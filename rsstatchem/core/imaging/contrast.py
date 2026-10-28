"""Process > Enhance Contrast (saturated stretch, normalize, histogram equalization).

Ported from ImageJ 1.54p ij/plugin/ContrastEnhancer.java with the histogram conventions of
ij/process/{Byte,Short,Float,Color}Statistics.java and StackStatistics.java (public domain,
NIH; W. Rasband et al.; equalization by Richard Kirk).

* 8-bit / RGB: 256-bin histogram (RGB: luminance ``(int)(r*rw+g*gw+b*bw+0.5)``);
* 16-bit: the full 65536-bin histogram (``histogram16``), limits are raw values;
* 32-bit: 256 bins between the (selection's) min and max, ``binSize = (max-min)/256``.

``saturated`` percent is split over both tails: ``threshold = (int)(count*saturated/200)``.
Without "normalize" a stretch only changes the display range (RGB: the pixels, as
``ColorProcessor.setMinAndMax``). Equalization uses the square root of the counts (counts < 2
kept) unless ``classic``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rsstatchem.core.imaging.math_ops import java_int, java_round


@dataclass
class HistStats:
    histogram: np.ndarray       # int64 counts (256 bins, or 65536 for 16-bit)
    hist_min: float
    bin_size: float
    pixel_count: int
    full16: bool = False        # histogram is the 65536-bin 16-bit histogram


def luminance(rgb: np.ndarray, weights=(1 / 3, 1 / 3, 1 / 3)) -> np.ndarray:
    r, g, b = (rgb[..., k].astype(np.float64) for k in range(3))
    return java_int(r * weights[0] + g * weights[1] + b * weights[2] + 0.5)


def histogram_stats(planes, mask: np.ndarray | None = None,
                    weights=(1 / 3, 1 / 3, 1 / 3)) -> HistStats:
    """Histogram of one plane or a list of planes (stack histogram) inside ``mask``."""
    if isinstance(planes, np.ndarray):
        planes = [planes]
    planes = list(planes)
    first = np.asarray(planes[0])

    def values(p):
        p = np.asarray(p)
        if p.ndim == 3:
            p = luminance(p, weights)
        return p if mask is None else p[mask]

    if first.ndim == 3 or first.dtype == np.uint8:
        hist = np.zeros(256, np.int64)
        for p in planes:
            hist += np.bincount(values(p).ravel().astype(np.int64), minlength=256)[:256]
        return HistStats(hist, 0.0, 1.0, int(hist.sum()))
    if first.dtype == np.uint16:
        hist = np.zeros(65536, np.int64)
        for p in planes:
            hist += np.bincount(values(p).ravel().astype(np.int64), minlength=65536)
        return HistStats(hist, 0.0, 1.0, int(hist.sum()), full16=True)
    vals = np.concatenate([values(p).ravel().astype(np.float64) for p in planes])
    lim = float(np.finfo(np.float32).max)   # FloatStatistics: -Float.MAX..Float.MAX
    vals = vals[(vals >= -lim) & (vals <= lim)]
    if vals.size == 0:
        return HistStats(np.zeros(256, np.int64), 0.0, 0.0, 0)
    vmin, vmax = float(vals.min()), float(vals.max())
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = 256 / (vmax - vmin)
        idx = java_int(scale * (vals - vmin))
    idx = np.minimum(idx, 255)
    hist = np.bincount(idx.astype(np.int64), minlength=256)[:256]
    return HistStats(hist.astype(np.int64), vmin, (vmax - vmin) / 256, int(vals.size))


def saturated_limits(stats: HistStats, saturated: float) -> tuple[int, int]:
    """``ContrastEnhancer.getMinAndMax``: (hmin, hmax) histogram indices."""
    hist = stats.histogram
    threshold = int(stats.pixel_count * saturated / 200.0) if saturated > 0.0 else 0
    csum = np.cumsum(hist)
    above = np.nonzero(csum > threshold)[0]
    hmin = int(above[0]) if above.size else len(hist) - 1
    rsum = np.cumsum(hist[::-1])
    above = np.nonzero(rsum > threshold)[0]
    hmax = len(hist) - 1 - int(above[0]) if above.size else 0
    return hmin, hmax


def stretch_range(stats: HistStats, saturated: float) -> tuple[float, float] | None:
    """The new (min, max) of a saturated stretch, or None when hmax <= hmin."""
    hmin, hmax = saturated_limits(stats, saturated)
    if hmax <= hmin:
        return None
    if stats.full16:
        return float(hmin), float(hmax)
    return stats.hist_min + hmin * stats.bin_size, stats.hist_min + hmax * stats.bin_size


def normalize(plane: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """``ContrastEnhancer.normalize``: 8/16-bit through a table to 0..255/65535, 32-bit to
    0..1 (``normalizeFloat``)."""
    a = np.asarray(plane)
    if a.dtype == np.float32:
        scale = 1.0 / (vmax - vmin) if vmax > vmin else 1.0
        v = a.astype(np.float64) - vmin
        v = np.where(v < 0.0, 0.0, v)
        v = v * scale
        v = np.where(v > 1.0, 1.0, v)
        return v.astype(np.float32)
    max2 = 65535 if a.dtype == np.uint16 else 255
    i = np.arange(max2 + 1, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        mid = java_int(((i - vmin) / (vmax - vmin)) * max2)
    lut = np.where(i <= vmin, 0, np.where(i >= vmax, max2, mid))
    return lut.astype(a.dtype)[a]


def rgb_set_min_max(rgb: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """``ColorProcessor.setMinAndMax`` (changes the pixels of all three channels)."""
    if vmax < vmin:
        return rgb.copy()
    lo = int(java_int(vmin))
    i = np.arange(256, dtype=np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = java_int(256.0 * (i - lo) / (vmax - vmin))
    lut = np.clip(v, 0, 255).astype(np.uint8)
    return lut[rgb]


def equalize_table(histogram: np.ndarray, sixteen: bool, classic: bool = False
                   ) -> np.ndarray:
    """``ContrastEnhancer.equalize(ip, histogram)`` lookup table (sequential sums, as Java)."""
    mx = 65535 if sixteen else 255
    hist = np.asarray(histogram, dtype=np.int64)
    if len(hist) < mx + 1:
        hist = np.concatenate([hist, np.zeros(mx + 1 - len(hist), np.int64)])
    h = hist.astype(np.float64)
    wv = h if classic else np.where(hist < 2, h, np.sqrt(h))
    wl = wv.tolist()
    total = wl[0]
    for k in range(1, mx):
        total += 2 * wl[k]
    total += wl[mx]
    scale = mx / total if total else math.inf
    lut = np.zeros(mx + 1, dtype=np.float64)
    s = wl[0]
    vals = []
    for k in range(1, mx):
        delta = wl[k]
        s += delta
        vals.append(s * scale)
        s += delta
    lut[1:mx] = vals
    out = java_round(lut)
    out[0] = 0
    out[mx] = mx
    return out


def equalize(plane: np.ndarray, histogram: np.ndarray | None = None, classic: bool = False
             ) -> np.ndarray:
    """Histogram equalization of an 8-bit, 16-bit or RGB plane (32-bit is not supported in
    ImageJ). ``histogram`` defaults to the plane's own histogram (RGB: luminance)."""
    a = np.asarray(plane)
    if a.dtype == np.float32:
        raise ValueError("Equalization of 32-bit images not supported.")
    sixteen = a.dtype == np.uint16
    if histogram is None:
        histogram = histogram_stats(a).histogram
    lut = equalize_table(histogram, sixteen, classic)
    if sixteen:
        return lut.astype(np.int64).astype(np.uint16)[a]
    return (lut.astype(np.int64) & 255).astype(np.uint8)[a]

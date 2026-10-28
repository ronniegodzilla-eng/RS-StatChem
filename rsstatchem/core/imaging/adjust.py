"""Image > Adjust: Brightness/Contrast (auto, reset, set, apply), Window/Level, Color Balance,
Apply LUT, and the B&C slider maths.

Ported from ImageJ 1.54 ij/plugin/frame/ContrastAdjuster.java (autoAdjust with
AUTO_THRESHOLD = 5000 halved on every further click, brightness/contrast slider mapping, Apply),
ij/plugin/filter/LutApplier.java, ij/process/ColorProcessor.setMinAndMax and the 256-bin
histograms of ij/process/{Byte,Short,Float,Color}Statistics.java (public domain, NIH;
W. Rasband).
"""
from __future__ import annotations

import numpy as np

from rsstatchem.core.imaging.convert import java_int, rgb_to_byte
from rsstatchem.core.imaging.image import COLOR_RGB, GRAY16, GRAY32, ImageDoc

AUTO_THRESHOLD = 5000
SLIDER_RANGE = 256


def histogram256(plane: np.ndarray, mask: np.ndarray | None = None, weights=None):
    """(histogram[256], histMin, binSize, min, max, pixelCount) as ImageStatistics computes them
    for "raw" statistics (8-bit/RGB: bins 0..255; 16-bit: (max-min+1)/256; 32-bit: (max-min)/256)."""
    if plane.ndim == 3:
        from rsstatchem.core.imaging.convert import UNWEIGHTED
        plane = rgb_to_byte(plane, weights or UNWEIGHTED)
    v = plane[mask] if mask is not None else plane.ravel()
    if v.dtype == np.uint8:
        h = np.bincount(v, minlength=256)
        nz = np.nonzero(h)[0]
        lo, hi = (float(nz[0]), float(nz[-1])) if nz.size else (0.0, 0.0)
        return h, 0.0, 1.0, lo, hi, int(v.size)
    if v.dtype == np.uint16:
        lo, hi = int(v.min()), int(v.max())
        bin_size = (hi - lo + 1) / 256.0
        idx = np.minimum(java_int((1.0 / bin_size) * (v.astype(np.int64) - lo)), 255)
        return np.bincount(idx, minlength=256), float(lo), bin_size, float(lo), float(hi), \
            int(v.size)
    f = v[np.isfinite(v)].astype(np.float32)
    if f.size == 0:
        return np.zeros(256, np.int64), 0.0, 0.0, 0.0, 0.0, 0
    lo, hi = float(f.min()), float(f.max())
    bin_size = (hi - lo) / 256.0
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = 256 / (hi - lo) if hi > lo else 0.0
        idx = np.minimum(java_int(scale * (f.astype(np.float64) - lo)), 255)
    return np.bincount(idx, minlength=256), lo, bin_size, lo, hi, int(f.size)


def auto_range(plane: np.ndarray, auto_threshold: int = 0, mask: np.ndarray | None = None):
    """ContrastAdjuster.autoAdjust: returns (min, max, new_auto_threshold) or None when the
    histogram gives hmax < hmin (ImageJ then resets). Pass the returned threshold back in for
    the next click (each click halves it; 0 = start again at 5000)."""
    hist, hmin0, bin_size, vmin, vmax, count = histogram256(plane, mask)
    limit = count // 10
    auto_threshold = AUTO_THRESHOLD if auto_threshold < 10 else auto_threshold // 2
    threshold = count // auto_threshold
    i = -1
    while True:
        i += 1
        c = int(hist[i])
        if c > limit:
            c = 0
        if c > threshold or i >= 255:
            found = c > threshold
            break
    hmin = i
    i = 256
    while True:
        i -= 1
        c = int(hist[i])
        if c > limit:
            c = 0
        if c > threshold or i <= 0:
            break
    hmax = i
    del found
    if hmax < hmin:
        return None
    lo = hmin0 + hmin * bin_size
    hi = hmin0 + hmax * bin_size
    if lo == hi:
        lo, hi = vmin, vmax
    return lo, hi, auto_threshold


def default_range(doc: ImageDoc) -> tuple[float, float]:
    """The B&C "default" range: 0-255 for 8-bit/RGB, the plane min/max for 16/32-bit."""
    if doc.bit_depth in (16, 32):
        return doc.plane_min_max()
    return 0.0, 255.0


def sliders_from_range(vmin, vmax, dmin, dmax, slider_range=SLIDER_RANGE):
    """ContrastAdjuster.updateScrollBars: (brightness, contrast) slider values."""
    mid = slider_range / 2
    span = (vmax - vmin) or 1e-12
    c = ((dmax - dmin) / span) * mid
    if c > mid:
        c = slider_range - (span / (dmax - dmin)) * mid
    level = vmin + span / 2.0
    normalized = 1.0 - (level - dmin) / ((dmax - dmin) or 1e-12)
    return int(normalized * slider_range), int(c)


def range_from_brightness(bvalue, vmin, vmax, dmin, dmax, slider_range=SLIDER_RANGE):
    center = dmin + (dmax - dmin) * ((slider_range - bvalue) / slider_range)
    width = vmax - vmin
    return center - width / 2.0, center + width / 2.0


def range_from_contrast(cvalue, vmin, vmax, dmin, dmax, slider_range=SLIDER_RANGE):
    center = vmin + (vmax - vmin) / 2.0
    rng = dmax - dmin
    mid = slider_range / 2
    slope = cvalue / mid if cvalue <= mid else mid / max(slider_range - cvalue, 1e-12)
    if slope > 0.0:
        return center - (0.5 * rng) / slope, center + (0.5 * rng) / slope
    return vmin, vmax


def window_level(vmin, vmax):
    """(window, level) of a display range."""
    return vmax - vmin, vmin + (vmax - vmin) / 2.0


def range_from_window_level(window, level):
    return level - window / 2.0, level + window / 2.0


def apply_table(doc_type: int, vmin: float, vmax: float) -> np.ndarray:
    """LutApplier / ContrastAdjuster.apply table: (int)((i-min)/(max-min)*range)."""
    size = 65536 if doc_type == GRAY16 else 256
    rng = size
    lo, hi = int(vmin), int(vmax)
    i = np.arange(size, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = java_int(((i - lo) / (hi - lo)) * rng) if hi != lo else np.zeros(size, np.int64)
    t = np.where(i <= lo, 0, np.where(i >= hi, rng - 1, t))
    return t.astype(np.uint16 if size == 65536 else np.uint8)


def rgb_display_table(vmin: float, vmax: float) -> np.ndarray:
    """ColorProcessor.setMinAndMax table: (int)(256*(i-(int)min)/(max-min)) clipped."""
    i = np.arange(256)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = java_int(256.0 * (i - int(vmin)) / (vmax - vmin))
    return np.clip(v, 0, 255).astype(np.uint8)


def apply_rgb_range(rgb: np.ndarray, vmin: float, vmax: float, channels: int = 7) -> np.ndarray:
    """Destructive RGB display range (``channels`` bit mask 4 = red, 2 = green, 1 = blue)."""
    table = rgb_display_table(vmin, vmax)
    out = rgb.copy()
    for k, bit in enumerate((4, 2, 1)):
        if channels & bit:
            out[..., k] = table[rgb[..., k]]
    return out


def apply_lut(doc: ImageDoc, all_slices: bool = True, mask: np.ndarray | None = None) -> None:
    """Image > Lookup Tables > Apply LUT (and B&C "Apply"): map the display range to the full
    pixel range (8/16-bit, RGB); 16-bit images get display range 0-65535 afterwards."""
    if doc.type == GRAY32:
        raise ValueError('"Apply" does not work with 32-bit images')
    vmin, vmax = doc.display_range
    idx = range(doc.stack_size) if all_slices else [doc.current_index]
    if doc.type == COLOR_RGB:
        for i in idx:
            c, z, t = doc.position_of(i)
            p = doc.plane(c, z, t)
            new = apply_rgb_range(p, vmin, vmax)
            if mask is not None:
                new = np.where(mask[..., None], new, p)
            doc.set_plane(new, c, z, t)
        doc.display_ranges = [(0.0, 255.0)]
        return
    channel = doc.position[0]
    table = apply_table(doc.type, vmin, vmax)
    for i in idx:
        c, z, t = doc.position_of(i)
        if doc.is_composite and c != channel:
            continue
        p = doc.plane(c, z, t)
        new = table[p]
        if mask is not None:
            new = np.where(mask, new, p)
        doc.set_plane(new, c, z, t)
    if doc.type == GRAY16:
        doc.set_display_range(0, 65535)
    else:
        doc.set_display_range(0, 255)


def color_balance(rgb: np.ndarray, channel: str, vmin: float, vmax: float) -> np.ndarray:
    """Image > Adjust > Color Balance on one channel ("Red", "Green", "Blue", "All", or the
    two-channel names ImageJ shows: "Cyan" = G+B, "Magenta" = R+B, "Yellow" = R+G)."""
    bits = {"Red": 4, "Green": 2, "Blue": 1, "Cyan": 3, "Magenta": 5, "Yellow": 6,
            "All": 7}[channel]
    return apply_rgb_range(rgb, vmin, vmax, bits)

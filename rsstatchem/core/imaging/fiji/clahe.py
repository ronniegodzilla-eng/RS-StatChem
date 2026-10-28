"""Enhance Local Contrast (CLAHE), written clean-room from

* Pizer S.M. et al. (1987) "Adaptive histogram equalization and its variations", CVGIP 39;
* Zuiderveld K. (1994) "Contrast Limited Adaptive Histogram Equalization", Graphics Gems IV.

No code of Fiji's (GPL) CLAHE plugin is used; the parameters follow its dialog: ``blocksize``
(side of the square neighbourhood, default 127), ``bins`` (histogram bins, 256) and
``slope`` (maximum slope of the mapping = clip limit relative to a uniform histogram, 3).

Accurate mode: every pixel is equalised with the clipped histogram of the ``blocksize`` square
centred on it (cut at the image border). Fast mode: Zuiderveld's tiles - one clipped mapping
per ``blocksize`` tile, bilinearly interpolated between tile centres. The clipped excess is
redistributed uniformly (repeatedly, so that no bin ends above the limit when possible); the
mapping of bin ``q`` is the centred cumulative count, so ``slope=1`` (a fully uniform clipped
histogram) is the identity. 16/32-bit planes are binned between their minimum and maximum and
mapped back to that range; RGB is equalised on the HSB brightness.
"""
from __future__ import annotations

import numpy as np

_REDISTRIBUTE_ITERATIONS = 12


def _quantize(a: np.ndarray, bins: int, lo: float, hi: float) -> np.ndarray:
    if a.dtype == np.uint8 and bins == 256 and lo == 0 and hi == 255:
        return a.astype(np.int32)
    span = hi - lo if hi > lo else 1.0
    q = np.floor((a.astype(np.float64) - lo) / span * (bins - 1) + 0.5)
    q = np.nan_to_num(q, nan=0.0)
    return np.clip(q, 0, bins - 1).astype(np.int32)


def clip_histograms(hist: np.ndarray, n: np.ndarray, slope: float) -> np.ndarray:
    """Clip histograms (bins first axis, counts ``n`` per histogram) at ``slope * n / bins``
    and redistribute the excess uniformly over all bins."""
    bins = hist.shape[0]
    h = hist.astype(np.float64)
    limit = slope * n / bins          # slope >= 1, so limit * bins >= n can always be met
    for _ in range(_REDISTRIBUTE_ITERATIONS):
        h = np.minimum(h, limit)
        excess = n - h.sum(axis=0)
        if np.all(excess <= 1e-9 * np.maximum(n, 1)):
            break
        h = h + excess / bins
    h = np.minimum(h, limit)
    h += (n - h.sum(axis=0)) / bins      # what is left (limit*bins < n cannot be clipped)
    return h


def mapping(hist: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Equalised bin (float, 0..bins-1) of every bin: centred cumulative count."""
    bins = hist.shape[0]
    cs = np.cumsum(hist, axis=0)
    mid = cs - hist / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        m = mid / np.maximum(n, 1e-12) * bins - 0.5
    return np.clip(m, 0, bins - 1)


def _accurate(q: np.ndarray, bins: int, blocksize: int, slope: float) -> np.ndarray:
    h, w = q.shape
    r = blocksize // 2
    chunk = max(8, min(128, int(4e6 // (bins * w))))
    out = np.zeros((h, w), np.float64)
    xs = np.arange(w)
    x0 = np.clip(xs - r, 0, w)
    x1 = np.clip(xs + r + 1, 0, w)
    present = np.unique(q)
    for y0 in range(0, h, chunk):
        y1 = min(y0 + chunk, h)
        s0, s1 = max(0, y0 - r), min(h, y1 + r)
        slab = q[s0:s1]
        ys = np.arange(y0, y1)
        top = np.clip(ys - r, 0, h) - s0
        bot = np.clip(ys + r + 1, 0, h) - s0
        hist = np.zeros((bins, y1 - y0, w), np.float32)
        for b in present:
            ii = np.zeros((slab.shape[0] + 1, w + 1), np.int32)
            np.cumsum(np.cumsum(slab == b, axis=0), axis=1, out=ii[1:, 1:])
            hist[b] = (ii[bot][:, x1] - ii[top][:, x1] - ii[bot][:, x0] + ii[top][:, x0])
        n = ((bot - top)[:, None] * (x1 - x0)[None, :]).astype(np.float64)
        clipped = clip_histograms(hist, n, slope)
        qq = q[y0:y1][None]
        cs = np.cumsum(clipped, axis=0)
        below = np.take_along_axis(cs, qq, axis=0)[0]
        own = np.take_along_axis(clipped, qq, axis=0)[0]
        out[y0:y1] = np.clip((below - own / 2.0) / n * bins - 0.5, 0, bins - 1)
    return out


def _fast(q: np.ndarray, bins: int, blocksize: int, slope: float) -> np.ndarray:
    h, w = q.shape
    ny, nx = max(1, -(-h // blocksize)), max(1, -(-w // blocksize))
    th, tw = h / ny, w / nx
    maps = np.zeros((ny, nx, bins), np.float64)
    for ty in range(ny):
        for tx in range(nx):
            ya, yb = int(round(ty * th)), int(round((ty + 1) * th))
            xa, xb = int(round(tx * tw)), int(round((tx + 1) * tw))
            tile = q[ya:yb, xa:xb]
            hist = np.bincount(tile.ravel(), minlength=bins).astype(np.float64)
            n = np.array(float(tile.size))
            maps[ty, tx] = mapping(clip_histograms(hist, n, slope), n)
    fy = (np.arange(h) + 0.5) / th - 0.5
    fx = (np.arange(w) + 0.5) / tw - 0.5
    iy0 = np.clip(np.floor(fy).astype(int), 0, ny - 1)
    ix0 = np.clip(np.floor(fx).astype(int), 0, nx - 1)
    iy1 = np.minimum(iy0 + 1, ny - 1)
    ix1 = np.minimum(ix0 + 1, nx - 1)
    wy = np.clip(fy - iy0, 0, 1)[:, None]
    wx = np.clip(fx - ix0, 0, 1)[None, :]
    yy0, yy1 = iy0[:, None], iy1[:, None]
    xx0, xx1 = ix0[None, :], ix1[None, :]
    m00 = maps[yy0, xx0, q]
    m01 = maps[yy0, xx1, q]
    m10 = maps[yy1, xx0, q]
    m11 = maps[yy1, xx1, q]
    return ((1 - wy) * ((1 - wx) * m00 + wx * m01) + wy * ((1 - wx) * m10 + wx * m11))


def clahe(plane: np.ndarray, blocksize: int = 127, bins: int = 256, slope: float = 3.0,
          mask: np.ndarray | None = None, fast: bool = False) -> np.ndarray:
    """CLAHE of a 2-D uint8/uint16/float32 plane or an (H, W, 3) RGB plane; returns a new
    array of the same dtype. Pixels where ``mask`` is False keep their value."""
    a = np.asarray(plane)
    if a.ndim == 3:
        from rsstatchem.core.imaging.fft import get_brightness, set_brightness
        bri = np.floor(get_brightness(a).astype(np.float64) * 255.0 + 0.5).astype(np.uint8)
        new = clahe(bri, blocksize, bins, slope, mask, fast)
        out = set_brightness(a, (new.astype(np.float32) / np.float32(255.0)))
        if mask is not None:
            out = np.where(mask[..., None], out, a)
        return out
    blocksize = max(int(blocksize), 3)
    bins = int(min(max(int(bins), 2), 65536))
    slope = max(float(slope), 1.0)
    if a.dtype == np.uint8:
        lo, hi = 0.0, 255.0
    else:
        finite = a[np.isfinite(a)] if a.dtype == np.float32 else a
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 0.0)
    q = _quantize(a, bins, lo, hi)
    m = _fast(q, bins, blocksize, slope) if fast else _accurate(q, bins, blocksize, slope)
    span = hi - lo
    vals = lo + m / (bins - 1) * span
    if a.dtype == np.float32:
        out = vals.astype(np.float32)
        out = np.where(np.isfinite(a), out, a)
    else:
        info = np.iinfo(a.dtype)
        out = np.clip(np.floor(vals + 0.5), info.min, info.max).astype(a.dtype)
    if mask is not None:
        out = np.where(mask, out, a)
    return out

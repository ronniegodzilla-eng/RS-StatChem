"""Kymographs along a line or polyline of a time series (KymographBuilder-style).

Follows the approach of KymographBuilder (H. Hadrien, MIT licence; Fiji): the path is sampled
at 1-pixel steps with bilinear interpolation, each sample averaged over ``width`` pixels
perpendicular to the path, one row per frame, so the kymograph has x = distance along the path
and y = time. The implementation is written for numpy (no Java code copied).
"""
from __future__ import annotations

import numpy as np


def sample_path(xs, ys, step: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                    np.ndarray]:
    """Points every ``step`` pixels along the polyline (xs, ys) and the unit normals there."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    seg = np.hypot(np.diff(xs), np.diff(ys))
    keep = seg > 0
    if not keep.any():
        return xs[:1], ys[:1], np.zeros(1), np.ones(1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    d = np.arange(0.0, cum[-1] + 1e-9, step)
    k = np.clip(np.searchsorted(cum, d, side="right") - 1, 0, len(seg) - 1)
    good = np.nonzero(keep)[0]
    pos = np.searchsorted(good, k, side="right") - 1
    k = good[np.clip(pos, 0, len(good) - 1)]
    frac = (d - cum[k]) / seg[k]
    px = xs[k] + frac * (xs[k + 1] - xs[k])
    py = ys[k] + frac * (ys[k + 1] - ys[k])
    ux = (xs[k + 1] - xs[k]) / seg[k]
    uy = (ys[k + 1] - ys[k]) / seg[k]
    return px, py, -uy, ux


def _bilinear(plane: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = plane.shape
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    p = plane.astype(np.float64)
    top = p[y0, x0] + fx * (p[y0, x0 + 1] - p[y0, x0])
    bot = p[y0 + 1, x0] + fx * (p[y0 + 1, x0 + 1] - p[y0 + 1, x0])
    return top + fy * (bot - top)


def kymograph(frames: np.ndarray, xs, ys, width: int = 1) -> np.ndarray:
    """(T, Y, X) or (T, Y, X, 3) frames -> kymograph (T, L) float32 (or (T, L, 3) uint8)."""
    # pixel centres: ImageJ line coordinates refer to pixel corners
    px, py, nx, ny = sample_path(np.asarray(xs) - 0.5, np.asarray(ys) - 0.5)
    offsets = np.arange(width) - (width - 1) / 2.0
    if frames.ndim == 4:
        chans = [kymograph(frames[..., k], xs, ys, width) for k in range(3)]
        return np.clip(np.floor(np.stack(chans, axis=-1) + 0.5), 0, 255).astype(np.uint8)
    out = np.zeros((frames.shape[0], px.size), dtype=np.float64)
    for t, plane in enumerate(frames):
        acc = np.zeros(px.size)
        for o in offsets:
            acc += _bilinear(plane, px + o * nx, py + o * ny)
        out[t] = acc / len(offsets)
    return out.astype(np.float32)


def to_dtype(kymo: np.ndarray, dtype) -> np.ndarray:
    if np.dtype(dtype).kind == "f" or kymo.dtype == np.uint8:
        return kymo.astype(dtype)
    info = np.iinfo(dtype)
    return np.clip(np.floor(kymo + 0.5), info.min, info.max).astype(dtype)

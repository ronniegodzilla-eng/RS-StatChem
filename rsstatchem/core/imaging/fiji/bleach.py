"""Bleach correction of time series (Fiji "Bleach Correction" equivalent, written from the
literature, not from the GPL plugin): K. Miura et al., "ImageJ plugin CorrectBleach",
Zenodo (2014) and K. Miura, Bioimage Data Analysis (Wiley 2016), ch. 4.

Methods:

* ``simple_ratio``: frame t is multiplied by mean_0 / mean_t (means of the whole frame or of the
  ``mask``), after subtracting a constant ``background`` which is added back.
* ``exponential_fit``: the frame means are fitted with y = a*exp(-b*t) + c (least squares,
  scipy), and frame t is multiplied by f(0)/f(t); the fit parameters are returned.
* ``histogram_matching``: every frame's histogram is matched to that of the first frame
  (cumulative distribution mapping, per integer value for 8/16-bit, 4096 quantiles for 32-bit).

All functions take frames as an array (T, Y, X) and return a corrected array of the same dtype
(clipped and rounded for integer types).
"""
from __future__ import annotations

import numpy as np

METHODS = ("Simple Ratio", "Exponential Fit", "Histogram Matching")


def _store(f: np.ndarray, dtype) -> np.ndarray:
    if np.dtype(dtype).kind == "f":
        return f.astype(dtype)
    info = np.iinfo(dtype)
    return np.clip(np.floor(f + 0.5), info.min, info.max).astype(dtype)


def frame_means(frames: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    f = frames.astype(np.float64)
    if mask is not None:
        return np.array([np.nanmean(p[mask]) for p in f])
    return np.nanmean(f.reshape(f.shape[0], -1), axis=1)


def simple_ratio(frames: np.ndarray, background: float = 0.0,
                 mask: np.ndarray | None = None) -> np.ndarray:
    means = frame_means(frames, mask) - background
    ratio = np.where(means != 0, means[0] / np.where(means == 0, 1, means), 1.0)
    f = (frames.astype(np.float64) - background) * ratio[:, None, None] + background
    return _store(f, frames.dtype)


def exponential_fit(frames: np.ndarray, mask: np.ndarray | None = None
                    ) -> tuple[np.ndarray, tuple[float, float, float]]:
    from scipy.optimize import curve_fit
    y = frame_means(frames, mask)
    t = np.arange(len(y), dtype=np.float64)

    def model(tt, a, b, c):
        return a * np.exp(-b * tt) + c

    a0 = max(y[0] - y[-1], 1e-6)
    b0 = 1.0 / max(len(y), 1)
    try:
        p, _ = curve_fit(model, t, y, p0=(a0, b0, y[-1]), maxfev=20000)
    except (RuntimeError, ValueError):
        p = np.array([0.0, 0.0, y[0]])
    fit = model(t, *p)
    scale = np.where(fit != 0, fit[0] / np.where(fit == 0, 1, fit), 1.0)
    out = frames.astype(np.float64) * scale[:, None, None]
    return _store(out, frames.dtype), (float(p[0]), float(p[1]), float(p[2]))


def histogram_matching(frames: np.ndarray) -> np.ndarray:
    ref = frames[0]
    out = np.empty_like(frames)
    out[0] = ref
    if frames.dtype.kind in "ui":
        n = int(max(frames.max(), 0)) + 1
        ref_cdf = np.cumsum(np.bincount(ref.ravel().astype(np.int64), minlength=n))
        ref_cdf = ref_cdf / ref_cdf[-1]
        for i in range(1, frames.shape[0]):
            src = frames[i].ravel().astype(np.int64)
            cdf = np.cumsum(np.bincount(src, minlength=n))
            cdf = cdf / cdf[-1]
            lut = np.searchsorted(ref_cdf, cdf, side="left").clip(0, n - 1)
            out[i] = lut[frames[i]].astype(frames.dtype)
        return out
    q = np.linspace(0, 1, 4096)
    rq = np.nanquantile(ref, q)
    for i in range(1, frames.shape[0]):
        sq = np.nanquantile(frames[i], q)
        out[i] = np.interp(frames[i], sq, rq).astype(frames.dtype)
    return out


def correct(frames: np.ndarray, method: str = "Simple Ratio", background: float = 0.0,
            mask: np.ndarray | None = None):
    """Dispatch by the dialog name; returns (corrected frames, info dict)."""
    if method == "Simple Ratio":
        return simple_ratio(frames, background, mask), {}
    if method == "Exponential Fit":
        out, (a, b, c) = exponential_fit(frames, mask)
        return out, {"a": a, "b": b, "c": c}
    if method == "Histogram Matching":
        return histogram_matching(frames), {}
    raise ValueError(f"Unknown bleach correction method {method!r}")

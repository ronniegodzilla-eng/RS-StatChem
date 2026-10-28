"""Origin's FFT Filters, IIR Filter Design and the extra smoothers, without Qt.

* FFT filters on evenly spaced data: low-pass, high-pass, band-pass, band-block, threshold
  (drop components whose amplitude is below a threshold) and low-pass parabolic
  (weight ``1 − (f/fc)²``); cutoffs in the frequency unit of X (1 / X unit). The mean (DC)
  can be kept and the end-point trend removed before filtering, as Origin offers.
* IIR filters (``scipy.signal``): Butterworth, Chebyshev I / II, Elliptic and Bessel; low /
  high / band-pass / band-stop; applied causally (``sosfilt``) or with zero phase
  (``sosfiltfilt``); frequency, phase, group-delay, impulse and step responses.
* Smoothers: LOESS (local quadratic) and LOWESS (local linear) with tricube weights and
  optional robust bisquare iterations (Cleveland 1979), binomial smoothing, weighted
  (triangular) adjacent averaging, and Savitzky-Golay with derivative order and boundary
  mode.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from scipy import signal as ss
from scipy.special import comb

FFT_FILTERS = {"lowpass": "Low Pass", "highpass": "High Pass", "bandpass": "Band Pass",
               "bandblock": "Band Block", "threshold": "Threshold",
               "parabolic": "Low Pass Parabolic"}
IIR_TYPES = {"butter": "Butterworth", "cheby1": "Chebyshev Type I",
             "cheby2": "Chebyshev Type II", "ellip": "Elliptic", "bessel": "Bessel"}
BAND_TYPES = {"lowpass": "Low Pass", "highpass": "High Pass", "bandpass": "Band Pass",
              "bandstop": "Band Stop"}
SG_MODES = {"interp": "Polynomial fit at the ends", "mirror": "Reflect", "nearest": "Repeat",
            "constant": "Pad with zero", "wrap": "Periodic"}
SMOOTH_EXTRA = {"loess": "LOESS (local quadratic)", "lowess": "LOWESS (local linear)",
                "binomial": "Binomial", "triangular": "Weighted adjacent averaging"}


# ---------------------------------------------------------------------- FFT filters
def fft_filter(y, dt: float = 1.0, kind: str = "lowpass", cutoff: float = 0.1,
               cutoff2: float | None = None, threshold: float = 0.0, keep_dc: bool = True,
               trend: str = "none") -> np.ndarray:
    """Filter evenly spaced ``y`` (sampling interval ``dt``) in the frequency domain.
    ``trend="ends"`` subtracts the line through the end points first (less ringing) and adds
    it back; ``threshold`` is an amplitude (same units as y) for ``kind="threshold"``."""
    y = np.asarray(y, float)
    n = len(y)
    if n < 4:
        raise ValueError("At least 4 points are needed for an FFT filter.")
    base = np.linspace(y[0], y[-1], n) if trend == "ends" else np.zeros(n)
    v = y - base
    spec = np.fft.rfft(v)
    f = np.fft.rfftfreq(n, d=dt)
    nyq = 0.5 / dt
    lo, hi = float(cutoff), float(cutoff2) if cutoff2 is not None else nyq
    if kind in ("bandpass", "bandblock") and hi <= lo:
        raise ValueError("The upper cutoff must be above the lower cutoff.")
    if kind == "lowpass":
        gain = (f <= lo).astype(float)
    elif kind == "highpass":
        gain = (f >= lo).astype(float)
    elif kind == "bandpass":
        gain = ((f >= lo) & (f <= hi)).astype(float)
    elif kind == "bandblock":
        gain = ((f < lo) | (f > hi)).astype(float)
    elif kind == "parabolic":
        gain = np.clip(1 - (f / lo) ** 2, 0, None) if lo > 0 else np.zeros_like(f)
    elif kind == "threshold":
        amp = np.abs(spec) / n
        amp[1:] *= 2
        if n % 2 == 0:
            amp[-1] /= 2
        gain = (amp >= threshold).astype(float)
    else:
        raise ValueError(f"Unknown FFT filter '{kind}'")
    if keep_dc:
        gain[0] = 1.0
    return np.fft.irfft(spec * gain, n) + base


# ---------------------------------------------------------------------- IIR
def iir_design(ftype: str = "butter", btype: str = "lowpass", order: int = 4,
               cutoff=0.1, fs: float = 1.0, rp: float = 1.0, rs: float = 40.0):
    """Second-order sections of the filter; ``cutoff`` is a frequency (or a pair for band
    filters) in the units of ``fs`` (1 / X unit); ``rp`` / ``rs`` are the pass-band ripple
    and stop-band attenuation in dB."""
    wn = np.atleast_1d(np.asarray(cutoff, float))
    if btype in ("bandpass", "bandstop"):
        if len(wn) != 2 or wn[1] <= wn[0]:
            raise ValueError("A band filter needs a lower and a higher cutoff frequency.")
    else:
        wn = wn[:1]
    if np.any(wn <= 0) or np.any(wn >= fs / 2):
        raise ValueError(f"Cutoff frequencies must lie between 0 and the Nyquist frequency "
                         f"{fs / 2:g}.")
    wn = wn if len(wn) > 1 else float(wn[0])
    order = int(order)
    if ftype == "butter":
        return ss.butter(order, wn, btype=btype, fs=fs, output="sos")
    if ftype == "cheby1":
        return ss.cheby1(order, rp, wn, btype=btype, fs=fs, output="sos")
    if ftype == "cheby2":
        return ss.cheby2(order, rs, wn, btype=btype, fs=fs, output="sos")
    if ftype == "ellip":
        return ss.ellip(order, rp, rs, wn, btype=btype, fs=fs, output="sos")
    if ftype == "bessel":
        return ss.bessel(order, wn, btype=btype, fs=fs, output="sos", norm="phase")
    raise ValueError(f"Unknown IIR type '{ftype}'")


def iir_apply(y, sos, zero_phase: bool = True) -> np.ndarray:
    y = np.asarray(y, float)
    if zero_phase:
        padlen = min(3 * (2 * len(sos) + 1), len(y) - 1)
        return ss.sosfiltfilt(sos, y, padlen=max(padlen, 0))
    return ss.sosfilt(sos, y)


def iir_response(sos, fs: float = 1.0, n: int = 512) -> dict:
    """Frequency response: ``freq``, ``gain`` (|H|), ``db``, ``phase`` (degrees, unwrapped)
    and ``group_delay`` (samples)."""
    f, h = ss.sosfreqz(sos, worN=n, fs=fs)
    b, a = ss.sos2tf(sos)
    with np.errstate(divide="ignore", invalid="ignore"):
        _, gd = ss.group_delay((b, a), w=f, fs=fs)
    return {"freq": f, "gain": np.abs(h), "db": 20 * np.log10(np.maximum(np.abs(h), 1e-300)),
            "phase": np.degrees(np.unwrap(np.angle(h))), "group_delay": gd}


def impulse_response(sos, n: int = 100) -> np.ndarray:
    x = np.zeros(n)
    x[0] = 1.0
    return ss.sosfilt(sos, x)


def step_response(sos, n: int = 100) -> np.ndarray:
    return ss.sosfilt(sos, np.ones(n))


# ---------------------------------------------------------------------- LOESS / LOWESS
def loess(x, y, frac: float = 0.3, degree: int = 2, robust_iters: int = 0,
          x_eval=None) -> np.ndarray:
    """Locally weighted regression (Cleveland 1979; Cleveland & Devlin 1988). For each
    point the ``k = int(frac·n)`` nearest neighbours are fitted by a weighted polynomial of
    ``degree`` (1 = LOWESS, 2 = LOESS) with tricube weights; ``robust_iters`` bisquare
    re-weighting passes (residual scale 6·median|r|) down-weight outliers."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    if n < degree + 2:
        raise ValueError("Too few points for LOESS.")
    xe = x if x_eval is None else np.asarray(x_eval, float)
    k = int(min(max(frac * n + 1e-10, degree + 1), n))
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    robust = np.ones(n)

    def fit_at(x0: float, rw: np.ndarray) -> float:
        d = np.abs(xs - x0)
        idx = np.argpartition(d, k - 1)[:k]
        h = d[idx].max()
        if h <= 0:
            return float(np.average(ys[idx], weights=rw[idx])) if rw[idx].sum() else \
                float(ys[idx].mean())
        u = d[idx] / h
        w = np.clip(1 - u ** 3, 0, None) ** 3 * rw[idx]
        if w.sum() <= 0:
            return float(ys[idx].mean())
        X = np.vander(xs[idx] - x0, degree + 1, increasing=True)
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(X * sw[:, None], ys[idx] * sw, rcond=None)
        return float(beta[0])

    for it in range(robust_iters + 1):
        fitted = np.array([fit_at(v, robust) for v in xs])
        if it == robust_iters:
            break
        res = ys - fitted
        s = 6 * np.median(np.abs(res))
        if s <= 0:
            break
        robust = np.clip(1 - (res / s) ** 2, 0, None) ** 2
    if x_eval is None:
        out = np.empty(n)
        out[order] = fitted
        return out
    return np.array([fit_at(v, robust) for v in xe])


# ---------------------------------------------------------------------- other smoothers
def binomial(y, order: int = 2) -> np.ndarray:
    """Binomial smoothing: convolution with the normalised binomial coefficients
    C(order, k) / 2^order (window ``order + 1``; order 2 = [1 2 1]/4), ends reflected."""
    order = max(1, int(order))
    kernel = comb(order, np.arange(order + 1)) / 2.0 ** order
    return ndimage.convolve1d(np.asarray(y, float), kernel, mode="reflect")


def triangular(y, window: int = 5) -> np.ndarray:
    """Adjacent averaging with triangular weights (Origin's weighted average)."""
    w = max(1, int(window)) | 1
    half = w // 2
    kernel = np.concatenate([np.arange(1, half + 2), np.arange(half, 0, -1)]).astype(float)
    kernel /= kernel.sum()
    return ndimage.convolve1d(np.asarray(y, float), kernel, mode="nearest")


def savgol(y, window: int = 7, polyorder: int = 2, deriv: int = 0, delta: float = 1.0,
           mode: str = "interp") -> np.ndarray:
    """Savitzky-Golay smoothing or derivative (``deriv``) with Origin's boundary options."""
    y = np.asarray(y, float)
    w = max(3, int(window) | 1)
    if w > len(y):
        w = len(y) if len(y) % 2 else len(y) - 1
    if polyorder >= w:
        raise ValueError("The polynomial order must be smaller than the window size.")
    if deriv > polyorder:
        raise ValueError("The derivative order cannot exceed the polynomial order.")
    return ss.savgol_filter(y, w, int(polyorder), deriv=int(deriv), delta=delta, mode=mode)


def savgol_coefficients(window: int, polyorder: int, deriv: int = 0) -> np.ndarray:
    """Convolution weights (central point), e.g. the Savitzky & Golay (1964) tables."""
    return ss.savgol_coeffs(int(window), int(polyorder), deriv=int(deriv), use="dot")


def smooth_extra(x, y, method: str, window: int = 7, frac: float = 0.3, robust: bool = False,
                 order: int = 2) -> np.ndarray:
    """Dispatcher for the extra smoothers (results in the original point order)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if method in ("loess", "lowess"):
        return loess(x, y, frac, 2 if method == "loess" else 1, 4 if robust else 0)
    o = np.argsort(x, kind="stable")
    ys = y[o]
    s = binomial(ys, order) if method == "binomial" else triangular(ys, window) \
        if method == "triangular" else None
    if s is None:
        raise ValueError(f"Unknown smoothing method '{method}'")
    out = np.empty(len(y))
    out[o] = s
    return out


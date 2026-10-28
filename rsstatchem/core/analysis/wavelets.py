"""Wavelet transforms (Origin's Signal Processing > Wavelet), without Qt, on the optional
PyWavelets package (``pywt``; :func:`available` says whether it is installed — the menus
hide the Wavelet entries otherwise).

* CWT scalogram (Morlet, Mexican hat, Gaussian derivative, complex Morlet…) with scales and
  pseudo-frequencies.
* DWT / IDWT (one level) and multilevel decomposition / reconstruction of each level
  (approximation and details) — Origin's Multi-Scale DWT.
* Denoising (Donoho & Johnstone 1994): universal threshold ``σ √(2 ln N)`` with σ from the
  median absolute deviation of the finest details (/0.6745), soft or hard thresholding, or a
  level-dependent threshold; smoothing by dropping the finest detail levels.
"""
from __future__ import annotations

import math

import numpy as np

DWT_WAVELETS = {"haar": "Haar", "db2": "Daubechies 2", "db4": "Daubechies 4",
                "db6": "Daubechies 6", "db8": "Daubechies 8", "sym4": "Symlet 4",
                "sym8": "Symlet 8", "coif2": "Coiflet 2", "bior3.5": "Biorthogonal 3.5"}
CWT_WAVELETS = {"morl": "Morlet", "mexh": "Mexican Hat", "gaus1": "Gaussian 1st derivative",
                "gaus2": "Gaussian 2nd derivative", "cmor1.5-1.0": "Complex Morlet"}
MODES = {"symmetric": "Symmetric", "periodization": "Periodization", "zero": "Zero padding",
         "reflect": "Reflect", "periodic": "Periodic", "smooth": "Smooth"}


class WaveletUnavailable(RuntimeError):
    pass


def available() -> bool:
    try:
        import pywt  # noqa: F401
    except ImportError:
        return False
    return True


def _pywt():
    try:
        import pywt
    except ImportError:
        raise WaveletUnavailable("Wavelet transforms need the optional PyWavelets package, "
                                 "which is not installed.") from None
    return pywt


def cwt(y, dt: float = 1.0, wavelet: str = "morl", scales=None, n_scales: int = 64) -> dict:
    """Continuous wavelet transform: ``coef`` (scales × points), ``scales``, ``freqs``
    (pseudo-frequencies in 1/X units) and ``power`` = |coef|²."""
    pywt = _pywt()
    y = np.asarray(y, float)
    if scales is None:
        scales = np.geomspace(1, max(2.0, len(y) / 4), int(n_scales))
    scales = np.asarray(scales, float)
    coef, freqs = pywt.cwt(y, scales, wavelet, sampling_period=dt)
    return {"coef": coef, "scales": scales, "freqs": freqs, "power": np.abs(coef) ** 2}


def dwt(y, wavelet: str = "db4", mode: str = "symmetric") -> dict:
    pywt = _pywt()
    ca, cd = pywt.dwt(np.asarray(y, float), wavelet, mode=mode)
    return {"approx": ca, "detail": cd}


def idwt(approx, detail, wavelet: str = "db4", mode: str = "symmetric") -> np.ndarray:
    return _pywt().idwt(approx, detail, wavelet, mode=mode)


def max_level(n: int, wavelet: str = "db4") -> int:
    pywt = _pywt()
    return pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len)


def decompose(y, wavelet: str = "db4", level: int | None = None,
              mode: str = "symmetric") -> dict:
    """Multilevel DWT: ``coeffs`` [cA_n, cD_n … cD_1] and the signal reconstructed from
    each part separately (``components``: names → arrays of the signal length; they add up
    to the signal)."""
    pywt = _pywt()
    y = np.asarray(y, float)
    level = level or max_level(len(y), wavelet)
    coeffs = pywt.wavedec(y, wavelet, mode=mode, level=level)
    comps = {}
    for i in range(len(coeffs)):
        keep = [c if j == i else np.zeros_like(c) for j, c in enumerate(coeffs)]
        name = f"A{level}" if i == 0 else f"D{level - i + 1}"
        comps[name] = pywt.waverec(keep, wavelet, mode=mode)[:len(y)]
    return {"coeffs": coeffs, "components": comps, "level": level}


def reconstruct(coeffs, wavelet: str = "db4", mode: str = "symmetric", n: int | None = None) \
        -> np.ndarray:
    out = _pywt().waverec(coeffs, wavelet, mode=mode)
    return out[:n] if n else out


def universal_threshold(detail_finest, n: int) -> float:
    sigma = np.median(np.abs(detail_finest)) / 0.6745
    return float(sigma * math.sqrt(2 * math.log(n)))


def denoise(y, wavelet: str = "db4", level: int | None = None, mode: str = "soft",
            threshold: float | None = None, rule: str = "universal",
            ext: str = "symmetric") -> dict:
    """Wavelet noise removal. ``rule="universal"``: one threshold σ√(2 ln N) for every detail
    level (σ from the finest level); ``"level"``: the universal threshold with σ estimated
    per level. A fixed ``threshold`` overrides the rule."""
    pywt = _pywt()
    y = np.asarray(y, float)
    n = len(y)
    level = level or max_level(n, wavelet)
    coeffs = pywt.wavedec(y, wavelet, mode=ext, level=level)
    thr_all = threshold if threshold is not None else universal_threshold(coeffs[-1], n)
    new = [coeffs[0]]
    used = []
    for c in coeffs[1:]:
        t = thr_all if (threshold is not None or rule == "universal") else \
            universal_threshold(c, n)
        used.append(t)
        new.append(pywt.threshold(c, t, mode=mode))
    return {"signal": pywt.waverec(new, wavelet, mode=ext)[:n], "thresholds": used,
            "level": level}


def smooth(y, wavelet: str = "db4", drop_levels: int = 1, level: int | None = None,
           ext: str = "symmetric") -> np.ndarray:
    """Wavelet smoothing: zero the ``drop_levels`` finest detail levels and reconstruct."""
    pywt = _pywt()
    y = np.asarray(y, float)
    level = level or max_level(len(y), wavelet)
    coeffs = pywt.wavedec(y, wavelet, mode=ext, level=level)
    for k in range(1, min(drop_levels, len(coeffs) - 1) + 1):
        coeffs[-k] = np.zeros_like(coeffs[-k])
    return pywt.waverec(coeffs, wavelet, mode=ext)[:len(y)]


def dwt2(z, wavelet: str = "haar", mode: str = "symmetric") -> dict:
    """One-level 2D DWT of a matrix: approximation and horizontal/vertical/diagonal
    details."""
    ca, (ch, cv, cd) = _pywt().dwt2(np.nan_to_num(np.asarray(z, float)), wavelet, mode=mode)
    return {"approx": ca, "horizontal": ch, "vertical": cv, "diagonal": cd}

"""Origin's Signal Processing transforms, without Qt: FFT / IFFT with every output type, STFT,
Hilbert transform and envelope, auto/cross correlation, coherence, convolution /
deconvolution, decimation and 2D FFT / IFFT / correlation.

FFT conventions (Origin's FFT page): ``F(k) = Σ x(n) e^(−2πikn/N)`` (no normalisation),
frequencies ``k / (N Δt)``. With a window ``w``:

* ``amplitude`` = ``|F| / Σw`` (two-sided); one-sided values (``one_sided=True``) are doubled
  except at DC and Nyquist, so a sinusoid ``A sin(2πft)`` on a bin gives ``A``.
* ``magnitude`` = ``|F|``; ``db`` = ``20 log10(amplitude / ref)``.
* ``msa`` (mean square amplitude) = ``|F|² / (M Σw²)`` (M = FFT length) (one-sided doubled except DC/Nyquist):
  ``Σ msa = mean(x²)`` for any window (Parseval).
* ``psd`` = ``|F|² Δt / Σw²`` (one-sided doubled): ``Σ psd · Δf = mean(x²)``.
* ``real``, ``imag``, ``phase`` (degrees or radians, optionally unwrapped).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import signal as ss

WINDOWS = {"rect": "Rectangle", "hann": "Hanning", "hamming": "Hamming",
           "blackman": "Blackman", "bartlett": "Bartlett (triangular)",
           "flattop": "Flat Top", "welch": "Welch", "kaiser": "Kaiser (β = 8.6)"}
SPECTRA = {"amplitude": "Amplitude", "magnitude": "Magnitude", "msa": "Mean Square Amplitude",
           "psd": "Power Spectral Density", "db": "Amplitude in dB"}
CORR_NORMS = {"none": "None (sum of products)", "biased": "Biased (÷N)",
              "unbiased": "Unbiased (÷(N−|lag|))", "coeff": "Correlation coefficient"}


def window(name: str, n: int) -> np.ndarray:
    if name in ("rect", "none", "", None):
        return np.ones(n)
    if name == "welch":
        k = np.arange(n)
        return 1 - ((k - (n - 1) / 2) / ((n + 1) / 2)) ** 2
    if name == "kaiser":
        return ss.get_window(("kaiser", 8.6), n, fftbins=False)
    return ss.get_window(name, n, fftbins=False)


def sampling_interval(x) -> float:
    """Δt of an X column (mean spacing; ValueError when X does not increase)."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return 1.0
    d = np.diff(x)
    dt = float(np.mean(d))
    if dt <= 0:
        raise ValueError("X must increase for a transform.")
    return dt


def _pad_length(n: int, pad: str | int | None) -> int:
    if pad in (None, "", "none", 0):
        return n
    if pad == "pow2":
        return 1 << (n - 1).bit_length()
    return max(n, int(pad))


# ---------------------------------------------------------------------- FFT
def fft(y, dt: float = 1.0, window_name: str = "rect", one_sided: bool = True,
        shift: bool = False, pad=None, detrend: str = "none", ref: float = 1.0,
        phase_unit: str = "deg", unwrap: bool = False) -> dict:
    """Spectrum of ``y`` sampled every ``dt``: arrays ``freq``, ``real``, ``imag``,
    ``magnitude``, ``amplitude``, ``msa``, ``psd``, ``db``, ``phase`` (one-sided or
    two-sided, the latter optionally shifted so that 0 is in the middle)."""
    y = np.asarray(y, float)
    n = len(y)
    if n < 2:
        raise ValueError("At least 2 points are needed for an FFT.")
    if not np.isfinite(y).all():
        raise ValueError("The signal contains missing values.")
    if detrend == "mean":
        y = y - y.mean()
    elif detrend == "linear":
        y = ss.detrend(y, type="linear")
    w = window(window_name, n)
    m = _pad_length(n, pad)
    spec = np.fft.fft(y * w, m)
    freq = np.fft.fftfreq(m, d=dt)
    s1, s2 = float(np.sum(w)), float(np.sum(w ** 2))
    mag = np.abs(spec)
    amp = mag / s1
    msa = mag ** 2 / (m * s2)
    psd = mag ** 2 * dt / s2
    if one_sided:
        k = m // 2 + 1
        freq = np.abs(freq[:k])
        spec, mag, amp, msa, psd = spec[:k], mag[:k], amp[:k].copy(), msa[:k].copy(), \
            psd[:k].copy()
        double = np.ones(k, dtype=bool)
        double[0] = False
        if m % 2 == 0:
            double[-1] = False
        amp[double] *= 2
        msa[double] *= 2
        psd[double] *= 2
    elif shift:
        freq, spec, mag, amp, msa, psd = (np.fft.fftshift(a) for a in
                                          (freq, spec, mag, amp, msa, psd))
    phase = np.angle(spec)
    if unwrap:
        phase = np.unwrap(phase)
    if phase_unit == "deg":
        phase = np.degrees(phase)
    with np.errstate(divide="ignore"):
        db = 20 * np.log10(np.maximum(amp, 1e-300) / ref)
    return {"freq": freq, "real": spec.real, "imag": spec.imag, "magnitude": mag,
            "amplitude": amp, "msa": msa, "psd": psd, "db": db, "phase": phase, "n": n,
            "n_fft": m, "dt": dt, "df": 1.0 / (m * dt), "window": window_name}


def ifft(real, imag=None, df: float | None = None, shifted: bool = False) -> dict:
    """Inverse FFT of a two-sided spectrum (``real`` + i ``imag``): the time signal (real and
    imaginary parts) and its time axis ``t = n / (N Δf)``."""
    re_ = np.asarray(real, float)
    im = np.zeros_like(re_) if imag is None else np.asarray(imag, float)
    spec = re_ + 1j * im
    if shifted:
        spec = np.fft.ifftshift(spec)
    x = np.fft.ifft(spec)
    n = len(x)
    dt = 1.0 / (n * df) if df else 1.0
    return {"t": np.arange(n) * dt, "real": x.real, "imag": x.imag, "dt": dt}


def ifft_from_polar(amplitude, phase, df: float | None = None, deg: bool = True) -> dict:
    ph = np.radians(phase) if deg else np.asarray(phase, float)
    a = np.asarray(amplitude, float)
    return ifft(a * np.cos(ph), a * np.sin(ph), df)


# ---------------------------------------------------------------------- STFT
def stft(y, dt: float = 1.0, nperseg: int = 64, noverlap: int | None = None,
         window_name: str = "hann", scale: str = "magnitude") -> dict:
    """Short-time Fourier transform (``scipy.signal.ShortTimeFFT``): ``times``, ``freqs`` and
    the ``values`` matrix (rows = frequencies) as magnitude, power or dB."""
    y = np.asarray(y, float)
    nperseg = int(min(max(4, nperseg), len(y)))
    noverlap = nperseg // 2 if noverlap is None else int(min(noverlap, nperseg - 1))
    hop = max(1, nperseg - noverlap)
    w = window(window_name, nperseg)
    sft = ss.ShortTimeFFT(w, hop=hop, fs=1.0 / dt, scale_to="magnitude")
    z = sft.stft(y)
    mag = np.abs(z)
    values = {"magnitude": mag, "power": mag ** 2,
              "db": 20 * np.log10(np.maximum(mag, 1e-300))}[scale]
    return {"times": sft.t(len(y)), "freqs": sft.f, "values": values, "complex": z}


# ---------------------------------------------------------------------- Hilbert
def hilbert(y) -> dict:
    """Hilbert transform (imaginary part of the analytic signal), envelope ``|z|``,
    instantaneous phase (rad, unwrapped) and instantaneous frequency (cycles/sample)."""
    y = np.asarray(y, float)
    z = ss.hilbert(y)
    phase = np.unwrap(np.angle(z))
    inst = np.gradient(phase) / (2 * math.pi)
    return {"hilbert": z.imag, "envelope": np.abs(z), "phase": phase, "frequency": inst,
            "analytic": z}


def envelope(y) -> np.ndarray:
    return hilbert(y)["envelope"]


# ---------------------------------------------------------------------- correlation
def correlate(a, b=None, normalize: str = "none", demean: bool = False,
              circular: bool = False, dt: float = 1.0) -> dict:
    """Auto (``b`` None) or cross correlation ``r(k) = Σ a(n+k) b(n)`` for lags
    ``−(N−1)…(N−1)`` (circular: ``0…N−1``), normalised as in :data:`CORR_NORMS`."""
    a = np.asarray(a, float)
    b = a if b is None else np.asarray(b, float)
    if demean:
        a, b = a - a.mean(), b - b.mean()
    n = max(len(a), len(b))
    if circular:
        if len(a) != len(b):
            raise ValueError("Circular correlation needs signals of equal length.")
        r = np.fft.ifft(np.fft.fft(a) * np.conj(np.fft.fft(b))).real
        lags = np.arange(n)
    else:
        r = ss.correlate(a, b, mode="full", method="auto")
        lags = ss.correlation_lags(len(a), len(b), mode="full")
    if normalize == "biased":
        r = r / n
    elif normalize == "unbiased":
        r = r / np.maximum(n - np.abs(lags), 1)
    elif normalize == "coeff":
        d = math.sqrt(float(np.sum(a * a) * np.sum(b * b)))
        r = r / d if d else r
    return {"lag": lags * dt, "lags": lags, "r": r}


def coherence(a, b, dt: float = 1.0, nperseg: int = 64, window_name: str = "hann") -> dict:
    """Magnitude-squared coherence (Welch averaging, ``scipy.signal.coherence``)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    nperseg = int(min(max(4, nperseg), len(a)))
    w = window(window_name, nperseg)
    f, c = ss.coherence(a, b, fs=1.0 / dt, window=w, nperseg=nperseg)
    return {"freq": f, "coherence": c}


# ---------------------------------------------------------------------- convolution
def convolve(signal, response, mode: str = "linear", normalize: bool = False) -> np.ndarray:
    """Convolution of ``signal`` with ``response`` (Origin's Convolution): ``linear``
    (length N+M−1) or ``circular`` (length N); ``normalize`` scales the response to unit
    sum."""
    x = np.asarray(signal, float)
    h = np.asarray(response, float)
    if normalize:
        s = h.sum()
        if s == 0:
            raise ValueError("The response sums to zero and cannot be normalized.")
        h = h / s
    if mode == "circular":
        if len(h) > len(x):
            raise ValueError("The response is longer than the signal.")
        hp = np.zeros(len(x))
        hp[:len(h)] = h
        return np.fft.ifft(np.fft.fft(x) * np.fft.fft(hp)).real
    return ss.fftconvolve(x, h, mode="full")


def deconvolve(y, response, mode: str = "linear", normalize: bool = False,
               regularization: float = 0.0) -> np.ndarray:
    """Deconvolution by FFT division (Origin's Deconvolution). ``linear``: ``y`` is a
    linear convolution of length N+M−1, the result has length N = len(y)−M+1; ``circular``:
    same length. ``regularization`` > 0 gives the Wiener-type ``Y·H* / (|H|² + ε·max|H|²)``
    for noisy data."""
    y = np.asarray(y, float)
    h = np.asarray(response, float)
    if normalize:
        s = h.sum()
        if s == 0:
            raise ValueError("The response sums to zero and cannot be normalized.")
        h = h / s
    m = len(y)
    if len(h) > m:
        raise ValueError("The response is longer than the signal.")
    hp = np.zeros(m)
    hp[:len(h)] = h
    Y, H = np.fft.fft(y), np.fft.fft(hp)
    p = np.abs(H) ** 2
    if regularization > 0:
        X = Y * np.conj(H) / (p + regularization * p.max())
    else:
        if np.min(np.abs(H)) < 1e-12 * max(np.max(np.abs(H)), 1e-300):
            raise ValueError("The response has zeros in its spectrum; use a regularization "
                             "> 0.")
        X = Y / H
    x = np.fft.ifft(X).real
    return x[:m - len(h) + 1] if mode == "linear" else x


def decimate(y, factor: int, ftype: str = "iir", zero_phase: bool = True) -> np.ndarray:
    """Down-sample by an integer ``factor`` after an anti-alias filter
    (``scipy.signal.decimate``: order-8 Chebyshev I IIR or 30-tap FIR)."""
    factor = int(factor)
    if factor < 2:
        return np.asarray(y, float).copy()
    return ss.decimate(np.asarray(y, float), factor, ftype=ftype, zero_phase=zero_phase)


# ---------------------------------------------------------------------- 2D
def fft2(z, shift: bool = True, window_name: str = "rect") -> dict:
    """2D FFT of a matrix: amplitude ``|F|/(R·C)``, phase (degrees), real, imaginary and
    ``log`` = log10(1+|F|) for display; zero frequency centred when ``shift``."""
    z = np.nan_to_num(np.asarray(z, float))
    r, c = z.shape
    w = np.outer(window(window_name, r), window(window_name, c))
    f = np.fft.fft2(z * w)
    if shift:
        f = np.fft.fftshift(f)
    mag = np.abs(f)
    return {"amplitude": mag / (r * c), "phase": np.degrees(np.angle(f)), "real": f.real,
            "imag": f.imag, "log": np.log10(1 + mag), "complex": f}


def ifft2(real, imag=None, shifted: bool = True) -> np.ndarray:
    f = np.asarray(real, float) + 1j * (0 if imag is None else np.asarray(imag, float))
    if shifted:
        f = np.fft.ifftshift(f)
    return np.fft.ifft2(f).real


def correlate2d(a, b=None, normalize: bool = True) -> np.ndarray:
    """2D cross correlation (auto when ``b`` is None), FFT based, ``full`` output;
    ``normalize`` divides by sqrt(Σa²·Σb²) after removing the means."""
    a = np.nan_to_num(np.asarray(a, float))
    b = a if b is None else np.nan_to_num(np.asarray(b, float))
    if normalize:
        a, b = a - a.mean(), b - b.mean()
    r = ss.fftconvolve(a, b[::-1, ::-1], mode="full")
    if normalize:
        d = math.sqrt(float(np.sum(a * a) * np.sum(b * b)))
        r = r / d if d else r
    return r


def filter2d(z, kind: str = "lowpass", cutoff: float = 0.2, cutoff2: float = 0.5) \
        -> np.ndarray:
    """2D FFT filter with an ideal circular mask; cutoffs are fractions of the Nyquist
    radius (``bandpass`` / ``bandblock`` use both)."""
    z = np.asarray(z, float)
    finite = np.isfinite(z)
    fill = np.where(finite, z, np.nanmean(z) if finite.any() else 0.0)
    r, c = z.shape
    fy = np.fft.fftfreq(r)[:, None] / 0.5
    fx = np.fft.fftfreq(c)[None, :] / 0.5
    rad = np.sqrt(fx ** 2 + fy ** 2)
    keep = {"lowpass": rad <= cutoff, "highpass": rad > cutoff,
            "bandpass": (rad >= cutoff) & (rad <= cutoff2),
            "bandblock": (rad < cutoff) | (rad > cutoff2)}[kind]
    out = np.fft.ifft2(np.fft.fft2(fill) * keep).real
    return np.where(finite, out, np.nan)

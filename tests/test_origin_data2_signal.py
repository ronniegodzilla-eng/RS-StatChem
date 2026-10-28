"""WP-2 signal processing: FFT normalisations (Parseval, coherent gain), IFFT round trip,
FFT filters, IIR designs against scipy (zero phase), STFT, Hilbert / envelope, correlation
against numpy, coherence, convolution / deconvolution, decimation, 2D FFT, Savitzky-Golay
coefficients against the Savitzky & Golay (1964) tables (with the Steinier et al. 1972
corrections), LOWESS against statsmodels, binomial smoothing and wavelets (optional
PyWavelets: perfect reconstruction, Donoho-Johnstone universal threshold)."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import signal as ss

from rsstatchem.core.analysis import filters as flt
from rsstatchem.core.analysis import transforms as tr
from rsstatchem.core.analysis import wavelets as wv

RNG = np.random.default_rng(20261027)


# ====================================================================== FFT
def test_fft_amplitude_of_sinusoid_and_dc_and_parseval():
    dt = 1e-3
    t = np.arange(1000) * dt
    y = 3.0 * np.sin(2 * np.pi * 50 * t) + 1.5
    r = tr.fft(y, dt)
    k = int(np.argmax(r["amplitude"][1:])) + 1
    assert r["freq"][k] == pytest.approx(50.0)
    assert r["amplitude"][k] == pytest.approx(3.0, rel=1e-12)
    assert r["amplitude"][0] == pytest.approx(1.5, rel=1e-12)
    assert r["msa"].sum() == pytest.approx(np.mean(y ** 2), rel=1e-12)      # Parseval
    assert (r["psd"] * r["df"]).sum() == pytest.approx(np.mean(y ** 2), rel=1e-12)
    assert r["db"][k] == pytest.approx(20 * math.log10(3.0))
    # Hann window: amplitude still corrected by the coherent gain on a bin frequency
    rh = tr.fft(y, dt, "hann")
    assert rh["amplitude"][k] == pytest.approx(3.0, rel=2e-3)
    w = tr.window("hann", 1000)
    assert rh["msa"].sum() == pytest.approx(np.sum((y * w) ** 2) / np.sum(w ** 2))


def test_fft_two_sided_shift_matches_numpy_and_ifft_round_trip():
    y = RNG.normal(size=64)
    r = tr.fft(y, 0.5, one_sided=False, shift=True)
    ref = np.fft.fftshift(np.fft.fft(y))
    np.testing.assert_allclose(r["real"], ref.real, atol=1e-12)
    np.testing.assert_allclose(r["freq"], np.fft.fftshift(np.fft.fftfreq(64, 0.5)))
    back = tr.ifft(r["real"], r["imag"], r["df"], shifted=True)
    np.testing.assert_allclose(back["real"], y, atol=1e-12)
    assert back["dt"] == pytest.approx(0.5)
    polar = tr.ifft_from_polar(np.fft.ifftshift(r["magnitude"]),
                               np.fft.ifftshift(r["phase"]), r["df"])
    np.testing.assert_allclose(polar["real"], y, atol=1e-12)
    padded = tr.fft(y[:50], pad="pow2")
    assert padded["n_fft"] == 64 and len(padded["freq"]) == 33


def test_fft_filters_lowpass_highpass_band_threshold():
    dt = 0.01
    t = np.arange(512) * dt
    slow, fast = np.sin(2 * np.pi * 1.5625 * t), 0.5 * np.sin(2 * np.pi * 25 * t)
    y = slow + fast + 2.0
    np.testing.assert_allclose(flt.fft_filter(y, dt, "lowpass", 5.0), slow + 2.0, atol=1e-9)
    np.testing.assert_allclose(flt.fft_filter(y, dt, "highpass", 5.0, keep_dc=False), fast,
                               atol=1e-9)
    np.testing.assert_allclose(flt.fft_filter(y, dt, "bandpass", 20, 30, keep_dc=False), fast,
                               atol=1e-9)
    np.testing.assert_allclose(flt.fft_filter(y, dt, "bandblock", 20, 30), slow + 2, atol=1e-9)
    np.testing.assert_allclose(flt.fft_filter(y, dt, "threshold", threshold=0.75),
                               slow + 2.0, atol=1e-9)
    par = flt.fft_filter(y, dt, "parabolic", 50.0, keep_dc=False)
    gain = 1 - (25 / 50) ** 2
    # the parabolic gain is 1 at 0 Hz, so the DC offset always passes
    assert np.max(par - 2.0 - slow * (1 - (1.5625 / 50) ** 2)) == pytest.approx(0.5 * gain,
                                                                                rel=1e-6)


@pytest.mark.parametrize("ftype", list(flt.IIR_TYPES))
@pytest.mark.parametrize("btype", ["lowpass", "highpass", "bandpass", "bandstop"])
def test_iir_designs_match_scipy(ftype, btype):
    fs = 100.0
    cut = [10.0, 20.0] if btype.startswith("band") else 10.0
    sos = flt.iir_design(ftype, btype, 4, cut, fs, rp=0.5, rs=50)
    ref = {"butter": lambda: ss.butter(4, cut, btype, fs=fs, output="sos"),
           "cheby1": lambda: ss.cheby1(4, 0.5, cut, btype, fs=fs, output="sos"),
           "cheby2": lambda: ss.cheby2(4, 50, cut, btype, fs=fs, output="sos"),
           "ellip": lambda: ss.ellip(4, 0.5, 50, cut, btype, fs=fs, output="sos"),
           "bessel": lambda: ss.bessel(4, cut, btype, fs=fs, output="sos",
                                       norm="phase")}[ftype]()
    np.testing.assert_allclose(sos, ref)
    x = RNG.normal(size=400)
    np.testing.assert_allclose(flt.iir_apply(x, sos, False), ss.sosfilt(ref, x))


def test_iir_butterworth_response_zero_phase_and_impulse():
    fs = 1000.0
    sos = flt.iir_design("butter", "lowpass", 5, 50.0, fs)
    r = flt.iir_response(sos, fs, 4096)
    k = int(np.argmin(np.abs(r["freq"] - 50.0)))
    assert r["db"][k] == pytest.approx(-3.0103, abs=0.05)            # -3 dB at the cutoff
    t = np.arange(2000) / fs
    y = np.sin(2 * np.pi * 5 * t)
    out = flt.iir_apply(y, sos, zero_phase=True)                    # no phase lag
    assert np.max(np.abs(out[200:-200] - y[200:-200])) < 1e-3
    lagged = flt.iir_apply(y, sos, zero_phase=False)
    assert np.max(np.abs(lagged[200:-200] - y[200:-200])) > 0.05
    imp = flt.impulse_response(sos, 3000)
    assert imp.sum() == pytest.approx(1.0, abs=1e-6)                # DC gain 1
    assert flt.step_response(sos, 3000)[-1] == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError):
        flt.iir_design("butter", "lowpass", 4, 600.0, fs)


# ====================================================================== transforms
def test_stft_hilbert_envelope():
    fs = 1000.0
    t = np.arange(2000) / fs
    y = np.where(t < 1, np.sin(2 * np.pi * 50 * t), np.sin(2 * np.pi * 200 * t))
    r = tr.stft(y, 1 / fs, 128, 64)
    peak = r["freqs"][np.argmax(r["values"], axis=0)]
    early = peak[(r["times"] > 0.2) & (r["times"] < 0.8)]
    late = peak[(r["times"] > 1.2) & (r["times"] < 1.8)]
    assert np.all(np.abs(early - 50) < 8) and np.all(np.abs(late - 200) < 8)
    am = (1 + 0.5 * np.cos(2 * np.pi * 2 * t)) * np.cos(2 * np.pi * 100 * t)
    env = tr.envelope(am)
    np.testing.assert_allclose(env[100:-100], (1 + 0.5 * np.cos(2 * np.pi * 2 * t))[100:-100],
                               atol=2e-3)
    h = tr.hilbert(np.cos(2 * np.pi * 10 * t))
    np.testing.assert_allclose(h["hilbert"][200:-200], np.sin(2 * np.pi * 10 * t)[200:-200],
                               atol=1e-3)
    np.testing.assert_allclose(h["frequency"][200:-200] * fs, 10, atol=1e-2)


def test_correlation_matches_numpy_and_normalisations():
    a = RNG.normal(size=40)
    b = np.roll(a, 3) + 0.01 * RNG.normal(size=40)
    r = tr.correlate(b, a)
    np.testing.assert_allclose(r["r"], np.correlate(b, a, "full"), atol=1e-10)
    assert r["lags"][np.argmax(r["r"])] == 3
    auto = tr.correlate(a, normalize="coeff")
    assert auto["r"][auto["lags"] == 0][0] == pytest.approx(1.0)
    unb = tr.correlate(a, normalize="unbiased")
    k = 5
    assert unb["r"][unb["lags"] == k][0] == pytest.approx(np.sum(a[k:] * a[:-k]) / (40 - k))
    circ = tr.correlate(a, b, circular=True)
    ref = np.array([np.sum(np.roll(a, -k) * b) for k in range(40)])   # Σ a(n+k) b(n)
    np.testing.assert_allclose(circ["r"], ref, atol=1e-10)


def test_coherence_convolution_deconvolution_decimate():
    x = RNG.normal(size=2048)
    y = ss.lfilter([1, 0.5], [1], x) + 0.01 * RNG.normal(size=2048)
    c = tr.coherence(x, y, 1.0, 128)
    assert np.mean(c["coherence"]) > 0.95
    s = RNG.normal(size=100)
    h = np.array([0.5, 0.3, 0.2])
    full = tr.convolve(s, h)
    np.testing.assert_allclose(full, np.convolve(s, h), atol=1e-12)
    np.testing.assert_allclose(tr.deconvolve(full, h), s, atol=1e-9)
    circ = tr.convolve(s, h, "circular")
    np.testing.assert_allclose(tr.deconvolve(circ, h, "circular"), s, atol=1e-9)
    np.testing.assert_allclose(tr.convolve(s, [2.0, 2.0], normalize=True),
                               np.convolve(s, [0.5, 0.5]))
    np.testing.assert_allclose(tr.decimate(s, 4), ss.decimate(s, 4, ftype="iir",
                                                              zero_phase=True))
    with pytest.raises(ValueError):
        tr.deconvolve(full, np.array([1.0, 1.0]))      # zero at Nyquist without ε


def test_fft2_ifft2_and_filter2d_and_correlate2d():
    z = RNG.normal(size=(16, 20))
    f = tr.fft2(z)
    np.testing.assert_allclose(tr.ifft2(f["real"], f["imag"]), z, atol=1e-12)
    assert f["amplitude"][8, 10] == pytest.approx(abs(z.sum()) / z.size)   # centred DC
    smooth = tr.filter2d(np.ones((8, 8)) * 3, "lowpass", 0.1)
    np.testing.assert_allclose(smooth, 3.0)
    c = tr.correlate2d(z)
    assert c.shape == (31, 39) and c[15, 19] == pytest.approx(1.0)


# ====================================================================== smoothing
@pytest.mark.parametrize("window, table", [
    (5, [-3, 12, 17, 12, -3]),                          # Savitzky & Golay 1964, quadratic
    (7, [-2, 3, 6, 7, 6, 3, -2]),
    (9, [-21, 14, 39, 54, 59, 54, 39, 14, -21]),
    (11, [-36, 9, 44, 69, 84, 89, 84, 69, 44, 9, -36])])
def test_savitzky_golay_coefficients_match_published_tables(window, table):
    norm = sum(table)
    np.testing.assert_allclose(flt.savgol_coefficients(window, 2), np.array(table) / norm,
                               atol=1e-12)
    # cubic gives the same smoothing weights as quadratic (Savitzky-Golay tables I)
    np.testing.assert_allclose(flt.savgol_coefficients(window, 3), np.array(table) / norm,
                               atol=1e-12)


def test_savitzky_golay_first_derivative_table_and_boundaries():
    # 5-point quadratic first derivative: (-2 -1 0 1 2) / 10
    np.testing.assert_allclose(flt.savgol_coefficients(5, 2, 1), np.array([-2, -1, 0, 1, 2])
                               / 10, atol=1e-12)
    x = np.linspace(0, 1, 41)
    y = 3 * x ** 2
    np.testing.assert_allclose(flt.savgol(y, 7, 2, 1, x[1] - x[0]), 6 * x, atol=1e-9)
    for mode in flt.SG_MODES:
        assert flt.savgol(np.sin(x * 6), 7, 2, mode=mode).shape == (41,)


def test_lowess_matches_statsmodels_and_loess_reproduces_quadratics():
    from statsmodels.nonparametric.smoothers_lowess import lowess
    x = np.sort(RNG.uniform(0, 10, 120))
    y = np.sin(x) + RNG.normal(0, 0.3, 120)
    y[7] += 6
    for it in (0, 3):
        ref = lowess(y, x, frac=0.25, it=it, delta=0.0, return_sorted=False)
        np.testing.assert_allclose(flt.loess(x, y, 0.25, 1, it), ref, atol=1e-10)
    q = 0.5 * x ** 2 - x + 2
    np.testing.assert_allclose(flt.loess(x, q, 0.2, 2), q, atol=1e-8)   # degree 2 exact
    shuffled = RNG.permutation(len(x))
    out = flt.smooth_extra(x[shuffled], q[shuffled], "loess", frac=0.2)
    np.testing.assert_allclose(out, q[shuffled], atol=1e-8)            # original order


def test_binomial_and_triangular_smoothing():
    y = np.zeros(11)
    y[5] = 16.0
    np.testing.assert_allclose(flt.binomial(y, 4)[3:8], [1, 4, 6, 4, 1])  # C(4,k)/16·16
    np.testing.assert_allclose(flt.binomial(np.full(9, 2.0), 2), 2.0)
    np.testing.assert_allclose(flt.triangular(y, 5)[3:8], 16 * np.array([1, 2, 3, 2, 1]) / 9)


# ====================================================================== wavelets
def test_wavelets_optional():
    if not wv.available():
        with pytest.raises(wv.WaveletUnavailable):
            wv.dwt(np.ones(8))
        return
    x = RNG.normal(size=256)
    d = wv.dwt(x, "db4")
    np.testing.assert_allclose(wv.idwt(d["approx"], d["detail"], "db4")[:256], x, atol=1e-10)
    dec = wv.decompose(x, "sym4", 4)
    np.testing.assert_allclose(sum(dec["components"].values()), x, atol=1e-10)
    np.testing.assert_allclose(wv.reconstruct(dec["coeffs"], "sym4", n=256), x, atol=1e-10)
    # Donoho-Johnstone: Doppler-like test signal + noise, universal threshold
    n = 1024
    t = np.linspace(0, 1, n)
    clean = np.sqrt(t * (1 - t)) * np.sin(2 * np.pi * 1.05 / (t + 0.05))
    sigma = 0.1
    noisy = clean + sigma * RNG.normal(size=n)
    r = wv.denoise(noisy, "sym8", 5)
    import pywt
    finest = pywt.wavedec(noisy, "sym8", level=5)[-1]
    assert r["thresholds"][0] == pytest.approx(np.median(np.abs(finest)) / 0.6745
                                               * math.sqrt(2 * math.log(n)))
    assert r["thresholds"][0] == pytest.approx(sigma * math.sqrt(2 * math.log(n)), rel=0.15)
    err_noisy = np.sqrt(np.mean((noisy - clean) ** 2))
    err_den = np.sqrt(np.mean((r["signal"] - clean) ** 2))
    assert err_den < 0.6 * err_noisy
    smooth = wv.smooth(noisy, "db4", 2)
    assert np.sqrt(np.mean((smooth - clean) ** 2)) < err_noisy
    c = wv.cwt(np.sin(2 * np.pi * np.arange(512) / 32), 1.0, "morl", n_scales=48)
    best = c["freqs"][np.argmax(c["power"][:, 200:300].mean(axis=1))]
    assert best == pytest.approx(1 / 32, rel=0.15)

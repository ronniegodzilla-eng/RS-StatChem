"""Mathematics and signal processing checked against analytic results, scipy and statsmodels."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import interpolate as si
from scipy import signal as ss

from rsstatchem.core.analysis import mathematics as m
from rsstatchem.core.analysis import signal as sg
from rsstatchem.core.dataset import Dataset

RNG = np.random.default_rng(7)
FWHM_PER_SIGMA = 2 * math.sqrt(2 * math.log(2))


def gauss(x, h, c, s):
    return h * np.exp(-0.5 * ((x - c) / s) ** 2)


# ---------------------------------------------------------------------- mathematics
def test_derivative_exact_for_quadratic_and_uneven_grid():
    x = np.sort(RNG.uniform(0, 5, 40))
    y = 3 * x ** 2 - 2 * x + 1
    np.testing.assert_allclose(m.differentiate(x, y), 6 * x - 2, atol=1e-9)
    np.testing.assert_allclose(m.differentiate(x, y, order=2), 6, atol=1e-6)
    shuffled = RNG.permutation(len(x))
    np.testing.assert_allclose(m.differentiate(x[shuffled], y[shuffled]),
                               (6 * x - 2)[shuffled], atol=1e-9)


def test_derivative_savgol_matches_scipy():
    x = np.linspace(0, 2 * np.pi, 200)
    y = np.sin(x) + RNG.normal(0, 0.01, len(x))
    d = m.differentiate(x, y, method="savgol", window=15, polyorder=3)
    ref = ss.savgol_filter(y, 15, 3, deriv=1, delta=x[1] - x[0], mode="interp")
    np.testing.assert_allclose(d, ref, atol=1e-10)
    assert np.sqrt(np.mean((d - np.cos(x)) ** 2)) < 0.06


def test_integrate_area_baselines():
    x = np.linspace(0, 3, 3001)
    r = m.integrate(x, x ** 2)
    assert r["area"] == pytest.approx(9.0, rel=1e-6)
    r = m.integrate(x, x ** 2, baseline="line")
    assert r["area"] == pytest.approx(9.0 - 13.5, rel=1e-6)
    r = m.integrate(x, x ** 2, x_from=1.0, x_to=2.0)
    assert r["area"] == pytest.approx(7 / 3, rel=1e-6)
    x = np.linspace(0, 2 * np.pi, 4001)
    r = m.integrate(x, np.sin(x))
    assert r["area"] == pytest.approx(0, abs=1e-9)
    assert r["abs_area"] == pytest.approx(4.0, rel=1e-6)
    xg = np.linspace(-10, 10, 2001)
    r = m.integrate(xg, gauss(xg, 2, 0, 1.5) + 1, baseline="constant", value=1)
    assert r["area"] == pytest.approx(2 * 1.5 * math.sqrt(2 * math.pi), rel=1e-6)
    assert r["fwhm"] == pytest.approx(1.5 * FWHM_PER_SIGMA, rel=1e-4)
    assert r["centroid"] == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize("method", ["linear", "cubic", "akima", "pchip"])
def test_interpolate_matches_scipy(method):
    x = np.sort(RNG.uniform(0, 10, 15))
    y = np.sin(x)
    xn = np.linspace(x[0], x[-1], 57)
    ref = {"linear": lambda: np.interp(xn, x, y),
           "cubic": lambda: si.CubicSpline(x, y)(xn),
           "akima": lambda: si.Akima1DInterpolator(x, y)(xn),
           "pchip": lambda: si.PchipInterpolator(x, y)(xn)}[method]()
    np.testing.assert_allclose(m.interpolate(x, y, xn, method), ref, atol=1e-12)
    outside = m.interpolate(x, y, [x[0] - 1, x[-1] + 1], method)
    assert np.all(np.isnan(outside))
    assert np.all(np.isfinite(m.interpolate(x, y, [x[0] - 1, x[-1] + 1], method, True)))


def test_interpolate_extrapolates_line_exactly_and_averages_duplicates():
    x = np.array([0, 1, 1, 2, 3.0])
    y = np.array([0, 1, 3, 4, 6.0])
    np.testing.assert_allclose(m.interpolate(x, y, [1.0]), [2.0])
    np.testing.assert_allclose(m.interpolate([0, 1, 2.0], [1, 3, 5.0], [4.0], "linear", True),
                               [9.0])


def test_normalize():
    y = np.array([1.0, 2, 3, 4])
    np.testing.assert_allclose(m.normalize(y, "max"), y / 4)
    np.testing.assert_allclose(m.normalize(y, "range"), [0, 1 / 3, 2 / 3, 1])
    z = m.normalize(y, "zscore")
    assert z.mean() == pytest.approx(0) and z.std(ddof=1) == pytest.approx(1)
    np.testing.assert_allclose(m.normalize(y, "sum"), y / 10)
    np.testing.assert_allclose(m.normalize(y, "value", value=2), y / 2)
    x = np.linspace(0, 3, 4)
    a = m.normalize(y, "area", x)
    assert np.trapezoid(a, x) == pytest.approx(1)
    with pytest.raises(ValueError):
        m.normalize(np.zeros(3), "max")


# ---------------------------------------------------------------------- smoothing / FFT
def test_smoothers_match_references():
    x = np.linspace(0, 10, 301)
    y = np.sin(x) + RNG.normal(0, 0.2, len(x))
    np.testing.assert_allclose(sg.smooth(x, y, "savgol", 21, 3),
                               ss.savgol_filter(y, 21, 3, mode="interp"))
    avg = sg.smooth(x, y, "average", 5)
    assert avg[100] == pytest.approx(np.mean(y[98:103]))
    med = sg.smooth(x, y, "percentile", 7, percentile=50)
    assert med[50] == pytest.approx(np.median(y[47:54]))
    from statsmodels.nonparametric.smoothers_lowess import lowess
    np.testing.assert_allclose(sg.smooth(x, y, "lowess", frac=0.2),
                               lowess(y, x, frac=0.2, it=3, return_sorted=False))
    for method in ("savgol", "average", "lowess", "fft"):
        s = sg.smooth(x, y, method, 21, 3, cutoff=0.05, frac=0.15)
        assert np.std(s - np.sin(x)) < 0.6 * np.std(y - np.sin(x)), method


def test_fft_lowpass_removes_high_frequency():
    t = np.arange(512) / 512
    slow, fast = np.sin(2 * np.pi * 3 * t), 0.5 * np.sin(2 * np.pi * 120 * t)
    s = sg.smooth(t, slow + fast, "fft", cutoff=0.2)
    assert np.max(np.abs(s - slow)[20:-20]) < 0.02


def test_power_spectrum_amplitude():
    n, fs = 1000, 100.0
    t = np.arange(n) / fs
    y = 0.7 + 2.0 * np.sin(2 * np.pi * 5 * t) + 0.5 * np.cos(2 * np.pi * 12 * t)
    for window in ("none", "hann", "blackman"):
        r = sg.power_spectrum(t, y, window)
        i5, i12 = np.argmin(np.abs(r["freq"] - 5)), np.argmin(np.abs(r["freq"] - 12))
        assert r["freq"][i5] == pytest.approx(5) and r["dt"] == pytest.approx(0.01)
        assert r["amplitude"][i5] == pytest.approx(2.0, rel=0.01), window
        assert r["amplitude"][i12] == pytest.approx(0.5, rel=0.01)
        assert r["power"][i5] == pytest.approx(2.0, rel=0.02)
    r = sg.power_spectrum(t, y, "none", detrend=False)
    assert r["amplitude"][0] == pytest.approx(0.7, rel=1e-6)


# ---------------------------------------------------------------------- baseline / peaks
def _spectrum():
    x = np.linspace(0, 100, 1001)
    base = 5 + 0.03 * x + 0.0004 * x ** 2
    peaks = gauss(x, 10, 30, 2.0) + gauss(x, 6, 55, 3.0) + gauss(x, 8, 80, 1.5)
    return x, base, peaks


def test_baselines_recover_smooth_background():
    x, base, peaks = _spectrum()
    y = base + peaks + RNG.normal(0, 0.02, len(x))
    b_als = sg.baseline(x, y, "als", lam=1e6, p=0.001)
    assert np.sqrt(np.mean((b_als - base) ** 2)) < 0.15
    b_poly = sg.baseline(x, y, "poly", order=2)
    assert np.sqrt(np.mean((b_poly - base) ** 2)) < 0.05


def test_find_peaks_positions_heights_widths_areas():
    x, _, peaks = _spectrum()
    found = sg.find_peaks(x, peaks, min_prominence=0.1)
    assert [round(p["x"]) for p in found] == [30, 55, 80]
    for p, (h, s) in zip(found, [(10, 2.0), (6, 3.0), (8, 1.5)]):
        assert p["y"] == pytest.approx(h, rel=1e-3)
        assert p["fwhm"] == pytest.approx(s * FWHM_PER_SIGMA, rel=5e-3)
        assert p["area"] == pytest.approx(h * s * math.sqrt(2 * math.pi), rel=5e-3)
        assert p["centroid"] == pytest.approx(p["x"], abs=0.05)
    neg = sg.find_peaks(x, -peaks, direction="negative", min_prominence=0.1)
    assert [round(p["x"]) for p in neg] == [30, 55, 80]
    assert sg.find_peaks(x, peaks, min_height=7) and \
        len(sg.find_peaks(x, peaks, min_height=7)) == 2


@pytest.mark.parametrize("shape", ["gauss", "lorentz"])
def test_multi_peak_fit_recovers_overlapping_peaks(shape):
    x = np.linspace(0, 20, 400)
    if shape == "gauss":
        truth = [(8.0, 1.5, 12.0), (11.0, 2.0, 20.0)]
        f = lambda c, w, a: a / (w / FWHM_PER_SIGMA * math.sqrt(2 * math.pi)) * np.exp(  # noqa
            -0.5 * ((x - c) / (w / FWHM_PER_SIGMA)) ** 2)
    else:
        truth = [(8.0, 1.0, 12.0), (11.0, 1.5, 20.0)]
        f = lambda c, w, a: (2 * a / np.pi) * w / (4 * (x - c) ** 2 + w ** 2)  # noqa
    y = 0.5 + sum(f(*t) for t in truth) + RNG.normal(0, 0.03, len(x))
    res = sg.multi_peak_fit(x, y, 2, shape, "constant")
    rows = sg.peak_summary(res)
    for r, (c, w, a) in zip(rows, truth):
        assert abs(r["xc"] - c) < 4 * r["xc_se"]
        assert abs(r["fwhm"] - w) < 4 * r["fwhm_se"]
        assert abs(r["area"] - a) < 4 * r["area_se"]
    assert sum(r["area_pct"] for r in rows) == pytest.approx(100)
    assert res.r2 > 0.999


def test_multi_peak_voigt_and_centres():
    x = np.linspace(0, 20, 400)
    from rsstatchem.core.analysis import functions as fl
    y = 0.2 + fl.LIBRARY["Voigt"](x, 0, 10, 15, 1.2, 0.8) + RNG.normal(0, 0.02, len(x))
    res = sg.multi_peak_fit(x, y, None, "voigt", "constant", centers=[9.5])
    q = res.params
    assert abs(q["xc1"] - 10) < 0.05 and abs(q["A1"] - 15) < 4 * res.errors[-1]


# ---------------------------------------------------------------------- dataset reports
def test_reports_run_and_save():
    x, base, peaks = _spectrum()
    y = base + peaks + RNG.normal(0, 0.02, len(x))
    y[10] = np.nan
    ds = Dataset(pd.DataFrame({"wn": x, "abs": y, "abs2": y * 2}))
    out = m.differentiate_report(ds, "wn", ["abs", "abs2"])
    assert set(out.saved) == {"D1_abs", "D1_abs2"} and np.isnan(out.saved["D1_abs"][1][10])
    assert m.integrate_report(ds, "wn", ["abs"], "line", save_cumulative=True).saved
    out = m.interpolate_report(ds, "wn", "abs", "pchip", n_points=50)
    assert len(out.saved["IP_abs"][1]) == ds.n_cases
    assert m.normalize_report(ds, ["abs"], "range").saved
    assert sg.smooth_report(ds, None, ["abs"], "savgol", 11, 2).saved
    assert sg.fft_report(ds, "wn", "abs").section.items
    out = sg.baseline_report(ds, "wn", "abs", "als", lam=1e6, p=0.001)
    corr = out.saved["BC_abs"][1]
    ds.add_variable("corr", values=corr.to_numpy())
    pk = sg.peaks_report(ds, "wn", "corr", min_prominence=0.1)
    table = next(i for i in pk.section.items if getattr(i, "title", "") == "Peak Analysis")
    assert len(table.rows) == 3
    out = sg.multi_peak_report(ds, "wn", "corr", 3, "gauss", "constant", save_fitted=True)
    assert "PKFIT" in out.saved

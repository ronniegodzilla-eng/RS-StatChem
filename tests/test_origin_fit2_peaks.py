"""WP-1 Peak Analyzer, baselines and spectroscopy helpers.

* NIST StRD Gauss1/2/3 (two Gaussian peaks on an exponential baseline) as the multi-peak
  acceptance test: the Peak Analyzer (anchor-point baseline, 2nd-derivative peak finding,
  Gaussian fit) finds the two peaks near the certified centres, and the certified model
  started from the Peak Analyzer result reproduces all certified values to ≥ 6 digits.
* Baselines: Shirley self-consistency (background built from the cumulative peak area is
  recovered exactly), Tougaard end conditions, arPLS / airPLS / ALS / SNIP / rolling ball /
  anchors / end points on a known sloping background, the lower-envelope anchor filter.
* Find peaks (local maximum, window, 1st / 2nd derivative, residual) including a hidden
  shoulder; integration against analytic Gaussian and Lorentzian areas; fits of every peak
  shape, shared widths, baseline co-fit and fixed centres.
* Spectroscopy: unit conversions (CODATA constants), Kubelka-Munk, absorbance ↔
  transmittance, Tauc band gap of a synthetic direct-gap absorption edge.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from nist_strd import load_nls, lre
from rsstatchem.core.analysis import baselines as bl
from rsstatchem.core.analysis import fitting as fit
from rsstatchem.core.analysis import functions as fl
from rsstatchem.core.analysis import peak_analyzer as pa
from rsstatchem.core.analysis import spectroscopy as sp

RNG = np.random.default_rng(5)
G = fl.LIBRARY["Gaussian"]
L = fl.LIBRARY["Lorentz"]


def _spectrum(noise=0.005):
    x = np.linspace(0, 100, 801)
    base = 0.3 + 0.002 * x
    y = base + G(x, 0, 30, 4, 10) + G(x, 0, 42, 5, 6) + G(x, 0, 46, 5, 4) + G(x, 0, 70, 3, 8)
    return x, y + RNG.normal(0, noise, x.size), base


# ====================================================================== NIST Gauss1-3
@pytest.mark.parametrize("name", ["Gauss1", "Gauss2", "Gauss3"])
def test_nist_gauss_peaks(name):
    d = load_nls(name)
    b = d.certified
    an = pa.PeakAnalysis(d.x, d.y, "fit", "anchors", {"n_anchors": 10, "interp": "spline"},
                         find_opts={"method": "second_derivative", "height_pct": 10,
                                    "max_peaks": 2},
                         fit_opts={"shape": "gauss", "baseline": "none"}).run()
    centres = sorted(p["center"] for p in an.fit.peaks)
    assert len(centres) == 2
    # the anchor baseline differs from the certified exponential one under the overlapping
    # peaks of Gauss2/3, so the Peak Analyzer centres are only close; the certified model,
    # started from them, must then reproduce the certified values
    assert centres == pytest.approx([b[3], b[6]], abs=7.0)
    # the certified model: Gaussian peaks (GaussAmp form: σ = b5/√2) on b1·exp(−b2·x)
    f = fl.parse_function("y = b1*exp(-b2*x) + b3*exp(-(x-b4)^2/b5^2) + b6*exp(-(x-b7)^2/b8^2)")
    p0 = {"b1": 90, "b2": 0.01, "b3": an.fit.peaks[0]["height"], "b4": centres[0],
          "b5": an.fit.peaks[0]["fwhm"] / (2 * math.sqrt(math.log(2))),
          "b6": an.fit.peaks[1]["height"], "b7": centres[1],
          "b8": an.fit.peaks[1]["fwhm"] / (2 * math.sqrt(math.log(2)))}
    res = fit.fit_curve(d.x, d.y, f, params0=p0)
    assert min(lre(res.params[n], c) for n, c in zip(d.names, b)) >= 6


# ====================================================================== baselines
def test_shirley_recovers_its_own_background_and_tougaard_endpoints():
    x = np.linspace(280, 300, 801)
    P = G(x, 0, 290, 1.5, 20)
    cum = np.concatenate([[0], np.cumsum(0.5 * (P[1:] + P[:-1]) * np.diff(x))])
    B = 1 + 0.7 * cum / cum[-1]
    assert np.max(np.abs(bl.shirley(x, P + B) - B)) < 1e-6
    assert np.max(np.abs(bl.shirley(x[::-1], (P + B)[::-1]) - B[::-1])) < 1e-6
    t = bl.tougaard(x, P + B)
    assert t[0] == pytest.approx((P + B)[0]) and t[-1] == pytest.approx((P + B)[-1])
    assert np.all(np.diff(t) >= -1e-12)


@pytest.mark.parametrize("mode,tol", [("line", 0.01), ("endpoints", 0.01), ("anchors", 0.015),
                                      ("arpls", 0.01), ("als", 0.04), ("airpls", 0.04),
                                      ("snip", 0.04), ("rolling_ball", 0.04), ("poly", 0.01)])
def test_baselines_on_a_known_background(mode, tol):
    x, y, base = _spectrum()
    b = bl.create_baseline(x, y, mode)
    far = (np.abs(x - 30) > 15) & (np.abs(x - 44) > 15) & (np.abs(x - 70) > 12)
    assert np.sqrt(np.mean((b - base)[far] ** 2)) < tol
    assert np.median(y - b) < 0.05


def test_anchor_finding_and_interpolation():
    x, y, base = _spectrum()
    ax, ay = bl.find_anchors(x, y, 10, "minima")
    assert ax[0] == x[0] and ax[-1] == x[-1]
    assert np.all(ay - np.interp(ax, x, base) < 0.03)   # no anchor on a peak / valley
    for method in bl.ANCHOR_METHODS:
        ax2, _ = bl.find_anchors(x, y, 8, method)
        assert len(ax2) >= 3
    for interp in ("linear", "spline", "pchip", "fit"):
        b = bl.anchors_baseline(x, y, ax, ay, interp)
        assert b.shape == y.shape and np.all(np.isfinite(b))
    fb = bl.create_baseline(x, y, "anchors", anchors_x=ax, anchors_y=ay, interp="function",
                            function="Line")
    assert np.max(np.abs(fb - base)) < 0.02
    ds = bl.create_baseline(x, y, "dataset", dataset=base)
    assert np.array_equal(ds, base)


# ====================================================================== find / integrate
def test_find_peaks_methods_and_hidden_shoulder():
    x, y, base = _spectrum()
    v = y - base
    for m in ("local_max", "window", "first_derivative"):
        p = pa.find_peaks(x, v, m, height_pct=5)
        assert [round(q["x"]) for q in p] == [30, 42, 70] or \
            [round(q["x"]) for q in p] == [30, 43, 70]
    d2 = pa.find_peaks(x, v, "second_derivative", height_pct=5)
    xs = [q["x"] for q in d2]
    assert len(xs) == 4 and xs[2] == pytest.approx(46, abs=1.0)
    assert d2[2]["hidden"] and not d2[0]["hidden"]
    res = pa.find_peaks(x, v, "residual", height_pct=5)
    assert any(q["hidden"] for q in res) and len(res) == 4
    neg = pa.find_peaks(x, -v, "local_max", "negative", height_pct=5)
    assert len(neg) == 3
    top2 = pa.find_peaks(x, v, "local_max", height_pct=5, max_peaks=2)
    assert [round(q["x"]) for q in top2] == [30, 70]


def test_integration_matches_analytic_areas():
    x = np.linspace(0, 100, 4001)
    y = G(x, 0, 30, 4, 10) + L(x, 0, 70, 2, 5)
    peaks = pa.find_peaks(x, y, "local_max")
    r = pa.integrate_peaks(x, y, peaks, limits=[(10, 50), (50, 99.9)])

    def lor(a, b):   # area of the Lorentzian (centre 70, FWHM 2, area 5) between a and b
        return 5 * (math.atan(2 * (b - 70) / 2) - math.atan(2 * (a - 70) / 2)) / math.pi

    def gau(a, b):   # area of the Gaussian (centre 30, FWHM 4, area 10) between a and b
        s = 4 / (2 * math.sqrt(2 * math.log(2)))
        r2 = s * math.sqrt(2)
        return 5 * (math.erf((b - 30) / r2) - math.erf((a - 30) / r2))
    assert r[0]["area"] == pytest.approx(gau(10, 50) + lor(10, 50), rel=1e-6)
    assert r[1]["area"] == pytest.approx(gau(50, 99.9) + lor(50, 99.9), rel=1e-6)
    assert r[0]["fwhm"] == pytest.approx(4, rel=1e-3)
    assert r[0]["centroid"] == pytest.approx(30, abs=0.1)
    assert sum(q["area_pct"] for q in r) == pytest.approx(100)
    auto = pa.integrate_peaks(x, y + 0.0, peaks)
    assert auto[0]["area"] == pytest.approx(10, rel=1e-2)


# ====================================================================== fit peaks
@pytest.mark.parametrize("shape", list(pa.SHAPES))
def test_fit_every_peak_shape_on_its_own_data(shape):
    spec = pa.SHAPES[shape]
    g = fl.get_function(spec["func"])
    x = np.linspace(0, 60, 1201)
    truth = []
    y = np.zeros_like(x)
    for c, h, w in ((20, 2.0, 4.0), (40, 1.0, 5.0)):
        q = spec["init"](c, h, w)
        if "mu" in q and shape == "pearson7":
            q["mu"] = 2.5
        args = [0.0 if p == "y0" else q[p] for p in g.params]
        y += g(x, *args)
        truth.append(q)
    y += RNG.normal(0, 0.002, x.size)
    found = pa.find_peaks(x, y, "local_max", height_pct=5)
    pf = pa.fit_peaks(x, y, found, shape)
    assert pf.result.r2 > 0.9999
    assert [r["center"] for r in pf.peaks] == pytest.approx([truth[0][spec["center"]],
                                                            truth[1][spec["center"]]], abs=0.05)
    xx = np.linspace(-40, 100, 40001)
    for i, q in enumerate(truth):
        args = [0.0 if p == "y0" else q[p] for p in g.params]
        area = q[spec["area"]] if spec["area"] else np.trapezoid(g(xx, *args), xx)
        assert pf.peaks[i]["area"] == pytest.approx(area, rel=0.01)
        assert pf.peaks[i]["area_se"] > 0


def test_shared_width_baseline_cofit_and_fixed_centres():
    x = np.linspace(0, 60, 601)
    y = 0.5 + 0.01 * x + G(x, 0, 20, 4, 8) + G(x, 0, 35, 4, 5) + RNG.normal(0, 0.002, x.size)
    found = pa.find_peaks(x, y - 0.5 - 0.01 * x, "local_max", height_pct=5)
    pf = pa.fit_peaks(x, y, found, "gauss", baseline="linear", shared_width=True)
    assert "w" in pf.model.params and pf.result.params["w"] == pytest.approx(4, abs=0.02)
    assert pf.result.params["y0"] == pytest.approx(0.5, abs=0.01)
    assert pf.result.params["k"] == pytest.approx(0.01, abs=5e-4)
    assert [r["area"] for r in pf.peaks] == pytest.approx([8, 5], rel=0.01)
    fx = pa.fit_peaks(x, y, [20.3, 35.0], "gauss", baseline="linear", fix_centers=True)
    assert fx.peaks[0]["center"] == 20.3 and not fx.result.free[fx.model.params.index("xc_1")]
    fw = pa.fit_peaks(x, y, found, "lorentz", baseline="linear", fixed_width=3.0)
    assert fw.result.params["w_1"] == 3.0


def test_peak_analysis_goals_and_batch():
    x, y, base = _spectrum()
    for goal in pa.GOALS:
        an = pa.PeakAnalysis(x, y, goal, "line", {"n_end": 20}).run()
        assert an.baseline is not None
        if goal in ("find", "integrate", "fit"):
            assert an.peaks
        if goal == "integrate":
            assert len(an.integration) == len(an.peaks)
        if goal == "fit":
            assert an.fit.result.r2 > 0.99
    t = pa.PeakAnalysis(x, y, "integrate", "line", {"n_end": 20})
    rows = pa.batch(x, [y, 2 * y], ["a", "b"], t)
    assert {r["dataset"] for r in rows} == {"a", "b"}
    a = [r["area"] for r in rows if r["dataset"] == "a"]
    b2 = [r["area"] for r in rows if r["dataset"] == "b"]
    assert len(a) == len(b2) and np.allclose(np.array(b2), 2 * np.array(a), rtol=0.05)


# ====================================================================== spectroscopy
def test_spectroscopy_conversions():
    assert sp.convert(500, "nm", "cm-1") == pytest.approx(20000)
    assert sp.convert(500, "nm", "eV") == pytest.approx(1239.84198 / 500, rel=1e-8)
    assert sp.convert(2.0, "eV", "nm") == pytest.approx(619.920992, rel=1e-8)
    assert sp.convert(sp.convert(1234.5, "cm-1", "THz"), "THz", "cm-1") == pytest.approx(1234.5)
    assert sp.convert(1.0, "um", "nm") == pytest.approx(1000)
    assert sp.kubelka_munk([50.0, 25.0]).tolist() == pytest.approx([0.25, 1.125])
    assert sp.kubelka_munk([0.5], percent=False)[0] == pytest.approx(0.25)
    assert sp.absorbance_to_transmittance([1.0])[0] == pytest.approx(10.0)
    assert sp.transmittance_to_absorbance([10.0])[0] == pytest.approx(1.0)
    assert sp.wavenumber_shift([550.0], 532.0)[0] == pytest.approx(1e7 / 532 - 1e7 / 550)


def test_tauc_band_gap():
    eg = 3.2
    E = np.linspace(2.5, 4.0, 301)
    alpha_hv = np.sqrt(np.clip(E - eg, 0, None)) * 5          # direct gap: (αhν)² ∝ (E − Eg)
    A = alpha_hv / E
    nm = sp.convert(E, "eV", "nm")
    t = sp.tauc(nm, A, "nm", "direct", (3.3, 3.8))
    assert t.band_gap == pytest.approx(eg, abs=1e-6) and t.r2 > 0.999999
    auto = sp.tauc(E, A, "eV", "direct")
    assert auto.band_gap == pytest.approx(eg, abs=0.01)
    with pytest.raises(ValueError):
        sp.tauc(E, A, "eV", "bogus")

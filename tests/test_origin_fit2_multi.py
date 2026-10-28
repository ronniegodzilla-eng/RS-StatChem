"""WP-1 multi-dataset fitting, errors-in-variables and surface fitting.

* Global fit with parameter sharing against a hand-stacked ``least_squares`` problem (values
  and joint-covariance standard errors), and an enzyme-inhibition global fit with a
  different fixed inhibitor concentration per dataset; replica, concatenate, Compare
  Datasets.
* In-house ODR against ``scipy.odr`` (ODRPACK; deprecated, used only as a test reference
  while the test environment still has it) and against York regression.
* York regression on Pearson's data with York's weights: York et al. (2004), Am. J. Phys.
  72, 367 (best line y = 5.4799 − 0.48053·x, σa = 0.2950, σb = 0.0580; Cantrell 2008 gives
  the same values to more digits).
* Implicit circle fit against the exact geometric circle fit; surface fits (Gauss2D, Plane on
  a matrix, the NIST Nelson model) and Simulate Curve / Surface.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy import optimize

from rsstatchem.core.analysis import fit_compare as fc
from rsstatchem.core.analysis import fit_multi as fm
from rsstatchem.core.analysis import fitting as fit
from rsstatchem.core.analysis import functions as fl
from rsstatchem.core.analysis import odr
from rsstatchem.core.analysis import surface_fit as sf

RNG = np.random.default_rng(11)


def _expdec_sets():
    x = np.linspace(0, 10, 40)
    out = []
    for y0, A in [(0.1, 3.0), (0.5, 2.0), (-0.2, 5.0)]:
        out.append((x, y0 + A * np.exp(-x / 2.5) + RNG.normal(0, 0.02, x.size)))
    return out


def test_global_fit_with_shared_parameter_matches_stacked_least_squares():
    data = _expdec_sets()
    mr = fm.global_fit(data, "ExpDec1", shared=["t1"])
    assert mr.names == ["y0_1", "A1_1", "t1", "y0_2", "A1_2", "y0_3", "A1_3"]

    def resid(p):
        return np.concatenate([p[1 + 2 * i] + p[2 + 2 * i] * np.exp(-x / p[0]) - y
                               for i, (x, y) in enumerate(data)])
    ref = optimize.least_squares(resid, [2, 0, 1, 0, 1, 0, 1], method="lm", ftol=1e-15,
                                 xtol=1e-15, gtol=1e-15)
    J = ref.jac
    cov = np.linalg.inv(J.T @ J) * 2 * ref.cost / (120 - 7)
    c = mr.combined
    assert c.params["t1"] == pytest.approx(ref.x[0], rel=1e-9)
    assert c.errors[mr.names.index("t1")] == pytest.approx(math.sqrt(cov[0, 0]), rel=1e-6)
    for i in range(3):
        assert c.params[f"y0_{i + 1}"] == pytest.approx(ref.x[1 + 2 * i], rel=1e-8, abs=1e-10)
        assert mr.results[i].params["t1"] == c.params["t1"]
    assert c.dof == 120 - 7
    rows = mr.table()
    assert len(rows) == 9 and sum(r["shared"] for r in rows) == 3


def test_global_enzyme_inhibition_with_fixed_concentration_per_dataset():
    s = np.array([0.5, 1, 2, 4, 8, 16, 32.0])
    f = fl.LIBRARY["CompInhib"]
    data = [(s, f(s, 10.0, 2.0, 0.8, I) * (1 + RNG.normal(0, 0.002, s.size)))
            for I in (0.0, 1.0, 3.0)]
    mr = fm.global_fit(data, "CompInhib", shared=["Vmax", "Km", "Ki"],
                       fixed=[{"I": 0.0}, {"I": 1.0}, {"I": 3.0}])
    p = mr.combined.params
    assert p["Vmax"] == pytest.approx(10.0, rel=0.01)
    assert p["Km"] == pytest.approx(2.0, rel=0.02)
    assert p["Ki"] == pytest.approx(0.8, rel=0.02)
    assert [r.params["I"] for r in mr.results] == [0.0, 1.0, 3.0]


def test_independent_concatenate_replica_and_compare_datasets():
    data = _expdec_sets()
    ind = fm.independent_fit(data, "ExpDec1")
    assert len(ind.results) == 3
    for r, (x, y) in zip(ind.results, data):
        assert r.params == pytest.approx(fit.fit_curve(x, y, "ExpDec1").params)
    con = fm.concatenate_fit(data[:1] * 2, "ExpDec1")
    assert con.combined.params["t1"] == pytest.approx(ind.results[0].params["t1"], rel=1e-8)
    x = np.linspace(0, 20, 400)
    g = fl.LIBRARY["Gauss"]
    y = 0.2 + g(x, 0, 6, 1.5, 3) + g(x, 0, 12, 2.0, 5) + RNG.normal(0, 0.005, x.size)
    rep = fm.replica_fit(x, y, "Gauss", 2)
    centres = sorted([rep.params["xc_1"], rep.params["xc_2"]])
    assert centres == pytest.approx([6, 12], abs=0.02) and rep.params["y0"] == pytest.approx(
        0.2, abs=0.01)
    same = fc.compare_datasets([data[0], (data[0][0], data[0][1] + RNG.normal(0, .02, 40))],
                               "ExpDec1")
    diff = fc.compare_datasets(data[:2], "ExpDec1")
    assert same["same"] and not diff["same"] and diff["p"] < 1e-6


# ====================================================================== ODR / York
X_P = np.array([0, 0.9, 1.8, 2.6, 3.3, 4.4, 5.2, 6.1, 6.5, 7.4])
Y_P = np.array([5.9, 5.4, 4.4, 4.6, 3.5, 3.7, 2.8, 2.8, 2.4, 1.5])
WX = np.array([1000, 1000, 500, 800, 200, 80, 60, 20, 1.8, 1.0])
WY = np.array([1, 1.8, 4, 8, 20, 20, 70, 70, 100, 500.0])


def test_york_regression_pearson_data():
    r = odr.york_fit(X_P, Y_P, 1 / np.sqrt(WX), 1 / np.sqrt(WY))
    assert r.slope == pytest.approx(-0.480533, abs=5e-7)
    assert r.intercept == pytest.approx(5.479910, abs=5e-6)
    assert r.se_slope == pytest.approx(0.0580, abs=5e-4)
    assert r.se_intercept == pytest.approx(0.2950, abs=5e-4)
    # with uniform errors York reduces to orthogonal regression (major axis)
    x = np.linspace(0, 10, 30)
    y = 1 + 0.7 * x + RNG.normal(0, 0.3, 30)
    yu = odr.york_fit(x, y, 1.0, 1.0)
    sxx, syy = np.var(x), np.var(y)
    sxy = np.mean((x - x.mean()) * (y - y.mean()))
    b_major = (syy - sxx + math.sqrt((syy - sxx) ** 2 + 4 * sxy ** 2)) / (2 * sxy)
    assert yu.slope == pytest.approx(b_major, rel=1e-10)


def test_odr_matches_york_and_scipy_odr():
    r = odr.odr_fit(X_P, Y_P, "Line", wx=WX, wy=WY, scale_errors=False)
    y = odr.york_fit(X_P, Y_P, 1 / np.sqrt(WX), 1 / np.sqrt(WY))
    assert r.params["Slope"] == pytest.approx(y.slope, rel=1e-8)
    assert r.params["Intercept"] == pytest.approx(y.intercept, rel=1e-8)
    assert r.errors[1] == pytest.approx(y.se_slope, rel=1e-6)
    assert r.sum_square == pytest.approx(y.S, rel=1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sodr = pytest.importorskip("scipy.odr")
    rs = odr.odr_fit(X_P, Y_P, "Line", wx=WX, wy=WY)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o = sodr.ODR(sodr.Data(X_P, Y_P, wd=WX, we=WY),
                     sodr.Model(lambda b, x: b[0] + b[1] * x), beta0=[5.0, -0.5],
                     sstol=1e-15, partol=1e-15, maxit=1000).run()
    assert rs.values == pytest.approx(o.beta, rel=1e-6)
    assert rs.errors == pytest.approx(o.sd_beta, rel=1e-5)
    assert rs.res_var == pytest.approx(o.res_var, rel=1e-8)
    # nonlinear model with errors in x and y
    x = np.linspace(0, 5, 30)
    yv = 2 * np.exp(-0.6 * x) + RNG.normal(0, 0.02, 30)
    xn = x + RNG.normal(0, 0.05, 30)
    rn = odr.odr_fit(xn, yv, "y = a*exp(-b*x)", sx=0.05, sy=0.02)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        on = sodr.ODR(sodr.RealData(xn, yv, sx=0.05, sy=0.02),
                      sodr.Model(lambda b, t: b[0] * np.exp(-b[1] * t)), beta0=[1.5, 0.5],
                      sstol=1e-15, partol=1e-15, maxit=1000).run()
    assert rn.values == pytest.approx(on.beta, rel=1e-6)
    assert rn.errors == pytest.approx(on.sd_beta, rel=1e-4)
    assert rn.sum_square == pytest.approx(on.sum_square, rel=1e-8)


def test_implicit_circle_and_ellipse():
    t = RNG.uniform(0, 2 * np.pi, 50)
    cx = 3 + 2 * np.cos(t) + RNG.normal(0, 0.03, 50)
    cy = -1 + 2 * np.sin(t) + RNG.normal(0, 0.03, 50)
    r = odr.implicit_fit(cx, cy, "Circle")
    ref = optimize.least_squares(lambda p: np.hypot(cx - p[0], cy - p[1]) - p[2], [3, -1, 2],
                                 ftol=1e-15, xtol=1e-15).x
    assert [r.params[k] for k in ("xc", "yc", "r")] == pytest.approx(ref, rel=1e-7)
    assert np.all(np.abs(r.distances - np.abs(np.hypot(cx - ref[0], cy - ref[1]) - ref[2]))
                  < 1e-6)
    ex = 1 + 3 * np.cos(t) + RNG.normal(0, 0.01, 50)
    ey = 2 + 1.5 * np.sin(t) + RNG.normal(0, 0.01, 50)
    e = odr.implicit_fit(ex, ey, "Ellipse")
    assert [e.params[k] for k in ("xc", "yc", "a", "b")] == pytest.approx([1, 2, 3, 1.5],
                                                                          abs=0.01)
    cx_, cy_ = r.curve()
    assert len(cx_) > 50


# ====================================================================== surfaces
def test_surface_fits_and_simulation():
    X = RNG.uniform(-3, 3, (400, 2))
    f = sf.SURFACES["Gauss2D"]
    z = f(X, 0.5, 2.0, 0.3, 0.8, -0.4, 1.2) + RNG.normal(0, 0.005, 400)
    r = sf.surface_fit(X[:, 0], X[:, 1], z, "Gauss2D")
    assert [r.params[k] for k in ("z0", "A", "xc", "w1", "yc", "w2")] == pytest.approx(
        [0.5, 2.0, 0.3, 0.8, -0.4, 1.2], abs=0.01)
    gx, gy, Z = sf.simulate_surface("Plane", {"z0": 1, "a": 2, "b": -1}, (0, 1), (0, 2), 5, 7)
    assert Z.shape == (7, 5) and Z[0, -1] == pytest.approx(3.0)
    m = sf.matrix_fit(Z, (0, 1, 0, 2), "Plane")
    assert m.params == pytest.approx({"z0": 1, "a": 2, "b": -1})
    u = sf.surface_fit(X[:, 0], X[:, 1], 1 + X[:, 0] * X[:, 1], "z = c + d*x*y")
    assert u.params == pytest.approx({"c": 1.0, "d": 1.0})
    for name, g in sf.SURFACES.items():
        assert g.nvar == 2 and g.jac is not None, name
    xx, yy = sf.simulate_curve("Gauss", {"y0": 0, "xc": 0, "w": 1, "A": 1}, (-3, 3), 61)
    assert yy[30] == pytest.approx(1 / math.sqrt(math.pi / 2))

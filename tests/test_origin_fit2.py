"""WP-1 NLFit upgrades, verified against the NIST Statistical Reference Datasets.

* NIST StRD nonlinear regression: all 27 datasets from both NIST starting points; the
  certified parameters must be matched to ≥ 6 significant digits (log relative error, LRE)
  for every dataset — lower, average and higher difficulty — and the certified standard
  deviations to ≥ 6 digits except Lanczos1, whose data are exact to 13 digits so that its
  certified SDs (≈1e-13) carry only ~3 meaningful digits (NIST notes this).
* NIST StRD linear regression: Norris, Pontius, NoInt1/2, Filip (10th degree), Longley
  (multiple regression) and Wampler1–5 (5th degree) — ≥ 7 digits.
* The extended function library: exact (automatic-differentiation) Jacobians against central
  differences, and a round-trip fit of every built-in function from its automatic guess.
* Linear constraints, Downhill Simplex start, weighting methods, information criteria
  (differences equal statsmodels' likelihood-based AIC/BIC), F-test by hand, derived
  parameters, correlation and dependency.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import optimize

from nist_strd import NLS_MODELS, load_lls, load_nls, lre
from rsstatchem.core.analysis import fit_compare as fc
from rsstatchem.core.analysis import fitting as fit
from rsstatchem.core.analysis import functions as fl

NLS = list(NLS_MODELS)


# ====================================================================== NIST nonlinear
@pytest.mark.parametrize("start", [1, 2])
@pytest.mark.parametrize("name", NLS)
def test_nist_strd_nonlinear(name, start):
    eq, difficulty = NLS_MODELS[name]
    d = load_nls(name)
    if name == "Nelson":   # log[y] = b1 - b2*x1*exp(-b3*x2): a surface z = f(x, y)
        f = fl.parse_function(eq, ("x", "y"))
        x, y = d.x, np.log(d.y)
    else:
        f = fl.parse_user_function(eq)
        x, y = d.x, d.y
    p0 = dict(zip(d.names, d.start1 if start == 1 else d.start2))
    res = fit.fit_curve(x, y, f, params0=p0)
    vals = [res.params[n] for n in d.names]
    errs = [res.errors[f.params.index(n)] for n in d.names]
    digits = min(lre(v, c) for v, c in zip(vals, d.certified))
    assert digits >= 6.0, f"{name} start {start}: {digits:.1f} digits"
    if name != "Lanczos1":
        sd_digits = min(lre(v, c) for v, c in zip(errs, d.sd))
        assert sd_digits >= 6.0, f"{name} SD: {sd_digits:.1f} digits"
        assert lre(res.rss, d.rss) >= 6.0
    assert res.success


def test_nist_certified_residual_sd():
    d = load_nls("Misra1a")
    res = fit.fit_curve(d.x, d.y, NLS_MODELS["Misra1a"][0], params0=dict(zip(d.names, d.start1)))
    assert lre(res.rmse, d.resid_sd) >= 8 and res.dof == 12


# ====================================================================== NIST linear
LLS = [("Norris", "Line", 12), ("Pontius", "Poly2", 11), ("NoInt1", "noint", 12),
       ("NoInt2", "noint", 12), ("Filip", "Poly10", 7), ("Longley", "mlr", 10),
       ("Wampler1", "Poly5", 9), ("Wampler2", "Poly5", 12), ("Wampler3", "Poly5", 10),
       ("Wampler4", "Poly5", 9), ("Wampler5", "Poly5", 7)]


@pytest.mark.parametrize("name,model,need", LLS)
def test_nist_strd_linear(name, model, need):
    d = load_lls(name)
    if model == "mlr":
        res = fit.fit_mlr(d.x, d.y)
    elif model == "noint":
        res = fit.fit_mlr(d.x[:, None], d.y, intercept=False)
    else:
        res = fit.fit_curve(d.x, d.y, model)
    assert min(lre(v, c) for v, c in zip(res.values, d.certified)) >= need
    assert min(lre(v, c) for v, c in zip(res.errors, d.sd)) >= need
    if d.resid_sd > 0:
        assert lre(res.rmse, d.resid_sd) >= need


# ====================================================================== function library
def test_library_size_and_categories():
    assert fl.library_size() >= 120
    for cat in ("Origin Basic Functions", "Peak Functions", "Chromatography", "Enzyme Kinetics",
                "Pharmacology", "Rheology", "Waveform", "Piecewise", "Statistics",
                "Electrophysiology", "Logarithm", "Spectroscopy"):
        assert cat in fl.CATEGORIES and fl.functions_in(cat)
    assert "Gauss" in [f.name for f in fl.functions_in("Origin Basic Functions")]
    assert fl.get_function("EMG").name == "GaussMod" and fl.get_function("Cubic").name == "Poly3"


def _check_jac(f, x, p):
    J = f.jacobian(x, p)
    Jn = fl.FitFunction(f.name, f.category, f.formula, f.params, f.func).jacobian(x, p)
    ok = np.isfinite(J) & np.isfinite(Jn)
    scale = np.maximum(np.nanmax(np.abs(np.where(ok, Jn, 0)), axis=0), 1e-12)
    return float(np.max(np.abs(np.where(ok, J - Jn, 0)) / scale))


def test_analytic_jacobians_match_finite_differences():
    rng = np.random.default_rng(4)
    checked = 0
    for f in fl.LIBRARY.values():
        if f.jac is None:
            continue
        x = np.linspace(0.2, 0.9, 9) if f.name == "Bradley" else np.linspace(1.3, 9.7, 9)
        p = rng.uniform(0.6, 1.6, len(f.params))
        if f.name == "Eyring":
            x, p = np.linspace(280, 350, 9), np.array([5e4, -20.0])
        if f.name == "PearsonVII":
            p[f.params.index("mu")] = 1.7
        assert _check_jac(f, x, p) < 1e-5, f.name
        checked += 1
    assert checked >= 80


TRUE = {"y0": 0.5, "A": 3.0, "xc": 5.0, "w": 1.5, "t1": 2.0, "t2": 6.0, "t3": 15.0, "A1": 2.0,
        "A2": 1.0, "A3": 0.5, "x0": 1.0, "k": 0.8, "n": 2.0, "a": 2.0, "b": 0.5, "c": 0.3,
        "d": 0.2, "mu": 0.5, "p": 0.4}
OVERRIDE = {"Richards": {"d": 2.0}, "Power1": {"xc": 0.0}, "MnMolecular": {"xc": 0.0},
            "Langevin": {"xc": 0.0}, "SWeibull1": {"xc": 0.5}, "ECS": {"a3": 0.1, "a4": 0.05},
            "GCAS": {"a3": 0.1, "a4": 0.05}, "PearsonVII": {"mu": 1.5},
            "TwoSiteComp": {"logx0_1": 3.0, "logx0_2": 7.0, "f": 0.4},
            "Carreau": {"n": 0.4, "lam": 1.0}, "CarreauYasuda": {"n": 0.4, "lam": 1.0, "a": 2.0},
            "VFT": {"T0": -5.0, "B": 10.0}}
DOMAIN = {"Bradley": (0.05, 0.9), "BET": (0.05, 0.6), "Arrhenius": (280, 360),
          "Eyring": (280, 360)}


@pytest.mark.parametrize("name", sorted(n for n in fl.LIBRARY if n not in ("Constant",
                                                                          "InvsPoly")))
def test_every_library_function_recovers_its_curve(name):
    """Data generated by the formula are fitted from the automatic initial values."""
    f = fl.LIBRARY[name]
    x = np.linspace(*DOMAIN.get(name, (1.0, 10.0)), 60)
    p = {k: TRUE.get(k, 1.0 + 0.3 * i) for i, k in enumerate(f.params)}
    p.update(OVERRIDE.get(name, {}))
    y = f(x, *[p[k] for k in f.params])
    assert np.all(np.isfinite(y)) and np.ptp(y) > 0
    res = fit.fit_curve(x, y, f)
    assert np.max(np.abs(res.fitted - y)) / np.ptp(y) < (1e-4 if name == "ExpDec3" else 1e-6)


def test_overparametrised_invspoly_and_constant():
    f = fl.LIBRARY["InvsPoly"]
    x = np.linspace(1, 10, 60)
    y = f(x, 0.5, 5.0, 1.5, 3.0, 1.0, 0.1, 0.01)
    res = fit.fit_curve(x, y, f, fixed={"w": 1.5},
                        params0={"y0": 0.4, "xc": 5.1, "A": 2.5, "A1": 1.2, "A2": 0.05, "A3": 0.0})
    assert res.params["A2"] == pytest.approx(0.1, rel=1e-5)
    c = fit.fit_curve(x, np.full(60, 2.0) + np.sin(x) * 1e-3, "Constant")
    assert c.params["y0"] == pytest.approx(2.0, abs=1e-3)


@pytest.mark.parametrize("name,area_key", [("Gauss", "A"), ("Gaussian", "A"), ("Lorentz", "A"),
                                           ("Voigt", "A"), ("PsdVoigt1", "A"),
                                           ("PsdVoigt2", "A"), ("PearsonVII", "A"),
                                           ("GaussMod", "A"), ("Laplace", "A")])
def test_peak_functions_area_parameter_is_the_area(name, area_key):
    f = fl.LIBRARY[name]
    p = {"y0": 0.0, "xc": 0.0, "w": 1.2, "A": 2.5, "wG": 0.9, "wL": 0.7, "mu": 1.8, "t0": 0.6}
    if name.startswith("PsdVoigt1"):
        p["mu"] = 0.4
    if name == "PsdVoigt2":
        p["mu"] = 0.3
    args = [p[k] for k in f.params]
    from scipy import integrate
    area, _ = integrate.quad(lambda t: float(f(np.array([t]), *args)[0]), -np.inf, np.inf,
                             limit=400)
    assert area == pytest.approx(2.5, rel=2e-6)


def test_peak_widths_and_derived_parameters():
    x = np.linspace(-10, 10, 2001)
    g = fl.LIBRARY["Gaussian"](x, 0, 0, 2.0, 1.0)
    half = x[g >= g.max() / 2]
    assert half[-1] - half[0] == pytest.approx(2.0, abs=0.02)
    pv7 = fl.LIBRARY["PearsonVII"](x, 0, 0, 1.0, 2.0, 3.0)
    half = x[pv7 >= pv7.max() / 2]
    assert half[-1] - half[0] == pytest.approx(2.0, abs=0.02)
    y = fl.LIBRARY["GaussAmp"](x, 0.1, 1.0, 1.3, 4.0) + np.sin(7 * x) * 1e-4
    res = fit.fit_curve(x, y, "GaussAmp")
    d = dict((n, (v, se)) for n, v, se in res.derived())
    assert d["FWHM"][0] == pytest.approx(2 * math.sqrt(2 * math.log(2)) * 1.3, rel=1e-4)
    assert d["Area"][0] == pytest.approx(4.0 * 1.3 * math.sqrt(2 * math.pi), rel=1e-4)
    assert d["Area"][1] > 0


def test_parser_conditionals_and_functions():
    f = fl.parse_user_function("y = if(x < c, a*x, a*c + b*(x - c))")
    assert f.params == ("c", "a", "b")
    x = np.array([1.0, 3.0])
    assert list(f(x, 2.0, 1.0, 3.0)) == [1.0, 5.0]
    J = f.jacobian(x, [2.0, 1.0, 3.0])
    assert J[1].tolist() == [1.0 - 3.0, 2.0, 1.0]
    g = fl.parse_function("z = a*x + b*y^2", ("x", "y"))
    assert g.nvar == 2 and g(np.array([[1.0, 2.0]]), 1.0, 2.0)[0] == 9.0
    with pytest.raises(fl.FunctionSyntaxError):
        fl.parse_user_function("y = a*x if a else x")


# ====================================================================== NLFit options
def test_linear_equality_and_inequality_constraints():
    rng = np.random.default_rng(2)
    x = np.linspace(0, 10, 50)
    y = 1.0 + 2.0 * np.exp(-x / 3.0) + rng.normal(0, 0.01, x.size)
    res = fit.fit_curve(x, y, "y = a*exp(-x/t) + b", constraints="a + b = 3.5")
    assert res.params["a"] + res.params["b"] == pytest.approx(3.5, abs=1e-10)
    assert res.dof == 50 - 3 + 1 and res.n_active == 1
    # reference: substitute b = 3.5 − a and fit the reduced model
    ref = optimize.least_squares(lambda p: p[0] * np.exp(-x / p[1]) + 3.5 - p[0] - y,
                                 [2.0, 3.0], method="lm", xtol=1e-15, ftol=1e-15)
    assert res.params["a"] == pytest.approx(ref.x[0], rel=1e-8)
    assert res.params["t"] == pytest.approx(ref.x[1], rel=1e-8)
    J = ref.jac
    cov = np.linalg.inv(J.T @ J) * 2 * ref.cost / (50 - 2)
    assert res.errors[res.names.index("a")] == pytest.approx(math.sqrt(cov[0, 0]), rel=1e-5)
    assert res.errors[res.names.index("b")] == pytest.approx(math.sqrt(cov[0, 0]), rel=1e-5)
    ineq = fit.fit_curve(x, y, "y = a*exp(-x/t) + b", constraints=["t <= 2.5"])
    assert ineq.params["t"] == pytest.approx(2.5, abs=1e-8) and ineq.n_active == 1
    free = fit.fit_curve(x, y, "y = a*exp(-x/t) + b", constraints=["t <= 10"])
    assert free.n_active == 0 and free.params["t"] == pytest.approx(3.0, rel=0.02)
    with pytest.raises(ValueError, match="not linear"):
        fit.fit_curve(x, y, "y = a*exp(-x/t) + b", constraints=["a*t = 1"])
    with pytest.raises(ValueError, match="unknown"):
        fit.fit_curve(x, y, "y = a*exp(-x/t) + b", constraints=["q < 1"])


def test_simplex_algorithm_and_default_fixed_offset():
    x = np.linspace(0, 10, 40)
    y = 0.3 + 2.0 * np.exp(-(x - 0.0) / 1.7)
    r1 = fit.fit_curve(x, y, "ExpDecay1")
    assert not r1.free[r1.names.index("x0")] and r1.params["t1"] == pytest.approx(1.7)
    r2 = fit.fit_curve(x, y, "y = a*exp(-x/t) + c", params0={"a": 1, "t": 5, "c": 0},
                       algorithm="simplex")
    assert r2.params["t"] == pytest.approx(1.7, rel=1e-8)


def test_weighting_methods():
    x = np.linspace(1, 10, 30)
    y = 3 * x + 1 + np.sin(x) * 0.1
    w = fit.make_weights("experimental", y)
    assert np.allclose(w, 1 / y ** 2)
    r = fit.fit_curve(x, y, "Line", weights=w)
    W = np.diag(w)
    B = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.solve(B.T @ W @ B, B.T @ W @ y)
    assert r.values == pytest.approx(beta, rel=1e-10)
    assert fit.make_weights("arbitrary", y, np.full(30, 2.0)).tolist() == [2.0] * 30


def test_information_criteria_against_statsmodels_and_f_test():
    import statsmodels.api as sm
    rng = np.random.default_rng(9)
    x = np.linspace(0, 5, 40)
    y = 1 + 0.5 * x + 0.3 * x ** 2 + rng.normal(0, 0.2, x.size)
    r1, r2 = fit.fit_curve(x, y, "Line"), fit.fit_curve(x, y, "Poly2")
    s1 = sm.OLS(y, sm.add_constant(x)).fit()
    s2 = sm.OLS(y, np.column_stack([np.ones_like(x), x, x ** 2])).fit()
    assert r2.aic - r1.aic == pytest.approx(s2.aic - s1.aic, rel=1e-10)
    assert r2.bic - r1.bic == pytest.approx(s2.bic - s1.bic, rel=1e-10)
    n, K = 40, 3
    assert r1.aic == pytest.approx(n * math.log(r1.rss / n) + 2 * K)
    assert r1.aicc == pytest.approx(r1.aic + 2 * K * (K + 1) / (n - K - 1))
    t = fc.f_test(r1, r2)
    F = ((r1.rss - r2.rss) / 1) / (r2.rss / 37)
    assert t["F"] == pytest.approx(F) and t["df1"] == 1 and t["df2"] == 37
    assert t["F"] == pytest.approx(s2.compare_f_test(s1)[0], rel=1e-9)
    cmp = fc.compare_models(x, y, ["Line", "Poly2"])
    assert cmp.best == "Poly2" and cmp.ftest["p"] < 1e-6
    w = [r["w_aicc"] for r in cmp.rows]
    assert sum(w) == pytest.approx(1.0)
    ranked = fc.rank_models(x, y, ["Line", "Poly2", "Poly3", "ExpGro1"])
    assert ranked[0]["model"] in ("Poly2", "ExpGro1") and ranked[0]["rank"] == 1


def test_correlation_dependency_and_covariance():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 10, 60)
    y = 2 + 3 * np.exp(-x / 2) + rng.normal(0, 0.02, x.size)
    r = fit.fit_curve(x, y, "ExpDec1")
    c = r.correlation()
    assert np.allclose(np.diag(c), 1.0) and np.allclose(c, c.T)
    dep = r.dependency()
    C = r.cov
    assert dep[0] == pytest.approx(1 - 1 / (C[0, 0] * np.linalg.inv(C)[0, 0]))
    assert np.all((dep >= 0) & (dep < 1))

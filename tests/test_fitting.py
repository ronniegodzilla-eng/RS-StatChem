"""Curve fitting and calibration checked against statsmodels, scipy, numpy and a hand-worked
calibration example (Miller & Miller, Statistics and Chemometrics for Analytical Chemistry)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import optimize, stats

from rsstatchem.core.analysis import calibration as cal
from rsstatchem.core.analysis import functions as fl
from rsstatchem.core.analysis.fitting import FitOptions, fit_curve, make_weights, run_fit
from rsstatchem.core.dataset import Dataset

RNG = np.random.default_rng(20261024)


def _line_data(n=25):
    x = np.linspace(0, 10, n)
    y = 1.5 + 0.8 * x + RNG.normal(0, 0.4, n)
    return x, y


# ---------------------------------------------------------------------- linear / polynomial
def test_linear_matches_statsmodels_ols():
    x, y = _line_data()
    res = fit_curve(x, y, "Line")
    ols = sm.OLS(y, sm.add_constant(x)).fit()
    np.testing.assert_allclose(res.values, ols.params, rtol=1e-10)
    np.testing.assert_allclose(res.errors, ols.bse, rtol=1e-8)
    np.testing.assert_allclose(res.p_values, ols.pvalues, rtol=1e-6)
    assert res.r2 == pytest.approx(ols.rsquared, rel=1e-10)
    assert res.adj_r2 == pytest.approx(ols.rsquared_adj, rel=1e-10)
    assert res.rss == pytest.approx(ols.ssr, rel=1e-10)
    assert res.red_chisq == pytest.approx(ols.mse_resid, rel=1e-10)
    lr = stats.linregress(x, y)
    assert res.pearson_r == pytest.approx(lr.rvalue, rel=1e-10)
    model, error, total = res.anova()
    assert model[2] == pytest.approx(ols.ess, rel=1e-10)
    assert model[4] == pytest.approx(ols.fvalue, rel=1e-8)
    assert model[5] == pytest.approx(ols.f_pvalue, rel=1e-6, abs=1e-300)
    assert total[1] == len(x) - 1


def test_linear_bands_match_statsmodels_prediction():
    x, y = _line_data()
    res = fit_curve(x, y, "Line")
    ols = sm.OLS(y, sm.add_constant(x)).fit()
    xn = np.array([-1.0, 2.5, 7.0, 12.0])
    frame = ols.get_prediction(sm.add_constant(xn)).summary_frame(alpha=0.05)
    lo, hi = res.confidence_band(xn, 0.95)
    np.testing.assert_allclose(lo, frame["mean_ci_lower"], rtol=1e-8)
    np.testing.assert_allclose(hi, frame["mean_ci_upper"], rtol=1e-8)
    lo, hi = res.prediction_band(xn, 0.95)
    np.testing.assert_allclose(lo, frame["obs_ci_lower"], rtol=1e-8)
    np.testing.assert_allclose(hi, frame["obs_ci_upper"], rtol=1e-8)


def test_instrumental_weighting_matches_wls_and_curve_fit():
    x, _ = _line_data(30)
    sigma = 0.1 + 0.05 * x
    y = 2.0 - 0.5 * x + RNG.normal(0, 1, len(x)) * sigma
    w = make_weights("instrumental", y, sigma)
    res = fit_curve(x, y, "Line", weights=w)
    wls = sm.WLS(y, sm.add_constant(x), weights=1 / sigma ** 2).fit()
    np.testing.assert_allclose(res.values, wls.params, rtol=1e-10)
    np.testing.assert_allclose(res.errors, wls.bse, rtol=1e-8)
    absolute = fit_curve(x, y, "Line", weights=w, scale_errors=False)
    p, cov = optimize.curve_fit(lambda t, a, b: a + b * t, x, y, sigma=sigma,
                                absolute_sigma=True)
    np.testing.assert_allclose(absolute.values, p, rtol=1e-7)
    np.testing.assert_allclose(absolute.errors, np.sqrt(np.diag(cov)), rtol=1e-6)
    assert res.rss == pytest.approx(np.sum((y - wls.fittedvalues) ** 2 / sigma ** 2))


def test_fixed_intercept_and_slope():
    x, y = _line_data()
    res = fit_curve(x, y, "Line", fixed={"Intercept": 0.0})
    ols = sm.OLS(y, x[:, None]).fit()
    assert res.values[1] == pytest.approx(ols.params[0], rel=1e-10)
    assert res.errors[1] == pytest.approx(ols.bse[0], rel=1e-8)
    assert math.isnan(res.errors[0]) and res.dof == len(x) - 1
    res = fit_curve(x, y, "Line", fixed={"Slope": 0.8})
    assert res.values[0] == pytest.approx(np.mean(y - 0.8 * x), rel=1e-10)


@pytest.mark.parametrize("order", [2, 3, 5, 9])
def test_polynomial_matches_polyfit_and_ols(order):
    x = np.linspace(0.5, 20, 40)
    coef = RNG.normal(0, 1, order + 1) / (10.0 ** np.arange(order + 1))
    y = np.polynomial.polynomial.polyval(x, coef) + RNG.normal(0, 0.05, len(x))
    res = fit_curve(x, y, fl.polynomial(order))
    ref = np.polyfit(x, y, order)[::-1]
    np.testing.assert_allclose(res.values, ref, rtol=1e-5, atol=1e-12)
    if order <= 5:
        ols = sm.OLS(y, np.vander(x, order + 1, increasing=True)).fit()
        np.testing.assert_allclose(res.errors, ols.bse, rtol=1e-5)
        assert res.r2 == pytest.approx(ols.rsquared, rel=1e-9)
    np.testing.assert_allclose(res.fitted, np.polyval(np.polyfit(x, y, order), x), rtol=1e-7)


# ---------------------------------------------------------------------- nonlinear
def test_nonlinear_matches_curve_fit():
    x = np.linspace(0, 8, 50)
    y = 0.4 + 3.0 * np.exp(-x / 1.7) + RNG.normal(0, 0.05, len(x))
    res = fit_curve(x, y, "ExpDec1")
    f = fl.LIBRARY["ExpDec1"].func
    p, cov = optimize.curve_fit(f, x, y, p0=[0.5, 2, 1])
    np.testing.assert_allclose(res.values, p, rtol=1e-6)
    np.testing.assert_allclose(res.errors, np.sqrt(np.diag(cov)), rtol=1e-4)
    assert res.r2 == pytest.approx(1 - np.sum((y - f(x, *p)) ** 2) / np.sum((y - y.mean()) ** 2))
    ss_reg, ss_res, ss_unc, ss_cor = [row[2] for row in res.anova()]
    assert ss_reg + ss_res == pytest.approx(ss_unc)
    assert ss_cor == pytest.approx(np.sum((y - y.mean()) ** 2))


CASES = {
    "ExpDec2": [0.5, 4.0, 0.6, 2.0, 5.0], "ExpGrow1": [1, 0.5, 3], "BoxLucas1": [5, 0.4],
    "Boltzmann": [1, 10, 5, 0.7], "Logistic": [1, 10, 3, 2.5], "Logistic5": [1, 10, 3, 2, 1.5],
    "Hill": [0.5, 9, 4, 2], "Gompertz": [10, 4, 1.2], "Richards": [10, 4, 2, 1.0],
    "DoseResp": [1, 10, 5, 0.8], "Weibull": [10, 1, 0.3, 2], "Gauss": [1, 5, 1.5, 10],
    "GaussAmp": [1, 5, 1, 4], "Lorentz": [1, 5, 1.5, 10], "Voigt": [1, 5, 10, 1, 0.8],
    "PsdVoigt1": [1, 5, 10, 1.2, 0.4], "LogNormal": [0.5, 4, 0.4, 10],
    "Allometric1": [2, 1.5], "Allometric2": [1, 2, 0.7], "Power1": [2, -1, 1.5],
    "Rational0": [0.2, 1, 3], "MichaelisMenten": [10, 2], "Langmuir": [20, 0.5],
    "Freundlich": [3, 2.5], "Temkin": [3, 4], "Exp": [1, 2, 0.2], "ExpAssoc": [1, 4, 0.5, 3, 4],
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_recovers_known_parameters(name):
    rng = np.random.default_rng(sum(map(ord, name)))
    x = np.linspace(0.1, 10, 80)
    true = np.array(CASES[name], float)
    f = fl.LIBRARY[name]
    yt = f(x, *true)
    y = yt + rng.normal(0, 0.01 * np.ptp(yt), len(x))
    res = fit_curve(x, y, name)
    assert res.success
    z = np.abs(res.values - true) / res.errors
    assert np.all(z < 4.5), (name, res.values, res.errors)
    assert res.r2 > 0.99


def test_chemistry_models():
    T = np.linspace(280, 360, 12)
    kt = 1e7 * np.exp(-50000 / (fl.R_GAS * T))
    k = kt * (1 + RNG.normal(0, 0.01, len(T)))
    res = fit_curve(T, k, "Arrhenius", weights=1 / (0.01 * kt) ** 2)
    assert abs(res.params["Ea"] - 50000) < 4 * res.errors[1]
    p, _ = optimize.curve_fit(fl.LIBRARY["Arrhenius"].func, T, k, p0=res.values,
                              sigma=0.01 * kt)
    np.testing.assert_allclose(res.values, p, rtol=1e-5)
    p = np.linspace(0.05, 0.35, 15)
    q = fl.LIBRARY["BET"](p, 2.0, 50.0) + RNG.normal(0, 0.01, len(p))
    res = fit_curve(p, q, "BET")
    assert np.all(np.abs(res.values - [2.0, 50.0]) < 4 * res.errors)


def test_fixed_parameter_and_bounds():
    x = np.linspace(0, 8, 40)
    y = 0.4 + 3.0 * np.exp(-x / 1.7) + RNG.normal(0, 0.03, len(x))
    res = fit_curve(x, y, "ExpDec1", fixed={"y0": 0.4})
    p, cov = optimize.curve_fit(lambda t, A, tau: 0.4 + A * np.exp(-t / tau), x, y, p0=[3, 2])
    np.testing.assert_allclose(res.values[1:], p, rtol=1e-6)
    np.testing.assert_allclose(res.errors[1:], np.sqrt(np.diag(cov)), rtol=1e-4)
    assert res.values[0] == 0.4 and res.dof == len(x) - 2
    res = fit_curve(x, y, "ExpDec1", bounds={"t1": (None, 1.2)})
    assert res.values[2] <= 1.2 + 1e-12
    assert res.at_bound[2]


def test_user_defined_function():
    x = np.linspace(0, 8, 40)
    y = 0.4 + 3.0 * np.exp(-x / 1.7) + RNG.normal(0, 0.03, len(x))
    ref = fit_curve(x, y, "ExpDec1")
    res = fit_curve(x, y, "y = A*exp(-x/t) + y0")
    assert res.names == ("A", "t", "y0")
    np.testing.assert_allclose([res.params["y0"], res.params["A"], res.params["t"]],
                               ref.values, rtol=1e-5)
    f = fl.parse_user_function("y = a*ln(x) + b*log(x) + c*x^2 + sqrt(abs(d)) + pi")
    assert f.params == ("a", "b", "c", "d")
    assert f(np.array([10.0]), 1, 1, 1, 4)[0] == pytest.approx(
        math.log(10) + 1 + 100 + 2 + math.pi)


@pytest.mark.parametrize("bad", ["__import__('os').system('x')", "y = a.b*x", "y = (lambda: 1)()",
                                 "y = a*x if a else x", "y = [a]", "y = open(a)", "y = x",
                                 "y = a*x + exp", "y = 'a'*x", "y = a**x)", "y = x[0]*a",
                                 "y = _a*x"])
def test_user_function_rejects_unsafe_or_invalid(bad):
    with pytest.raises(fl.FunctionSyntaxError):
        fl.parse_user_function(bad)


def test_find_x_and_y():
    x = np.linspace(0, 10, 30)
    y = 2 + 0.5 * x + RNG.normal(0, 0.1, len(x))
    res = fit_curve(x, y, "Line")
    a, b = res.values
    assert res.find_x(4.0) == [pytest.approx((4.0 - a) / b, rel=1e-9)]
    g = fit_curve(x, 1 + 5 * np.exp(-(x - 5) ** 2 / 2) + RNG.normal(0, .02, len(x)), "GaussAmp")
    sols = g.find_x(3.0)
    assert len(sols) == 2 and sols[0] < 5 < sols[1]


def test_lack_of_fit_with_replicates():
    x = np.repeat([1.0, 2, 3, 4, 5], 3)
    y = 1 + 0.5 * x + 0.1 * x ** 2 + RNG.normal(0, 0.05, len(x))
    res = fit_curve(x, y, "Line")
    lof = res.lack_of_fit()
    means = pd.Series(y).groupby(x).transform("mean").to_numpy()
    ss_pe = np.sum((y - means) ** 2)
    assert lof["ss_pe"] == pytest.approx(ss_pe)
    assert lof["df_pe"] == 10 and lof["df_lof"] == 3
    assert lof["p"] < 0.01


def test_run_fit_report_and_saved_variables():
    x = np.linspace(0, 8, 30)
    y = 0.4 + 3.0 * np.exp(-x / 1.7) + RNG.normal(0, 0.03, len(x))
    y[3] = np.nan
    ds = Dataset(pd.DataFrame({"t": x, "s": y, "e": np.full(len(x), 0.03)}))
    out = run_fit(ds, "t", "s", "ExpDec1", ename="e", weighting="instrumental",
                  opts=FitOptions(save_fitted=True, save_residuals=True, pred_band=True,
                                  find_x=[1.0], find_y=[2.0]))
    titles = [getattr(i, "title", "") for i in out.section.items]
    for t in ("Parameters", "Statistics", "ANOVA", "Find X from Y", "Find Y from X",
              "Fitted Curves Plot"):
        assert t in titles
    assert len(out.saved["FIT"][1]) == len(x) and np.isnan(out.saved["RES"][1][3])


# ---------------------------------------------------------------------- calibration
MM_X = np.array([0, 2, 4, 6, 8, 10, 12], float)
MM_Y = np.array([2.1, 5.0, 9.0, 12.6, 17.3, 21.0, 24.7])


def test_calibration_miller_example():
    c = cal.calibrate(MM_X, MM_Y)
    assert c.slope == pytest.approx(1.93, abs=0.005)
    assert c.intercept == pytest.approx(1.52, abs=0.005)
    assert c.r == pytest.approx(0.9989, abs=1e-4)
    assert c.s_yx == pytest.approx(0.4329, abs=1e-4)
    assert c.se_slope == pytest.approx(0.0409, abs=1e-4)
    assert c.se_intercept == pytest.approx(0.2950, abs=1e-4)
    for y0, x0, s in ((2.9, 0.72, 0.26), (13.5, 6.21, 0.24), (23.0, 11.13, 0.26)):
        got = c.inverse(y0, 1)
        assert got[0] == pytest.approx(x0, abs=0.006)
        assert got[1] == pytest.approx(s, abs=0.006)
    # LOD / LOQ by hand: σ = s(y/x) = 0.432848, s(a) = 0.294936, b = 1.930357
    assert c.lod == pytest.approx(3.3 * 0.432848 / 1.930357, rel=1e-5)
    assert c.loq == pytest.approx(10 * 0.432848 / 1.930357, rel=1e-5)
    ci = cal.calibrate(MM_X, MM_Y, sigma="intercept")
    assert ci.lod == pytest.approx(3.3 * 0.294936 / 1.930357, rel=1e-5)
    cb = cal.calibrate(MM_X, MM_Y, sigma="blank", blank_sd=0.1)
    assert cb.loq == pytest.approx(1.0 / 1.930357, rel=1e-5)


def test_calibration_replicates_and_weighting():
    c = cal.calibrate(MM_X, MM_Y)
    x0, s3, lo, hi = c.inverse(13.5, 3)
    assert s3 < c.inverse(13.5, 1)[1]
    q = stats.t.ppf(0.975, 5)
    assert hi - x0 == pytest.approx(q * s3)
    x = np.array([1, 2, 5, 10, 20, 50], float)
    y = 0.02 + 0.1 * x * (1 + RNG.normal(0, 0.02, len(x)))
    cw = cal.calibrate(x, y, weighting="1/x2")
    wls = sm.WLS(y, sm.add_constant(x), weights=1 / x ** 2).fit()
    np.testing.assert_allclose([cw.intercept, cw.slope], wls.params, rtol=1e-9)
    np.testing.assert_allclose([cw.se_intercept, cw.se_slope], wls.bse, rtol=1e-8)


def test_calibration_report():
    ds = Dataset(pd.DataFrame({"conc": list(MM_X) + [np.nan] * 2,
                               "signal": list(MM_Y) + [np.nan] * 2,
                               "unk": [2.9, 13.5, 23.0] + [np.nan] * 6}))
    out = cal.calibration_curve(ds, "conc", "signal", unknown="unk", unknown_values=[10.0],
                                save=True)
    titles = [getattr(i, "title", "") for i in out.section.items]
    assert "Limits of Detection and Quantitation" in titles
    assert "Inverse Prediction of Unknown Concentrations" in titles
    conc = out.saved["CONC"][1]
    assert conc[1] == pytest.approx((13.5 - 1.517857) / 1.930357, rel=1e-5)
    assert np.isnan(conc[5])

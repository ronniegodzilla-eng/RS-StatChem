"""WP2: Weight Estimation, 2SLS, Quantile, Ridge/Lasso/Elastic Net and Nonlinear regression
against statsmodels, scikit-learn (test-only) and scipy."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import optimize

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import regression_ext as rx


def num(text):
    return float(text.rstrip("ᵃᵇᶜ"))


def table(sec, title, k=0):
    return [t for t in sec.items if getattr(t, "title", "").startswith(title)][k]


@pytest.fixture
def data():
    rng = np.random.default_rng(31)
    n = 150
    x = rng.uniform(1, 10, n)
    x2 = rng.normal(size=n)
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    endo = 0.8 * z + 0.6 * u + rng.normal(size=n)
    y = 1 + 2 * x + 0.5 * x2 + rng.normal(0, 0.4 * x)
    c = 5 * np.exp(-0.3 * x) + rng.normal(0, 0.05, n)
    return Dataset(pd.DataFrame({"y": y, "x": x, "x2": x2, "z": z, "endo": endo,
                                 "yy": 1 + 1.5 * endo + 0.3 * x2 + u, "c": c,
                                 "g": rng.integers(1, 3, n) * 1.0}))


def test_weight_estimation_power_search(data):
    res = rx.weight_estimation(data, "y", ["x", "x2"], "x", save_weight=True)
    ll = table(res.section, "Log-Likelihood Values")
    best = next(r for r in ll.rows if r[2].endswith("ᵃ"))
    pw = num(best[1])
    df = data.df
    w = 1 / df.x ** pw
    ref = sm.WLS(df.y, sm.add_constant(df[["x", "x2"]]), weights=w).fit()
    coef = table(res.section, "Coefficients")
    assert math.isclose(num(coef.rows[1][1]), ref.params["x"], abs_tol=5e-4)
    assert math.isclose(num(coef.rows[1][2]), ref.bse["x"], abs_tol=5e-4)
    assert math.isclose(num(table(res.section, "Model Summary").rows[1][1]), ref.rsquared,
                        abs_tol=5e-4)
    lls = [num(r[2]) for r in ll.rows]
    assert max(lls) == num(best[2])
    assert pw > 0          # variance grows with x
    assert np.allclose(res.saved["WGT"][1], w)


def test_two_stage_least_squares(data):
    from statsmodels.sandbox.regression.gmm import IV2SLS

    res = rx.two_stage(data, "yy", ["endo", "x2"], ["z", "x2"], save=("pred", "resid"))
    df = data.df
    ref = IV2SLS(df.yy.to_numpy(), sm.add_constant(df[["endo", "x2"]].to_numpy()),
                 sm.add_constant(df[["z", "x2"]].to_numpy())).fit()
    coef = table(res.section, "Coefficients")
    assert math.isclose(num(coef.rows[1][2]), ref.params[1], abs_tol=5e-4)
    assert math.isclose(num(coef.rows[1][3]), ref.bse[1], abs_tol=5e-4)
    ms = table(res.section, "Model Summary")
    assert math.isclose(num(ms.rows[2][1]), ref.rsquared, abs_tol=5e-4)
    assert np.allclose(res.saved["ERR"][1], df.yy - res.saved["FIT"][1])


def test_quantile_matches_quantreg(data):
    from statsmodels.regression.quantile_regression import QuantReg

    res = rx.quantile_regression(data, "y", [], ["x", "x2"], quantiles=(0.25, 0.5))
    df = data.df
    X = sm.add_constant(df[["x", "x2"]].to_numpy())
    ref = QuantReg(df.y.to_numpy(), X).fit(q=0.25, vcov="robust", kernel="epa",
                                            bandwidth="bofinger")
    pe = table(res.section, "Parameter Estimates — Quantile = 0.25")
    assert math.isclose(num(pe.rows[1][1]), ref.params[1], abs_tol=5e-4)
    assert math.isclose(num(pe.rows[1][2]), ref.bse[1], abs_tol=5e-4)
    mq = table(res.section, "Model Quality — Quantile = 0.25")
    assert math.isclose(num(mq.rows[0][1]), ref.prsquared, abs_tol=5e-4)
    assert any(getattr(i, "kind", "") == "image" for i in res.section.items)


@pytest.mark.parametrize("kind,alpha,l1", [("ridge", 3.0, 0.0), ("lasso", 0.05, 1.0),
                                           ("enet", 0.05, 0.4)])
def test_regularized_matches_sklearn(data, kind, alpha, l1):
    from sklearn.linear_model import ElasticNet, Lasso, Ridge

    res = rx.regularized(data, "y", ["x", "x2", "z"], kind, alpha=alpha, l1_ratio=l1,
                         standardize=False, trace=False)
    df = data.df
    X, y = df[["x", "x2", "z"]].to_numpy(), df.y.to_numpy()
    model = {"ridge": Ridge(alpha=alpha), "lasso": Lasso(alpha=alpha, tol=1e-12,
                                                         max_iter=100000),
             "enet": ElasticNet(alpha=alpha, l1_ratio=l1, tol=1e-12, max_iter=100000)}[kind]
    model.fit(X, y)
    coef = table(res.section, "Coefficients")
    got = [num(r[1]) for r in coef.rows[1:]]
    assert np.allclose(got, model.coef_, atol=5e-4)
    assert math.isclose(num(coef.rows[0][1]), model.intercept_, abs_tol=5e-4)


def test_regularized_cv_picks_reasonable_alpha(data):
    res = rx.regularized(data, "y", ["x", "x2", "z"], "lasso", mode="cv",
                         alphas=[0.001, 0.01, 0.1, 1, 10, 100], folds=5, holdout=0.2)
    cv = table(res.section, "Cross-Validation Results")
    best = next(r for r in cv.rows if r[-1].endswith("ᵃ"))
    assert num(best[0]) < 10
    summary = table(res.section, "Model Summary")
    assert any(r[0] == "R Square (holdout)" for r in summary.rows)


def test_nonlinear_matches_curve_fit(data):
    res = rx.nonlinear(data, "c", "a * EXP(-b * x)", {"a": 1.0, "b": 0.1},
                       save=("pred", "resid", "deriv"))
    df = data.df
    popt, pcov = optimize.curve_fit(lambda x, a, b: a * np.exp(-b * x), df.x, df.c,
                                    p0=[1, 0.1])
    pe = table(res.section, "Parameter Estimates")
    assert math.isclose(num(pe.rows[0][1]), popt[0], abs_tol=5e-4)
    assert math.isclose(num(pe.rows[1][2]), math.sqrt(pcov[1, 1]), abs_tol=5e-4)
    anova = table(res.section, "ANOVA")
    sse = float(np.sum((df.c - popt[0] * np.exp(-popt[1] * df.x)) ** 2))
    assert math.isclose(num(anova.rows[1][1]), sse, abs_tol=5e-4)
    assert {"PRED", "RESID", "D_A", "D_B"} <= set(res.saved)
    con = rx.nonlinear(data, "c", "a * exp(-b * x)", {"a": 1.0, "b": 0.1},
                       constraints=["b >= 0.32", "a + b <= 10"])
    pe = table(con.section, "Parameter Estimates")
    assert math.isclose(num(pe.rows[1][1]), 0.32, abs_tol=2e-3)
    boot = rx.nonlinear(data, "c", "a * exp(-b * x)", {"a": 1.0, "b": 0.1}, bootstrap=40)
    assert "Bootstrap" in table(boot.section, "Parameter Estimates").footnotes[0]


def test_nonlinear_rejects_unknown_names(data):
    res = rx.nonlinear(data, "c", "a * exp(-k * x)", {"a": 1.0})
    assert any(getattr(i, "style", "") == "warning" for i in res.section.items)
    with pytest.raises(ValueError):
        rx.compile_model("__import__('os')")[0]({})

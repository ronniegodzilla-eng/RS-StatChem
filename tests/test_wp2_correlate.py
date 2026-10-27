"""WP2: correlation confidence intervals, cross-products, partial extras, Curve Estimation Save
and Means layers."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import correlate, curvefit_spss, means


def num(text):
    return float(text.rstrip("*ᵃᵇ"))


def table(sec, title):
    return next(t for t in sec.items if getattr(t, "title", "").startswith(title))


@pytest.fixture
def ds():
    rng = np.random.default_rng(4)
    n = 50
    a = rng.normal(size=n)
    b = 0.6 * a + rng.normal(size=n)
    c = 0.3 * b + rng.normal(size=n)
    df = pd.DataFrame({"a": a, "b": b, "c": c})
    df.loc[[3, 9], "c"] = np.nan
    return Dataset(df)


def test_pearson_ci_fisher_z_and_bias(ds):
    sec = correlate.bivariate(ds, ["a", "b"], ci=True, ci_level=0.95)
    t = table(sec, "Confidence Intervals")
    r, _ = stats.pearsonr(ds.df.a, ds.df.b)
    n = 50
    z = math.atanh(r)
    lo, hi = math.tanh(z - 1.959964 / math.sqrt(n - 3)), math.tanh(z + 1.959964 / math.sqrt(n - 3))
    assert math.isclose(num(t.rows[0][3]), lo, abs_tol=5e-4)
    assert math.isclose(num(t.rows[0][4]), hi, abs_tol=5e-4)
    # same as scipy's Fisher interval
    ci = stats.pearsonr(ds.df.a, ds.df.b).confidence_interval(0.95)
    assert math.isclose(num(t.rows[0][3]), ci.low, abs_tol=5e-4)
    lo_b, _ = correlate.correlation_ci(r, n, "pearson", 0.95, bias=True)
    zb = z - r / (2 * (n - 1))
    assert math.isclose(lo_b, math.tanh(zb - 1.959964 / math.sqrt(n - 3)), abs_tol=1e-6)


def test_rank_ci_standard_errors():
    lo, hi = correlate.correlation_ci(0.5, 30, "spearman", 0.95, rank_se="fieller")
    se = math.sqrt(1.06 / 27)
    assert math.isclose(lo, math.tanh(math.atanh(0.5) - 1.959964 * se), abs_tol=1e-6)
    lo, _ = correlate.correlation_ci(0.5, 30, "spearman", 0.95, rank_se="bonett")
    se = math.sqrt((1 + 0.125) / 27)
    assert math.isclose(lo, math.tanh(math.atanh(0.5) - 1.959964 * se), abs_tol=1e-6)
    lo, _ = correlate.correlation_ci(0.4, 30, "kendall", 0.95)
    se = math.sqrt(0.437 / 26)
    assert math.isclose(lo, math.tanh(math.atanh(0.4) - 1.959964 * se), abs_tol=1e-6)


def test_cross_products_and_covariance(ds):
    sec = correlate.bivariate(ds, ["a", "b"], cross_products=True)
    t = table(sec, "Correlations")
    sp = next(r for r in t.rows if r[1] == "Sum of Squares and Cross-products")
    cv = next(r for r in t.rows if r[1] == "Covariance")
    assert math.isclose(num(sp[3]), float(np.sum((ds.df.a - ds.df.a.mean())
                                                  * (ds.df.b - ds.df.b.mean()))), rel_tol=1e-3)
    assert math.isclose(num(cv[3]), float(np.cov(ds.df.a, ds.df.b)[0, 1]), abs_tol=5e-4)


def test_partial_pairwise_and_descriptives(ds):
    sec = correlate.partial(ds, ["a", "c"], ["b"], descriptives=True, listwise_deletion=False)
    assert table(sec, "Descriptive Statistics")
    R = ds.df.corr().to_numpy()
    P = np.linalg.inv(R[np.ix_([0, 2, 1], [0, 2, 1])])
    expected = -P[0, 1] / math.sqrt(P[0, 0] * P[1, 1])
    t = table(sec, "Correlations")
    assert math.isclose(num(t.rows[0][4]), expected, abs_tol=5e-4)
    assert t.rows[2][4] == str(48 - 3)
    import pingouin as pg

    lw = correlate.partial(ds, ["a", "c"], ["b"])
    ref = pg.partial_corr(data=ds.df.dropna(), x="a", y="c", covar="b")
    assert math.isclose(num(table(lw, "Correlations").rows[0][4]), ref["r"].iloc[0],
                        abs_tol=5e-4)


def test_curve_estimation_save_intervals():
    rng = np.random.default_rng(2)
    t = np.arange(1, 41, dtype=float)
    y = 3 * np.exp(0.05 * t) * np.exp(rng.normal(0, 0.05, 40))
    ds = Dataset(pd.DataFrame({"t": t, "y": y}))
    res = curvefit_spss.curve_estimation(ds, ["y"], "t", ("linear",), save=("pred", "resid",
                                                                            "cin"))
    f = sm.OLS(y, sm.add_constant(t)).fit()
    pi = f.get_prediction(sm.add_constant(t)).summary_frame(alpha=0.05)
    assert np.allclose(res.saved["FIT"][1], f.fittedvalues)
    assert np.allclose(res.saved["ERR"][1], f.resid)
    assert np.allclose(res.saved["LCL"][1], pi.obs_ci_lower)
    res = curvefit_spss.curve_estimation(ds, ["y"], "t", ("linear", "exponential"),
                                         save=("cin",))
    g = sm.OLS(np.log(y), sm.add_constant(t)).fit()
    gp = g.get_prediction(sm.add_constant(t)).summary_frame(alpha=0.05)
    assert set(res.saved) == {"LCL1", "UCL1", "LCL2", "UCL2"}
    assert np.allclose(res.saved["UCL2"][1], np.exp(gp.obs_ci_upper))


def test_means_three_layers():
    rng = np.random.default_rng(1)
    n = 120
    ds = Dataset(pd.DataFrame({"y": rng.normal(size=n), "a": rng.integers(1, 3, n) * 1.0,
                               "b": rng.integers(1, 3, n) * 1.0,
                               "c": rng.integers(1, 3, n) * 1.0}))
    sec = means.means(ds, ["y"], [["a"], ["b"], ["c"]])
    rep = table(sec, "Report")
    assert len(rep.columns) == 3 + 3
    sub = ds.df[(ds.df.a == 1) & (ds.df.b == 2) & (ds.df.c == 1)]
    row = next(r for r in rep.rows if r[2] == "1" and r[3] and abs(num(r[3]) - sub.y.mean())
               < 1e-3)
    assert row

"""Bayesian Statistics (WP5) against pingouin's Bayes factors (test-only reference), closed forms
and Monte Carlo integration."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pingouin as pg
import pytest
from scipy import integrate, stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import bayes as bs


@pytest.mark.parametrize("t,n1,n2,r", [(2.0, 30, None, 0.707), (3.1, 12, 15, 0.707),
                                       (-1.2, 50, 40, 1.0), (0.5, 8, None, 0.5)])
def test_jzs_ttest_matches_pingouin(t, n1, n2, r):
    assert bs.jzs_ttest_bf10(t, n1, n2, r) == pytest.approx(
        pg.bayesfactor_ttest(t, n1, n2, r=r), rel=1e-8)


@pytest.mark.parametrize("r,n", [(0.3, 30), (0.6, 12), (-0.1, 100)])
def test_pearson_bf_matches_pingouin(r, n):
    assert bs.pearson_bf10(r, n) == pytest.approx(pg.bayesfactor_pearson(r, n), rel=1e-8)


def test_binomial_and_poisson_bf():
    assert bs.binom_bf10(115, 200, 0.5) == pytest.approx(pg.bayesfactor_binom(115, 200, 0.5),
                                                         rel=1e-8)
    # Poisson: marginal likelihood by numerical integration over lambda
    x = np.array([2, 4, 3, 5, 1, 4])
    m1 = integrate.quad(lambda lam: np.prod(stats.poisson.pmf(x, lam)) *
                        stats.gamma.pdf(lam, 2, scale=1.0), 0, 50)[0]
    m0 = np.prod(stats.poisson.pmf(x, 2.0))
    assert bs.poisson_bf10(x.sum(), len(x), 2.0, 2.0, 1.0) == pytest.approx(m1 / m0, rel=1e-6)


def test_gonen_bf_by_integration():
    nd, df = 10, 38
    num = integrate.quad(lambda d: stats.nct.pdf(2.0, df, d * math.sqrt(nd)) *
                         stats.norm.pdf(d, 0.2, 0.8), -10, 10)[0]
    assert bs.gonen_bf10(2.0, 20, 20, 0.2, 0.64) == pytest.approx(num / stats.t.pdf(2.0, df),
                                                                  rel=1e-6)


def test_anova_rouder_two_groups_equals_jzs_t():
    # Rouder et al. (2012): a two-level fixed factor with scale r equals the JZS t test with
    # scale r*sqrt(2) (BayesFactor anovaBF "medium" .5 <-> ttestBF .707)
    rng = np.random.default_rng(5)
    y = np.r_[rng.normal(0, 1, 15), rng.normal(0.8, 1, 18)]
    g = np.r_[["a"] * 15, ["b"] * 18]
    t = stats.ttest_ind(y[g == "b"], y[g == "a"]).statistic
    assert bs.rouder_anova_bf10(y, g, 0.5) == pytest.approx(
        pg.bayesfactor_ttest(t, 15, 18, r=0.5 * math.sqrt(2)), rel=1e-5)


def test_regression_bf_priors():
    r2, n, p = 0.2, 40, 2
    rng = np.random.default_rng(11)
    g = 1 / rng.gamma(0.5, 1 / (n / 2), 2_000_000)      # Zellner-Siow: g ~ IG(1/2, n/2)
    mc = np.mean((1 + g) ** ((n - 1 - p) / 2) * (1 + g * (1 - r2)) ** (-(n - 1) / 2))
    assert bs.regression_bf10(r2, n, p, "jzs", r=1.0) == pytest.approx(mc, rel=0.01)
    gg = 40.0
    closed = (1 + gg) ** ((n - 1 - p) / 2) * (1 + gg * (1 - r2)) ** (-(n - 1) / 2)
    assert bs.regression_bf10(r2, n, p, "zellner") == pytest.approx(closed)
    hyper = integrate.quad(lambda x: (1 + x) ** ((n - 1 - p) / 2) *
                           (1 + x * (1 - r2)) ** (-(n - 1) / 2) * 0.5 * (1 + x) ** -1.5,
                           0, np.inf, limit=400)[0]
    assert bs.regression_bf10(r2, n, p, "hyperg", a=3) == pytest.approx(hyper, rel=1e-5)


def test_gunel_dickey_by_monte_carlo():
    tab = np.array([[10, 5], [3, 12]])
    rng = np.random.default_rng(2)
    cells = rng.dirichlet(np.ones(4), 400_000)
    l1 = np.mean(np.prod(cells ** tab.ravel(), axis=1))
    rows = rng.dirichlet(np.ones(2), 400_000)
    cols = rng.dirichlet(np.ones(2), 400_000)
    probs = (rows[:, :, None] * cols[:, None, :]).reshape(-1, 4)
    l0 = np.mean(np.prod(probs ** tab.ravel(), axis=1))
    assert bs.gunel_dickey_bf10(tab, "joint") == pytest.approx(l1 / l0, rel=0.03)


def test_posteriors():
    x = np.array([4.1, 5.3, 6.2, 4.8, 5.9, 5.1, 4.4])
    p = bs.normal_mean_posterior(x)
    se = x.std(ddof=1) / math.sqrt(len(x))
    lo, hi = stats.t.interval(0.95, len(x) - 1, x.mean(), se)
    assert (p.mean, p.lo, p.hi) == pytest.approx((x.mean(), lo, hi))
    known = bs.normal_mean_posterior(x, known_var=1.0)
    assert known.var == pytest.approx(1 / len(x))
    conj = bs.normal_mean_posterior(x, prior="conjugate", mu0=0, kappa0=1, a0=1, b0=1)
    assert conj.mean == pytest.approx(len(x) * x.mean() / (len(x) + 1))
    c = bs.correlation_posterior(0.5, 30)
    assert 0.1 < c.lo < c.mode < c.hi < 0.8
    assert c.mode == pytest.approx(0.5, abs=0.02)


@pytest.fixture()
def ds():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(0.4, 1, 40), "y": rng.normal(size=40),
                       "g": np.repeat([1.0, 2.0], 20), "k": np.tile([1.0, 2.0, 3.0, 4.0], 10),
                       "c": rng.poisson(3, 40).astype(float)})
    df["y"] += df["x"]
    df["z"] = df["y"] + rng.normal(0.3, 1, 40)
    return Dataset(df)


def test_procedures_output(ds):
    o = bs.BayesOptions()
    sec = bs.one_sample_normal(ds, ["x"], 0.0, o)
    bf_table = sec.items[1]
    t = stats.ttest_1samp(ds.df["x"], 0).statistic
    assert float(bf_table.rows[0][5]) == pytest.approx(1 / bs.jzs_ttest_bf10(t, 40), abs=1e-3)
    sec = bs.independent_normal(ds, ["x"], "g", 1.0, 2.0, bs.BayesOptions(bf10=True))
    assert "BF10" in sec.items[2].footnotes[0]
    sec = bs.linear_regression(ds, "y", ["g"], ["x"], o)
    assert [i.title for i in sec.items if i.kind == "table"][:2] == ["ANOVAᵃ", "Model Summary"]
    coef = next(i for i in sec.items if getattr(i, "title", "").startswith("Bayesian Estimates "
                                                                         "of Coef"))
    import statsmodels.api as sm
    X = sm.add_constant(np.column_stack([(ds.df.g == 1).astype(float), ds.df.x]))
    ols = sm.OLS(ds.df.y, X).fit()
    assert float(coef.rows[2][2]) == pytest.approx(ols.params.iloc[2], abs=1e-3)
    assert float(coef.rows[2][3]) == pytest.approx(ols.bse.iloc[2] ** 2 * 37 / 35, abs=1e-3)
    bs.oneway_anova(ds, ["y"], "k", o, "rouder")
    bs.oneway_anova(ds, ["y"], "k", o, "zellner")
    bs.repeated_measures_anova(ds, ["x", "y", "z"], o)
    bs.pearson_correlation(ds, ["x", "y", "z"], o)
    bs.one_sample_binomial(ds, ["g"], 2.0, 0.5, o)
    bs.one_sample_binomial(ds, ["k"], ("le", 2.0), 0.5, o)
    bs.one_sample_poisson(ds, ["c"], 3.0, o)
    bs.loglinear(ds, "g", "k", o, "rows")
    bs.selftest()

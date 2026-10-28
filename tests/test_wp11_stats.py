"""WP11 SPSS 28–32 extensions: distance correlation (IBM chapter / Székely formula), permutation
tests against scipy.stats.permutation_test, VAR against statsmodels, the Weibull plot against
scipy / lifelines, and TURF against brute force."""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import dcor, permutation, turf, var_model, weibull_plot


# ----------------------------------------------------------------------------- dCor
def _dcor_naive(x, y):
    """Direct transcription of Székely, Rizzo & Bakirov (2007), eq. 2.8–2.10."""
    x = np.asarray(x, float).reshape(len(x), -1)
    y = np.asarray(y, float).reshape(len(y), -1)
    n = len(x)
    a = np.array([[np.linalg.norm(x[i] - x[j]) for j in range(n)] for i in range(n)])
    b = np.array([[np.linalg.norm(y[i] - y[j]) for j in range(n)] for i in range(n)])
    A = a - a.mean(0) - a.mean(1)[:, None] + a.mean()
    B = b - b.mean(0) - b.mean(1)[:, None] + b.mean()
    v2xy, v2x, v2y = (A * B).sum() / n ** 2, (A * A).sum() / n ** 2, (B * B).sum() / n ** 2
    return math.sqrt(v2xy) / math.sqrt(math.sqrt(v2x * v2y))


def test_dcor_matches_formula_univariate_and_multivariate():
    r = np.random.default_rng(0)
    x = r.normal(size=40)
    y = x ** 2 + 0.3 * r.normal(size=40)
    assert dcor.distance_correlation(x, y) == pytest.approx(_dcor_naive(x, y), rel=1e-12)
    X = r.normal(size=(30, 3))
    Y = np.column_stack([X[:, 0] * X[:, 1], r.normal(size=30)])
    assert dcor.distance_correlation(X, Y) == pytest.approx(_dcor_naive(X, Y), rel=1e-12)


def test_dcor_properties():
    x = np.linspace(-1, 1, 25)
    assert dcor.distance_correlation(x, 3 * x + 2) == pytest.approx(1.0)
    assert dcor.distance_correlation(x, np.ones(25)) == 0.0     # zero distance variance
    # Pearson r of x and x² on a symmetric grid is 0; distance correlation is not
    assert abs(np.corrcoef(x, x ** 2)[0, 1]) < 1e-12
    assert dcor.distance_correlation(x, x ** 2) > 0.4


def test_dcor_fisher_ci_and_permutation_rules():
    lo, hi = dcor.fisher_ci(0.5, 28, 0.95)
    z = math.atanh(0.5)
    assert lo == pytest.approx(math.tanh(z - 1.959963985 / 5))
    assert hi == pytest.approx(math.tanh(z + 1.959963985 / 5))
    r = np.random.default_rng(1)
    x = r.normal(size=30)
    res = dcor.permutation_test(x, x + r.normal(scale=0.5, size=30), 300, seed=3)
    assert res["p"] < 0.01 and res["B"] == 300
    assert res["perm_ci"][0] < res["perm_ci"][1] < res["dcor"]
    res2 = dcor.permutation_test(x, r.normal(size=30), 300, seed=3)
    assert res2["p"] > 0.05


def test_dcor_procedure_tables():
    r = np.random.default_rng(2)
    ds = Dataset(pd.DataFrame({"a": r.normal(size=40), "b": r.normal(size=40),
                               "c": r.normal(size=40)}))
    sec = dcor.dcor_procedure(ds, ["a", "b", "c"], n_perm=50, seed=1)
    assert [getattr(i, "title", "") for i in sec.items][-2:] == ["Distance Correlations",
                                                                 "Distance Correlation Matrix"]
    sec = dcor.dcor_procedure(ds, x_set=["a", "b"], y_set=["c"], n_perm=50, seed=1)
    assert len(sec.result) == 1


# ----------------------------------------------------------------------------- permutation
def test_independent_exact_matches_scipy():
    x = np.array([12.1, 14.3, 11.8, 15.2, 13.9, 12.7])
    y = np.array([10.2, 11.9, 12.4, 10.8, 11.1])
    obs, null, exact = permutation.independent_null(x, y, "mean")
    p = permutation.p_values(obs, null, exact)
    ref = stats.permutation_test((x, y), lambda a, b: np.mean(a) - np.mean(b),
                                 permutation_type="independent", n_resamples=np.inf,
                                 vectorized=False)
    assert exact and len(null) == math.comb(11, 6)
    assert p["two-sided"] == pytest.approx(ref.pvalue, abs=1e-12)
    for alt, key in (("less", "less"), ("greater", "greater")):
        ref = stats.permutation_test((x, y), lambda a, b: np.mean(a) - np.mean(b),
                                     permutation_type="independent", n_resamples=np.inf,
                                     alternative=alt, vectorized=False)
        assert p[key] == pytest.approx(ref.pvalue, abs=1e-12)
    obs_t, null_t, _ = permutation.independent_null(x, y, "welch")
    ref = stats.permutation_test((x, y), lambda a, b: stats.ttest_ind(a, b, equal_var=False)
                                 .statistic, permutation_type="independent",
                                 n_resamples=np.inf, vectorized=False)
    assert permutation.p_values(obs_t, null_t, True)["two-sided"] == pytest.approx(
        ref.pvalue, abs=1e-12)


def test_paired_exact_matches_scipy():
    a = np.array([8.1, 7.4, 9.0, 6.8, 7.9, 8.6, 7.1, 8.3])
    b = np.array([7.6, 7.5, 8.1, 6.2, 7.7, 8.0, 7.3, 7.4])
    obs, null, exact = permutation.paired_null(a - b, "mean")
    ref = stats.permutation_test((a, b), lambda u, v: np.mean(u - v),
                                 permutation_type="samples", n_resamples=np.inf,
                                 vectorized=False)
    assert exact and permutation.p_values(obs, null, exact)["two-sided"] == pytest.approx(
        ref.pvalue, abs=1e-12)


def test_correlation_exact_matches_scipy():
    x = np.array([1.2, 2.3, 2.9, 4.1, 5.0, 6.2, 6.8])
    y = np.array([2.0, 2.1, 3.5, 3.9, 4.0, 6.5, 6.0])
    obs, null, exact = permutation.correlation_null(x, y, "pearson")
    ref = stats.permutation_test((x,), lambda u: stats.pearsonr(u, y)[0],
                                 permutation_type="pairings", n_resamples=np.inf,
                                 vectorized=False)
    assert exact and obs == pytest.approx(stats.pearsonr(x, y)[0])
    assert permutation.p_values(obs, null, exact)["two-sided"] == pytest.approx(ref.pvalue,
                                                                                abs=1e-12)


def test_monte_carlo_p_value_convention_and_regression():
    r = np.random.default_rng(4)
    x, y = r.normal(size=30), r.normal(0.8, 1, 30)
    obs, null, exact = permutation.independent_null(x, y, "mean", n_perm=2000, max_exact=10,
                                                    seed=7)
    p = permutation.p_values(obs, null, exact)
    assert not exact and p["B"] == 2000
    assert p["less"] == pytest.approx(((null <= obs + abs(obs) * permutation.EPS).sum() + 1)
                                      / 2001)
    ref = stats.ttest_ind(x, y).pvalue
    assert abs(p["two-sided"] - ref) < 0.03
    X = r.normal(size=(60, 2))
    yy = 1 + 0.6 * X[:, 0] + r.normal(size=60)
    res = permutation.freedman_lane(yy, X, 1999, seed=1)
    import statsmodels.api as sm

    fit = sm.OLS(yy, sm.add_constant(X)).fit()
    assert res[0]["two-sided"] < 0.01
    for j in range(2):
        assert res[j]["t"] == pytest.approx(fit.tvalues[j + 1])
        assert abs(res[j]["two-sided"] - fit.pvalues[j + 1]) < 0.02


def test_permutation_procedure_runs():
    r = np.random.default_rng(5)
    ds = Dataset(pd.DataFrame({"y": r.normal(size=24), "z": r.normal(size=24),
                               "g": np.repeat([1.0, 2.0, 3.0], 8)}))
    for test, kw in (("independent", dict(variables=["y"], group="g", groups=(1.0, 2.0))),
                     ("paired", dict(variables=[], pairs=[("y", "z")])),
                     ("oneway", dict(variables=["y"], group="g")),
                     ("correlation", dict(variables=["y", "z"]))):
        sec = permutation.permutation_procedure(ds, test, n_perm=200, seed=1, **kw)
        assert sec.items


# ----------------------------------------------------------------------------- VAR
def test_var_matches_statsmodels():
    from statsmodels.tsa.api import VAR

    r = np.random.default_rng(6)
    Y = np.zeros((150, 2))
    for t in range(1, 150):
        Y[t] = [[0.5, 0.1], [0.3, 0.4]] @ Y[t - 1] + r.normal(size=2)
    ds = Dataset(pd.DataFrame(Y, columns=["a", "b"]))
    res = var_model.var_procedure(ds, ["a", "b"], lags=2, forecast=3)
    ref = VAR(Y).fit(2)
    assert np.allclose(np.asarray(res.model.params), ref.params)
    tab = next(i for i in res.section.items if getattr(i, "title", "") == "Equation: a")
    assert float(tab.rows[1][1]) == pytest.approx(ref.params[1, 0], abs=1e-4)
    g = next(i for i in res.section.items if getattr(i, "title", "") ==
             "Granger Causality Tests")
    t = ref.test_causality(0, [1], kind="f")
    assert float(g.rows[0][2]) == pytest.approx(t.test_statistic, abs=1e-3)
    auto = var_model.var_procedure(ds, ["a", "b"], max_lags=6, criterion="bic")
    sel = VAR(Y).select_order(6)
    assert auto.model.k_ar == max(1, sel.selected_orders["bic"])


# ----------------------------------------------------------------------------- Weibull
def test_weibull_mle_matches_scipy_and_lifelines():
    r = np.random.default_rng(7)
    t = stats.weibull_min.rvs(1.8, scale=120, size=80, random_state=r)
    m = weibull_plot.weibull_mle(t, np.ones(80, bool))
    c, loc, scale = stats.weibull_min.fit(t, floc=0)
    assert m["beta"] == pytest.approx(c, rel=1e-4)
    assert m["eta"] == pytest.approx(scale, rel=1e-4)
    WeibullFitter = pytest.importorskip("lifelines").WeibullFitter   # optional test-only dep

    d = r.random(80) < 0.75
    m2 = weibull_plot.weibull_mle(t, d)
    wf = WeibullFitter().fit(t, event_observed=d)
    assert m2["beta"] == pytest.approx(wf.rho_, rel=1e-3)
    assert m2["eta"] == pytest.approx(wf.lambda_, rel=1e-3)


def test_median_ranks_and_rank_regression():
    t = np.array([16.0, 34, 53, 75, 93, 120])
    times, F = weibull_plot.median_ranks(t, np.ones(6, bool))
    assert np.allclose(F, (np.arange(1, 7) - 0.3) / 6.4)
    beta, eta = weibull_plot.rank_regression(times, F, "x")
    b, a = np.polyfit(np.log(-np.log(1 - F)), np.log(times), 1)
    assert beta == pytest.approx(1 / b) and eta == pytest.approx(math.exp(a))
    # Johnson's adjusted ranks: suspension at 2nd position
    times, F = weibull_plot.median_ranks(np.array([10.0, 20, 30, 40]),
                                         np.array([True, False, True, True]))
    ranks = F * 4.4 + 0.3
    assert np.allclose(ranks, [1.0, 1.0 + 4 / 3, 1.0 + 4 / 3 + (5 - 1 - 4 / 3) / 2])


def test_weibull_plot_procedure():
    r = np.random.default_rng(8)
    ds = Dataset(pd.DataFrame({"t": r.weibull(2, 50) * 10, "s": (r.random(50) < .8) * 1.0,
                               "g": np.repeat([1.0, 2.0], 25)}))
    for kw in ({}, {"status": "s", "event_values": [1.0], "method": "mle"},
               {"group": "g", "method": "rry"}):
        sec = weibull_plot.weibull_plot(ds, "t", **kw)
        assert any(getattr(i, "title", "") == "Weibull Plot" for i in sec.items)


# ----------------------------------------------------------------------------- TURF
def test_turf_matches_brute_force():
    r = np.random.default_rng(9)
    P = r.random((200, 7)) < np.linspace(0.1, 0.5, 7)
    res = turf.turf(P, 3, 5)
    for k in (1, 2, 3):
        best = max(itertools.combinations(range(7), k),
                   key=lambda c: (P[:, list(c)].any(1).sum(), P[:, list(c)].sum()))
        reach, freq, combo = res["results"][k][0]
        assert reach == P[:, list(best)].any(1).sum()
        assert freq == P[:, list(combo)].sum()
    heur = turf.turf(P, 3, 5, exhaustive_limit=5, beam=10)
    assert heur["heuristic"][3] and heur["results"][3][0][0] == res["results"][3][0][0]


def test_turf_procedure():
    r = np.random.default_rng(10)
    ds = Dataset(pd.DataFrame({f"v{i}": (r.random(60) < 0.3) * 1.0 for i in range(5)}))
    sec = turf.turf_analysis(ds, list(ds.names()), [1.0], 2, 3)
    assert sec.items[1].title == "Best Reach and Frequency by Group Size"

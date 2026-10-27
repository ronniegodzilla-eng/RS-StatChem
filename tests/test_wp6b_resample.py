"""WP6b: shared bootstrap engine (core/stats/resample.py) and the fast model refits
(core/stats/boot_models.py) against SciPy / statsmodels references."""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Table
from rsstatchem.core.stats import boot_models as bm
from rsstatchem.core.stats import classify, logistic, regression
from rsstatchem.core.stats import resample as rs


def _mean(a):
    return a.mean(axis=1)


@pytest.mark.parametrize("method", ["percentile", "bca"])
@pytest.mark.parametrize("seed", [1, 123, 2000000])
def test_mean_ci_matches_scipy(method, seed):
    x = np.random.default_rng(seed).gamma(2.0, 2.0, 57)
    mine = rs.bootstrap(x, _mean, rs.BootSpec(2000, seed, 0.95, method), vectorized=True)
    ref = stats.bootstrap((x,), np.mean, n_resamples=2000, method=method,
                          rng=np.random.default_rng(seed))
    assert mine.lower[0] == pytest.approx(ref.confidence_interval.low, abs=1e-12)
    assert mine.upper[0] == pytest.approx(ref.confidence_interval.high, abs=1e-12)
    assert mine.se[0] == pytest.approx(ref.standard_error, abs=1e-12)
    np.testing.assert_allclose(mine.boot[:, 0], ref.bootstrap_distribution)


@pytest.mark.parametrize("method", ["percentile", "bca"])
def test_paired_statistic_matches_scipy(method):
    rng = np.random.default_rng(4)
    x = rng.normal(size=40)
    y = 0.6 * x + rng.normal(size=40)

    def corr(a, b, axis=-1):
        return np.array([np.corrcoef(a[i], b[i])[0, 1] for i in range(a.shape[0])]) \
            if a.ndim > 1 else np.corrcoef(a, b)[0, 1]

    ref = stats.bootstrap((x, y), corr, paired=True, vectorized=False, n_resamples=999,
                          method=method, rng=np.random.default_rng(77))
    mine = rs.bootstrap(np.column_stack([x, y]), lambda a: np.corrcoef(a.T)[0, 1],
                        rs.BootSpec(999, 77, ci=method))
    assert mine.lower[0] == pytest.approx(ref.confidence_interval.low, abs=1e-10)
    assert mine.upper[0] == pytest.approx(ref.confidence_interval.high, abs=1e-10)


def test_level_and_chunking_do_not_change_results():
    x = np.random.default_rng(8).normal(size=300)
    a = rs.bootstrap_indices(300, lambda I: x[I].mean(axis=1)[:, None], x.mean(),
                             rs.BootSpec(1000, 5, 0.90), batch=7)
    b = rs.bootstrap_indices(300, lambda I: x[I].mean(axis=1)[:, None], x.mean(),
                             rs.BootSpec(1000, 5, 0.90), batch=1000)
    np.testing.assert_array_equal(a.boot, b.boot)
    ref = stats.bootstrap((x,), np.mean, n_resamples=1000, confidence_level=0.90,
                          method="percentile", rng=np.random.default_rng(5))
    assert a.lower[0] == pytest.approx(ref.confidence_interval.low)


def test_stratified_sampling_keeps_strata_sizes():
    strata = np.repeat(["a", "b", "c"], [5, 10, 25])
    rng = np.random.default_rng(1)
    codes = rs.strata_codes(strata)
    for idx in rs.index_batches(40, 50, rng, codes):
        for c, size in zip(range(3), (5, 10, 25)):
            assert (codes[idx] == c).sum(axis=1).tolist() == [size] * len(idx)


def test_bias_se_and_significance_definitions():
    x = np.random.default_rng(2).normal(0.3, 1, 80)
    res = rs.bootstrap(x, _mean, rs.BootSpec(1500, 3), vectorized=True)
    boot = res.boot[:, 0]
    assert res.bias[0] == pytest.approx(boot.mean() - x.mean())
    assert res.se[0] == pytest.approx(boot.std(ddof=1))
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    assert res.p[0] == pytest.approx(min(p, 1.0))
    # p < .05 exactly when the 95% percentile interval excludes 0
    assert (res.p[0] < 0.05) == (res.lower[0] > 0 or res.upper[0] < 0)


def test_spec_from_dialog_values_and_tables():
    assert rs.BootSpec.from_options({"perform": False}) is None
    spec = rs.BootSpec.from_options({"perform": True, "samples": 500, "use_seed": True,
                                     "seed": 9, "level": 90, "ci": "bca",
                                     "sampling": "stratified", "strata": ["g"]})
    assert (spec.samples, spec.seed, spec.level, spec.ci, spec.strata) == \
        (500, 9, 0.9, "bca", ("g",))
    t = rs.specifications_table(spec)
    assert t.title == "Bootstrap Specifications"
    assert ["Confidence Interval Type", "Bias-corrected and accelerated (BCa)"] in t.rows
    res = rs.bootstrap(np.arange(10.0), _mean, rs.BootSpec(100, 1), vectorized=True)
    tab = rs.insert_boot_rows(Table("T", ["", "Mean"], [["x", "4.5"], ["y", "1"]]),
                              {0: {1: (res, 0)}}, rs.BootSpec(100, 1))
    assert [r[2] for r in tab.rows[:5]] == ["", "Bias", "Std. Error", "95% CI Lower",
                                             "95% CI Upper"]
    assert tab.footnotes[-1].startswith("a. Unless otherwise noted, bootstrap results are "
                                        "based on 100 bootstrap samples")
    app = rs.append_boot_columns(Table("S", ["", "Value"], [["Phi", ".2"]]), {0: (res, 0)},
                                 rs.BootSpec(100, 1))
    assert app.columns[-1] == ("Bootstrapᵃ", "95% CI Upper") and len(app.rows[0]) == 7


# ============================================================================ model refits
def _sample_idx(n, seed=3, b=5):
    return np.random.default_rng(seed).integers(0, n, (b, n))


def test_ols_refit_matches_lstsq_and_wls():
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(60), rng.normal(size=(60, 2))])
    y = X @ [1, 2, -1] + rng.normal(size=60)
    w = rng.uniform(0.5, 2, 60)
    m = bm.ols_model(y, X, ["c", "a", "b"], w)
    idx = _sample_idx(60)
    got = m.refit_batch(idx)
    for i, r in enumerate(idx):
        sw = np.sqrt(w[r])
        ref = np.linalg.lstsq(X[r] * sw[:, None], y[r] * sw, rcond=None)[0]
        np.testing.assert_allclose(got[i], ref, rtol=1e-9)


def test_logit_refit_matches_statsmodels():
    import statsmodels.api as sm

    rng = np.random.default_rng(2)
    X = np.column_stack([rng.normal(size=(200, 2)), np.ones(200)])
    y = (rng.random(200) < 1 / (1 + np.exp(-(X @ [1, -0.5, 0.2])))).astype(float)
    est = sm.Logit(y, X).fit(disp=0).params
    m = bm.logit_model(y, X, ["a", "b", "c"], est)
    idx = _sample_idx(200)
    got = m.refit_batch(idx)
    for i, r in enumerate(idx):
        ref = sm.Logit(y[r], X[r]).fit(disp=0).params
        np.testing.assert_allclose(got[i], ref, atol=1e-6)


def test_multinomial_refit_matches_statsmodels():
    import statsmodels.api as sm

    rng = np.random.default_rng(3)
    X = np.column_stack([np.ones(250), rng.normal(size=(250, 2))])
    eta = np.column_stack([np.zeros(250), X @ [0.2, 1, 0], X @ [-0.3, 0, 1]])
    p = np.exp(eta) / np.exp(eta).sum(1, keepdims=True)
    y = np.array([rng.choice(3, p=pi) for pi in p])
    start = sm.MNLogit(y, X).fit(disp=0).params
    for r in _sample_idx(250, b=3):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = sm.MNLogit(y[r], X[r]).fit(disp=0, method="newton", maxiter=100).params
        np.testing.assert_allclose(bm.mnlogit_fit(y[r], X[r], 3, start), ref, atol=1e-5)


@pytest.mark.parametrize("link", ["logit", "probit", "cloglog", "nloglog", "cauchit"])
def test_ordinal_newton_matches_core_fit(link):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(300, 2))
    y = np.digitize(X @ [1.0, -0.5] + rng.logistic(size=300), [-1, 0.5, 1.5])
    th, beta, *_ = logistic._ordinal_fit(y, X, 4, link)
    par = bm.ordinal_fit(y, X, 4, link, th + 0.1, beta * 0.9)
    np.testing.assert_allclose(par, np.r_[th, beta], atol=2e-4)


def test_glm_irls_matches_statsmodels():
    import statsmodels.api as sm

    rng = np.random.default_rng(5)
    X = np.column_stack([np.ones(150), rng.normal(size=150)])
    y = rng.poisson(np.exp(X @ [0.3, 0.4])).astype(float)
    fam = sm.families.Poisson()
    est = sm.GLM(y, X, family=fam).fit().params
    m = bm.glm_model(y, X, fam, None, None, ["c", "x"], est)
    for r in _sample_idx(150, b=3):
        np.testing.assert_allclose(m.refit(r), sm.GLM(y[r], X[r], family=fam).fit().params,
                                   atol=1e-7)
    gam = sm.families.Gamma(sm.families.links.Log())
    yg = rng.gamma(2.0, np.exp(X @ [0.5, 0.2]) / 2.0)
    est = sm.GLM(yg, X, family=gam).fit().params
    m = bm.glm_model(yg, X, gam, None, None, ["c", "x"], est)
    r = _sample_idx(150, b=1)[0]
    np.testing.assert_allclose(m.refit(r), sm.GLM(yg[r], X[r], family=gam).fit().params,
                               atol=1e-6)


def test_discriminant_classification_functions_match_core():
    rng = np.random.default_rng(6)
    n = 120
    g = np.repeat([1.0, 2.0, 3.0], n // 3)
    X = rng.normal(size=(n, 2)) + g[:, None] * [0.8, -0.4]
    ds = Dataset(pd.DataFrame({"g": g, "a": X[:, 0], "b": X[:, 1]}))
    res = classify.discriminant(ds, "g", 1, 3, ["a", "b"], fisher=True)
    table = next(t for t in res.section.items if getattr(t, "title", "") ==
                 "Classification Function Coefficients")
    est = res.models[0].estimate.reshape(3, 3)       # rows a, b, constant; columns groups
    for i, row in enumerate(table.rows):
        assert [float(v) for v in row[1:]] == pytest.approx(est[i], abs=6e-4)


def test_regression_models_expose_coefficients():
    rng = np.random.default_rng(7)
    df = pd.DataFrame(rng.normal(size=(80, 3)), columns=["y", "a", "b"])
    res = regression.linear(Dataset(df), "y", blocks=[("enter", ["a"]), ("enter", ["b"])])
    assert [m.group for m in res.models] == ["1", "2"]
    assert res.models[1].names == ["(Constant)", "a", "b"]
    np.testing.assert_allclose(res.models[1].estimate,
                               np.linalg.lstsq(np.column_stack([np.ones(80), df[["a", "b"]]]),
                                               df["y"], rcond=None)[0])


def test_bootstrap_timing_default_1000_samples():
    rng = np.random.default_rng(9)
    n = 400
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    df["yb"] = (rng.random(n) < 1 / (1 + np.exp(-df["x1"]))).astype(float)
    df["yo"] = np.digitize(df["x1"] + rng.logistic(size=n), [-1, 0, 1]).astype(float)
    ds = Dataset(df)
    spec = rs.BootSpec(1000, 1)
    for res in (logistic.binary_logistic(ds, "yb", ["x1", "x2"]),
                logistic.ordinal(ds, "yo", [], ["x1", "x2"])):
        t0 = time.perf_counter()
        out = rs.bootstrap_model(res.models[0], spec)
        assert time.perf_counter() - t0 < 10
        assert out.min_valid == 1000 and np.all(out.se > 0)

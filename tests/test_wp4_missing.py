"""WP4 Missing Value Analysis, Multiple Imputation and Rubin's-rules pooling."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import imputation as mi
from rsstatchem.core.stats import missing as mv

DATA = Path(__file__).parent / "data_wp4"


def _table(section, title):
    return next(i for i in section.items if getattr(i, "title", "").startswith(title))


@pytest.fixture(scope="module")
def airquality():
    a = pd.read_csv(DATA / "airquality.csv").drop(columns="rownames")
    a.columns = ["Ozone", "Solar", "Wind", "Temp", "Month", "Day"]
    return a


# ------------------------------------------------------------------ MVA
def test_little_mcar_matches_naniar(airquality):
    """R: naniar::mcar_test(airquality) -> statistic 35.1, df 14, p.value 0.00142,
    missing.patterns 4."""
    chi, df, p, patterns = mv.little_mcar(airquality.to_numpy(float))
    assert chi == pytest.approx(35.1, abs=0.05)
    assert df == 14 and patterns == 4
    assert p == pytest.approx(0.00142, abs=2e-5)


def test_em_without_missing_is_ml_estimate(airquality):
    x = airquality[["Wind", "Temp", "Month"]].to_numpy(float)
    em = mv.em_normal(x)
    np.testing.assert_allclose(em.mean, x.mean(0))
    np.testing.assert_allclose(em.cov, np.cov(x, rowvar=False, ddof=0))


def test_em_fixed_point(airquality):
    """At convergence one more E- and M-step reproduces the estimates."""
    x = airquality[["Ozone", "Solar", "Wind", "Temp"]].to_numpy(float)
    em = mv.em_normal(x, max_iter=500, tol=1e-10)
    again = mv.em_normal(x, max_iter=1, mean0=em.mean, cov0=em.cov)
    np.testing.assert_allclose(again.mean, em.mean, atol=1e-7)
    np.testing.assert_allclose(again.cov, em.cov, atol=1e-6)
    assert em.mean[0] != pytest.approx(np.nanmean(x[:, 0]), abs=1e-3)   # EM moves Ozone


def test_mva_tables(airquality):
    ds = Dataset(airquality)
    sec = mv.mva(ds, ["Ozone", "Solar", "Wind", "Temp"], ["Month"],
                 ("listwise", "pairwise", "em", "regression"), ttest=True, crosstabs=True,
                 case_patterns=True)
    uni = _table(sec, "Univariate Statistics").rows
    assert uni[0][:5] == ["Ozone", "116", "42.13", "32.988", "37"]
    em_t = _table(sec, "EM Means")
    assert "DF = 8" in em_t.footnotes[0]
    lw = _table(sec, "Listwise Means").rows[0]
    assert lw[1] == "111"
    tt = _table(sec, "Separate Variance t Tests").rows
    a = airquality["Wind"][airquality["Ozone"].notna()]
    b = airquality["Wind"][airquality["Ozone"].isna()]
    t_ref = stats.ttest_ind(a, b, equal_var=False).statistic
    assert tt[0][0] == "Ozone" and tt[0][4] == f"{t_ref:.1f}".replace("0.", ".")
    pats = _table(sec, "Tabulated Patterns").rows
    assert [r[0] for r in pats] == ["111", "35", "5", "2"]
    assert pats[1][6] == "146"   # complete if Ozone is dropped


# ------------------------------------------------------------------ Rubin's rules
def test_rubin_rules_formulas():
    q = np.array([[1.0, 2.0], [1.2, 2.5], [0.9, 1.8], [1.1, 2.2], [1.05, 2.4]])
    u = np.array([[0.04, 0.09], [0.05, 0.08], [0.045, 0.1], [0.04, 0.09], [0.05, 0.11]])
    pl = mi.rubin(q, u)
    m = 5
    b = q.var(axis=0, ddof=1)
    ubar = u.mean(0)
    t = ubar + (1 + 1 / m) * b
    r = (1 + 1 / m) * b / ubar
    df = (m - 1) * (1 + 1 / r) ** 2
    np.testing.assert_allclose(pl.se, np.sqrt(t))
    np.testing.assert_allclose(pl.df, df)
    np.testing.assert_allclose(pl.fmi, (r + 2 / (df + 3)) / (r + 1))
    np.testing.assert_allclose(pl.re, 1 / (1 + pl.fmi / m))
    np.testing.assert_allclose(pl.p, 2 * stats.t.sf(np.abs(q.mean(0) / np.sqrt(t)), df))
    # Barnard & Rubin (1999) adjusted df
    br = mi.rubin(q, u, df_complete=30)
    lam = (1 + 1 / m) * b / t
    obs = 31 / 33 * 30 * (1 - lam)
    np.testing.assert_allclose(br.df, 1 / (1 / df + 1 / obs))


def _imputed_without_missing(n=60, m=3):
    rng = np.random.default_rng(2)
    x = rng.normal(size=n)
    y = 1 + x + rng.normal(size=n)
    g = (rng.random(n) < 0.5).astype(float)
    frames = [pd.DataFrame({"Imputation_": float(k), "x": x, "y": y, "g": g})
              for k in range(m + 1)]
    return Dataset(pd.concat(frames, ignore_index=True)), x, y, g


def test_pooling_with_identical_imputations_equals_single_analysis():
    ds, x, y, g = _imputed_without_missing()
    sec = mi.pooled_independent_t(ds, ["y"], "g", (0.0, 1.0))
    pooled = [r for r in _table(sec, "Independent Samples Test").rows if r[2] == "Pooled"][0]
    ref = stats.ttest_ind(y[g == 0], y[g == 1])
    assert float(pooled[3]) == pytest.approx(ref.statistic, abs=5e-4)
    assert pooled[-3:] == [".000", ".000", "1.000"]
    sec = mi.pooled_linear(ds, "y", ["x"])
    rows = [r for r in _table(sec, "Coefficients").rows]
    pooled_rows = rows[rows.index(next(r for r in rows if r[0] == "Pooled")):]
    import statsmodels.api as sm

    fit = sm.OLS(y, sm.add_constant(x)).fit()
    assert float(pooled_rows[1][2]) == pytest.approx(fit.params[1], abs=5e-4)
    assert float(pooled_rows[1][3]) == pytest.approx(fit.bse[1], abs=5e-4)
    sec = mi.pooled_correlations(ds, ["x", "y"])
    rows = _table(sec, "Correlations").rows
    pooled_r = [r for r in rows if r[0] == "Pooled"][0][4]
    assert float(pooled_r) == pytest.approx(np.corrcoef(x, y)[0, 1], abs=5e-4)


# ------------------------------------------------------------------ imputation
@pytest.fixture(scope="module")
def incomplete():
    rng = np.random.default_rng(7)
    n = 400
    x = rng.normal(size=n)
    z = 0.6 * x + rng.normal(size=n) * 0.8
    g = (rng.random(n) < 1 / (1 + np.exp(-1.5 * x))).astype(float)
    y = 1 + 2 * x - z + g + rng.normal(size=n)
    df = pd.DataFrame({"x": x, "z": z, "g": g, "y": y})
    full = df.copy()
    # MAR: missingness of y and z depends on the observed x
    df.loc[rng.random(n) < 1 / (1 + np.exp(-(x - 0.5))) * 0.5, "y"] = np.nan
    df.loc[rng.random(n) < 0.2, "z"] = np.nan
    df.loc[rng.random(n) < 0.15, "g"] = np.nan
    ds = Dataset(df)
    ds.var("g").measure = "nominal"
    return ds, full


def test_impute_structure_and_constraints(incomplete):
    ds, _ = incomplete
    res = mi.impute(ds, ["x", "z", "g", "y"], m=4, max_iter=5, seed=11,
                    constraints={"z": mi.Constraint(minimum=-1.5, maximum=1.5, rounding=0.5)})
    new = res.dataset
    assert new.names()[0] == "Imputation_" and new.n_cases == 5 * ds.n_cases
    imp = new.df["Imputation_"].to_numpy()
    orig = ds.df
    for k in range(1, 5):
        block = new.df[imp == k].reset_index(drop=True)
        assert not block[["z", "g", "y"]].isna().any().any()
        obs = orig["z"].notna()
        np.testing.assert_array_equal(block.loc[obs, "z"], orig.loc[obs, "z"])
        zi = block.loc[~obs, "z"]
        assert zi.between(-1.5, 1.5).all() and np.allclose(zi * 2, np.round(zi * 2))
        assert set(block.loc[orig["g"].isna(), "g"]) <= {0.0, 1.0}
    np.testing.assert_array_equal(new.df[imp == 0].reset_index(drop=True)["y"].isna(),
                                  orig["y"].isna())
    models = _table(res.section, "Imputation Models").rows
    assert models[1][1] == "Logistic Regression"
    again = mi.impute(ds, ["x", "z", "g", "y"], m=4, max_iter=5, seed=11,
                      constraints={"z": mi.Constraint(minimum=-1.5, maximum=1.5,
                                                      rounding=0.5)})
    pd.testing.assert_frame_equal(again.dataset.df, new.df)


def test_pmm_draws_observed_values(incomplete):
    ds, _ = incomplete
    res = mi.impute(ds, ["x", "z", "y"], m=2, max_iter=3, scale_model="pmm", seed=3)
    new = res.dataset
    observed = set(ds.df["y"].dropna().round(12))
    imputed = new.df.loc[(new.df["Imputation_"] > 0), "y"][
        np.tile(ds.df["y"].isna().to_numpy(), 2)]
    assert set(imputed.round(12)) <= observed


def test_pooled_regression_recovers_truth_under_mar(incomplete):
    ds, full = incomplete
    res = mi.impute(ds, ["x", "z", "g", "y"], m=10, max_iter=10, seed=5)
    sec = mi.pooled_linear(res.dataset, "y", ["x", "z", "g"])
    rows = _table(sec, "Coefficients").rows
    start = rows.index(next(r for r in rows if r[0] == "Pooled"))
    pooled = {r[1]: r for r in rows[start:]}
    for name, truth in (("x", 2.0), ("z", -1.0), ("g", 1.0)):
        lo, hi = float(pooled[name][7]), float(pooled[name][8])
        assert lo - 0.15 < truth < hi + 0.15
        assert float(pooled[name][9]) > 0          # fraction of missing information
    log = mi.pooled_logistic(res.dataset, "g", ["x"])
    rows = _table(log, "Variables in the Equation").rows
    b = float([r for r in rows if r[0] == "Pooled"][0][2])
    assert 0.8 < b < 2.4                            # true slope 1.5


def test_monotone_method():
    rng = np.random.default_rng(1)
    n = 200
    a = rng.normal(size=n)
    b = a + rng.normal(size=n)
    c = b + rng.normal(size=n)
    df = pd.DataFrame({"a": a, "b": b, "c": c})
    miss_b = rng.random(n) < 0.15
    df.loc[miss_b, "b"] = np.nan
    df.loc[miss_b | (rng.random(n) < 0.15), "c"] = np.nan
    ds = Dataset(df)
    res = mi.impute(ds, ["a", "b", "c"], m=2, method="auto")
    assert "Monotone" in _table(res.section, "Imputation Results").rows[0][2]
    df.loc[0, ["b", "c"]] = [np.nan, 1.0]
    df.loc[1, ["b", "c"]] = [1.0, np.nan]
    df.loc[2, ["b", "c"]] = [np.nan, 2.0]
    with pytest.raises(ValueError):
        mi.impute(Dataset(df), ["a", "b", "c"], m=2, method="monotone")


def test_analyze_patterns(incomplete):
    ds, _ = incomplete
    sec = mi.analyze_patterns(ds, ["x", "z", "g", "y"])
    overall = _table(sec, "Overall Summary of Missing Values").kind
    assert overall in ("image", "table")
    summary = [i for i in sec.items if i.kind == "table" and i.title.startswith("Overall")][0]
    assert summary.rows[0][3] == "3"      # three incomplete variables
    var = _table(sec, "Variable Summary").rows
    assert {r[0] for r in var} == {"y", "z", "g"}
    assert mi.is_imputed(ds) is False

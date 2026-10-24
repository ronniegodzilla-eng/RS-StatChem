"""PROCESS-style mediation / moderation checked against pingouin mediation_analysis and
statsmodels OLS (including HC standard errors)."""
import math

import numpy as np
import pingouin as pg
import pytest
import statsmodels.formula.api as smf
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import process


def num(text: str) -> float:
    return float(text.rstrip("*ᵃᵇ").replace(",", ""))


def table(section, title):
    items = [i for i in section.items if getattr(i, "kind", "") == "table"]
    exact = [i for i in items if i.title == title]
    return exact[0] if exact else next(i for i in items if i.title.startswith(title))


@pytest.fixture
def df():
    data = process.selftest_data(200, seed=11)
    rng = np.random.default_rng(2)
    data["m2"] = 0.3 * data["x"] + rng.normal(0, 1, 200)
    data["c"] = rng.normal(0, 1, 200)
    return data


def test_model4_matches_pingouin(df):
    res = process.process(Dataset(df), 4, "y", "x", ["m", "m2"], covariates=["c"], boot=5000,
                          seed=123)
    ref = pg.mediation_analysis(df, x="x", m=["m", "m2"], y="y", covar=["c"], n_boot=5000,
                                seed=1).set_index("path")
    e = res.effects
    assert e["total"] == pytest.approx(ref.loc["Total", "coef"], abs=1e-6)
    assert e["direct"] == pytest.approx(ref.loc["Direct", "coef"], abs=1e-6)
    assert e["indirect"][0] == pytest.approx(ref.loc["Indirect m", "coef"], abs=1e-6)
    assert e["indirect"][1] == pytest.approx(ref.loc["Indirect m2", "coef"], abs=1e-6)
    assert sum(e["indirect"]) == pytest.approx(e["total"] - e["direct"], abs=1e-9)
    lo, hi = e["indirect_ci"][0]
    assert lo == pytest.approx(ref.loc["Indirect m", "CI2.5"], abs=0.03)
    assert hi == pytest.approx(ref.loc["Indirect m", "CI97.5"], abs=0.03)
    a = table(res.section, "OUTCOME VARIABLE: m — Model").rows[1]
    assert num(a[1]) == pytest.approx(ref.loc["m ~ X", "coef"], abs=1e-4)
    assert num(a[2]) == pytest.approx(ref.loc["m ~ X", "se"], abs=1e-4)


def test_model4_bootstrap_reproducible_and_standardized(df):
    ds = Dataset(df)
    r1 = process.process(ds, 4, "y", "x", ["m"], boot=1000, seed=42)
    r2 = process.process(ds, 4, "y", "x", ["m"], boot=1000, seed=42)
    assert r1.effects["indirect_ci"] == r2.effects["indirect_ci"]
    sd = df["x"].std(ddof=1) / df["y"].std(ddof=1)
    assert r1.effects["std_indirect"][0] == pytest.approx(r1.effects["indirect"][0] * sd)
    sob = process.process(ds, 4, "y", "x", ["m"], boot=0, normal=True)
    row = table(sob.section, "Normal theory test").rows[0]
    a = smf.ols("m ~ x", df).fit()
    b = smf.ols("y ~ x + m", df).fit()
    se = math.sqrt(a.params.x ** 2 * b.bse.m ** 2 + b.params.m ** 2 * a.bse.x ** 2)
    assert num(row[2]) == pytest.approx(se, abs=1e-4)


def test_boot_ci_positions():
    v = np.arange(1, 5001, dtype=float)
    assert process.boot_ci(v, 0.95) == (125.0, 4876.0)


def test_model1_matches_statsmodels(df):
    res = process.process(Dataset(df), 1, "y", "x", w="w", boot=0)
    fit = smf.ols("y ~ x * w", df).fit()
    coef = table(res.section, "OUTCOME VARIABLE: y — Model").rows
    for row, name in zip(coef, ("Intercept", "x", "w", "x:w")):
        assert num(row[1]) == pytest.approx(fit.params[name], abs=1e-4)
        assert num(row[2]) == pytest.approx(fit.bse[name], abs=1e-4)
    summ = table(res.section, "OUTCOME VARIABLE: y — Model Summary").rows[0]
    assert num(summ[1]) == pytest.approx(fit.rsquared, abs=1e-4)
    assert num(summ[3]) == pytest.approx(fit.fvalue, abs=1e-3)
    inter = table(res.section, "Test(s) of highest order").rows[0]
    assert num(inter[2]) == pytest.approx(fit.tvalues["x:w"] ** 2, abs=1e-3)
    # conditional effect at the 16th percentile of w
    w16 = float(np.percentile(df["w"], 16, method="weibull"))
    cond = table(res.section, "Conditional effects of the focal predictor").rows[0]
    eff = fit.params["x"] + fit.params["x:w"] * w16
    V = fit.cov_params()
    se = math.sqrt(V.loc["x", "x"] + 2 * w16 * V.loc["x", "x:w"] + w16 ** 2 * V.loc["x:w", "x:w"])
    assert num(cond[0]) == pytest.approx(w16, abs=1e-4)
    assert num(cond[1]) == pytest.approx(eff, abs=1e-4)
    assert num(cond[2]) == pytest.approx(se, abs=1e-4)
    # Johnson-Neyman points are exactly at |t| = critical t
    crit = stats.t.ppf(0.975, fit.df_resid)
    for pt in res.effects["jn"]:
        eff = fit.params["x"] + fit.params["x:w"] * pt
        se = math.sqrt(V.loc["x", "x"] + 2 * pt * V.loc["x", "x:w"] + pt ** 2 *
                       V.loc["x:w", "x:w"])
        assert abs(eff / se) == pytest.approx(crit, abs=1e-6)
    assert any(i.kind == "image" for i in res.section.items)


@pytest.mark.parametrize("hc", ["HC0", "HC1", "HC2", "HC3"])
def test_hc_standard_errors(df, hc):
    res = process.process(Dataset(df), 1, "y", "x", w="w", boot=0, hc=hc, jn=False)
    fit = smf.ols("y ~ x * w", df).fit(cov_type=hc)
    coef = table(res.section, "OUTCOME VARIABLE: y — Model").rows
    for row, name in zip(coef, ("Intercept", "x", "w", "x:w")):
        assert num(row[2]) == pytest.approx(fit.bse[name], abs=1e-4)


def test_moments_and_centering(df):
    res = process.process(Dataset(df), 1, "y", "x", w="w", boot=0, center=True, moments=True)
    cond = table(res.section, "Conditional effects of the focal predictor").rows
    sd = df["w"].std(ddof=1)
    assert [num(r[0]) for r in cond] == pytest.approx([-sd, 0, sd], abs=1e-4)
    dfc = df.assign(x=df.x - df.x.mean(), w=df.w - df.w.mean())
    fit = smf.ols("y ~ x * w", dfc).fit()
    coef = table(res.section, "OUTCOME VARIABLE: y — Model").rows
    assert num(coef[1][1]) == pytest.approx(fit.params["x"], abs=1e-4)


def test_models_7_and_14_indices(df):
    ds = Dataset(df)
    r7 = process.process(ds, 7, "y", "x", ["m"], w="w", boot=500, seed=3)
    a = smf.ols("m ~ x * w", df).fit()
    b = smf.ols("y ~ x + m", df).fit()
    assert r7.effects["index"][0] == pytest.approx(a.params["x:w"] * b.params["m"], abs=1e-9)
    r14 = process.process(ds, 14, "y", "x", ["m"], w="w", boot=500, seed=3)
    a = smf.ols("m ~ x", df).fit()
    b = smf.ols("y ~ x + m * w", df).fit()
    assert r14.effects["index"][0] == pytest.approx(a.params["x"] * b.params["m:w"], abs=1e-9)
    w50 = float(np.percentile(df["w"], 50, method="weibull"))
    cond = r14.effects["conditional_indirect"][0][1]
    assert cond == pytest.approx(a.params["x"] * (b.params["m"] + b.params["m:w"] * w50))

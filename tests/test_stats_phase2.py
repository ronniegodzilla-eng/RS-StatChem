"""Phase 2 analyses checked against SciPy / statsmodels / pingouin reference implementations."""
import math

import numpy as np
import pandas as pd
import pingouin as pg
import pytest
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import (anova, common, correlate, crosstabs, explore, nonparametric,
                                   regression, reliability, ttest)


def num(text: str) -> float:
    return float(text.rstrip("*ᵃᵇᶜᵈ").replace(",", ""))


def table(section, title):
    return next(i for i in section.items if getattr(i, "title", "").startswith(title))


def row(tab, *heads):
    for r in tab.rows:
        if all(h in r for h in heads):
            return r
    raise AssertionError(f"row {heads} not found in {tab.title}")


@pytest.fixture
def ds():
    rng = np.random.default_rng(42)
    n = 90
    g3 = np.repeat([1, 2, 3], 30)
    g2 = np.tile([1, 2], 45)
    x1 = rng.normal(50, 10, n)
    x2 = 0.5 * x1 + rng.normal(0, 8, n)
    x3 = rng.normal(0, 1, n)
    y = 3 + 0.8 * x1 + 0.3 * x2 + g3 * 2 + rng.normal(0, 6, n)
    items = np.clip(np.round(rng.normal(3, 1, (n, 1)) + rng.normal(0, 0.7, (n, 5))), 1, 5)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3, "g3": g3, "g2": g2,
                       "pre": x1, "post": x1 + rng.normal(3, 5, n),
                       **{f"i{j + 1}": items[:, j] for j in range(5)}})
    data = Dataset(df)
    data.var("g3").value_labels = {1.0: "A", 2.0: "B", 3.0: "C"}
    data.var("g2").value_labels = {1.0: "Kontrol", 2.0: "Eksperimen"}
    return data


def test_fmt_spss_style():
    assert common.fmt(0.0451) == ".045"
    assert common.fmt(-0.5) == "-.500"
    assert common.fmt_p(0.0001) == ".000"
    assert common.fmt(12.3456, 2) == "12.35"


def test_trimmed_mean_matches_integer_trim():
    x = np.arange(1.0, 21.0)
    x[-1] = 1000
    assert math.isclose(common.trimmed_mean(x), np.mean(np.sort(x)[1:-1]))


def test_explore_normality(ds):
    sec = explore.explore(ds, ["y"], ["g2"])
    t = table(sec, "Tests of Normality")
    for value, label in ((1.0, "Kontrol"), (2.0, "Eksperimen")):
        x = ds.df.loc[ds.df.g2 == value, "y"].to_numpy()
        r = row(t, f"g2 = {label}")
        assert math.isclose(num(r[5]), stats.shapiro(x).statistic, abs_tol=5e-4)
        assert math.isclose(num(r[7]), stats.shapiro(x).pvalue, abs_tol=5e-4)
    assert any(i.kind == "image" for i in sec.items)


def test_one_sample(ds):
    sec = ttest.one_sample(ds, ["y"], test_value=50)
    t = table(sec, "One-Sample Test")
    ref = stats.ttest_1samp(ds.df.y, 50)
    assert math.isclose(num(t.rows[0][1]), ref.statistic, abs_tol=5e-4)
    assert math.isclose(num(t.rows[0][4]), ref.pvalue, abs_tol=5e-4)
    eff = table(sec, "One-Sample Effect Sizes")
    d = (ds.df.y.mean() - 50) / ds.df.y.std(ddof=1)
    assert math.isclose(num(eff.rows[0][3]), d, abs_tol=5e-4)


def test_independent(ds):
    sec = ttest.independent(ds, ["y"], "g2", groups=(1.0, 2.0))
    t = table(sec, "Independent Samples Test")
    a = ds.df.loc[ds.df.g2 == 1, "y"]
    b = ds.df.loc[ds.df.g2 == 2, "y"]
    eq, welch = stats.ttest_ind(a, b), stats.ttest_ind(a, b, equal_var=False)
    lev = stats.levene(a, b, center="mean")
    assert math.isclose(num(t.rows[0][2]), lev.statistic, abs_tol=5e-4)
    assert math.isclose(num(t.rows[0][4]), eq.statistic, abs_tol=5e-4)
    assert math.isclose(num(t.rows[0][7]), eq.pvalue, abs_tol=5e-4)
    assert math.isclose(num(t.rows[1][4]), welch.statistic, abs_tol=5e-4)
    assert math.isclose(num(t.rows[1][5]), welch.df, abs_tol=5e-3)
    eff = table(sec, "Independent Samples Effect Sizes")
    assert math.isclose(num(eff.rows[0][3]), pg.compute_effsize(a, b, eftype="cohen"),
                        abs_tol=5e-4)
    assert math.isclose(num(eff.rows[1][3]), pg.compute_effsize(a, b, eftype="hedges"),
                        abs_tol=2e-3)


def test_paired(ds):
    sec = ttest.paired(ds, [("pre", "post")])
    eff = table(sec, "Paired Samples Effect Sizes")
    assert eff.rows[0][5] and eff.rows[0][6]  # CI bounds present for a negative t
    t = table(sec, "Paired Samples Test")
    ref = stats.ttest_rel(ds.df.pre, ds.df.post)
    assert math.isclose(num(t.rows[0][7]), ref.statistic, abs_tol=5e-4)
    assert math.isclose(num(t.rows[0][10]), ref.pvalue, abs_tol=5e-4)


def test_oneway_and_posthoc(ds):
    sec = anova.oneway(ds, ["y"], "g3", welch=True, brown_forsythe=True,
                       posthoc_methods=("tukey", "bonferroni", "games_howell", "scheffe"))
    groups = [ds.df.loc[ds.df.g3 == k, "y"].to_numpy() for k in (1, 2, 3)]
    a = table(sec, "ANOVA")
    assert math.isclose(num(a.rows[0][5]), stats.f_oneway(*groups).statistic, abs_tol=5e-4)
    welch = pg.welch_anova(data=ds.df, dv="y", between="g3")
    robust = table(sec, "Robust Tests")
    assert math.isclose(num(robust.rows[0][2]), welch["F"].iloc[0], abs_tol=5e-3)
    mc = table(sec, "Multiple Comparisons")
    tuk = stats.tukey_hsd(*groups)
    assert math.isclose(num(mc.rows[0][6]), tuk.pvalue[0, 1], abs_tol=2e-3)
    gh = pg.pairwise_gameshowell(data=ds.df, dv="y", between="g3")
    gh_rows = [r for r in mc.rows if r[1] == "Games-Howell" or r is mc.rows[
        [x[1] for x in mc.rows].index("Games-Howell")]]
    gh_ab = gh[(gh.A == 1) & (gh.B == 2)]["pval"].iloc[0]
    assert math.isclose(num(gh_rows[0][6]), gh_ab, abs_tol=2e-3)
    eff = table(sec, "ANOVA Effect Sizes")
    assert math.isclose(num(eff.rows[0][2]),
                        pg.anova(data=ds.df, dv="y", between="g3")["np2"].iloc[0], abs_tol=5e-4)


def test_univariate_two_way_type3(ds):
    df = ds.df.iloc[:-7]  # unbalanced on purpose
    sub = Dataset(df.reset_index(drop=True))
    sec = anova.univariate(sub, "y", ["g3", "g2"], homogeneity=True, descriptives=True,
                           posthoc_factors=["g3"], posthoc_methods=("tukey",))
    t = table(sec, "Tests of Between-Subjects Effects")
    ref = pg.anova(data=df, dv="y", between=["g3", "g2"], ss_type=3)
    for source, key in (("g3", "g3"), ("g2", "g2"), ("g3 * g2", "g3 * g2")):
        r = row(t, source)
        ref_row = ref[ref.Source == key].iloc[0]
        assert math.isclose(num(r[1]), ref_row["SS"], rel_tol=1e-3)
        assert math.isclose(num(r[4]), ref_row["F"], abs_tol=5e-3)
        assert math.isclose(num(r[6]), ref_row["np2"], abs_tol=5e-4)
    assert table(sec, "Multiple Comparisons")


def test_ancova_runs(ds):
    sec = anova.univariate(ds, "y", ["g3"], covariates=["x1"])
    t = table(sec, "Tests of Between-Subjects Effects")
    ref = pg.ancova(data=ds.df, dv="y", covar="x1", between="g3")
    assert math.isclose(num(row(t, "g3")[4]), ref[ref.Source == "g3"]["F"].iloc[0],
                        abs_tol=5e-3)


def test_correlations(ds):
    sec = correlate.bivariate(ds, ["x1", "x2", "x3"], pearson=True, spearman=True,
                              kendall=True)
    p = sec.items[1]
    assert math.isclose(num(p.rows[0][3]), stats.pearsonr(ds.df.x1, ds.df.x2)[0], abs_tol=5e-4)
    assert p.rows[0][3].endswith("**")
    nonpar = sec.items[2]
    rho = row(nonpar, "Spearman's rho")
    assert math.isclose(num(rho[4]), stats.spearmanr(ds.df.x1, ds.df.x2)[0], abs_tol=5e-4)
    kt = nonpar.rows[0]
    assert math.isclose(num(kt[4]), stats.kendalltau(ds.df.x1, ds.df.x2)[0], abs_tol=5e-4)

    sec = correlate.partial(ds, ["y", "x2"], ["x1"])
    ref = pg.partial_corr(data=ds.df, x="y", y="x2", covar="x1")
    assert math.isclose(num(sec.items[-1].rows[0][4]), ref["r"].iloc[0], abs_tol=5e-4)


def test_regression_enter_and_assumptions(ds):
    res = regression.linear(ds, "y", ["x1", "x2", "x3"],
                            statistics=regression.DEFAULT_STATS + ("ci", "r2_change"),
                            save=("pred", "resid", "zresid"))
    sec = res.section
    import statsmodels.api as sm
    X = sm.add_constant(ds.df[["x1", "x2", "x3"]])
    fit = sm.OLS(ds.df.y, X).fit()
    summary = table(sec, "Model Summary")
    assert math.isclose(num(summary.rows[0][2]), fit.rsquared, abs_tol=5e-4)
    coef = table(sec, "Coefficients")
    assert math.isclose(num(row(coef, "x1")[2]), fit.params["x1"], abs_tol=5e-4)
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif_x1 = variance_inflation_factor(X.values, 1)
    assert math.isclose(num(row(coef, "x1")[-1]), vif_x1, abs_tol=5e-3)
    for title in ("One-Sample Kolmogorov-Smirnov Test", "Collinearity Statistics",
                  "Glejser Test", "Heteroscedasticity Tests", "Autocorrelation Tests"):
        assert table(sec, title)
    assert set(res.saved) == {"PRE", "RES", "ZRE"}
    assert np.allclose(res.saved["RES"][1].to_numpy(), fit.resid.to_numpy())
    assert sum(i.kind == "image" for i in sec.items) == 3


def test_regression_stepwise_drops_noise(ds):
    res = regression.linear(ds, "y", ["x1", "x2", "x3"], method="stepwise")
    entered = table(res.section, "Variables Entered")
    names = [r[1] for r in entered.rows]
    assert names[0] == "x1"
    assert "x3" not in names


def test_ks_and_runs_helpers():
    rng = np.random.default_rng(3)
    x = rng.normal(size=60)
    ks = nonparametric.ks_details(x)
    assert math.isclose(ks["absolute"], ks["statistic"], abs_tol=1e-9)
    from statsmodels.sandbox.stats.runs import runstest_1samp
    rd = nonparametric.runs_details(x)
    z, p = runstest_1samp(x, cutoff="median", correction=False)
    assert math.isclose(rd["z"], z, abs_tol=1e-6)


def test_nonparametric_tests(ds):
    sec = nonparametric.mann_whitney(ds, ["y"], "g2", (1.0, 2.0))
    a = ds.df.loc[ds.df.g2 == 1, "y"]
    b = ds.df.loc[ds.df.g2 == 2, "y"]
    ref = stats.mannwhitneyu(a, b, method="asymptotic", use_continuity=False)
    st = table(sec, "Test Statistics")
    assert math.isclose(num(row(st, "Mann-Whitney U")[1]), min(ref.statistic,
                                                               len(a) * len(b) - ref.statistic))
    assert math.isclose(num(row(st, "Asymp. Sig. (2-tailed)")[1]), ref.pvalue, abs_tol=5e-4)

    sec = nonparametric.kruskal_wallis(ds, ["y"], "g3")
    ref = stats.kruskal(*[ds.df.loc[ds.df.g3 == k, "y"] for k in (1, 2, 3)])
    assert math.isclose(num(table(sec, "Test Statistics").rows[0][1]), ref.statistic,
                        abs_tol=5e-4)

    sec = nonparametric.two_related(ds, [("pre", "post")], wilcoxon=True, sign=True)
    ref = stats.wilcoxon(ds.df.post - ds.df.pre, correction=False, method="approx")
    z_row = table(sec, "Test Statistics").rows
    assert math.isclose(num(z_row[1][1]), ref.pvalue, abs_tol=5e-4)

    sec = nonparametric.k_related(ds, ["i1", "i2", "i3"], friedman=True, kendall=True)
    ref = stats.friedmanchisquare(ds.df.i1, ds.df.i2, ds.df.i3)
    assert math.isclose(num(row(table(sec, "Test Statistics"), "Chi-Square")[1]),
                        ref.statistic, abs_tol=5e-4)

    sec = nonparametric.chi_square(ds, ["g3"])
    assert table(sec, "Test Statistics")
    sec = nonparametric.binomial(ds, ["g2"])
    assert num(table(sec, "Binomial Test").rows[0][6]) == 1.0
    assert nonparametric.ks_one_sample(ds, ["x1"]).items[-1].rows[0][2] == "90"


def test_crosstabs(ds):
    sec = crosstabs.crosstabs(ds, ["g3"], ["g2"], chisq=True, phi=True,
                              cells=("observed", "expected", "row"))
    obs = pd.crosstab(ds.df.g3, ds.df.g2).to_numpy()
    chi, p, _, _ = stats.chi2_contingency(obs, correction=False)
    ct = table(sec, "Chi-Square Tests")
    assert math.isclose(num(ct.rows[0][1]), chi, abs_tol=5e-4)
    sm = table(sec, "Symmetric Measures")
    assert math.isclose(num(row(sm, "Cramer's V")[2]), math.sqrt(chi / (obs.sum() * 1)),
                        abs_tol=5e-4)
    sec2 = crosstabs.crosstabs(ds, ["g2"], ["g2"], chisq=True)
    assert any("Fisher" in r[0] for r in table(sec2, "Chi-Square Tests").rows)


def test_reliability_and_validity(ds):
    items = [f"i{j}" for j in range(1, 6)]
    sec = reliability.reliability(ds, items, item_stats=True, scale_stats=True,
                                  validity=True)
    alpha = num(table(sec, "Reliability Statistics").rows[0][0])
    assert math.isclose(alpha, pg.cronbach_alpha(data=ds.df[items])[0], abs_tol=5e-4)
    assert math.isclose(reliability.r_table(30), 0.361, abs_tol=5e-4)  # classic r-table value
    val = table(sec, "Item Validity")
    assert all(r[-1] in ("Valid", "Not valid") for r in val.rows)

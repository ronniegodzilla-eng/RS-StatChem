"""Repeated Measures, MANOVA and Linear Mixed Models checked against statsmodels / pingouin and
closed-form results."""
import math

import numpy as np
import pandas as pd
import pingouin as pg
import pytest
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.anova import AnovaRM, anova_lm

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import manova, mixed, repeated


def num(text: str) -> float:
    return float(text.rstrip("*ᵃᵇᶜᵈᵉ").replace(",", ""))


def table(section, title):
    return next(i for i in section.items if getattr(i, "title", "").startswith(title))


def rows_of(tab, label, col=0):
    """Rows of a nested table starting at the row whose column ``col`` is ``label``."""
    start = next(i for i, r in enumerate(tab.rows) if r[col] == label)
    out = [tab.rows[start]]
    for r in tab.rows[start + 1:]:
        if r[col]:
            break
        out.append(r)
    return out


@pytest.fixture
def wide():
    rng = np.random.default_rng(3)
    n = 24
    subj = rng.normal(0, 2, n)
    g = np.repeat([1.0, 2.0, 3.0], 8)
    t = np.column_stack([subj + rng.normal(m, s, n)
                         for s, m in ((1, 0), (1.5, 0.8), (2.5, 1.0), (1, 2))])
    t[:, 2] += g * 0.7
    df = pd.DataFrame(t, columns=["t1", "t2", "t3", "t4"])
    df["g"] = g
    df["x"] = rng.normal(0, 1, n)
    return df


def _long(df, cols, between=None):
    keep = ["index"] + ([between] if between else [])
    return df.reset_index().melt(id_vars=keep, value_vars=cols, var_name="time",
                                 value_name="y")


# ------------------------------------------------------------------ repeated measures
def test_rm_oneway_matches_pingouin(wide):
    ds = Dataset(wide)
    cols = ["t1", "t2", "t3", "t4"]
    sec = repeated.repeated_measures(ds, [("time", 4)], cols)
    ref = pg.rm_anova(_long(wide, cols), dv="y", within="time", subject="index",
                      correction=True, detailed=True)
    within = table(sec, "Tests of Within-Subjects Effects")
    assert num(within.rows[0][2]) == pytest.approx(ref.loc[0, "SS"], abs=1e-3)
    assert num(within.rows[0][5]) == pytest.approx(ref.loc[0, "F"], abs=1e-3)
    assert num(within.rows[1][6]) == pytest.approx(ref.loc[0, "p_GG_corr"], abs=1e-3)
    spher = pg.sphericity(_long(wide, cols), dv="y", within="time", subject="index")
    m = table(sec, "Mauchly").rows[0]
    assert num(m[1]) == pytest.approx(spher.W, abs=1e-3)
    assert num(m[2]) == pytest.approx(spher.chi2, abs=1e-3)
    assert int(m[3]) == spher.dof
    assert num(m[5]) == pytest.approx(pg.epsilon(_long(wide, cols), dv="y", within="time",
                                                 subject="index"), abs=1e-3)
    assert num(m[6]) == pytest.approx(pg.epsilon(_long(wide, cols), dv="y", within="time",
                                                 subject="index", correction="hf"), abs=1e-3)
    assert num(m[7]) == pytest.approx(1 / 3, abs=1e-3)
    aov = AnovaRM(_long(wide, cols), "y", "index", within=["time"]).fit().anova_table
    assert num(within.rows[0][5]) == pytest.approx(aov["F Value"].iloc[0], abs=1e-3)


def test_rm_contrasts_sum_to_effect(wide):
    ds = Dataset(wide)
    sec = repeated.repeated_measures(ds, [("time", 4)], ["t1", "t2", "t3", "t4"])
    contrasts = rows_of(table(sec, "Tests of Within-Subjects Contrasts"), "time")
    within = table(sec, "Tests of Within-Subjects Effects")
    assert sum(num(r[2]) for r in contrasts) == pytest.approx(num(within.rows[0][2]), abs=2e-3)
    # linear contrast F equals a one-sample t^2 on the orthonormal linear scores
    c = repeated.poly_contrasts(4)[:, 0]
    scores = wide[["t1", "t2", "t3", "t4"]].to_numpy() @ c
    t = stats.ttest_1samp(scores, 0).statistic
    assert num(contrasts[0][5]) == pytest.approx(t * t, abs=2e-3)


def test_rm_multivariate_equals_hotelling(wide):
    ds = Dataset(wide)
    cols = ["t1", "t2", "t3", "t4"]
    sec = repeated.repeated_measures(ds, [("time", 4)], cols)
    D = wide[cols].to_numpy() @ repeated.poly_contrasts(4)
    frame = pd.DataFrame(D, columns=["d1", "d2", "d3"])
    ref = MANOVA.from_formula("d1 + d2 + d3 ~ 1", frame).mv_test().results["Intercept"]["stat"]
    mv = table(sec, "Multivariate Tests")
    assert num(mv.rows[1][2]) == pytest.approx(ref.loc["Wilks' lambda", "Value"], abs=1e-3)
    assert num(mv.rows[0][3]) == pytest.approx(ref.loc["Pillai's trace", "F Value"], abs=2e-3)


def test_rm_pairwise_is_paired_t(wide):
    ds = Dataset(wide)
    cols = ["t1", "t2", "t3", "t4"]
    for adj, factor in (("lsd", 1), ("bonferroni", 6)):
        sec = repeated.repeated_measures(ds, [("time", 4)], cols, emm=["time"], compare=True,
                                         adjustment=adj)
        pw = table(sec, "Pairwise Comparisons")
        r = pw.rows[0]  # 1 vs 2
        res = stats.ttest_rel(wide["t1"], wide["t2"])
        assert num(r[2]) == pytest.approx(wide["t1"].mean() - wide["t2"].mean(), abs=1e-3)
        assert num(r[4]) == pytest.approx(min(1, res.pvalue * factor), abs=1e-3)
    est = table(sec, "Estimates").rows
    assert num(est[0][1]) == pytest.approx(wide["t1"].mean(), abs=1e-3)
    assert num(est[0][2]) == pytest.approx(wide["t1"].std(ddof=1) / math.sqrt(24), abs=1e-3)


def test_rm_two_within_factors_match_anovarm():
    rng = np.random.default_rng(8)
    n = 20
    subj = rng.normal(0, 1.5, n)
    cols, data = [], {}
    for a in range(2):
        for b in range(3):
            name = f"a{a + 1}b{b + 1}"
            cols.append(name)
            data[name] = subj + rng.normal(0.4 * a + 0.3 * b + 0.5 * a * b, 1, n)
    df = pd.DataFrame(data)
    sec = repeated.repeated_measures(Dataset(df), [("A", 2), ("B", 3)], cols)
    long = df.reset_index().melt(id_vars="index", var_name="cell", value_name="y")
    long["A"] = long["cell"].str[1]
    long["B"] = long["cell"].str[3]
    aov = AnovaRM(long, "y", "index", within=["A", "B"]).fit().anova_table
    within = table(sec, "Tests of Within-Subjects Effects")
    for effect, key in (("A", "A"), ("B", "B"), ("A * B", "A:B")):
        r = rows_of(within, effect)[0]
        assert num(r[5]) == pytest.approx(aov.loc[key, "F Value"], abs=1e-3)
    wsf = table(sec, "Within-Subjects Factors")
    assert wsf.rows[1][1:] == ["2", "a1b2"]


def test_rm_mixed_design_matches_pingouin(wide):
    ds = Dataset(wide)
    cols = ["t1", "t2", "t3", "t4"]
    sec = repeated.repeated_measures(ds, [("time", 4)], cols, ["g"], homogeneity=True)
    ref = pg.mixed_anova(_long(wide, cols, "g"), dv="y", within="time", between="g",
                         subject="index").set_index("Source")
    within = table(sec, "Tests of Within-Subjects Effects")
    assert num(rows_of(within, "time")[0][5]) == pytest.approx(ref.loc["time", "F"], abs=1e-3)
    assert num(rows_of(within, "time * g")[0][5]) == pytest.approx(
        ref.loc["Interaction", "F"], abs=1e-3)
    between = table(sec, "Tests of Between-Subjects Effects")
    g_row = next(r for r in between.rows if r[0] == "g")
    assert num(g_row[4]) == pytest.approx(ref.loc["g", "F"], abs=1e-3)
    assert num(g_row[1]) == pytest.approx(ref.loc["g", "SS"], abs=1e-3)
    mauchly = table(sec, "Mauchly").rows[0]
    assert num(mauchly[1]) == pytest.approx(ref.loc["time", "W_spher"], abs=1e-3)
    assert table(sec, "Levene's Test") is not None
    assert table(sec, "Box's Test") is not None


def test_rm_covariate_and_plot(wide):
    ds = Dataset(wide)
    sec = repeated.repeated_measures(ds, [("time", 4)], ["t1", "t2", "t3", "t4"], ["g"],
                                     ["x"], emm=["g", "time*g"], compare=True,
                                     plots=[("time", "g")])
    assert any(i.kind == "image" for i in sec.items)
    est = table(sec, "Estimates — g")
    assert any("Covariates appearing" in f for f in est.footnotes)
    mv = table(sec, "Multivariate Tests")
    assert any(r[0] == "time * x" for r in mv.rows)


# ------------------------------------------------------------------ MANOVA
@pytest.fixture
def mv_data():
    rng = np.random.default_rng(1)
    n = 60
    g = np.repeat([1.0, 2.0, 3.0], 20)
    h = np.tile([1.0, 2.0], 30)
    y1 = rng.normal(0, 1, n) + g * 0.5
    y2 = 0.5 * y1 + rng.normal(0, 1, n) + h * 0.3
    y3 = rng.normal(0, 1, n)
    df = pd.DataFrame(dict(y1=y1, y2=y2, y3=y3, g=g, h=h, x=rng.normal(0, 1, n)))
    return df.iloc[3:].reset_index(drop=True)  # unbalanced


def test_manova_matches_statsmodels(mv_data):
    sec = manova.manova(Dataset(mv_data), ["y1", "y2", "y3"], ["g", "h"], ["x"])
    ref = MANOVA.from_formula("y1 + y2 + y3 ~ x + C(g, Sum) * C(h, Sum)", mv_data).mv_test()
    mv = table(sec, "Multivariate Tests")
    for effect, key in (("g", "C(g, Sum)"), ("g * h", "C(g, Sum):C(h, Sum)"), ("x", "x"),
                        ("Intercept", "Intercept")):
        stat = ref.results[key]["stat"]
        rows = rows_of(mv, effect)
        assert num(rows[0][2]) == pytest.approx(stat.loc["Pillai's trace", "Value"], abs=1e-3)
        assert num(rows[1][2]) == pytest.approx(stat.loc["Wilks' lambda", "Value"], abs=1e-3)
        assert num(rows[2][2]) == pytest.approx(stat.loc["Hotelling-Lawley trace", "Value"],
                                                abs=1e-3)
        assert num(rows[3][2]) == pytest.approx(stat.loc["Roy's greatest root", "Value"],
                                                abs=1e-3)
        for i, name in ((0, "Pillai's trace"), (1, "Wilks' lambda"), (3, "Roy's greatest root")):
            assert num(rows[i][3]) == pytest.approx(stat.loc[name, "F Value"], abs=2e-3)
            assert num(rows[i][6]) == pytest.approx(stat.loc[name, "Pr > F"], abs=1e-3)


def test_manova_hotelling_spss_approximation(mv_data):
    """SPSS uses the Pillai-Samson F for Hotelling's trace: df2 = 2(sn + 1)."""
    sec = manova.manova(Dataset(mv_data), ["y1", "y2", "y3"], ["g", "h"], ["x"])
    row = rows_of(table(sec, "Multivariate Tests"), "g")[2]
    p, q, ve = 3, 2, 50
    s, m, n = min(p, q), (abs(p - q) - 1) / 2, (ve - p - 1) / 2
    T = num(row[2])
    assert num(row[4]) == s * (2 * m + s + 1)
    assert num(row[5]) == 2 * (s * n + 1)
    assert num(row[3]) == pytest.approx(2 * (s * n + 1) * T / (s * s * (2 * m + s + 1)),
                                        abs=2e-3)


def test_manova_between_subjects_and_box(mv_data):
    sec = manova.manova(Dataset(mv_data), ["y1", "y2", "y3"], ["g", "h"])
    tab = table(sec, "Tests of Between-Subjects Effects")
    for dep in ("y1", "y2"):
        ref = anova_lm(smf.ols(f"{dep} ~ C(g, Sum) * C(h, Sum)", mv_data).fit(), typ=3)
        g_rows = rows_of(tab, "g")
        r = next(x for x in g_rows if x[1] == dep)
        assert num(r[2]) == pytest.approx(ref.loc["C(g, Sum)", "sum_sq"], abs=1e-3)
        assert num(r[5]) == pytest.approx(ref.loc["C(g, Sum)", "F"], abs=1e-3)
    box = table(sec, "Box's Test")
    long = mv_data.assign(cell=mv_data["g"].astype(str) + mv_data["h"].astype(str))
    ref = pg.box_m(long, dvs=["y1", "y2", "y3"], group="cell")
    assert num(box.rows[0][1]) == pytest.approx(
        ref["Chi2"].iloc[0] / (1 - _box_c1(long)), rel=1e-3)


def _box_c1(long):
    groups = [len(x) for _, x in long.groupby("cell")]
    p, g, N = 3, len(groups), sum(groups)
    return (sum(1 / (n - 1) for n in groups) - 1 / (N - g)) * (2 * p * p + 3 * p - 1) / \
        (6 * (p + 1) * (g - 1))


def test_manova_emm_and_posthoc(mv_data):
    sec = manova.manova(Dataset(mv_data), ["y1", "y2"], ["g"], emm_factors=["g"],
                        compare=True, posthoc_factors=["g"], posthoc_methods=["tukey"])
    est = table(sec, "Estimates — g")
    sub = mv_data[mv_data.g == 1.0]["y1"]
    resid_ms = sum(((mv_data[mv_data.g == k]["y1"] - mv_data[mv_data.g == k]["y1"].mean()) ** 2)
                   .sum() for k in (1.0, 2.0, 3.0)) / (len(mv_data) - 3)
    assert num(est.rows[0][2]) == pytest.approx(sub.mean(), abs=1e-3)
    assert num(est.rows[0][3]) == pytest.approx(math.sqrt(resid_ms / len(sub)), abs=1e-3)
    mc = table(sec, "Multiple Comparisons")
    ref = pg.pairwise_tukey(mv_data, dv="y1", between="g")
    assert num(mc.rows[0][6]) == pytest.approx(ref["p_tukey"].iloc[0], abs=2e-3)


# ------------------------------------------------------------------ mixed models
@pytest.fixture
def split_plot():
    rng = np.random.default_rng(5)
    ns, nt = 20, 4
    sid = np.repeat(np.arange(ns), nt)
    tm = np.tile(np.arange(1, nt + 1), ns).astype(float)
    grp = np.repeat(np.where(np.arange(ns) < 10, 1.0, 2.0), nt)
    u = rng.normal(0, 2, ns)[sid]
    slope = rng.normal(0, 0.5, ns)[sid]
    x = rng.normal(0, 1, ns * nt)
    y = 10 + u + (1 + slope) * tm + grp * 1.5 + rng.normal(0, 1, ns * nt) + 0.3 * x
    return pd.DataFrame(dict(id=sid.astype(float), time=tm, grp=grp, y=y, x=x, t=tm))


def test_mixed_balanced_split_plot_equals_anova(split_plot):
    sec = mixed.linear_mixed(Dataset(split_plot), "y", "id", ["grp", "time"])
    ref = pg.mixed_anova(split_plot, dv="y", within="time", between="grp",
                         subject="id").set_index("Source")
    t3 = table(sec, "Type III Tests of Fixed Effects")
    rows = {r[0]: r for r in t3.rows}
    assert num(rows["grp"][3]) == pytest.approx(ref.loc["grp", "F"], abs=1e-3)
    assert num(rows["time"][3]) == pytest.approx(ref.loc["time", "F"], abs=1e-3)
    assert num(rows["grp * time"][3]) == pytest.approx(ref.loc["Interaction", "F"], abs=1e-3)
    assert num(rows["grp"][2]) == pytest.approx(18, abs=1e-3)       # Satterthwaite = exact
    assert num(rows["time"][2]) == pytest.approx(54, abs=1e-3)


def test_mixed_covariance_parameters_closed_form(split_plot):
    sec = mixed.linear_mixed(Dataset(split_plot), "y", "id", ["grp", "time"])
    cov = table(sec, "Estimates of Covariance Parameters").rows
    s2, se_s2 = num(cov[0][2]), num(cov[0][3])
    su, se_su = num(cov[1][2]), num(cov[1][3])
    # balanced REML = ANOVA estimators; asymptotic variances in closed form
    assert se_s2 == pytest.approx(s2 * math.sqrt(2 / 54), abs=2e-3)
    k = 4
    var_su = 2 / k ** 2 * ((s2 + k * su) ** 2 / 18 + s2 ** 2 / 54)
    assert se_su == pytest.approx(math.sqrt(var_su), abs=3e-3)
    ref = smf.mixedlm("y ~ C(grp) * C(time)", split_plot, groups=split_plot["id"]).fit(reml=True)
    assert s2 == pytest.approx(ref.scale, abs=2e-3)
    assert su == pytest.approx(float(ref.cov_re.iloc[0, 0]), abs=2e-3)
    ic = table(sec, "Information Criteria").rows
    assert num(ic[0][1]) == pytest.approx(-2 * ref.llf, abs=2e-3)


def test_mixed_random_slope_matches_statsmodels(split_plot):
    sec = mixed.linear_mixed(Dataset(split_plot), "y", "id", ["grp"], ["t", "x"],
                             random_slopes=["t"], structure="UN")
    ref = smf.mixedlm("y ~ C(grp, Treatment(2.0)) + t + x", split_plot, groups=split_plot["id"],
                      re_formula="~t").fit(reml=True)
    est = {r[0]: r for r in table(sec, "Estimates of Fixed Effects").rows}
    assert num(est["Intercept"][1]) == pytest.approx(ref.fe_params["Intercept"], abs=2e-3)
    assert num(est["[grp=1]"][1]) == pytest.approx(
        ref.fe_params["C(grp, Treatment(2.0))[T.1.0]"], abs=2e-3)
    assert num(est["t"][1]) == pytest.approx(ref.fe_params["t"], abs=2e-3)
    assert est["[grp=2]"][1].startswith("0")
    cov = table(sec, "Estimates of Covariance Parameters").rows
    assert num(cov[1][2]) == pytest.approx(ref.cov_re.iloc[0, 0], abs=5e-3)
    assert num(cov[2][2]) == pytest.approx(ref.cov_re.iloc[1, 0], abs=5e-3)
    ic = table(sec, "Information Criteria").rows
    assert num(ic[0][1]) == pytest.approx(-2 * ref.llf, abs=5e-3)
    # t^2 = F for one-df terms
    t3 = {r[0]: r for r in table(sec, "Type III Tests of Fixed Effects").rows}
    assert num(t3["x"][3]) == pytest.approx(num(est["x"][4]) ** 2, rel=2e-3)
    assert num(ic[1][1]) == pytest.approx(num(ic[0][1]) + 2 * 4, abs=2e-3)  # 4 cov params


def test_mixed_ml_loglik(split_plot):
    sec = mixed.linear_mixed(Dataset(split_plot), "y", "id", ["grp"], ["t"], method="ML")
    ref = smf.mixedlm("y ~ C(grp) + t", split_plot, groups=split_plot["id"]).fit(reml=False)
    ic = table(sec, "Information Criteria").rows
    assert ic[0][0] == "-2 Log Likelihood"
    assert num(ic[0][1]) == pytest.approx(-2 * ref.llf, abs=2e-3)
    assert num(ic[1][1]) == pytest.approx(-2 * ref.llf + 2 * (3 + 2), abs=2e-3)

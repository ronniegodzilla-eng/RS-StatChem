"""WP10: Complex Samples (Taylor linearization as in IBM's Complex Samples algorithms).

Printed reference values:

* Lohr (1999) Example 5.2, one-stage cluster sample of 5 suites (GPA data printed in the book),
  analysed with the R survey package in the SDaA vignette "Reproduction of Analyses in Lohr
  (1999) using the survey package" (T. Verbeke, 2011, p. 9):
  ``svytotal(~gpa, svydesign(ids=~cluster, weights=~pwt))`` -> total 1130.4, SE 67.167
  (Stata: 1130.4, 67.16666).
* Lohr Examples 10.1, 10.2 and 10.4 as printed in the same vignette: Pearson X^2 = 2.281,
  8.2176 and 4.5621 (under simple random sampling CSTABULATE's X^2 is the Pearson statistic and
  the Rao-Scott Delta is n/(n-1) I, so F = X^2 (n-1)/(n df)).

Other references: Cochran's stratified-SRS variance formula, statsmodels cluster-robust
WLS/GLM (no small-sample correction) times G/(G-1), statsmodels OrderedModel with the
HC0 sandwich times n/(n-1), lifelines robust (clustered) Cox without ties times G/(G-1).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import complex_samples as cs

GPA = [3.08, 2.60, 3.44, 3.04, 2.36, 3.04, 3.28, 2.68, 2.00, 2.56, 2.52, 1.88, 3.00, 2.88,
       3.44, 3.64, 2.68, 1.92, 3.28, 3.20]


def _gpa():
    ds = Dataset(pd.DataFrame({"suite": np.repeat(np.arange(1, 6), 4).astype(float),
                               "gpa": GPA, "pwt": 20.0}))
    return ds, cs.Plan("pwt", [cs.Stage([], ["suite"], "WR")])


def _table(sec, title):
    return next(t for t in sec.items if getattr(t, "title", "") == title)


def test_lohr_gpa_total_matches_survey_package():
    ds, plan = _gpa()
    dsg = cs.Design(ds, plan)
    e = cs.total_est(dsg, np.array(GPA), np.ones(20, bool))
    assert e.value == pytest.approx(1130.4)
    assert round(e.se, 3) == 67.167
    assert dsg.df == 4 and dsg.n_psu == 5
    m = cs.ratio_est(dsg, np.array(GPA), np.ones(20), np.ones(20, bool))
    assert m.value == pytest.approx(2.826)
    assert m.se == pytest.approx(e.se / 400, rel=1e-9)   # equal cluster sizes: N-hat fixed
    sec = cs.cs_descriptives(ds, ["gpa"], plan, means=True, sums=True,
                             statistics=("se", "ci", "cv", "count", "deff"))
    t = _table(sec, "Univariate Statistics: Sum")
    assert t.rows[0][1:3] == ["1130.400", "67.167"]
    lo = 1130.4 - 67.16666 * 2.7764451
    assert float(t.rows[0][3]) == pytest.approx(lo, abs=2e-3)   # t(4) interval as Stata


@pytest.mark.parametrize("table, chi2", [([[119, 188], [88, 105]], 2.281),
                                         ([[46, 222], [41, 109], [17, 40], [8, 26]], 8.2176),
                                         ([[238, 376], [176, 210]], 4.5621)])
def test_lohr_chapter10_pearson_statistics(table, chi2):
    rows, cols = [], []
    for i, r in enumerate(table):
        for j, c in enumerate(r):
            rows += [i + 1.0] * c
            cols += [j + 1.0] * c
    n = len(rows)
    ds = Dataset(pd.DataFrame({"r": rows, "c": cols}))
    plan = cs.Plan(None, [cs.Stage()], srs_estimator="WR")
    dsg = cs.Design(ds, plan)
    RI = np.column_stack([(np.array(rows) == i + 1).astype(float) for i in range(len(table))])
    CI = np.column_stack([(np.array(cols) == j + 1).astype(float) for j in range(2)])
    res = cs.independence_tests(dsg, RI, CI, np.ones(n, bool))
    X2, F, d1, d2, p = res["Pearson"]
    df = len(table) - 1
    assert X2 == pytest.approx(chi2, abs=6e-4)
    assert d1 == pytest.approx(df, abs=1e-8)
    assert F == pytest.approx(X2 * (n - 1) / (n * df), rel=1e-8)
    sec = cs.cs_crosstabs(ds, "r", "c", plan, cells=("popsize", "row", "column", "table"),
                          statistics=("se", "ci", "cv", "deff", "count"), odds_ratio=True,
                          residuals=True)
    t = _table(sec, "Tests of Independence")
    assert t.rows[0][1] == f"{chi2:.3f}".lstrip("0")


def test_stratified_srswor_mean_matches_cochran():
    rng = np.random.default_rng(2)
    st = np.repeat([0, 1, 2], [20, 30, 25])
    y = rng.normal(10 + st, 2)
    Nh = {0: 200, 1: 900, 2: 400}
    nh = {h: int((st == h).sum()) for h in Nh}
    df = pd.DataFrame({"st": st.astype(float), "y": y, "N": [Nh[h] for h in st],
                       "w": [Nh[h] / nh[h] for h in st]})
    ds = Dataset(df)
    dsg = cs.Design(ds, cs.Plan("w", [cs.Stage(["st"], [], "EQUAL_WOR", "popsize", None,
                                               "N")]))
    e = cs.ratio_est(dsg, y, np.ones(len(y)), np.ones(len(y), bool))
    N = sum(Nh.values())
    mean = sum(Nh[h] / N * y[st == h].mean() for h in Nh)
    var = sum((Nh[h] / N) ** 2 * (1 - nh[h] / Nh[h]) * y[st == h].var(ddof=1) / nh[h]
              for h in Nh)
    assert e.value == pytest.approx(mean, rel=1e-12)
    assert e.se == pytest.approx(math.sqrt(var), rel=1e-10)
    assert dsg.df == len(y) - 3
    # sampling rate given as a value gives the same FPC as the population sizes here
    dsg2 = cs.Design(ds, cs.Plan("w", [cs.Stage(["st"], [], "WR")]))
    e2 = cs.ratio_est(dsg2, y, np.ones(len(y)), np.ones(len(y), bool))
    assert e2.se > e.se


def test_design_effect_srs_is_one_and_domain_estimation():
    rng = np.random.default_rng(4)
    y = rng.normal(size=80)
    g = rng.integers(0, 2, 80)
    ds = Dataset(pd.DataFrame({"y": y, "g": g.astype(float)}))
    plan = cs.Plan(None, [cs.Stage()], srs_estimator="WR")
    dsg = cs.Design(ds, plan)
    e = cs.ratio_est(dsg, y, np.ones(80), np.ones(80, bool))
    assert e.deff == pytest.approx(1.0, rel=1e-12)
    assert e.se == pytest.approx(y.std(ddof=1) / math.sqrt(80), rel=1e-12)
    dom = g == 1
    ed = cs.ratio_est(dsg, y, np.ones(80), dom)
    assert ed.value == pytest.approx(y[dom].mean())
    nd = dom.sum()
    # SRS domain mean: var = sum_dom (y - ybar_d)^2 * n/(n-1) / n_d^2
    v = np.sum((y[dom] - y[dom].mean()) ** 2) * 80 / 79 / nd ** 2
    assert ed.se == pytest.approx(math.sqrt(v), rel=1e-12)
    sec = cs.cs_descriptives(ds, ["y"], plan, subpops=["g"], statistics=("se",),
                             test_value=0.0)
    assert len(_table(sec, "Univariate Statistics: Mean").rows) == 3


def test_two_stage_variance_formula():
    rng = np.random.default_rng(6)
    psu = np.repeat(np.arange(6), 5)
    ssu = np.arange(30)
    y = rng.normal(size=30) + psu
    f1, f2 = 0.3, 0.25
    w = np.full(30, 1 / (f1 * f2))
    ds = Dataset(pd.DataFrame({"psu": psu.astype(float), "ssu": ssu.astype(float), "y": y,
                               "w": w}))
    plan = cs.Plan("w", [cs.Stage([], ["psu"], "EQUAL_WOR", "rate", f1),
                         cs.Stage([], [], "EQUAL_WOR", "rate", f2)])
    dsg = cs.Design(ds, plan)
    U = w * y
    T = np.array([U[psu == i].sum() for i in range(6)])
    v1 = (1 - f1) * 6 / 5 * np.sum((T - T.mean()) ** 2)
    v2 = 0.0
    for i in range(6):
        u = U[psu == i]
        v2 += f1 * (1 - f2) * 5 / 4 * np.sum((u - u.mean()) ** 2)
    assert dsg.vcov(U)[0, 0] == pytest.approx(v1 + v2, rel=1e-12)


def test_unequal_wor_yates_grundy():
    p = np.array([0.2, 0.3, 0.5, 0.6])
    y = np.array([4.0, 6.0, 9.0, 13.0])
    ds = Dataset(pd.DataFrame({"y": y, "pi": p, "w": 1 / p}))
    dsg = cs.Design(ds, cs.Plan("w", [cs.Stage([], [], "UNEQUAL_WOR", "inclusion", None,
                                               "pi")]))
    v = dsg.vcov(y / p)[0, 0]
    c = (4 - 1) / (4 - p)
    ref = 0.0
    for i in range(4):
        for j in range(i + 1, 4):
            pij = p[i] * p[j] * (c[i] + c[j]) / 2
            ref += (p[i] * p[j] - pij) / pij * (y[i] / p[i] - y[j] / p[j]) ** 2
    assert v == pytest.approx(ref)


def _model_data():
    rng = np.random.default_rng(5)
    n = 240
    g = np.repeat(np.arange(40), 6)
    x1 = rng.normal(size=n) + 0.3 * rng.normal(size=40)[g]
    y = 1 + 0.5 * x1 + rng.normal(size=n) + rng.normal(size=40)[g]
    yb = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.3 + 0.8 * x1)))).astype(float)
    w = rng.uniform(1, 3, n)
    yo = np.digitize(y, np.quantile(y, [0.3, 0.7])).astype(float)
    T = rng.exponential(1 / np.exp(0.5 * x1)) + 0.1
    E = (rng.uniform(size=n) < 0.8).astype(float)
    df = pd.DataFrame(dict(g=g.astype(float), x1=x1, y=y, yb=yb, w=w, yo=yo, T=T, E=E,
                           f=(g % 3).astype(float)))
    return Dataset(df), df


def _params(sec):
    t = _table(sec, "Parameter Estimates")
    return np.array([float(r[1]) for r in t.rows if r[2]]), \
        np.array([float(r[2]) for r in t.rows if r[2]])


def test_glm_matches_cluster_robust_wls():
    ds, df = _model_data()
    plan = cs.Plan("w", [cs.Stage([], ["g"], "WR")])
    r = cs.cs_glm(ds, "y", [], ["x1"], plan)
    m = sm.WLS(df.y, sm.add_constant(df.x1), weights=df.w).fit(
        cov_type="cluster", cov_kwds={"groups": df.g, "use_correction": False})
    b, se = _params(r.section)
    assert np.allclose(b, m.params, atol=6e-4)
    assert np.allclose(se, np.sqrt(np.diag(m.cov_params()) * 40 / 39), atol=6e-4)
    # one-df Wald F equals t^2 (Fellegi F with r = 1)
    eff = _table(r.section, "Tests of Model Effects")
    row = next(x for x in eff.rows if x[0] == "x1")
    se_exact = math.sqrt(m.cov_params().iloc[1, 1] * 40 / 39)
    assert float(row[1]) == pytest.approx((m.params.iloc[1] / se_exact) ** 2, rel=1e-4)
    assert row[2:4] == ["1", "39"]
    r = cs.cs_glm(ds, "y", ["f"], ["x1"], plan, test="ADJF")
    assert _table(r.section, "Tests of Model Effects").rows[0][0] == "Corrected Model"


def test_logistic_matches_cluster_robust_glm():
    ds, df = _model_data()
    plan = cs.Plan(None, [cs.Stage([], ["g"], "WR")])
    r = cs.cs_logistic(ds, "yb", [], ["x1"], plan, reference="first")
    gl = sm.GLM(df.yb, sm.add_constant(df.x1), family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": df.g, "use_correction": False})
    b, se = _params(r.section)
    assert np.allclose(b, gl.params, atol=6e-4)
    assert np.allclose(se, np.sqrt(np.diag(gl.cov_params()) * 40 / 39), atol=6e-4)
    # SPSS default reference = highest category: signs flip
    b2, _ = _params(cs.cs_logistic(ds, "yb", [], ["x1"], plan).section)
    assert np.allclose(b2, -b, atol=2e-3)
    r = cs.cs_logistic(ds, "yo", ["f"], ["x1"], plan)      # multinomial (3 categories)
    # 2 non-reference categories x (intercept, x1, [f=0], [f=1], [f=2] redundant)
    assert len(_table(r.section, "Parameter Estimates").rows) == 2 * 5


def test_ordinal_matches_statsmodels_ordered_model():
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    ds, df = _model_data()
    plan = cs.Plan(None, [cs.Stage()])
    r = cs.cs_ordinal(ds, "yo", [], ["x1"], plan)
    om = OrderedModel(df.yo.to_numpy(), df[["x1"]].to_numpy(), distr="logit").fit(
        method="bfgs", disp=False, gtol=1e-10)
    H = om.model.hessian(om.params)
    S = om.model.score_obs(om.params)
    Hi = np.linalg.inv(H)
    n = len(df)
    V = Hi @ (S.T @ S) @ Hi * n / (n - 1)
    b, se = _params(r.section)
    th1, inc = om.params[1], om.params[2]
    assert b[0] == pytest.approx(th1, abs=6e-4)
    assert b[1] == pytest.approx(th1 + math.exp(inc), abs=6e-4)
    assert b[2] == pytest.approx(om.params[0], abs=6e-4)
    assert se[0] == pytest.approx(math.sqrt(V[1, 1]), abs=6e-4)
    assert se[2] == pytest.approx(math.sqrt(V[0, 0]), abs=6e-4)
    for link in ("probit", "cloglog", "nloglog", "cauchit"):
        cs.cs_ordinal(ds, "yo", [], ["x1"], plan, link=link)


def test_cox_matches_lifelines_robust():
    lifelines = pytest.importorskip("lifelines")
    ds, df = _model_data()
    plan = cs.Plan("w", [cs.Stage([], ["g"], "WR")])
    r = cs.cs_cox(ds, "T", "E", [1.0], [], ["x1"], plan)
    cph = lifelines.CoxPHFitter().fit(df[["T", "E", "x1", "w", "g"]], "T", "E",
                                      weights_col="w", cluster_col="g", robust=True)
    b, se = _params(r.section)
    assert b[0] == pytest.approx(cph.params_.iloc[0], abs=6e-4)
    assert se[0] == pytest.approx(cph.standard_errors_.iloc[0] * math.sqrt(40 / 39), abs=6e-4)
    r = cs.cs_cox(ds, "T", "E", [1.0], ["f"], ["x1"], plan, ties="breslow")
    assert len(_table(r.section, "Parameter Estimates").rows) == 4


def test_score_residuals_sum_to_score_with_ties():
    rng = np.random.default_rng(1)
    T = np.round(rng.exponential(1, 60), 1) + 0.1
    E = (rng.uniform(size=60) < 0.7).astype(float)
    X = rng.normal(size=(60, 2))
    w = rng.uniform(1, 2, 60)
    for ties in ("efron", "breslow"):
        b, I, R, ll = cs.cox_fit(T, E, X, w, ties)
        assert np.allclose((w[:, None] * R).sum(axis=0), 0, atol=1e-8)


def test_wald_tests_variants():
    b = np.array([0.5, -0.2])
    V = np.array([[0.04, 0.01], [0.01, 0.09]])
    L = np.eye(2)
    X2 = b @ np.linalg.inv(V) @ b
    F, d1, d2, p = cs.wald_tests(L, b, V, V, 20, "F")
    assert F == pytest.approx((20 - 2 + 1) / (2 * 20) * X2) and (d1, d2) == (2, 19)
    c, d1, _, _ = cs.wald_tests(L, b, V, V, 20, "CHISQUARE")
    assert c == pytest.approx(X2) and d1 == 2
    a, d, d2, _ = cs.wald_tests(L, b, V, V, 20, "ADJF")   # Vsrs = V: Delta = I
    assert a == pytest.approx(X2 / 2) and d == pytest.approx(2) and d2 == pytest.approx(40)


def test_frequencies_ratios_and_plan_roundtrip():
    ds, plan = _gpa()
    ds = Dataset(ds.df.assign(hi=(np.array(GPA) > 2.8).astype(float)))
    sec = cs.cs_frequencies(ds, ["hi"], plan, statistics=("se", "ci", "deff", "count"),
                            cumulative=True)
    t = _table(sec, "hi")
    pct = [float(r[2]) for r in t.rows if r[0] == "% of Total" and r[1] != "Total"]
    assert sum(pct) == pytest.approx(100.0, abs=0.11)
    sec = cs.cs_ratios(ds, ["gpa"], ["pwt"], plan, population_size=True, test_value=0.1)
    assert _table(sec, "Ratios").rows[0][2] == ".1413"
    cs.set_plan(ds, plan)
    assert cs.get_plan(ds).to_dict() == plan.to_dict()
    sec = cs.prepare_plan(ds, plan)
    assert _table(sec, "Plan Summary").rows[0][1] == "pwt"


def test_sampling_wizard():
    df = pd.DataFrame({"st": np.repeat([1.0, 2.0], [40, 60]), "size": np.arange(1, 101.0),
                       "cl": np.repeat(np.arange(25), 4).astype(float)})
    ds = Dataset(df.copy())
    cs.select_sample(ds, "srswor", ["st"], [], 10, "count", seed=1)
    wcol = ds.df["SampleWeight_Final_"]
    sel = wcol.notna()
    assert sel.sum() == 20
    assert wcol[sel & (ds.df.st == 1)].iloc[0] == pytest.approx(4.0)
    assert wcol[sel].sum() == pytest.approx(100.0)
    plan = cs.get_plan(ds)
    assert plan.weight == "SampleWeight_Final_" and plan.stages[0].method == "EQUAL_WOR"
    ds2 = Dataset(df.copy())
    cs.select_sample(ds2, "pps_sys", [], ["cl"], 5, "count", measure="size", seed=3)
    assert ds2.df["InclusionProbability_1_"].notna().sum() == 20      # 5 clusters x 4
    ds3 = Dataset(df.copy())
    cs.select_sample(ds3, "srswr", ["st"], [], 0.2, "proportion", seed=2)
    assert cs.get_plan(ds3).stages[0].method == "WR"
    cs.cs_descriptives(ds3, ["size"], None, sums=True)


def test_no_plan_raises():
    with pytest.raises(ValueError):
        cs.cs_descriptives(Dataset(pd.DataFrame({"y": [1.0, 2.0]})), ["y"])


def test_selftest():
    cs.selftest()

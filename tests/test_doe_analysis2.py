"""WP-B analysis extensions of the RSM / DOE module, checked against published examples and
statsmodels / scipy:

* E2 chemical conversion CCD (Myers & Montgomery; Stat-Ease RSM tutorial data): quadratic
  coefficients and fit statistics;
* E3 Montgomery Example 6.2 filtration rate: effects, Lenth PSE / limits, reduced model;
* E4 Montgomery §6.8 curvature test (2² + 5 centre points);
* E6 Cornell yarn elongation: real / U_Pseudo equations;
* E15 logistic and Poisson regression against statsmodels GLM;
* logit / arcsine square-root transforms, higher-order and Slack / quartic mixture models,
  combined mixture-process models, several analyses per response, Durbin-Watson, covariance
  ratio, treatment means (LSD / Tukey) and the new analysis UI.
"""
import math
import os

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.stats.stattools import durbin_watson as sm_dw

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.doe import glm, model, report

SHOTS = os.environ.get("RSSTATCHEM_DX2B_SHOTS")


# ---------------------------------------------------------------------------- data
def conversion_ccd():
    """E2: chemical conversion CCD, time 40-50 min, temperature 80-90 °C, catalyst 2-3 %,
    alpha 1.682, six centre points (conversion and activity)."""
    a = 8 ** 0.25  # rotatable alpha 1.68179 (Stat-Ease)
    fact = [((-1, -1, -1), 74, 53.2), ((1, -1, -1), 51, 62.9), ((-1, 1, -1), 88, 53.4),
            ((1, 1, -1), 70, 62.6), ((-1, -1, 1), 71, 57.3), ((1, -1, 1), 90, 67.9),
            ((-1, 1, 1), 66, 59.8), ((1, 1, 1), 97, 67.8)]
    axial = [((-a, 0, 0), 76, 59.1), ((a, 0, 0), 79, 65.9), ((0, -a, 0), 85, 60.0),
             ((0, a, 0), 97, 60.7), ((0, 0, -a), 55, 57.4), ((0, 0, a), 81, 63.2)]
    centre = [((0, 0, 0), c, act) for c, act in zip([81, 75, 76, 83, 80, 91],
                                                    [59.2, 60.4, 59.1, 60.6, 60.8, 58.9])]
    rows = fact + axial + centre
    X = np.array([r[0] for r in rows], float)
    df = pd.DataFrame({"Time": 45 + 5 * X[:, 0], "Temp": 85 + 5 * X[:, 1],
                       "Cat": 2.5 + 0.5 * X[:, 2], "Conv": [r[1] for r in rows],
                       "Act": [r[2] for r in rows]})
    ds = Dataset(df)
    ds.meta["doe"] = {"kind": "process", "design": "ccd",
                      "factors": [{"name": "time", "column": "Time", "low": 40, "high": 50},
                                  {"name": "temperature", "column": "Temp", "low": 80,
                                   "high": 90},
                                  {"name": "catalyst", "column": "Cat", "low": 2, "high": 3}],
                      "responses": [{"name": "Conversion", "column": "Conv"},
                                    {"name": "Activity", "column": "Act"}]}
    return ds, X


FILTRATION = [45, 71, 48, 65, 68, 60, 80, 65, 43, 100, 45, 104, 75, 86, 70, 96]


def filtration_ds():
    X = np.array([[(-1) ** (1 + (i >> b & 1)) for b in range(4)] for i in range(16)], float)
    df = pd.DataFrame({c: 1.0 * X[:, j] for j, c in enumerate("ABCD")})
    df["Rate"] = FILTRATION
    ds = Dataset(df)
    ds.meta["doe"] = {"kind": "process", "design": "factorial",
                      "factors": [{"name": n, "column": n, "low": -1, "high": 1}
                                  for n in "ABCD"],
                      "responses": [{"name": "Filtration rate", "column": "Rate"}]}
    return ds


def curvature_ds():
    """E4: 2² factorial with five centre points (Montgomery §6.8)."""
    pts = [(-1, -1, 39.3), (1, -1, 40.9), (-1, 1, 40.0), (1, 1, 41.5)] + \
        [(0, 0, v) for v in (40.3, 40.5, 40.7, 40.2, 40.6)]
    df = pd.DataFrame(pts, columns=["A", "B", "Y"])
    ds = Dataset(df)
    ds.meta["doe"] = {"kind": "process", "design": "factorial",
                      "factors": [{"name": "Time", "column": "A", "low": -1, "high": 1},
                                  {"name": "Temp", "column": "B", "low": -1, "high": 1}],
                      "responses": [{"name": "Yield", "column": "Y"}]}
    return ds


def yarn_ds():
    pts = [((1, 0, 0), [11.0, 12.4]), ((0.5, 0.5, 0), [15.0, 14.8, 16.1]),
           ((0, 1, 0), [8.8, 10.0]), ((0, 0.5, 0.5), [10.0, 9.7, 11.8]),
           ((0, 0, 1), [16.8, 16.0]), ((0.5, 0, 0.5), [17.7, 16.4, 16.6])]
    rows = [(*x, v) for x, vals in pts for v in vals]
    ds = Dataset(pd.DataFrame(rows, columns=["PE", "PS", "PP", "Elong"]))
    ds.meta["doe"] = {"kind": "mixture", "mixture": {"total": 1.0},
                      "factors": [{"name": n, "column": n, "type": "mixture", "low": 0,
                                   "high": 1} for n in ("PE", "PS", "PP")],
                      "responses": [{"name": "Elongation", "column": "Elong"}]}
    return ds


def factorial_glm_ds(seed=3, reps=6, kind="binary"):
    """2³ factorial with replicates and a binary / binomial / count response."""
    rng = np.random.default_rng(seed)
    base = np.array([[(-1) ** (1 + (i >> b & 1)) for b in range(3)] for i in range(8)], float)
    X = np.repeat(base, reps, axis=0)
    eta = 0.2 + 0.9 * X[:, 0] - 0.6 * X[:, 1] + 0.3 * X[:, 2] + 0.4 * X[:, 0] * X[:, 1]
    df = pd.DataFrame({c: X[:, j] for j, c in enumerate("ABC")})
    if kind == "binary":
        df["Y"] = (rng.random(len(X)) < 1 / (1 + np.exp(-eta))).astype(float)
    elif kind == "binomial":
        df["Y"] = rng.binomial(20, 1 / (1 + np.exp(-eta))).astype(float)
        df["N"] = 20.0
    else:
        df["Y"] = rng.poisson(np.exp(1.5 + 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.2 * X[:, 2]))
    ds = Dataset(df)
    ds.meta["doe"] = {"kind": "process", "design": "factorial",
                      "factors": [{"name": n, "column": n, "low": -1, "high": 1}
                                  for n in "ABC"],
                      "responses": [{"name": "Y", "column": "Y"}]}
    return ds


def combined_ds(noise=0.0, seed=1):
    """Combined design: simplex lattice {3, 2} plus centroid (7 blends) x 2² process
    factors (28 runs); response from a known Quadratic x Linear crossed model."""
    blends = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (.5, .5, 0), (.5, 0, .5), (0, .5, .5),
              (1 / 3, 1 / 3, 1 / 3)]
    rows = []
    for b in blends:
        for p1 in (-1, 1):
            for p2 in (-1, 1):
                rows.append((*b, p1, p2))
    A = np.array(rows, float)
    x1, x2, x3, d, e = A.T
    y = (10 * x1 + 6 * x2 + 8 * x3 + 12 * x1 * x2 - 5 * x1 * x3 + 4 * x2 * x3
         + d * (2 * x1 - 1 * x2 + 0.5 * x3) + e * (1.5 * x1 + 0.7 * x2 - 0.8 * x3))
    y = y + noise * np.random.default_rng(seed).normal(size=len(y))
    df = pd.DataFrame({"M1": x1, "M2": x2, "M3": x3, "Temp": 50 + 10 * d, "Time": 30 + 5 * e,
                       "Y": y})
    ds = Dataset(df)
    ds.meta["doe"] = {"kind": "combined", "design": "combined", "mixture": {"total": 1.0},
                      "factors": [{"name": "M1", "column": "M1", "type": "mixture", "low": 0,
                                   "high": 1},
                                  {"name": "M2", "column": "M2", "type": "mixture", "low": 0,
                                   "high": 1},
                                  {"name": "M3", "column": "M3", "type": "mixture", "low": 0,
                                   "high": 1},
                                  {"name": "Temp", "column": "Temp", "low": 40, "high": 60},
                                  {"name": "Time", "column": "Time", "low": 25, "high": 35}],
                      "responses": [{"name": "Y", "column": "Y"}]}
    return ds


def terms_of(*names):
    letters = "ABCDEFGH"
    return [tuple(sorted(letters.index(ch) for ch in n)) for n in names]


# ---------------------------------------------------------------------------- E2
def test_e2_conversion_ccd_quadratic_coefficients():
    ds, X = conversion_ccd()
    a = model.analyze(ds, model.AnalysisSpec("Conv", model="quadratic"))
    coef = {c: v for v, c in a.fit.coded_equation()}
    expected = {"": 81.091, "A": 1.028, "B": 4.040, "C": 6.204, "AB": 2.125, "AC": 11.375,
                "BC": -3.875, "A²": -1.834, "B²": 2.938, "C²": -5.191}
    for k, v in expected.items():  # the report's values used alpha = 1.682
        assert coef[k] == pytest.approx(v, abs=2e-3), k
    st = a.fit.fit_statistics()
    assert st["r2"] == pytest.approx(0.9199, abs=1e-4)
    assert st["adj_r2"] == pytest.approx(0.8478, abs=1e-4)
    assert st["pred_r2"] == pytest.approx(0.7566, abs=1e-4)
    assert st["press"] == pytest.approx(676.28, abs=0.1)
    assert st["std_dev"] == pytest.approx(4.717, abs=1e-3)
    # Stat-Ease tutorial: B = 4.04057, AC = 11.375
    assert coef["B"] == pytest.approx(4.04057, abs=1e-5)
    # activity is linear: 60.51 + 3.583 A + 0.255 B + 2.23 C
    act = model.analyze(ds, model.AnalysisSpec("Act", model="linear"))
    c2 = {c: v for v, c in act.fit.coded_equation()}
    assert (c2[""], c2["A"], c2["B"], c2["C"]) == pytest.approx((60.51, 3.583, 0.255, 2.23),
                                                                abs=6e-3)
    # the extra fit statistics agree with statsmodels
    ols = sm.OLS(ds.df["Conv"].to_numpy(float), a.fit.mat.X).fit()
    assert a.fit.durbin_watson() == pytest.approx(sm_dw(ols.resid))
    assert a.fit.information()["m2ll"] == pytest.approx(-2 * ols.llf)
    infl = OLSInfluence(ols)
    assert np.allclose(a.fit.diagnostics()["covratio"], infl.cov_ratio)


def test_e2_fit_statistics_table_reports_dw_and_information():
    ds, _ = conversion_ccd()
    a = model.analyze(ds, model.AnalysisSpec("Conv", model="quadratic"))
    t, _ = report.fit_statistics_table(a.fit)
    cells = {r[i]: r[i + 1] for r in t.rows for i in (0, 2)}
    assert cells["R²"] == "0.9199" and "Durbin-Watson" in cells and "AICc" in cells


# ---------------------------------------------------------------------------- E3
def test_e3_filtration_effects_lenth_and_reduced_model():
    ds = filtration_ds()
    a = model.analyze(ds, model.AnalysisSpec("Rate", model="linear"))
    eff = {e["name"]: e["effect"] for e in model.factorial_effects(a.space, a.obs)}
    expected = {"A": 21.625, "B": 3.125, "C": 9.875, "D": 14.625, "AB": 0.125, "AC": -18.125,
                "AD": 16.625, "BC": 2.375, "BD": -0.375, "CD": -1.125, "ABC": 1.875,
                "ABD": 4.125, "ACD": -1.625, "BCD": -2.625, "ABCD": 1.375}
    for k, v in expected.items():
        assert eff[k] == pytest.approx(v, abs=1e-9), k
    e0 = model.factorial_effects(a.space, a.obs)[0]
    assert e0["pse"] == pytest.approx(2.625)
    assert e0["t_limit"] * e0["pse"] == pytest.approx(6.748, abs=1e-3)
    assert e0["bonferroni"] == pytest.approx(5.2474, abs=1e-4)
    assert e0["bonferroni"] * e0["pse"] == pytest.approx(13.77, abs=0.01)
    spec = model.AnalysisSpec("Rate", model="manual",
                              terms=[model.term_key(t) for t in terms_of("A", "C", "D", "AC",
                                                                          "AD")])
    fit = model.analyze(ds, spec).fit
    coef = {c: v for v, c in fit.coded_equation()}
    assert [coef[k] for k in ("", "A", "C", "D", "AC", "AD")] == pytest.approx(
        [70.0625, 10.8125, 4.9375, 7.3125, -9.0625, 8.3125])
    ss = {r[0].split("-")[0]: r[1] for r in fit.anova()["terms"]}
    assert [ss[k] for k in ("A", "C", "D", "AC", "AD")] == pytest.approx(
        [1870.5625, 390.0625, 855.5625, 1314.0625, 1105.5625])
    assert fit.anova()["residual"] == pytest.approx((195.125, 10))
    # Shapiro-Wilk of the effects left for error (scipy) and the normal effect scores
    sel = {"A", "C", "D", "AC", "AD"}
    effs = model.factorial_effects(a.space, a.obs)
    sw = model.effects_shapiro(effs, sel)
    w, p = stats.shapiro([e["effect"] for e in effs if e["name"] not in sel])
    assert (sw["w"], sw["p"], sw["n"]) == pytest.approx((w, p, 10))
    vals, q, names = model.normal_effect_scores(effs)
    assert names[0] == "AC" and names[-1] == "A" and np.all(np.diff(q) > 0)


# ---------------------------------------------------------------------------- E4
def test_e4_curvature_test():
    ds = curvature_ds()
    fit = model.analyze(ds, model.AnalysisSpec("Y", model="2fi")).fit
    an = fit.anova()
    ss = {r[0].split("-")[0]: r[1] for r in an["terms"]}
    assert (ss["A"], ss["B"], ss["AB"]) == pytest.approx((2.4025, 0.4225, 0.0025))
    c = an["curvature"]
    nf, nc, yf, yc = 4, 5, np.mean([39.3, 40.9, 40.0, 41.5]), np.mean([40.3, 40.5, 40.7, 40.2,
                                                                       40.6])
    assert c["ss"] == pytest.approx(nf * nc * (yf - yc) ** 2 / (nf + nc))
    assert c["ss"] == pytest.approx(0.00272, abs=1e-5)
    assert an["pure_error"] == pytest.approx((0.172, 4))
    assert an["residual"][1] == 4
    assert (c["f"], c["p"]) == pytest.approx((0.0633, 0.8137), abs=1e-4)
    assert c["estimate"] == pytest.approx(yc - yf)
    # predictions use the factorial model (no curvature)
    assert fit.predict(np.array([[0.0, 0.0]]))[0] == pytest.approx(yf)
    t, interp = report.anova_table(fit, "2FI")
    row = next(r for r in t.rows if r[0] == "Curvature")
    assert row[5] == "0.8137" and row[6] == "not significant"
    assert any("curvature" in s for s in interp)
    # pooled into the residual on request; not used with a quadratic term
    pooled = model.analyze(ds, model.AnalysisSpec("Y", model="2fi", curvature="residual")).fit
    assert pooled.anova()["curvature"] is None and pooled.df_res == 5
    assert pooled.ss_res == pytest.approx(0.172 + c["ss"])
    # Analysis Summary shows the curvature p-value of the stored analysis
    model.save_spec(ds, model.AnalysisSpec("Y", model="2fi"))
    tab = report.analysis_summary_table(ds)
    assert tab.rows[0][6] == "0.8137"


def test_curvature_not_applied_to_ccd():
    ds, _ = conversion_ccd()
    fit = model.analyze(ds, model.AnalysisSpec("Conv", model="2fi")).fit
    assert fit.anova()["curvature"] is None


# ---------------------------------------------------------------------------- transforms
def test_logit_and_arcsine_sqrt_transforms():
    tr = model.Transform("logit", lower=0, upper=100)
    y = np.array([5.0, 20.0, 50.0, 95.0])
    z = tr.apply(y)
    assert np.allclose(z, np.log(y / (100 - y)))
    assert np.allclose(tr.inverse(z), y)
    assert tr.check(np.array([0.0, 50])) is not None and tr.check(y) is None
    assert tr.label("Yield") == "Logit(Yield)" and "L = 0, U = 100" in tr.describe()
    assert model.Transform.from_meta(tr.to_meta()) == tr
    tr2 = model.Transform("asin_sqrt")
    p = np.array([0.0, 0.1, 0.5, 1.0])
    assert np.allclose(tr2.apply(p), np.arcsin(np.sqrt(p)))
    assert np.allclose(tr2.inverse(tr2.apply(p)), p)
    assert tr2.power is None and model.Transform.from_meta({"kind": "none"}).upper == 1.0
    # an analysis on the logit scale back-transforms its predictions into the bounds
    ds, X = conversion_ccd()
    spec = model.AnalysisSpec("Conv", model="quadratic",
                              transform=model.Transform("logit", lower=0, upper=100))
    a = model.analyze(ds, spec)
    z = np.log(ds.df["Conv"] / (100 - ds.df["Conv"]))
    ols = sm.OLS(z.to_numpy(), a.fit.mat.X).fit()
    assert np.allclose(a.fit.beta, ols.params)
    pred = a.fit.predict(np.array([[45, 85, 2.5]]))
    assert 0 < pred[0] < 100
    iv = a.fit.interval(np.array([[45, 85, 2.5]]))
    assert iv["ci"][0][0] < pred[0] < iv["ci"][1][0]


# ---------------------------------------------------------------------------- higher orders
def test_higher_process_orders_and_add_term():
    sp = model.Space([model.FactorDef(n, n, index=j) for j, n in enumerate("AB")])
    assert len(model.process_terms(2, "quartic")) == 14   # monomials of degree 1..4 in 2
    assert len(model.process_terms(2, "sixth")) == 27
    assert set(model.process_terms(2, "cubic")) <= set(model.process_terms(2, "quartic"))
    assert [o for _n, o in model.all_model_orders(sp)][-3:] == ["quartic", "fifth", "sixth"]
    assert model.parse_term("A2B", sp) == (0, 0, 1)
    assert model.parse_term("a^3", sp) == (0, 0, 0)
    assert model.parse_term("A²B²", sp) == (0, 0, 1, 1)
    with pytest.raises(ValueError):
        model.parse_term("AZ", sp)
    assert model.term_name((0, 0, 0, 1)) == "A³B"
    # a quartic in one factor on 7 levels is fitted exactly
    x = np.linspace(-1, 1, 7)
    y = 1 + x - 2 * x ** 2 + 0.5 * x ** 3 + 3 * x ** 4
    ds = Dataset(pd.DataFrame({"A": np.repeat(x, 2), "Y": np.repeat(y, 2)
                               + np.tile([0.01, -0.01], 7)}))
    ds.meta["doe"] = {"kind": "process", "factors": [{"name": "A", "column": "A", "low": -1,
                                                       "high": 1}],
                      "responses": [{"name": "Y", "column": "Y"}]}
    fit = model.analyze(ds, model.AnalysisSpec("Y", model="quartic")).fit
    coef = {c: v for v, c in fit.coded_equation()}
    assert [coef[k] for k in ("", "A", "A²", "A³", "A⁴")] == pytest.approx(
        [1, 1, -2, 0.5, 3], abs=1e-6)
    assert report.model_label(fit.space, fit.terms) == "Quartic"


# ---------------------------------------------------------------------------- mixtures
def test_mixture_quartic_terms_and_exact_fit():
    assert len(model.mixture_terms(3, "special_quartic")) == 3 + 3 + 3
    assert len(model.mixture_terms(3, "quartic")) == 3 + 3 + 1 + 3 + 3 + 3
    t = (-2, 0, 1)
    assert model.term_name(t) == "AB(A-B)²" and model.key_term(model.term_key(t)) == t
    assert model.term_degree(t) == 4
    # data from a full quartic Scheffé polynomial are reproduced
    rng = np.random.default_rng(0)
    P = np.vstack([rng.dirichlet(np.ones(3), 30), np.eye(3)])
    x1, x2, x3 = P.T
    y = (3 * x1 + 5 * x2 + 7 * x3 + 4 * x1 * x2 - 2 * x2 * x3 + 6 * x1 * x2 * (x1 - x2)
         + 9 * x1 * x2 * (x1 - x2) ** 2 + 11 * x1 ** 2 * x2 * x3)
    ds = Dataset(pd.DataFrame({"A": x1, "B": x2, "C": x3, "Y": y}))
    ds.meta["doe"] = {"kind": "mixture", "mixture": {"total": 1.0},
                      "factors": [{"name": n, "column": n, "type": "mixture", "low": 0,
                                   "high": 1} for n in "ABC"],
                      "responses": [{"name": "Y", "column": "Y"}]}
    fit = model.analyze(ds, model.AnalysisSpec("Y", model="quartic")).fit
    coef = {c: v for v, c in fit.coded_equation()}
    assert coef["AB(A-B)²"] == pytest.approx(9, abs=1e-6)
    assert coef["A²BC"] == pytest.approx(11, abs=1e-6)
    assert coef["AB(A-B)"] == pytest.approx(6, abs=1e-6)


def test_slack_model_matches_scheffe_quadratic():
    ds = yarn_ds()
    sch = model.analyze(ds, model.AnalysisSpec("Elong", model="quadratic")).fit
    slack = model.analyze(ds, model.AnalysisSpec("Elong", model="quadratic", slack=2)).fit
    assert slack.space.slack == 2 and slack.mat.intercept
    assert all(2 not in model.term_factors(t) for t in slack.terms)
    assert np.allclose(sch.fitted, slack.fitted)
    assert slack.ss_res == pytest.approx(sch.ss_res)
    pts = np.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]])
    assert np.allclose(sch.predict(pts), slack.predict(pts))
    assert slack.actual_equations()  # equation in actual components (PP left out)
    assert report.model_label(slack.space, slack.terms) == "Quadratic"
    # Slack is remembered with the analysis
    spec = model.AnalysisSpec.from_meta("Elong", model.AnalysisSpec(
        "Elong", slack=2).to_meta())
    assert spec.slack == 2


def test_mixture_real_and_upseudo_equations():
    ds = yarn_ds()
    for f in ds.meta["doe"]["factors"]:
        f["high"] = 0.8
    fit = model.analyze(ds, model.AnalysisSpec("Elong", model="quadratic")).fit
    up = model.mixture_actual_equation(fit, "upseudo")
    assert up is not None
    U = np.array([0.8, 0.8, 0.8])
    pts = np.array([[0.2, 0.3, 0.5], [0.5, 0.5, 0.0]])
    u = (U - pts) / (U.sum() - 1)
    terms = [t for t, _ in fit.mat.groups if t]
    A = model.raw_matrix(u, terms, intercept=False)
    assert np.allclose(A @ np.array([c for c, _ in up]), fit.predict(pts))
    titles = [t.title for t in report.equation_tables(fit)]
    assert "Final Equation in Terms of U_Pseudo Components" in titles


# ---------------------------------------------------------------------------- combined
def test_combined_mixture_process_model():
    ds = combined_ds()
    sp = model.space_from_dataset(ds)
    assert sp.combined and sp.scheffe and sp.mix_idx == [0, 1, 2] and sp.proc_idx == [3, 4]
    terms = model.combined_terms(3, 2, "quadratic", "linear")
    assert len(terms) == 6 + 6 * 2
    assert (0, 3) in terms and (0, 1, 4) in terms and () not in terms
    fit = model.analyze(ds, model.AnalysisSpec("Y", model="quadratic*linear")).fit
    assert not fit.mat.intercept
    coef = {c: v for v, c in fit.coded_equation()}
    # coded process factors are -1/+1, components already pseudo (= actual here)
    for k, v in (("A", 10), ("B", 6), ("C", 8), ("AB", 12), ("AC", -5), ("BC", 4), ("AD", 2),
                 ("BD", -1), ("CD", 0.5), ("AE", 1.5), ("BE", 0.7), ("CE", -0.8)):
        assert coef[k] == pytest.approx(v, abs=1e-8), k
    assert fit.anova()["terms"][0][0] == "Linear Mixture"
    assert report.model_label(fit.space, fit.terms) == "Quadratic x Linear"
    # Fit Summary walks the nested chain and suggests the true model with noise added
    noisy = combined_ds(noise=0.3)
    a = model.analyze(noisy, model.AnalysisSpec("Y"))
    assert a.summary.suggested_order == "quadratic*linear"
    # Mix x Process graphs: ternary at fixed process settings, process contour at a blend
    ref = report.default_point(a.space)
    assert ref[:3] == pytest.approx([1 / 3] * 3) and ref[3:] == pytest.approx([50, 30])
    gf = report.graph_figure(a, "ternary", [0, 1, 2], ref)
    assert gf.figure is not None
    gf2 = report.graph_figure(a, "contour", [3, 4], ref)
    assert gf2.probe(50, 30) is not None
    with pytest.raises(ValueError):
        report.graph_figure(a, "contour", [0, 3], ref)
    assert report.graph_section(a, "all_factors", [], ref).items


# ---------------------------------------------------------------------------- GLM (E15)
def _sm_glm(ds, fit, family):
    y = ds.df["Y"].to_numpy(float)
    if "N" in ds.df:
        endog = np.column_stack([y, ds.df["N"] - y])
    else:
        endog = y
    fam = sm.families.Binomial() if family == "binomial" else sm.families.Poisson()
    return sm.GLM(endog, fit.mat.X, family=fam).fit(tol=1e-12)


@pytest.mark.parametrize("kind,family", [("binary", "binomial"), ("binomial", "binomial"),
                                         ("poisson", "poisson")])
def test_glm_matches_statsmodels(kind, family):
    ds = factorial_glm_ds(kind=kind, reps=10)
    # binary data: a main-effects model (a 2FI model on 8 cells separates the data)
    spec = model.AnalysisSpec("Y", model="linear" if kind == "binary" else "2fi",
                              family=family,
                              trials="N" if kind == "binomial" else None)
    a = model.analyze(ds, spec)
    fit = a.fit
    assert glm.is_glm(fit) and fit.converged
    ref = _sm_glm(ds, fit, family)
    assert np.allclose(fit.beta, ref.params, atol=1e-6)
    assert np.allclose(np.sqrt(np.diag(fit.cov)), ref.bse, atol=1e-6)
    assert fit.deviance == pytest.approx(ref.deviance, abs=1e-6)
    assert fit.loglik == pytest.approx(ref.llf, abs=1e-6)
    assert fit.information()["aic"] == pytest.approx(ref.aic, abs=1e-6)
    assert fit.pearson() == pytest.approx(ref.pearson_chi2, abs=1e-6)
    assert fit.null["deviance"] == pytest.approx(ref.null_deviance, abs=1e-6)
    coefs = fit.coefficients()
    assert [c["p"] for c in coefs] == pytest.approx(list(ref.pvalues), abs=1e-6)
    # likelihood-ratio test of a term = drop in deviance when it is removed
    tname = "AB" if "AB" in fit.mat.labels else "A"
    j = fit.mat.labels.index(tname)
    keep = [i for i in range(fit.p) if i != j]
    red = sm.GLM(ref.model.endog if kind != "binomial" else
                 np.column_stack([ds.df["Y"], ds.df["N"] - ds.df["Y"]]),
                 fit.mat.X[:, keep], family=ref.model.family).fit(tol=1e-12)
    chi, pv = fit.term_p_values()[(0, 1) if tname == "AB" else (0,)]
    assert chi == pytest.approx(red.deviance - ref.deviance, abs=1e-6)
    assert pv == pytest.approx(stats.chi2.sf(chi, 1), abs=1e-9)
    # predictions are probabilities / expected counts
    x0 = np.array([[1.0, -1.0, 0.0]])
    mu = ref.predict(fit.row(fit.space.to_coded(x0)))
    assert fit.predict(x0) == pytest.approx(mu, abs=1e-6)
    iv = fit.interval(x0)
    assert iv["ci"][0][0] < iv["mean"][0] < iv["ci"][1][0]
    # reports
    t, _ = report.anova_table(fit, "Model")
    assert t.title.startswith("Deviance Table")
    st, _ = report.fit_statistics_table(fit)
    assert st.rows[0][0] == "-2 Log Likelihood"
    assert report.coefficient_table(fit).columns[7] in ("Odds Ratio", "Rate Ratio")
    assert report.goodness_of_fit_table(fit).rows
    sec = report.anova_section(a)
    assert sec.items
    d = fit.diagnostics()
    assert np.allclose(d["pearson"], ref.resid_pearson, atol=1e-6)
    assert np.allclose(d["deviance_resid"], ref.resid_deviance, atol=1e-6)
    assert report.diagnostics_section(a, plots_wanted=("normal", "cooks", "covratio")).items


def test_glm_fit_summary_and_auto_select():
    ds = factorial_glm_ds(kind="poisson", reps=4)
    a = model.analyze(ds, model.AnalysisSpec("Y", family="poisson"))
    assert a.summary.kind == "glm" and a.summary.suggested_order == "linear"
    assert report.glm_fit_summary_table(a.summary).rows[0][0] == "Mean"
    sel = model.analyze(ds, model.AnalysisSpec("Y", model="2fi", family="poisson",
                                               reduction="backward", alpha_out=0.05))
    assert glm.is_glm(sel.fit)
    assert set(sel.fit.terms) <= set(model.model_terms(sel.space, "2fi"))
    assert (0,) in sel.fit.terms
    with pytest.raises(ValueError):
        glm.GLMFit(a.space, a.obs, [], "binomial", trials=2)  # counts above the trials


# ---------------------------------------------------------------------------- treatment means
def test_treatment_means_lsd_and_tukey():
    rng = np.random.default_rng(5)
    groups = np.repeat([1.0, 2.0, 3.0, 4.0], 5)
    y = np.array([10, 12, 15, 11])[groups.astype(int) - 1] + rng.normal(0, 1, 20)
    ds = Dataset(pd.DataFrame({"G": groups, "Y": y}))
    ds.meta["doe"] = {"kind": "process", "factors": [{"name": "Catalyst", "column": "G",
                                                       "type": "categoric",
                                                       "levels": [1, 2, 3, 4],
                                                       "labels": ["a", "b", "c", "d"]}],
                      "responses": [{"name": "Y", "column": "Y"}]}
    fit = model.analyze(ds, model.AnalysisSpec("Y", model="linear")).fit
    tm = model.treatment_means(fit, 0)
    means = [y[groups == g].mean() for g in (1, 2, 3, 4)]
    assert [m["mean"] for m in tm["means"]] == pytest.approx(means)
    mse = fit.ms_res
    se = math.sqrt(mse * 2 / 5)
    ab = tm["pairs"][0]
    t = (means[0] - means[1]) / se
    assert ab["t"] == pytest.approx(t) and ab["p_lsd"] == pytest.approx(2 * stats.t.sf(abs(t),
                                                                                        16))
    hsd = stats.tukey_hsd(*[y[groups == g] for g in (1, 2, 3, 4)])
    assert [p["p_tukey"] for p in tm["pairs"]] == pytest.approx(
        [hsd.pvalue[i, j] for i in range(4) for j in range(i + 1, 4)], abs=1e-4)
    tables = report.treatment_tables(fit)
    assert [t.title for t in tables] == ["Treatment Means: A: Catalyst",
                                         "Pairwise Comparisons: A: Catalyst"]


# ---------------------------------------------------------------------------- several analyses
def test_several_analyses_per_response_roundtrip(tmp_path):
    from rsstatchem.ui.dx.document import DesignDocument
    from rsstatchem.ui.dx.examples import chemical_process

    doc = chemical_process()
    col = doc.response_cols[0]
    spec = doc.spec(col)
    assert model.analysis_names(spec) == ["Analysis 1"]
    spec2 = model.add_analysis(spec)
    spec2.transform = model.Transform("ln")
    doc.set_spec(spec2)
    s = doc.spec(col)
    assert model.analysis_names(s) == ["Analysis 1", "Analysis 2"] and s.active == 1
    assert doc.analyze(col).fit.transform.kind == "ln"
    doc.set_spec(model.select_analysis(s, 0))
    assert doc.analyze(col).fit.transform.kind == "none"
    assert model.spec_of(doc.spec(col), 1).transform.kind == "ln"
    path = tmp_path / "two.rsx"
    doc.save(str(path))
    back = DesignDocument.load(str(path))
    s3 = back.spec(col)
    assert model.analysis_names(s3) == ["Analysis 1", "Analysis 2"] and s3.active == 0
    rows = report.analysis_summary_table(back.ds).rows
    assert [r[1] for r in rows] == ["Analysis 1 *", "Analysis 2"]
    assert rows[1][2].startswith("Natural Log")
    one = model.remove_analysis(s3, 1)
    assert one.analyses is None and "analyses" not in one.to_meta()
    with pytest.raises(ValueError):
        model.remove_analysis(one, 0)
    # save_spec keeps the other analyses
    a = back.analyze(col)
    model.save_spec(back.ds, s3, a.fit)
    assert len(back.meta["analysis"][col]["analyses"]) == 2
    back.modified = doc.modified = False


def test_preferences_object():
    prefs = model.preferences()
    assert (prefs.alpha, prefs.ci_level, prefs.ti_proportion) == (0.05, 0.95, 0.99)
    p2 = model.Preferences().update({"alpha": 0.1, "unknown": 3})
    assert p2.alpha == 0.1 and p2.to_meta()["ci_level"] == 0.95


# ---------------------------------------------------------------------------- graphs
def test_new_model_graphs_render():
    ds, _ = conversion_ccd()
    a = model.analyze(ds, model.AnalysisSpec("Conv", model="quadratic"))
    ref = report.default_point(a.space)
    se = report.graph_figure(a, "se", [0, 1], ref)
    Xc = np.array([[45, 85, 2.5]])
    assert report.se_of_prediction(a.fit, Xc)[0] == pytest.approx(
        math.sqrt(a.fit.interval(Xc)["se_mean"][0] ** 2))
    assert se.probe is not None
    gf = report.graph_figure(a, "contour", [0, 1], ref, levels=[75, 80, 85], cmap="viridis")
    assert gf.figure is not None
    sec = report.graph_section(a, "one_factor", [0], ref, pi_bands=True)
    assert any("prediction band" in getattr(i, "text", "") for i in sec.items)
    assert report.graph_section(a, "all_factors", [], ref).items
    # LSD bars for a categoric factor
    from rsstatchem.ui.dx.examples import battery_life

    doc = battery_life()
    b = doc.analyze(doc.response_cols[0])
    sec = report.graph_section(b, "one_factor", [1], report.default_point(b.space), lsd=True)
    assert any("least significant difference" in getattr(i, "text", "") for i in sec.items)
    assert report.treatment_tables(b.fit)
    doc.modified = False


# ---------------------------------------------------------------------------- UI
@pytest.fixture
def ctx(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("RSSTATCHEM_HOME", str(tmp_path))
    from rsstatchem.app import AppContext

    context = AppContext(qapp)
    monkeypatch.setattr(context.app, "quit", lambda: None)
    return context


def _window(ctx, doc):
    from rsstatchem.ui.dx.window import DXMainWindow

    w = DXMainWindow(ctx, doc)
    ctx.register_window(w)
    w.resize(1400, 900)
    w.show()
    return w


def _shot(widget, name, qapp):
    if not SHOTS:
        return
    os.makedirs(SHOTS, exist_ok=True)
    for _ in range(3):
        qapp.processEvents()
    widget.window().grab().save(os.path.join(SHOTS, f"{name}.png"))


def test_ui_factorial_effects_tab_and_curvature(ctx, qapp):
    from rsstatchem.ui.dx.analysis import EffectsPage
    from rsstatchem.ui.dx.document import DesignDocument

    doc = DesignDocument.from_spec({"design": "factorial", "randomize": False,
                                    "options": {"center_points": 5},
                                    "factors": [{"name": "Time"}, {"name": "Temp"}],
                                    "responses": [{"name": "Yield"}]})
    vals = {(-1, -1): 39.3, (1, -1): 40.9, (-1, 1): 40.0, (1, 1): 41.5}
    centre = iter([40.3, 40.5, 40.7, 40.2, 40.6])
    col = doc.response_cols[0]
    doc.set_values([(i, col, vals.get(tuple(int(round(v)) for v in row)) or next(centre))
                    for i, row in enumerate(doc.coded_matrix())])
    w = _window(ctx, doc)
    view = w.show_node(f"analysis:{col}")
    assert view.factorial_effects and isinstance(view.fit_page, EffectsPage)
    view.set_tab("Effects")
    ep = view.fit_page
    ep.clear()
    ep.toggle_name("AB")  # with its parents A and B (hierarchy)
    assert set(doc.analyze(col).fit.terms) == {(0,), (1,), (0, 1)}
    ep.set_plot("normal")
    assert ep.graph.figure is not None and ep.graph.figure.effect_points
    _shot(view, "effects_normal_plot", qapp)
    ep.set_plot("pareto")
    assert ep.graph.png
    view.set_tab("ANOVA")
    anova = next(t for t in view.anova_page.report.tables() if t.title.startswith("ANOVA"))
    rows = {r[0]: r for r in anova.rows}
    assert rows["Curvature"][5] == "0.8137"
    _shot(view, "anova_curvature", qapp)
    view.set_tab("Model")
    assert view.model_page.curv.isVisible()
    view.model_page.curv.setCurrentIndex(view.model_page.curv.findData("residual"))
    assert doc.spec(col).curvature == "residual"
    doc.modified = False


def test_ui_transform_glm_model_diagnostics_graphs(ctx, qapp):
    from rsstatchem.ui.dx.examples import chemical_process

    doc = chemical_process()
    col = doc.response_cols[0]
    w = _window(ctx, doc)
    view = w.show_node(f"analysis:{col}")
    # several analyses: add one with a logit transform (yield in %)
    view.add_analysis()
    assert view.analysis_box.count() == 2 and view.analysis_box.currentIndex() == 1
    view.set_tab("Transform")
    tp = view.transform_page
    tp.upper.setValue(100)
    tp.radios["logit"].setChecked(True)
    assert doc.spec(col).transform.kind == "logit" and doc.spec(col).transform.upper == 100
    assert doc.analyze(col).fit.transform.kind == "logit"
    _shot(view, "transform_logit", qapp)
    view.analysis_box.setCurrentIndex(0)
    assert doc.analyze(col).fit.transform.kind == "none"
    # Model tab: higher order list and Add Term
    view.set_tab("Model")
    mp = view.model_page
    assert mp.order.findData("quartic") >= 0
    mp.term_edit.setText("A2B")
    mp.add_term()
    assert (0, 0, 1) in doc.analyze(col).fit.terms
    _shot(view, "model_add_term", qapp)
    # Diagnostics: covariance ratio, residual type, colour-by
    view.set_tab("Diagnostics")
    dp = view.diag_page
    keys = [dp.list.item(i).data(0x0100) for i in range(dp.list.count())]
    dp.residual.setCurrentIndex(dp.residual.findData("internal"))
    dp.color_by.setCurrentIndex(dp.color_by.findData("factor:0"))
    dp.list.setCurrentRow(keys.index("covratio"))
    assert dp.graph.png and "Durbin-Watson" in dp.dw.text()
    _shot(view, "diagnostics_covratio", qapp)
    # Model Graphs: standard error, all factors, custom contours and colours
    view.set_tab("Model Graphs")
    gp = view.graph_page
    gp.set_graph("se")
    assert gp.graph.figure is not None
    _shot(view, "graph_std_error", qapp)
    gp.set_graph("all_factors")
    assert gp.graph.png
    gp.levels.setText("78, 79, 80")
    gp.cmap.setCurrentIndex(gp.cmap.findData("viridis"))
    gp.set_graph("contour")
    assert gp.graph.figure is not None
    _shot(view, "graph_custom_contours", qapp)
    gp.pi_bands.setChecked(True)
    gp.set_graph("one_factor")
    assert gp.graph.png
    # Analysis Summary view (WP-A wires it into the tree)
    from rsstatchem.ui.dx.analysis import AnalysisSummaryView

    sv = AnalysisSummaryView(doc)
    sv.resize(1200, 300)
    sv.show()
    rows = sv.report.tables()[0].rows
    assert [r[1] for r in rows] == ["Analysis 1 *", "Analysis 2"]
    if SHOTS:
        for _ in range(3):
            qapp.processEvents()
        sv.grab().save(os.path.join(SHOTS, "analysis_summary.png"))
    sv.close()
    # remove the second analysis again
    view.analysis_box.setCurrentIndex(1)
    view.remove_analysis()
    assert view.analysis_box.count() == 1
    doc.modified = False


def test_ui_logistic_regression(ctx, qapp):
    from rsstatchem.ui.dx.document import DesignDocument

    doc = DesignDocument.from_spec({"design": "factorial", "randomize": False,
                                    "options": {"replicates": 6},
                                    "factors": [{"name": "A"}, {"name": "B"}, {"name": "C"}],
                                    "responses": [{"name": "Pass"}]})
    rng = np.random.default_rng(3)
    X = doc.coded_matrix()
    eta = 0.2 + 0.9 * X[:, 0] - 0.6 * X[:, 1]
    col = doc.response_cols[0]
    doc.set_values([(i, col, float(rng.random() < 1 / (1 + math.exp(-e))))
                    for i, e in enumerate(eta)])
    w = _window(ctx, doc)
    view = w.show_node(f"analysis:{col}")
    view.set_tab("Transform")
    view.transform_page.family_radios["binomial"].setChecked(True)
    assert doc.spec(col).family == "binomial"
    a = doc.analyze(col)
    assert glm.is_glm(a.fit)
    _shot(view, "transform_logistic", qapp)
    view.set_tab("ANOVA")
    titles = [t.title for t in view.anova_page.report.tables()]
    assert any(t.startswith("Deviance Table") for t in titles)
    assert "Goodness of Fit Tests" in titles
    _shot(view, "anova_logistic", qapp)
    view.set_tab("Model Graphs")
    assert view.graph_page.graph.figure is not None or view.graph_page.graph.png
    _shot(view, "graph_logistic", qapp)
    view.set_tab("Diagnostics")
    assert view.diag_page.graph.png
    doc.modified = False

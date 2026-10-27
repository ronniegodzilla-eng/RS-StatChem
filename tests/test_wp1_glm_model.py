"""WP1: GLM custom models, SS types, contrasts, EMMs, random factors, options and Save,
checked against statsmodels / pingouin and Montgomery's RCBD example."""
import math

import numpy as np
import pandas as pd
import pingouin as pg
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import OLSInfluence

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import anova, glm_output as go
from rsstatchem.core.stats.glm_model import (FactorInfo, GLMFit, TermSpec, contrast_matrix,
                                             parse_terms, parameter_estimates)


def num(text: str) -> float:
    return float(text.rstrip("*ᵃᵇᶜᵈᵉ").replace(",", ""))


def table(section, title):
    return next(i for i in section.items if getattr(i, "title", "").startswith(title))


def row(tab, *heads):
    for r in tab.rows:
        if all(h in r for h in heads):
            return r
    raise AssertionError(f"row {heads} not found in {tab.title}")


# Montgomery, Design and Analysis of Experiments, Example 4.1 (vascular graft RCBD):
# 4 extrusion pressures x 6 batches of resin (blocks).
MONTGOMERY = [90.3, 89.2, 98.2, 93.9, 87.4, 97.9, 92.5, 89.5, 90.6, 94.7, 87.0, 95.8,
              85.5, 90.8, 89.6, 86.2, 88.0, 93.4, 82.5, 89.5, 85.6, 87.4, 78.9, 90.7]


@pytest.fixture
def rcbd():
    df = pd.DataFrame({"y": MONTGOMERY, "pressure": np.repeat([8500., 8700, 8900, 9100], 6),
                       "batch": np.tile(np.arange(1., 7), 4)})
    return Dataset(df)


@pytest.fixture
def unbalanced():
    rng = np.random.default_rng(7)
    n = 72
    a = rng.integers(1, 4, n).astype(float)
    b = rng.integers(1, 3, n).astype(float)
    x = rng.normal(10, 2, n)
    y = 2 + a + 1.5 * b + 0.4 * a * b + 0.8 * x + rng.normal(0, 1.5, n)
    return pd.DataFrame({"y": y, "a": a, "b": b, "x": x})


def test_rcbd_matches_montgomery(rcbd):
    opts = go.GLMOptions(terms=[TermSpec(("pressure",)), TermSpec(("batch",))],
                         effect_size=True)
    sec = go.univariate(rcbd, "y", ["pressure", "batch"], [], opts).section
    t = table(sec, "Tests of Between-Subjects Effects")
    # Montgomery Table 4.4: SS 178.17 / 192.25 / 109.89, F = 8.11, P = 0.0019
    assert math.isclose(num(row(t, "pressure")[1]), 178.17, abs_tol=0.01)
    assert math.isclose(num(row(t, "batch")[1]), 192.25, abs_tol=0.01)
    assert math.isclose(num(row(t, "Error")[1]), 109.89, abs_tol=0.01)
    assert math.isclose(num(row(t, "pressure")[4]), 8.11, abs_tol=0.01)
    assert row(t, "Corrected Total")[2] == "23"
    assert "pressure * batch" not in [r[0] for r in t.rows]


def test_ss_types_match_statsmodels(unbalanced):
    df = unbalanced
    fa = FactorInfo("a", [1., 2., 3.], ["1", "2", "3"], (df.a - 1).astype(int).to_numpy())
    fb = FactorInfo("b", [1., 2.], ["1", "2"], (df.b - 1).astype(int).to_numpy())
    model = smf.ols("y ~ x + C(a, Sum) * C(b, Sum)", df).fit()
    for typ in (2, 3):
        fit = GLMFit(df.y.to_numpy(), [fa, fb], {"x": df.x.to_numpy()}, None, True, typ)
        ref = anova_lm(model, typ=typ)
        for term, key in (("x", "x"), ("a", "C(a, Sum)"), ("b", "C(b, Sum)"),
                          ("a * b", "C(a, Sum):C(b, Sum)")):
            t = next(tt for tt in fit.terms if tt.label == term)
            assert math.isclose(fit.hypothesis(t)[0, 0], ref.loc[key, "sum_sq"], rel_tol=1e-9)
            assert fit.term_df(t) == ref.loc[key, "df"]
    # Type I = sequential RSS differences in model order (covariate first, as in SPSS)
    fit = GLMFit(df.y.to_numpy(), [fa, fb], {"x": df.x.to_numpy()}, None, True, 1)
    formulas = ["y ~ 1", "y ~ x", "y ~ x + C(a)", "y ~ x + C(a) + C(b)",
                "y ~ x + C(a) * C(b)"]
    rss = [smf.ols(f, df).fit().ssr for f in formulas]
    for i, term in enumerate(["x", "a", "b", "a * b"]):
        t = next(tt for tt in fit.terms if tt.label == term)
        assert math.isclose(fit.hypothesis(t)[0, 0], rss[i] - rss[i + 1], rel_tol=1e-9)


def test_partial_eta_and_power(unbalanced):
    ds = Dataset(unbalanced)
    opts = go.GLMOptions(effect_size=True, power=True)
    sec = go.univariate(ds, "y", ["a", "b"], [], opts).section
    t = table(sec, "Tests of Between-Subjects Effects")
    ref = pg.anova(data=unbalanced, dv="y", between=["a", "b"], ss_type=3)
    r = row(t, "a")
    assert math.isclose(num(r[6]), ref[ref.Source == "a"]["np2"].iloc[0], abs_tol=5e-4)
    F, df1 = num(r[4]), int(r[2])
    df2 = int(row(t, "Error")[2])
    lam = F * df1
    power = stats.ncf.sf(stats.f.ppf(0.95, df1, df2), df1, df2, lam)
    assert math.isclose(num(r[7]), lam, abs_tol=2e-3 * lam)
    assert math.isclose(num(r[8]), power, abs_tol=1e-3)


def test_nested_design_type3():
    # balanced nested design: 3 schools, 4 classes within each school, 3 pupils per class
    rng = np.random.default_rng(3)
    school = np.repeat([1., 2., 3.], 12)
    klass = np.tile(np.repeat([1., 2., 3., 4.], 3), 3)
    y = school * 2 + rng.normal(0, 1, 36) + np.repeat(rng.normal(0, 1, 12), 3)
    df = pd.DataFrame({"y": y, "school": school, "klass": klass})
    terms = parse_terms("school klass(school)", ["school", "klass"], [])
    assert terms[1] == TermSpec(("klass",), (), ("school",)) and terms[1].label == \
        "klass(school)"
    sec = go.univariate(Dataset(df), "y", ["school", "klass"], [],
                        go.GLMOptions(terms=terms)).section
    t = table(sec, "Tests of Between-Subjects Effects")
    cell = df.groupby(["school", "klass"]).y.mean()
    smean = df.groupby("school").y.mean()
    ss_nested = 3 * sum((cell[(s, k)] - smean[s]) ** 2 for s, k in cell.index)
    assert math.isclose(num(row(t, "klass(school)")[1]), ss_nested, rel_tol=1e-6, abs_tol=6e-4)
    assert row(t, "klass(school)")[2] == "9"
    ref = anova_lm(smf.ols("y ~ C(school) + C(school):C(klass)", df).fit())
    assert math.isclose(num(row(t, "school")[1]), ref["sum_sq"].iloc[0], rel_tol=1e-6, abs_tol=6e-4)


def test_no_intercept_model(rcbd):
    opts = go.GLMOptions(terms=[TermSpec(("pressure",))], intercept=False)
    sec = go.univariate(rcbd, "y", ["pressure"], [], opts).section
    t = table(sec, "Tests of Between-Subjects Effects")
    assert t.rows[0][0] == "Model" and "Corrected Total" not in [r[0] for r in t.rows]
    y = np.array(MONTGOMERY)
    # with no intercept the pressure effect tests all four cell means = 0 (df = 4)
    ss = sum(6 * y[i * 6:(i + 1) * 6].mean() ** 2 for i in range(4))
    assert math.isclose(num(row(t, "pressure")[1]), ss, abs_tol=6e-4)
    assert row(t, "pressure")[2] == "4"


def test_parameter_estimates_and_robust(unbalanced):
    df = unbalanced
    fa = FactorInfo("a", [1., 2., 3.], ["1", "2", "3"], (df.a - 1).astype(int).to_numpy())
    fit = GLMFit(df.y.to_numpy(), [fa], {"x": df.x.to_numpy()})
    ref = smf.ols("y ~ x + C(a, Treatment(3))", df).fit()
    est = {p[0]: p for p in parameter_estimates(fit)}
    assert est["[a=3]"][5]  # redundant
    assert math.isclose(est["[a=1]"][1], ref.params["C(a, Treatment(3))[T.1.0]"], rel_tol=1e-9)
    assert math.isclose(est["x"][2], ref.bse["x"], rel_tol=1e-9)
    hc = {p[0]: p for p in parameter_estimates(fit, hc="HC3")}
    ref3 = smf.ols("y ~ x + C(a, Treatment(3))", df).fit(cov_type="HC3")
    assert math.isclose(hc["x"][2], ref3.bse["x"], rel_tol=1e-9)
    hc1 = {p[0]: p for p in parameter_estimates(fit, hc="HC1")}
    assert math.isclose(hc1["Intercept"][2],
                        smf.ols("y ~ x + C(a, Treatment(3))", df).fit(cov_type="HC1")
                        .bse["Intercept"], rel_tol=1e-9)


def test_heteroskedasticity_tests(unbalanced):
    df = unbalanced
    fa = FactorInfo("a", [1., 2., 3.], ["1", "2", "3"], (df.a - 1).astype(int).to_numpy())
    fit = GLMFit(df.y.to_numpy(), [fa], {"x": df.x.to_numpy()})
    from rsstatchem.core.stats.glm_model import heteroskedasticity

    res = heteroskedasticity(fit)
    exog = sm.add_constant(fit.fitted[:, 0])
    lm, lmp, f, fp = het_breuschpagan(fit.resid[:, 0], exog, robust=True)
    assert math.isclose(res["mbp"][0], lm, rel_tol=1e-8)
    assert math.isclose(res["f"][0], f, rel_tol=1e-8)
    lm0, *_ = het_breuschpagan(fit.resid[:, 0], exog, robust=False)
    assert math.isclose(res["bp"][0], lm0, rel_tol=1e-8)


def test_save_diagnostics_match_statsmodels(unbalanced):
    ds = Dataset(unbalanced)
    opts = go.GLMOptions(terms=[TermSpec(covariates=("x",)), TermSpec(("a",))],
                         save=["PRE", "RES", "ZRE", "SRE", "DRE", "COO", "LEV", "SEP"])
    res = go.univariate(ds, "y", ["a"], ["x"], opts)
    ref = smf.ols("y ~ x + C(a)", unbalanced).fit()
    infl = OLSInfluence(ref)
    np.testing.assert_allclose(res.saved["PRE"][1], ref.fittedvalues, rtol=1e-9)
    np.testing.assert_allclose(res.saved["RES"][1], ref.resid, rtol=1e-9)
    np.testing.assert_allclose(res.saved["COO"][1], infl.cooks_distance[0], rtol=1e-8)
    np.testing.assert_allclose(res.saved["LEV"][1], infl.hat_matrix_diag, rtol=1e-8)
    np.testing.assert_allclose(res.saved["SRE"][1], infl.resid_studentized_internal,
                               rtol=1e-8)
    assert res.saved["PRE"][0].startswith("Predicted Value for y")


def test_emm_and_simple_effects(unbalanced):
    df = unbalanced
    ds = Dataset(df)
    opts = go.GLMOptions(emm=["a", "a*b"], compare=True, simple=True, adjust="lsd")
    sec = go.univariate(ds, "y", ["a", "b"], [], opts).section
    cells = df.groupby(["a", "b"]).y
    cm = cells.mean()
    est = table(sec, "Estimates — a")
    for i, lv in enumerate([1., 2., 3.]):
        assert math.isclose(num(est.rows[i][1]), (cm[(lv, 1.)] + cm[(lv, 2.)]) / 2,
                            abs_tol=5e-4)
    simple = table(sec, "Pairwise Comparisons — a within b")
    ns = cells.size()
    mse = smf.ols("y ~ C(a) * C(b)", df).fit().mse_resid
    r = simple.rows[0]  # b = 1: a=1 vs a=2
    assert math.isclose(num(r[3]), cm[(1., 1.)] - cm[(2., 1.)], abs_tol=5e-4)
    assert math.isclose(num(r[4]), math.sqrt(mse * (1 / ns[(1., 1.)] + 1 / ns[(2., 1.)])),
                        abs_tol=5e-4)
    tests = table(sec, "Univariate Tests — a within b")
    assert len([r for r in tests.rows if r[1] == "Contrast"]) == 2
    # simple-effect F of a at b=1 equals the one-way F within b=1 with the pooled MSE
    sub = df[df.b == 1]
    g = sub.groupby("a").y
    ssa = float((g.size() * (g.mean() - (g.mean() * g.size()).sum() / g.size().sum()) ** 2)
                .sum())
    assert math.isclose(num(tests.rows[0][2]), ssa, rel_tol=1e-3)


def test_contrasts_cover_main_effect(rcbd):
    terms = [TermSpec(("pressure",)), TermSpec(("batch",))]
    for kind in ("deviation", "simple", "difference", "helmert", "repeated", "polynomial"):
        opts = go.GLMOptions(terms=terms, contrasts={"pressure": (kind, "last")})
        sec = go.univariate(rcbd, "y", ["pressure", "batch"], [], opts).section
        test = table(sec, "Test Results — pressure")
        assert math.isclose(num(test.rows[0][1]), 178.171, abs_tol=1e-3)
    C, labels = contrast_matrix("simple", 4, "first")
    assert labels[0] == "Level 2 vs. Level 1" and C.shape == (3, 4)
    C, labels = contrast_matrix("helmert", 3)
    np.testing.assert_allclose(C, [[1, -.5, -.5], [0, 1, -1]])
    opts = go.GLMOptions(terms=terms, contrasts={"pressure": ("simple", "last")})
    sec = go.univariate(rcbd, "y", ["pressure", "batch"], [], opts).section
    k = table(sec, "Contrast Results (K Matrix)")
    y = np.array(MONTGOMERY).reshape(4, 6).mean(axis=1)
    assert math.isclose(num(k.rows[0][4]), y[0] - y[3], abs_tol=5e-4)


def test_random_factor_mixed_model():
    # balanced two-way mixed model: A fixed (3), B random (4), n = 2 per cell
    rng = np.random.default_rng(5)
    a = np.repeat([1., 2., 3.], 8)
    b = np.tile(np.repeat([1., 2., 3., 4.], 2), 3)
    y = a + np.repeat(rng.normal(0, 1, 4), 2)[np.tile(np.arange(8), 3)] + rng.normal(0, 1, 24)
    df = pd.DataFrame({"y": y, "a": a, "b": b})
    sec = go.univariate(Dataset(df), "y", ["a"], [], go.GLMOptions(random=["b"])).section
    t = table(sec, "Tests of Between-Subjects Effects")
    ref = anova_lm(smf.ols("y ~ C(a) * C(b)", df).fit())
    ms = ref["sum_sq"] / ref["df"]
    fa = ms["C(a)"] / ms["C(a):C(b)"]
    fb = ms["C(b)"] / ms["C(a):C(b)"]
    fab = ms["C(a):C(b)"] / ms["Residual"]
    rows = {r[0]: r for r in t.rows if r[1] == "Hypothesis"}
    assert math.isclose(num(rows["a"][5]), fa, rel_tol=1e-6, abs_tol=6e-4)
    assert math.isclose(num(rows["b"][5]), fb, rel_tol=1e-6, abs_tol=6e-4)
    assert math.isclose(num(rows["a * b"][5]), fab, rel_tol=1e-6, abs_tol=6e-4)
    ems = table(sec, "Expected Mean Squares")
    emsrow = row(ems, "a")
    assert emsrow[-1] == "a" and num(emsrow[2]) == pytest.approx(2.0)  # Var(a*b) coef = n


def test_lack_of_fit():
    x = np.repeat([1., 2., 3., 4., 5.], 3)
    rng = np.random.default_rng(2)
    y = 1 + 0.5 * x + 0.3 * x ** 2 + rng.normal(0, 0.5, 15)
    df = pd.DataFrame({"y": y, "x": x})
    sec = go.univariate(Dataset(df), "y", [], ["x"], go.GLMOptions(lack_of_fit=True)).section
    t = table(sec, "Lack of Fit Tests")
    pe = sum(((g - g.mean()) ** 2).sum() for _, g in df.groupby("x").y)
    sse = smf.ols("y ~ x", df).fit().ssr
    assert math.isclose(num(row(t, "Pure Error")[1]), pe, rel_tol=1e-4, abs_tol=6e-4)
    assert math.isclose(num(row(t, "Lack of Fit")[1]), sse - pe, rel_tol=1e-4, abs_tol=6e-4)
    assert row(t, "Lack of Fit")[2] == "3"


def test_wls_weight(unbalanced):
    df = unbalanced.copy()
    df["w"] = np.linspace(0.5, 2, len(df))
    sec = go.univariate(Dataset(df), "y", ["a"], ["x"], go.GLMOptions(wls="w")).section
    t = table(sec, "Tests of Between-Subjects Effects")
    ref = smf.wls("y ~ x + C(a, Sum)", df, weights=df.w).fit()
    f_a = ref.f_test("C(a, Sum)[S.1.0] = C(a, Sum)[S.2.0] = 0").fvalue
    assert math.isclose(num(row(t, "a")[4]), float(np.squeeze(f_a)), rel_tol=1e-3)
    assert any("Weighted Least Squares" in f for f in t.footnotes)


def test_posthoc_and_profile_in_glm(rcbd):
    opts = go.GLMOptions(terms=[TermSpec(("pressure",)), TermSpec(("batch",))],
                         posthoc_factors=["pressure"],
                         posthoc_methods=["tukey", "duncan", "snk", "dunnett"],
                         plots=[("pressure", None, None)], error_bars="ci")
    sec = go.univariate(rcbd, "y", ["pressure", "batch"], [], opts).section
    mc = table(sec, "Multiple Comparisons")
    # Montgomery Example 4.1: MSE = 7.33 on 15 df; Tukey q(.05;4,15) = 4.08
    r = row(mc, "Tukey HSD", "8500", "9100") if any(
        x[1] == "Tukey HSD" and x[3] == "9100" for x in mc.rows) else None
    tuk = [x for x in mc.rows if x[3] == "9100"][0]
    assert math.isclose(num(tuk[4]), 7.05, abs_tol=0.01)
    hs = table(sec, "Homogeneous Subsets")
    assert hs.columns[3] == ("Subset for alpha = 0.05", "1")
    assert any(i.kind == "image" for i in sec.items)
    del r


def test_old_signature_still_works(unbalanced):
    sec = anova.univariate(Dataset(unbalanced), "y", ["a", "b"], ["x"], descriptives=True)
    assert table(sec, "Tests of Between-Subjects Effects")


def test_manova_custom_model_and_sscp(unbalanced):
    from statsmodels.multivariate.manova import MANOVA

    from rsstatchem.core.stats import manova

    df = unbalanced.copy()
    df["y2"] = df.y * 0.3 + np.random.default_rng(1).normal(0, 1, len(df))
    opts = go.GLMOptions(terms=[TermSpec(("a",)), TermSpec(("b",))], residual_sscp=True,
                         power=True, save=["RES"])
    res = manova.manova(Dataset(df), ["y", "y2"], ["a", "b"], options=opts)
    mv = table(res.section, "Multivariate Tests")
    ref = MANOVA.from_formula("y + y2 ~ C(a, Sum) + C(b, Sum)", df).mv_test()
    pillai = ref.results["C(a, Sum)"]["stat"].loc["Pillai's trace", "Value"]
    i = next(k for k, r in enumerate(mv.rows) if r[0] == "a")
    assert math.isclose(num(mv.rows[i][2]), pillai, abs_tol=5e-4)
    model = smf.ols("y ~ C(a) + C(b)", df).fit()
    model2 = smf.ols("y2 ~ C(a) + C(b)", df).fit()
    r = np.corrcoef(model.resid, model2.resid)[0, 1]
    bart = table(res.section, "Bartlett's Test of Sphericity")
    assert math.isclose(num(bart.rows[0][1]), 1 - r * r, abs_tol=5e-4)
    np.testing.assert_allclose(res.saved["RES_y"][1], model.resid, atol=1e-8)


def test_repeated_measures_options():
    from rsstatchem.core.stats import repeated

    rng = np.random.default_rng(9)
    n = 24
    g = np.repeat([1., 2., 3.], 8)
    base = rng.normal(0, 1, n)
    df = pd.DataFrame({"t1": base + rng.normal(0, 1, n), "t2": base + 1 + rng.normal(0, 1, n),
                       "t3": base + 1.5 + rng.normal(0, 1, n), "g": g})
    opts = go.GLMOptions(effect_size=True, power=True, emm=["time*g"], simple=True,
                         posthoc_factors=["g"], posthoc_methods=["snk", "t3"],
                         plots=[("time", "g")], error_bars="se", save=["PRE"])
    res = repeated.repeated_measures(Dataset(df), [("time", 3)], ["t1", "t2", "t3"], ["g"],
                                     options=opts, within_contrasts={"time": "repeated"})
    tc = table(res.section, "Tests of Within-Subjects Contrasts")
    d = (df.t1 - df.t2).to_numpy()
    r = next(x for x in tc.rows if x[1] == "Level 1 vs. Level 2")
    # unnormalized repeated contrast: SS = N * mean(d)^2 adjusted for the g effects (Type III
    # intercept of d in the effect-coded one-way model = mean of the group means of d)
    gm = pd.Series(d).groupby(g).mean().mean()
    assert math.isclose(num(r[2]), n * gm ** 2, rel_tol=2e-3)
    assert any(t.title.startswith("Pairwise Comparisons — time within g")
               for t in res.section.items if t.kind == "table")
    assert table(res.section, "Homogeneous Subsets")
    w = table(res.section, "Tests of Within-Subjects Effects")
    assert w.columns[-1].startswith("Observed Power")
    assert set(res.saved) == {"PRE_t1", "PRE_t2", "PRE_t3"}

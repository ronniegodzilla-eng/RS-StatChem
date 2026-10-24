"""CB-SEM (AMOS / lavaan style) checked against published lavaan output for Holzinger &
Swineford (1939) and Bollen's Political Democracy model, against semopy (reference only,
never imported by app code), OLS for saturated path models, and hand-computed CR / AVE /
HTMT."""
import math

import numpy as np
import pandas as pd
import pytest

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import sem, sem_fit as sf, sem_syntax as sx

HS = "visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6\nspeed =~ x7 + x8 + x9"
PD_CONSTRAINED = """ind60 =~ x1 + x2 + x3
dem60 =~ y1 + a*y2 + b*y3 + c*y4
dem65 =~ y5 + a*y6 + b*y7 + c*y8
dem60 ~ ind60
dem65 ~ ind60 + dem60
y1 ~~ y5
y2 ~~ y4 + y6
y3 ~~ y7
y4 ~~ y8
y6 ~~ y8"""


def table(section, title):
    items = [i for i in section.items if getattr(i, "kind", "") == "table"]
    exact = [i for i in items if i.title == title]
    return exact[0] if exact else next(i for i in items if i.title.startswith(title))


def num(text):
    return float(text.replace("***", "0").rstrip("*"))


def row(tab, label):
    return next(r for r in tab.rows if r[0] == label)


@pytest.fixture(scope="module")
def hs():
    semopy = pytest.importorskip("semopy")
    from semopy.examples import holzinger39
    del semopy
    return holzinger39.get_data().reset_index(drop=True)


@pytest.fixture(scope="module")
def pd_data():
    pytest.importorskip("semopy")
    from semopy.examples import political_democracy
    return political_democracy.get_data().reset_index(drop=True), political_democracy.get_model()


def fit(df, syntax, **kw):
    model = sx.build_model(syntax, list(df.columns), std_lv=kw.pop("std_lv", False))
    X = df[model.observed].to_numpy(float)
    return sf.estimate(model, X, **kw), X


# ------------------------------------------------------------------ syntax
def test_parser_modifiers_labels_and_continuation():
    text = """# comment
    f =~ x1 + NA*x2 + a*x3 +
         start(0.5)*x4   ! another comment
    f ~~ 1*f ; y ~ b*f
    x1 ~~ x2
    ind := a*b"""
    m = sx.build_model(text, ["x1", "x2", "x3", "x4", "y"])
    p = {q.name: q for q in m.params}
    assert not p["f=~x1"].free and p["f=~x1"].value == 1      # marker imposed by default
    assert p["f~~f"].free is False and p["f~~f"].value == 1
    assert p["f=~x3"].label == "a" and p["y~f"].label == "b"
    assert p["f=~x4"].start == 0.5
    assert "x1~~x2" in p
    assert m.defined[0].names == ["a", "b"]


def test_marker_and_std_lv_defaults():
    m = sx.build_model("f =~ x1 + x2 + x3", ["x1", "x2", "x3"])
    p = {q.name: q for q in m.params}
    assert not p["f=~x1"].free and p["f=~x1"].value == 1 and p["f~~f"].free
    m = sx.build_model("f =~ x1 + x2 + x3", ["x1", "x2", "x3"], std_lv=True)
    p = {q.name: q for q in m.params}
    assert p["f=~x1"].free and not p["f~~f"].free and p["f~~f"].value == 1
    m = sx.build_model("f =~ NA*x1 + x2 + x3\nf ~~ 1*f", ["x1", "x2", "x3"])
    assert {q.name: q for q in m.params}["f=~x1"].free


def test_auto_parameters_like_lavaan():
    m = sx.build_model("y1 ~ x1 + x2\ny2 ~ x1", ["x1", "x2", "y1", "y2"])
    names = {q.name for q in m.params}
    assert "x1~~x2" in names                 # exogenous covariates correlated
    assert "y1~~y2" in names                 # auto.cov.y among pure outcomes
    m = sx.build_model("y1 ~ x1 + x2\ny2 ~ x1", ["x1", "x2", "y1", "y2"], auto_cov_y=False)
    assert "y1~~y2" not in {q.name for q in m.params}


def test_equality_by_shared_label_and_constraint():
    m = sx.build_model("f =~ x1 + a*x2 + a*x3", ["x1", "x2", "x3"])
    p = {q.name: q for q in m.params}
    assert p["f=~x2"].index == p["f=~x3"].index
    m = sx.build_model("f =~ x1 + a*x2 + b*x3\na == b", ["x1", "x2", "x3"])
    p = {q.name: q for q in m.params}
    assert p["f=~x2"].index == p["f=~x3"].index
    m = sx.build_model("f =~ x1 + a*x2 + x3\na == 0.5", ["x1", "x2", "x3"])
    p = {q.name: q for q in m.params}
    assert not p["f=~x2"].free and p["f=~x2"].value == 0.5


@pytest.mark.parametrize("text, fragment", [
    ("f =~ x1 + x2 + q9", "'q9' is not a variable"),
    ("y ~ 1 + x1", "Intercepts"),
    ("x1 x2", "No operator"),
    ("y ~ a*x1\nz := a*c", "Unknown label 'c'"),
    ("x1 =~ x2 + x3", "also an observed variable"),
    ("f =~ x1 + x2\nf ~ f", "regressed on itself"),
    ("y ~ a*x1\na > 0", "Inequality"),
])
def test_validation_messages(text, fragment):
    ok, msg = sx.validate(text, ["x1", "x2", "x3", "y", "z"])
    assert not ok and fragment in msg


def test_validation_reports_df_and_identification():
    ok, msg = sx.validate(HS, [f"x{i}" for i in range(1, 10)])
    assert ok and "21 free parameters" in msg and "df = 24" in msg
    ok, msg = sx.validate("f =~ x1 + x2", ["x1", "x2"])
    assert not ok and "not identified" in msg


def test_case_insensitive_names_take_dataset_spelling():
    m = sx.build_model("F =~ X1 + x2 + X3", ["x1", "x2", "x3"])
    assert m.observed == ["x1", "x2", "x3"]


# ------------------------------------------------------------------ Holzinger & Swineford
def test_hs_cfa_matches_lavaan(hs):
    est, _ = fit(hs, HS)
    fm = sf.fit_measures(est)
    assert est.converged
    assert fm["chi2"] == pytest.approx(85.306, abs=1e-3)
    assert fm["df"] == 24 and fm["npar"] == 21
    assert fm["cfi"] == pytest.approx(0.931, abs=5e-4)
    assert fm["tli"] == pytest.approx(0.896, abs=5e-4)
    assert fm["rmsea"] == pytest.approx(0.092, abs=5e-4)
    assert fm["rmsea_lo"] == pytest.approx(0.071, abs=5e-4)
    assert fm["rmsea_hi"] == pytest.approx(0.114, abs=5e-4)
    assert fm["srmr"] == pytest.approx(0.065, abs=5e-4)
    assert fm["chi2_base"] == pytest.approx(918.852, abs=1e-2)
    assert fm["logl"] == pytest.approx(-3737.745, abs=1e-2)
    assert fm["aic"] == pytest.approx(7517.490, abs=1e-2)
    assert fm["bic"] == pytest.approx(7595.339, abs=1e-2)
    # lavaan parameter estimates and standard errors (expected information)
    se = np.sqrt(np.diag(est.cov))
    ref = {"visual=~x2": (0.554, 0.100), "visual=~x3": (0.729, 0.109),
           "textual=~x5": (1.113, 0.065), "textual=~x6": (0.926, 0.055),
           "speed=~x8": (1.180, 0.165), "speed=~x9": (1.082, 0.151),
           "visual~~textual": (0.408, 0.074), "visual~~speed": (0.262, 0.056),
           "textual~~speed": (0.173, 0.049), "visual~~visual": (0.809, 0.145),
           "x1~~x1": (0.549, 0.114), "x7~~x7": (0.799, 0.081)}
    for p in est.model.params:
        if p.name in ref:
            assert est.theta[p.index] == pytest.approx(ref[p.name][0], abs=6e-4), p.name
            assert se[p.index] == pytest.approx(ref[p.name][1], abs=6e-4), p.name


def test_hs_modification_indices_match_lavaan(hs):
    est, _ = fit(hs, HS)
    mis = {(m.lhs, m.op, m.rhs): m for m in sf.modification_indices(est)}
    m1 = mis[("visual", "=~", "x9")]
    assert m1.mi == pytest.approx(36.411, abs=2e-3)
    assert m1.epc == pytest.approx(0.577, abs=1e-3)
    assert m1.sepc == pytest.approx(0.515, abs=1e-3)
    m2 = mis[("x7", "~~", "x8")]
    assert m2.mi == pytest.approx(34.145, abs=2e-3)
    assert m2.epc == pytest.approx(0.536, abs=1e-3)
    assert m2.sepc == pytest.approx(0.859, abs=1e-3)


def test_hs_estimates_match_semopy(hs):
    semopy = pytest.importorskip("semopy")
    est, _ = fit(hs, HS)
    model = semopy.Model(HS)
    model.fit(hs, obj="MLW")
    ins = model.inspect(std_est=True)
    for _, r in ins.iterrows():
        if r["op"] != "~" or r["Estimate"] == 1.0:
            continue
        p = next(q for q in est.model.params if q.op == "=~" and q.lhs == r["rval"]
                 and q.rhs == r["lval"])
        assert est.theta[p.index] == pytest.approx(r["Estimate"], abs=3e-3)
        assert math.sqrt(est.cov[p.index, p.index]) == pytest.approx(float(r["Std. Err"]),
                                                                     abs=3e-3)


def test_std_lv_gives_same_fit_and_standardized_solution(hs):
    ds = Dataset(hs)
    a = sem.cb_sem(ds, HS, outputs={"diagram": False})
    b = sem.cb_sem(ds, HS, identification="std.lv", outputs={"diagram": False})
    assert a.fit["chi2"] == pytest.approx(b.fit["chi2"], abs=1e-6)
    la = table(a.section, "Factor Loadings")
    lb = table(b.section, "Factor Loadings")
    for ra, rb in zip(la.rows, lb.rows):
        assert ra[5] == rb[5]          # standardized estimates identical
    assert num(row(lb, "x1 <--- visual")[1]) == pytest.approx(0.900, abs=1e-3)  # lavaan


def test_wishart_likelihood_scales_chi_square(hs):
    a, _ = fit(hs, HS)
    b, _ = fit(hs, HS, likelihood="wishart")
    n = len(hs)
    assert b.chi2 == pytest.approx(a.chi2 * (n - 1) / n, rel=1e-6)
    k = next(p.index for p in a.model.params if p.name == "visual~~visual")
    assert b.theta[k] == pytest.approx(a.theta[k] * n / (n - 1), rel=1e-5)


def test_gls_and_uls_match_semopy(hs):
    semopy = pytest.importorskip("semopy")
    for obj in ("GLS", "ULS"):
        est, _ = fit(hs, HS, estimator=obj)
        assert est.converged
        model = semopy.Model(HS)
        model.fit(hs, obj=obj)
        ins = model.inspect()
        for _, r in ins.iterrows():
            if r["op"] != "~" or r["Estimate"] == 1.0:
                continue
            p = next(q for q in est.model.params if q.op == "=~" and q.lhs == r["rval"]
                     and q.rhs == r["lval"])
            assert est.theta[p.index] == pytest.approx(r["Estimate"], abs=5e-3), (obj, p.name)


def test_observed_information_and_robust_close_to_expected(hs):
    a, X = fit(hs, HS)
    b, _ = fit(hs, HS, information="observed")
    c, _ = fit(hs, HS, robust=True)
    sa, sb, sc = (np.sqrt(np.diag(e.cov)) for e in (a, b, c))
    assert np.allclose(sa, sb, rtol=0.25)
    assert np.allclose(sa, sc, rtol=0.6)
    fm = sf.fit_measures(c)
    assert fm["chi2_scaled"] == pytest.approx(fm["chi2"] / fm["scale"])
    assert 0.7 < fm["scale"] < 1.5


def test_satorra_bentler_near_one_for_normal_data():
    rng = np.random.default_rng(3)
    n = 3000
    f = rng.normal(size=n)
    df = pd.DataFrame({f"x{k}": l * f + rng.normal(scale=0.7, size=n)
                       for k, l in enumerate((1, 0.8, 0.9, 1.1), 1)})
    est, _ = fit(df, "f =~ x1 + x2 + x3 + x4", robust=True)
    assert est.robust["scale"] == pytest.approx(1.0, abs=0.15)
    a, _ = fit(df, "f =~ x1 + x2 + x3 + x4")
    assert np.allclose(np.sqrt(np.diag(est.cov)), np.sqrt(np.diag(a.cov)), rtol=0.1)


# ------------------------------------------------------------------ Political Democracy
def test_political_democracy_matches_lavaan(pd_data):
    df, syntax = pd_data
    est, _ = fit(df, syntax)
    fm = sf.fit_measures(est)
    assert fm["chi2"] == pytest.approx(38.125, abs=1e-3)
    assert fm["df"] == 35
    assert fm["p"] == pytest.approx(0.329, abs=1e-3)
    assert fm["cfi"] == pytest.approx(0.995, abs=5e-4)
    assert fm["tli"] == pytest.approx(0.993, abs=5e-4)
    assert fm["rmsea"] == pytest.approx(0.035, abs=5e-4)
    assert fm["rmsea_hi"] == pytest.approx(0.092, abs=5e-4)
    assert fm["srmr"] == pytest.approx(0.044, abs=5e-4)
    se = np.sqrt(np.diag(est.cov))
    ref = {"dem60~ind60": (1.483, 0.399), "dem65~ind60": (0.572, 0.221),
           "dem65~dem60": (0.837, 0.098), "ind60=~x2": (2.180, 0.139),
           "dem60=~y2": (1.257, 0.182), "y2~~y6": (2.153, 0.734), "dem65~~dem65":
           (0.172, 0.215)}
    for p in est.model.params:
        if p.name in ref:
            assert est.theta[p.index] == pytest.approx(ref[p.name][0], abs=6e-4), p.name
            assert se[p.index] == pytest.approx(ref[p.name][1], abs=6e-4), p.name


def test_political_democracy_equality_constraints_match_lavaan(pd_data):
    df, _ = pd_data
    est, _ = fit(df, PD_CONSTRAINED)
    assert est.chi2 == pytest.approx(40.179, abs=1e-3)
    assert est.df == 38


def test_political_democracy_vs_semopy(pd_data):
    semopy = pytest.importorskip("semopy")
    df, syntax = pd_data
    est, _ = fit(df, syntax)
    model = semopy.Model(syntax)
    model.fit(df, obj="MLW")
    ins = model.inspect()
    for _, r in ins[ins["op"] == "~"].iterrows():
        if r["Estimate"] == 1.0:
            continue
        lhs, rhs = r["lval"], r["rval"]
        p = next((q for q in est.model.params if (q.op == "~" and q.lhs == lhs and q.rhs == rhs)
                  or (q.op == "=~" and q.lhs == rhs and q.rhs == lhs)))
        assert est.theta[p.index] == pytest.approx(r["Estimate"], abs=5e-3), p.name


def test_effects_and_bootstrap_indirect(pd_data):
    df, syntax = pd_data
    text = syntax.replace("dem60 ~ ind60", "dem60 ~ a*ind60").replace(
        "dem65 ~ ind60 + dem60", "dem65 ~ c*ind60 + b*dem60") + "\nind := a*b\ntot := c + a*b"
    res = sem.cb_sem(Dataset(df), text, boot=1000, seed=7, outputs={"diagram": False})
    eff = table(res.section, "Total, Direct and Indirect Effects")
    r = row(eff, "dem65 <--- ind60")
    a, b, c = 1.483, 0.837, 0.572
    assert num(r[2]) == pytest.approx(a * b, abs=2e-3)
    assert num(r[3]) == pytest.approx(c + a * b, abs=2e-3)
    defined = table(res.section, "Defined Parameters")
    ind = row(defined, "ind := a*b")
    est_ind = num(ind[1])
    assert est_ind == pytest.approx(a * b, abs=2e-3)
    boot = table(res.section, "Bootstrap Confidence Intervals (95%, unstandardized)")
    br = row(boot, "ind := a*b")
    lo_p, hi_p, lo_bc, hi_bc = (num(x) for x in br[3:7])
    assert lo_p < est_ind < hi_p and lo_bc < est_ind < hi_bc
    assert lo_p > 0                               # significant indirect effect
    spec = table(res.section, "Indirect Effects (Delta Method)")
    assert num(row(spec, "ind60 → dem60 → dem65")[1]) == pytest.approx(est_ind, abs=1e-3)
    again = sem.cb_sem(Dataset(df), text, boot=1000, seed=7, outputs={"diagram": False})
    assert table(again.section, "Bootstrap Confidence Intervals (95%").rows == boot.rows


# ------------------------------------------------------------------ path analysis / mediation
def test_saturated_mediation_equals_ols():
    import statsmodels.api as smapi
    rng = np.random.default_rng(5)
    n = 250
    x = rng.normal(size=n)
    m = 0.5 * x + rng.normal(size=n)
    y = 0.3 * x + 0.4 * m + rng.normal(size=n)
    df = pd.DataFrame({"x": x, "m": m, "y": y})
    res = sem.cb_sem(Dataset(df), sx.mediation_syntax("x", "m", "y"), boot=500, seed=3,
                     outputs={"diagram": False})
    assert res.fit["df"] == 0 and res.fit["chi2"] == pytest.approx(0, abs=1e-8)
    ols_m = smapi.OLS(m, smapi.add_constant(x)).fit()
    ols_y = smapi.OLS(y, smapi.add_constant(np.column_stack([x, m]))).fit()
    reg = table(res.section, "Regression Weights")
    assert num(row(reg, "m <--- x")[1]) == pytest.approx(ols_m.params[1], abs=1e-3)
    assert num(row(reg, "y <--- x")[1]) == pytest.approx(ols_y.params[1], abs=1e-3)
    assert num(row(reg, "y <--- m")[1]) == pytest.approx(ols_y.params[2], abs=1e-3)
    # ML SE = OLS SE * sqrt((n - k) / n)
    k = 3
    assert num(row(reg, "y <--- m")[2]) == pytest.approx(
        ols_y.bse[2] * math.sqrt((n - k) / n), abs=1e-3)
    d = res.extra["derived"]
    i = d.sl["def"].start
    ab = ols_m.params[1] * ols_y.params[2]
    assert res.extra["dvals"][i] == pytest.approx(ab, abs=1e-5)
    boot = res.extra["boot"]["derived"][:, i]
    lo, hi = np.quantile(boot, [0.025, 0.975])
    assert lo < ab < hi


def test_r2_in_path_model():
    rng = np.random.default_rng(8)
    n = 300
    x1, x2 = rng.normal(size=(2, n))
    y = 0.5 * x1 - 0.3 * x2 + rng.normal(size=n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    res = sem.cb_sem(Dataset(df), "y ~ x1 + x2", outputs={"diagram": False})
    r2 = num(row(table(res.section, "Squared Multiple Correlations"), "y")[1])
    ref = np.corrcoef(np.column_stack([x1, x2]) @ np.linalg.lstsq(
        np.column_stack([np.ones(n), x1, x2]), y, rcond=None)[0][1:], y)[0, 1] ** 2
    assert r2 == pytest.approx(ref, abs=1e-3)


# ------------------------------------------------------------------ measurement quality
def _two_factor(n=500, seed=1):
    rng = np.random.default_rng(seed)
    f1 = rng.normal(size=n)
    f2 = 0.5 * f1 + rng.normal(scale=math.sqrt(0.75), size=n)
    cols = {}
    for k, l in enumerate((0.8, 0.7, 0.6), 1):
        cols[f"a{k}"] = l * f1 + rng.normal(scale=math.sqrt(1 - l * l), size=n)
    for k, l in enumerate((0.85, 0.75, 0.65, 0.55), 1):
        cols[f"b{k}"] = l * f2 + rng.normal(scale=math.sqrt(1 - l * l), size=n)
    return pd.DataFrame(cols)


def test_cr_ave_alpha_htmt_by_hand():
    df = _two_factor()
    res = sem.cb_sem(Dataset(df), "A =~ a1 + a2 + a3\nB =~ b1 + b2 + b3 + b4",
                     outputs={"diagram": False})
    q = res.extra["quality"]
    for f, items in (("A", ["a1", "a2", "a3"]), ("B", ["b1", "b2", "b3", "b4"])):
        lam = q[f]["loadings"]
        theta = 1 - lam ** 2
        assert q[f]["cr"] == pytest.approx(lam.sum() ** 2 / (lam.sum() ** 2 + theta.sum()))
        assert q[f]["ave"] == pytest.approx(np.mean(lam ** 2))
        X = df[items].to_numpy()
        k = len(items)
        alpha = k / (k - 1) * (1 - X.var(axis=0, ddof=1).sum() / X.sum(axis=1).var(ddof=1))
        assert q[f]["alpha"] == pytest.approx(alpha)
    R = df.corr().to_numpy()
    a, b = [0, 1, 2], [3, 4, 5, 6]
    het = np.abs(R[np.ix_(a, b)]).mean()
    def mono(ii):
        return (np.abs(R[np.ix_(ii, ii)]).sum() - len(ii)) / (len(ii) * (len(ii) - 1))
    htmt = het / math.sqrt(mono(a) * mono(b))
    assert res.extra["discriminant"]["htmt"][1, 0] == pytest.approx(htmt)
    fl = table(res.section, "Discriminant Validity: Fornell-Larcker")
    assert num(fl.rows[0][1]) == pytest.approx(math.sqrt(q["A"]["ave"]), abs=1e-3)
    assert num(fl.rows[1][1]) == pytest.approx(res.extra["discriminant"]["latent_corr"][1, 0],
                                               abs=1e-3)
    # standardized loadings recover the generating values
    assert np.allclose(q["B"]["loadings"], [0.85, 0.75, 0.65, 0.55], atol=0.07)


# ------------------------------------------------------------------ problems
def test_not_identified_message():
    df = _two_factor()
    res = sem.cb_sem(Dataset(df), "A =~ NA*a1 + a2 + a3", outputs={"diagram": False})
    texts = " ".join(i.text for i in res.section.items if getattr(i, "kind", "") == "text")
    assert "not identified" in texts and res.est is None


def test_negative_df_message():
    df = _two_factor()
    res = sem.cb_sem(Dataset(df), "A =~ a1 + a2", outputs={"diagram": False})
    texts = " ".join(i.text for i in res.section.items if getattr(i, "kind", "") == "text")
    assert "not identified" in texts


def test_heywood_case_warning():
    target = np.array([[1, .8, .8], [.8, 1, .4], [.8, .4, 1]])
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(200, 3))
    Z = (Z - Z.mean(0)) @ np.linalg.inv(np.linalg.cholesky(np.cov(Z.T, bias=True))).T
    X = Z @ np.linalg.cholesky(target).T
    df = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    res = sem.cb_sem(Dataset(df), "f =~ x1 + x2 + x3", outputs={"diagram": False})
    texts = " ".join(i.text for i in res.section.items if getattr(i, "kind", "") == "text")
    assert "Heywood" in texts and "greater than 1" in texts
    k = next(p.index for p in res.est.model.params if p.name == "x1~~x1")
    assert res.est.theta[k] == pytest.approx(1 - 1.6, abs=1e-4)


def test_syntax_error_and_missing_values_listwise():
    df = _two_factor(200)
    df.loc[[3, 10, 50], "a2"] = np.nan
    res = sem.cb_sem(Dataset(df), "A =~ a1 + a2 + a3\nB =~ b1 + b2 + b3",
                     outputs={"diagram": False})
    summary = table(res.section, "Model Summary")
    assert row(summary, "Number of observations used")[1] == "197"
    bad = sem.cb_sem(Dataset(df), "A =~ a1 + zz")
    assert any(getattr(i, "style", "") == "warning" and "zz" in i.text
               for i in bad.section.items)


def test_uls_reports_no_chi_square_indices(hs):
    res = sem.cb_sem(Dataset(hs), HS, estimator="ULS", outputs={"diagram": False})
    fit_t = table(res.section, "Model Fit")
    assert row(fit_t, "Chi-square (χ²)")[1] == "n/a"
    assert not any(r[0] == "CFI" for r in fit_t.rows)
    assert any(r[0] == "GFI" for r in fit_t.rows)


# ------------------------------------------------------------------ outputs
def test_full_output_and_factor_scores():
    df = _two_factor(300, seed=4)
    res = sem.cb_sem(Dataset(df), "A =~ a1 + a2 + a3\nB =~ b1 + b2 + b3 + b4\nB ~ A",
                     outputs={"residuals": True, "sample_cov": True, "implied_cov": True},
                     save_scores=True)
    titles = [getattr(i, "title", "") for i in res.section.items]
    for t in ("Model Summary", "Assessment of Normality", "Model Fit",
              "Factor Loadings (Measurement Model)", "Regression Weights (Structural Model)",
              "Variances", "Squared Multiple Correlations (R²)",
              "Total, Direct and Indirect Effects", "Construct Reliability and Validity",
              "Discriminant Validity: HTMT", "Standardized Residual Covariances",
              "Sample Covariances", "Implied Covariances", "Path Diagram"):
        assert t in titles, t
    img = next(i for i in res.section.items if getattr(i, "kind", "") == "image")
    assert img.png[:8] == b"\x89PNG\r\n\x1a\n"
    fa, fb = res.saved["FS_A"][1], res.saved["FS_B"][1]
    assert len(fa) == len(df) and fa.notna().all()
    assert np.corrcoef(fa, df[["a1", "a2", "a3"]].mean(axis=1))[0, 1] > 0.9
    assert np.corrcoef(fa, fb)[0, 1] > 0.3


def test_diagram_handles_second_order_and_mediation():
    from rsstatchem.core.stats.sem_diagram import path_diagram
    m = sx.build_model("A =~ a1 + a2\nB =~ b1 + b2\nG =~ A + B\nM =~ m1 + m2\nY =~ y1 + y2\n"
                       "M ~ G\nY ~ M + G\na1 ~~ b1",
                       ["a1", "a2", "b1", "b2", "m1", "m2", "y1", "y2"])
    png = path_diagram(m, {p.name: 0.5 for p in m.params}, {"M": 0.3, "Y": 0.4})
    assert png[:4] == b"\x89PNG"


def test_questionnaire_example_runs():
    from rsstatchem.io.files import load_project
    ds = load_project("examples/contoh_kuesioner.rsd")
    res = sem.cb_sem(ds, "X =~ X1 + X2 + X3 + X4 + X5 + X6\nY =~ Y1 + Y2 + Y3 + Y4 + Y5\n"
                         "Y ~ X", boot=300, seed=1)
    assert res.est.converged and res.fit["df"] == 43
    assert 0 < res.fit["cfi"] <= 1


def test_dialog_offscreen(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from rsstatchem.io.files import load_project
    from rsstatchem.ui.dialogs import sem as dlgmod
    assert dlgmod.MENUS == {"Analyze/Structural Equation Modeling":
                            [("CB-SEM...", dlgmod.CBSEMDialog)]}
    dlgmod.CBSEMDialog._state.clear()
    ds = load_project("examples/contoh_kuesioner.rsd")
    dlg = dlgmod.CBSEMDialog(ds)
    assert [c for c, _ in dlg.constructs()] == ["X", "Y"]    # auto-detected item groups
    dlg.p_from.setCurrentText("X")
    dlg.p_to.setCurrentText("Y")
    dlg._add_pair(dlg.paths, dlg.p_from, dlg.p_to, "→")
    dlg.template.setCurrentIndex(dlg.template.findData("sem"))
    dlg._generate()
    assert "Y ~ p1*X" in dlg.syntax()
    assert dlg.validate() is None
    dlg.opts["out_diagram"] = False
    res = dlg.compute()
    assert res.est is not None and res.est.converged
    dlg.editor.setPlainText("X =~ X1 + nope")
    assert "nope" in dlg.validate()
    app.processEvents()


def test_selftest():
    from rsstatchem.ui.dialogs.sem import selftest
    selftest()

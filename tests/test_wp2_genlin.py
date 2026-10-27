"""WP2: GENLIN / GEE / Probit / Loglinear against statsmodels, scipy and textbook formulas."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import special, stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import genlin


def num(text: str) -> float:
    return float(text.rstrip("ᵃᵇᶜᵈᵉ*"))


def table(sec, title, k=0):
    return [t for t in sec.items if getattr(t, "title", "").startswith(title)][k]


def prow(t, name):
    return next(r for r in t.rows if r[0] == name)


@pytest.fixture
def data():
    rng = np.random.default_rng(12)
    n = 300
    x = rng.normal(size=n)
    g = rng.integers(1, 4, n).astype(float)
    h = rng.integers(1, 3, n).astype(float)
    mu = np.exp(0.4 + 0.5 * x + 0.3 * (g == 1) - 0.2 * (g == 2))
    cnt = rng.poisson(mu).astype(float)
    nb = rng.negative_binomial(2, 2 / (2 + mu)).astype(float)
    pos = rng.gamma(3.0, np.exp(0.2 + 0.3 * x) / 3.0)
    b = (rng.random(n) < special.expit(-0.2 + 0.8 * x)).astype(float)
    y = 1 + 0.5 * x + 0.4 * (g == 1) + rng.normal(size=n)
    return Dataset(pd.DataFrame({"cnt": cnt, "nb": nb, "pos": pos, "b": b, "y": y, "x": x,
                                 "g": g, "h": h, "expo": rng.uniform(1, 3, n)}))


def design(df):
    return pd.DataFrame({"const": 1.0, "g1": (df.g == 1) * 1.0, "g2": (df.g == 2) * 1.0,
                         "x": df.x})


def test_poisson_matches_statsmodels(data):
    sec = genlin.genlin(data, "cnt", ["g"], ["x"], "poisson", "log").section
    df = data.df
    ref = sm.GLM(df.cnt, design(df), family=sm.families.Poisson()).fit()
    pe = table(sec, "Parameter Estimates")
    assert math.isclose(num(prow(pe, "x")[1]), ref.params["x"], abs_tol=5e-4)
    assert math.isclose(num(prow(pe, "x")[2]), ref.bse["x"], abs_tol=5e-4)
    assert math.isclose(num(prow(pe, "[g=1]")[1]), ref.params["g1"], abs_tol=5e-4)
    assert prow(pe, "[g=3]")[1].startswith("0")
    gof = table(sec, "Goodness of Fit")
    assert math.isclose(num(prow(gof, "Log Likelihoodᵇ")[1]), ref.llf, abs_tol=5e-3)
    assert math.isclose(num(prow(gof, "Akaike's Information Criterion (AIC)")[1]), ref.aic,
                        abs_tol=5e-3)
    assert math.isclose(num(prow(gof, "Deviance")[1]), ref.deviance, abs_tol=5e-3)
    omni = table(sec, "Omnibus Test")
    assert math.isclose(num(omni.rows[0][0]), 2 * (ref.llf - ref.llnull), abs_tol=5e-3)
    # main-effects model: Type III Wald for g = joint Wald of its dummies
    eff = table(sec, "Tests of Model Effects")
    w = ref.wald_test("g1 = 0, g2 = 0", scalar=True).statistic
    assert math.isclose(num(prow(eff, "g")[1]), w, rel_tol=1e-3)


def test_normal_ml_scale_and_se(data):
    sec = genlin.genlin(data, "y", ["g"], ["x"]).section
    df = data.df
    X = design(df)
    ols = sm.OLS(df.y, X).fit()
    n = len(df)
    phi = ols.ssr / n
    pe = table(sec, "Parameter Estimates")
    scale = prow(pe, "(Scale)")
    assert math.isclose(num(scale[1]), phi, abs_tol=5e-4)
    assert math.isclose(num(scale[2]), phi * math.sqrt(2 / n), abs_tol=5e-4)
    Xn = X.to_numpy()
    se_x = math.sqrt(phi * np.linalg.inv(Xn.T @ Xn)[3, 3])
    assert math.isclose(num(prow(pe, "x")[2]), se_x, abs_tol=5e-4)
    gof = table(sec, "Goodness of Fit")
    ll = -n / 2 * (math.log(2 * math.pi * phi) + 1)
    assert math.isclose(num(prow(gof, "Log Likelihoodᵇ")[1]), ll, abs_tol=5e-3)
    aic = -2 * ll + 2 * 5
    assert math.isclose(num(prow(gof, "Akaike's Information Criterion (AIC)")[1]), aic,
                        abs_tol=5e-3)


def test_gamma_log_ml_scale_and_fisher(data):
    df = data.df
    X = pd.DataFrame({"const": 1.0, "x": df.x})
    fam = sm.families.Gamma(sm.families.links.Log())
    ref = sm.GLM(df.pos, X, family=fam).fit()
    sec = genlin.genlin(data, "pos", [], ["x"], "gamma", "log", method="fisher").section
    pe = table(sec, "Parameter Estimates")
    phi = num(prow(pe, "(Scale)")[1])
    # ML scale maximises the gamma log likelihood at the fitted means
    f = lambda s: fam.loglike(df.pos, ref.fittedvalues, scale=s)  # noqa: E731
    assert f(phi) >= f(phi * 1.01) and f(phi) >= f(phi * 0.99)
    ref_phi = sm.GLM(df.pos, X, family=fam).fit(scale=phi)
    assert math.isclose(num(prow(pe, "x")[2]), ref_phi.bse["x"], abs_tol=5e-4)
    assert math.isclose(num(prow(pe, "x")[1]), ref.params["x"], abs_tol=5e-4)


def test_binary_models_lower_category_and_exp(data):
    sec = genlin.genlin(data, "b", [], ["x"], "binomial", "logit", exp_params=True).section
    df = data.df
    ref = sm.Logit(df.b, sm.add_constant(df.x)).fit(disp=0)
    pe = table(sec, "Parameter Estimates")
    assert math.isclose(num(prow(pe, "x")[1]), -ref.params["x"], abs_tol=5e-4)
    assert math.isclose(num(prow(pe, "x")[8]), math.exp(-ref.params["x"]), abs_tol=5e-4)
    info = table(sec, "Model Information")
    assert "models 0 as the response" in info.footnotes[0]
    sec = genlin.genlin(data, "b", [], ["x"], "binomial", "logit", reference="first").section
    assert math.isclose(num(prow(table(sec, "Parameter Estimates"), "x")[1]),
                        ref.params["x"], abs_tol=5e-4)


def test_events_trials_and_offset(data):
    df = data.df.copy()
    df["n"] = 10.0
    rng = np.random.default_rng(3)
    df["ev"] = rng.binomial(10, special.expit(0.3 * df.x)).astype(float)
    ds = Dataset(df)
    sec = genlin.genlin(ds, "ev", [], ["x"], "binomial", "probit", trials="n").section
    ref = sm.GLM(np.column_stack([df.ev, 10 - df.ev]), sm.add_constant(df.x),
                 family=sm.families.Binomial(sm.families.links.Probit())).fit()
    assert math.isclose(num(prow(table(sec, "Parameter Estimates"), "x")[1]), ref.params["x"],
                        abs_tol=5e-4)
    assert math.isclose(num(prow(table(sec, "Goodness of Fit"), "Log Likelihoodᵇ")[1]),
                        ref.llf, abs_tol=5e-3)
    df["lexpo"] = np.log(df.expo)
    ds = Dataset(df)
    sec = genlin.genlin(ds, "cnt", [], ["x"], "poisson", "log", offset="lexpo").section
    ref = sm.GLM(df.cnt, sm.add_constant(df.x), family=sm.families.Poisson(),
                 offset=df.lexpo).fit()
    assert math.isclose(num(prow(table(sec, "Parameter Estimates"), "x")[1]), ref.params["x"],
                        abs_tol=5e-4)


def test_robust_covariance(data):
    sec = genlin.genlin(data, "cnt", ["g"], ["x"], "poisson", "log", cov_type="robust").section
    df = data.df
    ref = sm.GLM(df.cnt, design(df), family=sm.families.Poisson()).fit(cov_type="HC0")
    assert math.isclose(num(prow(table(sec, "Parameter Estimates"), "x")[2]), ref.bse["x"],
                        abs_tol=5e-4)


def test_type1_lr_and_type3_interaction(data):
    sec = genlin.genlin(data, "cnt", ["g", "h"], ["x"], "poisson", "log",
                        terms=["g", "h", "g*h", "x"], analysis_type=13, chi_type="lr").section
    df = data.df
    t1 = table(sec, "Tests of Model Effects", 0)
    f0 = sm.GLM(df.cnt, np.ones(len(df)), family=sm.families.Poisson()).fit()
    Xg = pd.DataFrame({"c": 1.0, "g1": (df.g == 1) * 1.0, "g2": (df.g == 2) * 1.0})
    fg = sm.GLM(df.cnt, Xg, family=sm.families.Poisson()).fit()
    assert math.isclose(num(prow(t1, "g")[1]), f0.deviance - fg.deviance, abs_tol=5e-3)
    t3 = table(sec, "Tests of Model Effects", 1)
    # Type III LR for the interaction = deviance difference without g*h
    Xm = Xg.assign(h1=(df.h == 1) * 1.0, x=df.x)
    Xi = Xm.assign(i1=Xm.g1 * Xm.h1, i2=Xm.g2 * Xm.h1)
    fm = sm.GLM(df.cnt, Xm, family=sm.families.Poisson()).fit()
    fi = sm.GLM(df.cnt, Xi, family=sm.families.Poisson()).fit()
    assert math.isclose(num(prow(t3, "g * h")[1]), fm.deviance - fi.deviance, abs_tol=5e-3)


def test_negative_binomial_estimated(data):
    sec = genlin.genlin(data, "nb", [], ["x"], "negbin", "log", nb_estimate=True).section
    df = data.df
    ref = sm.NegativeBinomial(df.nb, sm.add_constant(df.x)).fit(disp=0)
    pe = table(sec, "Parameter Estimates")
    assert math.isclose(num(prow(pe, "(Negative binomial)")[1]), ref.params["alpha"],
                        rel_tol=2e-2)
    assert math.isclose(num(prow(pe, "x")[1]), ref.params["x"], abs_tol=2e-3)


def test_emmeans_poisson(data):
    sec = genlin.genlin(data, "cnt", ["g"], ["x"], "poisson", "log", emmeans=["g"],
                        emm_adjust="bonferroni").section
    df = data.df
    ref = sm.GLM(df.cnt, design(df), family=sm.families.Poisson()).fit()
    est = table(sec, "Estimates")
    m1 = math.exp(ref.params["const"] + ref.params["g1"] + ref.params["x"] * df.x.mean())
    assert math.isclose(num(est.rows[0][1]), m1, abs_tol=5e-4)
    assert table(sec, "Pairwise Comparisons")
    assert table(sec, "Overall Test Results")


def test_saved_diagnostics(data):
    res = genlin.genlin(data, "cnt", ["g"], ["x"], "poisson", "log",
                        save=tuple(genlin.GENLIN_SAVE))
    df = data.df
    ref = sm.GLM(df.cnt, design(df), family=sm.families.Poisson()).fit()
    inf = ref.get_influence()
    sv = {k: v[1].to_numpy() for k, v in res.saved.items()}
    assert np.allclose(sv["MeanPredicted"], ref.fittedvalues)
    assert np.allclose(sv["Leverage"], inf.hat_matrix_diag)
    assert np.allclose(sv["StdPearsonResidual"], inf.resid_studentized)
    assert np.allclose(sv["CooksDistance"], inf.cooks_distance[0])
    assert np.allclose(sv["DevianceResidual"], ref.resid_deviance)
    assert np.all(sv["CIMeanPredictedLower"] < sv["MeanPredicted"])


def test_gee_exchangeable_matches_statsmodels(data):
    df = data.df.copy()
    df["s"] = np.repeat(np.arange(75), 4).astype(float)
    df["t"] = np.tile(np.arange(1, 5), 75).astype(float)
    ds = Dataset(df)
    sec = genlin.genlin(ds, "y", ["g"], ["x"], repeated={"subject": ["s"], "within": ["t"],
                                                         "corr": "exchangeable"}).section
    ref = sm.GEE(df.y.to_numpy(), design(df).to_numpy(), groups=df.s.to_numpy(),
                 time=df.t.to_numpy() - 1, cov_struct=sm.cov_struct.Exchangeable()).fit()
    pe = table(sec, "Parameter Estimates")
    assert math.isclose(num(prow(pe, "x")[1]), ref.params[3], abs_tol=5e-4)
    assert math.isclose(num(prow(pe, "x")[2]), ref.bse[3], abs_tol=5e-4)
    wc = table(sec, "Working Correlation Matrix")
    assert math.isclose(num(wc.rows[0][2]), ref.cov_struct.dep_params, abs_tol=5e-4)
    cds = table(sec, "Correlated Data Summary")
    assert cds.rows[1][2] == "75"
    assert table(sec, "Goodness of Fit")


FINNEY = pd.DataFrame({"dose": [10.2, 7.7, 5.1, 3.8, 2.6], "n": [50, 49, 46, 48, 50.],
                       "r": [44, 42, 24, 16, 6.]})   # Finney (1971) rotenone on Macrosiphoniella


def test_probit_finney_rotenone():
    ds = Dataset(FINNEY.copy())
    sec = genlin.probit(ds, "r", "n", "dose", transform="log10")
    model = sm.GLM(np.column_stack([FINNEY.r, FINNEY.n - FINNEY.r]),
                   sm.add_constant(np.log10(FINNEY.dose)),
                   family=sm.families.Binomial(sm.families.links.Probit()))
    ref = model.fit()
    pe = table(sec, "Parameter Estimates")
    # Finney's probit line: slope 4.21 per log10 unit
    assert math.isclose(num(pe.rows[0][2]), ref.params.iloc[1], abs_tol=2e-3)
    assert math.isclose(num(pe.rows[0][2]), 4.21, abs_tol=0.01)
    assert math.isclose(num(pe.rows[1][2]), ref.params.iloc[0], abs_tol=2e-3)
    # standard errors from the observed information (inverse Hessian), as SPSS PROBIT
    se_obs = np.sqrt(np.diag(np.linalg.inv(-model.hessian(ref.params, observed=True))))
    assert math.isclose(num(pe.rows[0][3]), se_obs[1], abs_tol=2e-3)
    chi = table(sec, "Chi-Square Tests")
    pearson = float(np.sum((FINNEY.r - FINNEY.n * ref.fittedvalues) ** 2
                           / (FINNEY.n * ref.fittedvalues * (1 - ref.fittedvalues))))
    assert math.isclose(num(chi.rows[0][2]), pearson, abs_tol=2e-3)
    cl = table(sec, "Confidence Limits")
    row50 = next(r for r in cl.rows if r[0] == ".500")
    a, b = ref.params.iloc[0], ref.params.iloc[1]
    ld50 = 10 ** (-a / b)
    assert math.isclose(num(row50[1]), ld50, abs_tol=2e-3)
    # LD50 of Finney's rotenone example is about 4.8 mg/l
    assert 4.5 < ld50 < 5.1
    lo, hi = num(row50[2]), num(row50[3])
    assert lo < ld50 < hi
    # Fieller: at the limits (a + b x - 0)^2 = z^2 Var(a + b x)
    V = np.linalg.inv(-model.hessian(ref.params, observed=True))
    z = stats.norm.ppf(0.975)
    for lim in (lo, hi):
        xl = math.log10(lim)
        lhs = (a + b * xl) ** 2
        rhs = z * z * (V[0, 0] + 2 * xl * V[0, 1] + xl * xl * V[1, 1])
        assert math.isclose(lhs, rhs, rel_tol=2e-2)


def test_probit_logit_factor_potency():
    df = pd.concat([FINNEY.assign(f=1.0), FINNEY.assign(f=2.0, dose=FINNEY.dose * 2)])
    sec = genlin.probit(Dataset(df.reset_index(drop=True)), "r", "n", "dose", factor="f",
                        model="logit", relative_potency=True, parallelism=True)
    rmp = table(sec, "Relative Median Potency")
    est = num(rmp.rows[0][2])
    assert math.isclose(est, 2.0, rel_tol=1e-3) or math.isclose(est, 0.5, rel_tol=1e-3)


def test_loglinear_general_independence_g2():
    rng = np.random.default_rng(7)
    a = rng.integers(1, 4, 400).astype(float)
    b = rng.integers(1, 3, 400).astype(float)
    ds = Dataset(pd.DataFrame({"a": a, "b": b}))
    sec = genlin.loglinear_general(ds, ["a", "b"], terms=["a", "b"])
    tab = pd.crosstab(a, b).to_numpy()
    g, p, dof, _ = stats.chi2_contingency(tab, correction=False, lambda_="log-likelihood")
    x2 = stats.chi2_contingency(tab, correction=False)[0]
    gof = table(sec, "Goodness-of-Fit")
    assert math.isclose(num(gof.rows[0][1]), g, abs_tol=5e-4)
    assert math.isclose(num(gof.rows[1][1]), x2, abs_tol=5e-4)
    assert gof.rows[0][2] == str(dof)
    sat = genlin.loglinear_general(ds, ["a", "b"])
    assert num(table(sat, "Goodness-of-Fit").rows[0][1]) == 0


def test_hiloglinear_backward_removes_noise():
    rng = np.random.default_rng(9)
    n = 600
    a = rng.integers(1, 3, n)
    b = np.where(rng.random(n) < 0.8, a, 3 - a)
    c = rng.integers(1, 3, n)
    ds = Dataset(pd.DataFrame({"a": a * 1.0, "b": b * 1.0, "c": c * 1.0}))
    sec = genlin.loglinear_selection(ds, ["a", "b", "c"])
    steps = table(sec, "Step Summary")
    final = [r for r in steps.rows if r[1].startswith("Generating Class")][-1][2]
    assert "a*b" in final and "a*b*c" not in final
    k = table(sec, "K-Way and Higher-Order Effects")
    assert k.rows[0][2] == "7"
    logit = genlin.loglinear_logit(ds, ["b"], ["a"])
    assert table(logit, "Analysis of Dispersion")

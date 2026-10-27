"""WP2: Binary logistic blocks / interactions / contrasts / saves, multinomial and ordinal
extras, compared with statsmodels and published SPSS coding tables."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import logistic


def num(text: str) -> float:
    return float(text.rstrip("ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ*"))


def tables(sec, title):
    return [t for t in sec.items if getattr(t, "title", "").startswith(title)]


@pytest.fixture
def data():
    rng = np.random.default_rng(21)
    n = 300
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    g = rng.integers(1, 4, n).astype(float)
    eta = -0.3 + 0.9 * x1 - 0.5 * x2 + 0.6 * x1 * x2 + 0.7 * (g == 2) - 0.4 * (g == 3)
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    return Dataset(pd.DataFrame({"y": y, "x1": x1, "x2": x2, "g": g}))


def test_contrast_codings_match_spss_tables():
    c = logistic.contrast_coding
    assert np.allclose(c(3, "simple"), [[2 / 3, -1 / 3], [-1 / 3, 2 / 3], [-1 / 3, -1 / 3]])
    assert np.allclose(c(3, "helmert"), [[2 / 3, 0], [-1 / 3, .5], [-1 / 3, -.5]])
    assert np.allclose(c(3, "difference"), [[-.5, -1 / 3], [.5, -1 / 3], [0, 2 / 3]])
    assert np.allclose(c(3, "repeated"), [[2 / 3, 1 / 3], [-1 / 3, 1 / 3], [-1 / 3, -2 / 3]])
    assert np.allclose(c(3, "polynomial"), [[-.7071, .4082], [0, -.8165], [.7071, .4082]],
                       atol=1e-4)


def test_blocks_with_interaction_match_statsmodels(data):
    res = logistic.binary_logistic(data, "y", blocks=[("enter", ["x1", "x2"]),
                                                      ("enter", ["x1*x2"])], hosmer=True)
    df = data.df
    X1 = sm.add_constant(df[["x1", "x2"]])
    X2 = X1.assign(x12=df.x1 * df.x2)
    f1 = sm.Logit(df.y, X1).fit(disp=0)
    f2 = sm.Logit(df.y, X2).fit(disp=0)
    omni = tables(res.section, "Omnibus Tests")
    assert len(omni) == 2
    block2 = omni[1]
    assert math.isclose(num(block2.rows[1][2]), 2 * (f2.llf - f1.llf), abs_tol=2e-3)
    assert math.isclose(num(block2.rows[2][2]), 2 * (f2.llf - f2.llnull), abs_tol=2e-3)
    vie = tables(res.section, "Variables in the Equation")[-1]
    row = next(r for r in vie.rows if r[1] == "x1 by x2")
    assert math.isclose(num(row[2]), f2.params["x12"], abs_tol=2e-3)
    assert math.isclose(num(row[3]), f2.bse["x12"], abs_tol=2e-3)


@pytest.mark.parametrize("contrast", ["simple", "helmert", "repeated", "polynomial",
                                      "difference"])
def test_contrast_reparametrisation_same_fit(data, contrast):
    res = logistic.binary_logistic(data, "y", ["x1", "g"], {"g": (contrast, "last")})
    ind = logistic.binary_logistic(data, "y", ["x1", "g"], {"g": ("indicator", "last")})
    ll = [num(tables(r.section, "Model Summary")[-1].rows[0][1]) for r in (res, ind)]
    assert math.isclose(ll[0], ll[1], abs_tol=1e-3)
    # simple contrast with last reference = indicator coefficients
    if contrast == "simple":
        a = tables(res.section, "Variables in the Equation")[-1]
        b = tables(ind.section, "Variables in the Equation")[-1]
        ra = next(r for r in a.rows if r[1] == "g(1)")
        rb = next(r for r in b.rows if r[1] == "g(1)")
        assert ra[2] == rb[2]


def test_saves_match_glm_influence(data):
    keys = ("prob", "group", "cook", "lever", "dfbeta", "resid", "lresid", "sresid",
            "zresid", "dev")
    res = logistic.binary_logistic(data, "y", ["x1", "x2"], save=keys, class_plot=True,
                                   casewise=True, correlations=True)
    df = data.df
    glm = sm.GLM(df.y, sm.add_constant(df[["x1", "x2"]]), family=sm.families.Binomial()).fit()
    infl = glm.get_influence()
    sv = {k: v[1].to_numpy() for k, v in res.saved.items()}
    assert np.allclose(sv["PRE"], glm.fittedvalues, atol=1e-6)
    assert np.allclose(sv["LEV"], infl.hat_matrix_diag, atol=1e-6)
    assert np.allclose(sv["ZRE"], glm.resid_pearson, atol=1e-6)
    assert np.allclose(sv["DEV"], glm.resid_deviance, atol=1e-6)
    assert np.allclose(sv["COO"], infl.cooks_distance[0] * 3, atol=1e-6)
    assert np.allclose(sv["RES"], df.y - glm.fittedvalues, atol=1e-6)
    # DfBeta = one-step change in the coefficients when the case is deleted
    i = int(np.argmax(sv["COO"]))
    refit = sm.Logit(df.y.drop(i), sm.add_constant(df[["x1", "x2"]]).drop(i)).fit(disp=0)
    one_step = sv["DFB1"][i]
    assert math.isclose(one_step, glm.params["x1"] - refit.params["x1"], rel_tol=0.25)
    assert any(getattr(t, "kind", "") == "image" for t in res.section.items)
    assert tables(res.section, "Correlation Matrix")


def test_conditional_forward_selects_signal(data):
    res = logistic.binary_logistic(data, "y", ["x1", "x2", "g"], {"g": ("indicator", "last")},
                                   method="fstep_cond")
    removed = tables(res.section, "Model if Term Removed")
    assert removed and "conditional" in removed[0].footnotes[0]
    vie = tables(res.section, "Variables in the Equation")[-1]
    assert any(r[1] == "x1" for r in vie.rows)
    res = logistic.binary_logistic(data, "y", ["x1", "x2"], method="bstep_cond")
    assert tables(res.section, "Model if Term Removed")


@pytest.fixture
def mdata():
    rng = np.random.default_rng(8)
    n = 400
    x = rng.normal(size=n)
    g = rng.integers(1, 3, n).astype(float)
    h = rng.integers(1, 3, n).astype(float)
    u = np.column_stack([np.zeros(n), 0.8 * x + 0.5 * (g == 1), -0.6 * x + 0.7 * (g == 1)
                         * (h == 1)])
    p = np.exp(u) / np.exp(u).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(3, p=pi) for pi in p]) + 1.0
    s = np.exp(0.5 * (g == 2))
    o = np.digitize((0.9 * x + 0.4 * (h == 1)) + rng.logistic(size=n) * s, [-1, 0.5]) + 1.0
    return Dataset(pd.DataFrame({"y": y, "x": x, "g": g, "h": h, "o": o, "noise":
                                 rng.normal(size=n)}))


def test_multinomial_custom_interaction_ic_and_saves(mdata):
    res = logistic.multinomial(mdata, "y", ["g", "h"], ["x"], model="custom",
                               terms=["x", "g", "h", "g*h"], info_criteria=True,
                               correlations=True, covariances=True, cell_probs=True,
                               save=("est", "pre", "max", "acp"))
    df = mdata.df
    X = pd.DataFrame({"x": df.x, "g1": (df.g == 1) * 1.0, "h1": (df.h == 1) * 1.0})
    X["gh"] = X.g1 * X.h1
    yr = (df.y - 1).astype(int).map({0: 1, 1: 2, 2: 0})
    ref = sm.MNLogit(yr, sm.add_constant(X)).fit(disp=0)
    pe = tables(res.section, "Parameter Estimates")[0]
    row = next(r for r in pe.rows if r[1] == "[g=1] * [h=1]")
    assert math.isclose(num(row[2]), ref.params.iloc[4, 0], abs_tol=2e-3)
    assert math.isclose(num(row[3]), ref.bse.iloc[4, 0], abs_tol=2e-3)
    assert any(r[1] == "[g=2] * [h=1]" and r[2].startswith("0") for r in pe.rows)
    mfi = tables(res.section, "Model Fitting Information")[0]
    k = 2 * 5
    m2 = num(mfi.rows[1][3])
    assert math.isclose(num(mfi.rows[1][1]), m2 + 2 * k, abs_tol=2e-3)
    assert tables(res.section, "Asymptotic Correlation Matrix")
    assert tables(res.section, "Observed and Predicted Frequencies")
    est = np.column_stack([res.saved[f"EST{j}"][1] for j in (1, 2, 3)])
    assert np.allclose(est.sum(axis=1), 1)
    acp = res.saved["ACP"][1].to_numpy()
    assert np.allclose(acp, est[np.arange(len(df)), (df.y - 1).astype(int)])


def test_multinomial_stepwise_and_scale(mdata):
    res = logistic.multinomial(mdata, "y", ["g"], ["x", "noise"], model="custom",
                               terms=["g"], stepwise_terms=["x", "noise"],
                               step_method="forward_entry")
    steps = tables(res.section, "Step Summary")[0]
    assert [r[2] for r in steps.rows[1:]] == ["x"]
    res2 = logistic.multinomial(mdata, "y", ["g"], ["x"], scale="pearson")
    res1 = logistic.multinomial(mdata, "y", ["g"], ["x"])
    s1 = num(tables(res1.section, "Parameter Estimates")[0].rows[1][3])
    s2 = num(tables(res2.section, "Parameter Estimates")[0].rows[1][3])
    gof = tables(res1.section, "Goodness-of-Fit")[0]
    phi = num(gof.rows[0][1]) / num(gof.rows[0][2])
    assert math.isclose(s2, s1 * math.sqrt(phi), rel_tol=2e-2)


def test_ordinal_scale_model_and_custom_location(mdata):
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    base = logistic.ordinal(mdata, "o", ["h"], ["x"], iteration_history=True)
    df = mdata.df
    exog = pd.DataFrame({"x": df.x, "h1": (df.h == 1) * 1.})
    om = OrderedModel((df.o - 1).astype(int), exog, distr="logit").fit(
        method="bfgs", disp=0, maxiter=5000)
    pe = tables(base.section, "Parameter Estimates")[0]
    assert math.isclose(num(next(r for r in pe.rows if r[1] == "x")[2]), om.params["x"],
                        abs_tol=2e-3)
    assert tables(base.section, "Iteration History")
    sc = logistic.ordinal(mdata, "o", ["h", "g"], ["x"], location=["x", "h"],
                          scale_terms=["g"], cell_info=True, correlations=True,
                          save=("est", "pre", "max", "acp"))
    pe = tables(sc.section, "Parameter Estimates")[0]
    scale_row = next(r for r in pe.rows if r[0] == "Scale")
    assert scale_row[1] == "[g=1]" and num(scale_row[2]) < 0   # g = 2 is more dispersed
    mfi = tables(sc.section, "Model Fitting Information")[0]
    assert mfi.rows[1][3] == "3"
    assert tables(sc.section, "Cell Information")
    est = np.column_stack([sc.saved[f"EST{j}"][1] for j in (1, 2, 3)])
    assert np.allclose(est.sum(axis=1), 1)

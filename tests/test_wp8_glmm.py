"""WP8: Generalized linear mixed models (SPSS GENLINMIXED, REPL-PQL; optional Laplace).

Reference values:
* SAS/STAT PROC GLIMMIX documentation, "Getting Started: Logistic Regressions with Random
  Intercepts" (multicenter side-effect data printed in the documentation,
  ``data_wp8/multicenter.csv``). GLIMMIX's default METHOD=RSPL is the same residual
  pseudo-likelihood / PQL linearisation IBM documents for GENLINMIXED (REPL-PQL, phi = 1 for
  the binomial): -2 Res Log Pseudo-Likelihood 81.44; Var(Intercept center) 0.6176 (SE
  0.3181); Intercept -0.8071 (0.2514), group A -0.4896 (0.2034); group LS-means on the
  probability scale A 0.2147 (SE 0.04385), B 0.3085 (SE 0.05363). (GLIMMIX's t tests use
  containment df = 14; GENLINMIXED's default residual df here is 30 - 2 = 28.)
* R ``lme4::glmer(cbind(incidence, size - incidence) ~ period + (1 | herd), binomial, cbpp)``
  (Laplace): fixed effects -1.3983, -0.9919, -1.1282, -1.5797 (SE 0.2312, 0.3032, 0.3228,
  0.4220), Var(herd) 0.4123, deviance 184.1, AIC 194.1. R-equivalent, not SPSS output.
* statsmodels MixedLM (REML) for the normal/identity analog, GLM for models without random
  effects, NegativeBinomial for the ancillary k, BinomialBayesMixedGLM (loose agreement).
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import glmm as gm

DATA = Path(__file__).parent / "data_wp8"


def _table(section, title):
    return next(t for t in section.items if getattr(t, "title", "").startswith(title))


def _row(table, label):
    return next(r for r in table.rows if r[0] == label)


@pytest.fixture(scope="module")
def multicenter():
    return Dataset(pd.read_csv(DATA / "multicenter.csv"))


@pytest.fixture(scope="module")
def cbpp():
    return Dataset(pd.read_csv(DATA / "cbpp.csv").astype(float))


def test_multicenter_matches_glimmix_rspl(multicenter):
    r = gm.glmm(multicenter, "sideeffect", factors=["group"], subjects=["center"],
                distribution="binomial", link="logit", trials="n",
                random_blocks=[gm.RandomBlock(subjects=["center"])], emm=["group"])
    f = r.fit
    assert f.beta == pytest.approx([-0.8071, -0.4896], abs=1e-4)
    assert np.sqrt(np.diag(f.C)) == pytest.approx([0.2514, 0.2034], abs=1e-4)
    assert f.natural[0] == pytest.approx(0.6176, abs=1e-4)
    assert math.sqrt(f.cov_nat[0, 0]) == pytest.approx(0.3181, abs=1e-4)
    assert f.neg2 == pytest.approx(81.44, abs=0.005)
    coef = _table(r.section, "Fixed Coefficients")
    assert _row(coef, "[group=A]")[1:3] == ["-.490", ".203"]
    assert _row(coef, "[group=B]")[1].startswith("0")
    cov = _table(r.section, "Covariance Parameter Estimates")
    assert cov.rows[0][1:4] == ["Var(Intercept)", ".618", ".318"]
    emm = _table(r.section, "Estimated Means: group")
    assert [row[1:3] for row in emm.rows] == [[".215", ".044"], [".309", ".054"]]
    fx = _table(r.section, "Fixed Effects")
    assert _row(fx, "group")[1:4] == ["5.791", "1", "28"]
    summary = _table(r.section, "Model Summary")
    assert "81.444" in summary.footnotes[0]


def test_cbpp_laplace_matches_glmer(cbpp):
    df = cbpp.df.copy()
    for k in (2, 3, 4):
        df[f"p{k}"] = (df["period"] == k).astype(float)
    ds = Dataset(df)
    r = gm.glmm(ds, "incidence", covariates=["p2", "p3", "p4"], subjects=["herd"],
                distribution="binomial", link="logit", trials="size",
                random_blocks=[gm.RandomBlock(subjects=["herd"])], method="laplace")
    f = r.fit
    assert f.method == "laplace"
    # glmer's printed estimates are slightly less converged than ours (its deviance at the
    # printed point is 1e-5 higher), hence the 1e-3 tolerance
    assert f.beta == pytest.approx([-1.3983, -0.9919, -1.1282, -1.5797], abs=1e-3)
    assert np.sqrt(np.diag(f.C)) == pytest.approx([0.2312, 0.3032, 0.3228, 0.4220], rel=0.02)
    assert f.natural[0] == pytest.approx(0.4123, abs=2e-3)
    assert f.neg2 == pytest.approx(184.05, abs=0.01)
    assert f.neg2 + 2 * 5 == pytest.approx(194.1, abs=0.05)


def test_cbpp_pql_factor_coding(cbpp):
    r = gm.glmm(cbpp, "incidence", factors=["period"], subjects=["herd"],
                distribution="binomial", link="logit", trials="size",
                random_blocks=[gm.RandomBlock(subjects=["herd"])], emm=["period"],
                contrast="pairwise", adjust="seqbonferroni")
    coef = _table(r.section, "Fixed Coefficients")
    assert _row(coef, "[period=4]")[1].startswith("0")      # last category redundant
    assert len(coef.rows[0]) == 10                            # Exp(Coefficient) columns
    con = _table(r.section, "Pairwise Contrasts")
    assert len(con.rows) == 6
    from statsmodels.stats.multitest import multipletests

    raw = [float(row[5]) if row[5] != "<.001" else 0.0005 for row in con.rows]
    adj = multipletests(raw, method="holm")[1]
    got = [float(row[6]) if row[6] != "<.001" else 0.0005 for row in con.rows]
    assert got == pytest.approx(list(adj), abs=2e-3)


def test_normal_identity_equals_mixedlm():
    import statsmodels.formula.api as smf

    rng = np.random.default_rng(5)
    ns, per = 25, 8
    g = np.repeat(np.arange(ns), per)
    x = rng.normal(size=ns * per)
    y = 2 + 0.7 * x + rng.normal(0, 1, ns)[g] + rng.normal(0, 0.4, ns)[g] * x + \
        rng.normal(0, 0.8, ns * per)
    df = pd.DataFrame({"y": y, "x": x, "g": g.astype(float)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        md = smf.mixedlm("y ~ x", df, groups=df["g"], re_formula="~x").fit(reml=True)
        mi = smf.mixedlm("y ~ x", df, groups=df["g"]).fit(reml=True)
    ds = Dataset(df)
    r = gm.glmm(ds, "y", covariates=["x"], subjects=["g"],
                random_blocks=[gm.RandomBlock(effects=["x"], subjects=["g"], covtype="UN")])
    f = r.fit
    assert f.beta == pytest.approx(md.fe_params.to_numpy(), abs=1e-5)
    assert f.neg2 == pytest.approx(-2 * md.llf, abs=1e-4)
    G = md.cov_re.to_numpy()
    assert f.natural[:3] == pytest.approx([G[0, 0], G[1, 0], G[1, 1]], abs=1e-4)
    assert f.phi == pytest.approx(md.scale, rel=1e-4)
    assert np.sqrt(np.diag(f.C)) == pytest.approx(md.bse_fe.to_numpy(), rel=2e-3)
    r2 = gm.glmm(ds, "y", covariates=["x"], subjects=["g"],
                 random_blocks=[gm.RandomBlock(subjects=["g"])])
    assert r2.fit.beta == pytest.approx(mi.fe_params.to_numpy(), abs=1e-5)
    assert r2.fit.natural[0] == pytest.approx(float(mi.cov_re.iloc[0, 0]), rel=1e-4)
    cov = _table(r2.section, "Covariance Parameter Estimates")
    assert cov.rows[0][0] == "Residual Effect"
    r2t = _table(r2.section, "Pseudo R Squared")
    assert len(r2t.rows) == 4


def test_no_random_effects_is_a_glm():
    import statsmodels.api as sm

    rng = np.random.default_rng(1)
    n = 200
    x = rng.normal(size=n)
    c = rng.poisson(np.exp(0.3 + 0.6 * x))
    ds = Dataset(pd.DataFrame({"c": c.astype(float), "x": x}))
    r = gm.glmm(ds, "c", covariates=["x"], distribution="poisson", link="log")
    glm = sm.GLM(c, sm.add_constant(x), family=sm.families.Poisson()).fit()
    assert r.fit.beta == pytest.approx(glm.params, abs=1e-6)
    assert np.sqrt(np.diag(r.fit.C)) == pytest.approx(glm.bse, rel=1e-5)


def test_negative_binomial_ancillary_and_gamma():
    from statsmodels.discrete.discrete_model import NegativeBinomial

    rng = np.random.default_rng(4)
    ns, per = 30, 6
    g = np.repeat(np.arange(ns), per)
    x = rng.normal(size=ns * per)
    u = rng.normal(0, 0.5, ns)[g]
    mu = np.exp(1 + 0.4 * x + u)
    yn = rng.negative_binomial(2, 2 / (2 + mu))
    yg = rng.gamma(4, mu / 4)
    ds = Dataset(pd.DataFrame({"yn": yn.astype(float), "yg": yg, "x": x, "g": g * 1.0}))
    r = gm.glmm(ds, "yn", covariates=["x"], subjects=["g"], distribution="negbin",
                link="log", random_blocks=[gm.RandomBlock(subjects=["g"])])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nb = NegativeBinomial(yn, np.column_stack([np.ones(len(x)), x])).fit(disp=0)
    assert r.fit.k == pytest.approx(nb.params[-1], rel=1e-4)
    assert abs(r.fit.beta[1] - 0.4) < 0.2
    r2 = gm.glmm(ds, "yg", covariates=["x"], subjects=["g"], distribution="gamma",
                 link="log", random_blocks=[gm.RandomBlock(subjects=["g"])],
                 save=["predicted", "ci", "pearson"])
    assert abs(r2.fit.beta[1] - 0.4) < 0.1 and r2.fit.phi > 0
    assert r2.fit.natural[0] == pytest.approx(0.25, abs=0.2)
    assert set(r2.saved) == {"PredictedValue", "CI_Lower", "CI_Upper", "PearsonResidual"}
    lo, pv, hi = (r2.saved[k][1] for k in ("CI_Lower", "PredictedValue", "CI_Upper"))
    assert ((lo <= pv + 1e-9) & (pv <= hi + 1e-9)).all()
    r2t = _table(r2.section, "Pseudo R Squared")
    assert "trigamma" in r2t.footnotes[0]


def test_binary_reference_category_and_bayes_glmm():
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    rng = np.random.default_rng(21)
    ns, per = 40, 10
    g = np.repeat(np.arange(ns), per)
    x = rng.normal(size=ns * per)
    eta = -0.3 + 0.9 * x + rng.normal(0, 1.0, ns)[g]
    y = (rng.random(ns * per) < 1 / (1 + np.exp(-eta))).astype(float)
    df = pd.DataFrame({"y": y, "x": x, "g": g * 1.0})
    ds = Dataset(df)
    kw = dict(covariates=["x"], subjects=["g"], distribution="binomial", link="logit",
              random_blocks=[gm.RandomBlock(subjects=["g"])])
    last = gm.glmm(ds, "y", **kw)
    first = gm.glmm(ds, "y", reference="first", save=["predicted", "probability"], **kw)
    # SPSS default: the last category (1) is the reference, so P(y = 0) is modelled
    assert last.fit.beta == pytest.approx(-first.fit.beta, abs=1e-6)
    summary = _table(last.section, "Model Summary")
    assert _row(summary, "Target Category Modeled")[1].startswith("0")
    cls = _table(first.section, "Classification")
    assert cls.rows[-1][3].endswith("%")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bm = BinomialBayesMixedGLM.from_formula("y ~ x", {"g": "0 + C(g)"}, df).fit_vb()
    assert first.fit.beta == pytest.approx(bm.fe_mean, abs=0.2)
    assert first.saved["PredictedProbability"][1].between(0, 1).all()


def test_robust_and_multiple_blocks():
    rng = np.random.default_rng(8)
    sch = np.repeat(np.arange(10), 20)
    cls = np.repeat(np.arange(40), 5)
    x = rng.normal(size=200)
    y = 1 + 0.5 * x + rng.normal(0, 0.8, 10)[sch] + rng.normal(0, 0.5, 40)[cls] + \
        rng.normal(0, 1, 200)
    ds = Dataset(pd.DataFrame({"y": y, "x": x, "school": sch * 1.0, "cls": cls * 1.0}))
    blocks = [gm.RandomBlock(subjects=["school"]), gm.RandomBlock(subjects=["school", "cls"],
                                                                   predictions=True)]
    model = gm.glmm(ds, "y", covariates=["x"], subjects=["school", "cls"],
                    random_blocks=blocks)
    rob = gm.glmm(ds, "y", covariates=["x"], subjects=["school", "cls"],
                  random_blocks=blocks, robust=True)
    assert model.fit.beta == pytest.approx(rob.fit.beta)
    assert not np.allclose(np.diag(model.fit.C), np.diag(_coef_var(rob)))
    assert model.fit.natural[0] == pytest.approx(0.64, abs=0.6)
    assert any(getattr(t, "title", "").startswith("Random Effect 2 Predictions")
               for t in model.section.items)
    ds_tab = _table(model.section, "Data Structure")
    assert [r_[2] for r_ in ds_tab.rows] == ["10", "40"]


def _coef_var(result):
    coef = _table(result.section, "Fixed Coefficients")
    return np.array([float(r_[2]) ** 2 for r_ in coef.rows if r_[2]])


def test_selftest_runs():
    gm.selftest()

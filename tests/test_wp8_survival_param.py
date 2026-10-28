"""WP8: Parametric AFT and Parametric Shared Frailty models (SPSS SURVREG AFT / RECURRENT).

Reference values:
* IBM SPSS Advanced Statistics 31, "Parametric Shared Frailty Models - A Case Study for
  Recurrent Data" (data printed in the manual, ``data_wp8/side_effect.csv``): the frailty LR
  test p = 0.168, B[female=0] = 0.017, acceleration factor 1.017 with 95% CI (.688, 1.504).
* R ``survival::survreg(Surv(futime, fustat) ~ ecog.ps + rx, ovarian, dist = "weibull")``
  (the ovarian data of the survival package, ``data_wp8/ovarian.csv``): coefficients 6.897,
  -0.385, 0.529, Scale = 0.884, Loglik(model) = -97.1, Loglik(intercept only) = -98,
  Chisq = 1.74 on 2 df, p = 0.42. R-equivalent values, not SPSS output.
* lifelines (test-only; skipped when absent): Weibull / log-normal / log-logistic AFT fitters,
  interval and left censoring, left truncation (entry times).
* statsmodels: the exponential AFT equals a Poisson GLM with offset log(t) (signs reversed).
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import survival_param as sp

DATA = Path(__file__).parent / "data_wp8"


def _table(section, title):
    return next(t for t in section.items if getattr(t, "title", "").startswith(title))


def _row(table, label):
    return next(r for r in table.rows if r[0] == label)


@pytest.fixture(scope="module")
def ovarian():
    return Dataset(pd.read_csv(DATA / "ovarian.csv").astype(float))


@pytest.fixture(scope="module")
def side_effect():
    return Dataset(pd.read_csv(DATA / "side_effect.csv").astype(float))


# ---------------------------------------------------------------------------- AFT vs R survreg
def test_weibull_ovarian_matches_survreg(ovarian):
    r = sp.parametric_survival(ovarian, "futime", "fustat", ["ecog_ps", "rx"],
                               distribution="weibull")
    par = r.fit["par"]
    assert par[:3] == pytest.approx([6.8967, -0.3850, 0.5286], abs=5e-4)
    assert math.exp(par[3]) == pytest.approx(0.884, abs=5e-4)
    assert r.fit["ll"] == pytest.approx(-97.1, abs=0.05)
    omni = _table(r.section, "Omnibus Test")
    assert float(omni.rows[0][0]) == pytest.approx(1.74, abs=0.005) and omni.rows[0][1] == "2"
    assert float(omni.rows[0][2]) == pytest.approx(0.42, abs=0.005)
    pe = _table(r.section, "Parameter Estimates")
    assert _row(pe, "rx")[1] == ".529"
    assert _row(pe, "(Scale)")[1] == ".884"


def test_exponential_equals_poisson_offset(ovarian):
    import statsmodels.api as sm

    df = ovarian.df
    r = sp.parametric_survival(ovarian, "futime", "fustat", ["age", "ecog_ps"],
                               distribution="exponential")
    X = sm.add_constant(df[["age", "ecog_ps"]])
    glm = sm.GLM(df["fustat"], X, family=sm.families.Poisson(),
                 offset=np.log(df["futime"])).fit()
    assert r.fit["par"] == pytest.approx(-glm.params.to_numpy(), abs=1e-5)
    assert np.sqrt(np.diag(r.fit["cov"])) == pytest.approx(glm.bse.to_numpy(), rel=1e-4)


@pytest.mark.parametrize("dist,fitter", [("weibull", "WeibullAFTFitter"),
                                         ("lognormal", "LogNormalAFTFitter"),
                                         ("loglogistic", "LogLogisticAFTFitter")])
def test_aft_against_lifelines(ovarian, dist, fitter):
    lifelines = pytest.importorskip("lifelines")
    df = ovarian.df[["futime", "fustat", "age", "ecog_ps"]]
    r = sp.parametric_survival(ovarian, "futime", "fustat", ["age", "ecog_ps"],
                               distribution=dist)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = getattr(lifelines, fitter)().fit(df, "futime", "fustat")
    first = m.params_.index.get_level_values(0)[0]
    coef = m.params_[first]
    assert r.fit["par"][0] == pytest.approx(coef["Intercept"], abs=2e-3)
    assert r.fit["par"][1] == pytest.approx(coef["age"], abs=1e-4)
    assert r.fit["par"][2] == pytest.approx(coef["ecog_ps"], abs=2e-3)
    assert r.fit["ll"] == pytest.approx(m.log_likelihood_, abs=1e-5)


def test_interval_and_left_censoring_against_lifelines():
    lifelines = pytest.importorskip("lifelines")
    rng = np.random.default_rng(3)
    n = 150
    x = rng.normal(size=n)
    t = np.exp(1.5 + 0.4 * x + 0.5 * np.log(rng.exponential(size=n)))
    lo = np.floor(t)
    hi = lo + 1
    lo = np.where(lo == 0, 0.0, lo)
    df = pd.DataFrame({"lo": lo, "hi": hi, "x": x})
    ds = Dataset(df)
    r = sp.parametric_survival(ds, start="lo", end="hi", covariates=["x"],
                               distribution="weibull")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = lifelines.WeibullAFTFitter().fit_interval_censoring(df, "lo", "hi")
    assert r.fit["par"][1] == pytest.approx(m.params_["lambda_"]["x"], abs=1e-3)
    assert r.fit["par"][0] == pytest.approx(m.params_["lambda_"]["Intercept"], abs=1e-3)
    assert r.fit["ll"] == pytest.approx(m.log_likelihood_, abs=1e-3)
    cps = _table(r.section, "Case Processing Summary")
    assert cps.rows[2][2] == fmt_count((lo == 0).sum())       # left censored (start 0)
    # left censoring in Survival mode with a status variable
    c = np.quantile(t, 0.3)
    obs = np.maximum(t, c)
    st = np.where(t < c, 2.0, 1.0)            # 2 = left censored at c
    ds2 = Dataset(pd.DataFrame({"t": obs, "st": st, "x": x}))
    r2 = sp.parametric_survival(ds2, "t", "st", ["x"], status_map={"event": [1],
                                                                  "left": [2]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m2 = lifelines.WeibullAFTFitter().fit_left_censoring(
            pd.DataFrame({"t": obs, "e": (st == 1).astype(int), "x": x}), "t", "e")
    assert r2.fit["par"][1] == pytest.approx(m2.params_["lambda_"]["x"], abs=1e-3)
    assert r2.fit["ll"] == pytest.approx(m2.log_likelihood_, abs=1e-3)


def fmt_count(k):
    return str(int(k))


def test_left_truncation_against_lifelines():
    lifelines = pytest.importorskip("lifelines")
    rng = np.random.default_rng(9)
    n = 400
    x = rng.normal(size=n)
    t = np.exp(2 + 0.5 * x + 0.7 * np.log(rng.exponential(size=n)))
    entry = rng.uniform(0, 4, n)
    keep = t > entry
    c = entry + rng.exponential(15, n)
    obs = np.minimum(t, c)
    df = pd.DataFrame({"t": obs, "e": (t <= c).astype(float), "x": x, "entry": entry})[keep]
    r = sp.parametric_survival(Dataset(df.reset_index(drop=True)), "t", "e", ["x"],
                               truncation="entry")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = lifelines.WeibullAFTFitter().fit(df, "t", "e", entry_col="entry")
    assert r.fit["par"][1] == pytest.approx(m.params_["lambda_"]["x"], abs=1e-3)
    assert r.fit["ll"] == pytest.approx(m.log_likelihood_, abs=1e-3)


def test_string_status_factors_and_predictions(ovarian):
    df = ovarian.df.copy()
    df["st"] = np.where(df["fustat"] == 1, "F", "R")
    df.loc[0, "st"] = "X"                           # unmapped -> excluded by default
    ds = Dataset(df)
    ds.var("st").type = "string"
    r = sp.parametric_survival(ds, "futime", "st", ["age"], ["rx"],
                               reference={"rx": "first"}, print_encoding=True,
                               print_initial=True, print_history=True, cox_snell=True,
                               function_plots=("survival", "hazard", "density"),
                               separate="rx", predict=("survival", "hazard", "cumhazard"))
    cps = _table(r.section, "Case Processing Summary")
    assert any("unmapped" in row[1] for row in cps.rows)
    pe = _table(r.section, "Parameter Estimates")
    assert _row(pe, "[rx=1]")[1].startswith("0")            # first category is the baseline
    assert sum(1 for i in r.section.items if type(i).__name__ == "Image") == 4
    S = r.saved["PredSurvival"][1].dropna()
    Hc = r.saved["PredCumHazard"][1].dropna()
    assert np.allclose(S, np.exp(-Hc))
    assert len(S) == 25
    # regular intervals give one variable per period
    r2 = sp.parametric_survival(ovarian, "futime", "fustat", ["age"],
                                predict=("survival",), predict_times="regular",
                                interval=100, periods=3)
    assert set(r2.saved) == {"PredSurvival_1", "PredSurvival_2", "PredSurvival_3"}
    assert (r2.saved["PredSurvival_1"][1] >= r2.saved["PredSurvival_3"][1]).all()


def test_start_end_derived_status():
    rng = np.random.default_rng(5)
    n = 80
    t = rng.weibull(1.5, n) * 10
    lo = np.round(t, 0)
    hi = lo.copy()
    hi[:20] = np.inf                                  # right censored at lo
    hi[20:40] = lo[20:40] + 2                         # interval
    df = pd.DataFrame({"lo": np.maximum(lo, 0.5), "hi": np.where(np.isinf(hi), np.nan,
                                                                   np.maximum(hi, 0.5))})
    r = sp.parametric_survival(Dataset(df), start="lo", end="hi", distribution="lognormal",
                               status_treatment="derive")
    cps = _table(r.section, "Case Processing Summary")
    counts = {row[1]: int(row[2]) for row in cps.rows if row[0] == "Included"}
    assert counts["Right Censoring"] == 20 and counts["Interval Censoring"] == 20
    assert counts["Failure/Event"] == 40


# ---------------------------------------------------------------------------- frailty
def test_ibm_shared_frailty_case_study(side_effect):
    r = sp.parametric_survival(side_effect, "endTime", "sideEffect", ["age"], ["female"],
                               subject="patID", frailty="gamma", distribution="weibull",
                               function_plots=("survival", "hazard"), separate="female")
    pe = _table(r.section, "Parameter Estimates")
    row = _row(pe, "[female=0]")
    assert row[1] == ".017"
    assert row[8:11] == ["1.017", ".688", "1.504"]
    lr = _table(r.section, "Likelihood Ratio Test for the Frailty Component")
    assert lr.rows[0][4] == ".168"
    assert r.fit["theta"] == pytest.approx(0.6026, abs=2e-3)
    titles = [i.title for i in r.section.items if type(i).__name__ == "Image"]
    assert titles == ["Unconditional Survival Function", "Unconditional Hazard Function"]


def test_inverse_gaussian_laplace_derivatives():
    theta = 0.7
    dist = stats.invgauss(mu=theta, scale=1 / theta)      # mean 1, variance theta
    assert dist.mean() == pytest.approx(1) and dist.var() == pytest.approx(theta)
    for s_, d_ in ((0.3, 0), (1.2, 1), (2.5, 3), (0.8, 5)):
        val, _ = integrate.quad(lambda z: z ** d_ * math.exp(-z * s_) * dist.pdf(z), 0, np.inf)
        got = sp._lt_logderiv("invgauss", theta, np.array([s_]), np.array([d_]))[0]
        assert got == pytest.approx(math.log(val), abs=1e-7)
    gam = stats.gamma(a=1 / theta, scale=theta)
    val, _ = integrate.quad(lambda z: z ** 2 * math.exp(-z * 1.4) * gam.pdf(z), 0, np.inf)
    got = sp._lt_logderiv("gamma", theta, np.array([1.4]), np.array([2]))[0]
    assert got == pytest.approx(math.log(val), abs=1e-7)


def test_inverse_gaussian_frailty_and_predictions(side_effect):
    r = sp.parametric_survival(side_effect, "endTime", "sideEffect", ["age"], ["female"],
                               subject="patID", frailty="invgauss", distribution="loglogistic",
                               predict=("survival", "uncond_survival", "uncond_hazard",
                                        "uncond_cumhazard"))
    assert r.fit["theta"] > 0
    su = r.saved["PredUnCondSurvival"][1]
    hu = r.saved["PredUncondCumHazard"][1]
    assert np.allclose(su, np.exp(-hu))
    assert ((su > 0) & (su < 1)).all()


def test_frailty_vanishes_without_clustering():
    rng = np.random.default_rng(12)
    n = 200
    x = rng.normal(size=n)
    t = np.exp(1 + 0.5 * x + 0.5 * np.log(rng.exponential(size=n)))
    ds = Dataset(pd.DataFrame({"t": t, "x": x, "id": np.arange(n) // 2 * 1.0}))
    r = sp.parametric_survival(ds, "t", None, ["x"], subject="id", frailty="gamma")
    assert r.fit["lr_frailty"] < 3.84
    plain = sp.parametric_survival(ds, "t", None, ["x"])
    assert r.fit["par"][1] == pytest.approx(plain.fit["par"][1], abs=0.05)

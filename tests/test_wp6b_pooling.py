"""WP6b: automatic pooled output for multiply imputed data (Rubin's rules) in the analysis
dialogs, checked against hand-computed values and per-imputation statsmodels fits."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import descriptive, mi_pool
from rsstatchem.core.stats import imputation as mi
from rsstatchem.core.stats.common import fmt
from rsstatchem.core.stats.descriptive import StatSpec
from rsstatchem.ui.dialogs import compare as cdlg
from rsstatchem.ui.dialogs import genlin as gdlg
from rsstatchem.ui.dialogs import glm as glmdlg
from rsstatchem.ui.dialogs import logistic as ldlg
from rsstatchem.ui.dialogs import regression as rdlg
from rsstatchem.ui.dialogs import survival as sdlg
from rsstatchem.ui.dialogs.base import VariablePickerDialog


def _run(cls, ds, memory, opts=None):
    VariablePickerDialog._memory[cls.__name__] = memory
    VariablePickerDialog._states[cls.__name__] = opts or {}
    dlg = cls(ds)
    try:
        assert dlg.validate() is None
        return dlg.compute()
    finally:
        VariablePickerDialog._memory.pop(cls.__name__, None)
        VariablePickerDialog._states.pop(cls.__name__, None)
        dlg.deleteLater()


def _sec(r):
    return r.section if hasattr(r, "section") else r


def _table(sec, title):
    return next(t for t in sec.items if getattr(t, "kind", "") == "table" and t.title == title)


def _pooled_rows(table):
    start = next(i for i, r in enumerate(table.rows) if r[0] == "Pooled")
    return table.rows[start:]


# ============================================================================ Rubin's rules
def test_rubin_hand_computed_example():
    """Five completed-data estimates (van Buuren, FIMD 2nd ed., §2.3.2, eqs. 2.16-2.30)."""
    q = [10.0, 10.4, 9.8, 10.2, 10.6]
    u = [1.0, 1.2, 0.9, 1.1, 1.0]
    pl = mi.rubin(np.array(q)[:, None], np.array(u)[:, None])
    # by hand: Qbar = 10.2, Ubar = 1.04, B = 0.4/4 = 0.1, T = 1.04 + 1.2*0.1 = 1.16
    assert pl.estimate[0] == pytest.approx(10.2)
    assert pl.within[0] == pytest.approx(1.04)
    assert pl.between[0] == pytest.approx(0.1)
    assert pl.se[0] == pytest.approx(math.sqrt(1.16))
    # r = 1.2*0.1/1.04 = 0.115385; nu_old = 4 (1 + 1/r)^2 = 373.777...
    assert pl.riv[0] == pytest.approx(0.12 / 1.04)
    assert pl.df[0] == pytest.approx(4 * (1 + 1.04 / 0.12) ** 2)
    nu = 4 * (1 + 1.04 / 0.12) ** 2
    r = 0.12 / 1.04
    assert pl.fmi[0] == pytest.approx((r + 2 / (nu + 3)) / (r + 1))
    assert pl.re[0] == pytest.approx(1 / (1 + pl.fmi[0] / 5))
    # Barnard-Rubin with nu_com = 20: lambda = 0.12/1.16, nu_obs = 21/23*20*(1-lambda)
    br = mi.rubin(np.array(q)[:, None], np.array(u)[:, None], df_complete=20)
    lam = 0.12 / 1.16
    obs = 21 / 23 * 20 * (1 - lam)
    assert br.df[0] == pytest.approx(nu * obs / (nu + obs))


# ============================================================================ datasets
@pytest.fixture(scope="module")
def imputed():
    rng = np.random.default_rng(42)
    n = 150
    x = rng.normal(size=n)
    z = 0.5 * x + rng.normal(size=n)
    y = 1 + 0.8 * x - 0.4 * z + rng.normal(size=n)
    g = (rng.random(n) < 1 / (1 + np.exp(-(0.5 * x - 0.2)))).astype(float)
    grp = rng.integers(1, 4, n).astype(float)
    t = rng.exponential(np.exp(-0.4 * x))
    d = (rng.random(n) < 0.8).astype(float)
    cnt = rng.poisson(np.exp(0.3 + 0.4 * x)).astype(float)
    subj = np.repeat(np.arange(30), 5).astype(float)
    df = pd.DataFrame({"y": y, "x": x, "z": z, "g": g, "grp": grp, "t": t, "d": d,
                       "cnt": cnt, "subj": subj})
    miss = rng.random(n) < 0.25
    df.loc[miss, "z"] = np.nan
    df.loc[rng.random(n) < 0.1, "y"] = np.nan
    ds = Dataset(df)
    ds.var("g").measure = "nominal"
    ds.var("grp").measure = "nominal"
    res = mi.impute(ds, ["y", "x", "z"], m=5, seed=7)
    return res.dataset


def _views(ds):
    return [v for lb, v in mi.imputation_views(ds, include_original=False)]


def test_auto_views_requires_imputations(imputed):
    assert mi_pool.auto_views(imputed) is not None
    plain = Dataset(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
    assert mi_pool.auto_views(plain) is None
    one = Dataset(pd.DataFrame({"Imputation_": [0.0, 0, 1, 1], "x": [1.0, 2, 3, 4]}))
    assert mi_pool.auto_views(one) is None          # a single imputation is not pooled


def test_linear_regression_pooled_coefficients(qapp, imputed):
    res = _run(rdlg.LinearRegressionDialog, imputed, {"dep": ["y"]},
               {"blocks": [["enter", ["x", "z"]]], "assumptions": [], "plots": []})
    t = _table(res, "Coefficients (Pooled)")
    q, u, dfr = [], [], None
    for v in _views(imputed):
        f = sm.OLS(v.df["y"], sm.add_constant(v.df[["x", "z"]])).fit()
        q.append(f.params.to_numpy())
        u.append(f.bse.to_numpy() ** 2)
        dfr = f.df_resid
    pl = mi.rubin(np.array(q), np.array(u), df_complete=dfr)
    rows = _pooled_rows(t)
    for i in range(3):
        b_col = t.columns.index("B")
        assert rows[i][b_col] == fmt(pl.estimate[i])
        assert rows[i][b_col + 1] == fmt(pl.se[i])
        assert float(rows[i][b_col + 3]) == pytest.approx(pl.df[i], abs=0.01)
        assert rows[i][-3] == fmt(pl.fmi[i], 3)
    assert any("Imputation Number = 3" in getattr(i, "title", "") for i in res.items)


def test_logistic_and_genlin_pooled(qapp, imputed):
    res = _run(ldlg.BinaryLogisticDialog, imputed, {"dep": ["g"]},
               {"blocks": [["enter", ["x", "z"]]], "hosmer": False})
    t = _table(res, "Variables in the Equation (Pooled)")
    q, u = [], []
    for v in _views(imputed):
        f = sm.Logit(v.df["g"], np.column_stack([v.df[["x", "z"]], np.ones(len(v.df))])
                     ).fit(disp=0)
        q.append(f.params)
        u.append(f.bse ** 2)
    pl = mi.rubin(np.array(q), np.array(u))
    rows = _pooled_rows(t)
    assert [r[t.columns.index("B")] for r in rows] == [fmt(b) for b in pl.estimate]
    assert rows[0][t.columns.index("Exp(B)")] == fmt(math.exp(pl.estimate[0]))
    res = _run(gdlg.GenlinDialog, imputed, {"dep": ["cnt"], "covs": ["x", "z"]},
               {"distribution": "poisson", "link": "log"})
    t = _table(res, "Parameter Estimates (Pooled)")
    q, u = [], []
    for v in _views(imputed):
        f = sm.GLM(v.df["cnt"], sm.add_constant(v.df[["x", "z"]]),
                   family=sm.families.Poisson()).fit()
        q.append(f.params.to_numpy())
        u.append(f.bse.to_numpy() ** 2)
    pl = mi.rubin(np.array(q), np.array(u))
    assert [r[t.columns.index("B")] for r in _pooled_rows(t)] == [fmt(b) for b in pl.estimate]


def test_cox_and_mixed_pooled(qapp, imputed):
    res = _run(sdlg.CoxRegressionDialog, imputed,
               {"time": ["t"], "status": ["d"], "cov": ["x", "z"]},
               {"event": {"spec": ["value", [1.0]], "text": "1"}})
    t = _table(res, "Variables in the Equation (Pooled)")
    assert _pooled_rows(t)[0][0] == "Pooled" and len(_pooled_rows(t)) == 2
    res = _run(glmdlg.LinearMixedDialog, imputed, {"dep": ["y"], "subject": ["subj"],
                                                   "covariates": ["x", "z"]},
               {"intercept": True, "estimates": True})
    t = _table(_sec(res), "Estimates of Fixed Effects (Pooled)")
    assert [r[1] for r in _pooled_rows(t)] == ["Intercept", "x", "z"]


def test_simple_procedures_pooled(qapp, imputed):
    sec = descriptive.descriptives(imputed, ["y", "z"], StatSpec(["mean", "std"]))
    t = _table(sec, "Descriptive Statistics")      # WP4 pooled table first
    pooled = [r for r in t.rows if r[1] == "Pooled"]
    means = [numeric_mean(v, "z") for v in _views(imputed)]
    assert pooled[1][5] == fmt(np.mean(means), 4)
    sec = descriptive.frequencies(imputed, ["z"], StatSpec(["mean"]), False)
    t = _table(sec, "Statistics (Pooled)")
    assert t.rows[-1][1] == "Pooled" and t.rows[-1][3] == fmt(np.mean(means), 4)
    sec = _run(cdlg.OneSampleTDialog, imputed, {"vars": ["z"]}, {"test_value": 0.0})
    assert any(r[1] == "Pooled" for r in _table(sec, "One-Sample Test").rows)
    sec = _run(cdlg.OneWayAnovaDialog, imputed, {"deps": ["z"], "factor": ["grp"]})
    t = _table(sec, "Descriptives (Pooled)")
    total = [r for r in t.rows if r[2] == "Pooled"][-1]
    assert total[4] == fmt(np.mean(means), 4)
    sec = _run(rdlg.PartialDialog, imputed, {"vars": ["y", "z"], "controls": ["x"]})
    t = _table(sec, "Correlations (Pooled)")
    zs, vs = [], []
    for v in _views(imputed):
        res_y = sm.OLS(v.df["y"], sm.add_constant(v.df["x"])).fit().resid
        res_z = sm.OLS(v.df["z"], sm.add_constant(v.df["x"])).fit().resid
        zs.append(np.arctanh(np.corrcoef(res_y, res_z)[0, 1]))
        vs.append(1 / (len(v.df) - 4))
    pl = mi.rubin(np.array(zs)[:, None], np.array(vs)[:, None])
    assert t.rows[-1][2] == fmt(np.tanh(pl.estimate[0]))
    sec = _run(ldlg.MeansDialog, imputed, {"deps": ["z"]}, {"layers": [["grp"]]})
    assert any(r[2] == "Pooled" for r in _table(sec, "Report (Pooled)").rows)
    sec = _run(rdlg.BivariateDialog, imputed, {"vars": ["y", "z"]})
    assert any(r[0] == "Pooled" for r in _table(sec, "Correlations").rows)


def numeric_mean(view, name):
    return float(pd.to_numeric(view.df[name]).mean())


def test_bootstrap_is_skipped_when_pooling(qapp, imputed):
    sec = _run(cdlg.OneSampleTDialog, imputed, {"vars": ["z"]},
               {"bootstrap": {"perform": True, "samples": 50}})
    assert any("not performed" in getattr(i, "text", "") for i in sec.items)


def test_pooled_df_t_distribution(imputed):
    """Pooled one-sample t: p from the t distribution with Rubin's df."""
    sec = mi.pooled_one_sample_t(imputed, ["z"])
    row = next(r for r in _table(sec, "One-Sample Test").rows if r[1] == "Pooled")
    t, df, p = float(row[2]), float(row[3]), float(row[4])
    assert p == pytest.approx(2 * stats.t.sf(abs(t), df), abs=1.5e-3)

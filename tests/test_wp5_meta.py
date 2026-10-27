"""Meta-analysis (WP5) against metafor's published results (BCG vaccine data, Viechtbauer 2005
estimator comparison) and statsmodels."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.contingency_tables import StratifiedTable
from statsmodels.stats.meta_analysis import combine_effects

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import meta as mt

BCG = mt.BCG


@pytest.fixture(scope="module")
def bcg_rr():
    return mt.binary_effects(BCG.tpos, BCG.tneg, BCG.cpos, BCG.cneg, "rr")


def test_escalc_log_rr(bcg_rr):
    # metafor escalc(measure="RR") printed values
    y, v = bcg_rr
    assert np.round(y, 4).tolist() == [-0.8893, -1.5854, -1.3481, -1.4416, -0.2175, -0.7861,
                                       -1.6209, 0.0120, -0.4694, -1.3713, -0.3394, 0.4459,
                                       -0.0173]
    assert np.round(v, 4).tolist() == [0.3256, 0.1946, 0.4154, 0.0200, 0.0512, 0.0069, 0.2230,
                                       0.0040, 0.0564, 0.0730, 0.0124, 0.5325, 0.0714]


def test_random_effects_reml_bcg(bcg_rr):
    # metafor: tau^2 = 0.3132, I^2 = 92.22%, H^2 = 12.86, Q(12) = 152.2330,
    # estimate -0.7145 (se 0.1798, z -3.9744), CI [-1.0669, -0.3622]; predict(exp) CR .15-1.55
    f = mt.fit(*bcg_rr, model="random", method="REML")
    assert f.tau2 == pytest.approx(0.3132, abs=5e-5)
    assert f.i2 == pytest.approx(92.22, abs=5e-3)
    assert f.h2 == pytest.approx(12.86, abs=5e-3)
    assert f.q == pytest.approx(152.2330, abs=5e-4)
    assert f.beta[0] == pytest.approx(-0.7145, abs=5e-5)
    assert f.se[0] == pytest.approx(0.1798, abs=5e-5)
    assert f.stat()[0] == pytest.approx(-3.9744, abs=5e-4)
    lo, hi = f.ci()
    assert (lo[0], hi[0]) == pytest.approx((-1.0669, -0.3622), abs=5e-5)
    plo, phi = f.prediction()
    assert (np.exp(plo), np.exp(phi)) == pytest.approx((0.15, 1.55), abs=5e-3)


def test_fixed_and_dl_vs_statsmodels(bcg_rr):
    y, v = bcg_rr
    sm = combine_effects(y, v, method_re="dl")
    dl = mt.fit(y, v, None, "random", "DL")
    fe = mt.fit(y, v, None, "fixed")
    assert dl.tau2 == pytest.approx(sm.tau2)
    assert dl.beta[0] == pytest.approx(sm.mean_effect_re)
    assert dl.se[0] == pytest.approx(sm.sd_eff_w_re)
    assert fe.beta[0] == pytest.approx(sm.mean_effect_fe)
    assert fe.beta[0] == pytest.approx(-0.4303, abs=5e-5)


EX1 = ([-0.581, 0.530, 0.771, 1.031, 0.553, 0.295, 0.078, 0.573, -0.176, -0.232],
       [0.023, 0.052, 0.060, 0.115, 0.095, 0.203, 0.200, 0.211, 0.051, 0.040])
EX2 = ([0.100, -0.162, -0.090, -0.049, -0.046, -0.010, -0.431, -0.261, 0.134, 0.019, 0.175,
        0.056, 0.045, 0.103, 0.121, -0.482, 0.290, 0.342],
       [0.016, 0.015, 0.050, 0.050, 0.032, 0.052, 0.036, 0.024, 0.034, 0.033, 0.031, 0.034,
        0.039, 0.167, 0.134, 0.096, 0.016, 0.035])
# metafor-project.org "analyses:viechtbauer2005": tau2, I2, H2 per estimator
REF1 = {"HS": (0.228, 77.23, 4.39), "HE": (0.148, 68.80, 3.21), "DL": (0.277, 80.44, 5.11),
        "ML": (0.197, 74.51, 3.92), "REML": (0.223, 76.84, 4.32), "EB": (0.192, 74.05, 3.85),
        "SJ": (0.199, 74.75, 3.96)}
REF2 = {"HS": (0.010, 22.93, 1.30), "HE": (0.000, 0.00, 1.00), "DL": (0.013, 27.53, 1.38),
        "ML": (0.013, 28.45, 1.40), "REML": (0.016, 32.02, 1.47), "EB": (0.010, 23.72, 1.31),
        "SJ": (0.025, 42.67, 1.74)}


@pytest.mark.parametrize("data,ref", [(EX1, REF1), (EX2, REF2)])
def test_tau2_estimators_viechtbauer2005(data, ref):
    for method, (tau2, i2, h2) in ref.items():
        f = mt.fit(*data, model="random", method=method)
        assert round(f.tau2, 3) == pytest.approx(tau2, abs=1e-9), method
        assert round(f.i2, 2) == pytest.approx(i2, abs=1e-9), method
        assert round(f.h2, 2) == pytest.approx(h2, abs=1e-9), method


def test_meta_regression_bcg(bcg_rr):
    # metafor: rma(yi, vi, mods = ~ alloc + ablat) (REML), reference level "alternate"
    y, v = bcg_rr
    X = np.column_stack([np.ones(13), (BCG.alloc == "random").astype(float),
                         (BCG.alloc == "systematic").astype(float), BCG.ablat])
    f = mt.fit(y, v, X, "random", "REML")
    assert f.tau2 == pytest.approx(0.1446, abs=5e-5)
    assert f.beta == pytest.approx([0.2932, -0.2675, 0.0585, -0.0273], abs=5e-5)
    assert f.se == pytest.approx([0.4050, 0.3504, 0.3795, 0.0092], abs=5e-5)
    assert f.qm == pytest.approx(11.0605, abs=2e-4)
    assert f.q == pytest.approx(26.2034, abs=5e-4)
    assert f.i2 == pytest.approx(70.11, abs=5e-3)
    assert f.h2 == pytest.approx(3.35, abs=5e-3)
    assert f.r2 == pytest.approx(53.84, abs=5e-3)
    # the same through the dataset-level procedure (factor coded with "alternate" first)
    ds = Dataset(pd.DataFrame({"yi": y, "sei": np.sqrt(v), "ablat": BCG.ablat,
                               "alloc": BCG.alloc.map({"alternate": 1.0, "random": 2.0,
                                                       "systematic": 3.0})}))
    sec = mt.meta_regression(ds, "yi", "sei", ["ablat"], ["alloc"])
    pe = sec.items[1]
    assert pe.title == "Parameter Estimates"
    assert pe.rows[-1][1] == "-.027"


def test_mantel_haenszel_vs_statsmodels():
    tabs = [np.array([[a, b], [c, d]]) for a, b, c, d in zip(BCG.tpos, BCG.tneg, BCG.cpos,
                                                                BCG.cneg)]
    st = StratifiedTable(tabs)
    est, se = mt.mantel_haenszel(BCG.tpos, BCG.tneg, BCG.cpos, BCG.cneg, "or")
    assert est == pytest.approx(np.log(st.oddsratio_pooled))
    assert se == pytest.approx(st.logodds_pooled_se)
    est, _ = mt.mantel_haenszel(BCG.tpos, BCG.tneg, BCG.cpos, BCG.cneg, "rr")
    assert est == pytest.approx(np.log(st.riskratio_pooled))


def test_effect_sizes_continuous_and_peto():
    d, vd = mt.continuous_effects([20], [10], [2], [20], [9], [2], "cohen")
    assert d[0] == pytest.approx(0.5)
    assert vd[0] == pytest.approx(40 / 400 + 0.25 / 80)
    g, vg = mt.continuous_effects([20], [10], [2], [20], [9], [2], "hedges")
    j = 1 - 3 / (4 * 38 - 1)
    assert g[0] == pytest.approx(j * 0.5) and vg[0] == pytest.approx(j ** 2 * vd[0])
    gl, _ = mt.continuous_effects([20], [10], [3], [20], [9], [2], "glass")
    assert gl[0] == pytest.approx(0.5)
    y, v = mt.binary_effects([5], [15], [10], [10], "peto")
    n, e = 40, 20 * 15 / 40
    var = 20 * 20 * 15 * 25 / (40 ** 2 * 39)
    assert y[0] == pytest.approx((5 - e) / var) and v[0] == pytest.approx(1 / var)
    y0, _ = mt.binary_effects([0], [10], [3], [7], "or")
    assert y0[0] == pytest.approx(np.log(0.5 * 7.5 / (10.5 * 3.5)))


def test_bias_trimfill_subgroups(bcg_rr):
    y, v = bcg_rr
    coef, se, z, p, df_t, _ = mt.egger_test(y, v)
    assert (z, p) == pytest.approx((-0.8033, 0.4218), abs=5e-4)
    tf = mt.trim_and_fill(y, v)
    assert tf.k0 == 1 and tf.side == "right"
    # symmetric data -> nothing to fill
    ys = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, -0.2, 0.2])
    vs = np.array([0.1, 0.2, 0.05, 0.2, 0.1, 0.3, 0.3])
    assert mt.trim_and_fill(ys, vs, side="left").k0 == 0
    loo = mt.leave_one_out(y, v)
    assert len(loo) == 13
    cum = mt.cumulative(y, v)
    assert cum[-1].beta[0] == pytest.approx(mt.fit(y, v).beta[0])


def test_full_procedure_output():
    ds = Dataset(BCG.drop(columns=["alloc"]))
    res = mt.meta_analysis(ds, "bin_raw", {"a": "tpos", "b": "tneg", "c": "cpos", "d": "cneg"},
                           "rr", mt.MetaOptions(egger=True, begg=True, trimfill=True,
                                                cumulative=True, leave_one_out=True,
                                                knapp="truncated",
                                                plots=("forest", "funnel", "galbraith",
                                                       "labbe"), save=("ES", "SE", "WT")),
                           study="author", order="year")
    titles = [getattr(i, "title", "") for i in res.section.items]
    for t in ("Effect Size Estimatesᵃ ᵇ", "Heterogeneity", "Trim-and-Fill Analysis",
              "Egger's Regression-Based Test", "Cumulative Effect Size Estimates",
              "Leave-One-Out Analysis", "Forest Plot", "Funnel Plot", "L'Abbé Plot"):
        assert t in titles, t
    est = res.section.items[titles.index("Effect Size Estimatesᵃ ᵇ")]
    assert est.columns[3] == "t"           # Knapp-Hartung -> t test
    assert set(res.saved) == {"ES", "ES_SE", "ES_WT"}
    assert res.saved["ES"][1].notna().sum() == 13
    mt.selftest()


def test_subgroup_analysis():
    y, v = mt.binary_effects(BCG.tpos, BCG.tneg, BCG.cpos, BCG.cneg, "rr")
    ds = Dataset(pd.DataFrame({"y": y, "se": np.sqrt(v), "g": BCG.alloc.map(
        {"random": 1.0, "alternate": 2.0, "systematic": 3.0})}))
    res = mt.meta_analysis(ds, "cont_es", {"es": "y", "se": "se"}, "es",
                           mt.MetaOptions(plots=("forest",)), group="g")
    t = next(i for i in res.section.items if getattr(i, "title", "") ==
             "Test of Subgroup Differences")
    sub = [mt.fit(y[(BCG.alloc == a).to_numpy()], v[(BCG.alloc == a).to_numpy()])
           for a in ("random", "alternate", "systematic")]
    b = np.array([s.beta[0] for s in sub])
    w = np.array([1 / s.cov[0, 0] for s in sub])
    qb = float((w * (b - (w * b).sum() / w.sum()) ** 2).sum())
    assert float(t.rows[0][1]) == pytest.approx(qb, abs=5e-4)

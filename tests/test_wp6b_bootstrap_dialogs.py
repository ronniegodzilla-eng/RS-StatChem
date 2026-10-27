"""WP6b: the Bootstrap… button in every SPSS procedure that offers it — output structure
(Bootstrap Specifications, bootstrap rows/columns, "Bootstrap for …" tables, footnotes),
values against scipy.stats.bootstrap, stratified sampling and timing."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import descriptive
from rsstatchem.core.stats.common import fmt, fmt_p
from rsstatchem.core.stats.descriptive import StatSpec
from rsstatchem.ui.dialogs import classify as kdlg
from rsstatchem.ui.dialogs import compare as cdlg
from rsstatchem.ui.dialogs import descriptive as ddlg
from rsstatchem.ui.dialogs import explore as edlg
from rsstatchem.ui.dialogs import genlin as gdlg
from rsstatchem.ui.dialogs import logistic as ldlg
from rsstatchem.ui.dialogs import regression as rdlg
from rsstatchem.ui.dialogs import survival as sdlg
from rsstatchem.ui.dialogs.base import VariablePickerDialog
from rsstatchem.ui.dialogs.resample_ui import BootstrapDialog

BOOT = {"perform": True, "samples": 1000, "use_seed": True, "seed": 2000000, "level": 95.0,
        "ci": "percentile", "sampling": "simple", "strata": []}


def _run(cls, ds, memory, opts=None):
    VariablePickerDialog._memory[cls.__name__] = memory
    VariablePickerDialog._states[cls.__name__] = {"bootstrap": dict(BOOT), **(opts or {})}
    dlg = cls(ds)
    try:
        assert any(b.text() == "Bootstrap..." for b in dlg.findChildren(
            type(dlg.ok_button)))
        assert dlg.validate() is None
        t0 = time.perf_counter()
        out = dlg.compute()
        assert time.perf_counter() - t0 < 15
        return out.section if hasattr(out, "section") else out
    finally:
        VariablePickerDialog._memory.pop(cls.__name__, None)
        VariablePickerDialog._states.pop(cls.__name__, None)
        dlg.deleteLater()


def _table(sec, title):
    return next(t for t in sec.items if getattr(t, "kind", "") == "table" and t.title == title)


def _has_spec(sec):
    t = _table(sec, "Bootstrap Specifications")
    assert ["Number of Samples", "1000"] in t.rows
    return t


def _footnoted(t):
    return any("bootstrap results are based on 1000 bootstrap samples" in f
               for f in t.footnotes)


@pytest.fixture(scope="module")
def ds():
    rng = np.random.default_rng(10)
    n = 120
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    grp = np.repeat([1.0, 2.0, 3.0], n // 3)
    y = 2 + 0.6 * x1 - 0.3 * x2 + 0.5 * grp + rng.normal(size=n)
    yb = (rng.random(n) < 1 / (1 + np.exp(-x1))).astype(float)
    yo = np.digitize(y, [2.5, 3.5]).astype(float)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "grp": grp, "yb": yb, "yo": yo,
                       "g2": np.tile([1.0, 2.0], n // 2),
                       "cnt": rng.poisson(np.exp(0.5 + 0.3 * x1)).astype(float),
                       "t": rng.exponential(np.exp(-0.5 * x1)),
                       "d": (rng.random(n) < 0.85).astype(float)})
    df.loc[[3, 17], "x2"] = np.nan
    return Dataset(df)


def test_bootstrap_subdialog(qapp, ds):
    dlg = BootstrapDialog(ds, {"perform": True, "samples": 2000, "ci": "bca",
                               "sampling": "stratified", "strata": ["grp"]})
    v = dlg.values()
    assert v["perform"] and v["samples"] == 2000 and v["ci"] == "bca"
    assert v["sampling"] == "stratified" and v["strata"] == ["grp"]
    dlg.perform.setChecked(False)
    assert not dlg.body.isEnabled()


def test_one_sample_t_matches_scipy(qapp, ds):
    sec = _run(cdlg.OneSampleTDialog, ds, {"vars": ["y"]}, {"test_value": 2.0})
    _has_spec(sec)
    t = _table(sec, "Bootstrap for One-Sample Test")
    assert _footnoted(t)
    y = ds.df["y"].to_numpy()
    ref = stats.bootstrap((y,), lambda a, axis=-1: a.mean(axis=axis) - 2.0,
                          n_resamples=1000, method="percentile",
                          rng=np.random.default_rng(2000000))
    row = t.rows[0]
    assert row[-2] == fmt(ref.confidence_interval.low)
    assert row[-1] == fmt(ref.confidence_interval.high)
    assert row[3] == fmt(ref.standard_error)
    stats_t = _table(sec, "One-Sample Statistics")
    assert stats_t.rows[1][1] == "Bootstrapᵃ" and stats_t.rows[1][2] == "Bias"


def test_t_tests_and_anova(qapp, ds):
    sec = _run(cdlg.IndependentTDialog, ds, {"vars": ["y"], "group": ["g2"]},
               {"group_def": {"var": "g2", "groups": (1.0, 2.0), "cut": None}})
    t = _table(sec, "Bootstrap for Independent Samples Test")
    assert [r[1] for r in t.rows] == ["Equal variances assumed", "Equal variances not assumed"]
    assert len(_table(sec, "Group Statistics").rows) == 10
    sec = _run(cdlg.PairedTDialog, ds, {"pairs": [("y", "x1")]})
    assert _table(sec, "Bootstrap for Paired Samples Test").rows[0][0] == "Pair 1"
    sec = _run(cdlg.OneWayAnovaDialog, ds, {"deps": ["y"], "factor": ["grp"]},
               {"posthoc": ["tukey"]})
    mc = _table(sec, "Bootstrap for Multiple Comparisons")
    assert len(mc.rows) == 6 and _footnoted(mc)
    desc = _table(sec, "Descriptives")
    assert sum(1 for r in desc.rows if r[2] == "Bias") == 4          # 3 groups + Total


def test_descriptives_and_frequencies_via_editor_params(qapp, ds):
    dlg = ddlg.DescriptivesDialog(ds)
    type(dlg)._bootstrap = dict(BOOT)
    try:
        p = dlg.params()
    finally:
        type(dlg)._bootstrap = dict(ddlg.BOOT_DEFAULTS)
        dlg.deleteLater()
    sec = descriptive.descriptives(ds, ["y", "x2"], p["stats"])
    _has_spec(sec)
    t = _table(sec, "Descriptive Statistics")
    x2 = ds.df["x2"].to_numpy()
    rng = np.random.default_rng(2000000)
    idx = rng.integers(0, len(x2), (1000, len(x2)))
    sds = np.array([np.nanstd(x2[i], ddof=1) for i in idx])
    sd_col = next(c for c, col in enumerate(t.columns) if col == "Std. Deviation")
    se_row = next(i for i, r in enumerate(t.rows) if r[0] == "x2") + 2
    assert t.rows[se_row][sd_col] == fmt(sds.std(ddof=1), 4)
    sec = descriptive.frequencies(ds, ["y"], StatSpec(["mean", "median", "quartiles"],
                                                      {"bootstrap": BOOT}), False)
    t = _table(sec, "Statistics")
    assert sum(1 for r in t.rows if r[2] == "Bias") == 5      # mean, median, 3 quartiles


def test_correlations_explore_means_crosstabs(qapp, ds):
    sec = _run(rdlg.BivariateDialog, ds, {"vars": ["y", "x1"]}, {"spearman": True})
    tabs = [t for t in sec.items if getattr(t, "kind", "") == "table"
            and t.title == "Correlations"]
    assert all(any(r[-2] == "Bias" or "Bias" in r for r in t.rows) for t in tabs)
    y, x = ds.df["y"].to_numpy(), ds.df["x1"].to_numpy()
    ref = stats.bootstrap((y, x), lambda a, b, axis=-1: np.array(
        [np.corrcoef(a[i], b[i])[0, 1] for i in range(len(a))]) if a.ndim > 1 else
        np.corrcoef(a, b)[0, 1], paired=True, n_resamples=1000, method="percentile",
        rng=np.random.default_rng(2000000))
    lower = next(r for r in tabs[0].rows if r[2] == "95% CI Lower")
    assert lower[4] == fmt(ref.confidence_interval.low)
    sec = _run(rdlg.PartialDialog, ds, {"vars": ["y", "x1"], "controls": ["x2"]})
    assert any(r[3] == "Bias" for r in _table(sec, "Correlations").rows)
    sec = _run(edlg.ExploreDialog, ds, {"deps": ["y"], "factors": ["g2"]},
               {"display": "statistics"})
    t = _table(sec, "Descriptives")
    assert ("Bootstrapᵃ", "Bias") in t.columns
    sec = _run(ldlg.MeansDialog, ds, {"deps": ["y"]}, {"layers": [["grp"]]})
    assert any(r[2] == "Std. Error" for r in _table(sec, "Report").rows)
    sec = _run(edlg.CrosstabsDialog, ds, {"rows": ["grp"], "cols": ["g2"]},
               {"phi": True, "gamma": True})
    t = _table(sec, "Symmetric Measures")
    assert t.columns[-1][1] == "95% CI Upper"
    assert t.rows[0][-1] != ""


def test_model_procedures(qapp, ds):
    sec = _run(rdlg.LinearRegressionDialog, ds, {"dep": ["y"]},
               {"blocks": [["enter", ["x1"]], ["enter", ["x2"]]], "assumptions": [],
                "plots": []})
    t = _table(sec, "Bootstrap for Coefficients")
    assert [r[0] for r in t.rows] == ["1", "", "2", "", ""] and _footnoted(t)
    frame = ds.df[["y", "x1", "x2"]].dropna()
    X = np.column_stack([np.ones(len(frame)), frame[["x1", "x2"]]])
    yv = frame["y"].to_numpy()
    idx = np.random.default_rng(2000000).integers(0, len(yv), (1000, len(yv)))
    bs = np.array([np.linalg.lstsq(X[i], yv[i], rcond=None)[0] for i in idx])
    assert t.rows[3][4] == fmt(bs[:, 1].std(ddof=1))
    lo, hi = np.quantile(bs[:, 2], [0.025, 0.975])
    assert (t.rows[4][-2], t.rows[4][-1]) == (fmt(lo), fmt(hi))
    p = min(1, 2 * min((bs[:, 2] <= 0).mean(), (bs[:, 2] >= 0).mean()))
    assert t.rows[4][5] == fmt_p(p)
    for cls, memory, opts, title in [
        (ldlg.BinaryLogisticDialog, {"dep": ["yb"]}, {"blocks": [["enter", ["x1", "x2"]]]},
         "Bootstrap for Variables in the Equation"),
        (ldlg.MultinomialDialog, {"dep": ["yo"], "covs": ["x1"]}, {},
         "Bootstrap for Parameter Estimates"),
        (ldlg.OrdinalDialog, {"dep": ["yo"], "covs": ["x1"], "factors": ["g2"]}, {},
         "Bootstrap for Parameter Estimates"),
        (gdlg.GenlinDialog, {"dep": ["cnt"], "covs": ["x1"]},
         {"distribution": "poisson", "link": "log"}, "Bootstrap for Parameter Estimates"),
        (sdlg.CoxRegressionDialog, {"time": ["t"], "status": ["d"], "cov": ["x1", "x2"]},
         {"event": {"spec": ["value", [1.0]], "text": "1"}},
         "Bootstrap for Variables in the Equation"),
        (kdlg.DiscriminantDialog, {"group": ["grp"], "ind": ["x1", "y"]},
         {"group_def": {"var": "grp", "lo": 1.0, "hi": 3.0}, "fisher": True},
         "Bootstrap for Classification Function Coefficients"),
        (cdlg.UnivariateDialog, {"dep": ["y"], "factors": ["grp"], "covariates": ["x1"]},
         {"parameters": True}, "Bootstrap for Parameter Estimates"),
    ]:
        sec = _run(cls, ds, memory, opts)
        _has_spec(sec)
        t = _table(sec, title)
        assert t.rows and all(r[-1] != "" for r in t.rows), (cls.__name__, t.rows)


def test_stratified_and_bca(qapp, ds):
    sec = _run(cdlg.OneSampleTDialog, ds, {"vars": ["y"]},
               {"bootstrap": {**BOOT, "ci": "bca", "sampling": "stratified",
                              "strata": ["grp"]}})
    spec = _table(sec, "Bootstrap Specifications")
    assert ["Sampling Method", "Stratified"] in spec.rows
    assert ["Strata Variables", "grp"] in spec.rows
    assert ["Confidence Interval Type", "Bias-corrected and accelerated (BCa)"] in spec.rows
    t = _table(sec, "Bootstrap for One-Sample Test")
    assert "stratified bootstrap samples" in t.footnotes[-1]
    # stratified resampling keeps the group means' sampling variability within strata only
    y = ds.df["y"].to_numpy()
    grp = ds.df["grp"].to_numpy()
    rng = np.random.default_rng(2000000)
    groups = [np.flatnonzero(grp == g) for g in (1.0, 2.0, 3.0)]
    idx = np.hstack([gi[rng.integers(0, len(gi), (1000, len(gi)))] for gi in groups])
    means = y[idx].mean(axis=1)
    assert float(t.rows[0][3]) == pytest.approx(np.std(means, ddof=1), abs=5e-4)
    simple = y[np.random.default_rng(2000000).integers(0, len(y), (1000, len(y)))].mean(1)
    assert np.std(means, ddof=1) < np.std(simple, ddof=1)      # strata remove between-group

"""WP11 dialogs offscreen: every Data Preparation / Matching / Conjoint / Quality Control /
Direct Marketing / extension dialog builds, validates and computes, and the menus are merged
in SPSS order."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import validation
from rsstatchem.ui.dialogs import conjoint as cjd
from rsstatchem.ui.dialogs import dataprep as dpd
from rsstatchem.ui.dialogs import marketing as mkd
from rsstatchem.ui.dialogs import matching as mtd
from rsstatchem.ui.dialogs import misc_ext as mxd
from rsstatchem.ui.dialogs import qc as qcd
from rsstatchem.ui.dialogs.base import VariablePickerDialog


def _ds(n=120, seed=0):
    r = np.random.default_rng(seed)
    g = r.integers(0, 2, n).astype(float)
    df = pd.DataFrame({
        "id": np.arange(1, n + 1, dtype=float), "x": r.normal(10, 1, n) + g,
        "y": r.normal(size=n), "g": g, "c": r.choice([1.0, 2.0, 3.0], n),
        "sub": np.repeat(np.arange(1, n // 5 + 1), 5).astype(float),
        "d": r.poisson(3, n).astype(float), "t": r.weibull(1.5, n) * 100,
        "e": (r.random(n) < .8).astype(float), "zip": r.integers(10, 15, n) * 100.0,
        "date": pd.Timestamp("2024-01-01") + pd.to_timedelta(r.integers(0, 300, n),
                                                             unit="D")})
    ds = Dataset(df)
    ds.var("c").measure = "nominal"
    ds.var("g").measure = "nominal"
    return ds


def _run(cls, ds, memory, opts=None):
    VariablePickerDialog._memory[cls.__name__] = memory
    VariablePickerDialog._states[cls.__name__] = {**cls.DEFAULTS, **(opts or {})}
    dlg = cls(ds, None)
    try:
        assert dlg.validate() is None, dlg.validate()
        return dlg.compute()
    finally:
        VariablePickerDialog._memory.pop(cls.__name__, None)
        VariablePickerDialog._states.pop(cls.__name__, None)
        dlg.deleteLater()


def _titles(res):
    sec = res.section if hasattr(res, "section") else res
    return [getattr(i, "title", "") for i in sec.items]


@pytest.mark.parametrize("chart,memory,opts", [
    ("xbar_r", {"process": ["x"], "subgroup": ["sub"]},
     {"capability": True, "lsl": "7", "usl": "14", "rules": ["ucl", "lcl", "r8upper"]}),
    ("xbar_s", {"process": ["x"], "subgroup": ["sub"]}, {}),
    ("ir", {"process": ["x"], "label": ["id"]}, {"capability": True, "usl": "14"}),
    ("p", {"process": ["g"], "subgroup": ["sub"]}, {"nonconforming": "1"}),
    ("np", {"process": ["d"]}, {"organization": "subgroups", "sample_size": 20}),
    ("c", {"process": ["d"], "subgroup": ["sub"]}, {}),
    ("u", {"process": ["d"], "nsize": ["sub"]}, {"organization": "subgroups"}),
])
def test_control_charts_dialog(qapp, chart, memory, opts):
    res = _run(qcd.ControlChartsDialog, _ds(), memory, {"chart": chart, **opts})
    assert "Control Limits" in _titles(res)


def test_pareto_and_capability_dialogs(qapp):
    ds = _ds()
    assert "Pareto Table" in _titles(_run(qcd.ParetoDialog, ds, {"category": ["c"],
                                                                "stack": ["g"]}))
    res = _run(qcd.ProcessCapabilityDialog, ds, {"var": ["x"], "subgroup": ["sub"]},
               {"lsl": "6", "usl": "15"})
    assert "Process Statistics" in _titles(res)


def test_validation_dialogs(qapp):
    ds = _ds()
    dlg = dpd.LoadPredefinedRulesDialog(ds, None)
    dlg.compute()
    assert validation.rules(ds)["single"]
    dlg = dpd.DefineRulesDialog(ds, None)
    dlg._new_cross()
    dlg.value["cross"][-1]["expression"] = "g = 1 AND x < 9"
    dlg._new_single()
    assert dlg.validate() is None
    dlg.compute()
    names = [r["name"] for r in validation.rules(ds)["cross"]]
    res = _run(dpd.ValidateDataDialog, ds, {"vars": ["x", "c"], "ids": ["id"]},
               {"assign": {"c": ["Likert 1 to 5"]}, "cross": names, "save_total": True})
    assert "Cross-Variable Rules" in _titles(res) and ds.has_name("ValidationRuleViolations")


def test_unusual_cases_binning_adp_dialogs(qapp):
    ds = _ds()
    res = _run(dpd.UnusualCasesDialog, ds, {"vars": ["x", "y", "c"], "id": ["id"]},
               {"save_index": True, "use_cutoff": False})
    assert "Anomaly Case Index List" in _titles(res) and ds.has_name("AnomalyIndex")
    res = _run(dpd.OptimalBinningDialog, ds, {"vars": ["x", "y"], "guide": ["g"]},
               {"syntax": True})
    assert "Model Entropy" in _titles(res) and ds.has_name("x_bin")
    res = _run(dpd.PrepareDataDialog, ds, {"target": ["y"], "inputs": ["x", "c", "date"]},
               {"reference": "2025-01-01", "select": False})
    assert "Field Actions" in _titles(res) and ds.has_name("x_transformed")


def test_matching_dialogs(qapp):
    ds = _ds()
    res = _run(mtd.PropensityMatchingDialog, ds, {"group": ["g"], "pred": ["x", "y"],
                                                  "id": ["id"]}, {"tolerance": 0.05})
    assert "Matching Statistics" in _titles(res) and ds.has_name("MatchID")
    res = _run(mtd.CaseControlDialog, ds, {"group": ["g"], "keys": ["c", "x"]},
               {"tolerances": "x=0.5", "match_name": "CCID", "group_name": "CCG"})
    assert "Match Summary" in _titles(res) and ds.has_name("CCG")
    res = _run(mtd.RakeWeightsDialog, ds, {"vars": ["c", "g"]},
               {"targets": {"c": "1=0.3; 2=0.3; 3=0.4", "g": "0=0.5; 1=0.5"}})
    assert "Control Totals" in _titles(res) and ds.has_name("rakewt")


def test_conjoint_dialogs(qapp):
    cjd.GenerateDesignDialog._state = {"factors": [
        {"name": "price", "label": "Price", "values": [1.0, 2.0, 3.0],
         "labels": ["low", "mid", "high"]},
        {"name": "brand", "label": "", "values": [1.0, 2.0, 3.0], "labels": []},
        {"name": "seal", "label": "", "values": [1.0, 2.0], "labels": ["no", "yes"]}],
        "holdout": 2}
    dlg = cjd.GenerateDesignDialog(None, None)
    assert dlg.validate() is None
    sec = dlg.compute()
    plan = dlg.new_dataset
    cjd.GenerateDesignDialog._state = {}
    assert "Plan Cards" in _titles(sec) and plan.has_name("STATUS_")
    res = _run(cjd.DisplayDesignDialog, plan, {"factors": ["price", "brand", "seal"]},
               {"profiles": True})
    assert "Plan Cards" in _titles(res)
    r = np.random.default_rng(1)
    for k in range(4):
        plan.add_variable(f"s{k}", values=r.integers(1, 10, plan.n_cases).astype(float))
    res = _run(cjd.ConjointDialog, plan, {"factors": ["price", "brand", "seal"],
                                          "resp": ["s0", "s1", "s2", "s3"]},
               {"models": {"price": "linear_less"}, "individual": True})
    assert {"Utilities", "Importance Values", "Coefficients"} <= set(_titles(res))


def test_marketing_dialogs(qapp):
    ds = _ds()
    res = _run(mkd.RFMDialog, ds, {"id": ["zip"], "date": ["date"], "amount": ["t"]},
               {"reference": "2025-01-01"})
    assert "RFM Bins" in _titles(res)
    res = _run(mkd.RFMDialog, ds, {"date": ["d"], "freq": ["sub"], "amount": ["t"]},
               {"format": "customers"})
    assert ds.has_name("RFM_Score")
    assert "Auto-Clustering" in _titles(_run(mkd.DMClusterDialog, ds, {"fields": ["x", "c"]}))
    assert "Profiles" in _titles(_run(mkd.ProspectProfilesDialog, ds,
                                      {"resp": ["g"], "fields": ["c", "x"]},
                                      {"min_group": 20}))
    res = _run(mkd.PostalCodeDialog, ds, {"resp": ["g"], "postal": ["zip"]},
               {"profit": True})
    assert "Response Rates by Postal Code" in _titles(res)
    res = _run(mkd.PropensityPurchaseDialog, ds, {"resp": ["g"], "pred": ["x", "c"]})
    assert "Overall Model Quality" in _titles(res) and res.saved
    res = _run(mkd.ControlPackageDialog, ds, {"campaign": ["c"], "eff": ["g"]})
    assert "Comparison with the Control Package" in _titles(res)
    dlg = mkd.ChooseTechniqueDialog()
    assert dlg.selected() is mkd.RFMDialog


def test_extension_dialogs(qapp):
    ds = _ds()
    assert "Best Reach and Frequency by Group Size" in _titles(
        _run(mxd.TurfDialog, ds, {"vars": ["g", "e"]}))
    r = np.random.default_rng(2)
    Y = np.zeros((80, 2))
    for i in range(1, 80):
        Y[i] = 0.4 * Y[i - 1] + r.normal(size=2)
    dv = Dataset(pd.DataFrame(Y, columns=["a", "b"]))
    res = _run(mxd.VarDialog, dv, {"vars": ["a", "b"]}, {"forecast": 4, "save_resid": True})
    assert "Granger Causality Tests" in _titles(res) and res.saved
    assert "Weibull Plot" in _titles(_run(mxd.WeibullPlotDialog, ds,
                                          {"var": ["t"], "status": ["e"]},
                                          {"method": "mle"}))
    assert "Distance Correlations" in _titles(_run(mxd.DistanceCorrelationDialog, ds,
                                                   {"vars": ["x", "y"]}, {"n_perm": 99}))
    assert "Distance Correlations" in _titles(_run(
        mxd.DistanceCorrelationDialog, ds, {"vars": ["x", "y"], "yset": ["t"]},
        {"mode": "sets", "n_perm": 99}))
    res = _run(mxd.PermutationTestDialog, ds, {"vars": ["x"], "group": ["g"]},
               {"n_perm": 300, "groups": "0 1"})
    assert any("Permutation Test" in t for t in _titles(res))
    res = _run(mxd.PermutationTestDialog, ds, {"pairs": [("x", "y")]},
               {"test": "paired", "n_perm": 300})
    assert any("Paired" in t for t in _titles(res))
    res = _run(mxd.RegressionPermutationDialog, ds, {"dep": ["x"], "pred": ["y", "t"]},
               {"n_perm": 200})
    assert "Coefficients (Freedman-Lane Permutation)" in _titles(res)


def test_menus_in_spss_order(qapp):
    from rsstatchem.ui.data_editor import menu_tree
    from rsstatchem.ui.menus import _find_menu, _key, failed_extensions

    assert not failed_extensions()
    tree = menu_tree()

    def labels(path):
        return [_key(e[0]) for e in _find_menu(tree, path) if isinstance(e, tuple)]

    data = labels("Data")
    for a, b in (("validation", "identify unusual cases"),
                 ("identify unusual cases", "compare datasets"),
                 ("restructure", "rake weights"), ("rake weights", "propensity score matching"),
                 ("propensity score matching", "case control matching"),
                 ("aggregate", "orthogonal design")):
        assert data.index(a) < data.index(b), (a, b)
    assert labels("Data/Validation") == ["load predefined rules", "define rules",
                                         "validate data"]
    tr = labels("Transform")
    assert tr.index("visual binning") < tr.index("optimal binning") < \
        tr.index("prepare data for modeling")
    an = labels("Analyze")
    assert an.index("quality control") < an.index("direct marketing")
    assert labels("Analyze/Quality Control") == ["control charts", "pareto charts",
                                                 "process capability"]
    assert "turf analysis" in labels("Analyze/Descriptive Statistics")
    assert "vector autoregression" in labels("Analyze/Forecasting")
    corr = labels("Analyze/Correlate")
    assert corr.index("distances") < corr.index("distance correlation")
    assert "weibull plot" in labels("Graphs")

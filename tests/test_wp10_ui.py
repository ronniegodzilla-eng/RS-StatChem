"""WP10 dialogs offscreen: every Categories and Complex Samples dialog builds, validates and
computes; the menus land in SPSS order."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import complex_samples as cs
from rsstatchem.ui.dialogs import complex_samples as cd
from rsstatchem.ui.dialogs import optscale as od
from rsstatchem.ui.dialogs.base import VariablePickerDialog


def _ds(plan=True):
    rng = np.random.default_rng(1)
    n = 120
    y = rng.integers(1, 6, n).astype(float)
    df = pd.DataFrame(dict(y=y, a=np.clip(y + rng.integers(-1, 2, n), 1, 5),
                           b=rng.integers(1, 4, n).astype(float),
                           c=rng.integers(1, 6, n).astype(float),
                           st=np.repeat([1.0, 2.0, 3.0], 40),
                           psu=np.repeat(np.arange(24), 5).astype(float),
                           w=rng.uniform(1, 4, n), x=rng.normal(size=n),
                           t=rng.exponential(2, n) + 0.1,
                           e=(rng.uniform(size=n) < .7).astype(float)))
    ds = Dataset(df)
    if plan:
        cs.set_plan(ds, cs.Plan("w", [cs.Stage(["st"], ["psu"], "WR")]))
    return ds


def _run(cls, ds, memory, opts=None):
    VariablePickerDialog._memory[cls.__name__] = memory
    if opts:
        VariablePickerDialog._states[cls.__name__] = {**cls.DEFAULTS, **opts}
    dlg = cls(ds, None)
    try:
        assert dlg.validate() is None
        res = dlg.compute()
        return res.section if hasattr(res, "section") else res
    finally:
        VariablePickerDialog._memory.pop(cls.__name__, None)
        VariablePickerDialog._states.pop(cls.__name__, None)
        dlg.deleteLater()


def _titles(sec):
    return [getattr(i, "title", "") for i in sec.items]


CASES = [
    (od.CatregDialog, {"dep": ["y"], "ind": ["a", "b", "c"]}, None, "Coefficients"),
    (od.CatregDialog, {"dep": ["y"], "ind": ["a", "b", "c"]},
     {"regularization": "lasso", "lasso": [0, .4, .2], "resampling": "cv", "folds": 4,
      "tplots": ["a"], "save_pred": True}, "Lasso Regression"),
    (od.CatpcaDialog, {"vars": ["y", "a", "b", "c"]},
     {"plots": ["objects", "biplot", "joint"], "quant": ["a"]}, "Component Loadings"),
    (od.McaDialog, {"vars": ["a", "b", "c"]}, None, "Discrimination Measures"),
    (od.PrefscalDialog, {"vars": ["a", "b", "c", "y"]}, {"maxiter": 100}, "Measures"),
    (cd.CSFrequenciesDialog, {"vars": ["b"]}, {"statistics": ["se", "ci", "deff"]}, "b"),
    (cd.CSDescriptivesDialog, {"vars": ["x"], "subpop": ["b"]}, {"sum": True},
     "Univariate Statistics: Sum"),
    (cd.CSCrosstabsDialog, {"row": ["b"], "col": ["e"]}, None, "Tests of Independence"),
    (cd.CSRatiosDialog, {"num": ["t"], "den": ["w"]}, None, "Ratios"),
    (cd.CSGLMDialog, {"dep": ["x"], "factors": ["b"], "covariates": ["t"]}, None,
     "Tests of Model Effects"),
    (cd.CSLogisticDialog, {"dep": ["e"], "factors": ["b"], "covariates": ["x"]}, None,
     "Classification"),
    (cd.CSOrdinalDialog, {"dep": ["y"], "covariates": ["x"]}, {"link": "probit"},
     "Parameter Estimates"),
    (cd.CSCoxDialog, {"time": ["t"], "status": ["e"], "covariates": ["x"]}, None,
     "Parameter Estimates"),
]


@pytest.mark.parametrize("cls, memory, opts, title", CASES)
def test_dialogs_compute(qapp, cls, memory, opts, title):
    sec = _run(cls, _ds(), memory, opts)
    assert title in _titles(sec)


def test_overals_dialog_sets(qapp):
    ds = _ds()
    dlg = od.OveralsDialog(ds, None)
    target = next(t for t in dlg._targets if t.key == "vars")
    for n in ("a", "b"):
        dlg._take(n, target)
    dlg._go(1)
    assert dlg.set_label.text() == "Set 2 of 2"
    for n in ("c", "y"):
        dlg._take(n, target)
    assert dlg.validate() is None
    sec = dlg.compute().section
    assert "Summary of Analysis" in _titles(sec)
    dlg._go(-1)
    assert dlg.selected("vars") == ["a", "b"]
    dlg.deleteLater()


def test_define_scale_suffix_and_chooser(qapp):
    ds = _ds()
    VariablePickerDialog._memory["CatregDialog"] = {"dep": ["y"], "ind": ["a"]}
    dlg = od.CatregDialog(ds, None)
    widget = next(t.widget for t in dlg._targets if t.key == "ind")
    assert widget.item(0).text() == "a (Spline Ordinal 2 2)"
    dlg.opts["scale"]["a"] = {"level": "nomi"}
    dlg._refresh("ind")
    assert widget.item(0).text() == "a (Nominal)"
    dlg.deleteLater()
    VariablePickerDialog._memory.pop("CatregDialog", None)
    ch = od.OptimalScalingDialog(None)
    assert ch.choice() is od.McaDialog
    ch.some.setChecked(True)
    assert ch.choice() is od.CatpcaDialog
    ch.multi.setChecked(True)
    assert ch.choice() is od.OveralsDialog
    ch.deleteLater()


def test_plan_and_sampling_dialogs(qapp):
    ds = _ds(plan=False)
    sec = None
    VariablePickerDialog._memory["CSDescriptivesDialog"] = {"vars": ["x"]}
    dlg = cd.CSDescriptivesDialog(ds, None)
    assert "Prepare for Analysis" in dlg.validate()
    dlg.deleteLater()
    VariablePickerDialog._memory.pop("CSDescriptivesDialog", None)
    sec = _run(cd.PreparePlanDialog, ds, {"strata": ["st"], "clusters": ["psu"],
                                          "weight": ["w"]})
    assert "Plan Summary" in _titles(sec) and cs.get_plan(ds).weight == "w"
    bad = cd.PreparePlanDialog(ds, None)
    bad.opts["method"] = "EQUAL_WOR"
    assert "without-replacement" in bad.validate()
    bad.deleteLater()
    ds2 = _ds(plan=False)
    sec = _run(cd.SelectSampleDialog, ds2, {"strata": ["st"]}, {"size": 5.0})
    assert ds2.df["SampleWeight_Final_"].notna().sum() == 15
    assert cs.get_plan(ds2).weight == "SampleWeight_Final_"


def test_menus_registered_in_spss_order():
    from rsstatchem.ui import menus

    assert "rsstatchem.ui.dialogs.optscale" in menus.EXTENSIONS
    assert "rsstatchem.ui.dialogs.complex_samples" in menus.EXTENSIONS
    tree = menus.build_tree({"Analyze": []})
    analyze = tree["Analyze"]

    def sub(entries, name):
        return next(e[1] for e in entries if isinstance(e, tuple) and isinstance(e[1], list)
                    and e[0] == name)

    labels = [e[0] if isinstance(e, tuple) else e for e in sub(analyze, "Complex Samples")]
    assert labels == ["Prepare for Analysis...", "Select a Sample...", None,
                      "Frequencies...", "Descriptives...", "Crosstabs...", "Ratios...",
                      "General Linear Model...", "Logistic Regression...",
                      "Ordinal Regression...", "Cox Regression..."]
    assert any(isinstance(e, tuple) and e[0] == "Optimal Scaling (CATREG)..."
               for e in sub(analyze, "Regression"))
    assert any(isinstance(e, tuple) and e[0] == "Optimal Scaling..."
               for e in sub(analyze, "Dimension Reduction"))
    assert any(isinstance(e, tuple) and e[0] == "Multidimensional Unfolding (PREFSCAL)..."
               for e in sub(analyze, "Scale"))
    top = [e[0] if isinstance(e, tuple) else e for e in analyze]
    assert top.index("Complex Samples") > top.index("Survival")


def test_module_selftests():
    od.selftest()
    cd.selftest()

"""WP4 dialogs offscreen: every Survival / Missing Values / Multiple Imputation / Forecasting
dialog builds, validates and computes, and the menus are merged."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsstatchem.core.dataset import Dataset
from rsstatchem.ui.dialogs import missing as mdlg
from rsstatchem.ui.dialogs import survival as sdlg
from rsstatchem.ui.dialogs import timeseries as tdlg
from rsstatchem.ui.dialogs.base import VariablePickerDialog

DATA = Path(__file__).parent / "data_wp4"


def _run(cls, ds, memory=None, opts=None, parent=None):
    VariablePickerDialog._memory[cls.__name__] = memory or {}
    if opts:
        VariablePickerDialog._states[cls.__name__] = opts
    dlg = cls(ds, parent)
    try:
        if hasattr(dlg, "validate"):
            assert dlg.validate() is None
        return dlg.compute()
    finally:
        VariablePickerDialog._memory.pop(cls.__name__, None)
        VariablePickerDialog._states.pop(cls.__name__, None)
        dlg.deleteLater()


def _section(result):
    return result.section if hasattr(result, "section") else result


@pytest.fixture()
def surv_ds():
    g = pd.read_csv(DATA / "gehan.csv")
    g["rx"] = (g["treat"] == "control").astype(float)
    g["wbc"] = np.random.default_rng(1).normal(size=len(g))
    return Dataset(g[["time", "cens", "rx", "wbc"]])


def test_survival_dialogs(qapp, surv_ds):
    res = _run(sdlg.KaplanMeierDialog, surv_ds,
               {"time": ["time"], "status": ["cens"], "factor": ["rx"]},
               {"tests": ["logrank"], "plots": ["survival"], "save": ["survival"]})
    assert res.saved and _section(res).title == "Kaplan-Meier"
    sec = _run(sdlg.LifeTablesDialog, surv_ds,
               {"time": ["time"], "status": ["cens"], "factor": ["rx"]},
               {"compare": "overall", "plots": ["survival"]})
    assert sec.title == "Life Tables"
    res = _run(sdlg.CoxRegressionDialog, surv_ds,
               {"time": ["time"], "status": ["cens"], "cov": ["rx", "wbc"]},
               {"categorical": {"rx": ["indicator", "first"]}, "method": "fstep_wald",
                "plots": ["survival"], "separate": "rx", "save": ["survival", "xbeta"]})
    assert res.saved
    res = _run(sdlg.CoxTimeDependentDialog, surv_ds,
               {"time": ["time"], "status": ["cens"], "cov": ["rx"]},
               {"tcov": "wbc * LN(T_)"})
    VariablePickerDialog._states.pop("CoxTimeDependentDialog", None)
    assert "Time-Dependent" in _section(res).title


def test_missing_and_imputation_dialogs(qapp):
    rng = np.random.default_rng(3)
    n = 80
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                       "c": rng.integers(0, 2, n).astype(float)})
    df["y"] = df.a + df.b + df.c + rng.normal(size=n)
    df.loc[rng.random(n) < 0.2, "b"] = np.nan
    df.loc[rng.random(n) < 0.1, "c"] = np.nan
    ds = Dataset(df)
    ds.var("c").measure = "nominal"
    sec = _run(mdlg.MissingValueAnalysisDialog, ds, {"quant": ["a", "b", "y"], "cat": ["c"]},
               {"ttest": True, "pairwise": True})
    assert any(getattr(i, "title", "").startswith("EM Means") for i in sec.items)
    sec = _run(mdlg.AnalyzePatternsDialog, ds, {"vars": ["a", "b", "c", "y"]})
    assert sec.items
    VariablePickerDialog._memory["ImputeDialog"] = {"vars": ["a", "b", "c", "y"]}
    dlg = mdlg.ImputeDialog(ds, None)
    dlg.m_spin.setValue(3)
    assert dlg.validate() is None
    sec = dlg.compute()
    new = dlg.result_dataset
    dlg.deleteLater()
    assert new.n_cases == 4 * n and sec.title == "Multiple Imputation"
    for cls, memory in ((mdlg.PooledDescriptivesDialog, {"vars": ["b"]}),
                        (mdlg.PooledOneSampleTDialog, {"vars": ["b"]}),
                        (mdlg.PooledCorrelationsDialog, {"vars": ["a", "b"]}),
                        (mdlg.PooledLinearDialog, {"dep": ["y"], "ind": ["a", "b", "c"]}),
                        (mdlg.PooledLogisticDialog, {"dep": ["c"], "ind": ["a"]}),
                        (mdlg.PooledPairedTDialog, {"pairs": [("a", "b")]})):
        assert _run(cls, new, memory).items
    sec = _run(mdlg.PooledIndependentTDialog, new, {"vars": ["y"], "group": ["c"]},
               {"group_def": {"var": "c", "groups": (0.0, 1.0), "cut": None}})
    assert "Pooled" in str(sec.items[-1].rows)
    VariablePickerDialog._memory["PooledDescriptivesDialog"] = {"vars": ["b"]}
    dlg = mdlg.PooledDescriptivesDialog(ds, None)
    assert dlg.validate() is not None          # not an imputed dataset
    dlg.deleteLater()


def test_forecasting_dialogs(qapp):
    a = pd.read_csv(DATA / "airpassengers.csv")
    ds = Dataset(pd.DataFrame({"pass": a["value"].astype(float),
                               "x": np.random.default_rng(2).normal(size=len(a))}))
    dlg = tdlg.DefineDatesDialog(ds)
    dlg.first.setValue(1949)
    dlg.second.setValue(1)
    dlg.compute()
    dlg.deleteLater()
    assert ds.has_name("DATE_")
    sec = _run(tdlg.CreateTimeSeriesDialog, ds, {"vars": ["pass"]}, {"func": "DIFF"})
    assert ds.has_name("pass_1") and sec.title == "Create Time Series"
    for cls, memory in ((tdlg.SequenceChartsDialog, {"vars": ["pass"], "axis": ["DATE_"]}),
                        (tdlg.AutocorrelationsDialog, {"vars": ["pass"]}),
                        (tdlg.CrossCorrelationsDialog, {"vars": ["pass", "x"]}),
                        (tdlg.SpectralDialog, {"vars": ["pass"]})):
        assert _run(cls, ds, memory).items
    res = _run(tdlg.SeasonalDecompositionDialog, ds, {"vars": ["pass"]})
    assert set(res.saved) == {"ERR", "SAS", "SAF", "STC"}
    sec = _run(tdlg.TimeSeriesModelerDialog, ds, {"dep": ["pass"]},
               {"method": "arima", "p": 0, "d": 1, "q": 1, "sp": 0, "sd": 1, "sq": 1,
                "transform": "log", "constant": False, "store": True, "horizon": 6})
    assert any(getattr(i, "title", "") == "ARIMA Model Parameters" for i in sec.items)
    dlg = tdlg.ApplyModelsDialog(ds)
    assert dlg.list.count() == 1
    sec = dlg.compute()
    dlg.deleteLater()
    assert sec.title == "Apply Time Series Models"


def test_menus_merged(qapp):
    from rsstatchem.ui import menus
    from rsstatchem.ui.data_editor import menu_tree

    tree = menu_tree()
    analyze = {e[0]: e[1] for e in tree["Analyze"] if isinstance(e, tuple)}
    for title in ("Survival", "Multiple Imputation", "Forecasting"):
        assert isinstance(analyze[title], list)
    assert analyze["Missing Value Analysis..."] is mdlg.MissingValueAnalysisDialog
    assert any(isinstance(e, tuple) and e[0] == "Create Time Series..." for e in
               tree["Transform"])
    for name in ("rsstatchem.ui.dialogs.survival", "rsstatchem.ui.dialogs.missing",
                 "rsstatchem.ui.dialogs.timeseries"):
        assert name not in menus.failed_extensions()

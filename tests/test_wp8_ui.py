"""WP8 dialogs offscreen: Parametric AFT, Parametric Shared Frailty and Generalized Linear
Mixed Models build, validate and compute, and land in the SPSS menu positions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsstatchem.core.dataset import Dataset
from rsstatchem.ui.dialogs import glmm as gdlg
from rsstatchem.ui.dialogs import survival_param as sdlg
from rsstatchem.ui.dialogs.base import VariablePickerDialog

DATA = Path(__file__).parent / "data_wp8"


def _run(cls, ds, memory=None, opts=None, check=None):
    VariablePickerDialog._memory[cls.__name__] = memory or {}
    if opts:
        VariablePickerDialog._states[cls.__name__] = opts
    dlg = cls(ds)
    try:
        if check:
            check(dlg)
        assert dlg.validate() is None
        return dlg.compute()
    finally:
        VariablePickerDialog._memory.pop(cls.__name__, None)
        VariablePickerDialog._states.pop(cls.__name__, None)
        dlg.deleteLater()


def _section(result):
    return result.section if hasattr(result, "section") else result


def test_aft_dialog(qapp):
    ds = Dataset(pd.read_csv(DATA / "ovarian.csv").astype(float))
    res = _run(sdlg.ParametricAFTDialog, ds,
               {"time": ["futime"], "status": ["fustat"], "cov": ["age"], "factors": ["rx"]},
               {"distribution": "lognormal", "plots": ["survival"], "separate": "rx",
                "cox_snell": True, "predict": ["survival"], "print_encoding": True})
    assert res.saved and _section(res).title == "Parametric Accelerated Failure Time Models"
    sec = _run(sdlg.ParametricAFTDialog, ds, {"time": ["futime"]},
               {"mode": "survival"})
    assert any(getattr(t, "title", "") == "Parameter Estimates" for t in sec.items)


def test_aft_dialog_start_end_mode(qapp):
    df = pd.DataFrame({"a": [1.0, 2, 3, 4, 2, 5, 1, 3], "b": [2.0, 3, 5, 4, 6, 7, 2, 4],
                       "x": [0.1, 0.5, -0.2, 0.3, 1.0, -1.0, 0.0, 0.7]})

    def to_startend(dlg):
        dlg.rb_se.setChecked(True)
        assert not dlg._target("time").widget.isEnabled()
        dlg._take("a", dlg._target("start"))
        dlg._take("b", dlg._target("end"))

    sec = _run(sdlg.ParametricAFTDialog, Dataset(df), {"cov": ["x"]},
               {"status_treatment": "derive"}, check=to_startend)
    rows = next(t for t in sec.items if getattr(t, "title", "") == "Case Processing Summary")
    assert any(r[1] == "Interval Censoring" and r[2] == "7" for r in rows.rows)


def test_frailty_dialog(qapp):
    ds = Dataset(pd.read_csv(DATA / "side_effect.csv").astype(float))
    res = _run(sdlg.SharedFrailtyDialog, ds,
               {"time": ["endTime"], "status": ["sideEffect"], "subject": ["patID"],
                "cov": ["age"], "factors": ["female"]},
               {"plots": ["survival", "hazard"], "separate": "female",
                "predict": ["uncond_survival"]})
    sec = _section(res)
    assert sec.title == "Parametric Shared Frailty Models"
    pe = next(t for t in sec.items if getattr(t, "title", "") == "Parameter Estimates")
    assert next(r for r in pe.rows if r[0] == "[female=0]")[1] == ".017"


def test_status_map_dialog(qapp):
    ds = Dataset(pd.DataFrame({"s": ["F", "R", "L"], "t": [1.0, 2, 3]}))
    ds.var("s").type = "string"
    dlg = sdlg.StatusMapDialog(ds.var("s"), None, "exclude", ["event", "right", "left"])
    smap, unm = dlg.result()
    assert smap == {"event": ["F"], "right": ["R"], "left": ["L"]} and unm == "exclude"
    dlg.deleteLater()


def test_glmm_dialog(qapp):
    ds = Dataset(pd.read_csv(DATA / "multicenter.csv"))
    res = _run(gdlg.GeneralizedLinearMixedDialog, ds,
               {"subjects": ["center"], "target": ["sideeffect"], "trials": ["n"],
                "factors": ["group"]},
               {"preset": "binary_logistic", "distribution": "binomial", "link": "logit",
                "emm": ["group"], "contrast": "pairwise", "save": ["predicted"]})
    sec = _section(res)
    assert res.saved and sec.title == "Generalized Linear Mixed Models"
    coef = next(t for t in sec.items if getattr(t, "title", "") == "Fixed Coefficients")
    assert next(r for r in coef.rows if r[0] == "[group=A]")[1] == "-.490"


def test_glmm_dialog_validation_and_blocks(qapp):
    rng = np.random.default_rng(3)
    g = np.repeat(np.arange(8), 6)
    ds = Dataset(pd.DataFrame({"y": rng.poisson(3, 48) * 1.0, "x": rng.normal(size=48),
                               "g": g * 1.0}))
    VariablePickerDialog._memory["GeneralizedLinearMixedDialog"] = {
        "subjects": ["g"], "target": ["y"], "covariates": ["x"]}
    VariablePickerDialog._states["GeneralizedLinearMixedDialog"] = {
        "preset": "custom", "distribution": "poisson", "link": "logit"}
    dlg = gdlg.GeneralizedLinearMixedDialog(ds)
    try:
        assert "binomial" in dlg.validate()
        dlg.link.setCurrentIndex(dlg.link.findData("log"))
        assert dlg.validate() is None
        blocks = dlg._blocks()
        assert blocks == [{"effects": [], "intercept": True, "subjects": ["g"],
                           "covtype": "VC", "predictions": False}]
        assert "Subject: g" in gdlg.block_text(blocks[0])
        dlg.opts["blocks"] = [dict(blocks[0], effects=["x"], covtype="UN")]
        sec = _section(dlg.compute())
        cov = next(t for t in sec.items
                   if getattr(t, "title", "") == "Covariance Parameter Estimates")
        assert [r[1] for r in cov.rows] == ["Var(Intercept)", "Cov(x, Intercept)", "Var(x)"]
    finally:
        VariablePickerDialog._memory.pop("GeneralizedLinearMixedDialog", None)
        VariablePickerDialog._states.pop("GeneralizedLinearMixedDialog", None)
        dlg.deleteLater()


def test_menus_and_selftests():
    from rsstatchem.ui import menus

    tree = menus.build_tree({"Analyze": [("Survival", []), ("Mixed Models", [])]})
    sub = {e[0]: e[1] for e in tree["Analyze"] if e}
    surv = [e[0] for e in sub["Survival"] if e]
    assert surv[-2:] == ["Parametric Accelerated Failure Time (AFT) Models...",
                         "Parametric Shared Frailty Models..."]
    assert "Generalized Linear..." in [e[0] for e in sub["Mixed Models"] if e]
    sdlg.selftest()
    gdlg.selftest()

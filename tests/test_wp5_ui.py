"""WP5 dialogs offscreen: every Power Analysis / Meta Analysis / Bayesian Statistics /
Correspondence / MDS dialog builds, validates and computes, and the menus are merged."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import meta as mt
from rsstatchem.ui.dialogs import bayes as bd
from rsstatchem.ui.dialogs import correspondence as cd
from rsstatchem.ui.dialogs import mds as md
from rsstatchem.ui.dialogs import meta as mtd
from rsstatchem.ui.dialogs import power as pd_
from rsstatchem.ui.dialogs.base import VariablePickerDialog


def _run(cls, ds, memory=None, opts=None):
    VariablePickerDialog._memory[cls.__name__] = memory or {}
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


def _tables(sec):
    return [i for i in sec.items if getattr(i, "kind", "") == "table"]


ALL_POWER = [pd_.OneSampleTPowerDialog, pd_.PairedTPowerDialog, pd_.IndependentTPowerDialog,
             pd_.AnovaPowerDialog, pd_.OneBinomialPowerDialog, pd_.RelatedBinomialPowerDialog,
             pd_.IndependentBinomialPowerDialog, pd_.PearsonPowerDialog,
             pd_.SpearmanPowerDialog, pd_.PartialPowerDialog, pd_.RegressionPowerDialog]


@pytest.mark.parametrize("cls", ALL_POWER)
def test_power_dialogs(qapp, cls):
    pd_.PowerDialog._memory.pop(cls.__name__, None)
    dlg = cls(None, None)
    assert cls.needs_data is False
    assert dlg.check_params() is None
    dlg.opts["plots"] = {"by_n": True}
    sec = dlg.compute()
    t = _tables(sec)[0]
    assert t.title == "Power Analysis Table" and t.rows[0][1] not in ("", ".")
    dlg.est_power.setChecked(True)
    dlg.n_edit.setText("40 60")
    sec = dlg.compute()
    assert len(_tables(sec)[0].rows) == 2
    dlg.accept()
    assert cls.__name__ in pd_.PowerDialog._memory
    dlg.deleteLater()


def test_power_one_sample_values_and_contrasts(qapp):
    pd_.PowerDialog._memory.pop("AnovaPowerDialog", None)
    pd_.PowerDialog._memory.pop("OneSampleTPowerDialog", None)
    dlg = pd_.OneSampleTPowerDialog(None, None)
    sec = dlg.compute()
    assert _tables(sec)[0].rows[0][1:3] == ["34", ".808"]
    dlg.deleteLater()
    dlg = pd_.AnovaPowerDialog(None, None)
    dlg.opts.update(contrasts=[[-1, 0, 1]], pairwise=True, adjust="sidak")
    assert dlg.check_params() is None
    sec = dlg.compute()
    contrast = _tables(sec)[1]
    assert contrast.title.startswith("Power Analysis Table (Contrast")
    assert len(contrast.rows) == 4
    dlg.opts["contrasts"] = [[1, 1, 1]]
    assert "sum to 0" in dlg.check_params()
    dlg.deleteLater()
    assert pd_.parse_numbers("0,8 0,9") == [0.8, 0.9]
    assert pd_.parse_numbers("30,40,50") == [30, 40, 50]
    assert pd_.parse_numbers("0.8, 0.9") == [0.8, 0.9]


@pytest.fixture()
def bcg_ds():
    d = mt.BCG.copy()
    d["alloc_code"] = d["alloc"].map({"alternate": 1.0, "random": 2.0, "systematic": 3.0})
    return Dataset(d.drop(columns=["alloc"]))


def test_meta_dialogs(qapp, bcg_ds):
    sec = _run(mtd.MetaBinaryRawDialog, bcg_ds,
               {"study": ["author"], "a": ["tpos"], "b": ["tneg"], "c": ["cpos"],
                "d": ["cneg"], "order": ["year"]},
               {"measure": "rr", "egger": True, "trimfill": True, "cumulative": True,
                "plots": ["forest", "funnel", "labbe"]})
    est = _tables(sec)[1]
    assert est.rows[0][1:3] == ["-.715", ".180"]
    y, v = mt.binary_effects(bcg_ds.df.tpos, bcg_ds.df.tneg, bcg_ds.df.cpos, bcg_ds.df.cneg,
                             "rr")
    bcg_ds.add_variable("yi", values=pd.Series(y))
    bcg_ds.add_variable("vi", values=pd.Series(v))
    sec = _run(mtd.MetaContinuousESDialog, bcg_ds, {"es": ["yi"], "se": ["vi"],
                                                    "group": ["alloc_code"]},
               {"se_kind": "var", "method": "DL"})
    assert any(t.title == "Test of Subgroup Differences" for t in _tables(sec))
    sec = _run(mtd.MetaBinaryESDialog, bcg_ds, {"es": ["yi"], "se": ["vi"]},
               {"se_kind": "var", "measure": "rr", "model": "fixed"})
    assert _tables(sec)[1].rows[0][1] == "-.430"
    rng = np.random.default_rng(3)
    k = 10
    cont = Dataset(pd.DataFrame({"n1": rng.integers(20, 60, k).astype(float),
                                 "m1": rng.normal(10.5, 0.5, k), "s1": rng.uniform(1.5, 2.5, k),
                                 "n2": rng.integers(20, 60, k).astype(float),
                                 "m2": rng.normal(10.0, 0.5, k),
                                 "s2": rng.uniform(1.5, 2.5, k)}))
    sec = _run(mtd.MetaContinuousRawDialog, cont,
               {r: [r] for r in ("n1", "m1", "s1", "n2", "m2", "s2")},
               {"measure": "cohen", "knapp": "truncated", "loo": True, "begg": True})
    assert "Cohen's d" in _tables(sec)[1].footnotes[0]
    sec = _run(mtd.MetaRegressionDialog, bcg_ds, {"es": ["yi"], "se": ["vi"],
                                                  "cov": ["ablat"], "fac": ["alloc_code"]},
               {"se_kind": "var"})
    assert _tables(sec)[0].title == "Parameter Estimates"
    assert _tables(sec)[0].rows[-1][1] == "-.027"


@pytest.fixture()
def bayes_ds():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"pre": rng.normal(50, 10, 40), "g": np.repeat([1.0, 2.0], 20),
                       "k": np.tile([1.0, 2.0, 3.0, 4.0], 10),
                       "cnt": rng.poisson(3, 40) * 1.0})
    df["post"] = df.pre + rng.normal(3, 5, 40) + 2 * df.g
    df["t3"] = df.post + rng.normal(1, 5, 40)
    return Dataset(df)


def test_bayes_dialogs(qapp, bayes_ds):
    ds = bayes_ds
    sec = _run(bd.BayesOneSampleNormalDialog, ds, {"vars": ["pre"]},
               {"test_value": 50.0, "plots": True})
    assert any(i.kind == "image" for i in sec.items)
    _run(bd.BayesRelatedNormalDialog, ds, {"pairs": [("pre", "post")]})
    _run(bd.BayesIndependentNormalDialog, ds, {"vars": ["post"], "group": ["g"]},
         {"group_def": {"var": "g", "groups": (1.0, 2.0), "cut": None}, "method": "gonen",
          "equal": False})
    _run(bd.BayesPearsonDialog, ds, {"vars": ["pre", "post", "t3"]})
    _run(bd.BayesRegressionDialog, ds, {"dep": ["post"], "cov": ["pre"], "fac": ["g"]},
         {"prior": "hyperg"})
    _run(bd.BayesAnovaDialog, ds, {"deps": ["post"], "factor": ["k"]}, {"analysis": "bf"})
    _run(bd.BayesRepeatedDialog, ds, {"vars": ["pre", "post", "t3"]})
    _run(bd.BayesBinomialDialog, ds, {"vars": ["g"]}, {"success": "value",
                                                       "success_value": "2"})
    _run(bd.BayesPoissonDialog, ds, {"vars": ["cnt"]}, {"test_value": 3.0})
    sec = _run(bd.BayesLoglinearDialog, ds, {"row": ["g"], "col": ["k"]},
               {"analysis": "posterior"})
    assert not any(t.title == "Tests of Independence" for t in _tables(sec))


def test_ca_and_mds_dialogs(qapp):
    from rsstatchem.core.stats.correspondence import SMOKE
    rows = np.repeat(np.arange(1, 6), 4)
    cols = np.tile(np.arange(1, 5), 5)
    counts = SMOKE.ravel().astype(int)
    ds = Dataset(pd.DataFrame({"staff": np.repeat(rows, counts).astype(float),
                               "smoking": np.repeat(cols, counts).astype(float)}))
    sec = _run(cd.CorrespondenceDialog, ds, {"row": ["staff"], "col": ["smoking"]},
               {"normalization": "principal", "plots": ["biplot", "rows"],
                "row_range": {"var": "staff", "range": [1, 4]}})
    assert len(_tables(sec)[0].rows) == 5
    rng = np.random.default_rng(2)
    pts = rng.normal(size=(7, 2))
    D = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
    dm = Dataset(pd.DataFrame(D, columns=[f"v{i}" for i in range(7)]))
    names = [f"v{i}" for i in range(7)]
    _run(md.ProxscalDialog, dm, {"vars": names}, {"level": "ordinal",
                                                  "plots": ["common", "residuals"]})
    _run(md.AlscalDialog, dm, {"vars": names}, {"level": "interval"})
    raw = Dataset(pd.DataFrame(rng.normal(size=(20, 5)), columns=list("abcde")))
    _run(md.ProxscalDialog, raw, {"vars": list("abcde")}, {"source": "data",
                                                            "standardize": True})
    _run(md.AlscalDialog, raw, {"vars": list("abc")}, {"source": "data", "between": "cases",
                                                        "measure": "cityblock"})


def test_menus_merged(qapp):
    from rsstatchem.ui import menus

    names = [m.__name__ for m in menus.extension_modules()]
    for mod in ("power", "meta", "bayes", "correspondence", "mds"):
        assert f"rsstatchem.ui.dialogs.{mod}" in names
    from rsstatchem.ui.data_editor import menu_tree

    analyze = dict((e[0], e[1]) for e in menu_tree()["Analyze"] if isinstance(e, tuple))
    assert "Power Analysis" in analyze and "Meta Analysis" in analyze
    assert "Bayesian Statistics" in analyze
    dr = [e[0] for e in analyze["Dimension Reduction"] if isinstance(e, tuple)]
    assert "Correspondence Analysis..." in dr
    sc = [e[0] for e in analyze["Scale"] if isinstance(e, tuple)]
    assert "Multidimensional Scaling (PROXSCAL)..." in sc
    for mod in (pd_, mtd, bd, cd, md):
        mod.selftest()

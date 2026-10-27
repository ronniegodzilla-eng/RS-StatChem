"""WP2 dialogs driven headlessly: blocks (Previous/Next), interaction terms, compute()."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rsstatchem.core.dataset import Dataset


@pytest.fixture
def ds():
    rng = np.random.default_rng(3)
    n = 120
    x1, x2, m = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    y = 1 + x1 + 0.5 * x2 + 0.4 * x1 * m + rng.normal(size=n)
    b = (rng.random(n) < 1 / (1 + np.exp(-(x1 - x2)))).astype(float)
    g = rng.integers(1, 4, n).astype(float)
    cnt = rng.poisson(np.exp(0.3 + 0.4 * x1)).astype(float)
    o = np.digitize(x1 + rng.logistic(size=n) * 0.7, [-0.5, 0.5]) + 1.0
    return Dataset(pd.DataFrame({"y": y, "x1": x1, "x2": x2, "m": m, "x1m": x1 * m, "b": b,
                                 "g": g, "cnt": cnt, "o": o, "w": rng.uniform(.5, 2, n),
                                 "s": np.repeat(np.arange(30), 4).astype(float),
                                 "t": np.tile(np.arange(1, 5), 30).astype(float)}))


def _pick(dlg, key, names):
    target = next(t for t in dlg._targets if t.key == key)
    for name in names:
        assert dlg._take(name, target)
    dlg._update_arrows()


def test_linear_dialog_blocks(qapp, ds):
    from rsstatchem.ui.dialogs.regression import LinearRegressionDialog

    dlg = LinearRegressionDialog(ds)
    _pick(dlg, "dep", ["y"])
    _pick(dlg, "ind", ["x1", "m"])
    assert dlg.next_button.isEnabled()
    dlg._go(1)
    assert dlg.block_label.text() == "Block 2 of 2"
    _pick(dlg, "ind", ["x1m"])
    dlg._go(-1)
    assert dlg.selected("ind") == ["x1", "m"]
    dlg.opts["statistics"] = ["estimates", "model_fit", "r2_change"]
    dlg.opts["save"] = ["pred", "cook", "mcin"]
    assert dlg.validate() is None
    res = dlg.compute()
    summary = next(t for t in res.section.items if getattr(t, "title", "").startswith(
        "Model Summary"))
    assert len(summary.rows) == 2
    assert {"PRE", "COO", "LMCI", "UMCI"} <= set(res.saved)


def test_genlin_gee_probit_loglinear_dialogs(qapp, ds):
    from rsstatchem.ui.dialogs import genlin as G

    dlg = G.GenlinDialog(ds)
    _pick(dlg, "dep", ["cnt"])
    _pick(dlg, "factors", ["g"])
    _pick(dlg, "covs", ["x1"])
    dlg.opts.update(model_type="poisson", distribution="poisson", link="log",
                    emmeans=["g"], save=["mean", "leverage"])
    assert dlg.validate() is None
    res = dlg.compute()
    assert {"MeanPredicted", "Leverage"} <= set(res.saved)
    dlg = G.GeeDialog(ds)
    _pick(dlg, "dep", ["y"])
    _pick(dlg, "covs", ["x1"])
    _pick(dlg, "subject", ["s"])
    _pick(dlg, "within", ["t"])
    res = dlg.compute()
    assert any(getattr(t, "title", "") == "Working Correlation Matrix"
               for t in res.section.items)
    for cls, picks in ((G.LoglinearGeneralDialog, {"factors": ["g", "b"]}),
                       (G.LoglinearSelectionDialog, {"factors": ["g", "b", "o"]}),
                       (G.LoglinearLogitDialog, {"deps": ["b"], "factors": ["g"]})):
        dlg = cls(ds)
        for key, names in picks.items():
            _pick(dlg, key, names)
        assert dlg.compute().items


def test_regression_ext_dialogs(qapp, ds):
    from rsstatchem.ui.dialogs import regression_ext as R

    ds.add_variable("pos2", values=(np.abs(ds.df["x2"]) + 1).to_numpy())
    dlg = R.NonlinearDialog(ds)
    _pick(dlg, "dep", ["y"])
    dlg.model.setText("a + b * x1")
    dlg.opts["parameters"] = "a=0\nb=1"
    assert dlg.validate() is None
    assert dlg.compute().section.items
    dlg = R.TwoStageDialog(ds)
    _pick(dlg, "dep", ["y"])
    _pick(dlg, "expl", ["x1", "x2"])
    _pick(dlg, "inst", ["m", "x2"])
    assert dlg.selected("inst") == ["m", "x2"]
    assert dlg.source.count() == ds.n_vars - 1
    assert dlg.compute().section.items
    dlg = R.WeightEstimationDialog(ds)
    _pick(dlg, "dep", ["y"])
    _pick(dlg, "ind", ["x1"])
    _pick(dlg, "source", ["pos2"])
    assert dlg.validate() is None and dlg.compute().section.items
    dlg = R.QuantileDialog(ds)
    _pick(dlg, "dep", ["y"])
    _pick(dlg, "covs", ["x1"])
    dlg.quantiles.setText("0.25 0.75")
    assert dlg.validate() is None and dlg.compute().section.items
    for cls in (R.RidgeDialog, R.LassoDialog, R.ElasticNetDialog):
        dlg = cls(ds)
        _pick(dlg, "dep", ["y"])
        _pick(dlg, "preds", ["x1", "x2"])
        dlg.opts["mode"] = "cv"
        assert dlg.validate() is None and dlg.compute().section.items
    dlg = R.DistancesDialog(ds)
    _pick(dlg, "vars", ["x1", "x2", "m"])
    dlg.opts["between"] = "variables"
    assert dlg.validate() is None and dlg.compute().items
    dlg = R.CanonicalDialog(ds)
    _pick(dlg, "set1", ["x1", "x2"])
    _pick(dlg, "set2", ["y", "m"])
    assert dlg.compute().section.items
    dlg = R.RocAnalysisDialog(ds)
    _pick(dlg, "tests", ["x1", "x2"])
    _pick(dlg, "state", ["b"])
    dlg.positive.setText("1")
    assert dlg.validate() is None and dlg.compute().items


def test_menu_tree_contains_wp2_entries(qapp):
    from rsstatchem.ui import menus

    tree = menus.build_tree({"Analyze": []})
    analyze = tree["Analyze"]
    sub = {e[0]: e[1] for e in analyze if isinstance(e, tuple) and isinstance(e[1], list)}
    assert "Generalized Linear Models" in sub and "Loglinear" in sub
    reg = {e[0]: e[1] for e in sub["Regression"] if isinstance(e, tuple)}
    assert "Probit..." in reg and "Linear OLS Alternatives" in reg
    from rsstatchem.ui.dialogs.regression_ext import DistancesDialog, NonlinearDialog

    assert reg["Nonlinear..."] is NonlinearDialog
    cor = {e[0]: e[1] for e in sub["Correlate"] if isinstance(e, tuple)}
    assert cor["Distances..."] is DistancesDialog
    assert "ROC Analysis..." in [e[0] for e in sub["Classify"] if isinstance(e, tuple)]


def test_logistic_dialog_interaction_blocks(qapp, ds):
    from rsstatchem.ui.dialogs.logistic import BinaryLogisticDialog

    dlg = BinaryLogisticDialog(ds)
    _pick(dlg, "dep", ["b"])
    _pick(dlg, "cov", ["x1", "x2"])
    dlg._go(1)
    for i in range(dlg.source.count()):
        item = dlg.source.item(i)
        item.setSelected(item.text() in ("x2", "m"))
    dlg._add_interaction()
    assert dlg._cov_terms() == ["x2*m"]
    dlg._move_back(dlg._cov_target(), all_items=True)
    assert dlg._cov_terms() == []
    dlg._add_term_item("x1*x2")
    assert dlg.validate() is None
    res = dlg.compute()
    assert sum(1 for t in res.section.items if getattr(t, "title", "") ==
               "Omnibus Tests of Model Coefficients") == 2

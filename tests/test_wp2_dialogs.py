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
                                 "g": g, "cnt": cnt, "o": o, "w": rng.uniform(.5, 2, n)}))


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

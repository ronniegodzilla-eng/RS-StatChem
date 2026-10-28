"""WP9 dialogs offscreen: every Tree / TwoStep / Silhouettes / Nearest Neighbor / MLP / RBF /
PLS / MARS / Kernel Ridge dialog builds, validates and computes, and the menus are merged in
SPSS order."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

from rsstatchem.core.dataset import Dataset
from rsstatchem.ui.dialogs import classify_ml as cd
from rsstatchem.ui.dialogs import neural as nd
from rsstatchem.ui.dialogs import regression_ml as rd
from rsstatchem.ui.dialogs.base import VariablePickerDialog


def _ds():
    X, y = load_iris(return_X_y=True)
    r = np.random.default_rng(0)
    df = pd.DataFrame(X, columns=["sl", "sw", "pl", "pw"])
    df["species"] = y + 1.0
    df["grp"] = r.integers(1, 3, 150).astype(float)
    ds = Dataset(df)
    ds.var("species").measure = "nominal"
    ds.var("grp").measure = "nominal"
    return ds


def _run(cls, ds, memory, opts=None):
    VariablePickerDialog._memory[cls.__name__] = memory
    VariablePickerDialog._states[cls.__name__] = {**cls.DEFAULTS, **(opts or {})}
    dlg = cls(ds, None)
    try:
        assert dlg.ok_button.isEnabled()
        assert dlg.validate() is None
        return dlg.compute()
    finally:
        VariablePickerDialog._memory.pop(cls.__name__, None)
        VariablePickerDialog._states.pop(cls.__name__, None)
        dlg.deleteLater()


def _titles(res):
    return [getattr(i, "title", "") for i in res.section.items]


@pytest.mark.parametrize("method", ["chaid", "exhaustive", "crt", "quest"])
def test_tree_dialog(qapp, method):
    res = _run(cd.TreeDialog, _ds(), {"dep": ["species"], "indep": ["sl", "sw", "pl", "pw",
                                                                    "grp"]},
               {"method": method, "min_parent": 10, "min_child": 5, "out_table": True,
                "save": ["node", "predicted"], "validation": "cv", "folds": 5})
    assert "Tree Diagram" in _titles(res) and "Tree Table" in _titles(res)
    assert {"NodeID", "PredictedValue"} <= set(res.saved)


def test_tree_dialog_validation_and_costs(qapp):
    ds = _ds()
    VariablePickerDialog._memory["TreeDialog"] = {"dep": ["pl"], "indep": ["sl"]}
    VariablePickerDialog._states["TreeDialog"] = {**cd.TreeDialog.DEFAULTS, "method": "quest"}
    dlg = cd.TreeDialog(ds, None)
    assert "QUEST" in dlg.validate()
    dlg.deleteLater()
    VariablePickerDialog._memory.pop("TreeDialog")
    VariablePickerDialog._states.pop("TreeDialog")
    res = _run(cd.TreeDialog, ds, {"dep": ["species"], "indep": ["sl", "sw"]},
               {"method": "crt", "min_parent": 10, "min_child": 5, "priors": "equal",
                "costs": "0 1 1; 5 0 1; 1 1 0"})
    assert res.model.costs is not None and res.model.priors is not None


def test_twostep_and_silhouettes_dialogs(qapp):
    ds = _ds()
    res = _run(cd.TwoStepDialog, ds, {"cat": ["grp"], "cont": ["sl", "sw", "pl", "pw"]},
               {"save": True})
    assert "Auto-Clustering" in _titles(res) and "TSC" in res.saved
    res = _run(cd.TwoStepDialog, ds, {"cat": [], "cont": ["pl", "pw"]},
               {"measure": "euclidean", "auto": False, "fixed_k": 3})
    assert res.model["k"] == 3
    res = _run(cd.SilhouettesDialog, ds, {"cluster": ["species"], "vars": ["pl", "pw"]},
               {"save": True})
    assert "Silhouette" in res.saved


def test_knn_dialog(qapp):
    res = _run(cd.NearestNeighborDialog, _ds(),
               {"target": ["species"], "features": ["sl", "sw", "pl", "pw", "grp"],
                "forced": ["pl"], "focal": []},
               {"auto_k": True, "select": True, "save_prob": True})
    assert "Classification Table" in _titles(res)
    assert "KNN_PredictedProbability_1" in res.saved


def test_neural_dialogs(qapp):
    ds = _ds()
    mem = {"deps": ["species"], "factors": ["grp"], "covariates": ["sl", "sw", "pl", "pw"]}
    res = _run(nd.MultilayerPerceptronDialog, ds, mem, {"out_weights": True,
                                                        "save_prob": True})
    assert "Parameter Estimates" in _titles(res)
    res = _run(nd.MultilayerPerceptronDialog, ds, mem,
               {"auto_arch": False, "layers": 2, "units_auto": False, "units1": 5,
                "units2": 3, "mode": "minibatch", "algorithm": "gd"})
    assert res.model.net.sizes[1:3] == [5, 3]
    res = _run(nd.RadialBasisFunctionDialog, ds, mem, {"units": "fixed", "u_fixed": 6})
    assert len(res.model.centres) <= 6
    VariablePickerDialog._memory["MultilayerPerceptronDialog"] = mem
    VariablePickerDialog._states["MultilayerPerceptronDialog"] = {
        **nd.MultilayerPerceptronDialog.DEFAULTS, "mode": "online"}
    dlg = nd.MultilayerPerceptronDialog(ds, None)
    assert "batch" in dlg.validate()
    assert dlg.tabs.count() == 7
    dlg.deleteLater()
    VariablePickerDialog._memory.pop("MultilayerPerceptronDialog")
    VariablePickerDialog._states.pop("MultilayerPerceptronDialog")


def test_regression_dialogs(qapp):
    ds = _ds()
    res = _run(rd.PartialLeastSquaresDialog, ds, {"deps": ["pl"], "indep": ["sl", "sw", "grp"],
                                                  "id": []}, {"n_factors": 2,
                                                              "save_pred": True})
    assert "Proportion of Variance Explained" in _titles(res)
    res = _run(rd.MarsDialog, ds, {"dep": ["pl"], "pred": ["sl", "sw", "grp"]},
               {"save_pred": True})
    assert "Coefficients" in _titles(res) and "MARS_Predicted" in res.saved
    res = _run(rd.KernelRidgeDialog, ds, {"deps": ["pl"], "pred": ["sl", "sw"]},
               {"kernels": ["rbf", "linear"], "alphas": "1 0,1", "gammas": "auto 0.5"})
    assert "Cross-Validation Results" in _titles(res)


def test_menus_registered_in_spss_order():
    from rsstatchem.ui import menus

    for name in ("rsstatchem.ui.dialogs.classify_ml", "rsstatchem.ui.dialogs.neural",
                 "rsstatchem.ui.dialogs.regression_ml"):
        assert name in menus.EXTENSIONS
    tree = menus.build_tree({"Analyze": []})
    analyze = tree["Analyze"]
    labels = [e[0] for e in analyze if isinstance(e, tuple)]
    assert labels.index("Neural Networks") < labels.index("Classify")

    def sub(path):
        return [e[0] if isinstance(e, tuple) else e for e in menus._find_menu(tree, path)
                if e is not None]
    assert sub("Analyze/Neural Networks") == ["Multilayer Perceptron...",
                                              "Radial Basis Function..."]
    cls = sub("Analyze/Classify")
    assert cls.index("TwoStep Cluster...") < cls.index("Cluster Silhouettes...") < \
        cls.index("Tree...") < cls.index("Nearest Neighbor...")
    assert "Multiple Adaptive Regression Splines..." in sub(
        "Analyze/Generalized Linear Models")
    reg = sub("Analyze/Regression")
    assert "Partial Least Squares..." in reg and "Kernel Ridge..." in reg

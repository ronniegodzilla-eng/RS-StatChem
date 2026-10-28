"""WP9 Multilayer Perceptron and Radial Basis Function: back-propagation gradients, training
behaviour against scikit-learn's MLP (structural: IBM's random initialisation cannot be
reproduced), IBM stopping rules, sensitivity importance and the SPSS output tables."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier, MLPRegressor

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import neural as nn


@pytest.mark.parametrize("out_act,err,hid", [("softmax", "ce", "tanh"),
                                             ("identity", "sse", "tanh"),
                                             ("tanh", "sse", "tanh"),
                                             ("sigmoid", "sse", "sigmoid")])
def test_backprop_gradient(out_act, err, hid):
    r = np.random.default_rng(0)
    X = r.normal(size=(25, 3))
    T = np.eye(3)[r.integers(0, 3, 25)]
    net = nn.Net([3, 4, 2, 3], hid, out_act, err, [[0, 1, 2]])
    w = r.normal(size=net.n_weights())
    _, g = net.loss_grad(w, X, T)
    num = np.zeros_like(w)
    for i in range(len(w)):
        e = np.zeros_like(w)
        e[i] = 1e-6
        num[i] = (net.loss_grad(w + e, X, T)[0] - net.loss_grad(w - e, X, T)[0]) / 2e-6
    np.testing.assert_allclose(g, num, atol=1e-6)


def _iris():
    X, y = load_iris(return_X_y=True)
    return (X - X.mean(0)) / X.std(0, ddof=1), y


def test_mlp_classifier_trains_like_sklearn():
    X, y = _iris()
    T = np.eye(3)[y]
    spec = [nn.TargetSpec("y", "categorical", [0, 1, 2], list("abc"), cols=[0, 1, 2])]
    net, w, rule, steps = nn.mlp_fit(X, T, spec, (4,), seed=3)
    ours = np.mean(net.predict(w, X).argmax(1) == y)
    sk = MLPClassifier(hidden_layer_sizes=(4,), activation="tanh", solver="lbfgs",
                       alpha=1e-8, max_iter=2000, random_state=0).fit(X, y)
    assert ours >= 0.97 and sk.score(X, y) >= 0.97
    assert net.out_act == "softmax" and net.error == "ce"
    np.testing.assert_allclose(net.predict(w, X).sum(1), 1.0)
    for alg, mode in (("gd", "batch"), ("gd", "minibatch")):
        s = nn.TrainSettings(algorithm=alg, mode=mode)
        net, w, _, _ = nn.mlp_fit(X, T, spec, (4,), settings=s, seed=3)
        assert np.mean(net.predict(w, X).argmax(1) == y) > 0.9


def test_mlp_regression_close_to_sklearn():
    r = np.random.default_rng(1)
    X = r.uniform(-2, 2, (300, 2))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2
    ys = (y - y.mean()) / y.std(ddof=1)
    spec = [nn.TargetSpec("y", "scale", cols=[0])]
    net, w, _, _ = nn.mlp_fit(X, ys[:, None], spec, (8,), seed=2,
                              settings=nn.TrainSettings(min_rel_change=1e-7, max_steps=3000))
    pred = net.predict(w, X)[:, 0]
    r2 = 1 - ((ys - pred) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()
    sk = MLPRegressor(hidden_layer_sizes=(8,), activation="tanh", solver="lbfgs",
                      alpha=1e-8, max_iter=3000, random_state=0).fit(X, ys)
    assert r2 > 0.97 and r2 > sk.score(X, ys) - 0.03


def test_early_stopping_on_testing_sample():
    X, y = _iris()
    T = np.eye(3)[y]
    spec = [nn.TargetSpec("y", "categorical", [0, 1, 2], list("abc"), cols=[0, 1, 2])]
    r = np.random.default_rng(0)
    te = r.random(150) < 0.3
    net, w, rule, steps = nn.mlp_fit(X[~te], T[~te], spec, (6,), Xte=X[te], Tte=T[te], seed=1)
    assert "no decrease in error" in rule
    e_final = net.error_value(net.predict(w, X[te]), T[te])
    assert e_final < net.error_value(np.full((te.sum(), 3), 1 / 3), T[te])


def test_sensitivity_importance_identifies_relevant_input():
    r = np.random.default_rng(4)
    X = r.normal(size=(200, 3))
    imp = nn.sensitivity(lambda Z: (3 * Z[:, :1] + 0.1 * Z[:, 1:2]), X, [[0], [1], [2]],
                         ["cov"] * 3)
    assert imp.sum() == pytest.approx(1.0) and imp.argmax() == 0 and imp[2] == 0


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


def test_mlp_procedure_tables():
    ds = _ds()
    res = nn.mlp_procedure(ds, ["species"], ["grp"], ["sl", "sw", "pl", "pw"], seed=7,
                           output={"weights": True}, save=("predicted", "probability",
                                                           "partition"))
    titles = [getattr(i, "title", "") for i in res.section.items]
    for t in ("Case Processing Summary", "Network Information", "Model Summary",
              "Classification", "ROC Curve", "Area Under the Curve", "Gain", "Lift",
              "Network Diagram", "Parameter Estimates", "Independent Variable Importance"):
        assert t in titles, t
    summ = next(i for i in res.section.items if getattr(i, "title", "") == "Model Summary")
    assert summ.rows[0][1] == "Cross Entropy Error"
    auc = next(i for i in res.section.items if getattr(i, "title", "") ==
               "Area Under the Curve")
    assert all(float(r[2]) > 0.8 for r in auc.rows)
    assert "MLP_PseudoProbability_1" in res.saved and "Partition" in res.saved
    res = nn.mlp_procedure(ds, ["pl"], [], ["sl", "sw"], auto_arch=False, layers=2,
                           units=(4, 3), seed=7)
    summ = next(i for i in res.section.items if getattr(i, "title", "") == "Model Summary")
    assert summ.rows[0][1] == "Sum of Squares Error" and summ.rows[1][1] == "Relative Error"
    assert res.model.net.sizes == [2, 4, 3, 1]


def test_rbf_fit_and_procedure():
    X, y = _iris()
    T = np.eye(3)[y]
    m = nn.rbf_fit_units(X, T, [], 4, 6, True, None)
    assert np.mean(m.predict(X).argmax(1) == y) > 0.8
    H = m.hidden(X)
    np.testing.assert_allclose(H.sum(1), 1.0)            # normalized RBF
    m2 = nn.rbf_fit_units(X, T, [], 4, 6, False, None)
    assert (m2.hidden(X) <= 1).all()
    ds = _ds()
    res = nn.rbf_procedure(ds, ["species"], ["grp"], ["sl", "sw", "pl", "pw"], seed=3,
                           output={"weights": True}, save=("predicted",))
    titles = [getattr(i, "title", "") for i in res.section.items]
    assert "Parameter Estimates" in titles and "Number of Hidden Units" in titles
    cls = next(i for i in res.section.items if getattr(i, "title", "") == "Classification")
    assert float(cls.rows[3][-1].rstrip("%")) > 80

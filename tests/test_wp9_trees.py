"""WP9 Decision Trees (CHAID, Exhaustive CHAID, CRT, QUEST) against scikit-learn's CART,
R rpart's printed iris tree, scipy/statsmodels tests and Kass's Bonferroni multipliers."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import trees


def _scale_preds(X):
    return [trees.Predictor(f"x{i}", "scale", X[:, i], [], "scale") for i in range(X.shape[1])]


def iris_ds():
    X, y = load_iris(return_X_y=True)
    df = pd.DataFrame(X, columns=["sl", "sw", "pl", "pw"])
    df["species"] = y + 1.0
    ds = Dataset(df)
    ds.var("species").measure = "nominal"
    ds.var("species").value_labels = {1.0: "setosa", 2.0: "versicolor", 3.0: "virginica"}
    return ds


def test_crt_regression_equals_sklearn():
    X, y = load_diabetes(return_X_y=True)
    m = trees.fit_tree(y, "scale", [], _scale_preds(X), "crt",
                       {"max_depth": 4, "min_parent": 2, "min_child": 1,
                        "min_improvement": 0.0, "max_surrogates": 0})
    sk = DecisionTreeRegressor(max_depth=4).fit(X, y)
    pred = np.array([m.nodes[t].mean for t in m.assign()])
    np.testing.assert_allclose(pred, sk.predict(X), atol=1e-9)
    m3 = trees.fit_tree(y, "scale", [], _scale_preds(X), "crt",
                        {"max_depth": 3, "min_parent": 2, "min_child": 1,
                         "min_improvement": 0.0, "max_surrogates": 0})
    sk3 = DecisionTreeRegressor(max_depth=3).fit(X, y)   # depth 4 has an exact s1/s2 tie
    imp = trees.importance(m3)
    np.testing.assert_allclose(imp / imp.sum(), sk3.feature_importances_, atol=1e-12)
    # IBM improvement p(t)·Δi equals scikit-learn's weighted impurity decrease at the root
    t = sk.tree_
    a, b = t.children_left[0], t.children_right[0]
    root_dec = t.impurity[0] - (t.weighted_n_node_samples[a] * t.impurity[a] +
                                t.weighted_n_node_samples[b] * t.impurity[b]) / len(y)
    assert m.nodes[1].split.stat["improvement"] == pytest.approx(root_dec, rel=1e-10)


def test_crt_classification_gini_matches_sklearn():
    X, y = load_breast_cancer(return_X_y=True)
    X = X[:, :8]
    m = trees.fit_tree(y.astype(float), "categorical", ["0", "1"], _scale_preds(X), "crt",
                       {"max_depth": 3, "min_parent": 2, "min_child": 1,
                        "min_improvement": 0.0, "max_surrogates": 0})
    sk = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    pred = np.array([m.nodes[t].pred for t in m.assign()])
    assert np.mean(pred == sk.predict(X)) == 1.0
    sp = m.nodes[1].split
    assert sp.var == sk.tree_.feature[0]
    assert sp.value == pytest.approx(sk.tree_.threshold[0], abs=1e-6)


def test_crt_iris_matches_rpart_printout():
    # R: rpart(Species ~ ., iris) prints Petal.Length< 2.45 (50, setosa) and, for the other
    # 100, Petal.Width< 1.75 with 54 cases (49 versicolor, 5 virginica) and 46 cases
    # (1 versicolor, 45 virginica).
    X, y = load_iris(return_X_y=True)
    m = trees.fit_tree(y.astype(float), "categorical", list("abc"), _scale_preds(X), "crt",
                       {"max_depth": 2, "min_parent": 20, "min_child": 7})
    root = m.nodes[1].split
    assert root.var in (2, 3)          # petal length and width tie at the root
    left, right = (m.nodes[c] for c in m.nodes[1].children)
    assert left.counts.tolist() == [50, 0, 0]
    r = m.nodes[right.id].split
    assert r.var == 3 and r.value == pytest.approx(1.75)
    c4, c5 = (m.nodes[c] for c in right.children)
    assert c4.counts.tolist() == [0, 49, 5] and c5.counts.tolist() == [0, 1, 45]


def test_bonferroni_multipliers_kass():
    # nominal: Stirling numbers of the second kind S(I, r); ordinal: C(I-1, r-1)
    assert trees.bonferroni(4, 2, "nominal") == pytest.approx(7)
    assert trees.bonferroni(5, 3, "nominal") == pytest.approx(25)
    assert trees.bonferroni(5, 3, "ordinal") == pytest.approx(6)
    assert trees.bonferroni(10, 4, "ordinal") == pytest.approx(84)
    # floating category: C(I-2, r-2) + r C(I-2, r-1)
    assert trees.bonferroni(6, 3, "ordinal", floating=True) == pytest.approx(4 + 3 * 6)
    assert trees.bonferroni(4, 4, "nominal") == 1.0
    assert trees.bonferroni(5, 2, "nominal", exhaustive=True) == pytest.approx(1 + 3 + 6 + 10)


def test_chaid_pvalues_match_scipy_and_statsmodels():
    t = np.array([[20, 5, 3], [10, 12, 8], [4, 9, 15]], float)
    stat, lp, df = trees._chi2_logp(t)
    chi2, p, dof, _ = stats.chi2_contingency(t, correction=False)
    assert stat == pytest.approx(chi2) and math.exp(lp) == pytest.approx(p) and df == dof
    g, lpg, _ = trees._chi2_logp(t, lr=True)
    g2, pg, _, _ = stats.chi2_contingency(t, correction=False, lambda_="log-likelihood")
    assert g == pytest.approx(g2) and math.exp(lpg) == pytest.approx(pg)
    S = np.array([[10, 50.0, 300.0], [12, 80.0, 600.0], [8, 30.0, 140.0]])
    F, lpf, (d1, d2) = trees._anova_logp(S)
    assert (d1, d2) == (2, 27) and F > 0
    # Goodman row-effects LR statistic = deviance difference of two Poisson log-linear models
    import statsmodels.api as sm
    I, J = t.shape
    yv = t.ravel()
    rows = np.repeat(np.arange(I), J)
    cols = np.tile(np.arange(J), I)
    D0 = np.column_stack([np.ones(I * J)] + [(rows == i) * 1.0 for i in range(1, I)] +
                         [(cols == j) * 1.0 for j in range(1, J)])
    D1 = np.column_stack([D0] + [(rows == i) * (cols + 1.0) for i in range(1, I)])
    dev0 = sm.GLM(yv, D0, family=sm.families.Poisson()).fit().deviance
    dev1 = sm.GLM(yv, D1, family=sm.families.Poisson()).fit().deviance
    h2, lph, dfh = trees._row_effects_logp(t, np.arange(1.0, J + 1))
    assert h2 == pytest.approx(dev0 - dev1, rel=1e-6) and dfh == I - 1


def test_chaid_binning_and_growth():
    x = np.arange(1.0, 101.0)
    codes, breaks = trees._bin_scale(x, 10)
    assert len(breaks) == 9 and np.bincount(codes.astype(int)).tolist() == [10] * 10
    ds = iris_ds()
    for method in ("chaid", "exhaustive"):
        res = trees.tree_procedure(ds, "species", ["sl", "sw", "pl", "pw"], method,
                                   settings={"min_parent": 10, "min_child": 5},
                                   output={"table": True})
        m = res.model
        assert m.nodes[1].split is not None and len(m.nodes[1].children) >= 2
        acc = np.mean([m.nodes[t].pred for t in m.assign()] == m.y)
        assert acc > 0.9
        tab = next(i for i in res.section.items if getattr(i, "title", "") == "Tree Table")
        assert tab.rows[1][-4] != ""      # adjusted p of the first split


def test_quest_and_pruning_and_procedure_outputs():
    ds = iris_ds()
    res = trees.tree_procedure(ds, "species", ["sl", "sw", "pl", "pw"], "quest",
                               settings={"min_parent": 10, "min_child": 5},
                               save=("node", "predicted", "probability"))
    m = res.model
    assert len(m.terminals()) >= 3
    assert set(res.saved) == {"NodeID", "PredictedValue", "PredictedProbability_1",
                              "PredictedProbability_2", "PredictedProbability_3"}
    probs = sum(res.saved[f"PredictedProbability_{j}"][1] for j in (1, 2, 3))
    np.testing.assert_allclose(probs.to_numpy(), 1.0)
    X, y = load_breast_cancer(return_X_y=True)
    full = trees.fit_tree(y.astype(float), "categorical", ["0", "1"], _scale_preds(X[:, :8]),
                          "crt", {"max_depth": 6, "min_parent": 10, "min_child": 5})
    n_full = len(full.nodes)
    removed = trees.prune(full, 1.0)
    assert removed > 0 and len(full.nodes) == n_full - removed
    titles = [getattr(i, "title", "") for i in res.section.items]
    for t in ("Model Summary", "Tree Diagram", "Gain Summary for Nodes", "Risk",
              "Classification"):
        assert t in titles
    risk = next(i for i in res.section.items if getattr(i, "title", "") == "Risk")
    pred = np.array([m.nodes[t].pred for t in m.assign()])
    assert float(risk.rows[0][0]) == pytest.approx(np.mean(pred != m.y), abs=5e-4)


def test_cross_validation_and_split_sample_and_scale_target():
    ds = iris_ds()
    res = trees.tree_procedure(ds, "species", ["sl", "sw", "pl", "pw"], "crt",
                               settings={"min_parent": 10, "min_child": 5}, validation="cv",
                               folds=5)
    risk = next(i for i in res.section.items if getattr(i, "title", "") == "Risk")
    assert risk.rows[1][0] == "Cross-Validation" and 0 < float(risk.rows[1][1]) < 0.2
    res = trees.tree_procedure(ds, "pl", ["sl", "sw"], "chaid",
                               settings={"min_parent": 20, "min_child": 10},
                               validation="split", output={"table": True})
    titles = [getattr(i, "title", "") for i in res.section.items]
    assert "Gain Summary for Nodes (Training)" in titles and "Gain Summary for Nodes (Test)" \
        in titles
    gains = trees.gains(res.model, res.model.assign(), res.model.y, None)
    assert [g["mean"] for g in gains] == sorted([g["mean"] for g in gains], reverse=True)


def test_force_first_and_rules():
    ds = iris_ds()
    res = trees.tree_procedure(ds, "species", ["sw", "pl", "pw"], "crt", force_first=True,
                               settings={"min_parent": 10, "min_child": 5},
                               output={"rules": True})
    assert res.model.predictors[res.model.nodes[1].split.var].name == "sw"
    note = next(i for i in res.section.items if "Selection Rules" in getattr(i, "title", ""))
    assert "DO IF" in note.text and "VALUE(sw)" in note.text

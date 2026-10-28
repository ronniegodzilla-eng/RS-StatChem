"""WP9 TwoStep Cluster, Cluster Silhouettes and Nearest Neighbor against scikit-learn
(silhouette_samples, KNeighbors*) and the IBM TwoStep algorithm's structural properties."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score, silhouette_samples
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import knn, twostep


def _blobs(seed=3, n=80, means=(0, 3, 6), d=2, sd=0.45):
    r = np.random.default_rng(seed)
    X = np.vstack([r.normal(m, sd, (n, d)) for m in means])
    return X, np.repeat(np.arange(len(means)), n)


def test_xi_and_loglikelihood_distance():
    space = twostep.CFSpace(1, [], np.array([1.0]))
    # ξ of a cluster = −N/2 · log(σ²_k + σ²_vk); merging two singletons at distance δ
    a = (1.0, np.array([0.0]), np.array([0.0]), np.zeros(0))
    b = (np.array([1.0]), np.array([[2.0]]), np.array([[4.0]]), np.zeros((1, 0)))
    d = twostep.distance(space, a, b)[0]
    assert d == pytest.approx(-0.0 - 0.0 + 2 * 0.5 * math.log(1 + 1.0))
    space_c = twostep.CFSpace(0, [2], np.zeros(0))
    x = twostep.xi(space_c, np.array([4.0]), np.zeros((1, 0)), np.zeros((1, 0)),
                   np.array([[2.0, 2.0]]))
    assert x[0] == pytest.approx(-4 * math.log(2))      # −N · entropy


def test_twostep_auto_clusters_continuous_and_euclidean():
    X, truth = _blobs()
    for measure in ("ll", "euclidean"):
        res = twostep.twostep(X, np.zeros((len(X), 0), int), [], measure)
        assert res["k"] == 3
        assert adjusted_rand_score(truth, res["labels"]) == pytest.approx(1.0)
    rows = res["auto"]
    assert [r["J"] for r in rows][:3] == [1, 2, 3]
    assert rows[1]["ratio"] == pytest.approx(1.0)
    # BIC(J) = −2 Σ ξ_j + m_J log N, m_J = J(2 K_A + Σ(L_k − 1))
    h = res["hier"]
    assert rows[2]["crit"] == pytest.approx(-2 * h.xi_sum[3] + 3 * 4 * math.log(len(X)))


def test_twostep_mixed_and_fixed_and_outliers():
    X, truth = _blobs(n=60)
    cat = truth[:, None].copy()
    res = twostep.twostep(np.zeros((len(X), 0)), cat, [3], "ll")
    assert res["k"] == 3
    assert adjusted_rand_score(truth, res["labels"]) == pytest.approx(1.0)
    res = twostep.twostep(X, cat, [3], "ll", k=2)
    assert res["k"] == 2 and set(res["labels"]) == {1, 2}
    Xo = np.vstack([X, [[40.0, -40.0]]])
    res = twostep.twostep(Xo, np.zeros((len(Xo), 0), int), [], "ll", outliers=True)
    assert res["labels"][-1] == -1


def test_cf_tree_rebuilds_when_full():
    X, _ = _blobs(n=400, d=3)
    res = twostep.twostep(X, np.zeros((len(X), 0), int), [], "ll", max_entries=100)
    assert res["n_sub"] <= 100 and res["tree"].rebuilds >= 1 and res["k"] == 3


def test_silhouettes_equal_sklearn():
    X, truth = _blobs(sd=1.2)
    for metric, skm in (("euclidean", "euclidean"), ("cityblock", "manhattan")):
        D = twostep.pairwise(X, metric)
        s, nb = twostep.silhouette_samples(D, truth)
        np.testing.assert_allclose(s, silhouette_samples(X, truth, metric=skm), atol=1e-10)
    sil, s_ibm = twostep.ibm_silhouette(X, np.zeros((len(X), 0)), [], truth)
    assert -1 <= sil <= 1 and sil > 0.3


def _ds():
    r = np.random.default_rng(5)
    X, truth = _blobs(n=50)
    df = pd.DataFrame(X, columns=["a", "b"])
    df["noise"] = r.normal(size=len(df))
    df["g"] = r.integers(1, 3, len(df)).astype(float)
    df["cls"] = truth + 1.0
    df["y"] = X[:, 0] * 2 + r.normal(0, 0.1, len(df))
    ds = Dataset(df)
    for v in ("g", "cls"):
        ds.var(v).measure = "nominal"
    return ds


def test_twostep_and_silhouette_procedures():
    ds = _ds()
    res = twostep.twostep_procedure(ds, ["g"], ["a", "b"], save=True)
    titles = [getattr(i, "title", "") for i in res.section.items]
    for t in ("Auto-Clustering", "Cluster Distribution", "Centroids", "g", "Model Summary",
              "Predictor Importance"):
        assert t in titles
    assert res.saved["TSC"][1].notna().all()
    res = twostep.silhouettes_procedure(ds, "cls", ["a", "b"], save=True)
    tab = res.section.items[1]
    assert tab.rows[-1][0] == "Overall" and float(tab.rows[-1][2]) > 0.6
    with pytest.raises(Exception):
        twostep.twostep_procedure(ds, ["g"], ["a"], measure="euclidean")


def test_knn_neighbours_and_predictions_equal_sklearn():
    X, y = _blobs(sd=1.3)
    r = np.random.default_rng(0)
    perm = r.permutation(len(X))
    tr, te = perm[:180], perm[180:]
    for metric, skm in (("euclidean", "euclidean"), ("cityblock", "manhattan")):
        nb, d = knn.neighbours(X[te], X[tr], 5, metric)
        sk = KNeighborsClassifier(5, metric=skm).fit(X[tr], y[tr])
        dd, ii = sk.kneighbors(X[te])
        np.testing.assert_allclose(d, dd, atol=1e-10)
        pred, prob = knn.predict_from(nb, y[tr], 3, np.bincount(y[tr]).astype(float))
        np.testing.assert_array_equal(pred, sk.predict(X[te]))
        # Laplace correction (n_j + 1)/(k + J)
        np.testing.assert_allclose(prob, (sk.predict_proba(X[te]) * 5 + 1) / 8)
    yc = X[:, 0] + X[:, 1]
    nb, _ = knn.neighbours(X[te], X[tr], 4, "euclidean")
    pred, _ = knn.predict_from(nb, yc[tr], 0, None)
    np.testing.assert_allclose(pred, KNeighborsRegressor(4).fit(X[tr], yc[tr]).predict(X[te]))


def test_knn_feature_selection_and_k_selection():
    ds = _ds()
    res = knn.knn_procedure(ds, "cls", ["a", "b", "noise", "g"], auto_k=True, k_min=3, k_max=7,
                            select_features=True, max_features=None,
                            save=("predicted", "probability", "partition"))
    model = res.model
    assert 3 <= model.k <= 7
    names = ["g", "a", "b", "noise"]      # factors first, then covariates
    chosen = [names[g] for g in model.features]
    assert "a" in chosen and "noise" not in chosen
    titles = [getattr(i, "title", "") for i in res.section.items]
    for t in ("Classification Table", "Error Summary", "k Selection Error Log",
              "Feature Selection Error Log", "Feature Importance"):
        assert t in titles
    assert {"KNN_PredictedValue", "KNN_PredictedProbability_1", "Partition"} <= set(res.saved)
    res = knn.knn_procedure(ds, "y", ["a", "b"], k=3, scale_fn="median")
    err = next(i for i in res.section.items if getattr(i, "title", "") == "Error Summary")
    assert err.rows[0][1] == "Sum of Squares Error"

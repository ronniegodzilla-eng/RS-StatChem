"""WP9 Partial Least Squares (against scikit-learn's PLSRegression), MARS (against R earth's
published ``trees`` example) and Kernel Ridge (against scikit-learn's KernelRidge and
GridSearchCV)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, KFold

from rsstatchem.core.analysis import chemometrics as cm
from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import kernel_ridge as krr
from rsstatchem.core.stats import mars, pls_regression

# R datasets::trees (Ryan, Joiner & Ryan 1976), 31 black cherry trees
GIRTH = [8.3, 8.6, 8.8, 10.5, 10.7, 10.8, 11.0, 11.0, 11.1, 11.2, 11.3, 11.4, 11.4, 11.7, 12.0,
         12.9, 12.9, 13.3, 13.7, 13.8, 14.0, 14.2, 14.5, 16.0, 16.3, 17.3, 17.5, 17.9, 18.0,
         18.0, 20.6]
HEIGHT = [70, 65, 63, 72, 81, 83, 66, 75, 80, 75, 79, 76, 76, 69, 75, 74, 85, 86, 71, 64, 78,
          80, 74, 72, 77, 81, 82, 80, 80, 80, 87]
VOLUME = [10.3, 10.3, 10.2, 16.4, 18.8, 19.7, 15.6, 18.2, 22.6, 19.9, 24.2, 21.0, 21.4, 21.3,
          19.1, 22.2, 33.8, 27.4, 25.7, 24.9, 34.5, 31.7, 36.3, 38.3, 42.6, 55.4, 55.7, 58.3,
          51.5, 51.0, 77.0]


def trees_ds():
    ds = Dataset(pd.DataFrame({"Girth": GIRTH, "Height": np.array(HEIGHT, float),
                               "Volume": VOLUME}))
    for v in ds.variables:
        v.measure = "scale"
    return ds


def test_trees_data_sanity():
    A = np.column_stack([np.ones(31), GIRTH, HEIGHT])
    b = np.linalg.lstsq(A, VOLUME, rcond=None)[0]
    np.testing.assert_allclose(b, [-57.9877, 4.7082, 0.3393], atol=1e-4)   # lm(Volume ~ .)


def test_mars_reproduces_r_earth_trees():
    # R: summary(earth(Volume ~ ., data = trees)) prints
    #   (Intercept) 29.0599535; h(14.2-Girth) -3.4198062; h(Girth-14.2) 6.2295143;
    #   h(Height-75) 0.5813644; Selected 4 of 5 terms, and 2 of 2 predictors;
    #   Termination condition: RSq changed by less than 0.001 at 5 terms;
    #   GCV 11.25439  RSS 209.1139  GRSq 0.959692  RSq 0.9742029
    X = np.column_stack([GIRTH, HEIGHT]).astype(float)
    m = mars.earth(X, np.array(VOLUME), ["Girth", "Height"])
    coef = {t.label(m.names): c for t, c in zip(m.terms, m.coef)}
    assert coef["(Intercept)"] == pytest.approx(29.0599535, abs=5e-7)
    assert coef["h(14.2-Girth)"] == pytest.approx(-3.4198062, abs=5e-7)
    assert coef["h(Girth-14.2)"] == pytest.approx(6.2295143, abs=5e-7)
    assert coef["h(Height-75)"] == pytest.approx(0.5813644, abs=5e-7)
    assert (len(m.terms), m.forward_terms) == (4, 5)
    assert m.termination == "RSq changed by less than 0.001 at 5 terms"
    assert m.gcv == pytest.approx(11.25439, abs=5e-6)
    assert m.rss == pytest.approx(209.1139, abs=5e-5)
    assert m.grsq == pytest.approx(0.959692, abs=5e-7)
    assert m.rsq == pytest.approx(0.9742029, abs=5e-8)
    imp = {name: n for name, n, _, _ in mars.evimp(m)}
    assert imp == {"Girth": 3, "Height": 1}


def test_mars_procedure_degree2_and_binomial():
    ds = trees_ds()
    res = mars.mars_procedure(ds, "Volume", ["Girth", "Height"], save=("predicted",
                                                                        "residuals"))
    summ = next(i for i in res.section.items if getattr(i, "title", "") == "Model Summary")
    assert dict((r[0], r[1]) for r in summ.rows)["GCV"] == "11.2544"
    assert res.saved["MARS_Predicted"][1].notna().all()
    res = mars.mars_procedure(ds, "Volume", ["Girth", "Height"], degree=2)
    assert res.model.penalty == 3.0
    r = np.random.default_rng(0)
    x = r.uniform(-3, 3, 300)
    yb = (r.random(300) < 1 / (1 + np.exp(-2 * np.maximum(0, x)))).astype(float)
    m = mars.earth(x[:, None], yb, ["x"], binomial=True)
    import statsmodels.api as sm
    fit = sm.GLM(yb, m.basis(x[:, None]), family=sm.families.Binomial()).fit()
    np.testing.assert_allclose(m.coef, fit.params, rtol=1e-6, atol=1e-8)
    assert m.glm["deviance"] == pytest.approx(fit.deviance, rel=1e-8)


def test_pls_equals_sklearn():
    r = np.random.default_rng(3)
    X = r.normal(size=(40, 5))
    Y = np.column_stack([X @ [1, -1, 0.5, 0, 0], X @ [0, 1, 1, -0.5, 0]]) + \
        0.3 * r.normal(size=(40, 2))
    m = cm.pls_nipals(X, Y, 3, scale=True)
    sk = PLSRegression(3, scale=True, tol=1e-14, max_iter=10000).fit(X, Y)
    # PLS2: NIPALS iterated to convergence on both sides
    np.testing.assert_allclose(m.coef, sk.coef_.T, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(np.abs(m.x_scores), np.abs(sk.x_scores_), rtol=1e-6,
                               atol=1e-6)
    m1 = cm.pls_nipals(X, Y[:, 0], 3, scale=True)        # PLS1: exact (one iteration)
    sk1 = PLSRegression(3, scale=True).fit(X, Y[:, 0])
    np.testing.assert_allclose(m1.coef[:, 0], sk1.coef_.ravel(), rtol=1e-10, atol=1e-12)
    vip = pls_regression.cumulative_vip(m, X, Y)
    np.testing.assert_allclose(vip[:, -1], cm.vip(m, X, Y))
    # VIP² averages to 1 over the predictors (Wold)
    assert np.mean(vip[:, -1] ** 2) == pytest.approx(1.0)


def test_pls_procedure_tables():
    r = np.random.default_rng(1)
    df = pd.DataFrame(r.normal(size=(50, 3)), columns=["a", "b", "c"])
    df["g"] = r.integers(1, 4, 50).astype(float)
    df["y"] = df["a"] * 2 - df["b"] + df["g"] + r.normal(0, 0.2, 50)
    ds = Dataset(df)
    ds.var("g").measure = "nominal"
    res = pls_regression.pls_procedure(ds, ["y"], ["g"], ["a", "b", "c"], 3,
                                       plots=("vip", "scores", "weights"),
                                       save=("predicted", "residuals", "distance", "scores"))
    pve = res.section.items[1]
    assert pve.title == "Proportion of Variance Explained" and len(pve.rows) == 3
    r2 = [float(row[4]) for row in pve.rows]
    assert r2 == sorted(r2) and r2[-1] > 0.9
    par = next(i for i in res.section.items if getattr(i, "title", "") == "Parameters")
    assert par.rows[0][0] == "(Constant)" and len(par.rows) == 1 + 3 + 3
    assert {"PredictedValue_1", "Residual_1", "DModX", "LatentFactor_1"} <= set(res.saved)


@pytest.mark.parametrize("kind,params", [("rbf", {}), ("rbf", {"gamma": 0.3}),
                                         ("poly", {"degree": 2, "coef0": 0.5}),
                                         ("linear", {}), ("laplacian", {}),
                                         ("sigmoid", {"gamma": 0.1}), ("cosine", {}),
                                         ("chi2", {"gamma": 0.7}), ("additive_chi2", {})])
def test_kernel_ridge_equals_sklearn(kind, params):
    r = np.random.default_rng(2)
    X = r.uniform(0, 1, (35, 3))
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2
    sk = KernelRidge(alpha=0.5, kernel=kind, **params).fit(X, y)
    p = {"gamma": params.get("gamma"), "degree": params.get("degree", 3),
         "coef0": params.get("coef0", 1)}
    dual = krr.fit(X, y, 0.5, kind, **p)
    np.testing.assert_allclose(dual, sk.dual_coef_, rtol=1e-7, atol=1e-9)
    Xn = r.uniform(0, 1, (5, 3))
    np.testing.assert_allclose(krr.kernel(Xn, X, kind, **p) @ dual, sk.predict(Xn),
                               rtol=1e-7, atol=1e-9)


def test_kernel_ridge_grid_search_equals_gridsearchcv():
    r = np.random.default_rng(5)
    X = r.normal(size=(60, 2))
    y = X[:, 0] ** 2 + X[:, 1] + 0.1 * r.normal(size=60)
    grid = [{"alpha": a, "kind": "rbf", "gamma": g, "degree": 3, "coef0": 1.0}
            for a in (1.0, 0.1, 0.01) for g in (0.1, 1.0)]
    rows, rank = krr.grid_search(X, y, grid, 5)
    gs = GridSearchCV(KernelRidge(kernel="rbf"), {"alpha": [1.0, 0.1, 0.01],
                                                  "gamma": [0.1, 1.0]},
                      cv=KFold(5)).fit(X, y)
    ours = {(p["alpha"], p["gamma"]): m for p, m, _ in rows}
    for params, score in zip(gs.cv_results_["params"], gs.cv_results_["mean_test_score"]):
        assert ours[(params["alpha"], params["gamma"])] == pytest.approx(score, rel=1e-9)
    best = rows[int(np.argmin(rank))][0]
    assert (best["alpha"], best["gamma"]) == (gs.best_params_["alpha"],
                                              gs.best_params_["gamma"])
    ds = Dataset(pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "y": y}))
    res = krr.krr_procedure(ds, ["y"], ["x1", "x2"], ["rbf"], [1.0, 0.1, 0.01], [0.1, 1.0],
                            save=("predicted", "residuals"))
    titles = [getattr(i, "title", "") for i in res.section.items]
    assert "Cross-Validation Results" in titles and "Model Summary" in titles
    assert res.model["params"]["alpha"] == gs.best_params_["alpha"]

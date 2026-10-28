"""WP10: SPSS Categories (CATREG, CATPCA, MCA, OVERALS, PREFSCAL).

References (no R available, so closed forms that the Gifi/homals solutions provably equal,
plus statsmodels / scikit-learn as test-only references):

* CATREG with every variable numeric = standardized OLS; ridge = (R + lambda2 I)^-1 r;
  lasso/elastic net = the coordinate-descent solutions with alpha = lambda1/2 (IBM CATREG
  Algorithms, van der Kooij 2007 update rules);
* one nominal predictor: CATREG R^2 = the correlation ratio eta^2;
* CATPCA numeric = PCA of the correlation matrix; MCA eigenvalues = m x principal inertias of
  correspondence analysis of the indicator matrix (Gifi 1990 ch. 3; the closed form that
  homals and SPSS MCA converge to); CATPCA single nominal in one dimension = first MCA
  dimension (Gifi 1990, sec. 4.x);
* OVERALS with two numeric sets: (K*eigenvalue - 1)/(K - 1) = canonical correlations;
* Max (1960) minimum-distortion quantizer of N(0,1) (SPSS discretization "grouping, normal"):
  3 levels thresholds +-0.6120, 4 levels 0 and +-0.9816 (Max's Table I).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import optscale as osc
from rsstatchem.core.stats import optscale_engine as E


def _data(n=150, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(1, 6, size=(n, 3)).astype(float)
    y = 0.5 * X[:, 0] - 0.8 * X[:, 1] + 0.3 * X[:, 2] ** 1.5 + rng.normal(0, 1, n)
    y = np.round(y - y.min() + 1)
    return X, y


def _fit(level, lam1=0.0, lam2=0.0, eps=1e-12):
    X, y = _data()
    resp = E.make_var("y", y, level)
    preds = [E.make_var(f"x{j}", X[:, j], level) for j in range(3)]
    return X, y, E.catreg_fit(resp, preds, np.ones(len(y)), lam1, lam2, 2000, eps)


def _std(X):
    return (X - X.mean(0)) / X.std(0)


def test_catreg_numeric_equals_ols():
    X, y, f = _fit("nume")
    ols = sm.OLS(_std(y[:, None])[:, 0], _std(X)).fit()
    assert f.r2 == pytest.approx(ols.rsquared, abs=1e-9)
    assert np.allclose(f.b, ols.params, atol=1e-6)


def test_catreg_ridge_closed_form():
    X, y, f = _fit("nume", lam2=0.3)
    R = np.corrcoef(X.T)
    r = np.array([np.corrcoef(X[:, j], y)[0, 1] for j in range(3)])
    assert np.allclose(f.b, np.linalg.solve(R + 0.3 * np.eye(3), r), atol=1e-8)


def test_catreg_lasso_and_elastic_net_match_coordinate_descent():
    lm = pytest.importorskip("sklearn.linear_model")
    X, y, f = _fit("nume", lam1=0.2)
    las = lm.Lasso(alpha=0.1, fit_intercept=False, tol=1e-14, max_iter=100000).fit(
        _std(X), _std(y[:, None])[:, 0])
    assert np.allclose(f.b, las.coef_, atol=1e-7)
    X, y, f = _fit("nume", lam1=0.2, lam2=0.5)
    # sklearn: alpha*l1_ratio = lambda1/2, alpha*(1 - l1_ratio) = lambda2
    alpha = 0.1 + 0.5
    en = lm.ElasticNet(alpha=alpha, l1_ratio=0.1 / alpha, fit_intercept=False, tol=1e-14,
                       max_iter=100000).fit(_std(X), _std(y[:, None])[:, 0])
    assert np.allclose(f.b, en.coef_, atol=1e-7)


def test_catreg_one_nominal_predictor_is_eta_squared():
    rng = np.random.default_rng(3)
    g = rng.integers(1, 5, 200).astype(float)
    y = np.round(np.array([0, 2, 1, 3])[g.astype(int) - 1] + rng.normal(0, 1, 200)) + 5
    f = E.catreg_fit(E.make_var("y", y, "nume"), [E.make_var("g", g, "nomi")],
                     np.ones(200), eps=1e-12, maxiter=500)
    means = pd.Series(y).groupby(g).transform("mean").to_numpy()
    eta2 = np.sum((means - y.mean()) ** 2) / np.sum((y - y.mean()) ** 2)
    assert f.r2 == pytest.approx(eta2, abs=1e-8)


def test_catreg_levels_order_and_monotone_quantifications():
    r2 = {lv: _fit(lv, eps=1e-8)[2].r2 for lv in ("nume", "spord", "ordi", "nomi")}
    assert r2["nume"] <= r2["ordi"] + 1e-6 <= r2["nomi"] + 2e-6
    assert r2["spord"] <= r2["ordi"] + 1e-6
    f = _fit("ordi", eps=1e-8)[2]
    for y in [f.yr] + f.yp:
        assert np.all(np.diff(y) >= -1e-10) or np.all(np.diff(y) <= 1e-10)


def test_catpca_numeric_equals_pca():
    X, y = _data()
    Z = np.column_stack([X, y])
    vs = [E.make_var(f"v{j}", Z[:, j], "nume") for j in range(4)]
    fit = E.gifi_pca(vs, np.ones(len(y)), 2, eps=1e-10, maxiter=500)
    ev, V = np.linalg.eigh(np.corrcoef(Z.T))
    ev, V = ev[::-1], V[:, ::-1]
    assert np.allclose(fit.eig, ev[:2], atol=1e-7)
    load = np.array(fit.loadings)
    assert np.allclose(np.abs(load), np.abs(V[:, :2] * np.sqrt(ev[:2])), atol=1e-5)


def test_mca_equals_correspondence_analysis_of_indicator_matrix():
    X, y = _data()
    Z = np.column_stack([X, y])
    vs = [E.make_var(f"v{j}", Z[:, j], "mnom") for j in range(4)]
    fit = E.gifi_pca(vs, np.ones(len(y)), 3, eps=1e-10, maxiter=1000)
    assert np.allclose(fit.eig, E.mca_closed_form(vs, np.ones(len(y)), 3), atol=1e-6)


def test_catpca_single_nominal_one_dimension_equals_mca_first_dimension():
    X, y = _data()
    Z = np.column_stack([X, y])
    mc = E.mca_closed_form([E.make_var(f"v{j}", Z[:, j], "mnom") for j in range(4)],
                           np.ones(len(y)), 1)
    fit = E.gifi_pca([E.make_var(f"v{j}", Z[:, j], "nomi") for j in range(4)],
                     np.ones(len(y)), 1, eps=1e-11, maxiter=5000)
    assert fit.eig[0] == pytest.approx(mc[0], abs=1e-6)


def test_overals_two_numeric_sets_equals_canonical_correlation():
    from statsmodels.multivariate.cancorr import CanCorr

    rng = np.random.default_rng(7)
    A = rng.normal(size=(150, 2))
    B = A @ np.array([[1, .5], [.2, 1]]) + rng.normal(size=(150, 2))
    Ad = np.round(A * 2)
    Bd = np.round(B * 2)
    Ad -= Ad.min(0) - 1
    Bd -= Bd.min(0) - 1
    sets = [[E.make_var(f"a{j}", Ad[:, j], "nume") for j in range(2)],
            [E.make_var(f"b{j}", Bd[:, j], "nume") for j in range(2)]]
    fit = E.overals(sets, np.ones(150), 2, eps=1e-11, maxiter=3000)
    assert np.allclose(2 * fit.eig - 1, CanCorr(Bd, Ad).cancorr, atol=1e-6)
    assert fit.fit == pytest.approx(fit.eig.sum())


def test_max_quantizer_thresholds():
    assert np.allclose(E.lloyd_max_normal(3), [-0.6120, 0.6120], atol=1e-4)
    assert np.allclose(E.lloyd_max_normal(4), [-0.9816, 0, 0.9816], atol=1e-4)
    d = E.discretize(np.random.default_rng(0).normal(size=500), "normal", 7)
    assert set(np.unique(d)) <= set(range(1, 8)) and len(np.unique(d)) == 7


def test_discretization_methods():
    x = np.array([0.5, 1.5, 1.5, 3.2, 8.0, np.nan])
    assert list(E.discretize(x, "rank")[:5]) == [1, 2, 2, 3, 4]
    assert np.isnan(E.discretize(x, "rank")[5])
    assert list(E.discretize(x, "equal", interval=2.0)[:5]) == [1, 1, 1, 2, 4]
    m = E.discretize(x, "multiply")
    assert np.nanmin(m) == 1
    assert list(E.discretize(np.array(["b", "a", "b"], object), strings=True)) == [2, 1, 2]


def test_ispline_basis_monotone():
    S = E.ispline_basis(np.arange(1, 11, dtype=float), np.ones(10), 2, 2)
    assert S.shape == (10, 4)
    assert np.all(np.diff(S, axis=0) >= -1e-12)
    assert np.allclose(S[0], 0) and np.allclose(S[-1], 1)


def test_prefscal_recovers_perfect_unfolding():
    rng = np.random.default_rng(3)
    R = rng.normal(size=(12, 2))
    C = rng.normal(size=(7, 2)) * 1.3
    D = E._dist(R, C)
    fit = E.prefscal(D * 3, 2, "none", False, maxiter=3000)
    m = E.unfolding_measures(fit, D * 3, np.ones_like(D), False)
    assert m["normalized"] < 1e-4 and m["rpo"] > 0.99
    fit = E.prefscal(np.exp(D), 2, "ordinal", False, maxiter=3000)
    m = E.unfolding_measures(fit, np.exp(D), np.ones_like(D), False)
    assert m["normalized"] < 1e-3 and m["spearman"] > 0.99
    assert m["cv_trans"] > 0.2   # the penalty keeps the transformation non-degenerate


def _ds():
    rng = np.random.default_rng(1)
    n = 120
    y = rng.integers(1, 6, n).astype(float)
    df = pd.DataFrame({"y": y, "a": np.clip(y + rng.integers(-1, 2, n), 1, 5),
                       "b": rng.integers(1, 4, n).astype(float),
                       "c": rng.integers(1, 6, n).astype(float),
                       "s": rng.choice(["lo", "mid", "hi"], n)})
    df.loc[[3, 10], "b"] = np.nan
    return Dataset(df)


def _table(sec, title):
    return next(t for t in sec.items if getattr(t, "title", "") == title)


def test_catreg_output_tables_and_save():
    ds = _ds()
    r = osc.catreg(ds, {"name": "y", "level": "nume"},
                   [{"name": "a", "level": "ordi"}, {"name": "b", "level": "nomi"},
                    {"name": "s", "level": "nomi"}], bootstrap_se=20, show_history=True,
                   correlations=True, transformation_plots=("a",), save_predicted=True,
                   save_residuals=True, save_transformed=True)
    ms = _table(r.section, "Model Summary")
    r2 = float(ms.rows[0][1])
    assert 0 < r2 < 1 and float(ms.rows[0][3]) == pytest.approx(1 - r2, abs=2e-3)
    assert _table(r.section, "Case Processing Summary").rows[-1][1] == "118"
    coef = _table(r.section, "Coefficients")
    assert len(coef.rows) == 3 and "Bootstrap (20)" in coef.columns[2][1]
    assert {"PRE", "RES", "TRA1"} <= set(r.saved)
    imp = [float(x[4]) for x in _table(r.section, "Correlations and Tolerance").rows]
    assert sum(imp) == pytest.approx(1.0, abs=5e-3)   # Pratt importances sum to one


def test_catreg_regularization_with_cross_validation():
    ds = _ds()
    r = osc.catreg(ds, {"name": "y", "level": "nume"},
                   [{"name": c, "level": "nume"} for c in "abc"], regularization="ridge",
                   ridge=(0.0, 0.4, 0.2), resampling="cv", folds=5, bootstrap_se=0)
    t = _table(r.section, "Ridge Regression")
    assert len(t.rows) == 3 and t.rows[0][4] == "1.000"     # unpenalized: sum ratio 1
    assert all(row[2] != "" for row in t.rows)               # EPE filled
    r = osc.catreg(ds, {"name": "y", "level": "nume"},
                   [{"name": c, "level": "nume"} for c in "abc"], regularization="enet",
                   ridge=(0.0, 0.5, 0.5), lasso=(0.0, 0.2, 0.2), bootstrap_se=0)
    assert len(_table(r.section, "Elastic Net Regression").rows) == 4


def test_catpca_and_mca_outputs():
    ds = _ds()
    r = osc.catpca(ds, [{"name": "y", "level": "ordi"}, {"name": "a", "level": "spord"},
                        {"name": "b", "level": "mnom"}, {"name": "c", "level": "nume"}],
                   quantifications=("a", "b"), correlations_transformed=True,
                   object_scores=True, plots=("objects", "biplot", "loadings", "joint"),
                   transformation_plots=("a",), save_scores=True, save_transformed=True)
    ms = _table(r.section, "Model Summary")
    eig = [float(x[2]) for x in ms.rows[:2]]
    assert float(ms.rows[2][2]) == pytest.approx(sum(eig), abs=2e-3)
    assert "OBSCO1" in r.saved and "TRA1" in r.saved
    assert sum(1 for i in r.section.items if getattr(i, "kind", "") == "image") == 5
    r = osc.mca(ds, ["y", "a", "c"])
    ms = _table(r.section, "Model Summary")
    vs = [E.make_var(c, ds.df[c].to_numpy(float), "mnom") for c in ("y", "a", "c")]
    assert float(ms.rows[0][2]) == pytest.approx(E.mca_closed_form(vs, np.ones(120), 1)[0],
                                                 abs=2e-3)
    assert float(ms.rows[0][3]) == pytest.approx(float(ms.rows[0][2]) / 3, abs=2e-3)
    disc = _table(r.section, "Discrimination Measures")
    assert float(disc.rows[3][1]) == pytest.approx(float(ms.rows[0][2]), abs=3e-3)


def test_catpca_missing_strategies():
    ds = _ds()
    for miss in ("passive", "mode", "extra", "listwise"):
        r = osc.catpca(ds, [{"name": c, "level": "ordi"} for c in "yab"], missing=miss,
                       plots=())
        used = _table(r.section, "Case Processing Summary").rows[-1][1]
        assert used == ("118" if miss == "listwise" else "120")


def test_overals_and_prefscal_outputs():
    ds = _ds()
    r = osc.overals(ds, [[{"name": "y", "level": "nume"}, {"name": "b", "level": "mnom"}],
                         [{"name": "a", "level": "ordi"}, {"name": "c", "level": "nomi"}]],
                    centroids=True, frequencies=True, object_scores=True,
                    plots=("objects", "loadings", "centroids"), save_scores=True)
    s = _table(r.section, "Summary of Analysis")
    eig = [float(x) for x in s.rows[3][1:3]]
    assert float(s.rows[4][3]) == pytest.approx(sum(eig), abs=2e-3)
    r = osc.prefscal(ds, ["y", "a", "b", "c"], transform="linear", maxiter=100,
                     plots=("joint", "transformation", "shepard"), history=True)
    m = _table(r.section, "Measures")
    assert m.rows[0][2] != "" and len(_table(r.section, "Final Column Coordinates").rows) == 4


def test_selftest():
    osc.selftest()

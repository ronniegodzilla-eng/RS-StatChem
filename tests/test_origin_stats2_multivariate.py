"""WP-3 multivariate analyses and chemometrics: PLS (NIPALS / SIMPLS), PCA, k-means,
hierarchical clustering and discriminant analysis against scikit-learn / SciPy (test-only
references) on a synthetic NIR-like set and Fisher's Iris data."""
from __future__ import annotations

import numpy as np
import pytest

from rsstatchem.core.analysis import chemometrics as cm
from rsstatchem.core.stats import classify as cl
from rsstatchem.ui.origin import stat_tests as st
from rsstatchem.ui.origin.model import Workbook

sklearn = pytest.importorskip("sklearn")


def _book(cols, name="Book1") -> Workbook:
    b = Workbook.new(name, n_cols=len(cols))
    ws = b.sheets[0]
    ws.set_rows(max(len(c) for c in cols))
    for j, c in enumerate(cols):
        ws.set_column_values(j, list(c))
    return b


def _col(j: int) -> str:
    s = ""
    j += 1
    while j:
        j, r = divmod(j - 1, 26)
        s = chr(65 + r) + s
    return s


def R(j: int) -> str:
    return f"[Book1]Sheet1!{_col(j)}"


def nir(n=40, p=60, seed=0):
    """Synthetic NIR-like spectra: three Gaussian bands mixed by concentrations + noise."""
    rng = np.random.default_rng(seed)
    wl = np.linspace(0, 1, p)
    conc = rng.uniform(0, 1, (n, 3))
    bands = np.vstack([np.exp(-((wl - m) / 0.08) ** 2) for m in (0.3, 0.5, 0.7)])
    X = conc @ bands + 0.01 * rng.standard_normal((n, p)) + 0.2 * wl
    return X, conc


def iris():
    from sklearn.datasets import load_iris
    d = load_iris()
    return d.data, d.target


def _cell(report, table, row, col):
    t = report.table(table)
    r = next(r for r in t["rows"] if r[0] == row)
    j = t["columns"].index(col) if isinstance(col, str) else col
    return float(r[j])


# ====================================================================== PLS
@pytest.mark.parametrize("ncomp", [1, 2, 4, 6])
def test_pls1_nipals_matches_sklearn(ncomp):
    from sklearn.cross_decomposition import PLSRegression
    X, C = nir()
    y = C[:, 0]
    ref = PLSRegression(ncomp).fit(X, y)
    m = cm.pls_nipals(X, y, ncomp)
    assert m.predict(X).ravel() == pytest.approx(ref.predict(X).ravel(), abs=1e-8)
    assert np.abs(m.x_scores) == pytest.approx(np.abs(ref.x_scores_), abs=1e-8)
    assert m.x_weights == pytest.approx(ref.x_weights_, abs=1e-8)
    assert m.x_loadings == pytest.approx(ref.x_loadings_, abs=1e-8)
    assert m.coef.ravel() == pytest.approx(np.ravel(ref.coef_), abs=1e-8)
    assert m.x_rotations == pytest.approx(ref.x_rotations_, abs=1e-8)
    assert m.transform(X) == pytest.approx(ref.transform(X), abs=1e-8)
    # SIMPLS gives the same model for one response (de Jong 1993)
    s = cm.pls_simpls(X, y, ncomp)
    assert s.predict(X).ravel() == pytest.approx(m.predict(X).ravel(), abs=1e-9)


def test_pls2_nipals_and_simpls():
    from sklearn.cross_decomposition import PLSRegression
    X, C = nir(50, 80, 3)
    Y = C[:, :2]
    ref = PLSRegression(4, tol=1e-15, max_iter=10000).fit(X, Y)
    m = cm.pls_nipals(X, Y, 4)
    assert m.predict(X) == pytest.approx(ref.predict(X), abs=1e-6)
    assert np.abs(m.x_scores) == pytest.approx(np.abs(ref.x_scores_), abs=1e-5)
    s = cm.pls_simpls(X, Y, 4)
    Xs = (X - X.mean(0)) / X.std(0, ddof=1)
    # de Jong's definitions: T = X0 R with orthonormal scores, B = R Q'
    assert s.x_scores == pytest.approx(Xs @ s.x_rotations, abs=1e-10)
    assert s.x_scores.T @ s.x_scores == pytest.approx(np.eye(4), abs=1e-10)
    assert s.coef_scaled == pytest.approx(s.x_rotations @ s.y_loadings.T, abs=1e-12)
    # SIMPLS and NIPALS fit PLS2 data comparably well
    e_n = ((Y - m.predict(X)) ** 2).sum()
    e_s = ((Y - s.predict(X)) ** 2).sum()
    assert e_s == pytest.approx(e_n, rel=0.05)


def test_pls_cross_validation_and_vip():
    X, C = nir(30, 40, 5)
    y = C[:, 1]
    cv = cm.pls_cv(X, y, 4, how="loo")
    press = np.zeros(4)
    for i in range(len(y)):   # manual leave-one-out
        tr = np.arange(len(y)) != i
        for a in range(1, 5):
            mdl = cm.pls_nipals(X[tr], y[tr], a)
            press[a - 1] += float((y[i] - mdl.predict(X[i:i + 1])[0, 0]) ** 2)
    assert cv["press"] == pytest.approx(press, rel=1e-10)
    assert cv["q2"] == pytest.approx(1 - press / ((y - y.mean()) ** 2).sum())
    assert len(cm.cv_folds(30, "venetian", 7)) == 7 and len(cm.cv_folds(30, "kfold", 5)) == 5
    m = cm.pls_nipals(X, y, 3)
    v = cm.vip(m, X, y)
    assert np.mean(v ** 2) == pytest.approx(1.0)   # VIP property: Σ VIP² = p
    ex = cm.explained(m, X, y)
    pred = m.predict(X).ravel()
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    assert ex["y"].sum() == pytest.approx(r2, rel=1e-8)


def test_pls_report():
    X, C = nir(30, 40, 6)
    cols = [X[:, j] for j in range(X.shape[1])] + [C[:, 0]]
    books = {"Book1": _book(cols)}
    rep, text = st.run("pls", books, {"ranges": [R(j) for j in range(40)],
                                      "response": [R(40)], "cv": "loo", "max_comp": 6})
    ncomp = int(dict(rep.table("Notes")["rows"])["Number of Components"])
    cv = cm.pls_cv(X, C[:, 0], 6)
    assert ncomp == int(np.argmin(cv["press"]) + 1)
    m = cm.pls_nipals(X, C[:, 0], ncomp)
    assert _cell(rep, "Coefficients", "Intercept", 1) == pytest.approx(m.intercept[0],
                                                                       rel=1e-5)
    assert rep.table("Variable Importance in the Projection (VIP)") is not None
    assert len(rep.graphs()) >= 5
    rep2, _ = st.run("pls", books, rep.op["settings"], report=rep)
    assert rep2 is rep


# ====================================================================== PCA
def test_pca_against_sklearn_iris():
    from sklearn.decomposition import PCA
    X, _ = iris()
    cov = cm.pca(X, "covariance")
    ref = PCA().fit(X)
    assert cov.eigenvalues == pytest.approx(ref.explained_variance_, rel=1e-10)
    S = ref.transform(X)
    for k in range(4):   # up to sign
        assert np.abs(np.corrcoef(cov.scores[:, k], S[:, k])[0, 1]) == pytest.approx(1.0)
        assert np.abs(cov.scores[:, k]) == pytest.approx(np.abs(S[:, k]), abs=1e-9)
    cor = cm.pca(X, "correlation")
    Z = (X - X.mean(0)) / X.std(0, ddof=1)
    ref2 = PCA().fit(Z)
    assert cor.eigenvalues == pytest.approx(ref2.explained_variance_, rel=1e-10)
    assert cor.eigenvalues.sum() == pytest.approx(4.0)
    from rsstatchem.core.stats.factor import extract_pc
    assert np.abs(cor.loadings[:, :2]) == pytest.approx(
        np.abs(extract_pc(np.corrcoef(X, rowvar=False), 2)), abs=1e-10)


def test_pca_report_iris():
    X, y = iris()
    books = {"Book1": _book([X[:, j] for j in range(4)] + [[str(v) for v in y]])}
    rep, _ = st.run("pca", books, {"ranges": [R(j) for j in range(4)], "group": R(4)})
    ev = rep.table("Eigenvalues of the Correlation Matrix")["rows"]
    assert float(ev[0][1]) == pytest.approx(2.91850, rel=1e-5)   # classic Iris value
    assert [i["title"] for i in rep.graphs()] == ["Scree Plot", "Loading Plot", "Score Plot",
                                                  "Biplot"]


# ====================================================================== clustering
def test_kmeans_against_sklearn():
    from sklearn.cluster import KMeans
    X, _ = iris()
    init = cl.initial_centres(X, 3)
    res = cl.kmeans(X, 3, 300, 0.0, centres=init)
    ref = KMeans(3, init=init, n_init=1, max_iter=300, tol=0, algorithm="lloyd").fit(X)
    assert res["labels"].tolist() == ref.labels_.tolist()
    assert res["final"] == pytest.approx(ref.cluster_centers_, abs=1e-10)
    books = {"Book1": _book([X[:, j] for j in range(4)])}
    rep, _ = st.run("kmeans", books, {"ranges": [R(j) for j in range(4)], "k": 3,
                                      "max_iter": 300})
    sizes = sorted(int(r[1]) for r in rep.table("Cluster Summary")["rows"])
    assert sizes == sorted(np.bincount(ref.labels_).tolist())


@pytest.mark.parametrize("method,scipy_method,measure", [
    ("single", "single", "euclid"), ("complete", "complete", "euclid"),
    ("baverage", "average", "euclid"), ("ward", "ward", "seuclid")])
def test_hierarchical_heights_against_scipy(method, scipy_method, measure):
    from scipy.cluster.hierarchy import linkage
    rng = np.random.default_rng(4)
    X = rng.normal(size=(25, 3))
    merges, Z = cl.agglomerate(cl.proximities(X, measure), method)
    ref = linkage(X, scipy_method)
    if method == "ward":   # SPSS coefficients: cumulative error sum of squares
        coef = np.array([c for _, _, c in merges])
        assert coef == pytest.approx(np.cumsum(ref[:, 2] ** 2 / 2), rel=1e-9)
    else:
        assert np.sort(Z[:, 2]) == pytest.approx(np.sort(ref[:, 2]), rel=1e-9)
    from rsstatchem.ui.origin.stat_multivariate import dendrogram_segments
    xs, ys, leaves = dendrogram_segments(Z, 25)
    assert sorted(leaves) == list(range(25)) and np.isnan(xs).sum() == 24


def test_hca_report():
    X, _ = iris()
    books = {"Book1": _book([X[:, j] for j in range(4)])}
    rep, _ = st.run("hca", books, {"ranges": [R(j) for j in range(4)], "method": "ward",
                                   "measure": "seuclid", "k": 3})
    assert len(rep.table("Agglomeration Schedule")["rows"]) == 149
    assert {r[1] for r in rep.table("Cluster Membership (3 clusters)")["rows"]} == \
        {"1", "2", "3"}
    rep2, _ = st.run("hca", books, {"ranges": [R(j) for j in range(4)],
                                    "cluster": "variables", "measure": "correlation"})
    assert len(rep2.table("Agglomeration Schedule")["rows"]) == 3


# ====================================================================== discriminant
def test_lda_against_sklearn_iris():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    X, y = iris()
    ref = LinearDiscriminantAnalysis().fit(X, y)
    names = np.array(["setosa", "versicolor", "virginica"])[y]
    books = {"Book1": _book([X[:, j] for j in range(4)] + [list(names)])}
    rep, _ = st.run("lda", books, {"ranges": [R(j) for j in range(4)], "group": R(4)})
    ev = rep.table("Canonical Discriminant Functions: Eigenvalues")["rows"]
    assert [float(r[2]) / 100 for r in ev] == pytest.approx(ref.explained_variance_ratio_,
                                                            rel=1e-4)
    assert float(ev[0][1]) == pytest.approx(32.192, rel=1e-4)   # Fisher's Iris (SPSS)
    from rsstatchem.ui.origin.stat_multivariate import lda_fit
    fit = lda_fit(X, y, 3, np.full(3, 1 / 3))
    assert fit["pred"].tolist() == ref.predict(X).tolist()
    tr = rep.table("Classification Summary (Training)")
    assert "98" in tr["notes"][0]   # 147 of 150 correct
    assert rep.table("Classification Summary (Leave-One-Out)") is not None
    # canonical scores: within-group variance 1 (SPSS scaling)
    sc = fit["scores"]
    w = sum(((sc[y == g] - sc[y == g].mean(0)) ** 2).sum(0) for g in range(3)) / (150 - 3)
    assert w == pytest.approx([1.0, 1.0], rel=1e-8)

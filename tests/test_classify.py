"""K-Means, hierarchical cluster and discriminant analysis checked against scipy / sklearn /
pingouin / statsmodels reference implementations (sklearn is only used here, not at runtime)."""
import math

import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import classify


def num(text: str) -> float:
    return float(text.rstrip("*ᵃᵇᶜᵈ").replace(",", ""))


def table(section, title, nth=0):
    return [i for i in section.items if getattr(i, "title", "").startswith(title)][nth]


@pytest.fixture
def blobs():
    rng = np.random.default_rng(21)
    centres = np.array([[0, 0, 0], [4, 4, 0], [0, 5, 3]], float)
    g = np.repeat([1.0, 2.0, 3.0], [30, 35, 25])
    X = centres[(g - 1).astype(int)] + rng.normal(0, 1.2, (90, 3))
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    df["g"] = g
    df["d"] = rng.normal(size=90)
    perm = rng.permutation(90)
    return Dataset(df.iloc[perm].reset_index(drop=True)), df.iloc[perm].reset_index(drop=True)


# ------------------------------------------------------------------ K-Means
def test_initial_centres_rules():
    X = np.array([[0, 0], [1, 0], [10, 0], [0.5, 0], [20, 0]], float)
    # case 3 (10,0) is farther from its nearest centre than the two centres are apart
    M = classify.initial_centres(X, 2)
    assert sorted(M[:, 0].tolist()) == [0.0, 20.0]


def test_kmeans_matches_lloyd_from_same_start(blobs):
    from sklearn.cluster import KMeans

    ds, df = blobs
    X = df[["a", "b", "c"]].to_numpy()
    res = classify.kmeans(X, 3, max_iter=50)
    ref = KMeans(3, init=res["initial"], n_init=1, algorithm="lloyd", max_iter=50,
                 tol=0).fit(X)
    assert np.allclose(np.sort(res["final"], axis=0), np.sort(ref.cluster_centers_, axis=0))
    out = classify.kmeans_cluster(ds, ["a", "b", "c"], 3, max_iter=50, anova=True,
                                  save=("cluster", "distance"))
    counts = table(out.section, "Number of Cases")
    assert sorted(int(num(r[2])) for r in counts.rows[:3]) == sorted(
        np.bincount(ref.labels_).tolist())
    an = table(out.section, "ANOVA")
    lab = res["labels"]
    from scipy import stats

    f = stats.f_oneway(*[X[lab == j, 0] for j in range(3)]).statistic
    assert num(an.rows[0][5]) == pytest.approx(f, rel=1e-3)
    dist = out.saved["QCL_DIST"][1].to_numpy()
    assert np.allclose(dist, res["distance"])
    hist = table(out.section, "Iteration History")
    assert hist.rows[-1][1:] == [".000"] * 3
    assert "Convergence achieved" in hist.footnotes[0]


def test_kmeans_running_means_and_classify_only(blobs):
    ds, df = blobs
    X = df[["a", "b", "c"]].to_numpy()
    r = classify.kmeans(X, 3, running_means=True, max_iter=30)
    assert len(np.unique(r["labels"])) == 3
    c = classify.kmeans(X, 3, iterate=False)
    assert np.allclose(c["final"], c["initial"]) and not c["history"]


# ------------------------------------------------------------------ hierarchical
@pytest.mark.parametrize("method,scipy_method,measure", [
    ("single", "single", "euclid"), ("complete", "complete", "euclid"),
    ("baverage", "average", "seuclid"), ("centroid", "centroid", "seuclid"),
    ("median", "median", "seuclid"), ("ward", "ward", "seuclid")])
def test_agglomeration_matches_scipy(method, scipy_method, measure):
    rng = np.random.default_rng(4)
    X = rng.normal(size=(30, 3))
    merges, Z = classify.agglomerate(classify.proximities(X, measure), method)
    if method in ("centroid", "median", "ward"):
        ref = linkage(X, scipy_method)
        heights = ref[:, 2] ** 2
        if method == "ward":
            heights = np.cumsum(heights / 2)  # SPSS: cumulative within-cluster SS
    else:
        D = pdist(X, "sqeuclidean" if measure == "seuclid" else "euclidean")
        ref = linkage(D, scipy_method)
        heights = ref[:, 2]
    assert np.allclose([m[2] for m in merges], heights)
    for k in (2, 3, 5):
        mine = classify.memberships(merges, 30, k)
        theirs = fcluster(ref, k, criterion="maxclust")
        # same partition (labels may differ)
        assert len(set(zip(mine, theirs))) == k


def test_within_groups_linkage_hand_example():
    X = np.array([[0.0], [1.0], [5.0], [6.0]])
    P = classify.proximities(X, "seuclid")
    merges, _ = classify.agglomerate(P, "waverage")
    assert merges[0][2] == pytest.approx(1.0)
    assert merges[1][2] == pytest.approx(1.0)
    # all six pairwise squared distances averaged: (1+25+36+16+25+1)/6
    assert merges[2][2] == pytest.approx(104 / 6)


def test_schedule_stage_bookkeeping():
    X = np.array([[0.0], [0.1], [5.0], [5.3], [20.0]])
    ds = Dataset(pd.DataFrame({"x": X[:, 0]}))
    sec = classify.hierarchical(ds, ["x"], "single", "euclid", solution=(2, 3),
                                save=(2, 2)).section
    sch = table(sec, "Agglomeration Schedule").rows
    assert sch[0][1:3] == ["1", "2"] and sch[1][1:3] == ["3", "4"]
    assert sch[2][1:3] == ["1", "3"] and sch[2][4:6] == ["1", "2"]
    assert sch[0][6] == "3" and sch[3][6] == "0"
    mem = table(sec, "Cluster Membership")
    assert [r[2] for r in mem.rows] == ["1", "1", "1", "1", "2"]
    assert any(getattr(i, "kind", "") == "image" for i in sec.items)


def test_correlation_similarity_merges_largest_first():
    X = np.array([[1, 2, 3], [2, 4, 6.1], [3, 2, 1], [1, 5, 2.0]])
    P = classify.proximities(X, "correlation")
    assert np.allclose(P, np.corrcoef(X))
    merges, _ = classify.agglomerate(P, "baverage", similarity=True)
    assert merges[0][:2] == (0, 1) and merges[0][2] == pytest.approx(P[0, 1])


def test_standardize_z():
    X = np.array([[1, 10], [2, 20], [3, 60.0]])
    Z = classify.standardize(X, "z")
    assert np.allclose(Z.mean(axis=0), 0) and np.allclose(Z.std(axis=0, ddof=1), 1)


# ------------------------------------------------------------------ discriminant
def test_discriminant_matches_lda(blobs):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import LeaveOneOut, cross_val_predict

    ds, df = blobs
    X, y = df[["a", "b", "c"]].to_numpy(), df["g"].to_numpy()
    sec = classify.discriminant(ds, "g", 1, 3, ["a", "b", "c"], fisher=True,
                                unstandardized=True).section
    ev = table(sec, "Eigenvalues")
    # eigenvalues of W^-1 B
    means = np.array([X[y == k].mean(axis=0) for k in (1, 2, 3)])
    W = sum((X[y == k] - means[i]).T @ (X[y == k] - means[i]) for i, k in enumerate((1, 2, 3)))
    T = (X - X.mean(0)).T @ (X - X.mean(0))
    lam = np.sort(np.linalg.eigvals(np.linalg.solve(W, T - W)).real)[::-1][:2]
    assert num(ev.rows[0][1]) == pytest.approx(lam[0], abs=6e-4)
    assert num(ev.rows[1][1]) == pytest.approx(lam[1], abs=6e-4)
    lda = LinearDiscriminantAnalysis(priors=[1 / 3] * 3, solver="eigen").fit(X, y)
    assert num(ev.rows[0][2]) == pytest.approx(100 * lam[0] / lam.sum(), abs=0.06)
    wl = table(sec, "Wilks' Lambda")
    assert num(wl.rows[0][1]) == pytest.approx(np.prod(1 / (1 + lam)), abs=6e-4)
    cr = table(sec, "Classification Results")
    pred = lda.predict(X)
    correct = 100 * np.mean(pred == y)
    assert cr.footnotes[0].startswith(f"a. {correct:.1f}%")
    loo = cross_val_predict(LinearDiscriminantAnalysis(priors=[1 / 3] * 3), X, y,
                            cv=LeaveOneOut())
    assert f"{100 * np.mean(loo == y):.1f}%" in cr.footnotes[2]
    # centroids of the canonical scores have pooled within-group variance 1
    fc = table(sec, "Functions at Group Centroids")
    assert len(fc.rows) == 3
    uc = table(sec, "Canonical Discriminant Function Coefficients")
    v = np.array([[num(c) for c in r[1:]] for r in uc.rows[:3]])
    S = W / (len(y) - 3)
    assert np.allclose(np.diag(v.T @ S @ v), 1, atol=2e-3)


def test_discriminant_univariate_and_box_m(blobs):
    import pingouin as pg
    from scipy import stats

    ds, df = blobs
    sec = classify.discriminant(ds, "g", 1, 3, ["a", "b", "c"]).section
    te = table(sec, "Tests of Equality of Group Means")
    f = stats.f_oneway(*[df.loc[df.g == k, "a"] for k in (1, 2, 3)]).statistic
    assert num(te.rows[0][2]) == pytest.approx(f, abs=2e-3)
    box = table(sec, "Test Results")
    ref = pg.box_m(df, dvs=["a", "b", "c"], group="g")
    # pingouin reports the chi-square approximation; Box's M statistic itself must agree
    nis = df.g.value_counts().to_numpy()
    p, g, n = 3, 3, len(df)
    c = (np.sum(1 / (nis - 1)) - 1 / (n - g)) * (2 * p * p + 3 * p - 1) / (6 * (p + 1) * (g - 1))
    assert num(box.rows[0][2]) * (1 - c) == pytest.approx(ref["Chi2"].iloc[0], rel=1e-3)


def test_discriminant_stepwise_picks_informative(blobs):
    ds, df = blobs
    res = classify.discriminant(ds, "g", 1, 3, ["d", "a", "b", "c"], method="stepwise",
                                save=("group", "scores", "probs"))
    er = table(res.section, "Variables Entered/Removed")
    entered = [r[1] for r in er.rows if r[1]]
    assert "d" not in entered and len(entered) >= 2
    from statsmodels.multivariate.manova import MANOVA

    cols = entered
    m = MANOVA.from_formula(" + ".join(cols) + " ~ C(g)", data=df).mv_test()
    wilks = m.results["C(g)"]["stat"].loc["Wilks' lambda", "Value"]
    assert num(er.rows[-1][3]) == pytest.approx(wilks, abs=6e-4)
    assert "Dis" in res.saved and "Dis1" in res.saved and "Dis1_P" in res.saved
    probs = sum(res.saved[f"Dis{j}_P"][1] for j in (1, 2, 3))
    assert np.allclose(probs, 1)


def test_discriminant_out_of_range_excluded(blobs):
    ds, df = blobs
    sec = classify.discriminant(ds, "g", 1, 2, ["a", "b"]).section
    cps = table(sec, "Analysis Case Processing Summary")
    assert cps.rows[0][2] == str(int((df.g <= 2).sum()))
    assert cps.rows[1][2] == str(int((df.g == 3).sum()))
    ev = table(sec, "Eigenvalues")
    assert len(ev.rows) == 1 and not math.isnan(num(ev.rows[0][1]))

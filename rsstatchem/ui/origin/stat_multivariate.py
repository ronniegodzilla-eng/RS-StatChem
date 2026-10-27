"""Origin's Statistics > Multivariate Analysis with report sheets (no Qt).

* Principal Component Analysis (correlation or covariance matrix): eigenvalues, extracted
  eigenvectors, correlation loadings, KMO and Bartlett's test, scores; scree plot, loading
  plot, score plot and biplot (Gabriel, α = 0) as live graphs.
* K-Means Cluster Analysis (:func:`rsstatchem.core.stats.classify.kmeans`: SPSS QUICK CLUSTER
  initial centres and iterations), with ANOVA per variable and the membership table.
* Hierarchical Cluster Analysis (:func:`classify.agglomerate`, Lance-Williams) of
  observations or variables with the agglomeration schedule, memberships and a dendrogram
  drawn as line segments.
* Discriminant Analysis (linear, pooled covariance): tests of group means, canonical
  discriminant functions (eigenvalues, canonical correlations, Wilks' lambda), standardized
  / unstandardized coefficients, centroids, Fisher's classification functions and the
  classification table (training and leave-one-out).
* Partial Least Squares regression (:mod:`rsstatchem.core.analysis.chemometrics`: NIPALS or
  SIMPLS, cross-validation, VIP).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

from rsstatchem.core.analysis import chemometrics as cm
from rsstatchem.core.analysis.common import num, pval
from rsstatchem.core.graph.themes import PALETTES
from rsstatchem.core.stats import classify as cl
from rsstatchem.core.stats.factor import extract_pc, kmo
from rsstatchem.ui.origin import report_graphs as rg
from rsstatchem.ui.origin import stat_tests as st
from rsstatchem.ui.origin.model import ReportSheet
from rsstatchem.ui.origin.stat_extra import aligned, new_report, report_text

StatError = st.StatError
PAL = PALETTES.get("Origin Classic")
MAX_TABLE_ROWS = 5000


def _matrix(books: dict, settings: dict, extra_keys=("labels", "group")):
    """Listwise-complete data of the variable ranges plus optional label / group columns:
    (inputs, X, row labels, groups or None, extra inputs)."""
    inputs = st.read_ranges(books, settings.get("ranges", []))
    m, ok = aligned(inputs)
    extra = {}
    for key in extra_keys:
        text = settings.get(key, "")
        if text and text.strip():
            extra[key] = st.read_range(books, text)
    n = len(m)
    if "group" in extra:
        g = extra["group"]
        ok = ok[:min(n, len(g.texts))] & np.array([g.texts[i] != "" for i in
                                                   range(min(n, len(g.texts)))])
        n = len(ok)
        m = m[:n]
    rows = np.flatnonzero(ok)
    X = m[rows]
    start = inputs[0].rows[0] if inputs[0].rows else 1
    if "labels" in extra:
        lab = extra["labels"]
        labels = [lab.texts[i] if i < len(lab.texts) and lab.texts[i] != "" else str(i + start)
                  for i in rows]
    else:
        labels = [str(i + start) for i in rows]
    groups = [extra["group"].texts[i] for i in rows] if "group" in extra else None
    return inputs, X, labels, groups, list(extra.values())


def _labels_column(g, labels, name="Labels"):
    return rg._key(g, name, np.array(labels, dtype=object), kind="text")


def _scatter_by_group(g, x, y, groups, labels=None, size=5.0):
    if groups is None:
        p = rg.add_series(g, x, y, "Observations", "scatter", "#1f5fbf", size=size,
                          legend=False)
        if labels is not None and len(labels) <= 60:
            p.labels = _labels_column(g, labels)
        return
    levels = st._sort_levels(groups)
    garr = np.array(groups, dtype=object)
    for i, lv in enumerate(levels):
        sel = garr == lv
        rg.add_series(g, x[sel], y[sel], str(lv), "scatter", PAL[i % len(PAL)],
                      symbol=("s", "o", "^", "v", "D")[i % 5], size=size)


# ====================================================================== PCA
def pca_report(books: dict, settings: dict, report: ReportSheet | None = None,
               mode: str = "manual"):
    inputs, X, labels, groups, extra = _matrix(books, settings)
    if len(inputs) < 2:
        raise StatError("PCA needs two or more variables.")
    matrix = settings.get("matrix", "correlation")
    try:
        res = cm.pca(X, matrix)
    except cm.ChemometricsError as exc:
        raise StatError(str(exc)) from None
    p = X.shape[1]
    ncomp = int(settings.get("ncomp", 0))
    if ncomp <= 0:   # Kaiser criterion (eigenvalue > mean eigenvalue)
        ncomp = max(2 if p >= 2 else 1, int(np.sum(res.eigenvalues > res.eigenvalues.mean())))
    ncomp = min(ncomp, p)
    names = [r.title for r in inputs]
    first = inputs[0]
    report, fresh = new_report(first, f"Principal Component Analysis of {first.text} …",
                               "PCA", report, "pca", settings, mode)
    st._notes(fresh, "Principal Component Analysis", inputs + extra,
              [["Analyze", "Correlation Matrix" if matrix == "correlation" else
                "Covariance Matrix"], ["Number of Components", str(ncomp)],
               ["N (listwise)", str(len(X))]])
    fresh.add_table("Descriptive Statistics", ["", "N", "Mean", "SD"],
                    [[nm, str(len(X)), num(X[:, j].mean()), num(X[:, j].std(ddof=1))]
                     for j, nm in enumerate(names)])
    R = np.corrcoef(X, rowvar=False) if matrix == "correlation" else np.cov(X, rowvar=False)
    fresh.add_table("Correlation Matrix" if matrix == "correlation" else "Covariance Matrix",
                    [""] + names, [[names[i]] + [num(R[i, j], 5) for j in range(p)]
                                   for i in range(p)])
    chi, df, pb = cm.bartlett_sphericity(X)
    rows = [["Bartlett's Test of Sphericity", num(chi, 5), str(df), pval(pb)]]
    try:
        k_all, _, _ = kmo(np.corrcoef(X, rowvar=False))
        rows.insert(0, ["Kaiser-Meyer-Olkin Measure of Sampling Adequacy", num(k_all, 5), "",
                        ""])
    except np.linalg.LinAlgError:
        pass
    fresh.add_table("Sampling Adequacy", ["", "Value / Chi-Square", "DF", "Prob"], rows)
    fresh.add_table("Eigenvalues of the " + ("Correlation" if matrix == "correlation" else
                                             "Covariance") + " Matrix",
                    ["", "Eigenvalue", "Percentage of Variance", "Cumulative"],
                    [[str(i + 1), num(res.eigenvalues[i]), num(res.percent[i]) + "%",
                      num(res.cumulative[i]) + "%"] for i in range(p)])
    comp = [f"PC {i + 1}" for i in range(ncomp)]
    fresh.add_table("Extracted Eigenvectors", [""] + comp,
                    [[names[j]] + [num(res.vectors[j, i], 5) for i in range(ncomp)]
                     for j in range(p)],
                    ["Coefficients of the principal components (unit length)."])
    L = res.loadings
    if matrix == "correlation":   # same as the factor-analysis PC extraction
        L[:, :ncomp] = extract_pc(np.corrcoef(X, rowvar=False), ncomp)
    fresh.add_table("Loadings", [""] + comp,
                    [[names[j]] + [num(L[j, i], 5) for i in range(ncomp)] for j in range(p)],
                    ["Correlations between the variables and the components."
                     if matrix == "correlation" else
                     "Covariances between the variables and the unit-variance scores."])
    sc = res.scores[:, :ncomp]
    fresh.add_table("Scores", ["Observation"] + comp,
                    [[labels[i]] + [num(sc[i, k], 5) for k in range(ncomp)]
                     for i in range(min(len(sc), MAX_TABLE_ROWS))], group="Scores")
    if settings.get("scree", True):
        g = rg.new_graph("Scree Plot", 11.0, 7.5)
        x = np.arange(1, p + 1, dtype=float)
        rg.add_series(g, x, res.eigenvalues, "Eigenvalue", "line_symbol", "#1f5fbf", size=6.0)
        fresh.add_graph("Scree Plot", rg.finish(g, "Component Number", "Eigenvalue"),
                        width=380, group="Plots")
    if ncomp >= 2:
        if settings.get("loading_plot", True):
            g = rg.new_graph("Loading Plot", 10.0, 9.0)
            rg.add_series(g, [-1, 1], [0, 0], "x0", "line", "#c0c0c0", width=0.6,
                          legend=False)
            rg.add_series(g, [0, 0], [-1, 1], "y0", "line", "#c0c0c0", width=0.6,
                          legend=False)
            pl = rg.add_series(g, L[:, 0], L[:, 1], "Loadings", "scatter", rg.FIT_RED,
                               size=6.0, legend=False)
            pl.labels = _labels_column(g, names, "Variables")
            fresh.add_graph("Loading Plot", rg.finish(g, f"PC 1 ({num(res.percent[0], 4)}%)",
                                                      f"PC 2 ({num(res.percent[1], 4)}%)"),
                            width=380, group="Plots")
        if settings.get("score_plot", True):
            g = rg.new_graph("Score Plot", 10.0, 9.0)
            _scatter_by_group(g, sc[:, 0], sc[:, 1], groups, labels)
            fresh.add_graph("Score Plot", rg.finish(g, "PC 1 Score", "PC 2 Score",
                                                    legend=groups is not None),
                            width=380, group="Plots")
        if settings.get("biplot", True):
            s2, l2 = res.biplot(0, 1)
            g = rg.new_graph("Biplot", 10.0, 9.0)
            _scatter_by_group(g, s2[:, 0], s2[:, 1], groups, None, size=4.0)
            xs, ys = [], []
            for j in range(p):
                xs += [0.0, l2[j, 0], np.nan]
                ys += [0.0, l2[j, 1], np.nan]
            rg.add_series(g, xs, ys, "Loading vectors", "line", rg.FIT_RED, width=1.2,
                          legend=False)
            tips = rg.add_series(g, l2[:, 0], l2[:, 1], "Variables", "scatter", rg.FIT_RED,
                                 size=3.0, legend=False)
            tips.labels = _labels_column(g, names, "Variable names")
            fresh.add_graph("Biplot", rg.finish(g, "PC 1", "PC 2", legend=groups is not None),
                            width=380, group="Plots")
    report = st._finish(report, fresh)
    return report, report_text(first.book, "Principal Component Analysis", report,
                               [f"  first {ncomp} components explain "
                                f"{num(res.cumulative[ncomp - 1], 4)}% of the variance"])


# ====================================================================== k-means
STANDARDIZE = {"none": "None", "z": "Z scores", "range01": "Range 0 to 1"}


def kmeans_report(books: dict, settings: dict, report: ReportSheet | None = None,
                  mode: str = "manual"):
    inputs, X0, labels, _, extra = _matrix(books, settings, ("labels",))
    k = int(settings.get("k", 3))
    if not 2 <= k < len(X0):
        raise StatError("The number of clusters must be at least 2 and less than N.")
    how = settings.get("standardize", "none")
    X = cl.standardize(X0, how)
    res = cl.kmeans(X, k, int(settings.get("max_iter", 10)), 0.0)
    lab = res["labels"]
    names = [r.title for r in inputs]
    first = inputs[0]
    report, fresh = new_report(first, f"K-Means Cluster Analysis of {first.text} …",
                               "KMeans", report, "kmeans", settings, mode)
    st._notes(fresh, "K-Means Cluster Analysis", inputs + extra,
              [["Number of Clusters", str(k)], ["Standardize", STANDARDIZE.get(how, how)],
               ["Iterations", f"{res['iterations']} ({'converged' if res['converged'] else 'maximum reached'})"],
               ["N (listwise)", str(len(X))]])
    cols = [f"Cluster {j + 1}" for j in range(k)]
    fresh.add_table("Initial Cluster Centers", [""] + cols,
                    [[nm] + [num(res["initial"][j, v]) for j in range(k)]
                     for v, nm in enumerate(names)])
    fresh.add_table("Final Cluster Centers", [""] + cols,
                    [[nm] + [num(res["final"][j, v]) for j in range(k)]
                     for v, nm in enumerate(names)])
    summ = []
    for j in range(k):
        dj = res["distance"][lab == j]
        summ.append([cols[j], str(int((lab == j).sum())), num(float(np.sum(dj ** 2))),
                     num(dj.mean() if len(dj) else math.nan),
                     num(dj.max() if len(dj) else math.nan)])
    fresh.add_table("Cluster Summary", ["", "Number of Observations", "Within Cluster Sum of "
                                        "Squares", "Average Distance from Centroid",
                                        "Maximum Distance from Centroid"], summ)
    d = np.sqrt(((res["final"][:, None, :] - res["final"][None, :, :]) ** 2).sum(2))
    fresh.add_table("Distances between Final Cluster Centers", [""] + cols,
                    [[cols[i]] + [num(d[i, j]) for j in range(k)] for i in range(k)])
    an = []
    for v, nm in enumerate(names):
        groups = [X[lab == j, v] for j in range(k) if (lab == j).sum() > 0]
        if len(groups) > 1:
            f, p = stats.f_oneway(*groups)
            an.append([nm, num(f, 5), pval(p)])
    fresh.add_table("ANOVA", ["", "F Value", "Prob>F"], an,
                    ["The F tests are descriptive only: the clusters were chosen to maximize "
                     "the differences."])
    fresh.add_table("Cluster Membership", ["Observation", "Cluster", "Distance"],
                    [[labels[i], str(lab[i] + 1), num(res["distance"][i])]
                     for i in range(min(len(X), MAX_TABLE_ROWS))], group="Membership")
    # plot on the first two variables (or first two principal components)
    if X.shape[1] >= 2:
        if X.shape[1] == 2:
            P, C, xt, yt = X, res["final"], names[0], names[1]
        else:
            pr = cm.pca(X, "covariance")
            P = pr.scores[:, :2]
            C = ((res["final"] - pr.mean) / pr.scale) @ pr.vectors[:, :2]
            xt, yt = "PC 1", "PC 2"
        g = rg.new_graph("Cluster Plot", 10.0, 9.0)
        _scatter_by_group(g, P[:, 0], P[:, 1], [f"Cluster {v + 1}" for v in lab])
        rg.add_series(g, C[:, 0], C[:, 1], "Centers", "scatter", rg.DATA_BLACK, symbol="*",
                      size=11.0)
        fresh.add_graph("Cluster Plot", rg.finish(g, xt, yt, legend=True), width=400)
    report = st._finish(report, fresh)
    return report, report_text(first.book, "K-Means Cluster Analysis", report,
                               ["  sizes: " + ", ".join(r[1] for r in summ)])


# ====================================================================== hierarchical
HCA_METHODS = {"baverage": "Group Average (between groups)", "single": "Nearest Neighbor",
               "complete": "Furthest Neighbor", "centroid": "Centroid", "median": "Median",
               "ward": "Ward"}
HCA_MEASURES = {"seuclid": "Squared Euclidean", "euclid": "Euclidean",
                "block": "City Block (Manhattan)", "chebychev": "Chebychev",
                "correlation": "Correlation (1 − r)", "cosine": "Cosine (1 − cos)"}


def dendrogram_segments(Z: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """(x, y) of the U-shaped links (NaN-separated) and the leaf order, from a SciPy-style
    linkage matrix; leaves at x = 1…n."""
    from scipy.cluster.hierarchy import dendrogram
    dd = dendrogram(Z, no_plot=True)
    xs, ys = [], []
    for ic, dc in zip(dd["icoord"], dd["dcoord"]):
        xs += [v / 10 + 0.5 for v in ic] + [np.nan]
        ys += list(dc) + [np.nan]
    return np.array(xs), np.array(ys), list(dd["leaves"])


def hca_report(books: dict, settings: dict, report: ReportSheet | None = None,
               mode: str = "manual"):
    inputs, X0, labels, _, extra = _matrix(books, settings, ("labels",))
    what = settings.get("cluster", "observations")
    names = [r.title for r in inputs]
    how = settings.get("standardize", "none")
    X = cl.standardize(X0, how)
    if what == "variables":
        X = X.T
        labels = names
    n = len(X)
    if n < 3:
        raise StatError("Hierarchical clustering needs at least 3 items.")
    method = settings.get("method", "baverage")
    measure = settings.get("measure", "seuclid")
    P = cl.proximities(X, measure)
    if measure in ("cosine", "correlation"):
        P = 1 - P
    merges, Z = cl.agglomerate(P, method)
    k = int(settings.get("k", 3))
    k = max(1, min(k, n))
    member = cl.memberships(merges, n, k)
    first = inputs[0]
    report, fresh = new_report(first, f"Hierarchical Cluster Analysis of {first.text} …",
                               "HCA", report, "hca", settings, mode)
    st._notes(fresh, "Hierarchical Cluster Analysis", inputs + extra,
              [["Cluster", "Variables" if what == "variables" else "Observations"],
               ["Method", HCA_METHODS.get(method, method)],
               ["Distance", HCA_MEASURES.get(measure, measure)],
               ["Standardize", STANDARDIZE.get(how, how)], ["N", str(n)]])
    first_stage: dict[int, int] = {}
    sched = []
    for s, (i, j, coef) in enumerate(merges):
        nxt = next((t + 1 for t in range(s + 1, len(merges)) if i in merges[t][:2]), 0)
        sched.append([str(s + 1), labels[i], labels[j], num(coef),
                      str(first_stage.get(i, 0)), str(first_stage.get(j, 0)), str(nxt)])
        first_stage[i] = s + 1
    fresh.add_table("Agglomeration Schedule", ["Stage", "Cluster 1", "Cluster 2",
                                               "Coefficient", "Stage Cluster 1 First Appears",
                                               "Stage Cluster 2 First Appears", "Next Stage"],
                    sched, ["A cluster is named after its first member."])
    fresh.add_table(f"Cluster Membership ({k} clusters)", ["Item", "Cluster"],
                    [[labels[i], str(member[i])] for i in range(min(n, MAX_TABLE_ROWS))])
    if what == "observations" and k > 1:
        cen = [[nm] + [num(X0[member == c, v].mean()) for c in range(1, k + 1)]
               for v, nm in enumerate(names)]
        fresh.add_table("Cluster Centers", [""] + [f"Cluster {c}" for c in range(1, k + 1)],
                        cen)
    if settings.get("dendrogram", True) and n <= 500:
        xs, ys, leaves = dendrogram_segments(Z, n)
        g = rg.new_graph("Dendrogram", 13.0, 8.5)
        x_labels = [labels[i] for i in leaves]
        p = rg.add_series(g, xs, ys, "Dendrogram", "line", "#1f3f8f", width=1.2,
                          legend=False)
        if n <= 80:   # leaf names at the bottom of the tree
            leaf = rg.add_series(g, np.arange(1, n + 1, dtype=float), np.zeros(n), "Leaves",
                                 "scatter", "#1f3f8f", size=2.0, legend=False)
            leaf.labels = _labels_column(g, x_labels, "Leaf labels")
            leaf.options["label_size"] = 7.0
        lay = g.layers[0]
        if k > 1 and len(Z) >= k - 1:
            cut = (Z[n - k, 2] + Z[n - k - 1, 2]) / 2 if n - k - 1 >= 0 else Z[n - k, 2]
            rg.add_series(g, [0.5, n + 0.5], [cut, cut], f"Cut at {k} clusters", "line",
                          rg.FIT_RED, dash="--", width=1.0, legend=False)
        lay.x.show_labels = False
        fresh.add_graph("Dendrogram", rg.finish(g, "", "Distance"), width=520)
    report = st._finish(report, fresh)
    return report, report_text(first.book, "Hierarchical Cluster Analysis", report)


# ====================================================================== discriminant
def lda_fit(X: np.ndarray, gidx: np.ndarray, g: int, priors: np.ndarray) -> dict:
    n, p = X.shape
    means, grand, W, T = cl._lda(X, gidx, g)
    B = T - W
    S = W / (n - g)
    Si = np.linalg.pinv(S)
    coef = means @ Si                                   # g × p
    const = -0.5 * np.sum(coef * means, axis=1) + np.log(priors)
    vals, vecs = cl._eig(W, B)
    m = min(g - 1, p)
    vals, vecs = vals[:m], vecs[:, :m] * math.sqrt(n - g)   # v' S v = 1
    for j in range(m):   # sign: positive centroid of the first group at the largest |coef|
        if vecs[np.argmax(np.abs(vecs[:, j])), j] < 0:
            vecs[:, j] *= -1
    scores = (X - grand) @ vecs
    disc = X @ coef.T + const
    return {"means": means, "grand": grand, "W": W, "T": T, "S": S, "coef": coef,
            "const": const, "eig": vals, "vecs": vecs, "scores": scores,
            "pred": np.argmax(disc, axis=1), "m": m}


def lda_report(books: dict, settings: dict, report: ReportSheet | None = None,
               mode: str = "manual"):
    inputs, X, labels, groups, extra = _matrix(books, settings, ("labels", "group"))
    if groups is None:
        raise StatError("Select the group range (class of each observation).")
    levels = st._sort_levels(groups)
    g = len(levels)
    if g < 2:
        raise StatError("The group column needs at least two groups.")
    gidx = np.array([levels.index(v) for v in groups])
    n, p = X.shape
    if n - g < p:
        raise StatError("Too few observations for the number of variables.")
    counts = np.bincount(gidx, minlength=g).astype(float)
    priors = np.full(g, 1.0 / g) if settings.get("priors", "equal") == "equal" else \
        counts / n
    fit = lda_fit(X, gidx, g, priors)
    names = [r.title for r in inputs]
    first = inputs[0]
    report, fresh = new_report(first, f"Discriminant Analysis of {first.text} …",
                               "Discriminant", report, "lda", settings, mode)
    st._notes(fresh, "Discriminant Analysis (linear)", inputs + extra,
              [["Groups", ", ".join(levels)],
               ["Prior Probabilities", "Equal" if settings.get("priors", "equal") == "equal"
                else "Proportional to group sizes"], ["N (listwise)", str(n)]])
    gs = []
    for j, lv in enumerate(levels):
        for v, nm in enumerate(names):
            x = X[gidx == j, v]
            gs.append([lv if v == 0 else "", nm, str(len(x)), num(x.mean()),
                       num(x.std(ddof=1) if len(x) > 1 else math.nan)])
    fresh.add_table("Group Statistics", ["Group", "Variable", "N", "Mean", "SD"], gs)
    eq = []
    for v, nm in enumerate(names):
        lam = fit["W"][v, v] / fit["T"][v, v] if fit["T"][v, v] > 0 else math.nan
        f = (1 - lam) / lam * (n - g) / (g - 1) if lam > 0 else math.nan
        eq.append([nm, num(lam, 5), num(f, 5), str(g - 1), str(n - g),
                   pval(float(stats.f.sf(f, g - 1, n - g)))])
    fresh.add_table("Tests of Equality of Group Means", ["", "Wilks' Lambda", "F", "DF1",
                                                         "DF2", "Prob>F"], eq)
    ev = fit["eig"]
    m = fit["m"]
    tot = ev.sum()
    fresh.add_table("Canonical Discriminant Functions: Eigenvalues",
                    ["Function", "Eigenvalue", "% of Variance", "Cumulative %",
                     "Canonical Correlation"],
                    [[str(j + 1), num(ev[j], 5), num(ev[j] / tot * 100, 5),
                      num(ev[:j + 1].sum() / tot * 100, 5), num(math.sqrt(ev[j] / (1 + ev[j])), 5)]
                     for j in range(m)])
    wl = []
    for j in range(m):
        lam = float(np.prod(1 / (1 + ev[j:])))
        chi = -(n - 1 - (p + g) / 2) * math.log(lam)
        df = (p - j) * (g - j - 1)
        wl.append([f"{j + 1} through {m}" if j < m - 1 else str(j + 1), num(lam, 5),
                   num(chi, 5), str(df), pval(float(stats.chi2.sf(chi, df)))])
    fresh.add_table("Wilks' Lambda", ["Test of Function(s)", "Wilks' Lambda", "Chi-Square",
                                      "DF", "Prob"], wl)
    sd_w = np.sqrt(np.diag(fit["S"]))
    fn = [f"Function {j + 1}" for j in range(m)]
    fresh.add_table("Standardized Canonical Discriminant Function Coefficients", [""] + fn,
                    [[nm] + [num(fit["vecs"][v, j] * sd_w[v], 5) for j in range(m)]
                     for v, nm in enumerate(names)])
    const = -fit["grand"] @ fit["vecs"]
    fresh.add_table("Canonical Discriminant Function Coefficients (Unstandardized)",
                    [""] + fn, [[nm] + [num(fit["vecs"][v, j], 5) for j in range(m)]
                                for v, nm in enumerate(names)] +
                    [["(Constant)"] + [num(const[j], 5) for j in range(m)]])
    cent = (fit["means"] - fit["grand"]) @ fit["vecs"]
    fresh.add_table("Functions at Group Centroids", ["Group"] + fn,
                    [[lv] + [num(cent[i, j], 5) for j in range(m)] for i, lv in
                     enumerate(levels)])
    fresh.add_table("Classification Function Coefficients", [""] + levels,
                    [[nm] + [num(fit["coef"][j, v], 5) for j in range(g)]
                     for v, nm in enumerate(names)] +
                    [["(Constant)"] + [num(fit["const"][j], 5) for j in range(g)]],
                    ["Fisher's linear discriminant functions (prior probabilities in the "
                     "constants)."])

    def table(pred, title):
        rows = []
        for i, lv in enumerate(levels):
            c = np.bincount(pred[gidx == i], minlength=g)
            rows.append([lv] + [str(int(v)) for v in c] + [str(int(c.sum())),
                                                           num(c[i] / c.sum() * 100, 4) + "%"])
        acc = float(np.mean(pred == gidx) * 100)
        fresh.add_table(title, ["Actual \\ Predicted"] + levels + ["Total", "% Correct"], rows,
                        [f"{num(acc, 4)}% of the observations are correctly classified."],
                        group="Classification")
        return acc
    acc = table(fit["pred"], "Classification Summary (Training)")
    if settings.get("loo", True):
        loo = cl._loo_predict(X, gidx, g, priors)
        table(np.asarray(loo, int), "Classification Summary (Leave-One-Out)")
    if m >= 1:
        gr = rg.new_graph("Canonical Score Plot", 10.0, 9.0)
        sc = fit["scores"]
        y = sc[:, 1] if m >= 2 else np.zeros(len(sc))
        _scatter_by_group(gr, sc[:, 0], y, groups)
        rg.add_series(gr, cent[:, 0], cent[:, 1] if m >= 2 else np.zeros(g), "Centroids",
                      "scatter", rg.DATA_BLACK, symbol="*", size=11.0)
        fresh.add_graph("Canonical Score Plot", rg.finish(gr, "Function 1",
                                                          "Function 2" if m >= 2 else "",
                                                          legend=True), width=400)
    report = st._finish(report, fresh)
    return report, report_text(first.book, "Discriminant Analysis", report,
                               [f"  {num(acc, 4)}% correctly classified (training)"])


# ====================================================================== PLS
CV_SCHEMES = {"none": "None", "loo": "Leave One Out", "kfold": "K-fold (contiguous blocks)",
              "venetian": "Venetian blinds"}


def pls_report(books: dict, settings: dict, report: ReportSheet | None = None,
               mode: str = "manual"):
    xin = st.read_ranges(books, settings.get("ranges", []))
    yin = [st.read_range(books, t) for t in settings.get("response", []) if t.strip()]
    if not yin:
        raise StatError("Select the response (Y) columns.")
    lab = settings.get("labels", "")
    m, ok = aligned(xin + yin)
    rows = np.flatnonzero(ok)
    X, Y = m[rows, :len(xin)], m[rows, len(xin):]
    n, p = X.shape
    if n < 4:
        raise StatError("PLS needs at least 4 complete observations.")
    method = settings.get("method", "nipals")
    scale = bool(settings.get("scale", True))
    scheme = settings.get("cv", "loo")
    folds = int(settings.get("folds", 7))
    amax = min(int(settings.get("max_comp", 10)), p, n - 2)
    ncomp = int(settings.get("ncomp", 0))
    cv = None
    if scheme != "none":
        try:
            cv = cm.pls_cv(X, Y, max(amax, 1), method, scale, scheme, folds)
        except cm.ChemometricsError as exc:
            raise StatError(str(exc)) from None
    if ncomp <= 0:
        if cv is None:
            raise StatError("Choose the number of components or a cross-validation scheme.")
        ncomp = int(np.nanargmin(cv["press"]) + 1)
    ncomp = min(ncomp, p, n - 1)
    try:
        model = cm.pls_fit(X, Y, ncomp, method, scale)
    except cm.ChemometricsError as exc:
        raise StatError(str(exc)) from None
    ex = cm.explained(model, X, Y)
    vip = cm.vip(model, X, Y)
    xn = [r.title for r in xin]
    yn = [r.title for r in yin]
    start = xin[0].rows[0] if xin[0].rows else 1
    labels = [str(i + start) for i in rows]
    extra = []
    if lab and lab.strip():
        lr = st.read_range(books, lab)
        extra = [lr]
        labels = [lr.texts[i] if i < len(lr.texts) and lr.texts[i] != "" else labels[k]
                  for k, i in enumerate(rows)]
    first = xin[0]
    report, fresh = new_report(first, f"Partial Least Squares of {yn[0]}" +
                               (" …" if len(yn) > 1 else ""), "PLS", report, "pls", settings,
                               mode)
    st._notes(fresh, "Partial Least Squares Regression", xin + yin + extra,
              [["Algorithm", "NIPALS" if method == "nipals" else "SIMPLS (de Jong 1993)"],
               ["Scaling", "Center and scale (unit variance)" if scale else "Center only"],
               ["Cross Validation", CV_SCHEMES.get(scheme, scheme) +
                (f" ({cv['folds']} folds)" if cv is not None else "")],
               ["Number of Components", str(ncomp)], ["N", str(n)],
               ["X Variables", str(p)], ["Y Variables", str(len(yn))]])
    cx, cy = np.cumsum(ex["x"]), np.cumsum(ex["y"])
    fresh.add_table("Variance Explained", ["Component", "X Variance %", "Cumulative X %",
                                           "Y Variance %", "Cumulative Y %"],
                    [[str(a + 1), num(ex["x"][a] * 100, 5), num(cx[a] * 100, 5),
                      num(ex["y"][a] * 100, 5), num(cy[a] * 100, 5)] for a in range(ncomp)])
    if cv is not None:
        best = int(np.nanargmin(cv["press"]) + 1)
        fresh.add_table("Cross Validation", ["Components", "PRESS", "RMSECV", "Q²"],
                        [[str(a + 1), num(cv["press"][a]), num(cv["rmsecv"][a]),
                          num(cv["q2"][a], 5)] for a in range(len(cv["press"]))],
                        [f"Minimum PRESS at {best} component{'s' if best > 1 else ''}."])
    pred = model.predict(X)
    resid = Y - pred
    fit_rows = []
    for j, nm in enumerate(yn):
        ss = float(((Y[:, j] - Y[:, j].mean()) ** 2).sum())
        r2 = 1 - float((resid[:, j] ** 2).sum()) / ss if ss > 0 else math.nan
        fit_rows.append([nm, num(r2, 5), num(math.sqrt(float((resid[:, j] ** 2).mean()))),
                         num(cv["q2_y"][ncomp - 1, j], 5) if cv is not None and
                         ncomp <= len(cv["press"]) else "--"])
    fresh.add_table("Fit Statistics", ["Response", "R² (calibration)", "RMSEC", "Q² (CV)"],
                    fit_rows)
    B, b0 = model.coef, model.intercept
    fresh.add_table("Coefficients", [""] + yn,
                    [["Intercept"] + [num(v) for v in b0]] +
                    [[xn[i]] + [num(B[i, j]) for j in range(len(yn))] for i in range(p)],
                    ["Regression coefficients in the original units of X and Y."])
    fresh.add_table("Variable Importance in the Projection (VIP)", ["", "VIP"],
                    [[xn[i], num(vip[i], 5)] for i in range(p)],
                    ["Variables with VIP > 1 are the most influential."], group="Loadings")
    comp = [f"Component {a + 1}" for a in range(ncomp)]
    fresh.add_table("X Weights", [""] + comp,
                    [[xn[i]] + [num(model.x_weights[i, a], 5) for a in range(ncomp)]
                     for i in range(p)], group="Loadings")
    fresh.add_table("X Loadings", [""] + comp,
                    [[xn[i]] + [num(model.x_loadings[i, a], 5) for a in range(ncomp)]
                     for i in range(p)], group="Loadings")
    fresh.add_table("Y Loadings", [""] + comp,
                    [[yn[j]] + [num(model.y_loadings[j, a], 5) for a in range(ncomp)]
                     for j in range(len(yn))], group="Loadings")
    fresh.add_table("X Scores", ["Observation"] + comp,
                    [[labels[i]] + [num(model.x_scores[i, a], 5) for a in range(ncomp)]
                     for i in range(min(n, MAX_TABLE_ROWS))], group="Scores")
    fresh.add_table("Predicted Values and Residuals",
                    ["Observation"] + [f"{nm} (predicted)" for nm in yn] +
                    [f"{nm} (residual)" for nm in yn],
                    [[labels[i]] + [num(pred[i, j]) for j in range(len(yn))] +
                     [num(resid[i, j]) for j in range(len(yn))]
                     for i in range(min(n, MAX_TABLE_ROWS))], group="Scores")
    # graphs
    g = rg.new_graph("Variance Explained", 11.0, 7.5)
    xs = np.arange(1, ncomp + 1, dtype=float)
    rg.add_series(g, xs, cx * 100, "Cumulative X %", "line_symbol", "#1f5fbf", size=6.0)
    rg.add_series(g, xs, cy * 100, "Cumulative Y %", "line_symbol", rg.FIT_RED, symbol="o",
                  size=6.0)
    fresh.add_graph("Variance Explained", rg.finish(g, "Number of Components",
                                                    "Variance Explained (%)", legend=True),
                    width=380, group="Plots")
    if cv is not None:
        g = rg.new_graph("RMSECV", 11.0, 7.5)
        rg.add_series(g, np.arange(1, len(cv["rmsecv"]) + 1, dtype=float), cv["rmsecv"],
                      "RMSECV", "line_symbol", "#1f5fbf", size=6.0)
        fresh.add_graph("Cross Validation Plot", rg.finish(g, "Number of Components",
                                                           "RMSECV"), width=380, group="Plots")
    for j, nm in enumerate(yn):
        g = rg.new_graph(f"Predicted vs. Actual: {nm}", 10.0, 9.0)
        lo, hi = float(min(Y[:, j].min(), pred[:, j].min())), float(max(Y[:, j].max(),
                                                                        pred[:, j].max()))
        rg.add_series(g, [lo, hi], [lo, hi], "y = x", "line", "#808080", dash="--",
                      width=1.0, legend=False)
        rg.add_series(g, Y[:, j], pred[:, j], nm, "scatter", "#1f5fbf", size=5.0,
                      legend=False)
        fresh.add_graph(f"Predicted vs. Actual: {nm}", rg.finish(g, f"Actual {nm}",
                                                                 f"Predicted {nm}"),
                        width=380, group="Plots")
    g = rg.new_graph("VIP", 12.0, 7.5)
    if p <= 40:
        rg.add_series(g, None, vip, "VIP", "column", "#1f5fbf", x_labels=xn)
    else:
        rg.add_series(g, np.arange(1, p + 1, dtype=float), vip, "VIP", "line", "#1f5fbf")
    rg.add_series(g, [0.5, p + 0.5], [1.0, 1.0], "VIP = 1", "line", rg.FIT_RED, dash="--",
                  width=1.0, legend=False)
    fresh.add_graph("VIP Plot", rg.finish(g, "X Variable", "VIP"), width=420, group="Plots")
    if p > 10:   # spectra-like: loadings against the variable index
        g = rg.new_graph("X Loadings", 12.0, 7.5)
        for a in range(min(ncomp, 3)):
            rg.add_series(g, np.arange(1, p + 1, dtype=float), model.x_loadings[:, a],
                          f"Component {a + 1}", "line", PAL[a % len(PAL)], width=1.2)
        fresh.add_graph("X Loadings Plot", rg.finish(g, "X Variable Index", "Loading",
                                                     legend=True), width=420, group="Plots")
    if ncomp >= 2:
        g = rg.new_graph("Score Plot", 10.0, 9.0)
        _scatter_by_group(g, model.x_scores[:, 0], model.x_scores[:, 1], None, labels)
        fresh.add_graph("Score Plot", rg.finish(g, "t1", "t2"), width=380, group="Plots")
    report = st._finish(report, fresh)
    return report, report_text(first.book, "Partial Least Squares", report,
                               [f"  {ncomp} components: " + ", ".join(
                                   f"R²({r[0]}) = {r[1]}" for r in fit_rows)])


# ====================================================================== specs
def _sel_split(sel):
    return sel[:-1] if len(sel) > 1 else sel, sel[-1:] if len(sel) > 1 else []


SPECS = {
    "pca": lambda sel: ("Principal Component Analysis", [
        ("Input", [("ranges", "ranges", "Variables", sel),
                   ("range", "labels", "Observation Labels (optional)", ""),
                   ("range", "group", "Group for Score Plot (optional)", "")]),
        ("Settings", [("choice", "matrix", "Analyze", "correlation",
                       {"correlation": "Correlation Matrix", "covariance": "Covariance Matrix"}),
                      ("int", "ncomp", "Number of Components (0 = eigenvalue > mean)", 0,
                       (0, 1000))]),
        ("Plots", [("bool", "scree", "Scree Plot", True),
                   ("bool", "loading_plot", "Loading Plot", True),
                   ("bool", "score_plot", "Score Plot", True),
                   ("bool", "biplot", "Biplot", True)])],
        "Principal components of the variables (listwise deletion).", "report sheet"),
    "kmeans": lambda sel: ("K-Means Cluster Analysis", [
        ("Input", [("ranges", "ranges", "Variables", sel),
                   ("range", "labels", "Observation Labels (optional)", "")]),
        ("Settings", [("int", "k", "Number of Clusters", 3, (2, 1000)),
                      ("choice", "standardize", "Standardize Variables", "none", STANDARDIZE),
                      ("int", "max_iter", "Maximum Iterations", 10, (1, 1000))])],
        "Partitions the observations into k clusters.", "report sheet"),
    "hca": lambda sel: ("Hierarchical Cluster Analysis", [
        ("Input", [("ranges", "ranges", "Variables", sel),
                   ("range", "labels", "Observation Labels (optional)", "")]),
        ("Settings", [("choice", "cluster", "Cluster", "observations",
                       {"observations": "Observations", "variables": "Variables"}),
                      ("choice", "method", "Cluster Method", "baverage", HCA_METHODS),
                      ("choice", "measure", "Distance Type", "seuclid", HCA_MEASURES),
                      ("choice", "standardize", "Standardize Variables", "none", STANDARDIZE),
                      ("int", "k", "Number of Clusters", 3, (1, 1000)),
                      ("bool", "dendrogram", "Dendrogram", True)])],
        "Agglomerative clustering with the schedule and a dendrogram.", "report sheet"),
    "lda": lambda sel: ("Discriminant Analysis", [
        ("Input", [("ranges", "ranges", "Training Data (variables)", _sel_split(sel)[0]),
                   ("range", "group", "Group Range", (_sel_split(sel)[1] or [""])[0]),
                   ("range", "labels", "Observation Labels (optional)", "")]),
        ("Settings", [("choice", "priors", "Prior Probabilities", "equal",
                       {"equal": "Equal", "sizes": "Proportional to group sizes"}),
                      ("bool", "loo", "Leave-One-Out Classification", True)])],
        "Linear discriminant analysis with canonical functions and classification.",
        "report sheet"),
    "pls": lambda sel: ("Partial Least Squares", [
        ("Input", [("ranges", "ranges", "Independent Data (X)", _sel_split(sel)[0]),
                   ("ranges", "response", "Dependent Data (Y)", _sel_split(sel)[1]),
                   ("range", "labels", "Observation Labels (optional)", "")]),
        ("Settings", [("choice", "method", "Method", "nipals",
                       {"nipals": "NIPALS", "simpls": "SIMPLS"}),
                      ("bool", "scale", "Scale Variables (unit variance)", True),
                      ("int", "ncomp", "Number of Components (0 = by cross validation)", 0,
                       (0, 1000)),
                      ("int", "max_comp", "Maximum Components for Cross Validation", 10,
                       (1, 1000)),
                      ("choice", "cv", "Cross Validation", "loo", CV_SCHEMES),
                      ("int", "folds", "Number of Folds (K-fold / Venetian)", 7, (2, 1000))])],
        "PLS regression for collinear predictors (e.g. spectra), with VIP and scores.",
        "report sheet"),
}
FUNCS = {"pca": pca_report, "kmeans": kmeans_report, "hca": hca_report, "lda": lda_report,
         "pls": pls_report}
for _k, _f in FUNCS.items():
    st.register(_k, _f, SPECS[_k])


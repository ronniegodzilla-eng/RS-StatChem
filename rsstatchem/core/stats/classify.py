"""Analyze > Classify: K-Means Cluster (SPSS QUICK CLUSTER), Hierarchical Cluster (CLUSTER) and
Discriminant (DISCRIMINANT).

* K-Means follows the SPSS algorithm: initial centres chosen from the data by the two
  replacement tests of QUICK CLUSTER, then Lloyd iterations (or running means) until the
  largest change in any centre is <= CONVERGE x the minimum distance between initial centres.
* Hierarchical clustering uses the Lance-Williams recurrence on the proximity matrix, so all
  seven SPSS methods (including within-groups linkage) work with any measure; clusters are
  identified by their lowest case number as in the SPSS Agglomeration Schedule; Ward
  coefficients are the cumulative within-cluster sum of squares.
* Discriminant analysis computes Fisher's canonical discriminant functions from the pooled
  within-groups and total SSCP matrices; classification uses Fisher's classification functions
  with the chosen priors and leave-one-out cross-validation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from rsstatchem.core.dataset import Dataset, format_code
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_df, fmt_int, fmt_p, numeric, sup, value_text
from rsstatchem.core.stats.roc import figure, png, style


@dataclass
class ClassifyResult:
    section: Section
    saved: dict[str, tuple[str, pd.Series]] = field(default_factory=dict)


def _frame(ds: Dataset, names: list[str]) -> pd.DataFrame:
    return pd.concat({n: numeric(ds, n) for n in names}, axis=1)


def _case_labels(ds: Dataset, index, label_var: str | None) -> list[str]:
    if not label_var:
        return [str(i + 1) for i in index]
    s = ds.df[label_var] if label_var in ds.df else ds.analysis_series(label_var)
    var = ds.var(label_var)
    out = []
    for i in index:
        v = s.iloc[i]
        out.append(f"{i + 1}:{'' if pd.isna(v) else value_text(var, v)}")
    return out


# ============================================================================ K-Means
def initial_centres(X: np.ndarray, k: int) -> np.ndarray:
    """SPSS QUICK CLUSTER initial cluster centre selection."""
    M = X[:k].astype(float).copy()
    for x in X[k:]:
        d = np.sqrt(((M - x) ** 2).sum(axis=1))
        dm = np.sqrt(((M[:, None, :] - M[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(dm, np.inf)
        m, n_ = np.unravel_index(np.argmin(dm), dm.shape)
        order = np.argsort(d, kind="mergesort")
        q, p = order[0], order[1]
        if d[q] > dm[m, n_]:
            M[m if d[m] <= d[n_] else n_] = x
        elif d[p] > dm[q].min():
            M[q] = x
    return M


def kmeans(X: np.ndarray, k: int, max_iter: int = 10, converge: float = 0.0,
           running_means: bool = False, iterate: bool = True,
           centres: np.ndarray | None = None) -> dict:
    init = initial_centres(X, k) if centres is None else np.asarray(centres, float)
    dmi = np.sqrt(((init[:, None, :] - init[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dmi, np.inf)
    min_init = float(dmi.min()) if k > 1 else 0.0
    M = init.copy()
    history, converged, it = [], False, 0
    if iterate:
        for it in range(1, max_iter + 1):
            old = M.copy()
            if running_means:
                counts = np.zeros(k)
                for x in X:
                    j = int(np.argmin(((M - x) ** 2).sum(axis=1)))
                    counts[j] += 1
                    M[j] += (x - M[j]) / counts[j]
            else:
                lab = np.argmin(((X[:, None, :] - M[None, :, :]) ** 2).sum(axis=2), axis=1)
                for j in range(k):
                    if (lab == j).any():
                        M[j] = X[lab == j].mean(axis=0)
            change = np.sqrt(((M - old) ** 2).sum(axis=1))
            history.append(change)
            if change.max() <= converge * min_init:
                converged = True
                break
    d2 = ((X[:, None, :] - M[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(d2, axis=1)
    return {"initial": init, "final": M, "labels": labels,
            "distance": np.sqrt(d2[np.arange(len(X)), labels]), "history": history,
            "converged": converged, "iterations": it, "min_init": min_init}


def kmeans_cluster(ds: Dataset, names: list[str], k: int = 2, max_iter: int = 10,
                   converge: float = 0.0, running_means: bool = False, iterate: bool = True,
                   initial: bool = True, anova: bool = True, membership: bool = False,
                   distances: bool = False, label_var: str | None = None,
                   save=()) -> ClassifyResult:
    section = Section("Quick Cluster")
    meth = ("KMEANS(UPDATE)" if running_means else "KMEANS(NOUPDATE)") if iterate \
        else "CLASSIFY"
    section.log(f"QUICK CLUSTER {' '.join(names)}\n  /MISSING=LISTWISE\n"
                f"  /CRITERIA=CLUSTER({k}) MXITER({max_iter}) CONVERGE({converge:g})\n"
                f"  /METHOD={meth}"
                + ("\n  /SAVE " + " ".join({"cluster": "CLUSTER", "distance": "DISTANCE"}[s]
                                          for s in save) if save else "")
                + "\n  /PRINT " + " ".join(["INITIAL"] * initial + ["ANOVA"] * anova
                                          + ["CLUSTER"] * membership + ["DISTAN"] * distances)
                + ".")
    result = ClassifyResult(section)
    frame = _frame(ds, names).dropna()
    X = frame.to_numpy(float)
    n = len(X)
    if n < k or k < 2:
        section.note("Warning", "The number of valid cases must be at least the number of "
                     "clusters (k >= 2).", style="warning")
        return result
    res = kmeans(X, k, max_iter, converge, running_means, iterate)
    titles = [ds.var(c).title for c in names]
    ccols = [""] + [("Cluster", str(j + 1)) for j in range(k)]
    decs = [ds.var(c).decimals for c in names]
    if initial:
        section.table(Table("Initial Cluster Centers", ccols, [
            [t] + [fmt(res["initial"][j, i], decs[i]) for j in range(k)]
            for i, t in enumerate(titles)]))
    if iterate:
        h = res["history"]
        last = h[-1] if h else np.zeros(k)
        note = ("a. Convergence achieved due to no or small change in cluster centers. "
                if res["converged"] else
                "a. Iterations stopped because the maximum number of iterations was "
                "performed. Iterations failed to converge. ") + \
            (f"The maximum absolute coordinate change for any center is "
             f"{fmt(float(last.max()) if len(last) else 0.0)}. The current iteration is "
             f"{res['iterations']}. The minimum distance between initial centers is "
             f"{fmt(res['min_init'])}.")
        section.table(Table("Iteration Historyᵃ", ["Iteration"] + [
            ("Change in Cluster Centers", str(j + 1)) for j in range(k)],
            [[str(i + 1)] + [fmt(v) for v in row] for i, row in enumerate(h)],
            footnotes=[note]))
    if membership:
        labels = _case_labels(ds, frame.index, label_var)
        section.table(Table("Cluster Membership", ["Case Number", "Cluster", "Distance"], [
            [labels[i], str(res["labels"][i] + 1), fmt(res["distance"][i])]
            for i in range(n)]))
    section.table(Table("Final Cluster Centers", ccols, [
        [t] + [fmt(res["final"][j, i], decs[i] + 2) for j in range(k)]
        for i, t in enumerate(titles)]))
    if distances:
        M = res["final"]
        dm = np.sqrt(((M[:, None, :] - M[None, :, :]) ** 2).sum(axis=2))
        section.table(Table("Distances between Final Cluster Centers", ["Cluster"] + [
            str(j + 1) for j in range(k)], [[str(i + 1)] + ["" if i == j else fmt(dm[i, j])
                                                           for j in range(k)]
                                            for i in range(k)]))
    if anova:
        rows = []
        lab = res["labels"]
        present = len(np.unique(lab))
        for i, t in enumerate(titles):
            x = X[:, i]
            grand = x.mean()
            ssb = sum(((x[lab == j].mean() - grand) ** 2) * (lab == j).sum()
                      for j in range(k) if (lab == j).any())
            ssw = float(((x - grand) ** 2).sum() - ssb)
            df1, df2 = present - 1, n - present
            msb, msw = ssb / df1 if df1 else float("nan"), ssw / df2 if df2 else float("nan")
            f = msb / msw if msw > 0 else float("nan")
            rows.append([t, fmt(msb), fmt_int(df1), fmt(msw), fmt_int(df2), fmt(f),
                         fmt_p(stats.f.sf(f, df1, df2)) if f == f else ""])
        section.table(Table("ANOVA", ["", ("Cluster", "Mean Square"), ("Cluster", "df"),
                                      ("Error", "Mean Square"), ("Error", "df"), "F", "Sig."],
                            rows, footnotes=[
                                "The F tests should be used only for descriptive purposes "
                                "because the clusters have been chosen to maximize the "
                                "differences among cases in different clusters. The observed "
                                "significance levels are not corrected for this and thus "
                                "cannot be interpreted as tests of the hypothesis that the "
                                "cluster means are equal."]))
    counts = np.bincount(res["labels"], minlength=k)
    crow = [["Cluster" if j == 0 else "", str(j + 1), fmt(counts[j], 3)] for j in range(k)]
    crow += [["Valid", "", fmt(n, 3)], ["Missing", "", fmt(ds.n_cases - n, 3)]]
    section.table(Table("Number of Cases in each Cluster", ["", "", ""], crow,
                        row_header_cols=2))
    index = ds.df.index
    if "cluster" in save:
        result.saved["QCL"] = ("Cluster Number of Case", pd.Series(
            res["labels"] + 1.0, index=frame.index).reindex(index))
    if "distance" in save:
        result.saved["QCL_DIST"] = ("Distance of Case from its Classification Cluster Center",
                                    pd.Series(res["distance"], index=frame.index)
                                    .reindex(index))
    return result


# ============================================================================ Hierarchical
LINKAGES = {"baverage": "Average Linkage (Between Groups)",
            "waverage": "Average Linkage (Within Group)", "single": "Single Linkage",
            "complete": "Complete Linkage", "centroid": "Centroid Method",
            "median": "Median Method", "ward": "Ward Linkage"}
MEASURES = {"seuclid": "Squared Euclidean Distance", "euclid": "Euclidean Distance",
            "correlation": "Pearson Correlation", "cosine": "Cosine",
            "block": "Block", "chebychev": "Chebychev"}
SIMILARITIES = ("correlation", "cosine")
STANDARDIZE = {"none": "None", "z": "Z scores", "range01": "Range 0 to 1",
               "range11": "Range -1 to 1", "max1": "Maximum magnitude of 1"}


def standardize(X: np.ndarray, how: str) -> np.ndarray:
    X = np.asarray(X, float)
    if how == "z":
        sd = X.std(axis=0, ddof=1)
        return (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1)
    if how == "range01":
        r = np.ptp(X, axis=0)
        return (X - X.min(axis=0)) / np.where(r > 0, r, 1)
    if how == "range11":
        r = np.ptp(X, axis=0)
        return X / np.where(r > 0, r, 1)
    if how == "max1":
        m = np.abs(X).max(axis=0)
        return X / np.where(m > 0, m, 1)
    return X


def proximities(X: np.ndarray, measure: str) -> np.ndarray:
    diff = X[:, None, :] - X[None, :, :]
    if measure == "seuclid":
        return (diff ** 2).sum(axis=2)
    if measure == "euclid":
        return np.sqrt((diff ** 2).sum(axis=2))
    if measure == "block":
        return np.abs(diff).sum(axis=2)
    if measure == "chebychev":
        return np.abs(diff).max(axis=2)
    if measure == "cosine":
        norm = np.sqrt((X ** 2).sum(axis=1))
        return (X @ X.T) / np.outer(norm, norm)
    if measure == "correlation":
        Z = X - X.mean(axis=1, keepdims=True)
        norm = np.sqrt((Z ** 2).sum(axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            return (Z @ Z.T) / np.outer(norm, norm)
    raise ValueError(measure)


def agglomerate(P: np.ndarray, method: str, similarity: bool = False):
    """Lance-Williams agglomeration. Returns list of (i, j, coefficient) with i < j case
    indices (0-based, cluster = lowest case index) and a SciPy-style linkage matrix."""
    n = len(P)
    D = (-P if similarity else P).astype(float).copy()
    B = D.copy()
    np.fill_diagonal(B, 0)
    W = np.zeros(n)
    np.fill_diagonal(D, np.inf)
    size = np.ones(n)
    active = np.ones(n, bool)
    zid = np.arange(n)
    merges, Z = [], np.zeros((max(n - 1, 0), 4))
    ess = 0.0
    for stage in range(n - 1):
        flat = int(np.argmin(D))
        i, j = divmod(flat, n)
        if i > j:
            i, j = j, i
        dij = D[i, j]
        ni, nj = size[i], size[j]
        nk = size
        if method == "single":
            new = np.minimum(D[i], D[j])
        elif method == "complete":
            new = np.maximum(D[i], D[j])
        elif method == "baverage":
            new = (ni * D[i] + nj * D[j]) / (ni + nj)
        elif method == "centroid":
            new = (ni * D[i] + nj * D[j]) / (ni + nj) - ni * nj * dij / (ni + nj) ** 2
        elif method == "median":
            new = (D[i] + D[j]) / 2 - dij / 4
        elif method == "ward":
            new = ((nk + ni) * D[i] + (nk + nj) * D[j] - nk * dij) / (nk + ni + nj)
        elif method == "waverage":
            Bij = B[i] + B[j]
            Wij = W[i] + W[j] + B[i, j]
            tot = nk + ni + nj
            new = (W + Wij + Bij) / (tot * (tot - 1) / 2)
            B[i], B[:, i] = Bij, Bij
            W[i] = Wij
        else:
            raise ValueError(method)
        if method == "ward":
            ess += dij / 2
            coef = ess
        else:
            coef = -dij if similarity else dij
        merges.append((i, j, coef))
        Z[stage] = [zid[i], zid[j], abs(dij) if method != "ward" else math.sqrt(2 * ess),
                    ni + nj]
        zid[i] = n + stage
        size[i] = ni + nj
        active[j] = False
        new[~active] = np.inf
        new[i] = np.inf
        D[i], D[:, i] = new, new
        D[j], D[:, j] = np.inf, np.inf
    return merges, Z


def memberships(merges, n: int, k: int) -> np.ndarray:
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j, _ in merges[:max(n - k, 0)]:
        parent[find(j)] = find(i)
    roots, out = {}, np.zeros(n, int)
    for c in range(n):
        r = find(c)
        roots.setdefault(r, len(roots) + 1)
        out[c] = roots[r]
    return out


def _schedule_rows(merges) -> list[list[str]]:
    last_stage: dict[int, int] = {}
    rows, nxt = [], {}
    for s, (i, j, _) in enumerate(merges, start=1):
        for c in (i, j):
            if c in last_stage:
                nxt[last_stage[c]] = s
        last_stage[i] = s
        last_stage.pop(j, None)
    seen: dict[int, int] = {}
    for s, (i, j, coef) in enumerate(merges, start=1):
        rows.append([str(s), str(i + 1), str(j + 1), fmt(coef), str(seen.get(i, 0)),
                     str(seen.get(j, 0)), str(nxt.get(s, 0))])
        seen[i] = s
        seen.pop(j, None)
    return rows


def _dendrogram(Z: np.ndarray, coefs: np.ndarray, labels: list[str], title: str,
                similarity: bool) -> bytes:
    from scipy.cluster.hierarchy import dendrogram

    c = np.asarray(coefs, float)
    if similarity:
        c = -c
    lo, hi = float(c.min()), float(c.max())
    resc = 1 + 24 * (c - lo) / (hi - lo) if hi > lo else np.full(len(c), 1.0)
    Zr = Z.copy()
    Zr[:, 2] = np.maximum.accumulate(resc)
    d = dendrogram(Zr, no_plot=True, labels=labels, orientation="right")
    n = len(labels)
    fig = figure(6.4, max(2.6, 0.17 * n + 1.2))
    ax = fig.add_subplot()
    for xs, ys in zip(d["icoord"], d["dcoord"]):
        ax.plot(ys, xs, color="#C62828", lw=1)
    ax.set_yticks(np.arange(5, 10 * n, 10), d["ivl"], fontsize=7 if n > 40 else 8)
    ax.set_ylim(0, 10 * n)
    ax.set_xlim(0, 25.5)
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    style(ax, title, "Rescaled Distance Cluster Combine", "")
    return png(fig)


def hierarchical(ds: Dataset, names: list[str], method: str = "baverage",
                 measure: str = "seuclid", transform: str = "none",
                 label_var: str | None = None, schedule: bool = True,
                 proximity: bool = False, solution: tuple[int, int] | None = None,
                 dendrogram: bool = True, save: tuple[int, int] | None = None
                 ) -> ClassifyResult:
    """``solution``/``save`` are (min, max) numbers of clusters (equal for a single
    solution)."""
    section = Section("Cluster")
    meth_kw = {"baverage": "BAVERAGE", "waverage": "WAVERAGE", "single": "SINGLE",
               "complete": "COMPLETE", "centroid": "CENTROID", "median": "MEDIAN",
               "ward": "WARD"}[method]
    section.log(f"CLUSTER {' '.join(names)}\n  /METHOD {meth_kw}\n  /MEASURE={measure.upper()}"
                + (f"\n  /STANDARDIZE=VARIABLE {transform.upper()}" if transform != "none"
                   else "") + (f"\n  /ID={label_var}" if label_var else "")
                + "\n  /PRINT " + " ".join(["SCHEDULE"] * schedule + (
                    [f"CLUSTER({solution[0]},{solution[1]})"] if solution else []))
                + ("\n  /PRINT DISTANCE" if proximity else "")
                + ("\n  /PLOT DENDROGRAM" if dendrogram else "\n  /PLOT NONE")
                + (f"\n  /SAVE CLUSTER({save[0]},{save[1]})" if save else "") + ".")
    result = ClassifyResult(section)
    frame = _frame(ds, names).dropna()
    n = len(frame)
    total = ds.n_cases
    section.table(Table("Case Processing Summaryᵃ", [
        (g, h) for g in ("Valid", "Missing", "Total") for h in ("N", "Percent")],
        [[fmt_int(n), fmt(100 * n / total, 1) + "%", fmt_int(total - n),
          fmt(100 * (total - n) / total, 1) + "%", fmt_int(total), "100.0%"]],
        row_header_cols=0, footnotes=[f"a. {MEASURES[measure]} used"]))
    if n < 2:
        section.note("Warning", "At least two valid cases are needed.", style="warning")
        return result
    X = standardize(frame.to_numpy(float), transform)
    P = proximities(X, measure)
    sim = measure in SIMILARITIES
    labels = _case_labels(ds, frame.index, label_var)
    if proximity:
        if n <= 60:
            section.table(Table("Proximity Matrix", ["Case"] + [(MEASURES[measure], lb)
                                                               for lb in labels],
                                [[labels[i]] + ["" if i == j and not sim else fmt(P[i, j])
                                               for j in range(n)] for i in range(n)],
                                footnotes=["This is a similarity matrix" if sim
                                           else "This is a dissimilarity matrix"]))
        else:
            section.note("", "Proximity matrix not shown for more than 60 cases.")
    merges, Z = agglomerate(P, method, sim)
    title = LINKAGES[method]
    if schedule:
        section.note(title, "")
        section.table(Table("Agglomeration Schedule", [
            "Stage", ("Cluster Combined", "Cluster 1"), ("Cluster Combined", "Cluster 2"),
            "Coefficients", ("Stage Cluster First Appears", "Cluster 1"),
            ("Stage Cluster First Appears", "Cluster 2"), "Next Stage"],
            _schedule_rows(merges)))
    if solution:
        lo, hi = max(2, solution[0]), min(n - 1, solution[1])
        ks = list(range(hi, lo - 1, -1))
        if ks:
            mem = {k: memberships(merges, n, k) for k in ks}
            section.table(Table("Cluster Membership", ["Case"] + [f"{k} Clusters"
                                                                  for k in ks],
                                [[labels[i]] + [str(mem[k][i]) for k in ks]
                                 for i in range(n)]))
    if dendrogram:
        if n <= 400:
            section.image(f"Dendrogram using {title}", _dendrogram(
                Z, np.array([m[2] for m in merges]), labels, f"Dendrogram using {title}", sim),
                width=560)
        else:
            section.note("", "Dendrogram not drawn for more than 400 cases.")
    if save:
        lo, hi = max(2, save[0]), min(n - 1, save[1])
        for k in range(hi, lo - 1, -1):
            result.saved[f"CLU{k}"] = (f"{title}", pd.Series(
                memberships(merges, n, k).astype(float), index=frame.index)
                .reindex(ds.df.index))
    return result


# ============================================================================ Discriminant
def _wilks_f(lam: float, p: int, g: int, n: int) -> tuple[float, float, float, bool]:
    """Rao's F approximation for Wilks' lambda (exact when p <= 2 or g <= 3)."""
    vh, ve = g - 1, n - g
    if p * p + vh * vh - 5 > 0:
        t = math.sqrt((p * p * vh * vh - 4) / (p * p + vh * vh - 5))
    else:
        t = 1.0
    w = ve + vh - (p + vh + 1) / 2
    df1 = p * vh
    df2 = w * t - (p * vh - 2) / 2
    lt = lam ** (1 / t)
    f = (1 - lt) / lt * df2 / df1 if lt > 0 else float("inf")
    return f, df1, df2, (p <= 2 or g <= 3)


def _lda(X: np.ndarray, gidx: np.ndarray, g: int):
    n, p = X.shape
    grand = X.mean(axis=0)
    means = np.array([X[gidx == j].mean(axis=0) for j in range(g)])
    Xc = X - means[gidx]
    W = Xc.T @ Xc
    T = (X - grand).T @ (X - grand)
    return means, grand, W, T


def _wilks(cols: list[int], W, T) -> float:
    if not cols:
        return 1.0
    sub = np.ix_(cols, cols)
    return float(np.linalg.det(W[sub]) / np.linalg.det(T[sub]))


def discriminant(ds: Dataset, group: str, lo: float, hi: float, names: list[str],
                 method: str = "enter", priors: str = "equal", means: bool = True,
                 univariate: bool = True, box_m: bool = True, fisher: bool = False,
                 unstandardized: bool = False, loo: bool = True, f_enter: float = 3.84,
                 f_remove: float = 2.71, casewise: bool = False, save=(),
                 tolerance: float = 0.001) -> ClassifyResult:
    gvar = ds.var(group)
    section = Section("Discriminant")
    section.log(f"DISCRIMINANT\n  /GROUPS={group}({lo:g} {hi:g})\n  /VARIABLES={' '.join(names)}"
                + ("\n  /ANALYSIS ALL\n  /METHOD=WILKS\n  /FIN={:g}\n  /FOUT={:g}".format(
                    f_enter, f_remove) if method == "stepwise" else "\n  /ANALYSIS ALL")
                + f"\n  /PRIORS {'EQUAL' if priors == 'equal' else 'SIZE'}"
                + "\n  /STATISTICS=" + " ".join(["MEAN STDDEV"] * means + ["UNIVF"] * univariate
                                               + ["BOXM"] * box_m + ["COEFF"] * fisher
                                               + ["RAW"] * unstandardized + ["TABLE"]
                                               + ["CROSSVALID"] * loo)
                + "\n  /CLASSIFY=NONMISSING POOLED.")
    result = ClassifyResult(section)
    gser = pd.to_numeric(ds.analysis_series(group), errors="coerce")
    data = _frame(ds, names)
    total = ds.n_cases
    in_range = gser.between(lo, hi) & gser.notna()
    miss_x = data.isna().any(axis=1)
    valid = in_range & ~miss_x
    n_gonly = int((~in_range & ~miss_x).sum())
    n_xonly = int((in_range & miss_x).sum())
    n_both = int((~in_range & miss_x).sum())
    nv = int(valid.sum())
    section.table(Table("Analysis Case Processing Summary", ["Unweighted Cases", "", "N",
                                                             "Percent"], [
        ["Valid", "", fmt_int(nv), fmt(100 * nv / total, 1)],
        ["Excluded", "Missing or out-of-range group codes", fmt_int(n_gonly),
         fmt(100 * n_gonly / total, 1)],
        ["", "At least one missing discriminating variable", fmt_int(n_xonly),
         fmt(100 * n_xonly / total, 1)],
        ["", "Both missing or out-of-range group codes and at least one missing "
             "discriminating variable", fmt_int(n_both), fmt(100 * n_both / total, 1)],
        ["", "Total", fmt_int(total - nv), fmt(100 * (total - nv) / total, 1)],
        ["Total", "", fmt_int(total), "100.0"]], row_header_cols=2))
    X = data[valid].to_numpy(float)
    gv = gser[valid].to_numpy(float)
    levels = sorted(np.unique(gv))
    g = len(levels)
    gidx = np.searchsorted(levels, gv)
    n, p = X.shape
    glabels = [value_text(gvar, v) for v in levels]
    titles = [ds.var(c).title for c in names]
    if g < 2 or n - g < 1:
        section.note("Warning", "At least two non-empty groups are needed.", style="warning")
        return result
    sizes = np.bincount(gidx, minlength=g)
    if means:
        rows = []
        for j in list(range(g)) + [None]:
            Xg = X if j is None else X[gidx == j]
            for i, t in enumerate(titles):
                d = ds.var(names[i]).decimals
                rows.append([("Total" if j is None else glabels[j]) if i == 0 else "", t,
                             fmt(Xg[:, i].mean(), d + 2),
                             fmt(Xg[:, i].std(ddof=1) if len(Xg) > 1 else float("nan"), d + 3),
                             fmt_int(len(Xg)), fmt(len(Xg), 3)])
        section.table(Table("Group Statistics", [gvar.title, "", "Mean", "Std. Deviation",
                                                 ("Valid N (listwise)", "Unweighted"),
                                                 ("Valid N (listwise)", "Weighted")], rows,
                            row_header_cols=2))
    gmeans, grand, W, T = _lda(X, gidx, g)
    if univariate:
        rows = []
        for i, t in enumerate(titles):
            lam = W[i, i] / T[i, i] if T[i, i] > 0 else float("nan")
            f = ((1 - lam) / (g - 1)) / (lam / (n - g)) if lam > 0 else float("nan")
            rows.append([t, fmt(lam), fmt(f), fmt_int(g - 1), fmt_int(n - g),
                         fmt_p(stats.f.sf(f, g - 1, n - g))])
        section.table(Table("Tests of Equality of Group Means", ["", "Wilks' Lambda", "F",
                                                                 "df1", "df2", "Sig."], rows))

    # ---------------------------------------------------------------- stepwise selection
    selected = list(range(p))
    if method == "stepwise":
        selected = []
        steps = []
        for _ in range(2 * p + 2):
            changed = False
            q = len(selected)
            best = None
            for c in range(p):
                if c in selected:
                    continue
                cols = selected + [c]
                tol = _tolerance(W, selected, c)
                if tol < tolerance:
                    continue
                lam_new = _wilks(cols, W, T)
                f_in = (n - g - q) / (g - 1) * (_wilks(selected, W, T) / lam_new - 1)
                if best is None or lam_new < best[1]:
                    best = (c, lam_new, f_in)
            if best is not None and best[2] >= f_enter:
                selected.append(best[0])
                steps.append(("enter", best[0], list(selected)))
                changed = True
                q = len(selected)
                worst = None
                for c in selected:
                    rest = [x for x in selected if x != c]
                    f_out = (n - g - q + 1) / (g - 1) * (_wilks(rest, W, T)
                                                          / _wilks(selected, W, T) - 1)
                    if worst is None or f_out < worst[1]:
                        worst = (c, f_out)
                if worst is not None and worst[1] < f_remove and worst[0] != best[0]:
                    selected.remove(worst[0])
                    steps.append(("remove", worst[0], list(selected)))
            if not changed:
                break
        _stepwise_tables(section, steps, titles, W, T, n, g, p, f_enter, f_remove, tolerance)
        if not selected:
            section.note("Warning", "No variables qualify for analysis.", style="warning")
            return result
    cols = selected
    Xs = X[:, cols]
    gm = gmeans[:, cols]
    Ws, Ts = W[np.ix_(cols, cols)], T[np.ix_(cols, cols)]
    ps = len(cols)
    stitles = [titles[c] for c in cols]
    Sp = Ws / (n - g)

    if box_m:
        _box_m(section, Xs, gidx, g, glabels, Sp, n)

    B = Ts - Ws
    evals, evecs = _eig(Ws, B)
    m = min(ps, g - 1)
    evals, evecs = evals[:m], evecs[:, :m]
    scale = np.sqrt(np.einsum("ij,jk,ki->i", evecs.T, Sp, evecs))
    V = evecs / scale
    sd_w = np.sqrt(np.diag(Sp))
    Rw = Sp / np.outer(sd_w, sd_w)
    std = V * sd_w[:, None]
    struct = Rw @ std
    for k in range(m):
        if struct[np.argmax(np.abs(struct[:, k])), k] < 0:
            V[:, k] *= -1
            std[:, k] *= -1
            struct[:, k] *= -1
    const = -grand[cols] @ V
    tot = evals.sum()
    erows = []
    for k in range(m):
        erows.append([str(k + 1), fmt(evals[k]) + sup("a"), fmt(100 * evals[k] / tot, 1),
                      fmt(100 * evals[:k + 1].sum() / tot, 1),
                      fmt(math.sqrt(evals[k] / (1 + evals[k])))])
    section.note("Summary of Canonical Discriminant Functions", "")
    section.table(Table("Eigenvalues", ["Function", "Eigenvalue", "% of Variance",
                                        "Cumulative %", "Canonical Correlation"], erows,
                        footnotes=[f"a. First {m} canonical discriminant functions were used "
                                   "in the analysis."]))
    wrows = []
    for k in range(m):
        lam = float(np.prod(1 / (1 + evals[k:])))
        chi = -(n - 1 - (ps + g) / 2) * math.log(lam)
        df = (ps - k) * (g - k - 1)
        wrows.append([f"{k + 1} through {m}" if k < m - 1 else str(k + 1), fmt(lam), fmt(chi),
                      fmt_int(df), fmt_p(stats.chi2.sf(chi, df))])
    section.table(Table("Wilks' Lambda", ["Test of Function(s)", "Wilks' Lambda", "Chi-square",
                                          "df", "Sig."], wrows))
    fcols = [""] + [("Function", str(k + 1)) for k in range(m)]
    section.table(Table("Standardized Canonical Discriminant Function Coefficients", fcols,
                        [[t] + [fmt(v) for v in std[i]] for i, t in enumerate(stitles)]))
    best_f = np.argmax(np.abs(struct), axis=1)
    order = sorted(range(ps), key=lambda i: (best_f[i], -abs(struct[i, best_f[i]])))
    srows = [[stitles[i]] + [fmt(struct[i, k]) + ("*" if k == best_f[i] and m > 1 else "")
                             for k in range(m)] for i in order]
    removed = [c for c in range(p) if c not in cols]
    if removed:
        Wfull = W / (n - g)
        sdf = np.sqrt(np.diag(Wfull))
        Rfull = Wfull / np.outer(sdf, sdf)
        for c in removed:
            # SPSS also lists variables not used in the analysis, correlated with the functions
            corr = Rfull[c, cols] @ std
            srows.append([titles[c] + sup("b")] + [fmt(v) for v in corr])
    notes = ["Pooled within-groups correlations between discriminating variables and "
             "standardized canonical discriminant functions",
             "Variables ordered by absolute size of correlation within function."]
    if m > 1:
        notes.append("*. Largest absolute correlation between each variable and any "
                     "discriminant function")
    if removed:
        notes.append("b. This variable not used in the analysis.")
    section.table(Table("Structure Matrix", fcols, srows, footnotes=notes))
    if unstandardized:
        section.table(Table("Canonical Discriminant Function Coefficients", fcols,
                            [[t] + [fmt(v) for v in V[i]] for i, t in enumerate(stitles)]
                            + [["(Constant)"] + [fmt(v) for v in const]],
                            footnotes=["Unstandardized coefficients"]))
    cent = gm @ V + const
    section.table(Table("Functions at Group Centroids", [gvar.title] + fcols[1:], [
        [glabels[j]] + [fmt(v) for v in cent[j]] for j in range(g)], footnotes=[
        "Unstandardized canonical discriminant functions evaluated at group means"]))

    prior = np.full(g, 1 / g) if priors == "equal" else sizes / n
    section.note("Classification Statistics", "")
    section.table(Table("Prior Probabilities for Groups", [gvar.title, "Prior",
                                                           ("Cases Used in Analysis",
                                                            "Unweighted"),
                                                           ("Cases Used in Analysis",
                                                            "Weighted")],
                        [[glabels[j], fmt(prior[j]), fmt_int(sizes[j]), fmt(sizes[j], 3)]
                         for j in range(g)] + [["Total", fmt(prior.sum()), fmt_int(n),
                                                fmt(n, 3)]]))
    Spi = np.linalg.inv(Sp)
    coef = gm @ Spi
    ccon = -0.5 * np.einsum("ij,ij->i", coef, gm) + np.log(prior)
    if fisher:
        section.table(Table("Classification Function Coefficients", [""] + [
            (gvar.title, lb) for lb in glabels],
            [[t] + [fmt(coef[j, i]) for j in range(g)] for i, t in enumerate(stitles)]
            + [["(Constant)"] + [fmt(ccon[j]) for j in range(g)]],
            footnotes=["Fisher's linear discriminant functions"]))
    scores = Xs @ coef.T + ccon
    pred = scores.argmax(axis=1)
    rows = _class_rows("Original", gidx, pred, g, glabels)
    notes = [f"a. {fmt(100 * (pred == gidx).mean(), 1)}% of original grouped cases correctly "
             "classified."]
    if loo:
        cv = _loo_predict(Xs, gidx, g, prior)
        rows += _class_rows("Cross-validatedᵇ", gidx, cv, g, glabels)
        notes += ["b. Cross validation is done only for those cases in the analysis. In cross "
                  "validation, each case is classified by the functions derived from all cases "
                  "other than that case.",
                  f"c. {fmt(100 * (cv == gidx).mean(), 1)}% of cross-validated grouped cases "
                  "correctly classified."]
    section.table(Table("Classification Resultsᵃ" + (" ᶜ" if loo else ""), [
        "", "", gvar.title] + [("Predicted Group Membership", lb) for lb in glabels]
        + ["Total"], rows, row_header_cols=3, footnotes=notes))
    if save:
        allX = data[[names[c] for c in cols]].to_numpy(float)
        ok = ~np.isnan(allX).any(axis=1)
        sc = np.full((total, g), np.nan)
        sc[ok] = allX[ok] @ coef.T + ccon
        index = ds.df.index
        if "group" in save:
            grp = np.full(total, np.nan)
            grp[ok] = np.asarray(levels)[sc[ok].argmax(axis=1)]
            result.saved["Dis"] = ("Predicted Group for Analysis 1", pd.Series(grp,
                                                                                index=index))
        if "scores" in save:
            fs = np.full((total, m), np.nan)
            fs[ok] = allX[ok] @ V + const
            for k in range(m):
                result.saved[f"Dis{k + 1}"] = (
                    f"Discriminant Scores from Function {k + 1} for Analysis 1",
                    pd.Series(fs[:, k], index=index))
        if "probs" in save:
            post = np.full((total, g), np.nan)
            e = np.exp(sc[ok] - sc[ok].max(axis=1, keepdims=True))
            post[ok] = e / e.sum(axis=1, keepdims=True)
            for j in range(g):
                result.saved[f"Dis{j + 1}_P"] = (
                    f"Probabilities of Membership in Group {format_code(levels[j])} for "
                    "Analysis 1", pd.Series(post[:, j], index=index))
    return result


def _tolerance(W: np.ndarray, selected: list[int], c: int) -> float:
    if not selected:
        return 1.0
    R = W / np.sqrt(np.outer(np.diag(W), np.diag(W)))
    sub = R[np.ix_(selected, selected)]
    r = R[selected, c]
    return float(1 - r @ np.linalg.solve(sub, r))


def _eig(W: np.ndarray, B: np.ndarray):
    from scipy.linalg import eigh

    vals, vecs = eigh(B, W)
    order = np.argsort(vals)[::-1]
    return np.clip(vals[order], 0, None), vecs[:, order]


def _class_rows(label: str, truth, pred, g, glabels) -> list[list[str]]:
    rows = []
    for j in range(g):
        c = [int(((truth == j) & (pred == k)).sum()) for k in range(g)]
        rows.append([label if j == 0 else "", "Count" if j == 0 else "", glabels[j]]
                    + [fmt_int(v) for v in c] + [fmt_int(sum(c))])
    for j in range(g):
        c = [int(((truth == j) & (pred == k)).sum()) for k in range(g)]
        s = sum(c)
        rows.append(["", "%" if j == 0 else "", glabels[j]]
                    + [fmt(100 * v / s, 1) if s else "" for v in c] + ["100.0"])
    return rows


def _loo_predict(X: np.ndarray, gidx: np.ndarray, g: int, prior: np.ndarray) -> np.ndarray:
    n, p = X.shape
    sizes = np.bincount(gidx, minlength=g).astype(float)
    means = np.array([X[gidx == j].mean(axis=0) for j in range(g)])
    Xc = X - means[gidx]
    W = Xc.T @ Xc
    out = np.empty(n, int)
    for i in range(n):
        j = gidx[i]
        nj = sizes[j]
        if nj < 2:
            out[i] = -1
            continue
        d = X[i] - means[j]
        Wi = W - nj / (nj - 1) * np.outer(d, d)
        mi = means.copy()
        mi[j] = (nj * means[j] - X[i]) / (nj - 1)
        S = Wi / (n - 1 - g)
        try:
            Si = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            Si = np.linalg.pinv(S)
        coef = mi @ Si
        score = coef @ X[i] - 0.5 * np.einsum("ij,ij->i", coef, mi) + np.log(prior)
        out[i] = int(np.argmax(score))
    return out


def _box_m(section: Section, X, gidx, g, glabels, Sp, n) -> None:
    p = X.shape[1]
    rows, logdet, nis = [], [], []
    m_stat = (n - g) * np.linalg.slogdet(Sp)[1]
    ok = True
    for j in range(g):
        Xg = X[gidx == j]
        nj = len(Xg)
        nis.append(nj)
        if nj < 2:
            rows.append([glabels[j], "0", ""])
            ok = False
            continue
        S = np.cov(Xg, rowvar=False).reshape(p, p)
        rank = np.linalg.matrix_rank(S)
        sign, ld = np.linalg.slogdet(S)
        rows.append([glabels[j], str(rank), fmt(ld) if sign > 0 else ""])
        if rank < p or sign <= 0:
            ok = False
        else:
            m_stat -= (nj - 1) * ld
    sign, ldp = np.linalg.slogdet(Sp)
    rows.append(["Pooled within-groups", str(np.linalg.matrix_rank(Sp)), fmt(ldp)])
    section.table(Table("Log Determinants", ["", "Rank", "Log Determinant"], rows, footnotes=[
        "The ranks and natural logarithms of determinants printed are those of the group "
        "covariance matrices."]))
    if not ok:
        section.note("Box's Test of Equality of Covariance Matrices",
                     "Box's M cannot be computed: at least one group covariance matrix is "
                     "singular.", style="warning")
        return
    nis = np.asarray(nis, float)
    c1 = (np.sum(1 / (nis - 1)) - 1 / (n - g)) * (2 * p * p + 3 * p - 1) / (6 * (p + 1) * (g - 1))
    c2 = (np.sum(1 / (nis - 1) ** 2) - 1 / (n - g) ** 2) * (p - 1) * (p + 2) / (6 * (g - 1))
    df1 = (g - 1) * p * (p + 1) / 2
    if c2 - c1 * c1 > 0:
        df2 = (df1 + 2) / (c2 - c1 * c1)
        b = df1 / (1 - c1 - df1 / df2)
        f = m_stat / b
    else:
        df2 = (df1 + 2) / (c1 * c1 - c2)
        b = df2 / (1 - c1 + 2 / df2)
        f = df2 * m_stat / (df1 * (b - m_stat))
    section.note("Box's Test of Equality of Covariance Matrices", "")
    section.table(Table("Test Results", ["", "", ""], [
        ["Box's M", "", fmt(m_stat)], ["F", "Approx.", fmt(f)], ["", "df1", fmt_int(df1)],
        ["", "df2", fmt(df2)], ["", "Sig.", fmt_p(stats.f.sf(f, df1, df2))]],
        row_header_cols=2, footnotes=["Tests null hypothesis of equal population covariance "
                                      "matrices."]))


def _stepwise_tables(section, steps, titles, W, T, n, g, p, f_enter, f_remove,
                     tolerance) -> None:
    erows, inrows, outrows, wrows = [], [], [], []
    for s, (action, c, sel) in enumerate(steps, start=1):
        lam = _wilks(sel, W, T)
        q = len(sel)
        f, df1, df2, exact = _wilks_f(lam, q, g, n)
        erows.append([str(s), titles[c] if action == "enter" else "",
                      titles[c] if action == "remove" else "", fmt(lam), fmt_int(q),
                      fmt_int(g - 1), fmt_int(n - g), fmt(f), fmt_int(df1), fmt_df(df2),
                      fmt_p(stats.f.sf(f, df1, df2))])
        wrows.append([str(s), fmt_int(q), fmt(lam), fmt_int(q), fmt_int(g - 1),
                      fmt_int(n - g), fmt(f), fmt_int(df1), fmt_df(df2),
                      fmt_p(stats.f.sf(f, df1, df2))])
        for j, v in enumerate(sel):
            rest = [x for x in sel if x != v]
            f_out = (n - g - q + 1) / (g - 1) * (_wilks(rest, W, T) / lam - 1)
            inrows.append([str(s) if j == 0 else "", titles[v],
                           fmt(_tolerance(W, rest, v)), fmt(f_out),
                           fmt(_wilks(rest, W, T)) if q > 1 else ""])
    for s, sel in enumerate([[]] + [st[2] for st in steps]):
        first = True
        for c in range(p):
            if c in sel:
                continue
            tol = _tolerance(W, sel, c)
            lam_new = _wilks(sel + [c], W, T)
            f_in = (n - g - len(sel)) / (g - 1) * (_wilks(sel, W, T) / lam_new - 1)
            min_tol = min([tol] + [_tolerance(W, [x for x in sel + [c] if x != v], v)
                                   for v in sel])
            outrows.append([str(s) if first else "", titles[c], fmt(tol), fmt(min_tol),
                            fmt(f_in), fmt(lam_new)])
            first = False
    section.table(Table("Variables Entered/Removedᵃ ᵇ ᶜ ᵈ", [
        "Step", "Entered", "Removed", ("Wilks' Lambda", "Statistic"), ("Wilks' Lambda", "df1"),
        ("Wilks' Lambda", "df2"), ("Wilks' Lambda", "df3"), ("Exact F", "Statistic"),
        ("Exact F", "df1"), ("Exact F", "df2"), ("Exact F", "Sig.")], erows, footnotes=[
        "At each step, the variable that minimizes the overall Wilks' Lambda is entered.",
        "a. Maximum number of steps is " + str(2 * p) + ".",
        f"b. Minimum partial F to enter is {f_enter:g}.",
        f"c. Maximum partial F to remove is {f_remove:g}.",
        "d. F level, tolerance, or VIN insufficient for further computation.",
        "F is Rao's approximation where the exact F is not available."]))
    section.table(Table("Variables in the Analysis", ["Step", "", "Tolerance", "F to Remove",
                                                      "Wilks' Lambda"], inrows))
    section.table(Table("Variables Not in the Analysis", ["Step", "", "Tolerance",
                                                          "Min. Tolerance", "F to Enter",
                                                          "Wilks' Lambda"], outrows))
    section.table(Table("Wilks' Lambda", ["Step", "Number of Variables", "Lambda", "df1",
                                          "df2", "df3", ("Exact F", "Statistic"),
                                          ("Exact F", "df1"), ("Exact F", "df2"),
                                          ("Exact F", "Sig.")], wrows))

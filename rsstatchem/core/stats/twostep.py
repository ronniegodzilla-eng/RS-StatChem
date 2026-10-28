"""Analyze > Classify > TwoStep Cluster and Cluster Silhouettes (SPSS TWOSTEP CLUSTER and the
STATS CLUS SIL extension), following IBM's "TWOSTEP CLUSTER Algorithms" and "Cluster
Evaluation Algorithms" chapters (Chiu, Fang, Chen, Wang & Jeris 2001; Zhang et al. 1996).

* **Pre-cluster**: a BIRCH cluster-feature (CF) tree (branching factor B, maximum depth D,
  threshold starting at 0) is built record by record; a record joins the closest leaf entry when
  its distance is below the threshold, leaf nodes split around the farthest pair, and when the
  tree outgrows its maximum size it is rebuilt with a larger threshold (the mean closest-pair
  distance of the leaf entries). Each CF holds N, the sums and sums of squares of the
  continuous variables and the category counts of the categorical variables.
* **Distance**: log-likelihood, d(j, s) = ξ_j + ξ_s − ξ_<j,s> with
  ξ_v = −N_v (Σ_k ½ log(σ̂²_k + σ̂²_vk) + Σ_k Ê_vk) (Ê = category entropy), or Euclidean
  distance between cluster centres (continuous variables only).
* **Cluster**: agglomerative hierarchical clustering of the sub-clusters; the number of clusters
  is chosen by IBM's two-stage rule: BIC (or AIC) change ratio R1(J) = dBIC(J)/dBIC(1) < 0.04
  for the initial estimate, then the ratio of distance changes R2(J) = d_min(C_J)/d_min(C_J+1)
  with the 1.15 rule.
* **Outlier handling** (optional): leaf entries smaller than a fraction (25%) of the largest
  are set aside; a record whose log-likelihood distance to its closest cluster exceeds
  log V (V = product of the continuous ranges and the category counts) is noise (cluster -1).
* **Quality**: the IBM silhouette (A/B as distances to cluster centroids, continuous inputs
  normalised to [-1, 1], categorical inputs one-of-c) and predictor importance
  (−log10 p of the F / chi-square test across clusters).

SPSS's pre-clustering depends on the case order (the IBM chapter says so); this implementation
reproduces the algorithm, not SPSS's exact sub-clusters. Cluster Silhouettes computes the
Kaufman-Rousseeuw silhouette s(i) = (b − a)/max(a, b) from all pairwise distances.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int
from rsstatchem.core.stats.ml_common import (MLError, MLResult, categorical, full_series,
                                             importance_chart, raw_column)
from rsstatchem.core.stats.roc import figure, png, style


# ============================================================================ CF algebra
@dataclass
class CFSpace:
    """Global quantities the log-likelihood distance needs."""
    kc: int                       # continuous variables
    ncat: list[int]               # categories per categorical variable
    var: np.ndarray               # σ̂²_k (overall variances of the continuous variables)
    measure: str = "ll"           # ll | euclidean

    def __post_init__(self):
        self.offsets = np.r_[0, np.cumsum(self.ncat)].astype(int)


def xi(space: CFSpace, N, S, Q, C) -> np.ndarray:
    """ξ_v for arrays of CFs: N (m,), S/Q (m, kc), C (m, sum ncat)."""
    N = np.asarray(N, float)
    out = np.zeros(len(N))
    if space.kc:
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = S / N[:, None]
            v = np.maximum(Q / N[:, None] - mean ** 2, 0.0)
        out += 0.5 * np.log(space.var[None, :] + v).sum(1)
    off = space.offsets
    for k in range(len(space.ncat)):
        c = C[:, off[k]:off[k + 1]]
        with np.errstate(divide="ignore", invalid="ignore"):
            p = c / N[:, None]
            out += -np.where(p > 0, p * np.log(p), 0.0).sum(1)
    return -N * out


def distance(space: CFSpace, a: tuple, B: tuple) -> np.ndarray:
    """Distance between one CF ``a`` (scalars/1-d) and every CF in ``B`` (arrays)."""
    Na, Sa, Qa, Ca = a
    Nb, Sb, Qb, Cb = B
    if space.measure == "euclidean":
        return np.sqrt(((Sb / Nb[:, None] - Sa[None, :] / Na) ** 2).sum(1))
    xa = xi(space, np.array([Na]), Sa[None, :], Qa[None, :], Ca[None, :])[0]
    xb = xi(space, Nb, Sb, Qb, Cb)
    xab = xi(space, Nb + Na, Sb + Sa[None, :], Qb + Qa[None, :], Cb + Ca[None, :])
    return np.maximum(xa + xb - xab, 0.0)


class _Entries:
    """A growable list of CFs stored as arrays."""

    def __init__(self, kc: int, kcat: int):
        self.N = np.zeros(0)
        self.S = np.zeros((0, kc))
        self.Q = np.zeros((0, kc))
        self.C = np.zeros((0, kcat))

    def __len__(self):
        return len(self.N)

    def add(self, cf):
        self.N = np.r_[self.N, cf[0]]
        self.S = np.vstack([self.S, cf[1]])
        self.Q = np.vstack([self.Q, cf[2]])
        self.C = np.vstack([self.C, cf[3]])

    def get(self, i):
        return (self.N[i], self.S[i], self.Q[i], self.C[i])

    def absorb(self, i, cf):
        self.N[i] += cf[0]
        self.S[i] += cf[1]
        self.Q[i] += cf[2]
        self.C[i] += cf[3]

    def arrays(self):
        return self.N, self.S, self.Q, self.C

    def subset(self, idx):
        e = _Entries(self.S.shape[1], self.C.shape[1])
        e.N, e.S, e.Q, e.C = self.N[idx], self.S[idx], self.Q[idx], self.C[idx]
        return e

    def total(self):
        return (self.N.sum(), self.S.sum(0), self.Q.sum(0), self.C.sum(0))


class _Node:
    def __init__(self, leaf: bool, kc: int, kcat: int):
        self.leaf = leaf
        self.entries = _Entries(kc, kcat)
        self.children: list[_Node] = []
        self.members: list[list[int]] = []     # record ids per leaf entry


class CFTree:
    def __init__(self, space: CFSpace, branch: int = 8, depth: int = 3,
                 threshold: float = 0.0, max_entries: int = 600):
        self.space, self.B, self.D = space, int(branch), int(depth)
        self.T = float(threshold)
        self.max_entries = int(max_entries)
        kcat = int(sum(space.ncat))
        self.kc, self.kcat = space.kc, kcat
        self.root = _Node(True, space.kc, kcat)
        self.height = 1
        self.n_leaf_entries = 0
        self.rebuilds = 0

    # -------------------------------------------------------------- insertion
    def insert(self, cf, members: list[int]) -> None:
        split = self._insert(self.root, cf, members)
        if split is not None:
            new_root = _Node(False, self.kc, self.kcat)
            for child in split:
                new_root.children.append(child)
                new_root.entries.add(child.entries.total())
            self.root = new_root
            self.height += 1

    def _insert(self, node: _Node, cf, members):
        if node.leaf:
            if len(node.entries):
                d = distance(self.space, cf, node.entries.arrays())
                i = int(np.argmin(d))
                if d[i] <= self.T:
                    node.entries.absorb(i, cf)
                    node.members[i].extend(members)
                    return None
            node.entries.add(cf)
            node.members.append(list(members))
            self.n_leaf_entries += 1
            if len(node.entries) > self.B:
                return self._split(node)
            return None
        d = distance(self.space, cf, node.entries.arrays())
        i = int(np.argmin(d))
        child = node.children[i]
        res = self._insert(child, cf, members)
        if res is None:
            node.entries.absorb(i, cf)
            return None
        a, b = res
        node.children[i:i + 1] = [a, b]
        node.entries = _Entries(self.kc, self.kcat)
        for child_node in node.children:
            node.entries.add(child_node.entries.total())
        if len(node.children) > self.B:
            return self._split(node)
        return None

    def _split(self, node: _Node):
        E = node.entries
        m = len(E)
        D = np.zeros((m, m))
        for i in range(m):
            D[i] = distance(self.space, E.get(i), E.arrays())
        i, j = np.unravel_index(np.argmax(D), D.shape)
        if i == j:
            i, j = 0, m - 1
        side = np.where(D[:, i] <= D[:, j], 0, 1)
        side[i], side[j] = 0, 1
        out = []
        for s in (0, 1):
            idx = np.where(side == s)[0]
            nn = _Node(node.leaf, self.kc, self.kcat)
            nn.entries = E.subset(idx)
            if node.leaf:
                nn.members = [node.members[k] for k in idx]
            else:
                nn.children = [node.children[k] for k in idx]
            out.append(nn)
        return out

    # -------------------------------------------------------------- leaves
    def leaves(self) -> list[_Node]:
        out, stack = [], [self.root]
        while stack:
            n = stack.pop()
            if n.leaf:
                out.append(n)
            else:
                stack.extend(n.children)
        return out[::-1]

    def leaf_entries(self):
        ents = _Entries(self.kc, self.kcat)
        members = []
        for lf in self.leaves():
            for i in range(len(lf.entries)):
                ents.add(lf.entries.get(i))
                members.append(lf.members[i])
        return ents, members

    def too_big(self) -> bool:
        return self.height > self.D + 1 or self.n_leaf_entries > self.max_entries

    def rebuild(self) -> None:
        ents, members = self.leaf_entries()
        mins = []
        for lf in self.leaves():
            m = len(lf.entries)
            if m < 2:
                continue
            best = math.inf
            for i in range(m):
                d = distance(self.space, lf.entries.get(i), lf.entries.arrays())
                d[i] = math.inf
                best = min(best, float(d.min()))
            mins.append(best)
        newT = float(np.mean(mins)) if mins else self.T * 2
        self.T = max(newT, self.T * 1.5, 1e-9)
        self.root = _Node(True, self.kc, self.kcat)
        self.height = 1
        self.n_leaf_entries = 0
        self.rebuilds += 1
        for i in range(len(ents)):
            self.insert(ents.get(i), members[i])


# ============================================================================ clustering
def precluster(space: CFSpace, Xc: np.ndarray, Xcat: np.ndarray, branch=8, depth=3,
               threshold=0.0, max_entries=600, outliers: bool = False,
               outlier_frac: float = 0.25):
    """Returns (sub-cluster entries, member lists, outlier record ids)."""
    tree = CFTree(space, branch, depth, threshold, max_entries)
    off = space.offsets
    n = len(Xc) if space.kc else len(Xcat)
    held: list[tuple] = []
    for r in range(n):
        C = np.zeros(int(sum(space.ncat)))
        for k in range(len(space.ncat)):
            C[off[k] + int(Xcat[r, k])] = 1.0
        S = Xc[r] if space.kc else np.zeros(0)
        cf = (1.0, S.astype(float), (S ** 2).astype(float), C)
        tree.insert(cf, [r])
        while tree.too_big():
            if outliers:
                ents, members = tree.leaf_entries()
                big = ents.N.max()
                small = [i for i in range(len(ents)) if ents.N[i] < outlier_frac * big]
                keep = [i for i in range(len(ents)) if i not in set(small)]
                held += [(ents.get(i), members[i]) for i in small]
                _reset(tree, ents.subset(keep), [members[i] for i in keep])
                if not tree.too_big():
                    break
            tree.rebuild()
    for cf, mem in held:  # try to fit set-aside entries back without growing the tree
        tree.insert(cf, mem)
        if tree.too_big():
            tree.rebuild()
    ents, members = tree.leaf_entries()
    out_ids: list[int] = []
    # with a small file the tree never fills and every entry holds one record, so nothing can
    # be "small relative to the largest entry": coarsen the tree until entries aggregate
    while outliers and len(ents) > 1 and ents.N.max() < 2 / max(outlier_frac, 1e-9) and \
            tree.rebuilds < 12:
        tree.rebuild()
        ents, members = tree.leaf_entries()
    if outliers and len(ents) > 1:
        big = ents.N.max()
        keep = [i for i in range(len(ents)) if ents.N[i] >= outlier_frac * big]
        out_ids = [r for i in range(len(ents)) if i not in set(keep) for r in members[i]]
        ents, members = ents.subset(keep), [members[i] for i in keep]
    return ents, members, out_ids, tree


def _reset(tree: CFTree, ents: _Entries, members):
    tree.root = _Node(True, tree.kc, tree.kcat)
    tree.height = 1
    tree.n_leaf_entries = 0
    for i in range(len(ents)):
        tree.insert(ents.get(i), members[i])


@dataclass
class Hierarchy:
    merges: list            # (J clusters before the merge, distance)
    labels: dict            # J -> sub-cluster labels (0..J-1)
    xi_sum: dict            # J -> Σ ξ_j


def agglomerate(space: CFSpace, ents: _Entries, max_k: int) -> Hierarchy:
    N, S, Q, C = [a.copy() for a in ents.arrays()]
    m = len(N)
    alive = np.ones(m, bool)
    label = np.arange(m)
    D = np.full((m, m), np.inf)
    for i in range(m):
        d = distance(space, (N[i], S[i], Q[i], C[i]), (N, S, Q, C))
        D[i] = d
    np.fill_diagonal(D, np.inf)
    merges = []
    labels, xs = {}, {}
    xcur = xi(space, N, S, Q, C)
    J = m
    if J <= max_k:
        labels[J] = _relabel(label)
        xs[J] = float(xcur.sum())
    rmin = D.min(1)
    rarg = D.argmin(1)
    while J > 1:
        i = int(np.argmin(rmin))
        j = int(rarg[i])
        if i > j:
            i, j = j, i
        merges.append((J, float(D[i, j])))
        N[i] += N[j]
        S[i] += S[j]
        Q[i] += Q[j]
        C[i] += C[j]
        alive[j] = False
        label[label == j] = i
        D[j, :] = np.inf
        D[:, j] = np.inf
        d = distance(space, (N[i], S[i], Q[i], C[i]), (N, S, Q, C))
        d[~alive] = np.inf
        d[i] = np.inf
        D[i, :] = d
        D[:, i] = d
        rmin[j] = np.inf
        rmin[i], rarg[i] = (float(d.min()), int(d.argmin())) if alive.sum() > 1 else \
            (np.inf, i)
        stale = np.where(alive & ((rarg == i) | (rarg == j)))[0]
        for r in stale:
            if r != i:
                rmin[r], rarg[r] = float(D[r].min()), int(D[r].argmin())
        better = alive & (d < rmin)
        better[i] = False
        rmin[better] = d[better]
        rarg[better] = i
        xcur[i] = xi(space, N[i:i + 1], S[i:i + 1], Q[i:i + 1], C[i:i + 1])[0]
        xcur[j] = 0.0
        J -= 1
        if J <= max_k:
            labels[J] = _relabel(label)
            xs[J] = float(xcur[alive].sum())
    return Hierarchy(merges, labels, xs)


def _relabel(label: np.ndarray) -> np.ndarray:
    _, inv = np.unique(label, return_inverse=True)
    return inv


def auto_clusters(h: Hierarchy, space: CFSpace, N: int, max_k: int,
                  criterion: str = "bic") -> tuple[int, list[dict]]:
    """IBM's two-stage number-of-clusters rule. Returns (k, table rows)."""
    Ka = space.kc
    mpar = lambda J: J * (2 * Ka + sum(L - 1 for L in space.ncat))  # noqa: E731
    Js = sorted(h.xi_sum)
    crit = {}
    for J in Js:
        pen = mpar(J) * (math.log(N) if criterion == "bic" else 2.0)
        crit[J] = -2 * h.xi_sum[J] + pen
    dist = {J: d for J, d in h.merges}     # d_min(C_J): merge distance from J to J-1
    rows = []
    top = min(max_k, max(Js))
    for J in range(1, top + 1):
        row = {"J": J, "crit": crit.get(J, float("nan"))}
        if J > 1:
            row["change"] = crit[J] - crit[J - 1]
            row["ratio"] = row["change"] / (crit[2] - crit[1]) if crit[2] != crit[1] else \
                float("nan")
        if J > 1 and J + 1 in dist and dist[J + 1] > 0:
            row["dist_ratio"] = dist[J] / dist[J + 1]
        rows.append(row)
    if top < 2:
        return 1, rows
    dB = {J: crit[J] - crit[J + 1] for J in range(1, top)}
    if dB[1] < 0:
        return 1, rows
    k = None
    for J in range(1, top):
        if dB[J] / dB[1] < 0.04:
            k = J
            break
    if k is None:
        k = top
    k = max(k, 2)
    R2 = {}
    for J in range(k, 1, -1):
        if J + 1 in dist and dist[J + 1] > 0:
            R2[J] = dist[J] / dist[J + 1]
        elif J + 1 not in dist:      # C_k holds every sub-cluster: nothing finer to compare
            R2[J] = math.inf
    if not R2:
        return k, rows
    order = sorted(R2, key=lambda J: -R2[J])
    if len(order) == 1:
        return order[0], rows
    a, b = order[0], order[1]
    if R2[a] > 1.15 * R2[b]:
        return a, rows
    return max(a, b), rows


# ============================================================================ quality
def ibm_silhouette(Xc: np.ndarray, Xcat: np.ndarray, ncat: list[int],
                   labels: np.ndarray) -> tuple[float, np.ndarray]:
    """IBM Cluster Evaluation silhouette (distances to centroids)."""
    parts = []
    if Xc.size:
        lo, hi = Xc.min(0), Xc.max(0)
        rng_ = np.where(hi > lo, hi - lo, 1.0)
        parts.append(2 * (Xc - lo) / rng_ - 1)
    for k, L in enumerate(ncat):
        Z = np.zeros((len(labels), L))
        Z[np.arange(len(labels)), Xcat[:, k].astype(int)] = 1.0 / math.sqrt(2)
        parts.append(Z)
    Z = np.column_stack(parts)
    ks = sorted(set(labels.tolist()))
    if len(ks) < 2:
        return float("nan"), np.full(len(labels), np.nan)
    cent = np.array([Z[labels == k].mean(0) for k in ks])
    d = np.sqrt(((Z[:, None, :] - cent[None, :, :]) ** 2).sum(2))
    pos = np.searchsorted(ks, labels)
    A = d[np.arange(len(labels)), pos]
    d[np.arange(len(labels)), pos] = np.inf
    B = d.min(1)
    mx = np.maximum(A, B)
    s = np.where(mx > 0, (B - A) / np.where(mx > 0, mx, 1), np.nan)
    return float(np.nanmean(s)), s


def predictor_importance(Xc, Xcat, labels, names_c, names_cat) -> tuple[list, np.ndarray]:
    names, logp = [], []
    ks = sorted(set(labels.tolist()))
    for k, name in enumerate(names_c):
        groups = [Xc[labels == c, k] for c in ks]
        if len(groups) < 2:
            continue
        F, p = stats.f_oneway(*groups)
        lp = stats.f.logsf(F, len(groups) - 1, len(labels) - len(groups)) \
            if np.isfinite(F) else -745.0
        names.append(name)
        logp.append(-lp / math.log(10))
    for k, name in enumerate(names_cat):
        tab = np.zeros((len(ks), int(Xcat[:, k].max()) + 1))
        np.add.at(tab, (np.searchsorted(ks, labels), Xcat[:, k].astype(int)), 1)
        tab = tab[:, tab.sum(0) > 0]
        if tab.shape[1] < 2:
            names.append(name)
            logp.append(0.0)
            continue
        chi2, _, df, _ = stats.chi2_contingency(tab, correction=False)
        names.append(name)
        logp.append(-stats.chi2.logsf(chi2, df) / math.log(10))
    v = np.array(logp, float)
    v = np.where(np.isfinite(v), v, 745 / math.log(10))
    return names, v


# ============================================================================ procedure
@dataclass
class TwoStepModel:
    space: CFSpace
    centres_c: np.ndarray
    centres_cat: list
    k: int
    labels: np.ndarray
    sub_labels: np.ndarray = field(default=None)


def twostep(Xc: np.ndarray, Xcat: np.ndarray, ncat: list[int], measure: str = "ll",
            k: int | None = None, max_k: int = 15, criterion: str = "bic",
            branch: int = 8, depth: int = 3, threshold: float = 0.0,
            max_entries: int = 600, outliers: bool = False, outlier_frac: float = 0.25):
    """Core TwoStep on prepared data (continuous already standardised if wanted).
    Returns a dict with labels (cluster 1..k, -1 = outlier), k, auto rows and pieces."""
    n = len(Xc) if Xc.size else len(Xcat)
    kc = Xc.shape[1] if Xc.size else 0
    var = Xc.var(0) if kc else np.zeros(0)
    var = np.where(var > 0, var, 1.0)
    space = CFSpace(kc, list(ncat), var, measure)
    if measure == "euclidean" and ncat:
        raise MLError("The Euclidean distance needs continuous variables only.")
    Xc2 = Xc if kc else np.zeros((n, 0))
    Xcat2 = Xcat if len(ncat) else np.zeros((n, 0), int)
    ents, members, out_ids, tree = precluster(space, Xc2, Xcat2, branch, depth, threshold,
                                             max_entries, outliers, outlier_frac)
    m = len(ents)
    max_k = max(1, min(max_k, m))
    h = agglomerate(space, ents, max_k)
    llspace = CFSpace(kc, list(ncat), var, "ll")
    if measure == "euclidean":
        hll = Hierarchy(h.merges, h.labels, {J: float(xi(llspace, *_cluster_cfs(
            ents, h.labels[J], J)).sum()) for J in h.labels})
    else:
        hll = h
    auto_rows: list = []
    if k is None:
        k, auto_rows = auto_clusters(hll, space, n, max_k, criterion)
        if measure == "euclidean":
            _, auto_rows = auto_clusters(hll, space, n, max_k, criterion)
    k = int(max(1, min(k, m)))
    sub_lab = h.labels[k] if k in h.labels else np.zeros(m, int)
    cl = _cluster_cfs(ents, sub_lab, k)
    # final assignment of each record to the closest cluster
    labels = np.full(n, -1, int)
    off = space.offsets
    Ncl, Scl, Qcl, Ccl = cl
    noise_crit = None
    if outliers:
        V = 1.0
        if kc:
            V *= float(np.prod(np.maximum(Xc2.max(0) - Xc2.min(0), 1e-12)))
        for L in ncat:
            V *= L
        noise_crit = math.log(V) if V > 0 else math.inf
    for r in range(n):
        C = np.zeros(int(sum(ncat)))
        for kk in range(len(ncat)):
            C[off[kk] + int(Xcat2[r, kk])] = 1.0
        cf = (1.0, Xc2[r], Xc2[r] ** 2, C)
        d = distance(space, cf, cl)
        j = int(np.argmin(d))
        if outliers and measure == "ll" and d[j] > noise_crit:
            labels[r] = -1
        elif outliers and measure == "euclidean" and r in set(out_ids):
            labels[r] = -1
        else:
            labels[r] = j + 1
    return {"labels": labels, "k": k, "auto": auto_rows, "space": space, "tree": tree,
            "n_sub": m, "criterion": criterion, "hier": h}


def _cluster_cfs(ents: _Entries, lab: np.ndarray, k: int):
    N = np.array([ents.N[lab == j].sum() for j in range(k)])
    S = np.array([ents.S[lab == j].sum(0) for j in range(k)])
    Q = np.array([ents.Q[lab == j].sum(0) for j in range(k)])
    C = np.array([ents.C[lab == j].sum(0) for j in range(k)])
    return N, S, Q, C


def twostep_procedure(ds: Dataset, categorical_vars: list[str], continuous_vars: list[str],
                      measure: str = "ll", auto: bool = True, max_k: int = 15,
                      fixed_k: int = 3, criterion: str = "bic", standardize: bool = True,
                      outliers: bool = False, outlier_pct: float = 25.0, branch: int = 8,
                      depth: int = 3, threshold: float = 0.0,
                      output: dict | None = None, save: bool = False) -> MLResult:
    out = {"pivot": True, "charts": True, "evaluation": True, **(output or {})}
    names = list(categorical_vars) + list(continuous_vars)
    if not names:
        raise MLError("Select at least one variable.")
    if measure == "euclidean" and categorical_vars:
        raise MLError("The Euclidean distance measure can be used only when all variables are "
                      "continuous.")
    ok = np.ones(ds.n_cases, bool)
    for nm in names:
        ok &= raw_column(ds, nm).notna().to_numpy()
    rows = np.where(ok)[0]
    if len(rows) < 2:
        raise MLError("Too few valid cases.")
    cats = [categorical(ds, nm, rows) for nm in categorical_vars]
    Xcat = np.column_stack([c.codes for c in cats]) if cats else np.zeros((len(rows), 0), int)
    ncat = [len(c.levels) for c in cats]
    Xraw = np.column_stack([raw_column(ds, nm).to_numpy(float)[rows] for nm in
                            continuous_vars]) if continuous_vars else np.zeros((len(rows), 0))
    Xc = Xraw.copy()
    if standardize and Xc.size:
        sd = Xc.std(0, ddof=1)
        Xc = (Xc - Xc.mean(0)) / np.where(sd > 0, sd, 1)
    res = twostep(Xc, Xcat, ncat, measure, None if auto else int(fixed_k), int(max_k),
                  criterion, branch, depth, threshold, outliers=outliers,
                  outlier_frac=outlier_pct / 100)
    labels, k = res["labels"], res["k"]
    section = Section("TwoStep Cluster")
    numc = f"AUTO {max_k} {criterion.upper()}" if auto else f"FIXED={fixed_k}"
    section.log(f"TWOSTEP CLUSTER\n  /CATEGORICAL VARIABLES={' '.join(categorical_vars)}\n"
                f"  /CONTINUOUS VARIABLES={' '.join(continuous_vars)}\n"
                f"  /DISTANCE {'LIKELIHOOD' if measure == 'll' else 'EUCLIDEAN'}\n"
                f"  /NUMCLUSTERS {numc}\n"
                f"  /HANDLENOISE {outlier_pct:g}" * bool(outliers) + "\n  /CRITERIA "
                f"INITTHRESHOLD({threshold:g}) MXBRANCH({branch}) MXLEVEL({depth}).")
    if auto and res["auto"]:
        crit_name = "Schwarz's Bayesian Criterion (BIC)" if criterion == "bic" else \
            "Akaike's Information Criterion (AIC)"
        short = "BIC" if criterion == "bic" else "AIC"
        rows_t = [[str(r["J"]), fmt(r["crit"], 3), fmt(r.get("change"), 3),
                   fmt(r.get("ratio"), 3), fmt(r.get("dist_ratio"), 3)] for r in res["auto"]]
        section.table(Table("Auto-Clustering", ["Number of Clusters", crit_name,
                                                f"{short} Changeᵃ",
                                                f"Ratio of {short} Changesᵇ",
                                                "Ratio of Distance Measuresᶜ"], rows_t,
                            footnotes=["a. The changes are from the previous number of "
                                       "clusters in the table.",
                                       "b. The ratios of changes are relative to the change "
                                       "for the two cluster solution.",
                                       "c. The ratios of distance measures are based on the "
                                       "current number of clusters against the previous "
                                       "number of clusters."]))
    n = len(rows)
    excluded = ds.n_cases - n
    rows_d = []
    valid = labels > 0
    for j in range(1, k + 1):
        c = int((labels == j).sum())
        rows_d.append(["Cluster", str(j), fmt_int(c), f"{100 * c / max(valid.sum(), 1):.1f}%",
                       f"{100 * c / ds.n_cases:.1f}%"])
    if (labels == -1).any():
        c = int((labels == -1).sum())
        rows_d.append(["", "Outlier (-1)", fmt_int(c), f"{100 * c / n:.1f}%",
                       f"{100 * c / ds.n_cases:.1f}%"])
    rows_d.append(["", "Combined", fmt_int(n), "100.0%", f"{100 * n / ds.n_cases:.1f}%"])
    rows_d.append(["Excluded Cases", "", fmt_int(excluded), "",
                   f"{100 * excluded / ds.n_cases:.1f}%"])
    rows_d.append(["Total", "", fmt_int(ds.n_cases), "", "100.0%"])
    section.table(Table("Cluster Distribution", ["", "", "N", "% of Combined", "% of Total"],
                        rows_d, row_header_cols=2))
    if continuous_vars:
        cols = [""] + [(nm, h) for nm in continuous_vars for h in ("Mean", "Std. Deviation")]
        rows_c = []
        for j in list(range(1, k + 1)) + ([-1] if (labels == -1).any() else []) + [0]:
            sel = valid if j == 0 else labels == j
            lab = "Combined" if j == 0 else ("Outlier (-1)" if j == -1 else str(j))
            r = [lab]
            for v in range(Xraw.shape[1]):
                x = Xraw[sel, v]
                r += [fmt(x.mean(), 4) if len(x) else "",
                      fmt(x.std(ddof=1), 4) if len(x) > 1 else ""]
            rows_c.append(r)
        section.table(Table("Centroids", cols, rows_c))
    for c, nm in zip(cats, categorical_vars):
        cols = [""] + [(lab, h) for lab in c.labels for h in ("Frequency", "Percent")]
        rows_f = []
        for j in list(range(1, k + 1)) + [0]:
            sel = valid if j == 0 else labels == j
            r = ["Combined" if j == 0 else str(j)]
            for li in range(len(c.levels)):
                f = int(((c.codes == li) & sel).sum())
                tot = int(((c.codes == li) & valid).sum())
                r += [fmt_int(f), f"{100 * f / tot:.1f}%" if tot else ""]
            rows_f.append(r)
        section.table(Table(nm, cols, rows_f, footnotes=["Percent within each category."]))
    sil = float("nan")
    if k >= 2 and out.get("evaluation"):
        sil, _ = ibm_silhouette(Xraw[valid] if Xraw.size else np.zeros((valid.sum(), 0)),
                                Xcat[valid] if Xcat.size else np.zeros((valid.sum(), 0)),
                                ncat, labels[valid])
        section.table(Table("Model Summary", ["", ""], [
            ["Algorithm", "TwoStep"], ["Inputs", fmt_int(len(names))],
            ["Clusters", fmt_int(k)],
            ["Silhouette measure of cohesion and separation", fmt(sil, 3)],
            ["Cluster Quality", "Good" if sil > 0.5 else "Fair" if sil > 0.2 else "Poor"]]))
        if out.get("charts"):
            section.image("Cluster Quality", _quality_chart(sil), width=460)
            sizes = [int((labels == j).sum()) for j in range(1, k + 1)]
            section.image("Cluster Sizes", _pie(sizes), width=420)
            pn, pv = predictor_importance(Xraw[valid] if Xraw.size else np.zeros((0, 0)),
                                          Xcat[valid] if Xcat.size else np.zeros((0, 0)),
                                          labels[valid], continuous_vars, categorical_vars)
            if len(pv) and pv.max() > 0:
                section.table(Table("Predictor Importance", ["", "Importance"], [
                    [nm, fmt(v / pv.max(), 3)] for nm, v in zip(pn, pv)]))
                section.image("Predictor Importance", importance_chart(
                    pn, pv, "Predictor Importance", "Importance"))
    saved = {}
    if save:
        saved["TSC"] = ("Two Step Cluster Number", full_series(ds, rows, labels.astype(float)))
    return MLResult(section, saved, res)


def _quality_chart(sil: float) -> bytes:
    fig = figure(5.6, 1.9)
    ax = fig.add_subplot(111)
    for lo, hi, col, lab in ((-1, 0.2, "#f4c7c3", "Poor"), (0.2, 0.5, "#fce8b2", "Fair"),
                             (0.5, 1.0, "#b7e1cd", "Good")):
        ax.barh(0, hi - lo, left=lo, color=col, height=0.8)
        ax.text((lo + hi) / 2, 0.25, lab, ha="center", va="center", fontsize=8)
    ax.barh(0, (sil if np.isfinite(sil) else 0) + 1, left=-1, height=0.3, color="#4f81bd")
    ax.set_xlim(-1, 1)
    ax.set_yticks([])
    style(ax, "Cluster Quality", "Silhouette measure of cohesion and separation", "")
    return png(fig)


def _pie(sizes: list[int]) -> bytes:
    fig = figure(4.6, 4.0)
    ax = fig.add_subplot(111)
    ax.pie(sizes, labels=[f"{i + 1}" for i in range(len(sizes))], autopct="%1.1f%%",
           textprops={"fontsize": 8})
    ax.set_title("Cluster Sizes", fontsize=11, fontweight="bold")
    return png(fig)


# ============================================================================ silhouettes
def pairwise(X: np.ndarray, metric: str = "euclidean", ranges=None, nominal=None) -> np.ndarray:
    if metric == "euclidean":
        sq = (X ** 2).sum(1)
        return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0))
    if metric in ("cityblock", "manhattan"):
        return np.abs(X[:, None, :] - X[None, :, :]).sum(2)
    if metric == "gower":
        nominal = np.zeros(X.shape[1], bool) if nominal is None else np.asarray(nominal)
        rg = np.where(np.asarray(ranges) > 0, ranges, 1.0)
        D = np.zeros((len(X), len(X)))
        for k in range(X.shape[1]):
            if nominal[k]:
                D += (X[:, None, k] != X[None, :, k])
            else:
                D += np.abs(X[:, None, k] - X[None, :, k]) / rg[k]
        return D / X.shape[1]
    raise MLError(f"Unknown distance {metric!r}.")


def silhouette_samples(D: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kaufman-Rousseeuw silhouette and the nearest other cluster of each case."""
    ks = sorted(set(labels.tolist()))
    n = len(labels)
    pos = np.searchsorted(ks, labels)
    M = np.zeros((n, len(ks)))
    sizes = np.array([(labels == k).sum() for k in ks], float)
    for j, k in enumerate(ks):
        M[:, j] = D[:, labels == k].sum(1)
    own = sizes[pos] - 1
    a = np.where(own > 0, M[np.arange(n), pos] / np.where(own > 0, own, 1), 0.0)
    other = M / sizes[None, :]
    other[np.arange(n), pos] = np.inf
    nb = other.argmin(1)
    b = other[np.arange(n), nb]
    mx = np.maximum(a, b)
    s = np.where((own > 0) & (mx > 0), (b - a) / np.where(mx > 0, mx, 1), 0.0)
    return s, np.array(ks)[nb]


def silhouettes_procedure(ds: Dataset, cluster_var: str, variables: list[str],
                          metric: str = "euclidean", standardize: bool = False,
                          plot: bool = True, save: bool = False) -> MLResult:
    names = [cluster_var] + list(variables)
    ok = np.ones(ds.n_cases, bool)
    for nm in names:
        ok &= raw_column(ds, nm).notna().to_numpy()
    rows = np.where(ok)[0]
    if len(rows) < 3:
        raise MLError("Too few valid cases.")
    cl = categorical(ds, cluster_var, rows)
    labels = cl.codes
    if len(cl.levels) < 2:
        raise MLError("The cluster variable needs at least two clusters.")
    nominal, cols = [], []
    for nm in variables:
        if metric == "gower" and ds.var(nm).measure == "nominal":
            cols.append(categorical(ds, nm, rows).codes.astype(float))
            nominal.append(True)
        else:
            x = raw_column(ds, nm).to_numpy(float)[rows]
            if ds.var(nm).type != "numeric":
                raise MLError(f"{nm} is not numeric.")
            cols.append(x)
            nominal.append(False)
    X = np.column_stack(cols)
    if standardize:
        cont = ~np.array(nominal)
        sd = X[:, cont].std(0, ddof=1)
        X[:, cont] = (X[:, cont] - X[:, cont].mean(0)) / np.where(sd > 0, sd, 1)
    D = pairwise(X, metric, X.max(0) - X.min(0), nominal)
    s, nb = silhouette_samples(D, labels)
    section = Section("Cluster Silhouettes")
    section.log(f"STATS CLUS SIL CLUSTER={cluster_var} VARIABLES={' '.join(variables)}\n"
                f"  DISSIMILARITY={metric.upper()}.")
    rows_t = []
    for j, lab in enumerate(cl.labels):
        sel = labels == j
        rows_t.append([lab, fmt_int(sel.sum()), fmt(s[sel].mean(), 4), fmt(s[sel].min(), 4),
                       fmt(s[sel].max(), 4)])
    rows_t.append(["Overall", fmt_int(len(s)), fmt(s.mean(), 4), fmt(s.min(), 4),
                   fmt(s.max(), 4)])
    section.table(Table("Silhouette Statistics by Cluster", [cluster_var, "N", "Mean",
                                                             "Minimum", "Maximum"], rows_t,
                        footnotes=[f"Dissimilarity: {metric}. Silhouette s = (b - a) / "
                                   "max(a, b) (Kaufman & Rousseeuw 1990)."]))
    if plot:
        fig = figure(5.6, max(3.2, 0.012 * len(s) + 2))
        ax = fig.add_subplot(111)
        y0 = 0
        for j, lab in enumerate(cl.labels):
            v = np.sort(s[labels == j])[::-1]
            ax.barh(np.arange(y0, y0 + len(v)), v, height=1.0, color=f"C{j}",
                    label=f"{lab} (mean {v.mean():.2f})")
            y0 += len(v) + 3
        ax.axvline(s.mean(), color="#c00", ls="--", lw=0.8)
        ax.set_yticks([])
        ax.legend(fontsize=7, title=cluster_var)
        style(ax, "Silhouette Plot", "Silhouette", "Cases by cluster")
        section.image("Silhouette Plot", png(fig))
    saved = {}
    if save:
        saved["Silhouette"] = ("Silhouette value", full_series(ds, rows, s))
        nbv = np.array([cl.levels[int(k)] if not isinstance(cl.levels[int(k)], str)
                        else int(k) + 1 for k in nb], float)
        saved["NextCluster"] = ("Nearest other cluster", full_series(ds, rows, nbv))
    return MLResult(section, saved, s)


def selftest() -> None:
    r = np.random.default_rng(3)
    X = np.vstack([r.normal(m, 0.4, (60, 2)) for m in (0, 3, 6)])
    res = twostep(X, np.zeros((180, 0), int), [], "ll")
    assert res["k"] == 3, res["k"]
    D = pairwise(X)
    s, _ = silhouette_samples(D, res["labels"])
    assert s.mean() > 0.6

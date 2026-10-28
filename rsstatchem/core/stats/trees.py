"""Analyze > Classify > Tree (SPSS Decision Trees, TREE procedure): CHAID, Exhaustive CHAID,
CRT and QUEST, following the IBM SPSS Statistics Algorithms chapter "TREE Algorithms".

* **CHAID** (Kass 1980) / **Exhaustive CHAID** (Biggs et al. 1991): scale predictors are
  binned into (default) 10 ordinal categories with IBM's rank-based break points; categories
  are merged by the largest pairwise p-value above alpha-merge (adjacent pairs only for ordinal
  predictors; optional re-splitting of compound categories), missing values form a "floating"
  category for ordinal predictors; p-values come from Pearson / likelihood-ratio chi-square
  (nominal target), Goodman's row-effects likelihood ratio (ordinal target) or the ANOVA F
  test (scale target), adjusted with Kass's Bonferroni multipliers. Exhaustive CHAID merges
  down to two categories and keeps the set with the smallest p-value.
* **CRT** (Breiman et al. 1984): binary splits maximising the Gini, twoing or ordered-twoing
  decrease (least-squares deviation for a scale target), priors and misclassification costs,
  surrogate splits for missing values and variable importance (primary + surrogate
  improvements), optional cost-complexity pruning with the b-SE rule.
* **QUEST** (Loh & Shih 1997): predictor selection by ANOVA F / Pearson chi-square with
  Bonferroni alpha/M and Levene fallback; split points from quadratic discriminant analysis on
  two super-classes (2-means on class means); nominal predictors are mapped once to their
  largest discriminant coordinate.

Assignment, resubstitution / test-sample / V-fold cross-validated risk, the gain summary
(node-by-node and cumulative, response and index) and cost-complexity pruning follow the
"Assignment and Risk Estimation", "Gain Summary" and "Cost-Complexity Pruning" sections.
The tree diagram is drawn as an image like the SPSS tree editor (node tables with category
percentages and n, split statistics above the branches).
"""
from __future__ import annotations

import io
import itertools
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from scipy.special import gammaln

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int
from rsstatchem.core.stats.ml_common import (MLError, MLResult, categorical,
                                             classification_rows, full_series, importance_chart,
                                             importance_table, measure_of,
                                             raw_column, rng)
from rsstatchem.core.stats.roc import figure, png, style

METHODS = {"chaid": "CHAID", "exhaustive": "Exhaustive CHAID", "crt": "CRT", "quest": "QUEST"}
DEFAULT_DEPTH = {"chaid": 3, "exhaustive": 3, "crt": 5, "quest": 5}


# ============================================================================ data model
@dataclass
class Predictor:
    name: str
    kind: str                 # nominal | ordinal | scale (as used by the growing method)
    x: np.ndarray             # float; category code for nominal/ordinal; NaN = missing
    labels: list[str] = field(default_factory=list)   # category labels (nominal/ordinal)
    orig_kind: str = ""
    breaks: np.ndarray | None = None   # CHAID bins of a scale predictor
    xi: np.ndarray | None = None       # QUEST discriminant coordinate per category


@dataclass
class Split:
    var: int
    kind: str                 # "sets" (CHAID multiway) | "le" | "in"
    groups: list = field(default_factory=list)   # sets: list of code lists (-1 = missing)
    value: float = float("nan")                  # le: x <= value goes left
    left: frozenset = frozenset()                # in: codes going left
    stat: dict = field(default_factory=dict)
    surrogates: list = field(default_factory=list)
    majority: int = 0
    default: int = 0          # child for unseen categories (CHAID)


@dataclass
class Node:
    id: int
    parent: int | None
    depth: int
    idx: np.ndarray           # training cases (indices into model arrays)
    children: list[int] = field(default_factory=list)
    split: Split | None = None
    branch: str = ""          # text on the edge from the parent
    counts: np.ndarray | None = None
    mean: float = float("nan")
    sd: float = float("nan")
    pred: object = None
    probs: np.ndarray | None = None

    @property
    def terminal(self) -> bool:
        return not self.children


@dataclass
class TreeModel:
    method: str
    dep: str
    ytype: str                 # categorical | scale
    y: np.ndarray              # codes (categorical) or values
    ylabels: list[str]
    yordinal: bool
    predictors: list[Predictor]
    nodes: dict[int, Node]
    priors: np.ndarray | None
    costs: np.ndarray | None
    settings: dict

    @property
    def J(self) -> int:
        return len(self.ylabels)

    def terminals(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.terminal]

    def depth(self) -> int:
        return max(n.depth for n in self.nodes.values())

    def used_vars(self) -> list[str]:
        seen = []
        for n in self.nodes.values():
            if n.split is not None:
                name = self.predictors[n.split.var].name
                if name not in seen:
                    seen.append(name)
        return seen

    # ------------------------------------------------------------------ scoring
    def route(self, node: Node, rows: np.ndarray, X: list[np.ndarray]) -> np.ndarray:
        """Child position (0..k-1) of each case in ``rows`` at ``node``."""
        sp = node.split
        out = _apply_rule(sp, X[sp.var][rows], self.predictors[sp.var])
        if sp.kind != "sets":
            miss = out < 0
            for sur in sp.surrogates:
                if not miss.any():
                    break
                v = _apply_rule(sur["split"], X[sur["split"].var][rows[miss]],
                                self.predictors[sur["split"].var])
                if sur.get("reverse"):
                    v = np.where(v >= 0, 1 - v, v)
                sub = np.where(miss)[0]
                ok = v >= 0
                out[sub[ok]] = v[ok]
                miss = out < 0
            out[out < 0] = sp.majority
        else:
            out[out < 0] = sp.default
        return out

    def assign(self, X: list[np.ndarray] | None = None, n: int | None = None) -> np.ndarray:
        """Terminal node id of each case (training data when X is None)."""
        if X is None:
            X = [p.x for p in self.predictors]
        n = len(X[0]) if n is None else n
        node_of = np.full(n, 1, int)
        stack = [1]
        while stack:
            nid = stack.pop()
            node = self.nodes[nid]
            if node.terminal:
                continue
            rows = np.where(node_of == nid)[0]
            pos = self.route(node, rows, X)
            for k, cid in enumerate(node.children):
                node_of[rows[pos == k]] = cid
                stack.append(cid)
        return node_of


def _apply_rule(sp: Split, x: np.ndarray, pred: Predictor) -> np.ndarray:
    out = np.full(len(x), -1, int)
    ok = ~np.isnan(x)
    if sp.kind == "sets":
        code = np.where(ok, x, -1).astype(int)
        for k, g in enumerate(sp.groups):
            out[np.isin(code, g)] = k
        return out
    if sp.kind == "le":
        out[ok] = np.where(x[ok] <= sp.value, 0, 1)
        return out
    codes = x[ok].astype(int)
    out[ok] = np.where(np.isin(codes, list(sp.left)), 0, 1)
    return out


# ============================================================================ statistics
def _chi2_logp(table: np.ndarray, lr: bool = False) -> tuple[float, float, int]:
    t = table[table.sum(1) > 0][:, table.sum(0) > 0]
    if t.shape[0] < 2 or t.shape[1] < 2:
        return 0.0, 0.0, 0
    n = t.sum()
    e = np.outer(t.sum(1), t.sum(0)) / n
    if lr:
        with np.errstate(divide="ignore", invalid="ignore"):
            stat = 2 * float(np.nansum(np.where(t > 0, t * np.log(t / e), 0.0)))
    else:
        stat = float(((t - e) ** 2 / e).sum())
    df = (t.shape[0] - 1) * (t.shape[1] - 1)
    return stat, float(stats.chi2.logsf(stat, df)), df


def _row_effects_logp(table: np.ndarray, scores: np.ndarray) -> tuple[float, float, int]:
    """Goodman's (1979) row-effects model against independence (ordinal target)."""
    keep_r, keep_c = table.sum(1) > 0, table.sum(0) > 0
    t = table[keep_r][:, keep_c]
    s = scores[keep_c]
    I, J = t.shape
    if I < 2 or J < 2:
        return 0.0, 0.0, 0
    n = t.sum()
    ind = np.outer(t.sum(1), t.sum(0)) / n
    y = t.ravel()
    sc = s - s.mean()
    cols = [np.ones(I * J)]
    for i in range(1, I):
        cols.append(np.repeat(np.eye(I)[i], J))
    for j in range(1, J):
        cols.append(np.tile(np.eye(J)[j], I))
    for i in range(1, I):
        cols.append(np.repeat(np.eye(I)[i], J) * np.tile(sc, I))
    D = np.column_stack(cols)
    beta = np.linalg.lstsq(D, np.log(ind.ravel()), rcond=None)[0]
    for _ in range(100):
        mu = np.exp(np.clip(D @ beta, -700, 700))
        grad = D.T @ (y - mu)
        H = D.T @ (D * mu[:, None])
        step = np.linalg.lstsq(H, grad, rcond=None)[0]
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    m = np.exp(np.clip(D @ beta, -700, 700)).reshape(I, J)
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = 2 * float(np.nansum(np.where(m > 0, m * np.log(m / ind), 0.0)))
    stat = max(stat, 0.0)
    return stat, float(stats.chi2.logsf(stat, I - 1)), I - 1


def _anova_logp(stats_rows: np.ndarray) -> tuple[float, float, tuple[int, int]]:
    """stats_rows: (n, sum, sumsq) per group."""
    r = stats_rows[stats_rows[:, 0] > 0]
    k = len(r)
    N = r[:, 0].sum()
    if k < 2 or N - k <= 0:
        return 0.0, 0.0, (max(k - 1, 0), int(max(N - k, 0)))
    gm = r[:, 1].sum() / N
    ssb = float((r[:, 1] ** 2 / r[:, 0]).sum() - N * gm ** 2)
    ssw = float((r[:, 2] - r[:, 1] ** 2 / r[:, 0]).sum())
    df1, df2 = k - 1, int(N - k)
    if ssw <= 1e-300 * max(1.0, abs(ssb)):
        return (math.inf, -math.inf, (df1, df2)) if ssb > 1e-12 else (0.0, 0.0, (df1, df2))
    F = max(ssb, 0.0) / df1 / (ssw / df2)
    return float(F), float(stats.f.logsf(F, df1, df2)), (df1, df2)


def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -math.inf
    return float(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1))


def bonferroni(I: int, r: int, kind: str, floating: bool = False,
               exhaustive: bool = False) -> float:
    """Kass's multipliers (CHAID) or the summed merge counts (Exhaustive CHAID)."""
    if r >= I or I <= 1:
        return 1.0
    if exhaustive:
        if kind == "nominal":
            return float(sum(v * (v - 1) // 2 for v in range(2, I + 1)))
        return float(I * (I - 1) // 2) + (1.0 if floating else 0.0)
    if kind == "nominal":
        tot = 0.0
        for v in range(r):
            tot += (-1) ** v * (r - v) ** I / (math.factorial(v) * math.factorial(r - v))
        return max(tot, 1.0)
    if floating:
        return math.exp(_log_comb(I - 2, r - 2)) + r * math.exp(_log_comb(I - 2, r - 1))
    return math.exp(_log_comb(I - 1, r - 1))


# ============================================================================ CHAID
class _ChaidTester:
    def __init__(self, model: TreeModel, settings: dict):
        self.m = model
        self.lr = settings.get("chisq", "pearson") == "lr"
        self.scores = np.arange(model.J, dtype=float) + 1 if model.ytype == "categorical" \
            else None

    def group_stats(self, codes: np.ndarray, y: np.ndarray, levels: list[int]) -> dict:
        out = {}
        for c in levels:
            sel = codes == c
            if self.m.ytype == "categorical":
                out[c] = np.bincount(y[sel].astype(int), minlength=self.m.J).astype(float)
            else:
                v = y[sel]
                out[c] = np.array([len(v), v.sum(), (v ** 2).sum()])
        return out

    def test(self, rows: list[np.ndarray]) -> tuple[float, float, object]:
        T = np.array(rows)
        if self.m.ytype == "scale":
            F, lp, df = _anova_logp(T)
            return F, lp, df
        if self.m.yordinal:
            return _row_effects_logp(T, self.scores)
        return _chi2_logp(T, self.lr)


def _bin_scale(x: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    ok = ~np.isnan(x)
    v = x[ok]
    if len(v) == 0:
        return np.full(len(x), np.nan), np.array([])
    r = stats.rankdata(v)
    N = len(v)
    k = np.floor(r * K / (N + 1)).astype(int)
    order = np.argsort(v, kind="mergesort")
    breaks = []
    for b in range(K):
        sel = k[order] == b
        if sel.any():
            breaks.append(v[order][sel].max())
    breaks = np.unique(np.array(breaks[:-1]))
    codes = np.full(len(x), np.nan)
    codes[ok] = np.searchsorted(breaks, v, side="left")
    return codes, breaks


def _bin_labels(breaks: np.ndarray) -> list[str]:
    if len(breaks) == 0:
        return ["all values"]
    f = lambda b: f"{b:.6g}"  # noqa: E731
    labels = [f"<= {f(breaks[0])}"]
    labels += [f"({f(a)}, {f(b)}]" for a, b in zip(breaks[:-1], breaks[1:])]
    labels.append(f"> {f(breaks[-1])}")
    return labels


def _chaid_merge(tester: _ChaidTester, gstats: dict, cats: list[int], kind: str,
                 has_missing: bool, s: dict, exhaustive: bool):
    """Returns (groups, adjusted log p, stat, df, raw log p)."""
    am = s.get("alpha_merge", 0.05)
    resplit = s.get("resplit", False)
    asm = s.get("alpha_split_merge", 0.05)

    def pstat(groups):
        return tester.test([sum(gstats[c] for c in g) for g in groups])

    def pair_p(g1, g2):
        return pstat([g1, g2])[1]

    def allowable(groups):
        pairs = []
        for i, j in itertools.combinations(range(len(groups)), 2):
            if kind == "ordinal" and j != i + 1:
                continue
            pairs.append((i, j))
        return pairs

    def best_binary_split(g):
        best = None
        if kind == "ordinal":
            cands = [(g[:k], g[k:]) for k in range(1, len(g))]
        else:
            cands = []
            rest = g[1:]
            for r in range(0, len(rest)):
                for comb in itertools.combinations(rest, r):
                    a = [g[0], *comb]
                    b = [c for c in g if c not in a]
                    if b:
                        cands.append((a, b))
        for a, b in cands:
            lp = pair_p(a, b)
            if best is None or lp < best[0]:
                best = (lp, a, b)
        return best

    groups = [[c] for c in cats]
    if not groups:
        return [], 0.0, 0.0, 0, 0.0
    stat, lp, df = 0.0, 0.0, 0
    if len(groups) >= 2 and exhaustive:
        history = [([list(g) for g in groups], pstat(groups))]
        while len(groups) > 2:
            pairs = allowable(groups)
            i, j = max(pairs, key=lambda ij: pair_p(groups[ij[0]], groups[ij[1]]))
            new = sorted(groups[i] + groups[j])
            groups = [g for k, g in enumerate(groups) if k not in (i, j)]
            groups.insert(i, new)
            history.append(([list(g) for g in groups], pstat(groups)))
        groups, (stat, lp, df) = min(history, key=lambda h: h[1][1])
    elif len(groups) >= 2:
        guard = 0
        while len(groups) > 2 and guard < 200:
            guard += 1
            pairs = allowable(groups)
            ps = [pair_p(groups[i], groups[j]) for i, j in pairs]
            k = int(np.argmax(ps))
            if math.exp(ps[k]) <= am:
                break
            i, j = pairs[k]
            new = sorted(groups[i] + groups[j])
            groups = [g for t, g in enumerate(groups) if t not in (i, j)]
            groups.insert(i, new)
            if resplit and len(new) >= 3:
                bs = best_binary_split(new)
                if bs is not None and math.exp(bs[0]) <= asm:
                    idx = groups.index(new)
                    groups[idx:idx + 1] = [sorted(bs[1]), sorted(bs[2])]
        stat, lp, df = pstat(groups)
    floating = False
    if has_missing:
        miss = [-1]
        if gstats.get(-1) is not None and gstats[-1].sum() > 0:
            if kind == "ordinal":
                floating = True
                sims = [pair_p(g, miss) for g in groups]
                k = int(np.argmax(sims))
                merged = [g + miss if t == k else g for t, g in enumerate(groups)]
                sep = groups + [miss]
                a, b = pstat(merged), pstat(sep)
                groups = merged if a[1] <= b[1] else sep
                stat, lp, df = a if a[1] <= b[1] else b
            else:
                pass
    I = len(cats) + (1 if floating else 0)
    B = bonferroni(I, len(groups), kind, floating, exhaustive) if s.get("bonferroni", True) \
        else 1.0
    return groups, lp + math.log(B), stat, df, lp


def _merge_small(tester, gstats, groups, sizes_fn, min_child, kind):
    groups = [list(g) for g in groups]
    while len(groups) > 1:
        sizes = [sizes_fn(g) for g in groups]
        small = [i for i, n in enumerate(sizes) if n < min_child]
        if not small:
            break
        i = min(small, key=lambda k: sizes[k])
        cands = [j for j in range(len(groups)) if j != i and
                 (kind != "ordinal" or abs(j - i) == 1 or -1 in groups[i] or -1 in groups[j])]
        if not cands:
            cands = [j for j in range(len(groups)) if j != i]
        j = max(cands, key=lambda k: tester.test([sum(gstats[c] for c in groups[i]),
                                                  sum(gstats[c] for c in groups[k])])[1])
        a, b = sorted((i, j))
        groups[a] = sorted(groups[a] + groups[b])
        del groups[b]
    return groups


def _grow_chaid(model: TreeModel, s: dict) -> None:
    tester = _ChaidTester(model, s)
    exhaustive = model.method == "exhaustive"
    queue = [1]
    while queue:
        nid = queue.pop(0)
        node = model.nodes[nid]
        idx = node.idx
        y = model.y[idx]
        if _stop_basic(model, node, s):
            continue
        best = None
        forced = s.get("force_first") and nid == 1
        for m, pr in enumerate(model.predictors[:1] if forced else model.predictors):
            x = pr.x[idx]
            codes = np.where(np.isnan(x), -1, x).astype(int)
            cats = sorted(set(codes[codes >= 0].tolist()))
            has_missing = bool((codes < 0).any())
            if len(cats) + has_missing < 2:
                continue
            levels = cats + ([-1] if has_missing else [])
            gstats = tester.group_stats(codes, y, levels)
            kind = "nominal" if pr.kind == "nominal" else "ordinal"
            if has_missing and kind == "nominal":
                cats_all = levels
            else:
                cats_all = cats
            groups, alp, stat, df, lp = _chaid_merge(tester, gstats, cats_all, kind,
                                                     has_missing and kind == "ordinal", s,
                                                     exhaustive)
            if has_missing and kind == "ordinal" and not any(-1 in g for g in groups):
                groups = groups + [[-1]]
            if len(groups) < 2:
                continue
            key = (alp, -stat)
            if best is None or key < best[0]:
                best = (key, m, groups, alp, stat, df, gstats, kind)
        if best is None:
            continue
        _, m, groups, alp, stat, df, gstats, kind = best
        if math.exp(alp) > s.get("alpha_split", 0.05) and not forced:
            continue
        x = model.predictors[m].x[idx]
        codes = np.where(np.isnan(x), -1, x).astype(int)

        def size(g):
            return int(np.isin(codes, g).sum())
        groups = _merge_small(tester, gstats, groups, size, s["min_child"], kind)
        if len(groups) < 2:
            continue
        sp = Split(m, "sets", groups=groups, stat={"logp": alp, "stat": stat, "df": df,
                                                    "test": "F" if model.ytype == "scale"
                                                    else ("lr" if (tester.lr or model.yordinal)
                                                          else "chi2")})
        sizes = [size(g) for g in groups]
        sp.default = int(np.argmax(sizes))
        _make_children(model, node, sp, codes_to_child(codes, groups))
        queue.extend(node.children)


def codes_to_child(codes: np.ndarray, groups: list) -> np.ndarray:
    out = np.full(len(codes), -1, int)
    for k, g in enumerate(groups):
        out[np.isin(codes, g)] = k
    return out


def _stop_basic(model: TreeModel, node: Node, s: dict) -> bool:
    idx = node.idx
    if node.depth >= s["max_depth"] or len(idx) < s["min_parent"] or len(idx) < 2:
        return True
    y = model.y[idx]
    if np.all(y == y[0]):
        return True
    for p in model.predictors:
        x = np.where(np.isnan(p.x[idx]), -np.inf, p.x[idx])
        if x.min() != x.max():
            return False
    return True


def _make_children(model: TreeModel, node: Node, sp: Split, pos: np.ndarray) -> None:
    node.split = sp
    nxt = max(model.nodes) + 1
    pr = model.predictors[sp.var]
    for k in range(int(pos.max()) + 1 if len(pos) else 0):
        child = Node(nxt, node.id, node.depth + 1, node.idx[pos == k])
        child.branch = _branch_text(sp, k, pr)
        model.nodes[nxt] = child
        node.children.append(nxt)
        nxt += 1
    for cid in node.children:
        _node_stats(model, model.nodes[cid])


def _branch_text(sp: Split, k: int, pr: Predictor) -> str:
    def lab(c):
        if c < 0:
            return "<missing>"
        return pr.labels[c] if c < len(pr.labels) else str(c)

    if sp.kind == "sets":
        g = sp.groups[k]
        if pr.orig_kind == "scale" and pr.breaks is not None:
            cs = [c for c in g if c >= 0]
            parts = []
            if cs:
                lo, hi = min(cs), max(cs)
                b = pr.breaks
                if lo == 0 and hi >= len(b):
                    parts.append("all values")
                elif lo == 0:
                    parts.append(f"<= {b[hi]:.6g}")
                elif hi >= len(b):
                    parts.append(f"> {b[lo - 1]:.6g}")
                else:
                    parts.append(f"({b[lo - 1]:.6g}, {b[hi]:.6g}]")
            if -1 in g:
                parts.append("<missing>")
            return "; ".join(parts)
        return "; ".join(lab(c) for c in g)
    if sp.kind == "le":
        if pr.kind == "ordinal" and pr.labels:
            cut = int(math.floor(sp.value))
            cats = range(0, cut + 1) if k == 0 else range(cut + 1, len(pr.labels))
            return "; ".join(lab(c) for c in cats)
        return (f"<= {sp.value:.6g}" if k == 0 else f"> {sp.value:.6g}")
    codes = sorted(sp.left) if k == 0 else sorted(set(range(len(pr.labels))) - set(sp.left))
    return "; ".join(lab(c) for c in codes)


# ============================================================================ node stats
def _probs(model: TreeModel, counts: np.ndarray) -> np.ndarray:
    if model.priors is None:
        tot = counts.sum()
        return counts / tot if tot else np.full(len(counts), 1 / len(counts))
    root = model.nodes[1].counts
    with np.errstate(divide="ignore", invalid="ignore"):
        pj = np.where(root > 0, model.priors * counts / root, 0.0)
    tot = pj.sum()
    return pj / tot if tot else np.full(len(counts), 1 / len(counts))


def _node_stats(model: TreeModel, node: Node) -> None:
    y = model.y[node.idx]
    if model.ytype == "categorical":
        node.counts = np.bincount(y.astype(int), minlength=model.J).astype(float)
        p = _probs(model, node.counts)
        node.probs = p
        if model.costs is not None:
            exp_cost = model.costs @ p      # costs[i, j] = C(i|j): predict i when truth j
            cand = np.where(np.isclose(exp_cost, exp_cost.min()))[0]
        else:
            cand = np.where(np.isclose(p, p.max()))[0]
        pos = [j for j in cand if node.counts[j] > 0]
        node.pred = int(pos[0] if pos else cand[0])
    else:
        node.mean = float(y.mean()) if len(y) else float("nan")
        node.sd = float(y.std(ddof=1)) if len(y) > 1 else float("nan")
        node.pred = node.mean


# ============================================================================ CRT
def _impurity_split_values(model: TreeModel, yl: np.ndarray, Ltot: np.ndarray,
                           Rtot: np.ndarray, s: dict, scale_stats=None) -> np.ndarray:
    """Improvement p(t)·Δi for every candidate (rows of class counts left/right)."""
    if model.ytype == "scale":
        nL, sL, qL, nR, sR, qR, N = scale_stats
        with np.errstate(divide="ignore", invalid="ignore"):
            sse_t = (qL + qR) - (sL + sR) ** 2 / (nL + nR)
            sse = (qL - sL ** 2 / nL) + (qR - sR ** 2 / nR)
        return np.where((nL > 0) & (nR > 0), (sse_t - sse) / N, -np.inf)
    root = model.nodes[1].counts
    pri = model.priors if model.priors is not None else root / root.sum()
    w = np.where(root > 0, pri / np.where(root > 0, root, 1), 0.0)
    pL = Ltot * w            # p(j, tL)
    pR = Rtot * w
    ptL, ptR = pL.sum(1), pR.sum(1)
    pt = ptL + ptR
    with np.errstate(divide="ignore", invalid="ignore"):
        cL = pL / ptL[:, None]
        cR = pR / ptR[:, None]
        ct = (pL + pR) / pt[:, None]
        crit = s.get("impurity", "gini")
        if crit == "twoing":
            val = ptL * ptR / pt ** 2 / 4 * np.abs(cL - cR).sum(1) ** 2 * pt
        elif crit == "ordered":
            cumL, cumR, cumT = np.cumsum(cL, 1)[:, :-1], np.cumsum(cR, 1)[:, :-1], \
                np.cumsum(ct, 1)[:, :-1]
            iT = cumT * (1 - cumT)
            iL = cumL * (1 - cumL)
            iR = cumR * (1 - cumR)
            d = iT - (ptL / pt)[:, None] * iL - (ptR / pt)[:, None] * iR
            val = d.max(1) * pt if d.shape[1] else np.zeros(len(pt))
        else:
            gi = lambda c: 1 - (c ** 2).sum(1)  # noqa: E731
            val = pt * (gi(ct) - ptL / pt * gi(cL) - ptR / pt * gi(cR))
    ok = (Ltot.sum(1) > 0) & (Rtot.sum(1) > 0)
    return np.where(ok, np.nan_to_num(val, nan=-np.inf), -np.inf)


def _best_split_predictor(model: TreeModel, idx: np.ndarray, m: int, s: dict,
                          min_child: int, forced: bool = False):
    """Best binary split of predictor m on node cases idx: (improvement, Split, left mask)."""
    pr = model.predictors[m]
    x = pr.x[idx]
    ok = ~np.isnan(x)
    if ok.sum() < 2:
        return None
    xv, yv = x[ok], model.y[idx][ok]
    N_root = len(model.y)
    if pr.kind == "nominal":
        codes = xv.astype(int)
        cats = sorted(set(codes.tolist()))
        if len(cats) < 2:
            return None
        if model.ytype == "categorical":
            C = np.array([np.bincount(yv[codes == c].astype(int), minlength=model.J)
                          for c in cats], float)
            if model.J == 2 or len(cats) > 10:
                p1 = C[:, 0] / np.maximum(C.sum(1), 1)
                order = np.argsort(p1, kind="mergesort")
                subsets = [frozenset(cats[i] for i in order[:k]) for k in range(1, len(cats))]
            else:
                subsets = []
                rest = cats[1:]
                for r in range(0, len(rest)):
                    for comb in itertools.combinations(rest, r):
                        subsets.append(frozenset([cats[0], *comb]))
            Ls = np.array([C[[cats.index(c) for c in S]].sum(0) for S in subsets])
            Rs = C.sum(0)[None, :] - Ls
            vals = _impurity_split_values(model, yv, Ls, Rs, s)
            nL, nR = Ls.sum(1), Rs.sum(1)
        else:
            stats_c = np.array([[np.sum(codes == c), yv[codes == c].sum(),
                                 (yv[codes == c] ** 2).sum()] for c in cats])
            order = np.argsort(stats_c[:, 1] / stats_c[:, 0], kind="mergesort")
            subsets = [frozenset(cats[i] for i in order[:k]) for k in range(1, len(cats))]
            L = np.array([stats_c[[cats.index(c) for c in S]].sum(0) for S in subsets])
            T = stats_c.sum(0)
            R = T[None, :] - L
            vals = _impurity_split_values(model, yv, None, None, s, (
                L[:, 0], L[:, 1], L[:, 2], R[:, 0], R[:, 1], R[:, 2], N_root))
            nL, nR = L[:, 0], R[:, 0]
        vals = np.where((nL >= min_child) & (nR >= min_child), vals, -np.inf)
        if not len(vals) or not np.isfinite(vals.max()):
            return None
        k = int(np.argmax(vals))
        sp = Split(m, "in", left=subsets[k])
        return float(vals[k]), sp
    order = np.argsort(xv, kind="mergesort")
    xs, ys = xv[order], yv[order]
    n = len(xs)
    cut = np.where(xs[1:] > xs[:-1])[0]          # split after position cut
    if not len(cut):
        return None
    nL = cut + 1
    nR = n - nL
    if model.ytype == "categorical":
        Y = np.zeros((n, model.J))
        Y[np.arange(n), ys.astype(int)] = 1
        cum = np.cumsum(Y, 0)
        Ls = cum[cut]
        Rs = cum[-1][None, :] - Ls
        vals = _impurity_split_values(model, ys, Ls, Rs, s)
    else:
        cs, cq = np.cumsum(ys), np.cumsum(ys ** 2)
        vals = _impurity_split_values(model, ys, None, None, s, (
            nL.astype(float), cs[cut], cq[cut], nR.astype(float), cs[-1] - cs[cut],
            cq[-1] - cq[cut], N_root))
    vals = np.where((nL >= min_child) & (nR >= min_child), vals, -np.inf)
    if not np.isfinite(vals.max()):
        return None
    k = int(np.argmax(vals))
    if pr.kind == "ordinal":
        value = float(xs[cut[k]])
    else:
        value = float((xs[cut[k]] + xs[cut[k] + 1]) / 2)
    return float(vals[k]), Split(m, "le", value=value)


def _surrogates(model: TreeModel, idx: np.ndarray, sp: Split, s: dict) -> list:
    pr0 = model.predictors[sp.var]
    side = _apply_rule(sp, pr0.x[idx], pr0)
    valid0 = side >= 0
    if valid0.sum() == 0:
        return []
    pL = float((side[valid0] == 0).mean())
    pR = 1 - pL
    out = []
    for m, pr in enumerate(model.predictors):
        if m == sp.var:
            continue
        x = pr.x[idx]
        both = valid0 & ~np.isnan(x)
        if both.sum() < 2:
            continue
        xv, sv = x[both], side[both]
        best = None
        if pr.kind == "nominal":
            codes = xv.astype(int)
            left = frozenset(c for c in set(codes.tolist())
                             if (sv[codes == c] == 0).sum() >= (sv[codes == c] == 1).sum())
            agree = float(np.mean(np.isin(codes, list(left)) == (sv == 0)))
            best = (agree, Split(m, "in", left=left), False)
        else:
            order = np.argsort(xv, kind="mergesort")
            xs, ss = xv[order], sv[order]
            cut = np.where(xs[1:] > xs[:-1])[0]
            if not len(cut):
                continue
            leftL = np.cumsum(ss == 0)[cut]
            rightR = (ss == 1).sum() - np.cumsum(ss == 1)[cut]
            a1 = (leftL + rightR) / len(ss)
            a2 = 1 - a1
            k1, k2 = int(np.argmax(a1)), int(np.argmax(a2))
            if a1[k1] >= a2[k2]:
                k, agree, rev = k1, float(a1[k1]), False
            else:
                k, agree, rev = k2, float(a2[k2]), True
            value = float(xs[cut[k]]) if pr.kind == "ordinal" else \
                float((xs[cut[k]] + xs[cut[k] + 1]) / 2)
            best = (agree, Split(m, "le", value=value), rev)
        mn = min(pL, pR)
        assoc = (mn - (1 - best[0])) / mn if mn > 0 else 0.0
        if assoc > 0:
            imp = _split_improvement(model, idx, best[1], s)
            out.append({"split": best[1], "reverse": best[2], "assoc": assoc,
                        "improvement": imp})
    out.sort(key=lambda d: -d["assoc"])
    return out[:s.get("max_surrogates", len(model.predictors) - 1)]


def _split_improvement(model: TreeModel, idx: np.ndarray, sp: Split, s: dict) -> float:
    pr = model.predictors[sp.var]
    side = _apply_rule(sp, pr.x[idx], pr)
    ok = side >= 0
    y = model.y[idx][ok]
    sd = side[ok]
    if model.ytype == "categorical":
        L = np.bincount(y[sd == 0].astype(int), minlength=model.J)[None, :].astype(float)
        R = np.bincount(y[sd == 1].astype(int), minlength=model.J)[None, :].astype(float)
        v = _impurity_split_values(model, y, L, R, s)[0]
    else:
        a, b = y[sd == 0], y[sd == 1]
        v = _impurity_split_values(model, y, None, None, s, (
            np.array([len(a)], float), np.array([a.sum()]), np.array([(a ** 2).sum()]),
            np.array([len(b)], float), np.array([b.sum()]), np.array([(b ** 2).sum()]),
            len(model.y)))[0]
    return float(v) if np.isfinite(v) else 0.0


def _finish_binary(model: TreeModel, node: Node, sp: Split, s: dict) -> None:
    idx = node.idx
    sp.surrogates = _surrogates(model, idx, sp, s) if s.get("max_surrogates", 1) > 0 else []
    pr = model.predictors[sp.var]
    side = _apply_rule(sp, pr.x[idx], pr)
    sp.majority = int((side == 1).sum() > (side == 0).sum())
    node.split = sp
    pos = model.route(node, np.arange(len(idx)), [p.x[idx] for p in model.predictors])
    node.split = None
    _make_children(model, node, sp, pos)


def _grow_crt(model: TreeModel, s: dict) -> None:
    queue = [1]
    while queue:
        node = model.nodes[queue.pop(0)]
        if _stop_basic(model, node, s):
            continue
        best = None
        forced = s.get("force_first") and node.id == 1
        for m in range(1 if forced else len(model.predictors)):
            r = _best_split_predictor(model, node.idx, m, s, s["min_child"])
            if r is not None and (best is None or r[0] > best[0] + 1e-15):
                best = r
        if best is None or (best[0] < s.get("min_improvement", 0.0001) and not forced):
            continue
        sp = best[1]
        sp.stat = {"improvement": best[0]}
        _finish_binary(model, node, sp, s)
        queue.extend(node.children)


# ============================================================================ QUEST
def _quest_discriminant(model: TreeModel) -> None:
    y = model.y.astype(int)
    for pr in model.predictors:
        if pr.kind != "nominal" or len(pr.labels) <= 2:
            continue
        ok = ~np.isnan(pr.x)
        codes = pr.x[ok].astype(int)
        I = len(pr.labels)
        V = np.zeros((ok.sum(), I))
        V[np.arange(len(codes)), codes] = 1
        yy = y[ok]
        vbar = V.mean(0)
        B = np.zeros((I, I))
        for j in range(model.J):
            sel = yy == j
            if sel.any():
                d = V[sel].mean(0) - vbar
                B += sel.sum() * np.outer(d, d)
        Tm = (V - vbar).T @ (V - vbar)
        d, Q = np.linalg.eigh(Tm)
        dinv = np.where(d > 1e-10 * max(d.max(), 1e-300), 1 / np.sqrt(np.abs(d)), 0.0)
        M = (dinv[:, None] * Q.T) @ B @ (Q * dinv[None, :])
        ev, vec = np.linalg.eigh((M + M.T) / 2)
        a = vec[:, -1]
        pr.xi = (a @ (dinv[:, None] * Q.T))     # coordinate of each category's dummy


def _quest_select(model: TreeModel, idx: np.ndarray, alpha: float):
    y = model.y[idx].astype(int)
    M = len(model.predictors)
    cont = [m for m, p in enumerate(model.predictors) if p.kind != "nominal"]
    best, bestp = None, math.inf
    for m, pr in enumerate(model.predictors):
        x = pr.x[idx]
        ok = ~np.isnan(x)
        if ok.sum() < 3:
            continue
        if pr.kind == "nominal":
            T = np.zeros((len(pr.labels), model.J))
            np.add.at(T, (x[ok].astype(int), y[ok]), 1)
            _, lp, df = _chi2_logp(T)
            if df == 0:
                continue
        else:
            groups = [x[ok][y[ok] == j] for j in range(model.J) if (y[ok] == j).sum()]
            if len(groups) < 2:
                continue
            S = np.array([[len(g), g.sum(), (g ** 2).sum()] for g in groups])
            _, lp, _ = _anova_logp(S)
        if lp < bestp:
            best, bestp = m, lp
    if best is not None and math.exp(bestp) < alpha / M:
        return best
    best, bestp = None, math.inf
    for m in cont:
        x = model.predictors[m].x[idx]
        ok = ~np.isnan(x)
        groups = [x[ok][y[ok] == j] for j in range(model.J) if (y[ok] == j).sum()]
        if len(groups) < 2:
            continue
        z = [np.abs(g - g.mean()) for g in groups]
        S = np.array([[len(g), g.sum(), (g ** 2).sum()] for g in z])
        _, lp, _ = _anova_logp(S)
        if lp < bestp:
            best, bestp = m, lp
    if best is not None and math.exp(bestp) < alpha / (M + len(cont)):
        return best
    return None


def _quest_point(model: TreeModel, xv: np.ndarray, yv: np.ndarray, probs_fn) -> float:
    classes = [j for j in range(model.J) if (yv == j).any()]
    means = {j: xv[yv == j].mean() for j in classes}
    if len(classes) == 2:
        A, B = [classes[0]], [classes[1]]
    else:
        mv = np.array([means[j] for j in classes])
        if np.allclose(mv, mv[0]):
            sizes = [(yv == j).sum() for j in classes]
            A = [classes[int(np.argmax(sizes))]]
            B = [j for j in classes if j not in A]
        else:
            c = np.array([mv.min(), mv.max()])
            for _ in range(100):
                lab = np.abs(mv[:, None] - c[None, :]).argmin(1)
                newc = np.array([mv[lab == k].mean() if (lab == k).any() else c[k]
                                 for k in range(2)])
                if np.allclose(newc, c):
                    break
                c = newc
            A = [j for j, lb in zip(classes, lab) if lb == 0]
            B = [j for j, lb in zip(classes, lab) if lb == 1]
            if not A or not B:
                A, B = classes[:1], classes[1:]
    xa, xb = xv[np.isin(yv, A)], xv[np.isin(yv, B)]
    p = probs_fn(yv)
    pa, pb = p[A].sum(), p[B].sum()
    ma, mb = xa.mean(), xb.mean()
    va = xa.var(ddof=1) if len(xa) > 1 else 0.0
    vb = xb.var(ddof=1) if len(xb) > 1 else 0.0
    if va > vb:
        ma, mb, va, vb, pa, pb = mb, ma, vb, va, pb, pa
    eps = 1e-12
    if va < eps:
        va = vb * eps if vb > eps else eps
    if vb < eps:
        vb = eps
    sa, sb = math.sqrt(va), math.sqrt(vb)
    a = va - vb
    b = 2 * (ma * vb - mb * va)
    c = (mb * sa) ** 2 - (ma * sb) ** 2 + 2 * va * vb * math.log(max(pa, eps) * sb
                                                                    / (max(pb, eps) * sa))
    lo, hi = xv.min(), xv.max()
    roots = []
    if abs(a) < 1e-12 * max(va, vb):
        if abs(b) > 0:
            roots = [-c / b]
    else:
        disc = b * b - 4 * a * c
        if disc >= 0:
            roots = [(-b + math.sqrt(disc)) / (2 * a), (-b - math.sqrt(disc)) / (2 * a)]
    roots = [r for r in roots if lo <= r < hi]
    if roots:
        return float(min(roots, key=lambda r: abs(r - ma)))
    return float((ma + mb) / 2)


def _grow_quest(model: TreeModel, s: dict) -> None:
    _quest_discriminant(model)
    queue = [1]
    while queue:
        node = model.nodes[queue.pop(0)]
        if _stop_basic(model, node, s):
            continue
        if s.get("force_first") and node.id == 1:
            m = 0
        else:
            m = _quest_select(model, node.idx, s.get("alpha_quest", 0.05))
        if m is None:
            continue
        pr = model.predictors[m]
        x = pr.x[node.idx]
        ok = ~np.isnan(x)
        yv = model.y[node.idx][ok].astype(int)
        probs_fn = lambda yy: _probs(model, np.bincount(yy, minlength=model.J)  # noqa: E731
                                     .astype(float))
        if pr.kind == "nominal":
            codes = x[ok].astype(int)
            cats = sorted(set(codes.tolist()))
            if len(cats) < 2:
                continue
            if len(cats) == 2:
                sp = Split(m, "in", left=frozenset([cats[0]]))
            else:
                xi = pr.xi[codes] if pr.xi is not None else codes.astype(float)
                d = _quest_point(model, xi, yv, probs_fn)
                left = frozenset(c for c in cats if pr.xi[c] <= d)
                if not left or len(left) == len(cats):
                    continue
                sp = Split(m, "in", left=left)
        else:
            d = _quest_point(model, x[ok], yv, probs_fn)
            if pr.kind == "ordinal":
                d = float(math.floor(d))
            sp = Split(m, "le", value=d)
        side = _apply_rule(sp, x, pr)
        nL, nR = (side == 0).sum(), (side == 1).sum()
        if nL < s["min_child"] or nR < s["min_child"] or nL == 0 or nR == 0:
            continue
        sp.stat = {"improvement": _split_improvement(model, node.idx, sp, s)}
        _finish_binary(model, node, sp, s)
        queue.extend(node.children)


# ============================================================================ risk, pruning
def node_risk(model: TreeModel, node: Node) -> float:
    """Resubstitution risk contribution R(t) of a node as if it were terminal."""
    N = len(model.y)
    if model.ytype == "scale":
        y = model.y[node.idx]
        return float(((y - y.mean()) ** 2).sum() / N) if len(y) else 0.0
    C = model.costs if model.costs is not None else 1 - np.eye(model.J)
    if model.priors is None:
        return float((C[node.pred] * node.counts).sum() / N)
    root = model.nodes[1].counts
    w = np.where(root > 0, model.priors / np.where(root > 0, root, 1), 0.0)
    return float((C[node.pred] * node.counts * w).sum())


def risk_estimate(model: TreeModel, node_of: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Risk of the tree on cases with terminal nodes ``node_of`` and outcomes ``y``."""
    N = len(y)
    if N == 0:
        return float("nan"), float("nan")
    if model.ytype == "scale":
        pred = np.array([model.nodes[t].mean for t in node_of])
        e2 = (y - pred) ** 2
        R = float(e2.mean())
        se = float(math.sqrt(max((e2 ** 2).mean() - R ** 2, 0.0) / N))
        return R, se
    C = model.costs if model.costs is not None else 1 - np.eye(model.J)
    pred = np.array([model.nodes[t].pred for t in node_of])
    loss = C[pred, y.astype(int)]
    if model.priors is None:
        R = float(loss.mean())
        se = float(math.sqrt(max((loss ** 2).mean() - R ** 2, 0.0) / N))
        return R, se
    R, var = 0.0, 0.0
    for j in range(model.J):
        sel = y == j
        Nj = sel.sum()
        if Nj:
            mj = loss[sel].mean()
            R += model.priors[j] * mj
            var += model.priors[j] ** 2 * ((loss[sel] ** 2).mean() - mj ** 2) / Nj
    return float(R), float(math.sqrt(max(var, 0.0)))


def prune(model: TreeModel, b: float = 1.0) -> int:
    """Cost-complexity pruning (Breiman et al.): weakest-link sequence, then the b-SE rule
    on the resubstitution risk. Returns the number of nodes removed."""
    sequence = [set(model.nodes)]
    current = {k: list(v.children) for k, v in model.nodes.items()}

    def leaves(t):
        if not current[t]:
            return [t]
        return [lf for c in current[t] for lf in leaves(c)]

    R = {k: node_risk(model, v) for k, v in model.nodes.items()}
    while current[1]:
        g = {}
        alive_now = _alive(current)
        for t, ch in current.items():
            if ch and t in alive_now:
                lv = leaves(t)
                g[t] = (R[t] - sum(R[x] for x in lv)) / max(len(lv) - 1, 1)
        gmin = min(g.values())
        for t in [t for t, v in g.items() if v <= gmin + 1e-12]:
            current[t] = []
        sequence.append(_alive(current))
    risks = []
    node_of = model.assign()
    for alive in sequence:
        term = [t for t in alive if not any(c in alive for c in model.nodes[t].children)]
        # map cases to their alive ancestor that is terminal in this subtree
        mapped = np.array([_ancestor_in(model, t, set(term)) for t in node_of])
        risks.append(risk_estimate(model, mapped, model.y))
    rs = np.array([r[0] for r in risks])
    kmin = int(np.argmin(rs))
    limit = rs[kmin] + b * risks[kmin][1]
    k = max(i for i in range(len(rs)) if rs[i] <= limit + 1e-12)
    keep = sequence[k]
    removed = [t for t in model.nodes if t not in keep]
    for t in removed:
        del model.nodes[t]
    for nd in model.nodes.values():
        if nd.children and not all(c in model.nodes for c in nd.children):
            nd.children = []
            nd.split = None
    return len(removed)


def _alive(current: dict) -> set:
    out, stack = set(), [1]
    while stack:
        t = stack.pop()
        out.add(t)
        stack.extend(current[t])
    return out


def _ancestor_in(model: TreeModel, t: int, term: set) -> int:
    while t not in term:
        t = model.nodes[t].parent
    return t


# ============================================================================ fitting
def fit_tree(y, ytype: str, ylabels: list[str], predictors: list[Predictor], method: str,
             settings: dict, priors=None, costs=None, yordinal: bool = False,
             dep: str = "y") -> TreeModel:
    s = dict(settings)
    s.setdefault("max_depth", DEFAULT_DEPTH[method])
    s.setdefault("min_parent", 100)
    s.setdefault("min_child", 50)
    s.setdefault("max_surrogates", max(len(predictors) - 1, 0))
    if method == "quest" and ytype != "categorical":
        raise MLError("QUEST requires a nominal dependent variable.")
    model = TreeModel(method, dep, ytype, np.asarray(y, float), list(ylabels), yordinal,
                      predictors, {}, None if priors is None else np.asarray(priors, float),
                      None if costs is None else np.asarray(costs, float), s)
    root = Node(1, None, 0, np.arange(len(model.y)))
    model.nodes[1] = root
    _node_stats(model, root)
    if method in ("chaid", "exhaustive"):
        _grow_chaid(model, s)
    elif method == "crt":
        _grow_crt(model, s)
    else:
        _grow_quest(model, s)
    if s.get("prune") and method in ("crt", "quest"):
        prune(model, s.get("prune_se", 1.0))
    return model


def importance(model: TreeModel) -> np.ndarray:
    """IBM Variable Importance: summed primary + surrogate improvements."""
    imp = np.zeros(len(model.predictors))
    for nd in model.nodes.values():
        if nd.split is None:
            continue
        imp[nd.split.var] += nd.split.stat.get("improvement", 0.0)
        for sur in nd.split.surrogates:
            imp[sur["split"].var] += sur.get("improvement", 0.0)
    return imp


def _prepare(ds: Dataset, dep: str, indep: list[str], method: str, s: dict, rows: np.ndarray,
             dep_kind: str):
    preds = []
    for name in indep:
        kind = measure_of(ds, name)
        if method in ("chaid", "exhaustive") and kind == "scale":
            x = raw_column(ds, name).to_numpy(float)[rows]
            codes, breaks = _bin_scale(x, int(s.get("intervals", 10)))
            preds.append(Predictor(name, "ordinal", codes, _bin_labels(breaks), "scale",
                                   breaks))
        elif kind == "scale":
            preds.append(Predictor(name, "scale", raw_column(ds, name).to_numpy(float)[rows],
                                   [], "scale"))
        else:
            cat = categorical(ds, name, rows)
            x = np.where(cat.codes >= 0, cat.codes, np.nan).astype(float)
            preds.append(Predictor(name, kind, x, cat.labels, kind))
    if dep_kind == "scale":
        y = raw_column(ds, dep).to_numpy(float)[rows]
        return y, [], preds, False
    cat = categorical(ds, dep, rows)
    return cat.codes.astype(float), cat.labels, preds, dep_kind == "ordinal"


# ============================================================================ procedure
def tree_procedure(ds: Dataset, dep: str, indep: list[str], method: str = "chaid",
                   force_first: bool = False, settings: dict | None = None,
                   validation: str = "none", folds: int = 10, train_pct: float = 50.0,
                   split_var: str | None = None, seed: int | None = None,
                   target: list | None = None, priors: str = "empirical",
                   custom_priors: list | None = None, costs=None,
                   output: dict | None = None, save: tuple = ()) -> MLResult:
    s = dict(settings or {})
    out = {"diagram": True, "table": False, "summary": True, "risk": True,
           "classification": True, "gains": True, "cumulative": True, "importance": True,
           "surrogates": False, "rules": False, "gain_chart": True, "index_chart": False,
           "response_chart": False, **(output or {})}
    if method not in METHODS:
        raise MLError(f"Unknown growing method {method!r}.")
    if not indep:
        raise MLError("Select at least one independent variable.")
    dep_kind = measure_of(ds, dep)
    if method == "quest" and dep_kind == "scale":
        raise MLError("QUEST requires a categorical (nominal or ordinal) dependent variable.")
    ys = raw_column(ds, dep)
    all_missing = np.ones(ds.n_cases, bool)
    for n in indep:
        all_missing &= raw_column(ds, n).isna().to_numpy()
    valid = ys.notna().to_numpy() & ~all_missing
    if split_var:
        valid &= raw_column(ds, split_var).notna().to_numpy()
    rows = np.where(valid)[0]
    if len(rows) < 2:
        raise MLError("Too few valid cases.")
    y, ylabels, preds, yord = _prepare(ds, dep, indep, method, s, rows, dep_kind)
    ytype = "scale" if dep_kind == "scale" else "categorical"
    J = len(ylabels)
    if ytype == "categorical" and J < 2:
        raise MLError("The dependent variable needs at least two categories.")
    pri = None
    if ytype == "categorical" and method in ("crt", "quest"):
        if priors == "equal":
            pri = np.full(J, 1 / J)
        elif priors == "custom" and custom_priors:
            v = np.asarray(custom_priors, float)[:J]
            pri = v / v.sum()
    cost = None
    if costs is not None and ytype == "categorical":
        cost = np.asarray(costs, float)[:J, :J]
    sample = np.zeros(len(rows), int)
    if validation == "split":
        if split_var:
            v = raw_column(ds, split_var).to_numpy(float)[rows]
            sample = np.where(v == 1, 0, 1)
        else:
            sample = (rng(seed).random(len(rows)) >= train_pct / 100).astype(int)
    train = np.where(sample == 0)[0]
    test = np.where(sample == 1)[0]
    if len(train) < 2:
        raise MLError("The training sample has too few cases.")

    def sub(pr, idx):
        return Predictor(pr.name, pr.kind, pr.x[idx], pr.labels, pr.orig_kind, pr.breaks)

    s["force_first"] = bool(force_first)
    model = fit_tree(y[train], ytype, ylabels, [sub(p, train) for p in preds], method, s,
                     pri, cost, yord, dep)
    section = Section("Tree")
    section.log(_syntax(dep, indep, method, s, validation))
    X_all = [p.x for p in preds]
    node_all = model.assign(X_all, len(rows))
    imp = importance(model) if method in ("crt", "quest") else None
    _summary_table(section, model, dep, indep, method, s, validation, folds, train_pct)
    if out["diagram"]:
        section.image("Tree Diagram", tree_diagram(model, ds), width=_diagram_width(model))
    if out["table"]:
        _tree_table(section, model)
    if out["surrogates"] and method in ("crt", "quest"):
        _surrogate_table(section, model)
    samples = [("Training", train)] + ([("Test", test)] if len(test) else [])
    label_all = len(samples) == 1 and validation != "split"
    if out["gains"]:
        for lab, idx in samples:
            _gains_tables(section, model, node_all[idx], y[idx], target, out,
                          None if label_all else lab)
    if out["risk"]:
        rows_r = []
        for lab, idx in samples:
            R, se = risk_estimate(model, node_all[idx], y[idx])
            rows_r.append(([] if label_all else [lab]) + [fmt(R, 3), fmt(se, 3)])
        if validation == "cv":
            R, se = cross_validated_risk(y, ytype, ylabels, preds, method, s, pri, cost, yord,
                                         folds, seed)
            rows_r = [["Resubstitution", *rows_r[0]], ["Cross-Validation", fmt(R, 3),
                                                        fmt(se, 3)]]
        cols = (["Method"] if validation == "cv" else ([] if label_all else ["Sample"])) + \
            ["Estimate", "Std. Error"]
        foot = [f"Growing Method: {METHODS[method]}", f"Dependent Variable: {dep}"]
        section.table(Table("Risk", cols, rows_r, row_header_cols=1 if len(cols) > 2 else 0,
                            footnotes=foot))
    if out["classification"] and ytype == "categorical":
        lead = ["Observed"] if label_all else ["Sample", "Observed"]
        cols = lead + [("Predicted", c) for c in ylabels] + [("Predicted", "Percent Correct")]
        rows_c = []
        for lab, idx in samples:
            pred = np.array([model.nodes[t].pred for t in node_all[idx]])
            rows_c += classification_rows(y[idx].astype(int), pred, ylabels,
                                          None if label_all else lab)
        section.table(Table("Classification", cols, rows_c, row_header_cols=len(lead),
                            footnotes=[f"Growing Method: {METHODS[method]}",
                                       f"Dependent Variable: {dep}"]))
    if out["importance"] and imp is not None and imp.sum() > 0:
        importance_table(section, indep, imp)
        section.image("Independent Variable Importance",
                      importance_chart(indep, imp, f"Independent Variable Importance"))
    if out["rules"]:
        section.note("Selection Rules (SPSS syntax)", selection_rules(model))
    saved = {}
    if save:
        node_full = node_all
        if "node" in save:
            saved["NodeID"] = ("Terminal node identifier",
                               full_series(ds, rows, node_full))
        if "predicted" in save:
            pv = np.array([model.nodes[t].pred for t in node_full], float)
            if ytype == "categorical":
                cat = categorical(ds, dep, rows)
                pv = np.array([float(cat.levels[int(k)]) if not isinstance(cat.levels[int(k)],
                                                                         str) else np.nan
                               for k in pv])
            saved["PredictedValue"] = ("Predicted value", full_series(ds, rows, pv))
        if "probability" in save and ytype == "categorical":
            for j in range(J):
                pj = np.array([model.nodes[t].probs[j] for t in node_full])
                saved[f"PredictedProbability_{j + 1}"] = (
                    f"Predicted probability for {dep} = {ylabels[j]}",
                    full_series(ds, rows, pj))
        if "sample" in save and validation == "split":
            saved["SampleAssign"] = ("Sample assignment (1 = training, 0 = test)",
                                     full_series(ds, rows, (sample == 0).astype(float)))
    return MLResult(section, saved, model)


def cross_validated_risk(y, ytype, ylabels, preds, method, s, pri, cost, yord, folds: int,
                         seed) -> tuple[float, float]:
    n = len(y)
    fold = rng(seed).permutation(n) % max(2, int(folds))
    losses = np.zeros(n)
    for v in range(max(2, int(folds))):
        te = np.where(fold == v)[0]
        tr = np.where(fold != v)[0]
        if not len(te) or len(tr) < 2:
            continue
        sub = [Predictor(p.name, p.kind, p.x[tr], p.labels, p.orig_kind, p.breaks)
               for p in preds]
        m = fit_tree(y[tr], ytype, ylabels, sub, method, s, pri, cost, yord)
        node_te = m.assign([p.x[te] for p in preds], len(te))
        if ytype == "scale":
            pred = np.array([m.nodes[t].mean for t in node_te])
            losses[te] = (y[te] - pred) ** 2
        else:
            C = cost if cost is not None else 1 - np.eye(len(ylabels))
            pred = np.array([m.nodes[t].pred for t in node_te])
            losses[te] = C[pred, y[te].astype(int)]
    R = float(losses.mean())
    return R, float(math.sqrt(max((losses ** 2).mean() - R ** 2, 0.0) / n))


# ============================================================================ output
def _syntax(dep, indep, method, s, validation) -> str:
    return (f"TREE {dep} [{', '.join(indep)}]\n  /METHOD TYPE={METHODS[method].upper()}\n"
            f"  /GROWTHLIMIT MAXDEPTH={s.get('max_depth', DEFAULT_DEPTH[method])} "
            f"MINPARENTSIZE={s.get('min_parent', 100)} MINCHILDSIZE={s.get('min_child', 50)}"
            f"\n  /VALIDATION TYPE={validation.upper()}.")


def _summary_table(section, model, dep, indep, method, s, validation, folds, train_pct):
    vtxt = {"none": "None", "cv": f"Cross Validation ({folds} folds)",
            "split": f"Split Sample ({train_pct:g}% training)"}[validation]
    rows = [["Specifications", "Growing Method", METHODS[method]],
            ["", "Dependent Variable", dep],
            ["", "Independent Variables", ", ".join(indep)],
            ["", "Validation", vtxt],
            ["", "Maximum Tree Depth", fmt_int(s.get("max_depth", DEFAULT_DEPTH[method]))],
            ["", "Minimum Cases in Parent Node", fmt_int(s.get("min_parent", 100))],
            ["", "Minimum Cases in Child Node", fmt_int(s.get("min_child", 50))]]
    if method in ("crt",):
        rows.append(["", "Impurity Measure", {"gini": "Gini", "twoing": "Twoing",
                                              "ordered": "Ordered Twoing"}.get(
            s.get("impurity", "gini"), "Gini") if model.ytype == "categorical"
            else "Least Squares Deviation"])
    used = model.used_vars()
    rows += [["Results", "Independent Variables Included", ", ".join(used) or "(none)"],
             ["", "Number of Nodes", fmt_int(len(model.nodes))],
             ["", "Number of Terminal Nodes", fmt_int(len(model.terminals()))],
             ["", "Depth", fmt_int(model.depth())]]
    section.table(Table("Model Summary", ["", "", ""], rows, row_header_cols=2))


def _p_text(sp: Split) -> str:
    lp = sp.stat.get("logp", 0.0)
    return fmt(min(1.0, math.exp(lp)), 3)


def _tree_table(section: Section, model: TreeModel) -> None:
    N = len(model.y)
    chaid = model.method in ("chaid", "exhaustive")
    if model.ytype == "categorical":
        cols = ["Node"]
        for lab in model.ylabels:
            cols += [(lab, "N"), (lab, "Percent")]
        cols += [("Total", "N"), ("Total", "Percent"), "Predicted Category", "Parent Node"]
    else:
        cols = ["Node", "Mean", "Std. Deviation", "N", "Percent", "Predicted Mean",
                "Parent Node"]
    if chaid:
        stat = "F" if model.ytype == "scale" else "Chi-Square"
        cols += [("Primary Independent Variable", "Variable"),
                 ("Primary Independent Variable", "Sig.ᵃ"),
                 ("Primary Independent Variable", stat),
                 ("Primary Independent Variable", "df"),
                 ("Primary Independent Variable", "Split Values")]
    else:
        cols += [("Primary Independent Variable", "Variable"),
                 ("Primary Independent Variable", "Improvement"),
                 ("Primary Independent Variable", "Split Values")]
    rows = []
    for nid in sorted(model.nodes):
        nd = model.nodes[nid]
        n = len(nd.idx)
        if model.ytype == "categorical":
            r = [str(nid)]
            for j in range(model.J):
                c = nd.counts[j]
                r += [fmt_int(c), f"{100 * c / n:.1f}%" if n else ""]
            r += [fmt_int(n), f"{100 * n / N:.1f}%", model.ylabels[nd.pred],
                  "" if nd.parent is None else str(nd.parent)]
        else:
            r = [str(nid), fmt(nd.mean, 3), fmt(nd.sd, 3), fmt_int(n), f"{100 * n / N:.1f}%",
                 fmt(nd.mean, 3), "" if nd.parent is None else str(nd.parent)]
        if nd.parent is None:
            r += [""] * (5 if chaid else 3)
        else:
            par = model.nodes[nd.parent]
            sp = par.split
            name = model.predictors[sp.var].name
            if chaid:
                df = sp.stat.get("df")
                dft = f"{df[0]}, {df[1]}" if isinstance(df, tuple) else fmt_int(df)
                r += [name, _p_text(sp), fmt(sp.stat.get("stat"), 3), dft, nd.branch]
            else:
                r += [name, fmt(sp.stat.get("improvement"), 3), nd.branch]
        rows.append(r)
    foot = ["a. Bonferroni adjusted"] if chaid else []
    foot += [f"Growing Method: {METHODS[model.method]}", f"Dependent Variable: {model.dep}"]
    section.table(Table("Tree Table", cols, rows, footnotes=foot))


def _surrogate_table(section: Section, model: TreeModel) -> None:
    rows = []
    for nid in sorted(model.nodes):
        nd = model.nodes[nid]
        if nd.split is None:
            continue
        name = model.predictors[nd.split.var].name
        rows.append([str(nid), "Primary", name, fmt(nd.split.stat.get("improvement"), 3), ""])
        for k, sur in enumerate(nd.split.surrogates, 1):
            rows.append(["", f"Surrogate {k}", model.predictors[sur["split"].var].name,
                         fmt(sur["improvement"], 3), fmt(sur["assoc"], 3)])
    section.table(Table("Surrogates", ["Parent Node", "", "Independent Variable",
                                       "Improvement", "Association"], rows,
                        row_header_cols=2))


def gains(model: TreeModel, node_of: np.ndarray, y: np.ndarray, target: int | None):
    terms = sorted(set(node_of.tolist()))
    N = len(y)
    out = []
    if model.ytype == "categorical":
        Nt = float((y == target).sum())
        base = Nt / N if N else float("nan")
        for t in terms:
            sel = node_of == t
            n = int(sel.sum())
            g = int((y[sel] == target).sum())
            resp = g / n if n else 0.0
            out.append({"node": t, "n": n, "pct": n / N, "gain": g,
                        "gain_pct": g / Nt if Nt else float("nan"), "response": resp,
                        "index": resp / base if base else float("nan")})
        out.sort(key=lambda d: (-d["response"], d["node"]))
    else:
        mean_all = float(y.mean())
        for t in terms:
            sel = node_of == t
            n = int(sel.sum())
            m = float(y[sel].mean())
            out.append({"node": t, "n": n, "pct": n / N, "mean": m,
                        "index": m / mean_all if mean_all else float("nan")})
        out.sort(key=lambda d: (-d["mean"], d["node"]))
    return out


def _gains_tables(section, model, node_of, y, target, out, sample_label):
    title_s = f" ({sample_label})" if sample_label else ""
    if model.ytype == "categorical":
        targets = list(range(model.J)) if target is None else [int(t) for t in target]
        targets = targets[:1] if target is None else targets
        for tj in targets:
            g = gains(model, node_of, y, tj)
            rows = [[str(d["node"]), fmt_int(d["n"]), f"{100 * d['pct']:.1f}%",
                     fmt_int(d["gain"]), f"{100 * d['gain_pct']:.1f}%",
                     f"{100 * d['response']:.1f}%", f"{100 * d['index']:.1f}%"] for d in g]
            cols = ["Node", ("Node", "N"), ("Node", "Percent"), ("Gain", "N"),
                    ("Gain", "Percent"), "Response", "Index"]
            cols[0] = "Node"
            section.table(Table(f"Gain Summary for Nodes{title_s}", cols, rows, footnotes=[
                f"Growing Method: {METHODS[model.method]}", f"Dependent Variable: {model.dep}",
                f"Target Category: {model.ylabels[tj]}"]))
            if out.get("cumulative"):
                cn = np.cumsum([d["n"] for d in g])
                cg = np.cumsum([d["gain"] for d in g])
                N, Nt = cn[-1], max(cg[-1], 1)
                base = Nt / N
                rows = [[str(d["node"]), fmt_int(a), f"{100 * a / N:.1f}%", fmt_int(b),
                         f"{100 * b / Nt:.1f}%", f"{100 * b / a:.1f}%",
                         f"{100 * (b / a) / base:.1f}%"] for d, a, b in zip(g, cn, cg)]
                section.table(Table(f"Cumulative Gain Summary{title_s}", [
                    "Node", ("Node", "N"), ("Node", "Percent"), ("Gain", "N"),
                    ("Gain", "Percent"), "Response", "Index"], rows, footnotes=[
                    f"Target Category: {model.ylabels[tj]}"]))
            if sample_label in (None, "Training"):
                _gains_charts(section, g, out, model.ylabels[tj])
    else:
        g = gains(model, node_of, y, None)
        rows = [[str(d["node"]), fmt_int(d["n"]), f"{100 * d['pct']:.1f}%", fmt(d["mean"], 4)]
                for d in g]
        section.table(Table(f"Gain Summary for Nodes{title_s}",
                            ["Node", "N", "Percent", "Mean"], rows,
                            footnotes=[f"Growing Method: {METHODS[model.method]}",
                                       f"Dependent Variable: {model.dep}"]))


def _gains_charts(section, g, out, target_label):
    cn = np.cumsum([d["n"] for d in g])
    cg = np.cumsum([d["gain"] for d in g])
    x = np.r_[0, 100 * cn / cn[-1]]
    gain = np.r_[0, 100 * cg / max(cg[-1], 1)]
    if out.get("gain_chart"):
        fig = figure(5.0, 4.2)
        ax = fig.add_subplot(111)
        ax.plot(x, gain, marker="o", ms=3)
        ax.plot([0, 100], [0, 100], color="#999", lw=0.8)
        style(ax, f"Gains for target category {target_label}", "Percentile", "Gain (%)")
        section.image("Gains Chart", png(fig), width=460)
    with np.errstate(divide="ignore", invalid="ignore"):
        resp = 100 * cg / cn
        idx = resp / (100 * cg[-1] / cn[-1]) * 100
    if out.get("index_chart"):
        fig = figure(5.0, 4.2)
        ax = fig.add_subplot(111)
        ax.plot(x[1:], idx, marker="o", ms=3)
        ax.axhline(100, color="#999", lw=0.8)
        style(ax, f"Index for target category {target_label}", "Percentile", "Index (%)")
        section.image("Index Chart", png(fig), width=460)
    if out.get("response_chart"):
        fig = figure(5.0, 4.2)
        ax = fig.add_subplot(111)
        ax.plot(x[1:], resp, marker="o", ms=3)
        style(ax, f"Response for target category {target_label}", "Percentile",
              "Response (%)")
        section.image("Response Chart", png(fig), width=460)


def selection_rules(model: TreeModel) -> str:
    lines = []
    for t in sorted(n.id for n in model.terminals()):
        conds = []
        cur = model.nodes[t]
        while cur.parent is not None:
            par = model.nodes[cur.parent]
            k = par.children.index(cur.id)
            conds.append(_rule_condition(model, par.split, k))
            cur = par
        conds = conds[::-1] or ["1"]
        nd = model.nodes[t]
        pred = model.ylabels[nd.pred] if model.ytype == "categorical" else fmt(nd.mean, 4)
        lines.append(f"/* Node {t} */.\nDO IF {' AND '.join(conds)}.\n"
                     f"COMPUTE nod_001 = {t}.\nCOMPUTE pre_001 = {pred!r}.\nEND IF.")
    return "\n".join(lines) + "\nEXECUTE."


def _rule_condition(model: TreeModel, sp: Split, k: int) -> str:
    pr = model.predictors[sp.var]
    name = pr.name
    if sp.kind == "le":
        return f"(VALUE({name}) <= {sp.value:.6g})" if k == 0 else \
            f"(VALUE({name}) > {sp.value:.6g})"
    if sp.kind == "sets":
        if pr.orig_kind == "scale" and pr.breaks is not None:
            cs = [c for c in sp.groups[k] if c >= 0]
            b = pr.breaks
            parts = []
            if cs:
                lo, hi = min(cs), max(cs)
                if lo > 0:
                    parts.append(f"VALUE({name}) > {b[lo - 1]:.6g}")
                if hi < len(b):
                    parts.append(f"VALUE({name}) <= {b[hi]:.6g}")
            cond = " AND ".join(parts) or "1"
            if -1 in sp.groups[k]:
                cond = f"({cond}) OR SYSMIS({name})"
            return f"({cond})"
        labs = [pr.labels[c] if c >= 0 else "missing" for c in sp.groups[k]]
        return f"({name} IN [{', '.join(labs)}])"
    codes = sorted(sp.left) if k == 0 else sorted(set(range(len(pr.labels))) - set(sp.left))
    return f"({name} IN [{', '.join(pr.labels[c] for c in codes)}])"


# ============================================================================ diagram
def _layout(model: TreeModel) -> dict[int, tuple[float, int]]:
    pos: dict[int, tuple[float, int]] = {}
    counter = [0]

    def visit(t):
        nd = model.nodes[t]
        if nd.terminal:
            pos[t] = (counter[0], nd.depth)
            counter[0] += 1
        else:
            for c in nd.children:
                visit(c)
            xs = [pos[c][0] for c in nd.children]
            pos[t] = ((min(xs) + max(xs)) / 2, nd.depth)
    visit(1)
    return pos


def _diagram_width(model: TreeModel) -> int:
    leaves = len(model.terminals())
    return int(min(1800, max(560, 190 * leaves)))


def tree_diagram(model: TreeModel, ds: Dataset | None = None) -> bytes:
    """The SPSS tree editor view: node tables, split statistics and branch values."""
    pos = _layout(model)
    leaves = max(1, len(model.terminals()))
    depth = model.depth()
    N = len(model.y)
    cat = model.ytype == "categorical"
    box_lines = (model.J + 2) if cat else 5
    W = max(6.0, min(30.0, 2.3 * leaves))
    H = max(3.5, (depth + 1) * (0.32 * box_lines + 1.5))
    fig = figure(W, H)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-0.6, leaves - 0.4)
    level_h = 1.0
    ax.set_ylim(-(depth + 1) * level_h + 0.1, 0.12)
    ax.axis("off")
    fs = max(5.0, min(8.0, 70.0 / max(leaves, 1)))
    bw = 0.86
    bh = min(level_h * 0.6, 0.052 * (box_lines + 1) + 0.03)
    for nid, nd in model.nodes.items():
        x, d = pos[nid]
        top = -d * level_h
        if nd.parent is not None:
            px, pd_ = pos[nd.parent]
            ptop = -pd_ * level_h - bh
            ymid = ptop - (level_h - bh) * 0.55
            ax.plot([px, px], [ptop, ymid], color="#555", lw=0.8)
            ax.plot([px, x], [ymid, ymid], color="#555", lw=0.8)
            ax.plot([x, x], [ymid, top], color="#555", lw=0.8)
            ax.text(x, (ymid + top) / 2, nd.branch, ha="center", va="center", fontsize=fs,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
        ax.add_patch(_rect(x - bw / 2, top - bh, bw, bh))
        n = len(nd.idx)
        lines = [f"Node {nid}"]
        if cat:
            lines.append(f"{'Category':<12}{'%':>7}{'n':>6}")
            for j, lab in enumerate(model.ylabels):
                c = nd.counts[j]
                mark = "■" if j == nd.pred else " "
                lines.append(f"{mark}{lab[:11]:<11}{100 * c / n if n else 0:7.1f}{int(c):6d}")
            lines.append(f" {'Total':<11}{100 * n / N:7.1f}{n:6d}")
        else:
            lines += [f"Mean       {nd.mean:9.3f}", f"Std. Dev.  {nd.sd:9.3f}",
                      f"n          {n:9d}", f"%          {100 * n / N:9.1f}",
                      f"Predicted  {nd.mean:9.3f}"]
        ax.text(x, top - 0.02, "\n".join(lines), ha="center", va="top", fontsize=fs,
                family="monospace")
        if nd.split is not None:
            sp = nd.split
            name = model.predictors[sp.var].name
            if model.method in ("chaid", "exhaustive"):
                df = sp.stat.get("df")
                stat_name = "F" if model.ytype == "scale" else "Chi-square"
                dft = f"{df[0]}, {df[1]}" if isinstance(df, tuple) else f"{df}"
                info = (f"{name}\nAdj. P-value={_p_text(sp)}, {stat_name}={sp.stat['stat']:.3f}"
                        f", df={dft}")
            else:
                info = f"{name}\nImprovement={sp.stat.get('improvement', 0):.3f}"
            ax.text(x, top - bh - (level_h - bh) * 0.28, info, ha="center", va="center",
                    fontsize=fs, color="#1f3c88",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#eef2fb", ec="#99a"))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    return buf.getvalue()


def _rect(x, y, w, h):
    from matplotlib.patches import FancyBboxPatch
    return FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01", fc="#fbfbfb", ec="#444",
                          lw=0.8)


# ============================================================================ selftest
def selftest() -> None:
    r = np.random.default_rng(1)
    n = 400
    x1 = r.integers(0, 4, n).astype(float)
    x2 = r.normal(size=n)
    y = ((x1 >= 2) ^ (x2 > 0.5)).astype(float)
    for method in METHODS:
        preds = [Predictor("x1", "nominal", x1, ["a", "b", "c", "d"], "nominal")]
        if method in ("chaid", "exhaustive"):
            codes, br = _bin_scale(x2, 10)
            preds.append(Predictor("x2", "ordinal", codes, _bin_labels(br), "scale", br))
        else:
            preds.append(Predictor("x2", "scale", x2, [], "scale"))
        m = fit_tree(y, "categorical", ["0", "1"], preds, method,
                     {"min_parent": 20, "min_child": 10})
        assert len(m.terminals()) >= 2, method
        acc = np.mean([m.nodes[t].pred for t in m.assign()] == y)
        assert acc > 0.8, (method, acc)
    tree_diagram(m)

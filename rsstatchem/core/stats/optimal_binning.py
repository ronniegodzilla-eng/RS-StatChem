"""Transform > Optimal Binning (SPSS OPTIMAL BINNING, Data Preparation add-on), following IBM's
"OPTIMAL BINNING Algorithms" chapter: supervised MDLP discretisation of scale variables against
a categorical guide variable (Fayyad & Irani 1993).

* Ent(S) = −Σ P(C_i, S) log₂ P(C_i, S); E(A, T; S) = |S₁|/|S| Ent(S₁) + |S₂|/|S| Ent(S₂) with
  S₁ = {A ≤ T}; Gain = Ent(S) − E.
* Candidate cut points are the distinct values d_i for which d_i and d_{i+1} do not belong to
  the same class (a value belongs to a class when all its cases have that class); the cut with
  the smallest E is accepted iff Gain > log₂(N − 1)/N + Δ/N with
  Δ = log₂(3^k − 2) − [k Ent(S) − k₁ Ent(S₁) − k₂ Ent(S₂)]. IBM's worked example prints the
  threshold 0.6530774 for its 7-case table, which is reproduced (to 4 decimals) when the
  entropies inside Δ use natural logarithms, so Δ is computed that way; Gain uses log₂ as
  printed (0.5916728). MDLPCut recurses on both halves (iteratively, with a stack).
* Hybrid MDLP: with more distinct values than the pre-binning maximum (default 1000), equal
  frequency pre-bins (empirical percentiles) form the candidate set first.
* Sparse bins (interior bins whose size divided by the smaller neighbour is below the merging
  threshold, default 5%) are merged, in one pass, with the neighbour giving the lower class
  information entropy.
* Model entropy E_m = Σ P(A_i) (−Σ P(C_j|A_i) log₂ P(C_j|A_i)).

Bins are reported SPSS style with inclusive lower limits (Lower ≤ x < Upper), the upper end
point being the next observed value above the accepted cut (identical bins on the data); the
"exclusive" option uses Lower < x ≤ Upper with the cut values themselves.
"""
from __future__ import annotations

import math

import numpy as np

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int
from rsstatchem.core.stats.ml_common import MLError, categorical, raw_column

BinError = MLError


def ent(counts, base: float = 2.0) -> float:
    c = np.asarray(counts, float)
    n = c.sum()
    if n <= 0:
        return 0.0
    p = c[c > 0] / n
    return float(-(p * np.log(p)).sum() / math.log(base))


def mdlp_threshold(left, right) -> tuple[float, float]:
    """(Gain, acceptance threshold) for a binary split given class counts on both sides."""
    left, right = np.asarray(left, float), np.asarray(right, float)
    S = left + right
    N = S.sum()
    e = (left.sum() * ent(left) + right.sum() * ent(right)) / N
    gain = ent(S) - e
    k, k1, k2 = (int((c > 0).sum()) for c in (S, left, right))
    delta = math.log2(3 ** k - 2) - (k * ent(S, math.e) - k1 * ent(left, math.e)
                                     - k2 * ent(right, math.e))
    return gain, math.log2(N - 1) / N + delta / N


def class_info_entropy(M: np.ndarray, lo: int, hi: int, cut: int) -> float:
    """E(A, T; S) for rows lo..hi (inclusive) of the value-by-class matrix, cut after row
    ``cut``."""
    left = M[lo:cut + 1].sum(0)
    right = M[cut + 1:hi + 1].sum(0)
    n = left.sum() + right.sum()
    return (left.sum() * ent(left) + right.sum() * ent(right)) / n


def _candidates(M: np.ndarray, lo: int, hi: int) -> list[int]:
    pure = [(M[i] > 0).sum() == 1 for i in range(lo, hi + 1)]
    cls = [int(np.argmax(M[i])) for i in range(lo, hi + 1)]
    out = []
    for j in range(hi - lo):
        if not (pure[j] and pure[j + 1] and cls[j] == cls[j + 1]):
            out.append(lo + j)
    return out


def mdlp(M: np.ndarray) -> list[int]:
    """Indices i of accepted cuts (cut after row i) for the sorted value-by-class matrix."""
    cuts = []
    stack = [(0, len(M) - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi <= lo:
            continue
        cand = _candidates(M, lo, hi)
        if not cand:
            continue
        E = [class_info_entropy(M, lo, hi, c) for c in cand]
        best = cand[int(np.argmin(E))]
        gain, thr = mdlp_threshold(M[lo:best + 1].sum(0), M[best + 1:hi + 1].sum(0))
        if gain > thr:
            cuts.append(best)
            stack.append((lo, best))
            stack.append((best + 1, hi))
    return sorted(cuts)


def value_class_matrix(x: np.ndarray, y_codes: np.ndarray, n_classes: int,
                       prebin: int | None = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Sorted distinct values (or equal-frequency pre-bin upper values) and their class
    counts."""
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], y_codes[order]
    values = np.unique(xs)
    if prebin and len(values) > prebin:
        q = np.quantile(xs, np.arange(1, prebin + 1) / prebin, method="inverted_cdf")
        values = np.unique(q)
    idx = np.searchsorted(values, xs, side="left")
    idx = np.minimum(idx, len(values) - 1)
    M = np.zeros((len(values), n_classes))
    np.add.at(M, (idx, ys), 1)
    keep = M.sum(1) > 0
    return values[keep], M[keep]


def merge_sparse(bins: list[np.ndarray], threshold: float) -> tuple[list[np.ndarray], list[int]]:
    """IBM's single pass over interior bins; returns the merged count rows and, for each
    surviving boundary, the index of the original boundary kept."""
    rows = [b.astype(float) for b in bins]
    bounds = list(range(len(rows) - 1))      # boundary j sits between rows j and j+1
    i = 1
    while i < len(rows) - 1:
        size = rows[i].sum()
        ratio = size / max(min(rows[i - 1].sum(), rows[i + 1].sum()), 1e-300)
        if ratio < threshold:
            left = rows[i - 1] + rows[i]
            right = rows[i] + rows[i + 1]
            e_left = ent(left) * left.sum() + ent(rows[i + 1]) * rows[i + 1].sum()
            e_right = ent(right) * right.sum() + ent(rows[i - 1]) * rows[i - 1].sum()
            if e_left <= e_right:
                rows[i - 1] = left
                del rows[i]
                del bounds[i - 1]
            else:
                rows[i + 1] = right
                del rows[i]
                del bounds[i]
            continue
        i += 1
    return rows, bounds


def model_entropy(bin_counts: list[np.ndarray]) -> float:
    tot = sum(b.sum() for b in bin_counts)
    return float(sum(b.sum() / tot * ent(b) for b in bin_counts if b.sum() > 0))


def bin_variable(x: np.ndarray, y_codes: np.ndarray, n_classes: int, prebin: int | None = 1000,
                 merge: bool = True, threshold: float = 0.05,
                 lowerlimit: str = "inclusive") -> dict:
    values, M = value_class_matrix(x, y_codes, n_classes, prebin)
    cut_idx = mdlp(M) if len(values) > 1 else []
    edges = [-1] + cut_idx + [len(values) - 1]
    bins = [M[a + 1:b + 1].sum(0) for a, b in zip(edges[:-1], edges[1:])]
    if merge and len(bins) > 2:
        bins, keep = merge_sparse(bins, threshold)
        cut_idx = [cut_idx[j] for j in keep]
    xs = np.sort(np.unique(x))
    cuts = [float(values[i]) for i in cut_idx]
    if lowerlimit == "inclusive":
        ends = [float(xs[np.searchsorted(xs, c, side="right")]) for c in cuts]
    else:
        ends = list(cuts)
    return {"cuts": cuts, "ends": ends, "bins": bins, "entropy": model_entropy(bins),
            "distinct": int(len(np.unique(x))), "lowerlimit": lowerlimit}


def assign(x: np.ndarray, ends: list[float], lowerlimit: str = "inclusive") -> np.ndarray:
    x = np.asarray(x, float)
    side = "right" if lowerlimit == "inclusive" else "left"
    out = np.searchsorted(np.asarray(ends, float), x, side=side) + 1.0
    out[~np.isfinite(x)] = np.nan
    return out


def optimal_binning(ds: Dataset, variables: list[str], guide: str, prebin: int | None = 1000,
                    merge: bool = True, merge_threshold: float = 5.0,
                    lowerlimit: str = "inclusive", missing: str = "pairwise",
                    save: dict | None = None, syntax_rules: bool = False,
                    descriptives: bool = True, entropy_table: bool = True,
                    summaries: bool = True) -> Section:
    """``save`` maps a binned variable to the name of its new variable (created in ``ds``)."""
    if not variables or not guide:
        raise BinError("Select variables to bin and a guide variable.")
    if guide in variables:
        raise BinError("The guide variable cannot also be binned.")
    for v in variables:
        if ds.var(v).type != "numeric":
            raise BinError(f"{v} is not numeric; only scale variables can be binned.")
    section = Section("Optimal Binning")
    section.log(f"OPTIMAL BINNING\n  /VARIABLES GUIDE={guide} BIN={' '.join(variables)}"
                + (f" SAVE=YES (INTO={' '.join(save.values())})" if save else "")
                + f"\n  /CRITERIA METHOD=MDLP PREPROCESS="
                + (f"EQUALFREQ (BINS={prebin})" if prebin else "NONE")
                + f" FORCEMERGE={merge_threshold if merge else 0:g}"
                + f" LOWERLIMIT={lowerlimit.upper()}\n  /MISSING SCOPE="
                + ("PAIRWISE" if missing == "pairwise" else "LISTWISE") + ".")
    g = raw_column(ds, guide)
    ok_all = g.notna().to_numpy()
    if missing == "listwise":
        for v in variables:
            ok_all &= raw_column(ds, v).notna().to_numpy()
    cat = categorical(ds, guide)
    labels = cat.labels
    results = {}
    for v in variables:
        x = raw_column(ds, v).to_numpy(float)
        ok = ok_all & np.isfinite(x)
        if ok.sum() < 2:
            raise BinError(f"Too few valid cases for {v}.")
        r = bin_variable(x[ok], cat.codes[ok], len(cat.levels), prebin, merge,
                         merge_threshold / 100, lowerlimit)
        r["n"], r["min"], r["max"] = int(ok.sum()), float(x[ok].min()), float(x[ok].max())
        r["x"], r["ok"] = x, ok
        results[v] = r
    if descriptives:
        section.table(Table("Descriptive Statistics", ["", "N", "Minimum", "Maximum",
                                                       "Number of Distinct Values",
                                                       "Number of Bins"],
                            [[v, fmt_int(r["n"]), fmt(r["min"], 2), fmt(r["max"], 2),
                              fmt_int(r["distinct"]), fmt_int(len(r["bins"]))]
                             for v, r in results.items()]))
    if entropy_table:
        section.table(Table("Model Entropy", ["", "Model Entropy"],
                            [[v, fmt(r["entropy"], 3)] for v, r in results.items()],
                            footnotes=["Smaller model entropy indicates higher predictive "
                                       f"accuracy of the binned variable on guide variable "
                                       f"{guide}."]))
    if summaries:
        rel = "<=" if lowerlimit == "inclusive" else "<"
        rel2 = "<" if lowerlimit == "inclusive" else "<="
        for v, r in results.items():
            ends = r["ends"]
            rows = []
            for k, b in enumerate(r["bins"]):
                lo = fmt(ends[k - 1], 2) if k > 0 else "ᵃ"
                hi = fmt(ends[k], 2) if k < len(ends) else "ᵃ"
                rows.append([str(k + 1), lo, hi] + [fmt_int(c) for c in b] + [fmt_int(b.sum())])
            tot = np.sum(r["bins"], axis=0)
            rows.append(["Total", "", ""] + [fmt_int(c) for c in tot] + [fmt_int(tot.sum())])
            cols = ["Bin", ("End Point", "Lower"), ("End Point", "Upper")] + \
                [(f"Number of Cases by Level of {guide}", lab) for lab in labels] + ["Total"]
            fn = [f"Each bin is computed as Lower {rel} {v} {rel2} Upper.",
                  "a. Unbounded."]
            if len(r["bins"]) == 1:
                fn.append(f"No cut point was accepted; {v} forms a single bin.")
            section.table(Table(v, cols, rows, footnotes=fn))
    if syntax_rules:
        lines = []
        for v, r in results.items():
            target = (save or {}).get(v, f"{v}_bin")
            parts = []
            ends = r["ends"]
            for k in range(len(ends) + 1):
                if lowerlimit == "inclusive":
                    cond = " AND ".join(c for c in (
                        f"{v} >= {ends[k - 1]:g}" if k > 0 else "",
                        f"{v} < {ends[k]:g}" if k < len(ends) else "") if c)
                else:
                    cond = " AND ".join(c for c in (
                        f"{v} > {ends[k - 1]:g}" if k > 0 else "",
                        f"{v} <= {ends[k]:g}" if k < len(ends) else "") if c)
                parts.append(f"IF (MISSING({v}) = 0 AND {cond or '1 = 1'}) {target} = {k + 1}.")
            lines.append(f"*{v}.\n" + "\n".join(parts))
        section.note("Binning Rules (syntax)", "\n".join(lines))
    if save:
        created = []
        for v, name in save.items():
            r = results[v]
            vals = assign(r["x"], r["ends"], lowerlimit)
            if missing == "listwise":
                vals[~r["ok"]] = np.nan
            if ds.has_name(name):
                raise BinError(f"A variable named {name} already exists.")
            var = ds.add_variable(name, values=vals)
            var.label = f"{v} (Binned)"
            var.measure = "ordinal"
            var.decimals = 0
            var.value_labels = {float(k + 1): _bin_label(v, r["ends"], k, lowerlimit)
                                for k in range(len(r["bins"]))}
            created.append(name)
        section.table(Table("Binned Variables", ["", "Binned Variable"],
                            [[v, n] for v, n in save.items()]))
    section.result = results  # type: ignore[attr-defined]
    return section


def _bin_label(v: str, ends: list[float], k: int, lowerlimit: str) -> str:
    lo = ends[k - 1] if k > 0 else None
    hi = ends[k] if k < len(ends) else None
    a, b = ("<=", "<") if lowerlimit == "inclusive" else ("<", "<=")
    if lo is None:
        return f"{v} {b} {hi:g}" if hi is not None else "All"
    if hi is None:
        return f"{v} {'>=' if a == '<=' else '>'} {lo:g}"
    return f"{lo:g} {a} {v} {b} {hi:g}"


def selftest() -> None:
    gain, thr = mdlp_threshold([1, 0], [0, 6])
    assert abs(gain - 0.5916728) < 1e-6 and abs(thr - 0.6530774) < 2e-4
    x = np.r_[np.linspace(0, 1, 30), np.linspace(2, 3, 30)]
    y = np.r_[np.zeros(30, int), np.ones(30, int)]
    r = bin_variable(x, y, 2)
    assert len(r["bins"]) == 2 and r["ends"] == [2.0]

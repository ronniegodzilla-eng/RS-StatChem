"""Analyze > Correlate > Distance Correlation (SPSS 31), following IBM's "Distance Correlation"
algorithm chapter (Székely, Rizzo & Bakirov 2007).

Euclidean distance matrices a_ij, b_ij are double-centred (A = a − row mean − column mean +
grand mean); V²(X, Y) = mean(A·B), V²(X, X) = mean(A²), V²(Y, Y) = mean(B²) and
R(X, Y) = V(X, Y) / √(V(X, X) V(Y, Y)) (0 when a distance variance is 0, clamped to [0, 1]).
Inference as IBM documents it: a Fisher-z interval tanh(atanh R ± z_{α/2}/√(n − 3)) and a
permutation test that shuffles Y B times (default 1000, seeded), with p = mean(R* ≥ R) and a
percentile interval of the permutation distribution (linear interpolation at the
(α/2)·B and (1 − α/2)·B positions). X and Y may each be several variables (multivariate
distance correlation), or a list of variables gives the pairwise matrix.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int
from rsstatchem.core.stats.ml_common import MLError, raw_column, rng

DcorError = MLError


def centered(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(2))
    return d - d.mean(0)[None, :] - d.mean(1)[:, None] + d.mean()


def dcor_from_centered(A: np.ndarray, B: np.ndarray) -> tuple[float, float, float, float]:
    """(R, V²xy, V²xx, V²yy)."""
    vxy = float((A * B).mean())
    vxx = max(0.0, float((A * A).mean()))
    vyy = max(0.0, float((B * B).mean()))
    if vxx == 0 or vyy == 0:
        return 0.0, vxy, vxx, vyy
    r = math.sqrt(max(vxy, 0.0)) / math.sqrt(math.sqrt(vxx * vyy))
    return min(1.0, max(0.0, r)), vxy, vxx, vyy


def distance_correlation(x, y) -> float:
    return dcor_from_centered(centered(x), centered(y))[0]


def fisher_ci(r: float, n: int, level: float = 0.95) -> tuple[float, float]:
    if r == 0.0:
        return 0.0, 0.0
    if r == 1.0:
        return 1.0, 1.0
    rc = min(max(r, 1e-10), 1 - 1e-10)
    z = 0.5 * math.log((1 + rc) / (1 - rc))
    se = 1 / math.sqrt(n - 3) if n > 3 else float("inf")
    zc = stats.norm.ppf(1 - (1 - level) / 2)
    return max(0.0, math.tanh(z - zc * se)), min(1.0, math.tanh(z + zc * se))


def _interp(sorted_vals: np.ndarray, idx: float) -> float:
    lo = int(math.floor(idx))
    frac = idx - lo
    if lo + 1 >= len(sorted_vals):
        return float(sorted_vals[-1])
    return float(sorted_vals[lo] + frac * (sorted_vals[lo + 1] - sorted_vals[lo]))


def permutation_test(x, y, n_perm: int = 1000, alpha: float = 0.05, level: float = 0.95,
                     seed: int | None = None) -> dict:
    A = centered(x)
    Yv = np.asarray(y, float)
    B = centered(Yv)
    r, vxy, vxx, vyy = dcor_from_centered(A, B)
    n = len(A)
    g = rng(seed)
    perm = []
    for _ in range(int(n_perm)):
        idx = g.permutation(n)
        Bp = B[np.ix_(idx, idx)]             # centred distances of the permuted sample
        rp, _, _, vyp = dcor_from_centered(A, Bp)
        if vxx == 0 or vyp == 0 or not np.isfinite(rp):
            continue
        perm.append(rp)
    perm = np.sort(np.array(perm))
    if len(perm):
        lo = _interp(perm, alpha / 2 * len(perm))
        hi = _interp(perm, (1 - alpha / 2) * len(perm))
        p = float((perm >= r).mean())
    else:
        lo = hi = p = float("nan")
    return {"dcor": r, "dcov2": vxy, "dvarx2": vxx, "dvary2": vyy, "n": n,
            "fisher": fisher_ci(r, n, level), "perm_ci": (lo, hi), "p": p,
            "B": len(perm)}


def _matrix(ds: Dataset, names: list[str]) -> np.ndarray:
    return np.column_stack([raw_column(ds, v).to_numpy(float) for v in names])


def dcor_procedure(ds: Dataset, variables: list[str] | None = None,
                   x_set: list[str] | None = None, y_set: list[str] | None = None,
                   n_perm: int = 1000, alpha: float = 0.05, level: float = 0.95,
                   seed: int | None = None, missing: str = "pairwise") -> Section:
    pairs: list[tuple[list[str], list[str]]] = []
    if x_set and y_set:
        pairs = [(list(x_set), list(y_set))]
    elif variables and len(variables) >= 2:
        pairs = [([a], [b]) for i, a in enumerate(variables) for b in variables[i + 1:]]
    else:
        raise DcorError("Select at least two variables (or an X set and a Y set).")
    allv = sorted({v for a, b in pairs for v in a + b}, key=lambda v: ds.index_of(v))
    for v in allv:
        if ds.var(v).type != "numeric":
            raise DcorError(f"{v} is not numeric.")
    listwise_ok = np.isfinite(_matrix(ds, allv)).all(1)
    section = Section("Distance Correlation")
    section.log("DISTANCE CORRELATION /VARIABLES="
                + (f"{' '.join(x_set)} WITH {' '.join(y_set)}" if x_set and y_set
                   else " ".join(variables))
                + f"\n  /CRITERIA PERMUTATIONS={n_perm} CILEVEL={level * 100:g}"
                f" SIGLEVEL={alpha:g}" + (f" SEED={seed}" if seed is not None else "")
                + f"\n  /MISSING={missing.upper()}.")
    rows = []
    results = {}
    for a, b in pairs:
        X, Y = _matrix(ds, a), _matrix(ds, b)
        ok = listwise_ok if missing == "listwise" else (np.isfinite(X).all(1)
                                                        & np.isfinite(Y).all(1))
        if ok.sum() < 4:
            raise DcorError("Too few valid cases.")
        res = permutation_test(X[ok], Y[ok], n_perm, alpha, level, seed)
        results[(tuple(a), tuple(b))] = res
        pr = stats.pearsonr(X[ok, 0], Y[ok, 0])[0] if len(a) == 1 and len(b) == 1 else None
        rows.append([" ".join(a), " ".join(b), fmt_int(res["n"]), fmt(res["dcor"], 3),
                     fmt(res["fisher"][0], 3), fmt(res["fisher"][1], 3),
                     fmt(res["perm_ci"][0], 3), fmt(res["perm_ci"][1], 3), fmt(res["p"], 3),
                     fmt(pr, 3) if pr is not None else ""])
    lv = f"{level * 100:g}%"
    pl = f"{(1 - alpha) * 100:g}%"
    section.table(Table("Distance Correlations", [
        "X", "Y", "N", "Distance Correlation", (f"Fisher {lv} CI", "Lower"),
        (f"Fisher {lv} CI", "Upper"), (f"Permutation {pl} Interval", "Lower"),
        (f"Permutation {pl} Interval", "Upper"), "Permutation Sig.", "Pearson r"], rows,
        footnotes=[f"Permutation test with {n_perm} random permutations of Y; Sig. = "
                   "proportion of permuted distance correlations at least as large as the "
                   "observed one.",
                   "The permutation interval is the percentile interval of the null "
                   "(permutation) distribution."]))
    if variables and not (x_set and y_set) and len(variables) > 2:
        k = len(variables)
        M = np.eye(k)
        for i in range(k):
            for j in range(i + 1, k):
                M[i, j] = M[j, i] = results[((variables[i],), (variables[j],))]["dcor"]
        section.table(Table("Distance Correlation Matrix", [""] + list(variables),
                            [[variables[i]] + [fmt(M[i, j], 3) for j in range(k)]
                             for i in range(k)]))
    section.result = results  # type: ignore[attr-defined]
    return section


def selftest() -> None:
    x = np.linspace(-1, 1, 21)
    assert distance_correlation(x, x) > 0.999
    assert distance_correlation(x, x ** 2) > 0.3

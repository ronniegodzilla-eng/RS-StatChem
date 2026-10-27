"""Analyze > Correlate > Canonical Correlation (SPSS 25+ CANCORR output).

Canonical correlations with eigenvalues and sequential Wilks tests (Rao's F approximation),
standardized and raw canonical coefficients, canonical loadings, cross loadings and the
proportion of variance explained (redundancy). Canonical variate scores can be saved.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_p, listwise


@dataclass
class CanCorrResult:
    section: Section
    saved: dict = field(default_factory=dict)


def _inv_sqrt(S: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(S)
    return V @ np.diag(1 / np.sqrt(np.clip(w, 1e-12, None))) @ V.T


def canonical(X: np.ndarray, Y: np.ndarray):
    """Returns r, A (standardized coefficients set 1, p x k), B (set 2, q x k)."""
    Zx = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    Zy = (Y - Y.mean(axis=0)) / Y.std(axis=0, ddof=1)
    n = len(X)
    Rxx = Zx.T @ Zx / (n - 1)
    Ryy = Zy.T @ Zy / (n - 1)
    Rxy = Zx.T @ Zy / (n - 1)
    Kx, Ky = _inv_sqrt(Rxx), _inv_sqrt(Ryy)
    M = Kx @ Rxy @ Ky
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    k = min(X.shape[1], Y.shape[1])
    A = Kx @ U[:, :k]
    B = Ky @ Vt.T[:, :k]
    for i in range(k):           # sign convention: loadings of set 1 sum positive
        if (Rxx @ A[:, i]).sum() < 0:
            A[:, i] *= -1
            B[:, i] *= -1
    return s[:k], A, B, Rxx, Ryy, Rxy, Zx, Zy


def wilks_tests(r: np.ndarray, n: int, p: int, q: int):
    rows = []
    for k in range(len(r)):
        lam = float(np.prod(1 - r[k:] ** 2))
        pk, qk = p - k, q - k
        df1 = pk * qk
        w = n - 1.5 - (p + q) / 2
        t = math.sqrt((pk ** 2 * qk ** 2 - 4) / (pk ** 2 + qk ** 2 - 5)) \
            if pk ** 2 + qk ** 2 - 5 > 0 else 1.0
        df2 = w * t - df1 / 2 + 1
        root = lam ** (1 / t)
        F = (1 - root) / root * df2 / df1
        rows.append((lam, F, df1, df2, float(stats.f.sf(F, df1, df2))))
    return rows


def canonical_correlation(ds: Dataset, set1: list[str], set2: list[str],
                          n_vars: int | None = None, save: bool = False) -> CanCorrResult:
    section = Section("Canonical Correlation")
    result = CanCorrResult(section)
    section.log("STATS CANCORR SET1=" + " ".join(set1) + " SET2=" + " ".join(set2)
                + (f"\n  /OPTIONS NCOMP={n_vars}" if n_vars else "")
                + ("\n  /SAVE SCORES" if save else "") + ".")
    frame = listwise(ds, list(dict.fromkeys(set1 + set2)))
    n = len(frame)
    X, Y = frame[set1].to_numpy(float), frame[set2].to_numpy(float)
    p, q = X.shape[1], Y.shape[1]
    if n <= p + q + 1:
        section.note("Warning", "Not enough valid cases.", style="warning")
        return result
    r, A, B, Rxx, Ryy, Rxy, Zx, Zy = canonical(X, Y)
    k = len(r) if not n_vars else min(n_vars, len(r))
    tests = wilks_tests(r, n, p, q)
    rows = [[str(i + 1), fmt(r[i]), fmt(r[i] ** 2 / (1 - r[i] ** 2)), fmt(tests[i][0]),
             fmt(tests[i][1]), fmt(tests[i][2], 3), fmt(tests[i][3], 3), fmt_p(tests[i][4])]
            for i in range(k)]
    section.table(Table("Canonical Correlations", ["", "Correlation", "Eigenvalue",
                                                   "Wilks Statistic", "F", "Num D.F",
                                                   "Denom D.F.", "Sig."], rows,
                        footnotes=["H0 for Wilks test is that the correlations in the current "
                                   "and following rows are zero"]))
    sx, sy = X.std(axis=0, ddof=1), Y.std(axis=0, ddof=1)
    names1 = [ds.var(v).title for v in set1]
    names2 = [ds.var(v).title for v in set2]
    heads = [str(i + 1) for i in range(k)]

    def mat(title, names, M):
        section.table(Table(title, ["Variable"] + heads,
                            [[nm] + [fmt(M[j, i]) for i in range(k)]
                             for j, nm in enumerate(names)]))

    mat("Set 1 Standardized Canonical Correlation Coefficients", names1, A)
    mat("Set 2 Standardized Canonical Correlation Coefficients", names2, B)
    mat("Set 1 Unstandardized Canonical Correlation Coefficients", names1, A / sx[:, None])
    mat("Set 2 Unstandardized Canonical Correlation Coefficients", names2, B / sy[:, None])
    Lx = Rxx @ A
    Ly = Ryy @ B
    mat("Set 1 Canonical Loadings", names1, Lx)
    mat("Set 2 Canonical Loadings", names2, Ly)
    mat("Set 1 Cross Loadings", names1, Lx * r[None, :])
    mat("Set 2 Cross Loadings", names2, Ly * r[None, :])
    prow = []
    for i in range(k):
        s1 = float(np.mean(Lx[:, i] ** 2))
        s2 = float(np.mean(Ly[:, i] ** 2))
        prow.append([str(i + 1), fmt(s1), fmt(s1 * r[i] ** 2), fmt(s2), fmt(s2 * r[i] ** 2)])
    section.table(Table("Proportion of Variance Explained", [
        "Canonical Variable", "Set 1 by Self", "Set 1 by Set 2", "Set 2 by Self",
        "Set 2 by Set 1"], prow))
    if save:
        U = Zx @ A
        V = Zy @ B
        for i in range(k):
            for tag, M in (("s1_cv", U), ("s2_cv", V)):
                s = pd.Series(np.nan, index=ds.df.index)
                s[frame.index] = M[:, i]
                result.saved[f"{tag}{i + 1:03d}"] = (
                    f"Canonical variate {i + 1} of set {tag[1]}", s)
    return result


def selftest() -> None:
    rng = np.random.default_rng(3)
    n = 80
    a = rng.normal(size=(n, 3))
    b = a[:, :2] @ np.array([[0.6, 0.1], [0.2, 0.5]]) + rng.normal(size=(n, 2))
    ds = Dataset(pd.DataFrame(np.column_stack([a, b]), columns=["a1", "a2", "a3", "b1", "b2"]))
    canonical_correlation(ds, ["a1", "a2", "a3"], ["b1", "b2"], save=True)

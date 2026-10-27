"""Analyze > Correlate > Distances (SPSS PROXIMITIES).

Distances between cases or between variables. Interval measures (Euclidean, squared Euclidean,
Chebychev, block, Minkowski, customized power, Pearson correlation, cosine), count measures
(chi-square, phi-square) and the binary measures of SPSS (computed from the 2x2 table a, b, c,
d of present/absent values). Values can be standardised (z scores, range -1 to 1, range 0 to
1, maximum magnitude 1, mean 1, standard deviation 1) by variable or by case, and measures
transformed (absolute values, change sign, rescale to 0-1).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, numeric, value_text

INTERVAL_DISSIM = {"euclid": "Euclidean distance", "seuclid": "Squared Euclidean distance",
                   "chebychev": "Chebychev", "block": "Block", "minkowski": "Minkowski",
                   "power": "Customized"}
INTERVAL_SIM = {"correlation": "Pearson correlation", "cosine": "Cosine"}
COUNT = {"chisq": "Chi-square measure", "ph2": "Phi-square measure"}
BINARY_DISSIM = {"beuclid": "Euclidean distance", "bseuclid": "Squared Euclidean distance",
                 "size": "Size difference", "pattern": "Pattern difference",
                 "variance": "Variance", "bshape": "Shape", "lw": "Lance and Williams"}
BINARY_SIM = {"rr": "Russell and Rao", "sm": "Simple matching", "jaccard": "Jaccard",
              "dice": "Dice", "ss1": "Sokal and Sneath 1", "rt": "Rogers and Tanimoto",
              "ss2": "Sokal and Sneath 2", "k1": "Kulczynski 1", "ss3": "Sokal and Sneath 3",
              "k2": "Kulczynski 2", "ss4": "Sokal and Sneath 4", "hamann": "Hamann",
              "lambda": "Lambda", "d": "Anderberg's D", "y": "Yule's Y", "q": "Yule's Q",
              "ochiai": "Ochiai", "ss5": "Sokal and Sneath 5", "phi": "Phi 4-point correlation",
              "disper": "Dispersion"}
SIMILARITIES = set(INTERVAL_SIM) | set(BINARY_SIM)
STANDARDIZE = {"none": "None", "z": "Z scores", "range1": "Range -1 to 1",
               "range01": "Range 0 to 1", "max1": "Maximum magnitude of 1",
               "mean1": "Mean of 1", "sd1": "Standard deviation of 1"}


def _standardize(M: np.ndarray, how: str, axis: int) -> np.ndarray:
    """``axis`` 0 = by variable (columns), 1 = by case (rows)."""
    if how == "none":
        return M
    A = M if axis == 0 else M.T
    with np.errstate(divide="ignore", invalid="ignore"):
        if how == "z":
            A = (A - A.mean(axis=0)) / A.std(axis=0, ddof=1)
        elif how == "range1":
            A = A / (A.max(axis=0) - A.min(axis=0))
        elif how == "range01":
            A = (A - A.min(axis=0)) / (A.max(axis=0) - A.min(axis=0))
        elif how == "max1":
            A = A / np.abs(A).max(axis=0)
        elif how == "mean1":
            A = A / A.mean(axis=0)
        elif how == "sd1":
            A = A / A.std(axis=0, ddof=1)
    A = np.nan_to_num(A)
    return A if axis == 0 else A.T


def _binary_counts(u: np.ndarray, v: np.ndarray, present: float, absent: float):
    pu, pv = u == present, v == present
    au, av = u == absent, v == absent
    ok = (pu | au) & (pv | av)
    a = float(np.sum(pu & pv & ok))
    b = float(np.sum(pu & av & ok))
    c = float(np.sum(au & pv & ok))
    d = float(np.sum(au & av & ok))
    return a, b, c, d


def binary_measure(m: str, a: float, b: float, c: float, d: float) -> float:
    n = a + b + c + d
    with np.errstate(divide="ignore", invalid="ignore"):
        if m == "beuclid":
            return math.sqrt(b + c)
        if m == "bseuclid":
            return b + c
        if m == "size":
            return (b - c) ** 2 / n ** 2
        if m == "pattern":
            return b * c / n ** 2
        if m == "variance":
            return (b + c) / (4 * n)
        if m == "bshape":
            return (n * (b + c) - (b - c) ** 2) / n ** 2
        if m == "lw":
            return _div(b + c, 2 * a + b + c)
        if m == "rr":
            return a / n
        if m == "sm":
            return (a + d) / n
        if m == "jaccard":
            return _div(a, a + b + c)
        if m == "dice":
            return _div(2 * a, 2 * a + b + c)
        if m == "ss1":
            return _div(2 * (a + d), 2 * (a + d) + b + c)
        if m == "rt":
            return _div(a + d, a + d + 2 * (b + c))
        if m == "ss2":
            return _div(a, a + 2 * (b + c))
        if m == "k1":
            return _div(a, b + c)
        if m == "ss3":
            return _div(a + d, b + c)
        if m == "k2":
            return (_div(a, a + b) + _div(a, a + c)) / 2
        if m == "ss4":
            return (_div(a, a + b) + _div(a, a + c) + _div(d, b + d) + _div(d, c + d)) / 4
        if m == "hamann":
            return ((a + d) - (b + c)) / n
        if m in ("lambda", "d"):
            t1 = max(a, b) + max(c, d) + max(a, c) + max(b, d)
            t2 = max(a + c, b + d) + max(a + b, c + d)
            return _div(t1 - t2, 2 * n - t2) if m == "lambda" else (t1 - t2) / (2 * n)
        if m == "y":
            return _div(math.sqrt(a * d) - math.sqrt(b * c), math.sqrt(a * d) + math.sqrt(b * c))
        if m == "q":
            return _div(a * d - b * c, a * d + b * c)
        if m == "ochiai":
            return math.sqrt(_div(a, a + b) * _div(a, a + c))
        if m == "ss5":
            return _div(a * d, math.sqrt((a + b) * (a + c) * (b + d) * (c + d)))
        if m == "phi":
            return _div(a * d - b * c, math.sqrt((a + b) * (a + c) * (b + d) * (c + d)))
        if m == "disper":
            return (a * d - b * c) / n ** 2
    raise ValueError(m)


def _div(x, y):
    return x / y if y else float("nan")


def proximity(M: np.ndarray, measure: str, power: float = 2.0, root: float = 2.0,
              present: float = 1.0, absent: float = 0.0) -> np.ndarray:
    """Proximities between the rows of ``M``."""
    k = M.shape[0]
    out = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            u, v = M[i], M[j]
            if measure == "euclid":
                val = math.sqrt(np.sum((u - v) ** 2))
            elif measure == "seuclid":
                val = float(np.sum((u - v) ** 2))
            elif measure == "chebychev":
                val = float(np.max(np.abs(u - v)))
            elif measure == "block":
                val = float(np.sum(np.abs(u - v)))
            elif measure == "minkowski":
                val = float(np.sum(np.abs(u - v) ** power) ** (1 / power))
            elif measure == "power":
                val = float(np.sum(np.abs(u - v) ** power) ** (1 / root))
            elif measure == "correlation":
                val = float(np.corrcoef(u, v)[0, 1]) if np.ptp(u) and np.ptp(v) else \
                    float("nan")
            elif measure == "cosine":
                den = math.sqrt(np.sum(u * u) * np.sum(v * v))
                val = float(np.sum(u * v) / den) if den else float("nan")
            elif measure in COUNT:
                tab = np.vstack([u, v])
                tot = tab.sum()
                exp_ = np.outer(tab.sum(axis=1), tab.sum(axis=0)) / tot
                with np.errstate(divide="ignore", invalid="ignore"):
                    chi = float(np.nansum((tab - exp_) ** 2 / exp_))
                val = math.sqrt(chi) if measure == "chisq" else math.sqrt(chi / tot)
            else:
                val = binary_measure(measure, *_binary_counts(u, v, present, absent))
            out[i, j] = val
    return out


def distances(ds: Dataset, variables: list[str], between: str = "cases",
              measure: str = "euclid", standardize: str = "none", by: str = "variable",
              power: float = 2.0, root: float = 2.0, present: float = 1.0,
              absent: float = 0.0, absolute: bool = False, change_sign: bool = False,
              rescale: bool = False, label: str | None = None,
              max_cases: int = 60) -> Section:
    """SPSS PROXIMITIES. ``between`` cases / variables."""
    section = Section("Proximities")
    kind = "similarity" if measure in SIMILARITIES else "dissimilarity"
    mkw = measure.upper() + (f"({power:g},{root:g})" if measure == "power" else
                             f"({power:g})" if measure == "minkowski" else
                             f"({present:g},{absent:g})" if measure in BINARY_DISSIM
                             or measure in BINARY_SIM else "")
    section.log(f"PROXIMITIES {' '.join(variables)}"
                + (f"\n  /STANDARDIZE={by.upper()} {standardize.upper()}"
                   if standardize != "none" else "\n  /STANDARDIZE=NONE")
                + f"\n  /VIEW={'CASE' if between == 'cases' else 'VARIABLE'}"
                + f"\n  /MEASURE={mkw}"
                + ("\n  /TRANSFORM " + " ".join(["ABSOLUTE"] * absolute
                                                + ["REVERSE"] * change_sign
                                                + ["RESCALE"] * rescale)
                   if absolute or change_sign or rescale else "")
                + "\n  /PRINT PROXIMITIES" + (f"\n  /ID={label}" if label else "") + ".")
    frame = pd.DataFrame({v: numeric(ds, v) for v in variables})
    total = len(frame)
    valid = frame.dropna()
    nv = len(valid)
    section.table(Table("Case Processing Summary", [("Cases", "Valid N"), ("Cases", "Percent"),
                                                    ("Cases", "Missing N"),
                                                    ("Cases", "Percent"), ("Cases", "Total N"),
                                                    ("Cases", "Percent")],
                        [[fmt_int(nv), fmt(100 * nv / total, 1) + "%", fmt_int(total - nv),
                          fmt(100 * (total - nv) / total, 1) + "%", fmt_int(total),
                          "100.0%"]], row_header_cols=0))
    M = valid.to_numpy(float)
    M = _standardize(M, standardize, 0 if by == "variable" else 1)
    if between == "variables":
        R = proximity(M.T, measure, power, root, present, absent)
        labels = [ds.var(v).title for v in variables]
    else:
        if nv > max_cases:
            section.note("Warning", f"The proximity matrix is shown for the first {max_cases} "
                         "cases only.", style="warning")
            M = M[:max_cases]
            valid = valid.iloc[:max_cases]
        R = proximity(M, measure, power, root, present, absent)
        if label:
            lv = ds.analysis_series(label)
            labels = [f"{i + 1}:{value_text(ds.var(label), lv.iloc[i])}" for i in valid.index]
        else:
            labels = [str(i + 1) for i in valid.index]
    if absolute:
        R = np.abs(R)
    if change_sign:
        R = -R
    if rescale:
        lo, hi = np.nanmin(R), np.nanmax(R)
        R = (R - lo) / (hi - lo) if hi > lo else R * 0
    title = {**INTERVAL_DISSIM, **INTERVAL_SIM, **COUNT, **BINARY_DISSIM,
             **BINARY_SIM}[measure]
    rows = [[labels[i]] + [fmt(R[i, j]) for j in range(len(labels))] for i in range(len(R))]
    section.table(Table("Proximity Matrix", [""] + [(title, lab) for lab in labels], rows,
                        footnotes=[f"This is a {kind} matrix"]))
    return section


def selftest() -> None:
    rng = np.random.default_rng(0)
    ds = Dataset(pd.DataFrame({"a": rng.normal(size=10), "b": rng.normal(size=10),
                               "c": rng.integers(0, 2, 10) * 1.0,
                               "d": rng.integers(0, 2, 10) * 1.0}))
    distances(ds, ["a", "b"], "cases", "euclid", "z")
    distances(ds, ["c", "d"], "variables", "jaccard")
    distances(ds, ["a", "b"], "variables", "correlation")

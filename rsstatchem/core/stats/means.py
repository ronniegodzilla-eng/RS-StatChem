"""Analyze > Compare Means > Means (SPSS MEANS).

Report table of cell statistics for dependent variables within categories of one or more
layers of independent variables, plus the optional ANOVA table (with test for linearity) and
Measures of Association (Eta, Eta Squared) for the first layer, as SPSS computes them.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, fmt_p, numeric, value_text
from rsstatchem.core.stats.descriptive import se_kurtosis, se_skewness

STATISTICS = {"mean": "Mean", "count": "N", "stddev": "Std. Deviation", "median": "Median",
              "gmedian": "Grouped Median", "semean": "Std. Error of Mean", "sum": "Sum",
              "min": "Minimum", "max": "Maximum", "range": "Range", "first": "First",
              "last": "Last", "variance": "Variance", "kurt": "Kurtosis",
              "sekurt": "Std. Error of Kurtosis", "skew": "Skewness",
              "seskew": "Std. Error of Skewness", "harmonic": "Harmonic Mean",
              "geometric": "Geometric Mean", "spct": "% of Total Sum", "npct": "% of Total N"}
DEFAULT = ("mean", "count", "stddev")
KEYWORD = {"mean": "MEAN", "count": "COUNT", "stddev": "STDDEV", "median": "MEDIAN",
           "gmedian": "GMEDIAN", "semean": "SEMEAN", "sum": "SUM", "min": "MIN", "max": "MAX",
           "range": "RANGE", "first": "FIRST", "last": "LAST", "variance": "VAR",
           "kurt": "KURT", "sekurt": "SEKURT", "skew": "SKEW", "seskew": "SESKEW",
           "harmonic": "HARMONIC", "geometric": "GEOMETRIC", "spct": "SPCT", "npct": "NPCT"}


def grouped_median(x: np.ndarray) -> float:
    """Median of data treated as grouped: each distinct value is the midpoint of its class."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0:
        return float("nan")
    vals, freq = np.unique(x, return_counts=True)
    if len(vals) == 1:
        return float(vals[0])
    lower = np.r_[vals[0] - (vals[1] - vals[0]) / 2, (vals[:-1] + vals[1:]) / 2]
    upper = np.r_[(vals[:-1] + vals[1:]) / 2, vals[-1] + (vals[-1] - vals[-2]) / 2]
    cum = np.cumsum(freq)
    i = int(np.searchsorted(cum, n / 2))
    before = cum[i - 1] if i > 0 else 0
    return float(lower[i] + (n / 2 - before) / freq[i] * (upper[i] - lower[i]))


def cell_stats(x: np.ndarray, total_sum: float, total_n: int) -> dict:
    x = np.asarray(x, float)
    n = len(x)
    out = {"count": n}
    if n == 0:
        return out
    s = pd.Series(x)
    sd = float(s.std(ddof=1)) if n > 1 else float("nan")
    out.update(mean=float(x.mean()), stddev=sd, median=float(np.median(x)),
               gmedian=grouped_median(x), semean=sd / math.sqrt(n) if n > 1 else float("nan"),
               sum=float(x.sum()), min=float(x.min()), max=float(x.max()),
               range=float(np.ptp(x)), first=float(x[0]), last=float(x[-1]),
               variance=sd ** 2 if n > 1 else float("nan"),
               kurt=float(s.kurt()) if n > 3 else float("nan"), sekurt=se_kurtosis(n),
               skew=float(s.skew()) if n > 2 else float("nan"), seskew=se_skewness(n),
               harmonic=float(stats.hmean(x)) if (x > 0).all() else float("nan"),
               geometric=float(stats.gmean(x)) if (x > 0).all() else float("nan"),
               spct=100 * float(x.sum()) / total_sum if total_sum else float("nan"),
               npct=100 * n / total_n if total_n else float("nan"))
    return out


def _format(key: str, value, decimals: int) -> str:
    if key == "count":
        return fmt_int(value)
    if key in ("spct", "npct"):
        return fmt(value, 1) + "%" if value == value else ""
    if key in ("kurt", "sekurt", "skew", "seskew"):
        return fmt(value, 3)
    if key in ("min", "max", "range", "first", "last", "sum"):
        return fmt(value, decimals)
    if key in ("variance",):
        return fmt(value, decimals + 3)
    return fmt(value, decimals + (2 if key in ("mean", "median", "gmedian", "harmonic",
                                               "geometric") else 3))


def _sort_key(v):
    return (isinstance(v, str), v)


def means(ds: Dataset, dependents: list[str], layers: list[list[str]],
          statistics=DEFAULT, anova: bool = False, linearity: bool = False) -> Section:
    statistics = [k for k in STATISTICS if k in statistics] or list(DEFAULT)
    layers = [lay for lay in layers if lay]
    section = Section("Means")
    section.log("MEANS TABLES=" + " ".join(dependents)
                + "".join(" BY " + " ".join(lay) for lay in layers)
                + "\n  /CELLS=" + " ".join(KEYWORD[k] for k in statistics)
                + (("\n  /STATISTICS ANOVA" + (" LINEARITY" if linearity else ""))
                   if anova else "") + ".")
    combos = list(itertools.product(*layers)) if layers else [()]

    # Case Processing Summary
    cps = []
    for combo in combos:
        for dep in dependents:
            names = [dep, *combo]
            frame = pd.DataFrame({n: (numeric(ds, n) if n == dep else ds.analysis_series(n))
                                  for n in dict.fromkeys(names)})
            total = len(frame)
            inc = int(frame.notna().all(axis=1).sum())
            label = " * ".join(ds.var(n).title for n in names)
            cps.append([label, fmt_int(inc), fmt(100 * inc / total, 1) + "%",
                        fmt_int(total - inc), fmt(100 * (total - inc) / total, 1) + "%",
                        fmt_int(total), "100.0%"])
    section.table(Table("Case Processing Summary", [""] + [
        (g, h) for g in ("Included", "Excluded", "Total") for h in ("N", "Percent")], cps))

    for combo in combos:
        frame = pd.DataFrame({d: numeric(ds, d) for d in dependents})
        for v in combo:
            frame[f"__{v}"] = ds.analysis_series(v)
        gcols = [f"__{v}" for v in combo]
        if gcols:
            frame = frame[frame[gcols].notna().all(axis=1)]
        totals = {d: (float(frame[d].dropna().sum()), int(frame[d].notna().sum()))
                  for d in dependents}
        cells: list[tuple[list[str], pd.DataFrame]] = []

        def walk(sub: pd.DataFrame, depth: int, heads: list[str]) -> None:
            if depth == len(combo):
                cells.append((heads, sub))
                return
            var = ds.var(combo[depth])
            col = gcols[depth]
            for v in sorted(sub[col].unique(), key=_sort_key):
                walk(sub[sub[col] == v], depth + 1, heads + [value_text(var, v)])
            walk(sub, depth + 1, heads + ["Total"])

        if combo:
            walk(frame, 0, [])
        else:
            cells.append(([], frame))
        nh = len(combo)
        rows = []
        single = len(dependents) == 1
        prev: list[str] = []
        for heads, sub in cells:
            shown = [h if heads[:i + 1] != prev[:i + 1] else "" for i, h in enumerate(heads)]
            prev = heads
            st = {d: cell_stats(sub[d].dropna().to_numpy(float), *totals[d])
                  for d in dependents}
            if single:
                d = dependents[0]
                dec = ds.var(d).decimals
                rows.append(shown + [_format(k, st[d].get(k, float("nan")), dec)
                                     for k in statistics])
            else:
                for j, k in enumerate(statistics):
                    rows.append((shown if j == 0 else [""] * nh) + [STATISTICS[k]]
                                + [_format(k, st[d].get(k, float("nan")), ds.var(d).decimals)
                                   for d in dependents])
        heads = [ds.var(v).title for v in combo]
        if single:
            dtitle = ds.var(dependents[0]).title
            cols = heads + [(dtitle, STATISTICS[k]) for k in statistics]
            section.table(Table("Report", cols, rows, row_header_cols=max(nh, 1)))
        else:
            cols = heads + [""] + [ds.var(d).title for d in dependents]
            section.table(Table("Report", cols, rows, row_header_cols=nh + 1))

        if anova and combo:
            _anova(section, ds, dependents, combo[0], linearity)
    return section


def _anova(section: Section, ds: Dataset, dependents: list[str], factor: str,
           linearity: bool) -> None:
    fvar = ds.var(factor)
    arows, mrows = [], []
    for dep in dependents:
        frame = pd.DataFrame({"y": numeric(ds, dep), "g": ds.analysis_series(factor)}).dropna()
        y = frame["y"].to_numpy(float)
        levels = sorted(frame["g"].unique(), key=_sort_key)
        n, k = len(y), len(levels)
        if k < 2 or n - k < 1:
            continue
        grand = y.mean()
        sst = float(((y - grand) ** 2).sum())
        ssb = float(sum(((frame.loc[frame["g"] == v, "y"].mean() - grand) ** 2)
                        * (frame["g"] == v).sum() for v in levels))
        ssw = sst - ssb
        msw = ssw / (n - k)
        label = f"{ds.var(dep).title} * {fvar.title}"
        f = (ssb / (k - 1)) / msw if msw > 0 else float("nan")
        arows.append([label, "Between Groups", "(Combined)", fmt(ssb), fmt_int(k - 1),
                      fmt(ssb / (k - 1)), fmt(f), fmt_p(stats.f.sf(f, k - 1, n - k))])
        eta2 = ssb / sst if sst > 0 else float("nan")
        mrow = [label]
        if linearity:
            x = pd.to_numeric(frame["g"], errors="coerce").to_numpy(float)
            if np.isnan(x).any():
                x = frame["g"].map({v: i for i, v in enumerate(levels)}).to_numpy(float)
            r = float(np.corrcoef(x, y)[0, 1]) if np.ptp(x) > 0 else float("nan")
            ss_lin = r ** 2 * sst
            ss_dev = ssb - ss_lin
            f_lin = ss_lin / msw if msw > 0 else float("nan")
            arows.append(["", "", "Linearity", fmt(ss_lin), "1", fmt(ss_lin), fmt(f_lin),
                          fmt_p(stats.f.sf(f_lin, 1, n - k))])
            if k > 2:
                f_dev = (ss_dev / (k - 2)) / msw if msw > 0 else float("nan")
                arows.append(["", "", "Deviation from Linearity", fmt(ss_dev), fmt_int(k - 2),
                              fmt(ss_dev / (k - 2)), fmt(f_dev),
                              fmt_p(stats.f.sf(f_dev, k - 2, n - k))])
            mrow += [fmt(r), fmt(r ** 2)]
        arows += [["", "Within Groups", "", fmt(ssw), fmt_int(n - k), fmt(msw), "", ""],
                  ["", "Total", "", fmt(sst), fmt_int(n - 1), "", "", ""]]
        mrows.append(mrow + [fmt(math.sqrt(eta2)), fmt(eta2)])
    if not arows:
        return
    section.table(Table("ANOVA Table", ["", "", "", "Sum of Squares", "df", "Mean Square", "F",
                                        "Sig."], arows, row_header_cols=3))
    section.table(Table("Measures of Association", [""] + (["R", "R Squared"] if linearity
                                                           else []) + ["Eta", "Eta Squared"],
                        mrows))

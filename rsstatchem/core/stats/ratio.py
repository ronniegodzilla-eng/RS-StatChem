"""Analyze > Descriptive Statistics > Ratio… (SPSS RATIO STATISTICS).

Summary statistics for the ratio of two scale variables (e.g. assessed value / sale price),
optionally by group: median, mean and weighted mean with confidence intervals, dispersion
(AAD, COD, PRD, median- and mean-centred COV, SD, range), concentration indices and the
price-related bias (PRB, SPSS 28). Formulas follow the IAAO Standard on Ratio Studies (2013):

    COD = 100 · mean|r − median| / median
    PRD = mean(r) / weighted mean,  weighted mean = Σ numerator / Σ denominator
    PRB: OLS slope of (r − median)/median on log2(0.5 · numerator/median + 0.5 · denominator)

The median CI uses binomial order statistics (actual coverage ≥ the requested level); the mean
CI is t based; the weighted-mean CI uses the delta-method (ratio estimator) standard error.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, numeric, value_text

STATISTICS = ("median", "mean", "wmean", "ci", "min", "max", "std", "range", "aad", "cod",
              "prd", "mdcov", "mncov", "prb")
DEFAULT_STATS = ("median", "mean", "wmean", "aad", "cod", "prd", "mdcov", "mncov", "std",
                 "range", "min", "max")


def median_ci(r: np.ndarray, level: float = 0.95) -> tuple[float, float, float]:
    rs = np.sort(np.asarray(r, float))
    n = len(rs)
    if n == 0:
        return (float("nan"),) * 3
    alpha = 1 - level
    lo = max(int(stats.binom.ppf(alpha / 2, n, 0.5)), 1)
    hi = min(int(stats.binom.ppf(1 - alpha / 2, n, 0.5)) + 1, n)
    cov = float(stats.binom.cdf(hi - 1, n, 0.5) - stats.binom.cdf(lo - 1, n, 0.5))
    return float(rs[lo - 1]), float(rs[hi - 1]), cov


def ratio_stats(num: np.ndarray, den: np.ndarray, level: float = 0.95,
                low: float | None = None, high: float | None = None,
                within: float | None = None) -> dict:
    """All RATIO STATISTICS quantities for one group."""
    a = np.asarray(num, float)
    s = np.asarray(den, float)
    r = a / s
    n = len(r)
    out: dict = {"n": n}
    if n == 0:
        return out
    med = float(np.median(r))
    mean = float(r.mean())
    wmean = float(a.sum() / s.sum())
    sd = float(r.std(ddof=1)) if n > 1 else float("nan")
    aad = float(np.mean(np.abs(r - med)))
    out.update(median=med, mean=mean, wmean=wmean, std=sd, min=float(r.min()),
               max=float(r.max()), range=float(r.max() - r.min()), aad=aad,
               cod=100 * aad / med if med else float("nan"),
               prd=mean / wmean if wmean else float("nan"),
               mdcov=100 * math.sqrt(np.sum((r - med) ** 2) / (n - 1)) / med
               if n > 1 and med else float("nan"),
               mncov=100 * sd / mean if mean and n > 1 else float("nan"))
    tq = stats.t.ppf(0.5 + level / 2, n - 1) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    out["mean_ci"] = (mean - tq * se, mean + tq * se)
    out["median_ci"] = median_ci(r, level)
    if n > 1:
        resid = a - wmean * s
        se_w = math.sqrt(np.sum(resid ** 2) / (n * (n - 1))) / s.mean()
        out["wmean_ci"] = (wmean - tq * se_w, wmean + tq * se_w)
    else:
        out["wmean_ci"] = (float("nan"), float("nan"))
    if low is not None and high is not None:
        out["between"] = 100.0 * float(np.mean((r >= low) & (r <= high)))
    if within is not None:
        lo, hi = med * (1 - within / 100), med * (1 + within / 100)
        out["within"] = 100.0 * float(np.mean((r >= lo) & (r <= hi)))
    out["prb"] = prb(a, s, med, level)
    return out


def prb(num: np.ndarray, den: np.ndarray, median: float, level: float = 0.95) -> dict:
    """Price-related bias (IAAO 2013): slope, its standard error and CI."""
    a = np.asarray(num, float)
    s = np.asarray(den, float)
    n = len(a)
    if n < 3 or not median:
        return {"value": float("nan"), "se": float("nan"), "ci": (float("nan"),) * 2}
    r = a / s
    y = (r - median) / median
    value = 0.5 * a / median + 0.5 * s
    x = np.log(value) / math.log(2)
    xc = x - x.mean()
    sxx = float(np.sum(xc ** 2))
    if sxx <= 0:
        return {"value": float("nan"), "se": float("nan"), "ci": (float("nan"),) * 2}
    slope = float(np.sum(xc * (y - y.mean())) / sxx)
    resid = y - y.mean() - slope * xc
    se = math.sqrt(np.sum(resid ** 2) / (n - 2) / sxx)
    tq = stats.t.ppf(0.5 + level / 2, n - 2)
    return {"value": slope, "se": se, "ci": (slope - tq * se, slope + tq * se)}


def ratio(ds: Dataset, numerator: str, denominator: str, group: str | None = None,
          order: str = "ascending", stats_=DEFAULT_STATS, ci: bool = True,
          level: float = 0.95, low: float | None = None, high: float | None = None,
          within: float | None = None) -> Section:
    """Ratio Statistics for <numerator> / <denominator>."""
    nvar, dvar = ds.var(numerator), ds.var(denominator)
    gvar = ds.var(group) if group else None
    section = Section("Ratio Statistics")
    printed = " ".join(s.upper() for s in STATISTICS if s in stats_)
    log = (f"RATIO STATISTICS {numerator} WITH {denominator}"
           + (f" BY {group} ({order.upper()[:1]})" if group else "")
           + "\n  /MISSING=EXCLUDE\n  /PRINT=" + printed + (f" CIN({level * 100:g})" if ci
                                                           else ""))
    if low is not None and high is not None:
        log += f" CONCENTRATION({low:g} {high:g})"
    if within is not None:
        log += f" CONCENTRATION({within:g})"
    section.log(log + ".")
    frame = pd.DataFrame({"a": numeric(ds, numerator), "s": numeric(ds, denominator)})
    if group:
        frame["g"] = ds.analysis_series(group)
    total = len(frame)
    valid = frame.dropna()
    valid = valid[valid["s"] != 0]
    excluded = total - len(valid)

    groups: list[tuple[str, pd.DataFrame]] = []
    if group:
        cats = sorted(valid["g"].unique(), key=lambda v: (isinstance(v, str), v),
                      reverse=order == "descending")
        groups = [(value_text(gvar, c), valid[valid["g"] == c]) for c in cats]
    cps = []
    if group:
        for i, (label, sub) in enumerate(groups):
            cps.append([gvar.title if i == 0 else "", label, fmt_int(len(sub)),
                        f"{100 * len(sub) / len(valid):.1f}%" if len(valid) else ""])
    cps.append(["Overall", "", fmt_int(len(valid)), "100.0%" if len(valid) else ""])
    cps.append(["Excluded", "", fmt_int(excluded), ""])
    cps.append(["Total", "", fmt_int(total), ""])
    section.table(Table("Case Processing Summary", ["", "", "Count", "Percent"], cps,
                        row_header_cols=2))
    if valid.empty:
        section.note("Warning", "No valid cases (missing values or zero denominators).",
                     style="warning")
        return section

    results = [(label, ratio_stats(sub["a"], sub["s"], level, low, high, within))
               for label, sub in groups] + \
        [("Overall", ratio_stats(valid["a"], valid["s"], level, low, high, within))]
    lv = f"{level * 100:g}%"
    spec: list[tuple[str, str, object]] = []
    want = set(stats_)
    if "mean" in want:
        spec.append(("Mean", "", "mean"))
        if ci:
            spec += [(f"{lv} Confidence Interval for Mean", "Lower Bound", ("mean_ci", 0)),
                     ("", "Upper Bound", ("mean_ci", 1))]
    if "median" in want:
        spec.append(("Median", "", "median"))
        if ci:
            spec += [(f"{lv} Confidence Interval for Median", "Lower Bound",
                      ("median_ci", 0)), ("", "Upper Bound", ("median_ci", 1)),
                     ("", "Actual Coverage", ("median_ci", 2))]
    if "wmean" in want:
        spec.append(("Weighted Mean", "", "wmean"))
        if ci:
            spec += [(f"{lv} Confidence Interval for Weighted Mean", "Lower Bound",
                      ("wmean_ci", 0)), ("", "Upper Bound", ("wmean_ci", 1))]
    for key, label in (("min", "Minimum"), ("max", "Maximum"), ("std", "Std. Deviation"),
                       ("range", "Range"), ("aad", "Average Absolute Deviation"),
                       ("prd", "Price Related Differential"),
                       ("cod", "Coefficient of Dispersion")):
        if key in want:
            spec.append((label, "", key))
    if "mdcov" in want:
        spec.append(("Coefficient of Variation", "Median Centered", "mdcov"))
    if "mncov" in want:
        spec.append(("" if "mdcov" in want else "Coefficient of Variation", "Mean Centered",
                     "mncov"))
    if low is not None and high is not None:
        spec.append(("Concentration Index", f"Coefficient of Concentration ({low:g} – "
                                             f"{high:g})", "between"))
    if within is not None:
        spec.append(("Concentration Index" if low is None else "",
                     f"Within {within:g}% of Median", "within"))
    if "prb" in want:
        spec.append(("Price Related Bias", "Value", ("prb", "value")))
        spec.append(("", "Std. Error", ("prb", "se")))
        if ci:
            spec += [("", f"{lv} CI Lower Bound", ("prb", "ci", 0)),
                     ("", f"{lv} CI Upper Bound", ("prb", "ci", 1))]

    def cell(res: dict, key) -> str:
        if res.get("n", 0) == 0:
            return ""
        if isinstance(key, tuple):
            if key[0] == "prb":
                v = res["prb"][key[1]]
                if len(key) == 3:
                    v = v[key[2]]
            else:
                v = res[key[0]][key[1]]
                if key == ("median_ci", 2):
                    return f"{100 * v:.1f}%"
        else:
            v = res.get(key)
            if key in ("cod", "mdcov", "mncov", "between", "within"):
                return f"{fmt(v, 1)}%" if v == v else ""
        return fmt(v, 3)

    rows = [[a, b] + [cell(res, key) for _, res in results] for a, b, key in spec]
    columns = ["", ""] + [label for label, _ in results] if group else ["", "", "Value"]
    if not group:
        rows = [r[:3] for r in rows]
    notes = []
    if ci and "median" in want:
        notes.append("The confidence interval for the median is constructed from binomial "
                     "order statistics; its actual coverage is at least the specified level.")
    if "prb" in want:
        notes.append("PRB: slope of (ratio − median)/median regressed on "
                     "log2(0.5 · numerator/median + 0.5 · denominator) (IAAO, 2013).")
    section.table(Table(f"Ratio Statistics for {nvar.title} / {dvar.title}", columns, rows,
                        row_header_cols=2, footnotes=notes))
    return section

"""Analyze > Descriptive Statistics: Descriptives and Frequencies.

Formulas follow SPSS: sample (n-1) variance, bias-corrected skewness (G1) and excess kurtosis
(G2) with their standard errors, and percentiles by the weighted average at (n+1)p (HAVERAGE).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from rsstatchem.core.dataset import (NUMERIC, STRING, Dataset, Variable, format_code,
                                     format_number)
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats import charts
from rsstatchem.core.stats.common import fmt, percentile, sup, value_text

DESCRIPTIVE_STATS = ("mean", "sum", "std", "var", "range", "min", "max", "sem", "kurt", "skew")
DEFAULT_DESCRIPTIVES = ("mean", "std", "min", "max")
FREQ_STATS = ("quartiles", "mean", "median", "mode", "sum", "std", "var", "range", "min",
              "max", "sem", "skew", "kurt")


class StatSpec(tuple):
    """The statistic keywords plus the extra sub-dialog options (percentiles, display order,
    format…). It is a plain tuple for callers that only know the keywords, which is how the
    Data Editor hands the Frequencies/Descriptives settings to :func:`frequencies` and
    :func:`descriptives`."""

    def __new__(cls, keys=(), options: dict | None = None):
        obj = super().__new__(cls, tuple(keys))
        obj.options = dict(options or {})
        return obj

    def __reduce__(self):
        return (StatSpec, (tuple(self), self.options))


def _options(stats) -> dict:
    return dict(getattr(stats, "options", {}) or {})


def se_skewness(n: int) -> float:
    if n < 3:
        return float("nan")
    return math.sqrt(6.0 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3)))


def se_kurtosis(n: int) -> float:
    if n < 4:
        return float("nan")
    return 2.0 * se_skewness(n) * math.sqrt((n * n - 1.0) / ((n - 3) * (n + 5)))


def summary(values: pd.Series) -> dict:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(x)
    out = {"n": n}
    if n == 0:
        return out
    s = pd.Series(x)
    out.update(
        mean=float(x.mean()), sum=float(x.sum()), min=float(x.min()), max=float(x.max()),
        range=float(x.max() - x.min()),
        std=float(s.std(ddof=1)) if n > 1 else float("nan"),
        var=float(s.var(ddof=1)) if n > 1 else float("nan"),
        skew=float(s.skew()) if n > 2 else float("nan"),
        kurt=float(s.kurt()) if n > 3 else float("nan"),
        se_skew=se_skewness(n), se_kurt=se_kurtosis(n),
    )
    out["sem"] = out["std"] / math.sqrt(n) if n > 1 else float("nan")
    out["median"] = float(np.median(x))
    out["p25"], out["p50"], out["p75"] = (
        float(v) for v in percentile(x, [25, 50, 75]))
    counts = s.value_counts()
    top = counts[counts == counts.max()].index.sort_values()
    out["mode"] = float(top[0])
    out["multimode"] = len(top) > 1
    return out


def _numeric_vars(ds: Dataset, names: list[str]) -> list[Variable]:
    return [ds.var(n) for n in names if ds.var(n).type == NUMERIC]


# ------------------------------------------------------------------ Descriptives
def descriptives(ds: Dataset, names: list[str], stats=DEFAULT_DESCRIPTIVES,
                 order: str | None = None) -> Section:
    """``order`` (or ``stats.options["order"]``): "variable" (list order), "alpha",
    "amean" (ascending means) or "dmean" (descending means), as SPSS Display Order."""
    order = order or _options(stats).get("order", "variable")
    variables = _numeric_vars(ds, names)
    if order == "alpha":
        variables.sort(key=lambda v: v.name.lower())
    elif order in ("amean", "dmean"):
        means = {v.name: summary(ds.analysis_series(v.name)).get("mean", float("nan"))
                 for v in variables}
        variables.sort(key=lambda v: (math.isnan(means[v.name]), means[v.name]),
                       reverse=order == "dmean")
        if order == "dmean":   # keep variables without a mean last
            variables.sort(key=lambda v: math.isnan(means[v.name]))
    stats = [s for s in DESCRIPTIVE_STATS if s in stats]
    section = Section("Descriptives")
    sort_kw = {"variable": "", "alpha": "\n  /SORT=NAME (A)", "amean": "\n  /SORT=MEAN (A)",
               "dmean": "\n  /SORT=MEAN (D)"}.get(order, "")
    section.log(f"DESCRIPTIVES VARIABLES={' '.join(v.name for v in variables)}\n"
                f"  /STATISTICS={' '.join(s.upper() for s in stats)}{sort_kw}.")
    if not variables:
        section.note("Warning", "No numeric variables were selected.", style="warning")
        return section

    order = ["range", "min", "max", "sum", "mean", "std", "var", "skew", "kurt"]
    grouped = any(s in stats for s in ("sem", "skew", "kurt"))
    names_map = {"range": "Range", "min": "Minimum", "max": "Maximum", "sum": "Sum",
                 "mean": "Mean", "std": "Std. Deviation", "var": "Variance",
                 "skew": "Skewness", "kurt": "Kurtosis"}
    columns: list = [""]
    columns.append(("N", "Statistic") if grouped else "N")
    for key in order:
        if key == "mean" and ("mean" in stats or "sem" in stats):
            if "mean" in stats:
                columns.append(("Mean", "Statistic") if grouped else "Mean")
            if "sem" in stats:
                columns.append(("Mean", "Std. Error"))
            continue
        if key in stats:
            if key in ("skew", "kurt"):
                columns += [(names_map[key], "Statistic"), (names_map[key], "Std. Error")]
            else:
                columns.append((names_map[key], "Statistic") if grouped else names_map[key])

    rows = []
    for var in variables:
        r = summary(ds.analysis_series(var.name))
        d = var.decimals
        cells = [var.title, str(r["n"])]
        if r["n"] == 0:
            rows.append(cells + [""] * (len(columns) - 2))
            continue
        for key in order:
            if key == "mean":
                if "mean" in stats:
                    cells.append(fmt(r["mean"], d + 2))
                if "sem" in stats:
                    cells.append(fmt(r["sem"], d + 2))
            elif key in stats:
                if key in ("range", "min", "max", "sum"):
                    cells.append(fmt(r[key], d))
                elif key in ("std", "var"):
                    cells.append(fmt(r[key], d + 3 if key == "var" else d + 2))
                else:
                    cells += [fmt(r[key], 3), fmt(r["se_" + key], 3)]
        rows.append(cells)

    listwise = pd.concat([ds.analysis_series(v.name) for v in variables], axis=1).dropna()
    rows.append(["Valid N (listwise)", str(len(listwise))] + [""] * (len(columns) - 2))
    section.table(Table("Descriptive Statistics", columns, rows))
    return section


def standardize(ds: Dataset, names: list[str]) -> list[str]:
    """Save z-scores as new variables (Z + name), like SPSS' 'Save standardized values'."""
    created = []
    for var in _numeric_vars(ds, names):
        x = pd.to_numeric(ds.analysis_series(var.name), errors="coerce")
        sd = x.std(ddof=1)
        z = (x - x.mean()) / sd if sd and not math.isnan(sd) else x * np.nan
        base = ("Z" + var.name)[:64]
        name, k = base, 1
        while ds.has_name(name):
            k += 1
            name = f"{base[:60]}_{k}"
        new = ds.add_variable(name, NUMERIC, values=z.to_numpy())
        new.decimals = 5
        new.label = f"Zscore({var.title})"
        created.append(name)
    return created


# ------------------------------------------------------------------ Frequencies
def grouped_percentile(values: np.ndarray, counts: np.ndarray, p: float) -> float:
    """Percentile when 'values are group midpoints' (SPSS FREQUENCIES /GROUPED): each value is
    the midpoint of an interval whose bounds lie halfway between adjacent values."""
    values = np.asarray(values, float)
    counts = np.asarray(counts, float)
    order = np.argsort(values)
    values, counts = values[order], counts[order]
    k = len(values)
    if k == 0:
        return float("nan")
    if k == 1:
        return float(values[0])
    mids = (values[1:] + values[:-1]) / 2
    lower = np.concatenate([[values[0] - (mids[0] - values[0])], mids])
    upper = np.concatenate([mids, [values[-1] + (values[-1] - mids[-1])]])
    total = counts.sum()
    target = p / 100 * total
    cum = np.cumsum(counts)
    i = int(np.searchsorted(cum, target, side="left"))
    i = min(i, k - 1)
    before = cum[i - 1] if i > 0 else 0.0
    if counts[i] == 0:
        return float(lower[i])
    return float(lower[i] + (target - before) / counts[i] * (upper[i] - lower[i]))


def _percentile_list(o: dict, stats) -> list[float]:
    ps: set[float] = set()
    if "quartiles" in stats:
        ps |= {25.0, 50.0, 75.0}
    n_cut = int(o.get("cut_points") or 0)
    if n_cut >= 2:
        ps |= {round(100.0 * i / n_cut, 10) for i in range(1, n_cut)}
    ps |= {float(p) for p in o.get("percentiles", []) if 0 <= float(p) <= 100}
    return sorted(ps)


def _statistics_table(ds: Dataset, variables: list[Variable], stats, o: dict | None = None,
                      title: str = "Statistics") -> Table:
    o = o or {}
    grouped = bool(o.get("grouped"))
    columns = ["", ""] + [v.title for v in variables]
    results = [summary(ds.analysis_series(v.name)) if v.type == NUMERIC else None
               for v in variables]
    valid = [int(ds.analysis_series(v.name).notna().sum()) for v in variables]
    rows = [["N", "Valid"] + [str(n) for n in valid],
            ["", "Missing"] + [str(ds.n_cases - n) for n in valid]]
    spec = [("mean", "Mean", 2), ("sem", "Std. Error of Mean", 3), ("median", "Median", 2),
            ("mode", "Mode", 0), ("std", "Std. Deviation", 3), ("var", "Variance", 3),
            ("skew", "Skewness", None), ("kurt", "Kurtosis", None), ("range", "Range", 0),
            ("min", "Minimum", 0), ("max", "Maximum", 0), ("sum", "Sum", 0)]
    footnotes = []
    grouped_values = {}
    if grouped:
        for var in variables:
            if var.type != NUMERIC:
                continue
            x = pd.to_numeric(ds.analysis_series(var.name), errors="coerce").dropna()
            vc = x.value_counts().sort_index()
            grouped_values[var.name] = (vc.index.to_numpy(float), vc.to_numpy(float))
    multi = "mode" in stats and any(r and r.get("multimode") for r in results)
    gmark = ""
    if grouped:
        gmark = "ᵇ" if multi else "ᵃ"
    for key, label, extra in spec:
        if key not in stats:
            continue
        cells, se_cells = [], []
        for var, r in zip(variables, results):
            if not r or r["n"] == 0:
                cells.append("")
                se_cells.append("")
                continue
            if extra is None:
                cells.append(fmt(r[key], 3))
                se_cells.append(fmt(r["se_" + key], 3))
            else:
                value = r[key]
                if key == "median" and grouped and var.name in grouped_values:
                    value = grouped_percentile(*grouped_values[var.name], 50.0)
                text = fmt(value, var.decimals + extra)
                if key == "mode" and r["multimode"]:
                    text += "ᵃ"
                    note = "a. Multiple modes exist. The smallest value is shown"
                    if note not in footnotes:
                        footnotes.insert(0, note)
                if key == "median" and grouped:
                    text += gmark
                cells.append(text)
        rows.append([label, ""] + cells)
        if extra is None:
            rows.append([f"Std. Error of {label}", ""] + se_cells)
    plist = _percentile_list(o, stats)
    for i, p in enumerate(plist):
        cells = []
        for v, r in zip(variables, results):
            if not r or not r["n"]:
                cells.append("")
                continue
            if grouped and v.name in grouped_values:
                value = grouped_percentile(*grouped_values[v.name], p)
            else:
                x = pd.to_numeric(ds.analysis_series(v.name), errors="coerce").dropna()
                value = float(percentile(x.to_numpy(float), [p])[0])
            pmark = ""
            if grouped:
                pmark = sup(chr(ord("a") + int(multi) + int("median" in stats)))
            cells.append(fmt(value, v.decimals + 2) + pmark)
        rows.append(["Percentiles" if i == 0 else "", f"{p:g}"] + cells)
    if grouped:
        letter = "b" if multi else "a"
        if "median" in stats:
            footnotes.append(f"{letter}. Calculated from grouped data.")
        if plist:
            footnotes.append(f"{chr(ord(letter) + ('median' in stats))}. Percentiles are "
                             "calculated from grouped data.")
    return Table(title, columns, rows, row_header_cols=2, footnotes=footnotes)


def _frequency_table(ds: Dataset, var: Variable, order: str = "avalue"
                     ) -> tuple[Table, list[str], list[float], int]:
    raw = ds.df[var.name]
    total = len(raw)
    clean = ds.analysis_series(var.name)
    empty = (raw == "") if var.type == STRING else pd.Series(False, index=raw.index)
    system_missing = int(raw.isna().sum() + empty.sum())
    user_missing = raw[raw.notna() & clean.isna() & ~empty]
    counts = clean.dropna().value_counts().sort_index()
    if order == "dvalue":
        counts = counts.iloc[::-1]
    elif order in ("afreq", "dfreq"):
        frame = pd.DataFrame({"v": counts.index, "n": counts.to_numpy()})
        frame["pos"] = range(len(frame))
        frame = frame.sort_values(["n", "pos"], ascending=[order == "afreq", True])
        counts = pd.Series(frame["n"].to_numpy(), index=frame["v"].to_numpy())
    n_valid = int(counts.sum())

    columns = ["", "", "Frequency", "Percent", "Valid Percent", "Cumulative Percent"]
    rows = []
    cumulative = 0.0
    labels, freqs = [], []
    for i, (value, count) in enumerate(counts.items()):
        pct = 100.0 * count / total if total else 0.0
        vpct = 100.0 * count / n_valid if n_valid else 0.0
        cumulative += vpct
        text = value_text(var, value)
        labels.append(text)
        freqs.append(float(count))
        rows.append(["Valid" if i == 0 else "", text, str(int(count)), fmt(pct, 1), fmt(vpct, 1),
                     fmt(cumulative, 1)])
    if n_valid:
        rows.append(["", "Total", str(n_valid), fmt(100.0 * n_valid / total, 1), "100.0", ""])
    n_missing = total - n_valid
    if n_missing:
        first = True
        for value, count in user_missing.value_counts().sort_index().items():
            rows.append(["Missing" if first else "", value_text(var, value), str(int(count)),
                         fmt(100.0 * count / total, 1), "", ""])
            first = False
        if system_missing:
            rows.append(["Missing" if first else "", "System", str(system_missing),
                         fmt(100.0 * system_missing / total, 1), "", ""])
        rows.append(["", "Total", str(n_missing), fmt(100.0 * n_missing / total, 1), "", ""])
    rows.append(["Total", "", str(total), "100.0" if total else "", "", ""])
    return Table(var.title, columns, rows, row_header_cols=2), labels, freqs, len(counts)


def frequencies(ds: Dataset, names: list[str], stats=(), show_tables: bool = True,
                chart: str = "none", normal_curve: bool = True, **options) -> Section:
    """FREQUENCIES. Extra options (keyword arguments or ``stats.options``): ``percentiles``
    (list), ``cut_points`` (n equal groups), ``grouped`` (values are group midpoints),
    ``chart_values`` ("frequencies" | "percentages"), ``order`` ("avalue" | "dvalue" |
    "afreq" | "dfreq"), ``organize`` ("compare" | "variable") and ``limit`` (suppress tables
    with more than n categories)."""
    o = _options(stats)
    o.update(options)
    variables = [ds.var(n) for n in names]
    section = Section("Frequencies")
    order = o.get("order", "avalue")
    limit = o.get("limit")
    log = f"FREQUENCIES VARIABLES={' '.join(v.name for v in variables)}"
    if not show_tables:
        log += "\n  /FORMAT=NOTABLE"
    elif order != "avalue" or limit:
        log += f"\n  /FORMAT={order.upper()}" + (f" LIMIT({int(limit)})" if limit else "")
    if o.get("cut_points"):
        log += f"\n  /NTILES={int(o['cut_points'])}"
    if o.get("percentiles"):
        log += "\n  /PERCENTILES=" + " ".join(f"{float(p):g}" for p in o["percentiles"])
    keywords = [s.upper() for s in stats if s != "quartiles"]
    if "quartiles" in stats:
        log += "\n  /NTILES=4"
    if keywords:
        log += f"\n  /STATISTICS={' '.join(keywords)}"
    if chart != "none":
        kind = "HISTOGRAM" + (" NORMAL" if normal_curve else "") if chart == "histogram" \
            else f"{chart.upper()}CHART " + ("PERCENT" if o.get("chart_values") ==
                                             "percentages" else "FREQ")
        log += f"\n  /{kind}"
    if o.get("grouped"):
        log += f"\n  /GROUPED={' '.join(v.name for v in variables)}"
    if o.get("organize") == "variable":
        log += "\n  /ORDER=VARIABLE"
    section.log(log + ".")
    if not variables:
        return section

    by_variable = o.get("organize") == "variable" and len(variables) > 1
    if not by_variable:
        section.table(_statistics_table(ds, variables, stats, o))
    tables = [_frequency_table(ds, v, order) for v in variables]
    for var, (table, labels, freqs, n_cats) in zip(variables, tables):
        if by_variable:
            section.table(_statistics_table(ds, [var], stats, o))
        if show_tables and not (limit and n_cats > int(limit)):
            section.table(table)
        if by_variable:
            _chart(section, ds, var, labels, freqs, chart, normal_curve, o)
    if not by_variable:
        for var, (_, labels, freqs, _) in zip(variables, tables):
            _chart(section, ds, var, labels, freqs, chart, normal_curve, o)
    return section


def _chart(section, ds, var, labels, freqs, chart, normal_curve, o) -> None:
    percent = o.get("chart_values") == "percentages"
    values = freqs
    if percent and freqs:
        total = sum(freqs)
        values = [100.0 * f / total for f in freqs]
    if chart == "bar" and freqs:
        section.image(var.title, charts.bar_chart(labels, values, var.title,
                                                  "Percent" if percent else "Frequency"))
    elif chart == "pie" and freqs:
        section.image(var.title, charts.pie_chart(labels, values, var.title))
    elif chart == "histogram" and var.type == NUMERIC:
        x = pd.to_numeric(ds.analysis_series(var.name), errors="coerce").dropna()
        if len(x):
            section.image(var.title, charts.histogram(x.to_numpy(), var.title, var.title,
                                                      normal_curve))


__all__ = ["descriptives", "frequencies", "standardize", "summary", "format_code", "StatSpec",
           "grouped_percentile"]

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
from rsstatchem.core.stats.common import fmt, percentile, value_text

DESCRIPTIVE_STATS = ("mean", "sum", "std", "var", "range", "min", "max", "sem", "kurt", "skew")
DEFAULT_DESCRIPTIVES = ("mean", "std", "min", "max")
FREQ_STATS = ("quartiles", "mean", "median", "mode", "sum", "std", "var", "range", "min",
              "max", "sem", "skew", "kurt")


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
def descriptives(ds: Dataset, names: list[str], stats=DEFAULT_DESCRIPTIVES) -> Section:
    variables = _numeric_vars(ds, names)
    stats = [s for s in DESCRIPTIVE_STATS if s in stats]
    section = Section("Descriptives")
    section.log(f"DESCRIPTIVES VARIABLES={' '.join(v.name for v in variables)}\n"
                f"  /STATISTICS={' '.join(s.upper() for s in stats)}.")
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
def _statistics_table(ds: Dataset, variables: list[Variable], stats) -> Table:
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
    for key, title, extra in spec:
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
                text = fmt(r[key], var.decimals + extra)
                if key == "mode" and r["multimode"]:
                    text += "ᵃ"
                    note = "a. Multiple modes exist. The smallest value is shown"
                    if note not in footnotes:
                        footnotes.append(note)
                cells.append(text)
        rows.append([title, ""] + cells)
        if extra is None:
            rows.append([f"Std. Error of {title}", ""] + se_cells)
    if "quartiles" in stats:
        for i, (p, key) in enumerate(((25, "p25"), (50, "p50"), (75, "p75"))):
            rows.append(["Percentiles" if i == 0 else "", str(p)] +
                        [fmt(r[key], v.decimals + 2) if r and r["n"] else ""
                         for v, r in zip(variables, results)])
    return Table("Statistics", columns, rows, row_header_cols=2, footnotes=footnotes)


def _frequency_table(ds: Dataset, var: Variable) -> tuple[Table, list[str], list[float]]:
    raw = ds.df[var.name]
    total = len(raw)
    clean = ds.analysis_series(var.name)
    empty = (raw == "") if var.type == STRING else pd.Series(False, index=raw.index)
    system_missing = int(raw.isna().sum() + empty.sum())
    user_missing = raw[raw.notna() & clean.isna() & ~empty]
    counts = clean.dropna().value_counts().sort_index()
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
    return Table(var.title, columns, rows, row_header_cols=2), labels, freqs


def frequencies(ds: Dataset, names: list[str], stats=(), show_tables: bool = True,
                chart: str = "none", normal_curve: bool = True) -> Section:
    variables = [ds.var(n) for n in names]
    section = Section("Frequencies")
    log = f"FREQUENCIES VARIABLES={' '.join(v.name for v in variables)}"
    if stats:
        log += f"\n  /STATISTICS={' '.join(s.upper() for s in stats)}"
    if chart != "none":
        log += f"\n  /{chart.upper()}"
    section.log(log + ("" if show_tables else "\n  /FORMAT=NOTABLE") + ".")
    if not variables:
        return section

    section.table(_statistics_table(ds, variables, stats))
    tables = [_frequency_table(ds, v) for v in variables]
    if show_tables:
        for table, _, _ in tables:
            section.table(table)

    for var, (_, labels, freqs) in zip(variables, tables):
        if chart == "bar" and freqs:
            section.image(var.title, charts.bar_chart(labels, freqs, var.title))
        elif chart == "pie" and freqs:
            section.image(var.title, charts.pie_chart(labels, freqs, var.title))
        elif chart == "histogram" and var.type == NUMERIC:
            values = pd.to_numeric(ds.analysis_series(var.name), errors="coerce").dropna()
            if len(values):
                section.image(var.title, charts.histogram(values.to_numpy(), var.title,
                                                          var.title, normal_curve))
    return section


__all__ = ["descriptives", "frequencies", "standardize", "summary", "format_code"]

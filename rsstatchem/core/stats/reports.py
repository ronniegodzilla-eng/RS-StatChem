"""SPSS Analyze > Reports and File > Display Data File Information.

* :func:`codebook` (CODEBOOK): dictionary information plus counts/percentages for categorical
  variables and mean/SD/quartiles for scale variables;
* :func:`olap_cubes` (OLAP CUBES): sum, N, mean, SD, % of total sum and N by grouping
  variables (with a Total row per grouping level);
* :func:`case_summaries` (SUMMARIZE): case listing (optionally limited) and statistics by
  grouping variables;
* :func:`report_rows` / :func:`report_columns` (REPORT): summary statistics per break group in
  rows or in columns;
* :func:`file_info` (DISPLAY DICTIONARY): Variable Information and Variable Values tables.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from rsstatchem.core.dataset import DATE, NUMERIC, SCALE, STRING, Dataset, format_code
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, percentile, value_text

STATS = {"n": "N", "sum": "Sum", "mean": "Mean", "median": "Median", "std": "Std. Deviation",
         "var": "Variance", "min": "Minimum", "max": "Maximum", "range": "Range",
         "se": "Std. Error of Mean", "first": "First", "last": "Last",
         "pct_sum": "% of Total Sum", "pct_n": "% of Total N", "skew": "Skewness",
         "kurt": "Kurtosis", "gmean": "Geometric Mean", "hmean": "Harmonic Mean"}


def _format_spec(var) -> str:
    if var.type == STRING:
        return f"A{var.width}"
    if var.type == DATE:
        return "DATE11"
    return f"F{var.width}.{var.decimals}"


def _stat(x: np.ndarray, stat: str, total_sum: float = np.nan, total_n: float = np.nan,
          raw: np.ndarray | None = None) -> float:
    x = x[~np.isnan(x)]
    n = len(x)
    if stat == "n":
        return float(n)
    if stat == "pct_n":
        return 100 * n / total_n if total_n else float("nan")
    if n == 0:
        return float("nan")
    if stat == "sum":
        return float(x.sum())
    if stat == "pct_sum":
        return 100 * float(x.sum()) / total_sum if total_sum else float("nan")
    if stat == "mean":
        return float(x.mean())
    if stat == "median":
        return float(np.median(x))
    if stat == "min":
        return float(x.min())
    if stat == "max":
        return float(x.max())
    if stat == "range":
        return float(x.max() - x.min())
    if stat == "first":
        return float(x[0])
    if stat == "last":
        return float(x[-1])
    if stat == "gmean":
        return float(np.exp(np.log(x).mean())) if (x > 0).all() else float("nan")
    if stat == "hmean":
        return float(n / (1 / x).sum()) if (x > 0).all() else float("nan")
    if n < 2:
        return float("nan")
    if stat == "std":
        return float(x.std(ddof=1))
    if stat == "var":
        return float(x.var(ddof=1))
    if stat == "se":
        return float(x.std(ddof=1) / math.sqrt(n))
    if stat in ("skew", "kurt"):
        from scipy import stats as sps
        if stat == "skew":
            return float(sps.skew(x, bias=False)) if n > 2 else float("nan")
        return float(sps.kurtosis(x, bias=False)) if n > 3 else float("nan")
    raise ValueError(stat)


def _fmt_stat(stat: str, value: float, decimals: int) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if stat == "n":
        return fmt_int(value)
    if stat in ("pct_sum", "pct_n"):
        return f"{value:.1f}%"
    return fmt(value, decimals)


def _groups(ds: Dataset, by: list[str]) -> list[tuple[tuple, np.ndarray]]:
    """(labels per grouping variable, row mask) for every observed combination."""
    if not by:
        return [((), np.ones(ds.n_cases, dtype=bool))]
    series = [ds.analysis_series(b) for b in by]
    frame = pd.concat(series, axis=1, keys=range(len(by)))
    out = []
    keys = frame.dropna().drop_duplicates()
    try:
        keys = keys.sort_values(list(keys.columns))
    except TypeError:
        pass
    for _, key in keys.iterrows():
        mask = np.ones(ds.n_cases, dtype=bool)
        labels = []
        for i, b in enumerate(by):
            mask &= (series[i] == key[i]).to_numpy()
            labels.append(value_text(ds.var(b), key[i]))
        out.append((tuple(labels), mask))
    return out


def _numbers(ds: Dataset, name: str) -> np.ndarray:
    return pd.to_numeric(ds.analysis_series(name), errors="coerce").to_numpy(float)


# ------------------------------------------------------------------ codebook
def codebook(ds: Dataset, names: list[str], max_categories: int = 200,
             show_counts: bool = True, show_percents: bool = True,
             scale_stats: tuple = ("mean", "std", "p25", "p50", "p75")) -> Section:
    section = Section("Codebook")
    section.log("CODEBOOK " + " ".join(names) +
                "\n  /VARINFO POSITION LABEL TYPE FORMAT MEASURE ROLE VALUELABELS MISSING"
                "\n  /STATISTICS COUNT PERCENT MEAN STDDEV QUARTILES.")
    n = ds.n_cases
    for name in names:
        var = ds.var(name)
        rows = [["Standard Attributes", "Position", "", fmt_int(ds.index_of(name) + 1)],
                ["", "Label", "", var.label or "<none>"],
                ["", "Type", "", var.type.capitalize()],
                ["", "Format", "", _format_spec(var)],
                ["", "Measurement", "", var.measure.capitalize()],
                ["", "Role", "", var.role.capitalize()]]
        if var.missing:
            rows.append(["", "Missing Values", "", ", ".join(format_code(m) for m in var.missing)])
        s = ds.analysis_series(name)
        raw = ds.df[var.name]
        n_valid = int(s.notna().sum())
        rows.append(["N", "Valid", "", fmt_int(n_valid)])
        rows.append(["", "Missing", "", fmt_int(n - n_valid)])
        if var.measure == SCALE and var.type == NUMERIC:
            x = _numbers(ds, name)
            xv = x[~np.isnan(x)]
            labels = {"mean": "Mean", "std": "Standard Deviation", "p25": "Percentile 25",
                      "p50": "Percentile 50", "p75": "Percentile 75"}
            for k, st in enumerate(scale_stats):
                if st.startswith("p") and st[1:].isdigit():
                    value = float(percentile(xv, int(st[1:]))) if len(xv) else float("nan")
                else:
                    value = _stat(x, st)
                rows.append(["Central Tendency and Dispersion" if k == 0 else "",
                             labels.get(st, st), "", fmt(value, var.decimals + 2)])
            if var.value_labels:
                for k, (code, label) in enumerate(var.value_labels.items()):
                    rows.append(["Labeled Values" if k == 0 else "", format_code(code), label,
                                 ""])
        else:
            counts = raw.value_counts(dropna=False)
            codes = list(var.value_labels) + [c for c in counts.index
                                              if c not in var.value_labels and not pd.isna(c)
                                              and c != ""]
            try:
                codes = sorted(set(codes))
            except TypeError:
                codes = sorted(set(codes), key=str)
            for k, code in enumerate(codes[:max_categories]):
                count = int((raw == code).sum())
                pct = 100 * count / n if n else float("nan")
                label = var.label_for(code) or ""
                tag = "ᵃ" if code in var.missing else ""
                rows.append(["Valid Values" if k == 0 else "", format_code(code) + tag, label,
                             (fmt_int(count) if show_counts else "")
                             + ("  " if show_counts and show_percents else "")
                             + (f"{pct:.1f}%" if show_percents else "")])
            sysmis = int(raw.isna().sum()) + (int((raw == "").sum()) if var.type == STRING
                                              else 0)
            if sysmis:
                rows.append(["Missing Values", "System", "",
                             f"{sysmis}  {100 * sysmis / n:.1f}%" if n else str(sysmis)])
        notes = ["a. Missing value"] if var.missing else []
        section.table(Table(f"{var.name}" + (f" ({var.label})" if var.label else ""),
                            ["", "", "", "Value" if var.measure == SCALE else "Count / Percent"],
                            rows, row_header_cols=3, footnotes=notes))
    return section


# ------------------------------------------------------------------ OLAP cubes
def olap_cubes(ds: Dataset, summaries: list[str], groups: list[str],
               stats: tuple = ("sum", "n", "mean", "std", "pct_sum", "pct_n")) -> Section:
    section = Section("OLAP Cubes")
    section.log("OLAP CUBES " + " ".join(summaries) + (" BY " + " BY ".join(groups)
                                                        if groups else "")
                + "\n  /CELLS=" + " ".join(s.upper() for s in stats) + ".")
    total_mask = np.ones(ds.n_cases, dtype=bool)
    for g in groups:
        total_mask &= ds.analysis_series(g).notna().to_numpy()
    case_rows = []
    for s in summaries:
        x = _numbers(ds, s)
        included = int(np.sum(total_mask & ~np.isnan(x)))
        case_rows.append([ds.var(s).title, fmt_int(included),
                          f"{100 * included / ds.n_cases:.1f}%" if ds.n_cases else "",
                          fmt_int(ds.n_cases - included),
                          f"{100 * (ds.n_cases - included) / ds.n_cases:.1f}%"
                          if ds.n_cases else ""])
    section.table(Table("Case Processing Summary",
                        ["", ("Included", "N"), ("Included", "Percent"), ("Excluded", "N"),
                         ("Excluded", "Percent")], case_rows))
    for s in summaries:
        var = ds.var(s)
        x = _numbers(ds, s)
        valid_all = total_mask & ~np.isnan(x)
        total_sum = float(np.nansum(x[valid_all]))
        total_n = float(valid_all.sum())
        rows = []
        for labels, mask in (_groups(ds, groups) if groups else []):
            vals = x[mask & total_mask]
            rows.append(list(labels) + [_fmt_stat(st, _stat(vals, st, total_sum, total_n),
                                                  var.decimals + 2) for st in stats])
        totals = [_fmt_stat(st, _stat(x[total_mask], st, total_sum, total_n), var.decimals + 2)
                  for st in stats]
        rows.append(["Total"] + [""] * max(len(groups) - 1, 0) + totals)
        cols = [ds.var(g).title for g in groups] or [""]
        cols += [STATS[st] for st in stats]
        section.table(Table(f"OLAP Cubes: {var.title}", cols, rows,
                            row_header_cols=max(len(groups), 1)))
    return section


# ------------------------------------------------------------------ case summaries
def case_summaries(ds: Dataset, names: list[str], groups: list[str] | None = None,
                   display_cases: bool = True, limit: int | None = 100,
                   valid_only: bool = False, case_numbers: bool = True,
                   stats: tuple = ("n",), title: str = "Case Summaries") -> Section:
    groups = groups or []
    section = Section("Summarize")
    section.log("SUMMARIZE\n  /TABLES=" + " ".join(names) +
                (" BY " + " BY ".join(groups) if groups else "") +
                f"\n  /FORMAT={'VALIDLIST' if valid_only else 'LIST'} "
                f"{'CASENUM' if case_numbers else 'NOCASENUM'} TOTAL"
                + (f" LIMIT={limit}" if limit else "") +
                f"\n  /CELLS={' '.join(s.upper() for s in stats)}.")
    n = ds.n_cases
    proc = []
    for name in names:
        valid = int(ds.analysis_series(name).notna().sum())
        proc.append([ds.var(name).title, fmt_int(valid), f"{100 * valid / n:.1f}%" if n else "",
                     fmt_int(n - valid), f"{100 * (n - valid) / n:.1f}%" if n else "",
                     fmt_int(n), "100.0%"])
    section.table(Table("Case Processing Summary",
                        ["", ("Included", "N"), ("Included", "Percent"), ("Excluded", "N"),
                         ("Excluded", "Percent"), ("Total", "N"), ("Total", "Percent")], proc))
    rows = []
    rh = len(groups) + 1
    for labels, mask in _groups(ds, groups):
        idx = np.flatnonzero(mask)
        if valid_only:
            ok = np.ones(n, dtype=bool)
            for name in names:
                ok &= ds.analysis_series(name).notna().to_numpy()
            idx = np.flatnonzero(mask & ok)
        if display_cases:
            shown = idx[:limit] if limit else idx
            for k, i in enumerate(shown):
                head = list(labels) if k == 0 else [""] * len(groups)
                values = []
                for name in names:
                    v = ds.df[ds.var(name).name].iloc[i]
                    values.append("" if pd.isna(v) else value_text(ds.var(name), v))
                rows.append(head + [str(i + 1) if case_numbers else str(k + 1)] + values)
        for st in stats:
            stat_row = []
            for name in names:
                var = ds.var(name)
                if var.type == NUMERIC:
                    stat_row.append(_fmt_stat(st, _stat(_numbers(ds, name)[mask], st),
                                              var.decimals + 2))
                elif st == "n":
                    stat_row.append(fmt_int(ds.analysis_series(name)[mask].notna().sum()))
                else:
                    stat_row.append("")
            head = (list(labels) if not display_cases else [""] * len(groups)) if groups \
                else []
            label = "Total " + STATS[st] if display_cases else STATS[st]
            rows.append(head + [label] + stat_row)
    if groups:
        for st in stats:
            stat_row = []
            for name in names:
                var = ds.var(name)
                if var.type == NUMERIC:
                    stat_row.append(_fmt_stat(st, _stat(_numbers(ds, name), st),
                                              var.decimals + 2))
                else:
                    stat_row.append(fmt_int(ds.analysis_series(name).notna().sum())
                                    if st == "n" else "")
            rows.append(["Total"] + [""] * (len(groups) - 1) + [STATS[st]] + stat_row)
    cols = [ds.var(g).title for g in groups] + [""] + [ds.var(nm).title for nm in names]
    notes = [f"a. Limited to first {limit} cases."] if display_cases and limit and \
        any(len(np.flatnonzero(m)) > limit for _, m in _groups(ds, groups)) else []
    section.table(Table(title + ("ᵃ" if notes else ""), cols, rows, row_header_cols=rh,
                        footnotes=notes))
    return section


# ------------------------------------------------------------------ report summaries
def report_rows(ds: Dataset, columns: list[str], breaks: list[str],
                stats: tuple = ("mean",), list_cases: bool = False,
                title: str = "Report") -> Section:
    """Report Summaries in Rows: one block per break group, summary lines under it."""
    section = Section("Report")
    section.log(f"REPORT FORMAT={'LIST' if list_cases else 'AUTOMATIC'}\n  /VARIABLES="
                + " ".join(columns) + "\n  /BREAK=" + " ".join(breaks) +
                "\n  /SUMMARY=" + " ".join(s.upper() for s in stats) + ".")
    rows = []
    for labels, mask in _groups(ds, breaks):
        if list_cases:
            for k, i in enumerate(np.flatnonzero(mask)):
                rows.append((list(labels) if k == 0 else [""] * len(breaks)) + [""] +
                            [value_text(ds.var(c), ds.df[ds.var(c).name].iloc[i])
                             if not pd.isna(ds.df[ds.var(c).name].iloc[i]) else ""
                             for c in columns])
        for k, st in enumerate(stats):
            head = list(labels) if (k == 0 and not list_cases) else [""] * len(breaks)
            rows.append(head + [STATS[st]] + [
                _fmt_stat(st, _stat(_numbers(ds, c)[mask], st), ds.var(c).decimals + 2)
                for c in columns])
    for k, st in enumerate(stats):
        rows.append((["Grand Total"] if k == 0 else [""]) + [""] * (len(breaks) - 1) +
                    [STATS[st]] + [_fmt_stat(st, _stat(_numbers(ds, c), st),
                                             ds.var(c).decimals + 2) for c in columns])
    cols = [ds.var(b).title for b in breaks] + [""] + [ds.var(c).title for c in columns]
    section.table(Table(title, cols, rows, row_header_cols=len(breaks) + 1))
    return section


def report_columns(ds: Dataset, columns: list[tuple[str, str]], breaks: list[str],
                   grand_total: bool = True, title: str = "Report") -> Section:
    """Report Summaries in Columns: each data column is (variable, statistic)."""
    section = Section("Report")
    section.log("REPORT FORMAT=AUTOMATIC\n  /VARIABLES=" + " ".join(
        f"{v} ({s.upper()})" for v, s in columns) + "\n  /BREAK=" + " ".join(breaks) + ".")
    rows = []
    for labels, mask in _groups(ds, breaks):
        rows.append(list(labels) + [_fmt_stat(s, _stat(_numbers(ds, v)[mask], s),
                                              ds.var(v).decimals + 2) for v, s in columns])
    if grand_total:
        rows.append(["Grand Total"] + [""] * (len(breaks) - 1) +
                    [_fmt_stat(s, _stat(_numbers(ds, v), s), ds.var(v).decimals + 2)
                     for v, s in columns])
    cols = [ds.var(b).title for b in breaks] + [
        (ds.var(v).title, STATS[s]) for v, s in columns]
    section.table(Table(title, cols, rows, row_header_cols=max(len(breaks), 1)))
    return section


# ------------------------------------------------------------------ file information
def file_info(ds: Dataset, source: str = "working file") -> Section:
    """DISPLAY DICTIONARY: variable information and value labels."""
    section = Section("File Information")
    section.log("DISPLAY DICTIONARY." if source == "working file"
                else f"SYSFILE INFO FILE='{source}'.")
    rows = []
    for i, v in enumerate(ds.variables, 1):
        missing = ", ".join(format_code(m) for m in v.missing) if v.missing else ""
        rows.append([v.name, str(i), v.label or "", v.measure.capitalize(), v.role.capitalize(),
                     str(v.columns), v.align.capitalize(), _format_spec(v), _format_spec(v),
                     missing])
    section.table(Table("Variable Information",
                        ["Variable", "Position", "Label", "Level of Measurement", "Role",
                         "Column Width", "Alignment", "Print Format", "Write Format",
                         "Missing Values"], rows,
                        footnotes=[f"Source: {source}. Cases: {ds.n_cases}. "
                                   f"Variables: {ds.n_vars}."]))
    labels = []
    for v in ds.variables:
        for k, (code, label) in enumerate(v.value_labels.items()):
            labels.append([v.name if k == 0 else "", format_code(code)
                           + ("ᵃ" if code in v.missing else ""), label])
    if labels:
        section.table(Table("Variable Values", ["Variable", "Value", "Label"], labels,
                            row_header_cols=2,
                            footnotes=["a. Missing value"] if any(
                                v.missing for v in ds.variables) else []))
    comments = ds.meta.get("comments")
    if comments:
        section.note("Data File Comments", comments)
    return section

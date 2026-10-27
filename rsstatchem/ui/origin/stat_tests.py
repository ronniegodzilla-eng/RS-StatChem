"""Statistics menu analyses with Origin-style dialogs and report sheets (no Qt).

Each analysis takes the workbooks (``books``: name -> Workbook) and a ``settings`` dict as
collected by its dialog (:mod:`rsstatchem.ui.origin.stat_dialogs`): input ranges in Origin's
notation (``[Book1]Sheet1!B`` or ``[Book1]Sheet1!B[3:10]``) and the options. It writes a
report sheet into the book of the first input — Descriptive Statistics, Normality Test,
Correlation, t-tests, ANOVA with means comparisons, nonparametric tests — and returns
``(report, results-log text)``. The report remembers the analysis in ``report.op`` so it can
be recalculated when the source data change (:func:`run` with ``report=``). Statistics on
Rows and Frequency Counts write worksheet columns instead and record themselves in the
sheet's ``ops`` (Origin's lock on output columns) so they can be recalculated too. Report
plots are live graph objects (:mod:`rsstatchem.ui.origin.report_graphs`).

Numbers come from scipy / statsmodels and, for the means comparisons, from
:func:`rsstatchem.core.stats.anova.posthoc` (Tukey, Bonferroni, Dunn-Šidák, Scheffé, Fisher
LSD) plus Dunnett's many-to-one test against a chosen control level — two-sided, "< control"
or "> control" with one-sided bounds (multivariate t, as scipy.stats.dunnett) — and Holm's
step-down Holm-Bonferroni / Holm-Šidák procedures. Two-Way ANOVA takes raw (one column per
level combination) or indexed data, balanced or not, with Type I / II / III sums of squares.
The Chen-Shapiro normality test uses the QH* statistic of Chen & Shapiro (1995) with a
Monte Carlo p-value.
"""
from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from scipy import stats

from rsstatchem.core.analysis.common import num, pval
from rsstatchem.core.stats.anova import posthoc
from rsstatchem.core.stats.common import Group
from rsstatchem.ui.origin import report_graphs as rg
from rsstatchem.ui.origin.model import (ReportSheet, Workbook, Worksheet, jsonable,
                                        parse_range, row_window, rows_text, to_float)


class StatError(ValueError):
    pass


# ====================================================================== inputs
@dataclass
class RangeData:
    """One input range resolved against the project."""
    book: Workbook
    sheet: Worksheet
    col: int
    rows: tuple[int, int] | None
    values: np.ndarray            # numeric values of the range (NaN = missing / text)
    texts: list[str]              # the same cells as text

    @property
    def column(self):
        return self.sheet.columns[self.col]

    @property
    def title(self) -> str:
        return self.column.title

    @property
    def text(self) -> str:
        c = self.column
        return f"[{self.book.name}]{self.sheet.name}!{c.name}" + \
            (f'"{c.long_name}"' if c.long_name else "") + \
            rows_text(*(self.rows or (None, None)))

    @property
    def finite(self) -> np.ndarray:
        return self.values[np.isfinite(self.values)]

    @property
    def n_missing(self) -> int:
        return int(np.sum(~np.isfinite(self.values)))


def read_range(books: dict, text: str, book: str = "", sheet: str = "") -> RangeData:
    try:
        ref = parse_range(text, book, sheet)
    except ValueError as exc:
        raise StatError(str(exc)) from None
    wb = books.get(ref.book)
    if not isinstance(wb, Workbook):
        raise StatError(f"There is no workbook {ref.book or '(none)'} for the range {text}.")
    ws = wb.sheet(ref.sheet) if ref.sheet else wb.active_sheet()
    if not isinstance(ws, Worksheet):
        raise StatError(f"There is no worksheet {ref.sheet} in {ref.book}.")
    try:
        col = ws.find(ref.col)
    except KeyError:
        raise StatError(f"There is no column {ref.col} in [{ref.book}]{ws.name}.") from None
    rows = ref.rows
    n = max(ws.data_rows(), 1)
    lo, hi = row_window(n, rows)
    hi = max(hi, lo)
    c = ws.columns[col]
    values = c.numeric(max(hi, 1))[lo:hi]
    texts = c.texts(max(hi, 1))[lo:hi]
    masked = _masked_rows(ws, col)
    if masked:   # masked cells are left out of the analysis (Origin's mask)
        values = np.array(values, float)
        texts = list(texts)
        for r in masked:
            if lo <= r < hi:
                values[r - lo] = np.nan
                texts[r - lo] = ""
    return RangeData(wb, ws, col, (lo + 1, hi) if rows else None, values, texts)


def _masked_rows(ws: Worksheet, col: int) -> set[int]:
    """0-based masked rows of a column through the worksheet mask API
    (``Worksheet.masked_rows(col)``) when the model provides it."""
    fn = getattr(ws, "masked_rows", None)
    if fn is None:
        return set()
    try:
        return {int(r) for r in (fn(col) or ())}
    except (TypeError, ValueError, KeyError, IndexError, AttributeError):
        return set()


def read_ranges(books: dict, texts: list[str]) -> list[RangeData]:
    out = [read_range(books, t) for t in texts if t and t.strip()]
    if not out:
        raise StatError("Select the input data (Input > Range).")
    return out


def _stamp(book: Workbook) -> str:
    return f'[{datetime.now():%Y-%m-%d %H:%M} "/{book.name}"]'


def _new_report(first: RangeData, title: str, base: str, report: ReportSheet | None,
                key: str, settings: dict, mode: str) -> tuple[ReportSheet, ReportSheet]:
    """(target report, fresh report to fill). The target is the existing report on a
    recalculation, else a new sheet added to the first input's book."""
    fresh = ReportSheet(title=title)
    if report is None:
        report = fresh
        report.name = first.book.unique_sheet_name(base)
        first.book.sheets.append(report)
    fresh.op = {"kind": "stats", "test": key, "settings": jsonable(settings),
                "mode": report.op.get("mode", mode) if report.op else mode, "dirty": False}
    return report, fresh


def _finish(report: ReportSheet, fresh: ReportSheet) -> ReportSheet:
    if report is not fresh:
        report.replace_items(fresh)
        report.op = fresh.op
    return report


def _notes(fresh: ReportSheet, description: str, inputs: list[RangeData], extra=()) -> None:
    fresh.add_table("Notes", ["", "Value"], [["Description", description]] +
                    [["Input Data", ", ".join(r.text for r in inputs)]] + list(extra))


def _decision(p: float, alpha: float, yes: str, no: str) -> str:
    if not np.isfinite(p):
        return "--"
    return yes if p < alpha else no


def _level(alpha: float) -> str:
    return f"{alpha * 100:g}%"


def _sort_levels(values: list[str]) -> list[str]:
    def key(v):
        f = to_float(v)
        return (0, f, "") if np.isfinite(f) else (1, 0.0, v)
    return sorted(set(values), key=key)


def indexed_groups(data: RangeData, factor: RangeData) -> list[tuple[str, np.ndarray]]:
    """Groups of an indexed data set: (level, values) sorted by level."""
    n = min(len(data.values), len(factor.texts))
    labels = factor.texts[:n]
    vals = data.values[:n]
    ok = [i for i in range(n) if labels[i] != "" and np.isfinite(vals[i])]
    levels = _sort_levels([labels[i] for i in ok])
    return [(lv, np.array([vals[i] for i in ok if labels[i] == lv], float)) for lv in levels]


def _groups(books: dict, settings: dict, minimum: int = 2) -> tuple[list[RangeData],
                                                                   list[tuple[str, np.ndarray]]]:
    """Groups for t-tests / ANOVA / nonparametric tests from raw (one column per group) or
    indexed (data + group column) input."""
    if settings.get("form", "raw") == "indexed":
        data = read_range(books, settings.get("data", ""))
        factor = read_range(books, settings.get("factor", ""))
        groups = indexed_groups(data, factor)
        inputs = [data, factor]
    else:
        inputs = read_ranges(books, settings.get("ranges", []))
        groups = [(r.title, r.finite) for r in inputs]
    if len(groups) < minimum:
        raise StatError(f"This test needs at least {minimum} groups (got {len(groups)}).")
    for name, v in groups:
        if len(v) < 2:
            raise StatError(f"Group {name} needs at least 2 values.")
    return inputs, groups


# ====================================================================== descriptive statistics
DESC_QUANTITIES = {
    "n": "N total", "missing": "N missing", "mean": "Mean", "sd": "Standard Deviation",
    "se": "SE of mean", "lcl": "Lower 95% CI of Mean", "ucl": "Upper 95% CI of Mean",
    "var": "Variance", "sum": "Sum", "skew": "Skewness", "kurt": "Kurtosis",
    "uss": "Uncorrected Sum of Squares", "css": "Corrected Sum of Squares",
    "cv": "Coefficient of Variation", "mad": "Mean Absolute Deviation",
    "gmean": "Geometric Mean", "hmean": "Harmonic Mean", "mode": "Mode",
    "min": "Minimum", "q1": "1st Quartile (Q1)", "median": "Median",
    "q3": "3rd Quartile (Q3)", "max": "Maximum", "range": "Range",
    "iqr": "Interquartile Range (Q3-Q1)", "imin": "Index of Minimum",
    "imax": "Index of Maximum",
}
DESC_DEFAULT = ("n", "missing", "mean", "sd", "se", "lcl", "ucl", "var", "sum", "skew", "kurt",
                "min", "q1", "median", "q3", "max", "range")


def describe(values: np.ndarray, moment: str = "n-1", level: float = 0.95,
             first_row: int = 1) -> dict[str, float]:
    """Origin's descriptive quantities of one data set (``moment``: divisor "n-1" or "n")."""
    allv = np.asarray(values, float)
    ok = np.isfinite(allv)
    v = allv[ok]
    n = len(v)
    out = {"n": float(n), "missing": float(len(allv) - n)}
    if n == 0:
        return out
    ddof = 0 if moment == "n" else 1
    mean = float(v.mean())
    sd = float(v.std(ddof=ddof)) if n > ddof else math.nan
    se = sd / math.sqrt(n) if np.isfinite(sd) else math.nan
    t = stats.t.ppf(0.5 + level / 2, n - 1) if n > 1 else math.nan
    q1, med, q3 = (float(q) for q in np.percentile(v, [25, 50, 75]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        vals, counts = np.unique(v, return_counts=True)
        idx = np.flatnonzero(ok)
        out.update({
            "mean": mean, "sd": sd, "se": se, "lcl": mean - t * se, "ucl": mean + t * se,
            "var": sd ** 2, "sum": float(v.sum()),
            "skew": float(stats.skew(v, bias=moment == "n")) if n > 2 else math.nan,
            "kurt": float(stats.kurtosis(v, bias=moment == "n")) if n > 3 else math.nan,
            "uss": float(np.sum(v ** 2)), "css": float(np.sum((v - mean) ** 2)),
            "cv": sd / mean if mean != 0 else math.nan,
            "mad": float(np.mean(np.abs(v - mean))),
            "gmean": float(stats.gmean(v)) if np.all(v > 0) else math.nan,
            "hmean": float(stats.hmean(v)) if np.all(v > 0) else math.nan,
            "mode": float(vals[np.argmax(counts)]) if counts.max() > 1 else math.nan,
            "min": float(v.min()), "q1": q1, "median": med, "q3": q3, "max": float(v.max()),
            "range": float(v.max() - v.min()), "iqr": q3 - q1,
            "imin": float(idx[int(np.argmin(v))] + first_row),
            "imax": float(idx[int(np.argmax(v))] + first_row)})
    return out


def _fmt_quantity(key: str, value) -> str:
    if key in ("n", "missing", "imin", "imax"):
        return "--" if value is None or not np.isfinite(value) else str(int(value))
    return num(value)


def desc_stats(books: dict, settings: dict, report: ReportSheet | None = None,
               mode: str = "manual") -> tuple[ReportSheet, str]:
    """Statistics > Descriptive Statistics > Statistics on Columns."""
    inputs = read_ranges(books, settings.get("ranges", []))
    inputs = [r for r in inputs if not r.column.is_text()] or inputs
    keys = [k for k in settings.get("quantities", DESC_DEFAULT) if k in DESC_QUANTITIES]
    if not keys:
        raise StatError("Select at least one quantity to compute.")
    moment = settings.get("moment", "n-1")
    level = float(settings.get("level", 0.95))
    first = inputs[0]
    report, fresh = _new_report(first, f"Descriptive Statistics of {first.text}"
                                + (" …" if len(inputs) > 1 else ""), "DescStatsOnCols",
                                report, "desc", settings, mode)
    _notes(fresh, "Statistics on Columns", inputs, [
        ["Moment Computation Method", "Divisor is N" if moment == "n" else
         "Divisor is N − 1 (sample statistics)"],
        ["Quantiles Interpolation", "Linear (type 7)"]])
    headers = [DESC_QUANTITIES[k].replace("95%", f"{level * 100:g}%") for k in keys]
    rows, results = [], []
    for r in inputs:
        d = describe(r.values, moment, level, (r.rows[0] if r.rows else 1))
        results.append(d)
        rows.append([f"{r.column.name}  {r.column.long_name}".strip() +
                     rows_text(*(r.rows or (None, None)))] +
                    [_fmt_quantity(k, d.get(k, math.nan)) for k in keys])
    fresh.add_table("Descriptive Statistics", [""] + headers, rows)
    if settings.get("histogram"):
        for r in inputs:
            if len(r.finite):
                fresh.add_graph(f"Histogram of {r.title}", rg.histogram_graph(
                    r.finite, f"Histogram of {r.title}", r.column.axis_title), width=380,
                    group="Histograms")
    if settings.get("box"):
        data = [(r.title, r.finite) for r in inputs if len(r.finite)]
        if data:
            fresh.add_graph("Box Chart", rg.box_graph(data), width=420, group="Box Charts")
    report = _finish(report, fresh)
    text = f"{_stamp(first.book)}\nStatistics on Columns → [{first.book.name}]{report.name}"
    for r, d in zip(inputs, results):
        text += f"\n  {r.title:<14} N = {int(d['n'])}  Mean = {num(d.get('mean'))}  " \
                f"SD = {num(d.get('sd'))}"
    return report, text


ROW_QUANTITIES = {"mean": "Mean", "sd": "Standard Deviation", "se": "SE of Mean",
                  "n": "N", "sum": "Sum", "min": "Minimum", "median": "Median",
                  "max": "Maximum", "range": "Range"}


def _sheet_op(kind: str, settings: dict, mode: str, op: dict | None) -> dict:
    """The recalculation record of an output-in-worksheet operation (kept on the sheet)."""
    new = {"kind": kind, "settings": jsonable(settings),
           "mode": op.get("mode", mode) if op else mode, "dirty": False}
    if op is not None:
        op.update(new)
        return op
    return new


def stats_on_rows(books: dict, settings: dict, op: dict | None = None,
                  mode: str = "manual") -> tuple[Worksheet, list[int], str]:
    """Statistics on Rows: append the chosen quantities as new columns of the source sheet
    (one value per row across the input columns). The columns carry Origin's lock: the
    operation is recorded in ``ws.ops`` and, with ``op`` (Recalculate), its output columns
    are rewritten in place."""
    inputs = read_ranges(books, settings.get("ranges", []))
    ws = inputs[0].sheet
    if any(r.sheet is not ws for r in inputs):
        raise StatError("Statistics on Rows needs columns of one worksheet.")
    if len(inputs) < 2:
        raise StatError("Select two or more columns.")
    keys = [k for k in settings.get("quantities", ("mean", "sd", "n")) if k in ROW_QUANTITIES]
    if not keys:
        raise StatError("Select at least one quantity to compute.")
    n = max(len(r.values) for r in inputs)
    lo = (inputs[0].rows[0] - 1) if inputs[0].rows else 0
    m = np.full((n, len(inputs)), np.nan)
    for j, r in enumerate(inputs):
        m[:len(r.values), j] = r.values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cnt = np.isfinite(m).sum(axis=1).astype(float)
        out = {"mean": np.nanmean(m, axis=1),
               "sd": np.where(cnt > 1, np.nanstd(m, axis=1, ddof=1), np.nan),
               "n": cnt, "sum": np.where(cnt > 0, np.nansum(m, axis=1), np.nan),
               "min": np.nanmin(m, axis=1), "median": np.nanmedian(m, axis=1),
               "max": np.nanmax(m, axis=1)}
        out["se"] = out["sd"] / np.sqrt(cnt)
        out["range"] = out["max"] - out["min"]
    names = "".join(r.column.name for r in inputs)
    old = [c for c in ((op or {}).get("outputs") or []) if c in ws.names()]
    if op is not None:   # drop output columns no longer asked for
        ws.delete_columns([ws.index_of(c) for c in old[len(keys):]])
        old = old[:len(keys)]
    added = []
    for k_i, k in enumerate(keys):
        if k_i < len(old):
            col = ws.column(old[k_i])
            col.designation = "yEr" if k in ("sd", "se") else "Y"
        else:
            col = ws.add_column("yEr" if k in ("sd", "se") else "Y")
        col.long_name, col.comments = ROW_QUANTITIES[k], f"{ROW_QUANTITIES[k]} of {names}"
        idx = ws.columns.index(col)
        vals = [None] * ws.n_rows
        for i, v in enumerate(out[k]):
            if lo + i < ws.n_rows and np.isfinite(v):
                vals[lo + i] = float(v)
        ws.set_column_values(idx, vals)
        added.append(idx)
    record = _sheet_op("rows", settings, mode, op)
    record["outputs"] = [ws.columns[i].name for i in added]
    if op is None:
        ws.ops.append(record)
    return ws, added, (f"{_stamp(inputs[0].book)}\nStatistics on Rows of {names}: " +
                       ", ".join(ROW_QUANTITIES[k] for k in keys))


# ====================================================================== frequency counts
def frequency_counts(books: dict, settings: dict, out: Worksheet | None = None,
                     mode: str = "manual") -> tuple[Worksheet, str]:
    """Statistics > Descriptive Statistics > Frequency Counts: a worksheet of bin centers,
    counts, cumulative counts and relative frequencies (discrete counts for text data).
    The sheet records the operation (``ws.ops``, Origin's lock on its columns); with
    ``out`` (Recalculate) that sheet is refilled in place."""
    r = read_range(books, settings.get("range", ""))
    book = r.book
    sheet = Worksheet(name=out.name if out is not None else
                      book.unique_sheet_name("FreqCounts"))
    discrete = r.column.is_text() or settings.get("discrete", False)
    if discrete:
        labels = [t for t in r.texts if t != ""]
        levels = _sort_levels(labels)
        counts = np.array([labels.count(lv) for lv in levels], float)
        total = counts.sum() or 1.0
        cols = [("Data", "L", levels), ("Count", "Y", counts),
                ("Percent", "Y", counts / total * 100),
                ("Cumulative Count", "Y", np.cumsum(counts)),
                ("Cumulative Percent", "Y", np.cumsum(counts) / total * 100)]
        sheet.columns.clear()
        sheet.set_rows(max(len(levels), 1))
        for title, desig, vals in cols:
            c = sheet.add_column("X" if title == "Data" else desig)
            c.long_name = title
            sheet.set_column_values(len(sheet.columns) - 1, list(vals))
        sheet.columns[0].designation = "X"
        text = (f"{_stamp(book)}\nDiscrete Frequency of {r.text}: {len(levels)} "
                f"categories → [{book.name}]{sheet.name}")
    else:
        v = r.finite
        if len(v) == 0:
            raise StatError(f"{r.text} has no numeric data.")
        if settings.get("auto", True) or not settings.get("step"):
            edges = np.histogram_bin_edges(v, bins="sturges")
            step = float(edges[1] - edges[0]) if len(edges) > 1 else 1.0
            lo, hi = float(edges[0]), float(edges[-1])
        else:
            lo, hi = float(settings.get("from", v.min())), float(settings.get("to", v.max()))
            step = float(settings["step"])
            if step <= 0 or hi <= lo:
                raise StatError("Use From < To and a positive Step Size.")
        nb = max(1, int(math.ceil((hi - lo) / step - 1e-9)))
        edges = lo + step * np.arange(nb + 1)
        counts, _ = np.histogram(v[(v >= lo) & (v <= edges[-1])], bins=edges)
        counts = counts.astype(float)
        total = float(len(v))
        centers = (edges[:-1] + edges[1:]) / 2
        cols = [("Bin Center", "X", centers), ("Bin Begin", "Y", edges[:-1]),
                ("Bin End", "Y", edges[1:]), ("Count", "Y", counts),
                ("Cumulative Count", "Y", np.cumsum(counts)),
                ("Relative Frequency", "Y", counts / total),
                ("Cumulative Frequency", "Y", np.cumsum(counts) / total)]
        sheet.set_rows(len(centers))
        for title, desig, vals in cols:
            c = sheet.add_column(desig)
            c.long_name = title
            sheet.set_column_values(len(sheet.columns) - 1, list(vals))
        sheet.columns[3].comments = f"counts of {r.text}"
        outside = int(np.sum((v < lo) | (v > edges[-1])))
        text = (f"{_stamp(book)}\nFrequency Counts of {r.text}: {nb} bins of width "
                f"{num(step)} from {num(lo)} → [{book.name}]{sheet.name}" +
                (f" ({outside} values outside the bins)" if outside else ""))
    old = out.ops[0] if out is not None and out.ops else None
    record = _sheet_op("freq", settings, mode, old)
    record["outputs"] = None   # the whole sheet
    if out is None:
        sheet.ops = [record]
        book.sheets.append(sheet)
        return sheet, text
    # keep the sheet object (graphs link to its name and columns)
    out.columns, out.n_rows, out.ops = sheet.columns, sheet.n_rows, [record]
    return out, text


# ====================================================================== normality
NORMALITY_TESTS = {"shapiro": "Shapiro-Wilk", "ks": "Kolmogorov-Smirnov",
                   "lilliefors": "Lilliefors", "anderson": "Anderson-Darling",
                   "dagostino": "D'Agostino-K squared", "jb": "Jarque-Bera",
                   "chen_shapiro": "Chen-Shapiro"}
CHEN_SHAPIRO_REPLICATES = 10000
_CS_NULL: dict[int, np.ndarray] = {}


def chen_shapiro_statistic(v: np.ndarray) -> float:
    """Chen & Shapiro's normalized-spacings statistic QH* = √n (1 − QH), with
    QH = Σ (x₍ᵢ₊₁₎ − x₍ᵢ₎) / (Hᵢ₊₁ − Hᵢ) / ((n − 1) s) and Hᵢ = Φ⁻¹((i − 0.375)/(n + 0.25))
    (Chen, L. & Shapiro, S. S. 1995, "An alternative test for normality based on normalized
    spacings", J. Statist. Comput. Simul. 53, 269–287; the form used in Romão, Delgado &
    Costa 2010, J. Statist. Comput. Simul. 80, 545–591). Large values reject normality."""
    x = np.sort(np.asarray(v, float))
    n = len(x)
    s = x.std(ddof=1)
    if n < 3 or not s > 0:
        return math.nan
    h = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    qh = float(np.sum(np.diff(x) / np.diff(h)) / ((n - 1) * s))
    return math.sqrt(n) * (1.0 - qh)


def _chen_shapiro_null(n: int) -> np.ndarray:
    """Null distribution of QH* for sample size ``n`` by Monte Carlo (QH* does not depend
    on the mean or the variance, so standard normal samples give the exact null law up to
    simulation error); cached per ``n`` with a fixed seed, so p-values are reproducible."""
    if n not in _CS_NULL:
        rng = np.random.default_rng(1995 + n)
        h = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
        dh = np.diff(h)
        out = []
        chunk = max(1, min(CHEN_SHAPIRO_REPLICATES, 2_000_000 // n))
        left = CHEN_SHAPIRO_REPLICATES
        while left > 0:
            m = min(chunk, left)
            x = np.sort(rng.standard_normal((m, n)), axis=1)
            qh = np.sum(np.diff(x, axis=1) / dh, axis=1) / ((n - 1) * x.std(axis=1, ddof=1))
            out.append(math.sqrt(n) * (1.0 - qh))
            left -= m
        _CS_NULL[n] = np.sort(np.concatenate(out))
    return _CS_NULL[n]


def chen_shapiro(v: np.ndarray) -> tuple[float, float]:
    """(QH*, Monte Carlo p-value) of the Chen-Shapiro test (3 ≤ n ≤ 5000)."""
    v = np.asarray(v, float)
    n = len(v)
    stat = chen_shapiro_statistic(v)
    if not np.isfinite(stat) or n > 5000:
        return math.nan, math.nan
    null = _chen_shapiro_null(n)
    exceed = len(null) - np.searchsorted(null, stat - 1e-12, side="left")
    return stat, float((exceed + 1) / (len(null) + 1))


def chen_shapiro_critical(n: int, alpha: float) -> float:
    """Upper ``alpha`` critical value of QH* (Monte Carlo)."""
    return float(np.quantile(_chen_shapiro_null(n), 1 - alpha))


def normality_values(v: np.ndarray, test: str) -> tuple[float, float]:
    """(statistic, p) of one normality test."""
    v = np.asarray(v, float)
    n = len(v)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if test == "shapiro":
            if n < 3 or n > 5000:
                return math.nan, math.nan
            w, p = stats.shapiro(v)
            return float(w), float(p)
        if test == "ks":
            if n < 3:
                return math.nan, math.nan
            sd = v.std(ddof=1)
            if sd == 0:
                return math.nan, math.nan
            res = stats.kstest((v - v.mean()) / sd, "norm")
            return float(res.statistic), float(res.pvalue)
        if test == "lilliefors":
            from statsmodels.stats.diagnostic import lilliefors
            if n < 4:
                return math.nan, math.nan
            d, p = lilliefors(v, dist="norm", pvalmethod="table")
            return float(d), float(p)
        if test == "anderson":
            from statsmodels.stats.diagnostic import normal_ad
            if n < 8:
                return math.nan, math.nan
            a2, p = normal_ad(v)
            return float(a2), float(p)
        if test == "dagostino":
            if n < 8:
                return math.nan, math.nan
            k2, p = stats.normaltest(v)
            return float(k2), float(p)
        if test == "jb":
            if n < 3:
                return math.nan, math.nan
            res = stats.jarque_bera(v)
            return float(res.statistic), float(res.pvalue)
        if test == "chen_shapiro":
            return chen_shapiro(v)
    raise StatError(f"Unknown normality test {test}.")


def normality(books: dict, settings: dict, report: ReportSheet | None = None,
              mode: str = "manual") -> tuple[ReportSheet, str]:
    inputs = read_ranges(books, settings.get("ranges", []))
    tests = [t for t in settings.get("tests", ["shapiro"]) if t in NORMALITY_TESTS]
    if not tests:
        raise StatError("Select at least one normality test.")
    alpha = float(settings.get("alpha", 0.05))
    first = inputs[0]
    report, fresh = _new_report(first, f"Normality Test of {first.text}", "NormalityTest",
                                report, "normality", settings, mode)
    _notes(fresh, "Normality Test", inputs,
           [["Tests", ", ".join(NORMALITY_TESTS[t] for t in tests)],
            ["Significance Level", num(alpha)]])
    fresh.add_table("Descriptive Statistics", ["", "N", "Mean", "SD", "Minimum", "Median",
                                               "Maximum"],
                    [[r.title, str(len(r.finite)), num(r.finite.mean() if len(r.finite) else
                                                       math.nan),
                      num(r.finite.std(ddof=1) if len(r.finite) > 1 else math.nan),
                      num(r.finite.min() if len(r.finite) else math.nan),
                      num(np.median(r.finite) if len(r.finite) else math.nan),
                      num(r.finite.max() if len(r.finite) else math.nan)] for r in inputs])
    lines = []
    for t in tests:
        rows = []
        for r in inputs:
            s, p = normality_values(r.finite, t)
            rows.append([r.title, str(len(r.finite)), num(s, 5), pval(p),
                         _decision(p, alpha, "Reject normality", "Can't reject normality")])
            lines.append(f"  {NORMALITY_TESTS[t]:<22} {r.title}: statistic = {num(s, 5)}, "
                         f"p = {pval(p)}")
        notes = [f"At the {num(alpha)} level, the data were significantly drawn from "
                 "a normally distributed population unless the decision rejects normality."]
        if t == "chen_shapiro":
            notes.append("Statistic QH* = √n(1 − QH) of Chen & Shapiro (1995), normalized "
                         "spacings with Blom scores; the p-value comes from a Monte Carlo "
                         f"null distribution ({CHEN_SHAPIRO_REPLICATES:,} normal samples of "
                         "the same size).")
        fresh.add_table(NORMALITY_TESTS[t], ["", "DF", "Statistic", "p-value",
                                             f"Decision at level({_level(alpha)})"], rows,
                        notes)
    if settings.get("qq", True):
        for r in inputs:
            if len(r.finite) >= 3:
                fresh.add_graph(f"Q-Q Plot of {r.title}", rg.qq_graph(
                    r.finite, f"Q-Q Plot of {r.title}", r.column.axis_title), width=380,
                    group="Q-Q Plots")
    report = _finish(report, fresh)
    return report, f"{_stamp(first.book)}\nNormality Test → [{first.book.name}]{report.name}\n" \
        + "\n".join(lines)


# ====================================================================== correlation
CORR_METHODS = {"pearson": "Pearson", "spearman": "Spearman", "kendall": "Kendall"}


def corr_pair(a: np.ndarray, b: np.ndarray, method: str) -> tuple[float, float, int]:
    ok = np.isfinite(a) & np.isfinite(b)
    n = int(ok.sum())
    if n < 3:
        return math.nan, math.nan, n
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == "pearson":
            r, p = stats.pearsonr(a[ok], b[ok])
        elif method == "spearman":
            r, p = stats.spearmanr(a[ok], b[ok])
        else:
            r, p = stats.kendalltau(a[ok], b[ok])
    return float(r), float(p), n


def correlation(books: dict, settings: dict, report: ReportSheet | None = None,
                mode: str = "manual") -> tuple[ReportSheet, str]:
    inputs = read_ranges(books, settings.get("ranges", []))
    if len(inputs) < 2:
        raise StatError("Select two or more columns.")
    methods = [m for m in settings.get("methods", ["pearson"]) if m in CORR_METHODS]
    if not methods:
        raise StatError("Select at least one correlation type.")
    alpha = float(settings.get("alpha", 0.05))
    flag = settings.get("flag", True)
    first = inputs[0]
    report, fresh = _new_report(first, f"Correlation Coefficient of {first.text} …",
                                "CorrCoef", report, "correlation", settings, mode)
    _notes(fresh, "Correlation Coefficient", inputs,
           [["Correlation Types", ", ".join(CORR_METHODS[m] for m in methods)],
            ["Missing Values", "Exclude pairwise"]])
    n = max(len(r.values) for r in inputs)
    data = []
    for r in inputs:
        v = np.full(n, np.nan)
        v[:len(r.values)] = r.values
        data.append(v)
    names = [r.title for r in inputs]
    lines = []
    for m in methods:
        rr, pp, nn = [], [], []
        flagged = False
        for i in range(len(inputs)):
            rrow, prow, nrow = [names[i]], [names[i]], [names[i]]
            for j in range(len(inputs)):
                r, p, k = corr_pair(data[i], data[j], m)
                star = "*" if flag and i != j and np.isfinite(p) and p < alpha else ""
                flagged |= bool(star)
                rrow.append(num(r, 5) + star if np.isfinite(r) else "--")
                prow.append(pval(p) if i != j else "--")
                nrow.append(str(k))
                if j > i:
                    lines.append(f"  {CORR_METHODS[m]} r({names[i]}, {names[j]}) = {num(r, 5)}"
                                 f", p = {pval(p)}, N = {k}")
            rr.append(rrow)
            pp.append(prow)
            nn.append(nrow)
        notes = [f"*: correlation is significant at the {num(alpha)} level (two-tailed)."] \
            if flagged else []
        fresh.add_table(f"{CORR_METHODS[m]} Correlation Coefficients", [""] + names, rr, notes,
                        group=CORR_METHODS[m])
        fresh.add_table(f"{CORR_METHODS[m]} Correlation P-values", [""] + names, pp,
                        ["Two-tailed test of H0: no correlation."], group=CORR_METHODS[m])
        fresh.add_table("Number of Points", [""] + names, nn, group=CORR_METHODS[m])
    if settings.get("scatter") and len(inputs) == 2:
        a, b = data[0], data[1]
        ok = np.isfinite(a) & np.isfinite(b)
        fresh.add_graph("Scatter Plot", rg.scatter_graph(
            a[ok], b[ok], "Scatter Plot", inputs[0].column.axis_title,
            inputs[1].column.axis_title), width=380)
    report = _finish(report, fresh)
    return report, f"{_stamp(first.book)}\nCorrelation Coefficient → " \
                   f"[{first.book.name}]{report.name}\n" + "\n".join(lines)


# ====================================================================== t-tests
TAILS = {"two": "Two-tailed (≠)", "upper": "Upper-tailed (>)", "lower": "Lower-tailed (<)"}
_ALT = {"two": "two-sided", "upper": "greater", "lower": "less"}
_SYM = {"two": "<>", "upper": ">", "lower": "<"}


def _ci(mean: float, se: float, df: float, alpha: float, tail: str) -> tuple[float, float]:
    if tail == "two":
        q = stats.t.ppf(1 - alpha / 2, df)
        return mean - q * se, mean + q * se
    q = stats.t.ppf(1 - alpha, df)
    return (mean - q * se, math.inf) if tail == "upper" else (-math.inf, mean + q * se)


def _desc_row(title: str, v: np.ndarray) -> list[str]:
    n = len(v)
    sd = v.std(ddof=1) if n > 1 else math.nan
    return [title, str(n), num(v.mean() if n else math.nan), num(sd),
            num(sd / math.sqrt(n) if n > 1 else math.nan)]


def one_sample_t(books: dict, settings: dict, report: ReportSheet | None = None,
                 mode: str = "manual") -> tuple[ReportSheet, str]:
    inputs = read_ranges(books, settings.get("ranges", []))
    mu = float(settings.get("mean", 0.0))
    tail = settings.get("tail", "two")
    alpha = float(settings.get("alpha", 0.05))
    first = inputs[0]
    report, fresh = _new_report(first, f"One-Sample t-Test of {first.text}", "OneSampleT",
                                report, "ttest1", settings, mode)
    _notes(fresh, "One-Sample t-Test", inputs,
           [["Null Hypothesis", f"Mean = {num(mu)}"],
            ["Alternative Hypothesis", f"Mean {_SYM[tail]} {num(mu)}"],
            ["Significance Level", num(alpha)]])
    fresh.add_table("Descriptive Statistics", ["", "N", "Mean", "SD", "SEM"],
                    [_desc_row(r.title, r.finite) for r in inputs])
    rows, lines = [], []
    for r in inputs:
        v = r.finite
        if len(v) < 2:
            raise StatError(f"{r.title}: a t-test needs at least 2 values.")
        res = stats.ttest_1samp(v, mu, alternative=_ALT[tail])
        se = v.std(ddof=1) / math.sqrt(len(v))
        lo, hi = _ci(v.mean(), se, len(v) - 1, alpha, tail)
        rows.append([r.title, num(res.statistic, 5), str(len(v) - 1), pval(res.pvalue),
                     num(lo), num(hi)])
        lines.append(f"  {r.title}: t = {num(res.statistic, 5)}, DF = {len(v) - 1}, "
                     f"p = {pval(res.pvalue)}")
        concl = ("significantly" if res.pvalue < alpha else "not significantly")
        note = f"At the {num(alpha)} level, the population mean of {r.title} is {concl} " \
               f"{'different from' if tail == 'two' else ('greater than' if tail == 'upper' else 'less than')} {num(mu)}."
        rows[-1].append(note)
    fresh.add_table("t-Test Statistics", ["", "t Statistic", "DF", "Prob>|t|" if tail == "two"
                                          else "Prob", f"Lower {100 - alpha * 100:g}% CL",
                                          f"Upper {100 - alpha * 100:g}% CL"],
                    [row[:-1] for row in rows], [row[-1] for row in rows])
    report = _finish(report, fresh)
    return report, f"{_stamp(first.book)}\nOne-Sample t-Test (mean = {num(mu)}) → " \
                   f"[{first.book.name}]{report.name}\n" + "\n".join(lines)


def two_sample_t(books: dict, settings: dict, report: ReportSheet | None = None,
                 mode: str = "manual") -> tuple[ReportSheet, str]:
    inputs, groups = _groups(books, settings, 2)
    if len(groups) != 2:
        raise StatError("A two-sample t-test needs exactly two groups "
                        f"(the grouping column has {len(groups)}).")
    (n1, a), (n2, b) = groups
    diff0 = float(settings.get("diff", 0.0))
    tail = settings.get("tail", "two")
    alpha = float(settings.get("alpha", 0.05))
    first = inputs[0]
    report, fresh = _new_report(first, f"Two-Sample t-Test of {n1} and {n2}", "TwoSampleT",
                                report, "ttest2", settings, mode)
    _notes(fresh, "Two-Sample t-Test", inputs,
           [["Null Hypothesis", f"mean1 − mean2 = {num(diff0)}"],
            ["Alternative Hypothesis", f"mean1 − mean2 {_SYM[tail]} {num(diff0)}"],
            ["Significance Level", num(alpha)]])
    fresh.add_table("Descriptive Statistics", ["", "N", "Mean", "SD", "SEM"],
                    [_desc_row(n1, a), _desc_row(n2, b)])
    rows, lines = [], []
    for equal in (True, False):
        res = stats.ttest_ind(a - diff0, b, equal_var=equal, alternative=_ALT[tail])
        va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
        if equal:
            df = len(a) + len(b) - 2
            sp = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / df
            se = math.sqrt(sp * (1 / len(a) + 1 / len(b)))
        else:
            se = math.sqrt(va + vb)
            df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
        d = a.mean() - b.mean()
        lo, hi = _ci(d, se, df, alpha, tail)
        label = "Equal Variance Assumed" if equal else "Equal Variance Not Assumed"
        rows.append([label, num(res.statistic, 5), num(df), pval(res.pvalue), num(d),
                     num(lo), num(hi)])
        lines.append(f"  {label}: t = {num(res.statistic, 5)}, DF = {num(df)}, "
                     f"p = {pval(res.pvalue)}")
    p_eq = float(stats.ttest_ind(a - diff0, b, alternative=_ALT[tail]).pvalue)
    fresh.add_table("t-Test Statistics", ["", "t Statistic", "DF", "Prob", "Mean Difference",
                                          f"Lower {100 - alpha * 100:g}% CL",
                                          f"Upper {100 - alpha * 100:g}% CL"], rows,
                    [f"At the {num(alpha)} level, the difference of the population means is "
                     f"{'significantly' if p_eq < alpha else 'not significantly'} different "
                     f"from {num(diff0)} (equal variance assumed)."])
    if settings.get("equal_var_test", True):
        w, p = stats.levene(a, b, center="mean")
        f_stat = a.var(ddof=1) / b.var(ddof=1)
        p_f = 2 * min(stats.f.cdf(f_stat, len(a) - 1, len(b) - 1),
                      stats.f.sf(f_stat, len(a) - 1, len(b) - 1))
        fresh.add_table("Equality of Variance Test", ["", "Statistic", "DF1", "DF2", "Prob",
                                                      f"Decision at level({_level(alpha)})"],
                        [["Levene (absolute deviations)", num(w, 5), "1",
                          str(len(a) + len(b) - 2), pval(p),
                          _decision(p, alpha, "Variances differ", "Equal variances")],
                         ["F-test (two-tailed)", num(f_stat, 5), str(len(a) - 1),
                          str(len(b) - 1), pval(p_f),
                          _decision(p_f, alpha, "Variances differ", "Equal variances")]])
    report = _finish(report, fresh)
    return report, f"{_stamp(first.book)}\nTwo-Sample t-Test of {n1} and {n2} → " \
                   f"[{first.book.name}]{report.name}\n" + "\n".join(lines)


def paired_t(books: dict, settings: dict, report: ReportSheet | None = None,
             mode: str = "manual") -> tuple[ReportSheet, str]:
    r1 = read_range(books, settings.get("range1", ""))
    r2 = read_range(books, settings.get("range2", ""))
    n = min(len(r1.values), len(r2.values))
    a, b = r1.values[:n], r2.values[:n]
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 2:
        raise StatError("A pair-sample t-test needs at least 2 complete pairs.")
    diff0 = float(settings.get("diff", 0.0))
    tail = settings.get("tail", "two")
    alpha = float(settings.get("alpha", 0.05))
    report, fresh = _new_report(r1, f"Pair-Sample t-Test of {r1.title} and {r2.title}",
                                "PairSampleT", report, "ttest_pair", settings, mode)
    _notes(fresh, "Pair-Sample t-Test", [r1, r2],
           [["Null Hypothesis", f"mean1 − mean2 = {num(diff0)}"],
            ["Alternative Hypothesis", f"mean1 − mean2 {_SYM[tail]} {num(diff0)}"],
            ["Significance Level", num(alpha)]])
    d = a - b
    fresh.add_table("Descriptive Statistics", ["", "N", "Mean", "SD", "SEM"],
                    [_desc_row(r1.title, a), _desc_row(r2.title, b),
                     _desc_row("Difference", d)])
    res = stats.ttest_rel(a - diff0, b, alternative=_ALT[tail])
    se = d.std(ddof=1) / math.sqrt(len(d))
    lo, hi = _ci(d.mean(), se, len(d) - 1, alpha, tail)
    fresh.add_table("t-Test Statistics", ["", "t Statistic", "DF", "Prob", "Mean Difference",
                                          f"Lower {100 - alpha * 100:g}% CL",
                                          f"Upper {100 - alpha * 100:g}% CL"],
                    [[f"{r1.title} − {r2.title}", num(res.statistic, 5), str(len(d) - 1),
                      pval(res.pvalue), num(d.mean()), num(lo), num(hi)]],
                    [f"At the {num(alpha)} level, the mean difference is "
                     f"{'significantly' if res.pvalue < alpha else 'not significantly'} "
                     f"different from {num(diff0)}."])
    report = _finish(report, fresh)
    return report, f"{_stamp(r1.book)}\nPair-Sample t-Test: t = {num(res.statistic, 5)}, " \
                   f"DF = {len(d) - 1}, p = {pval(res.pvalue)} → [{r1.book.name}]{report.name}"


# ====================================================================== ANOVA
MEANS_METHODS = {"tukey": "Tukey", "bonferroni": "Bonferroni", "sidak": "Dunn-Sidak",
                 "lsd": "Fisher LSD", "scheffe": "Scheffe", "dunnett": "Dunnett",
                 "holm_bonferroni": "Holm-Bonferroni", "holm_sidak": "Holm-Sidak"}
DUNNETT_SEED = 20240521   # fixed: the multivariate t integration is randomized QMC


def _pairs(gs: list[Group], mse: float, df_error: float):
    """(i, j, mean diff, SE, t, unadjusted two-sided p) for i > j, Origin's order."""
    out = []
    for i in range(len(gs)):
        for j in range(i):
            d = float(gs[i].data.mean() - gs[j].data.mean())
            se = math.sqrt(mse * (1 / gs[i].n + 1 / gs[j].n))
            t = d / se if se > 0 else math.nan
            out.append((i, j, d, se, t, float(2 * stats.t.sf(abs(t), df_error))))
    return out


def holm_rows(groups: list[tuple[str, np.ndarray]], method: str, mse: float,
              df_error: float, alpha: float) -> list[list[str]]:
    """Holm-Bonferroni / Holm-Sidak step-down comparisons of all pairs (pooled MSE t tests,
    as Origin): the p-values are ranked; the i-th smallest (of m) is tested at
    α/(m − i + 1) (Holm-Bonferroni) or 1 − (1 − α)^(1/(m − i + 1)) (Holm-Sidak), and every
    comparison after the first non-significant one is non-significant. "Alpha" is the
    step's critical level and LCL/UCL the interval at that level."""
    gs = [Group(k, name, v) for k, (name, v) in enumerate(groups)]
    pairs = _pairs(gs, mse, df_error)
    m = len(pairs)
    order = sorted(range(m), key=lambda k: (pairs[k][5], k))
    level, sig = {}, {}
    still = True
    for rank, k in enumerate(order):
        left = m - rank
        a = alpha / left if method == "holm_bonferroni" else 1 - (1 - alpha) ** (1 / left)
        level[k] = a
        still = still and pairs[k][5] < a
        sig[k] = still
    rows = []
    for k, (i, j, d, se, t, p) in enumerate(pairs):
        crit = stats.t.ppf(1 - level[k] / 2, df_error)
        rows.append([f"{gs[i].label}  {gs[j].label}", num(d), num(se), num(t, 5), pval(p),
                     num(level[k], 5), "1" if sig[k] else "0", num(d - crit * se),
                     num(d + crit * se)])
    return rows


DUNNETT_TAILS = {"two": "two-sided", "lower": "less", "upper": "greater"}  # scipy names
DUNNETT_TAIL_TEXT = {"two": "two-sided", "lower": "one-sided, mean < control",
                     "upper": "one-sided, mean > control"}


def _dunnett_cdf(c: np.ndarray, rho: np.ndarray, df: float, two_sided: bool = True
                 ) -> np.ndarray:
    """P(|T_j| < c for all j) (two-sided) or P(T_j < c for all j) (one-sided) of the
    multivariate t with correlation ``rho``, for every value of ``c`` (Dunnett 1955;
    integrated like scipy.stats.dunnett)."""
    c = np.atleast_1d(np.asarray(c, float))
    m = rho.shape[0]
    if m == 1:
        return 1 - 2 * stats.t.sf(np.abs(c), df) if two_sided else stats.t.cdf(c, df)
    mvt = stats.multivariate_t(shape=rho, df=df, seed=DUNNETT_SEED)
    if two_sided:
        x = np.repeat(np.abs(c)[:, None], m, axis=1)
        return np.atleast_1d(mvt.cdf(x, lower_limit=-x))
    x = np.repeat(c[:, None], m, axis=1)
    return np.atleast_1d(mvt.cdf(x, lower_limit=np.full_like(x, -np.inf)))


def dunnett_values(groups: list[tuple[str, np.ndarray]], control: int, mse: float,
                   df_error: float, alpha: float, tail: str = "two") -> list[tuple]:
    """Dunnett's many-to-one comparisons against group ``control`` with the pooled MSE:
    ``(i, mean diff, SE, t, p, LCL, UCL)`` for every other group.

    ``tail`` is the alternative hypothesis as in Origin / scipy.stats.dunnett: ``"two"``
    (mean <> control), ``"lower"`` (mean < control, scipy's ``less``) or ``"upper"``
    (mean > control, ``greater``). The one-sided tests use the one-sided critical value
    of the multivariate t and give one-sided simultaneous confidence bounds (LCL = -inf for
    "< control", UCL = +inf for "> control"), like scipy's ``confidence_interval``."""
    if tail not in DUNNETT_TAILS:
        raise StatError(f"Unknown Dunnett alternative {tail!r}.")
    others = [i for i in range(len(groups)) if i != control]
    n0 = len(groups[control][1])
    ns = np.array([len(groups[i][1]) for i in others], float)
    r = n0 / ns + 1
    rho = 1 / np.sqrt(r[:, None] * r[None, :])
    np.fill_diagonal(rho, 1.0)
    m0 = float(np.mean(groups[control][1]))
    diffs = np.array([float(np.mean(groups[i][1])) - m0 for i in others])
    se = np.sqrt(mse * (1 / ns + 1 / n0))
    t = diffs / se
    two = tail == "two"
    stat = -t if tail == "lower" else t   # "< control": large values of -t reject
    p = np.clip(1 - _dunnett_cdf(stat, rho, df_error, two), 0.0, 1.0)
    from scipy.optimize import brentq
    f = lambda c: 1 - _dunnett_cdf(c, rho, df_error, two)[0] - alpha   # noqa: E731
    hi = 3.0
    while f(hi) > 0 and hi < 200:
        hi *= 2
    crit = brentq(f, 1e-6 if two else -50.0, hi, xtol=1e-6)
    lo_b = np.full_like(diffs, -np.inf) if tail == "lower" else diffs - crit * se
    hi_b = np.full_like(diffs, np.inf) if tail == "upper" else diffs + crit * se
    return [(i, float(diffs[k]), float(se[k]), float(t[k]), float(p[k]),
             float(lo_b[k]), float(hi_b[k])) for k, i in enumerate(others)]


def control_index(groups: list[tuple[str, np.ndarray]], control: str | None) -> int:
    """Index of the Dunnett control level (by name; the first level when blank)."""
    if control is not None and str(control).strip():
        for k, (name, _) in enumerate(groups):
            if str(name) == str(control).strip():
                return k
        raise StatError(f"There is no level “{control}” for the Dunnett control "
                        f"({', '.join(str(n) for n, _ in groups)}).")
    return 0


def factor_levels(books: dict, settings: dict, key: str) -> list[str]:
    """The levels a Dunnett control can be chosen from, for the dialog's dropdown:
    ``"anova1"`` (one column per level, or the levels of the factor column), ``"anova2_a"``
    / ``"anova2_b"`` (Factor A / B of a Two-Way ANOVA). Empty while the input cannot be
    read."""
    try:
        if key == "anova1":
            if settings.get("form", "raw") == "indexed":
                data = read_range(books, settings.get("data", ""))
                factor = read_range(books, settings.get("factor", ""))
                return [str(name) for name, _ in indexed_groups(data, factor)]
            return [r.title for r in read_ranges(books, settings.get("ranges", []))]
        if key in ("anova2_a", "anova2_b"):
            a = key == "anova2_a"
            if settings.get("form", "indexed") == "raw":
                n = int(settings.get("levels_a" if a else "levels_b", 2))
                return level_names(settings.get("names_a" if a else "names_b", ""), n,
                                   "A" if a else "B")
            data = read_range(books, settings.get("data", ""))
            fac = read_range(books, settings.get("factor_a" if a else "factor_b", ""))
            n = min(len(data.values), len(fac.texts))
            return _sort_levels([fac.texts[i] for i in range(n)
                                 if fac.texts[i] != "" and np.isfinite(data.values[i])])
    except (StatError, KeyError, ValueError, TypeError, IndexError, AttributeError):
        return []
    return []


def means_comparison_rows(groups: list[tuple[str, np.ndarray]], method: str, mse: float,
                          df_error: float, alpha: float, control: str | None = None,
                          tail: str = "two") -> list[list[str]]:
    """Origin's Means Comparisons table rows ("B A" = mean of B − mean of A); ``tail`` is
    the Dunnett alternative (see :func:`dunnett_values`)."""
    if method in ("holm_bonferroni", "holm_sidak"):
        return holm_rows(groups, method, mse, df_error, alpha)
    if method == "dunnett":
        c = control_index(groups, control)
        return [[f"{groups[i][0]}  {groups[c][0]}", num(d), num(se), num(t, 5), pval(p),
                 num(alpha), "1" if p < alpha else "0", num(lo), num(hi)]
                for i, d, se, t, p, lo, hi in dunnett_values(groups, c, mse, df_error,
                                                             alpha, tail)]
    gs = [Group(k, name, v) for k, (name, v) in enumerate(groups)]
    comps = {(i, j): (d, se, p, lo, hi) for i, j, d, se, p, lo, hi in
             posthoc(gs, method, mse, df_error, 1 - alpha)}
    rows = []
    for i in range(len(gs)):
        for j in range(i):
            d, se, p, lo, hi = comps[(i, j)]
            stat = abs(d) / se * math.sqrt(2) if method == "tukey" else d / se
            rows.append([f"{gs[i].label}  {gs[j].label}", num(d), num(se), num(stat, 5),
                         pval(p), num(alpha), "1" if p < alpha else "0", num(lo), num(hi)])
    return rows


def _means_tables(fresh: ReportSheet, groups, methods, mse, df_error, alpha,
                  factor: str = "", control: str | None = None, tail: str = "two") -> None:
    for m in methods:
        stat = "q Value" if m == "tukey" else "t Value"
        notes = [f"Sig equals 1 indicates that the difference of the means is "
                 f"significant at the {num(alpha)} level.",
                 "Sig equals 0 indicates that the difference of the means is not "
                 f"significant at the {num(alpha)} level."]
        if m == "dunnett":
            c = control_index(groups, control)
            notes.insert(0, f"Each level is compared with the control level "
                            f"{groups[c][0]} ({DUNNETT_TAIL_TEXT.get(tail, tail)} Dunnett "
                            "test).")
            if tail != "two":
                notes.insert(1, "LCL / UCL are one-sided simultaneous confidence bounds "
                                f"({'upper' if tail == 'lower' else 'lower'} bound only).")
        elif m.startswith("holm"):
            notes.insert(0, "Prob is the unadjusted p-value; Alpha is the step-down "
                            "critical level of that comparison (Holm's sequential "
                            "procedure).")
        fresh.add_table(f"{MEANS_METHODS[m]} Test" + (f" ({factor})" if factor else ""),
                        ["", "MeanDiff", "SEM", stat, "Prob", "Alpha", "Sig", "LCL", "UCL"],
                        means_comparison_rows(groups, m, mse, df_error, alpha, control,
                                              tail),
                        notes, group="Means Comparisons")


def centered_groups(groups: list[tuple[str, np.ndarray]]) -> list[np.ndarray]:
    """The group values minus a common shift, the subtraction done in exact decimal
    arithmetic on each value's shortest round-trip text, so data with many constant leading
    digits (NIST StRD SmLs07–09: 1000000000000.4 …) keep their full precision in the sums
    of squares."""
    from decimal import Decimal, InvalidOperation
    allv = np.concatenate([v for _, v in groups])
    if not len(allv) or not np.all(np.isfinite(allv)):
        return [np.asarray(v, float) for _, v in groups]
    try:
        shift = Decimal(repr(float(np.median(allv))))
        return [np.array([float(Decimal(repr(float(x))) - shift) for x in v], float)
                for _, v in groups]
    except (InvalidOperation, ValueError):
        return [np.asarray(v, float) - float(np.median(allv)) for _, v in groups]


def one_way_anova(books: dict, settings: dict, report: ReportSheet | None = None,
                  mode: str = "manual") -> tuple[ReportSheet, str]:
    inputs, groups = _groups(books, settings, 2)
    alpha = float(settings.get("alpha", 0.05))
    methods = [m for m in settings.get("means", ["tukey"]) if m in MEANS_METHODS]
    first = inputs[0]
    report, fresh = _new_report(first, f"One-Way ANOVA of {first.text}" +
                                (" …" if len(inputs) > 1 else ""), "ANOVA1Way", report,
                                "anova1", settings, mode)
    _notes(fresh, "One-Way ANOVA", inputs,
           [["Data Form", "Indexed" if settings.get("form") == "indexed" else "Raw"],
            ["Significance Level", num(alpha)]])
    fresh.add_table("Descriptive Statistics", ["", "N Analysis", "Mean", "SD", "SE of Mean"],
                    [_desc_row(name, v) for name, v in groups])
    allv = np.concatenate([v for _, v in groups])
    grand = allv.mean()
    shifted = centered_groups(groups)
    sall = np.concatenate(shifted)
    sgrand = sall.mean()
    ss_b = float(sum(len(v) * (v.mean() - sgrand) ** 2 for v in shifted))
    ss_w = float(sum(((v - v.mean()) ** 2).sum() for v in shifted))
    df_b, df_w = len(groups) - 1, len(allv) - len(groups)
    mse = ss_w / df_w
    f = (ss_b / df_b) / mse if mse > 0 else math.nan
    p = float(stats.f.sf(f, df_b, df_w)) if np.isfinite(f) else math.nan
    fresh.add_table("Overall ANOVA", ["", "DF", "Sum of Squares", "Mean Square", "F Value",
                                      "Prob>F"],
                    [["Model", str(df_b), num(ss_b), num(ss_b / df_b), num(f, 5), pval(p)],
                     ["Error", str(df_w), num(ss_w), num(mse), "", ""],
                     ["Total", str(df_b + df_w), num(ss_b + ss_w), "", "", ""]],
                    [f"At the {num(alpha)} level, the population means are "
                     f"{'significantly' if p < alpha else 'not significantly'} different."])
    r2 = ss_b / (ss_b + ss_w) if ss_b + ss_w > 0 else math.nan
    fresh.add_table("Fit Statistics", ["R-Square", "Coeff Var", "Root MSE", "Data Mean"],
                    [[num(r2), num(math.sqrt(mse) / grand if grand else math.nan),
                      num(math.sqrt(mse)), num(grand)]])
    _means_tables(fresh, groups, methods, mse, df_w, alpha,
                  control=settings.get("control") or None,
                  tail=settings.get("dunnett_tail", "two"))
    homog = settings.get("homogeneity", "levene")
    if homog != "none":
        center = "median" if homog == "brown_forsythe" else "mean"
        w, ph = stats.levene(*[v for _, v in groups], center=center)
        fresh.add_table("Homogeneity of Variance Test",
                        ["", "DF", "Statistic", "Prob>F", f"Decision at level({_level(alpha)})"],
                        [["Brown-Forsythe" if center == "median" else
                          "Levene's Test (Absolute Deviations)", f"{df_b}, {df_w}", num(w, 5),
                          pval(ph), _decision(ph, alpha, "Variances differ",
                                              "Equal variances")]])
    if settings.get("means_plot", True):
        fresh.add_graph("Means Plot", rg.means_graph(groups), width=420)
    report = _finish(report, fresh)
    return report, f"{_stamp(first.book)}\nOne-Way ANOVA: F({df_b}, {df_w}) = {num(f, 5)}, " \
                   f"p = {pval(p)} → [{first.book.name}]{report.name}"


SS_TYPES = {"3": "Type III", "2": "Type II", "1": "Type I (sequential)"}


def level_names(text: str, n: int, prefix: str) -> list[str]:
    """Level names typed as "Male, Female" (missing ones become A1, A2…)."""
    names = [t.strip() for t in re.split(r"[,;\t]", text or "") if t.strip()]
    names = names[:n]
    used = set(names)
    k = len(names)
    while len(names) < n:
        k += 1
        cand = f"{prefix}{k}"
        if cand not in used:
            names.append(cand)
    if len(set(names)) != len(names):
        raise StatError(f"Level names must be different ({', '.join(names)}).")
    return names


@dataclass
class TwoWayData:
    inputs: list[RangeData]
    rows: list[tuple[str, str, float]]   # (level of A, level of B, response)
    a_title: str
    b_title: str
    levels_a: list[str]
    levels_b: list[str]
    data_title: str
    raw: bool


def two_way_data(books: dict, settings: dict) -> TwoWayData:
    """The observations of a Two-Way ANOVA from Origin's two input forms.

    **Raw**: one column per factor-level combination, Factor A outermost — with levels
    A1, A2 of A and B1, B2, B3 of B the ranges are A1B1, A1B2, A1B3, A2B1, A2B2, A2B3 (Origin's
    raw layout, e.g. Male SV, Male LV, Male Nor, Female SV…); the factor names, numbers of
    levels and level names come from the dialog. Columns may have different lengths
    (unbalanced design). **Indexed**: a data column and two factor columns."""
    if settings.get("form", "indexed") == "raw":
        inputs = read_ranges(books, settings.get("ranges", []))
        na, nb = int(settings.get("levels_a", 2)), int(settings.get("levels_b", 2))
        if na < 2 or nb < 2:
            raise StatError("Each factor needs at least two levels.")
        if len(inputs) != na * nb:
            raise StatError(f"Raw data need one column per level combination: {na} × {nb} = "
                            f"{na * nb} ranges (got {len(inputs)}).")
        a_title = (settings.get("name_a") or "Factor A").strip()
        b_title = (settings.get("name_b") or "Factor B").strip()
        la = level_names(settings.get("names_a", ""), na, "A")
        lb = level_names(settings.get("names_b", ""), nb, "B")
        rows = [(la[k // nb], lb[k % nb], float(v)) for k, r in enumerate(inputs)
                for v in r.finite]
        titles = {r.column.long_name for r in inputs if r.column.long_name}
        return TwoWayData(inputs, rows, a_title, b_title, la, lb,
                          titles.pop() if len(titles) == 1 else "Data", True)
    data = read_range(books, settings.get("data", ""))
    fa = read_range(books, settings.get("factor_a", ""))
    fb = read_range(books, settings.get("factor_b", ""))
    n = min(len(data.values), len(fa.texts), len(fb.texts))
    rows = [(fa.texts[i], fb.texts[i], float(data.values[i])) for i in range(n)
            if fa.texts[i] != "" and fb.texts[i] != "" and np.isfinite(data.values[i])]
    return TwoWayData([data, fa, fb], rows, fa.title, fb.title,
                      _sort_levels([r[0] for r in rows]), _sort_levels([r[1] for r in rows]),
                      data.title, False)


def two_way_anova(books: dict, settings: dict, report: ReportSheet | None = None,
                  mode: str = "manual") -> tuple[ReportSheet, str]:
    """Two-Way ANOVA of raw or indexed data, balanced or unbalanced, with Type I, II or III
    sums of squares (Origin uses Type III; statsmodels ``anova_lm`` with sum-to-zero
    contrasts gives the numbers)."""
    import pandas as pd
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    tw = two_way_data(books, settings)
    if len(tw.rows) < 4:
        raise StatError("Two-Way ANOVA needs data with two grouping columns.")
    frame = pd.DataFrame(tw.rows, columns=["A", "B", "y"])
    la, lb = [lv for lv in tw.levels_a if (frame.A == lv).any()], \
        [lv for lv in tw.levels_b if (frame.B == lv).any()]
    if len(la) < 2 or len(lb) < 2:
        raise StatError("Each factor needs at least two levels.")
    interaction = settings.get("interaction", True)
    alpha = float(settings.get("alpha", 0.05))
    ss_type = str(settings.get("ss", "3"))
    if ss_type not in SS_TYPES:
        raise StatError(f"Unknown type of sums of squares {ss_type}.")
    methods = [m for m in settings.get("means", ["tukey"]) if m in MEANS_METHODS]
    counts = frame.groupby(["A", "B"]).size()
    if interaction and len(counts) < len(la) * len(lb):
        missing = [f"{a} {b}" for a in la for b in lb if (a, b) not in counts.index]
        raise StatError("Some level combinations have no data (" + ", ".join(missing) +
                        "): turn off Interactions to fit the main effects only.")
    balanced = counts.nunique() == 1 and len(counts) == len(la) * len(lb)
    rhs = "C(A, Sum) * C(B, Sum)" if interaction else "C(A, Sum) + C(B, Sum)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.ols(f"y ~ {rhs}", data=frame).fit()
        table = anova_lm(model, typ=int(ss_type))
    df_e, ss_e = float(model.df_resid), float(model.ssr)
    if df_e <= 0:
        raise StatError("No degrees of freedom for error: each cell needs replicates "
                        "(or turn off the interaction).")
    mse = ss_e / df_e
    an, bn = tw.a_title, tw.b_title
    names = {"C(A, Sum)": an, "C(B, Sum)": bn, "C(A, Sum):C(B, Sum)": f"{an} * {bn}"}
    first = tw.inputs[0]
    report, fresh = _new_report(first, f"Two-Way ANOVA of {first.text}" +
                                (" …" if tw.raw and len(tw.inputs) > 1 else ""),
                                "ANOVA2Way", report, "anova2", settings, mode)
    extra = [["Data Form", "Raw" if tw.raw else "Indexed"]]
    if tw.raw:
        extra += [[f"{an} Levels", ", ".join(tw.levels_a)],
                  [f"{bn} Levels", ", ".join(tw.levels_b)]]
    _notes(fresh, "Two-Way ANOVA", tw.inputs, extra + [
        ["Interactions", "Yes" if interaction else "No"],
        ["Sums of Squares", SS_TYPES[ss_type]],
        ["Design", "Balanced" if balanced else "Unbalanced"],
        ["Significance Level", num(alpha)]])
    desc = []
    for a_ in la:
        for b_ in lb:
            v = frame.y[(frame.A == a_) & (frame.B == b_)].to_numpy(float)
            if len(v):
                desc.append([f"{a_}  {b_}"] + _desc_row("", v)[1:])
    fresh.add_table("Descriptive Statistics", [f"{an}  {bn}", "N", "Mean", "SD", "SEM"], desc)
    out_rows, lines = [], []
    for term, row in table.iterrows():
        if term in ("Intercept", "Residual"):
            continue
        ss, df, f, p = (float(row["sum_sq"]), float(row["df"]), float(row["F"]),
                        float(row["PR(>F)"]))
        out_rows.append([names.get(term, term), num(df), num(ss), num(ss / df), num(f, 5),
                         pval(p)])
        lines.append(f"  {names.get(term, term)}: F({num(df)}, {num(df_e)}) = {num(f, 5)}, "
                     f"p = {pval(p)}")
    ss_tot = float(((frame.y - frame.y.mean()) ** 2).sum())
    ss_model_total = ss_tot - ss_e
    df_model_total = float(model.df_model)
    f_m = (ss_model_total / df_model_total) / mse
    out_rows += [["Model", num(df_model_total), num(ss_model_total),
                  num(ss_model_total / df_model_total), num(f_m, 5),
                  pval(float(stats.f.sf(f_m, df_model_total, df_e)))],
                 ["Error", num(df_e), num(ss_e), num(mse), "", ""],
                 ["Corrected Total", num(len(frame) - 1), num(ss_tot), "", "", ""]]
    notes = [f"At the {num(alpha)} level, a term with Prob>F < {num(alpha)} has a "
             "significant effect."]
    if not balanced:
        notes.append(f"Unbalanced design: {SS_TYPES[ss_type]} sums of squares" +
                     (" depend on the order of the terms." if ss_type == "1" else "."))
    fresh.add_table("Overall ANOVA", ["", "DF", "Sum of Squares", "Mean Square", "F Value",
                                      "Prob>F"], out_rows, notes)
    fresh.add_table("Fit Statistics", ["R-Square", "Coeff Var", "Root MSE", "Data Mean"],
                    [[num(ss_model_total / ss_tot if ss_tot else math.nan),
                      num(math.sqrt(mse) / frame.y.mean() if frame.y.mean() else math.nan),
                      num(math.sqrt(mse)), num(frame.y.mean())]])
    legacy = settings.get("control") or None   # one control name typed for either factor
    for title, col, levels, key in ((an, "A", la, "control_a"), (bn, "B", lb, "control_b")):
        groups = [(lv, frame.y[frame[col] == lv].to_numpy(float)) for lv in levels]
        control = settings.get(key) or (legacy if legacy in levels else None)
        _means_tables(fresh, groups, methods, mse, df_e, alpha, title, control,
                      settings.get("dunnett_tail", "two"))
    if settings.get("means_plot", True):
        means = np.array([[frame.y[(frame.A == a_) & (frame.B == b_)].mean() for b_ in lb]
                          for a_ in la], float)
        fresh.add_graph("Interaction Plot", rg.interaction_graph(la, lb, means, an, bn,
                                                                 tw.data_title), width=420)
    report = _finish(report, fresh)
    return report, f"{_stamp(first.book)}\nTwo-Way ANOVA ({SS_TYPES[ss_type]}) → " \
                   f"[{first.book.name}]{report.name}\n" + "\n".join(lines)


# ====================================================================== nonparametric
NONPARAMETRIC = {"mann_whitney": "Mann-Whitney Test", "wilcoxon": "Paired Sample Wilcoxon "
                 "Signed Rank Test", "kruskal": "Kruskal-Wallis ANOVA",
                 "friedman": "Friedman ANOVA"}


def nonparametric(books: dict, settings: dict, report: ReportSheet | None = None,
                  mode: str = "manual") -> tuple[ReportSheet, str]:
    test = settings.get("test", "mann_whitney")
    alpha = float(settings.get("alpha", 0.05))
    tail = settings.get("tail", "two")
    if test in ("wilcoxon", "friedman"):
        inputs = read_ranges(books, settings.get("ranges", []))
        need = 2 if test == "wilcoxon" else 3
        if len(inputs) < need or (test == "wilcoxon" and len(inputs) != 2):
            raise StatError(f"{NONPARAMETRIC[test]} needs "
                            f"{'two' if test == 'wilcoxon' else 'three or more'} columns.")
        n = min(len(r.values) for r in inputs)
        m = np.column_stack([r.values[:n] for r in inputs])
        m = m[np.isfinite(m).all(axis=1)]
        if len(m) < 2:
            raise StatError("Not enough complete rows.")
        groups = [(r.title, m[:, k]) for k, r in enumerate(inputs)]
    else:
        inputs, groups = _groups(books, settings, 2)
        if test == "mann_whitney" and len(groups) != 2:
            raise StatError("The Mann-Whitney test compares exactly two groups.")
    first = inputs[0]
    report, fresh = _new_report(first, f"{NONPARAMETRIC[test]} of {first.text}",
                                {"mann_whitney": "MannWhitney", "wilcoxon": "Wilcoxon",
                                 "kruskal": "KWANOVA", "friedman": "FriedmanANOVA"}[test],
                                report, "nonparam", settings, mode)
    _notes(fresh, NONPARAMETRIC[test], inputs, [["Significance Level", num(alpha)]])
    allv = np.concatenate([v for _, v in groups])
    ranks = stats.rankdata(allv)
    pos = 0
    desc = []
    for name, v in groups:
        r = ranks[pos:pos + len(v)]
        pos += len(v)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        desc.append([name, str(len(v)), num(med), num(q1), num(q3), num(r.mean())])
    fresh.add_table("Descriptive Statistics", ["", "N", "Median", "Q1", "Q3", "Mean Rank"],
                    desc)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if test == "mann_whitney":
            (n1, a), (n2, b) = groups
            res = stats.mannwhitneyu(a, b, alternative=_ALT[tail], method="asymptotic")
            u = float(res.statistic)
            mu, sd = len(a) * len(b) / 2, math.sqrt(len(a) * len(b) * (len(a) + len(b) + 1)
                                                     / 12)
            z = (u - mu) / sd
            rows = [[f"{n1} − {n2}", num(u), num(z, 5), pval(res.pvalue)]]
            header = ["", "U", "Z", "Prob"]
            p = float(res.pvalue)
        elif test == "wilcoxon":
            (n1, a), (n2, b) = groups
            res = stats.wilcoxon(a, b, alternative=_ALT[tail])
            rows = [[f"{n1} − {n2}", num(res.statistic), str(len(a)), pval(res.pvalue)]]
            header = ["", "W", "N", "Prob"]
            p = float(res.pvalue)
        elif test == "kruskal":
            res = stats.kruskal(*[v for _, v in groups])
            rows = [["Kruskal-Wallis", num(res.statistic, 5), str(len(groups) - 1),
                     pval(res.pvalue)]]
            header = ["", "Chi-Square", "DF", "Prob>Chi-Square"]
            p = float(res.pvalue)
        else:
            res = stats.friedmanchisquare(*[v for _, v in groups])
            rows = [["Friedman", num(res.statistic, 5), str(len(groups) - 1),
                     pval(res.pvalue)]]
            header = ["", "Chi-Square", "DF", "Prob>Chi-Square"]
            p = float(res.pvalue)
    fresh.add_table("Test Statistics", header, rows,
                    [f"At the {num(alpha)} level, the populations are "
                     f"{'significantly' if p < alpha else 'not significantly'} different."])
    report = _finish(report, fresh)
    return report, f"{_stamp(first.book)}\n{NONPARAMETRIC[test]}: p = {pval(p)} → " \
                   f"[{first.book.name}]{report.name}"


# ====================================================================== registry / recalc
REPORT_TESTS = {"desc": desc_stats, "normality": normality, "correlation": correlation,
                "ttest1": one_sample_t, "ttest2": two_sample_t, "ttest_pair": paired_t,
                "anova1": one_way_anova, "anova2": two_way_anova, "nonparam": nonparametric}


# analyses added by other modules (stat_extra, stat_multivariate, stat_survival,
# stat_power_roc) register themselves here on import; :func:`load_extensions` imports them
# so recalculating a saved report works without the menus being built first.
EXTENSION_MODULES = ("rsstatchem.ui.origin.stat_extra", "rsstatchem.ui.origin.stat_multivariate",
                     "rsstatchem.ui.origin.stat_survival", "rsstatchem.ui.origin.stat_power_roc")
_EXTENSIONS_LOADED = False


def load_extensions() -> None:
    global _EXTENSIONS_LOADED
    if _EXTENSIONS_LOADED:
        return
    _EXTENSIONS_LOADED = True
    import importlib
    for name in EXTENSION_MODULES:
        importlib.import_module(name)


DIALOG_SPECS: dict = {}   # key -> spec(selection) -> (title, groups, description, output)


def register(key: str, fn, spec=None) -> None:
    """Add a report analysis ``fn(books, settings, report=None, mode="manual")`` and the
    field spec of its dialog (see :func:`stat_dialogs.spec`)."""
    REPORT_TESTS[key] = fn
    if spec is not None:
        DIALOG_SPECS[key] = spec


def run(key: str, books: dict, settings: dict, report: ReportSheet | None = None,
        mode: str = "manual") -> tuple[ReportSheet, str]:
    """Run a report analysis (new report, or recalculate ``report`` in place)."""
    if key not in REPORT_TESTS:
        load_extensions()
    if key not in REPORT_TESTS:
        raise StatError(f"Unknown analysis {key!r}.")
    try:
        return REPORT_TESTS[key](books, settings, report=report, mode=mode)
    except (ValueError, ZeroDivisionError, np.linalg.LinAlgError) as exc:
        if isinstance(exc, StatError):
            raise
        raise StatError(str(exc)) from None


def range_strings(settings: dict) -> list[str]:
    """Every input range in a settings dict."""
    out = []
    for key, value in settings.items():
        if key in RANGE_LIST_KEYS and isinstance(value, list):
            out += [v for v in value if isinstance(v, str) and v.strip()]
        elif key in RANGE_KEYS and isinstance(value, str) and value.strip():
            out.append(value)
    return out


RANGE_LIST_KEYS = ("ranges", "response", "covariates")
RANGE_KEYS = ("range", "range1", "range2", "data", "factor", "factor_a", "factor_b",
              "factor_c", "time", "censor", "group", "state", "labels", "subject")

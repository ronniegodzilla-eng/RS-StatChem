"""Statistics menu analyses with Origin-style dialogs and report sheets (no Qt).

Each analysis takes the workbooks (``books``: name -> Workbook) and a ``settings`` dict as
collected by its dialog (:mod:`rsstatchem.ui.origin.stat_dialogs`): input ranges in Origin's
notation (``[Book1]Sheet1!B`` or ``[Book1]Sheet1!B[3:10]``) and the options. It writes a
report sheet into the book of the first input — Descriptive Statistics, Normality Test,
Correlation, t-tests, ANOVA with means comparisons, nonparametric tests — and returns
``(report, results-log text)``. The report remembers the analysis in ``report.op`` so it can
be recalculated when the source data change (:func:`run` with ``report=``).

Numbers come from scipy / statsmodels and, for the means comparisons, from
:func:`rsstatchem.core.stats.anova.posthoc` (Tukey, Bonferroni, Šidák, Scheffé, Fisher LSD).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from scipy import stats

from rsstatchem.core.analysis.common import figure, num, png, pval
from rsstatchem.core.stats.anova import posthoc
from rsstatchem.core.stats.common import Group
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
    return RangeData(wb, ws, col, (lo + 1, hi) if rows else None,
                     c.numeric(max(hi, 1))[lo:hi], c.texts(max(hi, 1))[lo:hi])


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


def _fig_png(draw, w: float = 5.0, h: float = 3.4) -> bytes:
    fig = figure(w, h)
    ax = fig.add_subplot(111)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        draw(ax)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    ax.tick_params(direction="out", labelsize=8)
    return png(fig)


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
            v = r.finite
            if len(v):
                fresh.add_image(f"Histogram of {r.title}", _fig_png(lambda ax, v=v, r=r: (
                    ax.hist(v, bins="auto", color="#e57373", edgecolor="#000", lw=0.6),
                    ax.set_xlabel(r.column.axis_title), ax.set_ylabel("Counts"))),
                    width=380, group="Histograms")
    if settings.get("box"):
        data = [r.finite for r in inputs if len(r.finite)]
        if data:
            fresh.add_image("Box Chart", _fig_png(lambda ax: (
                ax.boxplot(data, tick_labels=[r.title for r in inputs if len(r.finite)],
                           patch_artist=True,
                           boxprops={"facecolor": "#9ec2f0"}, medianprops={"color": "#000"}),
                ax.set_ylabel("Value"))), width=420, group="Box Charts")
    report = _finish(report, fresh)
    text = f"{_stamp(first.book)}\nStatistics on Columns → [{first.book.name}]{report.name}"
    for r, d in zip(inputs, results):
        text += f"\n  {r.title:<14} N = {int(d['n'])}  Mean = {num(d.get('mean'))}  " \
                f"SD = {num(d.get('sd'))}"
    return report, text


ROW_QUANTITIES = {"mean": "Mean", "sd": "Standard Deviation", "se": "SE of Mean",
                  "n": "N", "sum": "Sum", "min": "Minimum", "median": "Median",
                  "max": "Maximum", "range": "Range"}


def stats_on_rows(books: dict, settings: dict) -> tuple[Worksheet, list[int], str]:
    """Statistics on Rows: append the chosen quantities as new columns of the source sheet
    (one value per row across the input columns)."""
    inputs = read_ranges(books, settings.get("ranges", []))
    ws = inputs[0].sheet
    if any(r.sheet is not ws for r in inputs):
        raise StatError("Statistics on Rows needs columns of one worksheet.")
    if len(inputs) < 2:
        raise StatError("Select two or more columns.")
    keys = [k for k in settings.get("quantities", ("mean", "sd", "n")) if k in ROW_QUANTITIES]
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
    added = []
    for k in keys:
        col = ws.add_column("yEr" if k in ("sd", "se") else "Y")
        col.long_name, col.comments = ROW_QUANTITIES[k], f"{ROW_QUANTITIES[k]} of {names}"
        idx = len(ws.columns) - 1
        vals = [None] * ws.n_rows
        for i, v in enumerate(out[k]):
            if lo + i < ws.n_rows and np.isfinite(v):
                vals[lo + i] = float(v)
        ws.set_column_values(idx, vals)
        added.append(idx)
    return ws, added, (f"{_stamp(inputs[0].book)}\nStatistics on Rows of {names}: " +
                       ", ".join(ROW_QUANTITIES[k] for k in keys))


# ====================================================================== frequency counts
def frequency_counts(books: dict, settings: dict) -> tuple[Worksheet, str]:
    """Statistics > Descriptive Statistics > Frequency Counts: a worksheet of bin centers,
    counts, cumulative counts and relative frequencies (discrete counts for text data)."""
    r = read_range(books, settings.get("range", ""))
    book = r.book
    out = Worksheet(name=book.unique_sheet_name("FreqCounts"))
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
        out.columns.clear()
        out.set_rows(max(len(levels), 1))
        for title, desig, vals in cols:
            c = out.add_column("X" if title == "Data" else desig)
            c.long_name = title
            out.set_column_values(len(out.columns) - 1, list(vals))
        out.columns[0].designation = "X"
        book.sheets.append(out)
        return out, (f"{_stamp(book)}\nDiscrete Frequency of {r.text}: {len(levels)} "
                     f"categories → [{book.name}]{out.name}")
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
    out.set_rows(len(centers))
    for title, desig, vals in cols:
        c = out.add_column(desig)
        c.long_name = title
        out.set_column_values(len(out.columns) - 1, list(vals))
    out.columns[3].comments = f"counts of {r.text}"
    book.sheets.append(out)
    outside = int(np.sum((v < lo) | (v > edges[-1])))
    return out, (f"{_stamp(book)}\nFrequency Counts of {r.text}: {nb} bins of width "
                 f"{num(step)} from {num(lo)} → [{book.name}]{out.name}" +
                 (f" ({outside} values outside the bins)" if outside else ""))


# ====================================================================== normality
NORMALITY_TESTS = {"shapiro": "Shapiro-Wilk", "ks": "Kolmogorov-Smirnov",
                   "lilliefors": "Lilliefors", "anderson": "Anderson-Darling",
                   "dagostino": "D'Agostino-K squared", "jb": "Jarque-Bera"}


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
        fresh.add_table(NORMALITY_TESTS[t], ["", "DF", "Statistic", "p-value",
                                             f"Decision at level({_level(alpha)})"], rows,
                        [f"At the {num(alpha)} level, the data were significantly drawn from "
                         "a normally distributed population unless the decision rejects "
                         "normality."])
    if settings.get("qq", True):
        for r in inputs:
            v = r.finite
            if len(v) >= 3:
                def draw(ax, v=v):
                    (osm, osr), (slope, icpt, _) = stats.probplot(v, dist="norm")
                    ax.plot(osm, osr, "s", ms=4, mfc="#1f5fbf", mec="#1f5fbf")
                    ax.plot(osm, slope * np.asarray(osm) + icpt, "-", color="#d62728", lw=1.2)
                    ax.set_xlabel("Theoretical Quantiles (Normal)")
                    ax.set_ylabel(r.column.axis_title)
                fresh.add_image(f"Q-Q Plot of {r.title}", _fig_png(draw), width=380,
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
        fresh.add_image("Scatter Plot", _fig_png(lambda ax: (
            ax.plot(a[ok], b[ok], "s", ms=4, color="#1f5fbf"),
            ax.set_xlabel(inputs[0].column.axis_title),
            ax.set_ylabel(inputs[1].column.axis_title))), width=380)
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
MEANS_METHODS = {"tukey": "Tukey", "bonferroni": "Bonferroni", "sidak": "Sidak",
                 "scheffe": "Scheffe", "lsd": "Fisher LSD"}


def means_comparison_rows(groups: list[tuple[str, np.ndarray]], method: str, mse: float,
                          df_error: float, alpha: float) -> list[list[str]]:
    """Origin's Means Comparisons table rows ("B A" = mean of B − mean of A)."""
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
                  factor: str = "") -> None:
    for m in methods:
        stat = "q Value" if m == "tukey" else "t Value"
        fresh.add_table(f"{MEANS_METHODS[m]} Test" + (f" ({factor})" if factor else ""),
                        ["", "MeanDiff", "SEM", stat, "Prob", "Alpha", "Sig", "LCL", "UCL"],
                        means_comparison_rows(groups, m, mse, df_error, alpha),
                        [f"Sig equals 1 indicates that the difference of the means is "
                         f"significant at the {num(alpha)} level.",
                         "Sig equals 0 indicates that the difference of the means is not "
                         f"significant at the {num(alpha)} level."],
                        group="Means Comparisons")


def _means_plot(groups, alpha: float) -> bytes:
    def draw(ax):
        names = [g for g, _ in groups]
        means = [v.mean() for _, v in groups]
        se = [v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else 0 for _, v in groups]
        ax.errorbar(range(len(groups)), means, yerr=se, fmt="s-", color="#1f1f1f",
                    mfc="#d62728", capsize=4, lw=1)
        ax.set_xticks(range(len(groups)), names)
        ax.set_ylabel("Mean ± SE")
        ax.margins(x=0.15)
    return _fig_png(draw, 4.8, 3.2)


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
    ss_b = float(sum(len(v) * (v.mean() - grand) ** 2 for _, v in groups))
    ss_w = float(sum(((v - v.mean()) ** 2).sum() for _, v in groups))
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
    _means_tables(fresh, groups, methods, mse, df_w, alpha)
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
        fresh.add_image("Means Plot", _means_plot(groups, alpha), width=420)
    report = _finish(report, fresh)
    return report, f"{_stamp(first.book)}\nOne-Way ANOVA: F({df_b}, {df_w}) = {num(f, 5)}, " \
                   f"p = {pval(p)} → [{first.book.name}]{report.name}"


def two_way_anova(books: dict, settings: dict, report: ReportSheet | None = None,
                  mode: str = "manual") -> tuple[ReportSheet, str]:
    """Two-Way ANOVA (indexed data, Type III sums of squares as in Origin)."""
    import pandas as pd
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    data = read_range(books, settings.get("data", ""))
    fa = read_range(books, settings.get("factor_a", ""))
    fb = read_range(books, settings.get("factor_b", ""))
    n = min(len(data.values), len(fa.texts), len(fb.texts))
    rows = [(fa.texts[i], fb.texts[i], data.values[i]) for i in range(n)
            if fa.texts[i] != "" and fb.texts[i] != "" and np.isfinite(data.values[i])]
    if len(rows) < 4:
        raise StatError("Two-Way ANOVA needs data with two grouping columns.")
    frame = pd.DataFrame(rows, columns=["A", "B", "y"])
    la, lb = _sort_levels(list(frame.A)), _sort_levels(list(frame.B))
    if len(la) < 2 or len(lb) < 2:
        raise StatError("Each factor needs at least two levels.")
    interaction = settings.get("interaction", True)
    alpha = float(settings.get("alpha", 0.05))
    methods = [m for m in settings.get("means", ["tukey"]) if m in MEANS_METHODS]
    rhs = "C(A, Sum) * C(B, Sum)" if interaction else "C(A, Sum) + C(B, Sum)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.ols(f"y ~ {rhs}", data=frame).fit()
        table = anova_lm(model, typ=3)
    df_e, ss_e = float(model.df_resid), float(model.ssr)
    if df_e <= 0:
        raise StatError("No degrees of freedom for error: each cell needs replicates "
                        "(or turn off the interaction).")
    mse = ss_e / df_e
    an, bn = fa.title, fb.title
    names = {"C(A, Sum)": an, "C(B, Sum)": bn, "C(A, Sum):C(B, Sum)": f"{an} * {bn}"}
    report, fresh = _new_report(data, f"Two-Way ANOVA of {data.text}", "ANOVA2Way", report,
                                "anova2", settings, mode)
    _notes(fresh, "Two-Way ANOVA", [data, fa, fb],
           [["Interactions", "Yes" if interaction else "No"],
            ["Sums of Squares", "Type III"], ["Significance Level", num(alpha)]])
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
    fresh.add_table("Overall ANOVA", ["", "DF", "Sum of Squares", "Mean Square", "F Value",
                                      "Prob>F"], out_rows,
                    [f"At the {num(alpha)} level, a term with Prob>F < {num(alpha)} has a "
                     "significant effect."])
    fresh.add_table("Fit Statistics", ["R-Square", "Coeff Var", "Root MSE", "Data Mean"],
                    [[num(ss_model_total / ss_tot if ss_tot else math.nan),
                      num(math.sqrt(mse) / frame.y.mean() if frame.y.mean() else math.nan),
                      num(math.sqrt(mse)), num(frame.y.mean())]])
    for title, col, levels in ((an, "A", la), (bn, "B", lb)):
        groups = [(lv, frame.y[frame[col] == lv].to_numpy(float)) for lv in levels]
        _means_tables(fresh, groups, methods, mse, df_e, alpha, title)
    if settings.get("means_plot", True):
        def draw(ax):
            for k, b_ in enumerate(lb):
                means = [frame.y[(frame.A == a_) & (frame.B == b_)].mean() for a_ in la]
                ax.plot(range(len(la)), means, "s-", label=f"{bn} = {b_}", lw=1.2, ms=5)
            ax.set_xticks(range(len(la)), la)
            ax.set_xlabel(an)
            ax.set_ylabel(f"Mean of {data.title}")
            ax.legend(fontsize=7, frameon=False)
        fresh.add_image("Interaction Plot", _fig_png(draw, 4.8, 3.2), width=420)
    report = _finish(report, fresh)
    return report, f"{_stamp(data.book)}\nTwo-Way ANOVA → [{data.book.name}]{report.name}\n" \
        + "\n".join(lines)


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


def run(key: str, books: dict, settings: dict, report: ReportSheet | None = None,
        mode: str = "manual") -> tuple[ReportSheet, str]:
    """Run a report analysis (new report, or recalculate ``report`` in place)."""
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
        if key in ("ranges",) and isinstance(value, list):
            out += [v for v in value if isinstance(v, str)]
        elif key in ("range", "range1", "range2", "data", "factor", "factor_a", "factor_b") \
                and isinstance(value, str) and value:
            out.append(value)
    return out

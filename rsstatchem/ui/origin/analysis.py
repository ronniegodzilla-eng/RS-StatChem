"""Origin-style analyses on worksheet columns / graph plots (no Qt).

Results are placed the way Origin places them: fit reports become report sheets
(``FitLinear1``, ``FitPolynomial1``, ``FitNL1``…) plus a fitted-curve data sheet
(``FitLinearCurve1``…) in the source workbook, with a short text for the Results Log;
signal-processing and mathematics outputs become new columns of the source sheet; statistics
become report sheets (``DescStatsOnCols1``…). The numbers come from
:mod:`rsstatchem.core.analysis`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from scipy import stats

from rsstatchem.core.analysis import fitting as fit
from rsstatchem.core.analysis import mathematics as maths
from rsstatchem.core.analysis import signal as sig
from rsstatchem.core.analysis.common import num, pval
from rsstatchem.core.analysis.fitting import FitResult
from rsstatchem.ui.origin import report_graphs as rg
from rsstatchem.ui.origin.model import (ReportSheet, Workbook, Worksheet, jsonable, row_window,
                                        rows_text)


class AnalysisError(ValueError):
    pass


# ====================================================================== input data
@dataclass
class XYData:
    """One (x, y) data set taken from a worksheet: the rows used and their values."""
    book: Workbook
    sheet: Worksheet
    ycol: int
    xcol: int | None
    rows: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yerr: np.ndarray | None = None
    row_range: tuple[int, int] | None = None   # 1-based inclusive, None = whole column

    @property
    def xtitle(self) -> str:
        return self.sheet.columns[self.xcol].axis_title if self.xcol is not None else "Row"

    @property
    def ytitle(self) -> str:
        return self.sheet.columns[self.ycol].axis_title

    @property
    def yname(self) -> str:
        return self.sheet.columns[self.ycol].title

    @property
    def range_text(self) -> str:
        """Origin's range notation, e.g. ``[Book1]Sheet1!B"Signal"`` or, for a row range,
        ``[Book1]Sheet1!B"Signal"[3:10]``."""
        c = self.sheet.columns[self.ycol]
        long = f'"{c.long_name}"' if c.long_name else ""
        return f"[{self.book.name}]{self.sheet.name}!{c.name}{long}" + \
            rows_text(*(self.row_range or (None, None)))

    def source(self) -> dict:
        """The input as stored in a report's recalculation info."""
        ws = self.sheet
        return {"book": self.book.name, "sheet": ws.name, "y": ws.columns[self.ycol].name,
                "x": ws.columns[self.xcol].name if self.xcol is not None else None,
                "rows": list(self.row_range) if self.row_range else None}


def xy_data(book: Workbook, ws: Worksheet, ycol: int, xcol: int | None = -1,
            with_error: bool = False, rows: tuple[int, int] | None = None) -> XYData:
    """Column ``ycol`` against its X (``xcol=-1``: the designated X; None: row numbers),
    over the 1-based inclusive row range ``rows`` (None: the whole column)."""
    if xcol == -1:
        xcol = ws.x_of(ycol)
    n = ws.n_rows
    lo, hi = row_window(n, rows) if rows else (0, n)
    y = ws.columns[ycol].numeric(n)
    x = ws.columns[xcol].numeric(n) if xcol is not None else np.arange(1, n + 1, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    ok[:lo] = False
    ok[hi:] = False
    e = None
    ecol = ws.yerr_of(ycol) if with_error else None
    if ecol is not None:   # error bars / instrumental weights, when complete
        e_all = ws.columns[ecol].numeric(n)[ok]
        e = e_all if np.isfinite(e_all).all() else None
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        raise AnalysisError(f"Column {ws.columns[ycol].name} has no numeric data" +
                            (" in the selected rows." if rows else "."))
    return XYData(book, ws, ycol, xcol, idx, x[ok], y[ok], e,
                  (lo + 1, hi) if rows else None)


def _stamp(book: Workbook) -> str:
    return f'[{datetime.now():%Y-%m-%d %H:%M} "/{book.name}"]'


def add_report(book: Workbook, report: ReportSheet, base: str) -> ReportSheet:
    report.name = book.unique_sheet_name(base)
    book.sheets.append(report)
    return report


def _table_into(report: ReportSheet, table) -> None:
    report.add_table(table.title, table.columns, table.rows, table.footnotes)


# ====================================================================== fitting
@dataclass
class FitOutcome:
    result: FitResult
    report: ReportSheet
    curve: Worksheet
    log: str
    data: XYData
    curve_x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    curve_y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    curve_name: str = ""


FIT_KINDS = {"linear": ("FitLinear", "Linear Fit"), "poly": ("FitPolynomial", "Polynomial Fit"),
             "nlfit": ("FitNL", "Nonlinear Curve Fit")}


def run_fit(data: XYData, model, kind: str = "nlfit", *, weighting: str = "none",
            params0=None, fixed=None, bounds=None, level: float = 0.95,
            conf_band: bool = True, pred_band: bool = False, residual_plot: bool = True,
            points: int = 200, output_fitted: bool = False, report: ReportSheet | None = None,
            curve: Worksheet | None = None, mode: str = "manual") -> FitOutcome:
    """Fit ``model`` (library name / FitFunction / "y = ..." equation) to ``data`` and put an
    Origin-style report sheet and fitted-curve sheet into the source workbook. With
    ``report`` / ``curve`` (a recalculation) those sheets are updated in place."""
    book, ws = data.book, data.sheet
    f = fit.resolve_model(model)
    w = fit.make_weights(weighting, data.y, data.yerr) if weighting != "none" else None
    if weighting in ("instrumental", "direct") and data.yerr is None:
        raise AnalysisError("Instrumental weighting needs a Y Error column to the right of the "
                            "Y column.")
    try:
        res = fit.fit_curve(data.x, data.y, f, params0=params0, bounds=bounds, fixed=fixed,
                            weights=w, weighting=weighting)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        raise AnalysisError(str(exc)) from None
    base, label = FIT_KINDS.get(kind, FIT_KINDS["nlfit"])
    rerun = report is not None
    new = ReportSheet(title=f"{label} ({f.name}) of {data.range_text}")
    if not rerun:
        report = add_report(book, new, base)
    yname = data.yname
    new.add_table("Notes", ["", "Value"], [
        ["Description", label + (f" ({f.name})" if kind == "nlfit" else "")],
        ["Input Data", f"X = {data.xtitle};  Y = {data.range_text}"],
        ["Model", f.name], ["Equation", f.formula],
        ["Weight", fit.WEIGHTINGS.get(weighting, weighting)],
        ["Number of Points", str(res.n)], ["Fit Status", res.message or "Succeeded"]])
    src = data.source()
    rng = rows_text(*(data.row_range or (None, None)))
    new.add_table("Input Data", ["", "Range", "Data"], [
        ["X", f"[{book.name}]{ws.name}!{src['x']}{rng}" if src["x"] else "(row numbers)",
         data.xtitle],
        ["Y", f"[{book.name}]{ws.name}!{src['y']}{rng}", data.ytitle]] +
        ([["Y Error", f"[{book.name}]{ws.name}!{ws.columns[ws.yerr_of(data.ycol)].name}{rng}",
           "weights"]] if data.yerr is not None and weighting != "none" else []))
    for note in res.notes:
        new.add_text("Warning", note, "warning")
    _table_into(new, fit.parameter_table(res, yname, level))
    _table_into(new, fit.statistics_table(res, yname))
    if res.dof > 0:
        _table_into(new, fit.anova_table(res, yname))
    # fitted curve sheet (Origin's FitLinearCurve1)
    lo, hi = float(np.min(data.x)), float(np.max(data.x))
    xx = np.linspace(lo, hi, max(points, 10)) if hi > lo else np.array([lo])
    yy = res.predict(xx)
    curve_name = curve.name if curve is not None else book.unique_sheet_name(base + "Curve")
    new.add_graph("Fitted Curves Plot",
                  rg.fit_graph(res, data.xtitle, data.ytitle, level, conf_band, pred_band,
                               data.yerr if weighting == "instrumental" else None,
                               title=f"{label} of {yname}"),
                  width=520, open_spec={"kind": "fit", "curve": curve_name, **src})
    if residual_plot:
        new.add_graph("Residual vs. Independent Plot",
                      rg.residual_graph(res, data.xtitle, data.ytitle), width=460,
                      group="Residual Plots")
        new.add_graph("Histogram of the Residual", rg.residual_histogram_graph(res, yname),
                      width=460, group="Residual Plots")
    new.op = {"kind": "fit", "fit_kind": kind, "input": src, "curve": curve_name,
              "mode": report.op.get("mode", mode) if rerun else mode,
              "settings": jsonable({"model": model if isinstance(model, str) else f.name,
                                    "weighting": weighting, "params0": params0,
                                    "fixed": fixed, "bounds": bounds, "level": level,
                                    "conf_band": conf_band, "pred_band": pred_band,
                                    "residual_plot": residual_plot,
                                    "output_fitted": output_fitted, "points": points})}
    if rerun:
        report.replace_items(new)
        report.op.update({k: v for k, v in new.op.items()})
    else:
        report.op = new.op
    report.op["dirty"] = False
    fresh = Worksheet(name=curve_name, n_rows=len(xx))
    cx = fresh.add_column("X")
    cx.long_name, cx.comments = "Independent Variable", data.xtitle
    fresh.set_column_values(0, list(xx))
    curve_title = f"{label} of {yname}"
    cy = fresh.add_column("Y")
    cy.long_name, cy.comments = curve_title, f.formula
    fresh.set_column_values(1, list(yy))
    if conf_band and res.dof > 0:
        a, b = res.confidence_band(xx, level)
        for title, vals in ((f"Lower {level * 100:g}% Confidence Limit", a),
                            (f"Upper {level * 100:g}% Confidence Limit", b)):
            c = fresh.add_column("Y")
            c.long_name = title
            fresh.set_column_values(len(fresh.columns) - 1, list(vals))
    if curve is None:
        curve = fresh
        book.sheets.append(curve)
    else:   # keep the sheet object (graph links point at its name and columns)
        curve.columns, curve.n_rows = fresh.columns, fresh.n_rows
    if output_fitted:
        _append_column(ws, f"Fitted Y ({f.name})", data.rows, res.fitted,
                       comments=f"{label} of {yname}", reuse=rerun)
        _append_column(ws, f"Residual ({f.name})", data.rows, res.residuals,
                       comments=f"{label} of {yname}", reuse=rerun)
    lines = [f"{_stamp(book)}", f"{label} ({f.name}) of {data.range_text} vs X = {data.xtitle}",
             f"  Equation: {f.formula}"]
    for i, pname in enumerate(res.names):
        err = num(res.errors[i]) if res.free[i] else "(fixed)"
        lines.append(f"  {pname:<12} = {num(res.values[i]):>14}  ± {err}")
    lines.append(f"  R-Square (COD) = {num(res.r2)}   Adj. R-Square = {num(res.adj_r2)}")
    if f.meta.get("order") == 1:
        lines.append(f"  Pearson's r = {num(res.pearson_r)}")
    lines.append(f"  Reduced Chi-Sqr = {num(res.red_chisq)}   N = {res.n}")
    lines.append(f"  Report: [{book.name}]{report.name}   Curve: [{book.name}]{curve.name}")
    return FitOutcome(res, report, curve, "\n".join(lines), data, xx, yy, curve_title)


def refit(book: Workbook, report: ReportSheet, books: dict) -> FitOutcome:
    """Re-run the fit that produced ``report`` (Recalculate) with its stored settings."""
    op = report.op
    src = op["input"]
    src_book = books.get(src["book"])
    ws = src_book.sheet(src["sheet"]) if src_book is not None else None
    if not isinstance(ws, Worksheet):
        raise AnalysisError(f"The input sheet [{src['book']}]{src['sheet']} no longer exists.")
    try:
        y = ws.index_of(src["y"])
        x = ws.index_of(src["x"]) if src.get("x") else None
    except KeyError as exc:
        raise AnalysisError(f"The input column {exc} no longer exists.") from None
    data = xy_data(src_book, ws, y, x, with_error=True,
                   rows=tuple(src["rows"]) if src.get("rows") else None)
    s = dict(op.get("settings", {}))
    curve = book.sheet(op.get("curve", ""))
    bounds = {k: tuple(v) for k, v in (s.get("bounds") or {}).items()} or None
    return run_fit(data, s.get("model", "Line"), op.get("fit_kind", "nlfit"),
                   weighting=s.get("weighting", "none"), params0=s.get("params0"),
                   fixed=s.get("fixed"), bounds=bounds, level=s.get("level", 0.95),
                   conf_band=s.get("conf_band", True), pred_band=s.get("pred_band", False),
                   residual_plot=s.get("residual_plot", True), points=s.get("points", 200),
                   output_fitted=s.get("output_fitted", False), report=report,
                   curve=curve if isinstance(curve, Worksheet) else None)


def _append_column(ws: Worksheet, long_name: str, rows, values, comments: str = "",
                   designation: str = "Y", reuse: bool = False) -> int:
    existing = next((i for i, c in enumerate(ws.columns) if c.long_name == long_name and
                     c.comments == comments), None) if reuse else None
    if existing is None:
        col = ws.add_column(designation)
        col.long_name, col.comments = long_name, comments
        idx = len(ws.columns) - 1
    else:
        idx = existing
    out = [None] * ws.n_rows
    for r, v in zip(rows, values):
        out[int(r)] = float(v) if np.isfinite(v) else None
    ws.set_column_values(idx, out)
    return idx


# ====================================================================== signal & mathematics
def smooth_column(data: XYData, method: str = "savgol", window: int = 7,
                  polyorder: int = 2) -> tuple[int, str]:
    try:
        s = sig.smooth(data.x, data.y, method, window, polyorder)
    except ValueError as exc:
        raise AnalysisError(str(exc)) from None
    desc = f"{sig.SMOOTHERS.get(method, method)}, {window} points"
    idx = _append_column(data.sheet, f"Smoothed {data.yname}", data.rows, s, desc)
    return idx, f"{_stamp(data.book)}\nSmoothing of {data.range_text}: {desc} → column " \
                f"{data.sheet.columns[idx].name}"


def differentiate_column(data: XYData, order: int = 1) -> tuple[int, str]:
    try:
        d = maths.differentiate(data.x, data.y, order)
    except ValueError as exc:
        raise AnalysisError(str(exc)) from None
    name = "Derivative" if order == 1 else f"Derivative (order {order})"
    idx = _append_column(data.sheet, f"{name} of {data.yname}", data.rows, d,
                         f"d{'' if order == 1 else order}Y/dX{'' if order == 1 else order}")
    return idx, f"{_stamp(data.book)}\nDifferentiate {data.range_text} (order {order}) → " \
                f"column {data.sheet.columns[idx].name}"


def integrate_column(data: XYData, baseline: str = "zero") -> tuple[int, str]:
    o = np.argsort(data.x, kind="stable")
    try:
        out = maths.integrate(data.x, data.y, baseline)
    except ValueError as exc:
        raise AnalysisError(str(exc)) from None
    cum = np.empty(len(o))
    cum[o] = maths.cumulative_area(data.x[o], data.y[o])
    idx = _append_column(data.sheet, f"Integrated {data.yname}", data.rows, cum,
                         "cumulative trapezoidal area")
    text = (f"{_stamp(data.book)}\nIntegrate {data.range_text} (baseline: "
            f"{maths.BASELINES.get(baseline, baseline)})\n  Area = {num(out['area'])}   "
            f"Absolute area = {num(out['abs_area'])}\n  X at max = {num(out['x_max'])}   "
            f"Height = {num(out['y_max'])}   Centroid = {num(out['centroid'])}   "
            f"FWHM = {num(out['fwhm'])}")
    return idx, text


def normalize_column(data: XYData, method: str = "max") -> tuple[int, str]:
    try:
        v = maths.normalize(data.y, method, x=data.x)
    except ValueError as exc:
        raise AnalysisError(str(exc)) from None
    idx = _append_column(data.sheet, f"Normalized {data.yname}", data.rows, v,
                         maths.NORMALIZE.get(method, method))
    return idx, f"{_stamp(data.book)}\nNormalize {data.range_text} " \
                f"({maths.NORMALIZE.get(method, method)}) → column {data.sheet.columns[idx].name}"


def baseline_columns(data: XYData, method: str = "als", order: int = 2) -> tuple[int, str]:
    try:
        b = sig.baseline(data.x, data.y, method, order)
    except ValueError as exc:
        raise AnalysisError(str(exc)) from None
    _append_column(data.sheet, f"Baseline of {data.yname}", data.rows, b,
                   sig.BASELINE_METHODS.get(method, method))
    idx = _append_column(data.sheet, f"Subtracted {data.yname}", data.rows, data.y - b,
                         "data − baseline")
    return idx, f"{_stamp(data.book)}\nBaseline ({sig.BASELINE_METHODS.get(method, method)}) " \
                f"of {data.range_text} → baseline and subtracted columns"


def fft_sheet(data: XYData, window: str = "hann") -> tuple[Worksheet, str]:
    try:
        out = sig.power_spectrum(data.x, data.y, window)
    except ValueError as exc:
        raise AnalysisError(str(exc)) from None
    ws = Worksheet(name=data.book.unique_sheet_name("FFTResult"), n_rows=len(out["freq"]))
    for title, key, units in (("Frequency", "freq", ""), ("Amplitude", "amplitude", ""),
                              ("Power (MSA)", "power", ""), ("Phase", "phase", "deg")):
        c = ws.add_column("X" if key == "freq" else "Y")
        c.long_name, c.units = title, units
        ws.set_column_values(len(ws.columns) - 1, list(out[key]))
    data.book.sheets.append(ws)
    k = int(np.argmax(out["amplitude"][1:])) + 1 if len(out["amplitude"]) > 1 else 0
    return ws, (f"{_stamp(data.book)}\nFFT of {data.range_text} (window "
                f"{sig.WINDOWS.get(window, window)}): N = {out['n']}, dt = {num(out['dt'])}\n"
                f"  Dominant frequency = {num(out['freq'][k])}  amplitude = "
                f"{num(out['amplitude'][k])}  → sheet {ws.name}")


def peaks_report(data: XYData, direction: str = "positive",
                 min_prominence: float = 0.05) -> tuple[ReportSheet, list[dict], str]:
    peaks = sig.find_peaks(data.x, data.y, direction, min_prominence=min_prominence)
    report = ReportSheet(title=f"Peak Analysis of {data.range_text}")
    add_report(data.book, report, "PeakAnalysis")
    rows = [[str(i + 1), num(p["x"]), num(p["y"]), num(p["prominence"]), num(p["fwhm"]),
             num(p["area"]), num(p["centroid"])] for i, p in enumerate(peaks)]
    report.add_table("Peak Table", ["Peak Index", "X", "Y (Height)", "Prominence", "FWHM",
                                    "Area", "Centroid"], rows,
                     [f"{len(peaks)} peak(s); minimum prominence {min_prominence * 100:g}% of "
                      "the Y range."])
    text = f"{_stamp(data.book)}\nFind Peaks in {data.range_text}: {len(peaks)} peak(s) → " \
           f"[{data.book.name}]{report.name}"
    for i, p in enumerate(peaks[:12]):
        text += f"\n  {i + 1:>2}. X = {num(p['x'])}  Y = {num(p['y'])}  FWHM = {num(p['fwhm'])}"
    return report, peaks, text


def multi_peak_fit(data: XYData, n_peaks: int | None, shape: str = "gauss",
                   baseline: str = "constant") -> FitOutcome:
    try:
        res = sig.multi_peak_fit(data.x, data.y, n_peaks, shape, baseline)
    except (ValueError, RuntimeError) as exc:
        raise AnalysisError(str(exc)) from None
    book = data.book
    report = ReportSheet(title=f"Multiple Peak Fit of {data.range_text}")
    add_report(book, report, "MultiPeakFit")
    report.add_table("Notes", ["", "Value"], [
        ["Input Data", data.range_text], ["Peak Function", sig.SHAPES.get(shape, shape)],
        ["Baseline", baseline], ["Number of Peaks", str(res.model.meta["n_peaks"])],
        ["Fit Status", res.message or "Succeeded"]])
    _table_into(report, fit.parameter_table(res, data.yname))
    _table_into(report, fit.statistics_table(res, data.yname))
    summary = sig.peak_summary(res)
    report.add_table("Peak Summary", ["Peak", "Center", "Center SE", "Height", "FWHM", "Area",
                                      "Area SE", "Area %"],
                     [[str(r["peak"]), num(r["xc"]), num(r["xc_se"]), num(r["height"]),
                       num(r["fwhm"]), num(r["area"]), num(r["area_se"]), num(r["area_pct"], 4)]
                      for r in summary])
    xx = np.linspace(float(np.min(data.x)), float(np.max(data.x)), 400)
    comps = sig.peak_components(res, xx)
    report.add_graph("Fitted Curves Plot", rg.fit_graph(
        res, data.xtitle, data.ytitle, conf=False, grid=xx,
        components=[(f"Peak {i + 1}", c) for i, c in enumerate(comps)],
        title=f"Multiple Peak Fit of {data.yname}"), width=520)
    curve = Worksheet(name=book.unique_sheet_name("MultiPeakFitCurve"), n_rows=len(xx))
    c = curve.add_column("X")
    c.long_name = "Independent Variable"
    curve.set_column_values(0, list(xx))
    c = curve.add_column("Y")
    c.long_name = f"Cumulative Fit Peak of {data.yname}"
    curve.set_column_values(1, list(res.predict(xx)))
    for i, comp in enumerate(comps):
        c = curve.add_column("Y")
        c.long_name = f"Peak{i + 1}"
        curve.set_column_values(len(curve.columns) - 1, list(comp))
    book.sheets.append(curve)
    text = f"{_stamp(book)}\nMultiple Peak Fit ({sig.SHAPES.get(shape, shape)}) of " \
           f"{data.range_text}: R² = {num(res.r2)}"
    for r in summary:
        text += f"\n  Peak {r['peak']}: center = {num(r['xc'])}  FWHM = {num(r['fwhm'])}  " \
                f"area = {num(r['area'])}"
    return FitOutcome(res, report, curve, text, data, xx, res.predict(xx),
                      f"Cumulative Fit Peak of {data.yname}")


# ====================================================================== statistics
DESC_QUANTITIES = ("N total", "N missing", "Mean", "Standard Deviation", "SE of mean",
                   "Lower 95% CI of Mean", "Upper 95% CI of Mean", "Variance", "Sum",
                   "Skewness", "Kurtosis", "Minimum", "1st Quartile (Q1)", "Median",
                   "3rd Quartile (Q3)", "Maximum", "Range")


def _desc(values: np.ndarray, n_rows: int) -> dict[str, float]:
    v = values[np.isfinite(values)]
    n = len(v)
    out = {"N total": float(n), "N missing": float(n_rows - n)}
    if n == 0:
        return out
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else math.nan
    se = sd / math.sqrt(n) if n > 1 else math.nan
    t = stats.t.ppf(0.975, n - 1) if n > 1 else math.nan
    q1, med, q3 = (float(q) for q in np.percentile(v, [25, 50, 75]))
    out.update({"Mean": mean, "Standard Deviation": sd, "SE of mean": se,
                "Lower 95% CI of Mean": mean - t * se, "Upper 95% CI of Mean": mean + t * se,
                "Variance": sd ** 2, "Sum": float(v.sum()),
                "Skewness": float(stats.skew(v, bias=False)) if n > 2 else math.nan,
                "Kurtosis": float(stats.kurtosis(v, bias=False)) if n > 3 else math.nan,
                "Minimum": float(v.min()), "1st Quartile (Q1)": q1, "Median": med,
                "3rd Quartile (Q3)": q3, "Maximum": float(v.max()),
                "Range": float(v.max() - v.min())})
    return out


def statistics_on_columns(book: Workbook, ws: Worksheet, cols: list[int],
                          quantities=DESC_QUANTITIES) -> tuple[ReportSheet, str]:
    """Statistics > Descriptive Statistics > Statistics on Columns."""
    cols = [c for c in cols if not ws.columns[c].is_text()]
    if not cols:
        raise AnalysisError("Select one or more numeric columns.")
    n_rows = ws.data_rows()
    rows = []
    for c in cols:
        d = _desc(ws.columns[c].numeric(n_rows), n_rows)
        rows.append([f"{ws.columns[c].name}  {ws.columns[c].long_name}".strip()] +
                    [num(d.get(q, math.nan)) if not q.startswith("N ") else
                     str(int(d.get(q, 0))) for q in quantities])
    report = ReportSheet(title=f"Descriptive Statistics of [{book.name}]{ws.name}")
    add_report(book, report, "DescStatsOnCols")
    report.add_table("Notes", ["", "Value"], [
        ["Description", "Statistics on Columns"],
        ["Input Data", ", ".join(f"[{book.name}]{ws.name}!{ws.columns[c].name}" for c in cols)],
        ["Moment Computation Method", "Divisor is N − 1 (sample statistics)"],
        ["Quantiles Interpolation", "Linear (type 7)"]])
    report.add_table("Descriptive Statistics", [""] + list(quantities), rows)
    text = f"{_stamp(book)}\nStatistics on Columns of [{book.name}]{ws.name}"
    for r in rows:
        text += f"\n  {r[0]:<14} N = {r[1]}  Mean = {r[3]}  SD = {r[4]}  Min = {r[12]}  " \
                f"Max = {r[16]}"
    return report, text


def statistics_on_rows(ws: Worksheet, cols: list[int]) -> list[int]:
    """Statistics on Rows: append Mean, SD and N columns over the selected columns."""
    cols = [c for c in cols if not ws.columns[c].is_text()]
    if len(cols) < 2:
        raise AnalysisError("Select two or more numeric columns.")
    n = ws.data_rows()
    m = np.column_stack([ws.columns[c].numeric(n) for c in cols])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        counts = np.isfinite(m).sum(axis=1)
        mean = np.nanmean(m, axis=1)
        sd = np.where(counts > 1, np.nanstd(m, axis=1, ddof=1), np.nan)
    rows = np.arange(n)
    names = "".join(ws.columns[c].name for c in cols)
    out = [_append_column(ws, "Mean", rows, mean, f"mean of {names}"),
           _append_column(ws, "SD", rows, sd, f"standard deviation of {names}", "yEr"),
           _append_column(ws, "N", rows, counts.astype(float), "number of values")]
    return out


def correlation(book: Workbook, ws: Worksheet, cols: list[int]) -> tuple[ReportSheet, str]:
    cols = [c for c in cols if not ws.columns[c].is_text()]
    if len(cols) < 2:
        raise AnalysisError("Select two or more numeric columns.")
    n = ws.data_rows()
    names = [ws.columns[c].title for c in cols]
    m = [ws.columns[c].numeric(n) for c in cols]
    r_rows, p_rows, n_rows = [], [], []
    for i in range(len(cols)):
        rr, pp, nn = [names[i]], [names[i]], [names[i]]
        for j in range(len(cols)):
            ok = np.isfinite(m[i]) & np.isfinite(m[j])
            if ok.sum() > 2:
                r, p = stats.pearsonr(m[i][ok], m[j][ok])
                rr.append(num(r, 5))
                pp.append(pval(p) if i != j else "--")
            else:
                rr.append("--")
                pp.append("--")
            nn.append(str(int(ok.sum())))
        r_rows.append(rr)
        p_rows.append(pp)
        n_rows.append(nn)
    report = ReportSheet(title=f"Correlation Coefficient of [{book.name}]{ws.name}")
    add_report(book, report, "CorrCoef")
    report.add_table("Pearson Correlation Coefficients", [""] + names, r_rows)
    report.add_table("Pearson Correlation P-values", [""] + names, p_rows,
                     ["Two-tailed test of H0: ρ = 0."])
    report.add_table("Number of Points", [""] + names, n_rows)
    return report, f"{_stamp(book)}\nCorrelation Coefficient (Pearson) of " \
                   f"{', '.join(names)} → [{book.name}]{report.name}"


def normality_test(book: Workbook, ws: Worksheet, cols: list[int]) -> tuple[ReportSheet, str]:
    cols = [c for c in cols if not ws.columns[c].is_text()]
    if not cols:
        raise AnalysisError("Select one or more numeric columns.")
    n = ws.data_rows()
    rows = []
    for c in cols:
        v = ws.columns[c].numeric(n)
        v = v[np.isfinite(v)]
        if len(v) < 3:
            rows.append([ws.columns[c].title, str(len(v)), "--", "--", "too few points"])
            continue
        w, p = stats.shapiro(v)
        rows.append([ws.columns[c].title, str(len(v)), num(w, 5), pval(p),
                     "Normal" if p >= 0.05 else "Not normal"])
    report = ReportSheet(title=f"Normality Test of [{book.name}]{ws.name}")
    add_report(book, report, "NormalityTest")
    report.add_table("Shapiro-Wilk", ["", "DF", "Statistic", "p-value", "Decision at level(5%)"],
                     rows, ["At the 0.05 level, the data were drawn from a normally "
                            "distributed population unless the decision says otherwise."])
    return report, f"{_stamp(book)}\nNormality Test (Shapiro-Wilk) → [{book.name}]{report.name}"


def one_sample_t(book: Workbook, ws: Worksheet, col: int, mu: float = 0.0) \
        -> tuple[ReportSheet, str]:
    v = ws.columns[col].numeric(ws.data_rows())
    v = v[np.isfinite(v)]
    if len(v) < 2:
        raise AnalysisError("A t-test needs at least 2 values.")
    t, p = stats.ttest_1samp(v, mu)
    se = v.std(ddof=1) / math.sqrt(len(v))
    q = stats.t.ppf(0.975, len(v) - 1)
    report = ReportSheet(title=f"One-Sample t-Test of {ws.columns[col].title}")
    add_report(book, report, "TTest1")
    report.add_table("Descriptive Statistics", ["", "N", "Mean", "SD", "SEM"],
                     [[ws.columns[col].title, str(len(v)), num(v.mean()), num(v.std(ddof=1)),
                       num(se)]])
    report.add_table("t-Test Statistics", ["", "t Statistic", "DF", "Prob>|t|",
                                           "Lower 95% CI", "Upper 95% CI"],
                     [[ws.columns[col].title, num(t, 5), str(len(v) - 1), pval(p),
                       num(v.mean() - q * se), num(v.mean() + q * se)]],
                     [f"Null hypothesis: mean = {num(mu)}; alternative: mean <> {num(mu)}.",
                      "At the 0.05 level, the population mean is " +
                      ("significantly" if p < 0.05 else "not significantly") +
                      f" different from {num(mu)}."])
    return report, f"{_stamp(book)}\nOne-Sample t-Test of {ws.columns[col].title}: " \
                   f"t = {num(t, 5)}, p = {pval(p)}"


def two_sample_t(book: Workbook, ws: Worksheet, col1: int, col2: int, paired: bool = False,
                 equal_var: bool = True) -> tuple[ReportSheet, str]:
    n = ws.data_rows()
    a, b = ws.columns[col1].numeric(n), ws.columns[col2].numeric(n)
    if paired:
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        t, p = stats.ttest_rel(a, b)
        df = len(a) - 1
    else:
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        t, p = stats.ttest_ind(a, b, equal_var=equal_var)
        df = len(a) + len(b) - 2 if equal_var else float("nan")
    if min(len(a), len(b)) < 2:
        raise AnalysisError("Each column needs at least 2 values.")
    name = "Pair-Sample t-Test" if paired else "Two-Sample t-Test"
    report = ReportSheet(title=f"{name} of {ws.columns[col1].title} and "
                               f"{ws.columns[col2].title}")
    add_report(book, report, "TTestPair" if paired else "TTest2")
    report.add_table("Descriptive Statistics", ["", "N", "Mean", "SD", "SEM"],
                     [[ws.columns[c].title, str(len(v)), num(v.mean()), num(v.std(ddof=1)),
                       num(v.std(ddof=1) / math.sqrt(len(v)))]
                      for c, v in ((col1, a), (col2, b))])
    report.add_table("t-Test Statistics", ["", "t Statistic", "DF", "Prob>|t|"],
                     [["Equal Variance Assumed" if equal_var or paired else
                       "Equal Variance Not Assumed", num(t, 5), num(df), pval(p)]],
                     ["Null hypothesis: mean1 − mean2 = 0.",
                      "At the 0.05 level, the difference of the population means is " +
                      ("significantly" if p < 0.05 else "not significantly") +
                      " different from 0."])
    return report, f"{_stamp(book)}\n{name}: t = {num(t, 5)}, p = {pval(p)}"


def one_way_anova(book: Workbook, ws: Worksheet, cols: list[int]) -> tuple[ReportSheet, str]:
    n = ws.data_rows()
    groups = [(ws.columns[c].title, ws.columns[c].numeric(n)) for c in cols
              if not ws.columns[c].is_text()]
    groups = [(t, v[np.isfinite(v)]) for t, v in groups]
    if len(groups) < 2 or any(len(v) < 2 for _, v in groups):
        raise AnalysisError("Select two or more numeric columns (one group per column) with "
                            "at least 2 values each.")
    f, p = stats.f_oneway(*[v for _, v in groups])
    allv = np.concatenate([v for _, v in groups])
    grand = allv.mean()
    ss_b = sum(len(v) * (v.mean() - grand) ** 2 for _, v in groups)
    ss_w = sum(((v - v.mean()) ** 2).sum() for _, v in groups)
    df_b, df_w = len(groups) - 1, len(allv) - len(groups)
    report = ReportSheet(title=f"One-Way ANOVA of [{book.name}]{ws.name}")
    add_report(book, report, "ANOVA1Way")
    report.add_table("Descriptive Statistics", ["", "N", "Mean", "SD", "SE of Mean"],
                     [[t, str(len(v)), num(v.mean()), num(v.std(ddof=1)),
                       num(v.std(ddof=1) / math.sqrt(len(v)))] for t, v in groups])
    report.add_table("Overall ANOVA", ["", "DF", "Sum of Squares", "Mean Square", "F Value",
                                       "Prob>F"],
                     [["Model", str(df_b), num(ss_b), num(ss_b / df_b), num(f, 5), pval(p)],
                      ["Error", str(df_w), num(ss_w), num(ss_w / df_w), "", ""],
                      ["Total", str(df_b + df_w), num(ss_b + ss_w), "", "", ""]],
                     ["At the 0.05 level, the population means are " +
                      ("significantly" if p < 0.05 else "not significantly") + " different."])
    return report, f"{_stamp(book)}\nOne-Way ANOVA: F({df_b}, {df_w}) = {num(f, 5)}, " \
                   f"p = {pval(p)}"

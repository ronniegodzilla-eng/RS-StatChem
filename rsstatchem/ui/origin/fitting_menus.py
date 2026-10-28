"""WP-1 menu hook: Analysis > Fitting / Peaks and Baseline / Spectroscopy, Tools > Fitting
Function Organizer and Matrix > Surface Fit for the Origin-style workspace.

The integrator calls :func:`install` once from ``OriginMainWindow._build_menus``::

    from rsstatchem.ui.origin import fitting_menus
    fitting_menus.install(self)

Everything here only *calls* public (and a few underscore helper) methods of the main
window (``_need_inputs``, ``resolve``, ``_run_in_book``, ``_input_dialog``, ``result``,
``_warn``, ``book_window``, ``new_book``, ``new_matrix``, ``_add_fit_to_graph``,
``recalculate_report``); ``window.py`` is not edited. Each command is also a module
function taking the main window (and optional settings), so tests can run them without a
modal dialog loop. ``install`` also wraps ``main.change_report_parameters`` so that
"Change Parameters" on a Peak Analyzer report reopens the Peak Analyzer wizard.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu

from rsstatchem.core.analysis import fit_multi as fm
from rsstatchem.core.analysis import fitting as fit
from rsstatchem.core.analysis import functions as fl
from rsstatchem.core.analysis import odr
from rsstatchem.core.analysis import peak_analyzer as pa
from rsstatchem.core.analysis import spectroscopy as sp
from rsstatchem.core.analysis import surface_fit as sf
from rsstatchem.core.analysis.calibration import CAL_WEIGHTS, SIGMAS
from rsstatchem.core.analysis.common import num, parse_numbers
from rsstatchem.ui.origin import analysis as an
from rsstatchem.ui.origin import fit_reports as fr
from rsstatchem.ui.origin import recalc
from rsstatchem.ui.origin.model import MatrixBook, ReportSheet, Workbook, Worksheet
from rsstatchem.ui.origin.plotting import add_xy_plot

FIT_COLOR = "#FF0000"
ERRORS = (an.AnalysisError, ValueError, KeyError, RuntimeError, np.linalg.LinAlgError)


# ====================================================================== install
def _submenu(menu: QMenu, title: str) -> QMenu:
    for a in menu.actions():
        m = a.menu()
        if m is not None and m.title().replace("&", "") == title:
            return m
    return menu.addMenu("&" + title)


def _add(main, menu: QMenu, text: str, slot, key: str) -> QAction:
    act = QAction(text, main)
    act.triggered.connect(lambda _=False: slot(main))
    act.setObjectName(f"wp1_{key}")
    menu.addAction(act)
    main.wp1_actions[key] = act
    return act


def install(main) -> None:
    """Add the WP-1 entries to the Origin main window's menus."""
    if getattr(main, "wp1_actions", None):   # idempotent: once per window
        return
    from rsstatchem.ui.origin import fit_organizer
    fit_organizer.register_all()
    main.wp1_actions = {}
    am = main.menus["analysis"]
    fm_ = _submenu(am, "Fitting")
    fm_.addSeparator()
    _add(main, fm_, "Nonlinear Curve &Fit (Multiple Data, Global, Constraints)...", nlfit,
         "nlfit")
    _add(main, fm_, "Multiple &Linear Regression...", mlr_dialog, "mlr")
    _add(main, fm_, "&Sigmoidal Fit (Boltzmann)...", sigmoidal_fit, "sigmoidal")
    _add(main, fm_, "Fit Linear with &X Error (York)...", lambda m: odr_dialog(m, "york"), "york")
    _add(main, fm_, "&Orthogonal Distance Regression...", lambda m: odr_dialog(m, "odr"), "odr")
    _add(main, fm_, "&Implicit Fit (circle, ellipse, f(x,y)=0)...", implicit_dialog,
         "implicit")
    _add(main, fm_, "S&urface Fit (XYZ or Matrix)...", surface_dialog, "surface")
    fm_.addSeparator()
    _add(main, fm_, "Compare &Models...", compare_models_dialog, "compare_models")
    _add(main, fm_, "Compare &Datasets...", compare_datasets_dialog, "compare_datasets")
    _add(main, fm_, "&Rank Models...", rank_models_dialog, "rank_models")
    fm_.addSeparator()
    _add(main, fm_, "Simulate &Curve...", simulate_curve_dialog, "simulate_curve")
    _add(main, fm_, "Simulate Sur&face...", simulate_surface_dialog, "simulate_surface")
    _add(main, fm_, "Ca&libration Curve (LOD/LOQ)...", calibration_dialog, "calibration")
    pk = _submenu(am, "Peaks and Baseline")
    pk.addSeparator()
    _add(main, pk, "&Peak Analyzer...", peak_analyzer, "peak_analyzer")
    _add(main, pk, "&Quick Peaks (find and integrate)", quick_peaks, "quick_peaks")
    _add(main, pk, "&Batch Peak Analysis...", batch_peaks_dialog, "batch_peaks")
    spm = _submenu(am, "Spectroscopy")
    _add(main, spm, "Peak &Deconvolution...", lambda m: peak_analyzer(m, goal="fit"),
         "deconvolution")
    _add(main, spm, "&Unit Conversion (nm, cm⁻¹, eV)...", unit_conversion_dialog, "units")
    _add(main, spm, "&Tauc Plot (Band Gap)...", tauc_dialog, "tauc")
    _add(main, spm, "&Kubelka-Munk...", kubelka_munk, "kubelka_munk")
    _add(main, spm, "&Absorbance ↔ Transmittance...", absorbance_dialog, "absorbance")
    tools = main.menus["tools"]
    tools.addSeparator()
    _add(main, tools, "Fitting &Function Organizer...", organizer, "organizer")
    if "matrix" in main.menus:
        main.menus["matrix"].addSeparator()
        _add(main, main.menus["matrix"], "Surface &Fit...", surface_dialog, "matrix_surface")
    _wrap_change_parameters(main)


def _wrap_change_parameters(main) -> None:
    orig = main.change_report_parameters
    if getattr(orig, "_wp1", False):
        return

    def change(book, report):
        op = getattr(report, "op", {}) or {}
        if op.get("kind") == "fit" and op.get("fit_kind") == "peaks":
            return peak_analyzer(main, settings=op.get("settings"), report=(book, report))
        return orig(book, report)
    change._wp1 = True
    main.change_report_parameters = change


# ====================================================================== helpers
def _show(main, book: Workbook, sheet) -> None:
    bw = main.book_window(book)
    if bw is not None:
        bw.refresh()
        bw.show_sheet(sheet)


def _finish_report(main, book, report, text, stamp: bool = False) -> None:
    if stamp:
        recalc.stamp(main.project, report)
    main.result(text)
    _show(main, book, report)


def _plot_curve(main, gwin, book: Workbook, curve: Worksheet, xcol: int, ycol: int,
                color: str = FIT_COLOR) -> None:
    li = gwin.active_index()
    c = curve.columns
    gwin.mutate(lambda: add_xy_plot(
        gwin.graph, li, c[xcol].numeric(curve.n_rows), c[ycol].numeric(curve.n_rows),
        c[ycol].long_name or c[ycol].name, "line", color,
        link={"book": book.name, "sheet": curve.name, "x": c[xcol].name, "y": c[ycol].name},
        source={"book": book.name, "sheet": curve.name, "x": c[xcol].name,
                "y": c[ycol].name}))


def _inputs(main):
    got = main._need_inputs()
    return got


# ====================================================================== NLFit
def nlfit(main, values: dict | None = None):
    """Analysis > Fitting > Nonlinear Curve Fit (multiple data / global / constraints)."""
    got = _inputs(main)
    if got is None:
        return None
    choices, current, gwin = got
    if values is None:
        from rsstatchem.ui.origin.nlfit_dialog import NLFitDialog
        dlg = NLFitDialog(choices, main, current, graph_available=gwin is not None)
        if not dlg.exec():
            return None
        try:
            values = dlg.values()
        except ValueError as exc:
            main._warn(str(exc))
            return None
    return run_nlfit(main, values, gwin)


def run_nlfit(main, v: dict, gwin=None):
    inputs = v.get("inputs") or [v["input"]]
    mode = v.get("multi_mode", "global")
    try:
        datas = [main.resolve(c) for c in inputs]
        if len(datas) == 1 and mode != "replica":
            data = datas[0]
            out = main._run_in_book(data.book, "Fit", lambda: an.run_fit(
                data, v["model"], "nlfit", weighting=v.get("weighting", "none"),
                params0=v.get("params0"), fixed=v.get("fixed"), bounds=v.get("bounds"),
                level=v.get("level", 0.95), conf_band=v.get("conf_band", True),
                pred_band=v.get("pred_band", False),
                residual_plot=v.get("residual_plot", True),
                output_fitted=v.get("output_fitted", False), mode=v.get("mode", "manual"),
                constraints=v.get("constraints"), algorithm=v.get("algorithm", "auto"),
                residual_extra=v.get("residual_extra", False)))
            recalc.stamp(main.project, out.report)
            main.result(out.log)
            if gwin is not None and v.get("to_graph", True):
                main._add_fit_to_graph(gwin, out, inputs[0])
            else:
                _show(main, data.book, out.report)
            return out
        out = main._run_in_book(datas[0].book, "Fit", lambda: fr.run_multi_fit(
            datas, v["model"], mode, v.get("shared") or (), v.get("params0"), v.get("fixed"),
            v.get("bounds"), v.get("constraints"), v.get("replicas", 2),
            v.get("weighting", "none"), v.get("algorithm", "auto"), v.get("level", 0.95)))
    except ERRORS as exc:
        main._warn(f"Fit failed: {exc}", "Fitting")
        return None
    main.result(out.log)
    book = datas[0].book
    if gwin is not None and v.get("to_graph", True):
        cols = out.curve.columns
        for j in range(1, len(cols)):
            if cols[j].designation == "Y":
                x = max(i for i in range(j) if cols[i].designation == "X")
                _plot_curve(main, gwin, book, out.curve, x, j,
                            fr.PALETTE[(j // 2) % len(fr.PALETTE)] if mode != "replica"
                            else FIT_COLOR)
                if mode == "replica":
                    break
    else:
        _show(main, book, out.report)
    return out


def sigmoidal_fit(main, values: dict | None = None):
    got = main._input_dialog("Sigmoidal Fit", [
        ("model", "Function:", "choice", "Boltzmann",
         {k: k for k in ("Boltzmann", "Logistic", "DoseResp", "Hill", "SLogistic1",
                         "Gompertz", "Richards")})]) if values is None else (values, None)
    if not got:
        return None
    v, gwin = got
    return run_nlfit(main, {"inputs": [v["input"]], "model": v["model"], "multi_mode":
                            "independent"}, gwin)


def mlr_dialog(main, values: dict | None = None):
    """Multiple Linear Regression of a Y column on several X columns of the active sheet."""
    bw = main.active_book_window()
    ws = bw.worksheet() if bw is not None else None
    if ws is None:
        main._warn("Activate a worksheet first.")
        return None
    cols = {c.name: ws.label(i) for i, c in enumerate(ws.columns)}
    if values is None:
        from rsstatchem.ui.origin.dialogs import ParamDialog
        sel = [ws.columns[i].name for i in bw.selected_columns()]
        dlg = ParamDialog("Multiple Linear Regression", [
            ("y", "Dependent (Y):", "choice", sel[-1] if sel else ws.columns[-1].name, cols),
            ("x", "Independents (e.g. A B C):", "text", " ".join(sel[:-1]) or
             ws.columns[0].name), ("intercept", "Fit Intercept", "bool", True)], main)
        if not dlg.exec():
            return None
        values = dlg.values()
    try:
        xs = [ws.index_of(n) for n in str(values["x"]).replace(",", " ").split()]
        yi = ws.index_of(values["y"])
        n = ws.data_rows()
        M = np.column_stack([ws.columns[i].numeric(n) for i in xs])
        y = ws.columns[yi].numeric(n)
        ok = np.isfinite(M).all(axis=1) & np.isfinite(y)
        names = [ws.columns[i].title for i in xs]
        res = fit.fit_mlr(M[ok], y[ok], bool(values.get("intercept", True)),
                          names=[f"b{k + 1} ({nm})" for k, nm in enumerate(names)])
    except ERRORS as exc:
        main._warn(f"Multiple Linear Regression: {exc}")
        return None
    book = bw.book
    report = ReportSheet(title=f"Multiple Linear Regression of {ws.columns[yi].title}")
    an.add_report(book, report, "MR")
    report.add_table("Notes", ["", "Value"], [["Dependent", ws.columns[yi].title],
                                              ["Independents", ", ".join(names)],
                                              ["Number of Points", str(res.n)]])
    an._table_into(report, fit.parameter_table(res, ws.columns[yi].title))
    an._table_into(report, fit.statistics_table(res, ws.columns[yi].title))
    if res.dof > 0:
        an._table_into(report, fit.anova_table(res, ws.columns[yi].title, False))
    fr.add_matrices(report, res)
    text = f"{an._stamp(book)}\nMultiple Linear Regression: R² = {num(res.r2)} → " \
           f"[{book.name}]{report.name}"
    _finish_report(main, book, report, text)
    return report


# ====================================================================== comparison
def _model_list(main, title, n_models, multi=False, category=False, values=None):
    got = _inputs(main)
    if got is None:
        return None
    choices, current, _ = got
    if values is None:
        from rsstatchem.ui.origin.nlfit_dialog import ModelListDialog
        dlg = ModelListDialog(title, choices, main, current, n_models, multi, category)
        if not dlg.exec():
            return None
        values = dlg.values()
    if not values.get("inputs"):
        main._warn("Select the input data.")
        return None
    return values


def compare_models_dialog(main, values: dict | None = None):
    v = _model_list(main, "Compare Models", 2, values=values)
    if v is None:
        return None
    try:
        data = main.resolve(v["inputs"][0])
        report, text = main._run_in_book(data.book, "Compare Models", lambda:
                                         fr.compare_models_report(data, v["models"],
                                                                  v.get("criterion", "aicc")))
    except ERRORS as exc:
        main._warn(f"Compare Models: {exc}")
        return None
    _finish_report(main, data.book, report, text)
    return report


def rank_models_dialog(main, values: dict | None = None):
    v = _model_list(main, "Rank Models", 0, category=True, values=values)
    if v is None:
        return None
    try:
        data = main.resolve(v["inputs"][0])
        report, text = main._run_in_book(data.book, "Rank Models", lambda:
                                         fr.rank_models_report(data, v.get("models") or None,
                                                               v.get("category"),
                                                               v.get("criterion", "aicc")))
    except ERRORS as exc:
        main._warn(f"Rank Models: {exc}")
        return None
    _finish_report(main, data.book, report, text)
    return report


def compare_datasets_dialog(main, values: dict | None = None):
    v = _model_list(main, "Compare Datasets", 1, multi=True, values=values)
    if v is None:
        return None
    try:
        datas = [main.resolve(c) for c in v["inputs"]]
        report, text = main._run_in_book(datas[0].book, "Compare Datasets", lambda:
                                         fr.compare_datasets_report(datas, v["models"][0],
                                                                    v.get("criterion", "aicc")))
    except ERRORS as exc:
        main._warn(f"Compare Datasets: {exc}")
        return None
    _finish_report(main, datas[0].book, report, text)
    return report


# ====================================================================== errors in x
def odr_dialog(main, method: str = "odr", values: dict | None = None):
    if values is None:
        got = main._input_dialog(
            "Fit Linear with X Error (York)" if method == "york" else
            "Orthogonal Distance Regression",
            ([] if method == "york" else [("model", "Function:", "text", "Line")]) +
            [("sx", "X Std. Dev. (0 = xEr column):", "float", 0.0, (0.0, 1e12, 6)),
             ("sy", "Y Std. Dev. (0 = yEr column):", "float", 0.0, (0.0, 1e12, 6))],
            note="Error columns: an xEr / yEr column to the right of the Y column.")
        if not got:
            return None
        values = got[0]
    try:
        data = main.resolve(values["input"])
        ws = data.sheet
        n = ws.n_rows

        def err(value, col):
            if value:
                return float(value)
            if col is None:
                return None
            e = ws.columns[col].numeric(n)[data.rows]
            return e if np.isfinite(e).all() and (e > 0).all() else None
        sx = err(values.get("sx"), ws.xerr_of(data.ycol))
        sy = err(values.get("sy"), ws.yerr_of(data.ycol))
        if method == "york" and (sx is None or sy is None):
            raise ValueError("York regression needs X and Y errors (xEr and yEr columns, or "
                             "constant standard deviations).")
        report, text = main._run_in_book(data.book, "ODR", lambda: fr.odr_report(
            data, values.get("model", "Line"), sx, sy, method))
    except ERRORS as exc:
        main._warn(str(exc))
        return None
    _finish_report(main, data.book, report, text)
    return report


def implicit_dialog(main, values: dict | None = None):
    if values is None:
        got = main._input_dialog("Implicit Fit", [
            ("model", "Function:", "choice", "Circle", {k: k for k in odr.IMPLICIT}),
            ("equation", "or Equation f(x, y) = 0:", "text", "")],
            note="Fits by orthogonal distance; X and Y are the X and Y columns.")
        if not got:
            return None
        values = got[0]
    try:
        data = main.resolve(values["input"])
        model = values.get("equation") or values.get("model", "Circle")
        report, text = main._run_in_book(data.book, "Implicit Fit", lambda: fr.implicit_report(
            data.book, data.x, data.y, data.xtitle, data.ytitle, model))
    except ERRORS as exc:
        main._warn(str(exc))
        return None
    _finish_report(main, data.book, report, text)
    return report


def surface_dialog(main, values: dict | None = None):
    """Surface fit z = f(x, y) of XYZ columns (active worksheet) or of the active matrix."""
    models = {k: k for k in sf.SURFACES}
    mw = main.active_matrix_window()
    if mw is not None:
        ms = mw.sheet()
        if values is None:
            from rsstatchem.ui.origin.dialogs import ParamDialog
            dlg = ParamDialog("Surface Fit (Matrix)", [
                ("model", "Function:", "choice", "Gauss2D", models),
                ("equation", "or z = f(x, y):", "text", "")], main)
            if not dlg.exec():
                return None
            values = dlg.values()
        x, y, z = sf.matrix_xyz(ms.data, (ms.x1, ms.x2, ms.y1, ms.y2))
        book = Workbook.new(main.project.unique_name("SurfaceFit"), 1, 1)
        main.new_book(book)
        titles = (ms.x_title(), ms.y_title(), ms.z_title())
        source = f"[{mw.book.name}]{ms.name}"
    else:
        bw = main.active_book_window()
        ws = bw.worksheet() if bw is not None else None
        if ws is None:
            main._warn("Activate a worksheet with X, Y, Z columns or a matrix.")
            return None
        cols = {c.name: ws.label(i) for i, c in enumerate(ws.columns)}

        def first(d, k=0):
            ii = [c.name for c in ws.columns if c.designation == d]
            return ii[k] if len(ii) > k else ws.columns[min(k, len(ws.columns) - 1)].name
        if values is None:
            from rsstatchem.ui.origin.dialogs import ParamDialog
            dlg = ParamDialog("Surface Fit (XYZ)", [
                ("x", "X:", "choice", first("X"), cols), ("y", "Y:", "choice", first("Y"), cols),
                ("z", "Z:", "choice", first("Z"), cols),
                ("model", "Function:", "choice", "Plane", models),
                ("equation", "or z = f(x, y):", "text", "")], main)
            if not dlg.exec():
                return None
            values = dlg.values()
        try:
            n = ws.data_rows()
            x, y, z = (ws.columns[ws.index_of(values[k])].numeric(n) for k in ("x", "y", "z"))
        except ERRORS as exc:
            main._warn(str(exc))
            return None
        book = bw.book
        titles = tuple(ws.columns[ws.index_of(values[k])].axis_title for k in ("x", "y", "z"))
        source = f"[{book.name}]{ws.name}!" + ",".join(values[k] for k in ("x", "y", "z"))
    model = values.get("equation") or values.get("model", "Plane")
    try:
        report, res, text = main._run_in_book(book, "Surface Fit", lambda: fr.surface_report(
            book, x, y, z, titles, model, source=source))
    except ERRORS as exc:
        main._warn(f"Surface Fit: {exc}")
        return None
    _finish_report(main, book, report, text)
    return report


# ====================================================================== simulate
def simulate_curve_dialog(main, values: dict | None = None):
    if values is None:
        from rsstatchem.ui.origin.dialogs import ParamDialog
        dlg = ParamDialog("Simulate Curve", [
            ("model", "Function (name or y = ...):", "text", "Gauss"),
            ("params", "Parameters (a = 1, b = 2):", "text", "y0 = 0, xc = 5, w = 1, A = 1"),
            ("x1", "X From:", "float", 0.0, (-1e12, 1e12, 6)),
            ("x2", "X To:", "float", 10.0, (-1e12, 1e12, 6)),
            ("n", "Points:", "int", 200, (2, 1000000))], main)
        if not dlg.exec():
            return None
        values = dlg.values()
    from rsstatchem.ui.origin.fit_organizer import parse_initial
    try:
        f = fit.resolve_model(values["model"])
        p = parse_initial(values["params"])
        missing = [k for k in f.params if k not in p]
        if missing:
            raise ValueError(f"Give values for {', '.join(missing)}.")
        xx, yy = sf.simulate_curve(f, p, (values["x1"], values["x2"]), int(values["n"]))
    except ERRORS as exc:
        main._warn(f"Simulate Curve: {exc}")
        return None
    book = Workbook.new(main.project.unique_name("SimCurve"), 2, len(xx))
    ws = book.sheets[0]
    ws.columns[0].designation, ws.columns[0].long_name = "X", "X"
    ws.columns[1].long_name, ws.columns[1].comments = f"{f.name}", f.formula
    ws.set_column_values(0, list(map(float, xx)))
    ws.set_column_values(1, [float(v) if np.isfinite(v) else None for v in yy])
    main.new_book(book)
    main.result(f"Simulate Curve: {f.name} with " + ", ".join(f"{k} = {v:g}" for k, v in
                                                            p.items()) + f" → [{book.name}]")
    return book


def simulate_surface_dialog(main, values: dict | None = None):
    if values is None:
        from rsstatchem.ui.origin.dialogs import ParamDialog
        dlg = ParamDialog("Simulate Surface", [
            ("model", "Function (name or z = ...):", "text", "Gauss2D"),
            ("params", "Parameters:", "text", "z0 = 0, A = 1, xc = 0, w1 = 1, yc = 0, w2 = 1"),
            ("x1", "X From:", "float", -3.0, (-1e12, 1e12, 6)),
            ("x2", "X To:", "float", 3.0, (-1e12, 1e12, 6)),
            ("y1", "Y From:", "float", -3.0, (-1e12, 1e12, 6)),
            ("y2", "Y To:", "float", 3.0, (-1e12, 1e12, 6)),
            ("n", "Points per side:", "int", 41, (2, 2000))], main)
        if not dlg.exec():
            return None
        values = dlg.values()
    from rsstatchem.ui.origin.fit_organizer import parse_initial
    try:
        f = sf.resolve_surface(values["model"])
        p = parse_initial(values["params"])
        gx, gy, Z = sf.simulate_surface(f, p, (values["x1"], values["x2"]),
                                        (values["y1"], values["y2"]), int(values["n"]),
                                        int(values["n"]))
    except ERRORS as exc:
        main._warn(f"Simulate Surface: {exc}")
        return None
    mb = MatrixBook.new(main.project.unique_name("SimSurface"), Z.shape[0], Z.shape[1])
    ms = mb.sheets[0]
    ms.data = Z
    ms.x1, ms.x2, ms.y1, ms.y2 = float(gx[0]), float(gx[-1]), float(gy[0]), float(gy[-1])
    ms.long_name = f.name
    main.new_matrix(mb)
    main.result(f"Simulate Surface: {f.name} → [{mb.name}]")
    return mb


# ====================================================================== calibration
def calibration_dialog(main, values: dict | None = None):
    if values is None:
        got = main._input_dialog("Calibration Curve", [
            ("weighting", "Weighting:", "choice", "none", CAL_WEIGHTS),
            ("sigma", "σ for LOD/LOQ:", "choice", "residual", SIGMAS),
            ("blank_sd", "SD of the Blank:", "float", 0.0, (0.0, 1e12, 6)),
            ("unknowns", "Unknown Responses (y0):", "text", ""),
            ("replicates", "Replicates per Unknown:", "int", 1, (1, 1000))],
            note="X = concentration of the standards, Y = response.")
        if not got:
            return None
        values = got[0]
    try:
        data = main.resolve(values["input"])
        unknowns = parse_numbers(values.get("unknowns", "")) if values.get("unknowns") else []
        report, text = main._run_in_book(data.book, "Calibration", lambda: fr.calibration_report(
            data, values.get("weighting", "none"), values.get("sigma", "residual"),
            values.get("blank_sd") or None, unknowns, int(values.get("replicates", 1))))
    except ERRORS as exc:
        main._warn(f"Calibration: {exc}")
        return None
    _finish_report(main, data.book, report, text)
    return report


# ====================================================================== peaks
def peak_analyzer(main, goal: str | None = None, settings: dict | None = None,
                  report=None, values: dict | None = None):
    """Peak Analyzer wizard; with ``values`` ({"input", "settings", "mode"}) it runs without
    the dialog. ``report=(book, report)`` re-opens an existing report (Change Parameters)."""
    if report is not None:
        book, rep = report
        src = rep.op.get("input") or {}
        from rsstatchem.ui.origin.dialogs import InputChoice
        from rsstatchem.ui.origin.model import range_text
        choices = [InputChoice(range_text(src["book"], src["sheet"], src["y"]), src["book"],
                               src["sheet"], src["y"], src.get("x"), None,
                               tuple(src["rows"]) if src.get("rows") else None)]
        current, gwin = 0, None
    else:
        got = _inputs(main)
        if got is None:
            return None
        choices, current, gwin = got
    if values is None:
        from rsstatchem.ui.origin.peak_analyzer_dialog import PeakAnalyzerDialog
        try:
            dlg = PeakAnalyzerDialog(choices, main, current, main.resolve, settings, goal)
        except ERRORS as exc:
            main._warn(str(exc))
            return None
        if not dlg.exec():
            return None
        values = dlg.values()
    try:
        data = main.resolve(values["input"])
        anl = fr.analysis_from_settings(data.x, data.y, values["settings"])
        if report is not None:
            book, rep = report
            rep.op["settings"] = fr.analysis_settings(anl)
            main.recalculate_report(book, rep)
            return rep
        out = main._run_in_book(data.book, "Peak Analyzer", lambda: fr.peak_report(
            data, anl, mode=values.get("mode", "manual")))
    except ERRORS as exc:
        main._warn(f"Peak Analyzer: {exc}")
        return None
    recalc.stamp(main.project, out.report)
    main.result(out.log)
    if gwin is not None:
        cols = out.curve.columns
        names = [c.long_name for c in cols]
        if "Baseline" in names:
            _plot_curve(main, gwin, data.book, out.curve, 0, names.index("Baseline"),
                        "#0000FF")
        if "Cumulative Fit Peak" in names and "Baseline" not in names:
            _plot_curve(main, gwin, data.book, out.curve, 0, names.index("Cumulative Fit Peak"))
    else:
        _show(main, data.book, out.report)
    return out


def quick_peaks(main, values: dict | None = None):
    """Find peaks (local maximum, 5 % height) and integrate them, no dialog."""
    got = _inputs(main)
    if got is None:
        return None
    choices, current, _ = got
    v = values or {"input": choices[current], "mode": "manual",
                   "settings": {"goal": "integrate", "baseline_mode": "none",
                                "find_opts": {"method": "local_max", "height_pct": 5.0}}}
    return peak_analyzer(main, values=v)


def batch_peaks_dialog(main, values: dict | None = None):
    """Apply one Peak Analyzer setting to every Y column of the active sheet."""
    got = _inputs(main)
    if got is None:
        return None
    choices, current, _ = got
    if values is None:
        from rsstatchem.ui.origin.peak_analyzer_dialog import PeakAnalyzerDialog
        dlg = PeakAnalyzerDialog(choices, main, current, main.resolve)
        dlg.setWindowTitle("Batch Peak Analysis - settings for all datasets")
        if not dlg.exec():
            return None
        values = dlg.values()
    try:
        datas = [main.resolve(c) for c in choices]
        template = fr.analysis_from_settings(datas[0].x, datas[0].y, values["settings"])
        template.keep_peaks = False
        report, text = main._run_in_book(datas[0].book, "Batch Peak Analysis", lambda:
                                         fr.batch_peaks_report(datas, template))
    except ERRORS as exc:
        main._warn(f"Batch Peak Analysis: {exc}")
        return None
    _finish_report(main, datas[0].book, report, text)
    return report


# ====================================================================== spectroscopy
def _column_result(main, data, idx_text):
    idx, text = idx_text
    main.result(text)
    bw = main.book_window(data.book)
    if bw is not None:
        bw.refresh()
    return idx


def unit_conversion_dialog(main, values: dict | None = None):
    if values is None:
        got = main._input_dialog("Unit Conversion", [
            ("from_unit", "X Unit:", "choice", "nm", sp.UNITS),
            ("to_unit", "Convert to:", "choice", "cm-1", sp.UNITS)])
        if not got:
            return None
        values = got[0]
    try:
        data = main.resolve(values["input"])
        return _column_result(main, data, main._run_in_book(
            data.book, "Unit Conversion",
            lambda: fr.convert_units(data, values["from_unit"], values["to_unit"])))
    except ERRORS as exc:
        main._warn(str(exc))
        return None


def kubelka_munk(main, values: dict | None = None):
    got = main._input_dialog("Kubelka-Munk", [], note="F(R) = (1 − R)²/2R of the "
                             "reflectance column (fraction or %).") if values is None \
        else (values, None)
    if not got:
        return None
    try:
        data = main.resolve(got[0]["input"])
        return _column_result(main, data, main._run_in_book(
            data.book, "Kubelka-Munk", lambda: fr.kubelka_munk_column(data)))
    except ERRORS as exc:
        main._warn(str(exc))
        return None


def absorbance_dialog(main, values: dict | None = None):
    if values is None:
        got = main._input_dialog("Absorbance ↔ Transmittance", [
            ("to", "Convert to:", "choice", "transmittance",
             {"transmittance": "%T from absorbance", "absorbance": "Absorbance from %T"})])
        if not got:
            return None
        values = got[0]
    try:
        data = main.resolve(values["input"])
        return _column_result(main, data, main._run_in_book(
            data.book, "Absorbance", lambda: fr.absorbance_column(data, values["to"])))
    except ERRORS as exc:
        main._warn(str(exc))
        return None


def tauc_dialog(main, values: dict | None = None):
    if values is None:
        got = main._input_dialog("Tauc Plot", [
            ("x_unit", "X Unit:", "choice", "nm", sp.UNITS),
            ("transition", "Transition:", "choice", "direct",
             {k: v[0] for k, v in sp.TAUC.items()}),
            ("e1", "Linear Range From (eV, 0 = auto):", "float", 0.0, (0.0, 1e6, 4)),
            ("e2", "Linear Range To (eV):", "float", 0.0, (0.0, 1e6, 4))],
            note="Y = absorbance (∝ α).")
        if not got:
            return None
        values = got[0]
    try:
        data = main.resolve(values["input"])
        rng = (values["e1"], values["e2"]) if values.get("e1") and values.get("e2") else None
        report, ws, text = main._run_in_book(data.book, "Tauc Plot", lambda: fr.tauc_report(
            data, values.get("x_unit", "nm"), values.get("transition", "direct"), rng))
    except ERRORS as exc:
        main._warn(f"Tauc Plot: {exc}")
        return None
    _finish_report(main, data.book, report, text)
    return report


def organizer(main):
    from rsstatchem.ui.origin.fit_organizer import FunctionOrganizerDialog
    dlg = FunctionOrganizerDialog(main)
    dlg.exec()
    return dlg


# ====================================================================== selftest
def selftest() -> None:
    """Headless check of the WP-1 cores (no widgets): library, NLFit with constraints,
    global fit, ODR / York, surface fit, baselines and the Peak Analyzer."""
    rng = np.random.default_rng(0)
    assert fl.library_size() >= 120
    x = np.linspace(0, 10, 40)
    y = 1 + 3 * np.exp(-x / 2.5) + rng.normal(0, 0.01, x.size)
    r = fit.fit_curve(x, y, "ExpDec1")
    assert abs(r.params["t1"] - 2.5) < 0.05 and np.isfinite(r.aicc)
    rc = fit.fit_curve(x, y, "y = a*exp(-x/t) + b", constraints=["a + b = 4"])
    assert abs(rc.params["a"] + rc.params["b"] - 4) < 1e-8
    mr = fm.global_fit([(x, y), (x, 2 * y)], "ExpDec1", shared=["t1"])
    assert abs(mr.combined.params["t1"] - 2.5) < 0.05
    yr = odr.york_fit(x, 2 * x + 1, np.full(40, 0.1), np.full(40, 0.1))
    assert abs(yr.slope - 2) < 1e-9
    X = rng.uniform(-2, 2, (200, 2))
    z = 1 + 2 * np.exp(-0.5 * (X[:, 0] ** 2 + X[:, 1] ** 2))
    s = sf.surface_fit(X[:, 0], X[:, 1], z, "Gauss2D")
    assert abs(s.params["A"] - 2) < 1e-6
    xs = np.linspace(0, 100, 501)
    sig_ = 0.5 + 0.01 * xs + 5 * np.exp(-0.5 * ((xs - 40) / 3) ** 2)
    anl = pa.PeakAnalysis(xs, sig_, "fit", "anchors", {"n_anchors": 6}).run()
    assert anl.fit is not None and abs(anl.fit.peaks[0]["center"] - 40) < 0.1
    assert abs(sp.convert(500, "nm", "cm-1") - 20000) < 1e-9

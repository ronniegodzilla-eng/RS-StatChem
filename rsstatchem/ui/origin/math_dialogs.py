"""Analysis > Mathematics tools of the Origin-style window (and the shared dialog framework
used by :mod:`signal_dialogs`, :mod:`data_menus`).

Every tool has a ``run_*(main, values)`` function that does the work from a settings dict
(used by the dialogs and by tests) and a ``*_dialog(main)`` that asks for the settings.
Inputs are the data sets of the active worksheet or graph layer (the main window's
``analysis_inputs``); masked points are left out (:meth:`Worksheet.xy_arrays`). Results go
where Origin puts them: new columns of the source sheet (aligned with the input rows),
a new sheet in the source book, or a new matrix book; started from a graph, curves are
also plotted into the active layer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFormLayout, QGroupBox, QLabel, QLineEdit,
                               QSpinBox, QVBoxLayout, QWidget)

from rsstatchem.core.analysis import mathematics as maths
from rsstatchem.core.analysis import matrix as mx
from rsstatchem.core.analysis import transforms as tr
from rsstatchem.core.analysis.common import num
from rsstatchem.ui.origin.model import (MatrixBook, MatrixSheet, Workbook, Worksheet,
                                        rows_text)

RESULT_COLOR = "#FF0000"
RESULT_COLORS = ("#FF0000", "#0000FF", "#008000", "#FF00FF", "#FF8000", "#008080")


# ====================================================================== form dialog
class FormDialog(QDialog):
    """Fields ``(key, label, kind, default, extra)``; kinds: int (extra = (lo, hi)), float
    (extra = (lo, hi, decimals)), choice (extra = {value: text}), bool, text."""

    def __init__(self, title: str, fields: list[tuple], parent=None, note: str = ""):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.widgets: dict[str, QWidget] = {}
        form = QFormLayout()
        for key, label, kind, default, *extra in fields:
            extra = extra[0] if extra else None
            if kind == "int":
                w = QSpinBox()
                lo, hi = extra or (0, 1_000_000)
                w.setRange(lo, hi)
                w.setValue(int(default))
            elif kind == "float":
                w = QDoubleSpinBox()
                lo, hi, dec = extra or (-1e12, 1e12, 6)
                w.setRange(lo, hi)
                w.setDecimals(dec)
                w.setValue(float(default))
            elif kind == "choice":
                w = QComboBox()
                for value, text in extra.items():
                    w.addItem(str(text), value)
                w.setCurrentIndex(max(w.findData(default), 0))
            elif kind == "bool":
                w = QCheckBox()
                w.setChecked(bool(default))
            else:
                w = QLineEdit(str(default))
            w.setObjectName(key)
            self.widgets[key] = w
            form.addRow(label, w)
        layout = QVBoxLayout(self)
        if note:
            lab = QLabel(note)
            lab.setWordWrap(True)
            lab.setStyleSheet("color:#444")
            layout.addWidget(lab)
        box = QGroupBox("Settings")
        box.setLayout(form)
        layout.addWidget(box)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)
        self.resize(430, 0)

    def values(self) -> dict:
        out = {}
        for key, w in self.widgets.items():
            if isinstance(w, (QSpinBox, QDoubleSpinBox)):
                out[key] = w.value()
            elif isinstance(w, QComboBox):
                out[key] = w.currentData()
            elif isinstance(w, QCheckBox):
                out[key] = w.isChecked()
            else:
                out[key] = w.text()
        return out


# ====================================================================== inputs
@dataclass
class Series:
    """A resolved (x, y) data set of a worksheet (masked / empty rows removed)."""
    book: Workbook
    ws: Worksheet
    ycol: int
    xcol: int | None
    idx: np.ndarray
    x: np.ndarray
    y: np.ndarray
    rows: tuple | None = None

    @property
    def label(self) -> str:
        c = self.ws.columns[self.ycol]
        return f"[{self.book.name}]{self.ws.name}!{c.name}" + \
            (f'"{c.long_name}"' if c.long_name else "") + rows_text(*(self.rows or
                                                                      (None, None)))

    @property
    def title(self) -> str:
        return self.ws.columns[self.ycol].title

    @property
    def x_title(self) -> str:
        return self.ws.columns[self.xcol].title if self.xcol is not None else "Row"


class InputError(ValueError):
    pass


def inputs(main):
    """``(choices, current, graph window or None)`` from the main window."""
    return main.analysis_inputs()


def resolve(main, choice, honor_mask: bool = True) -> Series:
    book = main.project.windows.get(choice.book)
    ws = book.sheet(choice.sheet) if isinstance(book, Workbook) else None
    if not isinstance(ws, Worksheet):
        raise InputError(f"Sheet {choice.sheet} not found in {choice.book}.")
    y = ws.index_of(choice.ycol)
    x = ws.index_of(choice.xcol) if choice.xcol else None
    idx, xv, yv, _ = ws.xy_arrays(y, x, choice.rows, honor_mask)
    if len(idx) == 0:
        raise InputError(f"Column {choice.ycol} has no (unmasked) numeric data.")
    return Series(book, ws, y, x, idx, xv, yv, choice.rows)


def series_from(book: Workbook, ws: Worksheet, ycol: int, xcol: int | None = -1) -> Series:
    """A Series straight from a sheet (tests, matrix-free tools)."""
    idx, x, y, _ = ws.xy_arrays(ycol, xcol)
    xi = ws.x_of(ycol) if xcol == -1 else xcol
    return Series(book, ws, ycol, xi, idx, x, y)


def ask(main, title: str, fields: list[tuple], note: str = "", n_inputs: int = 1,
        input_labels=("Input:", "Second input:")):
    """Show a FormDialog with input choices first; returns ``(values, gwin)`` or None.
    Inputs are returned as choice objects under ``input`` / ``input2``."""
    choices, current, gwin = inputs(main)
    if not choices:
        main._warn("Select a Y column of a worksheet (or activate a graph with data plots) "
                   "first.")
        return None
    options = {i: c.label for i, c in enumerate(choices)}
    head = [("input", input_labels[0], "choice", current, options)]
    if n_inputs > 1:
        head.append(("input2", input_labels[1], "choice", min(current + 1, len(choices) - 1),
                     options))
    dlg = FormDialog(title, head + fields, main, note)
    if not dlg.exec():
        return None
    v = dlg.values()
    v["input"] = choices[v["input"]]
    if n_inputs > 1:
        v["input2"] = choices[v["input2"]]
    return v, gwin


def ask_plain(main, title: str, fields: list[tuple], note: str = ""):
    dlg = FormDialog(title, fields, main, note)
    return dlg.values() if dlg.exec() else None


# ====================================================================== outputs
def _change(main, book, text: str, fn):
    view = main.book_window(book) if isinstance(book, Workbook) else \
        main.matrix_window(book)
    if view is None:
        return fn()
    return view.change(text, fn)


def add_columns(main, src: Series, columns: list[dict], text: str) -> list[int]:
    """New columns right of the sheet's last column, values placed at the input rows.
    ``columns``: dicts with ``long_name``, ``values`` (one per input point) and optional
    ``designation``, ``units``, ``comments``, ``rows`` (other target rows)."""
    ws = src.ws

    def fn():
        out = []
        for spec in columns:
            rows = np.asarray(spec.get("rows", src.idx), int)
            ws.ensure_rows(int(rows.max()) + 1 if len(rows) else 1)
            c = ws.add_column(spec.get("designation", "Y"))
            c.long_name = spec.get("long_name", "")
            c.units = spec.get("units", "")
            c.comments = spec.get("comments", f"{text} of {src.label}")
            vals = [None] * ws.n_rows
            for r, v in zip(rows, spec["values"]):
                fv = float(v)
                vals[int(r)] = fv if np.isfinite(fv) else None
            c.values = vals
            out.append(len(ws.columns) - 1)
        return out
    return _change(main, src.book, text, fn)


def add_sheet(main, book: Workbook, sheet: Worksheet, text: str, show: bool = True) \
        -> Worksheet:
    sheet.name = book.unique_sheet_name(sheet.name.rstrip("0123456789") or "Sheet") \
        if book.sheet(sheet.name) else sheet.name

    def fn():
        book.sheets.append(sheet)
        if show:
            book.active = len(book.sheets) - 1
    _change(main, book, text, fn)
    return sheet


def result_sheet(name: str, columns: list[tuple]) -> Worksheet:
    """A worksheet from ``(long_name, values, designation[, units[, comments]])`` tuples."""
    n = max((len(c[1]) for c in columns), default=1)
    ws = Worksheet(name=name[:31], n_rows=max(n, 1))
    for spec in columns:
        long_name, values, desig = spec[0], spec[1], spec[2]
        c = ws.add_column(desig)
        c.long_name = long_name
        c.units = spec[3] if len(spec) > 3 else ""
        c.comments = spec[4] if len(spec) > 4 else ""
        ws.set_column_values(len(ws.columns) - 1,
                             [float(v) if np.isfinite(v) else None for v in
                              np.asarray(values, float)])
    return ws


def new_matrix_book(main, data, mapping, name: str, long_name: str = "",
                    xyz_titles=("X", "Y", "Z")) -> MatrixBook:
    data = np.asarray(data, float)
    ms = MatrixSheet(name="MSheet1", data=data, x1=float(mapping[0]), x2=float(mapping[1]),
                     y1=float(mapping[2]), y2=float(mapping[3]), x_name=xyz_titles[0],
                     y_name=xyz_titles[1], long_name=xyz_titles[2])
    mb = MatrixBook(name=main.project.unique_name("MBook"), long_name=long_name or name,
                    sheets=[ms])
    main.new_matrix(mb)
    return mb


def plot_into(main, gwin, book: Workbook, ws: Worksheet, xcol: int | None, ycol: int,
              color: str = RESULT_COLOR, kind: str = "line") -> None:
    """Add sheet column ``ycol`` (against ``xcol``) to the active layer of ``gwin``."""
    if gwin is None:
        return
    from rsstatchem.ui.origin.plotting import add_xy_plot
    li = gwin.active_index()
    if color == RESULT_COLOR:      # successive results in different colours
        k = max(len(gwin.graph.layers[li].plots) - 1, 0)
        color = RESULT_COLORS[k % len(RESULT_COLORS)]
    n = ws.n_rows
    x = ws.columns[xcol].numeric(n) if xcol is not None else np.arange(1, n + 1, dtype=float)
    y = ws.columns[ycol].numeric(n)
    link = {"book": book.name, "sheet": ws.name,
            "x": ws.columns[xcol].name if xcol is not None else None,
            "y": ws.columns[ycol].name}

    def fn():
        add_xy_plot(gwin.graph, li, x, y, ws.columns[ycol].title, kind, color, link=link,
                    source=dict(link))
    gwin.mutate(fn)


def report(main, text: str) -> None:
    main.result(text)


def _guard(main, fn, title: str = "Analysis"):
    try:
        return fn()
    except (ValueError, KeyError, RuntimeError, np.linalg.LinAlgError) as exc:
        main._warn(f"{title}: {exc}", title)
        return None


def _floats(text) -> list[float]:
    from rsstatchem.core.analysis.common import parse_numbers
    return parse_numbers(str(text)) if text not in (None, "") else []


def sheet_columns(main) -> tuple:
    """(book window, worksheet, {index: label}) of the active sheet."""
    bw = main.active_book_window()
    ws = bw.worksheet() if bw is not None else None
    if ws is None:
        raise InputError("Activate a worksheet first.")
    return bw, ws, {i: ws.label(i) + (f"  {c.long_name}" if c.long_name else "")
                    for i, c in enumerate(ws.columns)}


# ====================================================================== interpolate
def run_interpolate(main, v: dict, gwin=None) -> Worksheet:
    s = resolve(main, v["input"])
    lo = v.get("x_from") if v.get("x_from") not in (None, "") else float(np.min(s.x))
    hi = v.get("x_to") if v.get("x_to") not in (None, "") else float(np.max(s.x))
    if v.get("step"):
        xn = np.arange(float(lo), float(hi) + float(v["step"]) * 1e-9, float(v["step"]))
    else:
        xn = np.linspace(float(lo), float(hi), int(v.get("points", 100)))
    yn = maths.interpolate(s.x, s.y, xn, v.get("method", "linear"),
                           bool(v.get("extrapolate", False)))
    sheet = result_sheet("Interpolated", [
        (s.x_title, xn, "X", "", f"Interpolated X of {s.label}"),
        (s.title, yn, "Y", "", f"{maths.INTERP_METHODS[v.get('method', 'linear')]} "
                               f"interpolation of {s.label}")])
    add_sheet(main, s.book, sheet, "Interpolate")
    report(main, f"Interpolate/Extrapolate {s.label}: {len(xn)} points, "
                 f"{maths.INTERP_METHODS[v.get('method', 'linear')]}")
    plot_into(main, gwin, s.book, sheet, 0, 1)
    return sheet


def interpolate_dialog(main) -> None:
    got = ask(main, "Interpolate/Extrapolate", [
        ("method", "Method:", "choice", "linear", maths.INTERP_METHODS),
        ("points", "Number of Points:", "int", 100, (2, 10_000_000)),
        ("step", "X Increment (0 = use number of points):", "float", 0.0, (0, 1e12, 6)),
        ("x_from", "X Minimum (empty = data):", "text", ""),
        ("x_to", "X Maximum (empty = data):", "text", ""),
        ("extrapolate", "Extrapolate outside the data:", "bool", False)])
    if got:
        v, gwin = got
        for k in ("x_from", "x_to"):
            v[k] = _floats(v[k])[0] if _floats(v[k]) else None
        _guard(main, lambda: run_interpolate(main, v, gwin), "Interpolate")


def run_interp_y_from_x(main, v: dict) -> int:
    """Interpolate Y at the X values of column ``xs_col`` of the source sheet; the result
    is a new column next to it (rows aligned with those X values)."""
    s = resolve(main, v["input"])
    ws = s.ws
    xc = int(v["xs_col"])
    n = ws.n_rows
    xq = ws.columns[xc].numeric(n)
    rows = np.flatnonzero(np.isfinite(xq))
    yq = maths.interpolate(s.x, s.y, xq[rows], v.get("method", "linear"),
                           bool(v.get("extrapolate", True)))
    idx = add_columns(main, s, [{"long_name": f"{s.title} at {ws.columns[xc].title}",
                                 "values": yq, "rows": rows}], "Interpolate Y from X")
    report(main, f"Interpolate Y from X: {len(rows)} values of {s.label}")
    return idx[0]


def interp_y_from_x_dialog(main) -> None:
    try:
        _, _, cols = sheet_columns(main)
    except InputError as exc:
        main._warn(str(exc))
        return
    got = ask(main, "Interpolate/Extrapolate Y from X", [
        ("xs_col", "X values to interpolate at (column):", "choice", 0, cols),
        ("method", "Method:", "choice", "linear", maths.INTERP_METHODS),
        ("extrapolate", "Extrapolate:", "bool", True)])
    if got:
        _guard(main, lambda: run_interp_y_from_x(main, got[0]), "Interpolate")


def _xyz_of(ws: Worksheet, zcol: int) -> tuple[int, int, int]:
    y = ws.y_of(zcol)
    x = ws.x_of(y if y is not None else zcol)
    if x is None or y is None:
        raise InputError("XYZ data need X, Y and Z columns (designations X, Y, Z).")
    return x, y, zcol


def run_xyz_interpolate(main, v: dict, ws: Worksheet | None = None,
                        book: Workbook | None = None) -> MatrixBook:
    """2D interpolation of scattered XYZ data onto a rows × cols grid (new matrix book)."""
    if ws is None:
        bw, ws, _ = sheet_columns(main)
        book = bw.book
    x, y, z = _xyz_of(ws, int(v["zcol"]))
    n = ws.n_rows
    keep = ~ws.point_mask([x, y, z], n)
    X, Y, Z = (ws.columns[c].numeric(n)[keep] for c in (x, y, z))
    ok = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    X, Y, Z = X[ok], Y[ok], Z[ok]
    cols, rows = int(v.get("cols", 50)), int(v.get("rows", 50))
    gx = np.linspace(X.min(), X.max(), cols)
    gy = np.linspace(Y.min(), Y.max(), rows)
    GX, GY = np.meshgrid(gx, gy)
    G = maths.interpolate_2d(X, Y, Z, GX, GY, v.get("method", "linear"),
                             float(v.get("smoothing", 0.0)))
    mb = new_matrix_book(main, G, (gx[0], gx[-1], gy[0], gy[-1]), "XYZInterp",
                         f"{book.name} {ws.name} XYZ interpolation",
                         (ws.columns[x].title, ws.columns[y].title, ws.columns[z].title))
    report(main, f"XYZ interpolation ({maths.INTERP2D[v.get('method', 'linear')]}): "
                 f"{rows} × {cols} grid → {mb.name}")
    return mb


def xyz_interpolate_dialog(main) -> None:
    try:
        bw, ws, cols = sheet_columns(main)
    except InputError as exc:
        main._warn(str(exc))
        return
    zs = {i: t for i, t in cols.items() if ws.columns[i].designation == "Z"} or cols
    v = ask_plain(main, "XYZ Interpolation (2D)", [
        ("zcol", "Z column:", "choice", next(iter(zs)), zs),
        ("method", "Method:", "choice", "linear", maths.INTERP2D),
        ("rows", "Rows:", "int", 50, (2, 5000)), ("cols", "Columns:", "int", 50, (2, 5000)),
        ("smoothing", "RBF smoothing:", "float", 0.0, (0, 1e9, 4))])
    if v:
        _guard(main, lambda: run_xyz_interpolate(main, v, ws, bw.book), "Interpolate")


def run_interpolate_3d(main, v: dict, ws: Worksheet | None = None,
                       book: Workbook | None = None) -> int:
    """3D interpolation of XYZ-V data at the target points of three columns (new column)."""
    if ws is None:
        bw, ws, _ = sheet_columns(main)
        book = bw.book
    n = ws.n_rows
    src = [int(v[k]) for k in ("x", "y", "z", "v")]
    keep = ~ws.point_mask(src, n)
    X, Y, Z, V = (ws.columns[c].numeric(n)[keep] for c in src)
    tgt = [int(v[k]) for k in ("tx", "ty", "tz")]
    TX, TY, TZ = (ws.columns[c].numeric(n) for c in tgt)
    rows = np.flatnonzero(np.isfinite(TX) & np.isfinite(TY) & np.isfinite(TZ))
    out = maths.interpolate_3d(X, Y, Z, V, TX[rows], TY[rows], TZ[rows],
                               v.get("method", "linear"))
    s = Series(book, ws, src[3], None, rows, TX[rows], out)
    idx = add_columns(main, s, [{"long_name": f"{ws.columns[src[3]].title} (interpolated)",
                                 "values": out, "rows": rows}], "3D Interpolation")
    report(main, f"3D interpolation: {len(rows)} points")
    return idx[0]


def interpolate_3d_dialog(main) -> None:
    try:
        bw, ws, cols = sheet_columns(main)
    except InputError as exc:
        main._warn(str(exc))
        return
    k = list(cols)
    pick = lambda i: k[min(i, len(k) - 1)]  # noqa: E731
    v = ask_plain(main, "3D Interpolation (XYZ → V)", [
        ("x", "X:", "choice", pick(0), cols), ("y", "Y:", "choice", pick(1), cols),
        ("z", "Z:", "choice", pick(2), cols), ("v", "V (values):", "choice", pick(3), cols),
        ("tx", "Target X:", "choice", pick(4), cols), ("ty", "Target Y:", "choice", pick(5),
                                                       cols),
        ("tz", "Target Z:", "choice", pick(6), cols),
        ("method", "Method:", "choice", "linear",
         {"linear": "Linear", "nearest": "Nearest", "thin_plate_spline": "Thin-plate spline",
          "multiquadric": "Multiquadric"})])
    if v:
        _guard(main, lambda: run_interpolate_3d(main, v, ws, bw.book), "Interpolate")


# ====================================================================== curves
def run_average_curves(main, v: dict, choices=None, gwin=None) -> Worksheet:
    choices = choices if choices is not None else inputs(main)[0]
    series = []
    for c in choices:            # curves without (unmasked) numeric data are skipped
        try:
            series.append(resolve(main, c))
        except (InputError, KeyError):
            continue
    if not series:
        raise InputError("No curves to average.")
    res = maths.average_curves([(s.x, s.y) for s in series], v.get("x_mode", "union"),
                               v.get("method", "linear"), v.get("step") or None,
                               v.get("points") or None)
    book = series[0].book
    names = ", ".join(s.label for s in series)
    sheet = result_sheet("Average", [
        (series[0].x_title, res["x"], "X"), ("Mean", res["mean"], "Y", "",
                                             f"Average of {len(series)} curves"),
        ("SD", res["sd"], "yEr"), ("N", res["n"].astype(float), "Y")])
    add_sheet(main, book, sheet, "Average Multiple Curves")
    report(main, f"Average Multiple Curves: {names} → [{book.name}]{sheet.name}")
    plot_into(main, gwin, book, sheet, 0, 1)
    return sheet


def average_curves_dialog(main) -> None:
    choices, _, gwin = inputs(main)
    if len(choices) < 2:
        main._warn("Average Multiple Curves needs at least two curves: activate a graph "
                   "layer with several plots or a sheet with several Y columns.")
        return
    v = ask_plain(main, "Average Multiple Curves", [
        ("x_mode", "Common X:", "choice", "union", maths.AVERAGE_X),
        ("method", "Interpolation:", "choice", "linear", maths.INTERP_METHODS),
        ("step", "X step (for evenly spaced X):", "float", 0.0, (0, 1e12, 6)),
        ("points", "Points (for evenly spaced X, 0 = auto):", "int", 0, (0, 10_000_000))],
        f"Curves: {len(choices)} — " + "; ".join(c.label for c in choices[:6]))
    if v:
        _guard(main, lambda: run_average_curves(main, v, choices, gwin), "Average")


def run_curve_math(main, v: dict, gwin=None) -> int:
    a = resolve(main, v["input"])
    if v.get("operand", "curve") == "constant":
        y = maths.curve_math(a.x, a.y, v["op"], constant=float(v.get("constant", 0.0)))
        what = f"{a.title} {v['op']} {num(float(v.get('constant', 0.0)))}"
    else:
        b = resolve(main, v["input2"])
        y = maths.curve_math(a.x, a.y, v["op"], b.x, b.y, method=v.get("method", "linear"),
                             extrapolate=bool(v.get("extrapolate", False)))
        what = f"{a.title} {v['op']} {b.title}"
    idx = add_columns(main, a, [{"long_name": what, "values": y}], "Simple Curve Math")
    report(main, f"Simple Curve Math: {what}")
    plot_into(main, gwin, a.book, a.ws, a.xcol, idx[0])
    return idx[0]


def curve_math_dialog(main) -> None:
    got = ask(main, "Simple Curve Math", [
        ("op", "Operator:", "choice", "-", maths.CURVE_OPS),
        ("operand", "Operand:", "choice", "curve", {"curve": "Second input (interpolated)",
                                                    "constant": "Constant"}),
        ("constant", "Constant:", "float", 0.0),
        ("method", "Interpolation:", "choice", "linear", maths.INTERP_METHODS),
        ("extrapolate", "Extrapolate the second curve:", "bool", False)], n_inputs=2,
        input_labels=("Curve 1:", "Curve 2:"))
    if got:
        _guard(main, lambda: run_curve_math(main, got[0], got[1]), "Curve Math")


def run_column_math(main, v: dict, ws: Worksheet | None = None,
                    book: Workbook | None = None) -> int:
    """Simple Column Math: column A op (column B or constant), row by row."""
    if ws is None:
        bw, ws, _ = sheet_columns(main)
        book = bw.book
    n = ws.n_rows
    a = ws.columns[int(v["a"])].numeric(n)
    if v.get("operand") == "constant":
        b = float(v.get("constant", 0.0))
        what = f"{ws.columns[int(v['a'])].title} {v['op']} {num(b)}"
        masked = ws.point_mask([int(v["a"])], n)
    else:
        b = ws.columns[int(v["b"])].numeric(n)
        what = f"{ws.columns[int(v['a'])].title} {v['op']} {ws.columns[int(v['b'])].title}"
        masked = ws.point_mask([int(v["a"]), int(v["b"])], n)
    out = maths.column_math(a, b, v["op"])
    rows = np.flatnonzero(np.isfinite(out) & ~masked)
    s = Series(book, ws, int(v["a"]), None, rows, rows.astype(float), out[rows])
    idx = add_columns(main, s, [{"long_name": what, "values": out[rows], "rows": rows,
                                 "comments": "Simple Column Math"}], "Simple Column Math")
    return idx[0]


def column_math_dialog(main) -> None:
    try:
        bw, ws, cols = sheet_columns(main)
    except InputError as exc:
        main._warn(str(exc))
        return
    k = list(cols)
    v = ask_plain(main, "Simple Column Math", [
        ("a", "Column A:", "choice", k[0], cols), ("op", "Operator:", "choice", "+",
                                                   maths.CURVE_OPS),
        ("operand", "Operand:", "choice", "column", {"column": "Column B",
                                                     "constant": "Constant"}),
        ("b", "Column B:", "choice", k[min(1, len(k) - 1)], cols),
        ("constant", "Constant:", "float", 0.0)])
    if v:
        _guard(main, lambda: run_column_math(main, v, ws, bw.book), "Column Math")


def run_subtract_reference(main, v: dict, gwin=None) -> int:
    a, r = resolve(main, v["input"]), resolve(main, v["input2"])
    y = maths.subtract_reference(a.x, a.y, r.x, r.y, v.get("method", "linear"))
    idx = add_columns(main, a, [{"long_name": f"{a.title} − {r.title}", "values": y}],
                      "Subtract Reference Data")
    plot_into(main, gwin, a.book, a.ws, a.xcol, idx[0])
    return idx[0]


def subtract_reference_dialog(main) -> None:
    got = ask(main, "Subtract Reference Data", [
        ("method", "Interpolation:", "choice", "linear", maths.INTERP_METHODS)], n_inputs=2,
        input_labels=("Input:", "Reference:"))
    if got:
        _guard(main, lambda: run_subtract_reference(main, got[0], got[1]), "Subtract")


def run_subtract_line(main, v: dict, gwin=None) -> int:
    s = resolve(main, v["input"])
    p1 = p2 = None
    if not v.get("end_points", True):
        p1 = (float(v["x1"]), float(v["y1"]))
        p2 = (float(v["x2"]), float(v["y2"]))
    y = maths.subtract_line(s.x, s.y, p1, p2)
    idx = add_columns(main, s, [{"long_name": f"{s.title} − line", "values": y}],
                      "Subtract Straight Line")
    plot_into(main, gwin, s.book, s.ws, s.xcol, idx[0])
    return idx[0]


def subtract_line_dialog(main) -> None:
    got = ask(main, "Subtract Straight Line", [
        ("end_points", "Line through the end points:", "bool", True),
        ("x1", "Point 1 X:", "float", 0.0), ("y1", "Point 1 Y:", "float", 0.0),
        ("x2", "Point 2 X:", "float", 1.0), ("y2", "Point 2 Y:", "float", 0.0)])
    if got:
        _guard(main, lambda: run_subtract_line(main, got[0], got[1]), "Subtract")


TRANSLATE_TO = {"": "Add the offset", "min": "Minimum to the reference",
                "max": "Maximum to the reference", "first": "First point to the reference",
                "last": "Last point to the reference", "mean": "Mean to the reference"}


def run_translate(main, v: dict, gwin=None) -> int:
    s = resolve(main, v["input"])
    vertical = v.get("direction", "vertical") == "vertical"
    vals = s.y if vertical else s.x
    out = maths.translate(vals, float(v.get("offset", 0.0)), v.get("to") or None,
                          float(v.get("reference", 0.0)))
    if vertical:
        idx = add_columns(main, s, [{"long_name": f"{s.title} (translated)",
                                     "values": out}], "Vertical Translate")
        plot_into(main, gwin, s.book, s.ws, s.xcol, idx[0])
    else:
        idx = add_columns(main, s, [{"long_name": f"{s.x_title} (translated)",
                                     "values": out, "designation": "X"},
                                    {"long_name": s.title, "values": s.y}],
                          "Horizontal Translate")
        plot_into(main, gwin, s.book, s.ws, idx[0], idx[1])
    return idx[0]


def translate_dialog(main, direction: str = "vertical") -> None:
    got = ask(main, f"{direction.title()} Translate", [
        ("to", "Translate:", "choice", "", TRANSLATE_TO),
        ("offset", "Offset:", "float", 0.0), ("reference", "Reference value:", "float", 0.0)])
    if got:
        got[0]["direction"] = direction
        _guard(main, lambda: run_translate(main, got[0], got[1]), "Translate")


def run_polygon_area(main, v: dict) -> dict:
    s = resolve(main, v["input"])
    r = maths.polygon_area(s.x, s.y)
    report(main, f"Polygon Area of {s.label}: area = {num(r['area'])} (signed "
                 f"{num(r['signed_area'])}), centroid = ({num(r['cx'])}, {num(r['cy'])}), "
                 f"perimeter = {num(r['perimeter'])}, {r['points']} points")
    return r


def polygon_area_dialog(main) -> None:
    got = ask(main, "Polygon Area", [])
    if got:
        r = _guard(main, lambda: run_polygon_area(main, got[0]), "Polygon Area")
        if r:
            main.message(f"Polygon area = {num(r['area'])}")


def run_set_operation(main, v: dict, ws: Worksheet | None = None,
                      book: Workbook | None = None) -> int:
    if ws is None:
        bw, ws, _ = sheet_columns(main)
        book = bw.book
    n = ws.n_rows
    a, b = int(v["a"]), int(v["b"])

    def unmasked(c):
        col = ws.columns[c]
        return [x for r, x in enumerate((col.values + [None] * n)[:n]) if r not in col.mask]
    res = maths.set_operation(unmasked(a), unmasked(b), v.get("op", "union"))

    def fn():
        c = ws.add_column("Y")
        c.long_name = f"{maths.SET_OPS[v.get('op', 'union')]} of {ws.columns[a].name}, " \
                      f"{ws.columns[b].name}"
        ws.set_column_values(len(ws.columns) - 1, [x if not isinstance(x, (float, np.floating))
                                                   else float(x) for x in res])
        return len(ws.columns) - 1
    return _change(main, book, "Set Operation", fn)


def set_operation_dialog(main) -> None:
    try:
        bw, ws, cols = sheet_columns(main)
    except InputError as exc:
        main._warn(str(exc))
        return
    k = list(cols)
    v = ask_plain(main, "Set Operations", [
        ("a", "Column A:", "choice", k[0], cols),
        ("b", "Column B:", "choice", k[min(1, len(k) - 1)], cols),
        ("op", "Operation:", "choice", "union", maths.SET_OPS)])
    if v:
        _guard(main, lambda: run_set_operation(main, v, ws, bw.book), "Set Operations")


# ====================================================================== matrix
def _matrix(main):
    mw = main.active_matrix_window()
    ms = mw.sheet() if mw is not None else None
    if ms is None:
        raise InputError("Activate a matrix window first.")
    return mw, ms


def _mapping(ms: MatrixSheet):
    return (ms.x1, ms.x2, ms.y1, ms.y2)


def add_matrix_sheet(main, mw, data, mapping, name: str, long_name: str = "",
                     like: MatrixSheet | None = None) -> MatrixSheet:
    book = mw.book
    ms = MatrixSheet(name=book.unique_sheet_name("MSheet"), data=np.asarray(data, float),
                     x1=float(mapping[0]), x2=float(mapping[1]), y1=float(mapping[2]),
                     y2=float(mapping[3]), long_name=long_name or name)
    if like is not None:
        ms.x_name, ms.x_units, ms.y_name, ms.y_units = like.x_name, like.x_units, \
            like.y_name, like.y_units

    def fn():
        book.sheets.append(ms)
        book.active = len(book.sheets) - 1
    mw.change(name, fn)
    return ms


def run_matrix_op(main, op: str, v: dict | None = None) -> MatrixSheet | dict | None:
    v = v or {}
    mw, ms = _matrix(main)
    m = _mapping(ms)
    if op == "math":
        other = v.get("other")
        b = float(v.get("constant", 0.0)) if other in (None, "", "constant") else \
            _find_matrix(main, other).data
        data = mx.arithmetic(ms.data, b, v.get("op", "+"))
        return add_matrix_sheet(main, mw, data, m, "Simple Math", like=ms)
    if op == "inverse":
        return add_matrix_sheet(main, mw, mx.inverse(ms.data), m, "Inverse", like=ms)
    if op == "filter":
        data = mx.smooth_filter(ms.data, v.get("kind", "mean"), int(v.get("size", 3)),
                                float(v.get("sigma", 1.0)))
        return add_matrix_sheet(main, mw, data, m, f"Filter ({v.get('kind', 'mean')})",
                                like=ms)
    if op == "crop":
        data, mp = mx.crop(ms.data, m, v["r1"], v["r2"], v["c1"], v["c2"])
        return add_matrix_sheet(main, mw, data, mp, "Crop", like=ms)
    if op == "rotate":
        data = mx.rotate_any(ms.data, float(v.get("angle", 30.0)), bool(v.get("reshape", True)))
        return add_matrix_sheet(main, mw, data, (1, data.shape[1], 1, data.shape[0]),
                                f"Rotate {v.get('angle', 30.0)}°")
    if op == "resample":
        data = mx.resample(ms.data, m, int(v.get("rows", 64)), int(v.get("cols", 64)),
                           v.get("method", "linear"))
        return add_matrix_sheet(main, mw, data, m, "2D Interpolate", like=ms)
    if op == "integrate":
        r = mx.integrate_volume(ms.data, m, float(v.get("baseline", 0.0)))
        report(main, f"Matrix Integrate [{mw.book.name}]{ms.name}: volume = "
                     f"{num(r['volume'])}, |volume| = {num(r['abs_volume'])}, "
                     f"area = {num(r['area'])}")
        return r
    if op == "stats":
        r = mx.statistics(ms.data)
        report(main, f"Matrix Statistics [{mw.book.name}]{ms.name}: " +
               ", ".join(f"{k} = {num(val) if isinstance(val, float) else val}"
                         for k, val in r.items()))
        return r
    if op == "fft2":
        r = tr.fft2(ms.data, True, v.get("window", "rect"))
        out = add_matrix_sheet(main, mw, r["amplitude"], (-0.5, 0.5, -0.5, 0.5),
                               "2D FFT Amplitude")
        add_matrix_sheet(main, mw, r["phase"], (-0.5, 0.5, -0.5, 0.5), "2D FFT Phase")
        return out
    if op == "fft2_filter":
        data = tr.filter2d(ms.data, v.get("kind", "lowpass"), float(v.get("cutoff", 0.2)),
                           float(v.get("cutoff2", 0.5)))
        return add_matrix_sheet(main, mw, data, m, "2D FFT Filter", like=ms)
    if op == "correlate2d":
        data = tr.correlate2d(ms.data)
        return add_matrix_sheet(main, mw, data, (1, data.shape[1], 1, data.shape[0]),
                                "2D Autocorrelation")
    raise ValueError(f"Unknown matrix operation {op}")


def _find_matrix(main, ref: str) -> MatrixSheet:
    """``[MBook1]MSheet1`` → that matrix sheet."""
    import re
    m = re.match(r"^\[(.+)\](.+)$", ref)
    if not m:
        raise InputError(f"Unknown matrix {ref}")
    book = main.project.windows.get(m.group(1))
    ms = book.sheet(m.group(2)) if isinstance(book, MatrixBook) else None
    if ms is None:
        raise InputError(f"Unknown matrix {ref}")
    return ms


def matrix_dialog(main, op: str) -> None:
    try:
        mw, ms = _matrix(main)
    except InputError as exc:
        main._warn(str(exc))
        return
    fields = {
        "math": [("op", "Operator:", "choice", "+", mx.MATRIX_OPS),
                 ("other", "Operand:", "choice", "constant",
                  {"constant": "Constant", **{f"[{b.name}]{s.name}": f"[{b.name}]{s.name}"
                                              for b in main.project.matrices()
                                              for s in b.sheets}}),
                 ("constant", "Constant:", "float", 0.0)],
        "filter": [("kind", "Filter:", "choice", "mean", mx.MATRIX_FILTERS),
                   ("size", "Size (cells):", "int", 3, (1, 99)),
                   ("sigma", "Gaussian σ (cells):", "float", 1.0, (0.01, 100, 2))],
        "crop": [("r1", "First row:", "int", 1, (1, ms.n_rows)),
                 ("r2", "Last row:", "int", ms.n_rows, (1, ms.n_rows)),
                 ("c1", "First column:", "int", 1, (1, ms.n_cols)),
                 ("c2", "Last column:", "int", ms.n_cols, (1, ms.n_cols))],
        "rotate": [("angle", "Angle (degrees, counter-clockwise):", "float", 30.0,
                    (-360, 360, 2)), ("reshape", "Enlarge to fit:", "bool", True)],
        "resample": [("rows", "Rows:", "int", ms.n_rows * 2, (2, 10000)),
                     ("cols", "Columns:", "int", ms.n_cols * 2, (2, 10000)),
                     ("method", "Method:", "choice", "linear",
                      {"linear": "Bilinear", "cubic": "Bicubic", "nearest": "Nearest"})],
        "integrate": [("baseline", "Baseline Z:", "float", 0.0)],
        "fft2": [("window", "Window:", "choice", "rect", tr.WINDOWS)],
        "fft2_filter": [("kind", "Filter:", "choice", "lowpass",
                         {"lowpass": "Low Pass", "highpass": "High Pass",
                          "bandpass": "Band Pass", "bandblock": "Band Block"}),
                        ("cutoff", "Cutoff (fraction of Nyquist):", "float", 0.2, (0, 2, 3)),
                        ("cutoff2", "Upper cutoff:", "float", 0.5, (0, 2, 3))],
    }.get(op)
    v = {}
    if fields:
        v = ask_plain(main, f"Matrix: {op}", fields)
        if v is None:
            return
    _guard(main, lambda: run_matrix_op(main, op, v), "Matrix")

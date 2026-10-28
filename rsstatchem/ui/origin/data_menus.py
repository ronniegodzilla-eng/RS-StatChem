"""WP-2 menus of the Origin-style window: data import / export, worksheet manipulation,
masking, Mathematics, Signal Processing and Matrix tools.

:func:`install` adds Origin-named submenus and actions to the main window's menus
(``main.menus[key]``) — the main window calls it from ``_build_menus``; it is idempotent. All
actions are kept in ``main.data_actions`` (name → QAction) for tests and toolbars.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QFileDialog, QMenu

from rsstatchem.core.analysis import datamanip as dm
from rsstatchem.core.analysis import exporters as ex
from rsstatchem.core.analysis import wavelets as wv
from rsstatchem.ui.origin import import_wizard as iw
from rsstatchem.ui.origin import math_dialogs as md
from rsstatchem.ui.origin import signal_dialogs as sd
from rsstatchem.ui.origin import worksheet_ops as wo
from rsstatchem.ui.origin.model import ReportSheet, Workbook, Worksheet


# ====================================================================== helpers
def _target(main):
    bw = main.active_book_window()
    ws = bw.worksheet() if bw is not None else None
    if ws is None:
        main._warn("Activate a worksheet first.")
        return None
    return bw, ws


def _cols_dict(ws: Worksheet) -> dict:
    return {i: ws.label(i) + (f"  {c.long_name}" if c.long_name else "")
            for i, c in enumerate(ws.columns)}


def _new_sheet(main, bw, sheet: Worksheet, text: str) -> Worksheet:
    return md.add_sheet(main, bw.book, sheet, text)


def _guard(main, fn, title):
    return md._guard(main, fn, title)


# ====================================================================== masks
def mask_action(main, how: str) -> int:
    bw = main.active_book_window()
    if bw is None or bw.worksheet() is None:
        main._warn("Select worksheet cells to mask first.")
        return 0
    if how == "unmask_all":
        bw.unmask_all()
        return 0
    if how == "unmask_columns":
        bw.unmask_columns()
        return 0
    n = bw.mask_selection(how)
    main.show_status(f"{n} cell(s): {how}")
    return n


def mask_condition_dialog(main) -> None:
    t = _target(main)
    if t is None:
        return
    bw, ws = t
    cols = bw.selected_columns() or [0]
    v = md.ask_plain(main, "Mask by Condition", [
        ("col", "Column to mask:", "choice", cols[0], _cols_dict(ws)),
        ("expr", "Condition (e.g. B > 5 or col(A) < 0):", "text", ""),
        ("mode", "Action:", "choice", "mask", {"mask": "Mask", "unmask": "Unmask"})])
    if v and v["expr"].strip():
        def fn():
            return wo.mask_condition(ws, int(v["col"]), v["expr"], v["mode"])
        n = _guard(main, lambda: bw.change("Mask by Condition", fn, recalc=False), "Mask")
        if n is not None:
            main.message(f"{v['mode'].title()}ed {n} cell(s) of {ws.columns[int(v['col'])].name}")


# ====================================================================== worksheet ops
def stack_dialog(main) -> None:
    t = _target(main)
    if t is None:
        return
    bw, ws = t
    sel = bw.selected_columns() or [i for i, c in enumerate(ws.columns)
                                    if c.designation == "Y"]
    v = md.ask_plain(main, "Stack Columns", [
        ("cols", "Columns to stack (e.g. B:D):", "text", _range_text(ws, sel)),
        ("ids", "Identifier columns (repeated):", "text", ""),
        ("drop", "Skip empty cells:", "bool", True)])
    if v:
        _guard(main, lambda: _new_sheet(main, bw, wo.stack(
            ws, _parse_cols(ws, v["cols"]), _parse_cols(ws, v["ids"]),
            drop_missing=v["drop"]), "Stack Columns"), "Stack Columns")


def _range_text(ws: Worksheet, cols) -> str:
    return ",".join(ws.columns[c].name for c in cols)


def _parse_cols(ws: Worksheet, text: str) -> list[int]:
    """``"B:D, F"`` (short names, ranges by sheet order) → column indices."""
    out: list[int] = []
    for part in [p.strip() for p in str(text).replace(";", ",").split(",") if p.strip()]:
        if ":" in part:
            a, b = (ws.find(x.strip()) for x in part.split(":", 1))
            out.extend(range(min(a, b), max(a, b) + 1))
        else:
            out.append(ws.find(part))
    return list(dict.fromkeys(out))


def unstack_dialog(main) -> None:
    t = _target(main)
    if t is None:
        return
    bw, ws = t
    cols = _cols_dict(ws)
    k = list(cols)
    v = md.ask_plain(main, "Unstack Columns", [
        ("value", "Data column:", "choice", k[min(1, len(k) - 1)], cols),
        ("group", "Group column:", "choice", k[0], cols),
        ("key", "Match rows by (optional):", "choice", -1, {-1: "(none)", **cols})])
    if v:
        key = None if v["key"] == -1 else int(v["key"])
        _guard(main, lambda: _new_sheet(main, bw, wo.unstack(ws, int(v["value"]),
                                                             int(v["group"]), key),
                                        "Unstack Columns"), "Unstack Columns")


def pivot_dialog(main) -> None:
    t = _target(main)
    if t is None:
        return
    bw, ws = t
    cols = _cols_dict(ws)
    k = list(cols)
    v = md.ask_plain(main, "Pivot Table", [
        ("rows", "Row source:", "choice", k[0], cols),
        ("cols", "Column source:", "choice", k[min(1, len(k) - 1)], cols),
        ("values", "Data:", "choice", k[min(2, len(k) - 1)], cols),
        ("agg", "Aggregate:", "choice", "mean", dm.AGGREGATES)])
    if v:
        _guard(main, lambda: _new_sheet(main, bw, wo.pivot_table(
            ws, int(v["rows"]), int(v["cols"]), int(v["values"]), v["agg"]), "Pivot Table"),
            "Pivot Table")


def reduce_rows_dialog(main) -> None:
    t = _target(main)
    if t is None:
        return
    bw, ws = t
    v = md.ask_plain(main, "Reduce Rows", [
        ("n", "Every N rows:", "int", 2, (1, 1_000_000)),
        ("method", "Method:", "choice", "first", {"first": "Keep one row (delete the others)",
                                                   **{k: t for k, t in dm.AGGREGATES.items()
                                                      if k in ("mean", "sum", "min", "max",
                                                               "median")}}),
        ("start", "Start at row:", "int", 1, (1, 1_000_000))])
    if v:
        _guard(main, lambda: _new_sheet(main, bw, wo.reduce_rows(ws, v["n"], v["method"],
                                                                 v["start"] - 1),
                                        "Reduce Rows"), "Reduce Rows")


def _x_and_ys(bw, ws):
    sel = bw.selected_columns()
    ys = [c for c in sel if ws.columns[c].designation != "X"] or \
        [i for i, c in enumerate(ws.columns) if c.designation == "Y"]
    x = ws.x_of(ys[0]) if ys else None
    return x, ys


def reduce_dup_x_dialog(main) -> None:
    t = _target(main)
    if t is None:
        return
    bw, ws = t
    x, ys = _x_and_ys(bw, ws)
    if x is None:
        main._warn("Reduce Duplicate X needs an X column.")
        return
    v = md.ask_plain(main, "Reduce Duplicate X", [
        ("method", "Merge Y by:", "choice", "mean", {k: t for k, t in dm.AGGREGATES.items()
                                                     if k != "count"})])
    if v:
        _guard(main, lambda: _new_sheet(main, bw, wo.reduce_duplicate_x(ws, x, ys,
                                                                        v["method"]),
                                        "Reduce Duplicate X"), "Reduce Duplicate X")


def reduce_xy_dialog(main) -> None:
    t = _target(main)
    if t is None:
        return
    bw, ws = t
    x, ys = _x_and_ys(bw, ws)
    if x is None:
        main._warn("Reduce to evenly spaced X needs an X column.")
        return
    v = md.ask_plain(main, "Reduce to Evenly Spaced X", [
        ("step", "X increment (0 = use number of points):", "float", 0.0, (0, 1e12, 6)),
        ("points", "Number of points:", "int", 100, (1, 10_000_000)),
        ("method", "Y in each interval:", "choice", "mean",
         {k: t for k, t in dm.AGGREGATES.items() if k in ("mean", "median", "min", "max",
                                                          "sum", "first", "last")})])
    if v:
        _guard(main, lambda: _new_sheet(main, bw, wo.reduce_xy(
            ws, x, ys, v["step"] or None, None if v["step"] else v["points"], v["method"]),
            "Reduce XY"), "Reduce XY")


def remove_duplicates_dialog(main) -> None:
    t = _target(main)
    if t is None:
        return
    bw, ws = t
    sel = bw.selected_columns()
    v = md.ask_plain(main, "Remove Duplicate Rows", [
        ("keys", "Compare columns (empty = all):", "text", _range_text(ws, sel))])
    if v is not None:
        keys = _parse_cols(ws, v["keys"]) if v["keys"].strip() else None
        _guard(main, lambda: _new_sheet(main, bw, wo.remove_duplicates(ws, keys),
                                        "Remove Duplicates"), "Remove Duplicates")


def query_dialog(main) -> None:
    t = _target(main)
    if t is None:
        return
    bw, ws = t
    v = md.ask_plain(main, "Worksheet Query", [
        ("expr", "Condition:", "text", f"{ws.columns[0].name} > 0"),
        ("action", "Output:", "choice", "extract", {"extract": "Extract rows to a new sheet",
                                                     "select": "Select the rows",
                                                     "mask": "Mask the other rows"})],
        "Columns by short name, col(A) or col(\"Long Name\"); operators = <> < > <= >= "
        "and or not, functions abs sqrt log ln exp isnan between(x, a, b).")
    if not v:
        return

    def run():
        rows = wo.query_rows(ws, v["expr"])
        if v["action"] == "extract":
            _new_sheet(main, bw, wo.extract_rows(ws, rows, "Query"), "Worksheet Query")
        elif v["action"] == "mask":
            other = sorted(set(range(ws.data_rows())) - set(rows.tolist()))

            def fn():
                for c in range(len(ws.columns)):
                    ws.set_mask(c, other, True)
            bw.change("Mask by Query", fn, recalc=False)
        else:
            view = bw.sheet_view()
            if view is not None and len(rows):
                from PySide6.QtCore import QItemSelection, QItemSelectionModel

                from rsstatchem.ui.origin.sheet_view import HEADER_ROWS
                sel = QItemSelection()
                for r in rows:
                    sel.select(view.model_.index(HEADER_ROWS + int(r), 0),
                               view.model_.index(HEADER_ROWS + int(r), len(ws.columns) - 1))
                view.selectionModel().select(sel, QItemSelectionModel.SelectionFlag
                                             .ClearAndSelect)
        main.message(f"Worksheet Query “{v['expr']}”: {len(rows)} row(s)")
        return rows
    _guard(main, run, "Worksheet Query")


def sort_dialog(main) -> None:
    t = _target(main)
    if t is None:
        return
    bw, ws = t
    cols = _cols_dict(ws)
    none = {-1: "(none)", **cols}
    sel = bw.selected_columns() or [0]
    order = {True: "Ascending", False: "Descending"}
    v = md.ask_plain(main, "Sort Worksheet (Nested)", [
        ("k1", "First key:", "choice", sel[0], cols), ("a1", "Order:", "choice", True, order),
        ("k2", "Second key:", "choice", -1, none), ("a2", "Order:", "choice", True, order),
        ("k3", "Third key:", "choice", -1, none), ("a3", "Order:", "choice", True, order)])
    if v:
        keys = [(int(v[f"k{i}"]), bool(v[f"a{i}"])) for i in (1, 2, 3) if v[f"k{i}"] != -1]
        _guard(main, lambda: bw.change("Sort Worksheet", lambda: wo.sort_rows(ws, keys)),
               "Sort")


def split_dialog(main) -> None:
    t = _target(main)
    if t is None:
        return
    bw, ws = t
    v = md.ask_plain(main, "Split Worksheet", [
        ("by", "Split:", "choice", "column", {"column": "By the values of a column",
                                              "rows": "Every N rows"}),
        ("col", "Column:", "choice", (bw.selected_columns() or [0])[0], _cols_dict(ws)),
        ("n", "Rows per sheet:", "int", 10, (1, 10_000_000))])
    if not v:
        return

    def run():
        parts = wo.split(ws, int(v["col"]) if v["by"] == "column" else None,
                         None if v["by"] == "column" else v["n"])

        def fn():
            for p in parts:
                p.name = bw.book.unique_sheet_name(p.name[:26]) if bw.book.sheet(p.name) \
                    else p.name
                bw.book.sheets.append(p)
        bw.change("Split Worksheet", fn, recalc=False)
        main.message(f"Split {ws.name} into {len(parts)} sheets")
        return parts
    _guard(main, run, "Split")


def trim_dialog(main) -> None:
    t = _target(main)
    if t is None:
        return
    bw, ws = t
    v = md.ask_plain(main, "Remove Rows with Missing Values", [
        ("how", "Remove a row when:", "choice", "any", {"any": "any cell is empty",
                                                        "all": "all cells are empty"})])
    if v:
        _guard(main, lambda: _new_sheet(main, bw, wo.trim_missing(
            ws, bw.selected_columns() or None, v["how"]), "Trim Missing"), "Trim")


def paste_transpose(main, text: str | None = None) -> None:
    bw = main.active_book_window()
    view = bw.sheet_view() if bw is not None else None
    if view is None:
        return
    text = QApplication.clipboard().text() if text is None else text
    if not text:
        return
    block = [ln.split("\t") for ln in text.replace("\r\n", "\n").rstrip("\n").split("\n")]
    from rsstatchem.ui.origin.sheet_view import HEADER_ROWS
    cur = view.currentIndex()
    top = max((cur.row() if cur.isValid() else HEADER_ROWS) - HEADER_ROWS, 0)
    left = max(cur.column(), 0) if cur.isValid() else 0
    bw.change("Paste Transpose", lambda: wo.paste_transpose(view.ws, block, top, left))


# ====================================================================== export
def export_xlsx(main, path: str | None = None, which: str | None = None) -> Path | None:
    """Export worksheets to Excel: the active sheet, the active book (all data sheets) or
    every book of the project (one Excel sheet per worksheet)."""
    bw = main.active_book_window()
    if which is None:
        v = md.ask_plain(main, "Export to Excel", [
            ("which", "Export:", "choice", "book", {"sheet": "Active worksheet",
                                                    "book": "All sheets of the active book",
                                                    "project": "All books of the project"}),
            ("masked", "Masked cells:", "choice", "keep", {"keep": "Export values",
                                                           "blank": "Leave empty"})])
        if not v:
            return None
        which, masked = v["which"], v["masked"]
    else:
        masked = "keep"
    sheets = []
    if which == "project":
        for b in main.project.books():
            sheets += [(f"{b.name}_{s.name}", s) for s in b.sheets if isinstance(s, Worksheet)]
    elif bw is not None:
        if which == "sheet" and bw.worksheet() is not None:
            sheets = [(bw.worksheet().name, bw.worksheet())]
        else:
            sheets = [(s.name, s) for s in bw.book.sheets if isinstance(s, Worksheet)]
    if not sheets:
        main._warn("There is no worksheet to export.")
        return None
    if path is None:
        start = Path(main._start_dir()) / f"{bw.book.name if bw else 'Project'}.xlsx"
        path, _ = QFileDialog.getSaveFileName(main, "Export to Excel", str(start),
                                              "Excel (*.xlsx)")
        if not path:
            return None
    out = _guard(main, lambda: ex.write_xlsx(sheets, path, masked=masked), "Export")
    if out is not None:
        main.message(f"Exported {len(sheets)} sheet(s) to {out}")
    return out


def export_ascii(main, path: str | None = None, options: dict | None = None) -> Path | None:
    t = _target(main)
    if t is None:
        return None
    bw, ws = t
    if options is None:
        options = md.ask_plain(main, "Export ASCII", [
            ("delimiter", "Delimiter:", "choice", ",", {",": "Comma", "\t": "Tab",
                                                       ";": "Semicolon", " ": "Space"}),
            ("decimal", "Decimal separator:", "choice", ".", {".": "Dot", ",": "Comma"}),
            ("short", "Short names row:", "bool", False),
            ("long", "Long Name row:", "bool", True), ("units", "Units row:", "bool", True),
            ("comments", "Comments row:", "bool", False),
            ("masked", "Masked cells:", "choice", "keep", {"keep": "Export values",
                                                           "blank": "Leave empty"})])
        if not options:
            return None
    if path is None:
        path, _ = QFileDialog.getSaveFileName(main, "Export ASCII",
                                              str(Path(main._start_dir()) / f"{ws.name}.csv"),
                                              "CSV (*.csv);;Text (*.txt *.dat)")
        if not path:
            return None
    headers = [h for h, key in (("name", "short"), ("long_name", "long"), ("units", "units"),
                                ("comments", "comments")) if options.get(key)]
    out = _guard(main, lambda: ex.write_csv(ws, path, options.get("delimiter", ","),
                                            options.get("decimal", "."), headers,
                                            options.get("masked", "keep")), "Export")
    if out is not None:
        main.message(f"Exported {ws.name} to {out}")
    return out


def export_report(main, path: str | None = None) -> Path | None:
    bw = main.active_book_window()
    rep = bw.sheet() if bw is not None else None
    if not isinstance(rep, ReportSheet):
        main._warn("Activate a report sheet to export it.")
        return None
    if path is None:
        path, _ = QFileDialog.getSaveFileName(main, "Export Report",
                                              str(Path(main._start_dir()) / f"{rep.name}.html"),
                                              "HTML (*.html);;Word (*.docx)")
        if not path:
            return None
    from rsstatchem.ui.origin.report_view import item_png, report_html
    if str(path).lower().endswith(".docx"):
        out = _guard(main, lambda: ex.write_report_docx(rep, path, item_png), "Export")
    else:
        out = _guard(main, lambda: ex.write_html(report_html(rep), path), "Export")
    if out is not None:
        main.message(f"Exported report {rep.name} to {out}")
    return out


def export_all_graphs(main, folder: str | None = None, fmt: str = "png", dpi: int = 300) \
        -> list[Path]:
    """Batch export of every graph window of the project (``core.graph.export``)."""
    from rsstatchem.core.graph.export import export_graph
    graphs = main.project.graphs()
    if not graphs:
        main._warn("The project has no graphs.")
        return []
    if folder is None:
        folder = QFileDialog.getExistingDirectory(main, "Export All Graphs To",
                                                  main._start_dir())
        if not folder:
            return []
    if hasattr(main, "sync_models"):
        main.sync_models()
    out = []
    for g in graphs:
        p = Path(folder) / f"{g.name}.{fmt}"
        _guard(main, lambda g=g, p=p: export_graph(g.graph, p, fmt, dpi), "Export")
        out.append(p)
    main.message(f"Exported {len(out)} graph(s) to {folder}")
    return out


# ====================================================================== install
def _add(main, menu: QMenu, text: str, slot, key: str) -> QAction:
    act = QAction(text, main)
    act.triggered.connect(lambda _checked=False: slot())
    menu.addAction(act)
    main.data_actions[key] = act
    return act


def install(main) -> None:
    """Add the WP-2 menus to the Origin-style main window (idempotent: a second call on
    the same window does nothing)."""
    if getattr(main, "_data_menus_installed", False):
        return
    main._data_menus_installed = True
    main.data_actions = {}
    m = main.menus
    a = lambda menu, text, slot, key: _add(main, menu, text, slot, key)  # noqa: E731

    # File > Import / Export
    f = m["file"]
    imp_menu = next((act.menu() for act in f.actions() if act.menu() is not None and
                     "Import" in act.text()), None) or f.addMenu("&Import")
    imp_menu.addSeparator()
    a(imp_menu, "Import &Wizard...", lambda: iw.wizard(main), "import_wizard")
    a(imp_menu, "&Multiple ASCII...", lambda: iw.import_dialog(main, ["ascii"],
                                                               "Import Multiple ASCII"),
      "import_multi")
    a(imp_menu, "Excel (&all sheets)...", lambda: iw.import_dialog(main, ["excel"],
                                                                   "Import Excel"),
      "import_excel_all")
    a(imp_menu, "&JCAMP-DX...", lambda: iw.import_dialog(main, ["jcamp"], "Import JCAMP-DX"),
      "import_jcamp")
    a(imp_menu, "Thermo Galactic &SPC...", lambda: iw.import_dialog(main, ["spc"],
                                                                    "Import SPC"),
      "import_spc")
    a(imp_menu, "MAT&LAB...", lambda: iw.import_dialog(main, ["matlab"], "Import MATLAB"),
      "import_matlab")
    a(imp_menu, "&HDF5 / NetCDF...", lambda: iw.import_dialog(main, ["hdf"], "Import HDF5"),
      "import_hdf")
    a(imp_menu, "Sound (&WAV)...", lambda: iw.import_dialog(main, ["wav"], "Import WAV"),
      "import_wav")
    a(imp_menu, "&Image to Matrix...", lambda: iw.import_dialog(main, ["image"],
                                                                "Import Image"),
      "import_image")
    exp_menu = next((act.menu() for act in f.actions() if act.menu() is not None and
                     "Export" in act.text()), None) or f.addMenu("&Export")
    exp_menu.addSeparator()
    a(exp_menu, "ASCII (&options)...", lambda: export_ascii(main), "export_ascii_opts")
    a(exp_menu, "E&xcel...", lambda: export_xlsx(main), "export_xlsx")
    a(exp_menu, "&Report Sheet (HTML / Word)...", lambda: export_report(main),
      "export_report")
    a(exp_menu, "All &Graphs...", lambda: export_all_graphs(main), "export_all_graphs")

    # Edit
    e = m["edit"]
    e.addSeparator()
    a(e, "Paste &Transpose", lambda: paste_transpose(main), "paste_transpose")

    # Column > Mask
    c = m["column"]
    c.addSeparator()
    mk = c.addMenu("&Mask")
    a(mk, "&Mask", lambda: mask_action(main, "mask"), "mask")
    a(mk, "&Unmask", lambda: mask_action(main, "unmask"), "unmask")
    a(mk, "&Swap Mask", lambda: mask_action(main, "swap"), "swap_mask")
    a(mk, "Mask by &Condition...", lambda: mask_condition_dialog(main), "mask_condition")
    mk.addSeparator()
    a(mk, "Unmask Selected &Columns", lambda: mask_action(main, "unmask_columns"),
      "unmask_columns")
    a(mk, "Unmask &All", lambda: mask_action(main, "unmask_all"), "unmask_all")
    a(c, "Simple Column &Math...", lambda: md.column_math_dialog(main), "column_math")

    # Worksheet
    w = m["worksheet"]
    w.addSeparator()
    a(w, "Sort Worksheet (&Nested)...", lambda: sort_dialog(main), "sort_nested")
    a(w, "Worksheet &Query...", lambda: query_dialog(main), "query")
    a(w, "S&plit Worksheet...", lambda: split_dialog(main), "split")
    a(w, "Stac&k Columns...", lambda: stack_dialog(main), "stack")
    a(w, "&Unstack Columns...", lambda: unstack_dialog(main), "unstack")
    a(w, "Pivot &Table...", lambda: pivot_dialog(main), "pivot")
    red = w.addMenu("&Reduce")
    a(red, "Reduce &Rows...", lambda: reduce_rows_dialog(main), "reduce_rows")
    a(red, "Reduce &Duplicate X...", lambda: reduce_dup_x_dialog(main), "reduce_dup_x")
    a(red, "Reduce to &Evenly Spaced X...", lambda: reduce_xy_dialog(main), "reduce_xy")
    a(w, "Remove D&uplicate Rows...", lambda: remove_duplicates_dialog(main),
      "remove_duplicates")
    a(w, "Remove Rows with &Missing Values...", lambda: trim_dialog(main), "trim")

    # Analysis > Mathematics / Signal Processing
    an = m["analysis"]
    math_menu = next((act.menu() for act in an.actions() if act.menu() is not None and
                      "Mathematics" in act.text()), None) or an.addMenu("&Mathematics")
    math_menu.addSeparator()
    a(math_menu, "Interpolate/E&xtrapolate...", lambda: md.interpolate_dialog(main),
      "interpolate")
    a(math_menu, "Interpolate/Extrapolate &Y from X...",
      lambda: md.interp_y_from_x_dialog(main), "interp_y_from_x")
    a(math_menu, "XYZ Interpolation (&2D)...", lambda: md.xyz_interpolate_dialog(main),
      "interp_2d")
    a(math_menu, "&3D Interpolation...", lambda: md.interpolate_3d_dialog(main),
      "interp_3d")
    math_menu.addSeparator()
    a(math_menu, "&Average Multiple Curves...", lambda: md.average_curves_dialog(main),
      "average_curves")
    a(math_menu, "Simple &Curve Math...", lambda: md.curve_math_dialog(main), "curve_math")
    a(math_menu, "Simple Column &Math...", lambda: md.column_math_dialog(main),
      "column_math2")
    a(math_menu, "Subtract &Reference Data...", lambda: md.subtract_reference_dialog(main),
      "subtract_reference")
    a(math_menu, "Subtract &Straight Line...", lambda: md.subtract_line_dialog(main),
      "subtract_line")
    a(math_menu, "&Vertical Translate...", lambda: md.translate_dialog(main, "vertical"),
      "translate_v")
    a(math_menu, "&Horizontal Translate...", lambda: md.translate_dialog(main, "horizontal"),
      "translate_h")
    a(math_menu, "&Polygon Area...", lambda: md.polygon_area_dialog(main), "polygon_area")
    a(math_menu, "S&et Operations...", lambda: md.set_operation_dialog(main), "set_ops")

    sig_menu = next((act.menu() for act in an.actions() if act.menu() is not None and
                     "Signal" in act.text()), None) or an.addMenu("&Signal Processing")
    sig_menu.addSeparator()
    a(sig_menu, "Smoothing (LOESS, &Binomial, SG...)...", lambda: sd.smooth_dialog(main),
      "smooth2")
    fft_m = sig_menu.addMenu("F&FT")
    a(fft_m, "&FFT...", lambda: sd.fft_dialog(main), "fft2")
    a(fft_m, "&IFFT...", lambda: sd.ifft_dialog(main), "ifft")
    a(fft_m, "FFT &Filters...", lambda: sd.fft_filter_dialog(main), "fft_filter")
    a(fft_m, "&STFT...", lambda: sd.stft_dialog(main), "stft")
    a(sig_menu, "&IIR Filter...", lambda: sd.iir_dialog(main), "iir")
    a(sig_menu, "&Convolution...", lambda: sd.convolution_dialog(main), "convolution")
    a(sig_menu, "&Deconvolution...", lambda: sd.convolution_dialog(main, True),
      "deconvolution")
    a(sig_menu, "C&orrelation...", lambda: sd.correlation_dialog(main), "correlation")
    a(sig_menu, "Co&herence...", lambda: sd.coherence_dialog(main), "coherence")
    a(sig_menu, "&Hilbert Transform...", lambda: sd.hilbert_dialog(main), "hilbert")
    a(sig_menu, "&Envelope...", lambda: sd.hilbert_dialog(main, True), "envelope")
    a(sig_menu, "Deci&mation...", lambda: sd.decimate_dialog(main), "decimate")
    wav = sig_menu.addMenu("&Wavelet")
    for key, text in (("cwt", "&Continuous Wavelet..."), ("dwt", "&Decompose (DWT)..."),
                      ("decompose", "&Multi-Scale DWT..."), ("denoise", "&Denoise..."),
                      ("smooth", "&Smooth...")):
        a(wav, text, lambda k=key: sd.wavelet_dialog(main, k), f"wavelet_{key}")
    wav.menuAction().setVisible(wv.available())   # optional PyWavelets

    # Matrix
    mm = m["matrix"]
    mm.addSeparator()
    for key, text in (("math", "Simple &Math..."), ("inverse", "&Inverse"),
                      ("filter", "Smoothing / &Filter..."), ("crop", "&Crop..."),
                      ("rotate", "Rotate (&any angle)..."),
                      ("resample", "2D &Interpolate..."), ("integrate", "In&tegrate..."),
                      ("stats", "&Statistics")):
        a(mm, text, lambda k=key: md.matrix_dialog(main, k), f"matrix_{key}")
    sig2 = mm.addMenu("Signal Processin&g")
    for key, text in (("fft2", "2D &FFT..."), ("fft2_filter", "2D FFT &Filter..."),
                      ("correlate2d", "2D &Correlation")):
        a(sig2, text, lambda k=key: md.matrix_dialog(main, k), f"matrix_{key}")

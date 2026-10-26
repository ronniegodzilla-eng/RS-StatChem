"""Graph & Analysis (OriginLab-style) module: worksheet model, formulas, plotting, fitting,
statistics, project files and the main window."""
from __future__ import annotations

import numpy as np
import pytest

from rsstatchem.ui.origin import analysis as an
from rsstatchem.ui.origin import formula as fx
from rsstatchem.ui.origin.model import (Folder, GraphDoc, Notes, Project, ReportSheet, Workbook,
                                        Worksheet, format_cell, load_project, parse_cell,
                                        save_project)
from rsstatchem.ui.origin.plotting import PLOT_TYPES, PlotError, plot_columns, refresh_links


def _book(n: int = 11) -> Workbook:
    book = Workbook.new("Book1")
    ws = book.sheets[0]
    x = np.linspace(0, 10, n)
    ws.set_column_values(0, list(x))
    ws.set_column_values(1, list(2.0 * x + 1.0 + np.sin(x) * 0.2))
    ws.columns[0].long_name, ws.columns[0].units = "Concentration", "mg/L"
    ws.columns[1].long_name = "Absorbance"
    ws.add_column("yEr")
    ws.set_column_values(2, [0.1] * n)
    ws.add_column("Y")
    ws.set_column_values(3, list(x ** 2))
    return book


@pytest.fixture
def ctx(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("RSSTATCHEM_HOME", str(tmp_path))
    from rsstatchem.app import AppContext

    context = AppContext(qapp)
    quits = []
    monkeypatch.setattr(context.app, "quit", lambda: quits.append(True))
    context.quits = quits
    return context


def _no_prompts(monkeypatch):
    """Fail the test if anything tries to open a modal message box."""
    from PySide6.QtWidgets import QMessageBox

    def boom(*args, **kwargs):
        raise AssertionError("unexpected modal dialog")
    for name in ("question", "warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, boom)
    monkeypatch.setattr(QMessageBox, "exec", boom)


# ====================================================================== model
def test_worksheet_labels_and_designations():
    ws = Worksheet.new()
    assert [ws.label(i) for i in range(2)] == ["A(X)", "B(Y)"]
    ws.add_column("yEr")
    ws.add_column("X")
    ws.add_column("Y")
    assert [ws.label(i) for i in range(5)] == ["A(X1)", "B(Y1)", "C(yEr±)", "D(X2)", "E(Y2)"]
    assert ws.x_of(1) == 0 and ws.x_of(4) == 3
    assert ws.yerr_of(1) == 2 and ws.yerr_of(4) is None
    ws.rename_column(1, "Abs")
    assert ws.columns[1].name == "Abs"
    with pytest.raises(ValueError):
        ws.rename_column(2, "abs")
    assert ws.next_short_name() == "B"
    assert parse_cell("1,5") == 1.5 and parse_cell("abc") == "abc" and parse_cell("") is None
    assert format_cell(2.0) == "2" and format_cell(0.1 + 0.2) == "0.3"


def test_formula_translation_and_evaluation():
    book = _book(5)
    ws = book.sheets[0]
    vals = fx.evaluate_formula(ws, "col(A)^2 + i")
    assert vals[:5] == [0.0 + 1, 6.25 + 2, 25 + 3, 56.25 + 4, 100 + 5]
    assert vals[5] is None   # rows without data stay empty
    t = fx.evaluate_formula(ws, 'tan(col("Concentration"))')
    assert t[1] == pytest.approx(np.tan(2.5))
    assert fx.evaluate_formula(ws, "log(100) + ln(1) + sqrt(16)")[0] == pytest.approx(6.0)
    assert fx.evaluate_formula(ws, "B*0 + pi")[0] == pytest.approx(np.pi)
    with pytest.raises(fx.FormulaError, match="No column Q"):
        fx.evaluate_formula(ws, "col(Q) + 1")
    rnd = fx.evaluate_formula(ws, "rnd()")
    assert all(0 <= v <= 1 for v in rnd)
    fx.apply_formula(ws, 3, "col(A)*10")
    assert ws.columns[3].formula == "col(A)*10"
    ws.set_cell(0, 0, 7.0)
    assert fx.recalculate(ws) == [3] and ws.cell(0, 3) == 70.0


# ====================================================================== plotting
def test_plot_selected_columns_origin_style():
    book = _book()
    ws = book.sheets[0]
    g = plot_columns(book, ws, [1, 3], "line_symbol", "Graph1")
    layer = g.layers[0]
    assert [p.name for p in layer.plots] == ["Absorbance", "D"]
    assert layer.x.title == "Concentration (mg/L)" and layer.y.title == "Absorbance"
    assert layer.x.tick_dir == "out" and layer.legend.visible and not layer.legend.frame
    assert layer.plots[0].yerr == "[Book1]Sheet1!C"   # the Y Error column to the right
    assert layer.plots[0].options["source"] == {"book": "Book1", "sheet": "Sheet1", "x": "A",
                                                "y": "B", "yerr": "C"}
    # graphs stay linked to their worksheet
    ws.set_cell(0, 1, 99.0)
    assert refresh_links(g, {"Book1": book})
    assert g.values(layer.plots[0].y)[0] == 99.0
    assert not refresh_links(g, {"Book1": book})
    with pytest.raises(PlotError):
        plot_columns(book, ws, [], "line", "G")


@pytest.mark.parametrize("kind", sorted(PLOT_TYPES))
def test_every_plot_type_builds_and_renders(kind):
    from rsstatchem.core.graph import export_bytes

    book = _book()
    ws = book.sheets[0]
    cols = [1, 3]
    if kind in ("contour", "heatmap", "surface3d", "scatter3d", "ternary"):
        ws.columns[3].designation = "Z"
        cols = [3]
    g = plot_columns(book, ws, cols, kind, "G")
    assert g.layers and any(layer.plots for layer in g.layers)
    assert export_bytes(g, "png", dpi=40)[:4] == b"\x89PNG"


# ====================================================================== analysis
def test_linear_fit_report_matches_numpy():
    book = _book()
    ws = book.sheets[0]
    out = an.run_fit(an.xy_data(book, ws, 1), "Line", "linear")
    slope, intercept = np.polyfit(ws.columns[0].numeric()[:11], ws.columns[1].numeric()[:11], 1)
    assert out.result.params["Slope"] == pytest.approx(slope)
    assert out.result.params["Intercept"] == pytest.approx(intercept)
    names = [s.name for s in book.sheets]
    assert names == ["Sheet1", "FitLinear1", "FitLinearCurve1"]
    report = book.sheet("FitLinear1")
    assert isinstance(report, ReportSheet)
    titles = [t["title"] for t in report.tables()]
    assert titles[:4] == ["Notes", "Parameters", "Statistics", "ANOVA"]
    params = report.table("Parameters")
    assert params["rows"][1][1] == "Slope" and float(params["rows"][1][2]) == \
        pytest.approx(slope, rel=1e-5)
    assert any(i["kind"] == "image" for i in report.items)
    curve = book.sheet("FitLinearCurve1")
    assert curve.columns[1].long_name == "Linear Fit of Absorbance"
    assert "Slope" in out.log and "R-Square" in out.log
    # a second fit numbers its sheets like Origin
    an.run_fit(an.xy_data(book, ws, 1), "Poly2", "poly")
    assert book.sheet("FitPolynomial1") is not None


def test_nonlinear_fit_and_column_operations():
    book = Workbook.new()
    ws = book.sheets[0]
    x = np.linspace(0, 5, 40)
    ws.set_column_values(0, list(x))
    ws.set_column_values(1, list(3.0 * np.exp(-x / 1.5) + 0.5))
    out = an.run_fit(an.xy_data(book, ws, 1), "ExpDec1", "nlfit")
    assert out.result.params["t1"] == pytest.approx(1.5, rel=1e-4)
    data = an.xy_data(book, ws, 1)
    idx, text = an.smooth_column(data, "savgol", 7, 2)
    assert ws.columns[idx].long_name.startswith("Smoothed") and "Smoothing" in text
    idx, _ = an.differentiate_column(an.xy_data(book, ws, 1))
    assert ws.columns[idx].numeric()[5] == pytest.approx(-2 * np.exp(-x[5] / 1.5), rel=0.05)
    idx, text = an.integrate_column(an.xy_data(book, ws, 1))
    assert "Area" in text
    sheet, _ = an.fft_sheet(an.xy_data(book, ws, 1))
    assert sheet.columns[0].long_name == "Frequency"


def test_descriptive_statistics_report():
    book = _book()
    ws = book.sheets[0]
    report, text = an.statistics_on_columns(book, ws, [0, 1])
    assert report.name == "DescStatsOnCols1"
    table = report.table("Descriptive Statistics")
    b = ws.columns[1].numeric()[:11]
    row = table["rows"][1]
    header = table["columns"]
    assert row[header.index("N total")] == "11"
    assert float(row[header.index("Mean")]) == pytest.approx(b.mean(), rel=1e-5)
    assert float(row[header.index("Standard Deviation")]) == pytest.approx(b.std(ddof=1),
                                                                           rel=1e-5)
    assert "Statistics on Columns" in text
    rep, _ = an.one_way_anova(book, ws, [1, 3])
    assert rep.table("Overall ANOVA") is not None
    rep, _ = an.two_sample_t(book, ws, 1, 3)
    assert rep.table("t-Test Statistics") is not None


# ====================================================================== project file
def test_project_save_open_round_trip(tmp_path):
    book = _book()
    ws = book.sheets[0]
    ws.columns[3].formula = "col(A)^2"
    an.run_fit(an.xy_data(book, ws, 1), "Line", "linear")
    p = Project()
    p.add(book)
    sub = Folder("Results")
    p.root.folders.append(sub)
    g = plot_columns(book, ws, [1], "scatter", "Graph1")
    p.add(GraphDoc("Graph1", g), sub)
    p.add(Notes("Notes1", "lab notes"))
    p.hidden.add("Notes1")
    path = tmp_path / "proj.rsop"
    save_project(p, path, {"active": "Graph1"})
    q, view = load_project(path)
    assert view["active"] == "Graph1"
    assert q.name == "proj" and set(q.windows) == {"Book1", "Graph1", "Notes1"}
    b2 = q.windows["Book1"]
    w2 = b2.sheets[0]
    assert [c.designation for c in w2.columns] == ["X", "Y", "yEr", "Y"]
    assert w2.columns[0].units == "mg/L" and w2.columns[3].formula == "col(A)^2"
    assert w2.columns[1].values == ws.columns[1].values
    assert [s.name for s in b2.sheets] == ["Sheet1", "FitLinear1", "FitLinearCurve1"]
    assert b2.sheets[1].table("Parameters") == book.sheets[1].table("Parameters")
    assert q.root.folders[0].name == "Results" and q.root.folders[0].windows == ["Graph1"]
    assert q.windows["Graph1"].graph.layers[0].plots[0].name == "Absorbance"
    assert q.hidden == {"Notes1"} and q.windows["Notes1"].text == "lab notes"


# ====================================================================== Qt: workbook window
def test_book_window_editing_undo_and_paste(qapp):
    from rsstatchem.ui.origin.sheet_view import HEADER_ROWS, BookWindow

    book = Workbook.new()
    bw = BookWindow(book)
    model = bw.sheet_view().model_
    assert model.headerData(0, __import__("PySide6.QtCore").QtCore.Qt.Orientation.Horizontal) \
        == "A(X)"
    assert model.rowCount() == HEADER_ROWS + 32
    assert model.setData(model.index(HEADER_ROWS, 0), "1.5")
    assert book.sheets[0].cell(0, 0) == 1.5
    assert model.setData(model.index(0, 1), "Signal")          # Long Name row
    assert book.sheets[0].columns[1].long_name == "Signal"
    assert model.setData(model.index(3, 1), "col(A)*2")        # F(x)= row
    assert book.sheets[0].cell(0, 1) == 3.0
    bw.undo_stack.undo()
    assert book.sheets[0].columns[1].formula == "" and book.sheets[0].cell(0, 1) is None
    bw.undo_stack.redo()
    assert book.sheets[0].cell(0, 1) == 3.0
    # Excel-style paste with a header row: header -> Long Name
    bw.paste_block([["Time", "Temp", "Pressure"], ["1", "20.5", "1.0"], ["2", "21", "1.1"]],
                   HEADER_ROWS, 0)
    ws = book.sheets[0]
    assert [c.long_name for c in ws.columns] == ["Time", "Temp", "Pressure"]
    assert ws.cell(1, 2) == 1.1 and len(ws.columns) == 3
    bw.select_columns([1])
    assert bw.copy() == "20.5\n21"
    assert bw.copy(with_headers=True).split("\n")[:2] == ["Temp", ""]
    bw.add_sheet()
    assert [s.name for s in book.sheets] == ["Sheet1", "Sheet2"] and bw.tabs.count() == 2
    bw.undo_stack.undo()
    assert bw.tabs.count() == 1


# ====================================================================== Qt: main window
def test_open_module_and_close_without_prompt(ctx, monkeypatch):
    _no_prompts(monkeypatch)
    from rsstatchem.ui.modules import open_module

    win = open_module(ctx, "graph")
    assert win.isVisible() and win.ctx is ctx and win in ctx.windows
    titles = [a.text().replace("&", "") for a in win.menuBar().actions()]
    assert titles == ["File", "Edit", "View", "Plot", "Column", "Worksheet", "Analysis",
                      "Statistics", "Tools", "Format", "Window", "Help"]
    help_items = [a.text().replace("&", "") for a in win.menus["help"].actions()]
    assert "Check for Updates..." in help_items and any("About" in t for t in help_items)
    win.menus["window"].aboutToShow.emit()
    assert any(a.text() == "&Modules" for a in win.menus["window"].actions())
    assert list(win.project.windows) == ["Book1"] and not win.modified
    names = {d.objectName() for d in win.findChildren(type(win.pe_dock))}
    assert {"projectExplorer", "objectManager", "resultsLog", "messagesLog"} <= names
    win.close()
    assert win not in ctx.windows and ctx.quits == [True]


def test_main_window_workflow(ctx, monkeypatch, tmp_path):
    _no_prompts(monkeypatch)
    from rsstatchem.ui.origin import open_workspace
    from rsstatchem.ui.origin.sheet_view import HEADER_ROWS

    win = open_workspace(ctx)
    bw = win.active_book_window()
    x = np.arange(1, 9, dtype=float)
    rows = [["Time", "Signal"]] + [[f"{v:g}", f"{3 * v + 2:g}"] for v in x]
    bw.paste_block(rows, HEADER_ROWS, 0)
    assert win.modified
    bw.select_columns([1])
    win.set_designation("Y")
    gwin = win.plot("scatter")
    assert gwin is not None and win.active_graph_window() is gwin
    assert [a.text().replace("&", "") for a in win.menuBar().actions()][3] == "Graph"
    assert gwin.graph.layers[0].x.title == "Time"
    choices, current, g = win.analysis_inputs()
    assert g is gwin and choices[0].ycol == "B"
    out = win.run_fit("linear", {"input": choices[0], "model": "Line", "to_graph": True}, gwin)
    assert out.result.params["Slope"] == pytest.approx(3.0)
    assert len(gwin.graph.layers[0].plots) == 2 and gwin.graph.layers[0].annotations
    assert "Linear Fit" in win.results.toPlainText()
    # editing the worksheet refreshes the linked graph
    win.show_window("Book1")
    bw.tabs.setCurrentIndex(0)
    model = bw.sheet_view().model_
    model.setData(model.index(HEADER_ROWS, 1), "50")
    plot = gwin.graph.layers[0].plots[0]
    assert gwin.graph.values(plot.y)[0] == 50.0
    # statistics report into the workbook
    bw.select_columns([1])
    win.statistics_on_columns()
    assert bw.book.sheet("DescStatsOnCols1") is not None
    # new folder + notes, then save and reopen the project
    win.new_folder("Analysis")
    win.new_notes()
    path = tmp_path / "work.rsop"
    assert win.save_to(str(path)) and not win.modified
    win2 = open_workspace(ctx, str(path))
    assert set(win2.project.windows) == set(win.project.windows)
    g2 = win2.view_of(gwin.graph.title)
    assert len(g2.graph.layers[0].plots) == 2
    assert [f.name for f in win2.project.root.folders] == ["Analysis"]
    # rename / duplicate / delete windows
    assert win2.rename_window("Book1", "Calibration")
    src = g2.graph.layers[0].plots[0].options["source"]
    assert src["book"] == "Calibration"
    win2.duplicate_window(gwin.graph.title)
    assert len(win2.project.graphs()) == 2
    win2.delete_windows(["Graph1"])
    assert "Graph1" not in win2.project.windows and "Graph1" not in win2.subs
    win2.modified = False
    win2.close()
    win.close()
    assert ctx.windows == []


def test_open_rsg_and_import_csv(ctx, monkeypatch, tmp_path):
    _no_prompts(monkeypatch)
    from rsstatchem.core.graph.export import save_graph
    from rsstatchem.ui.modules import open_module

    book = _book()
    g = plot_columns(book, book.sheets[0], [1], "line", "Spectrum")
    rsg = tmp_path / "spectrum.rsg"
    save_graph(g, rsg)
    win = open_module(ctx, "graph", str(rsg))
    assert "spectrum" in win.project.windows and win.active_graph_window() is not None
    assert not win.modified
    csv = tmp_path / "data.csv"
    csv.write_text("Wavelength,Absorbance\n400,0.1\n450,0.5\n500,0.2\n", encoding="utf-8")
    bw = win.import_file(str(csv))
    ws = bw.worksheet()
    assert [c.long_name for c in ws.columns] == ["Wavelength", "Absorbance"]
    assert ws.columns[0].designation == "X" and ws.cell(1, 1) == 0.5
    assert ws.name == "data"
    win.modified = False
    win.close()


def test_column_commands_and_set_values(ctx, monkeypatch):
    _no_prompts(monkeypatch)
    from rsstatchem.ui.origin import open_workspace

    win = open_workspace(ctx)
    bw = win.active_book_window()
    ws = bw.worksheet()
    bw.select_columns([0])
    win.fill_columns("rows")
    assert ws.cell(31, 0) == 32.0
    win.add_columns(2)
    assert [c.name for c in ws.columns] == ["A", "B", "C", "D"]
    win.set_column_values(bw, 1, "col(A)^2")
    assert ws.cell(3, 1) == 16.0 and ws.columns[1].formula == "col(A)^2"
    with pytest.raises(fx.FormulaError):
        win.set_column_values(bw, 2, "col(Z)")
    bw.select_columns([2, 3])
    win.set_designation("yEr")
    assert ws.label(2) == "C(yEr±)"
    bw.select_columns([3])
    win.delete_columns()
    assert len(ws.columns) == 3
    bw.undo_stack.undo()
    assert len(bw.worksheet().columns) == 4
    win.modified = False
    win.close()

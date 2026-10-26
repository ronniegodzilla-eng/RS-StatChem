"""RSM / DOE module (Design-Expert-style workspace): wizard -> design layout -> analysis ->
optimization -> .rsx file, checked against Montgomery's textbook examples."""
import math

import numpy as np
import pandas as pd
import pytest

from rsstatchem.core.doe import model


@pytest.fixture
def ctx(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("RSSTATCHEM_HOME", str(tmp_path))
    from rsstatchem.app import AppContext

    context = AppContext(qapp)
    quits = []
    monkeypatch.setattr(context.app, "quit", lambda: quits.append(True))
    context.quits = quits
    return context


@pytest.fixture
def chem():
    from rsstatchem.ui.dx.examples import chemical_process

    return chemical_process()


@pytest.fixture
def win(ctx, chem):
    from rsstatchem.ui.dx.window import DXMainWindow

    w = DXMainWindow(ctx, chem)
    ctx.register_window(w)
    w.show()
    yield w
    if w.doc is not None:
        w.doc.modified = False


def pump(qapp, n=5):
    for _ in range(n):
        qapp.processEvents()


# ---------------------------------------------------------------------------- module window
def test_open_module_returns_window_without_modal_loop(ctx, qapp):
    from rsstatchem.ui.dx.window import DXMainWindow
    from rsstatchem.ui.modules import open_module

    w = open_module(ctx, "rsm")
    assert isinstance(w, DXMainWindow) and w.isVisible() and w.ctx is ctx
    assert w in ctx.windows and w.doc is None
    menus = [a.text().replace("&", "") for a in w.menuBar().actions()]
    assert menus == ["File", "Edit", "View", "Display Options", "Design Tools", "Analysis",
                     "Optimization", "Window", "Help"]
    help_menu = w.menuBar().actions()[-1].menu()
    texts = [a.text().replace("&", "") for a in help_menu.actions()]
    assert "Check for Updates..." in texts and any(t.startswith("About") for t in texts)
    w._fill_window_menu()
    assert any("Modules" in a.text() for a in w.window_menu.actions())
    pump(qapp)  # the New Design wizard appears on the empty workspace, non-modally
    assert w.wizard is not None and w.wizard.isVisible()
    w.close()  # untouched: no save prompt
    assert ctx.windows == [] and ctx.quits == [True]


def test_open_workspace_with_file_and_bad_file(ctx, tmp_path, chem):
    from rsstatchem.ui.dx import open_workspace

    path = tmp_path / "ccd.rsx"
    chem.save(str(path))
    w = open_workspace(ctx, str(path))
    assert w.doc is not None and w.doc.n_runs == 13
    assert str(path.resolve()) in ctx.recent_files()
    bad = tmp_path / "bad.rsx"
    bad.write_text("not json")
    w2 = open_workspace(ctx, str(bad), wizard=False)
    assert w2.doc is None  # error shown without blocking
    w.close()
    w2.close()


# ---------------------------------------------------------------------------- wizard
def _ccd_wizard(qapp):
    from rsstatchem.ui.dx.wizard import NewDesignWizard

    wiz = NewDesignWizard(None, "rsm", "ccd")
    wiz.set_factors([{"name": "Time", "units": "min", "low": 80, "high": 90},
                     {"name": "Temperature", "units": "°C", "low": 170, "high": 180}])
    return wiz


def test_wizard_central_composite(qapp):
    wiz = _ccd_wizard(qapp)
    assert wiz.design == "ccd" and wiz.table_columns[-2:] == ["-alpha", "+alpha"]
    r = wiz.table_columns.index("+alpha")
    assert float(wiz.table.item(0, r).text()) == pytest.approx(85 + 5 * 2 ** 0.5, rel=1e-5)
    wiz.go(1)
    wiz.set_responses([("Yield", "%"), ("Purity", "%")])
    got = []
    wiz.created.connect(got.append)
    wiz.finish()
    doc = got[0]
    assert doc.n_runs == 13 and doc.meta["design"] == "ccd"
    assert [f["name"] for f in doc.factors] == ["Time", "Temperature"]
    assert [r["name"] for r in doc.responses] == ["Yield", "Purity"]
    assert doc.factors[0]["low"] == 80 and doc.factors[0]["high"] == 90
    assert sorted(doc.ds.df["Run"]) == list(range(1, 14))
    assert doc.modified


def test_wizard_ranges_in_terms_of_alpha(qapp):
    wiz = _ccd_wizard(qapp)
    wiz.ccd_alpha_entry.setChecked(True)
    spec = wiz.spec()
    a = 2 ** 0.5
    assert spec["factors"][0]["low"] == pytest.approx(85 - 5 / a)
    assert spec["factors"][0]["high"] == pytest.approx(85 + 5 / a)


@pytest.mark.parametrize("family,design", [
    ("factorial", "factorial"), ("factorial", "general"), ("factorial", "pb"),
    ("factorial", "taguchi"), ("rsm", "bbd"), ("rsm", "threelevel"), ("rsm", "dopt"),
    ("mixture", "lattice"), ("mixture", "centroid"), ("mixture", "ev"),
    ("mixture", "mixdopt")])
def test_wizard_builds_every_design(qapp, family, design):
    from rsstatchem.ui.dx.wizard import NewDesignWizard

    wiz = NewDesignWizard(None, family, design)
    assert wiz.design == design
    wiz.finish()
    assert wiz.document is not None and wiz.document.n_runs > 0, wiz.error.text()
    assert wiz.document.meta["design"] == design


def test_wizard_fractional_factorial(qapp):
    from rsstatchem.ui.dx.wizard import NewDesignWizard

    wiz = NewDesignWizard(None, "factorial", "factorial")
    wiz.k_spin.setValue(5)
    fractions = [wiz._opt["fraction"].itemData(i) for i in range(wiz._opt["fraction"].count())]
    assert fractions[:2] == [0, 1]
    wiz.set_option("fraction", 1)
    wiz.finish()
    assert wiz.document.n_runs == 16 and wiz.document.meta["resolution"] == 5


# ---------------------------------------------------------------------------- design layout
def test_layout_headers_paste_undo_and_coded(win, qapp):
    page = win.show_node("design")
    m = page.model
    kinds = [k for k, _ in m.columns]
    assert kinds == ["select", "std", "run", "factor", "factor", "response"]
    from PySide6.QtCore import Qt

    assert m.headerData(3, Qt.Orientation.Horizontal) == "Factor 1\nA:Time\nmin"
    assert m.headerData(5, Qt.Orientation.Horizontal) == "Response 1\nR1:Yield\n%"
    runs = [int(m.data(m.index(r, 2))) for r in range(m.rowCount())]
    assert runs == list(range(1, 14))  # shown in run order
    col = win.doc.response_cols[0]
    old = [win.doc.value(m.rows[r], col) for r in range(3)]
    n = m.paste_text("1,5\n2.5\n3\n", 0, 5)  # Excel block, decimal comma accepted
    assert n == 3
    assert [win.doc.value(m.rows[r], col) for r in range(3)] == [1.5, 2.5, 3.0]
    win.undo.undo()
    assert [win.doc.value(m.rows[r], col) for r in range(3)] == old
    assert m.paste_text("9\t9", 0, 3) == 0  # factor settings are fixed by the design
    m.setData(m.index(0, 5), "")
    assert math.isnan(win.doc.value(m.rows[0], col))
    win.undo.undo()
    win.set_coded(True)
    coded = sorted({m.data(m.index(r, 3)) for r in range(m.rowCount())}, key=float)
    assert coded == ["-1.414", "-1", "0", "1", "1.414"]
    assert win.nav_items["design"].text(0) == "Design (Coded)"
    m.set_order("std")
    assert [int(m.data(m.index(r, 1))) for r in range(m.rowCount())] == list(range(1, 14))


def test_ignore_run_drops_out_of_analysis(win):
    doc = win.doc
    col = doc.response_cols[0]
    assert doc.analyze(col).fit.n == 13
    page = win.show_node("design")
    from PySide6.QtCore import Qt

    page.model.setData(page.model.index(0, 0), Qt.CheckState.Unchecked.value,
                       Qt.ItemDataRole.CheckStateRole)
    assert len(doc.ignored) == 1 and doc.analyze(col).fit.n == 12
    win.undo.undo()
    assert not doc.ignored and doc.analyze(col).fit.n == 13


# ---------------------------------------------------------------------------- analysis
def test_fit_summary_and_anova_match_montgomery(win):
    col = win.doc.response_cols[0]
    view = win.show_node(f"analysis:{col}")
    assert view.strip.labels() == ["Transform", "Fit Summary", "Model", "ANOVA", "Diagnostics",
                                   "Model Graphs"]
    view.set_tab("Fit Summary")
    fs = view.fit_page.report.tables()[0]
    suggested = [r for r in fs.rows if r[-1] == "Suggested"]
    assert suggested and suggested[0][0] == "Quadratic"
    view.set_tab("ANOVA")
    tables = {t.title: t for t in view.anova_page.report.tables()}
    anova = tables["ANOVA for Quadratic model"]
    assert anova.rows[0][:3] == ["Model", "28.25", "5"] and anova.rows[0][6] == "significant"
    lof = next(r for r in anova.rows if r[0].strip() == "Lack of Fit")
    assert lof[6] == "not significant"
    coef = {r[0]: float(r[1]) for r in tables["Coefficients in Terms of Coded Factors"].rows}
    assert coef["Intercept"] == pytest.approx(79.94, abs=0.005)
    assert coef["A²"] == pytest.approx(-1.376, abs=0.001)
    stats_rows = tables["Fit Statistics"].rows
    assert stats_rows[0][3] == "0.9827"
    html = view.anova_page.report.toHtml()
    assert "#d0021b" in html.lower()  # p < 0.01 in Design-Expert red


def test_model_tab_terms_and_auto_select(win):
    col = win.doc.response_cols[0]
    view = win.show_node(f"analysis:{col}")
    view.set_tab("Model")
    page = view.model_page
    marks = {page.table.item(r, 1).text(): page.table.item(r, 0).text()
             for r in range(page.table.rowCount())}
    assert marks["AB"] == "M" and marks["A²B"] == "e"
    assert page.table.item([page.table.item(r, 1).text() for r in
                            range(page.table.rowCount())].index("A³"), 2).text() == "Aliased"
    row_ab = [page.table.item(r, 1).text() for r in range(page.table.rowCount())].index("AB")
    page._clicked(row_ab, 0)  # AB out of the model
    assert (0, 1) not in win.doc.analyze(col).fit.terms
    page._clicked(row_ab, 0)
    assert (0, 1) in win.doc.analyze(col).fit.terms
    page.alpha.setValue(0.10)
    page.auto_select()  # AB has p = 0.1025 > 0.10
    terms = win.doc.analyze(col).fit.terms
    assert (0, 1) not in terms and (0, 0) in terms


def test_transform_tab_sets_transform(win):
    col = win.doc.response_cols[0]
    view = win.show_node(f"analysis:{col}")
    view.set_tab("Transform")
    view.transform_page.radios["ln"].setChecked(True)
    assert win.doc.spec(col).transform.kind == "ln"
    assert win.doc.analyze(col).fit.transform.kind == "ln"
    view.transform_page.radios["none"].setChecked(True)
    assert win.doc.spec(col).transform.kind == "none"


def test_diagnostics_and_model_graphs_render(win):
    from rsstatchem.ui.dx.analysis import DIAGNOSTICS, PROCESS_GRAPHS

    col = win.doc.response_cols[0]
    view = win.show_node(f"analysis:{col}")
    view.set_tab("Diagnostics")
    page = view.diag_page
    for i, (key, _) in enumerate(DIAGNOSTICS):
        page.list.setCurrentRow(i)
        if key == "report":
            assert page.report.tables()[0].title == "Diagnostics Case Statistics"
        else:
            assert page.graph.png, key
    view.set_tab("Model Graphs")
    gp = view.graph_page
    for key, _, _ in PROCESS_GRAPHS:
        if not gp.buttons[key].isEnabled():
            continue
        gp.graph.set_png(None)
        gp.set_graph(key)
        assert gp.graph.png, key
    gp.set_graph("contour")
    before = gp.graph.png
    gp.tool.set_axes([1, 0])  # swap the axes
    gp.render()
    assert gp.graph.png and gp.graph.png != before


def test_copy_tables_and_graph(win, qapp, tmp_path):
    col = win.doc.response_cols[0]
    view = win.show_node(f"analysis:{col}")
    view.set_tab("ANOVA")
    win.copy()
    mime = qapp.clipboard().mimeData()
    assert "ANOVA for Quadratic model" in mime.html() and "<table" in mime.html()
    assert "Sum of Squares\tdf" in mime.text()
    view.set_tab("Model Graphs")
    g = win.current_graph()
    assert g is not None and g.png
    win.copy()
    assert not qapp.clipboard().image().isNull()
    from rsstatchem.ui.dx.widgets import save_png

    save_png(g.png, str(tmp_path / "contour.png"))
    save_png(g.png, str(tmp_path / "contour.jpg"))
    assert (tmp_path / "contour.png").stat().st_size > 1000
    assert (tmp_path / "contour.jpg").stat().st_size > 1000


# ---------------------------------------------------------------------------- design branch
def test_summary_evaluation_graph_columns(win):
    s = win.show_node("summary")
    titles = [t.title for t in s.report.tables()]
    assert titles[:3] == ["Build Information", "Factors", "Responses"]
    ev = win.show_node("evaluation")
    ev.order.setCurrentIndex(ev.order.findData("quadratic"))
    t = {x.title: x for x in ev.report.tables()}
    df = dict(t["Degrees of Freedom for Evaluation (Quadratic model)"].rows)
    assert (df["Model"], df["Residuals"], df["  Pure Error"]) == ("5", "7", "4")
    terms = {r[0]: r for r in t["Terms"].rows}
    assert float(terms["A"][1]) == pytest.approx(0.3536, abs=1e-4)
    assert float(terms["AB"][1]) == pytest.approx(0.5, abs=1e-4)
    assert float(terms["A²"][3]) == pytest.approx(1.017, abs=1e-3)
    assert ev.graph.png
    gc = win.show_node("graphcols")
    assert gc.graph.png
    notes = win.show_node("notes")
    notes.edit.setPlainText("hello")
    notes.commit()
    assert win.doc.notes == "hello"


# ---------------------------------------------------------------------------- optimization
def test_numerical_optimization_finds_stationary_point(win):
    col = win.doc.response_cols[0]
    view = win.show_node("numerical")
    view.set_goal(col, "maximize", 75, 81)
    view.strip.set_current(1)
    sols = view.solutions
    best = sols[0]
    # Montgomery: stationary point x1 = 0.389, x2 = 0.306 (86.95 min, 176.53 °C), 80.21 %
    assert best["x"][0] == pytest.approx(86.95, abs=0.03)
    assert best["x"][1] == pytest.approx(176.53, abs=0.03)
    assert best["pred"][col] == pytest.approx(80.21, abs=0.01)
    assert view.sol_graph.png
    view._show_solution_view(1)
    assert view.sol_graph.png
    view._show_solution_view(2)
    assert view.sol_report.tables()
    view.strip.set_current(2)
    assert view.graph.png
    pp = win.show_node("prediction")
    pp.use_optimum()
    pp.update_prediction()
    pred = next(t for t in pp.report.tables() if t.title == "Prediction")
    assert float(pred.rows[0][1]) == pytest.approx(80.21, abs=0.01)


def test_graphical_overlay_and_confirmation(win):
    col = win.doc.response_cols[0]
    g = win.show_node("graphical")
    g.set_limits(col, 79.0, None)
    g.strip.set_current(1)
    assert g.graph.png and 0 < g.feasible_fraction < 1
    c = win.show_node("confirmation")
    c.set_data(col, 3, 80.0)
    fit = win.doc.fits()[col]
    iv = fit.interval(np.array([[85.0, 175.0]]))
    assert float(c.table.item(0, 1).text()) == pytest.approx(iv["mean"][0], abs=0.005)
    assert c.table.item(0, 8).text().startswith("Confirmed")
    c.set_data(col, 3, 85.0)
    assert c.table.item(0, 8).text() == "Not confirmed"
    co = win.show_node("coefficients")
    t = co.report.tables()[0]
    assert t.columns[:2] == ["Term", "R1:Yield"]


# ---------------------------------------------------------------------------- files
def test_rsx_round_trip(tmp_path, chem):
    from rsstatchem.ui.dx.document import DesignDocument

    col = chem.response_cols[0]
    spec = chem.spec(col)
    spec.model, spec.terms = "manual", ["0", "1", "0.0", "1.1"]
    spec.transform = model.Transform("sqrt")
    chem.set_spec(spec)
    chem.notes = "Montgomery 11.2"
    chem.set_ignored([2], True)
    chem.optimization["numerical"] = [{"name": col, "kind": "response", "goal": "maximize",
                                       "lower": 75.0, "upper": 81.0, "target": None,
                                       "w_lower": 1.0, "w_upper": 1.0, "importance": 3}]
    path = tmp_path / "design.rsx"
    chem.save(str(path))
    assert not chem.modified
    back = DesignDocument.load(str(path))
    pd.testing.assert_frame_equal(back.ds.df, chem.ds.df)
    assert back.factors == chem.factors and back.responses == chem.responses
    assert back.notes == "Montgomery 11.2" and back.ignored == {2}
    s2 = back.spec(col)
    assert s2.terms == ["0", "1", "0.0", "1.1"] and s2.transform.kind == "sqrt"
    assert back.analyze(col).fit.terms == chem.analyze(col).fit.terms
    assert back.optimization["numerical"][0]["goal"] == "maximize"
    assert back.ds.var("Space_Type").value_labels == chem.ds.var("Space_Type").value_labels
    with pytest.raises(ValueError):
        bad = tmp_path / "x.rsx"
        bad.write_text('{"format": "other"}')
        DesignDocument.load(str(bad))


def test_save_through_window_and_import_responses(win, tmp_path, monkeypatch):
    path = tmp_path / "saved.rsx"
    assert win._write(str(path))
    assert path.exists() and not win.doc.modified
    assert "saved.rsx" in win.windowTitle()
    col = win.doc.response_cols[0]
    std = win.doc.ds.df["Std"].to_numpy(int)
    frame = pd.DataFrame({"Std": np.arange(1, 14), "Yield": np.arange(1, 14) * 1.5})
    csv = tmp_path / "data.csv"
    frame.to_csv(csv, index=False)
    assert win.import_responses_from(str(csv)) == ["R1:Yield"]
    assert np.allclose(win.doc.ds.df[col].to_numpy(float), std * 1.5)
    win.undo.undo()
    assert win.doc.ds.df[col].iloc[0] != std[0] * 1.5
    out = tmp_path / "layout.csv"
    win.export_layout_to(str(out))
    exported = pd.read_csv(out)
    assert list(exported.columns[-1:]) == ["R1:Yield"] and len(exported) == 13


# ---------------------------------------------------------------------------- other designs
def test_mixture_design_flow(win):
    from rsstatchem.ui.dx.document import DesignDocument

    doc = DesignDocument.from_spec({
        "design": "lattice", "factors": [{"name": n} for n in ("Water", "Oil", "Soap")],
        "responses": [{"name": "Viscosity", "units": "cP"}],
        "options": {"degree": 2, "augment": True, "center_points": 2}, "seed": 1})
    X = doc.ds.df[doc.factor_cols].to_numpy(float)
    rng = np.random.default_rng(0)
    y = 10 * X[:, 0] + 20 * X[:, 1] + 5 * X[:, 2] + 30 * X[:, 0] * X[:, 1] + \
        rng.normal(0, 0.3, len(X))
    doc.set_values([(i, doc.response_cols[0], float(v)) for i, v in enumerate(y)])
    doc.modified = False
    win.doc.modified = False
    w2 = win.adopt(doc)
    assert w2 is not win and w2.doc is doc
    view = w2.show_node(f"analysis:{doc.response_cols[0]}")
    view.set_tab("Model Graphs")
    assert view.graph_page.graph_key == "ternary" and view.graph_page.graph.png
    view.graph_page.set_graph("trace")
    assert view.graph_page.graph.png
    n = w2.show_node("numerical")
    n.set_goal(doc.response_cols[0], "maximize", float(y.min()), float(y.max()))
    sols = n.solve()
    assert sols and abs(sols[0]["x"].sum() - 1) < 1e-6
    doc.modified = False
    w2.close()
    assert w2 not in win.ctx.windows


def test_filtration_factorial_analysis(win):
    from rsstatchem.ui.dx.examples import filtration_rate

    doc = filtration_rate()
    a = doc.analyze(doc.response_cols[0])
    assert a.fit.n == 16
    spec = doc.spec(doc.response_cols[0])
    spec.model = "2fi"
    spec.reduction = "backward"
    doc.set_spec(spec)
    names = {model.term_name(t) for t in doc.analyze(doc.response_cols[0]).fit.terms}
    assert {"A", "C", "D", "AC", "AD"} <= names and "BC" not in names

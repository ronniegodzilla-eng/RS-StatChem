"""RSM / DOE workspace additions: categoric and hard-to-change factors in the wizard, split-plot
layout and REML analysis, Auto Select settings, DFBETAS, live (rotatable) 3D graphs and contour
flags, foldover augmentation, min-run designs and .rsx compatibility."""
import json

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent

from rsstatchem.core.doe import model


@pytest.fixture
def ctx(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("RSSTATCHEM_HOME", str(tmp_path))
    from rsstatchem.app import AppContext

    context = AppContext(qapp)
    monkeypatch.setattr(context.app, "quit", lambda: None)
    return context


def make_window(ctx, doc):
    from rsstatchem.ui.dx.window import DXMainWindow

    w = DXMainWindow(ctx, doc)
    ctx.register_window(w)
    w.show()
    return w


def pump(qapp, n=5):
    for _ in range(n):
        qapp.processEvents()


@pytest.fixture
def chem_win(ctx):
    from rsstatchem.ui.dx.examples import chemical_process

    w = make_window(ctx, chemical_process())
    yield w
    w.doc.modified = False


@pytest.fixture
def pulp_win(ctx):
    from rsstatchem.ui.dx.examples import paper_pulp

    w = make_window(ctx, paper_pulp())
    yield w
    w.doc.modified = False


# ---------------------------------------------------------------------------- wizard
def test_wizard_type_and_changes_columns(qapp):
    from PySide6.QtWidgets import QComboBox

    from rsstatchem.ui.dx.wizard import NewDesignWizard

    wiz = NewDesignWizard(None, "splitplot", "dopt")
    assert "Changes" in wiz.table_columns and "Level Names" in wiz.table_columns
    assert wiz.split()  # the split-plot family starts with a hard-to-change factor
    wiz.set_factors([{"name": "Temperature", "units": "°C", "low": 150, "high": 200,
                      "changes": "hard"},
                     {"name": "Time", "units": "min", "low": 10, "high": 30},
                     {"name": "Catalyst", "type": "categoric", "levels": "Pd, Pt, Ni"}])
    t = wiz.table_columns.index("Type")
    box = wiz.table.cellWidget(2, t)
    assert isinstance(box, QComboBox) and box.currentText() == "Categoric"
    lv = wiz.table_columns.index("Levels")
    assert wiz.table.item(2, lv).text() == "3"
    assert not wiz.table.item(2, wiz.table_columns.index("Minimum")).flags() \
        & Qt.ItemFlag.ItemIsEditable
    spec = wiz.spec()
    assert spec["factors"][0]["changes"] == "hard"
    assert spec["factors"][2] == {"name": "Catalyst", "units": "", "type": "categoric",
                                  "levels": ["Pd", "Pt", "Ni"]}
    assert spec["options"]["groups"] == 8 and spec["options"]["group_size"] == 4
    # the Type column is editable: make Time categoric, then numeric again
    tb = wiz.table.cellWidget(1, t)
    tb.setCurrentIndex(1)
    assert wiz.spec()["factors"][1]["type"] == "categoric"
    wiz.table.cellWidget(1, t).setCurrentIndex(0)
    assert "type" not in wiz.spec()["factors"][1]
    wiz.go(1)
    got = []
    wiz.created.connect(got.append)
    wiz.finish()
    doc = got[0]
    assert doc.split_plot and doc.group_col == "Group" and doc.n_runs == 32
    assert doc.factors[2]["type"] == "categoric" and doc.factors[2]["labels"] == ["Pd", "Pt",
                                                                                   "Ni"]


def test_wizard_ccd_with_categoric_and_min_run(qapp):
    from rsstatchem.ui.dx.wizard import NewDesignWizard

    wiz = NewDesignWizard(None, "rsm", "ccd")
    wiz.set_factors([{"name": "Time", "low": 80, "high": 90},
                     {"name": "Temperature", "low": 170, "high": 180},
                     {"name": "Solvent", "type": "categoric", "levels": "Water, Ethanol"}])
    wiz.finish()
    assert wiz.document.n_runs == 2 * 13  # the CCD at each solvent
    wiz = NewDesignWizard(None, "factorial", "mr5")
    assert wiz.k_spin.value() == 6
    wiz._update_summary()
    assert "22 runs" in wiz.summary.text()
    wiz.finish()
    assert wiz.document.n_runs == 22 and wiz.document.meta["min_run"] == "MR5"
    wiz = NewDesignWizard(None, "factorial", "mr4")
    wiz.k_spin.setValue(7)
    wiz.finish()
    assert wiz.document.n_runs == 16
    wiz = NewDesignWizard(None, "splitplot", "general")
    wiz.set_factors([{"name": "Pulp", "levels": "a1, a2, a3", "changes": "hard"},
                     {"name": "Temp", "levels": "200, 225, 250, 275"}])
    wiz.set_option("replicates", 3)
    wiz.set_option("block_per_replicate", True)
    wiz.finish()
    doc = wiz.document
    assert doc.n_runs == 36 and doc.n_blocks == 3 and doc.meta["split_plot"]["groups"] == 9


# ---------------------------------------------------------------------------- split plot
def test_split_plot_layout_and_reml_analysis(pulp_win, qapp, tmp_path):
    w = pulp_win
    doc = w.doc
    page = w.show_node("design")
    heads = [page.model.headerData(c, Qt.Orientation.Horizontal)
             for c in range(page.model.columnCount())]
    assert "Group" in heads and any("(HTC)" in str(h) for h in heads)
    # restricted randomisation survives re-randomising
    doc.randomize(3)
    g = doc.ds.df.sort_values("Run")["Group"].to_numpy()
    assert np.sum(g[1:] != g[:-1]) == 8
    col = doc.response_cols[0]
    view = w.show_node(f"analysis:{col}")
    assert view.strip.buttons[3].text() == "ANOVA (REML)"
    view.set_tab("Model")
    mp = view.model_page
    mp.order.setCurrentIndex(mp.order.findData("2fi"))
    view.set_tab("ANOVA")
    titles = [t.title for t in view.anova_page.report.tables()]
    assert "ANOVA (REML) for 2FI model" in titles and "Variance Components" in titles
    reml = view.anova_page.report.tables()[titles.index("ANOVA (REML) for 2FI model")]
    rows = {r[0].strip(): r for r in reml.rows}
    assert rows["A-Pulp Prep"][2] == "4.000" and rows["A-Pulp Prep"][3] == "7.08"
    assert rows["B-Temp"][2] == "18.00" and rows["AB"][3] == "3.15"
    a = doc.analyze(col)
    assert isinstance(a.fit, model.SplitPlotFit)
    # diagnostics, graphs, point prediction and optimisation work on the REML fit
    view.set_tab("Diagnostics")
    keys = [view.diag_page.list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(view.diag_page.list.count())]
    view.diag_page.list.setCurrentRow(keys.index("dfbetas"))
    assert view.diag_page.graph.png
    w.show_node("prediction")
    w.show_node("summary")
    text = " ".join(" ".join(map(str, r)) for t in w.pages["summary"].report.tables()
                    for r in t.rows)
    assert "9 groups" in text and "(HTC)" in text
    # saved and reloaded: groups, hard-to-change flags and the model
    from rsstatchem.ui.dx.document import DesignDocument

    path = tmp_path / "pulp.rsx"
    doc.save(str(path))
    back = DesignDocument.load(str(path))
    assert back.split_plot and back.factors[0]["changes"] == "hard"
    assert np.array_equal(back.ds.df["Group"], doc.ds.df["Group"])
    assert back.analyze(col).fit.term_p_values() == pytest.approx(a.fit.term_p_values())


# ---------------------------------------------------------------------------- auto select
def test_auto_select_settings_saved(chem_win, tmp_path):
    w = chem_win
    col = w.doc.response_cols[0]
    view = w.show_node(f"analysis:{col}")
    view.set_tab("Model")
    mp = view.model_page
    mp.set_selection("forward", "aicc")
    assert not mp.alpha_in.isEnabled() and not mp.alpha.isEnabled()
    mp.set_selection("stepwise", "aicc")  # AICc / BIC: forward or backward only
    assert mp.method.currentData() == "forward"
    mp.set_selection("stepwise", "p", 0.05, 0.10)
    assert mp.alpha_in.isEnabled() and mp.alpha.isEnabled()
    steps = mp.auto_select()
    assert steps and steps[0][0] == "add"
    terms = set(w.doc.analyze(col).fit.terms)
    assert terms == {(0,), (1,), (0, 0), (1, 1)}
    sel = w.doc.spec(col).selection
    assert sel["method"] == "stepwise" and sel["alpha_in"] == 0.05
    assert "Stepwise" in mp.status.text()
    view.set_tab("ANOVA")
    assert view.anova_page.report.tables()[0].title.startswith("Stepwise Selection")
    path = tmp_path / "sel.rsx"
    w.doc.save(str(path))
    from rsstatchem.ui.dx.document import DesignDocument

    back = DesignDocument.load(str(path))
    assert back.spec(col).selection["steps"] == json.loads(json.dumps(sel["steps"]))


# ---------------------------------------------------------------------------- live graphs
def _drag(qapp, canvas, start: QPoint, delta: QPoint, steps: int = 10):
    def send(kind, pos, buttons):
        ev = QMouseEvent(kind, QPointF(pos), QPointF(canvas.mapToGlobal(pos)),
                         Qt.MouseButton.LeftButton, buttons, Qt.KeyboardModifier.NoModifier)
        qapp.sendEvent(canvas, ev)

    send(QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton)
    for i in range(1, steps + 1):
        send(QEvent.Type.MouseMove, start + delta * i / steps, Qt.MouseButton.LeftButton)
        pump(qapp, 1)
    send(QEvent.Type.MouseButtonRelease, start + delta, Qt.MouseButton.NoButton)
    pump(qapp, 2)


def test_rotatable_surface_and_reset_view(chem_win, qapp, tmp_path):
    import matplotlib as mpl

    w = chem_win
    col = w.doc.response_cols[0]
    view = w.show_node(f"analysis:{col}")
    view.set_tab("Model Graphs")
    gp = view.graph_page
    gp.set_graph("surface")
    pump(qapp)
    g = gp.graph
    assert g.canvas is not None and g.axes3d() is not None and g.reset_btn.isVisible()
    start = g.view()
    style = mpl.rcParams.get("axes3d.mouserotationstyle")
    _drag(qapp, g.canvas, g.canvas.rect().center(), QPoint(120, 30))
    turned = g.view()
    assert abs(turned[1] - start[1]) > 5  # the azimuth follows the mouse
    assert mpl.rcParams.get("axes3d.mouserotationstyle") == style  # global default untouched
    png_turned = g.png
    g.reset_view()
    assert g.view()[:2] == pytest.approx(start[:2])
    assert g.png != png_turned  # export / copy take the current view
    g.set_view(40, -60, zoom=1.5)
    assert g.view() == pytest.approx((40, -60, 0, 1.5))
    path = tmp_path / "surface.png"
    from rsstatchem.ui.dx.widgets import save_png

    save_png(g.png, str(path))
    assert path.read_bytes()[:4] == b"\x89PNG"
    # redrawing the same graph (a slider moved) keeps the user's view
    gp.tool.set_point(gp.tool.point() + np.array([1.0, 0.0]))
    gp.render()
    assert g.view()[:2] == pytest.approx((40, -60))
    assert w.current_graph() is g and g.has_graph


def test_contour_flags(chem_win, qapp):
    w = chem_win
    col = w.doc.response_cols[0]
    view = w.show_node(f"analysis:{col}")
    view.set_tab("Model Graphs")
    gp = view.graph_page
    gp.set_graph("contour")
    g = gp.graph
    assert g.probe is not None and g.clear_btn.isVisible()
    lines = g.add_flag(86.9, 176.5)
    fit = w.doc.analyze(col).fit
    pred = fit.predict(np.array([[86.9, 176.5]]))[0]
    assert lines[0].startswith("A: Time = 86.9") and f"{pred:.4g}"[:5] in lines[2]
    assert len(g.flags) == 1
    g.clear_flags()
    assert g.flags == []
    assert g.add_flag(50.0, 176.0) is None  # outside the factor range


def test_mixture_3d_surface_is_live(ctx, qapp):
    from rsstatchem.ui.dx.document import DesignDocument

    doc = DesignDocument.from_spec({"design": "lattice", "factors": [{"name": n} for n in "ABC"],
                                    "options": {"degree": 2, "augment": True}, "seed": 1})
    rng = np.random.default_rng(0)
    doc.set_values([(i, doc.response_cols[0], float(10 + rng.normal()))
                    for i in range(doc.n_runs)])
    w = make_window(ctx, doc)
    view = w.show_node(f"analysis:{doc.response_cols[0]}")
    view.set_tab("Model Graphs")
    gp = view.graph_page
    gp.set_graph("surface3")
    assert gp.graph.axes3d() is not None
    gp.set_graph("ternary")
    assert gp.graph.probe is not None
    assert gp.graph.add_flag(0.5, 0.3) is not None
    doc.modified = False


# ---------------------------------------------------------------------------- foldover
def test_foldover_augments_a_new_block(ctx, qapp):
    from rsstatchem.ui.dx.document import DesignDocument

    doc = DesignDocument.from_spec({"design": "factorial", "randomize": False,
                                    "factors": [{"name": n} for n in "ABCDEFG"],
                                    "options": {"fraction": 4}})
    assert doc.meta["resolution"] == 3 and doc.two_level()
    w = make_window(ctx, doc)
    dlg = w.foldover()
    dlg.kind.setCurrentIndex(0)
    dlg.apply()
    pump(qapp)
    assert dlg.added == 8 and doc.n_runs == 16 and doc.n_blocks == 2
    assert doc.meta["resolution"] == 4 and doc.meta["foldovers"][0]["kind"] == "full"
    coded = doc.coded_matrix()
    assert np.allclose(coded[8:], -coded[:8])
    heads = [w.pages["design"].model.headerData(c, Qt.Orientation.Horizontal)
             for c in range(w.pages["design"].model.columnCount())]
    assert "Block" in heads
    n = doc.foldover("semifold", factor=0, keep_factor=1, keep_level=1)
    assert n == 8 and doc.meta["aliases"] == []  # half of the 16 runs, irregular now
    doc.modified = False
    split = DesignDocument.from_spec({"design": "factorial", "factors": [
        {"name": "a", "changes": "hard"}, {"name": "B"}, {"name": "C"}]})
    assert not split.two_level()
    with pytest.raises(ValueError):
        split.foldover()


# ---------------------------------------------------------------------------- compatibility
def test_old_rsx_files_still_load(tmp_path):
    """A design saved by 0.6.0 (no Group column, no changes / type flags, an analysis spec
    without the Auto Select fields) opens and analyses as before."""
    from rsstatchem.ui.dx.document import DesignDocument
    from rsstatchem.ui.dx.examples import chemical_process

    doc = chemical_process()
    d = doc.to_dict()
    d["design"]["analysis"] = {doc.response_cols[0]: {
        "model": "quadratic", "terms": None, "reduction": "backward", "alpha_out": 0.1,
        "hierarchy": True, "transform": {"kind": "none", "k": 0.0, "lam": 1.0},
        "factors": None}}
    for f in d["design"]["factors"]:
        f.pop("changes", None)
    path = tmp_path / "old.rsx"
    path.write_text(json.dumps(d))
    back = DesignDocument.load(str(path))
    assert not back.split_plot and back.group_col is None
    a = back.analyze(back.response_cols[0])
    assert type(a.fit) is model.Fit and (0, 1) not in a.fit.terms  # backward removed AB
    assert a.steps and a.steps[0][0] == "remove"


def test_tutorial_examples_open(ctx):
    from rsstatchem.ui.dx.examples import EXAMPLES
    from rsstatchem.ui.dx.window import DXMainWindow

    w = DXMainWindow(ctx, None)
    ctx.register_window(w)
    for key in ("battery", "pulp"):
        win = w.open_example(key)
        assert win.doc is not None and win.doc.response_count(win.doc.response_cols[0]) == 36
        win.doc.modified = False
        assert key in EXAMPLES

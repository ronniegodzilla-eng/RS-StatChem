"""RSM / DOE workspace after 0.7.0: Min-Run designs up to 50 factors in the wizard, split-plot
CCD with categoric factors, split-plot factorials blocked by confounding, the Effects tab with
whole-plot and subplot half-normal plots, and profile-likelihood variance intervals."""
from types import SimpleNamespace

import numpy as np
import pytest

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


def split_factorial_doc():
    from rsstatchem.ui.dx.document import DesignDocument

    doc = DesignDocument.from_spec({
        "design": "factorial", "seed": 4, "name": "Split-plot 2^5",
        "options": {"groups": 8, "replicates": 2},
        "factors": [{"name": "Temp", "changes": "hard"}, {"name": "Time", "changes": "hard"},
                    {"name": "Conc"}, {"name": "Stir"}, {"name": "pH"}],
        "responses": [{"name": "Yield"}]})
    df = doc.ds.df
    rng = np.random.default_rng(1)
    cols = doc.factor_cols
    g = df["Group"].to_numpy(int)
    s = {c: np.sign(df[c].to_numpy(float)) for c in cols}
    y = (60 + 5 * s[cols[0]] + 3 * s[cols[0]] * s[cols[1]] + 4 * s[cols[2]]
         + 2.5 * s[cols[2]] * s[cols[3]] + rng.normal(0, 2.5, g.max() + 1)[g - 1]
         + rng.normal(0, 0.8, len(g)))
    doc.set_values([(i, doc.response_cols[0], float(v)) for i, v in enumerate(y)])
    doc.modified = False
    return doc


# ---------------------------------------------------------------------------- wizard
def test_wizard_min_run_up_to_fifty_and_split_plot_blocks(qapp):
    from rsstatchem.ui.dx.wizard import NewDesignWizard

    wiz = NewDesignWizard(None, "factorial", "mr5")
    assert wiz.k_spin.maximum() == 50
    wiz.k_spin.setValue(50)
    wiz._update_summary()
    assert "1276 runs" in wiz.summary.text()
    wiz = NewDesignWizard(None, "factorial", "mr4")
    assert wiz.k_spin.maximum() == 50
    wiz.k_spin.setValue(40)
    wiz.finish()
    assert wiz.document.n_runs == 82
    # split-plot two-level factorial: blocks stay available (confounded with whole plots)
    wiz = NewDesignWizard(None, "splitplot", "factorial")
    wiz.k_spin.setValue(4)
    wiz.set_option("groups", 4)
    wiz.set_option("blocks", 2)
    assert wiz._opt["blocks"].isEnabled()
    spec = wiz.spec()
    assert spec["options"]["blocks"] == 2 and spec["options"]["groups"] == 4
    wiz.finish()
    doc = wiz.document
    assert doc.split_plot and doc.n_blocks == 2 and doc.meta["block_generators"] == ["ABCD"]


def test_wizard_split_plot_ccd_with_categoric(qapp):
    from rsstatchem.ui.dx.wizard import NewDesignWizard

    wiz = NewDesignWizard(None, "splitplot", "ccd")
    wiz.set_factors([{"name": "Gap", "low": 1, "high": 3, "changes": "hard"},
                     {"name": "Angle", "low": 10, "high": 20},
                     {"name": "Foil", "type": "categoric", "levels": "Alu, Steel",
                      "changes": "hard"}])
    wiz.set_option("center_points", 1)
    wiz.finish()
    doc = wiz.document
    assert doc.split_plot and doc.n_runs == 22 and doc.meta["split_plot"]["groups"] == 10
    assert doc.factors[2]["type"] == "categoric" and doc.factors[2]["changes"] == "hard"


# ---------------------------------------------------------------------------- effects tab
def test_split_plot_effects_tab(ctx, qapp):
    doc = split_factorial_doc()
    w = make_window(ctx, doc)
    col = doc.response_cols[0]
    view = w.show_node(f"analysis:{col}")
    from rsstatchem.ui.dx.analysis import EffectsPage

    assert view.effects_mode and isinstance(view.fit_page, EffectsPage)
    assert view.strip.buttons[1].text() == "Effects"
    view.set_tab("Effects")
    ep = view.fit_page
    ep.clear()
    assert doc.analyze(col).fit.terms == []
    ep.set_kind("whole")
    names = [e["name"] for e in ep.rows]
    assert sorted(names) == ["A", "AB", "ABCDE", "ACDE", "B", "BCDE", "CDE"]
    assert ep.graph.figure is not None and len(ep.graph.figure.effect_points) == 7
    # clicking AB on the plot adds it with its parents A and B (hierarchy)
    fig = ep.graph.figure
    ax = fig.axes[0]
    x, y, _n = next(p for p in fig.effect_points if p[2] == "AB")
    px, py = ax.transData.transform((x, y))
    ep._pick(SimpleNamespace(button=1, inaxes=ax, x=px, y=py))
    assert set(doc.analyze(col).fit.terms) == {(0,), (1,), (0, 1)}
    # subplot effects: select C and CD from the list
    ep.set_kind("sub")
    assert ep.buttons["sub"].isChecked() and len(ep.rows) == 24
    for name in ("C", "CD"):
        row = next(r for r, e in enumerate(ep.rows) if e["name"] == name)
        ep._clicked(row, 0)
    fit = doc.analyze(col).fit
    assert set(fit.terms) == {(0,), (1,), (0, 1), (2,), (3,), (2, 3)}
    marks = {ep.table.item(r, 1).text(): ep.table.item(r, 0).text()
             for r in range(ep.table.rowCount())}
    assert marks["C"] == "M" and marks["BE"] == "e"
    # clicking a selected effect returns it to error
    ep.toggle_name("CD")
    assert (2, 3) not in doc.analyze(col).fit.terms
    ep.toggle_name("CD")
    view.set_tab("ANOVA (REML)")
    tables = {t.title: t for t in view.anova_page.report.tables()}
    reml = next(t for k, t in tables.items() if k.startswith("ANOVA (REML)"))
    rows = {r[0].strip(): r for r in reml.rows}
    assert rows["Whole-plot"][2] == "12.00"   # 16 groups - 1 - 3 whole-plot terms
    assert rows["A-Temp"][4] == "< 0.0001" and rows["CD"][4] == "< 0.0001"
    vc = tables["Variance Components"]
    assert float(vc.rows[0][3]) >= 0 and any("profile-likelihood" in f for f in vc.footnotes)
    doc.modified = False


def test_ordinary_split_plot_keeps_fit_summary(ctx):
    from rsstatchem.ui.dx.analysis import FitSummaryPage
    from rsstatchem.ui.dx.examples import paper_pulp

    w = make_window(ctx, paper_pulp())
    view = w.show_node(f"analysis:{w.doc.response_cols[0]}")
    assert not view.effects_mode and isinstance(view.fit_page, FitSummaryPage)
    view.set_tab("Fit Summary")
    texts = " ".join(getattr(i, "text", "") for i in view.fit_page.report.items)
    assert "Stat-Ease documents no split-plot Fit Summary" in texts
    w.doc.modified = False


def test_blocked_split_plot_summary_and_rsx(ctx, tmp_path):
    from rsstatchem.ui.dx.document import DesignDocument

    doc = DesignDocument.from_spec({"design": "factorial", "seed": 4,
                                    "options": {"groups": 4, "blocks": 2},
                                    "factors": [{"name": "a", "changes": "hard"}, {"name": "B"},
                                                {"name": "C"}, {"name": "D"}]})
    w = make_window(ctx, doc)
    w.show_node("summary")
    text = " ".join(" ".join(map(str, r)) for t in w.pages["summary"].report.tables()
                    for r in t.rows)
    assert "ABCD (confounded with whole-plot interactions)" in text
    path = tmp_path / "blocked.rsx"
    doc.save(str(path))
    back = DesignDocument.load(str(path))
    assert back.n_blocks == 2 and back.split_plot
    assert back.meta["block_generators"] == ["ABCD"]
    assert np.array_equal(back.ds.df["Block"], doc.ds.df["Block"])
    sp = back.space()
    assert sp.split_plot and sp.block_col == "Block"
    doc.modified = False


def test_large_min_run_evaluation_is_fast():
    """Evaluating a large MR5 (466 runs here, 1276 for 50 factors) used to take minutes (alias
    checks term by term, an unoptimised einsum); now it is a second of linear algebra."""
    import time

    from rsstatchem.core.doe import designs, evaluation

    rng = np.random.default_rng(0)
    D = designs.min_run_res5(30, rng)
    sp = model.Space([model.FactorDef(f"X{i}", f"X{i}", index=i) for i in range(30)])
    t = time.perf_counter()
    ev = evaluation.evaluate(sp, D, "2fi")
    assert not ev["aliased"] and ev["p"] == 466 and ev["df"]["Residuals"] == 0
    evaluation.fds_curve(sp, ev, n_points=500)
    assert time.perf_counter() - t < 20


def test_split_plot_ccd_categoric_graphs_and_optimisation(ctx, qapp):
    from PySide6.QtWidgets import QComboBox

    from rsstatchem.ui.dx.document import DesignDocument

    doc = DesignDocument.from_spec({
        "design": "ccd", "seed": 2, "options": {"center_points": 2, "alpha_type": "practical"},
        "factors": [{"name": "Gap", "low": 1, "high": 3, "changes": "hard"},
                    {"name": "Angle", "low": 10, "high": 20},
                    {"name": "Foil", "type": "categoric", "levels": ["Alu", "Steel"],
                     "changes": "hard"}],
        "responses": [{"name": "Lift"}]})
    df = doc.ds.df
    rng = np.random.default_rng(3)
    g = df["Group"].to_numpy(int)
    gap, ang, foil = doc.factor_cols
    x1, x2 = df[gap].to_numpy(float) - 2, (df[ang].to_numpy(float) - 15) / 5
    steel = df[foil].to_numpy(int) == 2
    y = (100 + 8 * x1 + 5 * x2 - 4 * x1 ** 2 - 3 * x2 ** 2 + 2 * x1 * x2 + 6 * steel
         + rng.normal(0, 2, g.max() + 1)[g - 1] + rng.normal(0, 1, len(g)))
    col = doc.response_cols[0]
    doc.set_values([(i, col, float(v)) for i, v in enumerate(y)])
    w = make_window(ctx, doc)
    view = w.show_node(f"analysis:{col}")
    view.set_tab("Model")
    view.model_page.order.setCurrentIndex(view.model_page.order.findData("quadratic"))
    fit = doc.analyze(col).fit
    assert isinstance(fit, model.SplitPlotFit)
    assert {r[1] for r in fit.anova()["whole"]} >= {"A-Gap", "C-Foil"}
    view.set_tab("Model Graphs")
    page = view.graph_page
    assert page.tool.axes() == [0, 1]  # numeric factors on the axes
    combo = page.tool.edits[2]
    assert isinstance(combo, QComboBox) and combo.count() == 2
    combo.setCurrentIndex(1)  # Foil = Steel
    page.render()
    assert page.graph.has_graph and page.tool.point()[2] == 2.0
    n = w.show_node("numerical")
    n.set_goal(col, "maximize", float(y.min()), float(y.max()))
    sols = n.solve()
    assert sols[0]["x"][2] == 2.0  # optimisation over the categoric levels
    doc.modified = False

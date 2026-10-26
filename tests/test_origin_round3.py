"""Graph & Analysis module, third round: Two-Way ANOVA with raw data and Type I/II/III sums
of squares, Dunnett / Holm-Bonferroni / Holm-Sidak means comparisons, the Chen-Shapiro
normality test, live graphs in reports, the roll-up range picker, Recalculate for Statistics
on Rows / Frequency Counts and Rotate90 coordinate remapping. Numbers are cross-checked
against scipy, statsmodels, independent least-squares sums of squares and the published
Dunnett (1955) example."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from rsstatchem.core.analysis import matrix as mx
from rsstatchem.ui.origin import recalc
from rsstatchem.ui.origin import stat_tests as st
from rsstatchem.ui.origin.model import (Project, ReportSheet, SheetOp, Workbook, Worksheet,
                                        load_project, save_project)


@pytest.fixture
def ctx(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("RSSTATCHEM_HOME", str(tmp_path))
    from rsstatchem.app import AppContext

    context = AppContext(qapp)
    monkeypatch.setattr(context.app, "quit", lambda: None)
    return context


def _no_prompts(monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    def boom(*args, **kwargs):
        raise AssertionError("unexpected modal dialog")
    for name in ("question", "warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, boom)
    monkeypatch.setattr(QMessageBox, "exec", boom)


def _sheet_book(columns: list[list], name: str = "Book1") -> Workbook:
    book = Workbook.new(name, n_cols=len(columns))
    ws = book.sheets[0]
    ws.set_rows(max(len(c) for c in columns))
    for j, col in enumerate(columns):
        ws.columns[j].designation = "Y"
        ws.set_column_values(j, list(col))
    return book


def _value(report: ReportSheet, table: str, row: str, col: int) -> float:
    t = report.table(table)
    r = next(r for r in t["rows"] if r[0] == row)
    return float(r[col])


# ====================================================================== matrix Rotate90
def test_rotate90_remaps_every_value_coordinate():
    data = np.arange(12.0).reshape(3, 4)
    mapping = (0.0, 3.0, 10.0, 30.0)
    X, Y = mx.cell_coordinates(data.shape, mapping)
    cx, cy = 1.5, 20.0
    for angle in (90, 180, 270):
        r, rm = mx.rotate(data, mapping, angle)
        X2, Y2 = mx.cell_coordinates(r.shape, rm)
        for (i, j), v in np.ndenumerate(data):
            x, y = X[i, j], Y[i, j]
            if angle == 90:
                want = (cx + (y - cy), cy - (x - cx))
            elif angle == 270:
                want = (cx - (y - cy), cy + (x - cx))
            else:
                want = (2 * cx - x, 2 * cy - y)
            k = tuple(np.argwhere(r == v)[0])
            assert (X2[k], Y2[k]) == pytest.approx(want)
    # spans swap, the centre stays, and four turns give the original back
    r, rm = mx.rotate(data, mapping, 90)
    assert rm == (-8.5, 11.5, 18.5, 21.5)
    assert (rm[0] + rm[1]) / 2 == cx and (rm[2] + rm[3]) / 2 == cy
    d, m = data, mapping
    for _ in range(4):
        d, m = mx.rotate(d, m, 90)
    assert np.array_equal(d, data) and m == mapping
    # cw90 undoes ccw90
    d, m = mx.rotate(*mx.rotate(data, mapping, 90), 270)
    assert np.array_equal(d, data) and m == mapping


# ====================================================================== Two-Way ANOVA
def _lstsq_rss(X: np.ndarray, y: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(np.sum((y - X @ beta) ** 2))


def _coded(levels: list[str], values: list[str], kind: str) -> np.ndarray:
    """Columns of a factor: dummy (treatment) or sum-to-zero coding."""
    k = len(levels)
    out = np.zeros((len(values), k - 1))
    for r, v in enumerate(values):
        i = levels.index(v)
        if i < k - 1:
            out[r, i] = 1.0
        elif kind == "sum":
            out[r, :] = -1.0
    return out


def _independent_ss(a: list[str], b: list[str], y: np.ndarray, la, lb) -> dict:
    """Type I / II / III sums of squares from plain least squares (no statsmodels)."""
    one = np.ones((len(y), 1))
    out = {}
    for kind in ("dummy", "sum"):
        A, B = _coded(la, a, kind), _coded(lb, b, kind)
        AB = np.column_stack([A[:, i] * B[:, j] for i in range(A.shape[1])
                              for j in range(B.shape[1])])
        out[kind] = (one, A, B, AB)
    one, A, B, AB = out["dummy"]
    rss = {"1": _lstsq_rss(one, y), "A": _lstsq_rss(np.hstack([one, A]), y),
           "B": _lstsq_rss(np.hstack([one, B]), y),
           "AB": _lstsq_rss(np.hstack([one, A, B]), y),
           "full": _lstsq_rss(np.hstack([one, A, B, AB]), y)}
    one, A, B, AB = out["sum"]
    full = rss["full"]
    return {
        "1": {"A": rss["1"] - rss["A"], "B": rss["A"] - rss["AB"], "AB": rss["AB"] - full},
        "2": {"A": rss["B"] - rss["AB"], "B": rss["A"] - rss["AB"], "AB": rss["AB"] - full},
        "3": {"A": _lstsq_rss(np.hstack([one, B, AB]), y) - full,
              "B": _lstsq_rss(np.hstack([one, A, AB]), y) - full, "AB": rss["AB"] - full},
        "error": full}


def _unbalanced_raw():
    """Origin's raw layout: one column per combination, Factor A outermost
    (Male SV, Male LV, Male Nor, Female SV, Female LV, Female Nor), unequal lengths."""
    rng = np.random.default_rng(11)
    sizes = [5, 4, 6, 3, 5, 4]
    shift = [0.0, 1.5, -1.0, 2.0, 3.5, 0.5]
    return [list(np.round(120 + s + rng.normal(0, 2.0, n), 2)) for n, s in zip(sizes, shift)]


@pytest.mark.parametrize("ss", ["1", "2", "3"])
def test_two_way_anova_raw_unbalanced_matches_least_squares(ss):
    cols = _unbalanced_raw()
    book = _sheet_book(cols)
    books = {"Book1": book}
    ranges = [f"[Book1]Sheet1!{c.name}" for c in book.sheets[0].columns]
    settings = {"form": "raw", "ranges": ranges, "name_a": "Sex", "levels_a": 2,
                "names_a": "Male, Female", "name_b": "Diet", "levels_b": 3,
                "names_b": "SV, LV, Nor", "ss": ss, "means": ["tukey"], "alpha": 0.05}
    rep, text = st.two_way_anova(books, settings)
    la, lb = ["Male", "Female"], ["SV", "LV", "Nor"]
    a, b, y = [], [], []
    for k, col in enumerate(cols):
        a += [la[k // 3]] * len(col)
        b += [lb[k % 3]] * len(col)
        y += col
    ref = _independent_ss(a, b, np.array(y), la, lb)
    assert _value(rep, "Overall ANOVA", "Sex", 2) == pytest.approx(ref[ss]["A"], rel=1e-4)
    assert _value(rep, "Overall ANOVA", "Diet", 2) == pytest.approx(ref[ss]["B"], rel=1e-4)
    assert _value(rep, "Overall ANOVA", "Sex * Diet", 2) == \
        pytest.approx(ref[ss]["AB"], rel=1e-4)
    assert _value(rep, "Overall ANOVA", "Error", 2) == pytest.approx(ref["error"], rel=1e-4)
    notes = dict((r[0], r[1]) for r in rep.table("Notes")["rows"])
    assert notes["Data Form"] == "Raw" and notes["Design"] == "Unbalanced"
    assert notes["Sums of Squares"] == st.SS_TYPES[ss]
    # descriptive statistics follow the raw layout (Female LV = 5th column)
    desc = rep.table("Descriptive Statistics")
    row = next(r for r in desc["rows"] if r[0] == "Female  LV")
    assert int(row[1]) == len(cols[4]) and float(row[2]) == pytest.approx(np.mean(cols[4]),
                                                                           rel=1e-5)
    assert rep.table("Tukey Test (Sex)") is not None
    assert any(i["kind"] == "graph" and i["title"] == "Interaction Plot" for i in rep.items)
    assert st.SS_TYPES[ss] in text


def test_two_way_anova_raw_equals_indexed_and_balanced_types_agree():
    rng = np.random.default_rng(3)
    cols = [list(np.round(10 + k + rng.normal(0, 1, 4), 3)) for k in range(4)]
    book = _sheet_book(cols)
    ws = book.sheets[0]
    # the same data, indexed, in a second sheet
    idx = Worksheet.new("Index", 3, 16)
    fa, fb, yy = [], [], []
    for k, col in enumerate(cols):
        fa += [["lo", "hi"][k // 2]] * 4
        fb += [["x", "y"][k % 2]] * 4
        yy += col
    idx.set_column_values(0, fa)
    idx.set_column_values(1, fb)
    idx.set_column_values(2, yy)
    book.sheets.append(idx)
    books = {"Book1": book}
    raw = {"form": "raw", "ranges": [f"[Book1]Sheet1!{c.name}" for c in ws.columns],
           "levels_a": 2, "levels_b": 2, "names_a": "lo, hi", "names_b": "x, y",
           "name_a": "A", "name_b": "B", "means": []}
    indexed = {"form": "indexed", "data": "[Book1]Index!C", "factor_a": "[Book1]Index!A",
               "factor_b": "[Book1]Index!B", "means": []}
    r1, _ = st.two_way_anova(books, raw)
    r2, _ = st.two_way_anova(books, indexed)
    # report cells hold rounded text (6 significant digits), so the last digit may differ
    # between platforms' floating-point summation orders
    for row in ("Error", "Corrected Total"):
        assert _value(r1, "Overall ANOVA", row, 2) == pytest.approx(
            _value(r2, "Overall ANOVA", row, 2), rel=1e-5)
    ss = [st.two_way_anova(books, {**raw, "ss": t})[0] for t in ("1", "2", "3")]
    for term in ("A", "B", "A * B"):
        vals = [_value(r, "Overall ANOVA", term, 2) for r in ss]
        assert vals[0] == pytest.approx(vals[1], rel=1e-5) == pytest.approx(vals[2], rel=1e-5)
    assert dict((r[0], r[1]) for r in ss[0].table("Notes")["rows"])["Design"] == "Balanced"
    # wrong number of raw columns / missing cells are reported
    with pytest.raises(st.StatError, match="2 × 3"):
        st.two_way_anova(books, {**raw, "levels_b": 3})
    with pytest.raises(st.StatError, match="different"):
        st.level_names("a, a", 2, "A")


# ====================================================================== means comparisons
DUNNETT_CONTROL = [7.40, 8.50, 7.20, 8.24, 9.84, 8.32]   # Dunnett (1955), scipy docs
DUNNETT_A = [9.76, 8.80, 7.68, 9.36]
DUNNETT_B = [12.80, 9.68, 12.16, 9.20, 10.55]


def _pooled(groups):
    ss = sum(((np.asarray(v) - np.mean(v)) ** 2).sum() for _, v in groups)
    df = sum(len(v) for _, v in groups) - len(groups)
    return ss / df, df


def test_dunnett_matches_published_example_and_scipy():
    groups = [("Control", np.array(DUNNETT_CONTROL)), ("Drug A", np.array(DUNNETT_A)),
              ("Drug B", np.array(DUNNETT_B))]
    mse, df = _pooled(groups)
    vals = st.dunnett_values(groups, 0, mse, df, 0.05)
    ref = stats.dunnett(np.array(DUNNETT_A), np.array(DUNNETT_B),
                        control=np.array(DUNNETT_CONTROL), rng=1)
    # the published p-values (scipy.stats.dunnett documentation)
    assert [v[4] for v in vals] == pytest.approx([0.62004941, 0.0059035], abs=2e-3)
    assert [v[4] for v in vals] == pytest.approx(list(ref.pvalue), abs=2e-3)
    assert [v[3] for v in vals] == pytest.approx(list(ref.statistic), rel=1e-9)
    ci = ref.confidence_interval(0.95)
    assert [v[5] for v in vals] == pytest.approx(list(ci.low), abs=2e-2)
    assert [v[6] for v in vals] == pytest.approx(list(ci.high), abs=2e-2)
    # through the One-Way ANOVA report, with a named control level
    book = _sheet_book([DUNNETT_A, DUNNETT_CONTROL, DUNNETT_B])
    for c, name in zip(book.sheets[0].columns, ("DrugA", "Control", "DrugB")):
        c.long_name = name
    books = {"Book1": book}
    rep, _ = st.one_way_anova(books, {"ranges": ["[Book1]Sheet1!A", "[Book1]Sheet1!B",
                                                 "[Book1]Sheet1!C"],
                                      "means": ["dunnett"], "control": "Control"})
    t = rep.table("Dunnett Test")
    assert [r[0] for r in t["rows"]] == ["DrugA  Control", "DrugB  Control"]
    assert float(t["rows"][1][4]) == pytest.approx(0.0059, abs=2e-3)
    assert t["rows"][1][6] == "1" and t["rows"][0][6] == "0"
    with pytest.raises(st.StatError, match="control"):
        st.one_way_anova(books, {"ranges": ["[Book1]Sheet1!A", "[Book1]Sheet1!B"],
                                 "means": ["dunnett"], "control": "Nope"})


@pytest.mark.parametrize("method,sm", [("holm_bonferroni", "holm"),
                                       ("holm_sidak", "holm-sidak")])
def test_holm_procedures_match_statsmodels(method, sm):
    from statsmodels.stats.multitest import multipletests

    rng = np.random.default_rng(5)
    groups = [(f"G{k}", rng.normal(10 + 0.6 * k, 1.0, 8)) for k in range(5)]
    mse, df = _pooled(groups)
    rows = st.holm_rows(groups, method, mse, df, 0.05)
    p = []
    for i in range(len(groups)):
        for j in range(i):
            d = groups[i][1].mean() - groups[j][1].mean()
            se = np.sqrt(mse * (1 / 8 + 1 / 8))
            p.append(2 * stats.t.sf(abs(d / se), df))
    reject, adj, _, _ = multipletests(p, alpha=0.05, method=sm)
    assert [float(r[4]) for r in rows] == pytest.approx(p, rel=1e-3, abs=1e-6)
    assert [r[6] == "1" for r in rows] == list(reject)
    assert any(reject) and not all(reject)
    # Alpha is the step-down level of each comparison's rank
    order = np.argsort(p, kind="stable")
    m = len(p)
    for rank, k in enumerate(order):
        want = 0.05 / (m - rank) if method == "holm_bonferroni" else \
            1 - 0.95 ** (1 / (m - rank))
        assert float(rows[k][5]) == pytest.approx(want, rel=1e-3)
    assert st.MEANS_METHODS["sidak"] == "Dunn-Sidak"


# ====================================================================== Chen-Shapiro
def test_chen_shapiro_statistic_and_monte_carlo_p():
    x = np.array([2.1, 3.4, 1.9, 5.6, 4.4, 3.3, 2.8, 4.0, 3.7, 3.1])
    n = len(x)
    xs = np.sort(x)
    h = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    qh = np.sum(np.diff(xs) / np.diff(h)) / ((n - 1) * xs.std(ddof=1))
    assert st.chen_shapiro_statistic(x) == pytest.approx(np.sqrt(n) * (1 - qh))
    # location / scale invariant
    assert st.chen_shapiro_statistic(5 * x - 3) == pytest.approx(st.chen_shapiro_statistic(x))
    # under H0 the test rejects about alpha of normal samples; skewed data are rejected
    rng = np.random.default_rng(42)
    p = [st.chen_shapiro(rng.normal(3, 2, 30))[1] for _ in range(300)]
    rate = np.mean(np.array(p) < 0.05)
    assert 0.02 <= rate <= 0.09
    _, p_exp = st.chen_shapiro(rng.exponential(1.0, 60))
    assert p_exp < 0.01
    crit = st.chen_shapiro_critical(30, 0.05)
    s, pp = st.chen_shapiro(rng.normal(size=30))
    assert (s > crit) == (pp < 0.05) or abs(pp - 0.05) < 0.01
    # like Shapiro-Wilk, it agrees on clearly normal / clearly skewed data
    v = rng.lognormal(0, 1, 80)
    assert st.chen_shapiro(v)[1] < 0.05 and stats.shapiro(v).pvalue < 0.05
    book = _sheet_book([list(np.round(rng.normal(5, 1, 25), 3))])
    rep, _ = st.normality({"Book1": book}, {"ranges": ["[Book1]Sheet1!A"],
                                            "tests": ["shapiro", "chen_shapiro"]})
    t = rep.table("Chen-Shapiro")
    assert t is not None and any("Monte Carlo" in n for n in t["notes"])


# ====================================================================== live report graphs
def test_report_graphs_are_live_models_redrawn_on_recalculate(qapp):
    from rsstatchem.core.graph.model import graph_from_json_dict
    from rsstatchem.ui.origin.report_view import LiveGraph, ReportView

    book = _sheet_book([[1, 2, 3, 4, 5, 6], [2, 3, 3, 5, 6, 8], [3, 3, 4, 4, 9, 9]])
    project = Project()
    project.windows["Book1"] = book
    books = project.data_windows()
    rep, _ = st.one_way_anova(books, {"ranges": ["[Book1]Sheet1!A", "[Book1]Sheet1!B",
                                                 "[Book1]Sheet1!C"], "means_plot": True})
    item = next(i for i in rep.items if i["title"] == "Means Plot")
    assert item["kind"] == "graph" and "png" not in item
    g = graph_from_json_dict(item["graph"])
    assert g.layers[0].plots[0].type == "line_symbol"
    assert g.values(g.layers[0].plots[0].y)[0] == pytest.approx(3.5)
    view = ReportView(rep)
    assert len(view.graphs) == 1 and isinstance(view.graphs[0], LiveGraph)
    assert not view.graphs[0].rmap.errors
    view.resize(500, 300)
    view.show()
    qapp.processEvents()
    qapp.processEvents()
    # wrapped notes must not let the scroll area squeeze the page (graphs would overlap)
    assert view.widget().height() >= view.widget().sizeHint().height()
    view.expand_all(False)
    qapp.processEvents()
    qapp.processEvents()
    assert view.widget().height() < view.widget().sizeHint().height() + 400
    view.expand_all(True)
    recalc.stamp(project, rep)
    book.sheets[0].set_cell(0, 0, 100.0)   # the source data change
    assert recalc.stale_reports(project) == [(book, rep)]
    recalc.recalculate(project, book, rep)
    view.refresh()
    g2 = view.graphs[0].graph
    assert g2.values(g2.layers[0].plots[0].y)[0] == pytest.approx((100 + 2 + 3 + 4 + 5 + 6) / 6)
    # copied as a picture
    from rsstatchem.ui.origin.report_view import item_png, report_html
    assert item_png(rep.graphs()[0])[:4] == b"\x89PNG"
    assert "data:image/png;base64," in report_html(rep)


def test_linked_report_graph_window_follows_recalculation(ctx, monkeypatch):
    _no_prompts(monkeypatch)
    from rsstatchem.ui.origin import open_workspace
    from rsstatchem.ui.origin.sheet_view import HEADER_ROWS

    win = open_workspace(ctx)
    bw = win.active_book_window()
    rows = [["A", "B"]] + [[f"{v}", f"{v * 1.5 + 2}"] for v in range(1, 11)]
    bw.paste_block(rows, HEADER_ROWS, 0)
    rep = win.run_stat("anova1", {"ranges": ["[Book1]Sheet1!A", "[Book1]Sheet1!B"],
                                  "means": ["holm_bonferroni"], "mode": "manual"})
    assert rep.table("Holm-Bonferroni Test") is not None
    item = next(i for i in rep.items if i["title"] == "Means Plot")
    gwin = win.open_report_graph(bw.book, rep, item)
    assert gwin is not None and gwin.graph.meta["report_link"]["report"] == rep.name
    y0 = gwin.graph.values(gwin.graph.layers[0].plots[0].y)[0]
    gwin.graph.layers[0].plots[0].line.width = 3.0   # the user's formatting is kept
    bw.tabs.setCurrentIndex(0)
    model = bw.sheet_view().model_
    model.setData(model.index(HEADER_ROWS, 0), "101")
    assert rep.dirty
    assert win.recalculate_report(bw.book, rep)
    y1 = gwin.graph.values(gwin.graph.layers[0].plots[0].y)[0]
    assert y1 == pytest.approx(y0 + 10.0) and gwin.graph.layers[0].plots[0].line.width == 3.0
    # a fit's Fitted Curves Plot still opens linked to the worksheet data
    from rsstatchem.ui.origin import analysis as an
    out = an.run_fit(an.xy_data(bw.book, bw.book.sheets[0], 1), "Line", "linear")
    fit_item = next(i for i in out.report.items if i["title"] == "Fitted Curves Plot")
    assert fit_item["kind"] == "graph" and fit_item["open"]["kind"] == "fit"
    g = win.open_report_graph(bw.book, out.report, fit_item)
    assert any("Sheet1" in k for k in g.graph.data)
    win.modified = False
    win.close()


# ====================================================================== range picker
def test_range_picker_rolls_up_for_drag_selection(ctx, monkeypatch):
    _no_prompts(monkeypatch)
    from PySide6.QtCore import Qt, QTimer

    from rsstatchem.ui.origin import open_workspace
    from rsstatchem.ui.origin import stat_dialogs as sd
    from rsstatchem.ui.origin.sheet_view import HEADER_ROWS

    win = open_workspace(ctx)
    bw = win.active_book_window()
    rows = [["X", "Y", "Z"]] + [[f"{v}", f"{v * 2}", f"{v * 3}"] for v in range(1, 13)]
    bw.paste_block(rows, HEADER_ROWS, 0)
    bw.select_columns([0])
    dlg = sd.make_dialog("ttest_pair", win, None, win.selection_ranges())
    seen = {}

    def step1():
        assert dlg.isVisible() and dlg.windowModality() == Qt.WindowModality.ApplicationModal
        dlg.widget("range2").button.click()            # ▸: roll up
        seen["rolled"] = (dlg.picking is not None, dlg.body.isVisible(),
                          dlg.pick_bar.isVisible(), dlg.windowModality())
        bw.select_block(3, 1, 9, 1)                     # drag over B3:B9 in the sheet
        dlg._show_picked()
        seen["bar"] = dlg.pick_bar.edit.text()
        dlg.pick_bar.back.click()                       # ▸: back to the dialog
        seen["back"] = (dlg.picking, dlg.body.isVisible(), dlg.windowModality())
        dlg.accept()

    QTimer.singleShot(0, step1)
    assert dlg.exec() == 1                              # hiding to roll up kept exec alive
    assert seen["rolled"] == (True, False, True, Qt.WindowModality.NonModal)
    assert seen["bar"] == "[Book1]Sheet1!B[3:9]"
    assert seen["back"] == (None, True, Qt.WindowModality.ApplicationModal)
    assert dlg.values()["range2"] == "[Book1]Sheet1!B[3:9]"
    # a list picker adds the selected columns; Esc while rolled up only restores
    dlg2 = sd.make_dialog("anova1", win, {"ranges": []}, [])
    lst = dlg2.widget("ranges")
    dlg2.show()
    lst.start_pick()
    bw.select_columns([1, 2])
    dlg2.reject()
    assert dlg2.picking is None and dlg2.isVisible() and lst.texts() == []
    lst.start_pick()
    dlg2.accept()                                       # Enter takes the selection
    assert lst.texts() == ["[Book1]Sheet1!B", "[Book1]Sheet1!C"] and dlg2.isVisible()
    dlg2.reject()
    assert not dlg2.isVisible()
    win.modified = False
    win.close()


# ====================================================================== recalc rows / freq
def test_statistics_on_rows_and_frequency_counts_recalculate(ctx, monkeypatch, tmp_path):
    _no_prompts(monkeypatch)
    from rsstatchem.ui.origin import open_workspace
    from rsstatchem.ui.origin import stat_dialogs as sd
    from rsstatchem.ui.origin.sheet_view import HEADER_ROWS

    win = open_workspace(ctx)
    bw = win.active_book_window()
    rows = [["A", "B"]] + [[f"{v}", f"{v + 2}"] for v in range(1, 9)]
    bw.paste_block(rows, HEADER_ROWS, 0)
    for key in ("rows", "freq"):    # both dialogs now have the Recalculate mode
        assert "mode" in sd.make_dialog(key, win, None, []).values()
    added = win.run_stat("rows", {"ranges": ["[Book1]Sheet1!A", "[Book1]Sheet1!B"],
                                  "quantities": ["mean", "sd"], "mode": "auto"})
    ws = bw.book.sheets[0]
    assert added == [2, 3] and ws.columns[2].numeric()[0] == pytest.approx(2.0)
    op = ws.op_of(2)
    assert op is ws.op_of(3) and op["kind"] == "rows" and op["hash"]
    assert ws.op_of(0) is None
    model = bw.sheet_view().model_
    from PySide6.QtCore import Qt
    assert model.headerData(2, Qt.Orientation.Horizontal,
                            Qt.ItemDataRole.DecorationRole) is not None
    assert model.headerData(0, Qt.Orientation.Horizontal,
                            Qt.ItemDataRole.DecorationRole) is None
    assert "Statistics on Rows" in model.headerData(2, Qt.Orientation.Horizontal,
                                                    Qt.ItemDataRole.ToolTipRole)
    # Auto: editing an input recalculates the output columns in place
    model.setData(model.index(HEADER_ROWS, 0), "11")
    assert len(ws.columns) == 4 and ws.columns[2].numeric()[0] == pytest.approx(7.0)
    # Manual: flagged (dirty lock), then Recalculate from the column's lock menu
    target = recalc.sheet_op(bw.book, ws, 2)
    win.set_report_mode(bw.book, target, "manual")
    model.setData(model.index(HEADER_ROWS, 0), "21")
    assert ws.op_of(2)["dirty"] and ws.columns[2].numeric()[0] == pytest.approx(7.0)
    bw.select_columns([2])
    menu = win.column_lock_menu(bw, 2)
    texts = [a.text() for a in menu.actions()]
    assert "&Recalculate" in texts and "&Change Parameters..." in texts
    next(a for a in menu.actions() if a.text() == "&Recalculate").trigger()
    assert not ws.op_of(2)["dirty"] and ws.columns[2].numeric()[0] == pytest.approx(12.0)
    # Change Parameters: fewer quantities remove the extra output column
    monkeypatch.setattr(sd.StatDialog, "exec", lambda self: True)
    real = sd.make_dialog

    def spy(key, main=None, settings=None, selection=None):
        d = real(key, main, settings, selection)
        if key == "rows":
            d.widget("quantities").boxes["sd"].setChecked(False)
            d.widget("quantities").boxes["max"].setChecked(True)
        return d
    monkeypatch.setattr(sd, "make_dialog", spy)
    win.change_report_parameters(bw.book, recalc.sheet_op(bw.book, ws, 2))
    assert len(ws.columns) == 4 and ws.columns[3].long_name == "Maximum"
    assert ws.op_of(3)["settings"]["quantities"] == ["mean", "max"]
    # Frequency Counts: its sheet is locked and refilled in place
    out = win.run_stat("freq", {"range": "[Book1]Sheet1!B", "auto": False, "from": 0.0,
                                "to": 12.0, "step": 4.0, "mode": "manual"})
    assert out.ops[0]["kind"] == "freq" and out.op_of(0) is out.ops[0]
    counts = out.columns[3].numeric()
    assert list(counts[:3]) == [1.0, 4.0, 3.0]
    bw.tabs.setCurrentIndex(0)
    model = bw.sheet_view().model_
    model.setData(model.index(HEADER_ROWS + 1, 1), "1")   # 4 -> 1
    assert out.ops[0]["dirty"]
    win.recalculate_report(bw.book, SheetOp(out, out.ops[0]))
    assert bw.book.sheet(out.name) is out and list(out.columns[3].numeric()[:3]) == \
        [2.0, 3.0, 3.0]
    assert not out.ops[0]["dirty"]
    # saved with the project
    path = tmp_path / "ops.rsop"
    save_project(win.project, path)
    p2, _ = load_project(path)
    b2 = p2.windows["Book1"]
    assert b2.sheets[0].op_of(2)["kind"] == "rows" and b2.sheet(out.name).ops[0]["kind"] == \
        "freq"
    assert len(recalc.reports(p2)) == 2
    win.modified = False
    win.close()


def test_matrix_rotate_command_in_window(ctx, monkeypatch):
    _no_prompts(monkeypatch)
    from rsstatchem.ui.origin import open_workspace

    win = open_workspace(ctx)
    mw = win.new_matrix()
    win.set_matrix_dimensions(mw, {"rows": 3, "cols": 4, "x1": 0.0, "x2": 3.0, "y1": 10.0,
                                   "y2": 30.0})
    win.set_matrix_values(mw, "x + 100*y")
    ms = mw.sheet()
    win.matrix_geometry("rot90")
    ms = mw.sheet()
    assert ms.data.shape == (4, 3) and (ms.x1, ms.x2, ms.y1, ms.y2) == (-8.5, 11.5, 18.5, 21.5)
    win.matrix_geometry("rot270")
    ms = mw.sheet()
    assert (ms.x1, ms.x2, ms.y1, ms.y2) == (0.0, 3.0, 10.0, 30.0)
    assert ms.data[0, 0] == pytest.approx(1000.0)
    win.modified = False
    win.close()

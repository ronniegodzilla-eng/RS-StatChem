"""WP-3 Statistics menu hook (``stats_menus.install``), dialogs and report sheets in the
Origin-style window (offscreen, session ``qapp``)."""
from __future__ import annotations

import numpy as np
import pytest

from rsstatchem.ui.origin import stat_tests as st


@pytest.fixture
def win(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("RSSTATCHEM_HOME", str(tmp_path))
    from PySide6.QtWidgets import QMessageBox

    from rsstatchem.app import AppContext
    from rsstatchem.ui.origin import open_workspace, stats_menus

    def boom(*args, **kwargs):
        raise AssertionError("unexpected modal dialog")
    for name in ("question", "warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, boom)
    context = AppContext(qapp)
    monkeypatch.setattr(context.app, "quit", lambda: None)
    w = open_workspace(context)
    stats_menus.install(w)
    return w


def _fill(win, columns):
    from rsstatchem.ui.origin.sheet_view import HEADER_ROWS
    bw = win.active_book_window()
    ws = bw.worksheet()
    while len(ws.columns) < len(columns):
        ws.add_column("Y")
    n = max(len(c) for c in columns)
    rows = [[("" if i >= len(c) else str(c[i])) for c in columns] for i in range(n)]
    bw.paste_block(rows, HEADER_ROWS, 0)
    return bw


def _menu_titles(menu):
    out = {}
    for act in menu.actions():
        if act.menu() is not None:
            out[act.text().replace("&", "")] = [a.text().replace("&", "")
                                                for a in act.menu().actions() if a.text()]
    return out


def test_install_adds_origin_statistics_menus(win):
    titles = _menu_titles(win.menus["statistics"])
    for sub in ("Descriptive Statistics", "Hypothesis Testing", "ANOVA",
                "Nonparametric Tests", "Multivariate Analysis", "Survival Analysis",
                "ROC Curve", "Power and Sample Size"):
        assert sub in titles, titles.keys()
    assert "Grubbs Test..." in titles["Descriptive Statistics"]
    assert "Normality Test..." in titles["Descriptive Statistics"]   # built-ins kept
    assert "Partial Least Squares..." in titles["Multivariate Analysis"]
    assert len(titles["Power and Sample Size"]) == 8
    assert len(win.wp3_actions) == 27 + 8


def test_every_dialog_builds_and_round_trips(win):
    from rsstatchem.ui.origin import stat_dialogs as sd
    st.load_extensions()
    for key in st.DIALOG_SPECS:
        dlg = sd.make_dialog(key, win, None, ["[Book1]Sheet1!A", "[Book1]Sheet1!B"])
        values = dlg.values()
        again = sd.make_dialog(key, win, values)
        assert again.values() == values, key
        dlg.deleteLater()
        again.deleteLater()


def test_power_dialog_visibility_by_procedure(win):
    from rsstatchem.ui.origin import stat_dialogs as sd
    dlg = sd.make_dialog("power", win, {"procedure": "var2"})
    dlg.show()
    vis = {k: w.isVisible() for k, w in dlg.widgets.items()}
    assert vis["ratio_var"] and vis["ratio_n"] and not vis["mean1"] and not vis["sd"]
    combo = dlg.widgets["procedure"]
    combo.setCurrentIndex(combo.findData("t2"))
    assert dlg.widgets["mean1"].isVisible() and dlg.widgets["sd"].isVisible()
    assert not dlg.widgets["ratio_var"].isVisible()
    dlg.close()


def test_run_from_menu_runner_with_recalculation(win):
    from rsstatchem.ui.origin import stats_menus
    v = [10.0, 10.1, 9.9, 10.05, 9.95, 10.02, 9.98, 10.08, 9.92, 13.0]
    bw = _fill(win, [v])
    rep = stats_menus.run(win, "grubbs", {"ranges": ["[Book1]Sheet1!A"], "mode": "auto"})
    assert rep is not None and rep.op["test"] == "grubbs" and rep.mode == "auto"
    assert rep.table("Grubbs Test")["rows"][0][-1] == "Outlier"
    from rsstatchem.ui.origin.sheet_view import HEADER_ROWS
    bw.tabs.setCurrentIndex(0)
    model = bw.sheet_view().model_
    model.setData(model.index(HEADER_ROWS + 9, 0), "10.1")
    assert rep.table("Grubbs Test")["rows"][0][-1] == "Not an outlier"   # Auto recalc


def test_no_input_analyses_go_to_the_active_book(win):
    from rsstatchem.ui.origin import stats_menus
    _fill(win, [[1, 2, 3]])
    rep = stats_menus.run(win, "power", {"procedure": "t1", "mean": 0.5, "null": 0, "sd": 1,
                                         "estimate": "power", "sizes": "10, 20",
                                         "mode": "manual"})
    book = win.active_book_window().book
    assert rep in book.sheets and rep.table("Power") is not None
    rep2 = stats_menus.run(win, "prop2", {"form": "summary", "x1": 10, "n1": 30, "x2": 20,
                                          "n2": 30, "mode": "manual"})
    assert rep2 in book.sheets
    # Recalculate (Change Parameters path) works for a report without input ranges
    assert win.recalculate_report(book, rep)


def test_open_dialog_cancel_and_ok(win, monkeypatch):
    from rsstatchem.ui.origin import stat_dialogs as sd
    from rsstatchem.ui.origin import stats_menus
    _fill(win, [[1.0, 2.0, 2.5, 3.0, 8.0], [2.0, 2.5, 3.1, 3.3, 2.9]])
    monkeypatch.setattr(sd.StatDialog, "exec", lambda self: 0)
    assert stats_menus.open_dialog(win, "dixon") is None
    monkeypatch.setattr(sd.StatDialog, "exec", lambda self: 1)
    rep = stats_menus.open_dialog(win, "var1")
    assert rep is not None and rep.op["test"] == "var1"


def test_report_graphs_render(win):
    from rsstatchem.ui.origin import stats_menus
    from rsstatchem.ui.origin.report_view import ReportView, item_png
    rng = np.random.default_rng(3)
    cols = [list(np.round(rng.normal(size=30), 4)) for _ in range(3)]
    bw = _fill(win, cols)
    rep = stats_menus.run(win, "pca", {"ranges": ["[Book1]Sheet1!A", "[Book1]Sheet1!B",
                                                  "[Book1]Sheet1!C"], "mode": "manual"})
    for item in rep.graphs():
        assert item_png(item)[:4] == b"\x89PNG"
    view = ReportView(rep, bw)
    view.resize(900, 700)
    view.show()
    view.close()

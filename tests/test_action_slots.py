"""Qt's ``checked`` flag must never arrive as a real argument of a menu slot.

``save_as(self, path=None)`` connected directly to ``triggered`` was called as
``save_as(False)``: File > Save As and File > Print of Chemical Drawing did nothing but raise.
"""
import subprocess
import sys
from pathlib import Path

import pytest
from PySide6.QtGui import QAction
from PySide6.QtPrintSupport import QPrintDialog
from PySide6.QtWidgets import QFileDialog

ROOT = Path(__file__).resolve().parent.parent


def test_connect_triggered_passes_the_flag_only_where_it_belongs(qapp):
    from rsstatchem.ui.actions import connect_triggered, flag_use

    calls = []

    def optional(path=None):
        calls.append(("optional", path))

    def required(on):
        calls.append(("required", on))

    def none():
        calls.append(("none",))

    assert [flag_use(f) for f in (optional, required, none)] == ["optional", "required",
                                                                  "none"]
    plain, check = QAction("plain"), QAction("check")
    check.setCheckable(True)
    for slot in (optional, required, none):
        connect_triggered(plain, slot)
        connect_triggered(check, slot)
    plain.trigger()
    assert calls == [("optional", None), ("required", False), ("none",)]
    calls.clear()
    check.trigger()                       # becomes checked: slots that can take it get True
    assert calls == [("optional", True), ("required", True), ("none",)]


def test_no_slot_takes_the_checked_flag_for_an_argument():
    """Static audit of every direct triggered/clicked connection in the UI code."""
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "slot_audit.py")],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout


@pytest.fixture
def ctx(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("RSSTATCHEM_HOME", str(tmp_path))
    from rsstatchem.app import AppContext

    context = AppContext(qapp)
    monkeypatch.setattr(context.app, "quit", lambda: None)
    return context


def test_chemdraw_save_as_and_print_from_the_menu(ctx, tmp_path, monkeypatch):
    from rsstatchem.ui import modules

    w = modules.open_module(ctx, "chem")
    w.load_text("CCO")
    target = tmp_path / "etanol.rsc"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(target), "")))
    w.act_save_as.trigger()                      # was save_as(False): TypeError, no file
    assert target.exists() and w.path == str(target)
    asked = []
    monkeypatch.setattr(QPrintDialog, "exec", lambda self: asked.append(1) or 0)
    w.act_print.trigger()                        # was print_document(False): no dialog at all
    assert asked == [1]
    w.undo_stack.setClean()


def test_origin_tab_menu_add_appends_the_sheet(ctx):
    from rsstatchem.ui import modules

    w = modules.open_module(ctx, "graph")
    bw = w.active_book_window()
    first = bw.book.sheets[0]
    ws = bw.add_sheet()                          # what the tab menu's "Add" now calls
    assert bw.book.sheets[0] is first and bw.book.sheets[-1] is ws

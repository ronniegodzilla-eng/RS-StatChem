"""Print / Page Setup dialogs are destroyed with their command: left behind as children of the
main window they kept a pointer to a deleted QPrinter and crashed when the window was hidden."""
import gc

import pytest
from PySide6.QtPrintSupport import QPageSetupDialog, QPrintDialog, QPrinter
from PySide6.QtWidgets import QDialog, QWidget


@pytest.fixture
def no_modal(monkeypatch):
    for cls in (QPageSetupDialog, QPrintDialog):
        monkeypatch.setattr(cls, "exec", lambda self: 0)


def test_exec_and_delete_destroys_the_dialog(qapp, no_modal):
    import shiboken6

    from rsstatchem.ui.qt_lifecycle import exec_and_delete

    parent = QWidget()
    printer = QPrinter()
    dlg = QPageSetupDialog(printer, parent)
    assert exec_and_delete(dlg) == QDialog.DialogCode.Rejected.value
    assert not shiboken6.isValid(dlg) and parent.findChildren(QPageSetupDialog) == []
    parent.deleteLater()


def test_chemdraw_page_setup_and_print_leave_no_dialog(qapp, tmp_path, monkeypatch, no_modal):
    monkeypatch.setenv("RSSTATCHEM_HOME", str(tmp_path))
    from rsstatchem.app import AppContext
    from rsstatchem.ui import modules

    ctx = AppContext(qapp)
    monkeypatch.setattr(ctx.app, "quit", lambda: None)
    w = modules.open_module(ctx, "chem")
    assert w.page_setup() is False and w.print_document() is False
    gc.collect()
    assert w.findChildren(QPageSetupDialog) == [] and w.findChildren(QPrintDialog) == []
    w.hide()                         # crashed here before: a child dialog with a dead printer
    qapp.processEvents()

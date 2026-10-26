"""RSM / DOE (Design-Expert-style) module. Placeholder until the workspace is built; see rsstatchem.ui.modules."""
from __future__ import annotations


def open_workspace(ctx, path: str | None = None):
    from PySide6.QtWidgets import QLabel, QMainWindow

    class _Placeholder(QMainWindow):
        def closeEvent(self, event) -> None:
            event.accept()
            ctx.window_closed(self)

    win = _Placeholder()
    win.ctx = ctx
    win.setWindowTitle("RSM / DOE (Design-Expert-style)")
    win.setCentralWidget(QLabel("This module is being built."))
    win.resize(800, 500)
    win.show()
    return win

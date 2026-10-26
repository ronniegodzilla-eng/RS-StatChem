"""Graph window inside the Origin workspace.

Reuses the Graph Studio window (:class:`rsstatchem.ui.graph.window.GraphWindow`) for the page
canvas, picking, double-click property dialogs, annotations, layers, undo and export, but as
an MDI child: its own menu bar, toolbar, docks and status bar are hidden — the Origin main
window provides the menus, the Tools toolbar, the Object Manager (it hosts this window's
object tree) and the status bar (data reader). Changes mark the project as modified instead
of prompting to save a separate ``.rsg`` file.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QToolBar, QWidget

from rsstatchem.core.graph.model import Graph
from rsstatchem.ui.graph import window as gw


class OriginGraphWindow(gw.GraphWindow):
    main = None   # the Origin main window (set after construction)

    def __init__(self, graph: Graph, main=None, path: str | None = None):
        super().__init__(graph, None, path)
        self.main = main
        if self in gw.WINDOWS:   # lifetime is managed by the Origin main window
            gw.WINDOWS.remove(self)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.menuBar().setNativeMenuBar(False)
        self.menuBar().hide()
        for tb in self.findChildren(QToolBar):
            tb.hide()
        for act in self.findChildren(QAction):   # the main window owns the shortcuts
            act.setShortcut(QKeySequence())
        self.statusBar().hide()
        if main is not None:
            self.statusBar().messageChanged.connect(main.show_status)
        self.prop_dock.hide()
        self.tree_dock.hide()
        # the object tree moves into the main window's Object Manager dock
        self.tree_dock.setWidget(QWidget())
        self.setMinimumSize(200, 150)

    # ------------------------------------------------------------------ integration hooks
    def _set_modified(self, on: bool) -> None:
        self.modified = on
        if on and self.main is not None:
            self.main.set_modified(True)
            self.main.graph_changed(self)

    def _update_title(self) -> None:
        self.setWindowTitle(self.graph.title)

    def _update_reader(self, event) -> None:
        super()._update_reader(event)
        if self.main is not None:
            self.main.show_reader(self.reader.text())

    def _settings(self):
        ctx = getattr(self.main, "ctx", None)
        return getattr(ctx, "settings", None)

    def add_plot(self) -> None:
        if self.main is not None:
            self.main.add_plot_to_layer(self)

    def add_layer(self, where: str) -> None:
        host = self.active_layer()
        from rsstatchem.core.graph import edit

        def fn():
            layer = edit.add_inset(self.graph, host) if where == "inset" else \
                edit.add_panel(self.graph, where, host)
            self.selected = ("layer", self.graph.layers.index(layer))
        self.mutate(fn)

    def send_to_output(self) -> None:   # no SPSS Output Viewer in this workspace
        self.copy_to_clipboard()

    def closeEvent(self, event) -> None:  # noqa: N802
        event.accept()   # the Origin main window asks Delete / Hide for the MDI window

"""Graph window (one per graph, like an Origin graph window).

Page canvas with zoom / pan / rescale, a data reader in the status bar, an Object Manager tree
(Graph › Layer › Axes / Plots / Legend / Annotations) and a docked property editor.
Double-click an axis, plot, legend or annotation on the page to open its properties; drag a
legend or text to move it; insert Text / Arrow / Line / Rectangle annotations with the
toolbar. Every edit is undoable. The model lives in ``self.graph``; the canvas is redrawn from
it after each change.
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib.axis
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QImage, QKeySequence
from PySide6.QtWidgets import (QApplication, QComboBox, QDockWidget, QFileDialog, QGridLayout,
                               QInputDialog, QLabel, QMainWindow, QMenu, QMessageBox,
                               QScrollArea, QTreeWidget, QTreeWidgetItem, QWidget)

from rsstatchem.core.graph import edit
from rsstatchem.core.graph.export import (EXTENSION, export_bytes, export_graph, load_graph,
                                          png_bytes, save_graph)
from rsstatchem.core.graph.model import Graph, from_dict, set_path, to_dict
from rsstatchem.core.graph.render import render
from rsstatchem.core.graph.themes import (THEMES, apply_template, apply_theme,
                                          delete_user_template, list_user_templates,
                                          load_user_template, save_user_template)
from rsstatchem.ui.graph import gicons
from rsstatchem.ui.graph.dialogs import ExportDialog, PageSetupDialog
from rsstatchem.ui.graph.properties import PropertiesDialog, PropertyForm, schema_for

WINDOWS: list["GraphWindow"] = []
REF_ROLE = Qt.ItemDataRole.UserRole
ZOOMS = [25, 50, 75, 100, 125, 150, 200, 300]
_PRIORITY = {"annotation": 0, "legend": 1, "plot": 2, "ref": 3, "axis": 3, "colorbar": 4,
             "roi": 5, "layer": 5}
_counter = [0]


def next_title() -> str:
    _counter[0] += 1
    return f"Graph{_counter[0]}"


class _Nav(NavigationToolbar2QT):
    toolitems = [t for t in NavigationToolbar2QT.toolitems if t[0] in ("Pan", "Zoom")]

    def set_message(self, s):   # the window has its own data reader
        pass


class GraphWindow(QMainWindow):
    def __init__(self, graph: Graph, editor=None, path: str | None = None):
        super().__init__()
        self.graph = graph
        self.editor = editor
        self.path = path
        self.modified = False
        self.zoom = 1.0
        self.fit = True
        self.rmap = None
        self.selected: tuple = ("graph",)
        self.undo_stack: list[dict] = []
        self.redo_stack: list[dict] = []
        self._last_push = (None, 0.0)
        self._drag = None
        self._place = None
        self._place_start = None
        self.export_settings: dict = {"fmt": "png", "dpi": 600}
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(1240, 800)
        self.setWindowIcon(gicons.icon("graph"))

        self.figure = Figure(dpi=96)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.nav = _Nav(self.canvas, self, coordinates=False)
        self.nav.hide()
        page = QWidget()
        page.setObjectName("graphPage")
        page.setStyleSheet("#graphPage { background: #8a8a8a; }")
        grid = QGridLayout(page)
        grid.setContentsMargins(24, 24, 24, 24)
        grid.addWidget(self.canvas, 0, 0, Qt.AlignmentFlag.AlignCenter)
        self.scroll = QScrollArea()
        self.scroll.setWidget(page)
        self.scroll.setWidgetResizable(True)
        self.scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCentralWidget(self.scroll)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemSelectionChanged.connect(self._tree_selected)
        self.tree.itemDoubleClicked.connect(lambda *_: self.open_properties())
        self.tree.itemChanged.connect(self._tree_item_changed)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._tree_menu)
        self.tree_dock = QDockWidget("Object Manager", self)
        self.tree_dock.setObjectName("objectManager")
        self.tree_dock.setWidget(self.tree)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.tree_dock)
        self.prop_scroll = QScrollArea()
        self.prop_scroll.setWidgetResizable(True)
        self.prop_dock = QDockWidget("Properties", self)
        self.prop_dock.setObjectName("properties")
        self.prop_dock.setWidget(self.prop_scroll)
        self.prop_dock.setMinimumWidth(330)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.prop_dock)

        self.reader = QLabel("")
        self.statusBar().addPermanentWidget(self.reader, 1)

        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self.redraw)
        self._update_title()
        self.redraw(rebuild_tree=True)
        WINDOWS.append(self)

    # ================================================================== actions / menus
    def _act(self, text, slot, shortcut=None, icon=None, checkable=False) -> QAction:
        act = QAction(text, self)
        if icon:
            act.setIcon(gicons.icon(icon))
        if shortcut:
            act.setShortcut(QKeySequence(shortcut))
        act.setCheckable(checkable)
        act.triggered.connect(lambda checked=False: slot())
        return act

    def _build_actions(self) -> None:
        a = self._act
        self.act_save = a("&Save Graph", self.save, QKeySequence.StandardKey.Save, "save")
        self.act_save_as = a("Save Graph &As...", self.save_as, "Ctrl+Shift+S")
        self.act_open = a("&Open Graph...", self.open_graph, None, "open")
        self.act_export = a("&Export Graph...", self.export, "Ctrl+E", "export")
        self.act_copy = a("&Copy Graph to Clipboard", self.copy_to_clipboard,
                          "Ctrl+Shift+C", "copy")
        self.act_output = a("Send to &Output Viewer", self.send_to_output, None, "output")
        self.act_close = a("&Close", self.close, QKeySequence.StandardKey.Close)
        self.act_undo = a("&Undo", self.undo, QKeySequence.StandardKey.Undo, "undo")
        self.act_redo = a("&Redo", self.redo, QKeySequence.StandardKey.Redo, "redo")
        self.act_delete = a("&Delete Selected Object", self.delete_selected, "Del")
        self.act_props = a("&Properties...", self.open_properties, "Alt+Return")
        self.act_rescale = a("&Rescale", self.rescale, "Ctrl+R", "rescale")
        self.act_add_plot = a("&Add Plot to Layer...", self.add_plot, None, "add_plot")
        self.act_page = a("&Page Setup...", self.page_setup)
        self.act_right_y = a("Show &Right Y Axis", self.toggle_right_y, checkable=True)
        self.act_panel_below = a("New Panel &Below", lambda: self.add_layer("below"))
        self.act_panel_right = a("New Panel &Right", lambda: self.add_layer("right"))
        self.act_inset = a("&Inset Layer", lambda: self.add_layer("inset"))
        self.act_remove_layer = a("Remove &Layer", self.remove_layer)
        self.act_save_tpl = a("Save as &Template...", self.save_template)
        self.act_del_tpl = a("&Delete User Template...", self.delete_template)

        group = QActionGroup(self)
        self.act_pointer = a("Pointer", lambda: self.set_tool(None), "Esc", "pointer", True)
        self.act_zoom = a("Zoom", lambda: self.set_tool("zoom"), "Z", "zoom", True)
        self.act_pan = a("Pan", lambda: self.set_tool("pan"), "P", "pan", True)
        self.act_text = a("Add Text", lambda: self.set_tool("text"), "T", "text", True)
        self.act_arrow = a("Add Arrow", lambda: self.set_tool("arrow"), None, "arrow", True)
        self.act_line = a("Add Line", lambda: self.set_tool("line"), None, "line", True)
        self.act_rect = a("Add Rectangle", lambda: self.set_tool("rect"), None, "rect", True)
        for act in (self.act_pointer, self.act_zoom, self.act_pan, self.act_text,
                    self.act_arrow, self.act_line, self.act_rect):
            group.addAction(act)
        self.act_pointer.setChecked(True)

    def _build_menus(self) -> None:
        mb = self.menuBar()
        f = mb.addMenu("&File")
        f.addActions([self.act_save, self.act_save_as, self.act_open])
        f.addSeparator()
        f.addActions([self.act_export, self.act_copy, self.act_output])
        f.addSeparator()
        f.addAction(self.act_close)
        e = mb.addMenu("&Edit")
        e.addActions([self.act_undo, self.act_redo])
        e.addSeparator()
        e.addActions([self.act_delete, self.act_props])
        v = mb.addMenu("&View")
        v.addAction(self._act("Zoom &In", lambda: self.set_zoom(self.zoom * 1.25), "Ctrl++"))
        v.addAction(self._act("Zoom &Out", lambda: self.set_zoom(self.zoom / 1.25), "Ctrl+-"))
        v.addAction(self._act("&Actual Size (100%)", lambda: self.set_zoom(1.0), "Ctrl+0"))
        v.addAction(self._act("&Fit Page in Window", self.fit_page, "Ctrl+9"))
        v.addSeparator()
        v.addAction(self.tree_dock.toggleViewAction())
        v.addAction(self.prop_dock.toggleViewAction())
        g = mb.addMenu("&Graph")
        g.addActions([self.act_add_plot, self.act_rescale])
        g.addSeparator()
        layers = g.addMenu("Add &Layer")
        layers.addActions([self.act_panel_below, self.act_panel_right, self.act_inset])
        g.addAction(self.act_right_y)
        g.addAction(self.act_remove_layer)
        g.addSeparator()
        g.addAction(self.act_page)
        fm = mb.addMenu("F&ormat")
        themes = fm.addMenu("&Theme")
        for name in THEMES:
            themes.addAction(self._act(name, lambda n=name: self.apply_theme(n)))
        self.tpl_menu = fm.addMenu("Apply &User Template")
        self.tpl_menu.aboutToShow.connect(self._fill_templates)
        fm.addActions([self.act_save_tpl, self.act_del_tpl])
        ins = mb.addMenu("&Insert")
        ins.addActions([self.act_text, self.act_arrow, self.act_line, self.act_rect])

    def _build_toolbar(self) -> None:
        tb = self.addToolBar("Graph")
        tb.setObjectName("graphToolbar")
        tb.setMovable(False)
        tb.addActions([self.act_save, self.act_undo, self.act_redo])
        tb.addSeparator()
        tb.addActions([self.act_pointer, self.act_zoom, self.act_pan, self.act_rescale])
        tb.addSeparator()
        tb.addActions([self.act_text, self.act_arrow, self.act_line, self.act_rect])
        tb.addSeparator()
        tb.addAction(self.act_add_plot)
        tb.addSeparator()
        self.zoom_combo = QComboBox()
        self.zoom_combo.addItem("Fit")
        for z in ZOOMS:
            self.zoom_combo.addItem(f"{z}%")
        self.zoom_combo.setToolTip("Page zoom")
        self.zoom_combo.activated.connect(self._zoom_chosen)
        tb.addWidget(self.zoom_combo)
        self.theme_combo = QComboBox()
        self.theme_combo.addItem("Theme…")
        for name in THEMES:
            self.theme_combo.addItem(name)
        self.theme_combo.setToolTip("Apply a graph theme")
        self.theme_combo.setMaximumWidth(240)
        self.theme_combo.activated.connect(self._theme_chosen)
        tb.addWidget(self.theme_combo)
        tb.addSeparator()
        tb.addActions([self.act_export, self.act_copy, self.act_output])

    # ================================================================== rendering
    def schedule(self) -> None:
        self._timer.start()

    def redraw(self, rebuild_tree: bool = False) -> None:
        self.rmap = render(self.graph, self.figure)
        self._apply_view()
        self.canvas.draw()
        self.nav.update()
        if rebuild_tree:
            self.rebuild_tree()
        layer = self.active_layer()
        self.act_right_y.setChecked(bool(layer and layer.show_y2))
        if self.rmap.errors:
            self.statusBar().showMessage("; ".join(self.rmap.errors[:3]), 8000)

    def _apply_view(self) -> None:
        w_in = self.graph.page.width_cm / 2.54
        h_in = self.graph.page.height_cm / 2.54
        if self.fit:
            vp = self.scroll.viewport().size()
            avail_w, avail_h = max(200, vp.width() - 52), max(150, vp.height() - 52)
            self.zoom = max(0.2, min(4.0, avail_w / (w_in * 96), avail_h / (h_in * 96)))
        dpi = 96 * self.zoom
        ratio = self.canvas.device_pixel_ratio or 1
        self.figure._original_dpi = dpi
        self.figure.set_dpi(dpi * ratio)
        self.figure.set_size_inches(w_in, h_in, forward=False)
        self.canvas.setFixedSize(int(round(w_in * dpi)), int(round(h_in * dpi)))
        self.zoom_combo.setItemText(0, f"Fit ({self.zoom * 100:.0f}%)" if self.fit else "Fit")
        if not self.fit:
            idx = self.zoom_combo.findText(f"{round(self.zoom * 100)}%")
            self.zoom_combo.setCurrentIndex(idx if idx >= 0 else -1)
        else:
            self.zoom_combo.setCurrentIndex(0)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self.fit and self.rmap is not None:
            QTimer.singleShot(0, self._refit)

    def _refit(self) -> None:
        if self.fit:
            self._apply_view()
            self.canvas.draw()

    def set_zoom(self, zoom: float) -> None:
        self.fit = False
        self.zoom = max(0.2, min(4.0, zoom))
        self._apply_view()
        self.canvas.draw_idle()

    def fit_page(self) -> None:
        self.fit = True
        self._refit()

    def _zoom_chosen(self, index: int) -> None:
        if index == 0:
            self.fit_page()
        else:
            self.set_zoom(ZOOMS[index - 1] / 100)

    def _theme_chosen(self, index: int) -> None:
        if index > 0:
            self.apply_theme(self.theme_combo.itemText(index))
        self.theme_combo.setCurrentIndex(0)

    # ================================================================== model edits
    def push_undo(self, key=None) -> None:
        now = time.monotonic()
        if key is not None and self._last_push[0] == key and now - self._last_push[1] < 1.5:
            self._last_push = (key, now)
            return
        self._last_push = (key, now)
        self.undo_stack.append(to_dict(self.graph))
        del self.undo_stack[:-60]
        self.redo_stack.clear()
        self._set_modified(True)

    def _set_modified(self, on: bool) -> None:
        self.modified = on
        self.setWindowModified(on)

    def mutate(self, fn, rebuild_tree: bool = True) -> None:
        self.push_undo()
        try:
            fn()
        except ValueError as exc:
            self.undo_stack.pop()
            QMessageBox.warning(self, "Graph", str(exc))
            return
        self.redraw(rebuild_tree=rebuild_tree)
        self.show_properties(self.selected)

    def apply_change(self, obj, path: str, value, push: bool = True) -> None:
        if push:
            self.push_undo((id(obj), path))
        if path.startswith("@entry:"):
            layer = self.active_layer()
            pid = path.split(":", 1)[1]
            if value and pid in layer.legend.hidden:
                layer.legend.hidden.remove(pid)
            elif not value and pid not in layer.legend.hidden:
                layer.legend.hidden.append(pid)
        else:
            set_path(obj, path, value)
            if path == "type" and value in ("line", "scatter", "line_symbol"):
                obj.line.visible = value != "scatter"
                obj.symbol.visible = value != "line"
            elif path == "type":
                obj.fill.visible = True
        if path in ("name", "title", "type", "show_y2", "y_axis", "visible") or \
                path.startswith("@entry"):
            QTimer.singleShot(0, self._refresh_tree_texts)
        if path == "title" and obj is self.graph:
            self._update_title()
        self.schedule()

    def undo(self) -> None:
        if not self.undo_stack:
            return
        self.redo_stack.append(to_dict(self.graph))
        self.graph = from_dict(Graph, self.undo_stack.pop())
        self._after_restore()

    def redo(self) -> None:
        if not self.redo_stack:
            return
        self.undo_stack.append(to_dict(self.graph))
        self.graph = from_dict(Graph, self.redo_stack.pop())
        self._after_restore()

    def _after_restore(self) -> None:
        self._set_modified(True)
        self._last_push = (None, 0.0)
        if self.resolve(self.selected) is None:
            self.selected = ("graph",)
        self.redraw(rebuild_tree=True)
        self.show_properties(self.selected)
        self._update_title()

    # ================================================================== selection
    def resolve(self, ref: tuple):
        g = self.graph
        kind = ref[0]
        if kind == "graph":
            return g
        li = ref[1]
        if li >= len(g.layers):
            return None
        layer = g.layers[li]
        if kind in ("layer", "colorbar", "roi"):
            return layer
        if kind == "ref":
            return next((r for r in layer.refs if r.id == ref[2]), None)
        if kind == "axis":
            return layer.axis(ref[2])
        if kind == "legend":
            return layer.legend
        items = layer.plots if kind == "plot" else layer.annotations
        return next((i for i in items if i.id == ref[2]), None)

    def active_layer(self):
        if len(self.selected) > 1 and self.selected[1] < len(self.graph.layers):
            return self.graph.layers[self.selected[1]]
        return self.graph.layers[0] if self.graph.layers else None

    def active_index(self) -> int:
        return self.selected[1] if len(self.selected) > 1 else 0

    def rebuild_tree(self) -> None:
        self.tree.blockSignals(True)
        self.tree.clear()
        root = QTreeWidgetItem([self.graph.title])
        root.setIcon(0, gicons.icon("graph"))
        root.setData(0, REF_ROLE, ("graph",))
        self.tree.addTopLevelItem(root)
        for li, layer in enumerate(self.graph.layers):
            kind = {"cartesian": "", "polar": " (polar)", "3d": " (3D)",
                    "ternary": " (ternary)", "pie": " (pie)"}[layer.kind]
            inset = " (inset)" if layer.rect else ""
            item = QTreeWidgetItem([f"Layer {li + 1}{kind}{inset}"])
            item.setIcon(0, gicons.icon("layer"))
            item.setData(0, REF_ROLE, ("layer", li))
            root.addChild(item)
            for which, name in self._axis_names(layer):
                ax_item = QTreeWidgetItem([name])
                ax_item.setIcon(0, gicons.icon("axis"))
                ax_item.setData(0, REF_ROLE, ("axis", li, which))
                item.addChild(ax_item)
            plots = QTreeWidgetItem(["Plots"])
            plots.setIcon(0, gicons.icon("folder"))
            plots.setData(0, REF_ROLE, ("layer", li))
            item.addChild(plots)
            for p in layer.plots:
                pi = QTreeWidgetItem([self._plot_text(p)])
                pi.setIcon(0, gicons.icon("plot"))
                pi.setData(0, REF_ROLE, ("plot", li, p.id))
                pi.setFlags(pi.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                pi.setCheckState(0, Qt.CheckState.Checked if p.visible
                                 else Qt.CheckState.Unchecked)
                plots.addChild(pi)
            if layer.kind != "3d":
                lg = QTreeWidgetItem(["Legend"])
                lg.setIcon(0, gicons.icon("legend"))
                lg.setData(0, REF_ROLE, ("legend", li))
                lg.setFlags(lg.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                lg.setCheckState(0, Qt.CheckState.Checked if layer.legend.visible
                                 else Qt.CheckState.Unchecked)
                item.addChild(lg)
            if layer.refs:
                rf = QTreeWidgetItem(["Reference Lines"])
                rf.setIcon(0, gicons.icon("folder"))
                rf.setData(0, REF_ROLE, ("layer", li))
                item.addChild(rf)
                for r in layer.refs:
                    text = r.label or (f"{r.axis.upper()} = {r.value:g}" if r.mode == "value"
                                       else f"{r.axis.upper()} {r.mode.replace('_', ' ± ')}")
                    ri = QTreeWidgetItem([("Band: " if r.kind == "band" else "") + text])
                    ri.setIcon(0, gicons.icon("annotation"))
                    ri.setData(0, REF_ROLE, ("ref", li, r.id))
                    rf.addChild(ri)
            if layer.annotations:
                an = QTreeWidgetItem(["Annotations"])
                an.setIcon(0, gicons.icon("folder"))
                an.setData(0, REF_ROLE, ("layer", li))
                item.addChild(an)
                for a in layer.annotations:
                    text = a.text if a.kind == "text" else a.kind.capitalize()
                    ai = QTreeWidgetItem([text[:40]])
                    ai.setIcon(0, gicons.icon("annotation"))
                    ai.setData(0, REF_ROLE, ("annotation", li, a.id))
                    an.addChild(ai)
        self.tree.expandAll()
        self.tree.blockSignals(False)
        self._select_in_tree(self.selected)
        if self.prop_scroll.widget() is None:
            self.show_properties(self.selected)

    @staticmethod
    def _plot_text(p) -> str:
        return f"{p.name or '(unnamed)'}  [{p.type.replace('_', ' + ')}]"

    def _axis_names(self, layer) -> list[tuple[str, str]]:
        if layer.kind == "pie":
            return []
        if layer.kind == "ternary":
            return [("x", "A Axis"), ("y", "B Axis"), ("z", "C Axis")]
        if layer.kind == "polar":
            return [("x", "Angular Axis (θ)"), ("y", "Radial Axis (r)")]
        names = [("x", "X Axis"), ("y", "Y Axis")]
        if layer.kind == "3d":
            names.append(("z", "Z Axis"))
        elif layer.show_y2 or layer.y2_formula or any(p.y_axis == "right" or
                                                      p.type == "pareto" for p in layer.plots):
            names.append(("y2", "Right Y Axis"))
        if layer.kind == "cartesian" and (layer.show_x2 or any(p.x_axis == "top"
                                                                for p in layer.plots)):
            names.append(("x2", "Top X Axis"))
        return names

    def _refresh_tree_texts(self) -> None:
        self.rebuild_tree()

    def _iter_items(self):
        stack = [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]
        while stack:
            item = stack.pop()
            yield item
            stack.extend(item.child(i) for i in range(item.childCount()))

    def _select_in_tree(self, ref) -> None:
        for item in self._iter_items():
            if item.data(0, REF_ROLE) == ref and item.text(0) not in ("Plots", "Annotations"):
                self.tree.blockSignals(True)
                self.tree.setCurrentItem(item)
                self.tree.blockSignals(False)
                return

    def _tree_selected(self) -> None:
        item = self.tree.currentItem()
        if item is None:
            return
        self.select(tuple(item.data(0, REF_ROLE)), from_tree=True)

    def _tree_item_changed(self, item, column) -> None:
        ref = item.data(0, REF_ROLE)
        if not ref or ref[0] not in ("plot", "legend"):
            return
        obj = self.resolve(tuple(ref))
        on = item.checkState(0) == Qt.CheckState.Checked
        if obj is not None and obj.visible != on:
            self.apply_change(obj, "visible", on)

    def _tree_menu(self, pos) -> None:
        item = self.tree.itemAt(pos)
        if item is None:
            return
        self.select(tuple(item.data(0, REF_ROLE)))
        menu = QMenu(self)
        menu.addAction(self.act_props)
        if self.selected[0] in ("plot", "annotation", "layer"):
            menu.addAction(self.act_delete)
        if self.selected[0] in ("layer", "plot", "axis"):
            menu.addAction(self.act_add_plot)
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def select(self, ref: tuple, from_tree: bool = False) -> None:
        if self.resolve(ref) is None:
            return
        self.selected = ref
        if not from_tree:
            self._select_in_tree(ref)
        self.show_properties(ref)
        layer = self.active_layer()
        self.act_right_y.setChecked(bool(layer and layer.show_y2))

    def show_properties(self, ref: tuple) -> None:
        obj = self.resolve(ref)
        if obj is None:
            return
        sections, layer = schema_for(obj, ref, self.graph)
        form = PropertyForm(obj, sections, lambda p, v, o=obj: self.apply_change(o, p, v),
                            layer)
        self.prop_scroll.setWidget(form)
        self.prop_dock.setWindowTitle(f"Properties — {self._ref_title(ref)}")

    def _ref_title(self, ref) -> str:
        kind = ref[0]
        if kind == "graph":
            return "Graph"
        base = f"Layer {ref[1] + 1}"
        if kind == "axis":
            layer = self.graph.layers[ref[1]]
            return f"{base} › " + dict(self._axis_names(layer)).get(ref[2], ref[2])
        if kind == "plot":
            return f"{base} › {self.resolve(ref).name or 'Plot'}"
        if kind == "annotation":
            return f"{base} › {self.resolve(ref).kind.capitalize()}"
        if kind == "ref":
            return f"{base} › Reference Line"
        return f"{base} › {kind.capitalize()}" if kind not in ("layer",) else base

    def open_properties(self, ref: tuple | None = None) -> None:
        ref = ref or self.selected
        obj = self.resolve(ref)
        if obj is None:
            return
        snapshot = to_dict(self.graph)
        self.push_undo()
        sections, layer = schema_for(obj, ref, self.graph)
        dlg = PropertiesDialog(self._ref_title(ref), obj, sections,
                               lambda p, v: self.apply_change(obj, p, v, push=False), layer,
                               self)
        if dlg.exec():
            self.redraw(rebuild_tree=True)
            self.show_properties(self.selected)
        else:
            self.undo_stack.pop()
            self.graph = from_dict(Graph, snapshot)
            self.redraw(rebuild_tree=True)
            self.show_properties(self.selected)

    # ================================================================== picking & tools
    def set_tool(self, tool: str | None) -> None:
        if self.nav.mode.name == "ZOOM" and tool != "zoom":
            self.nav.zoom()
        if self.nav.mode.name == "PAN" and tool != "pan":
            self.nav.pan()
        if tool == "zoom" and self.nav.mode.name != "ZOOM":
            self.nav.zoom()
        if tool == "pan" and self.nav.mode.name != "PAN":
            self.nav.pan()
        self._place = tool if tool in ("text", "arrow", "line", "rect") else None
        cursor = Qt.CursorShape.CrossCursor if self._place else Qt.CursorShape.ArrowCursor
        self.canvas.setCursor(cursor)
        {None: self.act_pointer, "zoom": self.act_zoom, "pan": self.act_pan,
         "text": self.act_text, "arrow": self.act_arrow, "line": self.act_line,
         "rect": self.act_rect}[tool].setChecked(True)

    def hit_test(self, event):
        if self.rmap is None:
            return None
        renderer = self.canvas.get_renderer()
        hits = []
        for order, (artist, ref) in enumerate(self.rmap.artists):
            try:
                if isinstance(artist, matplotlib.axis.Axis):
                    bb = artist.get_tightbbox(renderer)
                    hit = bb is not None and bb.contains(event.x, event.y)
                    if not hit:
                        hit = self._near_spine(artist, event)
                elif ref[0] == "layer":
                    hit = artist.contains(event)[0]
                else:
                    hit = artist.get_visible() and artist.contains(event)[0]
            except Exception:  # noqa: BLE001 - some artists don't implement contains
                continue
            if hit:
                hits.append((_PRIORITY.get(ref[0], 9), -order, ref, artist))
        if not hits:
            return None
        hits.sort(key=lambda h: (h[0], h[1]))
        return hits[0][2], hits[0][3]

    @staticmethod
    def _near_spine(mpl_axis, event) -> bool:
        ax = mpl_axis.axes
        if ax.name != "rectilinear":
            return False
        bb = ax.bbox
        tol = 6
        if isinstance(mpl_axis, matplotlib.axis.XAxis):
            return bb.x0 - tol <= event.x <= bb.x1 + tol and abs(event.y - bb.y0) <= tol
        if mpl_axis.get_label_position() == "right" or ax.get_label() == "twin":
            return bb.y0 - tol <= event.y <= bb.y1 + tol and abs(event.x - bb.x1) <= tol
        return bb.y0 - tol <= event.y <= bb.y1 + tol and abs(event.x - bb.x0) <= tol

    def _layer_at(self, event) -> int:
        for li, info in self.rmap.axes.items():
            if info["ax"].bbox.contains(event.x, event.y):
                if info["layer"].rect:      # insets win over the host layer
                    return li
        for li, info in self.rmap.axes.items():
            if info["ax"].bbox.contains(event.x, event.y):
                return li
        return self.active_index()

    def _on_press(self, event) -> None:
        if self.nav.mode.name in ("ZOOM", "PAN") or event.button != 1:
            return
        if self._place:
            self._place_start = (event.x, event.y, self._layer_at(event))
            return
        hit = self.hit_test(event)
        if hit is None:
            return
        ref, artist = hit
        self.select(ref)
        if event.dblclick:
            QTimer.singleShot(0, lambda r=ref: self.open_properties(r))
            return
        if ref[0] in ("legend", "annotation"):
            self._drag = {"ref": ref, "artist": artist, "x0": event.x, "y0": event.y,
                          "moved": False}

    def _on_motion(self, event) -> None:
        if self._drag is not None and event.x is not None:
            d = self._drag
            dx, dy = event.x - d["x0"], event.y - d["y0"]
            if abs(dx) + abs(dy) > 2:
                d["moved"] = True
                self._drag_preview(d, dx, dy)
            return
        self._update_reader(event)

    def _drag_preview(self, d, dx, dy) -> None:
        artist = d["artist"]
        if d["ref"][0] == "legend":
            if "bbox" not in d:
                d["bbox"] = artist.get_window_extent()
            ax = artist.axes
            x, y = ax.transAxes.inverted().transform((d["bbox"].x0 + dx, d["bbox"].y0 + dy))
            artist.set_loc("lower left")
            artist.set_bbox_to_anchor((x, y), transform=ax.transAxes)
            d["new"] = (float(x), float(y))
        else:
            obj = self.resolve(d["ref"])
            if obj is None or obj.kind != "text":
                return
            if "start" not in d:
                d["start"] = artist.get_transform().transform((obj.x, obj.y))
            sx, sy = d["start"]
            x, y = artist.get_transform().inverted().transform((sx + dx, sy + dy))
            artist.set_position((x, y))
            d["new"] = (float(x), float(y))
        self.canvas.draw_idle()

    def _on_release(self, event) -> None:
        if self.nav.mode.name in ("ZOOM", "PAN"):
            QTimer.singleShot(0, self._capture_view)
            return
        if self._place and self._place_start is not None:
            self._finish_placement(event)
            return
        if self._capture_3d_view():
            self._drag = None
            return
        d, self._drag = self._drag, None
        if d is None or not d["moved"]:
            return
        ref = d["ref"]
        if ref[0] == "annotation":
            obj = self.resolve(ref)
            if obj is None:
                return
            if obj.kind == "text" and "new" in d:
                self.push_undo()
                obj.x, obj.y = d["new"]
            elif obj.kind != "text" and event.x is not None:
                tr = d["artist"].get_transform()
                (x0, y0), (x1, y1) = tr.inverted().transform([(d["x0"], d["y0"]),
                                                              (event.x, event.y)])
                self.push_undo()
                obj.x, obj.y = obj.x + x1 - x0, obj.y + y1 - y0
                obj.x2, obj.y2 = obj.x2 + x1 - x0, obj.y2 + y1 - y0
        elif ref[0] == "legend" and "new" in d:
            lg = self.resolve(ref)
            self.push_undo()
            lg.loc, (lg.x, lg.y) = "custom", d["new"]
        self.redraw()
        self.show_properties(self.selected)

    def _capture_view(self) -> None:
        """After a zoom/pan, store the visible ranges in the axes (Origin changes scales)."""
        if self.rmap is None:
            return
        changed = False
        snapshot = to_dict(self.graph)
        for li, info in self.rmap.axes.items():
            layer = info["layer"]
            if layer.kind != "cartesian":
                continue
            ax = info["ax"]
            pairs = [("x", ax.get_xlim()), ("y", ax.get_ylim())]
            if info["ax2"] is not None:
                pairs.append(("y2", info["ax2"].get_ylim()))
            for which, (a, b) in pairs:
                axis = layer.axis(which)
                lo, hi = sorted((float(a), float(b)))
                if axis.min is None or abs(axis.min - lo) > 1e-12 or \
                        abs((axis.max or 0) - hi) > 1e-12:
                    axis.min, axis.max = float(f"{lo:.6g}"), float(f"{hi:.6g}")
                    changed = True
        if changed:
            self.undo_stack.append(snapshot)
            self.redo_stack.clear()
            self._set_modified(True)
            self.redraw()
            self.show_properties(self.selected)

    def _capture_3d_view(self) -> bool:
        """After dragging a 3D layer (matplotlib rotates it with the mouse), keep the new
        elevation / azimuth in the layer so redraws, saving and export use them."""
        if self.rmap is None:
            return False
        changes = []
        for info in self.rmap.axes.values():
            layer, ax = info["layer"], info["ax"]
            if layer.kind != "3d" or getattr(ax, "name", "") != "3d":
                continue
            elev, azim = float(ax.elev), float(ax.azim)
            if abs(elev - layer.elev) > 0.05 or abs(azim - layer.azim) > 0.05:
                changes.append((layer, round(elev, 1), round(azim, 1)))
        if not changes:
            return False
        self.push_undo()
        for layer, elev, azim in changes:
            layer.elev, layer.azim = elev, azim
        self.redraw()
        self.show_properties(self.selected)
        return True

    def rotate_3d(self, d_azim: float = 0.0, d_elev: float = 0.0) -> None:
        """Rotate the active 3D layer (keyboard / menu equivalent of dragging it)."""
        layer = self.active_layer()
        if layer is None or layer.kind != "3d":
            layer = next((lay for lay in self.graph.layers if lay.kind == "3d"), None)
        if layer is None:
            return

        def fn():
            layer.azim = (layer.azim + d_azim + 180) % 360 - 180
            layer.elev = max(-90.0, min(90.0, layer.elev + d_elev))
        self.mutate(fn, rebuild_tree=False)

    def rescale(self) -> None:
        def fn():
            for layer in self.graph.layers:
                for which in ("x", "y", "y2", "z"):
                    axis = layer.axis(which)
                    axis.min = axis.max = None
        self.mutate(fn, rebuild_tree=False)

    def _finish_placement(self, event) -> None:
        x0, y0, li = self._place_start
        self._place_start = None
        kind = self._place
        info = self.rmap.axes.get(li)
        if info is None:
            return
        ax = info["ax"]
        x1, y1 = (event.x, event.y) if event.x is not None else (x0, y0)
        inv = ax.transAxes.inverted()
        (ax0, ay0), (ax1, ay1) = inv.transform([(x0, y0), (x1, y1)])
        if kind != "text" and abs(x1 - x0) + abs(y1 - y0) < 6:
            ax1, ay1 = ax0 + 0.2, ay0 + (0.12 if kind != "line" else 0.0)
        layer = self.graph.layers[li]
        if kind == "text":
            text, ok = QInputDialog.getText(self, "Add Text",
                                            "Text (H$_2$O, cm$^{-1}$, \\+(sup), \\-(sub)):")
            if not ok or not text:
                self.set_tool(None)
                return
            ann = edit.new_annotation("text", ax0, ay0,
                                      font_size=layer.x.label_font.size)
            ann.text = text
        else:
            ann = edit.new_annotation(kind, ax0, ay0, ax1, ay1)
            ann.line.width = max(0.8, layer.x.line_width)
        self.push_undo()
        layer.annotations.append(ann)
        self.selected = ("annotation", li, ann.id)
        self.set_tool(None)
        self.redraw(rebuild_tree=True)
        self.show_properties(self.selected)

    def _update_reader(self, event) -> None:
        if self.rmap is None or event.inaxes is None or event.xdata is None:
            self.reader.setText("")
            return
        ax = event.inaxes
        if getattr(ax, "name", "") == "3d":
            self.reader.setText(f"3D layer — drag to rotate (elevation {ax.elev:.0f}°, "
                                f"azimuth {ax.azim:.0f}°).")
            return
        li = next((k for k, info in self.rmap.axes.items()
                   if info["ax"] is ax or info["ax2"] is ax), None)
        text = f"x = {event.xdata:.6g}   y = {event.ydata:.6g}"
        if li is not None:
            text = f"Layer {li + 1}:  " + text
        best = None
        for pli, pid, pax, xs, ys in self.rmap.points:
            if pli != li or not len(xs):
                continue
            pts = pax.transData.transform(np.column_stack([xs, ys]))
            d = np.hypot(pts[:, 0] - event.x, pts[:, 1] - event.y)
            k = int(np.nanargmin(d)) if np.isfinite(d).any() else -1
            if k >= 0 and d[k] < 10 and (best is None or d[k] < best[0]):
                best = (d[k], pid, k, xs[k], ys[k])
        if best is not None:
            plot = self.graph.find(best[1])
            text += f"     ▶ {plot.name if plot else ''} [{best[2] + 1}]  " \
                    f"X = {best[3]:.6g}, Y = {best[4]:.6g}"
        self.reader.setText(text)

    # ================================================================== graph commands
    def delete_selected(self) -> None:
        ref = self.selected
        if ref[0] == "plot":
            def fn():
                layer = self.graph.layers[ref[1]]
                layer.plots = [p for p in layer.plots if p.id != ref[2]]
                self.graph.prune_data()
        elif ref[0] == "annotation":
            def fn():
                layer = self.graph.layers[ref[1]]
                layer.annotations = [a for a in layer.annotations if a.id != ref[2]]
        elif ref[0] == "ref":
            def fn():
                layer = self.graph.layers[ref[1]]
                layer.refs = [r for r in layer.refs if r.id != ref[2]]
        elif ref[0] == "roi":
            def fn():
                self.graph.layers[ref[1]].options.pop("roi", None)
        elif ref[0] in ("layer", "colorbar"):
            def fn():
                edit.remove_layer(self.graph, ref[1])
        else:
            return
        self.selected = ("graph",) if ref[0] in ("layer", "colorbar") else ("layer", ref[1])
        self.mutate(fn)

    def remove_layer(self) -> None:
        li = self.active_index()
        self.selected = ("graph",)
        self.mutate(lambda: edit.remove_layer(self.graph, li))

    def add_layer(self, where: str) -> None:
        host = self.active_layer()

        def fn():
            if where == "inset":
                layer = edit.add_inset(self.graph, host)
            else:
                layer = edit.add_panel(self.graph, where, host)
            self.selected = ("layer", self.graph.layers.index(layer))
        self.mutate(fn)
        if self.editor is not None and getattr(self.editor, "ds", None) is not None:
            if QMessageBox.question(self, "Add Layer", "Add a plot to the new layer now?") == \
                    QMessageBox.StandardButton.Yes:
                self.add_plot()

    def toggle_right_y(self) -> None:
        layer = self.active_layer()
        if layer is None or layer.kind != "cartesian":
            self.act_right_y.setChecked(False)
            return
        on = not layer.show_y2
        self.mutate(lambda: setattr(layer, "show_y2", on))

    def add_plot(self) -> None:
        editor = self.editor
        ds = getattr(editor, "ds", None) if editor is not None else None
        if ds is None:
            QMessageBox.information(self, "Add Plot", "Open the graph from a Data Editor to "
                                                      "add plots from its dataset.")
            return
        from rsstatchem.core.graph.build import build_graph
        from rsstatchem.ui.graph.setup_dialog import PlotSetupDialog
        layer = self.active_layer()
        default = {"pie": "pie", "3d": "scatter3d", "ternary": "ternary",
                   "polar": "polar"}.get(layer.kind, "line_symbol")
        dlg = PlotSetupDialog(ds, self, kind=default, add_mode=True)
        if not dlg.exec():
            return
        spec = dlg.spec()
        spec.theme = self.graph.theme if self.graph.theme in THEMES else "Origin Default"
        try:
            src = build_graph(ds, spec)
        except ValueError as exc:
            QMessageBox.warning(self, "Add Plot", str(exc))
            return
        if len(src.layers) > 1:
            QMessageBox.warning(self, "Add Plot", "Multi-panel templates create their own "
                                                  "graph; choose a single-layer plot type.")
            return

        def fn():
            ids = edit.merge_plots(self.graph, layer, src)
            if src.layers[0].show_y2:
                layer.show_y2 = True
            li = self.graph.layers.index(layer)
            if ids:
                self.selected = ("plot", li, ids[-1])
        self.mutate(fn)

    def page_setup(self) -> None:
        dlg = PageSetupDialog(self.graph, self)
        if dlg.exec():
            w, h = dlg.values()
            self.mutate(lambda: (setattr(self.graph.page, "width_cm", w),
                                 setattr(self.graph.page, "height_cm", h)), False)

    def apply_theme(self, name: str) -> None:
        self.mutate(lambda: apply_theme(self.graph, name), rebuild_tree=False)

    def _fill_templates(self) -> None:
        self.tpl_menu.clear()
        names = list_user_templates()
        if not names:
            act = self.tpl_menu.addAction("(no user templates yet)")
            act.setEnabled(False)
        for name in names:
            self.tpl_menu.addAction(name, lambda n=name: self.mutate(
                lambda: apply_template(self.graph, load_user_template(n)), False))

    def save_template(self) -> None:
        name, ok = QInputDialog.getText(self, "Save as Template", "Template name:",
                                        text=self.graph.title)
        if ok and name.strip():
            path = save_user_template(name.strip(), self.graph)
            self.statusBar().showMessage(f"Template saved: {path}", 6000)

    def delete_template(self) -> None:
        names = list(list_user_templates())
        if not names:
            QMessageBox.information(self, "Templates", "There are no user templates.")
            return
        name, ok = QInputDialog.getItem(self, "Delete User Template", "Template:", names, 0,
                                        False)
        if ok:
            delete_user_template(name)

    # ================================================================== files & export
    def _settings(self):
        ctx = getattr(self.editor, "ctx", None)
        return getattr(ctx, "settings", None)

    def _start_dir(self) -> str:
        s = self._settings()
        return str(s.value("last_dir", str(Path.home()))) if s is not None else str(Path.home())

    def _remember_dir(self, path: str) -> None:
        s = self._settings()
        if s is not None:
            s.setValue("last_dir", str(Path(path).parent))

    def _update_title(self) -> None:
        name = Path(self.path).name if self.path else self.graph.title
        self.setWindowTitle(f"{name}[*] — Graph Studio")

    def save(self) -> bool:
        if not self.path:
            return self.save_as()
        save_graph(self.graph, self.path)
        self._set_modified(False)
        self.statusBar().showMessage(f"Saved {self.path}", 5000)
        return True

    def save_as(self) -> bool:
        start = str(Path(self._start_dir()) / f"{self.graph.title}{EXTENSION}")
        path, _ = QFileDialog.getSaveFileName(self, "Save Graph", start,
                                              f"RS-StatChem Graph (*{EXTENSION})")
        if not path:
            return False
        if not path.lower().endswith(EXTENSION):
            path += EXTENSION
        self.path = path
        self._remember_dir(path)
        self._update_title()
        return self.save()

    def open_graph(self) -> None:
        open_graph_dialog(self.editor, self)

    def export(self) -> None:
        dlg = ExportDialog(self.graph, self, self.export_settings)
        if not dlg.exec():
            return
        values = dlg.values()
        self.export_settings = values
        fmt = values["fmt"]
        start = str(Path(self._start_dir()) / f"{self.graph.title}.{fmt}")
        path, _ = QFileDialog.getSaveFileName(self, "Export Graph", start,
                                              f"{fmt.upper()} (*.{fmt})")
        if not path:
            return
        if not path.lower().endswith("." + fmt) and not (fmt == "tif" and
                                                         path.lower().endswith(".tiff")):
            path += "." + fmt
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            export_graph(self.graph, path, fmt, values["dpi"], values["width_cm"],
                         values["height_cm"], values["transparent"])
        except Exception as exc:  # noqa: BLE001 - report any writer failure
            QMessageBox.critical(self, "Export Graph", f"Could not export:\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        self._remember_dir(path)
        self.statusBar().showMessage(f"Exported {path}", 6000)

    def copy_to_clipboard(self) -> None:
        data = export_bytes(self.graph, "png", 300)
        QApplication.clipboard().setImage(QImage.fromData(data, "PNG"))
        self.statusBar().showMessage("Graph copied to the clipboard (300 DPI).", 4000)

    def output_section(self):
        from rsstatchem.core.output import Section
        section = Section(f"Graph: {self.graph.title}")
        spec = self.graph.meta.get("spec", {})
        syntax = "GRAPH /STUDIO"
        if spec:
            syntax += f" /TYPE={spec.get('kind', '').upper()}"
            if spec.get("x"):
                syntax += f" /X={spec['x']}"
            if spec.get("ys"):
                syntax += " /Y=" + " ".join(spec["ys"])
            if spec.get("group"):
                syntax += f" /BY={spec['group']}"
        section.log(syntax + ".")
        width = int(round(self.graph.page.width_cm / 2.54 * 96))
        section.image(self.graph.title, png_bytes(self.graph, dpi=192),
                      width=max(300, min(900, width)))
        return section

    def send_to_output(self) -> None:
        ctx = getattr(self.editor, "ctx", None)
        if ctx is None:
            QMessageBox.information(self, "Output Viewer", "No Output Viewer is available.")
            return
        ctx.show_output(self.output_section())

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.modified:
            answer = QMessageBox.question(
                self, "Graph Studio", f"Save changes to {self.graph.title}?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel)
            if answer == QMessageBox.StandardButton.Cancel or \
                    (answer == QMessageBox.StandardButton.Save and not self.save()):
                event.ignore()
                return
        if self in WINDOWS:
            WINDOWS.remove(self)
        super().closeEvent(event)


def show_graph(graph: Graph, editor=None, path: str | None = None) -> GraphWindow:
    win = GraphWindow(graph, editor, path)
    win.show()
    win.raise_()
    win.activateWindow()
    return win


def open_graph_dialog(editor=None, parent=None) -> GraphWindow | None:
    ctx = getattr(editor, "ctx", None)
    settings = getattr(ctx, "settings", None)
    start = str(settings.value("last_dir", str(Path.home()))) if settings is not None \
        else str(Path.home())
    path, _ = QFileDialog.getOpenFileName(parent or editor, "Open Graph", start,
                                          f"RS-StatChem Graph (*{EXTENSION});;All files (*)")
    if not path:
        return None
    try:
        graph = load_graph(path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        QMessageBox.critical(parent or editor, "Open Graph", f"Could not open graph:\n{exc}")
        return None
    if settings is not None:
        settings.setValue("last_dir", str(Path(path).parent))
    return show_graph(graph, editor, path)

"""Graph window inside the Origin workspace.

Reuses the Graph Studio window (:class:`rsstatchem.ui.graph.window.GraphWindow`) for the page
canvas, picking, double-click property dialogs, annotations, layers, undo and export, but as
an MDI child: its own menu bar, toolbar, docks and status bar are hidden — the Origin main
window provides the menus, the Tools toolbar, the Object Manager (it hosts this window's
object tree) and the status bar (data reader). Changes mark the project as modified instead
of prompting to save a separate ``.rsg`` file.

Double-clicking a text, an axis title or a layer title edits it in place (Origin's in-place
text editing) with the main window's Format toolbar active; the Text tool also places a new
text and opens it for editing. Dragging a 3D layer rotates it.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QFont, QKeySequence
from PySide6.QtWidgets import QLineEdit, QToolBar, QWidget

from rsstatchem.core.graph import edit
from rsstatchem.core.graph.model import Graph
from rsstatchem.core.graph.text import resolve_family
from rsstatchem.ui.graph import window as gw
from rsstatchem.ui.origin.format_bar import text_target


class _TextEditor(QLineEdit):
    """In-place editor over the graph canvas: Enter commits, Esc cancels."""

    def __init__(self, gwin: "OriginGraphWindow"):
        super().__init__(gwin.canvas)
        self.gwin = gwin
        self.setObjectName("graphTextEditor")
        self.setStyleSheet("QLineEdit#graphTextEditor { border: 1px dashed #1f5fbf; "
                           "background: #ffffff; padding: 1px 3px; }")
        self.returnPressed.connect(gwin.commit_text)
        self.hide()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.gwin.cancel_text()
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event) -> None:  # noqa: N802
        super().focusOutEvent(event)
        # the Format toolbar takes the focus while it is used: commit only when the focus
        # went somewhere else
        QTimer.singleShot(150, self, self.gwin.commit_if_unfocused)   # dies with the editor


class OriginGraphWindow(gw.GraphWindow):
    main = None   # the Origin main window (set after construction)
    gadget = None  # active ROI gadget tool (ui/origin/gadgets_ui.py)

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
        self.text_editor = _TextEditor(self)
        self._editing = None   # (ref, owner, text path) while editing text in place

    @staticmethod
    def _plot_text(p) -> str:
        """Object Manager entry: the plot name and its data reference (Origin shows the
        range, e.g. ``Absorbance  [Book1]Sheet1!B[3:10]``)."""
        rng = (p.options.get("source") or {})
        text = p.name or "(unnamed)"
        if rng:
            from rsstatchem.ui.origin.plotting import plot_range
            ref = plot_range(p, "y")
            if ref:
                return f"{text}  {ref}"
        return f"{text}  [{p.type.replace('_', ' + ')}]"

    def rebuild_tree(self) -> None:
        super().rebuild_tree()
        for item in self._iter_items():   # the full data reference as a tooltip
            ref = item.data(0, gw.REF_ROLE)
            if ref and ref[0] == "plot":
                p = self.resolve(ref)
                if p is not None and p.options.get("data_range"):
                    item.setToolTip(0, p.options["data_range"])

    # ------------------------------------------------------------------ in-place text
    def _artist_rect(self, ref: tuple):
        """Widget rectangle (x, y, w, h) of the drawn artist of ``ref`` on the canvas."""
        if self.rmap is None:
            return None
        renderer = self.canvas.get_renderer()
        ratio = self.canvas.device_pixel_ratio or 1
        h_px = self.figure.bbox.height
        for artist, r in self.rmap.artists:
            if r != ref:
                continue
            try:
                if ref[0] == "axis":
                    label = artist.label if hasattr(artist, "label") else None
                    bb = (label or artist).get_window_extent(renderer)
                else:
                    bb = artist.get_window_extent(renderer)
            except Exception:  # noqa: BLE001 - some artists have no extent
                continue
            return (bb.x0 / ratio, (h_px - bb.y1) / ratio, bb.width / ratio,
                    bb.height / ratio)
        return None

    def edit_text(self, ref: tuple, select_all: bool = True) -> bool:
        """Start editing the text of ``ref`` (annotation, axis title, layer title) in place."""
        tgt = text_target(self.graph, ref)
        if tgt is None or tgt[2] is None:
            return False
        owner, font_path, text_path = tgt
        self.select(ref)
        self._editing = (ref, owner, text_path)
        ed = self.text_editor
        ed.setText(getattr(owner, text_path) or "")
        self.style_editor()
        rect = self._artist_rect(ref)
        cw, ch = self.canvas.width(), self.canvas.height()
        if rect is None or rect[2] <= 0:
            rect = (cw / 2 - 80, ch / 2 - 12, 160, 24)
        w = int(max(140, min(cw - 4, rect[2] + 60)))
        h = int(max(22, ed.sizeHint().height()))
        x = int(max(2, min(cw - w - 2, rect[0] - 6)))
        y = int(max(2, min(ch - h - 2, rect[1] + rect[3] / 2 - h / 2)))
        ed.setGeometry(x, y, w, h)
        ed.show()
        ed.raise_()
        ed.setFocus(Qt.FocusReason.OtherFocusReason)
        if select_all:
            ed.selectAll()
        if self.main is not None:
            self.main.format_bar.sync()
        return True

    def style_editor(self) -> None:
        """Show the editor in the font of the edited object."""
        if self._editing is None:
            return
        ref, owner, _ = self._editing
        tgt = text_target(self.graph, ref)
        if tgt is None:
            return
        font = getattr(tgt[0], tgt[1])
        qf = QFont(resolve_family(font.family or self.graph.font_family))
        qf.setPointSizeF(max(6.0, min(28.0, font.size * self.zoom * 1.1)))
        qf.setBold(font.bold)
        qf.setItalic(font.italic)
        self.text_editor.setFont(qf)

    def commit_text(self) -> None:
        if self._editing is None:
            return
        ref, owner, text_path = self._editing
        self._editing = None
        text = self.text_editor.text()
        self.text_editor.hide()
        if text != (getattr(owner, text_path) or ""):
            if not text.strip() and ref[0] == "annotation":
                self.selected = ref
                self.delete_selected()
            else:
                self.apply_change(owner, text_path, text)
                self.redraw(rebuild_tree=True)
        self.canvas.setFocus()
        if self.main is not None:
            self.main.format_bar.sync()

    def cancel_text(self) -> None:
        self._editing = None
        self.text_editor.hide()
        self.canvas.setFocus()
        if self.main is not None:
            self.main.format_bar.sync()

    def commit_if_unfocused(self) -> None:
        if self._editing is None:
            return
        from PySide6.QtWidgets import QApplication
        focus = QApplication.focusWidget()
        bar = getattr(self.main, "format_bar", None)
        if focus is self.text_editor or (bar is not None and focus is not None and
                                         (focus is bar or bar.isAncestorOf(focus))):
            return
        self.commit_text()

    def set_tool(self, tool) -> None:
        if self.gadget is not None:
            self.gadget = None
        super().set_tool(tool)

    def _on_motion(self, event) -> None:
        if self.gadget is not None:
            self.gadget.motion(event)
            return
        super()._on_motion(event)

    def _on_release(self, event) -> None:
        if self.gadget is not None:
            self.gadget.release(event)
            return
        super()._on_release(event)

    def _on_press(self, event) -> None:
        if self.gadget is not None:
            self.gadget.press(event)
            return
        if self._editing is not None:
            self.commit_text()
        if event.dblclick and event.button == 1 and not self._place and \
                self.nav.mode.name not in ("ZOOM", "PAN"):
            hit = self.hit_test(event)
            if hit is not None:
                tgt = text_target(self.graph, hit[0])
                if tgt is not None and tgt[2] is not None and \
                        (hit[0][0] != "axis" or self._on_axis_title(hit[0], event)):
                    self._drag = None
                    QTimer.singleShot(0, lambda r=hit[0]: self.edit_text(r))
                    return
        super()._on_press(event)

    def _on_axis_title(self, ref: tuple, event) -> bool:
        """Is the click on the axis title (not on its tick labels)?"""
        info = self.rmap.axes.get(ref[1]) if self.rmap is not None else None
        if info is None:
            return False
        ax = info["ax"]
        which = ref[2]
        mpl_axis = {"x": getattr(ax, "xaxis", None), "y": getattr(ax, "yaxis", None)}.get(
            which)
        if which == "y2" and info.get("ax2") is not None:
            mpl_axis = info["ax2"].yaxis
        if which == "z":
            mpl_axis = getattr(ax, "zaxis", None)
        label = getattr(mpl_axis, "label", None)
        if label is None or not label.get_text():
            return False
        try:
            return label.get_window_extent(self.canvas.get_renderer()).expanded(
                1.1, 1.3).contains(event.x, event.y)
        except Exception:  # noqa: BLE001
            return False

    def _finish_placement(self, event) -> None:
        if self._place != "text":
            super()._finish_placement(event)
            return
        x0, y0, li = self._place_start
        self._place_start = None
        info = self.rmap.axes.get(li) if self.rmap is not None else None
        if info is None:
            return
        ax0, ay0 = info["ax"].transAxes.inverted().transform((x0, y0))
        layer = self.graph.layers[li]
        ann = edit.new_annotation("text", float(ax0), float(ay0),
                                  font_size=layer.x.label_font.size)
        ann.text = "Text"
        self.push_undo()
        layer.annotations.append(ann)
        self.selected = ("annotation", li, ann.id)
        self.set_tool(None)
        if self.main is not None:
            self.main.set_tool("pointer", quiet=True)
        self.redraw(rebuild_tree=True)
        self.edit_text(self.selected)

    def show_properties(self, ref: tuple) -> None:
        super().show_properties(ref)
        if self.main is not None and hasattr(self.main, "format_bar"):
            self.main.format_bar.sync()

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

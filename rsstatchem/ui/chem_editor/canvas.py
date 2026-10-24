"""Drawing canvas of the structure editor: a QGraphicsView whose scene is rebuilt from the
document after each change, with ChemDraw-like tools and hover hotkeys.

Hit testing works on the document model (nearest atom / bond / object), the scene only
renders. Every edit is wrapped in a snapshot undo command (``QUndoStack``), so undo and
redo restore the exact document state.

Tool ids: ``select``, ``lasso``, ``rotate``, ``eraser``, ``bond:<kind>``, ``chain``,
``ring:<template>``, ``atom:<symbol>``, ``label``, ``charge+``, ``charge-``, ``radical``,
``map``, ``arrow:<kind>``, ``plus``, ``text``, ``bracket``, ``template``.
"""
from __future__ import annotations

import math

from PySide6.QtCore import QEvent, QLineF, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (QAction, QBrush, QColor, QPainter, QPainterPath, QPen,
                           QPolygonF, QUndoCommand)
from PySide6.QtWidgets import (QGraphicsLineItem, QGraphicsPathItem, QGraphicsRectItem,
                               QGraphicsScene, QGraphicsView, QLineEdit, QMenu,
                               QPlainTextEdit)

from rsstatchem.core.chem import molecule as mo
from rsstatchem.core.chem_draw import abbrev, convert, ops
from rsstatchem.core.chem_draw import geometry as geo
from rsstatchem.core.chem_draw.model import Document
from rsstatchem.ui.chem_editor.render import SELECT_COLOR, BuildInfo, Builder

__all__ = ["ChemCanvas", "SnapshotCommand"]

_HOTKEY_ELEMENTS = {"c": "C", "n": "N", "o": "O", "s": "S", "p": "P", "f": "F", "i": "I",
                    "h": "H", "l": "Cl", "b": "Br", "B": "B", "k": "K"}


class SnapshotCommand(QUndoCommand):
    """Undo step holding the document before and after an edit."""

    def __init__(self, canvas: "ChemCanvas", text: str, before: dict, after: dict):
        super().__init__(text)
        self.canvas, self.before, self.after = canvas, before, after
        self._first = True

    def redo(self) -> None:
        if self._first:
            self._first = False
            return
        self.canvas.set_state(self.after)

    def undo(self) -> None:
        self.canvas.set_state(self.before)


class _LabelEdit(QLineEdit):
    cancelled = Signal()

    def keyPressEvent(self, e) -> None:  # noqa: N802
        if e.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()
            return
        super().keyPressEvent(e)


class _TextEdit(QPlainTextEdit):
    committed = Signal()
    cancelled = Signal()

    def keyPressEvent(self, e) -> None:  # noqa: N802
        if e.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()
            return
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and \
                e.modifiers() & (Qt.KeyboardModifier.ControlModifier |
                                 Qt.KeyboardModifier.MetaModifier):
            self.committed.emit()
            return
        super().keyPressEvent(e)

    def focusOutEvent(self, e) -> None:  # noqa: N802
        super().focusOutEvent(e)
        self.committed.emit()


class ChemCanvas(QGraphicsView):
    doc_changed = Signal()
    selection_changed = Signal()
    status_message = Signal(str)
    tool_changed = Signal(str)

    def __init__(self, doc: Document, undo_stack, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.scene().setSceneRect(-4000, -4000, 10000, 10000)
        self.setRenderHints(QPainter.RenderHint.Antialiasing |
                            QPainter.RenderHint.TextAntialiasing)
        self.setMouseTracking(True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QBrush(QColor("#FFFFFF")))
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.doc = doc
        self.undo_stack = undo_stack
        self.tool = "bond:single"
        self.selection: set = set()
        self.hover = None
        self.info = BuildInfo()
        self.problems: dict[int, str] = {}
        self.show_grid = False
        self.template_smiles = "c1ccccc1"
        self._overlay: list = []
        self._preview: list = []
        self._press = None
        self._press_hit = None
        self._press_screen = None
        self._dragging = False
        self._mode = ""
        self._before = None
        self._last = None
        self._lasso: list = []
        self._editor = None
        self._edit_target = None
        self._rot = None
        self.scale(1.6, 1.6)
        self.rebuild()

    # --- state / undo ----------------------------------------------------------------------
    @property
    def bl(self) -> float:
        return self.doc.style.bond_length

    def zoom(self) -> float:
        return self.transform().m11()

    def begin(self) -> None:
        self._before = self.doc.to_dict()

    def commit(self, text: str) -> bool:
        before, self._before = self._before, None
        if before is None:
            self.rebuild()
            return False
        after = self.doc.to_dict()
        if after == before:
            self.rebuild()
            return False
        self.undo_stack.push(SnapshotCommand(self, text, before, after))
        self._prune_selection()
        self.rebuild()
        self.doc_changed.emit()
        return True

    def edit(self, text: str, fn) -> object:
        """Run ``fn()`` as one undoable edit."""
        self.begin()
        try:
            result = fn()
        except Exception:
            self.set_state(self._before)
            self._before = None
            raise
        self.commit(text)
        return result

    def set_state(self, data: dict) -> None:
        new = Document.from_dict(data)
        d = self.doc
        d.style, d.meta, d._next = new.style, new.meta, new._next
        for kind in ("atoms", "bonds", "arrows", "pluses", "texts", "brackets"):
            setattr(d, kind, getattr(new, kind))
        self._prune_selection()
        self.rebuild()
        self.doc_changed.emit()

    def _prune_selection(self) -> None:
        sel = {k for k in self.selection if self.doc.get(k) is not None}
        if sel != self.selection:
            self.selection = sel
            self.selection_changed.emit()
        if self.hover is not None and self.doc.get(self.hover) is None:
            self.hover = None

    # --- rendering -----------------------------------------------------------------------
    def rebuild(self) -> None:
        self._overlay, self._preview = [], []
        self.scene().clear()
        self.info = Builder(self.doc.style).build(self.scene(), self.doc)
        try:
            self.problems = convert.structure_problems(self.doc)
        except Exception:  # noqa: BLE001 - feedback must never break drawing
            self.problems = {}
        self.update_overlays()

    def _add(self, item, z: float, store: list) -> None:
        item.setZValue(z)
        self.scene().addItem(item)
        store.append(item)

    def _clear(self, store: list) -> None:
        for it in store:
            if it.scene() is not None:
                self.scene().removeItem(it)
        store.clear()

    def update_overlays(self) -> None:
        self._clear(self._overlay)
        bl = self.bl
        sel_brush = QColor(SELECT_COLOR)
        sel_brush.setAlpha(70)
        for key in self.doc.with_bonds(self.selection) if self.selection else ():
            self._mark(key, sel_brush, None, 0.5)
        if self.hover is not None and self.hover not in self.selection:
            hov = QColor(SELECT_COLOR)
            hov.setAlpha(40)
            self._mark(self.hover, hov, QPen(QColor(SELECT_COLOR), 1.0 / self.zoom()), 0.6)
        red = QPen(QColor("#E53935"), max(0.8, 1.2 / self.zoom()))
        for aid in self.problems:
            a = self.doc.atoms.get(aid)
            if a is None:
                continue
            box = self.info.label_boxes.get(aid) or QRectF(a.x - bl * 0.22, a.y - bl * 0.22,
                                                           bl * 0.44, bl * 0.44)
            item = QGraphicsRectItem(box.adjusted(-1, -1, 1, 1))
            item.setPen(red)
            self._add(item, 9, self._overlay)
        if len(self.selection) > 1:
            r = self._selection_rect()
            if r is not None:
                item = QGraphicsRectItem(r)
                pen = QPen(QColor(SELECT_COLOR), 1.0 / self.zoom(), Qt.PenStyle.DashLine)
                item.setPen(pen)
                self._add(item, 9, self._overlay)

    def _selection_rect(self) -> QRectF | None:
        keys = self.doc.with_bonds(self.selection)
        rects = []
        for k, i in keys:
            if k == "atom":
                a = self.doc.atoms[i]
                rects.append(self.info.label_boxes.get(i) or QRectF(a.x - 2, a.y - 2, 4, 4))
            elif k == "text" and i in self.info.text_boxes:
                rects.append(self.info.text_boxes[i])
            elif k == "plus" and i in self.info.plus_boxes:
                rects.append(self.info.plus_boxes[i])
            elif k == "bracket" and i in self.info.bracket_boxes:
                rects.append(self.info.bracket_boxes[i])
            elif k == "arrow" and i in self.doc.arrows:
                ar = self.doc.arrows[i]
                rects.append(QRectF(QPointF(ar.x1, ar.y1), QPointF(ar.x2, ar.y2)).normalized())
        if not rects:
            return None
        r = rects[0]
        for x in rects[1:]:
            r = r.united(x)
        m = self.bl * 0.25
        return r.adjusted(-m, -m, m, m)

    def _mark(self, key, brush: QColor, pen, z: float) -> None:
        kind, i = key
        bl = self.bl
        obj = self.doc.get(key)
        if obj is None:
            return
        if kind == "atom":
            box = self.info.label_boxes.get(i) or QRectF(obj.x - bl * 0.16, obj.y - bl * 0.16,
                                                         bl * 0.32, bl * 0.32)
            item = QGraphicsRectItem(box)
            item.setBrush(QBrush(brush))
            item.setPen(pen or QPen(Qt.PenStyle.NoPen))
        elif kind == "bond":
            a, b = self.doc.atoms[obj.a], self.doc.atoms[obj.b]
            item = QGraphicsLineItem(a.x, a.y, b.x, b.y)
            p = QPen(brush, bl * 0.22)
            p.setCapStyle(Qt.PenCapStyle.RoundCap)
            item.setPen(p)
        elif kind == "arrow":
            pts = geo.arrow_path_points(obj)
            path = QPainterPath(QPointF(*pts[0]))
            for x, y in pts[1:]:
                path.lineTo(x, y)
            item = QGraphicsPathItem(path)
            p = QPen(brush, bl * 0.25)
            p.setCapStyle(Qt.PenCapStyle.RoundCap)
            item.setPen(p)
            for x, y in ((obj.x1, obj.y1), (obj.x2, obj.y2)):
                h = QGraphicsRectItem(x - 2.5 / self.zoom(), y - 2.5 / self.zoom(),
                                      5 / self.zoom(), 5 / self.zoom())
                h.setBrush(QBrush(QColor(SELECT_COLOR)))
                h.setPen(QPen(Qt.PenStyle.NoPen))
                self._add(h, z + 8, self._overlay)
        else:
            box = {"text": self.info.text_boxes, "plus": self.info.plus_boxes,
                   "bracket": self.info.bracket_boxes}[kind].get(i)
            if box is None:
                return
            item = QGraphicsRectItem(box.adjusted(-2, -2, 2, 2))
            item.setBrush(QBrush(brush))
            item.setPen(pen or QPen(Qt.PenStyle.NoPen))
        self._add(item, z, self._overlay)

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802
        super().drawBackground(painter, rect)
        if not self.show_grid:
            return
        step = self.bl
        pen = QPen(QColor(225, 230, 238), 0)
        painter.setPen(pen)
        x = math.floor(rect.left() / step) * step
        while x < rect.right():
            painter.drawLine(QLineF(x, rect.top(), x, rect.bottom()))
            x += step
        y = math.floor(rect.top() / step) * step
        while y < rect.bottom():
            painter.drawLine(QLineF(rect.left(), y, rect.right(), y))
            y += step

    # --- tools ---------------------------------------------------------------------------
    def set_tool(self, tool: str) -> None:
        self.tool = tool
        self._cancel_drag()
        cursor = Qt.CursorShape.ArrowCursor if tool in ("select", "lasso") else \
            Qt.CursorShape.CrossCursor
        self.viewport().setCursor(cursor)
        self.tool_changed.emit(tool)

    def hit(self, p: QPointF, kinds=None):
        x, y = p.x(), p.y()
        z = self.zoom()
        bl = self.bl
        kinds = kinds or ("atom", "bond", "arrow", "text", "plus", "bracket")
        if "atom" in kinds:
            a = self.doc.atom_at(x, y, max(bl * 0.3, 7 / z))
            if a is not None:
                return ("atom", a.id)
            for aid, box in self.info.label_boxes.items():
                if box.contains(p):
                    return ("atom", aid)
        if "text" in kinds:
            for tid, box in self.info.text_boxes.items():
                if box.adjusted(-2, -2, 2, 2).contains(p):
                    return ("text", tid)
        if "bond" in kinds:
            b = self.doc.bond_at(x, y, max(bl * 0.18, 5 / z))
            if b is not None:
                return ("bond", b.id)
        if "arrow" in kinds:
            for aid, rects in self.info.arrow_text.items():
                if any(r.contains(p) for r in rects):
                    return ("arrow", aid)
            ar = self.doc.arrow_at(x, y, max(bl * 0.2, 5 / z))
            if ar is not None:
                return ("arrow", ar.id)
        if "plus" in kinds:
            for pid, box in self.info.plus_boxes.items():
                if box.adjusted(-3, -3, 3, 3).contains(p):
                    return ("plus", pid)
        if "bracket" in kinds:
            for bid, br in self.doc.brackets.items():
                tol = max(3.0, 5 / z)
                for xx in (br.x1, br.x2):
                    if abs(x - xx) < tol * 1.5 and br.y1 - tol <= y <= br.y2 + tol:
                        return ("bracket", bid)
        return None

    # --- mouse ---------------------------------------------------------------------------
    def mousePressEvent(self, e) -> None:  # noqa: N802
        if self._editor is not None:
            self._finish_editor()
        if e.button() == Qt.MouseButton.MiddleButton:
            super().mousePressEvent(e)
            return
        if e.button() != Qt.MouseButton.LeftButton:
            return
        p = self.mapToScene(e.position().toPoint())
        self._press, self._last = p, p
        self._press_screen = e.position()
        self._press_hit = self.hit(p)
        self._dragging = False
        self._mode = ""
        self._mods = e.modifiers()
        self.setFocus()

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        p = self.mapToScene(e.position().toPoint())
        if not (e.buttons() & Qt.MouseButton.LeftButton) or self._press is None:
            h = self.hit(p)
            if h != self.hover:
                self.hover = h
                self.update_overlays()
                self._describe(h)
            return
        if not self._dragging:
            d = e.position() - self._press_screen
            if math.hypot(d.x(), d.y()) < 4:
                return
            self._dragging = True
            self._drag_start(self._press)
        self._drag_move(p, e.modifiers())
        self._last = p

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if e.button() != Qt.MouseButton.LeftButton or self._press is None:
            return
        p = self.mapToScene(e.position().toPoint())
        try:
            if self._dragging:
                self._drag_end(p, e.modifiers())
            else:
                self.click(self._press, self._press_hit, e.modifiers())
        finally:
            self._press = None
            self._dragging = False
            self._clear(self._preview)

    def mouseDoubleClickEvent(self, e) -> None:  # noqa: N802
        p = self.mapToScene(e.position().toPoint())
        h = self.hit(p)
        if h is None:
            return
        kind, i = h
        if kind == "text":
            self.edit_text(i)
        elif kind == "arrow" and self.tool in ("select", "lasso"):
            self.arrow_properties(i)
        elif kind == "atom" and self.tool not in ("select", "lasso", "rotate"):
            self.edit_label(i)
        elif kind == "atom" and self.tool in ("select", "lasso"):
            if e.modifiers() & Qt.KeyboardModifier.AltModifier:
                self.atom_properties(i)
            else:
                self.select_keys(self._fragment_keys(i))
        elif kind == "bond":
            self.select_keys(self._fragment_keys(self.doc.bonds[i].a))

    def wheelEvent(self, e) -> None:  # noqa: N802
        if e.modifiers() & (Qt.KeyboardModifier.ControlModifier |
                            Qt.KeyboardModifier.MetaModifier):
            f = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
            self.set_zoom(self.zoom() * f)
            return
        super().wheelEvent(e)

    def contextMenuEvent(self, e) -> None:  # noqa: N802
        p = self.mapToScene(e.pos())
        h = self.hit(p)
        menu = self.context_menu(h, p)
        if menu is not None:
            menu.exec(e.globalPos())

    # --- clicks ---------------------------------------------------------------------------
    def click(self, p: QPointF, h, mods=Qt.KeyboardModifier.NoModifier) -> None:
        t = self.tool
        x, y = p.x(), p.y()
        kind, i = h if h else (None, None)
        if t in ("select", "lasso", "rotate"):
            if h is None:
                if not (mods & Qt.KeyboardModifier.ShiftModifier):
                    self.select_keys(set())
                return
            keys = self.doc.group_members({h})
            if mods & Qt.KeyboardModifier.ShiftModifier:
                sel = set(self.selection)
                sel = sel - keys if keys <= sel else sel | keys
                self.select_keys(sel)
            else:
                self.select_keys(keys)
            return
        if t == "eraser":
            if h is not None:
                self.edit("Erase", lambda: self.erase([h]))
            return
        if t.startswith("bond:") or t == "chain":
            tool = t.split(":", 1)[1] if t.startswith("bond:") else "single"
            order, style = ops.BOND_TOOLS[tool]
            if kind == "atom":
                self.edit("Add Bond", lambda: ops.add_bond_at_atom(self.doc, i, order, style))
            elif kind == "bond":
                b = self.doc.bonds[i]
                near = min((b.a, b.b), key=lambda a: math.hypot(self.doc.atoms[a].x - x,
                                                                 self.doc.atoms[a].y - y))
                self.edit("Change Bond", lambda: ops.apply_bond_tool(self.doc, i, tool, near))
            elif h is None:
                self.edit("Add Bond", lambda: ops.add_bond_free(self.doc, x, y, order, style))
            return
        if t.startswith("ring:"):
            tpl = t.split(":", 1)[1]
            if kind == "atom":
                self.edit("Add Ring", lambda: ops.stamp_ring(self.doc, tpl, atom=i))
            elif kind == "bond":
                self.edit("Add Ring", lambda: ops.stamp_ring(self.doc, tpl, bond=i,
                                                             pointer=(x, y)))
            elif h is None:
                self.edit("Add Ring", lambda: ops.stamp_ring(self.doc, tpl, point=(x, y)))
            return
        if t.startswith("atom:"):
            sym = t.split(":", 1)[1]
            if kind == "atom":
                self.edit("Change Atom", lambda: ops.set_element(self.doc, i, sym))
            elif h is None:
                self.edit("Add Atom", lambda: self.doc.add_atom(x, y, sym))
            return
        if t == "label":
            if kind == "atom":
                self.edit_label(i)
            elif h is None:
                aid = self.edit("Add Atom", lambda: self.doc.add_atom(x, y).id)
                self.edit_label(aid)
            return
        if t in ("charge+", "charge-", "radical", "map") and kind == "atom":
            fn = {"charge+": lambda: ops.change_charge(self.doc, i, 1),
                  "charge-": lambda: ops.change_charge(self.doc, i, -1),
                  "radical": lambda: ops.cycle_radical(self.doc, i),
                  "map": lambda: ops.toggle_map_number(self.doc, i)}[t]
            self.edit({"map": "Atom Map"}.get(t, "Change Atom"), fn)
            return
        if t.startswith("arrow:"):
            ak = t.split(":", 1)[1]
            if kind == "arrow":
                self.edit("Change Arrow", lambda: setattr(self.doc.arrows[i], "kind", ak))
                return
            if h is None:
                L = self.bl * (2.0 if ak in ("curved", "fishhook") else 3.0)
                bend = -L * 0.35 if ak in ("curved", "fishhook") else 0.0
                self.edit("Add Arrow", lambda: self.doc.add_arrow(x, y, x + L, y, ak,
                                                                  bend=bend))
            return
        if t == "plus" and h is None:
            self.edit("Add Plus", lambda: self.doc.add_plus(x, y))
            return
        if t == "text":
            if kind == "text":
                self.edit_text(i)
            elif kind == "arrow":
                self.arrow_properties(i)
            elif h is None:
                tid = self.edit("Add Text", lambda: self.doc.add_text(x, y, "").id)
                self.edit_text(tid, new=True)
            return
        if t == "bracket" and self.selection:
            r = self._selection_rect()
            if r is not None:
                self.edit("Add Brackets", lambda: self.doc.add_bracket(
                    r.left(), r.top(), r.right(), r.bottom()))
            return
        if t == "template" and h is None and self.template_smiles:
            self.insert_text(self.template_smiles, x, y, "Insert Template")

    def erase(self, keys) -> None:
        """ChemDraw eraser: delete objects; carbons left without bonds disappear too."""
        doc = self.doc
        keys = list(keys)
        touched = set()
        for k, i in keys:
            if k == "atom" and i in doc.atoms:
                touched.update(doc.neighbors(i))
            elif k == "bond" and i in doc.bonds:
                touched.update((doc.bonds[i].a, doc.bonds[i].b))
        doc.remove(keys)
        doc.remove([("atom", n) for n in touched if n in doc.atoms and not doc.bonds_of(n)
                    and doc.atoms[n].symbol == "C" and not doc.atoms[n].label
                    and not doc.atoms[n].charge and not doc.atoms[n].isotope])

    # --- drags ---------------------------------------------------------------------------
    def _drag_start(self, p: QPointF) -> None:
        t = self.tool
        h = self._press_hit
        self.begin()
        if t in ("select", "lasso"):
            end = self._arrow_handle(p)
            if end is not None:
                self._mode, self._handle = "arrow-handle", end
                return
            if h is not None:
                if h not in self.selection:
                    self.select_keys(self.doc.group_members({h}))
                self._mode = "move"
                return
            self._mode = "lasso" if t == "lasso" else "marquee"
            self._lasso = [p]
            return
        if t == "rotate":
            keys = set(self.selection)
            if not keys and h is not None:
                keys = self._fragment_keys(h[1] if h[0] == "atom" else
                                           self.doc.bonds[h[1]].a) if h[0] in ("atom", "bond") \
                    else {h}
            if not keys:
                keys = set(self.doc.keys())
            self._rot = (self.doc.with_bonds(keys), self.doc.center(self.doc.with_bonds(keys)),
                         self.doc.to_dict())
            self._mode = "rotate"
            return
        if t == "eraser":
            self._mode = "erase"
            if h is not None:
                self.erase([h])
                self.rebuild()
            return
        if t.startswith("bond:") or t == "chain":
            self._mode = "bond" if t != "chain" else "chain"
            return
        if t.startswith("arrow:"):
            self._mode = "arrow"
            return
        if t == "bracket":
            self._mode = "bracket"
            return
        self._mode = "click"

    def _start_point(self) -> tuple[float, float, int | None]:
        h = self._press_hit
        if h and h[0] == "atom":
            a = self.doc.atoms[h[1]]
            return a.x, a.y, a.id
        return self._press.x(), self._press.y(), None

    def _bond_end(self, p: QPointF, sx: float, sy: float, start_id, free: bool):
        target = self.doc.atom_at(p.x(), p.y(), max(self.bl * 0.3, 7 / self.zoom()),
                                  exclude={start_id} if start_id else ())
        if target is not None:
            return target.x, target.y, target.id
        ang = math.atan2(p.y() - sy, p.x() - sx)
        if not free:
            ang = geo.snap_angle(ang, 15)
        ex, ey = geo.bond_end_point(sx, sy, ang, self.bl)
        return ex, ey, None

    def _drag_move(self, p: QPointF, mods) -> None:
        m = self._mode
        free = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        if m == "move":
            d = p - self._last
            self.doc.move(self.doc.with_bonds(self.selection), d.x(), d.y())
            self.rebuild()
        elif m == "arrow-handle":
            ar = self.doc.arrows[self._handle[0]]
            if self._handle[1] == "tail":
                ar.x1, ar.y1 = p.x(), p.y()
            elif self._handle[1] == "head":
                ar.x2, ar.y2 = p.x(), p.y()
            else:
                dx, dy = ar.x2 - ar.x1, ar.y2 - ar.y1
                L = math.hypot(dx, dy) or 1.0
                mx, my = (ar.x1 + ar.x2) / 2, (ar.y1 + ar.y2) / 2
                ar.bend = ((p.x() - mx) * -dy + (p.y() - my) * dx) / L
            self.rebuild()
        elif m in ("marquee", "lasso"):
            self._lasso.append(p)
            self._clear(self._preview)
            pen = QPen(QColor(SELECT_COLOR), 1 / self.zoom(), Qt.PenStyle.DashLine)
            if m == "marquee":
                item = QGraphicsRectItem(QRectF(self._press, p).normalized())
            else:
                path = QPainterPath(self._lasso[0])
                for q in self._lasso[1:]:
                    path.lineTo(q)
                item = QGraphicsPathItem(path)
            item.setPen(pen)
            self._add(item, 20, self._preview)
        elif m == "rotate":
            keys, (cx, cy), snap = self._rot
            a0 = math.atan2(self._press.y() - cy, self._press.x() - cx)
            a1 = math.atan2(p.y() - cy, p.x() - cx)
            ang = a1 - a0
            if free:
                ang = geo.snap_angle(ang, 15)
            self.set_state_quiet(snap)
            self.doc.rotate(keys, ang, cx, cy)
            self.rebuild()
            self.status_message.emit(f"Rotate {math.degrees(ang):.0f}°")
        elif m == "erase":
            h = self.hit(p)
            if h is not None:
                self.erase([h])
                self.rebuild()
        elif m == "bond":
            sx, sy, sid = self._start_point()
            ex, ey, _ = self._bond_end(p, sx, sy, sid, free)
            self._preview_line([(sx, sy), (ex, ey)])
        elif m == "chain":
            sx, sy, sid = self._start_point()
            pts = self._chain_preview(p, sx, sy, sid)
            self._preview_line([(sx, sy)] + pts)
            self.status_message.emit(f"Chain: {len(pts)} bonds")
        elif m == "arrow":
            sx, sy = self._press.x(), self._press.y()
            ex, ey = self._arrow_end(p, free)
            self._preview_line([(sx, sy), (ex, ey)])
        elif m == "bracket":
            self._clear(self._preview)
            item = QGraphicsRectItem(QRectF(self._press, p).normalized())
            item.setPen(QPen(QColor(SELECT_COLOR), 1 / self.zoom(), Qt.PenStyle.DashLine))
            self._add(item, 20, self._preview)

    def set_state_quiet(self, data: dict) -> None:
        new = Document.from_dict(data)
        for kind in ("atoms", "bonds", "arrows", "pluses", "texts", "brackets"):
            setattr(self.doc, kind, getattr(new, kind))

    def _arrow_end(self, p: QPointF, free: bool) -> tuple[float, float]:
        sx, sy = self._press.x(), self._press.y()
        if free or self.tool in ("arrow:curved", "arrow:fishhook"):
            return p.x(), p.y()
        ang = geo.snap_angle(math.atan2(p.y() - sy, p.x() - sx), 15)
        L = math.hypot(p.x() - sx, p.y() - sy)
        return sx + L * math.cos(ang), sy + L * math.sin(ang)

    def _chain_preview(self, p: QPointF, sx, sy, sid) -> list:
        ang = geo.snap_angle(math.atan2(p.y() - sy, p.x() - sx), 15)
        dist = math.hypot(p.x() - sx, p.y() - sy)
        n = max(1, int(round(dist / (self.bl * math.cos(geo.D30)))))
        sign = 1
        if sid is not None and self.doc.neighbors(sid):
            o = self.doc.atoms[self.doc.neighbors(sid)[0]]
            back = math.atan2(o.y - sy, o.x - sx)
            sign = max((1, -1), key=lambda s: abs(math.remainder(ang - s * geo.D30 - back,
                                                                   2 * math.pi)))
        self._chain = (ang, n, sign)
        return geo.chain_points(sx, sy, ang, n, self.bl, sign)

    def _preview_line(self, pts) -> None:
        self._clear(self._preview)
        path = QPainterPath(QPointF(*pts[0]))
        for x, y in pts[1:]:
            path.lineTo(x, y)
        item = QGraphicsPathItem(path)
        pen = QPen(QColor(SELECT_COLOR), max(self.doc.style.line_width, 1 / self.zoom()))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        item.setPen(pen)
        self._add(item, 20, self._preview)

    def _drag_end(self, p: QPointF, mods) -> None:
        m = self._mode
        free = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        doc = self.doc
        if m == "move":
            moved = doc.atoms_of(doc.with_bonds(self.selection))
            doc.merge_coincident(moved, self.bl * 0.2)
            self.commit("Move")
        elif m == "arrow-handle":
            self.commit("Edit Arrow")
        elif m in ("marquee", "lasso"):
            self._before = None
            self._select_region(p, m, mods)
        elif m == "rotate":
            self.commit("Rotate")
        elif m == "erase":
            self.commit("Erase")
        elif m == "bond":
            sx, sy, sid = self._start_point()
            ex, ey, eid = self._bond_end(p, sx, sy, sid, free)
            tool = self.tool.split(":", 1)[1]
            order, style = ops.BOND_TOOLS[tool]
            start = sid if sid is not None else (sx, sy)
            end = eid if eid is not None else (ex, ey)
            ops.add_bond_between(doc, start, end, order, style)
            self.commit("Add Bond")
        elif m == "chain":
            sx, sy, sid = self._start_point()
            self._chain_preview(p, sx, sy, sid)
            ang, n, sign = self._chain
            ops.add_chain(doc, sid if sid is not None else (sx, sy), ang, n, sign)
            self.commit("Add Chain")
        elif m == "arrow":
            ex, ey = self._arrow_end(p, free)
            ak = self.tool.split(":", 1)[1]
            L = math.hypot(ex - self._press.x(), ey - self._press.y())
            bend = -L * 0.35 if ak in ("curved", "fishhook") else 0.0
            doc.add_arrow(self._press.x(), self._press.y(), ex, ey, ak, bend=bend)
            self.commit("Add Arrow")
        elif m == "bracket":
            r = QRectF(self._press, p).normalized()
            doc.add_bracket(r.left(), r.top(), r.right(), r.bottom())
            self.commit("Add Brackets")
        else:
            self._before = None
            self.click(self._press, self._press_hit, mods)

    def _cancel_drag(self) -> None:
        if self._dragging and self._before is not None:
            self.set_state(self._before)
        self._before = None
        self._press = None
        self._dragging = False
        self._clear(self._preview)

    def _arrow_handle(self, p: QPointF):
        tol = max(3.0, 6 / self.zoom())
        for ar in self.doc.arrows.values():
            if math.hypot(p.x() - ar.x2, p.y() - ar.y2) <= tol:
                return (ar.id, "head")
            if math.hypot(p.x() - ar.x1, p.y() - ar.y1) <= tol:
                return (ar.id, "tail")
            if ar.kind in ("curved", "fishhook") and ("arrow", ar.id) in self.selection:
                pts = geo.arrow_path_points(ar)
                mid = pts[len(pts) // 2]
                if math.hypot(p.x() - mid[0], p.y() - mid[1]) <= tol:
                    return (ar.id, "bend")
        return None

    def _select_region(self, p: QPointF, mode: str, mods) -> None:
        doc = self.doc
        if mode == "marquee":
            r = QRectF(self._press, p).normalized()

            def inside(x, y):
                return r.contains(QPointF(x, y))
        else:
            poly = QPolygonF(self._lasso + [self._lasso[0]])

            def inside(x, y):
                return poly.containsPoint(QPointF(x, y), Qt.FillRule.OddEvenFill)
        keys = set()
        for a in doc.atoms.values():
            if inside(a.x, a.y):
                keys.add(("atom", a.id))
        for b in doc.bonds.values():
            if ("atom", b.a) in keys and ("atom", b.b) in keys:
                keys.add(("bond", b.id))
        for ar in doc.arrows.values():
            if inside(ar.x1, ar.y1) and inside(ar.x2, ar.y2):
                keys.add(("arrow", ar.id))
        for pl in doc.pluses.values():
            if inside(pl.x, pl.y):
                keys.add(("plus", pl.id))
        for t in doc.texts.values():
            box = self.info.text_boxes.get(t.id)
            if box is not None and inside(box.center().x(), box.center().y()):
                keys.add(("text", t.id))
        for br in doc.brackets.values():
            if inside(br.x1, br.y1) and inside(br.x2, br.y2):
                keys.add(("bracket", br.id))
        keys = doc.group_members(keys)
        if mods & Qt.KeyboardModifier.ShiftModifier:
            keys |= self.selection
        self.select_keys(keys)

    # --- selection -------------------------------------------------------------------------
    def select_keys(self, keys) -> None:
        self.selection = set(keys)
        self.update_overlays()
        self.selection_changed.emit()

    def select_all(self) -> None:
        self.select_keys(set(self.doc.keys()))

    def _fragment_keys(self, aid: int) -> set:
        comp = self.doc.fragment_of(aid)
        return self.doc.with_bonds({("atom", a) for a in comp})

    def selected_atoms(self) -> set[int]:
        return self.doc.atoms_of(self.doc.with_bonds(self.selection))

    def target_keys(self) -> set:
        """Selection (with bonds), or everything when nothing is selected."""
        return self.doc.with_bonds(self.selection) if self.selection else set(self.doc.keys())

    # --- keyboard ------------------------------------------------------------------------
    def keyPressEvent(self, e) -> None:  # noqa: N802
        key, text = e.key(), e.text()
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if self.selection:
                self.edit("Delete", lambda: self.erase(self.doc.with_bonds(self.selection)))
            elif self.hover is not None:
                h = self.hover
                self.hover = None
                self.edit("Delete", lambda: self.erase([h]))
            return
        if key == Qt.Key.Key_Escape:
            self._cancel_drag()
            self.select_keys(set())
            return
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down) and \
                self.selection:
            step = 10.0 if e.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1.0
            dx = {Qt.Key.Key_Left: -step, Qt.Key.Key_Right: step}.get(key, 0.0)
            dy = {Qt.Key.Key_Up: -step, Qt.Key.Key_Down: step}.get(key, 0.0)
            self.edit("Nudge", lambda: self.doc.move(self.doc.with_bonds(self.selection),
                                                     dx, dy))
            return
        if self.hover is not None and self.hotkey(self.hover, text, key):
            return
        super().keyPressEvent(e)

    def hotkey(self, target, text: str, key=None) -> bool:
        """ChemDraw-style hover hotkeys on atoms and bonds."""
        kind, i = target
        if kind == "atom":
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.edit_label(i)
                return True
            if text in _HOTKEY_ELEMENTS:
                sym = _HOTKEY_ELEMENTS[text]
                self.edit("Change Atom", lambda: ops.set_element(self.doc, i, sym))
                return True
            if text in ("+", "="):
                self.edit("Charge", lambda: ops.change_charge(self.doc, i, 1))
                return True
            if text == "-":
                self.edit("Charge", lambda: ops.change_charge(self.doc, i, -1))
                return True
            if text == ".":
                self.edit("Radical", lambda: ops.cycle_radical(self.doc, i))
                return True
            if text == "1":
                self.edit("Add Bond", lambda: ops.add_bond_at_atom(self.doc, i))
                return True
        if kind == "bond":
            tools = {"1": "single", "2": "double", "3": "triple", "w": "wedge", "h": "hash",
                     "y": "wavy", "a": "aromatic"}
            if text in tools:
                tool = tools[text]

                def fn():
                    b = self.doc.bonds[i]
                    if tool in ("single", "triple"):
                        b.order, b.style = ops.BOND_TOOLS[tool]
                    else:
                        ops.apply_bond_tool(self.doc, i, tool)
                self.edit("Change Bond", fn)
                return True
        return False

    def _describe(self, h) -> None:
        if h is None:
            self.status_message.emit("")
            return
        kind, i = h
        if kind == "atom":
            a = self.doc.atoms[i]
            msg = f"Atom {a.label or a.symbol}"
            if a.charge:
                msg += f"  charge {a.charge:+d}"
            if i in self.problems:
                msg += f"  —  {self.problems[i]}"
            self.status_message.emit(msg)
        elif kind == "bond":
            b = self.doc.bonds[i]
            name = {1: "single", 2: "double", 3: "triple", 1.5: "aromatic"}.get(b.order, "")
            self.status_message.emit(f"Bond: {name} {b.style}".strip())
        else:
            self.status_message.emit(kind.capitalize())

    # --- inline editors --------------------------------------------------------------------
    def edit_label(self, aid: int) -> None:
        a = self.doc.atoms.get(aid)
        if a is None:
            return
        self._finish_editor()
        ed = _LabelEdit(self.viewport())
        ed.setText(a.label or (a.symbol if a.symbol != "C" or a.show_carbon else ""))
        ed.selectAll()
        ed.setFixedWidth(90)
        pos = self.mapFromScene(QPointF(a.x, a.y))
        ed.move(pos.x() - 12, pos.y() - ed.sizeHint().height() // 2)
        ed.setPlaceholderText("N, OH, Ph…")
        ed.returnPressed.connect(self._finish_editor)
        ed.cancelled.connect(self._cancel_editor)
        ed.show()
        ed.setFocus()
        self._editor, self._edit_target = ed, ("atom", aid)

    def edit_text(self, tid: int, new: bool = False) -> None:
        t = self.doc.texts.get(tid)
        if t is None:
            return
        self._finish_editor()
        ed = _TextEdit(self.viewport())
        ed.setPlainText(t.text)
        ed.resize(220, 64)
        pos = self.mapFromScene(QPointF(t.x, t.y))
        ed.move(pos.x(), pos.y())
        ed.setToolTip("Ctrl+Enter to finish, Esc to cancel")
        ed.committed.connect(self._finish_editor)
        ed.cancelled.connect(self._cancel_editor)
        ed.show()
        ed.setFocus()
        self._editor, self._edit_target = ed, ("text", tid, new)

    def _cancel_editor(self) -> None:
        ed, target = self._editor, self._edit_target
        self._editor = self._edit_target = None
        if ed is None:
            return
        ed.hide()
        ed.deleteLater()
        if target[0] == "text" and target[2] and target[1] in self.doc.texts and \
                not self.doc.texts[target[1]].text:
            self.edit("Delete Text", lambda: self.doc.remove([("text", target[1])]))
        self.setFocus()

    def _finish_editor(self) -> None:
        ed, target = self._editor, self._edit_target
        if ed is None:
            return
        self._editor = self._edit_target = None
        text = ed.text() if isinstance(ed, QLineEdit) else ed.toPlainText()
        ed.hide()
        ed.deleteLater()
        self.apply_editor_text(target, text)
        self.setFocus()

    def apply_editor_text(self, target, text: str) -> None:
        if target[0] == "atom" and target[1] in self.doc.atoms:
            if text.strip():
                self.edit("Atom Label", lambda: convert.apply_label_text(self.doc, target[1],
                                                                         text))
        elif target[0] == "text" and target[1] in self.doc.texts:
            tid = target[1]
            if text.strip():
                self.edit("Edit Text", lambda: setattr(self.doc.texts[tid], "text", text))
            else:
                self.edit("Delete Text", lambda: self.doc.remove([("text", tid)]))

    # --- dialogs / menus -------------------------------------------------------------------
    def atom_properties(self, aid: int) -> None:
        from rsstatchem.ui.chem_editor.dialogs import AtomPropertiesDialog
        dlg = AtomPropertiesDialog(self.doc.atoms[aid], self)
        if dlg.exec():
            self.edit("Atom Properties", lambda: dlg.apply(self.doc, aid))

    def arrow_properties(self, arid: int) -> None:
        from rsstatchem.ui.chem_editor.dialogs import ArrowDialog
        dlg = ArrowDialog(self.doc.arrows[arid], self)
        if dlg.exec():
            self.edit("Arrow Properties", lambda: dlg.apply(self.doc.arrows[arid]))

    def context_menu(self, h, p: QPointF) -> QMenu:
        menu = QMenu(self)
        if h is not None and h[0] == "atom":
            i = h[1]
            a = self.doc.atoms[i]
            menu.addAction("Atom Properties...", lambda: self.atom_properties(i))
            menu.addAction("Edit Label...", lambda: self.edit_label(i))
            if a.label and abbrev.is_abbreviation(a.label):
                menu.addAction("Expand Label",
                               lambda: self.edit("Expand Label",
                                                 lambda: convert.expand_label(self.doc, i)))
            if a.symbol == "C" and not a.label:
                act = menu.addAction("Show Carbon Label")
                act.setCheckable(True)
                act.setChecked(a.show_carbon)
                act.triggered.connect(lambda c: self.edit(
                    "Carbon Label", lambda: setattr(self.doc.atoms[i], "show_carbon", c)))
            sub = menu.addMenu("Element")
            for sym in ("C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si"):
                sub.addAction(sym, lambda s=sym: self.edit(
                    "Change Atom", lambda: ops.set_element(self.doc, i, s)))
            menu.addAction("Increase Charge",
                           lambda: self.edit("Charge", lambda: ops.change_charge(self.doc, i, 1)))
            menu.addAction("Decrease Charge",
                           lambda: self.edit("Charge", lambda: ops.change_charge(self.doc, i,
                                                                                   -1)))
            menu.addSeparator()
            menu.addAction("Delete", lambda: self.edit("Delete", lambda: self.erase([h])))
        elif h is not None and h[0] == "bond":
            i = h[1]
            for text, tool in (("Single", "single"), ("Double", "double"),
                               ("Triple", "triple"), ("Aromatic (dashed)", "aromatic"),
                               ("Wedge", "wedge"), ("Hashed Wedge", "hash"),
                               ("Wavy", "wavy"), ("Bold", "bold"), ("Dashed", "dashed")):
                def set_bond(_=False, tool=tool):
                    def fn():
                        b = self.doc.bonds[i]
                        b.order, b.style = ops.BOND_TOOLS[tool]
                    self.edit("Change Bond", fn)
                menu.addAction(text, set_bond)
            menu.addSeparator()

            def reverse():
                b = self.doc.bonds[i]
                b.a, b.b = b.b, b.a
            menu.addAction("Reverse Direction", lambda: self.edit("Reverse Bond", reverse))
            sub = menu.addMenu("Double Bond Position")
            for text, pos in (("Automatic", "auto"), ("Centered", "center"),
                              ("Left", "left"), ("Right", "right")):
                sub.addAction(text, lambda pos=pos: self.edit(
                    "Double Bond Position", lambda: setattr(self.doc.bonds[i], "double_pos",
                                                            pos)))
            menu.addSeparator()
            menu.addAction("Delete", lambda: self.edit("Delete", lambda: self.erase([h])))
        elif h is not None and h[0] == "arrow":
            i = h[1]
            menu.addAction("Arrow Properties...", lambda: self.arrow_properties(i))
            menu.addAction("Delete", lambda: self.edit("Delete", lambda: self.erase([h])))
        elif h is not None:
            if h[0] == "text":
                menu.addAction("Edit Text", lambda: self.edit_text(h[1]))
            menu.addAction("Delete", lambda: self.edit("Delete", lambda: self.erase([h])))
        else:
            win = self.window()
            for name in ("paste", "select_all", "clean"):
                act = getattr(win, f"act_{name}", None)
                if isinstance(act, QAction):
                    menu.addAction(act)
        return menu

    # --- view / insertion ------------------------------------------------------------------
    def set_zoom(self, z: float) -> None:
        z = max(0.2, min(12.0, z))
        f = z / self.zoom()
        self.scale(f, f)
        self.update_overlays()

    def fit(self) -> None:
        r = self.scene().itemsBoundingRect()
        if r.isEmpty():
            self.resetTransform()
            self.scale(1.6, 1.6)
            self.centerOn(0, 0)
            return
        m = self.bl
        self.fitInView(r.adjusted(-m, -m, m, m), Qt.AspectRatioMode.KeepAspectRatio)
        if self.zoom() > 4:
            self.set_zoom(4)
        self.centerOn(r.center())
        self.update_overlays()

    def fit_later(self) -> None:
        """Fit once the view has its real size (on first show)."""
        self._fit_pending = True
        if self.isVisible():
            QTimer.singleShot(0, self._do_pending_fit)

    def _do_pending_fit(self) -> None:
        if getattr(self, "_fit_pending", False):
            self._fit_pending = False
            self.fit()
            if self.zoom() > 2.0:
                self.set_zoom(2.0)

    def showEvent(self, e) -> None:  # noqa: N802
        super().showEvent(e)
        if getattr(self, "_fit_pending", False):
            QTimer.singleShot(0, self._do_pending_fit)

    def view_center(self) -> QPointF:
        return self.mapToScene(self.viewport().rect().center())

    def insert_text(self, text: str, x: float | None = None, y: float | None = None,
                    title: str = "Insert Structure") -> list:
        if x is None or y is None:
            c = self.view_center()
            x, y = c.x(), c.y()
        keys = self.edit(title, lambda: convert.insert_text(self.doc, text, x, y))
        self.select_keys(set(keys or []))
        return keys

    def paste_document(self, other: Document) -> list:
        bb = other.bbox()
        c = self.view_center()
        dx = dy = 0.0
        if bb is not None:
            dx, dy = c.x() - (bb[0] + bb[2]) / 2, c.y() - (bb[1] + bb[3]) / 2
            if self.doc.bbox() is not None and abs(dx) < 1 and abs(dy) < 1:
                dx += self.bl
                dy += self.bl
        if other.style.bond_length and abs(other.style.bond_length - self.bl) > 1e-6:
            other = other.copy()
            other.scale(self.bl / other.style.bond_length)
            bb = other.bbox()
            if bb is not None:
                dx, dy = c.x() - (bb[0] + bb[2]) / 2, c.y() - (bb[1] + bb[3]) / 2
        keys = self.edit("Paste", lambda: self.doc.paste(other, dx, dy))
        self.select_keys(set(keys or []))
        return keys

    def clean(self) -> bool:
        atoms = self.selected_atoms() or None
        return bool(self.edit("Clean Structure",
                              lambda: convert.clean_structure(self.doc, atoms)))

    def event(self, e) -> bool:
        if e.type() == QEvent.Type.ShortcutOverride and self.hover is not None and \
                self._editor is None and e.text() and e.text() in "cnospfihlbBk+-=.123wya":
            e.accept()
            return True
        return super().event(e)

    def problems_text(self) -> str:
        if not self.problems:
            return ""
        return "; ".join(sorted(set(self.problems.values())))[:200]

    def mol_or_none(self):
        try:
            return convert.to_mol(self.doc, self.selected_atoms() or None)
        except mo.ChemError:
            return None

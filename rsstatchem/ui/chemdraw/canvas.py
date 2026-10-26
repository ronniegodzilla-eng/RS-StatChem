"""Document view of the Chemical Drawing workspace: the structure canvas drawn as a white
page (US Letter, 612 x 792 pt) on a gray desk, with ChemDraw's zoom percentages (100 % =
actual size on a 96 dpi screen) and the Object menu's Fixed Lengths / Fixed Angles."""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen

from rsstatchem.core.chem_draw import geometry as geo
from rsstatchem.ui.chem_editor.canvas import ChemCanvas

__all__ = ["PageCanvas", "PAGE_SIZES", "ACTUAL"]

ACTUAL = 96.0 / 72.0           # view scale of 100 % (1 pt = 1/72 in on a 96 dpi screen)
PAGE_SIZES = {"Letter": (612.0, 792.0), "A4": (595.0, 842.0)}
DESK = QColor("#9A9A9A")
ZOOM_STEPS = (12.5, 25, 33, 50, 66, 75, 100, 125, 150, 200, 300, 400, 600, 800)


class PageCanvas(ChemCanvas):
    zoom_changed = Signal(float)        # percent
    # class defaults: the base constructor already rebuilds the scene
    page = QRectF(0, 0, *PAGE_SIZES["Letter"])
    fixed_angles = True
    fixed_lengths = True
    _centered = False

    def __init__(self, doc, undo_stack, parent=None):
        super().__init__(doc, undo_stack, parent)
        self.setBackgroundBrush(QBrush(DESK))
        self.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        self.resetTransform()
        self.scale(ACTUAL, ACTUAL)

    # --- page ---------------------------------------------------------------------------
    def set_page_size(self, name: str) -> None:
        self.page = QRectF(0, 0, *PAGE_SIZES[name])
        self._update_scene_rect()
        self.viewport().update()

    def _update_scene_rect(self) -> None:
        m = 36.0
        r = self.page.adjusted(-m, -m, m, m)
        items = self.scene().itemsBoundingRect()
        if not items.isEmpty():
            r = r.united(items.adjusted(-m, -m, m, m))
        self.scene().setSceneRect(r)

    def rebuild(self) -> None:
        super().rebuild()
        self._update_scene_rect()

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802
        painter.fillRect(rect, DESK)
        page = self.page
        shadow = page.translated(3.0 / self.zoom(), 3.0 / self.zoom())
        painter.fillRect(shadow, QColor(0, 0, 0, 70))
        painter.fillRect(page, QColor("#FFFFFF"))
        painter.setPen(QPen(QColor("#5E5E5E"), 0))
        painter.drawRect(page)
        if self.show_grid:
            step = self.bl
            painter.setPen(QPen(QColor(222, 228, 238), 0))
            x = page.left() + step
            while x < page.right():
                painter.drawLine(QPointF(x, page.top()), QPointF(x, page.bottom()))
                x += step
            y = page.top() + step
            while y < page.bottom():
                painter.drawLine(QPointF(page.left(), y), QPointF(page.right(), y))
                y += step

    def page_center(self) -> QPointF:
        return self.page.center()

    # --- zoom ---------------------------------------------------------------------------
    def zoom_percent(self) -> float:
        return self.zoom() / ACTUAL * 100.0

    def set_zoom_percent(self, pct: float) -> None:
        self.set_zoom(ACTUAL * pct / 100.0)

    def set_zoom(self, z: float) -> None:
        super().set_zoom(max(ACTUAL * 0.1, min(ACTUAL * 8.0, z)))
        self.zoom_changed.emit(self.zoom_percent())

    def step_zoom(self, up: bool) -> None:
        pct = self.zoom_percent()
        if up:
            nxt = next((s for s in ZOOM_STEPS if s > pct + 0.5), ZOOM_STEPS[-1])
        else:
            nxt = next((s for s in reversed(ZOOM_STEPS) if s < pct - 0.5), ZOOM_STEPS[0])
        self.set_zoom_percent(nxt)

    def fit(self) -> None:
        """Fit to Window: the drawing (or the page when it is empty) fills the view."""
        r = self.scene().itemsBoundingRect()
        if r.isEmpty():
            r = self.page
        m = self.bl
        self.fitInView(r.adjusted(-m, -m, m, m), Qt.AspectRatioMode.KeepAspectRatio)
        if self.zoom() > ACTUAL * 4:          # a lone small molecule: stop at 400 %
            self.set_zoom(ACTUAL * 4)
        self.centerOn(r.center())
        self.update_overlays()
        self.zoom_changed.emit(self.zoom_percent())

    def actual_size(self) -> None:
        self.set_zoom(ACTUAL)

    def show_page_top(self) -> None:
        """Initial view: the top of the page, horizontally centred."""
        vh = self.viewport().height() / max(self.zoom(), 1e-6)
        self.centerOn(self.page.center().x(), self.page.top() - 24 + vh / 2)

    def show_content(self) -> None:
        """Bring the drawing into view at the current zoom (zoom out only if it is larger
        than the window)."""
        r = self.scene().itemsBoundingRect()
        if r.isEmpty():
            self.show_page_top()
            return
        vw = self.viewport().width() / self.zoom()
        vh = self.viewport().height() / self.zoom()
        if r.width() + 2 * self.bl > vw or r.height() + 2 * self.bl > vh:
            self.fit()
            return
        self.centerOn(r.center())

    def fit_later(self) -> None:
        self._fit_pending = True
        if self.isVisible():
            QTimer.singleShot(0, self._do_pending_fit)

    def _do_pending_fit(self) -> None:
        if getattr(self, "_fit_pending", False):
            self._fit_pending = False
            self.show_content()

    def showEvent(self, e) -> None:  # noqa: N802
        super().showEvent(e)
        if not self._centered:
            self._centered = True
            if not getattr(self, "_fit_pending", False):
                QTimer.singleShot(0, self.show_page_top)

    def wheelEvent(self, e) -> None:  # noqa: N802
        if e.modifiers() & (Qt.KeyboardModifier.ControlModifier |
                            Qt.KeyboardModifier.MetaModifier):
            self.step_zoom(e.angleDelta().y() > 0)
            return
        super().wheelEvent(e)

    # --- Fixed Lengths / Fixed Angles (Object menu) -------------------------------------
    def _bond_end(self, p: QPointF, sx: float, sy: float, start_id, free: bool):
        target = self.doc.atom_at(p.x(), p.y(), max(self.bl * 0.3, 7 / self.zoom()),
                                  exclude={start_id} if start_id else ())
        if target is not None:
            return target.x, target.y, target.id
        ang = math.atan2(p.y() - sy, p.x() - sx)
        if self.fixed_angles and not free:
            ang = geo.snap_angle(ang, 15)
        length = self.bl
        if not self.fixed_lengths:
            length = max(self.bl * 0.25, math.hypot(p.x() - sx, p.y() - sy))
        ex, ey = geo.bond_end_point(sx, sy, ang, length)
        return ex, ey, None

    def _arrow_end(self, p: QPointF, free: bool) -> tuple[float, float]:
        return super()._arrow_end(p, free or not self.fixed_angles)

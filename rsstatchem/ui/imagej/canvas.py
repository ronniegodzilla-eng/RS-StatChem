"""ImageCanvas: draws an :class:`~rsstatchem.core.imaging.image.ImageDoc` with ImageJ's zoom
levels, the ROI and the overlay. **Owned by WP-A after Phase 0**; the API below is the contract
other packages call (WP-D zoom commands, WP-C ROI tools/overlays):

* ``magnification`` / ``set_magnification(m, center=None)`` / ``zoom_in(at)`` / ``zoom_out(at)``
* ``refresh()`` - re-render the pixels (after the image or its display range/LUT changed)
* ``repaint_overlay()`` - redraw ROI/overlay only
* ``image_to_screen(x, y)`` / ``screen_to_image(px, py)`` (image coordinates as floats)
* signal ``mouse_moved(float x, float y)`` in image coordinates (the status line uses it)
* ``paint_hooks``: list of ``fn(painter, canvas)`` called after the ROI/overlay are drawn
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from rsstatchem.core.imaging.lut import render

# ImageCanvas.zoomLevels of ImageJ 1.54
ZOOM_LEVELS = (1 / 72, 1 / 48, 1 / 32, 1 / 24, 1 / 16, 1 / 12, 1 / 8, 1 / 6, 1 / 4, 1 / 3,
               1 / 2, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0)


def higher_zoom(m: float) -> float:
    return next((z for z in ZOOM_LEVELS if z > m * 1.0001), ZOOM_LEVELS[-1])


def lower_zoom(m: float) -> float:
    return next((z for z in reversed(ZOOM_LEVELS) if z < m / 1.0001), ZOOM_LEVELS[0])


class ImageCanvas(QWidget):
    mouse_moved = Signal(float, float)

    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.magnification = 1.0
        self.src_x = 0.0      # image coordinate shown at the canvas' left/top edge
        self.src_y = 0.0
        self.paint_hooks: list = []
        self._qimage: QImage | None = None
        self._buffer: np.ndarray | None = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.refresh()

    # ------------------------------------------------------------------ rendering
    def refresh(self) -> None:
        rgb = np.ascontiguousarray(render(self.doc))
        h, w = rgb.shape[:2]
        self._buffer = rgb  # QImage does not own the memory
        self._qimage = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.update()

    def repaint_overlay(self) -> None:
        self.update()

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(max(1, round(self.doc.width * self.magnification)),
                     max(1, round(self.doc.height * self.magnification)))

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(255, 255, 255))
        if self._qimage is not None:
            m = self.magnification
            target = QRectF(0, 0, self.width(), self.height())
            source = QRectF(self.src_x, self.src_y, self.width() / m, self.height() / m)
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, m < 1.0)
            p.drawImage(target, self._qimage, source)
        overlay = self.doc.overlay
        if overlay is not None and not overlay.hidden:
            for roi in overlay:
                self._draw_roi(p, roi, QColor(roi.stroke_color or "#00ffff"))
        if self.doc.roi is not None:
            self._draw_roi(p, self.doc.roi, QColor(roi_color(self.doc.roi)))
        for hook in list(self.paint_hooks):
            hook(p, self)
        p.end()

    def _draw_roi(self, p: QPainter, roi, color: QColor) -> None:
        from rsstatchem.core.imaging.roi import shapes as S

        p.setPen(QPen(color, max(1.0, roi.stroke_width * self.magnification)))
        p.setBrush(Qt.BrushStyle.NoBrush)
        if isinstance(roi, S.CompositeRoi):
            for _, part in roi.parts:
                self._draw_roi(p, part, color)
            return
        if isinstance(roi, (S.RectRoi, S.OvalRoi)):
            x0, y0 = self.image_to_screen(roi.x, roi.y)
            x1, y1 = self.image_to_screen(roi.x + roi.width, roi.y + roi.height)
            rect = QRectF(x0, y0, x1 - x0, y1 - y0)
            if isinstance(roi, S.OvalRoi):
                p.drawEllipse(rect)
            else:
                p.drawRect(rect)
            return
        xs, ys = roi.polygon()
        pts = [QPointF(*self.image_to_screen(x, y)) for x, y in zip(xs, ys)]
        if isinstance(roi, S.PointRoi):
            for pt in pts:
                p.drawLine(pt - QPointF(3, 0), pt + QPointF(3, 0))
                p.drawLine(pt - QPointF(0, 3), pt + QPointF(0, 3))
            return
        if roi.is_area():
            p.drawPolygon(QPolygonF(pts))
        else:
            path = QPainterPath()
            if pts:
                path.moveTo(pts[0])
                for pt in pts[1:]:
                    path.lineTo(pt)
            p.drawPath(path)

    # ------------------------------------------------------------------ coordinates & zoom
    def image_to_screen(self, x: float, y: float) -> tuple[float, float]:
        m = self.magnification
        return (x - self.src_x) * m, (y - self.src_y) * m

    def screen_to_image(self, px: float, py: float) -> tuple[float, float]:
        m = self.magnification
        return px / m + self.src_x, py / m + self.src_y

    def set_magnification(self, m: float, center: tuple[float, float] | None = None) -> None:
        """Zoom to ``m`` keeping image point ``center`` (default: the view centre) in place."""
        m = min(max(float(m), ZOOM_LEVELS[0]), ZOOM_LEVELS[-1])
        if center is None:
            center = self.screen_to_image(self.width() / 2, self.height() / 2)
        cx, cy = center
        sx, sy = self.image_to_screen(cx, cy)
        self.magnification = m
        self.src_x = cx - sx / m
        self.src_y = cy - sy / m
        self._clamp_source()
        self.updateGeometry()
        self.update()
        win = self.window()
        if hasattr(win, "update_title"):
            win.update_title()

    def zoom_in(self, at: tuple[float, float] | None = None) -> None:
        self.set_magnification(higher_zoom(self.magnification), at)

    def zoom_out(self, at: tuple[float, float] | None = None) -> None:
        self.set_magnification(lower_zoom(self.magnification), at)

    def _clamp_source(self) -> None:
        m = self.magnification
        vw, vh = self.width() / m, self.height() / m
        self.src_x = 0.0 if vw >= self.doc.width else min(max(self.src_x, 0.0),
                                                           self.doc.width - vw)
        self.src_y = 0.0 if vh >= self.doc.height else min(max(self.src_y, 0.0),
                                                            self.doc.height - vh)

    def resizeEvent(self, event) -> None:  # noqa: N802
        self._clamp_source()
        super().resizeEvent(event)

    # ------------------------------------------------------------------ events
    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        x, y = self.screen_to_image(event.position().x(), event.position().y())
        self.mouse_moved.emit(x, y)
        super().mouseMoveEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802
        if event.modifiers() & (Qt.KeyboardModifier.ControlModifier |
                                Qt.KeyboardModifier.MetaModifier):
            at = self.screen_to_image(event.position().x(), event.position().y())
            (self.zoom_in if event.angleDelta().y() > 0 else self.zoom_out)(at)
            event.accept()
            return
        super().wheelEvent(event)


def roi_color(roi) -> str:
    return roi.stroke_color or "#ffff00"

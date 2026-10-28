"""ImageCanvas: draws an :class:`~rsstatchem.core.imaging.image.ImageDoc` with ImageJ's zoom
levels, the threshold display, the selection (with handles), the overlay and the tools'
previews (``ij/gui/ImageCanvas.java`` behaviour). **Owned by WP-A.** API contract used by
other packages (WP-D zoom commands, WP-C tools/overlays):

* ``magnification`` / ``set_magnification(m, center=None)`` / ``zoom_in(at)`` / ``zoom_out(at)``
* ``refresh()`` - re-render the pixels (after the image or its display range/LUT changed)
* ``repaint_overlay()`` - redraw ROI/overlay only
* ``image_to_screen(x, y)`` / ``screen_to_image(px, py)`` (image coordinates as floats)
* signal ``mouse_moved(float x, float y)`` in image coordinates (the status line uses it)
* ``paint_hooks``: list of ``fn(painter, canvas)`` called after the ROI/overlay are drawn

Zooming follows ImageJ: the window grows with the zoom until it would leave the screen, then
the view is centred on the zoom target (``ImageCanvas.zoomIn/adjustSourceRect``); a small
navigation rectangle shows the visible part when the image is larger than the window.
Mouse and key events go to ``handler`` (:class:`rsstatchem.ui.imagej.roi_tools.ToolHandler`).
"""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (QBrush, QColor, QFont, QImage, QPainter, QPainterPath, QPen,
                           QPolygonF)
from PySide6.QtWidgets import QWidget

from rsstatchem.core.imaging.lut import render

# ImageCanvas.zoomLevels of ImageJ 1.54
ZOOM_LEVELS = (1 / 72, 1 / 48, 1 / 32, 1 / 24, 1 / 16, 1 / 12, 1 / 8, 1 / 6, 1 / 4, 1 / 3,
               1 / 2, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0)
HANDLE = 5
POINT_SIZES = {"Tiny": 1, "Small": 3, "Medium": 5, "Large": 7, "Extra Large": 11, "XXL": 17,
               "XXXL": 25}
COUNTER_COLORS = ("#ffff00", "#00ffff", "#ff00ff", "#00ff00", "#ff0000", "#0000ff",
                  "#ffc800", "#ffffff")


def higher_zoom(m: float) -> float:
    return next((z for z in ZOOM_LEVELS if z > m * 1.0001), ZOOM_LEVELS[-1])


def lower_zoom(m: float) -> float:
    return next((z for z in reversed(ZOOM_LEVELS) if z < m / 1.0001), ZOOM_LEVELS[0])


def zoom_text(m: float) -> str:
    pct = m * 100
    return f"{pct:.0f}%" if abs(pct - round(pct)) < 1e-6 else f"{pct:.1f}%"


def threshold_rgb(rgb: np.ndarray, plane: np.ndarray, threshold) -> np.ndarray:
    """ImageJ's threshold display: red (in range red), black & white, over/under (blue below,
    green above)."""
    if threshold is None or threshold.mode == "none" or plane.ndim != 2:
        return rgb
    p = plane
    lo, hi = threshold.lower, threshold.upper
    inside = (p >= lo) & (p <= hi)
    out = rgb.copy()
    if threshold.mode == "red":
        out[inside] = (255, 0, 0)
    elif threshold.mode == "bw":
        out[inside] = (0, 0, 0)
        out[~inside] = (255, 255, 255)
    elif threshold.mode == "over_under":
        out[p < lo] = (0, 0, 255)
        out[p > hi] = (0, 255, 0)
    return out


class ImageCanvas(QWidget):
    mouse_moved = Signal(float, float)

    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.magnification = 1.0
        self.src_x = 0.0      # image coordinate shown at the canvas' left/top edge
        self.src_y = 0.0
        self.paint_hooks: list = []
        self.handler = None              # roi_tools.ToolHandler
        self.selection_color = "#ffff00"
        self.options = None              # session Options (point tool, colours)
        self.show_all_rois: list = []    # ROI Manager "Show All" (WP-C may fill)
        self.interpolate = False
        self._qimage: QImage | None = None
        self._buffer: np.ndarray | None = None
        self._outline_cache: dict = {}
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(16, 16)
        self.refresh()

    # ------------------------------------------------------------------ rendering
    def refresh(self) -> None:
        rgb = np.ascontiguousarray(render(self.doc))
        if self.doc.threshold is not None and not self.doc.is_rgb:
            rgb = np.ascontiguousarray(threshold_rgb(rgb, self.doc.plane(), self.doc.threshold))
        h, w = rgb.shape[:2]
        self._buffer = rgb  # QImage does not own the memory
        self._qimage = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self._outline_cache.clear()
        self._clamp_source()
        self.update()

    def repaint_overlay(self) -> None:
        self._outline_cache.clear()
        self.update()

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(max(1, round(self.doc.width * self.magnification)),
                     max(1, round(self.doc.height * self.magnification)))

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(255, 255, 255))
        m = self.magnification
        if self._qimage is not None:
            iw, ih = self.doc.width * m, self.doc.height * m
            target = QRectF(0, 0, min(self.width(), iw - self.src_x * m),
                            min(self.height(), ih - self.src_y * m))
            source = QRectF(self.src_x, self.src_y, target.width() / m, target.height() / m)
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform,
                            m < 1.0 or (self.interpolate and m > 1.0))
            p.drawImage(target, self._qimage, source)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        overlay = self.doc.overlay
        if overlay is not None and not overlay.hidden:
            for i, roi in enumerate(overlay):
                if not self._on_position(roi):
                    continue
                self.draw_roi(p, roi, roi.stroke_color or "#00ffff", handles=False)
                if overlay.show_labels or overlay.draw_names:
                    self._label(p, roi, roi.name if overlay.draw_names and roi.name
                                else str(i + 1), overlay.label_color or "#ffffff")
        for i, roi in enumerate(self.show_all_rois):
            self.draw_roi(p, roi, "#00ffff", handles=False)
            self._label(p, roi, str(i + 1), "#00ffff")
        roi = self.doc.roi
        if roi is not None:
            self.draw_roi(p, roi, roi.stroke_color or self.selection_color, handles=True)
        for hook in list(self.paint_hooks):
            hook(p, self)
        if self.handler is not None:
            self.handler.paint(p, self)
        self._navigator(p)
        p.end()

    def _on_position(self, roi) -> bool:
        pos = getattr(roi, "position", (0, 0, 0)) or (0, 0, 0)
        c, z, t = (tuple(pos) + (0, 0, 0))[:3]
        dc, dz, dt = self.doc.position
        if c == 0 and t == 0 and z:
            return z == self.doc.current_index + 1 if not self.doc.hyperstack else z == dz + 1
        return (c == 0 or c == dc + 1) and (z == 0 or z == dz + 1) and (t == 0 or t == dt + 1)

    def _label(self, p: QPainter, roi, text: str, color: str) -> None:
        x, y, w, h = roi.bounds()
        sx, sy = self.image_to_screen(x + w / 2, y + h / 2)
        f = QFont()
        f.setPixelSize(max(9, int(9 * min(self.magnification, 2))))
        p.setFont(f)
        p.setPen(QColor(color))
        p.drawText(QRectF(sx - 30, sy - 8, 60, 16), Qt.AlignmentFlag.AlignCenter, text)

    def _qpts(self, xs, ys) -> list[QPointF]:
        return [QPointF(*self.image_to_screen(x, y)) for x, y in zip(xs, ys)]

    def draw_roi(self, p: QPainter, roi, color, handles: bool = False) -> None:
        """Draw any ROI type as ImageJ shows it (outline, optional fill, handles)."""
        from rsstatchem.core.imaging.roi import shapes as S

        if type(roi).__name__ == "ImageRoi" and hasattr(roi, "pixels"):
            self._draw_image_roi(p, roi, handles)
            return
        if getattr(roi, "image_data", None) is not None:
            self._draw_image_roi(p, _ImageData(roi), handles)
            return
        qc = QColor(color)
        width = roi.stroke_width * self.magnification if roi.stroke_width and \
            roi.stroke_width > 1 else 1.0
        pen = QPen(qc, width)
        p.setPen(pen)
        fill = QColor(roi.fill_color) if roi.fill_color else None
        p.setBrush(QBrush(fill) if fill is not None else Qt.BrushStyle.NoBrush)
        hpts: list[tuple[float, float]] = []
        if isinstance(roi, S.CompositeRoi):
            path = QPainterPath()
            path.setFillRule(Qt.FillRule.OddEvenFill)
            for loop in self._composite_loops(roi):
                pts = self._qpts(loop[:, 0], loop[:, 1])
                path.addPolygon(QPolygonF(pts + pts[:1]))
            p.drawPath(path)
            x, y, w, h = roi.bounds()
            hpts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        elif isinstance(roi, S.TextRoi):
            f = QFont(roi.font)
            f.setPixelSize(max(1, int(round(roi.font_size * self.magnification))))
            p.setFont(f)
            sx, sy = self.image_to_screen(roi.x, roi.y)
            p.save()
            p.translate(sx, sy)
            p.rotate(-roi.angle)
            p.setPen(qc)
            lines = roi.text.split("\n") if roi.text else [""]
            lh = f.pixelSize() * 1.2
            for i, line in enumerate(lines):
                p.drawText(QPointF(0, (i + 0.85) * lh), line)
            p.restore()
            if handles:
                x, y, w, h = roi.bounds()
                p.setPen(QPen(qc, 1, Qt.PenStyle.DashLine))
                p.setBrush(Qt.BrushStyle.NoBrush)
                x0, y0 = self.image_to_screen(x, y)
                x1, y1 = self.image_to_screen(x + w, y + h)
                p.drawRect(QRectF(x0, y0, x1 - x0, y1 - y0))
            return
        elif isinstance(roi, (S.RectRoi, S.OvalRoi)):
            x0, y0 = self.image_to_screen(roi.x, roi.y)
            x1, y1 = self.image_to_screen(roi.x + roi.width, roi.y + roi.height)
            rect = QRectF(x0, y0, x1 - x0, y1 - y0)
            if isinstance(roi, S.OvalRoi):
                p.drawEllipse(rect)
            elif roi.corner_diameter:
                d = roi.corner_diameter * self.magnification / 2
                p.drawRoundedRect(rect, d, d)
            else:
                p.drawRect(rect)
            hpts = handle_points(roi)
        elif isinstance(roi, S.PointRoi):
            self._draw_points(p, roi, qc)
            return
        elif isinstance(roi, S.LineRoi):
            a = QPointF(*self.image_to_screen(roi.x1, roi.y1))
            b = QPointF(*self.image_to_screen(roi.x2, roi.y2))
            lw = max(roi.width, roi.stroke_width or 0)
            if lw > 1 and not roi.arrow:
                from rsstatchem.ui.imagej.commands_edit import wide_line_polygon
                p.setPen(QPen(qc, 1))
                p.setBrush(Qt.BrushStyle.NoBrush)
                pts = wide_line_polygon(roi, lw)
                p.drawPolygon(QPolygonF([QPointF(*self.image_to_screen(x, y)) for x, y in pts]))
                p.setPen(QPen(qc, 1, Qt.PenStyle.DotLine))
            p.drawLine(a, b)
            if roi.arrow:
                self._arrow_head(p, a, b, qc, roi)
            hpts = handle_points(roi)
        else:
            xs, ys = roi.polygon()
            pts = self._qpts(xs, ys)
            if roi.is_area():
                p.drawPolygon(QPolygonF(pts))
            elif pts:
                path = QPainterPath(pts[0])
                for pt in pts[1:]:
                    path.lineTo(pt)
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawPath(path)
            hpts = handle_points(roi)
        if handles and hpts:
            self._handles(p, hpts)

    def _draw_image_roi(self, p: QPainter, roi, handles: bool) -> None:
        """ImageRoi overlay element (Image > Overlay > Add Image): RGB pixels at (x, y) with
        opacity; zero pixels transparent when ``zero_transparent``."""
        key = ("img", id(roi), id(roi.pixels))
        qimg = self._outline_cache.get(key)
        if qimg is None:
            px = np.asarray(roi.pixels)
            if px.ndim == 2:
                px = np.repeat(px[..., None], 3, axis=2)
            px = np.ascontiguousarray(px[..., :3].astype(np.uint8))
            alpha = np.full(px.shape[:2], 255, np.uint8)
            if getattr(roi, "zero_transparent", False):
                alpha[np.all(px == 0, axis=2)] = 0
            rgba = np.ascontiguousarray(np.dstack([px, alpha]))
            h, w = rgba.shape[:2]
            qimg = QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
            self._outline_cache[key] = qimg
        x0, y0 = self.image_to_screen(roi.x, roi.y)
        w = roi.width or qimg.width()
        h = roi.height or qimg.height()
        x1, y1 = self.image_to_screen(roi.x + w, roi.y + h)
        p.save()
        p.setOpacity(max(0.0, min(1.0, float(getattr(roi, "opacity", 1.0)))))
        p.drawImage(QRectF(x0, y0, x1 - x0, y1 - y0), qimg)
        p.restore()
        if handles:
            p.setPen(QPen(QColor(self.selection_color), 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(x0, y0, x1 - x0, y1 - y0))

    def _composite_loops(self, roi):
        key = id(roi)
        loops = self._outline_cache.get(key)
        if loops is None:
            from rsstatchem.core.imaging.io.roi_codec import mask_outlines
            x, y, w, h = roi.bounds()
            x0, y0 = int(math.floor(x)) - 1, int(math.floor(y)) - 1
            W, H = int(math.ceil(x + w)) - x0 + 2, int(math.ceil(y + h)) - y0 + 2
            moved = roi.copy()
            moved.translate(-x0, -y0)
            loops = [lp + [x0, y0] for lp in mask_outlines(moved.get_mask((H, W)))]
            self._outline_cache[key] = loops
        return loops

    def _arrow_head(self, p, a: QPointF, b: QPointF, qc, roi) -> None:
        size = float(roi.properties.get("_arrow_head", 10) or 10) * \
            max(1.0, roi.stroke_width or 1) ** 0.5 * self.magnification
        ang = math.atan2(b.y() - a.y(), b.x() - a.x())
        p1 = QPointF(b.x() - size * math.cos(ang - 0.4), b.y() - size * math.sin(ang - 0.4))
        p2 = QPointF(b.x() - size * math.cos(ang + 0.4), b.y() - size * math.sin(ang + 0.4))
        p.setBrush(QBrush(qc))
        p.drawPolygon(QPolygonF([b, p1, p2]))

    def _draw_points(self, p: QPainter, roi, qc: QColor) -> None:
        opts = self.options
        size = POINT_SIZES.get(getattr(opts, "point_size", "Small"), 3)
        kind = getattr(opts, "point_type", "Hybrid")
        xs, ys = roi.polygon()
        counters = roi.counters if roi.counters is not None else [0] * len(xs)
        for i, (x, y) in enumerate(zip(xs, ys)):
            sx, sy = self.image_to_screen(x + 0.5, y + 0.5)
            c = QColor(COUNTER_COLORS[int(counters[i]) % len(COUNTER_COLORS)]) if \
                roi.counters is not None and any(counters) else qc
            r = size / 2 + 1
            if kind in ("Hybrid", "Cross"):
                p.setPen(QPen(QColor(255, 255, 255) if kind == "Hybrid" else c, 1))
                p.drawLine(QPointF(sx - r - 3, sy), QPointF(sx + r + 3, sy))
                p.drawLine(QPointF(sx, sy - r - 3), QPointF(sx, sy + r + 3))
            if kind in ("Hybrid", "Dot"):
                p.fillRect(QRectF(sx - r, sy - r, 2 * r, 2 * r), c)
            if kind == "Circle":
                p.setPen(QPen(c, 1))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(QPointF(sx, sy), r + 2, r + 2)
            if getattr(opts, "point_label", False) and len(xs) > 1:
                p.setPen(c)
                p.drawText(QPointF(sx + r + 2, sy - r), str(i + 1))

    def _handles(self, p: QPainter, pts) -> None:
        p.setPen(QPen(QColor(255, 255, 255), 1))
        p.setBrush(QBrush(QColor(0, 0, 0)))
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        for x, y in pts:
            sx, sy = self.image_to_screen(x, y)
            p.drawRect(QRectF(sx - HANDLE / 2, sy - HANDLE / 2, HANDLE, HANDLE))
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    def _navigator(self, p: QPainter) -> None:
        """ImageJ's zoom indicator: the visible part inside the whole image."""
        m = self.magnification
        vw, vh = self.width() / m, self.height() / m
        if vw >= self.doc.width - 0.5 and vh >= self.doc.height - 0.5:
            return
        scale = 64.0 / max(self.doc.width, self.doc.height)
        W, H = self.doc.width * scale, self.doc.height * scale
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        p.setPen(QPen(QColor(128, 128, 255), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QRectF(10, 10, W, H))
        p.drawRect(QRectF(10 + self.src_x * scale, 10 + self.src_y * scale,
                          min(vw, self.doc.width) * scale, min(vh, self.doc.height) * scale))

    # ------------------------------------------------------------------ coordinates & zoom
    def image_to_screen(self, x: float, y: float) -> tuple[float, float]:
        m = self.magnification
        return (x - self.src_x) * m, (y - self.src_y) * m

    def screen_to_image(self, px: float, py: float) -> tuple[float, float]:
        m = self.magnification
        return px / m + self.src_x, py / m + self.src_y

    def set_magnification(self, m: float, center: tuple[float, float] | None = None) -> None:
        """Zoom to ``m`` keeping image point ``center`` (default: the view centre) in place;
        the window grows to show more of the image when it can (ImageJ)."""
        m = min(max(float(m), ZOOM_LEVELS[0]), ZOOM_LEVELS[-1])
        if center is None:
            center = self.screen_to_image(self.width() / 2, self.height() / 2)
        cx, cy = center
        sx, sy = self.image_to_screen(cx, cy)
        self.magnification = m
        self._grow_window()
        self.src_x = cx - sx / m
        self.src_y = cy - sy / m
        self._clamp_source()
        self._changed()

    def zoom_to(self, m: float, target: tuple[float, float] | None = None) -> None:
        """ImageJ zoom: enlarge the window if possible, else centre the view on ``target``."""
        m = min(max(float(m), ZOOM_LEVELS[0]), ZOOM_LEVELS[-1])
        if target is None:
            target = self.screen_to_image(self.width() / 2, self.height() / 2)
        self.magnification = m
        w, h = self._grow_window()
        vw, vh = w / m, h / m
        self.src_x = target[0] - vw / 2
        self.src_y = target[1] - vh / 2
        self._clamp_source(w, h)
        self._changed()

    def zoom_in(self, at: tuple[float, float] | None = None) -> None:
        if self.magnification >= ZOOM_LEVELS[-1]:
            return
        self.zoom_to(higher_zoom(self.magnification), at)

    def zoom_out(self, at: tuple[float, float] | None = None) -> None:
        if self.magnification <= ZOOM_LEVELS[0]:
            return
        self.zoom_to(lower_zoom(self.magnification), at)

    def original_scale(self) -> None:
        self.zoom_to(1.0, (self.doc.width / 2, self.doc.height / 2))

    def scale_to_fit(self, width: int, height: int) -> None:
        m = min(width / self.doc.width, height / self.doc.height)
        self.magnification = max(m, ZOOM_LEVELS[0])
        self.src_x = self.src_y = 0.0
        self._grow_window()
        self._changed()

    def _grow_window(self) -> tuple[int, int]:
        win = self.window()
        want = (max(1, round(self.doc.width * self.magnification)),
                max(1, round(self.doc.height * self.magnification)))
        if hasattr(win, "grow_canvas_to"):
            return win.grow_canvas_to(*want)
        return self.width(), self.height()

    def _changed(self) -> None:
        self.updateGeometry()
        self.update()
        win = self.window()
        if hasattr(win, "update_title"):
            win.update_title()

    def scroll_by(self, dx_screen: float, dy_screen: float) -> None:
        m = self.magnification
        self.src_x -= dx_screen / m
        self.src_y -= dy_screen / m
        self._clamp_source()
        self.update()

    def _clamp_source(self, w: float | None = None, h: float | None = None) -> None:
        m = self.magnification
        vw = (self.width() if w is None else w) / m
        vh = (self.height() if h is None else h) / m
        self.src_x = 0.0 if vw >= self.doc.width else min(max(self.src_x, 0.0),
                                                           self.doc.width - vw)
        self.src_y = 0.0 if vh >= self.doc.height else min(max(self.src_y, 0.0),
                                                            self.doc.height - vh)

    def resizeEvent(self, event) -> None:  # noqa: N802
        self._clamp_source()
        super().resizeEvent(event)

    # ------------------------------------------------------------------ events
    def _image_pos(self, event) -> tuple[float, float]:
        pos = event.position()
        return self.screen_to_image(pos.x(), pos.y())

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.setFocus()
        if self.handler is not None:
            self.handler.mouse_press(self, event, *self._image_pos(event))

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        x, y = self._image_pos(event)
        if self.handler is not None:
            self.handler.mouse_move(self, event, x, y)
        self.mouse_moved.emit(x, y)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self.handler is not None:
            self.handler.mouse_release(self, event, *self._image_pos(event))

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if self.handler is not None:
            self.handler.mouse_double_click(self, event, *self._image_pos(event))

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if self.handler is not None and self.handler.key_press(self, event):
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: N802
        if self.handler is not None and self.handler.key_release(self, event):
            return
        super().keyReleaseEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802
        if event.modifiers() & (Qt.KeyboardModifier.ControlModifier |
                                Qt.KeyboardModifier.MetaModifier):
            at = self.screen_to_image(event.position().x(), event.position().y())
            (self.zoom_in if event.angleDelta().y() > 0 else self.zoom_out)(at)
            event.accept()
            return
        win = self.window()
        if self.doc.stack_size > 1 and hasattr(win, "step_slice"):
            win.step_slice(-1 if event.angleDelta().y() > 0 else 1)
            event.accept()
            return
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.scroll_by(event.angleDelta().y() / 2, 0)
        else:
            self.scroll_by(event.angleDelta().x() / 2, event.angleDelta().y() / 2)
        event.accept()


class _ImageData:
    """Adapter: a selection carrying ``image_data`` (WP-C Edit > Selection > Image to
    Selection) drawn like an ImageRoi."""

    def __init__(self, roi):
        x, y, w, h = roi.bounds()
        self.x, self.y, self.width, self.height = x, y, w, h
        self.pixels = roi.image_data
        props = getattr(roi, "properties", {}) or {}
        try:
            op = float(props.get("opacity", 100))
        except ValueError:
            op = 100.0
        self.opacity = op / 100.0 if op > 1 else op
        self.zero_transparent = str(props.get("zero_transparent", "False")) == "True"


def handle_points(roi) -> list[tuple[float, float]]:
    """Handle positions of a selection (ImageJ: 8 for rectangles/ovals, end points for lines,
    vertices for polygons)."""
    from rsstatchem.core.imaging.roi import shapes as S

    if isinstance(roi, (S.RectRoi, S.OvalRoi)):
        x, y, w, h = roi.x, roi.y, roi.width, roi.height
        return [(x, y), (x + w / 2, y), (x + w, y), (x + w, y + h / 2), (x + w, y + h),
                (x + w / 2, y + h), (x, y + h), (x, y + h / 2)]
    if isinstance(roi, S.LineRoi):
        return [(roi.x1, roi.y1), ((roi.x1 + roi.x2) / 2, (roi.y1 + roi.y2) / 2),
                (roi.x2, roi.y2)]
    if isinstance(roi, (S.EllipseRoi, S.RotatedRectRoi)):
        return [(roi.x1, roi.y1), (roi.x2, roi.y2)]
    if isinstance(roi, S.PolygonRoi) and roi.type in (S.POLYGON, S.POLYLINE, S.ANGLE):
        xs, ys = roi.polygon()
        return list(zip(xs.tolist(), ys.tolist()))
    return []


def roi_color(roi) -> str:
    return roi.stroke_color or "#ffff00"

"""Image > Stacks > Orthogonal Views (ImageJ ij/plugin/Orthogonal_Views): XZ and YZ views of
a stack through the crosshair point, scaled in z by the voxel depth, kept in sync with the
image window (the crosshair follows clicks and drags in the image; arrow keys / the spin boxes
move it). **Owned by WP-D.**
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QMainWindow, QSpinBox, QVBoxLayout,
                               QWidget)

from rsstatchem.core.imaging.lut import LUT, apply_lut, to_8bit
from rsstatchem.core.imaging.project3d import orthogonal_slices

_OPEN: dict[int, "OrthoViews"] = {}


def show_ortho_views(bar, image) -> "OrthoViews":
    from rsstatchem.ui.imagej import windows as W
    win = _OPEN.get(image.id)
    if win is None or win not in W.WINDOWS:
        win = OrthoViews(bar, image)
        _OPEN[image.id] = win
        W.register(win)
    win.show()
    win.raise_()
    return win


def _to_rgb(doc, plane: np.ndarray, channel: int) -> np.ndarray:
    if plane.ndim == 3:
        return np.ascontiguousarray(plane)
    lo, hi = doc.display_ranges[channel]
    lut = doc.luts[channel] if doc.composite_mode != "grayscale" else LUT.grays()
    return np.ascontiguousarray(apply_lut(to_8bit(plane, lo, hi), lut))


def _qimage(rgb: np.ndarray) -> QImage:
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class OrthoViews(QMainWindow):
    def __init__(self, bar, image):
        super().__init__()
        self.bar = bar
        self.session = bar.session
        self.doc = image
        self.x = image.width // 2
        self.y = image.height // 2
        self.setWindowTitle(f"Orthogonal Views of {image.title}")
        central = QWidget()
        lay = QVBoxLayout(central)
        row = QHBoxLayout()
        self.xs = QSpinBox()
        self.xs.setRange(0, image.width - 1)
        self.ys = QSpinBox()
        self.ys.setRange(0, image.height - 1)
        self.xs.setValue(self.x)
        self.ys.setValue(self.y)
        self.xs.valueChanged.connect(lambda v: self.set_point(v, self.y))
        self.ys.valueChanged.connect(lambda v: self.set_point(self.x, v))
        row.addWidget(QLabel("x:"))
        row.addWidget(self.xs)
        row.addWidget(QLabel("y:"))
        row.addWidget(self.ys)
        row.addStretch(1)
        lay.addLayout(row)
        views = QHBoxLayout()
        self.yz_label = QLabel()
        self.xz_label = QLabel()
        col = QVBoxLayout()
        col.addWidget(QLabel("XZ"))
        col.addWidget(self.xz_label)
        views.addLayout(col)
        col2 = QVBoxLayout()
        col2.addWidget(QLabel("YZ"))
        col2.addWidget(self.yz_label)
        views.addLayout(col2)
        lay.addLayout(views)
        self.setCentralWidget(central)
        self._hooked = None
        self._attach()
        self.session.on("image_updated", self._updated)
        self.session.on("image_closed", self._closed)
        self.refresh()

    # ------------------------------------------------------------------ sync with the image
    def _attach(self) -> None:
        from rsstatchem.ui.imagej.windows import window_for
        win = window_for(self.doc)
        if win is None:
            return
        self._hooked = win
        win.canvas.paint_hooks.append(self._paint_crosshair)
        win.canvas.installEventFilter(self)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        et = event.type()
        if et in (event.Type.MouseButtonPress, event.Type.MouseMove) and \
                (et == event.Type.MouseButtonPress or
                 event.buttons() & Qt.MouseButton.LeftButton):
            cv = self._hooked.canvas if self._hooked else None
            if cv is not None:
                x, y = cv.screen_to_image(event.position().x(), event.position().y())
                self.set_point(int(x), int(y))
        return False

    def _paint_crosshair(self, painter, canvas) -> None:
        painter.setPen(QPen(QColor(255, 255, 0), 1))
        x0, y0 = canvas.image_to_screen(self.x + 0.5, 0)
        x1, y1 = canvas.image_to_screen(self.x + 0.5, self.doc.height)
        painter.drawLine(QPointF(x0, y0), QPointF(x1, y1))
        x0, y0 = canvas.image_to_screen(0, self.y + 0.5)
        x1, y1 = canvas.image_to_screen(self.doc.width, self.y + 0.5)
        painter.drawLine(QPointF(x0, y0), QPointF(x1, y1))

    def set_point(self, x: int, y: int) -> None:
        self.x = min(max(int(x), 0), self.doc.width - 1)
        self.y = min(max(int(y), 0), self.doc.height - 1)
        for spin, v in ((self.xs, self.x), (self.ys, self.y)):
            spin.blockSignals(True)
            spin.setValue(v)
            spin.blockSignals(False)
        self.refresh()
        if self._hooked is not None:
            self._hooked.canvas.repaint_overlay()

    def _updated(self, doc) -> None:
        if doc is self.doc:
            self.refresh()

    def _closed(self, doc) -> None:
        if doc is self.doc:
            self.close()

    # ------------------------------------------------------------------ rendering
    def views(self) -> tuple[np.ndarray, np.ndarray]:
        """(XZ, YZ) RGB arrays as displayed (z scaled by voxel depth / pixel width)."""
        from rsstatchem.core.imaging import transform as T
        doc = self.doc
        c = doc.position[0]
        if doc.is_composite and doc.composite_mode == "composite":
            parts = []
            for ch in range(doc.n_channels):
                if not doc.active_channels[ch]:
                    continue
                xz, yz = orthogonal_slices(doc, self.x, self.y, ch)
                parts.append((_to_rgb(doc, xz, ch).astype(np.int32),
                              _to_rgb(doc, yz, ch).astype(np.int32)))
            xz = np.minimum(sum(p[0] for p in parts), 255).astype(np.uint8)
            yz = np.minimum(sum(p[1] for p in parts), 255).astype(np.uint8)
        else:
            xz, yz = orthogonal_slices(doc, self.x, self.y, c)
            xz, yz = _to_rgb(doc, xz, c), _to_rgb(doc, yz, c)
        cal = doc.calibration
        zscale = abs(cal.pixel_depth / (cal.pixel_width or 1.0))
        if zscale != 1.0 and doc.n_slices > 1:
            nz = max(1, int(round(doc.n_slices * zscale)))
            xz = T.resize_plane(xz, "rgb", xz.shape[1], nz, T.BILINEAR)
            yz = T.resize_plane(yz, "rgb", nz, yz.shape[0], T.BILINEAR)
        return np.ascontiguousarray(xz), np.ascontiguousarray(yz)

    def refresh(self) -> None:
        xz, yz = self.views()
        z = self.doc.position[1]
        zscale = xz.shape[0] / max(self.doc.n_slices, 1)
        for label, rgb, horizontal in ((self.xz_label, xz, True), (self.yz_label, yz, False)):
            img = _qimage(rgb)
            p = QPainter(img)
            p.setPen(QPen(QColor(255, 255, 0), 1))
            if horizontal:
                yy = (z + 0.5) * zscale
                p.drawLine(QPointF(0, yy), QPointF(img.width(), yy))
                p.drawLine(QPointF(self.x + 0.5, 0), QPointF(self.x + 0.5, img.height()))
            else:
                xx = (z + 0.5) * zscale
                p.drawLine(QPointF(xx, 0), QPointF(xx, img.height()))
                p.drawLine(QPointF(0, self.y + 0.5), QPointF(img.width(), self.y + 0.5))
            p.end()
            label.setPixmap(QPixmap.fromImage(img))

    def closeEvent(self, event) -> None:  # noqa: N802
        from rsstatchem.ui.imagej import windows as W
        self.session.off("image_updated", self._updated)
        self.session.off("image_closed", self._closed)
        if self._hooked is not None:
            try:
                self._hooked.canvas.paint_hooks.remove(self._paint_crosshair)
                self._hooked.canvas.removeEventFilter(self)
            except (ValueError, RuntimeError):
                pass
        W.unregister(self)
        _OPEN.pop(self.doc.id, None)
        event.accept()

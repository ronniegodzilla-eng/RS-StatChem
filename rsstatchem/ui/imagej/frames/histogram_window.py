"""The histogram window (ImageJ ``HistogramWindow``): the bars with the LUT ramp below, the
statistics (N, Mean, StdDev, Min, Max, Mode, Bins, Bin Width), the value/count under the mouse,
and the buttons **List** (table of bin starts and counts), **Copy**, **Log** (log-scaled
counts shown in grey behind the linear ones), **Live** (follows the selection), and for RGB
images **RGB** (Gray / Red / Green / Blue).
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel, QMainWindow,
                               QPushButton, QVBoxLayout, QWidget)

from rsstatchem.core.imaging.histogram import HistogramData, RGB_MODES
from rsstatchem.ui.imagej import windows as W
from rsstatchem.ui.imagej.frames.plot_window import valid


class HistogramCanvas(QWidget):
    W, H, LEFT, TOP = 256, 128, 10, 10

    def __init__(self, data: HistogramData, lut=None, parent=None):
        super().__init__(parent)
        self.data = data
        self.lut = lut
        self.log = False
        self.y_max = None
        self.setFixedSize(self.W + 2 * self.LEFT, self.H + self.TOP + 34)
        self.setMouseTracking(True)
        self.hover_cb = None

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(255, 255, 255))
        d = self.data
        counts = np.asarray(d.counts, float)
        n = len(counts)
        frame = QRectF(self.LEFT, self.TOP, self.W, self.H)
        if n:
            ymax = self.y_max or float(np.sort(counts)[-2] * 1.5 if n > 1 and
                                       counts.max() > 2 * np.sort(counts)[-2] else counts.max())
            ymax = ymax or 1.0
            xs = np.linspace(0, self.W, n + 1)
            if self.log:
                lg = np.log(np.maximum(counts, 1)) * (counts > 0)
                lmax = lg.max() or 1.0
                p.setPen(Qt.PenStyle.NoPen)
                for i in range(n):
                    h = lg[i] / lmax * self.H
                    p.fillRect(QRectF(self.LEFT + xs[i], self.TOP + self.H - h,
                                      max(xs[i + 1] - xs[i], 1), h), QColor(160, 160, 160))
            for i in range(n):
                h = min(counts[i] / ymax, 1.0) * self.H
                p.fillRect(QRectF(self.LEFT + xs[i], self.TOP + self.H - h,
                                  max(xs[i + 1] - xs[i], 1), h), QColor(0, 0, 0))
        p.setPen(QPen(QColor(0, 0, 0)))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(frame)
        ramp_top = self.TOP + self.H + 2
        lut = self.lut
        for x in range(self.W):
            idx = int(x * 255 / (self.W - 1))
            c = QColor(*(int(v) for v in lut[idx])) if lut is not None else QColor(idx, idx, idx)
            p.fillRect(QRectF(self.LEFT + x, ramp_top, 1, 10), c)
        p.drawRect(QRectF(self.LEFT, ramp_top, self.W, 10))
        p.setFont(QFont("SansSerif", 9))
        lo = d.bin_starts[0] if len(d.bin_starts) else 0
        hi = d.hist_max if d.integer_bins else d.hist_max
        p.drawText(QPointF(self.LEFT, ramp_top + 22), f"{lo:.4g}")
        txt = f"{hi:.4g}"
        p.drawText(QPointF(self.LEFT + self.W - 6 * len(txt), ramp_top + 22), txt)
        p.end()

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        x = e.position().x() - self.LEFT
        d = self.data
        if 0 <= x < self.W and len(d.counts) and self.hover_cb is not None:
            i = min(int(x / self.W * len(d.counts)), len(d.counts) - 1)
            v = d.bin_starts[i]
            self.hover_cb(f"bin start: {v:.4g}  count: {int(d.counts[i])}" if
                          not d.integer_bins or d.bin_width != 1 else
                          f"value: {v:g}  count: {int(d.counts[i])}")


class HistogramWindow(QMainWindow):
    def __init__(self, bar, image, data: HistogramData, source: dict | None = None):
        super().__init__()
        self.bar = bar
        self.image = image
        self.data = data
        self.source = source or {}
        self.setWindowTitle(f"Histogram of {data.title.rsplit('.', 1)[0] if data.title else ''}"
                            if self.source.get("kind") != "distribution" else data.title)
        central = QWidget(self)
        lay = QVBoxLayout(central)
        lut = None
        if image is not None and not image.is_rgb:
            lut = image.lut.as_array()
        self.canvas = HistogramCanvas(data, lut, central)
        if self.source.get("y_max"):
            self.canvas.y_max = self.source["y_max"]
        lay.addWidget(self.canvas, 0, Qt.AlignmentFlag.AlignHCenter)
        self.stats = QLabel(central)
        self.stats.setFont(QFont("Menlo", 10))
        self.stats.setText(data.stats_text())
        lay.addWidget(self.stats)
        self.hover = QLabel(" ", central)
        lay.addWidget(self.hover)
        self.canvas.hover_cb = self.hover.setText
        row = QHBoxLayout()
        b = QPushButton("List", central)
        b.clicked.connect(self.list_values)
        row.addWidget(b)
        b = QPushButton("Copy", central)
        b.clicked.connect(self.copy)
        row.addWidget(b)
        self.log_box = QCheckBox("Log", central)
        self.log_box.toggled.connect(self._log)
        row.addWidget(self.log_box)
        self.live = QCheckBox("Live", central)
        self.live.setEnabled(image is not None and self.source.get("kind") == "image")
        self.live.toggled.connect(self._live_toggled)
        row.addWidget(self.live)
        self.rgb = None
        if image is not None and image.is_rgb:
            self.rgb = QComboBox(central)
            self.rgb.addItems(RGB_MODES)
            self.rgb.currentTextChanged.connect(lambda _t: self.recompute())
            row.addWidget(self.rgb)
        lay.addLayout(row)
        self.setCentralWidget(central)
        self._listening = False

    @property
    def session(self):
        return getattr(self.bar, "session", None)

    def set_data(self, data: HistogramData) -> None:
        self.data = data
        self.canvas.data = data
        self.stats.setText(data.stats_text())
        self.canvas.update()

    def list_values(self):
        rt = self.data.table("Histogram of " + (self.data.title or ""))
        if self.session is not None:
            self.session.show_table(rt)
        return rt

    def copy(self) -> str:
        text = self.data.table().to_text("\t", row_numbers=False)
        QGuiApplication.clipboard().setText(text)
        return text

    def _log(self, on: bool) -> None:
        self.canvas.log = on
        self.canvas.update()

    def recompute(self) -> None:
        if self.image is None:
            return
        from rsstatchem.core.imaging.histogram import histogram
        src = self.source
        mode = self.rgb.currentText() if self.rgb is not None else "Gray"
        data = histogram(self.image, src.get("bins", 256), stack=src.get("stack", False),
                         rgb_mode=mode)
        self.set_data(data)

    def _live_toggled(self, on: bool) -> None:
        s = self.session
        if s is None:
            return
        if on and not self._listening:
            s.on("image_updated", self._image_updated)
            self._listening = True
        elif not on and self._listening:
            s.off("image_updated", self._image_updated)
            self._listening = False

    def _image_updated(self, doc) -> None:
        if valid(self) and doc is self.image:
            self.recompute()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._live_toggled(False)
        event.accept()
        W.unregister(self)


def show_histogram(bar, image, data: HistogramData, source: dict | None = None):
    win = HistogramWindow(bar, image, data, source)
    W.register(win)
    win.show()
    return win

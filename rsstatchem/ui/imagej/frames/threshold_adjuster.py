"""Image > Adjust > Threshold... window (ImageJ ``ij.plugin.frame.ThresholdAdjuster``). **WP-B.**

A non-modal window that follows the active image: histogram with the thresholded range in red,
lower/upper sliders on ImageJ's 0-255 scale (raw levels shown in the fields; 16/32-bit levels
are scaled with the display range like ``scaleAndSetThreshold``), the 17 methods, display mode
(Red / B&W / Over/Under), Dark background, Stack histogram, Don't reset range, Raw values,
16-bit histogram, and Auto / Apply / Reset / Set. The threshold is stored on the image
(``doc.threshold = Threshold(lower, upper, mode)``, the Phase 0 contract) and drawn on image
windows through the canvas ``paint_hooks`` (:func:`paint_threshold`), so it stays visible after
the window is closed, as in ImageJ.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QSizePolicy, QSlider, QVBoxLayout,
                               QWidget)

from rsstatchem.core.imaging import threshold as T
from rsstatchem.core.imaging.image import COLOR_RGB, GRAY8, Threshold
from rsstatchem.ui.imagej import windows as W

OVER_COLOR = (0, 255, 0)     # ImageJ defaults (Threshold "Over/Under" colours)
UNDER_COLOR = (0, 0, 255)


# ---------------------------------------------------------------------------- painting
def threshold_overlay(doc, black_background: bool = False) -> np.ndarray | None:
    """RGBA uint8 (H, W, 4) of the threshold display of the current plane, or None."""
    th = getattr(doc, "threshold", None)
    if th is None or th.mode == "none" or doc.type == COLOR_RGB:
        return None
    plane = doc.plane()
    inside = T.threshold_mask(plane, th.lower, th.upper)
    h, w = plane.shape
    out = np.zeros((h, w, 4), np.uint8)
    if th.mode == "red":
        out[inside] = (255, 0, 0, 255)
    elif th.mode == "bw":
        fg = 255 if black_background else 0
        out[..., :3] = 255 - fg
        out[..., 3] = 255
        out[inside, :3] = fg
    elif th.mode == "over_under":
        a = np.asarray(plane)
        lo = a < (np.float32(th.lower) if a.dtype == np.float32 else th.lower)
        hi = a > (np.float32(th.upper) if a.dtype == np.float32 else th.upper)
        out[lo] = UNDER_COLOR + (255,)
        out[hi] = OVER_COLOR + (255,)
    return out


def paint_threshold(painter: QPainter, canvas) -> None:
    """Canvas paint hook: draws the image's threshold (red / B&W / over-under)."""
    doc = canvas.doc
    bb = False
    bar = getattr(canvas.window(), "bar", None)
    if bar is not None:
        bb = bar.session.options.black_background
    rgba = threshold_overlay(doc, bb)
    if rgba is None:
        return
    h, w = rgba.shape[:2]
    rgba = np.ascontiguousarray(rgba)
    img = QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
    x0, y0 = canvas.image_to_screen(0, 0)
    x1, y1 = canvas.image_to_screen(w, h)
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
    painter.drawImage(QRectF(x0, y0, x1 - x0, y1 - y0), img)
    painter.restore()


def install_paint_hook(doc) -> None:
    win = W.window_for(doc)
    canvas = getattr(win, "canvas", None)
    if canvas is not None and paint_threshold not in canvas.paint_hooks:
        canvas.paint_hooks.append(paint_threshold)


# ---------------------------------------------------------------------------- histogram
class HistogramPlot(QWidget):
    """256-bin histogram with the thresholded range drawn in red (ImageJ ThresholdPlot)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.hist = np.zeros(256, np.int64)
        self.lower = -1
        self.upper = -1
        self.mode = "red"
        self.setMinimumSize(258, 82)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def set_data(self, hist, lower: float, upper: float, mode: str) -> None:
        self.hist = np.asarray(hist, np.int64)[:256]
        self.lower, self.upper, self.mode = lower, upper, mode
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(255, 255, 255))
        h = self.height() - 2
        hist = self.hist.astype(float)
        if hist.size and hist.max() > 0:
            # ImageJ clips the mode at 1.5x the second-highest count for display
            order = np.sort(hist)
            hmax = order[-1]
            if len(order) > 1 and order[-1] > 2 * order[-2] > 0:
                hmax = 1.5 * order[-2]
            for i, v in enumerate(hist):
                bar = int(round(min(v, hmax) / hmax * h)) if hmax else 0
                inside = self.lower >= 0 and self.lower <= i <= self.upper
                color = QColor(255, 0, 0) if inside and self.mode != "over_under" else \
                    QColor(0, 0, 0)
                if self.mode == "over_under" and self.lower >= 0:
                    color = QColor(*UNDER_COLOR) if i < self.lower else (
                        QColor(*OVER_COLOR) if i > self.upper else QColor(0, 0, 0))
                p.setPen(color)
                p.drawLine(i + 1, h + 1, i + 1, h + 1 - bar)
        p.setPen(QPen(QColor(0, 0, 0)))
        p.drawRect(0, 0, 257, self.height() - 1)
        p.end()


# ---------------------------------------------------------------------------- window
class ThresholdAdjuster(QWidget):
    instance: "ThresholdAdjuster | None" = None

    def __init__(self, bar):
        super().__init__(None, Qt.WindowType.Window)
        self.bar = bar
        self.session = bar.session
        self.doc = None
        self._busy = False
        self.setWindowTitle("Threshold")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        self.plot = HistogramPlot()
        lay.addWidget(self.plot, 0, Qt.AlignmentFlag.AlignHCenter)
        self.percent = QLabel(" ")
        self.percent.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        lay.addWidget(self.percent)
        grid = QGridLayout()
        self.min_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_field = QLineEdit()
        self.max_field = QLineEdit()
        for row, (s, f) in enumerate(((self.min_slider, self.min_field),
                                      (self.max_slider, self.max_field))):
            s.setRange(0, 255)
            f.setFixedWidth(80)
            grid.addWidget(s, row, 0)
            grid.addWidget(f, row, 1)
        self.max_slider.setValue(255)
        lay.addLayout(grid)
        row1 = QHBoxLayout()
        self.method = QComboBox()
        self.method.addItems(T.METHODS)
        m = self.session.options.threshold_method
        self.method.setCurrentText(m if m in T.METHODS else "Default")
        self.display = QComboBox()
        self.display.addItems(T.DISPLAY_MODES)
        row1.addWidget(self.method)
        row1.addWidget(self.display)
        lay.addLayout(row1)
        checks = QGridLayout()
        self.dark = QCheckBox("Dark background")
        self.stack = QCheckBox("Stack histogram")
        self.no_reset = QCheckBox("Don't reset range")
        self.raw = QCheckBox("Raw values")
        self.sixteen = QCheckBox("16-bit histogram")
        for i, cb in enumerate((self.dark, self.stack, self.no_reset, self.raw, self.sixteen)):
            checks.addWidget(cb, i // 2, i % 2)
        lay.addLayout(checks)
        buttons = QHBoxLayout()
        self.buttons = {}
        for name, fn in (("Auto", self.auto), ("Apply", self.apply),
                         ("Reset", self.reset_threshold), ("Set", self.set_dialog)):
            b = QPushButton(name)
            b.clicked.connect(lambda _=False, f=fn: f())
            buttons.addWidget(b)
            self.buttons[name] = b
        lay.addLayout(buttons)
        # signals
        self.min_slider.valueChanged.connect(self._min_moved)
        self.max_slider.valueChanged.connect(self._max_moved)
        self.method.currentTextChanged.connect(lambda _t: self.auto())
        self.dark.toggled.connect(lambda _b: self.auto())
        self.stack.toggled.connect(lambda _b: self.auto())
        self.sixteen.toggled.connect(lambda _b: self.auto())
        self.display.currentTextChanged.connect(lambda _t: self._mode_changed())
        self.min_field.returnPressed.connect(self._fields_entered)
        self.max_field.returnPressed.connect(self._fields_entered)
        self.session.on("active_changed", self._active_changed)
        self.session.on("image_updated", self._image_updated)
        self.resize(300, 330)

    # ------------------------------------------------------------------ opening
    @classmethod
    def show_for(cls, bar) -> "ThresholdAdjuster":
        win = cls.instance
        if win is None or win.bar is not bar:
            win = cls(bar)
            cls.instance = win
            W.register(win)
        win.setup(bar.session.active)
        win.show()
        win.raise_()
        return win

    @property
    def mode(self) -> str:
        return T.MODE_KEYS[self.display.currentText()]

    # ------------------------------------------------------------------ image state
    def _usable(self, doc) -> bool:
        return doc is not None and not doc.is_rgb and not (
            doc.is_composite and doc.composite_mode == "composite")

    def setup(self, doc) -> None:
        """Attach to ``doc``: show its threshold or compute the automatic one."""
        self.doc = doc if self._usable(doc) else None
        if self.doc is None:
            self.plot.set_data(np.zeros(256), -1, -1, self.mode)
            self.percent.setText("No grayscale image")
            return
        install_paint_hook(self.doc)
        th = self.doc.threshold
        if th is None or (th.mode == "none" and th.lower == th.upper):
            self.auto()
        else:
            self._sync_from_doc()

    def _active_changed(self, doc) -> None:
        if doc is not self.doc and self.isVisible():
            self.setup(doc)

    def _image_updated(self, doc) -> None:
        if doc is self.doc and not self._busy:
            self._sync_from_doc()

    def _scale_down(self, value: float) -> float:
        if self.doc.bit_depth == 8:
            return value
        vmin, vmax = self.doc.display_range
        if vmax > vmin:
            return min(max((value - vmin) / (vmax - vmin) * 255.0, 0.0), 255.0)
        return 0.0

    def histogram(self) -> np.ndarray:
        doc = self.doc
        if self.stack.isChecked() and doc.stack_size > 1:
            planes = [doc.plane_at(i) for i in range(doc.stack_size)]
            return T.stack_histogram(planes, doc.bit_depth)[0][:256]
        return T.image_histogram(doc.plane(), doc.bit_depth, doc.display_range)

    def _sync_from_doc(self) -> None:
        doc = self.doc
        if doc is None:
            return
        th = doc.threshold
        lo = self._scale_down(th.lower) if th is not None else -1
        hi = self._scale_down(th.upper) if th is not None else -1
        self._busy = True
        try:
            if th is not None:
                self.min_slider.setValue(int(round(lo)))
                self.max_slider.setValue(int(round(hi)))
                self.min_field.setText(_fmt(th.lower, doc))
                self.max_field.setText(_fmt(th.upper, doc))
                if th.mode in ("red", "bw", "over_under"):
                    names = {v: k for k, v in T.MODE_KEYS.items()}
                    self.display.blockSignals(True)
                    self.display.setCurrentText(names[th.mode])
                    self.display.blockSignals(False)
            hist = self.histogram()
            self.plot.set_data(hist, round(lo), round(hi), self.mode)
            self._update_percent(th)
        finally:
            self._busy = False

    def _update_percent(self, th) -> None:
        if th is None:
            self.percent.setText(" ")
            return
        mask = T.threshold_mask(self.doc.plane(), th.lower, th.upper)
        self.percent.setText(f"{100.0 * mask.mean():.2f} %")

    def _store(self, lower: float, upper: float) -> None:
        doc = self.doc
        lower, upper = T.java_threshold_levels(lower, upper, doc.bit_depth)
        doc.threshold = Threshold(lower, upper, self.mode)
        self._busy = True
        try:
            self.session.image_updated(doc)
        finally:
            self._busy = False
        self._sync_from_doc()

    def _scale_up_and_set(self, lo255: float, hi255: float) -> None:
        doc = self.doc
        if doc.bit_depth == 8:
            self._store(lo255, hi255)
            return
        vmin, vmax = doc.display_range
        lower, upper = T._scale_and_set(float(lo255), float(hi255), doc.bit_depth, vmin, vmax)
        self._store(lower, upper)

    # ------------------------------------------------------------------ actions
    def auto(self) -> None:
        if self.doc is None or self._busy:
            return
        doc = self.doc
        hist16 = self.sixteen.isChecked() and doc.bit_depth != 32
        T.set_auto_threshold(doc, self.method.currentText(), self.dark.isChecked(),
                             stack_histogram=self.stack.isChecked(),
                             no_reset=self.no_reset.isChecked(), sixteen_bit=hist16,
                             mode=self.mode)
        self.session.options.threshold_method = self.method.currentText()
        if self.session.recorder is not None:
            opts = self.method.currentText() + (" dark" if self.dark.isChecked() else "") + \
                (" 16-bit" if hist16 else "") + (" stack" if self.stack.isChecked() else "")
            self.session.recorder.record_line(f'setAutoThreshold("{opts}");')
        self._busy = True
        try:
            self.session.image_updated(doc)
        finally:
            self._busy = False
        self._sync_from_doc()

    def _min_moved(self, value: int) -> None:
        if self._busy or self.doc is None:
            return
        hi = self.max_slider.value()
        if hi < value:
            self._busy = True
            self.max_slider.setValue(value)
            self._busy = False
            hi = value
        self._scale_up_and_set(value, hi)

    def _max_moved(self, value: int) -> None:
        if self._busy or self.doc is None:
            return
        lo = self.min_slider.value()
        if lo > value:
            self._busy = True
            self.min_slider.setValue(value)
            self._busy = False
            lo = value
        self._scale_up_and_set(lo, value)

    def _mode_changed(self) -> None:
        if self.doc is None or self.doc.threshold is None:
            return
        th = self.doc.threshold
        self._store(th.lower, th.upper)

    def _fields_entered(self) -> None:
        try:
            lo = float(self.min_field.text())
            hi = float(self.max_field.text())
        except ValueError:
            return
        self.set_levels(lo, hi)

    def set_levels(self, lower: float, upper: float) -> None:
        """The Set button / typed levels: raw values (clamped to the image range for single
        images like ImageJ's doSet)."""
        if self.doc is None:
            return
        doc = self.doc
        if upper < lower:
            upper = lower
        if doc.stack_size == 1 and doc.bit_depth != 8:
            vmin, vmax = T.float_min_max(doc.plane())
            lower, upper = max(lower, vmin), min(upper, vmax)
        self._store(float(lower), float(upper))
        if self.session.recorder is not None:
            self.session.recorder.record_line(f"setThreshold({lower:g}, {upper:g});")

    def set_dialog(self) -> None:
        if self.doc is None:
            return
        from rsstatchem.core.imaging import commands as C
        from rsstatchem.ui.imagej.dialogs.generic import GenericDialog

        th = self.doc.threshold
        lo, hi = (th.lower, th.upper) if th is not None else self.doc.plane_min_max()
        digits = 4 if self.doc.type not in (GRAY8,) and self.doc.bit_depth == 32 else 0
        values = GenericDialog("Set Threshold Levels", [
            C.number("Lower threshold level", lo, digits, key="lower"),
            C.number("Upper threshold level", hi, digits, key="upper")], self).get()
        if values is not None:
            self.set_levels(values["lower"], values["upper"])

    def reset_threshold(self) -> None:
        if self.doc is None:
            return
        self.doc.threshold = None
        if self.doc.bit_depth != 8 and not self.no_reset.isChecked():
            self.doc.reset_display_range()
        if self.session.recorder is not None:
            self.session.recorder.record_line("resetThreshold();")
        self._busy = True
        try:
            self.session.image_updated(self.doc)
        finally:
            self._busy = False
        self._sync_from_doc()

    def apply(self) -> None:
        """Apply = Convert to Mask (ImageJ runs the Thresholder with the window's method)."""
        if self.doc is None:
            return
        if self.doc.threshold is None:
            self.bar.error("Threshold is not set", "Thresholder")
            return
        if self.session.active is not self.doc:
            self.session.set_active(self.doc)
        self.bar.run_command("Convert to Mask")
        self.setup(self.session.active)

    # ------------------------------------------------------------------ closing
    def closeEvent(self, event) -> None:  # noqa: N802
        self.session.off("active_changed", self._active_changed)
        self.session.off("image_updated", self._image_updated)
        W.unregister(self)
        if ThresholdAdjuster.instance is self:
            ThresholdAdjuster.instance = None
        event.accept()


def _fmt(value: float, doc) -> str:
    if doc.bit_depth == 32:
        return f"{value:.4g}" if abs(value) < 1e7 else f"{value:.3e}"
    return f"{value:.0f}"

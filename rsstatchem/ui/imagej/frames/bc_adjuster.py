"""Image > Adjust > Brightness/Contrast, Window/Level and Color Balance windows (ImageJ
ContrastAdjuster). **Owned by WP-D.**

One window per mode follows the session's active image: a histogram with the display-range
line, Minimum/Maximum/Brightness/Contrast sliders (Level/Window in W&L mode; a channel choice in
Color Balance mode) and Auto, Reset, Set and Apply buttons. The maths (auto contrast with
AUTO_THRESHOLD = 5000, slider mapping, apply tables) is in ``core/imaging/adjust.py``. RGB images
are adjusted destructively from a snapshot as in ImageJ; "Reset" restores it.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QComboBox, QGridLayout, QHBoxLayout, QLabel, QMainWindow,
                               QPushButton, QSlider, QVBoxLayout, QWidget)

from rsstatchem.core.imaging import adjust as A
from rsstatchem.core.imaging.image import COLOR_RGB, GRAY16, GRAY32

TITLES = {"bc": "B&C", "wl": "W&L", "balance": "Color"}
BALANCE_CHANNELS = ("Red", "Green", "Blue", "Cyan", "Magenta", "Yellow", "All")
_OPEN: dict[str, "ContrastAdjuster"] = {}


def show_adjuster(bar, mode: str = "bc") -> "ContrastAdjuster":
    from rsstatchem.ui.imagej import windows as W
    win = _OPEN.get(mode)
    if win is None or win not in W.WINDOWS:
        win = ContrastAdjuster(bar, mode)
        _OPEN[mode] = win
        W.register(win)
    win.set_image(bar.session.active)
    win.show()
    win.raise_()
    return win


class Histogram(QLabel):
    W, H = 256, 128

    def __init__(self):
        super().__init__()
        self.setFixedSize(self.W + 2, self.H + 2)
        self.hist = np.zeros(256)
        self.vmin = 0.0
        self.vmax = 255.0
        self.dmin = 0.0
        self.dmax = 255.0
        self.color = QColor(128, 128, 128)

    def set_data(self, hist, vmin, vmax, dmin, dmax, color=None):
        self.hist = np.asarray(hist, dtype=float)
        self.vmin, self.vmax, self.dmin, self.dmax = vmin, vmax, dmin, dmax
        if color is not None:
            self.color = color
        self._render()

    def _render(self):
        img = QImage(self.W + 2, self.H + 2, QImage.Format.Format_RGB32)
        img.fill(QColor(255, 255, 255))
        p = QPainter(img)
        p.setPen(QColor(0, 0, 0))
        p.drawRect(0, 0, self.W + 1, self.H + 1)
        h = self.hist.copy()
        if h.size and h.max() > 0:
            # ImageJ scales to the second-highest count when the maximum dominates
            srt = np.sort(h)
            top = srt[-1]
            if srt.size > 1 and srt[-2] > 0 and top > 2 * srt[-2]:
                top = srt[-2] * 1.5
            p.setPen(self.color)
            for i, v in enumerate(h[:256]):
                y = int(min(v / top, 1.0) * self.H)
                if y:
                    p.drawLine(i + 1, self.H, i + 1, self.H - y + 1)
        span = (self.dmax - self.dmin) or 1.0
        x0 = (self.vmin - self.dmin) / span * self.W
        x1 = (self.vmax - self.dmin) / span * self.W
        p.setPen(QPen(QColor(0, 0, 0), 1))
        p.drawLine(QPointF(x0 + 1, self.H), QPointF(x1 + 1, 1))
        p.end()
        self.setPixmap(QPixmap.fromImage(img))


class ContrastAdjuster(QMainWindow):
    def __init__(self, bar, mode: str = "bc"):
        super().__init__()
        self.bar = bar
        self.session = bar.session
        self.mode = mode
        self.image = None
        self.auto_threshold = 0
        self._rgb_snapshot = None
        self._updating = False
        self.setWindowTitle(TITLES.get(mode, "B&C"))
        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(6, 6, 6, 6)
        self.histogram = Histogram()
        lay.addWidget(self.histogram, 0, Qt.AlignmentFlag.AlignHCenter)
        self.range_label = QLabel("")
        self.range_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.range_label)
        grid = QGridLayout()
        self.sliders: dict[str, QSlider] = {}
        names = {"bc": ("Minimum", "Maximum", "Brightness", "Contrast"),
                 "wl": ("Level", "Window"),
                 "balance": ("Minimum", "Maximum", "Brightness")}[mode]
        for row, name in enumerate(names):
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(0, A.SLIDER_RANGE)
            s.valueChanged.connect(lambda v, n=name: self._slider_moved(n, v))
            grid.addWidget(QLabel(name), row, 0)
            grid.addWidget(s, row, 1)
            self.sliders[name] = s
        lay.addLayout(grid)
        self.channel = None
        if mode == "balance":
            self.channel = QComboBox()
            self.channel.addItems(BALANCE_CHANNELS)
            self.channel.currentTextChanged.connect(lambda _t: self.refresh())
            lay.addWidget(self.channel)
        buttons = QHBoxLayout()
        self.buttons = {}
        for name in ("Auto", "Reset", "Set", "Apply"):
            b = QPushButton(name)
            b.clicked.connect(lambda _=False, n=name: self._button(n))
            buttons.addWidget(b)
            self.buttons[name] = b
        lay.addLayout(buttons)
        self.setCentralWidget(central)
        self.session.on("active_changed", self._active_changed)
        self.session.on("image_updated", self._image_updated)
        self.resize(300, 360)

    # ------------------------------------------------------------------ image tracking
    def _active_changed(self, image) -> None:
        if self.isVisible():
            self.set_image(image)

    def _image_updated(self, image) -> None:
        if image is self.image and not self._updating and self.isVisible():
            self.refresh()

    def set_image(self, image) -> None:
        if image is not self.image:
            self._rgb_snapshot = None
            self.auto_threshold = 0
        self.image = image
        self.refresh()

    def _current_range(self):
        im = self.image
        if im.type == COLOR_RGB:
            return im.display_ranges[0]
        if self.mode == "balance" and im.is_composite and self.channel is not None:
            name = self.channel.currentText()
            idx = BALANCE_CHANNELS.index(name) if name in BALANCE_CHANNELS[:3] else None
            if idx is not None and idx < im.n_channels:
                return im.display_ranges[idx]
        return im.display_range

    def defaults(self):
        im = self.image
        if im.bit_depth in (16, 32):
            lo, hi = im.plane_min_max()
            return (lo, hi) if hi > lo else (lo, lo + 1)
        return 0.0, 255.0

    def refresh(self) -> None:
        im = self.image
        enabled = im is not None
        for w in list(self.sliders.values()) + list(self.buttons.values()):
            w.setEnabled(enabled)
        if not enabled:
            self.range_label.setText("(no image)")
            return
        vmin, vmax = self._current_range()
        dmin, dmax = self.defaults()
        plane = im.plane()
        if im.type == COLOR_RGB and self._rgb_snapshot is not None:
            plane = self._rgb_snapshot
        if im.type == COLOR_RGB and self.mode == "balance" and self.channel is not None:
            name = self.channel.currentText()
            k = {"Red": 0, "Green": 1, "Blue": 2}.get(name)
            if k is not None:
                plane = np.ascontiguousarray(plane[..., k])
        hist, *_ = A.histogram256(plane)
        if im.type in (GRAY16, GRAY32):
            # the B&C histogram spans the default range
            v = plane.ravel().astype(np.float64)
            v = v[np.isfinite(v)]
            span = (dmax - dmin) or 1.0
            idx = np.clip(((v - dmin) / span * 256).astype(int), 0, 255)
            hist = np.bincount(idx, minlength=256)
        color = QColor(128, 128, 128)
        if im.is_composite:
            r, g, b = im.lut.as_array()[255]
            color = QColor(int(r), int(g), int(b))
        self.histogram.set_data(hist, vmin, vmax, dmin, dmax, color)
        if im.calibration.calibrated() and im.type != COLOR_RGB:
            lo, hi = im.calibration.cvalue(vmin), im.calibration.cvalue(vmax)
        else:
            lo, hi = vmin, vmax
        self.range_label.setText(f"{_fmt(lo)}  -  {_fmt(hi)}")
        self._updating = True
        try:
            span = (dmax - dmin) or 1.0
            b, c = A.sliders_from_range(vmin, vmax, dmin, dmax)
            vals = {"Minimum": (vmin - dmin) / span * A.SLIDER_RANGE,
                    "Maximum": (vmax - dmin) / span * A.SLIDER_RANGE,
                    "Brightness": b, "Contrast": c,
                    "Level": ((vmin + vmax) / 2 - dmin) / span * A.SLIDER_RANGE,
                    "Window": (vmax - vmin) / span * A.SLIDER_RANGE / 2}
            for name, s in self.sliders.items():
                s.blockSignals(True)
                s.setValue(int(round(min(max(vals[name], 0), A.SLIDER_RANGE))))
                s.blockSignals(False)
        finally:
            self._updating = False

    # ------------------------------------------------------------------ changes
    def _slider_moved(self, name: str, value: int) -> None:
        im = self.image
        if im is None or self._updating:
            return
        vmin, vmax = self._current_range()
        dmin, dmax = self.defaults()
        span = dmax - dmin
        if name == "Minimum":
            vmin = min(dmin + span * value / A.SLIDER_RANGE, vmax)
        elif name == "Maximum":
            vmax = max(dmin + span * value / A.SLIDER_RANGE, vmin)
        elif name == "Brightness":
            vmin, vmax = A.range_from_brightness(value, vmin, vmax, dmin, dmax)
        elif name == "Contrast":
            vmin, vmax = A.range_from_contrast(value, vmin, vmax, dmin, dmax)
        elif name == "Level":
            w, _ = A.window_level(vmin, vmax)
            vmin, vmax = A.range_from_window_level(w, dmin + span * value / A.SLIDER_RANGE)
        elif name == "Window":
            _, lev = A.window_level(vmin, vmax)
            vmin, vmax = A.range_from_window_level(2 * span * value / A.SLIDER_RANGE, lev)
        self.set_range(vmin, vmax)

    def set_range(self, vmin: float, vmax: float) -> None:
        im = self.image
        if im is None:
            return
        if im.type == COLOR_RGB and self.mode == "balance":
            if self._rgb_snapshot is None:
                self._rgb_snapshot = im.plane().copy()
            ch = self.channel.currentText() if self.channel is not None else "All"
            im.set_plane(A.color_balance(self._rgb_snapshot, ch, vmin, vmax))
            im.display_ranges = [(vmin, vmax)]
        elif im.type == COLOR_RGB:
            im.display_ranges = [(float(vmin), float(vmax))]
        elif self.mode == "balance" and im.is_composite and self.channel is not None and \
                self.channel.currentText() in BALANCE_CHANNELS[:3]:
            k = BALANCE_CHANNELS.index(self.channel.currentText())
            if k < im.n_channels:
                im.set_display_range(vmin, vmax, k)
        else:
            im.set_display_range(vmin, vmax)
        self._updating = True
        try:
            self.session.image_updated(im)
        finally:
            self._updating = False
        self.refresh()

    def _button(self, name: str) -> None:
        im = self.image
        if im is None:
            return
        if name == "Auto":
            plane = self._rgb_snapshot if self._rgb_snapshot is not None else im.plane()
            mask = None
            if im.roi is not None and im.roi.is_area():
                mask = im.roi.get_mask((im.height, im.width))
            res = A.auto_range(plane, self.auto_threshold, mask)
            if res is None:
                self._button("Reset")
                return
            lo, hi, self.auto_threshold = res
            self.set_range(lo, hi)
            self.session.record("Enhance Contrast", "saturated=0.35")
        elif name == "Reset":
            if im.type == COLOR_RGB and self._rgb_snapshot is not None:
                im.set_plane(self._rgb_snapshot)
                self._rgb_snapshot = None
            self.auto_threshold = 0
            self.set_range(*self.defaults())
        elif name == "Set":
            self._set_dialog()
        elif name == "Apply":
            self.apply()

    def _set_dialog(self) -> None:
        from rsstatchem.core.imaging import commands as C
        from rsstatchem.ui.imagej.dialogs.generic import GenericDialog
        im = self.image
        vmin, vmax = self._current_range()
        cal = im.calibration
        if cal.calibrated() and im.type != COLOR_RGB:
            vmin, vmax = cal.cvalue(vmin), cal.cvalue(vmax)
        digits = 0 if im.bit_depth in (8, 16, 24) and not cal.calibrated() else 4
        fields = [C.number("Minimum displayed value: ", vmin, digits, key="minimum"),
                  C.number("Maximum displayed value: ", vmax, digits, key="maximum"),
                  C.checkbox("Propagate to all other open images", False, key="propagate")]
        if im.is_composite:
            fields.append(C.checkbox(f"Propagate to the other {im.n_channels - 1} channels "
                                     "of this image", False, key="all"))
        vals = GenericDialog("Set Display Range", fields, self).get()
        if vals is None:
            return
        lo, hi = float(vals["minimum"]), float(vals["maximum"])
        if cal.calibrated() and im.type != COLOR_RGB:
            lo, hi = cal.raw_value(lo, im.bit_depth), cal.raw_value(hi, im.bit_depth)
        self.set_range(lo, hi)
        if vals.get("all"):
            im.display_ranges = [(lo, hi)] * im.n_channels
        if vals.get("propagate"):
            for other in self.session.images:
                if other is not im and other.type == im.type:
                    other.set_display_range(lo, hi)
                    self.session.image_updated(other)
        if self.session.recorder is not None:
            self.session.recorder.record_line(f"setMinAndMax({_fmt(lo)}, {_fmt(hi)});")

    def apply(self) -> None:
        im = self.image
        if im.type == GRAY32:
            self.bar.error('"Apply" does not work with 32-bit images', "B&C")
            return
        if im.type == COLOR_RGB and self._rgb_snapshot is not None:
            self._rgb_snapshot = None          # colour balance already applied
            im.display_ranges = [(0.0, 255.0)]
            im.changed = True
            self.session.image_updated(im)
            self.refresh()
            return
        im.snapshot("Apply LUT")
        A.apply_lut(im, all_slices=im.stack_size == 1)
        im.changed = True
        self.session.record("Apply LUT", "")
        self.session.image_updated(im)
        self.refresh()

    def closeEvent(self, event) -> None:  # noqa: N802
        from rsstatchem.ui.imagej import windows as W
        self.session.off("active_changed", self._active_changed)
        self.session.off("image_updated", self._image_updated)
        W.unregister(self)
        if _OPEN.get(self.mode) is self:
            _OPEN.pop(self.mode, None)
        event.accept()


def _fmt(v: float) -> str:
    v = float(v)
    if v == int(v) and abs(v) < 1e9:
        return f"{v:.0f}"
    return f"{v:.1f}" if abs(v) >= 100 else f"{v:.4g}"

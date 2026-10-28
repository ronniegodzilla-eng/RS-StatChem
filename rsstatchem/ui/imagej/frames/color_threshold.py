"""Image > Adjust > Color Threshold... window (ImageJ ``ij.plugin.frame.ColorThresholder``).
**WP-B.**

Thresholds the current plane of an RGB image in HSB, RGB, CIE Lab or YUV: one histogram per
channel with min/max sliders and a Pass checkbox (unchecked = stop band), the thresholding
method + Dark background for the automatic ranges, the threshold colour (Red / White / Black /
B&W), and the buttons Original, Filtered, Select, Sample, Stack, Macro and Help. While the
window is open the result is previewed on the image through the canvas ``paint_hooks``;
"Filtered" writes it into the pixels (undoable), "Original" restores them, "Select" turns the
mask into an exact area selection.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout,
                               QLabel, QMessageBox, QPushButton, QSizePolicy, QSlider,
                               QVBoxLayout, QWidget)

from rsstatchem.core.imaging import threshold as T
from rsstatchem.core.imaging.image import COLOR_RGB
from rsstatchem.ui.imagej import windows as W

HELP = ("Color Thresholder\n\n"
        "[Pass]: everything within range is thresholded, otherwise everything outside the "
        "range is thresholded.\n"
        "[Default] [Huang] ...: selects the automatic thresholding method.\n"
        "[Red] [White] [Black] [B&W]: selects the threshold colour.\n"
        "[Dark background]: auto-thresholding methods assume light features and dark "
        "background.\n"
        "[Original]: shows the original image.\n[Filtered]: shows the filtered image.\n"
        "[Stack]: processes all slices with the current settings.\n"
        "[Macro]: records the current settings.\n"
        "[Sample]: sets the ranges from the pixels in the selection.\n"
        "[HSB] [RGB] [Lab] [YUV]: selects the colour space.")


class BandPlot(QWidget):
    """Histogram of one channel with its band (ImageJ BandPlot)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.hist = np.zeros(256)
        self.colors = np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, axis=1)
        self.lo, self.hi = 0, 255
        self.setMinimumSize(258, 66)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def set_data(self, hist, colors) -> None:
        self.hist = np.asarray(hist, float)
        self.colors = colors
        self.update()

    def set_band(self, lo: int, hi: int) -> None:
        self.lo, self.hi = lo, hi
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(255, 255, 255))
        h = self.height() - 2
        hmax = self.hist.max() * 1.15 if self.hist.size and self.hist.max() > 0 else 1.0
        for i, v in enumerate(self.hist):
            r, g, b = (int(c) for c in self.colors[i])
            p.setPen(QColor(r, g, b))
            p.drawLine(i + 1, h + 1, i + 1, h + 1 - int(v / hmax * h))
        p.setPen(QColor(0, 0, 0))
        p.drawRect(0, 0, 257, self.height() - 1)
        p.setPen(QColor(255, 0, 0))
        p.drawLine(self.lo + 1, 0, self.lo + 1, h + 1)
        p.drawLine(self.hi + 1, 0, self.hi + 1, h + 1)
        p.end()


def _channel_colors(space: str, k: int) -> np.ndarray:
    i = np.arange(256)
    z = np.zeros(256, int)
    if space == "RGB":
        cols = [np.stack([i, z, z], 1), np.stack([z, i, z], 1), np.stack([z, z, i], 1)][k]
    elif space == "HSB":
        if k == 0:      # spectrum, Color.getHSBColor(i/255, 1, 1)
            import colorsys
            cols = np.array([[int(c * 255 + 0.5) for c in colorsys.hsv_to_rgb(v / 255.0, 1, 1)]
                             for v in range(256)])
        elif k == 1:
            cols = np.stack([np.full(256, 255), 255 - i, 255 - i], 1)
        else:
            cols = np.stack([i, i, i], 1)
    elif space == "Lab":
        cols = [np.stack([i, i, i], 1), np.stack([i, 255 - i, z], 1),
                np.stack([i, i, 255 - i], 1)][k]
    else:
        cols = [np.stack([i, i, i], 1),
                np.stack([(36 + (255 - i) / 1.4).astype(int), 255 - i, i], 1),
                np.stack([i, 255 - i, (83 + (255 - i) / 2.87).astype(int)], 1)][k]
    return np.clip(cols, 0, 255).astype(np.uint8)


class ColorThresholdWindow(QWidget):
    instance: "ColorThresholdWindow | None" = None

    def __init__(self, bar):
        super().__init__(None, Qt.WindowType.Window)
        self.bar = bar
        self.session = bar.session
        self.doc = None
        self.original_plane = None
        self.chans = None
        self.preview = True
        self._busy = False
        self.setWindowTitle("Threshold Color")
        lay = QVBoxLayout(self)
        self.plots, self.mins, self.maxs, self.passes, self.labels = [], [], [], [], []
        for k in range(3):
            box = QGroupBox()
            g = QGridLayout(box)
            plot = BandPlot()
            g.addWidget(plot, 0, 0, 1, 3)
            lo, hi = QSlider(Qt.Orientation.Horizontal), QSlider(Qt.Orientation.Horizontal)
            for s, v in ((lo, 0), (hi, 255)):
                s.setRange(0, 255)
                s.setValue(v)
                s.valueChanged.connect(lambda _v, kk=k: self._slider_moved(kk))
            lab = QLabel()
            pas = QCheckBox("Pass")
            pas.setChecked(True)
            pas.toggled.connect(lambda _b: self._changed())
            g.addWidget(lo, 1, 0, 1, 2)
            g.addWidget(hi, 2, 0, 1, 2)
            g.addWidget(pas, 1, 2)
            g.addWidget(lab, 2, 2)
            self.plots.append(plot)
            self.mins.append(lo)
            self.maxs.append(hi)
            self.passes.append(pas)
            self.labels.append(box)
            lay.addWidget(box)
        opts = QGridLayout()
        opts.addWidget(QLabel("Thresholding method:"), 0, 0)
        self.method = QComboBox()
        self.method.addItems(T.METHODS)
        opts.addWidget(self.method, 0, 1)
        opts.addWidget(QLabel("Threshold color:"), 1, 0)
        self.color = QComboBox()
        self.color.addItems(T.COLOR_MODES)
        opts.addWidget(self.color, 1, 1)
        opts.addWidget(QLabel("Color space:"), 2, 0)
        self.space = QComboBox()
        self.space.addItems(T.COLOR_SPACES)
        opts.addWidget(self.space, 2, 1)
        self.dark = QCheckBox("Dark background")
        opts.addWidget(self.dark, 3, 0, 1, 2)
        lay.addLayout(opts)
        row = QHBoxLayout()
        self.buttons = {}
        for name, fn in (("Original", self.original), ("Filtered", self.filtered),
                         ("Select", self.select), ("Sample", self.sample),
                         ("Stack", self.stack), ("Macro", self.macro), ("Help", self.help)):
            b = QPushButton(name)
            b.clicked.connect(lambda _=False, f=fn: f())
            row.addWidget(b)
            self.buttons[name] = b
        lay.addLayout(row)
        self.space.currentTextChanged.connect(lambda _t: self._space_changed())
        self.method.currentTextChanged.connect(lambda _t: self.auto_ranges())
        self.dark.toggled.connect(lambda _b: self.auto_ranges())
        self.color.currentTextChanged.connect(lambda _t: self._changed())
        self.session.on("active_changed", self._active_changed)
        self._update_titles()

    # ------------------------------------------------------------------ opening
    @classmethod
    def show_for(cls, bar) -> "ColorThresholdWindow":
        win = cls.instance
        if win is None or win.bar is not bar:
            win = cls(bar)
            cls.instance = win
            W.register(win)
        win.setup(bar.session.active)
        win.show()
        win.raise_()
        return win

    def setup(self, doc) -> None:
        self._remove_hook()
        if doc is None or doc.type != COLOR_RGB:
            self.doc = None
            self.bar.show_status("No RGB image")
            return
        self.doc = doc
        self.original_plane = doc.plane().copy()
        win = W.window_for(doc)
        canvas = getattr(win, "canvas", None)
        if canvas is not None and self._paint not in canvas.paint_hooks:
            canvas.paint_hooks.append(self._paint)
        self._compute_channels()
        self.auto_ranges()

    def _active_changed(self, doc) -> None:
        if doc is not self.doc and doc is not None and doc.type == COLOR_RGB:
            self.setup(doc)

    def _compute_channels(self) -> None:
        space = self.space.currentText()
        self.chans = T.color_channels(self.original_plane, space)
        for k, ch in enumerate(self.chans):
            hist = np.bincount(np.asarray(ch).ravel(), minlength=256)[:256]
            self.plots[k].set_data(hist, _channel_colors(space, k))
        self._update_titles()

    def _update_titles(self) -> None:
        names = T.CHANNEL_NAMES[self.space.currentText()]
        for box, name in zip(self.labels, names):
            box.setTitle(name)

    # ------------------------------------------------------------------ ranges
    def ranges(self) -> list[tuple[int, int, bool]]:
        return [(self.mins[k].value(), self.maxs[k].value(), self.passes[k].isChecked())
                for k in range(3)]

    def set_range(self, k: int, lo: int, hi: int, passing: bool | None = None) -> None:
        self._busy = True
        try:
            self.mins[k].setValue(int(lo))
            self.maxs[k].setValue(int(hi))
            if passing is not None:
                self.passes[k].setChecked(bool(passing))
        finally:
            self._busy = False
        self._changed()

    def auto_ranges(self) -> None:
        if self.chans is None:
            return
        rng = T.color_auto_ranges(self.chans, self.space.currentText(),
                                  self.method.currentText(), self.dark.isChecked())
        self._busy = True
        try:
            for k, (lo, hi) in enumerate(rng):
                self.mins[k].setValue(lo)
                self.maxs[k].setValue(hi)
        finally:
            self._busy = False
        self._changed()

    def _slider_moved(self, k: int) -> None:
        if self._busy:
            return
        if self.mins[k].value() > self.maxs[k].value():
            self._busy = True
            self.maxs[k].setValue(self.mins[k].value())
            self._busy = False
        self._changed()

    def _space_changed(self) -> None:
        if self.doc is None:
            self._update_titles()
            return
        self._compute_channels()
        self.auto_ranges()

    def current_mask(self) -> np.ndarray | None:
        if self.chans is None:
            return None
        return T.color_threshold(self.chans, self.space.currentText(), self.ranges())

    def _changed(self) -> None:
        for k in range(3):
            self.plots[k].set_band(self.mins[k].value(), self.maxs[k].value())
        if self.doc is not None:
            win = W.window_for(self.doc)
            if win is not None:
                win.canvas.update()

    # ------------------------------------------------------------------ preview
    def _paint(self, painter: QPainter, canvas) -> None:
        if not self.preview or canvas.doc is not self.doc or self.chans is None:
            return
        mask = self.current_mask()
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), np.uint8)
        mode = self.color.currentText()
        if mode == "B&W":
            fg = 255 if self.session.options.black_background else 0
            rgba[..., :3] = 255 - fg
            rgba[..., 3] = 255
            rgba[mask, :3] = fg
        else:
            col = {"Red": (255, 0, 0), "White": (255, 255, 255), "Black": (0, 0, 0)}[mode]
            rgba[mask] = col + (255,)
        img = QImage(np.ascontiguousarray(rgba).data, w, h, 4 * w,
                     QImage.Format.Format_RGBA8888)
        x0, y0 = canvas.image_to_screen(0, 0)
        x1, y1 = canvas.image_to_screen(w, h)
        painter.drawImage(QRectF(x0, y0, x1 - x0, y1 - y0), img)

    def _remove_hook(self) -> None:
        for win in W.image_windows():
            canvas = getattr(win, "canvas", None)
            if canvas is not None and self._paint in canvas.paint_hooks:
                canvas.paint_hooks.remove(self._paint)

    # ------------------------------------------------------------------ buttons
    def _updated(self) -> None:
        self.session.image_updated(self.doc)

    def original(self) -> None:
        if self.doc is None:
            return
        self.doc.set_plane(self.original_plane)
        self.preview = True
        self._updated()

    def filtered(self) -> None:
        if self.doc is None:
            return
        self.doc.snapshot("Color Threshold")
        mask = self.current_mask()
        self.doc.set_plane(T.apply_color_mode(self.original_plane, mask,
                                              self.color.currentText(),
                                              self.session.options.black_background))
        self.doc.changed = True
        self.preview = False
        self._updated()

    def select(self) -> None:
        if self.doc is None:
            return
        self.doc.set_plane(self.original_plane)
        self.doc.roi = T.mask_to_roi(self.current_mask())
        self.preview = True
        self._updated()

    def sample(self) -> None:
        if self.doc is None:
            return
        roi = self.doc.roi
        if roi is None or not roi.is_area():
            self.bar.error("Selection required", "Threshold Color")
            return
        sel = roi.get_mask((self.doc.height, self.doc.width))
        res = T.sample_ranges(self.chans, sel, self.space.currentText())
        self.doc.roi = None
        if res is None:
            return
        for k, (lo, hi, passing) in enumerate(res):
            self.set_range(k, lo, hi, passing)
        self._updated()

    def stack(self) -> None:
        """Filter every slice with the current settings."""
        if self.doc is None:
            return
        doc = self.doc
        doc.snapshot("Color Threshold")
        space = self.space.currentText()
        for c, z, t in doc.iter_planes(True):
            plane = self.original_plane if (c, z, t) == doc.position else doc.plane(c, z, t)
            mask = T.color_threshold(plane, space, self.ranges())
            doc.set_plane(T.apply_color_mode(plane, mask, self.color.currentText(),
                                             self.session.options.black_background), c, z, t)
        doc.changed = True
        self.preview = False
        self._updated()

    def macro_options(self) -> str:
        parts = [f"space={self.space.currentText()}"]
        for k, (lo, hi, passing) in enumerate(self.ranges(), start=1):
            parts.append(f"min{k}={lo} max{k}={hi} filter{k}={'pass' if passing else 'stop'}")
        parts.append(f"color={self.color.currentText()}")
        return " ".join(parts)

    def macro(self) -> None:
        rec = self.session.recorder
        if rec is None:
            self.bar.error("Command recorder is not running", "Threshold Color")
            return
        rec.record("Color Threshold...", self.macro_options() + " filtered")

    def help(self) -> None:
        QMessageBox.information(self, "Help", HELP)

    # ------------------------------------------------------------------ closing
    def closeEvent(self, event) -> None:  # noqa: N802
        self._remove_hook()
        self.session.off("active_changed", self._active_changed)
        W.unregister(self)
        if ColorThresholdWindow.instance is self:
            ColorThresholdWindow.instance = None
        if self.doc is not None:
            win = W.window_for(self.doc)
            if win is not None:
                win.canvas.update()
        event.accept()

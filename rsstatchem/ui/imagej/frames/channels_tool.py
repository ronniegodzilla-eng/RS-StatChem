"""Image > Color > Channels Tool (ImageJ ij/plugin/frame/Channels): display mode of the active
composite (Composite / Color / Grayscale), the channel check boxes and a "More" menu with the
related Image > Color commands and channel colours. **Owned by WP-D.**
"""
from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QComboBox, QMainWindow, QMenu, QPushButton,
                               QVBoxLayout, QWidget)

from rsstatchem.core.imaging.lut import named_lut

MODES = ("Composite", "Color", "Grayscale")
COLORS = ("Red", "Green", "Blue", "Grays", "Cyan", "Magenta", "Yellow")
_OPEN: list = []


def show_channels_tool(bar) -> "ChannelsTool":
    from rsstatchem.ui.imagej import windows as W
    win = _OPEN[0] if _OPEN and _OPEN[0] in W.WINDOWS else None
    if win is None:
        win = ChannelsTool(bar)
        _OPEN[:] = [win]
        W.register(win)
    win.set_image(bar.session.active)
    win.show()
    win.raise_()
    return win


class ChannelsTool(QMainWindow):
    def __init__(self, bar):
        super().__init__()
        self.bar = bar
        self.session = bar.session
        self.image = None
        self.setWindowTitle("Channels")
        central = QWidget()
        self.lay = QVBoxLayout(central)
        self.mode = QComboBox()
        self.mode.addItems(MODES)
        self.mode.currentTextChanged.connect(self._mode_changed)
        self.lay.addWidget(self.mode)
        self.boxes_widget = QWidget()
        self.boxes_layout = QVBoxLayout(self.boxes_widget)
        self.boxes_layout.setContentsMargins(0, 0, 0, 0)
        self.lay.addWidget(self.boxes_widget)
        self.boxes: list[QCheckBox] = []
        self.more = QPushButton("More >>")
        menu = QMenu(self.more)
        for label in ("Make Composite", "Convert to RGB", "Split Channels", "Merge Channels...",
                      "Arrange Channels...", "Invert LUTs", "Edit LUT..."):
            act = menu.addAction(label)
            act.triggered.connect(lambda _=False, lb=label: self._more(lb))
        menu.addSeparator()
        for color in COLORS:
            act = menu.addAction(color)
            act.triggered.connect(lambda _=False, c=color: self._set_color(c))
        self.more.setMenu(menu)
        self.lay.addWidget(self.more)
        self.setCentralWidget(central)
        self.session.on("active_changed", self._active_changed)
        self.session.on("image_updated", self._updated)
        self.resize(180, 220)

    def _active_changed(self, image) -> None:
        if self.isVisible():
            self.set_image(image)

    def _updated(self, image) -> None:
        if image is self.image and self.isVisible():
            self._sync()

    def set_image(self, image) -> None:
        self.image = image
        for b in self.boxes:
            b.setParent(None)
        self.boxes = []
        if image is not None and image.is_composite:
            for c in range(image.n_channels):
                box = QCheckBox(f"Channel {c + 1}")
                box.toggled.connect(lambda on, k=c: self._toggle(k, on))
                self.boxes_layout.addWidget(box)
                self.boxes.append(box)
        self._sync()

    def _sync(self) -> None:
        im = self.image
        ok = im is not None and im.is_composite
        self.mode.setEnabled(ok)
        if not ok:
            return
        self.mode.blockSignals(True)
        self.mode.setCurrentText(im.composite_mode.capitalize())
        self.mode.blockSignals(False)
        for c, box in enumerate(self.boxes):
            box.blockSignals(True)
            box.setChecked(c < len(im.active_channels) and im.active_channels[c])
            box.setEnabled(im.composite_mode == "composite")
            box.blockSignals(False)

    def _mode_changed(self, text: str) -> None:
        im = self.image
        if im is None or not im.is_composite:
            return
        im.composite_mode = text.lower()
        self.session.record("Make Composite" if text == "Composite" else "Channels Tool...",
                            "")
        self.session.image_updated(im)

    def _toggle(self, c: int, on: bool) -> None:
        im = self.image
        if im is None:
            return
        active = list(im.active_channels)
        active[c] = on
        im.active_channels = active
        if self.session.recorder is not None:
            flags = "".join("1" if a else "0" for a in active)
            self.session.recorder.record_line(f'Stack.setActiveChannels("{flags}");')
        self.session.image_updated(im)

    def _more(self, label: str) -> None:
        if label == "Convert to RGB":
            self.bar.run_command("Stack to RGB")
        else:
            self.bar.run_command(label)

    def _set_color(self, color: str) -> None:
        im = self.image
        if im is None or im.is_rgb:
            return
        luts = list(im.luts)
        luts[im.position[0]] = named_lut(color)
        im.luts = luts
        self.session.image_updated(im)

    def closeEvent(self, event) -> None:  # noqa: N802
        from rsstatchem.ui.imagej import windows as W
        self.session.off("active_changed", self._active_changed)
        self.session.off("image_updated", self._updated)
        W.unregister(self)
        if self in _OPEN:
            _OPEN.remove(self)
        event.accept()

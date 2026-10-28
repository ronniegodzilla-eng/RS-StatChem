"""The Color Picker window (``ij/plugin/frame/ColorPicker.java``): a palette of hues, shades
and greys; click = foreground colour, alt- or right-click = background colour; the swatches
show both and can be swapped. Used by double-clicking the colour-picker tool and by Image >
Color > Color Picker... (WP-D's command calls :func:`show_color_picker`). **Owned by WP-A.**"""
from __future__ import annotations

import colorsys

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QWidget

COLS, ROWS, CELL = 10, 20, 14


def palette() -> list[list[str]]:
    """ROWS x COLS colours: 10 hues by 16 brightness/saturation steps, then greys."""
    rows = []
    for r in range(ROWS - 4):
        row = []
        for c in range(COLS):
            h = c / COLS
            t = r / (ROWS - 5)
            s, v = (min(1.0, 2 * t), 1.0) if t <= 0.5 else (1.0, max(0.0, 2 - 2 * t))
            rr, gg, bb = colorsys.hsv_to_rgb(h, s if t > 0 else 0.0, v)
            row.append(f"#{int(rr * 255):02x}{int(gg * 255):02x}{int(bb * 255):02x}")
        rows.append(row)
    for r in range(4):
        row = []
        for c in range(COLS):
            g = int(round(255 * (1 - (r * COLS + c) / (4 * COLS - 1))))
            row.append(f"#{g:02x}{g:02x}{g:02x}")
        rows.append(row)
    return rows


class _Palette(QWidget):
    picked = Signal(str, bool)

    def __init__(self):
        super().__init__()
        self.colors = palette()
        self.setFixedSize(COLS * CELL, ROWS * CELL)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        for r, row in enumerate(self.colors):
            for c, col in enumerate(row):
                p.fillRect(QRectF(c * CELL, r * CELL, CELL, CELL), QColor(col))
        p.setPen(QPen(QColor(90, 90, 90)))
        p.drawRect(0, 0, self.width() - 1, self.height() - 1)
        p.end()

    def color_at(self, x: float, y: float) -> str | None:
        c, r = int(x // CELL), int(y // CELL)
        if 0 <= r < ROWS and 0 <= c < COLS:
            return self.colors[r][c]
        return None

    def mousePressEvent(self, event) -> None:  # noqa: N802
        col = self.color_at(event.position().x(), event.position().y())
        if col:
            bg = event.button() == Qt.MouseButton.RightButton or \
                bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
            self.picked.emit(col, bg)


class _Swatch(QWidget):
    def __init__(self, picker):
        super().__init__()
        self.picker = picker
        self.setFixedSize(COLS * CELL, 40)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        o = self.picker.bar.session.options
        p.fillRect(QRectF(30, 14, 40, 22), QColor(o.background))
        p.setPen(QPen(QColor(0, 0, 0)))
        p.drawRect(QRectF(30, 14, 40, 22))
        p.fillRect(QRectF(10, 4, 40, 22), QColor(o.foreground))
        p.drawRect(QRectF(10, 4, 40, 22))
        p.drawText(QRectF(80, 4, 60, 32), Qt.AlignmentFlag.AlignVCenter,
                   o.foreground + "\n" + o.background)
        p.end()


class ColorPicker(QMainWindow):
    def __init__(self, bar):
        super().__init__()
        self.bar = bar
        self.setWindowTitle("CP")
        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(4, 4, 4, 4)
        self.swatch = _Swatch(self)
        lay.addWidget(self.swatch)
        self.palette = _Palette()
        self.palette.picked.connect(self.pick)
        lay.addWidget(self.palette)
        swap = QPushButton("Swap")
        swap.clicked.connect(self.swap)
        lay.addWidget(swap)
        self.setCentralWidget(central)
        bar.session.on("options_changed", lambda *_: self.swatch.update())

    def pick(self, color: str, background: bool) -> None:
        self.bar.set_color(color, background)
        self.swatch.update()

    def swap(self) -> None:
        o = self.bar.session.options
        fg, bg = o.foreground, o.background
        self.bar.set_color(bg, False)
        self.bar.set_color(fg, True)
        self.swatch.update()

    def closeEvent(self, event) -> None:  # noqa: N802
        from rsstatchem.ui.imagej import windows as W
        event.accept()
        W.unregister(self)


def show_color_picker(bar) -> ColorPicker:
    from rsstatchem.ui.imagej import windows as W
    for w in W.WINDOWS:
        if isinstance(w, ColorPicker):
            w.show()
            w.raise_()
            return w
    win = ColorPicker(bar)
    W.register(win)
    win.show()
    return win

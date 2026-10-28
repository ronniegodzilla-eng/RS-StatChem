"""Image > Color > Edit LUT... dialog (ImageJ LUT_Editor, simplified): shows the 256 colours,
sets a colour or a colour gradient over an index range, inverts, and opens named LUTs.
**Owned by WP-D.**
"""
from __future__ import annotations

import numpy as np
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (QColorDialog, QComboBox, QDialog, QDialogButtonBox, QGridLayout,
                               QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout)

from rsstatchem.core.imaging.lut import LUT, NAMED_LUTS


class LutEditor(QDialog):
    def __init__(self, lut: LUT, parent=None):
        super().__init__(parent)
        self.setWindowTitle("LUT Editor")
        self._table = lut.as_array().astype(np.uint8).copy()
        self._name = lut.name
        lay = QVBoxLayout(self)
        self.strip = QLabel()
        lay.addWidget(self.strip)
        grid = QGridLayout()
        self.first = QSpinBox()
        self.first.setRange(0, 255)
        self.last = QSpinBox()
        self.last.setRange(0, 255)
        self.last.setValue(255)
        self.c1 = QColor(0, 0, 0)
        self.c2 = QColor(255, 255, 255)
        self.b1 = QPushButton("Start colour")
        self.b2 = QPushButton("End colour")
        self.b1.clicked.connect(lambda: self._pick(1))
        self.b2.clicked.connect(lambda: self._pick(2))
        grid.addWidget(QLabel("First index:"), 0, 0)
        grid.addWidget(self.first, 0, 1)
        grid.addWidget(self.b1, 0, 2)
        grid.addWidget(QLabel("Last index:"), 1, 0)
        grid.addWidget(self.last, 1, 1)
        grid.addWidget(self.b2, 1, 2)
        lay.addLayout(grid)
        row = QHBoxLayout()
        grad = QPushButton("Gradient")
        grad.clicked.connect(self.gradient)
        inv = QPushButton("Invert")
        inv.clicked.connect(self.invert)
        self.named = QComboBox()
        self.named.addItems(["(open LUT)"] + list(NAMED_LUTS))
        self.named.currentTextChanged.connect(self._open_named)
        row.addWidget(grad)
        row.addWidget(inv)
        row.addWidget(self.named)
        lay.addLayout(row)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self._render()

    def _pick(self, which: int) -> None:
        start = self.c1 if which == 1 else self.c2
        c = QColorDialog.getColor(start, self, "Colour")
        if c.isValid():
            if which == 1:
                self.c1 = c
            else:
                self.c2 = c

    def gradient(self) -> None:
        a, b = self.first.value(), self.last.value()
        if b < a:
            a, b = b, a
        n = b - a + 1
        t = np.linspace(0.0, 1.0, n)[:, None]
        c1 = np.array([self.c1.red(), self.c1.green(), self.c1.blue()], float)
        c2 = np.array([self.c2.red(), self.c2.green(), self.c2.blue()], float)
        self._table[a:b + 1] = np.round(c1 + (c2 - c1) * t).astype(np.uint8)
        self._render()

    def invert(self) -> None:
        self._table = self._table[::-1].copy()
        self._render()

    def _open_named(self, name: str) -> None:
        if name in NAMED_LUTS:
            lut = NAMED_LUTS[name]()
            self._table = lut.as_array().copy()
            self._name = lut.name
            self._render()

    def _render(self) -> None:
        strip = np.repeat(self._table[None, :, :], 24, axis=0)
        strip = np.ascontiguousarray(strip)
        img = QImage(strip.data, 256, 24, 3 * 256, QImage.Format.Format_RGB888).copy()
        self.strip.setPixmap(QPixmap.fromImage(img).scaled(512, 24))

    def lut(self) -> LUT:
        return LUT.from_array(self._table, self._name or "Edited")

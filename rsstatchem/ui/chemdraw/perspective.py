"""*Structure Perspective*: a rotatable 3D view of the selected molecule (RDKit ETKDG conformer,
MMFF94-optimised; see :mod:`rsstatchem.core.chem_draw.threed`). Drag to rotate (Shift: turn
in the plane), wheel to zoom; *Apply to Drawing* replaces the drawing by the rotated
projection, as ChemDraw's Structure Perspective tool does."""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QHBoxLayout, QLabel,
                               QPushButton, QVBoxLayout, QWidget)

from rsstatchem.core.chem_draw import threed

__all__ = ["MoleculeView3D", "StructurePerspectiveDialog", "STYLES"]

STYLES = ("Ball and Stick", "Stick", "Space Filling", "Wire Frame")


class MoleculeView3D(QWidget):
    """Painter-based 3D view (depth-sorted spheres and bonds, no OpenGL needed)."""

    rotated = Signal()

    def __init__(self, model: threed.Model3D, parent=None):
        super().__init__(parent)
        self.model = model
        self.yaw, self.pitch, self.roll = 0.0, 0.0, 0.0
        self.zoom = 1.0
        self.show_h = True
        self.style = STYLES[0]
        self._last = None
        self.setMinimumSize(360, 300)
        self.setMouseTracking(False)
        self.setAutoFillBackground(False)

    def rotation(self) -> np.ndarray:
        return threed.rotation(self.yaw, self.pitch, self.roll)

    def reset(self) -> None:
        self.yaw = self.pitch = self.roll = 0.0
        self.zoom = 1.0
        self.update()
        self.rotated.emit()

    def set_angles(self, yaw: float, pitch: float, roll: float = 0.0) -> None:
        self.yaw, self.pitch, self.roll = yaw, pitch, roll
        self.update()
        self.rotated.emit()

    # --- interaction ---------------------------------------------------------------------
    def mousePressEvent(self, e) -> None:  # noqa: N802
        self._last = e.position()

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        if self._last is None:
            return
        d = e.position() - self._last
        self._last = e.position()
        if e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self.roll += d.x() * 0.01
        else:
            self.yaw += d.x() * 0.01
            self.pitch += d.y() * 0.01
        self.update()
        self.rotated.emit()

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        self._last = None

    def wheelEvent(self, e) -> None:  # noqa: N802
        self.zoom = max(0.3, min(5.0, self.zoom * (1.12 if e.angleDelta().y() > 0 else
                                                   1 / 1.12)))
        self.update()

    # --- painting ------------------------------------------------------------------------
    def _scale(self, pts) -> float:
        ext = float(np.max(np.abs(pts[:, :2]))) if len(pts) else 1.0
        return 0.42 * min(self.width(), self.height()) / max(ext + 1.2, 2.0) * self.zoom

    def paintEvent(self, e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QColor("#FFFFFF"))
        m = self.model
        if not m.symbols:
            p.end()
            return
        pts = threed.project(m, self.rotation())
        k = self._scale(m.coords)
        cx, cy = self.width() / 2, self.height() / 2
        zmin, zmax = float(pts[:, 2].min()), float(pts[:, 2].max())

        def screen(i):
            return QPointF(cx + pts[i][0] * k, cy - pts[i][1] * k)

        def depth_fade(z):              # far atoms a little lighter (depth cue)
            return 0.0 if zmax - zmin < 1e-6 else 0.35 * (zmax - z) / (zmax - zmin)

        visible = [i for i, s in enumerate(m.symbols) if self.show_h or s != "H"]
        vis = set(visible)
        style = self.style
        if style == "Space Filling":
            radius = {i: threed.VDW.get(m.symbols[i], 1.8) for i in visible}
        elif style == "Ball and Stick":
            radius = {i: 0.28 * threed.VDW.get(m.symbols[i], 1.8) ** 0.8 for i in visible}
        else:
            radius = {i: 0.0 for i in visible}
        items = []
        if style != "Space Filling":
            for a, b, order in m.bonds:
                if a in vis and b in vis:
                    items.append(((pts[a][2] + pts[b][2]) / 2 - 0.01, "bond", (a, b, order)))
        for i in visible:
            items.append((pts[i][2], "atom", i))
        items.sort(key=lambda it: it[0])
        bond_w = {"Ball and Stick": 0.12, "Stick": 0.22, "Wire Frame": 0.04}.get(style, 0.1)
        for z, what, data in items:
            if what == "bond":
                a, b, order = data
                pa, pb = screen(a), screen(b)
                dx, dy = pb.x() - pa.x(), pb.y() - pa.y()
                L = math.hypot(dx, dy) or 1.0
                nx, ny = -dy / L, dx / L
                n = 1 if style == "Stick" or order == 1 else (3 if order == 3 else 2)
                w = max(1.2, bond_w * k / (1.6 if n > 1 else 1))
                gap = w * 1.6
                fade = depth_fade(z)
                for j in range(n):
                    off = (j - (n - 1) / 2) * gap
                    o = QPointF(nx * off, ny * off)
                    mid = QPointF((pa.x() + pb.x()) / 2, (pa.y() + pb.y()) / 2) + o
                    for end, idx in ((pa + o, a), (pb + o, b)):
                        col = QColor(threed.CPK.get(m.symbols[idx], "#C080C0")) \
                            if style in ("Stick", "Wire Frame") else QColor("#8C8C8C")
                        col = col.darker(115) if col.lightness() > 240 else col
                        col = _fade(col, fade)
                        pen = QPen(col, w)
                        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                        p.setPen(pen)
                        p.drawLine(end, mid)
            else:
                i = data
                r = radius[i] * k
                if r <= 0:
                    continue
                c = screen(i)
                base = _fade(QColor(threed.CPK.get(m.symbols[i], "#C080C0")), depth_fade(z))
                g = QRadialGradient(c.x() - r * 0.35, c.y() - r * 0.35, r * 1.3)
                g.setColorAt(0.0, QColor("#FFFFFF"))
                g.setColorAt(0.45, base)
                g.setColorAt(1.0, base.darker(170))
                p.setBrush(QBrush(g))
                p.setPen(QPen(base.darker(200), 0.6))
                p.drawEllipse(QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r))
        p.setPen(QColor("#7A7A7A"))
        p.drawText(QRectF(6, self.height() - 20, self.width() - 12, 16),
                   Qt.AlignmentFlag.AlignLeft, f"3D model: RDKit {m.method}")
        p.end()


def _fade(c: QColor, f: float) -> QColor:
    return QColor(int(c.red() + (255 - c.red()) * f), int(c.green() + (255 - c.green()) * f),
                  int(c.blue() + (255 - c.blue()) * f))


class StructurePerspectiveDialog(QDialog):
    """Modeless window around :class:`MoleculeView3D` for one molecule of a drawing."""

    applied = Signal()

    def __init__(self, canvas, model: threed.Model3D, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Structure Perspective")
        self.setObjectName("chemdraw_perspective")
        self.canvas = canvas
        self.view = MoleculeView3D(model, self)
        self.style_box = QComboBox()
        self.style_box.addItems(STYLES)
        self.style_box.currentTextChanged.connect(self._style)
        self.h_box = QCheckBox("Show Hydrogens")
        self.h_box.setChecked(True)
        self.h_box.toggled.connect(self._show_h)
        self.angles = QLabel()
        self.view.rotated.connect(self._show_angles)
        reset = QPushButton("Reset")
        reset.clicked.connect(self.view.reset)
        self.apply_btn = QPushButton("Apply to Drawing")
        self.apply_btn.setToolTip("Replace the drawing of this molecule by the rotated "
                                  "projection (stereochemistry is kept)")
        self.apply_btn.clicked.connect(self.apply)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        top = QHBoxLayout()
        top.addWidget(QLabel("Display:"))
        top.addWidget(self.style_box)
        top.addWidget(self.h_box)
        top.addStretch(1)
        top.addWidget(self.angles)
        bottom = QHBoxLayout()
        bottom.addWidget(QLabel("Drag to rotate, Shift+drag to turn, wheel to zoom."))
        bottom.addStretch(1)
        for b in (reset, self.apply_btn, close):
            bottom.addWidget(b)
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.view, 1)
        lay.addLayout(bottom)
        self.resize(560, 520)
        self._show_angles()

    def _style(self, text: str) -> None:
        self.view.style = text
        self.view.update()

    def _show_h(self, on: bool) -> None:
        self.view.show_h = on
        self.view.update()

    def _show_angles(self) -> None:
        v = self.view
        self.angles.setText(f"x {math.degrees(v.pitch) % 360:.0f}°  y "
                            f"{math.degrees(v.yaw) % 360:.0f}°  z "
                            f"{math.degrees(v.roll) % 360:.0f}°")

    def apply(self) -> bool:
        model, rot = self.view.model, self.view.rotation()
        doc = self.canvas.doc
        if not all(a in doc.atoms for a in model.atoms):
            self.angles.setText("The molecule was changed; reopen the view.")
            return False
        ok = self.canvas.edit("Structure Perspective",
                              lambda: threed.apply_to_drawing(doc, model, rot))
        if ok:
            self.applied.emit()
        return bool(ok)

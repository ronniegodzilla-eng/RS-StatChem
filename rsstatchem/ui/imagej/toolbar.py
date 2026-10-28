"""The row of tool icons in the ImageJ bar. **Owned by WP-A after Phase 0** (variants on
right-click, double-click options, ImageJ-like icons, the ">>" tool sets).

Contract: ``ToolBar.current`` (tool id), ``set_tool(id)``, signal ``tool_changed(int)``; tool
ids are ImageJ's ``ij.gui.Toolbar`` constants below.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QToolButton, QWidget

RECTANGLE, OVAL, POLYGON, FREEROI, LINE, POLYLINE, FREELINE, POINT, WAND, TEXT = range(10)
MAGNIFIER, HAND, DROPPER, ANGLE = 11, 12, 13, 14
# (id, tooltip) in ImageJ's left-to-right order
TOOLS = ((RECTANGLE, "Rectangle"), (OVAL, "Oval"), (POLYGON, "Polygon"),
         (FREEROI, "Freehand"), (LINE, "Straight Line"), (ANGLE, "Angle"),
         (POINT, "Point"), (WAND, "Wand (tracing)"), (TEXT, "Text"),
         (MAGNIFIER, "Magnifying glass"), (HAND, "Scrolling"), (DROPPER, "Color picker"))


def tool_icon(tool: int, size: int = 22) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QPen(QColor(30, 30, 30), 1.4))
    s = size
    r = QRectF(4, 6, s - 8, s - 12)
    if tool == RECTANGLE:
        p.drawRect(r)
    elif tool == OVAL:
        p.drawEllipse(r)
    elif tool == POLYGON:
        p.drawPolygon(QPolygonF([QPointF(4, 8), QPointF(12, 4), QPointF(s - 4, 10),
                                 QPointF(s - 7, s - 5), QPointF(6, s - 6)]))
    elif tool == FREEROI:
        p.drawEllipse(QRectF(4, 5, s - 9, s - 11))
    elif tool == LINE:
        p.drawLine(QPointF(4, s - 5), QPointF(s - 4, 5))
    elif tool == ANGLE:
        p.drawLine(QPointF(4, s - 5), QPointF(s - 4, s - 5))
        p.drawLine(QPointF(4, s - 5), QPointF(s - 6, 5))
    elif tool == POINT:
        p.drawLine(QPointF(s / 2, 4), QPointF(s / 2, s - 4))
        p.drawLine(QPointF(4, s / 2), QPointF(s - 4, s / 2))
    elif tool == WAND:
        p.drawLine(QPointF(5, s - 4), QPointF(s - 7, 7))
        p.drawEllipse(QPointF(s - 6, 6), 2.5, 2.5)
    elif tool == TEXT:
        f = p.font()
        f.setBold(True)
        f.setPointSize(int(s * 0.55))
        p.setFont(f)
        p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, "A")
    elif tool == MAGNIFIER:
        p.drawEllipse(QPointF(s / 2 - 2, s / 2 - 2), 6, 6)
        p.drawLine(QPointF(s / 2 + 2, s / 2 + 2), QPointF(s - 4, s - 4))
    elif tool == HAND:
        p.drawRoundedRect(QRectF(6, 8, s - 12, s - 10), 3, 3)
        for i in range(4):
            p.drawLine(QPointF(7 + i * 3, 9), QPointF(7 + i * 3, 4))
    elif tool == DROPPER:
        p.drawLine(QPointF(5, s - 4), QPointF(s - 8, 8))
        p.drawRect(QRectF(s - 9, 4, 5, 5))
    p.end()
    return QIcon(pm)


class ToolBar(QWidget):
    tool_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(1)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons: dict[int, QToolButton] = {}
        for tool, name in TOOLS:
            b = QToolButton(self)
            b.setCheckable(True)
            b.setIcon(tool_icon(tool))
            b.setIconSize(QSize(22, 22))
            b.setToolTip(name)
            b.setAutoRaise(True)
            self.group.addButton(b, tool)
            self.buttons[tool] = b
            lay.addWidget(b)
        lay.addStretch(1)
        self.current = RECTANGLE
        self.buttons[RECTANGLE].setChecked(True)
        self.group.idClicked.connect(self.set_tool)

    def set_tool(self, tool: int) -> None:
        if tool in self.buttons:
            self.buttons[tool].setChecked(True)
        if tool != self.current:
            self.current = tool
            self.tool_changed.emit(tool)

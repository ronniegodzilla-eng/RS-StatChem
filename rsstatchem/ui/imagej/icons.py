"""Tool icons of the ImageJ bar, drawn with QPainter in the style of ImageJ 1.54's toolbar
(thin dark outlines on a light button), including the variants of the rectangle, oval, line
and point tools and the colour-picker swatch."""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF

INK = QColor(30, 30, 30)

_CACHE: dict = {}


def tool_icon(tool: int, variant: str = "", size: int = 22, fg: str = "#ffffff",
              bg: str = "#000000") -> QIcon:
    key = (tool, variant, size, fg if tool == 13 else "", bg if tool == 13 else "")
    icon = _CACHE.get(key)
    if icon is None:
        icon = QIcon(tool_pixmap(tool, variant, size, fg, bg))
        _CACHE[key] = icon
    return icon


def clear_cache() -> None:
    _CACHE.clear()


def tool_pixmap(tool: int, variant: str = "", size: int = 22, fg: str = "#ffffff",
                bg: str = "#000000") -> QPixmap:
    from rsstatchem.ui.imagej import toolbar as T

    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QPen(INK, 1.3))
    s = float(size)
    r = QRectF(3.5, 6.5, s - 7, s - 13)
    if tool == T.RECTANGLE:
        if variant == "Rounded Rectangle":
            p.drawRoundedRect(r, 4, 4)
        elif variant == "Rotated Rectangle":
            p.drawPolygon(QPolygonF([QPointF(4, 12), QPointF(12, 4), QPointF(s - 3, 11),
                                     QPointF(s - 11, s - 3)]))
        else:
            p.drawRect(r)
    elif tool == T.OVAL:
        if variant == "Elliptical selections":
            p.save()
            p.translate(s / 2, s / 2)
            p.rotate(-30)
            p.drawEllipse(QRectF(-s / 2 + 2, -s / 4, s - 4, s / 2))
            p.restore()
        elif variant == "Selection Brush Tool":
            p.setBrush(QBrush(INK))
            p.drawEllipse(QPointF(s / 2, s / 2 + 3), 4, 4)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawLine(QPointF(s / 2 + 2, s / 2), QPointF(s - 4, 4))
        else:
            p.drawEllipse(r)
    elif tool == T.POLYGON:
        p.drawPolygon(QPolygonF([QPointF(3, 9), QPointF(11, 4), QPointF(s - 3, 9),
                                 QPointF(s - 6, s - 4), QPointF(8, s - 7)]))
    elif tool == T.FREEROI:
        path = QPainterPath(QPointF(5, 10))
        path.cubicTo(QPointF(2, 2), QPointF(14, 2), QPointF(s - 4, 7))
        path.cubicTo(QPointF(s, 14), QPointF(10, s - 1), QPointF(7, s - 6))
        path.cubicTo(QPointF(5, s - 8), QPointF(7, 12), QPointF(5, 10))
        p.drawPath(path)
    elif tool in (T.LINE, T.POLYLINE, T.FREELINE):
        if variant == "Segmented Line" or tool == T.POLYLINE:
            p.drawPolyline(QPolygonF([QPointF(3, s - 5), QPointF(8, 6), QPointF(14, s - 7),
                                      QPointF(s - 3, 5)]))
        elif variant == "Freehand Line" or tool == T.FREELINE:
            path = QPainterPath(QPointF(3, s - 6))
            path.cubicTo(QPointF(7, 2), QPointF(12, s), QPointF(s - 3, 5))
            p.drawPath(path)
        elif variant == "Arrow tool":
            p.drawLine(QPointF(4, s - 4), QPointF(s - 6, 6))
            p.setBrush(QBrush(INK))
            p.drawPolygon(QPolygonF([QPointF(s - 3, 3), QPointF(s - 10, 6),
                                     QPointF(s - 6, 10)]))
        else:
            p.drawLine(QPointF(3, s - 5), QPointF(s - 3, 5))
    elif tool == T.ANGLE:
        p.drawLine(QPointF(3, s - 5), QPointF(s - 3, s - 5))
        p.drawLine(QPointF(3, s - 5), QPointF(s - 6, 4))
        path = QPainterPath()
        path.arcMoveTo(QRectF(-5, s - 15, 20, 20), 0)
        path.arcTo(QRectF(-5, s - 15, 20, 20), 0, 55)
        p.drawPath(path)
    elif tool == T.POINT:
        def cross(cx, cy, k=4.0):
            p.drawLine(QPointF(cx - k, cy), QPointF(cx + k, cy))
            p.drawLine(QPointF(cx, cy - k), QPointF(cx, cy + k))
        if variant == "Multi-point Tool":
            cross(6, 7, 3)
            cross(s - 6, 9, 3)
            cross(s / 2, s - 6, 3)
        else:
            cross(s / 2, s / 2, 5)
            p.fillRect(QRectF(s / 2 - 1, s / 2 - 1, 2, 2), QColor(255, 255, 255))
    elif tool == T.WAND:
        p.drawLine(QPointF(4, s - 3), QPointF(s - 9, 9))
        for a, b in ((QPointF(s - 6, 2), QPointF(s - 6, 6)), (QPointF(s - 8, 4),
                                                             QPointF(s - 4, 4)),
                     (QPointF(s - 3, 7), QPointF(s - 3, 10)), (QPointF(s - 9, 1),
                                                             QPointF(s - 10, 2))):
            p.drawLine(a, b)
    elif tool == T.TEXT:
        f = p.font()
        f.setBold(True)
        f.setPixelSize(int(s * 0.62))
        p.setFont(f)
        p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, "A")
    elif tool == T.MAGNIFIER:
        p.drawEllipse(QPointF(s / 2 - 2, s / 2 - 2), 6, 6)
        p.setPen(QPen(INK, 2.4))
        p.drawLine(QPointF(s / 2 + 3, s / 2 + 3), QPointF(s - 3, s - 3))
    elif tool == T.HAND:
        path = QPainterPath(QPointF(7, s - 3))
        path.lineTo(5, 13)
        path.lineTo(5, 9)
        for i, top in enumerate((5, 3, 4, 7)):
            x = 7 + i * 3
            path.lineTo(x - 1, 10)
            path.lineTo(x - 1, top)
            path.lineTo(x + 1, top)
            path.lineTo(x + 1, 10)
        path.lineTo(s - 4, 12)
        path.lineTo(s - 6, s - 3)
        path.closeSubpath()
        p.drawPath(path)
    elif tool == T.DROPPER:
        p.fillRect(QRectF(s / 2 - 1, s / 2 - 1, s / 2 - 2, s / 2 - 2), QColor(bg))
        p.drawRect(QRectF(s / 2 - 1, s / 2 - 1, s / 2 - 2, s / 2 - 2))
        p.fillRect(QRectF(2, 2, s / 2, s / 2), QColor(fg))
        p.drawRect(QRectF(2, 2, s / 2, s / 2))
    p.end()
    return pm

"""Tool palette icons drawn with QPainter (no image files), in the style of ChemDraw's
tool palette: bond types, chain, ring templates, arrows, charges, text and selection."""
from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QBrush, QColor, QFont, QIcon, QPainter, QPainterPath, QPen, QPixmap,
                           QPolygonF)

__all__ = ["tool_icon"]

_S = 32
_CACHE: dict[str, QIcon] = {}


def _pen(w=1.6, style=Qt.PenStyle.SolidLine, color="#222222"):
    p = QPen(QColor(color), w, style)
    p.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return p


def _ring(p: QPainter, n: int, inner: tuple = ()) -> None:
    r = 11.0
    start = -math.pi / 2 + (math.pi / n if n % 4 == 0 else 0)
    pts = [QPointF(16 + r * math.cos(start + 2 * math.pi * k / n),
                   16 + r * math.sin(start + 2 * math.pi * k / n)) for k in range(n)]
    p.setPen(_pen(1.5))
    p.drawPolygon(QPolygonF(pts))
    for k in inner:
        a, b = pts[k], pts[(k + 1) % n]
        c = QPointF(16, 16)
        f = 0.25
        a2 = a + (c - a) * f
        b2 = b + (c - b) * f
        m = (a2 + b2) / 2
        p.drawLine(m + (a2 - m) * 0.8, m + (b2 - m) * 0.8)


def _arrow_head(p: QPainter, tip: QPointF, ux: float, uy: float, half: int = 0) -> None:
    hl, hw = 7.0, 3.2
    nx, ny = -uy, ux
    back = QPointF(tip.x() - ux * hl, tip.y() - uy * hl)
    notch = QPointF(tip.x() - ux * hl * 0.75, tip.y() - uy * hl * 0.75)
    left = QPointF(back.x() + nx * hw, back.y() + ny * hw)
    right = QPointF(back.x() - nx * hw, back.y() - ny * hw)
    pts = [tip, left, notch] if half > 0 else ([tip, notch, right] if half < 0 else
                                                [tip, left, notch, right])
    p.setBrush(QBrush(QColor("#222222")))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawPolygon(QPolygonF(pts))
    p.setBrush(Qt.BrushStyle.NoBrush)


def _text(p: QPainter, text: str, size: float = 13, bold: bool = True, color="#222222"):
    f = QFont("Arial")
    f.setPixelSize(int(size))
    f.setBold(bold)
    p.setFont(f)
    p.setPen(QColor(color))
    p.drawText(QRectF(0, 0, _S, _S), Qt.AlignmentFlag.AlignCenter, text)


def _draw(p: QPainter, name: str) -> None:
    a, b = QPointF(6, 24), QPointF(26, 8)
    kind, _, arg = name.partition(":")
    if name == "select":
        p.setPen(_pen(1.3, Qt.PenStyle.DashLine))
        p.drawRect(QRectF(6, 7, 20, 18))
    elif name == "lasso":
        p.setPen(_pen(1.3, Qt.PenStyle.DashLine))
        path = QPainterPath(QPointF(8, 20))
        path.cubicTo(2, 6, 28, 2, 26, 16)
        path.cubicTo(25, 24, 12, 26, 10, 22)
        p.drawPath(path)
    elif name == "rotate":
        p.setPen(_pen(1.6))
        p.drawArc(QRectF(7, 7, 18, 18), 30 * 16, 280 * 16)
        _arrow_head(p, QPointF(24, 10), 0.3, -1)
    elif name == "eraser":
        p.setPen(_pen(1.3))
        p.setBrush(QBrush(QColor("#F48FB1")))
        p.drawPolygon(QPolygonF([QPointF(6, 20), QPointF(16, 8), QPointF(26, 16),
                                 QPointF(18, 26), QPointF(10, 26)]))
    elif kind == "bond":
        if arg == "single":
            p.setPen(_pen(1.8))
            p.drawLine(a, b)
        elif arg in ("double", "aromatic"):
            p.setPen(_pen(1.6))
            p.drawLine(QPointF(5, 21), QPointF(23, 6))
            p.setPen(_pen(1.6, Qt.PenStyle.DashLine if arg == "aromatic" else
                          Qt.PenStyle.SolidLine))
            p.drawLine(QPointF(9, 26), QPointF(27, 11))
        elif arg == "triple":
            p.setPen(_pen(1.4))
            for d in (-4, 0, 4):
                p.drawLine(QPointF(6 + d * 0.6, 24 + d * 0.75),
                           QPointF(26 + d * 0.6, 8 + d * 0.75))
        elif arg == "wedge":
            p.setBrush(QBrush(QColor("#222222")))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawPolygon(QPolygonF([QPointF(6, 24), QPointF(24, 6), QPointF(28, 11)]))
        elif arg == "hash":
            p.setPen(_pen(1.3))
            for k in range(1, 8):
                t = k / 7
                cx, cy = 6 + 20 * t, 24 - 16 * t
                w = 0.5 + 3.5 * t
                p.drawLine(QPointF(cx - w * 0.6, cy - w * 0.75),
                           QPointF(cx + w * 0.6, cy + w * 0.75))
        elif arg == "wavy":
            p.setPen(_pen(1.5))
            path = QPainterPath(a)
            for k in range(1, 41):
                t = k / 40
                x, y = 6 + 20 * t, 24 - 16 * t
                s = 2.5 * math.sin(t * 5 * math.pi)
                path.lineTo(x + s * 0.62, y + s * 0.78)
            p.drawPath(path)
        elif arg == "bold":
            p.setPen(_pen(4.5))
            p.drawLine(a, b)
        elif arg == "dashed":
            p.setPen(_pen(1.8, Qt.PenStyle.DashLine))
            p.drawLine(a, b)
    elif name == "chain":
        p.setPen(_pen(1.6))
        p.drawPolyline(QPolygonF([QPointF(3, 20), QPointF(10, 12), QPointF(17, 20),
                                  QPointF(24, 12), QPointF(30, 20)]))
    elif kind == "ring":
        sizes = {"cyclopropane": 3, "cyclobutane": 4, "cyclopentane": 5, "cyclohexane": 6,
                 "cycloheptane": 7, "cyclooctane": 8, "benzene": 6, "cyclopentadiene": 5}
        inner = {"benzene": (0, 2, 4), "cyclopentadiene": (0, 2)}.get(arg, ())
        _ring(p, sizes.get(arg, 6), inner)
    elif kind == "atom":
        _text(p, arg, 15)
    elif name == "label":
        _text(p, "A", 17)
    elif name == "periodic":
        p.setPen(_pen(1.0))
        for r in range(3):
            for c in range(5):
                if r == 0 and 0 < c < 4:
                    continue
                p.drawRect(QRectF(4 + c * 5, 9 + r * 5, 4.5, 4.5))
    elif name in ("charge+", "charge-"):
        p.setPen(_pen(1.4))
        p.drawEllipse(QRectF(7, 7, 18, 18))
        p.drawLine(QPointF(11, 16), QPointF(21, 16))
        if name == "charge+":
            p.drawLine(QPointF(16, 11), QPointF(16, 21))
    elif name == "radical":
        _text(p, "C", 14, False)
        p.setBrush(QBrush(QColor("#222222")))
        p.drawEllipse(QRectF(14.5, 3, 3, 3))
    elif name == "map":
        _text(p, "1:", 13)
    elif kind == "arrow":
        p.setPen(_pen(1.5))
        if arg == "forward":
            p.drawLine(QPointF(4, 16), QPointF(22, 16))
            _arrow_head(p, QPointF(28, 16), 1, 0)
        elif arg == "equilibrium":
            p.drawLine(QPointF(4, 13), QPointF(24, 13))
            _arrow_head(p, QPointF(28, 13), 1, 0, half=1)
            p.setPen(_pen(1.5))
            p.drawLine(QPointF(8, 19), QPointF(28, 19))
            _arrow_head(p, QPointF(4, 19), -1, 0, half=1)
        elif arg == "retro":
            p.drawLine(QPointF(4, 13.5), QPointF(23, 13.5))
            p.drawLine(QPointF(4, 18.5), QPointF(23, 18.5))
            p.drawPolyline(QPolygonF([QPointF(21, 9), QPointF(28, 16), QPointF(21, 23)]))
        elif arg == "resonance":
            p.drawLine(QPointF(9, 16), QPointF(23, 16))
            _arrow_head(p, QPointF(28, 16), 1, 0)
            _arrow_head(p, QPointF(4, 16), -1, 0)
        elif arg in ("curved", "fishhook"):
            path = QPainterPath(QPointF(5, 24))
            path.quadTo(14, 0, 24, 18)
            p.drawPath(path)
            _arrow_head(p, QPointF(26, 23), 0.35, 0.94, half=-1 if arg == "fishhook" else 0)
        else:
            p.drawLine(QPointF(4, 16), QPointF(28, 16))
    elif name == "plus":
        p.setPen(_pen(2.0))
        p.drawLine(QPointF(9, 16), QPointF(23, 16))
        p.drawLine(QPointF(16, 9), QPointF(16, 23))
    elif name == "text":
        _text(p, "T", 18)
    elif name == "bracket":
        p.setPen(_pen(1.6))
        p.drawPolyline(QPolygonF([QPointF(11, 6), QPointF(6, 6), QPointF(6, 26),
                                  QPointF(11, 26)]))
        p.drawPolyline(QPolygonF([QPointF(21, 6), QPointF(26, 6), QPointF(26, 26),
                                  QPointF(21, 26)]))
        _text(p, "n", 10, False)
    elif name == "template":
        _ring(p, 6, (0, 2, 4))
        p.setPen(_pen(2.0, color="#C62828"))
        p.drawLine(QPointF(22, 27), QPointF(30, 27))
        p.drawLine(QPointF(26, 23), QPointF(26, 31))
    else:
        _text(p, name[:2], 12)


def tool_icon(name: str) -> QIcon:
    if name not in _CACHE:
        pm = QPixmap(_S * 2, _S * 2)
        pm.setDevicePixelRatio(2.0)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        _draw(p, name)
        p.end()
        _CACHE[name] = QIcon(pm)
    return _CACHE[name]

"""Toolbar and variable-type icons drawn from small inline SVGs (no image files to ship)."""
from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

D = "#424242"
R = "#C62828"

_SVG = {
    "undo": f'<path d="M9 14L4 9l5-5" fill="none" stroke="{D}" stroke-width="1.9" '
            f'stroke-linejoin="round"/><path d="M4 9h10a6 6 0 0 1 0 12h-3" fill="none" '
            f'stroke="{R}" stroke-width="1.9"/>',
    "redo": f'<path d="M15 14l5-5-5-5" fill="none" stroke="{D}" stroke-width="1.9" '
            f'stroke-linejoin="round"/><path d="M20 9H10a6 6 0 0 0 0 12h3" fill="none" '
            f'stroke="{R}" stroke-width="1.9"/>',
    "new": f'<path d="M6 3h8l4 4v14H6z" fill="none" stroke="{D}" stroke-width="1.6"/>'
           f'<path d="M14 3v4h4" fill="none" stroke="{D}" stroke-width="1.6"/>'
           f'<path d="M12 11v6M9 14h6" stroke="{R}" stroke-width="2"/>',
    "open": f'<path d="M3 7h6l2 2h10v10H3z" fill="none" stroke="{D}" stroke-width="1.6"/>'
            f'<path d="M3 19l3-7h17l-3 7" fill="{R}" fill-opacity="0.18" stroke="{R}" '
            f'stroke-width="1.6"/>',
    "save": f'<path d="M4 4h13l3 3v13H4z" fill="none" stroke="{D}" stroke-width="1.6"/>'
            f'<rect x="7" y="4" width="8" height="5" fill="none" stroke="{D}" stroke-width="1.4"/>'
            f'<rect x="7" y="13" width="10" height="7" fill="{R}" fill-opacity="0.85"/>',
    "import": f'<path d="M4 15v5h16v-5" fill="none" stroke="{D}" stroke-width="1.6"/>'
              f'<path d="M12 3v11M7.5 9.5L12 14l4.5-4.5" fill="none" stroke="{R}" stroke-width="2"/>',
    "export": f'<path d="M4 15v5h16v-5" fill="none" stroke="{D}" stroke-width="1.6"/>'
              f'<path d="M12 14V3M7.5 7.5L12 3l4.5 4.5" fill="none" stroke="{R}" stroke-width="2"/>',
    "output": f'<rect x="3" y="4" width="18" height="16" rx="1.5" fill="none" stroke="{D}" '
              f'stroke-width="1.6"/><path d="M3 8h18" stroke="{D}" stroke-width="1.4"/>'
              f'<rect x="6" y="15" width="2.5" height="3" fill="{R}"/>'
              f'<rect x="10.5" y="12" width="2.5" height="6" fill="{R}"/>'
              f'<rect x="15" y="10" width="2.5" height="8" fill="{R}"/>',
    "goto": f'<circle cx="10" cy="10" r="6" fill="none" stroke="{D}" stroke-width="1.8"/>'
            f'<path d="M14.5 14.5L20 20" stroke="{R}" stroke-width="2.4"/>',
    "insert_var": f'<rect x="3" y="4" width="5" height="16" fill="none" stroke="{D}" '
                  f'stroke-width="1.4"/><rect x="16" y="4" width="5" height="16" fill="none" '
                  f'stroke="{D}" stroke-width="1.4"/><rect x="9.5" y="4" width="5" height="16" '
                  f'fill="{R}" fill-opacity="0.25" stroke="{R}" stroke-width="1.4"/>'
                  f'<path d="M12 9v6M9.5 12h5" stroke="{R}" stroke-width="1.8"/>',
    "insert_case": f'<rect x="4" y="3" width="16" height="5" fill="none" stroke="{D}" '
                   f'stroke-width="1.4"/><rect x="4" y="16" width="16" height="5" fill="none" '
                   f'stroke="{D}" stroke-width="1.4"/><rect x="4" y="9.5" width="16" height="5" '
                   f'fill="{R}" fill-opacity="0.25" stroke="{R}" stroke-width="1.4"/>'
                   f'<path d="M12 10.5v3M10.5 12h3" stroke="{R}" stroke-width="1.8"/>',
    "labels": f'<path d="M3 12l8-8h9v9l-8 8z" fill="none" stroke="{D}" stroke-width="1.6"/>'
              f'<circle cx="16" cy="8" r="1.8" fill="{R}"/>'
              f'<path d="M8 13l3 3" stroke="{R}" stroke-width="1.8"/>',
    "sort": f'<path d="M7 4v16M3.5 16.5L7 20l3.5-3.5" fill="none" stroke="{D}" stroke-width="1.8"/>'
            f'<path d="M14 6h7M14 11h5M14 16h3" stroke="{R}" stroke-width="2"/>',
    "descriptives": f'<path d="M17 5H7l6 7-6 7h10" fill="none" stroke="{R}" stroke-width="2.2" '
                    f'stroke-linejoin="round"/>',
    "frequencies": f'<path d="M3 20h18" stroke="{D}" stroke-width="1.6"/>'
                   f'<rect x="5" y="12" width="3" height="8" fill="{R}"/>'
                   f'<rect x="10.5" y="6" width="3" height="14" fill="{R}"/>'
                   f'<rect x="16" y="9" width="3" height="11" fill="{R}"/>',
    "chart": f'<path d="M3 3v18h18" fill="none" stroke="{D}" stroke-width="1.6"/>'
             f'<path d="M6 16l4-5 3 3 6-8" fill="none" stroke="{R}" stroke-width="2"/>',
    "molecule": f'<path d="M12 3l7.8 4.5v9L12 21l-7.8-4.5v-9z" fill="none" stroke="{R}" '
                f'stroke-width="1.8"/><circle cx="12" cy="12" r="3.5" fill="none" stroke="{D}" '
                f'stroke-width="1.4"/>',
    "data_editor": f'<rect x="3" y="4" width="18" height="16" fill="none" stroke="{D}" '
                   f'stroke-width="1.6"/><path d="M3 9h18M3 14h18M9 4v16M15 4v16" stroke="{D}" '
                   f'stroke-width="1"/><rect x="9" y="9" width="6" height="5" fill="{R}"/>',
    # variable measurement levels
    "scale": '<rect x="2" y="8" width="20" height="8" rx="1" fill="#FFE082" stroke="#B28704"/>'
             '<path d="M6 8v4M10 8v3M14 8v4M18 8v3" stroke="#6D5300" stroke-width="1.2"/>',
    "ordinal": f'<rect x="3" y="14" width="4" height="6" fill="{R}"/>'
               f'<rect x="10" y="9" width="4" height="11" fill="{R}"/>'
               f'<rect x="17" y="4" width="4" height="16" fill="{R}"/>',
    "nominal": '<circle cx="7" cy="8" r="3.4" fill="#1E88E5"/>'
               '<circle cx="17" cy="8" r="3.4" fill="#E53935"/>'
               '<circle cx="12" cy="16.5" r="3.4" fill="#43A047"/>',
    "string": f'<text x="3" y="17" font-family="Arial" font-size="13" font-weight="bold" '
              f'fill="{D}">a</text><text x="11" y="17" font-family="Arial" font-size="13" '
              f'font-weight="bold" fill="{R}">b</text>',
    "date": f'<rect x="3" y="5" width="18" height="15" rx="1.5" fill="none" stroke="{D}" '
            f'stroke-width="1.5"/><path d="M3 9h18" stroke="{R}" stroke-width="2.4"/>'
            f'<path d="M8 3v4M16 3v4" stroke="{D}" stroke-width="1.6"/>',
}


@lru_cache(maxsize=None)
def icon(name: str) -> QIcon:
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">{_SVG[name]}</svg>'
    renderer = QSvgRenderer(QByteArray(svg.encode()))
    result = QIcon()
    for size in (16, 20, 24, 32, 48):
        pm = QPixmap(size, size)
        pm.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        renderer.render(painter, QRectF(0, 0, size, size))
        painter.end()
        result.addPixmap(pm)
    return result


def variable_icon(var) -> QIcon:
    if var.type == "string":
        return icon("string") if var.measure == "scale" else icon(var.measure)
    if var.type == "date":
        return icon("date")
    return icon(var.measure)

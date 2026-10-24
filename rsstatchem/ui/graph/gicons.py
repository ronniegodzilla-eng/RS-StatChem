"""Small inline-SVG icons for the Graph Studio toolbar (no image files to ship)."""
from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

D = "#424242"
R = "#C62828"
_S = f'fill="none" stroke="{D}" stroke-width="1.7" stroke-linecap="round" ' \
     f'stroke-linejoin="round"'

_SVG = {
    "pointer": f'<path d="M6 3l12 9-5.5 1 3 6.5-2.3 1-3-6.5L6 18z" fill="{D}"/>',
    "zoom": f'<circle cx="10" cy="10" r="6" {_S}/><path d="M14.5 14.5L20 20M7.5 10h5M10 7.5v5"'
            f' {_S}/>',
    "pan": f'<path d="M12 3v18M3 12h18M12 3l-3 3M12 3l3 3M12 21l-3-3M12 21l3-3M3 12l3-3M3 12'
           f'l3 3M21 12l-3-3M21 12l-3 3" {_S}/>',
    "rescale": f'<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" {_S}/>'
               f'<path d="M8 15l3-4 2 2 3-4" fill="none" stroke="{R}" stroke-width="1.7"/>',
    "text": f'<path d="M5 5h14M12 5v15M9 20h6" {_S}/>',
    "arrow": f'<path d="M4 20L19 5M19 5h-7M19 5v7" {_S}/>',
    "line": f'<path d="M4 20L20 4" {_S}/>',
    "rect": f'<rect x="4" y="6" width="16" height="12" {_S}/>',
    "export": f'<path d="M4 15v5h16v-5M12 4v11M8 8l4-4 4 4" {_S}/>',
    "copy": f'<rect x="8" y="8" width="12" height="12" {_S}/><path d="M16 8V4H4v12h4" {_S}/>',
    "output": f'<rect x="3" y="4" width="18" height="16" rx="1.5" {_S}/><path d="M7 15l3-4 3 3'
              f' 4-5" fill="none" stroke="{R}" stroke-width="1.8"/>',
    "save": f'<path d="M4 4h13l3 3v13H4z" {_S}/><path d="M8 4v5h8V4M8 20v-6h8v6" {_S}/>',
    "open": f'<path d="M3 7h6l2 2h10v10H3z" {_S}/>',
    "undo": f'<path d="M9 14L4 9l5-5" {_S}/><path d="M4 9h10a6 6 0 010 12h-3" {_S}/>',
    "redo": f'<path d="M15 14l5-5-5-5" {_S}/><path d="M20 9H10a6 6 0 000 12h3" {_S}/>',
    "layer": f'<path d="M12 4l9 5-9 5-9-5z" {_S}/><path d="M3 14l9 5 9-5" {_S}/>',
    "plot": f'<path d="M3 3v18h18" {_S}/><path d="M6 16l4-6 4 3 5-8" fill="none" '
            f'stroke="{R}" stroke-width="1.8"/>',
    "add_plot": f'<path d="M3 3v18h18" {_S}/><path d="M14 7h7M17.5 3.5v7" fill="none" '
                f'stroke="{R}" stroke-width="2"/>',
    "graph": f'<rect x="3" y="3" width="18" height="18" rx="1.5" {_S}/><path d="M7 16l3-5 3 2'
             f' 4-6" fill="none" stroke="{R}" stroke-width="1.8"/>',
    "axis": f'<path d="M5 3v16h16" {_S}/><path d="M5 7h2M5 11h2M5 15h2M9 19v-2M13 19v-2M17 '
            f'19v-2" {_S}/>',
    "legend": f'<rect x="3" y="5" width="18" height="14" {_S}/><path d="M6 9h4M6 15h4" '
              f'stroke="{R}" stroke-width="2"/><path d="M12 9h6M12 15h6" {_S}/>',
    "annotation": f'<path d="M4 4h16v12H9l-5 4z" {_S}/>',
    "folder": f'<path d="M3 6h7l2 2h9v11H3z" {_S}/>',
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

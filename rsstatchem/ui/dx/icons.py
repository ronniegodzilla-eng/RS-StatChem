"""Icons of the RSM module, drawn as small inline SVGs (own artwork, Design-Expert-like roles).

Icons are not cached at module level: a QIcon that outlives the QApplication crashes on exit.
"""
from __future__ import annotations

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

B = "#1F4E8C"   # blue
L = "#5B9BE0"   # light blue
G = "#2E9E4F"   # green
O = "#E8822A"   # orange
R = "#D0342C"   # red
Y = "#F2C230"   # yellow
D = "#3B4654"   # dark grey

_SVG = {
    "new": f'<path d="M5 2h9l5 5v15H5z" fill="#fff" stroke="{D}" stroke-width="1.4"/>'
           f'<path d="M14 2v5h5" fill="none" stroke="{D}" stroke-width="1.4"/>'
           f'<circle cx="17" cy="17" r="5" fill="{G}"/><path d="M17 14.5v5M14.5 17h5" '
           f'stroke="#fff" stroke-width="1.8"/>',
    "open": f'<path d="M2 6h7l2 2h10v12H2z" fill="{Y}" stroke="{D}" stroke-width="1.3"/>'
            f'<path d="M2 20l3-8h18l-3 8z" fill="#FFE07A" stroke="{D}" stroke-width="1.3"/>',
    "save": f'<path d="M3 3h15l3 3v15H3z" fill="{B}" stroke="{B}" stroke-width="1"/>'
            f'<rect x="6" y="3" width="10" height="6" fill="#fff"/>'
            f'<rect x="12.5" y="4" width="2" height="4" fill="{B}"/>'
            f'<rect x="6" y="13" width="12" height="8" fill="#E8EEF6"/>',
    "copy": f'<rect x="3" y="3" width="12" height="14" fill="#fff" stroke="{D}" '
            f'stroke-width="1.3"/><rect x="9" y="7" width="12" height="14" fill="#E3EEFB" '
            f'stroke="{B}" stroke-width="1.3"/>',
    "paste": f'<rect x="4" y="4" width="14" height="17" rx="1" fill="#E9C98A" stroke="{D}" '
             f'stroke-width="1.2"/><rect x="8" y="2" width="6" height="4" fill="#fff" '
             f'stroke="{D}" stroke-width="1.2"/><rect x="10" y="10" width="11" height="12" '
             f'fill="#fff" stroke="{B}" stroke-width="1.2"/>',
    "export": f'<rect x="3" y="5" width="16" height="14" fill="#fff" stroke="{D}" '
              f'stroke-width="1.3"/><path d="M5 16l4-5 3 3 3-4 3 6z" fill="{G}"/>'
              f'<circle cx="8" cy="9" r="1.6" fill="{Y}"/><path d="M15 3h6v6M21 3l-6 6" '
              f'stroke="{R}" stroke-width="1.8" fill="none"/>',
    "undo": f'<path d="M9 14L4 9l5-5" fill="none" stroke="{B}" stroke-width="2" '
            f'stroke-linejoin="round"/><path d="M4 9h10a6 6 0 0 1 0 12h-3" fill="none" '
            f'stroke="{B}" stroke-width="2"/>',
    "redo": f'<path d="M15 14l5-5-5-5" fill="none" stroke="{B}" stroke-width="2" '
            f'stroke-linejoin="round"/><path d="M20 9H10a6 6 0 0 0 0 12h3" fill="none" '
            f'stroke="{B}" stroke-width="2"/>',
    "tips": f'<path d="M12 2a7 7 0 0 0-4 12.7V18h8v-3.3A7 7 0 0 0 12 2z" fill="{Y}" '
            f'stroke="{D}" stroke-width="1.2"/><rect x="8.5" y="19" width="7" height="3" '
            f'rx="1" fill="{D}"/><path d="M10 11l2 2 2-2" fill="none" stroke="{O}" '
            f'stroke-width="1.3"/>',
    "coded": f'<rect x="2" y="5" width="20" height="14" rx="2" fill="#fff" stroke="{D}" '
             f'stroke-width="1.3"/><text x="12" y="16" font-size="9" font-family="Arial" '
             f'font-weight="bold" text-anchor="middle" fill="{B}">±1</text>',
    # navigation nodes
    "design": f'<rect x="2" y="3" width="20" height="18" fill="#fff" stroke="{D}" '
              f'stroke-width="1.2"/><path d="M2 8h20M2 13h20M2 18h20M8 3v18M15 3v18" '
              f'stroke="{L}" stroke-width="1"/><rect x="2" y="3" width="20" height="5" '
              f'fill="{B}"/>',
    "summary": f'<rect x="4" y="2" width="16" height="20" fill="#fff" stroke="{D}" '
               f'stroke-width="1.2"/><path d="M7 7h10M7 11h10M7 15h7" stroke="{B}" '
               f'stroke-width="1.6"/>',
    "graphcols": f'<path d="M3 3v18h18" fill="none" stroke="{D}" stroke-width="1.4"/>'
                 + "".join(f'<circle cx="{x}" cy="{y}" r="1.8" fill="{c}"/>'
                           for x, y, c in ((7, 17, B), (10, 13, G), (13, 14, O), (15, 9, R),
                                           (18, 6, B))),
    "evaluation": f'<circle cx="10" cy="10" r="6.5" fill="#E3EEFB" stroke="{B}" '
                  f'stroke-width="1.8"/><path d="M15 15l6 6" stroke="{D}" stroke-width="2.6"/>'
                  f'<path d="M7 10l2 2 4-4" fill="none" stroke="{G}" stroke-width="1.8"/>',
    "notes": f'<path d="M4 3h16v13l-5 5H4z" fill="#FFF3B0" stroke="{D}" stroke-width="1.2"/>'
             f'<path d="M15 21v-5h5" fill="#F2DE7A" stroke="{D}" stroke-width="1.2"/>'
             f'<path d="M7 8h10M7 12h8" stroke="{O}" stroke-width="1.4"/>',
    "folder": f'<path d="M2 6h7l2 2h11v12H2z" fill="#D9E6F5" stroke="{B}" stroke-width="1.2"/>',
    "response": f'<circle cx="12" cy="12" r="9.5" fill="#E3EEFB" stroke="{B}" stroke-width="1.4"/>'
                f'<path d="M5 16c3-8 5-10 7-10s4 4 7 10" fill="none" stroke="{R}" '
                f'stroke-width="1.8"/>',
    "response_empty": f'<circle cx="12" cy="12" r="9.5" fill="#fff" stroke="#9AA6B4" '
                      f'stroke-width="1.4" stroke-dasharray="2 2"/>'
                      f'<path d="M5 16c3-8 5-10 7-10s4 4 7 10" fill="none" stroke="#9AA6B4" '
                      f'stroke-width="1.6"/>',
    "numerical": f'<circle cx="12" cy="12" r="9.5" fill="#fff" stroke="{R}" stroke-width="1.6"/>'
                 f'<circle cx="12" cy="12" r="5.5" fill="#fff" stroke="{R}" stroke-width="1.6"/>'
                 f'<circle cx="12" cy="12" r="2" fill="{R}"/>',
    "graphical": f'<rect x="2" y="2" width="20" height="20" fill="#C8C8C8"/>'
                 f'<path d="M2 22C6 10 12 6 22 5v17z" fill="{Y}"/>'
                 f'<path d="M2 22C6 10 12 6 22 5" fill="none" stroke="{B}" stroke-width="1.6"/>'
                 f'<path d="M8 22c2-6 7-10 14-11" fill="none" stroke="{R}" stroke-width="1.6"/>',
    "prediction": f'<circle cx="12" cy="12" r="8" fill="none" stroke="{B}" stroke-width="1.6"/>'
                  f'<path d="M12 1v6M12 17v6M1 12h6M17 12h6" stroke="{B}" stroke-width="1.6"/>'
                  f'<circle cx="12" cy="12" r="2.4" fill="{O}"/>',
    "confirmation": f'<rect x="3" y="3" width="18" height="18" rx="3" fill="#E7F6EC" '
                    f'stroke="{G}" stroke-width="1.6"/><path d="M7 12l4 4 6-8" fill="none" '
                    f'stroke="{G}" stroke-width="2.2"/>',
    "coefficients": f'<rect x="2" y="4" width="20" height="16" fill="#fff" stroke="{D}" '
                    f'stroke-width="1.2"/><path d="M2 9h20M9 4v16" stroke="{D}" '
                    f'stroke-width="1"/><rect x="11" y="11" width="4" height="3" fill="{R}"/>'
                    f'<rect x="16" y="15" width="4" height="3" fill="{G}"/>'
                    f'<rect x="11" y="15" width="4" height="3" fill="{L}"/>',
    # analysis tabs
    "transform": f'<path d="M4 20c4 0 5-16 9-16" fill="none" stroke="{B}" stroke-width="1.8"/>'
                 f'<text x="15" y="20" font-size="11" font-family="Arial" font-weight="bold" '
                 f'fill="{O}">λ</text>',
    "fitsummary": f'<path d="M3 21h18" stroke="{D}" stroke-width="1.4"/>'
                  f'<rect x="4" y="13" width="3.5" height="8" fill="{L}"/>'
                  f'<rect x="9" y="8" width="3.5" height="13" fill="{B}"/>'
                  f'<rect x="14" y="4" width="3.5" height="17" fill="{G}"/>'
                  f'<path d="M14 2l1.75 1.5L17.5 2" fill="none" stroke="{O}" stroke-width="1.4"/>',
    "model": f'<text x="1" y="17" font-size="12" font-style="italic" font-family="Times New '
             f'Roman, serif" font-weight="bold" fill="{B}">f(x)</text>',
    "anova": f'<rect x="2" y="3" width="20" height="18" fill="#fff" stroke="{D}" '
             f'stroke-width="1.2"/><path d="M2 8h20M2 13h20M2 18h20M10 3v18" stroke="{L}" '
             f'stroke-width="1"/><text x="3.5" y="7.2" font-size="5" font-family="Arial" '
             f'font-weight="bold" fill="{B}">SS</text><text x="12" y="17.3" font-size="5" '
             f'font-family="Arial" font-weight="bold" fill="{R}">F p</text>',
    "diagnostics": f'<path d="M3 3v18h18" fill="none" stroke="{D}" stroke-width="1.3"/>'
                   f'<path d="M4 20L21 4" stroke="{R}" stroke-width="1.4"/>'
                   + "".join(f'<circle cx="{x}" cy="{y}" r="1.6" fill="{B}"/>'
                             for x, y in ((6, 17), (9, 15), (11, 12), (14, 10), (17, 7))),
    "modelgraphs": f'<path d="M2 17l8-4 12 3-7 5z" fill="#D9E6F5" stroke="{D}" '
                   f'stroke-width="1"/><path d="M2 17C5 6 9 3 11 5s5 8 11 11" fill="none" '
                   f'stroke="{R}" stroke-width="1.8"/><path d="M5 15c2-5 4-7 6-6s3 5 8 7" '
                   f'fill="none" stroke="{G}" stroke-width="1.3"/>',
    # graph types
    "contour": f'<rect x="2" y="2" width="20" height="20" fill="#1f3fbf"/>'
               f'<ellipse cx="13" cy="11" rx="9" ry="7.5" fill="#22a7d6"/>'
               f'<ellipse cx="13" cy="11" rx="6.5" ry="5" fill="#2fbf5f"/>'
               f'<ellipse cx="13" cy="11" rx="4" ry="3" fill="#e8e23a"/>'
               f'<ellipse cx="13" cy="11" rx="1.8" ry="1.3" fill="#e2442b"/>',
    "surface": f'<path d="M2 16l9-5 11 3-9 6z" fill="#22a7d6" stroke="{D}" stroke-width="0.8"/>'
               f'<path d="M4 15c3-9 6-12 8-11s4 6 8 10l-7 4z" fill="#2fbf5f" stroke="{D}" '
               f'stroke-width="0.8"/><path d="M9 12c1-4 2-6 3-6s2 3 4 6z" fill="#e8e23a"/>'
               f'<path d="M11 9c.5-1.5 1-2.5 1.5-2.5s1 1 1.6 2.5z" fill="#e2442b"/>',
    "perturbation": f'<path d="M3 3v18h18" fill="none" stroke="{D}" stroke-width="1.2"/>'
                    f'<path d="M4 18C9 6 15 6 20 12" fill="none" stroke="{B}" stroke-width="1.7"/>'
                    f'<path d="M4 9C10 13 14 15 20 16" fill="none" stroke="{R}" '
                    f'stroke-width="1.7"/><path d="M12 3v18" stroke="#999" '
                    f'stroke-width="0.8" stroke-dasharray="2 2"/>',
    "onefactor": f'<path d="M3 3v18h18" fill="none" stroke="{D}" stroke-width="1.2"/>'
                 f'<path d="M4 17C9 7 14 6 20 9" fill="none" stroke="{B}" stroke-width="1.8"/>'
                 f'<path d="M4 14C9 4 14 3 20 6M4 20C9 10 14 9 20 12" fill="none" '
                 f'stroke="{B}" stroke-width="0.9" stroke-dasharray="2 1.5"/>',
    "interaction": f'<path d="M3 3v18h18" fill="none" stroke="{D}" stroke-width="1.2"/>'
                   f'<path d="M4 17L20 6" stroke="{R}" stroke-width="1.8"/>'
                   f'<path d="M4 8L20 15" stroke="{G}" stroke-width="1.8"/>',
    "cube": f'<path d="M4 8l8-4 8 4v9l-8 4-8-4z" fill="#E3EEFB" stroke="{B}" '
            f'stroke-width="1.4"/><path d="M4 8l8 4 8-4M12 12v9" fill="none" stroke="{B}" '
            f'stroke-width="1.2"/>' + "".join(f'<circle cx="{x}" cy="{y}" r="1.5" fill="{R}"/>'
                                              for x, y in ((4, 8), (12, 4), (20, 8), (4, 17),
                                                           (20, 17), (12, 21), (12, 12))),
    "ternary": f'<path d="M12 3L22 20H2z" fill="#22a7d6" stroke="{D}" stroke-width="1.2"/>'
               f'<path d="M12 8l6 10H6z" fill="#2fbf5f"/><path d="M12 12l3 5H9z" '
               f'fill="#e8e23a"/>',
    "trace": f'<path d="M3 3v18h18" fill="none" stroke="{D}" stroke-width="1.2"/>'
             f'<path d="M4 16C8 8 14 7 20 9" fill="none" stroke="{B}" stroke-width="1.6"/>'
             f'<path d="M4 7C9 11 14 14 20 18" fill="none" stroke="{G}" stroke-width="1.6"/>'
             f'<path d="M4 12C9 12 14 10 20 13" fill="none" stroke="{R}" stroke-width="1.6"/>',
    "ramps": f'<path d="M2 18h6l6-10h8" fill="none" stroke="{G}" stroke-width="1.8"/>'
             f'<circle cx="11" cy="13" r="2.4" fill="{R}"/>',
    "bars": f'<rect x="3" y="4" width="14" height="4" fill="{G}"/><rect x="3" y="10" '
            f'width="18" height="4" fill="{G}"/><rect x="3" y="16" width="9" height="4" '
            f'fill="{B}"/>',
    "report": f'<rect x="4" y="2" width="16" height="20" fill="#fff" stroke="{D}" '
              f'stroke-width="1.2"/><path d="M7 6h10M7 10h10M7 14h10M7 18h6" stroke="{D}" '
              f'stroke-width="1.1"/>',
    "start": f'<rect x="2" y="2" width="20" height="20" rx="3" fill="{B}"/>'
             f'<path d="M9 7l8 5-8 5z" fill="#fff"/>',
}


def svg(name: str) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">{_SVG[name]}</svg>'


def pixmap(name: str, size: int = 24) -> QPixmap:
    renderer = QSvgRenderer(QByteArray(svg(name).encode()))
    pix = QPixmap(size * 2, size * 2)
    pix.setDevicePixelRatio(2.0)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    renderer.render(painter, QRectF(0, 0, size, size))
    painter.end()
    return pix


def icon(name: str) -> QIcon:
    ic = QIcon()
    for size in (16, 24, 32):
        ic.addPixmap(pixmap(name, size))
    return ic


NAMES = tuple(_SVG)

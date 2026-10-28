"""Inline-SVG icons of the Chemical Drawing workspace (drawn for RS-StatChem, 24 x 24 grid):
the Main Toolbar tools in the arrangement of ChemDraw's palette, the General toolbar and the
Style toolbar.

Icons are cached in ``rsstatchem.ui.chem_editor.icons._CACHE`` (cleared by
:mod:`rsstatchem.ui.qt_lifecycle` before the QApplication goes away).
"""
from __future__ import annotations

import math

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

__all__ = ["cd_icon", "color_icon", "svg_markup"]

K = "#1A1A1A"      # ink
B = "#2F6FD6"      # accent (selection tools, info)
G = "#6B6B6B"


def _line(x1, y1, x2, y2, w=1.6, color=K, dash="") -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" '
            f'stroke-width="{w}" stroke-linecap="round"{d}/>')


def _poly(pts, fill=K, stroke="none", w=1.0) -> str:
    p = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
    return f'<polygon points="{p}" fill="{fill}" stroke="{stroke}" stroke-width="{w}"/>'


def _text(x, y, s, size=13, weight="bold", color=K, family="Arial", style="normal") -> str:
    return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" font-style="{style}" fill="{color}" '
            f'text-anchor="middle">{s}</text>')


def _ring(n: int, inner=(), r=8.6, cx=12.0, cy=12.4) -> str:
    start = -math.pi / 2 + (math.pi / n if n % 2 == 0 and n % 4 == 0 else 0)
    if n == 4:
        start = -math.pi / 4
    pts = [(cx + r * math.cos(start + 2 * math.pi * k / n),
            cy + r * math.sin(start + 2 * math.pi * k / n)) for k in range(n)]
    out = [_poly(pts, fill="none", stroke=K, w=1.4)]
    for k in inner:
        (ax, ay), (bx, by) = pts[k], pts[(k + 1) % n]
        f = 0.27
        ax2, ay2 = ax + (cx - ax) * f, ay + (cy - ay) * f
        bx2, by2 = bx + (cx - bx) * f, by + (cy - by) * f
        mx, my = (ax2 + bx2) / 2, (ay2 + by2) / 2
        out.append(_line(mx + (ax2 - mx) * 0.78, my + (ay2 - my) * 0.78,
                         mx + (bx2 - mx) * 0.78, my + (by2 - my) * 0.78, 1.4))
    return "".join(out)


def _chair(mirror: bool) -> str:
    from rsstatchem.core.chem_draw.ops import CHAIR_TEMPLATES
    pts = CHAIR_TEMPLATES["chair2" if mirror else "chair1"]
    pts = [(12 + x * 8.6, 12 + y * 13) for x, y in pts]
    return _poly(pts, fill="none", stroke=K, w=1.4)


def _arrow_head(x, y, ux, uy, half=0, hl=5.2, hw=2.6) -> str:
    nx, ny = -uy, ux
    bx, by = x - ux * hl, y - uy * hl
    left = (bx + nx * hw, by + ny * hw)
    right = (bx - nx * hw, by - ny * hw)
    notch = (x - ux * hl * 0.72, y - uy * hl * 0.72)
    pts = [(x, y), left, notch] if half > 0 else (
        [(x, y), notch, right] if half < 0 else [(x, y), left, notch, right])
    return _poly(pts)


def _bond_icon(kind: str) -> str:
    a, b = (4.5, 19.5), (19.5, 4.5)
    if kind == "single":
        return _line(*a, *b, 1.8)
    if kind in ("double", "triple", "aromatic"):
        off = 2.6 if kind != "triple" else 3.3
        s = [_line(a[0] - off * 0.7, a[1] - off * 0.7, b[0] - off * 0.7, b[1] - off * 0.7, 1.5),
             _line(a[0] + off * 0.7, a[1] + off * 0.7, b[0] + off * 0.7, b[1] + off * 0.7, 1.5,
                   dash="2.2 1.8" if kind == "aromatic" else "")]
        if kind == "triple":
            s.append(_line(*a, *b, 1.5))
        return "".join(s)
    if kind == "quadruple":
        return "".join(_line(a[0] + d * 0.7, a[1] + d * 0.7, b[0] + d * 0.7, b[1] + d * 0.7, 1.2)
                       for d in (-4.2, -1.4, 1.4, 4.2))
    if kind == "bolddouble":
        return (_line(a[0] - 1.6, a[1] - 1.6, b[0] - 1.6, b[1] - 1.6, 3.6) +
                _line(a[0] + 2.4, a[1] + 2.4, b[0] + 2.4, b[1] + 2.4, 1.4))
    if kind == "dashed":
        return _line(*a, *b, 1.8, dash="2.4 2.2")
    if kind == "bold":
        return _line(*a, *b, 4.2)
    if kind == "wedge":
        return _poly([(4.5, 19.5), (20.8, 5.8), (18.2, 3.2)])
    if kind == "hash":
        out = []
        for k in range(1, 8):
            t = k / 7.5
            cx, cy = 4.5 + 15 * t, 19.5 - 15 * t
            hw = 0.4 + 2.0 * t
            out.append(_line(cx - hw * 0.707, cy - hw * 0.707, cx + hw * 0.707,
                             cy + hw * 0.707, 1.1))
        return "".join(out)
    if kind == "wavy":
        return ('<path d="M4 20 q2.6 -0.4 2.2 -3 t2.2 -3 t2.2 -3 t2.2 -3 t2.2 -3 t2.6 -3" '
                f'fill="none" stroke="{K}" stroke-width="1.5" stroke-linecap="round"/>')
    if kind == "hollow":
        return _poly([(4.5, 19.5), (20.8, 5.8), (18.2, 3.2)], fill="#FFFFFF", stroke=K, w=1.2)
    if kind == "hashed":
        out = []
        for k in range(1, 8):
            t = k / 7.5
            cx, cy = 4.5 + 15 * t, 19.5 - 15 * t
            out.append(_line(cx - 1.6, cy - 1.6, cx + 1.6, cy + 1.6, 1.1))
        return "".join(out)
    if kind == "dative":
        return _line(4.5, 19.5, 15.5, 8.5, 1.6) + _arrow_head(19.5, 4.5, 0.707, -0.707,
                                                             hl=6, hw=3)
    return ""


def _lobe_path(cx, cy, ang, L, fat=1.0) -> list[tuple[float, float]]:
    pts = []
    for k in range(25):
        phi = -math.pi / 2 + math.pi * k / 24
        r = L * math.cos(phi) ** 2
        u, v = r * math.cos(phi), r * math.sin(phi) * fat
        pts.append((cx + u * math.cos(ang) - v * math.sin(ang),
                    cy + u * math.sin(ang) + v * math.cos(ang)))
    return pts


def _orbital_icon(kind: str, fill: str) -> str:
    up = -math.pi / 2
    if kind == "s":
        lobes = [None]
    elif kind == "lobe":
        lobes = [(up, 10.5, 1.0)]
    elif kind == "hybrid":
        lobes = [(math.pi / 2, 4.2, 1.0), (up, 10.5, 1.15)]
    elif kind == "dxy":
        lobes = [(math.pi / 4 + k * math.pi / 2, 9.5, 0.95) for k in range(4)]
    elif kind == "dz2":
        lobes = ["ring", (up, 9.5, 1.0), (math.pi / 2, 9.5, 1.0)]
    else:
        lobes = [(up, 10.5, 1.0), (math.pi / 2, 10.5, 1.0)]
    out = []
    for k, lobe in enumerate(lobes):
        f = {"plain": "#FFFFFF", "shaded": "#A8A8A8", "filled": K}.get(fill, "#A8A8A8")
        if fill == "phase":
            f = K if (k % 2 == 0) != (kind in ("dz2", "hybrid")) else "#FFFFFF"
        if lobe is None:
            out.append(f'<circle cx="12" cy="12" r="7.5" fill="{f}" stroke="{K}" '
                       'stroke-width="1"/>')
        elif lobe == "ring":
            out.append(f'<ellipse cx="12" cy="12" rx="5.5" ry="2.2" fill="{f}" stroke="{K}" '
                       'stroke-width="1"/>')
        else:
            out.append(_poly(_lobe_path(12, 12, *lobe), fill=f, stroke=K, w=1.0))
    return "".join(out)


def _arrow_icon(kind: str) -> str:
    y = 12
    if kind == "forward":
        return _line(3, y, 18, y, 1.5) + _arrow_head(21.5, y, 1, 0)
    if kind == "equilibrium":
        return (_line(3, y - 2.2, 18, y - 2.2, 1.3) + _arrow_head(21.5, y - 2.2, 1, 0, half=-1) +
                _line(6, y + 2.2, 21, y + 2.2, 1.3) + _arrow_head(2.5, y + 2.2, -1, 0, half=-1))
    if kind == "retro":
        return (_line(3, y - 2.4, 17, y - 2.4, 1.3) + _line(3, y + 2.4, 17, y + 2.4, 1.3) +
                f'<polyline points="14.5,5.5 21,12 14.5,18.5" fill="none" stroke="{K}" '
                f'stroke-width="1.3" stroke-linejoin="miter"/>')
    if kind == "resonance":
        return _line(6, y, 18, y, 1.5) + _arrow_head(21.5, y, 1, 0) + _arrow_head(2.5, y, -1, 0)
    if kind in ("curved", "fishhook"):
        half = 1 if kind == "fishhook" else 0
        return (f'<path d="M3.5 17 Q10 2 19.5 13" fill="none" stroke="{K}" stroke-width="1.5"/>'
                + _arrow_head(20.5, 15.2, 0.45, 0.89, half=half))
    if kind == "line":
        return _line(3, y, 21, y, 1.5)
    if kind == "nogo":
        return (_line(3, y, 18, y, 1.5) + _arrow_head(21.5, y, 1, 0) +
                _line(9, 8, 15, 16, 1.4) + _line(9, 16, 15, 8, 1.4))
    if kind == "unbalanced":
        return (_line(3, y - 2.2, 18, y - 2.2, 1.3) + _arrow_head(21.5, y - 2.2, 1, 0, half=-1) +
                _line(10, y + 2.2, 15, y + 2.2, 1.3) + _arrow_head(7, y + 2.2, -1, 0, half=-1))
    if kind == "block":
        return _poly([(3, 9.5), (14, 9.5), (14, 6), (21.5, 12), (14, 18), (14, 14.5),
                      (3, 14.5)], fill="#FFFFFF", stroke=K, w=1.2)
    if kind == "double":
        return (f'<path d="M5 16 Q12 4 19 16" fill="none" stroke="{K}" stroke-width="1.5"/>'
                + _arrow_head(20.2, 17.6, 0.5, 0.86) + _arrow_head(3.8, 17.6, -0.5, 0.86))
    if kind.startswith("arc"):
        sweep = kind.partition(":")[2] or "180"
        d = {"90": "M5 17 A10 10 0 0 1 17 5", "180": "M3.5 15 A8.5 8.5 0 0 1 20.5 15",
             "270": "M8 19.5 A7.5 7.5 0 1 1 19.5 13"}.get(sweep, "M3.5 15 A8.5 8.5 0 0 1 20.5 15")
        tip = {"90": (18.5, 5.2, 1, 0.1), "180": (20.5, 17, 0, 1),
               "270": (19.4, 15.5, -0.2, 1)}.get(sweep, (20.5, 17, 0, 1))
        return (f'<path d="{d}" fill="none" stroke="{K}" stroke-width="1.5"/>' +
                _arrow_head(*tip))
    if kind == "dipole":
        return (_line(3, y, 18, y, 1.5) + _arrow_head(21.5, y, 1, 0) +
                _line(6.5, y - 3.5, 6.5, y + 3.5, 1.5))
    return ""


def _symbol_icon(kind: str) -> str:
    dot = '<circle cx="{0}" cy="{1}" r="1.7" fill="#111"/>'
    if kind == "lonepair":
        return _text(12, 20, "O", 12, "normal") + dot.format(9.5, 5) + dot.format(14.5, 5)
    if kind == "electron":
        return dot.format(12, 12)
    if kind in ("circleplus", "circleminus"):
        s = (f'<circle cx="12" cy="12" r="7" fill="none" stroke="{K}" stroke-width="1.3"/>' +
             _line(8.5, 12, 15.5, 12, 1.5))
        return s + (_line(12, 8.5, 12, 15.5, 1.5) if kind == "circleplus" else "")
    if kind in ("radicalcation", "radicalanion"):
        s = dot.format(6.5, 12) + _line(11, 12, 19, 12, 1.6)
        return s + (_line(15, 8, 15, 16, 1.6) if kind == "radicalcation" else "")
    text = {"deltaplus": "δ+", "deltaminus": "δ−", "dagger": "†",
            "ddagger": "‡"}.get(kind, "?")
    return _text(12, 17, text, 12, "normal")


def _shape_icon(arg: str) -> str:
    kind, _, sweep = arg.partition(":")
    st = f'fill="none" stroke="{K}" stroke-width="1.4"'
    if kind == "circle":
        return f'<circle cx="12" cy="12" r="8" {st}/>'
    if kind == "ellipse":
        return f'<ellipse cx="12" cy="12" rx="9" ry="6" {st}/>'
    if kind == "roundrect":
        return f'<rect x="3.5" y="5.5" width="17" height="13" rx="3.5" {st}/>'
    if kind == "shadowrect":
        return ('<rect x="5.5" y="7.5" width="16" height="12" fill="#333"/>'
                f'<rect x="3" y="5" width="16" height="12" fill="#fff" stroke="{K}" '
                'stroke-width="1.4"/>')
    if kind == "arc":
        d = {"90": "M4 18 A12 12 0 0 1 16 6", "120": "M3.5 16 A10 10 0 0 1 20.5 13",
             "270": "M8 19.5 A7.5 7.5 0 1 1 19.5 13"}.get(sweep,
                                                          "M3.5 16 A8.5 8.5 0 0 1 20.5 16")
        return f'<path d="{d}" {st}/>'
    return f'<rect x="3.5" y="5.5" width="17" height="13" {st}/>'


def _clipart_icon(name: str) -> str:
    from rsstatchem.core.chem_draw.clipart import CLIPART
    item = CLIPART.get(name)
    if item is None:
        return _text(12, 16, "?", 12)
    paths, details, _inner = item[1]()
    w, h = item[2]
    k = 19.0 / max(w, h)
    bw, bh = w * k, h * k
    ox, oy = (24 - bw) / 2, (24 - bh) / 2
    out = []
    for pts, closed in paths + details:
        p = " ".join(f"{ox + x * bw:.2f},{oy + y * bh:.2f}" for x, y in pts)
        tag = "polygon" if closed else "polyline"
        out.append(f'<{tag} points="{p}" fill="none" stroke="{K}" stroke-width="1.1" '
                   'stroke-linejoin="round"/>')
    return "".join(out)


def _tool_svg(tool: str) -> str:
    kind, _, arg = tool.partition(":")
    if tool == "lasso":
        return ('<path d="M7 17 C1 12 4 4 12.5 4 C20 4 22.5 9.5 18.5 13.5 C15.5 16.5 10 16 '
                f'8.5 15" fill="none" stroke="{K}" stroke-width="1.3" stroke-dasharray="2 1.6"/>'
                f'<path d="M8.5 15 q-2.5 2 -0.5 5" fill="none" stroke="{K}" stroke-width="1.3"/>')
    if tool == "select":
        return (f'<rect x="4" y="5" width="16" height="14" fill="none" stroke="{K}" '
                'stroke-width="1.3" stroke-dasharray="2.2 1.6"/>')
    if tool == "rotate":
        return (f'<path d="M18.5 9 A7.2 7.2 0 1 0 19 15" fill="none" stroke="{K}" '
                'stroke-width="1.6"/>' + _arrow_head(18.6, 5.4, 0.15, -1, hl=4.5, hw=2.4) +
                _ring(6, r=3.2))
    if tool == "eraser":
        return (_poly([(9, 20), (3.5, 14.5), (12.5, 5.5), (20.5, 13.5), (14, 20)],
                      fill="#F2B8C6", stroke=K, w=1.2) +
                _line(7.4, 10.6, 15.4, 18.6, 1.1) + _line(9, 20, 20, 20, 1.2))
    if kind == "bond":
        return _bond_icon(arg)
    if kind == "orbital":
        k, _, f = arg.partition(":")
        return _orbital_icon(k, f or "shaded")
    if tool == "pen":
        # a fountain-pen nib drawing a Bezier curve (handle line + points in blue)
        return (f'<path d="M2.5 9 C 5 2.5, 9 2.5, 11 6" fill="none" stroke="{B}" '
                'stroke-width="1.3"/>' + _line(2.5, 9, 6.5, 3, 0.8, B) +
                f'<rect x="1.4" y="7.9" width="2.2" height="2.2" fill="{B}"/>' +
                _poly([(4.5, 21), (8, 12.5), (13.5, 7.5), (18.5, 12.5), (13, 17)], fill=K) +
                _line(5.2, 20.3, 10.8, 14.7, 0.9, "#FFFFFF") +
                '<circle cx="11.3" cy="14.2" r="1.25" fill="#FFFFFF"/>' +
                _line(15, 6, 20, 11, 2.6))
    if tool == "fragment":
        return (_ring(6, r=5.2, cx=8, cy=13) + _line(13, 10.5, 18, 8, 1.3) +
                _line(15.5, 3, 15.5, 21, 1.2, "#C62828", "2 1.6") +
                _text(20, 21.5, "m/z", 5.5, "bold", B))
    if tool == "perspective":
        return (_poly([(5, 8), (12, 4.5), (19, 8), (12, 11.5)], fill="#DCE6F5", stroke=K,
                      w=1.0) +
                _poly([(5, 8), (12, 11.5), (12, 20), (5, 16.5)], fill="#B9CBEA", stroke=K,
                      w=1.0) +
                _poly([(19, 8), (12, 11.5), (12, 20), (19, 16.5)], fill="#93AEDD", stroke=K,
                      w=1.0))
    if tool == "chain":
        return (f'<polyline points="2.5,16 7.5,8 12,16 16.5,8 21.5,16" fill="none" '
                f'stroke="{K}" stroke-width="1.5" stroke-linejoin="round"/>')
    if tool == "text":
        return _text(12, 19, "A", 18, family="Times New Roman")
    if kind == "arrow":
        return _arrow_icon(arg)
    if kind == "symbol":
        return _symbol_icon(arg)
    if kind == "shape":
        return _shape_icon(arg)
    if kind == "clipart":
        return _clipart_icon(arg)
    if tool == "tlc":
        return (f'<rect x="5" y="2.5" width="14" height="19" fill="#fff" stroke="{K}" '
                'stroke-width="1.2"/>' + _line(5, 18.5, 19, 18.5, 0.9) +
                _line(5, 5.5, 19, 5.5, 0.9, dash="1.5 1.2") +
                '<ellipse cx="9" cy="13" rx="1.8" ry="1.3" fill="#111"/>'
                '<ellipse cx="15" cy="9" rx="1.8" ry="1.3" fill="#111"/>')
    if tool == "gel":
        return (f'<rect x="3.5" y="3" width="17" height="18" fill="#EEF1F4" stroke="{K}" '
                'stroke-width="1.2"/>' + _line(6, 8, 10.5, 8, 1.6) + _line(6, 12, 10.5, 12, 1.6)
                + _line(6, 17, 10.5, 17, 1.6) + _line(13.5, 10, 18, 10, 1.6) +
                _line(13.5, 15, 18, 15, 1.6))
    if tool == "plus":
        return _line(12, 5, 12, 19, 1.8) + _line(5, 12, 19, 12, 1.8)
    if kind == "bracket" and ":" in arg:
        shape, _, side = arg.partition(":")
        full = _tool_svg("bracket:" + shape)
        clip = ('<clipPath id="c"><rect x="0" y="0" width="12" height="24"/></clipPath>'
                if side == "left" else
                '<clipPath id="c"><rect x="12" y="0" width="12" height="24"/></clipPath>')
        return f'<defs>{clip}</defs><g clip-path="url(#c)">{full}</g>'
    if kind == "bracket":
        if arg == "round":
            return (f'<path d="M8 3.5 Q3 12 8 20.5 M16 3.5 Q21 12 16 20.5" fill="none" '
                    f'stroke="{K}" stroke-width="1.5"/>')
        if arg == "curly":
            return (f'<path d="M8.5 3.5 Q5.5 3.5 6 7.5 T3.5 12 Q6.5 12.5 6 16.5 T8.5 20.5 '
                    f'M15.5 3.5 Q18.5 3.5 18 7.5 T20.5 12 Q17.5 12.5 18 16.5 T15.5 20.5" '
                    f'fill="none" stroke="{K}" stroke-width="1.4"/>')
        return (f'<polyline points="8,3.5 4.5,3.5 4.5,20.5 8,20.5" fill="none" stroke="{K}" '
                f'stroke-width="1.5"/><polyline points="16,3.5 19.5,3.5 19.5,20.5 16,20.5" '
                f'fill="none" stroke="{K}" stroke-width="1.5"/>')
    if tool in ("charge+", "charge-"):
        s = (f'<circle cx="12" cy="12" r="7.5" fill="none" stroke="{K}" stroke-width="1.3"/>' +
             _line(8, 12, 16, 12, 1.6))
        return s + (_line(12, 8, 12, 16, 1.6) if tool == "charge+" else "")
    if tool == "radical":
        return _text(12, 19, "C", 13, "normal") + '<circle cx="12" cy="4.5" r="1.7" fill="#111"/>'
    if tool == "map":
        return _text(10, 17, "C", 12, "normal") + _text(18, 21, "1", 8, color=B)
    if tool == "periodic":
        cells = []
        for r, cols in enumerate(((0, 6), (0, 1, 2, 3, 4, 5, 6), (0, 1, 2, 3, 4, 5, 6))):
            for c in cols:
                fill = "#9EC0EE" if r == 0 or c in (0, 6) else "#FFFFFF"
                cells.append(f'<rect x="{2.5 + c * 2.75:.2f}" y="{6 + r * 4:.2f}" width="2.4" '
                             f'height="3.6" fill="{fill}" stroke="{K}" stroke-width="0.5"/>')
        return "".join(cells)
    if tool == "template":
        return (_ring(6, (0, 2, 4), r=6.8, cx=11, cy=11) +
                _line(17, 19.5, 22, 19.5, 1.6, "#C62828") + _line(19.5, 17, 19.5, 22, 1.6,
                                                                   "#C62828"))
    if kind == "atom":
        return _text(12, 17, arg, 13 if len(arg) == 1 else 11)
    if kind == "ring":
        if arg in ("chair1", "chair2"):
            return _chair(arg == "chair2")
        n = {"cyclopropane": 3, "cyclobutane": 4, "cyclopentane": 5, "cyclohexane": 6,
             "cycloheptane": 7, "cyclooctane": 8, "benzene": 6, "cyclopentadiene": 5}[arg]
        inner = {"benzene": (0, 2, 4), "cyclopentadiene": (1, 3)}.get(arg, ())
        return _ring(n, inner)
    return _text(12, 16, "?", 12)


_GENERAL = {
    "new": (f'<path d="M6 2.5h8l4.5 4.5v14.5H6z" fill="#fff" stroke="{K}" stroke-width="1.2"/>'
            f'<path d="M14 2.5v4.5h4.5" fill="none" stroke="{K}" stroke-width="1.2"/>'),
    "open": (f'<path d="M2.5 6h7l2 2h9.5v11.5h-18.5z" fill="#F4C95D" stroke="{K}" '
             'stroke-width="1.1"/><path d="M2.5 19.5l3-8h17l-3 8z" fill="#F9DB8A" '
             f'stroke="{K}" stroke-width="1.1"/>'),
    "save": (f'<rect x="3" y="3" width="18" height="18" rx="1.5" fill="#4A78C2" stroke="{K}" '
             'stroke-width="1.1"/><rect x="6.5" y="3.5" width="11" height="6.5" fill="#fff"/>'
             '<rect x="6" y="13" width="12" height="7.5" fill="#E8EEF8"/>'
             '<rect x="14" y="4.5" width="2" height="4" fill="#4A78C2"/>'),
    "print": (f'<rect x="6" y="3" width="12" height="6" fill="#fff" stroke="{K}" '
              f'stroke-width="1.1"/><rect x="2.5" y="9" width="19" height="8" rx="1.5" '
              f'fill="#9AA4B2" stroke="{K}" stroke-width="1.1"/><rect x="6" y="14" width="12" '
              f'height="7" fill="#fff" stroke="{K}" stroke-width="1.1"/>'),
    "cut": (f'<circle cx="7" cy="17.5" r="3" fill="none" stroke="{K}" stroke-width="1.4"/>'
            f'<circle cx="17" cy="17.5" r="3" fill="none" stroke="{K}" stroke-width="1.4"/>'
            + _line(9, 15.5, 17, 3, 1.4) + _line(15, 15.5, 7, 3, 1.4)),
    "copy": (f'<rect x="3.5" y="3.5" width="11" height="13" fill="#fff" stroke="{K}" '
             f'stroke-width="1.1"/><rect x="9.5" y="7.5" width="11" height="13" fill="#fff" '
             f'stroke="{K}" stroke-width="1.1"/>'),
    "paste": (f'<rect x="4" y="4.5" width="13" height="16" rx="1" fill="#C89B5A" stroke="{K}" '
              f'stroke-width="1.1"/><rect x="7.5" y="2.5" width="6" height="4" rx="1" '
              f'fill="#ddd" stroke="{K}" stroke-width="1"/><rect x="10" y="9" width="10.5" '
              f'height="12.5" fill="#fff" stroke="{K}" stroke-width="1.1"/>'),
    "undo": (f'<path d="M8.5 7.5H15a5.5 5.5 0 0 1 0 11H8" fill="none" stroke="{K}" '
             'stroke-width="1.8"/>' + _arrow_head(3.5, 7.5, -1, 0, hl=5.5, hw=3.4)),
    "redo": (f'<path d="M15.5 7.5H9a5.5 5.5 0 0 0 0 11h7" fill="none" stroke="{K}" '
             'stroke-width="1.8"/>' + _arrow_head(20.5, 7.5, 1, 0, hl=5.5, hw=3.4)),
    "zoom_in": (f'<circle cx="10" cy="10" r="6.5" fill="#fff" stroke="{K}" stroke-width="1.6"/>'
                + _line(15, 15, 21, 21, 2.6) + _line(7, 10, 13, 10, 1.6) +
                _line(10, 7, 10, 13, 1.6)),
    "zoom_out": (f'<circle cx="10" cy="10" r="6.5" fill="#fff" stroke="{K}" stroke-width="1.6"/>'
                 + _line(15, 15, 21, 21, 2.6) + _line(7, 10, 13, 10, 1.6)),
    "actual": (f'<rect x="3.5" y="3.5" width="17" height="17" fill="#fff" stroke="{K}" '
               f'stroke-width="1.2"/>' + _text(12, 16, "1:1", 8.5)),
    "clean": (_ring(6, r=5.6, cx=9, cy=13) +
              '<path d="M14 3l1.2 2.8 2.8 1.2-2.8 1.2-1.2 2.8-1.2-2.8-2.8-1.2 2.8-1.2z" '
              f'fill="{B}"/><path d="M19.5 9l.7 1.6 1.6.7-1.6.7-.7 1.6-.7-1.6-1.6-.7 1.6-.7z" '
              f'fill="{B}"/>'),
    "check": (_ring(6, r=5.6, cx=9, cy=12) +
              f'<path d="M13.5 15.5l3 3.5 5.5-8" fill="none" stroke="#2E7D32" '
              'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>'),
    "name2struct": (_text(7, 11, "abc", 7.5, "bold", B) + _line(4, 14, 10, 14, 1.2) +
                    _arrow_head(12.5, 14, 1, 0, hl=3, hw=1.8) + _ring(6, r=4.6, cx=17.5,
                                                                         cy=14)),
    "struct2name": (_ring(6, r=4.6, cx=6.5, cy=10) + _line(12, 14, 15, 14, 1.2) +
                    _arrow_head(17.5, 14, 1, 0, hl=3, hw=1.8) +
                    _text(17, 22, "abc", 7.5, "bold", B)),
    "analysis": (f'<rect x="3" y="3" width="18" height="18" rx="1.5" fill="#fff" stroke="{K}" '
                 f'stroke-width="1.1"/>' + _text(12, 11, "C₈H", 6.5, "bold", K) +
                 _line(6, 15, 18, 15, 1, G) + _line(6, 18, 15, 18, 1, G)),
    "properties": (f'<rect x="3" y="3" width="18" height="18" rx="1.5" fill="#fff" '
                   f'stroke="{K}" stroke-width="1.1"/>' + _text(12, 12, "bp", 7.5, "bold", B)
                   + _line(6, 15, 18, 15, 1, G) + _line(6, 18, 15, 18, 1, G)),
    "output": (f'<rect x="3" y="3.5" width="18" height="17" rx="1.5" fill="#fff" stroke="{K}" '
               f'stroke-width="1.1"/><rect x="6" y="12" width="2.6" height="6" fill="{B}"/>'
               f'<rect x="10.5" y="8" width="2.6" height="10" fill="{B}"/>'
               f'<rect x="15" y="10" width="2.6" height="8" fill="{B}"/>'),
    "bold": _text(12, 18, "B", 16, "bold"),
    "italic": _text(12, 18, "I", 16, "normal", family="Times New Roman", style="italic"),
    "underline": _text(12, 16.5, "U", 14, "normal") + _line(6, 20, 18, 20, 1.4),
    "formula": _text(9.5, 15, "C", 13, "normal") + _text(16.5, 20, "2", 9, "normal"),
    "subscript": _text(9, 15, "x", 14, "normal") + _text(17, 20.5, "2", 9, "normal", B),
    "superscript": _text(9, 19, "x", 14, "normal") + _text(17, 11, "2", 9, "normal", B),
    "plain": _text(12, 17, "Aa", 12, "normal"),
    "color": (f'<rect x="3" y="3" width="8" height="8" fill="#D32F2F"/><rect x="13" y="3" '
              f'width="8" height="8" fill="#1976D2"/><rect x="3" y="13" width="8" height="8" '
              f'fill="#388E3C"/><rect x="13" y="13" width="8" height="8" fill="#111"/>'),
}


def svg_markup(name: str, flyout: bool = False) -> str:
    body = _GENERAL[name] if name in _GENERAL else _tool_svg(name)
    if flyout:          # ChemDraw marks tools with a fly-out palette by a corner triangle
        body += _poly([(24, 18.5), (24, 24), (18.5, 24)], fill="#3A3A3A")
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">{body}</svg>'


def _cache() -> dict:
    from rsstatchem.ui.chem_editor import icons as chem_icons
    return chem_icons._CACHE


def cd_icon(name: str, flyout: bool = False) -> QIcon:
    key = f"chemdraw:{name}:{int(flyout)}"
    cache = _cache()
    if key not in cache:
        renderer = QSvgRenderer(QByteArray(svg_markup(name, flyout).encode("utf-8")))
        ic = QIcon()
        for size in (16, 20, 24, 32, 48, 64):
            pm = QPixmap(size, size)
            pm.fill(Qt.GlobalColor.transparent)
            p = QPainter(pm)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            renderer.render(p, QRectF(0, 0, size, size))
            p.end()
            ic.addPixmap(pm)
        cache[key] = ic
    return cache[key]


def color_icon(color: str) -> QIcon:
    key = f"chemdraw:color:{color}"
    cache = _cache()
    if key not in cache:
        pm = QPixmap(16, 16)
        pm.fill(QColor(color or "#000000"))
        p = QPainter(pm)
        p.setPen(QColor("#555555"))
        p.drawRect(0, 0, 15, 15)
        p.end()
        cache[key] = QIcon(pm)
    return cache[key]

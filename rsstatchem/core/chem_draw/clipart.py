"""Clipware: laboratory glassware drawn as vector outlines (no Qt).

ChemDraw's clipware templates are copyrighted art, so these outlines were drawn for
RS-StatChem from simple geometric primitives (see ``resources/templates/ATTRIBUTION.md``);
no third-party artwork is used. Each item is defined in a unit box (x right, y down, both
0..1): stroked paths, optional extra detail strokes (graduations, rims) and the inner outline
of the vessel, which is filled up to the liquid level of a :class:`~.model.Clipart`.
"""
from __future__ import annotations

import math

__all__ = ["CLIPART", "clipart_names", "clipart_title", "clipart_primitives", "clip_below"]


def _arc(cx, cy, rx, ry, a0, a1, n=24) -> list[tuple[float, float]]:
    """Points of an elliptical arc from angle a0 to a1 (degrees, screen y down)."""
    out = []
    for k in range(n + 1):
        a = math.radians(a0 + (a1 - a0) * k / n)
        out.append((cx + rx * math.cos(a), cy + ry * math.sin(a)))
    return out


def _beaker():
    outer = [(0.03, 0.0), (0.1, 0.07), (0.1, 0.96), (0.14, 1.0), (0.86, 1.0), (0.9, 0.96),
             (0.9, 0.04), (0.94, 0.0)]
    ticks = [[(0.74, y), (0.9, y)] for y in (0.3, 0.45, 0.6, 0.75)]
    inner = [(0.1, 0.07), (0.1, 0.96), (0.14, 1.0), (0.86, 1.0), (0.9, 0.96), (0.9, 0.04)]
    return [(outer, False)], [(t, False) for t in ticks], inner


def _erlenmeyer():
    outer = [(0.38, 0.0), (0.38, 0.32), (0.04, 0.94), (0.09, 1.0), (0.91, 1.0), (0.96, 0.94),
             (0.62, 0.32), (0.62, 0.0)]
    rim = [(0.34, 0.0), (0.66, 0.0)]
    return [(outer, False)], [(rim, False)], outer[1:-1]


def _round_bottom():
    cx, cy, r = 0.5, 0.64, 0.35
    a = math.degrees(math.acos(-0.1 / r))          # where the neck (x = 0.4 / 0.6) meets it
    body = _arc(cx, cy, r, r, -a, -180 + a + 360, 48)[::-1]
    outer = [(0.4, 0.0)] + [(0.4, cy - r * math.sin(math.radians(a)))] + body[1:-1] + \
        [(0.6, cy - r * math.sin(math.radians(a))), (0.6, 0.0)]
    rim = [(0.36, 0.0), (0.64, 0.0)]
    return [(outer, False)], [(rim, False)], outer[1:-1]


def _test_tube():
    outer = [(0.3, 0.0), (0.3, 0.82)] + _arc(0.5, 0.82, 0.2, 0.18, 180, 0, 20)[1:-1] + \
        [(0.7, 0.82), (0.7, 0.0)]
    rim = [(0.25, 0.0), (0.75, 0.0)]
    return [(outer, False)], [(rim, False)], outer


def _cylinder():
    outer = [(0.28, 0.0), (0.33, 0.04), (0.33, 0.92), (0.67, 0.92), (0.67, 0.04), (0.7, 0.0)]
    foot = [(0.12, 0.92), (0.88, 0.92), (0.88, 1.0), (0.12, 1.0)]
    ticks = [[(0.33, 0.12 + 0.08 * k), (0.33 + (0.14 if k % 2 == 0 else 0.08),
                                        0.12 + 0.08 * k)] for k in range(10)]
    return [(outer, False), (foot, True)], [(t, False) for t in ticks], outer[1:-1]


def _funnel():
    outer = [(0.02, 0.0), (0.44, 0.5), (0.44, 1.0), (0.56, 1.0), (0.56, 0.5), (0.98, 0.0)]
    rim = [(0.02, 0.0), (0.98, 0.0)]
    return [(outer, False)], [(rim, False)], outer


def _separatory_funnel():
    bulb = [(0.42, 0.1), (0.22, 0.32), (0.18, 0.45), (0.3, 0.58), (0.46, 0.7),
            (0.46, 0.76), (0.54, 0.76), (0.54, 0.7), (0.7, 0.58), (0.82, 0.45),
            (0.78, 0.32), (0.58, 0.1)]
    stopper = [(0.4, 0.0), (0.6, 0.0), (0.58, 0.1), (0.42, 0.1)]
    stem = [(0.48, 0.76), (0.48, 1.0), (0.52, 1.0), (0.52, 0.76)]
    cock = [(0.36, 0.79), (0.64, 0.79), (0.64, 0.84), (0.36, 0.84)]
    return [(bulb, False), (stopper, True), (stem, False), (cock, True)], [], bulb


def _condenser():
    jacket = [(0.3, 0.12), (0.7, 0.12), (0.7, 0.88), (0.3, 0.88)]
    tube = [[(0.45, 0.0), (0.45, 1.0)], [(0.55, 0.0), (0.55, 1.0)]]
    inlet = [[(0.7, 0.78), (0.9, 0.78)], [(0.7, 0.84), (0.9, 0.84)]]
    outlet = [[(0.1, 0.16), (0.3, 0.16)], [(0.1, 0.22), (0.3, 0.22)]]
    return ([(jacket, True)] + [(t, False) for t in tube + inlet + outlet], [], None)


def _burette():
    tube = [(0.42, 0.0), (0.42, 0.78), (0.58, 0.78), (0.58, 0.0)]
    cock = [(0.3, 0.78), (0.7, 0.78), (0.7, 0.84), (0.3, 0.84)]
    tip = [(0.46, 0.84), (0.48, 1.0), (0.52, 1.0), (0.54, 0.84)]
    ticks = [[(0.42, 0.05 + 0.07 * k), (0.42 + (0.1 if k % 2 == 0 else 0.06),
                                        0.05 + 0.07 * k)] for k in range(10)]
    return [(tube, False), (cock, True), (tip, False)], [(t, False) for t in ticks], tube


def _thermometer():
    stem = [(0.45, 0.82), (0.45, 0.03)] + _arc(0.5, 0.03, 0.05, 0.03, 180, 360, 8)[1:-1] + \
        [(0.55, 0.03), (0.55, 0.82)]
    bulb = _arc(0.5, 0.9, 0.1, 0.1, -60, 240, 30)
    column = [[(0.5, 0.86), (0.5, 0.35)]]
    ticks = [[(0.55, 0.1 + 0.08 * k), (0.62, 0.1 + 0.08 * k)] for k in range(9)]
    return ([(stem, False), (bulb, False)], [(c, False) for c in column + ticks], None)


def _petri_dish():
    top = _arc(0.5, 0.35, 0.48, 0.2, 0, 360, 48)
    wall = [[(0.02, 0.35), (0.02, 0.62)], [(0.98, 0.35), (0.98, 0.62)]]
    bottom = _arc(0.5, 0.62, 0.48, 0.2, 0, 180, 24)
    return [(top, True), (bottom, False)] + [(w, False) for w in wall], [], None


def _vial():
    cap = [(0.3, 0.0), (0.7, 0.0), (0.7, 0.16), (0.3, 0.16)]
    body = [(0.32, 0.16), (0.32, 0.96), (0.36, 1.0), (0.64, 1.0), (0.68, 0.96), (0.68, 0.16)]
    return [(cap, True), (body, False)], [], body


# name -> (title, builder, default size (w, h) in bond lengths)
CLIPART = {
    "beaker": ("Beaker", _beaker, (2.0, 2.4)),
    "erlenmeyer": ("Erlenmeyer flask", _erlenmeyer, (2.2, 2.8)),
    "roundbottom": ("Round-bottom flask", _round_bottom, (2.2, 2.8)),
    "testtube": ("Test tube", _test_tube, (0.9, 3.0)),
    "cylinder": ("Graduated cylinder", _cylinder, (1.4, 3.6)),
    "funnel": ("Funnel", _funnel, (2.0, 2.4)),
    "sepfunnel": ("Separatory funnel", _separatory_funnel, (2.0, 3.6)),
    "condenser": ("Liebig condenser", _condenser, (1.4, 4.0)),
    "burette": ("Burette", _burette, (1.0, 4.4)),
    "thermometer": ("Thermometer", _thermometer, (0.8, 3.6)),
    "petri": ("Petri dish", _petri_dish, (2.6, 1.2)),
    "vial": ("Vial", _vial, (1.0, 2.0)),
}


def clipart_names() -> list[str]:
    return list(CLIPART)


def clipart_title(name: str) -> str:
    return CLIPART.get(name, (name,))[0]


def default_size(name: str, bond_length: float) -> tuple[float, float]:
    w, h = CLIPART.get(name, ("", None, (2.0, 2.0)))[2]
    return w * bond_length, h * bond_length


def clip_below(poly, y: float) -> list[tuple[float, float]]:
    """Part of a polygon with y >= ``y`` (Sutherland-Hodgman against one half-plane)."""
    out = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        pin, qin = p[1] >= y, q[1] >= y
        if pin:
            out.append(p)
        if pin != qin:
            t = (y - p[1]) / ((q[1] - p[1]) or 1e-12)
            out.append((p[0] + (q[0] - p[0]) * t, y))
    return out


LIQUID = "#BBD8F2"


def clipart_primitives(c, line_width: float) -> list[tuple]:
    """Drawing primitives of a :class:`~.model.Clipart` (liquid first, then outlines)."""
    item = CLIPART.get(c.name)
    if item is None:
        return [("shape", [(c.x, c.y), (c.x + c.w, c.y), (c.x + c.w, c.y + c.h),
                           (c.x, c.y + c.h)], line_width, True, "", True)]
    paths, details, inner = item[1]()

    def tr(pts):
        return [(c.x + px * c.w, c.y + py * c.h) for px, py in pts]
    out: list[tuple] = []
    if inner and c.fill > 0:
        ys = [py for _, py in inner]
        top, bottom = min(ys), max(ys)
        level = bottom - (bottom - top) * max(0.0, min(1.0, c.fill))
        liquid = clip_below(inner, level)
        if len(liquid) >= 3:
            out.append(("shape", tr(liquid), 0.0, False, LIQUID, True))
    for pts, closed in paths:
        out.append(("shape", tr(pts), line_width, False, "white" if closed else "", closed))
    for pts, closed in details:
        out.append(("shape", tr(pts), line_width * 0.8, False, "", closed))
    return out

"""Drawing geometry of the editor (no Qt): new-bond angles (120 deg zig-zag), chains, ring
templates, angle snapping, double-bond placement and the line/polygon primitives of bonds
and arrows, following ChemDraw's conventions (offset double bonds inside rings, wedges whose
wide end has the bold width, hashed wedges with a fixed hash spacing, notched arrowheads).

Also the Pen tool's cubic Bezier curves (ChemDraw's Curves menu styles: dashed, bold, doubled,
closed, filled, shaded, full / half arrowheads at either end) and the orbital shapes (s, p,
single lobe, hybrid, d(xy), d(z2); plain, shaded, filled or phased lobes).

Arrow kinds also cover ChemDraw's no-go (crossed), unbalanced equilibrium, block (hollow,
filled or shaded), double-headed, arc (90 / 180 / 270 deg) and dipole arrows, each optionally
dashed or bold. Bonds include quadruple and bold-double bonds.

Angles are in radians in screen coordinates (y down), so -30 deg points up-right.
"""
from __future__ import annotations

import math

from rsstatchem.core.chem_draw.model import BENT_ARROWS, Arrow, Bond, Curve, Document, Orbital
from rsstatchem.core.chem_draw.style import Style

__all__ = ["snap_angle", "new_bond_angle", "chain_points", "free_ring", "ring_on_bond",
           "ring_on_atom", "double_bond_side", "bond_primitives", "arrow_primitives",
           "arrow_path_points", "head_size", "ray_box_exit", "bracket_primitives",
           "bond_end_point", "curve_path_points", "curve_segments", "curve_primitives",
           "smooth_curve_points", "orbital_outline", "orbital_primitives", "bezier_point"]

D30 = math.radians(30)


def _ang(dx: float, dy: float) -> float:
    return math.atan2(dy, dx)


def _norm(a: float) -> float:
    while a <= -math.pi:
        a += 2 * math.pi
    while a > math.pi:
        a -= 2 * math.pi
    return a


def snap_angle(angle: float, step_deg: float = 15.0) -> float:
    step = math.radians(step_deg)
    return round(angle / step) * step


def bond_end_point(x: float, y: float, angle: float, length: float) -> tuple[float, float]:
    return x + length * math.cos(angle), y + length * math.sin(angle)


def new_bond_angle(doc: Document, aid: int) -> float:
    """Direction of a bond added to atom ``aid`` (ChemDraw: 120 deg zig-zag, linear at sp
    centres, largest gap otherwise)."""
    atom = doc.atoms[aid]
    nbrs = doc.neighbors(aid)
    if not nbrs:
        return -D30
    angles = [_ang(doc.atoms[n].x - atom.x, doc.atoms[n].y - atom.y) for n in nbrs]
    if len(nbrs) == 1:
        n = nbrs[0]
        base = angles[0]
        bond = doc.bond_between(aid, n)
        doubles = sum(1 for b in doc.bonds_of(aid) if b.order == 2)
        if bond is not None and (bond.order == 3 or doubles >= 2):
            return _norm(base + math.pi)
        cands = [_norm(base + math.radians(120)), _norm(base - math.radians(120))]
        others = [m for m in doc.neighbors(n) if m != aid]
        if others:
            nb = doc.atoms[n]
            m = doc.atoms[others[0]]
            target = _ang(nb.x - m.x, nb.y - m.y)     # continue the zig-zag (trans)
            return min(cands, key=lambda c: abs(_norm(c - target)))
        return max(cands, key=lambda c: (round(math.cos(c), 6), -math.sin(c)))
    angles.sort()
    best_gap, best_mid = -1.0, 0.0
    for i, a in enumerate(angles):
        b = angles[(i + 1) % len(angles)] + (2 * math.pi if i == len(angles) - 1 else 0)
        if b - a > best_gap:
            best_gap, best_mid = b - a, (a + b) / 2
    return _norm(best_mid)


def chain_points(x0: float, y0: float, direction: float, n: int, length: float,
                 first_sign: int = 1) -> list[tuple[float, float]]:
    """n zig-zag bond end points from (x0, y0) along ``direction`` (bonds at +-30 deg)."""
    pts, x, y = [], x0, y0
    sign = -first_sign
    for _ in range(n):
        a = direction + sign * D30
        x, y = x + length * math.cos(a), y + length * math.sin(a)
        pts.append((x, y))
        sign = -sign
    return pts


def free_ring(cx: float, cy: float, n: int, length: float) -> list[tuple[float, float]]:
    """Regular ring centred on (cx, cy): pointed top (benzene with vertical sides),
    flat top for 4- and 8-membered rings."""
    r = length / (2 * math.sin(math.pi / n))
    start = -math.pi / 2 + (math.pi / n if n % 4 == 0 else 0.0)
    return [(cx + r * math.cos(start + 2 * math.pi * k / n),
             cy + r * math.sin(start + 2 * math.pi * k / n)) for k in range(n)]


def ring_on_bond(x1, y1, x2, y2, n: int, side: int) -> list[tuple[float, float]]:
    """Vertices of a regular n-ring fused on the edge (x1, y1)-(x2, y2), starting with the
    two edge atoms; ``side`` (+1/-1) picks the half-plane of the ring centre (+1 = left
    normal (-dy, dx))."""
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy) or 1.0
    apothem = L / (2 * math.tan(math.pi / n))
    nx, ny = -dy / L * side, dx / L * side
    cx, cy = (x1 + x2) / 2 + nx * apothem, (y1 + y2) / 2 + ny * apothem
    r = math.hypot(x1 - cx, y1 - cy)
    a1, a2 = _ang(x1 - cx, y1 - cy), _ang(x2 - cx, y2 - cy)
    step = _norm(a2 - a1)
    return [(cx + r * math.cos(a1 + step * k), cy + r * math.sin(a1 + step * k))
            for k in range(n)]


def ring_on_atom(x: float, y: float, n: int, direction: float,
                 length: float) -> list[tuple[float, float]]:
    """Ring that contains the vertex (x, y) and whose centre lies along ``direction``."""
    r = length / (2 * math.sin(math.pi / n))
    cx, cy = x + r * math.cos(direction), y + r * math.sin(direction)
    a0 = _ang(x - cx, y - cy)
    return [(cx + r * math.cos(a0 + 2 * math.pi * k / n),
             cy + r * math.sin(a0 + 2 * math.pi * k / n)) for k in range(n)]


def away_direction(doc: Document, aid: int) -> float:
    """Direction pointing away from the neighbours of an atom (up when it has none)."""
    a = doc.atoms[aid]
    sx = sy = 0.0
    for n in doc.neighbors(aid):
        o = doc.atoms[n]
        d = math.hypot(o.x - a.x, o.y - a.y) or 1.0
        sx += (o.x - a.x) / d
        sy += (o.y - a.y) / d
    if math.hypot(sx, sy) < 1e-6:
        return new_bond_angle(doc, aid) if doc.neighbors(aid) else -math.pi / 2
    return _ang(-sx, -sy)


def ring_side_for_bond(doc: Document, bond: Bond, px: float | None = None,
                       py: float | None = None) -> int:
    """Side (+1 left normal / -1) for fusing a ring on a bond: away from the atoms already
    attached to the bond, else towards the pointer."""
    a, b = doc.atoms[bond.a], doc.atoms[bond.b]
    dx, dy = b.x - a.x, b.y - a.y
    mx, my = (a.x + b.x) / 2, (a.y + b.y) / 2
    score = 0.0
    for end in (bond.a, bond.b):
        for n in doc.neighbors(end):
            if n in (bond.a, bond.b):
                continue
            o = doc.atoms[n]
            score += (-dy) * (o.x - mx) + dx * (o.y - my)
    if abs(score) > 1e-6:
        return -1 if score > 0 else 1
    if px is not None:
        s = (-dy) * (px - mx) + dx * (py - my)
        if abs(s) > 1e-6:
            return 1 if s > 0 else -1
    return 1 if dx >= 0 else -1


# --- double bonds ---------------------------------------------------------------------------

def double_bond_side(doc: Document, bond: Bond, labelled=None, rings=None) -> int:
    """+1 / -1: the second line lies on the left (-dy, dx) / right normal of a->b;
    0: centred. ``labelled`` = set of atom ids with visible labels."""
    if bond.double_pos == "center":
        return 0
    if bond.double_pos == "left":
        return 1
    if bond.double_pos == "right":
        return -1
    labelled = labelled or set()
    a, b = doc.atoms[bond.a], doc.atoms[bond.b]
    dx, dy = b.x - a.x, b.y - a.y
    ring = None
    if rings is not None:
        cands = [r for r in rings if bond.a in r and bond.b in r and
                 _adjacent_in(r, bond.a, bond.b)]
        if cands:
            ring = min(cands, key=lambda r: (len(r), -_doubles_in(doc, r)))
    else:
        ring = doc.ring_of_bond(bond.id)
    if ring:
        cx = sum(doc.atoms[i].x for i in ring) / len(ring)
        cy = sum(doc.atoms[i].y for i in ring) / len(ring)
        s = (-dy) * (cx - a.x) + dx * (cy - a.y)
        return 1 if s > 0 else -1
    if bond.a in labelled and bond.b in labelled:
        return 0
    left = right = 0
    ends_other = {}
    for end in (bond.a, bond.b):
        others = [n for n in doc.neighbors(end) if n not in (bond.a, bond.b)]
        ends_other[end] = others
        for n in others:
            o = doc.atoms[n]
            s = (-dy) * (o.x - a.x) + dx * (o.y - a.y)
            if s > 1e-6:
                left += 1
            elif s < -1e-6:
                right += 1
    if not ends_other[bond.a] and not ends_other[bond.b]:
        return 0
    for end in (bond.a, bond.b):
        if not ends_other[end] and end in labelled:
            return 0
    if left == right:
        return 1 if left else 0
    return 1 if left > right else -1


def _adjacent_in(ring, a, b) -> bool:
    i, j = ring.index(a), ring.index(b)
    return abs(i - j) in (1, len(ring) - 1)


def _doubles_in(doc: Document, ring) -> int:
    n = 0
    for i in range(len(ring)):
        b = doc.bond_between(ring[i], ring[(i + 1) % len(ring)])
        if b is not None and b.order in (2, 1.5):
            n += 1
    return n


# --- primitives -----------------------------------------------------------------------------
# ("line", x1, y1, x2, y2, width, dashed)   ("poly", [(x, y), ...])   ("path", pts, width)

def _shorten(doc: Document, end: int, other: int, side: int, nx: float, ny: float,
             spacing: float, length: float) -> float:
    e, o = doc.atoms[end], doc.atoms[other]
    ux, uy = o.x - e.x, o.y - e.y
    d = math.hypot(ux, uy) or 1.0
    ux, uy = ux / d, uy / d
    best = None
    for n in doc.neighbors(end):
        if n == other:
            continue
        m = doc.atoms[n]
        vx, vy = m.x - e.x, m.y - e.y
        dv = math.hypot(vx, vy) or 1.0
        vx, vy = vx / dv, vy / dv
        if side and (vx * nx + vy * ny) * side <= 0:
            continue
        theta = math.acos(max(-1.0, min(1.0, ux * vx + uy * vy)))
        if theta < 1e-3:
            continue
        s = spacing / math.tan(theta / 2)
        best = s if best is None else max(best, s)
    if best is None:
        return 0.1 * length
    return max(0.05 * length, min(best, 0.35 * length))


def bond_primitives(doc: Document, bond: Bond, style: Style, trim_a: float = 0.0,
                    trim_b: float = 0.0, side: int | None = None,
                    labelled=None) -> list[tuple]:
    a, b = doc.atoms[bond.a], doc.atoms[bond.b]
    dx, dy = b.x - a.x, b.y - a.y
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return []
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux
    lw = style.line_width
    sp = style.spacing
    p = (a.x + ux * trim_a, a.y + uy * trim_a)
    q = (b.x - ux * trim_b, b.y - uy * trim_b)
    if (q[0] - p[0]) * ux + (q[1] - p[1]) * uy <= 0.5:
        return []
    out: list[tuple] = []

    def line(p1, p2, width=lw, dashed=False):
        out.append(("line", p1[0], p1[1], p2[0], p2[1], width, dashed))

    def off(pt, k):
        return (pt[0] + nx * k, pt[1] + ny * k)

    order = bond.order
    st = bond.style
    if order == 4:                          # quadruple bond
        for k in (-1.5, -0.5, 0.5, 1.5):
            line(off(p, sp * k), off(q, sp * k))
        return out
    if order in (2, 1.5) and st != "either":
        if side is None:
            side = double_bond_side(doc, bond, labelled)
        if side == 0 and order == 2:
            if st == "bold":
                w = style.bold_width / 2
                c = off(p, sp / 2), off(q, sp / 2)
                out.append(("poly", [off(c[0], w), off(c[1], w), off(c[1], -w),
                                     off(c[0], -w)]))
            else:
                line(off(p, sp / 2), off(q, sp / 2))
            line(off(p, -sp / 2), off(q, -sp / 2))
            return out
        side = side or 1
        if st == "bold" and order == 2:     # Bold Double Bond: the main line is bold
            w = style.bold_width / 2
            out.append(("poly", [off(p, w), off(q, w), off(q, -w), off(p, -w)]))
        else:
            line(p, q)
        sa = trim_a if bond.a in (labelled or ()) else max(
            trim_a, _shorten(doc, bond.a, bond.b, side, nx, ny, sp, L))
        sb = trim_b if bond.b in (labelled or ()) else max(
            trim_b, _shorten(doc, bond.b, bond.a, side, nx, ny, sp, L))
        p2 = (a.x + ux * sa + nx * sp * side, a.y + uy * sa + ny * sp * side)
        q2 = (b.x - ux * sb + nx * sp * side, b.y - uy * sb + ny * sp * side)
        if (q2[0] - p2[0]) * ux + (q2[1] - p2[1]) * uy > 0.5:
            line(p2, q2, dashed=order == 1.5)
        return out
    if order == 2 and st == "either":
        line(off(p, sp / 2), off(q, -sp / 2))
        line(off(p, -sp / 2), off(q, sp / 2))
        return out
    if order == 3:
        line(p, q)
        line(off(p, sp), off(q, sp))
        line(off(p, -sp), off(q, -sp))
        return out
    if st == "wedge":
        w = style.wedge_width / 2
        tip = lw / 2
        out.append(("poly", [off(p, tip), off(q, w), off(q, -w), off(p, -tip)]))
        return out
    if st == "hollow":                      # Hollow Wedged Bond: outlined wedge
        w = style.wedge_width / 2
        out.append(("shape", [p, off(q, w), off(q, -w)], lw, False, "white", True))
        return out
    if st == "hashed":                      # Hashed Bond: parallel strokes of equal width
        w = style.wedge_width * 0.4
        length = math.hypot(q[0] - p[0], q[1] - p[1])
        n = max(2, int(length / style.hash_spacing))
        for k in range(n + 1):
            t = (k + 0.5) / (n + 1)
            c = (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)
            line(off(c, w), off(c, -w))
        return out
    if st == "dative":                      # Dative Bond: arrow from a (donor) to b
        hl, hw = head_size(style)
        hl, hw = hl * 0.75, hw * 0.75
        notch = (q[0] - ux * hl * 0.78, q[1] - uy * hl * 0.78)
        line(p, notch)
        out.append(("poly", _head(q, ux, uy, hl, hw)))
        return out
    if st == "hash":
        w = style.wedge_width / 2
        length = math.hypot(q[0] - p[0], q[1] - p[1])
        n = max(2, int(length / style.hash_spacing))
        for k in range(n + 1):
            t = k / n
            if k == 0:
                t = 0.5 / n
            c = (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)
            hw = lw / 2 + (w - lw / 2) * t
            line(off(c, hw), off(c, -hw))
        return out
    if st == "wavy":
        length = math.hypot(q[0] - p[0], q[1] - p[1])
        waves = max(2, round(length / (style.bond_length * 0.16)))
        amp = style.wedge_width * 0.4
        pts = []
        steps = waves * 12
        for k in range(steps + 1):
            t = k / steps
            c = (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)
            pts.append(off(c, amp * math.sin(t * waves * math.pi)))
        out.append(("path", pts, lw))
        return out
    if st == "bold":
        w = style.bold_width / 2
        out.append(("poly", [off(p, w), off(q, w), off(q, -w), off(p, -w)]))
        return out
    if st == "dashed":
        line(p, q, dashed=True)
        return out
    line(p, q)
    return out


def ray_box_exit(cx: float, cy: float, ux: float, uy: float,
                 box: tuple[float, float, float, float]) -> float:
    """Distance from (cx, cy), which lies inside ``box`` (x0, y0, x1, y1), along (ux, uy)
    to the border of the box."""
    x0, y0, x1, y1 = box
    ts = []
    if ux > 1e-9:
        ts.append((x1 - cx) / ux)
    elif ux < -1e-9:
        ts.append((x0 - cx) / ux)
    if uy > 1e-9:
        ts.append((y1 - cy) / uy)
    elif uy < -1e-9:
        ts.append((y0 - cy) / uy)
    ts = [t for t in ts if t >= 0]
    return min(ts) if ts else 0.0


# --- arrows ---------------------------------------------------------------------------------

def head_size(style: Style) -> tuple[float, float]:
    """(length, half width) of an arrowhead."""
    hl = max(4.5, style.bond_length * 0.32 + style.line_width * 1.5)
    return hl, hl * 0.36


def arc_points(x1, y1, x2, y2, sweep_deg: float, side: int, samples: int = 36
               ) -> list[tuple[float, float]]:
    """Circular arc from (x1, y1) to (x2, y2) spanning ``sweep_deg`` degrees and bulging to
    ``side`` (+1 = left normal (-dy, dx) of the chord, -1 = right)."""
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    th = math.radians(max(10.0, min(350.0, sweep_deg)))
    if L < 1e-9:
        return [(x1, y1), (x2, y2)]
    nx, ny = -dy / L, dx / L
    r = L / (2 * math.sin(th / 2))
    d = r * math.cos(th / 2)
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    cx, cy = mx - side * nx * d, my - side * ny * d
    a0 = math.atan2(y1 - cy, x1 - cx)
    best = None
    for direction in (1, -1):
        pts = [(cx + r * math.cos(a0 + direction * th * k / samples),
                cy + r * math.sin(a0 + direction * th * k / samples))
               for k in range(samples + 1)]
        mid = pts[samples // 2]
        on_side = ((mid[0] - mx) * nx + (mid[1] - my) * ny) * side
        err = math.hypot(pts[-1][0] - x2, pts[-1][1] - y2)
        score = (err > 1e-3 * max(1.0, L), -on_side)
        if best is None or score < best[0]:
            best = (score, pts)
    pts = best[1]
    pts[-1] = (x2, y2)
    return pts


def arrow_path_points(ar: Arrow, samples: int = 24) -> list[tuple[float, float]]:
    """Points along the arrow shaft (a quadratic curve for curved arrows, a circular arc
    for arc arrows)."""
    if ar.kind == "arc":
        return arc_points(ar.x1, ar.y1, ar.x2, ar.y2, ar.sweep, -1 if ar.bend < 0 else 1,
                          max(samples, 36))
    if ar.kind not in BENT_ARROWS or abs(ar.bend) < 1e-6:
        return [(ar.x1, ar.y1), (ar.x2, ar.y2)]
    dx, dy = ar.x2 - ar.x1, ar.y2 - ar.y1
    L = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / L, dx / L
    mx, my = (ar.x1 + ar.x2) / 2, (ar.y1 + ar.y2) / 2
    cx, cy = mx + nx * 2 * ar.bend, my + ny * 2 * ar.bend
    pts = []
    for k in range(samples + 1):
        t = k / samples
        x = (1 - t) ** 2 * ar.x1 + 2 * (1 - t) * t * cx + t * t * ar.x2
        y = (1 - t) ** 2 * ar.y1 + 2 * (1 - t) * t * cy + t * t * ar.y2
        pts.append((x, y))
    return pts


def _head(tip, ux, uy, hl, hw, half: int = 0) -> list[tuple[float, float]]:
    """Notched arrowhead polygon; ``half`` = +1/-1 keeps one barb (left/right normal)."""
    nx, ny = -uy, ux
    back = (tip[0] - ux * hl, tip[1] - uy * hl)
    notch = (tip[0] - ux * hl * 0.78, tip[1] - uy * hl * 0.78)
    left = (back[0] + nx * hw, back[1] + ny * hw)
    right = (back[0] - nx * hw, back[1] - ny * hw)
    if half > 0:
        return [tip, left, notch]
    if half < 0:
        return [tip, notch, right]
    return [tip, left, notch, right]


def _dash_lines(out: list, dashed: bool) -> list:
    """Make the shaft lines / paths of an arrow dashed."""
    if not dashed:
        return out
    res = []
    for prim in out:
        if prim[0] == "line":
            res.append(prim[:6] + (True,))
        elif prim[0] == "path":
            res.append(("shape", prim[1], prim[2], True, "", False))
        else:
            res.append(prim)
    return res


def arrow_primitives(ar: Arrow, style: Style) -> list[tuple]:
    out = _arrow_primitives(ar, style)
    return _dash_lines(out, getattr(ar, "dashed", False))


def _block_arrow(tail, tip, ux, uy, L, style: Style, fill: str, lw: float) -> list[tuple]:
    nx, ny = -uy, ux
    w1 = max(style.bond_length * 0.13, 2.5)
    w2 = w1 * 2.2
    hl = min(L * 0.45, w2 * 1.7)
    neck = (tip[0] - ux * hl, tip[1] - uy * hl)
    pts = [(tail[0] + nx * w1, tail[1] + ny * w1), (neck[0] + nx * w1, neck[1] + ny * w1),
           (neck[0] + nx * w2, neck[1] + ny * w2), tip,
           (neck[0] - nx * w2, neck[1] - ny * w2), (neck[0] - nx * w1, neck[1] - ny * w1),
           (tail[0] - nx * w1, tail[1] - ny * w1)]
    f = {"filled": "solid", "shaded": "shade"}.get(fill, "white")
    return [("shape", pts, lw, False, f, True)]


def _arrow_primitives(ar: Arrow, style: Style) -> list[tuple]:
    lw = style.bold_width * 0.75 if getattr(ar, "bold", False) else style.line_width
    hl, hw = head_size(style)
    if getattr(ar, "bold", False):
        hl, hw = hl * 1.25, hw * 1.35
    out: list[tuple] = []
    dx, dy = ar.x2 - ar.x1, ar.y2 - ar.y1
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return out
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux
    tail, tip = (ar.x1, ar.y1), (ar.x2, ar.y2)
    k = ar.kind
    if k in ("curved", "fishhook", "double", "arc"):
        pts = arrow_path_points(ar)
        hl2, hw2 = hl * 0.9, hw * 0.9
        cut = hl2 * 0.7
        heads = []
        half = 0
        if k == "fishhook":
            half = 1 if ar.bend >= 0 else -1
        ex, ey = _end_tangent(pts, True)
        heads.append(("poly", _head(tip, ex, ey, hl2, hw2 * (1.15 if half else 1), half)))
        pts = _trim(pts, cut, True)
        if k == "double":
            sx, sy = _end_tangent(pts, False)
            heads.append(("poly", _head(tail, sx, sy, hl2, hw2)))
            pts = _trim(pts, cut, False)
        out.append(("path", pts, lw))
        return out + heads
    if k == "block":
        return _block_arrow(tail, tip, ux, uy, L, style, getattr(ar, "fill", ""),
                            style.line_width)
    if k == "nogo":
        notch = (tip[0] - ux * hl * 0.78, tip[1] - uy * hl * 0.78)
        out.append(("line", tail[0], tail[1], notch[0], notch[1], lw, False))
        out.append(("poly", _head(tip, ux, uy, hl, hw)))
        mx, my = (tail[0] + tip[0]) / 2, (tail[1] + tip[1]) / 2
        c = max(hw * 1.8, style.bond_length * 0.18)
        for s in (1, -1):
            ax, ay = (ux + s * nx) * c / math.sqrt(2), (uy + s * ny) * c / math.sqrt(2)
            out.append(("line", mx - ax, my - ay, mx + ax, my + ay, lw, False))
        return out
    if k == "dipole":
        notch = (tip[0] - ux * hl * 0.78, tip[1] - uy * hl * 0.78)
        out.append(("line", tail[0], tail[1], notch[0], notch[1], lw, False))
        out.append(("poly", _head(tip, ux, uy, hl, hw)))
        cx, cy = tail[0] + ux * min(L * 0.15, hl * 1.2), tail[1] + uy * min(L * 0.15, hl * 1.2)
        c = hw * 1.6
        out.append(("line", cx - nx * c, cy - ny * c, cx + nx * c, cy + ny * c, lw, False))
        return out
    if k == "unbalanced":
        # unbalanced equilibrium: long forward harpoon on top, short reverse harpoon below
        g = max(hw * 0.9, 2.2 * lw)
        t1 = (tail[0] - nx * g / 2, tail[1] - ny * g / 2)
        h1 = (tip[0] - nx * g / 2, tip[1] - ny * g / 2)
        out.append(("line", t1[0], t1[1], h1[0] - ux * hl * 0.6, h1[1] - uy * hl * 0.6, lw,
                    False))
        out.append(("poly", _head(h1, ux, uy, hl, hw * 1.1, half=-1)))
        f0, f1 = 0.3, 0.7
        t2 = (tail[0] + ux * L * f1 + nx * g / 2, tail[1] + uy * L * f1 + ny * g / 2)
        h2 = (tail[0] + ux * L * f0 + nx * g / 2, tail[1] + uy * L * f0 + ny * g / 2)
        out.append(("line", t2[0], t2[1], h2[0] + ux * hl * 0.6, h2[1] + uy * hl * 0.6, lw,
                    False))
        out.append(("poly", _head(h2, -ux, -uy, hl, hw * 1.1, half=-1)))
        return out
    if k == "line":
        out.append(("line", tail[0], tail[1], tip[0], tip[1], lw, False))
        return out
    if k == "forward":
        notch = (tip[0] - ux * hl * 0.78, tip[1] - uy * hl * 0.78)
        out.append(("line", tail[0], tail[1], notch[0], notch[1], lw, False))
        out.append(("poly", _head(tip, ux, uy, hl, hw)))
        return out
    if k == "resonance":
        n1 = (tip[0] - ux * hl * 0.78, tip[1] - uy * hl * 0.78)
        n2 = (tail[0] + ux * hl * 0.78, tail[1] + uy * hl * 0.78)
        out.append(("line", n2[0], n2[1], n1[0], n1[1], lw, False))
        out.append(("poly", _head(tip, ux, uy, hl, hw)))
        out.append(("poly", _head(tail, -ux, -uy, hl, hw)))
        return out
    if k == "equilibrium":
        # forward harpoon on top (screen up = -n when the arrow points right), barbs out
        g = max(hw * 0.9, 2.2 * lw)
        t1 = (tail[0] - nx * g / 2, tail[1] - ny * g / 2)
        h1 = (tip[0] - nx * g / 2, tip[1] - ny * g / 2)
        t2 = (tip[0] + nx * g / 2, tip[1] + ny * g / 2)
        h2 = (tail[0] + nx * g / 2, tail[1] + ny * g / 2)
        out.append(("line", t1[0], t1[1], h1[0] - ux * hl * 0.6, h1[1] - uy * hl * 0.6, lw,
                    False))
        out.append(("poly", _head(h1, ux, uy, hl, hw * 1.1, half=-1)))
        out.append(("line", t2[0], t2[1], h2[0] + ux * hl * 0.6, h2[1] + uy * hl * 0.6, lw,
                    False))
        out.append(("poly", _head(h2, -ux, -uy, hl, hw * 1.1, half=-1)))
        return out
    if k == "retro":
        g = max(hw * 0.9, 2.5 * lw)
        wing = g * 1.9
        back = (tip[0] - ux * hl, tip[1] - uy * hl)
        for s in (1, -1):
            # parallel lines stop where they meet the open chevron
            f = 1 - (g / 2) / wing
            end = (tip[0] - ux * hl * f + nx * s * g / 2, tip[1] - uy * hl * f + ny * s * g / 2)
            out.append(("line", tail[0] + nx * s * g / 2, tail[1] + ny * s * g / 2, end[0],
                        end[1], lw, False))
        out.append(("path", [(back[0] + nx * wing, back[1] + ny * wing), tip,
                             (back[0] - nx * wing, back[1] - ny * wing)], lw))
        return out
    out.append(("line", tail[0], tail[1], tip[0], tip[1], lw, False))
    return out


# --- curves (Pen tool) -----------------------------------------------------------------------

def bezier_point(p0, c0, c1, p1, t: float) -> tuple[float, float]:
    u = 1 - t
    return (u ** 3 * p0[0] + 3 * u * u * t * c0[0] + 3 * u * t * t * c1[0] + t ** 3 * p1[0],
            u ** 3 * p0[1] + 3 * u * u * t * c0[1] + 3 * u * t * t * c1[1] + t ** 3 * p1[1])


def curve_segments(c: Curve) -> list[tuple]:
    """Cubic segments (p0, c0, c1, p1) of a curve (closing segment included)."""
    pts = c.points
    n = len(pts) // 3
    segs = []
    for i in range(n - 1):
        segs.append((pts[3 * i + 1], pts[3 * i + 2], pts[3 * i + 3], pts[3 * i + 4]))
    if c.closed and n > 1:
        segs.append((pts[3 * n - 2], pts[3 * n - 1], pts[0], pts[1]))
    return segs


def curve_path_points(c: Curve, samples: int = 20) -> list[tuple[float, float]]:
    """Polyline approximation of the curve."""
    segs = curve_segments(c)
    if not segs:
        return [tuple(p) for p in c.anchors()]
    out = [tuple(segs[0][0])]
    for seg in segs:
        for k in range(1, samples + 1):
            out.append(bezier_point(*seg, k / samples))
    return out


def smooth_curve_points(anchors, closed: bool = False, tension: float = 1 / 6
                        ) -> list[list[float]]:
    """Curve points (3 per anchor) of a smooth curve through ``anchors`` (Catmull-Rom
    handles), as the Pen tool makes when every point is dragged."""
    n = len(anchors)
    out = []
    for i, (x, y) in enumerate(anchors):
        if closed:
            px, py = anchors[(i - 1) % n]
            nx, ny = anchors[(i + 1) % n]
        else:
            px, py = anchors[max(0, i - 1)]
            nx, ny = anchors[min(n - 1, i + 1)]
        tx, ty = (nx - px) * tension, (ny - py) * tension
        out += [[x - tx, y - ty], [float(x), float(y)], [x + tx, y + ty]]
    return out


def _offset_polyline(pts, d: float) -> list[tuple[float, float]]:
    out = []
    n = len(pts)
    for i in range(n):
        a = pts[max(0, i - 1)]
        b = pts[min(n - 1, i + 1)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy) or 1.0
        out.append((pts[i][0] - dy / L * d, pts[i][1] + dx / L * d))
    return out


def _end_tangent(pts, at_end: bool) -> tuple[float, float]:
    """Unit direction of travel at the end (or reversed, at the start) of a polyline."""
    if at_end:
        tip = pts[-1]
        seq = reversed(pts[:-1])
    else:
        tip = pts[0]
        seq = pts[1:]
    for q in seq:
        dx, dy = tip[0] - q[0], tip[1] - q[1]
        L = math.hypot(dx, dy)
        if L > 1e-6:
            return dx / L, dy / L
    return 1.0, 0.0


def _trim(pts, cut: float, at_end: bool) -> list:
    """Remove the part of a polyline within ``cut`` of its end (or start)."""
    seq = list(pts) if at_end else list(reversed(pts))
    tip = seq[-1]
    kept = [q for q in seq[:-1] if math.hypot(q[0] - tip[0], q[1] - tip[1]) > cut]
    if not kept:
        kept = [seq[0]]
    ux, uy = _end_tangent(seq, True)
    kept.append((tip[0] - ux * cut, tip[1] - uy * cut))
    return kept if at_end else list(reversed(kept))


def curve_primitives(c: Curve, style: Style) -> list[tuple]:
    """Drawing primitives of a Pen-tool curve with its Curves-menu style."""
    pts = curve_path_points(c)
    if len(pts) < 2:
        return []
    lw = style.bold_width if c.bold else style.line_width
    out: list[tuple] = []
    closed = c.closed and len(c.anchors()) > 2
    arrows = not closed and (c.arrow_start or c.arrow_end)
    fill = ""
    if closed and not c.dashed:
        fill = "shade" if c.shaded else ("solid" if c.filled else "")
        if not fill and getattr(c, "faded", False):
            fill = "fade"
    hl, hw = head_size(style)
    hl, hw = hl * (1.15 if c.bold else 0.95), hw * (1.25 if c.bold else 0.95)
    heads = []
    if arrows:
        for kind, at_end in ((c.arrow_end, True), (c.arrow_start, False)):
            if not kind:
                continue
            ux, uy = _end_tangent(pts, at_end)
            tip = pts[-1] if at_end else pts[0]
            # half arrowheads keep the barb on the outside (left of travel) as in ChemDraw
            half = 0 if kind == "full" else (1 if at_end else -1)
            heads.append(("poly", _head(tip, ux, uy, hl, hw * (1.15 if half else 1), half)))
            pts = _trim(pts, hl * 0.7, at_end)
    if c.doubled and not fill and not c.dashed and not arrows:
        g = max(style.spacing / 2, lw * 1.5)
        for d in (g, -g):
            out.append(("shape", _offset_polyline(pts, d), lw, False, "", closed))
    else:
        out.append(("shape", pts, lw, c.dashed, fill, closed))
    return out + heads


# --- orbitals -------------------------------------------------------------------------------

def _lobe(cx, cy, angle: float, length: float, fat: float = 1.0, n: int = 40
          ) -> list[tuple[float, float]]:
    """Teardrop lobe with its point at (cx, cy) along ``angle`` (r = L cos^2 phi)."""
    pts = []
    ca, sa = math.cos(angle), math.sin(angle)
    for k in range(n + 1):
        phi = -math.pi / 2 + math.pi * k / n
        r = length * math.cos(phi) ** 2
        u, v = r * math.cos(phi), r * math.sin(phi) * fat
        pts.append((cx + u * ca - v * sa, cy + u * sa + v * ca))
    return pts


def _ellipse(cx, cy, a: float, b: float, angle: float, n: int = 48):
    ca, sa = math.cos(angle), math.sin(angle)
    return [(cx + a * math.cos(t) * ca - b * math.sin(t) * sa,
             cy + a * math.cos(t) * sa + b * math.sin(t) * ca)
            for t in (2 * math.pi * k / n for k in range(n))]


def orbital_outline(o: Orbital, bond_length: float) -> list[tuple[list, str]]:
    """[(polygon, fill)] of an orbital, back to front; fill is "white" (unshaded),
    "shade" (gradient) or "solid"."""
    L = o.size or bond_length
    a, x, y = o.angle, o.x, o.y
    if o.kind == "s":
        lobes = [_ellipse(x, y, L * 0.55, L * 0.55, a)]
    elif o.kind == "lobe":
        lobes = [_lobe(x, y, a, L)]
    elif o.kind == "hybrid":
        lobes = [_lobe(x, y, a + math.pi, L * 0.38), _lobe(x, y, a, L, 1.15)]
    elif o.kind == "dxy":
        lobes = [_lobe(x, y, a + math.pi / 4 + k * math.pi / 2, L * 0.9, 0.95)
                 for k in range(4)]
    elif o.kind == "dz2":
        lobes = [_ellipse(x, y, L * 0.2, L * 0.5, a), _lobe(x, y, a, L),
                 _lobe(x, y, a + math.pi, L)]
    else:                                                   # p
        lobes = [_lobe(x, y, a, L), _lobe(x, y, a + math.pi, L)]
    base = {"plain": "white", "shaded": "shade", "filled": "solid"}.get(o.fill, "shade")
    out = []
    for k, lobe in enumerate(lobes):
        fill = base
        if o.fill == "phase":
            if o.kind == "dz2":
                fill = "white" if k == 0 else "solid"
            elif o.kind == "dxy":
                fill = "solid" if k % 2 == 0 else "white"
            elif o.kind == "hybrid":
                fill = "white" if k == 0 else "solid"
            else:
                fill = "solid" if k == 0 else "white"
        out.append((lobe, fill))
    return out


def orbital_primitives(o: Orbital, style: Style) -> list[tuple]:
    lw = style.line_width
    return [("shape", pts, lw, False, fill, True)
            for pts, fill in orbital_outline(o, style.bond_length)]


def bracket_primitives(x1, y1, x2, y2, kind: str, style: Style,
                       sides: str = "both") -> list[tuple]:
    """Bracket pair (or a single left / right bracket with ``sides``)."""
    lw = style.line_width
    h = y2 - y1
    tick = max(3.0, min(h * 0.12, style.bond_length * 0.3))
    out: list[tuple] = []
    ends = {"left": ((x1, 1),), "right": ((x2, -1),)}.get(sides, ((x1, 1), (x2, -1)))
    for x, s in ends:
        if kind == "round":
            pts = []
            for k in range(17):
                t = k / 16
                pts.append((x + s * tick * 1.2 * (1 - math.sin(math.pi * t)), y1 + h * t))
            out.append(("path", pts, lw))
        elif kind == "curly":
            m = (y1 + y2) / 2
            pts = [(x + s * tick, y1), (x + s * tick * 0.4, y1 + tick * 0.3),
                   (x + s * tick * 0.4, m - tick * 0.6), (x, m),
                   (x + s * tick * 0.4, m + tick * 0.6), (x + s * tick * 0.4, y2 - tick * 0.3),
                   (x + s * tick, y2)]
            out.append(("path", pts, lw))
        else:
            out.append(("path", [(x + s * tick, y1), (x, y1), (x, y2), (x + s * tick, y2)],
                        lw))
    return out

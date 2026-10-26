"""Editing operations behind the editor's tools (no Qt), so drawings can be built and
tested programmatically exactly as the mouse tools build them.

- bonds: click an atom (new bond at the zig-zag angle), drag between points/atoms, click a
  bond with a bond tool (single cycles 1 -> 2 -> 3, double moves the second line, wedge /
  hash set or reverse the stereo bond, ...);
- chains (zig-zag), ring templates (free, spiro on an atom, fused on a bond, benzene /
  cyclopentadiene with a Kekule pattern that fits existing double bonds);
- atoms: element, typed label / nickname, charge, radical, isotope, explicit H, map number.
"""
from __future__ import annotations

import math

from rsstatchem.core.chem_draw import geometry as geo
from rsstatchem.core.chem_draw.labels import bond_sum
from rsstatchem.core.chem_draw.model import Document

__all__ = ["BOND_TOOLS", "RING_TEMPLATES", "add_bond_at_atom", "add_bond_free",
           "add_bond_between", "apply_bond_tool", "add_chain", "stamp_ring", "set_element",
           "change_charge", "cycle_radical", "toggle_map_number", "atom_or_new", "snap_tol",
           "CHAIR_TEMPLATES", "stamp_chair"]

# tool name -> (order, style)
BOND_TOOLS = {
    "single": (1, ""), "double": (2, ""), "triple": (3, ""), "wedge": (1, "wedge"),
    "hash": (1, "hash"), "wavy": (1, "wavy"), "bold": (1, "bold"), "dashed": (1, "dashed"),
    "aromatic": (1.5, ""), "hollow": (1, "hollow"), "hashed": (1, "hashed"),
    "dative": (1, "dative"),
}
# name -> (ring size, edge-order pattern or None)
RING_TEMPLATES = {
    "cyclopropane": (3, None), "cyclobutane": (4, None), "cyclopentane": (5, None),
    "cyclohexane": (6, None), "cycloheptane": (7, None), "cyclooctane": (8, None),
    "benzene": (6, (2, 1, 2, 1, 2, 1)), "cyclopentadiene": (5, (2, 1, 2, 1, 1)),
}
_MAXVAL = {"C": 4, "N": 3, "O": 2, "S": 6, "P": 5, "B": 3, "Si": 4}


def snap_tol(doc: Document) -> float:
    return doc.style.bond_length * 0.3


def atom_or_new(doc: Document, x: float, y: float, exclude=()) -> int:
    """Existing atom within the snap tolerance of (x, y), or a new carbon there."""
    a = doc.atom_at(x, y, snap_tol(doc), exclude=exclude)
    return a.id if a is not None else doc.add_atom(x, y).id


def add_bond_at_atom(doc: Document, aid: int, order: float = 1, style: str = "",
                     angle: float | None = None) -> tuple[int, int]:
    """New bond from atom ``aid`` at the ChemDraw angle; returns (new atom, bond id)."""
    a = doc.atoms[aid]
    ang = geo.new_bond_angle(doc, aid) if angle is None else angle
    x, y = geo.bond_end_point(a.x, a.y, ang, doc.style.bond_length)
    other = atom_or_new(doc, x, y, exclude={aid})
    if doc.bond_between(aid, other) is not None:          # avoid stacking on a neighbour
        other = doc.add_atom(x, y).id
    b = doc.add_bond(aid, other, order, style)
    return other, b.id


def add_bond_free(doc: Document, x: float, y: float, order: float = 1, style: str = "",
                  angle: float = -geo.D30) -> tuple[int, int, int]:
    """Bond drawn by clicking empty space: begins at (x, y)."""
    a = atom_or_new(doc, x, y)
    other, bid = add_bond_at_atom(doc, a, order, style,
                                  angle if not doc.neighbors(a) else None)
    return a, other, bid


def add_bond_between(doc: Document, start, end, order: float = 1, style: str = "") -> int:
    """Bond between two atoms or points ``(x, y)`` (points snap onto existing atoms)."""
    a = start if isinstance(start, int) else atom_or_new(doc, *start)
    b = end if isinstance(end, int) else atom_or_new(doc, *end, exclude={a})
    old = doc.bond_between(a, b)
    if old is not None:
        old.order, old.style = order, style
        return old.id
    return doc.add_bond(a, b, order, style).id


def apply_bond_tool(doc: Document, bid: int, tool: str, near: int | None = None) -> None:
    """Click on an existing bond with a bond tool."""
    b = doc.bonds[bid]
    order, style = BOND_TOOLS[tool]
    if tool == "single":
        if b.style:
            b.style = ""
            b.order = 1
        else:
            b.order = {1: 2, 2: 3, 3: 1}.get(b.order, 1)
        b.double_pos = "auto"
        return
    if tool == "double":
        if b.order == 2 and not b.style:
            seq = ["auto", "center", "left", "right"]
            b.double_pos = seq[(seq.index(b.double_pos) + 1) % 4] \
                if b.double_pos in seq else "auto"
        else:
            b.order, b.style, b.double_pos = 2, "", "auto"
        return
    if tool in ("wedge", "hash", "hollow", "dative"):
        narrow = near if near in (b.a, b.b) else b.a
        if b.style == style and b.order == 1:
            b.a, b.b = b.b, b.a                     # second click reverses the wedge
        else:
            b.order, b.style = 1, style
            if b.a != narrow:
                b.a, b.b = b.b, b.a
        return
    b.order, b.style = order, style
    if order != 2:
        b.double_pos = "auto"


def add_chain(doc: Document, start, direction: float, n: int,
              first_sign: int | None = None) -> list[int]:
    """Zig-zag chain of ``n`` bonds from an atom id or point, along ``direction``."""
    a = start if isinstance(start, int) else atom_or_new(doc, *start)
    atom = doc.atoms[a]
    bl = doc.style.bond_length
    if first_sign is None:
        first_sign = 1
        nbrs = doc.neighbors(a)
        if nbrs:
            o = doc.atoms[nbrs[0]]
            back = math.atan2(o.y - atom.y, o.x - atom.x)

            def gap(sign):
                first = direction + (-sign) * geo.D30
                return abs(math.remainder(first - back, 2 * math.pi))
            first_sign = max((1, -1), key=gap)
    pts = geo.chain_points(atom.x, atom.y, direction, n, bl, first_sign)
    ids = [a]
    for x, y in pts:
        nid = atom_or_new(doc, x, y, exclude=set(ids))
        doc.add_bond(ids[-1], nid, 1)
        ids.append(nid)
    return ids


def _pattern_score(doc: Document, verts: list[int], pattern) -> float:
    n = len(verts)
    score = 0.0
    for k in range(n):
        if pattern[k] != 2:
            continue
        i, j = verts[k], verts[(k + 1) % n]
        b = doc.bond_between(i, j)
        if b is not None and b.order == 2:
            score += 2
            continue
        for v in (i, j):
            atom = doc.atoms.get(v)
            if atom is None:
                continue
            extra = [x for x in doc.bonds_of(v) if x.order >= 2 and
                     {x.a, x.b} != {i, j}]
            if extra or bond_sum(doc, v) + 1 > _MAXVAL.get(atom.symbol, 4):
                score -= 3
    return score


def stamp_ring(doc: Document, template: str, atom: int | None = None,
               bond: int | None = None, point: tuple[float, float] | None = None,
               pointer: tuple[float, float] | None = None) -> list[int]:
    """Place a ring template free at ``point``, on ``atom`` (sharing it) or fused on
    ``bond``. Returns the ring atom ids in order."""
    n, pattern = RING_TEMPLATES[template]
    bl = doc.style.bond_length
    if bond is not None:
        b = doc.bonds[bond]
        side = geo.ring_side_for_bond(doc, b, *(pointer or (None, None)))
        p, q = doc.atoms[b.a], doc.atoms[b.b]
        pos = geo.ring_on_bond(p.x, p.y, q.x, q.y, n, side)
        fixed = {0: b.a, 1: b.b}
    elif atom is not None:
        a = doc.atoms[atom]
        pos = geo.ring_on_atom(a.x, a.y, n, geo.away_direction(doc, atom), bl)
        fixed = {0: atom}
    else:
        x, y = point if point is not None else (0.0, 0.0)
        pos = geo.free_ring(x, y, n, bl)
        fixed = {}
    verts: list[int] = []
    for k, (x, y) in enumerate(pos):
        if k in fixed:
            verts.append(fixed[k])
        else:
            verts.append(atom_or_new(doc, x, y, exclude=set(verts)))
    if pattern is None:
        orders = [1] * n
    else:
        best = max(range(n), key=lambda r: (_pattern_score(
            doc, verts, [pattern[(k + r) % n] for k in range(n)]), -r))
        orders = [pattern[(k + best) % n] for k in range(n)]
    for k in range(n):
        i, j = verts[k], verts[(k + 1) % n]
        old = doc.bond_between(i, j)
        if old is None:
            doc.add_bond(i, j, orders[k])
        elif orders[k] == 2 and old.order == 1 and not old.style:
            ok = all(not [x for x in doc.bonds_of(v) if x.order >= 2]
                     for v in (i, j))
            if ok:
                old.order = 2
    return verts


# Cyclohexane chair projections (ChemDraw's two chair tools), in bond lengths, y down: a 3D
# chair (puckering +-0.25) seen 20 deg above its mean plane, so opposite bonds are parallel.
_CHAIR = [(1.15 * x, 1.15 * y) for x, y in ((1.0, -0.235), (0.5, -0.059), (-0.5, -0.529),
                                            (-1.0, 0.235), (-0.5, 0.059), (0.5, 0.529))]
CHAIR_TEMPLATES = {"chair1": _CHAIR, "chair2": [(-x, y) for x, y in _CHAIR]}


def stamp_chair(doc: Document, variant: str, point: tuple[float, float]) -> list[int]:
    """Cyclohexane chair centred on ``point``; returns the ring atom ids in order."""
    bl = doc.style.bond_length
    pts = CHAIR_TEMPLATES[variant]
    cx = sum(x for x, _ in pts) / len(pts)
    cy = sum(y for _, y in pts) / len(pts)
    verts: list[int] = []
    for x, y in pts:
        verts.append(atom_or_new(doc, point[0] + (x - cx) * bl, point[1] + (y - cy) * bl,
                                 exclude=set(verts)))
    for k in range(len(verts)):
        doc.add_bond(verts[k], verts[(k + 1) % len(verts)])
    return verts


def set_element(doc: Document, aid: int, symbol: str) -> None:
    a = doc.atoms[aid]
    a.symbol, a.label, a.explicit_h, a.show_carbon = symbol, "", None, False


def change_charge(doc: Document, aid: int, delta: int) -> None:
    doc.atoms[aid].charge += delta


def cycle_radical(doc: Document, aid: int) -> None:
    a = doc.atoms[aid]
    a.radicals = (a.radicals + 1) % 3


def toggle_map_number(doc: Document, aid: int) -> None:
    a = doc.atoms[aid]
    a.map_number = 0 if a.map_number else doc.next_map_number()

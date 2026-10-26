"""ChemDraw's *Mass Fragmentation* tool (no Qt): a line drawn across bonds of a structure
cleaves them and labels each piece with its formula and monoisotopic mass.

Each fragment keeps the hydrogens its atoms carry in the intact molecule (homolytic cleavage:
the pieces are radicals, as in ChemDraw's tool), so the fragment masses add up to the
molecule's exact mass. Masses use the most abundant isotopes (``core.chem.formula``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction

from rsstatchem.core.chem import formula as fm
from rsstatchem.core.chem_draw.model import Document

__all__ = ["Fragment", "crossed_bonds", "cleave", "add_fragmentation"]


@dataclass
class Fragment:
    atoms: list = field(default_factory=list)      # document atom ids
    formula: str = ""                              # Hill formula
    mass: float = 0.0                              # monoisotopic mass
    side: int = 0                                  # +1 / -1: side of the cleavage line


def _cross(ax, ay, bx, by, cx, cy, dx, dy) -> bool:
    """Do the segments a-b and c-d properly intersect?"""
    def orient(px, py, qx, qy, rx, ry):
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)
    o1 = orient(ax, ay, bx, by, cx, cy)
    o2 = orient(ax, ay, bx, by, dx, dy)
    o3 = orient(cx, cy, dx, dy, ax, ay)
    o4 = orient(cx, cy, dx, dy, bx, by)
    return o1 * o2 < 0 and o3 * o4 < 0


def crossed_bonds(doc: Document, x1, y1, x2, y2) -> list[int]:
    out = []
    for b in doc.bonds.values():
        p, q = doc.atoms[b.a], doc.atoms[b.b]
        if _cross(x1, y1, x2, y2, p.x, p.y, q.x, q.y):
            out.append(b.id)
    return out


def cleave(doc: Document, x1, y1, x2, y2) -> list[Fragment]:
    """Fragments of the molecule(s) cut by the line (x1, y1)-(x2, y2); [] when the line
    crosses no bond or does not split a molecule."""
    from rsstatchem.core.chem_draw import convert

    cut = set(crossed_bonds(doc, x1, y1, x2, y2))
    if not cut:
        return []
    atoms = set()
    for bid in cut:
        atoms.update(doc.fragment_of(doc.bonds[bid].a))
    mol = convert.to_mol(doc, sorted(atoms))
    # hydrogens and element of each document atom (nicknames contribute all their atoms)
    counts: dict[int, dict] = {}
    for a in mol.GetAtoms():
        if not a.HasProp("doc_id") or a.GetAtomicNum() == 0:
            continue
        d = counts.setdefault(a.GetIntProp("doc_id"), {})
        key = (a.GetSymbol(), a.GetIsotope() or None)
        d[key] = d.get(key, 0) + 1
        if a.GetTotalNumHs():
            d[("H", None)] = d.get(("H", None), 0) + a.GetTotalNumHs()
    adj: dict[int, list[int]] = {i: [] for i in atoms}
    for b in doc.bonds.values():
        if b.id in cut or b.a not in adj or b.b not in adj:
            continue
        adj[b.a].append(b.b)
        adj[b.b].append(b.a)
    seen: set[int] = set()
    pieces = []
    for start in sorted(atoms):
        if start in seen:
            continue
        comp, stack = [], [start]
        seen.add(start)
        while stack:
            i = stack.pop()
            comp.append(i)
            for j in adj[i]:
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
        pieces.append(sorted(comp))
    if len(pieces) < 2:
        return []
    dx, dy = x2 - x1, y2 - y1
    out = []
    for comp in pieces:
        total: dict = {}
        for aid in comp:
            for k, n in counts.get(aid, {}).items():
                total[k] = total.get(k, 0) + n
        f = fm.Formula({k: Fraction(v) for k, v in total.items()}, 0)
        cx = sum(doc.atoms[a].x for a in comp) / len(comp)
        cy = sum(doc.atoms[a].y for a in comp) / len(comp)
        side = 1 if dx * (cy - y1) - dy * (cx - x1) > 0 else -1
        out.append(Fragment(comp, f.hill(), f.monoisotopic_mass, side))
    out.sort(key=lambda fr: (fr.side, fr.atoms[0]))
    return out


def add_fragmentation(doc: Document, x1, y1, x2, y2, decimals: int = 4) -> list:
    """Draw the cleavage line (a dashed curve) and a caption "formula / mass" beside each
    fragment, as ChemDraw's tool does. Returns the new keys ([] if nothing was cleaved)."""
    frags = cleave(doc, x1, y1, x2, y2)
    if not frags:
        return []
    keys = []
    line = doc.add_curve([(x1, y1), (x1, y1), (x1, y1), (x2, y2), (x2, y2), (x2, y2)],
                         dashed=True)
    keys.append(("curve", line.id))
    size = doc.style.caption_size * 0.9
    L = math.hypot(x2 - x1, y2 - y1) or 1.0
    ux, uy = (x2 - x1) / L, (y2 - y1) / L
    nx, ny = -uy, ux
    # one end of the line per side (ChemDraw writes each fragment at an end of the line),
    # pushed out beyond the end and towards the fragment's side
    ends = {1: ((x1, y1), -1.0), -1: ((x2, y2), 1.0)}
    by_side: dict[int, list[Fragment]] = {}
    for fr in frags:
        by_side.setdefault(fr.side, []).append(fr)
    for side, items in by_side.items():
        (ex, ey), out = ends[side]
        for k, fr in enumerate(items):
            text = f"{fr.formula}\n{fr.mass:.{decimals}f}"
            w = max(len(fr.formula), len(f"{fr.mass:.{decimals}f}")) * size * 0.55
            h = size * 2.5
            reach = size * 0.6 + k * h * 1.1
            lateral = (abs(nx) * w + abs(ny) * h) / 2 + size * 0.3
            cx = ex + out * ux * reach + side * nx * lateral
            cy = ey + out * uy * reach + side * ny * lateral
            t = doc.add_text(cx - w / 2, cy - h / 2, text, chemical=True, size=size)
            keys.append(("text", t.id))
    return keys

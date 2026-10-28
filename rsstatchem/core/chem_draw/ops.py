"""Editing operations behind the editor's tools (no Qt), so drawings can be built and
tested programmatically exactly as the mouse tools build them.

- bonds: click an atom (new bond at the zig-zag angle), drag between points/atoms, click a
  bond with a bond tool (single cycles 1 -> 2 -> 3, double moves the second line, wedge /
  hash set or reverse the stereo bond, ...);
- chains (zig-zag), ring templates (free, spiro on an atom, fused on a bond, benzene /
  cyclopentadiene with a Kekule pattern that fits existing double bonds);
- atoms: element, typed label / nickname, charge, radical, isotope, explicit H, map number;
- ChemDraw 17.1 "enhanced hotkeys" (User Guide pp. 54-58): number keys sprout groups on an
  atom (1 linear bond, 0 cyclic bond, 2 carbonyl / acetyl, 3 or a phenyl, 4 wedge, 5 hashed
  wedge, 6 cyclohexyl, 7 cyclopentyl, 8 methylidene, 9 dimethyl, z alkyne) and fuse rings or
  chairs on a bond (a benzene, z cyclopentadiene, v cyclopropane, 4-8 rings, 9 / 0 chairs);
- stereo indicators: :func:`refresh_stereo` fills ``Atom.stereo`` / ``Bond.stereo``.
"""
from __future__ import annotations

import math

from rsstatchem.core.chem_draw import geometry as geo
from rsstatchem.core.chem_draw.labels import bond_sum
from rsstatchem.core.chem_draw.model import Document

__all__ = ["BOND_TOOLS", "RING_TEMPLATES", "add_bond_at_atom", "add_bond_free",
           "add_bond_between", "apply_bond_tool", "add_chain", "stamp_ring", "set_element",
           "change_charge", "cycle_radical", "toggle_map_number", "atom_or_new", "snap_tol",
           "CHAIR_TEMPLATES", "stamp_chair", "sprout", "SPROUT_KEYS", "fuse_on_bond",
           "FUSE_KEYS", "stamp_chair_on_bond", "refresh_stereo", "stereo_labels",
           "is_aromatic_atom", "heavy_degree"]

# tool name -> (order, style)
BOND_TOOLS = {
    "single": (1, ""), "double": (2, ""), "triple": (3, ""), "wedge": (1, "wedge"),
    "hash": (1, "hash"), "wavy": (1, "wavy"), "bold": (1, "bold"), "dashed": (1, "dashed"),
    "aromatic": (1.5, ""), "hollow": (1, "hollow"), "hashed": (1, "hashed"),
    "dative": (1, "dative"), "quadruple": (4, ""), "bolddouble": (2, "bold"),
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


# --- ChemDraw 17.1 enhanced hotkeys ---------------------------------------------------------

SPROUT_KEYS = ("0", "1", "2", "3", "a", "4", "5", "6", "7", "8", "9", "z")
FUSE_KEYS = {"a": "benzene", "z": "cyclopentadiene", "v": "cyclopropane",
             "4": "cyclobutane", "5": "cyclopentane", "6": "cyclohexane",
             "7": "cycloheptane", "8": "cyclooctane", "9": "chair1", "0": "chair2"}


def heavy_degree(doc: Document, aid: int) -> int:
    return len(doc.neighbors(aid))


def is_aromatic_atom(doc: Document, aid: int) -> bool:
    """Aromatic ring atom: aromatic bonds, or a 5/6-ring with alternating double bonds."""
    if any(b.order == 1.5 for b in doc.bonds_of(aid)):
        return True
    for ring in doc.rings(8):
        if aid not in ring or len(ring) not in (5, 6):
            continue
        doubles = sum(1 for k in range(len(ring))
                      if (b := doc.bond_between(ring[k], ring[(k + 1) % len(ring)]))
                      is not None and b.order == 2)
        if doubles >= len(ring) // 2:
            return True
    return False


def _cyclic_angle(doc: Document, aid: int) -> float:
    """Bond direction of the "cyclic mode" sprout (0): the zig-zag turns back (cis)."""
    a = doc.atoms[aid]
    nbrs = doc.neighbors(aid)
    if len(nbrs) != 1:
        return geo.new_bond_angle(doc, aid)
    n = doc.atoms[nbrs[0]]
    base = math.atan2(n.y - a.y, n.x - a.x)
    trans = geo.new_bond_angle(doc, aid)
    cands = [base + math.radians(120), base - math.radians(120)]
    return max(cands, key=lambda c: abs(math.remainder(c - trans, 2 * math.pi)))


def _vertical_angle(doc: Document, aid: int) -> float:
    a = doc.atoms[aid]
    angs = [math.atan2(doc.atoms[n].y - a.y, doc.atoms[n].x - a.x)
            for n in doc.neighbors(aid)]

    def gap(c):
        return min((abs(math.remainder(c - t, 2 * math.pi)) for t in angs), default=math.pi)
    return max((-math.pi / 2, math.pi / 2), key=gap)


def _methyl_pair(doc: Document, aid: int) -> list[int]:
    """Two methyls on ``aid`` at the angles of a gem-dimethyl."""
    out = []
    for _ in range(2):
        other, _b = add_bond_at_atom(doc, aid)
        out.append(other)
    return out


def sprout(doc: Document, aid: int, key: str) -> list[int]:
    """Atom hotkey ``key`` of ChemDraw 17.1 on atom ``aid`` (pp. 55-57); returns the new
    atom ids. Primary / secondary / tertiary refers to the number of bonded neighbours;
    aromatic and tertiary hotspots first get a C-C bond, as described in the guide."""
    before = set(doc.atoms)
    deg = heavy_degree(doc, aid)
    arom = is_aromatic_atom(doc, aid)
    carbon = doc.atoms[aid].symbol == "C" and not doc.atoms[aid].label
    if key == "1":
        add_bond_at_atom(doc, aid)
    elif key == "0":
        add_bond_at_atom(doc, aid, angle=_cyclic_angle(doc, aid))
    elif key == "2":                            # carbonyl / acetyl
        if carbon and deg == 2 and not arom:
            add_bond_at_atom(doc, aid, 2)                   # C=O on the hotspot
            _last_to(doc, before, "O")
        else:
            c = aid
            if not (carbon and deg <= 1):
                c, _b = add_bond_at_atom(doc, aid)         # C-C bond, then the acetyl
            o, _b = add_bond_at_atom(doc, c, 2)
            doc.atoms[o].symbol = "O"
            add_bond_at_atom(doc, c)
    elif key in ("3", "a"):                     # phenyl
        c = aid
        if not (carbon and deg <= 1) or arom:
            c, _b = add_bond_at_atom(doc, aid)
        if c == aid and deg == 0:
            stamp_ring(doc, "benzene", atom=aid)
        elif c == aid:
            stamp_ring(doc, "benzene", atom=aid)
        else:
            stamp_ring(doc, "benzene", atom=c)
    elif key in ("4", "5"):                     # wedged / hashed wedged bond (+ methyl)
        style = "wedge" if key == "4" else "hash"
        c = aid
        if arom:
            c, _b = add_bond_at_atom(doc, aid)
        angle = _vertical_angle(doc, c) if heavy_degree(doc, c) >= 3 else None
        add_bond_at_atom(doc, c, 1, style, angle=angle)
    elif key in ("6", "7"):                     # cyclohexane / cyclopentane
        tpl = "cyclohexane" if key == "6" else "cyclopentane"
        c = aid
        if deg >= 3 or arom or not carbon:
            c, _b = add_bond_at_atom(doc, aid)
        stamp_ring(doc, tpl, atom=c)
    elif key == "8":                            # methylidene
        add_bond_at_atom(doc, aid, 2)
    elif key == "9":                            # dimethyl / gem-dimethyl / isopropyl
        c = aid
        if deg >= 3 or not carbon:
            c, _b = add_bond_at_atom(doc, aid)
        _methyl_pair(doc, c)
    elif key == "z":                            # alkyne with linear geometry
        c1, _b = add_bond_at_atom(doc, aid)
        a, b1 = doc.atoms[aid], doc.atoms[c1]
        ang = math.atan2(b1.y - a.y, b1.x - a.x)
        add_bond_at_atom(doc, c1, 3, angle=ang)
    return sorted(set(doc.atoms) - before)


def _last_to(doc: Document, before: set, symbol: str) -> None:
    new = sorted(set(doc.atoms) - before)
    if new:
        doc.atoms[new[-1]].symbol = symbol


def stamp_chair_on_bond(doc: Document, bid: int, variant: str) -> list[int]:
    """Cyclohexane chair sharing bond ``bid`` (bond hotkeys 9 and 0), on the free side."""
    b = doc.bonds[bid]
    A, B = doc.atoms[b.a], doc.atoms[b.b]
    bl = doc.style.bond_length
    tpl = [(x * bl, y * bl) for x, y in CHAIR_TEMPLATES[variant]]
    (px, py), (qx, qy) = tpl[0], tpl[1]
    vt = (qx - px, qy - py)
    vb = (B.x - A.x, B.y - A.y)
    lt, lb = math.hypot(*vt) or 1.0, math.hypot(*vb) or 1.0
    rot = math.atan2(vb[1], vb[0]) - math.atan2(vt[1], vt[0])
    k = lb / lt
    c, s = math.cos(rot) * k, math.sin(rot) * k
    pts = [(A.x + c * (x - px) - s * (y - py), A.y + s * (x - px) + c * (y - py))
           for x, y in tpl]
    want = geo.ring_side_for_bond(doc, b)
    nx, ny = -vb[1] / lb, vb[0] / lb
    cx = sum(x for x, _ in pts) / 6
    cy = sum(y for _, y in pts) / 6
    side = 1 if (cx - A.x) * nx + (cy - A.y) * ny > 0 else -1
    if side != want:                                # mirror across the bond
        out = []
        for x, y in pts:
            d = (x - A.x) * nx + (y - A.y) * ny
            out.append((x - 2 * d * nx, y - 2 * d * ny))
        pts = out
    verts = [b.a, b.b]
    for x, y in pts[2:]:
        verts.append(atom_or_new(doc, x, y, exclude=set(verts)))
    for i in range(6):
        doc.add_bond(verts[i], verts[(i + 1) % 6])
    return verts


def fuse_on_bond(doc: Document, bid: int, key: str) -> list[int]:
    """Bond hotkey ring fusion (a, z, v, 4-8) or chair (9, 0)."""
    tpl = FUSE_KEYS[key]
    if tpl in CHAIR_TEMPLATES:
        return stamp_chair_on_bond(doc, bid, tpl)
    return stamp_ring(doc, tpl, bond=bid)


# --- stereo indicators (Object > Show Stereochemistry) ----------------------------------

def stereo_labels(doc: Document) -> dict:
    """{("atom", id): "R"/"S"/"r"/"s"/"M"/"P", ("bond", id): "E"/"Z"}: WP-A's
    :func:`rsstatchem.core.chem_draw.stereo.labels` when available, else RDKit's new CIP
    labeller on each molecule of the drawing."""
    try:
        from rsstatchem.core.chem_draw import stereo        # chemistry-intelligence module
        return dict(stereo.labels(doc))
    except Exception:  # noqa: BLE001 - module missing or failed: open fallback below
        pass
    from rdkit.Chem import rdCIPLabeler

    from rsstatchem.core.chem_draw import convert
    out: dict = {}
    for comp, mol in convert.fragment_mols(doc):
        if isinstance(mol, Exception):
            continue
        try:
            rdCIPLabeler.AssignCIPLabels(mol)
        except Exception:  # noqa: BLE001
            continue
        origin: dict[int, list[int]] = {}
        for at in mol.GetAtoms():
            if at.HasProp("doc_id"):
                origin.setdefault(at.GetIntProp("doc_id"), []).append(at.GetIdx())
        idx_doc = {v[0]: k for k, v in origin.items() if len(v) == 1}
        for at in mol.GetAtoms():
            if at.HasProp("_CIPCode") and at.GetIdx() in idx_doc:
                out[("atom", idx_doc[at.GetIdx()])] = at.GetProp("_CIPCode")
        for bd in mol.GetBonds():
            if not bd.HasProp("_CIPCode"):
                continue
            a, b = idx_doc.get(bd.GetBeginAtomIdx()), idx_doc.get(bd.GetEndAtomIdx())
            db = doc.bond_between(a, b) if a is not None and b is not None else None
            if db is not None:
                out[("bond", db.id)] = bd.GetProp("_CIPCode")
    return out


def refresh_stereo(doc: Document) -> None:
    """Store the current stereo labels in ``Atom.stereo`` / ``Bond.stereo``."""
    try:
        labels = stereo_labels(doc)
    except Exception:  # noqa: BLE001 - indicators must never break editing
        labels = {}
    for a in doc.atoms.values():
        a.stereo = str(labels.get(("atom", a.id), ""))
    for b in doc.bonds.values():
        b.stereo = str(labels.get(("bond", b.id), ""))

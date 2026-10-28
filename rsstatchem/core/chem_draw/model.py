"""Drawing document of the structure / reaction editor (the ChemDraw document model).

A :class:`Document` holds atoms and bonds (any number of molecules), reaction arrows (also
curved electron-pushing arrows), "+" signs, free text boxes, brackets, Bezier curves (the Pen
tool, ChemDraw's Curves menu), orbitals, drawing elements (circles, ellipses, boxes, arcs),
chemical symbols (lone pairs, delta charges, daggers... optionally attached to an atom), TLC
plates, gel electrophoresis plates and clipware (glassware). Coordinates are in
points with y pointing down (screen / scene convention). Every object has an integer id that
is unique within the document; objects are addressed by keys ``(kind, id)`` with kind in
:data:`KINDS`.

The document serialises to plain JSON (``.rsc`` files, :func:`save_rsc` / :func:`load_rsc`)
and back without loss, which the editor also uses for undo snapshots and the clipboard.
"""
from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass, field, fields

from rsstatchem.core.chem_draw.style import Style

__all__ = ["Atom", "Bond", "Arrow", "Plus", "TextBox", "Bracket", "Curve", "Orbital",
           "Shape", "Symbol", "TLCPlate", "GelPlate", "Clipart",
           "Document", "KINDS", "FORMAT", "VERSION", "save_rsc", "load_rsc", "ARROW_KINDS",
           "BOND_STYLES", "ORBITAL_KINDS", "ORBITAL_FILLS", "OBJECT_KINDS", "SHAPE_KINDS",
           "SYMBOL_KINDS", "BRACKET_KINDS", "POLYMER_TYPES", "REPEAT_PATTERNS",
           "ANCHORED_ARROWS"]

FORMAT = "rs-statchem-drawing"
VERSION = 1
KINDS = ("atom", "bond", "arrow", "plus", "text", "bracket", "curve", "orbital", "shape",
         "symbol", "tlc", "gel", "clipart")
OBJECT_KINDS = KINDS[2:]                                                 # not atoms/bonds
# nogo = crossed ("no reaction"), unbalanced = unbalanced equilibrium, block = hollow /
# filled block arrow, double = double-headed (bendable), arc = circular arc (sweep 90, 180,
# 270 deg), dipole = dipole arrow (crossed tail)
ARROW_KINDS = ("forward", "equilibrium", "retro", "resonance", "curved", "fishhook", "line",
               "nogo", "unbalanced", "block", "double", "arc", "dipole")
REACTION_ARROWS = ("forward", "equilibrium", "retro", "resonance", "nogo", "unbalanced",
                   "block")
BENT_ARROWS = ("curved", "fishhook", "double")          # quadratic shaft with a bend
ANCHORED_ARROWS = ("curved", "fishhook", "double")      # mechanism arrows (anchor to atoms)
SHAPE_KINDS = ("circle", "ellipse", "rect", "roundrect", "shadowrect", "arc")
# chemical symbols (ChemDraw's Chemical Symbols palette)
SYMBOL_KINDS = ("lonepair", "electron", "plus", "minus", "circleplus", "circleminus",
                "radicalcation", "radicalanion", "deltaplus", "deltaminus", "dagger",
                "ddagger")
BRACKET_KINDS = ("square", "round", "curly")
# Bracket Properties (polymer Sgroups): structural repeating unit and copolymer types
POLYMER_TYPES = ("", "SRU", "copolymer", "alternating", "random", "block", "graft", "mer",
                 "monomer", "crosslink", "mixture", "component")
REPEAT_PATTERNS = ("", "ht", "hh", "eu")        # head-to-tail, head-to-head, either/unknown
# hollow = hollow wedged, hashed = plain (non-wedged) hashed bond, dative = arrow a -> b
BOND_STYLES = ("", "wedge", "hash", "wavy", "bold", "dashed", "either", "hollow", "hashed",
               "dative")
ORBITAL_KINDS = ("s", "p", "lobe", "hybrid", "dxy", "dz2")
ORBITAL_FILLS = ("plain", "shaded", "filled", "phase")


@dataclass
class Atom:
    id: int
    x: float
    y: float
    symbol: str = "C"
    charge: int = 0
    isotope: int = 0
    explicit_h: int | None = None   # None = implicit hydrogens from the valence
    radicals: int = 0
    map_number: int = 0
    label: str = ""                 # nickname or generic label (Ph, OMe, R1...)
    show_carbon: bool = False       # force the "C" label of a carbon
    group: int = 0
    color: str = ""                 # "" = automatic (black / colour by element), else #RRGGBB
    number: int = 0                 # displayed atom number (Structure > Add Atom Numbers)
    stereo: str = ""                # CIP label shown by Show Stereochemistry (R, S, r, s...)
    show_stereo: bool = True        # per-atom: hide its stereo indicator when False
    charge_circle: bool = False     # draw the charge as a circled +/- (Chemical Symbols)
    attachment: int = 0             # attachment point number (hotkey "."), 0 = none


@dataclass
class Bond:
    id: int
    a: int                          # begin atom (narrow end of wedges)
    b: int
    order: float = 1                # 1, 2, 3 or 1.5 (aromatic / delocalised)
    style: str = ""                 # see BOND_STYLES
    double_pos: str = "auto"        # auto | center | left | right (seen from a to b)
    group: int = 0
    color: str = ""
    stereo: str = ""                # E / Z label shown by Show Stereochemistry


@dataclass
class Arrow:
    id: int
    x1: float
    y1: float
    x2: float                       # head
    y2: float
    kind: str = "forward"           # see ARROW_KINDS
    bend: float = 0.0               # curved arrows: sagitta (pt, + = left of a->b)
    above: str = ""                 # reaction conditions written above / below the arrow
    below: str = ""
    z: float = 0.0
    group: int = 0
    color: str = ""
    dashed: bool = False
    bold: bool = False
    fill: str = ""                  # block arrows: "" (hollow) | "filled" | "shaded"
    sweep: float = 180.0            # arc arrows: angle of the arc in degrees
    tail_ref: str = ""              # mechanism arrows: "atom:12", "bond:5", "symbol:7"
    head_ref: str = ""
    tail_dx: float = 0.0            # offsets of the end points from their anchors
    tail_dy: float = 0.0
    head_dx: float = 0.0
    head_dy: float = 0.0


@dataclass
class Plus:
    id: int
    x: float
    y: float
    z: float = 0.0
    group: int = 0
    color: str = ""


@dataclass
class TextBox:
    id: int
    x: float                        # top-left corner of the text
    y: float
    text: str = ""
    chemical: bool = True           # subscripts in formulas, superscript charges
    size: float = 0.0               # pt; 0 = the style's caption size
    bold: bool = False
    z: float = 0.0
    group: int = 0
    color: str = ""
    italic: bool = False
    underline: bool = False
    font: str = ""                  # "" = the style's font family
    autonumber: int = 0             # compound number written by Autonumber Reaction


@dataclass
class Bracket:
    id: int
    x1: float
    y1: float
    x2: float
    y2: float
    kind: str = "square"            # square | round | curly
    label: str = ""                 # subscript at the lower right (n, 2, ‡ ...)
    z: float = 0.0
    group: int = 0
    color: str = ""
    sides: str = "both"             # both | left | right (single brackets)
    polymer: str = ""               # see POLYMER_TYPES (SRU, copolymer...)
    repeat: str = ""                # see REPEAT_PATTERNS (head-to-tail...)


@dataclass
class Curve:
    """Cubic Bezier curve drawn with the Pen tool. ``points`` holds three points per anchor
    in ChemDraw's CurvePoints order: [control in, anchor, control out, control in, anchor,
    ...] (the first "in" and last "out" handles of an open curve are unused)."""
    id: int
    points: list = field(default_factory=list)     # [[x, y], ...], 3 per anchor
    closed: bool = False
    dashed: bool = False
    bold: bool = False
    doubled: bool = False
    filled: bool = False
    shaded: bool = False
    arrow_start: str = ""           # "" | "full" | "half"
    arrow_end: str = ""
    z: float = 0.0
    group: int = 0
    color: str = ""
    faded: bool = False             # Curves > Faded: fill fading from the colour to white

    def anchors(self) -> list[list[float]]:
        return [self.points[k] for k in range(1, len(self.points), 3)]


@dataclass
class Orbital:
    """Atomic / hybrid orbital centred on (x, y); ``angle`` (radians, screen y down) is the
    direction of the main lobe."""
    id: int
    x: float
    y: float
    kind: str = "p"                 # see ORBITAL_KINDS
    fill: str = "shaded"            # see ORBITAL_FILLS (phase = alternate filled lobes)
    size: float = 0.0               # lobe length in pt; 0 = the style's bond length
    angle: float = -math.pi / 2     # up
    z: float = 0.0
    group: int = 0
    color: str = ""


@dataclass
class Shape:
    """Drawing element (ChemDraw's Drawing Elements palette) in the box (x1, y1)-(x2, y2):
    circle / ellipse (inscribed), rectangle, rounded or shadowed rectangle, or an elliptical
    arc of ``sweep`` degrees starting at ``start`` (0 = right, clockwise on screen)."""
    id: int
    x1: float
    y1: float
    x2: float
    y2: float
    kind: str = "rect"              # see SHAPE_KINDS
    sweep: float = 180.0
    start: float = 180.0
    dashed: bool = False
    bold: bool = False
    filled: bool = False
    shaded: bool = False
    faded: bool = False
    z: float = 0.0
    group: int = 0
    color: str = ""


@dataclass
class Symbol:
    """Chemical symbol (lone pair, electron, circled charges, radical ions, delta charges,
    daggers). ``atom`` (id) associates it with an atom: it moves and is deleted with it.
    ``angle`` orients lone pairs (direction from the atom, screen y down)."""
    id: int
    x: float
    y: float
    kind: str = "lonepair"          # see SYMBOL_KINDS
    atom: int = 0
    angle: float = -math.pi / 2
    z: float = 0.0
    group: int = 0
    color: str = ""


@dataclass
class TLCPlate:
    """Thin layer chromatography plate. ``origin`` and ``front`` are the heights of the
    origin line and the solvent front as fractions of the plate height (from the bottom).
    ``lanes`` holds one list of spots per lane; a spot is a dict with ``rf``, ``w`` / ``h``
    (size in pt, 0 = default), ``tail`` (crescent / smear length in pt), ``style``
    ("filled", "shaded", "hollow"), ``color`` and ``show_rf``."""
    id: int
    x1: float
    y1: float
    x2: float
    y2: float
    lanes: list = field(default_factory=list)
    origin: float = 0.12
    front: float = 0.9
    show_origin: bool = True
    show_front: bool = True
    show_border: bool = True
    show_ticks: bool = True
    show_rf: bool = False
    transparent: bool = False
    z: float = 0.0
    group: int = 0
    color: str = ""


@dataclass
class GelPlate:
    """Gel electrophoresis plate: lanes (dicts with ``label`` and ``bands``; a band is a
    dict with ``mass`` (kDa), ``style`` ("filled", "hollow", "dashed", "doubled"),
    ``show`` (size label) and ``color``) on a logarithmic mass axis from ``lo`` to ``hi``
    (highest mass at the top)."""
    id: int
    x1: float
    y1: float
    x2: float
    y2: float
    lanes: list = field(default_factory=list)
    lo: float = 10.0
    hi: float = 1000.0
    units: str = "kDa"              # kDa | log | distance (scale labels)
    show_scale: bool = True
    show_labels: bool = True
    z: float = 0.0
    group: int = 0
    color: str = ""


@dataclass
class Clipart:
    """Clipware (laboratory glassware) drawn in the box (x, y, w, h); ``fill`` is the liquid
    level (0 = empty, 1 = full)."""
    id: int
    x: float
    y: float
    w: float = 60.0
    h: float = 80.0
    name: str = "beaker"
    fill: float = 0.0
    z: float = 0.0
    group: int = 0
    color: str = ""


_CLASSES = {"atom": Atom, "bond": Bond, "arrow": Arrow, "plus": Plus, "text": TextBox,
            "bracket": Bracket, "curve": Curve, "orbital": Orbital, "shape": Shape,
            "symbol": Symbol, "tlc": TLCPlate, "gel": GelPlate, "clipart": Clipart}
_STORES = {"atom": "atoms", "bond": "bonds", "arrow": "arrows", "plus": "pluses",
           "text": "texts", "bracket": "brackets", "curve": "curves", "orbital": "orbitals",
           "shape": "shapes", "symbol": "symbols", "tlc": "tlcs", "gel": "gels",
           "clipart": "cliparts"}
_BOXED = ("arrow", "bracket", "shape", "tlc", "gel")      # objects with x1, y1, x2, y2


def _from_dict(cls, data: dict):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


class Document:
    """All objects of one drawing plus its style."""

    def __init__(self, style: Style | None = None):
        self.style = style or Style()
        self.atoms: dict[int, Atom] = {}
        self.bonds: dict[int, Bond] = {}
        self.arrows: dict[int, Arrow] = {}
        self.pluses: dict[int, Plus] = {}
        self.texts: dict[int, TextBox] = {}
        self.brackets: dict[int, Bracket] = {}
        self.curves: dict[int, Curve] = {}
        self.orbitals: dict[int, Orbital] = {}
        self.shapes: dict[int, Shape] = {}
        self.symbols: dict[int, Symbol] = {}
        self.tlcs: dict[int, TLCPlate] = {}
        self.gels: dict[int, GelPlate] = {}
        self.cliparts: dict[int, Clipart] = {}
        self.meta: dict = {}
        self._next = 1

    # --- generic access ----------------------------------------------------------------
    def store(self, kind: str) -> dict:
        return getattr(self, _STORES[kind])

    def get(self, key: tuple[str, int]):
        return self.store(key[0]).get(key[1])

    def keys(self, kinds=KINDS) -> list[tuple[str, int]]:
        return [(k, i) for k in kinds for i in self.store(k)]

    def new_id(self) -> int:
        i = self._next
        self._next += 1
        return i

    def is_empty(self) -> bool:
        return not any(self.store(k) for k in KINDS)

    def clear(self) -> None:
        for k in KINDS:
            self.store(k).clear()

    # --- creation ----------------------------------------------------------------------
    def add_atom(self, x: float, y: float, symbol: str = "C", **kw) -> Atom:
        a = Atom(self.new_id(), float(x), float(y), symbol, **kw)
        self.atoms[a.id] = a
        return a

    def add_bond(self, a: int, b: int, order: float = 1, style: str = "", **kw) -> Bond:
        if a == b:
            raise ValueError("A bond needs two different atoms")
        old = self.bond_between(a, b)
        if old is not None:
            return old
        bond = Bond(self.new_id(), a, b, order, style, **kw)
        self.bonds[bond.id] = bond
        return bond

    def add_arrow(self, x1, y1, x2, y2, kind: str = "forward", **kw) -> Arrow:
        ar = Arrow(self.new_id(), float(x1), float(y1), float(x2), float(y2), kind, **kw)
        self.arrows[ar.id] = ar
        return ar

    def add_plus(self, x: float, y: float) -> Plus:
        p = Plus(self.new_id(), float(x), float(y))
        self.pluses[p.id] = p
        return p

    def add_text(self, x: float, y: float, text: str, **kw) -> TextBox:
        t = TextBox(self.new_id(), float(x), float(y), text, **kw)
        self.texts[t.id] = t
        return t

    def add_bracket(self, x1, y1, x2, y2, kind: str = "square", label: str = "",
                    **kw) -> Bracket:
        b = Bracket(self.new_id(), min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), kind,
                    label, **kw)
        self.brackets[b.id] = b
        return b

    def add_shape(self, x1, y1, x2, y2, kind: str = "rect", **kw) -> Shape:
        s = Shape(self.new_id(), min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), kind,
                  **kw)
        self.shapes[s.id] = s
        return s

    def add_symbol(self, x: float, y: float, kind: str = "lonepair", **kw) -> Symbol:
        s = Symbol(self.new_id(), float(x), float(y), kind, **kw)
        self.symbols[s.id] = s
        return s

    def add_tlc(self, x1, y1, x2, y2, lanes: int | None = None, **kw) -> TLCPlate:
        """TLC plate; without ``lanes`` the width sets the number of lanes (one per ~0.9
        bond lengths, as ChemDraw's TLC tool), each with one spot at Rf 0.5."""
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        if lanes is None:
            lanes = max(1, int(round((x2 - x1) / (self.style.bond_length * 1.2))))
        kw.setdefault("lanes", [[{"rf": 0.5}] for _ in range(lanes)])
        t = TLCPlate(self.new_id(), x1, y1, x2, y2, **kw)
        self.tlcs[t.id] = t
        return t

    def add_gel(self, x1, y1, x2, y2, lanes: int | None = None, **kw) -> GelPlate:
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        if lanes is None:
            lanes = max(1, int(round((x2 - x1) / (self.style.bond_length * 1.2))))
        kw.setdefault("lanes", [{"label": str(k + 1), "bands": []} for k in range(lanes)])
        g = GelPlate(self.new_id(), x1, y1, x2, y2, **kw)
        self.gels[g.id] = g
        return g

    def add_clipart(self, x: float, y: float, name: str = "beaker", **kw) -> Clipart:
        c = Clipart(self.new_id(), float(x), float(y), name=name, **kw)
        self.cliparts[c.id] = c
        return c

    # --- associations ----------------------------------------------------------------
    def symbols_of(self, aid: int) -> list[Symbol]:
        return [s for s in self.symbols.values() if s.atom == aid]

    def anchor_point(self, ref: str) -> tuple[float, float] | None:
        """Position of an arrow anchor "atom:<id>", "bond:<id>" (mid-point) or
        "symbol:<id>"; None when the object is gone."""
        kind, _, i = (ref or "").partition(":")
        try:
            i = int(i)
        except ValueError:
            return None
        if kind == "atom" and i in self.atoms:
            return self.atoms[i].x, self.atoms[i].y
        if kind == "bond" and i in self.bonds:
            b = self.bonds[i]
            if b.a in self.atoms and b.b in self.atoms:
                p, q = self.atoms[b.a], self.atoms[b.b]
                return (p.x + q.x) / 2, (p.y + q.y) / 2
        if kind == "symbol" and i in self.symbols:
            return self.symbols[i].x, self.symbols[i].y
        return None

    def anchor(self, ar: Arrow, end: str, ref: str) -> None:
        """Attach the tail or head of a mechanism arrow to an object (keeps the offset)."""
        p = self.anchor_point(ref)
        if p is None:
            return
        if end == "tail":
            ar.tail_ref, ar.tail_dx, ar.tail_dy = ref, ar.x1 - p[0], ar.y1 - p[1]
        else:
            ar.head_ref, ar.head_dx, ar.head_dy = ref, ar.x2 - p[0], ar.y2 - p[1]

    def follow_anchors(self) -> None:
        """Move anchored arrow ends to their objects (anchors to deleted objects drop)."""
        for ar in self.arrows.values():
            for end in ("tail", "head"):
                ref = getattr(ar, end + "_ref")
                if not ref:
                    continue
                p = self.anchor_point(ref)
                if p is None:
                    setattr(ar, end + "_ref", "")
                    continue
                if end == "tail":
                    ar.x1, ar.y1 = p[0] + ar.tail_dx, p[1] + ar.tail_dy
                else:
                    ar.x2, ar.y2 = p[0] + ar.head_dx, p[1] + ar.head_dy

    def add_curve(self, points, **kw) -> Curve:
        """Curve from ``points`` (3 per anchor, see :class:`Curve`)."""
        c = Curve(self.new_id(), [[float(x), float(y)] for x, y in points], **kw)
        self.curves[c.id] = c
        return c

    def add_orbital(self, x: float, y: float, kind: str = "p", **kw) -> Orbital:
        o = Orbital(self.new_id(), float(x), float(y), kind, **kw)
        self.orbitals[o.id] = o
        return o

    # --- graph queries -----------------------------------------------------------------
    def bonds_of(self, aid: int) -> list[Bond]:
        return [b for b in self.bonds.values() if b.a == aid or b.b == aid]

    def neighbors(self, aid: int) -> list[int]:
        return [b.b if b.a == aid else b.a for b in self.bonds_of(aid)]

    def bond_between(self, a: int, b: int) -> Bond | None:
        for bond in self.bonds.values():
            if (bond.a == a and bond.b == b) or (bond.a == b and bond.b == a):
                return bond
        return None

    def adjacency(self) -> dict[int, list[int]]:
        adj: dict[int, list[int]] = {i: [] for i in self.atoms}
        for b in self.bonds.values():
            if b.a in adj and b.b in adj:
                adj[b.a].append(b.b)
                adj[b.b].append(b.a)
        return adj

    def fragments(self) -> list[list[int]]:
        """Connected molecules as sorted atom-id lists, ordered left to right."""
        adj = self.adjacency()
        seen: set[int] = set()
        out = []
        for start in sorted(adj):
            if start in seen:
                continue
            comp, queue = [], deque([start])
            seen.add(start)
            while queue:
                i = queue.popleft()
                comp.append(i)
                for j in adj[i]:
                    if j not in seen:
                        seen.add(j)
                        queue.append(j)
            out.append(sorted(comp))
        out.sort(key=lambda c: (min(self.atoms[i].x for i in c), c[0]))
        return out

    def fragment_of(self, aid: int) -> list[int]:
        for comp in self.fragments():
            if aid in comp:
                return comp
        return []

    def ring_of_bond(self, bid: int, max_size: int = 10) -> list[int] | None:
        """Atoms of the smallest ring through a bond (ordered around the ring) or None."""
        bond = self.bonds.get(bid)
        if bond is None:
            return None
        adj = self.adjacency()
        prev = {bond.a: None}
        queue = deque([bond.a])
        while queue:
            i = queue.popleft()
            if i == bond.b:
                break
            depth = 0
            j = i
            while prev[j] is not None:
                j = prev[j]
                depth += 1
            if depth >= max_size - 1:
                continue
            for n in adj[i]:
                if (i, n) in ((bond.a, bond.b), (bond.b, bond.a)) or n in prev:
                    continue
                prev[n] = i
                queue.append(n)
        if bond.b not in prev:
            return None
        path, j = [], bond.b
        while j is not None:
            path.append(j)
            j = prev[j]
        return path[::-1]

    def rings(self, max_size: int = 10) -> list[list[int]]:
        """Smallest ring through each ring bond (duplicates removed)."""
        out, seen = [], set()
        for bid in self.bonds:
            r = self.ring_of_bond(bid, max_size)
            if r and frozenset(r) not in seen:
                seen.add(frozenset(r))
                out.append(r)
        return out

    def ring_bond_ids(self) -> set[int]:
        out = set()
        for r in self.rings():
            for i in range(len(r)):
                b = self.bond_between(r[i], r[(i + 1) % len(r)])
                if b is not None:
                    out.add(b.id)
        return out

    # --- hit testing (model coordinates) -----------------------------------------------
    def atom_at(self, x: float, y: float, radius: float, exclude=()) -> Atom | None:
        best, best_d = None, radius
        for a in self.atoms.values():
            if a.id in exclude:
                continue
            d = math.hypot(a.x - x, a.y - y)
            if d <= best_d:
                best, best_d = a, d
        return best

    def bond_at(self, x: float, y: float, tol: float) -> Bond | None:
        best, best_d = None, tol
        for b in self.bonds.values():
            p, q = self.atoms[b.a], self.atoms[b.b]
            d, t = point_segment(x, y, p.x, p.y, q.x, q.y)
            if 0.1 <= t <= 0.9 and d <= best_d:
                best, best_d = b, d
        return best

    def arrow_at(self, x: float, y: float, tol: float) -> Arrow | None:
        from rsstatchem.core.chem_draw.geometry import arrow_path_points
        best, best_d = None, tol
        for ar in self.arrows.values():
            pts = arrow_path_points(ar)
            for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
                d, _ = point_segment(x, y, x1, y1, x2, y2)
                if d <= best_d:
                    best, best_d = ar, d
        return best

    def curve_at(self, x: float, y: float, tol: float) -> Curve | None:
        from rsstatchem.core.chem_draw.geometry import curve_path_points
        best, best_d = None, tol
        for c in self.curves.values():
            pts = curve_path_points(c)
            for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
                d, _ = point_segment(x, y, x1, y1, x2, y2)
                if d <= best_d:
                    best, best_d = c, d
            if c.closed and (c.filled or c.shaded) and _inside(x, y, pts) and best is None:
                best = c
        return best

    def orbital_at(self, x: float, y: float, tol: float) -> Orbital | None:
        from rsstatchem.core.chem_draw.geometry import orbital_outline
        best, best_d = None, None
        for o in self.orbitals.values():
            for lobe, _fill in orbital_outline(o, self.style.bond_length):
                if _inside(x, y, lobe) or min(math.hypot(x - px, y - py)
                                              for px, py in lobe) <= tol:
                    d = math.hypot(x - o.x, y - o.y)
                    if best_d is None or d < best_d:
                        best, best_d = o, d
                    break
        return best

    # --- edits ---------------------------------------------------------------------------
    def remove(self, keys) -> None:
        keys = list(keys)
        atom_ids = {i for k, i in keys if k == "atom"}
        for k, i in keys:
            self.store(k).pop(i, None)
        if atom_ids:
            for bid in [b.id for b in self.bonds.values() if b.a in atom_ids or
                        b.b in atom_ids]:
                del self.bonds[bid]
            for sid in [s.id for s in self.symbols.values() if s.atom in atom_ids]:
                del self.symbols[sid]
        self.follow_anchors()

    def erase(self, keys) -> None:
        """Delete like ChemDraw's eraser: removing bonds also removes carbon atoms that are
        left without bonds (labelled atoms stay)."""
        keys = list(keys)
        touched = set()
        for k, i in keys:
            if k == "bond" and i in self.bonds:
                touched.update((self.bonds[i].a, self.bonds[i].b))
        self.remove(keys)
        orphans = [("atom", a) for a in touched if a in self.atoms and not self.bonds_of(a)
                   and self.atoms[a].symbol == "C" and not self.atoms[a].label
                   and not self.atoms[a].charge and not self.atoms[a].isotope]
        self.remove(orphans)

    def atoms_of(self, keys) -> set[int]:
        """Atom ids addressed by keys (bonds contribute both atoms)."""
        out = set()
        for k, i in keys:
            if k == "atom" and i in self.atoms:
                out.add(i)
            elif k == "bond" and i in self.bonds:
                out.update((self.bonds[i].a, self.bonds[i].b))
        return out

    def with_bonds(self, keys) -> set[tuple[str, int]]:
        """Keys plus every bond whose two atoms are both selected."""
        out = set(keys)
        atoms = self.atoms_of(keys)
        for b in self.bonds.values():
            if b.a in atoms and b.b in atoms:
                out.add(("bond", b.id))
        for a in atoms:
            out.add(("atom", a))
        for s in self.symbols.values():
            if s.atom in atoms:
                out.add(("symbol", s.id))
        return out

    def _points(self, keys):
        """Mutable point references [(obj, xattr, yattr)] for the objects of ``keys``."""
        pts = []
        atoms = self.atoms_of(keys)
        for a in atoms:
            pts.append((self.atoms[a], "x", "y"))
        keyset = set(keys)
        for s in self.symbols.values():         # symbols travel with their atoms
            if s.atom in atoms and ("symbol", s.id) not in keyset:
                pts.append((s, "x", "y"))
        for k, i in keys:
            o = self.store(k).get(i)
            if o is None:
                continue
            if k in _BOXED:
                pts += [(o, "x1", "y1"), (o, "x2", "y2")]
            elif k in ("plus", "text", "orbital", "symbol", "clipart"):
                pts.append((o, "x", "y"))
            elif k == "curve":
                pts += [(p, 0, 1) for p in o.points]
        return pts

    def _detach_moved_arrows(self, keys) -> None:
        """Arrows moved without the objects they are anchored to come loose."""
        keyset = set(self.with_bonds(keys))
        for k, i in keys:
            ar = self.arrows.get(i) if k == "arrow" else None
            if ar is None:
                continue
            for end in ("tail", "head"):
                ref = getattr(ar, end + "_ref")
                kind, _, rid = ref.partition(":")
                if ref and (kind, int(rid or 0)) not in keyset:
                    setattr(ar, end + "_ref", "")

    def move(self, keys, dx: float, dy: float) -> None:
        keys = list(keys)
        self._detach_moved_arrows(keys)
        for o, xa, ya in self._points(keys):
            _set(o, xa, _get(o, xa) + dx)
            _set(o, ya, _get(o, ya) + dy)
        self.follow_anchors()

    def center(self, keys) -> tuple[float, float]:
        pts = [(_get(o, xa), _get(o, ya)) for o, xa, ya in self._points(keys)]
        if not pts:
            return 0.0, 0.0
        xs, ys = zip(*pts)
        return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2

    def rotate(self, keys, angle: float, cx: float | None = None,
               cy: float | None = None) -> None:
        """Rotate by ``angle`` radians (clockwise on screen) around (cx, cy)."""
        keys = list(keys)
        if cx is None or cy is None:
            cx, cy = self.center(keys)
        self._detach_moved_arrows(keys)
        c, s = math.cos(angle), math.sin(angle)
        for o, xa, ya in self._points(keys):
            x, y = _get(o, xa) - cx, _get(o, ya) - cy
            _set(o, xa, cx + c * x - s * y)
            _set(o, ya, cy + s * x + c * y)
        for k, i in keys:
            if k in ("bracket", "shape", "tlc", "gel") and i in self.store(k):
                self._normalize_bracket(self.store(k)[i])
            elif k == "orbital" and i in self.orbitals:
                self.orbitals[i].angle += angle
        for sym in self.symbols.values():
            if sym.atom in self.atoms_of(keys) or ("symbol", sym.id) in keys:
                sym.angle += angle
        self._rotate_anchor_offsets(keys, angle)
        self.follow_anchors()

    def flip(self, keys, horizontal: bool = True) -> None:
        """Mirror left-right (``horizontal``) or top-bottom around the selection centre."""
        keys = list(keys)
        cx, cy = self.center(keys)
        self._detach_moved_arrows(keys)
        for o, xa, ya in self._points(keys):
            if horizontal:
                _set(o, xa, 2 * cx - _get(o, xa))
            else:
                _set(o, ya, 2 * cy - _get(o, ya))
        for k, i in keys:
            if k == "arrow" and i in self.arrows:
                self.arrows[i].bend = -self.arrows[i].bend
            if k == "orbital" and i in self.orbitals:
                o = self.orbitals[i]
                o.angle = (math.pi - o.angle) if horizontal else -o.angle
            if k in ("bracket", "shape", "tlc", "gel") and i in self.store(k):
                self._normalize_bracket(self.store(k)[i])
        for sym in self.symbols.values():
            if sym.atom in self.atoms_of(keys) or ("symbol", sym.id) in keys:
                sym.angle = (math.pi - sym.angle) if horizontal else -sym.angle
        for k, i in keys:
            ar = self.arrows.get(i) if k == "arrow" else None
            if ar is not None:
                if horizontal:
                    ar.tail_dx, ar.head_dx = -ar.tail_dx, -ar.head_dx
                else:
                    ar.tail_dy, ar.head_dy = -ar.tail_dy, -ar.head_dy
        self.follow_anchors()

    def _rotate_anchor_offsets(self, keys, angle: float) -> None:
        c, s = math.cos(angle), math.sin(angle)
        for k, i in keys:
            ar = self.arrows.get(i) if k == "arrow" else None
            if ar is None:
                continue
            ar.tail_dx, ar.tail_dy = c * ar.tail_dx - s * ar.tail_dy, \
                s * ar.tail_dx + c * ar.tail_dy
            ar.head_dx, ar.head_dy = c * ar.head_dx - s * ar.head_dy, \
                s * ar.head_dx + c * ar.head_dy

    @staticmethod
    def _normalize_bracket(b: Bracket) -> None:
        b.x1, b.x2 = min(b.x1, b.x2), max(b.x1, b.x2)
        b.y1, b.y2 = min(b.y1, b.y2), max(b.y1, b.y2)

    def scale(self, factor: float, cx: float = 0.0, cy: float = 0.0) -> None:
        for o, xa, ya in self._points(self.keys()):
            _set(o, xa, cx + (_get(o, xa) - cx) * factor)
            _set(o, ya, cy + (_get(o, ya) - cy) * factor)
        for ar in self.arrows.values():
            ar.bend *= factor
            for f in ("tail_dx", "tail_dy", "head_dx", "head_dy"):
                setattr(ar, f, getattr(ar, f) * factor)
        for o in self.orbitals.values():
            if o.size:
                o.size *= factor
        for c in self.cliparts.values():
            c.w *= factor
            c.h *= factor
        for t in self.tlcs.values():
            for lane in t.lanes:
                for sp in lane:
                    for f in ("w", "h", "tail"):
                        if sp.get(f):
                            sp[f] *= factor
        self.follow_anchors()

    def merge_atoms(self, drop: int, keep: int) -> None:
        """Fuse atom ``drop`` into ``keep`` (bonds are re-attached; duplicates dropped)."""
        if drop == keep or drop not in self.atoms or keep not in self.atoms:
            return
        for b in list(self.bonds_of(drop)):
            other = b.b if b.a == drop else b.a
            if other == keep or self.bond_between(keep, other) is not None:
                del self.bonds[b.id]
                continue
            if b.a == drop:
                b.a = keep
            else:
                b.b = keep
        d, k = self.atoms[drop], self.atoms[keep]
        if k.symbol == "C" and not k.label and (d.symbol != "C" or d.label):
            k.symbol, k.label, k.charge = d.symbol, d.label, d.charge
        del self.atoms[drop]

    def merge_coincident(self, atom_ids, tol: float) -> dict[int, int]:
        """Fuse each of ``atom_ids`` into a different atom lying within ``tol``."""
        mapping = {}
        for aid in list(atom_ids):
            if aid not in self.atoms:
                continue
            a = self.atoms[aid]
            other = self.atom_at(a.x, a.y, tol, exclude={aid} | set(mapping))
            if other is not None and other.id not in atom_ids:
                self.merge_atoms(aid, other.id)
                mapping[aid] = other.id
        return mapping

    def bbox(self, keys=None) -> tuple[float, float, float, float] | None:
        keys = self.keys() if keys is None else keys
        pts = [(_get(o, xa), _get(o, ya)) for o, xa, ya in self._points(keys)]
        for k, i in keys:                               # orbitals extend around the centre
            o = self.orbitals.get(i) if k == "orbital" else None
            if o is not None:
                r = o.size or self.style.bond_length
                pts += [(o.x - r, o.y - r), (o.x + r, o.y + r)]
            c = self.cliparts.get(i) if k == "clipart" else None
            if c is not None:
                pts.append((c.x + c.w, c.y + c.h))
        if not pts:
            return None
        xs, ys = zip(*pts)
        return min(xs), min(ys), max(xs), max(ys)

    def next_map_number(self) -> int:
        return max([a.map_number for a in self.atoms.values()] + [0]) + 1

    def next_group(self) -> int:
        objs = [o for k in KINDS for o in self.store(k).values()]
        return max([o.group for o in objs] + [0]) + 1

    def group_members(self, keys) -> set[tuple[str, int]]:
        groups = {self.get(k).group for k in keys if self.get(k) is not None} - {0}
        out = set(keys)
        if groups:
            for k in KINDS:
                for i, o in self.store(k).items():
                    if o.group in groups:
                        out.add((k, i))
        return out

    def top_z(self) -> float:
        zs = [o.z for k in OBJECT_KINDS for o in self.store(k).values()]
        return max(zs + [0.0])

    # --- (de)serialisation ---------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "format": FORMAT, "version": VERSION, "style": self.style.to_dict(),
            "meta": dict(self.meta), "next_id": self._next,
            **{_STORES[k]: [asdict(o) for o in self.store(k).values()] for k in KINDS},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        if data.get("format", FORMAT) != FORMAT:
            raise ValueError("Not an RS-StatChem drawing")
        doc = cls(Style.from_dict(data.get("style")))
        doc.meta = dict(data.get("meta") or {})
        for k in KINDS:
            store = doc.store(k)
            for item in data.get(_STORES[k], []):
                o = _from_dict(_CLASSES[k], item)
                store[o.id] = o
        ids = [o.id for k in KINDS for o in doc.store(k).values()]
        doc._next = max([int(data.get("next_id", 1))] + [i + 1 for i in ids])
        return doc

    def copy(self) -> "Document":
        return Document.from_dict(json.loads(json.dumps(self.to_dict())))

    def subset(self, keys) -> "Document":
        """New document with the given objects (and bonds between selected atoms)."""
        keys = self.with_bonds(keys)
        sub = Document(self.style.updated())
        atoms = self.atoms_of(keys)
        for a in atoms:
            sub.atoms[a] = _from_dict(Atom, asdict(self.atoms[a]))
        for k, i in keys:
            if k == "atom":
                continue
            o = self.store(k).get(i)
            if o is None or (k == "bond" and not (o.a in atoms and o.b in atoms)):
                continue
            sub.store(k)[i] = _from_dict(_CLASSES[k], asdict(o))
        for s in sub.symbols.values():
            if s.atom and s.atom not in atoms:
                s.atom = 0
        for ar in sub.arrows.values():
            for end in ("tail", "head"):
                ref = getattr(ar, end + "_ref")
                kind, _, rid = ref.partition(":")
                if ref and sub.store(kind).get(int(rid or 0)) is None:
                    setattr(ar, end + "_ref", "")
        sub._next = self._next
        return sub

    def paste(self, other: "Document", dx: float = 0.0, dy: float = 0.0) -> list:
        """Insert a copy of ``other`` (new ids), shifted by (dx, dy); returns the new keys."""
        amap: dict[int, int] = {}
        omap: dict[tuple[str, int], int] = {}
        new_keys = []
        for a in other.atoms.values():
            d = asdict(a)
            d.update(id=self.new_id(), x=a.x + dx, y=a.y + dy, group=0)
            na = _from_dict(Atom, d)
            self.atoms[na.id] = na
            amap[a.id] = na.id
            omap[("atom", a.id)] = na.id
            new_keys.append(("atom", na.id))
        for b in other.bonds.values():
            if b.a in amap and b.b in amap:
                nb = self.add_bond(amap[b.a], amap[b.b], b.order, b.style,
                                   double_pos=b.double_pos, color=b.color, stereo=b.stereo)
                omap[("bond", b.id)] = nb.id
                new_keys.append(("bond", nb.id))
        new_objs = []
        for k in OBJECT_KINDS:
            for o in other.store(k).values():
                d = asdict(o)
                d["id"] = self.new_id()
                d["group"] = 0
                for xa, ya in (("x", "y"), ("x1", "y1"), ("x2", "y2")):
                    if xa in d:
                        d[xa] += dx
                        d[ya] += dy
                if "points" in d:
                    d["points"] = [[px + dx, py + dy] for px, py in d["points"]]
                if k == "symbol":
                    d["atom"] = amap.get(d["atom"], 0)
                no = _from_dict(_CLASSES[k], d)
                self.store(k)[no.id] = no
                omap[(k, o.id)] = no.id
                new_keys.append((k, no.id))
                new_objs.append((k, no))
        for k, no in new_objs:                  # re-point mechanism arrow anchors
            if k != "arrow":
                continue
            for end in ("tail", "head"):
                ref = getattr(no, end + "_ref")
                kind, _, rid = ref.partition(":")
                nid = omap.get((kind, int(rid or 0))) if ref else None
                setattr(no, end + "_ref", f"{kind}:{nid}" if nid else "")
        return new_keys

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=1)

    @classmethod
    def from_json(cls, text: str) -> "Document":
        return cls.from_dict(json.loads(text))


def _get(o, a):
    return o[a] if isinstance(o, list) else getattr(o, a)


def _set(o, a, v) -> None:
    if isinstance(o, list):
        o[a] = v
    else:
        setattr(o, a, v)


def _inside(x: float, y: float, poly) -> bool:
    """Point in polygon (even-odd rule)."""
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / ((y2 - y1) or 1e-12):
            inside = not inside
    return inside


def point_segment(px, py, x1, y1, x2, y2) -> tuple[float, float]:
    """(distance, t) from a point to the segment (x1, y1)-(x2, y2); t in [0, 1]."""
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / L2))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy)), t


def save_rsc(doc: Document, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc.to_json())


def load_rsc(path: str) -> Document:
    with open(path, encoding="utf-8") as fh:
        return Document.from_json(fh.read())

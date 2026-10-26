"""Drawing document of the structure / reaction editor (the ChemDraw document model).

A :class:`Document` holds atoms and bonds (any number of molecules), reaction arrows (also
curved electron-pushing arrows), "+" signs, free text boxes and brackets. Coordinates are in
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
from dataclasses import asdict, dataclass, fields

from rsstatchem.core.chem_draw.style import Style

__all__ = ["Atom", "Bond", "Arrow", "Plus", "TextBox", "Bracket", "Document", "KINDS",
           "FORMAT", "VERSION", "save_rsc", "load_rsc", "ARROW_KINDS", "BOND_STYLES"]

FORMAT = "rs-statchem-drawing"
VERSION = 1
KINDS = ("atom", "bond", "arrow", "plus", "text", "bracket")
ARROW_KINDS = ("forward", "equilibrium", "retro", "resonance", "curved", "fishhook", "line")
REACTION_ARROWS = ("forward", "equilibrium", "retro", "resonance")
BOND_STYLES = ("", "wedge", "hash", "wavy", "bold", "dashed", "either")


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


_CLASSES = {"atom": Atom, "bond": Bond, "arrow": Arrow, "plus": Plus, "text": TextBox,
            "bracket": Bracket}
_STORES = {"atom": "atoms", "bond": "bonds", "arrow": "arrows", "plus": "pluses",
           "text": "texts", "bracket": "brackets"}


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

    def add_bracket(self, x1, y1, x2, y2, kind: str = "square", label: str = "") -> Bracket:
        b = Bracket(self.new_id(), min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), kind,
                    label)
        self.brackets[b.id] = b
        return b

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
        return out

    def _points(self, keys):
        """Mutable point references [(obj, xattr, yattr)] for the objects of ``keys``."""
        pts = []
        for a in self.atoms_of(keys):
            pts.append((self.atoms[a], "x", "y"))
        for k, i in keys:
            o = self.store(k).get(i)
            if o is None:
                continue
            if k in ("arrow", "bracket"):
                pts += [(o, "x1", "y1"), (o, "x2", "y2")]
            elif k in ("plus", "text"):
                pts.append((o, "x", "y"))
        return pts

    def move(self, keys, dx: float, dy: float) -> None:
        for o, xa, ya in self._points(keys):
            setattr(o, xa, getattr(o, xa) + dx)
            setattr(o, ya, getattr(o, ya) + dy)

    def center(self, keys) -> tuple[float, float]:
        pts = [(getattr(o, xa), getattr(o, ya)) for o, xa, ya in self._points(keys)]
        if not pts:
            return 0.0, 0.0
        xs, ys = zip(*pts)
        return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2

    def rotate(self, keys, angle: float, cx: float | None = None,
               cy: float | None = None) -> None:
        """Rotate by ``angle`` radians (clockwise on screen) around (cx, cy)."""
        if cx is None or cy is None:
            cx, cy = self.center(keys)
        c, s = math.cos(angle), math.sin(angle)
        for o, xa, ya in self._points(keys):
            x, y = getattr(o, xa) - cx, getattr(o, ya) - cy
            setattr(o, xa, cx + c * x - s * y)
            setattr(o, ya, cy + s * x + c * y)
        for k, i in keys:
            if k == "bracket" and i in self.brackets:
                self._normalize_bracket(self.brackets[i])

    def flip(self, keys, horizontal: bool = True) -> None:
        """Mirror left-right (``horizontal``) or top-bottom around the selection centre."""
        cx, cy = self.center(keys)
        for o, xa, ya in self._points(keys):
            if horizontal:
                setattr(o, xa, 2 * cx - getattr(o, xa))
            else:
                setattr(o, ya, 2 * cy - getattr(o, ya))
        for k, i in keys:
            if k == "arrow" and i in self.arrows:
                self.arrows[i].bend = -self.arrows[i].bend
            if k == "bracket" and i in self.brackets:
                self._normalize_bracket(self.brackets[i])

    @staticmethod
    def _normalize_bracket(b: Bracket) -> None:
        b.x1, b.x2 = min(b.x1, b.x2), max(b.x1, b.x2)
        b.y1, b.y2 = min(b.y1, b.y2), max(b.y1, b.y2)

    def scale(self, factor: float, cx: float = 0.0, cy: float = 0.0) -> None:
        for o, xa, ya in self._points(self.keys()):
            setattr(o, xa, cx + (getattr(o, xa) - cx) * factor)
            setattr(o, ya, cy + (getattr(o, ya) - cy) * factor)
        for ar in self.arrows.values():
            ar.bend *= factor

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
        pts = [(getattr(o, xa), getattr(o, ya)) for o, xa, ya in self._points(keys)]
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
        zs = [o.z for k in ("arrow", "plus", "text", "bracket") for o in self.store(k).values()]
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
        sub._next = self._next
        return sub

    def paste(self, other: "Document", dx: float = 0.0, dy: float = 0.0) -> list:
        """Insert a copy of ``other`` (new ids), shifted by (dx, dy); returns the new keys."""
        amap: dict[int, int] = {}
        new_keys = []
        for a in other.atoms.values():
            d = asdict(a)
            d.update(id=self.new_id(), x=a.x + dx, y=a.y + dy, group=0)
            na = _from_dict(Atom, d)
            self.atoms[na.id] = na
            amap[a.id] = na.id
            new_keys.append(("atom", na.id))
        for b in other.bonds.values():
            if b.a in amap and b.b in amap:
                nb = self.add_bond(amap[b.a], amap[b.b], b.order, b.style,
                                   double_pos=b.double_pos, color=b.color)
                new_keys.append(("bond", nb.id))
        for k in ("arrow", "plus", "text", "bracket"):
            for o in other.store(k).values():
                d = asdict(o)
                d["id"] = self.new_id()
                d["group"] = 0
                for xa, ya in (("x", "y"), ("x1", "y1"), ("x2", "y2")):
                    if xa in d:
                        d[xa] += dx
                        d[ya] += dy
                no = _from_dict(_CLASSES[k], d)
                self.store(k)[no.id] = no
                new_keys.append((k, no.id))
        return new_keys

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=1)

    @classmethod
    def from_json(cls, text: str) -> "Document":
        return cls.from_dict(json.loads(text))


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

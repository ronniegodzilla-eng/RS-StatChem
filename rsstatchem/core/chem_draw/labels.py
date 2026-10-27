"""Atom labels as ChemDraw draws them: which atoms show a label, implicit hydrogens, on
which side the hydrogens go (``OH`` vs ``HO``, ``NH`` above/below in rings), charges,
isotopes and how nicknames are written (``CO2H`` / ``HO2C``).

A :class:`LabelLayout` lists the text runs (normal / subscript / superscript) of one
horizontal label line plus an optional stacked ``H`` run placed above or below, and the
character range (``anchor``) that is centred on the atom position.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from rsstatchem.core.chem_draw import abbrev
from rsstatchem.core.chem_draw.model import Atom, Document

__all__ = ["VALENCES", "implicit_h", "label_visible", "h_side", "LabelLayout", "layout",
           "charge_text", "split_digits", "bond_sum", "approx_extent"]

VALENCES = {"H": (1,), "B": (3,), "C": (4,), "N": (3, 5), "O": (2,), "F": (1,),
            "Al": (3,), "Si": (4,), "P": (3, 5), "S": (2, 4, 6), "Cl": (1,), "Ge": (4,),
            "As": (3, 5), "Se": (2, 4, 6), "Br": (1,), "Te": (2, 4, 6), "I": (1, 3, 5)}
_LEFT_H = {"O", "S", "Se", "Te", "F", "Cl", "Br", "I"}
MINUS = "−"


def bond_sum(doc: Document, aid: int) -> int:
    # a dative bond adds nothing to its donor (begin atom), as in RDKit
    orders = [b.order for b in doc.bonds_of(aid) if not (b.style == "dative" and b.a == aid)]
    arom = sum(1 for o in orders if o == 1.5)
    total = sum(int(o) for o in orders if o != 1.5) + arom + (1 if arom >= 2 else 0)
    return total


def implicit_h(doc: Document, atom: Atom) -> int:
    """Hydrogens implied by the valence (or the explicit count when set)."""
    if atom.label:
        return 0
    if atom.explicit_h is not None:
        return max(0, int(atom.explicit_h))
    vals = VALENCES.get(atom.symbol)
    if not vals:
        return 0
    q = atom.charge
    if atom.symbol in ("C", "Si", "Ge"):
        vals = tuple(v - abs(q) for v in vals)
    elif atom.symbol in ("B", "Al"):
        vals = tuple(v - q for v in vals)
    elif atom.symbol != "H":
        vals = tuple(v + q for v in vals)
    else:
        vals = tuple(v - abs(q) for v in vals)
    used = bond_sum(doc, atom.id) + atom.radicals
    for v in vals:
        if v >= used:
            return v - used
    return 0


def label_visible(doc: Document, atom: Atom, degree: int | None = None) -> bool:
    if atom.label or atom.symbol != "C":
        return True
    circled = getattr(atom, "charge_circle", False)     # circled charge: carbon stays bare
    if (atom.charge and not circled) or atom.isotope or atom.show_carbon:
        return True
    if atom.radicals and not circled:
        return True
    if degree is None:
        degree = len(doc.neighbors(atom.id))
    return degree == 0


def _unit(doc: Document, atom: Atom) -> list[tuple[float, float]]:
    out = []
    for n in doc.neighbors(atom.id):
        o = doc.atoms[n]
        dx, dy = o.x - atom.x, o.y - atom.y
        d = math.hypot(dx, dy) or 1.0
        out.append((dx / d, dy / d))
    return out


def h_side(doc: Document, atom: Atom) -> str:
    """'right', 'left', 'up' or 'down' for the implicit hydrogens of a label."""
    vecs = _unit(doc, atom)
    if not vecs:
        return "left" if atom.symbol in _LEFT_H else "right"
    if len(vecs) == 1:
        return "left" if vecs[0][0] > 0.1 else "right"
    best, best_score = "right", -1e9
    for name, (cx, cy), bonus in (("right", (1, 0), 12), ("left", (-1, 0), 10),
                                   ("down", (0, 1), 0), ("up", (0, -1), 0)):
        score = min(math.degrees(math.acos(max(-1.0, min(1.0, cx * vx + cy * vy))))
                    for vx, vy in vecs) + bonus
        if score > best_score:
            best, best_score = name, score
    return best


def from_right(doc: Document, atom: Atom) -> bool:
    """True when the bonds leave the label towards the right (text then runs leftwards)."""
    vecs = _unit(doc, atom)
    return bool(vecs) and sum(v[0] for v in vecs) > 0.1


def charge_text(q: int) -> str:
    if not q:
        return ""
    sign = "+" if q > 0 else MINUS
    return sign if abs(q) == 1 else f"{abs(q)}{sign}"


def split_digits(text: str) -> list[tuple[str, str]]:
    """Formula-style runs: digits after a letter or ')' become subscripts."""
    runs: list[tuple[str, str]] = []
    for i, ch in enumerate(text):
        kind = "sub" if ch.isdigit() and i and (text[i - 1].isalpha() or
                                                text[i - 1] in ")]" or
                                                (runs and runs[-1][1] == "sub")) else "n"
        if runs and runs[-1][1] == kind:
            runs[-1] = (runs[-1][0] + ch, kind)
        else:
            runs.append((ch, kind))
    return runs


@dataclass
class LabelLayout:
    visible: bool
    runs: list = field(default_factory=list)          # [(text, 'n' | 'sub' | 'sup')]
    anchor: tuple[int, int] = (0, 0)                  # character range on the atom
    stacked: list = field(default_factory=list)       # H runs above / below the anchor
    stacked_pos: str = ""                             # '' | 'up' | 'down'
    radicals: int = 0

    @property
    def text(self) -> str:
        return "".join(t for t, _ in self.runs)


def layout(doc: Document, atom: Atom) -> LabelLayout:
    degree = len(doc.neighbors(atom.id))
    if not label_visible(doc, atom, degree):
        return LabelLayout(False, radicals=atom.radicals)
    if atom.label:
        text, anchor = abbrev.label_text(atom.label, from_right(doc, atom))
        runs = split_digits(text)
        if anchor and from_right(doc, atom) and text.endswith(anchor):
            rng = (len(text) - len(anchor), len(text))
        elif anchor and text.startswith(anchor):
            rng = (0, len(anchor))
        else:
            rng = (0, len(text))
        if atom.charge and not getattr(atom, "charge_circle", False):
            runs.append((charge_text(atom.charge), "sup"))
        return LabelLayout(True, runs, rng, radicals=atom.radicals)
    nh = implicit_h(doc, atom)
    side = h_side(doc, atom) if nh else "right"
    hruns = [("H", "n")] + ([(str(nh), "sub")] if nh > 1 else []) if nh else []
    runs: list[tuple[str, str]] = []
    if side == "left":
        runs += hruns
    if atom.isotope:
        runs.append((str(atom.isotope), "sup"))
    start = sum(len(t) for t, _ in runs)
    runs.append((atom.symbol, "n"))
    rng = (start, start + len(atom.symbol))
    if side == "right":
        runs += hruns
    if atom.charge and not getattr(atom, "charge_circle", False):
        runs.append((charge_text(atom.charge), "sup"))
    stacked, pos = ([], "")
    if side in ("up", "down"):
        stacked, pos = hruns, side
    return LabelLayout(True, runs, rng, stacked, pos, atom.radicals)


def approx_extent(doc: Document, atom: Atom, font_size: float) -> tuple[float, float]:
    """Approximate (left, right) extent of an atom's label from the atom position (pt),
    for layouts done without font metrics."""
    lay = layout(doc, atom)
    if not lay.visible:
        return 0.0, 0.0
    widths = []
    for text, kind in lay.runs:
        f = 0.45 if kind in ("sub", "sup") else 0.68
        widths += [f * font_size] * len(text)
    a0, a1 = lay.anchor
    left = sum(widths[:a0]) + sum(widths[a0:a1]) / 2
    right = sum(widths[a1:]) + sum(widths[a0:a1]) / 2
    return left, right

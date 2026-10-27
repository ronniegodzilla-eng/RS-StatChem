"""Template library of the editor (ChemDraw's template toolbars): amino acids,
nucleobases, sugars, heterocycles, aromatics, bicyclics and cages, functional groups,
protecting groups, conformations and projections (Newman, Fischer, Haworth, chair with
axial / equatorial bonds, boat) and clipware (laboratory glassware).

Most templates are SMILES; ``*`` marks an attachment point (drawn as a generic "R" label).
Stereocentres are given with the natural configuration (L-amino acids, D-sugars).
Projections and clipware cannot be written as SMILES: their entries are ``@tpl:<key>`` /
``@clip:<name>`` and are drawn by :func:`insert_drawn` (atoms, bonds, lines and circles, so
they can be relabelled like any drawing).
"""
from __future__ import annotations

import math

__all__ = ["TEMPLATES", "categories", "find", "DRAWN", "is_drawn", "insert_drawn",
           "build_drawn"]

TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "Amino acids": [
        ("Glycine", "NCC(=O)O"), ("L-Alanine", "C[C@H](N)C(=O)O"),
        ("L-Valine", "CC(C)[C@H](N)C(=O)O"), ("L-Leucine", "CC(C)C[C@H](N)C(=O)O"),
        ("L-Isoleucine", "CC[C@H](C)[C@H](N)C(=O)O"), ("L-Proline", "OC(=O)[C@@H]1CCCN1"),
        ("L-Phenylalanine", "N[C@@H](Cc1ccccc1)C(=O)O"),
        ("L-Tyrosine", "N[C@@H](Cc1ccc(O)cc1)C(=O)O"),
        ("L-Tryptophan", "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O"),
        ("L-Serine", "N[C@@H](CO)C(=O)O"), ("L-Threonine", "C[C@@H](O)[C@H](N)C(=O)O"),
        ("L-Cysteine", "N[C@@H](CS)C(=O)O"), ("L-Methionine", "CSCC[C@H](N)C(=O)O"),
        ("L-Asparagine", "NC(=O)C[C@H](N)C(=O)O"),
        ("L-Glutamine", "NC(=O)CC[C@H](N)C(=O)O"),
        ("L-Aspartic acid", "N[C@@H](CC(=O)O)C(=O)O"),
        ("L-Glutamic acid", "N[C@@H](CCC(=O)O)C(=O)O"),
        ("L-Lysine", "NCCCC[C@H](N)C(=O)O"),
        ("L-Arginine", "N[C@@H](CCCNC(=N)N)C(=O)O"),
        ("L-Histidine", "N[C@@H](Cc1c[nH]cn1)C(=O)O"),
    ],
    "Nucleobases": [
        ("Adenine", "Nc1ncnc2[nH]cnc12"), ("Guanine", "Nc1nc2[nH]cnc2c(=O)[nH]1"),
        ("Cytosine", "Nc1cc[nH]c(=O)n1"), ("Thymine", "Cc1c[nH]c(=O)[nH]c1=O"),
        ("Uracil", "O=c1cc[nH]c(=O)[nH]1"),
    ],
    "Sugars": [
        ("β-D-Glucopyranose", "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O"),
        ("α-D-Glucopyranose", "OC[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@@H]1O"),
        ("β-D-Galactopyranose", "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@H]1O"),
        ("β-D-Ribofuranose", "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H]1O"),
        ("2-Deoxy-β-D-ribofuranose", "OC[C@H]1O[C@@H](O)C[C@@H]1O"),
        ("β-D-Fructofuranose", "OC[C@H]1O[C@@](O)(CO)[C@@H](O)[C@@H]1O"),
    ],
    "Rings & heterocycles": [
        ("Naphthalene", "c1ccc2ccccc2c1"), ("Anthracene", "c1ccc2cc3ccccc3cc2c1"),
        ("Pyridine", "c1ccncc1"), ("Pyrimidine", "c1cncnc1"), ("Pyrrole", "c1cc[nH]c1"),
        ("Furan", "c1ccoc1"), ("Thiophene", "c1ccsc1"), ("Imidazole", "c1c[nH]cn1"),
        ("Indole", "c1ccc2[nH]ccc2c1"), ("Quinoline", "c1ccc2ncccc2c1"),
        ("Piperidine", "C1CCNCC1"), ("Morpholine", "C1COCCN1"),
        ("Tetrahydrofuran", "C1CCOC1"), ("Cyclohexane", "C1CCCCC1"),
        ("Steroid core", "C1CCC2C(C1)CCC1C2CCC2CCCC21"),
    ],
    "Aromatics": [
        ("Benzene", "c1ccccc1"), ("Biphenyl", "c1ccc(-c2ccccc2)cc1"),
        ("Phenanthrene", "c1ccc2c(c1)ccc1ccccc12"), ("Pyrene", "c1cc2ccc3cccc4ccc(c1)c2c34"),
        ("Azulene", "c1ccc2cccc2cc1"), ("Cyclopentadienide", "[cH-]1cccc1"),
        ("Tropylium", "[cH+]1cccccc1"), ("Fluorene", "c1ccc2c(c1)Cc1ccccc12"),
    ],
    "Bicyclics & cages": [
        ("Norbornane", "C1CC2CCC1C2"), ("Norbornene", "C1=CC2CCC1C2"),
        ("Bicyclo[2.2.2]octane", "C1CC2CCC1CC2"), ("Decalin", "C1CCC2CCCCC2C1"),
        ("Camphor", "CC1(C)C2CCC1(C)C(=O)C2"), ("Adamantane", "C1C2CC3CC1CC(C2)C3"),
        ("Cubane", "C12C3C4C1C5C2C3C45"), ("Spiro[4.5]decane", "C1CCC2(C1)CCCCC2"),
    ],
    "Conformations & projections": [
        ("Newman projection, staggered", "@tpl:newman_staggered"),
        ("Newman projection, eclipsed", "@tpl:newman_eclipsed"),
        ("Fischer projection, 1 centre (D-glyceraldehyde)", "@tpl:fischer1"),
        ("Fischer projection, 2 centres", "@tpl:fischer2"),
        ("Fischer projection, D-glucose", "@tpl:fischer_glucose"),
        ("Haworth, β-D-glucopyranose", "@tpl:haworth_pyranose"),
        ("Haworth, β-D-ribofuranose", "@tpl:haworth_furanose"),
        ("Chair with axial / equatorial bonds", "@tpl:chair_axial"),
        ("Boat cyclohexane", "@tpl:boat"),
    ],
    "Clipware (glassware)": [
        ("Beaker", "@clip:beaker"), ("Erlenmeyer flask", "@clip:erlenmeyer"),
        ("Round-bottom flask", "@clip:roundbottom"), ("Test tube", "@clip:testtube"),
        ("Graduated cylinder", "@clip:cylinder"), ("Funnel", "@clip:funnel"),
        ("Separatory funnel", "@clip:sepfunnel"), ("Liebig condenser", "@clip:condenser"),
        ("Burette", "@clip:burette"), ("Thermometer", "@clip:thermometer"),
        ("Petri dish", "@clip:petri"), ("Vial", "@clip:vial"),
    ],
    "Functional groups": [
        ("Carboxylic acid", "*C(=O)O"), ("Ester", "*C(=O)OC"), ("Amide", "*C(N)=O"),
        ("Aldehyde", "*C=O"), ("Ketone", "*C(C)=O"), ("Nitrile", "*C#N"),
        ("Nitro", "*[N+](=O)[O-]"), ("Sulfonamide", "*S(N)(=O)=O"),
        ("Sulfonic acid", "*S(=O)(=O)O"), ("Phenyl", "*c1ccccc1"), ("Phenol", "Oc1ccccc1"),
        ("Aniline", "Nc1ccccc1"), ("Acetyl chloride", "CC(Cl)=O"), ("Urea", "NC(N)=O"),
        ("Guanidine", "NC(N)=N"), ("Azide", "*N=[N+]=[N-]"),
    ],
    "Protecting groups": [
        ("Boc", "*C(=O)OC(C)(C)C"), ("Cbz", "*C(=O)OCc1ccccc1"),
        ("Fmoc", "*C(=O)OCC1c2ccccc2-c2ccccc12"), ("Acetyl (Ac)", "*C(C)=O"),
        ("Benzyl (Bn)", "*Cc1ccccc1"), ("PMB", "*Cc1ccc(OC)cc1"),
        ("TBS", "*[Si](C)(C)C(C)(C)C"), ("TMS", "*[Si](C)(C)C"),
        ("Tosyl (Ts)", "*S(=O)(=O)c1ccc(C)cc1"), ("Mesyl (Ms)", "*S(C)(=O)=O"),
        ("THP", "*C1CCCCO1"), ("MOM", "*COC"),
    ],
}


def categories() -> list[str]:
    return list(TEMPLATES)


def find(name: str) -> str | None:
    for items in TEMPLATES.values():
        for n, smi in items:
            if n.lower() == name.lower():
                return smi
    return None


# --- drawn templates (projections, conformers, clipware) ----------------------------------

def is_drawn(text: str) -> bool:
    return isinstance(text, str) and text.startswith(("@tpl:", "@clip:"))


def _label(doc, aid: int, text: str) -> None:
    from rsstatchem.core.chem_draw import convert
    convert.apply_label_text(doc, aid, text)


def _atoms(doc, pts, bl, labels=None):
    ids = [doc.add_atom(x * bl, y * bl).id for x, y in pts]
    for k, text in (labels or {}).items():
        _label(doc, ids[k], text)
    return ids


def _newman(doc, bl, eclipsed: bool) -> None:
    r = bl * 0.55
    c = doc.add_atom(0, 0)
    for k in range(3):
        a = math.radians(-90 + 120 * k)
        end = doc.add_atom(bl * math.cos(a), bl * math.sin(a))
        doc.add_bond(c.id, end.id)
    doc.add_shape(-r, -r, r, r, "circle")
    off = 20 if eclipsed else 60
    for k in range(3):
        a = math.radians(-90 + off + 120 * k)
        doc.add_arrow(r * math.cos(a), r * math.sin(a), bl * 1.05 * math.cos(a),
                      bl * 1.05 * math.sin(a), "line")


def _fischer(doc, bl, rights: list[bool], top="CHO", bottom="CH2OH") -> None:
    n = len(rights)
    ys = [(k - (n - 1) / 2) for k in range(n)]
    spine = [(0, ys[0] - 1)] + [(0, y) for y in ys] + [(0, ys[-1] + 1)]
    ids = _atoms(doc, spine, bl, {0: top, len(spine) - 1: bottom})
    for a, b in zip(ids, ids[1:]):
        doc.add_bond(a, b)
    for k, right in enumerate(rights):
        y = ys[k]
        left = doc.add_atom(-bl, y * bl)
        rgt = doc.add_atom(bl, y * bl)
        _label(doc, left.id, "H" if right else "OH")
        _label(doc, rgt.id, "OH" if right else "H")
        doc.add_bond(ids[k + 1], left.id)
        doc.add_bond(ids[k + 1], rgt.id)


def _haworth(doc, bl, ring, front, subs) -> None:
    """``ring`` [(x, y, label)], ``front`` = indices of ring bonds drawn bold (k -> k+1),
    ``subs`` [(ring index, up?, label[, dx])] (dx tilts the bond, in bond lengths)."""
    ids = []
    for x, y, lab in ring:
        a = doc.add_atom(x * bl, y * bl)
        if lab:
            _label(doc, a.id, lab)
        ids.append(a.id)
    n = len(ids)
    for k in range(n):
        doc.add_bond(ids[k], ids[(k + 1) % n], 1, "bold" if k in front else "")
    for k, up, lab, *dx in subs:
        a = doc.atoms[ids[k]]
        s = doc.add_atom(a.x + (dx[0] if dx else 0.0) * bl, a.y + (-0.8 if up else 0.8) * bl)
        _label(doc, s.id, lab)
        doc.add_bond(ids[k], s.id)


def _chair_axial(doc, bl) -> None:
    from rsstatchem.core.chem_draw.ops import CHAIR_TEMPLATES
    pts = CHAIR_TEMPLATES["chair1"]
    ids = _atoms(doc, pts, bl)
    for k in range(6):
        doc.add_bond(ids[k], ids[(k + 1) % 6])
    cx = sum(x for x, _ in pts) / 6
    cy = sum(y for _, y in pts) / 6
    for k in range(6):
        x, y = pts[k]
        prev_y, next_y = pts[k - 1][1], pts[(k + 1) % 6][1]
        up = y < (prev_y + next_y) / 2              # vertex above its neighbours
        ax = doc.add_atom(x * bl, (y + (-0.75 if up else 0.75)) * bl)
        doc.add_bond(ids[k], ax.id)
        p1, p2 = pts[(k + 1) % 6], pts[(k + 2) % 6]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        L = math.hypot(dx, dy) or 1.0
        dx, dy = dx / L, dy / L
        if (x - cx) * dx + (y - cy) * dy < 0:
            dx, dy = -dx, -dy
        eq = doc.add_atom((x + 0.75 * dx) * bl, (y + 0.75 * dy) * bl)
        doc.add_bond(ids[k], eq.id)


def _boat(doc, bl) -> None:
    pts = [(1.3, -0.55), (0.65, 0.35), (-0.65, 0.35), (-1.3, -0.55), (-0.6, -0.15),
           (0.6, -0.15)]
    ids = _atoms(doc, pts, bl)
    for k in range(6):
        doc.add_bond(ids[k], ids[(k + 1) % 6])


DRAWN = {
    "newman_staggered": lambda d, bl: _newman(d, bl, False),
    "newman_eclipsed": lambda d, bl: _newman(d, bl, True),
    "fischer1": lambda d, bl: _fischer(d, bl, [True]),
    "fischer2": lambda d, bl: _fischer(d, bl, [False, True]),
    "fischer_glucose": lambda d, bl: _fischer(d, bl, [True, False, True, True]),
    "haworth_pyranose": lambda d, bl: _haworth(
        d, bl, [(0.95, -0.7, "O"), (1.9, 0.05, ""), (1.0, 0.8, ""), (-0.65, 0.8, ""),
                (-1.55, 0.05, ""), (-0.65, -0.7, "")], {1, 2, 3},
        [(1, True, "OH"), (2, False, "OH"), (3, True, "OH"), (4, False, "OH"),
         (5, True, "CH2OH")]),
    "haworth_furanose": lambda d, bl: _haworth(
        d, bl, [(0.0, -0.6, "O"), (1.2, -0.05, ""), (0.75, 0.7, ""), (-0.75, 0.7, ""),
                (-1.2, -0.05, "")], {1, 2, 3},
        [(1, True, "OH"), (2, False, "OH"), (3, False, "OH"), (4, True, "CH2OH", -0.6)]),
    "chair_axial": _chair_axial,
    "boat": _boat,
}


def build_drawn(text: str, style=None):
    """Document holding a drawn template (``@tpl:<key>`` or ``@clip:<name>``)."""
    from rsstatchem.core.chem_draw.clipart import CLIPART, default_size
    from rsstatchem.core.chem_draw.model import Document
    from rsstatchem.core.chem_draw.style import Style
    doc = Document((style or Style()).updated())
    bl = doc.style.bond_length
    kind, _, key = text.partition(":")
    if kind == "@clip":
        if key not in CLIPART:
            raise KeyError(key)
        w, h = default_size(key, bl)
        doc.add_clipart(-w / 2, -h / 2, key, w=w, h=h)
    else:
        DRAWN[key](doc, bl)
    return doc


def insert_drawn(doc, text: str, x: float, y: float) -> list:
    """Paste a drawn template centred on (x, y); returns the new keys."""
    sub = build_drawn(text, doc.style)
    bb = sub.bbox()
    dx, dy = (x - (bb[0] + bb[2]) / 2, y - (bb[1] + bb[3]) / 2) if bb else (x, y)
    return doc.paste(sub, dx, dy)

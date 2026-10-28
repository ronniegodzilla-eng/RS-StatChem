"""Drawing <-> chemistry, built on the stage-1 RDKit bridge of :mod:`rsstatchem.core.chem.molecule`
(``AtomSpec`` / ``BondSpec`` / ``mol_from_graph`` / ``graph_from_mol``).

- Nicknames are expanded to their full fragments (coordinates included, so stereo from
  wedges next to a nickname still works); generic labels (R, X, R1) become dummy atoms.
- Stereo: wedge / hash bonds give tetrahedral centres, the 2D geometry gives E/Z.
- Reactions: the drawing is split into reactants / agents / products by the position of
  the reaction arrow (ChemDraw "Reaction interpretation"); export as reaction SMILES / RXN.
- Import: SMILES, MOL/SDF, InChI, reaction SMILES and RXN text are placed into a document
  at the document's fixed bond length.
- Clean Structure: re-layout with ``compute_2d`` keeping stereo (wedges are re-assigned).
- Enhanced stereochemistry (ABS / AND / OR groups) read from MOL V3000 / CXSMILES is kept in
  ``doc.meta["stereo_groups"]`` and written back to MOL / RXN V3000.
- Clean Up Reaction (ChemDraw): even spacing of the components on one line, "+" signs
  between them, an arrow long enough for its conditions.
- HELM strings are recognised on insert (see :mod:`.sequence`).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import lru_cache

from rdkit import Chem
from rdkit.Chem import rdChemReactions

from rsstatchem.core.chem import molecule as mo
from rsstatchem.core.chem.elements import is_symbol
from rsstatchem.core.chem_draw import abbrev
from rsstatchem.core.chem_draw.labels import approx_extent, implicit_h
from rsstatchem.core.chem_draw.model import REACTION_ARROWS, Document

__all__ = ["ANGSTROM", "to_specs", "to_mol", "fragment_mols", "structure_problems",
           "to_smiles", "to_molblock", "to_sdf", "to_inchi", "to_inchikey", "insert_mol",
           "insert_text", "insert_reaction", "split_reaction", "ReactionSplit",
           "reaction_smiles", "reaction_rxn", "reaction_object", "clean_structure",
           "expand_label", "apply_label_text", "mols_from_text", "stereo_groups",
           "set_stereo_groups", "clean_up_reaction", "reaction_components"]

ANGSTROM = 1.5          # RDKit 2D bond length; one document bond length maps onto it


def _scale(doc: Document) -> float:
    return ANGSTROM / (doc.style.bond_length or 30.0)


def _median_bond(mol: Chem.Mol) -> float:
    """Median 2D bond length (CoordGen draws 1.0, the classic depictor 1.5)."""
    if mol.GetNumBonds() == 0 or mol.GetNumConformers() == 0:
        return ANGSTROM
    conf = mol.GetConformer()
    ls = sorted((conf.GetAtomPosition(b.GetBeginAtomIdx()) -
                 conf.GetAtomPosition(b.GetEndAtomIdx())).Length() for b in mol.GetBonds())
    return ls[len(ls) // 2] or ANGSTROM


@lru_cache(maxsize=None)
def _nickname_fragment(name: str):
    """(atoms [(symbol, charge, x, y)], bonds [(i, j, order)], attach index, dir angle)."""
    ab = abbrev.ABBREVIATIONS[name]
    mol = Chem.MolFromSmiles(ab.smiles)
    mo.compute_2d(mol)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    dummy = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0)
    attach = mol.GetAtomWithIdx(dummy).GetNeighbors()[0].GetIdx()
    conf = mol.GetConformer()
    f = ANGSTROM / _median_bond(mol)
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (p.x * f, p.y * f, 0.0))
    keep = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() != dummy]
    index = {old: new for new, old in enumerate(keep)}
    atoms = []
    for i in keep:
        a = mol.GetAtomWithIdx(i)
        p = conf.GetAtomPosition(i)
        atoms.append((a.GetSymbol(), a.GetFormalCharge(), p.x, p.y))
    bonds = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if dummy in (i, j):
            continue
        bonds.append((index[i], index[j], {Chem.BondType.DOUBLE: 2,
                                            Chem.BondType.TRIPLE: 3}.get(b.GetBondType(), 1)))
    pa, pd = conf.GetAtomPosition(attach), conf.GetAtomPosition(dummy)
    return atoms, bonds, index[attach], math.atan2(pd.y - pa.y, pd.x - pa.x)


def to_specs(doc: Document, atom_ids=None, expand: bool = True):
    """(AtomSpec list, BondSpec list, origin) for the given atoms; ``origin[i]`` is the
    document atom id each spec atom comes from (expanded nickname atoms point to their
    label atom)."""
    ids = sorted(doc.atoms if atom_ids is None else atom_ids)
    idset = set(ids)
    k = _scale(doc)
    specs: list[mo.AtomSpec] = []
    origin: list[int] = []
    index: dict[int, int] = {}
    bonds: list[mo.BondSpec] = []
    nick_atoms = []
    for aid in ids:
        a = doc.atoms[aid]
        x, y = a.x * k, -a.y * k
        nick = abbrev.lookup(a.label) if a.label else None
        if nick is not None and expand:
            nick_atoms.append((aid, nick))
            index[aid] = -1
            continue
        if a.label and nick is None:
            spec = mo.AtomSpec("*", x, y, a.charge, 0, None, 0, a.map_number, a.label)
        else:
            sym, iso = a.symbol, a.isotope
            if sym in ("D", "T"):
                sym, iso = "H", (2 if a.symbol == "D" else 3)
            elif nick is not None:   # not expanded: placeholder for layouts
                sym = "*"
            spec = mo.AtomSpec(sym, x, y, a.charge, iso, a.explicit_h, a.radicals,
                               a.map_number, a.label if nick is not None else "")
        index[aid] = len(specs)
        specs.append(spec)
        origin.append(aid)
    for aid, nick in nick_atoms:
        a = doc.atoms[aid]
        frag_atoms, frag_bonds, attach, dir_angle = _nickname_fragment(nick.name)
        nbrs = [n for n in doc.neighbors(aid) if n in idset]
        px, py = a.x * k, -a.y * k
        if nbrs:
            mx = sum(doc.atoms[n].x * k for n in nbrs) / len(nbrs)
            my = sum(-doc.atoms[n].y * k for n in nbrs) / len(nbrs)
            want = math.atan2(my - py, mx - px)
        else:
            want = math.pi
        rot = want - dir_angle
        c, s = math.cos(rot), math.sin(rot)
        ax, ay = frag_atoms[attach][2], frag_atoms[attach][3]
        base = len(specs)
        for j, (sym, q, fx, fy) in enumerate(frag_atoms):
            rx, ry = fx - ax, fy - ay
            specs.append(mo.AtomSpec(sym, px + c * rx - s * ry, py + s * rx + c * ry, q,
                                     map_number=a.map_number if j == attach else 0))
            origin.append(aid)
        for i, j, order in frag_bonds:
            bonds.append(mo.BondSpec(base + i, base + j, order))
        index[aid] = base + attach
    for b in doc.bonds.values():
        if b.a not in idset or b.b not in idset:
            continue
        stereo = ""
        order = b.order
        if b.order == 1 and b.style in ("wedge", "hash", "wavy"):
            stereo = b.style
        elif b.order == 1 and b.style == "hollow":     # hollow wedge: also "up"
            stereo = "wedge"
        elif b.order == 2 and b.style == "either":
            stereo = "either"
        if b.style == "dative":
            order = 0                                   # RDKit dative bond (a -> b)
        bonds.append(mo.BondSpec(index[b.a], index[b.b], order, stereo))
    return specs, bonds, origin


def to_mol(doc: Document, atom_ids=None, sanitize: bool = True, expand: bool = True,
           remove_hs: bool = True) -> Chem.Mol:
    """RDKit molecule of the drawing (or of the given atoms). Raises ``ChemError``."""
    specs, bonds, origin = to_specs(doc, atom_ids, expand)
    if not specs:
        raise mo.ChemError("Nothing drawn")
    mol = mo.mol_from_graph(specs, bonds, sanitize=sanitize)
    for i, aid in enumerate(origin):
        mol.GetAtomWithIdx(i).SetIntProp("doc_id", aid)
    if sanitize and doc.meta.get("stereo_groups"):
        mol = _apply_stereo_groups(doc, mol)
    if sanitize and remove_hs and any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
        try:
            ps = Chem.RemoveHsParameters()
            ps.removeOnlyHNeighbors = False      # keep H2
            ps.removeDegreeZero = False          # keep a lone H atom / H+
            ps.removeWithWedgedBond = True
            ps.removeMapped = False
            mol = Chem.RemoveHs(mol, ps)
        except Exception:  # noqa: BLE001 - keep explicit H atoms if they cannot go
            pass
    return mol


def fragment_mols(doc: Document, atom_ids=None) -> list[tuple[list[int], object]]:
    """[(atom ids, Mol or ChemError)] for each molecule of the drawing."""
    out = []
    sel = None if atom_ids is None else set(atom_ids)
    for comp in doc.fragments():
        if sel is not None:
            comp = [a for a in comp if a in sel]
            if not comp:
                continue
        try:
            out.append((comp, to_mol(doc, comp)))
        except mo.ChemError as exc:
            out.append((comp, exc))
    return out


def structure_problems(doc: Document) -> dict[int, str]:
    """{document atom id: message} for valence problems and unknown labels."""
    out: dict[int, str] = {}
    for aid, a in doc.atoms.items():
        if a.label and not abbrev.is_abbreviation(a.label):
            continue
        if not a.label and a.symbol not in ("D", "T") and not is_symbol(a.symbol):
            out[aid] = f"Unknown element '{a.symbol}'"
    for comp in doc.fragments():
        try:
            specs, bonds, origin = to_specs(doc, comp)
            mol = mo.mol_from_graph(specs, bonds, sanitize=False)
        except Exception as exc:  # noqa: BLE001
            for aid in comp:
                out.setdefault(aid, str(exc))
            continue
        for idx, msg in mo.problems(mol):
            if 0 <= idx < len(origin):
                out.setdefault(origin[idx], msg)
            elif idx < 0 and comp:
                out.setdefault(comp[0], msg)
    return out


_GROUP_TYPES = {"ABS": Chem.StereoGroupType.STEREO_ABSOLUTE,
                "AND": Chem.StereoGroupType.STEREO_AND, "OR": Chem.StereoGroupType.STEREO_OR}


def stereo_groups(doc: Document) -> list[dict]:
    """Enhanced stereo groups of the drawing: [{"type": "ABS"|"AND"|"OR", "id": n,
    "atoms": [document atom ids]}] (ChemDraw's Structure > Enhanced Stereochemistry)."""
    out = []
    for g in doc.meta.get("stereo_groups") or []:
        atoms = [a for a in g.get("atoms", []) if a in doc.atoms]
        if atoms and g.get("type") in _GROUP_TYPES:
            out.append({"type": g["type"], "id": int(g.get("id", 0)), "atoms": atoms})
    return out


def set_stereo_groups(doc: Document, groups: list[dict]) -> None:
    doc.meta["stereo_groups"] = [dict(g, atoms=list(g["atoms"])) for g in groups]


def _apply_stereo_groups(doc: Document, mol: Chem.Mol) -> Chem.Mol:
    index: dict[int, int] = {}
    for a in mol.GetAtoms():
        if a.HasProp("doc_id"):
            index.setdefault(a.GetIntProp("doc_id"), a.GetIdx())
    rw = Chem.RWMol(mol)
    groups = []
    for g in stereo_groups(doc):
        idx = [index[a] for a in g["atoms"] if a in index and
               rw.GetAtomWithIdx(index[a]).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED]
        if idx:
            try:
                groups.append(Chem.CreateStereoGroup(_GROUP_TYPES[g["type"]], rw, idx,
                                                     g["id"]))
            except Exception:  # noqa: BLE001 - older RDKit signature
                groups.append(Chem.CreateStereoGroup(_GROUP_TYPES[g["type"]], rw, idx))
    if not groups:
        return mol
    rw.SetStereoGroups(groups)
    return rw.GetMol()


def to_smiles(doc: Document, atom_ids=None) -> str:
    return mo.to_smiles(to_mol(doc, atom_ids))


def to_molblock(doc: Document, atom_ids=None, v3000: bool = False) -> str:
    """MOL block (V2000, or V3000 with enhanced stereo collections when ``v3000``)."""
    m = _clean_props(to_mol(doc, atom_ids))
    return Chem.MolToV3KMolBlock(m) if v3000 else Chem.MolToMolBlock(m)


def to_sdf(doc: Document, atom_ids=None, v3000: bool = False) -> str:
    out = []
    for comp, m in fragment_mols(doc, atom_ids):
        if isinstance(m, Exception):
            raise m
        m = _clean_props(m)
        out.append((Chem.MolToV3KMolBlock(m) if v3000 else Chem.MolToMolBlock(m)) +
                   "$$$$\n")
    return "".join(out)


def to_inchi(doc: Document, atom_ids=None) -> str:
    return mo.to_inchi(to_mol(doc, atom_ids))


def to_inchikey(doc: Document, atom_ids=None) -> str:
    return mo.to_inchikey(to_mol(doc, atom_ids))


def _clean_props(mol: Chem.Mol) -> Chem.Mol:
    m = Chem.Mol(mol)
    for a in m.GetAtoms():
        if a.HasProp("doc_id"):
            a.ClearProp("doc_id")
    return m


# --- import -------------------------------------------------------------------------------

def insert_mol(doc: Document, mol: Chem.Mol, x: float, y: float,
               keep_coords: bool = True) -> list[tuple[str, int]]:
    """Place a molecule centred on (x, y) at the document bond length; returns new keys."""
    m = Chem.Mol(mol)
    if not keep_coords or not mo.has_2d(m) or not _finite(m):
        mo.compute_2d(m)
        if not _finite(m):                      # CoordGen fails on e.g. H2
            mo.compute_2d(m, coordgen=False)
    atoms, bonds = mo.graph_from_mol(m)
    if not atoms:
        return []
    lengths = sorted(math.hypot(atoms[b.begin].x - atoms[b.end].x,
                                atoms[b.begin].y - atoms[b.end].y) for b in bonds)
    ref = lengths[len(lengths) // 2] if lengths else ANGSTROM
    k = doc.style.bond_length / (ref or ANGSTROM)
    xs = [a.x for a in atoms]
    ys = [a.y for a in atoms]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    ids, keys = [], []
    for a in atoms:
        sym = a.symbol if a.symbol != "*" else "C"
        label = (a.label or "R") if a.symbol == "*" else ""
        na = doc.add_atom(x + (a.x - cx) * k, y - (a.y - cy) * k, sym, charge=a.charge,
                          isotope=a.isotope, radicals=a.radicals, map_number=a.map_number,
                          label=label)
        ids.append(na.id)
        keys.append(("atom", na.id))
    explicit_any = {bd.GetIdx() for bd in m.GetBonds()
                    if bd.GetStereo() == Chem.BondStereo.STEREOANY or
                    bd.GetBondDir() == Chem.BondDir.EITHERDOUBLE}
    for idx, b in enumerate(bonds):
        stereo = b.stereo if b.stereo in ("wedge", "hash", "wavy", "either") else ""
        # unspecified double bonds are drawn plainly (as ChemDraw does); only an explicit
        # "either" double bond becomes a crossed bond
        if stereo == "either" and (b.order != 2 or idx not in explicit_any):
            stereo = ""
        if b.order == 0:
            nb = doc.add_bond(ids[b.begin], ids[b.end], 1, "dative")
        else:
            nb = doc.add_bond(ids[b.begin], ids[b.end], b.order, stereo)
        keys.append(("bond", nb.id))
    for a, aid in zip(atoms, ids):
        if a.explicit_h is not None and a.symbol != "*":
            na = doc.atoms[aid]
            if implicit_h(doc, na) != a.explicit_h:
                na.explicit_h = a.explicit_h
    _keep_stereo_groups(doc, m, ids)
    return keys


def _keep_stereo_groups(doc: Document, mol: Chem.Mol, ids: list[int]) -> None:
    """Enhanced stereo groups of an inserted molecule, as document atom ids in doc.meta."""
    try:
        groups = list(mol.GetStereoGroups())
    except Exception:  # noqa: BLE001
        return
    if not groups:
        return
    names = {v: k for k, v in _GROUP_TYPES.items()}
    have = stereo_groups(doc)
    used = {(g["type"], g["id"]) for g in have}
    for g in groups:
        kind = names.get(g.GetGroupType())
        if kind is None:
            continue
        atoms = [ids[a.GetIdx()] for a in g.GetAtoms() if a.GetIdx() < len(ids)]
        try:
            num = int(g.GetReadId()) or 1
        except Exception:  # noqa: BLE001
            num = 1
        if kind != "ABS":
            while (kind, num) in used:
                num += 1
        used.add((kind, num))
        have.append({"type": kind, "id": 0 if kind == "ABS" else num, "atoms": atoms})
    set_stereo_groups(doc, have)


def _finite(m: Chem.Mol) -> bool:
    if m.GetNumConformers() == 0:
        return False
    return all(math.isfinite(v) for p in m.GetConformer().GetPositions() for v in p)


def mols_from_text(text: str) -> list[Chem.Mol]:
    """Molecules from SMILES (one per line, '.'-separated kept together), MOL, SDF or
    InChI text."""
    t = text.strip()
    fmt = mo.detect_format(t)
    if fmt == "sdf":
        blocks = re.split(r"\$\$\$\$[^\n]*\n?", text)
        return [mo.mol_from_text(_fix_molblock(b), "molblock") for b in blocks
                if "V2000" in b or "V3000" in b]
    if fmt == "molblock":
        return [mo.mol_from_text(_fix_molblock(text), "molblock")]
    if fmt == "inchi":
        return [mo.mol_from_text(t, fmt)]
    out = []
    for line in t.splitlines():
        if line.strip() and not line.strip().startswith("#"):
            out.append(mo.mol_from_text(line.strip(), "smiles"))
    if not out:
        raise mo.ChemError("No structure found")
    return out


def _fix_molblock(text: str) -> str:
    """Re-attach exactly three header lines before the counts line (pasted MOL blocks
    often lose a blank title line or gain extra blank lines)."""
    lines = text.splitlines()
    idx = next((i for i, ln in enumerate(lines) if "V2000" in ln or "V3000" in ln), None)
    if idx is None:
        return text
    head = lines[max(0, idx - 3):idx]
    head = [""] * (3 - len(head)) + head
    return "\n".join(head + lines[idx:]) + "\n"


def insert_text(doc: Document, text: str, x: float, y: float) -> list[tuple[str, int]]:
    """Insert SMILES / MOL / SDF / InChI / reaction SMILES / RXN text at (x, y)."""
    t = (text or "").strip()
    if not t:
        raise mo.ChemError("Empty structure")
    if t.startswith("$RXN") or mo.detect_format(t) == "reaction":
        return insert_reaction(doc, reaction_object(t), x, y)
    from rsstatchem.core.chem_draw import sequence
    if sequence.looks_like_helm(t):
        return insert_mol(doc, sequence.mol_from_helm(t), x, y)
    mols = mols_from_text(text)
    if len(mols) == 1:
        return insert_mol(doc, mols[0], x, y)
    keys: list = []
    cursor = x
    for m in mols:
        keys += _place_left(doc, m, cursor, y)
        cursor = _right_edge(doc, keys) + doc.style.bond_length * 1.2
    return keys


def _pad(doc: Document) -> float:
    return max(doc.style.bond_length * 0.3, doc.style.font_size * 0.5)


def _edges(doc: Document, keys) -> tuple[float, float]:
    lo, hi = math.inf, -math.inf
    for k, i in keys:
        if k == "atom" and i in doc.atoms:
            a = doc.atoms[i]
            left, right = approx_extent(doc, a, doc.style.font_size)
            lo, hi = min(lo, a.x - left), max(hi, a.x + right)
    return (lo, hi) if lo <= hi else (0.0, 0.0)


def _right_edge(doc: Document, keys) -> float:
    return _edges(doc, keys)[1] + _pad(doc)


def _place_left(doc: Document, mol: Chem.Mol, left: float, y: float) -> list:
    """Insert ``mol`` so that its left edge (labels included) is at ``left``."""
    m = Chem.Mol(mol)
    m.UpdatePropertyCache(strict=False)
    keys = insert_mol(doc, m, 0.0, y)
    if keys:
        doc.move(keys, left + _pad(doc) - _edges(doc, keys)[0], 0.0)
    return keys


def reaction_object(text: str):
    t = text.strip()
    if t.startswith("$RXN"):
        rxn = rdChemReactions.ReactionFromRxnBlock(t)
        if rxn is None:
            raise mo.ChemError("Invalid RXN block")
        for tmpl in list(rxn.GetReactants()) + list(rxn.GetProducts()):
            try:
                tmpl.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(tmpl, catchErrors=True)
            except Exception:  # noqa: BLE001
                pass
        return rxn
    return mo.reaction_from_text(t)


def insert_reaction(doc: Document, rxn, x: float, y: float) -> list[tuple[str, int]]:
    """Lay out reactants + ... -> products (agents above the arrow) starting at x."""
    bl = doc.style.bond_length
    keys: list = []
    cursor = x

    def place(mols):
        nonlocal cursor
        for i, m in enumerate(mols):
            new = _place_left(doc, m, cursor, y)
            keys.extend(new)
            cursor = _right_edge(doc, new)
            if i < len(mols) - 1:
                cursor += bl * 0.5
                keys.append(("plus", doc.add_plus(cursor, y).id))
                cursor += bl * 0.5
    place(list(rxn.GetReactants()))
    cursor += bl * 0.5
    start = cursor
    agent_keys: list = []
    ax = start + bl * 0.3
    for m in rxn.GetAgents():
        new = _place_left(doc, m, ax, y - bl * 2.0)
        agent_keys += new
        ax = _right_edge(doc, new) + bl * 0.3
    alen = max(bl * 3.5, ax - start + bl * 0.3)
    keys.append(("arrow", doc.add_arrow(start, y, start + alen, y, "forward").id))
    keys += agent_keys
    cursor = start + alen + bl * 0.5
    place(list(rxn.GetProducts()))
    return keys


# --- reactions ------------------------------------------------------------------------------

@dataclass
class ReactionSplit:
    arrow: object = None
    reactants: list = field(default_factory=list)   # lists of atom ids
    agents: list = field(default_factory=list)
    products: list = field(default_factory=list)
    retro: bool = False
    notes: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.arrow is not None and bool(self.reactants) and bool(self.products)


def split_reaction(doc: Document) -> ReactionSplit:
    """Assign each molecule to reactants / agents / products from the main arrow: before
    the tail -> reactants, beyond the head -> products, alongside the arrow -> agents.
    Retrosynthetic arrows (target => precursors) are read backwards."""
    arrows = sorted((a for a in doc.arrows.values() if a.kind in REACTION_ARROWS),
                    key=lambda a: min(a.x1, a.x2))
    res = ReactionSplit()
    if not arrows:
        res.notes.append("No reaction arrow in the drawing.")
        return res
    ar = arrows[0]
    if len(arrows) > 1:
        res.notes.append("Several arrows found; the first (leftmost) step is used.")
    res.arrow = ar
    dx, dy = ar.x2 - ar.x1, ar.y2 - ar.y1
    L = math.hypot(dx, dy) or 1.0
    ux, uy = dx / L, dy / L
    for comp in doc.fragments():
        cx = sum(doc.atoms[i].x for i in comp) / len(comp)
        cy = sum(doc.atoms[i].y for i in comp) / len(comp)
        t = (cx - ar.x1) * ux + (cy - ar.y1) * uy
        if t < 0:
            res.reactants.append(comp)
        elif t > L:
            res.products.append(comp)
        else:
            res.agents.append(comp)
    if ar.kind == "retro":
        res.retro = True
        res.reactants, res.products = res.products, res.reactants
    return res


def reaction_smiles(doc: Document, split: ReactionSplit | None = None,
                    agents: bool = True) -> str:
    sp = split or split_reaction(doc)
    if not sp.ok:
        raise mo.ChemError("Draw reactants, a reaction arrow and products first.")

    def side(comps):
        return ".".join(to_smiles(doc, c) for c in comps)
    return f"{side(sp.reactants)}>{side(sp.agents) if agents else ''}>{side(sp.products)}"


def reaction_rxn(doc: Document, split: ReactionSplit | None = None,
                 v3000: bool = False) -> str:
    rxn = _rxn(doc, split)
    if v3000:
        return rdChemReactions.ReactionToV3KRxnBlock(rxn)
    return rdChemReactions.ReactionToRxnBlock(rxn)


def _rxn(doc: Document, split: ReactionSplit | None = None):
    sp = split or split_reaction(doc)
    if not sp.ok:
        raise mo.ChemError("Draw reactants, a reaction arrow and products first.")
    rxn = rdChemReactions.ChemicalReaction()
    for c in sp.reactants:
        rxn.AddReactantTemplate(_clean_props(to_mol(doc, c)))
    for c in sp.agents:
        rxn.AddAgentTemplate(_clean_props(to_mol(doc, c)))
    for c in sp.products:
        rxn.AddProductTemplate(_clean_props(to_mol(doc, c)))
    return rxn


# --- editing helpers ------------------------------------------------------------------------

def apply_label_text(doc: Document, aid: int, text: str) -> None:
    """Set an atom from typed text: element symbol, element + H ("OH", "NH2"), nickname
    ("OMe", "Ph", "CO2H") or a generic label ("R1")."""
    a = doc.atoms[aid]
    t = (text or "").strip()
    if not t:
        return
    a.label, a.explicit_h, a.show_carbon = "", None, False
    if t in ("D", "T"):
        a.symbol = t
        return
    eh = abbrev.element_with_h(t, is_symbol)
    if t in abbrev.ABBREVIATIONS and (eh is None or eh[1] is None):
        eh = None                       # "Ac", "Pr", "Ts" are nicknames, not elements
    if eh is not None:
        sym, nh = eh
        a.symbol = sym
        if sym == "C" and nh is not None:
            a.show_carbon = True
        if nh is not None:
            a.explicit_h = None
        return
    if abbrev.is_abbreviation(t):
        a.symbol, a.label = "C", abbrev.lookup(t).name
        return
    a.symbol, a.label = "C", t


def expand_label(doc: Document, aid: int) -> list[int]:
    """Replace a nickname atom by its atoms (ChemDraw "Expand Label"); new atom ids."""
    a = doc.atoms[aid]
    nick = abbrev.lookup(a.label) if a.label else None
    if nick is None:
        return []
    frag_atoms, frag_bonds, attach, dir_angle = _nickname_fragment(nick.name)
    bl = doc.style.bond_length
    k = bl / ANGSTROM
    nbrs = doc.neighbors(aid)
    if nbrs:
        mx = sum(doc.atoms[n].x for n in nbrs) / len(nbrs)
        my = sum(doc.atoms[n].y for n in nbrs) / len(nbrs)
        want = math.atan2(-(my - a.y), mx - a.x)
    else:
        want = math.pi
    rot = want - dir_angle
    c, s = math.cos(rot), math.sin(rot)
    ax, ay = frag_atoms[attach][2], frag_atoms[attach][3]
    new = []
    for j, (sym, q, fx, fy) in enumerate(frag_atoms):
        rx, ry = fx - ax, fy - ay
        x, y = c * rx - s * ry, s * rx + c * ry
        if j == attach:
            a.symbol, a.label, a.charge = sym, "", q
            new.append(aid)
        else:
            new.append(doc.add_atom(a.x + x * k, a.y - y * k, sym, charge=q).id)
    for i, j, order in frag_bonds:
        doc.add_bond(new[i], new[j], order)
    return new


def clean_structure(doc: Document, atom_ids=None) -> bool:
    """Regenerate 2D coordinates of the molecules containing ``atom_ids`` (all when None),
    keeping their centres and stereochemistry. Returns False if nothing could be done."""
    comps = doc.fragments()
    if atom_ids is not None:
        sel = set(atom_ids)
        comps = [c for c in comps if sel & set(c)]
    done = False
    bl = doc.style.bond_length
    for comp in comps:
        if len(comp) < 2:
            continue
        specs, bonds, origin = to_specs(doc, comp, expand=False)
        try:
            mol = mo.mol_from_graph(specs, bonds, sanitize=True)
            stereo_ok = True
        except mo.ChemError:
            mol = mo.mol_from_graph(specs, bonds, sanitize=False)
            mol.UpdatePropertyCache(strict=False)
            Chem.FastFindRings(mol)
            stereo_ok = False
        cx0 = sum(doc.atoms[i].x for i in comp) / len(comp)
        cy0 = sum(doc.atoms[i].y for i in comp) / len(comp)
        mo.compute_2d(mol)
        conf = mol.GetConformer()
        pos = [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
        k = bl / _median_bond(mol)
        mx = sum(p.x for p in pos) / len(pos)
        my = sum(p.y for p in pos) / len(pos)
        for i, aid in enumerate(origin):
            doc.atoms[aid].x = cx0 + (pos[i].x - mx) * k
            doc.atoms[aid].y = cy0 - (pos[i].y - my) * k
        if stereo_ok:
            _rewedge(doc, comp, mol, origin)
        done = True
    return done


def _rewedge(doc: Document, comp, mol: Chem.Mol, origin) -> None:
    m = Chem.Mol(mol)
    for b in m.GetBonds():
        if b.GetBondDir() in (Chem.BondDir.BEGINWEDGE, Chem.BondDir.BEGINDASH):
            b.SetBondDir(Chem.BondDir.NONE)
    Chem.WedgeMolBonds(m, m.GetConformer())
    sel = set(comp)
    for b in doc.bonds.values():
        if b.a in sel and b.style in ("wedge", "hash"):
            b.style = ""
    for b in m.GetBonds():
        d = b.GetBondDir()
        if d not in (Chem.BondDir.BEGINWEDGE, Chem.BondDir.BEGINDASH):
            continue
        i, j = origin[b.GetBeginAtomIdx()], origin[b.GetEndAtomIdx()]
        db = doc.bond_between(i, j)
        if db is None:
            continue
        if db.a != i:
            db.a, db.b = i, j
        db.style = "wedge" if d == Chem.BondDir.BEGINWEDGE else "hash"


# --- Clean Up Reaction ---------------------------------------------------------------------

def reaction_components(doc: Document, split: ReactionSplit | None = None) -> dict:
    """{"reactants" | "agents" | "products": [set of keys per component]} of the main
    reaction, each component with its captions (text boxes just below or above it)."""
    sp = split or split_reaction(doc)
    out = {"reactants": [], "agents": [], "products": []}
    if not sp.ok:
        return out
    bl = doc.style.bond_length
    taken: set[int] = set()
    for side in ("reactants", "agents", "products"):
        for comp in getattr(sp, side):
            keys = doc.with_bonds({("atom", a) for a in comp})
            bb = doc.bbox(keys)
            if bb is not None and side != "agents":
                for t in doc.texts.values():
                    if t.id in taken:
                        continue
                    if bb[0] - bl <= t.x <= bb[2] + bl and \
                            bb[3] - bl * 0.2 <= t.y <= bb[3] + bl * 2.5:
                        keys.add(("text", t.id))
                        taken.add(t.id)
            out[side].append(keys)
    return out


def clean_up_reaction(doc: Document) -> bool:
    """ChemDraw's Structure > Clean Up Reaction for a single-step reaction with a straight
    arrow: the components are spread evenly on the arrow's line, "+" signs are put between
    reactants and between products (old ones removed) and the arrow is made long enough for
    the conditions drawn on it. Returns False when there is no interpretable reaction."""
    sp = split_reaction(doc)
    if not sp.ok or abs(getattr(sp.arrow, "bend", 0.0)) > 1e-6:
        return False
    comps = reaction_components(doc, sp)
    if sp.retro:
        comps["reactants"], comps["products"] = comps["products"], comps["reactants"]
    ar = sp.arrow
    bl = doc.style.bond_length
    gap = bl * 1.6
    y = (ar.y1 + ar.y2) / 2
    old_plus = [("plus", p.id) for p in doc.pluses.values()
                if abs(p.y - y) < bl * 2.5]
    doc.remove(old_plus)

    def place(units, cursor):
        for i, keys in enumerate(units):
            bb = doc.bbox(keys)
            cy = _atom_center_y(doc, keys, bb)
            doc.move(keys, cursor - bb[0], y - cy)
            cursor += bb[2] - bb[0]
            if i < len(units) - 1:
                doc.add_plus(cursor + gap / 2, y)
                cursor += gap
        return cursor
    left = min((doc.bbox(k)[0] for k in comps["reactants"]), default=ar.x1)
    cursor = place(comps["reactants"], left)
    start = cursor + gap * 0.6
    agent_w = sum(doc.bbox(k)[2] - doc.bbox(k)[0] for k in comps["agents"]) + \
        bl * 0.6 * max(0, len(comps["agents"]) - 1)
    text_w = max(len(ar.above or ""), len(ar.below or "")) * doc.style.caption_size * 0.55
    length = max(bl * 3.5, agent_w + bl, text_w + bl)
    ax = start + (length - agent_w) / 2
    for keys in comps["agents"]:
        bb = doc.bbox(keys)
        doc.move(keys, ax - bb[0], (y - bl * 0.6) - bb[3])
        ax += bb[2] - bb[0] + bl * 0.6
    ar.x1, ar.y1, ar.x2, ar.y2 = start, y, start + length, y      # tail -> head, rightwards
    place(comps["products"], start + length + gap * 0.6)
    return True


def _atom_center_y(doc: Document, keys, bb) -> float:
    ys = [doc.atoms[i].y for k, i in keys if k == "atom" and i in doc.atoms]
    return (min(ys) + max(ys)) / 2 if ys else (bb[1] + bb[3]) / 2

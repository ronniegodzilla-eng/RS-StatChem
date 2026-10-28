"""CIP stereo descriptors of a drawing (ChemDraw's *Object > Show Stereochemistry*, 17.1 guide
pp. 193-195) as plain data; the renderer (WP-B) decides how to draw them.

``labels(doc, atoms=None)`` returns ``{("atom", id): "R" | "S" | "r" | "s",
("bond", id): "E" | "Z"}`` for every stereocentre and stereogenic double bond whose
configuration is defined by the drawing (wedges / hashes for centres, the 2D geometry for
double bonds). Labels come from RDKit's new CIP labeller (``rdCIPLabeler``, an implementation
of the IUPAC 2013 rules after Hanson et al., J. Chem. Inf. Model. 58 (2018) 1755), which also
gives the lower-case pseudo-asymmetric descriptors r / s (e.g. cis- and trans-decalin,
cis-1,4-dimethylcyclohexane).

Not available: M / P for allenes and atropisomers (RDKit's labeller does not assign them), and
centres inside a contracted nickname label (Ph, OAc ...), which have no atom on the canvas.
Undefined centres (plain bonds) get no label - ChemDraw then shows nothing too.
No Qt here.
"""
from __future__ import annotations

from rsstatchem.core.chem import molecule as mo
from rsstatchem.core.chem_draw import convert
from rsstatchem.core.chem_draw.model import Document

__all__ = ["labels", "mol_labels", "format_label", "ATOM_LABELS", "BOND_LABELS"]

ATOM_LABELS = ("R", "S", "r", "s")
BOND_LABELS = ("E", "Z")


def mol_labels(mol) -> tuple[dict[int, str], dict[int, str]]:
    """({atom index: label}, {bond index: label}) of an RDKit molecule (CIP, IUPAC 2013).
    The molecule is copied; it must carry its stereo (chiral tags / bond stereo)."""
    from rdkit import Chem
    from rdkit.Chem import rdCIPLabeler

    m = Chem.Mol(mol)
    for obj in list(m.GetAtoms()) + list(m.GetBonds()):    # drop legacy-perception labels
        if obj.HasProp("_CIPCode"):
            obj.ClearProp("_CIPCode")
    try:
        rdCIPLabeler.AssignCIPLabels(m)
    except Exception:  # noqa: BLE001 - very large / odd molecules: no labels rather than a crash
        return {}, {}
    atoms = {a.GetIdx(): a.GetProp("_CIPCode") for a in m.GetAtoms()
             if a.HasProp("_CIPCode") and a.GetProp("_CIPCode") in ATOM_LABELS}
    bonds = {b.GetIdx(): b.GetProp("_CIPCode") for b in m.GetBonds()
             if b.HasProp("_CIPCode") and b.GetProp("_CIPCode") in BOND_LABELS}
    return atoms, bonds


def labels(doc: Document, atoms=None) -> dict[tuple[str, int], str]:
    """CIP labels of the drawing (or of the molecules touching the given atom ids).

    Keys are document keys: ``("atom", atom id)`` and ``("bond", bond id)``. Molecules that
    cannot be converted (valence errors ...) are skipped."""
    out: dict[tuple[str, int], str] = {}
    sel = None if atoms is None else set(atoms)
    for comp in doc.fragments():
        if sel is not None and not sel.intersection(comp):
            continue
        try:
            mol = convert.to_mol(doc, comp)
        except (mo.ChemError, Exception):  # noqa: BLE001 - one bad fragment must not stop others
            continue
        a_lab, b_lab = mol_labels(mol)
        if not a_lab and not b_lab:
            continue
        origin = {}
        count: dict[int, int] = {}
        for at in mol.GetAtoms():
            if at.HasProp("doc_id"):
                did = at.GetIntProp("doc_id")
                origin[at.GetIdx()] = did
                count[did] = count.get(did, 0) + 1
        for idx, lab in a_lab.items():
            did = origin.get(idx)
            # atoms of an expanded nickname all point to the label atom: not drawable
            if did is None or count.get(did, 0) != 1 or doc.atoms[did].label:
                continue
            out[("atom", did)] = lab
        for idx, lab in b_lab.items():
            b = mol.GetBondWithIdx(idx)
            da, db = origin.get(b.GetBeginAtomIdx()), origin.get(b.GetEndAtomIdx())
            if da is None or db is None or count.get(da) != 1 or count.get(db) != 1:
                continue
            bond = doc.bond_between(da, db)
            if bond is not None:
                out[("bond", bond.id)] = lab
    return out


def format_label(label: str, parens: bool = True) -> str:
    """'(R)' or 'R' (ChemDraw's 'Include parentheses' preference)."""
    return f"({label})" if parens else label

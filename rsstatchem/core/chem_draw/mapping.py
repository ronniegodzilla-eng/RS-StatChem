"""Automatic reaction atom-atom mapping (ChemDraw's Structure > Map Reaction Atoms) and Clear
Reaction Map.

The automatic mapper is EPAM Indigo's ``automap`` (package ``epam.indigo``, Apache-2.0). It
is optional: without it :func:`available` is False, :func:`why_unavailable` says why and the
menu command is greyed out; Clear Reaction Map and the manual map tool always work.

The reaction is read from the drawing exactly as Reaction SMILES export reads it
(:func:`convert.split_reaction`), handed to Indigo as an RXN block built from the same RDKit
molecules (so atom order is known) and the map numbers are written back to the drawn atoms.
"""
from __future__ import annotations

from rsstatchem.core.chem import molecule as mo
from rsstatchem.core.chem_draw import convert
from rsstatchem.core.chem_draw.model import Document

__all__ = ["available", "why_unavailable", "indigo_version", "map_reaction", "clear_map",
           "MappingError"]

_STATE: dict = {}


class MappingError(ValueError):
    pass


def _indigo():
    if "indigo" not in _STATE:
        try:
            from indigo import Indigo
            _STATE["indigo"] = Indigo()
            _STATE["error"] = ""
        except Exception as exc:  # noqa: BLE001 - not installed, or its library fails to load
            _STATE["indigo"] = None
            _STATE["error"] = str(exc) or exc.__class__.__name__
    return _STATE["indigo"]


def available() -> bool:
    return _indigo() is not None


def why_unavailable() -> str:
    if available():
        return ""
    return ("Automatic atom mapping needs the optional Indigo toolkit (epam.indigo, "
            "Apache-2.0), which is not installed in this copy of RS-StatChem"
            + (f" ({_STATE.get('error')})" if _STATE.get("error") else "")
            + ". Map atoms by hand with the Atom-to-atom map tool instead.")


def indigo_version() -> str:
    ind = _indigo()
    return ind.version() if ind is not None else ""


def _side_mols(doc: Document, comps):
    mols = []
    for comp in comps:
        m = convert.to_mol(doc, comp)
        ids = [a.GetIntProp("doc_id") if a.HasProp("doc_id") else -1 for a in m.GetAtoms()]
        mols.append((m, ids))
    return mols


def map_reaction(doc: Document, mode: str = "discard ignore_charges") -> dict[int, int]:
    """Map the drawn reaction automatically; sets ``Atom.map_number`` and returns
    {document atom id: map number}. Raises ``MappingError``."""
    ind = _indigo()
    if ind is None:
        raise MappingError(why_unavailable())
    split = convert.split_reaction(doc)
    if not split.ok:
        raise MappingError("Draw reactants, a reaction arrow and products first.")
    reactants, products = split.reactants, split.products
    try:
        left = _side_mols(doc, reactants)
        right = _side_mols(doc, products)
    except mo.ChemError as exc:
        raise MappingError(f"Invalid structure: {exc}") from None
    from rdkit import Chem
    from rdkit.Chem import rdChemReactions
    rxn = rdChemReactions.ChemicalReaction()
    for m, _ in left:
        rxn.AddReactantTemplate(Chem.Mol(m))
    for m, _ in right:
        rxn.AddProductTemplate(Chem.Mol(m))
    block = rdChemReactions.ReactionToRxnBlock(rxn)
    try:
        r = ind.loadReaction(block)
        r.automap(mode)
    except Exception as exc:  # noqa: BLE001 - IndigoException
        raise MappingError(f"Indigo could not map the reaction: {exc}") from None
    result: dict[int, int] = {}
    sides = list(r.iterateReactants()) + list(r.iterateProducts())
    for (_m, ids), im in zip(left + right, sides):
        for atom in im.iterateAtoms():
            idx = atom.index()
            num = r.atomMappingNumber(atom)
            if idx < len(ids) and ids[idx] >= 0 and num > 0:
                result.setdefault(ids[idx], num)
    comps = [a for c in reactants + products for a in c]
    for aid in comps:
        doc.atoms[aid].map_number = result.get(aid, 0)
    if not result:
        raise MappingError("Indigo found no atom correspondence for this reaction.")
    return result


def clear_map(doc: Document, atom_ids=None) -> int:
    """Clear Reaction Map: remove map numbers (of ``atom_ids`` or all); returns the count."""
    n = 0
    for aid in (doc.atoms if atom_ids is None else atom_ids):
        a = doc.atoms.get(aid)
        if a is not None and a.map_number:
            a.map_number = 0
            n += 1
    return n

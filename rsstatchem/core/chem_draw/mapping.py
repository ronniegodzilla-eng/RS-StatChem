"""Automatic reaction atom-atom mapping (ChemDraw's Structure > Map Reaction Atoms) and Clear
Reaction Map.

The automatic mapper is EPAM Indigo's ``automap`` (package ``epam.indigo``, Apache-2.0). It
is optional: without it :func:`available` is False, :func:`why_unavailable` says why and the
menu command is greyed out; Clear Reaction Map and the manual map tool always work.

The reaction is read from the drawing exactly as Reaction SMILES export reads it
(:func:`convert.split_reaction`), handed to Indigo as an RXN block built from the same RDKit
molecules (so atom order is known) and the map numbers are written back to the drawn atoms.

Indigo's result is checked: every number must belong to one reactant atom and at most one
product atom of the same element. With several products Indigo sometimes gives the same
reactant atoms to two products (acetic anhydride in the aspirin synthesis); such a mapping is
replaced by a maximum-common-substructure mapping (:func:`mcs_mapping`, RDKit) that matches the
largest product first and never uses a reactant atom twice.
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
    numbers = [[r.atomMappingNumber(atom) for atom in im.iterateAtoms()]
               for im in list(r.iterateReactants()) + list(r.iterateProducts())]
    if not valid_mapping([m for m, _ in left], [m for m, _ in right], numbers):
        numbers = mcs_mapping([m for m, _ in left], [m for m, _ in right])
    result: dict[int, int] = {}
    for (_m, ids), nums in zip(left + right, numbers):
        for idx, num in enumerate(nums):
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


# ---------------------------------------------------------------------------------- checking
def valid_mapping(reactants, products, numbers) -> bool:
    """``numbers``: map numbers per molecule (reactants then products, atom order). Valid when
    each number is on exactly one reactant atom and at most one product atom of the same
    element, and something is mapped."""
    n_r = len(reactants)
    owner: dict[int, str] = {}
    seen_products: set[int] = set()
    for k, (mol, nums) in enumerate(zip(list(reactants) + list(products), numbers)):
        for idx, num in enumerate(nums):
            if num <= 0:
                continue
            element = mol.GetAtomWithIdx(idx).GetSymbol() if idx < mol.GetNumAtoms() else "?"
            if k < n_r:
                if num in owner:
                    return False
                owner[num] = element
            else:
                if num in seen_products or owner.get(num) != element:
                    return False
                seen_products.add(num)
    return bool(seen_products)


def mcs_mapping(reactants, products, timeout: int = 2) -> list[list[int]]:
    """A one-to-one atom mapping from maximum common substructures: products from the largest,
    each matched again and again to the reactant whose still-free atoms share the largest
    substructure with its still-unmapped atoms (bond orders may change, elements may not).
    Returns map numbers per molecule (reactants then products)."""
    from rdkit import Chem
    from rdkit.Chem import rdFMCS

    def part(mol, keep: list[int]):
        """Sub-molecule of the atoms ``keep`` (bonds between them) and its index map."""
        rw = Chem.RWMol(Chem.Mol(mol))
        for a in rw.GetAtoms():
            a.SetIntProp("orig", a.GetIdx())
            a.SetNoImplicit(True)
        for idx in sorted(set(range(mol.GetNumAtoms())) - set(keep), reverse=True):
            rw.RemoveAtom(idx)
        sub = rw.GetMol()
        try:
            sub.UpdatePropertyCache(strict=False)
        except Exception:  # noqa: BLE001
            pass
        return sub, [a.GetIntProp("orig") for a in sub.GetAtoms()]

    free = [set(range(m.GetNumAtoms())) for m in reactants]
    pairs: list[tuple[int, int, int, int]] = []        # (reactant, r atom, product, p atom)
    for pk in sorted(range(len(products)), key=lambda k: -products[k].GetNumAtoms()):
        pmol = products[pk]
        open_atoms = set(range(pmol.GetNumAtoms()))
        while open_atoms:
            best = None
            psub, pmap = part(pmol, sorted(open_atoms))
            for rk, rmol in enumerate(reactants):
                if not free[rk]:
                    continue
                rsub, rmap = part(rmol, sorted(free[rk]))
                res = rdFMCS.FindMCS([psub, rsub], timeout=timeout,
                                     atomCompare=rdFMCS.AtomCompare.CompareElements,
                                     bondCompare=rdFMCS.BondCompare.CompareAny,
                                     ringMatchesRingOnly=False, completeRingsOnly=False)
                if res.numAtoms == 0 or (best is not None and res.numAtoms <= best[0]):
                    continue
                query = Chem.MolFromSmarts(res.smartsString)
                pm = psub.GetSubstructMatch(query)
                rm = rsub.GetSubstructMatch(query)
                if pm and rm:
                    best = (res.numAtoms, rk, [(rmap[r], pmap[q]) for q, r in zip(pm, rm)])
            if best is None:
                break
            _, rk, matched = best
            for ra, pa in matched:
                pairs.append((rk, ra, pk, pa))
                free[rk].discard(ra)
                open_atoms.discard(pa)
    numbers = [[0] * m.GetNumAtoms() for m in list(reactants) + list(products)]
    n_r = len(reactants)
    for n, (rk, ra, pk, pa) in enumerate(sorted(pairs), 1):
        numbers[rk][ra] = n
        numbers[n_r + pk][pa] = n
    return numbers

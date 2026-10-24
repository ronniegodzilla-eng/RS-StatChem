"""ChemDraw-style "Analysis" values, reaction balance check and the stoichiometry grid.

- :func:`analyze` - formula (Hill, with charge), exact mass, molecular weight, m/z isotope
  peaks, elemental analysis, cLogP, TPSA, H-bond donors/acceptors and the Lipinski summary
  of the selection or the whole drawing (stage-1 ``formula`` / ``molecule`` cores).
- :func:`balance_check` - atom counts per side of a drawn reaction and, via
  ``reaction.balance``, the balanced coefficients or what is missing.
- :func:`stoich_grid` - ChemDraw's reaction stoichiometry table (formula, MW, equivalents,
  mmol, mass, limiting reagent, theoretical yield) through ``reaction.stoichiometry``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction

from rdkit import Chem

from rsstatchem.core.chem import formula as fm
from rsstatchem.core.chem import molecule as mo
from rsstatchem.core.chem import reaction as rx
from rsstatchem.core.chem.sections import isotope_pattern
from rsstatchem.core.chem_draw import convert
from rsstatchem.core.chem_draw.model import Document

__all__ = ["Analysis", "analyze", "formula_of", "BalanceCheck", "balance_check",
           "StoichSpecies", "StoichLine", "stoich_species", "stoich_grid"]


def formula_of(mol: Chem.Mol) -> fm.Formula:
    """Stage-1 :class:`Formula` of an RDKit molecule (isotopes kept, H counted)."""
    counts: dict = {}
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 0:
            continue
        key = (a.GetSymbol(), a.GetIsotope() or None)
        counts[key] = counts.get(key, 0) + 1
        h = a.GetTotalNumHs()
        if h:
            counts[("H", None)] = counts.get(("H", None), 0) + h
    f = fm.Formula({k: Fraction(v) for k, v in counts.items()}, Chem.GetFormalCharge(mol))
    f.text = f.hill()
    return f


@dataclass
class Analysis:
    ok: bool = False
    message: str = ""
    formula: str = ""
    formula_html: str = ""
    exact_mass: float = math.nan
    mw: float = math.nan
    mz: list = field(default_factory=list)           # [(m/z, relative %)]
    elemental: list = field(default_factory=list)    # [(symbol, mass %)]
    clogp: float = math.nan
    tpsa: float = math.nan
    hbd: int = 0
    hba: int = 0
    rotb: int = 0
    lipinski: str = ""
    smiles: str = ""
    inchikey: str = ""
    n_molecules: int = 0
    charge: int = 0
    stereo: int = 0


def analyze(doc: Document, atom_ids=None) -> Analysis:
    res = Analysis()
    ids = list(doc.atoms) if atom_ids is None else [a for a in atom_ids if a in doc.atoms]
    if not ids:
        res.message = "Nothing drawn"
        return res
    try:
        mol = convert.to_mol(doc, ids)
    except mo.ChemError as exc:
        res.message = str(exc)
        return res
    res.ok = True
    res.n_molecules = len(Chem.GetMolFrags(mol))
    f = formula_of(mol)
    res.charge = f.charge
    res.formula = f.hill()
    res.formula_html = fm.format_formula(res.formula, "html")
    res.mw = f.molar_mass
    res.exact_mass = f.monoisotopic_mass
    try:
        peaks = isotope_pattern(f, threshold=1.0)
        z = abs(f.charge) or 1
        top = sorted(peaks, key=lambda p: -p[1])[:4]
        res.mz = [(m / z, a) for m, a in sorted(top)]
    except Exception:  # noqa: BLE001
        res.mz = []
    res.elemental = [(r.symbol, r.percent) for r in f.composition()]
    p = mo.properties(mol)
    res.clogp, res.tpsa = p["cLogP"], p["TPSA"]
    res.hbd, res.hba, res.rotb = p["HBD"], p["HBA"], p["RotB"]
    res.stereo = p["Stereo"]
    lip = mo.drug_likeness(mol, ["Lipinski"])[0]
    res.lipinski = (f"{'Pass' if lip.passed else 'Fail'} ({lip.violations} violation"
                    f"{'' if lip.violations == 1 else 's'})")
    res.smiles = mo.to_smiles(mol)
    try:
        res.inchikey = mo.to_inchikey(mol)
    except Exception:  # noqa: BLE001
        res.inchikey = ""
    return res


# --- reactions ------------------------------------------------------------------------------

@dataclass
class BalanceCheck:
    ok: bool
    message: str
    rows: list = field(default_factory=list)       # (element, left, right)
    balanced: bool = False
    equation: str = ""                             # as drawn
    suggestion: str = ""                           # balanced equation from reaction.balance
    species: list = field(default_factory=list)    # (side, formula text)
    result: object = None


def _side_species(doc: Document, comps, sign: int) -> list:
    out = []
    for comp in comps:
        mol = convert.to_mol(doc, comp)
        f = formula_of(mol)
        text = f.hill() + (fm.format_charge(f.charge) if f.charge else "")
        out.append(rx.Species(text, f, sign))
    return out


def balance_check(doc: Document) -> BalanceCheck:
    sp = convert.split_reaction(doc)
    if not sp.ok:
        msg = " ".join(sp.notes) or "Draw reactants, a reaction arrow and products."
        return BalanceCheck(False, msg)
    try:
        species = _side_species(doc, sp.reactants, 1) + _side_species(doc, sp.products, -1)
    except mo.ChemError as exc:
        return BalanceCheck(False, f"Invalid structure: {exc}")
    keys: list = []
    for s in species:
        for k in s.formula.counts:
            if k not in keys:
                keys.append(k)
    keys.sort(key=lambda k: ({"C": 0, "H": 1}.get(k[0], 2), k[0], k[1] or 0))
    rows = []
    for k in keys:
        left = sum(s.formula.counts.get(k, 0) for s in species if s.side > 0)
        right = sum(s.formula.counts.get(k, 0) for s in species if s.side < 0)
        rows.append((k[0] if k[1] is None else f"{k[1]}{k[0]}", int(left), int(right)))
    ql = sum(s.formula.charge for s in species if s.side > 0)
    qr = sum(s.formula.charge for s in species if s.side < 0)
    if ql or qr:
        rows.append(("Charge", ql, qr))
    balanced = all(l == r for _, l, r in rows)

    def eq(coefs=None):
        sides = []
        for sign in (1, -1):
            terms = []
            for i, s in enumerate(species):
                if s.side == sign:
                    c = coefs[i] if coefs else 1
                    terms.append(("" if c == 1 else f"{c} ") + s.text)
            sides.append(" + ".join(terms))
        return f"{sides[0]} → {sides[1]}"
    res = BalanceCheck(True, "", rows, balanced, eq(), species=[(s.side, s.text)
                                                                  for s in species])
    try:
        br = rx.balance(species)
    except Exception as exc:  # noqa: BLE001
        br = None
        res.message = str(exc)
    res.result = br
    if balanced:
        res.message = "The reaction is balanced as drawn."
    elif br is not None and br.coefficients is not None:
        res.suggestion = eq(br.coefficients)
        res.message = ("The reaction is not balanced as drawn. Balanced coefficients: "
                       + res.suggestion)
    else:
        missing = [f"{e}: {l} vs {r}" for e, l, r in rows if l != r]
        res.message = ("The reaction is not balanced (" + "; ".join(missing) + "). " +
                       (br.message if br is not None else "") +
                       " A by-product (e.g. H2O) may be missing from the drawing.")
    if sp.notes:
        res.message += " " + " ".join(sp.notes)
    return res


@dataclass
class StoichSpecies:
    index: int
    role: str            # Reactant | Product
    name: str            # SMILES of the drawn molecule
    formula: str
    mw: float
    coefficient: int
    atom_ids: list = field(default_factory=list)


@dataclass
class StoichLine:
    species: StoichSpecies
    equiv: float | None
    mmol: float | None
    mass_mg: float | None
    limiting: bool = False
    note: str = ""


def stoich_species(doc: Document) -> tuple[list[StoichSpecies], object, str]:
    """Species of the drawn reaction with their balanced coefficients (1 when the drawing
    cannot be balanced), plus the BalanceResult and a note."""
    sp = convert.split_reaction(doc)
    if not sp.ok:
        raise ValueError(" ".join(sp.notes) or "Draw reactants, an arrow and products.")
    comps = [(c, "Reactant", 1) for c in sp.reactants] + \
            [(c, "Product", -1) for c in sp.products]
    species, out = [], []
    for i, (comp, role, sign) in enumerate(comps):
        mol = convert.to_mol(doc, comp)
        f = formula_of(mol)
        text = f.hill() + (fm.format_charge(f.charge) if f.charge else "")
        species.append(rx.Species(text, f, sign))
        out.append(StoichSpecies(i, role, mo.to_smiles(mol), text, f.molar_mass, 1, comp))
    note = ""
    br = rx.balance(species)
    if br.coefficients is None:
        br = rx.BalanceResult(species, "→", "drawn", [1] * len(species))
        note = ("The drawn reaction cannot be balanced; coefficients of 1 are used "
                "(check for a missing by-product).")
    elif br.status == "not_unique":
        note = "Coefficients are not unique; the smallest positive set is used."
    for s, c in zip(out, br.coefficients):
        s.coefficient = c
    return out, br, note


def stoich_grid(doc: Document, amounts: dict[int, tuple[float, str]],
                reference: int | None = None) -> tuple[list[StoichLine], list[str]]:
    """Rows of the stoichiometry table. ``amounts`` maps species index -> (value, unit)
    with unit 'mmol', 'mg', 'g' or 'equiv' (equivalents need a reference reactant with an
    amount in mmol/mass). Products get their theoretical yield."""
    species, br, note = stoich_species(doc)
    notes = [note] if note else []
    real: dict[int, rx.Amount] = {}
    for i, (v, unit) in amounts.items():
        if v is None or unit == "equiv" or not (v > 0):
            continue
        real[i] = rx.Amount(float(v), unit)
    if any(u == "equiv" for _, u in amounts.values()):
        ref = reference if reference in real else next(
            (i for i in real if species[i].role == "Reactant"), None)
        if ref is None:
            raise ValueError("Give the amount (mmol or mass) of one reactant.")
        n_ref = real[ref].moles(species[ref].mw) / species[ref].coefficient
        for i, (v, unit) in amounts.items():
            if unit == "equiv" and v and v > 0:
                real[i] = rx.Amount(v * n_ref * species[i].coefficient * 1e3, "mmol")
    if not real:
        raise ValueError("Enter the amount of at least one reactant.")
    st = rx.stoichiometry(br, real, molar_masses={s.index: s.mw for s in species})
    notes += st.notes
    lines = []
    for s, row in zip(species, st.rows):
        if s.role == "Reactant":
            n = row.given_mol
            if n is None:
                lines.append(StoichLine(s, None, None, None, False, "excess / not given"))
                continue
            equiv = (n / s.coefficient) / st.extent if st.extent else None
            lines.append(StoichLine(s, equiv, n * 1e3, n * s.mw * 1e3, row.limiting))
        else:
            lines.append(StoichLine(s, None, row.reacting_mol * 1e3, row.reacting_g * 1e3,
                                    False, "theoretical yield"))
    return lines, notes

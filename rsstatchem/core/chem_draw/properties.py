"""Physico-chemical properties for ChemDraw's *Chemical Properties* window.

Experimental values come first: when the compound is in the offline database of open
experimental data (:mod:`exp_props`; melting / boiling point, logP, water solubility, vapour
pressure, critical constants) that value is shown with its sources. Otherwise the estimate whose
error on the open data is lowest (``resources/properties/benchmarks.json``, made by
``scripts/benchmark_properties.py``) among the methods that cover the molecule is shown, and
every method's value and benchmark error is listed in the row's ``alternatives``. Every value
is labelled with the method that produced it:

- machine-learning models (:mod:`ml_props`) for melting point, boiling point and logP, with
  their held-out expected error and an applicability-domain flag;
- open logP methods of the CDK (:mod:`cdk`: JPlogP, XLogP, ALogP, AMR) and a SwissADME-style
  consensus logP - open-source alternatives, not BioByte CLogP/CMR;

- Joback group contribution (K. G. Joback, R. C. Reid, Chem. Eng. Commun. 57 (1987) 233):
  normal boiling point, melting point, critical temperature / pressure / volume, ideal-gas
  heat and Gibbs energy of formation at 298 K. Only molecules made entirely of Joback groups
  get Joback values.
- Further published group-contribution methods (tables in :mod:`gc_tables`): the boiling
  point by Stein & Brown (J. Chem. Inf. Comput. Sci. 34 (1994) 581); Marrero & Gani first-,
  second- and third-order groups (Fluid Phase Equilib. 183-184 (2001) 183; third order in
  :mod:`mg_third`) for every Joback property; Abdulelah-Gani for the critical constants;
  Nannoolal / Rarey (:mod:`nannoolal`) for the boiling point, the critical constants (Tc with
  the experimental Tb when known) and the vapour pressure at 25 °C (with the experimental Tb,
  else the default Tb estimate; solids converted with the melting point by Walden's rule).
  Each value says which method gave it; a value stays n/a when no method covers every atom
  (no guessing).
- Crippen LogP and molar refractivity (Wildman & Crippen 1999, as implemented by RDKit).
- Topological polar surface area (Ertl 2000, RDKit).
- ESOL aqueous solubility LogS (Delaney 2004): 0.16 - 0.63 cLogP - 0.0062 MW + 0.066 RB
  - 0.74 AP (AP = aromatic heavy-atom proportion).
- pKa of the strongest acidic and basic group (:mod:`pka`): measured values of the Mansouri
  et al. 2019 data set, else gradient-boosted trees trained on it (open replacement for the
  licensed MOSES pKa).
- Henry's law constant (:mod:`henry`): PHYSPROP measured values, else the better of an ML
  model and the ratio of vapour pressure to water solubility.

These are not ChemDraw's licensed ChemProp / BioByte (CLogP, CMR) values.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

__all__ = ["JOBACK", "JobackResult", "joback", "PropertyValue", "MethodValue",
           "chemical_properties", "stein_brown", "marrero_gani", "abdulelah_gani",
           "GCResult", "METHODS",
           "OTHER_METHODS", "benchmarks", "benchmark_text", "ranked", "method_label",
           "EXP_SHORT", "NOT_BIOBYTE"]

METHODS = {
    "joback": ("Joback", "Joback group contribution estimate (Joback & Reid 1987)"),
    "sb": ("Stein-Brown", "Stein & Brown group contribution estimate (1994), boiling point"),
    "mg": ("Marrero-Gani", "Marrero & Gani group contribution estimate (2001), first- plus "
           "second-order groups"),
    "mg1": ("Marrero-Gani 1st", "Marrero & Gani first-order group contribution estimate "
            "(2001)"),
    "mg3": ("Marrero-Gani +3rd", "Marrero & Gani group contribution estimate (2001), first- "
            "plus second- plus third-order groups"),
    "nan": ("Nannoolal", "Nannoolal / Rarey group contribution estimate (Tb 2004, critical "
            "properties 2007, vapour pressure 2008)"),
    "ag": ("Abdulelah-Gani", "Abdulelah-Gani GC-simple group contribution estimate (Alshehri "
           "et al. 2022), primary and secondary groups"),
}

# group -> (Tc, Pc, Vc, Tb, Tm, Hform, Gform); None = not given by Joback
JOBACK: dict[str, tuple] = {
    "-CH3": (0.0141, -0.0012, 65, 23.58, -5.10, -76.45, -43.96),
    "-CH2-": (0.0189, 0.0000, 56, 22.88, 11.27, -20.64, 8.42),
    ">CH-": (0.0164, 0.0020, 41, 21.74, 12.64, 29.89, 58.36),
    ">C<": (0.0067, 0.0043, 27, 18.25, 46.43, 82.23, 116.02),
    "=CH2": (0.0113, -0.0028, 56, 18.18, -4.32, -9.630, 3.77),
    "=CH-": (0.0129, -0.0006, 46, 24.96, 8.73, 37.97, 48.53),
    "=C<": (0.0117, 0.0011, 38, 24.14, 11.14, 83.99, 92.36),
    "=C=": (0.0026, 0.0028, 36, 26.15, 17.78, 142.14, 136.70),
    "#CH": (0.0027, -0.0008, 46, 9.20, -11.18, 79.30, 77.71),
    "#C-": (0.0020, 0.0016, 37, 27.38, 64.32, 115.51, 109.82),
    "ring -CH2-": (0.0100, 0.0025, 48, 27.15, 7.75, -26.80, -3.68),
    "ring >CH-": (0.0122, 0.0004, 38, 21.78, 19.88, 8.67, 40.99),
    "ring >C<": (0.0042, 0.0061, 27, 21.32, 60.15, 79.72, 87.88),
    "ring =CH-": (0.0082, 0.0011, 41, 26.73, 8.13, 2.09, 11.30),
    "ring =C<": (0.0143, 0.0008, 32, 31.01, 37.02, 46.43, 54.05),
    "-F": (0.0111, -0.0057, 27, -0.03, -15.78, -251.92, -247.19),
    "-Cl": (0.0105, -0.0049, 58, 38.13, 13.55, -71.55, -64.31),
    "-Br": (0.0133, 0.0057, 71, 66.86, 43.43, -29.48, -38.06),
    "-I": (0.0068, -0.0034, 97, 93.84, 41.69, 21.06, 5.74),
    "-OH (alcohol)": (0.0741, 0.0112, 28, 92.88, 44.45, -208.04, -189.20),
    "-OH (phenol)": (0.0240, 0.0184, -25, 76.34, 82.83, -221.65, -197.37),
    "-O-": (0.0168, 0.0015, 18, 22.42, 22.23, -132.22, -105.00),
    "ring -O-": (0.0098, 0.0048, 13, 31.22, 23.05, -138.16, -98.22),
    ">C=O": (0.0380, 0.0031, 62, 76.75, 61.20, -133.22, -120.50),
    "ring >C=O": (0.0284, 0.0028, 55, 94.97, 75.97, -164.50, -126.27),
    "O=CH- (aldehyde)": (0.0379, 0.0030, 82, 72.24, 36.90, -162.03, -143.48),
    "-COOH": (0.0791, 0.0077, 89, 169.09, 155.50, -426.72, -387.87),
    "-COO-": (0.0481, 0.0005, 82, 81.10, 53.60, -337.92, -301.95),
    "=O (other)": (0.0143, 0.0101, 36, -10.50, 2.08, -247.61, -250.83),
    "-NH2": (0.0243, 0.0109, 38, 73.23, 66.89, -22.02, 14.07),
    ">NH": (0.0295, 0.0077, 35, 50.17, 52.66, 53.47, 89.39),
    "ring >NH": (0.0130, 0.0114, 29, 52.82, 101.51, 31.65, 75.61),
    ">N-": (0.0169, 0.0074, 9, 11.74, 48.84, 123.34, 163.16),
    "-N=": (0.0255, -0.0099, None, 74.60, None, 23.61, None),
    "ring -N=": (0.0085, 0.0076, 34, 57.55, 68.40, 55.52, 79.93),
    "=NH": (None, None, None, 83.08, 68.91, 93.70, 119.66),
    "-CN": (0.0496, -0.0101, 91, 125.66, 59.89, 88.43, 89.22),
    "-NO2": (0.0437, 0.0064, 91, 152.54, 127.24, -66.57, -16.83),
    "-SH": (0.0031, 0.0084, 63, 63.56, 20.09, -17.33, -22.99),
    "-S-": (0.0119, 0.0049, 54, 68.78, 34.40, 41.87, 33.12),
    "ring -S-": (0.0019, 0.0051, 38, 52.10, 79.93, 39.10, 27.76),
}
_HALOGEN = {"F": "-F", "Cl": "-Cl", "Br": "-Br", "I": "-I"}


@dataclass
class JobackResult:
    groups: dict = field(default_factory=dict)      # group -> count
    missing: str = ""                               # why no estimate is possible
    tb: float | None = None                         # K
    tm: float | None = None                         # K
    tc: float | None = None                         # K
    pc: float | None = None                         # bar
    vc: float | None = None                         # cm3/mol
    hf: float | None = None                         # kJ/mol
    gf: float | None = None                         # kJ/mol

    @property
    def ok(self) -> bool:
        return not self.missing


def _ring(atom) -> str:
    return "ring " if atom.IsInRing() else ""


def _groups(mol: Chem.Mol) -> tuple[dict, str]:
    """Joback groups of a molecule, or ({}, reason) when an atom fits no group."""
    aromatic = {a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()}
    m = Chem.RWMol(mol)
    try:
        Chem.Kekulize(m, clearAromaticFlags=True)
    except Exception:  # noqa: BLE001
        return {}, "the structure could not be kekulized"
    groups: dict[str, int] = {}
    used: set[int] = set()

    def add(name: str, *atoms) -> None:
        groups[name] = groups.get(name, 0) + 1
        used.update(atoms)

    def bonds(a):
        return [(b, b.GetOtherAtom(a)) for b in a.GetBonds()]

    def terminal(o) -> bool:
        return o.GetDegree() == 1

    # multi-atom groups first: nitro, carboxylic acid, ester, carbonyls, nitrile
    for a in m.GetAtoms():
        i = a.GetIdx()
        if a.GetSymbol() == "N" and i not in used:
            oxy = [o for _, o in bonds(a) if o.GetSymbol() == "O" and terminal(o)]
            if len(oxy) == 2 and a.GetFormalCharge() == 1:
                add("-NO2", i, *(o.GetIdx() for o in oxy))
        if a.GetSymbol() != "C" or i in used:
            continue
        dbl_o = [o for b, o in bonds(a) if o.GetSymbol() == "O" and terminal(o) and
                 b.GetBondType() == Chem.BondType.DOUBLE]
        if dbl_o:
            single_o = [o for b, o in bonds(a) if o.GetSymbol() == "O" and
                        b.GetBondType() == Chem.BondType.SINGLE]
            oh = [o for o in single_o if terminal(o) and o.GetTotalNumHs() == 1]
            ether = [o for o in single_o if o.GetDegree() == 2 and o.GetIdx() not in used]
            if oh:
                add("-COOH", i, dbl_o[0].GetIdx(), oh[0].GetIdx())
            elif ether:
                add("-COO-", i, dbl_o[0].GetIdx(), ether[0].GetIdx())
            elif a.GetTotalNumHs() == 1:
                add("O=CH- (aldehyde)", i, dbl_o[0].GetIdx())
            elif a.GetTotalNumHs() == 0:
                add(_ring(a) + ">C=O", i, dbl_o[0].GetIdx())
            else:
                return {}, "formaldehyde has no Joback group"
            continue
        nitrile = [o for b, o in bonds(a) if o.GetSymbol() == "N" and terminal(o) and
                   b.GetBondType() == Chem.BondType.TRIPLE]
        if nitrile:
            add("-CN", i, nitrile[0].GetIdx())
    for a in m.GetAtoms():
        i = a.GetIdx()
        if i in used:
            continue
        if a.GetFormalCharge():
            return {}, f"no Joback group for the charged atom {a.GetSymbol()}"
        sym, h = a.GetSymbol(), a.GetTotalNumHs()
        types = [b.GetBondType() for b, _ in bonds(a)]
        n_dbl = types.count(Chem.BondType.DOUBLE)
        n_tri = types.count(Chem.BondType.TRIPLE)
        heavy = a.GetDegree()
        r = _ring(a)
        name = None
        if sym == "C":
            if n_tri:
                name = "#CH" if h == 1 else "#C-"
            elif n_dbl == 2:
                name = "=C="
            elif n_dbl == 1:
                name = {2: "=CH2", 1: r + "=CH-", 0: r + "=C<"}.get(h)
                if h == 2 and r:
                    name = None
            else:
                name = {3: "-CH3", 2: r + "-CH2-", 1: r + ">CH-", 0: r + ">C<"}.get(h)
                if h == 3 and heavy == 0:
                    name = None                     # methane
        elif sym in _HALOGEN and heavy == 1:
            name = _HALOGEN[sym]
        elif sym == "O":
            if n_dbl == 1:
                name = "=O (other)"
            elif h == 1 and heavy == 1:
                nb = a.GetNeighbors()[0]
                name = "-OH (phenol)" if nb.GetIdx() in aromatic else "-OH (alcohol)"
            elif h == 0 and heavy == 2:
                name = r + "-O-"
        elif sym == "N" and not a.GetFormalCharge():
            if n_dbl == 1:
                name = {0: r + "-N=", 1: None if r else "=NH"}.get(h)
            elif not n_tri:
                if h == 2:
                    name = "-NH2"
                elif h == 1:
                    name = r + ">NH"
                elif h == 0 and not r:
                    name = ">N-"
        elif sym == "S" and not n_dbl:
            if h == 1 and heavy == 1:
                name = "-SH"
            elif h == 0 and heavy == 2:
                name = r + "-S-"
        if name is None or name not in JOBACK:
            where = "ring " if a.IsInRing() else ""
            return {}, f"no Joback group for a {where}{sym} atom with {h} H"
        add(name, i)
    return groups, ""


def joback(mol: Chem.Mol) -> JobackResult:
    res = JobackResult()
    if mol is None or mol.GetNumAtoms() == 0:
        res.missing = "nothing drawn"
        return res
    if len(Chem.GetMolFrags(mol)) > 1:
        res.missing = "select a single molecule"
        return res
    groups, why = _groups(mol)
    res.groups = groups
    if why:
        res.missing = why
        return res

    def total(k: int):
        s = 0.0
        for g, n in groups.items():
            v = JOBACK[g][k]
            if v is None:
                return None
            s += n * v
        return s

    n_atoms = Chem.AddHs(mol).GetNumAtoms()
    stc, spc, svc, stb, stm, shf, sgf = (total(k) for k in range(7))
    if stb is not None:
        res.tb = 198.2 + stb
        if stc is not None:
            d = 0.584 + 0.965 * stc - stc * stc
            res.tc = res.tb / d if d > 0 else None
    if stm is not None:
        res.tm = 122.5 + stm
    if spc is not None:
        q = 0.113 + 0.0032 * n_atoms - spc
        res.pc = q ** -2 if q > 0 else None
    if svc is not None:
        res.vc = 17.5 + svc
    if shf is not None:
        res.hf = 68.29 + shf
    if sgf is not None:
        res.gf = 53.88 + sgf
    return res


# --- other group-contribution methods (fallbacks) ---------------------------------------------

@dataclass
class GCResult:
    method: str                                     # key of METHODS
    groups: dict = field(default_factory=dict)
    missing: str = ""
    values: dict = field(default_factory=dict)      # tb, tm, tc, pc, vc, hf, gf (None = n/a)
    second: dict = field(default_factory=dict)      # Marrero-Gani second-order groups
    third: dict = field(default_factory=dict)       # Marrero-Gani third-order groups

    @property
    def ok(self) -> bool:
        return not self.missing


def _split_smarts(smarts: str) -> list[str]:
    """Alternatives separated by top-level commas (commas inside [] or () belong to SMARTS)."""
    parts, depth, cur = [], 0, ""
    for ch in smarts:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [x for x in parts if x]


_QUERIES: dict = {}


def _queries(key: str, items):
    """[(group, [query mols])] compiled once; ``items`` = [(group, SMARTS)]."""
    if key not in _QUERIES:
        out = []
        for name, smarts in items:
            qs = [q for q in (Chem.MolFromSmarts(x) for x in _split_smarts(smarts))
                  if q is not None]
            if qs:
                out.append((name, qs))
        _QUERIES[key] = out
    return _QUERIES[key]


def _forms(mol: Chem.Mol) -> list[Chem.Mol]:
    """The molecule with RDKit aromaticity, with MDL aromaticity and fully Kekule: group
    tables differ in what they call aromatic (e.g. caffeine's pyrimidinedione ring)."""
    out = [mol]
    try:
        kek = Chem.Mol(mol)
        Chem.Kekulize(kek, clearAromaticFlags=True)
        mdl = Chem.Mol(kek)
        Chem.SetAromaticity(mdl, Chem.AromaticityModel.AROMATICITY_MDL)
        out += [mdl, kek]
    except Exception:  # noqa: BLE001
        pass
    return out


def _fragment(mol: Chem.Mol, queries) -> tuple[dict, list[int]]:
    """Greedy exclusive matching (larger groups first, then table order): group counts and
    the heavy atoms no group covers."""
    used: set[int] = set()
    counts: dict[str, int] = {}
    for name, qs in queries:
        for q in qs:
            for match in mol.GetSubstructMatches(q, uniquify=True):
                if used.intersection(match):
                    continue
                used.update(match)
                counts[name] = counts.get(name, 0) + 1
    left = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in used]
    return counts, left


def _best_fragmentation(mol: Chem.Mol, key: str, items, with_form: bool = False):
    """(group counts, why not) - plus the molecule form that was fragmented (RDKit / MDL
    aromaticity or Kekule) when ``with_form``."""
    queries = _queries(key, items)
    first_left = None
    for form in _forms(mol):
        counts, left = _fragment(form, queries)
        if not left:
            return (counts, "", form) if with_form else (counts, "")
        if first_left is None:
            first_left = left
    a = mol.GetAtomWithIdx(first_left[0])
    where = "ring " if a.IsInRing() else ""
    why = f"no group for a {where}{a.GetSymbol()} atom with {a.GetTotalNumHs()} H"
    return ({}, why, None) if with_form else ({}, why)


_MG2: list | None = None


def _mg_second_order(form: Chem.Mol) -> dict:
    """Marrero-Gani second-order group occurrences (overlaps allowed) in the fragmented form."""
    global _MG2
    from rsstatchem.core.chem_draw.gc_tables import MARRERO_GANI_2
    if _MG2 is None:
        _MG2 = [(row[0], Chem.MolFromSmarts(row[1]), row[2:]) for row in MARRERO_GANI_2]
    out = {}
    for name, q, _vals in _MG2:
        if q is None:
            continue
        n = len(form.GetSubstructMatches(q, uniquify=True))
        if n:
            out[name] = n
    return out


def _gc_precheck(mol: Chem.Mol) -> str:
    if mol is None or mol.GetNumAtoms() == 0:
        return "nothing drawn"
    if len(Chem.GetMolFrags(mol)) > 1:
        return "select a single molecule"
    if any(a.GetFormalCharge() for a in mol.GetAtoms()) and not _is_nitro_only(mol):
        return "charged atoms have no group"
    return ""


def _is_nitro_only(mol: Chem.Mol) -> bool:
    for a in mol.GetAtoms():
        q = a.GetFormalCharge()
        if not q:
            continue
        if a.GetSymbol() == "N" and q == 1 and sum(
                1 for n in a.GetNeighbors() if n.GetSymbol() == "O") >= 2:
            continue
        if a.GetSymbol() == "O" and q == -1 and a.GetNeighbors() and \
                a.GetNeighbors()[0].GetSymbol() == "N":
            continue
        return False
    return True


def stein_brown(mol: Chem.Mol) -> GCResult:
    """Normal boiling point by Stein & Brown (1994), with their correction."""
    from rsstatchem.core.chem_draw.gc_tables import STEIN_BROWN, STEIN_BROWN_SMARTS

    res = GCResult("sb")
    res.missing = _gc_precheck(mol)
    if res.missing:
        return res
    counts, why = _best_fragmentation(mol, "sb", STEIN_BROWN_SMARTS)
    if why:
        res.missing = why
        return res
    res.groups = counts
    tb = 198.2 + sum(n * STEIN_BROWN[g] for g, n in counts.items())
    tb = tb - 94.84 + 0.5577 * tb - 0.0007705 * tb * tb if tb <= 700 else \
        tb + 282.7 - 0.5209 * tb
    res.values = {"tb": tb}
    return res


_MG_ITEMS: list | None = None


def marrero_gani(mol: Chem.Mol, second_order: bool = True,
                 third_order: bool = False) -> GCResult:
    """Marrero & Gani (2001) estimates of Tb, Tm, Tc, Pc, Vc, Hf, Gf: first-order groups
    plus (by default) the second-order corrections, and optionally the third-order groups
    (:mod:`mg_third`)."""
    global _MG_ITEMS
    from rsstatchem.core.chem_draw.gc_tables import (MARRERO_GANI, MARRERO_GANI_2,
                                                      MARRERO_GANI_3, MG_FIELDS)

    res = GCResult("mg3" if third_order else ("mg" if second_order else "mg1"))
    res.missing = _gc_precheck(mol)
    if res.missing:
        return res
    if _MG_ITEMS is None:
        _MG_ITEMS = sorted(((g, row[0]) for g, row in MARRERO_GANI.items()),
                           key=lambda it: -max(Chem.MolFromSmarts(x).GetNumAtoms()
                                               for x in _split_smarts(it[1])
                                               if Chem.MolFromSmarts(x) is not None))
    counts, why, form = _best_fragmentation(mol, "mg", _MG_ITEMS, with_form=True)
    if why:
        res.missing = why
        return res
    res.groups = dict(counts)
    second = _mg_second_order(form) if second_order else {}
    res.second = second
    mg2 = {row[0]: row[2:] for row in MARRERO_GANI_2}
    third: dict = {}
    if third_order:
        from rsstatchem.core.chem_draw.mg_third import third_order_groups
        third = third_order_groups(form)
        res.third = third
    mg3 = {row[0]: row[2:] for row in MARRERO_GANI_3}
    for prop in ("tb", "tm", "tc", "pc", "vc", "hf", "gf"):
        k = MG_FIELDS.index(prop)
        vals = [(n, MARRERO_GANI[g][k]) for g, n in counts.items()]
        if any(v is None for _, v in vals):
            res.values[prop] = None
            continue
        t = sum(n * v for n, v in vals)
        t += sum(n * (mg2[g][k - 1] or 0.0) for g, n in second.items())
        t += sum(n * (mg3[g][k - 1] or 0.0) for g, n in third.items())
        if prop in ("tb", "tm", "tc"):
            c = {"tb": 222.543, "tm": 147.45, "tc": 231.239}[prop]
            res.values[prop] = c * math.log(t) if t > 0 else None
        elif prop == "pc":
            res.values[prop] = (t + 0.108998) ** -2 + 5.9827
        elif prop == "vc":
            res.values[prop] = 7.95 + t
        elif prop == "hf":
            res.values[prop] = 5.549 + t
        else:
            res.values[prop] = -34.967 + t
    return res


# --- Abdulelah-Gani 'GC-simple' (critical constants, formation energies) ---------------------

# Ring aromaticity as defined for the Abdulelah-Gani data set (ugropy, MIT licence): rings RDKit
# calls aromatic but the method treats as non-aromatic (e.g. caffeine's pyrimidinedione ring,
# furan), and the rings it accepts as aromatic.
_AG_NON_AROMATIC = (
    "n1[cH0;$([cH0]=O)]nccc1", "n1ccnc[cH0;$([cH0]=O)]1",
    "n1ccn[cH0;$([cH0]=O)][cH0;$([cH0]=O)]1",
    "[nH0]1[cH0;$([cH0]=O)][nH0]cn[cH0;$([cH0]=O)]1", "c1ccco1",
    "n1[nH0][cH0;$([cH0]=O)]ccc1", "[cH0]1[cH0]cc[cH0]cc1", "[nH0;R2]1ccccc1",
    "n1[cH0;$([cH0]=*)]cccc1", "n1c[cH0;$([cH0]=*)]ccc1", "n1cc[cH0;$([cH0]=*)]cc1",
    "n1ccc[cH0;$([cH0]=*)]c1", "n1cccc[cH0;$([cH0]=*)]1", "n1cn[cH0;$([cH0]=O)]cc1")
_AG_AROMATIC = ("c1ccccc1", "c1ncncn1", "n1ccccc1", "n1ccncc1", "[nH0;R1]1[c;R1][nH0]ccc1",
                "[nH0]1[nH0]cccc1", "[nX2]1[nX2]c[nX2]cc1", "[nH0]1c[nH0]ccc1")
_AG_Q: dict = {}


def _ag_queries():
    if not _AG_Q:
        from rsstatchem.core.chem_draw.gc_tables import AG_PRIMARY, AG_SECONDARY
        _AG_Q["non"] = [Chem.MolFromSmarts(s) for s in _AG_NON_AROMATIC]
        _AG_Q["arom"] = [Chem.MolFromSmarts(s) for s in _AG_AROMATIC]
        _AG_Q["primary"] = [(row[0], Chem.MolFromSmarts(row[1]), row[2:]) for row in AG_PRIMARY]
        _AG_Q["secondary"] = [(row[0], Chem.MolFromSmarts(row[1]), row[2:])
                              for row in AG_SECONDARY]
    return _AG_Q


def _ag_preprocess(mol: Chem.Mol) -> Chem.Mol:
    """RDKit-aromatic rings re-classified with the method's own aromaticity rules."""
    q = _ag_queries()
    rw = Chem.RWMol(mol)
    kek = Chem.Mol(mol)
    Chem.Kekulize(kek, clearAromaticFlags=True)
    rings = [r for r in rw.GetRingInfo().AtomRings()
             if all(rw.GetAtomWithIdx(i).GetIsAromatic() for i in r)]

    def same_ring(patterns, ring):
        rs = set(ring)
        return any(set(m) == rs for p in patterns if p is not None
                   for m in rw.GetSubstructMatches(p))
    verdicts = []
    for ring in rings:
        if same_ring(q["non"], ring):
            verdicts.append((ring, False))
        elif same_ring(q["arom"], ring):
            verdicts.append((ring, True))
        else:
            verdicts.append((ring, all(rw.GetAtomWithIdx(i).GetSymbol() == "C"
                                       for i in ring)))
    for ring, aromatic in sorted(verdicts, key=lambda v: v[1]):   # non-aromatic first
        for i in ring:
            rw.GetAtomWithIdx(i).SetIsAromatic(aromatic)
        for k in range(len(ring)):
            a, b = ring[k], ring[(k + 1) % len(ring)]
            bond = rw.GetBondBetweenAtoms(a, b)
            bond.SetBondType(Chem.BondType.AROMATIC if aromatic else
                             kek.GetBondBetweenAtoms(a, b).GetBondType())
            bond.SetIsAromatic(aromatic)
    out = rw.GetMol()
    out.UpdatePropertyCache(False)
    return out


def _exact_cover(mol: Chem.Mol, queries, max_steps: int = 20000):
    """First exact cover of the heavy atoms by non-overlapping groups (depth-first, the most
    constrained atom first, larger groups first), as {group index: count}, or None."""
    n = mol.GetNumAtoms()
    matches = []
    for gi, (_name, q, _vals) in enumerate(queries):
        if q is None:
            continue
        for m in mol.GetSubstructMatches(q, uniquify=True, maxMatches=2000):
            matches.append((gi, frozenset(m)))
    by_atom: dict = {a: [] for a in range(n)}
    for k, (_gi, atoms) in enumerate(matches):
        for a in atoms:
            by_atom[a].append(k)
    used = [False] * n
    chosen: list[int] = []
    steps = [0]

    def search() -> bool:
        steps[0] += 1
        if steps[0] > max_steps:
            return False
        free = [a for a in range(n) if not used[a]]
        if not free:
            return True
        a = min(free, key=lambda x: len(by_atom[x]))
        options = [k for k in by_atom[a] if not any(used[b] for b in matches[k][1])]
        options.sort(key=lambda k: -len(matches[k][1]))
        for k in options:
            for b in matches[k][1]:
                used[b] = True
            chosen.append(k)
            if search():
                return True
            chosen.pop()
            for b in matches[k][1]:
                used[b] = False
        return False
    if not search():
        return None
    counts: dict = {}
    for k in chosen:
        counts[matches[k][0]] = counts.get(matches[k][0], 0) + 1
    return counts


def abdulelah_gani(mol: Chem.Mol) -> GCResult:
    """Abdulelah-Gani GC-simple estimates of Tc, Pc, Vc, Hf and Gf (primary + secondary
    groups; Alshehri, Tula, You & Gani, AIChE J. 68 (2022) e17469)."""
    from rsstatchem.core.chem_draw.gc_tables import AG_BIAS, AG_FIELDS

    res = GCResult("ag")
    res.missing = _gc_precheck(mol)
    if res.missing:
        return res
    try:
        form = _ag_preprocess(mol)
    except Exception:  # noqa: BLE001
        res.missing = "the structure could not be kekulized"
        return res
    q = _ag_queries()
    counts = _exact_cover(form, q["primary"])
    if counts is None:
        res.missing = "no complete set of Abdulelah-Gani groups"
        return res
    res.groups = {q["primary"][gi][0]: n for gi, n in counts.items()}
    sums = [0.0] * len(AG_FIELDS)
    for gi, n in counts.items():
        for k, v in enumerate(q["primary"][gi][2]):
            sums[k] += n * v
    for name, sq, vals in q["secondary"]:
        if sq is None:
            continue
        n = len(form.GetSubstructMatches(sq, uniquify=True))
        if n:
            res.second[name] = n
            for k, v in enumerate(vals):
                sums[k] += n * v
    for k, prop in enumerate(AG_FIELDS):
        s = sums[k]
        b1, b2 = AG_BIAS[prop]
        if prop == "tc":
            v = b1 * math.log(s) if s > 0 else None
        elif prop == "pc":
            v = (s + b2) ** -2 + b1 if s + b2 > 0 else None
        else:
            v = s + b1
        res.values[prop] = v
    return res


# --- the rows of the Chemical Properties window -----------------------------------------------

# short labels of the experimental data sources (see resources/properties/README.md)
EXP_SHORT = {"B": "Bradley", "O": "PHYSPROP", "A": "AqSolDB", "H": "HSDB", "I": "ICSC",
             "N": "NIOSH", "D": "DOE PAC"}
# values of one compound spread wider than this are reported as a range
EXP_CONFLICT = {"mp": 10.0, "bp": 10.0, "logp": 0.5, "logs": 0.5, "logvp": 0.5, "tc": 10.0,
                "pc": 3.0, "vc": 20.0}

# estimation methods beyond the group contributions: key -> (short label, description)
OTHER_METHODS = {
    "ml": ("ML", "machine-learning estimate: gradient-boosted trees trained on the open "
                 "experimental data (scripts/train_property_models.py)"),
    "crippen": ("Crippen", "Crippen logP / molar refractivity (Wildman & Crippen 1999, RDKit)"),
    "jplogp": ("JPlogP", "JPlogP (Plante & Werner 2018), CDK"),
    "xlogp": ("XLogP", "XLogP v2 (Wang, Fu & Lai 1997), CDK"),
    "alogp": ("ALogP", "ALogP (Ghose, Viswanadhan & Wendoloski 1998), CDK"),
    "amr": ("AMR", "AMR molar refractivity (Ghose et al. 1998), CDK"),
    "consensus": ("Consensus", "mean of Crippen, JPlogP, XLogP, ALogP and the ML logP "
                               "(SwissADME-style consensus)"),
    "consensus4": ("Consensus (4)", "mean of the four open atom/fragment methods Crippen, "
                                    "JPlogP, XLogP and ALogP"),
    "esol": ("ESOL", "ESOL estimate, Delaney 2004 (log mol/L)"),
    "ertl": ("Ertl", "topological polar surface area, Ertl 2000 (RDKit)"),
}
NOT_BIOBYTE = ("open-source alternatives; not BioByte CLogP/CMR - values differ")
# candidate estimation methods per property, in the order used when no benchmark is available
CANDIDATES = {"mp": ("ml", "mg1", "mg", "mg3", "joback"),
              "bp": ("ml", "nan", "sb", "mg", "mg3", "mg1", "joback"),
              "tc": ("nan", "mg", "mg3", "mg1", "ag", "joback"),
              "pc": ("mg3", "mg", "mg1", "nan", "ag", "joback"),
              "vc": ("mg", "mg3", "mg1", "nan", "ag", "joback"),
              "hf": ("joback", "mg", "mg3", "ag"), "gf": ("joback", "mg", "mg3", "ag"),
              "logp": ("consensus", "ml", "consensus4", "jplogp", "xlogp", "alogp", "crippen")}
UNITS = {"mp": "K", "bp": "K", "tc": "K", "pc": "bar", "vc": "cm³/mol", "logp": "",
         "logs": "", "logvp": "", "hf": "kJ/mol", "gf": "kJ/mol"}


def method_label(key: str) -> tuple[str, str]:
    return METHODS.get(key) or OTHER_METHODS.get(key) or (key, key)


_BENCH: dict | None = None


def benchmarks() -> dict:
    """Benchmark of every method against the open experimental data
    (``resources/properties/benchmarks.json``, ``scripts/benchmark_properties.py``)."""
    global _BENCH
    if _BENCH is None:
        import json

        from rsstatchem.paths import resource
        try:
            _BENCH = json.loads(resource("properties/benchmarks.json").read_text("utf-8"))
        except (OSError, ValueError):
            _BENCH = {}
    return _BENCH


def benchmark_text(prop: str, method: str) -> str:
    """'RMSE 38 K, MAE 27 K, n = 20512' for a method, or ''."""
    s = benchmarks().get(prop, {}).get("methods", {}).get(method)
    if not s:
        return ""
    unit = UNITS.get(prop, "")
    d = 1 if unit in ("K", "cm³/mol") else 2
    u = f" {unit}" if unit else ""
    cv = " (5-fold CV)" if method in ("ml", "consensus") else ""
    return f"RMSE {s['rmse']:.{d}f}{u}, MAE {s['mae']:.{d}f}{u}, n = {s['n']}{cv}"


def ranked(prop: str, available, out_of_domain: bool = False) -> list[str]:
    """Available methods ordered by their benchmark RMSE on the compounds every compared
    method covers (for a molecule outside the ML training domain: on such compounds), falling
    back to the overall RMSE and then to the static order."""
    order = list(CANDIDATES.get(prop, ()))
    b = benchmarks().get(prop, {})
    common = (b.get("ood_common") if out_of_domain else None) or b.get("common") or {}
    overall = b.get("methods", {})

    def score(m):
        s = common.get("methods", {}).get(m) or overall.get(m)
        return (0, s["rmse"]) if s else (1, order.index(m) if m in order else 99)
    return sorted((m for m in available), key=score)


@dataclass
class MethodValue:
    """One method's value for a property (the details list)."""
    key: str
    label: str
    value: float | None
    text: str
    benchmark: str = ""      # RMSE on the open data
    note: str = ""           # why missing / domain flag / sources
    default: bool = False


@dataclass
class PropertyValue:
    name: str
    value: float | None
    unit: str
    method: str
    decimals: int = 2
    note: str = ""          # why the value is missing
    short: str = ""         # short method name shown next to the value (Joback, ...)
    key: str = ""           # property key (mp, bp, logp, ...)
    kind: str = "estimate"  # "experimental", "estimate" or "computed"
    detail: str = ""        # shown after the short label: sources, expected error, flags
    warn: bool = False      # out of the ML domain / conflicting experimental values
    display: str = ""       # preformatted value (instead of value + unit)
    alternatives: list = field(default_factory=list)   # [MethodValue]

    def text(self) -> str:
        if self.value is None or (isinstance(self.value, float) and math.isnan(self.value)):
            return "n/a"
        if self.display:
            return self.display
        u = f" {self.unit}" if self.unit else ""
        return f"{self.value:.{self.decimals}f}{u}"

    def tag(self) -> str:
        """The bracketed label: '[experimental: Bradley, HSDB]', '[ML ±41 K]' ..."""
        if self.value is None or not self.short:
            return ""
        return f"[{self.short}{': ' + self.detail if self.detail else ''}]"


def _fmt(v, unit: str, decimals: int) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    return f"{v:.{decimals}f}" + (f" {unit}" if unit else "")


def _exp_detail(prop: str, e, unit: str, decimals: int) -> tuple[str, bool]:
    names = ", ".join(EXP_SHORT.get(c, c) for c in e.sources)
    detail, warn = names, False
    if e.n > 1 and e.spread > EXP_CONFLICT.get(prop, 0):
        detail += (f"; {e.n} values from {_fmt(e.low, '', decimals)} to "
                   f"{_fmt(e.high, unit, decimals)}")
        warn = True
    if e.stereo_note:
        detail += f"; {e.stereo_note}"
    return detail, warn


def _ml_detail(p, unit: str) -> tuple[str, bool]:
    if p.expected_rmse is None:
        return "", False
    d = 0 if unit == "K" else 2
    detail = f"±{p.expected_rmse:.{d}f}{' ' + unit if unit else ''} expected"
    if not p.in_domain:
        sim = math.floor(p.similarity * 100) / 100       # 0.299 must not print as 0.30
        detail += f"; outside training domain (similarity {sim:.2f} < 0.30)"
    return detail, not p.in_domain


def chemical_properties(mol: Chem.Mol | None, cdk_values: dict | None = None,
                        cdk_state: str = "") -> list[PropertyValue]:
    """Rows of the Chemical Properties window for one molecule (or an empty list).

    For each property: an experimental value from the offline database when the compound is
    in it, else the estimate whose benchmark error on the open data is lowest among the methods
    that cover the molecule; every method's value is listed in ``alternatives``.
    ``cdk_values`` are the CDK helper's results (JPlogP, XLogP, ALogP, AMR), computed in the
    background by the caller; ``cdk_state`` says why they are missing ('computing...',
    'Java not found', ...)."""
    if mol is None or mol.GetNumAtoms() == 0:
        return []
    from rsstatchem.core.chem_draw import exp_props, ml_props

    jb = joback(mol)
    jv = {"tb": jb.tb, "tm": jb.tm, "tc": jb.tc, "pc": jb.pc, "vc": jb.vc, "hf": jb.hf,
          "gf": jb.gf}
    gc: dict = {}

    def gc_result(key: str) -> GCResult:
        if key not in gc:
            try:
                gc[key] = {"sb": stein_brown, "mg": marrero_gani, "ag": abdulelah_gani,
                           "mg1": lambda m: marrero_gani(m, second_order=False),
                           "mg3": lambda m: marrero_gani(m, third_order=True)}[key](mol)
            except Exception as exc:  # noqa: BLE001 - an estimate must never break drawing
                gc[key] = GCResult(key, missing=str(exc))
        return gc[key]

    try:
        exp = exp_props.lookup(mol)
    except Exception:  # noqa: BLE001
        exp = {}
    try:
        ml = ml_props.predict_all(mol)
    except Exception:  # noqa: BLE001
        ml = {}
    gc_field = {"mp": "tm", "bp": "tb", "tc": "tc", "pc": "pc", "vc": "vc", "hf": "hf",
                "gf": "gf"}
    nan_cache: dict = {}

    def nan_result() -> dict:
        """Nannoolal estimates; Tc uses the experimental Tb when the compound has one."""
        if not nan_cache:
            from rsstatchem.core.chem_draw import nannoolal
            eb = exp.get("bp")
            try:
                nan_cache.update(nannoolal.nannoolal(mol, tb=eb.value if eb else None))
            except Exception as exc:  # noqa: BLE001 - an estimate must never break drawing
                nan_cache.update({"why": {k: str(exc) for k in ("tb", "tc", "pc", "vc",
                                                                "db")}, "tc_tb": ""})
        return nan_cache

    def estimate(prop: str, method: str):
        """(value, note, ML prediction or None)."""
        if method == "joback":
            v = jv[gc_field[prop]] if jb.ok else None
            return v, ("" if v is not None else f"{jb.missing or 'no Joback parameter'}"), None
        if method == "nan":
            r = nan_result()
            if prop not in ("bp", "tc", "pc", "vc"):
                return None, "Tb, critical constants and vapour pressure only", None
            key = {"bp": "tb"}.get(prop, prop)
            v = r.get(key)
            why = r["why"].get(key, "")
            if v is not None and prop == "tc":
                why = "with the experimental Tb" if r["tc_tb"] == "given" else \
                    "with the Nannoolal Tb estimate"
                return v, why, None
            return v, ("" if v is not None else (why or "no parameter")), None
        if method in ("sb", "mg", "mg1", "mg3", "ag"):
            if method == "sb" and prop != "bp":
                return None, "boiling point only", None
            if method == "ag" and prop in ("mp", "bp"):
                return None, "critical constants and energies only", None
            r = gc_result(method)
            v = r.values.get(gc_field[prop]) if r.ok else None
            if v is not None and not math.isfinite(v):
                v = None
            return v, ("" if v is not None else (r.missing or "no parameter")), None
        if method == "ml":
            p = ml.get(prop)
            if p is None:
                return None, "no model", None
            return p.value, p.error, p
        return None, "", None

    def gc_row(name: str, prop: str, decimals: int = 2, extra: str = "",
               use_exp: bool = True) -> PropertyValue:
        unit = UNITS[prop]
        alts: list[MethodValue] = []
        e = exp.get(prop) if use_exp else None
        if e is not None:
            detail, warn = _exp_detail(prop, e, unit, decimals)
            alts.append(MethodValue("exp", "experimental", e.value, _fmt(e.value, unit, decimals),
                                    note=exp_props.source_text(e.sources), default=True))
        values = {}
        for m in CANDIDATES[prop]:
            v, why, p = estimate(prop, m)
            values[m] = (v, why, p)
        avail = [m for m, (v, _w, _p) in values.items() if v is not None]
        mlp = values.get("ml", (None, "", None))[2]
        ood = bool(mlp is not None and mlp.value is not None and not mlp.in_domain)
        # Nannoolal Tc without an experimental Tb: judged by its benchmark with estimated Tb
        bkey = {m: m for m in CANDIDATES[prop]}
        if prop == "tc" and nan_result().get("tc_tb") == "estimated":
            bkey["nan"] = "nan_tbest"
        back = {bkey[m]: m for m in avail}
        order = [back[k] for k in ranked(prop, [bkey[m] for m in avail], ood)]
        best = order[0] if order else None
        for m in CANDIDATES[prop]:
            v, why, p = values[m]
            note = why
            if m == "ml" and p is not None and p.value is not None:
                note = _ml_detail(p, unit)[0]
            alts.append(MethodValue(m, method_label(m)[0], v, _fmt(v, unit, decimals),
                                    benchmark_text(prop, bkey[m]), note,
                                    default=(e is None and m == best)))
        if e is not None:
            return PropertyValue(name, e.value, unit,
                                 "experimental value: " + exp_props.source_text(e.sources),
                                 decimals, short="experimental", key=prop,
                                 kind="experimental", detail=detail, warn=warn,
                                 alternatives=alts)
        if best is None:
            reasons = "; ".join(f"{method_label(m)[0]}: {values[m][1] or 'no parameter'}"
                                for m in CANDIDATES[prop] if m != "ml" or values[m][1])
            return PropertyValue(name, None, unit, method_label(CANDIDATES[prop][0])[1] + extra,
                                 decimals, reasons, key=prop, alternatives=alts)
        v, why, p = values[best]
        short, long = method_label(best)
        detail, warn = (_ml_detail(p, unit) if best == "ml" else ("", False))
        if best == "nan" and prop == "tc":
            detail = why
        return PropertyValue(name, v, unit, long + extra, decimals, short=short, key=prop,
                             detail=detail, warn=warn, alternatives=alts)

    rows = [gc_row("Boiling Point", "bp")]
    bp = rows[0]
    rows.append(PropertyValue("Boiling Point (°C)", None if bp.value is None else
                              bp.value - 273.15, "°C", bp.method, 2, bp.note, bp.short,
                              key="bp_c", kind=bp.kind, detail=bp.detail, warn=bp.warn))
    rows += [
        gc_row("Melting Point", "mp"),
        gc_row("Critical Temp.", "tc"),
        gc_row("Critical Pres.", "pc"),
        gc_row("Critical Vol.", "vc", 1),
        gc_row("Gibbs Energy", "gf", extra=" (ideal gas, 298 K)", use_exp=False),
        gc_row("Heat of Form.", "hf", extra=" (ideal gas, 298 K)", use_exp=False),
    ]
    rows += _logp_rows(mol, exp, ml, cdk_values, cdk_state)
    rows[-1] = _vp_row(exp, bp, rows[2], nan_result())
    by_key = {r.key: r for r in rows}
    rows += _pka_rows(mol)
    rows.append(_henry_row(mol, by_key.get("logvp"), by_key.get("logs")))
    return rows


def _logp_rows(mol, exp: dict, ml: dict, cdk_values: dict | None, cdk_state: str):
    """LogP (experimental or the best open estimate, consensus and components in the
    details), MR, tPSA, LogS (experimental or ESOL) and the experimental vapour pressure."""
    from rsstatchem.core.chem_draw import exp_props
    try:
        crippen = Crippen.MolLogP(mol)
        mr = Crippen.MolMR(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        mw = Descriptors.MolWt(mol)
        rb = Descriptors.NumRotatableBonds(mol)
        heavy = mol.GetNumHeavyAtoms() or 1
        ap = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()) / heavy
        esol = 0.16 - 0.63 * crippen - 0.0062 * mw + 0.066 * rb - 0.74 * ap
    except Exception:  # noqa: BLE001
        crippen = mr = tpsa = esol = None
    cdk = cdk_values or {}
    cdk_note = cdk_state or ("" if cdk_values else "CDK not available")
    mlp = ml.get("logp")
    vals = {"crippen": (crippen, ""),
            "ml": (mlp.value if mlp else None,
                   (_ml_detail(mlp, "")[0] if mlp and mlp.value is not None else
                    (mlp.error if mlp else "no model")))}
    for k in ("jplogp", "xlogp", "alogp"):
        v = cdk.get(k)
        vals[k] = (v, "" if v is not None else (cdk.get("error") or cdk_note))
    comps4 = [vals[k][0] for k in ("crippen", "jplogp", "xlogp", "alogp")]
    if all(c is not None for c in comps4):
        vals["consensus4"] = (sum(comps4) / 4, "")
        if vals["ml"][0] is not None:
            vals["consensus"] = ((sum(comps4) + vals["ml"][0]) / 5, "")
        else:
            vals["consensus"] = (None, "needs the ML logP")
    else:
        why = cdk_note or "a component is missing"
        vals["consensus4"] = (None, why)
        vals["consensus"] = (None, why)
    e = exp.get("logp")
    alts = []
    if e is not None:
        alts.append(MethodValue("exp", "experimental", e.value, f"{e.value:.2f}",
                                note=exp_props.source_text(e.sources), default=True))
    avail = [m for m in CANDIDATES["logp"] if vals[m][0] is not None]
    ood = bool(mlp is not None and mlp.value is not None and not mlp.in_domain)
    order = ranked("logp", avail, ood)
    best = order[0] if order else None
    for m in CANDIDATES["logp"]:
        v, note = vals[m]
        alts.append(MethodValue(m, method_label(m)[0], v, _fmt(v, "", 2),
                                benchmark_text("logp", m), note,
                                default=(e is None and m == best)))
    if e is not None:
        detail, warn = _exp_detail("logp", e, "", 2)
        logp = PropertyValue("LogP", e.value, "", "experimental value (log Kow): " +
                             exp_props.source_text(e.sources), 2, short="experimental",
                             key="logp", kind="experimental", detail=detail, warn=warn,
                             alternatives=alts)
    elif best is not None:
        short, long = method_label(best)
        detail, warn = (_ml_detail(mlp, "") if best == "ml" else ("", False))
        if best != "ml" and benchmarks().get("logp", {}).get("methods", {}).get(best):
            detail = f"RMSE {benchmarks()['logp']['methods'][best]['rmse']:.2f} on open data"
        cons = vals["consensus"][0]
        if best != "consensus" and cons is not None:
            detail = (detail + "; " if detail else "") + f"consensus of 5 methods {cons:.2f}"
        if cdk_state and best in ("ml", "crippen"):
            detail = (detail + "; " if detail else "") + f"CDK methods: {cdk_state}"
        logp = PropertyValue("LogP", vals[best][0], "", long + "; " + NOT_BIOBYTE, 2,
                             short=short, key="logp", detail=detail, warn=warn,
                             alternatives=alts)
    else:
        logp = PropertyValue("LogP", None, "", "no logP method covers this structure", 2,
                             "no method", key="logp", alternatives=alts)
    amr = cdk.get("amr")
    mr_alts = [MethodValue("crippen", "Crippen", mr, _fmt(mr, "cm³/mol", 2), default=True),
               MethodValue("amr", "AMR", amr, _fmt(amr, "cm³/mol", 2),
                           note="" if amr is not None else (cdk.get("error") or cdk_note))]
    es = exp.get("logs")
    logs_alts = []
    if es is not None:
        logs_alts.append(MethodValue("exp", "experimental", es.value, f"{es.value:.2f}",
                                     note=exp_props.source_text(es.sources), default=True))
    logs_alts.append(MethodValue("esol", "ESOL", esol, _fmt(esol, "", 2),
                                 benchmark_text("logs", "esol"), default=es is None))
    if es is not None:
        detail, warn = _exp_detail("logs", es, "", 2)
        logs = PropertyValue("LogS", es.value, "", "experimental water solubility, log10(mol/L)"
                             " near 25 °C: " + exp_props.source_text(es.sources), 2,
                             short="experimental", key="logs", kind="experimental",
                             detail=detail, warn=warn, alternatives=logs_alts)
    else:
        logs = PropertyValue("LogS", esol, "", OTHER_METHODS["esol"][1], 2, short="ESOL",
                             key="logs", alternatives=logs_alts)
    vp = PropertyValue("Vapor Pres.", None, "Pa", "no estimate", 2, "", key="logvp")
    return [logp,
            PropertyValue("MR", mr, "cm³/mol", "Crippen molar refractivity (RDKit); " +
                          NOT_BIOBYTE.replace("CLogP/", ""), 2, short="Crippen", key="mr",
                          kind="estimate", alternatives=mr_alts),
            PropertyValue("tPSA", tpsa, "Å²", "Topological polar surface area, Ertl (RDKit)", 2,
                          short="Ertl", key="tpsa", kind="computed"),
            logs, vp]


VP_T = 298.15


def _vp_row(exp: dict, bp: PropertyValue, mp: PropertyValue, nan: dict) -> PropertyValue:
    """Vapour pressure at 25 °C: experimental, else the Nannoolal (2008) vapour-pressure
    equation with the experimental Tb (or the default Tb estimate of the Boiling Point row) and
    the group-contribution slope dB; below the melting point (experimental, else the default
    estimate) the liquid value is converted to the solid's with Walden's rule."""
    from rsstatchem.core.chem_draw import exp_props, nannoolal
    ev = exp.get("logvp")
    alts: list[MethodValue] = []
    if ev is not None:
        pa = 10 ** ev.value
        alts.append(MethodValue("exp", "experimental", pa, f"{_sig(pa)} Pa", default=True,
                                note=exp_props.source_text(ev.sources)))
    db = nan.get("db")
    est, note, detail, warn = None, "", "", False
    if db is None:
        note = nan.get("why", {}).get("db") or "no dB parameter"
    elif bp.value is None:
        note = "needs a boiling point"
    else:
        tb_src = "the experimental Tb" if bp.kind == "experimental" else \
            f"the {bp.short} Tb estimate"
        try:
            log_l = math.log10(nannoolal.vapour_pressure(VP_T, bp.value, db))
        except (ValueError, OverflowError, ZeroDivisionError):
            log_l = None
        if log_l is None or not math.isfinite(log_l):
            note = "out of range"
        else:
            tm = mp.value
            corr = nannoolal.solid_correction(VP_T, tm)
            est = 10 ** (log_l + corr)
            b = benchmarks().get("logvp", {}).get("subsets", {})
            src_key = "tb_experimental" if bp.kind == "experimental" else "tb_estimated"
            state = "solids" if corr else "liquids"
            sub = (b.get(f"{src_key}_{state}") or b.get(src_key) or {}).get("nan_s")
            detail = f"with {tb_src}"
            if corr:
                src = "experimental" if mp.kind == "experimental" else f"{mp.short} estimate"
                detail += f"; solid below Tm {tm:.0f} K ({src}), Walden conversion"
            if sub:
                detail += f"; ±{sub['rmse']:.1f} log units expected"
                warn = sub["rmse"] > 1.0
            note = detail
    bench = ""
    s = benchmarks().get("logvp", {}).get("methods", {}).get("nan_s")
    if s:
        bench = (f"RMSE {s['rmse']:.2f}, MAE {s['mae']:.2f} log10(Pa), n = {s['n']}")
    alts.append(MethodValue("nan", "Nannoolal", est, f"{_sig(est)} Pa" if est else "n/a",
                            bench, note, default=ev is None and est is not None))
    if ev is not None:
        pa = 10 ** ev.value
        detail_e, warn_e = _exp_detail("logvp", ev, "", 2)
        return PropertyValue("Vapor Pres.", pa, "Pa", "experimental vapour pressure at 25 °C: " +
                             exp_props.source_text(ev.sources), 2, short="experimental",
                             key="logvp", kind="experimental", detail=detail_e, warn=warn_e,
                             display=f"{_sig(pa)} Pa at 25 °C", alternatives=alts)
    if est is None:
        return PropertyValue("Vapor Pres.", None, "Pa", method_label("nan")[1], 2,
                             f"Nannoolal: {note}", key="logvp", alternatives=alts)
    return PropertyValue("Vapor Pres.", est, "Pa", "Nannoolal et al. (2008) vapour-pressure "
                         "estimate at 25 °C (group contribution dB)", 2, short="Nannoolal",
                         key="logvp", detail=detail, warn=warn,
                         display=f"{_sig(est)} Pa at 25 °C", alternatives=alts)


def _sig(v: float) -> str:
    """Three significant figures without scientific notation for ordinary magnitudes."""
    if v == 0 or not math.isfinite(v):
        return "0"
    if 1e-3 <= abs(v) < 1e6:
        d = max(0, 2 - int(math.floor(math.log10(abs(v)))))
        return f"{v:,.{d}f}"
    return f"{v:.2e}"


# --- pKa and Henry's law constant (WP-A) -------------------------------------------------------

def _ml_bench(metrics: dict) -> str:
    cv = metrics.get("random_cv") or {}
    if not cv:
        return ""
    return f"RMSE {cv['rmse']:.2f}, MAE {cv['mae']:.2f}, n = {cv['n']} (5-fold CV)"


def _pka_rows(mol) -> list[PropertyValue]:
    """'pKa (acidic)' and 'pKa (basic)' (pKa of the conjugate acid BH+)."""
    from rsstatchem.core.chem_draw import pka
    names = {"acid": ("pKa (acidic)", "pka_a", "strongest acidic group"),
             "base": ("pKa (basic)", "pka_b", "strongest basic group, pKa of BH+")}
    try:
        res = pka.predict(mol)
    except Exception as exc:  # noqa: BLE001 - an estimate must never break drawing
        return [PropertyValue(n, None, "", "pKa", 2, f"failed: {exc}", key=k)
                for n, k, _d in names.values()]
    rows = []
    for kind in ("acid", "base"):
        name, key, what = names[kind]
        v = getattr(res, kind)
        metrics = pka.pka_models.metrics(key)
        bench = _ml_bench(metrics)
        alts = []
        if v.experimental:
            alts.append(MethodValue("exp", "experimental", v.value, f"{v.value:.2f}",
                                    note=v.note, default=True))
        p = v.prediction
        if v.source == "ML" and p is not None:
            alts.append(MethodValue("ml", "ML", p.value, f"{p.value:.2f}", bench,
                                    _ml_detail(p, "")[0], default=True))
        if v.experimental:
            rows.append(PropertyValue(name, v.value, "", f"experimental pKa ({what}): "
                                      + pka.DATA_SOURCE, 2, short="experimental", key=key,
                                      kind="experimental", detail="Mansouri 2019 data" +
                                      (f"; {v.note}" if v.note != pka.DATA_SOURCE else ""),
                                      alternatives=alts))
        elif v.value is not None and p is not None:
            detail, warn = _ml_detail(p, "")
            rows.append(PropertyValue(name, v.value, "", f"ML estimate of the {what} "
                                      f"(gradient-boosted trees on {pka.DATA_SOURCE}); "
                                      f"groups found: {v.note}", 2, short="ML", key=key,
                                      detail=detail, warn=warn, alternatives=alts))
        else:
            rows.append(PropertyValue(name, None, "", f"pKa ({what})", 2, v.note, key=key,
                                      alternatives=alts))
    return rows


def _henry_row(mol, vp: PropertyValue | None, logs: PropertyValue | None) -> PropertyValue:
    """Henry's law constant at 25 °C in Pa m3/mol: experimental, else the estimate (ML or
    P_vap / S_w) with the lower expected error on the PHYSPROP data."""
    from rsstatchem.core.chem_draw import henry
    name, unit = "Henry's Law", "Pa·m³/mol"
    try:
        exp = henry.experimental(mol)
        ml = henry.ml(mol)
        if vp is not None and logs is not None and vp.value is not None and \
                logs.value is not None:
            ws = henry.vp_ws_estimate(mol, vp.value, logs.value,
                                      vp.kind == "experimental" and logs.kind == "experimental")
        else:
            ws = henry.HenryEstimate(None, "vpws", "needs a vapour pressure and a water "
                                     "solubility")
    except Exception as exc:  # noqa: BLE001 - an estimate must never break drawing
        return PropertyValue(name, None, unit, "Henry's law constant", 2, f"failed: {exc}",
                             key="hlc")
    vb = henry.vpws_benchmark()
    ws_sub = vb.get("both_experimental" if ws.experimental_inputs else "estimated") or {}
    ws_rmse = ws_sub.get("rmse")
    ml_rmse = ml.prediction.expected_rmse if ml.prediction is not None and \
        ml.value is not None else None
    cands = [(r, e) for r, e in ((ml_rmse, ml), (ws_rmse, ws)) if e.value is not None]
    cands.sort(key=lambda c: c[0] if c[0] is not None else 99.0)
    best = cands[0][1] if cands else None

    def txt(v):
        return f"{_sig(v)} {unit}" if v else "n/a"
    ml_bench = _ml_bench(henry.henry_models.metrics("loghl"))
    ws_bench = (f"RMSE {ws_rmse:.2f}, MAE {ws_sub['mae']:.2f} log units, n = {ws_sub['n']}"
                if ws_rmse is not None else "")
    alts = []
    if exp is not None:
        alts.append(MethodValue("exp", "experimental", exp.value, txt(exp.value),
                                note=exp.note, default=True))
    ml_note = (_ml_detail(ml.prediction, "")[0].replace("expected", "log units expected")
               if ml.value is not None else ml.note)
    alts.append(MethodValue("ml", "ML", ml.value, txt(ml.value), ml_bench + " (log10)",
                            ml_note, default=exp is None and best is ml))
    alts.append(MethodValue("vpws", "P_vap / S_w", ws.value, txt(ws.value), ws_bench,
                            ws.note, default=exp is None and best is ws))
    if exp is not None:
        return PropertyValue(name, exp.value, unit, "experimental Henry's law constant at "
                             "25 °C: " + henry.DATA_SOURCE, 2, short="experimental",
                             key="hlc", kind="experimental", detail="PHYSPROP" +
                             (f"; {exp.note}" if exp.note != henry.DATA_SOURCE else ""),
                             display=txt(exp.value), alternatives=alts)
    if best is None:
        return PropertyValue(name, None, unit, "Henry's law constant at 25 °C", 2,
                             f"ML: {ml.note}; P_vap / S_w: {ws.note}", key="hlc",
                             alternatives=alts)
    if best is ml:
        detail, warn = _ml_detail(ml.prediction, "")
        detail = detail.replace("expected", "log units expected")
        method = ("ML estimate of the Henry's law constant at 25 °C (gradient-boosted trees "
                  "on " + henry.DATA_SOURCE + ")")
        short = "ML"
    else:
        detail = f"{ws.note}; ±{ws_rmse:.1f} log units expected" if ws_rmse else ws.note
        warn = bool(ws_rmse and ws_rmse > 1.0)
        method = "Henry's law constant from vapour pressure / water solubility (25 °C)"
        short = "P/S"
    return PropertyValue(name, best.value, unit, method, 2, short=short, key="hlc",
                         detail=detail, warn=warn, display=txt(best.value), alternatives=alts)

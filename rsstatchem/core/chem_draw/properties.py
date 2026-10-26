"""Estimated physico-chemical properties for ChemDraw's *Chemical Properties* window.

Every value here is an *estimate* and is labelled with the method that produced it:

- Joback group contribution (K. G. Joback, R. C. Reid, Chem. Eng. Commun. 57 (1987) 233):
  normal boiling point, melting point, critical temperature / pressure / volume, ideal-gas
  heat and Gibbs energy of formation at 298 K. Only molecules made entirely of Joback groups
  get Joback values.
- When Joback lacks a group (e.g. caffeine's ring N-CH3), other published group-contribution
  methods are tried in turn (tables in :mod:`gc_tables`): the boiling point by Stein & Brown
  (J. Chem. Inf. Comput. Sci. 34 (1994) 581), then Marrero & Gani first-order groups (Fluid
  Phase Equilib. 183-184 (2001) 183) for every Joback property. Each value says which method
  gave it; a value stays n/a when no method covers every atom (no guessing).
- Crippen LogP and molar refractivity (Wildman & Crippen 1999, as implemented by RDKit).
- Topological polar surface area (Ertl 2000, RDKit).
- ESOL aqueous solubility LogS (Delaney 2004): 0.16 - 0.63 cLogP - 0.0062 MW + 0.066 RB
  - 0.74 AP (AP = aromatic heavy-atom proportion).

These are not ChemDraw's licensed ChemProp / BioByte (CLogP, CMR) values.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

__all__ = ["JOBACK", "JobackResult", "joback", "PropertyValue", "chemical_properties",
           "stein_brown", "marrero_gani", "GCResult", "METHODS"]

METHODS = {
    "joback": ("Joback", "Joback group contribution estimate (Joback & Reid 1987)"),
    "sb": ("Stein-Brown", "Stein & Brown group contribution estimate (1994), boiling point"),
    "mg": ("Marrero-Gani", "Marrero & Gani first-order group contribution estimate (2001)"),
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


def _best_fragmentation(mol: Chem.Mol, key: str, items) -> tuple[dict, str]:
    queries = _queries(key, items)
    first_left = None
    for form in _forms(mol):
        counts, left = _fragment(form, queries)
        if not left:
            return counts, ""
        if first_left is None:
            first_left = left
    a = mol.GetAtomWithIdx(first_left[0])
    where = "ring " if a.IsInRing() else ""
    return {}, f"no group for a {where}{a.GetSymbol()} atom with {a.GetTotalNumHs()} H"


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


def marrero_gani(mol: Chem.Mol) -> GCResult:
    """Marrero & Gani (2001) first-order estimates of Tb, Tm, Tc, Pc, Vc, Hf, Gf."""
    from rsstatchem.core.chem_draw.gc_tables import MARRERO_GANI, MG_FIELDS

    res = GCResult("mg")
    res.missing = _gc_precheck(mol)
    if res.missing:
        return res
    items = sorted(((g, row[0]) for g, row in MARRERO_GANI.items()),
                   key=lambda it: -max(Chem.MolFromSmarts(x).GetNumAtoms()
                                       for x in _split_smarts(it[1])
                                       if Chem.MolFromSmarts(x) is not None))
    counts, why = _best_fragmentation(mol, "mg", items)
    if why:
        res.missing = why
        return res
    res.groups = counts
    for prop in ("tb", "tm", "tc", "pc", "vc", "hf", "gf"):
        k = MG_FIELDS.index(prop)
        vals = [(n, MARRERO_GANI[g][k]) for g, n in counts.items()]
        if any(v is None for _, v in vals):
            res.values[prop] = None
            continue
        t = sum(n * v for n, v in vals)
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


@dataclass
class PropertyValue:
    name: str
    value: float | None
    unit: str
    method: str
    decimals: int = 2
    note: str = ""          # why the value is missing
    short: str = ""         # short method name shown next to the value (Joback, ...)

    def text(self) -> str:
        if self.value is None or (isinstance(self.value, float) and math.isnan(self.value)):
            return "n/a"
        u = f" {self.unit}" if self.unit else ""
        return f"{self.value:.{self.decimals}f}{u}"


def chemical_properties(mol: Chem.Mol | None) -> list[PropertyValue]:
    """Rows of the Chemical Properties window for one molecule (or an empty list)."""
    if mol is None or mol.GetNumAtoms() == 0:
        return []
    jb = joback(mol)
    jv = {"tb": jb.tb, "tm": jb.tm, "tc": jb.tc, "pc": jb.pc, "vc": jb.vc, "hf": jb.hf,
          "gf": jb.gf}
    lazy: dict = {}

    def fallback(key: str) -> GCResult:
        if key not in lazy:
            try:
                lazy[key] = stein_brown(mol) if key == "sb" else marrero_gani(mol)
            except Exception as exc:  # noqa: BLE001 - an estimate must never break drawing
                lazy[key] = GCResult(key, missing=str(exc))
        return lazy[key]

    def pick(prop: str):
        """(value, method key, why missing) from the first method that covers the
        molecule: Joback, then Stein-Brown (boiling point), then Marrero-Gani."""
        if jb.ok and jv[prop] is not None:
            return jv[prop], "joback", ""
        reasons = [f"Joback: {jb.missing or 'no parameter'}"]
        for key in (("sb", "mg") if prop == "tb" else ("mg",)):
            r = fallback(key)
            v = r.values.get(prop) if r.ok else None
            if v is not None and math.isfinite(v):
                return v, key, ""
            reasons.append(f"{METHODS[key][0]}: {r.missing or 'no parameter'}")
        return None, "joback", "; ".join(reasons)

    def row(name, prop, unit, decimals=2, extra="", convert=None):
        v, key, why = pick(prop)
        short, long = METHODS[key]
        if convert is not None and v is not None:
            v = convert(v)
        return PropertyValue(name, v, unit, long + extra, decimals, why,
                             short if v is not None else "")

    rows = [
        row("Boiling Point", "tb", "K"),
        row("Boiling Point (°C)", "tb", "°C", convert=lambda v: v - 273.15),
        row("Melting Point", "tm", "K"),
        row("Critical Temp.", "tc", "K"),
        row("Critical Pres.", "pc", "bar"),
        row("Critical Vol.", "vc", "cm³/mol", 1),
        row("Gibbs Energy", "gf", "kJ/mol", extra=" (ideal gas, 298 K)"),
        row("Heat of Form.", "hf", "kJ/mol", extra=" (ideal gas, 298 K)"),
    ]
    try:
        logp = Crippen.MolLogP(mol)
        mr = Crippen.MolMR(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        mw = Descriptors.MolWt(mol)
        rb = Descriptors.NumRotatableBonds(mol)
        heavy = mol.GetNumHeavyAtoms() or 1
        ap = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()) / heavy
        logs = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rb - 0.74 * ap
    except Exception:  # noqa: BLE001
        logp = mr = tpsa = logs = None
    rows += [
        PropertyValue("LogP", logp, "", "Crippen estimate (RDKit); not BioByte CLogP", 2,
                      short="Crippen"),
        PropertyValue("MR", mr, "cm³/mol", "Crippen molar refractivity (RDKit); not CMR", 2,
                      short="Crippen"),
        PropertyValue("tPSA", tpsa, "Å²", "Topological polar surface area, Ertl (RDKit)", 2,
                      short="Ertl"),
        PropertyValue("LogS", logs, "", "ESOL estimate, Delaney 2004 (log mol/L)", 2,
                      short="ESOL"),
    ]
    return rows

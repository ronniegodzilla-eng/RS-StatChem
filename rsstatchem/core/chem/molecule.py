"""Molecules with RDKit: reading/writing, properties, drug-likeness and an editor bridge.

Mirrors ChemDraw's "Chemical Properties" / "Analysis" windows and the descriptor panels of
DataWarrior / SwissADME for the descriptors chemists report in theses: molecular formula,
MW, exact mass, Crippen cLogP and MR, TPSA, H-bond donors/acceptors, rotatable bonds, rings,
aromatic rings, heavy atoms, fraction Csp3, formal charge, QED, plus Lipinski (Ro5), Veber,
Ghose and Egan rules.

Stage-2 editor bridge: :class:`AtomSpec` / :class:`BondSpec` lists ⇄ RDKit molecules
(:func:`mol_from_graph`, :func:`graph_from_mol`), with stereo perceived from wedge/hash bonds
and 2D coordinates, plus :func:`problems` for live valence feedback. All RDKit errors are
converted to :class:`ChemError` (a ``ValueError``) with a readable message.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Callable

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, QED, rdChemReactions, rdDepictor, rdMolDescriptors

__all__ = ["ChemError", "mol_from_text", "detect_format", "read_file", "write_file",
           "to_smiles", "to_molblock", "to_inchi", "to_inchikey", "to_sdf_text",
           "molecular_formula", "compute_2d", "has_2d", "standardize", "largest_fragment",
           "properties", "Descriptor", "DESCRIPTORS", "descriptor", "RuleResult",
           "drug_likeness", "RULES", "AtomSpec", "BondSpec", "mol_from_graph",
           "graph_from_mol", "problems", "reaction_from_text", "reaction_to_smiles",
           "substructure_matches"]

RDLogger.DisableLog("rdApp.*")
rdDepictor.SetPreferCoordGen(True)


class ChemError(ValueError):
    """Invalid structure input or an RDKit failure, with a user-readable message."""


# --- reading / writing ---------------------------------------------------------------------

def detect_format(text: str) -> str:
    t = text.strip()
    if t.startswith("InChI="):
        return "inchi"
    if "M  END" in t or "V2000" in t or "V3000" in t:
        return "sdf" if "$$$$" in t else "molblock"
    if ">>" in t or (t.count(">") >= 2 and " " not in t.split()[0]):
        return "reaction"
    return "smiles"


def _sanitize(mol: Chem.Mol, what: str) -> Chem.Mol:
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:  # noqa: BLE001 - RDKit raises several exception types
        raise ChemError(f"{what}: {_clean(exc)}") from None
    return mol


def _clean(exc: Exception) -> str:
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    return text.replace("Explicit valence", "explicit valence")


def mol_from_text(text: str, fmt: str | None = None, sanitize: bool = True) -> Chem.Mol:
    """Molecule from SMILES, SMARTS (``fmt="smarts"``), InChI, MOL block or SDF text.

    The format is detected when ``fmt`` is None. A SMILES may be followed by a name
    (``"CCO ethanol"``); the name becomes the ``_Name`` property."""
    if text is None or not str(text).strip():
        raise ChemError("Empty structure")
    text = str(text)
    fmt = (fmt or detect_format(text)).lower()
    if fmt == "reaction":
        raise ChemError("This is a reaction; use reaction_from_text().")
    if fmt == "smiles":
        parts = text.strip().split(None, 1)
        mol = Chem.MolFromSmiles(parts[0], sanitize=sanitize)
        if mol is None:
            raw = Chem.MolFromSmiles(parts[0], sanitize=False)
            if raw is None:
                raise ChemError(f"Invalid SMILES '{parts[0]}'")
            _sanitize(raw, "Invalid SMILES")
            raise ChemError(f"Invalid SMILES '{parts[0]}'")
        if len(parts) > 1:
            mol.SetProp("_Name", parts[1].strip())
        return mol
    if fmt == "smarts":
        mol = Chem.MolFromSmarts(text.strip())
        if mol is None:
            raise ChemError(f"Invalid SMARTS '{text.strip()}'")
        return mol
    if fmt == "inchi":
        mol = Chem.MolFromInchi(text.strip(), sanitize=sanitize)
        if mol is None:
            raise ChemError("Invalid InChI")
        return mol
    if fmt in ("molblock", "mol", "sdf"):
        block = text.split("$$$$")[0]
        mol = Chem.MolFromMolBlock(block, sanitize=sanitize, removeHs=sanitize)
        if mol is None:
            raw = Chem.MolFromMolBlock(block, sanitize=False, removeHs=False)
            if raw is None:
                raise ChemError("Invalid MOL block")
            _sanitize(raw, "Invalid MOL block")
            raise ChemError("Invalid MOL block")
        return mol
    raise ChemError(f"Unknown structure format '{fmt}'")


def read_file(path: str) -> list[Chem.Mol]:
    """All molecules of a .mol/.sdf/.sd/.smi/.txt/.mol2/.pdb/.inchi file (invalid ones are
    skipped; raises if none could be read)."""
    ext = os.path.splitext(path)[1].lower()
    mols: list = []
    if ext in (".sdf", ".sd"):
        mols = [m for m in Chem.SDMolSupplier(path) if m is not None]
    elif ext == ".mol":
        m = Chem.MolFromMolFile(path)
        mols = [m] if m is not None else []
    elif ext == ".mol2":
        m = Chem.MolFromMol2File(path)
        mols = [m] if m is not None else []
    elif ext == ".pdb":
        m = Chem.MolFromPDBFile(path)
        mols = [m] if m is not None else []
    else:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or line.lower().startswith("smiles"):
                    continue
                try:
                    mols.append(mol_from_text(line))
                except ChemError:
                    continue
    if not mols:
        raise ChemError(f"No valid molecule could be read from {os.path.basename(path)}")
    return mols


def write_file(mols, path: str) -> None:
    """Write molecules to .sdf, .mol (first molecule) or .smi."""
    mols = [mols] if isinstance(mols, Chem.Mol) else list(mols)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".sdf", ".sd"):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(to_sdf_text(mols))
    elif ext == ".mol":
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(to_molblock(mols[0]))
    else:
        with open(path, "w", encoding="utf-8") as fh:
            for m in mols:
                name = m.GetProp("_Name") if m.HasProp("_Name") else ""
                fh.write(f"{to_smiles(m)} {name}".rstrip() + "\n")


def to_smiles(mol: Chem.Mol, canonical: bool = True, isomeric: bool = True,
              kekule: bool = False) -> str:
    return Chem.MolToSmiles(mol, canonical=canonical, isomericSmiles=isomeric,
                            kekuleSmiles=kekule)


def to_molblock(mol: Chem.Mol, v3000: bool = False, coords: bool = True) -> str:
    m = Chem.Mol(mol)
    if coords and not has_2d(m) and m.GetNumConformers() == 0:
        compute_2d(m)
    return Chem.MolToV3KMolBlock(m) if v3000 else Chem.MolToMolBlock(m)


def to_sdf_text(mols) -> str:
    out = []
    for m in ([mols] if isinstance(mols, Chem.Mol) else mols):
        block = to_molblock(m)
        props = "".join(f">  <{k}>\n{v}\n\n" for k, v in m.GetPropsAsDict().items()
                        if not k.startswith("_"))
        out.append(block + props + "$$$$\n")
    return "".join(out)


def to_inchi(mol: Chem.Mol) -> str:
    return Chem.MolToInchi(mol) or ""


def to_inchikey(mol: Chem.Mol) -> str:
    return Chem.MolToInchiKey(mol) or ""


def molecular_formula(mol: Chem.Mol) -> str:
    return rdMolDescriptors.CalcMolFormula(mol)


# --- coordinates and clean-up --------------------------------------------------------------

def has_2d(mol: Chem.Mol) -> bool:
    if mol.GetNumConformers() == 0:
        return False
    conf = mol.GetConformer()
    if conf.Is3D():
        return False
    pos = conf.GetPositions()
    return mol.GetNumAtoms() <= 1 or float(abs(pos[:, :2]).sum()) > 0


def compute_2d(mol: Chem.Mol, template: Chem.Mol | None = None,
               coordgen: bool = True) -> Chem.Mol:
    """(Re)generate 2D coordinates in place ("Clean Structure"); optional template
    alignment keeps a common core in the same orientation. Returns the molecule."""
    rdDepictor.SetPreferCoordGen(coordgen)
    try:
        if template is not None and mol.HasSubstructMatch(template):
            if not has_2d(template):
                rdDepictor.Compute2DCoords(template)
            rdDepictor.GenerateDepictionMatching2DStructure(mol, template)
        else:
            rdDepictor.Compute2DCoords(mol)
    finally:
        rdDepictor.SetPreferCoordGen(True)
    return mol


def largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    return max(frags, key=lambda m: (m.GetNumHeavyAtoms(), m.GetNumAtoms())) if frags else mol


def standardize(mol: Chem.Mol, strip_salts: bool = True, neutralize: bool = True) -> Chem.Mol:
    """RDKit MolStandardize clean-up, largest fragment (salt stripping), neutralisation."""
    from rdkit.Chem.MolStandardize import rdMolStandardize
    m = rdMolStandardize.Cleanup(mol)
    if strip_salts:
        m = rdMolStandardize.FragmentParent(m)
    if neutralize:
        m = rdMolStandardize.Uncharger().uncharge(m)
    return m


# --- descriptors ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Descriptor:
    key: str           # short name, also the base name of saved variables
    label: str
    fn: Callable
    decimals: int = 3
    integer: bool = False


DESCRIPTORS: list[Descriptor] = [
    Descriptor("MW", "Molecular weight (g/mol)", Descriptors.MolWt, 3),
    Descriptor("ExactMass", "Exact (monoisotopic) mass", Descriptors.ExactMolWt, 5),
    Descriptor("cLogP", "cLogP (Crippen)", Crippen.MolLogP, 3),
    Descriptor("MR", "Molar refractivity (Crippen)", Crippen.MolMR, 3),
    Descriptor("TPSA", "Topological polar surface area (Å²)", rdMolDescriptors.CalcTPSA, 2),
    Descriptor("HBD", "H-bond donors", Descriptors.NumHDonors, 0, True),
    Descriptor("HBA", "H-bond acceptors", Descriptors.NumHAcceptors, 0, True),
    Descriptor("RotB", "Rotatable bonds", Descriptors.NumRotatableBonds, 0, True),
    Descriptor("Rings", "Rings", rdMolDescriptors.CalcNumRings, 0, True),
    Descriptor("AroRings", "Aromatic rings", rdMolDescriptors.CalcNumAromaticRings, 0, True),
    Descriptor("HeavyAtoms", "Heavy atoms", lambda m: m.GetNumHeavyAtoms(), 0, True),
    Descriptor("Atoms", "Atoms (incl. H)", lambda m: Chem.AddHs(m).GetNumAtoms(), 0, True),
    Descriptor("Heteroatoms", "Heteroatoms", rdMolDescriptors.CalcNumHeteroatoms, 0, True),
    Descriptor("FCsp3", "Fraction Csp3", rdMolDescriptors.CalcFractionCSP3, 3),
    Descriptor("Charge", "Formal charge", Chem.GetFormalCharge, 0, True),
    Descriptor("Stereo", "Stereocentres",
               lambda m: len(Chem.FindMolChiralCenters(m, includeUnassigned=True,
                                                       useLegacyImplementation=False)), 0, True),
    Descriptor("QED", "QED drug-likeness", QED.qed, 3),
]
_BY_KEY = {d.key: d for d in DESCRIPTORS}


def descriptor(key: str) -> Descriptor:
    return _BY_KEY[key]


def properties(mol: Chem.Mol, keys: list[str] | None = None) -> dict:
    """Ordered dict ``{"Formula": ..., "MW": ..., "cLogP": ...}`` of the chosen descriptors
    (all by default). Values are floats/ints; a failing descriptor gives NaN."""
    out: dict = {"Formula": molecular_formula(mol)}
    for d in DESCRIPTORS:
        if keys is not None and d.key not in keys:
            continue
        try:
            v = d.fn(mol)
            out[d.key] = int(v) if d.integer else float(v)
        except Exception:  # noqa: BLE001 - one bad descriptor must not lose the others
            out[d.key] = math.nan
    return out


# --- drug-likeness -------------------------------------------------------------------------

@dataclass
class RuleResult:
    rule: str
    reference: str
    criteria: list = field(default_factory=list)   # (criterion, value, limit text, passed)
    violations: int = 0
    allowed: int = 0
    passed: bool = True


def _check(name, value, lo, hi):
    ok = (lo is None or value >= lo) and (hi is None or value <= hi)
    if lo is None:
        limit = f"≤ {hi:g}"
    elif hi is None:
        limit = f"≥ {lo:g}"
    else:
        limit = f"{lo:g} – {hi:g}"
    return (name, value, limit, ok)


def _lipinski(p):
    return [_check("Molecular weight", p["MW"], None, 500), _check("cLogP", p["cLogP"], None, 5),
            _check("H-bond donors", p["HBD"], None, 5),
            _check("H-bond acceptors", p["HBA"], None, 10)]


def _veber(p):
    return [_check("Rotatable bonds", p["RotB"], None, 10), _check("TPSA", p["TPSA"], None, 140)]


def _ghose(p):
    return [_check("Molecular weight", p["MW"], 160, 480), _check("cLogP", p["cLogP"], -0.4, 5.6),
            _check("Molar refractivity", p["MR"], 40, 130), _check("Atoms", p["Atoms"], 20, 70)]


def _egan(p):
    return [_check("cLogP", p["cLogP"], None, 5.88), _check("TPSA", p["TPSA"], None, 131.6)]


# name: (function, violations allowed, reference)
RULES = {
    "Lipinski": (_lipinski, 1, "Lipinski et al. (1997), Rule of Five; ≤ 1 violation allowed"),
    "Veber": (_veber, 0, "Veber et al. (2002), oral bioavailability"),
    "Ghose": (_ghose, 0, "Ghose et al. (1999), drug-like ranges"),
    "Egan": (_egan, 0, "Egan et al. (2000), absorption"),
}


def drug_likeness(mol: Chem.Mol, rules: list[str] | None = None) -> list[RuleResult]:
    p = properties(mol)
    out = []
    for name in rules or list(RULES):
        fn, allowed, ref = RULES[name]
        crit = fn(p)
        v = sum(1 for c in crit if not c[3])
        out.append(RuleResult(name, ref, crit, v, allowed, v <= allowed))
    return out


def substructure_matches(mol: Chem.Mol, query: str) -> tuple[list[int], list[int]]:
    """Atom and bond indices matched by a SMARTS (or SMILES) query, for highlighting."""
    q = Chem.MolFromSmarts(query) or Chem.MolFromSmiles(query)
    if q is None:
        raise ChemError(f"Invalid substructure query '{query}'")
    atoms: set[int] = set()
    bonds: set[int] = set()
    for match in mol.GetSubstructMatches(q):
        atoms.update(match)
        for b in q.GetBonds():
            bond = mol.GetBondBetweenAtoms(match[b.GetBeginAtomIdx()], match[b.GetEndAtomIdx()])
            if bond is not None:
                bonds.add(bond.GetIdx())
    return sorted(atoms), sorted(bonds)


# --- reactions -----------------------------------------------------------------------------

def reaction_from_text(text: str, smarts: bool | None = None):
    """``ChemicalReaction`` from reaction SMILES (``CCO.CC(=O)O>>CCOC(C)=O``) or SMARTS."""
    t = (text or "").strip()
    if ">" not in t:
        raise ChemError("A reaction needs '>>' between reactants and products.")
    use_smiles = not smarts if smarts is not None else not any(c in t for c in "&;$~")
    try:
        rxn = rdChemReactions.ReactionFromSmarts(t.split()[0], useSmiles=use_smiles)
    except Exception as exc:  # noqa: BLE001
        raise ChemError(f"Invalid reaction: {_clean(exc)}") from None
    if rxn is None or rxn.GetNumReactantTemplates() + rxn.GetNumProductTemplates() == 0:
        raise ChemError("Invalid reaction")
    for i in range(rxn.GetNumReactantTemplates()):
        _try_sanitize(rxn.GetReactantTemplate(i))
    for i in range(rxn.GetNumProductTemplates()):
        _try_sanitize(rxn.GetProductTemplate(i))
    return rxn


def _try_sanitize(m) -> None:
    try:
        m.UpdatePropertyCache(strict=False)
        flags = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
        Chem.SanitizeMol(m, flags,
                         catchErrors=True)
    except Exception:  # noqa: BLE001
        pass


def reaction_to_smiles(rxn) -> str:
    return rdChemReactions.ReactionToSmiles(rxn)


# --- editor bridge (stage 2) ---------------------------------------------------------------

_ORDER = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE,
          1.5: Chem.BondType.AROMATIC, 0: Chem.BondType.DATIVE, 4: Chem.BondType.QUADRUPLE}
_ORDER_BACK = {v: k for k, v in _ORDER.items()}


@dataclass
class AtomSpec:
    """One atom of an editor drawing (2D coordinates in Å-like units, y up)."""

    symbol: str = "C"
    x: float = 0.0
    y: float = 0.0
    charge: int = 0
    isotope: int = 0
    explicit_h: int | None = None     # None = implicit hydrogens from valence
    radicals: int = 0
    map_number: int = 0
    label: str = ""                   # free text shown instead of the symbol (abbreviation)


@dataclass
class BondSpec:
    """A bond: order 1, 2, 3, 1.5 (aromatic), 0 (dative); stereo '' | 'wedge' | 'hash' |
    'wavy' (single bonds, wide end at ``end``) or 'either' (crossed double bond)."""

    begin: int
    end: int
    order: float = 1
    stereo: str = ""


def mol_from_graph(atoms: list[AtomSpec], bonds: list[BondSpec],
                   sanitize: bool = True) -> Chem.Mol:
    """Build an RDKit molecule from an editor graph; tetrahedral stereo comes from
    wedge/hash bonds and double-bond geometry from the 2D coordinates."""
    rw = Chem.RWMol()
    conf = Chem.Conformer(len(atoms))
    for i, a in enumerate(atoms):
        sym = "*" if a.symbol in ("R", "*", "X", "A") else a.symbol
        atom = Chem.Atom(sym)
        atom.SetFormalCharge(int(a.charge))
        if a.isotope:
            atom.SetIsotope(int(a.isotope))
        if a.explicit_h is not None:
            atom.SetNumExplicitHs(int(a.explicit_h))
            atom.SetNoImplicit(True)
        if a.radicals:
            atom.SetNumRadicalElectrons(int(a.radicals))
        if a.map_number:
            atom.SetAtomMapNum(int(a.map_number))
        if a.label:
            atom.SetProp("atomLabel", a.label)
        rw.AddAtom(atom)
        conf.SetAtomPosition(i, (float(a.x), float(a.y), 0.0))
    for b in bonds:
        if b.begin == b.end or rw.GetBondBetweenAtoms(b.begin, b.end) is not None:
            continue
        rw.AddBond(int(b.begin), int(b.end), _ORDER.get(b.order, Chem.BondType.SINGLE))
        bond = rw.GetBondBetweenAtoms(int(b.begin), int(b.end))
        if b.order == 1.5:
            bond.SetIsAromatic(True)
            rw.GetAtomWithIdx(b.begin).SetIsAromatic(True)
            rw.GetAtomWithIdx(b.end).SetIsAromatic(True)
        if b.stereo == "wedge":
            bond.SetBondDir(Chem.BondDir.BEGINWEDGE)
        elif b.stereo == "hash":
            bond.SetBondDir(Chem.BondDir.BEGINDASH)
        elif b.stereo == "wavy":
            bond.SetBondDir(Chem.BondDir.UNKNOWN)
        elif b.stereo == "either":
            bond.SetBondDir(Chem.BondDir.EITHERDOUBLE)
    conf.Set3D(False)
    rw.AddConformer(conf, assignId=True)
    mol = rw.GetMol()
    if not sanitize:
        return mol
    _sanitize(mol, "Invalid structure")
    Chem.AssignChiralTypesFromBondDirs(mol)
    Chem.DetectBondStereochemistry(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    for b in mol.GetBonds():
        if b.GetBondDir() == Chem.BondDir.EITHERDOUBLE:
            b.SetStereo(Chem.BondStereo.STEREOANY)
    return mol


def graph_from_mol(mol: Chem.Mol, kekulize: bool = True,
                   coords: bool = True) -> tuple[list[AtomSpec], list[BondSpec]]:
    """Editor graph from a molecule (2D coordinates generated when missing; wedges chosen
    by RDKit to express the stereocentres)."""
    m = Chem.Mol(mol)
    if coords and not has_2d(m):
        compute_2d(m)
    unspecified = set()
    try:
        for info in Chem.FindPotentialStereo(m):
            if info.type == Chem.StereoType.Bond_Double and \
                    info.specified == Chem.StereoSpecified.Unspecified:
                unspecified.add(info.centeredOn)
    except Exception:  # noqa: BLE001
        pass
    if kekulize:
        try:
            Chem.Kekulize(m, clearAromaticFlags=True)
        except Exception:  # noqa: BLE001 - leave aromatic bonds as 1.5
            pass
    Chem.WedgeMolBonds(m, m.GetConformer()) if m.GetNumConformers() else None
    conf = m.GetConformer() if m.GetNumConformers() else None
    atoms = []
    for a in m.GetAtoms():
        pos = conf.GetAtomPosition(a.GetIdx()) if conf is not None else None
        atoms.append(AtomSpec(
            a.GetSymbol() if a.GetAtomicNum() else "*",
            float(pos.x) if pos is not None else 0.0, float(pos.y) if pos is not None else 0.0,
            a.GetFormalCharge(), a.GetIsotope(),
            a.GetNumExplicitHs() if a.GetNoImplicit() else None,
            a.GetNumRadicalElectrons(), a.GetAtomMapNum(),
            a.GetProp("atomLabel") if a.HasProp("atomLabel") else ""))
    bonds = []
    for b in m.GetBonds():
        d = b.GetBondDir()
        stereo = {Chem.BondDir.BEGINWEDGE: "wedge", Chem.BondDir.BEGINDASH: "hash",
                  Chem.BondDir.UNKNOWN: "wavy"}.get(d, "")
        if b.GetStereo() == Chem.BondStereo.STEREOANY or b.GetIdx() in unspecified:
            stereo = "either"
        bonds.append(BondSpec(b.GetBeginAtomIdx(), b.GetEndAtomIdx(),
                              _ORDER_BACK.get(b.GetBondType(), 1), stereo))
    return atoms, bonds


def problems(mol: Chem.Mol) -> list[tuple[int, str]]:
    """(atom index, message) for valence problems — for red marks in an editor; an empty
    list means the structure sanitises."""
    m = Chem.Mol(mol)
    out = []
    for a in m.GetAtoms():
        a.UpdatePropertyCache(strict=False)
    try:
        issues = Chem.DetectChemistryProblems(m)
    except Exception as exc:  # noqa: BLE001
        return [(-1, _clean(exc))]
    for issue in issues:
        idx = issue.GetAtomIdx() if hasattr(issue, "GetAtomIdx") else -1
        out.append((idx, issue.Message()))
    return out

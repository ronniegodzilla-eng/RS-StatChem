"""Biopolymer sequences for the drawing: HELM and FASTA (ChemDraw's Edit > Paste Special >
HELM / FASTA Peptide / FASTA DNA / FASTA RNA and Copy As > HELM), on RDKit's sequence support
(``MolFromHELM``, ``MolFromFASTA``, ``MolFromSequence``; BSD-3).

Pasting builds the full-atom structure of the sequence. Copying works for molecules built from
a sequence (RDKit keeps the monomer information) and, for drawn structures, for linear
peptides of the 20 standard amino acids: the peptide bonds are cut, every residue is matched
against the free amino acids and the residues are ordered from the N-terminus.
"""
from __future__ import annotations

from rdkit import Chem

__all__ = ["KINDS", "parse_fasta", "mols_from_fasta", "mol_from_helm", "looks_like_helm",
           "looks_like_fasta", "peptide_sequence", "to_helm", "to_fasta", "SequenceError"]

# FASTA flavour -> RDKit MolFromSequence flavor (L-peptide, uncapped RNA / DNA)
KINDS = {"peptide": 0, "rna": 2, "dna": 6}
_PEPTIDE_BOND = Chem.MolFromSmarts("[NX3;!$(N=*)][CX4][CX3](=O)!@[NX3][CX4]")


class SequenceError(ValueError):
    pass


def parse_fasta(text: str) -> list[tuple[str, str]]:
    """[(header, sequence)] of FASTA text; plain sequences without a header are accepted."""
    records: list[tuple[str, list[str]]] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith(">"):
            records.append((line[1:].strip(), []))
            continue
        if not records:
            records.append(("", []))
        records[-1][1].append("".join(ch for ch in line if not ch.isspace() and ch not in
                                      "*-0123456789"))
    out = [(h, "".join(parts).upper()) for h, parts in records]
    return [(h, s) for h, s in out if s]


def looks_like_fasta(text: str) -> bool:
    t = (text or "").strip()
    return t.startswith(">") and "\n" in t


def looks_like_helm(text: str) -> bool:
    t = (text or "").strip()
    return "{" in t and "}" in t and t.split("{", 1)[0].rstrip("0123456789") in (
        "PEPTIDE", "RNA", "CHEM", "BLOB")


def mols_from_fasta(text: str, kind: str = "peptide") -> list[Chem.Mol]:
    """Structures of the FASTA records (kind: peptide, dna or rna)."""
    flavor = KINDS.get(kind.lower())
    if flavor is None:
        raise SequenceError(f"Unknown sequence type '{kind}'")
    recs = parse_fasta(text)
    if not recs:
        raise SequenceError("No sequence found")
    mols = []
    for header, seq in recs:
        if kind == "rna":
            seq = seq.replace("T", "U")
        elif kind == "dna":
            seq = seq.replace("U", "T")
        m = Chem.MolFromSequence(seq, flavor=flavor)
        if m is None or m.GetNumAtoms() == 0:
            raise SequenceError(f"'{seq[:30]}' is not a valid {kind} sequence"
                                + (f" ({header})" if header else ""))
        mols.append(m)
    return mols


def mol_from_helm(text: str) -> Chem.Mol:
    t = " ".join((text or "").split())
    m = Chem.MolFromHELM(t)
    if (m is None or m.GetNumAtoms() == 0) and "$" not in t:
        m = Chem.MolFromHELM(t + "$$$$")
    if m is None or m.GetNumAtoms() == 0:
        raise SequenceError("Not a HELM string RDKit can read (peptides and simple RNA)")
    return m


def _residue_table() -> tuple[dict, dict]:
    iso, flat = {}, {}
    for code in "ACDEFGHIKLMNPQRSTVWY":
        m = Chem.MolFromSequence(code)
        iso[Chem.MolToSmiles(m)] = code
        flat[Chem.MolToSmiles(m, isomericSmiles=False)] = code
    return iso, flat


_TABLE: tuple | None = None


def peptide_sequence(mol: Chem.Mol) -> tuple[str, bool]:
    """(one-letter sequence N->C, stereo given) of a drawn linear peptide; raises
    ``SequenceError`` if the molecule is not one."""
    global _TABLE
    if _TABLE is None:
        _TABLE = _residue_table()
    iso, flat = _TABLE
    m = Chem.RemoveHs(mol)
    bonds, links = [], []
    for match in m.GetSubstructMatches(_PEPTIDE_BOND):
        b = m.GetBondBetweenAtoms(match[2], match[4])
        if b is not None and b.GetIdx() not in bonds:
            bonds.append(b.GetIdx())
            links.append((match[2], match[4]))
    if not bonds:
        raise SequenceError("No peptide bond found")
    frag = Chem.FragmentOnBonds(m, bonds, addDummies=True)
    rw = Chem.RWMol(frag)
    for a in rw.GetAtoms():
        if a.GetAtomicNum() == 0:
            # the dummy on the carbonyl carbon becomes OH, the one on the nitrogen H
            nb = a.GetNeighbors()[0]
            a.SetAtomicNum(1 if nb.GetAtomicNum() == 7 else 8)
            a.SetIsotope(0)
            a.SetNoImplicit(False)
    frag = rw.GetMol()
    Chem.SanitizeMol(frag)
    atom_frag: dict[int, int] = {}
    pieces = Chem.GetMolFrags(frag, asMols=False, sanitizeFrags=False)
    for k, atoms in enumerate(pieces):
        for i in atoms:
            atom_frag[i] = k
    frag_mols = Chem.GetMolFrags(frag, asMols=True)
    codes, stereo = [], True
    for fm in frag_mols:
        fm = Chem.RemoveHs(fm)
        smi = Chem.MolToSmiles(fm)
        code = iso.get(smi)
        if code is None:
            code = flat.get(Chem.MolToSmiles(fm, isomericSmiles=False))
            if code is None:
                raise SequenceError("A residue is not one of the 20 standard amino acids")
            if code != "G":
                stereo = False
        codes.append(code)
    nxt = {atom_frag[c]: atom_frag[n] for c, n in links}
    starts = set(range(len(codes))) - set(nxt.values())
    if len(starts) != 1 or len(nxt) != len(codes) - 1:
        raise SequenceError("Only linear, unbranched peptides can be written as a sequence")
    order, cur = [], starts.pop()
    while cur is not None:
        order.append(cur)
        cur = nxt.get(cur)
    if len(order) != len(codes):
        raise SequenceError("Only linear peptides can be written as a sequence")
    return "".join(codes[i] for i in order), stereo


def to_helm(mol: Chem.Mol) -> str:
    """HELM of a molecule built from a sequence, or of a drawn linear peptide."""
    try:
        helm = Chem.MolToHELM(mol)
    except Exception:  # noqa: BLE001
        helm = ""
    if helm:
        return helm
    seq, _ = peptide_sequence(mol)
    return "PEPTIDE1{" + ".".join(seq) + "}$$$$"


def to_fasta(mol: Chem.Mol, title: str = "") -> str:
    try:
        fasta = Chem.MolToFASTA(mol)
    except Exception:  # noqa: BLE001
        fasta = ""
    if fasta and fasta.strip() not in ("", ">"):
        lines = fasta.strip().splitlines()
        return f">{title or lines[0][1:].strip() or 'sequence'}\n" + "\n".join(lines[1:]) + "\n"
    seq, _ = peptide_sequence(mol)
    body = "\n".join(seq[i:i + 60] for i in range(0, len(seq), 60))
    return f">{title or 'peptide'}\n{body}\n"

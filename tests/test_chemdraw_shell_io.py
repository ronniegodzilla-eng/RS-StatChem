"""Chemical Drawing, WP-C file formats and interop (no windows): binary CDX import / export,
MOL / RXN V3000 with enhanced stereo, HELM / FASTA sequences, Clean Up Reaction and the
optional Indigo atom mapping."""
from __future__ import annotations

import struct

import pytest
from rdkit import Chem
from rdkit.Chem import rdChemDraw, rdChemReactions

from rsstatchem.core.chem_draw import cdx, cdxml, convert, mapping, sequence
from rsstatchem.core.chem_draw.model import Document
from rsstatchem.core.chem_draw.style import preset

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def _doc(text: str = ASPIRIN) -> Document:
    d = Document(preset("ACS Document 1996"))
    convert.insert_text(d, text, 200.0, 200.0)
    return d


def _rdkit_smiles_of_cdx(data: bytes) -> list[str]:
    """Chemistry of CDX bytes as RDKit reads it (through the CDXML twin: RDKit's own CDX
    reader fails on binary input in the Python wheel)."""
    return sorted(Chem.MolToSmiles(m) for m in
                  rdChemDraw.MolsFromChemDrawBlock(cdx.cdx_to_cdxml(data)))


# --- CDX: hand-built file in ChemDraw's layout ----------------------------------------------

def _prop(tag: int, data: bytes) -> bytes:
    return struct.pack("<HH", tag, len(data)) + data


def _chemdraw_style_cdx() -> bytes:
    """A CDX file assembled byte by byte as ChemDraw writes it (28-byte header, document
    properties, page, fragment of C-O with an O label, a caption): independent of our
    writer, so it checks the reader's record layouts."""
    coord = 65536
    out = b"VjCD0100" + struct.pack("<I", 0x01020304) + bytes(16)
    out += _prop(0x0003, struct.pack("<H", 0) + b"ChemDraw 17.1")
    out += _prop(0x0100, struct.pack("<HHHHH", 1, 1, 3, 1252, 5) + b"Arial")
    out += _prop(0x0805, struct.pack("<i", int(14.4 * coord)))          # BondLength
    out += struct.pack("<HI", 0x8001, 1)                                 # page
    out += struct.pack("<HI", 0x8003, 2)                                 # fragment
    out += struct.pack("<HI", 0x8004, 3)                                 # node C
    out += _prop(0x0200, struct.pack("<2i", 100 * coord, 50 * coord))    # y=100, x=50
    out += b"\0\0"
    out += struct.pack("<HI", 0x8004, 4)                                 # node O
    out += _prop(0x0200, struct.pack("<2i", 100 * coord, int(64.4 * coord)))
    out += _prop(0x0402, struct.pack("<h", 8))
    out += _prop(0x0421, struct.pack("<b", -1))                          # charge -1
    out += b"\0\0"
    out += struct.pack("<HI", 0x8005, 5)                                 # bond 3-4
    out += _prop(0x0604, struct.pack("<I", 3)) + _prop(0x0605, struct.pack("<I", 4))
    out += b"\0\0"
    out += b"\0\0"                                                       # end fragment
    text = b"Methoxide"
    out += struct.pack("<HI", 0x8006, 6)
    out += _prop(0x0200, struct.pack("<2i", 130 * coord, 40 * coord))
    out += _prop(0x0700, struct.pack("<H", 1) + struct.pack("<5H", 0, 3, 1, 200, 0) + text)
    out += b"\0\0"
    out += b"\0\0"                                                       # end page
    out += b"\0\0"                                                       # end document
    return out


def test_cdx_reader_decodes_chemdraw_layout():
    data = _chemdraw_style_cdx()
    assert cdx.is_cdx(data)
    xml = cdx.cdx_to_cdxml(data)
    assert 'CreationProgram="ChemDraw 17.1"' in xml and 'BondLength="14.4"' in xml
    assert 'p="50 100"' in xml                       # Point2D stored y, x
    assert 'Charge="-1"' in xml and 'name="Arial"' in xml
    assert [Chem.MolToSmiles(m) for m in rdChemDraw.MolsFromChemDrawBlock(xml)] == ["C[O-]"]
    doc = cdx.from_cdx(data)
    assert convert.to_smiles(doc) == "C[O-]"
    [t] = doc.texts.values()
    assert t.text == "Methoxide" and t.bold
    a = min(doc.atoms.values(), key=lambda a: a.x)
    assert (a.x, a.y) == (50.0, 100.0)


def test_cdx_rejects_other_data():
    with pytest.raises(ValueError):
        cdx.cdx_to_cdxml(b"<CDXML/>")
    assert not cdx.is_cdx(b"PK\x03\x04")


# --- CDX: our writer ------------------------------------------------------------------------

def test_cdx_round_trip_structures_and_graphics(tmp_path):
    d = _doc()
    convert.insert_text(d, "[NH4+].[Cl-]", 420.0, 200.0)
    d.add_text(150.0, 300.0, "Aspirin and NH4Cl", chemical=False, bold=True)
    d.add_curve([[300, 300], [300, 300], [320, 280], [340, 280], [360, 300], [360, 300]],
                arrow_end="full", color="#E00000")
    d.add_orbital(500.0, 350.0, "p", fill="shaded")
    d.add_bracket(150.0, 150.0, 260.0, 250.0, "round")
    path = tmp_path / "a.cdx"
    cdx.save_cdx(d, str(path))
    data = path.read_bytes()
    assert data[:8] == b"VjCD0100" and len(data) > 500
    assert _rdkit_smiles_of_cdx(data) == sorted([ASPIRIN, "[NH4+]", "[Cl-]"])
    back = cdx.load_cdx(str(path))
    assert convert.to_smiles(back) == convert.to_smiles(d)
    assert sorted((round(a.x, 1), round(a.y, 1)) for a in back.atoms.values()) == \
        sorted((round(a.x, 1), round(a.y, 1)) for a in d.atoms.values())
    assert [t.text for t in back.texts.values()] == ["Aspirin and NH4Cl"]
    assert next(iter(back.texts.values())).bold
    [c] = back.curves.values()
    assert c.arrow_end == "full" and c.color == "#E00000"
    assert len(back.orbitals) == 1 and next(iter(back.orbitals.values())).fill == "shaded"
    assert next(iter(back.brackets.values())).kind == "round"
    assert abs(back.style.bond_length - d.style.bond_length) < 0.01


def test_cdx_reaction_scheme_is_read_as_reaction(tmp_path):
    d = _doc("CC(=O)O.OCC>>CC(=O)OCC.O")
    data = cdx.to_cdx(d)
    xml = cdx.cdx_to_cdxml(data)
    assert "<scheme" in xml and "ReactionStepReactants" in xml
    path = tmp_path / "r.cdxml"
    path.write_text(xml, encoding="utf-8")
    [rxn] = rdChemDraw.ReactionsFromChemDrawFile(str(path))
    assert rdChemReactions.ReactionToSmiles(rxn) == "CC(=O)O.CCO>>CCOC(C)=O.O"
    assert convert.reaction_smiles(cdx.from_cdx(data)) == "CC(=O)O.CCO>>CCOC(C)=O.O"
    arrow = next(iter(cdx.from_cdx(data).arrows.values()))
    a0 = next(iter(d.arrows.values()))
    assert (round(arrow.x1, 1), round(arrow.x2, 1)) == (round(a0.x1, 1), round(a0.x2, 1))


def test_cdx_from_rdkit_cdxml_and_back():
    """CDXML written by RDKit (Revvity's code) survives CDXML -> CDX -> CDXML."""
    for smi in ("c1ccc2ccccc2c1", "C[C@H](N)C(=O)O", "O=C([O-])c1ccccc1.[Na+]"):
        block = rdChemDraw.MolToChemDrawBlock(Chem.MolFromSmiles(smi))
        again = cdx.cdx_to_cdxml(cdx.cdxml_to_cdx(block))
        got = sorted(Chem.MolToSmiles(m) for m in rdChemDraw.MolsFromChemDrawBlock(again))
        want = sorted(Chem.MolToSmiles(m) for m in rdChemDraw.MolsFromChemDrawBlock(block))
        assert got == want


def test_cdx_duplicate_ids_are_renumbered():
    """Our CDXML gives the page id 1 and an atom id 1; the CDX ids must stay unique and the
    bonds must keep pointing at their atoms."""
    d = _doc("CC(=O)O")
    assert convert.to_smiles(cdx.from_cdx(cdx.to_cdx(d))) == "CC(=O)O"


# --- V3000 ----------------------------------------------------------------------------------

def test_v3000_export_keeps_stereo_groups():
    mb = Chem.MolToV3KMolBlock(Chem.MolFromSmiles("C[C@H](O)[C@@H](N)C |&1:1,o1:3|"))
    d = Document()
    convert.insert_text(d, mb, 0.0, 0.0)
    groups = convert.stereo_groups(d)
    assert sorted(g["type"] for g in groups) == ["AND", "OR"]
    out = convert.to_molblock(d, v3000=True)
    assert "V3000" in out and "MDLV30/STERAC1" in out and "MDLV30/STEREL1" in out
    m = Chem.MolFromMolBlock(out)
    kinds = sorted(str(g.GetGroupType()) for g in m.GetStereoGroups())
    assert kinds == ["STEREO_AND", "STEREO_OR"]
    # the groups live in the document and survive .rsc
    again = Document.from_json(d.to_json())
    assert "MDLV30/STERAC1" in convert.to_molblock(again, v3000=True)
    assert "V2000" in convert.to_molblock(_doc("CCO"))


def test_rxn_v3000_and_sdf_v3000():
    d = _doc("CC(=O)O.OCC>>CC(=O)OCC.O")
    rxn = convert.reaction_rxn(d, v3000=True)
    assert rxn.startswith("$RXN V3000")
    r = rdChemReactions.ReactionFromRxnBlock(rxn)
    assert r.GetNumReactantTemplates() == 2 and r.GetNumProductTemplates() == 2
    assert convert.to_sdf(_doc("CCO.CCN"), v3000=True).count("V3000") == 2


# --- HELM / FASTA ---------------------------------------------------------------------------

def test_fasta_and_helm_to_structure():
    assert sequence.parse_fasta(">pep one\nAG C\n>two\nkw\n") == [("pep one", "AGC"),
                                                                 ("two", "KW")]
    [m] = sequence.mols_from_fasta(">x\nAGC", "peptide")
    assert Chem.MolToSmiles(m) == "C[C@H](N)C(=O)NCC(=O)N[C@@H](CS)C(=O)O"
    [dna] = sequence.mols_from_fasta("ACGT", "dna")
    [rna] = sequence.mols_from_fasta("ACGU", "rna")
    assert dna.GetNumAtoms() > 60 and rna.GetNumAtoms() > 60
    assert Chem.MolToSmiles(sequence.mol_from_helm("PEPTIDE1{A.G.C}$$$$")) == \
        Chem.MolToSmiles(m)
    assert sequence.looks_like_helm("PEPTIDE1{A.G}$$$$") and not sequence.looks_like_helm("CCO")
    d = Document()
    convert.insert_text(d, "PEPTIDE1{A.G.C}$$$$", 0.0, 0.0)          # HELM pasted as text
    assert convert.to_smiles(d) == Chem.MolToSmiles(m)
    with pytest.raises(sequence.SequenceError):
        sequence.mols_from_fasta("", "peptide")


def test_drawn_peptide_to_helm_and_fasta():
    for seq in ("AGC", "GPKW", "ACDEFGHIKLMNPQRSTVWY"):
        drawn = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSequence(seq)))
        assert sequence.peptide_sequence(drawn) == (seq, True)
        assert sequence.to_helm(drawn) == "PEPTIDE1{" + ".".join(seq) + "}$$$$"
        assert sequence.to_fasta(drawn).splitlines()[1] == seq
    # no stereo drawn: residues still found, flagged as unspecified
    assert sequence.peptide_sequence(Chem.MolFromSmiles("CC(N)C(=O)NCC(=O)O")) == ("AG", False)
    for bad in ("CCO", "C1CC(=O)NCC(=O)N1"):
        with pytest.raises(sequence.SequenceError):
            sequence.to_helm(Chem.MolFromSmiles(bad))
    # a document round trip (drawn structure loses RDKit's monomer info)
    d = Document()
    convert.insert_text(d, "PEPTIDE1{G.A.V}$$$$", 0.0, 0.0)
    assert sequence.to_helm(convert.to_mol(d)) == "PEPTIDE1{G.A.V}$$$$"


# --- Clean Up Reaction ----------------------------------------------------------------------

def test_clean_up_reaction_spacing_plus_and_arrow():
    d = _doc("CC(=O)O.OCC>>CC(=O)OCC.O")
    first = d.fragments()[0]
    d.move({("atom", a) for a in first}, -60.0, 45.0)              # messy drawing
    d.remove([("plus", p) for p in d.pluses])
    assert convert.clean_up_reaction(d)
    assert convert.reaction_smiles(d) == "CC(=O)O.CCO>>CCOC(C)=O.O"
    assert len(d.pluses) == 2
    [ar] = d.arrows.values()
    y = ar.y1
    assert ar.y2 == y and ar.x2 > ar.x1 and ar.x2 - ar.x1 >= d.style.bond_length * 3.5
    spans = [(min(d.atoms[i].x for i in c), max(d.atoms[i].x for i in c),
              (min(d.atoms[i].y for i in c) + max(d.atoms[i].y for i in c)) / 2)
             for c in d.fragments()]
    for x0, x1, cy in spans:
        assert abs(cy - y) < 1e-6                                   # all on the arrow line
    assert spans[1][1] < ar.x1 < ar.x2 < spans[2][0]                # reactants | arrow | products
    for p in d.pluses.values():
        assert abs(p.y - y) < 1e-6
    assert not convert.clean_up_reaction(_doc("CCO"))


# --- atom mapping (Indigo optional) ---------------------------------------------------------

def test_mapping_unavailable_explains(monkeypatch):
    monkeypatch.setattr(mapping, "_STATE", {"indigo": None, "error": "No module named "
                                            "'indigo'"})
    assert not mapping.available()
    assert "Indigo" in mapping.why_unavailable() and "by hand" in mapping.why_unavailable()
    with pytest.raises(mapping.MappingError):
        mapping.map_reaction(_doc("CBr.[OH-]>>CO.[Br-]"))
    d = _doc("CBr.[OH-]>>CO.[Br-]")
    for a in d.atoms.values():
        a.map_number = 3
    assert mapping.clear_map(d) == len(d.atoms)
    assert all(a.map_number == 0 for a in d.atoms.values())


@pytest.mark.skipif(not mapping.available(), reason="Indigo (epam.indigo) is not installed")
def test_indigo_automap():
    d = _doc("CCBr.[OH-]>>CCO.[Br-]")
    result = mapping.map_reaction(d)
    assert result and mapping.indigo_version()
    rsmi = convert.reaction_smiles(d)
    left, _, right = rsmi.split(">")
    nums = {n for n in range(1, 20) if f":{n}]" in left and f":{n}]" in right}
    assert len(nums) >= 2                      # the ethyl carbons are mapped on both sides


@pytest.mark.skipif(not mapping.available(), reason="Indigo (epam.indigo) is not installed")
def test_automap_two_products_uses_each_atom_once():
    """Aspirin synthesis: Indigo gives anhydride atoms to both products; the checked result
    maps every reactant atom once and each number to one product atom of the same element."""
    d = _doc("OC(=O)c1ccccc1O.CC(=O)OC(C)=O>>CC(=O)Oc1ccccc1C(=O)O.CC(=O)O")
    mapping.map_reaction(d)
    split = convert.split_reaction(d)
    left = [d.atoms[a] for comp in split.reactants for a in comp]
    right = [d.atoms[a] for comp in split.products for a in comp]
    nums = [a.map_number for a in right if a.map_number]
    assert len(nums) == len(set(nums)) == 17            # 13 + 4 product atoms, all mapped
    by_number = {a.map_number: a for a in left}
    assert len(by_number) == 17 and 0 not in by_number
    for a in right:
        assert a.symbol == by_number[a.map_number].symbol


def test_mapping_check_and_mcs_fallback():
    from rdkit import Chem

    r = [Chem.MolFromSmiles("CC(=O)OC(C)=O")]
    p = [Chem.MolFromSmiles("CC(=O)O"), Chem.MolFromSmiles("CC(=O)O")]
    # the same reactant atoms in both products: invalid
    assert not mapping.valid_mapping(r, p, [[1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4], [1, 2, 3, 4]])
    numbers = mapping.mcs_mapping(r, p)
    assert mapping.valid_mapping(r, p, numbers)
    assert sorted(n for nums in numbers[1:] for n in nums) == [0] + list(range(1, 8))


def test_rdkit_cdx_support_flag():
    assert isinstance(cdx.rdkit_cdx_support(), bool)
    assert cdxml.to_cdxml(_doc()).startswith("<?xml")

"""Chemistry stage 2: the ChemDraw-style structure / reaction editor.

Core (no Qt): document model, .rsc round trip, editing operations -> canonical SMILES /
InChIKey (references from RDKit on the published SMILES / InChIKeys), nicknames, stereo
from wedges and from 2D geometry, label rules, reactions (split, balance, stoichiometry).
UI (offscreen Qt): simulated mouse events on the canvas, undo/redo, exports (SVG parses,
PNG carries the requested dpi), Output Viewer / Data Editor integration.
"""
from __future__ import annotations

import json
import math
import os
import xml.etree.ElementTree as ET

import pytest
from rdkit import Chem
from rdkit.Chem import rdChemReactions

from rsstatchem.core.chem import molecule as mo
from rsstatchem.core.chem_draw import abbrev, analysis, convert, geometry, labels, ops
from rsstatchem.core.chem_draw.model import Document, load_rsc, save_rsc
from rsstatchem.core.chem_draw.style import ACS_1996, preset

S_IBUPROFEN = "HEFNNWSXXWATRW-JTQLQIEISA-N"
R_IBUPROFEN = "HEFNNWSXXWATRW-SNVBAGLBSA-N"


def key_of(smiles: str) -> str:
    return mo.to_inchikey(mo.mol_from_text(smiles))


def canon(smiles: str) -> str:
    return mo.to_smiles(mo.mol_from_text(smiles))


# --- document model --------------------------------------------------------------------

def test_model_basics_and_fragments():
    d = Document()
    a = d.add_atom(0, 0)
    b = d.add_atom(30, 0, "O")
    bond = d.add_bond(a.id, b.id)
    assert d.add_bond(b.id, a.id) is bond          # no duplicate bonds
    c = d.add_atom(200, 0, "N")
    assert d.fragments() == [[a.id, b.id], [c.id]]
    assert d.neighbors(a.id) == [b.id]
    d.remove([("atom", b.id)])
    assert not d.bonds and b.id not in d.atoms


def test_ring_detection_and_merge():
    d = Document()
    ring = ops.stamp_ring(d, "cyclohexane", point=(0, 0))
    assert len(d.rings()) == 1 and sorted(d.rings()[0]) == sorted(ring)
    assert all(len(d.neighbors(i)) == 2 for i in ring)
    x = d.add_atom(d.atoms[ring[0]].x + 1, d.atoms[ring[0]].y)
    y, _ = ops.add_bond_at_atom(d, x.id)
    d.merge_coincident([x.id], 5.0)
    assert x.id not in d.atoms and len(d.neighbors(ring[0])) == 3


def test_rsc_round_trip(tmp_path):
    d = Document(preset("ACS Document 1996"))
    convert.insert_text(d, "CC(=O)O.OCC>>CC(=O)OCC.O", 0, 0)
    ar = next(iter(d.arrows.values()))
    ar.above, ar.below = "H2SO4 (cat.)", "reflux, 2 h"
    d.add_text(10, 80, "Fischer esterification", bold=True)
    d.add_bracket(0, 0, 50, 40, "round", "n")
    d.add_arrow(0, 100, 40, 100, "fishhook", bend=-12)
    first = next(iter(d.atoms.values()))
    first.isotope, first.map_number, first.radicals = 13, 4, 1
    path = tmp_path / "rx.rsc"
    save_rsc(d, str(path))
    again = load_rsc(str(path))
    assert again.to_dict() == d.to_dict()
    assert json.loads(path.read_text())["format"] == "rs-statchem-drawing"
    assert again.style.bond_length == pytest.approx(14.4)
    assert convert.reaction_smiles(again) == convert.reaction_smiles(d)


def test_subset_and_paste_keep_chemistry():
    d = Document()
    convert.insert_text(d, "c1ccccc1O", 0, 0)
    sub = d.subset([("atom", i) for i in d.atoms])
    d2 = Document()
    keys = d2.paste(sub, 100, 0)
    assert len([k for k in keys if k[0] == "atom"]) == 7
    assert convert.to_smiles(d2) == "Oc1ccccc1"


# --- editing operations -> chemistry -----------------------------------------------------

def test_benzene_template():
    d = Document()
    ops.stamp_ring(d, "benzene", point=(0, 0))
    assert convert.to_smiles(d) == "c1ccccc1"
    assert convert.to_inchikey(d) == "UHOVQNZJYSORNB-UHFFFAOYSA-N"


def test_naphthalene_by_fusing_on_a_bond_keeps_kekule():
    d = Document()
    r = ops.stamp_ring(d, "benzene", point=(0, 0))
    b = next(b for b in d.bonds.values() if b.order == 2)
    ops.stamp_ring(d, "benzene", bond=b.id)
    assert convert.to_smiles(d) == "c1ccc2ccccc2c1"
    assert not convert.structure_problems(d)
    assert len(r) == 6 and len(d.atoms) == 10


def test_aspirin_from_bonds_and_nicknames():
    d = Document()
    r = ops.stamp_ring(d, "benzene", point=(0, 0))
    a, _ = ops.add_bond_at_atom(d, r[1])
    convert.apply_label_text(d, a, "COOH")
    b, _ = ops.add_bond_at_atom(d, r[2])
    convert.apply_label_text(d, b, "OAc")
    assert convert.to_smiles(d) == canon("CC(=O)Oc1ccccc1C(=O)O")
    assert convert.to_inchikey(d) == "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"


def test_aspirin_atom_by_atom():
    d = Document()
    r = ops.stamp_ring(d, "benzene", point=(0, 0))
    c, _ = ops.add_bond_at_atom(d, r[1])
    o1, _ = ops.add_bond_at_atom(d, c, 2)
    o2, _ = ops.add_bond_at_atom(d, c)
    o3, _ = ops.add_bond_at_atom(d, r[2])
    c2, _ = ops.add_bond_at_atom(d, o3)
    o4, _ = ops.add_bond_at_atom(d, c2, 2)
    ops.add_bond_at_atom(d, c2)
    for i in (o1, o2, o3, o4):
        ops.set_element(d, i, "O")
    assert convert.to_inchikey(d) == "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"


def _ibuprofen(stereo: str) -> Document:
    d = Document()
    r = ops.stamp_ring(d, "benzene", point=(0, 0))
    c, _ = ops.add_bond_at_atom(d, r[0])                 # stereocentre above the ring
    me, bme = ops.add_bond_at_atom(d, c)                 # methyl, upper left
    co, _ = ops.add_bond_at_atom(d, c)                   # carboxyl C, upper right
    o1, _ = ops.add_bond_at_atom(d, co, 2)
    o2, _ = ops.add_bond_at_atom(d, co)
    ops.set_element(d, o1, "O")
    ops.set_element(d, o2, "O")
    chain = ops.add_chain(d, r[3], math.pi / 2, 2)       # CH2-CH below the ring
    ops.add_bond_at_atom(d, chain[-1])
    ops.add_bond_at_atom(d, chain[-1])
    assert d.atoms[me].x < d.atoms[c].x < d.atoms[co].x
    ops.apply_bond_tool(d, bme, stereo, near=c)          # narrow end on the stereocentre
    return d


def test_ibuprofen_stereocentre_from_wedge_and_hash():
    wedge, hashed = _ibuprofen("wedge"), _ibuprofen("hash")
    # COOH upper right -> aryl below -> CH3 upper left runs clockwise; with the methyl
    # wedged towards the viewer the H points away, so the wedge drawing is (R).
    mw, mh = convert.to_mol(wedge), convert.to_mol(hashed)
    assert Chem.FindMolChiralCenters(mw) == [(Chem.FindMolChiralCenters(mw)[0][0], "R")]
    assert Chem.FindMolChiralCenters(mh)[0][1] == "S"
    assert convert.to_inchikey(wedge) == R_IBUPROFEN
    assert convert.to_inchikey(hashed) == S_IBUPROFEN
    assert convert.to_smiles(hashed) == canon("CC(C)Cc1ccc(cc1)[C@H](C)C(=O)O")


def test_wedge_click_reverses_and_sets_narrow_end():
    d = _ibuprofen("wedge")
    b = next(b for b in d.bonds.values() if b.style == "wedge")
    a0 = b.a
    ops.apply_bond_tool(d, b.id, "wedge")
    assert b.a != a0                                     # second click reverses
    ops.apply_bond_tool(d, b.id, "wedge")
    assert b.a == a0


def test_2_butene_E_and_Z_from_geometry():
    d = Document()
    ids = ops.add_chain(d, (0, 0), 0.0, 3)
    d.bond_between(ids[1], ids[2]).order = 2
    assert convert.to_inchikey(d) == key_of("C/C=C/C")
    # mirror the last carbon through the C=C axis -> cis
    a, b, c = (d.atoms[i] for i in ids[1:])
    ux, uy = b.x - a.x, b.y - a.y
    L = math.hypot(ux, uy)
    ux, uy = ux / L, uy / L
    px, py = c.x - b.x, c.y - b.y
    dot = px * ux + py * uy
    c.x, c.y = b.x + 2 * dot * ux - px, b.y + 2 * dot * uy - py
    assert convert.to_inchikey(d) == key_of("C/C=C\\C")
    assert convert.to_smiles(d) == "C/C=C\\C"


def test_glycine_zwitterion_with_charges():
    d = Document()
    n = d.add_atom(0, 0, "N")
    c, _ = ops.add_bond_at_atom(d, n.id)
    c2, _ = ops.add_bond_at_atom(d, c)
    o1, _ = ops.add_bond_at_atom(d, c2, 2)
    o2, _ = ops.add_bond_at_atom(d, c2)
    ops.set_element(d, o1, "O")
    ops.set_element(d, o2, "O")
    ops.change_charge(d, n.id, 1)
    ops.change_charge(d, o2, -1)
    assert convert.to_smiles(d) == canon("[NH3+]CC(=O)[O-]")
    lay = labels.layout(d, d.atoms[n.id])
    assert lay.text == "H3N+"                            # bond to the right: H on the left
    res = analysis.analyze(d)
    assert res.formula == "C2H5NO2" and res.charge == 0


def test_nickname_expansion_methyl_benzoate():
    d = Document()
    a = d.add_atom(0, 0)
    b, _ = ops.add_bond_at_atom(d, a.id)
    convert.apply_label_text(d, a.id, "Ph")
    convert.apply_label_text(d, b, "CO2Me")
    assert convert.to_smiles(d) == canon("COC(=O)c1ccccc1")
    assert labels.layout(d, d.atoms[b]).text == "CO2Me"
    assert labels.layout(d, d.atoms[a.id]).text == "Ph"
    new = convert.expand_label(d, b)
    assert len(new) == 4 and convert.to_smiles(d) == canon("COC(=O)c1ccccc1")


@pytest.mark.parametrize("label, smiles", [
    ("Me", "Cc1ccccc1"), ("Et", "CCc1ccccc1"), ("Pr", "CCCc1ccccc1"),
    ("iPr", "CC(C)c1ccccc1"), ("tBu", "CC(C)(C)c1ccccc1"), ("Ph", "c1ccc(-c2ccccc2)cc1"),
    ("Bn", "c1ccc(Cc2ccccc2)cc1"), ("Ac", "CC(=O)c1ccccc1"), ("OMe", "COc1ccccc1"),
    ("OH", "Oc1ccccc1"), ("COOH", "OC(=O)c1ccccc1"), ("CHO", "O=Cc1ccccc1"),
    ("CN", "N#Cc1ccccc1"), ("NO2", "O=[N+]([O-])c1ccccc1"), ("NH2", "Nc1ccccc1"),
    ("CF3", "FC(F)(F)c1ccccc1"), ("Boc", "CC(C)(C)OC(=O)c1ccccc1"),
    ("Ts", "Cc1ccc(S(=O)(=O)c2ccccc2)cc1"), ("TMS", "C[Si](C)(C)c1ccccc1"),
])
def test_abbreviations_on_benzene(label, smiles):
    d = Document()
    r = ops.stamp_ring(d, "benzene", point=(0, 0))
    a, _ = ops.add_bond_at_atom(d, r[0])
    convert.apply_label_text(d, a, label)
    assert convert.to_inchikey(d) == key_of(smiles)


def test_typed_labels_become_elements():
    d = Document()
    a = d.add_atom(0, 0)
    b, _ = ops.add_bond_at_atom(d, a.id)
    convert.apply_label_text(d, b, "OH")
    assert d.atoms[b].symbol == "O" and not d.atoms[b].label
    convert.apply_label_text(d, b, "NH2")
    assert d.atoms[b].symbol == "N"
    convert.apply_label_text(d, b, "R1")
    assert d.atoms[b].label == "R1" and "*" in convert.to_smiles(d)
    assert abbrev.label_text("CO2H", True) == ("HO2C", "C")


def test_isotope_radical_map_number():
    d = Document()
    a = d.add_atom(0, 0)
    a.isotope = 13
    assert convert.to_smiles(d) == "[13CH4]"
    b = d.add_atom(100, 0)
    ops.cycle_radical(d, b.id)
    ops.toggle_map_number(d, b.id)
    m = convert.to_mol(d, [b.id])
    atom = m.GetAtomWithIdx(0)
    assert atom.GetNumRadicalElectrons() == 1 and atom.GetAtomMapNum() == 1
    assert atom.GetTotalNumHs() == 3


def test_valence_problems_marked():
    d = Document()
    c = d.add_atom(0, 0)
    for _ in range(5):
        ops.add_bond_at_atom(d, c.id)
    probs = convert.structure_problems(d)
    assert list(probs) == [c.id] and "valence" in probs[c.id]
    assert not analysis.analyze(d).ok


# --- labels / geometry -------------------------------------------------------------------

def test_label_rules():
    d = Document()
    convert.insert_text(d, "OCC(=O)C", 0, 0)
    o = next(a for a in d.atoms.values() if a.symbol == "O" and len(d.neighbors(a.id)) == 1
             and d.bond_between(a.id, d.neighbors(a.id)[0]).order == 1)
    lay = labels.layout(d, o)
    assert lay.text in ("OH", "HO")
    nb = d.atoms[d.neighbors(o.id)[0]]
    assert (lay.text == "HO") == (nb.x - o.x > 0.1 * d.style.bond_length)
    carbons = [a for a in d.atoms.values() if a.symbol == "C"]
    assert not any(labels.layout(d, a).visible for a in carbons)
    ketone = next(b for b in d.bonds.values() if b.order == 2)
    assert geometry.double_bond_side(d, ketone, {o.id for o in d.atoms.values()
                                                 if o.symbol == "O"}) == 0
    methane = Document()
    methane.add_atom(0, 0)
    assert labels.layout(methane, next(iter(methane.atoms.values()))).text == "CH4"
    water = Document()
    water.add_atom(0, 0, "O")
    assert labels.layout(water, next(iter(water.atoms.values()))).text == "H2O"


def test_pyrrole_nh_points_out_of_ring_and_ring_double_bonds_inside():
    d = Document()
    r = ops.stamp_ring(d, "cyclopentadiene", point=(0, 0))
    sp3 = next(i for i in r if all(b.order == 1 for b in d.bonds_of(i)))
    ops.set_element(d, sp3, "N")
    assert convert.to_smiles(d) == "c1cc[nH]c1"
    cx = sum(d.atoms[i].x for i in r) / 5
    cy = sum(d.atoms[i].y for i in r) / 5
    n = d.atoms[sp3]
    vec = {"right": (1, 0), "left": (-1, 0), "up": (0, -1), "down": (0, 1)}
    v = vec[labels.h_side(d, n)]
    assert v[0] * (n.x - cx) + v[1] * (n.y - cy) > 0
    # N at the bottom of a ring (bonds up-left / up-right): H stacked below, as ChemDraw
    m = Document()
    nn = m.add_atom(0, 0, "N")
    for dx in (-26, 26):
        m.add_bond(nn.id, m.add_atom(dx, -15).id)
    lay = labels.layout(m, nn)
    assert lay.stacked_pos == "down" and lay.stacked[0][0] == "H"
    for b in d.bonds.values():
        if b.order != 2:
            continue
        side = geometry.double_bond_side(d, b)
        a1, a2 = d.atoms[b.a], d.atoms[b.b]
        nx, ny = -(a2.y - a1.y), a2.x - a1.x
        assert side * (nx * (cx - a1.x) + ny * (cy - a1.y)) > 0


def test_new_bond_angle_zigzag_and_chain():
    d = Document()
    ids = ops.add_chain(d, (0, 0), 0.0, 6)
    for i in range(1, 6):
        a, b, c = (d.atoms[j] for j in ids[i - 1:i + 2])
        v1 = (a.x - b.x, a.y - b.y)
        v2 = (c.x - b.x, c.y - b.y)
        ang = math.degrees(math.acos((v1[0] * v2[0] + v1[1] * v2[1]) /
                                     (math.hypot(*v1) * math.hypot(*v2))))
        assert ang == pytest.approx(120, abs=1e-6)
    for b in d.bonds.values():
        p, q = d.atoms[b.a], d.atoms[b.b]
        assert math.hypot(p.x - q.x, p.y - q.y) == pytest.approx(d.style.bond_length)
    assert geometry.snap_angle(math.radians(37)) == pytest.approx(math.radians(30))


def test_bond_primitives():
    st = ACS_1996
    d = Document(st)
    convert.insert_text(d, "C[C@H](O)CC", 0, 0)
    wedge = next(b for b in d.bonds.values() if b.style in ("wedge", "hash"))
    prims = geometry.bond_primitives(d, wedge, st)
    if wedge.style == "wedge":
        assert prims[0][0] == "poly" and len(prims[0][1]) == 4
    else:
        assert all(p[0] == "line" for p in prims) and len(prims) > 3
    ar = d.add_arrow(0, 0, 50, 0, "equilibrium")
    kinds = [p[0] for p in geometry.arrow_primitives(ar, st)]
    assert kinds.count("poly") == 2 and kinds.count("line") == 2


# --- import / export ---------------------------------------------------------------------

@pytest.mark.parametrize("smiles", [
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "CC(C)Cc1ccc(cc1)[C@H](C)C(=O)O",
    "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O", "C/C=C\\C", "[NH3+]CC(=O)[O-]",
    "c1ccc2[nH]ccc2c1", "O=[N+]([O-])c1ccccc1", "[13CH3]O", "CC#N", "C1CC2CCC1C2",
])
def test_smiles_round_trip(smiles):
    d = Document()
    convert.insert_text(d, smiles, 0, 0)
    assert convert.to_smiles(d) == canon(smiles)


def test_mol_round_trip():
    src = mo.to_molblock(mo.mol_from_text("CC(C)Cc1ccc(cc1)[C@H](C)C(=O)O"))
    d = Document()
    convert.insert_text(d, src, 0, 0)
    out = convert.to_molblock(d)
    m = Chem.MolFromMolBlock(out)
    assert m is not None and m.GetNumAtoms() == 15
    assert mo.to_inchikey(m) == S_IBUPROFEN
    assert "V2000" in out
    sdf = convert.to_sdf(d)
    assert sdf.count("$$$$") == 1


def test_inchi_import():
    d = Document()
    convert.insert_text(d, "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3", 0, 0)
    assert convert.to_smiles(d) == "CCO"
    assert convert.to_inchi(d) == "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"


def test_clean_structure_keeps_stereo_and_bond_length():
    d = Document()
    convert.insert_text(d, "CC(C)Cc1ccc(cc1)[C@H](C)C(=O)O", 0, 0)
    for a in d.atoms.values():           # scramble the layout
        a.x += (a.id % 3) * 7.0
        a.y -= (a.id % 5) * 5.0
    before = convert.to_inchikey(d)
    assert convert.clean_structure(d)
    assert convert.to_inchikey(d) == before == S_IBUPROFEN
    lengths = [math.hypot(d.atoms[b.a].x - d.atoms[b.b].x, d.atoms[b.a].y - d.atoms[b.b].y)
               for b in d.bonds.values()]
    assert sorted(lengths)[len(lengths) // 2] == pytest.approx(d.style.bond_length, rel=0.02)


# --- reactions ---------------------------------------------------------------------------

def _esterification(water: bool = True) -> Document:
    d = Document()
    convert.insert_text(d, "CC(=O)O", 0, 0)
    convert.insert_text(d, "OCC", 90, 0)
    d.add_plus(45, 0)
    d.add_arrow(140, 0, 220, 0, "forward", above="H2SO4")
    convert.insert_text(d, "CC(=O)OCC", 300, 0)
    if water:
        d.add_plus(360, 0)
        convert.insert_text(d, "O", 400, 0)
    return d


def test_reaction_split_and_smiles():
    d = _esterification()
    sp = convert.split_reaction(d)
    assert sp.ok and len(sp.reactants) == 2 and len(sp.products) == 2 and not sp.agents
    assert convert.reaction_smiles(d) == "CC(=O)O.CCO>>CCOC(C)=O.O"
    rxn = rdChemReactions.ReactionFromRxnBlock(convert.reaction_rxn(d))
    assert rxn.GetNumReactantTemplates() == 2 and rxn.GetNumProductTemplates() == 2
    retro = Document()
    convert.insert_text(retro, "CC(=O)OCC", 0, 0)
    retro.add_arrow(60, 0, 120, 0, "retro")
    convert.insert_text(retro, "CC(=O)O.OCC", 200, 0)
    assert convert.reaction_smiles(retro) == "CC(=O)O.CCO>>CCOC(C)=O"


def test_reaction_balance_check():
    ok = analysis.balance_check(_esterification())
    assert ok.balanced and ("C", 4, 4) in ok.rows and ("O", 3, 3) in ok.rows
    bad = analysis.balance_check(_esterification(water=False))
    assert not bad.balanced and "not balanced" in bad.message
    h2 = Document()
    convert.insert_text(h2, "[H][H]", 0, 0)
    convert.insert_text(h2, "O=O", 60, 0)
    h2.add_arrow(100, 0, 160, 0)
    convert.insert_text(h2, "O", 200, 0)
    res = analysis.balance_check(h2)
    assert not res.balanced and res.suggestion == "2 H2 + O2 → 2 H2O"


def test_stoichiometry_grid():
    d = _esterification()
    lines, notes = analysis.stoich_grid(d, {0: (10.0, "mmol"), 1: (1.5, "equiv")})
    acid, etoh, ester, water = lines
    assert acid.limiting and not etoh.limiting
    assert etoh.mmol == pytest.approx(15.0) and etoh.equiv == pytest.approx(1.5)
    assert acid.mass_mg == pytest.approx(10 * 60.052, rel=1e-4)
    assert ester.mmol == pytest.approx(10.0) and ester.mass_mg == pytest.approx(881.06,
                                                                                  rel=1e-4)
    lines, _ = analysis.stoich_grid(d, {0: (600.52, "mg"), 1: (230.35, "mg")})
    assert lines[1].limiting and lines[2].mmol == pytest.approx(5.0, rel=1e-4)


def test_reaction_import_layout():
    d = Document()
    convert.insert_text(d, "CC(=O)O.OCC>>CC(=O)OCC.O", 0, 0)
    assert len(d.arrows) == 1 and len(d.pluses) == 2
    assert convert.reaction_smiles(d) == "CC(=O)O.CCO>>CCOC(C)=O.O"


def test_analysis_values_aspirin():
    d = Document()
    convert.insert_text(d, "CC(=O)Oc1ccccc1C(=O)O", 0, 0)
    res = analysis.analyze(d)
    assert res.formula == "C9H8O4"
    assert res.mw == pytest.approx(180.159, abs=1e-3)
    assert res.exact_mass == pytest.approx(180.042259, abs=1e-5)
    assert dict(res.elemental)["C"] == pytest.approx(60.00, abs=0.01)
    assert res.mz[0][0] == pytest.approx(180.0423, abs=1e-3)
    assert res.lipinski.startswith("Pass") and res.tpsa == pytest.approx(63.6, abs=0.01)


def test_selftest_runs():
    from rsstatchem.ui.chem_editor import selftest
    selftest()


# --- Qt: canvas, window, exports ---------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def win(qapp):
    from rsstatchem.ui.chem_editor import open_structure_editor
    w = open_structure_editor(None)
    w.resize(1100, 760)
    qapp.processEvents()
    w.canvas.resetTransform()
    w.canvas.centerOn(0, 0)
    yield w
    w.confirm_close = False
    w.close()


def _mouse(canvas, kind, pt, buttons=None):
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication
    vp = canvas.viewport()
    pos = canvas.mapFromScene(QPointF(*pt))
    typ = {"press": QMouseEvent.Type.MouseButtonPress, "move": QMouseEvent.Type.MouseMove,
           "release": QMouseEvent.Type.MouseButtonRelease,
           "double": QMouseEvent.Type.MouseButtonDblClick}[kind]
    btn = Qt.MouseButton.NoButton if kind == "move" else Qt.MouseButton.LeftButton
    if buttons is None:
        buttons = Qt.MouseButton.LeftButton if kind in ("press", "move", "double") else \
            Qt.MouseButton.NoButton
    ev = QMouseEvent(typ, QPointF(pos), QPointF(vp.mapToGlobal(pos)), btn, buttons,
                     Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(vp, ev)


def click(canvas, pt):
    _mouse(canvas, "press", pt)
    _mouse(canvas, "release", pt)


def drag(canvas, a, b, steps=6):
    _mouse(canvas, "press", a)
    for k in range(1, steps + 1):
        t = k / steps
        _mouse(canvas, "move", (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    _mouse(canvas, "release", b)


def test_canvas_mouse_tools(win):
    from PySide6.QtCore import Qt
    cv, doc = win.canvas, win.doc
    win.set_tool("bond:single")
    click(cv, (0, 0))                                     # click empty: new bond
    assert len(doc.atoms) == 2 and len(doc.bonds) == 1
    end = max(doc.atoms.values(), key=lambda a: a.x)
    click(cv, (end.x, end.y))                             # click atom: zig-zag bond
    assert len(doc.atoms) == 3 and len(doc.bonds) == 2
    a, b, c = sorted(doc.atoms.values(), key=lambda a: a.x)
    v1, v2 = (a.x - b.x, a.y - b.y), (c.x - b.x, c.y - b.y)
    ang = math.degrees(math.acos((v1[0] * v2[0] + v1[1] * v2[1]) /
                                 (math.hypot(*v1) * math.hypot(*v2))))
    assert ang == pytest.approx(120, abs=0.5)
    drag(cv, (c.x, c.y), (c.x + 10, c.y + 40))            # drag from atom: snapped bond
    assert len(doc.bonds) == 3
    new = max(doc.atoms.values(), key=lambda a: a.y)
    assert math.hypot(new.x - c.x, new.y - c.y) == pytest.approx(doc.style.bond_length)
    bond = doc.bond_between(a.id, b.id)
    mid = ((a.x + b.x) / 2, (a.y + b.y) / 2)
    click(cv, mid)                                        # click bond: cycle order
    assert bond.order == 2
    click(cv, mid)
    assert bond.order == 3
    click(cv, mid)
    assert bond.order == 1
    assert convert.to_smiles(doc) == "CCCC"

    win.set_tool("ring:benzene")
    click(cv, (200, 0))                                   # ring stamped on empty space
    assert convert.to_smiles(doc, doc.fragments()[-1]) == "c1ccccc1"
    n_atoms = len(doc.atoms)
    ring_bond = next(b for b in doc.bonds.values() if b.order == 1 and
                     doc.atoms[b.a].x > 150 and doc.ring_of_bond(b.id))
    p, q = doc.atoms[ring_bond.a], doc.atoms[ring_bond.b]
    click(cv, ((p.x + q.x) / 2, (p.y + q.y) / 2))         # fused on a bond
    assert len(doc.atoms) == n_atoms + 4
    assert convert.to_smiles(doc, doc.fragments()[-1]) == "c1ccc2ccccc2c1"

    win.set_tool("select")
    target = doc.atoms[a.id]
    x0, y0 = target.x, target.y
    drag(cv, (x0, y0), (x0 - 20, y0 + 15))                # move the atom
    assert (doc.atoms[a.id].x, doc.atoms[a.id].y) == pytest.approx((x0 - 20, y0 + 15),
                                                                     abs=1.0)
    drag(cv, (-60, -60), (60, 60))                        # marquee
    assert ("atom", b.id) in cv.selection
    before = {i: (at.x, at.y) for i, at in doc.atoms.items()}
    drag(cv, (doc.atoms[b.id].x, doc.atoms[b.id].y),
         (doc.atoms[b.id].x + 30, doc.atoms[b.id].y))     # move the selection
    moved = [i for i in before if abs(doc.atoms[i].x - before[i][0] - 30) < 1]
    assert b.id in moved and len(moved) >= 3

    win.set_tool("eraser")
    n_bonds = len(doc.bonds)
    bb = doc.bond_between(b.id, c.id)
    p, q = doc.atoms[bb.a], doc.atoms[bb.b]
    click(cv, ((p.x + q.x) / 2, (p.y + q.y) / 2))
    assert len(doc.bonds) == n_bonds - 1 and bb.id not in doc.bonds

    # undo / redo restore exact states
    state = doc.to_dict()
    win.undo_stack.undo()
    assert bb.id in doc.bonds
    win.undo_stack.redo()
    assert doc.to_dict() == state
    for _ in range(win.undo_stack.count()):
        win.undo_stack.undo()
    assert doc.is_empty()
    for _ in range(win.undo_stack.count()):
        win.undo_stack.redo()
    assert doc.to_dict() == state
    assert Qt.MouseButton.LeftButton                      # imported for the helpers


def test_canvas_chain_atom_label_charge_and_hotkeys(win):
    cv, doc = win.canvas, win.doc
    win.set_tool("chain")
    bl = doc.style.bond_length
    drag(cv, (0, 0), (4 * bl * math.cos(math.radians(30)), 0), steps=10)
    assert len(doc.bonds) == 4 and convert.to_smiles(doc) == "CCCCC"
    end = max(doc.atoms.values(), key=lambda a: a.x)
    win.set_tool("atom:O")
    click(cv, (end.x, end.y))
    assert doc.atoms[end.id].symbol == "O"
    win.set_tool("label")
    first = min(doc.atoms.values(), key=lambda a: a.x)
    click(cv, (first.x, first.y))                         # opens the label editor
    assert cv._editor is not None
    cv._editor.setText("Ph")
    cv._finish_editor()
    assert convert.to_smiles(doc) == canon("OCCCc1ccccc1")      # Ph replaced a carbon
    win.set_tool("charge-")
    click(cv, (end.x, end.y))
    assert doc.atoms[end.id].charge == -1
    cv.hover = ("atom", end.id)
    assert cv.hotkey(cv.hover, "n")
    assert doc.atoms[end.id].symbol == "N"
    win.set_tool("bond:wedge")
    b = doc.bond_between(end.id, doc.neighbors(end.id)[0])
    p, q = doc.atoms[b.a], doc.atoms[b.b]
    click(cv, ((p.x * 0.6 + q.x * 0.4), (p.y * 0.6 + q.y * 0.4)))
    assert b.style == "wedge"


def test_canvas_arrows_text_and_reaction(win):
    cv, doc = win.canvas, win.doc
    cv.insert_text("CC(=O)O", 0, 0)
    cv.insert_text("OCC", 120, 0)
    win.set_tool("plus")
    click(cv, (60, 0))
    win.set_tool("arrow:forward")
    drag(cv, (180, 2), (270, 0))
    ar = next(iter(doc.arrows.values()))
    assert ar.y1 == pytest.approx(ar.y2, abs=1e-6)       # snapped to horizontal
    cv.insert_text("CC(=O)OCC", 380, 0)
    cv.insert_text("O", 520, 0)
    assert convert.reaction_smiles(doc) == "CC(=O)O.CCO>>CCOC(C)=O.O"
    win.set_tool("text")
    click(cv, (0, 120))
    tid = next(iter(doc.texts))
    assert cv._editor is not None                         # in-place text editor is open
    cv._editor.setPlainText("H2SO4, 80 °C")
    cv._finish_editor()
    assert doc.texts[tid].text == "H2SO4, 80 °C"
    from rsstatchem.ui.chem_editor.render import chem_text_lines
    runs = chem_text_lines("H2SO4, 80 °C")[0]
    assert ("2", "sub") in runs and ("4", "sub") in runs and ("80", "n") in runs
    win.set_tool("arrow:curved")
    drag(cv, (0, 200), (60, 200))
    curved = [a for a in doc.arrows.values() if a.kind == "curved"]
    assert curved and abs(curved[0].bend) > 1


def test_style_presets_rescale(win):
    doc = win.doc
    win.canvas.insert_text("c1ccccc1", 0, 0)
    win.apply_style(preset("ACS Document 1996"))
    b = next(iter(doc.bonds.values()))
    p, q = doc.atoms[b.a], doc.atoms[b.b]
    assert math.hypot(p.x - q.x, p.y - q.y) == pytest.approx(14.4)
    assert doc.style.line_width == 0.6 and doc.style.bold_width == 2.0
    assert doc.style.bond_spacing == 18 and doc.style.font_family == "Arial"
    assert doc.style.font_size == 10
    win.undo_stack.undo()
    assert doc.style.bond_length == 30


def test_exports(win, tmp_path):
    from PySide6.QtGui import QImage

    from rsstatchem.ui.chem_editor import export
    win.canvas.insert_text("CC(=O)Oc1ccccc1C(=O)O", 0, 0)
    svg = tmp_path / "a.svg"
    assert win.export_as("svg", str(svg))
    root = ET.parse(svg).getroot()
    assert root.tag.endswith("svg")
    vb = [float(v) for v in root.get("viewBox").split()]
    assert len(vb) == 4 and vb[2] > 50 and vb[3] > 30
    assert len(list(root.iter())) > 10
    for dpi in (300, 600):
        png = tmp_path / f"a{dpi}.png"
        assert win.export_as("png", str(png), dpi=dpi, transparent=True)
        img = QImage(str(png))
        assert img.dotsPerMeterX() == pytest.approx(dpi / 0.0254, abs=1)
        _, rect = export.render_scene(win.doc)
        assert img.width() == pytest.approx(rect.width() * dpi / 72, abs=1)
        assert img.pixelColor(0, 0).alpha() == 0
    assert QImage(str(tmp_path / "a600.png")).width() == pytest.approx(
        2 * QImage(str(tmp_path / "a300.png")).width(), abs=2)
    pdf = tmp_path / "a.pdf"
    assert win.export_as("pdf", str(pdf))
    assert pdf.read_bytes()[:4] == b"%PDF"
    for fmt, check in (("smi", lambda t: t.strip() == canon("CC(=O)Oc1ccccc1C(=O)O")),
                       ("mol", lambda t: "M  END" in t), ("sdf", lambda t: "$$$$" in t),
                       ("inchi", lambda t: "BSYNRYMUTXBXSQ-UHFFFAOYSA-N" in t)):
        path = tmp_path / f"a.{fmt}"
        assert win.export_as(fmt, str(path))
        assert check(path.read_text())
    md = export.mime_data(win.doc)
    assert md.text() == canon("CC(=O)Oc1ccccc1C(=O)O") and md.hasImage()
    assert md.hasFormat(export.NATIVE_MIME)


def test_copy_paste_and_rsc_in_window(win, tmp_path):
    from PySide6.QtGui import QGuiApplication
    win.canvas.insert_text("c1ccccc1O", 0, 0)
    win.canvas.select_all()
    win.copy()
    win.paste()
    assert len(win.doc.fragments()) == 2
    QGuiApplication.clipboard().setText("CCN")
    win.paste()
    assert len(win.doc.fragments()) == 3
    path = tmp_path / "d.rsc"
    win.path = str(path)
    assert win.save() and win.undo_stack.isClean()
    from rsstatchem.ui.chem_editor import _WINDOWS, open_drawing_file
    other = open_drawing_file(None, str(path))
    try:
        assert other.doc.to_dict() == win.doc.to_dict()
    finally:
        other.confirm_close = False
        other.close()
    assert other not in _WINDOWS


def test_open_with_content_output_and_data_editor(qapp):
    import pandas as pd

    from rsstatchem.core.dataset import Dataset
    from rsstatchem.ui.chem_editor import open_reaction_editor, open_structure_editor

    shown = []

    class Ctx:
        settings = None

        def show_output(self, section):
            shown.append(section)

    class Doc:
        def __init__(self, ds):
            self.ds = ds

        def mutate(self, fn, text=""):
            return fn(self.ds)

    class Editor:
        def __init__(self):
            self.ds = Dataset(pd.DataFrame({"name": ["a", "b"]}))
            self.doc = Doc(self.ds)
            self.ctx = Ctx()

    ed = Editor()
    w = open_structure_editor(ed, smiles="CC(C)Cc1ccc(cc1)[C@H](C)C(=O)O")
    try:
        assert convert.to_inchikey(w.doc) == S_IBUPROFEN
        assert w.undo_stack.count() == 0
        assert w.send_to_output()
        sec = shown[-1]
        kinds = [it.kind for it in sec.items]
        assert "image" in kinds and "table" in kinds
        assert any("SMILES" in str(r) for it in sec.items if it.kind == "table"
                   for r in it.rows)
        assert w.write_smiles_to_data("SMILES", True, 1, convert.to_smiles(w.doc))
        assert ed.ds.df.loc[1, "SMILES"] == canon("CC(C)Cc1ccc(cc1)[C@H](C)C(=O)O")
    finally:
        w.confirm_close = False
        w.close()
    r = open_reaction_editor(ed, smiles="CC(=O)O.OCC>>CC(=O)OCC.O")
    try:
        assert r.canvas.tool == "arrow:forward"
        assert r.send_to_output()
        titles = [getattr(it, "title", "") for it in shown[-1].items]
        assert "Atom Balance" in titles
    finally:
        r.confirm_close = False
        r.close()


def test_dialogs_build(win):
    from rsstatchem.ui.chem_editor.dialogs import (AtomPropertiesDialog, BalanceDialog,
                                                   PeriodicTableDialog, StoichiometryDialog,
                                                   pt_position)
    assert pt_position(1) == (0, 0) and pt_position(2) == (0, 17)
    assert pt_position(6) == (1, 13) and pt_position(26) == (3, 7)
    assert pt_position(57) == (8, 2) and pt_position(72) == (5, 3)
    assert pt_position(71) == (8, 16) and pt_position(104) == (6, 3)
    PeriodicTableDialog(win)
    win.canvas.insert_text("CC(=O)O.OCC>>CC(=O)OCC.O", 0, 0)
    dlg = StoichiometryDialog(win.doc, win)
    dlg.set_amount(0, 10.0, "mmol")
    dlg.set_amount(1, 2.0, "equiv")
    assert dlg.lines[1].mmol == pytest.approx(20.0) and dlg.lines[0].limiting
    assert dlg.section().items[-1].rows[2][5] == "10.000"
    bal = BalanceDialog(win.doc, win)
    assert bal.res.balanced
    atom = next(iter(win.doc.atoms.values()))
    ap = AtomPropertiesDialog(atom, win)
    ap.charge.setValue(1)
    ap.isotope.setValue(13)
    ap.apply(win.doc, atom.id)
    assert atom.charge == 1 and atom.isotope == 13

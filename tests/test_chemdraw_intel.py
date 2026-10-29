"""WP-A chemistry intelligence: CIP labels, the rule-based IUPAC namer (OPSIN round trip), pKa
and Henry's law models, NMR prediction, JCAMP-DX and the NMR window / menu actions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

from rsstatchem.core.chem_draw import (convert, henry, iupac_namer, jcamp, names, pka,
                                       stereo)
from rsstatchem.core.chem_draw import nmr_predict as npd
from rsstatchem.core.chem_draw.model import Document
from rsstatchem.paths import resource

DATA = Path(__file__).parent / "data_chem_intel"


def doc_of(smiles: str) -> Document:
    d = Document()
    convert.insert_text(d, smiles, 100, 100)
    return d


# --- stereo ------------------------------------------------------------------------------------

def test_acceptance_stereo_labels():
    d = doc_of("C[C@H](N)C(=O)O")                    # L-alanine
    assert list(stereo.labels(d).values()) == ["S"]
    assert list(stereo.labels(doc_of("C/C=C/C")).values()) == ["E"]
    assert list(stereo.labels(doc_of("C/C=C\\C")).values()) == ["Z"]
    cis_decalin = stereo.labels(doc_of("[H][C@@]12CCCC[C@]1([H])CCCC2"))
    assert sorted(cis_decalin.values()) == ["s", "s"]
    trans_decalin = stereo.labels(doc_of("[H][C@@]12CCCC[C@@]1([H])CCCC2"))
    assert sorted(trans_decalin.values()) == ["r", "r"]
    assert stereo.labels(doc_of("CC(N)C(=O)O")) == {}     # undefined centre: no label
    assert stereo.format_label("R") == "(R)" and stereo.format_label("R", False) == "R"


CIP_SET = ["C[C@@H](O)CC", "OC(=O)[C@H](O)[C@@H](O)C(=O)O", "N[C@@H](Cc1ccccc1)C(=O)O",
           "C[C@H]1CC[C@@H](C)CC1", "C/C=C/C(=O)O", "C/C(Cl)=C(\\F)Br", "CC[C@H](C)[C@H](N)C(=O)O",
           "O[C@H]1CCCC[C@@H]1O", "C[C@@H](Cl)/C=C/C", "Br[C@@H](Cl)F"]


@pytest.mark.parametrize("smi", CIP_SET)
def test_stereo_labels_match_rdkit(smi):
    """Labels on the drawing equal RDKit's new CIP labeller on the SMILES molecule."""
    ref = Chem.MolFromSmiles(smi)
    rdCIPLabeler.AssignCIPLabels(ref)
    want = sorted([a.GetProp("_CIPCode") for a in ref.GetAtoms() if a.HasProp("_CIPCode")] +
                  [b.GetProp("_CIPCode") for b in ref.GetBonds() if b.HasProp("_CIPCode")])
    d = doc_of(smi)
    got = stereo.labels(d)
    assert sorted(got.values()) == want
    for (kind, i), _lab in got.items():
        assert (d.atoms if kind == "atom" else d.bonds)[i] is not None


def test_stereo_labels_subset_and_atoms():
    d = doc_of("C[C@H](N)C(=O)O")
    convert.insert_text(d, "C/C=C/C", 400, 100)
    lab = stereo.labels(d)
    assert sorted(lab.values()) == ["E", "S"]
    centre = next(i for (k, i) in lab if k == "atom")
    assert stereo.labels(d, atoms=[centre]) == {("atom", centre): "S"}


# --- IUPAC namer -----------------------------------------------------------------------------

EXPECTED = {
    "CC(C)CC(C)C": "2,4-dimethylpentane", "C=C(C)CC": "2-methylbut-1-ene",
    "C/C=C/C": "(2E)-but-2-ene", "C[C@H](O)CC": "(2S)-butan-2-ol",
    "OCCO": "ethane-1,2-diol", "CCOC(C)=O": "ethyl acetate", "CN(C)C=O": "N,N-dimethylformamide",
    "CCC#N": "propanenitrile", "CCC(=O)CC": "pentan-3-one", "CC(=O)Cl": "acetyl chloride",
    "CC(=O)OC(C)=O": "acetic anhydride", "CCN(CC)CC": "N,N-diethylethanamine",
    "Clc1ccc(Cl)cc1": "1,4-dichlorobenzene", "OC1CCCCC1": "cyclohexanol",
    "C[C@H](N)C(=O)O": "(2S)-2-aminopropanoic acid", "CCOC(=O)CC(=O)OCC": "diethyl propanedioate",
    "Oc1ccc(Cl)cc1": "4-chlorophenol", "OC(=O)c1ccccc1O": "2-hydroxybenzoic acid",
    "CC(=O)Oc1ccccc1C(=O)O": "2-acetyloxybenzoic acid",
    "O=CC1CCCCC1": "cyclohexanecarbaldehyde", "C=CC=C": "buta-1,3-diene",
    "CC(=O)c1ccccc1": "1-phenylethan-1-one", "NC(=O)CCC(N)=O": "butanediamide",
}


@pytest.mark.parametrize("smi,name", sorted(EXPECTED.items()))
def test_namer_candidate_names(smi, name):
    assert iupac_namer.candidate_name(Chem.MolFromSmiles(smi)) == name


@pytest.mark.parametrize("smi", ["c1ccncc1", "C1CC2CCC1C2", "CS(C)=O", "c1ccc2ccccc2c1",
                                 "CC(=O)[O-]", "OC1CCOC1", "C1CCC2(CC1)CC2"])
def test_namer_refuses_out_of_scope(smi):
    with pytest.raises(iupac_namer.NamerError):
        iupac_namer.candidate_name(Chem.MolFromSmiles(smi))


def test_namer_never_returns_unverified_names():
    mols = [Chem.MolFromSmiles(s) for s in ("CCO", "CCCC")]
    # a fake OPSIN that parses every name as methane: nothing may be returned
    assert iupac_namer.verified_names(mols, opsin=lambda ns, t: ["C"] * len(ns)) == [None, None]
    # a fake OPSIN that cannot parse anything
    assert iupac_namer.verified_names(mols, opsin=lambda ns, t: [None] * len(ns)) == [None, None]
    ok = iupac_namer.verified_names(mols, opsin=lambda ns, t: ["CCO", "CCCC"])
    assert ok == ["ethanol", "butane"]


def _teaching_set():
    rows = []
    for line in (DATA / "teaching_set.tsv").read_text("utf-8").splitlines():
        if line and not line.startswith("#"):
            cls, smi = line.split("\t")
            rows.append((cls, smi))
    return rows


@pytest.mark.skipif(not names.opsin_available(), reason="OPSIN (jar + Java) not available")
def test_namer_teaching_set_round_trips():
    """>= 90 % of the teaching set gets a name that OPSIN parses back to the same InChIKey;
    the rest get None (never a wrong name, which verified_names guarantees)."""
    rows = _teaching_set()
    assert len(rows) >= 200
    mols = [Chem.MolFromSmiles(s) for _c, s in rows]
    got = iupac_namer.verified_names(mols)
    rate = sum(g is not None for g in got) / len(got)
    assert rate >= 0.90, rate
    # every class is covered
    per = {}
    for (cls, _s), g in zip(rows, got):
        per.setdefault(cls, []).append(g is not None)
    assert all(sum(v) / len(v) >= 0.8 for v in per.values()), per


def test_names_tier_order(monkeypatch):
    calls = []

    def fake(ns, t):
        calls.append(list(ns))
        return ["CCCC(C)Cl" if "chloro" in n else None for n in ns]
    names._GENERATED.clear()
    mol = Chem.MolFromSmiles("CCCC(C)Cl")
    hit = names.structure_name(mol, opsin=fake)
    assert hit.name == "2-chloropentane" and hit.source == names.GENERATED_SOURCE
    # cached: OPSIN is not asked again
    assert names.structure_name(mol, opsin=fake).name == "2-chloropentane" and len(calls) == 1
    # the curated offline list comes first
    assert names.structure_name(Chem.MolFromSmiles("CC(=O)O"), opsin=fake).source == \
        "offline list"
    # when OPSIN cannot run, the database still answers and nothing generated is cached
    names._GENERATED.clear()

    def broken(ns, t):
        raise names.NameLookupError("no java", offline=True)
    nic = Chem.MolFromSmiles("CN1CCC[C@H]1c1cccnc1")
    assert names.structure_name(nic, opsin=broken).source == names.DB_SOURCE
    assert names.generated_name(Chem.MolFromSmiles("CCCCCl"), opsin=broken) is None
    # an OPSIN failure is not remembered (out-of-scope structures may be cached as None)
    from rsstatchem.core.chem import molecule as mo
    assert mo.to_inchikey(Chem.MolFromSmiles("CCCCCl")) not in names._GENERATED


# --- pKa and Henry's law ---------------------------------------------------------------------

def test_pka_model_file_and_benchmark():
    ms = pka.pka_models.load()
    assert set(ms) == {"pka_a", "pka_b"}
    for t in ("pka_a", "pka_b"):
        m = pka.pka_models.metrics(t)
        assert m["random_cv"]["n"] > 2000
        assert m["random_cv"]["rmse"] < 2.0 and m["holdout_75_25"]["rmse"] < 2.0
        assert m["scaffold"]["rmse"] < 2.2


@pytest.mark.parametrize("smi,kind,lit", [
    ("CC(=O)O", "acid", 4.76), ("OC(=O)c1ccccc1", "acid", 4.20), ("Oc1ccccc1", "acid", 9.99),
    ("c1ccncc1", "base", 5.23), ("Nc1ccccc1", "base", 4.60), ("CN", "base", 10.62),
    ("c1c[nH]cn1", "base", 6.95)])
def test_pka_literature_values(smi, kind, lit):
    """Measured values of the training data agree with handbook values."""
    v = getattr(pka.predict(Chem.MolFromSmiles(smi)), kind)
    assert v.experimental and v.value == pytest.approx(lit, abs=0.3)


def test_pka_model_predictions_and_gating():
    r = pka.predict(Chem.MolFromSmiles("ClCCCC(=O)O"))         # not in the data; lit. 4.52
    assert r.acid.source == "ML" and r.acid.value == pytest.approx(4.52, abs=0.6)
    assert r.acid.prediction.expected_rmse is not None
    assert r.base.value is None and "no basic group" in r.base.note
    r = pka.predict(Chem.MolFromSmiles("CCCCCC"))
    assert r.acid.value is None and r.base.value is None
    # phenol is filed with a "basic" pKa in the data: not shown (no basic group)
    assert pka.predict(Chem.MolFromSmiles("Oc1ccccc1")).base.value is None


def test_henry_values_and_rows():
    chcl3 = henry.experimental(Chem.MolFromSmiles("ClC(Cl)Cl"))
    assert chcl3.value == pytest.approx(0.00367 * henry.ATM_M3, rel=0.05)   # lit. 3.67e-3
    est = henry.vp_ws_estimate(None, vp_pa=12700.0, logs=-1.6, exp_inputs=True)
    assert est.value == pytest.approx(12700.0 / (10 ** -1.6 * 1000))
    m = henry.henry_models.metrics("loghl")
    assert m["opera_test"]["n"] > 100 and m["opera_test"]["rmse"] < 1.2
    vb = henry.vpws_benchmark()
    assert vb["all"]["n"] > 500
    from rsstatchem.core.chem_draw.properties import chemical_properties
    rows = {p.name: p for p in chemical_properties(Chem.MolFromSmiles("CCCCCC"))}
    hl = rows["Henry's Law"]
    assert hl.value is not None and hl.unit == "Pa·m³/mol" and hl.short in ("ML", "P/S")
    assert {a.key for a in hl.alternatives} == {"ml", "vpws"}
    acid = {p.name: p for p in chemical_properties(Chem.MolFromSmiles("CC(=O)O"))}
    assert acid["pKa (acidic)"].kind == "experimental"
    assert acid["pKa (basic)"].value is None


# --- NMR prediction ------------------------------------------------------------------------------

def test_nmr_benchmark_meets_targets():
    info = json.loads(resource("nmr/benchmark.json").read_text("utf-8"))
    b = info["benchmark"]
    assert info["n_test"] >= 200
    assert b["13C"]["mae"] <= 3.0 and b["13C"]["n"] > 2000
    assert b["1H"]["mae"] <= 0.3 and b["1H"]["n"] > 1500
    assert "CC0" in info["source"]


def _by_atoms(peaks):
    return {tuple(p.atoms): p for p in peaks}


def test_nmr_literature_ethyl_acetate():
    """CDCl3 literature values: 1H 4.12 (q), 2.05 (s), 1.26 (t); 13C 171.1, 60.5, 21.0, 14.2."""
    mol = Chem.MolFromSmiles("CCOC(C)=O")       # atoms: 0 CH3, 1 CH2, 2 O, 3 C=O, 4 CH3CO
    h = _by_atoms(npd.predict_detailed(mol, "1H"))
    assert h[(1,)].shift == pytest.approx(4.12, abs=0.25) and h[(1,)].multiplicity == "q"
    assert h[(4,)].shift == pytest.approx(2.05, abs=0.25) and h[(4,)].multiplicity == "s"
    assert h[(0,)].shift == pytest.approx(1.26, abs=0.25) and h[(0,)].multiplicity == "t"
    assert h[(1,)].count == 2 and h[(0,)].count == 3 and h[(0,)].J == (7.0,)
    c = _by_atoms(npd.predict_detailed(mol, "13C"))
    for atoms, lit in (((3,), 171.1), ((1,), 60.5), ((4,), 21.0), ((0,), 14.2)):
        assert c[atoms].shift == pytest.approx(lit, abs=3.0)
    # the WP-A data contract
    shifts = npd.predict(mol, "1H")
    assert all(len(s) == 4 for s in shifts)
    assert {s.atom_idx for s in shifts} == {0, 1, 4}


def test_nmr_exchangeable_protons_at_their_group_position():
    """OH/NH are not the D2O water signal (~4.7 ppm) of the archive spectra but the usual CDCl3
    position of their group: COOH ~11, phenol ~5.5, alcohol ~2, amide ~6.5, amine ~1.5."""
    for smiles, idx, lo, hi in (("CC(=O)Oc1ccccc1C(=O)O", 12, 10.0, 13.0),   # aspirin COOH
                                ("Oc1ccccc1", 0, 4.0, 8.0),                  # phenol
                                ("CCO", 2, 0.5, 5.0),                        # ethanol
                                ("CC(=O)Nc1ccccc1", 3, 5.0, 8.5),            # acetanilide NH
                                ("CCN", 2, 0.5, 3.0),                        # ethylamine
                                ("c1ccc2[nH]ccc2c1", 4, 7.5, 12.0)):         # indole NH
        mol = Chem.MolFromSmiles(smiles)
        peak = next(p for p in npd.predict_detailed(mol, "1H") if idx in p.atoms)
        assert peak.exchangeable and lo <= peak.shift <= hi, smiles
        assert "typical" in peak.note


def test_nmr_multiplets_and_equivalence():
    iso = _by_atoms(npd.predict_detailed(Chem.MolFromSmiles("CC(C)O"), "1H"))
    assert iso[(1,)].multiplicity == "sept" and iso[(0, 2)].multiplicity == "d"
    assert iso[(0, 2)].count == 6 and iso[(3,)].exchangeable
    tol = npd.predict_detailed(Chem.MolFromSmiles("Cc1ccccc1"), "13C")
    assert len(tol) == 5                                       # 7 carbons, 5 environments
    ch3 = [p for p in tol if p.atoms == [0]][0]
    assert ch3.shift == pytest.approx(21.4, abs=2.0)
    trans = _by_atoms(npd.predict_detailed(Chem.MolFromSmiles("C/C=C/C(=O)O"), "1H"))
    assert 16.0 in trans[(1,)].J or 16.0 in trans[(2,)].J
    lines = npd.multiplet_lines(1.0, [(7.0, 2)], 300.0, 3)
    assert [round(i, 3) for _x, i in lines] == [0.75, 1.5, 0.75]
    assert lines[0][0] == pytest.approx(1.0 - 7.0 / 300.0)
    assert npd.multiplicity_name([(7.0, 1), (2.5, 1)]) == "dd"
    x, y = npd.spectrum(npd.predict_detailed(Chem.MolFromSmiles("CCO"), "1H"), 400.0)
    assert x[0] > x[-1] and y.max() > 0


# --- JCAMP-DX ------------------------------------------------------------------------------------

def test_asdf_decoding():
    x, ys, dif = jcamp.decode_asdf_line("10 A00KT%Tl")
    assert x == 10 and ys == [100, 102, 104, 104, 104, 101] and dif
    _x, ys, _d = jcamp.decode_asdf_line("1 @eB")
    assert ys == [0, -5, 2]
    assert jcamp.encode_difdup([100, 102, 104, 104, 104, 101]) == "A00KT%Tl"


def test_jcamp_round_trip_and_peak_table(tmp_path):
    x = np.linspace(10, 0, 501)
    y = np.exp(-((x - 3.0) / 0.05) ** 2) + 0.5 * np.exp(-((x - 7.2) / 0.05) ** 2)
    path = tmp_path / "s.jdx"
    jcamp.write_jcamp(path, x, y, "test", "1H", 400.0)
    sp = jcamp.read_jcamp(path)
    assert sp.nucleus == "1H" and sp.frequency == pytest.approx(400.0)
    assert len(sp.x) == 501 and sp.x[0] == pytest.approx(10) and sp.x[-1] == pytest.approx(0)
    assert np.max(np.abs(sp.y - y)) < 1e-5
    text = ("##TITLE= peaks\n##JCAMP-DX= 4.24\n##DATA TYPE= NMR PEAK TABLE\n"
            "##.OBSERVE FREQUENCY= 100.0\n##.OBSERVE NUCLEUS= ^13C\n##XUNITS= HZ\n"
            "##YUNITS= ARBITRARY UNITS\n##PEAK TABLE= (XY..XY)\n17710, 100 6050, 80\n"
            "2100,60\n##END=\n")
    pk = jcamp.parse_jcamp(text)
    assert pk.kind == "peaks" and pk.nucleus == "13C"
    assert list(np.round(pk.x, 2)) == [177.1, 60.5, 21.0]


def test_jcamp_ntuples_and_errors():
    text = ("##TITLE= bruker-like\n##JCAMP-DX= 5.00\n##DATA TYPE= NMR SPECTRUM\n"
            "##.OBSERVE FREQUENCY= 300\n##.OBSERVE NUCLEUS= ^1H\n##NTUPLES= NMR SPECTRUM\n"
            "##VAR_NAME= FREQUENCY, SPECTRUM/REAL, SPECTRUM/IMAG\n##UNITS= HZ, ARB, ARB\n"
            "##FIRST= 3000, 0, 0\n##LAST= 0, 0, 0\n##FACTOR= 1, 1, 1\n##VAR_DIM= 4, 4, 4\n"
            "##PAGE= N=1\n##DATA TABLE= (X++(R..R)), XYDATA\n3000 1 2 3 4\n"
            "##PAGE= N=2\n##DATA TABLE= (X++(I..I)), XYDATA\n3000 0 0 0 0\n"
            "##END NTUPLES= NMR SPECTRUM\n##END=\n")
    sp = jcamp.parse_jcamp(text)
    assert list(sp.y) == [1, 2, 3, 4] and sp.x[0] == pytest.approx(10.0)
    with pytest.raises(jcamp.JcampError):
        jcamp.parse_jcamp("not a spectrum")


# --- window integration --------------------------------------------------------------------------

@pytest.fixture
def ctx(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("RSSTATCHEM_HOME", str(tmp_path))
    from rsstatchem.app import AppContext
    context = AppContext(qapp)
    monkeypatch.setattr(context.app, "quit", lambda: None)
    return context


@pytest.fixture
def win(ctx, qapp):
    from rsstatchem.ui.chemdraw import intel_actions
    from rsstatchem.ui.modules import open_module
    w = open_module(ctx, "chem")
    w.resize(1200, 820)
    w.online_lookup = False
    intel_actions.install(w)
    qapp.processEvents()
    yield w
    w.confirm_close = False
    w.close()


def test_install_adds_menu_items(win):
    acts = win.intel_actions
    assert {"nmr_1h", "nmr_13c", "stereo", "struct2name", "open_spectrum"} <= set(acts)
    texts = [a.text() for m in win.menuBar().actions() if m.menu()
             for a in m.menu().actions()]
    assert "Predict ¹H-NMR Shifts" in texts and "Open Spectrum (JCAMP-DX)..." in texts


def test_nmr_window_assignment_highlighting(win, qapp, tmp_path):
    from rsstatchem.ui.chemdraw import intel_actions
    win.load_text("CCOC(C)=O")
    nw = intel_actions.show_nmr(win, "1H")
    qapp.processEvents()
    assert [p.multiplicity for p in nw.peaks] == ["q", "s", "t"]
    assert nw.table.rowCount() == 3
    nw.select_peak(0)                                 # the CH2 quartet
    qapp.processEvents()
    sel = win.canvas.selected_atoms()
    assert len(sel) == 1 and win.doc.atoms[next(iter(sel))].symbol == "C"
    assert 0 in nw.view.marked
    # canvas -> peaks
    ch3 = [nw.origin[a] for p in nw.peaks if p.multiplicity == "t" for a in p.atoms]
    win.canvas.select_keys({("atom", ch3[0])})
    qapp.processEvents()
    assert nw.view.marked == {2}
    assert "q, J = 7.0 Hz, 2H" in nw.peak_list_text()
    # export and re-import as overlay
    out = tmp_path / "pred.jdx"
    assert nw.export_spectrum(str(out))
    assert nw.open_spectrum(str(out)) and nw.view.overlay is not None
    nw.nucleus_box.setCurrentText("13C")
    qapp.processEvents()
    assert len(nw.peaks) == 4 and all(p.multiplicity == "s" for p in nw.peaks)
    assert nw.grab().width() > 0


def test_show_stereochemistry_toggle(win):
    from rsstatchem.ui.chemdraw import intel_actions
    win.load_text("C[C@H](N)C(=O)O")
    labs = intel_actions.toggle_stereo(win, True)
    assert list(labs.values()) == ["S"]
    assert "(S)" in win.statusBar().currentMessage()


@pytest.mark.skipif(not names.opsin_available(), reason="OPSIN (jar + Java) not available")
def test_struct2name_uses_verified_namer(win):
    from rsstatchem.ui.chemdraw import intel_actions
    names._GENERATED.clear()
    win.load_text("CC(Cl)CC(C)(C)C")                  # not in the offline list / database
    found = intel_actions.convert_structure_to_name(win)
    assert found == ["4-chloro-2,2-dimethylpentane"]     # locants {2,2,4} < {2,4,4}
    assert any(t.text == found[0] for t in win.doc.texts.values())


def test_properties_window_shows_pka_rows(win, qapp):
    win.load_text("CC(=O)O")
    win.properties.refresh()
    qapp.processEvents()
    vals = win.properties.values()
    assert vals["pKa (acidic)"].startswith("4.70")
    assert "Pa·m³/mol" in vals["Henry's Law"]

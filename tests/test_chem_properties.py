"""Chemical Properties data and methods: the offline experimental database, the numpy-only ML
models (expected error, applicability domain), Marrero-Gani second-order and Abdulelah-Gani
group contributions, the benchmark-driven choice of the shown estimate, the property-database
parsers and the CDK logP helper (runs only when the CDK jar, the helper and Java are present)."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from rsstatchem.core.chem_draw import cdk, exp_props, ml_props
from rsstatchem.core.chem_draw.properties import (abdulelah_gani, benchmarks,
                                                  chemical_properties, marrero_gani, ranked)

ROOT = Path(__file__).resolve().parents[1]
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
NOVEL = "O=C(NC1CC2(C1)OCCO2)c1ccc(F)c(Cl)c1"
HAS_CDK = cdk.cdk_available()


def mol(smiles):
    return Chem.MolFromSmiles(smiles)


# --- experimental database --------------------------------------------------------------------

def test_experimental_database_lookup_and_sources():
    assert exp_props.db_size() > 30000
    caf = exp_props.lookup(mol(CAFFEINE))
    assert caf["mp"].value == pytest.approx(510.6, abs=1.0)   # 235-238 degC
    assert set("BOHI") <= set(caf["mp"].sources) and caf["mp"].n >= 4
    assert caf["logp"].value == pytest.approx(-0.07, abs=0.02)
    benzene = exp_props.lookup(mol("c1ccccc1"))
    assert benzene["bp"].value == pytest.approx(353.2, abs=0.5)
    assert benzene["tc"].value == pytest.approx(562.0, abs=1.0)
    assert benzene["logvp"].value == pytest.approx(np.log10(12700), abs=0.05)   # 95 mmHg
    # every source is listed with its licence
    assert exp_props.SOURCES["B"] == ("Bradley Open Melting Point Dataset", "CC0 1.0")
    assert all(lic for _label, lic in exp_props.SOURCES.values())
    assert "CC BY 4.0" in exp_props.source_text("I")
    assert exp_props.lookup(mol("CC(F)(Cl)C(Br)CCCC#N")) == {}


def test_experimental_lookup_falls_back_to_stereo_unspecified_record():
    # (S)-ibuprofen: only the racemate / unspecified record has a boiling point etc.
    exact = exp_props.lookup(inchikey="HEFNNWSXXWATRW-UHFFFAOYSA-N")
    s_ibu = exp_props.lookup(mol("CC(C)Cc1ccc(cc1)[C@H](C)C(=O)O"))
    assert exact and s_ibu
    for prop, v in s_ibu.items():
        if v.stereo_note:
            assert v.value == exact[prop].value
    # conflicting values keep their range
    asp = exp_props.lookup(mol(ASPIRIN))["mp"]
    assert asp.low is not None and asp.high >= asp.low


def test_rows_show_experimental_first_with_details():
    rows = {p.name: p for p in chemical_properties(mol(CAFFEINE))}
    mp = rows["Melting Point"]
    assert mp.kind == "experimental" and mp.tag().startswith("[experimental: Bradley")
    keys = [a.key for a in mp.alternatives]
    assert keys[0] == "exp" and {"ml", "mg", "mg1", "joback"} <= set(keys)
    assert rows["LogS"].kind == "experimental"
    assert rows["Vapor Pres."].value is None            # caffeine: no open VP record
    tol = {p.name: p for p in chemical_properties(mol("Cc1ccccc1"))}
    assert tol["Vapor Pres."].kind == "experimental" and "Pa at 25 °C" in tol["Vapor Pres."].text()


# --- machine-learning models (numpy only) -----------------------------------------------------

def test_gbtrees_inference_on_a_hand_made_tree():
    # one tree: x0 <= 1.5 ? (x1 <= 0 ? 1 : 2) : 3 ; NaN in x0 goes right
    t = ml_props.GBTrees(feature=np.array([0, 1, 0, 0, 0]),
                         threshold=np.array([1.5, 0.0, 0, 0, 0]),
                         left=np.array([1, 3, -1, -1, -1]), right=np.array([2, 4, -1, -1, -1]),
                         value=np.array([0, 0, 3.0, 1.0, 2.0]),
                         missing_left=np.array([False, True, False, False, False]),
                         offsets=np.array([0]), baseline=10.0)
    X = np.array([[1.0, -1.0], [1.0, 5.0], [2.0, 0.0], [np.nan, 0.0], [0.0, np.nan]])
    assert t.predict(X).tolist() == [11.0, 12.0, 13.0, 13.0, 11.0]


def test_ml_models_metrics_and_applicability_domain():
    ms = ml_props.models()
    assert set(ms) == {"mp", "bp", "logp"}
    m = ms["mp"].metrics
    assert m["random_cv"]["n"] > 15000 and 20 < m["random_cv"]["rmse"] < 60
    assert m["scaffold"]["rmse"] > 0 and m["ad_bins"][0][2] > m["ad_bins"][-1][2]
    caf = ml_props.predict(mol(CAFFEINE), "mp")
    assert caf.value == pytest.approx(510.6, abs=80) and caf.in_domain
    assert caf.expected_rmse < 60
    novel = ml_props.predict_all(mol(NOVEL))
    assert 300 < novel["mp"].value < 550 and 0 < novel["mp"].similarity < 1
    weird = ml_props.predict(mol("C" * 30 + "[Se]" + "C" * 5), "logp")
    assert weird.value is not None and weird.expected_rmse is not None
    assert ml_props.predict(mol("[Na+].[Cl-]"), "mp").value is None


def test_app_never_imports_scikit_learn():
    for path in (ROOT / "rsstatchem").rglob("*.py"):
        tree = ast.parse(path.read_text("utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            assert not any(n.split(".")[0] == "sklearn" for n in names), path


# --- group contributions -----------------------------------------------------------------------

@pytest.mark.parametrize("smiles,prop,first,second", [
    ("Nc1ccc(Nc2ccccc2)cc1", "tb", 614.62, 616.03),          # Marrero & Gani 2001, ex. 1
    ("CN1C(=O)C(O)N=C(c2ccccc2F)c2cc(Cl)ccc21", "tm", 469.19, 473.37),   # flutemazepam
])
def test_marrero_gani_second_order_reproduces_the_paper(smiles, prop, first, second):
    m = mol(smiles)
    assert marrero_gani(m, second_order=False).values[prop] == pytest.approx(first, abs=0.01)
    res = marrero_gani(m)
    assert res.values[prop] == pytest.approx(second, abs=0.01) and res.second


def test_abdulelah_gani_fills_critical_constants():
    caf = abdulelah_gani(mol(CAFFEINE))
    assert caf.ok and 700 < caf.values["tc"] < 1000 and caf.values["pc"] > 10
    tol = abdulelah_gani(mol("Cc1ccccc1")).values
    assert tol["tc"] == pytest.approx(591.8, abs=15) and tol["pc"] == pytest.approx(41.1, abs=3)
    assert not abdulelah_gani(mol("[Na+].[Cl-]")).ok


# --- benchmark-driven choice ---------------------------------------------------------------------

def test_benchmark_file_drives_the_default_estimate():
    b = benchmarks()
    for prop in ("mp", "bp", "tc", "pc", "logp", "logs"):
        assert b[prop]["methods"], prop
    assert b["mp"]["methods"]["ml"]["rmse"] < b["mp"]["methods"]["joback"]["rmse"]
    assert ranked("mp", ["joback", "mg", "ml"])[0] == "ml"
    assert set(b["logp"]["methods"]) >= {"crippen", "jplogp", "xlogp", "alogp", "consensus"}
    rows = {p.name: p for p in chemical_properties(mol(NOVEL))}
    mp = rows["Melting Point"]
    assert mp.short == "ML" and "expected" in mp.detail
    assert [a for a in mp.alternatives if a.default][0].key == "ml"
    assert all("RMSE" in a.benchmark for a in mp.alternatives if a.key != "exp")


# --- property database parsers ----------------------------------------------------------------

def test_property_db_parsers():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import build_property_db as b
    finally:
        sys.path.pop(0)
    assert b._temperature("235-238 °C", False) == pytest.approx(509.65)
    assert b._temperature("MP: 176 °F", False) == pytest.approx(353.15)
    assert b._temperature("173-175 °C (slight decomposition)", False) == pytest.approx(447.15)
    assert b._temperature("450 °C decomposes", True) is None          # not a boiling point
    assert b._temperature("BP: 139-140 °C at 25 mm Hg", True) is None   # reduced pressure
    assert b._temperature("232-234 °C @ 760 MM HG", True) == pytest.approx(506.15)
    assert b._temperature("MP: 227-279 °C /Vilazodone hydrochloride/", False) is None
    assert b._temperature("Sublimes at 93 °C", True) is None
    assert b._logp("log Kow = 1.19") == pytest.approx(1.19)
    assert b._logp("log Kow = 0.87 /Estimated/") is None
    assert b._logp("log Kow = 0.73 (pH 5); -0.77 (pH 7)") is None
    assert b._critical("Critical temperature: 278.4 °C; critical pressure: 34.9 atm") == [
        ("tc", pytest.approx(551.55)), ("pc", pytest.approx(35.3624, abs=1e-3))]
    assert b._vapour_pressure("2.20X10-6 mm Hg at 25 °C") == pytest.approx(
        np.log10(2.2e-6 * 133.322368))
    assert b._vapour_pressure("0.97 MM HG AT 20 °C") is None


# --- CDK logP helper (Java) ----------------------------------------------------------------------

@pytest.mark.skipif(not HAS_CDK, reason="CDK jar / helper or Java not available")
def test_cdk_helper_values_and_restart():
    helper = cdk.CdkHelper()
    try:
        asp = helper.compute(ASPIRIN, timeout=120)
        assert asp["jplogp"] == pytest.approx(1.24, abs=0.05)
        assert asp["xlogp"] == pytest.approx(1.98, abs=0.05)
        assert asp["alogp"] == pytest.approx(1.20, abs=0.05)
        assert asp["amr"] == pytest.approx(43.9, abs=0.5)
        bad = helper.compute("C1CC(", timeout=60)
        assert bad["jplogp"] is None and "SMILES" in bad["error"]
        assert helper.starts == 1
        helper._proc.kill()                       # the helper dies: restarted on next call
        helper._proc.wait()
        again = helper.compute("CCO", timeout=120)
        assert again["xlogp"] is not None and helper.starts == 2
    finally:
        helper.close()
    assert not helper.alive()


@pytest.mark.skipif(not HAS_CDK, reason="CDK jar / helper or Java not available")
def test_consensus_logp_rows_with_cdk_values():
    m = mol(NOVEL)
    values = cdk.cdk_values(Chem.MolToSmiles(m))
    rows = {p.name: p for p in chemical_properties(m, values)}
    alts = {a.key: a for a in rows["LogP"].alternatives}
    comps = [alts[k].value for k in ("crippen", "jplogp", "xlogp", "alogp", "ml")]
    assert all(c is not None for c in comps)
    assert alts["consensus"].value == pytest.approx(np.mean(comps))
    assert alts["consensus4"].value == pytest.approx(np.mean(comps[:4]))
    assert "not BioByte" in rows["LogP"].method and "consensus" in rows["LogP"].detail
    mr = {a.key: a for a in rows["MR"].alternatives}
    assert mr["amr"].value == pytest.approx(values["amr"])


def test_properties_window_without_cdk_never_blocks(qapp):
    from rsstatchem.ui.chemdraw.window import ChemDrawWindow  # noqa: F401 - import check
    rows = chemical_properties(mol(NOVEL), None, "computing...")
    logp = {p.name: p for p in rows}["LogP"]
    assert logp.value is not None and "computing" in logp.detail


@pytest.mark.skipif(not HAS_CDK, reason="CDK jar / helper or Java not available")
def test_properties_window_fills_cdk_values_in_background(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("RSSTATCHEM_HOME", str(tmp_path))
    from rsstatchem.app import AppContext
    from rsstatchem.ui.modules import open_module

    ctx = AppContext(qapp)
    monkeypatch.setattr(ctx.app, "quit", lambda: None)
    win = open_module(ctx, "chem")
    win.load_text(NOVEL)
    props = win.properties
    props.refresh()
    assert props.wait_cdk(120)
    alts = {a.key: a for a in {p.name: p for p in props.rows}["LogP"].alternatives}
    assert alts["jplogp"].value is not None and alts["consensus"].value is not None
    props.details_btn.setChecked(True)
    html = props.details.toHtml()
    assert "JPlogP" in html and "Consensus" in html and "RMSE" in html
    assert "XLogP" in props.labels["LogP"].toolTip()

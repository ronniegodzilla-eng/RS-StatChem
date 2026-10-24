"""Chemistry module (stage 1): formulas, balancing, stoichiometry, calculators, RDKit core.

Reference values: IUPAC standard atomic weights (textbook molar masses), textbook balanced
equations, direct RDKit calls, published InChIKeys and textbook pH examples.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from rsstatchem.core.chem import elements as el
from rsstatchem.core.chem import formula as fm
from rsstatchem.core.chem import reaction as rx
from rsstatchem.core.chem import sections
from rsstatchem.core.chem import solutions as sol
from rsstatchem.core.dataset import Dataset

# --- formulas ------------------------------------------------------------------------------


@pytest.mark.parametrize("text, expected, tol", [
    ("H2O", 18.015, 0.0005),
    ("C6H12O6", 180.156, 0.0005),
    ("CuSO4·5H2O", 249.68, 0.005),
    ("CuSO4.5H2O", 249.68, 0.005),
    ("CuSO4*5H2O", 249.68, 0.005),
    ("K4[Fe(CN)6]", 368.35, 0.005),
    ("Ca3(PO4)2", 310.18, 0.01),
    ("(NH4)2SO4", 132.14, 0.01),
    ("Al2(SO4)3·18H2O", 666.43, 0.05),
    ("CaSO4·0.5H2O", 145.15, 0.01),
    ("NaCl", 58.44, 0.005),
])
def test_molar_mass(text, expected, tol):
    assert fm.molar_mass(text) == pytest.approx(expected, abs=tol)


def test_hydrate_forms_agree_and_hill():
    a, b, c = (fm.parse_formula(t) for t in ("CuSO4·5H2O", "CuSO4.5H2O", "CuSO4*5H2O"))
    assert a.same_as(b) and b.same_as(c)
    assert a.hill() == "CuH10O9S"
    assert fm.parse_formula("C2H5OH").hill() == "C2H6O"
    assert fm.parse_formula("K4[Fe(CN)6]").hill() == "C6FeK4N6"
    assert fm.parse_formula("NaCl").hill() == "ClNa"


@pytest.mark.parametrize("text, hill, charge", [
    ("SO4^2-", "O4S", -2), ("SO4^{2-}", "O4S", -2), ("SO42-", "O4S", -2),
    ("SO₄²⁻", "O4S", -2), ("[Fe(CN)6]3-", "C6FeN6", -3), ("[Fe(CN)6]^4-", "C6FeN6", -4),
    ("NH4+", "H4N", 1), ("Fe3+", "Fe", 3), ("Fe+++", "Fe", 3), ("NO3-", "NO3", -1),
    ("Cr2O72-", "Cr2O7", -2), ("Hg22+", "Hg2", 2), ("Cl-", "Cl", -1), ("Na+(aq)", "Na", 1),
    ("O^2-", "O", -2), ("PO43-", "O4P", -3),
])
def test_charges(text, hill, charge):
    f = fm.parse_formula(text)
    assert f.hill(with_charge=False) == hill
    assert f.charge == charge


def test_isotopes_and_monoisotopic():
    assert fm.parse_formula("H2O").monoisotopic_mass == pytest.approx(18.010565, abs=1e-5)
    f = fm.parse_formula("[13C]H4")
    assert f.monoisotopic_mass == pytest.approx(13.0033548 + 4 * 1.00782503, abs=1e-5)
    assert fm.parse_formula("¹³CH4").same_as(f)
    d = fm.parse_formula("CD3OD")
    assert d.elements["H"] == 4 and ("H", 2) in d.counts
    assert d.molar_mass == pytest.approx(12.011 + 4 * 2.014101778 + 15.999, abs=1e-4)
    # ions: monoisotopic mass corrected for the electron mass
    so4 = fm.parse_formula("SO4^2-")
    assert so4.monoisotopic_mass == pytest.approx(31.97207 + 4 * 15.99491 + 2 * el.ELECTRON_MASS,
                                                  abs=1e-4)
    assert so4.mz == pytest.approx(so4.monoisotopic_mass / 2)


@pytest.mark.parametrize("bad", ["Xx2", "H2O)", "(H2O", "", "h2o", "C1.5.", "Cb"])
def test_invalid_formulas(bad):
    with pytest.raises(fm.FormulaError):
        fm.parse_formula(bad)


def test_composition():
    comp = {r.symbol: r.percent for r in fm.parse_formula("H2O").composition()}
    assert comp["H"] == pytest.approx(11.19, abs=0.01)
    assert comp["O"] == pytest.approx(88.81, abs=0.01)
    pct = fm.parse_formula("C6H12O6").mass_percent()
    assert pct["C"] == pytest.approx(40.00, abs=0.01)
    assert sum(pct.values()) == pytest.approx(100.0)


def test_empirical_and_dbe():
    assert fm.parse_formula("C6H12O6").empirical().hill() == "CH2O"
    assert fm.parse_formula("C8H10N4O2").empirical().hill() == "C4H5N2O"
    assert fm.parse_formula("C6H6").dbe() == 4
    assert fm.parse_formula("C9H8O4").dbe() == 6
    assert fm.parse_formula("C2H3Cl").dbe() == 1


@pytest.mark.parametrize("formula, molar", [("C6H12O6", 180.156), ("C8H10N4O2", 194.19),
                                            ("Fe2O3", None), ("C9H8O4", 180.16),
                                            ("C2H4O2", 60.05)])
def test_composition_round_trip(formula, molar):
    f = fm.parse_formula(formula)
    pct = {k: round(v, 2) for k, v in f.mass_percent().items()}
    res = fm.empirical_from_composition(pct, molar)
    assert res.empirical.hill() == f.empirical().hill()
    if molar:
        assert res.molecular.hill() == f.hill()


def test_empirical_by_difference_and_combustion():
    res = fm.empirical_from_composition({"C": 40.0, "H": 6.71, "O": None})
    assert res.by_difference == "O" and res.empirical.hill() == "CH2O"
    # 10.0 mg glucose burns to 14.66 mg CO2 and 6.00 mg H2O
    pct = fm.combustion_composition(10.0, 14.66, 6.00)
    assert pct["C"] == pytest.approx(40.0, abs=0.1) and pct["O"] is None
    assert fm.empirical_from_composition(pct, 180.16).molecular.hill() == "C6H12O6"
    with pytest.raises(ValueError):
        fm.empirical_from_composition({"C": None, "O": None})


def test_formatting():
    assert fm.format_formula("H2SO4") == "H₂SO₄"
    assert fm.format_formula("SO4^2-") == "SO₄²⁻"
    assert fm.format_formula("CuSO4.5H2O") == "CuSO₄·5H₂O"
    assert fm.format_formula("K4[Fe(CN)6]") == "K₄[Fe(CN)₆]"
    assert fm.format_formula("[13C]O2") == "¹³CO₂"
    assert fm.format_formula("NH4+", "html") == "NH<sub>4</sub><sup>+</sup>"
    assert fm.format_formula("H2O", "mathtext") == "H$_{2}$O"
    assert fm.format_formula("SO4^2-", "mathtext") == "SO$_{4}^{2-}$"
    assert fm.format_formula("Fe3+", "html") == "Fe<sup>3+</sup>"
    assert fm.format_formula("not a formula!") == "not a formula!"
    # mathtext output must be valid for matplotlib
    from matplotlib.mathtext import MathTextParser
    MathTextParser("path").parse(fm.format_formula("SO4^2-", "mathtext"))


def test_isotope_pattern_chlorine():
    pattern = sections.isotope_pattern(fm.parse_formula("Cl2"))
    rel = {round(m): a for m, a in pattern}
    assert rel[70] == pytest.approx(100.0)
    assert rel[72] == pytest.approx(2 * 24.24 / 75.76 * 100, rel=0.01)


# --- balancing -----------------------------------------------------------------------------

@pytest.mark.parametrize("eq, coefs", [
    ("C3H8 + O2 -> CO2 + H2O", [1, 5, 3, 4]),
    ("KMnO4 + HCl -> KCl + MnCl2 + H2O + Cl2", [2, 16, 2, 2, 8, 5]),
    ("Cu + HNO3 -> Cu(NO3)2 + NO + H2O", [3, 8, 3, 2, 4]),
    ("Fe + O2 -> Fe2O3", [4, 3, 2]),
    ("C6H12O6 + O2 → CO2 + H2O", [1, 6, 6, 6]),
    ("MnO4- + Fe2+ + H+ -> Mn2+ + Fe3+ + H2O", [1, 5, 8, 1, 5, 4]),
    ("Cr2O7^2- + Fe^2+ + H^+ = Cr^3+ + Fe^3+ + H2O", [1, 6, 14, 2, 6, 7]),
    ("MnO4^- + H^+ + e^- -> Mn^2+ + H2O", [1, 8, 5, 1, 4]),
    ("Fe3++e- -> Fe2+", [1, 1, 1]),
    ("Ag+ + Cl- -> AgCl", [1, 1, 1]),
    ("NH4+ + OH- <=> NH3 + H2O", [1, 1, 1, 1]),
    ("CuSO4·5H2O -> CuSO4 + H2O", [1, 1, 5]),
    ("K4[Fe(CN)6] + H2SO4 + H2O -> K2SO4 + FeSO4 + (NH4)2SO4 + CO", [1, 6, 6, 2, 1, 3, 6]),
])
def test_balance(eq, coefs):
    res = rx.balance(eq)
    assert res.status == "balanced"
    assert res.coefficients == coefs
    assert all(a == b for _, a, b in res.balance_table())


def test_balance_reversible_and_given():
    res = rx.balance("2 H2 + O2 -> 2 H2O")
    assert res.given_balanced is True
    assert rx.balance("H2 + O2 -> H2O").given_balanced is None
    assert rx.balance("H2 + O2 -> 2 H2O").given_balanced is False
    assert rx.balance("1 H2 + 1 O2 -> 1 H2O").given_balanced is False
    assert rx.balance("NH4+ + OH- <=> NH3 + H2O").arrow == "⇌"


def test_balance_impossible_and_not_unique():
    assert rx.balance("H2O -> H2O2").status == "impossible"
    res = rx.balance("NaCl -> Na + Cl2 + O2")
    assert res.status == "impossible" and "O" in res.message
    res = rx.balance("CO2 -> CO + C")
    assert res.status == "no_positive" and "C" in res.message
    res = rx.balance("H2 + O2 -> H2O + H2O2")
    assert res.status == "not_unique" and len(res.basis) == 2
    assert all(a == b for _, a, b in res.balance_table())
    with pytest.raises(fm.FormulaError):
        rx.balance("H2 + O2")


def test_stoichiometry_limiting_reagent():
    res = rx.balance("2 H2 + O2 -> 2 H2O")
    st = rx.stoichiometry(res, {0: rx.Amount(4.0, "g"), 1: rx.Amount(16.0, "g")}, {2: 15.0})
    n_h2 = 4.0 / (2 * 1.008)
    assert st.limiting == [1]
    assert st.extent == pytest.approx(16.0 / 31.998)
    water = st.rows[2]
    assert water.reacting_mol == pytest.approx(2 * 16.0 / 31.998)
    assert water.reacting_g == pytest.approx(2 * 16.0 / 31.998 * 18.015)
    assert st.rows[0].remaining_mol == pytest.approx(n_h2 - 2 * 16.0 / 31.998)
    assert st.rows[1].remaining_mol == 0
    assert water.percent_yield == pytest.approx(100 * 15.0 / water.reacting_g)


def test_stoichiometry_units_and_reverse():
    res = rx.balance("HCl + NaOH -> NaCl + H2O")
    st = rx.stoichiometry(res, {0: rx.Amount(25, "mL (solution)", molarity=0.1),
                                1: rx.Amount(2.0, "mmol")})
    assert st.limiting == [1] and st.extent == pytest.approx(0.002)
    assert rx.Amount(22.414, "L (gas)").moles(1.0) == pytest.approx(1.0, abs=1e-4)
    # product only: reactants needed for 1 mol water
    st = rx.stoichiometry(rx.balance("2 H2 + O2 -> 2 H2O"), {2: rx.Amount(1, "mol")})
    assert st.basis == "product" and st.rows[1].reacting_mol == pytest.approx(0.5)
    with pytest.raises(ValueError):
        rx.stoichiometry(res, {0: rx.Amount(1, "L (solution)")})


# --- solutions -----------------------------------------------------------------------------

def test_ph():
    assert sol.ph_weak_acid(0.1, 1.8e-5).pH == pytest.approx(2.87, abs=0.01)
    assert sol.ph_weak_acid(0.1, 1.8e-5).ionization_pct == pytest.approx(1.33, abs=0.01)
    assert sol.ph_weak_acid(0.1, 1.8e-5, water=True).pH == pytest.approx(2.875, abs=0.001)
    assert sol.ph_weak_base(0.1, 1.8e-5).pH == pytest.approx(11.13, abs=0.01)
    assert sol.ph_strong_acid(0.01).pH == pytest.approx(2.0, abs=1e-6)
    assert sol.ph_strong_acid(1e-8).pH == pytest.approx(6.978, abs=0.001)
    assert sol.ph_strong_base(0.05, 2).pH == pytest.approx(13.0, abs=1e-6)
    r = sol.ph_weak_acid(0.1, 1.8e-5)
    assert r.pH + r.pOH == pytest.approx(14.0)


def test_buffer():
    assert sol.buffer_ph(4.76, 0.1, 0.1).pH == pytest.approx(4.76)
    assert sol.buffer_ph(4.76, 0.1, 0.2).pH == pytest.approx(4.76 + math.log10(2))
    # 0.01 mol HCl added to 0.1/0.1 mol acetate buffer
    assert sol.buffer_ph(4.76, 0.1, 0.1, added_strong_acid=0.01).pH == \
        pytest.approx(4.76 + math.log10(0.09 / 0.11))
    rec = sol.buffer_recipe(4.76, 5.0, 0.1, 1.0, 60.052, 82.034)
    assert rec["base_mol"] / rec["acid_mol"] == pytest.approx(10 ** 0.24)
    assert rec["acid_mol"] + rec["base_mol"] == pytest.approx(0.1)


def test_solution_calculators():
    assert sol.mass_for_solution(0.1, 0.25, 58.44) == pytest.approx(1.461)
    assert sol.mass_for_solution(0.1, 0.25, 58.44, purity_pct=99.5) == \
        pytest.approx(1.461 / 0.995)
    assert sol.molarity(0.5, mass_g=5.844, molar_mass=58.44) == pytest.approx(0.2)
    assert sol.stock_molarity(37, 1.19, 36.46) == pytest.approx(12.08, abs=0.01)
    assert sol.molality(0.1, 0.5) == pytest.approx(0.2)
    assert sol.normality(0.1, 2) == pytest.approx(0.2)
    d = sol.dilution(c1=1.0, c2=0.1, v2=100)
    assert d["v1"] == pytest.approx(10) and d["solvent"] == pytest.approx(90)
    with pytest.raises(ValueError):
        sol.dilution(c1=0.1, v1=10, c2=1.0)
    steps = sol.serial_dilution(1000, 10, 4, 10)
    assert [s.concentration for s in steps] == pytest.approx([100, 10, 1, 0.1])
    assert steps[0].transfer_volume == pytest.approx(1)


def test_concentration_and_unit_conversion():
    cc = sol.convert_concentration
    assert cc(1, "M (mol/L)", "mg/L", molar_mass=58.44) == pytest.approx(58440)
    assert cc(1, "% (w/v)", "g/L") == pytest.approx(10)
    assert cc(5, "mg/L", "ppm (w/w)") == pytest.approx(5)
    assert cc(37, "% (w/w)", "M (mol/L)", molar_mass=36.46, density=1.19) == \
        pytest.approx(sol.stock_molarity(37, 1.19, 36.46))
    assert cc(cc(0.5, "mol/kg", "M (mol/L)", 58.44, 1.02), "M (mol/L)", "mol/kg", 58.44,
              1.02) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        cc(1, "M (mol/L)", "g/L")
    assert sol.convert(25, "°C", "K") == pytest.approx(298.15)
    assert sol.convert(1, "atm", "kPa") == pytest.approx(101.325)
    assert sol.convert(1, "kcal", "kJ") == pytest.approx(4.184)
    assert sol.ideal_gas(P=1, n=1, T=273.15)["V"] == pytest.approx(22.414, abs=0.001)


# --- RDKit ---------------------------------------------------------------------------------

rdkit = pytest.importorskip("rdkit")


def test_identifiers_and_formats():
    from rsstatchem.core.chem import molecule as mo
    caffeine = mo.mol_from_text("Cn1cnc2c1c(=O)n(C)c(=O)n2C")
    assert mo.to_inchikey(caffeine) == "RYYVLZVUVIJVGH-UHFFFAOYSA-N"
    assert mo.molecular_formula(caffeine) == "C8H10N4O2"
    same = mo.mol_from_text(mo.to_inchi(caffeine))
    assert mo.to_smiles(same) == mo.to_smiles(caffeine)
    block = mo.to_molblock(caffeine)
    assert mo.detect_format(block) == "molblock"
    assert mo.to_smiles(mo.mol_from_text(block)) == mo.to_smiles(caffeine)
    named = mo.mol_from_text("CCO ethanol")
    assert named.GetProp("_Name") == "ethanol"
    with pytest.raises(mo.ChemError):
        mo.mol_from_text("C1CC")
    with pytest.raises(mo.ChemError, match="valence"):
        mo.mol_from_text("C(C)(C)(C)(C)C")


def test_descriptors_equal_rdkit():
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors
    from rsstatchem.core.chem import molecule as mo
    for smi in ("CC(=O)Oc1ccccc1C(=O)O", "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
                "CC(C)Cc1ccc(cc1)[C@H](C)C(=O)O", "C[N+](C)(C)CC(=O)[O-]"):
        m = Chem.MolFromSmiles(smi)
        p = mo.properties(mo.mol_from_text(smi))
        assert p["MW"] == pytest.approx(Descriptors.MolWt(m))
        assert p["ExactMass"] == pytest.approx(Descriptors.ExactMolWt(m))
        assert p["cLogP"] == pytest.approx(Crippen.MolLogP(m))
        assert p["MR"] == pytest.approx(Crippen.MolMR(m))
        assert p["TPSA"] == pytest.approx(rdMolDescriptors.CalcTPSA(m))
        assert p["HBD"] == Descriptors.NumHDonors(m)
        assert p["HBA"] == Descriptors.NumHAcceptors(m)
        assert p["RotB"] == Descriptors.NumRotatableBonds(m)
        assert p["Rings"] == rdMolDescriptors.CalcNumRings(m)
        assert p["AroRings"] == rdMolDescriptors.CalcNumAromaticRings(m)
        assert p["HeavyAtoms"] == m.GetNumHeavyAtoms()
        assert p["FCsp3"] == pytest.approx(rdMolDescriptors.CalcFractionCSP3(m))
        assert p["Charge"] == Chem.GetFormalCharge(m)
        assert p["QED"] == pytest.approx(QED.qed(m))
        # RDKit MW and the IUPAC formula tool agree
        assert fm.molar_mass(p["Formula"]) == pytest.approx(p["MW"], abs=0.01)


def test_drug_likeness():
    from rsstatchem.core.chem import molecule as mo
    aspirin = mo.drug_likeness(mo.mol_from_text("CC(=O)Oc1ccccc1C(=O)O"))
    assert all(r.passed for r in aspirin)
    assert aspirin[0].rule == "Lipinski" and aspirin[0].violations == 0
    wax = mo.drug_likeness(mo.mol_from_text("C" * 40), ["Lipinski", "Veber"])
    assert not wax[0].passed and wax[0].violations == 2
    assert not wax[1].passed


def test_depiction():
    from rsstatchem.core.chem import depict, molecule as mo
    m = mo.mol_from_text("C[C@H](N)C(=O)O")
    png = depict.depict_png(m, 300, 200, transparent=True, highlight_smarts="C(=O)O",
                            atom_indices=True, stereo_labels=True, legend="Ala")
    assert png[:4] == b"\x89PNG"
    from PIL import Image  # matplotlib dependency
    import io
    img = Image.open(io.BytesIO(png))
    assert img.size == (600, 400) and img.mode == "RGBA"
    assert img.getpixel((0, 0))[3] == 0  # transparent corner
    svg = depict.depict_svg(m, 300, 200, black_white=True)
    assert svg.lstrip().startswith("<?xml") and "<svg" in svg
    rxn_png = depict.depict_reaction_png("CC(=O)O.OCC>>CC(=O)OCC.O", 500, 150)
    assert rxn_png[:4] == b"\x89PNG"
    assert "<svg" in depict.depict_reaction_svg("CC(=O)O.OCC>>CC(=O)OCC.O")
    grid = depict.depict_grid_png([m, mo.mol_from_text("CCO")], ["a", "b"], cols=2)
    assert Image.open(io.BytesIO(grid)).size == (2 * 320 * 2, 240 * 2)
    with pytest.raises(TypeError):
        depict.depict_png(m, nonsense=True)


def test_editor_bridge_round_trip():
    from rsstatchem.core.chem import molecule as mo
    for smi in ("C[C@H](N)C(=O)O", "C/C=C\\Cl", "F/C=C/C=C\\C[C@](Cl)(Br)I",
                "O=C1CC[C@@H](C)[C@H]1O", "c1ccccc1O", "[NH4+].[Cl-]", "[13CH4]"):
        m = mo.mol_from_text(smi)
        atoms, bonds = mo.graph_from_mol(m)
        assert all(isinstance(a, mo.AtomSpec) for a in atoms)
        back = mo.mol_from_graph(atoms, bonds)
        assert mo.to_smiles(back) == mo.to_smiles(m), smi
    atoms, bonds = mo.graph_from_mol(mo.mol_from_text("CC=CC"))
    assert bonds[1].stereo == "either"
    bad = mo.mol_from_graph([mo.AtomSpec("C"), mo.AtomSpec("O", 1.0)],
                            [mo.BondSpec(0, 1, 3)], sanitize=False)
    assert mo.problems(bad) and mo.problems(bad)[0][0] == 1
    with pytest.raises(mo.ChemError):
        mo.mol_from_graph([mo.AtomSpec("C"), mo.AtomSpec("O", 1.0)], [mo.BondSpec(0, 1, 3)])


def test_reaction_and_files(tmp_path):
    from rsstatchem.core.chem import molecule as mo
    rxn = mo.reaction_from_text("CC(=O)O.OCC>>CC(=O)OCC.O")
    assert rxn.GetNumReactantTemplates() == 2 and rxn.GetNumProductTemplates() == 2
    assert mo.detect_format("CC(=O)O.OCC>>CC(=O)OCC.O") == "reaction"
    mols = [mo.mol_from_text("CCO ethanol"), mo.mol_from_text("c1ccccc1 benzene")]
    path = tmp_path / "x.sdf"
    mo.write_file(mols, str(path))
    back = mo.read_file(str(path))
    assert [m.GetProp("_Name") for m in back] == ["ethanol", "benzene"]
    smi = tmp_path / "x.smi"
    mo.write_file(mols, str(smi))
    assert len(mo.read_file(str(smi))) == 2


def test_descriptors_from_dataset():
    df = pd.DataFrame({"smi": ["CCO", "not_smiles", "", "c1ccccc1", "CC(=O)[O-].[Na+]"],
                       "f": ["C2H6O", "Xx", "H2O", "C6H6", "C2H3NaO2"]})
    ds = Dataset(df)
    out = sections.descriptors_from_dataset(ds, "smi", "f", ["MW", "HBD"], ["FW", "DBE"],
                                            strip_salts=True)
    mw = out.saved["MW"][1]
    assert mw.iloc[0] == pytest.approx(46.069, abs=1e-3)
    assert math.isnan(mw.iloc[1]) and math.isnan(mw.iloc[2])
    assert mw.iloc[4] == pytest.approx(60.052, abs=1e-3)  # acetate neutralised to acid
    assert out.saved["FW"][1].iloc[3] == pytest.approx(78.114, abs=1e-3)
    assert math.isnan(out.saved["FW"][1].iloc[1])
    assert out.saved["DBE"][1].iloc[3] == 4
    warnings = [i.text for i in out.section.items if getattr(i, "style", "") == "warning"]
    assert any("1 case(s) with invalid SMILES" in w for w in warnings)
    assert any("invalid formulas" in w for w in warnings)


def test_sections_build():
    from rsstatchem.core.chem import molecule as mo
    from rsstatchem.core.output import Image, Table
    s = sections.formula_section(["H2SO4", "SO4^2-", "bad("], isotopes=True)
    titles = [i.title for i in s.items]
    assert "Formula Summary" in titles and any("Isotope" in t for t in titles)
    summary = next(i for i in s.items if isinstance(i, Table))
    assert summary.rows[0][0] == "H₂SO₄"
    s = sections.molecule_section([("aspirin", mo.mol_from_text("CC(=O)Oc1ccccc1C(=O)O"))],
                                  rules=["Lipinski"])
    assert any(isinstance(i, Image) for i in s.items)
    res = rx.balance("C3H8 + O2 -> CO2 + H2O")
    st = rx.stoichiometry(res, {0: rx.Amount(44.1, "g"), 1: rx.Amount(4, "mol")})
    s = sections.balance_section(res, st)
    assert any(isinstance(i, Table) and i.title == "Stoichiometry" for i in s.items)
    s = sections.structure_section("CC(=O)O.OCC>>CC(=O)OCC.O")
    assert isinstance(s.items[-1], Image)


def test_ui_dialogs_offscreen(qapp):
    import rsstatchem.ui.chem as chem
    from rsstatchem.core.output import Section
    labels = [e[0] for e in chem.MENUS["Chemistry"] if e]
    assert "Balance Equation && Stoichiometry..." in labels
    for _, cls in (e for e in chem.MENUS["Chemistry"] if e):
        if cls is chem.DescriptorsDialog:
            continue
        assert cls.needs_data is False
        dlg = cls(None, None)
        dlg._refresh()
        assert isinstance(dlg.build(), Section)
    calc = chem.SolutionCalculatorDialog(None, None)
    for i in range(len(calc.pages)):
        calc.stack.setCurrentIndex(i)
        assert isinstance(calc.build(), Section), i
    ds = Dataset(pd.DataFrame({"smi": ["CCO", "c1ccccc1"]}))
    dlg = chem.DescriptorsDialog(ds, None)
    assert dlg.validate() is not None
    dlg._take("smi", dlg._targets[0])
    assert dlg.validate() is None
    assert set(dlg.compute().saved) >= {"MW", "cLogP"}
    chem.selftest()

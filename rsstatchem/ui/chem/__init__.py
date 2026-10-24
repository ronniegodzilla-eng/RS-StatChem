"""Chemistry menu, stage 1: formula tools, molecule properties and depiction (RDKit),
equation balancing and stoichiometry, solution calculators and dataset descriptors.

The calculators work without data (``needs_data = False``); "Descriptors from SMILES" saves
new variables through the ``.saved`` protocol. The structure/reaction editors are stage 2 and
build on :mod:`rsstatchem.core.chem` (see ``molecule.mol_from_graph`` / ``graph_from_mol``).
"""
from __future__ import annotations

from rsstatchem.ui.chem.descriptor_dialog import DescriptorsDialog
from rsstatchem.ui.chem.formula_dialogs import EmpiricalFormulaDialog, FormulaMassDialog
from rsstatchem.ui.chem.molecule_dialogs import InsertStructureDialog, MoleculePropertiesDialog
from rsstatchem.ui.chem.reaction_dialog import BalanceEquationDialog
from rsstatchem.ui.chem.solution_dialog import SolutionCalculatorDialog

__all__ = ["MENUS", "selftest", "FormulaMassDialog", "EmpiricalFormulaDialog",
           "MoleculePropertiesDialog", "DescriptorsDialog", "InsertStructureDialog",
           "BalanceEquationDialog", "SolutionCalculatorDialog"]

MENUS = {
    "Chemistry": [
        ("Formula && Molar Mass...", FormulaMassDialog),
        ("Empirical Formula...", EmpiricalFormulaDialog),
        None,
        ("Molecule Properties...", MoleculePropertiesDialog),
        ("Descriptors from SMILES...", DescriptorsDialog),
        ("Insert Structure into Output...", InsertStructureDialog),
        None,
        ("Balance Equation && Stoichiometry...", BalanceEquationDialog),
        ("Solution Calculator...", SolutionCalculatorDialog),
    ],
}


def selftest() -> None:
    """Headless check of the chemistry core (formulas, balancing, calculators, RDKit)."""
    import math

    import pandas as pd

    from rsstatchem.core.chem import formula as fm
    from rsstatchem.core.chem import reaction as rx
    from rsstatchem.core.chem import sections
    from rsstatchem.core.chem import solutions as sol
    from rsstatchem.core.dataset import Dataset

    def check(cond: bool, what: str) -> None:
        if not cond:
            raise AssertionError(f"chemistry selftest failed: {what}")

    check(abs(fm.molar_mass("CuSO4·5H2O") - 249.677) < 0.01, "molar mass of hydrate")
    check(abs(fm.molar_mass("K4[Fe(CN)6]") - 368.35) < 0.01, "molar mass with brackets")
    check(fm.parse_formula("[Fe(CN)6]3-").charge == -3, "charge parsing")
    check(fm.format_formula("SO4^2-") == "SO₄²⁻", "unicode formatting")
    emp = fm.empirical_from_composition({"C": 40.0, "H": 6.71, "O": None}, 180.16)
    check(emp.empirical.hill() == "CH2O" and emp.molecular.hill() == "C6H12O6", "empirical")
    r = rx.balance("KMnO4 + HCl -> KCl + MnCl2 + H2O + Cl2")
    check(r.coefficients == [2, 16, 2, 2, 8, 5], "redox balancing")
    check(rx.balance("MnO4- + Fe2+ + H+ -> Mn2+ + Fe3+ + H2O").coefficients ==
          [1, 5, 8, 1, 5, 4], "ionic balancing")
    check(rx.balance("H2O -> H2O2").status == "impossible", "impossible equation")
    st = rx.stoichiometry(rx.balance("2 H2 + O2 -> 2 H2O"),
                          {0: rx.Amount(4.0, "g"), 1: rx.Amount(16.0, "g")})
    check(st.limiting == [1], "limiting reagent")
    check(abs(sol.ph_weak_acid(0.1, 1.8e-5).pH - 2.875) < 0.01, "weak acid pH")
    check(abs(sol.buffer_ph(4.76, 0.1, 0.2).pH - (4.76 + math.log10(2))) < 1e-9, "buffer")
    check(abs(sol.dilution(c1=1.0, c2=0.1, v2=100)["v1"] - 10) < 1e-9, "dilution")
    sections.formula_section(["H2SO4", "SO4^2-"])
    sections.balance_section(r)
    try:
        from rsstatchem.core.chem import depict, molecule as mo
    except ImportError as exc:  # RDKit missing from the build
        raise AssertionError(f"chemistry selftest: RDKit unavailable ({exc})") from None
    caffeine = mo.mol_from_text("Cn1cnc2c1c(=O)n(C)c(=O)n2C")
    check(mo.to_inchikey(caffeine) == "RYYVLZVUVIJVGH-UHFFFAOYSA-N", "InChIKey")
    check(depict.depict_png(caffeine, 200, 150)[:4] == b"\x89PNG", "PNG depiction")
    check("<svg" in depict.depict_svg(caffeine, 200, 150), "SVG depiction")
    check(all(x.passed for x in mo.drug_likeness(mo.mol_from_text("CC(=O)Oc1ccccc1C(=O)O"),
                                                 ["Lipinski"])), "Lipinski")
    ds = Dataset(pd.DataFrame({"smi": ["CCO", "not_a_smiles", "c1ccccc1"]}))
    out = sections.descriptors_from_dataset(ds, "smi", None, ["MW"])
    mw = out.saved["MW"][1]
    check(math.isnan(mw.iloc[1]) and abs(mw.iloc[0] - 46.069) < 0.01, "dataset descriptors")

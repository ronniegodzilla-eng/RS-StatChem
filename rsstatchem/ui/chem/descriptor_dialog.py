"""Chemistry > Descriptors from SMILES: compute RDKit descriptors (and formula masses) for
every case and save them as new variables (MW_1, cLogP_1, …), SPSS "Save" style."""
from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QGridLayout, QGroupBox

from rsstatchem.core.chem import sections
from rsstatchem.ui.dialogs.base import VariablePickerDialog

DESCRIPTOR_KEYS = ["MW", "ExactMass", "cLogP", "MR", "TPSA", "HBD", "HBA", "RotB", "Rings",
                   "AroRings", "HeavyAtoms", "Atoms", "Heteroatoms", "FCsp3", "Charge", "Stereo",
                   "QED"]
DEFAULT_ON = {"MW", "cLogP", "TPSA", "HBD", "HBA", "RotB"}
LABELS = {"MW": "Molecular weight", "ExactMass": "Exact mass", "cLogP": "cLogP (Crippen)",
          "MR": "Molar refractivity", "TPSA": "TPSA", "HBD": "H-bond donors",
          "HBA": "H-bond acceptors", "RotB": "Rotatable bonds", "Rings": "Rings",
          "AroRings": "Aromatic rings", "HeavyAtoms": "Heavy atoms", "Atoms": "Atoms (incl. H)",
          "Heteroatoms": "Heteroatoms", "FCsp3": "Fraction Csp3", "Charge": "Formal charge",
          "Stereo": "Stereocentres", "QED": "QED"}


def is_string(var) -> bool:
    return var.type == "string"


class DescriptorsDialog(VariablePickerDialog):
    DEFAULTS = {**{f"d_{k}": k in DEFAULT_ON for k in DESCRIPTOR_KEYS},
                "f_FW": True, "f_MonoMass": False, "f_DBE": False, "strip": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Molecular Descriptors from SMILES", parent)
        self.add_target("smiles", "SMILES Variable:", accepts=is_string, single=True,
                        required=False)
        self.add_target("formula", "Formula Variable:", accepts=is_string, single=True,
                        required=False)
        box = QGroupBox("Descriptors from SMILES (saved as new variables)")
        grid = QGridLayout(box)
        for i, key in enumerate(DESCRIPTOR_KEYS):
            grid.addWidget(self._check(f"d_{key}", LABELS[key]), i // 4, i % 4)
        self.options_layout.addWidget(box)
        box = QGroupBox("From Formula Variable")
        grid = QGridLayout(box)
        for i, (key, text) in enumerate((("f_FW", "Molar mass"),
                                         ("f_MonoMass", "Monoisotopic mass"),
                                         ("f_DBE", "Double-bond equivalents"))):
            grid.addWidget(self._check(key, text), 0, i)
        self.options_layout.addWidget(box)
        self.add_checkbox("strip", "Strip salts and neutralise before computing "
                          "(largest fragment)")
        self.finish()

    def _check(self, key: str, text: str) -> QCheckBox:
        cb = QCheckBox(text)
        cb.setChecked(bool(self.opts.get(key)))
        cb.toggled.connect(lambda on, k=key: self.opts.__setitem__(k, on))
        return cb

    def _keys(self) -> list[str]:
        return [k for k in DESCRIPTOR_KEYS if self.opts.get(f"d_{k}")]

    def _formula_keys(self) -> list[str]:
        return [k for k in ("FW", "MonoMass", "DBE") if self.opts.get(f"f_{k}")]

    def validate(self) -> str | None:
        smiles, formula = self.selected_one("smiles"), self.selected_one("formula")
        if not smiles and not formula:
            return "Select a string variable with SMILES and/or one with formulas."
        if smiles and not self._keys():
            return "Select at least one descriptor."
        if formula and not self._formula_keys():
            return "Select at least one formula quantity."
        return None

    def compute(self):
        smiles, formula = self.selected_one("smiles"), self.selected_one("formula")
        return sections.descriptors_from_dataset(
            self.ds, smiles, formula, self._keys() if smiles else [],
            self._formula_keys() if formula else [], self.opts.get("strip", False))

"""Chemistry > Balance Equation & Stoichiometry dialog."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
                               QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout)

from rsstatchem.core.chem import formula as fm
from rsstatchem.core.chem import reaction as rx
from rsstatchem.core.chem import sections
from rsstatchem.ui.chem.widgets import CalculatorDialog, line, parse_number

EXAMPLES = [
    "C3H8 + O2 -> CO2 + H2O",
    "KMnO4 + HCl -> KCl + MnCl2 + H2O + Cl2",
    "Cu + HNO3 -> Cu(NO3)2 + NO + H2O",
    "MnO4- + Fe2+ + H+ -> Mn2+ + Fe3+ + H2O",
    "Cr2O7^2- + I- + H+ -> Cr^3+ + I2 + H2O",
    "Fe + O2 -> Fe2O3",
    "C6H12O6 + O2 -> CO2 + H2O",
    "CuSO4·5H2O -> CuSO4 + H2O",
]
UNITS = ["", *rx.MASS_UNITS, *rx.AMOUNT_UNITS, *rx.SOLUTION_UNITS, *rx.GAS_UNITS]
COLS = ["Species", "Role", "Coef.", "M (g/mol)", "Amount", "Unit", "Molarity (M)",
        "Actual yield (g)"]


class BalanceEquationDialog(CalculatorDialog):
    """Balance an equation (exact rational null space) and run a stoichiometry table."""

    def __init__(self, ds=None, parent=None):
        super().__init__(ds, parent, "Balance Equation and Stoichiometry")
        self.setMinimumSize(780, 520)
        row = QHBoxLayout()
        row.addWidget(QLabel("Equation:"))
        self.eq = line(EXAMPLES[0], "e.g. Fe + O2 -> Fe2O3   (ions: Fe3+, SO4^2-, e-)")
        self.remember("eq", self.eq)
        self.eq.textChanged.connect(self.schedule)
        row.addWidget(self.eq, 1)
        self.examples = QComboBox()
        self.examples.addItem("Examples…")
        self.examples.addItems(EXAMPLES)
        self.examples.activated.connect(self._example)
        row.addWidget(self.examples)
        self.body.addLayout(row)
        self.result_label = QLabel()
        self.result_label.setTextFormat(Qt.TextFormat.RichText)
        self.result_label.setWordWrap(True)
        self.result_label.setMinimumHeight(48)
        self.result_label.setStyleSheet("QLabel { font-size: 15px; padding: 6px; "
                                        "background: #fafafa; border: 1px solid #ddd; }")
        self.body.addWidget(self.result_label)
        self.stoich = QGroupBox("Stoichiometry (optional: enter amounts of reactants; actual "
                                "yields of products for % yield)")
        self.stoich.setCheckable(True)
        self.stoich.setChecked(True)
        lay = QVBoxLayout(self.stoich)
        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setMinimumSectionSize(70)
        self.table.verticalHeader().setVisible(False)
        lay.addWidget(self.table)
        gas = QHBoxLayout()
        form = QFormLayout()
        self.temp = line("0", "°C", 80)
        self.pres = line("1", "atm", 80)
        self.remember("temp", self.temp)
        self.remember("pres", self.pres)
        form.addRow("Gas volumes at T (°C):", self.temp)
        gas.addLayout(form)
        form = QFormLayout()
        form.addRow("P (atm):", self.pres)
        gas.addLayout(form)
        gas.addStretch()
        lay.addLayout(gas)
        self.body.addWidget(self.stoich, 1)
        self._species_key = None
        self.preview()

    def _example(self, i: int) -> None:
        if i > 0:
            self.eq.setText(self.examples.itemText(i))
            self.examples.setCurrentIndex(0)

    def preview(self) -> None:
        try:
            res = rx.balance(self.eq.text())
        except fm.FormulaError as exc:
            self.result_label.setText(f"<span style='color:#b00020'>{exc}</span>")
            self._fill_table(None)
            return
        color = "#1b5e20" if res.status == "balanced" else "#b00020"
        text = res.equation("html") if res.coefficients is not None else \
            res.equation("html", [None] * len(res.species))
        self.result_label.setText(f"{text}<br><span style='font-size:12px; color:{color}'>"
                            f"{res.message}</span>")
        self._fill_table(res)

    def _fill_table(self, res) -> None:
        key = None if res is None or res.coefficients is None else \
            tuple(s.text for s in res.species)
        if key == self._species_key:
            if res is not None and res.coefficients is not None:
                for r, c in enumerate(res.coefficients):
                    self.table.item(r, 2).setText(str(c))
            return
        old = self._amounts_by_species()
        self._species_key = key
        self.table.setRowCount(0)
        if key is None:
            return
        self.table.setRowCount(len(res.species))
        for r, s in enumerate(res.species):
            name = QLabel(fm.format_formula(s.text, "html"))
            name.setTextFormat(Qt.TextFormat.RichText)
            name.setContentsMargins(4, 0, 0, 0)
            self.table.setCellWidget(r, 0, name)
            for c, text in ((1, s.role), (2, str(res.coefficients[r]))):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(r, c, item)
            self.table.setItem(r, 3, QTableWidgetItem(f"{s.formula.molar_mass:.3f}"))
            prev = old.get(s.text, {})
            self.table.setItem(r, 4, QTableWidgetItem(prev.get("amount", "")))
            unit = QComboBox()
            unit.addItems(UNITS)
            unit.setCurrentText(prev.get("unit", "g"))
            self.table.setCellWidget(r, 5, unit)
            self.table.setItem(r, 6, QTableWidgetItem(prev.get("molarity", "")))
            actual = QTableWidgetItem(prev.get("actual", ""))
            if s.side > 0:
                actual.setFlags(Qt.ItemFlag.NoItemFlags)
            self.table.setItem(r, 7, actual)

    def _amounts_by_species(self) -> dict:
        out = {}
        if self._species_key is None:
            return out
        for r, name in enumerate(self._species_key):
            if r >= self.table.rowCount():
                break
            out[name] = {"amount": self._cell(r, 4), "unit": self.table.cellWidget(r, 5)
                         .currentText(), "molarity": self._cell(r, 6),
                         "actual": self._cell(r, 7)}
        return out

    def _cell(self, r: int, c: int) -> str:
        item = self.table.item(r, c)
        return item.text().strip() if item else ""

    def build(self):
        text = self.eq.text().strip()
        try:
            res = rx.balance(text)
        except fm.FormulaError as exc:
            raise ValueError(str(exc)) from None
        stoich = None
        if self.stoich.isChecked() and res.coefficients is not None and \
                self._species_key == tuple(s.text for s in res.species):
            amounts, actual, masses = {}, {}, {}
            t = parse_number(self.temp.text())
            p = parse_number(self.pres.text())
            for r, s in enumerate(res.species):
                m = parse_number(self._cell(r, 3))
                if m is not None:
                    if m <= 0:
                        raise ValueError(f"Molar mass of {s.text} must be positive.")
                    masses[r] = m
                value = parse_number(self._cell(r, 4))
                if value is not None:
                    unit = self.table.cellWidget(r, 5).currentText() or "g"
                    amounts[r] = rx.Amount(value, unit, parse_number(self._cell(r, 6)),
                                           273.15 + (0.0 if t is None else t),
                                           1.0 if p is None else p)
                a = parse_number(self._cell(r, 7))
                if a is not None and s.side < 0:
                    actual[r] = a
            if amounts:
                stoich = rx.stoichiometry(res, amounts, actual, masses)
        return sections.balance_section(res, stoich, text)

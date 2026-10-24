"""Chemistry > Formula & Molar Mass and Chemistry > Empirical Formula dialogs."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QDoubleSpinBox, QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QHeaderView, QLabel, QPlainTextEdit,
                               QPushButton, QSpinBox, QTableWidget,
                               QTableWidgetItem, QVBoxLayout)

from rsstatchem.core.chem import formula as fm
from rsstatchem.core.chem import sections
from rsstatchem.ui.chem.widgets import CalculatorDialog, line, parse_number


def _split_formulas(text: str) -> list[str]:
    out = []
    for raw in text.replace(";", "\n").splitlines():
        raw = raw.strip()
        if raw:
            out.append(raw)
    return out


class FormulaMassDialog(CalculatorDialog):
    """Molar mass, monoisotopic mass, composition and Hill formula of typed formulas."""

    def __init__(self, ds=None, parent=None):
        super().__init__(ds, parent, "Formula and Molar Mass")
        self.setMinimumWidth(620)
        self.body.addWidget(QLabel("Formulas (one per line), e.g.  H2SO4,  CuSO4·5H2O,  "
                                   "K4[Fe(CN)6],  SO4^2-,  [13C]H4:"))
        self.text = QPlainTextEdit()
        self.text.setProperty("default_value", "H2SO4")
        self.text.setPlainText("H2SO4")
        self.text.setFixedHeight(96)
        self.remember("text", self.text)
        self.text.textChanged.connect(self.schedule)
        self.body.addWidget(self.text)
        box = QGroupBox("Preview")
        lay = QVBoxLayout(box)
        self.preview_label = QLabel()
        self.preview_label.setTextFormat(Qt.TextFormat.RichText)
        self.preview_label.setWordWrap(True)
        self.preview_label.setMinimumHeight(118)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        lay.addWidget(self.preview_label)
        self.body.addWidget(box)
        opts = QGroupBox("Display")
        row = QHBoxLayout(opts)
        self.comp = QCheckBox("Elemental composition (mass %)")
        self.comp.setChecked(True)
        self.comp.setProperty("default_value", True)
        self.iso = QCheckBox("Isotope pattern")
        self.iso.setProperty("default_value", False)
        self.remember("comp", self.comp)
        self.remember("iso", self.iso)
        row.addWidget(self.comp)
        row.addWidget(self.iso)
        row.addStretch()
        self.body.addWidget(opts)
        hint = QLabel("Hydrates/adducts: · . or *   Charges: ^2- or trailing 2-   Isotopes: "
                      "[13C], D, T")
        hint.setStyleSheet("color: #666;")
        self.body.addWidget(hint)
        self.preview()

    def preview(self) -> None:
        lines = []
        for t in _split_formulas(self.text.toPlainText())[:8]:
            try:
                f = fm.parse_formula(t)
                lines.append(f"<b>{fm.format_formula(t, 'html')}</b> &nbsp; M = "
                             f"{f.molar_mass:.3f} g/mol &nbsp;·&nbsp; Hill: "
                             f"{fm.format_formula(f.hill(), 'html')}")
            except fm.FormulaError as exc:
                lines.append(f"<span style='color:#b00020'>{t}: {exc}</span>")
        self.preview_label.setText("<br>".join(lines) or "<i>Type a formula</i>")

    def build(self):
        texts = _split_formulas(self.text.toPlainText())
        if not texts:
            raise ValueError("Enter at least one formula.")
        errors = []
        for t in texts:
            try:
                fm.parse_formula(t)
            except fm.FormulaError as exc:
                errors.append(f"{t}: {exc}")
        if len(errors) == len(texts):
            raise ValueError("\n".join(errors))
        return sections.formula_section(texts, self.comp.isChecked(), self.iso.isChecked())


class EmpiricalFormulaDialog(CalculatorDialog):
    """Empirical and molecular formula from mass % (or CHN combustion data)."""

    ROWS = 8

    def __init__(self, ds=None, parent=None):
        super().__init__(ds, parent, "Empirical Formula")
        self.setMinimumWidth(640)
        top = QHBoxLayout()
        left = QVBoxLayout()
        lbl = QLabel("Mass composition (leave one % empty to take that element by "
                     "difference):")
        lbl.setWordWrap(True)
        left.addWidget(lbl)
        self.table = QTableWidget(self.ROWS, 2)
        self.table.setHorizontalHeaderLabels(["Element", "Mass %"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.setFixedHeight(24 * self.ROWS + 30)
        for r, (sym, pct) in enumerate([("C", "40.00"), ("H", "6.71"), ("O", "")]):
            self.table.setItem(r, 0, QTableWidgetItem(sym))
            self.table.setItem(r, 1, QTableWidgetItem(pct))
        self.table.itemChanged.connect(self.schedule)
        left.addWidget(self.table)
        left.addStretch()
        top.addLayout(left, 1)

        right = QVBoxLayout()
        box = QGroupBox("Molecular Formula")
        form = QFormLayout(box)
        self.molar = line("", "optional, g/mol", 120)
        self.molar.textChanged.connect(self.schedule)
        self.remember("molar", self.molar)
        form.addRow("Molar mass:", self.molar)
        right.addWidget(box)
        box = QGroupBox("Whole-number Search")
        form = QFormLayout(box)
        self.tol = QDoubleSpinBox()
        self.tol.setRange(0.01, 0.4)
        self.tol.setSingleStep(0.01)
        self.tol.setValue(0.1)
        self.tol.setProperty("default_value", 0.1)
        self.maxk = QSpinBox()
        self.maxk.setRange(1, 30)
        self.maxk.setValue(12)
        self.maxk.setProperty("default_value", 12)
        self.remember("tol", self.tol)
        self.remember("maxk", self.maxk)
        self.tol.valueChanged.connect(self.schedule)
        self.maxk.valueChanged.connect(self.schedule)
        form.addRow("Tolerance:", self.tol)
        form.addRow("Largest multiplier:", self.maxk)
        right.addWidget(box)
        box = QGroupBox("Combustion Analysis (CHN)")
        grid = QGridLayout(box)
        self.sample = line("", "mg")
        self.co2 = line("", "mg")
        self.h2o = line("", "mg")
        self.n2 = line("", "mg (optional)")
        for i, (lbl, w) in enumerate([("Sample:", self.sample), ("CO₂:", self.co2),
                                      ("H₂O:", self.h2o), ("N₂:", self.n2)]):
            grid.addWidget(QLabel(lbl), i, 0)
            grid.addWidget(w, i, 1)
        fill = QPushButton("Fill Mass %")
        fill.setAutoDefault(False)
        fill.clicked.connect(self._fill_combustion)
        grid.addWidget(fill, 4, 0, 1, 2)
        right.addWidget(box)
        right.addStretch()
        top.addLayout(right)
        self.body.addLayout(top)
        self.result_label = QLabel()
        self.result_label.setTextFormat(Qt.TextFormat.RichText)
        self.result_label.setWordWrap(True)
        self.body.addWidget(self.result_label)
        self._source = ""
        self.preview()

    def _percents(self) -> dict:
        out = {}
        for r in range(self.table.rowCount()):
            sym_item, pct_item = self.table.item(r, 0), self.table.item(r, 1)
            sym = sym_item.text().strip() if sym_item else ""
            pct = pct_item.text().strip() if pct_item else ""
            if not sym:
                if pct:
                    raise ValueError(f"Row {r + 1}: enter the element symbol.")
                continue
            sym = sym[0].upper() + sym[1:].lower()
            if sym in out:
                raise ValueError(f"{sym} is listed twice.")
            out[sym] = parse_number(pct)
        if not out:
            raise ValueError("Enter element symbols and mass percentages.")
        return out

    def _fill_combustion(self) -> None:
        from PySide6.QtWidgets import QMessageBox
        try:
            sample = parse_number(self.sample.text())
            co2, h2o = parse_number(self.co2.text()), parse_number(self.h2o.text())
            if sample is None or co2 is None or h2o is None:
                raise ValueError("Enter the sample, CO₂ and H₂O masses.")
            pct = fm.combustion_composition(sample, co2, h2o, parse_number(self.n2.text()) or 0)
        except ValueError as exc:
            QMessageBox.warning(self, self.windowTitle(), str(exc))
            return
        self.table.blockSignals(True)
        self.table.clearContents()
        for r, (sym, v) in enumerate(pct.items()):
            self.table.setItem(r, 0, QTableWidgetItem(sym))
            self.table.setItem(r, 1, QTableWidgetItem("" if v is None else f"{v:.3f}"))
        self.table.blockSignals(False)
        self._source = (f"Combustion analysis: sample {sample:g} mg, CO₂ {co2:g} mg, "
                        f"H₂O {h2o:g} mg" + (f", N₂ {self.n2.text()} mg" if self.n2.text()
                                             else "") + "; O by difference.")
        self.preview()

    def _result(self):
        return fm.empirical_from_composition(self._percents(), parse_number(self.molar.text()),
                                             self.tol.value(), self.maxk.value())

    def preview(self) -> None:
        try:
            res = self._result()
        except (ValueError, KeyError) as exc:
            msg = str(exc).strip(chr(39))
            self.result_label.setText(f"<span style='color:#b00020'>{msg}</span>")
            return
        text = (f"Empirical formula: <b>{fm.format_formula(res.empirical.hill(), 'html')}</b>"
                f" (M = {res.empirical_mass:.3f} g/mol)")
        if res.molecular is not None:
            text += (f" &nbsp;→&nbsp; molecular formula: <b>"
                     f"{fm.format_formula(res.molecular.hill(), 'html')}</b> (n = {res.n:.2f})")
        if res.warnings:
            text += "<br><span style='color:#b00020'>" + "<br>".join(res.warnings) + "</span>"
        self.result_label.setText(text)

    def build(self):
        return sections.empirical_section(self._result(), self._source)

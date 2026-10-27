"""Dialogs of the structure editor: periodic table, atom properties, arrow properties,
document settings (style presets), structure import, image export options, reaction
balance check, stoichiometry grid and "Insert SMILES into Data Editor"."""
from __future__ import annotations

import math

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QDoubleSpinBox, QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QMessageBox, QPlainTextEdit, QPushButton, QSpinBox,
                               QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout)

from rsstatchem.core.chem import elements as el
from rsstatchem.core.chem.formula import format_formula
from rsstatchem.core.chem_draw import analysis
from rsstatchem.core.chem_draw.model import (ARROW_KINDS, BRACKET_KINDS, POLYMER_TYPES,
                                             REPEAT_PATTERNS, Arrow, Atom, Bond, Bracket,
                                             Document)
from rsstatchem.core.chem_draw.style import PRESETS, Style

__all__ = ["PeriodicTableDialog", "AtomPropertiesDialog", "ArrowDialog",
           "DocumentSettingsDialog", "ImportDialog", "ImageOptionsDialog", "BalanceDialog",
           "StoichiometryDialog", "DataInsertDialog", "pt_position", "BondDialog",
           "BracketDialog", "ARROW_NAMES"]

ARROW_NAMES = {"forward": "Reaction (→)", "equilibrium": "Equilibrium (⇌)",
               "retro": "Retrosynthetic (⇒)", "resonance": "Resonance (↔)",
               "curved": "Curved, two-electron", "fishhook": "Curved, one-electron (fishhook)",
               "line": "Plain line", "nogo": "No-go (crossed, no reaction)",
               "unbalanced": "Unbalanced equilibrium", "block": "Block (hollow / filled)",
               "double": "Double-headed (bendable)", "arc": "Arc arrow",
               "dipole": "Dipole"}


def _buttons(dlg: QDialog, ok_text: str = "OK") -> QDialogButtonBox:
    bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                          QDialogButtonBox.StandardButton.Cancel)
    bb.button(QDialogButtonBox.StandardButton.Ok).setText(ok_text)
    bb.accepted.connect(dlg.accept)
    bb.rejected.connect(dlg.reject)
    return bb


def pt_position(z: int) -> tuple[int, int]:
    """(row, column) of element Z in the 18-column periodic table; lanthanides/actinides
    in rows 8 and 9."""
    if z <= 2:
        return 0, 0 if z == 1 else 17
    for period, (lo, hi) in enumerate(((3, 10), (11, 18)), start=1):
        if lo <= z <= hi:
            k = z - lo
            return period, k if k < 2 else k + 10
    for period, lo in ((3, 19), (4, 37)):
        if lo <= z < lo + 18:
            return period, z - lo
    for period, lo, frow in ((5, 55, 8), (6, 87, 9)):
        if lo <= z < lo + 32:
            k = z - lo
            if k < 2:
                return period, k
            if k < 17:
                return frow, k
            return period, k - 14
    return 9, 0


class PeriodicTableDialog(QDialog):
    def __init__(self, parent=None, current: str = "C"):
        super().__init__(parent)
        self.setWindowTitle("Periodic Table")
        self.symbol = current
        grid = QGridLayout()
        grid.setSpacing(2)
        common = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"}
        for z, e in sorted(el.BY_Z.items()):
            r, c = pt_position(z)
            b = QToolButton()
            b.setText(e.symbol)
            b.setToolTip(f"{z} {e.name}\n{e.weight:g}")
            b.setFixedSize(34, 30)
            if e.symbol in common:
                b.setStyleSheet("font-weight: bold;")
            b.clicked.connect(lambda _=False, s=e.symbol: self._pick(s))
            grid.addWidget(b, r + (1 if r >= 8 else 0), c)
        grid.setRowMinimumHeight(8, 8)
        lay = QVBoxLayout(self)
        lay.addLayout(grid)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _pick(self, s: str) -> None:
        self.symbol = s
        self.accept()


class AtomPropertiesDialog(QDialog):
    """Element, label, charge, isotope, fixed hydrogen count, radical, atom map."""

    def __init__(self, atom: Atom, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Atom Properties")
        form = QFormLayout()
        row = QHBoxLayout()
        self.symbol = QLineEdit(atom.symbol)
        pt = QPushButton("Periodic Table...")
        pt.setAutoDefault(False)
        pt.clicked.connect(self._table)
        row.addWidget(self.symbol)
        row.addWidget(pt)
        form.addRow("Element:", row)
        self.label = QLineEdit(atom.label)
        self.label.setPlaceholderText("nickname (Ph, OMe, Boc...) or generic label (R1)")
        form.addRow("Label:", self.label)
        self.charge = QSpinBox()
        self.charge.setRange(-8, 8)
        self.charge.setValue(atom.charge)
        form.addRow("Charge:", self.charge)
        self.isotope = QSpinBox()
        self.isotope.setRange(0, 300)
        self.isotope.setSpecialValueText("natural")
        self.isotope.setValue(atom.isotope)
        form.addRow("Isotope (mass number):", self.isotope)
        hrow = QHBoxLayout()
        self.fix_h = QCheckBox("Fixed")
        self.h = QSpinBox()
        self.h.setRange(0, 6)
        self.fix_h.setChecked(atom.explicit_h is not None)
        self.h.setValue(atom.explicit_h or 0)
        self.h.setEnabled(atom.explicit_h is not None)
        self.fix_h.toggled.connect(self.h.setEnabled)
        hrow.addWidget(self.fix_h)
        hrow.addWidget(self.h)
        form.addRow("Hydrogens:", hrow)
        self.radical = QComboBox()
        self.radical.addItems(["None", "Doublet (•)", "Two radicals (••)"])
        self.radical.setCurrentIndex(min(2, atom.radicals))
        form.addRow("Radical:", self.radical)
        self.map = QSpinBox()
        self.map.setRange(0, 999)
        self.map.setSpecialValueText("none")
        self.map.setValue(atom.map_number)
        form.addRow("Atom map number:", self.map)
        self.show_c = QCheckBox("Show carbon label")
        self.show_c.setChecked(atom.show_carbon)
        form.addRow("", self.show_c)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(_buttons(self))

    def _table(self) -> None:
        dlg = PeriodicTableDialog(self, self.symbol.text())
        if dlg.exec():
            self.symbol.setText(dlg.symbol)

    def accept(self) -> None:
        s = self.symbol.text().strip()
        if not el.is_symbol(s):
            QMessageBox.warning(self, "Atom Properties", f"'{s}' is not an element symbol.")
            return
        super().accept()

    def apply(self, doc: Document, aid: int) -> None:
        a = doc.atoms[aid]
        a.symbol = self.symbol.text().strip()
        a.label = self.label.text().strip()
        a.charge = self.charge.value()
        a.isotope = self.isotope.value()
        a.explicit_h = self.h.value() if self.fix_h.isChecked() else None
        a.radicals = self.radical.currentIndex()
        a.map_number = self.map.value()
        a.show_carbon = self.show_c.isChecked()


class ArrowDialog(QDialog):
    def __init__(self, arrow: Arrow, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Arrow Properties")
        form = QFormLayout()
        self.kind = QComboBox()
        for k in ARROW_KINDS:
            self.kind.addItem(ARROW_NAMES[k], k)
        self.kind.setCurrentIndex(ARROW_KINDS.index(arrow.kind))
        form.addRow("Arrow:", self.kind)
        self.above = QLineEdit(arrow.above)
        self.above.setPlaceholderText("reagents, e.g. H2SO4 (cat.), EtOH")
        self.below = QLineEdit(arrow.below)
        self.below.setPlaceholderText("conditions, e.g. reflux, 2 h")
        form.addRow("Above the arrow:", self.above)
        form.addRow("Below the arrow:", self.below)
        self.dashed = QCheckBox("Dashed")
        self.dashed.setChecked(arrow.dashed)
        self.bold = QCheckBox("Bold")
        self.bold.setChecked(arrow.bold)
        row = QHBoxLayout()
        row.addWidget(self.dashed)
        row.addWidget(self.bold)
        form.addRow("Line:", row)
        self.fill = QComboBox()
        for text, v in (("Hollow", ""), ("Filled", "filled"), ("Shaded", "shaded")):
            self.fill.addItem(text, v)
        self.fill.setCurrentIndex(max(0, self.fill.findData(arrow.fill)))
        form.addRow("Block arrow fill:", self.fill)
        self.sweep = QComboBox()
        for v in (90, 120, 180, 270):
            self.sweep.addItem(f"{v}°", float(v))
        self.sweep.setCurrentIndex(max(0, self.sweep.findData(float(arrow.sweep))))
        form.addRow("Arc:", self.sweep)
        note = QLabel("Use \\n for a line break. Formulas get subscripts automatically.")
        note.setStyleSheet("color: #666;")
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(note)
        lay.addWidget(_buttons(self))

    def apply(self, arrow: Arrow) -> None:
        arrow.kind = self.kind.currentData()
        arrow.above = self.above.text().replace("\\n", "\n")
        arrow.below = self.below.text().replace("\\n", "\n")
        arrow.dashed = self.dashed.isChecked()
        arrow.bold = self.bold.isChecked()
        arrow.fill = self.fill.currentData()
        arrow.sweep = self.sweep.currentData()
        if arrow.kind in ("curved", "fishhook", "double") and abs(arrow.bend) < 1e-6:
            arrow.bend = -0.35 * math.hypot(arrow.x2 - arrow.x1, arrow.y2 - arrow.y1)


class BondDialog(QDialog):
    """Bond Properties (bond hotkey "/"): order, display style and double-bond position."""

    ORDERS = (("Single", 1), ("Double", 2), ("Triple", 3), ("Quadruple", 4),
              ("Aromatic", 1.5))
    STYLES = (("Plain", ""), ("Wedged", "wedge"), ("Hashed Wedged", "hash"),
              ("Hollow Wedged", "hollow"), ("Hashed", "hashed"), ("Bold", "bold"),
              ("Dashed", "dashed"), ("Wavy", "wavy"), ("Dative", "dative"),
              ("Either (crossed double)", "either"))

    def __init__(self, bond: Bond, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bond Properties")
        form = QFormLayout()
        self.order = QComboBox()
        for text, v in self.ORDERS:
            self.order.addItem(text, v)
        self.order.setCurrentIndex(max(0, self.order.findData(bond.order)))
        self.style_box = QComboBox()
        for text, v in self.STYLES:
            self.style_box.addItem(text, v)
        self.style_box.setCurrentIndex(max(0, self.style_box.findData(bond.style)))
        self.pos = QComboBox()
        for text, v in (("Automatic", "auto"), ("Centered", "center"), ("Left", "left"),
                        ("Right", "right")):
            self.pos.addItem(text, v)
        self.pos.setCurrentIndex(max(0, self.pos.findData(bond.double_pos)))
        form.addRow("Order:", self.order)
        form.addRow("Display:", self.style_box)
        form.addRow("Double position:", self.pos)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(_buttons(self))

    def apply(self, bond: Bond) -> None:
        bond.order = self.order.currentData()
        bond.style = self.style_box.currentData()
        bond.double_pos = self.pos.currentData()


POLYMER_NAMES = {"": "(none)", "SRU": "Structural repeating unit (SRU)",
                 "copolymer": "Copolymer", "alternating": "Alternating copolymer",
                 "random": "Random copolymer", "block": "Block copolymer",
                 "graft": "Graft copolymer", "mer": "Mer", "monomer": "Monomer",
                 "crosslink": "Crosslink", "mixture": "Mixture", "component": "Component"}
REPEAT_NAMES = {"": "(none)", "ht": "Head-to-tail", "hh": "Head-to-head",
                "eu": "Either / unknown"}


class BracketDialog(QDialog):
    """Bracket Properties: bracket shape, single or paired, repeat count label and the
    polymer semantics (SRU / copolymer type, repeat pattern)."""

    def __init__(self, br: Bracket, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bracket Properties")
        form = QFormLayout()
        self.kind = QComboBox()
        for k, text in zip(BRACKET_KINDS, ("Square [ ]", "Round ( )", "Curly { }")):
            self.kind.addItem(text, k)
        self.kind.setCurrentIndex(max(0, self.kind.findData(br.kind)))
        self.sides = QComboBox()
        for text, v in (("Pair", "both"), ("Left only", "left"), ("Right only", "right")):
            self.sides.addItem(text, v)
        self.sides.setCurrentIndex(max(0, self.sides.findData(br.sides)))
        self.label = QLineEdit(br.label)
        self.label.setPlaceholderText("n, 2, ‡ ...")
        self.polymer = QComboBox()
        for k in POLYMER_TYPES:
            self.polymer.addItem(POLYMER_NAMES.get(k, k), k)
        self.polymer.setCurrentIndex(max(0, self.polymer.findData(br.polymer)))
        self.repeat = QComboBox()
        for k in REPEAT_PATTERNS:
            self.repeat.addItem(REPEAT_NAMES.get(k, k), k)
        self.repeat.setCurrentIndex(max(0, self.repeat.findData(br.repeat)))
        form.addRow("Bracket:", self.kind)
        form.addRow("Sides:", self.sides)
        form.addRow("Repeat count:", self.label)
        form.addRow("Polymer type:", self.polymer)
        form.addRow("Repeat pattern:", self.repeat)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(_buttons(self))

    def apply(self, br: Bracket) -> None:
        br.kind = self.kind.currentData()
        br.sides = self.sides.currentData()
        br.label = self.label.text()
        br.polymer = self.polymer.currentData()
        br.repeat = self.repeat.currentData()
        if br.polymer and not br.label:
            br.label = "n"


class DocumentSettingsDialog(QDialog):
    """ChemDraw File > Document Settings: preset plus individual drawing settings."""

    def __init__(self, style: Style, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Document Settings")
        form = QFormLayout()
        self.preset = QComboBox()
        self.preset.addItem("(custom)")
        self.preset.addItems(list(PRESETS))
        self.preset.activated.connect(self._preset)
        form.addRow("Apply settings from:", self.preset)

        def spin(v, lo, hi, dec=1, suffix=" pt"):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(dec)
            s.setValue(v)
            s.setSuffix(suffix)
            return s
        self.bond_length = spin(style.bond_length, 5, 100)
        self.line_width = spin(style.line_width, 0.1, 10, 2)
        self.bold_width = spin(style.bold_width, 0.5, 20, 2)
        self.spacing = spin(style.bond_spacing, 5, 40, 0, " % of length")
        self.hash = spin(style.hash_spacing, 0.5, 10, 2)
        self.margin = spin(style.margin, 0, 10, 2)
        self.font = QLineEdit(style.font_family)
        self.font_size = spin(style.font_size, 4, 48)
        self.caption = spin(style.caption_size, 4, 48)
        self.color = QCheckBox("Color atom labels by element")
        self.color.setChecked(style.color_by_element)
        for label, w in (("Fixed bond length:", self.bond_length),
                         ("Line width:", self.line_width), ("Bold width:", self.bold_width),
                         ("Bond spacing:", self.spacing), ("Hash spacing:", self.hash),
                         ("Margin width:", self.margin), ("Label font:", self.font),
                         ("Label size:", self.font_size), ("Caption size:", self.caption),
                         ("", self.color)):
            form.addRow(label, w)
        self._name = style.name
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(_buttons(self))

    def _preset(self, idx: int) -> None:
        if idx <= 0:
            return
        s = PRESETS[self.preset.currentText()]
        self._name = s.name
        self.bond_length.setValue(s.bond_length)
        self.line_width.setValue(s.line_width)
        self.bold_width.setValue(s.bold_width)
        self.spacing.setValue(s.bond_spacing)
        self.hash.setValue(s.hash_spacing)
        self.margin.setValue(s.margin)
        self.font.setText(s.font_family)
        self.font_size.setValue(s.font_size)
        self.caption.setValue(s.caption_size)
        self.color.setChecked(s.color_by_element)

    def style(self) -> Style:
        return Style(self._name, self.bond_length.value(), self.line_width.value(),
                     self.bold_width.value(), self.spacing.value(), self.hash.value(),
                     self.margin.value(), self.font.text().strip() or "Arial",
                     self.font_size.value(), self.caption.value(), self.color.isChecked())


class ImportDialog(QDialog):
    """Paste SMILES, InChI, MOL/SDF, reaction SMILES or RXN text."""

    def __init__(self, parent=None, text: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Import Structure")
        self.resize(520, 360)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Paste a SMILES, InChI, MOL / SDF block, reaction SMILES "
                             "(A.B>>C) or RXN block:"))
        self.edit = QPlainTextEdit(text)
        lay.addWidget(self.edit, 1)
        self.kind = QLabel("")
        self.kind.setStyleSheet("color: #666;")
        lay.addWidget(self.kind)
        self.edit.textChanged.connect(self._detect)
        self._detect()
        lay.addWidget(_buttons(self, "Insert"))

    def _detect(self) -> None:
        from rsstatchem.core.chem import molecule as mo
        t = self.edit.toPlainText().strip()
        if not t:
            self.kind.setText("")
            return
        fmt = "RXN" if t.startswith("$RXN") else mo.detect_format(t).upper()
        self.kind.setText(f"Detected format: {fmt}")

    def text(self) -> str:
        return self.edit.toPlainText()


class ImageOptionsDialog(QDialog):
    def __init__(self, parent=None, dpi: int = 300, transparent: bool = True):
        super().__init__(parent)
        self.setWindowTitle("PNG Options")
        form = QFormLayout()
        self.dpi = QComboBox()
        for d in (150, 300, 600, 1200):
            self.dpi.addItem(f"{d} dpi", d)
        self.dpi.setCurrentIndex(max(0, [150, 300, 600, 1200].index(dpi)
                                     if dpi in (150, 300, 600, 1200) else 1))
        self.transparent = QCheckBox("Transparent background")
        self.transparent.setChecked(transparent)
        form.addRow("Resolution:", self.dpi)
        form.addRow("", self.transparent)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(_buttons(self))

    def values(self) -> tuple[int, bool]:
        return int(self.dpi.currentData()), self.transparent.isChecked()


def _table(columns: list[str]) -> QTableWidget:
    t = QTableWidget(0, len(columns))
    t.setHorizontalHeaderLabels(columns)
    t.verticalHeader().setVisible(False)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    t.horizontalHeader().setStretchLastSection(True)
    return t


def _ro(text: str) -> QTableWidgetItem:
    it = QTableWidgetItem(text)
    it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return it


class BalanceDialog(QDialog):
    """Check reaction balance: atom counts per side and suggested coefficients."""

    def __init__(self, doc: Document, parent=None, send=None):
        super().__init__(parent)
        self.setWindowTitle("Check Reaction Balance")
        self.resize(520, 420)
        self.res = analysis.balance_check(doc)
        lay = QVBoxLayout(self)
        eq = QLabel()
        eq.setTextFormat(Qt.TextFormat.RichText)
        eq.setWordWrap(True)
        if self.res.ok:
            eq.setText("<b>As drawn:</b> " + _html_eq(self.res.equation) +
                       ("<br><b>Balanced:</b> " + _html_eq(self.res.suggestion)
                        if self.res.suggestion else ""))
        lay.addWidget(eq)
        self.table = _table(["Element", "Reactants", "Products", "Difference"])
        for e, lft, rgt in self.res.rows:
            r = self.table.rowCount()
            self.table.insertRow(r)
            for c, v in enumerate((e, str(lft), str(rgt), f"{rgt - lft:+d}" if lft != rgt
                                   else "0")):
                it = _ro(v)
                if lft != rgt:
                    it.setForeground(Qt.GlobalColor.red)
                self.table.setItem(r, c, it)
        lay.addWidget(self.table, 1)
        msg = QLabel(self.res.message)
        msg.setWordWrap(True)
        msg.setStyleSheet("color: #1B5E20;" if self.res.balanced else "color: #B71C1C;")
        lay.addWidget(msg)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.reject)
        if send is not None and self.res.ok:
            b = bb.addButton("Send to Output Viewer", QDialogButtonBox.ButtonRole.ActionRole)
            b.clicked.connect(lambda: send(self.section()))
        lay.addWidget(bb)

    def section(self):
        from rsstatchem.core.output import Section, Table
        s = Section("Reaction Balance")
        s.log(f"CHEM BALANCE DRAWING '{self.res.equation}'.")
        s.table(Table("Atom Balance", ["Element", "Reactants", "Products", "Balanced"],
                      [[e, str(lft), str(rgt), "Yes" if lft == rgt else "No"]
                       for e, lft, rgt in self.res.rows], footnotes=[self.res.message]))
        return s


def _html_eq(text: str) -> str:
    parts = []
    for tok in text.split(" "):
        parts.append(format_formula(tok, "html") if tok not in ("+", "→") else tok)
    return " ".join(parts)


class StoichiometryDialog(QDialog):
    """ChemDraw-style reaction stoichiometry grid."""

    UNITS = ["mmol", "mg", "g", "equiv"]

    def __init__(self, doc: Document, parent=None, send=None):
        super().__init__(parent)
        self.setWindowTitle("Reaction Stoichiometry")
        self.resize(860, 380)
        self.doc = doc
        self.species, _, note = analysis.stoich_species(doc)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Enter the amount of each reactant (mmol, mass or equivalents "
                             "relative to the first reactant with an amount)."))
        cols = ["Role", "Structure", "Formula", "MW", "Coef.", "Amount", "Unit", "Equiv.",
                "mmol", "Mass (mg)", "Limiting / yield"]
        self.table = _table(cols)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._units = []
        for s in self.species:
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, _ro(s.role))
            self.table.setItem(r, 1, _ro(s.name))
            self.table.setItem(r, 2, _ro(format_formula(s.formula, "unicode")))
            self.table.setItem(r, 3, _ro(f"{s.mw:.2f}"))
            self.table.setItem(r, 4, _ro(str(s.coefficient)))
            amount = QTableWidgetItem("")
            if s.role != "Reactant":
                amount.setFlags(amount.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(r, 5, amount)
            unit = QComboBox()
            unit.addItems(self.UNITS)
            unit.setEnabled(s.role == "Reactant")
            unit.currentIndexChanged.connect(self.recalc)
            self.table.setCellWidget(r, 6, unit)
            self._units.append(unit)
            for c in range(7, 11):
                self.table.setItem(r, c, _ro(""))
        first = next((i for i, s in enumerate(self.species) if s.role == "Reactant"), None)
        if first is not None:
            self.table.item(first, 5).setText("1.0")
        self.table.itemChanged.connect(lambda _it: self.recalc())
        lay.addWidget(self.table, 1)
        self.notes = QLabel(note)
        self.notes.setWordWrap(True)
        self.notes.setStyleSheet("color: #555;")
        lay.addWidget(self.notes)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(self.reject)
        if send is not None:
            b = bb.addButton("Send to Output Viewer", QDialogButtonBox.ButtonRole.ActionRole)
            b.clicked.connect(lambda: send(self.section()))
        lay.addWidget(bb)
        self.lines = []
        self.recalc()

    def set_amount(self, row: int, value: float, unit: str) -> None:
        self._units[row].setCurrentText(unit)
        self.table.item(row, 5).setText(f"{value:g}")

    def amounts(self) -> dict:
        out = {}
        for r, s in enumerate(self.species):
            if s.role != "Reactant":
                continue
            t = (self.table.item(r, 5).text() or "").strip().replace(",", ".")
            if not t:
                continue
            try:
                out[r] = (float(t), self._units[r].currentText())
            except ValueError:
                continue
        return out

    def recalc(self) -> None:
        self.table.blockSignals(True)
        try:
            self.lines, notes = analysis.stoich_grid(self.doc, self.amounts())
            for r, ln in enumerate(self.lines):
                fmt = (lambda v, d: "" if v is None else f"{v:.{d}f}")
                self.table.item(r, 7).setText(fmt(ln.equiv, 2))
                self.table.item(r, 8).setText(fmt(ln.mmol, 3))
                self.table.item(r, 9).setText(fmt(ln.mass_mg, 1))
                self.table.item(r, 10).setText("limiting reagent" if ln.limiting else ln.note)
            self.notes.setText(" ".join(notes))
        except ValueError as exc:
            self.lines = []
            self.notes.setText(str(exc))
        finally:
            self.table.blockSignals(False)

    def section(self):
        from rsstatchem.core.output import Section, Table
        s = Section("Reaction Stoichiometry")
        s.log("CHEM STOICHIOMETRY DRAWING " + " ".join(
            f"'{ln.species.name}'" for ln in self.lines) + ".")
        rows = []
        for ln in self.lines:
            sp = ln.species
            f = (lambda v, d: "" if v is None else f"{v:.{d}f}")
            rows.append([sp.role, format_formula(sp.formula, "unicode"), f"{sp.mw:.2f}",
                         str(sp.coefficient), f(ln.equiv, 2), f(ln.mmol, 3), f(ln.mass_mg, 1),
                         "Yes" if ln.limiting else ("" if sp.role == "Reactant" else
                                                    "theoretical")])
        s.table(Table("Stoichiometry", ["Role", "Formula", "MW (g/mol)", "Coef.", "Equiv.",
                                        "mmol", "Mass (mg)", "Limiting"], rows,
                      footnotes=[self.notes.text()] if self.notes.text() else []))
        return s


class DataInsertDialog(QDialog):
    """Write a SMILES into a string variable of the active Data Editor row."""

    def __init__(self, ds, smiles: str, row: int = 0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Insert SMILES into Data Editor")
        self.ds = ds
        form = QFormLayout()
        self.var = QComboBox()
        strings = [v.name for v in ds.variables if v.type == "string"]
        self.var.addItems(strings + ["(new variable)"])
        form.addRow("Variable:", self.var)
        self.new_name = QLineEdit("SMILES")
        form.addRow("New variable name:", self.new_name)
        self.row = QSpinBox()
        self.row.setRange(1, max(1, ds.n_cases + 1))
        self.row.setValue(row + 1)
        form.addRow("Case (row):", self.row)
        self.value = QLineEdit(smiles)
        form.addRow("SMILES:", self.value)
        self.var.currentIndexChanged.connect(self._toggle)
        self._toggle()
        lay = QVBoxLayout(self)
        box = QGroupBox("Target")
        box.setLayout(form)
        lay.addWidget(box)
        lay.addWidget(_buttons(self, "Insert"))

    def _toggle(self) -> None:
        self.new_name.setEnabled(self.var.currentText() == "(new variable)")

    def target(self) -> tuple[str, bool, int, str]:
        new = self.var.currentText() == "(new variable)"
        name = self.new_name.text().strip() if new else self.var.currentText()
        return name, new, self.row.value() - 1, self.value.text().strip()

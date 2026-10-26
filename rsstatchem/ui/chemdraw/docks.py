"""ChemDraw's floating *Analysis* and *Chemical Properties* windows for the selection (or the
whole document when nothing is selected), each with a check box per value and a **Paste**
button that puts the checked values into the document as a caption under the structure."""
from __future__ import annotations

import html
import math

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QCheckBox, QGridLayout, QHBoxLayout, QLabel, QPushButton,
                               QSpinBox, QVBoxLayout, QWidget)

from rsstatchem.core.chem import molecule as mo
from rsstatchem.core.chem_draw import analysis, convert
from rsstatchem.core.chem_draw.properties import chemical_properties

__all__ = ["AnalysisWindow", "ChemicalPropertiesWindow", "ANALYSIS_ROWS"]

ANALYSIS_ROWS = ("Chemical Formula", "Exact Mass", "Molecular Weight", "m/z",
                 "Elemental Analysis")


def _num(x, d) -> str:
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


class _ValueWindow(QWidget):
    """Common frame: scope line, rows of (check box, value), Paste / Copy buttons."""

    paste_requested = Signal(str)

    def __init__(self, canvas, names, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.scope = QLabel()
        self.scope.setObjectName("chemdraw_scope")
        self.checks: dict[str, QCheckBox] = {}
        self.labels: dict[str, QLabel] = {}
        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(8)
        self.grid.setVerticalSpacing(3)
        for r, name in enumerate(names):
            self._add_row(r, name)
        self.grid.setColumnStretch(1, 1)
        self.paste_btn = QPushButton("Paste")
        self.paste_btn.setToolTip("Paste the checked values into the document as a caption")
        self.paste_btn.clicked.connect(lambda: self.paste_requested.emit(self.text()))
        self.copy_btn = QPushButton("Copy")
        self.copy_btn.setToolTip("Copy the checked values as text")
        self.copy_btn.clicked.connect(self.copy)
        self.buttons = QHBoxLayout()
        self.buttons.addStretch(1)
        self.buttons.addWidget(self.copy_btn)
        self.buttons.addWidget(self.paste_btn)
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setObjectName("chemdraw_note")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 8)
        lay.setSpacing(6)
        lay.addWidget(self.scope)
        lay.addLayout(self.grid)
        lay.addWidget(self.note)
        lay.addStretch(1)
        lay.addLayout(self.buttons)
        self._plain: dict[str, str] = {}
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self.refresh)
        canvas.doc_changed.connect(self._timer.start)
        canvas.selection_changed.connect(self._timer.start)

    def _add_row(self, r: int, name: str) -> None:
        cb = QCheckBox(name + ":")
        cb.setChecked(True)
        val = QLabel()
        val.setTextFormat(Qt.TextFormat.RichText)
        val.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        val.setWordWrap(True)
        val.setObjectName("chemdraw_value")
        self.grid.addWidget(cb, r, 0, Qt.AlignmentFlag.AlignTop)
        self.grid.addWidget(val, r, 1)
        self.checks[name], self.labels[name] = cb, val

    def scope_atoms(self):
        atoms = self.canvas.selected_atoms()
        self.scope.setText("Selection" if atoms else "Whole document")
        return atoms or None

    def showEvent(self, e) -> None:  # noqa: N802
        super().showEvent(e)
        self._timer.start()

    def values(self) -> dict[str, str]:
        """Plain-text values (what Paste and Copy use)."""
        return dict(self._plain)

    def text(self) -> str:
        return "\n".join(f"{k}: {v}" for k, v in self._plain.items()
                         if v and k in self.checks and self.checks[k].isChecked())

    def copy(self) -> None:
        from PySide6.QtGui import QGuiApplication
        QGuiApplication.clipboard().setText(self.text() + "\n")

    def set_value(self, name: str, plain: str, rich: str | None = None) -> None:
        self._plain[name] = plain
        self.labels[name].setText(rich if rich is not None else html.escape(plain))


class AnalysisWindow(_ValueWindow):
    """Chemical Formula, Exact Mass, Molecular Weight, m/z and Elemental Analysis."""

    def __init__(self, canvas, parent=None):
        super().__init__(canvas, ANALYSIS_ROWS, parent)
        self.setObjectName("chemdraw_analysis")
        self.decimals = QSpinBox()
        self.decimals.setRange(0, 6)
        self.decimals.setValue(2)
        self.decimals.setPrefix("Decimals: ")
        self.decimals.valueChanged.connect(self.refresh)
        self.buttons.insertWidget(0, self.decimals)
        self.result = analysis.Analysis()
        self.refresh()

    def refresh(self) -> None:
        atoms = self.scope_atoms()
        try:
            self.result = analysis.analyze(self.canvas.doc, atoms)
        except Exception as exc:  # noqa: BLE001 - the window must never break drawing
            self.result = analysis.Analysis(message=str(exc))
        res, d = self.result, self.decimals.value()
        if not res.ok:
            for name in ANALYSIS_ROWS:
                self.set_value(name, "")
            self.note.setText(res.message if res.message != "Nothing drawn" else "")
            return
        self.set_value("Chemical Formula", res.formula, res.formula_html)
        self.set_value("Exact Mass", _num(res.exact_mass, d))
        self.set_value("Molecular Weight", _num(res.mw, d))
        mz = ", ".join(f"{m:.{d}f} ({a:.1f}%)" for m, a in res.mz)
        self.set_value("m/z", mz)
        ea = "; ".join(f"{s}, {p:.{d}f}" for s, p in res.elemental)
        self.set_value("Elemental Analysis", ea)
        self.note.setText(f"{res.n_molecules} molecules (combined values)"
                          if res.n_molecules > 1 else "")


class ChemicalPropertiesWindow(_ValueWindow):
    """Estimated properties of one molecule (Joback, Crippen, Ertl, ESOL; see
    :mod:`rsstatchem.core.chem_draw.properties`)."""

    NAMES = ("Boiling Point", "Melting Point", "Critical Temp.", "Critical Pres.",
             "Critical Vol.", "Gibbs Energy", "Heat of Form.", "LogP", "MR", "tPSA", "LogS")

    def __init__(self, canvas, parent=None):
        super().__init__(canvas, self.NAMES, parent)
        self.setObjectName("chemdraw_properties")
        self.rows = []
        self.refresh()

    def refresh(self) -> None:
        atoms = self.scope_atoms()
        doc = self.canvas.doc
        mol, msg = None, ""
        if doc.atoms:
            try:
                mol = convert.to_mol(doc, atoms)
            except mo.ChemError as exc:
                msg = str(exc)
        if mol is not None:
            from rdkit import Chem
            if len(Chem.GetMolFrags(mol)) > 1:
                mol, msg = None, "Select one molecule to see its properties."
        self.rows = chemical_properties(mol)
        by_name = {p.name: p for p in self.rows}
        missing: dict = {}
        used = set()
        for name in self.NAMES:
            p = by_name.get(name)
            if p is None:
                self.set_value(name, "")
                self.labels[name].setToolTip("")
                continue
            text = p.text()
            if name == "Boiling Point" and p.value is not None:
                text += f" ({p.value - 273.15:.2f} °C)"
            rich = html.escape(text)
            if p.short and p.value is not None:
                text += f" [{p.short}]"
                sep = "<br>" if len(rich) > 16 else " "     # long values: tag on its own line
                rich += f"{sep}<span style='color:#7A7A7A;'>[{html.escape(p.short)}]</span>"
            self.set_value(name, text, rich)
            self.labels[name].setToolTip(p.method + (f"\nNot available: {p.note}"
                                                     if p.note else ""))
            self.checks[name].setToolTip(p.method)
            if p.note:
                missing.setdefault(p.note, []).append(name)
            elif p.short and name not in ("LogP", "MR", "tPSA", "LogS"):
                used.add(p.short)
        notes = []
        if msg:
            notes.append(msg)
        if self.rows:
            gc = ", ".join(m for m in ("Joback", "Stein-Brown", "Marrero-Gani") if m in used)
            notes.append("All values are estimates (method in brackets): boiling/melting "
                         f"point, critical data and energies by group contribution "
                         f"({gc or 'Joback'}; Joback first, published fallbacks when it "
                         "lacks a group); LogP and MR by Crippen (RDKit); tPSA by Ertl; "
                         "LogS by ESOL. Not ChemDraw's CLogP/CMR.")
        if missing:
            for why, props in missing.items():
                notes.append(f"n/a ({', '.join(props)}): {why}.")
        self.note.setText("\n".join(notes))

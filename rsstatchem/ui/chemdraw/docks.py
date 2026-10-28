"""ChemDraw's floating *Analysis* and *Chemical Properties* windows for the selection (or the
whole document when nothing is selected), each with a check box per value and a **Paste**
button that puts the checked values into the document as a caption under the structure."""
from __future__ import annotations

import html
import math
import threading

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (QCheckBox, QGridLayout, QHBoxLayout, QLabel, QPushButton,
                               QSpinBox, QTextBrowser, QVBoxLayout, QWidget)

from rsstatchem.core.chem import molecule as mo
from rsstatchem.core.chem_draw import analysis, convert
from rsstatchem.core.chem_draw.properties import NOT_BIOBYTE, chemical_properties

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


class _CdkBridge(QObject):
    """Carries CDK results from the worker thread to the window (queued signal)."""

    done = Signal(str, object, str)          # SMILES, values (or None), state / error


class ChemicalPropertiesWindow(_ValueWindow):
    """Properties of one molecule: experimental values from the offline database first, else
    the estimate with the lowest benchmark error (see
    :mod:`rsstatchem.core.chem_draw.properties`). Each value's tooltip lists every method with
    its error on the open data; **Details** shows the same as a table. CDK logP methods run in
    a background thread (one persistent Java helper) and fill in when ready."""

    NAMES = ("Boiling Point", "Melting Point", "Critical Temp.", "Critical Pres.",
             "Critical Vol.", "Gibbs Energy", "Heat of Form.", "LogP", "MR", "tPSA", "LogS",
             "Vapor Pres.", "pKa (acidic)", "pKa (basic)", "Henry's Law")
    WARN = "#B05A00"
    GREY = "#7A7A7A"

    def __init__(self, canvas, parent=None):
        super().__init__(canvas, self.NAMES, parent)
        self.setObjectName("chemdraw_properties")
        self.rows = []
        self.use_cdk = True
        self._mol = None
        self._smiles = ""
        self._cdk: dict[str, tuple] = {}         # SMILES -> (values or None, state)
        self._pending: set[str] = set()
        self._bridge = _CdkBridge(self)
        self._bridge.done.connect(self._cdk_done)
        self.details_btn = QPushButton("Details")
        self.details_btn.setCheckable(True)
        self.details_btn.setToolTip("Every method's value and its error on open experimental "
                                    "data")
        self.details_btn.toggled.connect(self._toggle_details)
        self.buttons.insertWidget(0, self.details_btn)
        self.details = QTextBrowser()
        self.details.setObjectName("chemdraw_details")
        self.details.setOpenExternalLinks(False)
        self.details.setMinimumHeight(180)
        self.details.setVisible(False)
        lay = self.layout()
        lay.insertWidget(lay.indexOf(self.note) + 1, self.details, 10)
        self.refresh()

    # -- CDK in the background ---------------------------------------------------------------
    def _cdk_for(self, smiles: str):
        """(values, state): cached CDK values, or None with 'computing...' / why not."""
        from rsstatchem.core.chem_draw import cdk
        if not self.use_cdk or not smiles:
            return None, ""
        if smiles in self._cdk:
            return self._cdk[smiles]
        if not cdk.cdk_available():
            state = "not available (CDK jar or Java missing)"
            self._cdk[smiles] = (None, state)
            return None, state
        if smiles not in self._pending:
            self._pending.add(smiles)
            bridge = self._bridge
            threading.Thread(target=self._cdk_worker, args=(bridge, smiles), daemon=True,
                             name="cdk-props").start()
        return None, "computing..."

    @staticmethod
    def _cdk_worker(bridge, smiles: str) -> None:
        from rsstatchem.core.chem_draw import cdk
        try:
            values, state = cdk.cdk_values(smiles), ""
        except cdk.CdkError as exc:
            values, state = None, str(exc)
        except Exception as exc:  # noqa: BLE001 - the worker must never raise
            values, state = None, f"failed ({exc})"
        try:
            bridge.done.emit(smiles, values, state)
        except RuntimeError:                       # window already gone
            pass

    def _cdk_done(self, smiles: str, values, state: str) -> None:
        self._pending.discard(smiles)
        self._cdk[smiles] = (values, state)
        if smiles == self._smiles and self._mol is not None:
            self._render(self._mol)

    def wait_cdk(self, timeout: float = 60.0) -> bool:
        """Block (processing events) until the CDK values of the current molecule arrived."""
        import time

        from PySide6.QtCore import QCoreApplication
        end = time.monotonic() + timeout
        while self._smiles in self._pending and time.monotonic() < end:
            QCoreApplication.processEvents()
            time.sleep(0.01)
        QCoreApplication.processEvents()
        return self._smiles in self._cdk and self._cdk[self._smiles][0] is not None

    # -- refresh -----------------------------------------------------------------------------
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
        self._msg = msg
        self._mol = mol
        self._render(mol)

    def _render(self, mol) -> None:
        smiles = ""
        if mol is not None:
            try:
                from rdkit import Chem
                kek = Chem.Mol(mol)
                Chem.Kekulize(kek, clearAromaticFlags=True)
                smiles = Chem.MolToSmiles(kek, kekuleSmiles=True)
            except Exception:  # noqa: BLE001
                smiles = ""
        self._smiles = smiles
        values, state = self._cdk_for(smiles) if mol is not None else (None, "")
        try:
            self.rows = chemical_properties(mol, values, state)
        except Exception as exc:  # noqa: BLE001 - the window must never break drawing
            self.rows = []
            self._msg = f"Properties failed: {exc}"
        by_name = {p.name: p for p in self.rows}
        missing: dict = {}
        kinds = set()
        for name in self.NAMES:
            p = by_name.get(name)
            if p is None:
                self.set_value(name, "")
                self.labels[name].setToolTip("")
                continue
            text = p.text()
            if p.key in ("bp", "mp") and p.value is not None:
                text += f" ({p.value - 273.15:.1f} °C)"
            rich = html.escape(text)
            tag = p.tag()
            if tag:
                text += f" {tag}"
                color = self.WARN if p.warn else self.GREY
                sep = "<br>" if len(rich) + len(tag) > 30 else " "
                rich += f"{sep}<span style='color:{color};'>{html.escape(tag)}</span>"
            self.set_value(name, text, rich)
            self.labels[name].setToolTip(self._tooltip(p))
            self.checks[name].setToolTip(p.method)
            if p.note and p.value is None:
                missing.setdefault(p.note, []).append(name)
            if p.value is not None:
                kinds.add(p.kind)
        notes = []
        if self._msg:
            notes.append(self._msg)
        if self.rows:
            notes.append("Experimental values (open databases, sources in brackets) are shown "
                         "first; otherwise the estimate with the lowest error on open "
                         "experimental data. Hover a value or press Details for every method "
                         "and its error. LogP/MR: " + NOT_BIOBYTE + ". pKa: open model "
                         "(Mansouri et al. 2019 data), not MOSES; pKa (basic) is the pKa of "
                         "the conjugate acid.")
        for why, props in missing.items():
            notes.append(f"n/a ({', '.join(props)}): {why}.")
        self.note.setText("\n".join(notes))
        if self.details.isVisible():
            self.details.setHtml(self.details_html())

    # -- tooltip and details -----------------------------------------------------------------
    def _tooltip(self, p) -> str:
        head = f"<b>{html.escape(p.name)}</b> - {html.escape(p.method)}"
        if p.note and p.value is None:
            head += f"<br>Not available: {html.escape(p.note)}"
        if not p.alternatives:
            return head
        return head + self._table(p.alternatives)

    @staticmethod
    def _table(alts) -> str:
        cells = ["<table cellspacing='0' cellpadding='2'>"
                 "<tr><th align='left'>Method</th><th align='left'>Value</th>"
                 "<th align='left'>Error on open data</th><th align='left'>Note</th></tr>"]
        for a in alts:
            mark = "&#9654; " if a.default else ""
            cells.append(f"<tr><td>{mark}{html.escape(a.label)}</td>"
                         f"<td>{html.escape(a.text)}</td>"
                         f"<td>{html.escape(a.benchmark)}</td>"
                         f"<td>{html.escape(a.note)}</td></tr>")
        cells.append("</table>")
        return "".join(cells)

    def details_html(self) -> str:
        parts = ["<p style='color:#555'>&#9654; = value shown. Errors are RMSE / MAE against "
                 "the open experimental data of the property database (n compounds; ML and "
                 "consensus values 5-fold cross-validated, the published methods on data that "
                 "may overlap their own fitting sets). " + html.escape(NOT_BIOBYTE) + ".</p>"]
        for p in self.rows:
            if not p.alternatives:
                continue
            parts.append(f"<p><b>{html.escape(p.name)}</b>: {html.escape(p.text())} "
                         f"<span style='color:#777'>{html.escape(p.tag())}</span></p>")
            parts.append(self._table(p.alternatives))
        return "".join(parts)

    def _toggle_details(self, on: bool) -> None:
        self.details.setVisible(on)
        if on:
            self.details.setHtml(self.details_html())

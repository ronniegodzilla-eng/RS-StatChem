"""Dock panels of the structure editor: ChemDraw's Analysis window (live) and the template
library."""
from __future__ import annotations

import html
import math

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                               QPushButton, QTextBrowser, QVBoxLayout, QWidget)

from rsstatchem.core.chem_draw import analysis, templates
from rsstatchem.core.chem_draw.model import Document
from rsstatchem.core.chem_draw.style import preset

__all__ = ["AnalysisPanel", "TemplatePanel", "analysis_html", "analysis_text"]


def _f(x, d) -> str:
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def analysis_rows(res: analysis.Analysis) -> list[tuple[str, str, str]]:
    """(label, html value, plain value) rows of the Analysis window."""
    if not res.ok:
        return []
    mz = ", ".join(f"{m:.4f} ({a:.1f}%)" for m, a in res.mz)
    ea = "; ".join(f"{s}, {p:.2f}" for s, p in res.elemental)
    return [
        ("Chemical Formula", res.formula_html, res.formula),
        ("Exact Mass", _f(res.exact_mass, 4), _f(res.exact_mass, 4)),
        ("Molecular Weight", _f(res.mw, 4), _f(res.mw, 4)),
        ("m/z", mz, mz),
        ("Elemental Analysis", ea, ea),
        ("cLogP", _f(res.clogp, 2), _f(res.clogp, 2)),
        ("TPSA", _f(res.tpsa, 2) + " Å²", _f(res.tpsa, 2) + " Å²"),
        ("H-bond donors / acceptors", f"{res.hbd} / {res.hba}", f"{res.hbd} / {res.hba}"),
        ("Rotatable bonds", str(res.rotb), str(res.rotb)),
        ("Stereocentres", str(res.stereo), str(res.stereo)),
        ("Lipinski (Ro5)", res.lipinski, res.lipinski),
        ("SMILES", html.escape(res.smiles), res.smiles),
        ("InChIKey", res.inchikey, res.inchikey),
    ]


def analysis_html(res: analysis.Analysis, scope: str, problems: list[str]) -> str:
    out = [f"<p style='color:#666;margin:0 0 4px 0'>{html.escape(scope)}</p>"]
    if not res.ok:
        out.append(f"<p style='color:#B71C1C'>{html.escape(res.message)}</p>")
    else:
        out.append("<table cellspacing='0' cellpadding='2'>")
        for label, value, _ in analysis_rows(res):
            out.append(f"<tr><td style='color:#555;padding-right:8px;white-space:nowrap' "
                       f"valign='top'>{label}:</td><td>{value}</td></tr>")
        out.append("</table>")
        if res.n_molecules > 1:
            out.append(f"<p style='color:#666'>{res.n_molecules} molecules (values for the "
                       "combined set).</p>")
    if problems:
        out.append("<p style='color:#B71C1C'><b>Structure problems:</b><br>" +
                   "<br>".join(html.escape(p) for p in problems[:8]) + "</p>")
    return "".join(out)


def analysis_text(res: analysis.Analysis) -> str:
    return "\n".join(f"{label}: {plain}" for label, _, plain in analysis_rows(res)[:7])


class AnalysisPanel(QWidget):
    """Live ChemDraw-style Analysis of the selection (or the whole drawing)."""

    paste_requested = Signal(str)

    def __init__(self, canvas, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.browser = QTextBrowser()
        self.browser.setOpenLinks(False)
        self.browser.setMinimumWidth(250)
        self.result = analysis.Analysis()
        copy = QPushButton("Copy")
        copy.setToolTip("Copy the analysis as text")
        copy.clicked.connect(self.copy)
        paste = QPushButton("Paste into Drawing")
        paste.setToolTip("Insert formula, masses and elemental analysis as a text box")
        paste.clicked.connect(lambda: self.paste_requested.emit(analysis_text(self.result)))
        row = QHBoxLayout()
        row.addWidget(copy)
        row.addWidget(paste)
        row.addStretch(1)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(self.browser, 1)
        lay.addLayout(row)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self.refresh)
        canvas.doc_changed.connect(self._timer.start)
        canvas.selection_changed.connect(self._timer.start)
        self.refresh()

    def refresh(self) -> None:
        doc = self.canvas.doc
        atoms = self.canvas.selected_atoms()
        scope = "Selection" if atoms else "Whole drawing"
        try:
            self.result = analysis.analyze(doc, atoms or None)
        except Exception as exc:  # noqa: BLE001 - panel must never break the editor
            self.result = analysis.Analysis(message=str(exc))
        probs = sorted(set(self.canvas.problems.values()))
        self.browser.setHtml(analysis_html(self.result, scope, probs))

    def copy(self) -> None:
        from PySide6.QtGui import QGuiApplication
        QGuiApplication.clipboard().setText(analysis_text(self.result) + "\n")


class TemplatePanel(QWidget):
    """Template library: pick a template, then click on the canvas to place it."""

    template_chosen = Signal(str, str)      # name, smiles
    insert_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.category = QComboBox()
        self.category.addItems(templates.categories())
        self.list = QListWidget()
        self.list.setViewMode(QListWidget.ViewMode.IconMode)
        self.list.setIconSize(QSize(84, 64))
        self.list.setGridSize(QSize(100, 92))
        self.list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.list.setMovement(QListWidget.Movement.Static)
        self.list.setWordWrap(True)
        self.list.itemClicked.connect(self._chosen)
        self.list.itemDoubleClicked.connect(
            lambda it: self.insert_requested.emit(it.data(Qt.ItemDataRole.UserRole)))
        self.category.currentIndexChanged.connect(self._fill)
        hint = QLabel("Click a template, then click in the drawing (double-click inserts at "
                      "the centre). * = attachment point R.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(self.category)
        lay.addWidget(self.list, 1)
        lay.addWidget(hint)
        self._icons: dict[str, QIcon] = {}
        self._fill()

    def _icon(self, smiles: str) -> QIcon:
        if smiles not in self._icons:
            from rsstatchem.core.chem_draw import convert
            from rsstatchem.ui.chem_editor.export import image
            doc = Document(preset("ACS Document 1996"))
            doc.style.font_size = 9.0
            try:
                if templates.is_drawn(smiles):
                    templates.insert_drawn(doc, smiles, 0, 0)
                else:
                    convert.insert_text(doc, smiles, 0, 0)
                img = image(doc, dpi=110, transparent=True)
                pm = QPixmap.fromImage(img).scaled(
                    84, 64, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation)
                self._icons[smiles] = QIcon(pm)
            except Exception:  # noqa: BLE001
                self._icons[smiles] = QIcon()
        return self._icons[smiles]

    def _fill(self) -> None:
        self.list.clear()
        for name, smi in templates.TEMPLATES[self.category.currentText()]:
            it = QListWidgetItem(self._icon(smi), name)
            it.setData(Qt.ItemDataRole.UserRole, smi)
            it.setToolTip(name if templates.is_drawn(smi) else f"{name}\n{smi}")
            self.list.addItem(it)

    def _chosen(self, it: QListWidgetItem) -> None:
        self.template_chosen.emit(it.text(), it.data(Qt.ItemDataRole.UserRole))

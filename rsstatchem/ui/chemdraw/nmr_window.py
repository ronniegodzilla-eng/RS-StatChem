"""ChemDraw-style NMR window (Structure > Predict ¹H-NMR / ¹³C-NMR Shifts): the predicted
spectrum of the structure on the canvas (first-order ¹H multiplets at the chosen spectrometer
frequency), the peak list, assignment highlighting in both directions (select a peak -> its
atoms are selected on the canvas; select atoms -> their peaks are marked) and a measured
JCAMP-DX spectrum as an overlay. Prediction: :mod:`rsstatchem.core.chem_draw.nmr_predict`."""
from __future__ import annotations

import html

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QFileDialog,
                               QHBoxLayout, QHeaderView, QLabel, QPushButton, QSplitter,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from rsstatchem.core.chem import molecule as mo
from rsstatchem.core.chem_draw import convert, jcamp
from rsstatchem.core.chem_draw import nmr_predict as npd

__all__ = ["NmrWindow", "SpectrumView", "FREQUENCIES", "SOLVENTS"]

FREQUENCIES = (60, 90, 300, 400, 500, 600, 800)
SOLVENTS = ("CDCl3", "DMSO-d6", "D2O", "CD3OD", "acetone-d6")
PRED = QColor("#1f4e9a")
MEAS = QColor("#c0392b")
MARK = QColor("#f2b400")


class SpectrumView(QWidget):
    """Plot of one spectrum (ppm axis decreasing to the right), optional overlay, peak marks."""

    peak_clicked = Signal(int)          # index into peaks

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(520, 240)
        self.x = np.zeros(0)
        self.y = np.zeros(0)
        self.overlay: jcamp.JcampSpectrum | None = None
        self.peaks: list = []
        self.marked: set[int] = set()
        self.solvent_ppm: float | None = None
        self.nucleus = "1H"
        self.lo, self.hi = -0.5, 12.5
        self.setMouseTracking(True)

    def set_data(self, x, y, peaks, nucleus: str, lo: float, hi: float) -> None:
        self.x, self.y, self.peaks, self.nucleus = x, y, peaks, nucleus
        self.lo, self.hi = lo, hi
        self.update()

    def _plot_rect(self) -> QRectF:
        return QRectF(12, 10, max(10, self.width() - 24), max(10, self.height() - 44))

    def _to_px(self, ppm: float, r: QRectF) -> float:
        return r.left() + (self.hi - ppm) / (self.hi - self.lo) * r.width()

    def _to_ppm(self, px: float, r: QRectF) -> float:
        return self.hi - (px - r.left()) / r.width() * (self.hi - self.lo)

    def wheelEvent(self, e) -> None:  # noqa: N802
        r = self._plot_rect()
        c = self._to_ppm(e.position().x(), r)
        f = 0.8 if e.angleDelta().y() > 0 else 1.25
        lo, hi = c - (c - self.lo) * f, c + (self.hi - c) * f
        if hi - lo > 0.05:
            self.lo, self.hi = lo, hi
            self.update()

    def mouseDoubleClickEvent(self, e) -> None:  # noqa: N802
        self.lo, self.hi = (-0.5, 12.5) if self.nucleus == "1H" else (-5.0, 225.0)
        self.update()

    def mousePressEvent(self, e) -> None:  # noqa: N802
        if not self.peaks:
            return
        r = self._plot_rect()
        best, dist = None, 12.0
        for i, p in enumerate(self.peaks):
            if p.shift is None:
                continue
            d = abs(self._to_px(p.shift, r) - e.position().x())
            if d < dist:
                best, dist = i, d
        if best is not None:
            self.peak_clicked.emit(best)

    def paintEvent(self, e) -> None:  # noqa: N802
        qp = QPainter(self)
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        qp.fillRect(self.rect(), QColor("white"))
        r = self._plot_rect()
        qp.setPen(QPen(QColor("#888888"), 1))
        qp.drawLine(QPointF(r.left(), r.bottom()), QPointF(r.right(), r.bottom()))
        # axis ticks
        span = self.hi - self.lo
        step = next(s for s in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50) if span / s <= 14)
        f = QFont(self.font())
        f.setPointSizeF(max(7.0, f.pointSizeF() - 1))
        qp.setFont(f)
        t = np.ceil(self.lo / step) * step
        while t <= self.hi + 1e-9:
            px = self._to_px(t, r)
            qp.drawLine(QPointF(px, r.bottom()), QPointF(px, r.bottom() + 4))
            qp.drawText(QRectF(px - 20, r.bottom() + 5, 40, 14), Qt.AlignmentFlag.AlignCenter,
                        f"{round(float(t), 6) + 0.0:g}")
            t += step
        qp.drawText(QRectF(r.left(), r.bottom() + 18, r.width(), 14),
                    Qt.AlignmentFlag.AlignCenter, f"δ (ppm), {self.nucleus}")
        if self.solvent_ppm is not None and self.lo < self.solvent_ppm < self.hi:
            px = self._to_px(self.solvent_ppm, r)
            qp.setPen(QPen(QColor("#999999"), 1, Qt.PenStyle.DashLine))
            qp.drawLine(QPointF(px, r.top()), QPointF(px, r.bottom()))
        # marked peaks
        for i in self.marked:
            if i < len(self.peaks) and self.peaks[i].shift is not None:
                px = self._to_px(self.peaks[i].shift, r)
                qp.fillRect(QRectF(px - 5, r.top(), 10, r.height()), QColor(242, 180, 0, 70))
        self._curve(qp, r, self.x, self.y, PRED)
        if self.overlay is not None:
            ov = self.overlay
            if ov.kind == "peaks":
                ymax = float(np.max(np.abs(ov.y))) or 1.0
                qp.setPen(QPen(MEAS, 1.2))
                for xx, yy in zip(ov.x, ov.y):
                    if self.lo <= xx <= self.hi:
                        px = self._to_px(float(xx), r)
                        qp.drawLine(QPointF(px, r.bottom()),
                                    QPointF(px, r.bottom() - 0.9 * r.height() * yy / ymax))
            else:
                self._curve(qp, r, ov.x, ov.y, MEAS)
        qp.end()

    def _curve(self, qp, r: QRectF, x, y, color) -> None:
        if len(x) < 2:
            return
        sel = (x >= self.lo) & (x <= self.hi)
        if sel.sum() < 2:
            return
        xs, ys = x[sel], y[sel]
        ymax = float(np.max(ys)) or 1.0
        path = QPainterPath()
        # decimate to the pixel width
        n = len(xs)
        stride = max(1, n // int(max(200, r.width() * 2)))
        first = True
        for k in range(0, n, stride):
            seg = ys[k:k + stride]
            v = float(seg.max()) if len(seg) else float(ys[k])
            pt = QPointF(self._to_px(float(xs[k]), r), r.bottom() - 0.9 * r.height() * v / ymax)
            if first:
                path.moveTo(pt)
                first = False
            else:
                path.lineTo(pt)
        qp.setPen(QPen(color, 1.2))
        qp.drawPath(path)


class NmrWindow(QDialog):
    """Predicted ¹H / ¹³C spectrum of the structure (selection's molecule, or the drawing)."""

    def __init__(self, canvas, nucleus: str = "1H", parent=None):
        super().__init__(parent)
        self.setObjectName("chemdraw_nmr")
        self.setWindowTitle("NMR Prediction")
        self.canvas = canvas
        self.peaks: list = []
        self.mol = None
        self.smiles = ""
        self.origin: dict[int, int] = {}       # mol atom index -> document atom id
        self._sync = False
        self.nucleus_box = QComboBox()
        self.nucleus_box.addItems(["1H", "13C"])
        self.nucleus_box.setCurrentText(npd.normalise(nucleus))
        self.freq_box = QComboBox()
        self.freq_box.addItems([f"{f} MHz" for f in FREQUENCIES])
        self.freq_box.setCurrentText("300 MHz")
        self.solvent_box = QComboBox()
        self.solvent_box.addItems(SOLVENTS)
        self.open_btn = QPushButton("Open JCAMP-DX...")
        self.open_btn.setToolTip("Overlay a measured spectrum (.jdx, .dx)")
        self.clear_btn = QPushButton("Clear Overlay")
        self.export_btn = QPushButton("Export JCAMP-DX...")
        self.copy_btn = QPushButton("Copy Peak List")
        top = QHBoxLayout()
        for w in (QLabel("Nucleus:"), self.nucleus_box, QLabel("Frequency:"), self.freq_box,
                  QLabel("Solvent:"), self.solvent_box):
            top.addWidget(w)
        top.addStretch(1)
        for w in (self.open_btn, self.clear_btn, self.export_btn, self.copy_btn):
            top.addWidget(w)
        self.view = SpectrumView()
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["δ (ppm)", "Mult.", "J (Hz)", "Nuclei", "Atoms",
                                              "Basis"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setObjectName("chemdraw_note")
        self.note.setTextFormat(Qt.TextFormat.RichText)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.view)
        split.addWidget(self.table)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(split, 1)
        lay.addWidget(self.note)
        self.resize(900, 640)
        self.nucleus_box.currentTextChanged.connect(lambda _t: self.refresh(force=True))
        self.freq_box.currentTextChanged.connect(lambda _t: self.refresh(force=True))
        self.solvent_box.currentTextChanged.connect(lambda _t: self.refresh(force=True))
        self.open_btn.clicked.connect(lambda: self.open_spectrum())
        self.clear_btn.clicked.connect(self.clear_overlay)
        self.export_btn.clicked.connect(lambda: self.export_spectrum())
        self.copy_btn.clicked.connect(self.copy_peaks)
        self.table.itemSelectionChanged.connect(self._rows_selected)
        self.view.peak_clicked.connect(self.select_peak)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self.refresh)
        canvas.doc_changed.connect(self._timer.start)
        canvas.selection_changed.connect(self._canvas_selection)
        self.refresh(force=True)

    # -- properties ------------------------------------------------------------------------
    @property
    def nucleus(self) -> str:
        return self.nucleus_box.currentText()

    @property
    def mhz(self) -> float:
        return float(self.freq_box.currentText().split()[0])

    # -- structure and prediction ------------------------------------------------------------
    def _structure(self):
        doc = self.canvas.doc
        if not doc.atoms:
            return None, {}, "Draw a structure to predict its spectrum."
        sel = self.canvas.selected_atoms()
        frags = doc.fragments()
        if sel:
            frags = [f for f in frags if sel.intersection(f)]
        if len(frags) != 1:
            return None, {}, "Select one molecule (the drawing holds several)."
        try:
            mol = convert.to_mol(doc, frags[0])
        except mo.ChemError as exc:
            return None, {}, f"Invalid structure: {exc}"
        origin = {a.GetIdx(): a.GetIntProp("doc_id") for a in mol.GetAtoms()
                  if a.HasProp("doc_id")}
        return mol, origin, ""

    def refresh(self, force: bool = False) -> None:
        from rdkit import Chem
        mol, origin, msg = self._structure()
        smiles = Chem.MolToSmiles(mol) if mol is not None else ""
        if not force and smiles == self.smiles and origin == self.origin:
            return
        self.mol, self.origin, self.smiles = mol, origin, smiles
        self.peaks = npd.predict_detailed(mol, self.nucleus) if mol is not None else []
        lo, hi = (-0.5, 12.5) if self.nucleus == "1H" else (-5.0, 225.0)
        if self.peaks:
            x, y = npd.spectrum(self.peaks, self.mhz, lo, hi, points=16384)
        else:
            x, y = np.zeros(0), np.zeros(0)
        self.view.solvent_ppm = npd.SOLVENT_PEAKS.get(self.solvent_box.currentText(),
                                                      {}).get(self.nucleus)
        self.view.marked = set()
        self.view.set_data(x, y, self.peaks, self.nucleus, lo, hi)
        self._fill_table()
        self._set_note(msg)

    def _fill_table(self) -> None:
        self._sync = True
        self.table.setRowCount(len(self.peaks))
        for r, p in enumerate(self.peaks):
            atoms = ", ".join(f"{self._atom_text(a)}" for a in p.atoms)
            if p.radius >= 0:
                basis = f"sphere {p.radius}, {p.n_ref} ref."
                if p.radius <= 1:
                    basis += " (low confidence)"
            else:
                basis = p.note or "no data"
            if p.exchangeable:
                basis += "; exchangeable"
            vals = [f"{p.shift:.2f}" if p.shift is not None else "n/a",
                    "br s" if p.exchangeable else p.multiplicity,
                    ", ".join(f"{j:g}" for j in p.J),
                    f"{p.count}{'H' if self.nucleus == '1H' else 'C'}", atoms, basis]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c == 5 and p.radius <= 1:
                    it.setForeground(QColor("#B05A00"))
                self.table.setItem(r, c, it)
        self.table.resizeColumnsToContents()
        self._sync = False

    def _atom_text(self, idx: int) -> str:
        return f"{self.mol.GetAtomWithIdx(idx).GetSymbol()}{idx + 1}"

    def _set_note(self, msg: str) -> None:
        info = npd.table_info()
        parts = []
        if msg:
            parts.append(html.escape(msg))
        if not info:
            parts.append("The shift table (resources/nmr/hose_shifts.npz) is missing.")
        else:
            b = info.get("benchmark", {}).get(self.nucleus, {})
            err = (f" Held-out error: MAE {b['mae']:.2f} ppm, RMSE {b['rmse']:.2f} ppm "
                   f"({b['n']} shifts of {info.get('n_test', '?')} compounds)."
                   if b.get("mae") is not None else "")
            parts.append(
                f"Predicted from {info.get('n_compounds', '?')} assigned spectra of the BMRB "
                "metabolomics archive (CC0) with HOSE-style atom environments; sphere = how "
                "many bonds of context matched (5 best)." + err +
                " Most reference spectra were recorded in D2O/CDCl3; the solvent choice only "
                "marks the residual solvent signal. First-order multiplets with typical "
                "coupling constants. Not ChemDraw's ChemNMR.")
        if self.view.overlay is not None:
            ov = self.view.overlay
            parts.append(f"<span style='color:{MEAS.name()}'>Overlay</span>: "
                         f"{html.escape(ov.title or 'measured spectrum')}"
                         f"{' (' + ov.nucleus + ')' if ov.nucleus else ''}.")
        self.note.setText("<br>".join(parts))

    # -- assignment highlighting ---------------------------------------------------------------
    def select_peak(self, row: int) -> None:
        self.table.selectRow(row)

    def _rows_selected(self) -> None:
        if self._sync:
            return
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        self.view.marked = set(rows)
        self.view.update()
        keys = set()
        for r in rows:
            for a in self.peaks[r].atoms:
                did = self.origin.get(a)
                if did is not None:
                    keys.add(("atom", did))
        if keys:
            self._sync = True
            try:
                self.canvas.select_keys(keys)
            finally:
                self._sync = False

    def _canvas_selection(self) -> None:
        if self._sync:
            return
        sel = self.canvas.selected_atoms()
        if sel and sel <= set(self.origin.values()):
            # atoms of the current molecule: mark their peaks, keep the prediction
            rows = set(self.peaks_for_atoms(sel))
            self._sync = True
            self.table.clearSelection()
            self._sync = False
            self.view.marked = rows
            self.view.update()
            return
        self._timer.start()

    def peaks_for_atoms(self, doc_ids) -> list[int]:
        ids = set(doc_ids)
        return [r for r, p in enumerate(self.peaks)
                if any(self.origin.get(a) in ids for a in p.atoms)]

    # -- files ------------------------------------------------------------------------------
    def open_spectrum(self, path: str | None = None) -> bool:
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "Open Spectrum", "",
                                                  "JCAMP-DX (*.jdx *.dx *.jcamp);;All files (*)")
            if not path:
                return False
        try:
            sp = jcamp.read_jcamp(path)
        except (OSError, jcamp.JcampError, ValueError) as exc:
            self.note.setText(f"Could not read the spectrum: {html.escape(str(exc))}")
            return False
        if sp.nucleus in ("1H", "13C") and sp.nucleus != self.nucleus:
            self.nucleus_box.setCurrentText(sp.nucleus)
        self.view.overlay = sp
        if sp.x.size:
            lo, hi = float(np.nanmin(sp.x)), float(np.nanmax(sp.x))
            self.view.lo, self.view.hi = min(self.view.lo, lo), max(self.view.hi, hi)
        self.view.update()
        self._set_note("")
        return True

    def clear_overlay(self) -> None:
        self.view.overlay = None
        self.view.update()
        self._set_note("")

    def export_spectrum(self, path: str | None = None) -> bool:
        if not self.peaks:
            return False
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Export Spectrum", "prediction.jdx",
                                                  "JCAMP-DX (*.jdx)")
            if not path:
                return False
        x, y = npd.spectrum(self.peaks, self.mhz, self.view.lo, self.view.hi, points=8192)
        jcamp.write_jcamp(path, x, y, f"Predicted {self.nucleus} spectrum of {self.smiles}",
                          self.nucleus, self.mhz)
        return True

    def peak_list_text(self) -> str:
        """ACS-style peak list: '1H NMR (300 MHz, CDCl3) δ 4.12 (q, J = 7.0 Hz, 2H), ...'."""
        items = []
        for p in self.peaks:
            if p.shift is None:
                continue
            if self.nucleus == "1H":
                mult = "br s" if p.exchangeable else p.multiplicity
                j = f", J = {', '.join(f'{x:.1f}' for x in p.J)} Hz" if p.J else ""
                items.append(f"{p.shift:.2f} ({mult}{j}, {p.count}H)")
            else:
                items.append(f"{p.shift:.1f}")
        head = f"{self.nucleus} NMR ({self.mhz:g} MHz, {self.solvent_box.currentText()}) δ "
        return head + ", ".join(items) + " (predicted)"

    def copy_peaks(self) -> None:
        from PySide6.QtGui import QGuiApplication
        QGuiApplication.clipboard().setText(self.peak_list_text())

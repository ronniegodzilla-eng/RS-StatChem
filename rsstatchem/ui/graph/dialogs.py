"""Export Graph and Page Setup dialogs for the Graph Studio (Origin's "Export Graphs")."""
from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QPushButton, QVBoxLayout)

from rsstatchem.core.graph.export import DPI_CHOICES, FORMATS, RASTER
from rsstatchem.core.graph.model import Graph

PAGE_PRESETS = [("Journal single column (8.3 cm)", 8.3, 6.4),
                ("Journal 1.5 column (12.9 cm)", 12.9, 9.0),
                ("Journal double column (17.4 cm)", 17.4, 9.5),
                ("Origin default (12 × 9 cm)", 12.0, 9.0),
                ("Presentation 4:3 (16 × 12 cm)", 16.0, 12.0),
                ("Presentation 16:9 (24 × 13.5 cm)", 24.0, 13.5)]


class ExportDialog(QDialog):
    def __init__(self, graph: Graph, parent=None, settings: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Export Graph")
        self.graph = graph
        s = settings or {}
        self.fmt = QComboBox()
        for key, text in FORMATS.items():
            self.fmt.addItem(f"{text} (*.{key})", key)
        self.fmt.setCurrentIndex(max(0, self.fmt.findData(s.get("fmt", "png"))))
        self.dpi = QComboBox()
        self.dpi.setEditable(True)
        for d in DPI_CHOICES:
            self.dpi.addItem(str(d))
        self.dpi.setCurrentText(str(s.get("dpi", 600)))
        self.custom = QCheckBox("Custom size (graph is re-laid out; fonts keep their size)")
        self.custom.setChecked(bool(s.get("custom", False)))
        self.unit = QComboBox()
        self.unit.addItems(["cm", "inch"])
        self.unit.setCurrentText(s.get("unit", "cm"))
        self.width = QDoubleSpinBox()
        self.height = QDoubleSpinBox()
        for w in (self.width, self.height):
            w.setRange(0.5, 200)
            w.setDecimals(2)
        self.aspect = QCheckBox("Keep aspect ratio")
        self.aspect.setChecked(True)
        self.transparent = QCheckBox("Transparent background (PNG, TIFF, SVG, PDF)")
        self.transparent.setChecked(bool(s.get("transparent", False)))
        self.info = QLabel()
        self.info.setStyleSheet("color:#555;")

        form = QFormLayout()
        form.addRow("Format:", self.fmt)
        form.addRow("Resolution (DPI):", self.dpi)
        size_box = QGroupBox("Size")
        size_form = QFormLayout(size_box)
        size_form.addRow(self.custom)
        row = QHBoxLayout()
        row.addWidget(self.width)
        row.addWidget(QLabel("×"))
        row.addWidget(self.height)
        row.addWidget(self.unit)
        size_form.addRow("Width × Height:", row)
        size_form.addRow(self.aspect)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Export...")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(size_box)
        lay.addWidget(self.transparent)
        lay.addWidget(self.info)
        lay.addWidget(buttons)

        self._unit = self.unit.currentText()
        self._set_size_from_page()
        self.fmt.currentIndexChanged.connect(self._update)
        self.dpi.currentTextChanged.connect(self._update)
        self.custom.toggled.connect(self._update)
        self.unit.currentTextChanged.connect(self._unit_changed)
        self.width.valueChanged.connect(lambda v: self._linked(True))
        self.height.valueChanged.connect(lambda v: self._linked(False))
        self._update()

    def _factor(self) -> float:
        return 1.0 if self.unit.currentText() == "cm" else 1 / 2.54

    def _set_size_from_page(self) -> None:
        f = self._factor()
        self._busy = True
        self.width.setValue(self.graph.page.width_cm * f)
        self.height.setValue(self.graph.page.height_cm * f)
        self._busy = False

    def _unit_changed(self, unit: str) -> None:
        conv = 1 / 2.54 if unit == "inch" else 2.54
        self._busy = True
        self.width.setValue(self.width.value() * conv)
        self.height.setValue(self.height.value() * conv)
        self._busy = False
        self._update()

    def _linked(self, from_width: bool) -> None:
        if getattr(self, "_busy", False):
            return
        if self.aspect.isChecked():
            ratio = self.graph.page.height_cm / self.graph.page.width_cm
            self._busy = True
            if from_width:
                self.height.setValue(self.width.value() * ratio)
            else:
                self.width.setValue(self.height.value() / ratio)
            self._busy = False
        self._update()

    def _update(self) -> None:
        custom = self.custom.isChecked()
        for w in (self.width, self.height, self.unit, self.aspect):
            w.setEnabled(custom)
        if not custom:
            self._set_size_from_page()
        raster = self.fmt.currentData() in RASTER
        self.dpi.setEnabled(raster or self.fmt.currentData() in ("pdf", "eps", "svg"))
        w_cm, h_cm = self.size_cm()
        if raster:
            px = (round(w_cm / 2.54 * self.dpi_value()), round(h_cm / 2.54 * self.dpi_value()))
            self.info.setText(f"{w_cm:.2f} × {h_cm:.2f} cm → {px[0]} × {px[1]} pixels")
        else:
            self.info.setText(f"{w_cm:.2f} × {h_cm:.2f} cm, vector (fonts embedded)")

    def dpi_value(self) -> int:
        try:
            return max(50, min(2400, int(float(self.dpi.currentText()))))
        except ValueError:
            return 300

    def size_cm(self) -> tuple[float, float]:
        f = self._factor()
        return self.width.value() / f, self.height.value() / f

    def values(self) -> dict:
        w, h = self.size_cm()
        custom = self.custom.isChecked()
        return {"fmt": self.fmt.currentData(), "dpi": self.dpi_value(), "custom": custom,
                "unit": self.unit.currentText(), "width_cm": w if custom else None,
                "height_cm": h if custom else None,
                "transparent": self.transparent.isChecked()}


class PageSetupDialog(QDialog):
    def __init__(self, graph: Graph, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Page Setup")
        self.width = QDoubleSpinBox()
        self.height = QDoubleSpinBox()
        for w, v in ((self.width, graph.page.width_cm), (self.height, graph.page.height_cm)):
            w.setRange(2, 60)
            w.setDecimals(2)
            w.setSuffix(" cm")
            w.setValue(v)
        form = QFormLayout()
        form.addRow("Width:", self.width)
        form.addRow("Height:", self.height)
        presets = QVBoxLayout()
        for text, w, h in PAGE_PRESETS:
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, w=w, h=h: (self.width.setValue(w),
                                                          self.height.setValue(h)))
            presets.addWidget(b)
        box = QGroupBox("Presets")
        box.setLayout(presets)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(box)
        lay.addWidget(buttons)

    def values(self) -> tuple[float, float]:
        return self.width.value(), self.height.value()

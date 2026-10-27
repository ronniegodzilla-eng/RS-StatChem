"""Graphs > Chart Builder: SPSS-style gallery of the charts Graph Studio has no template for.

Pick a chart in the gallery, assign variables to its slots, adjust the options and watch the
preview; OK puts the chart in the Output Viewer (as SPSS does). Runs through the editor's
analysis protocol, so Split File / Select Cases apply.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFormLayout, QGroupBox, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QScrollArea, QSpinBox, QVBoxLayout)

from rsstatchem.core.dataset import NUMERIC, Dataset
from rsstatchem.core.graph import spss_charts as sc

_CHOICES = {"error": [("ci", "Confidence interval"), ("se", "±2 standard errors"),
                      ("sd", "±1 standard deviation")],
            "summary": [("mean", "Mean"), ("median", "Median")],
            "diagonal": [("histogram", "Histogram")]}
_LABELS = {"fit": "Fit line (linear)", "box": "Show box plot inside", "bins": "Number of bins "
           "(0 = automatic):", "percent": "Show percentages", "normal": "Display normal curve",
           "error": "Error bars:", "level": "Level (%):", "summary": "Statistic:",
           "diagonal": "Diagonal:"}


class ChartBuilderDialog(QDialog):
    PRESET: str | None = None
    _last: dict = {}

    def __init__(self, ds: Dataset, parent=None):
        super().__init__(parent)
        self.ds = ds
        self.setWindowTitle("Chart Builder")
        self.resize(1040, 640)
        self.gallery = QListWidget()
        for cat in sc.CATEGORIES:
            head = QListWidgetItem(cat)
            head.setFlags(Qt.ItemFlag.NoItemFlags)
            font = head.font()
            font.setBold(True)
            head.setFont(font)
            self.gallery.addItem(head)
            for chart in sc.CHARTS.values():
                if chart.category == cat:
                    item = QListWidgetItem("    " + chart.label)
                    item.setData(Qt.ItemDataRole.UserRole, chart.key)
                    item.setToolTip(chart.hint)
                    self.gallery.addItem(item)
        self.gallery.currentItemChanged.connect(lambda *_: self._chart_changed())
        self.slots_box = QGroupBox("Variables")
        self.slots_form = QFormLayout(self.slots_box)
        self.options_box = QGroupBox("Element Properties")
        self.options_form = QFormLayout(self.options_box)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color:#555;")
        self.preview = QLabel("Preview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(460, 380)
        self.preview.setStyleSheet("background:white; border:1px solid #ccc;")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.preview)
        self.status = QLabel()
        self.status.setStyleSheet("color:#8E1B1B;")
        self.widgets: dict = {}
        self.option_widgets: dict = {}
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        left = QVBoxLayout()
        left.addWidget(QLabel("Gallery — choose from:"))
        left.addWidget(self.gallery, 1)
        middle = QVBoxLayout()
        middle.addWidget(self.slots_box)
        middle.addWidget(self.options_box)
        middle.addWidget(self.hint)
        middle.addStretch()
        right = QVBoxLayout()
        right.addWidget(scroll, 1)
        right.addWidget(self.status)
        body = QHBoxLayout()
        body.addLayout(left, 2)
        body.addLayout(middle, 3)
        body.addLayout(right, 5)
        layout = QVBoxLayout(self)
        layout.addLayout(body, 1)
        layout.addWidget(buttons)
        key = self.PRESET or self._last.get("key") or "splom"
        self.select_chart(key)

    # ------------------------------------------------------------------ gallery
    def select_chart(self, key: str) -> None:
        for i in range(self.gallery.count()):
            if self.gallery.item(i).data(Qt.ItemDataRole.UserRole) == key:
                self.gallery.setCurrentRow(i)
                return

    def chart(self) -> sc.ChartType:
        item = self.gallery.currentItem()
        key = item.data(Qt.ItemDataRole.UserRole) if item is not None else "splom"
        return sc.CHARTS[key or "splom"]

    def _clear(self, form: QFormLayout) -> None:
        while form.rowCount():
            form.removeRow(0)

    def _chart_changed(self) -> None:
        chart = self.chart()
        self._clear(self.slots_form)
        self._clear(self.options_form)
        self.widgets, self.option_widgets = {}, {}
        remembered = self._last.get("roles", {})
        for slot in chart.slots:
            names = [v.name for v in self.ds.variables
                     if not slot.numeric or v.type == NUMERIC]
            if slot.multiple:
                w = QListWidget()
                w.setMaximumHeight(130)
                for n in names:
                    item = QListWidgetItem(self.ds.var(n).title + f" [{n}]")
                    item.setData(Qt.ItemDataRole.UserRole, n)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(Qt.CheckState.Checked if n in
                                       (remembered.get(slot.key) or []) else
                                       Qt.CheckState.Unchecked)
                    w.addItem(item)
                w.itemChanged.connect(lambda *_: self.update_preview())
            else:
                w = QComboBox()
                w.addItem("(none)" if not slot.required else "— choose —", "")
                for n in names:
                    w.addItem(self.ds.var(n).title + f" [{n}]", n)
                prev = remembered.get(slot.key)
                if isinstance(prev, str) and prev in names:
                    w.setCurrentIndex(names.index(prev) + 1)
                w.currentIndexChanged.connect(lambda *_: self.update_preview())
            self.widgets[slot.key] = w
            self.slots_form.addRow(slot.label, w)
        for key, default in chart.options.items():
            if isinstance(default, bool):
                w = QCheckBox(_LABELS.get(key, key))
                w.setChecked(default)
                w.toggled.connect(lambda *_: self.update_preview())
                self.options_form.addRow(w)
            elif key in _CHOICES:
                w = QComboBox()
                for value, text in _CHOICES[key]:
                    w.addItem(text, value)
                w.setCurrentIndex([v for v, _ in _CHOICES[key]].index(default))
                w.currentIndexChanged.connect(lambda *_: self.update_preview())
                self.options_form.addRow(_LABELS.get(key, key), w)
            elif isinstance(default, int):
                w = QSpinBox()
                w.setRange(0, 200)
                w.setValue(default)
                w.valueChanged.connect(lambda *_: self.update_preview())
                self.options_form.addRow(_LABELS.get(key, key), w)
            else:
                w = QDoubleSpinBox()
                w.setRange(50, 99.9)
                w.setValue(float(default))
                w.valueChanged.connect(lambda *_: self.update_preview())
                self.options_form.addRow(_LABELS.get(key, key), w)
            self.option_widgets[key] = w
        self.options_box.setVisible(bool(chart.options))
        self.hint.setText(chart.hint)
        self.setWindowTitle(f"Chart Builder — {chart.label}")
        self.update_preview()

    # ------------------------------------------------------------------ values
    def roles(self) -> dict:
        out = {}
        for key, w in self.widgets.items():
            if isinstance(w, QListWidget):
                out[key] = [w.item(i).data(Qt.ItemDataRole.UserRole) for i in range(w.count())
                            if w.item(i).checkState() == Qt.CheckState.Checked]
            else:
                out[key] = w.currentData() or None
        return out

    def options(self) -> dict:
        out = {}
        for key, w in self.option_widgets.items():
            if isinstance(w, QCheckBox):
                out[key] = w.isChecked()
            elif isinstance(w, QComboBox):
                out[key] = w.currentData()
            else:
                out[key] = w.value()
        return out

    def set_roles(self, roles: dict) -> None:
        for key, value in roles.items():
            w = self.widgets.get(key)
            if isinstance(w, QListWidget):
                for i in range(w.count()):
                    w.item(i).setCheckState(Qt.CheckState.Checked if w.item(i).data(
                        Qt.ItemDataRole.UserRole) in value else Qt.CheckState.Unchecked)
            elif isinstance(w, QComboBox):
                index = w.findData(value)
                w.setCurrentIndex(max(index, 0))
        self.update_preview()

    def update_preview(self) -> None:
        if not hasattr(self, "ok_button"):
            return
        chart = self.chart()
        roles = self.roles()
        error = sc.check_roles(chart, roles)
        self.ok_button.setEnabled(error is None)
        if error:
            self.preview.setPixmap(QPixmap())
            self.preview.setText("Assign the variables to see a preview.")
            self.status.setText("")
            return
        try:
            fig, _ = sc.render(self.ds, chart.key, roles, self.options())
            image = QImage.fromData(sc._png(fig))
            pix = QPixmap.fromImage(image)
            if pix.width() > 620:
                pix = pix.scaledToWidth(620, Qt.TransformationMode.SmoothTransformation)
            self.preview.setPixmap(pix)
            self.status.setText("")
        except Exception as exc:  # noqa: BLE001 - show the problem instead of a chart
            self.preview.setPixmap(QPixmap())
            self.preview.setText("")
            self.status.setText(str(exc))

    def accept(self) -> None:
        type(self)._last.update({"key": self.chart().key, "roles": self.roles()})
        super().accept()

    def compute(self):
        return sc.chart_section(self.ds, self.chart().key, self.roles(), self.options())


def preset(key: str, title: str) -> type:
    """A Chart Builder subclass that opens on chart ``key`` (for menu entries)."""
    return type(f"ChartBuilder_{key}", (ChartBuilderDialog,), {"PRESET": key,
                                                               "__doc__": title})


"""Analyze > Descriptive Statistics > Frequencies… / Descriptives…"""
from __future__ import annotations

from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QDialog, QDialogButtonBox, QGridLayout,
                               QGroupBox, QRadioButton, QVBoxLayout)

from rsstatchem.core.dataset import NUMERIC
from rsstatchem.core.stats.descriptive import DEFAULT_DESCRIPTIVES
from rsstatchem.ui.dialogs.base import VariablePickerDialog


def _numeric(var) -> bool:
    return var.type == NUMERIC


class _CheckSubDialog(QDialog):
    """Sub-dialog of grouped check boxes (e.g. Descriptives: Options…)."""

    def __init__(self, title: str, groups: list[tuple[str, list[tuple[str, str]]]],
                 checked: set[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.boxes: dict[str, QCheckBox] = {}
        layout = QVBoxLayout(self)
        grid = QGridLayout()
        for i, (group_title, items) in enumerate(groups):
            box = QGroupBox(group_title)
            inner = QVBoxLayout(box)
            for key, text in items:
                cb = QCheckBox(text)
                cb.setChecked(key in checked)
                self.boxes[key] = cb
                inner.addWidget(cb)
            inner.addStretch()
            grid.addWidget(box, i // 2, i % 2)
        layout.addLayout(grid)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Continue")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def checked(self) -> set[str]:
        return {k for k, cb in self.boxes.items() if cb.isChecked()}


class DescriptivesDialog(VariablePickerDialog):
    _stats: set[str] = set(DEFAULT_DESCRIPTIVES)
    _save_z = False

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Descriptives", parent)
        self.add_target("vars", "Variable(s):", accepts=_numeric)
        self.add_option_button("Options...", self._options)
        self.save_z = QCheckBox("Save standardized values as variables")
        self.save_z.setChecked(type(self)._save_z)
        self.options_layout.addWidget(self.save_z)
        self.finish()

    def _options(self) -> None:
        dlg = _CheckSubDialog("Descriptives: Options", [
            ("Central tendency", [("mean", "Mean"), ("sum", "Sum")]),
            ("Dispersion", [("std", "Std. deviation"), ("var", "Variance"), ("range", "Range"),
                            ("min", "Minimum"), ("max", "Maximum"), ("sem", "S.E. mean")]),
            ("Characterize Posterior Distribution", [("kurt", "Kurtosis"),
                                                     ("skew", "Skewness")]),
        ], type(self)._stats, self)
        if dlg.exec():
            type(self)._stats = dlg.checked()

    def accept(self) -> None:
        type(self)._save_z = self.save_z.isChecked()
        super().accept()

    def params(self) -> dict:
        return {"names": self.selected("vars"), "stats": tuple(sorted(type(self)._stats)),
                "save_z": self.save_z.isChecked()}


class FrequenciesDialog(VariablePickerDialog):
    _stats: set[str] = set()
    _chart = "none"
    _normal = True
    _tables = True

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Frequencies", parent)
        self.add_target("vars", "Variable(s):")
        self.add_option_button("Statistics...", self._statistics)
        self.add_option_button("Charts...", self._charts)
        self.show_tables = QCheckBox("Display frequency tables")
        self.show_tables.setChecked(type(self)._tables)
        self.options_layout.addWidget(self.show_tables)
        self.finish()

    def _statistics(self) -> None:
        dlg = _CheckSubDialog("Frequencies: Statistics", [
            ("Percentile Values", [("quartiles", "Quartiles")]),
            ("Central Tendency", [("mean", "Mean"), ("median", "Median"), ("mode", "Mode"),
                                  ("sum", "Sum")]),
            ("Dispersion", [("std", "Std. deviation"), ("var", "Variance"), ("range", "Range"),
                            ("min", "Minimum"), ("max", "Maximum"), ("sem", "S.E. mean")]),
            ("Characterize Posterior Distribution", [("skew", "Skewness"),
                                                     ("kurt", "Kurtosis")]),
        ], type(self)._stats, self)
        if dlg.exec():
            type(self)._stats = dlg.checked()

    def _charts(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Frequencies: Charts")
        layout = QVBoxLayout(dlg)
        box = QGroupBox("Chart Type")
        inner = QVBoxLayout(box)
        group = QButtonGroup(dlg)
        radios = {}
        for key, text in (("none", "None"), ("bar", "Bar charts"), ("pie", "Pie charts"),
                          ("histogram", "Histograms")):
            rb = QRadioButton(text)
            rb.setChecked(key == type(self)._chart)
            group.addButton(rb)
            radios[key] = rb
            inner.addWidget(rb)
        normal = QCheckBox("Show normal curve on histogram")
        normal.setChecked(type(self)._normal)
        normal.setEnabled(radios["histogram"].isChecked())
        radios["histogram"].toggled.connect(normal.setEnabled)
        inner.addWidget(normal)
        layout.addWidget(box)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Continue")
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)
        if dlg.exec():
            type(self)._chart = next(k for k, rb in radios.items() if rb.isChecked())
            type(self)._normal = normal.isChecked()

    def accept(self) -> None:
        type(self)._tables = self.show_tables.isChecked()
        super().accept()

    def params(self) -> dict:
        return {"names": self.selected("vars"), "stats": tuple(sorted(type(self)._stats)),
                "show_tables": self.show_tables.isChecked(), "chart": type(self)._chart,
                "normal_curve": type(self)._normal}

"""Analyze > Descriptive Statistics > Frequencies… / Descriptives…

The Data Editor runs these two dialogs itself (``params()`` → ``descriptive.frequencies`` /
``descriptive.descriptives``), so the extra SPSS options travel inside ``params()["stats"]``,
a :class:`~rsstatchem.core.stats.descriptive.StatSpec` (a tuple of keywords with ``.options``).
"""
from __future__ import annotations


from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QDialog, QDialogButtonBox, QGridLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QPushButton, QRadioButton, QSpinBox, QVBoxLayout)

from rsstatchem.core.dataset import NUMERIC
from rsstatchem.core.stats.descriptive import DEFAULT_DESCRIPTIVES, StatSpec
from rsstatchem.ui.dialogs.base import VariablePickerDialog


def _numeric(var) -> bool:
    return var.type == NUMERIC


def _buttons(dlg: QDialog, layout) -> None:
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                               | QDialogButtonBox.StandardButton.Cancel)
    buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Continue")
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    layout.addWidget(buttons)


def _radios(dlg: QDialog, layout, options: list[tuple[str, str]], current: str) -> dict:
    group = QButtonGroup(dlg)
    out = {}
    for key, text in options:
        rb = QRadioButton(text)
        rb.setChecked(key == current)
        group.addButton(rb)
        layout.addWidget(rb)
        out[key] = rb
    if not any(rb.isChecked() for rb in out.values()):
        next(iter(out.values())).setChecked(True)
    return out


def _chosen(radios: dict) -> str:
    return next(k for k, rb in radios.items() if rb.isChecked())


class _CheckSubDialog(QDialog):
    """Sub-dialog of grouped check boxes (e.g. Descriptives: Options…)."""

    def __init__(self, title: str, groups: list[tuple[str, list[tuple[str, str]]]],
                 checked: set[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.boxes: dict[str, QCheckBox] = {}
        self.layout_ = QVBoxLayout(self)
        self.grid = QGridLayout()
        for i, (group_title, items) in enumerate(groups):
            box = QGroupBox(group_title)
            inner = QVBoxLayout(box)
            for key, text in items:
                cb = QCheckBox(text)
                cb.setChecked(key in checked)
                self.boxes[key] = cb
                inner.addWidget(cb)
            inner.addStretch()
            self.grid.addWidget(box, i // 2, i % 2)
        self.layout_.addLayout(self.grid)

    def finish(self) -> None:
        _buttons(self, self.layout_)

    def checked(self) -> set[str]:
        return {k for k, cb in self.boxes.items() if cb.isChecked()}


class DescriptivesDialog(VariablePickerDialog):
    _stats: set[str] = set(DEFAULT_DESCRIPTIVES)
    _save_z = False
    _order = "variable"

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
            ("", [("mean", "Mean"), ("sum", "Sum")]),
            ("Dispersion", [("std", "Std. deviation"), ("var", "Variance"), ("range", "Range"),
                            ("min", "Minimum"), ("max", "Maximum"), ("sem", "S.E. mean")]),
            ("Characterize Posterior Distribution", [("kurt", "Kurtosis"),
                                                     ("skew", "Skewness")]),
        ], type(self)._stats, self)
        box = QGroupBox("Display Order")
        inner = QVBoxLayout(box)
        radios = _radios(dlg, inner, [("variable", "Variable list"), ("alpha", "Alphabetic"),
                                      ("amean", "Ascending means"),
                                      ("dmean", "Descending means")], type(self)._order)
        dlg.grid.addWidget(box, 1, 1)
        dlg.finish()
        if dlg.exec():
            type(self)._stats = dlg.checked()
            type(self)._order = _chosen(radios)

    def accept(self) -> None:
        type(self)._save_z = self.save_z.isChecked()
        super().accept()

    def params(self) -> dict:
        return {"names": self.selected("vars"),
                "stats": StatSpec(sorted(type(self)._stats), {"order": type(self)._order}),
                "save_z": self.save_z.isChecked()}


class FrequenciesDialog(VariablePickerDialog):
    _stats: set[str] = set()
    _chart = "none"
    _normal = True
    _tables = True
    _extra: dict = {"percentiles": [], "cut_points": 10, "use_cut": False, "grouped": False,
                    "chart_values": "frequencies", "order": "avalue", "organize": "compare",
                    "use_limit": False, "limit": 10}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Frequencies", parent)
        self.add_target("vars", "Variable(s):")
        self.add_option_button("Statistics...", self._statistics)
        self.add_option_button("Charts...", self._charts)
        self.add_option_button("Format...", self._format)
        self.show_tables = QCheckBox("Display frequency tables")
        self.show_tables.setChecked(type(self)._tables)
        self.options_layout.addWidget(self.show_tables)
        self.finish()

    def _statistics(self) -> None:
        extra = type(self)._extra
        dlg = _CheckSubDialog("Frequencies: Statistics", [
            ("Percentile Values", [("quartiles", "Quartiles")]),
            ("Central Tendency", [("mean", "Mean"), ("median", "Median"), ("mode", "Mode"),
                                  ("sum", "Sum")]),
            ("Dispersion", [("std", "Std. deviation"), ("var", "Variance"), ("range", "Range"),
                            ("min", "Minimum"), ("max", "Maximum"), ("sem", "S.E. mean")]),
            ("Characterize Posterior Distribution", [("skew", "Skewness"),
                                                     ("kurt", "Kurtosis")]),
        ], type(self)._stats, self)
        box = dlg.grid.itemAtPosition(0, 0).widget()
        inner = box.layout()
        inner.takeAt(inner.count() - 1)        # drop the stretch, then add the extra rows
        cut_row = QHBoxLayout()
        use_cut = QCheckBox("Cut points for")
        use_cut.setChecked(extra["use_cut"])
        cut = QSpinBox()
        cut.setRange(2, 100)
        cut.setValue(int(extra["cut_points"]))
        cut_row.addWidget(use_cut)
        cut_row.addWidget(cut)
        cut_row.addWidget(QLabel("equal groups"))
        cut_row.addStretch()
        inner.addLayout(cut_row)
        use_pct = QCheckBox("Percentile(s):")
        use_pct.setChecked(bool(extra["percentiles"]))
        inner.addWidget(use_pct)
        pct_row = QHBoxLayout()
        entry = QLineEdit()
        entry.setPlaceholderText("0–100")
        entry.setMaximumWidth(80)
        listing = QListWidget()
        listing.setMaximumHeight(80)
        for p in extra["percentiles"]:
            listing.addItem(f"{float(p):g}")
        add, remove = QPushButton("Add"), QPushButton("Remove")

        def do_add():
            try:
                v = float(entry.text().replace(",", "."))
            except ValueError:
                return
            if 0 <= v <= 100:
                listing.addItem(f"{v:g}")
                entry.clear()
                use_pct.setChecked(True)

        add.clicked.connect(do_add)
        entry.returnPressed.connect(do_add)
        remove.clicked.connect(lambda: [listing.takeItem(listing.row(i))
                                        for i in listing.selectedItems()])
        col = QVBoxLayout()
        col.addWidget(entry)
        col.addWidget(add)
        col.addWidget(remove)
        pct_row.addLayout(col)
        pct_row.addWidget(listing)
        inner.addLayout(pct_row)
        grouped = QCheckBox("Values are group midpoints")
        grouped.setChecked(extra["grouped"])
        cbox = dlg.grid.itemAtPosition(0, 1).widget().layout()
        cbox.takeAt(cbox.count() - 1)
        cbox.addWidget(grouped)
        cbox.addStretch()
        dlg.finish()
        if dlg.exec():
            type(self)._stats = dlg.checked()
            extra["use_cut"] = use_cut.isChecked()
            extra["cut_points"] = cut.value()
            extra["percentiles"] = [float(listing.item(i).text()) for i in
                                    range(listing.count())] if use_pct.isChecked() else []
            extra["grouped"] = grouped.isChecked()

    def _charts(self) -> None:
        extra = type(self)._extra
        dlg = QDialog(self)
        dlg.setWindowTitle("Frequencies: Charts")
        layout = QVBoxLayout(dlg)
        box = QGroupBox("Chart Type")
        inner = QVBoxLayout(box)
        radios = _radios(dlg, inner, [("none", "None"), ("bar", "Bar charts"),
                                      ("pie", "Pie charts"), ("histogram", "Histograms")],
                         type(self)._chart)
        normal = QCheckBox("Show normal curve on histogram")
        normal.setChecked(type(self)._normal)
        normal.setEnabled(radios["histogram"].isChecked())
        radios["histogram"].toggled.connect(normal.setEnabled)
        inner.addWidget(normal)
        layout.addWidget(box)
        vbox = QGroupBox("Chart Values")
        vinner = QVBoxLayout(vbox)
        values = _radios(dlg, vinner, [("frequencies", "Frequencies"),
                                       ("percentages", "Percentages")], extra["chart_values"])
        layout.addWidget(vbox)
        _buttons(dlg, layout)
        if dlg.exec():
            type(self)._chart = _chosen(radios)
            type(self)._normal = normal.isChecked()
            extra["chart_values"] = _chosen(values)

    def _format(self) -> None:
        extra = type(self)._extra
        dlg = QDialog(self)
        dlg.setWindowTitle("Frequencies: Format")
        layout = QVBoxLayout(dlg)
        row = QHBoxLayout()
        box = QGroupBox("Order by")
        order = _radios(dlg, QVBoxLayout(box), [("avalue", "Ascending values"),
                                               ("dvalue", "Descending values"),
                                               ("afreq", "Ascending counts"),
                                               ("dfreq", "Descending counts")], extra["order"])
        row.addWidget(box)
        box2 = QGroupBox("Multiple Variables")
        organize = _radios(dlg, QVBoxLayout(box2), [("compare", "Compare variables"),
                                                   ("variable", "Organize output by variables")],
                           extra["organize"])
        row.addWidget(box2)
        layout.addLayout(row)
        limit_row = QHBoxLayout()
        use_limit = QCheckBox("Suppress tables with many categories")
        use_limit.setChecked(extra["use_limit"])
        limit = QSpinBox()
        limit.setRange(1, 10000)
        limit.setValue(int(extra["limit"]))
        limit_row.addWidget(use_limit)
        limit_row.addWidget(QLabel("Maximum number of categories:"))
        limit_row.addWidget(limit)
        layout.addLayout(limit_row)
        _buttons(dlg, layout)
        if dlg.exec():
            extra["order"] = _chosen(order)
            extra["organize"] = _chosen(organize)
            extra["use_limit"] = use_limit.isChecked()
            extra["limit"] = limit.value()

    def accept(self) -> None:
        type(self)._tables = self.show_tables.isChecked()
        super().accept()

    def params(self) -> dict:
        extra = type(self)._extra
        options = {"percentiles": list(extra["percentiles"]),
                   "cut_points": extra["cut_points"] if extra["use_cut"] else None,
                   "grouped": extra["grouped"], "chart_values": extra["chart_values"],
                   "order": extra["order"], "organize": extra["organize"],
                   "limit": extra["limit"] if extra["use_limit"] else None}
        return {"names": self.selected("vars"),
                "stats": StatSpec(sorted(type(self)._stats), options),
                "show_tables": self.show_tables.isChecked(), "chart": type(self)._chart,
                "normal_curve": type(self)._normal}

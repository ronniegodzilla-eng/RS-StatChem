"""Multiple Response: Define Sets, Frequencies and Crosstabs (SPSS Analyze > Multiple
Response, Data > Define Multiple Response Sets, Analyze > Tables > Multiple Response Sets).

All three "define" entries edit the same sets, stored with the data (``ds.meta["mrsets"]``),
so a set defined once is available to MR Frequencies/Crosstabs and to Custom Tables.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QDialog, QDialogButtonBox, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QMessageBox, QPushButton, QRadioButton,
                               QVBoxLayout, QWidget)

from rsstatchem.core import mresponse as mr
from rsstatchem.core.dataset import NUMERIC, Dataset, format_code
from rsstatchem.ui.dialogs.base import NAME_ROLE, SubDialog, variable_item
from rsstatchem.ui.dialogs.picker import ListPickerDialog


class DefineSetsDialog(QDialog):
    """Define Multiple Response Sets: name, label, dichotomies (counted value) or categories
    (optional range), variables."""

    def __init__(self, ds: Dataset, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Define Multiple Response Sets")
        self.resize(780, 520)
        self.ds = ds
        self.sets = mr.get_sets(ds)
        self.variables = QListWidget()
        self.variables.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        for v in ds.variables:
            self.variables.addItem(variable_item(v))
        self.members = QListWidget()
        self.members.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        add = QPushButton("▶")
        add.setObjectName("arrow")
        add.clicked.connect(self._add_vars)
        remove = QPushButton("◀")
        remove.setObjectName("arrow")
        remove.clicked.connect(self._remove_vars)
        self.members.itemDoubleClicked.connect(lambda _: self._remove_vars())
        self.variables.itemDoubleClicked.connect(lambda _: self._add_vars())

        enc = QGroupBox("Variables Are Coded As")
        el = QVBoxLayout(enc)
        self.dich = QRadioButton("Dichotomies   Counted value:")
        self.cat = QRadioButton("Categories   Range (optional):")
        self.counted = QLineEdit("1")
        self.lo, self.hi = QLineEdit(), QLineEdit()
        group = QButtonGroup(self)
        group.addButton(self.dich)
        group.addButton(self.cat)
        self.dich.setChecked(True)
        row1 = QHBoxLayout()
        row1.addWidget(self.dich)
        row1.addWidget(self.counted)
        row2 = QHBoxLayout()
        row2.addWidget(self.cat)
        row2.addWidget(self.lo)
        row2.addWidget(QLabel("through"))
        row2.addWidget(self.hi)
        el.addLayout(row1)
        el.addLayout(row2)
        self.use_counted_labels = QCheckBox("Category labels from the counted values' labels "
                                            "(default: variable labels)")
        el.addWidget(self.use_counted_labels)

        self.name = QLineEdit()
        self.name.setPlaceholderText("$name")
        self.label = QLineEdit()
        form = QFormLayout()
        form.addRow("Set name:", self.name)
        form.addRow("Set label:", self.label)

        self.set_list = QListWidget()
        self.set_list.currentRowChanged.connect(self._load)
        b_add = QPushButton("Add")
        b_change = QPushButton("Change")
        b_remove = QPushButton("Remove")
        b_add.clicked.connect(self._add_set)
        b_change.clicked.connect(self._change_set)
        b_remove.clicked.connect(self._remove_set)
        self._refresh()

        left = QVBoxLayout()
        left.addWidget(QLabel("Set Definition — variables:"))
        left.addWidget(self.variables)
        arrows = QVBoxLayout()
        arrows.addStretch()
        arrows.addWidget(add)
        arrows.addWidget(remove)
        arrows.addStretch()
        mid = QVBoxLayout()
        mid.addWidget(QLabel("Variables in Set:"))
        mid.addWidget(self.members)
        mid.addWidget(enc)
        mid.addLayout(form)
        right = QVBoxLayout()
        right.addWidget(QLabel("Multiple Response Sets:"))
        right.addWidget(self.set_list, 1)
        for b in (b_add, b_change, b_remove):
            right.addWidget(b)
        body = QHBoxLayout()
        body.addLayout(left, 2)
        body.addLayout(arrows)
        body.addLayout(mid, 3)
        body.addLayout(right, 2)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(body, 1)
        layout.addWidget(buttons)

    def _add_vars(self) -> None:
        present = {self.members.item(i).data(NAME_ROLE) for i in range(self.members.count())}
        for item in self.variables.selectedItems():
            name = item.data(NAME_ROLE)
            if name not in present:
                self.members.addItem(variable_item(self.ds.var(name)))

    def _remove_vars(self) -> None:
        for item in self.members.selectedItems():
            self.members.takeItem(self.members.row(item))

    def _refresh(self) -> None:
        self.set_list.blockSignals(True)
        self.set_list.clear()
        for s in self.sets:
            self.set_list.addItem(f"{s.name}  ({s.kind}, {len(s.variables)} variables)")
        self.set_list.blockSignals(False)

    def _value(self, text: str, numeric: bool):
        text = text.strip()
        if not text:
            return None
        return float(text.replace(",", ".")) if numeric else text

    def current_set(self) -> mr.MRSet:
        names = [self.members.item(i).data(NAME_ROLE) for i in range(self.members.count())]
        numeric = bool(names) and self.ds.var(names[0]).type == NUMERIC
        name = self.name.text().strip()
        if name and not name.startswith("$"):
            name = "$" + name
        if self.dich.isChecked():
            counted = self._value(self.counted.text(), numeric)
            return mr.MRSet(name, self.label.text().strip(), "dichotomy", names,
                            counted if counted is not None else 1.0, None,
                            "countedlabels" if self.use_counted_labels.isChecked()
                            else "varlabels")
        lo, hi = self._value(self.lo.text(), True), self._value(self.hi.text(), True)
        rng = [lo, hi] if lo is not None or hi is not None else None
        return mr.MRSet(name, self.label.text().strip(), "category", names, 1.0, rng)

    def _add_set(self) -> None:
        try:
            s = self.current_set()
        except ValueError as exc:
            QMessageBox.warning(self, self.windowTitle(), str(exc))
            return
        error = mr.validate(self.ds, s, self.sets)
        if error:
            QMessageBox.warning(self, self.windowTitle(), error)
            return
        self.sets.append(s)
        self._refresh()
        self.set_list.setCurrentRow(len(self.sets) - 1)

    def _change_set(self) -> None:
        row = self.set_list.currentRow()
        if row < 0:
            return
        s = self.current_set()
        error = mr.validate(self.ds, s, [x for i, x in enumerate(self.sets) if i != row])
        if error:
            QMessageBox.warning(self, self.windowTitle(), error)
            return
        self.sets[row] = s
        self._refresh()
        self.set_list.setCurrentRow(row)

    def _remove_set(self) -> None:
        row = self.set_list.currentRow()
        if row >= 0:
            del self.sets[row]
            self._refresh()

    def _load(self, row: int) -> None:
        if row < 0 or row >= len(self.sets):
            return
        s = self.sets[row]
        self.name.setText(s.name)
        self.label.setText(s.label)
        self.members.clear()
        for v in s.variables:
            if self.ds.has_name(v):
                self.members.addItem(variable_item(self.ds.var(v)))
        if s.kind == "dichotomy":
            self.dich.setChecked(True)
            self.counted.setText(format_code(s.counted))
            self.use_counted_labels.setChecked(s.category_labels == "countedlabels")
        else:
            self.cat.setChecked(True)
            lo, hi = (s.range or [None, None])
            self.lo.setText("" if lo is None else format_code(lo))
            self.hi.setText("" if hi is None else format_code(hi))


def define_sets(editor) -> None:
    dlg = DefineSetsDialog(editor.ds, editor)
    if dlg.exec():
        sets = dlg.sets
        editor.doc.mutate(lambda ds: mr.save_sets(ds, sets), "Define Multiple Response Sets")
        editor.statusBar().showMessage(f"{len(sets)} multiple response set(s) defined.", 5000)


def _need_sets(ds: Dataset, parent) -> bool:
    if mr.get_sets(ds):
        return True
    QMessageBox.information(parent, "Multiple Response",
                            "No multiple response sets are defined yet. Use Analyze > Multiple "
                            "Response > Define Variable Sets first.")
    return False


class MRFrequenciesDialog(ListPickerDialog):
    def __init__(self, ds: Dataset, parent=None):
        super().__init__(ds, "Multiple Response Frequencies", parent, variables=False)
        self.add_target("sets", "Table(s) for:")
        self.finish()

    def exec(self) -> int:
        if not _need_sets(self.ds, self.parent()):
            return 0
        return super().exec()

    def compute(self):
        return mr.frequencies(self.ds, self.selected("sets"))


class MRCrosstabsDialog(ListPickerDialog):
    DEFAULTS = {"row": True, "col": False, "total": False, "base": "cases"}

    def __init__(self, ds: Dataset, parent=None):
        super().__init__(ds, "Multiple Response Crosstabs", parent)
        self.add_target("rows", "Row(s):")
        self.add_target("cols", "Column(s):")
        self.add_target("layers", "Layer(s):", min_items=0)
        self.add_button("Options...", self._options)
        self.finish()

    def exec(self) -> int:
        if not _need_sets(self.ds, self.parent()):
            return 0
        return super().exec()

    def _options(self) -> None:
        dlg = SubDialog("Multiple Response Crosstabs: Options", self)
        box = dlg.group("Cell Percentages")
        dlg.check(box, "row", "Row", self.opts["row"])
        dlg.check(box, "col", "Column", self.opts["col"])
        dlg.check(box, "total", "Total", self.opts["total"])
        box2 = dlg.group("Percentages Based on")
        dlg.radio(box2, "base", [("cases", "Cases (respondents)"), ("responses", "Responses")],
                  self.opts["base"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def compute(self):
        return mr.crosstabs(self.ds, self.selected("rows"), self.selected("cols"),
                            self.selected("layers") or None, self.opts["row"],
                            self.opts["col"], self.opts["total"], self.opts["base"])


MENUS = {
    "Data": [("Define Multiple Response Sets...", define_sets)],
    "Analyze/Tables": [("Multiple Response Sets...", define_sets)],
    "Analyze/Multiple Response": [("Define Variable Sets...", define_sets),
                                  ("Frequencies...", MRFrequenciesDialog),
                                  ("Crosstabs...", MRCrosstabsDialog)],
}


def selftest() -> None:
    rng = np.random.default_rng(3)
    n = 40
    df = pd.DataFrame({f"m{i}": rng.integers(0, 2, n).astype(float) for i in range(1, 4)})
    df["c1"] = rng.integers(1, 5, n).astype(float)
    df["c2"] = rng.integers(1, 5, n).astype(float)
    df["g"] = rng.integers(1, 3, n).astype(float)
    ds = Dataset(df)
    mr.save_sets(ds, [mr.MRSet("$media", "Media", "dichotomy", ["m1", "m2", "m3"], 1.0),
                      mr.MRSet("$choice", "Choices", "category", ["c1", "c2"], 1.0, [1, 4])])
    counts = mr.frequency_counts(ds, "$media")
    assert list(counts["n"]) == [int(df[f"m{i}"].sum()) for i in range(1, 4)]
    sec = mr.crosstabs(ds, ["$media"], ["g"], row_pct=True)
    assert sec.items[-1].rows
    assert mr.frequencies(ds, ["$choice"]).items

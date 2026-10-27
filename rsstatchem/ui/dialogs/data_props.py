"""Data / Transform / Utilities items of the SPSS Data Editor that WP6 adds.

Data: Define Variable Properties, Copy Data Properties, Compare Datasets, Sort Variables,
Split into Files, Copy Dataset. Transform: Create Dummy Variables, Date and Time Wizard,
Random Number Generators. Utilities: Variables, Data File Comments, Define / Use Variable
Sets, Show All Variables. The cores are in :mod:`rsstatchem.core.data_props`.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMessageBox, QPlainTextEdit, QPushButton,
                               QRadioButton, QSpinBox, QStackedWidget, QTableWidget,
                               QTableWidgetItem, QTextBrowser, QVBoxLayout, QWidget)

from rsstatchem.core import data_props as dp
from rsstatchem.core import expr_functions as X
from rsstatchem.core.dataset import (DATE, MEASURES, NUMERIC, ROLES, STRING, Dataset,
                                     format_code)
from rsstatchem.ui.dialogs.base import NAME_ROLE, VariablePickerDialog, variable_item


def _buttons(dialog: QDialog) -> QDialogButtonBox:
    box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                           | QDialogButtonBox.StandardButton.Cancel)
    box.accepted.connect(dialog.accept)
    box.rejected.connect(dialog.reject)
    return box


def _pick_variables(ds: Dataset, title: str, parent, prompt: str = "Variables:",
                    preselected: list[str] | None = None) -> list[str] | None:
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.resize(420, 460)
    lst = QListWidget()
    lst.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
    for v in ds.variables:
        item = variable_item(v)
        lst.addItem(item)
        item.setSelected(bool(preselected) and v.name in preselected)
    layout = QVBoxLayout(dlg)
    layout.addWidget(QLabel(prompt))
    layout.addWidget(lst, 1)
    layout.addWidget(_buttons(dlg))
    if not dlg.exec():
        return None
    return [i.data(NAME_ROLE) for i in lst.selectedItems()]


# ================================================================== Define Variable Properties
class DefinePropertiesDialog(QDialog):
    """Scan the values of the chosen variables, then edit label, measurement level (with a
    suggestion), value labels and missing values in one grid per variable."""

    def __init__(self, ds: Dataset, names: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Define Variable Properties")
        self.resize(900, 600)
        self.ds = ds
        self.names = names
        self.state = {n: {"label": ds.var(n).label, "measure": ds.var(n).measure,
                          "role": ds.var(n).role, "labels": dict(ds.var(n).value_labels),
                          "missing": list(ds.var(n).missing)} for n in names}
        self.current: str | None = None
        self.var_list = QListWidget()
        for n in names:
            self.var_list.addItem(variable_item(ds.var(n)))
        self.var_list.currentRowChanged.connect(self._show)
        self.label = QLineEdit()
        self.measure = QComboBox()
        self.measure.addItems([m.capitalize() for m in MEASURES])
        self.role = QComboBox()
        self.role.addItems([r.capitalize() for r in ROLES])
        suggest = QPushButton("Suggest...")
        suggest.clicked.connect(self._suggest)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.grid = QTableWidget(0, 4)
        self.grid.setHorizontalHeaderLabels(["Missing", "Count", "Value", "Label"])
        self.grid.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        form = QFormLayout()
        form.addRow("Label:", self.label)
        mrow = QHBoxLayout()
        mrow.addWidget(self.measure)
        mrow.addWidget(suggest)
        mrow.addStretch()
        form.addRow("Measurement Level:", mrow)
        form.addRow("Role:", self.role)
        right = QVBoxLayout()
        right.addLayout(form)
        right.addWidget(self.info)
        right.addWidget(QLabel("Value Label grid (edit labels, tick user-missing values; "
                               "add a row for an unlisted value):"))
        right.addWidget(self.grid, 1)
        add_row = QPushButton("Add Value")
        add_row.clicked.connect(self._add_row)
        right.addWidget(add_row, 0, Qt.AlignmentFlag.AlignLeft)
        body = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(QLabel("Scanned Variable List:"))
        left.addWidget(self.var_list)
        body.addLayout(left, 1)
        body.addLayout(right, 3)
        layout = QVBoxLayout(self)
        layout.addLayout(body, 1)
        layout.addWidget(_buttons(self))
        self.var_list.setCurrentRow(0)

    def _store(self) -> None:
        n = self.current
        if n is None:
            return
        st = self.state[n]
        st["label"] = self.label.text()
        st["measure"] = MEASURES[self.measure.currentIndex()]
        st["role"] = ROLES[self.role.currentIndex()]
        labels, missing = {}, []
        numeric = self.ds.var(n).type == NUMERIC
        for r in range(self.grid.rowCount()):
            value_item = self.grid.item(r, 2)
            if value_item is None or not value_item.text().strip():
                continue
            raw = value_item.text().strip()
            try:
                value = float(raw.replace(",", ".")) if numeric else raw
            except ValueError:
                continue
            label_item = self.grid.item(r, 3)
            if label_item is not None and label_item.text().strip():
                labels[value] = label_item.text().strip()
            check = self.grid.item(r, 0)
            if check is not None and check.checkState() == Qt.CheckState.Checked:
                missing.append(value)
        st["labels"], st["missing"] = labels, missing

    def _show(self, row: int) -> None:
        self._store()
        if row < 0:
            return
        n = self.names[row]
        self.current = n
        st = self.state[n]
        var = self.ds.var(n)
        self.label.setText(st["label"])
        self.measure.setCurrentIndex(MEASURES.index(st["measure"]))
        self.role.setCurrentIndex(ROLES.index(st["role"]) if st["role"] in ROLES else 0)
        values = dp.scan_values(self.ds, n)
        total = sum(c for _, c, _ in values)
        self.info.setText(f"{var.name}: {var.type}, {len(values)} distinct values in "
                          f"{total} valid cases.")
        self.grid.setRowCount(0)
        for value, count, _ in values:
            self._add_row(value, count, st["labels"].get(value, ""), value in st["missing"])

    def _add_row(self, value=None, count: int | None = None, label: str = "",
                 missing: bool = False) -> None:
        r = self.grid.rowCount()
        self.grid.insertRow(r)
        check = QTableWidgetItem()
        check.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
        check.setCheckState(Qt.CheckState.Checked if missing else Qt.CheckState.Unchecked)
        self.grid.setItem(r, 0, check)
        c = QTableWidgetItem("" if count is None else str(count))
        c.setFlags(Qt.ItemFlag.ItemIsEnabled)
        self.grid.setItem(r, 1, c)
        self.grid.setItem(r, 2, QTableWidgetItem("" if value is None else format_code(value)))
        self.grid.setItem(r, 3, QTableWidgetItem(label))

    def _suggest(self) -> None:
        if self.current is None:
            return
        level, reason = dp.suggest_measure(self.ds, self.current)
        answer = QMessageBox.question(self, "Suggest Measurement Level",
                                      f"Suggested level: {level.capitalize()}.\n\n{reason}\n\n"
                                      "Use it?")
        if answer == QMessageBox.StandardButton.Yes:
            self.measure.setCurrentIndex(MEASURES.index(level))

    def accept(self) -> None:
        self._store()
        super().accept()

    def apply(self, ds: Dataset) -> None:
        for n, st in self.state.items():
            dp.apply_properties(ds, n, label=st["label"], measure=st["measure"],
                                value_labels=st["labels"], missing=st["missing"],
                                role=st["role"])


def define_properties(editor) -> None:
    ds = editor.ds
    if not ds.n_vars:
        return
    names = _pick_variables(ds, "Define Variable Properties", editor,
                            "Variables to scan (values are listed with their counts):")
    if not names:
        return
    dlg = DefinePropertiesDialog(ds, names, editor)
    if dlg.exec():
        editor.doc.mutate(dlg.apply, "Define Variable Properties")


# ================================================================== other datasets
def _other_dataset(editor, title: str) -> tuple[Dataset | None, str]:
    """Pick another open dataset or a file; returns (dataset, description)."""
    others = [e for e in editor.ctx.editors if e is not editor and e.ds.n_vars]
    choices = [f"Open dataset: {e.windowTitle().replace('[*]', '')}" for e in others]
    choices.append("External data file...")
    from PySide6.QtWidgets import QInputDialog

    choice, ok = QInputDialog.getItem(editor, title, "Source:", choices, 0, False)
    if not ok:
        return None, ""
    index = choices.index(choice)
    if index < len(others):
        return others[index].ds, choice
    start = editor.ctx.settings.value("last_dir", str(Path.home()))
    path, _ = QFileDialog.getOpenFileName(editor, title, start,
                                          "Data files (*.rsd *.sav *.dta *.sas7bdat *.xpt "
                                          "*.por)")
    if not path:
        return None, ""
    return load_any(path), path


def load_any(path: str) -> Dataset:
    from rsstatchem.io import files

    kind = files.kind_of(path)
    reader = {"project": files.load_project, "sav": files.read_sav, "por": files.read_por,
              "stata": files.read_stata, "sas": files.read_sas}.get(kind)
    if reader is None:
        raise ValueError("Choose an .rsd, .sav, .por, .dta, .sas7bdat or .xpt file.")
    return reader(path)


class CopyPropertiesDialog(QDialog):
    def __init__(self, source: Dataset, target: Dataset, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Copy Data Properties")
        self.resize(560, 520)
        self.source, self.target = source, target
        self.matches = QListWidget()
        for v in source.variables:
            if target.has_name(v.name):
                item = QListWidgetItem(v.name)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked)
                self.matches.addItem(item)
        self.attrs = {}
        attr_box = QGroupBox("Variable properties to copy")
        al = QVBoxLayout(attr_box)
        for a, text in (("label", "Variable label"), ("value_labels", "Value labels"),
                        ("missing", "Missing values"), ("measure", "Measurement level"),
                        ("role", "Role"), ("width", "Width"), ("decimals", "Decimals"),
                        ("columns", "Columns"), ("align", "Alignment")):
            cb = QCheckBox(text)
            cb.setChecked(a in ("label", "value_labels", "missing", "measure"))
            self.attrs[a] = cb
            al.addWidget(cb)
        self.merge = QCheckBox("Merge value labels (keep labels the target already has)")
        al.addWidget(self.merge)
        file_box = QGroupBox("Dataset properties")
        fl = QVBoxLayout(file_box)
        self.file_props = {}
        for key, text in (("comments", "File comments"), ("mrsets", "Multiple response sets"),
                          ("varsets", "Variable sets"), ("weight", "Weight variable")):
            cb = QCheckBox(text)
            self.file_props[key] = cb
            fl.addWidget(cb)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Matching variables (same name in source and active dataset):"))
        layout.addWidget(self.matches, 1)
        row = QHBoxLayout()
        row.addWidget(attr_box)
        row.addWidget(file_box)
        layout.addLayout(row)
        layout.addWidget(_buttons(self))

    def run(self, ds: Dataset):
        pairs = [(self.matches.item(i).text(), self.matches.item(i).text())
                 for i in range(self.matches.count())
                 if self.matches.item(i).checkState() == Qt.CheckState.Checked]
        attrs = tuple(a for a, cb in self.attrs.items() if cb.isChecked())
        files_ = tuple(k for k, cb in self.file_props.items() if cb.isChecked())
        return dp.copy_properties(self.source, ds, pairs, attrs, files_,
                                  replace_labels=not self.merge.isChecked())


def copy_data_properties(editor) -> None:
    source, _ = _other_dataset(editor, "Copy Data Properties")
    if source is None:
        return
    dlg = CopyPropertiesDialog(source, editor.ds, editor)
    if dlg.exec():
        section = editor.doc.mutate(dlg.run, "Copy Data Properties")
        editor.ctx.show_output(section)


def compare_datasets(editor) -> None:
    other, desc = _other_dataset(editor, "Compare Datasets")
    if other is None:
        return
    common = [v.name for v in editor.ds.variables if other.has_name(v.name)]
    if not common:
        QMessageBox.information(editor, "Compare Datasets", "The datasets share no variables.")
        return
    ids = _pick_variables(editor.ds, "Compare Datasets: Case ID", editor,
                          "Case ID variables (none = compare cases by position):") or []
    flag = QMessageBox.question(editor, "Compare Datasets",
                                "Create a variable flagging mismatched cases (CasesCompare)?"
                                ) == QMessageBox.StandardButton.Yes
    if flag:
        section = editor.doc.mutate(lambda ds: dp.compare_datasets(ds, other, ids,
                                                                   flag="CasesCompare"),
                                    "Compare Datasets")
    else:
        section = dp.compare_datasets(editor.ds, other, ids)
    section.note("", f"Compared with: {desc}")
    editor.ctx.show_output(section)


# ================================================================== sort / split / copy
class SortVariablesDialog(QDialog):
    def __init__(self, ds: Dataset, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sort Variables")
        self.key = QListWidget()
        for k in dp.SORT_KEYS:
            self.key.addItem({"value_labels": "Values", "missing": "Missing"}.get(
                k, k.capitalize()))
        self.key.setCurrentRow(0)
        self.asc = QRadioButton("Ascending")
        self.desc = QRadioButton("Descending")
        self.asc.setChecked(True)
        self.save = QCheckBox("Save current (pre-sorted) order in attribute:")
        self.attr = QLineEdit("OriginalOrder")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Variable View Columns:"))
        layout.addWidget(self.key)
        row = QHBoxLayout()
        row.addWidget(self.asc)
        row.addWidget(self.desc)
        layout.addLayout(row)
        r2 = QHBoxLayout()
        r2.addWidget(self.save)
        r2.addWidget(self.attr)
        layout.addLayout(r2)
        layout.addWidget(_buttons(self))

    def apply(self, ds: Dataset) -> None:
        dp.sort_variables(ds, dp.SORT_KEYS[self.key.currentRow()], self.asc.isChecked(),
                          self.attr.text().strip() if self.save.isChecked() else None)


def sort_variables(editor) -> None:
    dlg = SortVariablesDialog(editor.ds, editor)
    if dlg.exec():
        editor.doc.mutate(dlg.apply, "Sort Variables")


def split_into_files(editor) -> None:
    names = _pick_variables(editor.ds, "Split into Files", editor, "Split cases by:")
    if not names:
        return
    start = editor.ctx.settings.value("last_dir", str(Path.home()))
    folder = QFileDialog.getExistingDirectory(editor, "Split into Files: Output folder", start)
    if not folder:
        return
    _, section = dp.split_into_files(editor.ds, names, folder)
    editor.ctx.show_output(section)


def copy_dataset(editor) -> None:
    from rsstatchem.core.dataset import Variable

    ds = editor.ds
    new = Dataset(ds.df.copy(), [Variable.from_dict(v.to_dict()) for v in ds.variables])
    for v_new, v_old in zip(new.variables, ds.variables):
        v_new.value_labels = dict(v_old.value_labels)
    new.meta = copy.deepcopy(ds.meta)
    new.modified = True
    editor.ctx.new_editor(new)


# ================================================================== Create Dummy Variables
class CreateDummiesDialog(VariablePickerDialog):
    modifies_data = True
    DEFAULTS = {"omit": "none", "labels": True, "roots": ""}

    def __init__(self, ds: Dataset, parent=None):
        super().__init__(ds, "Create Dummy Variables", parent)
        self.add_target("vars", "Create Dummy Variables for:")
        form = QFormLayout()
        self.roots = QLineEdit(self.opts["roots"])
        self.roots.setPlaceholderText("Root names, one per variable (blank = variable name)")
        self.roots.textChanged.connect(lambda t: self.opts.__setitem__("roots", t))
        form.addRow("Root names:", self.roots)
        self.omit = QComboBox()
        for key, text in (("none", "Create a dummy for every category"),
                          ("first", "Omit the first category (reference)"),
                          ("last", "Omit the last category (reference)")):
            self.omit.addItem(text, key)
        self.omit.setCurrentIndex(["none", "first", "last"].index(self.opts["omit"]))
        self.omit.currentIndexChanged.connect(
            lambda _: self.opts.__setitem__("omit", self.omit.currentData()))
        form.addRow("Categories:", self.omit)
        self.options_layout.addLayout(form)
        self.add_checkbox("labels", "Use value labels in the dummy variable labels")
        self.finish()

    def compute(self):
        roots = [r for r in self.opts["roots"].replace(",", " ").split()]
        return dp.create_dummies(self.ds, self.selected("vars"), roots or None,
                                 self.opts["omit"], self.opts["labels"])


# ================================================================== Date and Time Wizard
class DateTimeWizard(QDialog):
    TASKS = [("parse", "Create a date/time variable from a string containing a date"),
             ("parts", "Create a date/time variable from variables holding parts of dates"),
             ("diff", "Calculate with dates: difference between two dates"),
             ("add", "Calculate with dates: add or subtract a duration"),
             ("extract", "Extract a part of a date/time variable (year, month, weekday…)")]
    UNITS = ("years", "quarters", "months", "weeks", "days", "hours", "minutes", "seconds")

    def __init__(self, ds: Dataset, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Date and Time Wizard")
        self.resize(640, 460)
        self.ds = ds
        self.group = QButtonGroup(self)
        tasks = QGroupBox("What do you want to do?")
        tl = QVBoxLayout(tasks)
        for i, (_, text) in enumerate(self.TASKS):
            rb = QRadioButton(text)
            self.group.addButton(rb, i)
            tl.addWidget(rb)
        self.group.button(0).setChecked(True)
        self.stack = QStackedWidget()
        strings = [v.name for v in ds.variables if v.type == STRING]
        dates = [v.name for v in ds.variables if v.type in (DATE, NUMERIC)]
        nums = [v.name for v in ds.variables if v.type == NUMERIC]

        def combo(items, blank=False):
            c = QComboBox()
            if blank:
                c.addItem("(none)", "")
            for n in items:
                c.addItem(n, n)
            return c

        self.w = {}
        # parse
        page = QWidget()
        f = QFormLayout(page)
        self.w["p_src"] = combo(strings)
        self.w["p_fmt"] = QComboBox()
        for label, code in dp.INPUT_FORMATS.items():
            self.w["p_fmt"].addItem(label, code)
        f.addRow("String variable:", self.w["p_src"])
        f.addRow("Pattern:", self.w["p_fmt"])
        self.stack.addWidget(page)
        # parts
        page = QWidget()
        f = QFormLayout(page)
        for key, text in (("y", "Year:"), ("m", "Month:"), ("d", "Day of month:"),
                          ("h", "Hours:"), ("mi", "Minutes:"), ("s", "Seconds:")):
            self.w[f"t_{key}"] = combo(nums, blank=key != "y")
            f.addRow(text, self.w[f"t_{key}"])
        self.stack.addWidget(page)
        # diff
        page = QWidget()
        f = QFormLayout(page)
        self.w["d_a"] = combo(dates)
        self.w["d_b"] = combo(dates)
        self.w["d_unit"] = combo(self.UNITS)
        self.w["d_unit"].setCurrentText("days")
        f.addRow("Later date:", self.w["d_a"])
        f.addRow("Minus earlier date:", self.w["d_b"])
        f.addRow("Unit (whole units):", self.w["d_unit"])
        self.stack.addWidget(page)
        # add
        page = QWidget()
        f = QFormLayout(page)
        self.w["a_src"] = combo(dates)
        self.w["a_amount"] = QLineEdit("1")
        self.w["a_unit"] = combo(self.UNITS)
        self.w["a_unit"].setCurrentText("days")
        f.addRow("Date:", self.w["a_src"])
        f.addRow("Duration (number or variable; negative subtracts):", self.w["a_amount"])
        f.addRow("Unit:", self.w["a_unit"])
        self.stack.addWidget(page)
        # extract
        page = QWidget()
        f = QFormLayout(page)
        self.w["e_src"] = combo(dates)
        self.w["e_part"] = combo(dp.DATE_PARTS)
        f.addRow("Date/time variable:", self.w["e_src"])
        f.addRow("Unit to extract:", self.w["e_part"])
        self.stack.addWidget(page)
        self.group.idClicked.connect(self.stack.setCurrentIndex)
        self.target = QLineEdit()
        self.label = QLineEdit()
        form = QFormLayout()
        form.addRow("Result variable:", self.target)
        form.addRow("Variable label:", self.label)
        layout = QVBoxLayout(self)
        layout.addWidget(tasks)
        layout.addWidget(self.stack, 1)
        layout.addLayout(form)
        layout.addWidget(_buttons(self))

    def task(self) -> str:
        return self.TASKS[self.group.checkedId()][0]

    def accept(self) -> None:
        if not self.target.text().strip():
            QMessageBox.warning(self, self.windowTitle(), "Enter a name for the result.")
            return
        super().accept()

    def apply(self, ds: Dataset):
        t, target, label = self.task(), self.target.text().strip(), self.label.text().strip()
        w = self.w
        if t == "parse":
            return dp.wizard_parse(ds, w["p_src"].currentData(), target,
                                   w["p_fmt"].currentData(), label)
        if t == "parts":
            g = {k: w[f"t_{k}"].currentData() or None for k in ("y", "m", "d", "h", "mi", "s")}
            return dp.wizard_from_parts(ds, target, g["y"], g["m"], g["d"], g["h"], g["mi"],
                                        g["s"], label)
        if t == "diff":
            return dp.wizard_difference(ds, w["d_a"].currentData(), w["d_b"].currentData(),
                                        target, w["d_unit"].currentData(), label)
        if t == "add":
            amount = w["a_amount"].text().strip()
            try:
                value: float | str = float(amount.replace(",", "."))
            except ValueError:
                value = amount
            return dp.wizard_add(ds, w["a_src"].currentData(), target, value,
                                 w["a_unit"].currentData(), label)
        return dp.wizard_extract(ds, w["e_src"].currentData(), target,
                                 w["e_part"].currentData(), label)


def date_time_wizard(editor) -> None:
    if not editor.ds.n_vars:
        return
    dlg = DateTimeWizard(editor.ds, editor)
    if dlg.exec():
        section = editor.doc.mutate(dlg.apply, "Date and Time Wizard")
        editor.ctx.show_output(section)


# ================================================================== Random Number Generators
class RandomGeneratorsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Random Number Generators")
        state = X._RNG_HOLDER
        gen = QGroupBox("Active Generator")
        gl = QVBoxLayout(gen)
        self.mt = QRadioButton("Mersenne Twister (PCG64 stream)")
        self.compat = QRadioButton("SPSS 12 Compatible (same stream; for reproducible seeds)")
        (self.compat if state["generator"].startswith("SPSS") else self.mt).setChecked(True)
        gl.addWidget(self.mt)
        gl.addWidget(self.compat)
        start = QGroupBox("Active Generator Initialization")
        sl = QVBoxLayout(start)
        self.random = QRadioButton("Random")
        self.fixed = QRadioButton("Fixed Value:")
        self.seed = QSpinBox()
        self.seed.setRange(1, 2_000_000_000)
        self.seed.setValue(state["seed"] or 2000000)
        (self.fixed if state["seed"] is not None else self.random).setChecked(True)
        sl.addWidget(self.random)
        row = QHBoxLayout()
        row.addWidget(self.fixed)
        row.addWidget(self.seed)
        sl.addLayout(row)
        layout = QVBoxLayout(self)
        layout.addWidget(gen)
        layout.addWidget(start)
        layout.addWidget(QLabel("Applies to RV.* and the random functions of Compute Variable "
                                "and to random case samples."))
        layout.addWidget(_buttons(self))

    def apply(self) -> None:
        X.set_seed(self.seed.value() if self.fixed.isChecked() else None,
                   "SPSS 12 Compatible" if self.compat.isChecked() else "Mersenne Twister")


def random_generators(editor) -> None:
    dlg = RandomGeneratorsDialog(editor)
    if dlg.exec():
        dlg.apply()
        seed = X._RNG_HOLDER["seed"]
        editor.statusBar().showMessage(
            f"Random number seed: {seed}" if seed is not None else "Random seed: random", 5000)


# ================================================================== Utilities
class VariablesDialog(QDialog):
    """Utilities > Variables: variable information browser with Go To."""

    def __init__(self, editor):
        super().__init__(editor)
        self.setWindowTitle("Variables")
        self.resize(640, 460)
        self.editor = editor
        ds = editor.ds
        self.list = QListWidget()
        for v in ds.variables:
            self.list.addItem(variable_item(v))
        self.info = QTextBrowser()
        self.list.currentRowChanged.connect(self._show)
        goto = QPushButton("Go To")
        goto.clicked.connect(self._goto)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        body = QHBoxLayout()
        body.addWidget(self.list, 1)
        body.addWidget(self.info, 2)
        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(goto)
        buttons.addWidget(close)
        layout = QVBoxLayout(self)
        layout.addLayout(body, 1)
        layout.addLayout(buttons)
        if ds.n_vars:
            self.list.setCurrentRow(0)

    def _show(self, row: int) -> None:
        if row < 0:
            return
        v = self.editor.ds.variables[row]
        labels = "".join(f"<tr><td>{format_code(k)}</td><td>{t}</td></tr>"
                         for k, t in v.value_labels.items())
        missing = ", ".join(format_code(m) for m in v.missing) or "None"
        self.info.setHtml(
            f"<b>{v.name}</b><br>Label: {v.label or '&lt;none&gt;'}<br>Type: {v.type} "
            f"(width {v.width}, {v.decimals} decimals)<br>Missing Values: {missing}<br>"
            f"Measurement Level: {v.measure.capitalize()}<br>Role: {v.role.capitalize()}"
            + (f"<br><br>Value Labels:<table>{labels}</table>" if labels else ""))

    def _goto(self) -> None:
        row = self.list.currentRow()
        if row >= 0:
            self.editor.go_to_variable_index(row)
        self.accept()


def variables_browser(editor) -> None:
    VariablesDialog(editor).exec()


def data_file_comments(editor) -> None:
    dlg = QDialog(editor)
    dlg.setWindowTitle("Data File Comments")
    dlg.resize(560, 360)
    edit = QPlainTextEdit(dp.comments(editor.ds))
    show = QCheckBox("Display comments in output")
    layout = QVBoxLayout(dlg)
    layout.addWidget(QLabel("Comments (saved with the data file):"))
    layout.addWidget(edit, 1)
    layout.addWidget(show)
    layout.addWidget(_buttons(dlg))
    if dlg.exec():
        text = edit.toPlainText()
        editor.doc.mutate(lambda ds: dp.set_comments(ds, text), "Data File Comments")
        if show.isChecked() and text.strip():
            from rsstatchem.core.output import Section

            section = Section("Data File Comments")
            section.log("ADD DOCUMENT\n  '" + text.replace("\n", "'\n  '") + "'.")
            section.note("", text)
            editor.ctx.show_output(section)


class VariableSetsDialog(QDialog):
    def __init__(self, ds: Dataset, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Define Variable Sets")
        self.resize(720, 460)
        self.ds = ds
        self.sets = dp.variable_sets(ds)
        self.variables = QListWidget()
        self.variables.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        for v in ds.variables:
            self.variables.addItem(variable_item(v))
        self.members = QListWidget()
        self.members.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        add, remove = QPushButton("▶"), QPushButton("◀")
        add.setObjectName("arrow")
        remove.setObjectName("arrow")
        add.clicked.connect(self._add)
        remove.clicked.connect(lambda: [self.members.takeItem(self.members.row(i))
                                        for i in self.members.selectedItems()])
        self.name = QLineEdit()
        self.set_list = QListWidget()
        self.set_list.currentTextChanged.connect(self._load)
        b_add, b_remove = QPushButton("Add Set"), QPushButton("Remove Set")
        b_add.clicked.connect(self._add_set)
        b_remove.clicked.connect(self._remove_set)
        self._refresh()
        mid = QVBoxLayout()
        mid.addWidget(QLabel("Set Name:"))
        mid.addWidget(self.name)
        mid.addWidget(QLabel("Variables in Set:"))
        mid.addWidget(self.members, 1)
        arrows = QVBoxLayout()
        arrows.addStretch()
        arrows.addWidget(add)
        arrows.addWidget(remove)
        arrows.addStretch()
        right = QVBoxLayout()
        right.addWidget(QLabel("Variable Sets:"))
        right.addWidget(self.set_list, 1)
        right.addWidget(b_add)
        right.addWidget(b_remove)
        body = QHBoxLayout()
        body.addWidget(self.variables, 2)
        body.addLayout(arrows)
        body.addLayout(mid, 2)
        body.addLayout(right, 2)
        layout = QVBoxLayout(self)
        layout.addLayout(body, 1)
        layout.addWidget(_buttons(self))

    def _add(self) -> None:
        present = {self.members.item(i).data(NAME_ROLE) for i in range(self.members.count())}
        for item in self.variables.selectedItems():
            if item.data(NAME_ROLE) not in present:
                self.members.addItem(variable_item(self.ds.var(item.data(NAME_ROLE))))

    def _refresh(self) -> None:
        self.set_list.blockSignals(True)
        self.set_list.clear()
        self.set_list.addItems(list(self.sets))
        self.set_list.blockSignals(False)

    def _add_set(self) -> None:
        name = self.name.text().strip()
        members = [self.members.item(i).data(NAME_ROLE) for i in range(self.members.count())]
        if not name or not members:
            QMessageBox.warning(self, self.windowTitle(), "Enter a set name and variables.")
            return
        self.sets[name] = members
        self._refresh()

    def _remove_set(self) -> None:
        item = self.set_list.currentItem()
        if item is not None:
            self.sets.pop(item.text(), None)
            self._refresh()

    def _load(self, name: str) -> None:
        if name not in self.sets:
            return
        self.name.setText(name)
        self.members.clear()
        for v in self.sets[name]:
            if self.ds.has_name(v):
                self.members.addItem(variable_item(self.ds.var(v)))


def define_variable_sets(editor) -> None:
    dlg = VariableSetsDialog(editor.ds, editor)
    if dlg.exec():
        sets = dlg.sets
        editor.doc.mutate(lambda ds: dp.set_variable_sets(ds, sets), "Define Variable Sets")
        editor.apply_variable_sets()


def use_variable_sets(editor) -> None:
    sets = dp.variable_sets(editor.ds)
    if not sets:
        QMessageBox.information(editor, "Use Variable Sets",
                                "No variable sets are defined (Utilities > Define Variable "
                                "Sets).")
        return
    dlg = QDialog(editor)
    dlg.setWindowTitle("Use Variable Sets")
    lst = QListWidget()
    active = set(editor.ds.meta.get("varsets_active", []))
    for name in sets:
        item = QListWidgetItem(f"{name}")
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked if name in active or not active
                           else Qt.CheckState.Unchecked)
        lst.addItem(item)
    layout = QVBoxLayout(dlg)
    layout.addWidget(QLabel("Select variable sets to apply (Data View shows their "
                            "variables only):"))
    layout.addWidget(lst)
    layout.addWidget(_buttons(dlg))
    if dlg.exec():
        chosen = [lst.item(i).text() for i in range(lst.count())
                  if lst.item(i).checkState() == Qt.CheckState.Checked]
        dp.use_variable_sets(editor.ds, chosen)
        editor.apply_variable_sets()


def show_all_variables(editor) -> None:
    dp.use_variable_sets(editor.ds, None)
    editor.apply_variable_sets()


MENUS = {
    "Data": [("Define Variable Properties...", define_properties),
             ("Copy Data Properties...", copy_data_properties),
             ("Compare Datasets...", compare_datasets),
             ("Sort Variables...", sort_variables),
             ("Split into Files...", split_into_files),
             ("Copy Dataset", copy_dataset)],
    "Transform": [("Create Dummy Variables...", CreateDummiesDialog),
                  ("Date and Time Wizard...", date_time_wizard),
                  ("Random Number Generators...", random_generators)],
    "Utilities": [("Variables...", variables_browser),
                  ("Data File Comments...", data_file_comments),
                  ("Define Variable Sets...", define_variable_sets),
                  ("Use Variable Sets...", use_variable_sets),
                  ("Show All Variables", show_all_variables)],
}


def selftest() -> None:
    df = pd.DataFrame({"tgl": ["17/08/1945", "01/01/2000", "bad"],
                       "g": [1.0, 2.0, 1.0], "x": [1.5, 2.5, np.nan]})
    ds = Dataset(df)
    dp.wizard_parse(ds, "tgl", "tanggal", "%d/%m/%Y")
    assert ds.var("tanggal").type == DATE and ds.df["tanggal"].isna().sum() == 1
    dp.wizard_extract(ds, "tanggal", "tahun", "year")
    assert list(ds.df["tahun"][:2]) == [1945.0, 2000.0]
    dp.create_dummies(ds, ["g"])
    assert list(ds.df["g_1"]) == [1.0, 0.0, 1.0]
    dp.sort_variables(ds, "name")
    assert ds.names() == sorted(ds.names(), key=str.lower)
    assert dp.find_cells(ds, "2.50", [ds.index_of("x")]) == [(1, ds.index_of("x"))]

"""SPSS Transform and Data menu dialogs beyond Compute / Recode.

Transform: Count Values within Cases, Automatic Recode, Visual Binning, Rank Cases, Replace
Missing Values, Shift Values. Data: Identify Duplicate Cases, Merge Files (Add Cases, Add
Variables), Aggregate, Transpose, Restructure (Variables to Cases / Cases to Variables).
The layouts, captions and defaults follow the SPSS Statistics dialogs; the computations live
in :mod:`rsstatchem.core.data_ops`.
"""
from __future__ import annotations

import copy
import io
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QLocale, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QInputDialog,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
                               QPushButton, QRadioButton, QSpinBox, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from rsstatchem.core import data_ops as do
from rsstatchem.core.dataset import NUMERIC, STRING, Dataset, Variable, format_code, is_valid_name
from rsstatchem.ui.dialogs.base import (NAME_ROLE, SubDialog, VariablePickerDialog, is_numeric,
                                        variable_item)
from rsstatchem.ui.dialogs.transform import IfDialog

SPEC_ROLE = Qt.ItemDataRole.UserRole + 10
GREY = "color:#555;"
_windows: list = []  # strong references to top-level windows opened from here


def _if_label(condition: str) -> str:
    return f"If: {condition}" if condition else "(optional case selection condition)"


def _continue_buttons(dlg: QDialog, accept=None) -> QDialogButtonBox:
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                               | QDialogButtonBox.StandardButton.Cancel)
    buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Continue")
    buttons.accepted.connect(accept or dlg.accept)
    buttons.rejected.connect(dlg.reject)
    return buttons


def _parse_number(text: str) -> float:
    text = text.strip()
    if not text:
        raise ValueError("Enter a value.")
    return float(text.replace(",", "."))


def _radio_group(parent: QWidget, layout, options: list[tuple[object, str]], current
                 ) -> dict:
    group = QButtonGroup(parent)
    buttons = {}
    for value, text in options:
        rb = QRadioButton(text)
        rb.setChecked(value == current)
        group.addButton(rb)
        layout.addWidget(rb)
        buttons[value] = rb
    if not any(b.isChecked() for b in buttons.values()):
        next(iter(buttons.values())).setChecked(True)
    return buttons


def _checked(buttons: dict):
    return next(v for v, b in buttons.items() if b.isChecked())


def _editor_ctx(widget: QWidget):
    parent = widget.parent()
    return getattr(parent, "ctx", None)


def _open_new_dataset(widget: QWidget, dataset: Dataset) -> None:
    ctx = _editor_ctx(widget)
    if ctx is None:
        raise RuntimeError("No application context to open the new dataset.")
    ctx.new_editor(dataset)


# ================================================================== value specification
class ValuesSpecDialog(QDialog):
    """'Count Values within Cases: Values to Count' — the Old Value panel of Recode."""

    def __init__(self, numeric: bool, specs: list[tuple], parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Count Values within Cases: Values to Count")
        self.numeric = numeric
        self.specs = list(specs)
        box = QGroupBox("Value")
        grid = QGridLayout(box)
        self.r_value, self.e_value = QRadioButton("Value:"), QLineEdit()
        self.r_sysmis = QRadioButton("System-missing")
        self.r_missing = QRadioButton("System- or user-missing")
        self.r_range = QRadioButton("Range:")
        self.e_lo, self.e_hi = QLineEdit(), QLineEdit()
        self.r_lowest, self.e_lowest = QRadioButton("Range, LOWEST through value:"), QLineEdit()
        self.r_highest, self.e_highest = QRadioButton("Range, value through HIGHEST:"), \
            QLineEdit()
        group = QButtonGroup(self)
        range_row = QWidget()
        rr = QHBoxLayout(range_row)
        rr.setContentsMargins(22, 0, 0, 0)
        rr.addWidget(self.e_lo)
        rr.addWidget(QLabel("through"))
        rr.addWidget(self.e_hi)
        rows = [(self.r_value, self.e_value), (self.r_sysmis, None), (self.r_missing, None),
                (self.r_range, None), (None, range_row), (self.r_lowest, None),
                (None, self.e_lowest), (self.r_highest, None), (None, self.e_highest)]
        for r, (radio, fld) in enumerate(rows):
            if radio is not None:
                group.addButton(radio)
                grid.addWidget(radio, r, 0, 1, 2 if fld is None else 1)
            if fld is not None:
                if radio is None and fld is not range_row:
                    fld.setContentsMargins(22, 0, 0, 0)
                grid.addWidget(fld, r, 1 if radio is not None else 0, 1, 1 if radio else 2)
        grid.setRowStretch(len(rows), 1)
        self.r_value.setChecked(True)
        if not numeric:
            for w in (self.r_sysmis, self.r_range, self.e_lo, self.e_hi, self.r_lowest,
                      self.e_lowest, self.r_highest, self.e_highest):
                w.setEnabled(False)
        for edit, radio in ((self.e_value, self.r_value), (self.e_lo, self.r_range),
                            (self.e_hi, self.r_range), (self.e_lowest, self.r_lowest),
                            (self.e_highest, self.r_highest)):
            edit.textEdited.connect(lambda _, r=radio: r.setChecked(True))

        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._pick)
        add, change, remove = QPushButton("Add"), QPushButton("Change"), QPushButton("Remove")
        add.clicked.connect(self._add)
        change.clicked.connect(self._change)
        remove.clicked.connect(self._remove)
        col = QVBoxLayout()
        for b in (add, change, remove):
            col.addWidget(b)
        col.addStretch()
        right = QVBoxLayout()
        right.addWidget(QLabel("Values to Count:"))
        lr = QHBoxLayout()
        lr.addLayout(col)
        lr.addWidget(self.list, 1)
        right.addLayout(lr, 1)
        self.error = QLabel()
        self.error.setStyleSheet("color:#B71C1C;")
        body = QHBoxLayout()
        body.addWidget(box, 1)
        body.addLayout(right, 1)
        layout = QVBoxLayout(self)
        layout.addLayout(body, 1)
        layout.addWidget(self.error)
        layout.addWidget(_continue_buttons(self, self._accept))
        self._refresh()

    def _value(self, text: str):
        if self.numeric:
            return _parse_number(text)
        text = text.strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
            text = text[1:-1]
        return text

    def _current(self) -> tuple:
        if self.r_value.isChecked():
            return ("value", self._value(self.e_value.text()))
        if self.r_sysmis.isChecked():
            return ("sysmis",)
        if self.r_missing.isChecked():
            return ("missing",)
        if self.r_range.isChecked():
            lo, hi = _parse_number(self.e_lo.text()), _parse_number(self.e_hi.text())
            if lo > hi:
                raise ValueError("The lower range value is larger than the upper one.")
            return ("range", lo, hi)
        if self.r_lowest.isChecked():
            return ("lowest", _parse_number(self.e_lowest.text()))
        return ("highest", _parse_number(self.e_highest.text()))

    def _add(self) -> None:
        try:
            spec = self._current()
        except ValueError as exc:
            self.error.setText(str(exc) or "Values must be numbers.")
            return
        self.error.clear()
        if spec not in self.specs:
            self.specs.append(spec)
        self._refresh()
        self.e_value.clear()
        self.e_value.setFocus()

    def _change(self) -> None:
        row = self.list.currentRow()
        if row < 0:
            return
        try:
            self.specs[row] = self._current()
        except ValueError as exc:
            self.error.setText(str(exc) or "Values must be numbers.")
            return
        self._refresh()

    def _remove(self) -> None:
        row = self.list.currentRow()
        if row >= 0:
            del self.specs[row]
            self._refresh()

    def _refresh(self) -> None:
        self.list.clear()
        for s in self.specs:
            self.list.addItem(do.spec_text(s))

    def _pick(self, row: int) -> None:
        if not 0 <= row < len(self.specs):
            return
        s = self.specs[row]
        {"value": self.r_value, "sysmis": self.r_sysmis, "missing": self.r_missing,
         "range": self.r_range, "lowest": self.r_lowest, "highest": self.r_highest}[
            s[0]].setChecked(True)
        if s[0] == "value":
            self.e_value.setText(format_code(s[1]))
        elif s[0] == "range":
            self.e_lo.setText(format_code(s[1]))
            self.e_hi.setText(format_code(s[2]))
        elif s[0] == "lowest":
            self.e_lowest.setText(format_code(s[1]))
        elif s[0] == "highest":
            self.e_highest.setText(format_code(s[1]))

    def _accept(self) -> None:
        if not self.specs:
            self.error.setText("Add at least one value to count.")
            return
        self.accept()


# ================================================================== COUNT
class CountValuesDialog(VariablePickerDialog):
    modifies_data = True
    DEFAULTS = {"target": "", "label": "", "specs": [], "condition": ""}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Count Occurrences of Values within Cases", parent)
        self.target_edit = QLineEdit(self.opts["target"])
        self.label_edit = QLineEdit(self.opts["label"])
        top = QGridLayout()
        top.addWidget(QLabel("Target Variable:"), 0, 0)
        top.addWidget(QLabel("Target Label:"), 0, 1)
        top.addWidget(self.target_edit, 1, 0)
        top.addWidget(self.label_edit, 1, 1)
        self.layout().insertLayout(0, top)
        self.add_target("vars", "Variables:")
        self.add_option_button("Define Values...", self._define)
        self.add_option_button("If...", self._if)
        self.values_label = QLabel()
        self.values_label.setStyleSheet(GREY)
        self.values_label.setWordWrap(True)
        self.if_label = QLabel(_if_label(self.opts["condition"]))
        self.if_label.setStyleSheet(GREY)
        self.options_layout.addWidget(self.values_label)
        self.options_layout.addWidget(self.if_label)
        self._show_values()
        self.finish()

    def _show_values(self) -> None:
        specs = self.opts["specs"]
        self.values_label.setText("Values to count: " + (", ".join(do.spec_text(s)
                                                              for s in specs)
                                                         if specs else "(none — click Define "
                                                                       "Values…)"))

    def _numeric(self) -> bool:
        names = self.selected("vars")
        return not names or self.ds.var(names[0]).type != STRING

    def _define(self) -> None:
        dlg = ValuesSpecDialog(self._numeric(), self.opts["specs"], self)
        if dlg.exec():
            self.opts["specs"] = dlg.specs
            self._show_values()

    def _if(self) -> None:
        dlg = IfDialog(self.ds, self.opts["condition"], self, "Count Values within Cases")
        if dlg.exec():
            self.opts["condition"] = dlg.condition()
            self.if_label.setText(_if_label(self.opts["condition"]))

    def validate(self):
        name = self.target_edit.text().strip()
        if not is_valid_name(name):
            return "Enter a valid name for the target variable."
        kinds = {self.ds.var(n).type == STRING for n in self.selected("vars")}
        if len(kinds) > 1:
            return "The variables must be all numeric or all string."
        if not self.opts["specs"]:
            return "Click Define Values… and add at least one value to count."
        string = kinds.pop() if kinds else False
        for s in self.opts["specs"]:
            if s[0] == "value" and isinstance(s[1], str) != string:
                return "The values to count do not match the variable type. Define them again."
            if string and s[0] in ("range", "lowest", "highest", "sysmis"):
                return "Ranges and system-missing apply to numeric variables only."
        if self.ds.has_name(name):
            answer = QMessageBox.question(self, self.windowTitle(),
                                          f"Change existing variable {self.ds.var(name).name}?")
            if answer != QMessageBox.StandardButton.Yes:
                return "Choose another target variable name."
        self.opts.update(target=name, label=self.label_edit.text().strip())
        return None

    def compute(self):
        o = self.opts
        return do.count_values(self.ds, o["target"], self.selected("vars"), o["specs"],
                               o["label"], o["condition"] or None)


# ================================================================== input --> output lists
class _OutputNamesMixin:
    """Input --> output name list (Automatic Recode, Replace Missing Values, Shift Values)."""

    var_list: QListWidget

    def _outputs(self) -> dict:
        return self.opts.setdefault("outputs", {})

    def _default_spec(self, src: str, taken: set[str]) -> dict:  # pragma: no cover
        raise NotImplementedError

    def _item_text(self, src: str, spec: dict) -> str:  # pragma: no cover
        raise NotImplementedError

    def targets_changed(self) -> None:
        if not hasattr(self, "var_list"):
            return
        outputs = self._outputs()
        names = self.selected("vars")
        taken = {outputs[s]["name"].lower() for s in names if s in outputs}
        for i in range(self.var_list.count()):
            item = self.var_list.item(i)
            src = item.data(NAME_ROLE)
            if src not in outputs or self.ds.has_name(outputs[src]["name"]):
                outputs[src] = self._default_spec(src, taken)
                taken.add(outputs[src]["name"].lower())
            item.setText(self._item_text(src, outputs[src]))

    def _current_source(self) -> str | None:
        item = self.var_list.currentItem() or (self.var_list.item(0)
                                               if self.var_list.count() == 1 else None)
        return item.data(NAME_ROLE) if item is not None else None

    def _check_output_names(self) -> str | None:
        outputs = self._outputs()
        names = [outputs[s]["name"] for s in self.selected("vars")]
        for n in names:
            if not is_valid_name(n):
                return f"'{n}' is not a valid variable name."
            if self.ds.has_name(n):
                return f"A variable named {n} already exists."
        if len({n.lower() for n in names}) != len(names):
            return "The new variable names must be different."
        return None


# ================================================================== AUTORECODE
class AutoRecodeDialog(_OutputNamesMixin, VariablePickerDialog):
    modifies_data = True
    DEFAULTS = {"outputs": {}, "descending": False, "group": False, "blank_missing": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Automatic Recode", parent)
        self.var_list = self.add_target("vars", "Variable -> New Name")
        self.var_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.var_list.currentItemChanged.connect(lambda *_: self._load())
        row = QHBoxLayout()
        row.addWidget(QLabel("New Name:"))
        self.name_edit = QLineEdit()
        row.addWidget(self.name_edit, 1)
        add = QPushButton("Add New Name")
        add.clicked.connect(self._set_name)
        row.addWidget(add)
        self.options_layout.addLayout(row)
        box = QGroupBox("Recode Starting from")
        bl = QHBoxLayout(box)
        self.order = _radio_group(self, bl, [(False, "Lowest value"), (True, "Highest value")],
                                  self.opts["descending"])
        bl.addStretch()
        self.options_layout.addWidget(box)
        self.add_checkbox("group", "Use the same recoding scheme for all variables")
        self.add_checkbox("blank_missing", "Treat blank string values as user-missing")
        self.finish()

    def _default_spec(self, src, taken):
        return {"name": do.unique_name(self.ds, f"{src}_rec", taken)}

    def _item_text(self, src, spec):
        return f"{src} --> {spec['name']}"

    def _load(self) -> None:
        src = self._current_source()
        if src:
            self.name_edit.setText(self._outputs().get(src, {}).get("name", ""))

    def _set_name(self) -> None:
        src = self._current_source()
        name = self.name_edit.text().strip()
        if src is None:
            QMessageBox.information(self, self.windowTitle(), "Select a variable first.")
            return
        if not is_valid_name(name) or self.ds.has_name(name):
            QMessageBox.warning(self, self.windowTitle(),
                                f"'{name}' is not a valid new variable name.")
            return
        self._outputs()[src] = {"name": name}
        self.targets_changed()

    def validate(self):
        self.opts["descending"] = _checked(self.order)
        if self.opts["group"]:
            kinds = {self.ds.var(n).type == STRING for n in self.selected("vars")}
            if len(kinds) > 1:
                return "With the same recoding scheme, all variables must be numeric or all " \
                       "string."
        return self._check_output_names()

    def compute(self):
        o = self.opts
        src = self.selected("vars")
        return do.automatic_recode(self.ds, src, [o["outputs"][s]["name"] for s in src],
                                   o["descending"], o["blank_missing"], o["group"])


# ================================================================== VISUAL BINNING
class MakeCutpointsDialog(QDialog):
    def __init__(self, x: np.ndarray, decimals: int, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Make Cutpoints")
        self.x = x[~np.isnan(x)]
        self.cuts: list[float] = []
        lo, hi = (float(self.x.min()), float(self.x.max())) if len(self.x) else (0.0, 1.0)
        n0 = 3
        width0 = (hi - lo) / (n0 + 1) if hi > lo else 1.0
        dec = max(decimals, 0)

        def spin(value, minimum=-1e12, maximum=1e12, d=dec + 2):
            s = QDoubleSpinBox()
            s.setLocale(QLocale.c())
            s.setRange(minimum, maximum)
            s.setDecimals(d)
            s.setValue(value)
            return s

        self.r_width = QRadioButton("Equal Width Intervals")
        self.r_pct = QRadioButton("Equal Percentiles Based on Scanned Cases")
        self.r_sd = QRadioButton("Cutpoints at Mean and Selected Standard Deviations Based on "
                                 "Scanned Cases")
        group = QButtonGroup(self)
        for r in (self.r_width, self.r_pct, self.r_sd):
            group.addButton(r)
        self.r_width.setChecked(True)
        self.first = spin(round(lo + width0, dec + 2))
        self.n_width = QSpinBox()
        self.n_width.setRange(1, 999)
        self.n_width.setValue(n0)
        self.width = spin(round(width0, dec + 2), 1e-9)
        self.last = QLabel()
        wf = QFormLayout()
        wf.addRow("First Cutpoint Location:", self.first)
        wf.addRow("Number of Cutpoints:", self.n_width)
        wf.addRow("Width:", self.width)
        wf.addRow("Last Cutpoint Location:", self.last)
        self.n_pct = QSpinBox()
        self.n_pct.setRange(1, 99)
        self.n_pct.setValue(3)
        self.pct_width = QLabel()
        pf = QFormLayout()
        pf.addRow("Number of Cutpoints:", self.n_pct)
        pf.addRow("Width(%):", self.pct_width)
        self.sd = [QCheckBox(f"+/- {k} Std. Deviation") for k in (1, 2, 3)]
        self.sd[0].setChecked(True)
        sl = QVBoxLayout()
        for cb in self.sd:
            sl.addWidget(cb)
        for w in (self.first, self.width):
            w.valueChanged.connect(self._update)
        self.n_width.valueChanged.connect(self._update)
        self.n_pct.valueChanged.connect(self._update)
        layout = QVBoxLayout(self)
        layout.addWidget(self.r_width)
        box1 = QWidget()
        box1.setLayout(wf)
        wf.setContentsMargins(24, 0, 0, 0)
        layout.addWidget(box1)
        layout.addWidget(self.r_pct)
        box2 = QWidget()
        box2.setLayout(pf)
        pf.setContentsMargins(24, 0, 0, 0)
        layout.addWidget(box2)
        layout.addWidget(self.r_sd)
        box3 = QWidget()
        box3.setLayout(sl)
        sl.setContentsMargins(24, 0, 0, 0)
        layout.addWidget(box3)
        note = QLabel("Apply will replace the current cutpoint definitions with this "
                      "specification. A final interval will include all remaining values: "
                      "N cutpoints produce N+1 intervals.")
        note.setWordWrap(True)
        note.setStyleSheet(GREY)
        layout.addWidget(note)
        buttons = _continue_buttons(self, self._apply)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Apply")
        layout.addWidget(buttons)
        self._update()

    def _update(self) -> None:
        last = self.first.value() + (self.n_width.value() - 1) * self.width.value()
        self.last.setText(format_code(round(last, 8)))
        self.pct_width.setText(format_code(round(100 / (self.n_pct.value() + 1), 2)))

    def _apply(self) -> None:
        try:
            if self.r_width.isChecked():
                self.cuts = do.equal_width_cutpoints(self.first.value(), self.n_width.value(),
                                                     self.width.value())
            elif self.r_pct.isChecked():
                self.cuts = do.percentile_cutpoints(self.x, self.n_pct.value())
            else:
                ks = [k for k, cb in zip((1, 2, 3), self.sd) if cb.isChecked()]
                self.cuts = do.sd_cutpoints(self.x, ks)
        except do.DataError as exc:
            QMessageBox.warning(self, self.windowTitle(), str(exc))
            return
        self.accept()


def _histogram_png(x: np.ndarray, cuts: list[float], width: int = 520,
                   height: int = 170) -> bytes:
    fig = Figure(figsize=(width / 100, height / 100), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0.07, 0.18, 0.91, 0.76))
    x = x[~np.isnan(x)]
    if len(x):
        ax.hist(x, bins=min(30, max(5, int(np.sqrt(len(x))) * 2)), color="#EF9A9A",
                edgecolor="#B71C1C", linewidth=0.6)
    for c in cuts:
        ax.axvline(c, color="#212121", linewidth=1.2)
    ax.tick_params(labelsize=7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    return buf.getvalue()


class VisualBinningDialog(QDialog):
    modifies_data = True

    def __init__(self, ds: Dataset, parent: QWidget | None = None):
        super().__init__(parent)
        self.ds = ds
        self.setWindowTitle("Visual Binning")
        self.resize(900, 600)
        self.cuts: list[float] = []
        self.labels: list[str] = []
        self.auto_labels = True
        self._filling = False

        self.scanned = QListWidget()
        for var in ds.variables:
            if var.type == NUMERIC:
                self.scanned.addItem(variable_item(var))
        self.scanned.currentItemChanged.connect(lambda *_: self._load_variable())
        left = QVBoxLayout()
        left.addWidget(QLabel("Scanned Variable List:"))
        left.addWidget(self.scanned, 1)

        self.current = QLabel("")
        self.cur_label = QLabel("")
        self.bin_name = QLineEdit()
        self.bin_label = QLineEdit()
        names = QGridLayout()
        names.addWidget(QLabel("Name:"), 0, 1)
        names.addWidget(QLabel("Label:"), 0, 2)
        names.addWidget(QLabel("Current Variable:"), 1, 0)
        names.addWidget(self.current, 1, 1)
        names.addWidget(self.cur_label, 1, 2)
        names.addWidget(QLabel("Binned Variable:"), 2, 0)
        names.addWidget(self.bin_name, 2, 1)
        names.addWidget(self.bin_label, 2, 2)
        names.setColumnStretch(2, 1)

        self.range_label = QLabel("")
        self.range_label.setStyleSheet(GREY)
        self.hist = QLabel()
        self.hist.setMinimumHeight(170)
        self.hist.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hist.setStyleSheet("background:white; border:1px solid #ddd;")

        self.grid = QTableWidget(0, 3)
        self.grid.setHorizontalHeaderLabels(["Value", "Label", "Count"])
        self.grid.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.grid.verticalHeader().setDefaultSectionSize(22)
        self.grid.itemChanged.connect(self._grid_edited)
        make = QPushButton("Make Cutpoints...")
        make.clicked.connect(self._make_cutpoints)
        labels = QPushButton("Make Labels")
        labels.clicked.connect(self._make_labels)
        delete = QPushButton("Delete Cutpoint")
        delete.clicked.connect(self._delete_row)
        self.reverse = QCheckBox("Reverse scale")
        self.reverse.toggled.connect(self._refresh)
        upper = QGroupBox("Upper Endpoints")
        ul = QVBoxLayout(upper)
        self.upper = _radio_group(self, ul, [(True, "Included (<=)"), (False, "Excluded (<)")],
                                  True)
        self.upper[True].toggled.connect(self._upper_changed)
        side = QVBoxLayout()
        side.addWidget(upper)
        side.addWidget(make)
        side.addWidget(labels)
        side.addWidget(delete)
        side.addWidget(self.reverse)
        side.addStretch()
        grid_row = QHBoxLayout()
        grid_row.addWidget(self.grid, 1)
        grid_row.addLayout(side)

        right = QVBoxLayout()
        right.addLayout(names)
        right.addWidget(self.range_label)
        right.addWidget(self.hist)
        right.addWidget(QLabel("Grid: enter interval cutpoints or click Make Cutpoints for "
                               "automatic intervals. A cutpoint value of 10 defines an interval "
                               "starting above the previous interval and ending at 10."))
        right.itemAt(right.count() - 1).widget().setWordWrap(True)
        right.itemAt(right.count() - 1).widget().setStyleSheet(GREY)
        right.addLayout(grid_row, 1)

        body = QHBoxLayout()
        body.addLayout(left, 1)
        body.addLayout(right, 3)
        buttons = QDialogButtonBox()
        self.ok = buttons.addButton(QDialogButtonBox.StandardButton.Ok)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(body, 1)
        layout.addWidget(buttons)
        if self.scanned.count():
            self.scanned.setCurrentRow(0)
        else:
            self.ok.setEnabled(False)

    # ------------------------------------------------------------------ state
    def source(self) -> str | None:
        item = self.scanned.currentItem()
        return item.data(NAME_ROLE) if item is not None else None

    def _values(self) -> np.ndarray:
        return self.ds.analysis_series(self.source()).to_numpy(dtype=float)

    def _load_variable(self) -> None:
        name = self.source()
        if not name:
            return
        var = self.ds.var(name)
        self.current.setText(var.name)
        self.cur_label.setText(var.label)
        self.bin_name.setText(do.unique_name(self.ds, f"{var.name}_bin"))
        self.bin_label.setText(f"{var.title} (Binned)")
        x = self._values()
        valid = x[~np.isnan(x)]
        n_missing = int(np.isnan(x).sum())
        if len(valid):
            self.range_label.setText(f"Minimum: {format_code(float(valid.min()))}    "
                                     f"Maximum: {format_code(float(valid.max()))}    "
                                     f"Nonmissing Values: {len(valid)}    "
                                     f"Missing Values: {n_missing}")
        self.cuts, self.labels, self.auto_labels = [], [], True
        self._refresh()

    def _decimals(self) -> int:
        return self.ds.var(self.source()).decimals if self.source() else 0

    def _upper_changed(self) -> None:
        if self.auto_labels and self.cuts:
            self.labels = do.bin_labels(self.cuts, self._decimals(), _checked(self.upper))
        self._refresh()

    def _refresh(self) -> None:
        if not self.source():
            return
        upper = _checked(self.upper)
        counts = do.bin_counts(self.ds, self.source(), self.cuts, upper) if self.cuts \
            else [int((~np.isnan(self._values())).sum())]
        k = len(self.cuts) + 1
        labels = (self.labels + [""] * k)[:k]
        self._filling = True
        self.grid.setRowCount(k + 1)
        for r in range(k + 1):
            if r < len(self.cuts):
                value = format_code(round(self.cuts[r], 10))
            elif r == len(self.cuts):
                value = "HIGH"
            else:
                value = ""
            cells = [value, labels[r] if r < k else "", str(counts[r]) if r < k else ""]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c == 2 or (r == len(self.cuts) and c == 0):
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    item.setForeground(Qt.GlobalColor.darkGray)
                if c == 2:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                          | Qt.AlignmentFlag.AlignVCenter)
                self.grid.setItem(r, c, item)
        self._filling = False
        self.hist.setPixmap(QPixmap.fromImage(_png_image(_histogram_png(self._values(),
                                                                        self.cuts))))

    def _grid_edited(self, item: QTableWidgetItem) -> None:
        if self._filling:
            return
        r, c = item.row(), item.column()
        if c == 1:
            k = len(self.cuts) + 1
            self.labels = (self.labels + [""] * k)[:k]
            if r < k:
                self.labels[r] = item.text()
                self.auto_labels = False
            return
        if c != 0:
            return
        text = item.text().strip()
        try:
            value = _parse_number(text) if text else None
        except ValueError:
            QMessageBox.warning(self, self.windowTitle(), f"'{text}' is not a number.")
            self._refresh()
            return
        if r < len(self.cuts):
            if value is None:
                del self.cuts[r]
            else:
                self.cuts[r] = value
        elif value is not None:
            self.cuts.append(value)
        self.cuts = sorted(set(self.cuts))
        if self.auto_labels:
            self.labels = do.bin_labels(self.cuts, self._decimals(), _checked(self.upper)) \
                if self.cuts else []
        self._refresh()

    def _delete_row(self) -> None:
        r = self.grid.currentRow()
        if 0 <= r < len(self.cuts):
            del self.cuts[r]
            if self.auto_labels:
                self.labels = do.bin_labels(self.cuts, self._decimals(), _checked(self.upper)) \
                    if self.cuts else []
            elif r < len(self.labels):
                del self.labels[r]
            self._refresh()

    def _make_cutpoints(self) -> None:
        if not self.source():
            return
        dlg = MakeCutpointsDialog(self._values(), self._decimals(), self)
        if dlg.exec():
            self.set_cutpoints(dlg.cuts)

    def set_cutpoints(self, cuts: list[float]) -> None:
        self.cuts = sorted(set(float(c) for c in cuts))
        self.auto_labels = True
        self.labels = do.bin_labels(self.cuts, self._decimals(), _checked(self.upper))
        self._refresh()

    def _make_labels(self) -> None:
        if self.cuts:
            self.labels = do.bin_labels(self.cuts, self._decimals(), _checked(self.upper))
            self.auto_labels = True
            self._refresh()

    def accept(self) -> None:
        name = self.bin_name.text().strip()
        if not self.source():
            return
        if not is_valid_name(name) or self.ds.has_name(name):
            QMessageBox.warning(self, self.windowTitle(),
                                "Enter a new, valid name for the binned variable.")
            return
        if not self.cuts:
            QMessageBox.warning(self, self.windowTitle(),
                                "Define at least one cutpoint (or click Make Cutpoints…).")
            return
        super().accept()

    def compute(self):
        k = len(self.cuts) + 1
        labels = (self.labels + [""] * k)[:k]
        return do.visual_binning(self.ds, self.source(), self.bin_name.text().strip(),
                                 self.cuts, labels, _checked(self.upper),
                                 self.reverse.isChecked(), self.bin_label.text().strip())


def _png_image(png: bytes):
    from PySide6.QtGui import QImage

    image = QImage()
    image.loadFromData(png, "PNG")
    return image


# ================================================================== RANK
class RankCasesDialog(VariablePickerDialog):
    modifies_data = True
    DEFAULTS = {"functions": ["RANK"], "ntiles": 4, "fraction": "BLOM", "ties": "MEAN",
                "ascending": True, "summary": True}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Rank Cases", parent)
        self.add_target("vars", "Variable(s):", accepts=is_numeric)
        self.add_target("by", "By:", required=False)
        box = QGroupBox("Assign Rank 1 to")
        bl = QVBoxLayout(box)
        self.order = _radio_group(self, bl, [(True, "Smallest value"),
                                             (False, "Largest value")], self.opts["ascending"])
        row = QHBoxLayout()
        row.addWidget(box)
        cb = QCheckBox("Display summary tables")
        cb.setChecked(self.opts["summary"])
        cb.toggled.connect(lambda on: self.opts.__setitem__("summary", on))
        row.addWidget(cb, 0, Qt.AlignmentFlag.AlignBottom)
        row.addStretch()
        self.options_layout.addLayout(row)
        self.add_option_button("Rank Types...", self._rank_types)
        self.add_option_button("Ties...", self._ties)
        self.finish()

    def _rank_types(self) -> None:
        dlg = SubDialog("Rank Cases: Types", self, columns=1)
        box = dlg.group("Rank Types")
        funcs = self.opts["functions"]
        for key, text in (("RANK", "Rank"), ("SAVAGE", "Savage score"),
                          ("RFRACTION", "Fractional rank"),
                          ("PERCENT", "Fractional rank as %"),
                          ("N", "Sum of case weights")):
            dlg.check(box, key, text, key in funcs)
        nt = QHBoxLayout()
        nt_cb = QCheckBox("Ntiles:")
        nt_cb.setChecked("NTILES" in funcs)
        nt_spin = QSpinBox()
        nt_spin.setRange(2, 999)
        nt_spin.setValue(self.opts["ntiles"])
        nt.addWidget(nt_cb)
        nt.addWidget(nt_spin)
        nt.addStretch()
        box.addLayout(nt)
        dlg._getters["NTILES"] = nt_cb.isChecked
        dlg._getters["ntiles"] = nt_spin.value
        dlg.check(box, "PROPORTION", "Proportion estimates", "PROPORTION" in funcs)
        dlg.check(box, "NORMAL", "Normal scores", "NORMAL" in funcs)
        fbox = dlg.group("Proportion Estimation Formula")
        dlg.radio(fbox, "fraction", [("BLOM", "Blom"), ("TUKEY", "Tukey"),
                                     ("RANKIT", "Rankit"), ("VW", "Van der Waerden")],
                  self.opts["fraction"])
        values = dlg.run()
        if values:
            order = ["RANK", "SAVAGE", "RFRACTION", "PERCENT", "N", "NTILES", "PROPORTION",
                     "NORMAL"]
            chosen = [f for f in order if values.get(f)]
            if not chosen:
                QMessageBox.warning(self, self.windowTitle(),
                                    "Select at least one rank type.")
                return
            self.opts.update(functions=chosen, ntiles=int(values["ntiles"]),
                             fraction=values["fraction"])

    def _ties(self) -> None:
        dlg = SubDialog("Rank Cases: Ties", self, columns=1)
        box = dlg.group("Rank Assigned to Ties")
        dlg.radio(box, "ties", [("MEAN", "Mean"), ("LOW", "Low"), ("HIGH", "High"),
                                ("CONDENSE", "Sequential ranks to unique values")],
                  self.opts["ties"])
        values = dlg.run()
        if values:
            self.opts["ties"] = values["ties"]

    def validate(self):
        self.opts["ascending"] = _checked(self.order)
        return None

    def compute(self):
        o = self.opts
        return do.rank_cases(self.ds, self.selected("vars"), o["functions"],
                             self.selected("by"), o["ties"], o["ascending"], o["ntiles"],
                             o["fraction"], o["summary"])


# ================================================================== REPLACE MISSING VALUES
class ReplaceMissingDialog(_OutputNamesMixin, VariablePickerDialog):
    modifies_data = True
    DEFAULTS = {"outputs": {}, "method": "SMEAN", "span": 2, "all": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Replace Missing Values", parent)
        self.var_list = self.add_target("vars", "New Variable(s):", accepts=is_numeric)
        self.var_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.var_list.currentItemChanged.connect(lambda *_: self._load())
        box = QGroupBox("Name and Method")
        form = QFormLayout(box)
        self.name_edit = QLineEdit()
        self.method = QComboBox()
        for key, text in do.RMV_METHODS.items():
            self.method.addItem(text, key)
        self.method.setCurrentIndex(list(do.RMV_METHODS).index(self.opts["method"]))
        span_box = QGroupBox("Span of nearby points")
        sl = QHBoxLayout(span_box)
        self.r_number = QRadioButton("Number:")
        self.span = QSpinBox()
        self.span.setRange(1, 999)
        self.span.setValue(self.opts["span"])
        self.r_all = QRadioButton("All")
        (self.r_all if self.opts["all"] else self.r_number).setChecked(True)
        sl.addWidget(self.r_number)
        sl.addWidget(self.span)
        sl.addWidget(self.r_all)
        sl.addStretch()
        change = QPushButton("Change")
        change.clicked.connect(self._change)
        form.addRow("Name:", self.name_edit)
        form.addRow("Method:", self.method)
        form.addRow(span_box)
        form.addRow("", change)
        self.method.currentIndexChanged.connect(self._span_enabled)
        self.options_layout.addWidget(box)
        self._span_enabled()
        self.finish()

    def _span_enabled(self) -> None:
        nearby = self.method.currentData() in ("MEAN", "MEDIAN")
        for w in (self.r_number, self.span, self.r_all):
            w.setEnabled(nearby)

    def _default_spec(self, src, taken):
        return {"name": do.unique_name(self.ds, f"{src}_1", taken),
                "method": self.method.currentData() if hasattr(self, "method")
                else self.opts["method"],
                "span": None if getattr(self, "r_all", None) and self.r_all.isChecked()
                else (self.span.value() if hasattr(self, "span") else 2)}

    def _item_text(self, src, spec):
        return f"{spec['name']}={do.rmv_function_text(spec['method'], src, spec['span'])}"

    def _load(self) -> None:
        src = self._current_source()
        spec = self._outputs().get(src) if src else None
        if not spec:
            return
        self.name_edit.setText(spec["name"])
        self.method.setCurrentIndex(list(do.RMV_METHODS).index(spec["method"]))
        if spec["span"] is None:
            self.r_all.setChecked(True)
        else:
            self.r_number.setChecked(True)
            self.span.setValue(spec["span"])

    def _change(self) -> None:
        src = self._current_source()
        if src is None:
            QMessageBox.information(self, self.windowTitle(), "Select a variable first.")
            return
        name = self.name_edit.text().strip()
        if not is_valid_name(name) or self.ds.has_name(name):
            QMessageBox.warning(self, self.windowTitle(), f"'{name}' is not a valid new name.")
            return
        method = self.method.currentData()
        span = (None if self.r_all.isChecked() else self.span.value()) \
            if method in ("MEAN", "MEDIAN") else 2
        self._outputs()[src] = {"name": name, "method": method, "span": span}
        self.opts.update(method=method, span=self.span.value(), all=self.r_all.isChecked())
        self.targets_changed()

    def validate(self):
        return self._check_output_names()

    def compute(self):
        out = self._outputs()
        specs = [(s, out[s]["name"], out[s]["method"], out[s]["span"])
                 for s in self.selected("vars")]
        return do.replace_missing(self.ds, specs)


# ================================================================== SHIFT VALUES
class ShiftValuesDialog(_OutputNamesMixin, VariablePickerDialog):
    modifies_data = True
    DEFAULTS = {"outputs": {}, "lag": True, "n": 1}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Shift Values", parent)
        self.var_list = self.add_target("vars", "Variable -> New name:")
        self.var_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.var_list.currentItemChanged.connect(lambda *_: self._load())
        box = QGroupBox("Name and Method")
        bl = QVBoxLayout(box)
        form = QFormLayout()
        self.name_edit = QLineEdit()
        form.addRow("Name:", self.name_edit)
        bl.addLayout(form)
        self.direction = _radio_group(self, bl, [(True, "Get value from earlier case (lag)"),
                                                 (False, "Get value from following case "
                                                         "(lead)")], self.opts["lag"])
        nf = QFormLayout()
        self.n = QSpinBox()
        self.n.setRange(1, 99999)
        self.n.setValue(self.opts["n"])
        nf.addRow("Number of cases to shift:", self.n)
        bl.addLayout(nf)
        change = QPushButton("Change")
        change.clicked.connect(self._change)
        bl.addWidget(change, 0, Qt.AlignmentFlag.AlignLeft)
        self.options_layout.addWidget(box)
        self.finish()

    def _default_spec(self, src, taken):
        lag = _checked(self.direction) if hasattr(self, "direction") else self.opts["lag"]
        n = self.n.value() if hasattr(self, "n") else self.opts["n"]
        return {"name": do.unique_name(self.ds, f"{src}_{'lag' if lag else 'lead'}{n}", taken),
                "lag": lag, "n": n}

    def _item_text(self, src, spec):
        return f"{src} --> {spec['name']} ({'LAG' if spec['lag'] else 'LEAD'}={spec['n']})"

    def _load(self) -> None:
        src = self._current_source()
        spec = self._outputs().get(src) if src else None
        if spec:
            self.name_edit.setText(spec["name"])
            self.direction[spec["lag"]].setChecked(True)
            self.n.setValue(spec["n"])

    def _change(self) -> None:
        src = self._current_source()
        if src is None:
            QMessageBox.information(self, self.windowTitle(), "Select a variable first.")
            return
        name = self.name_edit.text().strip()
        if not is_valid_name(name) or self.ds.has_name(name):
            QMessageBox.warning(self, self.windowTitle(), f"'{name}' is not a valid new name.")
            return
        self._outputs()[src] = {"name": name, "lag": _checked(self.direction),
                                "n": self.n.value()}
        self.opts.update(lag=_checked(self.direction), n=self.n.value())
        self.targets_changed()

    def validate(self):
        return self._check_output_names()

    def compute(self):
        out = self._outputs()
        return do.shift_values(self.ds, [(s, out[s]["name"], out[s]["lag"], out[s]["n"])
                                         for s in self.selected("vars")])


# ================================================================== IDENTIFY DUPLICATES
class IdentifyDuplicatesDialog(VariablePickerDialog):
    modifies_data = True
    DEFAULTS = {"ascending": True, "primary": True, "primary_kind": "last",
                "primary_name": "PrimaryLast", "sequence": False,
                "sequence_name": "MatchSequence", "move_top": True, "freq": True}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Identify Duplicate Cases", parent)
        self.setMinimumSize(640, 560)
        self.add_target("match", "Define matching cases by:")
        self.add_target("sort", "Sort within matching groups by:", required=False)
        sort_box = QGroupBox("Sort")
        sb = QHBoxLayout(sort_box)
        self.sort_order = _radio_group(self, sb, [(True, "Ascending"), (False, "Descending")],
                                       self.opts["ascending"])
        sb.addStretch()
        self.options_layout.addWidget(sort_box)
        box = QGroupBox("Variables to Create")
        grid = QGridLayout(box)
        self.primary = QCheckBox("Indicator of primary cases (1=unique or primary, "
                                 "0=duplicate)")
        self.primary.setChecked(self.opts["primary"])
        self.kind = {}
        group = QButtonGroup(self)
        for r, (value, text) in enumerate((("last", "Last case in each group is primary"),
                                           ("first", "First case in each group is primary"))):
            rb = QRadioButton(text)
            rb.setChecked(self.opts["primary_kind"] == value)
            group.addButton(rb)
            self.kind[value] = rb
            grid.addWidget(rb, 1 + r, 0)
            rb.toggled.connect(self._kind_changed)
        self.primary_name = QLineEdit(self.opts["primary_name"])
        self.sequence = QCheckBox("Sequential count of matching case in each group "
                                  "(0=nonmatching)")
        self.sequence.setChecked(self.opts["sequence"])
        self.sequence_name = QLineEdit(self.opts["sequence_name"])
        grid.addWidget(self.primary, 0, 0, 1, 3)
        grid.addWidget(QLabel("Name:"), 1, 1)
        grid.addWidget(self.primary_name, 1, 2)
        grid.addWidget(self.sequence, 3, 0, 1, 3)
        grid.addWidget(QLabel("Name:"), 4, 1)
        grid.addWidget(self.sequence_name, 4, 2)
        for w in (self.primary_name, self.sequence_name):
            w.setMaximumWidth(160)
        grid.setColumnStretch(0, 1)
        self.options_layout.addWidget(box)
        self.move_top = QCheckBox("Move matching cases to the top of the file")
        self.move_top.setChecked(self.opts["move_top"])
        self.freq = QCheckBox("Display frequencies for created variables")
        self.freq.setChecked(self.opts["freq"])
        self.options_layout.addWidget(self.move_top)
        self.options_layout.addWidget(self.freq)
        self.finish()

    def _kind_changed(self) -> None:
        kind = "last" if self.kind["last"].isChecked() else "first"
        if self.primary_name.text() in ("PrimaryLast", "PrimaryFirst", ""):
            self.primary_name.setText("PrimaryLast" if kind == "last" else "PrimaryFirst")

    def validate(self):
        if not self.primary.isChecked() and not self.sequence.isChecked():
            return "Select at least one variable to create."
        names = []
        if self.primary.isChecked():
            names.append(self.primary_name.text().strip())
        if self.sequence.isChecked():
            names.append(self.sequence_name.text().strip())
        for n in names:
            if not is_valid_name(n) or self.ds.has_name(n):
                return f"'{n}' is not a valid new variable name."
        if len({n.lower() for n in names}) != len(names):
            return "The new variable names must be different."
        self.opts.update(ascending=_checked(self.sort_order), primary=self.primary.isChecked(),
                         primary_kind="last" if self.kind["last"].isChecked() else "first",
                         primary_name=self.primary_name.text().strip(),
                         sequence=self.sequence.isChecked(),
                         sequence_name=self.sequence_name.text().strip(),
                         move_top=self.move_top.isChecked(), freq=self.freq.isChecked())
        return None

    def compute(self):
        o = self.opts
        return do.identify_duplicates(
            self.ds, self.selected("match"), self.selected("sort"), o["ascending"],
            o["primary_kind"] if o["primary"] else None, o["primary_name"],
            o["sequence_name"] if o["sequence"] else None, o["move_top"], o["freq"])


# ================================================================== MERGE FILES
def _clone(ds: Dataset) -> Dataset:
    variables = [Variable.from_dict(v.to_dict()) for v in ds.variables]
    clone = Dataset(ds.df.copy(), variables, name=ds.name)
    clone.path = ds.path
    return clone


def load_other_dataset(path: str) -> Dataset:
    from rsstatchem.io import files

    suffix = Path(path).suffix.lower()
    if suffix == ".sav":
        return files.read_sav(path)
    if suffix == files.PROJECT_EXT:
        return files.load_project(path)
    raise do.DataError("Choose an RS-StatChem (.rsd) or SPSS (.sav) data file.")


class _MergeBase(QDialog):
    """Dataset chooser shared by Add Cases and Add Variables (SPSS 'Merge Files' step 1)."""

    modifies_data = True
    TITLE = ""

    def __init__(self, ds: Dataset, parent: QWidget | None = None):
        super().__init__(parent)
        self.ds = ds
        self.other: Dataset | None = None
        self.other_label = ""
        self.setWindowTitle(f"{self.TITLE} to {ds.name}")
        box = QGroupBox("Select a dataset from the list of open datasets or from a file to "
                        "merge with the active dataset")
        bl = QVBoxLayout(box)
        self.r_open = QRadioButton("An open dataset")
        self.r_file = QRadioButton("An external SPSS Statistics or RS-StatChem data file")
        group = QButtonGroup(self)
        group.addButton(self.r_open)
        group.addButton(self.r_file)
        self.open_list = QListWidget()
        self.open_list.setMaximumHeight(70)
        self._editors = []
        ctx = getattr(parent, "ctx", None)
        for editor in getattr(ctx, "editors", []) if ctx else []:
            if editor is parent or editor.ds is ds:
                continue
            path = editor.ds.path
            self.open_list.addItem(editor.ds.name + (f"  ({Path(path).name})" if path else ""))
            self._editors.append(editor)
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        prow = QHBoxLayout()
        prow.addWidget(self.path_edit, 1)
        prow.addWidget(browse)
        bl.addWidget(self.r_open)
        bl.addWidget(self.open_list)
        bl.addWidget(self.r_file)
        bl.addLayout(prow)
        (self.r_open if self._editors else self.r_file).setChecked(True)
        self.r_open.setEnabled(bool(self._editors))
        self.open_list.setEnabled(bool(self._editors))
        self.open_list.currentRowChanged.connect(self._pick_open)
        self.source_box = box

        buttons = QDialogButtonBox()
        self.ok = buttons.addButton(QDialogButtonBox.StandardButton.Ok)
        reset = buttons.addButton(QDialogButtonBox.StandardButton.Reset)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        reset.clicked.connect(self._reset)
        self.buttons = buttons
        self.ok.setEnabled(False)

    def _browse(self) -> None:
        ctx = _editor_ctx(self)
        start = str(ctx.settings.value("last_dir", "")) if ctx else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Read File", start,
            "SPSS Statistics / RS-StatChem data (*.sav *.rsd);;SPSS Statistics (*.sav);;"
            "RS-StatChem data (*.rsd)")
        if path:
            self.r_file.setChecked(True)
            self.set_other_path(path)

    def set_other_path(self, path: str) -> None:
        try:
            other = load_other_dataset(path)
        except Exception as exc:  # noqa: BLE001 - show any read failure
            QMessageBox.critical(self, self.windowTitle(), f"Could not read {path}:\n{exc}")
            return
        self.path_edit.setText(path)
        self.r_file.setChecked(True)
        self.open_list.clearSelection()
        self.set_other(other, Path(path).name)

    def _pick_open(self, row: int) -> None:
        if 0 <= row < len(self._editors):
            self.r_open.setChecked(True)
            editor = self._editors[row]
            self.set_other(_clone(editor.ds), editor.ds.name)

    def set_other(self, other: Dataset, label: str) -> None:
        self.other = other
        self.other_label = label
        self.ok.setEnabled(True)
        self.other_loaded()

    def other_loaded(self) -> None:  # pragma: no cover - overridden
        pass

    def _reset(self) -> None:
        if self.other is not None:
            self.other_loaded()


class AddCasesDialog(_MergeBase):
    TITLE = "Add Cases"

    def __init__(self, ds: Dataset, parent: QWidget | None = None):
        super().__init__(ds, parent)
        self.resize(700, 560)
        self.unpaired = QListWidget()
        self.unpaired.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.new_list = QListWidget()
        self.new_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.arrow = QPushButton("▶")
        self.arrow.setObjectName("arrow")
        self.arrow.clicked.connect(self._arrow)
        pair = QPushButton("Pair")
        pair.clicked.connect(self._pair)
        rename = QPushButton("Rename...")
        rename.clicked.connect(self._rename)
        left = QVBoxLayout()
        left.addWidget(QLabel("Unpaired Variables:"))
        left.addWidget(self.unpaired, 1)
        lb = QHBoxLayout()
        lb.addWidget(pair)
        lb.addWidget(rename)
        lb.addStretch()
        left.addLayout(lb)
        mid = QVBoxLayout()
        mid.addStretch()
        mid.addWidget(self.arrow)
        mid.addStretch()
        right = QVBoxLayout()
        right.addWidget(QLabel("Variables in New Active Dataset:"))
        right.addWidget(self.new_list, 1)
        lists = QHBoxLayout()
        lists.addLayout(left, 1)
        lists.addLayout(mid)
        lists.addLayout(right, 1)
        self.source_cb = QCheckBox("Indicate case source as variable:")
        self.source_name = QLineEdit("source01")
        self.source_name.setMaximumWidth(160)
        srow = QHBoxLayout()
        srow.addWidget(self.source_cb)
        srow.addWidget(self.source_name)
        srow.addStretch()
        legend = QLabel("(*)=Active dataset   (+)=" + "other dataset")
        legend.setStyleSheet(GREY)
        self.legend = legend
        layout = QVBoxLayout(self)
        layout.addWidget(self.source_box)
        layout.addLayout(lists, 1)
        layout.addWidget(legend)
        layout.addLayout(srow)
        layout.addWidget(self.buttons)
        self.unpaired.itemSelectionChanged.connect(self._arrows)
        self.new_list.itemSelectionChanged.connect(self._arrows)
        self.renames: dict[str, str] = {}

    @staticmethod
    def _item(text: str, data: tuple) -> QListWidgetItem:
        item = QListWidgetItem(text)
        item.setData(SPEC_ROLE, data)
        return item

    def other_loaded(self) -> None:
        self.unpaired.clear()
        self.new_list.clear()
        self.legend.setText(f"(*)=Active dataset   (+)={self.other_label}")
        paired, un_a, un_b = do.match_variables(self.ds, self.other)
        for a, b in paired:
            self.new_list.addItem(self._item(a if a == b else f"{a} & {b}", ("pair", a, b)))
        for a in un_a:
            self.unpaired.addItem(self._item(f"{a}(*)", ("a", a)))
        for b in un_b:
            self.unpaired.addItem(self._item(f"{b}(+)", ("b", b)))

    def _arrows(self) -> None:
        back = bool(self.new_list.selectedItems()) and not self.unpaired.selectedItems()
        self.arrow.setText("◀" if back else "▶")

    def _arrow(self) -> None:
        if self.new_list.selectedItems() and not self.unpaired.selectedItems():
            for item in self.new_list.selectedItems():
                data = item.data(SPEC_ROLE)
                self.new_list.takeItem(self.new_list.row(item))
                if data[0] == "pair":
                    self.unpaired.addItem(self._item(f"{data[1]}(*)", ("a", data[1])))
                    self.unpaired.addItem(self._item(f"{data[2]}(+)", ("b", data[2])))
                else:
                    self.unpaired.addItem(item)
        else:
            for item in self.unpaired.selectedItems():
                data = item.data(SPEC_ROLE)
                if data[0] == "b" and self.ds.has_name(data[1]):
                    QMessageBox.warning(self, self.windowTitle(),
                                        f"{data[1]} already exists in the active dataset. "
                                        "Rename it or pair it with a variable.")
                    continue
                self.unpaired.takeItem(self.unpaired.row(item))
                self.new_list.addItem(item)
        self._arrows()

    def _pair(self) -> None:
        items = self.unpaired.selectedItems()
        kinds = sorted(i.data(SPEC_ROLE)[0] for i in items)
        if len(items) != 2 or kinds != ["a", "b"]:
            QMessageBox.information(self, self.windowTitle(),
                                    "Select one variable from the active dataset (*) and one "
                                    "from the other dataset (+) to pair them.")
            return
        a = next(i for i in items if i.data(SPEC_ROLE)[0] == "a").data(SPEC_ROLE)[1]
        b = next(i for i in items if i.data(SPEC_ROLE)[0] == "b").data(SPEC_ROLE)[1]
        va, vb = self.ds.var(a), self.other.var(b)
        if (va.type == STRING) != (vb.type == STRING):
            QMessageBox.warning(self, self.windowTitle(),
                                "Numeric and string variables cannot be paired.")
            return
        for i in items:
            self.unpaired.takeItem(self.unpaired.row(i))
        self.new_list.addItem(self._item(f"{a} & {b}", ("pair", a, b)))

    def _rename(self) -> None:
        items = self.unpaired.selectedItems()
        if len(items) != 1 or items[0].data(SPEC_ROLE)[0] != "b":
            QMessageBox.information(self, self.windowTitle(),
                                    "Select one variable of the other dataset (+) to rename.")
            return
        old = items[0].data(SPEC_ROLE)[1]
        name, ok = QInputDialog.getText(self, "Rename", f"New name for {old}:", text=old)
        name = name.strip()
        if not ok or name == old:
            return
        if not is_valid_name(name) or self.other.has_name(name) or self.ds.has_name(name):
            QMessageBox.warning(self, self.windowTitle(), f"'{name}' cannot be used.")
            return
        self.other.rename_variable(self.other.index_of(old), name)
        items[0].setText(f"{old}->{name}(+)")
        items[0].setData(SPEC_ROLE, ("b", name))

    def _plan(self) -> tuple[list, list, list]:
        pairs, keep_a, keep_b = [], [], []
        for i in range(self.new_list.count()):
            data = self.new_list.item(i).data(SPEC_ROLE)
            if data[0] == "pair":
                pairs.append((data[1], data[2]))
            elif data[0] == "a":
                keep_a.append(data[1])
            else:
                keep_b.append(data[1])
        return pairs, keep_a, keep_b

    def accept(self) -> None:
        if self.other is None:
            return
        if not self.new_list.count():
            QMessageBox.warning(self, self.windowTitle(),
                                "The new active dataset would have no variables.")
            return
        if self.source_cb.isChecked():
            name = self.source_name.text().strip()
            if not is_valid_name(name) or any(
                    n.lower() == name.lower() for n in self.ds.names() + self.other.names()):
                QMessageBox.warning(self, self.windowTitle(),
                                    f"'{name}' is not a valid new name for the source "
                                    "variable.")
                return
        super().accept()

    def compute(self):
        pairs, keep_a, keep_b = self._plan()
        return do.add_cases(self.ds, self.other, pairs, keep_a, keep_b,
                            self.source_name.text().strip() if self.source_cb.isChecked()
                            else None, self.other_label)


class AddVariablesDialog(_MergeBase):
    TITLE = "Add Variables"

    def __init__(self, ds: Dataset, parent: QWidget | None = None):
        super().__init__(ds, parent)
        self.resize(760, 640)
        mbox = QGroupBox("Merge Method")
        ml = QVBoxLayout(mbox)
        self.method = _radio_group(self, ml, [
            ("order", "One-to-one merge based on file order"),
            ("key", "One-to-one merge based on key values"),
            ("table", "One-to-many merge based on key values")], "key")
        lbox = QGroupBox("Select Lookup Table")
        ll = QVBoxLayout(lbox)
        self.lookup = _radio_group(self, ll, [("active", "Active dataset"),
                                              ("other", "Other dataset")], "other")
        sort_cb = QCheckBox("Sort files by key values before merging")
        sort_cb.setChecked(True)
        sort_cb.setEnabled(False)
        sort_cb.setToolTip("Both files are always sorted by the key variables.")
        ml.addWidget(sort_cb)
        self.lookup_box = lbox
        top = QHBoxLayout()
        top.addWidget(mbox, 2)
        top.addWidget(lbox, 1)
        for b in self.method.values():
            b.toggled.connect(self._method_changed)

        self.excluded = QListWidget()
        self.included = QListWidget()
        self.keys = QListWidget()
        for w in (self.excluded, self.included, self.keys):
            w.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        to_inc, to_key = QPushButton("▶"), QPushButton("▶")
        for b in (to_inc, to_key):
            b.setObjectName("arrow")
        to_inc.clicked.connect(lambda: self._move(self.excluded, self.included))
        to_key.clicked.connect(lambda: self._move(self.excluded, self.keys))
        back_inc, back_key = QPushButton("◀"), QPushButton("◀")
        for b in (back_inc, back_key):
            b.setObjectName("arrow")
        back_inc.clicked.connect(lambda: self._move(self.included, self.excluded))
        back_key.clicked.connect(lambda: self._move(self.keys, self.excluded))
        rename = QPushButton("Rename...")
        rename.clicked.connect(self._rename)
        left = QVBoxLayout()
        left.addWidget(QLabel("Excluded Variables:"))
        left.addWidget(self.excluded, 1)
        left.addWidget(rename, 0, Qt.AlignmentFlag.AlignLeft)
        arrows = QVBoxLayout()
        arrows.addStretch()
        arrows.addWidget(to_inc)
        arrows.addWidget(back_inc)
        arrows.addSpacing(30)
        arrows.addWidget(to_key)
        arrows.addWidget(back_key)
        arrows.addStretch()
        right = QVBoxLayout()
        right.addWidget(QLabel("Included Variables:"))
        right.addWidget(self.included, 2)
        self.key_label = QLabel("Key Variables:")
        right.addWidget(self.key_label)
        right.addWidget(self.keys, 1)
        lists = QHBoxLayout()
        lists.addLayout(left, 1)
        lists.addLayout(arrows)
        lists.addLayout(right, 1)
        self.legend = QLabel("(*)=Active dataset   (+)=other dataset")
        self.legend.setStyleSheet(GREY)
        layout = QVBoxLayout(self)
        layout.addWidget(self.source_box)
        layout.addLayout(top)
        layout.addLayout(lists, 1)
        layout.addWidget(self.legend)
        layout.addWidget(self.buttons)
        self.renames: dict[str, str] = {}
        self._method_changed()

    def _method_changed(self) -> None:
        method = _checked(self.method)
        self.lookup_box.setEnabled(method == "table")
        self.keys.setEnabled(method != "order")
        self.key_label.setEnabled(method != "order")

    def other_loaded(self) -> None:
        self.excluded.clear()
        self.included.clear()
        self.keys.clear()
        self.renames = {}
        self.lookup["other"].setText(self.other_label)
        self.legend.setText(f"(*)=Active dataset   (+)={self.other_label}")
        for var in self.ds.variables:
            item = QListWidgetItem(f"{var.name}(*)")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            item.setForeground(Qt.GlobalColor.darkGray)
            self.included.addItem(item)
        for var in self.other.variables:
            item = QListWidgetItem(f"{var.name}(+)")
            item.setData(SPEC_ROLE, var.name)
            if self.ds.has_name(var.name):
                self.excluded.addItem(item)
            else:
                self.included.addItem(item)

    def _move(self, src: QListWidget, dst: QListWidget) -> None:
        for item in src.selectedItems():
            name = item.data(SPEC_ROLE)
            if name is None:
                continue
            new_name = self.renames.get(name, name)
            if dst is self.keys:
                if not self.ds.has_name(name) or name in self.renames:
                    QMessageBox.warning(self, self.windowTitle(),
                                        f"{name} is not in the active dataset and cannot be a "
                                        "key variable.")
                    continue
                if not do._compatible(self.ds.var(name), self.other.var(name)):
                    QMessageBox.warning(self, self.windowTitle(),
                                        f"{name} has different types in the two datasets.")
                    continue
            if dst is self.included and self.ds.has_name(new_name):
                QMessageBox.warning(self, self.windowTitle(),
                                    f"{new_name} already exists in the active dataset. "
                                    "Rename it first.")
                continue
            src.takeItem(src.row(item))
            dst.addItem(item)

    def _rename(self) -> None:
        items = self.excluded.selectedItems()
        if len(items) != 1:
            QMessageBox.information(self, self.windowTitle(),
                                    "Select one excluded variable to rename.")
            return
        old = items[0].data(SPEC_ROLE)
        current = self.renames.get(old, old)
        name, ok = QInputDialog.getText(self, "Rename", f"New name for {old}:", text=current)
        name = name.strip()
        if not ok or not name:
            return
        if not is_valid_name(name) or self.ds.has_name(name):
            QMessageBox.warning(self, self.windowTitle(), f"'{name}' cannot be used.")
            return
        self.renames[old] = name
        items[0].setText(f"{old}->{name}(+)")

    def _names(self, widget: QListWidget) -> list[str]:
        return [widget.item(i).data(SPEC_ROLE) for i in range(widget.count())
                if widget.item(i).data(SPEC_ROLE) is not None]

    def accept(self) -> None:
        if self.other is None:
            return
        method = _checked(self.method)
        if method != "order" and not self._names(self.keys):
            QMessageBox.warning(self, self.windowTitle(),
                                "Move at least one variable that is in both datasets to the "
                                "Key Variables list.")
            return
        super().accept()

    def compute(self):
        method = _checked(self.method)
        include = self._names(self.included)
        keys = self._names(self.keys) if method != "order" else []
        return do.add_variables(self.ds, self.other, method, keys, include,
                                {k: v for k, v in self.renames.items() if k in include},
                                _checked(self.lookup), self.other_label)


# ================================================================== AGGREGATE
class AggregateFunctionDialog(QDialog):
    GROUPS = [("Summary Statistics", [("MEAN", "Mean"), ("MEDIAN", "Median"), ("SUM", "Sum"),
                                      ("SD", "Standard deviation")]),
              ("Specific Values", [("FIRST", "First"), ("LAST", "Last"), ("MIN", "Minimum"),
                                   ("MAX", "Maximum")]),
              ("Number of cases", [("N", "Weighted"), ("NMISS", "Weighted missing"),
                                   ("NU", "Unweighted"), ("NUMISS", "Unweighted missing")])]

    def __init__(self, spec: do.AggSpec, string: bool, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Aggregate Data: Aggregate Function")
        self.buttons: dict[str, QRadioButton] = {}
        group = QButtonGroup(self)
        grid = QGridLayout()
        for c, (title, items) in enumerate(self.GROUPS):
            box = QGroupBox(title)
            bl = QVBoxLayout(box)
            for key, text in items:
                rb = QRadioButton(text)
                group.addButton(rb)
                bl.addWidget(rb)
                self.buttons[key] = rb
                if string and key in ("MEAN", "MEDIAN", "SUM", "SD"):
                    rb.setEnabled(False)
            grid.addWidget(box, 0, c)
        self.value = QLineEdit()
        self.low = QLineEdit()
        self.high = QLineEdit()
        for c, (prefix, title) in enumerate((("P", "Percentages"), ("F", "Fractions"),
                                             ("C", "Counts"))):
            box = QGroupBox(title)
            bl = QVBoxLayout(box)
            for suffix, text in (("GT", "Above"), ("LT", "Below"), ("IN", "Inside"),
                                 ("OUT", "Outside")):
                rb = QRadioButton(text)
                rb.setEnabled(not string)
                group.addButton(rb)
                bl.addWidget(rb)
                self.buttons[prefix + suffix] = rb
            grid.addWidget(box, 1, c)
        vals = QFormLayout()
        vals.addRow("Value:", self.value)
        row = QHBoxLayout()
        row.addWidget(QLabel("Low:"))
        row.addWidget(self.low)
        row.addWidget(QLabel("High:"))
        row.addWidget(self.high)
        vals.addRow(row)
        self.buttons.get(spec.function, self.buttons["MEAN"]).setChecked(True)
        if len(spec.args) == 1:
            self.value.setText(format_code(spec.args[0]))
        elif len(spec.args) == 2:
            self.low.setText(format_code(spec.args[0]))
            self.high.setText(format_code(spec.args[1]))
        self.error = QLabel()
        self.error.setStyleSheet("color:#B71C1C;")
        layout = QVBoxLayout(self)
        layout.addLayout(grid)
        layout.addLayout(vals)
        layout.addWidget(self.error)
        layout.addWidget(_continue_buttons(self, self._accept))
        self.function, self.args = spec.function, spec.args

    def _accept(self) -> None:
        func = next(k for k, b in self.buttons.items() if b.isChecked())
        nargs = do.AGG_FUNCTIONS[func][0]
        try:
            if nargs == 1:
                args = (_parse_number(self.value.text()),)
            elif nargs == 2:
                args = (_parse_number(self.low.text()), _parse_number(self.high.text()))
            else:
                args = ()
        except ValueError:
            self.error.setText("Enter a numeric Value (or Low and High) for this function.")
            return
        self.function, self.args = func, args
        self.accept()


class AggregateDialog(VariablePickerDialog):
    modifies_data = True
    DEFAULTS = {"n_break": False, "n_name": "N_BREAK", "mode": "add",
                "dataset_name": "DataSet_agg", "file": ""}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Aggregate Data", parent)
        self.setMinimumSize(700, 600)
        self.add_target("break", "Break Variable(s):", required=False)
        self.summ = self._list_widget()
        self.summ_arrow = QPushButton("▶")
        self.summ_arrow.setObjectName("arrow")
        self.summ_arrow.clicked.connect(self._summ_arrow)
        self.summ.itemSelectionChanged.connect(self._update_arrows)
        self.summ.itemDoubleClicked.connect(lambda _: self._remove_summaries())
        self._target_block("Summaries of Variable(s):", self.summ, self.summ_arrow, 1)
        frow = QHBoxLayout()
        fbtn = QPushButton("Function...")
        fbtn.clicked.connect(self._function)
        nbtn = QPushButton("Name && Label...")
        nbtn.clicked.connect(self._name_label)
        frow.addStretch()
        frow.addWidget(fbtn)
        frow.addWidget(nbtn)
        self._targets_column.addLayout(frow)
        nrow = QHBoxLayout()
        self.n_cb = QCheckBox("Number of cases")
        self.n_cb.setChecked(self.opts["n_break"])
        self.n_cb.toggled.connect(lambda _: self._update_ok())
        self.n_name = QLineEdit(self.opts["n_name"])
        self.n_name.setMaximumWidth(160)
        nrow.addWidget(self.n_cb)
        nrow.addWidget(QLabel("Name:"))
        nrow.addWidget(self.n_name)
        nrow.addStretch()
        self._targets_column.addLayout(nrow)
        save = QGroupBox("Save")
        sl = QGridLayout(save)
        group = QButtonGroup(self)
        self.r_add = QRadioButton("Add aggregated variables to active dataset")
        self.r_new = QRadioButton("Create a new dataset containing only the aggregated "
                                  "variables")
        self.r_file = QRadioButton("Write a new data file containing only the aggregated "
                                   "variables")
        for r in (self.r_add, self.r_new, self.r_file):
            group.addButton(r)
        {"add": self.r_add, "new": self.r_new, "file": self.r_file}[
            self.opts["mode"]].setChecked(True)
        self.ds_name = QLineEdit(self.opts["dataset_name"])
        file_btn = QPushButton("File...")
        file_btn.clicked.connect(self._choose_file)
        self.file_label = QLabel(self.opts["file"] or "(no file chosen)")
        self.file_label.setStyleSheet(GREY)
        sl.addWidget(self.r_add, 0, 0, 1, 3)
        sl.addWidget(self.r_new, 1, 0, 1, 3)
        sl.addWidget(QLabel("Dataset name:"), 2, 0, Qt.AlignmentFlag.AlignRight)
        sl.addWidget(self.ds_name, 2, 1)
        sl.addWidget(self.r_file, 3, 0, 1, 3)
        sl.addWidget(file_btn, 4, 0, Qt.AlignmentFlag.AlignRight)
        sl.addWidget(self.file_label, 4, 1, 1, 2)
        sl.setColumnStretch(1, 1)
        self.options_layout.addWidget(save)
        self.specs: list[do.AggSpec] = []
        self.finish()

    # ------------------------------------------------------------------ summaries
    def _spec_text(self, spec: do.AggSpec) -> str:
        label = f' "{spec.label}"' if spec.label else ""
        args = "".join(f", {format_code(a)}" for a in spec.args)
        return f"{spec.name}{label} = {spec.function}({spec.source}{args})"

    def _refresh_summ(self) -> None:
        self.summ.clear()
        for spec in self.specs:
            item = QListWidgetItem(self._spec_text(spec))
            self.summ.addItem(item)
        self._update_ok()

    def targets_changed(self) -> None:
        if not hasattr(self, "summ"):
            return
        back = bool(self.summ.selectedItems()) and not self.source.selectedItems()
        self.summ_arrow.setText("◀" if back else "▶")

    def _taken(self) -> set[str]:
        return {s.name.lower() for s in self.specs}

    def add_summary(self, source: str, function: str | None = None) -> do.AggSpec:
        var = self.ds.var(source)
        function = function or ("FIRST" if var.type == STRING else "MEAN")
        name = do.default_agg_name(var.name, function)
        if self.r_add.isChecked():
            name = do.unique_name(self.ds, name, self._taken())
        else:
            base, k = name, 0
            while name.lower() in self._taken():
                k += 1
                name = f"{base}_{k}"
        spec = do.AggSpec(name, function, var.name)
        self.specs.append(spec)
        self._refresh_summ()
        return spec

    def _summ_arrow(self) -> None:
        if self.summ.selectedItems() and not self.source.selectedItems():
            self._remove_summaries()
            return
        for item in self.source.selectedItems():
            self.add_summary(item.data(NAME_ROLE))
        self.source.clearSelection()
        self._update_arrows()

    def _remove_summaries(self) -> None:
        rows = sorted((self.summ.row(i) for i in self.summ.selectedItems()), reverse=True)
        for r in rows:
            del self.specs[r]
        self._refresh_summ()
        self._update_arrows()

    def _selected_specs(self) -> list[do.AggSpec]:
        rows = sorted(self.summ.row(i) for i in self.summ.selectedItems())
        if not rows and len(self.specs) == 1:
            rows = [0]
        return [self.specs[r] for r in rows]

    def _function(self) -> None:
        specs = self._selected_specs()
        if not specs:
            QMessageBox.information(self, self.windowTitle(),
                                    "Select a summary variable in the list first.")
            return
        string = any(self.ds.var(s.source).type == STRING for s in specs)
        dlg = AggregateFunctionDialog(specs[0], string, self)
        if dlg.exec():
            for spec in specs:
                old_default = spec.name.lower().startswith(
                    do.default_agg_name(spec.source, spec.function).lower())
                spec.function, spec.args = dlg.function, dlg.args
                if old_default:
                    others = self._taken() - {spec.name.lower()}
                    base = do.default_agg_name(spec.source, spec.function)
                    name, k = base, 0
                    while name.lower() in others or (self.r_add.isChecked()
                                                     and self.ds.has_name(name)):
                        k += 1
                        name = f"{base}_{k}"
                    spec.name = name
            self._refresh_summ()

    def _name_label(self) -> None:
        specs = self._selected_specs()
        if len(specs) != 1:
            QMessageBox.information(self, self.windowTitle(),
                                    "Select one summary variable in the list.")
            return
        spec = specs[0]
        dlg = QDialog(self)
        dlg.setWindowTitle("Aggregate Data: Variable Name and Label")
        form = QFormLayout(dlg)
        form.addRow(QLabel(f"{spec.function}({spec.source})"))
        name = QLineEdit(spec.name)
        label = QLineEdit(spec.label)
        form.addRow("Name:", name)
        form.addRow("Label:", label)
        form.addRow(_continue_buttons(dlg))
        if dlg.exec():
            new = name.text().strip()
            if not is_valid_name(new) or new.lower() in self._taken() - {spec.name.lower()}:
                QMessageBox.warning(self, self.windowTitle(), f"'{new}' cannot be used.")
                return
            spec.name, spec.label = new, label.text().strip()
            self._refresh_summ()

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Aggregate Data: Output File", self.opts["file"] or "aggr.rsd",
            "RS-StatChem data (*.rsd);;SPSS Statistics (*.sav)")
        if path:
            self.opts["file"] = path
            self.file_label.setText(path)
            self.r_file.setChecked(True)

    def _mode(self) -> str:
        return "add" if self.r_add.isChecked() else "new" if self.r_new.isChecked() else "file"

    def _update_ok(self) -> None:
        super()._update_ok()
        if hasattr(self, "specs") and hasattr(self, "ok_button"):
            self.ok_button.setEnabled(bool(self.specs) or self.n_cb.isChecked())

    def validate(self):
        mode = self._mode()
        if not self.specs and not self.n_cb.isChecked():
            return "Select at least one summary variable or Number of cases."
        if mode == "file" and not self.opts["file"]:
            return "Click File… to choose the output data file."
        if mode == "add":
            for s in self.specs:
                if self.ds.has_name(s.name):
                    return f"{s.name} already exists in the active dataset."
        if self.n_cb.isChecked():
            n = self.n_name.text().strip()
            if not is_valid_name(n) or (mode == "add" and self.ds.has_name(n)):
                return f"'{n}' is not a valid new name for the number of cases."
        self.opts.update(n_break=self.n_cb.isChecked(), n_name=self.n_name.text().strip(),
                         mode=mode, dataset_name=self.ds_name.text().strip() or "DataSet_agg")
        self.modifies_data = mode == "add"
        return None

    def compute(self):
        o = self.opts
        specs = copy.deepcopy(self.specs)
        mode = "add" if o["mode"] == "add" else "new"
        new_ds, section = do.aggregate(self.ds, self.selected("break"), specs, mode,
                                       o["n_name"] if o["n_break"] else None,
                                       dataset_name=o["dataset_name"])
        if o["mode"] == "new":
            _open_new_dataset(self, new_ds)
        elif o["mode"] == "file":
            from rsstatchem.io import files

            path = o["file"]
            if Path(path).suffix.lower() == ".sav":
                files.write_sav(new_ds, path)
            else:
                if Path(path).suffix.lower() != files.PROJECT_EXT:
                    path += files.PROJECT_EXT
                files.save_project(new_ds, path)
            section.note("", f"Aggregated data written to {path}.")
        return section


# ================================================================== TRANSPOSE
class TransposeDialog(VariablePickerDialog):
    modifies_data = False

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Transpose", parent)
        self.add_target("vars", "Variable(s):")
        self.add_target("name", "Name Variable:", single=True, required=False)
        note = QLabel("The transposed data open in a new Data Editor window. String variables "
                      "(other than the name variable) cannot be transposed.")
        note.setWordWrap(True)
        note.setStyleSheet(GREY)
        self.options_layout.addWidget(note)
        self.finish()

    def validate(self):
        if not any(self.ds.var(v).type == NUMERIC for v in self.selected("vars")):
            return "Select at least one numeric variable to transpose."
        return None

    def compute(self):
        new_ds, section = do.transpose(self.ds, self.selected("vars"),
                                       self.selected_one("name"))
        _open_new_dataset(self, new_ds)
        return section


# ================================================================== RESTRUCTURE
class VarsToCasesDialog(VariablePickerDialog):
    modifies_data = False
    DEFAULTS = {"target": "trans1", "id_mode": "case", "id_name": "id", "index": "index1",
                "index_mode": "sequential", "drop_null": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Restructure Data Wizard - Variables to Cases", parent)
        self.setMinimumSize(720, 600)
        id_box = QGroupBox("Case Group Identification")
        il = QHBoxLayout(id_box)
        self.id_mode = QComboBox()
        for key, text in (("case", "Use case number"), ("var", "Use selected variable"),
                          ("none", "None")):
            self.id_mode.addItem(text, key)
        self.id_mode.setCurrentIndex(["case", "var", "none"].index(self.opts["id_mode"]))
        self.id_name = QLineEdit(self.opts["id_name"])
        self.id_name.setMaximumWidth(140)
        il.addWidget(self.id_mode)
        il.addWidget(QLabel("Name:"))
        il.addWidget(self.id_name)
        il.addStretch()
        self.layout().insertWidget(0, id_box)
        self.add_target("id", "Case group identification variable:", single=True,
                        required=False)
        trow = QHBoxLayout()
        trow.addWidget(QLabel("Target Variable:"))
        self.target = QLineEdit(self.opts["target"])
        trow.addWidget(self.target, 1)
        self._targets_column.addLayout(trow)
        self.add_target("vars", "Variables to be Transposed:")
        self.add_target("fixed", "Fixed Variable(s):", required=False)
        index_box = QGroupBox("Index Variable")
        ib = QHBoxLayout(index_box)
        ib.addWidget(QLabel("Name:"))
        self.index_name = QLineEdit(self.opts["index"])
        self.index_name.setMaximumWidth(140)
        ib.addWidget(self.index_name)
        self.index_mode = _radio_group(self, ib, [("sequential", "Sequential numbers"),
                                                  ("names", "Variable names")],
                                       self.opts["index_mode"])
        ib.addStretch()
        miss_box = QGroupBox("Handling of All Missing Values")
        mb = QHBoxLayout(miss_box)
        self.null = _radio_group(self, mb, [(False, "Keep and treat as system-missing"),
                                            (True, "Discard the data")],
                                 self.opts["drop_null"])
        mb.addStretch()
        self.options_layout.addWidget(index_box)
        self.options_layout.addWidget(miss_box)
        note = QLabel("Variables left in the list on the left are dropped from the new file. "
                      "The restructured data open in a new Data Editor window.")
        note.setStyleSheet(GREY)
        note.setWordWrap(True)
        self.options_layout.addWidget(note)
        self.id_mode.currentIndexChanged.connect(self._id_mode_changed)
        self.finish()
        self._id_mode_changed()

    def _id_mode_changed(self) -> None:
        mode = self.id_mode.currentData()
        self.id_name.setEnabled(mode == "case")

    def validate(self):
        mode = self.id_mode.currentData()
        if mode == "var" and not self.selected("id"):
            return "Move a variable to the case group identification list or choose " \
                   "another identification."
        names = [self.target.text().strip(), self.index_name.text().strip()]
        if mode == "case":
            names.append(self.id_name.text().strip())
        for n in names:
            if not is_valid_name(n):
                return f"'{n}' is not a valid variable name."
        lowered = [n.lower() for n in names] + [f.lower() for f in self.selected("fixed")]
        if len(set(lowered)) != len(lowered):
            return "The new variable names must be different from each other and from the " \
                   "fixed variables."
        kinds = {self.ds.var(v).type for v in self.selected("vars")}
        if len(kinds) > 1:
            return "The variables to be transposed must all be of the same type."
        self.opts.update(target=names[0], index=names[1], id_mode=mode,
                         id_name=self.id_name.text().strip(),
                         index_mode=_checked(self.index_mode), drop_null=_checked(self.null))
        return None

    def compute(self):
        o = self.opts
        id_var = self.selected_one("id") if o["id_mode"] == "var" else None
        fixed = self.selected("fixed")
        if o["id_mode"] != "var" and self.selected("id"):
            fixed = self.selected("id") + fixed
        new_ds, section = do.vars_to_cases(
            self.ds, [(o["target"], self.selected("vars"))], o["index"], o["index_mode"],
            o["id_name"] if o["id_mode"] == "case" else None, id_var, fixed, o["drop_null"])
        _open_new_dataset(self, new_ds)
        return section


class CasesToVarsDialog(VariablePickerDialog):
    modifies_data = False
    DEFAULTS = {"sort": True, "groupby": "variable", "separator": ".", "autofix": True}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Restructure Data Wizard - Cases to Variables", parent)
        self.add_target("id", "Identifier Variable(s):")
        self.add_target("index", "Index Variable(s):", required=False)
        self.add_checkbox("sort", "Sort the current data by the identifier and index "
                                  "variables")
        box = QGroupBox("Order of New Variable Groups")
        bl = QHBoxLayout(box)
        self.groupby = _radio_group(self, bl, [("variable", "Group by original variable"),
                                               ("index", "Group by index")],
                                    self.opts["groupby"])
        bl.addStretch()
        self.options_layout.addWidget(box)
        srow = QHBoxLayout()
        srow.addWidget(QLabel("Separator between variable name and index:"))
        self.sep = QLineEdit(self.opts["separator"])
        self.sep.setMaximumWidth(50)
        srow.addWidget(self.sep)
        srow.addStretch()
        self.options_layout.addLayout(srow)
        self.add_checkbox("autofix", "Keep variables that are constant within ID groups as "
                                     "single fixed variables")
        note = QLabel("All other variables are spread into one new variable per index value. "
                      "Without an index variable the cases of each group are numbered 1, 2, … "
                      "The restructured data open in a new Data Editor window.")
        note.setStyleSheet(GREY)
        note.setWordWrap(True)
        self.options_layout.addWidget(note)
        self.finish()

    def validate(self):
        sep = self.sep.text()
        if sep and not all(ch.isalnum() or ch in "._" for ch in sep):
            return "The separator may only contain letters, digits, '.' or '_'."
        self.opts.update(groupby=_checked(self.groupby), separator=sep)
        return None

    def compute(self):
        o = self.opts
        new_ds, section = do.cases_to_vars(self.ds, self.selected("id"),
                                           self.selected("index") or None, o["groupby"],
                                           o["separator"], o["autofix"], o["sort"])
        _open_new_dataset(self, new_ds)
        return section


class RestructureWizardDialog(QDialog):
    """Step 1 of the SPSS Restructure Data Wizard: what to do."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Restructure Data Wizard")
        self.setMinimumWidth(520)
        title = QLabel("<b>Welcome to the Restructure Data Wizard!</b><br>This wizard helps "
                       "you to restructure your data from multiple variables (columns) in a "
                       "single case to groups of related cases (rows) or vice versa, or you "
                       "can choose to transpose your data.")
        title.setWordWrap(True)
        box = QGroupBox("What do you want to do?")
        bl = QVBoxLayout(box)
        self.choice = {}
        group = QButtonGroup(self)
        for key, text, hint in (
                ("v2c", "Restructure selected variables into cases",
                 "Use when each case has several variables you want to stack into one "
                 "variable in separate cases (wide → long)."),
                ("c2v", "Restructure selected cases into variables",
                 "Use when groups of related cases should become one case with new variables "
                 "(long → wide)."),
                ("flip", "Transpose all data",
                 "All cases become variables and the selected variables become cases.")):
            rb = QRadioButton(text)
            group.addButton(rb)
            bl.addWidget(rb)
            lbl = QLabel(hint)
            lbl.setWordWrap(True)
            lbl.setStyleSheet(GREY + "margin-left:22px;")
            bl.addWidget(lbl)
            self.choice[key] = rb
        self.choice["v2c"].setChecked(True)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Next >")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(box)
        layout.addWidget(buttons)

    def selected(self) -> str:
        return _checked(self.choice)


def open_restructure(editor) -> None:
    dlg = RestructureWizardDialog(editor)
    if not dlg.exec():
        return
    cls = {"v2c": VarsToCasesDialog, "c2v": CasesToVarsDialog,
           "flip": TransposeDialog}[dlg.selected()]
    editor.run_analysis(cls)


# ================================================================== menus
MENUS = {
    "Transform": [
        ("Count Values within Cases...", CountValuesDialog),
        ("Shift Values...", ShiftValuesDialog),
        None,
        ("Automatic Recode...", AutoRecodeDialog),
        ("Visual Binning...", VisualBinningDialog),
        None,
        ("Rank Cases...", RankCasesDialog),
        None,
        ("Replace Missing Values...", ReplaceMissingDialog),
    ],
    "Data": [
        ("Identify Duplicate Cases...", IdentifyDuplicatesDialog),
        None,
        ("Merge Files", [("Add Cases...", AddCasesDialog),
                         ("Add Variables...", AddVariablesDialog)]),
        ("Restructure...", open_restructure),
        ("Transpose...", TransposeDialog),
        ("Aggregate...", AggregateDialog),
    ],
}


# ================================================================== selftest
def selftest() -> None:
    """Headless check of the core procedures on synthetic data (no widgets)."""
    rng = np.random.default_rng(7)
    df = pd.DataFrame({"id": np.arange(1, 13, dtype=float),
                       "g": np.repeat([1.0, 2.0, 3.0], 4),
                       "x": rng.normal(50, 10, 12).round(1),
                       "s": list("abcabcabcabc")})
    df.loc[3, "x"] = np.nan
    ds = Dataset(df)
    do.count_values(ds, "c", ["x", "g"], [("range", 1, 2), ("sysmis",)])
    assert ds.df["c"].iloc[3] == 2 and ds.df["c"].iloc[0] == 1
    do.automatic_recode(ds, ["s"], ["s_n"])
    assert list(ds.df["s_n"][:3]) == [1.0, 2.0, 3.0]
    do.visual_binning(ds, "x", "x_bin", do.percentile_cutpoints(ds.df["x"].to_numpy(), 2))
    assert ds.df["x_bin"].dropna().between(1, 3).all()
    do.rank_cases(ds, ["x"], ["RANK", "NORMAL"], by=["g"])
    from scipy import stats as _stats
    sub = ds.df.loc[ds.df["g"] == 1, "x"].dropna()
    assert np.allclose(ds.df.loc[sub.index, "Rx"], _stats.rankdata(sub))
    do.replace_missing(ds, [("x", "x_1", "LINT", None)])
    assert ds.df["x_1"].iloc[3] == (ds.df["x"].iloc[2] + ds.df["x"].iloc[4]) / 2
    do.shift_values(ds, [("x", "x_lag1", True, 1)])
    assert ds.df["x_lag1"].iloc[1] == ds.df["x"].iloc[0]
    new, _ = do.aggregate(ds, ["g"], [do.AggSpec("m", "MEAN", "x")], mode="new",
                          n_break="N_BREAK")
    ref = ds.df.groupby("g")["x"].mean().to_numpy()
    assert np.allclose(new.df["m"].to_numpy(), ref) and list(new.df["N_BREAK"]) == [4] * 3
    long, _ = do.vars_to_cases(ds, [("v", ["x", "x_1"])], id_var="id", fixed=["g"])
    assert long.n_cases == 24
    wide, _ = do.cases_to_vars(long, ["id"], ["index1"])
    assert np.allclose(wide.df["v.2"].to_numpy(), ds.df["x_1"].to_numpy())
    flipped, _ = do.transpose(ds, ["x", "g"])
    assert flipped.n_cases == 2 and flipped.n_vars == 13
    other = Dataset(pd.DataFrame({"id": [13.0], "x": [1.0]}))
    n = ds.n_cases
    _, unpaired, _ = do.match_variables(ds, other)
    do.add_cases(ds, other, keep_active=unpaired, source_var="src")
    assert ds.n_cases == n + 1 and ds.df["src"].iloc[-1] == 1
    look = Dataset(pd.DataFrame({"g": [1.0, 2.0], "town": ["A", "B"]}))
    do.add_variables(ds, look, "table", ["g"])
    assert ds.df.loc[ds.df["g"] == 2, "town"].eq("B").all()
    do.identify_duplicates(ds, ["g"], primary="last")
    assert int(ds.df["PrimaryLast"].sum()) == ds.df["g"].nunique(dropna=False)

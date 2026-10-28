"""Helpers for the tabbed SPSS dialogs of WP9 (Nearest Neighbor, Multilayer Perceptron, Radial
Basis Function, Partial Least Squares): the variable picker becomes the "Variables" tab and
further tabs hold option widgets bound live to ``dialog.opts``."""
from __future__ import annotations

from typing import Callable

from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFormLayout, QGridLayout, QGroupBox, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem, QRadioButton,
                               QTabWidget, QVBoxLayout, QWidget)
from PySide6.QtCore import Qt

from rsstatchem.ui.dialogs.base import VariablePickerDialog


class Form:
    """Option widgets that write into ``opts`` as soon as they change."""

    def __init__(self, opts: dict, layout):
        self.opts, self.layout = opts, layout

    def group(self, title: str) -> "Form":
        box = QGroupBox(title)
        lay = QVBoxLayout(box)
        self.layout.addWidget(box)
        return Form(self.opts, lay)

    def label(self, text: str) -> QLabel:
        lab = QLabel(text)
        lab.setWordWrap(True)
        self.layout.addWidget(lab)
        return lab

    def check(self, key: str, text: str) -> QCheckBox:
        cb = QCheckBox(text)
        cb.setChecked(bool(self.opts.get(key)))
        cb.toggled.connect(lambda on: self.opts.__setitem__(key, on))
        self.layout.addWidget(cb)
        return cb

    def radio(self, key: str, options: list[tuple[object, str]]) -> dict:
        group = QButtonGroup(self.layout.parentWidget() or None)
        buttons = {}
        for value, text in options:
            rb = QRadioButton(text)
            rb.setChecked(self.opts.get(key) == value)
            rb.toggled.connect(lambda on, v=value: on and self.opts.__setitem__(key, v))
            group.addButton(rb)
            self.layout.addWidget(rb)
            buttons[value] = rb
        if not any(b.isChecked() for b in buttons.values()):
            next(iter(buttons.values())).setChecked(True)
        buttons["_group"] = group
        return buttons

    def number(self, key: str, text: str, minimum: float, maximum: float,
               decimals: int = 0, step: float | None = None) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        if step:
            spin.setSingleStep(step)
        spin.setValue(float(self.opts.get(key) or 0))
        conv = (lambda v: int(round(v))) if decimals == 0 else float
        spin.valueChanged.connect(lambda v: self.opts.__setitem__(key, conv(v)))
        self._row(text, spin)
        return spin

    def combo(self, key: str, text: str, options: list[tuple[object, str]]) -> QComboBox:
        box = QComboBox()
        for value, label in options:
            box.addItem(label, value)
        values = [v for v, _ in options]
        if self.opts.get(key) in values:
            box.setCurrentIndex(values.index(self.opts.get(key)))
        box.currentIndexChanged.connect(lambda i: self.opts.__setitem__(key, values[i]))
        self._row(text, box)
        return box

    def text(self, key: str, text: str, placeholder: str = "") -> QLineEdit:
        edit = QLineEdit(str(self.opts.get(key) or ""))
        edit.setPlaceholderText(placeholder)
        edit.textChanged.connect(lambda t: self.opts.__setitem__(key, t))
        self._row(text, edit)
        return edit

    def variable(self, key: str, text: str, names: list[str]) -> QComboBox:
        return self.combo(key, text, [(None, "(none)")] + [(n, n) for n in names])

    def _row(self, text: str, widget) -> None:
        row = QFormLayout()
        row.addRow(text, widget)
        self.layout.addLayout(row)

    def stretch(self) -> None:
        self.layout.addStretch()


class TabbedPickerDialog(VariablePickerDialog):
    """VariablePickerDialog whose picker sits on the first tab ("Variables" / "Fields")."""

    def make_tabs(self, pages: list[tuple[str, Callable[[Form], None]]],
                  first: str = "Variables") -> QTabWidget:
        layout = self.layout()
        body = layout.takeAt(0).layout()
        extra = layout.takeAt(0).layout()   # options_layout below the lists
        self.tabs = QTabWidget()
        page = QWidget()
        pl = QVBoxLayout(page)
        pl.addLayout(body, 1)
        pl.addLayout(extra)
        self.tabs.addTab(page, first)
        for title, build in pages:
            w = QWidget()
            lay = QVBoxLayout(w)
            build(Form(self.opts, lay))
            lay.addStretch()
            self.tabs.addTab(w, title)
        layout.insertWidget(0, self.tabs, 1)
        self.setMinimumSize(760, 560)
        return self.tabs


class OptionsDialog(QDialog):
    """A "Continue / Cancel" sub-dialog built from :class:`Form` groups on a copy of opts."""

    def __init__(self, title: str, opts: dict, parent=None, columns: int = 2):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.values = dict(opts)
        self._grid = QGridLayout()
        self._columns, self._n = columns, 0
        lay = QVBoxLayout(self)
        lay.addLayout(self._grid)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Continue")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def group(self, title: str, span: int = 1) -> Form:
        box = QGroupBox(title)
        inner = QVBoxLayout(box)
        row, col = divmod(self._n, self._columns)
        self._grid.addWidget(box, row, col, 1, span)
        self._n += span
        return Form(self.values, inner)

    def checklist(self, title: str, key: str, items: list[tuple[object, str]]) -> QListWidget:
        form = self.group(title)
        lst = QListWidget()
        chosen = set(self.values.get(key) or [])
        for value, text in items:
            it = QListWidgetItem(text)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if value in chosen else
                             Qt.CheckState.Unchecked)
            it.setData(Qt.ItemDataRole.UserRole, value)
            lst.addItem(it)

        def changed(_):
            self.values[key] = [lst.item(i).data(Qt.ItemDataRole.UserRole)
                                for i in range(lst.count())
                                if lst.item(i).checkState() == Qt.CheckState.Checked]
        lst.itemChanged.connect(changed)
        form.layout.addWidget(lst)
        return lst

    def run(self, target: dict) -> bool:
        if self.exec():
            target.update(self.values)
            return True
        return False


def parse_list(text, cast=float, allow_none: bool = False) -> list:
    out = []
    """Values separated by spaces or semicolons; a comma is read as a decimal point."""
    for tok in str(text or "").replace(";", " ").split():
        if allow_none and tok.lower() in ("auto", "none", "default"):
            out.append(None)
        else:
            out.append(cast(float(tok.replace(",", "."))) if cast is int else
                       cast(tok.replace(",", ".")))
    return out

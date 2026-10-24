"""The SPSS analysis-dialog pattern: source variable list, arrow buttons, target lists,
option buttons on the right and OK / Reset / Cancel at the bottom.

Dialogs remember their last selections and options for the session, as SPSS does.
"""
from __future__ import annotations

import copy
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup, QCheckBox,
                               QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
                               QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMessageBox, QPushButton,
                               QRadioButton, QVBoxLayout, QWidget)

from rsstatchem.core.dataset import NUMERIC, Dataset, Variable, format_code
from rsstatchem.ui import icons

NAME_ROLE = Qt.ItemDataRole.UserRole
BASE_TEXT_ROLE = Qt.ItemDataRole.UserRole + 2
PAIR_ROLE = Qt.ItemDataRole.UserRole + 3


def is_numeric(var: Variable) -> bool:
    return var.type == NUMERIC


def variable_item(var: Variable) -> QListWidgetItem:
    text = f"{var.label} [{var.name}]" if var.label else var.name
    item = QListWidgetItem(icons.variable_icon(var), text)
    item.setData(NAME_ROLE, var.name)
    item.setData(BASE_TEXT_ROLE, text)
    item.setToolTip(f"{var.name} — {var.type}, {var.measure}")
    return item


class _Target:
    def __init__(self, key: str, widget: QListWidget, accepts: Callable[[Variable], bool],
                 single: bool, min_items: int, arrow: QPushButton):
        self.key, self.widget, self.accepts = key, widget, accepts
        self.single, self.min_items, self.arrow = single, min_items, arrow


class _PairTarget:
    def __init__(self, key: str, widget: QListWidget, arrow: QPushButton,
                 accepts: Callable[[Variable], bool]):
        self.key, self.widget, self.arrow, self.accepts = key, widget, arrow, accepts
        self.pending: str | None = None


class SubDialog(QDialog):
    """Small builder for the "Options…", "Statistics…", "Post Hoc…" style sub-dialogs.

    Values are read back with :meth:`values`; widgets are keyed by option name.
    """

    def __init__(self, title: str, parent: QWidget | None = None, columns: int = 2):
        super().__init__(parent)
        self.setWindowTitle(title)
        self._getters: dict[str, Callable[[], object]] = {}
        self._grid = QGridLayout()
        self._columns = columns
        self._n = 0
        layout = QVBoxLayout(self)
        layout.addLayout(self._grid)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Continue")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def group(self, title: str, span: int = 1) -> QVBoxLayout:
        box = QGroupBox(title)
        inner = QVBoxLayout(box)
        row, col = divmod(self._n, self._columns)
        self._grid.addWidget(box, row, col, 1, span)
        self._n += span
        return inner

    def check(self, layout, key: str, text: str, checked: bool) -> QCheckBox:
        cb = QCheckBox(text)
        cb.setChecked(bool(checked))
        layout.addWidget(cb)
        self._getters[key] = cb.isChecked
        return cb

    def radio(self, layout, key: str, options: list[tuple[object, str]], current) -> dict:
        group = QButtonGroup(self)
        buttons = {}
        for value, text in options:
            rb = QRadioButton(text)
            rb.setChecked(value == current)
            group.addButton(rb)
            layout.addWidget(rb)
            buttons[value] = rb
        if not any(b.isChecked() for b in buttons.values()):
            next(iter(buttons.values())).setChecked(True)
        self._getters[key] = lambda: next(v for v, b in buttons.items() if b.isChecked())
        return buttons

    def number(self, layout, key: str, text: str, value: float, minimum: float,
               maximum: float, decimals: int = 2, suffix: str = "") -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setValue(float(value))
        spin.setSuffix(suffix)
        row = QFormLayout()
        row.addRow(text, spin)
        layout.addLayout(row)
        self._getters[key] = spin.value
        return spin

    def values(self) -> dict:
        return {k: get() for k, get in self._getters.items()}

    def run(self) -> dict | None:
        return self.values() if self.exec() else None


class VariablePickerDialog(QDialog):
    DEFAULTS: dict = {}
    _memory: dict[str, dict] = {}
    _states: dict[str, dict] = {}

    def __init__(self, ds: Dataset, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.ds = ds
        self.setWindowTitle(title)
        self.setMinimumSize(600, 380)
        self._targets: list[_Target] = []
        self._pairs: list[_PairTarget] = []
        name = type(self).__name__
        self.opts = copy.deepcopy(self.DEFAULTS)
        self.opts.update(copy.deepcopy(self._states.get(name, {})))

        self.source = QListWidget()
        self.source.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.source.itemDoubleClicked.connect(lambda _: self._move_to_first_accepting())
        self.source.itemSelectionChanged.connect(self._update_arrows)
        self.source.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        for var in ds.variables:
            self.source.addItem(variable_item(var))
        self._targets_column = QVBoxLayout()

        self._option_column = QVBoxLayout()
        self._option_column.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.options_layout = QVBoxLayout()

        buttons = QDialogButtonBox()
        self.ok_button = buttons.addButton(QDialogButtonBox.StandardButton.Ok)
        reset = buttons.addButton(QDialogButtonBox.StandardButton.Reset)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        self.ok_button.setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        reset.clicked.connect(self.reset)

        body = QHBoxLayout()
        body.addWidget(self.source, 1)
        body.addLayout(self._targets_column, 1)
        body.addLayout(self._option_column)
        layout = QVBoxLayout(self)
        layout.addLayout(body, 1)
        layout.addLayout(self.options_layout)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------ building
    def _target_block(self, label: str, widget: QListWidget, arrow: QPushButton,
                      stretch: int) -> None:
        arrow_column = QVBoxLayout()
        arrow_column.addSpacing(QLabel(label).sizeHint().height() + 4)
        arrow_column.addWidget(arrow)
        arrow_column.addStretch()
        list_column = QVBoxLayout()
        list_column.addWidget(QLabel(label))
        list_column.addWidget(widget, 1)
        block = QHBoxLayout()
        block.addLayout(arrow_column)
        block.addLayout(list_column, 1)
        self._targets_column.addLayout(block, stretch)

    @staticmethod
    def _list_widget() -> QListWidget:
        widget = QListWidget()
        widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return widget

    def add_target(self, key: str, label: str, accepts: Callable[[Variable], bool] | None = None,
                   single: bool = False, required: bool = True,
                   min_items: int | None = None) -> QListWidget:
        arrow = QPushButton("▶")
        arrow.setObjectName("arrow")
        widget = self._list_widget()
        if single:
            widget.setFixedHeight(30)
        minimum = (1 if required else 0) if min_items is None else min_items
        target = _Target(key, widget, accepts or (lambda v: True), single, minimum, arrow)
        self._targets.append(target)
        arrow.clicked.connect(lambda _=False, t=target: self._arrow_clicked(t))
        widget.itemDoubleClicked.connect(lambda _, t=target: self._move_back(t))
        widget.itemSelectionChanged.connect(self._update_arrows)
        self._target_block(label, widget, arrow, 0 if single else 1)
        self._update_ok()
        return widget

    def add_pair_target(self, key: str, label: str,
                        accepts: Callable[[Variable], bool] = is_numeric) -> QListWidget:
        """Paired variables (Variable1 – Variable2). Variables stay in the source list."""
        arrow = QPushButton("▶")
        arrow.setObjectName("arrow")
        widget = self._list_widget()
        target = _PairTarget(key, widget, arrow, accepts)
        self._pairs.append(target)
        arrow.clicked.connect(lambda _=False, t=target: self._pair_arrow(t))
        widget.itemDoubleClicked.connect(lambda item, t=target: self._remove_pair(t, item))
        widget.itemSelectionChanged.connect(self._update_arrows)
        self._target_block(label, widget, arrow, 1)
        return widget

    def add_option_button(self, text: str, callback: Callable[[], None]) -> QPushButton:
        button = QPushButton(text)
        button.clicked.connect(callback)
        self._option_column.addWidget(button)
        return button

    def add_checkbox(self, key: str, text: str) -> QCheckBox:
        cb = QCheckBox(text)
        cb.setChecked(bool(self.opts.get(key)))
        cb.toggled.connect(lambda on, k=key: self.opts.__setitem__(k, on))
        self.options_layout.addWidget(cb)
        return cb

    def finish(self) -> None:
        """Call at the end of the subclass constructor to restore remembered selections."""
        memory = self._memory.get(type(self).__name__, {})
        for target in self._targets:
            for name in memory.get(target.key, []):
                if self.ds.has_name(name):
                    self._take(name, target)
        for target in self._pairs:
            for a, b in memory.get(target.key, []):
                if self.ds.has_name(a) and self.ds.has_name(b):
                    self._add_pair(target, a, b)
        self._update_arrows()

    # ------------------------------------------------------------------ moving
    def _source_item(self, name: str) -> QListWidgetItem | None:
        for i in range(self.source.count()):
            if self.source.item(i).data(NAME_ROLE) == name:
                return self.source.item(i)
        return None

    def _take(self, name: str, target: _Target) -> bool:
        item = self._source_item(name)
        if item is None or not target.accepts(self.ds.var(name)):
            return False
        if target.single and target.widget.count():
            self._move_back(target, all_items=True)
        self.source.takeItem(self.source.row(item))
        target.widget.addItem(item)
        return True

    def _arrow_clicked(self, target: _Target) -> None:
        if target.widget.selectedItems() and not self.source.selectedItems():
            self._move_back(target)
            return
        names = [i.data(NAME_ROLE) for i in self.source.selectedItems()]
        if target.single:
            names = names[:1]
        if [n for n in names if not self._take(n, target)]:
            QApplication.beep()  # e.g. a string variable offered to a numeric-only list
        self._update_arrows()

    def _move_to_first_accepting(self) -> None:
        for name in [i.data(NAME_ROLE) for i in self.source.selectedItems()]:
            for target in self._targets:
                if (not target.single or not target.widget.count()) and self._take(name, target):
                    break
        self._update_arrows()

    def _move_back(self, target: _Target, all_items: bool = False) -> None:
        items = [target.widget.item(i) for i in range(target.widget.count())] if all_items \
            else target.widget.selectedItems()
        for item in items:
            target.widget.takeItem(target.widget.row(item))
            item.setText(item.data(BASE_TEXT_ROLE))
            order = self.ds.index_of(item.data(NAME_ROLE))
            pos = 0
            while pos < self.source.count() and \
                    self.ds.index_of(self.source.item(pos).data(NAME_ROLE)) < order:
                pos += 1
            self.source.insertItem(pos, item)
        self._update_arrows()

    # ------------------------------------------------------------------ pairs
    def _pair_text(self, a: str, b: str | None) -> str:
        return f"{a}  –  {b if b else '?'}"

    def _add_pair(self, target: _PairTarget, a: str, b: str | None) -> None:
        if b is None:
            target.pending = a
            item = QListWidgetItem(self._pair_text(a, None))
            item.setData(PAIR_ROLE, (a, None))
            target.widget.addItem(item)
            return
        last = target.widget.item(target.widget.count() - 1) if target.widget.count() else None
        if last is not None and last.data(PAIR_ROLE)[1] is None:
            target.widget.takeItem(target.widget.count() - 1)
        item = QListWidgetItem(self._pair_text(a, b))
        item.setData(PAIR_ROLE, (a, b))
        target.widget.addItem(item)
        target.pending = None

    def _pair_arrow(self, target: _PairTarget) -> None:
        if target.widget.selectedItems() and not self.source.selectedItems():
            for item in target.widget.selectedItems():
                self._remove_pair(target, item)
            return
        names = [i.data(NAME_ROLE) for i in self.source.selectedItems()]
        if any(not target.accepts(self.ds.var(n)) for n in names):
            QApplication.beep()
            return
        if target.pending:
            names = [target.pending] + names
            target.pending = None
        while len(names) >= 2:
            self._add_pair(target, names.pop(0), names.pop(0))
        if names:
            self._add_pair(target, names[0], None)
        self.source.clearSelection()
        self._update_arrows()

    def _remove_pair(self, target: _PairTarget, item: QListWidgetItem) -> None:
        if item.data(PAIR_ROLE)[1] is None:
            target.pending = None
        target.widget.takeItem(target.widget.row(item))
        self._update_arrows()

    def pairs(self, key: str) -> list[tuple[str, str]]:
        target = next(t for t in self._pairs if t.key == key)
        out = []
        for i in range(target.widget.count()):
            a, b = target.widget.item(i).data(PAIR_ROLE)
            if b is not None:
                out.append((a, b))
        return out

    # ------------------------------------------------------------------ state
    def _update_arrows(self) -> None:
        for target in self._targets:
            back = bool(target.widget.selectedItems()) and not self.source.selectedItems()
            target.arrow.setText("◀" if back else "▶")
        for target in self._pairs:
            back = bool(target.widget.selectedItems()) and not self.source.selectedItems()
            target.arrow.setText("◀" if back else "▶")
        self._update_ok()
        self.targets_changed()

    def targets_changed(self) -> None:
        """Hook for subclasses (e.g. enable "Define Groups…" when a grouping variable is set)."""

    def _update_ok(self) -> None:
        if not hasattr(self, "ok_button"):
            return
        ok = all(t.widget.count() >= t.min_items for t in self._targets)
        ok = ok and all(self.pairs(t.key) for t in self._pairs)
        self.ok_button.setEnabled(ok)

    def set_item_suffix(self, key: str, suffix: str) -> None:
        widget = next(t.widget for t in self._targets if t.key == key)
        for i in range(widget.count()):
            item = widget.item(i)
            item.setText(f"{item.data(NAME_ROLE)}{suffix}")

    def selected(self, key: str) -> list[str]:
        for target in self._targets:
            if target.key == key:
                return [target.widget.item(i).data(NAME_ROLE)
                        for i in range(target.widget.count())]
        raise KeyError(key)

    def selected_one(self, key: str) -> str | None:
        names = self.selected(key)
        return names[0] if names else None

    def reset(self) -> None:
        for target in self._targets:
            self._move_back(target, all_items=True)
        for target in self._pairs:
            target.widget.clear()
            target.pending = None
        self.opts = copy.deepcopy(self.DEFAULTS)
        self._update_arrows()

    def validate(self) -> str | None:
        """Return an error message to keep the dialog open."""
        return None

    def accept(self) -> None:
        error = self.validate()
        if error:
            QMessageBox.warning(self, self.windowTitle(), error)
            return
        memory = {t.key: self.selected(t.key) for t in self._targets}
        memory.update({t.key: self.pairs(t.key) for t in self._pairs})
        self._memory[type(self).__name__] = memory
        self._states[type(self).__name__] = copy.deepcopy(self.opts)
        super().accept()

    # ------------------------------------------------------------------ shared sub-dialogs
    def ci_missing_options(self, missing: bool = True) -> None:
        dlg = SubDialog(f"{self.windowTitle()}: Options", self, columns=1)
        box = dlg.group("Confidence Interval")
        dlg.number(box, "level", "Confidence Interval Percentage:", self.opts.get("level", 95),
                   50, 99.99, 2, " %")
        if missing:
            box = dlg.group("Missing Values")
            dlg.radio(box, "listwise", [(False, "Exclude cases analysis by analysis"),
                                        (True, "Exclude cases listwise")],
                      self.opts.get("listwise", False))
        values = dlg.run()
        if values:
            self.opts.update(values)

    def level(self) -> float:
        return float(self.opts.get("level", 95)) / 100


class DefineGroupsDialog(QDialog):
    """Group 1 / Group 2 codes, or a cut point (independent t test, Mann-Whitney)."""

    def __init__(self, var: Variable, current: dict | None, allow_cut: bool = True,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.var = var
        self.setWindowTitle("Define Groups")
        current = current or {}
        values = current.get("groups") or self._guess()
        self.use_values = QRadioButton("Use specified values")
        self.g1 = QLineEdit(format_code(values[0]) if values else "")
        self.g2 = QLineEdit(format_code(values[1]) if values and len(values) > 1 else "")
        self.use_cut = QRadioButton("Cut point:")
        self.cut = QLineEdit(format_code(current["cut"]) if current.get("cut") is not None
                             else "")
        form = QFormLayout()
        form.addRow(self.use_values)
        form.addRow("Group 1:", self.g1)
        form.addRow("Group 2:", self.g2)
        if allow_cut and var.type == NUMERIC:
            form.addRow(self.use_cut, self.cut)
        (self.use_cut if current.get("cut") is not None else self.use_values).setChecked(True)
        if var.value_labels:
            hint = ", ".join(f"{format_code(k)} = {v}" for k, v in var.value_labels.items())
            label = QLabel(hint)
            label.setWordWrap(True)
            label.setStyleSheet("color:#666")
            form.addRow(label)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Continue")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _guess(self) -> list:
        if self.var.value_labels and len(self.var.value_labels) >= 2:
            return list(self.var.value_labels)[:2]
        return []

    def result(self) -> dict | None:
        try:
            if self.use_cut.isChecked():
                return {"groups": None, "cut": float(self.cut.text().replace(",", "."))}
            parse = (lambda t: float(t.replace(",", "."))) if self.var.type == NUMERIC \
                else (lambda t: t.strip())
            return {"groups": (parse(self.g1.text()), parse(self.g2.text())), "cut": None}
        except ValueError:
            return None


class DefineRangeDialog(SubDialog):
    def __init__(self, current: tuple | None, parent: QWidget | None = None):
        super().__init__("Define Range", parent, columns=1)
        box = self.group("Range for Grouping Variable")
        lo, hi = current or (1, 2)
        self.number(box, "lo", "Minimum:", lo, -1e9, 1e9, 0)
        self.number(box, "hi", "Maximum:", hi, -1e9, 1e9, 0)

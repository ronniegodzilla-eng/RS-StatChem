"""A source/target list dialog whose items can be variables *or* multiple response sets.

Used by Custom Tables and the Multiple Response procedures, where ``$sets`` sit next to the
variables. Items move with the arrow buttons, by double-click or by drag and drop between the
lists (Custom Tables' "drag variables to rows and columns").
"""
from __future__ import annotations

import copy
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QDialog, QDialogButtonBox,
                               QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox,
                               QPushButton, QVBoxLayout, QWidget)

from rsstatchem.core import mresponse as mr
from rsstatchem.core.dataset import Dataset
from rsstatchem.ui import icons

NAME_ROLE = Qt.ItemDataRole.UserRole


def source_items(ds: Dataset, variables: bool = True, sets: bool = True,
                 accept: Callable | None = None) -> list[QListWidgetItem]:
    items = []
    if variables:
        for v in ds.variables:
            if accept is not None and not accept(v.name):
                continue
            text = f"{v.label} [{v.name}]" if v.label else v.name
            item = QListWidgetItem(icons.variable_icon(v), text)
            item.setData(NAME_ROLE, v.name)
            item.setToolTip(f"{v.name} — {v.type}, {v.measure}")
            items.append(item)
    if sets:
        for s in mr.get_sets(ds):
            item = QListWidgetItem(f"{s.label} [{s.name}]" if s.label else s.name)
            item.setIcon(icons.icon("labels"))
            item.setData(NAME_ROLE, s.name)
            item.setToolTip(f"Multiple response set ({s.kind}): {', '.join(s.variables)}")
            items.append(item)
    return items


class _DragList(QListWidget):
    def __init__(self, owner: "ListPickerDialog", key: str, single: bool = False):
        super().__init__()
        self.owner, self.key, self.single = owner, key, single
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def dropEvent(self, event) -> None:
        source = event.source()
        if isinstance(source, _DragList) and source is not self:
            names = [i.data(NAME_ROLE) for i in source.selectedItems()]
            self.owner.move_names(names, source, self)
            event.setDropAction(Qt.DropAction.IgnoreAction)
            event.accept()
            return
        super().dropEvent(event)
        self.owner.changed()


class ListPickerDialog(QDialog):
    DEFAULTS: dict = {}
    _memory: dict[str, dict] = {}
    _states: dict[str, dict] = {}

    def __init__(self, ds: Dataset, title: str, parent: QWidget | None = None,
                 variables: bool = True, sets: bool = True):
        super().__init__(parent)
        self.ds = ds
        self.setWindowTitle(title)
        self.setMinimumSize(640, 420)
        name = type(self).__name__
        self.opts = copy.deepcopy(self.DEFAULTS)
        self.opts.update(copy.deepcopy(self._states.get(name, {})))
        self.targets: dict[str, _DragList] = {}
        self.accepts: dict[str, Callable] = {}
        self.minimum: dict[str, int] = {}
        self.source = _DragList(self, "source")
        for item in source_items(ds, variables, sets):
            self.source.addItem(item)
        self.source.itemDoubleClicked.connect(lambda _: self._to_first())
        self._targets_col = QVBoxLayout()
        self.button_col = QVBoxLayout()
        self.button_col.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.options_layout = QVBoxLayout()
        buttons = QDialogButtonBox()
        self.ok_button = buttons.addButton(QDialogButtonBox.StandardButton.Ok)
        reset = buttons.addButton(QDialogButtonBox.StandardButton.Reset)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        reset.clicked.connect(self.reset)
        body = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(QLabel("Variables and sets:"))
        left.addWidget(self.source, 1)
        body.addLayout(left, 1)
        body.addLayout(self._targets_col, 1)
        body.addLayout(self.button_col)
        layout = QVBoxLayout(self)
        layout.addLayout(body, 1)
        layout.addLayout(self.options_layout)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------ building
    def add_target(self, key: str, label: str, single: bool = False,
                   accepts: Callable[[str], bool] | None = None, min_items: int = 1,
                   extra: QWidget | None = None) -> _DragList:
        widget = _DragList(self, key, single)
        if single:
            widget.setFixedHeight(30)
        arrow = QPushButton("▶")
        arrow.setObjectName("arrow")
        arrow.clicked.connect(lambda _=False, w=widget: self._arrow(w))
        widget.itemDoubleClicked.connect(lambda _, w=widget: self.move_names(
            [i.data(NAME_ROLE) for i in w.selectedItems()], w, self.source))
        self.targets[key] = widget
        self.accepts[key] = accepts or (lambda name: True)
        self.minimum[key] = min_items
        column = QVBoxLayout()
        head = QHBoxLayout()
        head.addWidget(QLabel(label))
        if extra is not None:
            head.addStretch()
            head.addWidget(extra)
        column.addLayout(head)
        column.addWidget(widget, 1)
        row = QHBoxLayout()
        arrows = QVBoxLayout()
        arrows.addSpacing(24)
        arrows.addWidget(arrow)
        arrows.addStretch()
        row.addLayout(arrows)
        row.addLayout(column, 1)
        self._targets_col.addLayout(row, 0 if single else 1)
        self.changed()
        return widget

    def add_button(self, text: str, callback: Callable[[], None]) -> QPushButton:
        b = QPushButton(text)
        b.clicked.connect(callback)
        self.button_col.addWidget(b)
        return b

    def finish(self) -> None:
        memory = self._memory.get(type(self).__name__, {})
        names = {self.source.item(i).data(NAME_ROLE) for i in range(self.source.count())}
        for key, chosen in memory.items():
            if key in self.targets:
                self.move_names([n for n in chosen if n in names], self.source,
                                self.targets[key])
        self.changed()

    # ------------------------------------------------------------------ moving
    def _item(self, widget: QListWidget, name: str) -> QListWidgetItem | None:
        for i in range(widget.count()):
            if widget.item(i).data(NAME_ROLE) == name:
                return widget.item(i)
        return None

    def move_names(self, names: list[str], src: QListWidget, dst: QListWidget) -> None:
        refused = False
        for name in names:
            item = self._item(src, name)
            if item is None:
                continue
            if dst is not self.source:
                key = dst.key
                if not self.accepts[key](name):
                    refused = True
                    continue
                if dst.single and dst.count():
                    self.move_names([dst.item(0).data(NAME_ROLE)], dst, self.source)
            src.takeItem(src.row(item))
            dst.addItem(item)
        if refused:
            QApplication.beep()
        if dst is self.source:
            self._resort_source()
        self.changed()

    def _resort_source(self) -> None:
        order = {item.data(NAME_ROLE): i for i, item in enumerate(source_items(self.ds))}
        items = [self.source.takeItem(0) for _ in range(self.source.count())]
        for item in sorted(items, key=lambda it: order.get(it.data(NAME_ROLE), 1e9)):
            self.source.addItem(item)

    def _arrow(self, widget: _DragList) -> None:
        if widget.selectedItems() and not self.source.selectedItems():
            self.move_names([i.data(NAME_ROLE) for i in widget.selectedItems()], widget,
                            self.source)
            return
        names = [i.data(NAME_ROLE) for i in self.source.selectedItems()]
        if widget.single:
            names = names[:1]
        self.move_names(names, self.source, widget)

    def _to_first(self) -> None:
        for name in [i.data(NAME_ROLE) for i in self.source.selectedItems()]:
            for key, widget in self.targets.items():
                if self.accepts[key](name) and (not widget.single or not widget.count()):
                    self.move_names([name], self.source, widget)
                    break

    def selected(self, key: str) -> list[str]:
        w = self.targets[key]
        return [w.item(i).data(NAME_ROLE) for i in range(w.count())]

    def set_selected(self, key: str, names: list[str]) -> None:
        self.move_names(names, self.source, self.targets[key])

    def changed(self) -> None:
        if hasattr(self, "ok_button"):
            self.ok_button.setEnabled(all(self.targets[k].count() >= m
                                          for k, m in self.minimum.items()))
        self.targets_changed()

    def targets_changed(self) -> None:
        """Hook for subclasses."""

    def reset(self) -> None:
        for key, widget in self.targets.items():
            self.move_names(self.selected(key), widget, self.source)
        self.opts = copy.deepcopy(self.DEFAULTS)

    def validate(self) -> str | None:
        return None

    def accept(self) -> None:
        error = self.validate()
        if error:
            QMessageBox.warning(self, self.windowTitle(), error)
            return
        self._memory[type(self).__name__] = {k: self.selected(k) for k in self.targets}
        self._states[type(self).__name__] = copy.deepcopy(self.opts)
        super().accept()

"""Qt wrapper around a :class:`Dataset` that broadcasts edits to every view showing it and
records them on an undo stack.

Single-cell edits store only the old and new value. Anything else (paste, clear, variable
properties, Compute/Recode, sort, insert/delete) stores a snapshot of the data before and
after, which keeps every command trivially correct. If a change raises midway the dataset is
rolled back to its snapshot, so a failed Compute never leaves half-written data.
"""
from __future__ import annotations

import copy
from typing import Callable

import pandas as pd
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QUndoCommand, QUndoStack

from rsstatchem.core.dataset import Dataset

UNDO_LIMIT = 50


class _Snapshot:
    def __init__(self, ds: Dataset):
        self.df = ds.df.copy(deep=True)
        self.variables = copy.deepcopy(ds.variables)
        self.meta = copy.deepcopy(ds.meta)

    def restore(self, ds: Dataset) -> None:
        ds.df = self.df.copy(deep=True)
        ds.variables = copy.deepcopy(self.variables)
        ds.meta = copy.deepcopy(self.meta)


class _SnapshotCommand(QUndoCommand):
    def __init__(self, doc: "DatasetDocument", text: str, before: _Snapshot, after: _Snapshot):
        super().__init__(text)
        self.doc, self.before, self.after = doc, before, after
        self._applied = True  # the change already happened when the command was pushed

    def redo(self) -> None:
        if self._applied:
            self._applied = False
            return
        self.after.restore(self.doc.ds)
        self.doc._after_edit(True)

    def undo(self) -> None:
        self.before.restore(self.doc.ds)
        self.doc._after_edit(True)


class _CellCommand(QUndoCommand):
    def __init__(self, doc: "DatasetDocument", row: int, col: int, old, new):
        super().__init__("Edit Cell")
        self.doc, self.row, self.col, self.old, self.new = doc, row, col, old, new
        self._applied = True

    def _put(self, value) -> None:
        self.doc.ds.df.iat[self.row, self.col] = value
        self.doc.ds.modified = True
        self.doc._after_edit(False, (self.row, self.col, self.row, self.col))

    def redo(self) -> None:
        if self._applied:
            self._applied = False
            return
        self._put(self.new)

    def undo(self) -> None:
        self._put(self.old)


class DatasetDocument(QObject):
    structure_changed = Signal()             # rows/variables added, removed, renamed, retyped
    values_changed = Signal(int, int, int, int)  # top, left, bottom, right
    modified_changed = Signal(bool)

    def __init__(self, dataset: Dataset | None = None, parent: QObject | None = None):
        super().__init__(parent)
        self.ds = dataset or Dataset()
        self.undo_stack = QUndoStack(self)
        self.undo_stack.setUndoLimit(UNDO_LIMIT)
        self.undo_stack.cleanChanged.connect(self._clean_changed)
        self._dirty_when_clean = self.ds.modified

    # ------------------------------------------------------------------ state
    def set_dataset(self, dataset: Dataset) -> None:
        self.ds = dataset
        self.undo_stack.clear()
        self._dirty_when_clean = dataset.modified  # e.g. imported CSV: unsaved but no edits
        self.structure_changed.emit()
        self.modified_changed.emit(dataset.modified)

    def set_modified(self, modified: bool) -> None:
        if self.ds.modified != modified:
            self.ds.modified = modified
        self.modified_changed.emit(modified)

    def mark_saved(self) -> None:
        self._dirty_when_clean = False
        self.undo_stack.setClean()
        self.set_modified(False)

    def _clean_changed(self, clean: bool) -> None:
        self.set_modified(not clean or self._dirty_when_clean)

    def _after_edit(self, structural: bool, block=None) -> None:
        if structural:
            self.structure_changed.emit()
        elif block:
            self.values_changed.emit(*block)
        self.ds.modified = True
        self.modified_changed.emit(True)

    def _recorded(self, text: str, change: Callable[[], object], block=None):
        """Apply ``change`` as one undoable step; roll back on failure."""
        before = _Snapshot(self.ds)
        try:
            result = change()
        except BaseException:
            before.restore(self.ds)
            self._after_edit(True)
            raise
        self.undo_stack.push(_SnapshotCommand(self, text, before, _Snapshot(self.ds)))
        self._after_edit(block is None, block)
        return result

    # ------------------------------------------------------------------ edits
    def set_value(self, row: int, col: int, text) -> None:
        grows = row >= self.ds.n_cases or col >= self.ds.n_vars
        if grows:
            self._recorded("Edit Cell", lambda: self.ds.set_value(row, col, text))
            return
        old = self.ds.get_value(row, col)
        self.ds.set_value(row, col, text)
        new = self.ds.get_value(row, col)
        if not (pd.isna(old) and pd.isna(new)) and not (old == new):
            self.undo_stack.push(_CellCommand(self, row, col, old, new))
        self._after_edit(False, (row, col, row, col))

    def paste_block(self, top: int, left: int, matrix: list[list[str]]) -> list[str]:
        """Write a 2-D block of text; returns messages for cells that could not be parsed."""
        n_rows = len(matrix)
        n_cols = max((len(r) for r in matrix), default=0)
        if not n_rows or not n_cols:
            return []
        grows = top + n_rows > self.ds.n_cases or left + n_cols > self.ds.n_vars
        errors: list[str] = []

        def paste():
            self.ds.ensure_size(top + n_rows, left + n_cols)
            for i, row in enumerate(matrix):
                for j, text in enumerate(row):
                    try:
                        self.ds.set_value(top + i, left + j, text)
                    except ValueError as exc:
                        errors.append(str(exc))

        block = None if grows else (top, left, top + n_rows - 1, left + n_cols - 1)
        self._recorded("Paste", paste, block)
        return errors

    def clear_cells(self, cells: list[tuple[int, int]]) -> None:
        cells = [(r, c) for r, c in cells if r < self.ds.n_cases and c < self.ds.n_vars]
        if not cells:
            return
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]

        def clear():
            for r, c in cells:
                self.ds.set_value(r, c, None)

        self._recorded("Clear", clear, (min(rows), min(cols), max(rows), max(cols)))

    def set_variable_attr(self, col: int, attr: str, value) -> None:
        def change():
            self.ds.ensure_size(0, col + 1)
            self.ds.set_variable_attr(col, attr, value)

        # Decimals, labels, alignment… all change how Data View renders, so refresh fully.
        title = {"value_labels": "Value Labels", "missing": "Missing Values"}.get(
            attr, attr.capitalize())
        self._recorded(f"Change {title}", change)

    def mutate(self, fn: Callable[[Dataset], object], text: str = "Change Data"):
        """Run a structural change (insert/delete/sort/compute) as one undoable step."""
        return self._recorded(text, lambda: fn(self.ds))

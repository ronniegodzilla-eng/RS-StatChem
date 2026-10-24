"""Spreadsheet-style views: Enter moves down, Delete clears, copy/paste with Excel (TSV)."""
from __future__ import annotations

from PySide6.QtCore import QModelIndex, Qt, Signal
from PySide6.QtGui import QGuiApplication, QKeySequence
from PySide6.QtWidgets import (QAbstractItemDelegate, QAbstractItemView, QComboBox, QHeaderView,
                               QSpinBox, QStyledItemDelegate, QTableView)


class SheetView(QTableView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked
                             | QAbstractItemView.EditTrigger.EditKeyPressed
                             | QAbstractItemView.EditTrigger.AnyKeyPressed)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.verticalHeader().setDefaultSectionSize(22)
        self.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.verticalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        self.horizontalHeader().setHighlightSections(True)
        self.verticalHeader().setHighlightSections(True)
        self._saved_index: tuple[int, int] | None = None

    def setModel(self, model) -> None:
        super().setModel(model)
        model.modelAboutToBeReset.connect(self._remember_current)
        model.modelReset.connect(self._restore_current)

    def _remember_current(self) -> None:
        idx = self.currentIndex()
        self._saved_index = (idx.row(), idx.column()) if idx.isValid() else None

    def _restore_current(self) -> None:
        if self._saved_index:
            idx = self.model().index(*self._saved_index)
            if idx.isValid():
                self.setCurrentIndex(idx)

    def move_current(self, d_row: int, d_col: int = 0) -> None:
        idx = self.currentIndex()
        if not idx.isValid():
            return
        target = self.model().index(
            min(max(idx.row() + d_row, 0), self.model().rowCount() - 1),
            min(max(idx.column() + d_col, 0), self.model().columnCount() - 1))
        self.setCurrentIndex(target)

    def closeEditor(self, editor, hint) -> None:
        if hint == QAbstractItemDelegate.EndEditHint.SubmitModelCache:
            super().closeEditor(editor, QAbstractItemDelegate.EndEditHint.NoHint)
            self.move_current(1)
            return
        super().closeEditor(editor, hint)

    def keyPressEvent(self, event) -> None:
        if self.state() != QAbstractItemView.State.EditingState:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.move_current(-1 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier
                                  else 1)
                return
        super().keyPressEvent(event)

    def selected_block(self) -> tuple[int, int, int, int] | None:
        indexes = self.selectionModel().selectedIndexes()
        if not indexes:
            idx = self.currentIndex()
            return (idx.row(), idx.column(), idx.row(), idx.column()) if idx.isValid() else None
        rows = [i.row() for i in indexes]
        cols = [i.column() for i in indexes]
        return min(rows), min(cols), max(rows), max(cols)


class DataGridView(SheetView):
    """Data View. Emits ``header_double_clicked(col)`` to jump to Variable View like SPSS."""

    header_double_clicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.horizontalHeader().setDefaultSectionSize(92)
        self.horizontalHeader().sectionDoubleClicked.connect(self.header_double_clicked)

    def keyPressEvent(self, event) -> None:
        if self.state() != QAbstractItemView.State.EditingState:
            if event.matches(QKeySequence.StandardKey.Copy):
                self.copy_selection()
                return
            if event.matches(QKeySequence.StandardKey.Paste):
                self.paste()
                return
            if event.matches(QKeySequence.StandardKey.Cut):
                self.copy_selection()
                self.clear_selection_cells()
                return
            if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
                self.clear_selection_cells()
                return
        super().keyPressEvent(event)

    def copy_selection(self) -> None:
        block = self.selected_block()
        if not block:
            return
        top, left, bottom, right = block
        model = self.model()
        ds = model.ds
        lines = []
        for r in range(top, min(bottom, ds.n_cases - 1) + 1):
            cells = []
            for c in range(left, min(right, ds.n_vars - 1) + 1):
                cells.append(model.data(model.index(r, c), Qt.ItemDataRole.EditRole) or "")
            lines.append("\t".join(cells))
        QGuiApplication.clipboard().setText("\n".join(lines))

    def paste(self) -> list[str]:
        text = QGuiApplication.clipboard().text()
        idx = self.currentIndex()
        if not text or not idx.isValid():
            return []
        lines = text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n").split("\n")
        matrix = [line.split("\t") for line in lines]
        errors = self.model().doc.paste_block(idx.row(), idx.column(), matrix)
        if errors:
            self.model().error.emit(f"{len(errors)} cell(s) could not be pasted: {errors[0]}")
        return errors

    def clear_selection_cells(self) -> None:
        indexes = self.selectionModel().selectedIndexes() or [self.currentIndex()]
        self.model().doc.clear_cells([(i.row(), i.column()) for i in indexes if i.isValid()])

    def selected_columns(self) -> list[int]:
        return sorted({i.column() for i in self.selectionModel().selectedColumns()})

    def selected_rows(self) -> list[int]:
        return sorted({i.row() for i in self.selectionModel().selectedRows()})


# ------------------------------------------------------------------ delegates
class ChoiceDelegate(QStyledItemDelegate):
    def __init__(self, choices: tuple[str, ...], parent=None):
        super().__init__(parent)
        self.choices = choices

    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        for value in self.choices:
            combo.addItem(value.capitalize(), value)
        return combo

    def setEditorData(self, editor: QComboBox, index: QModelIndex) -> None:
        value = index.data(Qt.ItemDataRole.EditRole)
        i = editor.findData(value)
        editor.setCurrentIndex(max(i, 0))
        editor.showPopup()

    def setModelData(self, editor: QComboBox, model, index: QModelIndex) -> None:
        model.setData(index, editor.currentData(), Qt.ItemDataRole.EditRole)


class SpinDelegate(QStyledItemDelegate):
    def __init__(self, minimum: int, maximum: int, parent=None):
        super().__init__(parent)
        self.minimum, self.maximum = minimum, maximum

    def createEditor(self, parent, option, index):
        spin = QSpinBox(parent)
        spin.setRange(self.minimum, self.maximum)
        return spin

    def setEditorData(self, editor: QSpinBox, index: QModelIndex) -> None:
        editor.setValue(int(index.data(Qt.ItemDataRole.EditRole) or 0))

    def setModelData(self, editor: QSpinBox, model, index: QModelIndex) -> None:
        editor.interpretText()
        model.setData(index, editor.value(), Qt.ItemDataRole.EditRole)

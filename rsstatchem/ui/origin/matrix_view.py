"""Matrix book window (Origin's MBook): matrix sheets in tabs, shown as a data grid or as an
image (View > Image Mode).

The grid's headers number the columns and rows (1, 2, 3…) or, with View > Show X/Y, show
the X and Y coordinates the columns and rows map to. Every change goes through
:meth:`MatrixWindow.change` (one undo step, then the main window refreshes linked graphs).
"""
from __future__ import annotations

import numpy as np
from matplotlib import colormaps
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QPoint, Qt, Signal
from PySide6.QtGui import QImage, QPixmap, QUndoCommand, QUndoStack
from PySide6.QtWidgets import (QAbstractItemView, QLabel, QSizePolicy, QStackedWidget,
                               QTableView, QTabWidget, QVBoxLayout, QWidget)

from rsstatchem.ui.origin.model import MatrixBook, MatrixSheet, format_cell, parse_cell

IMAGE_CMAP = "viridis"


def matrix_image(data: np.ndarray, cmap: str = IMAGE_CMAP) -> QImage:
    """The matrix as a colour-mapped image (row 1 at the top, empty cells white)."""
    data = np.asarray(data, float)
    finite = np.isfinite(data)
    rgba = np.full(data.shape + (4,), 255, dtype=np.uint8)
    if finite.any():
        lo, hi = float(np.nanmin(data)), float(np.nanmax(data))
        norm = (data - lo) / (hi - lo) if hi > lo else np.zeros_like(data)
        mapped = (colormaps[cmap](np.nan_to_num(norm, nan=0.0)) * 255).astype(np.uint8)
        rgba[finite] = mapped[finite]
    rgba = np.ascontiguousarray(rgba)
    h, w = data.shape
    return QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()


class _Snapshot(QUndoCommand):
    def __init__(self, window: "MatrixWindow", text: str, before, after):
        super().__init__(text)
        self.window, self.before, self.after = window, before, after
        self._first = True

    def redo(self) -> None:
        if self._first:
            self._first = False
            return
        self.window.restore(self.after)

    def undo(self) -> None:
        self.window.restore(self.before)


class MatrixModel(QAbstractTableModel):
    def __init__(self, window: "MatrixWindow", ms: MatrixSheet):
        super().__init__()
        self.window = window
        self.ms = ms

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else self.ms.n_rows

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else self.ms.n_cols

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        v = self.ms.data[index.row(), index.column()]
        if role == Qt.ItemDataRole.DisplayRole:   # 6 significant digits, like Origin's grid
            return "" if not np.isfinite(v) else format_cell(float(f"{v:.6g}"))
        if role == Qt.ItemDataRole.EditRole:
            return "" if not np.isfinite(v) else format_cell(float(v))
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if role == Qt.ItemDataRole.DisplayRole:
            if self.window.show_xy:
                vals = self.ms.xs() if orientation == Qt.Orientation.Horizontal else \
                    self.ms.ys()
                return format_cell(float(f"{vals[section]:.6g}")) if section < len(vals) \
                    else ""
            return str(section + 1)
        if role == Qt.ItemDataRole.ToolTipRole:
            if orientation == Qt.Orientation.Horizontal:
                return f"Column {section + 1}: X = {self.ms.xs()[section]:.6g}"
            return f"Row {section + 1}: Y = {self.ms.ys()[section]:.6g}"
        return None

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | \
            Qt.ItemFlag.ItemIsEditable

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole) -> bool:  # noqa: N802
        if role != Qt.ItemDataRole.EditRole or not index.isValid():
            return False
        return self.window.edit_cell(index.row(), index.column(), value)


class MatrixImageView(QLabel):
    """Image mode: the matrix drawn as pixels, stretched over the window."""

    def __init__(self, ms: MatrixSheet):
        super().__init__()
        self.ms = ms
        self.setObjectName("matrixImage")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setMinimumSize(60, 40)
        self._image = matrix_image(ms.data)

    def refresh(self) -> None:
        self._image = matrix_image(self.ms.data)
        self._rescale()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self.width() > 2 and self.height() > 2:
            self.setPixmap(QPixmap.fromImage(self._image).scaled(
                self.width() - 2, self.height() - 2, Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.FastTransformation))

    def image(self) -> QImage:
        return self._image


class MatrixSheetView(QStackedWidget):
    """One matrix sheet: the grid (page 0) and the image (page 1)."""

    def __init__(self, window: "MatrixWindow", ms: MatrixSheet):
        super().__init__()
        self.window = window
        self.ms = ms
        self.grid = QTableView()
        self.grid.setObjectName("matrixSheet")
        self.model_ = MatrixModel(window, ms)
        self.grid.setModel(self.model_)
        self.grid.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.grid.horizontalHeader().setDefaultSectionSize(84)
        self.grid.verticalHeader().setDefaultSectionSize(21)
        self.grid.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.grid.customContextMenuRequested.connect(
            lambda pos: window.context_menu(self.grid.viewport().mapToGlobal(pos)))
        self.image = MatrixImageView(ms)
        self.image.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.image.customContextMenuRequested.connect(
            lambda pos: window.context_menu(self.image.mapToGlobal(pos)))
        self.addWidget(self.grid)
        self.addWidget(self.image)
        self.setCurrentIndex(1 if window.image_mode else 0)

    def refresh(self) -> None:
        m = self.model_
        shape = (m.rowCount(), m.columnCount())
        if shape == getattr(self, "_shape", None) and shape[0] and shape[1]:
            m.dataChanged.emit(m.index(0, 0), m.index(shape[0] - 1, shape[1] - 1))
            m.headerDataChanged.emit(Qt.Orientation.Horizontal, 0, shape[1] - 1)
            m.headerDataChanged.emit(Qt.Orientation.Vertical, 0, shape[0] - 1)
        else:
            m.beginResetModel()
            m.endResetModel()
        self._shape = shape
        self.image.refresh()

    def selected_block(self) -> tuple[int, int, int, int] | None:
        idx = self.grid.selectionModel().selectedIndexes()
        if not idx:
            return None
        rows = [i.row() for i in idx]
        cols = [i.column() for i in idx]
        return min(rows), min(cols), max(rows), max(cols)


class MatrixWindow(QWidget):
    """MDI child for a matrix book. ``main`` is the Origin main window (None in tests)."""

    changed = Signal(object)

    def __init__(self, book: MatrixBook, main=None):
        super().__init__()
        self.book = book
        self.main = main
        self.image_mode = False
        self.show_xy = False
        self.undo_stack = QUndoStack(self)
        self.setObjectName("originMatrix")
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.South)
        self.tabs.setDocumentMode(True)
        self.tabs.setObjectName("sheetTabs")
        self.tabs.currentChanged.connect(self._tab_changed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)
        self.reload()

    # ------------------------------------------------------------------ views
    def reload(self) -> None:
        self.tabs.blockSignals(True)
        current = self.book.active
        while self.tabs.count():
            w = self.tabs.widget(0)
            self.tabs.removeTab(0)
            w.deleteLater()
        for ms in self.book.sheets:
            self.tabs.addTab(MatrixSheetView(self, ms), ms.name)
        self.tabs.setCurrentIndex(max(0, min(current, self.tabs.count() - 1)))
        self.tabs.blockSignals(False)
        self.book.active = self.tabs.currentIndex()
        self._notify()

    def refresh(self) -> None:
        for i in range(self.tabs.count()):
            w = self.tabs.widget(i)
            if w.ms is not self.book.sheets[i] or self.tabs.tabText(i) != w.ms.name:
                self.reload()
                return
            w.refresh()

    def _tab_changed(self, index: int) -> None:
        self.book.active = index
        self._notify()

    def _notify(self) -> None:
        if self.main is not None:
            self.main.matrix_view_changed(self)

    def sheet(self) -> MatrixSheet | None:
        return self.book.active_sheet()

    def view(self) -> MatrixSheetView | None:
        return self.tabs.currentWidget()

    def set_image_mode(self, on: bool) -> None:
        self.image_mode = on
        for i in range(self.tabs.count()):
            self.tabs.widget(i).setCurrentIndex(1 if on else 0)

    def set_show_xy(self, on: bool) -> None:
        self.show_xy = on
        for i in range(self.tabs.count()):
            m = self.tabs.widget(i).model_
            m.headerDataChanged.emit(Qt.Orientation.Horizontal, 0, max(m.columnCount() - 1, 0))
            m.headerDataChanged.emit(Qt.Orientation.Vertical, 0, max(m.rowCount() - 1, 0))

    # ------------------------------------------------------------------ changes
    def snapshot(self) -> dict:
        return {"sheets": [s.copy() for s in self.book.sheets], "active": self.book.active,
                "long_name": self.book.long_name}

    def restore(self, snap: dict) -> None:
        self.book.sheets = [s.copy() for s in snap["sheets"]]
        self.book.active = snap["active"]
        self.book.long_name = snap["long_name"]
        self.reload()
        self.changed.emit(self.book)

    def change(self, text: str, fn):
        """Apply ``fn()`` to the matrix book as one undo step."""
        before = self.snapshot()
        result = fn()
        self.undo_stack.push(_Snapshot(self, text, before, self.snapshot()))
        self.refresh()
        self.changed.emit(self.book)
        return result

    def edit_cell(self, row: int, col: int, value) -> bool:
        ms = self.sheet()
        new = parse_cell("" if value is None else str(value))
        if new is not None and not isinstance(new, float):
            return False   # matrices hold numbers only
        new = np.nan if new is None else float(new)
        old = ms.data[row, col]
        if (np.isnan(old) and np.isnan(new)) or old == new:
            return False
        self.change("Edit Cell", lambda: ms.data.__setitem__((row, col), new))
        return True

    def paste_text(self, text: str, top: int = 0, left: int = 0) -> None:
        """Paste tab-separated numbers at (top, left), growing the matrix if needed."""
        rows = [line.split("\t") for line in text.replace("\r\n", "\n").rstrip("\n").split("\n")]
        ms = self.sheet()

        def fn():
            need_r = top + len(rows)
            need_c = left + max(len(r) for r in rows)
            if need_r > ms.n_rows or need_c > ms.n_cols:
                ms.set_dimensions(max(need_r, ms.n_rows), max(need_c, ms.n_cols))
            for i, r in enumerate(rows):
                for j, v in enumerate(r):
                    p = parse_cell(v)
                    ms.data[top + i, left + j] = p if isinstance(p, float) else np.nan
        self.change("Paste", fn)

    def copy_text(self) -> str:
        v = self.view()
        ms = self.sheet()
        block = v.selected_block() if v is not None else None
        top, left, bottom, right = block or (0, 0, ms.n_rows - 1, ms.n_cols - 1)
        return "\n".join("\t".join("" if not np.isfinite(ms.data[r, c]) else
                                   format_cell(float(ms.data[r, c]))
                                   for c in range(left, right + 1))
                         for r in range(top, bottom + 1))

    def context_menu(self, global_pos: QPoint) -> None:
        if self.main is not None:
            self.main.matrix_menu(self, global_pos)

    def add_sheet(self) -> MatrixSheet:
        ref = self.sheet()
        ms = MatrixSheet.new(self.book.unique_sheet_name(), ref.n_rows if ref else 32,
                             ref.n_cols if ref else 32)

        def fn():
            self.book.sheets.append(ms)
            self.book.active = len(self.book.sheets) - 1
        self.change("Add Sheet", fn)
        self.reload()
        return ms

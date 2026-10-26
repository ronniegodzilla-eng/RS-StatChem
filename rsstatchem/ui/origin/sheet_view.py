"""Workbook window: Origin-style worksheet grid and report sheets in sheet tabs.

The grid shows the column header ``A(X)``, ``B(Y)``…, Origin's label rows (Long Name, Units,
Comments, F(x)=, and the optional Sparklines row with a tiny line plot of each column) above
the numbered data rows, and the sheet tabs at the bottom. Every change goes through
:meth:`BookWindow.change`, which records an undo step (a snapshot of the workbook's sheets),
recalculates F(x)= formulas and tells the main window so linked graphs refresh.

Report sheets are shown by :class:`ReportView` (see :mod:`rsstatchem.ui.origin.report_view`):
collapsible sections with copyable tables and live embedded graphs, and the lock button for
recalculation. Output columns of Statistics on Rows and Frequency Counts show the same lock
in their column header (Origin's locked output columns); right-click it to recalculate.
"""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QPoint, QPointF, Qt, Signal
from PySide6.QtGui import (QBrush, QColor, QKeySequence, QPainter, QPen, QPolygonF,
                           QUndoCommand, QUndoStack)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QHeaderView, QMenu,
                               QStyledItemDelegate, QTabBar, QTableView, QTabWidget,
                               QVBoxLayout, QWidget)

from rsstatchem.ui.origin import formula as fx
from rsstatchem.ui.origin.icons import icon
from rsstatchem.ui.origin.model import (DESIGNATIONS, LABEL_ROWS, RECALC_MODES, ReportSheet,
                                        Workbook, Worksheet, format_cell, parse_cell,
                                        to_float)
from rsstatchem.ui.origin.report_view import (ReportView, lock_icon_name,  # noqa: F401
                                              report_html)

OP_TITLES = {"rows": "Statistics on Rows", "freq": "Frequency Counts"}

SPARK_ROW = len(LABEL_ROWS)            # the Sparklines row, below F(x)=
HEADER_ROWS = len(LABEL_ROWS) + 1
SPARK_HEIGHT = 34
LABEL_BG = QColor("#f2f2f2")
FORMULA_BG = QColor("#eef3fb")
EXTRA_ROWS = 10   # rows added when typing into the last row


def label_attr(row: int) -> str | None:
    """Column attribute shown in header row ``row`` (None for the Sparklines row)."""
    return LABEL_ROWS[row][0] if row < len(LABEL_ROWS) else None


# ====================================================================== undo
def snapshot(book: Workbook) -> dict:
    """Copy of a workbook's state (reports too: recalculation changes them in place)."""
    return {"sheets": [s.copy() for s in book.sheets], "active": book.active,
            "long_name": book.long_name}


def restore(book: Workbook, snap: dict) -> None:
    book.sheets = [s.copy() for s in snap["sheets"]]
    book.active = snap["active"]
    book.long_name = snap["long_name"]


class SnapshotCommand(QUndoCommand):
    def __init__(self, window: "BookWindow", text: str, before: dict, after: dict):
        super().__init__(text)
        self.window, self.before, self.after = window, before, after
        self._first = True

    def redo(self) -> None:
        if self._first:   # the change has already been applied
            self._first = False
            return
        restore(self.window.book, self.after)
        self.window.reload()

    def undo(self) -> None:
        restore(self.window.book, self.before)
        self.window.reload()


# ====================================================================== grid model
class SheetModel(QAbstractTableModel):
    def __init__(self, window: "BookWindow", ws: Worksheet):
        super().__init__()
        self.window = window
        self.ws = ws

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else HEADER_ROWS + self.ws.n_rows

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self.ws.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        r, c = index.row(), index.column()
        col = self.ws.columns[c]
        if r == SPARK_ROW:
            if role == Qt.ItemDataRole.BackgroundRole:
                return QBrush(QColor("#ffffff"))
            if role == Qt.ItemDataRole.ToolTipRole:
                return f"Sparkline of {col.name}" + (f" ({col.long_name})" if col.long_name
                                                     else "")
            return None
        if r < HEADER_ROWS:
            attr = LABEL_ROWS[r][0]
            if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
                return getattr(col, attr)
            if role == Qt.ItemDataRole.BackgroundRole:
                return QBrush(FORMULA_BG if attr == "formula" else LABEL_BG)
            if role == Qt.ItemDataRole.ForegroundRole and attr == "formula":
                return QBrush(QColor("#1f4fa0"))
            if role == Qt.ItemDataRole.TextAlignmentRole:
                return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            if role == Qt.ItemDataRole.ToolTipRole and attr == "formula":
                return "Column formula (Origin syntax), e.g. col(A)*2 or sin(i/10)."
            return None
        value = self.ws.cell(r - HEADER_ROWS, c)
        if role == Qt.ItemDataRole.DisplayRole:
            return format_cell(value)
        if role == Qt.ItemDataRole.EditRole:
            return "" if value is None else (format_cell(value) if isinstance(value, float)
                                             else str(value))
        if role == Qt.ItemDataRole.TextAlignmentRole:
            left = isinstance(value, str)
            return int((Qt.AlignmentFlag.AlignLeft if left else Qt.AlignmentFlag.AlignRight)
                       | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.ForegroundRole and col.designation == "N":
            return QBrush(QColor("#9a9a9a"))
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return self.ws.label(section) if section < len(self.ws.columns) else ""
            if section == SPARK_ROW:
                return "Sparklines"
            return LABEL_ROWS[section][1] if section < HEADER_ROWS else \
                str(section - HEADER_ROWS + 1)
        if role == Qt.ItemDataRole.TextAlignmentRole and orientation == Qt.Orientation.Vertical:
            return int((Qt.AlignmentFlag.AlignLeft if section < HEADER_ROWS
                        else Qt.AlignmentFlag.AlignCenter) | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.DecorationRole and \
                orientation == Qt.Orientation.Horizontal:
            op = self.ws.op_of(section)
            return icon(lock_icon_name(op)) if op is not None else None
        if role == Qt.ItemDataRole.ToolTipRole and orientation == Qt.Orientation.Horizontal \
                and section < len(self.ws.columns):
            col = self.ws.columns[section]
            tip = f"{col.name}: {DESIGNATIONS[col.designation][0]}" + \
                (f" — {col.long_name}" if col.long_name else "")
            op = self.ws.op_of(section)
            if op is not None:
                tip += (f"\nOutput of {OP_TITLES.get(op.get('kind'), 'an analysis')} — "
                        f"Recalculate: {RECALC_MODES.get(op.get('mode', 'none'), 'None')}" +
                        (" (the source data changed)" if op.get("dirty") else "") +
                        "; right-click for Recalculate / Change Parameters.")
            return tip
        return None

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        if index.row() == SPARK_ROW:
            return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | \
            Qt.ItemFlag.ItemIsEditable

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole) -> bool:  # noqa: N802
        if role != Qt.ItemDataRole.EditRole or not index.isValid():
            return False
        return self.window.edit_cell(index.row(), index.column(), value)

    def refresh(self) -> bool:
        """Repaint after a model change; True when the shape changed (full reset)."""
        shape = (self.rowCount(), self.columnCount())
        if shape == getattr(self, "_shape", None) and shape[0] and shape[1]:
            self.dataChanged.emit(self.index(0, 0), self.index(shape[0] - 1, shape[1] - 1))
            self.headerDataChanged.emit(Qt.Orientation.Horizontal, 0, shape[1] - 1)
            return False
        self._shape = shape
        self.beginResetModel()
        self.endResetModel()
        return True


# ====================================================================== sparklines
def sparkline_points(ws: Worksheet, col: int, max_points: int = 240) \
        -> tuple[np.ndarray, np.ndarray]:
    """(x, y) of a column's sparkline: a Y column against its X column (sorted by X), other
    columns against the row number; at most ``max_points`` points (evenly decimated)."""
    n = ws.data_rows()
    c = ws.columns[col]
    if n == 0 or c.is_text():
        return np.zeros(0), np.zeros(0)
    y = c.numeric(n)
    xi = ws.x_of(col) if c.designation not in ("X",) else None
    x = ws.columns[xi].numeric(n) if xi is not None and not ws.columns[xi].is_text() else \
        np.arange(1, n + 1, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if xi is not None and len(x):
        order = np.argsort(x, kind="stable")
        x, y = x[order], y[order]
    if len(x) > max_points:
        keep = np.linspace(0, len(x) - 1, max_points).astype(int)
        x, y = x[keep], y[keep]
    return x, y


class SparklineDelegate(QStyledItemDelegate):
    """Paints Origin's sparkline (a tiny line plot of the column) in the Sparklines row."""

    def __init__(self, view: "SheetView"):
        super().__init__(view)
        self.view = view

    def paint(self, painter: QPainter, option, index) -> None:
        if index.row() != SPARK_ROW:
            super().paint(painter, option, index)
            return
        painter.save()
        painter.fillRect(option.rect, QColor("#ffffff"))
        x, y = sparkline_points(self.view.ws, index.column())
        r = option.rect.adjusted(5, 4, -5, -4)
        if len(x) >= 2 and r.width() > 4 and r.height() > 4:
            x0, x1 = float(x.min()), float(x.max())
            y0, y1 = float(y.min()), float(y.max())
            sx = (r.width()) / (x1 - x0) if x1 > x0 else 0.0
            sy = (r.height()) / (y1 - y0) if y1 > y0 else 0.0
            pts = QPolygonF([QPointF(r.left() + (xv - x0) * sx if sx else r.center().x(),
                                     r.bottom() - (yv - y0) * sy if sy else r.center().y())
                             for xv, yv in zip(x, y)])
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setPen(QPen(QColor("#1f4fa0"), 1.1))
            painter.drawPolyline(pts)
        painter.setPen(QPen(QColor("#d4d4d4"), 1))
        painter.drawLine(option.rect.bottomLeft(), option.rect.bottomRight())
        painter.restore()


# ====================================================================== grid view
class SheetView(QTableView):
    def __init__(self, window: "BookWindow", ws: Worksheet):
        super().__init__()
        self.window = window
        self.model_ = SheetModel(window, ws)
        self.model_._shape = (self.model_.rowCount(), self.model_.columnCount())
        self.setModel(self.model_)
        self.setObjectName("originSheet")
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked
                             | QAbstractItemView.EditTrigger.EditKeyPressed
                             | QAbstractItemView.EditTrigger.AnyKeyPressed)
        self.setWordWrap(False)
        self.setCornerButtonEnabled(True)
        hh = self.horizontalHeader()
        hh.setDefaultSectionSize(84)
        hh.setMinimumSectionSize(40)
        hh.setHighlightSections(True)
        hh.setSectionsClickable(True)
        hh.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        hh.customContextMenuRequested.connect(self._header_menu)
        hh.sectionDoubleClicked.connect(lambda c: self.window.column_properties(c))
        hh.sectionResized.connect(self._resized)
        vh = self.verticalHeader()
        vh.setDefaultSectionSize(21)
        vh.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        vh.setMinimumWidth(78)
        vh.setHighlightSections(False)
        vh.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        vh.customContextMenuRequested.connect(self._row_menu)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._cell_menu)
        self.setItemDelegateForRow(SPARK_ROW, SparklineDelegate(self))
        self.apply_widths()
        self.apply_sparklines()

    @property
    def ws(self) -> Worksheet:
        return self.model_.ws

    def apply_sparklines(self) -> None:
        """Show / hide the Sparklines header row as the sheet says."""
        self.setRowHidden(SPARK_ROW, not self.ws.sparklines)
        if self.ws.sparklines:
            self.setRowHeight(SPARK_ROW, SPARK_HEIGHT)

    def refresh_view(self) -> None:
        """Refresh the grid, keeping the current cell after a reset (rows/columns added)."""
        cur = self.currentIndex()
        row, col = (cur.row(), cur.column()) if cur.isValid() else (-1, -1)
        if self.model_.refresh() and row >= 0:
            m = self.model_
            if m.rowCount() and m.columnCount():
                self.setCurrentIndex(m.index(min(row, m.rowCount() - 1),
                                             min(col, m.columnCount() - 1)))
        self.apply_widths()
        self.apply_sparklines()

    def closeEditor(self, editor, hint) -> None:  # noqa: N802
        super().closeEditor(editor, hint)
        if hint == self.itemDelegate().EndEditHint.SubmitModelCache:   # Enter: next row
            cur = self.currentIndex()
            if cur.isValid() and cur.row() + 1 < self.model_.rowCount():
                self.setCurrentIndex(self.model_.index(cur.row() + 1, cur.column()))

    def apply_widths(self) -> None:
        for i, c in enumerate(self.ws.columns):
            if c.width:
                self.setColumnWidth(i, c.width)

    def _resized(self, index: int, _old: int, new: int) -> None:
        if index < len(self.ws.columns):
            self.ws.columns[index].width = new

    # ------------------------------------------------------------------ selection
    def selected_columns(self) -> list[int]:
        """Fully selected columns, or the columns of the selected cells."""
        sm = self.selectionModel()
        if sm is None:
            return []
        full = [i.column() for i in sm.selectedColumns()]
        if full:
            return sorted(set(full))
        cols = sorted({i.column() for i in sm.selectedIndexes()})
        if not cols and self.currentIndex().isValid():
            cols = [self.currentIndex().column()]
        return cols

    def selected_rows(self) -> list[int]:
        """Selected data rows (0-based)."""
        sm = self.selectionModel()
        rows = {i.row() - HEADER_ROWS for i in sm.selectedIndexes()} if sm else set()
        return sorted(r for r in rows if r >= 0)

    def selected_row_range(self) -> tuple[int, int] | None:
        """1-based inclusive data-row range of a selected block of cells (e.g. B3:C10), or
        None when whole columns (or a single row / all data rows) are selected — then plots
        and analyses use the whole columns, as in Origin."""
        sm = self.selectionModel()
        if sm is None or sm.selectedColumns():
            return None
        rows = self.selected_rows()
        if len(rows) < 2:
            return None
        lo, hi = rows[0] + 1, rows[-1] + 1
        n = self.ws.data_rows()
        if lo > n:
            return None
        hi = min(hi, max(n, lo))
        if lo == 1 and hi >= n:
            return None
        return lo, hi

    def select_columns(self, cols: list[int]) -> None:
        self.clearSelection()
        for c in cols:
            self.selectColumn(c) if len(cols) == 1 else \
                self.selectionModel().select(self.model_.index(0, c),
                                             self.selectionModel().SelectionFlag.Select
                                             | self.selectionModel().SelectionFlag.Columns)

    def selection_range(self) -> tuple[int, int, int, int] | None:
        """(top, left, bottom, right) of the selected cells in model rows/cols."""
        idx = self.selectionModel().selectedIndexes() if self.selectionModel() else []
        if not idx:
            cur = self.currentIndex()
            if not cur.isValid():
                return None
            return cur.row(), cur.column(), cur.row(), cur.column()
        rows = [i.row() for i in idx]
        cols = [i.column() for i in idx]
        return min(rows), min(cols), max(rows), max(cols)

    # ------------------------------------------------------------------ keys
    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and \
                self.state() != QAbstractItemView.State.EditingState:
            cur = self.currentIndex()
            if cur.isValid():
                self.setCurrentIndex(self.model_.index(min(cur.row() + 1,
                                                           self.model_.rowCount() - 1),
                                                       cur.column()))
                return
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and \
                self.state() != QAbstractItemView.State.EditingState:
            self.window.clear_selection()
            return
        if event.matches(QKeySequence.StandardKey.Copy):
            self.window.copy()
            return
        if event.matches(QKeySequence.StandardKey.Paste):
            self.window.paste()
            return
        super().keyPressEvent(event)

    # ------------------------------------------------------------------ menus
    def _header_menu(self, pos: QPoint) -> None:
        col = self.horizontalHeader().logicalIndexAt(pos)
        if col < 0:
            return
        if col not in self.selected_columns():
            self.selectColumn(col)
        self.window.column_menu(self.horizontalHeader().mapToGlobal(pos))

    def _cell_menu(self, pos: QPoint) -> None:
        self.window.cell_menu(self.viewport().mapToGlobal(pos))

    def _row_menu(self, pos: QPoint) -> None:
        row = self.verticalHeader().logicalIndexAt(pos)
        if row >= HEADER_ROWS and row - HEADER_ROWS not in self.selected_rows():
            self.selectRow(row)
        self.window.row_menu(self.verticalHeader().mapToGlobal(pos))


# ====================================================================== workbook window
class BookWindow(QWidget):
    """MDI child for a workbook. ``main`` is the Origin main window (may be None in tests)."""

    changed = Signal(object)   # emitted with the workbook after every recorded change

    def __init__(self, book: Workbook, main=None):
        super().__init__()
        self.book = book
        self.main = main
        self.undo_stack = QUndoStack(self)
        self.setObjectName("originBook")
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.South)
        self.tabs.setDocumentMode(True)
        self.tabs.setObjectName("sheetTabs")
        self.tabs.currentChanged.connect(self._tab_changed)
        self.tabs.tabBar().setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tabs.tabBar().customContextMenuRequested.connect(self._tab_menu)
        self.tabs.tabBar().tabBarDoubleClicked.connect(lambda i: self.rename_sheet(i))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)
        self.reload()

    # ------------------------------------------------------------------ views
    def reload(self) -> None:
        """Rebuild the sheet tabs from the model (after undo, new sheets, reports…)."""
        self.tabs.blockSignals(True)
        current = self.book.active
        while self.tabs.count():
            w = self.tabs.widget(0)
            self.tabs.removeTab(0)
            w.deleteLater()
        for sheet in self.book.sheets:
            view = SheetView(self, sheet) if isinstance(sheet, Worksheet) else \
                ReportView(sheet, self)
            self.tabs.addTab(view, sheet.name)
        self.tabs.setCurrentIndex(max(0, min(current, self.tabs.count() - 1)))
        self.tabs.blockSignals(False)
        self.book.active = self.tabs.currentIndex()
        self._notify()

    def refresh(self) -> None:
        """Repaint the current grid after an in-place change (same sheets)."""
        for i in range(self.tabs.count()):
            w = self.tabs.widget(i)
            if isinstance(w, SheetView):
                if w.ws is not self.book.sheets[i]:
                    self.reload()
                    return
                w.refresh_view()
            elif isinstance(w, ReportView):
                w.refresh()

    def _tab_changed(self, index: int) -> None:
        self.book.active = index
        self._notify()

    def _notify(self) -> None:
        if self.main is not None:
            self.main.book_view_changed(self)

    def view(self) -> SheetView | ReportView | None:
        return self.tabs.currentWidget()

    def sheet(self):
        return self.book.active_sheet()

    def worksheet(self) -> Worksheet | None:
        s = self.sheet()
        return s if isinstance(s, Worksheet) else None

    def sheet_view(self) -> SheetView | None:
        v = self.view()
        return v if isinstance(v, SheetView) else None

    def selected_columns(self) -> list[int]:
        v = self.sheet_view()
        return v.selected_columns() if v is not None else []

    def selected_row_range(self) -> tuple[int, int] | None:
        v = self.sheet_view()
        return v.selected_row_range() if v is not None else None

    def select_block(self, top: int, left: int, bottom: int, right: int) -> None:
        """Select data cells rows ``top..bottom`` (1-based) × columns ``left..right``."""
        v = self.sheet_view()
        if v is None:
            return
        from PySide6.QtCore import QItemSelection, QItemSelectionModel
        m = v.model_
        sel = QItemSelection(m.index(HEADER_ROWS + top - 1, left),
                             m.index(HEADER_ROWS + bottom - 1, right))
        v.selectionModel().select(sel, QItemSelectionModel.SelectionFlag.ClearAndSelect)
        v.setCurrentIndex(m.index(HEADER_ROWS + top - 1, left))
        v.selectionModel().select(sel, QItemSelectionModel.SelectionFlag.ClearAndSelect)

    def select_columns(self, cols: list[int]) -> None:
        v = self.sheet_view()
        if v is not None:
            v.select_columns(cols)

    def show_sheet(self, sheet) -> None:
        if sheet in self.book.sheets:
            self.tabs.setCurrentIndex(self.book.sheets.index(sheet))

    # ------------------------------------------------------------------ recorded changes
    def change(self, text: str, fn, recalc: bool = True, skip_col: int | None = None):
        """Apply ``fn()`` to the model as one undo step."""
        before = snapshot(self.book)
        result = fn()
        ws = self.worksheet()
        if recalc and ws is not None:
            fx.recalculate(ws, skip=skip_col)
        self.undo_stack.push(SnapshotCommand(self, text, before, snapshot(self.book)))
        if len(self.book.sheets) != self.tabs.count() or any(
                (isinstance(self.tabs.widget(i), SheetView) and
                 self.tabs.widget(i).ws is not s) or
                (isinstance(self.tabs.widget(i), ReportView) and
                 self.tabs.widget(i).report is not s) or self.tabs.tabText(i) != s.name
                for i, s in enumerate(self.book.sheets)):
            self.reload()
        else:
            self.refresh()
        self.changed.emit(self.book)
        return result

    def edit_cell(self, row: int, col: int, value) -> bool:
        ws = self.worksheet()
        if ws is None:
            return False
        text = "" if value is None else str(value)
        if row == SPARK_ROW:
            return False
        if row < HEADER_ROWS:
            attr = LABEL_ROWS[row][0]
            if getattr(ws.columns[col], attr) == text.strip():
                return False
            if attr == "formula":
                return self.set_formula(col, text)
            self.change(f"Edit {LABEL_ROWS[row][1]}",
                        lambda: setattr(ws.columns[col], attr, text.strip()))
            return True
        r = row - HEADER_ROWS
        new = parse_cell(text)
        if format_cell(ws.cell(r, col)) == format_cell(new) and \
                type(ws.cell(r, col)) is type(new):
            return False

        def fn():
            ws.set_cell(r, col, new)
            ws.columns[col].formula = ""   # typing over a formula column breaks the formula
            if r >= ws.n_rows - 1:
                ws.set_rows(ws.n_rows + EXTRA_ROWS)
        self.change("Edit Cell", fn)
        return True

    def set_formula(self, col: int, text: str) -> bool:
        ws = self.worksheet()
        text = text.strip()
        if not text:
            self.change("Clear Formula", lambda: setattr(ws.columns[col], "formula", ""))
            return True
        try:
            values = fx.evaluate_formula(ws, text)
        except (fx.FormulaError, ValueError) as exc:
            self._message(f"F(x)= of column {ws.columns[col].name}: {exc}", error=True)
            return False

        def fn():
            ws.set_column_values(col, values)
            ws.columns[col].formula = text
        self.change("Set Column Values", fn, skip_col=None)
        return True

    def _message(self, text: str, error: bool = False) -> None:
        if self.main is not None:
            self.main.message(text, error=error)

    # ------------------------------------------------------------------ clipboard
    def copy(self, with_headers: bool = False) -> str:
        v = self.view()
        if isinstance(v, ReportView):
            v.copy_all()
            return v.toPlainText()
        if not isinstance(v, SheetView):
            return ""
        rng = v.selection_range()
        if rng is None:
            return ""
        top, left, bottom, right = rng
        ws = v.ws
        full_cols = bool(v.selectionModel().selectedColumns())
        if full_cols:
            top = 0 if with_headers else HEADER_ROWS
            bottom = HEADER_ROWS + max(ws.data_rows(), 1) - 1
        lines = []
        for r in range(top, bottom + 1):
            if r == SPARK_ROW:
                continue
            cells = []
            for c in range(left, right + 1):
                if r < HEADER_ROWS:
                    cells.append(getattr(ws.columns[c], LABEL_ROWS[r][0]))
                else:
                    cells.append(format_cell(ws.cell(r - HEADER_ROWS, c)))
            lines.append("\t".join(cells))
        text = "\n".join(lines)
        QApplication.clipboard().setText(text)
        return text

    def paste(self, text: str | None = None) -> None:
        v = self.sheet_view()
        if v is None:
            return
        text = QApplication.clipboard().text() if text is None else text
        if not text:
            return
        rows = [line.split("\t") for line in text.replace("\r\n", "\n").rstrip("\n").split("\n")]
        cur = v.currentIndex()
        rng = v.selection_range()
        top = rng[0] if rng else (cur.row() if cur.isValid() else HEADER_ROWS)
        left = rng[1] if rng else max(cur.column(), 0)
        if v.selectionModel().selectedColumns():
            top = HEADER_ROWS
        self.paste_block(rows, top, left)

    def paste_block(self, rows: list[list[str]], top: int, left: int) -> None:
        """Paste a block of cells; a text first line pasted at row 1 above numbers becomes
        the Long Names (pasting a table with headers from Excel)."""
        ws = self.worksheet()
        if ws is None or not rows:
            return
        header = None
        if top == HEADER_ROWS and len(rows) > 1 and \
                all(c.strip() == "" or math.isnan(to_float(c)) for c in rows[0]) and \
                any(not math.isnan(to_float(c)) for r in rows[1:] for c in r):
            header, rows = rows[0], rows[1:]

        def fn():
            width = max(len(r) for r in rows)
            while len(ws.columns) < left + width:
                ws.add_column("Y")
            if header is not None:
                for j, name in enumerate(header):
                    if name.strip():
                        ws.columns[left + j].long_name = name.strip()
            for i, r in enumerate(rows):
                for j, value in enumerate(r):
                    row = top + i
                    if row == SPARK_ROW:
                        continue
                    if row < HEADER_ROWS:
                        setattr(ws.columns[left + j], LABEL_ROWS[row][0], value.strip())
                    else:
                        ws.set_cell(row - HEADER_ROWS, left + j, value)
                        ws.columns[left + j].formula = ""
            last = top - HEADER_ROWS + len(rows)
            if last >= ws.n_rows:
                ws.set_rows(last + EXTRA_ROWS)
        self.change("Paste", fn)

    def clear_selection(self) -> None:
        v = self.sheet_view()
        if v is None:
            return
        ws = v.ws
        idx = v.selectionModel().selectedIndexes()
        if not idx and v.currentIndex().isValid():
            idx = [v.currentIndex()]
        if not idx:
            return

        def fn():
            for i in idx:
                r, c = i.row(), i.column()
                if r == SPARK_ROW:
                    continue
                if r < HEADER_ROWS:
                    setattr(ws.columns[c], LABEL_ROWS[r][0], "")
                else:
                    ws.set_cell(r - HEADER_ROWS, c, None)
        self.change("Clear", fn)

    # ------------------------------------------------------------------ menus (delegated)
    def column_menu(self, global_pos) -> None:
        if self.main is not None:
            self.main.column_menu(self, global_pos)

    def cell_menu(self, global_pos) -> None:
        if self.main is not None:
            self.main.cell_menu(self, global_pos)

    def row_menu(self, global_pos) -> None:
        if self.main is None:
            return
        menu = QMenu(self)
        menu.addAction("Insert Rows", self.insert_rows)
        menu.addAction("Delete Rows", self.delete_rows)
        menu.addAction("Clear", self.clear_selection)
        menu.exec(global_pos)

    @staticmethod
    def lock_icon(target) -> str:
        """Icon name of the lock of a recalculable output (report or column operation)."""
        return lock_icon_name(target.op if target is not None else None)

    def column_properties(self, col: int) -> None:
        if self.main is not None:
            self.main.column_properties(col)

    def insert_rows(self) -> None:
        v = self.sheet_view()
        rows = v.selected_rows() if v else []
        if v is None or not rows:
            return
        self.change("Insert Rows", lambda: v.ws.insert_rows(rows[0], len(rows)))

    def delete_rows(self) -> None:
        v = self.sheet_view()
        rows = v.selected_rows() if v else []
        if v is None or not rows:
            return
        self.change("Delete Rows", lambda: v.ws.delete_rows(rows))

    # ------------------------------------------------------------------ sheets
    def _tab_menu(self, pos: QPoint) -> None:
        bar: QTabBar = self.tabs.tabBar()
        index = bar.tabAt(pos)
        menu = QMenu(self)
        menu.addAction("Insert", lambda: self.add_sheet(at=max(index, 0)))
        menu.addAction("Add", self.add_sheet)
        if index >= 0:
            menu.addAction("Rename", lambda: self.rename_sheet(index))
            menu.addAction("Duplicate", lambda: self.duplicate_sheet(index))
            menu.addSeparator()
            act = menu.addAction("Delete", lambda: self.delete_sheet(index))
            act.setEnabled(len(self.book.sheets) > 1)
        menu.exec(bar.mapToGlobal(pos))

    def add_sheet(self, at: int | None = None) -> Worksheet:
        rows = self.main.default_rows() if self.main is not None else 32
        ws = Worksheet.new(self.book.unique_sheet_name(), 2, rows)

        def fn():
            self.book.add_sheet(ws, at)
            self.book.active = self.book.sheets.index(ws)
        self.change("Add Sheet", fn, recalc=False)
        return ws

    def duplicate_sheet(self, index: int) -> None:
        src = self.book.sheets[index]
        copy_ = src.copy()
        copy_.name = self.book.unique_sheet_name(src.name.rstrip("0123456789") or "Sheet")
        self.change("Duplicate Sheet", lambda: self.book.sheets.insert(index + 1, copy_),
                    recalc=False)

    def delete_sheet(self, index: int) -> None:
        if len(self.book.sheets) <= 1:
            return

        def fn():
            del self.book.sheets[index]
            self.book.active = max(0, min(self.book.active, len(self.book.sheets) - 1))
        self.change("Delete Sheet", fn, recalc=False)

    def rename_sheet(self, index: int, name: str | None = None) -> bool:
        if index < 0 or index >= len(self.book.sheets):
            return False
        if name is None:
            from PySide6.QtWidgets import QInputDialog
            name, ok = QInputDialog.getText(self, "Rename Sheet", "Sheet name:",
                                            text=self.book.sheets[index].name)
            if not ok:
                return False
        name = name.strip()
        if not name or any(s.name.lower() == name.lower() for i, s in
                           enumerate(self.book.sheets) if i != index):
            self._message(f"Cannot rename the sheet to “{name}”.", error=True)
            return False
        old = self.book.sheets[index].name
        self.change("Rename Sheet", lambda: setattr(self.book.sheets[index], "name", name),
                    recalc=False)
        if self.main is not None:
            self.main.sheet_renamed(self.book, old, name)
        return True

    def set_sparklines(self, on: bool) -> None:
        """View > Sparklines for the active worksheet (undoable, saved with the project)."""
        ws = self.worksheet()
        if ws is None or ws.sparklines == on:
            return
        self.change("Sparklines", lambda: setattr(ws, "sparklines", on), recalc=False)
        v = self.sheet_view()
        if v is not None:
            v.apply_sparklines()

    def is_untouched(self) -> bool:
        return self.undo_stack.count() == 0 and self.book.is_empty()

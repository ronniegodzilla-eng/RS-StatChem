"""The Results window (and every other table window: Summary, Histogram lists, Coloc 2, ...),
ImageJ's ``TextWindow`` with a ``ResultsTable``.

Menus as ImageJ 1.54: File (Save As... csv/tsv/xls/xlsx, Rename..., Duplicate...),
Edit (Copy, Clear, Select All, Delete Rows, Sort..., Plot..., Summarize, Distribution...,
Set Decimal Places..., Options...), Font (sizes), Results (Clear Results, Summarize,
Distribution..., Set Measurements..., Rename..., Duplicate...). Row numbers, the Label column,
per-column decimals and NaN empty cells follow the table's formatting rules
(``ResultsTable.format_value``). Registered as ``windows.TABLE_WINDOW_FACTORY``.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QAction, QFont, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (QAbstractItemView, QFileDialog, QInputDialog, QMainWindow,
                               QTableView)

from rsstatchem.core.imaging.results import LABEL, ResultsTable


class TableModel(QAbstractTableModel):
    def __init__(self, table: ResultsTable, parent=None):
        super().__init__(parent)
        self.table = table
        self._heads = list(table.headings)

    def set_table(self, table: ResultsTable) -> None:
        self.beginResetModel()
        self.table = table
        self._heads = list(table.headings)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else self.table.size

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._heads)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return self.table.format_value(self._heads[index.column()], index.row())
        if role == Qt.ItemDataRole.TextAlignmentRole:
            h = self._heads[index.column()]
            return (Qt.AlignmentFlag.AlignLeft if h == LABEL else Qt.AlignmentFlag.AlignRight) \
                | Qt.AlignmentFlag.AlignVCenter
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self._heads[section] if section < len(self._heads) else None
        return str(section + self.table.base_row_number)


class ResultsWindow(QMainWindow):
    """``ResultsWindow(bar, table)``; ``set_table(table)`` refreshes it."""

    def __init__(self, bar=None, table: ResultsTable | None = None):
        super().__init__()
        self.bar = bar
        self.table = table if table is not None else ResultsTable.results()
        self.setWindowTitle(self.table.title)
        self.view = QTableView(self)
        self.model = TableModel(self.table, self.view)
        self.view.setModel(self.model)
        self.view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.view.setAlternatingRowColors(False)
        self.view.verticalHeader().setDefaultSectionSize(18)
        self.view.horizontalHeader().setStretchLastSection(False)
        self._font_size = 11
        self.view.setFont(QFont("Menlo", self._font_size))
        self.setCentralWidget(self.view)
        self._build_menus()
        self.resize(560, 300)

    # ------------------------------------------------------------------ contract
    def set_table(self, table: ResultsTable) -> None:
        self.table = table
        self.model.set_table(table)
        self.setWindowTitle(table.title)
        self.view.resizeColumnsToContents()
        if table.size:
            self.view.scrollToBottom()

    @property
    def session(self):
        return getattr(self.bar, "session", None)

    # ------------------------------------------------------------------ menus
    def _build_menus(self) -> None:
        mb = self.menuBar()
        f = mb.addMenu("File")
        self._act(f, "Save As...", self.save_as, QKeySequence.StandardKey.Save)
        self._act(f, "Rename...", self.rename)
        self._act(f, "Duplicate...", self.duplicate)
        e = mb.addMenu("Edit")
        self._act(e, "Copy", self.copy, QKeySequence.StandardKey.Copy)
        self._act(e, "Clear", self.delete_rows)
        self._act(e, "Select All", self.view.selectAll, QKeySequence.StandardKey.SelectAll)
        e.addSeparator()
        self._act(e, "Delete Rows...", self.delete_rows_dialog)
        self._act(e, "Sort...", self.sort_dialog)
        self._act(e, "Plot...", self.plot_dialog)
        self._act(e, "Summarize", self.summarize)
        self._act(e, "Distribution...", self.distribution)
        self._act(e, "Set Decimal Places...", self.decimals_dialog)
        fnt = mb.addMenu("Font")
        for size in (9, 10, 11, 12, 14, 16, 18, 24):
            self._act(fnt, f"{size}", lambda s=size: self.set_font_size(s))
        if self.table.title == "Results":
            r = mb.addMenu("Results")
            self._act(r, "Clear Results", self.clear_results)
            self._act(r, "Summarize", self.summarize)
            self._act(r, "Distribution...", self.distribution)
            self._act(r, "Set Measurements...", lambda: self._run("Set Measurements..."))
            self._act(r, "Rename...", self.rename)
            self._act(r, "Duplicate...", self.duplicate)

    def _act(self, menu, text, slot, shortcut=None):
        a = QAction(text, self)
        if shortcut is not None:
            a.setShortcut(shortcut)
        a.triggered.connect(lambda _=False: slot())
        menu.addAction(a)
        return a

    def _run(self, label, options=None):
        if self.bar is not None:
            return self.bar.run_command(label, options)
        return None

    # ------------------------------------------------------------------ actions
    def selected_rows(self) -> list[int]:
        return sorted({i.row() for i in self.view.selectionModel().selectedRows()})

    def text(self, rows: list[int] | None = None) -> str:
        t = self.table
        out = t.to_text("\t")
        if rows:
            lines = out.splitlines()
            out = "\n".join([lines[0]] + [lines[r + 1] for r in rows if r + 1 < len(lines)]) \
                + "\n"
        return out

    def copy(self) -> str:
        text = self.text(self.selected_rows() or None)
        QGuiApplication.clipboard().setText(text)
        return text

    def save(self, path: str) -> None:
        p = Path(path)
        ext = p.suffix.lower()
        if ext == ".xlsx":
            df = self.table.to_dataframe()
            try:
                df.to_excel(p, index=False)
            except ImportError:
                p = p.with_suffix(".xls")
                self.table.save(p)
        else:
            self.table.save(p)

    def save_as(self, path: str | None = None) -> str | None:
        if path is None:
            ext = ".csv"
            if self.session is not None:
                ext = self.session.options.results_extension or ".csv"
            name = (self.table.title if self.table.title != "Results" else "Results") + ext
            path, _ = QFileDialog.getSaveFileName(
                self, "Save as", name, "CSV (*.csv);;Text (*.txt *.tsv);;Excel text (*.xls);;"
                                       "Excel (*.xlsx)")
            if not path:
                return None
        self.save(path)
        return path

    def rename(self, title: str | None = None) -> None:
        if title is None:
            title, ok = QInputDialog.getText(self, "Rename", "Title:", text=self.table.title)
            if not ok or not title:
                return
        s = self.session
        old = self.table.title
        self.table.title = title
        if s is not None:
            if s.tables.get(old) is self.table:
                del s.tables[old]
            if old == "Results":
                s.results = ResultsTable.results()
                s.tables["Results"] = s.results
            s.tables[title] = self.table
            wm = getattr(self.bar, "window_manager", None)
            if wm is not None:
                wm.tables.pop(old, None)
                wm.tables[title] = self
        self.setWindowTitle(title)

    def duplicate(self, title: str | None = None):
        if title is None:
            title, ok = QInputDialog.getText(self, "Duplicate", "Title:",
                                             text=self.table.title + "-1")
            if not ok or not title:
                return None
        dup = self.table.copy(title)
        if self.session is not None:
            self.session.show_table(dup)
            return dup
        win = ResultsWindow(self.bar, dup)
        from rsstatchem.ui.imagej import windows as W
        W.register(win)
        win.show()
        return dup

    def delete_rows(self) -> None:
        rows = self.selected_rows()
        if not rows:
            if self.table.title == "Results":
                self.clear_results()
            return
        for r in reversed(rows):
            self.table.delete_row(r)
        self.set_table(self.table)

    def delete_rows_dialog(self, first: int | None = None, last: int | None = None) -> None:
        if first is None:
            text, ok = QInputDialog.getText(self, "Delete Rows", "Rows (e.g. 1-5):")
            if not ok:
                return
            a, _, b = text.partition("-")
            first, last = int(a), int(b or a)
        for r in range(int(last) - 1, int(first) - 2, -1):
            if 0 <= r < self.table.size:
                self.table.delete_row(r)
        self.set_table(self.table)

    def sort_dialog(self, column: str | None = None, ascending: bool = True) -> None:
        heads = [h for h in self.table.headings if h != LABEL]
        if not heads:
            return
        if column is None:
            column, ok = QInputDialog.getItem(self, "Sort", "Column:", heads, 0, False)
            if not ok:
                return
        self.table.sort(column, ascending)
        self.set_table(self.table)

    def plot_dialog(self, x: str | None = None, y: str | None = None):
        from rsstatchem.core.imaging.profile import PlotData
        heads = [h for h in self.table.headings if h != LABEL]
        if not heads:
            return None
        if y is None:
            y, ok = QInputDialog.getItem(self, "Plot", "Y column:", heads, 0, False)
            if not ok:
                return None
        plot = PlotData(f"{self.table.title} ({y})", x or "Row", y)
        import numpy as np
        xv = self.table.column(x) if x else np.arange(1, self.table.size + 1)
        plot.add(xv, self.table.column(y), style="connected circle")
        if self.bar is not None:
            from rsstatchem.ui.imagej.frames.plot_window import show_plot
            show_plot(self.bar, plot)
        return plot

    def summarize(self) -> None:
        if self.table.title == "Results" and self.bar is not None:
            self._run("Summarize", "")
            return
        from rsstatchem.core.imaging.measure import summarize
        summarize(self.table)
        self.set_table(self.table)

    def distribution(self) -> None:
        self._run("Distribution...")

    def clear_results(self) -> None:
        if self.bar is not None and self.table is getattr(self.session, "results", None):
            self._run("Clear Results", "")
        else:
            self.table.reset()
            self.set_table(self.table)

    def decimals_dialog(self, places: int | None = None) -> None:
        if places is None:
            places, ok = QInputDialog.getInt(self, "Set Decimal Places", "Decimal places:",
                                             self.table.precision if self.table.precision >= 0
                                             else 3, 0, 9)
            if not ok:
                return
        self.table.precision = int(places)
        self.table.decimal_places = {}
        self.set_table(self.table)

    def set_font_size(self, size: int) -> None:
        self._font_size = size
        self.view.setFont(QFont("Menlo", size))
        self.view.verticalHeader().setDefaultSectionSize(int(size * 1.6))
        self.view.resizeColumnsToContents()

    def closeEvent(self, event) -> None:  # noqa: N802
        from rsstatchem.ui.imagej import windows as W
        event.accept()
        W.unregister(self)

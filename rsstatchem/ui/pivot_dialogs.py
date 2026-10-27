"""Pivot Table Editor dialogs for the Output Viewer (SPSS: double-click a table).

* :class:`TableEditorDialog` — edit the title, column labels, any cell and the footnotes,
  hide/show rows and columns, and change the decimals of numeric cells.
* :class:`PivotTraysDialog` — SPSS's pivoting trays: move dimensions between Rows, Columns
  and Layers (and choose the layer category shown).
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QDialogButtonBox,
                               QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem, QPlainTextEdit,
                               QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from rsstatchem.core import pivot as pv
from rsstatchem.core.output import Table

_HEADER_BG = Qt.GlobalColor.lightGray


class TableEditorDialog(QDialog):
    """Edits a copy of the table; :meth:`result_table` returns it after OK."""

    def __init__(self, table: Table, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(f"Edit Table - {table.title}")
        self.resize(900, 600)
        self.table = pv.copy_table(table)
        t = self.table
        self.title_edit = QLineEdit(t.title)
        self.n_head = 2 if t.has_groups else 1

        self.grid = QTableWidget()
        self.grid.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._fill_grid()

        self.notes = QPlainTextEdit("\n".join(t.footnotes))
        self.notes.setMaximumHeight(80)
        self.notes.setPlaceholderText("One footnote per line")

        hide_box = QGroupBox("Show / hide")
        hide = QHBoxLayout(hide_box)
        self.row_list = QListWidget()
        self.col_list = QListWidget()
        for r, row in enumerate(t.rows):
            label = " ".join(str(x) for x in row[:t.row_header_cols] if x) or f"Row {r + 1}"
            item = QListWidgetItem(f"{r + 1}. {label}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked if r in t.hidden_rows
                               else Qt.CheckState.Checked)
            self.row_list.addItem(item)
        for c, col in enumerate(t.columns):
            label = " ".join(p for p in col if p) if isinstance(col, tuple) else str(col)
            item = QListWidgetItem(label or f"Column {c + 1}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            if c < t.row_header_cols:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            item.setCheckState(Qt.CheckState.Unchecked if c in t.hidden_cols
                               else Qt.CheckState.Checked)
            self.col_list.addItem(item)
        rows_col = QVBoxLayout()
        rows_col.addWidget(QLabel("Rows"))
        rows_col.addWidget(self.row_list)
        cols_col = QVBoxLayout()
        cols_col.addWidget(QLabel("Columns"))
        cols_col.addWidget(self.col_list)
        hide.addLayout(rows_col)
        hide.addLayout(cols_col)

        fmt_box = QGroupBox("Cell format")
        fmt = QHBoxLayout(fmt_box)
        self.decimals = QSpinBox()
        self.decimals.setRange(0, 8)
        self.decimals.setValue(3)
        apply_sel = QPushButton("Apply to Selected Cells")
        apply_all = QPushButton("Apply to All Numbers")
        apply_sel.clicked.connect(lambda: self._apply_decimals(selected=True))
        apply_all.clicked.connect(lambda: self._apply_decimals(selected=False))
        fmt.addWidget(QLabel("Decimals:"))
        fmt.addWidget(self.decimals)
        fmt.addWidget(apply_sel)
        fmt.addWidget(apply_all)
        fmt.addStretch()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        form = QFormLayout()
        form.addRow("Title:", self.title_edit)
        lower = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(QLabel("Footnotes:"))
        left.addWidget(self.notes)
        left.addWidget(fmt_box)
        lower.addLayout(left, 3)
        lower.addWidget(hide_box, 2)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(QLabel("Cells (grey rows are the column labels; double-click to edit):"))
        layout.addWidget(self.grid, 1)
        layout.addLayout(lower)
        layout.addWidget(buttons)

    def _fill_grid(self) -> None:
        t = self.table
        n_cols = len(t.columns)
        self.grid.setColumnCount(n_cols)
        self.grid.setRowCount(self.n_head + len(t.rows))
        for c, col in enumerate(t.columns):
            if self.n_head == 2:
                top, sub = (col if isinstance(col, tuple) else (str(col), ""))
                self._set(0, c, top, header=True)
                self._set(1, c, sub, header=True)
            else:
                self._set(0, c, str(col), header=True)
        for r, row in enumerate(t.rows):
            for c in range(n_cols):
                self._set(self.n_head + r, c, "" if c >= len(row) or row[c] is None
                          else str(row[c]))
        self.grid.resizeColumnsToContents()

    def _set(self, r: int, c: int, text: str, header: bool = False) -> None:
        item = QTableWidgetItem(text)
        if header:
            item.setBackground(_HEADER_BG)
        self.grid.setItem(r, c, item)

    def _sync_from_grid(self) -> None:
        t = self.table
        n_cols = len(t.columns)
        text = lambda r, c: (self.grid.item(r, c).text()  # noqa: E731
                             if self.grid.item(r, c) is not None else "")
        cols = []
        for c in range(n_cols):
            if self.n_head == 2:
                top, sub = text(0, c), text(1, c)
                cols.append((top, sub) if sub else top)
            else:
                cols.append(text(0, c))
        t.columns = cols
        t.rows = [[text(self.n_head + r, c) for c in range(n_cols)]
                  for r in range(len(t.rows))]

    def _apply_decimals(self, selected: bool) -> None:
        self._sync_from_grid()
        d = self.decimals.value()
        if selected:
            cells = {(i.row() - self.n_head, i.column()) for i in self.grid.selectedIndexes()
                     if i.row() >= self.n_head}
            for r, c in cells:
                self.table.rows[r][c] = pv.format_cell(self.table.rows[r][c], d)
        else:
            pv.set_decimals(self.table, d)
        self._fill_grid()

    def result_table(self) -> Table:
        self._sync_from_grid()
        t = self.table
        t.title = self.title_edit.text()
        t.footnotes = [line for line in self.notes.toPlainText().splitlines() if line.strip()]
        t.hidden_rows = [r for r in range(self.row_list.count())
                         if self.row_list.item(r).checkState() != Qt.CheckState.Checked]
        t.hidden_cols = [c for c in range(self.col_list.count())
                         if self.col_list.item(c).checkState() != Qt.CheckState.Checked
                         and c >= t.row_header_cols]
        return t


class PivotTraysDialog(QDialog):
    """Rows / Columns / Layers trays holding the table's dimensions."""

    def __init__(self, table: Table, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Pivoting Trays")
        self.resize(640, 420)
        self.cube = pv.source_cube(table)
        layout0 = pv.default_layout(table)
        self.trays = {k: QListWidget() for k in ("layers", "rows", "cols")}
        for key, dims in (("rows", layout0["rows"]), ("cols", layout0["cols"]),
                          ("layers", list(layout0["layers"]))):
            for d in dims:
                self._add(key, d)
        self.layer_values = dict(layout0["layers"])
        self.layer_box = QComboBox()
        self.layer_dim = QLabel("")
        self.trays["layers"].currentRowChanged.connect(self._show_layer)
        self.layer_box.currentTextChanged.connect(self._layer_changed)

        grid = QGridLayout()
        titles = {"layers": "Layer", "rows": "Row", "cols": "Column"}
        for i, key in enumerate(("layers", "rows", "cols")):
            box = QGroupBox(titles[key])
            v = QVBoxLayout(box)
            v.addWidget(self.trays[key])
            buttons = QHBoxLayout()
            for target in ("layers", "rows", "cols"):
                if target == key:
                    continue
                b = QPushButton(f"→ {titles[target]}")
                b.clicked.connect(lambda _=False, s=key, t=target: self._move(s, t))
                buttons.addWidget(b)
            up, down = QPushButton("▲"), QPushButton("▼")
            up.clicked.connect(lambda _=False, s=key: self._shift(s, -1))
            down.clicked.connect(lambda _=False, s=key: self._shift(s, 1))
            buttons.addWidget(up)
            buttons.addWidget(down)
            v.addLayout(buttons)
            grid.addWidget(box, 0, i)
        layer_row = QHBoxLayout()
        layer_row.addWidget(QLabel("Layer shown:"))
        layer_row.addWidget(self.layer_dim)
        layer_row.addWidget(self.layer_box, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Move dimensions between the trays to pivot the table "
                                "(SPSS Pivot > Pivoting Trays)."))
        layout.addLayout(grid, 1)
        layout.addLayout(layer_row)
        layout.addWidget(buttons)

    def _add(self, key: str, dim: int) -> None:
        item = QListWidgetItem(self.cube.names[dim])
        item.setData(Qt.ItemDataRole.UserRole, dim)
        self.trays[key].addItem(item)

    def _move(self, source: str, target: str) -> None:
        w = self.trays[source]
        row = w.currentRow()
        if row < 0:
            return
        item = w.takeItem(row)
        dim = item.data(Qt.ItemDataRole.UserRole)
        self._add(target, dim)
        if target == "layers":
            self.layer_values.setdefault(dim, self.cube.values[dim][0]
                                         if self.cube.values[dim] else "")
            self.trays["layers"].setCurrentRow(self.trays["layers"].count() - 1)
        elif source == "layers":
            self.layer_values.pop(dim, None)
            self._show_layer(self.trays["layers"].currentRow())

    def _shift(self, key: str, step: int) -> None:
        w = self.trays[key]
        row = w.currentRow()
        new = row + step
        if row < 0 or not 0 <= new < w.count():
            return
        item = w.takeItem(row)
        w.insertItem(new, item)
        w.setCurrentRow(new)

    def _show_layer(self, row: int) -> None:
        self.layer_box.blockSignals(True)
        self.layer_box.clear()
        if row >= 0:
            dim = self.trays["layers"].item(row).data(Qt.ItemDataRole.UserRole)
            self.layer_dim.setText(self.cube.names[dim] + ":")
            self.layer_box.addItems(self.cube.values[dim])
            self.layer_box.setCurrentText(self.layer_values.get(dim, ""))
        else:
            self.layer_dim.setText("")
        self.layer_box.blockSignals(False)

    def _layer_changed(self, text: str) -> None:
        row = self.trays["layers"].currentRow()
        if row >= 0:
            dim = self.trays["layers"].item(row).data(Qt.ItemDataRole.UserRole)
            self.layer_values[dim] = text

    def layout_spec(self) -> dict:
        dims = {k: [w.item(i).data(Qt.ItemDataRole.UserRole) for i in range(w.count())]
                for k, w in self.trays.items()}
        return {"rows": dims["rows"], "cols": dims["cols"],
                "layers": {d: self.layer_values.get(d, "") for d in dims["layers"]}}

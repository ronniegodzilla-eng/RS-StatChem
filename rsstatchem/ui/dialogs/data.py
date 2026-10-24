"""Data-management dialogs: Sort Cases, Value Labels, text import wizard."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QPushButton, QRadioButton, QSpinBox, QTableWidget,
                               QTableWidgetItem, QVBoxLayout)

from rsstatchem.core.dataset import NUMERIC, Variable, format_code
from rsstatchem.io import files
from rsstatchem.ui.dialogs.base import NAME_ROLE, VariablePickerDialog

ORDER_ROLE = Qt.ItemDataRole.UserRole + 1


class SortCasesDialog(VariablePickerDialog):
    def __init__(self, ds, parent=None):
        super().__init__(ds, "Sort Cases", parent)
        self.target = self.add_target("by", "Sort by:")
        box = QGroupBox("Sort Order")
        row = QHBoxLayout(box)
        self.asc = QRadioButton("Ascending")
        self.desc = QRadioButton("Descending")
        self.asc.setChecked(True)
        row.addWidget(self.asc)
        row.addWidget(self.desc)
        row.addStretch()
        self.options_layout.addWidget(box)
        self.target.model().rowsInserted.connect(self._tag_new)
        self.desc.toggled.connect(self._apply_order_to_selection)
        self.finish()

    def _tag(self, item, ascending: bool) -> None:
        item.setData(ORDER_ROLE, ascending)
        base = item.text().rsplit(" (", 1)[0] if item.text().endswith(("(A)", "(D)")) \
            else item.text()
        item.setText(f"{base} ({'A' if ascending else 'D'})")

    def _tag_new(self, _parent, first: int, last: int) -> None:
        for i in range(first, last + 1):
            item = self.target.item(i)
            if item is not None and item.data(ORDER_ROLE) is None:
                self._tag(item, self.asc.isChecked())

    def _apply_order_to_selection(self) -> None:
        for item in self.target.selectedItems():
            self._tag(item, self.asc.isChecked())

    def params(self) -> dict:
        items = [self.target.item(i) for i in range(self.target.count())]
        return {"names": [i.data(NAME_ROLE) for i in items],
                "ascending": [bool(i.data(ORDER_ROLE)) for i in items]}


class ValueLabelsDialog(QDialog):
    def __init__(self, var: Variable, parent=None):
        super().__init__(parent)
        self.var = var
        self.labels = dict(var.value_labels)
        self.setWindowTitle(f"Value Labels — {var.name}")
        self.setMinimumWidth(420)

        self.value = QLineEdit()
        self.label = QLineEdit()
        form = QFormLayout()
        form.addRow("Value:", self.value)
        form.addRow("Label:", self.label)

        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._pick)
        add, change, remove = QPushButton("Add"), QPushButton("Change"), QPushButton("Remove")
        add.clicked.connect(self._add)
        change.clicked.connect(self._add)
        remove.clicked.connect(self._remove)
        side = QVBoxLayout()
        for b in (add, change, remove):
            side.addWidget(b)
        side.addStretch()
        lists = QHBoxLayout()
        lists.addLayout(side)
        lists.addWidget(self.list, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        self.error = QLabel()
        self.error.setStyleSheet("color: #B71C1C;")

        box = QGroupBox("Value Labels")
        inner = QVBoxLayout(box)
        inner.addLayout(form)
        inner.addLayout(lists)
        layout = QVBoxLayout(self)
        layout.addWidget(box)
        layout.addWidget(self.error)
        layout.addWidget(buttons)
        self.label.returnPressed.connect(self._add)
        self._refresh()

    def _key(self, text: str):
        text = text.strip()
        if self.var.type == NUMERIC:
            return float(text.replace(",", "."))
        return text

    def _refresh(self) -> None:
        self.list.clear()
        for code in sorted(self.labels, key=lambda k: (isinstance(k, str), k)):
            item_text = f'{format_code(code)} = "{self.labels[code]}"'
            self.list.addItem(item_text)
            self.list.item(self.list.count() - 1).setData(NAME_ROLE, code)

    def _pick(self, row: int) -> None:
        if row < 0:
            return
        code = self.list.item(row).data(NAME_ROLE)
        self.value.setText(format_code(code))
        self.label.setText(self.labels[code])

    def _add(self) -> None:
        if not self.value.text().strip() or not self.label.text().strip():
            return
        try:
            key = self._key(self.value.text())
        except ValueError:
            self.error.setText("Value must be a number for a numeric variable.")
            return
        self.error.clear()
        self.labels[key] = self.label.text().strip()
        self._refresh()
        self.value.clear()
        self.label.clear()
        self.value.setFocus()

    def _remove(self) -> None:
        item = self.list.currentItem()
        if item is not None:
            self.labels.pop(item.data(NAME_ROLE), None)
            self._refresh()

    def _accept(self) -> None:
        if self.value.text().strip() and self.label.text().strip():
            self._add()
        self.accept()


class TextImportDialog(QDialog):
    """Delimiter / decimal / header / encoding options with a live preview."""

    DELIMITERS = (("Comma ( , )", ","), ("Semicolon ( ; )", ";"), ("Tab", "\t"),
                  ("Pipe ( | )", "|"), ("Space", " "))
    DECIMALS = (("Period ( . )", "."), ("Comma ( , )", ","))

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path = path
        self.setWindowTitle(f"Import Text Data — {Path(path).name}")
        self.resize(760, 520)
        encoding = files.detect_encoding(path)
        delimiter, decimal = files.sniff_text(path, encoding)

        self.encoding = QComboBox()
        self.encoding.addItems(files.ENCODINGS)
        self.encoding.setCurrentText(encoding)
        self.delimiter = QComboBox()
        for text, value in self.DELIMITERS:
            self.delimiter.addItem(text, value)
        self.delimiter.setCurrentIndex(max(self.delimiter.findData(delimiter), 0))
        self.decimal = QComboBox()
        for text, value in self.DECIMALS:
            self.decimal.addItem(text, value)
        self.decimal.setCurrentIndex(max(self.decimal.findData(decimal), 0))
        self.header = QCheckBox("First row contains variable names")
        self.header.setChecked(True)
        self.skip = QSpinBox()
        self.skip.setRange(0, 1000)

        form = QFormLayout()
        form.addRow("Encoding:", self.encoding)
        form.addRow("Delimiter:", self.delimiter)
        form.addRow("Decimal symbol:", self.decimal)
        form.addRow("Skip lines at top:", self.skip)
        form.addRow("", self.header)

        self.preview = QTableWidget()
        self.preview.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.status = QLabel()
        self.status.setStyleSheet("color: #666;")

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Import")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.ok = buttons.button(QDialogButtonBox.StandardButton.Ok)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(QLabel("Preview (first 100 rows):"))
        layout.addWidget(self.preview, 1)
        layout.addWidget(self.status)
        layout.addWidget(buttons)

        for w in (self.encoding, self.delimiter, self.decimal):
            w.currentIndexChanged.connect(self._update_preview)
        self.header.toggled.connect(self._update_preview)
        self.skip.valueChanged.connect(self._update_preview)
        self._update_preview()

    def params(self) -> dict:
        return {"delimiter": self.delimiter.currentData(), "decimal": self.decimal.currentData(),
                "header": self.header.isChecked(), "encoding": self.encoding.currentText(),
                "skip_rows": self.skip.value()}

    def _update_preview(self) -> None:
        try:
            df = files.read_text(self.path, nrows=100, **self.params())
        except Exception as exc:  # noqa: BLE001 - show any parser error to the user
            self.preview.clear()
            self.preview.setRowCount(0)
            self.preview.setColumnCount(0)
            self.status.setText(f"Cannot parse with these settings: {exc}")
            self.ok.setEnabled(False)
            return
        self.ok.setEnabled(True)
        self.preview.clear()
        self.preview.setColumnCount(df.shape[1])
        self.preview.setRowCount(df.shape[0])
        self.preview.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r in range(df.shape[0]):
            for c in range(df.shape[1]):
                value = df.iat[r, c]
                self.preview.setItem(r, c, QTableWidgetItem("" if value != value else str(value)))
        kinds = ", ".join(f"{c}: {'numeric' if str(t).startswith(('int', 'float')) else 'text'}"
                          for c, t in list(df.dtypes.items())[:6])
        self.status.setText(f"{df.shape[1]} variables detected — {kinds}"
                            + (" …" if df.shape[1] > 6 else ""))

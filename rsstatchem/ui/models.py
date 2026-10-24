"""Table models behind Data View and Variable View."""
from __future__ import annotations

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont

from rsstatchem.core.dataset import (ALIGNS, DATE, MEASURES, NUMERIC, ROLES, STRING, TYPES,
                                     format_code)
from rsstatchem.ui import icons
from rsstatchem.ui.document import DatasetDocument

_ALIGN = {"left": Qt.AlignmentFlag.AlignLeft, "right": Qt.AlignmentFlag.AlignRight,
          "center": Qt.AlignmentFlag.AlignHCenter}
_MUTED = QBrush(QColor("#A0A0A0"))
_TYPE_NAMES = {NUMERIC: "Numeric", STRING: "String", DATE: "Date"}


class DataTableModel(QAbstractTableModel):
    """Data View grid. Shows spare empty rows/columns; typing into them grows the dataset."""

    EXTRA_ROWS, MIN_ROWS = 50, 100
    EXTRA_COLS, MIN_COLS = 6, 24
    error = Signal(str)

    def __init__(self, doc: DatasetDocument, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.show_labels = False
        doc.structure_changed.connect(self._reset)
        doc.values_changed.connect(self._values_changed)

    @property
    def ds(self):
        return self.doc.ds

    def _reset(self) -> None:
        self.beginResetModel()
        self.endResetModel()

    def _values_changed(self, top: int, left: int, bottom: int, right: int) -> None:
        self.dataChanged.emit(self.index(top, left), self.index(bottom, right))

    def set_show_labels(self, on: bool) -> None:
        self.show_labels = on
        self._reset()

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else max(self.ds.n_cases + self.EXTRA_ROWS, self.MIN_ROWS)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else max(self.ds.n_vars + self.EXTRA_COLS, self.MIN_COLS)

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        r, c = index.row(), index.column()
        inside = r < self.ds.n_cases and c < self.ds.n_vars
        if role == Qt.ItemDataRole.DisplayRole:
            return self.ds.display_value(r, c, self.show_labels) if inside else ""
        if role == Qt.ItemDataRole.EditRole:
            if not inside:
                return ""
            if self.show_labels:
                label = self.ds.variables[c].label_for(self.ds.get_value(r, c))
                if label is not None:
                    return label
            return self.ds.edit_text(r, c)
        if role == Qt.ItemDataRole.TextAlignmentRole and c < self.ds.n_vars:
            return _ALIGN[self.ds.variables[c].align] | Qt.AlignmentFlag.AlignVCenter
        return None

    def headerData(self, section: int, orientation, role=Qt.ItemDataRole.DisplayRole):
        horizontal = orientation == Qt.Orientation.Horizontal
        if role == Qt.ItemDataRole.DisplayRole:
            if horizontal:
                return self.ds.variables[section].name if section < self.ds.n_vars else "var"
            return str(section + 1)
        if role == Qt.ItemDataRole.ForegroundRole:
            limit = self.ds.n_vars if horizontal else self.ds.n_cases
            if section >= limit or (not horizontal and not self._selected(section)):
                return _MUTED
            return None
        if role == Qt.ItemDataRole.FontRole and not horizontal and not self._selected(section):
            font = QFont()
            font.setStrikeOut(True)  # SPSS marks filtered-out cases on the row number
            return font
        if role == Qt.ItemDataRole.ToolTipRole and horizontal and section < self.ds.n_vars:
            var = self.ds.variables[section]
            return f"{var.name}: {var.label}" if var.label else var.name
        if role == Qt.ItemDataRole.DecorationRole and horizontal and section < self.ds.n_vars:
            return icons.variable_icon(self.ds.variables[section])
        return None

    def _selected(self, row: int) -> bool:
        name = self.ds.meta.get("filter")
        if not name or row >= self.ds.n_cases or not self.ds.has_name(name):
            return True
        value = self.ds.df[self.ds.var(name).name].iat[row]
        return bool(value) and value == value  # zero or missing = filtered out

    def flags(self, index: QModelIndex):
        return (Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEditable)

    def setData(self, index: QModelIndex, value, role=Qt.ItemDataRole.EditRole) -> bool:
        if role != Qt.ItemDataRole.EditRole:
            return False
        r, c = index.row(), index.column()
        if r >= self.ds.n_cases and c < self.ds.n_vars and str(value).strip() == "":
            return False
        if c >= self.ds.n_vars and str(value).strip() == "":
            return False
        try:
            self.doc.set_value(r, c, value)
        except ValueError as exc:
            self.error.emit(str(exc))
            return False
        return True


class VariableTableModel(QAbstractTableModel):
    """Variable View grid: one row per variable, SPSS columns."""

    COLUMNS = ("Name", "Type", "Width", "Decimals", "Label", "Values", "Missing", "Columns",
               "Align", "Measure", "Role")
    ATTR = ("name", "type", "width", "decimals", "label", "value_labels", "missing", "columns",
            "align", "measure", "role")
    CHOICES = {"type": TYPES, "align": ALIGNS, "measure": MEASURES, "role": ROLES}
    EXTRA_ROWS, MIN_ROWS = 30, 50
    error = Signal(str)

    def __init__(self, doc: DatasetDocument, parent=None):
        super().__init__(parent)
        self.doc = doc
        doc.structure_changed.connect(self._reset)

    @property
    def ds(self):
        return self.doc.ds

    def _reset(self) -> None:
        self.beginResetModel()
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else max(self.ds.n_vars + self.EXTRA_ROWS, self.MIN_ROWS)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.COLUMNS)

    @staticmethod
    def values_text(var) -> str:
        if not var.value_labels:
            return "None"
        code, label = next(iter(var.value_labels.items()))
        more = "..." if len(var.value_labels) > 1 else ""
        return f"{{{format_code(code)}, {label}}}{more}"

    @staticmethod
    def missing_text(var) -> str:
        return ", ".join(format_code(m) for m in var.missing) if var.missing else "None"

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        r, c = index.row(), index.column()
        if r >= self.ds.n_vars:
            return None
        var = self.ds.variables[r]
        attr = self.ATTR[c]
        if role == Qt.ItemDataRole.DisplayRole:
            if attr == "type":
                return _TYPE_NAMES[var.type]
            if attr == "value_labels":
                return self.values_text(var)
            if attr == "missing":
                return self.missing_text(var)
            if attr in ("align", "measure", "role"):
                return getattr(var, attr).capitalize()
            return str(getattr(var, attr))
        if role == Qt.ItemDataRole.EditRole:
            if attr == "missing":
                return ", ".join(format_code(m) for m in var.missing)
            return getattr(var, attr)
        if role == Qt.ItemDataRole.DecorationRole:
            if attr == "measure":
                return icons.icon(var.measure)
            if attr == "type" and var.type != NUMERIC:
                return icons.icon(var.type)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if attr in ("width", "decimals", "columns"):
                return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        return None

    def headerData(self, section: int, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return self.COLUMNS[section]
            return str(section + 1)
        if role == Qt.ItemDataRole.ForegroundRole and orientation == Qt.Orientation.Vertical:
            return _MUTED if section >= self.ds.n_vars else None
        return None

    def flags(self, index: QModelIndex):
        flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        attr = self.ATTR[index.column()]
        if attr != "value_labels" and (index.row() < self.ds.n_vars or attr == "name"):
            flags |= Qt.ItemFlag.ItemIsEditable
        if attr == "decimals" and index.row() < self.ds.n_vars \
                and self.ds.variables[index.row()].type != NUMERIC:
            flags &= ~Qt.ItemFlag.ItemIsEditable
        return flags

    def setData(self, index: QModelIndex, value, role=Qt.ItemDataRole.EditRole) -> bool:
        if role != Qt.ItemDataRole.EditRole:
            return False
        r, attr = index.row(), self.ATTR[index.column()]
        try:
            if r >= self.ds.n_vars:
                if attr != "name" or not str(value).strip():
                    return False
                self.doc.mutate(lambda ds: ds.ensure_size(0, r + 1))
            var = self.ds.variables[r]
            if attr in ("width", "decimals", "columns"):
                value = int(value)
            elif attr == "missing":
                value = self._parse_missing(var, str(value))
            elif attr == "name":
                value = str(value).strip()
                if value == var.name:
                    return False
            elif attr == "label":
                value = str(value)
            self.doc.set_variable_attr(r, attr, value)
        except ValueError as exc:
            self.error.emit(str(exc))
            return False
        return True

    @staticmethod
    def _parse_missing(var, text: str) -> list:
        parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
        if var.type == NUMERIC:
            try:
                return [float(p) for p in parts][:3]
            except ValueError:
                raise ValueError("Missing values must be numbers, separated by commas.") from None
        return parts[:3]

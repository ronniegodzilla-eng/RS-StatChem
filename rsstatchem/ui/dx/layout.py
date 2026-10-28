"""The design layout grid (Design-Expert's "Design (Actual)" spreadsheet).

Columns: Select, Std, Run, Block (when blocked), Group (split-plot whole plots), Space Type
(optional), one column per factor
headed "Factor 1 / A:Temperature / °C" and one per response headed "Response 1 / R1:Yield / %".
Rows are shown in run order (or standard order). Response cells are editable; blocks of cells
copied from Excel paste in one step; every change is undoable. Unticking Select ignores a run
(it stays in the layout, greyed, but drops out of every analysis). Factors are shown in actual
or coded units (Display Options > Process Factors).

Historical (imported) designs have editable factor cells. Structural edits (insert / delete
runs, add a response, round factor values, augment the design, ...) are undoable through
:class:`StructureCommand`, which restores a snapshot of the document.
"""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QKeySequence, QUndoCommand
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QHeaderView, QMenu,
                               QTableView)

from rsstatchem.core.doe import designs
from rsstatchem.ui.dx import style


def parse_number(text) -> float:
    t = str(text).strip().replace(" ", "").replace(" ", "")
    if not t:
        return float("nan")
    if t.count(",") == 1 and "." not in t:
        t = t.replace(",", ".")
    else:
        t = t.replace(",", "")
    return float(t)


def format_value(v: float) -> str:
    if v is None or not math.isfinite(v):
        return ""
    return f"{v:.6g}"


class SetValuesCommand(QUndoCommand):
    """Undoable change of response cells: [(row, column, old, new)]."""

    def __init__(self, doc, changes: list[tuple[int, str, float, float]], text: str):
        super().__init__(text)
        self.doc = doc
        self.changes = changes

    def redo(self) -> None:
        self.doc.set_values([(r, c, new) for r, c, _, new in self.changes])

    def undo(self) -> None:
        self.doc.set_values([(r, c, old) for r, c, old, _ in self.changes])


class StructureCommand(QUndoCommand):
    """Undo for a structural edit: document snapshots before and after."""

    def __init__(self, doc, before: dict, after: dict, text: str):
        super().__init__(text)
        self.doc, self.before, self.after = doc, before, after
        self._first = True

    def redo(self) -> None:
        if self._first:  # already applied when pushed
            self._first = False
            return
        self.doc.restore(self.after)

    def undo(self) -> None:
        self.doc.restore(self.before)


def structural(undo_stack, doc, text: str, fn):
    """Run ``fn()`` (a document edit) as one undoable step; returns its result. Errors
    leave the document unchanged and propagate."""
    before = doc.snapshot()
    try:
        result = fn()
    except Exception:
        doc.restore(before)
        raise
    if undo_stack is not None:
        undo_stack.push(StructureCommand(doc, before, doc.snapshot(), text))
    return result


class IgnoreCommand(QUndoCommand):
    def __init__(self, doc, rows: list[int], ignored: bool):
        super().__init__("Ignore Run" if ignored else "Include Run")
        self.doc, self.rows, self.ignored = doc, rows, ignored
        self.before = set(doc.ignored)

    def redo(self) -> None:
        self.doc.set_ignored(self.rows, self.ignored)

    def undo(self) -> None:
        self.doc.ignored = set(self.before)
        self.doc.touch("data", list(self.doc.response_cols))


class LayoutModel(QAbstractTableModel):
    def __init__(self, doc, undo_stack, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.undo = undo_stack
        self.coded = False
        self.show_type = False
        self.order = "run"
        self.rows: list[int] = []
        self.columns: list[tuple[str, str]] = []  # (kind, dataset column)
        self.rebuild()

    # ------------------------------------------------------------------ structure
    def rebuild(self) -> None:
        self.beginResetModel()
        doc = self.doc
        cols = [("select", ""), ("std", "Std"), ("run", "Run")]
        if doc.n_blocks > 1:
            cols.append(("block", "Block"))
        if doc.group_col:
            cols.append(("group", doc.group_col))
        if self.show_type:
            cols.append(("type", "Space_Type"))
        cols += [("factor", c) for c in doc.factor_cols]
        cols += [("response", c) for c in doc.response_cols]
        self.columns = cols
        self.rows = doc.row_order(self.order)
        self._coded = doc.coded_matrix() if self.coded else None
        self.endResetModel()

    def refresh(self) -> None:
        self.rows = self.doc.row_order(self.order)
        if self.coded:
            self._coded = self.doc.coded_matrix()
        if self.rowCount() and self.columnCount():
            self.dataChanged.emit(self.index(0, 0),
                                  self.index(self.rowCount() - 1, self.columnCount() - 1))
        self.headerDataChanged.emit(Qt.Orientation.Horizontal, 0, self.columnCount() - 1)

    def set_order(self, order: str) -> None:
        self.order = order
        self.rebuild()

    def set_coded(self, coded: bool) -> None:
        self.coded = coded
        self.rebuild()

    def set_show_type(self, on: bool) -> None:
        self.show_type = on
        self.rebuild()

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.columns)

    def response_column(self, c: int) -> str | None:
        kind, col = self.columns[c]
        return col if kind == "response" else None

    # ------------------------------------------------------------------ display
    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Vertical:
            if role == Qt.ItemDataRole.DisplayRole:
                return str(section + 1)
            return None
        if section >= len(self.columns):
            return None
        kind, col = self.columns[section]
        doc = self.doc
        if role == Qt.ItemDataRole.DisplayRole:
            if kind == "select":
                return "Select"
            if kind in ("std", "run", "block", "group"):
                return {"std": "Std", "run": "Run", "block": "Block", "group": "Group"}[kind]
            if kind == "type":
                return "Space Type"
            if kind == "factor":
                j = doc.factor_cols.index(col)
                f = doc.factors[j]
                units = f.get("units", "") if not self.coded else "coded"
                head = "Component" if doc.mixture or f.get("type") == "mixture" else "Factor"
                if f.get("changes") == "hard":
                    units = (units + " " if units else "") + "(HTC)"
                return f"{head} {j + 1}\n{designs.letter(j)}:{f['name']}\n{units}"
            i = doc.response_cols.index(col)
            r = doc.responses[i]
            return f"Response {i + 1}\nR{i + 1}:{r['name']}\n{r.get('units', '')}"
        if role == Qt.ItemDataRole.ToolTipRole:
            if kind == "select":
                return "Untick to ignore a run (it is left out of the analyses)."
            if kind == "group":
                return "Whole plot: the hard-to-change factors stay at the same setting for all " \
                       "runs of a group (split-plot design)."
            if kind == "response":
                return "Type or paste the measured values here."
            if kind == "factor":
                return "Factor settings for each run (Display Options > Process Factors " \
                       "switches between actual and coded units)."
        if role == Qt.ItemDataRole.ForegroundRole and kind == "response":
            return QBrush(QColor(style.BLUE_DARK))
        if role == Qt.ItemDataRole.FontRole and kind in ("factor", "response"):
            f = QFont()
            f.setBold(True)
            return f
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = self.rows[index.row()]
        kind, col = self.columns[index.column()]
        doc = self.doc
        ignored = row in doc.ignored
        if role == Qt.ItemDataRole.CheckStateRole and kind == "select":
            return Qt.CheckState.Unchecked if ignored else Qt.CheckState.Checked
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            if kind == "select":
                return None
            if kind in ("std", "run", "block", "group"):
                v = doc.value(row, col)
                if kind == "block" and math.isfinite(v):
                    return doc.ds.var("Block").value_labels.get(v, f"Block {v:g}")
                return f"{v:g}" if math.isfinite(v) else ""
            if kind == "type":
                return designs.SPACE_TYPES.get(int(doc.value(row, col)), "")
            if kind == "factor":
                j = doc.factor_cols.index(col)
                f = doc.factors[j]
                v = doc.value(row, col)
                if f.get("type") == "categoric":
                    labels = dict(zip(f.get("levels", []), f.get("labels", [])))
                    return str(labels.get(v, format_value(v)))
                if self.coded and self._coded is not None and f.get("type") != "categoric":
                    c = self._coded[row, j]
                    return f"{c:.4g}" if math.isfinite(c) else ""
                return format_value(v)
            v = doc.value(row, col)
            return format_value(v)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if kind == "select":
                return int(Qt.AlignmentFlag.AlignCenter)
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.BackgroundRole:
            if kind == "group":
                g = doc.value(row, col)
                return QBrush(QColor("#FCE8D2" if math.isfinite(g) and int(g) % 2 else
                                     "#F6F0E6"))
            if kind in ("std", "run", "block", "type", "select"):
                return QBrush(QColor("#EDF1F6"))
            if kind == "factor":
                if doc.factors[doc.factor_cols.index(col)].get("changes") == "hard":
                    return QBrush(QColor("#F7EBDD"))
                if doc.editable_factors and not math.isfinite(doc.value(row, col)):
                    return QBrush(QColor("#FFFBEA"))
                return QBrush(QColor(style.FACTOR_BG))
            if kind == "response" and not ignored and not math.isfinite(doc.value(row, col)):
                return QBrush(QColor("#FFFBEA"))
            return QBrush(QColor(style.RESPONSE_BG))
        if role == Qt.ItemDataRole.ForegroundRole:
            if ignored:
                return QBrush(QColor(style.IGNORED))
            if kind == "response":
                return QBrush(QColor("#000000"))
            return QBrush(QColor("#2A3440"))
        if role == Qt.ItemDataRole.FontRole and ignored:
            f = QFont()
            f.setStrikeOut(True)
            return f
        return None

    def flags(self, index):
        base = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        kind, _ = self.columns[index.column()]
        if kind == "select":
            return base | Qt.ItemFlag.ItemIsUserCheckable
        if kind == "response" or (kind == "factor" and self.doc.editable_factors):
            return base | Qt.ItemFlag.ItemIsEditable
        return base

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole) -> bool:
        if not index.isValid():
            return False
        kind, col = self.columns[index.column()]
        row = self.rows[index.row()]
        if kind == "select" and role == Qt.ItemDataRole.CheckStateRole:
            ignored = Qt.CheckState(value) != Qt.CheckState.Checked
            self.undo.push(IgnoreCommand(self.doc, [row], ignored))
            return True
        editable = kind == "response" or (kind == "factor" and self.doc.editable_factors)
        if not editable or role != Qt.ItemDataRole.EditRole:
            return False
        try:
            new = self.parse_cell(col, value)
        except ValueError:
            return False
        old = self.doc.value(row, col)
        if (math.isnan(old) and math.isnan(new)) or old == new:
            return False
        self.undo.push(SetValuesCommand(self.doc, [(row, col, old, new)],
                                        f"Edit {self.doc.column_title(col)}"))
        return True

    def parse_cell(self, col: str, value) -> float:
        """A typed value for a column: categoric factors accept a level name."""
        if col in self.doc.factor_cols:
            f = self.doc.factors[self.doc.factor_cols.index(col)]
            if f.get("type") == "categoric":
                text = str(value).strip()
                for lv, lab in zip(f.get("levels", []), f.get("labels", [])):
                    if text == str(lab):
                        return float(lv)
                v = parse_number(text)
                if not any(abs(v - lv) < 1e-9 for lv in f.get("levels", [])) and \
                        math.isfinite(v):
                    raise ValueError(f"{text} is not a level of {f['name']}.")
                return v
        return parse_number(value)

    def editable_column(self, c: int) -> str | None:
        kind, col = self.columns[c]
        if kind == "response" or (kind == "factor" and self.doc.editable_factors):
            return col
        return None

    # ------------------------------------------------------------------ bulk edits
    def paste_text(self, text: str, top: int, left: int) -> int:
        """Paste a tab/newline separated block at (top, left) view cell; only response
        columns take values. Returns the number of cells changed."""
        lines = [ln for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
        while lines and lines[-1] == "":
            lines.pop()
        changes = []
        bad = 0
        for r, line in enumerate(lines):
            vr = top + r
            if vr >= len(self.rows):
                break
            for c, cell in enumerate(line.split("\t")):
                vc = left + c
                if vc >= len(self.columns):
                    break
                col = self.editable_column(vc)
                if col is None:
                    continue
                try:
                    new = self.parse_cell(col, cell)
                except ValueError:
                    bad += 1
                    continue
                row = self.rows[vr]
                old = self.doc.value(row, col)
                if not ((math.isnan(old) and math.isnan(new)) or old == new):
                    changes.append((row, col, old, new))
        if changes:
            self.undo.push(SetValuesCommand(self.doc, changes, f"Paste {len(changes)} cells"))
        return len(changes)

    def clear_cells(self, indexes) -> int:
        changes = []
        for ix in indexes:
            col = self.editable_column(ix.column())
            if col is None:
                continue
            row = self.rows[ix.row()]
            old = self.doc.value(row, col)
            if math.isfinite(old):
                changes.append((row, col, old, float("nan")))
        if changes:
            self.undo.push(SetValuesCommand(self.doc, changes, "Clear cells"))
        return len(changes)

    def selection_text(self, indexes) -> str:
        if not indexes:
            return ""
        rows = sorted({i.row() for i in indexes})
        cols = sorted({i.column() for i in indexes})
        chosen = {(i.row(), i.column()) for i in indexes}
        lines = []
        for r in rows:
            cells = []
            for c in cols:
                if (r, c) in chosen:
                    v = self.data(self.index(r, c))
                    cells.append("" if v is None else str(v))
                else:
                    cells.append("")
            lines.append("\t".join(cells))
        return "\n".join(lines)

    def header_line(self, cols: list[int]) -> str:
        out = []
        for c in cols:
            h = self.headerData(c, Qt.Orientation.Horizontal) or ""
            out.append(" ".join(p for p in str(h).split("\n")[1:2]) or str(h).split("\n")[0])
        return "\t".join(out)


class LayoutView(QTableView):
    """Spreadsheet view of the design layout with copy/paste/delete and header menus."""

    status = Signal(str)

    def __init__(self, model: LayoutModel, parent=None):
        super().__init__(parent)
        self.setModel(model)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked
                             | QAbstractItemView.EditTrigger.EditKeyPressed
                             | QAbstractItemView.EditTrigger.AnyKeyPressed)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(23)
        hh = self.horizontalHeader()
        hh.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        hh.setMinimumHeight(62)
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        hh.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        hh.customContextMenuRequested.connect(self._header_menu)
        hh.sectionDoubleClicked.connect(self._header_double)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._cell_menu)
        self.setAlternatingRowColors(False)
        model.modelReset.connect(self.fit_columns)
        self.fit_columns()

    @property
    def layout_model(self) -> LayoutModel:
        return self.model()

    def fit_columns(self) -> None:
        m = self.layout_model
        for c, (kind, _) in enumerate(m.columns):
            if kind == "select":
                self.setColumnWidth(c, 52)
            elif kind in ("std", "run"):
                self.setColumnWidth(c, 46)
            elif kind in ("block", "type"):
                self.setColumnWidth(c, 92)
            elif kind == "group":
                self.setColumnWidth(c, 56)
            else:
                text = m.headerData(c, Qt.Orientation.Horizontal) or ""
                width = max(len(s) for s in str(text).split("\n")) * 7 + 24
                self.setColumnWidth(c, max(96, min(width, 220)))

    # ------------------------------------------------------------------ editing
    def keyPressEvent(self, event) -> None:
        if event.matches(QKeySequence.StandardKey.Paste):
            self.paste()
            return
        if event.matches(QKeySequence.StandardKey.Copy):
            self.copy()
            return
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace) and \
                self.state() != QAbstractItemView.State.EditingState:
            self.layout_model.clear_cells(self.selectedIndexes())
            return
        super().keyPressEvent(event)

    def paste(self) -> int:
        text = QApplication.clipboard().text()
        cur = self.currentIndex()
        sel = self.selectedIndexes()
        if sel:
            top = min(i.row() for i in sel)
            left = min(i.column() for i in sel)
        elif cur.isValid():
            top, left = cur.row(), cur.column()
        else:
            m = self.layout_model
            top, left = 0, next((c for c, (k, _) in enumerate(m.columns) if k == "response"), 0)
        n = self.layout_model.paste_text(text, top, left)
        self.status.emit(f"Pasted {n} response values." if n else
                         "Nothing pasted: select a response cell first (factor settings are "
                         "fixed by the design).")
        return n

    def copy(self) -> None:
        sel = self.selectedIndexes()
        if not sel:
            return
        QApplication.clipboard().setText(self.layout_model.selection_text(sel))

    # ------------------------------------------------------------------ menus
    def _header_double(self, section: int) -> None:
        kind, _ = self.layout_model.columns[section]
        if kind == "std":
            self.layout_model.set_order("std")
        elif kind == "run":
            self.layout_model.set_order("run")

    def _header_menu(self, pos) -> None:
        m = self.layout_model
        menu = QMenu(self)
        a = menu.addAction("Sort by Std Order", lambda: m.set_order("std"))
        a.setCheckable(True)
        a.setChecked(m.order == "std")
        b = menu.addAction("Sort by Run Order", lambda: m.set_order("run"))
        b.setCheckable(True)
        b.setChecked(m.order == "run")
        menu.addSeparator()
        c = menu.addAction("Space Point Type", lambda: m.set_show_type(not m.show_type))
        c.setCheckable(True)
        c.setChecked(m.show_type)
        menu.addSeparator()
        d = menu.addAction("Process Factors: Actual", lambda: m.set_coded(False))
        d.setCheckable(True)
        d.setChecked(not m.coded)
        e = menu.addAction("Process Factors: Coded", lambda: m.set_coded(True))
        e.setCheckable(True)
        e.setChecked(m.coded)
        menu.exec(self.horizontalHeader().mapToGlobal(pos))

    def _cell_menu(self, pos) -> None:
        m = self.layout_model
        menu = QMenu(self)
        menu.addAction("Copy", self.copy)
        menu.addAction("Paste", self.paste)
        menu.addAction("Clear Response Cells", lambda: m.clear_cells(self.selectedIndexes()))
        rows = sorted({m.rows[i.row()] for i in self.selectedIndexes()})
        if rows:
            menu.addSeparator()
            menu.addAction("Ignore Run(s)", lambda: m.undo.push(IgnoreCommand(m.doc, rows,
                                                                               True)))
            menu.addAction("Include Run(s)", lambda: m.undo.push(IgnoreCommand(m.doc, rows,
                                                                                False)))
        menu.addSeparator()
        menu.addAction("Insert Run", self.insert_run)
        if rows:
            menu.addAction(f"Delete {len(rows)} Run(s)", lambda: self.delete_runs(rows))
            menu.addAction("Duplicate Run(s)", lambda: self.duplicate_runs(rows))
        menu.exec(self.viewport().mapToGlobal(pos))

    # ------------------------------------------------------------------ structure
    def insert_run(self) -> int:
        m = self.layout_model
        n = structural(m.undo, m.doc, "Insert Run", lambda: m.doc.insert_runs(n=1))
        self.status.emit("Run added at the end of the layout." if n else "")
        return n

    def delete_runs(self, rows: list[int]) -> int:
        m = self.layout_model
        try:
            n = structural(m.undo, m.doc, f"Delete {len(rows)} Run(s)",
                           lambda: m.doc.delete_runs(rows))
        except ValueError as exc:
            self.status.emit(str(exc))
            return 0
        self.status.emit(f"Deleted {n} run(s); Std and Run renumbered.")
        return n

    def duplicate_runs(self, rows: list[int]) -> int:
        m = self.layout_model
        return structural(m.undo, m.doc, "Duplicate Run(s)",
                          lambda: m.doc.add_replicates(rows, new_block=False))


def response_stats(doc, col: str) -> dict:
    vals = doc.ds.df[col].to_numpy(float)
    keep = np.ones(len(vals), bool)
    keep[list(doc.ignored)] = False
    v = vals[keep & np.isfinite(vals)]
    return {"n": len(v), "min": float(v.min()) if len(v) else math.nan,
            "max": float(v.max()) if len(v) else math.nan,
            "mean": float(v.mean()) if len(v) else math.nan,
            "sd": float(v.std(ddof=1)) if len(v) > 1 else math.nan}

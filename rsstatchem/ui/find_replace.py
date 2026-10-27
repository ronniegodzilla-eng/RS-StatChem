"""Edit > Find / Replace for the Data Editor (SPSS: Find and Replace - Data View / Variable View).

Data View: searches the selected column (or all columns) by the text shown in the cells (value
labels too when they are displayed); Replace changes the current match or all matches as one
undoable step. Variable View: finds variables by name, label or value labels.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QRadioButton, QTabWidget, QVBoxLayout,
                               QWidget)

from rsstatchem.core import data_props as dp

_OPEN: dict = {}


class FindReplaceDialog(QDialog):
    def __init__(self, editor, replace: bool = False):
        super().__init__(editor)
        self.editor = editor
        self.setWindowTitle("Find and Replace - Data View")
        self.current: tuple[int, int] | None = None
        self.tabs = QTabWidget()
        page = QWidget()
        self.find_edit = QLineEdit()
        self.replace_edit = QLineEdit()
        self.do_replace = QCheckBox("Replace")
        self.do_replace.toggled.connect(self.replace_edit.setEnabled)
        self.do_replace.setChecked(replace)
        self.replace_edit.setEnabled(replace)
        self.column_only = QRadioButton("Selected column only")
        self.all_columns = QRadioButton("All columns")
        self.column_only.setChecked(True)
        self.match_case = QCheckBox("Match case")
        self.mode = QComboBox()
        for key, text in (("contains", "Contains"), ("entire", "Entire cell"),
                          ("begins", "Begins with"), ("ends", "Ends with")):
            self.mode.addItem(text, key)
        self.field = QComboBox()
        for key, text in (("name", "Name"), ("label", "Label"),
                          ("value_labels", "Value labels")):
            self.field.addItem(text, key)
        self.column_label = QLabel()
        form = QFormLayout(page)
        form.addRow("Column:", self.column_label)
        form.addRow("Find:", self.find_edit)
        rep = QHBoxLayout()
        rep.addWidget(self.do_replace)
        rep.addWidget(self.replace_edit, 1)
        form.addRow(rep)
        scope = QHBoxLayout()
        scope.addWidget(self.column_only)
        scope.addWidget(self.all_columns)
        form.addRow("Search:", scope)
        form.addRow("Match:", self.mode)
        form.addRow("", self.match_case)
        form.addRow("Variable View field:", self.field)
        self.tabs.addTab(page, "Find / Replace")
        self.status = QLabel()
        self.status.setWordWrap(True)
        b_next = QPushButton("Find Next")
        b_prev = QPushButton("Find Previous")
        b_rep = QPushButton("Replace")
        b_all = QPushButton("Replace All")
        b_close = QPushButton("Close")
        b_next.setDefault(True)
        b_next.clicked.connect(lambda: self.find_next())
        b_prev.clicked.connect(lambda: self.find_next(backwards=True))
        b_rep.clicked.connect(self.replace_one)
        b_all.clicked.connect(self.replace_everything)
        b_close.clicked.connect(self.close)
        self.replace_buttons = (b_rep, b_all)
        self.do_replace.toggled.connect(lambda on: [b.setEnabled(on)
                                                    for b in self.replace_buttons])
        for b in self.replace_buttons:
            b.setEnabled(replace)
        buttons = QHBoxLayout()
        for b in (b_next, b_prev, b_rep, b_all, b_close):
            buttons.addWidget(b)
        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)
        layout.addWidget(self.status)
        layout.addLayout(buttons)
        self.refresh_scope()

    # ------------------------------------------------------------------ helpers
    @property
    def ds(self):
        return self.editor.ds

    def in_variable_view(self) -> bool:
        return self.editor.tabs.currentIndex() == 1

    def refresh_scope(self) -> None:
        var_view = self.in_variable_view()
        self.setWindowTitle("Find and Replace - " + ("Variable View" if var_view
                                                     else "Data View"))
        self.field.setEnabled(var_view)
        col = self.editor.data_view.currentIndex().column()
        if 0 <= col < self.ds.n_vars:
            self.column_label.setText(self.ds.variables[col].name)
        else:
            self.column_label.setText("(none selected)")
            self.all_columns.setChecked(True)

    def _columns(self) -> list[int] | None:
        if self.all_columns.isChecked():
            return None
        col = self.editor.data_view.currentIndex().column()
        return [col] if 0 <= col < self.ds.n_vars else None

    def _hits(self) -> list[tuple[int, int]]:
        return dp.find_cells(self.ds, self.find_edit.text(), self._columns(),
                             self.match_case.isChecked(), self.mode.currentData(),
                             self.editor.data_model.show_labels)

    # ------------------------------------------------------------------ actions
    def find_next(self, backwards: bool = False) -> bool:
        self.refresh_scope()
        needle = self.find_edit.text()
        if self.in_variable_view():
            rows = dp.find_variables(self.ds, needle, self.field.currentData(),
                                     self.match_case.isChecked(), self.mode.currentData())
            if not rows:
                self.status.setText(f"“{needle}” was not found.")
                return False
            current = self.editor.var_view.currentIndex().row()
            later = [r for r in rows if (r < current if backwards else r > current)]
            row = (later[-1] if backwards else later[0]) if later else \
                (rows[-1] if backwards else rows[0])
            self.editor.var_view.setCurrentIndex(self.editor.var_model.index(row, 0))
            self.status.setText(f"{len(rows)} variable(s) match.")
            return True
        hits = self._hits()
        idx = self.editor.data_view.currentIndex()
        here = (idx.row(), idx.column()) if idx.isValid() else None
        nxt = dp.next_match(hits, here, backwards)
        if nxt is None:
            self.status.setText(f"“{needle}” was not found.")
            self.current = None
            return False
        self.current = nxt
        self.editor.select_cell(*nxt)
        self.status.setText(f"Match {hits.index(nxt) + 1} of {len(hits)}.")
        return True

    def replace_one(self) -> None:
        if self.in_variable_view():
            self.status.setText("Replace works in Data View.")
            return
        hits = self._hits()
        if self.current not in hits:
            self.find_next()
            return
        r, c = self.current
        new = dp.replace_cell_text(self.ds, r, c, self.find_edit.text(),
                                   self.replace_edit.text(), self.mode.currentData(),
                                   self.match_case.isChecked())
        try:
            self.editor.doc.set_value(r, c, new)
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.find_next()

    def replace_everything(self) -> None:
        if self.in_variable_view():
            self.status.setText("Replace works in Data View.")
            return
        needle, new = self.find_edit.text(), self.replace_edit.text()
        cols, case, mode = self._columns(), self.match_case.isChecked(), self.mode.currentData()
        try:
            n = self.editor.doc.mutate(lambda ds: dp.replace_all(ds, needle, new, cols, case,
                                                                 mode), "Replace All")
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.status.setText(f"{n} value(s) replaced.")


def show_find(editor, replace: bool = False) -> FindReplaceDialog:
    dlg = _OPEN.get(id(editor))
    if dlg is None:
        dlg = FindReplaceDialog(editor, replace)
        _OPEN[id(editor)] = dlg
        editor.destroyed.connect(lambda *_: _OPEN.pop(id(editor), None))
    dlg.do_replace.setChecked(replace or dlg.do_replace.isChecked())
    sel = editor.data_view.currentIndex()
    if sel.isValid() and sel.row() < editor.ds.n_cases and sel.column() < editor.ds.n_vars \
            and not dlg.find_edit.text():
        dlg.find_edit.setText(editor.ds.display_value(sel.row(), sel.column(),
                                                      editor.data_model.show_labels))
    dlg.refresh_scope()
    dlg.show()
    dlg.raise_()
    dlg.activateWindow()
    dlg.find_edit.setFocus()
    dlg.find_edit.selectAll()
    return dlg

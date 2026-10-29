"""Transform > Compute Variable, Recode into Same Variables, Recode into Different Variables."""
from __future__ import annotations

import copy

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QDialog,
                               QDialogButtonBox, QFormLayout, QGridLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QMessageBox, QPlainTextEdit, QPushButton, QRadioButton,
                               QSpinBox, QVBoxLayout, QWidget)

from rsstatchem.core import expression as expr
from rsstatchem.core import transform
from rsstatchem.core.dataset import NUMERIC, STRING, Dataset, format_code, is_valid_name
from rsstatchem.core.transform import Rule
from rsstatchem.ui.dialogs.base import NAME_ROLE, VariablePickerDialog, variable_item

KEYPAD = [["+", "<", ">", "7", "8", "9"],
          ["-", "<=", ">=", "4", "5", "6"],
          ["*", "=", "~=", "1", "2", "3"],
          ["/", "&", "|", "0", ".", ""],
          ["**", "~", "()", "Delete", "", ""]]


def _template(name: str) -> str:
    """'MEAN(numexpr, numexpr[, …])…' → 'MEAN(?,?)' as SPSS pastes it."""
    desc = expr.FUNCTIONS[name][2]
    inside = desc.split("(", 1)[1].split(")", 1)[0]
    required = [p for p in inside.split("[")[0].split(",") if p.strip()]
    return f"{name}({','.join('?' for _ in required) or '?'})"


class ExpressionBuilder(QWidget):
    """Variable list + keypad + function browser around an expression editor."""

    def __init__(self, ds: Dataset, caption: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.ds = ds
        self.variables = QListWidget()
        self.variables.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        for var in ds.variables:
            self.variables.addItem(variable_item(var))
        self.variables.itemDoubleClicked.connect(lambda it: self.insert(it.data(NAME_ROLE)))
        arrow = QPushButton("▶")
        arrow.setObjectName("arrow")
        arrow.clicked.connect(self._insert_selected_variable)

        self.editor = QPlainTextEdit()
        mono = QFont()
        mono.setFamilies(["Menlo", "Consolas", "Courier New"])
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(11)
        self.editor.setFont(mono)
        self.editor.setMinimumHeight(70)
        self.editor.setPlaceholderText("e.g.  MEAN(X1 TO X6)   or   (pretest + posttest) / 2")
        self.status = QLabel()
        self.status.setWordWrap(True)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self.validate)
        self.editor.textChanged.connect(self._timer.start)
        self.on_validated = None

        pad = QGridLayout()
        pad.setSpacing(3)
        for r, row in enumerate(KEYPAD):
            for c, key in enumerate(row):
                if not key:
                    continue
                b = QPushButton(key.replace("&", "&&"))  # a lone & is a Qt mnemonic marker
                b.setMinimumWidth(34)
                b.setStyleSheet("QPushButton { min-width: 34px; padding: 3px 4px; }")
                b.clicked.connect(lambda _=False, k=key: self._key(k))
                span = 2 if key == "Delete" else 1
                pad.addWidget(b, r, c, 1, span)

        self.groups = QListWidget()
        self.groups.addItems(expr.FUNCTION_GROUPS)
        self.groups.setMaximumHeight(130)
        self.groups.currentTextChanged.connect(self._fill_functions)
        self.functions = QListWidget()
        self.functions.currentTextChanged.connect(self._describe)
        self.functions.itemDoubleClicked.connect(lambda it: self.insert_function(it.text()))
        self.description = QLabel()
        self.description.setWordWrap(True)
        self.description.setStyleSheet("color:#444; background:#fafafa; padding:4px;")
        self.description.setMinimumHeight(44)
        up = QPushButton("▲")
        up.setObjectName("arrow")
        up.setToolTip("Insert the function into the expression")
        up.clicked.connect(lambda: self.functions.currentItem()
                           and self.insert_function(self.functions.currentItem().text()))

        middle = QVBoxLayout()
        middle.addWidget(QLabel(caption))
        middle.addWidget(self.editor, 1)
        lower = QHBoxLayout()
        lower.addLayout(pad)
        lower.addWidget(up, 0, Qt.AlignmentFlag.AlignTop)
        middle.addLayout(lower)
        middle.addWidget(self.description)
        middle.addWidget(self.status)

        right = QVBoxLayout()
        right.addWidget(QLabel("Function group:"))
        right.addWidget(self.groups)
        right.addWidget(QLabel("Functions:"))
        right.addWidget(self.functions, 1)

        arrow_col = QVBoxLayout()
        arrow_col.addSpacing(26)
        arrow_col.addWidget(arrow)
        arrow_col.addStretch()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.variables, 2)
        layout.addLayout(arrow_col)
        layout.addLayout(middle, 4)
        layout.addLayout(right, 2)
        self.groups.setCurrentRow(0)

    # ------------------------------------------------------------------ editing
    def text(self) -> str:
        return self.editor.toPlainText().strip()

    def set_text(self, text: str) -> None:
        self.editor.setPlainText(text)

    def insert(self, text: str) -> None:
        cursor = self.editor.textCursor()
        before = self.editor.toPlainText()[:cursor.position()] if not cursor.hasSelection() \
            else ""
        needs_space = before and before[-1].isalnum() and text[:1].isalnum()
        cursor.insertText((" " if needs_space else "") + text)
        self.editor.setTextCursor(cursor)
        self.editor.setFocus()

    def insert_function(self, name: str) -> None:
        cursor = self.editor.textCursor()
        start = cursor.position() if not cursor.hasSelection() else cursor.selectionStart()
        self.insert(_template(name))
        text = self.editor.toPlainText()
        q = text.find("?", start)
        if q >= 0:  # select the first placeholder so the next variable replaces it
            cursor = self.editor.textCursor()
            cursor.setPosition(q)
            cursor.setPosition(q + 1, QTextCursor.MoveMode.KeepAnchor)
            self.editor.setTextCursor(cursor)

    def _insert_selected_variable(self) -> None:
        item = self.variables.currentItem()
        if item is not None:
            self.insert(item.data(NAME_ROLE))

    def _key(self, key: str) -> None:
        if key == "Delete":
            cursor = self.editor.textCursor()
            if not cursor.hasSelection():
                cursor.movePosition(QTextCursor.MoveOperation.Left,
                                    QTextCursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()
            return
        if key == "()":
            cursor = self.editor.textCursor()
            selected = cursor.selectedText()
            cursor.insertText(f"({selected})")
            if not selected:
                cursor.movePosition(QTextCursor.MoveOperation.Left)
            self.editor.setTextCursor(cursor)
            return
        spaced = key if key[0].isdigit() or key == "." else f" {key} "
        self.editor.insertPlainText(spaced)
        self.editor.setFocus()

    def _fill_functions(self, group: str) -> None:
        self.functions.clear()
        for name, (_, g, _) in expr.FUNCTIONS.items():
            if group in ("All", g):
                self.functions.addItem(name)

    def _describe(self, name: str) -> None:
        self.description.setText(expr.FUNCTIONS[name][2] if name in expr.FUNCTIONS else "")

    def validate(self) -> bool:
        text = self.text()
        if not text:
            self.status.setText("")
            ok = False
        else:
            error = expr.check(self.ds, text)
            ok = error is None
            if ok:
                kind = "string" if self.ds.n_cases and expr.is_string_result(
                    expr.evaluate(self.ds, text)) else "numeric"
                self.status.setStyleSheet("color:#1b7f3b;")
                self.status.setText(f"✓ Valid {kind} expression")
            else:
                self.status.setStyleSheet("color:#B71C1C;")
                self.status.setText(f"✗ {error}")
        if self.on_validated:
            self.on_validated(ok)
        return ok


class IfDialog(QDialog):
    def __init__(self, ds: Dataset, condition: str, parent: QWidget | None = None, title="If"):
        super().__init__(parent)
        self.setWindowTitle(f"{title}: If Cases")
        self.resize(900, 420)
        self.all_cases = QRadioButton("Include all cases")
        self.some = QRadioButton("Include if case satisfies condition:")
        group = QButtonGroup(self)
        group.addButton(self.all_cases)
        group.addButton(self.some)
        self.builder = ExpressionBuilder(ds, "Condition:")
        self.builder.set_text(condition)
        (self.some if condition else self.all_cases).setChecked(True)
        self.builder.setEnabled(bool(condition))
        self.some.toggled.connect(self.builder.setEnabled)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Continue")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(self.all_cases)
        layout.addWidget(self.some)
        layout.addWidget(self.builder, 1)
        layout.addWidget(buttons)

    def _accept(self) -> None:
        if self.some.isChecked():
            if not self.builder.validate():
                QMessageBox.warning(self, self.windowTitle(),
                                    self.builder.status.text() or "Enter a condition.")
                return
            if expr.is_string_result(expr.evaluate(self.builder.ds, self.builder.text())):
                QMessageBox.warning(self, self.windowTitle(),
                                    "The condition must be logical, e.g. kelompok = 1.")
                return
        self.accept()

    def condition(self) -> str:
        return self.builder.text() if self.some.isChecked() else ""


def _if_label(condition: str) -> str:
    return f"If: {condition}" if condition else "(optional case selection condition)"


# ------------------------------------------------------------------ compute
class ComputeDialog(QDialog):
    modifies_data = True
    _last: dict = {"target": "", "expression": "", "condition": "", "type": None, "label": "",
                   "width": 8}

    def __init__(self, ds: Dataset, parent: QWidget | None = None):
        super().__init__(parent)
        self.ds = ds
        self.state = copy.deepcopy(self._last)
        self.setWindowTitle("Compute Variable")
        self.resize(980, 520)

        self.target = QLineEdit(self.state["target"])
        self.target.setPlaceholderText("new or existing variable")
        type_button = QPushButton("Type && Label...")
        type_button.clicked.connect(self._type_label)
        top = QHBoxLayout()
        top.addWidget(QLabel("Target Variable:"))
        top.addWidget(self.target, 1)
        top.addWidget(type_button)
        top.addWidget(QLabel("  ="))
        top.addStretch(2)

        self.builder = ExpressionBuilder(ds, "Numeric Expression:")
        self.builder.set_text(self.state["expression"])
        self.builder.on_validated = lambda _: self._update_ok()
        self.target.textChanged.connect(self._update_ok)

        self.if_button = QPushButton("If...")
        self.if_button.clicked.connect(self._if)
        self.if_label = QLabel(_if_label(self.state["condition"]))
        self.if_label.setStyleSheet("color:#555;")
        bottom = QHBoxLayout()
        bottom.addWidget(self.if_button)
        bottom.addWidget(self.if_label, 1)

        buttons = QDialogButtonBox()
        self.ok = buttons.addButton(QDialogButtonBox.StandardButton.Ok)
        reset = buttons.addButton(QDialogButtonBox.StandardButton.Reset)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        self.ok.setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        reset.clicked.connect(self._reset)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.builder, 1)
        layout.addLayout(bottom)
        layout.addWidget(buttons)
        self.builder.validate()
        self._update_ok()
        from rsstatchem.ui.dialogs.base import keep_ok_default

        keep_ok_default(self, self.ok)

    def _update_ok(self, *_):
        name = self.target.text().strip()
        valid_name = is_valid_name(name)
        self.target.setStyleSheet("" if valid_name or not name else "border:1px solid #B71C1C;")
        self.ok.setEnabled(valid_name and expr.check(self.ds, self.builder.text()) is None
                           if self.builder.text() else False)

    def _type_label(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Compute Variable: Type and Label")
        label_box = QGroupBox("Label")
        use_label = QRadioButton("Label:")
        use_expr = QRadioButton("Use expression as label")
        label_edit = QLineEdit(self.state["label"])
        use_label.setChecked(True)
        lf = QFormLayout(label_box)
        lf.addRow(use_label, label_edit)
        lf.addRow(use_expr)
        type_box = QGroupBox("Type")
        numeric = QRadioButton("Numeric")
        string = QRadioButton("String")
        width = QSpinBox()
        width.setRange(1, 32767)
        width.setValue(self.state["width"])
        (string if self.state["type"] == STRING else numeric).setChecked(True)
        tf = QFormLayout(type_box)
        tf.addRow(numeric)
        tf.addRow(string, width)
        if self.ds.has_name(self.target.text().strip()):
            type_box.setEnabled(False)
            type_box.setTitle("Type (fixed: existing variable)")
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Continue")
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout = QVBoxLayout(dlg)
        layout.addWidget(label_box)
        layout.addWidget(type_box)
        layout.addWidget(buttons)
        if dlg.exec():
            self.state["label"] = self.builder.text() if use_expr.isChecked() \
                else label_edit.text()
            self.state["type"] = STRING if string.isChecked() else NUMERIC
            self.state["width"] = width.value()

    def _if(self) -> None:
        dlg = IfDialog(self.ds, self.state["condition"], self, "Compute Variable")
        if dlg.exec():
            self.state["condition"] = dlg.condition()
            self.if_label.setText(_if_label(self.state["condition"]))

    def _reset(self) -> None:
        self.target.clear()
        self.builder.set_text("")
        self.state = {"target": "", "expression": "", "condition": "", "type": None,
                      "label": "", "width": 8}
        self.if_label.setText(_if_label(""))

    def accept(self) -> None:
        name = self.target.text().strip()
        if self.ds.has_name(name):
            answer = QMessageBox.question(self, "Compute Variable",
                                          f"Change existing variable {self.ds.var(name).name}?")
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.state.update(target=name, expression=self.builder.text())
        type(self)._last = copy.deepcopy(self.state)
        super().accept()

    def compute(self):
        s = self.state
        vtype = None if self.ds.has_name(s["target"]) else s["type"]
        return transform.compute(self.ds, s["target"], s["expression"], vtype,
                                 s["label"] or None,
                                 s["width"] if vtype == STRING else None, s["condition"] or None)


# ------------------------------------------------------------------ recode
class OldNewDialog(QDialog):
    def __init__(self, numeric_source: bool, into_different: bool, rules: list[Rule],
                 options: dict, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Recode: Old and New Values")
        self.numeric_source = numeric_source
        self.into_different = into_different
        self.rules = list(rules)
        self.options = dict(options)

        old_box = QGroupBox("Old Value")
        og = QGridLayout(old_box)
        self.old_group = QButtonGroup(self)
        self.o_value, self.o_value_edit = QRadioButton("Value:"), QLineEdit()
        self.o_sysmis = QRadioButton("System-missing")
        self.o_missing = QRadioButton("System- or user-missing")
        self.o_range = QRadioButton("Range:")
        self.o_lo, self.o_hi = QLineEdit(), QLineEdit()
        self.o_lowest, self.o_lowest_edit = QRadioButton("Range, LOWEST through value:"), \
            QLineEdit()
        self.o_highest, self.o_highest_edit = QRadioButton("Range, value through HIGHEST:"), \
            QLineEdit()
        self.o_else = QRadioButton("All other values")
        range_row = QWidget()
        rr = QHBoxLayout(range_row)
        rr.setContentsMargins(22, 0, 0, 0)
        rr.addWidget(self.o_lo)
        rr.addWidget(QLabel("through"))
        rr.addWidget(self.o_hi)
        rows = [(self.o_value, self.o_value_edit), (self.o_sysmis, None),
                (self.o_missing, None), (self.o_range, None), (None, range_row),
                (self.o_lowest, None), (None, self.o_lowest_edit),
                (self.o_highest, None), (None, self.o_highest_edit), (self.o_else, None)]
        for r, (radio, field) in enumerate(rows):
            if radio is not None:
                self.old_group.addButton(radio)
                og.addWidget(radio, r, 0, 1, 2 if field is None else 1)
            if field is not None:
                if radio is None and field is not range_row:
                    field.setContentsMargins(22, 0, 0, 0)
                og.addWidget(field, r, 1 if radio is not None else 0, 1, 1 if radio else 2)
        og.setRowStretch(len(rows), 1)
        self.o_value.setChecked(True)
        if not numeric_source:
            for w in (self.o_sysmis, self.o_range, self.o_lo, self.o_hi, self.o_lowest,
                      self.o_lowest_edit, self.o_highest, self.o_highest_edit):
                w.setEnabled(False)

        new_box = QGroupBox("New Value")
        ng = QGridLayout(new_box)
        self.new_group = QButtonGroup(self)
        self.n_value, self.n_value_edit = QRadioButton("Value:"), QLineEdit()
        self.n_sysmis = QRadioButton("System-missing")
        self.n_copy = QRadioButton("Copy old value(s)")
        for r, (radio, field) in enumerate([(self.n_value, self.n_value_edit),
                                            (self.n_sysmis, None), (self.n_copy, None)]):
            self.new_group.addButton(radio)
            ng.addWidget(radio, r, 0)
            if field is not None:
                ng.addWidget(field, r, 1)
        self.n_value.setChecked(True)
        self.n_copy.setVisible(into_different)

        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._pick)
        add, change, remove = QPushButton("Add"), QPushButton("Change"), QPushButton("Remove")
        add.clicked.connect(self._add)
        change.clicked.connect(self._change)
        remove.clicked.connect(self._remove)
        buttons_col = QVBoxLayout()
        for b in (add, change, remove):
            buttons_col.addWidget(b)
        buttons_col.addStretch()
        list_row = QHBoxLayout()
        list_row.addLayout(buttons_col)
        list_row.addWidget(self.list, 1)
        new_side = QVBoxLayout()
        new_side.addWidget(new_box)
        new_side.addWidget(QLabel("Old --> New:"))
        new_side.addLayout(list_row, 1)

        self.as_string = QCheckBox("Output variables are strings")
        self.width = QSpinBox()
        self.width.setRange(1, 32767)
        self.width.setValue(self.options.get("width", 8))
        self.as_string.setChecked(self.options.get("output_string", False))
        self.convert = QCheckBox("Convert numeric strings to numbers ('5'->5)")
        self.convert.setChecked(self.options.get("convert", False))
        self.convert.setEnabled(not numeric_source)
        if into_different:
            opt = QHBoxLayout()
            opt.addWidget(self.as_string)
            opt.addWidget(QLabel("Width:"))
            opt.addWidget(self.width)
            opt.addStretch()
            new_side.addLayout(opt)
            new_side.addWidget(self.convert)

        self.error = QLabel()
        self.error.setStyleSheet("color:#B71C1C;")
        dialog_buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                          | QDialogButtonBox.StandardButton.Cancel)
        dialog_buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Continue")
        dialog_buttons.accepted.connect(self._accept)
        dialog_buttons.rejected.connect(self.reject)

        body = QHBoxLayout()
        body.addWidget(old_box, 1)
        body.addLayout(new_side, 1)
        layout = QVBoxLayout(self)
        layout.addLayout(body, 1)
        layout.addWidget(self.error)
        layout.addWidget(dialog_buttons)
        for edit, radio in ((self.o_value_edit, self.o_value), (self.o_lo, self.o_range),
                            (self.o_hi, self.o_range), (self.o_lowest_edit, self.o_lowest),
                            (self.o_highest_edit, self.o_highest),
                            (self.n_value_edit, self.n_value)):
            edit.textEdited.connect(lambda _, r=radio: r.setChecked(True))
        self._refresh()

    def _output_numeric(self) -> bool:
        if not self.into_different:
            return self.numeric_source
        return not self.as_string.isChecked()

    def _parse(self, text: str, numeric: bool):
        text = text.strip()
        if not numeric:
            if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
                text = text[1:-1]
            return text
        if not text:
            raise ValueError("Enter a value.")
        return float(text.replace(",", "."))

    def _current_rule(self) -> Rule:
        n = self.numeric_source
        if self.o_value.isChecked():
            old = ("value", self._parse(self.o_value_edit.text(), n))
        elif self.o_sysmis.isChecked():
            old = ("sysmis",)
        elif self.o_missing.isChecked():
            old = ("missing",)
        elif self.o_range.isChecked():
            lo, hi = self._parse(self.o_lo.text(), True), self._parse(self.o_hi.text(), True)
            if lo > hi:
                raise ValueError("The lower range value is larger than the upper one.")
            old = ("range", lo, hi)
        elif self.o_lowest.isChecked():
            old = ("lowest", self._parse(self.o_lowest_edit.text(), True))
        elif self.o_highest.isChecked():
            old = ("highest", self._parse(self.o_highest_edit.text(), True))
        else:
            old = ("else",)
        if self.n_value.isChecked():
            new = ("value", self._parse(self.n_value_edit.text(), self._output_numeric()))
        elif self.n_sysmis.isChecked():
            new = ("sysmis",)
        else:
            new = ("copy",)
        return Rule(old, new)

    def _ordered(self) -> None:
        """SPSS keeps ELSE last so that it only catches values not mentioned before."""
        self.rules = [r for r in self.rules if r.old[0] != "else"] + \
            [r for r in self.rules if r.old[0] == "else"]

    def _add(self) -> None:
        try:
            rule = self._current_rule()
        except ValueError as exc:
            self.error.setText(str(exc) if str(exc) else "Values must be numbers.")
            return
        self.error.clear()
        self.rules = [r for r in self.rules if r.old != rule.old] + [rule]
        self._ordered()
        self._refresh()
        self.o_value_edit.clear()
        self.n_value_edit.clear()
        self.o_value_edit.setFocus()

    def _change(self) -> None:
        row = self.list.currentRow()
        if row < 0:
            return
        try:
            rule = self._current_rule()
        except ValueError as exc:
            self.error.setText(str(exc) if str(exc) else "Values must be numbers.")
            return
        self.rules[row] = rule
        self._ordered()
        self._refresh()

    def _remove(self) -> None:
        row = self.list.currentRow()
        if row >= 0:
            del self.rules[row]
            self._refresh()

    def _refresh(self) -> None:
        self.list.clear()
        for r in self.rules:
            self.list.addItem(QListWidgetItem(f"{r.old_text()} --> {r.new_text()}"))

    def _pick(self, row: int) -> None:
        if row < 0 or row >= len(self.rules):
            return
        rule = self.rules[row]
        kind = rule.old[0]
        widgets = {"value": self.o_value, "sysmis": self.o_sysmis, "missing": self.o_missing,
                   "range": self.o_range, "lowest": self.o_lowest, "highest": self.o_highest,
                   "else": self.o_else}
        widgets[kind].setChecked(True)
        if kind == "value":
            self.o_value_edit.setText(format_code(rule.old[1]))
        elif kind == "range":
            self.o_lo.setText(format_code(rule.old[1]))
            self.o_hi.setText(format_code(rule.old[2]))
        elif kind == "lowest":
            self.o_lowest_edit.setText(format_code(rule.old[1]))
        elif kind == "highest":
            self.o_highest_edit.setText(format_code(rule.old[1]))
        {"value": self.n_value, "sysmis": self.n_sysmis, "copy": self.n_copy}[
            rule.new[0]].setChecked(True)
        if rule.new[0] == "value":
            self.n_value_edit.setText(format_code(rule.new[1]))

    def _accept(self) -> None:
        if not self.rules and not (self.convert.isChecked() and self.convert.isEnabled()):
            self.error.setText("Add at least one Old --> New pair.")
            return
        self.options.update(output_string=self.as_string.isChecked() and self.into_different,
                            width=self.width.value(),
                            convert=self.convert.isChecked() and self.convert.isEnabled())
        self.accept()


class _RecodeBase(VariablePickerDialog):
    modifies_data = True
    DEFAULTS = {"rules": [], "condition": "", "output_string": False, "width": 8,
                "convert": False, "outputs": {}}
    INTO_DIFFERENT = False

    def _common_buttons(self) -> None:
        self.add_option_button("Old and New Values...", self._old_new)
        self.add_option_button("If...", self._if)
        self.if_label = QLabel(_if_label(self.opts["condition"]))
        self.if_label.setStyleSheet("color:#555;")
        self.options_layout.addWidget(self.if_label)
        self.rules_label = QLabel()
        self.rules_label.setStyleSheet("color:#555;")
        self.rules_label.setWordWrap(True)
        self.options_layout.addWidget(self.rules_label)
        self._show_rules()

    def _show_rules(self) -> None:
        rules = self.opts["rules"]
        self.rules_label.setText("Old --> New: " + ", ".join(
            f"{r.old_text()} → {r.new_text()}" for r in rules) if rules
            else "Old --> New: (none yet — click Old and New Values…)")

    def _source_numeric(self) -> bool | None:
        names = self.selected("vars")
        if not names:
            return None
        return self.ds.var(names[0]).type == NUMERIC

    def _old_new(self) -> None:
        numeric = self._source_numeric()
        if numeric is None:
            QMessageBox.information(self, self.windowTitle(), "Select the variables first.")
            return
        dlg = OldNewDialog(numeric, self.INTO_DIFFERENT, self.opts["rules"], self.opts, self)
        if dlg.exec():
            self.opts["rules"] = dlg.rules
            self.opts.update({k: dlg.options[k] for k in ("output_string", "width", "convert")})
            self._show_rules()

    def _if(self) -> None:
        dlg = IfDialog(self.ds, self.opts["condition"], self, self.windowTitle())
        if dlg.exec():
            self.opts["condition"] = dlg.condition()
            self.if_label.setText(_if_label(self.opts["condition"]))

    def validate(self):
        names = self.selected("vars")
        kinds = {self.ds.var(n).type == NUMERIC for n in names}
        if len(kinds) > 1:
            return "Numeric and string variables cannot be recoded together."
        if not self.opts["rules"] and not self.opts["convert"]:
            return "Click Old and New Values… and add at least one Old --> New pair."
        numeric = kinds.pop() if kinds else True
        for r in self.opts["rules"]:
            if r.old[0] == "value" and isinstance(r.old[1], str) == numeric:
                return "The old values do not match the variable type. Re-enter them."
        return None


class RecodeSameDialog(_RecodeBase):
    def __init__(self, ds, parent=None):
        super().__init__(ds, "Recode into Same Variables", parent)
        self.add_target("vars", "Variables:")
        self._common_buttons()
        self.finish()

    def compute(self):
        o = self.opts
        return transform.recode(self.ds, self.selected("vars"), o["rules"],
                                condition=o["condition"] or None)


class RecodeDifferentDialog(_RecodeBase):
    INTO_DIFFERENT = True

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Recode into Different Variables", parent)
        self.var_list = self.add_target("vars", "Input Variable --> Output Variable:")
        self.var_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.var_list.currentItemChanged.connect(lambda *_: self._load_output())
        box = QGroupBox("Output Variable")
        form = QFormLayout(box)
        self.out_name = QLineEdit()
        self.out_label = QLineEdit()
        change = QPushButton("Change")
        change.clicked.connect(self._change_output)
        form.addRow("Name:", self.out_name)
        form.addRow("Label:", self.out_label)
        form.addRow("", change)
        self._option_column.addWidget(box)
        self._common_buttons()
        self.finish()

    def _outputs(self) -> dict:
        return self.opts.setdefault("outputs", {})

    def targets_changed(self) -> None:
        if not hasattr(self, "var_list"):
            return
        outputs = self._outputs()
        for i in range(self.var_list.count()):
            item = self.var_list.item(i)
            src = item.data(NAME_ROLE)
            name = outputs.get(src, ("", ""))[0]
            item.setText(f"{src} --> {name or '?'}")

    def _load_output(self) -> None:
        item = self.var_list.currentItem()
        if item is None:
            return
        name, label = self._outputs().get(item.data(NAME_ROLE), ("", ""))
        self.out_name.setText(name)
        self.out_label.setText(label)

    def _change_output(self) -> None:
        item = self.var_list.currentItem() or (self.var_list.item(0)
                                               if self.var_list.count() == 1 else None)
        if item is None:
            QMessageBox.information(self, self.windowTitle(),
                                    "Select an input variable in the list first.")
            return
        name = self.out_name.text().strip()
        if not is_valid_name(name):
            QMessageBox.warning(self, self.windowTitle(), f"'{name}' is not a valid name.")
            return
        self._outputs()[item.data(NAME_ROLE)] = (name, self.out_label.text().strip())
        self.targets_changed()

    def validate(self):
        if self.out_name.text().strip() and self.var_list.count() == 1 and \
                not self._outputs().get(self.selected("vars")[0], ("",))[0]:
            self._change_output()  # Name typed but Change not clicked — SPSS would warn
        missing = [s for s in self.selected("vars") if not self._outputs().get(s, ("",))[0]]
        if missing:
            return ("Give each input variable an output name (type a Name and click Change): "
                    + ", ".join(missing))
        return super().validate()

    def compute(self):
        o = self.opts
        sources = self.selected("vars")
        outputs = self._outputs()
        return transform.recode(self.ds, sources, o["rules"],
                                targets=[outputs[s][0] for s in sources],
                                labels=[outputs[s][1] for s in sources],
                                output_string=o["output_string"], width=o["width"],
                                convert=o["convert"], condition=o["condition"] or None)

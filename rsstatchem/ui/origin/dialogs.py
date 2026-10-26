"""Dialogs of the Graph & Analysis workspace (Origin's Set Values, Column Properties, fitting
dialogs and small parameter dialogs). Each dialog only collects settings; the main window
applies them through the Qt-free functions, so everything is testable without a modal loop.
"""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QDoubleSpinBox, QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMenu, QPlainTextEdit, QPushButton,
                               QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from rsstatchem.core.analysis import functions as fl
from rsstatchem.core.analysis.fitting import WEIGHTINGS
from rsstatchem.ui.origin import formula as fx
from rsstatchem.ui.origin.model import DESIGNATIONS, RECALC_MODES, Worksheet
from rsstatchem.ui.origin.plotting import PLOT_TYPES


def _buttons(dlg: QDialog, ok_text: str = "OK") -> QDialogButtonBox:
    box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                           | QDialogButtonBox.StandardButton.Cancel)
    box.button(QDialogButtonBox.StandardButton.Ok).setText(ok_text)
    box.accepted.connect(dlg.accept)
    box.rejected.connect(dlg.reject)
    return box


# ====================================================================== column properties
class ColumnPropertiesDialog(QDialog):
    def __init__(self, ws: Worksheet, col: int, parent=None):
        super().__init__(parent)
        c = ws.columns[col]
        self.setWindowTitle(f"Column Properties - {ws.label(col)}")
        self.short = QLineEdit(c.name)
        self.long = QLineEdit(c.long_name)
        self.units = QLineEdit(c.units)
        self.comments = QPlainTextEdit(c.comments)
        self.comments.setFixedHeight(60)
        self.designation = QComboBox()
        for code, (text, _) in DESIGNATIONS.items():
            self.designation.addItem(text, code)
        self.designation.setCurrentIndex(list(DESIGNATIONS).index(c.designation))
        self.width = QSpinBox()
        self.width.setRange(30, 600)
        self.width.setValue(c.width or 84)
        self.formula = QLineEdit(c.formula)
        form = QFormLayout()
        form.addRow("Short Name:", self.short)
        form.addRow("Long Name:", self.long)
        form.addRow("Units:", self.units)
        form.addRow("Comments:", self.comments)
        form.addRow("Plot Designation:", self.designation)
        form.addRow("Column Width:", self.width)
        form.addRow("F(x)=:", self.formula)
        box = QGroupBox("Properties")
        box.setLayout(form)
        layout = QVBoxLayout(self)
        layout.addWidget(box)
        layout.addWidget(_buttons(self))
        self.resize(420, 0)

    def values(self) -> dict:
        return {"name": self.short.text().strip(), "long_name": self.long.text().strip(),
                "units": self.units.text().strip(),
                "comments": self.comments.toPlainText().strip(),
                "designation": self.designation.currentData(), "width": self.width.value(),
                "formula": self.formula.text().strip()}


# ====================================================================== set column values
class SetValuesDialog(QDialog):
    """Origin's Set Values dialog: Col(B) = <formula>, with function / column menus, row
    range, Apply and OK. ``apply(col, formula, first, last)`` raises ValueError on errors."""

    def __init__(self, ws: Worksheet, col: int, apply, parent=None):
        super().__init__(parent)
        self.ws, self.col, self._apply = ws, col, apply
        self.setWindowTitle(f"Set Values - [{ws.name}]Col({ws.columns[col].name})")
        self.target = QLabel()
        self.target.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.edit = QPlainTextEdit(ws.columns[col].formula)
        self.edit.setFont(QFont("Consolas", 10))
        self.edit.setFixedHeight(90)
        self.first = QLineEdit()
        self.first.setPlaceholderText("<auto>")
        self.last = QLineEdit()
        self.last.setPlaceholderText("<auto>")
        for e in (self.first, self.last):
            e.setFixedWidth(70)
        self.error = QLabel("")
        self.error.setStyleSheet("color:#b00000")
        self.error.setWordWrap(True)
        prev_b = QPushButton("<<")
        next_b = QPushButton(">>")
        prev_b.setObjectName("arrow")
        next_b.setObjectName("arrow")
        prev_b.clicked.connect(lambda: self._move(-1))
        next_b.clicked.connect(lambda: self._move(1))
        func_b = QPushButton("Function")
        func_menu = QMenu(func_b)
        for name in fx.FUNCTION_HELP.split():
            func_menu.addAction(name, lambda n=name: self._insert(n if n.endswith(")") or
                                                                    n == "pi" else n + "()"))
        func_b.setMenu(func_menu)
        col_b = QPushButton("Col(A)")
        col_menu = QMenu(col_b)
        for c in ws.columns:
            col_menu.addAction(f"col({c.name})" + (f"  {c.long_name}" if c.long_name else ""),
                               lambda n=c.name: self._insert(f"col({n})"))
        col_menu.addAction("i  (row number)", lambda: self._insert("i"))
        col_b.setMenu(col_menu)
        ex_b = QPushButton("Examples")
        ex_menu = QMenu(ex_b)
        for text, desc in fx.EXAMPLES:
            ex_menu.addAction(f"{text}    — {desc}", lambda t=text: self.edit.setPlainText(t))
        ex_b.setMenu(ex_menu)
        for b in (func_b, col_b, ex_b):
            b.setObjectName("menuButton")
        top = QHBoxLayout()
        top.addWidget(QLabel("Row (i):  From"))
        top.addWidget(self.first)
        top.addWidget(QLabel("To"))
        top.addWidget(self.last)
        top.addStretch(1)
        top.addWidget(func_b)
        top.addWidget(col_b)
        top.addWidget(ex_b)
        nav = QHBoxLayout()
        nav.addWidget(prev_b)
        nav.addWidget(self.target, 1)
        nav.addWidget(next_b)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Apply
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._ok)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(nav)
        layout.addWidget(self.edit)
        layout.addWidget(self.error)
        layout.addWidget(buttons)
        self._update_target()
        self.resize(560, 0)

    def _update_target(self) -> None:
        c = self.ws.columns[self.col]
        self.target.setText(f"Col({c.name}) =" + (f"   [{c.long_name}]" if c.long_name else ""))
        self.setWindowTitle(f"Set Values - [{self.ws.name}]Col({c.name})")

    def _move(self, step: int) -> None:
        self.col = max(0, min(len(self.ws.columns) - 1, self.col + step))
        self.edit.setPlainText(self.ws.columns[self.col].formula)
        self._update_target()

    def _insert(self, text: str) -> None:
        self.edit.insertPlainText(text)
        self.edit.setFocus()

    def _rows(self) -> tuple[int | None, int | None]:
        def parse(e: QLineEdit):
            t = e.text().strip()
            return int(t) if t.isdigit() else None
        return parse(self.first), parse(self.last)

    def apply(self) -> bool:
        first, last = self._rows()
        try:
            self._apply(self.col, self.edit.toPlainText().strip(), first, last)
        except ValueError as exc:
            self.error.setText(str(exc))
            return False
        self.error.setText("")
        return True

    def _ok(self) -> None:
        if self.apply():
            self.accept()


# ====================================================================== generic parameters
class ParamDialog(QDialog):
    """Small form: fields are (key, label, kind, default, extra) with kind in
    int / float / choice (extra = {value: text}) / bool / text."""

    def __init__(self, title: str, fields: list[tuple], parent=None, note: str = ""):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.widgets: dict[str, QWidget] = {}
        form = QFormLayout()
        for key, label, kind, default, *extra in fields:
            extra = extra[0] if extra else None
            if kind == "int":
                w = QSpinBox()
                lo, hi = extra or (0, 100000)
                w.setRange(lo, hi)
                w.setValue(int(default))
            elif kind == "float":
                w = QDoubleSpinBox()
                lo, hi, dec = extra or (-1e9, 1e9, 4)
                w.setRange(lo, hi)
                w.setDecimals(dec)
                w.setValue(float(default))
            elif kind == "choice":
                w = QComboBox()
                for value, text in extra.items():
                    w.addItem(text, value)
                idx = w.findData(default)
                w.setCurrentIndex(max(idx, 0))
            elif kind == "bool":
                w = QCheckBox()
                w.setChecked(bool(default))
            else:
                w = QLineEdit(str(default))
            self.widgets[key] = w
            form.addRow(label, w)
        layout = QVBoxLayout(self)
        if note:
            lab = QLabel(note)
            lab.setWordWrap(True)
            lab.setStyleSheet("color:#444")
            layout.addWidget(lab)
        box = QGroupBox("Settings")
        box.setLayout(form)
        layout.addWidget(box)
        layout.addWidget(_buttons(self))
        self.resize(400, 0)

    def values(self) -> dict:
        out = {}
        for key, w in self.widgets.items():
            if isinstance(w, QSpinBox):
                out[key] = w.value()
            elif isinstance(w, QDoubleSpinBox):
                out[key] = w.value()
            elif isinstance(w, QComboBox):
                out[key] = w.currentData()
            elif isinstance(w, QCheckBox):
                out[key] = w.isChecked()
            else:
                out[key] = w.text()
        return out


# ====================================================================== fitting
@dataclass
class InputChoice:
    """One data set the fitting dialogs can use: a Y column (or a graph's data plot)."""
    label: str
    book: str
    sheet: str
    ycol: str
    xcol: str | None
    plot_id: str | None = None
    rows: tuple[int, int] | None = None   # 1-based inclusive row range (None = all rows)


class FitDialog(QDialog):
    """Linear Fit / Polynomial Fit / Nonlinear Curve Fit (Origin's NLFit) settings."""

    def __init__(self, kind: str, inputs: list[InputChoice], parent=None, current: int = 0,
                 graph_available: bool = False):
        super().__init__(parent)
        self.kind = kind
        self.inputs = inputs
        self.setWindowTitle({"linear": "Linear Fit", "poly": "Polynomial Fit",
                             "nlfit": "NLFit - Nonlinear Curve Fit"}[kind])
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        self.mode = QComboBox()
        for key, text in RECALC_MODES.items():
            self.mode.addItem(text, key)
        top.addWidget(QLabel("Recalculate"))
        top.addWidget(self.mode)
        top.addStretch(1)
        layout.addLayout(top)
        inp = QGroupBox("Input Data")
        g = QFormLayout(inp)
        self.data = QComboBox()
        for c in inputs:
            self.data.addItem(c.label)
        self.data.setCurrentIndex(max(0, min(current, len(inputs) - 1)))
        g.addRow("Range:", self.data)
        self.weight = QComboBox()
        for key, text in WEIGHTINGS.items():
            self.weight.addItem(text, key)
        g.addRow("Weight:", self.weight)
        layout.addWidget(inp)
        self.order = None
        self.function = None
        if kind == "poly":
            box = QGroupBox("Fit Control")
            f = QFormLayout(box)
            self.order = QSpinBox()
            self.order.setRange(2, 9)
            self.order.setValue(2)
            f.addRow("Polynomial Order:", self.order)
            layout.addWidget(box)
        elif kind == "nlfit":
            layout.addWidget(self._function_box())
        opts = QGroupBox("Quantities && Output")
        o = QGridLayout(opts)
        self.level = QDoubleSpinBox()
        self.level.setRange(50, 99.99)
        self.level.setValue(95)
        self.level.setSuffix(" %")
        self.conf = QCheckBox("Confidence bands")
        self.conf.setChecked(True)
        self.pred = QCheckBox("Prediction bands")
        self.resid = QCheckBox("Residual vs. independent plot")
        self.resid.setChecked(True)
        self.fitted = QCheckBox("Fitted values and residuals to source sheet")
        self.to_graph = QCheckBox("Add fit curve to the graph")
        self.to_graph.setChecked(graph_available)
        self.to_graph.setEnabled(graph_available)
        o.addWidget(QLabel("Confidence level:"), 0, 0)
        o.addWidget(self.level, 0, 1)
        o.addWidget(self.conf, 1, 0)
        o.addWidget(self.pred, 1, 1)
        o.addWidget(self.resid, 2, 0, 1, 2)
        o.addWidget(self.fitted, 3, 0, 1, 2)
        o.addWidget(self.to_graph, 4, 0, 1, 2)
        layout.addWidget(opts)
        layout.addWidget(_buttons(self, "Fit"))
        self.resize(560 if kind == "nlfit" else 440, 0)

    def _function_box(self) -> QWidget:
        box = QGroupBox("Function Selection")
        v = QVBoxLayout(box)
        row = QHBoxLayout()
        self.category = QComboBox()
        self.category.addItems(fl.CATEGORIES + ["User Defined"])
        self.function = QComboBox()
        self.function.setMinimumWidth(180)
        row.addWidget(QLabel("Category:"))
        row.addWidget(self.category, 1)
        row.addWidget(QLabel("Function:"))
        row.addWidget(self.function, 1)
        v.addLayout(row)
        self.equation = QLineEdit()
        self.equation.setPlaceholderText("y = a*exp(-b*x) + c")
        self.formula_label = QLabel()
        self.formula_label.setStyleSheet("color:#1f3d7a; font-family: 'Times New Roman'; "
                                         "font-size: 11pt; font-style: italic;")
        self.formula_label.setWordWrap(True)
        v.addWidget(self.formula_label)
        v.addWidget(self.equation)
        self.params = QTableWidget(0, 5)
        self.params.setHorizontalHeaderLabels(["Parameter", "Value (initial)", "Fixed",
                                               "Lower Bound", "Upper Bound"])
        self.params.verticalHeader().hide()
        self.params.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.params.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.params.verticalHeader().setDefaultSectionSize(22)
        self.params.setMinimumHeight(170)
        v.addWidget(self.params)
        hint = QLabel("Leave initial values empty for automatic guesses (Origin's parameter "
                      "initialization).")
        hint.setStyleSheet("color:#555")
        v.addWidget(hint)
        self.category.currentTextChanged.connect(self._fill_functions)
        self.function.currentIndexChanged.connect(self._fill_params)
        self.equation.editingFinished.connect(self._fill_params)
        self._fill_functions(self.category.currentText())
        return box

    def _fill_functions(self, category: str) -> None:
        self.function.blockSignals(True)
        self.function.clear()
        user = category == "User Defined"
        self.equation.setVisible(user)
        if not user:
            for f in fl.functions_in(category):
                self.function.addItem(f.name)
        self.function.blockSignals(False)
        self._fill_params()

    def model(self):
        if self.kind == "linear":
            return "Line"
        if self.kind == "poly":
            return f"Poly{self.order.value()}"
        if self.category.currentText() == "User Defined":
            return self.equation.text().strip()
        return self.function.currentText()

    def _fill_params(self) -> None:
        model = self.model()
        try:
            f = fl.get_function(model) if self.category.currentText() != "User Defined" \
                else fl.parse_user_function(model)
        except ValueError as exc:
            f = None
            self.formula_label.setText(str(exc))
        self.params.setRowCount(0)
        if f is None:
            return
        self.formula_label.setText(f.formula + (f"   — {f.description}" if f.description
                                                 else ""))
        for i, name in enumerate(f.params):
            self.params.insertRow(i)
            item = QTableWidgetItem(name)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.params.setItem(i, 0, item)
            for j in (1, 3, 4):
                self.params.setItem(i, j, QTableWidgetItem(""))
            fixed = QTableWidgetItem("")
            fixed.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            fixed.setCheckState(Qt.CheckState.Unchecked)
            self.params.setItem(i, 2, fixed)

    def preset(self, s: dict) -> None:
        """Show the settings of an earlier fit (Change Parameters)."""
        self.weight.setCurrentIndex(max(0, self.weight.findData(s.get("weighting", "none"))))
        self.level.setValue(float(s.get("level", 0.95)) * 100)
        self.conf.setChecked(bool(s.get("conf_band", True)))
        self.pred.setChecked(bool(s.get("pred_band", False)))
        self.resid.setChecked(bool(s.get("residual_plot", True)))
        self.fitted.setChecked(bool(s.get("output_fitted", False)))
        self.to_graph.setChecked(False)
        model = str(s.get("model", ""))
        if self.kind == "poly" and model.startswith("Poly") and model[4:].isdigit():
            self.order.setValue(int(model[4:]))
        elif self.kind == "nlfit" and model:
            for cat in fl.CATEGORIES:
                if model in [f.name for f in fl.functions_in(cat)]:
                    self.category.setCurrentText(cat)
                    self.function.setCurrentText(model)
                    break
            else:
                self.category.setCurrentText("User Defined")
                self.equation.setText(model)
                self._fill_params()

    def values(self) -> dict:
        out = {"input": self.inputs[self.data.currentIndex()] if self.inputs else None,
               "mode": self.mode.currentData(),
               "model": self.model(), "weighting": self.weight.currentData(),
               "level": self.level.value() / 100, "conf_band": self.conf.isChecked(),
               "pred_band": self.pred.isChecked(), "residual_plot": self.resid.isChecked(),
               "output_fitted": self.fitted.isChecked(), "to_graph": self.to_graph.isChecked(),
               "params0": None, "fixed": None, "bounds": None}
        if self.kind == "nlfit":
            p0, fixed, bounds = {}, {}, {}
            for i in range(self.params.rowCount()):
                name = self.params.item(i, 0).text()

                def num(j):
                    t = self.params.item(i, j).text().strip().replace(",", ".")
                    return float(t) if t else None
                try:
                    v, lo, hi = num(1), num(3), num(4)
                except ValueError:
                    raise ValueError(f"Parameter {name}: enter numbers only.") from None
                if v is not None:
                    p0[name] = v
                if self.params.item(i, 2).checkState() == Qt.CheckState.Checked:
                    fixed[name] = v
                if lo is not None or hi is not None:
                    bounds[name] = (lo, hi)
            out.update(params0=p0 or None, fixed=fixed or None, bounds=bounds or None)
        return out


# ====================================================================== matrices
class MatrixValuesDialog(QDialog):
    """Origin's Set Matrix Values: ``Mat(1) = f(x, y, i, j)`` with Apply / OK.
    ``apply(formula)`` raises ValueError with a message on errors."""

    EXAMPLES = (("sin(x)*cos(y)", "surface of x and y"),
                ("exp(-((x-16)^2+(y-16)^2)/50)", "Gaussian peak"),
                ("i*j", "row × column"), ("sqrt(x^2+y^2)", "distance from the origin"),
                ("x^2 - y^2", "saddle"))

    def __init__(self, ms, apply, parent=None):
        super().__init__(parent)
        self._apply = apply
        self.setWindowTitle(f"Set Values - [{ms.name}]")
        head = QLabel(f"Mat({ms.name}) =")
        head.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        self.edit = QPlainTextEdit(ms.formula or "")
        self.edit.setFont(QFont("Consolas", 10))
        self.edit.setFixedHeight(80)
        info = QLabel(f"x: {ms.x1:g} … {ms.x2:g} ({ms.n_cols} columns)    "
                      f"y: {ms.y1:g} … {ms.y2:g} ({ms.n_rows} rows)    "
                      "i = row, j = column (1-based)")
        info.setStyleSheet("color:#444")
        ex_b = QPushButton("Examples")
        ex_b.setObjectName("menuButton")
        ex_menu = QMenu(ex_b)
        for text, desc in self.EXAMPLES:
            ex_menu.addAction(f"{text}    — {desc}", lambda t=text: self.edit.setPlainText(t))
        ex_b.setMenu(ex_menu)
        self.error = QLabel("")
        self.error.setStyleSheet("color:#b00000")
        self.error.setWordWrap(True)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Apply
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._ok)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply)
        top = QHBoxLayout()
        top.addWidget(head)
        top.addStretch(1)
        top.addWidget(ex_b)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.edit)
        layout.addWidget(info)
        layout.addWidget(self.error)
        layout.addWidget(buttons)
        self.resize(560, 0)

    def apply(self) -> bool:
        try:
            self._apply(self.edit.toPlainText().strip())
        except ValueError as exc:
            self.error.setText(str(exc))
            return False
        self.error.setText("")
        return True

    def _ok(self) -> None:
        if self.apply():
            self.accept()


class MatrixDimensionsDialog(QDialog):
    """Origin's Matrix Dimensions and Labels: rows × columns, the X/Y coordinate mapping
    (first / last X and Y) and the Long Name / Units of X, Y and Z."""

    def __init__(self, ms, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Matrix Dimensions and Labels - {ms.name}")
        self.cols = QSpinBox()
        self.rows = QSpinBox()
        for w, v in ((self.cols, ms.n_cols), (self.rows, ms.n_rows)):
            w.setRange(1, 100000)
            w.setValue(v)
        dims = QGroupBox("Dimensions")
        g = QFormLayout(dims)
        g.addRow("Columns:", self.cols)
        g.addRow("Rows:", self.rows)
        self.xy = {}
        mapping = QGroupBox("XY Mapping")
        m = QGridLayout(mapping)
        for r, (axis, lo, hi) in enumerate((("X", ms.x1, ms.x2), ("Y", ms.y1, ms.y2))):
            a, b = QDoubleSpinBox(), QDoubleSpinBox()
            for w, v in ((a, lo), (b, hi)):
                w.setRange(-1e15, 1e15)
                w.setDecimals(6)
                w.setValue(v)
            self.xy[axis] = (a, b)
            m.addWidget(QLabel(f"{axis}: first"), r, 0)
            m.addWidget(a, r, 1)
            m.addWidget(QLabel("last"), r, 2)
            m.addWidget(b, r, 3)
        labels = QGroupBox("Labels")
        lg = QGridLayout(labels)
        lg.addWidget(QLabel("Long Name"), 0, 1)
        lg.addWidget(QLabel("Units"), 0, 2)
        self.labels = {}
        for r, (axis, name, units) in enumerate((("X", ms.x_name, ms.x_units),
                                                 ("Y", ms.y_name, ms.y_units),
                                                 ("Z", ms.long_name, ms.units)), start=1):
            ln, un = QLineEdit(name), QLineEdit(units)
            self.labels[axis] = (ln, un)
            lg.addWidget(QLabel(axis), r, 0)
            lg.addWidget(ln, r, 1)
            lg.addWidget(un, r, 2)
        layout = QVBoxLayout(self)
        for box in (dims, mapping, labels):
            layout.addWidget(box)
        layout.addWidget(_buttons(self))
        self.resize(460, 0)

    def values(self) -> dict:
        return {"rows": self.rows.value(), "cols": self.cols.value(),
                "x1": self.xy["X"][0].value(), "x2": self.xy["X"][1].value(),
                "y1": self.xy["Y"][0].value(), "y2": self.xy["Y"][1].value(),
                "x_name": self.labels["X"][0].text().strip(),
                "x_units": self.labels["X"][1].text().strip(),
                "y_name": self.labels["Y"][0].text().strip(),
                "y_units": self.labels["Y"][1].text().strip(),
                "long_name": self.labels["Z"][0].text().strip(),
                "units": self.labels["Z"][1].text().strip()}


# ====================================================================== add plot to layer
class AddPlotDialog(QDialog):
    """Origin's Layer Contents / Plot Setup in short: pick a sheet, Y columns and a type."""

    def __init__(self, books: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Plot to Layer")
        self.books = books
        self.sheet = QComboBox()
        self.entries = []
        for book in books:
            for s in book.sheets:
                if isinstance(s, Worksheet):
                    self.entries.append((book, s))
                    self.sheet.addItem(f"[{book.name}]{s.name}")
        self.columns = QListWidget()
        self.columns.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.type = QComboBox()
        for key in ("line_symbol", "line", "scatter", "yerror", "column", "area"):
            self.type.addItem(PLOT_TYPES[key][0], key)
        self.first = QLineEdit()
        self.first.setPlaceholderText("1")
        self.last = QLineEdit()
        self.last.setPlaceholderText("<last>")
        for e in (self.first, self.last):
            e.setFixedWidth(70)
        rows = QHBoxLayout()
        rows.addWidget(QLabel("From"))
        rows.addWidget(self.first)
        rows.addWidget(QLabel("To"))
        rows.addWidget(self.last)
        rows.addStretch(1)
        form = QFormLayout()
        form.addRow("Worksheet:", self.sheet)
        form.addRow("Plot Type:", self.type)
        form.addRow("Rows:", rows)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(QLabel("Y columns (each is plotted against its designated X):"))
        layout.addWidget(self.columns)
        layout.addWidget(_buttons(self, "Add"))
        self.sheet.currentIndexChanged.connect(self._fill)
        self._fill(0)
        self.resize(380, 380)

    def _fill(self, index: int) -> None:
        self.columns.clear()
        if not self.entries:
            return
        _, ws = self.entries[index]
        for i, c in enumerate(ws.columns):
            item = QListWidgetItem(ws.label(i) + (f"   {c.long_name}" if c.long_name else ""))
            item.setData(Qt.ItemDataRole.UserRole, i)
            self.columns.addItem(item)

    def values(self) -> dict | None:
        if not self.entries:
            return None
        book, ws = self.entries[self.sheet.currentIndex()]
        cols = [it.data(Qt.ItemDataRole.UserRole) for it in self.columns.selectedItems()]
        lo, hi = self.first.text().strip(), self.last.text().strip()
        rows = (int(lo) if lo.isdigit() else 1, int(hi) if hi.isdigit() else 0) \
            if lo.isdigit() or hi.isdigit() else None
        return {"book": book, "sheet": ws, "cols": cols, "kind": self.type.currentData(),
                "rows": rows}

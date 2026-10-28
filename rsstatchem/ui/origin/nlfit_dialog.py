"""NLFit dialog with the OriginPro options (Analysis > Fitting > Nonlinear Curve Fit):
function selection over the whole library and the user's organizer functions, multiple
input datasets with Independent / Global (parameter sharing) / Concatenate / Replica fitting,
a parameter table with Value / Fixed / Share / bounds, general linear constraints, weighting
(incl. experimental and arbitrary dataset), iteration algorithm (L-M, TRF, Downhill Simplex)
and output options (bands, residual diagnostics). The dialog only collects settings;
:mod:`fitting_menus` runs them.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QDoubleSpinBox, QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QPlainTextEdit, QPushButton,
                               QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget,
                               QVBoxLayout, QWidget)

from rsstatchem.core.analysis import fit_multi as fm
from rsstatchem.core.analysis import fitting as fit
from rsstatchem.core.analysis import functions as fl
from rsstatchem.ui.origin.dialogs import InputChoice
from rsstatchem.ui.origin.model import RECALC_MODES

USER_EQUATION = "User Defined (type an equation)"


def _check_item(checked: bool = False) -> QTableWidgetItem:
    it = QTableWidgetItem("")
    it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
    it.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
    return it


class NLFitDialog(QDialog):
    COLS = ["Parameter", "Value (initial)", "Fixed", "Share", "Lower Bound", "Upper Bound"]

    def __init__(self, inputs: list[InputChoice], parent=None, current: int = 0,
                 graph_available: bool = False):
        super().__init__(parent)
        self.inputs = inputs
        self.setWindowTitle("NLFit - Nonlinear Curve Fit (Multiple Data, Global, Constraints)")
        tabs = QTabWidget()
        tabs.addTab(self._settings_page(current), "Settings")
        tabs.addTab(self._params_page(), "Parameters")
        tabs.addTab(self._constraints_page(), "Constraints")
        tabs.addTab(self._output_page(graph_available), "Output")
        self.tabs = tabs
        top = QHBoxLayout()
        self.recalc = QComboBox()
        for key, text in RECALC_MODES.items():
            self.recalc.addItem(text, key)
        top.addWidget(QLabel("Recalculate"))
        top.addWidget(self.recalc)
        top.addStretch(1)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                               | QDialogButtonBox.StandardButton.Cancel)
        box.button(QDialogButtonBox.StandardButton.Ok).setText("Fit")
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(tabs)
        lay.addWidget(box)
        self.resize(720, 560)
        self._fill_functions(self.category.currentText())

    # ------------------------------------------------------------------ pages
    def _settings_page(self, current: int) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        fbox = QGroupBox("Function Selection")
        fg = QGridLayout(fbox)
        self.category = QComboBox()
        self.category.addItems(list(dict.fromkeys(fl.CATEGORIES)) + [USER_EQUATION])
        self.function = QComboBox()
        self.function.setMinimumWidth(200)
        self.organizer_btn = QPushButton("Organizer...")
        self.organizer_btn.clicked.connect(self._organizer)
        fg.addWidget(QLabel("Category:"), 0, 0)
        fg.addWidget(self.category, 0, 1)
        fg.addWidget(QLabel("Function:"), 0, 2)
        fg.addWidget(self.function, 0, 3)
        fg.addWidget(self.organizer_btn, 0, 4)
        self.equation = QLineEdit()
        self.equation.setPlaceholderText("y = a*exp(-b*x) + c")
        self.formula = QLabel()
        self.formula.setWordWrap(True)
        self.formula.setStyleSheet("color:#1f3d7a; font-style: italic;")
        fg.addWidget(self.formula, 1, 0, 1, 5)
        fg.addWidget(self.equation, 2, 0, 1, 5)
        v.addWidget(fbox)
        dbox = QGroupBox("Data Selection")
        dg = QGridLayout(dbox)
        self.data = QListWidget()
        self.data.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        for i, c in enumerate(self.inputs):
            it = QListWidgetItem(c.label)
            it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if i == current else Qt.CheckState.Unchecked)
            self.data.addItem(it)
        self.data.setMaximumHeight(110)
        self.mode = QComboBox()
        for k, t in fm.MODES.items():
            self.mode.addItem(t, k)
        self.mode.setCurrentIndex(1)
        self.replicas = QSpinBox()
        self.replicas.setRange(1, 30)
        self.replicas.setValue(2)
        self.weight = QComboBox()
        for k, t in {**fit.WEIGHTINGS, "experimental": "Experimental (1/y²)",
                     "arbitrary": "Arbitrary Dataset (Y Error column as w)"}.items():
            self.weight.addItem(t, k)
        self.algorithm = QComboBox()
        for k, t in fit.ALGORITHMS.items():
            self.algorithm.addItem(t, k)
        dg.addWidget(self.data, 0, 0, 1, 4)
        dg.addWidget(QLabel("Multi-Data Fit Mode:"), 1, 0)
        dg.addWidget(self.mode, 1, 1)
        dg.addWidget(QLabel("Number of Replicas:"), 1, 2)
        dg.addWidget(self.replicas, 1, 3)
        dg.addWidget(QLabel("Weight:"), 2, 0)
        dg.addWidget(self.weight, 2, 1)
        dg.addWidget(QLabel("Iteration Algorithm:"), 2, 2)
        dg.addWidget(self.algorithm, 2, 3)
        v.addWidget(dbox)
        v.addStretch(1)
        self.category.currentTextChanged.connect(self._fill_functions)
        self.function.currentIndexChanged.connect(self._fill_params)
        self.equation.editingFinished.connect(self._fill_params)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        return w

    def _params_page(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self.params = QTableWidget(0, len(self.COLS))
        self.params.setHorizontalHeaderLabels(self.COLS)
        self.params.verticalHeader().hide()
        self.params.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.params.verticalHeader().setDefaultSectionSize(22)
        v.addWidget(self.params)
        hint = QLabel("Empty values: automatic initial values (per dataset for global fits). "
                      "Share: one value for all datasets (Global Fit).")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#555")
        v.addWidget(hint)
        return w

    def _constraints_page(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(QLabel("Linear constraints, one per line (=, <, <=, >, >=):"))
        self.constraints = QPlainTextEdit()
        self.constraints.setPlaceholderText("A1 + A2 = 1\nxc1 < xc2\nt1 >= 2*t2")
        v.addWidget(self.constraints)
        note = QLabel("Global fits use the joint names, e.g. y0_1, y0_2 for unshared "
                      "parameters.")
        note.setStyleSheet("color:#555")
        v.addWidget(note)
        return w

    def _output_page(self, graph_available: bool) -> QWidget:
        w = QWidget()
        g = QGridLayout(w)
        self.level = QDoubleSpinBox()
        self.level.setRange(50, 99.99)
        self.level.setValue(95)
        self.level.setSuffix(" %")
        self.conf = QCheckBox("Confidence bands")
        self.conf.setChecked(True)
        self.pred = QCheckBox("Prediction bands")
        self.resid = QCheckBox("Residual vs. independent and histogram")
        self.resid.setChecked(True)
        self.resid_extra = QCheckBox("Residual diagnostics: normal probability, order, lag")
        self.fitted = QCheckBox("Fitted values and residuals to source sheet")
        self.to_graph = QCheckBox("Add fit curve to the graph")
        self.to_graph.setChecked(graph_available)
        self.to_graph.setEnabled(graph_available)
        g.addWidget(QLabel("Confidence level:"), 0, 0)
        g.addWidget(self.level, 0, 1)
        for r, wd in enumerate((self.conf, self.pred, self.resid, self.resid_extra,
                                self.fitted, self.to_graph), start=1):
            g.addWidget(wd, r, 0, 1, 2)
        g.setRowStretch(8, 1)
        return w

    # ------------------------------------------------------------------ behaviour
    def _organizer(self) -> None:
        from rsstatchem.ui.origin.fit_organizer import FunctionOrganizerDialog
        FunctionOrganizerDialog(self).exec()
        cur = self.category.currentText()
        self.category.blockSignals(True)
        self.category.clear()
        self.category.addItems(list(dict.fromkeys(fl.CATEGORIES)) + [USER_EQUATION])
        self.category.setCurrentText(cur)
        self.category.blockSignals(False)
        self._fill_functions(self.category.currentText())

    def _mode_changed(self) -> None:
        self.replicas.setEnabled(self.mode.currentData() == "replica")

    def _fill_functions(self, category: str) -> None:
        self.function.blockSignals(True)
        self.function.clear()
        user = category == USER_EQUATION
        self.equation.setVisible(user)
        self.function.setVisible(not user)
        if not user:
            for f in fl.functions_in(category):
                self.function.addItem(f.name)
        self.function.blockSignals(False)
        self._mode_changed()
        self._fill_params()

    def set_function(self, name: str) -> None:
        f = fl.get_function(name)
        if f is None:
            self.category.setCurrentText(USER_EQUATION)
            self.equation.setText(name)
            self._fill_params()
            return
        self.category.setCurrentText(f.category)
        self.function.setCurrentText(f.name)

    def model(self) -> str:
        if self.category.currentText() == USER_EQUATION:
            return self.equation.text().strip()
        return self.function.currentText()

    def model_function(self):
        m = self.model()
        if not m:
            return None
        try:
            return fl.get_function(m) if self.category.currentText() != USER_EQUATION \
                else fl.parse_user_function(m)
        except ValueError as exc:
            self.formula.setText(str(exc))
            return None

    def _fill_params(self) -> None:
        f = self.model_function()
        self.params.setRowCount(0)
        if f is None:
            return
        self.formula.setText(f.formula + (f"   — {f.description}" if f.description else ""))
        default_fixed = f.meta.get("fixed") or {}
        for i, name in enumerate(f.params):
            self.params.insertRow(i)
            it = QTableWidgetItem(name)
            it.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.params.setItem(i, 0, it)
            val = default_fixed.get(name)
            self.params.setItem(i, 1, QTableWidgetItem("" if val is None else f"{val:g}"))
            self.params.setItem(i, 2, _check_item(name in default_fixed))
            self.params.setItem(i, 3, _check_item(False))
            for j in (4, 5):
                self.params.setItem(i, j, QTableWidgetItem(""))

    def set_param(self, name: str, value=None, fixed=None, share=None, lower=None,
                  upper=None) -> None:
        for i in range(self.params.rowCount()):
            if self.params.item(i, 0).text() != name:
                continue
            if value is not None:
                self.params.item(i, 1).setText(f"{value:g}")
            for col, flag in ((2, fixed), (3, share)):
                if flag is not None:
                    self.params.item(i, col).setCheckState(
                        Qt.CheckState.Checked if flag else Qt.CheckState.Unchecked)
            if lower is not None:
                self.params.item(i, 4).setText(f"{lower:g}")
            if upper is not None:
                self.params.item(i, 5).setText(f"{upper:g}")
            return
        raise KeyError(name)

    def check_inputs(self, indices: list[int]) -> None:
        for i in range(self.data.count()):
            self.data.item(i).setCheckState(Qt.CheckState.Checked if i in indices
                                            else Qt.CheckState.Unchecked)

    def selected_inputs(self) -> list[InputChoice]:
        return [self.inputs[i] for i in range(self.data.count())
                if self.data.item(i).checkState() == Qt.CheckState.Checked]

    def values(self) -> dict:
        p0, fixed, bounds, shared = {}, {}, {}, []
        for i in range(self.params.rowCount()):
            name = self.params.item(i, 0).text()

            def num(j):
                t = self.params.item(i, j).text().strip().replace(",", ".")
                return float(t) if t else None
            try:
                v, lo, hi = num(1), num(4), num(5)
            except ValueError:
                raise ValueError(f"Parameter {name}: enter numbers only.") from None
            if v is not None:
                p0[name] = v
            if self.params.item(i, 2).checkState() == Qt.CheckState.Checked:
                fixed[name] = v
            if self.params.item(i, 3).checkState() == Qt.CheckState.Checked:
                shared.append(name)
            if lo is not None or hi is not None:
                bounds[name] = (lo, hi)
        inputs = self.selected_inputs()
        if not inputs:
            raise ValueError("Select at least one dataset.")
        if not self.model():
            raise ValueError("Choose a function or enter an equation.")
        cons = [ln.strip() for ln in self.constraints.toPlainText().splitlines() if ln.strip()]
        return {"inputs": inputs, "input": inputs[0], "mode": self.recalc.currentData(),
                "multi_mode": self.mode.currentData(), "replicas": self.replicas.value(),
                "model": self.model(), "weighting": self.weight.currentData(),
                "algorithm": self.algorithm.currentData(), "params0": p0 or None,
                "fixed": fixed or None, "bounds": bounds or None, "shared": shared,
                "constraints": cons or None, "level": self.level.value() / 100,
                "conf_band": self.conf.isChecked(), "pred_band": self.pred.isChecked(),
                "residual_plot": self.resid.isChecked(),
                "residual_extra": self.resid_extra.isChecked(),
                "output_fitted": self.fitted.isChecked(), "to_graph": self.to_graph.isChecked()}


class ModelListDialog(QDialog):
    """Compare Models / Rank Models / Compare Datasets: inputs, models and the criterion."""

    def __init__(self, title: str, inputs: list[InputChoice], parent=None, current: int = 0,
                 n_models: int = 2, multi_input: bool = False, category: bool = False):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.inputs = inputs
        form = QFormLayout()
        self.data = QListWidget()
        for i, c in enumerate(inputs):
            it = QListWidgetItem(c.label)
            it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if (i == current or
                                                       (multi_input and i == current + 1))
                             else Qt.CheckState.Unchecked)
            self.data.addItem(it)
        self.data.setMaximumHeight(100)
        form.addRow("Input Data:", self.data)
        names = sorted(fl.LIBRARY) + [f"Poly{k}" for k in range(2, 10)] + ["Line"]
        self.models = []
        for k in range(n_models):
            cb = QComboBox()
            cb.setEditable(True)
            cb.addItems(names)
            cb.setCurrentText(["ExpDec1", "ExpDec2", "ExpDec3"][k] if k < 3 else "Line")
            self.models.append(cb)
            form.addRow(f"Model {k + 1}:" if n_models > 1 else "Model:", cb)
        self.category = None
        if category:
            self.category = QComboBox()
            self.category.addItems(fl.CATEGORIES)
            self.category.setCurrentText("Exponential")
            form.addRow("Category:", self.category)
        self.criterion = QComboBox()
        for k, t in {"aicc": "AICc", "aic": "AIC", "bic": "BIC", "adj_r2": "Adj. R-Square"}.items():
            self.criterion.addItem(t, k)
        form.addRow("Criterion:", self.criterion)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                               | QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(box)
        self.resize(520, 0)

    def values(self) -> dict:
        inputs = [self.inputs[i] for i in range(self.data.count())
                  if self.data.item(i).checkState() == Qt.CheckState.Checked]
        return {"inputs": inputs, "models": [c.currentText().strip() for c in self.models
                                             if c.currentText().strip()],
                "category": self.category.currentText() if self.category else None,
                "criterion": self.criterion.currentData()}

"""Analysis > Fitting dialogs modelled on OriginLab: Fit Linear, Fit Polynomial, Nonlinear
Curve Fit (function library by category, user-defined functions, parameter table with initial
values, fixing and bounds, weighting) and a Calibration Curve tool for analytical chemistry.

Exports ``MENUS`` for the whole top-level Analysis menu (the Mathematics, Signal Processing and
Peaks dialogs live in ``signal.py``) and ``selftest()``.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QDialogButtonBox,
                               QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QMessageBox, QPushButton, QTableWidget,
                               QTableWidgetItem, QVBoxLayout)

from rsstatchem.core.analysis import calibration as cal
from rsstatchem.core.analysis import functions as fl
from rsstatchem.core.analysis.common import column, num, parse_numbers
from rsstatchem.core.analysis.fitting import WEIGHTINGS, FitOptions, make_weights, run_fit
from rsstatchem.ui.dialogs.base import SubDialog, VariablePickerDialog, is_numeric
from rsstatchem.ui.dialogs.signal import (BaselineDialog, DifferentiateDialog, FFTDialog,
                                          FindPeaksDialog, FormMixin, IntegrateDialog,
                                          InterpolateDialog, MultiPeakFitDialog,
                                          NormalizeDialog, SmoothDialog)

_FIT_DEFAULTS = {"weighting": "none", "level": 95.0, "conf_band": True, "pred_band": False,
                 "residual_plot": True, "anova": True, "lack_of_fit": True,
                 "scale_errors": True, "find_x": "", "find_y": "", "save_fitted": False,
                 "save_residuals": False}


class _FindDialog(QDialog):
    def __init__(self, title: str, find_x: str, find_y: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.ex = QLineEdit(find_x)
        self.ey = QLineEdit(find_y)
        for e in (self.ex, self.ey):
            e.setPlaceholderText("e.g. 1.5, 2, 10")
            e.setMinimumWidth(260)
        form = QFormLayout()
        form.addRow("Find X from Y values:", self.ex)
        form.addRow("Find Y from X values:", self.ey)
        hint = QLabel("Separate values with commas (or semicolons with decimal commas).")
        hint.setStyleSheet("color:#666")
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Continue")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(hint)
        layout.addWidget(buttons)


class _FitDialog(FormMixin, VariablePickerDialog):
    """X, Y, optional Y error / weight column, Output…, Find X/Y…, Save… and weighting."""

    DEFAULTS = dict(_FIT_DEFAULTS)

    def setup_xy(self) -> None:
        self.setMinimumSize(max(self.minimumWidth(), 660), max(self.minimumHeight(), 480))
        self.add_target("x", "X (Independent):", accepts=is_numeric, single=True)
        self.add_target("y", "Y (Dependent):", accepts=is_numeric, single=True)
        self.add_target("err", "Y Error / Weight (optional):", accepts=is_numeric, single=True,
                        required=False)
        self.add_option_button("Output...", self._output)
        self.add_option_button("Find X/Y...", self._find)
        self.add_option_button("Save...", self._save)

    def weighting_row(self, form: QFormLayout) -> QComboBox:
        return self.opt_combo(form, "Weighting:", "weighting", WEIGHTINGS)

    def _output(self) -> None:
        dlg = SubDialog(f"{self.windowTitle()}: Output", self)
        box = dlg.group("Tables")
        dlg.check(box, "anova", "ANOVA table", self.opts["anova"])
        dlg.check(box, "lack_of_fit", "Lack of Fit test (replicated X)", self.opts["lack_of_fit"])
        dlg.check(box, "scale_errors", "Scale errors with sqrt(Reduced Chi-Sqr)",
                  self.opts["scale_errors"])
        dlg.number(box, "level", "Confidence Level:", self.opts["level"], 50, 99.99, 2, " %")
        box = dlg.group("Plots")
        dlg.check(box, "conf_band", "Confidence bands", self.opts["conf_band"])
        dlg.check(box, "pred_band", "Prediction bands", self.opts["pred_band"])
        dlg.check(box, "residual_plot", "Residual vs. Independent plot",
                  self.opts["residual_plot"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def _find(self) -> None:
        dlg = _FindDialog(f"{self.windowTitle()}: Find X/Y", self.opts["find_x"],
                          self.opts["find_y"], self)
        if dlg.exec():
            self.opts["find_x"], self.opts["find_y"] = dlg.ex.text(), dlg.ey.text()

    def _save(self) -> None:
        dlg = SubDialog(f"{self.windowTitle()}: Save", self, columns=1)
        box = dlg.group("Save as New Variables")
        dlg.check(box, "save_fitted", "Fitted values (FIT_n)", self.opts["save_fitted"])
        dlg.check(box, "save_residuals", "Residuals (RES_n)", self.opts["save_residuals"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def validate(self) -> str | None:
        if self.opts["weighting"] in ("instrumental", "direct") and not self.selected("err"):
            return "Instrumental and direct weighting need a Y Error / Weight variable."
        for key, what in (("find_x", "Find X from Y"), ("find_y", "Find Y from X")):
            try:
                parse_numbers(self.opts[key])
            except ValueError as exc:
                return f"{what}: {exc}"
        return None

    def fit_options(self) -> FitOptions:
        o = self.opts
        return FitOptions(level=o["level"] / 100, conf_band=o["conf_band"],
                          pred_band=o["pred_band"], residual_plot=o["residual_plot"],
                          anova=o["anova"], lack_of_fit=o["lack_of_fit"],
                          find_x=parse_numbers(o["find_x"]), find_y=parse_numbers(o["find_y"]),
                          save_fitted=o["save_fitted"], save_residuals=o["save_residuals"],
                          scale_errors=o["scale_errors"])

    def _run(self, model, fixed=None, params0=None, bounds=None, title=None, command="NLFIT"):
        return run_fit(self.ds, self.selected_one("x"), self.selected_one("y"), model,
                       ename=self.selected_one("err"), weighting=self.opts["weighting"],
                       params0=params0, bounds=bounds, fixed=fixed, opts=self.fit_options(),
                       title=title, command=command)


class FitLinearDialog(_FitDialog):
    DEFAULTS = {**_FIT_DEFAULTS, "fix_intercept": False, "intercept": 0.0, "fix_slope": False,
                "slope": 1.0}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Fit Linear", parent)
        self.setup_xy()
        form = self.form_group("Fit Control")
        self.weighting_row(form)
        self.opt_check(form, "fix_intercept", "Fix Intercept")
        self.w_int = self.opt_spin(form, "Intercept at:", "intercept", -1e12, 1e12, 6)
        form = self.form_group("Slope")
        self.opt_check(form, "fix_slope", "Fix Slope")
        self.w_slope = self.opt_spin(form, "Slope at:", "slope", -1e12, 1e12, 6)
        self.options_changed()
        self.finish()

    def options_changed(self):
        if hasattr(self, "w_slope"):
            self.w_int.setEnabled(self.opts["fix_intercept"])
            self.w_slope.setEnabled(self.opts["fix_slope"])

    def validate(self):
        if self.opts["fix_intercept"] and self.opts["fix_slope"]:
            return "Intercept and slope cannot both be fixed."
        return super().validate()

    def compute(self):
        o = self.opts
        fixed = {}
        if o["fix_intercept"]:
            fixed["Intercept"] = o["intercept"]
        if o["fix_slope"]:
            fixed["Slope"] = o["slope"]
        return self._run(fl.polynomial(1), fixed=fixed or None, title="Linear Fit",
                         command="FITLINEAR")


class FitPolynomialDialog(_FitDialog):
    DEFAULTS = {**_FIT_DEFAULTS, "order": 2, "fix_intercept": False, "intercept": 0.0}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Fit Polynomial", parent)
        self.setup_xy()
        form = self.form_group("Fit Control")
        self.opt_spin(form, "Polynomial Order:", "order", 1, 9)
        self.weighting_row(form)
        form = self.form_group("Intercept")
        self.opt_check(form, "fix_intercept", "Fix Intercept")
        self.w_int = self.opt_spin(form, "Intercept at:", "intercept", -1e12, 1e12, 6)
        self.options_changed()
        self.finish()

    def options_changed(self):
        if hasattr(self, "w_int"):
            self.w_int.setEnabled(self.opts["fix_intercept"])

    def compute(self):
        o = self.opts
        fixed = {"Intercept": o["intercept"]} if o["fix_intercept"] else None
        return self._run(fl.polynomial(int(o["order"])), fixed=fixed, title="Polynomial Fit",
                         command="FITPOLY")


_USER = "User Defined"


class NonlinearFitDialog(_FitDialog):
    DEFAULTS = {**_FIT_DEFAULTS, "category": "Exponential", "function": "ExpDec1",
                "user_expr": "y = A*exp(-x/t) + y0", "table": {}}
    COLS = ("Parameter", "Initial Value", "Fixed", "Lower Bound", "Upper Bound")

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Nonlinear Curve Fit", parent)
        self.setMinimumSize(780, 700)
        self._model: fl.FitFunction | None = None
        self.setup_xy()
        box = QGroupBox("Function")
        outer = QVBoxLayout(box)
        top = QHBoxLayout()
        self.category = QComboBox()
        self.category.addItems(fl.CATEGORIES + [_USER])
        self.function = QComboBox()
        self.function.setMinimumWidth(170)
        top.addWidget(QLabel("Category:"))
        top.addWidget(self.category, 1)
        top.addWidget(QLabel("Function:"))
        top.addWidget(self.function, 1)
        top.addWidget(QLabel("Weighting:"))
        self.weight = QComboBox()
        for k, v in WEIGHTINGS.items():
            self.weight.addItem(v, k)
        self.weight.setCurrentIndex(max(self.weight.findData(self.opts["weighting"]), 0))
        self.weight.currentIndexChanged.connect(
            lambda _: self.opts.__setitem__("weighting", self.weight.currentData()))
        top.addWidget(self.weight, 1)
        outer.addLayout(top)
        self.user = QLineEdit(self.opts["user_expr"])
        self.user.setPlaceholderText("y = A*exp(-x/t) + y0")
        self.user_hint = QLabel("Use x, parameters by name, + − * / ^, and exp, ln (natural), "
                                "log (base 10), sqrt, abs, sin, cos, tan, erf, pi. Parameters "
                                "are detected automatically.")
        self.user_hint.setWordWrap(True)
        self.user_hint.setStyleSheet("color:#666")
        outer.addWidget(self.user)
        outer.addWidget(self.user_hint)
        self.formula = QLabel()
        self.formula.setWordWrap(True)
        self.formula.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        outer.addWidget(self.formula)
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setMinimumHeight(150)
        outer.addWidget(self.table, 1)
        row = QHBoxLayout()
        init = QPushButton("Initialize")
        init.setToolTip("Compute automatic initial values from the selected X and Y data")
        init.clicked.connect(self._initialize)
        clear = QPushButton("Clear Values")
        clear.clicked.connect(self._clear)
        row.addWidget(QLabel("Empty initial values are estimated automatically."))
        row.addStretch()
        row.addWidget(init)
        row.addWidget(clear)
        outer.addLayout(row)
        self.options_layout.addWidget(box, 1)
        self.category.setCurrentText(self.opts["category"] if self.opts["category"] in
                                     fl.CATEGORIES + [_USER] else "Exponential")
        self._fill_functions()
        self.category.currentTextChanged.connect(self._category_changed)
        self.function.currentTextChanged.connect(self._function_changed)
        self.user.editingFinished.connect(self._user_changed)
        self._function_changed()
        self.finish()

    # -- function selection
    def _fill_functions(self) -> None:
        cat = self.category.currentText()
        self.function.blockSignals(True)
        self.function.clear()
        if cat == _USER:
            self.function.addItem("User")
        else:
            names = [f.name for f in fl.functions_in(cat)]
            self.function.addItems(names)
            if self.opts["function"] in names:
                self.function.setCurrentText(self.opts["function"])
        self.function.blockSignals(False)
        is_user = cat == _USER
        self.user.setVisible(is_user)
        self.user_hint.setVisible(is_user)

    def _category_changed(self) -> None:
        self._store_table()
        self.opts["category"] = self.category.currentText()
        self._fill_functions()
        self._function_changed()

    def _function_changed(self) -> None:
        self._store_table()
        name = self.function.currentText()
        if self.category.currentText() == _USER:
            self._user_changed()
            return
        self.opts["function"] = name
        self._model = fl.get_function(name)
        self._show_model()

    def _user_changed(self) -> None:
        self._store_table()
        self.opts["user_expr"] = self.user.text()
        try:
            self._model = fl.parse_user_function(self.user.text())
            self.user.setStyleSheet("")
        except fl.FunctionSyntaxError as exc:
            self._model = None
            self.user.setStyleSheet("background:#FFEBEE")
            self.formula.setText(f"<span style='color:#C62828'>{exc}</span>")
            self.table.setRowCount(0)
            return
        self._show_model()

    def _key(self) -> str:
        if self._model is None:
            return ""
        return f"user:{self.opts['user_expr']}" if self._model.user_defined else self._model.name

    def _show_model(self) -> None:
        f = self._model
        if f is None:
            return
        self.formula.setText(f"<b>{f.formula}</b><br><span style='color:#555'>"
                             f"{f.description}</span>")
        saved = self.opts["table"].get(self._key(), {})
        self.table.setRowCount(len(f.params))
        for r, p in enumerate(f.params):
            init, fixed, lo, hi = saved.get(p, ("", False, "", ""))
            name = QTableWidgetItem(p)
            name.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(r, 0, name)
            self.table.setItem(r, 1, QTableWidgetItem(init))
            check = QTableWidgetItem("")
            check.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            check.setCheckState(Qt.CheckState.Checked if fixed else Qt.CheckState.Unchecked)
            self.table.setItem(r, 2, check)
            self.table.setItem(r, 3, QTableWidgetItem(lo))
            self.table.setItem(r, 4, QTableWidgetItem(hi))

    def _row(self, r: int) -> tuple[str, bool, str, str]:
        def text(c):
            item = self.table.item(r, c)
            return item.text().strip() if item else ""
        check = self.table.item(r, 2)
        fixed = bool(check and check.checkState() == Qt.CheckState.Checked)
        return text(1), fixed, text(3), text(4)

    def _store_table(self) -> None:
        if self._model is None or self.table.rowCount() != len(self._model.params):
            return
        self.opts["table"] = dict(self.opts["table"])
        self.opts["table"][self._key()] = {p: self._row(r)
                                           for r, p in enumerate(self._model.params)}

    def _initialize(self) -> None:
        if self._model is None:
            return
        xn, yn = self.selected_one("x"), self.selected_one("y")
        if not xn or not yn:
            QMessageBox.information(self, self.windowTitle(), "Select the X and Y variables "
                                    "first.")
            return
        x, y = column(self.ds, xn).to_numpy(), column(self.ds, yn).to_numpy()
        try:
            e = column(self.ds, self.selected_one("err")).to_numpy() if \
                self.selected("err") else None
            w = make_weights(self.opts["weighting"], y, e)
        except ValueError:
            w = None
        guess = self._model.initial_guess(x, y, w)
        for r, p in enumerate(self._model.params):
            self.table.item(r, 1).setText(num(guess.get(p, 1.0)))

    def _clear(self) -> None:
        for r in range(self.table.rowCount()):
            self.table.item(r, 1).setText("")

    # -- results
    def _parameters(self):
        f = self._model
        params0, fixed, bounds = {}, {}, {}
        for r, p in enumerate(f.params):
            init, is_fixed, lo, hi = self._row(r)
            try:
                vals = [float(t.replace(",", ".")) if t else None for t in (init, lo, hi)]
            except ValueError:
                raise ValueError(f"Parameter {p}: enter numbers only.") from None
            if vals[0] is not None:
                params0[p] = vals[0]
            if is_fixed:
                fixed[p] = vals[0]
            if vals[1] is not None or vals[2] is not None:
                bounds[p] = (vals[1], vals[2])
        return params0, fixed, bounds

    def validate(self):
        if self.category.currentText() == _USER:
            try:
                fl.parse_user_function(self.user.text())
            except fl.FunctionSyntaxError as exc:
                return str(exc)
            self._user_changed()
        if self._model is None:
            return "The fitting function is not valid."
        try:
            params0, fixed, bounds = self._parameters()
        except ValueError as exc:
            return str(exc)
        for p, (lo, hi) in bounds.items():
            if lo is not None and hi is not None and lo >= hi:
                return f"Parameter {p}: the lower bound must be below the upper bound."
        if len(fixed) == len(self._model.params):
            return "At least one parameter must be free."
        self._store_table()
        return super().validate()

    def compute(self):
        params0, fixed, bounds = self._parameters()
        return self._run(self._model, fixed=fixed or None, params0=params0 or None,
                         bounds=bounds or None)


class CalibrationDialog(FormMixin, VariablePickerDialog):
    DEFAULTS = {"sigma": "residual", "blank_sd": 0.0, "weighting": "none", "replicates": 1,
                "unknown_text": "", "level": 95.0, "save": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Calibration Curve", parent)
        self.setMinimumSize(660, 520)
        self.add_target("x", "Concentration (X):", accepts=is_numeric, single=True)
        self.add_target("y", "Response (Y):", accepts=is_numeric, single=True)
        self.add_target("unknown", "Unknown Responses (optional):", accepts=is_numeric,
                        single=True, required=False)
        self.add_option_button("Options...", self._options)
        form = self.form_group("LOD / LOQ")
        self.opt_combo(form, "σ:", "sigma", cal.SIGMAS)
        self.w_blank = self.opt_spin(form, "SD of Blank:", "blank_sd", 0, 1e12, 6)
        form = self.form_group("Calibration")
        self.opt_combo(form, "Weighting:", "weighting", cal.CAL_WEIGHTS)
        self.opt_spin(form, "Replicates per Unknown (m):", "replicates", 1, 1000)
        form = self.form_group("Unknown Samples", new_row=True)
        self.opt_line(form, "Responses (typed):", "unknown_text", "e.g. 2.9, 13.5, 23.0")
        self.options_changed()
        self.finish()

    def options_changed(self):
        if hasattr(self, "w_blank"):
            self.w_blank.setEnabled(self.opts["sigma"] == "blank")

    def _options(self):
        dlg = SubDialog("Calibration Curve: Options", self, columns=1)
        box = dlg.group("Confidence Interval")
        dlg.number(box, "level", "Confidence Level:", self.opts["level"], 50, 99.99, 2, " %")
        box = dlg.group("Save")
        dlg.check(box, "save", "Predicted concentrations of the unknown variable "
                               "(CONC_n, CONCLO_n, CONCHI_n)", self.opts["save"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def validate(self):
        try:
            parse_numbers(self.opts["unknown_text"])
        except ValueError as exc:
            return f"Unknown responses: {exc}"
        if self.opts["sigma"] == "blank" and self.opts["blank_sd"] <= 0:
            return "Enter the standard deviation of the blank."
        return None

    def compute(self):
        o = self.opts
        return cal.calibration_curve(self.ds, self.selected_one("x"), self.selected_one("y"),
                                     self.selected_one("unknown"),
                                     parse_numbers(o["unknown_text"]), int(o["replicates"]),
                                     o["sigma"], o["blank_sd"], o["weighting"],
                                     o["level"] / 100, o["save"])


MENUS = {
    "Analysis/Fitting": [
        ("Fit Linear...", FitLinearDialog),
        ("Fit Polynomial...", FitPolynomialDialog),
        ("Nonlinear Curve Fit...", NonlinearFitDialog),
        None,
        ("Calibration Curve...", CalibrationDialog),
    ],
    "Analysis/Mathematics": [
        ("Differentiate...", DifferentiateDialog),
        ("Integrate...", IntegrateDialog),
        ("Interpolate/Extrapolate...", InterpolateDialog),
        ("Normalize...", NormalizeDialog),
    ],
    "Analysis/Signal Processing": [
        ("Smoothing...", SmoothDialog),
        ("FFT...", FFTDialog),
    ],
    "Analysis/Peaks and Baseline": [
        ("Baseline Correction...", BaselineDialog),
        ("Find Peaks...", FindPeaksDialog),
        ("Multiple Peak Fit...", MultiPeakFitDialog),
    ],
}


def selftest() -> None:
    """Headless check of the Analysis core on synthetic data (no widgets). Raises on failure."""
    import pandas as pd

    from rsstatchem.core.analysis import mathematics as m
    from rsstatchem.core.analysis import signal as sg
    from rsstatchem.core.analysis.fitting import fit_curve
    from rsstatchem.core.dataset import Dataset

    rng = np.random.default_rng(3)
    x = np.linspace(0.2, 10, 60)
    y = 0.5 + 4 * np.exp(-x / 2.0) + rng.normal(0, 0.02, len(x))
    res = fit_curve(x, y, "ExpDec1")
    if not (res.success and abs(res.params["t1"] - 2.0) < 5 * res.errors[2]):
        raise AssertionError(f"ExpDec1 fit failed: {res.params}")
    lo, hi = res.confidence_band(x, 0.95)
    if not np.all(lo <= res.fitted) or not np.all(res.fitted <= hi):
        raise AssertionError("confidence band does not contain the fit")
    user = fit_curve(x, y, "y = A*exp(-x/t) + y0")
    if abs(user.params["t"] - res.params["t1"]) > 1e-4:
        raise AssertionError("user-defined function disagrees with ExpDec1")
    line = fit_curve(x, 2 + 3 * x + rng.normal(0, .1, len(x)), "Line")
    if abs(line.values[1] - np.polyfit(line.x, line.y, 1)[0]) > 1e-9:
        raise AssertionError("linear fit disagrees with polyfit")
    c = cal.calibrate([0, 2, 4, 6, 8, 10, 12], [2.1, 5.0, 9.0, 12.6, 17.3, 21.0, 24.7])
    if abs(c.slope - 1.930357) > 1e-5 or abs(c.lod - 3.3 * c.s_yx / c.slope) > 1e-12:
        raise AssertionError("calibration arithmetic")
    xs = np.linspace(0, 50, 501)
    sig = 5 * np.exp(-0.5 * ((xs - 20) / 2) ** 2) + 3 * np.exp(-0.5 * ((xs - 32) / 1.5) ** 2)
    peaks = sg.find_peaks(xs, sig)
    if [round(p["x"]) for p in peaks] != [20, 32]:
        raise AssertionError("peak finding")
    mp = sg.multi_peak_fit(xs, sig + 0.3 + rng.normal(0, .01, len(xs)), 2, "gauss")
    if abs(mp.params["xc2"] - 32) > 0.05:
        raise AssertionError("multiple peak fit")
    if abs(m.integrate(xs, sig)["area"] - (5 * 2 + 3 * 1.5) * np.sqrt(2 * np.pi)) > 1e-3:
        raise AssertionError("integration")
    ds = Dataset(pd.DataFrame({"x": x, "y": y}))
    for out in (run_fit(ds, "x", "y", "ExpDec1", opts=FitOptions(save_fitted=True)),
                cal.calibration_curve(ds, "x", "y", unknown_values=[1.0]),
                m.differentiate_report(ds, "x", ["y"]), sg.smooth_report(ds, "x", ["y"]),
                sg.fft_report(ds, "x", "y"), sg.baseline_report(ds, "x", "y")):
        if not out.section.items:
            raise AssertionError("empty report")

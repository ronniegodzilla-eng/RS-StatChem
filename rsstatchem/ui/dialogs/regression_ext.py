"""Dialogs for the newer SPSS procedures of WP2:

* Analyze > Regression: Nonlinear..., Weight Estimation..., 2-Stage Least Squares...,
  Quantile..., Linear OLS Alternatives > Ridge... / Lasso... / Elastic Net...
* Analyze > Correlate: Distances... (replaces the disabled placeholder), Canonical
  Correlation...
* Analyze > Classify: ROC Analysis...
"""
from __future__ import annotations

from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QVBoxLayout)

from rsstatchem.core.dataset import NUMERIC, format_code
from rsstatchem.core.stats import cancorr, distances, regression_ext, roc_analysis
from rsstatchem.ui.dialogs.base import SubDialog, VariablePickerDialog, is_numeric
from rsstatchem.ui.dialogs.logistic import _multi


def _combo(owner, key, items):
    box = QComboBox()
    for value, text in items:
        box.addItem(text, value)
    box.setCurrentIndex(max(box.findData(owner.opts[key]), 0))
    box.currentIndexChanged.connect(lambda _: owner.opts.__setitem__(key, box.currentData()))
    return box


def _text_dialog(parent, title: str, label: str, text: str, hint: str = "") -> str | None:
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    edit = QPlainTextEdit(text)
    lay = QVBoxLayout(dlg)
    lay.addWidget(QLabel(label))
    lay.addWidget(edit)
    if hint:
        h = QLabel(hint)
        h.setWordWrap(True)
        h.setStyleSheet("color:#666")
        lay.addWidget(h)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                               | QDialogButtonBox.StandardButton.Cancel)
    buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Continue")
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    lay.addWidget(buttons)
    dlg.resize(420, 300)
    return edit.toPlainText() if dlg.exec() else None


# ============================================================================ Nonlinear
class NonlinearDialog(VariablePickerDialog):
    DEFAULTS = {"model": "", "parameters": "", "constraints": "", "loss": "",
                "bootstrap": 0, "level": 95.0, "save": []}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Nonlinear Regression", parent)
        self.add_target("dep", "Dependent:", accepts=is_numeric, single=True)
        self.model = QLineEdit(self.opts["model"])
        self.model.setPlaceholderText("e.g. a * EXP(-b * x)   (use variable and parameter "
                                      "names)")
        form = QFormLayout()
        form.addRow("Model Expression:", self.model)
        self.options_layout.addLayout(form)
        self.add_option_button("Parameters...", self._parameters)
        self.add_option_button("Loss...", self._loss)
        self.add_option_button("Constraints...", self._constraints)
        self.add_option_button("Save...", self._save)
        self.add_option_button("Options...", self._options)
        self.finish()

    def _parameters(self) -> None:
        text = _text_dialog(self, "Nonlinear Regression: Parameters",
                            "Parameters with starting values (one per line, name=value):",
                            self.opts["parameters"], "Example:\na=1\nb=0.1")
        if text is not None:
            self.opts["parameters"] = text

    def _constraints(self) -> None:
        text = _text_dialog(self, "Nonlinear Regression: Parameter Constraints",
                            "Linear constraints (one per line):", self.opts["constraints"],
                            "Examples: b >= 0   a + b <= 10   a = 2*b. Constrained models are "
                            "estimated by sequential quadratic programming.")
        if text is not None:
            self.opts["constraints"] = text

    def _loss(self) -> None:
        text = _text_dialog(self, "Nonlinear Regression: Loss Function",
                            "User-defined loss (empty = sum of squared residuals):",
                            self.opts["loss"], "Use RESID_ and PRED_, e.g. ABS(RESID_) for "
                                               "least absolute deviations.")
        if text is not None:
            self.opts["loss"] = text.strip()

    def _save(self) -> None:
        _multi(self, "Nonlinear Regression: Save New Variables", "save", [
            ("Save", [("pred", "Predicted values"), ("resid", "Residuals"),
                      ("deriv", "Derivatives"), ("loss", "Loss function values")])],
            columns=1)

    def _options(self) -> None:
        dlg = SubDialog("Nonlinear Regression: Options", self, columns=1)
        box = dlg.group("Bootstrap estimates of standard error")
        dlg.number(box, "bootstrap", "Number of bootstrap samples (0 = none):",
                   self.opts["bootstrap"], 0, 5000, 0)
        dlg.number(box, "level", "Confidence interval (%):", self.opts["level"], 50, 99.99, 2)
        values = dlg.run()
        if values:
            values["bootstrap"] = int(values["bootstrap"])
            self.opts.update(values)

    def validate(self):
        self.opts["model"] = self.model.text().strip()
        if not self.opts["model"]:
            return "Enter the model expression."
        try:
            regression_ext.compile_model(self.opts["model"])
            params = regression_ext.parse_parameters(self.opts["parameters"])
        except (ValueError, SyntaxError) as exc:
            return str(exc)
        if not params:
            return "Define the parameters and their starting values (Parameters...)."
        return None

    def compute(self):
        o = self.opts
        return regression_ext.nonlinear(
            self.ds, self.selected_one("dep"), o["model"],
            regression_ext.parse_parameters(o["parameters"]),
            [c for c in o["constraints"].splitlines() if c.strip()], o["loss"] or None,
            o["level"] / 100, int(o["bootstrap"]), save=tuple(o["save"]))


# ============================================================================ WLS
class WeightEstimationDialog(VariablePickerDialog):
    DEFAULTS = {"start": -2.0, "stop": 2.0, "step": 0.5, "constant": True,
                "each_power": False, "save_weight": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Weight Estimation", parent)
        self.add_target("dep", "Dependent:", accepts=is_numeric, single=True)
        self.add_target("ind", "Independent(s):", accepts=is_numeric)
        self.add_target("source", "Weight Variable:", accepts=is_numeric, single=True)
        self.add_option_button("Options...", self._options)
        box = QGroupBox("Power range")
        row = QHBoxLayout(box)
        for key, text in (("start", "from"), ("stop", "through"), ("step", "by")):
            from PySide6.QtWidgets import QDoubleSpinBox

            spin = QDoubleSpinBox()
            spin.setRange(-6.5, 7.5)
            spin.setSingleStep(0.5)
            spin.setValue(self.opts[key])
            spin.valueChanged.connect(lambda v, k=key: self.opts.__setitem__(k, v))
            row.addWidget(QLabel(text))
            row.addWidget(spin)
        self.options_layout.addWidget(box)
        from PySide6.QtWidgets import QCheckBox

        cb = QCheckBox("Include constant in equation")
        cb.setChecked(self.opts["constant"])
        cb.toggled.connect(lambda on: self.opts.__setitem__("constant", on))
        self.options_layout.addWidget(cb)
        self.finish()

    def _options(self) -> None:
        dlg = SubDialog("Weight Estimation: Options", self, columns=1)
        dlg.check(dlg.group("Save"), "save_weight", "Save best weight as new variable",
                  self.opts["save_weight"])
        dlg.radio(dlg.group("Display ANOVA and Estimates"), "each_power", [
            (False, "For best power"), (True, "For each power value")],
            self.opts["each_power"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def validate(self):
        if self.opts["step"] <= 0 or self.opts["stop"] < self.opts["start"]:
            return "The power range is not valid."
        return None

    def compute(self):
        o = self.opts
        return regression_ext.weight_estimation(
            self.ds, self.selected_one("dep"), self.selected("ind"),
            self.selected_one("source"), o["start"], o["stop"], o["step"], o["constant"],
            o["each_power"], o["save_weight"])


# ============================================================================ 2SLS
class TwoStageDialog(VariablePickerDialog):
    DEFAULTS = {"constant": True, "save": [], "correlations": True}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Two-Stage Least Squares", parent)
        self.add_target("dep", "Dependent:", accepts=is_numeric, single=True)
        self.add_target("expl", "Explanatory:", accepts=is_numeric)
        self.add_target("inst", "Instrumental:", accepts=is_numeric)
        self.add_option_button("Options...", self._options)
        self.finish()

    # A variable may be both explanatory and instrumental (an exogenous regressor), so the
    # Explanatory and Instrumental lists copy variables and leave them in the source list.
    def _take(self, name, target):
        from rsstatchem.ui.dialogs.base import variable_item

        if target.key not in ("expl", "inst"):
            return super()._take(name, target)
        if not self.ds.has_name(name) or not target.accepts(self.ds.var(name)) or \
                name in self.selected(target.key):
            return False
        target.widget.addItem(variable_item(self.ds.var(name)))
        return True

    def _move_back(self, target, all_items: bool = False) -> None:
        if target.key not in ("expl", "inst"):
            super()._move_back(target, all_items)
            return
        items = [target.widget.item(i) for i in range(target.widget.count())] if all_items \
            else target.widget.selectedItems()
        for item in items:
            target.widget.takeItem(target.widget.row(item))
        self._update_arrows()

    def _options(self) -> None:
        dlg = SubDialog("Two-Stage Least Squares: Options", self, columns=1)
        box = dlg.group("Save New Variables")
        for key, text in (("pred", "Predicted"), ("resid", "Residuals")):
            dlg.check(box, f"save::{key}", text, key in self.opts["save"])
        dlg.check(dlg.group("Display"), "correlations", "Display covariance of parameters "
                                                        "(correlations)",
                  self.opts["correlations"])
        dlg.check(dlg.group("Constant"), "constant", "Include constant in equation",
                  self.opts["constant"])
        values = dlg.run()
        if values:
            self.opts["save"] = [k for k in ("pred", "resid") if values.pop(f"save::{k}")]
            self.opts.update(values)

    def compute(self):
        o = self.opts
        return regression_ext.two_stage(self.ds, self.selected_one("dep"),
                                        self.selected("expl"), self.selected("inst"),
                                        o["constant"], tuple(o["save"]), o["correlations"])


# ============================================================================ Quantile
class QuantileDialog(VariablePickerDialog):
    DEFAULTS = {"quantiles": "0.5", "intercept": True, "iid": False, "bandwidth": "bofinger",
                "level": 95.0, "plot": True, "save": []}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Quantile Regression", parent)
        self.add_target("dep", "Target Variable:", accepts=is_numeric, single=True)
        self.add_target("factors", "Factor(s):", required=False)
        self.add_target("covs", "Covariate(s):", accepts=is_numeric, required=False)
        self.quantiles = QLineEdit(self.opts["quantiles"])
        form = QFormLayout()
        form.addRow("Quantiles (e.g. 0.25 0.5 0.75):", self.quantiles)
        self.options_layout.addLayout(form)
        self.add_option_button("Criteria...", self._criteria)
        self.add_option_button("Display...", self._display)
        self.add_option_button("Save...", lambda: _multi(
            self, "Quantile Regression: Save", "save", [("Save", [("pred", "Predicted values")])],
            columns=1))
        self.finish()

    def _criteria(self) -> None:
        dlg = SubDialog("Quantile Regression: Criteria", self, columns=1)
        box = dlg.group("Standard errors")
        dlg.check(box, "iid", "Assume errors are independent and identically distributed",
                  self.opts["iid"])
        dlg.radio(box, "bandwidth", [("bofinger", "Bofinger"), ("hsheather", "Hall-Sheather")],
                  self.opts["bandwidth"])
        dlg.number(box, "level", "Confidence interval (%):", self.opts["level"], 50, 99.99, 2)
        dlg.check(dlg.group("Model"), "intercept", "Include intercept", self.opts["intercept"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def _display(self) -> None:
        dlg = SubDialog("Quantile Regression: Display", self, columns=1)
        dlg.check(dlg.group("Plots"), "plot", "Parameter estimates across quantiles",
                  self.opts["plot"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def validate(self):
        try:
            qs = [float(t.replace(",", ".")) for t in self.quantiles.text().split()]
        except ValueError:
            return "Quantiles must be numbers between 0 and 1."
        if not qs or any(not 0 < q < 1 for q in qs):
            return "Quantiles must be numbers between 0 and 1."
        if not self.selected("factors") and not self.selected("covs"):
            return "Specify at least one factor or covariate."
        self.opts["quantiles"] = self.quantiles.text()
        return None

    def compute(self):
        o = self.opts
        qs = tuple(float(t.replace(",", ".")) for t in o["quantiles"].split())
        return regression_ext.quantile_regression(
            self.ds, self.selected_one("dep"), self.selected("factors"), self.selected("covs"),
            qs, o["intercept"], o["iid"], o["bandwidth"], o["level"] / 100, o["plot"],
            tuple(o["save"]))


# ============================================================================ Ridge / Lasso / EN
class _PenalizedDialog(VariablePickerDialog):
    KIND = "ridge"
    TITLE = "Linear Ridge Regression"
    DEFAULTS = {"mode": "fit", "alpha": 1.0, "l1_ratio": 0.5, "alpha_min": 0.001,
                "alpha_max": 100.0, "n_alpha": 21, "l1_list": "0.1 0.5 0.9", "folds": 5,
                "standardize": True, "holdout": 0.0, "seed": 2000000, "trace": True,
                "save": []}

    def __init__(self, ds, parent=None):
        super().__init__(ds, self.TITLE, parent)
        self.add_target("dep", "Target:", accepts=is_numeric, single=True)
        self.add_target("preds", "Predictors:", accepts=is_numeric)
        self.add_option_button("Options...", self._options)
        self.add_option_button("Partition...", self._partition)
        self.add_option_button("Save...", lambda: _multi(
            self, f"{self.TITLE}: Save", "save", [("Save", [("pred", "Predicted values")])],
            columns=1))
        form = QFormLayout()
        form.addRow("Mode:", _combo(self, "mode", [("fit", "Fit model with specified alpha"),
                                                    ("cv", "Cross-validate a grid of alphas")]))
        self.options_layout.addLayout(form)
        self.finish()

    def _options(self) -> None:
        dlg = SubDialog(f"{self.TITLE}: Options", self)
        box = dlg.group("Fit")
        dlg.number(box, "alpha", "Alpha (penalty):", self.opts["alpha"], 0, 1e6, 4)
        if self.KIND == "enet":
            dlg.number(box, "l1_ratio", "L1 ratio:", self.opts["l1_ratio"], 0.01, 1, 2)
        dlg.check(box, "standardize", "Standardize predictors", self.opts["standardize"])
        dlg.check(box, "trace", "Coefficient path plot", self.opts["trace"])
        box = dlg.group("Cross-validation grid")
        dlg.number(box, "alpha_min", "Smallest alpha:", self.opts["alpha_min"], 1e-6, 1e6, 6)
        dlg.number(box, "alpha_max", "Largest alpha:", self.opts["alpha_max"], 1e-6, 1e6, 4)
        dlg.number(box, "n_alpha", "Number of alphas (log scale):", self.opts["n_alpha"], 2,
                   200, 0)
        dlg.number(box, "folds", "Number of folds:", self.opts["folds"], 2, 50, 0)
        values = dlg.run()
        if values:
            values["n_alpha"], values["folds"] = int(values["n_alpha"]), int(values["folds"])
            self.opts.update(values)

    def _partition(self) -> None:
        dlg = SubDialog(f"{self.TITLE}: Partition", self, columns=1)
        box = dlg.group("Training and holdout")
        dlg.number(box, "holdout", "Holdout (proportion of cases):", self.opts["holdout"], 0,
                   0.9, 2)
        dlg.number(box, "seed", "Random seed:", self.opts["seed"], 1, 2 ** 31 - 1, 0)
        values = dlg.run()
        if values:
            values["seed"] = int(values["seed"])
            self.opts.update(values)

    def validate(self):
        if self.opts["alpha_min"] >= self.opts["alpha_max"]:
            return "The smallest alpha must be below the largest alpha."
        return None

    def compute(self):
        import numpy as np

        o = self.opts
        alphas = np.logspace(np.log10(o["alpha_min"]), np.log10(o["alpha_max"]),
                             int(o["n_alpha"]))
        l1s = [float(t) for t in o["l1_list"].split()] if self.KIND == "enet" else None
        return regression_ext.regularized(
            self.ds, self.selected_one("dep"), self.selected("preds"), self.KIND, o["alpha"],
            o["l1_ratio"], o["mode"], alphas, l1s, int(o["folds"]), o["standardize"],
            o["holdout"], int(o["seed"]), o["trace"], tuple(o["save"]))


class RidgeDialog(_PenalizedDialog):
    KIND, TITLE = "ridge", "Linear Ridge Regression"


class LassoDialog(_PenalizedDialog):
    KIND, TITLE = "lasso", "Linear Lasso Regression"
    DEFAULTS = dict(_PenalizedDialog.DEFAULTS, alpha=0.1)


class ElasticNetDialog(_PenalizedDialog):
    KIND, TITLE = "enet", "Linear Elastic Net Regression"
    DEFAULTS = dict(_PenalizedDialog.DEFAULTS, alpha=0.1)


# ============================================================================ Distances
class DistancesDialog(VariablePickerDialog):
    DEFAULTS = {"between": "cases", "kind": "dissimilarity", "data": "interval",
                "measure": "euclid", "power": 2.0, "root": 2.0, "present": 1.0, "absent": 0.0,
                "standardize": "none", "by": "variable", "absolute": False,
                "change_sign": False, "rescale": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Distances", parent)
        self.add_target("vars", "Variables:", accepts=is_numeric, min_items=1)
        self.add_target("label", "Label Cases by:", single=True, required=False)
        box = QGroupBox("Compute Distances / Measure")
        form = QFormLayout(box)
        form.addRow("Compute distances:", _combo(self, "between", [
            ("cases", "Between cases"), ("variables", "Between variables")]))
        form.addRow("Measure:", _combo(self, "kind", [("dissimilarity", "Dissimilarities"),
                                                      ("similarity", "Similarities")]))
        self.options_layout.addWidget(box)
        self.add_option_button("Measures...", self._measures)
        self.finish()

    def _measures(self) -> None:
        sim = self.opts["kind"] == "similarity"
        dlg = SubDialog("Distances: " + ("Similarity" if sim else "Dissimilarity")
                        + " Measures", self)
        box = dlg.group("Measure")
        groups = ([("interval", distances.INTERVAL_SIM), ("binary", distances.BINARY_SIM)]
                  if sim else [("interval", distances.INTERVAL_DISSIM),
                               ("counts", distances.COUNT),
                               ("binary", distances.BINARY_DISSIM)])
        items = [(k, f"{g.capitalize()}: {v}") for g, d in groups for k, v in d.items()]
        combo = QComboBox()
        for k, text in items:
            combo.addItem(text, k)
        combo.setCurrentIndex(max(combo.findData(self.opts["measure"]), 0))
        box.addWidget(combo)
        dlg.number(box, "power", "Power:", self.opts["power"], 1, 10, 1)
        dlg.number(box, "root", "Root:", self.opts["root"], 1, 10, 1)
        dlg.number(box, "present", "Binary — Present:", self.opts["present"], -1e6, 1e6, 2)
        dlg.number(box, "absent", "Binary — Absent:", self.opts["absent"], -1e6, 1e6, 2)
        box = dlg.group("Transform Values")
        dlg.radio(box, "standardize", list(distances.STANDARDIZE.items()),
                  self.opts["standardize"])
        dlg.radio(box, "by", [("variable", "By variable"), ("case", "By case")],
                  self.opts["by"])
        box = dlg.group("Transform Measures")
        dlg.check(box, "absolute", "Absolute values", self.opts["absolute"])
        dlg.check(box, "change_sign", "Change sign", self.opts["change_sign"])
        dlg.check(box, "rescale", "Rescale to 0-1 range", self.opts["rescale"])
        values = dlg.run()
        if values:
            self.opts.update(values)
            self.opts["measure"] = combo.currentData()

    def validate(self):
        sim_measure = self.opts["measure"] in distances.SIMILARITIES
        if sim_measure != (self.opts["kind"] == "similarity"):
            self.opts["measure"] = "correlation" if self.opts["kind"] == "similarity" \
                else "euclid"
        if self.opts["between"] == "variables" and len(self.selected("vars")) < 2:
            return "Select at least two variables to compute distances between variables."
        return None

    def compute(self):
        o = self.opts
        return distances.distances(
            self.ds, self.selected("vars"), o["between"], o["measure"], o["standardize"],
            o["by"], o["power"], o["root"], o["present"], o["absent"], o["absolute"],
            o["change_sign"], o["rescale"], self.selected_one("label"))


# ============================================================================ Canonical
class CanonicalDialog(VariablePickerDialog):
    DEFAULTS = {"ncomp": 0, "save": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Canonical Correlation", parent)
        self.add_target("set1", "Set 1:", accepts=is_numeric)
        self.add_target("set2", "Set 2:", accepts=is_numeric)
        self.add_option_button("Options...", self._options)
        self.finish()

    def _options(self) -> None:
        dlg = SubDialog("Canonical Correlation: Options", self, columns=1)
        box = dlg.group("Options")
        dlg.number(box, "ncomp", "Number of canonical correlations (0 = all):",
                   self.opts["ncomp"], 0, 50, 0)
        dlg.check(box, "save", "Save canonical variate scores", self.opts["save"])
        values = dlg.run()
        if values:
            values["ncomp"] = int(values["ncomp"])
            self.opts.update(values)

    def compute(self):
        o = self.opts
        return cancorr.canonical_correlation(self.ds, self.selected("set1"),
                                             self.selected("set2"), o["ncomp"] or None,
                                             o["save"])


# ============================================================================ ROC Analysis
class RocAnalysisDialog(VariablePickerDialog):
    DEFAULTS = {"positive": "", "larger_positive": True, "distribution": "free",
                "level": 95.0, "plot": True, "pr": True, "paired": True, "metrics": True,
                "coordinates": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "ROC Analysis", parent)
        self.add_target("tests", "Test Variables:", accepts=is_numeric)
        self.add_target("state", "State Variable:", single=True)
        self.add_target("group", "Grouping Variable:", single=True, required=False)
        self.positive = QLineEdit(self.opts["positive"])
        form = QFormLayout()
        form.addRow("Positive Class (value of state variable):", self.positive)
        self.options_layout.addLayout(form)
        self.add_option_button("Options...", self._options)
        self.add_option_button("Display...", self._display)
        self.finish()

    def targets_changed(self) -> None:
        if not hasattr(self, "positive"):
            return
        state = self.selected_one("state")
        if state and not self.positive.text().strip():
            codes = list(self.ds.var(state).value_labels)
            if len(codes) >= 2:
                self.positive.setText(format_code(max(codes)))

    def _options(self) -> None:
        dlg = SubDialog("ROC Analysis: Options", self, columns=1)
        dlg.radio(dlg.group("Test Direction"), "larger_positive", [
            (True, "Larger test result indicates more positive test"),
            (False, "Smaller test result indicates more positive test")],
            self.opts["larger_positive"])
        box = dlg.group("Standard Error and Confidence Interval")
        dlg.radio(box, "distribution", [("free", "Nonparametric"),
                                        ("negexpo", "Bi-negative exponential")],
                  self.opts["distribution"])
        dlg.number(box, "level", "Confidence level (%):", self.opts["level"], 50, 99.99, 2)
        values = dlg.run()
        if values:
            self.opts.update(values)

    def _display(self) -> None:
        dlg = SubDialog("ROC Analysis: Display", self, columns=1)
        box = dlg.group("Plots and statistics")
        for key, text in (("plot", "ROC curve"), ("pr", "Precision-recall curve"),
                          ("paired", "Paired-sample area difference"),
                          ("metrics", "Classifier evaluation metrics"),
                          ("coordinates", "Coordinate points")):
            dlg.check(box, key, text, self.opts[key])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def validate(self):
        text = self.positive.text().strip()
        if not text:
            return "Enter the value of the state variable that indicates the positive class."
        if self.ds.var(self.selected_one("state")).type == NUMERIC:
            try:
                float(text.replace(",", "."))
            except ValueError:
                return "The positive class must be a number."
        self.opts["positive"] = text
        return None

    def compute(self):
        o = self.opts
        return roc_analysis.roc_analysis(
            self.ds, self.selected("tests"), self.selected_one("state"), o["positive"],
            self.selected_one("group"), o["larger_positive"], o["distribution"],
            o["level"] / 100, o["plot"], o["pr"], o["paired"], o["metrics"],
            o["coordinates"])


MENUS = {
    "Analyze/Correlate": [("Distances...", DistancesDialog),
                          ("Canonical Correlation...", CanonicalDialog)],
    "Analyze/Regression": [("Nonlinear...", NonlinearDialog),
                           ("Weight Estimation...", WeightEstimationDialog),
                           ("2-Stage Least Squares...", TwoStageDialog),
                           ("Quantile...", QuantileDialog),
                           ("Linear OLS Alternatives", [("Ridge...", RidgeDialog),
                                                        ("Lasso...", LassoDialog),
                                                        ("Elastic Net...", ElasticNetDialog)])],
    "Analyze/Classify": [("ROC Analysis...", RocAnalysisDialog)],
}


def selftest() -> None:
    """Headless check of regression_ext, distances, cancorr and roc_analysis."""
    regression_ext.selftest()
    distances.selftest()
    cancorr.selftest()
    roc_analysis.selftest()

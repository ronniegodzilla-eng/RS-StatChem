"""Analyze > Regression (Binary Logistic, Multinomial Logistic, Ordinal, Curve Estimation),
Analyze > Compare Means > Means and Analyze > ROC Curve dialogs, laid out like SPSS.

``MENUS`` also carries the Analyze > Classify entries from :mod:`rsstatchem.ui.dialogs.classify`
so the integrator can merge one dictionary; ``selftest()`` exercises all of the core code.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QLineEdit, QRadioButton)

from rsstatchem.core.dataset import NUMERIC, format_code
from rsstatchem.core.stats import curvefit_spss, logistic, means, roc
from rsstatchem.ui.dialogs.base import (BASE_TEXT_ROLE, NAME_ROLE, SubDialog, VariablePickerDialog,
                                        is_numeric)


def _combo(owner: VariablePickerDialog, key: str, items: list[tuple[str, str]]) -> QComboBox:
    box = QComboBox()
    for value, text in items:
        box.addItem(text, value)
    box.setCurrentIndex(max(box.findData(owner.opts[key]), 0))
    box.currentIndexChanged.connect(lambda _: owner.opts.__setitem__(key, box.currentData()))
    return box


def _check(owner: VariablePickerDialog, key: str, text: str) -> QCheckBox:
    cb = QCheckBox(text)
    cb.setChecked(bool(owner.opts[key]))
    cb.toggled.connect(lambda on: owner.opts.__setitem__(key, on))
    return cb


def _multi(owner: VariablePickerDialog, title: str, key: str,
           groups: list[tuple[str, list[tuple[str, str]]]], extra=None, columns=2) -> None:
    """Checkbox sub-dialog whose ticked keys are stored as a list in ``owner.opts[key]``."""
    dlg = SubDialog(title, owner, columns=columns)
    for group_title, items in groups:
        box = dlg.group(group_title)
        for item_key, text in items:
            dlg.check(box, item_key, text, item_key in owner.opts[key])
    if extra:
        extra(dlg)
    values = dlg.run()
    if values is None:
        return
    keys = [k for _, items in groups for k, _ in items]
    owner.opts[key] = [k for k in keys if values.get(k)]
    for k, v in values.items():
        if k not in keys:
            owner.opts[k] = v


# ============================================================================ binary logistic
class BinaryLogisticDialog(VariablePickerDialog):
    DEFAULTS = {"method": "enter", "categorical": {}, "contrast": "indicator",
                "reference": "last", "save": [], "hosmer": True, "ci": False, "level": 95.0,
                "iteration_history": False, "each_step": True, "p_enter": 0.05,
                "p_remove": 0.10, "cut": 0.5, "max_iter": 20, "constant": True}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Logistic Regression", parent)
        self.add_target("dep", "Dependent:", single=True)
        self.add_target("cov", "Covariates:")
        self.add_option_button("Categorical...", self._categorical)
        self.add_option_button("Save...", self._save)
        self.add_option_button("Options...", self._options)
        form = QFormLayout()
        form.addRow("Method:", _combo(self, "method", list(logistic.METHODS.items())))
        self.options_layout.addLayout(form)
        self.finish()

    def targets_changed(self) -> None:
        if not hasattr(self, "_targets") or not self._targets:
            return
        covs = self.selected("cov") if len(self._targets) > 1 else []
        cats = self.opts["categorical"]
        for name in covs:
            if self.ds.var(name).type != NUMERIC:
                cats.setdefault(name, [self.opts["contrast"], self.opts["reference"]])
        widget = self._targets[1].widget if len(self._targets) > 1 else None
        if widget is None:
            return
        for i in range(widget.count()):
            item = widget.item(i)
            name = item.data(NAME_ROLE)
            if name in cats:
                c, r = cats[name]
                item.setText(f"{name}(Cat: {c[:3].capitalize()}"
                             f"{'(first)' if r == 'first' else ''})")
            else:
                item.setText(item.data(BASE_TEXT_ROLE) or name)

    def _categorical(self) -> None:
        covs = self.selected("cov")
        if not covs:
            return
        dlg = SubDialog("Logistic Regression: Define Categorical Variables", self)
        box = dlg.group("Categorical Covariates")
        for name in covs:
            dlg.check(box, f"cat::{name}", self.ds.var(name).title,
                      name in self.opts["categorical"]
                      or self.ds.var(name).type != NUMERIC)
        box = dlg.group("Change Contrast")
        dlg.radio(box, "contrast", [("indicator", "Indicator"), ("deviation", "Deviation")],
                  self.opts["contrast"])
        dlg.radio(box, "reference", [("last", "Reference Category: Last"),
                                     ("first", "Reference Category: First")],
                  self.opts["reference"])
        values = dlg.run()
        if values is None:
            return
        self.opts["contrast"], self.opts["reference"] = values["contrast"], values["reference"]
        self.opts["categorical"] = {n: [values["contrast"], values["reference"]] for n in covs
                                    if values.get(f"cat::{n}")}
        self.targets_changed()

    def _save(self) -> None:
        _multi(self, "Logistic Regression: Save", "save", [
            ("Predicted Values", [("prob", "Probabilities"), ("group", "Group membership")])])

    def _options(self) -> None:
        dlg = SubDialog("Logistic Regression: Options", self)
        box = dlg.group("Statistics and Plots")
        dlg.check(box, "hosmer", "Hosmer-Lemeshow goodness-of-fit", self.opts["hosmer"])
        dlg.check(box, "iteration_history", "Iteration history",
                  self.opts["iteration_history"])
        dlg.check(box, "ci", "CI for exp(B):", self.opts["ci"])
        dlg.number(box, "level", "Level (%):", self.opts["level"], 50, 99.99, 2)
        box = dlg.group("Display")
        dlg.radio(box, "each_step", [(True, "At each step"), (False, "At last step")],
                  self.opts["each_step"])
        box = dlg.group("Probability for Stepwise")
        dlg.number(box, "p_enter", "Entry:", self.opts["p_enter"], 0.0001, 0.99, 4)
        dlg.number(box, "p_remove", "Removal:", self.opts["p_remove"], 0.0001, 0.99, 4)
        box = dlg.group("Criteria")
        dlg.number(box, "cut", "Classification cutoff:", self.opts["cut"], 0.01, 0.99, 2)
        dlg.number(box, "max_iter", "Maximum Iterations:", self.opts["max_iter"], 1, 999, 0)
        dlg.check(box, "constant", "Include constant in model", self.opts["constant"])
        values = dlg.run()
        if values:
            values["max_iter"] = int(values["max_iter"])
            if values["p_enter"] >= values["p_remove"]:
                values["p_remove"] = min(0.99, values["p_enter"] + 0.05)
            self.opts.update(values)

    def validate(self):
        if self.selected_one("dep") in self.selected("cov"):
            return "The dependent variable cannot also be a covariate."
        return None

    def compute(self):
        o = self.opts
        covs = self.selected("cov")
        cats = {n: tuple(v) for n, v in o["categorical"].items() if n in covs}
        return logistic.binary_logistic(
            self.ds, self.selected_one("dep"), covs, cats, o["method"], o["cut"], o["p_enter"],
            o["p_remove"], int(o["max_iter"]), o["constant"], o["hosmer"], o["ci"],
            self.level(), o["iteration_history"], o["each_step"], tuple(o["save"]))


# ============================================================================ multinomial
class MultinomialDialog(VariablePickerDialog):
    DEFAULTS = {"reference": "last", "custom": "", "classification": True, "gof": True,
                "lr_tests": True, "level": 95.0, "save": []}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Multinomial Logistic Regression", parent)
        self.add_target("dep", "Dependent:", single=True)
        self.add_target("factors", "Factor(s):", required=False)
        self.add_target("covs", "Covariate(s):", accepts=is_numeric, required=False)
        self.ref_button = self.add_option_button("Reference Category...", self._reference)
        self.add_option_button("Statistics...", self._statistics)
        self.add_option_button("Save...", self._save)
        self.finish()

    def targets_changed(self) -> None:
        dep = self.selected_one("dep") if self._targets else None
        if dep:
            ref = self.opts["reference"]
            text = {"last": "Last", "first": "First"}.get(ref, f"Custom: {self.opts['custom']}")
            self.set_item_suffix("dep", f"({text})")

    def _reference(self) -> None:
        dlg = SubDialog("Multinomial Logistic Regression: Reference Category", self, columns=1)
        box = dlg.group("Reference Category")
        dlg.radio(box, "reference", [("first", "First Category"), ("last", "Last Category"),
                                     ("custom", "Custom")],
                  self.opts["reference"] if self.opts["reference"] in ("first", "last")
                  else "custom")
        edit = QLineEdit(self.opts["custom"])
        form = QFormLayout()
        form.addRow("Value:", edit)
        box.addLayout(form)
        dep = self.selected_one("dep")
        if dep and self.ds.var(dep).value_labels:
            from PySide6.QtWidgets import QLabel

            hint = QLabel(", ".join(f"{format_code(k)} = {v}"
                                    for k, v in self.ds.var(dep).value_labels.items()))
            hint.setWordWrap(True)
            hint.setStyleSheet("color:#666")
            box.addWidget(hint)
        values = dlg.run()
        if values:
            self.opts["reference"] = values["reference"]
            self.opts["custom"] = edit.text().strip()
            self.targets_changed()

    def _statistics(self) -> None:
        dlg = SubDialog("Multinomial Logistic Regression: Statistics", self)
        box = dlg.group("Model")
        dlg.check(box, "gof", "Goodness-of-fit", self.opts["gof"])
        dlg.check(box, "classification", "Classification table", self.opts["classification"])
        box = dlg.group("Parameters")
        dlg.check(box, "lr_tests", "Likelihood ratio tests", self.opts["lr_tests"])
        dlg.number(box, "level", "Confidence interval (%):", self.opts["level"], 50, 99.99, 2)
        values = dlg.run()
        if values:
            self.opts.update(values)

    def _save(self) -> None:
        _multi(self, "Multinomial Logistic Regression: Save", "save", [
            ("Saved Variables", [("est", "Estimated response probability (first category)"),
                                 ("pre", "Predicted category"),
                                 ("max", "Predicted category probability")])], columns=1)

    def validate(self):
        if not self.selected("factors") and not self.selected("covs"):
            return "Specify at least one factor or covariate."
        if self.opts["reference"] == "custom" and not self.opts["custom"]:
            return "Enter the value of the custom reference category."
        return None

    def compute(self):
        o = self.opts
        ref = o["custom"] if o["reference"] == "custom" else o["reference"]
        return logistic.multinomial(self.ds, self.selected_one("dep"), self.selected("factors"),
                                    self.selected("covs"), ref, self.level(),
                                    o["classification"], o["gof"], o["lr_tests"],
                                    tuple(o["save"]))


# ============================================================================ ordinal
class OrdinalDialog(VariablePickerDialog):
    DEFAULTS = {"link": "logit", "gof": True, "parallel": False, "classification": False,
                "level": 95.0, "save": []}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Ordinal Regression", parent)
        self.add_target("dep", "Dependent:", single=True)
        self.add_target("factors", "Factor(s):", required=False)
        self.add_target("covs", "Covariate(s):", accepts=is_numeric, required=False)
        self.add_option_button("Options...", self._options)
        self.add_option_button("Output...", self._output)
        self.finish()

    def _options(self) -> None:
        dlg = SubDialog("Ordinal Regression: Options", self, columns=1)
        box = dlg.group("Link")
        dlg.radio(box, "link", list(logistic.LINKS.items()), self.opts["link"])
        dlg.number(dlg.group("Confidence interval"), "level", "Confidence interval (%):",
                   self.opts["level"], 50, 99.99, 2)
        values = dlg.run()
        if values:
            self.opts.update(values)

    def _output(self) -> None:
        dlg = SubDialog("Ordinal Regression: Output", self)
        box = dlg.group("Display")
        dlg.check(box, "gof", "Goodness of fit statistics", self.opts["gof"])
        dlg.check(box, "parallel", "Test of parallel lines", self.opts["parallel"])
        dlg.check(box, "classification", "Classification table (predicted vs actual)",
                  self.opts["classification"])
        box = dlg.group("Saved Variables")
        for key, text in (("est", "Estimated response probability (first category)"),
                          ("pre", "Predicted category"),
                          ("max", "Predicted category probability")):
            dlg.check(box, f"save::{key}", text, key in self.opts["save"])
        values = dlg.run()
        if values:
            self.opts["save"] = [k for k in ("est", "pre", "max") if values.pop(f"save::{k}")]
            self.opts.update(values)

    def validate(self):
        if not self.selected("factors") and not self.selected("covs"):
            return "Specify at least one factor or covariate."
        return None

    def compute(self):
        o = self.opts
        return logistic.ordinal(self.ds, self.selected_one("dep"), self.selected("factors"),
                                self.selected("covs"), o["link"], self.level(), o["gof"],
                                o["parallel"], o["classification"], tuple(o["save"]))


# ============================================================================ curve estimation
class CurveEstimationDialog(VariablePickerDialog):
    DEFAULTS = {"models": ["linear"], "constant": True, "plot": True, "anova": False,
                "use_time": False, "upper": ""}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Curve Estimation", parent)
        self.add_target("deps", "Dependent(s):", accepts=is_numeric)
        self.add_target("ind", "Independent Variable:", accepts=is_numeric, single=True,
                        required=False)
        row = QHBoxLayout()
        self.var_radio = QRadioButton("Variable")
        self.time_radio = QRadioButton("Time")
        group = QButtonGroup(self)
        group.addButton(self.var_radio)
        group.addButton(self.time_radio)
        (self.time_radio if self.opts["use_time"] else self.var_radio).setChecked(True)
        self.time_radio.toggled.connect(lambda on: self.opts.__setitem__("use_time", on))
        row.addWidget(self.var_radio)
        row.addWidget(self.time_radio)
        row.addStretch()
        row.addWidget(_check(self, "constant", "Include constant in equation"))
        row.addWidget(_check(self, "plot", "Plot models"))
        self.options_layout.addLayout(row)
        box = QGroupBox("Models")
        grid = QGridLayout(box)
        self._model_checks = {}
        for i, (key, text) in enumerate(curvefit_spss.MODELS.items()):
            cb = QCheckBox(text)
            cb.setChecked(key in self.opts["models"])
            cb.toggled.connect(self._models_changed)
            self._model_checks[key] = cb
            grid.addWidget(cb, i % 4, i // 4)
        self.upper = QLineEdit(self.opts["upper"])
        self.upper.setPlaceholderText("infinity")
        self.upper.setMaximumWidth(110)
        form = QFormLayout()
        form.addRow("Upper bound:", self.upper)
        grid.addLayout(form, 3, 3)
        self.options_layout.addWidget(box)
        self.options_layout.addWidget(_check(self, "anova", "Display ANOVA table"))
        self.finish()

    def _models_changed(self) -> None:
        self.opts["models"] = [k for k, cb in self._model_checks.items() if cb.isChecked()]

    def validate(self):
        if not self.opts["models"]:
            return "Select at least one model."
        if not self.opts["use_time"] and not self.selected_one("ind"):
            return "Select an independent variable, or choose Time."
        text = self.upper.text().strip().replace(",", ".")
        if text:
            try:
                float(text)
            except ValueError:
                return "Upper bound must be a number."
        self.opts["upper"] = text
        return None

    def compute(self):
        o = self.opts
        upper = float(o["upper"]) if o["upper"] else None
        return curvefit_spss.curve_estimation(
            self.ds, self.selected("deps"), None if o["use_time"] else self.selected_one("ind"),
            tuple(o["models"]), o["constant"], o["plot"], upper, o["anova"])


# ============================================================================ means
class MeansDialog(VariablePickerDialog):
    DEFAULTS = {"statistics": list(means.DEFAULT), "anova": False, "linearity": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Means", parent)
        self.add_target("deps", "Dependent List:", accepts=is_numeric)
        self.add_target("layer1", "Layer 1 — Independent List:", required=False)
        self.add_target("layer2", "Layer 2 — Independent List:", required=False)
        self.add_option_button("Options...", self._options)
        self.finish()

    def _options(self) -> None:
        dlg = SubDialog("Means: Options", self)
        box = dlg.group("Cell Statistics")
        for key, text in means.STATISTICS.items():
            dlg.check(box, key, text, key in self.opts["statistics"])
        box = dlg.group("Statistics for First Layer")
        dlg.check(box, "anova", "Anova table and eta", self.opts["anova"])
        dlg.check(box, "linearity", "Test for linearity", self.opts["linearity"])
        values = dlg.run()
        if values:
            self.opts["statistics"] = [k for k in means.STATISTICS if values.get(k)]
            self.opts["anova"] = values["anova"] or values["linearity"]
            self.opts["linearity"] = values["linearity"]

    def compute(self):
        o = self.opts
        return means.means(self.ds, self.selected("deps"),
                           [self.selected("layer1"), self.selected("layer2")],
                           tuple(o["statistics"]), o["anova"], o["linearity"])


# ============================================================================ ROC
class RocDialog(VariablePickerDialog):
    DEFAULTS = {"plot": True, "reference_line": True, "se_ci": True, "coordinates": False,
                "larger_positive": True, "distribution": "free", "level": 95.0,
                "youden": False, "positive": ""}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "ROC Curve", parent)
        self.add_target("tests", "Test Variable:", accepts=is_numeric)
        self.add_target("state", "State Variable:", single=True)
        self.positive = QLineEdit(self.opts["positive"])
        form = QFormLayout()
        form.addRow("Value of State Variable:", self.positive)
        self.options_layout.addLayout(form)
        box = QGroupBox("Display")
        grid = QGridLayout(box)
        grid.addWidget(_check(self, "plot", "ROC Curve"), 0, 0)
        grid.addWidget(_check(self, "reference_line", "With diagonal reference line"), 0, 1)
        grid.addWidget(_check(self, "se_ci", "Standard error and confidence interval"), 1, 0)
        grid.addWidget(_check(self, "coordinates", "Coordinate points of the ROC Curve"), 1, 1)
        self.options_layout.addWidget(box)
        self.add_option_button("Options...", self._options)
        self.finish()

    def targets_changed(self) -> None:
        if not hasattr(self, "positive"):
            return
        state = self.selected_one("state")
        if state and not self.positive.text().strip():
            var = self.ds.var(state)
            codes = list(var.value_labels)
            if len(codes) >= 2:
                self.positive.setText(format_code(max(codes)))

    def _options(self) -> None:
        dlg = SubDialog("ROC Curve: Options", self)
        box = dlg.group("Test Direction")
        dlg.radio(box, "larger_positive", [
            (True, "Larger test result indicates more positive test"),
            (False, "Smaller test result indicates more positive test")],
            self.opts["larger_positive"])
        box = dlg.group("Parameters for Standard Error of Area")
        dlg.radio(box, "distribution", [("free", "Nonparametric (DeLong)"),
                                        ("negexpo", "Bi-negative exponential "
                                                    "(Hanley-McNeil)")],
                  self.opts["distribution"])
        dlg.number(box, "level", "Confidence level (%):", self.opts["level"], 50, 99.99, 2)
        box = dlg.group("Extra")
        dlg.check(box, "youden", "Optimal cut-off (Youden index)", self.opts["youden"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def validate(self):
        text = self.positive.text().strip()
        if not text:
            return "Enter the Value of State Variable that indicates a positive state."
        var = self.ds.var(self.selected_one("state"))
        if var.type == NUMERIC:
            try:
                float(text.replace(",", "."))
            except ValueError:
                return "Value of State Variable must be a number."
        self.opts["positive"] = text
        return None

    def compute(self):
        o = self.opts
        return roc.roc(self.ds, self.selected("tests"), self.selected_one("state"),
                       o["positive"], o["larger_positive"], o["distribution"], o["plot"],
                       o["reference_line"], o["se_ci"], o["coordinates"], self.level(),
                       o["youden"])


# ============================================================================ registry
def _menus() -> dict[str, list]:
    from rsstatchem.ui.dialogs.classify import (DiscriminantDialog, HierarchicalDialog,
                                                KMeansDialog)

    return {
        "Analyze/Compare Means": [("Means...", MeansDialog)],
        "Analyze/Regression": [None, ("Binary Logistic...", BinaryLogisticDialog),
                               ("Multinomial Logistic...", MultinomialDialog),
                               ("Ordinal...", OrdinalDialog),
                               ("Curve Estimation...", CurveEstimationDialog)],
        "Analyze/Classify": [("K-Means Cluster...", KMeansDialog),
                             ("Hierarchical Cluster...", HierarchicalDialog),
                             ("Discriminant...", DiscriminantDialog)],
        "Analyze": [("ROC Curve...", RocDialog)],
    }


MENUS = _menus()


def selftest() -> None:
    """Headless check of the logistic, curve estimation, means, ROC and classify cores."""
    import numpy as np
    import pandas as pd

    from rsstatchem.core.dataset import Dataset
    from rsstatchem.core.stats import classify

    rng = np.random.default_rng(7)
    n = 120
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    g = rng.integers(1, 4, n).astype(float)
    eta = 0.8 * x1 - 0.6 * x2 + 0.5 * (g == 2)
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    o = np.digitize(eta + rng.logistic(size=n), [-1.0, 0.0, 1.0]) + 1.0
    ds = Dataset(pd.DataFrame({"y": y, "x1": x1, "x2": x2, "g": g, "o": o,
                               "t": np.arange(1, n + 1, dtype=float),
                               "z": np.exp(0.02 * np.arange(1, n + 1)) + rng.random(n)}))
    res = logistic.binary_logistic(ds, "y", ["x1", "x2", "g"], {"g": ("indicator", "first")},
                                   method="fstep_lr", save=("prob", "group"))
    assert res.saved and any(t.title.startswith("Model Summary") for t in res.section.items
                             if hasattr(t, "rows"))
    logistic.binary_logistic(ds, "y", ["x1", "g"], {"g": ("deviation", "last")},
                             method="bstep_wald", iteration_history=True, ci=True)
    logistic.multinomial(ds, "o", ["g"], ["x1", "x2"], save=("pre",))
    logistic.ordinal(ds, "o", ["g"], ["x1", "x2"], link="cloglog", parallel=True)
    curvefit_spss.curve_estimation(ds, ["z"], "t", tuple(curvefit_spss.MODELS))
    means.means(ds, ["x1", "x2"], [["g"], ["y"]], ("mean", "count", "stddev", "median"),
                anova=True, linearity=True)
    sec = roc.roc(ds, ["x1", "x2"], "y", 1.0, coordinates=True)
    assert any(getattr(i, "kind", "") == "image" for i in sec.items)
    classify.kmeans_cluster(ds, ["x1", "x2"], 3, save=("cluster", "distance"))
    classify.hierarchical(ds, ["x1", "x2"], "ward", "seuclid", "z", solution=(2, 4))
    classify.discriminant(ds, "g", 1, 3, ["x1", "x2"], method="stepwise", f_enter=0.5,
                          f_remove=0.2, fisher=True)

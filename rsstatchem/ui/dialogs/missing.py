"""Analyze > Missing Value Analysis and Analyze > Multiple Imputation dialogs (SPSS MVA and
MULTIPLE IMPUTATION), plus the Pooled Analysis entries that pool results over imputations.
Core code: :mod:`rsstatchem.core.stats.missing` and :mod:`rsstatchem.core.stats.imputation`."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFormLayout, QLabel, QLineEdit, QSpinBox,
                               QTableWidget, QTableWidgetItem, QVBoxLayout)

from rsstatchem.core.dataset import NUMERIC, format_code, is_valid_name
from rsstatchem.core.stats import imputation as mi
from rsstatchem.core.stats import missing as mv
from rsstatchem.ui.dialogs.base import SubDialog, VariablePickerDialog, is_numeric
from rsstatchem.ui.dialogs.compare import GroupingMixin


# =========================================================================== MVA
class MissingValueAnalysisDialog(VariablePickerDialog):
    DEFAULTS = {"listwise": True, "pairwise": False, "em": True, "regression": False,
                "univariate": True, "ttest": False, "crosstabs": False, "threshold": 5.0,
                "patterns": True, "case_patterns": False, "em_iter": 25,
                "residual": "normal"}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Missing Value Analysis", parent)
        self.add_target("quant", "Quantitative Variables:", accepts=is_numeric, required=False)
        self.add_target("cat", "Categorical Variables:", required=False)
        self.add_option_button("Patterns...", self._patterns)
        self.add_option_button("Descriptives...", self._descriptives)
        self.add_option_button("EM...", self._em)
        self.add_option_button("Regression...", self._regression)
        label = QLabel("Estimation")
        label.setStyleSheet("font-weight:600")
        self.options_layout.addWidget(label)
        for key, text in (("listwise", "Listwise"), ("pairwise", "Pairwise"), ("em", "EM"),
                          ("regression", "Regression")):
            self.add_checkbox(key, text)
        self.finish()

    def _patterns(self):
        dlg = SubDialog("Missing Value Analysis: Patterns", self, columns=1)
        box = dlg.group("Display")
        dlg.check(box, "patterns", "Tabulated cases, grouped by missing value patterns",
                  self.opts["patterns"])
        dlg.check(box, "case_patterns", "Cases with missing values", self.opts["case_patterns"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def _descriptives(self):
        dlg = SubDialog("Missing Value Analysis: Descriptives", self, columns=1)
        box = dlg.group("")
        dlg.check(box, "univariate", "Univariate statistics", self.opts["univariate"])
        box = dlg.group("Indicator Variable Statistics")
        dlg.check(box, "ttest", "t tests with groups formed by indicator variables",
                  self.opts["ttest"])
        dlg.check(box, "crosstabs", "Crosstabulations of categorical and indicator variables",
                  self.opts["crosstabs"])
        dlg.number(box, "threshold", "Omit variables missing less than % of cases:",
                   self.opts["threshold"], 0, 100, 1)
        values = dlg.run()
        if values:
            self.opts.update(values)

    def _em(self):
        dlg = SubDialog("Missing Value Analysis: EM", self, columns=1)
        box = dlg.group("Distribution: Normal")
        dlg.number(box, "em_iter", "Maximum iterations:", self.opts["em_iter"], 1, 9999, 0)
        values = dlg.run()
        if values:
            self.opts["em_iter"] = int(values["em_iter"])

    def _regression(self):
        dlg = SubDialog("Missing Value Analysis: Regression", self, columns=1)
        box = dlg.group("Estimation Adjustment")
        dlg.radio(box, "residual", [("residual", "Residuals"), ("normal", "Normal variates"),
                                    ("none", "No adjustment")], self.opts["residual"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def validate(self):
        if not self.selected("quant") and not self.selected("cat"):
            return "Select at least one variable."
        return None

    def compute(self):
        o = self.opts
        est = tuple(k for k in ("listwise", "pairwise", "em", "regression") if o[k])
        return mv.mva(self.ds, self.selected("quant"), self.selected("cat"), est,
                      o["univariate"], o["ttest"], o["crosstabs"], o["threshold"],
                      o["patterns"], o["case_patterns"], int(o["em_iter"]), o["residual"])


# =========================================================================== MI: patterns
class AnalyzePatternsDialog(VariablePickerDialog):
    DEFAULTS = {"summary": True, "patterns": True, "variables": True, "min_pct": 10.0,
                "max_vars": 25}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Analyze Patterns", parent)
        self.add_target("vars", "Analyze Across Variables:")
        label = QLabel("Output")
        label.setStyleSheet("font-weight:600")
        self.options_layout.addWidget(label)
        self.add_checkbox("summary", "Summary of missingness")
        self.add_checkbox("patterns", "Patterns of missing values")
        self.add_checkbox("variables", "Variables with the highest frequency of missing values")
        form = QFormLayout()
        self.max_spin = QSpinBox()
        self.max_spin.setRange(1, 1000)
        self.max_spin.setValue(int(self.opts["max_vars"]))
        self.min_spin = QDoubleSpinBox()
        self.min_spin.setRange(0, 100)
        self.min_spin.setValue(float(self.opts["min_pct"]))
        form.addRow("Maximum number of variables displayed:", self.max_spin)
        form.addRow("Minimum percentage missing for variable to be displayed:", self.min_spin)
        self.options_layout.addLayout(form)
        self.finish()

    def compute(self):
        self.opts["max_vars"] = self.max_spin.value()
        self.opts["min_pct"] = self.min_spin.value()
        o = self.opts
        return mi.analyze_patterns(self.ds, self.selected("vars"), o["summary"], o["patterns"],
                                   o["variables"], o["min_pct"], o["max_vars"])


# =========================================================================== MI: impute
class ConstraintsDialog(QDialog):
    def __init__(self, names: list[str], current: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Impute Missing Data Values: Constraints")
        self.table = QTableWidget(len(names), 5)
        self.table.setHorizontalHeaderLabels(["Variable", "Role", "Minimum", "Maximum",
                                              "Rounding"])
        self.table.verticalHeader().hide()
        self.names = names
        self.roles = []
        for r, name in enumerate(names):
            cur = current.get(name, {})
            item = QTableWidgetItem(name)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(r, 0, item)
            combo = QComboBox()
            for key, text in (("both", "Impute and use as predictor"),
                              ("dependent", "Impute only"), ("predictor", "Use as predictor only")):
                combo.addItem(text, key)
            combo.setCurrentIndex(max(combo.findData(cur.get("role", "both")), 0))
            self.table.setCellWidget(r, 1, combo)
            self.roles.append(combo)
            for c, key in ((2, "minimum"), (3, "maximum"), (4, "rounding")):
                v = cur.get(key)
                self.table.setItem(r, c, QTableWidgetItem("" if v is None else format_code(v)))
        self.table.resizeColumnsToContents()
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Continue")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Define constraints for imputed values (leave blank for none)."))
        lay.addWidget(self.table)
        lay.addWidget(buttons)
        self.resize(560, 320)

    def result(self) -> dict:
        out = {}
        for r, name in enumerate(self.names):
            spec = {"role": self.roles[r].currentData()}
            for c, key in ((2, "minimum"), (3, "maximum"), (4, "rounding")):
                text = (self.table.item(r, c).text() if self.table.item(r, c) else "").strip()
                try:
                    spec[key] = float(text.replace(",", ".")) if text else None
                except ValueError:
                    spec[key] = None
            out[name] = spec
        return out


class ImputeDialog(VariablePickerDialog):
    DEFAULTS = {"m": 5, "dataset": "Imputed", "method": "auto", "max_iter": 10,
                "scale_model": "linear", "constraints": {}, "history": False, "seed": 2000000}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Impute Missing Data Values", parent)
        self.add_target("vars", "Variables in Model:", min_items=2)
        self.add_option_button("Method...", self._method)
        self.add_option_button("Constraints...", self._constraints)
        self.add_option_button("Output...", self._output)
        form = QFormLayout()
        self.m_spin = QSpinBox()
        self.m_spin.setRange(1, 100)
        self.m_spin.setValue(int(self.opts["m"]))
        self.name_edit = QLineEdit(self.opts["dataset"])
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_000_000_000)
        self.seed_spin.setValue(int(self.opts["seed"]))
        form.addRow("Imputations:", self.m_spin)
        form.addRow("Create a new dataset — Dataset name:", self.name_edit)
        form.addRow("Random number seed:", self.seed_spin)
        self.options_layout.addLayout(form)
        note = QLabel("Categorical variables (nominal/ordinal or string) are imputed with "
                      "logistic regression, scale variables with linear regression or "
                      "predictive mean matching.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#666")
        self.options_layout.addWidget(note)
        self.finish()

    def _method(self):
        dlg = SubDialog("Impute Missing Data Values: Method", self, columns=1)
        box = dlg.group("Imputation Method")
        dlg.radio(box, "method", [("auto", "Automatic"),
                                  ("fcs", "Custom: Fully conditional specification (MCMC)"),
                                  ("monotone", "Custom: Monotone")], self.opts["method"])
        dlg.number(box, "max_iter", "Maximum iterations:", self.opts["max_iter"], 1, 1000, 0)
        box = dlg.group("Model type for scale variables")
        dlg.radio(box, "scale_model", [("linear", "Linear regression"),
                                       ("pmm", "Predictive mean matching")],
                  self.opts["scale_model"])
        values = dlg.run()
        if values:
            values["max_iter"] = int(values["max_iter"])
            self.opts.update(values)

    def _constraints(self):
        names = self.selected("vars")
        if not names:
            return
        dlg = ConstraintsDialog(names, self.opts["constraints"], self)
        if dlg.exec():
            self.opts["constraints"] = dlg.result()

    def _output(self):
        dlg = SubDialog("Impute Missing Data Values: Output", self, columns=1)
        box = dlg.group("Display")
        dlg.check(box, "history", "Iteration history (FCS)", self.opts["history"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def validate(self):
        name = self.name_edit.text().strip()
        if not is_valid_name(name):
            return "Enter a valid name for the new dataset."
        if self.ds.has_name(mi.IMPUTATION_VAR):
            return f"The active dataset already contains {mi.IMPUTATION_VAR}."
        self.opts.update(m=self.m_spin.value(), dataset=name, seed=self.seed_spin.value())
        return None

    def compute(self):
        o = self.opts
        names = self.selected("vars")
        cons = {n: mi.Constraint(c.get("role", "both"), c.get("minimum"), c.get("maximum"),
                                 c.get("rounding"))
                for n, c in o["constraints"].items() if n in names}
        res = mi.impute(self.ds, names, int(o["m"]), o["method"], int(o["max_iter"]),
                        o["scale_model"], cons, int(o["seed"]), history=o["history"],
                        dataset_name=o["dataset"])
        ctx = getattr(self.parent(), "ctx", None)
        if ctx is not None:
            ctx.new_editor(res.dataset)
        self.result_dataset = res.dataset
        return res.section


# =========================================================================== pooled analyses
class _PooledMixin:
    def check_imputed(self) -> str | None:
        if not mi.is_imputed(self.ds):
            return (f"The active dataset has no {mi.IMPUTATION_VAR} variable. Run Analyze > "
                    "Multiple Imputation > Impute Missing Data Values first, then make the "
                    "imputed dataset active.")
        return None


class PooledDescriptivesDialog(_PooledMixin, VariablePickerDialog):
    def __init__(self, ds, parent=None):
        super().__init__(ds, "Pooled Descriptives", parent)
        self.add_target("vars", "Variable(s):", accepts=is_numeric)
        self.finish()

    def validate(self):
        return self.check_imputed()

    def compute(self):
        return mi.pooled_descriptives(self.ds, self.selected("vars"))


class PooledOneSampleTDialog(_PooledMixin, VariablePickerDialog):
    DEFAULTS = {"test": 0.0, "level": 95.0}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Pooled One-Sample T Test", parent)
        self.add_target("vars", "Test Variable(s):", accepts=is_numeric)
        self.add_option_button("Options...", lambda: self.ci_missing_options(False))
        form = QFormLayout()
        self.test_edit = QLineEdit(format_code(self.opts["test"]))
        form.addRow("Test Value:", self.test_edit)
        self.options_layout.addLayout(form)
        self.finish()

    def validate(self):
        try:
            self.opts["test"] = float(self.test_edit.text().replace(",", "."))
        except ValueError:
            return "Test Value must be a number."
        return self.check_imputed()

    def compute(self):
        return mi.pooled_one_sample_t(self.ds, self.selected("vars"), self.opts["test"],
                                      self.level())


class PooledIndependentTDialog(_PooledMixin, GroupingMixin, VariablePickerDialog):
    DEFAULTS = {"level": 95.0, "group_def": None}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Pooled Independent-Samples T Test", parent)
        self.add_target("vars", "Test Variable(s):", accepts=is_numeric)
        self.setup_grouping(allow_cut=False)
        self.add_option_button("Options...", lambda: self.ci_missing_options(False))
        self.finish()

    def validate(self):
        err = GroupingMixin.validate(self)
        return err or self.check_imputed()

    def compute(self):
        d = self.opts["group_def"]
        return mi.pooled_independent_t(self.ds, self.selected("vars"), self.selected_one("group"),
                                       tuple(d["groups"]), self.level())


class PooledPairedTDialog(_PooledMixin, VariablePickerDialog):
    DEFAULTS = {"level": 95.0}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Pooled Paired-Samples T Test", parent)
        self.add_pair_target("pairs", "Paired Variables:")
        self.add_option_button("Options...", lambda: self.ci_missing_options(False))
        self.finish()

    def validate(self):
        return self.check_imputed()

    def compute(self):
        return mi.pooled_paired_t(self.ds, self.pairs("pairs"), self.level())


class PooledCorrelationsDialog(_PooledMixin, VariablePickerDialog):
    def __init__(self, ds, parent=None):
        super().__init__(ds, "Pooled Bivariate Correlations", parent)
        self.add_target("vars", "Variables:", accepts=is_numeric, min_items=2)
        self.finish()

    def validate(self):
        return self.check_imputed()

    def compute(self):
        return mi.pooled_correlations(self.ds, self.selected("vars"))


class PooledLinearDialog(_PooledMixin, VariablePickerDialog):
    DEFAULTS = {"level": 95.0}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Pooled Linear Regression", parent)
        self.add_target("dep", "Dependent:", accepts=is_numeric, single=True)
        self.add_target("ind", "Independent(s):", accepts=is_numeric)
        self.add_option_button("Options...", lambda: self.ci_missing_options(False))
        self.finish()

    def validate(self):
        if self.selected_one("dep") in self.selected("ind"):
            return "The dependent variable cannot also be an independent variable."
        return self.check_imputed()

    def compute(self):
        return mi.pooled_linear(self.ds, self.selected_one("dep"), self.selected("ind"),
                                self.level())


class PooledLogisticDialog(PooledLinearDialog):
    def __init__(self, ds, parent=None):
        super().__init__(ds, parent)
        self.setWindowTitle("Pooled Binary Logistic Regression")

    def compute(self):
        return mi.pooled_logistic(self.ds, self.selected_one("dep"), self.selected("ind"),
                                  self.level())


MENUS = {
    "Analyze": [("Missing Value Analysis...", MissingValueAnalysisDialog),
                ("Multiple Imputation", [
                    ("Analyze Patterns...", AnalyzePatternsDialog),
                    ("Impute Missing Data Values...", ImputeDialog),
                    None,
                    ("Pooled Analysis", [
                        ("Descriptives...", PooledDescriptivesDialog),
                        ("One-Sample T Test...", PooledOneSampleTDialog),
                        ("Independent-Samples T Test...", PooledIndependentTDialog),
                        ("Paired-Samples T Test...", PooledPairedTDialog),
                        ("Bivariate Correlations...", PooledCorrelationsDialog),
                        ("Linear Regression...", PooledLinearDialog),
                        ("Binary Logistic Regression...", PooledLogisticDialog)])])],
}


def selftest() -> None:
    """Headless check of MVA, Little's MCAR test, FCS imputation and pooling."""
    from rsstatchem.core.dataset import Dataset

    rng = np.random.default_rng(5)
    n = 120
    x = rng.normal(size=n)
    z = 0.5 * x + rng.normal(size=n)
    g = (rng.random(n) < 0.5).astype(float)
    y = x - z + g + rng.normal(size=n)
    df = pd.DataFrame({"x": x, "z": z, "g": g, "y": y})
    df.loc[rng.random(n) < 0.2, "z"] = np.nan
    df.loc[rng.random(n) < 0.1, "g"] = np.nan
    ds = Dataset(df)
    ds.var("g").measure = "nominal"
    mv.mva(ds, ["x", "z", "y"], ["g"], ("listwise", "pairwise", "em", "regression"),
           ttest=True, crosstabs=True)
    chi, dfree, p, _ = mv.little_mcar(df[["x", "z", "y"]].to_numpy())
    assert dfree == 2 and chi >= 0 and 0 <= p <= 1
    mi.analyze_patterns(ds, ["x", "z", "g", "y"])
    res = mi.impute(ds, ["x", "z", "g", "y"], m=3, max_iter=3)
    new = res.dataset
    assert mi.is_imputed(new) and new.n_cases == 4 * n
    assert not np.isnan(new.df.loc[new.df["Imputation_"] > 0, "z"]).any()
    mi.pooled_descriptives(new, ["z"])
    mi.pooled_linear(new, "y", ["x", "z", "g"])
    mi.pooled_logistic(new, "g", ["x", "z"])
    mi.pooled_correlations(new, ["x", "z"])
    mi.pooled_independent_t(new, ["y"], "g", (0.0, 1.0))
    from rsstatchem.core.stats import mi_pool  # WP6b automatic pooling

    mi_pool.selftest()
    assert mi_pool.auto_views(new) is not None

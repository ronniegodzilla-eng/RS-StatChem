"""Analyze > Correlate and Analyze > Regression > Linear dialogs."""
from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGroupBox,
                               QHBoxLayout)

from rsstatchem.core.stats import correlate, regression
from rsstatchem.ui.dialogs.base import SubDialog, VariablePickerDialog, is_numeric


class BivariateDialog(VariablePickerDialog):
    DEFAULTS = {"pearson": True, "kendall": False, "spearman": False, "two_tailed": True,
                "flag": True, "descriptives": False, "listwise": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Bivariate Correlations", parent)
        self.add_target("vars", "Variables:", accepts=is_numeric, min_items=2)
        self.add_option_button("Options...", self._options)
        box = QGroupBox("Correlation Coefficients")
        row = QHBoxLayout(box)
        for key, text in (("pearson", "Pearson"), ("kendall", "Kendall's tau-b"),
                          ("spearman", "Spearman")):
            row.addWidget(self._check(key, text))
        self.options_layout.addWidget(box)
        box = QGroupBox("Test of Significance")
        row = QHBoxLayout(box)
        self.tails = QComboBox()
        self.tails.addItems(["Two-tailed", "One-tailed"])
        self.tails.setCurrentIndex(0 if self.opts["two_tailed"] else 1)
        self.tails.currentIndexChanged.connect(
            lambda i: self.opts.__setitem__("two_tailed", i == 0))
        row.addWidget(self.tails)
        row.addWidget(self._check("flag", "Flag significant correlations"))
        row.addStretch()
        self.options_layout.addWidget(box)
        self.finish()

    def _check(self, key, text):
        cb = QCheckBox(text)
        cb.setChecked(self.opts[key])
        cb.toggled.connect(lambda on, k=key: self.opts.__setitem__(k, on))
        return cb

    def _options(self) -> None:
        dlg = SubDialog("Bivariate Correlations: Options", self, columns=1)
        dlg.check(dlg.group("Statistics"), "descriptives", "Means and standard deviations",
                  self.opts["descriptives"])
        dlg.radio(dlg.group("Missing Values"), "listwise",
                  [(False, "Exclude cases pairwise"), (True, "Exclude cases listwise")],
                  self.opts["listwise"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def validate(self):
        if not (self.opts["pearson"] or self.opts["kendall"] or self.opts["spearman"]):
            return "Select at least one correlation coefficient."
        return None

    def compute(self):
        o = self.opts
        return correlate.bivariate(self.ds, self.selected("vars"), o["pearson"], o["kendall"],
                                   o["spearman"], o["two_tailed"], o["flag"], o["descriptives"],
                                   o["listwise"])


class PartialDialog(VariablePickerDialog):
    DEFAULTS = {"two_tailed": True, "flag": True, "zero_order": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Partial Correlations", parent)
        self.add_target("vars", "Variables:", accepts=is_numeric, min_items=2)
        self.add_target("controls", "Controlling for:", accepts=is_numeric)
        self.add_option_button("Options...", self._options)
        self.add_checkbox("flag", "Display actual significance level / flag significant")
        self.finish()

    def _options(self) -> None:
        dlg = SubDialog("Partial Correlations: Options", self, columns=1)
        box = dlg.group("Statistics")
        dlg.check(box, "zero_order", "Zero-order correlations", self.opts["zero_order"])
        dlg.radio(dlg.group("Test of Significance"), "two_tailed",
                  [(True, "Two-tailed"), (False, "One-tailed")], self.opts["two_tailed"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def compute(self):
        o = self.opts
        return correlate.partial(self.ds, self.selected("vars"), self.selected("controls"),
                                 o["two_tailed"], o["flag"], o["zero_order"])


class LinearRegressionDialog(VariablePickerDialog):
    DEFAULTS = {"method": "enter", "statistics": list(regression.DEFAULT_STATS),
                "plots": ["histogram", "pp", "scatter"], "save": [],
                "assumptions": list(regression.ASSUMPTIONS), "level": 95.0,
                "p_enter": 0.05, "p_remove": 0.10}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Linear Regression", parent)
        self.add_target("dep", "Dependent:", accepts=is_numeric, single=True)
        self.add_target("ind", "Independent(s):", accepts=is_numeric)
        self.add_option_button("Statistics...", self._statistics)
        self.add_option_button("Plots...", self._plots)
        self.add_option_button("Save...", self._save)
        self.add_option_button("Assumptions...", self._assumptions)
        self.add_option_button("Options...", self._options)
        self.method = QComboBox()
        for key, text in (("enter", "Enter"), ("stepwise", "Stepwise"), ("forward", "Forward"),
                          ("backward", "Backward")):
            self.method.addItem(text, key)
        self.method.setCurrentIndex(max(self.method.findData(self.opts["method"]), 0))
        self.method.currentIndexChanged.connect(
            lambda _: self.opts.__setitem__("method", self.method.currentData()))
        form = QFormLayout()
        form.addRow("Method:", self.method)
        self.options_layout.addLayout(form)
        self.finish()

    def _multi(self, title: str, key: str, groups: list[tuple[str, list[tuple[str, str]]]],
               extra=None) -> None:
        dlg = SubDialog(title, self)
        for group_title, items in groups:
            box = dlg.group(group_title)
            for item_key, text in items:
                dlg.check(box, item_key, text, item_key in self.opts[key])
        if extra:
            extra(dlg)
        values = dlg.run()
        if values is None:
            return
        keys = [k for _, items in groups for k, _ in items]
        self.opts[key] = [k for k in keys if values.get(k)]
        for k, v in values.items():
            if k not in keys:
                self.opts[k] = v

    def _statistics(self) -> None:
        def ci_level(dlg):
            dlg.number(dlg.group("Confidence Intervals"), "level", "Level (%):",
                       self.opts["level"], 50, 99.99, 2)

        self._multi("Linear Regression: Statistics", "statistics", [
            ("Regression Coefficients", [("estimates", "Estimates"),
                                         ("ci", "Confidence intervals")]),
            ("Model", [("model_fit", "Model fit"), ("r2_change", "R squared change"),
                       ("descriptives", "Descriptives"),
                       ("collinearity", "Collinearity diagnostics")]),
            ("Residuals", [("durbin_watson", "Durbin-Watson")]),
        ], ci_level)

    def _plots(self) -> None:
        self._multi("Linear Regression: Plots", "plots", [
            ("Standardized Residual Plots", [("histogram", "Histogram"),
                                             ("pp", "Normal probability plot")]),
            ("Scatter", [("scatter", "*ZRESID (Y) vs *ZPRED (X)")]),
        ])

    def _save(self) -> None:
        self._multi("Linear Regression: Save", "save", [
            ("Predicted Values", [("pred", "Unstandardized"), ("zpred", "Standardized")]),
            ("Residuals", [("resid", "Unstandardized"), ("zresid", "Standardized")]),
        ])

    def _assumptions(self) -> None:
        self._multi("Linear Regression: Classical Assumption Tests", "assumptions", [
            ("Tests", [("normality", "Normality of residuals (K-S, Shapiro-Wilk)"),
                       ("multicollinearity", "Multicollinearity (Tolerance, VIF)"),
                       ("heteroscedasticity", "Heteroscedasticity (Glejser, Breusch-Pagan, "
                                              "White)"),
                       ("autocorrelation", "Autocorrelation (Durbin-Watson, Breusch-Godfrey, "
                                           "Runs test)")]),
        ])

    def _options(self) -> None:
        dlg = SubDialog("Linear Regression: Options", self, columns=1)
        box = dlg.group("Stepping Method Criteria (probability of F)")
        dlg.number(box, "p_enter", "Entry:", self.opts["p_enter"], 0.0001, 0.99, 4)
        dlg.number(box, "p_remove", "Removal:", self.opts["p_remove"], 0.0001, 0.99, 4)
        values = dlg.run()
        if values:
            if values["p_enter"] >= values["p_remove"]:
                values["p_remove"] = min(0.99, values["p_enter"] + 0.05)
            self.opts.update(values)

    def compute(self):
        o = self.opts
        return regression.linear(self.ds, self.selected_one("dep"), self.selected("ind"),
                                 o["method"], tuple(o["statistics"]), tuple(o["plots"]),
                                 tuple(o["save"]), tuple(o["assumptions"]), self.level(),
                                 o["p_enter"], o["p_remove"])

"""Analyze > Descriptive Statistics > Explore / Crosstabs, and Analyze > Scale > Reliability."""
from __future__ import annotations

from PySide6.QtWidgets import (QButtonGroup, QComboBox, QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QRadioButton)

from rsstatchem.core.stats import crosstabs, explore, reliability
from rsstatchem.ui.dialogs.base import SubDialog, VariablePickerDialog, is_numeric


class ExploreDialog(VariablePickerDialog):
    DEFAULTS = {"display": "both", "boxplot": True, "boxplot_mode": "factor",
                "histogram": False, "stem_leaf": True, "normality": True, "level": 95.0,
                "descriptives": True, "m_estimators": False, "outliers": False,
                "percentiles": False, "spread": "none", "spread_power": "log",
                "missing": "listwise"}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Explore", parent)
        self.add_target("deps", "Dependent List:", accepts=is_numeric)
        self.add_target("factors", "Factor List:", required=False)
        self.add_target("label", "Label Cases by:", single=True, required=False)
        self.add_option_button("Statistics...", self._statistics)
        self.add_option_button("Plots...", self._plots)
        self.add_option_button("Options...", self._options)
        row = QHBoxLayout()
        row.addWidget(QLabel("Display:"))
        group = QButtonGroup(self)
        for key, text in (("both", "Both"), ("statistics", "Statistics"), ("plots", "Plots")):
            rb = QRadioButton(text)
            rb.setChecked(self.opts["display"] == key)
            rb.toggled.connect(lambda on, k=key: on and self.opts.__setitem__("display", k))
            group.addButton(rb)
            row.addWidget(rb)
        row.addStretch()
        self.options_layout.addLayout(row)
        self.finish()

    def _statistics(self) -> None:
        dlg = SubDialog("Explore: Statistics", self, columns=1)
        box = dlg.group("")
        dlg.check(box, "descriptives", "Descriptives", self.opts["descriptives"])
        dlg.number(box, "level", "Confidence Interval for Mean:", self.opts["level"], 50,
                   99.99, 2, " %")
        dlg.check(box, "m_estimators", "M-estimators", self.opts["m_estimators"])
        dlg.check(box, "outliers", "Outliers", self.opts["outliers"])
        dlg.check(box, "percentiles", "Percentiles", self.opts["percentiles"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def _plots(self) -> None:
        dlg = SubDialog("Explore: Plots", self, columns=2)
        box = dlg.group("Boxplots")
        dlg.radio(box, "boxplot_mode", [("factor", "Factor levels together"),
                                        ("dependents", "Dependents together"),
                                        ("none", "None")], self.opts["boxplot_mode"])
        box = dlg.group("Descriptive")
        dlg.check(box, "stem_leaf", "Stem-and-leaf", self.opts["stem_leaf"])
        dlg.check(box, "histogram", "Histogram", self.opts["histogram"])
        dlg.check(dlg.group("", span=2), "normality", "Normality plots with tests",
                  self.opts["normality"])
        box = dlg.group("Spread vs Level with Levene Test")
        dlg.radio(box, "spread", [("none", "None"), ("estimate", "Power estimation"),
                                  ("transformed", "Transformed"),
                                  ("untransformed", "Untransformed")], self.opts["spread"])
        box = dlg.group("Power (Transformed)")
        dlg.radio(box, "spread_power", [("log", "Natural log"), ("inv_sqrt", "1/square root"),
                                        ("reciprocal", "Reciprocal"),
                                        ("sqrt", "Square root"), ("square", "Square"),
                                        ("cube", "Cube")], self.opts["spread_power"])
        values = dlg.run()
        if values:
            self.opts.update(values)
            self.opts["boxplot"] = values["boxplot_mode"] != "none"

    def _options(self) -> None:
        dlg = SubDialog("Explore: Options", self, columns=1)
        dlg.radio(dlg.group("Missing Values"), "missing", [
            ("listwise", "Exclude cases listwise"), ("pairwise", "Exclude cases pairwise"),
            ("report", "Report values")], self.opts["missing"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def compute(self):
        o = self.opts
        show_stats = o["display"] in ("both", "statistics")
        show_plots = o["display"] in ("both", "plots")
        return explore.explore(
            self.ds, self.selected("deps"), self.selected("factors"),
            statistics=show_stats, normality=o["normality"],
            boxplot=show_plots and o["boxplot_mode"] != "none",
            qq=show_plots and o["normality"], histogram=show_plots and o["histogram"],
            level=self.level(), descriptives=o["descriptives"],
            m_estimators=o["m_estimators"], outliers=o["outliers"],
            percentiles=o["percentiles"], label_by=self.selected_one("label"),
            boxplot_mode=o["boxplot_mode"] if show_plots else "none",
            stem_leaf=show_plots and o["stem_leaf"],
            spread=o["spread"] if show_plots else "none", spread_power=o["spread_power"],
            missing=o["missing"])


class CrosstabsDialog(VariablePickerDialog):
    DEFAULTS = {"chisq": True, "phi": True, "cc": False, "corr": False, "lambda": False,
                "uc": False, "gamma": False, "d": False, "btau": False, "ctau": False,
                "kappa": False, "eta": False, "risk": False, "mcnemar": False, "cmh": False,
                "cmh_value": 1.0, "cells": ["observed"], "z_test": False, "bonferroni": True,
                "hide_small": False, "hide_n": 5, "row_order": "ascending",
                "bar": False, "suppress": False, "layers": [[]]}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Crosstabs", parent)
        self.add_target("rows", "Row(s):")
        self.add_target("cols", "Column(s):")
        self.add_target("layer", "Layer:", required=False)
        self.add_option_button("Statistics...", self._statistics)
        self.add_option_button("Cells...", self._cells)
        self.add_option_button("Format...", self._format)
        layer_row = QHBoxLayout()
        self.prev_button, self.next_button = QPushButton("Previous"), QPushButton("Next")
        self.layer_label = QLabel()
        self.prev_button.clicked.connect(lambda: self._switch_layer(-1))
        self.next_button.clicked.connect(lambda: self._switch_layer(1))
        layer_row.addWidget(self.prev_button)
        layer_row.addWidget(self.layer_label)
        layer_row.addWidget(self.next_button)
        layer_row.addStretch()
        self.options_layout.addLayout(layer_row)
        self.bar = self.add_checkbox("bar", "Display clustered bar charts")
        self.add_checkbox("suppress", "Suppress tables")
        self.finish()
        self.opts["layers"] = [list(lay) for lay in self.opts.get("layers") or [[]]] or [[]]
        self._layer_index = 0
        self._show_layer()

    # ------------------------------------------------------------------ layers
    def _layer_target(self):
        return next(t for t in self._targets if t.key == "layer")

    def _store_layer(self) -> None:
        self.opts["layers"][self._layer_index] = self.selected("layer")

    def _show_layer(self) -> None:
        target = self._layer_target()
        self._move_back(target, all_items=True)
        for name in self.opts["layers"][self._layer_index]:
            if self.ds.has_name(name):
                self._take(name, target)
        total = len(self.opts["layers"])
        self.layer_label.setText(f"Layer {self._layer_index + 1} of {total}")
        self.prev_button.setEnabled(self._layer_index > 0)
        self._update_arrows()

    def targets_changed(self) -> None:
        if hasattr(self, "_layer_index"):
            self.next_button.setEnabled(bool(self.selected("layer")) or
                                        self._layer_index < len(self.opts["layers"]) - 1)

    def _switch_layer(self, step: int) -> None:
        self._store_layer()
        new = self._layer_index + step
        if new < 0:
            return
        if new >= len(self.opts["layers"]):
            self.opts["layers"].append([])
        self._layer_index = new
        self._show_layer()

    def layers(self) -> list[list[str]]:
        self._store_layer()
        return [lay for lay in self.opts["layers"] if lay]

    def reset(self) -> None:
        super().reset()
        self.opts["layers"] = [[]]
        self._layer_index = 0
        self._show_layer()

    def accept(self) -> None:
        self._store_layer()
        self.opts["layers"] = [lay for lay in self.opts["layers"] if lay] or [[]]
        super().accept()

    # ------------------------------------------------------------------ sub-dialogs
    def _statistics(self) -> None:
        o = self.opts
        dlg = SubDialog("Crosstabs: Statistics", self)
        box = dlg.group("")
        dlg.check(box, "chisq", "Chi-square", o["chisq"])
        box = dlg.group("")
        dlg.check(box, "corr", "Correlations", o["corr"])
        box = dlg.group("Nominal")
        dlg.check(box, "cc", "Contingency coefficient", o["cc"])
        dlg.check(box, "phi", "Phi and Cramer's V", o["phi"])
        dlg.check(box, "lambda", "Lambda", o["lambda"])
        dlg.check(box, "uc", "Uncertainty coefficient", o["uc"])
        box = dlg.group("Ordinal")
        dlg.check(box, "gamma", "Gamma", o["gamma"])
        dlg.check(box, "d", "Somers' d", o["d"])
        dlg.check(box, "btau", "Kendall's tau-b", o["btau"])
        dlg.check(box, "ctau", "Kendall's tau-c", o["ctau"])
        box = dlg.group("Nominal by Interval")
        dlg.check(box, "eta", "Eta", o["eta"])
        box = dlg.group("")
        dlg.check(box, "kappa", "Kappa", o["kappa"])
        dlg.check(box, "risk", "Risk", o["risk"])
        dlg.check(box, "mcnemar", "McNemar", o["mcnemar"])
        box = dlg.group("", span=2)
        dlg.check(box, "cmh", "Cochran's and Mantel-Haenszel statistics", o["cmh"])
        dlg.number(box, "cmh_value", "Test common odds ratio equals:", o["cmh_value"], 0.0001,
                   1e6, 4)
        values = dlg.run()
        if values:
            o.update(values)

    def _cells(self) -> None:
        o = self.opts
        cells = o["cells"]
        dlg = SubDialog("Crosstabs: Cell Display", self)
        box = dlg.group("Counts")
        dlg.check(box, "observed", "Observed", "observed" in cells)
        dlg.check(box, "expected", "Expected", "expected" in cells)
        dlg.check(box, "hide_small", "Hide small counts", o["hide_small"])
        dlg.number(box, "hide_n", "Less than", o["hide_n"], 2, 1e6, 0)
        box = dlg.group("z-test")
        dlg.check(box, "z_test", "Compare column proportions", o["z_test"])
        dlg.check(box, "bonferroni", "Adjust p-values (Bonferroni method)", o["bonferroni"])
        box = dlg.group("Percentages")
        dlg.check(box, "row", "Row", "row" in cells)
        dlg.check(box, "column", "Column", "column" in cells)
        dlg.check(box, "total", "Total", "total" in cells)
        box = dlg.group("Residuals")
        dlg.check(box, "resid", "Unstandardized", "resid" in cells)
        dlg.check(box, "sresid", "Standardized", "sresid" in cells)
        dlg.check(box, "asresid", "Adjusted standardized", "asresid" in cells)
        values = dlg.run()
        if values:
            o["cells"] = [k for k in crosstabs.CELLS if values.get(k)] or ["observed"]
            for key in ("z_test", "bonferroni", "hide_small", "hide_n"):
                o[key] = values[key]

    def _format(self) -> None:
        dlg = SubDialog("Crosstabs: Table Format", self, columns=1)
        dlg.radio(dlg.group("Row Order"), "row_order", [("ascending", "Ascending"),
                                                        ("descending", "Descending")],
                  self.opts["row_order"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def compute(self):
        o = self.opts
        stats = tuple(k for k in crosstabs.STATISTICS if o.get(k))
        return crosstabs.crosstabs(
            self.ds, self.selected("rows"), self.selected("cols"), False, False, False,
            tuple(o["cells"]), layers=self.layers(), statistics=stats,
            hide_small=int(o["hide_n"]) if o["hide_small"] else None, z_test=o["z_test"],
            bonferroni=o["bonferroni"], row_order=o["row_order"],
            suppress_tables=o["suppress"], clustered_bar=o["bar"], cmh_value=o["cmh_value"])


class ReliabilityDialog(VariablePickerDialog):
    DEFAULTS = {"item": False, "scale": False, "deleted": True, "inter": False,
                "inter_cov": False, "s_means": False, "s_vars": False, "s_covs": False,
                "s_corrs": False, "anova": "none", "hotelling": False, "tukey": False,
                "icc": False, "icc_model": "twoway_mixed", "icc_type": "consistency",
                "icc_level": 95.0, "icc_test": 0.0, "fleiss": False, "fleiss_cats": True,
                "validity": True, "alpha": 0.05, "model": "alpha", "label": ""}
    MODELS = (("alpha", "Alpha"), ("split", "Split-half"), ("guttman", "Guttman"),
              ("parallel", "Parallel"), ("strict", "Strict parallel"), ("omega", "Omega"))

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Reliability Analysis", parent)
        self.add_target("items", "Items:", accepts=is_numeric, min_items=2)
        self.add_option_button("Statistics...", self._statistics)
        self.model = QComboBox()
        for key, text in self.MODELS:
            self.model.addItem(text, key)
        self.model.setCurrentIndex(max(self.model.findData(self.opts["model"]), 0))
        self.model.currentIndexChanged.connect(
            lambda _: self.opts.__setitem__("model", self.model.currentData()))
        self.scale_label = QLineEdit(self.opts["label"])
        self.scale_label.textChanged.connect(lambda t: self.opts.__setitem__("label", t))
        form = QFormLayout()
        form.addRow("Model:", self.model)
        form.addRow("Scale label:", self.scale_label)
        self.options_layout.addLayout(form)
        self.add_checkbox("validity", "Item validity (Pearson item-total r vs r table)")
        self.finish()

    def _statistics(self) -> None:
        o = self.opts
        dlg = SubDialog("Reliability Analysis: Statistics", self, columns=3)
        box = dlg.group("Descriptives for")
        dlg.check(box, "item", "Item", o["item"])
        dlg.check(box, "scale", "Scale", o["scale"])
        dlg.check(box, "deleted", "Scale if item deleted", o["deleted"])
        box = dlg.group("Inter-Item")
        dlg.check(box, "inter", "Correlations", o["inter"])
        dlg.check(box, "inter_cov", "Covariances", o["inter_cov"])
        box = dlg.group("Summaries")
        dlg.check(box, "s_means", "Means", o["s_means"])
        dlg.check(box, "s_vars", "Variances", o["s_vars"])
        dlg.check(box, "s_covs", "Covariances", o["s_covs"])
        dlg.check(box, "s_corrs", "Correlations", o["s_corrs"])
        box = dlg.group("ANOVA Table")
        dlg.radio(box, "anova", [("none", "None"), ("f", "F test"),
                                 ("friedman", "Friedman chi-square"),
                                 ("cochran", "Cochran chi-square")], o["anova"])
        box = dlg.group("")
        dlg.check(box, "hotelling", "Hotelling's T-square", o["hotelling"])
        dlg.check(box, "tukey", "Tukey's test of additivity", o["tukey"])
        box = dlg.group("Inter-Rater Agreement: Fleiss' Kappa")
        dlg.check(box, "fleiss", "Fleiss' kappa (items are raters)", o["fleiss"])
        dlg.check(box, "fleiss_cats", "Display agreement on individual categories",
                  o["fleiss_cats"])
        box = dlg.group("", span=3)
        dlg.check(box, "icc", "Intraclass correlation coefficient", o["icc"])
        row = QHBoxLayout()
        row.addWidget(QLabel("Model:"))
        model = QComboBox()
        for key, text in (("twoway_mixed", "Two-Way Mixed"), ("twoway_random", "Two-Way Random"),
                          ("oneway", "One-Way Random")):
            model.addItem(text, key)
        model.setCurrentIndex(max(model.findData(o["icc_model"]), 0))
        row.addWidget(model)
        row.addWidget(QLabel("Type:"))
        kind = QComboBox()
        for key, text in (("consistency", "Consistency"), ("absolute", "Absolute Agreement")):
            kind.addItem(text, key)
        kind.setCurrentIndex(max(kind.findData(o["icc_type"]), 0))
        row.addWidget(kind)
        row.addStretch()
        box.addLayout(row)
        dlg._getters["icc_model"] = model.currentData
        dlg._getters["icc_type"] = kind.currentData
        dlg.number(box, "icc_level", "Confidence interval:", o["icc_level"], 50, 99.99, 2, " %")
        dlg.number(box, "icc_test", "Test value:", o["icc_test"], -0.99, 0.99, 3)
        dlg.number(dlg.group("Item validity", span=3), "alpha",
                   "Significance level (r table):", o["alpha"], 0.001, 0.2, 3)
        values = dlg.run()
        if values:
            o.update(values)

    def compute(self):
        o = self.opts
        summaries = tuple(k for k, key in (("means", "s_means"), ("variances", "s_vars"),
                                           ("covariances", "s_covs"),
                                           ("correlations", "s_corrs")) if o[key])
        icc = {"model": o["icc_model"], "type": o["icc_type"], "level": o["icc_level"] / 100,
               "test_value": o["icc_test"]} if o["icc"] else None
        fleiss = {"individual": o["fleiss_cats"], "level": 0.95} if o["fleiss"] else None
        return reliability.reliability(
            self.ds, self.selected("items"), o["item"], o["scale"], o["deleted"], o["inter"],
            o["validity"], o["alpha"], o["label"].strip() or "ALL VARIABLES",
            model=o["model"], inter_cov=o["inter_cov"], summaries=summaries,
            anova=o["anova"], tukey=o["tukey"], hotelling=o["hotelling"], icc=icc,
            fleiss=fleiss)

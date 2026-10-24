"""Analyze > Descriptive Statistics > Explore / Crosstabs, and Analyze > Scale > Reliability."""
from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QFormLayout

from rsstatchem.core.stats import crosstabs, explore, reliability
from rsstatchem.ui.dialogs.base import SubDialog, VariablePickerDialog, is_numeric


class ExploreDialog(VariablePickerDialog):
    DEFAULTS = {"display": "both", "boxplot": True, "histogram": False, "normality": True,
                "level": 95.0}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Explore", parent)
        self.add_target("deps", "Dependent List:", accepts=is_numeric)
        self.add_target("factors", "Factor List:", required=False)
        self.add_option_button("Statistics...", lambda: self.ci_missing_options(missing=False))
        self.add_option_button("Plots...", self._plots)
        self.display = QComboBox()
        for key, text in (("both", "Both"), ("statistics", "Statistics"), ("plots", "Plots")):
            self.display.addItem(text, key)
        self.display.setCurrentIndex(max(self.display.findData(self.opts["display"]), 0))
        self.display.currentIndexChanged.connect(
            lambda _: self.opts.__setitem__("display", self.display.currentData()))
        form = QFormLayout()
        form.addRow("Display:", self.display)
        self.options_layout.addLayout(form)
        self.finish()

    def _plots(self) -> None:
        dlg = SubDialog("Explore: Plots", self, columns=1)
        box = dlg.group("Boxplots & Descriptive")
        dlg.check(box, "boxplot", "Boxplot (factor levels together)", self.opts["boxplot"])
        dlg.check(box, "histogram", "Histogram", self.opts["histogram"])
        dlg.check(dlg.group("Normality"), "normality", "Normality plots with tests",
                  self.opts["normality"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def compute(self):
        o = self.opts
        show_stats = o["display"] in ("both", "statistics")
        show_plots = o["display"] in ("both", "plots")
        return explore.explore(self.ds, self.selected("deps"), self.selected("factors"),
                               statistics=show_stats, normality=o["normality"],
                               boxplot=show_plots and o["boxplot"],
                               qq=show_plots and o["normality"],
                               histogram=show_plots and o["histogram"], level=self.level())


class CrosstabsDialog(VariablePickerDialog):
    DEFAULTS = {"chisq": True, "phi": True, "cc": False,
                "cells": ["observed"]}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Crosstabs", parent)
        self.add_target("rows", "Row(s):")
        self.add_target("cols", "Column(s):")
        self.add_option_button("Statistics...", self._statistics)
        self.add_option_button("Cells...", self._cells)
        self.finish()

    def _statistics(self) -> None:
        dlg = SubDialog("Crosstabs: Statistics", self)
        dlg.check(dlg.group("Tests"), "chisq", "Chi-square", self.opts["chisq"])
        box = dlg.group("Nominal")
        dlg.check(box, "phi", "Phi and Cramer's V", self.opts["phi"])
        dlg.check(box, "cc", "Contingency coefficient", self.opts["cc"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def _cells(self) -> None:
        dlg = SubDialog("Crosstabs: Cell Display", self)
        box = dlg.group("Counts")
        dlg.check(box, "observed", "Observed", "observed" in self.opts["cells"])
        dlg.check(box, "expected", "Expected", "expected" in self.opts["cells"])
        box = dlg.group("Percentages")
        dlg.check(box, "row", "Row", "row" in self.opts["cells"])
        dlg.check(box, "column", "Column", "column" in self.opts["cells"])
        dlg.check(box, "total", "Total", "total" in self.opts["cells"])
        values = dlg.run()
        if values:
            self.opts["cells"] = [k for k in crosstabs.CELLS if values.get(k)] or ["observed"]

    def compute(self):
        o = self.opts
        return crosstabs.crosstabs(self.ds, self.selected("rows"), self.selected("cols"),
                                   o["chisq"], o["phi"], o["cc"], tuple(o["cells"]))


class ReliabilityDialog(VariablePickerDialog):
    DEFAULTS = {"item": False, "scale": False, "deleted": True, "inter": False,
                "validity": True, "alpha": 0.05}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Reliability Analysis", parent)
        self.add_target("items", "Items:", accepts=is_numeric, min_items=2)
        self.add_option_button("Statistics...", self._statistics)
        self.add_checkbox("validity", "Item validity (Pearson item-total r vs r table)")
        self.finish()

    def _statistics(self) -> None:
        dlg = SubDialog("Reliability Analysis: Statistics", self)
        box = dlg.group("Descriptives for")
        dlg.check(box, "item", "Item", self.opts["item"])
        dlg.check(box, "scale", "Scale", self.opts["scale"])
        dlg.check(box, "deleted", "Scale if item deleted", self.opts["deleted"])
        dlg.check(dlg.group("Inter-Item"), "inter", "Correlations", self.opts["inter"])
        dlg.number(dlg.group("Item validity"), "alpha", "Significance level (r table):",
                   self.opts["alpha"], 0.001, 0.2, 3)
        values = dlg.run()
        if values:
            self.opts.update(values)

    def compute(self):
        o = self.opts
        return reliability.reliability(self.ds, self.selected("items"), o["item"], o["scale"],
                                       o["deleted"], o["inter"], o["validity"], o["alpha"])

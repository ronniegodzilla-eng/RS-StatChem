"""Analyze > Compare Means and General Linear Model dialogs."""
from __future__ import annotations

import pandas as pd
from PySide6.QtWidgets import QFormLayout, QLineEdit

from rsstatchem.core.dataset import format_code
from rsstatchem.core.stats import anova, ttest
from rsstatchem.core.stats.anova import POSTHOC, POSTHOC_ORDER
from rsstatchem.ui.dialogs.base import (DefineGroupsDialog, DefineRangeDialog, SubDialog,
                                        VariablePickerDialog, is_numeric)


class GroupingMixin:
    """A single "Grouping Variable" target with a Define Groups… / Define Range… button."""

    def setup_grouping(self: VariablePickerDialog, label: str = "Grouping Variable:",
                       allow_cut: bool = True, use_range: bool = False) -> None:
        self._allow_cut = allow_cut
        self._use_range = use_range
        self.add_target("group", label, single=True)
        self.define_button = self.add_option_button(
            "Define Range..." if use_range else "Define Groups...", self._define)

    def _definition(self):
        d = self.opts.get("group_def") or {}
        return d if d.get("var") == self.selected_one("group") else None

    def targets_changed(self) -> None:
        if not hasattr(self, "define_button"):
            return
        name = self.selected_one("group")
        self.define_button.setEnabled(name is not None)
        d = self._definition()
        if name is None:
            return
        if not d:
            self.set_item_suffix("group", "(? ?)")
        elif self._use_range:
            self.set_item_suffix("group", f"({d['lo']:g} {d['hi']:g})")
        elif d.get("cut") is not None:
            self.set_item_suffix("group", f"({d['cut']:g})")
        else:
            self.set_item_suffix("group", "(" + " ".join(format_code(g) for g in d["groups"])
                                 + ")")

    def _define(self) -> None:
        name = self.selected_one("group")
        if name is None:
            return
        current = self._definition()
        if self._use_range:
            values = DefineRangeDialog((current["lo"], current["hi"]) if current else None,
                                       self).run()
            if values:
                self.opts["group_def"] = {"var": name, **values}
        else:
            dlg = DefineGroupsDialog(self.ds.var(name), current, self._allow_cut, self)
            if dlg.exec():
                result = dlg.result()
                if result is None:
                    return
                self.opts["group_def"] = {"var": name, **result}
        self.targets_changed()

    def validate(self) -> str | None:
        name = self.selected_one("group")
        if not name or self._definition():
            return None
        # Sensible defaults instead of an error: the two labelled codes, or the data range.
        var = self.ds.var(name)
        if self._use_range:
            codes = pd.to_numeric(self.ds.analysis_series(name), errors="coerce").dropna()
            if len(codes):
                self.opts["group_def"] = {"var": name, "lo": float(codes.min()),
                                          "hi": float(codes.max())}
                return None
        elif len(var.value_labels) == 2:
            self.opts["group_def"] = {"var": name, "groups": tuple(var.value_labels),
                                      "cut": None}
            return None
        return "Click Define Groups… to set the two groups to compare."


class OneSampleTDialog(VariablePickerDialog):
    DEFAULTS = {"test_value": 0.0, "level": 95.0, "listwise": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "One-Sample T Test", parent)
        self.add_target("vars", "Test Variable(s):", accepts=is_numeric)
        self.add_option_button("Options...", self.ci_missing_options)
        self.test_value = QLineEdit(f"{self.opts['test_value']:g}")
        form = QFormLayout()
        form.addRow("Test Value:", self.test_value)
        self.options_layout.addLayout(form)
        self.finish()

    def validate(self):
        try:
            self.opts["test_value"] = float(self.test_value.text().replace(",", "."))
        except ValueError:
            return "Test Value must be a number."
        return None

    def compute(self):
        return ttest.one_sample(self.ds, self.selected("vars"), self.opts["test_value"],
                                self.level(), self.opts["listwise"])


class IndependentTDialog(GroupingMixin, VariablePickerDialog):
    DEFAULTS = {"level": 95.0, "listwise": False, "group_def": None}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Independent-Samples T Test", parent)
        self.add_target("vars", "Test Variable(s):", accepts=is_numeric)
        self.setup_grouping()
        self.add_option_button("Options...", self.ci_missing_options)
        self.finish()

    def compute(self):
        d = self._definition()
        return ttest.independent(self.ds, self.selected("vars"), self.selected_one("group"),
                                 d.get("groups"), d.get("cut"), self.level(),
                                 self.opts["listwise"])


class PairedTDialog(VariablePickerDialog):
    DEFAULTS = {"level": 95.0}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Paired-Samples T Test", parent)
        self.add_pair_target("pairs", "Paired Variables (Variable1 – Variable2):")
        self.add_option_button("Options...", lambda: self.ci_missing_options(missing=False))
        self.finish()

    def compute(self):
        return ttest.paired(self.ds, self.pairs("pairs"), self.level())


def _posthoc_dialog(owner: VariablePickerDialog, factors: list[str] | None = None) -> None:
    dlg = SubDialog(f"{owner.windowTitle()}: Post Hoc Multiple Comparisons", owner)
    if factors is not None:
        box = dlg.group("Post Hoc Tests for")
        for f in factors:
            dlg.check(box, f"ph_{f}", owner.ds.var(f).title, f in owner.opts["posthoc_factors"])
    box = dlg.group("Equal Variances Assumed")
    for key in POSTHOC_ORDER[:-1]:
        dlg.check(box, key, POSTHOC[key], key in owner.opts["posthoc"])
    box = dlg.group("Equal Variances Not Assumed")
    dlg.check(box, "games_howell", "Games-Howell", "games_howell" in owner.opts["posthoc"])
    box = dlg.group("Significance level")
    dlg.number(box, "alpha", "Significance level:", owner.opts.get("alpha", 0.05), 0.001, 0.5, 3)
    values = dlg.run()
    if values is None:
        return
    owner.opts["posthoc"] = [k for k in POSTHOC_ORDER if values.get(k)]
    owner.opts["alpha"] = values["alpha"]
    if factors is not None:
        owner.opts["posthoc_factors"] = [f for f in factors if values.get(f"ph_{f}")]


class OneWayAnovaDialog(VariablePickerDialog):
    DEFAULTS = {"descriptives": True, "homogeneity": True, "welch": False, "bf": False,
                "effect": True, "means_plot": False, "posthoc": [], "alpha": 0.05}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "One-Way ANOVA", parent)
        self.add_target("deps", "Dependent List:", accepts=is_numeric)
        self.add_target("factor", "Factor:", single=True)
        self.add_option_button("Post Hoc...", lambda: _posthoc_dialog(self))
        self.add_option_button("Options...", self._options)
        self.add_checkbox("effect", "Estimate effect size for overall tests")
        self.finish()

    def _options(self) -> None:
        dlg = SubDialog("One-Way ANOVA: Options", self, columns=1)
        box = dlg.group("Statistics")
        for key, text in (("descriptives", "Descriptive"),
                          ("homogeneity", "Homogeneity of variance test"),
                          ("bf", "Brown-Forsythe"), ("welch", "Welch")):
            dlg.check(box, key, text, self.opts[key])
        dlg.check(dlg.group("Plots"), "means_plot", "Means plot", self.opts["means_plot"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def compute(self):
        o = self.opts
        return anova.oneway(self.ds, self.selected("deps"), self.selected_one("factor"),
                            o["descriptives"], o["homogeneity"], o["welch"], o["bf"], o["effect"],
                            o["posthoc"], o["means_plot"], 1 - o["alpha"])


class UnivariateDialog(VariablePickerDialog):
    DEFAULTS = {"descriptives": True, "effect": True, "homogeneity": True, "posthoc": [],
                "posthoc_factors": [], "alpha": 0.05}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Univariate", parent)
        self.add_target("dep", "Dependent Variable:", accepts=is_numeric, single=True)
        self.add_target("factors", "Fixed Factor(s):", required=False)
        self.add_target("covariates", "Covariate(s):", accepts=is_numeric, required=False)
        self.add_option_button("Post Hoc...", lambda: _posthoc_dialog(
            self, self.selected("factors")))
        self.add_option_button("Options...", self._options)
        self.finish()

    def _options(self) -> None:
        dlg = SubDialog("Univariate: Options", self, columns=1)
        box = dlg.group("Display")
        for key, text in (("descriptives", "Descriptive statistics"),
                          ("effect", "Estimates of effect size"),
                          ("homogeneity", "Homogeneity tests")):
            dlg.check(box, key, text, self.opts[key])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def validate(self):
        if not self.selected("factors") and not self.selected("covariates"):
            return "Specify at least one fixed factor or covariate."
        return None

    def compute(self):
        o = self.opts
        factors = self.selected("factors")
        return anova.univariate(self.ds, self.selected_one("dep"), factors,
                                self.selected("covariates"), o["descriptives"], o["effect"],
                                o["homogeneity"],
                                [f for f in o["posthoc_factors"] if f in factors],
                                o["posthoc"], 1 - o["alpha"])

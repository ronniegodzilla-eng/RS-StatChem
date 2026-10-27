"""Analyze > Compare Means and General Linear Model > Univariate dialogs (SPSS layout)."""
from __future__ import annotations

import pandas as pd
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QPushButton)

from rsstatchem.core.dataset import format_code
from rsstatchem.core.stats import anova, boot_procs, glm_output, mi_pool, posthoc_ext, ttest
from rsstatchem.core.stats import imputation as mi
from rsstatchem.core.stats.anova import POSTHOC, POSTHOC_ORDER  # noqa: F401 (re-export)
from rsstatchem.ui.dialogs import glm_model_ui as gm
from rsstatchem.ui.dialogs.base import (DefineGroupsDialog, DefineRangeDialog, SubDialog,
                                        VariablePickerDialog, is_numeric)
from rsstatchem.ui.dialogs.resample_ui import ResampleMixin


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


class OneSampleTDialog(ResampleMixin, VariablePickerDialog):
    DEFAULTS = {"test_value": 0.0, "level": 95.0, "listwise": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "One-Sample T Test", parent)
        self.add_target("vars", "Test Variable(s):", accepts=is_numeric)
        self.add_option_button("Options...", self.ci_missing_options)
        self.add_bootstrap_button()
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
        names, tv, lw = self.selected("vars"), self.opts["test_value"], self.opts["listwise"]
        return self.resampled(
            lambda: ttest.one_sample(self.ds, names, tv, self.level(), lw),
            boot=lambda sec, spec: boot_procs.one_sample_t(sec, self.ds, names, tv, spec, lw),
            pool=lambda views: mi.pooled_one_sample_t(self.ds, names, tv, self.level()))


class IndependentTDialog(ResampleMixin, GroupingMixin, VariablePickerDialog):
    DEFAULTS = {"level": 95.0, "listwise": False, "group_def": None}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Independent-Samples T Test", parent)
        self.add_target("vars", "Test Variable(s):", accepts=is_numeric)
        self.setup_grouping()
        self.add_option_button("Options...", self.ci_missing_options)
        self.add_bootstrap_button()
        self.finish()

    def compute(self):
        d = self._definition()
        names, group, lw = self.selected("vars"), self.selected_one("group"), \
            self.opts["listwise"]
        groups, cut = d.get("groups"), d.get("cut")

        def pool(views):
            if groups is None:
                return []
            return mi.pooled_independent_t(self.ds, names, group, groups, self.level())

        return self.resampled(
            lambda: ttest.independent(self.ds, names, group, groups, cut, self.level(), lw),
            boot=lambda sec, spec: boot_procs.independent_t(sec, self.ds, names, group, groups,
                                                            cut, spec, lw),
            pool=pool)


class PairedTDialog(ResampleMixin, VariablePickerDialog):
    DEFAULTS = {"level": 95.0, "listwise": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Paired-Samples T Test", parent)
        self.add_pair_target("pairs", "Paired Variables (Variable1 – Variable2):")
        self.add_option_button("Options...", self.ci_missing_options)
        self.add_bootstrap_button()
        self.finish()

    def compute(self):
        pairs, lw = self.pairs("pairs"), self.opts["listwise"]
        return self.resampled(
            lambda: ttest.paired(self.ds, pairs, self.level(), lw),
            boot=lambda sec, spec: boot_procs.paired_t(sec, self.ds, pairs, spec, lw),
            pool=lambda views: mi.pooled_paired_t(self.ds, pairs, self.level()))


# ------------------------------------------------------------------ One-Way ANOVA
class OneWayAnovaDialog(ResampleMixin, VariablePickerDialog):
    DEFAULTS = {"descriptives": True, "homogeneity": True, "welch": False, "bf": False,
                "effects": False, "effect": True, "means_plot": False, "posthoc": [],
                "alpha": 0.05, "polynomial": False, "degree": 1, "contrasts": [],
                "control": "last", "dunnett_test": "2-sided", "k_ratio": 100.0,
                "listwise": False, "level": 95.0}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "One-Way ANOVA", parent)
        self.add_target("deps", "Dependent List:", accepts=is_numeric)
        self.add_target("factor", "Factor:", single=True)
        self.add_option_button("Contrasts...", self._contrasts)
        self.add_option_button("Post Hoc...", lambda: gm.posthoc_dialog(
            self, "One-Way ANOVA", None, significance=True))
        self.add_option_button("Options...", self._options)
        self.add_bootstrap_button()
        self.add_checkbox("effect", "Estimate effect size for overall tests")
        self.finish()

    def _contrasts(self) -> None:
        dlg = ContrastsOneWayDialog(self.opts, self)
        if dlg.exec():
            self.opts.update(dlg.result_values())

    def _options(self) -> None:
        dlg = SubDialog("One-Way ANOVA: Options", self, columns=1)
        box = dlg.group("Statistics")
        for key, text in (("descriptives", "Descriptive"),
                          ("effects", "Fixed and random effects"),
                          ("homogeneity", "Homogeneity of variance test"),
                          ("bf", "Brown-Forsythe"), ("welch", "Welch")):
            dlg.check(box, key, text, self.opts[key])
        dlg.check(dlg.group("Plots"), "means_plot", "Means plot", self.opts["means_plot"])
        dlg.radio(dlg.group("Missing Values"), "listwise",
                  [(False, "Exclude cases analysis by analysis"),
                   (True, "Exclude cases listwise")], self.opts["listwise"])
        dlg.number(dlg.group("Confidence Intervals"), "level", "Level(%):", self.opts["level"],
                   50, 99.99, 2)
        values = dlg.run()
        if values:
            self.opts.update(values)

    def compute(self):
        o = self.opts
        popts = posthoc_ext.PostHocOptions(1 - o["alpha"], o["control"], o["dunnett_test"],
                                           o["k_ratio"])
        deps, factor = self.selected("deps"), self.selected_one("factor")
        return self.resampled(
            lambda: anova.oneway(self.ds, deps, factor, o["descriptives"], o["homogeneity"],
                                 o["welch"], o["bf"], o["effect"], o["posthoc"],
                                 o["means_plot"], o["level"] / 100,
                                 o["degree"] if o["polynomial"] else 0, o["contrasts"],
                                 o["effects"], o["listwise"], popts, 1 - o["alpha"]),
            boot=lambda sec, spec: boot_procs.oneway(sec, self.ds, deps, factor, spec,
                                                     bool(o["posthoc"]), o["listwise"]),
            pool=lambda views: mi_pool.pooled_oneway(views, deps, factor, o["level"] / 100))


class ContrastsOneWayDialog(SubDialog):
    """One-Way ANOVA: Contrasts — polynomial trend (degree) and coefficient lists."""

    DEGREES = ["Linear", "Quadratic", "Cubic", "4th", "5th"]

    def __init__(self, opts: dict, parent=None):
        super().__init__("One-Way ANOVA: Contrasts", parent, columns=1)
        box = self.group("Polynomial")
        row = QHBoxLayout()
        self.poly = QCheckBox("Polynomial")
        self.poly.setChecked(bool(opts.get("polynomial")))
        self.degree = QComboBox()
        self.degree.addItems(self.DEGREES)
        self.degree.setCurrentIndex(int(opts.get("degree", 1)) - 1)
        row.addWidget(self.poly)
        row.addWidget(QLabel("Degree:"))
        row.addWidget(self.degree)
        row.addStretch()
        box.addLayout(row)
        box = self.group("Coefficients")
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("e.g. 1 -1 0")
        add, change, remove = QPushButton("Add"), QPushButton("Change"), QPushButton("Remove")
        add.clicked.connect(self._add)
        change.clicked.connect(self._change)
        remove.clicked.connect(lambda: self.list.takeItem(self.list.currentRow()))
        self.list = QListWidget()
        for c in opts.get("contrasts", []):
            self.list.addItem(" ".join(f"{x:g}" for x in c))
        self.list.currentRowChanged.connect(
            lambda r: self.edit.setText(self.list.item(r).text()) if r >= 0 else None)
        row = QHBoxLayout()
        row.addWidget(self.edit, 1)
        for b in (add, change, remove):
            row.addWidget(b)
        box.addLayout(row)
        box.addWidget(QLabel("Contrasts (one per row):"))
        box.addWidget(self.list)
        self.sum_label = QLabel("")
        box.addWidget(self.sum_label)
        self.edit.textChanged.connect(self._sum)

    def _parse(self) -> list[float] | None:
        try:
            vals = [float(t.replace(",", ".")) for t in self.edit.text().replace(";", " ").split()]
        except ValueError:
            return None
        return vals or None

    def _sum(self) -> None:
        vals = self._parse()
        self.sum_label.setText(f"Coefficient Total: {sum(vals):g}" if vals else "")

    def _add(self) -> None:
        vals = self._parse()
        if vals:
            self.list.addItem(" ".join(f"{x:g}" for x in vals))
            self.edit.clear()

    def _change(self) -> None:
        row = self.list.currentRow()
        vals = self._parse()
        if row >= 0 and vals:
            self.list.item(row).setText(" ".join(f"{x:g}" for x in vals))

    def result_values(self) -> dict:
        rows = [[float(x) for x in self.list.item(i).text().split()]
                for i in range(self.list.count())]
        return {"polynomial": self.poly.isChecked(), "degree": self.degree.currentIndex() + 1,
                "contrasts": rows}


# ------------------------------------------------------------------ GLM Univariate
class UnivariateDialog(ResampleMixin, VariablePickerDialog):
    DEFAULTS = {**gm.GLM_DEFAULTS, "descriptives": True, "effect": True, "homogeneity": True}
    TITLE = "Univariate"

    def __init__(self, ds, parent=None):
        super().__init__(ds, self.TITLE, parent)
        self.add_target("dep", "Dependent Variable:", accepts=is_numeric, single=True)
        self.add_target("factors", "Fixed Factor(s):", required=False)
        self.add_target("random", "Random Factor(s):", required=False)
        self.add_target("covariates", "Covariate(s):", accepts=is_numeric, required=False)
        self.add_target("wls", "WLS Weight:", accepts=is_numeric, single=True, required=False)
        t = self.TITLE
        self.add_option_button("Model...", lambda: gm.model_dialog(
            self, t, self.selected("factors"), self.selected("covariates"),
            self.selected("random")))
        self.add_option_button("Contrasts...", lambda: gm.contrasts_dialog(
            self, t, self.selected("factors") + self.selected("random")))
        self.add_option_button("Plots...", lambda: gm.plots_dialog(
            self, t, self.selected("factors") + self.selected("random")))
        self.add_option_button("Post Hoc...", lambda: gm.posthoc_dialog(
            self, t, self.selected("factors")))
        self.add_option_button("EM Means...", lambda: gm.emm_dialog(
            self, t, self.selected("factors") + self.selected("random"), self.opts.get("terms")))
        self.add_option_button("Save...", lambda: gm.save_dialog(self, t))
        self.add_option_button("Options...", lambda: gm.options_dialog(self, t))
        if self.TITLE == "Univariate":
            self.add_bootstrap_button()
        self.finish()

    def validate(self):
        if not self.selected("factors") and not self.selected("covariates") and \
                not self.selected("random") and not self.opts.get("terms"):
            return "Specify at least one fixed factor, random factor or covariate."
        if self.opts.get("terms"):
            try:
                gm.parse_terms(" ".join(self.opts["terms"]),
                               self.selected("factors") + self.selected("random"),
                               self.selected("covariates"))
            except ValueError as exc:
                return f"The custom model refers to variables that are no longer selected " \
                       f"({exc}). Open Model… and choose Full factorial or rebuild the terms."
        return None

    def compute(self):
        factors = self.selected("factors")
        random = self.selected("random")
        covs = self.selected("covariates")
        opts = gm.glm_options_from(self.opts, factors, covs, random)
        opts.wls = self.selected_one("wls")
        dep = self.selected_one("dep")
        res = self.resampled(
            lambda: glm_output.univariate(self.ds, dep, factors, covs, opts),
            boot=lambda r, spec: boot_procs.models(r.section, self.ds, r.models, spec),
            pool="models")
        return self.plain(res)

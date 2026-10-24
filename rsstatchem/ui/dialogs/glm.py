"""Analyze > General Linear Model > Multivariate / Repeated Measures and Analyze > Mixed Models >
Linear dialogs (SPSS layout). This module also exports the combined ``MENUS`` and
``selftest()`` for the dialogs in :mod:`rsstatchem.ui.dialogs.multivariate` (Factor, PROCESS).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMessageBox, QPushButton, QSpinBox, QVBoxLayout,
                               QWidget)

from rsstatchem.core.stats import manova, mixed, repeated
from rsstatchem.core.stats.anova import POSTHOC, POSTHOC_ORDER
from rsstatchem.ui.dialogs.base import (BASE_TEXT_ROLE, NAME_ROLE, SubDialog,
                                        VariablePickerDialog, is_numeric)

ADJUST_CHOICES = [("lsd", "LSD(none)"), ("bonferroni", "Bonferroni"), ("sidak", "Sidak")]


# ------------------------------------------------------------------ shared sub-dialogs
def _effects(factor_names: list[str]) -> list[str]:
    return [" * ".join(c).replace(" * ", "*") for r in range(1, len(factor_names) + 1)
            for c in itertools.combinations(factor_names, r)]


def emm_dialog(owner: VariablePickerDialog, factor_names: list[str]) -> None:
    """Estimated Marginal Means: factors / interactions, compare main effects, adjustment."""
    dlg = SubDialog(f"{owner.windowTitle()}: Estimated Marginal Means", owner)
    box = dlg.group("Display Means for")
    current = owner.opts.get("emm", [])
    for eff in ["(OVERALL)"] + _effects(factor_names):
        dlg.check(box, f"emm::{eff}", eff.replace("*", " * "), eff in current)
    box = dlg.group("Comparisons")
    dlg.check(box, "compare", "Compare main effects", owner.opts.get("compare", False))
    box.addWidget(QLabel("Confidence interval adjustment:"))
    dlg.radio(box, "adjust", ADJUST_CHOICES, owner.opts.get("adjust", "bonferroni"))
    values = dlg.run()
    if values is None:
        return
    owner.opts["emm"] = [k.split("::", 1)[1] for k, v in values.items()
                         if k.startswith("emm::") and v]
    owner.opts["compare"] = values["compare"]
    owner.opts["adjust"] = values["adjust"]


def posthoc_dialog(owner: VariablePickerDialog, factors: list[str]) -> None:
    dlg = SubDialog(f"{owner.windowTitle()}: Post Hoc Multiple Comparisons for Observed Means",
                    owner)
    box = dlg.group("Post Hoc Tests for")
    if not factors:
        box.addWidget(QLabel("(no between-subjects factors)"))
    for f in factors:
        dlg.check(box, f"ph_{f}", owner.ds.var(f).name, f in owner.opts["posthoc_factors"])
    box = dlg.group("Equal Variances Assumed")
    for key in POSTHOC_ORDER[:-1]:
        dlg.check(box, key, POSTHOC[key], key in owner.opts["posthoc"])
    box = dlg.group("Equal Variances Not Assumed")
    dlg.check(box, "games_howell", "Games-Howell", "games_howell" in owner.opts["posthoc"])
    values = dlg.run()
    if values is None:
        return
    owner.opts["posthoc"] = [k for k in POSTHOC_ORDER if values.get(k)]
    owner.opts["posthoc_factors"] = [f for f in factors if values.get(f"ph_{f}")]


def glm_options(owner: VariablePickerDialog, title: str) -> None:
    dlg = SubDialog(f"{title}: Options", owner, columns=1)
    box = dlg.group("Display")
    for key, text in (("descriptives", "Descriptive statistics"),
                      ("effect", "Estimates of effect size"),
                      ("homogeneity", "Homogeneity tests")):
        dlg.check(box, key, text, owner.opts[key])
    box = dlg.group("Significance level")
    dlg.number(box, "alpha", "Significance level:", owner.opts["alpha"], 0.001, 0.5, 3)
    box.addWidget(QLabel("Confidence intervals are (1 − significance level) × 100%."))
    values = dlg.run()
    if values:
        owner.opts.update(values)


# ------------------------------------------------------------------ Multivariate
class MultivariateDialog(VariablePickerDialog):
    DEFAULTS = {"descriptives": True, "effect": True, "homogeneity": True, "posthoc": [],
                "posthoc_factors": [], "alpha": 0.05, "emm": [], "compare": False,
                "adjust": "bonferroni"}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Multivariate", parent)
        self.add_target("deps", "Dependent Variables:", accepts=is_numeric, min_items=2)
        self.add_target("factors", "Fixed Factor(s):", required=False)
        self.add_target("covariates", "Covariate(s):", accepts=is_numeric, required=False)
        self.add_option_button("Post Hoc...", lambda: posthoc_dialog(
            self, self.selected("factors")))
        self.add_option_button("EM Means...", lambda: emm_dialog(
            self, [self.ds.var(f).name for f in self.selected("factors")]))
        self.add_option_button("Options...", lambda: glm_options(self, "Multivariate"))
        self.finish()

    def validate(self):
        if not self.selected("factors") and not self.selected("covariates"):
            return "Specify at least one fixed factor or covariate."
        return None

    def compute(self):
        o = self.opts
        factors = self.selected("factors")
        names = {self.ds.var(f).name for f in factors}
        emm = [e for e in o["emm"] if e == "(OVERALL)" or
               all(p in names for p in e.split("*"))]
        return manova.manova(self.ds, self.selected("deps"), factors,
                             self.selected("covariates"), o["descriptives"], o["effect"],
                             o["homogeneity"], [f for f in o["posthoc_factors"] if f in factors],
                             o["posthoc"], emm, o["compare"], o["adjust"], 1 - o["alpha"])


# ------------------------------------------------------------------ Repeated Measures
class RepeatedMeasuresDefineDialog(QDialog):
    """Repeated Measures Define Factor(s): within-subject factor names and levels, measure."""

    def __init__(self, factors: list[tuple[str, int]] | None = None, measure: str = "",
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Repeated Measures Define Factor(s)")
        self.name = QLineEdit("factor1")
        self.levels = QSpinBox()
        self.levels.setRange(0, 99)
        self.levels.setValue(0)
        self.levels.setSpecialValueText(" ")
        self.list = QListWidget()
        self.list.setFixedHeight(70)
        add, change, remove = QPushButton("Add"), QPushButton("Change"), QPushButton("Remove")
        add.clicked.connect(self._add)
        change.clicked.connect(self._change)
        remove.clicked.connect(self._remove)
        self.list.currentRowChanged.connect(self._select)
        grid = QGridLayout()
        grid.addWidget(QLabel("Within-Subject Factor Name:"), 0, 0)
        grid.addWidget(self.name, 0, 1)
        grid.addWidget(QLabel("Number of Levels:"), 1, 0)
        grid.addWidget(self.levels, 1, 1)
        buttons = QVBoxLayout()
        for b in (add, change, remove):
            buttons.addWidget(b)
        buttons.addStretch()
        row = QHBoxLayout()
        row.addLayout(buttons)
        row.addWidget(self.list, 1)
        grid.addLayout(row, 2, 0, 1, 2)

        self.measure = QLineEdit()
        self.mlist = QListWidget()
        self.mlist.setFixedHeight(46)
        madd, mremove = QPushButton("Add"), QPushButton("Remove")
        madd.clicked.connect(self._madd)
        mremove.clicked.connect(lambda: self.mlist.clear())
        mgrid = QGridLayout()
        mgrid.addWidget(QLabel("Measure Name:"), 0, 0)
        mgrid.addWidget(self.measure, 0, 1)
        mrow = QHBoxLayout()
        mb = QVBoxLayout()
        mb.addWidget(madd)
        mb.addWidget(mremove)
        mb.addStretch()
        mrow.addLayout(mb)
        mrow.addWidget(self.mlist, 1)
        mgrid.addLayout(mrow, 1, 0, 1, 2)
        mbox = QGroupBox("Measure")
        mbox.setLayout(mgrid)

        bb = QDialogButtonBox()
        self.define = bb.addButton("Define", QDialogButtonBox.ButtonRole.AcceptRole)
        reset = bb.addButton(QDialogButtonBox.StandardButton.Reset)
        bb.addButton(QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        reset.clicked.connect(self._reset)
        layout = QVBoxLayout(self)
        layout.addLayout(grid)
        layout.addWidget(mbox)
        layout.addWidget(bb)
        for name, k in factors or []:
            self.list.addItem(f"{name}({k})")
        if measure:
            self.mlist.addItem(measure)
        self.name.setText(f"factor{self.list.count() + 1}")
        self._update()

    def _parse(self, text: str) -> tuple[str, int]:
        name, k = text[:-1].rsplit("(", 1)
        return name, int(k)

    def _valid_name(self) -> str | None:
        name = self.name.text().strip()
        if not name or not (name[0].isalpha() or name[0] in "@#$") or " " in name:
            return None
        return name

    def _add(self) -> None:
        name = self._valid_name()
        k = self.levels.value()
        if name is None or k < 2:
            QMessageBox.warning(self, self.windowTitle(),
                                "Enter a valid factor name and at least 2 levels.")
            return
        if self.list.count() >= 2:
            QMessageBox.warning(self, self.windowTitle(),
                                "At most two within-subject factors are supported.")
            return
        if any(self._parse(self.list.item(i).text())[0].lower() == name.lower()
               for i in range(self.list.count())):
            QMessageBox.warning(self, self.windowTitle(), "That factor name is already used.")
            return
        self.list.addItem(f"{name}({k})")
        self.name.setText(f"factor{self.list.count() + 1}")
        self.levels.setValue(0)
        self._update()

    def _change(self) -> None:
        row = self.list.currentRow()
        name = self._valid_name()
        if row < 0 or name is None or self.levels.value() < 2:
            return
        self.list.item(row).setText(f"{name}({self.levels.value()})")

    def _remove(self) -> None:
        row = self.list.currentRow()
        if row >= 0:
            self.list.takeItem(row)
        self._update()

    def _select(self, row: int) -> None:
        if row >= 0:
            name, k = self._parse(self.list.item(row).text())
            self.name.setText(name)
            self.levels.setValue(k)

    def _madd(self) -> None:
        text = self.measure.text().strip()
        if text and " " not in text:
            self.mlist.clear()
            self.mlist.addItem(text)
            self.measure.clear()

    def _reset(self) -> None:
        self.list.clear()
        self.mlist.clear()
        self.name.setText("factor1")
        self.levels.setValue(0)
        self._update()

    def _update(self) -> None:
        self.define.setEnabled(self.list.count() > 0 or self.levels.value() >= 2)

    def accept(self) -> None:
        if self.list.count() == 0 and self._valid_name() and self.levels.value() >= 2:
            self._add()
        if self.list.count() == 0:
            return
        super().accept()

    def factors(self) -> list[tuple[str, int]]:
        return [self._parse(self.list.item(i).text()) for i in range(self.list.count())]

    def measure_name(self) -> str:
        return self.mlist.item(0).text() if self.mlist.count() else ""


class ProfilePlotsDialog(QDialog):
    def __init__(self, factors: list[str], plots: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Repeated Measures: Profile Plots")
        self.h = QComboBox()
        self.h.addItems(factors)
        self.s = QComboBox()
        self.s.addItems(["(none)"] + factors)
        self.list = QListWidget()
        for h, s in plots:
            if h in factors and (s is None or s in factors):
                self._add_item(h, s)
        add, remove = QPushButton("Add"), QPushButton("Remove")
        add.clicked.connect(self._add)
        remove.clicked.connect(lambda: self.list.takeItem(self.list.currentRow()))
        form = QFormLayout()
        form.addRow("Horizontal Axis:", self.h)
        form.addRow("Separate Lines:", self.s)
        row = QHBoxLayout()
        row.addWidget(add)
        row.addWidget(remove)
        row.addStretch()
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("Continue")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(row)
        layout.addWidget(QLabel("Plots:"))
        layout.addWidget(self.list)
        layout.addWidget(bb)

    def _add_item(self, h, s) -> None:
        item = QListWidgetItem(h + (f"*{s}" if s else ""))
        item.setData(Qt.ItemDataRole.UserRole, (h, s))
        self.list.addItem(item)

    def _add(self) -> None:
        h = self.h.currentText()
        s = self.s.currentText()
        s = None if s == "(none)" or s == h else s
        if h:
            self._add_item(h, s)

    def plots(self) -> list:
        return [tuple(self.list.item(i).data(Qt.ItemDataRole.UserRole))
                for i in range(self.list.count())]


class RepeatedMeasuresDialog(VariablePickerDialog):
    """Main Repeated Measures dialog; opens the Define Factor(s) dialog first (like SPSS)."""

    DEFAULTS = {"descriptives": True, "effect": True, "homogeneity": False, "posthoc": [],
                "posthoc_factors": [], "alpha": 0.05, "emm": [], "compare": True,
                "adjust": "bonferroni", "plots": []}
    _last_define: tuple[list[tuple[str, int]], str] | None = None

    def __init__(self, ds, parent=None, factors: list[tuple[str, int]] | None = None,
                 measure: str = ""):
        super().__init__(ds, "Repeated Measures", parent)
        self.within_factors: list[tuple[str, int]] = []
        self.measure = ""
        self.within = self.add_target("within", "Within-Subjects Variables:",
                                      accepts=is_numeric, required=False)
        self.add_target("between", "Between-Subjects Factor(s):", required=False)
        self.add_target("covariates", "Covariates:", accepts=is_numeric, required=False)
        self.within_label = next(w for w in self.findChildren(QLabel)
                                 if w.text() == "Within-Subjects Variables:")
        self.add_option_button("Define Factors...", self._redefine)
        self.add_option_button("Plots...", self._plots)
        self.add_option_button("Post Hoc...", lambda: posthoc_dialog(
            self, self.selected("between")))
        self.add_option_button("EM Means...", lambda: emm_dialog(self, self._factor_names()))
        self.add_option_button("Options...", lambda: glm_options(self, "Repeated Measures"))
        self.finish()
        if factors:
            self.set_factors(factors, measure)

    # ---------------------------------------------------------------- factor definition
    def exec(self) -> int:
        if not self.within_factors:
            last = type(self)._last_define or ([], "")
            dlg = RepeatedMeasuresDefineDialog(last[0], last[1], self)
            if not dlg.exec():
                return 0
            self.set_factors(dlg.factors(), dlg.measure_name())
        return super().exec()

    def _redefine(self) -> None:
        dlg = RepeatedMeasuresDefineDialog(self.within_factors, self.measure, self)
        if dlg.exec():
            self.set_factors(dlg.factors(), dlg.measure_name())

    def set_factors(self, factors: list[tuple[str, int]], measure: str = "") -> None:
        type(self)._last_define = (list(factors), measure)
        target = self._within_target()
        self._move_back(target, all_items=True)
        self.within.clear()
        self.within_factors = list(factors)
        self.measure = measure
        names = ",".join(n for n, _ in factors)
        self.within_label.setText(f"Within-Subjects Variables ({names}):")
        ks = [k for _, k in factors]
        for col in range(int(np.prod(ks))):
            item = QListWidgetItem(f"_?_({self._slot(col)})")
            item.setData(NAME_ROLE, None)
            self.within.addItem(item)
        remembered = self._memory.get(type(self).__name__, {}).get("within", [])
        for name in remembered:
            if self.ds.has_name(name):
                self._take(name, target)
        self._update_arrows()

    def _slot(self, col: int) -> str:
        ks = [k for _, k in self.within_factors]
        return ",".join(str(int(v) + 1) for v in np.unravel_index(col, ks))

    def _factor_names(self) -> list[str]:
        return [n for n, _ in self.within_factors] + \
            [self.ds.var(b).name for b in self.selected("between")]

    def _within_target(self):
        return next(t for t in self._targets if t.key == "within")

    # ---------------------------------------------------------------- slot handling
    def selected(self, key: str) -> list[str]:
        names = super().selected(key)
        return [n for n in names if n is not None] if key == "within" else names

    def slot_variables(self) -> list[str | None]:
        return [self.within.item(i).data(NAME_ROLE) for i in range(self.within.count())]

    def _take(self, name: str, target) -> bool:
        if target.key != "within":
            return super()._take(name, target)
        item = self._source_item(name)
        if item is None or not target.accepts(self.ds.var(name)):
            return False
        row = next((i for i in range(self.within.count())
                    if self.within.item(i).data(NAME_ROLE) is None), None)
        if row is None:
            return False
        self.source.takeItem(self.source.row(item))
        self.within.takeItem(row)
        item.setText(f"{name}({self._slot(row)})")
        self.within.insertItem(row, item)
        return True

    def _move_back(self, target, all_items: bool = False) -> None:
        if target.key != "within":
            super()._move_back(target, all_items)
            return
        items = [self.within.item(i) for i in range(self.within.count())] if all_items \
            else self.within.selectedItems()
        for item in items:
            name = item.data(NAME_ROLE)
            if name is None:
                continue
            row = self.within.row(item)
            self.within.takeItem(row)
            item.setText(item.data(BASE_TEXT_ROLE))
            order = self.ds.index_of(name)
            pos = 0
            while pos < self.source.count() and \
                    self.ds.index_of(self.source.item(pos).data(NAME_ROLE)) < order:
                pos += 1
            self.source.insertItem(pos, item)
            placeholder = QListWidgetItem(f"_?_({self._slot(row)})")
            placeholder.setData(NAME_ROLE, None)
            self.within.insertItem(row, placeholder)
        self._update_arrows()

    def _update_ok(self) -> None:
        super()._update_ok()
        if hasattr(self, "ok_button") and hasattr(self, "within"):
            filled = self.within.count() > 0 and all(
                v is not None for v in self.slot_variables())
            self.ok_button.setEnabled(self.ok_button.isEnabled() and filled)

    def reset(self) -> None:
        super().reset()
        type(self)._last_define = (self.within_factors, self.measure)

    # ---------------------------------------------------------------- options
    def _plots(self) -> None:
        dlg = ProfilePlotsDialog(self._factor_names(), self.opts["plots"], self)
        if dlg.exec():
            self.opts["plots"] = dlg.plots()

    def validate(self):
        if not self.within_factors:
            return "Define the within-subject factor(s) first."
        if any(v is None for v in self.slot_variables()):
            return "Assign a variable to every level of the within-subjects factor(s)."
        return None

    def compute(self):
        o = self.opts
        between = self.selected("between")
        names = set(self._factor_names())
        emm = [e for e in o["emm"] if e == "(OVERALL)" or all(p in names for p in e.split("*"))]
        plots = [(h, s) for h, s in o["plots"] if h in names and (s is None or s in names)]
        return repeated.repeated_measures(
            self.ds, self.within_factors, self.slot_variables(), between,
            self.selected("covariates"), self.measure or "MEASURE_1", o["descriptives"],
            o["effect"], o["homogeneity"], emm, o["compare"], o["adjust"],
            [f for f in o["posthoc_factors"] if f in between], o["posthoc"], plots,
            1 - o["alpha"])


# ------------------------------------------------------------------ Linear Mixed Models
class LinearMixedDialog(VariablePickerDialog):
    DEFAULTS = {"main_only": False, "intercept": True, "slopes": [], "structure": "VC",
                "method": "REML", "descriptives": False, "estimates": True, "covtests": True,
                "level": 95.0}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Linear Mixed Models", parent)
        self.add_target("subject", "Subjects:", single=True)
        self.add_target("dep", "Dependent Variable:", accepts=is_numeric, single=True)
        self.add_target("factors", "Factor(s):", required=False)
        self.add_target("covariates", "Covariate(s):", accepts=is_numeric, required=False)
        self.add_option_button("Fixed...", self._fixed)
        self.add_option_button("Random...", self._random)
        self.add_option_button("Estimation...", self._estimation)
        self.add_option_button("Statistics...", self._statistics)
        hint = QLabel("Data must be in long format: one row per observation, with a subject "
                      "identifier.")
        hint.setStyleSheet("color:#666")
        hint.setWordWrap(True)
        self.options_layout.addWidget(hint)
        self.finish()

    def _fixed(self) -> None:
        dlg = SubDialog("Linear Mixed Models: Fixed Effects", self, columns=1)
        box = dlg.group("Fixed effects model")
        dlg.radio(box, "main_only", [(False, "Full factorial (factors) + covariate main effects"),
                                     (True, "Main effects only")], self.opts["main_only"])
        box.addWidget(QLabel("Include intercept and Type III sums of squares are always used."))
        values = dlg.run()
        if values:
            self.opts.update(values)

    def _random(self) -> None:
        dlg = SubDialog("Linear Mixed Models: Random Effects", self)
        box = dlg.group("Random effects")
        dlg.check(box, "intercept", "Include intercept", self.opts["intercept"])
        covs = self.selected("covariates")
        if covs:
            box.addWidget(QLabel("Random slopes for covariates:"))
        for c in covs:
            dlg.check(box, f"slope::{c}", self.ds.var(c).name, c in self.opts["slopes"])
        box = dlg.group("Covariance Type")
        dlg.radio(box, "structure", [("VC", "Variance Components"), ("UN", "Unstructured")],
                  self.opts["structure"])
        subj = self.selected_one("subject")
        box.addWidget(QLabel(f"Subject grouping: {subj or '(choose Subjects first)'}"))
        values = dlg.run()
        if values is None:
            return
        self.opts["intercept"] = values["intercept"]
        self.opts["structure"] = values["structure"]
        self.opts["slopes"] = [c for c in covs if values.get(f"slope::{c}")]

    def _estimation(self) -> None:
        dlg = SubDialog("Linear Mixed Models: Estimation", self, columns=1)
        box = dlg.group("Method")
        dlg.radio(box, "method", [("REML", "Restricted Maximum Likelihood (REML)"),
                                  ("ML", "Maximum Likelihood (ML)")], self.opts["method"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def _statistics(self) -> None:
        dlg = SubDialog("Linear Mixed Models: Statistics", self, columns=1)
        box = dlg.group("Summary Statistics")
        dlg.check(box, "descriptives", "Descriptive statistics", self.opts["descriptives"])
        box = dlg.group("Model Statistics")
        dlg.check(box, "estimates", "Parameter estimates for fixed effects",
                  self.opts["estimates"])
        dlg.check(box, "covtests", "Tests for covariance parameters", self.opts["covtests"])
        dlg.number(box, "level", "Confidence interval (%):", self.opts["level"], 50, 99.99, 2)
        values = dlg.run()
        if values:
            self.opts.update(values)

    def validate(self):
        if self.selected_one("subject") in self.selected("factors") + self.selected("covariates"):
            return "The subject variable cannot also be a factor or covariate."
        return None

    def compute(self):
        o = self.opts
        covs = self.selected("covariates")
        return mixed.linear_mixed(self.ds, self.selected_one("dep"),
                                  self.selected_one("subject"), self.selected("factors"), covs,
                                  o["intercept"], [s for s in o["slopes"] if s in covs],
                                  o["structure"], o["method"], o["main_only"],
                                  o["descriptives"], o["estimates"], o["covtests"],
                                  o["level"] / 100)


# ------------------------------------------------------------------ menus + selftest
def _menus() -> dict:
    from rsstatchem.ui.dialogs import multivariate as mv

    return {
        "Analyze/General Linear Model": [("Multivariate...", MultivariateDialog),
                                         ("Repeated Measures...", RepeatedMeasuresDialog)],
        "Analyze/Mixed Models": [("Linear...", LinearMixedDialog)],
        "Analyze/Dimension Reduction": [("Factor...", mv.FactorDialog)],
        "Analyze/Mediation && Moderation": [
            ("PROCESS Model 4 (Mediation)...", mv.ProcessModel4Dialog),
            ("PROCESS Model 1 (Moderation)...", mv.ProcessModel1Dialog),
            None,
            ("PROCESS Model 7 (Moderated Mediation, first stage)...", mv.ProcessModel7Dialog),
            ("PROCESS Model 14 (Moderated Mediation, second stage)...",
             mv.ProcessModel14Dialog),
        ],
    }


MENUS = _menus()


def _check(cond: bool, what: str) -> None:
    if not cond:
        raise AssertionError(f"GLM/multivariate selftest failed: {what}")


def selftest() -> None:
    """Headless checks of Repeated Measures, MANOVA, Mixed, Factor and PROCESS core code."""
    from rsstatchem.core.dataset import Dataset
    from rsstatchem.core.stats import factor, process

    rng = np.random.default_rng(11)
    n = 30
    g = np.repeat([1.0, 2.0, 3.0], 10)
    base = rng.normal(0, 2, n)
    t = np.column_stack([base + rng.normal(m, 1, n) for m in (0, 1, 1.5)])
    df = pd.DataFrame({"t1": t[:, 0], "t2": t[:, 1], "t3": t[:, 2], "g": g})
    ds = Dataset(df)

    # Repeated measures: univariate SS must equal the textbook decomposition.
    sec = repeated.repeated_measures(ds, [("time", 3)], ["t1", "t2", "t3"])
    tab = next(i for i in sec.items if getattr(i, "title", "") ==
               "Tests of Within-Subjects Effects")
    grand = t.mean()
    ss_time = n * ((t.mean(axis=0) - grand) ** 2).sum()
    ss_subj = 3 * ((t.mean(axis=1) - grand) ** 2).sum()
    ss_err = ((t - grand) ** 2).sum() - ss_time - ss_subj
    f_ref = (ss_time / 2) / (ss_err / (2 * (n - 1)))
    _check(abs(float(tab.rows[0][5]) - f_ref) < 5e-3 * max(1, f_ref), "RM ANOVA F")
    mixed_sec = repeated.repeated_measures(ds, [("time", 3)], ["t1", "t2", "t3"], ["g"],
                                           emm=["time"], compare=True, plots=[("time", "g")])
    _check(any(i.kind == "image" for i in mixed_sec.items), "profile plot")

    # MANOVA: Wilks' lambda equals det(E)/det(E+H) for the factor.
    sec = manova.manova(ds, ["t1", "t2", "t3"], ["g"])
    Y = t
    E = sum(((Y[g == k] - Y[g == k].mean(0)).T @ (Y[g == k] - Y[g == k].mean(0)))
            for k in (1, 2, 3))
    T = (Y - Y.mean(0)).T @ (Y - Y.mean(0))
    wilks = np.linalg.det(E) / np.linalg.det(T)
    tab = next(i for i in sec.items if getattr(i, "title", "") == "Multivariate Tests")
    row = next(r for r in tab.rows if r[0] == "" and r[1] == "Wilks' Lambda"
               and tab.rows.index(r) > 4)
    _check(abs(float(row[2]) - wilks) < 1e-3, "MANOVA Wilks' lambda")

    # Mixed: random-intercept model on long data runs and yields the four tables.
    long = pd.DataFrame({"id": np.repeat(np.arange(n), 3).astype(float),
                         "time": np.tile([1.0, 2.0, 3.0], n), "y": t.ravel()})
    sec = mixed.linear_mixed(Dataset(long), "y", "id", ["time"])
    titles = [getattr(i, "title", "") for i in sec.items]
    _check(any(s.startswith("Type III Tests") for s in titles), "mixed Type III table")
    tab = next(i for i in sec.items if getattr(i, "title", "").startswith("Type III"))
    _check(abs(float(tab.rows[1][3]) - f_ref) < 5e-3 * max(1, f_ref), "mixed F = RM F")

    # Factor: PCA eigenvalues sum to the number of variables.
    X = rng.normal(size=(80, 5))
    X[:, 1] += X[:, 0]
    X[:, 3] += X[:, 2]
    fds = Dataset(pd.DataFrame(X, columns=[f"v{i}" for i in range(5)]))
    res = factor.factor_analysis(fds, fds.names(), "pc", rotation="varimax", save="regression")
    _check(len(res.saved) >= 1, "factor scores saved")
    s = res.saved["FAC1"][1]
    _check(abs(float(s.std(ddof=1)) - 1) < 1e-6, "PCA scores standardized")

    # PROCESS: indirect effect equals a*b.
    pds = Dataset(process.selftest_data())
    res = process.process(pds, 4, "y", "x", ["m"], boot=200, seed=1)
    a = res.effects["fits"][0].b[1]
    b = res.effects["fits"][1].b[2]
    _check(abs(res.effects["indirect"][0] - a * b) < 1e-10, "PROCESS indirect effect")
    res = process.process(pds, 1, "y", "x", w="w", boot=0)
    _check(any(i.kind == "image" for i in res.section.items), "PROCESS interaction plot")

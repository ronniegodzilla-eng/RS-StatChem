"""Analyze > General Linear Model > Multivariate / Repeated Measures and Analyze > Mixed Models >
Linear dialogs (SPSS layout). This module also exports the combined ``MENUS`` and
``selftest()`` for the dialogs in :mod:`rsstatchem.ui.dialogs.multivariate` (Factor, PROCESS).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMessageBox, QPushButton, QSpinBox, QVBoxLayout,
                               QWidget)

from rsstatchem.core.stats import manova, mixed, repeated
from rsstatchem.ui.dialogs import glm_model_ui as gm
from rsstatchem.ui.dialogs.resample_ui import ResampleMixin
from rsstatchem.ui.dialogs.base import (BASE_TEXT_ROLE, NAME_ROLE, SubDialog,
                                        VariablePickerDialog, is_numeric)

ADJUST_CHOICES = [("lsd", "LSD(none)"), ("bonferroni", "Bonferroni"), ("sidak", "Sidak")]


# ------------------------------------------------------------------ shared sub-dialogs
def emm_dialog(owner: VariablePickerDialog, factor_names: list[str]) -> None:
    gm.emm_dialog(owner, owner.windowTitle(), factor_names, owner.opts.get("terms"))


def posthoc_dialog(owner: VariablePickerDialog, factors: list[str]) -> None:
    gm.posthoc_dialog(owner, owner.windowTitle(), factors)


def glm_options(owner: VariablePickerDialog, title: str) -> None:
    gm.options_dialog(owner, title, multivariate=True)


# ------------------------------------------------------------------ Multivariate
class MultivariateDialog(VariablePickerDialog):
    DEFAULTS = {**gm.GLM_DEFAULTS, "descriptives": True, "effect": True, "homogeneity": True,
                "adjust": "bonferroni"}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Multivariate", parent)
        self.add_target("deps", "Dependent Variables:", accepts=is_numeric, min_items=2)
        self.add_target("factors", "Fixed Factor(s):", required=False)
        self.add_target("covariates", "Covariate(s):", accepts=is_numeric, required=False)
        self.add_target("wls", "WLS Weight:", accepts=is_numeric, single=True, required=False)
        t = "Multivariate"
        self.add_option_button("Model...", lambda: gm.model_dialog(
            self, t, self.selected("factors"), self.selected("covariates")))
        self.add_option_button("Contrasts...", lambda: gm.contrasts_dialog(
            self, t, self.selected("factors")))
        self.add_option_button("Plots...", lambda: gm.plots_dialog(
            self, t, self.selected("factors")))
        self.add_option_button("Post Hoc...", lambda: gm.posthoc_dialog(
            self, t, self.selected("factors")))
        self.add_option_button("EM Means...", lambda: gm.emm_dialog(
            self, t, self.selected("factors"), self.opts.get("terms")))
        self.add_option_button("Save...", lambda: gm.save_dialog(self, t))
        self.add_option_button("Options...", lambda: gm.options_dialog(self, t, True))
        self.finish()

    def validate(self):
        if not self.selected("factors") and not self.selected("covariates") and \
                not self.opts.get("terms"):
            return "Specify at least one fixed factor or covariate."
        return None

    def compute(self):
        factors = self.selected("factors")
        covs = self.selected("covariates")
        opts = gm.glm_options_from(self.opts, factors, covs)
        opts.wls = self.selected_one("wls")
        res = manova.manova(self.ds, self.selected("deps"), factors, covs, options=opts)
        return res if res.saved else res.section


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


class RepeatedMeasuresDialog(VariablePickerDialog):
    """Main Repeated Measures dialog; opens the Define Factor(s) dialog first (like SPSS)."""

    DEFAULTS = {**gm.GLM_DEFAULTS, "descriptives": True, "effect": True, "compare": True,
                "adjust": "bonferroni", "within_contrasts": {}}
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
        t = "Repeated Measures"
        self.add_option_button("Define Factors...", self._redefine)
        self.add_option_button("Model...", lambda: gm.model_dialog(
            self, f"{t} (between-subjects)", self._between_names(),
            [self.ds.var(c).name for c in self.selected("covariates")]))
        self.add_option_button("Contrasts...", self._contrasts)
        self.add_option_button("Plots...", lambda: gm.plots_dialog(
            self, t, self._factor_names()))
        self.add_option_button("Post Hoc...", lambda: gm.posthoc_dialog(
            self, t, self.selected("between")))
        self.add_option_button("EM Means...", lambda: gm.emm_dialog(
            self, t, self._factor_names()))
        self.add_option_button("Save...", lambda: gm.save_dialog(self, t))
        self.add_option_button("Options...", lambda: gm.options_dialog(self, t, True))
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

    def _between_names(self) -> list[str]:
        return [self.ds.var(b).name for b in self.selected("between")]

    def _contrasts(self) -> None:
        names = [n for n, _ in self.within_factors]
        current = {k: (v, "last") for k, v in self.opts.get("within_contrasts", {}).items()}
        dlg = gm.ContrastsDialog("Repeated Measures (within-subjects factors)", names, current,
                                 self)
        if dlg.exec():
            self.opts["within_contrasts"] = {f: v[0] for f, v in dlg.values.items()
                                             if v[0] != "none"}

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
    def validate(self):
        if not self.within_factors:
            return "Define the within-subject factor(s) first."
        if any(v is None for v in self.slot_variables()):
            return "Assign a variable to every level of the within-subjects factor(s)."
        return None

    def compute(self):
        between = self.selected("between")
        covs = self.selected("covariates")
        names = set(self._factor_names())
        opts = gm.glm_options_from(self.opts, self._between_names(),
                                   [self.ds.var(c).name for c in covs])
        opts.emm = [e for e in self.opts.get("emm", []) if e == "(OVERALL)" or
                    all(p in names for p in e.split("*"))]
        opts.plots = [tuple(p) for p in self.opts.get("plots", [])
                      if p[0] in names and all(x is None or x in names for x in p[1:])]
        opts.posthoc_factors = [f for f in self.opts.get("posthoc_factors", []) if f in between]
        wc = {k: v for k, v in self.opts.get("within_contrasts", {}).items()
              if k in {n for n, _ in self.within_factors}}
        res = repeated.repeated_measures(
            self.ds, self.within_factors, self.slot_variables(), between, covs,
            self.measure or "MEASURE_1", options=opts, within_contrasts=wc)
        return res if res.saved else res.section


# ------------------------------------------------------------------ Linear Mixed Models
class LinearMixedDialog(ResampleMixin, VariablePickerDialog):
    DEFAULTS = {"main_only": False, "intercept": True, "slopes": [], "structure": "VC",
                "method": "REML", "descriptives": False, "estimates": True, "covtests": True,
                "level": 95.0, "terms": None, "fixed_intercept": True, "rtype": "ID",
                "emm": [], "compare": False, "adjust": "lsd", "save": [], "icc": False,
                "repeated": None}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Linear Mixed Models", parent)
        self.add_target("subject", "Subjects:", single=True)
        self.add_target("dep", "Dependent Variable:", accepts=is_numeric, single=True)
        self.add_target("factors", "Factor(s):", required=False)
        self.add_target("covariates", "Covariate(s):", accepts=is_numeric, required=False)
        self.add_option_button("Fixed...", self._fixed)
        self.add_option_button("Random...", self._random)
        self.add_option_button("Repeated...", self._repeated)
        self.add_option_button("Estimation...", self._estimation)
        self.add_option_button("Statistics...", self._statistics)
        self.add_option_button("EM Means...", lambda: gm.emm_dialog(
            self, "Linear Mixed Models", self.selected("factors"), self.opts.get("terms")))
        self.add_option_button("Save...", self._save)
        hint = QLabel("Data must be in long format: one row per observation, with a subject "
                      "identifier (and a repeated-measures index for repeated covariance "
                      "structures).")
        hint.setStyleSheet("color:#666")
        hint.setWordWrap(True)
        self.options_layout.addWidget(hint)
        self.finish()

    def _fixed(self) -> None:
        dlg = gm.ModelDialog("Linear Mixed Models", self.selected("factors"),
                             self.selected("covariates"), self.opts.get("terms"),
                             self.opts.get("fixed_intercept", True), 3, self)
        dlg.setWindowTitle("Linear Mixed Models: Fixed Effects")
        dlg.ss.setEnabled(False)
        dlg.ss.setToolTip("MIXED tests fixed effects with Type III sums of squares.")
        if dlg.exec():
            v = dlg.result_values()
            self.opts["terms"] = v["terms"]
            self.opts["fixed_intercept"] = v["intercept"]

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

    def _repeated(self) -> None:
        """SPSS's Specify Subjects and Repeated step: the repeated variable (it may also be a
        factor) and the residual covariance type."""
        dlg = SubDialog("Linear Mixed Models: Repeated", self, columns=1)
        box = dlg.group("Repeated")
        combo = QComboBox()
        combo.addItem("(none)", None)
        for v in self.ds.variables:
            combo.addItem(v.name, v.name)
        cur = self.opts.get("repeated")
        combo.setCurrentIndex(max(combo.findData(cur), 0) if cur else 0)
        row = QFormLayout()
        row.addRow("Repeated variable:", combo)
        box.addLayout(row)
        dlg._getters["repeated"] = combo.currentData
        box = dlg.group("Repeated Covariance Type (residual covariance within a subject)")
        dlg.radio(box, "rtype", [(k, v) for k, v in mixed.REPEATED_TYPES.items()],
                  self.opts["rtype"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def _repeated_name(self) -> str | None:
        name = self.opts.get("repeated")
        return name if name and self.ds.has_name(name) else None

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
        dlg.check(box, "icc", "Intraclass correlation (random-intercept models)",
                  self.opts["icc"])
        dlg.number(box, "level", "Confidence interval (%):", self.opts["level"], 50, 99.99, 2)
        values = dlg.run()
        if values:
            self.opts.update(values)

    def _save(self) -> None:
        dlg = SubDialog("Linear Mixed Models: Save", self, columns=2)
        cur = self.opts.get("save", [])
        box = dlg.group("Fixed Predicted Values")
        dlg.check(box, "FIXPRED", "Predicted values", "FIXPRED" in cur)
        dlg.check(box, "SEFIXP", "Standard errors", "SEFIXP" in cur)
        box = dlg.group("Predicted Values && Residuals".replace("&&", "&"))
        dlg.check(box, "PRED", "Predicted values", "PRED" in cur)
        dlg.check(box, "RESID", "Residuals", "RESID" in cur)
        values = dlg.run()
        if values is not None:
            self.opts["save"] = [k for k in ("FIXPRED", "SEFIXP", "PRED", "RESID")
                                 if values.get(k)]

    def validate(self):
        others = self.selected("factors") + self.selected("covariates")
        if self.selected_one("subject") in others:
            return "The subject variable cannot also be a factor or covariate."
        if self.opts.get("rtype", "ID") != "ID" and not self._repeated_name():
            return "Choose a Repeated variable (Repeated… button) for the covariance type."
        if self.opts.get("terms"):
            try:
                gm.parse_terms(" ".join(self.opts["terms"]), self.selected("factors"),
                               self.selected("covariates"))
            except ValueError as exc:
                return f"The fixed-effects model refers to variables that are not selected " \
                       f"({exc})."
        return None

    def compute(self):
        o = self.opts
        covs = self.selected("covariates")
        factors = self.selected("factors")
        terms = gm.parse_terms(" ".join(o["terms"]), factors, covs) if o.get("terms") else None
        names = set(factors)
        emm = [e for e in o.get("emm", []) if e == "(OVERALL)" or
               all(p in names for p in e.split("*"))]
        rep = self._repeated_name()
        return self.resampled(lambda: self._mixed(covs, factors, terms, emm, rep),
                              pool="models", level=o["level"] / 100)

    def _mixed(self, covs, factors, terms, emm, rep):
        o = self.opts
        return mixed.linear_mixed(self.ds, self.selected_one("dep"),
                                  self.selected_one("subject"), factors, covs,
                                  o["intercept"], [s for s in o["slopes"] if s in covs],
                                  o["structure"], o["method"], o["main_only"],
                                  o["descriptives"], o["estimates"], o["covtests"],
                                  o["level"] / 100, terms, o.get("fixed_intercept", True),
                                  rep, o.get("rtype", "ID") if rep else "ID", emm,
                                  o.get("compare", False), o.get("adjust", "lsd"),
                                  o.get("save", []), o.get("icc", False))


# ------------------------------------------------------------------ Variance Components
class VarianceComponentsDialog(VariablePickerDialog):
    DEFAULTS = {"method": "minque1", "ss_type": 3, "terms": None, "intercept": True}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Variance Components", parent)
        self.add_target("dep", "Dependent Variable:", accepts=is_numeric, single=True)
        self.add_target("factors", "Fixed Factor(s):", required=False)
        self.add_target("random", "Random Factor(s):")
        self.add_target("covariates", "Covariate(s):", accepts=is_numeric, required=False)
        self.add_option_button("Model...", lambda: gm.model_dialog(
            self, "Variance Components", self.selected("factors"),
            self.selected("covariates"), self.selected("random")))
        self.add_option_button("Options...", self._options)
        self.finish()

    def _options(self) -> None:
        dlg = SubDialog("Variance Components: Options", self, columns=2)
        box = dlg.group("Method")
        dlg.radio(box, "method", [("minque1", "MINQUE (random-effect prior: uniform)"),
                                  ("minque0", "MINQUE (random-effect prior: zero)"),
                                  ("anova", "ANOVA"),
                                  ("ml", "Maximum likelihood"),
                                  ("reml", "Restricted maximum likelihood")],
                  self.opts["method"])
        box = dlg.group("Sum of Squares (ANOVA method)")
        dlg.radio(box, "ss_type", [(1, "Type I"), (3, "Type III")], self.opts["ss_type"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def compute(self):
        from rsstatchem.core.stats import varcomp

        o = self.opts
        factors = self.selected("factors")
        random = self.selected("random")
        covs = self.selected("covariates")
        terms = gm.parse_terms(" ".join(o["terms"]), factors + random, covs) \
            if o.get("terms") else None
        return varcomp.variance_components(self.ds, self.selected_one("dep"), factors, random,
                                           covs, o["method"], o["ss_type"], terms,
                                           o.get("intercept", True))


# ------------------------------------------------------------------ menus + selftest
def _menus() -> dict:
    from rsstatchem.ui.dialogs import multivariate as mv

    return {
        "Analyze/General Linear Model": [("Multivariate...", MultivariateDialog),
                                         ("Repeated Measures...", RepeatedMeasuresDialog),
                                         ("Variance Components...", VarianceComponentsDialog)],
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

    _selftest_wp1()


def _selftest_wp1() -> None:
    """GLM custom model (Montgomery RCBD), SPSS post hoc tests, mixed AR(1), VARCOMP."""
    from rsstatchem.core.dataset import Dataset
    from rsstatchem.core.stats import glm_output, posthoc_ext, varcomp
    from rsstatchem.core.stats.common import Group
    from rsstatchem.core.stats.glm_model import TermSpec

    y = [90.3, 89.2, 98.2, 93.9, 87.4, 97.9, 92.5, 89.5, 90.6, 94.7, 87.0, 95.8,
         85.5, 90.8, 89.6, 86.2, 88.0, 93.4, 82.5, 89.5, 85.6, 87.4, 78.9, 90.7]
    df = pd.DataFrame({"y": y, "p": np.repeat([1., 2, 3, 4], 6),
                       "b": np.tile(np.arange(1., 7), 4)})
    opts = glm_output.GLMOptions(terms=[TermSpec(("p",)), TermSpec(("b",))], emm=["p"],
                                 compare=True, posthoc_factors=["p"],
                                 posthoc_methods=["duncan", "dunnett"], save=["PRE"])
    res = glm_output.univariate(Dataset(df), "y", ["p", "b"], [], opts)
    tab = next(i for i in res.section.items
               if getattr(i, "title", "") == "Tests of Between-Subjects Effects")
    row = next(r for r in tab.rows if r[0] == "p")
    _check(abs(float(row[1]) - 178.171) < 1e-3, "RCBD treatment SS (Montgomery 4.1)")
    _check("PRE" in res.saved, "GLM save")
    groups = [Group(k, k, np.array(v)) for k, v in
              {"cc": [28.5, 21.7, 23.0], "fc": [14.9, 10.6, 13.1], "ff": [41.8, 39.2, 28.0],
               "oo": [38.2, 40.4, 32.1]}.items()]
    order, sets, _ = posthoc_ext.homogeneous_subsets(groups, "duncan", 22.489167, 8)
    _check(len(sets) == 3, "Duncan subsets")
    _check(abs(posthoc_ext.dunnett_crit(0.95, np.full(4, np.sqrt(0.5)), 20) - 2.65) < 6e-3,
           "Dunnett critical value")
    rng = np.random.default_rng(3)
    long = pd.DataFrame({"id": np.repeat(np.arange(12.), 3), "t": np.tile([1., 2., 3.], 12),
                         "y": rng.normal(size=36) + np.tile([0., 1., 2.], 12)})
    sec = mixed.linear_mixed(Dataset(long), "y", "id", ["t"], random_intercept=False,
                             repeated="t", repeated_type="AR1")
    _check(any(getattr(i, "title", "").startswith("Estimates of Covariance")
               for i in sec.items), "mixed AR(1)")
    sec = varcomp.variance_components(Dataset(long), "y", [], ["id"], method="reml")
    _check(any(getattr(i, "title", "") == "Variance Estimates" for i in sec.items), "VARCOMP")

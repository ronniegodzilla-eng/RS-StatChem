"""Analyze > Complex Samples dialogs laid out like SPSS: Prepare for Analysis... (Analysis
Preparation Wizard), Select a Sample... (Sampling Wizard), Frequencies..., Descriptives...,
Crosstabs..., Ratios..., General Linear Model..., Logistic Regression..., Ordinal
Regression... and Cox Regression....

SPSS keeps the design in a .csplan file; here the plan is stored with the dataset
(``ds.meta['csplan']``, saved in .rsd projects) and every analysis dialog uses it. Core code:
:mod:`rsstatchem.core.stats.complex_samples`.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QSpinBox)

from rsstatchem.core.dataset import NUMERIC
from rsstatchem.core.stats import complex_samples as cs
from rsstatchem.ui.dialogs.base import SubDialog, VariablePickerDialog, is_numeric
from rsstatchem.ui.dialogs.glm_model_ui import model_dialog

STATS = [("se", "Standard error"), ("ci", "Confidence interval"),
         ("cv", "Coefficient of variation"), ("count", "Unweighted count"),
         ("deff", "Design effect"), ("sqrt_deff", "Square root of design effect")]


def _combo(owner, key, items):
    box = QComboBox()
    for value, text in items:
        box.addItem(text, value)
    box.setCurrentIndex(max(box.findData(owner.opts[key]), 0))
    box.currentIndexChanged.connect(lambda _: owner.opts.__setitem__(key, box.currentData()))
    return box


# ===================================================================== plan wizard
class PreparePlanDialog(VariablePickerDialog):
    """Analysis Preparation Wizard: design variables (stage 1, optional stage 2), sample
    weight, estimation method and finite population correction."""
    modifies_data = True
    DEFAULTS = {"method": "WR", "fpc_kind": "none", "fpc_value": 0.0, "use_stage2": False,
                "method2": "WR", "fpc2_kind": "none", "fpc2_value": 0.0, "srs": "WOR",
                "df": 0}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Analysis Preparation Wizard", parent)
        self.setMinimumSize(820, 820)
        self.add_target("strata", "Stage 1 Strata:", required=False)
        self.add_target("clusters", "Stage 1 Clusters:", required=False)
        self.add_target("weight", "Sample Weight:", accepts=is_numeric, single=True,
                        required=False)
        self.add_target("fpcvar", "Stage 1 FPC variable (rate, population size or "
                        "inclusion probability):", accepts=is_numeric, single=True, required=False)
        self.add_target("strata2", "Stage 2 Strata:", required=False)
        self.add_target("clusters2", "Stage 2 Clusters:", required=False)
        self.add_target("fpcvar2", "Stage 2 FPC variable:",
                        accepts=is_numeric, single=True, required=False)
        box = QGroupBox("Estimation")
        form = QFormLayout(box)
        form.addRow("Stage 1 method:", _combo(self, "method", list(cs.METHODS.items())))
        form.addRow("Stage 1 FPC:", _combo(self, "fpc_kind", [
            ("none", "None"), ("rate", "Sampling rate"), ("popsize", "Population size"),
            ("inclusion", "Inclusion probability (unequal WOR)")]))
        self.fpc_value = QLineEdit(str(self.opts["fpc_value"] or ""))
        self.fpc_value.setPlaceholderText("value (when no variable is given)")
        self.fpc_value.textChanged.connect(lambda t: self._num("fpc_value", t))
        form.addRow("Stage 1 FPC value:", self.fpc_value)
        from PySide6.QtWidgets import QCheckBox

        st2 = QCheckBox("Use stage 2 (only with a without-replacement first stage)")
        st2.setChecked(self.opts["use_stage2"])
        st2.toggled.connect(lambda on: self.opts.__setitem__("use_stage2", on))
        form.addRow(st2)
        form.addRow("Stage 2 method:", _combo(self, "method2", [
            ("WR", "Sampling with replacement"), ("EQUAL_WOR", "Equal WOR")]))
        form.addRow("Stage 2 FPC:", _combo(self, "fpc2_kind", [
            ("none", "None"), ("rate", "Sampling rate"), ("popsize", "Population size")]))
        self.fpc2_value = QLineEdit(str(self.opts["fpc2_value"] or ""))
        self.fpc2_value.textChanged.connect(lambda t: self._num("fpc2_value", t))
        form.addRow("Stage 2 FPC value:", self.fpc2_value)
        form.addRow("SRS estimator for design effects:", _combo(self, "srs", [
            ("WOR", "Without replacement"), ("WR", "With replacement")]))
        self.options_layout.addWidget(box)
        self.finish()

    def _num(self, key, text):
        try:
            self.opts[key] = float(text.replace(",", "."))
        except ValueError:
            self.opts[key] = 0.0

    def plan(self) -> cs.Plan:
        o = self.opts
        s1 = cs.Stage(self.selected("strata"), self.selected("clusters"), o["method"],
                      o["fpc_kind"] if o["method"] != "WR" else "none",
                      o["fpc_value"] or None, self.selected_one("fpcvar"))
        stages = [s1]
        if o["use_stage2"] and o["method"] != "WR":
            stages.append(cs.Stage(self.selected("strata2"), self.selected("clusters2"),
                                   o["method2"], o["fpc2_kind"] if o["method2"] != "WR"
                                   else "none", o["fpc2_value"] or None,
                                   self.selected_one("fpcvar2")))
        return cs.Plan(self.selected_one("weight"), stages, o["srs"], None)

    def validate(self):
        o = self.opts
        if o["method"] != "WR" and o["fpc_kind"] == "none":
            return "A without-replacement design needs a sampling rate, population size or " \
                   "inclusion probability (value or variable)."
        if o["method"] == "UNEQUAL_WOR" and o["fpc_kind"] != "inclusion":
            return "Unequal-probability WOR needs inclusion probabilities."
        if o["fpc_kind"] != "none" and o["method"] != "WR" and not self.selected_one("fpcvar") \
                and not o["fpc_value"]:
            return "Give the stage 1 FPC as a value or a variable."
        return None

    def compute(self):
        return cs.prepare_plan(self.ds, self.plan())


class SelectSampleDialog(VariablePickerDialog):
    """Sampling Wizard: stratified / clustered SRS, systematic or PPS selection."""
    modifies_data = True
    DEFAULTS = {"method": "srswor", "size": 10.0, "size_kind": "count", "seed": 2000000}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Sampling Wizard", parent)
        self.add_target("strata", "Stratify By:", required=False)
        self.add_target("clusters", "Clusters:", required=False)
        self.add_target("mos", "Measure of Size (PPS):", accepts=is_numeric, single=True,
                        required=False)
        box = QGroupBox("Sampling Method and Size")
        form = QFormLayout(box)
        form.addRow("Type:", _combo(self, "method", list(cs.SAMPLING.items())))
        form.addRow("Units:", _combo(self, "size_kind", [("count", "Counts"),
                                                         ("proportion", "Proportions")]))
        size = QLineEdit(f"{self.opts['size']:g}")
        size.textChanged.connect(lambda t: self._set("size", t))
        form.addRow("Value (per stratum):", size)
        seed = QSpinBox()
        seed.setRange(1, 2 ** 31 - 1)
        seed.setValue(int(self.opts["seed"]))
        seed.valueChanged.connect(lambda v: self.opts.__setitem__("seed", int(v)))
        form.addRow("Random seed:", seed)
        self.options_layout.addWidget(box)
        self.finish()

    def _set(self, key, text):
        try:
            self.opts[key] = float(text.replace(",", "."))
        except ValueError:
            pass

    def validate(self):
        if self.opts["method"].startswith("pps") and not self.selected_one("mos"):
            return "PPS methods need a measure of size variable."
        if self.opts["size"] <= 0:
            return "The sample size must be positive."
        return None

    def compute(self):
        o = self.opts
        return cs.select_sample(self.ds, o["method"], self.selected("strata"),
                                self.selected("clusters"), o["size"], o["size_kind"],
                                self.selected_one("mos"), int(o["seed"]))


# ===================================================================== analyses
class _CSDialog(VariablePickerDialog):
    TITLE = "Complex Samples"

    def __init__(self, ds, parent=None):
        super().__init__(ds, self.TITLE, parent)
        self.plan = cs.get_plan(ds)
        note = QLabel(("Plan: " + self.plan.describe().replace("\n", "; ")) if self.plan else
                      "No analysis plan: run Complex Samples > Prepare for Analysis... first.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#555")
        self.options_layout.addWidget(note)

    def validate(self):
        if self.plan is None:
            return "No Complex Samples plan for this dataset. Use Analyze > Complex Samples " \
                   "> Prepare for Analysis... (or Select a Sample...) first."
        return self.extra_validate()

    def extra_validate(self):
        return None

    def _stats(self, title, extra=None):
        dlg = SubDialog(f"{self.TITLE}: {title}", self, columns=2)
        box = dlg.group("Statistics")
        for k, t in STATS:
            dlg.check(box, f"s::{k}", t, k in self.opts["statistics"])
        if extra:
            extra(dlg)
        box = dlg.group("Confidence level")
        dlg.number(box, "level", "Level (%):", self.opts.get("level", 95), 50, 99.99, 2)
        v = dlg.run()
        if v:
            self.opts["statistics"] = [k[3:] for k, on in v.items() if k.startswith("s::") and on]
            self.opts.update({k: x for k, x in v.items() if not k.startswith("s::")})

    def lev(self):
        return float(self.opts.get("level", 95)) / 100


class CSFrequenciesDialog(_CSDialog):
    TITLE = "Complex Samples Frequencies"
    DEFAULTS = {"statistics": ["se"], "popsize": True, "tablepct": True, "cumulative": False,
                "level": 95}

    def __init__(self, ds, parent=None):
        super().__init__(ds, parent)
        self.add_target("vars", "Frequency Tables:")
        self.add_target("subpop", "Subpopulations:", required=False)
        self.add_option_button("Statistics...", lambda: self._stats("Statistics", self._cells))
        self.finish()

    def _cells(self, dlg):
        box = dlg.group("Cells")
        dlg.check(box, "popsize", "Population size", self.opts["popsize"])
        dlg.check(box, "tablepct", "Table percent", self.opts["tablepct"])
        dlg.check(box, "cumulative", "Cumulative values", self.opts["cumulative"])

    def compute(self):
        o = self.opts
        return cs.cs_frequencies(self.ds, self.selected("vars"), self.plan,
                                 self.selected("subpop"), o["popsize"], o["tablepct"],
                                 tuple(o["statistics"]), self.lev(), o["cumulative"])


class CSDescriptivesDialog(_CSDialog):
    TITLE = "Complex Samples Descriptives"
    DEFAULTS = {"statistics": ["se"], "mean": True, "sum": False, "listwise": False,
                "test": False, "test_value": 0.0, "level": 95}

    def __init__(self, ds, parent=None):
        super().__init__(ds, parent)
        self.add_target("vars", "Measures:", accepts=is_numeric)
        self.add_target("subpop", "Subpopulations:", required=False)
        self.add_option_button("Statistics...", lambda: self._stats("Statistics", self._more))
        self.add_option_button("Missing Values...", self._missing)
        self.finish()

    def _more(self, dlg):
        box = dlg.group("Summaries")
        dlg.check(box, "mean", "Mean", self.opts["mean"])
        dlg.check(box, "sum", "Sum", self.opts["sum"])
        box = dlg.group("T-test")
        dlg.check(box, "test", "t-test (mean = value)", self.opts["test"])
        dlg.number(box, "test_value", "Test value:", self.opts["test_value"], -1e12, 1e12, 4)

    def _missing(self):
        dlg = SubDialog(f"{self.TITLE}: Missing Values", self, columns=1)
        box = dlg.group("Statistics for Measure Variables")
        dlg.radio(box, "listwise", [(False, "Use all available data"),
                                    (True, "Ensure consistent case base")],
                  self.opts["listwise"])
        v = dlg.run()
        if v:
            self.opts.update(v)

    def extra_validate(self):
        if not (self.opts["mean"] or self.opts["sum"]):
            return "Request at least the mean or the sum."
        return None

    def compute(self):
        o = self.opts
        return cs.cs_descriptives(self.ds, self.selected("vars"), self.plan, o["mean"],
                                  o["sum"], self.selected("subpop"), tuple(o["statistics"]),
                                  self.lev(), o["listwise"],
                                  o["test_value"] if o["test"] else None)


class CSCrosstabsDialog(_CSDialog):
    TITLE = "Complex Samples Crosstabs"
    DEFAULTS = {"statistics": ["se"], "cells": ["popsize", "row"], "tests": True,
                "odds": False, "residuals": False, "level": 95}

    def __init__(self, ds, parent=None):
        super().__init__(ds, parent)
        self.add_target("row", "Rows:", single=True)
        self.add_target("col", "Columns:", single=True)
        self.add_target("subpop", "Subpopulations:", required=False)
        self.add_option_button("Statistics...", lambda: self._stats("Statistics", self._more))
        self.finish()

    def _more(self, dlg):
        box = dlg.group("Cells")
        for k, t in (("popsize", "Population size"), ("row", "Row percent"),
                     ("column", "Column percent"), ("table", "Table percent")):
            dlg.check(box, f"c::{k}", t, k in self.opts["cells"])
        box = dlg.group("Tests and Summaries")
        dlg.check(box, "tests", "Tests of independence", self.opts["tests"])
        dlg.check(box, "odds", "Odds ratio / relative risk (2 x 2)", self.opts["odds"])
        dlg.check(box, "residuals", "Standardized residuals", self.opts["residuals"])

    def _stats(self, title, extra=None):
        super()._stats(title, extra)
        cells = [k[3:] for k, v in list(self.opts.items()) if k.startswith("c::") and v]
        if any(k.startswith("c::") for k in self.opts):
            self.opts["cells"] = cells or ["popsize"]
            for k in [k for k in self.opts if k.startswith("c::")]:
                del self.opts[k]

    def extra_validate(self):
        if self.selected_one("row") == self.selected_one("col"):
            return "Row and column variables must differ."
        return None

    def compute(self):
        o = self.opts
        return cs.cs_crosstabs(self.ds, self.selected_one("row"), self.selected_one("col"),
                               self.plan, self.selected("subpop"), tuple(o["cells"]),
                               tuple(o["statistics"]), self.lev(), o["tests"], o["odds"],
                               o["residuals"])


class CSRatiosDialog(_CSDialog):
    TITLE = "Complex Samples Ratios"
    DEFAULTS = {"statistics": ["se"], "popsize": False, "test": False, "test_value": 1.0,
                "level": 95}

    def __init__(self, ds, parent=None):
        super().__init__(ds, parent)
        self.add_target("num", "Numerators:", accepts=is_numeric)
        self.add_target("den", "Denominator:", accepts=is_numeric)
        self.add_target("subpop", "Subpopulations:", required=False)
        self.add_option_button("Statistics...", lambda: self._stats("Statistics", self._more))
        self.finish()

    def _more(self, dlg):
        box = dlg.group("Other")
        dlg.check(box, "popsize", "Population size", self.opts["popsize"])
        dlg.check(box, "test", "t-test (ratio = value)", self.opts["test"])
        dlg.number(box, "test_value", "Test value:", self.opts["test_value"], -1e12, 1e12, 4)

    def compute(self):
        o = self.opts
        return cs.cs_ratios(self.ds, self.selected("num"), self.selected("den"), self.plan,
                            self.selected("subpop"), tuple(o["statistics"]), self.lev(),
                            o["popsize"], o["test_value"] if o["test"] else None)


class _CSModelDialog(_CSDialog):
    COMMON = {"terms": None, "intercept": True, "test": "F", "deff": True, "level": 95,
              "domain_value": ""}

    def _add_common(self):
        self.add_target("domain", "Subpopulation:", single=True, required=False)
        self.add_option_button("Model...", lambda: model_dialog(
            self, self.TITLE, self.selected("factors"), self.selected("covariates")))
        self.add_option_button("Statistics...", self._statistics)
        self.add_option_button("Hypothesis Tests...", self._tests)
        self.add_option_button("Subpopulation value...", self._domain)

    def _statistics(self):
        dlg = SubDialog(f"{self.TITLE}: Statistics", self, columns=1)
        box = dlg.group("Model Parameters")
        dlg.check(box, "deff", "Design effect", self.opts["deff"])
        dlg.number(box, "level", "Confidence level (%):", self.opts["level"], 50, 99.99, 2)
        v = dlg.run()
        if v:
            self.opts.update(v)

    def _tests(self):
        dlg = SubDialog(f"{self.TITLE}: Hypothesis Tests", self, columns=1)
        box = dlg.group("Test Statistic")
        dlg.radio(box, "test", list(cs.TESTS.items()), self.opts["test"])
        v = dlg.run()
        if v:
            self.opts.update(v)

    def _domain(self):
        from PySide6.QtWidgets import QInputDialog

        text, ok = QInputDialog.getText(self, self.TITLE, "Subpopulation category value:",
                                        text=str(self.opts["domain_value"]))
        if ok:
            self.opts["domain_value"] = text

    def terms_text(self):
        t = self.opts.get("terms")
        if t is None:
            from rsstatchem.core.stats.glm_model import full_factorial

            return " ".join(x.syntax for x in full_factorial(self.selected("factors"),
                                                              self.selected("covariates")))
        return " ".join(t) if t else None

    def domain(self):
        name = self.selected_one("domain")
        if not name:
            return None
        raw = self.opts["domain_value"]
        if self.ds.var(name).type == NUMERIC:
            try:
                raw = float(str(raw).replace(",", "."))
            except ValueError:
                pass
        return (name, raw)


class CSGLMDialog(_CSModelDialog):
    TITLE = "Complex Samples General Linear Model"
    DEFAULTS = dict(_CSModelDialog.COMMON)

    def __init__(self, ds, parent=None):
        super().__init__(ds, parent)
        self.add_target("dep", "Dependent Variable:", accepts=is_numeric, single=True)
        self.add_target("factors", "Factor(s):", required=False)
        self.add_target("covariates", "Covariate(s):", accepts=is_numeric, required=False)
        self._add_common()
        self.finish()

    def compute(self):
        o = self.opts
        return cs.cs_glm(self.ds, self.selected_one("dep"), self.selected("factors"),
                         self.selected("covariates"), self.plan, self.terms_text(),
                         o["intercept"], self.lev(), o["test"], o["deff"], self.domain())


class CSLogisticDialog(_CSModelDialog):
    TITLE = "Complex Samples Logistic Regression"
    DEFAULTS = dict(_CSModelDialog.COMMON, reference="last", odds=True, classification=True)

    def __init__(self, ds, parent=None):
        super().__init__(ds, parent)
        self.add_target("dep", "Dependent Variable:", single=True)
        self.add_option_button("Reference Category...", self._reference)
        self.add_target("factors", "Factor(s):", required=False)
        self.add_target("covariates", "Covariate(s):", accepts=is_numeric, required=False)
        self._add_common()
        self.finish()

    def _reference(self):
        dlg = SubDialog(f"{self.TITLE}: Reference Category", self, columns=1)
        box = dlg.group("Reference Category")
        dlg.radio(box, "reference", [("last", "Highest value"), ("first", "Lowest value")],
                  self.opts["reference"])
        box = dlg.group("Output")
        dlg.check(box, "odds", "Odds ratios", self.opts["odds"])
        dlg.check(box, "classification", "Classification table", self.opts["classification"])
        v = dlg.run()
        if v:
            self.opts.update(v)

    def compute(self):
        o = self.opts
        return cs.cs_logistic(self.ds, self.selected_one("dep"), self.selected("factors"),
                              self.selected("covariates"), self.plan, o["reference"],
                              self.terms_text(), o["intercept"], self.lev(), o["test"],
                              o["deff"], o["odds"], o["classification"], self.domain())


class CSOrdinalDialog(_CSModelDialog):
    TITLE = "Complex Samples Ordinal Regression"
    DEFAULTS = dict(_CSModelDialog.COMMON, link="logit")

    def __init__(self, ds, parent=None):
        super().__init__(ds, parent)
        self.add_target("dep", "Dependent Variable:", single=True)
        self.add_target("factors", "Factor(s):", required=False)
        self.add_target("covariates", "Covariate(s):", accepts=is_numeric, required=False)
        self._add_common()
        row = QHBoxLayout()
        row.addWidget(QLabel("Link function:"))
        row.addWidget(_combo(self, "link", list(cs.LINKS.items())))
        row.addStretch()
        self.options_layout.addLayout(row)
        self.finish()

    def compute(self):
        o = self.opts
        return cs.cs_ordinal(self.ds, self.selected_one("dep"), self.selected("factors"),
                             self.selected("covariates"), self.plan, o["link"],
                             self.terms_text(), self.lev(), o["test"], o["deff"],
                             self.domain())


class CSCoxDialog(_CSModelDialog):
    TITLE = "Complex Samples Cox Regression"
    DEFAULTS = dict(_CSModelDialog.COMMON, event="1", ties="efron")

    def __init__(self, ds, parent=None):
        super().__init__(ds, parent)
        self.add_target("time", "Time:", accepts=is_numeric, single=True)
        self.add_target("status", "Status:", single=True)
        self.add_option_button("Define Event...", self._event)
        self.add_target("factors", "Factor(s):", required=False)
        self.add_target("covariates", "Covariate(s):", accepts=is_numeric, required=False)
        self._add_common()
        row = QHBoxLayout()
        row.addWidget(QLabel("Ties:"))
        row.addWidget(_combo(self, "ties", [("efron", "Efron"), ("breslow", "Breslow")]))
        row.addStretch()
        self.options_layout.addLayout(row)
        self.finish()

    def _event(self):
        from PySide6.QtWidgets import QInputDialog

        text, ok = QInputDialog.getText(self, self.TITLE, "Event value(s), separated by "
                                        "spaces:", text=str(self.opts["event"]))
        if ok:
            self.opts["event"] = text

    def events(self):
        name = self.selected_one("status")
        parts = str(self.opts["event"]).replace(",", " ").split()
        if name and self.ds.var(name).type == NUMERIC:
            out = []
            for p in parts:
                try:
                    out.append(float(p))
                except ValueError:
                    pass
            return out
        return parts

    def extra_validate(self):
        if not self.events():
            return "Define the event value(s) of the status variable."
        return None

    def compute(self):
        o = self.opts
        return cs.cs_cox(self.ds, self.selected_one("time"), self.selected_one("status"),
                         self.events(), self.selected("factors"), self.selected("covariates"),
                         self.plan, self.terms_text(), o["ties"], self.lev(), o["test"],
                         o["deff"], True, self.domain())


MENUS = {"Analyze/Complex Samples": [
    ("Prepare for Analysis...", PreparePlanDialog),
    ("Select a Sample...", SelectSampleDialog), None,
    ("Frequencies...", CSFrequenciesDialog),
    ("Descriptives...", CSDescriptivesDialog),
    ("Crosstabs...", CSCrosstabsDialog),
    ("Ratios...", CSRatiosDialog),
    ("General Linear Model...", CSGLMDialog),
    ("Logistic Regression...", CSLogisticDialog),
    ("Ordinal Regression...", CSOrdinalDialog),
    ("Cox Regression...", CSCoxDialog),
]}


def selftest() -> None:
    cs.selftest()

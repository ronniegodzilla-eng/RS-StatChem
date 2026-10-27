"""Analyze > Survival dialogs (Life Tables, Kaplan-Meier, Cox Regression, Cox w/ Time-Dep
Cov), laid out like SPSS. Core code: :mod:`rsstatchem.core.stats.survival`."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QRadioButton, QVBoxLayout)

from rsstatchem.core.dataset import NUMERIC, format_code
from rsstatchem.core.stats import boot_procs
from rsstatchem.core.stats import survival as sv
from rsstatchem.core.stats.wp4_util import CONTRASTS
from rsstatchem.ui.dialogs.resample_ui import ResampleMixin
from rsstatchem.ui.dialogs.base import (BASE_TEXT_ROLE, NAME_ROLE, DefineRangeDialog, SubDialog,
                                        VariablePickerDialog, is_numeric)


class DefineEventDialog(QDialog):
    """'Define Event for Status Variable': single value, range or list of values."""

    def __init__(self, var, current, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Define Event for Status Variable")
        current = current or ("single", 1.0)
        self.single = QRadioButton("Single value:")
        self.range_ = QRadioButton("Range of values:")
        self.list_ = QRadioButton("List of values:")
        group = QButtonGroup(self)
        for b in (self.single, self.range_, self.list_):
            group.addButton(b)
        self.v1 = QLineEdit(format_code(current[1]) if current[0] == "single" else "1")
        self.lo = QLineEdit(format_code(current[1]) if current[0] == "range" else "")
        self.hi = QLineEdit(format_code(current[2]) if current[0] == "range" else "")
        self.lst = QLineEdit(" ".join(format_code(v) for v in current[1])
                             if current[0] == "list" else "")
        {"single": self.single, "range": self.range_, "list": self.list_}[current[0]] \
            .setChecked(True)
        form = QFormLayout()
        form.addRow(QLabel("Value(s) Indicating Event Has Occurred"))
        form.addRow(self.single, self.v1)
        row = QHBoxLayout()
        row.addWidget(self.lo)
        row.addWidget(QLabel("through"))
        row.addWidget(self.hi)
        form.addRow(self.range_, row)
        form.addRow(self.list_, self.lst)
        if var.value_labels:
            hint = QLabel(", ".join(f"{format_code(k)} = {v}" for k, v in var.value_labels.items()))
            hint.setWordWrap(True)
            hint.setStyleSheet("color:#666")
            form.addRow(hint)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Continue")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(buttons)
        self.numeric = var.type == NUMERIC

    def _val(self, text):
        text = text.strip()
        return float(text.replace(",", ".")) if self.numeric else text

    def result(self):
        try:
            if self.single.isChecked():
                return ("single", self._val(self.v1.text()))
            if self.range_.isChecked():
                return ("range", self._val(self.lo.text()), self._val(self.hi.text()))
            vals = [self._val(t) for t in self.lst.text().replace(",", " ").split() if t]
            return ("list", vals) if vals else None
        except ValueError:
            return None


class _StatusMixin:
    """A Status target with a 'Define Event...' button (the event spec lives in opts)."""

    def setup_status(self: VariablePickerDialog) -> None:
        self.add_target("status", "Status:", single=True)
        self.event_button = self.add_option_button("Define Event...", self._define_event)

    def _event(self):
        e = self.opts.get("event") or {}
        return tuple(e["spec"]) if e.get("var") == self.selected_one("status") else None

    def _define_event(self) -> None:
        name = self.selected_one("status")
        if not name:
            return
        dlg = DefineEventDialog(self.ds.var(name), self._event(), self)
        if dlg.exec() and dlg.result() is not None:
            self.opts["event"] = {"var": name, "spec": list(dlg.result())}
        self._status_suffix()

    def _status_suffix(self) -> None:
        name = self.selected_one("status")
        if name:
            spec = self._event()
            from rsstatchem.core.stats.wp4_util import describe_event

            self.set_item_suffix("status", describe_event(spec) if spec else "(?)")

    def check_event(self) -> str | None:
        name = self.selected_one("status")
        if name and not self._event():
            s = self.ds.analysis_series(name).dropna()
            if self.ds.var(name).type == NUMERIC and s.isin([0, 1]).all() and (s == 1).any():
                self.opts["event"] = {"var": name, "spec": ["single", 1.0]}
                return None
            return "Click Define Event... to define the value that indicates the event."
        return None


# =========================================================================== Life Tables
class LifeTablesDialog(_StatusMixin, VariablePickerDialog):
    DEFAULTS = {"end": 0.0, "width": 0.0, "tables": True, "plots": [], "compare": "none",
                "range": None}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Life Tables", parent)
        self.add_target("time", "Time:", accepts=is_numeric, single=True)
        self.setup_status()
        self.add_target("factor", "First-order Factor:", accepts=is_numeric, single=True,
                        required=False)
        self.range_button = self.add_option_button("Define Range...", self._range)
        self.add_target("by", "By Factor:", single=True, required=False)
        self.add_option_button("Options...", self._options)
        form = QFormLayout()
        row = QHBoxLayout()
        self.end_spin = QDoubleSpinBox()
        self.end_spin.setRange(0, 1e9)
        self.end_spin.setDecimals(2)
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0, 1e9)
        self.width_spin.setDecimals(2)
        row.addWidget(QLabel("0 through"))
        row.addWidget(self.end_spin)
        row.addWidget(QLabel("by"))
        row.addWidget(self.width_spin)
        form.addRow("Display Time Intervals:", row)
        self.options_layout.addLayout(form)
        self.finish()
        if self.opts["end"]:
            self.end_spin.setValue(self.opts["end"])
            self.width_spin.setValue(self.opts["width"])

    def targets_changed(self) -> None:
        if not hasattr(self, "range_button"):
            return
        self._status_suffix()
        f = self.selected_one("factor")
        self.range_button.setEnabled(f is not None)
        if f:
            r = self.opts.get("range")
            self.set_item_suffix("factor", f"({r[0]:g} {r[1]:g})" if r else "(? ?)")
        t = self.selected_one("time")
        if t and not self.end_spin.value():
            x = pd.to_numeric(self.ds.analysis_series(t), errors="coerce").dropna()
            if len(x):
                width = _nice_width(float(x.max()))
                self.width_spin.setValue(width)
                self.end_spin.setValue(width * np.floor(x.max() / width))

    def _range(self):
        values = DefineRangeDialog(self.opts.get("range"), self).run()
        if values:
            self.opts["range"] = (values["lo"], values["hi"])
        self.targets_changed()

    def _options(self):
        dlg = SubDialog("Life Tables: Options", self)
        box = dlg.group("")
        dlg.check(box, "tables", "Life table(s)", self.opts["tables"])
        box = dlg.group("Plot")
        for key, text in (("survival", "Survival"), ("log_survival", "Log survival"),
                          ("hazard", "Hazard"), ("density", "Density"),
                          ("one_minus", "One minus survival")):
            dlg.check(box, f"plot::{key}", text, key in self.opts["plots"])
        box = dlg.group("Compare Levels of First Factor")
        dlg.radio(box, "compare", [("none", "None"), ("overall", "Overall"),
                                   ("pairwise", "Pairwise")], self.opts["compare"])
        values = dlg.run()
        if values:
            self.opts["tables"] = values["tables"]
            self.opts["compare"] = values["compare"]
            self.opts["plots"] = [k[6:] for k, v in values.items() if k.startswith("plot::") and v]

    def validate(self):
        err = self.check_event()
        if err:
            return err
        if self.width_spin.value() <= 0 or self.end_spin.value() <= 0:
            return "Specify the time intervals (0 through ... by ...)."
        if self.selected_one("factor") and not self.opts.get("range"):
            x = pd.to_numeric(self.ds.analysis_series(self.selected_one("factor")),
                              errors="coerce").dropna()
            self.opts["range"] = (float(x.min()), float(x.max())) if len(x) else (0, 0)
        self.opts["end"] = self.end_spin.value()
        self.opts["width"] = self.width_spin.value()
        return None

    def compute(self):
        o = self.opts
        return sv.life_tables(self.ds, self.selected_one("time"), self.selected_one("status"),
                              tuple(o["event"]["spec"]), o["width"], o["end"],
                              self.selected_one("factor"),
                              tuple(o["range"]) if self.selected_one("factor") else None,
                              self.selected_one("by"), o["tables"], tuple(o["plots"]),
                              o["compare"] != "none", o["compare"] == "pairwise")


def _nice_width(tmax: float) -> float:
    raw = tmax / 10 if tmax > 0 else 1
    mag = 10 ** np.floor(np.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return float(m * mag)
    return float(raw)


# =========================================================================== Kaplan-Meier
class KaplanMeierDialog(_StatusMixin, VariablePickerDialog):
    DEFAULTS = {"tests": [], "compare": "pooled", "trend": False, "table": True,
                "mean_median": True, "quartiles": False, "plots": [], "save": []}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Kaplan-Meier", parent)
        self.add_target("time", "Time:", accepts=is_numeric, single=True)
        self.setup_status()
        self.add_target("factor", "Factor:", single=True, required=False)
        self.add_target("strata", "Strata:", single=True, required=False)
        self.add_target("label", "Label Cases by:", single=True, required=False)
        self.add_option_button("Compare Factor...", self._compare)
        self.add_option_button("Save...", self._save)
        self.add_option_button("Options...", self._options)
        self.finish()

    def targets_changed(self):
        if hasattr(self, "event_button"):
            self._status_suffix()

    def _compare(self):
        dlg = SubDialog("Kaplan-Meier: Compare Factor Levels", self)
        box = dlg.group("Test Statistics")
        for key, text in sv.TESTS.items():
            dlg.check(box, f"t::{key}", text, key in self.opts["tests"])
        dlg.check(box, "trend", "Linear trend for factor levels", self.opts["trend"])
        box = dlg.group("")
        dlg.radio(box, "compare", [("pooled", "Pooled over strata"),
                                   ("stratum", "For each stratum"),
                                   ("pairwise", "Pairwise over strata"),
                                   ("pairwise_stratum", "Pairwise for each stratum")],
                  self.opts["compare"])
        values = dlg.run()
        if values:
            self.opts["tests"] = [k for k in sv.TESTS if values.get(f"t::{k}")]
            self.opts["compare"] = values["compare"]
            self.opts["trend"] = values["trend"]

    def _save(self):
        dlg = SubDialog("Kaplan-Meier: Save New Variables", self, columns=1)
        box = dlg.group("Save")
        for key, text in (("survival", "Survival"), ("se", "Standard error of survival"),
                          ("hazard", "Hazard"), ("cum_events", "Cumulative events")):
            dlg.check(box, key, text, key in self.opts["save"])
        values = dlg.run()
        if values is not None:
            self.opts["save"] = [k for k, v in values.items() if v]

    def _options(self):
        dlg = SubDialog("Kaplan-Meier: Options", self)
        box = dlg.group("Statistics")
        dlg.check(box, "table", "Survival table(s)", self.opts["table"])
        dlg.check(box, "mean_median", "Mean and median survival", self.opts["mean_median"])
        dlg.check(box, "quartiles", "Quartiles", self.opts["quartiles"])
        box = dlg.group("Plots")
        for key, text in sv.KM_PLOTS.items():
            dlg.check(box, f"p::{key}", text, key in self.opts["plots"])
        values = dlg.run()
        if values:
            for k in ("table", "mean_median", "quartiles"):
                self.opts[k] = values[k]
            self.opts["plots"] = [k for k in sv.KM_PLOTS if values.get(f"p::{k}")]

    def validate(self):
        err = self.check_event()
        if err:
            return err
        if self.opts["tests"] and not self.selected_one("factor"):
            return "Select a factor to compare survival distributions."
        return None

    def compute(self):
        o = self.opts
        return sv.kaplan_meier(self.ds, self.selected_one("time"), self.selected_one("status"),
                               tuple(o["event"]["spec"]), self.selected_one("factor"),
                               self.selected_one("strata"), self.selected_one("label"),
                               tuple(o["tests"]), o["compare"], o["trend"], o["table"],
                               o["mean_median"], o["quartiles"], tuple(o["plots"]),
                               tuple(o["save"]))


# =========================================================================== Cox
class CoxRegressionDialog(ResampleMixin, _StatusMixin, VariablePickerDialog):
    TITLE = "Cox Regression"
    TIME_DEPENDENT = False
    DEFAULTS = {"method": "enter", "categorical": {}, "contrast": "indicator",
                "reference": "last", "ci": True, "level": 95.0, "correlations": False,
                "baseline": False, "each_step": True, "p_enter": 0.05, "p_remove": 0.10,
                "max_iter": 20, "plots": [], "separate": "", "save": [], "tcov": "",
                "use_tcov": True}

    def __init__(self, ds, parent=None):
        super().__init__(ds, self.TITLE, parent)
        self.add_target("time", "Time:", accepts=is_numeric, single=True)
        self.setup_status()
        self.add_target("cov", "Covariates:", required=False)
        self.add_target("strata", "Strata:", single=True, required=False)
        self.add_option_button("Categorical...", self._categorical)
        self.add_option_button("Plots...", self._plots)
        self.add_option_button("Save...", self._save)
        self.add_option_button("Options...", self._options)
        if not self.TIME_DEPENDENT:
            self.add_bootstrap_button()
        form = QFormLayout()
        self.method_box = QComboBox()
        for key, text in sv.COX_METHODS.items():
            self.method_box.addItem(text, key)
        self.method_box.setCurrentIndex(max(self.method_box.findData(self.opts["method"]), 0))
        self.method_box.currentIndexChanged.connect(
            lambda _: self.opts.__setitem__("method", self.method_box.currentData()))
        form.addRow("Method:", self.method_box)
        if self.TIME_DEPENDENT:
            self.tcov_edit = QLineEdit(self.opts["tcov"])
            self.tcov_edit.setPlaceholderText("e.g.  T_ * age   or   (T_ > 100) * treatment")
            form.addRow("Expression for T_COV_:", self.tcov_edit)
            hint = QLabel("T_ is the time variable. T_COV_ is added to the covariates "
                          "(SPSS: Compute Time-Dependent Covariate).")
            hint.setWordWrap(True)
            hint.setStyleSheet("color:#666")
            form.addRow(hint)
        self.options_layout.addLayout(form)
        self.finish()

    def targets_changed(self):
        if not hasattr(self, "event_button"):
            return
        self._status_suffix()
        if len(self._targets) < 3:
            return
        covs = self.selected("cov")
        cats = self.opts["categorical"]
        for name in covs:
            if self.ds.var(name).type != NUMERIC:
                cats.setdefault(name, [self.opts["contrast"], self.opts["reference"]])
        widget = self._targets[2].widget
        for i in range(widget.count()):
            item = widget.item(i)
            name = item.data(NAME_ROLE)
            if name in cats:
                c, r = cats[name]
                item.setText(f"{name}(Cat: {c[:3].capitalize()}"
                             f"{'(first)' if r == 'first' else ''})")
            else:
                item.setText(item.data(BASE_TEXT_ROLE) or name)

    def _categorical(self):
        covs = self.selected("cov")
        if not covs:
            return
        dlg = SubDialog("Cox Regression: Define Categorical Covariates", self)
        box = dlg.group("Categorical Covariates")
        for name in covs:
            dlg.check(box, f"cat::{name}", self.ds.var(name).title,
                      name in self.opts["categorical"] or self.ds.var(name).type != NUMERIC)
        box = dlg.group("Change Contrast")
        dlg.radio(box, "contrast", list(CONTRASTS.items()), self.opts["contrast"])
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

    def _plots(self):
        dlg = SubDialog("Cox Regression: Plots", self)
        box = dlg.group("Plot Type")
        for key, text in (("survival", "Survival"), ("hazard", "Hazard"),
                          ("lml", "Log minus log"), ("one_minus", "One minus survival")):
            dlg.check(box, f"p::{key}", text, key in self.opts["plots"])
        box = dlg.group("Separate Lines for")
        cats = [n for n in self.selected("cov") if n in self.opts["categorical"]]
        dlg.radio(box, "separate", [("", "(none)")] + [(n, n) for n in cats],
                  self.opts["separate"] if self.opts["separate"] in cats else "")
        values = dlg.run()
        if values:
            self.opts["plots"] = [k[3:] for k, v in values.items() if k.startswith("p::") and v]
            self.opts["separate"] = values["separate"]

    def _save(self):
        dlg = SubDialog("Cox Regression: Save New Variables", self)
        box = dlg.group("Survival")
        for key, text in (("survival", "Function"), ("se", "Standard error"),
                          ("lml", "Log minus log")):
            dlg.check(box, key, text, key in self.opts["save"])
        box = dlg.group("Diagnostics")
        for key, text in (("hazard", "Hazard function"), ("partial", "Partial residuals"),
                          ("dfbeta", "DfBeta(s)"), ("xbeta", "X*Beta")):
            dlg.check(box, key, text, key in self.opts["save"])
        values = dlg.run()
        if values is not None:
            self.opts["save"] = [k for k, v in values.items() if v]

    def _options(self):
        dlg = SubDialog("Cox Regression: Options", self)
        box = dlg.group("Model Statistics")
        dlg.check(box, "ci", "CI for exp(B):", self.opts["ci"])
        dlg.number(box, "level", "Level (%):", self.opts["level"], 50, 99.99, 2)
        dlg.check(box, "correlations", "Correlation of estimates", self.opts["correlations"])
        dlg.radio(box, "each_step", [(True, "At each step"), (False, "At last step")],
                  self.opts["each_step"])
        box = dlg.group("Probability for Stepwise")
        dlg.number(box, "p_enter", "Entry:", self.opts["p_enter"], 0.0001, 0.99, 4)
        dlg.number(box, "p_remove", "Removal:", self.opts["p_remove"], 0.0001, 0.99, 4)
        dlg.number(box, "max_iter", "Maximum Iterations:", self.opts["max_iter"], 1, 999, 0)
        box = dlg.group("")
        dlg.check(box, "baseline", "Display baseline function", self.opts["baseline"])
        values = dlg.run()
        if values:
            values["max_iter"] = int(values["max_iter"])
            if values["p_enter"] >= values["p_remove"]:
                values["p_remove"] = min(0.99, values["p_enter"] + 0.05)
            self.opts.update(values)

    def validate(self):
        err = self.check_event()
        if err:
            return err
        if self.TIME_DEPENDENT:
            expr = self.tcov_edit.text().strip()
            self.opts["tcov"] = expr
            if not expr:
                return "Enter an expression for T_COV_ (use T_ for time)."
        elif not self.selected("cov"):
            return "Select at least one covariate."
        if self.selected_one("time") in self.selected("cov"):
            return "The time variable cannot also be a covariate."
        return None

    def compute(self):
        o = self.opts
        covs = self.selected("cov")
        cats = {n: tuple(v) for n, v in o["categorical"].items() if n in covs}
        tcov = None
        if self.TIME_DEPENDENT:
            tcov = o["tcov"]
            covs = covs + ["T_COV_"]
        return self.resampled(
            lambda: self._cox(covs, cats, tcov),
            boot=lambda r, spec: boot_procs.models(r.section, self.ds, r.models, spec),
            pool="models", level=self.level())

    def _cox(self, covs, cats, tcov):
        o = self.opts
        return sv.cox_regression(self.ds, self.selected_one("time"), self.selected_one("status"),
                                 tuple(o["event"]["spec"]), covs, cats, o["method"],
                                 self.selected_one("strata"), o["p_enter"], o["p_remove"],
                                 int(o["max_iter"]), o["ci"], self.level(), o["correlations"],
                                 o["baseline"], o["each_step"], tuple(o["plots"]),
                                 o["separate"] or None, tuple(o["save"]), tcov)


class CoxTimeDependentDialog(CoxRegressionDialog):
    TITLE = "Cox Regression with Time-Dependent Covariates"
    TIME_DEPENDENT = True


MENUS = {
    "Analyze/Survival": [("Life Tables...", LifeTablesDialog),
                         ("Kaplan-Meier...", KaplanMeierDialog),
                         ("Cox Regression...", CoxRegressionDialog),
                         ("Cox w/ Time-Dep Cov...", CoxTimeDependentDialog)],
}


def selftest() -> None:
    """Headless check of the survival core on synthetic data."""
    from rsstatchem.core.dataset import Dataset

    rng = np.random.default_rng(3)
    n = 80
    g = rng.integers(0, 2, n).astype(float)
    x = rng.normal(size=n)
    t = rng.exponential(1 / np.exp(0.7 * g + 0.4 * x)) * 10
    c = rng.exponential(15, n)
    ds = Dataset(pd.DataFrame({"t": np.minimum(t, c), "d": (t <= c).astype(float), "g": g,
                               "x": x}))
    r = sv.kaplan_meier(ds, "t", "d", ("single", 1.0), "g", tests=("logrank", "breslow"),
                        plots=("survival",), save=("survival",))
    assert r.saved and any(getattr(i, "title", "") == "Overall Comparisons"
                           for i in r.section.items)
    sv.life_tables(ds, "t", "d", ("single", 1.0), 2.0, 20.0, "g", (0, 1), compare=True,
                   plots=("survival",))
    res = sv.cox_regression(ds, "t", "d", ("single", 1.0), ["g", "x"], {"g": ("indicator",
                                                                               "first")},
                            method="fstep_lr", baseline=True, plots=("survival",),
                            save=("survival", "xbeta"))
    assert res.saved
    sv.cox_regression(ds, "t", "d", ("single", 1.0), ["x", "T_COV_"],
                      time_dependent="x * LN(T_ + 1)")

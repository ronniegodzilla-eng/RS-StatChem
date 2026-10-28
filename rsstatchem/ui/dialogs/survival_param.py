"""Analyze > Survival > Parametric Accelerated Failure Time (AFT) Models and Parametric Shared
Frailty Models, laid out like the SPSS 29+ dialogs (Time: Survival or Start/End, Status with
Define Event, Covariate(s), Fixed Factor(s) with Select Category, Left Truncation / Subject /
Interval; Criteria, Model, Estimate, Print, Predict and Plot panels). Core code:
:mod:`rsstatchem.core.stats.survival_param`."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QDialog, QDialogButtonBox,
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit, QRadioButton,
                               QVBoxLayout)

from rsstatchem.core.dataset import NUMERIC, format_code
from rsstatchem.core.stats import survival_param as sp
from rsstatchem.ui.dialogs.base import SubDialog, VariablePickerDialog, is_numeric


class StatusMapDialog(QDialog):
    """'Define Event': the values mapped to each status category and the treatment of
    unmapped values."""

    def __init__(self, var, current: dict | None, unmapped: str, categories: list[str],
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Define Events for Status Variable")
        self.numeric = var.type == NUMERIC
        cur = current or ({"event": [1.0], "right": [0.0]} if self.numeric
                          else {k: v for k, v in sp.STRING_DEFAULTS.items() if k in categories})
        form = QFormLayout()
        form.addRow(QLabel("Values (separate several values by spaces):"))
        self.edits = {}
        for key in categories:
            ed = QLineEdit(" ".join(format_code(v) if self.numeric else str(v)
                                    for v in cur.get(key, [])))
            self.edits[key] = ed
            form.addRow(sp.STATUS[key] + ":", ed)
        self.unmapped = QComboBox()
        self.unmapped.addItem("Exclude them from analysis", "exclude")
        for key in categories:
            self.unmapped.addItem(f"Map to {sp.STATUS[key]}", key)
        self.unmapped.setCurrentIndex(max(self.unmapped.findData(unmapped), 0))
        form.addRow("Unmapped Values Treatment:", self.unmapped)
        if var.value_labels:
            hint = QLabel(", ".join(f"{format_code(k)} = {v}"
                                    for k, v in var.value_labels.items()))
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

    def result(self) -> tuple[dict, str] | None:
        out = {}
        try:
            for key, ed in self.edits.items():
                vals = [t for t in ed.text().replace(",", " ").split() if t]
                out[key] = [float(v) for v in vals] if self.numeric else vals
        except ValueError:
            return None
        return out, self.unmapped.currentData()


class SelectCategoryDialog(SubDialog):
    """Baseline category of each fixed factor (Last, First or a chosen value)."""

    def __init__(self, ds, factors: list[str], current: dict, parent=None):
        super().__init__("Select Category", parent, columns=1)
        box = self.group("Baseline category of each factor")
        for f in factors:
            combo = QComboBox()
            combo.addItem("Last Category", "last")
            combo.addItem("First Category", "first")
            vals = sorted(ds.analysis_series(f).dropna().unique(),
                          key=lambda v: (isinstance(v, str), v))
            for v in vals[:200]:
                combo.addItem(f"Value: {format_code(v) if not isinstance(v, str) else v}", v)
            cur = current.get(f, "last")
            idx = combo.findData(cur)
            combo.setCurrentIndex(max(idx, 0))
            row = QFormLayout()
            row.addRow(f"{f}:", combo)
            box.addLayout(row)
            self._getters[f"ref::{f}"] = combo.currentData


class ParametricAFTDialog(VariablePickerDialog):
    TITLE = "Parametric Accelerated Failure Time (AFT) Models"
    RECURRENT = False
    DEFAULTS = {"mode": "survival", "status_map": None, "status_var": None,
                "unmapped": "exclude", "reference": {}, "level": 95.0,
                "status_treatment": "discard", "distribution": "weibull",
                "use_intercept": False, "init_intercept": 1.0, "init_scale": "ols",
                "scale_value": 1.0, "method": "auto", "max_iter": 20, "max_halving": 5,
                "print_encoding": False, "print_initial": False, "print_history": False,
                "history_steps": 1, "predict": [], "predict_times": "dependent",
                "interval": 1.0, "periods": 2, "duration": "", "past_time": "",
                "cox_snell": False, "plots": [], "n_points": 100, "separate": "",
                "max_lines": 10, "plot_values": "", "frailty": "gamma",
                "frailty_init": 0.0, "alpha": 0.05, "interval_treatment": "discard"}

    def __init__(self, ds, parent=None):
        super().__init__(ds, self.TITLE, parent)
        self.setMinimumSize(680, 600)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Time:"))
        self.rb_surv = QRadioButton("Survival")
        self.rb_se = QRadioButton("Start/End")
        grp = QButtonGroup(self)
        grp.addButton(self.rb_surv)
        grp.addButton(self.rb_se)
        mode_row.addWidget(self.rb_surv)
        mode_row.addWidget(self.rb_se)
        mode_row.addStretch()
        self.options_layout.addLayout(mode_row)
        self.w_time = self.add_target("time", "Survival:", accepts=is_numeric, single=True,
                                      required=False)
        self.w_start = self.add_target("start", "Start Time:", accepts=is_numeric,
                                       single=True, required=False)
        self.w_end = self.add_target("end", "End Time:", accepts=is_numeric, single=True,
                                     required=False)
        if self.RECURRENT:
            self.add_target("subject", "Subject:", single=True)
            self.add_target("interval_var", "Interval:", accepts=is_numeric, single=True,
                            required=False)
        self.add_target("status", "Status:", single=True, required=False)
        self.add_target("cov", "Covariate(s):", accepts=is_numeric, required=False)
        self.add_target("factors", "Fixed Factor(s):", required=False)
        if not self.RECURRENT:
            self.add_target("trunc", "Left Truncation:", accepts=is_numeric, single=True,
                            required=False)
        self.event_button = self.add_option_button("Define Event...", self._define_event)
        self.add_option_button("Select Category...", self._select_category)
        self.add_option_button("Criteria...", self._criteria)
        self.add_option_button("Model...", self._model)
        self.add_option_button("Estimate...", self._estimate)
        self.add_option_button("Print...", self._print)
        self.add_option_button("Predict...", self._predict)
        self.add_option_button("Plot...", self._plot)
        (self.rb_se if self.opts["mode"] == "startend" else self.rb_surv).setChecked(True)
        self.rb_surv.toggled.connect(self._mode_changed)
        self.finish()
        self._mode_changed()

    # ------------------------------------------------------------------ helpers
    def _target(self, key):
        return next(t for t in self._targets if t.key == key)

    def _mode_changed(self, *_):
        startend = self.rb_se.isChecked()
        self.opts["mode"] = "startend" if startend else "survival"
        pairs = [("time", not startend), ("start", startend), ("end", startend)]
        if not self.RECURRENT:
            pairs.append(("trunc", not startend))
        for key, on in pairs:
            t = self._target(key)
            t.widget.setEnabled(on)
            t.arrow.setEnabled(on)
            # a disabled list must not receive variables by double-click either
            t.accepts = is_numeric if on else (lambda v: False)
            if not on and t.widget.count():
                self._move_back(t, all_items=True)

    def _categories(self) -> list[str]:
        if self.RECURRENT:
            return ["event", "right"]
        return ["event", "right", "left"] + (["interval"] if self.rb_se.isChecked() else [])

    def _status_map(self):
        name = self.selected_one("status")
        if name and self.opts.get("status_var") == name:
            return self.opts.get("status_map")
        return None

    def _define_event(self):
        name = self.selected_one("status")
        if not name:
            return
        dlg = StatusMapDialog(self.ds.var(name), self._status_map(), self.opts["unmapped"],
                              self._categories(), self)
        if dlg.exec() and dlg.result() is not None:
            smap, unm = dlg.result()
            self.opts["status_map"] = smap
            self.opts["status_var"] = name
            self.opts["unmapped"] = unm
        self.targets_changed()

    def targets_changed(self):
        if not hasattr(self, "event_button"):
            return
        name = self.selected_one("status")
        self.event_button.setEnabled(bool(name))
        if name:
            smap = self._status_map()
            if smap:
                txt = ", ".join(f"{k[0].upper()}:{' '.join(format_code(v) if isinstance(v, float) else str(v) for v in vals)}"
                                for k, vals in smap.items() if vals)
                self.set_item_suffix("status", f"({txt})")
            else:
                self.set_item_suffix("status", "(default)")

    def _select_category(self):
        factors = self.selected("factors")
        if not factors:
            return
        values = SelectCategoryDialog(self.ds, factors, self.opts["reference"], self).run()
        if values:
            self.opts["reference"] = {k[5:]: v for k, v in values.items()
                                      if k.startswith("ref::")}

    def _criteria(self):
        dlg = SubDialog(f"{self.TITLE}: Criteria", self, columns=1)
        box = dlg.group("Criteria")
        dlg.number(box, "level", "Confidence Interval (%):", self.opts["level"], 50, 99.99, 2)
        if self.RECURRENT:
            dlg.number(box, "alpha", "Significance Level (frailty LR test):",
                       self.opts["alpha"], 0.0001, 0.9999, 4)
            box = dlg.group("Interval Treatment")
            dlg.radio(box, "interval_treatment",
                      [("discard", "Discard the conflicted records"),
                       ("discover", "Discover interval values based on the start and end "
                                    "time")], self.opts["interval_treatment"])
        else:
            box = dlg.group("Status Treatment (Start/End only)")
            dlg.radio(box, "status_treatment",
                      [("discard", "Discard conflicted record"),
                       ("status", "Obtain the time information according to the status"),
                       ("derive", "Derive the status according to the time information")],
                      self.opts["status_treatment"])
        box = dlg.group("Missing Values")
        box.addWidget(QLabel("User-missing and system-missing values are excluded."))
        values = dlg.run()
        if values:
            self.opts.update(values)

    def _model(self):
        dlg = SubDialog(f"{self.TITLE}: Model", self, columns=2)
        box = dlg.group("Distribution of Survival Time")
        dlg.radio(box, "distribution", list(sp.DISTRIBUTIONS.items()),
                  self.opts["distribution"])
        box = dlg.group("Initial Values")
        dlg.check(box, "use_intercept", "Initial value of intercept:",
                  self.opts["use_intercept"])
        dlg.number(box, "init_intercept", "Intercept:", self.opts["init_intercept"], -1e6,
                   1e6, 4)
        box.addWidget(QLabel("Initial value of scale parameter:"))
        dlg.radio(box, "init_scale",
                  [("ols", "Standard error of the corresponding OLS regression"),
                   ("inverse", "Inverse standard error of the corresponding OLS regression"),
                   ("user", "User-supplied value")], self.opts["init_scale"])
        dlg.number(box, "scale_value", "Value:", self.opts["scale_value"], 0.0001, 1e6, 4)
        if self.RECURRENT:
            box = dlg.group("Frailty Component")
            dlg.radio(box, "frailty", list(sp.FRAILTIES.items()), self.opts["frailty"])
            dlg.number(box, "frailty_init", "Initial value of variance (0 = default):",
                       self.opts["frailty_init"], 0, 1e4, 4)
        values = dlg.run()
        if values:
            self.opts.update(values)

    def _estimate(self):
        dlg = SubDialog(f"{self.TITLE}: Estimate", self, columns=1)
        box = dlg.group("Method")
        dlg.radio(box, "method", [("auto", "Auto"), ("newton", "Newton-Raphson")],
                  self.opts["method"])
        note = QLabel("Both use Newton-Raphson with step-halving here. L1 regularization "
                      "(ADMM feature selection) is not available.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#666")
        box.addWidget(note)
        box = dlg.group("Iteration")
        dlg.number(box, "max_iter", "Maximum iterations:", self.opts["max_iter"], 1, 300, 0)
        dlg.number(box, "max_halving", "Maximum step-halving:", self.opts["max_halving"], 1,
                   200, 0)
        values = dlg.run()
        if values:
            values["max_iter"] = int(values["max_iter"])
            values["max_halving"] = int(values["max_halving"])
            self.opts.update(values)

    def _print(self):
        dlg = SubDialog(f"{self.TITLE}: Print", self, columns=1)
        box = dlg.group("Print")
        dlg.check(box, "print_encoding", "Factor encoding details",
                  self.opts["print_encoding"])
        dlg.check(box, "print_initial", "Initial values assigned to the regression parameters",
                  self.opts["print_initial"])
        dlg.check(box, "print_history", "Model iteration history", self.opts["print_history"])
        dlg.number(box, "history_steps", "Number of steps:", self.opts["history_steps"], 1,
                   99999999, 0)
        values = dlg.run()
        if values:
            values["history_steps"] = int(values["history_steps"])
            self.opts.update(values)

    def _numeric_combo(self, current: str) -> QComboBox:
        combo = QComboBox()
        combo.addItem("(none)", "")
        for v in self.ds.variables:
            if v.type == NUMERIC:
                combo.addItem(v.name, v.name)
        combo.setCurrentIndex(max(combo.findData(current), 0))
        return combo

    def _predict(self):
        dlg = SubDialog(f"{self.TITLE}: Predict", self, columns=2)
        box = dlg.group("Time Values for Scoring")
        dlg.radio(box, "predict_times",
                  [("dependent", "Time values defined by dependent variable(s)"),
                   ("regular", "Regular intervals"), ("duration", "Time duration variable")],
                  self.opts["predict_times"])
        dlg.number(box, "interval", "Time interval:", self.opts["interval"], 1e-6, 1e9, 4)
        dlg.number(box, "periods", "Number of time periods:", self.opts["periods"], 2, 100, 0)
        dur = self._numeric_combo(self.opts["duration"])
        row = QFormLayout()
        row.addRow("Time duration:", dur)
        box.addLayout(row)
        dlg._getters["duration"] = dur.currentData
        box = dlg.group("Predictions")
        items = [("survival", "Survival (PredSurvival)"), ("hazard", "Hazard (PredHazard)"),
                 ("cumhazard", "Cumulative hazard (PredCumHazard)"),
                 ("conditional", "Conditional survival (PredConditionalSurvival)")]
        if self.RECURRENT:
            items += [("uncond_survival", "Unconditional survival (PredUnCondSurvival)"),
                      ("uncond_hazard", "Unconditional hazard (PredUncondHazard)"),
                      ("uncond_cumhazard", "Unconditional cum hazard (PredUncondCumHazard)")]
        for key, text in items:
            dlg.check(box, f"p::{key}", text, key in self.opts["predict"])
        past = self._numeric_combo(self.opts["past_time"])
        row = QFormLayout()
        row.addRow("Past survival time:", past)
        box.addLayout(row)
        dlg._getters["past_time"] = past.currentData
        values = dlg.run()
        if values:
            self.opts["predict"] = [k[3:] for k, v in values.items() if k.startswith("p::") and v]
            for key in ("predict_times", "interval", "duration", "past_time"):
                self.opts[key] = values[key]
            self.opts["periods"] = int(values["periods"])

    def _plot(self):
        dlg = SubDialog(f"{self.TITLE}: Plot", self, columns=2)
        if not self.RECURRENT:
            box = dlg.group("Cox-Snell Residual Plot")
            dlg.check(box, "cox_snell", "Display the plot", self.opts["cox_snell"])
        box = dlg.group("Function Plots")
        for key, text in (("survival", "Survival"), ("hazard", "Hazard"),
                          ("density", "Density")):
            dlg.check(box, f"f::{key}", text, key in self.opts["plots"])
        dlg.number(box, "n_points", "Number of points to display:", self.opts["n_points"], 1,
                   200, 0)
        box = dlg.group("Pattern")
        sep = QComboBox()
        sep.addItem("(none)", "")
        for f in self.selected("factors"):
            sep.addItem(f, f)
        sep.setCurrentIndex(max(sep.findData(self.opts["separate"]), 0))
        row = QFormLayout()
        row.addRow("Separate lines for:", sep)
        values_edit = QLineEdit(self.opts["plot_values"])
        values_edit.setPlaceholderText("e.g. age=40; female=1")
        row.addRow("Covariate / factor values:", values_edit)
        box.addLayout(row)
        dlg._getters["separate"] = sep.currentData
        dlg._getters["plot_values"] = values_edit.text
        dlg.number(box, "max_lines", "Maximum number of lines in a chart:",
                   self.opts["max_lines"], 1, 100, 0)
        values = dlg.run()
        if values:
            self.opts["plots"] = [k[3:] for k, v in values.items() if k.startswith("f::") and v]
            for key in ("separate", "plot_values"):
                self.opts[key] = values[key]
            if "cox_snell" in values:
                self.opts["cox_snell"] = values["cox_snell"]
            self.opts["n_points"] = int(values["n_points"])
            self.opts["max_lines"] = int(values["max_lines"])

    def _plot_values(self) -> dict:
        out = {}
        for part in self.opts.get("plot_values", "").replace(",", ";").split(";"):
            if "=" not in part:
                continue
            k, v = (s.strip() for s in part.split("=", 1))
            if not self.ds.has_name(k):
                continue
            name = self.ds.var(k).name
            if self.ds.var(name).type == NUMERIC:
                try:
                    out[name] = float(v)
                except ValueError:
                    continue
            else:
                out[name] = v
        return out

    # ------------------------------------------------------------------ run
    def validate(self):
        startend = self.rb_se.isChecked()
        if startend and not (self.selected_one("start") and self.selected_one("end")):
            return "Select the Start Time and End Time variables."
        if not startend and not self.selected_one("time"):
            return "Select the Survival time variable."
        used = [self.selected_one(k) for k in ("time", "start", "end", "status")]
        if set(self.selected("cov")) & set(self.selected("factors")):
            return "A variable cannot be both a covariate and a fixed factor."
        if any(u and (u in self.selected("cov") or u in self.selected("factors"))
               for u in used):
            return "The time and status variables cannot also be predictors."
        if self.opts["predict_times"] == "duration" and self.opts["predict"] and \
                not self.opts["duration"]:
            return "Choose the time duration variable (Predict...)."
        return None

    def compute(self):
        o = self.opts
        startend = self.rb_se.isChecked()
        ref = {f: v for f, v in o["reference"].items() if f in self.selected("factors")}
        kw = dict(
            time=None if startend else self.selected_one("time"),
            start=self.selected_one("start") if startend else None,
            end=self.selected_one("end") if startend else None,
            status=self.selected_one("status"), covariates=self.selected("cov"),
            factors=self.selected("factors"), status_map=self._status_map(),
            unmapped=o["unmapped"], distribution=o["distribution"], reference=ref,
            level=o["level"] / 100, status_treatment=o["status_treatment"],
            init_intercept=o["init_intercept"] if o["use_intercept"] else None,
            init_scale=o["init_scale"], scale_value=o["scale_value"],
            max_iter=int(o["max_iter"]), max_halving=int(o["max_halving"]),
            print_encoding=o["print_encoding"], print_initial=o["print_initial"],
            print_history=o["print_history"], history_steps=int(o["history_steps"]),
            cox_snell=o["cox_snell"] and not self.RECURRENT,
            function_plots=tuple(o["plots"]), n_points=int(o["n_points"]),
            plot_values=self._plot_values(),
            separate=o["separate"] if o["separate"] in self.selected("factors") else None,
            max_lines=int(o["max_lines"]), predict=tuple(o["predict"]),
            predict_times=o["predict_times"], interval=o["interval"],
            periods=int(o["periods"]), duration=o["duration"] or None,
            past_time=o["past_time"] or None)
        if self.RECURRENT:
            kw.update(subject=self.selected_one("subject"), frailty=o["frailty"],
                      frailty_init=o["frailty_init"] or None,
                      interval_var=self.selected_one("interval_var")
                      if o["interval_treatment"] == "discard" else None,
                      alpha_frailty=o["alpha"])
        else:
            kw.update(truncation=None if startend else self.selected_one("trunc"))
        res = sp.parametric_survival(self.ds, **kw)
        return res if res.saved else res.section


class SharedFrailtyDialog(ParametricAFTDialog):
    TITLE = "Parametric Shared Frailty Models"
    RECURRENT = True

    def validate(self):
        err = super().validate()
        if err:
            return err
        if self.selected_one("subject") in (self.selected("cov") + self.selected("factors")):
            return "The subject variable cannot also be a predictor."
        return None


MENUS = {
    "Analyze/Survival": [("Parametric Accelerated Failure Time (AFT) Models...",
                          ParametricAFTDialog),
                         ("Parametric Shared Frailty Models...", SharedFrailtyDialog)],
}


def selftest() -> None:
    """Headless check of the parametric survival core on synthetic data."""
    from rsstatchem.core.dataset import Dataset

    sp.selftest()
    rng = np.random.default_rng(8)
    n = 60
    lo = rng.uniform(1, 5, n)
    hi = lo + rng.uniform(0.5, 3, n)
    ds = Dataset(pd.DataFrame({"lo": lo, "hi": hi, "x": rng.normal(size=n)}))
    r = sp.parametric_survival(ds, start="lo", end="hi", covariates=["x"],
                               distribution="lognormal")
    assert r.fit is not None

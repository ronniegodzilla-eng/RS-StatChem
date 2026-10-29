"""Analyze > Generalized Linear Models (Generalized Linear Models..., Generalized Estimating
Equations...), Analyze > Regression > Probit... and Analyze > Loglinear (General..., Logit...,
Model Selection...) dialogs.

SPSS shows GENLIN as a tabbed dialog; here the tabs are the option buttons on the right
(Type of Model, Response, Model, Estimation, Statistics, EM Means, Save; Repeated for GEE).
"""
from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit

from rsstatchem.core.dataset import NUMERIC
from rsstatchem.core.stats import boot_procs, genlin
from rsstatchem.ui.dialogs.base import SubDialog, VariablePickerDialog, is_numeric
from rsstatchem.ui.dialogs.resample_ui import ResampleMixin
from rsstatchem.ui.dialogs.logistic import TermBuilderDialog, _multi


class GenlinDialog(ResampleMixin, VariablePickerDialog):
    GEE = False
    TITLE = "Generalized Linear Models"
    DEFAULTS = {"model_type": "linear", "distribution": "normal", "link": "identity",
                "power": 1.0, "tweedie_p": 1.5, "nb_k": 1.0, "nb_estimate": False,
                "reference": "last", "factor_order": "ascending", "trials_mode": "none",
                "trials_value": 1.0, "terms": [], "intercept": True, "method": "hybrid",
                "scale_method": "ml", "scale_value": 1.0, "cov_type": "model",
                "max_iter": 100, "level": 95.0, "analysis_type": 3, "chi_type": "wald",
                "exp_params": False, "cov_matrix": False, "corr_matrix": False,
                "descriptives": True, "iteration_history": False, "emmeans": [],
                "emm_scale": "original", "emm_adjust": "lsd", "emm_pairwise": True,
                "save": [], "corr": "exchangeable", "m": 1, "gee_cov": "robust"}

    def __init__(self, ds, parent=None):
        super().__init__(ds, self.TITLE, parent)
        self.setMinimumSize(660, 500)
        self.add_target("dep", "Dependent Variable:", single=True)
        lists = [self.add_target("factors", "Factors:", required=False),
                 self.add_target("covs", "Covariates:", accepts=is_numeric, required=False)]
        self.add_target("offset", "Offset Variable:", accepts=is_numeric, single=True,
                        required=False)
        self.add_target("weight", "Scale Weight Variable:", accepts=is_numeric, single=True,
                        required=False)
        self.add_target("trials", "Trials Variable (binomial events/trials):",
                        accepts=is_numeric, single=True, required=False)
        if self.GEE:
            lists.append(self.add_target("subject", "Subject Variables:", required=True))
            lists.append(self.add_target("within", "Within-subject Variables:",
                                         required=False))
        for widget in lists:
            widget.setMaximumHeight(110 if self.GEE else 150)
        self.add_option_button("Type of Model...", self._type)
        self.add_option_button("Response...", self._response)
        self.add_option_button("Model...", self._model)
        self.add_option_button("Estimation...", self._estimation)
        self.add_option_button("Statistics...", self._statistics)
        self.add_option_button("EM Means...", self._emmeans)
        self.add_option_button("Save...", self._save)
        if self.GEE:
            self.add_option_button("Repeated...", self._repeated)
        else:
            self.add_bootstrap_button()
        self.summary = QLabel()
        self.summary.setStyleSheet("color:#555")
        self.options_layout.addWidget(self.summary)
        self.finish()
        self._show_summary()

    def _show_summary(self) -> None:
        o = self.opts
        self.summary.setText(f"Distribution: {genlin.DISTRIBUTIONS[o['distribution']]}, "
                             f"Link: {genlin.LINKS[o['link']]}"
                             + (f", Working correlation: {genlin.CORR_TYPES[o['corr']]}"
                                if self.GEE else ""))

    # ------------------------------------------------------------- sub-dialogs
    def _type(self) -> None:
        dlg = SubDialog(f"{self.TITLE}: Type of Model", self)
        box = dlg.group("Preset")
        dlg.radio(box, "model_type", [
            ("linear", "Scale Response: Linear"), ("gamma_log", "Gamma with log link"),
            ("poisson", "Counts: Poisson loglinear"),
            ("negbin", "Negative binomial with log link"),
            ("binary_logistic", "Binary Response: Binary logistic"),
            ("binary_probit", "Binary probit"),
            ("interval_censored", "Interval censored survival"),
            ("tweedie_log", "Mixture: Tweedie with log link"),
            ("tweedie_identity", "Tweedie with identity link"),
            ("custom", "Custom")], self.opts["model_type"])
        box = dlg.group("Custom")
        dist = QComboBox()
        for k, v in genlin.DISTRIBUTIONS.items():
            dist.addItem(v, k)
        dist.setCurrentIndex(max(dist.findData(self.opts["distribution"]), 0))
        link = QComboBox()
        for k, v in genlin.LINKS.items():
            link.addItem(v, k)
        link.setCurrentIndex(max(link.findData(self.opts["link"]), 0))
        form = QFormLayout()
        form.addRow("Distribution:", dist)
        form.addRow("Link function:", link)
        box.addLayout(form)
        dlg.number(box, "power", "Power (link):", self.opts["power"], -10, 10, 2)
        dlg.number(box, "tweedie_p", "Tweedie power:", self.opts["tweedie_p"], 1.01, 1.99, 2)
        dlg.number(box, "nb_k", "Negative binomial ancillary:", self.opts["nb_k"], 0.001, 100,
                   3)
        dlg.check(box, "nb_estimate", "Estimate negative binomial ancillary parameter",
                  self.opts["nb_estimate"])
        values = dlg.run()
        if values:
            self.opts.update(values)
            if values["model_type"] == "custom":
                self.opts["distribution"], self.opts["link"] = dist.currentData(), \
                    link.currentData()
            else:
                self.opts["distribution"], self.opts["link"] = \
                    genlin.MODEL_TYPES[values["model_type"]]
            self._show_summary()

    def _response(self) -> None:
        dlg = SubDialog(f"{self.TITLE}: Response", self, columns=1)
        dlg.radio(dlg.group("Binary response: Reference Category"), "reference", [
            ("last", "Last (default)"), ("first", "First")], self.opts["reference"])
        box = dlg.group("Number of events occurring in a set of trials")
        dlg.radio(box, "trials_mode", [("none", "Dependent is binary (not events/trials)"),
                                       ("variable", "Trials variable (target list)"),
                                       ("fixed", "Fixed value")], self.opts["trials_mode"])
        dlg.number(box, "trials_value", "Fixed number of trials:", self.opts["trials_value"],
                   1, 1e6, 0)
        dlg.radio(dlg.group("Category Order for Factors"), "factor_order", [
            ("ascending", "Ascending"), ("descending", "Descending")],
            self.opts["factor_order"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def _model(self) -> None:
        names = self.selected("factors") + self.selected("covs")
        top = QHBoxLayout()
        dlg = TermBuilderDialog(f"{self.TITLE}: Model", names,
                                [("terms", "Model (empty = main effects):", self.opts["terms"])],
                                self, top)
        from PySide6.QtWidgets import QCheckBox

        cb = QCheckBox("Include intercept in model")
        cb.setChecked(self.opts["intercept"])
        top.addWidget(cb)
        if dlg.exec():
            self.opts["terms"] = dlg.terms("terms")
            self.opts["intercept"] = cb.isChecked()

    def _estimation(self) -> None:
        dlg = SubDialog(f"{self.TITLE}: Estimation", self)
        box = dlg.group("Parameter Estimation")
        dlg.radio(box, "method", [("hybrid", "Hybrid"), ("fisher", "Fisher"),
                                  ("newton", "Newton-Raphson")], self.opts["method"])
        dlg.radio(box, "scale_method", [("ml", "Scale: Maximum likelihood estimate"),
                                        ("deviance", "Deviance"), ("pearson", "Pearson chi-square"),
                                        ("fixed", "Fixed value")], self.opts["scale_method"])
        dlg.number(box, "scale_value", "Fixed scale value:", self.opts["scale_value"], 1e-6,
                   1e6, 4)
        box = dlg.group("Covariance Matrix / Iterations")
        dlg.radio(box, "cov_type", [("model", "Model-based estimator"),
                                    ("robust", "Robust estimator")], self.opts["cov_type"])
        dlg.number(box, "max_iter", "Maximum iterations:", self.opts["max_iter"], 1, 1000, 0)
        values = dlg.run()
        if values:
            values["max_iter"] = int(values["max_iter"])
            self.opts.update(values)

    def _statistics(self) -> None:
        dlg = SubDialog(f"{self.TITLE}: Statistics", self)
        box = dlg.group("Chi-Square Statistics / Analysis Type")
        dlg.radio(box, "chi_type", [("wald", "Wald"), ("lr", "Likelihood ratio")],
                  self.opts["chi_type"])
        dlg.radio(box, "analysis_type", [(3, "Type III"), (1, "Type I"), (13, "Type I and III")],
                  self.opts["analysis_type"])
        dlg.number(box, "level", "Confidence interval level (%):", self.opts["level"], 50,
                   99.99, 2)
        box = dlg.group("Print")
        for key, text in (("descriptives", "Descriptive statistics"),
                          ("exp_params", "Include exponential parameter estimates"),
                          ("cov_matrix", "Covariance matrix for parameter estimates"),
                          ("corr_matrix", "Correlation matrix for parameter estimates"),
                          ("iteration_history", "Iteration history")):
            dlg.check(box, key, text, self.opts[key])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def _emmeans(self) -> None:
        factors = self.selected("factors")
        dlg = SubDialog(f"{self.TITLE}: EM Means", self)
        box = dlg.group("Display Means for")
        for f in factors:
            dlg.check(box, f"emm::{f}", f, f in self.opts["emmeans"])
        inter = QLineEdit(" ".join(t for t in self.opts["emmeans"] if "*" in t))
        inter.setPlaceholderText("interactions, e.g. a*b")
        box.addWidget(inter)
        box = dlg.group("Options")
        dlg.check(box, "emm_pairwise", "Pairwise comparisons", self.opts["emm_pairwise"])
        dlg.radio(box, "emm_scale", [("original", "Compute means for response"),
                                     ("linear", "Compute means for linear predictor")],
                  self.opts["emm_scale"])
        dlg.radio(box, "emm_adjust", list(genlin.ADJUSTMENTS.items()),
                  self.opts["emm_adjust"])
        values = dlg.run()
        if values:
            chosen = [f for f in factors if values.pop(f"emm::{f}", False)]
            chosen += [t for t in inter.text().split() if "*" in t]
            self.opts["emmeans"] = chosen
            self.opts.update(values)

    def _save(self) -> None:
        _multi(self, f"{self.TITLE}: Save", "save", [
            ("Items to Save", [(k, v[1]) for k, v in genlin.GENLIN_SAVE.items()])], columns=1)

    def _repeated(self) -> None:
        dlg = SubDialog(f"{self.TITLE}: Repeated", self, columns=1)
        box = dlg.group("Working Correlation Matrix")
        dlg.radio(box, "corr", list(genlin.CORR_TYPES.items()), self.opts["corr"])
        dlg.number(box, "m", "M (M-dependent):", self.opts["m"], 1, 20, 0)
        dlg.radio(dlg.group("Covariance Matrix"), "gee_cov", [
            ("robust", "Robust estimator"), ("model", "Model-based estimator")],
            self.opts["gee_cov"])
        values = dlg.run()
        if values:
            values["m"] = int(values["m"])
            self.opts.update(values)
            self._show_summary()

    # ------------------------------------------------------------- run
    def validate(self):
        o = self.opts
        if o["distribution"] == "binomial" and o["trials_mode"] == "variable" and \
                not self.selected("trials"):
            return "Select the trials variable, or choose another response type."
        if o["distribution"] != "binomial" and self.ds.var(self.selected_one("dep")).type != \
                NUMERIC:
            return "The dependent variable must be numeric for this distribution."
        names = set(self.selected("factors") + self.selected("covs"))
        for t in o["terms"]:
            if not set(t.split("*")) <= names:
                return f"The model term {t} uses a variable that is no longer selected."
        return None

    def compute(self):
        o = self.opts
        trials = None
        if o["distribution"] == "binomial":
            if o["trials_mode"] == "variable":
                trials = self.selected_one("trials")
            elif o["trials_mode"] == "fixed":
                trials = float(o["trials_value"])
        repeated = None
        if self.GEE:
            repeated = {"subject": self.selected("subject"), "within": self.selected("within"),
                        "corr": o["corr"], "m": o["m"], "cov": o["gee_cov"]}
        factors = self.selected("factors")
        return self.resampled(
            lambda: self._genlin(trials, repeated, factors),
            boot=None if self.GEE else
            (lambda r, spec: boot_procs.models(r.section, self.ds, r.models, spec)),
            pool="models", level=o["level"] / 100)

    def _genlin(self, trials, repeated, factors):
        o = self.opts
        return genlin.genlin(
            self.ds, self.selected_one("dep"), factors, self.selected("covs"),
            o["distribution"], o["link"], terms=o["terms"] or None, intercept=o["intercept"],
            offset=self.selected_one("offset"), scale_weight=self.selected_one("weight"),
            trials=trials, reference=o["reference"], factor_order=o["factor_order"],
            power=o["power"], tweedie_p=o["tweedie_p"], nb_k=o["nb_k"],
            nb_estimate=o["nb_estimate"], scale_method=o["scale_method"],
            scale_value=o["scale_value"], cov_type=o["cov_type"], method=o["method"],
            max_iter=int(o["max_iter"]), level=o["level"] / 100,
            analysis_type=int(o["analysis_type"]), chi_type=o["chi_type"],
            exp_params=o["exp_params"], cov_matrix=o["cov_matrix"],
            corr_matrix=o["corr_matrix"], descriptives=o["descriptives"],
            iteration_history=o["iteration_history"],
            emmeans=[t for t in o["emmeans"] if set(t.split("*")) <= set(factors)],
            emm_scale=o["emm_scale"], emm_adjust=o["emm_adjust"],
            emm_pairwise=o["emm_pairwise"], save=tuple(o["save"]), repeated=repeated)


class GeeDialog(GenlinDialog):
    GEE = True
    TITLE = "Generalized Estimating Equations"
    DEFAULTS = dict(GenlinDialog.DEFAULTS)


# ============================================================================ Probit
class ProbitDialog(VariablePickerDialog):
    DEFAULTS = {"model": "probit", "transform": "log10", "natural": "none",
                "natural_value": 0.0, "frequencies": True, "potency": False,
                "parallelism": False, "fiducial": True, "het_alpha": 0.15, "level": 95.0}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Probit Analysis", parent)
        self.add_target("resp", "Response Frequency:", accepts=is_numeric, single=True)
        self.add_target("total", "Total Observed:", accepts=is_numeric, single=True)
        self.add_target("factor", "Factor:", single=True, required=False)
        self.add_target("cov", "Covariate(s):", accepts=is_numeric)
        self.add_option_button("Options...", self._options)
        row = QHBoxLayout()
        self.transform = QComboBox()
        for k, v in (("none", "None"), ("log10", "Log base 10"), ("ln", "Natural log")):
            self.transform.addItem(v, k)
        self.transform.setCurrentIndex(max(self.transform.findData(self.opts["transform"]), 0))
        self.transform.currentIndexChanged.connect(
            lambda _: self.opts.__setitem__("transform", self.transform.currentData()))
        self.model = QComboBox()
        self.model.addItem("Probit", "probit")
        self.model.addItem("Logit", "logit")
        self.model.setCurrentIndex(max(self.model.findData(self.opts["model"]), 0))
        self.model.currentIndexChanged.connect(
            lambda _: self.opts.__setitem__("model", self.model.currentData()))
        form = QFormLayout()
        form.addRow("Transform:", self.transform)
        form.addRow("Statistical Model:", self.model)
        row.addLayout(form)
        self.options_layout.addLayout(row)
        self.finish()

    def _options(self) -> None:
        dlg = SubDialog("Probit Analysis: Options", self)
        box = dlg.group("Statistics")
        dlg.check(box, "frequencies", "Frequencies", self.opts["frequencies"])
        dlg.check(box, "potency", "Relative median potency", self.opts["potency"])
        dlg.check(box, "parallelism", "Parallelism test", self.opts["parallelism"])
        dlg.check(box, "fiducial", "Fiducial confidence intervals", self.opts["fiducial"])
        dlg.number(box, "level", "Confidence level (%):", self.opts["level"], 50, 99.99, 2)
        box = dlg.group("Natural Response Rate")
        dlg.radio(box, "natural", [("none", "None"), ("estimate", "Calculate from data"),
                                   ("value", "Value:")], self.opts["natural"])
        dlg.number(box, "natural_value", "Value:", self.opts["natural_value"], 0, 0.99, 3)
        box = dlg.group("Criteria")
        dlg.number(box, "het_alpha", "Heterogeneity significance level:",
                   self.opts["het_alpha"], 0.001, 1, 3)
        values = dlg.run()
        if values:
            self.opts.update(values)

    def compute(self):
        o = self.opts
        return genlin.probit(self.ds, self.selected_one("resp"), self.selected_one("total"),
                             self.selected("cov"), self.selected_one("factor"), o["model"],
                             o["transform"], o["natural"], o["natural_value"],
                             o["level"] / 100, o["frequencies"], o["potency"],
                             o["parallelism"], o["fiducial"], o["het_alpha"])


# ============================================================================ Loglinear
class _LoglinearBase(VariablePickerDialog):
    def _model_terms(self, names: list[str], title: str) -> None:
        from PySide6.QtWidgets import QButtonGroup, QRadioButton

        top = QHBoxLayout()
        group = QButtonGroup(self)
        sat, cus = QRadioButton("Saturated"), QRadioButton("Custom")
        group.addButton(sat)
        group.addButton(cus)
        (cus if self.opts["custom"] else sat).setChecked(True)
        top.addWidget(sat)
        top.addWidget(cus)
        dlg = TermBuilderDialog(title, names, [("terms", "Terms in Model:", self.opts["terms"])],
                                self, top)
        if dlg.exec():
            self.opts["custom"] = cus.isChecked()
            self.opts["terms"] = dlg.terms("terms")

    def _display_options(self, extra=None) -> None:
        dlg = SubDialog(f"{self.windowTitle()}: Options", self)
        box = dlg.group("Display")
        dlg.check(box, "frequencies", "Frequencies", self.opts["frequencies"])
        dlg.check(box, "residuals", "Residuals", self.opts["residuals"])
        dlg.check(box, "estimates", "Estimates", self.opts["estimates"])
        if extra:
            extra(dlg, box)
        box = dlg.group("Criteria")
        dlg.number(box, "delta", "Delta (saturated models):", self.opts["delta"], 0, 1, 2)
        dlg.number(box, "level", "Confidence interval (%):", self.opts["level"], 50, 99.99, 2)
        values = dlg.run()
        if values:
            self.opts.update(values)


class LoglinearGeneralDialog(_LoglinearBase):
    DEFAULTS = {"custom": False, "terms": [], "distribution": "poisson", "frequencies": True,
                "residuals": True, "estimates": True, "delta": 0.5, "level": 95.0}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "General Loglinear Analysis", parent)
        self.add_target("factors", "Factor(s):", min_items=1)
        self.add_target("cellcov", "Cell Covariate(s):", accepts=is_numeric, required=False)
        self.add_option_button("Model...", lambda: self._model_terms(
            self.selected("factors") + self.selected("cellcov"),
            "General Loglinear Analysis: Model"))
        self.add_option_button("Options...", self._display_options)
        dist = QComboBox()
        dist.addItem("Poisson", "poisson")
        dist.addItem("Multinomial", "multinomial")
        dist.setCurrentIndex(max(dist.findData(self.opts["distribution"]), 0))
        dist.currentIndexChanged.connect(
            lambda _: self.opts.__setitem__("distribution", dist.currentData()))
        form = QFormLayout()
        form.addRow("Distribution of Cell Counts:", dist)
        self.options_layout.addLayout(form)
        self.finish()

    def compute(self):
        o = self.opts
        return genlin.loglinear_general(
            self.ds, self.selected("factors"), o["terms"] if o["custom"] and o["terms"]
            else None, self.selected("cellcov"), o["distribution"], o["delta"],
            o["level"] / 100, o["frequencies"], o["residuals"], o["estimates"])


class LoglinearLogitDialog(_LoglinearBase):
    DEFAULTS = {"custom": False, "terms": [], "frequencies": True, "residuals": True,
                "estimates": True, "delta": 0.5, "level": 95.0}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Logit Loglinear Analysis", parent)
        self.add_target("deps", "Dependent:", min_items=1)
        self.add_target("factors", "Factor(s):", min_items=1)
        self.add_option_button("Model...", lambda: self._model_terms(
            self.selected("deps") + self.selected("factors"),
            "Logit Loglinear Analysis: Model"))
        self.add_option_button("Options...", self._display_options)
        self.finish()

    def compute(self):
        o = self.opts
        return genlin.loglinear_logit(
            self.ds, self.selected("deps"), self.selected("factors"),
            o["terms"] if o["custom"] and o["terms"] else None, o["delta"], o["level"] / 100,
            o["frequencies"], o["residuals"], o["estimates"])


class LoglinearSelectionDialog(_LoglinearBase):
    DEFAULTS = {"custom": False, "terms": [], "method": "backward", "p_remove": 0.05,
                "max_steps": 10, "frequencies": True, "residuals": True, "estimates": True,
                "association": True, "delta": 0.5, "level": 95.0}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Model Selection Loglinear Analysis", parent)
        self.add_target("factors", "Factor(s):", min_items=2)
        self.add_option_button("Model...", lambda: self._model_terms(
            self.selected("factors"), "Loglinear Analysis: Model"))
        self.add_option_button("Options...", lambda: self._display_options(self._extra))
        method = QComboBox()
        method.addItem("Use backward elimination", "backward")
        method.addItem("Enter in single step", "enter")
        method.setCurrentIndex(max(method.findData(self.opts["method"]), 0))
        method.currentIndexChanged.connect(
            lambda _: self.opts.__setitem__("method", method.currentData()))
        form = QFormLayout()
        form.addRow("Model Building:", method)
        self.options_layout.addLayout(form)
        self.finish()

    def _extra(self, dlg, box) -> None:
        dlg.check(box, "association", "Association table", self.opts["association"])
        dlg.number(box, "p_remove", "Probability for removal:", self.opts["p_remove"], 0.0001,
                   1, 4)
        dlg.number(box, "max_steps", "Maximum steps:", self.opts["max_steps"], 1, 100, 0)

    def compute(self):
        o = self.opts
        return genlin.loglinear_selection(
            self.ds, self.selected("factors"), o["method"],
            o["terms"] if o["custom"] and o["terms"] else None, o["p_remove"],
            int(o["max_steps"]), o["delta"], o["frequencies"], o["residuals"],
            o["association"], o["estimates"], o["level"] / 100)


MENUS = {
    "Analyze/Generalized Linear Models": [
        ("Generalized Linear Models...", GenlinDialog),
        ("Generalized Estimating Equations...", GeeDialog)],
    "Analyze/Regression": [("Probit...", ProbitDialog)],
    "Analyze/Loglinear": [("General...", LoglinearGeneralDialog),
                          ("Logit...", LoglinearLogitDialog),
                          ("Model Selection...", LoglinearSelectionDialog)],
}


def selftest() -> None:
    """Headless check of the GENLIN, GEE, PROBIT and loglinear cores."""
    genlin.selftest()

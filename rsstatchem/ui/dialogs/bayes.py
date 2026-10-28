"""Analyze > Bayesian Statistics dialogs (SPSS 25+). Each dialog has SPSS's "Bayesian Analysis"
choice (Characterize Posterior Distribution / Estimate Bayes Factor / Use Both Methods) and
Criteria..., Priors... and Bayes Factor... sub-dialogs. Core code:
:mod:`rsstatchem.core.stats.bayes`."""
from __future__ import annotations

from PySide6.QtWidgets import (QButtonGroup, QFormLayout, QGroupBox, QHBoxLayout, QLineEdit,
                               QRadioButton)

from rsstatchem.core.dataset import NUMERIC
from rsstatchem.core.stats import bayes as bs
from rsstatchem.ui.dialogs.base import SubDialog, VariablePickerDialog, is_numeric
from rsstatchem.ui.dialogs.compare import GroupingMixin


def _num(text: str) -> float:
    return float(text.strip().replace(",", "."))


class _BayesDialog(VariablePickerDialog):
    BASE = {"analysis": "both", "level": 95.0, "bf10": False, "plots": False}
    DEFAULTS = dict(BASE)
    TEST_VALUE: str | None = None     # label of a "Test Value" line edit

    def setup_common(self) -> None:
        box = QGroupBox("Bayesian Analysis")
        lay = QHBoxLayout(box)
        self._an = {}
        group = QButtonGroup(self)
        for key, text in (("posterior", "Characterize Posterior Distribution"),
                          ("bf", "Estimate Bayes Factor"), ("both", "Use Both Methods")):
            rb = QRadioButton(text)
            rb.setChecked(self.opts["analysis"] == key)
            rb.toggled.connect(lambda on, k=key: on and self.opts.__setitem__("analysis", k))
            group.addButton(rb)
            lay.addWidget(rb)
            self._an[key] = rb
        self.options_layout.addWidget(box)
        if self.TEST_VALUE:
            form = QFormLayout()
            self.test_edit = QLineEdit(f"{self.opts.get('test_value', 0.0):g}")
            form.addRow(self.TEST_VALUE, self.test_edit)
            self.options_layout.addLayout(form)
        self.add_option_button("Criteria...", self._criteria)

    def _criteria(self):
        dlg = SubDialog(f"{self.windowTitle()}: Criteria", self, columns=1)
        box = dlg.group("Credible Interval")
        dlg.number(box, "level", "Credible interval percentage (%):", self.opts["level"], 50,
                   99.99, 2)
        box = dlg.group("Bayes Factor")
        dlg.radio(box, "bf10", [(False, "BF01: null versus alternative (SPSS default)"),
                                (True, "BF10: alternative versus null")], self.opts["bf10"])
        box = dlg.group("Plots")
        dlg.check(box, "plots", "Plot prior and posterior distributions", self.opts["plots"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def bopts(self) -> bs.BayesOptions:
        return bs.BayesOptions(self.opts["analysis"], self.opts["level"] / 100,
                               self.opts["bf10"], self.opts["plots"])

    def validate(self):
        if self.TEST_VALUE:
            try:
                self.opts["test_value"] = _num(self.test_edit.text())
            except ValueError:
                return "The test value must be a number."
        return None


class BayesOneSampleNormalDialog(_BayesDialog):
    DEFAULTS = {**_BayesDialog.BASE, "test_value": 0.0, "r": 1.0, "prior": "diffuse",
                "known": False, "known_var": 1.0, "mu0": 0.0, "kappa0": 1.0, "a0": 1.0,
                "b0": 1.0}
    TEST_VALUE = "Test Value:"
    TITLE = "Bayesian One Sample Inference: Normal"

    def __init__(self, ds, parent=None):
        super().__init__(ds, self.TITLE, parent)
        self.build_targets()
        self.setup_common()
        self.add_option_button("Priors...", self._priors)
        self.add_option_button("Bayes Factor...", self._bf)
        self.finish()

    def build_targets(self):
        self.add_target("vars", "Test Variable(s):", accepts=is_numeric)

    def _priors(self):
        dlg = SubDialog(f"{self.windowTitle()}: Prior Distribution", self)
        box = dlg.group("Data Variance")
        dlg.check(box, "known", "Variance known", self.opts["known"])
        dlg.number(box, "known_var", "Variance value:", self.opts["known_var"], 1e-9, 1e9, 4)
        box = dlg.group("Prior on Mean (and Variance)")
        dlg.radio(box, "prior", [("diffuse", "Diffuse (Jeffreys)"),
                                 ("conjugate", "Normal / inverse gamma (conjugate)")],
                  self.opts["prior"])
        dlg.number(box, "mu0", "Location parameter:", self.opts["mu0"], -1e9, 1e9, 4)
        dlg.number(box, "kappa0", "Scale (prior sample size kappa):", self.opts["kappa0"], 1e-6,
                   1e6, 4)
        dlg.number(box, "a0", "Inverse gamma shape:", self.opts["a0"], 1e-6, 1e6, 4)
        dlg.number(box, "b0", "Inverse gamma scale:", self.opts["b0"], 1e-6, 1e9, 4)
        values = dlg.run()
        if values:
            self.opts.update(values)

    def _bf(self):
        dlg = SubDialog(f"{self.windowTitle()}: Bayes Factor", self, columns=1)
        box = dlg.group("Rouder's JZS Method")
        dlg.number(box, "r", "Cauchy prior scale r (SPSS: 1; JASP: 0.707):", self.opts["r"],
                   0.01, 10, 4)
        values = dlg.run()
        if values:
            self.opts.update(values)

    def compute(self):
        o = self.opts
        return bs.one_sample_normal(self.ds, self.selected("vars"), o["test_value"],
                                    self.bopts(), o["r"], o["prior"],
                                    o["known_var"] if o["known"] else None, o["mu0"],
                                    o["kappa0"], o["a0"], o["b0"])


class BayesRelatedNormalDialog(BayesOneSampleNormalDialog):
    TEST_VALUE = None
    TITLE = "Bayesian Related Sample Inference: Normal"

    def build_targets(self):
        self.add_pair_target("pairs", "Paired Variables:")

    def compute(self):
        o = self.opts
        return bs.one_sample_normal(self.ds, [], 0.0, self.bopts(), o["r"], o["prior"],
                                    o["known_var"] if o["known"] else None, o["mu0"],
                                    o["kappa0"], o["a0"], o["b0"], paired=self.pairs("pairs"))


class BayesIndependentNormalDialog(GroupingMixin, _BayesDialog):
    DEFAULTS = {**_BayesDialog.BASE, "method": "rouder", "r": 1.0, "mu_delta": 0.0,
                "var_delta": 1.0, "hyper_a": -0.75, "equal": True, "group_def": None}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Bayesian Independent Sample Inference", parent)
        self.add_target("vars", "Test Variable(s):", accepts=is_numeric)
        self.setup_grouping(allow_cut=False)
        self.setup_common()
        self.add_option_button("Priors...", self._priors)
        self.add_option_button("Bayes Factor...", self._bf)
        self.finish()

    def _priors(self):
        dlg = SubDialog(f"{self.windowTitle()}: Prior Distribution", self, columns=1)
        box = dlg.group("Variances (posterior of the mean difference)")
        dlg.radio(box, "equal", [(True, "Assume equal variances"),
                                 (False, "Assume unequal variances")], self.opts["equal"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def _bf(self):
        dlg = SubDialog(f"{self.windowTitle()}: Bayes Factor", self, columns=1)
        box = dlg.group("Method")
        dlg.radio(box, "method", [("rouder", "Rouder method (JZS)"), ("gonen", "Gönen method"),
                                  ("hyper", "Hyper-prior method (Wang & Liu)")],
                  self.opts["method"])
        dlg.number(box, "r", "Rouder: Cauchy scale r (SPSS: 1):", self.opts["r"], 0.01, 10, 4)
        dlg.number(box, "mu_delta", "Gönen: effect size mean:", self.opts["mu_delta"], -10, 10,
                   4)
        dlg.number(box, "var_delta", "Gönen: effect size variance:", self.opts["var_delta"],
                   1e-6, 100, 4)
        dlg.number(box, "hyper_a", "Hyper-prior: a (SPSS: -0.75):", self.opts["hyper_a"],
                   -0.9999, -0.5, 4)
        values = dlg.run()
        if values:
            self.opts.update(values)

    def validate(self):
        return GroupingMixin.validate(self)

    def compute(self):
        o = self.opts
        g1, g2 = o["group_def"]["groups"]
        return bs.independent_normal(self.ds, self.selected("vars"), self.selected_one("group"),
                                     g1, g2, self.bopts(), o["method"], o["r"], o["mu_delta"],
                                     o["var_delta"], o["equal"], hyper_a=o["hyper_a"])


class BayesPearsonDialog(_BayesDialog):
    DEFAULTS = {**_BayesDialog.BASE, "kappa": 1.0, "method": "jzs"}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Bayesian Pearson Correlation", parent)
        self.add_target("vars", "Test Variables:", accepts=is_numeric, min_items=2)
        self.setup_common()
        self.add_option_button("Bayes Factor...", self._bf)
        self.finish()

    def _bf(self):
        dlg = SubDialog(f"{self.windowTitle()}: Bayes Factor", self, columns=1)
        box = dlg.group("Bayes Factor Method")
        dlg.radio(box, "method", [("jzs", "JZS prior (Wetzels & Wagenmakers) - SPSS"),
                                  ("stretched", "Stretched beta prior (Ly et al.) - JASP")],
                  self.opts["method"])
        dlg.number(box, "kappa", "Stretched beta: kappa (1 = uniform):", self.opts["kappa"],
                   0.01, 10, 4)
        values = dlg.run()
        if values:
            self.opts.update(values)

    def compute(self):
        return bs.pearson_correlation(self.ds, self.selected("vars"), self.bopts(),
                                      self.opts["kappa"], self.opts["method"])


_G_PRIORS = [("jzs", "JZS (Zellner-Siow) method"), ("zellner", "Zellner's method"),
             ("hyperg", "Hyper-prior (hyper-g) method")]


class BayesRegressionDialog(_BayesDialog):
    DEFAULTS = {**_BayesDialog.BASE, "prior": "jzs", "r": 1.0, "g": 0.0, "a": 3.0}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Bayesian Linear Regression Models", parent)
        self.add_target("dep", "Dependent:", accepts=is_numeric, single=True)
        self.add_target("fac", "Factor(s):", required=False)
        self.add_target("cov", "Covariate(s):", accepts=is_numeric, required=False)
        self.setup_common()
        self.add_option_button("Bayes Factor...", self._bf)
        self.finish()

    def _bf(self):
        dlg = SubDialog(f"{self.windowTitle()}: Bayes Factor", self, columns=1)
        box = dlg.group("Bayes Factor Method")
        dlg.radio(box, "prior", _G_PRIORS, self.opts["prior"])
        dlg.number(box, "r", "JZS scale r:", self.opts["r"], 0.01, 10, 4)
        dlg.number(box, "g", "Zellner g (0 = N):", self.opts["g"], 0, 1e9, 3)
        dlg.number(box, "a", "Hyper-g parameter a (2 < a <= 4):", self.opts["a"], 2.0001, 4, 3)
        values = dlg.run()
        if values:
            self.opts.update(values)

    def validate(self):
        if not self.selected("fac") and not self.selected("cov"):
            return "Select at least one factor or covariate."
        return None

    def compute(self):
        o = self.opts
        return bs.linear_regression(self.ds, self.selected_one("dep"), self.selected("fac"),
                                    self.selected("cov"), self.bopts(), o["prior"], o["r"],
                                    o["g"] or None, o["a"])


class BayesAnovaDialog(_BayesDialog):
    DEFAULTS = {**_BayesDialog.BASE, "method": "rouder", "r": 1.0, "g": 0.0, "a": 3.0}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Bayesian One-way ANOVA", parent)
        self.add_target("deps", "Dependent(s):", accepts=is_numeric)
        self.add_target("factor", "Factor:", single=True)
        self.setup_common()
        self.add_option_button("Bayes Factor...", self._bf)
        self.finish()

    def _bf(self):
        dlg = SubDialog(f"{self.windowTitle()}: Bayes Factor", self, columns=1)
        box = dlg.group("Bayes Factor Method")
        dlg.radio(box, "method", [("rouder", "Rouder's method (SPSS)")] + _G_PRIORS +
                  [("rouder2012", "Multivariate JZS (BayesFactor anovaBF, use scale 0.5)")],
                  self.opts["method"])
        dlg.number(box, "r", "Scale (Rouder s / JZS r):", self.opts["r"], 0.01, 10, 4)
        dlg.number(box, "g", "Zellner g (0 = N):", self.opts["g"], 0, 1e9, 3)
        dlg.number(box, "a", "Hyper-g parameter a:", self.opts["a"], 2.0001, 4, 3)
        values = dlg.run()
        if values:
            self.opts.update(values)

    def compute(self):
        o = self.opts
        return bs.oneway_anova(self.ds, self.selected("deps"), self.selected_one("factor"),
                               self.bopts(), o["method"], o["r"], o["g"] or None, o["a"])


class BayesRepeatedDialog(_BayesDialog):
    DEFAULTS = {**_BayesDialog.BASE, "r": 1.0, "pairwise": True}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Bayesian One-way Repeated Measures ANOVA", parent)
        self.add_target("vars", "Repeated Measures (levels):", accepts=is_numeric, min_items=2)
        self.setup_common()
        self.add_checkbox("pairwise", "Pairwise comparisons (JZS Bayes factors)")
        self.finish()

    def compute(self):
        return bs.repeated_measures_anova(self.ds, self.selected("vars"), self.bopts(),
                                          self.opts["r"], self.opts["pairwise"])


class BayesBinomialDialog(_BayesDialog):
    DEFAULTS = {**_BayesDialog.BASE, "test_value": 0.5, "success": "last", "success_value": "",
                "a": 1.0, "b": 1.0, "null_beta": False, "a0": 1.0, "b0": 1.0}
    TEST_VALUE = "Test Proportion:"

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Bayesian One Sample Inference: Binomial", parent)
        self.add_target("vars", "Test Variable(s):")
        self.setup_common()
        self.add_option_button("Success Categories...", self._success)
        self.add_option_button("Priors...", self._priors)
        self.finish()

    def _success(self):
        dlg = SubDialog(f"{self.windowTitle()}: Define Success", self, columns=1)
        box = dlg.group("Define Success for Categorical Fields")
        dlg.radio(box, "success", [("last", "Last category found"),
                                   ("value", "Use specified value"),
                                   ("cut", "Cut point (values <= cut point are successes)")],
                  self.opts["success"])
        edit = QLineEdit(str(self.opts["success_value"]))
        row = QFormLayout()
        row.addRow("Value / cut point:", edit)
        box.addLayout(row)
        values = dlg.run()
        if values:
            values["success_value"] = edit.text()
            self.opts.update(values)

    def _priors(self):
        dlg = SubDialog(f"{self.windowTitle()}: Prior Distribution", self, columns=1)
        box = dlg.group("Beta Prior (alternative hypothesis and posterior)")
        dlg.number(box, "a", "Shape parameter a:", self.opts["a"], 1e-6, 1e6, 4)
        dlg.number(box, "b", "Shape parameter b:", self.opts["b"], 1e-6, 1e6, 4)
        box = dlg.group("Null hypothesis prior (Bayes factor)")
        dlg.check(box, "null_beta", "Beta(a0, b0) prior under H0 (SPSS) instead of the point "
                  "null at the test proportion", self.opts["null_beta"])
        dlg.number(box, "a0", "Null shape a0:", self.opts["a0"], 1e-6, 1e6, 4)
        dlg.number(box, "b0", "Null shape b0:", self.opts["b0"], 1e-6, 1e6, 4)
        values = dlg.run()
        if values:
            self.opts.update(values)

    def validate(self):
        err = super().validate()
        if err:
            return err
        if not 0 < self.opts["test_value"] < 1:
            return "The test proportion must lie between 0 and 1."
        if self.opts["success"] != "last" and not str(self.opts["success_value"]).strip():
            return "Enter the success value or cut point."
        return None

    def _success_spec(self):
        o = self.opts
        if o["success"] == "last":
            return None
        if o["success"] == "cut":
            return ("le", _num(o["success_value"]))
        names = self.selected("vars")
        if names and self.ds.var(names[0]).type == NUMERIC:
            return _num(o["success_value"])
        return str(o["success_value"]).strip()

    def compute(self):
        return bs.one_sample_binomial(self.ds, self.selected("vars"), self._success_spec(),
                                      self.opts["test_value"], self.bopts(), self.opts["a"],
                                      self.opts["b"],
                                      null_a=self.opts["a0"] if self.opts["null_beta"] else None,
                                      null_b=self.opts["b0"] if self.opts["null_beta"] else None)


class BayesPoissonDialog(_BayesDialog):
    DEFAULTS = {**_BayesDialog.BASE, "test_value": 1.0, "shape": 2.0, "rate": 1.0}
    TEST_VALUE = "Test Rate:"

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Bayesian One Sample Inference: Poisson", parent)
        self.add_target("vars", "Test Variable(s):", accepts=is_numeric)
        self.setup_common()
        self.add_option_button("Priors...", self._priors)
        self.finish()

    def _priors(self):
        dlg = SubDialog(f"{self.windowTitle()}: Prior Distribution", self, columns=1)
        box = dlg.group("Gamma Prior")
        dlg.number(box, "shape", "Shape parameter:", self.opts["shape"], 1e-6, 1e6, 4)
        dlg.number(box, "rate", "Rate parameter:", self.opts["rate"], 1e-6, 1e6, 4)
        values = dlg.run()
        if values:
            self.opts.update(values)

    def validate(self):
        err = super().validate()
        if err:
            return err
        return "The test rate must be positive." if self.opts["test_value"] <= 0 else None

    def compute(self):
        return bs.one_sample_poisson(self.ds, self.selected("vars"), self.opts["test_value"],
                                     self.bopts(), self.opts["shape"], self.opts["rate"])


class BayesLoglinearDialog(_BayesDialog):
    DEFAULTS = {**_BayesDialog.BASE, "sampling": "joint", "a": 1.0}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Bayesian Log-Linear Regression Models", parent)
        self.add_target("row", "Row Variable:", single=True)
        self.add_target("col", "Column Variable:", single=True)
        self.setup_common()
        self.add_option_button("Bayes Factor...", self._bf)
        self.finish()

    def _bf(self):
        dlg = SubDialog(f"{self.windowTitle()}: Bayes Factor", self, columns=1)
        box = dlg.group("Sampling Model")
        dlg.radio(box, "sampling", [("joint", "Multinomial (total fixed)"),
                                    ("rows", "Independent multinomial: row totals fixed"),
                                    ("cols", "Independent multinomial: column totals fixed")],
                  self.opts["sampling"])
        dlg.number(box, "a", "Dirichlet prior parameter a:", self.opts["a"], 0.01, 100, 3)
        values = dlg.run()
        if values:
            self.opts.update(values)

    def validate(self):
        if self.selected_one("row") == self.selected_one("col"):
            return "Choose two different variables."
        return None

    def compute(self):
        return bs.loglinear(self.ds, self.selected_one("row"), self.selected_one("col"),
                            self.bopts(), self.opts["sampling"], self.opts["a"])


MENUS = {
    "Analyze/Bayesian Statistics": [
        ("One Sample Normal...", BayesOneSampleNormalDialog),
        ("One Sample Binomial...", BayesBinomialDialog),
        ("One Sample Poisson...", BayesPoissonDialog),
        ("Related Samples Normal...", BayesRelatedNormalDialog),
        ("Independent Samples Normal...", BayesIndependentNormalDialog),
        ("Pearson Correlation...", BayesPearsonDialog),
        ("Linear Regression...", BayesRegressionDialog),
        ("One-way ANOVA...", BayesAnovaDialog),
        ("One-way Repeated Measures ANOVA...", BayesRepeatedDialog),
        ("Log-Linear Models...", BayesLoglinearDialog),
    ],
}


def selftest() -> None:
    bs.selftest()

"""Smaller SPSS 28–32 procedures: Analyze > Descriptive Statistics > TURF Analysis,
Analyze > Forecasting > Vector Autoregression, Graphs > Weibull Plot, Analyze > Correlate >
Distance Correlation and the permutation tests (Compare Means and Regression). Core code:
:mod:`rsstatchem.core.stats.turf`, :mod:`~rsstatchem.core.stats.var_model`,
:mod:`~rsstatchem.core.stats.weibull_plot`, :mod:`~rsstatchem.core.stats.dcor`,
:mod:`~rsstatchem.core.stats.permutation`."""
from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QFormLayout, QLineEdit

from rsstatchem.core.stats import dcor, permutation, turf, var_model, weibull_plot
from rsstatchem.ui.dialogs.base import VariablePickerDialog, is_numeric
from rsstatchem.ui.dialogs.ml_tabs import OptionsDialog
from rsstatchem.ui.dialogs.wp11_util import parse_values


def _text_row(dialog, label: str, key: str, placeholder: str = "") -> QLineEdit:
    edit = QLineEdit(str(dialog.opts.get(key) or ""))
    edit.setPlaceholderText(placeholder)
    edit.textChanged.connect(lambda t: dialog.opts.__setitem__(key, t.strip()))
    row = QFormLayout()
    row.addRow(label, edit)
    dialog.options_layout.addLayout(row)
    return edit


class TurfDialog(VariablePickerDialog):
    DEFAULTS = {"positive": "1", "max_size": 3, "top": 10, "limit": 200000, "beam": 50,
                "chart": True, "detail": True}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "TURF Analysis", parent)
        self.add_target("vars", "Variables to Analyze:", min_items=2)
        _text_row(self, "Responses counted as positive:", "positive", "e.g. 1")
        self.add_option_button("Options...", self._options)
        self.finish()

    def _options(self):
        dlg = OptionsDialog("TURF Analysis: Options", self.opts, self, columns=1)
        g = dlg.group("Combinations")
        g.number("max_size", "Maximum variable combinations (group size):", 1, 50)
        g.number("top", "Number of combinations to display:", 1, 1000)
        g.number("limit", "Exhaustive search up to N combinations per group size:", 10, 1e8)
        g.number("beam", "Heuristic: combinations kept per group size:", 1, 10000)
        g = dlg.group("Output")
        g.check("chart", "Best reach by group size chart")
        g.check("detail", "Reach and frequency table for each group size")
        dlg.run(self.opts)

    def validate(self):
        if not parse_values(self.opts["positive"]):
            return "Enter at least one response value counted as positive."
        return None

    def compute(self):
        o = self.opts
        names = self.selected("vars")
        return turf.turf_analysis(self.ds, names, parse_values(o["positive"], self.ds, names[0]),
                                  int(o["max_size"]), int(o["top"]), int(o["limit"]),
                                  int(o["beam"]), o["chart"], o["detail"])


class VarDialog(VariablePickerDialog):
    DEFAULTS = {"auto": True, "lags": 1, "max_lags": 8, "criterion": "aic", "trend": "c",
                "granger": True, "irf": 10, "irf_plot": True, "fevd": True, "forecast": 0,
                "level": 95.0, "save_forecast": False, "save_resid": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Vector Autoregression", parent)
        self.add_target("vars", "Endogenous Series:", accepts=is_numeric, min_items=2)
        self.add_target("exog", "Exogenous Series (optional):", accepts=is_numeric,
                        required=False)
        self.add_option_button("Model...", self._model)
        self.add_option_button("Statistics...", self._stats)
        self.add_option_button("Save...", self._save)
        self.finish()

    def _model(self):
        dlg = OptionsDialog("Vector Autoregression: Model", self.opts, self, columns=1)
        g = dlg.group("Lag Order")
        g.radio("auto", [(True, "Select by information criterion"), (False, "Fixed lag order")])
        g.number("max_lags", "Maximum lags:", 1, 50)
        g.combo("criterion", "Criterion:", list(var_model.CRITERIA.items()))
        g.number("lags", "Fixed lag order:", 1, 50)
        g = dlg.group("Deterministic Terms")
        g.combo("trend", "Include:", list(var_model.TRENDS.items()))
        dlg.run(self.opts)

    def _stats(self):
        dlg = OptionsDialog("Vector Autoregression: Statistics", self.opts, self, columns=1)
        g = dlg.group("Statistics")
        g.check("granger", "Granger causality tests")
        g.number("irf", "Impulse response periods:", 0, 100)
        g.check("irf_plot", "Impulse response plots")
        g.check("fevd", "Forecast error variance decomposition")
        g.number("forecast", "Forecast horizon (0 = none):", 0, 1000)
        g.number("level", "Confidence level (%):", 50, 99.99, 2)
        dlg.run(self.opts)

    def _save(self):
        dlg = OptionsDialog("Vector Autoregression: Save", self.opts, self, columns=1)
        g = dlg.group("Save")
        g.check("save_forecast", "Fitted values (with a forecast horizon)")
        g.check("save_resid", "Residuals")
        dlg.run(self.opts)

    def validate(self):
        if set(self.selected("vars")) & set(self.selected("exog")):
            return "A series cannot be both endogenous and exogenous."
        return None

    def compute(self):
        o = self.opts
        return var_model.var_procedure(
            self.ds, self.selected("vars"), None if o["auto"] else int(o["lags"]),
            int(o["max_lags"]), o["criterion"], o["trend"], self.selected("exog") or None,
            o["granger"], int(o["irf"]), o["irf_plot"], o["fevd"], int(o["forecast"]),
            float(o["level"]) / 100, True, True, o["save_forecast"], o["save_resid"])


class WeibullPlotDialog(VariablePickerDialog):
    DEFAULTS = {"event": "1", "method": "rrx", "level": 95.0}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Weibull Plot", parent)
        self.add_target("var", "Variable (time to failure):", accepts=is_numeric, single=True)
        self.add_target("status", "Status (optional):", single=True, required=False)
        self.add_target("group", "Group by (optional):", single=True, required=False)
        _text_row(self, "Value(s) indicating failure:", "event", "e.g. 1")
        self.add_option_button("Options...", self._options)
        self.finish()

    def _options(self):
        dlg = OptionsDialog("Weibull Plot: Options", self.opts, self, columns=1)
        g = dlg.group("Fit")
        g.radio("method", [("rrx", "Rank regression on X (median ranks)"),
                           ("rry", "Rank regression on Y"), ("mle", "Maximum likelihood")])
        g.number("level", "Confidence level (%):", 50, 99.99, 2)
        dlg.run(self.opts)

    def compute(self):
        o = self.opts
        st = self.selected_one("status")
        return weibull_plot.weibull_plot(self.ds, self.selected_one("var"), st,
                                         parse_values(o["event"], self.ds, st) if st else None,
                                         self.selected_one("group"), o["method"],
                                         float(o["level"]) / 100)


class DistanceCorrelationDialog(VariablePickerDialog):
    DEFAULTS = {"mode": "pairs", "n_perm": 1000, "alpha": 0.05, "level": 95.0,
                "seed": 2000000, "missing": "pairwise"}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Distance Correlation", parent)
        top = QFormLayout()
        self.mode = QComboBox()
        self.mode.addItem("All pairs of the variables", "pairs")
        self.mode.addItem("X set with Y set (multivariate)", "sets")
        self.mode.setCurrentIndex(0 if self.opts["mode"] == "pairs" else 1)
        self.mode.currentIndexChanged.connect(
            lambda _i: self.opts.__setitem__("mode", self.mode.currentData()))
        top.addRow("Correlate:", self.mode)
        self.layout().insertLayout(0, top)
        self.add_target("vars", "Variables / X Set:", accepts=is_numeric)
        self.add_target("yset", "Y Set (multivariate mode):", accepts=is_numeric,
                        required=False)
        self.add_option_button("Options...", self._options)
        self.finish()

    def _options(self):
        dlg = OptionsDialog("Distance Correlation: Options", self.opts, self, columns=1)
        g = dlg.group("Inference")
        g.number("n_perm", "Number of permutations:", 10, 1e6)
        g.number("alpha", "Significance level:", 0.001, 0.5, 3)
        g.number("level", "Fisher confidence interval (%):", 50, 99.99, 2)
        g.number("seed", "Random seed:", 1, 2 ** 31 - 1)
        g = dlg.group("Missing Values")
        g.radio("missing", [("pairwise", "Exclude cases pairwise"),
                            ("listwise", "Exclude cases listwise")])
        dlg.run(self.opts)

    def validate(self):
        if self.opts["mode"] == "sets":
            if not self.selected("yset"):
                return "Select the Y set."
            if set(self.selected("vars")) & set(self.selected("yset")):
                return "A variable cannot be in both sets."
        elif len(self.selected("vars")) < 2:
            return "Select at least two variables."
        return None

    def compute(self):
        o = self.opts
        sets = o["mode"] == "sets"
        return dcor.dcor_procedure(self.ds, None if sets else self.selected("vars"),
                                   self.selected("vars") if sets else None,
                                   self.selected("yset") if sets else None, int(o["n_perm"]),
                                   float(o["alpha"]), float(o["level"]) / 100, int(o["seed"]),
                                   o["missing"])


class PermutationTestDialog(VariablePickerDialog):
    DEFAULTS = {"test": "independent", "groups": "", "statistic": "mean", "method": "pearson",
                "n_perm": 10000, "max_exact": 100000, "seed": 2000000, "plot": True}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Permutation Test", parent)
        top = QFormLayout()
        self.test = QComboBox()
        for key, text in (("independent", "Independent samples"),
                          ("paired", "Paired samples"), ("oneway", "One-way (k groups)"),
                          ("correlation", "Correlation")):
            self.test.addItem(text, key)
        self.test.setCurrentIndex(["independent", "paired", "oneway", "correlation"].index(
            self.opts["test"]))
        self.test.currentIndexChanged.connect(
            lambda _i: self.opts.__setitem__("test", self.test.currentData()))
        top.addRow("Test:", self.test)
        self.layout().insertLayout(0, top)
        self.add_target("vars", "Test Variables:", accepts=is_numeric, required=False)
        self.add_target("group", "Grouping Variable / Factor:", single=True, required=False)
        self.add_pair_target("pairs", "Paired Variables:")
        _text_row(self, "Group values (two, independent samples):", "groups", "e.g. 1 2")
        self.add_option_button("Options...", self._options)
        self.finish()

    def _update_ok(self):
        if hasattr(self, "ok_button"):
            self.ok_button.setEnabled(True)

    def _options(self):
        dlg = OptionsDialog("Permutation Test: Options", self.opts, self, columns=1)
        g = dlg.group("Statistic")
        g.radio("statistic", [("mean", "Mean difference"), ("welch", "Welch t"),
                              ("pooled", "t (equal variances); paired t for pairs")])
        g.radio("method", [("pearson", "Pearson correlation"),
                           ("spearman", "Spearman correlation")])
        g = dlg.group("Permutations")
        g.number("n_perm", "Number of random permutations:", 100, 1e7)
        g.number("max_exact", "Exact when at most N permutations:", 0, 1e7)
        g.number("seed", "Random seed:", 1, 2 ** 31 - 1)
        g.check("plot", "Permutation distribution charts")
        dlg.run(self.opts)

    def validate(self):
        t = self.opts["test"]
        if t == "paired":
            return None if self.pairs("pairs") else "Select at least one pair of variables."
        if t == "correlation":
            return None if len(self.selected("vars")) >= 2 else \
                "Select at least two test variables."
        if not self.selected("vars") or not self.selected_one("group"):
            return "Select test variables and a grouping variable."
        if t == "independent" and self.opts["groups"]:
            g = parse_values(self.opts["groups"], self.ds, self.selected_one("group"))
            if len(g) != 2:
                return "Enter exactly two group values."
        return None

    def compute(self):
        o = self.opts
        t = o["test"]
        grp = self.selected_one("group")
        groups = tuple(parse_values(o["groups"], self.ds, grp)) if o["groups"] and grp else None
        stat = o["statistic"]
        if t == "paired":
            stat = "mean" if stat == "mean" else "t"
        return permutation.permutation_procedure(
            self.ds, t, self.selected("vars"), grp, groups, self.pairs("pairs"), stat,
            o["method"], int(o["n_perm"]), int(o["max_exact"]), int(o["seed"]), o["plot"])


class RegressionPermutationDialog(VariablePickerDialog):
    DEFAULTS = {"n_perm": 5000, "seed": 2000000}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Linear Regression: Permutation Test", parent)
        self.add_target("dep", "Dependent:", accepts=is_numeric, single=True)
        self.add_target("pred", "Independent(s):", accepts=is_numeric)
        self.add_option_button("Options...", self._options)
        self.finish()

    def _options(self):
        dlg = OptionsDialog("Permutation Test: Options", self.opts, self, columns=1)
        g = dlg.group("Permutations")
        g.number("n_perm", "Number of permutations:", 100, 1e7)
        g.number("seed", "Random seed:", 1, 2 ** 31 - 1)
        dlg.run(self.opts)

    def compute(self):
        return permutation.regression_permutation(self.ds, self.selected_one("dep"),
                                                  self.selected("pred"),
                                                  int(self.opts["n_perm"]),
                                                  int(self.opts["seed"]))


MENUS = {
    "Analyze/Descriptive Statistics": [("TURF Analysis...", TurfDialog)],
    "Analyze/Forecasting": [("Vector Autoregression...", VarDialog)],
    "Analyze/Correlate": [("Distance Correlation...", DistanceCorrelationDialog)],
    "Analyze/Compare Means": [("Permutation Test...", PermutationTestDialog)],
    "Analyze/Regression": [("Permutation Test...", RegressionPermutationDialog)],
    "Graphs": [("Weibull Plot...", WeibullPlotDialog)],
}


def selftest() -> None:
    for m in (turf, var_model, weibull_plot, dcor, permutation):
        m.selftest()

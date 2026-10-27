"""Analyze > Meta Analysis dialogs (SPSS 28+): Continuous Outcomes and Binary Outcomes (raw data
or pre-calculated effect sizes) and Meta-Regression. Core code:
:mod:`rsstatchem.core.stats.meta`."""
from __future__ import annotations

from rsstatchem.core.stats import meta as mt
from rsstatchem.ui.dialogs.base import SubDialog, VariablePickerDialog, is_numeric


class _MetaDialog(VariablePickerDialog):
    MODE = "cont_raw"
    TITLE = "Meta-Analysis"
    ROLES: list[tuple[str, str]] = []
    DEFAULTS = {"measure": "hedges", "model": "random", "method": "REML", "knapp": "none",
                "level": 95.0, "zero": "only", "mh": False, "se_kind": "se",
                "individual": True, "homogeneity": True, "prediction": True, "exp": True,
                "cumulative": False, "loo": False, "egger": False, "begg": False,
                "trimfill": False, "tf_estimator": "L0", "tf_side": "auto",
                "plots": ["forest"], "forest_sort": "none", "save": []}

    def __init__(self, ds, parent=None):
        super().__init__(ds, self.TITLE, parent)
        self.setMinimumSize(640, 560)
        self.add_target("study", "Study ID:", single=True, required=False)
        self.add_target("label", "Study Label:", single=True, required=False)
        for key, text in self.ROLES:
            self.add_target(key, text, accepts=is_numeric, single=True)
        self.add_target("group", "Subgroup:", single=True, required=False)
        self.add_target("order", "Cumulative / Sort by:", single=True, required=False)
        self.add_option_button("Analysis...", self._analysis)
        self.add_option_button("Inference...", self._inference)
        self.add_option_button("Bias...", self._bias)
        self.add_option_button("Print...", self._print)
        self.add_option_button("Plot...", self._plot)
        self.add_option_button("Save...", self._save)
        self.finish()

    # ------------------------------------------------------------------ sub-dialogs
    def measure_items(self) -> list[tuple[str, str]]:
        return []

    def _analysis(self):
        dlg = SubDialog(f"{self.TITLE}: Analysis", self)
        items = self.measure_items()
        if items:
            box = dlg.group("Effect Size")
            dlg.radio(box, "measure", items, self.opts["measure"])
        box = dlg.group("Model")
        dlg.radio(box, "model", [("random", "Random-effects"), ("fixed", "Fixed-effects")],
                  self.opts["model"])
        if self.MODE == "bin_raw":
            dlg.check(box, "mh", "Mantel-Haenszel method (fixed effects; OR, RR, RD)",
                      self.opts["mh"])
            box = dlg.group("Zero Cells")
            dlg.radio(box, "zero", [("only", "Add 0.5 to studies with a zero cell"),
                                    ("all", "Add 0.5 to all studies"), ("none", "None")],
                      self.opts["zero"])
        if self.MODE.endswith("_es"):
            box = dlg.group("Standard Error Variable Contains")
            dlg.radio(box, "se_kind", [("se", "Standard errors"), ("var", "Variances")],
                      self.opts["se_kind"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def _inference(self):
        dlg = SubDialog(f"{self.TITLE}: Inference", self)
        box = dlg.group("Estimator (random effects)")
        dlg.radio(box, "method", [(k, v) for k, v in mt.ESTIMATORS.items()],
                  self.opts["method"])
        box = dlg.group("Standard Error Adjustment")
        dlg.radio(box, "knapp", [("none", "None"), ("untruncated", "Knapp-Hartung"),
                                 ("truncated", "Truncated Knapp-Hartung")], self.opts["knapp"])
        dlg.number(box, "level", "Confidence interval (%):", self.opts["level"], 50, 99.99, 2)
        values = dlg.run()
        if values:
            self.opts.update(values)

    def _bias(self):
        dlg = SubDialog(f"{self.TITLE}: Bias", self)
        box = dlg.group("Egger's Regression-Based Test")
        dlg.check(box, "egger", "Egger's test", self.opts["egger"])
        dlg.check(box, "begg", "Begg's rank correlation test", self.opts["begg"])
        box = dlg.group("Trim-and-Fill")
        dlg.check(box, "trimfill", "Trim-and-fill analysis", self.opts["trimfill"])
        dlg.radio(box, "tf_estimator", [("L0", "Linear (L0)"), ("R0", "Run (R0)")],
                  self.opts["tf_estimator"])
        dlg.radio(box, "tf_side", [("auto", "Side: estimated from Egger's test"),
                                   ("left", "Impute on the left"),
                                   ("right", "Impute on the right")], self.opts["tf_side"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def _print(self):
        dlg = SubDialog(f"{self.TITLE}: Print", self)
        box = dlg.group("Tables")
        dlg.check(box, "individual", "Effect size estimates for individual studies",
                  self.opts["individual"])
        dlg.check(box, "homogeneity", "Heterogeneity and homogeneity test",
                  self.opts["homogeneity"])
        dlg.check(box, "prediction", "Prediction interval", self.opts["prediction"])
        dlg.check(box, "exp", "Exponentiated estimates (log effect sizes)", self.opts["exp"])
        box = dlg.group("Sensitivity")
        dlg.check(box, "cumulative", "Cumulative analysis (by the Cumulative variable)",
                  self.opts["cumulative"])
        dlg.check(box, "loo", "Leave-one-out analysis", self.opts["loo"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def _plot(self):
        dlg = SubDialog(f"{self.TITLE}: Plot", self)
        box = dlg.group("Plots")
        plots = [("forest", "Forest plot"), ("funnel", "Funnel plot"),
                 ("galbraith", "Galbraith plot")]
        if self.MODE == "bin_raw":
            plots.append(("labbe", "L'Abbé plot"))
        for key, text in plots:
            dlg.check(box, f"p::{key}", text, key in self.opts["plots"])
        box = dlg.group("Forest Plot Order")
        dlg.radio(box, "forest_sort", [("none", "Data order"), ("asc", "Ascending effect size"),
                                       ("desc", "Descending effect size")],
                  self.opts["forest_sort"])
        values = dlg.run()
        if values:
            self.opts["plots"] = [k[3:] for k, v in values.items() if k.startswith("p::") and v]
            self.opts["forest_sort"] = values["forest_sort"]

    def _save(self):
        dlg = SubDialog(f"{self.TITLE}: Save", self, columns=1)
        box = dlg.group("Save to the active dataset")
        for key, text in (("ES", "Individual effect size"), ("SE", "Standard error"),
                          ("LCL", "Lower confidence limit"), ("UCL", "Upper confidence limit"),
                          ("WT", "Weight (%)")):
            dlg.check(box, f"s::{key}", text, key in self.opts["save"])
        values = dlg.run()
        if values:
            self.opts["save"] = [k[3:] for k, v in values.items() if v]

    # ------------------------------------------------------------------ run
    def variables(self) -> dict:
        return {key: self.selected_one(key) for key, _ in self.ROLES}

    def compute(self):
        o = self.opts
        opts = mt.MetaOptions(model=o["model"], method=o["method"], knapp=o["knapp"],
                              level=o["level"] / 100, mh=o["mh"], individual=o["individual"],
                              homogeneity=o["homogeneity"], prediction=o["prediction"],
                              exp=o["exp"], cumulative=o["cumulative"],
                              leave_one_out=o["loo"], egger=o["egger"], begg=o["begg"],
                              trimfill=o["trimfill"], tf_estimator=o["tf_estimator"],
                              tf_side=o["tf_side"], plots=tuple(o["plots"]),
                              forest_sort=o["forest_sort"], save=tuple(o["save"]))
        return mt.meta_analysis(self.ds, self.MODE, self.variables(), self.measure(), opts,
                                self.selected_one("study"), self.selected_one("label"),
                                self.selected_one("group"), self.selected_one("order"),
                                o["zero"], o["se_kind"])

    def measure(self) -> str:
        return self.opts["measure"]


class MetaContinuousRawDialog(_MetaDialog):
    MODE = "cont_raw"
    TITLE = "Meta-Analysis Continuous Outcomes: Raw Data"
    ROLES = [("n1", "Treatment Group: Study Size:"), ("m1", "Treatment Group: Mean:"),
             ("s1", "Treatment Group: Std. Deviation:"), ("n2", "Control Group: Study Size:"),
             ("m2", "Control Group: Mean:"), ("s2", "Control Group: Std. Deviation:")]

    def measure_items(self):
        return list(mt.CONT_MEASURES.items())

    def measure(self):
        m = self.opts["measure"]
        return m if m in mt.CONT_MEASURES else "hedges"


class MetaContinuousESDialog(_MetaDialog):
    MODE = "cont_es"
    TITLE = "Meta-Analysis Continuous Outcomes: Pre-Calculated Effect Size"
    ROLES = [("es", "Effect Size:"), ("se", "Standard Error:")]

    def measure(self):
        return "es"


class MetaBinaryRawDialog(_MetaDialog):
    MODE = "bin_raw"
    TITLE = "Meta-Analysis Binary Outcomes: Raw Data"
    ROLES = [("a", "Treatment Group: Events (Yes):"), ("b", "Treatment Group: Non-events (No):"),
             ("c", "Control Group: Events (Yes):"), ("d", "Control Group: Non-events (No):")]
    DEFAULTS = {**_MetaDialog.DEFAULTS, "measure": "or"}

    def measure_items(self):
        return list(mt.BIN_MEASURES.items())

    def measure(self):
        m = self.opts["measure"]
        return m if m in mt.BIN_MEASURES else "or"


class MetaBinaryESDialog(_MetaDialog):
    MODE = "bin_es"
    TITLE = "Meta-Analysis Binary Outcomes: Pre-Calculated Effect Size"
    ROLES = [("es", "Effect Size (log scale for OR/RR):"), ("se", "Standard Error:")]
    DEFAULTS = {**_MetaDialog.DEFAULTS, "measure": "or"}

    def measure_items(self):
        return [("or", "Log odds ratio"), ("rr", "Log risk ratio"), ("rd", "Risk difference"),
                ("peto", "Log Peto odds ratio")]

    def measure(self):
        m = self.opts["measure"]
        return m if m in mt.BIN_MEASURES else "or"


class MetaRegressionDialog(VariablePickerDialog):
    DEFAULTS = {"model": "random", "method": "REML", "knapp": "none", "level": 95.0,
                "se_kind": "se", "bubble": True}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Meta-Regression", parent)
        self.add_target("es", "Effect Size:", accepts=is_numeric, single=True)
        self.add_target("se", "Standard Error:", accepts=is_numeric, single=True)
        self.add_target("cov", "Covariate(s):", accepts=is_numeric, required=False)
        self.add_target("fac", "Factor(s):", required=False)
        self.add_option_button("Analysis...", self._analysis)
        self.add_option_button("Plot...", self._plot)
        self.finish()

    def _analysis(self):
        dlg = SubDialog("Meta-Regression: Analysis", self)
        box = dlg.group("Model")
        dlg.radio(box, "model", [("random", "Mixed-effects (random)"), ("fixed",
                                                                      "Fixed-effects")],
                  self.opts["model"])
        dlg.radio(box, "se_kind", [("se", "Standard errors"), ("var", "Variances")],
                  self.opts["se_kind"])
        box = dlg.group("Estimator")
        dlg.radio(box, "method", list(mt.ESTIMATORS.items()), self.opts["method"])
        box = dlg.group("Inference")
        dlg.radio(box, "knapp", [("none", "None"), ("untruncated", "Knapp-Hartung"),
                                 ("truncated", "Truncated Knapp-Hartung")], self.opts["knapp"])
        dlg.number(box, "level", "Confidence interval (%):", self.opts["level"], 50, 99.99, 2)
        values = dlg.run()
        if values:
            self.opts.update(values)

    def _plot(self):
        dlg = SubDialog("Meta-Regression: Plot", self, columns=1)
        box = dlg.group("Plot")
        dlg.check(box, "bubble", "Bubble plot (first covariate)", self.opts["bubble"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def validate(self):
        if not self.selected("cov") and not self.selected("fac"):
            return "Select at least one covariate or factor."
        return None

    def compute(self):
        o = self.opts
        return mt.meta_regression(self.ds, self.selected_one("es"), self.selected_one("se"),
                                  self.selected("cov"), self.selected("fac"), o["model"],
                                  o["method"], o["knapp"], o["level"] / 100, o["se_kind"],
                                  o["bubble"])


MENUS = {
    "Analyze/Meta Analysis": [
        ("Continuous Outcomes", [("Raw Data...", MetaContinuousRawDialog),
                                 ("Pre-Calculated Effect Size...", MetaContinuousESDialog)]),
        ("Binary Outcomes", [("Raw Data...", MetaBinaryRawDialog),
                             ("Pre-Calculated Effect Size...", MetaBinaryESDialog)]),
        ("Meta-Regression...", MetaRegressionDialog),
    ],
}


def selftest() -> None:
    mt.selftest()

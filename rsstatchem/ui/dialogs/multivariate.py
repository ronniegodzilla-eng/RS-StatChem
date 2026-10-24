"""Analyze > Dimension Reduction > Factor (SPSS layout) and Analyze > Mediation & Moderation
(PROCESS-style Models 1, 4, 7 and 14) dialogs. Menus are exported from
:mod:`rsstatchem.ui.dialogs.glm`.
"""
from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QFormLayout, QLabel

from rsstatchem.core.stats import factor, process
from rsstatchem.ui.dialogs.base import SubDialog, VariablePickerDialog, is_numeric


# ------------------------------------------------------------------ Factor
class FactorDialog(VariablePickerDialog):
    DEFAULTS = {"univariate": False, "initial": True, "coefficients": False,
                "significance": False, "determinant": False, "kmo": True, "inverse": False,
                "reproduced": False, "anti_image": False, "extraction": "pc",
                "unrotated": True, "scree": True, "criterion": "eigen", "min_eigen": 1.0,
                "n_factors": 2, "max_iter": 25, "rotation": "none", "delta": 0.0,
                "kappa": 4.0, "rot_max_iter": 25, "loading_plots": False, "save": False,
                "save_method": "regression", "score_coefficients": False, "sort": False,
                "suppress": False, "suppress_value": 0.10}
    modifies_data = False

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Factor Analysis", parent)
        self.add_target("vars", "Variables:", accepts=is_numeric, min_items=2)
        self.add_option_button("Descriptives...", self._descriptives)
        self.add_option_button("Extraction...", self._extraction)
        self.add_option_button("Rotation...", self._rotation)
        self.add_option_button("Scores...", self._scores)
        self.add_option_button("Options...", self._options)
        self.finish()

    def _apply(self, values) -> None:
        if values:
            self.opts.update(values)

    def _descriptives(self) -> None:
        o = self.opts
        dlg = SubDialog("Factor Analysis: Descriptives", self, columns=1)
        box = dlg.group("Statistics")
        dlg.check(box, "univariate", "Univariate descriptives", o["univariate"])
        dlg.check(box, "initial", "Initial solution", o["initial"])
        box = dlg.group("Correlation Matrix")
        for key, text in (("coefficients", "Coefficients"),
                          ("significance", "Significance levels"),
                          ("determinant", "Determinant"),
                          ("kmo", "KMO and Bartlett's test of sphericity"),
                          ("inverse", "Inverse"), ("reproduced", "Reproduced"),
                          ("anti_image", "Anti-image")):
            dlg.check(box, key, text, o[key])
        self._apply(dlg.run())

    def _extraction(self) -> None:
        o = self.opts
        dlg = SubDialog("Factor Analysis: Extraction", self)
        box = dlg.group("Method")
        dlg.radio(box, "extraction", [("pc", "Principal components"),
                                      ("paf", "Principal axis factoring"),
                                      ("ml", "Maximum likelihood")], o["extraction"])
        box.addWidget(QLabel("Analyze: Correlation matrix"))
        box = dlg.group("Display")
        dlg.check(box, "unrotated", "Unrotated factor solution", o["unrotated"])
        dlg.check(box, "scree", "Scree plot", o["scree"])
        box = dlg.group("Extract")
        dlg.radio(box, "criterion", [("eigen", "Based on Eigenvalue"),
                                     ("fixed", "Fixed number of factors")], o["criterion"])
        dlg.number(box, "min_eigen", "Eigenvalues greater than:", o["min_eigen"], 0, 100, 2)
        dlg.number(box, "n_factors", "Factors to extract:", o["n_factors"], 1, 99, 0)
        box = dlg.group("Convergence")
        dlg.number(box, "max_iter", "Maximum Iterations for Convergence:", o["max_iter"],
                   1, 9999, 0)
        values = dlg.run()
        if values:
            values["n_factors"] = int(values["n_factors"])
            values["max_iter"] = int(values["max_iter"])
        self._apply(values)

    def _rotation(self) -> None:
        o = self.opts
        dlg = SubDialog("Factor Analysis: Rotation", self)
        box = dlg.group("Method")
        dlg.radio(box, "rotation", [("none", "None"), ("quartimax", "Quartimax"),
                                    ("varimax", "Varimax"), ("equamax", "Equamax"),
                                    ("oblimin", "Direct Oblimin"), ("promax", "Promax")],
                  o["rotation"])
        dlg.number(box, "delta", "Delta (Direct Oblimin):", o["delta"], -100, 0.8, 2)
        dlg.number(box, "kappa", "Kappa (Promax):", o["kappa"], 1, 100, 0)
        box = dlg.group("Display")
        dlg.check(box, "loading_plots", "Loading plot(s)", o["loading_plots"])
        dlg.number(box, "rot_max_iter", "Maximum Iterations for Convergence:",
                   o["rot_max_iter"], 1, 9999, 0)
        values = dlg.run()
        if values:
            values["rot_max_iter"] = int(values["rot_max_iter"])
        self._apply(values)

    def _scores(self) -> None:
        o = self.opts
        dlg = SubDialog("Factor Analysis: Factor Scores", self, columns=1)
        box = dlg.group("Save")
        dlg.check(box, "save", "Save as variables", o["save"])
        dlg.radio(box, "save_method", [("regression", "Regression"), ("bartlett", "Bartlett"),
                                       ("anderson_rubin", "Anderson-Rubin")], o["save_method"])
        dlg.check(dlg.group("Display"), "score_coefficients",
                  "Display factor score coefficient matrix", o["score_coefficients"])
        self._apply(dlg.run())

    def _options(self) -> None:
        o = self.opts
        dlg = SubDialog("Factor Analysis: Options", self, columns=1)
        box = dlg.group("Missing Values")
        box.addWidget(QLabel("Exclude cases listwise"))
        box = dlg.group("Coefficient Display Format")
        dlg.check(box, "sort", "Sorted by size", o["sort"])
        dlg.check(box, "suppress", "Suppress small coefficients", o["suppress"])
        dlg.number(box, "suppress_value", "Absolute value below:", o["suppress_value"], 0, 1, 2)
        self._apply(dlg.run())

    def compute(self):
        o = self.opts
        return factor.factor_analysis(
            self.ds, self.selected("vars"), o["extraction"], o["criterion"], o["min_eigen"],
            o["n_factors"], o["rotation"], o["delta"], o["kappa"], o["max_iter"],
            o["rot_max_iter"], o["univariate"], o["initial"], o["coefficients"],
            o["significance"], o["determinant"], o["kmo"], o["inverse"], o["reproduced"],
            o["anti_image"], o["unrotated"], o["scree"], o["loading_plots"], o["sort"],
            o["suppress_value"] if o["suppress"] else None,
            o["save_method"] if o["save"] else None, o["score_coefficients"])


# ------------------------------------------------------------------ PROCESS
MODEL_TEXT = {
    1: "Model 1: X → Y moderated by W (simple moderation)",
    4: "Model 4: X → M → Y (simple or parallel multiple mediation)",
    7: "Model 7: X → M → Y with W moderating X → M (first-stage moderated mediation)",
    14: "Model 14: X → M → Y with W moderating M → Y (second-stage moderated mediation)",
}


class ProcessDialog(VariablePickerDialog):
    MODEL = 4
    DEFAULTS = {"boot": 5000, "seed": 0, "level": 95.0, "center": False, "moments": False,
                "jn": True, "total": False, "effsize": True, "contrast": False,
                "normal": False, "hc": "none", "plot": True}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "PROCESS (Mediation and Moderation)", parent)
        self.add_target("y", "Y variable:", accepts=is_numeric, single=True)
        self.add_target("x", "X variable:", accepts=is_numeric, single=True)
        self.add_target("m", "Mediator(s) M (Models 4, 7, 14):", accepts=is_numeric,
                        required=False)
        self.add_target("w", "Moderator variable W (Models 1, 7, 14):", accepts=is_numeric,
                        single=True, required=False)
        self.add_target("cov", "Covariate(s):", accepts=is_numeric, required=False)
        self.add_option_button("Options...", self._options)
        self.model = QComboBox()
        for k, text in MODEL_TEXT.items():
            self.model.addItem(text, k)
        self.model.setCurrentIndex(max(self.model.findData(self.opts.get("model",
                                                                          self.MODEL)), 0))
        self.model.currentIndexChanged.connect(
            lambda _: self.opts.__setitem__("model", self.model.currentData()))
        self.opts.setdefault("model", self.MODEL)
        form = QFormLayout()
        form.addRow("Model number:", self.model)
        self.options_layout.addLayout(form)
        self.finish()

    def _options(self) -> None:
        o = self.opts
        dlg = SubDialog("PROCESS: Options", self)
        box = dlg.group("Bootstrap")
        dlg.number(box, "boot", "Number of bootstrap samples:", o["boot"], 0, 100000, 0)
        dlg.number(box, "seed", "Random seed (0 = random):", o["seed"], 0, 2147483646, 0)
        dlg.number(box, "level", "Confidence level (%):", o["level"], 50, 99.99, 2)
        box = dlg.group("Moderation")
        dlg.check(box, "center", "Mean center variables that define products", o["center"])
        dlg.radio(box, "moments", [(False, "Conditioning values: 16th, 50th, 84th percentiles"),
                                   (True, "Conditioning values: −1SD, Mean, +1SD")],
                  o["moments"])
        dlg.check(box, "jn", "Johnson-Neyman output (Model 1)", o["jn"])
        dlg.check(box, "plot", "Interaction plot (Model 1)", o["plot"])
        box = dlg.group("Mediation")
        dlg.check(box, "total", "Show total effect model (Model 4)", o["total"])
        dlg.check(box, "effsize", "Effect size (completely standardized)", o["effsize"])
        dlg.check(box, "contrast", "Pairwise contrasts of indirect effects", o["contrast"])
        dlg.check(box, "normal", "Sobel test (normal theory)", o["normal"])
        box = dlg.group("Standard errors")
        dlg.radio(box, "hc", [("none", "OLS (homoscedasticity assumed)"), ("HC0", "HC0"),
                              ("HC1", "HC1"), ("HC2", "HC2"), ("HC3", "HC3"), ("HC4", "HC4")],
                  o["hc"])
        values = dlg.run()
        if values:
            values["boot"] = int(values["boot"])
            values["seed"] = int(values["seed"])
            self.opts.update(values)

    def validate(self):
        model = self.model.currentData()
        names = [self.selected_one("y"), self.selected_one("x")]
        if model in (4, 7, 14) and not self.selected("m"):
            return f"Model {model} needs at least one mediator (M)."
        if model in (1, 7, 14) and not self.selected_one("w"):
            return f"Model {model} needs a moderator (W)."
        if model == 1 and self.selected("m"):
            return "Model 1 has no mediators; remove the M variable(s)."
        if model == 4 and self.selected_one("w"):
            return "Model 4 has no moderator; remove W or choose Model 7 or 14."
        if None in names:
            return "Specify Y and X."
        return None

    def compute(self):
        o = self.opts
        model = self.model.currentData()
        return process.process(
            self.ds, model, self.selected_one("y"), self.selected_one("x"),
            self.selected("m"), self.selected_one("w"), self.selected("cov"),
            o["boot"], o["seed"] or None, o["level"] / 100, o["center"], o["moments"], o["jn"],
            o["total"], o["effsize"], o["contrast"], o["normal"],
            None if o["hc"] == "none" else o["hc"], o["plot"])


class ProcessModel4Dialog(ProcessDialog):
    MODEL = 4


class ProcessModel1Dialog(ProcessDialog):
    MODEL = 1


class ProcessModel7Dialog(ProcessDialog):
    MODEL = 7


class ProcessModel14Dialog(ProcessDialog):
    MODEL = 14

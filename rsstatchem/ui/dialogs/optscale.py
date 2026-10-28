"""SPSS Categories dialogs: Analyze > Regression > Optimal Scaling (CATREG)...,
Analyze > Dimension Reduction > Optimal Scaling... (the chooser that opens Multiple
Correspondence Analysis, Categorical Principal Components or Nonlinear Canonical Correlation)
and Analyze > Scale > Multidimensional Unfolding (PREFSCAL)....

Variables in the target lists carry their optimal scaling level as SPSS shows it, e.g.
``income (Spline Ordinal 2 2)``; select one or more and press "Define Scale...". Core code:
:mod:`rsstatchem.core.stats.optscale`.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QButtonGroup, QDialog, QDialogButtonBox, QGroupBox, QHBoxLayout,
                               QLabel, QPushButton, QRadioButton, QSpinBox, QVBoxLayout)

from rsstatchem.core.stats import optscale as osc
from rsstatchem.ui.dialogs.base import NAME_ROLE, SubDialog, VariablePickerDialog

SHORT = {"spord": "Spline Ordinal", "spnom": "Spline Nominal", "mnom": "Multiple Nominal",
         "ordi": "Ordinal", "nomi": "Nominal", "nume": "Numeric"}


class _ScaleMixin:
    """Per-variable optimal scaling levels / weights shown as item suffixes."""
    LEVEL_CHOICES = ("spord", "spnom", "ordi", "nomi", "nume")
    DEFAULT_LEVEL = "spord"
    SHOW_WEIGHT = False

    def _spec(self, name: str) -> dict:
        d = dict(self.opts.setdefault("scale", {}).get(name, {}))
        d.setdefault("level", self.DEFAULT_LEVEL)
        d.setdefault("degree", 2)
        d.setdefault("knots", 2)
        d.setdefault("weight", 1.0)
        disc = self.opts.setdefault("disc", {}).get(name, {})
        d["disc"] = disc.get("method", "auto")
        d["ncat"] = int(disc.get("ncat", 7))
        d["interval"] = float(disc.get("interval", 1.0))
        d["name"] = name
        return d

    def _suffix(self, name: str) -> str:
        s = self._spec(name)
        t = SHORT[s["level"]]
        if s["level"] in ("spord", "spnom"):
            t += f" {s['degree']} {s['knots']}"
        if self.SHOW_WEIGHT:
            t = f"{s['weight']:g} {t}"
        return f" ({t})"

    def _refresh(self, key: str) -> None:
        widget = next(t.widget for t in self._targets if t.key == key)
        for i in range(widget.count()):
            item = widget.item(i)
            name = item.data(NAME_ROLE)
            item.setText(name + self._suffix(name))

    def _define_scale(self, key: str, title: str) -> None:
        widget = next(t.widget for t in self._targets if t.key == key)
        items = widget.selectedItems() or [widget.item(i) for i in range(widget.count())]
        names = [it.data(NAME_ROLE) for it in items]
        if not names:
            return
        cur = self._spec(names[0])
        dlg = SubDialog(title, self, columns=1)
        if self.SHOW_WEIGHT:
            box = dlg.group("Variable weight")
            dlg.number(box, "weight", "Weight:", cur["weight"], 1, 99, 0)
        box = dlg.group("Optimal Scaling Level")
        dlg.radio(box, "level", [(k, SHORT[k]) for k in self.LEVEL_CHOICES], cur["level"])
        box = dlg.group("Spline")
        dlg.number(box, "degree", "Degree:", cur["degree"], 1, 4, 0)
        dlg.number(box, "knots", "Interior Knots:", cur["knots"], 0, 9, 0)
        values = dlg.run()
        if values:
            for n in names:
                s = self.opts.setdefault("scale", {}).setdefault(n, {})
                s["level"] = values["level"]
                s["degree"] = int(values["degree"])
                s["knots"] = int(values["knots"])
                if "weight" in values:
                    s["weight"] = float(values["weight"])
            self._refresh(key)

    def _discretize(self, names: list[str]) -> None:
        if not names:
            return
        dlg = SubDialog(f"{self.windowTitle()}: Discretization", self, columns=1)
        box = dlg.group(f"Method for {', '.join(names[:4])}{'...' if len(names) > 4 else ''}")
        cur = self.opts.setdefault("disc", {}).get(names[0], {})
        dlg.radio(box, "method", list(osc.DISCRETIZATION.items()), cur.get("method", "auto"))
        box = dlg.group("Grouping")
        dlg.number(box, "ncat", "Number of categories:", cur.get("ncat", 7), 2, 99, 0)
        dlg.number(box, "interval", "Equal intervals (size):", cur.get("interval", 1.0),
                   0.0001, 1e9, 4)
        values = dlg.run()
        if values:
            for n in names:
                self.opts["disc"][n] = {"method": values["method"], "ncat": int(values["ncat"]),
                                        "interval": float(values["interval"])}

    def _all_names(self) -> list[str]:
        out = []
        for t in self._targets:
            out += [t.widget.item(i).data(NAME_ROLE) for i in range(t.widget.count())]
        return out


def _dims_box(owner, key="ndim", label="Dimensions in solution:") -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(1, 20)
    spin.setValue(int(owner.opts.get(key, 2)))
    spin.valueChanged.connect(lambda v: owner.opts.__setitem__(key, int(v)))
    row = QHBoxLayout()
    row.addWidget(QLabel(label))
    row.addWidget(spin)
    row.addStretch()
    owner.options_layout.addLayout(row)
    return spin


# ===================================================================== CATREG
class CatregDialog(_ScaleMixin, VariablePickerDialog):
    DEFAULTS = {"scale": {}, "disc": {}, "missing": "listwise", "init": "numerical",
                "maxiter": 100, "eps": 0.00001, "regularization": None,
                "ridge": [0.0, 1.0, 0.02], "lasso": [0.0, 1.0, 0.02], "resampling": None,
                "folds": 10, "bootstrap_se": 0, "history": False, "correlations": False,
                "quantifications": True, "anova": True, "tplots": [], "save_pred": False,
                "save_res": False, "save_tra": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Categorical Regression", parent)
        self.add_target("dep", "Dependent Variable:", single=True)
        self.add_option_button("Define Scale (Dependent)...",
                               lambda: self._define_scale("dep", "Define Scale"))
        self.add_target("ind", "Independent Variable(s):")
        self.add_option_button("Define Scale (Independent)...",
                               lambda: self._define_scale("ind", "Define Scale"))
        self.add_option_button("Discretize...", lambda: self._discretize(self._all_names()))
        self.add_option_button("Missing...", self._missing)
        self.add_option_button("Options...", self._options)
        self.add_option_button("Regularization...", self._regularization)
        self.add_option_button("Output...", self._output)
        self.add_option_button("Save...", self._save)
        self.add_option_button("Plots...", self._plots)
        self.finish()

    def targets_changed(self):
        if hasattr(self, "_targets"):
            for k in ("dep", "ind"):
                try:
                    self._refresh(k)
                except StopIteration:
                    pass

    def _missing(self):
        dlg = SubDialog("Categorical Regression: Missing Values", self, columns=1)
        box = dlg.group("Strategy")
        dlg.radio(box, "missing", [("listwise", "Exclude objects with missing values on "
                                    "this variable"), ("mode", "Impute missing values: Mode"),
                                   ("extra", "Impute missing values: Extra category")],
                  self.opts["missing"])
        v = dlg.run()
        if v:
            self.opts.update(v)

    def _options(self):
        dlg = SubDialog("Categorical Regression: Options", self, columns=1)
        box = dlg.group("Initial Configuration")
        dlg.radio(box, "init", [("numerical", "Numerical"), ("random", "Random")],
                  self.opts["init"])
        box = dlg.group("Criteria")
        dlg.number(box, "eps", "Convergence:", self.opts["eps"], 1e-10, 1, 8)
        dlg.number(box, "maxiter", "Maximum iterations:", self.opts["maxiter"], 1, 99999, 0)
        v = dlg.run()
        if v:
            v["maxiter"] = int(v["maxiter"])
            self.opts.update(v)

    def _regularization(self):
        dlg = SubDialog("Categorical Regression: Regularization", self, columns=1)
        box = dlg.group("Method")
        dlg.radio(box, "regularization", [(None, "None"), ("ridge", "Ridge regression"),
                                          ("lasso", "Lasso"), ("enet", "Elastic Net")],
                  self.opts["regularization"])
        box = dlg.group("Ridge penalty (start, stop, increment)")
        r = self.opts["ridge"]
        dlg.number(box, "r0", "Start:", r[0], 0, 100, 3)
        dlg.number(box, "r1", "Stop:", r[1], 0, 100, 3)
        dlg.number(box, "r2", "Increment:", r[2], 0.001, 100, 3)
        box = dlg.group("Lasso penalty (start, stop, increment)")
        la = self.opts["lasso"]
        dlg.number(box, "l0", "Start:", la[0], 0, 100, 3)
        dlg.number(box, "l1", "Stop:", la[1], 0, 100, 3)
        dlg.number(box, "l2", "Increment:", la[2], 0.001, 100, 3)
        box = dlg.group("Resampling")
        dlg.radio(box, "resampling", [(None, "None"), ("cv", "Cross-validation")],
                  self.opts["resampling"])
        dlg.number(box, "folds", "Number of folds:", self.opts["folds"], 2, 100, 0)
        v = dlg.run()
        if v:
            self.opts["regularization"] = v["regularization"]
            self.opts["resampling"] = v["resampling"]
            self.opts["folds"] = int(v["folds"])
            self.opts["ridge"] = [v["r0"], v["r1"], v["r2"]]
            self.opts["lasso"] = [v["l0"], v["l1"], v["l2"]]

    def _output(self):
        dlg = SubDialog("Categorical Regression: Output", self, columns=1)
        box = dlg.group("Tables")
        for k, t in (("anova", "ANOVA"), ("history", "Iteration history"),
                     ("correlations", "Correlations of original and transformed variables"),
                     ("quantifications", "Category quantifications")):
            dlg.check(box, k, t, self.opts[k])
        box = dlg.group("Coefficients")
        dlg.number(box, "bootstrap_se", "Bootstrap samples for Std. Error (0 = none):",
                   self.opts["bootstrap_se"], 0, 10000, 0)
        v = dlg.run()
        if v:
            v["bootstrap_se"] = int(v["bootstrap_se"])
            self.opts.update(v)

    def _save(self):
        dlg = SubDialog("Categorical Regression: Save", self, columns=1)
        box = dlg.group("Save to Active Dataset")
        dlg.check(box, "save_pred", "Predicted values", self.opts["save_pred"])
        dlg.check(box, "save_res", "Residuals", self.opts["save_res"])
        dlg.check(box, "save_tra", "Transformed variables", self.opts["save_tra"])
        v = dlg.run()
        if v:
            self.opts.update(v)

    def _plots(self):
        names = self._all_names()
        dlg = SubDialog("Categorical Regression: Plots", self, columns=1)
        box = dlg.group("Transformation Plots")
        for n in names:
            dlg.check(box, f"t::{n}", n, n in self.opts["tplots"])
        v = dlg.run()
        if v:
            self.opts["tplots"] = [k[3:] for k, on in v.items() if on]

    def validate(self):
        if self.selected_one("dep") in self.selected("ind"):
            return "The dependent variable cannot also be an independent variable."
        return None

    def compute(self):
        o = self.opts
        return osc.catreg(self.ds, self._spec(self.selected_one("dep")),
                          [self._spec(n) for n in self.selected("ind")], o["missing"],
                          o["init"], int(o["maxiter"]), float(o["eps"]), o["regularization"],
                          tuple(o["ridge"]), tuple(o["lasso"]), o["resampling"],
                          int(o["folds"]), int(o["bootstrap_se"]), show_history=o["history"],
                          correlations=o["correlations"],
                          quantifications=o["quantifications"], anova=o["anova"],
                          transformation_plots=tuple(o["tplots"]),
                          save_predicted=o["save_pred"], save_residuals=o["save_res"],
                          save_transformed=o["save_tra"])


# ===================================================================== CATPCA
class CatpcaDialog(_ScaleMixin, VariablePickerDialog):
    SHOW_WEIGHT = True
    LEVEL_CHOICES = ("spord", "spnom", "mnom", "ordi", "nomi", "nume")
    DEFAULTS = {"scale": {}, "disc": {}, "ndim": 2, "missing": "passive",
                "normalization": "vprincipal", "maxiter": 100, "eps": 0.00001,
                "history": False, "descriptives": False, "vaf": True, "loadings": True,
                "quant": [], "corr": False, "objscores": False,
                "plots": ["objects", "loadings"], "tplots": [], "save_scores": False,
                "save_tra": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Categorical Principal Components", parent)
        self.add_target("vars", "Analysis Variables:", min_items=2)
        self.add_option_button("Define Scale and Weight...",
                               lambda: self._define_scale("vars", "Define Scale and Weight"))
        self.add_target("labels", "Labeling Variables:", single=True, required=False)
        _dims_box(self)
        self.add_option_button("Discretize...",
                               lambda: self._discretize(self.selected("vars")))
        self.add_option_button("Missing...", self._missing)
        self.add_option_button("Options...", self._options)
        self.add_option_button("Output...", self._output)
        self.add_option_button("Save...", self._save)
        self.add_option_button("Plots...", self._plots)
        self.finish()

    def targets_changed(self):
        if hasattr(self, "_targets"):
            try:
                self._refresh("vars")
            except StopIteration:
                pass

    def _missing(self):
        dlg = SubDialog(f"{self.windowTitle()}: Missing Values", self, columns=1)
        box = dlg.group("Strategy")
        dlg.radio(box, "missing", [("passive", "Exclude missing values (passive)"),
                                   ("mode", "Impute missing values: Mode"),
                                   ("extra", "Impute missing values: Extra category"),
                                   ("listwise", "Exclude objects with missing values")],
                  self.opts["missing"])
        v = dlg.run()
        if v:
            self.opts.update(v)

    def _options(self):
        dlg = SubDialog(f"{self.windowTitle()}: Options", self, columns=1)
        box = dlg.group("Normalization Method")
        dlg.radio(box, "normalization", list(osc.NORMALIZATIONS.items()),
                  self.opts["normalization"])
        box = dlg.group("Criteria")
        dlg.number(box, "eps", "Convergence:", self.opts["eps"], 1e-10, 1, 8)
        dlg.number(box, "maxiter", "Maximum iterations:", self.opts["maxiter"], 1, 99999, 0)
        v = dlg.run()
        if v:
            v["maxiter"] = int(v["maxiter"])
            self.opts.update(v)

    def _output(self):
        dlg = SubDialog(f"{self.windowTitle()}: Output", self, columns=2)
        box = dlg.group("Tables")
        for k, t in (("objscores", "Object scores"), ("loadings", "Component loadings"),
                     ("history", "Iteration history"),
                     ("corr", "Correlations of transformed variables"),
                     ("vaf", "Variance accounted for"), ("descriptives", "Descriptive "
                                                                        "statistics")):
            dlg.check(box, k, t, self.opts[k])
        box = dlg.group("Category Quantifications")
        for n in self.selected("vars"):
            dlg.check(box, f"q::{n}", n, n in self.opts["quant"])
        v = dlg.run()
        if v:
            self.opts["quant"] = [k[3:] for k, on in v.items() if k.startswith("q::") and on]
            self.opts.update({k: on for k, on in v.items() if not k.startswith("q::")})

    def _save(self):
        dlg = SubDialog(f"{self.windowTitle()}: Save", self, columns=1)
        box = dlg.group("Save to Active Dataset")
        dlg.check(box, "save_scores", "Object scores", self.opts["save_scores"])
        dlg.check(box, "save_tra", "Transformed variables", self.opts["save_tra"])
        v = dlg.run()
        if v:
            self.opts.update(v)

    def _plots(self):
        dlg = SubDialog(f"{self.windowTitle()}: Plots", self, columns=2)
        box = dlg.group("Object and Loading Plots")
        for k, t in (("objects", "Object points"), ("biplot", "Biplot objects and loadings"),
                     ("loadings", "Component loadings"), ("joint", "Joint plot of category "
                                                                   "points")):
            dlg.check(box, f"p::{k}", t, k in self.opts["plots"])
        box = dlg.group("Transformation Plots")
        for n in self.selected("vars"):
            dlg.check(box, f"t::{n}", n, n in self.opts["tplots"])
        v = dlg.run()
        if v:
            self.opts["plots"] = [k[3:] for k, on in v.items() if k.startswith("p::") and on]
            self.opts["tplots"] = [k[3:] for k, on in v.items() if k.startswith("t::") and on]

    def compute(self):
        o = self.opts
        return osc.catpca(self.ds, [self._spec(n) for n in self.selected("vars")],
                          int(o["ndim"]), o["missing"], o["normalization"], 0.0,
                          int(o["maxiter"]), float(o["eps"]), o["history"], o["descriptives"],
                          o["vaf"], o["loadings"], tuple(o["quant"]), o["corr"],
                          o["objscores"], tuple(o["plots"]), tuple(o["tplots"]),
                          self.selected_one("labels"), o["save_scores"], o["save_tra"])


# ===================================================================== MCA
class McaDialog(_ScaleMixin, VariablePickerDialog):
    SHOW_WEIGHT = True
    LEVEL_CHOICES = ("mnom",)
    DEFAULT_LEVEL = "mnom"
    DEFAULTS = {"scale": {}, "disc": {}, "ndim": 2, "missing": "passive",
                "normalization": "vprincipal", "maxiter": 100, "eps": 0.00001,
                "history": False, "discrimination": True, "objscores": False,
                "plots": ["objects", "categories", "discrimination"], "save_scores": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Multiple Correspondence Analysis", parent)
        self.add_target("vars", "Analysis Variables:", min_items=2)
        self.add_option_button("Define Variable Weight...",
                               lambda: self._define_scale("vars", "Define Variable Weight"))
        self.add_target("labels", "Labeling Variables:", single=True, required=False)
        _dims_box(self)
        self.add_option_button("Discretize...",
                               lambda: self._discretize(self.selected("vars")))
        self.add_option_button("Missing...", lambda: CatpcaDialog._missing(self))
        self.add_option_button("Options...", lambda: CatpcaDialog._options(self))
        self.add_option_button("Output...", self._output)
        self.add_option_button("Save...", self._save)
        self.add_option_button("Plots...", self._plots)
        self.finish()

    def _suffix(self, name):
        return f" ({self._spec(name)['weight']:g})"

    def targets_changed(self):
        if hasattr(self, "_targets"):
            try:
                self._refresh("vars")
            except StopIteration:
                pass

    def _output(self):
        dlg = SubDialog(f"{self.windowTitle()}: Output", self, columns=1)
        box = dlg.group("Tables")
        for k, t in (("objscores", "Object scores"), ("discrimination", "Discrimination "
                                                                         "measures"),
                     ("history", "Iteration history")):
            dlg.check(box, k, t, self.opts[k])
        v = dlg.run()
        if v:
            self.opts.update(v)

    def _save(self):
        dlg = SubDialog(f"{self.windowTitle()}: Save", self, columns=1)
        box = dlg.group("Save to Active Dataset")
        dlg.check(box, "save_scores", "Object scores", self.opts["save_scores"])
        v = dlg.run()
        if v:
            self.opts.update(v)

    def _plots(self):
        dlg = SubDialog(f"{self.windowTitle()}: Plots", self, columns=1)
        box = dlg.group("Plots")
        for k, t in (("objects", "Object points"), ("categories", "Joint category plot"),
                     ("discrimination", "Discrimination measures")):
            dlg.check(box, f"p::{k}", t, k in self.opts["plots"])
        v = dlg.run()
        if v:
            self.opts["plots"] = [k[3:] for k, on in v.items() if on]

    def compute(self):
        o = self.opts
        return osc.mca(self.ds, [self._spec(n) for n in self.selected("vars")],
                       int(o["ndim"]), o["missing"], o["normalization"], int(o["maxiter"]),
                       float(o["eps"]), o["discrimination"], ("*",), o["objscores"],
                       o["history"], tuple(o["plots"]), self.selected_one("labels"),
                       o["save_scores"])


# ===================================================================== OVERALS
class OveralsDialog(_ScaleMixin, VariablePickerDialog):
    LEVEL_CHOICES = ("ordi", "nomi", "mnom", "nume")
    DEFAULT_LEVEL = "ordi"
    DEFAULTS = {"scale": {}, "disc": {}, "sets": [[], []], "ndim": 2, "maxiter": 100,
                "eps": 0.00001, "history": False, "freq": False, "centroids": False,
                "weights": True, "quant": True, "objscores": False,
                "plots": ["objects", "loadings"], "save_scores": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Nonlinear Canonical Correlation Analysis", parent)
        self.opts["sets"] = [list(s) for s in self.opts.get("sets") or [[], []]]
        self._set = 0
        box = QGroupBox("Set")
        lay = QHBoxLayout(box)
        self.prev_btn = QPushButton("Previous")
        self.next_btn = QPushButton("Next")
        self.set_label = QLabel()
        self.prev_btn.clicked.connect(lambda: self._go(-1))
        self.next_btn.clicked.connect(lambda: self._go(1))
        lay.addWidget(self.prev_btn)
        lay.addWidget(self.set_label)
        lay.addWidget(self.next_btn)
        self.options_layout.addWidget(box)
        self.add_target("vars", "Variables:", required=False)
        self.add_option_button("Define Range and Scale...",
                               lambda: self._define_scale("vars", "Define Range and Scale"))
        self.add_target("labels", "Label Object Scores Plot(s) by:", single=True,
                        required=False)
        _dims_box(self)
        self.add_option_button("Options...", self._options)
        self.finish()
        self._load_set()

    def _store(self):
        while len(self.opts["sets"]) <= self._set:
            self.opts["sets"].append([])
        self.opts["sets"][self._set] = self.selected("vars")

    def _load_set(self):
        target = next(t for t in self._targets if t.key == "vars")
        self._move_back(target, all_items=True)
        for name in (self.opts["sets"][self._set] if self._set < len(self.opts["sets"])
                     else []):
            if self.ds.has_name(name):
                self._take(name, target)
        self._refresh("vars")
        n = max(len(self.opts["sets"]), self._set + 1)
        self.set_label.setText(f"Set {self._set + 1} of {n}")
        self.prev_btn.setEnabled(self._set > 0)

    def _go(self, step):
        self._store()
        # leaving the last set empty does not add a new set
        if step > 0 and not self.opts["sets"][self._set]:
            return
        self._set = max(0, self._set + step)
        while len(self.opts["sets"]) <= self._set:
            self.opts["sets"].append([])
        self._load_set()

    def targets_changed(self):
        if hasattr(self, "_targets") and hasattr(self, "set_label"):
            try:
                self._refresh("vars")
            except StopIteration:
                pass

    def _options(self):
        dlg = SubDialog(f"{self.windowTitle()}: Options", self, columns=2)
        box = dlg.group("Display")
        for k, t in (("freq", "Frequencies"), ("centroids", "Centroids"),
                     ("history", "Iteration history"), ("weights", "Weights and component "
                                                                    "loadings"),
                     ("quant", "Category quantifications"), ("objscores", "Object scores")):
            dlg.check(box, k, t, self.opts[k])
        box = dlg.group("Plot")
        for k, t in (("objects", "Category coordinates / object scores"),
                     ("loadings", "Component loadings"), ("centroids", "Centroids")):
            dlg.check(box, f"p::{k}", t, k in self.opts["plots"])
        dlg.check(box, "save_scores", "Save object scores", self.opts["save_scores"])
        box = dlg.group("Criteria")
        dlg.number(box, "maxiter", "Maximum iterations:", self.opts["maxiter"], 1, 99999, 0)
        dlg.number(box, "eps", "Convergence:", self.opts["eps"], 1e-10, 1, 8)
        v = dlg.run()
        if v:
            self.opts["plots"] = [k[3:] for k, on in v.items() if k.startswith("p::") and on]
            v = {k: x for k, x in v.items() if not k.startswith("p::")}
            v["maxiter"] = int(v["maxiter"])
            self.opts.update(v)

    def _sets(self):
        self._store()
        return [s for s in self.opts["sets"] if s]

    def validate(self):
        sets = self._sets()
        if len(sets) < 2:
            return "Define at least two sets of variables (use Next to start a new set)."
        return None

    def compute(self):
        o = self.opts
        sets = [[self._spec(n) for n in s] for s in self._sets()]
        return osc.overals(self.ds, sets, int(o["ndim"]), int(o["maxiter"]), float(o["eps"]),
                           o["history"], o["freq"], o["centroids"], o["weights"], o["quant"],
                           o["objscores"], tuple(o["plots"]), self.selected_one("labels"),
                           o["save_scores"])


# ===================================================================== chooser
class OptimalScalingDialog(QDialog):
    """Analyze > Dimension Reduction > Optimal Scaling...: choose MCA, CATPCA or OVERALS."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Optimal Scaling")
        lay = QVBoxLayout(self)
        g1 = QGroupBox("Optimal Scaling Level")
        l1 = QVBoxLayout(g1)
        self.all_mnom = QRadioButton("All variables multiple nominal")
        self.some = QRadioButton("Some variable(s) not multiple nominal")
        b1 = QButtonGroup(self)
        for rb in (self.all_mnom, self.some):
            b1.addButton(rb)
            l1.addWidget(rb)
        self.all_mnom.setChecked(True)
        g2 = QGroupBox("Number of Sets of Variables")
        l2 = QVBoxLayout(g2)
        self.one = QRadioButton("One set")
        self.multi = QRadioButton("Multiple sets")
        b2 = QButtonGroup(self)
        for rb in (self.one, self.multi):
            b2.addButton(rb)
            l2.addWidget(rb)
        self.one.setChecked(True)
        self.analysis = QLabel()
        for rb in (self.all_mnom, self.some, self.one, self.multi):
            rb.toggled.connect(self._update)
        lay.addWidget(g1)
        lay.addWidget(g2)
        box = QGroupBox("Selected Analysis")
        QVBoxLayout(box).addWidget(self.analysis)
        lay.addWidget(box)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Define")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self._update()

    def choice(self):
        if self.multi.isChecked():
            return OveralsDialog
        return McaDialog if self.all_mnom.isChecked() else CatpcaDialog

    def _update(self):
        self.analysis.setText({OveralsDialog: "Nonlinear canonical correlation",
                               McaDialog: "Multiple correspondence analysis",
                               CatpcaDialog: "Categorical principal components"}[self.choice()])


def open_optimal_scaling(editor) -> None:
    dlg = OptimalScalingDialog(editor)
    if dlg.exec():
        editor.run_analysis(dlg.choice())


# ===================================================================== PREFSCAL
class PrefscalDialog(VariablePickerDialog):
    DEFAULTS = {"ndim": 2, "similarities": False, "transform": "none", "conditional": True,
                "strength": 0.5, "range": 1.0, "maxiter": 5000, "eps": 0.000001,
                "minstress": 0.0001, "measures": True, "coords": True, "history": False,
                "plots": ["joint"]}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Multidimensional Unfolding", parent)
        self.add_target("vars", "Proximities:", accepts=lambda v: v.type == "numeric",
                        min_items=3)
        self.add_target("labels", "Row labels (optional):", single=True, required=False)
        self.add_option_button("Model...", self._model)
        self.add_option_button("Options...", self._options)
        self.add_option_button("Plots...", self._plots)
        self.add_option_button("Output...", self._output)
        self.finish()

    def _model(self):
        dlg = SubDialog("Multidimensional Unfolding: Model", self, columns=2)
        box = dlg.group("Proximities")
        dlg.radio(box, "similarities", [(False, "Dissimilarities"), (True, "Similarities")],
                  self.opts["similarities"])
        box = dlg.group("Dimensions")
        dlg.number(box, "ndim", "Dimensions:", self.opts["ndim"], 1, 10, 0)
        box = dlg.group("Proximity Transformations")
        dlg.radio(box, "transform", [("none", "None"), ("linear", "Linear"),
                                     ("ordinal", "Ordinal")], self.opts["transform"])
        box = dlg.group("Apply Transformations")
        dlg.radio(box, "conditional", [(True, "Within each row separately"),
                                       (False, "Across all sources simultaneously")],
                  self.opts["conditional"])
        v = dlg.run()
        if v:
            v["ndim"] = int(v["ndim"])
            self.opts.update(v)

    def _options(self):
        dlg = SubDialog("Multidimensional Unfolding: Options", self, columns=1)
        box = dlg.group("Penalty Term")
        dlg.number(box, "strength", "Strength:", self.opts["strength"], 0, 1, 3)
        dlg.number(box, "range", "Range:", self.opts["range"], 0, 1e6, 3)
        box = dlg.group("Iteration Criteria")
        dlg.number(box, "eps", "Stress convergence:", self.opts["eps"], 0, 1, 8)
        dlg.number(box, "minstress", "Minimum stress:", self.opts["minstress"], 0, 1, 6)
        dlg.number(box, "maxiter", "Maximum iterations:", self.opts["maxiter"], 1, 99999, 0)
        v = dlg.run()
        if v:
            v["maxiter"] = int(v["maxiter"])
            self.opts.update(v)

    def _plots(self):
        dlg = SubDialog("Multidimensional Unfolding: Plots", self, columns=1)
        box = dlg.group("Plots")
        for k, t in (("joint", "Final common space"), ("transformation", "Transformation "
                                                                         "plots"),
                     ("shepard", "Shepard plots")):
            dlg.check(box, f"p::{k}", t, k in self.opts["plots"])
        v = dlg.run()
        if v:
            self.opts["plots"] = [k[3:] for k, on in v.items() if on]

    def _output(self):
        dlg = SubDialog("Multidimensional Unfolding: Output", self, columns=1)
        box = dlg.group("Display")
        for k, t in (("measures", "Fit measures"), ("coords", "Final common space"),
                     ("history", "Iteration history")):
            dlg.check(box, k, t, self.opts[k])
        v = dlg.run()
        if v:
            self.opts.update(v)

    def compute(self):
        o = self.opts
        return osc.prefscal(self.ds, self.selected("vars"), int(o["ndim"]), o["similarities"],
                            o["transform"], o["conditional"], float(o["strength"]),
                            float(o["range"]), int(o["maxiter"]), float(o["eps"]),
                            float(o["minstress"]), self.selected_one("labels"), o["measures"],
                            o["coords"], o["history"], tuple(o["plots"]))


MENUS = {
    "Analyze/Regression": [("Optimal Scaling (CATREG)...", CatregDialog)],
    "Analyze/Dimension Reduction": [("Optimal Scaling...", open_optimal_scaling)],
    "Analyze/Scale": [("Multidimensional Unfolding (PREFSCAL)...", PrefscalDialog)],
}


def selftest() -> None:
    osc.selftest()

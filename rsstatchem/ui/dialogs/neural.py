"""Analyze > Neural Networks: Multilayer Perceptron and Radial Basis Function dialogs, tabbed
like SPSS (Variables, Partitions, Architecture, Training, Output, Save, Options). Core code:
:mod:`rsstatchem.core.stats.neural`."""
from __future__ import annotations

from rsstatchem.core.stats import neural
from rsstatchem.ui.dialogs.base import is_numeric
from rsstatchem.ui.dialogs.ml_tabs import TabbedPickerDialog

_RESCALE = [("standardized", "Standardized"), ("normalized", "Normalized"),
            ("adjusted", "Adjusted Normalized"), ("none", "None")]
_OUTPUT_KEYS = {"description": "Description (network information)",
                "diagram": "Diagram", "weights": "Synaptic weights (parameter estimates)",
                "summary": "Model summary", "classification": "Classification results",
                "roc": "ROC curve", "gains": "Cumulative gains chart", "lift": "Lift chart",
                "pred_obs": "Predicted by observed chart",
                "resid": "Residual by predicted chart (scale dependents)",
                "processing": "Case processing summary",
                "importance": "Independent variable importance analysis"}


class _NetworkDialog(TabbedPickerDialog):
    TITLE = ""
    PREFIX = ""
    DEFAULTS = {"rescale": "standardized", "train": 7.0, "test": 3.0, "hold": 0.0,
                "partition_var": None, "seed": 2000000, "save_pred": True, "save_prob": False,
                "save_part": False, **{f"out_{k}": k not in ("weights",) for k in _OUTPUT_KEYS}}

    def __init__(self, ds, parent=None):
        super().__init__(ds, self.TITLE, parent)
        self.add_target("deps", "Dependent Variables:")
        self.add_target("factors", "Factors:", required=False)
        self.add_target("covariates", "Covariates:", accepts=is_numeric, required=False)
        self.nums = [v.name for v in ds.variables if is_numeric(v)]
        self.make_tabs(self.pages())
        self.finish()

    def pages(self):
        return []

    def _variables_extra(self, f):
        f.combo("rescale", "Rescaling of Covariates:", _RESCALE)

    def _partitions(self, f):
        g = f.group("Randomly assign cases based on relative number of cases")
        g.number("train", "Training:", 0, 1e6, 1)
        g.number("test", "Testing:", 0, 1e6, 1)
        g.number("hold", "Holdout:", 0, 1e6, 1)
        g.number("seed", "Random seed (Mersenne Twister):", 1, 2 ** 31 - 1)
        g = f.group("Use partitioning variable to assign cases")
        g.variable("partition_var", "Variable (positive = training, 0 = testing, "
                                    "negative = holdout):", self.nums)

    def _output(self, f):
        g = f.group("Network Structure")
        for k in ("description", "diagram", "weights"):
            g.check(f"out_{k}", _OUTPUT_KEYS[k])
        g = f.group("Network Performance")
        for k in ("summary", "classification", "roc", "gains", "lift", "pred_obs", "resid"):
            g.check(f"out_{k}", _OUTPUT_KEYS[k])
        g = f.group("Other")
        g.check("out_processing", _OUTPUT_KEYS["processing"])
        g.check("out_importance", _OUTPUT_KEYS["importance"])

    def _save(self, f):
        g = f.group("Save for each dependent variable")
        g.check("save_pred", "Save predicted value or category")
        g.check("save_prob", "Save predicted pseudo-probability or category")
        g.check("save_part", "Save partition variable")

    def validate(self):
        if not self.selected("factors") and not self.selected("covariates"):
            return "Specify at least one factor or covariate."
        if set(self.selected("deps")) & (set(self.selected("factors")) |
                                         set(self.selected("covariates"))):
            return "A dependent variable cannot also be a predictor."
        o = self.opts
        if not o["partition_var"] and o["train"] <= 0:
            return "The training sample needs a positive relative number of cases."
        return None

    def _common(self):
        o = self.opts
        save = tuple(k for k, on in (("predicted", o["save_pred"]),
                                     ("probability", o["save_prob"]),
                                     ("partition", o["save_part"])) if on)
        out = {k: o[f"out_{k}"] for k in _OUTPUT_KEYS}
        return dict(rescale=o["rescale"], train_pct=o["train"], test_pct=o["test"],
                    hold_pct=o["hold"], partition_var=o["partition_var"], seed=int(o["seed"]),
                    output=out, save=save)


class MultilayerPerceptronDialog(_NetworkDialog):
    TITLE = "Multilayer Perceptron"
    DEFAULTS = {**_NetworkDialog.DEFAULTS, "auto_arch": True, "h_min": 1, "h_max": 50,
                "layers": 1, "units_auto": True, "units1": 4, "units2": 3,
                "hidden_act": "tanh", "out_act": "auto", "mode": "batch", "algorithm": "scg",
                "lam": 5e-7, "sigma": 5e-5, "center": 0.0, "offset": 0.5, "lr": 0.4,
                "lr_low": 0.001, "momentum": 0.9, "batch_size": 0, "lr_epochs": 10,
                "steps_no_decrease": 1, "max_time": 15.0, "max_steps": 0,
                "min_rel_change": 1e-4, "min_ratio": 1e-3, "check": "test"}

    def pages(self):
        return [("Partitions", self._partitions), ("Architecture", self._architecture),
                ("Training", self._training), ("Output", self._output), ("Save", self._save),
                ("Options", self._opts_page)]

    def __init__(self, ds, parent=None):
        super().__init__(ds, parent)
        from rsstatchem.ui.dialogs.ml_tabs import Form
        self._variables_extra(Form(self.opts, self.options_layout))

    def _architecture(self, f):
        f.radio("auto_arch", [(True, "Automatic architecture selection"),
                              (False, "Custom architecture")])
        g = f.group("Automatic: Number of Units in Hidden Layer")
        g.number("h_min", "Minimum:", 1, 1000)
        g.number("h_max", "Maximum:", 1, 1000)
        g = f.group("Custom: Hidden Layers")
        g.radio("layers", [(1, "One"), (2, "Two")])
        g.check("units_auto", "Automatically compute number of units")
        g.number("units1", "Units in hidden layer 1:", 1, 1000)
        g.number("units2", "Units in hidden layer 2:", 1, 1000)
        g.combo("hidden_act", "Hidden layer activation function:",
                [("tanh", "Hyperbolic tangent"), ("sigmoid", "Sigmoid")])
        g.combo("out_act", "Output layer activation function:",
                [("auto", "Automatic (softmax for categorical, identity for scale)"),
                 ("identity", "Identity"), ("softmax", "Softmax"),
                 ("tanh", "Hyperbolic tangent"), ("sigmoid", "Sigmoid")])

    def _training(self, f):
        g = f.group("Type of Training")
        g.radio("mode", [("batch", "Batch"), ("online", "Online"), ("minibatch", "Mini-batch")])
        g.number("batch_size", "Mini-batch size (0 = automatic):", 0, 1e6)
        g = f.group("Optimization Algorithm")
        g.radio("algorithm", [("scg", "Scaled conjugate gradient (batch only)"),
                              ("gd", "Gradient descent")])
        g = f.group("Training Options")
        g.number("lam", "Initial Lambda (SCG):", 0, 1, 9)
        g.number("sigma", "Initial Sigma (SCG):", 0, 1, 7)
        g.number("center", "Interval Center:", -10, 10, 3)
        g.number("offset", "Interval Offset:", 0, 10, 3)
        g.number("lr", "Initial Learning Rate (GD):", 0.0001, 10, 4)
        g.number("lr_low", "Lower Boundary of Learning Rate:", 0.00001, 10, 5)
        g.number("lr_epochs", "Learning rate reduction, in epochs:", 1, 1e6)
        g.number("momentum", "Momentum:", 0, 0.9999, 4)

    def _opts_page(self, f):
        g = f.group("Stopping Rules")
        g.number("steps_no_decrease", "Maximum steps without a decrease in error:", 1, 1000)
        g.radio("check", [("test", "Error to check: testing data (if any)"),
                          ("train", "Error to check: training data")])
        g.number("max_time", "Maximum training time (minutes):", 0.01, 1e4, 2)
        g.number("max_steps", "Maximum training epochs (0 = automatic):", 0, 1e7)
        g.number("min_rel_change", "Minimum relative change in training error:", 0, 1, 6)
        g.number("min_ratio", "Minimum relative change in training error ratio:", 0, 1, 6)

    def validate(self):
        err = super().validate()
        if err:
            return err
        o = self.opts
        if o["mode"] != "batch" and o["algorithm"] == "scg":
            return "Scaled conjugate gradient can only be used with batch training."
        if o["auto_arch"] and o["h_min"] > o["h_max"]:
            return "The minimum number of units must not exceed the maximum."
        return None

    def compute(self):
        o = self.opts
        s = neural.TrainSettings(algorithm=o["algorithm"], mode=o["mode"],
                                 batch_size=int(o["batch_size"]) or None, lr=o["lr"],
                                 lr_low=o["lr_low"], momentum=o["momentum"],
                                 lr_epochs=int(o["lr_epochs"]), lam=o["lam"], sigma=o["sigma"],
                                 interval_center=o["center"], interval_offset=o["offset"],
                                 max_steps=int(o["max_steps"]) or None,
                                 max_time=o["max_time"] * 60,
                                 steps_no_decrease=int(o["steps_no_decrease"]),
                                 min_rel_change=o["min_rel_change"], min_ratio=o["min_ratio"],
                                 check_test=o["check"] == "test")
        units = (None, None) if o["units_auto"] else (int(o["units1"]), int(o["units2"]))
        if not o["auto_arch"] and o["units_auto"]:
            units = (max(1, len(self.selected("covariates")) + 2), 3)
        return neural.mlp_procedure(
            self.ds, self.selected("deps"), self.selected("factors"),
            self.selected("covariates"), auto_arch=o["auto_arch"], h_min=int(o["h_min"]),
            h_max=int(o["h_max"]), layers=int(o["layers"]), units=units,
            hidden_act=o["hidden_act"], out_act=None if o["out_act"] == "auto" else o["out_act"],
            settings=s, **self._common())


class RadialBasisFunctionDialog(_NetworkDialog):
    TITLE = "Radial Basis Function"
    DEFAULTS = {**_NetworkDialog.DEFAULTS, "units": "auto", "u_min": 2, "u_max": 10,
                "u_fixed": 5, "activation": "normalized", "overlap_auto": True,
                "overlap": 1.5}

    def __init__(self, ds, parent=None):
        super().__init__(ds, parent)
        from rsstatchem.ui.dialogs.ml_tabs import Form
        self._variables_extra(Form(self.opts, self.options_layout))

    def pages(self):
        return [("Partitions", self._partitions), ("Architecture", self._architecture),
                ("Output", self._output), ("Save", self._save)]

    def _architecture(self, f):
        g = f.group("Number of Units in Hidden Layer")
        g.radio("units", [("auto", "Automatically compute number of units (best number "
                                   "within automatically computed range)"),
                          ("range", "Find the best number of units within a range"),
                          ("fixed", "Use a specified number of units")])
        g.number("u_min", "Minimum:", 1, 1000)
        g.number("u_max", "Maximum:", 1, 1000)
        g.number("u_fixed", "Number:", 1, 1000)
        g = f.group("Activation Function for Hidden Layer")
        g.radio("activation", [("normalized", "Normalized radial basis function"),
                               ("ordinary", "Ordinary radial basis function")])
        g = f.group("Overlap Among Hidden Units")
        g.check("overlap_auto", "Automatically compute overlapping factor (1 + 0.1 × inputs)")
        g.number("overlap", "Overlapping factor:", 0.01, 100, 3)

    def compute(self):
        o = self.opts
        return neural.rbf_procedure(
            self.ds, self.selected("deps"), self.selected("factors"),
            self.selected("covariates"), units=o["units"], u_min=int(o["u_min"]),
            u_max=int(o["u_max"]), u_fixed=int(o["u_fixed"]), activation=o["activation"],
            overlap=None if o["overlap_auto"] else o["overlap"], **self._common())


MENUS = {"Analyze/Neural Networks": [("Multilayer Perceptron...", MultilayerPerceptronDialog),
                                     ("Radial Basis Function...", RadialBasisFunctionDialog)]}


def selftest() -> None:
    neural.selftest()

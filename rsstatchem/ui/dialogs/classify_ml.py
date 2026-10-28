"""Analyze > Classify dialogs of the SPSS Decision Trees / Classify add-ons (WP9): Tree
(CHAID, Exhaustive CHAID, CRT, QUEST), TwoStep Cluster, Cluster Silhouettes and Nearest
Neighbor, laid out like SPSS. Core code: :mod:`rsstatchem.core.stats.trees`,
:mod:`~rsstatchem.core.stats.twostep`, :mod:`~rsstatchem.core.stats.knn`."""
from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QGroupBox, QHBoxLayout, QLabel, QVBoxLayout

from rsstatchem.core.stats import knn, trees, twostep
from rsstatchem.core.stats.ml_common import categorical, complete_rows, measure_of
from rsstatchem.ui.dialogs.base import VariablePickerDialog, is_numeric
from rsstatchem.ui.dialogs.ml_tabs import OptionsDialog, TabbedPickerDialog, parse_list


# ============================================================================ Tree
class TreeDialog(VariablePickerDialog):
    DEFAULTS = {"method": "chaid", "force_first": False, "depth_auto": True, "max_depth": 3,
                "min_parent": 100, "min_child": 50, "alpha_split": 0.05, "alpha_merge": 0.05,
                "chisq": "pearson", "bonferroni": True, "resplit": False, "intervals": 10,
                "impurity": "gini", "min_improvement": 0.0001, "prune": False, "prune_se": 1.0,
                "surrogates_auto": True, "max_surrogates": 1, "alpha_quest": 0.05,
                "validation": "none", "folds": 10, "train_pct": 50.0, "split_var": None,
                "seed": 2000000, "targets": [], "priors": "empirical", "custom_priors": "",
                "costs": "", "out_diagram": True, "out_table": False, "out_risk": True,
                "out_classification": True, "out_gains": True, "out_cumulative": True,
                "out_importance": True, "out_surrogates": False, "out_rules": False,
                "out_gain_chart": True, "out_index_chart": False, "out_response_chart": False,
                "save": []}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Decision Tree", parent)
        self.add_target("dep", "Dependent Variable:", single=True)
        self.add_target("indep", "Independent Variables:")
        self.add_checkbox("force_first", "Force first variable")
        row = QHBoxLayout()
        row.addWidget(QLabel("Growing Method:"))
        self.method_box = QComboBox()
        for key, text in trees.METHODS.items():
            self.method_box.addItem(text, key)
        self.method_box.setCurrentIndex(list(trees.METHODS).index(self.opts["method"]))
        self.method_box.currentIndexChanged.connect(
            lambda i: self.opts.__setitem__("method", list(trees.METHODS)[i]))
        row.addWidget(self.method_box, 1)
        self.options_layout.addLayout(row)
        self.add_option_button("Categories...", self._categories)
        self.add_option_button("Output...", self._output)
        self.add_option_button("Validation...", self._validation)
        self.add_option_button("Criteria...", self._criteria)
        self.add_option_button("Save...", self._save)
        self.add_option_button("Options...", self._options)
        self.finish()

    def _dep_levels(self):
        dep = self.selected_one("dep")
        if not dep or measure_of(self.ds, dep) == "scale":
            return None
        rows = complete_rows(self.ds, [dep])
        cat = categorical(self.ds, dep, rows)
        return cat

    def _categories(self):
        cat = self._dep_levels()
        dlg = OptionsDialog("Decision Tree: Categories", self.opts, self, columns=1)
        if cat is None:
            dlg.group("Categories").label("Target categories apply to a categorical dependent "
                                          "variable.")
        else:
            dlg.checklist("Target (gain summaries and charts)", "targets",
                          list(enumerate(cat.labels)))
        dlg.run(self.opts)

    def _output(self):
        dlg = OptionsDialog("Decision Tree: Output", self.opts, self)
        g = dlg.group("Tree")
        g.check("out_diagram", "Tree diagram")
        g.check("out_table", "Tree in table format")
        g = dlg.group("Statistics")
        g.check("out_risk", "Risk")
        g.check("out_classification", "Classification table")
        g.check("out_importance", "Independent variable importance (CRT, QUEST)")
        g.check("out_surrogates", "Surrogates by split (CRT, QUEST)")
        g = dlg.group("Node Performance")
        g.check("out_gains", "Summary (gain summary for nodes)")
        g.check("out_cumulative", "Cumulative statistics")
        g = dlg.group("Plots")
        g.check("out_gain_chart", "Gain")
        g.check("out_index_chart", "Index")
        g.check("out_response_chart", "Response")
        g = dlg.group("Rules")
        g.check("out_rules", "Generate classification rules (SPSS syntax)")
        dlg.run(self.opts)

    def _validation(self):
        dlg = OptionsDialog("Decision Tree: Validation", self.opts, self, columns=1)
        g = dlg.group("Validation")
        g.radio("validation", [("none", "None"), ("cv", "Crossvalidation"),
                               ("split", "Split-sample validation")])
        g.number("folds", "Number of sample folds:", 2, 25)
        g.number("train_pct", "Training sample (%):", 1, 99, 1)
        g.variable("split_var", "Or use variable (1 = training):",
                   [v.name for v in self.ds.variables if is_numeric(v)])
        g.number("seed", "Random seed:", 1, 2 ** 31 - 1)
        dlg.run(self.opts)

    def _criteria(self):
        dlg = OptionsDialog("Decision Tree: Criteria", self.opts, self)
        g = dlg.group("Growth Limits")
        g.check("depth_auto", "Automatic maximum tree depth (CHAID 3, CRT/QUEST 5)")
        g.number("max_depth", "Custom maximum depth:", 1, 50)
        g.number("min_parent", "Minimum cases in parent node:", 2, 1e7)
        g.number("min_child", "Minimum cases in child node:", 1, 1e7)
        g = dlg.group("CHAID")
        g.number("alpha_split", "Significance level for splitting nodes:", 0.0001, 0.9999, 4,
                 0.01)
        g.number("alpha_merge", "Significance level for merging categories:", 0.0001, 0.9999,
                 4, 0.01)
        g.radio("chisq", [("pearson", "Pearson chi-square"), ("lr", "Likelihood ratio")])
        g.check("bonferroni", "Adjust significance values using Bonferroni method")
        g.check("resplit", "Allow resplitting of merged categories within a node")
        g.number("intervals", "Intervals for scale independent variables:", 2, 64)
        g = dlg.group("CRT")
        g.combo("impurity", "Impurity measure:", [("gini", "Gini"), ("twoing", "Twoing"),
                                                   ("ordered", "Ordered twoing")])
        g.number("min_improvement", "Minimum change in improvement:", 0, 1, 6, 0.0001)
        g.check("surrogates_auto", "Automatic maximum surrogates")
        g.number("max_surrogates", "Custom maximum surrogates:", 0, 100)
        g = dlg.group("QUEST / Pruning")
        g.number("alpha_quest", "QUEST significance level for splitting:", 0.0001, 0.9999, 4,
                 0.01)
        g.check("prune", "Prune tree to avoid overfitting (CRT, QUEST)")
        g.number("prune_se", "Maximum difference in risk (in standard errors):", 0, 10, 2)
        dlg.run(self.opts)

    def _save(self):
        dlg = OptionsDialog("Decision Tree: Save", self.opts, self, columns=1)
        dlg.checklist("Saved Variables", "save", [
            ("node", "Terminal node number"), ("predicted", "Predicted value"),
            ("probability", "Predicted probabilities"),
            ("sample", "Sample assignment (training/testing)")])
        dlg.run(self.opts)

    def _options(self):
        dlg = OptionsDialog("Decision Tree: Options", self.opts, self, columns=1)
        g = dlg.group("Prior Probabilities (CRT and QUEST)")
        g.radio("priors", [("empirical", "Obtain from training sample (empirical priors)"),
                           ("equal", "Equal across categories"), ("custom", "Custom")])
        g.text("custom_priors", "Custom priors:", "e.g. 0.3 0.7")
        g = dlg.group("Misclassification Costs")
        g.text("costs", "Custom cost matrix:", "rows = observed, e.g. 0 1; 2 0  (empty = equal)")
        dlg.run(self.opts)

    def validate(self):
        dep = self.selected_one("dep")
        if self.opts["method"] == "quest" and measure_of(self.ds, dep) == "scale":
            return "QUEST requires a nominal or ordinal dependent variable."
        if dep in self.selected("indep"):
            return "The dependent variable cannot also be an independent variable."
        return None

    def compute(self):
        o = self.opts
        m = o["method"]
        s = {"min_parent": int(o["min_parent"]), "min_child": int(o["min_child"]),
             "alpha_split": o["alpha_split"], "alpha_merge": o["alpha_merge"],
             "chisq": o["chisq"], "bonferroni": o["bonferroni"], "resplit": o["resplit"],
             "intervals": int(o["intervals"]), "impurity": o["impurity"],
             "min_improvement": o["min_improvement"], "prune": o["prune"],
             "prune_se": o["prune_se"], "alpha_quest": o["alpha_quest"]}
        s["max_depth"] = trees.DEFAULT_DEPTH[m] if o["depth_auto"] else int(o["max_depth"])
        if not o["surrogates_auto"]:
            s["max_surrogates"] = int(o["max_surrogates"])
        costs = None
        if str(o.get("costs") or "").strip():
            costs = [parse_list(r) for r in str(o["costs"]).split(";") if r.strip()]
            costs = [[float(v) for v in r] for r in costs]
            import numpy as np
            costs = np.array(costs).T   # C(i|j): rows predicted, columns observed
        out = {k[4:]: v for k, v in o.items() if k.startswith("out_")}
        out["diagram"] = o["out_diagram"]
        return trees.tree_procedure(
            self.ds, self.selected_one("dep"), self.selected("indep"), m, o["force_first"], s,
            o["validation"], int(o["folds"]), o["train_pct"], o["split_var"], int(o["seed"]),
            o["targets"] or None, o["priors"], parse_list(o["custom_priors"]) or None, costs,
            out, tuple(o["save"]))


# ============================================================================ TwoStep
class TwoStepDialog(VariablePickerDialog):
    DEFAULTS = {"measure": "ll", "auto": True, "max_k": 15, "fixed_k": 5, "criterion": "bic",
                "standardize": True, "outliers": False, "outlier_pct": 25.0, "threshold": 0.0,
                "branch": 8, "depth": 3, "pivot": True, "charts": True, "save": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "TwoStep Cluster Analysis", parent)
        self.add_target("cat", "Categorical Variables:", required=False)
        self.add_target("cont", "Continuous Variables:", accepts=is_numeric, required=False)
        self.add_option_button("Options...", self._options)
        self.add_option_button("Output...", self._output)
        from rsstatchem.ui.dialogs.ml_tabs import Form
        box = QGroupBox("")
        lay = QHBoxLayout(box)
        for title, build in (("Distance Measure", lambda f: f.radio("measure", [
                ("ll", "Log-likelihood"), ("euclidean", "Euclidean")])),
                ("Number of Clusters", self._numclusters),
                ("Clustering Criterion", lambda f: f.radio("criterion", [
                    ("bic", "Schwarz's Bayesian (BIC)"),
                    ("aic", "Akaike's Information (AIC)")]))):
            g = QGroupBox(title)
            gl = QVBoxLayout(g)
            build(Form(self.opts, gl))
            lay.addWidget(g)
        self.options_layout.addWidget(box)
        self.add_checkbox("standardize", "Standardize continuous variables")
        self.finish()

    def _numclusters(self, f):
        f.radio("auto", [(True, "Determine automatically"), (False, "Specify fixed")])
        f.number("max_k", "Maximum:", 1, 99)
        f.number("fixed_k", "Number:", 1, 99)

    def _options(self):
        dlg = OptionsDialog("TwoStep Cluster: Options", self.opts, self, columns=1)
        g = dlg.group("Outlier Treatment")
        g.check("outliers", "Use noise handling")
        g.number("outlier_pct", "Percentage:", 0, 100, 1)
        g = dlg.group("Advanced (CF tree tuning criteria)")
        g.number("threshold", "Initial Distance Change Threshold:", 0, 1e6, 3)
        g.number("branch", "Maximum Branches (per leaf node):", 2, 64)
        g.number("depth", "Maximum Tree Depth (levels):", 1, 10)
        dlg.run(self.opts)

    def _output(self):
        dlg = OptionsDialog("TwoStep Cluster: Output", self.opts, self, columns=1)
        g = dlg.group("Output")
        g.check("pivot", "Pivot tables")
        g.check("charts", "Charts and tables (model summary, cluster quality, sizes, "
                          "predictor importance)")
        g = dlg.group("Working Data File")
        g.check("save", "Create cluster membership variable")
        dlg.run(self.opts)

    def validate(self):
        if not self.selected("cat") and not self.selected("cont"):
            return "Select at least one categorical or continuous variable."
        if self.opts["measure"] == "euclidean" and self.selected("cat"):
            return "The Euclidean distance measure requires continuous variables only."
        return None

    def compute(self):
        o = self.opts
        return twostep.twostep_procedure(
            self.ds, self.selected("cat"), self.selected("cont"), o["measure"], o["auto"],
            int(o["max_k"]), int(o["fixed_k"]), o["criterion"], o["standardize"],
            o["outliers"], o["outlier_pct"], int(o["branch"]), int(o["depth"]),
            o["threshold"], {"charts": o["charts"], "evaluation": True}, o["save"])


# ============================================================================ Silhouettes
class SilhouettesDialog(VariablePickerDialog):
    DEFAULTS = {"metric": "euclidean", "standardize": False, "plot": True, "save": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Cluster Silhouettes", parent)
        self.add_target("cluster", "Cluster Number Variable:", single=True)
        self.add_target("vars", "Variables:")
        from rsstatchem.ui.dialogs.ml_tabs import Form
        box = QGroupBox("Dissimilarity Measure")
        f = Form(self.opts, QVBoxLayout(box))
        f.radio("metric", [("euclidean", "Euclidean"), ("cityblock", "Absolute value "
                                                                     "(city block)"),
                           ("gower", "Gower (mixed scale and nominal variables)")])
        self.options_layout.addWidget(box)
        self.add_checkbox("standardize", "Standardize scale variables (Z scores)")
        self.add_checkbox("plot", "Silhouette plot")
        self.add_checkbox("save", "Save silhouette values and next-best cluster")
        self.finish()

    def validate(self):
        bad = [v for v in self.selected("vars") if not is_numeric(self.ds.var(v))]
        if bad and self.opts["metric"] != "gower":
            return "String variables need the Gower measure: " + ", ".join(bad)
        return None

    def compute(self):
        o = self.opts
        return twostep.silhouettes_procedure(self.ds, self.selected_one("cluster"),
                                             self.selected("vars"), o["metric"],
                                             o["standardize"], o["plot"], o["save"])


# ============================================================================ KNN
class NearestNeighborDialog(TabbedPickerDialog):
    DEFAULTS = {"normalize": True, "auto_k": False, "k": 3, "k_min": 3, "k_max": 5,
                "metric": "euclidean", "weight": False, "scale_fn": "mean",
                "select": False, "stop": "fixed", "auto_features": True, "n_features": 3,
                "min_change": 0.01, "train_pct": 70.0, "hold_pct": 30.0, "partition_var": None,
                "folds": 10, "fold_var": None, "seed": 2000000, "save_pred": True,
                "save_prob": False, "save_part": False, "charts": True, "neighbours": True}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Nearest Neighbor Analysis", parent)
        self.add_target("target", "Target:", single=True)
        self.add_target("features", "Features:")
        self.add_target("forced", "Forced Entry Features (feature selection):", required=False)
        self.add_target("focal", "Focal Case Identification (1 = focal):", accepts=is_numeric,
                        single=True, required=False)
        self.add_checkbox("normalize", "Normalize scale features")
        nums = [v.name for v in ds.variables if is_numeric(v)]
        self.make_tabs([("Neighbors", self._neighbors), ("Features", self._features),
                        ("Partitions", lambda f: self._partitions(f, nums)),
                        ("Save", self._save), ("Output", self._output),
                        ("Options", lambda f: f.group("Cross-validation").number(
                            "folds", "V-fold cross-validation folds (k selection):", 2, 50))])
        self.finish()

    def _neighbors(self, f):
        g = f.group("Number of Nearest Neighbors (k)")
        g.radio("auto_k", [(False, "Specify fixed k"), (True, "Automatically select k")])
        g.number("k", "k:", 1, 100)
        g.number("k_min", "Minimum:", 1, 100)
        g.number("k_max", "Maximum:", 1, 100)
        g = f.group("Distance Computation")
        g.radio("metric", [("euclidean", "Euclidean metric"), ("cityblock",
                                                              "City block metric")])
        g.check("weight", "Weight features by importance when computing distances")
        g = f.group("Predictions for Scale Target")
        g.radio("scale_fn", [("mean", "Mean value of nearest neighbors"),
                             ("median", "Median value of nearest neighbors")])

    def _features(self, f):
        g = f.group("Feature Selection")
        g.check("select", "Perform feature selection (forced features stay in the model)")
        g = f.group("Stopping Criterion")
        g.radio("stop", [("fixed", "Stop when the specified number of features have been "
                                   "selected"),
                         ("change", "Stop when the change in the absolute error ratio is less "
                                    "than or equal to the minimum")])
        g.check("auto_features", "Automatic number of features (min(20, P) − forced)")
        g.number("n_features", "Number to select:", 1, 1000)
        g.number("min_change", "Minimum change:", 0, 1, 4, 0.005)

    def _partitions(self, f, nums):
        g = f.group("Randomly Assign Cases to Partitions")
        g.number("train_pct", "Training %:", 1, 100, 1)
        g.number("hold_pct", "Holdout %:", 0, 99, 1)
        g.number("seed", "Random seed:", 1, 2 ** 31 - 1)
        g = f.group("Use Variable to Assign Cases")
        g.variable("partition_var", "Partition variable (positive = training):", nums)
        g = f.group("V-fold Cross-Validation (k selection)")
        g.variable("fold_var", "Fold variable (optional):", nums)

    def _save(self, f):
        g = f.group("Variables to Save")
        g.check("save_pred", "Predicted value or category")
        g.check("save_prob", "Predicted probability (categorical target)")
        g.check("save_part", "Training/Holdout partition variable")

    def _output(self, f):
        g = f.group("Viewer Output")
        g.check("charts", "Charts and tables (feature importance, error logs)")
        g.check("neighbours", "k nearest neighbors and distances for focal cases")

    def validate(self):
        t = self.selected_one("target")
        if t in self.selected("features") + self.selected("forced"):
            return "The target cannot also be a feature."
        if self.opts["auto_k"] and self.opts["k_min"] > self.opts["k_max"]:
            return "The minimum k must not exceed the maximum."
        return None

    def compute(self):
        o = self.opts
        max_f = None
        if o["select"] and o["stop"] == "fixed":
            P = len(self.selected("features"))
            forced = len(self.selected("forced"))
            max_f = max(0, min(20, P) - forced) if o["auto_features"] else int(o["n_features"])
        save = tuple(k for k, on in (("predicted", o["save_pred"]),
                                     ("probability", o["save_prob"]),
                                     ("partition", o["save_part"])) if on)
        return knn.knn_procedure(
            self.ds, self.selected_one("target"),
            self.selected("forced") + self.selected("features"), int(o["k"]),
            o["auto_k"], int(o["k_min"]), int(o["k_max"]), o["metric"], o["weight"],
            o["normalize"], o["scale_fn"], o["select"], self.selected("forced"), max_f,
            o["min_change"], 100 - o["hold_pct"] if not o["partition_var"] else o["train_pct"],
            o["hold_pct"], o["partition_var"], int(o["folds"]), o["fold_var"],
            self.selected_one("focal"), int(o["seed"]),
            {"charts": o["charts"], "neighbours": o["neighbours"]}, save)


MENUS = {"Analyze/Classify": [("TwoStep Cluster...", TwoStepDialog),
                              ("Cluster Silhouettes...", SilhouettesDialog),
                              ("Tree...", TreeDialog),
                              ("Nearest Neighbor...", NearestNeighborDialog)]}


def selftest() -> None:
    trees.selftest()
    twostep.selftest()
    knn.selftest()

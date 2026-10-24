"""Analyze > Classify dialogs: K-Means Cluster, Hierarchical Cluster and Discriminant (SPSS
layout). Registered through ``MENUS`` in :mod:`rsstatchem.ui.dialogs.logistic`."""
from __future__ import annotations

from PySide6.QtWidgets import (QButtonGroup, QFormLayout, QHBoxLayout, QRadioButton, QSpinBox)

from rsstatchem.core.stats import classify
from rsstatchem.ui.dialogs.base import SubDialog, VariablePickerDialog, is_numeric
from rsstatchem.ui.dialogs.compare import GroupingMixin


def _range_group(dlg: SubDialog, title: str, prefix: str, current: list) -> None:
    """None / Single solution / Range of solutions (SPSS Cluster Membership boxes)."""
    box = dlg.group(title)
    mode, lo, hi = current
    dlg.radio(box, f"{prefix}_mode", [("none", "None"), ("single", "Single solution"),
                                      ("range", "Range of solutions")], mode)
    dlg.number(box, f"{prefix}_lo", "Number of clusters / Minimum:", lo, 2, 999, 0)
    dlg.number(box, f"{prefix}_hi", "Maximum number of clusters:", hi, 2, 999, 0)


def _solution(mode: str, lo: float, hi: float) -> tuple[int, int] | None:
    if mode == "none":
        return None
    lo = int(lo)
    return (lo, lo) if mode == "single" else (min(lo, int(hi)), max(lo, int(hi)))


class KMeansDialog(VariablePickerDialog):
    DEFAULTS = {"k": 2, "iterate": True, "max_iter": 10, "converge": 0.0,
                "running_means": False, "save": [], "initial": True, "anova": False,
                "membership": False, "distances": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "K-Means Cluster Analysis", parent)
        self.add_target("vars", "Variables:", accepts=is_numeric)
        self.add_target("label", "Label Cases by:", single=True, required=False)
        self.iterate_button = self.add_option_button("Iterate...", self._iterate)
        self.add_option_button("Save...", self._save)
        self.add_option_button("Options...", self._options)
        row = QHBoxLayout()
        self.k = QSpinBox()
        self.k.setRange(2, 99)
        self.k.setValue(int(self.opts["k"]))
        self.k.valueChanged.connect(lambda v: self.opts.__setitem__("k", v))
        form = QFormLayout()
        form.addRow("Number of Clusters:", self.k)
        row.addLayout(form)
        row.addSpacing(20)
        self.r_iter = QRadioButton("Iterate and classify")
        self.r_class = QRadioButton("Classify only")
        group = QButtonGroup(self)
        group.addButton(self.r_iter)
        group.addButton(self.r_class)
        (self.r_iter if self.opts["iterate"] else self.r_class).setChecked(True)
        self.r_iter.toggled.connect(self._method_changed)
        row.addWidget(self.r_iter)
        row.addWidget(self.r_class)
        row.addStretch()
        self.options_layout.addLayout(row)
        self.finish()
        self._method_changed(self.opts["iterate"])

    def _method_changed(self, on: bool) -> None:
        self.opts["iterate"] = on
        self.iterate_button.setEnabled(on)

    def _iterate(self) -> None:
        dlg = SubDialog("K-Means Cluster Analysis: Iterate", self, columns=1)
        box = dlg.group("Iterate")
        dlg.number(box, "max_iter", "Maximum Iterations:", self.opts["max_iter"], 1, 999, 0)
        dlg.number(box, "converge", "Convergence Criterion:", self.opts["converge"], 0, 1, 4)
        dlg.check(box, "running_means", "Use running means", self.opts["running_means"])
        values = dlg.run()
        if values:
            values["max_iter"] = int(values["max_iter"])
            self.opts.update(values)

    def _save(self) -> None:
        dlg = SubDialog("K-Means Cluster: Save New Variables", self, columns=1)
        box = dlg.group("Save")
        dlg.check(box, "cluster", "Cluster membership", "cluster" in self.opts["save"])
        dlg.check(box, "distance", "Distance from cluster center",
                  "distance" in self.opts["save"])
        values = dlg.run()
        if values:
            self.opts["save"] = [k for k in ("cluster", "distance") if values[k]]

    def _options(self) -> None:
        dlg = SubDialog("K-Means Cluster Analysis: Options", self, columns=1)
        box = dlg.group("Statistics")
        dlg.check(box, "initial", "Initial cluster centers", self.opts["initial"])
        dlg.check(box, "anova", "ANOVA table", self.opts["anova"])
        dlg.check(box, "membership", "Cluster information for each case",
                  self.opts["membership"])
        dlg.check(box, "distances", "Distances between final cluster centers",
                  self.opts["distances"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def compute(self):
        o = self.opts
        return classify.kmeans_cluster(self.ds, self.selected("vars"), int(o["k"]),
                                       int(o["max_iter"]), o["converge"], o["running_means"],
                                       o["iterate"], o["initial"], o["anova"], o["membership"],
                                       o["distances"], self.selected_one("label"),
                                       tuple(o["save"]))


class HierarchicalDialog(VariablePickerDialog):
    DEFAULTS = {"method": "baverage", "measure": "seuclid", "transform": "none",
                "schedule": True, "proximity": False, "member": ["none", 3, 5],
                "dendrogram": True, "save": ["none", 3, 5]}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Hierarchical Cluster Analysis", parent)
        self.add_target("vars", "Variable(s):", accepts=is_numeric)
        self.add_target("label", "Label Cases by:", single=True, required=False)
        self.add_option_button("Statistics...", self._statistics)
        self.add_option_button("Plots...", self._plots)
        self.add_option_button("Method...", self._method)
        self.add_option_button("Save...", self._save)
        self.finish()

    def _statistics(self) -> None:
        dlg = SubDialog("Hierarchical Cluster Analysis: Statistics", self)
        box = dlg.group("Statistics")
        dlg.check(box, "schedule", "Agglomeration schedule", self.opts["schedule"])
        dlg.check(box, "proximity", "Proximity matrix", self.opts["proximity"])
        _range_group(dlg, "Cluster Membership", "member", self.opts["member"])
        values = dlg.run()
        if values:
            self.opts["schedule"], self.opts["proximity"] = values["schedule"], \
                values["proximity"]
            self.opts["member"] = [values["member_mode"], int(values["member_lo"]),
                                   int(values["member_hi"])]

    def _plots(self) -> None:
        dlg = SubDialog("Hierarchical Cluster Analysis: Plots", self, columns=1)
        dlg.check(dlg.group("Plots"), "dendrogram", "Dendrogram", self.opts["dendrogram"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def _method(self) -> None:
        dlg = SubDialog("Hierarchical Cluster Analysis: Method", self, columns=3)
        dlg.radio(dlg.group("Cluster Method"), "method", list(classify.LINKAGES.items()),
                  self.opts["method"])
        dlg.radio(dlg.group("Measure (Interval)"), "measure", list(classify.MEASURES.items()),
                  self.opts["measure"])
        dlg.radio(dlg.group("Transform Values: Standardize (by variable)"), "transform",
                  list(classify.STANDARDIZE.items()), self.opts["transform"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def _save(self) -> None:
        dlg = SubDialog("Hierarchical Cluster Analysis: Save", self, columns=1)
        _range_group(dlg, "Cluster Membership", "save", self.opts["save"])
        values = dlg.run()
        if values:
            self.opts["save"] = [values["save_mode"], int(values["save_lo"]),
                                 int(values["save_hi"])]

    def compute(self):
        o = self.opts
        return classify.hierarchical(self.ds, self.selected("vars"), o["method"], o["measure"],
                                     o["transform"], self.selected_one("label"), o["schedule"],
                                     o["proximity"], _solution(*o["member"]), o["dendrogram"],
                                     _solution(*o["save"]))


class DiscriminantDialog(GroupingMixin, VariablePickerDialog):
    DEFAULTS = {"group_def": None, "method": "enter", "means": True, "univariate": True,
                "box_m": True, "fisher": False, "unstandardized": False, "priors": "equal",
                "loo": True, "f_enter": 3.84, "f_remove": 2.71, "save": []}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Discriminant Analysis", parent)
        self.setup_grouping(use_range=True)
        self.add_target("ind", "Independents:", accepts=is_numeric)
        self.stepwise_button = None
        self.add_option_button("Statistics...", self._statistics)
        self.method_button = self.add_option_button("Method...", self._method)
        self.add_option_button("Classify...", self._classify)
        self.add_option_button("Save...", self._save)
        row = QHBoxLayout()
        self.r_enter = QRadioButton("Enter independents together")
        self.r_step = QRadioButton("Use stepwise method")
        group = QButtonGroup(self)
        group.addButton(self.r_enter)
        group.addButton(self.r_step)
        (self.r_step if self.opts["method"] == "stepwise" else self.r_enter).setChecked(True)
        self.r_step.toggled.connect(self._method_changed)
        row.addWidget(self.r_enter)
        row.addWidget(self.r_step)
        row.addStretch()
        self.options_layout.addLayout(row)
        self.finish()
        self._method_changed(self.opts["method"] == "stepwise")

    def _method_changed(self, on: bool) -> None:
        self.opts["method"] = "stepwise" if on else "enter"
        self.method_button.setEnabled(on)

    def _statistics(self) -> None:
        dlg = SubDialog("Discriminant Analysis: Statistics", self)
        box = dlg.group("Descriptives")
        dlg.check(box, "means", "Means", self.opts["means"])
        dlg.check(box, "univariate", "Univariate ANOVAs", self.opts["univariate"])
        dlg.check(box, "box_m", "Box's M", self.opts["box_m"])
        box = dlg.group("Function Coefficients")
        dlg.check(box, "fisher", "Fisher's", self.opts["fisher"])
        dlg.check(box, "unstandardized", "Unstandardized", self.opts["unstandardized"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def _method(self) -> None:
        dlg = SubDialog("Discriminant Analysis: Stepwise Method", self, columns=1)
        box = dlg.group("Method: Wilks' lambda — Criteria (Use F value)")
        dlg.number(box, "f_enter", "Entry:", self.opts["f_enter"], 0.0001, 999, 4)
        dlg.number(box, "f_remove", "Removal:", self.opts["f_remove"], 0.0, 999, 4)
        values = dlg.run()
        if values:
            if values["f_remove"] >= values["f_enter"]:
                values["f_remove"] = max(0.0, values["f_enter"] - 0.5)
            self.opts.update(values)

    def _classify(self) -> None:
        dlg = SubDialog("Discriminant Analysis: Classification", self, columns=1)
        dlg.radio(dlg.group("Prior Probabilities"), "priors",
                  [("equal", "All groups equal"),
                   ("size", "Compute from group sizes")], self.opts["priors"])
        dlg.check(dlg.group("Display"), "loo", "Leave-one-out classification",
                  self.opts["loo"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def _save(self) -> None:
        dlg = SubDialog("Discriminant Analysis: Save", self, columns=1)
        box = dlg.group("Save")
        for key, text in (("group", "Predicted group membership"),
                          ("scores", "Discriminant scores"),
                          ("probs", "Probabilities of group membership")):
            dlg.check(box, key, text, key in self.opts["save"])
        values = dlg.run()
        if values:
            self.opts["save"] = [k for k in ("group", "scores", "probs") if values[k]]

    def compute(self):
        o = self.opts
        d = self._definition()
        return classify.discriminant(self.ds, self.selected_one("group"), d["lo"], d["hi"],
                                     self.selected("ind"), o["method"], o["priors"],
                                     o["means"], o["univariate"], o["box_m"], o["fisher"],
                                     o["unstandardized"], o["loo"], o["f_enter"],
                                     o["f_remove"], save=tuple(o["save"]))

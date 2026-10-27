"""Analyze > Scale > Multidimensional Scaling dialogs (PROXSCAL and ALSCAL), laid out like SPSS:
variables, "Distances: Data are distances / Create distances from data" with Shape... /
Measure..., Model... and Options.... Core code: :mod:`rsstatchem.core.stats.mds`."""
from __future__ import annotations

from PySide6.QtWidgets import QButtonGroup, QGroupBox, QRadioButton, QVBoxLayout

from rsstatchem.core.stats import mds
from rsstatchem.ui.dialogs.base import SubDialog, VariablePickerDialog, is_numeric


class _MDSDialog(VariablePickerDialog):
    TITLE = "Multidimensional Scaling"
    DEFAULTS = {"source": "matrix", "shape": "lower", "similarities": False,
                "measure": "euclidean", "between": "variables", "standardize": False,
                "dims": 2, "level": "ratio", "maxiter": 100, "eps": 0.0001,
                "init": "torgerson", "plots": ["common", "transformation"], "history": True}

    def __init__(self, ds, parent=None):
        super().__init__(ds, self.TITLE, parent)
        self.add_target("vars", "Variables:", accepts=is_numeric, min_items=3)
        self.add_target("label", "Label Cases by (distances between cases):", single=True,
                        required=False)
        box = QGroupBox("Distances")
        lay = QVBoxLayout(box)
        group = QButtonGroup(self)
        self._src = {}
        for key, text in (("matrix", "Data are distances (square matrix)"),
                          ("data", "Create distances from data")):
            rb = QRadioButton(text)
            rb.setChecked(self.opts["source"] == key)
            rb.toggled.connect(lambda on, k=key: on and self.opts.__setitem__("source", k))
            group.addButton(rb)
            lay.addWidget(rb)
        self.options_layout.addWidget(box)
        self.add_option_button("Shape...", self._shape)
        self.add_option_button("Measure...", self._measure)
        self.add_option_button("Model...", self._model)
        self.add_option_button("Options...", self._options)
        self.finish()

    def _shape(self):
        dlg = SubDialog(f"{self.TITLE}: Shape of Data", self, columns=1)
        box = dlg.group("Shape")
        dlg.radio(box, "shape", [("lower", "Lower-triangular matrix"),
                                 ("upper", "Upper-triangular matrix"),
                                 ("full", "Full matrix (symmetrized)")], self.opts["shape"])
        box = dlg.group("Proximities")
        dlg.radio(box, "similarities", [(False, "Dissimilarities"), (True, "Similarities")],
                  self.opts["similarities"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def _measure(self):
        dlg = SubDialog(f"{self.TITLE}: Create Measure from Data", self)
        box = dlg.group("Measure (Interval)")
        dlg.radio(box, "measure", list(mds.MEASURES.items()), self.opts["measure"])
        box = dlg.group("Create Distance Matrix")
        dlg.radio(box, "between", [("variables", "Between variables"),
                                   ("cases", "Between cases")], self.opts["between"])
        dlg.check(box, "standardize", "Standardize values (Z scores)", self.opts["standardize"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def _model(self):
        dlg = SubDialog(f"{self.TITLE}: Model", self)
        box = dlg.group("Level of Measurement / Transformation")
        dlg.radio(box, "level", list(mds.LEVELS.items()), self.opts["level"])
        box = dlg.group("Dimensions")
        dlg.number(box, "dims", "Dimensions:", self.opts["dims"], 1, 10, 0)
        values = dlg.run()
        if values:
            values["dims"] = int(values["dims"])
            self.opts.update(values)

    def _options(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def validate(self):
        n = len(self.selected("vars"))
        if self.opts["source"] == "matrix" and self.ds.n_cases < n:
            return f"A {n} x {n} distance matrix needs at least {n} cases."
        return None

    def _common(self) -> dict:
        o = self.opts
        return dict(source=o["source"], dims=int(o["dims"]), level=o["level"],
                    measure=o["measure"], between=o["between"], standardize=o["standardize"],
                    similarities=o["similarities"], shape=o["shape"],
                    label=self.selected_one("label"))


class ProxscalDialog(_MDSDialog):
    TITLE = "Multidimensional Scaling (PROXSCAL)"

    def _options(self):
        dlg = SubDialog(f"{self.TITLE}: Options", self)
        box = dlg.group("Initial Configuration")
        dlg.radio(box, "init", [("torgerson", "Torgerson"), ("simplex", "Simplex")],
                  self.opts["init"])
        box = dlg.group("Iteration Criteria")
        dlg.number(box, "eps", "Stress convergence:", self.opts["eps"], 1e-8, 0.1, 6)
        dlg.number(box, "maxiter", "Maximum iterations:", self.opts["maxiter"], 1, 10000, 0)
        box = dlg.group("Display")
        dlg.check(box, "history", "Iteration history", self.opts["history"])
        box = dlg.group("Plots")
        for key, text in (("common", "Common space"), ("transformation", "Transformations "
                                                                         "(Shepard)"),
                          ("residuals", "Residuals")):
            dlg.check(box, f"p::{key}", text, key in self.opts["plots"])
        values = dlg.run()
        if values:
            self.opts["plots"] = [k[3:] for k, v in values.items() if k.startswith("p::") and v]
            for k in ("init", "eps", "history"):
                self.opts[k] = values[k]
            self.opts["maxiter"] = int(values["maxiter"])

    def compute(self):
        o = self.opts
        return mds.proxscal(self.ds, self.selected("vars"), init=o["init"],
                            maxiter=int(o["maxiter"]), eps=o["eps"], history=o["history"],
                            plots=tuple(o["plots"]), **self._common())


class AlscalDialog(_MDSDialog):
    TITLE = "Multidimensional Scaling (ALSCAL)"
    DEFAULTS = {**_MDSDialog.DEFAULTS, "level": "ordinal", "maxiter": 30, "eps": 0.001,
                "plots": ["config", "fit"]}

    def _options(self):
        dlg = SubDialog(f"{self.TITLE}: Options", self)
        box = dlg.group("Display")
        for key, text in (("config", "Group plots (derived configuration)"),
                          ("fit", "Scatterplot of linear fit")):
            dlg.check(box, f"p::{key}", text, key in self.opts["plots"])
        box = dlg.group("Criteria")
        dlg.number(box, "eps", "S-stress convergence:", self.opts["eps"], 1e-6, 0.1, 5)
        dlg.number(box, "maxiter", "Maximum iterations:", self.opts["maxiter"], 1, 1000, 0)
        values = dlg.run()
        if values:
            self.opts["plots"] = [k[3:] for k, v in values.items() if k.startswith("p::") and v]
            self.opts["eps"] = values["eps"]
            self.opts["maxiter"] = int(values["maxiter"])

    def compute(self):
        o = self.opts
        return mds.alscal_section(self.ds, self.selected("vars"), maxiter=int(o["maxiter"]),
                                  eps=o["eps"], plots=tuple(o["plots"]), **self._common())


MENUS = {"Analyze/Scale": [("Multidimensional Scaling (PROXSCAL)...", ProxscalDialog),
                           ("Multidimensional Scaling (ALSCAL)...", AlscalDialog)]}


def selftest() -> None:
    mds.selftest()

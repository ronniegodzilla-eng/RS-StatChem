"""Analyze > Dimension Reduction > Correspondence Analysis dialog, laid out like SPSS (Row /
Column with Define Range..., Model..., Statistics..., Plots...). Core code:
:mod:`rsstatchem.core.stats.correspondence`."""
from __future__ import annotations

import pandas as pd

from rsstatchem.core.dataset import NUMERIC
from rsstatchem.core.stats import correspondence as ca
from rsstatchem.ui.dialogs.base import DefineRangeDialog, SubDialog, VariablePickerDialog


class CorrespondenceDialog(VariablePickerDialog):
    DEFAULTS = {"dims": 2, "normalization": "symmetrical", "q": 0.0, "table": True,
                "overview_rows": True, "overview_cols": True, "row_profiles": False,
                "col_profiles": False, "permutation": False, "plots": ["biplot"],
                "row_range": None, "col_range": None}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Correspondence Analysis", parent)
        self.add_target("row", "Row:", single=True)
        self.row_button = self.add_option_button("Define Range (Row)...",
                                                 lambda: self._range("row"))
        self.add_target("col", "Column:", single=True)
        self.col_button = self.add_option_button("Define Range (Column)...",
                                                 lambda: self._range("col"))
        self.add_option_button("Model...", self._model)
        self.add_option_button("Statistics...", self._statistics)
        self.add_option_button("Plots...", self._plots)
        self.finish()

    def targets_changed(self):
        if not hasattr(self, "col_button"):
            return
        for key, button in (("row", self.row_button), ("col", self.col_button)):
            name = self.selected_one(key)
            numeric = name is not None and self.ds.var(name).type == NUMERIC
            button.setEnabled(numeric)
            if name:
                r = self._current_range(key)
                self.set_item_suffix(key, f"({r[0]:g} {r[1]:g})" if r else "")

    def _current_range(self, key):
        r = self.opts.get(f"{key}_range")
        return tuple(r["range"]) if r and r.get("var") == self.selected_one(key) else None

    def _range(self, key):
        name = self.selected_one(key)
        if not name:
            return
        cur = self._current_range(key)
        if cur is None:
            x = pd.to_numeric(self.ds.analysis_series(name), errors="coerce").dropna()
            cur = (float(x.min()), float(x.max())) if len(x) else (1, 2)
        values = DefineRangeDialog(cur, self).run()
        if values:
            self.opts[f"{key}_range"] = {"var": name, "range": [values["lo"], values["hi"]]}
        self.targets_changed()

    def _model(self):
        dlg = SubDialog("Correspondence Analysis: Model", self)
        box = dlg.group("Dimensions in solution")
        dlg.number(box, "dims", "Dimensions:", self.opts["dims"], 1, 20, 0)
        box = dlg.group("Distance Measure")
        dlg.radio(box, "_dist", [("chisq", "Chi square")], "chisq")
        box = dlg.group("Normalization Method")
        dlg.radio(box, "normalization", list(ca.NORMALIZATIONS.items()),
                  self.opts["normalization"])
        dlg.number(box, "q", "Custom value (-1 to 1):", self.opts["q"], -1, 1, 2)
        values = dlg.run()
        if values:
            values.pop("_dist", None)
            values["dims"] = int(values["dims"])
            self.opts.update(values)

    def _statistics(self):
        dlg = SubDialog("Correspondence Analysis: Statistics", self, columns=1)
        box = dlg.group("")
        for key, text in (("table", "Correspondence table"),
                          ("overview_rows", "Overview of row points"),
                          ("overview_cols", "Overview of column points"),
                          ("permutation", "Permutations of the correspondence table"),
                          ("row_profiles", "Row profiles"), ("col_profiles", "Column profiles")):
            dlg.check(box, key, text, self.opts[key])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def _plots(self):
        dlg = SubDialog("Correspondence Analysis: Plots", self, columns=1)
        box = dlg.group("Scatterplots")
        for key, text in (("biplot", "Biplot"), ("rows", "Row points"),
                          ("cols", "Column points")):
            dlg.check(box, f"p::{key}", text, key in self.opts["plots"])
        values = dlg.run()
        if values:
            self.opts["plots"] = [k[3:] for k, v in values.items() if v]

    def validate(self):
        if self.selected_one("row") == self.selected_one("col"):
            return "The row and column variables must differ."
        return None

    def compute(self):
        o = self.opts
        return ca.correspondence(self.ds, self.selected_one("row"), self.selected_one("col"),
                                 self._current_range("row"), self._current_range("col"),
                                 int(o["dims"]), o["normalization"], o["q"], o["table"],
                                 o["overview_rows"], o["overview_cols"], o["row_profiles"],
                                 o["col_profiles"], o["permutation"], tuple(o["plots"]))


MENUS = {"Analyze/Dimension Reduction": [("Correspondence Analysis...", CorrespondenceDialog)]}


def selftest() -> None:
    ca.selftest()

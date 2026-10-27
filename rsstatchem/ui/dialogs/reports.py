"""Analyze > Reports: Codebook, OLAP Cubes, Case Summaries, Report Summaries in Rows and in
Columns (SPSS CODEBOOK, OLAP CUBES, SUMMARIZE, REPORT)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import reports as rp
from rsstatchem.ui.dialogs.base import SubDialog, VariablePickerDialog, is_numeric

_SUMMARY = ("n", "sum", "mean", "median", "std", "var", "min", "max", "range", "se", "first",
            "last", "pct_sum", "pct_n", "skew", "kurt", "gmean", "hmean")


def _stats_dialog(parent, title: str, current: list[str], allowed=_SUMMARY) -> list[str] | None:
    dlg = SubDialog(title, parent)
    box = dlg.group("Statistics")
    half = (len(allowed) + 1) // 2
    box2 = dlg.group(" ")
    for i, key in enumerate(allowed):
        dlg.check(box if i < half else box2, key, rp.STATS[key], key in current)
    values = dlg.run()
    if values is None:
        return None
    return [k for k in allowed if values.get(k)] or list(current)


class CodebookDialog(VariablePickerDialog):
    DEFAULTS = {"counts": True, "percents": True, "max_categories": 200}

    def __init__(self, ds: Dataset, parent=None):
        super().__init__(ds, "Codebook", parent)
        self.add_target("vars", "Codebook Variables:")
        self.add_option_button("Output...", self._output)
        self.finish()

    def _output(self) -> None:
        dlg = SubDialog("Codebook: Output and Statistics", self, columns=1)
        box = dlg.group("Categorical variables")
        dlg.check(box, "counts", "Count", self.opts["counts"])
        dlg.check(box, "percents", "Percent", self.opts["percents"])
        dlg.number(box, "max_categories", "Maximum number of categories:",
                   self.opts["max_categories"], 1, 10000, 0)
        values = dlg.run()
        if values:
            self.opts.update(values)

    def compute(self):
        return rp.codebook(self.ds, self.selected("vars"), int(self.opts["max_categories"]),
                           self.opts["counts"], self.opts["percents"])


class OlapCubesDialog(VariablePickerDialog):
    DEFAULTS = {"stats": ["sum", "n", "mean", "std", "pct_sum", "pct_n"]}

    def __init__(self, ds: Dataset, parent=None):
        super().__init__(ds, "OLAP Cubes", parent)
        self.add_target("summary", "Summary Variable(s):", is_numeric)
        self.add_target("groups", "Grouping Variable(s):", required=False)
        self.add_option_button("Statistics...", self._stats)
        self.finish()

    def _stats(self) -> None:
        chosen = _stats_dialog(self, "OLAP Cubes: Statistics", self.opts["stats"])
        if chosen:
            self.opts["stats"] = chosen

    def compute(self):
        return rp.olap_cubes(self.ds, self.selected("summary"), self.selected("groups"),
                             tuple(self.opts["stats"]))


class CaseSummariesDialog(VariablePickerDialog):
    DEFAULTS = {"display": True, "limit": True, "limit_n": 100, "valid_only": False,
                "case_numbers": True, "stats": ["n"], "title": "Case Summaries"}

    def __init__(self, ds: Dataset, parent=None):
        super().__init__(ds, "Summarize Cases", parent)
        self.add_target("vars", "Variables:")
        self.add_target("groups", "Grouping Variable(s):", required=False)
        self.add_checkbox("display", "Display cases")
        self.add_checkbox("limit", "Limit cases to first n (Statistics... sets n)")
        self.add_checkbox("valid_only", "Show only valid cases")
        self.add_checkbox("case_numbers", "Show case numbers")
        self.add_option_button("Statistics...", self._stats)
        self.add_option_button("Options...", self._options)
        self.finish()

    def _stats(self) -> None:
        chosen = _stats_dialog(self, "Summary Report: Statistics", self.opts["stats"])
        if chosen:
            self.opts["stats"] = chosen

    def _options(self) -> None:
        dlg = SubDialog("Summarize Cases: Options", self, columns=1)
        box = dlg.group("Listing")
        dlg.number(box, "limit_n", "Limit cases to first:", self.opts["limit_n"], 1, 100000, 0)
        values = dlg.run()
        if values:
            self.opts.update(values)

    def compute(self):
        o = self.opts
        return rp.case_summaries(self.ds, self.selected("vars"), self.selected("groups"),
                                 o["display"], int(o["limit_n"]) if o["limit"] else None,
                                 o["valid_only"], o["case_numbers"], tuple(o["stats"]),
                                 o["title"])


class ReportRowsDialog(VariablePickerDialog):
    DEFAULTS = {"stats": ["mean"], "list_cases": False}

    def __init__(self, ds: Dataset, parent=None):
        super().__init__(ds, "Report: Summaries in Rows", parent)
        self.add_target("columns", "Data Columns:", is_numeric)
        self.add_target("breaks", "Break Columns:")
        self.add_checkbox("list_cases", "Display cases")
        self.add_option_button("Summary...", self._stats)
        self.finish()

    def _stats(self) -> None:
        chosen = _stats_dialog(self, "Report: Summary Lines", self.opts["stats"])
        if chosen:
            self.opts["stats"] = chosen

    def compute(self):
        return rp.report_rows(self.ds, self.selected("columns"), self.selected("breaks"),
                              tuple(self.opts["stats"]), self.opts["list_cases"])


class ReportColumnsDialog(VariablePickerDialog):
    DEFAULTS = {"stat": "sum", "grand_total": True, "pairs": []}

    def __init__(self, ds: Dataset, parent=None):
        super().__init__(ds, "Report: Summaries in Columns", parent)
        self.add_target("columns", "Data Columns:", is_numeric)
        self.add_target("breaks", "Break Columns:")
        row = QHBoxLayout()
        row.addWidget(QLabel("Summary for every data column:"))
        self.stat_box = QComboBox()
        for key in _SUMMARY:
            self.stat_box.addItem(rp.STATS[key], key)
        self.stat_box.setCurrentIndex(_SUMMARY.index(self.opts["stat"]))
        self.stat_box.currentIndexChanged.connect(
            lambda _: self.opts.__setitem__("stat", self.stat_box.currentData()))
        row.addWidget(self.stat_box)
        row.addStretch()
        self.options_layout.addLayout(row)
        self.add_checkbox("grand_total", "Display grand total")
        self.finish()

    def compute(self):
        pairs = [(c, self.opts["stat"]) for c in self.selected("columns")]
        return rp.report_columns(self.ds, pairs, self.selected("breaks"),
                                 self.opts["grand_total"])


MENUS = {"Analyze/Reports": [("Codebook...", CodebookDialog),
                             ("OLAP Cubes...", OlapCubesDialog),
                             ("Case Summaries...", CaseSummariesDialog),
                             ("Report Summaries in Rows...", ReportRowsDialog),
                             ("Report Summaries in Columns...", ReportColumnsDialog)]}

def selftest() -> None:
    rng = np.random.default_rng(11)
    ds = Dataset(pd.DataFrame({"g": rng.integers(1, 3, 30).astype(float),
                               "x": rng.normal(10, 2, 30), "s": list("ab" * 15)}))
    ds.var("g").measure = "nominal"
    ds.var("g").value_labels = {1.0: "One", 2.0: "Two"}
    assert rp.codebook(ds, ["g", "x", "s"]).items
    sec = rp.olap_cubes(ds, ["x"], ["g"])
    total = sec.items[-1].rows[-1]
    assert total[0] == "Total"
    assert rp.case_summaries(ds, ["x", "s"], ["g"], limit=5).items
    assert rp.report_rows(ds, ["x"], ["g"], ("mean", "n")).items
    assert rp.report_columns(ds, [("x", "sum"), ("x", "mean")], ["g"]).items
    assert rp.file_info(ds).items

"""Analyze > Tables > Custom Tables: an SPSS-style table builder with a live preview.

Drag variables (or multiple response sets) to Rows, Columns and Layers, choose nesting or
stacking, the summary statistics, totals and category order, and the significance tests.
Measurement levels come from the Variable View (scale = summary statistics, nominal/ordinal =
categories); right-click a variable in a tray to treat it the other way.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QLabel, QLineEdit, QMenu, QTextBrowser)

from rsstatchem.core import ctables as ct
from rsstatchem.core import report
from rsstatchem.core.dataset import Dataset
from rsstatchem.ui.dialogs.base import SubDialog
from rsstatchem.ui.dialogs.picker import NAME_ROLE, ListPickerDialog

PREVIEW_CASES = 2000


class CustomTablesDialog(ListPickerDialog):
    DEFAULTS = {"row_mode": "nest", "col_mode": "nest", "cat_stats": ["count"],
                "scale_stats": ["mean"], "totals": False, "order": "ascending",
                "show_empty": True, "chisq": False, "compare_props": False,
                "compare_means": False, "alpha": 0.05, "adjust": "bonferroni",
                "title": "Custom Table", "decimals": -1, "scale_override": {}}

    def __init__(self, ds: Dataset, parent=None):
        super().__init__(ds, "Custom Tables", parent)
        self.resize(980, 700)
        self.row_mode = QComboBox()
        self.col_mode = QComboBox()
        for box, key in ((self.row_mode, "row_mode"), (self.col_mode, "col_mode")):
            box.addItem("Nest", "nest")
            box.addItem("Stack", "stack")
            box.setCurrentIndex(0 if self.opts[key] == "nest" else 1)
            box.currentIndexChanged.connect(
                lambda _=0, b=box, k=key: (self.opts.__setitem__(k, b.currentData()),
                                           self.update_preview()))
        self.add_target("rows", "Rows:", min_items=0, extra=self.row_mode)
        self.add_target("cols", "Columns:", min_items=0, extra=self.col_mode)
        self.add_target("layers", "Layers:", min_items=0)
        for key in ("rows", "cols", "layers"):
            w = self.targets[key]
            w.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            w.customContextMenuRequested.connect(lambda pos, k=key: self._measure_menu(k, pos))
        self.add_button("Summary Statistics...", self._stats)
        self.add_button("Categories and Totals...", self._categories)
        self.add_button("Test Statistics...", self._tests)
        self.add_button("Titles...", self._titles)
        self.preview = QTextBrowser()
        self.preview.setMinimumHeight(220)
        self.status = QLabel()
        self.status.setStyleSheet("color:#8E1B1B;")
        self.options_layout.addWidget(QLabel("Preview:"))
        self.options_layout.addWidget(self.preview, 1)
        self.options_layout.addWidget(self.status)
        self.finish()
        self.update_preview()

    # ------------------------------------------------------------------ options
    def _measure_menu(self, key: str, pos) -> None:
        w = self.targets[key]
        item = w.itemAt(pos)
        if item is None:
            return
        name = item.data(NAME_ROLE)
        if name.startswith("$"):
            return
        current = ct.measure_of(self.ds, name)
        override = self.opts["scale_override"].get(name)
        is_scale = override if override is not None else current == "scale"
        menu = QMenu(self)
        act_scale = menu.addAction("Scale (summary statistics)")
        act_cat = menu.addAction("Categorical (categories)")
        act_scale.setCheckable(True)
        act_cat.setCheckable(True)
        act_scale.setChecked(is_scale)
        act_cat.setChecked(not is_scale)
        chosen = menu.exec(w.viewport().mapToGlobal(pos))
        if chosen is act_scale:
            self.opts["scale_override"][name] = True
        elif chosen is act_cat:
            self.opts["scale_override"][name] = False
        self.update_preview()

    def _stats(self) -> None:
        dlg = SubDialog("Custom Tables: Summary Statistics", self)
        box = dlg.group("Categorical Variables")
        for key, label in ct.CAT_STATS.items():
            dlg.check(box, f"c_{key}", label, key in self.opts["cat_stats"])
        box2 = dlg.group("Scale Variables")
        for key, label in ct.SCALE_STATS.items():
            dlg.check(box2, f"s_{key}", label, key in self.opts["scale_stats"])
        dlg.number(box2, "decimals", "Decimals (-1 = automatic):", self.opts["decimals"],
                   -1, 8, 0)
        values = dlg.run()
        if values is None:
            return
        cats = [k for k in ct.CAT_STATS if values.get(f"c_{k}")]
        scale = [k for k in ct.SCALE_STATS if values.get(f"s_{k}")]
        self.opts["cat_stats"] = cats or ["count"]
        self.opts["scale_stats"] = scale or ["mean"]
        self.opts["decimals"] = int(values["decimals"])
        self.update_preview()

    def _categories(self) -> None:
        dlg = SubDialog("Custom Tables: Categories and Totals", self, columns=1)
        box = dlg.group("Totals")
        dlg.check(box, "totals", "Show totals (and subtotals of nested variables)",
                  self.opts["totals"])
        box2 = dlg.group("Category Order")
        dlg.radio(box2, "order", [("ascending", "Ascending values"),
                                  ("descending", "Descending values"),
                                  ("count_desc", "Counts, descending"),
                                  ("count_asc", "Counts, ascending"),
                                  ("label", "Labels, alphabetical")], self.opts["order"])
        box3 = dlg.group("Empty Categories")
        dlg.check(box3, "show_empty", "Include labelled categories that have no cases",
                  self.opts["show_empty"])
        values = dlg.run()
        if values:
            self.opts.update(values)
            self.update_preview()

    def _tests(self) -> None:
        dlg = SubDialog("Custom Tables: Test Statistics", self, columns=1)
        box = dlg.group("Tests")
        dlg.check(box, "chisq", "Tests of independence (Chi-square)", self.opts["chisq"])
        dlg.check(box, "compare_props", "Compare column proportions (z-tests)",
                  self.opts["compare_props"])
        dlg.check(box, "compare_means", "Compare column means (t-tests)",
                  self.opts["compare_means"])
        box2 = dlg.group("Settings")
        dlg.number(box2, "alpha", "Significance level:", self.opts["alpha"], 0.001, 0.5, 3)
        dlg.radio(box2, "adjust", [("bonferroni", "Bonferroni adjustment"),
                                   ("none", "No adjustment")], self.opts["adjust"])
        values = dlg.run()
        if values:
            self.opts.update(values)
            self.update_preview()

    def _titles(self) -> None:
        dlg = SubDialog("Custom Tables: Titles", self, columns=1)
        box = dlg.group("Title")
        edit = QLineEdit(self.opts["title"])
        box.addWidget(edit)
        if dlg.exec():
            self.opts["title"] = edit.text().strip() or "Custom Table"
            self.update_preview()

    # ------------------------------------------------------------------ spec / preview
    def _ctvar(self, name: str) -> ct.CTVar:
        override = self.opts["scale_override"].get(name)
        return ct.CTVar(name, scale=override, order=self.opts["order"],
                        show_empty=self.opts["show_empty"])

    def spec(self) -> ct.CTSpec:
        o = self.opts
        return ct.CTSpec(rows=[self._ctvar(n) for n in self.selected("rows")],
                         cols=[self._ctvar(n) for n in self.selected("cols")],
                         layers=[self._ctvar(n) for n in self.selected("layers")],
                         row_mode=o["row_mode"], col_mode=o["col_mode"],
                         cat_stats=tuple(o["cat_stats"]), scale_stats=tuple(o["scale_stats"]),
                         totals=o["totals"], chisq=o["chisq"],
                         compare_props=o["compare_props"], compare_means=o["compare_means"],
                         alpha=float(o["alpha"]), adjust=o["adjust"], title=o["title"],
                         decimals=None if o["decimals"] < 0 else int(o["decimals"]))

    def targets_changed(self) -> None:
        if hasattr(self, "preview"):
            self.update_preview()

    def update_preview(self) -> None:
        if not hasattr(self, "preview"):
            return
        has = bool(self.selected("rows") or self.selected("cols"))
        if hasattr(self, "ok_button"):
            self.ok_button.setEnabled(has)
        if not has:
            self.preview.setHtml("<p style='color:#777'>Drag variables to Rows and Columns."
                                 "</p>")
            self.status.setText("")
            return
        try:
            view = self.ds
            if self.ds.n_cases > PREVIEW_CASES:
                from rsstatchem.core.dataset import Dataset as _D

                view = _D(self.ds.df.iloc[:PREVIEW_CASES].reset_index(drop=True),
                          self.ds.variables)
                view.meta = self.ds.meta
            spec = self.spec()
            spec.layers = spec.layers[:0]
            res = ct.compute(view, spec)
            table = ct.to_table(view, spec, res)
            self.preview.setHtml(f"<style>{report.css(0.9)}</style>"
                                 + report.table_html(table))
            self.status.setText("Layers are shown as separate tables in the output."
                                if self.selected("layers") else "")
        except Exception as exc:  # noqa: BLE001 - show any spec problem in the preview
            self.preview.setHtml("")
            self.status.setText(str(exc))

    def validate(self) -> str | None:
        if not (self.selected("rows") or self.selected("cols")):
            return "Put at least one variable in Rows or Columns."
        try:
            ct.compute(self.ds, self.spec())
        except ct.CTableError as exc:
            return str(exc)
        return None

    def compute(self):
        return ct.custom_tables(self.ds, self.spec())


MENUS = {"Analyze/Tables": [("Custom Tables...", CustomTablesDialog)]}


def selftest() -> None:
    rng = np.random.default_rng(5)
    n = 80
    ds = Dataset(pd.DataFrame({"sex": rng.integers(1, 3, n).astype(float),
                               "grp": rng.integers(1, 4, n).astype(float),
                               "score": rng.normal(60, 10, n)}))
    for name in ("sex", "grp"):
        ds.var(name).measure = "nominal"
    spec = ct.CTSpec(rows=["grp"], cols=["sex"], cat_stats=("count", "col_pct"), totals=True,
                     chisq=True, compare_props=True)
    res = ct.compute(ds, spec)
    want = pd.crosstab(ds.df.grp, ds.df.sex)
    assert res.values[(0, 0, "count")] == want.iloc[0, 0]
    sec = ct.custom_tables(ds, ct.CTSpec(rows=["score"], cols=["sex"],
                                         scale_stats=("mean", "std"), compare_means=True))
    assert sec.items[1].rows

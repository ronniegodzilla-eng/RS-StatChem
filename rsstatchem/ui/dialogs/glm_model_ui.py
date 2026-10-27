"""SPSS GLM sub-dialogs shared by Univariate, Multivariate and Repeated Measures:
Model…, Contrasts…, Plots… (profile plots), Post Hoc…, EM Means…, Save… and Options….

Each function edits ``owner.opts`` (a :class:`VariablePickerDialog`'s remembered options);
:func:`glm_options_from` turns those options into a :class:`glm_output.GLMOptions`.
"""
from __future__ import annotations

import itertools

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QDoubleSpinBox, QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMessageBox, QPushButton, QRadioButton,
                               QVBoxLayout, QWidget)

from rsstatchem.core.stats import posthoc_ext
from rsstatchem.core.stats.glm_model import (CONTRAST_NAMES, TermSpec, full_factorial,
                                             interactions_of, parse_terms)
from rsstatchem.core.stats.glm_output import GLMOptions
from rsstatchem.ui.dialogs.base import SubDialog, VariablePickerDialog

GLM_DEFAULTS = {
    "terms": None, "intercept": True, "ss_type": 3, "contrasts": {}, "plots": [],
    "plot_type": "line", "error_bars": None, "se_mult": 2.0, "ref_line": False,
    "y_zero": False, "posthoc": [], "posthoc_factors": [], "control": "last",
    "dunnett_test": "2-sided", "k_ratio": 100.0, "emm": [], "compare": False,
    "simple": False, "adjust": "lsd", "save": [], "descriptives": False, "effect": False,
    "power": False, "parameters": False, "contrast_coef": False, "homogeneity": False,
    "spread_level": False, "residual_plot": False, "lack_of_fit": False, "het": [],
    "robust": None, "alpha": 0.05,
}


def _buttons(dlg: QDialog, layout) -> None:
    bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                          QDialogButtonBox.StandardButton.Cancel)
    bb.button(QDialogButtonBox.StandardButton.Ok).setText("Continue")
    bb.accepted.connect(dlg.accept)
    bb.rejected.connect(dlg.reject)
    layout.addWidget(bb)


# ------------------------------------------------------------------ Model
class ModelDialog(QDialog):
    """Univariate: Model — Full factorial / Build terms (main effects, interactions, all n-way,
    nested terms), Type I–IV sums of squares and Include intercept."""

    KINDS = [("interaction", "Interaction"), ("main", "Main effects"), ("all2", "All 2-way"),
             ("all3", "All 3-way"), ("all4", "All 4-way"), ("all5", "All 5-way")]

    def __init__(self, title: str, factors: list[str], covariates: list[str],
                 terms: list[str] | None, intercept: bool, ss_type: int, parent=None,
                 random: list[str] | None = None):
        super().__init__(parent)
        self.setWindowTitle(f"{title}: Model")
        self.factors = list(factors) + list(random or [])
        self.covariates = list(covariates)
        self.full = QRadioButton("Full factorial")
        self.build = QRadioButton("Build terms")
        group = QButtonGroup(self)
        group.addButton(self.full)
        group.addButton(self.build)
        top = QHBoxLayout()
        top.addWidget(QLabel("Specify Model"))
        top.addWidget(self.full)
        top.addWidget(self.build)
        top.addStretch()

        self.source = QListWidget()
        self.source.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        for f in factors:
            self.source.addItem(QListWidgetItem(f"{f}(F)"))
        for r in random or []:
            self.source.addItem(QListWidgetItem(f"{r}(R)"))
        for c in covariates:
            self.source.addItem(QListWidgetItem(f"{c}(C)"))
        self.kind = QComboBox()
        for _, text in self.KINDS:
            self.kind.addItem(text)
        add = QPushButton("▶")
        add.setObjectName("arrow")
        add.clicked.connect(self._add)
        remove = QPushButton("◀")
        remove.setObjectName("arrow")
        remove.clicked.connect(self._remove)
        self.model = QListWidget()
        self.model.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        mid = QVBoxLayout()
        mid.addStretch()
        mid.addWidget(QLabel("Build Term(s)"))
        mid.addWidget(QLabel("Type:"))
        mid.addWidget(self.kind)
        mid.addWidget(add)
        mid.addWidget(remove)
        mid.addStretch()
        lcol = QVBoxLayout()
        lcol.addWidget(QLabel("Factors && Covariates:".replace("&&", "&")))
        lcol.addWidget(self.source)
        rcol = QVBoxLayout()
        rcol.addWidget(QLabel("Model:"))
        rcol.addWidget(self.model)
        lists = QHBoxLayout()
        lists.addLayout(lcol, 1)
        lists.addLayout(mid)
        lists.addLayout(rcol, 1)

        self.custom = QLineEdit()
        self.custom.setPlaceholderText("e.g. B(A), A*B*X")
        by = QPushButton("By *")
        by.clicked.connect(lambda: self.custom.insert("*"))
        within = QPushButton("Within")
        within.clicked.connect(self._within)
        add_custom = QPushButton("Add")
        add_custom.clicked.connect(self._add_custom)
        crow = QHBoxLayout()
        crow.addWidget(QLabel("Build custom term:"))
        crow.addWidget(self.custom, 1)
        crow.addWidget(by)
        crow.addWidget(within)
        crow.addWidget(add_custom)
        self.custom_hint = QLabel("Select variables on the left and click By / Within to "
                                  "insert them; nested terms such as Plot(Block) are allowed.")
        self.custom_hint.setStyleSheet("color:#666")
        self.custom_hint.setWordWrap(True)

        self.ss = QComboBox()
        for k, text in ((1, "Type I"), (2, "Type II"), (3, "Type III"), (4, "Type IV")):
            self.ss.addItem(text, k)
        self.ss.setCurrentIndex(int(ss_type) - 1)
        self.intercept = QCheckBox("Include intercept in model")
        self.intercept.setChecked(intercept)
        bottom = QHBoxLayout()
        bottom.addWidget(QLabel("Sum of squares:"))
        bottom.addWidget(self.ss)
        bottom.addStretch()
        bottom.addWidget(self.intercept)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        self.build_box = QWidget()
        bl = QVBoxLayout(self.build_box)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addLayout(lists, 1)
        bl.addLayout(crow)
        bl.addWidget(self.custom_hint)
        layout.addWidget(self.build_box, 1)
        layout.addLayout(bottom)
        _buttons(self, layout)
        self.full.toggled.connect(lambda on: self.build_box.setEnabled(not on))
        if terms:
            for t in terms:
                self.model.addItem(t)
            self.build.setChecked(True)
        else:
            self.full.setChecked(True)
            self.build_box.setEnabled(False)
        self.resize(640, 460)

    def _names(self) -> list[str]:
        return [i.text()[:-3] for i in self.source.selectedItems()]

    def _existing(self) -> set[str]:
        return {self.model.item(i).text() for i in range(self.model.count())}

    def _add(self) -> None:
        names = self._names()
        if not names:
            return
        kind = self.KINDS[self.kind.currentIndex()][0]
        for t in interactions_of(names, kind, set(self.factors)):
            text = t.syntax
            if text not in self._existing():
                self.model.addItem(text)

    def _remove(self) -> None:
        for item in self.model.selectedItems():
            self.model.takeItem(self.model.row(item))

    def _within(self) -> None:
        names = self._names()
        text = self.custom.text().strip()
        if names:
            self.custom.setText(f"{text}({'*'.join(names)})" if text else "")
        else:
            self.custom.insert("(")

    def _add_custom(self) -> None:
        text = self.custom.text().strip()
        if not text:
            names = self._names()
            text = "*".join(names)
        try:
            terms = parse_terms(text, self.factors, self.covariates)
        except ValueError as exc:
            QMessageBox.warning(self, self.windowTitle(), str(exc))
            return
        for t in terms:
            if t.syntax not in self._existing():
                self.model.addItem(t.syntax)
        self.custom.clear()

    def accept(self) -> None:
        if self.build.isChecked() and self.model.count() == 0 and not self.intercept.isChecked():
            QMessageBox.warning(self, self.windowTitle(), "The model has no terms.")
            return
        super().accept()

    def result_values(self) -> dict:
        terms = None if self.full.isChecked() else \
            [self.model.item(i).text() for i in range(self.model.count())]
        return {"terms": terms, "intercept": self.intercept.isChecked(),
                "ss_type": self.ss.currentData()}


def model_dialog(owner: VariablePickerDialog, title: str, factors: list[str],
                 covariates: list[str], random: list[str] | None = None) -> None:
    dlg = ModelDialog(title, factors, covariates, owner.opts.get("terms"),
                      owner.opts.get("intercept", True), owner.opts.get("ss_type", 3), owner,
                      random)
    if dlg.exec():
        owner.opts.update(dlg.result_values())


# ------------------------------------------------------------------ Contrasts
class ContrastsDialog(QDialog):
    KINDS = ["none", "deviation", "simple", "difference", "helmert", "repeated", "polynomial"]

    def __init__(self, title: str, factors: list[str], current: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{title}: Contrasts")
        self.values = {f: list(current.get(f, ("none", "last"))) for f in factors}
        self.list = QListWidget()
        self.factors = factors
        for f in factors:
            self.list.addItem(self._text(f))
        self.combo = QComboBox()
        for k in self.KINDS:
            self.combo.addItem("None" if k == "none" else CONTRAST_NAMES[k], k)
        self.last = QRadioButton("Last")
        self.first = QRadioButton("First")
        self.last.setChecked(True)
        change = QPushButton("Change")
        change.clicked.connect(self._change)
        self.list.currentRowChanged.connect(self._select)
        box = QGroupBox("Change Contrast")
        g = QGridLayout(box)
        g.addWidget(QLabel("Contrast:"), 0, 0)
        g.addWidget(self.combo, 0, 1)
        g.addWidget(change, 0, 2)
        g.addWidget(QLabel("Reference Category:"), 1, 0)
        g.addWidget(self.last, 1, 1)
        g.addWidget(self.first, 1, 2)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Factors:"))
        layout.addWidget(self.list)
        layout.addWidget(box)
        _buttons(self, layout)
        if factors:
            self.list.setCurrentRow(0)

    def _text(self, f: str) -> str:
        kind, ref = self.values[f]
        name = "None" if kind == "none" else CONTRAST_NAMES[kind]
        extra = f" ({ref})" if kind in ("simple", "deviation") else ""
        return f"{f}({name}{extra})"

    def _select(self, row: int) -> None:
        if row < 0:
            return
        kind, ref = self.values[self.factors[row]]
        self.combo.setCurrentIndex(self.KINDS.index(kind))
        (self.first if ref == "first" else self.last).setChecked(True)

    def _change(self) -> None:
        row = self.list.currentRow()
        if row < 0:
            return
        f = self.factors[row]
        self.values[f] = [self.combo.currentData(), "first" if self.first.isChecked() else "last"]
        self.list.item(row).setText(self._text(f))


def contrasts_dialog(owner: VariablePickerDialog, title: str, factors: list[str]) -> None:
    dlg = ContrastsDialog(title, factors, owner.opts.get("contrasts", {}), owner)
    if dlg.exec():
        owner.opts["contrasts"] = {f: v for f, v in dlg.values.items() if v[0] != "none"}


# ------------------------------------------------------------------ Profile plots
class PlotsDialog(QDialog):
    def __init__(self, title: str, factors: list[str], opts: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{title}: Profile Plots")
        self.factors = factors
        self.h, self.s, self.p = QComboBox(), QComboBox(), QComboBox()
        self.h.addItems(factors)
        for c in (self.s, self.p):
            c.addItems(["(none)"] + factors)
        form = QFormLayout()
        form.addRow("Horizontal Axis:", self.h)
        form.addRow("Separate Lines:", self.s)
        form.addRow("Separate Plots:", self.p)
        add, change, remove = QPushButton("Add"), QPushButton("Change"), QPushButton("Remove")
        add.clicked.connect(self._add)
        change.clicked.connect(self._change)
        remove.clicked.connect(lambda: self.list.takeItem(self.list.currentRow()))
        self.list = QListWidget()
        for spec in opts.get("plots", []):
            h, s, p = (list(spec) + [None, None])[:3]
            if h in factors and (not s or s in factors) and (not p or p in factors):
                self._item(h, s, p)
        brow = QHBoxLayout()
        for b in (add, change, remove):
            brow.addWidget(b)
        brow.addStretch()
        chart = QGroupBox("Chart Type:")
        cl = QHBoxLayout(chart)
        self.line = QRadioButton("Line Chart")
        self.bar = QRadioButton("Bar Chart")
        cl.addWidget(self.line)
        cl.addWidget(self.bar)
        (self.bar if opts.get("plot_type") == "bar" else self.line).setChecked(True)
        err = QGroupBox("Error Bars")
        el = QGridLayout(err)
        self.include = QCheckBox("Include Error bars")
        self.include.setChecked(bool(opts.get("error_bars")))
        self.ci = QRadioButton("Confidence Interval")
        self.se = QRadioButton("Standard Error")
        (self.se if opts.get("error_bars") == "se" else self.ci).setChecked(True)
        self.mult = QDoubleSpinBox()
        self.mult.setRange(0.1, 10)
        self.mult.setValue(float(opts.get("se_mult", 2.0)))
        el.addWidget(self.include, 0, 0, 1, 3)
        el.addWidget(self.ci, 1, 0, 1, 3)
        el.addWidget(self.se, 2, 0)
        el.addWidget(QLabel("Multiplier:"), 2, 1)
        el.addWidget(self.mult, 2, 2)
        self.ref = QCheckBox("Include reference line for grand mean")
        self.ref.setChecked(bool(opts.get("ref_line")))
        self.zero = QCheckBox("Y axis starts at 0")
        self.zero.setChecked(bool(opts.get("y_zero")))
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(brow)
        layout.addWidget(QLabel("Plots:"))
        layout.addWidget(self.list)
        row = QHBoxLayout()
        row.addWidget(chart)
        row.addWidget(err)
        layout.addLayout(row)
        layout.addWidget(self.ref)
        layout.addWidget(self.zero)
        _buttons(self, layout)

    def _spec(self):
        h = self.h.currentText()
        s = self.s.currentText()
        p = self.p.currentText()
        s = None if s in ("(none)", h) else s
        p = None if p in ("(none)", h, s) else p
        return h, s, p

    def _item(self, h, s, p) -> None:
        item = QListWidgetItem("*".join(x for x in (h, s, p) if x))
        item.setData(Qt.ItemDataRole.UserRole, (h, s, p))
        self.list.addItem(item)

    def _add(self) -> None:
        if self.h.currentText():
            self._item(*self._spec())

    def _change(self) -> None:
        row = self.list.currentRow()
        if row >= 0:
            self.list.takeItem(row)
            h, s, p = self._spec()
            item = QListWidgetItem("*".join(x for x in (h, s, p) if x))
            item.setData(Qt.ItemDataRole.UserRole, (h, s, p))
            self.list.insertItem(row, item)

    def result_values(self) -> dict:
        return {"plots": [list(self.list.item(i).data(Qt.ItemDataRole.UserRole))
                          for i in range(self.list.count())],
                "plot_type": "bar" if self.bar.isChecked() else "line",
                "error_bars": None if not self.include.isChecked() else
                ("se" if self.se.isChecked() else "ci"),
                "se_mult": self.mult.value(), "ref_line": self.ref.isChecked(),
                "y_zero": self.zero.isChecked()}


def plots_dialog(owner: VariablePickerDialog, title: str, factors: list[str]) -> None:
    dlg = PlotsDialog(title, factors, owner.opts, owner)
    if dlg.exec():
        owner.opts.update(dlg.result_values())


# ------------------------------------------------------------------ Post hoc
def posthoc_dialog(owner: VariablePickerDialog, title: str, factors: list[str] | None,
                   significance: bool = False) -> None:
    """SPSS Post Hoc Multiple Comparisons (One-Way ANOVA when ``factors`` is None)."""
    dlg = SubDialog(f"{title}: Post Hoc Multiple Comparisons" +
                    (" for Observed Means" if factors is not None else ""), owner, columns=2)
    if factors is not None:
        box = dlg.group("Post Hoc Tests for:", span=2)
        if not factors:
            box.addWidget(QLabel("(no factors)"))
        for f in factors:
            dlg.check(box, f"ph::{f}", f, f in owner.opts.get("posthoc_factors", []))
    current = owner.opts.get("posthoc", [])
    box = dlg.group("Equal Variances Assumed", span=2)
    grid = QGridLayout()
    for i, (key, text) in enumerate(posthoc_ext.EQUAL):
        cb = QCheckBox(text)
        cb.setChecked(key in current)
        dlg._getters[key] = cb.isChecked
        grid.addWidget(cb, i % 5, i // 5)
    box.addLayout(grid)
    wd = QHBoxLayout()
    wd.addWidget(QLabel("Waller-Duncan  Type I/Type II Error Ratio:"))
    ratio = QDoubleSpinBox()
    ratio.setRange(2, 1000)
    ratio.setDecimals(0)
    ratio.setValue(float(owner.opts.get("k_ratio", 100)))
    dlg._getters["k_ratio"] = ratio.value
    wd.addWidget(ratio)
    wd.addStretch()
    box.addLayout(wd)
    drow = QHBoxLayout()
    drow.addWidget(QLabel("Dunnett  Control Category:"))
    control = QComboBox()
    control.addItems(["Last", "First"])
    control.setCurrentIndex(1 if owner.opts.get("control") == "first" else 0)
    dlg._getters["control"] = lambda: "first" if control.currentIndex() == 1 else "last"
    drow.addWidget(control)
    drow.addWidget(QLabel("   Test:"))
    test_group = QButtonGroup(dlg)
    tests = {}
    for value, text in (("2-sided", "2-sided"), ("<", "< Control"), (">", "> Control")):
        rb = QRadioButton(text)
        rb.setChecked(owner.opts.get("dunnett_test", "2-sided") == value)
        test_group.addButton(rb)
        drow.addWidget(rb)
        tests[value] = rb
    dlg._getters["dunnett_test"] = lambda: next(v for v, b in tests.items() if b.isChecked())
    drow.addStretch()
    box.addLayout(drow)
    box = dlg.group("Equal Variances Not Assumed", span=2)
    row = QHBoxLayout()
    for key, text in posthoc_ext.UNEQUAL:
        cb = QCheckBox(text)
        cb.setChecked(key in current)
        dlg._getters[key] = cb.isChecked
        row.addWidget(cb)
    row.addStretch()
    box.addLayout(row)
    if significance:
        box = dlg.group("Null hypothesis test", span=2)
        dlg.number(box, "alpha", "Significance level:", owner.opts.get("alpha", 0.05), 0.001,
                   0.5, 3)
    values = dlg.run()
    if values is None:
        return
    owner.opts["posthoc"] = [k for k in posthoc_ext.ORDER if values.get(k)]
    for key in ("k_ratio", "control", "dunnett_test"):
        owner.opts[key] = values[key]
    if significance:
        owner.opts["alpha"] = values["alpha"]
    if factors is not None:
        owner.opts["posthoc_factors"] = [f for f in factors if values.get(f"ph::{f}")]


# ------------------------------------------------------------------ EM Means
def effects_of(factors: list[str], terms: list[str] | None = None) -> list[str]:
    if terms:
        out = []
        for t in terms:
            parts = [p for p in t.replace("(", "*").replace(")", "").split("*") if p]
            if all(p in factors for p in parts):
                out.append("*".join(parts))
        return out
    return ["*".join(c) for r in range(1, len(factors) + 1)
            for c in itertools.combinations(factors, r)]


def emm_dialog(owner: VariablePickerDialog, title: str, factors: list[str],
               terms: list[str] | None = None) -> None:
    dlg = SubDialog(f"{title}: Estimated Marginal Means", owner, columns=1)
    box = dlg.group("Display Means for:")
    current = owner.opts.get("emm", [])
    for eff in ["(OVERALL)"] + effects_of(factors, terms):
        dlg.check(box, f"emm::{eff}", eff.replace("*", " * "), eff in current)
    box = dlg.group("Comparisons")
    dlg.check(box, "compare", "Compare main effects", owner.opts.get("compare", False))
    dlg.check(box, "simple", "Compare simple main effects", owner.opts.get("simple", False))
    row = QHBoxLayout()
    row.addWidget(QLabel("Confidence interval adjustment:"))
    combo = QComboBox()
    choices = [("lsd", "LSD(none)"), ("bonferroni", "Bonferroni"), ("sidak", "Sidak")]
    for k, text in choices:
        combo.addItem(text, k)
    combo.setCurrentIndex([k for k, _ in choices].index(owner.opts.get("adjust", "lsd")))
    dlg._getters["adjust"] = combo.currentData
    row.addWidget(combo)
    box.addLayout(row)
    values = dlg.run()
    if values is None:
        return
    owner.opts["emm"] = [k.split("::", 1)[1] for k, v in values.items()
                         if k.startswith("emm::") and v]
    for key in ("compare", "simple", "adjust"):
        owner.opts[key] = values[key]


# ------------------------------------------------------------------ Save / Options
SAVE_ITEMS = {
    "Predicted Values": [("PRE", "Unstandardized"), ("WPR", "Weighted"),
                         ("SEP", "Standard error")],
    "Diagnostics": [("COO", "Cook's distance"), ("LEV", "Leverage values")],
    "Residuals": [("RES", "Unstandardized"), ("WRE", "Weighted"), ("ZRE", "Standardized"),
                  ("SRE", "Studentized"), ("DRE", "Deleted")],
}


def save_dialog(owner: VariablePickerDialog, title: str) -> None:
    dlg = SubDialog(f"{title}: Save", owner, columns=2)
    current = owner.opts.get("save", [])
    for group, items in SAVE_ITEMS.items():
        box = dlg.group(group)
        for key, text in items:
            dlg.check(box, f"save::{key}", text, key in current)
    values = dlg.run()
    if values is not None:
        owner.opts["save"] = [k.split("::", 1)[1] for k, v in values.items() if v]


def options_dialog(owner: VariablePickerDialog, title: str, multivariate: bool = False) -> None:
    o = owner.opts
    dlg = SubDialog(f"{title}: Options", owner, columns=2)
    box = dlg.group("Display", span=2)
    grid = QGridLayout()
    items = [("descriptives", "Descriptive statistics"), ("effect", "Estimates of effect size"),
             ("power", "Observed power"), ("parameters", "Parameter estimates"),
             ("contrast_coef", "Contrast coefficient matrix"),
             ("homogeneity", "Homogeneity tests"), ("spread_level", "Spread vs. level plot"),
             ("residual_plot", "Residual plot"), ("lack_of_fit", "Lack of fit")]
    for i, (key, text) in enumerate(items):
        cb = QCheckBox(text)
        cb.setChecked(bool(o.get(key)))
        dlg._getters[key] = cb.isChecked
        grid.addWidget(cb, i % 5, i // 5)
    box.addLayout(grid)
    if multivariate:
        cb = QCheckBox("Residual SSCP matrix (with Bartlett's test of sphericity)")
        cb.setChecked(bool(o.get("residual_sscp")))
        dlg._getters["residual_sscp"] = cb.isChecked
        box.addWidget(cb)
    if not multivariate:
        box = dlg.group("Heteroskedasticity Tests")
        for key, text in (("mbp", "Modified Breusch-Pagan test"), ("f", "F test"),
                          ("bp", "Breusch-Pagan test"), ("white", "White's test")):
            dlg.check(box, f"het::{key}", text, key in o.get("het", []))
        box = dlg.group("Parameter estimates with robust standard errors")
        dlg.check(box, "robust_on", "Use robust standard errors", bool(o.get("robust")))
        dlg.radio(box, "robust_type", [(k, k) for k in ("HC0", "HC1", "HC2", "HC3", "HC4")],
                  o.get("robust") or "HC3")
    box = dlg.group("Significance level", span=2)
    dlg.number(box, "alpha", "Significance level:", o.get("alpha", 0.05), 0.001, 0.5, 3)
    box.addWidget(QLabel("Confidence intervals are (1 − significance level) × 100%."))
    values = dlg.run()
    if values is None:
        return
    for key, _ in items:
        o[key] = values[key]
    o["alpha"] = values["alpha"]
    if multivariate:
        o["residual_sscp"] = values["residual_sscp"]
    if not multivariate:
        o["het"] = [k for k in ("mbp", "f", "bp", "white") if values.get(f"het::{k}")]
        o["robust"] = values["robust_type"] if values["robust_on"] else None


# ------------------------------------------------------------------ conversion
def glm_options_from(o: dict, factors: list[str], covariates: list[str],
                     random: list[str] | None = None) -> GLMOptions:
    """Translate dialog options (variable names as typed) into GLMOptions."""
    random = random or []
    names = list(factors) + list(random)
    terms = None
    if o.get("terms"):
        terms = []
        for text in o["terms"]:
            try:
                terms += parse_terms(text, names, covariates)
            except ValueError:
                continue
    plots = [tuple(p) for p in o.get("plots", [])
             if p[0] in names and all(x is None or x in names for x in p[1:])]
    emm = [e for e in o.get("emm", []) if e == "(OVERALL)" or
           all(part in names for part in e.split("*"))]
    return GLMOptions(
        terms=terms, intercept=o.get("intercept", True), ss_type=o.get("ss_type", 3),
        random=list(random), contrasts={f: tuple(v) for f, v in o.get("contrasts", {}).items()
                                        if f in names},
        emm=emm, compare=o.get("compare", False), simple=o.get("simple", False),
        adjust=o.get("adjust", "lsd"), plots=plots, plot_type=o.get("plot_type", "line"),
        error_bars=o.get("error_bars"), se_mult=o.get("se_mult", 2.0),
        ref_line=o.get("ref_line", False), y_zero=o.get("y_zero", False),
        save=list(o.get("save", [])), descriptives=o.get("descriptives", False),
        effect_size=o.get("effect", False), power=o.get("power", False),
        parameters=o.get("parameters", False), contrast_coef=o.get("contrast_coef", False),
        homogeneity=o.get("homogeneity", False), spread_level=o.get("spread_level", False),
        residual_plot=o.get("residual_plot", False), lack_of_fit=o.get("lack_of_fit", False),
        het=list(o.get("het", [])), robust=o.get("robust"), alpha=o.get("alpha", 0.05),
        residual_sscp=o.get("residual_sscp", False),
        posthoc_factors=[f for f in o.get("posthoc_factors", []) if f in factors],
        posthoc_methods=list(o.get("posthoc", [])),
        posthoc=posthoc_ext.PostHocOptions(1 - o.get("alpha", 0.05), o.get("control", "last"),
                                           o.get("dunnett_test", "2-sided"),
                                           o.get("k_ratio", 100.0)))


__all__ = ["GLM_DEFAULTS", "model_dialog", "contrasts_dialog", "plots_dialog",
           "posthoc_dialog", "emm_dialog", "save_dialog", "options_dialog",
           "glm_options_from", "ModelDialog", "ContrastsDialog", "PlotsDialog",
           "TermSpec", "full_factorial", "parse_terms"]

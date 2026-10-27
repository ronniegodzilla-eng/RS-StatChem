"""Analyze > Nonparametric Tests > Legacy Dialogs (SPSS NPAR TESTS dialogs).

Each dialog has the SPSS *Options…* button (Descriptive / Quartiles statistics, Missing
Values test-by-test or listwise) and the test-type check boxes of SPSS.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QFormLayout, QGridLayout, QGroupBox,
                               QHBoxLayout, QLineEdit, QRadioButton, QVBoxLayout)

from rsstatchem.core.output import Section
from rsstatchem.core.stats import nonparametric as npt
from rsstatchem.ui.dialogs.base import SubDialog, VariablePickerDialog, is_numeric
from rsstatchem.ui.dialogs.compare import GroupingMixin

OPTION_DEFAULTS = {"descriptives": False, "quartiles": False, "listwise": False}


def _checks(dialog: VariablePickerDialog, title: str, items: list[tuple[str, str]],
            columns: int | None = None) -> QGroupBox:
    """Test-type check boxes laid out like SPSS (a 2 x 2 grid for four tests)."""
    box = QGroupBox(title)
    grid = QGridLayout(box)
    columns = columns or (2 if len(items) == 4 else len(items))
    dialog._check_widgets = getattr(dialog, "_check_widgets", {})
    for i, (key, text) in enumerate(items):
        cb = QCheckBox(text)
        cb.setChecked(bool(dialog.opts.get(key)))
        cb.toggled.connect(lambda on, k=key: dialog.opts.__setitem__(k, on))
        grid.addWidget(cb, i // columns, i % columns)
        dialog._check_widgets[key] = cb
    grid.setColumnStretch(columns, 1)
    dialog.options_layout.addWidget(box)
    return box


def _numbers(text: str) -> list[float]:
    return [float(p.replace(",", ".")) for p in text.replace(";", " ").split() if p.strip()]


def _number(text: str) -> float:
    return float(text.strip().replace(",", "."))


def merge(sections: list[Section]) -> Section:
    merged = Section(sections[0].title)
    for s in sections:
        merged.items += s.items
    return merged


class NparOptionsMixin:
    """The Options… sub-dialog shared by all legacy nonparametric dialogs."""

    def add_npar_options(self: VariablePickerDialog) -> None:
        self.add_option_button("Options...", self._npar_options)

    def _npar_options(self: VariablePickerDialog) -> None:
        dlg = SubDialog(f"{self.windowTitle()}: Options", self, columns=1)
        box = dlg.group("Statistics")
        dlg.check(box, "descriptives", "Descriptive", self.opts.get("descriptives", False))
        dlg.check(box, "quartiles", "Quartiles", self.opts.get("quartiles", False))
        box = dlg.group("Missing Values")
        dlg.radio(box, "listwise", [(False, "Exclude cases test-by-test"),
                                    (True, "Exclude cases listwise")],
                  self.opts.get("listwise", False))
        values = dlg.run()
        if values:
            self.opts.update(values)

    def npar_kwargs(self: VariablePickerDialog) -> dict:
        return {k: bool(self.opts.get(k, False)) for k in OPTION_DEFAULTS}


class ChiSquareDialog(NparOptionsMixin, VariablePickerDialog):
    DEFAULTS = {"expected": "", "use_range": False, "lower": "", "upper": "",
                "equal": True, **OPTION_DEFAULTS}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Chi-square Test", parent)
        self.add_target("vars", "Test Variable List:")
        self.add_npar_options()
        o = self.opts

        range_box = QGroupBox("Expected Range")
        rl = QVBoxLayout(range_box)
        group = QButtonGroup(self)
        self.from_data = QRadioButton("Get from data")
        self.use_range = QRadioButton("Use specified range")
        for rb in (self.from_data, self.use_range):
            group.addButton(rb)
            rl.addWidget(rb)
        (self.use_range if o["use_range"] else self.from_data).setChecked(True)
        self.lower, self.upper = QLineEdit(o["lower"]), QLineEdit(o["upper"])
        form = QFormLayout()
        form.addRow("Lower:", self.lower)
        form.addRow("Upper:", self.upper)
        rl.addLayout(form)

        exp_box = QGroupBox("Expected Values")
        el = QVBoxLayout(exp_box)
        group2 = QButtonGroup(self)
        self.all_equal = QRadioButton("All categories equal")
        self.use_values = QRadioButton("Values:")
        for rb in (self.all_equal, self.use_values):
            group2.addButton(rb)
            el.addWidget(rb)
        (self.all_equal if o["equal"] and not o["expected"] else self.use_values).setChecked(True)
        self.expected = QLineEdit(o["expected"])
        self.expected.setPlaceholderText("e.g. 1 2 1 (one value per category, in order)")
        el.addWidget(self.expected)
        row = QHBoxLayout()
        row.addWidget(range_box)
        row.addWidget(exp_box)
        self.options_layout.addLayout(row)
        self.finish()

    def validate(self):
        try:
            values = _numbers(self.expected.text()) if self.use_values.isChecked() else []
            if self.use_values.isChecked() and not values:
                return "Enter the expected values, or choose All categories equal."
            if any(v <= 0 for v in values):
                return "Expected values must be greater than 0."
            if self.use_range.isChecked():
                lo, hi = _number(self.lower.text()), _number(self.upper.text())
                if lo >= hi:
                    return "The upper bound must be greater than the lower bound."
        except ValueError:
            return "Expected values and the range bounds must be numbers."
        self.opts.update(expected=self.expected.text() if self.use_values.isChecked() else "",
                         equal=self.all_equal.isChecked(), use_range=self.use_range.isChecked(),
                         lower=self.lower.text(), upper=self.upper.text())
        return None

    def compute(self):
        o = self.opts
        rng = (_number(o["lower"]), _number(o["upper"])) if o["use_range"] else None
        return npt.chi_square(self.ds, self.selected("vars"), _numbers(o["expected"]) or None,
                              expected_range=rng, **self.npar_kwargs())


class BinomialDialog(NparOptionsMixin, VariablePickerDialog):
    DEFAULTS = {"prop": "0.50", "cut": "", **OPTION_DEFAULTS}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Binomial Test", parent)
        self.add_target("vars", "Test Variable List:")
        self.add_npar_options()
        self.prop = QLineEdit(self.opts["prop"])
        self.cut = QLineEdit(self.opts["cut"])
        self.cut.setPlaceholderText("Blank = get from data (dichotomy)")
        form = QFormLayout()
        form.addRow("Test Proportion:", self.prop)
        form.addRow("Cut point:", self.cut)
        self.options_layout.addLayout(form)
        self.finish()

    def validate(self):
        try:
            p = float(self.prop.text().replace(",", "."))
            if not 0 < p < 1:
                raise ValueError
            if self.cut.text().strip():
                float(self.cut.text().replace(",", "."))
        except ValueError:
            return "Test proportion must be between 0 and 1, and the cut point a number."
        self.opts.update(prop=self.prop.text(), cut=self.cut.text())
        return None

    def compute(self):
        cut = self.opts["cut"].strip()
        return npt.binomial(self.ds, self.selected("vars"),
                            float(self.opts["prop"].replace(",", ".")),
                            float(cut.replace(",", ".")) if cut else None, **self.npar_kwargs())


class RunsDialog(NparOptionsMixin, VariablePickerDialog):
    DEFAULTS = {"median": True, "mode": False, "mean": False, "custom": False, "value": "",
                **OPTION_DEFAULTS}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Runs Test", parent)
        self.add_target("vars", "Test Variable List:", accepts=is_numeric)
        self.add_npar_options()
        box = _checks(self, "Cut Point", [("median", "Median"), ("mode", "Mode"),
                                         ("mean", "Mean"), ("custom", "Custom:")], columns=1)
        self.value = QLineEdit(self.opts["value"])
        self.value.setMaximumWidth(90)
        box.layout().addWidget(self.value, 3, 1)
        box.layout().setColumnStretch(1, 0)
        box.layout().setColumnStretch(2, 1)
        self.finish()

    def validate(self):
        if self.opts["custom"]:
            try:
                _number(self.value.text())
            except ValueError:
                return "Enter a numeric custom cut point."
        self.opts["value"] = self.value.text()
        return None

    def compute(self):
        names = self.selected("vars")
        kw = self.npar_kwargs()
        sections = []
        for cut in ("median", "mode", "mean", "custom"):
            if self.opts[cut]:
                value = _number(self.opts["value"]) if cut == "custom" else None
                sections.append(npt.runs(self.ds, names, cut, value, **kw))
                kw["descriptives"] = kw["quartiles"] = False
        return merge(sections) if sections else npt.runs(self.ds, names, "median", **kw)


class OneSampleKSDialog(NparOptionsMixin, VariablePickerDialog):
    DEFAULTS = {"normal": True, "uniform": False, "poisson": False, "exponential": False,
                **OPTION_DEFAULTS}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "One-Sample Kolmogorov-Smirnov Test", parent)
        self.add_target("vars", "Test Variable List:", accepts=is_numeric)
        self.add_npar_options()
        _checks(self, "Test Distribution", [("normal", "Normal"), ("uniform", "Uniform"),
                                            ("poisson", "Poisson"),
                                            ("exponential", "Exponential")])
        self.finish()

    def validate(self):
        return None if any(self.opts[d] for d in npt.DISTRIBUTIONS) else \
            "Select a test distribution."

    def compute(self):
        kw = self.npar_kwargs()
        sections = []
        for dist in npt.DISTRIBUTIONS:
            if self.opts[dist]:
                sections.append(npt.ks_one_sample(self.ds, self.selected("vars"), dist, **kw))
                kw["descriptives"] = kw["quartiles"] = False
        return merge(sections)


class TwoIndependentDialog(NparOptionsMixin, GroupingMixin, VariablePickerDialog):
    DEFAULTS = {"group_def": None, "mw": True, "ks": False, "moses": False, "ww": False,
                **OPTION_DEFAULTS}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Two-Independent-Samples Tests", parent)
        self.add_target("vars", "Test Variable List:", accepts=is_numeric)
        self.setup_grouping(allow_cut=False)
        self.add_npar_options()
        _checks(self, "Test Type", [("mw", "Mann-Whitney U"), ("ks", "Kolmogorov-Smirnov Z"),
                                    ("moses", "Moses extreme reactions"),
                                    ("ww", "Wald-Wolfowitz runs")])
        self.finish()

    def validate(self):
        o = self.opts
        if not (o["mw"] or o["ks"] or o["moses"] or o["ww"]):
            return "Select a test type."
        return GroupingMixin.validate(self)

    def compute(self):
        o = self.opts
        return npt.two_independent(self.ds, self.selected("vars"), self.selected_one("group"),
                                   self._definition()["groups"], o["mw"], o["ks"], o["moses"],
                                   o["ww"], **self.npar_kwargs())


class KIndependentDialog(NparOptionsMixin, GroupingMixin, VariablePickerDialog):
    DEFAULTS = {"group_def": None, "kw": True, "median": False, "jt": False, **OPTION_DEFAULTS}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Tests for Several Independent Samples", parent)
        self.add_target("vars", "Test Variable List:", accepts=is_numeric)
        self.setup_grouping(use_range=True)
        self.add_npar_options()
        _checks(self, "Test Type", [("kw", "Kruskal-Wallis H"), ("median", "Median"),
                                    ("jt", "Jonckheere-Terpstra")])
        self.finish()

    def validate(self):
        o = self.opts
        if not (o["kw"] or o["median"] or o["jt"]):
            return "Select a test type."
        return GroupingMixin.validate(self)

    def compute(self):
        d, o = self._definition(), self.opts
        return npt.k_independent(self.ds, self.selected("vars"), self.selected_one("group"),
                                 d["lo"], d["hi"], o["kw"], o["median"], o["jt"],
                                 **self.npar_kwargs())


class TwoRelatedDialog(NparOptionsMixin, VariablePickerDialog):
    DEFAULTS = {"wilcoxon": True, "sign": False, "mcnemar": False, "marginal": False,
                **OPTION_DEFAULTS}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Two-Related-Samples Tests", parent)
        self.add_pair_target("pairs", "Test Pairs (Variable1 – Variable2):")
        self.add_npar_options()
        _checks(self, "Test Type", [("wilcoxon", "Wilcoxon"), ("sign", "Sign"),
                                    ("mcnemar", "McNemar"),
                                    ("marginal", "Marginal Homogeneity")])
        self.finish()

    def validate(self):
        o = self.opts
        return None if o["wilcoxon"] or o["sign"] or o["mcnemar"] or o["marginal"] else \
            "Select a test type."

    def compute(self):
        o = self.opts
        return npt.two_related(self.ds, self.pairs("pairs"), o["wilcoxon"], o["sign"],
                               o["mcnemar"], o["marginal"], **self.npar_kwargs())


class KRelatedDialog(NparOptionsMixin, VariablePickerDialog):
    DEFAULTS = {"friedman": True, "kendall": False, "cochran": False, **OPTION_DEFAULTS}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Tests for Several Related Samples", parent)
        self.add_target("vars", "Test Variables:", accepts=is_numeric, min_items=2)
        self.add_npar_options()
        _checks(self, "Test Type", [("friedman", "Friedman"), ("kendall", "Kendall's W"),
                                    ("cochran", "Cochran's Q")])
        self.finish()

    def validate(self):
        o = self.opts
        return None if o["friedman"] or o["kendall"] or o["cochran"] else "Select a test type."

    def compute(self):
        o = self.opts
        return npt.k_related(self.ds, self.selected("vars"), o["friedman"], o["kendall"],
                             o["cochran"], **self.npar_kwargs())

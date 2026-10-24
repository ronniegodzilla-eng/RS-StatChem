"""Analyze > Nonparametric Tests > Legacy Dialogs."""
from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QFormLayout, QGroupBox, QHBoxLayout, QLineEdit

from rsstatchem.core.stats import nonparametric as npt
from rsstatchem.ui.dialogs.base import VariablePickerDialog, is_numeric
from rsstatchem.ui.dialogs.compare import GroupingMixin


def _checks(dialog: VariablePickerDialog, title: str, items: list[tuple[str, str]]) -> None:
    box = QGroupBox(title)
    row = QHBoxLayout(box)
    for key, text in items:
        cb = QCheckBox(text)
        cb.setChecked(bool(dialog.opts.get(key)))
        cb.toggled.connect(lambda on, k=key: dialog.opts.__setitem__(k, on))
        row.addWidget(cb)
    row.addStretch()
    dialog.options_layout.addWidget(box)


def _numbers(text: str) -> list[float]:
    return [float(p.replace(",", ".")) for p in text.replace(";", " ").split() if p.strip()]


class ChiSquareDialog(VariablePickerDialog):
    DEFAULTS = {"expected": ""}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Chi-square Test", parent)
        self.add_target("vars", "Test Variable List:")
        self.expected = QLineEdit(self.opts["expected"])
        self.expected.setPlaceholderText("Blank = all categories equal; or e.g. 1 2 1")
        form = QFormLayout()
        form.addRow("Expected Values:", self.expected)
        self.options_layout.addLayout(form)
        self.finish()

    def validate(self):
        try:
            _numbers(self.expected.text())
        except ValueError:
            return "Expected values must be numbers separated by spaces."
        self.opts["expected"] = self.expected.text()
        return None

    def compute(self):
        return npt.chi_square(self.ds, self.selected("vars"),
                              _numbers(self.opts["expected"]) or None)


class BinomialDialog(VariablePickerDialog):
    DEFAULTS = {"prop": "0.50", "cut": ""}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Binomial Test", parent)
        self.add_target("vars", "Test Variable List:")
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
                            float(cut.replace(",", ".")) if cut else None)


class RunsDialog(VariablePickerDialog):
    DEFAULTS = {"median": True, "mean": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Runs Test", parent)
        self.add_target("vars", "Test Variable List:", accepts=is_numeric)
        _checks(self, "Cut Point", [("median", "Median"), ("mean", "Mean")])
        self.finish()

    def compute(self):
        from rsstatchem.core.output import Section

        names = self.selected("vars")
        sections = [npt.runs(self.ds, names, c) for c, on in
                    (("median", self.opts["median"]), ("mean", self.opts["mean"])) if on]
        if not sections:
            return npt.runs(self.ds, names, "median")
        merged = Section(sections[0].title)
        for s in sections:
            merged.items += s.items
        return merged


class OneSampleKSDialog(VariablePickerDialog):
    def __init__(self, ds, parent=None):
        super().__init__(ds, "One-Sample Kolmogorov-Smirnov Test", parent)
        self.add_target("vars", "Test Variable List:", accepts=is_numeric)
        self.finish()

    def compute(self):
        return npt.ks_one_sample(self.ds, self.selected("vars"))


class TwoIndependentDialog(GroupingMixin, VariablePickerDialog):
    DEFAULTS = {"group_def": None}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Two-Independent-Samples Tests (Mann-Whitney U)", parent)
        self.add_target("vars", "Test Variable List:", accepts=is_numeric)
        self.setup_grouping(allow_cut=False)
        self.finish()

    def compute(self):
        return npt.mann_whitney(self.ds, self.selected("vars"), self.selected_one("group"),
                                self._definition()["groups"])


class KIndependentDialog(GroupingMixin, VariablePickerDialog):
    DEFAULTS = {"group_def": None}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Tests for Several Independent Samples (Kruskal-Wallis H)", parent)
        self.add_target("vars", "Test Variable List:", accepts=is_numeric)
        self.setup_grouping(use_range=True)
        self.finish()

    def compute(self):
        d = self._definition()
        return npt.kruskal_wallis(self.ds, self.selected("vars"), self.selected_one("group"),
                                  d["lo"], d["hi"])


class TwoRelatedDialog(VariablePickerDialog):
    DEFAULTS = {"wilcoxon": True, "sign": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Two-Related-Samples Tests", parent)
        self.add_pair_target("pairs", "Test Pairs (Variable1 – Variable2):")
        _checks(self, "Test Type", [("wilcoxon", "Wilcoxon"), ("sign", "Sign")])
        self.finish()

    def validate(self):
        return None if self.opts["wilcoxon"] or self.opts["sign"] else "Select a test type."

    def compute(self):
        return npt.two_related(self.ds, self.pairs("pairs"), self.opts["wilcoxon"],
                               self.opts["sign"])


class KRelatedDialog(VariablePickerDialog):
    DEFAULTS = {"friedman": True, "kendall": False, "cochran": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Tests for Several Related Samples", parent)
        self.add_target("vars", "Test Variables:", accepts=is_numeric, min_items=2)
        _checks(self, "Test Type", [("friedman", "Friedman"), ("kendall", "Kendall's W"),
                                    ("cochran", "Cochran's Q")])
        self.finish()

    def validate(self):
        o = self.opts
        return None if o["friedman"] or o["kendall"] or o["cochran"] else "Select a test type."

    def compute(self):
        o = self.opts
        return npt.k_related(self.ds, self.selected("vars"), o["friedman"], o["kendall"],
                             o["cochran"])

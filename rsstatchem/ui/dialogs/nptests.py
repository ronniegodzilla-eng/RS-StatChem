"""Analyze > Nonparametric Tests > One Sample / Independent Samples / Related Samples.

Mirrors the SPSS NPTESTS dialogs: an *Objective* tab (automatic comparison or a customised
analysis), a *Fields* tab (test fields, groups) and a *Settings* tab (automatic test choice by
measurement level or custom tests, plus Test Options). Output starts with the SPSS
"Hypothesis Test Summary" followed by the per-test summary tables, pairwise comparisons and
charts that the SPSS model viewer shows.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
                               QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QRadioButton, QSpinBox, QTabWidget, QVBoxLayout, QWidget)

from rsstatchem.core.stats import nptests
from rsstatchem.ui.dialogs.base import VariablePickerDialog, is_numeric


class _NpTestsDialog(VariablePickerDialog):
    """Shared tabbed layout. Subclasses define OBJECTIVES, TESTS and the field targets."""

    OBJECTIVES: list[tuple[str, str, str]] = []
    TESTS: list[tuple[str, str]] = []
    BASE_DEFAULTS = {"objective": "auto", "mode": "auto", "alpha": 0.05, "level": 95.0,
                     "listwise": False, "multiple": "all"}

    def __init__(self, ds, title: str, parent=None):
        super().__init__(ds, title, parent)
        self.setMinimumSize(780, 560)
        self.build_fields()
        self._make_tabs()
        self.finish()

    # ------------------------------------------------------------------ layout
    def build_fields(self) -> None:
        raise NotImplementedError

    def _make_tabs(self) -> None:
        layout = self.layout()
        body = layout.takeAt(0).layout()
        self.tabs = QTabWidget()
        self.tabs.addTab(self._objective_page(), "Objective")
        fields = QWidget()
        fl = QVBoxLayout(fields)
        fl.addWidget(QLabel("Use custom field assignments"))
        fl.addLayout(body, 1)
        self.tabs.addTab(fields, "Fields")
        self.tabs.addTab(self._settings_page(), "Settings")
        layout.insertWidget(0, self.tabs, 1)

    def _objective_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        intro = QLabel("Identifies differences in single fields using one or more "
                             "nonparametric tests. Nonparametric tests do not assume your "
                             "data follow the normal distribution."
                             if isinstance(self, OneSampleDialog) else
                             "Identifies differences between two or more groups using "
                             "nonparametric tests. Nonparametric tests do not assume your "
                             "data follow the normal distribution."
                             if isinstance(self, IndependentSamplesDialog) else
                             "Identifies differences between two or more related fields "
                             "using nonparametric tests. Nonparametric tests do not assume "
                             "your data follow the normal distribution.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        box = QGroupBox("What is your objective?")
        bl = QVBoxLayout(box)
        group = QButtonGroup(self)
        self.objective_buttons = {}
        for key, text, _ in self.OBJECTIVES:
            rb = QRadioButton(text)
            rb.setChecked(key == self.opts["objective"])
            group.addButton(rb)
            bl.addWidget(rb)
            self.objective_buttons[key] = rb
            rb.toggled.connect(lambda on, k=key: on and self._objective_changed(k))
        lay.addWidget(box)
        desc_box = QGroupBox("Description")
        dl = QVBoxLayout(desc_box)
        self.description = QLabel()
        self.description.setWordWrap(True)
        dl.addWidget(self.description)
        lay.addWidget(desc_box)
        lay.addStretch()
        self._update_description()
        return page

    def _settings_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        choose = QGroupBox("Choose Tests")
        cl = QVBoxLayout(choose)
        group = QButtonGroup(self)
        self.auto_radio = QRadioButton("Automatically choose the tests based on the data")
        self.custom_radio = QRadioButton("Customize tests")
        for rb in (self.auto_radio, self.custom_radio):
            group.addButton(rb)
            cl.addWidget(rb)
        (self.auto_radio if self.opts["mode"] == "auto" else self.custom_radio).setChecked(True)
        self.test_boxes: dict[str, QCheckBox] = {}
        grid = QGridLayout()
        grid.setContentsMargins(20, 0, 0, 0)
        for i, (key, text) in enumerate(self.TESTS):
            cb = QCheckBox(text)
            cb.setChecked(key in self.opts["tests"])
            self.test_boxes[key] = cb
            grid.addWidget(cb, i // self.TEST_COLUMNS, i % self.TEST_COLUMNS)
        cl.addLayout(grid)
        self.extra_settings(cl)
        cl.addStretch()
        self.custom_radio.toggled.connect(self._enable_tests)
        self._enable_tests(self.custom_radio.isChecked())
        lay.addWidget(choose, 1)

        options = QGroupBox("Test Options")
        oh = QHBoxLayout(options)
        ol = QFormLayout()
        self.alpha = QDoubleSpinBox()
        self.alpha.setRange(0.0001, 0.9999)
        self.alpha.setDecimals(4)
        self.alpha.setSingleStep(0.01)
        self.alpha.setValue(float(self.opts["alpha"]))
        self.level_spin = QDoubleSpinBox()
        self.level_spin.setRange(50, 99.99)
        self.level_spin.setDecimals(2)
        self.level_spin.setSuffix(" %")
        self.level_spin.setValue(float(self.opts["level"]))
        ol.addRow("Significance level:", self.alpha)
        ol.addRow("Confidence interval (%):", self.level_spin)
        oh.addLayout(ol)
        self.listwise_box = None
        if not isinstance(self, RelatedSamplesDialog):
            excl = QGroupBox("Excluded Cases")
            el = QVBoxLayout(excl)
            g2 = QButtonGroup(self)
            self.test_by_test = QRadioButton("Exclude cases test-by-test")
            self.listwise_box = QRadioButton("Exclude cases listwise")
            for rb in (self.test_by_test, self.listwise_box):
                g2.addButton(rb)
                el.addWidget(rb)
            (self.listwise_box if self.opts["listwise"] else self.test_by_test).setChecked(True)
            oh.addWidget(excl)
        oh.addStretch()
        lay.addWidget(options)
        return page

    TEST_COLUMNS = 1

    def extra_settings(self, layout) -> None:
        """Hook for test parameters (hypothesized values, multiple comparisons…)."""

    def _multiple_combo(self, layout, label: str) -> None:
        self.multiple = QComboBox()
        self.multiple.addItem("All pairwise", "all")
        self.multiple.addItem("None", "none")
        self.multiple.setCurrentIndex(max(self.multiple.findData(self.opts["multiple"]), 0))
        form = QFormLayout()
        form.addRow(label, self.multiple)
        layout.addLayout(form)

    def _enable_tests(self, on: bool) -> None:
        for cb in self.test_boxes.values():
            cb.setEnabled(on)

    def _objective_changed(self, key: str) -> None:
        self.opts["objective"] = key
        preset = dict(self.OBJECTIVE_PRESETS).get(key)
        if key == "auto":
            self.auto_radio.setChecked(True)
        else:
            self.custom_radio.setChecked(True)
            if preset:
                for k, cb in self.test_boxes.items():
                    cb.setChecked(k in preset)
        self._update_description()

    OBJECTIVE_PRESETS: list[tuple[str, tuple]] = []

    def _update_description(self) -> None:
        text = next((d for k, _, d in self.OBJECTIVES if k == self.opts["objective"]), "")
        self.description.setText(text)

    # ------------------------------------------------------------------ values
    def collect(self) -> None:
        o = self.opts
        o["mode"] = "auto" if self.auto_radio.isChecked() else "custom"
        o["tests"] = [k for k, cb in self.test_boxes.items() if cb.isChecked()]
        o["alpha"] = self.alpha.value()
        o["level"] = self.level_spin.value()
        if self.listwise_box is not None:
            o["listwise"] = self.listwise_box.isChecked()
        if hasattr(self, "multiple"):
            o["multiple"] = self.multiple.currentData()
        self.collect_extra()

    def collect_extra(self) -> None:
        pass

    def chosen_tests(self) -> tuple | None:
        o = self.opts
        return None if o["mode"] == "auto" else tuple(o["tests"])

    def validate(self) -> str | None:
        self.collect()
        if self.opts["mode"] == "custom" and not self.opts["tests"]:
            return "Select at least one test on the Settings tab, or choose the tests " \
                   "automatically."
        return None


class OneSampleDialog(_NpTestsDialog):
    OBJECTIVES = [
        ("auto", "Automatically compare observed data to hypothesized",
         "Automatically compare observed data to hypothesized applies the Binomial test to "
         "categorical fields with only two categories, the Chi-Square test to all other "
         "categorical fields, and the Kolmogorov-Smirnov test to continuous fields."),
        ("runs", "Test sequence for randomness",
         "Test sequence for randomness uses the Runs test to test the observed sequence of "
         "data values for randomness."),
        ("custom", "Customize analysis",
         "Customize analysis allows you to have fine control over the tests performed and "
         "their options. Tests for multiple comparisons are available in the Settings tab."),
    ]
    OBJECTIVE_PRESETS = [("runs", ("runs",))]
    TESTS = [("binomial", "Compare observed binary probability to hypothesized (Binomial test)"),
             ("chisq", "Compare observed probabilities to hypothesized (Chi-Square test)"),
             ("ks", "Test observed distribution against hypothesized (Kolmogorov-Smirnov test)"),
             ("wilcoxon", "Compare median to hypothesized (Wilcoxon signed-rank test)"),
             ("runs", "Test sequence for randomness (Runs test)")]
    DEFAULTS = {**_NpTestsDialog.BASE_DEFAULTS, "tests": ["binomial", "chisq", "ks"],
                "prop": 0.5, "dists": ["normal"], "median": 0.0, "runs_cut": "median",
                "runs_value": 0.0}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "One-Sample Nonparametric Tests", parent)

    def build_fields(self) -> None:
        self.add_target("fields", "Test Fields:")

    def extra_settings(self, layout) -> None:
        o = self.opts
        form = QFormLayout()
        self.prop = QDoubleSpinBox()
        self.prop.setRange(0.0001, 0.9999)
        self.prop.setDecimals(4)
        self.prop.setValue(float(o["prop"]))
        form.addRow("Binomial: hypothesized proportion:", self.prop)
        dist_row = QHBoxLayout()
        self.dist_boxes = {}
        for d in ("normal", "uniform", "exponential", "poisson"):
            cb = QCheckBox(d.capitalize())
            cb.setChecked(d in o["dists"])
            self.dist_boxes[d] = cb
            dist_row.addWidget(cb)
        form.addRow("Kolmogorov-Smirnov test distributions:", dist_row)
        self.median = QLineEdit(f"{o['median']:g}")
        form.addRow("Wilcoxon: hypothesized median:", self.median)
        self.runs_cut = QComboBox()
        for key, text in (("median", "Sample median"), ("mean", "Sample mean"),
                          ("custom", "Custom")):
            self.runs_cut.addItem(text, key)
        self.runs_cut.setCurrentIndex(max(self.runs_cut.findData(o["runs_cut"]), 0))
        self.runs_value = QLineEdit(f"{o['runs_value']:g}")
        cut_row = QHBoxLayout()
        cut_row.addWidget(self.runs_cut)
        cut_row.addWidget(self.runs_value)
        form.addRow("Runs: cut point for continuous fields:", cut_row)
        layout.addLayout(form)

    def collect_extra(self) -> None:
        o = self.opts
        o["prop"] = self.prop.value()
        o["dists"] = [d for d, cb in self.dist_boxes.items() if cb.isChecked()] or ["normal"]
        o["runs_cut"] = self.runs_cut.currentData()
        try:
            o["median"] = float(self.median.text().replace(",", "."))
            o["runs_value"] = float(self.runs_value.text().replace(",", "."))
        except ValueError:
            o["median"] = o.get("median", 0.0)

    def compute(self):
        self.collect()
        o = self.opts
        return nptests.one_sample(self.ds, self.selected("fields"), self.chosen_tests(),
                                  o["alpha"], o["level"] / 100, o["listwise"], o["prop"],
                                  tuple(o["dists"]), o["median"], o["runs_cut"],
                                  o["runs_value"])


class IndependentSamplesDialog(_NpTestsDialog):
    OBJECTIVES = [
        ("auto", "Automatically compare distributions across groups",
         "Automatically compare distributions across groups applies the Mann-Whitney U test "
         "to data with 2 groups, or the Kruskal-Wallis 1-way ANOVA to data with k groups."),
        ("medians", "Compare medians across groups",
         "Compare medians across groups uses the Median test to compare the observed "
         "medians across groups."),
        ("custom", "Customize analysis",
         "Customize analysis allows you to have fine control over the tests performed and "
         "their options. Tests for multiple comparisons are available in the Settings tab."),
    ]
    OBJECTIVE_PRESETS = [("medians", ("median",))]
    TESTS = [("mw", "Mann-Whitney U (2 samples)"), ("kw", "Kruskal-Wallis 1-way ANOVA (k samples)"),
             ("ks", "Kolmogorov-Smirnov (2 samples)"), ("jt", "Test for ordered alternatives "
                                                            "(Jonckheere-Terpstra) (k samples)"),
             ("ww", "Test sequence for randomness (Wald-Wolfowitz for 2 samples)"),
             ("median", "Median test (k samples)"),
             ("moses", "Moses test of extreme reaction (2 samples)"),
             ("hl", "Hodges-Lehmann estimate (2 samples)")]
    TEST_COLUMNS = 2
    DEFAULTS = {**_NpTestsDialog.BASE_DEFAULTS, "tests": ["mw", "kw"], "moses_custom": False,
                "moses_h": 1}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Nonparametric Tests: Two or More Independent Samples", parent)

    def build_fields(self) -> None:
        self.add_target("fields", "Test Fields:", accepts=is_numeric)
        self.add_target("group", "Groups:", single=True)

    def extra_settings(self, layout) -> None:
        self._multiple_combo(layout, "Multiple comparisons (Kruskal-Wallis):")
        row = QHBoxLayout()
        self.moses_custom = QCheckBox("Moses: custom number of outliers")
        self.moses_custom.setChecked(bool(self.opts["moses_custom"]))
        self.moses_h = QSpinBox()
        self.moses_h.setRange(0, 1000)
        self.moses_h.setValue(int(self.opts["moses_h"]))
        row.addWidget(self.moses_custom)
        row.addWidget(self.moses_h)
        row.addStretch()
        layout.addLayout(row)

    def collect_extra(self) -> None:
        self.opts["moses_custom"] = self.moses_custom.isChecked()
        self.opts["moses_h"] = self.moses_h.value()

    def compute(self):
        self.collect()
        o = self.opts
        return nptests.independent_samples(
            self.ds, self.selected("fields"), self.selected_one("group"), self.chosen_tests(),
            o["alpha"], o["level"] / 100, o["listwise"], o["multiple"],
            int(o["moses_h"]) if o["moses_custom"] else None)


class RelatedSamplesDialog(_NpTestsDialog):
    OBJECTIVES = [
        ("auto", "Automatically compare observed data to hypothesized data",
         "Automatically compare observed data to hypothesized data applies McNemar's test to "
         "categorical data when 2 fields are specified, Cochran's Q to categorical data when "
         "more than 2 fields are specified, the Wilcoxon Matched-Pair Signed-Rank test to "
         "continuous data when 2 fields are specified, and Friedman's 2-Way ANOVA by Ranks "
         "to continuous data when more than 2 fields are specified."),
        ("custom", "Customize analysis",
         "Customize analysis allows you to have fine control over the tests performed and "
         "their options. Tests for multiple comparisons are available in the Settings tab."),
    ]
    TESTS = [("mcnemar", "McNemar's test (2 samples)"), ("cochran", "Cochran's Q (k samples)"),
             ("marginal", "Marginal Homogeneity test (2 samples)"),
             ("sign", "Sign test (2 samples)"),
             ("friedman", "Friedman's 2-way ANOVA by ranks (k samples)"),
             ("wilcoxon", "Wilcoxon matched-pair signed-rank (2 samples)"),
             ("kendall", "Kendall's coefficient of concordance (k samples)"),
             ("hl", "Hodges-Lehmann (2 samples)")]
    TEST_COLUMNS = 2
    DEFAULTS = {**_NpTestsDialog.BASE_DEFAULTS, "tests": ["wilcoxon", "friedman"]}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Nonparametric Tests: Two or More Related Samples", parent)

    def build_fields(self) -> None:
        self.add_target("fields", "Test Fields:", min_items=2)

    def extra_settings(self, layout) -> None:
        self._multiple_combo(layout, "Multiple comparisons (Friedman, Cochran's Q):")

    def compute(self):
        self.collect()
        o = self.opts
        return nptests.related_samples(self.ds, self.selected("fields"), self.chosen_tests(),
                                       o["alpha"], o["level"] / 100, o["multiple"])


MENUS = {
    "Analyze/Nonparametric Tests": [("One Sample...", OneSampleDialog),
                                    ("Independent Samples...", IndependentSamplesDialog),
                                    ("Related Samples...", RelatedSamplesDialog)],
}


def selftest() -> None:
    """Headless check of the legacy NPAR TESTS additions, the exact / Monte Carlo functions
    and the NPTESTS procedures."""
    import numpy as np
    import pandas as pd

    from rsstatchem.core.dataset import Dataset
    from rsstatchem.core.stats import nonparametric as npt

    rng = np.random.default_rng(11)
    n = 40
    ds = Dataset(pd.DataFrame({
        "y": rng.normal(10, 2, n).round(1), "g2": np.tile([1.0, 2.0], n // 2),
        "g3": np.repeat([1.0, 2.0, 3.0, 4.0], n // 4), "a": rng.integers(1, 4, n) * 1.0,
        "b": rng.integers(1, 4, n) * 1.0, "c": rng.integers(0, 2, n) * 1.0,
        "d": rng.integers(0, 2, n) * 1.0, "e": rng.integers(0, 2, n) * 1.0}))
    for sec in (npt.chi_square(ds, ["a"], expected_range=(1, 4), descriptives=True,
                               quartiles=True),
                npt.runs(ds, ["y"], "mode"), npt.ks_one_sample(ds, ["a"], "poisson"),
                npt.two_independent(ds, ["y"], "g2", (1.0, 2.0), True, True, True, True),
                npt.k_independent(ds, ["y"], "g3", 1, 4, True, True, True, listwise=True),
                npt.two_related(ds, [("a", "b"), ("c", "d")], True, True, True, True),
                nptests.one_sample(ds, ["y", "c"], charts_on=False),
                nptests.independent_samples(ds, ["y"], "g3", charts_on=False),
                nptests.related_samples(ds, ["c", "d", "e"], charts_on=False)):
        assert any(getattr(i, "rows", None) for i in sec.items), sec.title
    x = ds.df["y"].to_numpy()
    p = nptests.exact_mann_whitney(x[:8], x[8:16])
    assert 0 <= p.p <= 1 and p.method == "exact"
    mc = nptests.exact_kruskal([x[:10], x[10:20], x[20:30]], method="mc", samples_mc=500)
    assert mc.method == "monte carlo" and mc.lower <= mc.p <= mc.upper
    assert abs(nptests.jonckheere_test([x[:5], x[5:10]])["mean"] - 12.5) < 1e-9

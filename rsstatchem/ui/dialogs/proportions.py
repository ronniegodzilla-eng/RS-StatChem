"""Analyze > Compare Means: One-Sample / Independent-Samples / Paired-Samples Proportions
(SPSS 27.0.1+) and the Summary Independent-Samples T Test (t test from N, mean and SD)."""
from __future__ import annotations

from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
                               QGridLayout, QGroupBox, QLabel, QLineEdit, QMessageBox,
                               QVBoxLayout)

from rsstatchem.core.stats import proportions as pr
from rsstatchem.core.stats import ttest
from rsstatchem.ui.dialogs.base import SubDialog, VariablePickerDialog
from rsstatchem.ui.dialogs.compare import GroupingMixin


def _parse_codes(text: str) -> list:
    out = []
    for tok in text.replace(";", " ").replace(",", " ").split():
        try:
            out.append(float(tok))
        except ValueError:
            out.append(tok)
    return out


class _ProportionsBase(VariablePickerDialog):
    CI_KEYS: tuple = ()
    CI_NAMES: dict = {}
    TEST_KEYS: tuple = ()
    TEST_NAMES: dict = {}
    TITLE = ""

    def _success(self) -> None:
        dlg = SubDialog(f"{self.TITLE}: Define Success", self, columns=1)
        box = dlg.group("Define Success for Categorical Fields")
        dlg.radio(box, "rule", [("last", "Last value"), ("first", "First value"),
                                ("values", "Value(s):"), ("midpoint", "Midpoint"),
                                ("cutpoint", "Cutpoint:")], self.opts["rule"])
        values = QLineEdit(" ".join(str(v) for v in self.opts.get("values") or []))
        cut = QLineEdit("" if self.opts.get("cut") is None else f"{self.opts['cut']:g}")
        form = QFormLayout()
        form.addRow("Value(s):", values)
        form.addRow("Cutpoint:", cut)
        box.addLayout(form)
        box.addWidget(QLabel("Success values count as 1, all other non-missing values as 0."))
        res = dlg.run()
        if res is None:
            return
        self.opts["rule"] = res["rule"]
        self.opts["values"] = _parse_codes(values.text())
        try:
            self.opts["cut"] = float(cut.text().replace(",", ".")) if cut.text().strip() \
                else None
        except ValueError:
            self.opts["cut"] = None

    def _intervals(self) -> None:
        dlg = SubDialog(f"{self.TITLE}: Confidence Intervals", self, columns=1)
        box = dlg.group("Interval Types")
        for k in self.CI_KEYS:
            dlg.check(box, k, self.CI_NAMES[k].replace('"', ""), k in self.opts["ci"])
        box = dlg.group("Coverage Level")
        dlg.number(box, "level", "Coverage Level (%):", self.opts["level"], 50, 99.99, 2)
        res = dlg.run()
        if res is not None:
            self.opts["ci"] = [k for k in self.CI_KEYS if res.get(k)]
            self.opts["level"] = res["level"]

    def _tests(self) -> None:
        dlg = SubDialog(f"{self.TITLE}: Tests", self, columns=1)
        box = dlg.group("Test Types")
        for k in self.TEST_KEYS:
            dlg.check(box, k, self.TEST_NAMES[k], k in self.opts["tests"])
        if "test_value" in self.opts:
            dlg.number(dlg.group("Test Value"), "test_value", "Test Value:",
                       self.opts["test_value"], 0.0001, 0.9999, 4)
        res = dlg.run()
        if res is not None:
            self.opts["tests"] = [k for k in self.TEST_KEYS if res.get(k)]
            if "test_value" in res:
                self.opts["test_value"] = res["test_value"]

    def _spec(self):
        rule = self.opts["rule"]
        if rule == "values":
            return self.opts.get("values") or []
        if rule == "cutpoint":
            return self.opts.get("cut")
        return None

    def validate(self):
        if self.opts["rule"] == "cutpoint" and self.opts.get("cut") is None:
            return "Enter a cutpoint in Define Success…"
        if self.opts["rule"] == "values" and not self.opts.get("values"):
            return "Enter the success value(s) in Define Success…"
        return None


class OneSampleProportionsDialog(_ProportionsBase):
    TITLE = "One-Sample Proportions"
    CI_KEYS, CI_NAMES = pr.ONE_CI, pr.ONE_CI_NAMES
    TEST_KEYS, TEST_NAMES = pr.ONE_TEST, pr.ONE_TEST_NAMES
    DEFAULTS = {"rule": "last", "values": [], "cut": None, "ci": ["clopper_pearson"],
                "tests": ["score"], "test_value": 0.5, "level": 95.0, "listwise": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, self.TITLE, parent)
        self.add_target("vars", "Test Variable(s):")
        self.add_option_button("Confidence Intervals...", self._intervals)
        self.add_option_button("Tests...", self._tests)
        self.add_option_button("Define Success...", self._success)
        self.add_option_button("Missing Values...", self._missing)
        self.finish()

    def _missing(self) -> None:
        dlg = SubDialog(f"{self.TITLE}: Missing Values", self, columns=1)
        dlg.radio(dlg.group("Missing Values"), "listwise",
                  [(False, "Exclude cases test-by-test"), (True, "Exclude cases listwise")],
                  self.opts["listwise"])
        res = dlg.run()
        if res:
            self.opts.update(res)

    def compute(self):
        o = self.opts
        return pr.one_sample(self.ds, self.selected("vars"), o["test_value"], o["ci"],
                             o["tests"], o["rule"], self._spec(), o["level"] / 100,
                             o["listwise"])


class IndependentProportionsDialog(GroupingMixin, _ProportionsBase):
    TITLE = "Independent-Samples Proportions"
    CI_KEYS, CI_NAMES = pr.IND_CI, pr.IND_CI_NAMES
    TEST_KEYS, TEST_NAMES = pr.IND_TEST, pr.IND_TEST_NAMES
    DEFAULTS = {"rule": "last", "values": [], "cut": None, "ci": ["newcombe"],
                "tests": ["wald_h0"], "level": 95.0, "group_def": None}

    def __init__(self, ds, parent=None):
        super().__init__(ds, self.TITLE, parent)
        self.add_target("vars", "Test Variable(s):")
        self.setup_grouping()
        self.add_option_button("Confidence Intervals...", self._intervals)
        self.add_option_button("Tests...", self._tests)
        self.add_option_button("Define Success...", self._success)
        self.finish()

    def validate(self):
        return GroupingMixin.validate(self) or _ProportionsBase.validate(self)

    def compute(self):
        o = self.opts
        d = self._definition()
        return pr.independent(self.ds, self.selected("vars"), self.selected_one("group"),
                              d.get("groups"), d.get("cut"), o["ci"], o["tests"], o["rule"],
                              self._spec(), o["level"] / 100)


class PairedProportionsDialog(_ProportionsBase):
    TITLE = "Paired-Samples Proportions"
    CI_KEYS, CI_NAMES = pr.PAIR_CI, pr.PAIR_CI_NAMES
    TEST_KEYS, TEST_NAMES = pr.PAIR_TEST, pr.PAIR_TEST_NAMES
    DEFAULTS = {"rule": "last", "values": [], "cut": None, "ci": ["newcombe"],
                "tests": ["mcnemar"], "level": 95.0}

    def __init__(self, ds, parent=None):
        super().__init__(ds, self.TITLE, parent)
        self.add_pair_target("pairs", "Paired Variables:", accepts=lambda v: True)
        self.add_option_button("Confidence Intervals...", self._intervals)
        self.add_option_button("Tests...", self._tests)
        self.add_option_button("Define Success...", self._success)
        self.finish()

    def compute(self):
        o = self.opts
        return pr.paired(self.ds, self.pairs("pairs"), o["ci"], o["tests"], o["rule"],
                         self._spec(), o["level"] / 100)


class SummaryTTestDialog(QDialog):
    """Summary Independent-Samples T Test: enter N, mean and SD of two samples."""

    needs_data = False
    _last: dict = {}

    def __init__(self, ds=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Summary Independent-Samples T Test")
        self.fields = {}
        grid = QGridLayout()
        for c, head in enumerate(["", "Sample 1", "Sample 2"]):
            grid.addWidget(QLabel(f"<b>{head}</b>"), 0, c)
        for r, (key, label) in enumerate((("label", "Label:"), ("n", "Number of Cases:"),
                                          ("mean", "Mean:"), ("sd", "Standard Deviation:")),
                                         start=1):
            grid.addWidget(QLabel(label), r, 0)
            for c in (1, 2):
                default = self._last.get(f"{key}{c}", f"Sample {c}" if key == "label" else "")
                edit = QLineEdit(str(default))
                self.fields[f"{key}{c}"] = edit
                grid.addWidget(edit, r, c)
        box = QGroupBox("Summary Data")
        box.setLayout(grid)
        self.level = QDoubleSpinBox()
        self.level.setRange(50, 99.99)
        self.level.setValue(float(self._last.get("level", 95.0)))
        form = QFormLayout()
        form.addRow("Confidence Interval Level (%):", self.level)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(box)
        layout.addLayout(form)
        layout.addWidget(bb)
        self.values: dict = {}

    def accept(self) -> None:
        try:
            vals = {k: float(self.fields[k].text().replace(",", "."))
                    for k in ("n1", "mean1", "sd1", "n2", "mean2", "sd2")}
        except ValueError:
            QMessageBox.warning(self, self.windowTitle(),
                                "Enter numbers for N, mean and standard deviation.")
            return
        if vals["n1"] < 2 or vals["n2"] < 2 or vals["sd1"] < 0 or vals["sd2"] < 0:
            QMessageBox.warning(self, self.windowTitle(),
                                "Each sample needs N >= 2 and a non-negative SD.")
            return
        self.values = vals
        type(self)._last = {k: e.text() for k, e in self.fields.items()}
        type(self)._last["level"] = self.level.value()
        super().accept()

    def compute(self):
        v = self.values
        return ttest.summary_independent(v["n1"], v["mean1"], v["sd1"], v["n2"], v["mean2"],
                                         v["sd2"], self.level.value() / 100,
                                         self.fields["label1"].text() or "Sample 1",
                                         self.fields["label2"].text() or "Sample 2")


MENUS = {
    "Analyze/Compare Means": [
        ("One-Sample Proportions...", OneSampleProportionsDialog),
        ("Independent-Samples Proportions...", IndependentProportionsDialog),
        ("Paired-Samples Proportions...", PairedProportionsDialog),
        None,
        ("Summary Independent-Samples T Test...", SummaryTTestDialog),
    ],
}


def selftest() -> None:
    """Headless checks of the proportions and summary t-test core code."""
    lo, hi = pr.one_ci(81, 263, "wilson")
    if not (abs(lo - 0.2553) < 1e-4 and abs(hi - 0.3662) < 1e-4):
        raise AssertionError("Wilson interval")
    lo, hi = pr.ind_ci(56, 70, 48, 80, "newcombe")
    if not (abs(lo - 0.0524) < 1e-4 and abs(hi - 0.3339) < 1e-4):
        raise AssertionError("Newcombe interval")
    sec = ttest.summary_independent(20, 10.5, 2.1, 25, 9.2, 3.0)
    if not any(getattr(i, "title", "") == "Independent Samples Test" for i in sec.items):
        raise AssertionError("summary t test")

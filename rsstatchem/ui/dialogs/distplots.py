"""Analyze > Descriptive Statistics: Ratio…, P-P Plots…, Q-Q Plots…, Percentiles… and
Normality Analysis… (SPSS RATIO STATISTICS, PPLOT, the 29.0.1 Percentiles dialog and the
SPSS 30 Normality Analysis)."""
from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QRadioButton,
                               QSpinBox, QVBoxLayout)

from rsstatchem.core.stats import distplots, ratio
from rsstatchem.ui.dialogs.base import SubDialog, VariablePickerDialog, is_numeric


def _numbers(text: str) -> list[float]:
    return [float(p.replace(",", ".")) for p in text.replace(";", " ").split() if p.strip()]


# ============================================================================ Ratio
class RatioDialog(VariablePickerDialog):
    DEFAULTS = {"order": "ascending", "stats": list(ratio.DEFAULT_STATS), "ci": False,
                "level": 95.0, "between": False, "low": 0.9, "high": 1.1, "within_on": False,
                "within": 10.0}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Ratio Statistics", parent)
        self.add_target("num", "Numerator:", accepts=is_numeric, single=True)
        self.add_target("den", "Denominator:", accepts=is_numeric, single=True)
        self.add_target("group", "Group Variable:", single=True, required=False)
        self.add_option_button("Statistics...", self._statistics)
        box = QGroupBox("Sort by group variable")
        row = QHBoxLayout(box)
        for key, text in (("ascending", "Ascending order"), ("descending", "Descending order")):
            rb = QRadioButton(text)
            rb.setChecked(self.opts["order"] == key)
            rb.toggled.connect(lambda on, k=key: on and self.opts.__setitem__("order", k))
            row.addWidget(rb)
        row.addStretch()
        self.options_layout.addWidget(box)
        self.finish()

    def _statistics(self) -> None:
        dlg = SubDialog("Ratio Statistics: Statistics", self)
        chosen = set(self.opts["stats"])
        box = dlg.group("Central Tendency")
        for key, text in (("median", "Median"), ("mean", "Mean"), ("wmean", "Weighted Mean")):
            dlg.check(box, key, text, key in chosen)
        dlg.check(box, "ci", "Confidence intervals", self.opts["ci"])
        dlg.number(box, "level", "Level:", self.opts["level"], 50, 99.99, 2, " %")
        box = dlg.group("Dispersion")
        for key, text in (("aad", "AAD"), ("prd", "PRD"), ("cod", "COD"),
                          ("mdcov", "Median Centered COV"), ("mncov", "Mean Centered COV"),
                          ("std", "Standard deviation"), ("range", "Range"),
                          ("min", "Minimum"), ("max", "Maximum")):
            dlg.check(box, key, text, key in chosen)
        box = dlg.group("Concentration Index")
        dlg.check(box, "between", "Ratios Between Proportions", self.opts["between"])
        dlg.number(box, "low", "Low Proportion:", self.opts["low"], 0, 1e6, 3)
        dlg.number(box, "high", "High Proportion:", self.opts["high"], 0, 1e6, 3)
        dlg.check(box, "within_on", "Ratios Within Percent of Median", self.opts["within_on"])
        dlg.number(box, "within", "Percent of median:", self.opts["within"], 0.01, 99.99, 2,
                   " %")
        box = dlg.group("Price-Related Bias")
        dlg.check(box, "prb", "PRB (with standard error)", "prb" in chosen)
        values = dlg.run()
        if values:
            self.opts["stats"] = [k for k in ratio.STATISTICS if values.get(k)]
            for k in ("ci", "level", "between", "low", "high", "within_on", "within"):
                self.opts[k] = values[k]

    def validate(self):
        o = self.opts
        if o["between"] and o["low"] >= o["high"]:
            return "The low proportion must be smaller than the high proportion."
        return None

    def compute(self):
        o = self.opts
        return ratio.ratio(self.ds, self.selected_one("num"), self.selected_one("den"),
                           self.selected_one("group"), o["order"], tuple(o["stats"]), o["ci"],
                           o["level"] / 100, o["low"] if o["between"] else None,
                           o["high"] if o["between"] else None,
                           o["within"] if o["within_on"] else None)


# ============================================================================ P-P / Q-Q
class _ProbabilityPlotDialog(VariablePickerDialog):
    KIND = "pp"
    TITLE = "P-P Plots"
    DEFAULTS = {"dist": "normal", "df": 5.0, "estimate": True, "p1": 0.0, "p2": 1.0,
                "log": False, "standardize": False, "diff": False, "diff_n": 1,
                "sdiff": False, "sdiff_n": 1, "period": 12, "fraction": "blom",
                "ties": "mean"}

    def __init__(self, ds, parent=None):
        super().__init__(ds, self.TITLE, parent)
        self.add_target("vars", "Variables:", accepts=is_numeric)
        grid = QGridLayout()
        o = self.opts

        box = QGroupBox("Test Distribution")
        form = QFormLayout(box)
        self.dist = QComboBox()
        for key, (title, _, _) in distplots.DISTRIBUTIONS.items():
            self.dist.addItem(title, key)
        self.dist.setCurrentIndex(max(self.dist.findData(o["dist"]), 0))
        self.df = QDoubleSpinBox()
        self.df.setRange(0.01, 1e6)
        self.df.setValue(o["df"])
        form.addRow(self.dist)
        form.addRow("df:", self.df)
        grid.addWidget(box, 0, 0)

        box = QGroupBox("Distribution Parameters")
        form = QFormLayout(box)
        self.estimate = QCheckBox("Estimate from data")
        self.estimate.setChecked(o["estimate"])
        self.p1, self.p2 = QDoubleSpinBox(), QDoubleSpinBox()
        for spin, key in ((self.p1, "p1"), (self.p2, "p2")):
            spin.setRange(-1e9, 1e9)
            spin.setDecimals(4)
            spin.setValue(o[key])
        self.p1_label, self.p2_label = QLabel("Location:"), QLabel("Scale:")
        form.addRow(self.estimate)
        form.addRow(self.p1_label, self.p1)
        form.addRow(self.p2_label, self.p2)
        grid.addWidget(box, 0, 1)

        box = QGroupBox("Transform")
        inner = QGridLayout(box)
        self.log = QCheckBox("Natural log transform")
        self.std = QCheckBox("Standardize values")
        self.diff = QCheckBox("Difference:")
        self.sdiff = QCheckBox("Seasonally difference:")
        self.diff_n, self.sdiff_n, self.period = QSpinBox(), QSpinBox(), QSpinBox()
        for spin, key in ((self.diff_n, "diff_n"), (self.sdiff_n, "sdiff_n"),
                          (self.period, "period")):
            spin.setRange(1, 999)
            spin.setValue(int(o[key]))
        for cb, key in ((self.log, "log"), (self.std, "standardize"), (self.diff, "diff"),
                        (self.sdiff, "sdiff")):
            cb.setChecked(o[key])
        inner.addWidget(self.log, 0, 0)
        inner.addWidget(self.std, 1, 0)
        inner.addWidget(self.diff, 0, 1)
        inner.addWidget(self.diff_n, 0, 2)
        inner.addWidget(self.sdiff, 1, 1)
        inner.addWidget(self.sdiff_n, 1, 2)
        inner.addWidget(QLabel("Current Periodicity:"), 2, 1)
        inner.addWidget(self.period, 2, 2)
        grid.addWidget(box, 1, 0, 1, 2)

        box = QGroupBox("Proportion Estimation Formula")
        inner = QVBoxLayout(box)
        self.fraction = {}
        for key, (text, _, _) in distplots.FRACTIONS.items():
            rb = QRadioButton(text)
            rb.setChecked(o["fraction"] == key)
            self.fraction[key] = rb
            inner.addWidget(rb)
        grid.addWidget(box, 2, 0)
        box = QGroupBox("Rank Assigned to Ties")
        inner = QVBoxLayout(box)
        self.ties = {}
        for key, text in (("mean", "Mean"), ("high", "High"), ("low", "Low"),
                          ("break", "Break ties arbitrarily")):
            rb = QRadioButton(text)
            rb.setChecked(o["ties"] == key)
            self.ties[key] = rb
            inner.addWidget(rb)
        grid.addWidget(box, 2, 1)
        self.options_layout.addLayout(grid)
        self.dist.currentIndexChanged.connect(self._sync)
        self.estimate.toggled.connect(self._sync)
        self._sync()
        self.finish()

    def reset(self) -> None:
        super().reset()
        if not hasattr(self, "ties"):
            return
        o = self.opts
        self.dist.setCurrentIndex(max(self.dist.findData(o["dist"]), 0))
        self.df.setValue(o["df"])
        self.estimate.setChecked(o["estimate"])
        self.p1.setValue(o["p1"])
        self.p2.setValue(o["p2"])
        for cb, key in ((self.log, "log"), (self.std, "standardize"), (self.diff, "diff"),
                        (self.sdiff, "sdiff")):
            cb.setChecked(o[key])
        for spin, key in ((self.diff_n, "diff_n"), (self.sdiff_n, "sdiff_n"),
                          (self.period, "period")):
            spin.setValue(int(o[key]))
        self.fraction[o["fraction"]].setChecked(True)
        self.ties[o["ties"]].setChecked(True)
        self._sync()

    def _sync(self) -> None:
        key = self.dist.currentData()
        _, pnames, _ = distplots.DISTRIBUTIONS[key]
        self.df.setEnabled(key in ("chisq", "t"))
        free = not self.estimate.isChecked() and key not in ("chisq", "t")
        self.p1_label.setText(pnames[0] + ":")
        self.p2_label.setText((pnames[1] + ":") if len(pnames) > 1 else "")
        self.p1.setEnabled(free)
        self.p2.setEnabled(free and len(pnames) > 1)
        self.p2.setVisible(len(pnames) > 1)

    def _collect(self) -> None:
        o = self.opts
        o.update(dist=self.dist.currentData(), df=self.df.value(),
                 estimate=self.estimate.isChecked(), p1=self.p1.value(), p2=self.p2.value(),
                 log=self.log.isChecked(), standardize=self.std.isChecked(),
                 diff=self.diff.isChecked(), diff_n=self.diff_n.value(),
                 sdiff=self.sdiff.isChecked(), sdiff_n=self.sdiff_n.value(),
                 period=self.period.value(),
                 fraction=next(k for k, rb in self.fraction.items() if rb.isChecked()),
                 ties=next(k for k, rb in self.ties.items() if rb.isChecked()))

    def validate(self):
        self._collect()
        o = self.opts
        if o["dist"] == "uniform" and not o["estimate"] and o["p1"] >= o["p2"]:
            return "The uniform minimum must be smaller than the maximum."
        pnames = distplots.DISTRIBUTIONS[o["dist"]][1]
        if not o["estimate"] and o["dist"] not in ("chisq", "t", "uniform"):
            values = (o["p1"], o["p2"])[:len(pnames)]
            if any(v <= 0 for v, name in zip(values, pnames)
                   if name in ("Scale", "Shape", "Threshold")):
                return "Scale, shape and threshold parameters must be positive."
        return None

    def compute(self):
        o = self.opts
        pnames = distplots.DISTRIBUTIONS[o["dist"]][1]
        params = None if o["estimate"] else (o["p1"], o["p2"])[:len(pnames)]
        df = o["df"] if o["dist"] in ("chisq", "t") else None
        return distplots.pplot(self.ds, self.selected("vars"), self.KIND, o["dist"],
                               o["estimate"], params, df, o["log"], o["standardize"],
                               o["diff_n"] if o["diff"] else 0,
                               o["sdiff_n"] if o["sdiff"] else 0, o["period"], o["fraction"],
                               o["ties"])


class PPPlotsDialog(_ProbabilityPlotDialog):
    KIND = "pp"
    TITLE = "P-P Plots"


class QQPlotsDialog(_ProbabilityPlotDialog):
    KIND = "qq"
    TITLE = "Q-Q Plots"


# ============================================================================ Percentiles
class PercentilesDialog(VariablePickerDialog):
    DEFAULTS = {"points": "5 10 25 50 75 90 95", "method": "haverage", "hinges": True,
                "ci": True, "level": 95.0}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Percentiles", parent)
        self.add_target("vars", "Variables:", accepts=is_numeric)
        self.add_target("factor", "Factor Variable:", single=True, required=False)
        o = self.opts
        self.points = QLineEdit(o["points"])
        self.points.setPlaceholderText("e.g. 5 10 25 50 75 90 95")
        self.method = QComboBox()
        for key, text in (("haverage", "Weighted average at X(n+1)p (HAVERAGE)"),
                          ("waverage", "Weighted average at X(np) (WAVERAGE)"),
                          ("round", "Observation closest to np (ROUND)"),
                          ("empirical", "Empirical distribution function (EMPIRICAL)"),
                          ("aempirical", "Empirical distribution with averaging "
                                         "(AEMPIRICAL)")):
            self.method.addItem(text, key)
        self.method.setCurrentIndex(max(self.method.findData(o["method"]), 0))
        self.hinges = QCheckBox("Tukey's hinges")
        self.hinges.setChecked(o["hinges"])
        self.ci = QCheckBox("Confidence intervals (order statistics)")
        self.ci.setChecked(o["ci"])
        self.level = QDoubleSpinBox()
        self.level.setRange(50, 99.99)
        self.level.setSuffix(" %")
        self.level.setValue(o["level"])
        form = QFormLayout()
        form.addRow("Percentile(s):", self.points)
        form.addRow("Estimation Method:", self.method)
        row = QHBoxLayout()
        row.addWidget(self.hinges)
        row.addWidget(self.ci)
        row.addWidget(QLabel("Level:"))
        row.addWidget(self.level)
        row.addStretch()
        form.addRow(row)
        self.options_layout.addLayout(form)
        self.finish()

    def validate(self):
        try:
            points = _numbers(self.points.text())
        except ValueError:
            return "Percentiles must be numbers separated by spaces."
        if not points or any(not 0 < p < 100 for p in points):
            return "Enter one or more percentiles between 0 and 100."
        self.opts.update(points=self.points.text(), method=self.method.currentData(),
                         hinges=self.hinges.isChecked(), ci=self.ci.isChecked(),
                         level=self.level.value())
        return None

    def compute(self):
        o = self.opts
        return distplots.percentiles(self.ds, self.selected("vars"),
                                     self.selected_one("factor"), _numbers(o["points"]),
                                     o["method"], o["hinges"], o["ci"], o["level"] / 100)


# ============================================================================ Normality
class NormalityDialog(VariablePickerDialog):
    DEFAULTS = {"univariate": True, "extra": True, "mardia": True, "hz": True,
                "royston": True, "qq": True, "chisq_qq": True, "alpha": 0.05}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Normality Analysis", parent)
        self.add_target("vars", "Variables:", accepts=is_numeric)
        self.add_option_button("Statistics...", self._statistics)
        self.add_option_button("Plots...", self._plots)
        self.finish()

    def _statistics(self) -> None:
        dlg = SubDialog("Normality Analysis: Statistics", self)
        box = dlg.group("Univariate")
        dlg.check(box, "univariate", "Shapiro-Wilk and Kolmogorov-Smirnov (Lilliefors)",
                  self.opts["univariate"])
        dlg.check(box, "extra", "Anderson-Darling and D'Agostino tests", self.opts["extra"])
        box = dlg.group("Multivariate")
        dlg.check(box, "mardia", "Mardia's skewness and kurtosis", self.opts["mardia"])
        dlg.check(box, "hz", "Henze-Zirkler", self.opts["hz"])
        dlg.check(box, "royston", "Royston", self.opts["royston"])
        dlg.number(dlg.group("Decision"), "alpha", "Significance level:", self.opts["alpha"],
                   0.001, 0.2, 3)
        values = dlg.run()
        if values:
            self.opts.update(values)

    def _plots(self) -> None:
        dlg = SubDialog("Normality Analysis: Plots", self, columns=1)
        box = dlg.group("Plots")
        dlg.check(box, "qq", "Normal Q-Q plots", self.opts["qq"])
        dlg.check(box, "chisq_qq", "Chi-square Q-Q plot of Mahalanobis distances",
                  self.opts["chisq_qq"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def compute(self):
        o = self.opts
        return distplots.normality(self.ds, self.selected("vars"), o["univariate"],
                                   o["extra"], o["mardia"], o["hz"], o["royston"], o["qq"],
                                   o["chisq_qq"], o["alpha"])


MENUS = {
    "Analyze/Descriptive Statistics": [("Ratio...", RatioDialog),
                                       ("P-P Plots...", PPPlotsDialog),
                                       ("Q-Q Plots...", QQPlotsDialog),
                                       ("Percentiles...", PercentilesDialog),
                                       ("Normality Analysis...", NormalityDialog)],
}


def selftest() -> None:
    """Headless check of the PPLOT, Percentiles, Normality Analysis and Ratio cores."""
    import numpy as np
    import pandas as pd

    from rsstatchem.core.dataset import Dataset

    rng = np.random.default_rng(3)
    n = 60
    s = rng.lognormal(5, 0.4, n)
    a = s * rng.normal(1.0, 0.1, n)
    ds = Dataset(pd.DataFrame({"x": rng.normal(10, 2, n), "y": rng.gamma(3, 2, n),
                               "a": a, "s": s, "g": rng.integers(1, 3, n).astype(float)}))
    for dist in distplots.DISTRIBUTIONS:
        if dist == "beta":
            continue
        sec = distplots.pplot(ds, ["y"], "qq" if dist == "normal" else "pp", dist, df=4.0)
        assert any(getattr(i, "kind", "") == "image" for i in sec.items), dist
    assert abs(distplots.spss_percentile([1, 2, 3, 4], 50) - 2.5) < 1e-12
    sec = distplots.percentiles(ds, ["x", "y"], "g")
    assert any(getattr(t, "title", "") == "Percentile Confidence Intervals" for t in sec.items)
    sec = distplots.normality(ds, ["x", "y", "a"])
    assert any(getattr(t, "title", "") == "Tests of Multivariate Normality" for t in sec.items)
    sec = ratio.ratio(ds, "a", "s", "g", stats_=ratio.STATISTICS, ci=True, low=0.9,
                      high=1.1, within=10)
    assert any(getattr(t, "title", "").startswith("Ratio Statistics for") for t in sec.items)

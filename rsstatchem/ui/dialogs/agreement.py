"""Analyze > Scale > Weighted Kappa… (SPSS 27) and Bland-Altman… (SPSS 30).

``selftest()`` also exercises the WP3 cores that are reached through dialogs the Data Editor
imports directly (Frequencies, Descriptives, Explore, Crosstabs, Reliability).
"""
from __future__ import annotations

from rsstatchem.core.stats import agreement
from rsstatchem.ui.dialogs.base import SubDialog, VariablePickerDialog


class WeightedKappaDialog(VariablePickerDialog):
    DEFAULTS = {"weighting": "linear", "level": 95.0, "crosstab": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Weighted Kappa", parent)
        self.add_pair_target("pairs", "Pairs of raters (Rater 1 – Rater 2):",
                             accepts=lambda v: True)
        self.add_option_button("Criteria...", self._criteria)
        self.add_option_button("Print...", self._print)
        self.finish()

    def _criteria(self) -> None:
        dlg = SubDialog("Weighted Kappa: Criteria", self, columns=1)
        dlg.radio(dlg.group("Weighting"), "weighting", [("linear", "Linear"),
                                                         ("quadratic", "Quadratic")],
                  self.opts["weighting"])
        dlg.number(dlg.group("Asymptotic Confidence Interval"), "level", "Level (%):",
                   self.opts["level"], 50, 99.99, 2)
        values = dlg.run()
        if values:
            self.opts.update(values)

    def _print(self) -> None:
        dlg = SubDialog("Weighted Kappa: Print", self, columns=1)
        dlg.check(dlg.group("Print"), "crosstab", "Crosstabulation of the ratings",
                  self.opts["crosstab"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def compute(self):
        o = self.opts
        return agreement.weighted_kappa_analysis(self.ds, self.pairs("pairs"), o["weighting"],
                                                 self.level(), o["crosstab"])


class BlandAltmanDialog(VariablePickerDialog):
    DEFAULTS = {"level": 95.0, "proportional": True, "plot": True, "percent": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Bland-Altman", parent)
        self.add_pair_target("pairs", "Paired measurements (Method 1 – Method 2):")
        self.add_option_button("Statistics...", self._statistics)
        self.add_checkbox("plot", "Display Bland-Altman plot")
        self.finish()

    def _statistics(self) -> None:
        dlg = SubDialog("Bland-Altman: Statistics", self, columns=1)
        box = dlg.group("Statistics")
        dlg.number(box, "level", "Confidence level (%):", self.opts["level"], 50, 99.99, 2)
        dlg.check(box, "proportional", "Test for proportional bias (regression)",
                  self.opts["proportional"])
        dlg.check(box, "percent", "Differences as percentage of the mean",
                  self.opts["percent"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def compute(self):
        o = self.opts
        return agreement.bland_altman(self.ds, self.pairs("pairs"), self.level(),
                                      o["proportional"], o["plot"], o["percent"])


MENUS = {"Analyze/Scale": [("Weighted Kappa...", WeightedKappaDialog),
                           ("Bland-Altman...", BlandAltmanDialog)]}


def selftest() -> None:
    """Headless check of the WP3 cores: crosstabs, reliability, agreement, explore, frequencies."""
    import numpy as np
    import pandas as pd

    from rsstatchem.core.dataset import Dataset
    from rsstatchem.core.stats import crosstabs, descriptive, explore, reliability

    rng = np.random.default_rng(3)
    n = 80
    true = rng.normal(size=n)
    items = {f"i{j}": np.clip(np.round(3 + true + rng.normal(scale=0.8, size=n)), 1, 5)
             for j in range(1, 5)}
    ds = Dataset(pd.DataFrame(items | {"g": rng.integers(1, 3, n).astype(float),
                                       "h": rng.integers(1, 4, n).astype(float),
                                       "l": rng.integers(1, 3, n).astype(float),
                                       "a": true + rng.normal(scale=0.3, size=n),
                                       "b": true + 0.2 + rng.normal(scale=0.3, size=n)}))
    sec = crosstabs.crosstabs(ds, ["g"], ["h"], cells=crosstabs.CELLS, layers=[["l"]],
                              statistics=crosstabs.STATISTICS, z_test=True, clustered_bar=True)
    assert any(getattr(t, "title", "") == "Directional Measures" for t in sec.items)
    ex = crosstabs.exact_tests(np.array([[3, 1, 2], [1, 4, 0]]), "exact")
    assert 0 <= ex["fisher"]["p"] <= 1
    assert crosstabs.exact_tests(np.array([[3, 1], [1, 4]]), "mc", samples=500)["pearson"]
    names = list(items)
    for model in reliability.MODELS:
        reliability.reliability(ds, names, True, True, True, True, model=model,
                                summaries=("means", "correlations"), anova="friedman",
                                tukey=True, hotelling=True,
                                icc={"model": "twoway_random", "type": "absolute"},
                                fleiss={"individual": True})
    icc = agreement.icc(ds.df[names].to_numpy(float), "twoway_mixed", "consistency")
    assert 0 < icc["average"]["value"] <= 1
    agreement.weighted_kappa_analysis(ds, [("i1", "i2")], "quadratic", crosstab=True)
    agreement.bland_altman(ds, [("a", "b")])
    explore.explore(ds, ["a"], ["g"], m_estimators=True, outliers=True, percentiles=True,
                    stem_leaf=True, spread="estimate", boxplot_mode="dependents")
    descriptive.frequencies(ds, ["h", "a"], descriptive.StatSpec(
        ("quartiles", "median"), {"percentiles": [10, 90], "cut_points": 5, "grouped": True,
                                  "order": "dfreq", "organize": "variable"}), chart="bar")
    descriptive.descriptives(ds, ["a", "b"], descriptive.StatSpec(("mean",), {"order": "dmean"}))

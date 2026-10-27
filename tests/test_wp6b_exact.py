"""WP6b: Exact… option — exact / Monte Carlo significance rows for the legacy NPAR TESTS and
Crosstabs dialogs, checked against SciPy and full enumeration."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import npar_exact as ne
from rsstatchem.core.stats import nptests as T
from rsstatchem.ui.dialogs import explore as edlg
from rsstatchem.ui.dialogs import nonparametric as ndlg
from rsstatchem.ui.dialogs.base import VariablePickerDialog
from rsstatchem.ui.dialogs.resample_ui import ExactDialog


def _run(cls, ds, memory, opts):
    VariablePickerDialog._memory[cls.__name__] = memory
    VariablePickerDialog._states[cls.__name__] = opts
    dlg = cls(ds)
    try:
        assert dlg.validate() is None
        return dlg.compute()
    finally:
        VariablePickerDialog._memory.pop(cls.__name__, None)
        VariablePickerDialog._states.pop(cls.__name__, None)
        dlg.deleteLater()


def _table(sec, pred):
    return next(t for t in sec.items if getattr(t, "kind", "") == "table" and pred(t))


def _row(table, label):
    return next(r for r in table.rows if r[0] == label)


EXACT = {"method": "exact", "use_limit": True, "time_limit": 5.0}
MC = {"method": "mc", "mc_level": 99.0, "mc_samples": 5000}


# ============================================================================ distributions
def _enumerate_mw(a, b):
    allv = np.concatenate([a, b])
    ranks = stats.rankdata(allv)
    n1 = len(a)
    obs = ranks[:n1].sum()
    sums = np.array([ranks[list(c)].sum() for c in itertools.combinations(range(len(allv)),
                                                                           n1)])
    center = n1 * (len(allv) + 1) / 2
    two = np.mean(np.abs(sums - center) >= abs(obs - center) - 1e-9)
    one = min(np.mean(sums <= obs + 1e-9), np.mean(sums >= obs - 1e-9))
    point = np.mean(np.abs(sums - obs) < 1e-9)
    return two, one, point


def test_mann_whitney_exact_tails_match_enumeration_and_scipy():
    a = np.array([1.1, 2.3, 2.3, 4.0, 5.2, 6.1])
    b = np.array([2.3, 3.1, 6.5, 7.2, 8.0, 9.4, 9.9])
    got = ne.mann_whitney_exact(a, b)
    assert got == pytest.approx(_enumerate_mw(a, b), abs=1e-12)
    a2, b2 = np.array([1.0, 2.5, 3.1, 4.7]), np.array([2.2, 5.5, 6.1, 7.3, 8.8])
    assert ne.mann_whitney_exact(a2, b2)[0] == pytest.approx(
        stats.mannwhitneyu(a2, b2, method="exact").pvalue)
    assert ne.mann_whitney_exact(a, b)[0] == pytest.approx(T.exact_mann_whitney(a, b).p)


def test_wilcoxon_and_sign_exact():
    d = np.array([1.2, -0.4, 2.2, 3.1, -1.0, 0.7, 2.6, 1.9, -0.2, 1.4])
    two, one, point = ne.wilcoxon_exact(d)
    assert two == pytest.approx(stats.wilcoxon(d, method="exact").pvalue)
    assert one == pytest.approx(stats.wilcoxon(d, method="exact", alternative="greater").pvalue)
    s2, s1, sp = ne.sign_exact(d)
    assert s2 == pytest.approx(stats.binomtest(7, 10).pvalue)
    assert s1 == pytest.approx(stats.binomtest(7, 10, alternative="greater").pvalue)
    assert sp == pytest.approx(stats.binom.pmf(7, 10, 0.5))


def test_runs_exact_matches_enumeration():
    x = np.array([3, 5, 1, 7, 8, 2, 9, 4, 6, 10, 0.5, 11.0])
    value = float(np.median(x))
    above = x >= value
    n2 = int(above.sum())
    runs = []
    for c in itertools.combinations(range(len(x)), n2):
        s = np.zeros(len(x), bool)
        s[list(c)] = True
        runs.append(1 + np.count_nonzero(s[1:] != s[:-1]))
    runs = np.array(runs)
    obs = 1 + np.count_nonzero(above[1:] != above[:-1])
    mu = 2 * n2 * (len(x) - n2) / len(x) + 1
    two, _, point = ne.runs_exact(x, value)
    assert two == pytest.approx(np.mean(np.abs(runs - mu) >= abs(obs - mu) - 1e-9))
    assert point == pytest.approx(np.mean(runs == obs))


def test_kruskal_exact_matches_brute_force():
    samples = [np.array([1.0, 3.5, 2.2]), np.array([4.1, 5.0, 3.5]), np.array([6.0, 7.7])]
    allv = np.concatenate(samples)
    sizes = [3, 3, 2]
    h_obs = stats.kruskal(*samples).statistic
    hs = []
    for perm in set(itertools.permutations(np.repeat([0, 1, 2], sizes))):
        lab = np.array(perm)
        hs.append(stats.kruskal(*[allv[lab == g] for g in range(3)]).statistic)
    ref = np.mean(np.array(hs) >= h_obs - 1e-9)
    assert T.exact_kruskal(samples).p == pytest.approx(ref)


# ============================================================================ dialogs
@pytest.fixture()
def small_ds():
    rng = np.random.default_rng(11)
    n = 16
    return Dataset(pd.DataFrame({
        "a": rng.normal(size=n).round(2), "b": rng.normal(0.5, 1, n).round(2),
        "c": rng.normal(size=n).round(2), "g": np.repeat([1.0, 2.0], n // 2),
        "k": np.tile([1.0, 2.0, 3.0, 4.0], n // 4), "cat": np.tile([1.0, 2.0, 2.0, 3.0], 4),
        "bin": np.tile([0.0, 1.0, 1.0, 1.0], 4), "bin2": np.tile([1.0, 0.0, 1.0, 0.0], 4)}))


def test_two_independent_exact_rows(qapp, small_ds):
    sec = _run(ndlg.TwoIndependentDialog, small_ds, {"vars": ["a", "b"], "group": ["g"]},
               {"group_def": {"var": "g", "groups": (1.0, 2.0), "cut": None}, "mw": True,
                "ks": True, "exact": EXACT})
    t = _table(sec, lambda t: any(r[0] == "Mann-Whitney U" for r in t.rows))
    a = small_ds.df.loc[small_ds.df.g == 1, "a"].to_numpy()
    b = small_ds.df.loc[small_ds.df.g == 2, "a"].to_numpy()
    two, one, point = ne.mann_whitney_exact(a, b)
    assert _row(t, "Exact Sig. (2-tailed)")[1] == ne.fmt_p(two)
    assert _row(t, "Exact Sig. (1-tailed)")[1] == ne.fmt_p(one)
    assert _row(t, "Point Probability")[1] == ne.fmt(point)
    ks = _table(sec, lambda t: "Two-Sample Kolmogorov-Smirnov" in t.title)
    assert any(r[0] == "Exact Sig. (2-tailed)" for r in ks.rows)


def test_monte_carlo_rows_with_interval(qapp, small_ds):
    sec = _run(ndlg.KIndependentDialog, small_ds, {"vars": ["a"], "group": ["k"]},
               {"group_def": {"var": "k", "lo": 1.0, "hi": 4.0}, "kw": True, "exact": MC})
    t = _table(sec, lambda t: any(r[0] == "Kruskal-Wallis H" for r in t.rows))
    p = float(_row(t, "Monte Carlo Sig. (2-tailed)ᶜ")[1])
    lo = float(_row(t, "Monte Carlo 99% Confidence Interval: Lower Bound")[1])
    hi = float(_row(t, "Monte Carlo 99% Confidence Interval: Upper Bound")[1])
    assert lo <= p <= hi
    assert any("5000 sampled tables" in f for f in t.footnotes)


def test_related_and_one_sample_exact(qapp, small_ds):
    sec = _run(ndlg.TwoRelatedDialog, small_ds, {"pairs": [("a", "b")]},
               {"wilcoxon": True, "sign": True, "exact": EXACT})
    w = _table(sec, lambda t: any("Wilcoxon" in f for f in t.footnotes)
               and t.title.startswith("Test Statistics"))
    d = (small_ds.df["b"] - small_ds.df["a"]).to_numpy()
    d = d[d != 0]
    r = stats.rankdata(np.abs(d))                   # tie-aware permutation distribution
    signs = np.array(list(itertools.product([0, 1], repeat=len(d))))
    tplus = signs @ r
    obs, center = r[d > 0].sum(), r.sum() / 2
    ref = np.mean(np.abs(tplus - center) >= abs(obs - center) - 1e-9)
    assert _row(w, "Exact Sig. (2-tailed)")[1] == ne.fmt_p(ref)
    sec = _run(ndlg.KRelatedDialog, small_ds, {"vars": ["a", "b", "c"]},
               {"friedman": True, "kendall": True, "exact": MC})
    assert any(r[0].startswith("Monte Carlo Sig. (2-tailed)") for t in sec.items
               if getattr(t, "kind", "") == "table" for r in t.rows)
    sec = _run(ndlg.ChiSquareDialog, small_ds, {"vars": ["cat"]}, {"exact": EXACT})
    t = _table(sec, lambda t: t.title == "Test Statistics")
    obs = np.array([4, 8, 4])
    exp = np.full(3, 16 / 3)
    chi = ((obs - exp) ** 2 / exp).sum()
    ref = sum(stats.multinomial.pmf(c, 16, [1 / 3] * 3)
              for c in ([i, j, 16 - i - j] for i in range(17) for j in range(17 - i))
              if ((np.array(c) - exp) ** 2 / exp).sum() >= chi - 1e-9)
    assert _row(t, "Exact Sig. (2-tailed)")[1] == ne.fmt_p(ref)
    sec = _run(ndlg.BinomialDialog, small_ds, {"vars": ["bin"]}, {"exact": EXACT})
    t = _table(sec, lambda t: t.title == "Binomial Test")
    assert t.columns[-1] == "Point Probability"
    assert t.rows[0][-1] == ne.fmt(stats.binom.pmf(4, 16, 0.5))
    sec = _run(ndlg.RunsDialog, small_ds, {"vars": ["a"]}, {"median": True, "exact": EXACT})
    assert any(r[0] == "Exact Sig. (2-tailed)" for r in _table(
        sec, lambda t: t.title == "Runs Test").rows)


def test_crosstabs_exact_button(qapp, small_ds):
    sec = _run(edlg.CrosstabsDialog, small_ds, {"rows": ["g"], "cols": ["bin"]},
               {"chisq": True, "exact": EXACT})
    t = _table(sec, lambda t: t.title == "Chi-Square Tests")
    assert "Exact Sig. (2-sided)" in t.columns and "Point Probability" in t.columns
    fisher = _row(t, "Fisher's Exact Test")
    obs = pd.crosstab(small_ds.df.g, small_ds.df["bin"]).to_numpy()
    assert fisher[t.columns.index("Exact Sig. (2-sided)")] == ne.fmt_p(
        stats.fisher_exact(obs)[1])


def test_exact_dialog_values(qapp):
    dlg = ExactDialog({"method": "mc", "mc_samples": 2000})
    assert dlg.values()["method"] == "mc" and dlg.values()["mc_samples"] == 2000
    assert not dlg.limit.isEnabled()
    dlg.exact.setChecked(True)
    assert dlg.limit.isEnabled() and dlg.values()["method"] == "exact"
    dlg.asym.setChecked(True)
    assert dlg.values()["method"] == "asymptotic"

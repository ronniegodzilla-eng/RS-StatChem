"""WP1: Proportions procedures and the summary-statistics t test against statsmodels/scipy."""
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import (confint_proportions_2indep, proportion_confint)

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import proportions as pr
from rsstatchem.core.stats import ttest


def num(text: str) -> float:
    return float(text.rstrip("*ᵃᵇᶜᵈ").replace(",", ""))


@pytest.mark.parametrize("x,n", [(7, 25), (0, 12), (30, 30), (48, 120)])
def test_one_sample_intervals(x, n):
    methods = [("wilson", "wilson"), ("clopper_pearson", "beta"),
               ("agresti_coull", "agresti_coull")]
    if 0 < x < n:  # Jeffreys: bounds set to 0 / 1 at the extremes (Brown, Cai & DasGupta)
        methods.append(("jeffreys", "jeffreys"))
    for ours, theirs in methods:
        lo, hi = pr.one_ci(x, n, ours)
        rlo, rhi = proportion_confint(x, n, method=theirs)
        assert math.isclose(lo, max(rlo, 0), abs_tol=1e-7), ours
        assert math.isclose(hi, min(rhi, 1), abs_tol=1e-7), ours
    lo, hi = pr.one_ci(x, n, "wald")
    rlo, rhi = proportion_confint(x, n, method="normal")
    assert math.isclose(lo, max(rlo, 0), abs_tol=1e-9)
    # Wilson with continuity correction (Newcombe 1998, method 4) contains plain Wilson
    clo, chi = pr.one_ci(x, n, "wilson_cc")
    wlo, whi = pr.one_ci(x, n, "wilson")
    assert clo <= wlo + 1e-12 and chi >= whi - 1e-12


def test_newcombe_published_values():
    # Newcombe (1998), Stat Med 17:857-872, Table I: 81/263 -> Wilson 0.2553-0.3662,
    # with continuity correction 0.2535-0.3682
    lo, hi = pr.one_ci(81, 263, "wilson")
    assert (round(lo, 4), round(hi, 4)) == (0.2553, 0.3662)
    lo, hi = pr.one_ci(81, 263, "wilson_cc")
    assert (round(lo, 4), round(hi, 4)) == (0.2535, 0.3682)


def test_one_sample_tests():
    z, one, two = pr.one_test(16, 40, 0.5, "exact")
    ref = stats.binomtest(16, 40, 0.5, alternative="less").pvalue
    assert math.isclose(one, ref, rel_tol=1e-9) and math.isclose(two, 2 * ref, rel_tol=1e-9)
    z, one, two = pr.one_test(16, 40, 0.5, "score")
    assert math.isclose(z, (0.4 - 0.5) / math.sqrt(0.25 / 40), rel_tol=1e-12)
    z, *_ = pr.one_test(16, 40, 0.5, "wald")
    assert math.isclose(z, -0.1 / math.sqrt(0.4 * 0.6 / 40), rel_tol=1e-12)


def test_independent_intervals():
    x1, n1, x2, n2 = 56, 70, 48, 80
    for ours, theirs in (("newcombe", "newcomb"), ("agresti_caffo", "agresti-caffo"),
                         ("wald", "wald")):
        lo, hi = pr.ind_ci(x1, n1, x2, n2, ours)
        rlo, rhi = confint_proportions_2indep(x1, n1, x2, n2, method=theirs)
        assert math.isclose(lo, rlo, abs_tol=1e-7), ours
        assert math.isclose(hi, rhi, abs_tol=1e-7), ours
    # Newcombe (1998b) example 56/70 vs 48/80: hybrid score interval 0.0524 to 0.3339
    lo, hi = pr.ind_ci(56, 70, 48, 80, "newcombe")
    assert (round(lo, 4), round(hi, 4)) == (0.0524, 0.3339)
    z, one, two = pr.ind_test(x1, n1, x2, n2, "wald_h0")
    pb = (x1 + x2) / (n1 + n2)
    zr = (x1 / n1 - x2 / n2) / math.sqrt(pb * (1 - pb) * (1 / n1 + 1 / n2))
    assert math.isclose(z, zr, rel_tol=1e-12)
    chi2 = stats.chi2_contingency([[x1, n1 - x1], [x2, n2 - x2]], correction=False)[0]
    assert math.isclose(z * z, chi2, rel_tol=1e-9)


def test_paired():
    a, b, c, d = 30, 12, 4, 20
    _, one, two = pr.pair_test(a, b, c, d, "exact")
    assert math.isclose(two, mcnemar([[a, b], [c, d]], exact=True).pvalue, rel_tol=1e-9)
    z, _, _ = pr.pair_test(a, b, c, d, "mcnemar_cc")
    assert math.isclose(z * z, mcnemar([[a, b], [c, d]], exact=False, correction=True)
                        .statistic, rel_tol=1e-9)
    lo, hi = pr.pair_ci(a, b, c, d, "newcombe")
    diff = (b - c) / (a + b + c + d)
    assert lo < diff < hi
    # with no association (phi = 0) Newcombe's paired interval combines the two Wilson
    # intervals exactly like the independent-samples hybrid score interval
    a, b, c, d = 10, 10, 10, 10
    lo, hi = pr.pair_ci(a, b, c, d, "newcombe")
    l1, u1 = pr.one_ci(20, 40, "wilson")
    assert math.isclose(hi, math.sqrt(2) * (u1 - 0.5), rel_tol=1e-9)


def test_procedures_output():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"pass": rng.integers(0, 2, 60).astype(float),
                       "after": rng.integers(0, 2, 60).astype(float),
                       "grp": np.repeat([1., 2.], 30)})
    ds = Dataset(df)
    sec = pr.one_sample(ds, ["pass"], 0.5, pr.ONE_CI, pr.ONE_TEST)
    t = sec.items[1]
    assert t.title == "One-Sample Proportions Confidence Intervals" and len(t.rows) == 9
    sec = pr.independent(ds, ["pass"], "grp", groups=(1.0, 2.0),
                         intervals=pr.IND_CI, tests=pr.IND_TEST)
    titles = [getattr(i, "title", "") for i in sec.items]
    assert "Independent-Samples Proportions Tests" in titles
    sec = pr.paired(ds, [("pass", "after")], pr.PAIR_CI, pr.PAIR_TEST)
    assert sec.items[-1].title == "Paired-Samples Proportions Tests"


def test_summary_t_test():
    sec = ttest.summary_independent(20, 10.5, 2.1, 25, 9.2, 3.0)
    t = next(i for i in sec.items if getattr(i, "title", "") == "Independent Samples Test")
    pooled = stats.ttest_ind_from_stats(10.5, 2.1, 20, 9.2, 3.0, 25)
    welch = stats.ttest_ind_from_stats(10.5, 2.1, 20, 9.2, 3.0, 25, equal_var=False)
    assert math.isclose(num(t.rows[0][3]), pooled.statistic, abs_tol=5e-4)
    assert math.isclose(num(t.rows[0][6]), pooled.pvalue, abs_tol=5e-4)
    assert math.isclose(num(t.rows[1][3]), welch.statistic, abs_tol=5e-4)


def test_paired_t_listwise():
    df = pd.DataFrame({"a": [1., 2, 3, 4, 5], "b": [2., 2, 4, 5, 6],
                       "c": [1., np.nan, 2, 3, 4]})
    ds = Dataset(df)
    sec = ttest.paired(ds, [("a", "b"), ("a", "c")], listwise=True)
    stats_t = sec.items[1]
    assert stats_t.rows[0][3] == "4"  # case 2 dropped for every pair

"""Power Analysis (WP5) against G*Power published values, statsmodels.stats.power, pingouin and
direct enumeration."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.power import FTestAnovaPower, TTestIndPower, TTestPower
from statsmodels.stats.proportion import power_proportions_2indep

from rsstatchem.core.stats import power as pw


def test_one_sample_t_gpower():
    # G*Power: d = .5, alpha = .05 two-sided, power .80 -> N = 34, actual power .8078
    d = pw.OneSampleT(mean=0.5, null=0.0, sd=1.0)
    assert pw.solve_n(d, 0.8) == 34
    assert d.power(34) == pytest.approx(0.8077775, abs=1e-6)
    assert d.power(34) == pytest.approx(TTestPower().power(0.5, 34, 0.05), abs=1e-10)
    one = pw.OneSampleT(sided="one", mean=0.5, null=0.0, sd=1.0)
    assert pw.solve_n(one, 0.8) == 27  # G*Power one-tailed: 27
    assert one.power(27) == pytest.approx(TTestPower().power(0.5, 27, 0.05,
                                                             alternative="larger"), abs=1e-10)
    neg = pw.OneSampleT(sided="one", mean=-0.5, null=0.0, sd=1.0)
    assert neg.power(27) == pytest.approx(one.power(27))


def test_paired_t_from_sds_and_correlation():
    sd = pw.paired_sd(10, 12, 0.6)
    assert sd == pytest.approx(math.sqrt(100 + 144 - 2 * 0.6 * 120))
    d = pw.PairedT(mean=5, null=0, sd=sd)
    n = pw.solve_n(d, 0.9)
    assert TTestPower().power(5 / sd, n, 0.05) >= 0.9 > TTestPower().power(5 / sd, n - 1, 0.05)


def test_independent_t_gpower_and_ratio():
    d = pw.IndependentT(mean1=0.5, mean2=0, sd1=1)
    assert pw.solve_n(d, 0.8) == 64            # G*Power: 64 per group, actual .8015
    assert d.power(64) == pytest.approx(0.8014596, abs=1e-6)
    d2 = pw.IndependentT(mean1=0.5, mean2=0, sd1=1, ratio=2)
    n = pw.solve_n(d2, 0.8)
    assert d2.sizes(n) == [48, 96]
    assert d2.power(n) == pytest.approx(TTestIndPower().power(0.5, n, 0.05, ratio=2))
    # Welch: unequal SDs use the Satterthwaite df
    w = pw.IndependentT(mean1=1, mean2=0, sd1=1, sd2=2)
    v1, v2 = 1 / 20, 4 / 20
    df = (v1 + v2) ** 2 / (v1 ** 2 / 19 + v2 ** 2 / 19)
    tc = stats.t.ppf(0.975, df)
    ncp = 1 / math.sqrt(v1 + v2)
    assert w.power(20) == pytest.approx(stats.nct.sf(tc, df, ncp) + stats.nct.cdf(-tc, df, ncp))


def test_anova_gpower_and_contrasts():
    a = 0.75 / math.sqrt(5)          # four equally spaced means with f = .25
    d = pw.OneWayAnova(means=(-a, -a / 3, a / 3, a), sd=1.0)
    assert d.effect() == pytest.approx(0.25)
    assert pw.solve_n(d, 0.8) == 45  # G*Power: total N = 180
    assert d.power(45) == pytest.approx(FTestAnovaPower().power(0.25, 180, 0.05, 4))
    c = pw.AnovaContrast(means=(0, 0.5, 1.0), sd=1.0, coefficients=(-1, 0, 1))
    n = 20
    se = math.sqrt(2 / n)
    tc = stats.t.ppf(0.975, 3 * n - 3)
    assert c.power(n) == pytest.approx(stats.nct.sf(tc, 3 * n - 3, 1 / se) +
                                       stats.nct.cdf(-tc, 3 * n - 3, 1 / se))
    b = pw.AnovaContrast(means=(0, 0.5, 1.0), sd=1.0, coefficients=(-1, 0, 1), adjust_k=3,
                         adjust="bonferroni")
    assert b.power(n) < c.power(n)
    # unequal group-size weights
    wd = pw.OneWayAnova(means=(0, 1), sd=1.0, weights=(1, 2))
    assert list(wd.group_sizes(10)) == [10, 20]


def test_regression_gpower():
    # G*Power linear multiple regression, f2 = .15, 5 predictors, power .80 -> N = 92
    d = pw.LinearRegression(r2_full=0.15 / 1.15, predictors=5, tested=5)
    assert d.effect() == pytest.approx(0.15)
    assert pw.solve_n(d, 0.8) == 92
    assert d.power(92) == pytest.approx(0.8042, abs=5e-4)
    nested = pw.LinearRegression(r2_full=0.3, r2_reduced=0.2, predictors=4, tested=1)
    assert nested.partial_r2() == pytest.approx(0.125)


def test_correlations():
    d = pw.Correlation(rho=0.3)
    n_closed = ((stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) / math.atanh(0.3)) ** 2 + 3
    assert pw.solve_n(d, 0.8) == math.ceil(n_closed) == 85
    sp = pw.Correlation(rho=0.3, kind="spearman", spearman_var="fieller")
    assert pw.solve_n(sp, 0.8) > 85
    pc = pw.Correlation(rho=0.3, kind="partial", partial_k=2)
    assert pw.solve_n(pc, 0.8) == 87
    cc = pw.Correlation(rho=0.3, kind="spearman", spearman_var="caruso")
    z1, n = math.atanh(0.3), 50
    se1 = math.sqrt(1 / (n - 2) + z1 / (6 * n + 4 * math.sqrt(n)))
    se0 = math.sqrt(1 / (n - 2))
    zc = stats.norm.ppf(0.975)
    expect = stats.norm.cdf((z1 - zc * se0) / se1) + stats.norm.cdf((-z1 - zc * se0) / se1)
    assert cc.power(n) == pytest.approx(expect)


def test_proportions():
    d = pw.IndependentBinomial(p1=0.65, p2=0.85)
    n = pw.solve_n(d, 0.8)
    assert n == 73
    assert d.power(n) == pytest.approx(power_proportions_2indep(-0.2, 0.85, n).power)
    # exact one-sample binomial power equals direct enumeration of the rejection region
    e = pw.OneSampleBinomial(p=0.65, p0=0.5)
    n = 60
    k = np.arange(n + 1)
    reject = (stats.binom.cdf(k, n, 0.5) <= 0.025) | (stats.binom.sf(k - 1, n, 0.5) <= 0.025)
    assert e.power(n) == pytest.approx(stats.binom.pmf(k[reject], n, 0.65).sum())
    nn = pw.solve_n(e, 0.8)
    assert e.power(nn) >= 0.8 and all(e.power(m) < 0.8 for m in range(2, nn))
    # McNemar normal approximation (Connor 1987) closed form
    m = pw.RelatedBinomial(p12=0.25, p21=0.10, method="normal")
    psi, delta = 0.35, 0.15
    z = stats.norm.ppf(0.975)
    n = 100
    expect = stats.norm.cdf((delta * math.sqrt(n) - z * math.sqrt(psi)) /
                            math.sqrt(psi - delta ** 2))
    assert m.power(n) == pytest.approx(expect, abs=1e-4)
    exact = pw.RelatedBinomial(p12=0.25, p21=0.10)
    assert abs(exact.power(120) - m.power(120)) < 0.06
    f = pw.IndependentBinomial(p1=0.2, p2=0.6, test="fisher")
    assert 0 < f.power(20) < pw.IndependentBinomial(p1=0.2, p2=0.6).power(20)


def test_precision_and_output():
    sizes, w = pw.precision_n("one", 1.0, 0.25)
    n = sizes[0]
    assert stats.t.ppf(0.975, n - 1) / math.sqrt(n) <= 0.25 < \
        stats.t.ppf(0.975, n - 2) / math.sqrt(n - 1)
    sec = pw.power_analysis("one_t", [pw.OneSampleT(mean=0.5)], "n", (0.8, 0.9),
                            plots={"by_n": True, "by_effect": True},
                            precision={"kind": "one", "sd": 1.0, "half_widths": [0.25]})
    tables = [i for i in sec.items if i.kind == "table"]
    assert tables[0].title == "Power Analysis Table"
    assert [r[1] for r in tables[0].rows] == ["34", "44"]
    assert any("Two-sided" in f for f in tables[0].footnotes)
    assert sum(i.kind == "image" for i in sec.items) == 2
    pw.selftest()

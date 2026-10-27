"""WP3: stand-alone P-P / Q-Q plots, Percentiles and Normality Analysis.

References: numpy's Hyndman-Fan percentile definitions (HAVERAGE = Weibull, WAVERAGE =
interpolated inverted CDF, EMPIRICAL = inverted CDF, AEMPIRICAL = averaged inverted CDF);
Conover (1999) for the order-statistic median interval (n = 20 -> X(6), X(15), 95.86 %);
the R package MVN vignette (Korkmaz, Goksuluk & Zararsiz 2014, R Journal 6/2) for iris setosa:
Mardia skewness 25.6643 (p .1772), kurtosis 1.2950 (p .1953), Henze-Zirkler 0.9488
(p .0500), Royston 31.5180 (p 2.1877e-06).
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import distplots as D

SHOTS = ("/private/tmp/claude-501/-Users-ronisaputra-Aplikasi-statistik/"
         "a8ca05d2-7214-4d46-8f3d-8e0d4372bfcd/scratchpad/wp3")

SETOSA = np.array([float(v) for v in """
5.1 3.5 1.4 0.2 4.9 3.0 1.4 0.2 4.7 3.2 1.3 0.2 4.6 3.1 1.5 0.2 5.0 3.6 1.4 0.2
5.4 3.9 1.7 0.4 4.6 3.4 1.4 0.3 5.0 3.4 1.5 0.2 4.4 2.9 1.4 0.2 4.9 3.1 1.5 0.1
5.4 3.7 1.5 0.2 4.8 3.4 1.6 0.2 4.8 3.0 1.4 0.1 4.3 3.0 1.1 0.1 5.8 4.0 1.2 0.2
5.7 4.4 1.5 0.4 5.4 3.9 1.3 0.4 5.1 3.5 1.4 0.3 5.7 3.8 1.7 0.3 5.1 3.8 1.5 0.3
5.4 3.4 1.7 0.2 5.1 3.7 1.5 0.4 4.6 3.6 1.0 0.2 5.1 3.3 1.7 0.5 4.8 3.4 1.9 0.2
5.0 3.0 1.6 0.2 5.0 3.4 1.6 0.4 5.2 3.5 1.5 0.2 5.2 3.4 1.4 0.2 4.7 3.2 1.6 0.2
4.8 3.1 1.6 0.2 5.4 3.4 1.5 0.4 5.2 4.1 1.5 0.1 5.5 4.2 1.4 0.2 4.9 3.1 1.5 0.2
5.0 3.2 1.2 0.2 5.5 3.5 1.3 0.2 4.9 3.6 1.4 0.1 4.4 3.0 1.3 0.2 5.1 3.4 1.5 0.2
5.0 3.5 1.3 0.3 4.5 2.3 1.3 0.3 4.4 3.2 1.3 0.2 5.0 3.5 1.6 0.6 5.1 3.8 1.9 0.4
4.8 3.0 1.4 0.3 5.1 3.8 1.6 0.2 4.6 3.2 1.4 0.2 5.3 3.7 1.5 0.2 5.0 3.3 1.4 0.2
""".split()]).reshape(50, 4)


def _table(section, title):
    return next(t for t in section.items if getattr(t, "title", "") == title)


def _ds(n=80, seed=11):
    rng = np.random.default_rng(seed)
    return Dataset(pd.DataFrame({"x": rng.normal(50, 8, n), "y": rng.gamma(2.5, 3, n),
                                 "z": rng.normal(size=n), "g": rng.integers(1, 3, n) * 1.0}))


# ---------------------------------------------------------------------------- percentiles
@pytest.mark.parametrize("method,ref", [("haverage", "weibull"),
                                        ("waverage", "interpolated_inverted_cdf"),
                                        ("empirical", "inverted_cdf"),
                                        ("aempirical", "averaged_inverted_cdf")])
def test_percentile_definitions_match_numpy(method, ref):
    rng = np.random.default_rng(1)
    for n in (1, 2, 3, 5, 10, 17, 40):
        x = np.round(rng.normal(size=n), 2)
        for p in (1, 5, 10, 25, 30, 50, 62.5, 75, 90, 95, 99):
            assert D.spss_percentile(x, p, method) == pytest.approx(
                np.percentile(x, p, method=ref), abs=1e-12)


def test_round_definition_and_hinges():
    x = np.arange(1.0, 11.0)
    assert D.spss_percentile(x, 25, "round") == 3.0      # np = 2.5 -> X(3)
    assert D.spss_percentile(x, 24, "round") == 2.0      # np = 2.4 -> X(2)
    assert D.spss_percentile(x, 1, "round") == 1.0       # index 0 clamps to X(1)
    assert D.tukey_hinges(np.arange(1.0, 9.0)) == (2.5, 4.5, 6.5)
    assert D.tukey_hinges(np.arange(1.0, 11.0)) == (3.0, 5.5, 8.0)
    assert D.tukey_hinges(np.arange(1.0, 10.0)) == (3.0, 5.0, 7.0)


def test_order_statistic_interval():
    x = np.arange(1.0, 21.0)
    lo, hi, cov = D.percentile_ci(x, 50, 0.95)
    assert (lo, hi) == (6.0, 15.0)
    assert cov == pytest.approx(0.9586, abs=1e-4)
    for p in (10, 25, 75, 90):
        assert D.percentile_ci(np.arange(1.0, 201.0), p, 0.95)[2] >= 0.95


def test_percentiles_section_by_factor():
    ds = _ds()
    sec = D.percentiles(ds, ["x", "y"], "g", (10, 25, 50, 75, 90), "waverage")
    t = _table(sec, "Percentiles")
    x1 = ds.df.x[ds.df.g == 1].to_numpy()
    first = t.rows[0]
    assert first[2].startswith("Weighted Average(Definition 2)")
    assert float(first[5]) == pytest.approx(
        np.percentile(x1, 50, method="interpolated_inverted_cdf"), abs=0.01)
    hinge = t.rows[1]
    assert hinge[2] == "Tukey's Hinges" and hinge[3] == "" and hinge[4] != ""
    assert _table(sec, "Percentile Confidence Intervals").rows


# ---------------------------------------------------------------------------- normality
def test_mardia_hz_royston_match_mvn_setosa():
    m = D.mardia(SETOSA)
    assert m["skew"] == pytest.approx(25.6643, abs=5e-4)
    assert m["skew_p"] == pytest.approx(0.1772, abs=5e-4)
    assert m["kurt"] == pytest.approx(1.2950, abs=5e-4)
    assert m["kurt_p"] == pytest.approx(0.1953, abs=5e-4)
    h = D.henze_zirkler(SETOSA)
    assert h["hz"] == pytest.approx(0.9488, abs=5e-4)
    assert h["p"] == pytest.approx(0.0500, abs=5e-4)
    r = D.royston(SETOSA)
    assert r["h"] == pytest.approx(31.5180, abs=5e-3)
    assert r["p"] == pytest.approx(2.1877e-06, rel=2e-3)


def test_univariate_tests_match_references():
    from statsmodels.stats.diagnostic import normal_ad

    x = _ds().df.y.to_numpy()
    r = D.univariate_tests(x)
    assert r["sw"][0] == pytest.approx(stats.shapiro(x).statistic)
    assert r["ad"] == pytest.approx(normal_ad(x))
    assert r["omnibus"][0] == pytest.approx(stats.normaltest(x).statistic)
    assert D.shapiro_francia(SETOSA[:, 0]) == pytest.approx(
        np.corrcoef(np.sort(SETOSA[:, 0]),
                    stats.norm.ppf((np.arange(1, 51) - 0.375) / 50.25))[0, 1] ** 2)


def test_normality_section():
    ds = Dataset(pd.DataFrame(SETOSA, columns=["sl", "sw", "pl", "pw"]))
    sec = D.normality(ds, ["sl", "sw", "pl", "pw"])
    mv = _table(sec, "Tests of Multivariate Normality")
    assert mv.rows[0][1] == "25.664" and mv.rows[3][0] == "Henze-Zirkler"
    assert _table(sec, "Tests of Univariate Normality").rows[0][0] == "sl"
    assert sum(getattr(i, "kind", "") == "image" for i in sec.items) == 5


# ---------------------------------------------------------------------------- PPLOT
def test_fractions_ranks_and_transforms():
    r = np.array([1.0, 2.0, 3.0])
    assert D.fractions(r, 3, "blom") == pytest.approx((r - 0.375) / 3.25)
    assert D.fractions(r, 3, "rankit") == pytest.approx((r - 0.5) / 3)
    assert D.fractions(r, 3, "tukey") == pytest.approx((r - 1 / 3) / (3 + 1 / 3))
    assert D.fractions(r, 3, "vw") == pytest.approx(r / 4)
    x = np.array([3.0, 1.0, 3.0, 2.0])
    assert list(D.ranks(x, "mean")) == [3.5, 1, 3.5, 2]
    assert list(D.ranks(x, "high")) == [4, 1, 4, 2]
    assert list(D.ranks(x, "low")) == [3, 1, 3, 2]
    y = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
    assert list(D.transform(y, diff=1)) == [3, 5, 7, 9]
    assert list(D.transform(y, diff=2)) == [2, 2, 2]
    assert list(D.transform(y, sdiff=1, period=2)) == [8, 12, 16]
    z = D.transform(y, log=True, standardize=True)
    assert z.mean() == pytest.approx(0) and z.std(ddof=1) == pytest.approx(1)
    assert np.isnan(D.transform(np.array([-1.0, 2.0]), log=True)[0])


def test_estimators():
    y = _ds().df.y.to_numpy()
    m, s = y.mean(), y.std(ddof=1)
    assert D.estimate("normal", y) == pytest.approx((m, s))
    assert D.estimate("gamma", y) == pytest.approx((m * m / s ** 2, m / s ** 2))
    assert D.estimate("exponential", y) == pytest.approx((1 / m,))
    assert D.estimate("logistic", y) == pytest.approx((m, s * math.sqrt(3) / math.pi))
    assert D.estimate("laplace", y) == pytest.approx((m, s / math.sqrt(2)))
    lg = np.log(y)
    assert D.estimate("lognormal", y) == pytest.approx((math.exp(lg.mean()), lg.std(ddof=1)))
    assert D.estimate("uniform", y) == pytest.approx((y.min(), y.max()))
    assert D.estimate("pareto", y)[1] == pytest.approx(len(y) / np.log(y / y.min()).sum())
    assert D.estimate("chisq", y, 4) == (4.0,)
    # a gamma with the fitted moments has the sample mean and variance
    g = D.frozen("gamma", D.estimate("gamma", y))
    assert g.mean() == pytest.approx(m) and g.var() == pytest.approx(s ** 2)
    w = D.frozen("weibull", D.estimate("weibull", y))
    assert w.cdf(np.median(y)) == pytest.approx(0.5, abs=0.1)


def test_probability_points():
    x = np.array([2.0, 5.0, 1.0, 4.0, 3.0])
    p = D.probability_points(x, "normal", (3.0, 1.5), "qq", "blom")
    frac = (np.arange(1, 6) - 0.375) / 5.25
    assert p["x"] == pytest.approx(np.sort(x))
    assert p["y"] == pytest.approx(stats.norm.ppf(frac, 3.0, 1.5))
    pp = D.probability_points(x, "normal", (3.0, 1.5), "pp", "vw")
    assert pp["x"] == pytest.approx(np.arange(1, 6) / 6)
    assert pp["y"] == pytest.approx(stats.norm.cdf(np.sort(x), 3.0, 1.5))


def test_pplot_section_all_distributions():
    ds = _ds()
    ds.add_variable("u", values=np.linspace(0.05, 0.95, ds.n_cases))
    for dist in D.DISTRIBUTIONS:
        var = "u" if dist == "beta" else "y"
        sec = D.pplot(ds, [var], "qq", dist, df=5.0)
        est = _table(sec, "Estimated Distribution Parameters")
        assert est.rows[0][0].startswith(D.DISTRIBUTIONS[dist][0]), dist
        assert sum(getattr(i, "kind", "") == "image" for i in sec.items) == 2
    sec = D.pplot(ds, ["x"], "pp", "normal", estimate_params=False, params=(50.0, 8.0))
    model = _table(sec, "Model Description")
    assert ["", "Location", "50"] in model.rows
    est = _table(sec, "Estimated Distribution Parameters")
    assert est.rows[0][2] == "50.00000"
    sec = D.pplot(ds, ["z"], "pp", "lognormal")  # negative values -> warning, no plot
    assert any(getattr(i, "style", "") == "warning" for i in sec.items)


# ---------------------------------------------------------------------------- dialogs
def _take(dlg, key, *names):
    target = next(t for t in dlg._targets if t.key == key)
    for name in names:
        dlg._take(name, target)
    dlg._update_arrows()


def test_dialogs_compute_and_screenshots(qapp):
    from rsstatchem.ui.dialogs import distplots as U

    os.makedirs(SHOTS, exist_ok=True)
    ds = _ds()
    for cls, fill, shot in ((U.PPPlotsDialog, {"vars": ["x"]}, "pplot.png"),
                            (U.QQPlotsDialog, {"vars": ["y"]}, "qqplot.png"),
                            (U.PercentilesDialog, {"vars": ["x"], "factor": ["g"]},
                             "percentiles.png"),
                            (U.NormalityDialog, {"vars": ["x", "y", "z"]}, "normality.png")):
        dlg = cls(ds)
        for key, names in fill.items():
            _take(dlg, key, *names)
        assert dlg.ok_button.isEnabled()
        assert dlg.validate() is None
        dlg.grab().save(os.path.join(SHOTS, shot))
        sec = dlg.compute()
        assert sec.items, cls.__name__
    dlg = U.QQPlotsDialog(ds)
    dlg.dist.setCurrentIndex(dlg.dist.findData("t"))
    assert dlg.df.isEnabled() and not dlg.p1.isEnabled()
    assert set(U.MENUS["Analyze/Descriptive Statistics"][0]) == {"Ratio...", U.RatioDialog}
    U.selftest()

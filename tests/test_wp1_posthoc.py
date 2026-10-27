"""WP1: SPSS post hoc tests (Duncan, S-N-K, Tukey-b, REGW, GT2, Gabriel, Waller-Duncan,
Dunnett, Tamhane T2, Dunnett T3/C), homogeneous subsets and One-Way ANOVA extras."""
import math

import numpy as np
import pandas as pd
import pingouin as pg
import pytest
from scipy import optimize, stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import anova, posthoc_ext as P
from rsstatchem.core.stats.common import Group


def num(text: str) -> float:
    return float(text.rstrip("*ᵃᵇᶜᵈ").replace(",", ""))


def table(section, title):
    return next(i for i in section.items if getattr(i, "title", "").startswith(title))


# agricolae's 'sweetpotato' data (yield by virus treatment, CRD, 3 replicates)
SWEET = {"cc": [28.5, 21.7, 23.0], "fc": [14.9, 10.6, 13.1], "ff": [41.8, 39.2, 28.0],
         "oo": [38.2, 40.4, 32.1]}


@pytest.fixture
def sweet():
    groups = [Group(k, k, np.array(v)) for k, v in SWEET.items()]
    ssw = sum(((g.data - g.data.mean()) ** 2).sum() for g in groups)
    return groups, ssw / 8, 8


def _names(groups, order, sets):
    return [sorted(groups[order[i]].label for i in s) for s in sets]


def test_distribution_tables():
    # Duncan's (1955) significant studentized ranges r(.05; p, 20)
    for p, ref in ((2, 2.950), (3, 3.097), (4, 3.190), (5, 3.255)):
        assert math.isclose(stats.studentized_range.ppf(0.95 ** (p - 1), p, 20), ref,
                            abs_tol=6e-4)
    # Dunnett (1955) tables: 4 treatments + control, 20 df: 2.65 two-sided, 2.30 one-sided
    lam = np.full(4, math.sqrt(0.5))
    assert math.isclose(P.dunnett_crit(0.95, lam, 20), 2.65, abs_tol=6e-3)
    assert math.isclose(P.dunnett_crit(0.95, lam, 20, False), 2.30, abs_tol=6e-3)
    # studentized maximum modulus: k = 1 is |t|; Stoline-Ury table m(.05; 3, 20) = 2.59
    assert math.isclose(P.smm_ppf(0.95, 1, 12), stats.t.ppf(0.975, 12), rel_tol=1e-7)
    assert math.isclose(P.smm_ppf(0.95, 3, 20), 2.59, abs_tol=6e-3)


def test_waller_duncan_t():
    # Bayes rule limit (F -> inf, f -> inf) for K = 100: z with K = E[delta+]/E[delta-]
    f = lambda z: (stats.norm.pdf(z) + z * stats.norm.cdf(z)) / \
        (stats.norm.pdf(z) - z * stats.norm.sf(z)) - 100  # noqa: E731
    z = optimize.brentq(f, 0.5, 5)
    assert math.isclose(P.waller_t(100, 1e5, 3, 1000), z, abs_tol=5e-3)
    # decreasing in F, increasing in K
    assert P.waller_t(100, 2, 4, 20) > P.waller_t(100, 4, 4, 20) > P.waller_t(100, 10, 4, 20)
    assert P.waller_t(500, 5, 4, 30) > P.waller_t(100, 5, 4, 30) > P.waller_t(50, 5, 4, 30)
    assert P.waller_t(100, 0.8, 4, 20) == math.inf
    # agricolae waller.test(sweetpotato): F = 17.345, q = 3, f = 8 -> t = 2.235 (rounded)
    assert math.isclose(P.waller_t(100, 17.344783, 3, 8), 2.235, abs_tol=2e-3)


def test_subsets_sweetpotato(sweet):
    groups, mse, df = sweet
    opts = P.PostHocOptions()
    order, sets, sigs = P.homogeneous_subsets(groups, "tukey", mse, df, opts)
    # agricolae HSD.test: oo a, ff ab, cc bc, fc c
    assert _names(groups, order, sets) == [["cc", "fc"], ["cc", "ff"], ["ff", "oo"]]
    for meth in ("duncan", "snk", "regwq", "regwf", "tukeyb"):
        order, sets, _ = P.homogeneous_subsets(groups, meth, mse, df, opts)
        assert _names(groups, order, sets) == [["fc"], ["cc"], ["ff", "oo"]], meth
    order, sets, sigs = P.homogeneous_subsets(groups, "snk", mse, df, opts)
    q = (np.mean(SWEET["oo"]) - np.mean(SWEET["ff"])) / math.sqrt(mse / 3)
    assert math.isclose(sigs[-1], stats.studentized_range.sf(q, 2, df), rel_tol=1e-9)
    order, sets, sigs = P.homogeneous_subsets(groups, "waller", mse, df, opts, 17.344783)
    assert _names(groups, order, sets) == [["fc"], ["cc"], ["ff", "oo"]]


def test_pairwise_against_references(sweet):
    groups, mse, df = sweet
    opts = P.PostHocOptions()
    res = stats.dunnett(*[g.data for g in groups[:3]], control=groups[3].data,
                        random_state=1)
    dn = P.pairwise(groups, "dunnett", mse, df, opts)
    np.testing.assert_allclose([r[4] for r in dn], res.pvalue, atol=2e-4)
    ci = res.confidence_interval()
    np.testing.assert_allclose([r[5] for r in dn], ci.low, atol=5e-3)
    one = P.pairwise(groups, "dunnett", mse, df, P.PostHocOptions(dunnett_test="<"))
    ref1 = stats.dunnett(*[g.data for g in groups[:3]], control=groups[3].data,
                         alternative="less", random_state=1)
    np.testing.assert_allclose([r[4] for r in one], ref1.pvalue, atol=2e-4)
    first = P.pairwise(groups, "dunnett", mse, df, P.PostHocOptions(control="first"))
    assert all(r[1] == 0 for r in first)
    # equal n: Gabriel = GT2
    g2 = P.pairwise(groups, "gt2", mse, df, opts)
    ga = P.pairwise(groups, "gabriel", mse, df, opts)
    np.testing.assert_allclose([r[4] for r in g2], [r[4] for r in ga], rtol=1e-8)
    # with two groups, T2 and T3 reduce to Welch's t test
    two = groups[:2]
    welch = stats.ttest_ind(two[0].data, two[1].data, equal_var=False).pvalue
    assert math.isclose(P.pairwise(two, "t2", mse, df, opts)[0][4], welch, rel_tol=1e-8)
    assert math.isclose(P.pairwise(two, "t3", mse, df, opts)[0][4], welch, rel_tol=1e-6)
    c = P.pairwise(groups, "c", mse, df, opts)
    assert all(math.isnan(r[4]) for r in c) and all(r[5] < r[2] < r[6] for r in c)


def test_oneway_extras_and_posthoc_table():
    rng = np.random.default_rng(4)
    g = np.repeat([1., 2., 3., 4.], [8, 10, 9, 11])
    y = g * 1.2 + rng.normal(0, 2, len(g))
    df = pd.DataFrame({"y": y, "g": g})
    ds = Dataset(df)
    sec = anova.oneway(ds, ["y"], "g", posthoc_methods=["duncan", "snk", "dunnett", "t2", "c",
                                                       "gt2", "gabriel", "games_howell"],
                       polynomial=2, contrasts=[[1, -1, 0, 0], [1, 1, -1, -1]], effects=True)
    a = table(sec, "ANOVA")
    lin = next(r for r in a.rows if r[1] == "Linear Term")
    # unweighted linear SS = (c'm)^2 / sum(c^2/n) with orthonormal polynomial coefficients
    m = df.groupby("g").y.mean().to_numpy()
    n = df.groupby("g").y.size().to_numpy()
    c = np.array([-3, -1, 1, 3]) / math.sqrt(20)
    assert math.isclose(num(lin[3]), (c @ m) ** 2 / np.sum(c * c / n), rel_tol=1e-3)
    # weighted linear SS = regression SS of y on the level index
    import statsmodels.formula.api as smf

    df["x"] = df.g
    reg = smf.ols("y ~ x", df).fit()
    weighted = next(r for r in a.rows if r[2] == "Weighted")
    assert math.isclose(num(weighted[3]), reg.ess, rel_tol=1e-3)
    ct = table(sec, "Contrast Tests")
    r = ct.rows[0]
    msw = sum(((df.y[df.g == k] - df.y[df.g == k].mean()) ** 2).sum() for k in (1, 2, 3, 4)) \
        / (len(df) - 4)
    val = m[0] - m[1]
    assert math.isclose(num(r[3]), val, abs_tol=5e-4)
    assert math.isclose(num(r[4]), math.sqrt(msw * (1 / n[0] + 1 / n[1])), abs_tol=5e-4)
    welch_row = ct.rows[2]
    ref = stats.ttest_ind(df.y[df.g == 1], df.y[df.g == 2], equal_var=False)
    assert math.isclose(num(welch_row[5]), ref.statistic, abs_tol=5e-3)
    desc = table(sec, "Descriptives")
    assert any(r[2] == "Random Effects" for r in desc.rows)
    mc = table(sec, "Multiple Comparisons")
    labels = {r[1] for r in mc.rows if r[1]}
    assert {"Dunnett t (2-sided)ᵃ", "Tamhane", "Dunnett C", "Hochberg", "Gabriel",
            "Games-Howell"} <= labels
    hs = table(sec, "Homogeneous Subsets")
    assert any(r[0].startswith("Duncan") for r in hs.rows)
    assert any("harmonic mean" in f for f in hs.footnotes)
    gh = pg.pairwise_gameshowell(data=df, dv="y", between="g")
    ghr = [r for r in mc.rows if r[1] == "Games-Howell"][0]
    assert math.isclose(num(ghr[6]), gh.iloc[0]["pval"], abs_tol=2e-3)
    eff = table(sec, "ANOVA Effect Sizes")
    lo, hi = num(eff.rows[0][3]), num(eff.rows[0][4])
    assert lo <= num(eff.rows[0][2]) <= hi


def test_oneway_listwise_missing():
    df = pd.DataFrame({"y1": [1., 2, 3, 4, 5, 6, np.nan, 8, 9],
                       "y2": [2., 1, 4, 3, 6, 5, 8, 7, 9], "g": [1., 1, 1, 2, 2, 2, 3, 3, 3]})
    sec = anova.oneway(Dataset(df), ["y1", "y2"], "g", listwise=True)
    a = table(sec, "ANOVA")
    total_y2 = [r for r in a.rows if r[1] == "Total"][1]
    assert total_y2[3] == "7"  # 8 cases after listwise deletion

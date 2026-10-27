"""WP3: legacy NPAR TESTS additions (Analyze > Nonparametric Tests > Legacy Dialogs).

References: scipy.stats / statsmodels, brute-force permutation enumeration for the exact
null distributions (Moses: Siegel & Castellan 1988 §6.10 eq. 6.13; Wald-Wolfowitz runs:
Siegel & Castellan §6.7; Jonckheere-Terpstra tie-corrected variance: Hollander & Wolfe 1999
§6.2), and the SPSS NPAR TESTS algorithms for the marginal homogeneity statistic.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import nonparametric as npt
from rsstatchem.core.stats import nptests as T

SHOTS = "/private/tmp/claude-501/-Users-ronisaputra-Aplikasi-statistik/" \
        "a8ca05d2-7214-4d46-8f3d-8e0d4372bfcd/scratchpad/wp3"


def num(text: str) -> float:
    return float(text.rstrip("*ᵃᵇᶜᵈᵉ").replace(",", ""))


def table(section, title):
    return next(i for i in section.items if getattr(i, "title", "").startswith(title))


def row(tab, head):
    return next(r for r in tab.rows if r[0].startswith(head))


@pytest.fixture
def ds():
    rng = np.random.default_rng(2024)
    n = 60
    df = pd.DataFrame({
        "y": rng.normal(10, 2, n).round(1), "g2": np.tile([1.0, 2.0], n // 2),
        "g3": np.repeat([1.0, 2.0, 3.0], n // 3), "a": rng.integers(1, 4, n).astype(float),
        "b": rng.integers(1, 4, n).astype(float), "c": rng.integers(0, 2, n).astype(float),
        "d": rng.integers(0, 2, n).astype(float), "cnt": rng.poisson(3, n).astype(float),
        "w": rng.exponential(2.0, n), "u": rng.uniform(0, 5, n)})
    df.loc[3, "y"] = np.nan
    return Dataset(df)


# ------------------------------------------------------------------ chi-square / runs / K-S
def test_chi_square_expected_range_includes_empty_categories(ds):
    sec = npt.chi_square(ds, ["a"], expected_range=(0, 4))
    obs = [int((ds.df.a == k).sum()) for k in range(0, 5)]
    ref = stats.chisquare(obs)
    st = table(sec, "Test Statistics")
    assert num(row(st, "Chi-Square")[1]) == pytest.approx(ref.statistic, abs=5e-4)
    assert row(st, "df")[1] == "4"
    assert table(sec, "a").rows[0][2] == "0"  # category 0 is empty but listed


def test_chi_square_expected_values(ds):
    sec = npt.chi_square(ds, ["a"], expected=[1, 2, 1])
    obs = np.array([(ds.df.a == k).sum() for k in (1, 2, 3)], float)
    ref = stats.chisquare(obs, obs.sum() * np.array([.25, .5, .25]))
    assert num(table(sec, "Test Statistics").rows[0][1]) == pytest.approx(ref.statistic, abs=5e-4)


def test_runs_mode_and_custom_cut_points():
    from statsmodels.sandbox.stats.runs import runstest_1samp

    rng = np.random.default_rng(1)
    x = rng.integers(1, 6, 80).astype(float)
    vals, counts = np.unique(x, return_counts=True)
    mode = vals[counts == counts.max()].max()
    r = npt.runs_details(x, "mode")
    assert r["value"] == mode
    z, _ = runstest_1samp(x, cutoff=mode, correction=False)  # n >= 50: no correction in SPSS
    assert r["z"] == pytest.approx(z, abs=1e-6)
    r = npt.runs_details(x, "custom", 2.5)
    z, _ = runstest_1samp(x, cutoff=2.5, correction=False)
    assert r["value"] == 2.5 and r["z"] == pytest.approx(z, abs=1e-6)
    ds = Dataset(pd.DataFrame({"x": [1.0, 1, 2, 2, 3, 5, 5, 4]}))
    tab = table(npt.runs(ds, ["x"], "mode"), "Runs Test")
    assert any("Multiple modes" in f for f in tab.footnotes)
    assert tab.rows[0][1].startswith("5.00000")


@pytest.mark.parametrize("dist,col,args", [("uniform", "u", None), ("exponential", "w", None)])
def test_ks_one_sample_other_distributions(ds, dist, col, args):
    x = ds.df[col].to_numpy()
    r = T.ks_one_sample_test(x, dist)
    if dist == "uniform":
        ref = stats.kstest(x, "uniform", args=(x.min(), x.max() - x.min()), method="exact")
    else:
        ref = stats.kstest(x, "expon", args=(0, x.mean()), method="exact")
    assert r["absolute"] == pytest.approx(ref.statistic, abs=1e-12)
    assert r["p_exact"] == pytest.approx(ref.pvalue, abs=1e-10)
    assert r["p_kolmogorov"] == pytest.approx(stats.kstwobign.sf(math.sqrt(len(x)) *
                                                                  ref.statistic))
    sec = npt.ks_one_sample(ds, [col], dist, exact=True)
    tab = table(sec, "One-Sample Kolmogorov-Smirnov Test")
    assert f"a. Test distribution is {dist.capitalize()}." in tab.footnotes
    assert num(row(tab, "Exact Sig.")[2]) == pytest.approx(ref.pvalue, abs=5e-4)


def test_ks_poisson_matches_brute_force_supremum(ds):
    x = ds.df["cnt"].to_numpy()
    r = T.ks_one_sample_test(x, "poisson")
    lam = x.mean()
    grid = np.arange(-1, x.max() + 30, 0.25)  # sup over a fine grid of the real line
    s = np.array([(x <= g).mean() for g in grid])
    f = stats.poisson.cdf(np.floor(grid), lam)
    assert r["positive"] == pytest.approx(np.max(s - f), abs=1e-12)
    assert -r["negative"] == pytest.approx(max(np.max(f - s), 0), abs=1e-12)


def test_ks_exponential_skips_negative_values():
    ds = Dataset(pd.DataFrame({"x": [-1.0, 0.5, 1.2, 2.0, 3.1, 0.2, 4.0]}))
    tab = table(npt.ks_one_sample(ds, ["x"], "exponential"), "One-Sample")
    assert tab.rows[0][2] == "6"
    assert any("outside the specified distribution range" in f for f in tab.footnotes)


# ------------------------------------------------------------------ two independent
def test_two_sample_ks(ds):
    sec = npt.two_independent(ds, ["y"], "g2", (1.0, 2.0), False, True)
    a = ds.df.y[(ds.df.g2 == 1)].dropna()
    b = ds.df.y[(ds.df.g2 == 2)].dropna()
    ref = stats.ks_2samp(a, b)
    tab = table(sec, "Test Statisticsᵃ (Two-Sample")
    d = num(tab.rows[0][2])
    assert d == pytest.approx(ref.statistic, abs=5e-4)
    z = ref.statistic * math.sqrt(len(a) * len(b) / (len(a) + len(b)))
    assert num(row(tab, "Kolmogorov-Smirnov Z")[2]) == pytest.approx(z, abs=5e-4)
    assert num(row(tab, "Asymp. Sig.")[2]) == pytest.approx(stats.kstwobign.sf(z), abs=5e-4)


def _brute_span_cdf(m, n, h, span):
    hits = total = 0
    for pos in itertools.combinations(range(1, m + n + 1), m):
        p = sorted(pos)[h:m - h] if h else sorted(pos)
        total += 1
        hits += (p[-1] - p[0] + 1) <= span
    return hits / total


@pytest.mark.parametrize("m,n,h,span", [(5, 4, 0, 7), (5, 4, 1, 4), (6, 5, 1, 6), (4, 6, 0, 5)])
def test_moses_exact_distribution(m, n, h, span):
    assert T.moses_span_pvalue(span, m, n, h) == pytest.approx(_brute_span_cdf(m, n, h, span))


def test_moses_test_statistics():
    control = np.array([4.0, 5, 6, 7, 8, 9])
    exp_ = np.array([1.0, 2, 3, 10, 11, 12])
    r = T.moses_test(control, exp_)
    assert r["span"] == 6 and r["h"] == 1 and r["trimmed"] == 4
    assert r["p_span"] == pytest.approx(_brute_span_cdf(6, 6, 0, 6))


def test_wald_wolfowitz_exact_and_ties():
    rs, ps = T.runs_pmf(5, 4)
    counts = {}
    for pos in itertools.combinations(range(9), 5):
        lab = np.ones(9, int)
        lab[list(pos)] = 0
        r = 1 + int(np.count_nonzero(lab[1:] != lab[:-1]))
        counts[r] = counts.get(r, 0) + 1
    for r, p in zip(rs, ps):
        assert p == pytest.approx(counts[int(r)] / 126)
    a = np.array([1.0, 2, 2, 5, 7])
    b = np.array([2.0, 3, 5, 8])
    rmin, rmax, ties, cases = T.runs_range(np.r_[a, b], np.r_[np.zeros(5, int), np.ones(4, int)])
    seen = set()
    vals = np.r_[a, b]
    lab = np.r_[np.zeros(5, int), np.ones(4, int)]
    blocks = [lab[vals == v] for v in np.unique(vals)]
    for combo in itertools.product(*[set(itertools.permutations(bl)) for bl in blocks]):
        seq = np.concatenate([np.array(c) for c in combo])
        seen.add(1 + int(np.count_nonzero(seq[1:] != seq[:-1])))
    assert (rmin, rmax) == (min(seen), max(seen))
    assert ties == 2 and cases == 5


def test_two_independent_all_tables(ds):
    sec = npt.two_independent(ds, ["y"], "g2", (1.0, 2.0), True, True, True, True)
    titles = [getattr(i, "title", "") for i in sec.items]
    assert any("Moses" in t for t in titles) and any("Wald-Wolfowitz" in t for t in titles)
    mw = table(sec, "Test Statisticsᵃ")
    r = row(mw, "Effect size r")
    z = num(row(mw, "Z")[1])
    n = int(ds.df.y.notna().sum())
    assert num(r[1]) == pytest.approx(abs(z) / math.sqrt(n), abs=1e-3)
    assert any("not part of the SPSS output" in f for f in mw.footnotes)


# ------------------------------------------------------------------ k independent
def test_median_test_matches_scipy(ds):
    sec = npt.k_independent(ds, ["y"], "g3", 1, 3, kruskal=False, median=True)
    samples = [ds.df.y[ds.df.g3 == k].dropna() for k in (1, 2, 3)]
    ref = stats.median_test(*samples, ties="below", correction=False)
    tab = table(sec, "Test Statisticsᵃ (Median")
    assert num(row(tab, "Chi-Square")[1]) == pytest.approx(ref.statistic, abs=5e-4)
    assert num(row(tab, "Median")[1]) == pytest.approx(ref.median, abs=1e-6)
    freq = table(sec, "Frequencies")
    assert [int(v) for v in freq.rows[0][2:]] == list(ref.table[0])


def test_median_test_two_groups_yates_and_exact():
    rng = np.random.default_rng(3)
    a, b = rng.normal(0, 1, 25), rng.normal(.5, 1, 25)
    r = T.median_test([a, b])
    ref = stats.median_test(a, b, ties="below", correction=True)
    assert r["yates"] == pytest.approx(ref.statistic)
    assert r["p_exact"] == pytest.approx(stats.fisher_exact(ref.table)[1])


def _jt_perm_moments(samples):
    sizes = [len(s) for s in samples]
    values = np.concatenate(samples)
    js = []
    for lab in T._label_assignments(sizes):
        groups = [values[lab == g] for g in range(len(sizes))]
        js.append(T._jt_stat(groups))
    js = np.array(js)
    return js.mean(), js.var()


@pytest.mark.parametrize("ties", [False, True])
def test_jonckheere_variance_equals_permutation_variance(ties):
    rng = np.random.default_rng(7)
    x = rng.integers(1, 5, 9).astype(float) if ties else rng.normal(size=9)
    samples = [x[:3], x[3:5], x[5:]]
    r = T.jonckheere_test(samples)
    mean, var = _jt_perm_moments(samples)
    assert r["mean"] == pytest.approx(mean)
    assert r["sd"] ** 2 == pytest.approx(var, rel=1e-9)


def test_k_independent_jt_table(ds):
    sec = npt.k_independent(ds, ["y"], "g3", 1, 3, kruskal=True, jonckheere=True)
    tab = table(sec, "Jonckheere-Terpstra")
    assert row(tab, "Number of Levels")[1] == "3"
    assert row(tab, "Asymp. Sig.")[1]


# ------------------------------------------------------------------ two related
def test_mcnemar_matches_statsmodels(ds):
    from statsmodels.stats.contingency_tables import mcnemar

    sec = npt.two_related(ds, [("c", "d")], wilcoxon=False, mcnemar=True)
    t = pd.crosstab(ds.df.c, ds.df.d).to_numpy()
    changes = t[0, 1] + t[1, 0]
    tab = table(sec, "Test Statisticsᵃ (McNemar")
    if changes <= 25:
        assert num(tab.rows[-1][1]) == pytest.approx(mcnemar(t, exact=True).pvalue, abs=5e-4)
    else:
        ref = mcnemar(t, exact=False, correction=True)
        assert num(row(tab, "Chi-Square")[1]) == pytest.approx(ref.statistic, abs=5e-4)
        assert num(tab.rows[-1][1]) == pytest.approx(ref.pvalue, abs=5e-4)
    small = T.mcnemar_test(np.array([0, 0, 1, 1, 0, 1, 0, 0]), np.array([1, 1, 0, 1, 1, 1, 1, 0]))
    t2 = np.array([[1, 4], [1, 2]])
    assert small["exact"] and small["p"] == pytest.approx(mcnemar(t2, exact=True).pvalue)


def test_marginal_homogeneity_reduces_to_mcnemar_for_binary(ds):
    r = T.marginal_homogeneity_test(ds.df.c.to_numpy(), ds.df.d.to_numpy())
    t = pd.crosstab(ds.df.c, ds.df.d).to_numpy()
    b, c = t[0, 1], t[1, 0]
    assert r["z"] ** 2 == pytest.approx((b - c) ** 2 / (b + c))
    assert r["off"] == b + c


def test_marginal_homogeneity_moments_are_exact_permutation_moments():
    a = np.array([1.0, 2, 3, 3, 2, 1, 3])
    b = np.array([2.0, 2, 1, 3, 3, 3, 1])
    r = T.marginal_homogeneity_test(a, b)
    off = a != b
    sums = [sum(x if s else y for x, y, s in zip(a[off], b[off], swap))
            for swap in itertools.product([True, False], repeat=int(off.sum()))]
    assert r["mean"] == pytest.approx(np.mean(sums))
    assert r["sd"] == pytest.approx(np.std(sums))


def test_wilcoxon_effect_size_and_existing_rows(ds):
    sec = npt.two_related(ds, [("a", "b")], wilcoxon=True, sign=True)
    tab = table(sec, "Test Statisticsᵃ")
    d = (ds.df.b - ds.df.a).to_numpy()
    ref = stats.wilcoxon(d, correction=False, method="approx", zero_method="wilcox")
    assert num(tab.rows[1][1]) == pytest.approx(ref.pvalue, abs=5e-4)
    z = num(tab.rows[0][1])
    assert num(row(tab, "Effect size r")[1]) == pytest.approx(abs(z) / math.sqrt(len(d)),
                                                              abs=1e-3)


# ------------------------------------------------------------------ options
def test_descriptive_and_quartiles_table(ds):
    sec = npt.mann_whitney(ds, ["y"], "g2", (1.0, 2.0), descriptives=True, quartiles=True)
    tab = table(sec, "Descriptive Statistics")
    y = ds.df.y.dropna().to_numpy()
    r = tab.rows[0]
    assert r[1] == str(len(y))
    assert num(r[2]) == pytest.approx(y.mean(), abs=5e-3)
    assert num(r[3]) == pytest.approx(y.std(ddof=1), abs=5e-4)
    q = np.percentile(y, [25, 50, 75], method="weibull")
    assert [num(v) for v in r[6:9]] == pytest.approx(list(q), abs=5e-3)
    assert tab.rows[1][0] == "g2"  # the grouping variable is described too, as in SPSS


def test_listwise_exclusion(ds):
    sec = npt.k_related(ds, ["a", "b"], descriptives=True)
    assert table(sec, "Descriptive Statistics").rows[0][1] == "60"
    sec = npt.two_independent(ds, ["a", "y"], "g2", (1.0, 2.0), descriptives=True,
                              listwise=True)
    assert table(sec, "Descriptive Statistics").rows[0][1] == "59"
    assert "/MISSING LISTWISE" in sec.items[0].text


# ------------------------------------------------------------------ dialogs
def _take(dlg, key, *names):
    target = next(t for t in dlg._targets if t.key == key)
    for name in names:
        dlg._take(name, target)


def test_legacy_dialogs_compute_and_render(qapp, ds):
    from pathlib import Path

    from rsstatchem.ui.dialogs import nonparametric as ui

    Path(SHOTS).mkdir(parents=True, exist_ok=True)
    dlg = ui.TwoIndependentDialog(ds)
    _take(dlg, "vars", "y")
    _take(dlg, "group", "g2")
    dlg.opts.update(group_def={"var": "g2", "groups": (1.0, 2.0), "cut": None}, ks=True,
                    moses=True, ww=True, descriptives=True)
    sec = dlg.compute()
    assert table(sec, "Descriptive Statistics")
    dlg.grab().save(f"{SHOTS}/legacy_two_independent.png")

    dlg = ui.KIndependentDialog(ds)
    _take(dlg, "vars", "y")
    _take(dlg, "group", "g3")
    dlg.opts.update(group_def={"var": "g3", "lo": 1.0, "hi": 3.0}, median=True, jt=True)
    assert table(dlg.compute(), "Jonckheere-Terpstra")
    dlg.grab().save(f"{SHOTS}/legacy_k_independent.png")

    dlg = ui.TwoRelatedDialog(ds)
    dlg._add_pair(dlg._pairs[0], "c", "d")
    dlg.opts.update(mcnemar=True, marginal=True)
    assert table(dlg.compute(), "Marginal Homogeneity Test")
    dlg.grab().save(f"{SHOTS}/legacy_two_related.png")

    dlg = ui.OneSampleKSDialog(ds)
    _take(dlg, "vars", "w")
    dlg.opts.update(normal=True, exponential=True)
    sec = dlg.compute()
    assert sum(getattr(i, "title", "") == "One-Sample Kolmogorov-Smirnov Test"
               for i in sec.items) == 2
    dlg.grab().save(f"{SHOTS}/legacy_ks.png")

    dlg = ui.ChiSquareDialog(ds)
    _take(dlg, "vars", "a")
    dlg.use_range.setChecked(True)
    dlg.lower.setText("1")
    dlg.upper.setText("4")
    assert dlg.validate() is None
    assert table(dlg.compute(), "Test Statistics").rows[1][1] == "3"
    dlg.grab().save(f"{SHOTS}/legacy_chisquare.png")

    dlg = ui.RunsDialog(ds)
    _take(dlg, "vars", "y")
    dlg.opts.update(mode=True, custom=True)
    dlg.value.setText("10")
    assert dlg.validate() is None
    assert sum(getattr(i, "title", "") == "Runs Test" for i in dlg.compute().items) == 3
    dlg.grab().save(f"{SHOTS}/legacy_runs.png")

"""WP3: exact / Monte Carlo p-values and the SPSS NPTESTS procedures (Analyze >
Nonparametric Tests > One Sample / Independent Samples / Related Samples).

Exact p-values are checked against scipy (``mannwhitneyu``/``wilcoxon``/``ks_2samp`` exact,
and ``scipy.stats.permutation_test`` with full enumeration as an independent permutation
reference); asymptotic statistics against scipy and pingouin; Dunn's pairwise test against
the formula in Dunn (1964) as given by Siegel & Castellan (1988, eq. 8.5 with ties).
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pingouin as pg
import pytest
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import nptests as T

SHOTS = "/private/tmp/claude-501/-Users-ronisaputra-Aplikasi-statistik/" \
        "a8ca05d2-7214-4d46-8f3d-8e0d4372bfcd/scratchpad/wp3"


def table(section, title):
    return next(i for i in section.items if getattr(i, "title", "").startswith(title))


def num(text: str) -> float:
    return float(text.rstrip("ᵃᵇᶜᵈᵉ"))


# ------------------------------------------------------------------ exact p-values
def test_exact_mann_whitney_without_ties_equals_scipy():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=7), rng.normal(.8, 1, size=6)
    p = T.exact_mann_whitney(a, b)
    assert p.method == "exact"
    assert p.p == pytest.approx(stats.mannwhitneyu(a, b, method="exact").pvalue)


def test_exact_mann_whitney_with_ties_equals_full_permutation():
    a = np.array([1.0, 2, 2, 3, 5])
    b = np.array([2.0, 3, 4, 4, 6, 6])

    def stat(x, y):
        r = stats.rankdata(np.r_[x, y])[:len(x)].sum()
        return abs(r - len(x) * (len(x) + len(y) + 1) / 2)

    ref = stats.permutation_test((a, b), stat, permutation_type="independent",
                                 n_resamples=np.inf, alternative="greater").pvalue
    assert T.exact_mann_whitney(a, b).p == pytest.approx(ref)


def test_exact_wilcoxon_with_and_without_ties():
    rng = np.random.default_rng(2)
    d = rng.normal(.3, 1, 12)
    assert T.exact_wilcoxon(d).p == pytest.approx(stats.wilcoxon(d, method="exact").pvalue)
    d = np.array([1.0, -2, 2, 3, 3, -1, 4, 0, 5, 2])
    nz = d[d != 0]
    ranks = stats.rankdata(np.abs(nz))
    obs = ranks[nz > 0].sum()
    center = ranks.sum() / 2
    sums = [ranks[np.array(s, bool)].sum() for s in itertools.product([0, 1], repeat=len(nz))]
    ref = np.mean(np.abs(np.array(sums) - center) >= abs(obs - center) - 1e-9)
    assert T.exact_wilcoxon(d).p == pytest.approx(ref)


def test_monte_carlo_interval_and_seed():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=9), rng.normal(.7, 1, size=9)
    exact = T.exact_mann_whitney(a, b).p
    mc = T.exact_mann_whitney(a, b, method="mc")
    assert mc.method == "monte carlo" and mc.samples == T.MC_SAMPLES and mc.level == .99
    assert mc.lower <= exact <= mc.upper
    assert T.exact_mann_whitney(a, b, method="mc").p == mc.p  # SPSS default seed 2000000
    lo, hi = T.mc_interval(0.05, 10000, 0.99)
    assert (hi - lo) / 2 == pytest.approx(stats.norm.ppf(.995) * math.sqrt(.05 * .95 / 1e4))


def _kw(*s):
    return stats.kruskal(*s).statistic


def test_exact_kruskal_equals_permutation_test():
    s = [np.array([1.0, 3, 4]), np.array([2.0, 5, 7]), np.array([6.0, 8, 9])]
    ref = stats.permutation_test(s, _kw, permutation_type="independent", n_resamples=np.inf,
                                 alternative="greater").pvalue
    p = T.exact_kruskal(s)
    assert p.method == "exact" and p.p == pytest.approx(ref)


def test_exact_jonckheere_equals_permutation_test():
    s = [np.array([1.0, 2, 4]), np.array([3.0, 3, 6]), np.array([5.0, 8])]
    center = T.jonckheere_test(s)["mean"]

    def stat(*g):
        return abs(T._jt_stat(list(g)) - center)

    ref = stats.permutation_test(s, stat, permutation_type="independent", n_resamples=np.inf,
                                 alternative="greater").pvalue
    assert T.exact_jonckheere(s).p == pytest.approx(ref)


def test_exact_median_test_equals_permutation_test():
    s = [np.array([1.0, 2, 6, 7]), np.array([3.0, 8, 9]), np.array([4.0, 5, 10])]
    med = np.median(np.concatenate(s))

    def stat(*g):
        return T.median_test(list(g), med)["chi2"]

    ref = stats.permutation_test(s, stat, permutation_type="independent", n_resamples=np.inf,
                                 alternative="greater").pvalue
    assert T.exact_median_test(s).p == pytest.approx(ref)


def test_exact_friedman_and_cochran_equal_within_row_permutation():
    x = np.array([[1.0, 2, 3], [2, 3, 1], [1, 3, 2], [1, 2, 2], [3, 2, 1]])

    def fr(*cols):
        return T.friedman_test(np.column_stack(cols))["chi2"]

    ref = stats.permutation_test(tuple(x.T), fr, permutation_type="samples",
                                 n_resamples=np.inf, alternative="greater").pvalue
    assert T.exact_friedman(x).p == pytest.approx(ref)
    assert T.exact_kendall_w(x).p == pytest.approx(ref)
    b = np.array([[1.0, 0, 1], [1, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [1, 1, 1]])

    def q(*cols):
        return T.cochran_test(np.column_stack(cols))["q"]

    ref = stats.permutation_test(tuple(b.T), q, permutation_type="samples",
                                 n_resamples=np.inf, alternative="greater").pvalue
    assert T.exact_cochran(b).p == pytest.approx(ref)


def test_exact_chisq_gof_equals_multinomial_enumeration():
    obs = [3, 7, 2]
    n, k = sum(obs), 3
    stat_obs = T.chisq_gof(obs)["chi2"]
    total = 0.0
    for t in itertools.product(range(n + 1), repeat=k):
        if sum(t) != n:
            continue
        if T.chisq_gof(t)["chi2"] >= stat_obs - 1e-9:
            total += stats.multinomial.pmf(t, n, [1 / 3] * 3)
    assert T.exact_chisq_gof(obs).p == pytest.approx(total)


def test_exact_ks_runs_sign_binomial_mcnemar():
    rng = np.random.default_rng(4)
    a, b = rng.normal(size=8), rng.normal(1, 1, size=7)
    assert T.exact_ks_2samp(a, b).p == pytest.approx(stats.ks_2samp(a, b, method="exact").pvalue)
    d = np.array([1.0, -1, 2, 3, 0, 4, 5, -2, 6])
    assert T.exact_sign(d).p == pytest.approx(stats.binomtest(2, 8).pvalue)
    assert T.exact_binomial(3, 12, .5).p == pytest.approx(stats.binomtest(3, 12).pvalue)
    assert T.exact_mcnemar(2, 9).p == pytest.approx(stats.binomtest(2, 11).pvalue)
    x = np.array([1.0, 1, 2, 2, 1, 2, 2, 2, 1, 1])
    seqs, obs = [], 5  # runs: 11 | 22 | 1 | 222 | 11; E(R) = 6 for 5 + 5
    for pos in itertools.combinations(range(10), 5):
        lab = np.zeros(10, int)
        lab[list(pos)] = 1
        seqs.append(1 + int(np.count_nonzero(lab[1:] != lab[:-1])))
    seqs = np.array(seqs)
    ref = np.mean(np.abs(seqs - 6) >= abs(obs - 6))
    assert T.exact_runs(x, 1.5).p == pytest.approx(ref)


def test_exact_falls_back_to_monte_carlo_when_large():
    rng = np.random.default_rng(5)
    s = [rng.normal(size=15) for _ in range(3)]
    p = T.exact_kruskal(s, samples_mc=2000)
    assert p.method == "monte carlo" and p.samples == 2000
    assert p.lower <= stats.kruskal(*s).pvalue + 0.05


# ------------------------------------------------------------------ asymptotic statistics
@pytest.fixture
def frame():
    rng = np.random.default_rng(9)
    n = 45
    return pd.DataFrame({
        "y": rng.normal(10, 2, n).round(1), "g3": np.repeat([1.0, 2.0, 3.0], 15),
        "g2": np.tile([1.0, 2.0, 1.0], 15), "t1": rng.integers(1, 6, n).astype(float),
        "t2": rng.integers(1, 6, n).astype(float), "t3": rng.integers(2, 7, n).astype(float),
        "b1": rng.integers(0, 2, n).astype(float), "b2": rng.integers(0, 2, n).astype(float),
        "b3": (rng.random(n) < .7).astype(float), "sex": rng.choice(["L", "P"], n)})


def test_friedman_kendall_cochran_match_pingouin(frame):
    x = frame[["t1", "t2", "t3"]].to_numpy()
    r = T.friedman_test(x)
    ref = stats.friedmanchisquare(*x.T)
    assert r["chi2"] == pytest.approx(ref.statistic)
    long = frame[["t1", "t2", "t3"]].reset_index().melt(id_vars="index")
    w = pg.friedman(long, dv="value", within="variable", subject="index")
    assert r["w"] == pytest.approx(w["W"].iloc[0], rel=1e-6)
    b = frame[["b1", "b2", "b3"]]
    q = T.cochran_test(b.to_numpy())
    longb = b.reset_index().melt(id_vars="index")
    ref = pg.cochran(longb, dv="value", within="variable", subject="index")
    assert q["q"] == pytest.approx(ref["Q"].iloc[0])
    assert q["p"] == pytest.approx(ref["p_unc"].iloc[0] if "p_unc" in ref else
                                   ref["p-unc"].iloc[0])


def test_dunn_pairwise_against_formula(frame):
    samples = [frame.y[frame.g3 == k].to_numpy() for k in (1, 2, 3)]
    rows = T.dunn_pairwise(samples, ["A", "B", "C"])
    allx = frame.y.to_numpy()
    n = len(allx)
    ranks = stats.rankdata(allx)
    mr = [ranks[frame.g3.to_numpy() == k].mean() for k in (1, 2, 3)]
    _, t = np.unique(allx, return_counts=True)
    tie = (t ** 3 - t).sum()
    for r, (i, j) in zip(rows, itertools.combinations(range(3), 2)):
        se = math.sqrt((n * (n + 1) / 12 - tie / (12 * (n - 1))) * (1 / 15 + 1 / 15))
        z = (mr[i] - mr[j]) / se
        assert r["z"] == pytest.approx(z)
        assert r["adj"] == pytest.approx(min(1, 3 * 2 * stats.norm.sf(abs(z))))


def test_kruskal_tie_corrected_equals_scipy(frame):
    samples = [frame.y[frame.g3 == k] for k in (1, 2, 3)]
    assert T.kruskal_test(samples)["h"] == pytest.approx(stats.kruskal(*samples).statistic)


def test_hodges_lehmann_estimates():
    rng = np.random.default_rng(6)
    a, b = rng.normal(1, 1, 20), rng.normal(0, 1, 18)
    r = T.hodges_lehmann_two(a, b)
    assert r["estimate"] == pytest.approx(np.median(np.subtract.outer(a, b)))
    assert r["lower"] < r["estimate"] < r["upper"]
    # the CI end points are where the two-sided M-W z (no continuity) crosses 1.96
    for end, inside in ((r["lower"] - 1e-6, False), (r["lower"] + 1e-6, True)):
        p = stats.mannwhitneyu(a - end, b, method="asymptotic", use_continuity=False).pvalue
        assert (p > 0.04) == inside or abs(p - 0.05) < 0.01
    d = rng.normal(.5, 1, 15)
    r1 = T.hodges_lehmann_one(d)
    i, j = np.triu_indices(15)
    assert r1["estimate"] == pytest.approx(np.median((d[i] + d[j]) / 2))


# ------------------------------------------------------------------ NPTESTS procedures
def _ds(frame):
    ds = Dataset(frame)
    for v in ("g3", "g2", "b1", "b2", "b3"):
        ds.var(v).measure = "nominal"
    return ds


def test_one_sample_automatic_choice(frame):
    ds = _ds(frame)
    sec = T.one_sample(ds, ["y", "sex", "g3"], charts_on=False)
    hyp = table(sec, "Hypothesis Test Summary")
    tests = [r[2] for r in hyp.rows]
    assert tests == ["One-Sample Kolmogorov-Smirnov Test", "One-Sample Binomial Test",
                     "One-Sample Chi-Square Test"]
    first = frame.sex.iloc[0]
    k = int((frame.sex == first).sum())
    assert num(hyp.rows[1][3]) == pytest.approx(stats.binomtest(k, len(frame)).pvalue, abs=5e-4)
    assert any("Exact significance" in f for f in hyp.footnotes)
    assert hyp.rows[0][4] in ("Retain the null hypothesis.", "Reject the null hypothesis.")


def test_one_sample_custom_tests(frame):
    ds = _ds(frame)
    sec = T.one_sample(ds, ["y"], ("wilcoxon", "runs", "ks"), wilcoxon_median=10,
                       ks_dists=("normal", "uniform"), charts_on=False)
    hyp = table(sec, "Hypothesis Test Summary")
    assert len(hyp.rows) == 4
    ref = stats.wilcoxon(frame.y - 10, correction=False, method="approx")
    row = next(r for r in hyp.rows if "Wilcoxon" in r[2])
    assert num(row[3]) == pytest.approx(ref.pvalue, abs=5e-4)
    assert row[1] == "The median of y equals 10.00."


def test_independent_samples(frame):
    ds = _ds(frame)
    sec = T.independent_samples(ds, ["y"], "g3", charts_on=False)
    hyp = table(sec, "Hypothesis Test Summary")
    assert hyp.rows[0][2] == "Independent-Samples Kruskal-Wallis Test"
    samples = [frame.y[frame.g3 == k] for k in (1, 2, 3)]
    assert num(hyp.rows[0][3]) == pytest.approx(stats.kruskal(*samples).pvalue, abs=5e-4)
    pw = table(sec, "Pairwise Comparisons of g3")
    assert len(pw.rows) == 3 and "Bonferroni" in pw.footnotes[-1]
    sec = T.independent_samples(ds, ["y"], "g2", T.INDEPENDENT_TESTS, charts_on=False)
    hyp = table(sec, "Hypothesis Test Summary")
    assert len(hyp.rows) == 7  # HL gives a CI table instead of a hypothesis
    assert table(sec, "Confidence Interval Summary")
    a, b = frame.y[frame.g2 == 1], frame.y[frame.g2 == 2]
    mw = next(r for r in hyp.rows if "Mann-Whitney" in r[2])
    assert num(mw[3]) == pytest.approx(stats.mannwhitneyu(
        a, b, method="asymptotic", use_continuity=False).pvalue, abs=5e-4)


def test_related_samples_automatic(frame):
    ds = _ds(frame)
    sec = T.related_samples(ds, ["b1", "b2", "b3"], charts_on=False)
    hyp = table(sec, "Hypothesis Test Summary")
    assert hyp.rows[0][2] == "Related-Samples Cochran's Q Test"
    assert table(sec, "Pairwise Comparisons")
    sec = T.related_samples(ds, ["t1", "t2"], charts_on=False)
    assert table(sec, "Hypothesis Test Summary").rows[0][2] == \
        "Related-Samples Wilcoxon Signed Rank Test"
    sec = T.related_samples(ds, ["b1", "b2"], charts_on=False)
    assert table(sec, "Hypothesis Test Summary").rows[0][2] == \
        "Related-Samples McNemar Change Test"
    sec = T.related_samples(ds, ["t1", "t2", "t3"], T.RELATED_TESTS, charts_on=False)
    tests = [r[2] for r in table(sec, "Hypothesis Test Summary").rows]
    assert "Related-Samples Friedman's Two-Way Analysis of Variance by Ranks" in tests


def test_charts_are_rendered(frame):
    ds = _ds(frame)
    sec = T.independent_samples(ds, ["y"], "g3")
    assert any(getattr(i, "kind", "") == "image" for i in sec.items)


# ------------------------------------------------------------------ dialogs
def _take(dlg, key, *names):
    target = next(t for t in dlg._targets if t.key == key)
    for name in names:
        dlg._take(name, target)


def test_nptests_dialogs(qapp, frame):
    from pathlib import Path

    from rsstatchem.ui.dialogs import nptests as ui

    Path(SHOTS).mkdir(parents=True, exist_ok=True)
    ds = _ds(frame)
    dlg = ui.OneSampleDialog(ds)
    _take(dlg, "fields", "y", "sex")
    assert dlg.validate() is None
    assert table(dlg.compute(), "Hypothesis Test Summary")
    for i, name in enumerate(("objective", "fields", "settings")):
        dlg.tabs.setCurrentIndex(i)
        dlg.grab().save(f"{SHOTS}/nptests_one_sample_{name}.png")
    dlg.objective_buttons["runs"].setChecked(True)
    dlg.collect()
    assert dlg.opts["mode"] == "custom" and dlg.opts["tests"] == ["runs"]

    dlg = ui.IndependentSamplesDialog(ds)
    _take(dlg, "fields", "y")
    _take(dlg, "group", "g3")
    sec = dlg.compute()
    assert table(sec, "Pairwise Comparisons")
    dlg.tabs.setCurrentIndex(2)
    dlg.grab().save(f"{SHOTS}/nptests_independent_settings.png")

    dlg = ui.RelatedSamplesDialog(ds)
    _take(dlg, "fields", "t1", "t2", "t3")
    dlg.custom_radio.setChecked(True)
    for k, cb in dlg.test_boxes.items():
        cb.setChecked(k in ("friedman", "kendall"))
    assert dlg.validate() is None
    hyp = table(dlg.compute(), "Hypothesis Test Summary")
    assert len(hyp.rows) == 2
    dlg.tabs.setCurrentIndex(1)
    dlg.grab().save(f"{SHOTS}/nptests_related_fields.png")


def test_menus_and_selftest():
    from rsstatchem.ui.dialogs import nptests as ui

    assert [e[0] for e in ui.MENUS["Analyze/Nonparametric Tests"]] == [
        "One Sample...", "Independent Samples...", "Related Samples..."]
    ui.selftest()

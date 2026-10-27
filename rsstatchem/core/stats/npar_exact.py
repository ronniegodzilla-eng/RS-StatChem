"""SPSS *Exact Tests* rows for the legacy NPAR TESTS output (Exact… button).

With **Exact**, SPSS adds "Exact Sig. (2-tailed)", "Exact Sig. (1-tailed)" and "Point
Probability" rows to the Test Statistics tables; with **Monte Carlo** it adds "Monte Carlo Sig.
(2-tailed)" with its confidence interval (default 99%, 10000 samples, seed 2000000). The
p-values come from the permutation / Monte Carlo functions of
:mod:`rsstatchem.core.stats.nptests`; exact enumeration falls back to Monte Carlo when the
reference set is larger than the time limit allows (noted under the table).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats import nonparametric as npt
from rsstatchem.core.stats import nptests as T
from rsstatchem.core.stats.common import fmt_p, numeric, sup
from rsstatchem.core.stats.resample import ExactSpec, next_letter

_EPS = 1e-7


# ============================================================================ distributions
def _dist_tails(support, counts, obs: float, center: float) -> tuple[float, float, float]:
    """(two-sided, one-sided, point probability) from a discrete null distribution."""
    total = counts.sum()
    two = counts[np.abs(support - center) >= abs(obs - center) - _EPS].sum() / total
    lower = counts[support <= obs + _EPS].sum() / total
    upper = counts[support >= obs - _EPS].sum() / total
    point = counts[np.abs(support - obs) < _EPS].sum() / total
    return float(min(1.0, two)), float(min(lower, upper)), float(point)


def mann_whitney_exact(a, b, limit: int = 2 * 10 ** 8):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n = len(a), len(a) + len(b)
    ranks = stats.rankdata(np.concatenate([a, b]))
    doubled = np.round(2 * ranks).astype(int)
    if n * (n1 + 1) * doubled.sum() > limit:
        return None
    counts = T._sum_distribution(doubled, n1)
    return _dist_tails(np.arange(len(counts), dtype=float), counts,
                       float(doubled[:n1].sum()), n1 * (n + 1.0))


def wilcoxon_exact(d, limit: int = 10 ** 9):
    d = np.asarray(d, float)
    nz = d[d != 0]
    if not len(nz):
        return 1.0, 1.0, 1.0
    doubled = np.round(2 * stats.rankdata(np.abs(nz))).astype(int)
    if len(nz) * doubled.sum() > limit:
        return None
    counts = T._sum_distribution(doubled, None)
    return _dist_tails(np.arange(len(counts), dtype=float), counts,
                       float(doubled[nz > 0].sum()), doubled.sum() / 2.0)


def sign_exact(d):
    d = np.asarray(d, float)
    neg, pos = int((d < 0).sum()), int((d > 0).sum())
    n = neg + pos
    if n == 0:
        return 1.0, 1.0, 1.0
    k = np.arange(n + 1)
    pmf = stats.binom.pmf(k, n, 0.5)
    return _dist_tails(k.astype(float), pmf, float(pos), n / 2)


def runs_exact(x, value: float):
    x = np.asarray(x, float)
    above = x >= value
    n1, n2 = int((~above).sum()), int(above.sum())
    if n1 == 0 or n2 == 0:
        return None
    r = int(1 + np.count_nonzero(above[1:] != above[:-1]))
    rs_, ps = T.runs_pmf(n1, n2)
    return _dist_tails(np.asarray(rs_, float), np.asarray(ps, float), float(r),
                       2.0 * n1 * n2 / (n1 + n2) + 1)


# ============================================================================ table helpers
def _table_indices(section: Section, pred) -> list[int]:
    return [i for i, it in enumerate(section.items)
            if getattr(it, "kind", "") == "table" and pred(it)]


def _has_row(table: Table, label: str) -> bool:
    return any(row[0].startswith(label) for row in table.rows)


def _foot(table: Table, text: str) -> None:
    if text not in table.footnotes:
        table.footnotes.append(text)


def _add_rows(table: Table, results: dict[int, object], spec: ExactSpec,
              one_sided: bool = True, point: bool = True) -> None:
    """``results`` maps a column index to a PValue (Monte Carlo / exact two-sided) or a
    (two, one, point) tuple (full distribution)."""
    h = table.row_header_cols
    width = len(table.columns)

    def row(label):
        return [label] + [""] * (width - 1)

    if spec.method == "mc":
        mc = {c: r for c, r in results.items() if isinstance(r, T.PValue)}
        ex = {c: r for c, r in results.items() if not isinstance(r, T.PValue)}
        if mc:
            lab = f"{spec.mc_level * 100:g}% Confidence Interval"
            letter = next_letter(table)
            r1, r2, r3 = row("Monte Carlo Sig. (2-tailed)" + sup(letter)), \
                row(f"Monte Carlo {lab}: Lower Bound"), row(f"Monte Carlo {lab}: Upper Bound")
            for c, p in mc.items():
                r1[c], r2[c], r3[c] = fmt_p(p.p), fmt_p(p.lower), fmt_p(p.upper)
            table.rows += [r1, r2, r3]
            _foot(table, f"{letter}. Based on {spec.mc_samples} sampled tables with starting "
                         f"seed {spec.seed}.")
        results = ex
        if not results:
            return
    r2 = row("Exact Sig. (2-tailed)")
    r1 = row("Exact Sig. (1-tailed)")
    rp = row("Point Probability")
    fallback = False
    letter = next_letter(table)
    for c, res in results.items():
        if isinstance(res, T.PValue):
            r2[c] = fmt_p(res.p) + (sup(letter) if res.method == "monte carlo" else "")
            fallback |= res.method == "monte carlo"
            continue
        two, one, pt = res
        r2[c], r1[c], rp[c] = fmt_p(two), fmt_p(one), fmt(pt)
    table.rows.append(r2)
    if one_sided and any(r1[h:]):
        table.rows.append(r1)
    if point and any(rp[h:]):
        table.rows.append(rp)
    if fallback:
        _foot(table, f"{letter}. The exact computation exceeds the time limit; Monte Carlo "
                     f"significance based on {spec.mc_samples} sampled tables.")


def fmt(x) -> str:
    from rsstatchem.core.stats.common import fmt as _fmt
    return _fmt(x, 3)


def _col(table: Table, title: str) -> int | None:
    for c in range(table.row_header_cols, len(table.columns)):
        if table.columns[c] == title:
            return c
    return None


# ============================================================================ procedures
def chi_square(section, ds, names, spec: ExactSpec, expected=None, expected_range=None,
               listwise=False) -> Section:
    ds = npt._view(ds, names, listwise)
    idx = _table_indices(section, lambda t: t.title == "Test Statistics"
                         and _has_row(t, "Chi-Square"))
    if not idx:
        return section
    table = section.items[idx[0]]
    res = {}
    for name in names:
        s = ds.analysis_series(name).dropna()
        if expected_range is not None:
            lo, hi = int(math.ceil(expected_range[0])), int(math.floor(expected_range[1]))
            x = np.floor(pd.to_numeric(s, errors="coerce").dropna())
            x = x[(x >= lo) & (x <= hi)]
            observed = np.array([int((x == c).sum()) for c in range(lo, hi + 1)])
        else:
            cats = npt._categories(s)
            observed = np.array([int((s == c).sum()) for c in cats])
        w = expected if expected and len(expected) == len(observed) else None
        c = _col(table, ds.var(name).title)
        if c is not None and observed.sum() > 0:
            res[c] = T.exact_chisq_gof(observed, w, **spec.kwargs())
    _add_rows(table, res, spec, one_sided=False, point=False)
    return section


def binomial(section, ds, names, spec: ExactSpec, test_prop=0.5, cut=None,
             listwise=False) -> Section:
    """The binomial test is always exact; the Exact option adds the point probability."""
    ds = npt._view(ds, names, listwise)
    idx = _table_indices(section, lambda t: t.title == "Binomial Test")
    if not idx:
        return section
    table = section.items[idx[0]]
    table.columns = list(table.columns) + ["Point Probability"]
    for row in table.rows:
        row.append("")
    for r, row in enumerate(table.rows):
        if row[1] != "Group 1":
            continue
        k = int(float(row[3]))
        n = int(float(table.rows[r + 2][3]))
        row[-1] = fmt(stats.binom.pmf(k, n, test_prop))
    return section


def runs(section, ds, names, cutoff, value, spec: ExactSpec, listwise=False) -> Section:
    ds = npt._view(ds, names, listwise)
    for t_i in _table_indices(section, lambda t: t.title == "Runs Test"):
        table = section.items[t_i]
        res = {}
        for name in names:
            x = numeric(ds, name).dropna().to_numpy()
            if not len(x):
                continue
            c = _col(table, ds.var(name).title)
            out = runs_exact(x, npt.cut_value(x, cutoff, value)[0])
            if c is not None and out is not None:
                res[c] = out
        _add_rows(table, res, spec)
    return section


def two_independent(section, ds, names, group, groups, spec: ExactSpec, mw=True, ks=False,
                    listwise=False) -> Section:
    ds = npt._view(ds, names + [group], listwise)
    kw = spec.kwargs()
    if mw:
        idx = _table_indices(section, lambda t: _has_row(t, "Mann-Whitney U"))
        if idx:
            table = section.items[idx[0]]
            res = {}
            for name in names:
                a, b = npt._two_groups(ds, name, group, groups)
                c = _col(table, ds.var(name).title)
                if c is None or not len(a) or not len(b):
                    continue
                out = mann_whitney_exact(a, b, kw["limit"] * 10 ** 4) \
                    if spec.method == "exact" else None
                res[c] = out if out is not None else T.exact_mann_whitney(a, b, **kw)
            _add_rows(table, res, spec)
    if ks:
        idx = _table_indices(section, lambda t: "Two-Sample Kolmogorov-Smirnov" in t.title)
        if idx:
            table = section.items[idx[0]]
            res = {}
            for name in names:
                a, b = npt._two_groups(ds, name, group, groups)
                c = _col(table, ds.var(name).title)
                if c is not None and len(a) and len(b):
                    res[c] = T.exact_ks_2samp(a, b, **kw)
            _add_rows(table, res, spec, one_sided=False, point=False)
    return section


def k_independent(section, ds, names, group, lo, hi, spec: ExactSpec, kw_test=True,
                  median=False, jt=False, listwise=False) -> Section:
    ds = npt._view(ds, names + [group], listwise)
    kwargs = spec.kwargs("samples_mc")
    tests = [("kw", kw_test, lambda t: _has_row(t, "Kruskal-Wallis H"), T.exact_kruskal),
             ("median", median, lambda t: "(Median Test)" in t.title, T.exact_median_test),
             ("jt", jt, lambda t: t.title.startswith("Jonckheere-Terpstra"),
              T.exact_jonckheere)]
    for _, on, pred, fn in tests:
        if not on:
            continue
        idx = _table_indices(section, pred)
        if not idx:
            continue
        table = section.items[idx[0]]
        res = {}
        for name in names:
            _, samples = npt._k_groups(ds, name, group, lo, hi)
            samples = [s for s in samples if len(s)]
            c = _col(table, ds.var(name).title)
            if c is not None and len(samples) >= 2:
                res[c] = fn(samples, **kwargs)
        _add_rows(table, res, spec, one_sided=False, point=False)
    return section


def two_related(section, ds, pairs, spec: ExactSpec, wilcoxon=True, sign=False,
                listwise=False) -> Section:
    names = [n for p in pairs for n in p]
    ds = npt._view(ds, names, listwise)
    kw = spec.kwargs()
    diffs = {}
    for an, bn in pairs:
        frame = pd.DataFrame({"a": numeric(ds, an), "b": numeric(ds, bn)}).dropna()
        diffs[f"{ds.var(bn).title} - {ds.var(an).title}"] = \
            (frame["b"] - frame["a"]).to_numpy(float)
    if wilcoxon:
        idx = _table_indices(section, lambda t: t.title.startswith("Test Statistics")
                             and any("Wilcoxon" in f for f in t.footnotes))
        if idx:
            table = section.items[idx[0]]
            res = {}
            for title, d in diffs.items():
                c = _col(table, title)
                if c is None:
                    continue
                out = wilcoxon_exact(d, kw["limit"] * 10 ** 4) if spec.method == "exact" \
                    else None
                res[c] = out if out is not None else T.exact_wilcoxon(d, **kw)
            _add_rows(table, res, spec)
    if sign:
        idx = _table_indices(section, lambda t: "(Sign Test)" in t.title)
        if idx:
            table = section.items[idx[0]]
            res = {}
            for title, d in diffs.items():
                c = _col(table, title)
                if c is not None:
                    res[c] = sign_exact(d)
            table.rows = [r for r in table.rows if r[0] != "Exact Sig. (2-tailed)"]
            _add_rows(table, res, spec)
    return section


def k_related(section, ds, names, spec: ExactSpec, friedman=True, kendall=False,
              cochran=False, listwise=False) -> Section:
    frame = pd.concat({n: numeric(ds, n) for n in names}, axis=1).dropna()
    x = frame.to_numpy(float)
    if len(x) == 0:
        return section
    kw = spec.kwargs()
    plan = [(friedman, lambda t: any("Friedman" in f for f in t.footnotes), T.exact_friedman,
             x),
            (kendall, lambda t: "(Kendall's W)" in t.title, T.exact_kendall_w, x)]
    if cochran:
        values = np.unique(x)
        if len(values) <= 2:
            plan.append((True, lambda t: "(Cochran)" in t.title, T.exact_cochran,
                         (x == values.max()).astype(float)))
    for on, pred, fn, data in plan:
        if not on:
            continue
        idx = _table_indices(section, pred)
        if not idx:
            continue
        table = section.items[idx[0]]
        res = {1: fn(data, **kw)}
        _add_rows(table, res, spec, one_sided=False, point=False)
    return section

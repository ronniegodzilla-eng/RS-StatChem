"""Analyze > Compare Means (and Proportions) > One-Sample / Independent-Samples /
Paired-Samples Proportions, mirroring SPSS 27.0.1+ (PROPORTIONS command).

Confidence intervals: Agresti-Coull, Anscombe, Clopper-Pearson (Exact), Jeffreys, Logit,
Wald, Wald (Continuity Corrected), Wilson Score, Wilson Score (Continuity Corrected) for one
proportion; Agresti-Caffo, Brown-Li's Jeffreys, Hauck-Anderson, Newcombe (with/without CC),
Wald (with/without CC) for independent differences; Agresti-Min, Bonett-Price, Newcombe,
Wald for paired differences. Tests: exact and mid-p binomial, score and Wald z tests (with
continuity corrections), Hauck-Anderson, pooled Wald H0, exact / mid-p / asymptotic McNemar.

The interval and test functions are pure functions of the counts.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from rsstatchem.core.dataset import Dataset, format_code
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, fmt_p, pct, value_text

ONE_CI = ("agresti_coull", "anscombe", "clopper_pearson", "jeffreys", "logit", "wald",
          "wald_cc", "wilson", "wilson_cc")
ONE_CI_NAMES = {"agresti_coull": "Agresti-Coull", "anscombe": "Anscombe",
                "clopper_pearson": "Clopper-Pearson (\"Exact\")", "jeffreys": "Jeffreys",
                "logit": "Logit", "wald": "Wald", "wald_cc": "Wald (Continuity Corrected)",
                "wilson": "Wilson Score", "wilson_cc": "Wilson Score (Continuity Corrected)"}
ONE_TEST = ("exact", "midp", "score", "score_cc", "wald", "wald_cc")
ONE_TEST_NAMES = {"exact": "Exact Binomial", "midp": "Mid-p Adjusted Binomial",
                  "score": "Score", "score_cc": "Score (Continuity Corrected)", "wald": "Wald",
                  "wald_cc": "Wald (Continuity Corrected)"}
IND_CI = ("agresti_caffo", "brown_li", "hauck_anderson", "newcombe", "newcombe_cc", "wald",
          "wald_cc")
IND_CI_NAMES = {"agresti_caffo": "Agresti-Caffo", "brown_li": "Brown-Li's Jeffreys",
                "hauck_anderson": "Hauck-Anderson", "newcombe": "Newcombe",
                "newcombe_cc": "Newcombe (Continuity Corrected)", "wald": "Wald",
                "wald_cc": "Wald (Continuity Corrected)"}
IND_TEST = ("hauck_anderson", "wald", "wald_cc", "wald_h0", "wald_h0_cc")
IND_TEST_NAMES = {"hauck_anderson": "Hauck-Anderson", "wald": "Wald",
                  "wald_cc": "Wald (Continuity Corrected)", "wald_h0": "Wald H0",
                  "wald_h0_cc": "Wald H0 (Continuity Corrected)"}
PAIR_CI = ("agresti_min", "bonett_price", "newcombe", "newcombe_cc", "wald", "wald_cc")
PAIR_CI_NAMES = {"agresti_min": "Agresti-Min", "bonett_price": "Bonett-Price",
                 "newcombe": "Newcombe", "newcombe_cc": "Newcombe (Continuity Corrected)",
                 "wald": "Wald", "wald_cc": "Wald (Continuity Corrected)"}
PAIR_TEST = ("exact", "midp", "mcnemar", "mcnemar_cc", "wald", "wald_cc")
PAIR_TEST_NAMES = {"exact": "Exact McNemar", "midp": "Mid-p Adjusted Binomial",
                   "mcnemar": "McNemar", "mcnemar_cc": "McNemar (Continuity Corrected)",
                   "wald": "Wald", "wald_cc": "Wald (Continuity Corrected)"}


def _clip(lo: float, hi: float, a: float = 0.0, b: float = 1.0) -> tuple[float, float]:
    return max(a, lo), min(b, hi)


# ------------------------------------------------------------------ one proportion
def one_ci(x: int, n: int, method: str, level: float = 0.95) -> tuple[float, float]:
    z = stats.norm.ppf(0.5 + level / 2)
    a = 1 - level
    p = x / n
    q = 1 - p
    if method == "wald":
        h = z * math.sqrt(p * q / n)
        return _clip(p - h, p + h)
    if method == "wald_cc":
        h = z * math.sqrt(p * q / n) + 1 / (2 * n)
        return _clip(p - h, p + h)
    if method == "wilson":
        c = (p + z * z / (2 * n)) / (1 + z * z / n)
        h = z * math.sqrt(p * q / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return _clip(c - h, c + h)
    if method == "wilson_cc":
        den = 2 * (n + z * z)
        lo = 0.0 if x == 0 else (2 * x + z * z - 1 - z * math.sqrt(
            max(z * z - 2 - 1 / n + 4 * p * (n * q + 1), 0.0))) / den
        hi = 1.0 if x == n else (2 * x + z * z + 1 + z * math.sqrt(
            max(z * z + 2 - 1 / n + 4 * p * (n * q - 1), 0.0))) / den
        return _clip(lo, hi)
    if method == "agresti_coull":
        nt = n + z * z
        pt = (x + z * z / 2) / nt
        h = z * math.sqrt(pt * (1 - pt) / nt)
        return _clip(pt - h, pt + h)
    if method == "clopper_pearson":
        lo = 0.0 if x == 0 else float(stats.beta.ppf(a / 2, x, n - x + 1))
        hi = 1.0 if x == n else float(stats.beta.ppf(1 - a / 2, x + 1, n - x))
        return lo, hi
    if method == "jeffreys":
        lo = 0.0 if x == 0 else float(stats.beta.ppf(a / 2, x + 0.5, n - x + 0.5))
        hi = 1.0 if x == n else float(stats.beta.ppf(1 - a / 2, x + 0.5, n - x + 0.5))
        return lo, hi
    if method == "logit":
        if x == 0 or x == n:
            return float("nan"), float("nan")
        lam = math.log(x / (n - x))
        se = math.sqrt(n / (x * (n - x)))
        return 1 / (1 + math.exp(-(lam - z * se))), 1 / (1 + math.exp(-(lam + z * se)))
    if method == "anscombe":
        lam = math.log((x + 0.5) / (n - x + 0.5))
        se = math.sqrt((n + 1) * (n + 2) / (n * (x + 1) * (n - x + 1)))
        return 1 / (1 + math.exp(-(lam - z * se))), 1 / (1 + math.exp(-(lam + z * se)))
    raise ValueError(method)


def one_test(x: int, n: int, p0: float, method: str) -> tuple[float, float, float]:
    """(Z or NaN, one-sided p, two-sided p) for H0: pi = p0."""
    p = x / n
    q = 1 - p
    if method in ("exact", "midp"):
        upper = x > n * p0
        if method == "exact":
            one = stats.binom.sf(x - 1, n, p0) if upper else stats.binom.cdf(x, n, p0)
        else:
            pmf = stats.binom.pmf(x, n, p0)
            one = (stats.binom.sf(x, n, p0) if upper else stats.binom.cdf(x - 1, n, p0)) + \
                0.5 * pmf
        one = float(min(one, 1.0))
        return float("nan"), one, min(1.0, 2 * one)
    if method in ("score", "score_cc"):
        se = math.sqrt(p0 * (1 - p0) / n)
        d = p - p0
        if method == "score_cc":
            d = math.copysign(max(abs(d) - 1 / (2 * n), 0.0), d)
    else:
        se = math.sqrt(p * q / n)
        d = p - p0
        if method == "wald_cc":
            d = math.copysign(max(abs(d) - 1 / (2 * n), 0.0), d)
    z = d / se if se > 0 else float("nan")
    one = float(stats.norm.sf(abs(z)))
    return z, one, 2 * one


# ------------------------------------------------------------------ two independent
def ind_ci(x1, n1, x2, n2, method: str, level: float = 0.95) -> tuple[float, float]:
    z = stats.norm.ppf(0.5 + level / 2)
    p1, p2 = x1 / n1, x2 / n2
    d = p1 - p2
    if method in ("wald", "wald_cc"):
        h = z * math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        if method == "wald_cc":
            h += (1 / n1 + 1 / n2) / 2
        return _clip(d - h, d + h, -1, 1)
    if method == "agresti_caffo":
        a1, a2 = (x1 + 1) / (n1 + 2), (x2 + 1) / (n2 + 2)
        h = z * math.sqrt(a1 * (1 - a1) / (n1 + 2) + a2 * (1 - a2) / (n2 + 2))
        return _clip(a1 - a2 - h, a1 - a2 + h, -1, 1)
    if method == "brown_li":
        a1, a2 = (x1 + 0.5) / (n1 + 1), (x2 + 0.5) / (n2 + 1)
        h = z * math.sqrt(a1 * (1 - a1) / n1 + a2 * (1 - a2) / n2)
        return _clip(a1 - a2 - h, a1 - a2 + h, -1, 1)
    if method == "hauck_anderson":
        h = 1 / (2 * min(n1, n2)) + z * math.sqrt(p1 * (1 - p1) / (n1 - 1) +
                                                   p2 * (1 - p2) / (n2 - 1))
        return _clip(d - h, d + h, -1, 1)
    if method in ("newcombe", "newcombe_cc"):
        m = "wilson" if method == "newcombe" else "wilson_cc"
        l1, u1 = one_ci(x1, n1, m, level)
        l2, u2 = one_ci(x2, n2, m, level)
        return (d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
                d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))
    raise ValueError(method)


def ind_test(x1, n1, x2, n2, method: str) -> tuple[float, float, float]:
    p1, p2 = x1 / n1, x2 / n2
    d = p1 - p2
    cc = 0.0
    if method == "hauck_anderson":
        se = math.sqrt(p1 * (1 - p1) / (n1 - 1) + p2 * (1 - p2) / (n2 - 1))
        cc = 1 / (2 * min(n1, n2))
    elif method in ("wald", "wald_cc"):
        se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        cc = (1 / n1 + 1 / n2) / 2 if method == "wald_cc" else 0.0
    else:
        pb = (x1 + x2) / (n1 + n2)
        se = math.sqrt(pb * (1 - pb) * (1 / n1 + 1 / n2))
        cc = (1 / n1 + 1 / n2) / 2 if method == "wald_h0_cc" else 0.0
    dd = math.copysign(max(abs(d) - cc, 0.0), d)
    z = dd / se if se > 0 else float("nan")
    one = float(stats.norm.sf(abs(z)))
    return z, one, 2 * one


# ------------------------------------------------------------------ paired
def pair_ci(a, b, c, d, method: str, level: float = 0.95) -> tuple[float, float]:
    """a = both successes, b = success/failure, c = failure/success, d = both failures."""
    z = stats.norm.ppf(0.5 + level / 2)
    n = a + b + c + d
    diff = (b - c) / n
    if method in ("wald", "wald_cc"):
        h = z * math.sqrt(max((b + c) - (b - c) ** 2 / n, 0.0)) / n
        if method == "wald_cc":
            h += 1 / n
        return _clip(diff - h, diff + h, -1, 1)
    if method == "agresti_min":
        bt, ct, nt = b + 0.5, c + 0.5, n + 2
        dt = (bt - ct) / nt
        h = z * math.sqrt(max((bt + ct) - (bt - ct) ** 2 / nt, 0.0)) / nt
        return _clip(dt - h, dt + h, -1, 1)
    if method == "bonett_price":
        p12, p21 = (b + 1) / (n + 2), (c + 1) / (n + 2)
        dt = p12 - p21
        h = z * math.sqrt(max((p12 + p21 - dt * dt) / (n + 2), 0.0))
        return _clip(dt - h, dt + h, -1, 1)
    if method in ("newcombe", "newcombe_cc"):
        m = "wilson" if method == "newcombe" else "wilson_cc"
        x1, x2 = a + b, a + c
        p1, p2 = x1 / n, x2 / n
        l1, u1 = one_ci(x1, n, m, level)
        l2, u2 = one_ci(x2, n, m, level)
        den = (a + b) * (c + d) * (a + c) * (b + d)
        phi = (a * d - b * c) / math.sqrt(den) if den > 0 else 0.0
        lo = diff - math.sqrt(max((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) +
                                  (u2 - p2) ** 2, 0.0))
        hi = diff + math.sqrt(max((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) +
                                  (p2 - l2) ** 2, 0.0))
        return _clip(lo, hi, -1, 1)
    raise ValueError(method)


def pair_test(a, b, c, d, method: str) -> tuple[float, float, float]:
    n = a + b + c + d
    m = b + c
    if method in ("exact", "midp"):
        if m == 0:
            return float("nan"), 1.0, 1.0
        k = min(b, c)
        if method == "exact":
            one = float(stats.binom.cdf(k, m, 0.5))
        else:
            one = float(stats.binom.cdf(k - 1, m, 0.5) + 0.5 * stats.binom.pmf(k, m, 0.5))
        return float("nan"), min(one, 1.0), min(1.0, 2 * one)
    if method in ("mcnemar", "mcnemar_cc"):
        num = b - c
        if method == "mcnemar_cc":
            num = math.copysign(max(abs(num) - 1, 0.0), num)
        z = num / math.sqrt(m) if m > 0 else float("nan")
    else:
        se = math.sqrt(max((b + c) - (b - c) ** 2 / n, 0.0)) / n
        diff = (b - c) / n
        if method == "wald_cc":
            diff = math.copysign(max(abs(diff) - 1 / n, 0.0), diff)
        z = diff / se if se > 0 else float("nan")
    one = float(stats.norm.sf(abs(z)))
    return z, one, 2 * one


# ------------------------------------------------------------------ success definition
def success_mask(values: pd.Series, rule: str = "last", spec=None) -> pd.Series:
    """True for successes among the non-missing values. ``rule``: 'last' | 'first' |
    'values' (spec = list of codes) | 'midpoint' | 'cutpoint' (spec = cut, success >= cut)."""
    v = values.dropna()
    if rule in ("last", "first"):
        levels = sorted(v.unique(), key=lambda x: (isinstance(x, str), x))
        target = levels[-1] if rule == "last" else levels[0]
        return v == target
    if rule == "values":
        return v.isin(list(spec or []))
    num = pd.to_numeric(v, errors="coerce")
    if rule == "midpoint":
        return num >= (num.min() + num.max()) / 2
    return num >= float(spec)


def _success_text(var, values: pd.Series, rule: str, spec) -> str:
    v = values.dropna()
    if rule in ("last", "first"):
        levels = sorted(v.unique(), key=lambda x: (isinstance(x, str), x))
        if not levels:
            return ""
        target = levels[-1] if rule == "last" else levels[0]
        return f"{value_text(var, target)}"
    if rule == "values":
        return ", ".join(format_code(s) for s in spec or [])
    if rule == "midpoint":
        num = pd.to_numeric(v, errors="coerce")
        return f">= {fmt((num.min() + num.max()) / 2)}"
    return f">= {spec:g}"


# ------------------------------------------------------------------ procedures
def _ci_cols(level: float) -> list:
    return [(f"{pct(level)} Confidence Interval", "Lower"),
            (f"{pct(level)} Confidence Interval", "Upper")]


def one_sample(ds: Dataset, names: list[str], test_value: float = 0.5,
               intervals=("clopper_pearson", "wilson"), tests=("exact", "score"),
               rule: str = "last", spec=None, level: float = 0.95,
               listwise: bool = False) -> Section:
    section = Section("Proportions")
    section.log(f"PROPORTIONS\n  /ONESAMPLE {' '.join(names)} TESTVAL={test_value:g} "
                f"TESTTYPES={' '.join(t.upper() for t in tests)} "
                f"CITYPES={' '.join(c.upper() for c in intervals)}"
                f"\n  /SUCCESS VALUE={rule.upper()}\n  /CRITERIA CILEVEL={level * 100:g}"
                f"\n  /MISSING SCOPE={'LISTWISE' if listwise else 'ANALYSIS'}.")
    keep = None
    if listwise:
        keep = pd.DataFrame({n: ds.analysis_series(n) for n in names}).notna().all(axis=1)
    crows, trows = [], []
    for name in names:
        var = ds.var(name)
        s = ds.analysis_series(name)
        if keep is not None:
            s = s[keep]
        mask = success_mask(s, rule, spec)
        n, x = int(len(mask)), int(mask.sum())
        if n == 0:
            continue
        p = x / n
        ase = math.sqrt(p * (1 - p) / n)
        label = f"{var.title} = {_success_text(var, s, rule, spec)}"
        for i, m in enumerate([m for m in ONE_CI if m in intervals]):
            lo, hi = one_ci(x, n, m, level)
            crows.append([label if i == 0 else "", ONE_CI_NAMES[m], fmt_int(x), fmt_int(n),
                          fmt(p), fmt(ase), fmt(lo), fmt(hi)])
        for i, m in enumerate([m for m in ONE_TEST if m in tests]):
            z, one, two = one_test(x, n, test_value, m)
            se0 = math.sqrt(test_value * (1 - test_value) / n) if m.startswith("score") \
                else ase
            trows.append([label if i == 0 else "", ONE_TEST_NAMES[m], fmt_int(x), fmt_int(n),
                          fmt(p), fmt(p - test_value), fmt(se0) if z == z else "",
                          fmt(z) if z == z else "", fmt_p(one), fmt_p(two)])
    if crows:
        section.table(Table("One-Sample Proportions Confidence Intervals",
                            ["", "Interval Type", "Successes", "Trials", "Observed Proportion",
                             "Asymptotic Standard Error"] + _ci_cols(level), crows,
                            row_header_cols=2))
    if trows:
        section.table(Table("One-Sample Proportions Tests",
                            ["", "Test Type", "Successes", "Trials", "Observed Proportion",
                             "Observed - Test Value" + "ᵃ", "Asymptotic Standard Error",
                             "Z", ("Significance", "One-Sided p"),
                             ("Significance", "Two-Sided p")], trows, row_header_cols=2,
                            footnotes=[f"a. Test Value = {test_value:g}"]))
    return section


def independent(ds: Dataset, names: list[str], group_var: str, groups=None, cut=None,
                intervals=("newcombe", "wald"), tests=("wald_h0",), rule: str = "last",
                spec=None, level: float = 0.95, listwise: bool = False) -> Section:
    section = Section("Proportions")
    gdesc = f"({cut:g})" if cut is not None else \
        f"({' '.join(format_code(g) for g in groups)})"
    section.log(f"PROPORTIONS\n  /INDEPENDENTSAMPLES {' '.join(names)} BY {group_var}{gdesc} "
                f"TESTTYPES={' '.join(t.upper() for t in tests)} "
                f"CITYPES={' '.join(c.upper() for c in intervals)}"
                f"\n  /SUCCESS VALUE={rule.upper()}\n  /CRITERIA CILEVEL={level * 100:g}.")
    grows, crows, trows = [], [], []
    gtitle = ds.var(group_var).title
    for name in names:
        var = ds.var(name)
        raw = ds.analysis_series(name)
        succ = success_mask(raw, rule, spec)
        g = ds.analysis_series(group_var).loc[succ.index]
        gvar = ds.var(group_var)
        if cut is not None:
            gnum = pd.to_numeric(g, errors="coerce")
            a = succ[gnum >= cut].to_numpy(float)
            b = succ[gnum < cut].to_numpy(float)
            la, lb = f">= {cut:g}", f"< {cut:g}"
        else:
            a = succ[g == groups[0]].to_numpy(float)
            b = succ[g == groups[1]].to_numpy(float)
            la, lb = value_text(gvar, groups[0]), value_text(gvar, groups[1])
        n1, x1, n2, x2 = len(a), int(a.sum()), len(b), int(b.sum())
        if n1 < 2 or n2 < 2:
            section.note("Warning", f"{var.title}: each group needs at least two cases.",
                         style="warning")
            continue
        label = f"{var.title} = {_success_text(var, raw, rule, spec)}"
        for i, (lab, x, n) in enumerate(((la, x1, n1), (lb, x2, n2))):
            p = x / n
            grows.append([label if i == 0 else "", lab, fmt_int(x), fmt_int(n), fmt(p),
                          fmt(math.sqrt(p * (1 - p) / n))])
        d = x1 / n1 - x2 / n2
        ase = math.sqrt(x1 / n1 * (1 - x1 / n1) / n1 + x2 / n2 * (1 - x2 / n2) / n2)
        for i, m in enumerate([m for m in IND_CI if m in intervals]):
            lo, hi = ind_ci(x1, n1, x2, n2, m, level)
            crows.append([label if i == 0 else "", IND_CI_NAMES[m], fmt(d), fmt(ase), fmt(lo),
                          fmt(hi)])
        for i, m in enumerate([m for m in IND_TEST if m in tests]):
            z, one, two = ind_test(x1, n1, x2, n2, m)
            trows.append([label if i == 0 else "", IND_TEST_NAMES[m], fmt(d), fmt(ase), fmt(z),
                          fmt_p(one), fmt_p(two)])
    if grows:
        section.table(Table("Independent-Samples Proportions Group Statistics",
                            ["", gtitle, "Successes", "Trials", "Proportion",
                             "Asymptotic Standard Error"], grows, row_header_cols=2))
    if crows:
        section.table(Table("Independent-Samples Proportions Confidence Intervals",
                            ["", "Interval Type", "Difference in Proportions",
                             "Asymptotic Standard Error"] + _ci_cols(level), crows,
                            row_header_cols=2))
    if trows:
        section.table(Table("Independent-Samples Proportions Tests",
                            ["", "Test Type", "Difference in Proportions",
                             "Asymptotic Standard Error", "Z",
                             ("Significance", "One-Sided p"), ("Significance", "Two-Sided p")],
                            trows, row_header_cols=2))
    return section


def paired(ds: Dataset, pairs: list[tuple[str, str]], intervals=("newcombe", "wald"),
           tests=("exact", "mcnemar"), rule: str = "last", spec=None,
           level: float = 0.95) -> Section:
    section = Section("Proportions")
    section.log("PROPORTIONS\n  /RELATEDSAMPLES " + " ".join(a for a, _ in pairs) + " WITH "
                + " ".join(b for _, b in pairs) +
                f" (PAIRED) TESTTYPES={' '.join(t.upper() for t in tests)} "
                f"CITYPES={' '.join(c.upper() for c in intervals)}"
                f"\n  /SUCCESS VALUE={rule.upper()}\n  /CRITERIA CILEVEL={level * 100:g}.")
    srows, crows, trows = [], [], []
    for k, (an, bn) in enumerate(pairs, start=1):
        va, vb = ds.var(an), ds.var(bn)
        frame = pd.DataFrame({"a": ds.analysis_series(an), "b": ds.analysis_series(bn)}).dropna()
        if frame.empty:
            continue
        sa = success_mask(frame["a"], rule, spec).to_numpy()
        sb = success_mask(frame["b"], rule, spec).to_numpy()
        a = int(np.sum(sa & sb))
        b = int(np.sum(sa & ~sb))
        c = int(np.sum(~sa & sb))
        d = int(np.sum(~sa & ~sb))
        n = a + b + c + d
        label = f"Pair {k}"
        names = f"{va.title} - {vb.title}"
        for i, (v, s) in enumerate(((va, sa), (vb, sb))):
            p = s.mean()
            srows.append([label if i == 0 else "", v.title, fmt_int(s.sum()), fmt_int(n),
                          fmt(p), fmt(math.sqrt(p * (1 - p) / n))])
        diff = (b - c) / n
        ase = math.sqrt(max((b + c) - (b - c) ** 2 / n, 0.0)) / n
        for i, m in enumerate([m for m in PAIR_CI if m in intervals]):
            lo, hi = pair_ci(a, b, c, d, m, level)
            crows.append([label if i == 0 else "", names if i == 0 else "", PAIR_CI_NAMES[m],
                          fmt(diff), fmt(ase), fmt(lo), fmt(hi)])
        for i, m in enumerate([m for m in PAIR_TEST if m in tests]):
            z, one, two = pair_test(a, b, c, d, m)
            trows.append([label if i == 0 else "", names if i == 0 else "",
                          PAIR_TEST_NAMES[m], fmt(diff), fmt(ase), fmt(z) if z == z else "",
                          fmt_p(one), fmt_p(two)])
    if srows:
        section.table(Table("Paired-Samples Proportions Statistics",
                            ["", "", "Successes", "Trials", "Proportion",
                             "Asymptotic Standard Error"], srows, row_header_cols=2))
    if crows:
        section.table(Table("Paired-Samples Proportions Confidence Intervals",
                            ["", "", "Interval Type", "Difference in Proportions",
                             "Asymptotic Standard Error"] + _ci_cols(level), crows,
                            row_header_cols=3))
    if trows:
        section.table(Table("Paired-Samples Proportions Tests",
                            ["", "", "Test Type", "Difference in Proportions",
                             "Asymptotic Standard Error", "Z",
                             ("Significance", "One-Sided p"), ("Significance", "Two-Sided p")],
                            trows, row_header_cols=3))
    return section


__all__ = ["one_ci", "one_test", "ind_ci", "ind_test", "pair_ci", "pair_test", "one_sample",
           "independent", "paired", "success_mask"]

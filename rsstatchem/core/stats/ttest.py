"""Analyze > Compare Means: One-Sample, Independent-Samples and Paired-Samples T Tests.

Output follows SPSS 27+: one- and two-sided p values and an Effect Sizes table
(Cohen's d, Hedges' correction, Glass's delta) with noncentral-t confidence intervals.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from rsstatchem.core.dataset import Dataset, format_code
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import (fmt, fmt_df, fmt_int, fmt_p, numeric, pct,
                                          t_interval, value_text)


def hedges_j(df: float) -> float:
    """Exact small-sample correction factor J(df) used by SPSS for Hedges' g."""
    if df <= 1:
        return float("nan")
    return math.exp(special.gammaln(df / 2) - math.log(math.sqrt(df / 2))
                    - special.gammaln((df - 1) / 2))


def ncp_interval(t: float, df: float, level: float = 0.95) -> tuple[float, float]:
    """Confidence interval for the noncentrality parameter of a t statistic."""
    if not math.isfinite(t) or df <= 0:
        return float("nan"), float("nan")
    if t < 0:  # nct.cdf is unstable deep in the tail for negative t: solve for |t| and mirror
        lo, hi = ncp_interval(-t, df, level)
        return -hi, -lo
    alpha = 1 - level

    def solve(target):
        f = lambda ncp: stats.nct.cdf(t, df, ncp) - target  # noqa: E731
        lo, hi = t - 8 - 0.5 * t, t + 8 + 0.5 * t
        try:
            return optimize.brentq(f, lo, hi, xtol=1e-10)
        except ValueError:
            return float("nan")

    return solve(1 - alpha / 2), solve(alpha / 2)


def _effect_row(label: str, standardizer: float, t: float, df: float, factor: float,
                level: float, correction: float = 1.0) -> list[str]:
    """d = t * factor; CI from the noncentral t, scaled the same way."""
    lo, hi = ncp_interval(t, df, level)
    d = t * factor * correction
    return [label, fmt(standardizer), fmt(d), fmt(lo * factor * correction),
            fmt(hi * factor * correction)]


def _effect_columns(level: float) -> list:
    return ["", "", "Standardizerᵃ", "Point Estimate",
            (f"{pct(level)} Confidence Interval", "Lower"),
            (f"{pct(level)} Confidence Interval", "Upper")]


def _p_sides(t: float, df: float) -> tuple[float, float]:
    two = 2 * stats.t.sf(abs(t), df)
    return two / 2, two


# ------------------------------------------------------------------ one sample
def one_sample(ds: Dataset, names: list[str], test_value: float = 0.0,
               level: float = 0.95, listwise: bool = False) -> Section:
    section = Section("T-Test")
    section.log(f"T-TEST\n  /TESTVAL={test_value:g}\n  /VARIABLES={' '.join(names)}\n"
                f"  /CRITERIA=CI({level:.2f}).")
    data = {n: numeric(ds, n) for n in names}
    if listwise:
        frame = pd.DataFrame(data).dropna()
        data = {n: frame[n] for n in names}
    srows, trows, erows = [], [], []
    for name in names:
        var = ds.var(name)
        x = data[name].dropna().to_numpy(float)
        n = len(x)
        if n < 2:
            srows.append([var.title, fmt_int(n), "", "", ""])
            continue
        mean, sd = x.mean(), x.std(ddof=1)
        se = sd / math.sqrt(n)
        d = var.decimals
        srows.append([var.title, fmt_int(n), fmt(mean, d + 2), fmt(sd, d + 3), fmt(se, d + 3)])
        diff = mean - test_value
        t = diff / se if se > 0 else float("nan")
        one, two = _p_sides(t, n - 1)
        lo, hi = t_interval(diff, se, n - 1, level)
        trows.append([var.title, fmt(t), fmt_df(n - 1), fmt_p(one), fmt_p(two),
                      fmt(diff, d + 3), fmt(lo, d + 3), fmt(hi, d + 3)])
        factor = 1 / math.sqrt(n)
        erows.append([var.title] + _effect_row("Cohen's d", sd, t, n - 1, factor, level))
        j = hedges_j(n - 1)
        erows.append([""] + _effect_row("Hedges' correction", sd / j, t, n - 1, factor, level,
                                        j))
    section.table(Table("One-Sample Statistics", ["", "N", "Mean", "Std. Deviation",
                                                  "Std. Error Mean"], srows))
    group = f"Test Value = {test_value:g}"
    section.table(Table("One-Sample Test", [""] + [
        (group, "t"), (group, "df"), (group, "One-Sided p"), (group, "Two-Sided p"),
        (group, "Mean Difference"), (group, f"{pct(level)} CI Lower"),
        (group, f"{pct(level)} CI Upper")], trows))
    if erows:
        section.table(Table("One-Sample Effect Sizes", _effect_columns(level), erows,
                            row_header_cols=2, footnotes=[
                                "a. The denominator used in estimating the effect sizes. "
                                "Cohen's d uses the sample standard deviation. Hedges' "
                                "correction uses the sample standard deviation, plus a "
                                "correction factor."]))
    return section


# ------------------------------------------------------------------ independent
def split_two_groups(ds: Dataset, name: str, group_var: str, groups=None, cut=None,
                     frame: pd.DataFrame | None = None):
    g = ds.analysis_series(group_var)
    y = numeric(ds, name) if frame is None else frame[name]
    gvar = ds.var(group_var)
    if cut is not None:
        gnum = pd.to_numeric(g, errors="coerce")
        a = y[(gnum >= cut) & y.notna()].to_numpy(float)
        b = y[(gnum < cut) & y.notna()].to_numpy(float)
        return (f">= {cut:g}", a), (f"< {cut:g}", b)
    g1, g2 = groups
    a = y[(g == g1) & y.notna()].to_numpy(float)
    b = y[(g == g2) & y.notna()].to_numpy(float)
    return (value_text(gvar, g1), a), (value_text(gvar, g2), b)


def independent(ds: Dataset, names: list[str], group_var: str, groups=None, cut=None,
                level: float = 0.95, listwise: bool = False) -> Section:
    section = Section("T-Test")
    gdesc = f"({cut:g})" if cut is not None else f"({' '.join(format_code(g) for g in groups)})"
    section.log(f"T-TEST GROUPS={group_var}{gdesc}\n  /VARIABLES={' '.join(names)}\n"
                f"  /CRITERIA=CI({level:.2f}).")
    frame = None
    if listwise:
        frame = pd.DataFrame({n: numeric(ds, n) for n in names})
        frame = frame[frame.notna().all(axis=1)]
        frame = frame.reindex(ds.df.index)
    gvar = ds.var(group_var)
    grows, trows, erows = [], [], []
    for name in names:
        var = ds.var(name)
        d = var.decimals
        (la, a), (lb, b) = split_two_groups(ds, name, group_var, groups, cut, frame)
        for i, (label, x) in enumerate(((la, a), (lb, b))):
            if len(x):
                sd = x.std(ddof=1) if len(x) > 1 else float("nan")
                grows.append([var.title if i == 0 else "", label, fmt_int(len(x)),
                              fmt(x.mean(), d + 2), fmt(sd, d + 3),
                              fmt(sd / math.sqrt(len(x)), d + 3)])
            else:
                grows.append([var.title if i == 0 else "", label, "0", "", "", ""])
        n1, n2 = len(a), len(b)
        if n1 < 2 or n2 < 2:
            section.note("Warning", f"t cannot be computed for {var.title}: at least one "
                         "group has fewer than two cases.", style="warning")
            continue
        m1, m2 = a.mean(), b.mean()
        v1, v2 = a.var(ddof=1), b.var(ddof=1)
        diff = m1 - m2
        lev = stats.levene(a, b, center="mean") if (v1 > 0 or v2 > 0) else (np.nan, np.nan)
        df_pooled = n1 + n2 - 2
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / df_pooled
        se_pooled = math.sqrt(sp2 * (1 / n1 + 1 / n2))
        se_welch = math.sqrt(v1 / n1 + v2 / n2)
        df_welch = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1)
                                              + (v2 / n2) ** 2 / (n2 - 1)) \
            if se_welch > 0 else float("nan")
        for i, (se, df) in enumerate(((se_pooled, df_pooled), (se_welch, df_welch))):
            t = diff / se if se > 0 else float("nan")
            one, two = _p_sides(t, df)
            lo, hi = t_interval(diff, se, df, level)
            levene_cells = [fmt(lev[0]), fmt_p(lev[1])] if i == 0 else ["", ""]
            trows.append([var.title if i == 0 else "",
                          "Equal variances assumed" if i == 0 else "Equal variances not assumed"]
                         + levene_cells + [fmt(t), fmt_df(df), fmt_p(one), fmt_p(two),
                                           fmt(diff, d + 3), fmt(se, d + 3), fmt(lo, d + 3),
                                           fmt(hi, d + 3)])
        t_pooled = diff / se_pooled if se_pooled > 0 else float("nan")
        sp = math.sqrt(sp2)
        factor = math.sqrt(1 / n1 + 1 / n2)
        erows.append([var.title] + _effect_row("Cohen's d", sp, t_pooled, df_pooled, factor,
                                               level))
        j = hedges_j(df_pooled)
        erows.append([""] + _effect_row("Hedges' correction", sp / j, t_pooled, df_pooled,
                                        factor, level, j))
        sd2 = math.sqrt(v2)
        glass = diff / sd2 if sd2 > 0 else float("nan")
        g_lo, g_hi = ncp_interval(diff / (sd2 * math.sqrt(1 / n1 + 1 / n2)), n2 - 1, level) \
            if sd2 > 0 else (float("nan"), float("nan"))
        erows.append(["", "Glass's delta", fmt(sd2), fmt(glass), fmt(g_lo * factor),
                      fmt(g_hi * factor)])

    section.table(Table("Group Statistics", ["", gvar.title, "N", "Mean", "Std. Deviation",
                                             "Std. Error Mean"], grows, row_header_cols=2))
    lev_g = "Levene's Test for Equality of Variances"
    t_g = "t-test for Equality of Means"
    if trows:
        section.table(Table("Independent Samples Test", ["", "", (lev_g, "F"), (lev_g, "Sig."),
                            (t_g, "t"), (t_g, "df"), (t_g, "One-Sided p"),
                            (t_g, "Two-Sided p"), (t_g, "Mean Difference"),
                            (t_g, "Std. Error Difference"), (t_g, f"{pct(level)} CI Lower"),
                            (t_g, f"{pct(level)} CI Upper")], trows, row_header_cols=2))
        section.table(Table("Independent Samples Effect Sizes", _effect_columns(level), erows,
                            row_header_cols=2, footnotes=[
                                "a. The denominator used in estimating the effect sizes. "
                                "Cohen's d uses the pooled standard deviation. Hedges' "
                                "correction uses the pooled standard deviation, plus a "
                                "correction factor. Glass's delta uses the sample standard "
                                "deviation of the control (i.e., the second) group."]))
    return section


# ------------------------------------------------------------------ paired
def paired(ds: Dataset, pairs: list[tuple[str, str]], level: float = 0.95) -> Section:
    section = Section("T-Test")
    section.log("T-TEST PAIRS=" + " ".join(a for a, _ in pairs) + " WITH "
                + " ".join(b for _, b in pairs) + f" (PAIRED)\n  /CRITERIA=CI({level:.2f}).")
    srows, crows, trows, erows = [], [], [], []
    for k, (an, bn) in enumerate(pairs, start=1):
        va, vb = ds.var(an), ds.var(bn)
        frame = pd.DataFrame({"a": numeric(ds, an), "b": numeric(ds, bn)}).dropna()
        a, b = frame["a"].to_numpy(float), frame["b"].to_numpy(float)
        n = len(a)
        label = f"Pair {k}"
        for i, (v, x) in enumerate(((va, a), (vb, b))):
            sd = x.std(ddof=1) if n > 1 else float("nan")
            srows.append([label if i == 0 else "", v.title, fmt(x.mean() if n else np.nan,
                                                                 v.decimals + 2),
                          fmt_int(n), fmt(sd, v.decimals + 3),
                          fmt(sd / math.sqrt(n) if n else np.nan, v.decimals + 3)])
        if n < 2:
            continue
        r, rp = stats.pearsonr(a, b) if np.ptp(a) > 0 and np.ptp(b) > 0 else (np.nan, np.nan)
        crows.append([label, f"{va.title} & {vb.title}", fmt_int(n), fmt(r), fmt_p(rp / 2),
                      fmt_p(rp)])
        diff = a - b
        md, sdd = diff.mean(), diff.std(ddof=1)
        se = sdd / math.sqrt(n)
        t = md / se if se > 0 else float("nan")
        one, two = _p_sides(t, n - 1)
        lo, hi = t_interval(md, se, n - 1, level)
        dd = max(va.decimals, vb.decimals)
        trows.append([label, f"{va.title} - {vb.title}", fmt(md, dd + 3), fmt(sdd, dd + 3),
                      fmt(se, dd + 3), fmt(lo, dd + 3), fmt(hi, dd + 3), fmt(t), fmt_df(n - 1),
                      fmt_p(one), fmt_p(two)])
        factor = 1 / math.sqrt(n)
        erows.append([label, f"{va.title} - {vb.title}"]
                     + _effect_row("Cohen's d", sdd, t, n - 1, factor, level))
        j = hedges_j(n - 1)
        erows.append(["", ""] + _effect_row("Hedges' correction", sdd / j, t, n - 1, factor,
                                            level, j))
    section.table(Table("Paired Samples Statistics", ["", "", "Mean", "N", "Std. Deviation",
                                                      "Std. Error Mean"], srows,
                        row_header_cols=2))
    section.table(Table("Paired Samples Correlations", ["", "", "N", "Correlation",
                                                        ("Significance", "One-Sided p"),
                                                        ("Significance", "Two-Sided p")],
                        crows, row_header_cols=2))
    pd_g = "Paired Differences"
    section.table(Table("Paired Samples Test", ["", "", (pd_g, "Mean"),
                                                (pd_g, "Std. Deviation"),
                                                (pd_g, "Std. Error Mean"),
                                                (pd_g, f"{pct(level)} CI Lower"),
                                                (pd_g, f"{pct(level)} CI Upper"), "t", "df",
                                                ("Significance", "One-Sided p"),
                                                ("Significance", "Two-Sided p")],
                        trows, row_header_cols=2))
    if erows:
        section.table(Table("Paired Samples Effect Sizes", [""] + _effect_columns(level),
                            erows, row_header_cols=3, footnotes=[
                                "a. The denominator used in estimating the effect sizes. "
                                "Cohen's d uses the sample standard deviation of the mean "
                                "difference. Hedges' correction uses the sample standard "
                                "deviation of the mean difference, plus a correction factor."]))
    return section

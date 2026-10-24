"""Analyze > Descriptive Statistics > Explore: descriptives, tests of normality and plots."""
from __future__ import annotations

import numpy as np
import pandas as pd

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats import charts
from rsstatchem.core.stats.common import (Group, fmt, fmt_int, fmt_p, groups_of, lilliefors, numeric,
                                          pct, percentile, shapiro, t_interval, trimmed_mean)
from rsstatchem.core.stats.descriptive import summary


def _descriptive_rows(x: np.ndarray, decimals: int, level: float) -> list[tuple[str, str, str]]:
    r = summary(pd.Series(x))
    if r["n"] == 0:
        return []
    lo, hi = t_interval(r["mean"], r["sem"], r["n"] - 1, level)
    q1, q3 = percentile(x, [25, 75])
    d = decimals
    return [
        ("Mean", fmt(r["mean"], d + 2), fmt(r["sem"], d + 3)),
        (f"{pct(level)} Confidence Interval for Mean: Lower Bound", fmt(lo, d + 2), ""),
        (f"{pct(level)} Confidence Interval for Mean: Upper Bound", fmt(hi, d + 2), ""),
        ("5% Trimmed Mean", fmt(trimmed_mean(x), d + 2), ""),
        ("Median", fmt(r["median"], d + 2), ""),
        ("Variance", fmt(r["var"], d + 3), ""),
        ("Std. Deviation", fmt(r["std"], d + 3), ""),
        ("Minimum", fmt(r["min"], d), ""),
        ("Maximum", fmt(r["max"], d), ""),
        ("Range", fmt(r["range"], d), ""),
        ("Interquartile Range", fmt(q3 - q1, d + 2), ""),
        ("Skewness", fmt(r["skew"]), fmt(r["se_skew"])),
        ("Kurtosis", fmt(r["kurt"]), fmt(r["se_kurt"])),
    ]


def explore(ds: Dataset, dependents: list[str], factors: list[str] | None = None,
            statistics: bool = True, normality: bool = True, boxplot: bool = True,
            qq: bool = True, histogram: bool = False, level: float = 0.95) -> Section:
    factors = factors or []
    section = Section("Explore")
    log = f"EXAMINE VARIABLES={' '.join(dependents)}"
    if factors:
        log += f" BY {' '.join(factors)}"
    section.log(log + f"\n  /CINTERVAL {level * 100:g}\n  /MISSING LISTWISE.")
    if not dependents:
        return section

    frame = pd.concat({n: numeric(ds, n) for n in dependents} |
                      {f: ds.analysis_series(f) for f in factors}, axis=1)
    total = len(frame)
    frame = frame.dropna()

    # one "split" per factor, or the whole sample when no factor was given
    splits = [(f, ds.var(f)) for f in factors] or [(None, None)]

    has_factor = bool(factors)
    rh = 2 if has_factor else 1
    columns = [""] * rh + [("Valid", "N"), ("Valid", "Percent"), ("Missing", "N"),
                           ("Missing", "Percent"), ("Total", "N"), ("Total", "Percent")]
    rows = []
    for dep in dependents:
        title = ds.var(dep).title
        for factor, fvar in splits:
            for i, g in enumerate(_groups(ds, frame, dep, factor)):
                n_total = int((ds.analysis_series(factor) == g.value).sum()) if has_factor \
                    else total
                n_valid = g.n
                head = [title if i == 0 else "", f"{fvar.title} = {g.label}"] if has_factor \
                    else [title]
                rows.append(head + [str(n_valid), _pct(n_valid, n_total),
                                    str(n_total - n_valid), _pct(n_total - n_valid, n_total),
                                    str(n_total), "100.0%" if n_total else ""])
    section.table(Table("Case Processing Summary", columns, rows, row_header_cols=rh))

    if statistics:
        drows = []
        for dep in dependents:
            var = ds.var(dep)
            for factor, fvar in splits:
                for gi, g in enumerate(_groups(ds, frame, dep, factor)):
                    label, x = g.label, g.data
                    for si, (name, stat, se) in enumerate(_descriptive_rows(x, var.decimals,
                                                                              level)):
                        first = gi == 0 and si == 0
                        head = [var.title if first else ""]
                        if has_factor:
                            head.append(f"{fvar.title} = {label}" if si == 0 else "")
                        drows.append(head + [name, stat, se])
        section.table(Table("Descriptives", [""] * (rh + 1) + ["Statistic", "Std. Error"],
                            drows, row_header_cols=rh + 1))

    if normality:
        nrows, lower_bound = [], False
        for dep in dependents:
            var = ds.var(dep)
            for factor, fvar in splits:
                for gi, g in enumerate(_groups(ds, frame, dep, factor)):
                    label, x = g.label, g.data
                    d, p, lb = lilliefors(x)
                    w, sp = shapiro(x)
                    lower_bound |= lb
                    head = [var.title if gi == 0 else ""]
                    if has_factor:
                        head.append(f"{fvar.title} = {label}")
                    nrows.append(head + [fmt(d), fmt_int(len(x)), fmt_p(p) + ("*" if lb else ""),
                                         fmt(w), fmt_int(len(x)), fmt_p(sp)])
        notes = []
        if lower_bound:
            notes.append("*. This is a lower bound of the true significance.")
        notes.append("a. Lilliefors Significance Correction")
        section.table(Table("Tests of Normality", [""] * rh + [
            ("Kolmogorov-Smirnovᵃ", "Statistic"), ("Kolmogorov-Smirnovᵃ", "df"),
            ("Kolmogorov-Smirnovᵃ", "Sig."), ("Shapiro-Wilk", "Statistic"),
            ("Shapiro-Wilk", "df"), ("Shapiro-Wilk", "Sig.")], nrows, row_header_cols=rh,
            footnotes=notes))

    for dep in dependents:
        var = ds.var(dep)
        for factor, fvar in splits:
            groups = _groups(ds, frame, dep, factor)
            for g in groups:
                label, x = g.label, g.data
                suffix = f" ({fvar.title} = {label})" if factor else ""
                if histogram and len(x):
                    section.image(f"Histogram of {var.title}{suffix}",
                                  charts.histogram(x, f"Histogram{suffix}", var.title, False))
                if qq and len(x) > 2:
                    section.image(f"Normal Q-Q Plot of {var.title}{suffix}",
                                  charts.qq_plot(x, f"Normal Q-Q Plot of {var.title}{suffix}"))
            if boxplot and groups:
                section.image(f"Boxplot of {var.title}" + (f" by {fvar.title}" if factor else ""),
                              charts.boxplot([g.data for g in groups],
                                             [g.label for g in groups] if factor
                                             else [var.title],
                                             var.title, var.title))
    return section


def _groups(ds: Dataset, frame: pd.DataFrame, dep: str, factor: str | None) -> list[Group]:
    if factor is None:
        return [Group(None, "", frame[dep].to_numpy(float))]
    return groups_of(ds, dep, factor, frame=pd.DataFrame({"y": frame[dep], "g": frame[factor]}))


def _pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.1f}%" if whole else ""

"""Analyze > Nonparametric Tests > Legacy Dialogs (SPSS NPAR TESTS equivalents).

Chi-square (expected range / values), binomial, runs (median / mean / mode / custom), one-sample
K-S (normal with Lilliefors, uniform, Poisson, exponential), two independent samples
(Mann-Whitney U, Kolmogorov-Smirnov Z, Moses extreme reactions, Wald-Wolfowitz runs), k
independent samples (Kruskal-Wallis H, median, Jonckheere-Terpstra), two related samples
(Wilcoxon, sign, McNemar, marginal homogeneity) and k related samples (Friedman, Kendall's W,
Cochran's Q). Every procedure accepts the SPSS *Options…* settings: ``descriptives`` and
``quartiles`` ("Descriptive Statistics" table) and ``listwise`` (exclude cases listwise instead
of test-by-test). The statistics themselves live in :mod:`rsstatchem.core.stats.nptests`.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd
from scipy import stats

from rsstatchem.core.dataset import NUMERIC, Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats import nptests as T
from rsstatchem.core.stats.common import (fmt, fmt_int, fmt_p, lilliefors, numeric, sup,
                                          value_text)

EFFECT_NOTE = ("Effect size r = |Z| / √N (added by RS-StatChem; not part of the SPSS "
               "output).")
DISTRIBUTIONS = ("normal", "uniform", "poisson", "exponential")


# ------------------------------------------------------------------ helpers
def ks_details(x: np.ndarray) -> dict:
    """One-sample K-S against a normal with estimated mean/SD (Lilliefors p), as in SPSS."""
    x = np.asarray(x, float)
    n = len(x)
    out = {"n": n}
    if n < 2:
        return out
    mean, sd = x.mean(), x.std(ddof=1)
    out.update(mean=mean, sd=sd)
    if sd <= 0:
        return out
    values, counts = np.unique(x, return_counts=True)
    upper = np.cumsum(counts) / n
    lower = upper - counts / n
    cdf = stats.norm.cdf((values - mean) / sd)
    d_pos = float(np.max(upper - cdf))
    d_neg = float(np.max(cdf - lower))
    d, p, lower_bound = lilliefors(x)
    out.update(positive=d_pos, negative=-d_neg, absolute=max(d_pos, d_neg), statistic=d, p=p,
               lower_bound=lower_bound, p_exact=float(stats.kstwo.sf(max(d_pos, d_neg), n)))
    return out


def cut_value(x: np.ndarray, cutoff: str = "median", value: float | None = None
              ) -> tuple[float, bool]:
    """Runs-test cut point. Returns (value, multiple_modes)."""
    x = np.asarray(x, float)
    if cutoff == "mean":
        return float(np.mean(x)), False
    if cutoff == "mode":
        vals, counts = np.unique(x, return_counts=True)
        top = vals[counts == counts.max()]
        return float(top.max()), len(top) > 1  # SPSS uses the largest of multiple modes
    if cutoff == "custom":
        return float(value), False
    return float(np.median(x)), False


def runs_details(x: np.ndarray, cutoff: str = "median", value: float | None = None) -> dict:
    """Runs test around the median (default), mean, mode or a custom value; SPSS applies a
    continuity correction when there are fewer than 50 cases."""
    x = np.asarray(x, float)
    cut, multi = cut_value(x, cutoff, value) if len(x) else (float("nan"), False)
    out = T.runs_test(x, cut)
    out["multimode"] = multi
    return out


def _categories(series: pd.Series) -> list:
    return sorted(series.dropna().unique(), key=lambda v: (isinstance(v, str), v))


def _view(ds: Dataset, names: list[str], listwise: bool) -> Dataset:
    """The dataset restricted to cases valid on every variable (Missing Values: listwise)."""
    if not listwise or not names:
        return ds
    mask = pd.concat([ds.analysis_series(n).notna() for n in names], axis=1).all(axis=1)
    if mask.all():
        return ds
    return Dataset(ds.df[mask].reset_index(drop=True), copy.deepcopy(ds.variables), ds.name)


def descriptive_table(ds: Dataset, names: list[str], descriptives: bool = True,
                      quartiles: bool = False) -> Table | None:
    """The NPAR TESTS /STATISTICS DESCRIPTIVES QUARTILES table."""
    if not (descriptives or quartiles):
        return None
    cols: list = ["", "N"]
    if descriptives:
        cols += ["Mean", "Std. Deviation", "Minimum", "Maximum"]
    if quartiles:
        cols += [("Percentiles", "25th"), ("Percentiles", "50th (Median)"),
                 ("Percentiles", "75th")]
    rows, seen = [], set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        var = ds.var(name)
        if var.type != NUMERIC:
            continue
        x = numeric(ds, name).dropna().to_numpy(float)
        d = var.decimals
        r = [var.title, fmt_int(len(x))]
        if descriptives:
            r += [fmt(x.mean(), d + 2) if len(x) else "",
                  fmt(x.std(ddof=1), d + 3) if len(x) > 1 else "",
                  fmt(x.min(), d) if len(x) else "", fmt(x.max(), d) if len(x) else ""]
        if quartiles:
            r += [fmt(q, d + 2) for q in T.haverage(x, [25, 50, 75])] if len(x) else [""] * 3
        rows.append(r)
    return Table("Descriptive Statistics", cols, rows)


def _start(title: str, log: str, ds: Dataset, names: list[str], descriptives: bool,
           quartiles: bool, listwise: bool) -> tuple[Section, Dataset]:
    section = Section(title)
    extra = []
    if descriptives or quartiles:
        extra.append("/STATISTICS " + " ".join(k for k, on in (("DESCRIPTIVES", descriptives),
                                                                ("QUARTILES", quartiles)) if on))
    if listwise:
        extra.append("/MISSING LISTWISE")
    section.log(log.rstrip(".") + "".join("\n  " + e for e in extra) + ".")
    view = _view(ds, names, listwise)
    table = descriptive_table(view, names, descriptives, quartiles)
    if table is not None:
        section.table(table)
    return section, view


# ------------------------------------------------------------------ chi-square
def chi_square(ds: Dataset, names: list[str], expected: list[float] | None = None,
               expected_range: tuple[float, float] | None = None, descriptives: bool = False,
               quartiles: bool = False, listwise: bool = False) -> Section:
    rng_text = f"({expected_range[0]:g},{expected_range[1]:g})" if expected_range else ""
    section, ds = _start(
        "NPar Tests — Chi-Square Test",
        f"NPAR TESTS\n  /CHISQUARE={' '.join(names)}{rng_text}\n  /EXPECTED="
        f"{' '.join(f'{e:g}' for e in expected) if expected else 'EQUAL'}.",
        ds, names, descriptives, quartiles, listwise)
    test_cols, stat_rows = [], {"Chi-Square": [], "df": [], "Asymp. Sig.": []}
    notes = []
    for name in names:
        var = ds.var(name)
        s = ds.analysis_series(name).dropna()
        if expected_range is not None:
            lo, hi = int(math.ceil(expected_range[0])), int(math.floor(expected_range[1]))
            x = np.floor(pd.to_numeric(s, errors="coerce").dropna())
            x = x[(x >= lo) & (x <= hi)]
            cats = [float(c) for c in range(lo, hi + 1)]
            observed = np.array([int((x == c).sum()) for c in cats], float)
        else:
            cats = _categories(s)
            observed = np.array([int((s == c).sum()) for c in cats], float)
        n = observed.sum()
        if expected and len(expected) == len(cats):
            w = np.asarray(expected, float)
        else:
            if expected and len(expected) != len(cats):
                section.note("Warning", f"{var.title}: the number of expected values "
                             f"({len(expected)}) does not match the number of categories "
                             f"({len(cats)}); all categories are treated as equal.",
                             style="warning")
            w = np.ones(len(cats))
        exp = n * w / w.sum() if n else np.zeros(len(cats))
        if expected_range is not None:
            rows = [[str(i + 1), value_text(var, c), fmt_int(o), fmt(e, 1), fmt(o - e, 1)]
                    for i, (c, o, e) in enumerate(zip(cats, observed, exp))]
            rows.append(["Total", "", fmt_int(n), "", ""])
            section.table(Table(var.title, ["", "Category", "Observed N", "Expected N",
                                            "Residual"], rows))
        else:
            rows = [[value_text(var, c), fmt_int(o), fmt(e, 1), fmt(o - e, 1)]
                    for c, o, e in zip(cats, observed, exp)]
            rows.append(["Total", fmt_int(n), "", ""])
            section.table(Table(var.title, ["", "Observed N", "Expected N", "Residual"], rows))
        if len(cats) < 2 or n == 0:
            continue
        res = T.chisq_gof(observed, w)
        mark = sup(chr(ord("a") + len(notes)))
        stat_rows["Chi-Square"].append(fmt(res["chi2"]) + mark)
        stat_rows["df"].append(fmt_int(res["df"]))
        stat_rows["Asymp. Sig."].append(fmt_p(res["p"]))
        test_cols.append(var.title)
        letter = chr(ord("a") + len(notes))
        notes.append(f"{letter}. {res['small']} cells ({100 * res['small'] / len(cats):.1f}%) "
                     f"have expected frequencies less than 5. The minimum expected cell "
                     f"frequency is {res['expected'].min():.1f}.")
    if test_cols:
        section.table(Table("Test Statistics", [""] + test_cols,
                            [[k] + v for k, v in stat_rows.items()], footnotes=notes))
    return section


# ------------------------------------------------------------------ binomial
def binomial(ds: Dataset, names: list[str], test_prop: float = 0.5,
             cut: float | None = None, descriptives: bool = False, quartiles: bool = False,
             listwise: bool = False) -> Section:
    section, ds = _start(
        "NPar Tests — Binomial Test",
        f"NPAR TESTS\n  /BINOMIAL({test_prop:g})={' '.join(names)}"
        + (f"({cut:g})" if cut is not None else "") + ".",
        ds, names, descriptives, quartiles, listwise)
    rows = []
    two_sided = abs(test_prop - 0.5) < 1e-12
    for name in names:
        var = ds.var(name)
        s = ds.analysis_series(name).dropna()
        if cut is not None:
            x = pd.to_numeric(s, errors="coerce").dropna()
            labels = [f"<= {cut:g}", f"> {cut:g}"]
            counts = [int((x <= cut).sum()), int((x > cut).sum())]
        else:
            cats = _categories(s)
            if len(cats) != 2:
                section.note("Warning", f"{var.title} is not dichotomous ({len(cats)} distinct "
                             "values). Specify a cut point.", style="warning")
                continue
            # SPSS uses the first case's value as group 1
            first = s.iloc[0]
            cats = [first] + [c for c in cats if c != first]
            labels = [value_text(var, c) for c in cats]
            counts = [int((s == c).sum()) for c in cats]
        n = sum(counts)
        if n == 0:
            continue
        p = T.binomial_test(counts[0], n, test_prop)["p"]
        rows.append([var.title, "Group 1", labels[0], fmt_int(counts[0]), fmt(counts[0] / n, 2),
                     fmt(test_prop, 2), fmt_p(p)])
        # SPSS prints group 2's proportion as 1 - p1 (UCLA hsb2 female: 109/200 shown as .54)
        rows.append(["", "Group 2", labels[1], fmt_int(counts[1]), fmt(1 - counts[0] / n, 2),
                     "", ""])
        rows.append(["", "Total", "", fmt_int(n), "1.00", "", ""])
    sig = "Exact Sig. (2-tailed)" if two_sided else "Exact Sig. (1-tailed)"
    section.table(Table("Binomial Test", ["", "", "Category", "N", "Observed Prop.",
                                          "Test Prop.", sig], rows, row_header_cols=2))
    return section


# ------------------------------------------------------------------ K-S / runs
def ks_one_sample(ds: Dataset, names: list[str], dist: str = "normal", exact: bool = False,
                  descriptives: bool = False, quartiles: bool = False,
                  listwise: bool = False) -> Section:
    section, ds = _start("NPar Tests — One-Sample Kolmogorov-Smirnov Test",
                         f"NPAR TESTS\n  /K-S({dist.upper()})={' '.join(names)}.",
                         ds, names, descriptives, quartiles, listwise)
    if dist == "normal":
        results = [(ds.var(n), ks_details(numeric(ds, n).dropna().to_numpy())) for n in names]
        table = ks_table(results)
        if exact:
            table.rows.append(["Exact Sig. (2-tailed)", ""] + [
                fmt_p(r["p_exact"]) if "p_exact" in r else "" for _, r in results])
        section.table(table)
        return section
    results = []
    for n in names:
        x = numeric(ds, n).dropna().to_numpy(float)
        if dist == "poisson":
            keep = (x >= 0) & (x == np.round(x))
        elif dist == "exponential":
            keep = x >= 0
        else:
            keep = np.ones(len(x), bool)
        r = T.ks_one_sample_test(x[keep], dist)
        r["skipped"] = int((~keep).sum())
        results.append((ds.var(n), r))
    section.table(ks_table_dist(results, dist, exact))
    return section


def ks_table(results: list[tuple[object, dict]]) -> Table:
    lower = any(r.get("lower_bound") for _, r in results)

    def col(r: dict, key: str, d: int = 3) -> str:
        return fmt(r.get(key), d) if key in r else ""

    rows = [["N", ""] + [fmt_int(r["n"]) for _, r in results],
            ["Normal Parametersᵃ ᵇ", "Mean"] + [col(r, "mean", v.decimals + 4)
                                                for v, r in results],
            ["", "Std. Deviation"] + [col(r, "sd", v.decimals + 5) for v, r in results],
            ["Most Extreme Differences", "Absolute"] + [col(r, "absolute") for _, r in results],
            ["", "Positive"] + [col(r, "positive") for _, r in results],
            ["", "Negative"] + [col(r, "negative") for _, r in results],
            ["Test Statistic", ""] + [col(r, "statistic") for _, r in results],
            ["Asymp. Sig. (2-tailed)ᶜ", ""] + [
                (fmt_p(r["p"]) + ("ᵈ" if r.get("lower_bound") else "")) if "p" in r else ""
                for _, r in results]]
    notes = ["a. Test distribution is Normal.", "b. Calculated from data.",
             "c. Lilliefors Significance Correction."]
    if lower:
        notes.append("d. This is a lower bound of the true significance.")
    return Table("One-Sample Kolmogorov-Smirnov Test", ["", ""] + [v.title for v, _ in results],
                 rows, row_header_cols=2, footnotes=notes)


def ks_table_dist(results: list[tuple[object, dict]], dist: str, exact: bool = False) -> Table:
    """One-sample K-S table for the Uniform / Poisson / Exponential test distributions."""
    def col(r: dict, key: str, d: int = 3) -> str:
        return fmt(r.get(key), d) if key in r else ""

    def par(r: dict, key: str, d: int) -> str:
        return fmt(r.get("params", {}).get(key), d)

    rows = [["N", ""] + [fmt_int(r["n"]) for _, r in results]]
    if dist == "uniform":
        rows += [["Uniform Parametersᵃ ᵇ", "Minimum"] + [par(r, "min", v.decimals)
                                                        for v, r in results],
                 ["", "Maximum"] + [par(r, "max", v.decimals) for v, r in results]]
    elif dist == "poisson":
        rows.append(["Poisson Parameterᵃ ᵇ", "Mean"] + [par(r, "mean", v.decimals + 4)
                                                       for v, r in results])
    else:
        rows.append(["Exponential parameter.ᵃ ᵇ", "Mean"] + [par(r, "mean", v.decimals + 4)
                                                            for v, r in results])
    rows += [["Most Extreme Differences", "Absolute"] + [col(r, "absolute") for _, r in results],
             ["", "Positive"] + [col(r, "positive") for _, r in results],
             ["", "Negative"] + [col(r, "negative") for _, r in results],
             ["Kolmogorov-Smirnov Z", ""] + [col(r, "z") for _, r in results],
             ["Asymp. Sig. (2-tailed)", ""] + [fmt_p(r["p_kolmogorov"])
                                               if "p_kolmogorov" in r else ""
                                               for _, r in results]]
    if exact and dist != "poisson":
        rows.append(["Exact Sig. (2-tailed)", ""] + [fmt_p(r["p_exact"]) if "p_exact" in r
                                                     else "" for _, r in results])
    notes = [f"a. Test distribution is {dist.capitalize()}.", "b. Calculated from data."]
    letter = "c"
    for v, r in results:
        k = r.get("skipped", 0)
        if k:
            notes.append(f"{letter}. {v.title}: there {'is' if k == 1 else 'are'} {k} "
                         f"value{'' if k == 1 else 's'} outside the specified distribution "
                         f"range. {'This value is' if k == 1 else 'These values are'} skipped.")
            letter = chr(ord(letter) + 1)
    return Table("One-Sample Kolmogorov-Smirnov Test", ["", ""] + [v.title for v, _ in results],
                 rows, row_header_cols=2, footnotes=notes)


def runs_table(results: list[tuple[str, dict]], cutoff: str = "median") -> Table:
    rows = [["Test Valueᵃ"] + [fmt(r["value"], 5) for _, r in results],
            ["Cases < Test Value"] + [fmt_int(r["below"]) for _, r in results],
            ["Cases >= Test Value"] + [fmt_int(r["above"]) for _, r in results],
            ["Total Cases"] + [fmt_int(r["n"]) for _, r in results],
            ["Number of Runs"] + [fmt_int(r["runs"]) for _, r in results],
            ["Z"] + [fmt(r["z"]) for _, r in results],
            ["Asymp. Sig. (2-tailed)"] + [fmt_p(r["p"]) for _, r in results]]
    label = {"median": "Median", "mean": "Mean", "mode": "Mode",
             "custom": "User-Specified"}.get(cutoff, cutoff.capitalize())
    notes = [f"a. {label}"]
    if any(r.get("multimode") for _, r in results):
        notes.append("b. Multiple modes exist. The mode with the largest data value is used.")
        rows[0] = ["Test Valueᵃ"] + [fmt(r["value"], 5) + ("ᵇ" if r.get("multimode") else "")
                                     for _, r in results]
    return Table("Runs Test", [""] + [t for t, _ in results], rows, footnotes=notes)


def runs(ds: Dataset, names: list[str], cutoff: str = "median", value: float | None = None,
         descriptives: bool = False, quartiles: bool = False, listwise: bool = False) -> Section:
    spec = f"{value:g}" if cutoff == "custom" and value is not None else cutoff.upper()
    section, ds = _start("NPar Tests — Runs Test",
                         f"NPAR TESTS\n  /RUNS({spec})={' '.join(names)}.",
                         ds, names, descriptives, quartiles, listwise)
    results = [(ds.var(n).title, runs_details(numeric(ds, n).dropna().to_numpy(), cutoff, value))
               for n in names]
    section.table(runs_table(results, cutoff))
    return section


# ------------------------------------------------------------------ two independent
def _two_groups(ds: Dataset, name: str, group_var: str, groups) -> tuple[np.ndarray, np.ndarray]:
    g = ds.analysis_series(group_var)
    y = numeric(ds, name)
    g1, g2 = groups
    return (y[(g == g1) & y.notna()].to_numpy(float), y[(g == g2) & y.notna()].to_numpy(float))


def _frequencies_table(ds: Dataset, names: list[str], group_var: str, groups,
                       labels: tuple[str, str] | None = None) -> Table:
    gvar = ds.var(group_var)
    rows = []
    for name in names:
        a, b = _two_groups(ds, name, group_var, groups)
        first = labels or (value_text(gvar, groups[0]), value_text(gvar, groups[1]))
        rows += [[ds.var(name).title, first[0], fmt_int(len(a))],
                 ["", first[1], fmt_int(len(b))], ["", "Total", fmt_int(len(a) + len(b))]]
    return Table("Frequencies", ["", gvar.title, "N"], rows, row_header_cols=2)


def mann_whitney(ds: Dataset, names: list[str], group_var: str, groups,
                 effect_size: bool = True, descriptives: bool = False, quartiles: bool = False,
                 listwise: bool = False, _section: Section | None = None) -> Section:
    gvar = ds.var(group_var)
    g1, g2 = groups
    if _section is None:
        section, ds = _start("NPar Tests — Mann-Whitney Test",
                             f"NPAR TESTS\n  /M-W= {' '.join(names)} BY {group_var}({g1} {g2}).",
                             ds, names + [group_var], descriptives, quartiles, listwise)
    else:
        section = _section
    rank_rows, cols = [], []
    stat = {k: [] for k in ("Mann-Whitney U", "Wilcoxon W", "Z", "Asymp. Sig. (2-tailed)",
                            "Exact Sig. [2*(1-tailed Sig.)]", "Effect size rᶜ")}
    show_exact = False
    for name in names:
        var = ds.var(name)
        a, b = _two_groups(ds, name, group_var, groups)
        r = T.mann_whitney_test(a, b)
        n1, n2 = len(a), len(b)
        r1, r2 = r.get("r1", np.nan), r.get("r2", np.nan)
        for i, (label, nn, rs) in enumerate(((value_text(gvar, g1), n1, r1),
                                             (value_text(gvar, g2), n2, r2))):
            rank_rows.append([var.title if i == 0 else "", label, fmt_int(nn),
                              fmt(rs / nn if nn else np.nan, 2), fmt(rs, 2)])
        rank_rows.append(["", "Total", fmt_int(n1 + n2), "", ""])
        if n1 == 0 or n2 == 0:
            continue
        cols.append(var.title)
        stat["Mann-Whitney U"].append(fmt(r["u"]))
        stat["Wilcoxon W"].append(fmt(r["w"]))
        stat["Z"].append(fmt(r["z"]))
        stat["Asymp. Sig. (2-tailed)"].append(fmt_p(r["p"]))
        stat["Effect size rᶜ"].append(fmt(r["r"]))
        if "p_exact" in r:
            show_exact = True
            stat["Exact Sig. [2*(1-tailed Sig.)]"].append(fmt_p(r["p_exact"]) + "ᵇ")
        else:
            stat["Exact Sig. [2*(1-tailed Sig.)]"].append("")
    section.table(Table("Ranks", ["", gvar.title, "N", "Mean Rank", "Sum of Ranks"], rank_rows,
                        row_header_cols=2))
    if cols:
        letter = "c" if show_exact else "b"
        rows = [[k if k != "Effect size rᶜ" else "Effect size r" + sup(letter)] + v
                for k, v in stat.items()
                if (k != "Exact Sig. [2*(1-tailed Sig.)]" or show_exact)
                and (k != "Effect size rᶜ" or effect_size)]
        notes = [f"a. Grouping Variable: {gvar.title}"]
        if show_exact:
            notes.append("b. Not corrected for ties.")
        if effect_size:
            notes.append(f"{letter}. " + EFFECT_NOTE)
        section.table(Table("Test Statisticsᵃ", [""] + cols, rows, footnotes=notes))
    return section


def two_independent(ds: Dataset, names: list[str], group_var: str, groups,
                    mann_whitney_u: bool = True, ks: bool = False, moses: bool = False,
                    wald_wolfowitz: bool = False, moses_trim: int | None = None,
                    effect_size: bool = True, descriptives: bool = False,
                    quartiles: bool = False, listwise: bool = False) -> Section:
    """Two-Independent-Samples Tests with the four SPSS test types."""
    g1, g2 = groups
    gvar = ds.var(group_var)
    tests = [k for k, on in (("M-W", mann_whitney_u), ("MOSES", moses), ("K-S", ks),
                             ("W-W", wald_wolfowitz)) if on] or ["M-W"]
    log = "NPAR TESTS" + "".join(f"\n  /{t}= {' '.join(names)} BY {group_var}({g1} {g2})"
                                 for t in tests) + "."
    section, ds = _start("NPar Tests — Two-Independent-Samples Tests", log, ds,
                         names + [group_var], descriptives, quartiles, listwise)
    if "M-W" in tests:
        mann_whitney(ds, names, group_var, groups, effect_size, _section=section)
    if "MOSES" in tests:
        section.table(_frequencies_table(ds, names, group_var, groups,
                                         (f"{value_text(gvar, g1)} (Control)",
                                          f"{value_text(gvar, g2)} (Experimental)")))
        cols, rows = [], {k: [] for k in ("Observed Control Group Span", "Sig. (1-tailed)",
                                          "Trimmed Control Group Span", "Sig. (1-tailed) ",
                                          "Outliers Trimmed from each End")}
        for name in names:
            a, b = _two_groups(ds, name, group_var, groups)
            r = T.moses_test(a, b, moses_trim)
            if "span" not in r:
                continue
            cols.append(ds.var(name).title)
            for key, val in zip(rows, (fmt_int(r["span"]), fmt_p(r["p_span"]),
                                       fmt_int(r["trimmed"]), fmt_p(r["p_trimmed"]),
                                       fmt_int(r["h"]))):
                rows[key].append(val)
        if cols:
            section.table(Table("Test Statisticsᵃ ᵇ (Moses Test)", [""] + cols,
                                [[k.strip()] + v for k, v in rows.items()],
                                footnotes=["a. Moses Test",
                                           f"b. Grouping Variable: {gvar.title}"]))
    if "K-S" in tests:
        section.table(_frequencies_table(ds, names, group_var, groups))
        cols, rows = [], []
        results = []
        for name in names:
            a, b = _two_groups(ds, name, group_var, groups)
            r = T.ks_two_sample_test(a, b)
            if "z" in r:
                cols.append(ds.var(name).title)
                results.append(r)
        if cols:
            rows = [["Most Extreme Differences", "Absolute"]
                    + [fmt(r["absolute"]) for r in results],
                    ["", "Positive"] + [fmt(r["positive"]) for r in results],
                    ["", "Negative"] + [fmt(r["negative"]) for r in results],
                    ["Kolmogorov-Smirnov Z", ""] + [fmt(r["z"]) for r in results],
                    ["Asymp. Sig. (2-tailed)", ""] + [fmt_p(r["p"]) for r in results]]
            section.table(Table("Test Statisticsᵃ (Two-Sample Kolmogorov-Smirnov Test)",
                                ["", ""] + cols, rows, row_header_cols=2,
                                footnotes=[f"a. Grouping Variable: {gvar.title}"]))
    if "W-W" in tests:
        section.table(_frequencies_table(ds, names, group_var, groups))
        rows, notes, tie_notes = [], [], []
        for name in names:
            a, b = _two_groups(ds, name, group_var, groups)
            r = T.wald_wolfowitz_test(a, b)
            if "min" not in r:
                continue
            title = ds.var(name).title
            if r["exact_runs"]:
                d = r["min"]
                rows.append([title, "Exact Number of Runs", fmt_int(d["runs"]), fmt(d["z"]),
                             fmt_p(d["p_exact"])])
            else:
                for i, (label, d) in enumerate((("Minimum Possible", r["min"]),
                                                ("Maximum Possible", r["max"]))):
                    rows.append([title if i == 0 else "", label, fmt_int(d["runs"]),
                                 fmt(d["z"]), fmt_p(d["p_exact"])])
                tie_notes.append(f"{title}: there are {r['ties']} inter-group ties involving "
                                 f"{r['tie_cases']} cases.")
        if rows:
            notes = ["a. Wald-Wolfowitz Test", f"b. Grouping Variable: {gvar.title}"]
            notes += [f"{chr(ord('c') + i)}. {t}" for i, t in enumerate(tie_notes)]
            section.table(Table("Test Statisticsᵃ ᵇ (Wald-Wolfowitz Test)",
                                ["", "", "Number of Runs", "Z", "Exact Sig. (1-tailed)"], rows,
                                row_header_cols=2, footnotes=notes))
    return section


# ------------------------------------------------------------------ k independent
def _k_groups(ds: Dataset, name: str, group_var: str, lo, hi) -> tuple[list, list[np.ndarray]]:
    g = ds.analysis_series(group_var)
    if lo is not None and hi is not None:
        gnum = pd.to_numeric(g, errors="coerce")
        g = g.where((gnum >= lo) & (gnum <= hi))
    frame = pd.DataFrame({"y": numeric(ds, name), "g": g}).dropna()
    cats = _categories(frame["g"])
    return cats, [frame.loc[frame["g"] == c, "y"].to_numpy(float) for c in cats]


def kruskal_wallis(ds: Dataset, names: list[str], group_var: str,
                   lo: float | None = None, hi: float | None = None,
                   descriptives: bool = False, quartiles: bool = False,
                   listwise: bool = False, _section: Section | None = None) -> Section:
    gvar = ds.var(group_var)
    rng = f"({lo:g} {hi:g})" if lo is not None and hi is not None else ""
    if _section is None:
        section, ds = _start("NPar Tests — Kruskal-Wallis Test",
                             f"NPAR TESTS\n  /K-W={' '.join(names)} BY {group_var}{rng}.",
                             ds, names + [group_var], descriptives, quartiles, listwise)
    else:
        section = _section
    rank_rows, cols = [], []
    stat = {"Kruskal-Wallis H": [], "df": [], "Asymp. Sig.": []}
    for name in names:
        var = ds.var(name)
        cats, samples = _k_groups(ds, name, group_var, lo, hi)
        r = T.kruskal_test(samples)
        for i, (c, k, mr) in enumerate(zip(cats, r["sizes"], r["mean_ranks"])):
            rank_rows.append([var.title if i == 0 else "", value_text(gvar, c), fmt_int(k),
                              fmt(mr, 2)])
        rank_rows.append(["", "Total", fmt_int(r["n"]), ""])
        if len(samples) < 2 or not np.isfinite(r["h"]):
            continue
        cols.append(var.title)
        stat["Kruskal-Wallis H"].append(fmt(r["h"]))
        stat["df"].append(fmt_int(len(samples) - 1))
        stat["Asymp. Sig."].append(fmt_p(r["p"]))
    section.table(Table("Ranks", ["", gvar.title, "N", "Mean Rank"], rank_rows,
                        row_header_cols=2))
    if cols:
        section.table(Table("Test Statisticsᵃ ᵇ", [""] + cols,
                            [[k] + v for k, v in stat.items()],
                            footnotes=["a. Kruskal Wallis Test",
                                       f"b. Grouping Variable: {gvar.title}"]))
    return section


def k_independent(ds: Dataset, names: list[str], group_var: str, lo: float | None = None,
                  hi: float | None = None, kruskal: bool = True, median: bool = False,
                  jonckheere: bool = False, descriptives: bool = False,
                  quartiles: bool = False, listwise: bool = False) -> Section:
    """Tests for Several Independent Samples (Kruskal-Wallis H, Median, Jonckheere-Terpstra)."""
    gvar = ds.var(group_var)
    rng = f"({lo:g} {hi:g})" if lo is not None and hi is not None else ""
    tests = [k for k, on in (("K-W", kruskal), ("MEDIAN", median), ("J-T", jonckheere))
             if on] or ["K-W"]
    log = "NPAR TESTS" + "".join(f"\n  /{t}={' '.join(names)} BY {group_var}{rng}"
                                 for t in tests) + "."
    section, ds = _start("NPar Tests — Tests for Several Independent Samples", log, ds,
                         names + [group_var], descriptives, quartiles, listwise)
    if "K-W" in tests:
        kruskal_wallis(ds, names, group_var, lo, hi, _section=section)
    if "MEDIAN" in tests:
        for name in names:
            var = ds.var(name)
            cats, samples = _k_groups(ds, name, group_var, lo, hi)
            if len(samples) < 2:
                continue
            r = T.median_test(samples)
            labels = [value_text(gvar, c) for c in cats]
            section.table(Table("Frequencies", ["", ""] + [(gvar.title, lb) for lb in labels],
                                [[var.title, "> Median"] + [fmt_int(v) for v in r["above"]],
                                 ["", "<= Median"] + [fmt_int(v) for v in r["below"]]],
                                row_header_cols=2))
            rows = [["N", fmt_int(r["n"])], ["Median", fmt(r["median"], var.decimals + 4)]]
            notes = []
            if len(samples) == 2 and r["n"] <= 30 and "p_exact" in r:
                rows.append(["Exact Sig.", fmt_p(r["p_exact"])])
            elif np.isfinite(r.get("chi2", np.nan)):
                rows += [["Chi-Square", fmt(r["chi2"]) + "ᵇ"], ["df", fmt_int(r["df"])],
                         ["Asymp. Sig.", fmt_p(r["p"])]]
                if len(samples) == 2:
                    rows += [["Yates' Continuity Corrected", ""],
                             ["  Chi-Square", fmt(r["yates"])], ["  df", "1"],
                             ["  Asymp. Sig.", fmt_p(r["p_yates"])]]
                notes.append(f"b. {r['small']} cells ({100 * r['small'] / r['cells']:.1f}%) "
                             f"have expected frequencies less than 5. The minimum expected "
                             f"cell frequency is {r['expected'].min():.1f}.")
            section.table(Table("Test Statisticsᵃ (Median Test)", ["", var.title], rows,
                                footnotes=[f"a. Grouping Variable: {gvar.title}"] + notes))
    if "J-T" in tests:
        cols, results = [], []
        for name in names:
            cats, samples = _k_groups(ds, name, group_var, lo, hi)
            if len([s for s in samples if len(s)]) < 2:
                continue
            cols.append(ds.var(name).title)
            results.append(T.jonckheere_test(samples))
        if cols:
            rows = [[f"Number of Levels in {gvar.title}"] + [fmt_int(r["levels"]) for r in results],
                    ["N"] + [fmt_int(r["n"]) for r in results],
                    ["Observed J-T Statistic"] + [fmt(r["j"], 3) for r in results],
                    ["Mean J-T Statistic"] + [fmt(r["mean"], 3) for r in results],
                    ["Std. Deviation of J-T Statistic"] + [fmt(r["sd"], 3) for r in results],
                    ["Std. J-T Statistic"] + [fmt(r["z"]) for r in results],
                    ["Asymp. Sig. (2-tailed)"] + [fmt_p(r["p"]) for r in results]]
            section.table(Table("Jonckheere-Terpstra Testᵃ", [""] + cols, rows,
                                footnotes=[f"a. Grouping Variable: {gvar.title}"]))
    return section


# ------------------------------------------------------------------ two related
def two_related(ds: Dataset, pairs: list[tuple[str, str]], wilcoxon: bool = True,
                sign: bool = False, mcnemar: bool = False, marginal: bool = False,
                effect_size: bool = True, descriptives: bool = False, quartiles: bool = False,
                listwise: bool = False) -> Section:
    tests = " ".join(t for t, on in (("/WILCOXON", wilcoxon), ("/SIGN", sign),
                                     ("/MCNEMAR", mcnemar), ("/MH", marginal)) if on)
    names = [n for p in pairs for n in p]
    section, ds = _start("NPar Tests — Two Related Samples",
                         f"NPAR TESTS\n  {tests}={' '.join(a for a, _ in pairs)} WITH "
                         f"{' '.join(b for _, b in pairs)} (PAIRED).",
                         ds, names, descriptives, quartiles, listwise)
    w_rank_rows, w_cols, w_z, w_p, w_r, w_basis = [], [], [], [], [], []
    s_rows, s_cols, s_z, s_p = [], [], [], []
    s_exact = False
    for an, bn in pairs:
        va, vb = ds.var(an), ds.var(bn)
        title = f"{vb.title} - {va.title}"
        frame = pd.DataFrame({"a": numeric(ds, an), "b": numeric(ds, bn)}).dropna()
        d = (frame["b"] - frame["a"]).to_numpy(float)
        if wilcoxon:
            r = T.wilcoxon_test(d)
            neg, pos = r["neg"], r["pos"]
            w_rank_rows += [
                [title, "Negative Ranks", f"{neg}ᵃ", fmt(r["r_neg"] / neg if neg else np.nan, 2),
                 fmt(r["r_neg"], 2)],
                ["", "Positive Ranks", f"{pos}ᵇ", fmt(r["r_pos"] / pos if pos else np.nan, 2),
                 fmt(r["r_pos"], 2)],
                ["", "Ties", f"{r['ties']}ᶜ", "", ""],
                ["", "Total", fmt_int(len(d)), "", ""]]
            if r["n"] > 0:
                w_cols.append(title)
                w_z.append(fmt(r["z"]) + ("ᵇ" if r["basis"] == "negative" else "ᶜ"))
                w_p.append(fmt_p(r["p"]))
                w_r.append(fmt(abs(r["z"]) / math.sqrt(len(d))) if np.isfinite(r["z"]) else "")
                w_basis.append(r["basis"])
        if sign:
            r = T.sign_test(d)
            s_rows += [[title, "Negative Differencesᵃ", fmt_int(r["neg"])],
                       ["", "Positive Differencesᵇ", fmt_int(r["pos"])],
                       ["", "Tiesᶜ", fmt_int(r["ties"])], ["", "Total", fmt_int(len(d))]]
            if r["n"] == 0:
                continue
            s_cols.append(title)
            if r["exact"]:
                s_exact = True
                s_z.append("")
                s_p.append(fmt_p(r["p"]) + "ᵈ")
            else:
                s_z.append(fmt(r["z"]))
                s_p.append(fmt_p(r["p"]))
    pair_notes = ["a. " + " ; ".join(f"{ds.var(b).title} < {ds.var(a).title}" for a, b in pairs),
                  "b. " + " ; ".join(f"{ds.var(b).title} > {ds.var(a).title}" for a, b in pairs),
                  "c. " + " ; ".join(f"{ds.var(b).title} = {ds.var(a).title}" for a, b in pairs)]
    if wilcoxon:
        section.table(Table("Wilcoxon Signed Ranks Test — Ranks",
                            ["", "", "N", "Mean Rank", "Sum of Ranks"], w_rank_rows,
                            row_header_cols=2, footnotes=pair_notes))
        if w_cols:
            notes = ["a. Wilcoxon Signed Ranks Test"]
            if "negative" in w_basis:
                notes.append("b. Based on negative ranks.")
            if "positive" in w_basis:
                notes.append("c. Based on positive ranks.")
            rows = [["Z"] + w_z, ["Asymp. Sig. (2-tailed)"] + w_p]
            if effect_size:
                rows.append(["Effect size rᵈ"] + w_r)
                notes.append("d. " + EFFECT_NOTE)
            section.table(Table("Test Statisticsᵃ", [""] + w_cols, rows, footnotes=notes))
    if sign:
        section.table(Table("Sign Test — Frequencies", ["", "", "N"], s_rows, row_header_cols=2,
                            footnotes=pair_notes))
        if s_cols:
            rows = []
            if any(s_z):
                rows.append(["Z"] + s_z)
            rows.append(["Exact Sig. (2-tailed)" if s_exact and not any(s_z)
                         else "Asymp. Sig. (2-tailed)"] + s_p)
            notes = ["a. Sign Test"] + (["d. Binomial distribution used."] if s_exact else [])
            section.table(Table("Test Statisticsᵃ (Sign Test)", [""] + s_cols, rows,
                                footnotes=notes))
    if mcnemar:
        _mcnemar_tables(ds, section, pairs)
    if marginal:
        _marginal_table(ds, section, pairs)
    return section


def _mcnemar_tables(ds: Dataset, section: Section, pairs) -> None:
    cols, n_row, stat_row, p_row = [], [], [], []
    exact_any = asymp_any = False
    for an, bn in pairs:
        va, vb = ds.var(an), ds.var(bn)
        frame = pd.DataFrame({"a": ds.analysis_series(an),
                              "b": ds.analysis_series(bn)}).dropna()
        r = T.mcnemar_test(frame["a"].to_numpy(), frame["b"].to_numpy())
        if "error" in r:
            section.note("Warning", f"McNemar test: {va.title} & {vb.title} are not "
                         "dichotomous with the same two values. Use the Marginal Homogeneity "
                         "test.", style="warning")
            continue
        lo, hi = r["cats"]
        t = r["table"]
        section.table(Table(f"{va.title} & {vb.title}",
                            ["", (vb.title, value_text(vb, lo)), (vb.title, value_text(vb, hi))],
                            [[f"{va.title} = {value_text(va, lo)}", fmt_int(t[0, 0]),
                              fmt_int(t[0, 1])],
                             [f"{va.title} = {value_text(va, hi)}", fmt_int(t[1, 0]),
                              fmt_int(t[1, 1])]]))
        cols.append(f"{va.title} & {vb.title}")
        n_row.append(fmt_int(r["n"]))
        if r["exact"]:
            exact_any = True
            stat_row.append("")
            p_row.append(fmt_p(r["p"]) + "ᵇ")
        else:
            asymp_any = True
            stat_row.append(fmt(r["chi2"]) + "ᶜ")
            p_row.append(fmt_p(r["p"]))
    if not cols:
        return
    rows = [["N"] + n_row]
    if asymp_any:
        rows.append(["Chi-Square"] + stat_row)
    label = ("Exact Sig. (2-tailed)" if exact_any and not asymp_any else
             "Asymp. Sig." if not exact_any else "Sig. (2-tailed)")
    rows.append([label] + p_row)
    notes = ["a. McNemar Test"]
    if exact_any:
        notes.append("b. Binomial distribution used.")
    if asymp_any:
        notes.append("c. Continuity Corrected")
    section.table(Table("Test Statisticsᵃ (McNemar Test)", [""] + cols, rows, footnotes=notes))


def _marginal_table(ds: Dataset, section: Section, pairs) -> None:
    cols, results = [], []
    for an, bn in pairs:
        frame = pd.DataFrame({"a": numeric(ds, an), "b": numeric(ds, bn)}).dropna()
        cols.append(f"{ds.var(an).title} & {ds.var(bn).title}")
        results.append(T.marginal_homogeneity_test(frame["a"].to_numpy(), frame["b"].to_numpy()))
    rows = [["Distinct Values"] + [fmt_int(r["distinct"]) for r in results],
            ["Off-Diagonal Cases"] + [fmt_int(r["off"]) for r in results],
            ["Observed MH Statistic"] + [fmt(r["obs"], 3) for r in results],
            ["Mean MH Statistic"] + [fmt(r["mean"], 3) for r in results],
            ["Std. Deviation of MH Statistic"] + [fmt(r["sd"], 3) for r in results],
            ["Std. MH Statistic"] + [fmt(r["z"]) for r in results],
            ["Asymp. Sig. (2-tailed)"] + [fmt_p(r["p"]) for r in results]]
    section.table(Table("Marginal Homogeneity Test", [""] + cols, rows))


# ------------------------------------------------------------------ k related
def k_related(ds: Dataset, names: list[str], friedman: bool = True, kendall: bool = False,
              cochran: bool = False, descriptives: bool = False, quartiles: bool = False,
              listwise: bool = False) -> Section:
    tests = " ".join(t for t, on in (("/FRIEDMAN", friedman), ("/KENDALL", kendall),
                                     ("/COCHRAN", cochran)) if on)
    section, ds = _start("NPar Tests — Tests for Several Related Samples",
                         f"NPAR TESTS\n  {tests}={' '.join(names)}.",
                         ds, names, descriptives, quartiles, listwise)
    frame = pd.concat({n: numeric(ds, n) for n in names}, axis=1).dropna()
    n, k = frame.shape
    titles = [ds.var(v).title for v in names]
    if n == 0 or k < 2:
        section.note("Warning", "Not enough valid cases.", style="warning")
        return section
    x = frame.to_numpy(float)
    if friedman or kendall:
        fr = T.friedman_test(x)
        section.table(Table("Ranks", ["", "Mean Rank"],
                            [[t, fmt(r, 2)] for t, r in zip(titles, fr["mean_ranks"])]))
        if friedman:
            section.table(Table("Test Statisticsᵃ", ["", ""],
                                [["N", fmt_int(n)], ["Chi-Square", fmt(fr["chi2"])],
                                 ["df", fmt_int(k - 1)], ["Asymp. Sig.", fmt_p(fr["p"])]],
                                footnotes=["a. Friedman Test"]))
        if kendall:
            section.table(Table("Test Statistics (Kendall's W)", ["", ""],
                                [["N", fmt_int(n)], ["Kendall's Wᵃ", fmt(fr["w"])],
                                 ["Chi-Square", fmt(fr["chi2"])], ["df", fmt_int(k - 1)],
                                 ["Asymp. Sig.", fmt_p(fr["p"])]],
                                footnotes=["a. Kendall's Coefficient of Concordance"]))
    if cochran:
        values = np.unique(x)
        if len(values) > 2:
            section.note("Warning", "Cochran's Q requires dichotomous variables.",
                         style="warning")
            return section
        success = values.max()
        b = (x == success).astype(float)
        r = T.cochran_test(b)
        low = values.min() if len(values) == 2 else None
        rows = [[t, fmt_int(n - cj) if low is not None else "0", fmt_int(cj)]
                for t, cj in zip(titles, r["col_sums"])]
        section.table(Table("Frequencies", ["", ("Value", fmt(low, 0) if low is not None
                                                  else "0"),
                                            ("Value", fmt(success, 0))], rows))
        section.table(Table("Test Statistics (Cochran)", ["", ""],
                            [["N", fmt_int(n)], ["Cochran's Q", fmt(r["q"])],
                             ["df", fmt_int(k - 1)], ["Asymp. Sig.", fmt_p(r["p"])]],
                            footnotes=[f"{fmt(success, 0)} is treated as a success."]))
    return section

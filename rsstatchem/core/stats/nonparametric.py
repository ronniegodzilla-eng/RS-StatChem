"""Analyze > Nonparametric Tests > Legacy Dialogs (SPSS NPAR TESTS equivalents)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import (fmt, fmt_int, fmt_p, lilliefors, numeric,
                                          value_text)


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
               lower_bound=lower_bound)
    return out


def runs_details(x: np.ndarray, cutoff: str = "median") -> dict:
    """Runs test around the median (default) or mean; SPSS applies a continuity correction
    when there are fewer than 50 cases."""
    x = np.asarray(x, float)
    value = float(np.median(x)) if cutoff == "median" else float(np.mean(x))
    above = x >= value
    n1, n2 = int((~above).sum()), int(above.sum())
    n = n1 + n2
    runs = int(1 + np.count_nonzero(above[1:] != above[:-1])) if n else 0
    out = {"value": value, "below": n1, "above": n2, "n": n, "runs": runs,
           "z": float("nan"), "p": float("nan")}
    if n1 == 0 or n2 == 0 or n < 3:
        return out
    mu = 2.0 * n1 * n2 / n + 1
    var = 2.0 * n1 * n2 * (2.0 * n1 * n2 - n) / (n * n * (n - 1))
    diff = runs - mu
    if n < 50:
        diff = 0.0 if abs(diff) < 0.5 else diff - 0.5 * math.copysign(1, diff)
    z = diff / math.sqrt(var)
    out.update(z=z, p=2 * stats.norm.sf(abs(z)))
    return out


def _tie_term(ranks_source: np.ndarray) -> float:
    _, t = np.unique(ranks_source, return_counts=True)
    return float(((t ** 3) - t).sum())


def _categories(series: pd.Series) -> list:
    return sorted(series.dropna().unique(), key=lambda v: (isinstance(v, str), v))


# ------------------------------------------------------------------ chi-square
def chi_square(ds: Dataset, names: list[str], expected: list[float] | None = None) -> Section:
    section = Section("NPar Tests — Chi-Square Test")
    section.log(f"NPAR TESTS\n  /CHISQUARE={' '.join(names)}\n  /EXPECTED="
                f"{' '.join(f'{e:g}' for e in expected) if expected else 'EQUAL'}.")
    test_cols, stat_rows = [], {"Chi-Square": [], "df": [], "Asymp. Sig.": []}
    notes = []
    for idx, name in enumerate(names, start=1):
        var = ds.var(name)
        s = ds.analysis_series(name).dropna()
        cats = _categories(s)
        observed = np.array([int((s == c).sum()) for c in cats], float)
        n = observed.sum()
        if expected and len(expected) == len(cats):
            w = np.asarray(expected, float)
        else:
            w = np.ones(len(cats))
        exp = n * w / w.sum()
        rows = [[value_text(var, c), fmt_int(o), fmt(e, 1), fmt(o - e, 1)]
                for c, o, e in zip(cats, observed, exp)]
        rows.append(["Total", fmt_int(n), "", ""])
        section.table(Table(var.title, ["", "Observed N", "Expected N", "Residual"], rows))
        if len(cats) < 2:
            continue
        chi, p = stats.chisquare(observed, exp)
        small = int((exp < 5).sum())
        mark = chr(ord("a") + idx - 1)
        stat_rows["Chi-Square"].append(fmt(chi) + mark)
        stat_rows["df"].append(fmt_int(len(cats) - 1))
        stat_rows["Asymp. Sig."].append(fmt_p(p))
        test_cols.append(var.title)
        notes.append(f"{mark}. {small} cells ({100 * small / len(cats):.1f}%) have expected "
                     f"frequencies less than 5. The minimum expected cell frequency is "
                     f"{exp.min():.1f}.")
    if test_cols:
        section.table(Table("Test Statistics", [""] + test_cols,
                            [[k] + v for k, v in stat_rows.items()], footnotes=notes))
    return section


# ------------------------------------------------------------------ binomial
def binomial(ds: Dataset, names: list[str], test_prop: float = 0.5,
             cut: float | None = None) -> Section:
    section = Section("NPar Tests — Binomial Test")
    section.log(f"NPAR TESTS\n  /BINOMIAL({test_prop:g})={' '.join(names)}"
                + (f"({cut:g})" if cut is not None else "") + ".")
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
        k = counts[0]
        if two_sided:
            p = stats.binomtest(k, n, test_prop, alternative="two-sided").pvalue
        else:
            alt = "less" if k / n < test_prop else "greater"
            p = stats.binomtest(k, n, test_prop, alternative=alt).pvalue
        rows.append([var.title, "Group 1", labels[0], fmt_int(counts[0]), fmt(counts[0] / n, 2),
                     fmt(test_prop, 2), fmt_p(p)])
        rows.append(["", "Group 2", labels[1], fmt_int(counts[1]), fmt(counts[1] / n, 2), "", ""])
        rows.append(["", "Total", "", fmt_int(n), "1.00", "", ""])
    sig = "Exact Sig. (2-tailed)" if two_sided else "Exact Sig. (1-tailed)"
    section.table(Table("Binomial Test", ["", "", "Category", "N", "Observed Prop.",
                                          "Test Prop.", sig], rows, row_header_cols=2))
    return section


# ------------------------------------------------------------------ K-S / runs
def ks_one_sample(ds: Dataset, names: list[str]) -> Section:
    section = Section("NPar Tests — One-Sample Kolmogorov-Smirnov Test")
    section.log(f"NPAR TESTS\n  /K-S(NORMAL)={' '.join(names)}.")
    results = [(ds.var(n), ks_details(numeric(ds, n).dropna().to_numpy())) for n in names]
    section.table(ks_table(results))
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


def runs_table(results: list[tuple[str, dict]], cutoff: str = "median") -> Table:
    rows = [["Test Valueᵃ"] + [fmt(r["value"], 5) for _, r in results],
            ["Cases < Test Value"] + [fmt_int(r["below"]) for _, r in results],
            ["Cases >= Test Value"] + [fmt_int(r["above"]) for _, r in results],
            ["Total Cases"] + [fmt_int(r["n"]) for _, r in results],
            ["Number of Runs"] + [fmt_int(r["runs"]) for _, r in results],
            ["Z"] + [fmt(r["z"]) for _, r in results],
            ["Asymp. Sig. (2-tailed)"] + [fmt_p(r["p"]) for _, r in results]]
    return Table("Runs Test", [""] + [t for t, _ in results], rows,
                 footnotes=[f"a. {cutoff.capitalize()}"])


def runs(ds: Dataset, names: list[str], cutoff: str = "median") -> Section:
    section = Section("NPar Tests — Runs Test")
    section.log(f"NPAR TESTS\n  /RUNS({cutoff.upper()})={' '.join(names)}.")
    results = [(ds.var(n).title, runs_details(numeric(ds, n).dropna().to_numpy(), cutoff))
               for n in names]
    section.table(runs_table(results, cutoff))
    return section


# ------------------------------------------------------------------ two independent
def mann_whitney(ds: Dataset, names: list[str], group_var: str, groups) -> Section:
    gvar = ds.var(group_var)
    g1, g2 = groups
    section = Section("NPar Tests — Mann-Whitney Test")
    section.log(f"NPAR TESTS\n  /M-W= {' '.join(names)} BY {group_var}({g1} {g2}).")
    rank_rows, cols, stat = [], [], {k: [] for k in (
        "Mann-Whitney U", "Wilcoxon W", "Z", "Asymp. Sig. (2-tailed)",
        "Exact Sig. [2*(1-tailed Sig.)]")}
    show_exact = False
    g = ds.analysis_series(group_var)
    for name in names:
        var = ds.var(name)
        y = numeric(ds, name)
        a = y[(g == g1) & y.notna()].to_numpy(float)
        b = y[(g == g2) & y.notna()].to_numpy(float)
        n1, n2 = len(a), len(b)
        allx = np.concatenate([a, b])
        ranks = stats.rankdata(allx)
        r1, r2 = ranks[:n1].sum(), ranks[n1:].sum()
        for i, (label, nn, rs) in enumerate(((value_text(gvar, g1), n1, r1),
                                             (value_text(gvar, g2), n2, r2))):
            rank_rows.append([var.title if i == 0 else "", label, fmt_int(nn),
                              fmt(rs / nn if nn else np.nan, 2), fmt(rs, 2)])
        rank_rows.append(["", "Total", fmt_int(n1 + n2), "", ""])
        if n1 == 0 or n2 == 0:
            continue
        u1 = r1 - n1 * (n1 + 1) / 2
        u2 = n1 * n2 - u1
        u, w = (u1, r1) if u1 <= u2 else (u2, r2)
        n = n1 + n2
        var_u = n1 * n2 / 12.0 * ((n + 1) - _tie_term(allx) / (n * (n - 1)))
        z = (u - n1 * n2 / 2.0) / math.sqrt(var_u) if var_u > 0 else float("nan")
        p = 2 * stats.norm.sf(abs(z))
        cols.append(var.title)
        stat["Mann-Whitney U"].append(fmt(u))
        stat["Wilcoxon W"].append(fmt(w))
        stat["Z"].append(fmt(z))
        stat["Asymp. Sig. (2-tailed)"].append(fmt_p(p))
        if n1 * n2 <= 400:
            show_exact = True
            exact = stats.mannwhitneyu(a, b, alternative="two-sided", method="exact").pvalue
            stat["Exact Sig. [2*(1-tailed Sig.)]"].append(fmt_p(exact) + "ᵇ")
        else:
            stat["Exact Sig. [2*(1-tailed Sig.)]"].append("")
    section.table(Table("Ranks", ["", gvar.title, "N", "Mean Rank", "Sum of Ranks"], rank_rows,
                        row_header_cols=2))
    if cols:
        rows = [[k] + v for k, v in stat.items()
                if k != "Exact Sig. [2*(1-tailed Sig.)]" or show_exact]
        notes = [f"a. Grouping Variable: {gvar.title}"]
        if show_exact:
            notes.append("b. Not corrected for ties.")
        section.table(Table("Test Statisticsᵃ", [""] + cols, rows, footnotes=notes))
    return section


# ------------------------------------------------------------------ k independent
def kruskal_wallis(ds: Dataset, names: list[str], group_var: str,
                   lo: float | None = None, hi: float | None = None) -> Section:
    gvar = ds.var(group_var)
    section = Section("NPar Tests — Kruskal-Wallis Test")
    rng = f"({lo:g} {hi:g})" if lo is not None and hi is not None else ""
    section.log(f"NPAR TESTS\n  /K-W={' '.join(names)} BY {group_var}{rng}.")
    g = ds.analysis_series(group_var)
    if lo is not None and hi is not None:
        gnum = pd.to_numeric(g, errors="coerce")
        g = g.where((gnum >= lo) & (gnum <= hi))
    rank_rows, cols = [], []
    stat = {"Kruskal-Wallis H": [], "df": [], "Asymp. Sig.": []}
    for name in names:
        var = ds.var(name)
        frame = pd.DataFrame({"y": numeric(ds, name), "g": g}).dropna()
        cats = _categories(frame["g"])
        ranks = stats.rankdata(frame["y"].to_numpy(float))
        frame = frame.assign(r=ranks)
        samples = []
        for i, c in enumerate(cats):
            sub = frame[frame["g"] == c]
            samples.append(sub["y"].to_numpy(float))
            rank_rows.append([var.title if i == 0 else "", value_text(gvar, c),
                              fmt_int(len(sub)), fmt(sub["r"].mean(), 2)])
        rank_rows.append(["", "Total", fmt_int(len(frame)), ""])
        if len(samples) < 2:
            continue
        h, p = stats.kruskal(*samples)
        cols.append(var.title)
        stat["Kruskal-Wallis H"].append(fmt(h))
        stat["df"].append(fmt_int(len(samples) - 1))
        stat["Asymp. Sig."].append(fmt_p(p))
    section.table(Table("Ranks", ["", gvar.title, "N", "Mean Rank"], rank_rows,
                        row_header_cols=2))
    if cols:
        section.table(Table("Test Statisticsᵃ ᵇ", [""] + cols, [[k] + v for k, v in stat.items()],
                            footnotes=["a. Kruskal Wallis Test",
                                       f"b. Grouping Variable: {gvar.title}"]))
    return section


# ------------------------------------------------------------------ two related
def two_related(ds: Dataset, pairs: list[tuple[str, str]], wilcoxon: bool = True,
                sign: bool = False) -> Section:
    section = Section("NPar Tests — Two Related Samples")
    tests = " ".join(t for t, on in (("/WILCOXON", wilcoxon), ("/SIGN", sign)) if on)
    section.log(f"NPAR TESTS\n  {tests}={' '.join(a for a, _ in pairs)} WITH "
                f"{' '.join(b for _, b in pairs)} (PAIRED).")
    w_rank_rows, w_cols, w_z, w_p, w_basis = [], [], [], [], []
    s_rows, s_cols, s_z, s_p = [], [], [], []
    s_exact = False
    for an, bn in pairs:
        va, vb = ds.var(an), ds.var(bn)
        title = f"{vb.title} - {va.title}"
        frame = pd.DataFrame({"a": numeric(ds, an), "b": numeric(ds, bn)}).dropna()
        d = (frame["b"] - frame["a"]).to_numpy(float)
        neg, pos, ties = int((d < 0).sum()), int((d > 0).sum()), int((d == 0).sum())
        nz = d[d != 0]
        ranks = stats.rankdata(np.abs(nz)) if len(nz) else np.array([])
        r_neg, r_pos = ranks[nz < 0].sum(), ranks[nz > 0].sum()
        if wilcoxon:
            w_rank_rows += [
                [title, "Negative Ranks", f"{neg}ᵃ", fmt(r_neg / neg if neg else np.nan, 2),
                 fmt(r_neg, 2)],
                ["", "Positive Ranks", f"{pos}ᵇ", fmt(r_pos / pos if pos else np.nan, 2),
                 fmt(r_pos, 2)],
                ["", "Ties", f"{ties}ᶜ", "", ""],
                ["", "Total", fmt_int(len(d)), "", ""]]
            n = len(nz)
            if n > 0:
                t = min(r_neg, r_pos)
                var_t = n * (n + 1) * (2 * n + 1) / 24.0 - _tie_term(np.abs(nz)) / 48.0
                z = (t - n * (n + 1) / 4.0) / math.sqrt(var_t) if var_t > 0 else float("nan")
                w_cols.append(title)
                w_z.append(fmt(z) + ("ᵇ" if r_neg <= r_pos else "ᶜ"))
                w_p.append(fmt_p(2 * stats.norm.sf(abs(z))))
                w_basis.append("negative" if r_neg <= r_pos else "positive")
        if sign:
            s_rows += [[title, "Negative Differencesᵃ", fmt_int(neg)],
                       ["", "Positive Differencesᵇ", fmt_int(pos)],
                       ["", "Tiesᶜ", fmt_int(ties)], ["", "Total", fmt_int(len(d))]]
            n = neg + pos
            if n == 0:
                continue
            s_cols.append(title)
            if n <= 25:
                s_exact = True
                s_z.append("")
                s_p.append(fmt_p(stats.binomtest(min(neg, pos), n, 0.5).pvalue) + "ᵈ")
            else:
                z = (min(neg, pos) + 0.5 - n / 2.0) / (0.5 * math.sqrt(n))
                s_z.append(fmt(z))
                s_p.append(fmt_p(2 * stats.norm.sf(abs(z))))
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
            section.table(Table("Test Statisticsᵃ", [""] + w_cols,
                                [["Z"] + w_z, ["Asymp. Sig. (2-tailed)"] + w_p], footnotes=notes))
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
            section.table(Table("Test Statisticsᵃ", [""] + s_cols, rows, footnotes=notes))
    return section


# ------------------------------------------------------------------ k related
def k_related(ds: Dataset, names: list[str], friedman: bool = True, kendall: bool = False,
              cochran: bool = False) -> Section:
    section = Section("NPar Tests — Tests for Several Related Samples")
    tests = " ".join(t for t, on in (("/FRIEDMAN", friedman), ("/KENDALL", kendall),
                                     ("/COCHRAN", cochran)) if on)
    section.log(f"NPAR TESTS\n  {tests}={' '.join(names)}.")
    frame = pd.concat({n: numeric(ds, n) for n in names}, axis=1).dropna()
    n, k = frame.shape
    titles = [ds.var(v).title for v in names]
    if n == 0 or k < 2:
        section.note("Warning", "Not enough valid cases.", style="warning")
        return section
    ranks = np.apply_along_axis(stats.rankdata, 1, frame.to_numpy(float))
    mean_ranks = ranks.mean(axis=0)
    if friedman or kendall:
        section.table(Table("Ranks", ["", "Mean Rank"],
                            [[t, fmt(r, 2)] for t, r in zip(titles, mean_ranks)]))
        chi, p = stats.friedmanchisquare(*[frame[c].to_numpy(float) for c in names])
        if friedman:
            section.table(Table("Test Statisticsᵃ", ["", ""],
                                [["N", fmt_int(n)], ["Chi-Square", fmt(chi)],
                                 ["df", fmt_int(k - 1)], ["Asymp. Sig.", fmt_p(p)]],
                                footnotes=["a. Friedman Test"]))
        if kendall:
            w = chi / (n * (k - 1))
            section.table(Table("Test Statistics (Kendall's W)", ["", ""],
                                [["N", fmt_int(n)], ["Kendall's Wᵃ", fmt(w)],
                                 ["Chi-Square", fmt(chi)], ["df", fmt_int(k - 1)],
                                 ["Asymp. Sig.", fmt_p(p)]],
                                footnotes=["a. Kendall's Coefficient of Concordance"]))
    if cochran:
        values = np.unique(frame.to_numpy(float))
        if len(values) > 2:
            section.note("Warning", "Cochran's Q requires dichotomous variables.",
                         style="warning")
            return section
        success = values.max()
        x = (frame.to_numpy(float) == success).astype(float)
        c = x.sum(axis=0)
        r = x.sum(axis=1)
        total = x.sum()
        denom = k * total - (r ** 2).sum()
        q = (k - 1) * (k * (c ** 2).sum() - total ** 2) / denom if denom else float("nan")
        p = stats.chi2.sf(q, k - 1)
        low = values.min() if len(values) == 2 else None
        rows = [[t, fmt_int(n - cj) if low is not None else "0", fmt_int(cj)]
                for t, cj in zip(titles, c)]
        section.table(Table("Frequencies", ["", ("Value", fmt(low, 0) if low is not None
                                                  else "0"),
                                            ("Value", fmt(success, 0))], rows))
        section.table(Table("Test Statistics (Cochran)", ["", ""],
                            [["N", fmt_int(n)], ["Cochran's Q", fmt(q)], ["df", fmt_int(k - 1)],
                             ["Asymp. Sig.", fmt_p(p)]],
                            footnotes=[f"{fmt(success, 0)} is treated as a success."]))
    return section

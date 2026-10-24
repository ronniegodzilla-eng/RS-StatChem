"""One-Way ANOVA, post hoc multiple comparisons and GLM Univariate (factorial ANOVA / ANCOVA)."""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats import charts
from rsstatchem.core.stats.common import (Group, fmt, fmt_df, fmt_int, fmt_p, groups_of,
                                          levene_rows, numeric, pct, t_interval, value_text)

POSTHOC = {"lsd": "LSD", "bonferroni": "Bonferroni", "sidak": "Sidak", "scheffe": "Scheffe",
           "tukey": "Tukey HSD", "games_howell": "Games-Howell"}
POSTHOC_ORDER = ("tukey", "lsd", "bonferroni", "sidak", "scheffe", "games_howell")


# ------------------------------------------------------------------ post hoc
def posthoc(groups: list[Group], method: str, mse: float, df_error: float,
            level: float = 0.95) -> list[tuple[int, int, float, float, float, float, float]]:
    """All ordered pairs (i, j): mean difference, SE, p, CI lower, CI upper."""
    k = len(groups)
    m = k * (k - 1) / 2
    alpha = 1 - level
    out = []
    for i, j in itertools.permutations(range(k), 2):
        gi, gj = groups[i], groups[j]
        if gi.n == 0 or gj.n == 0:
            continue
        diff = gi.data.mean() - gj.data.mean()
        if method == "games_howell":
            vi = gi.data.var(ddof=1) / gi.n if gi.n > 1 else float("nan")
            vj = gj.data.var(ddof=1) / gj.n if gj.n > 1 else float("nan")
            se = math.sqrt(vi + vj)
            df = (vi + vj) ** 2 / (vi ** 2 / (gi.n - 1) + vj ** 2 / (gj.n - 1))
            q = abs(diff) / se * math.sqrt(2)
            p = float(stats.studentized_range.sf(q, k, df))
            half = stats.studentized_range.ppf(level, k, df) / math.sqrt(2) * se
            out.append((i, j, diff, se, p, diff - half, diff + half))
            continue
        se = math.sqrt(mse * (1 / gi.n + 1 / gj.n))
        t = diff / se if se > 0 else float("nan")
        p_lsd = 2 * stats.t.sf(abs(t), df_error)
        if method == "lsd":
            p, crit = p_lsd, stats.t.ppf(1 - alpha / 2, df_error)
        elif method == "bonferroni":
            p, crit = min(1.0, p_lsd * m), stats.t.ppf(1 - alpha / (2 * m), df_error)
        elif method == "sidak":
            a_s = 1 - (1 - alpha) ** (1 / m)
            p, crit = 1 - (1 - p_lsd) ** m, stats.t.ppf(1 - a_s / 2, df_error)
        elif method == "scheffe":
            f = t * t / (k - 1)
            p = float(stats.f.sf(f, k - 1, df_error))
            crit = math.sqrt((k - 1) * stats.f.ppf(level, k - 1, df_error))
        elif method == "tukey":
            q = abs(t) * math.sqrt(2)
            p = float(stats.studentized_range.sf(q, k, df_error))
            crit = stats.studentized_range.ppf(level, k, df_error) / math.sqrt(2)
        else:
            raise ValueError(method)
        out.append((i, j, diff, se, float(min(max(p, 0.0), 1.0)), diff - crit * se,
                    diff + crit * se))
    return out


def multiple_comparisons_table(title_rows: list[tuple[str, list[Group], str, list]],
                               level: float, factor_title: str) -> Table:
    """title_rows: (dependent title, groups, method key, comparisons)."""
    rows = []
    flagged = False
    for dep_title, groups, method, comps in title_rows:
        first_dep = True
        first_method = True
        last_i = None
        for i, j, diff, se, p, lo, hi in comps:
            star = "*" if p < 1 - level else ""
            flagged |= bool(star)
            rows.append([dep_title if first_dep else "",
                         POSTHOC[method] if first_method else "",
                         groups[i].label if i != last_i else "", groups[j].label,
                         fmt(diff) + star, fmt(se), fmt_p(p), fmt(lo), fmt(hi)])
            first_dep = first_method = False
            last_i = i
    notes = [f"*. The mean difference is significant at the {1 - level:.2f} level."] \
        if flagged else []
    return Table("Multiple Comparisons", ["Dependent Variable", "", f"(I) {factor_title}",
                                          f"(J) {factor_title}", "Mean Difference (I-J)",
                                          "Std. Error", "Sig.",
                                          (f"{pct(level)} Confidence Interval", "Lower Bound"),
                                          (f"{pct(level)} Confidence Interval", "Upper Bound")],
                 rows, row_header_cols=4, footnotes=notes)


# ------------------------------------------------------------------ one-way
def oneway(ds: Dataset, dependents: list[str], factor: str, descriptives: bool = True,
           homogeneity: bool = True, welch: bool = False, brown_forsythe: bool = False,
           effect_size: bool = True, posthoc_methods=(), means_plot: bool = False,
           level: float = 0.95) -> Section:
    fvar = ds.var(factor)
    section = Section("Oneway")
    stats_list = [s for s, on in (("DESCRIPTIVES", descriptives), ("HOMOGENEITY", homogeneity),
                                  ("BROWNFORSYTHE", brown_forsythe), ("WELCH", welch)) if on]
    log = f"ONEWAY {' '.join(dependents)} BY {factor}"
    if stats_list:
        log += f"\n  /STATISTICS {' '.join(stats_list)}"
    if posthoc_methods:
        log += (f"\n  /POSTHOC={' '.join(m.upper() for m in posthoc_methods)} "
                f"ALPHA({1 - level:.2f})")
    section.log(log + ".")

    desc_rows, lev_rows, anova_rows, eff_rows, robust_rows, mc = [], [], [], [], [], []
    for dep in dependents:
        var = ds.var(dep)
        d = var.decimals
        groups = [g for g in groups_of(ds, dep, factor) if g.n > 0]
        allx = np.concatenate([g.data for g in groups]) if groups else np.array([])
        k, n = len(groups), len(allx)
        if descriptives:
            for gi, g in enumerate(groups + [Group(None, "Total", allx)]):
                x = g.data
                sd = x.std(ddof=1) if len(x) > 1 else float("nan")
                se = sd / math.sqrt(len(x)) if len(x) else float("nan")
                lo, hi = t_interval(x.mean(), se, len(x) - 1, level) if len(x) > 1 \
                    else (float("nan"), float("nan"))
                desc_rows.append([var.title if gi == 0 else "", g.label, fmt_int(len(x)),
                                  fmt(x.mean(), d + 4), fmt(sd, d + 5), fmt(se, d + 5),
                                  fmt(lo, d + 4), fmt(hi, d + 4), fmt(x.min(), d),
                                  fmt(x.max(), d)])
        if k < 2 or n - k < 1:
            section.note("Warning", f"ANOVA cannot be computed for {var.title}: fewer than two "
                         "non-empty groups.", style="warning")
            continue
        if homogeneity:
            for li, (basis, f, df1, df2, p) in enumerate(levene_rows([g.data for g in groups])):
                lev_rows.append([var.title if li == 0 else "", basis, fmt(f), fmt_df(df1),
                                 fmt_df(df2), fmt_p(p)])
        grand = allx.mean()
        ssb = sum(g.n * (g.data.mean() - grand) ** 2 for g in groups)
        sst = float(((allx - grand) ** 2).sum())
        ssw = sst - ssb
        dfb, dfw = k - 1, n - k
        msb, msw = ssb / dfb, ssw / dfw
        f = msb / msw if msw > 0 else float("nan")
        p = float(stats.f.sf(f, dfb, dfw)) if msw > 0 else float("nan")
        anova_rows += [[var.title, "Between Groups", fmt(ssb), fmt_int(dfb), fmt(msb), fmt(f),
                        fmt_p(p)],
                       ["", "Within Groups", fmt(ssw), fmt_int(dfw), fmt(msw), "", ""],
                       ["", "Total", fmt(sst), fmt_int(n - 1), "", "", ""]]
        if effect_size:
            eta = ssb / sst if sst > 0 else float("nan")
            eps = (ssb - dfb * msw) / sst if sst > 0 else float("nan")
            omega = (ssb - dfb * msw) / (sst + msw) if sst > 0 else float("nan")
            eff_rows += [[var.title, "Eta-squared", fmt(eta)],
                         ["", "Epsilon-squared", fmt(eps)],
                         ["", "Omega-squared Fixed-effect", fmt(omega)]]
        if welch or brown_forsythe:
            from statsmodels.stats.oneway import anova_oneway

            first = True
            for title, use, on in (("Welch", "unequal", welch),
                                   ("Brown-Forsythe", "bf", brown_forsythe)):
                if not on:
                    continue
                try:
                    res = anova_oneway([g.data for g in groups], use_var=use,
                                       welch_correction=True)
                    df1, df2 = res.df
                    cells = [fmt(res.statistic), fmt_df(float(df1)), fmt(float(df2)),
                             fmt_p(res.pvalue)]
                except (ValueError, ZeroDivisionError):
                    cells = ["", "", "", ""]
                robust_rows.append([var.title if first else "", title] + cells)
                first = False
        for method in [m for m in POSTHOC_ORDER if m in posthoc_methods]:
            mc.append((var.title, groups, method, posthoc(groups, method, msw, dfw, level)))
        if means_plot:
            section.image(f"Means Plot: {var.title}",
                          charts.means_plot([g.label for g in groups],
                                            [g.data.mean() for g in groups],
                                            f"Mean of {var.title}", fvar.title,
                                            f"Mean of {var.title}"))

    if descriptives:
        section.table(Table("Descriptives", ["", "", "N", "Mean", "Std. Deviation",
                                             "Std. Error",
                                             (f"{pct(level)} CI for Mean", "Lower Bound"),
                                             (f"{pct(level)} CI for Mean", "Upper Bound"),
                                             "Minimum", "Maximum"], desc_rows,
                            row_header_cols=2))
    if lev_rows:
        section.table(Table("Tests of Homogeneity of Variances",
                            ["", "", "Levene Statistic", "df1", "df2", "Sig."], lev_rows,
                            row_header_cols=2))
    if anova_rows:
        section.table(Table("ANOVA", ["", "", "Sum of Squares", "df", "Mean Square", "F",
                                      "Sig."], anova_rows, row_header_cols=2))
    if eff_rows:
        section.table(Table("ANOVA Effect Sizes", ["", "", "Point Estimate"], eff_rows,
                            row_header_cols=2))
    if robust_rows:
        section.table(Table("Robust Tests of Equality of Means",
                            ["", "", "Statisticᵃ", "df1", "df2", "Sig."], robust_rows,
                            row_header_cols=2, footnotes=["a. Asymptotically F distributed."]))
    if mc:
        section.table(multiple_comparisons_table(mc, level, fvar.title))
    # Charts were appended while looping; keep tables first, as SPSS does.
    section.items.sort(key=lambda it: it.kind == "image")
    return section


# ------------------------------------------------------------------ GLM univariate
def univariate(ds: Dataset, dependent: str, factors: list[str],
               covariates: list[str] | None = None,
               descriptives: bool = False, effect_size: bool = True, homogeneity: bool = False,
               posthoc_factors: list[str] | None = None, posthoc_methods=(),
               level: float = 0.95) -> Section:
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    covariates = covariates or []
    posthoc_factors = posthoc_factors or []
    dvar = ds.var(dependent)
    section = Section("Univariate Analysis of Variance")
    log = f"UNIANOVA {dependent} BY {' '.join(factors)}"
    if covariates:
        log += f" WITH {' '.join(covariates)}"
    section.log(log + "\n  /METHOD=SSTYPE(3)\n  /DESIGN=" +
                " ".join(["*".join(c) for r in range(1, len(factors) + 1)
                          for c in itertools.combinations(factors, r)] + covariates) + ".")

    alias = {"y": dependent}
    data = {"y": numeric(ds, dependent)}
    fnames, cnames = [], []
    for i, f in enumerate(factors):
        data[f"f{i}"] = ds.analysis_series(f)
        alias[f"f{i}"] = f
        fnames.append(f"f{i}")
    for i, c in enumerate(covariates):
        data[f"c{i}"] = numeric(ds, c)
        alias[f"c{i}"] = c
        cnames.append(f"c{i}")
    frame = pd.DataFrame(data).dropna()
    if len(frame) < 3 or not factors and not covariates:
        section.note("Warning", "Not enough valid cases for the model.", style="warning")
        return section
    for fname in fnames:  # keep categories as strings so codes like 1.0 stay categorical
        frame[fname] = frame[fname].map(lambda v: repr(v))

    terms = [":".join(f"C({f}, Sum)" for f in combo)
             for r in range(1, len(fnames) + 1) for combo in itertools.combinations(fnames, r)]
    formula = "y ~ " + " + ".join(terms + cnames) if terms or cnames else "y ~ 1"
    model = smf.ols(formula, data=frame).fit()
    table = anova_lm(model, typ=3)

    y = frame["y"].to_numpy(float)
    n = len(y)
    sse = float(model.ssr)
    df_e = float(model.df_resid)
    mse = sse / df_e if df_e > 0 else float("nan")
    ss_ct = float(((y - y.mean()) ** 2).sum())
    ss_model = ss_ct - sse
    df_model = float(model.df_model)

    def label(term: str) -> str:
        if term == "Intercept":
            return "Intercept"
        parts = []
        for piece in term.split(":"):
            key = piece.replace("C(", "").replace(", Sum)", "")
            parts.append(ds.var(alias[key]).name)
        return " * ".join(parts)

    def row(source, ss, df, f=None, p=None):
        cells = [source, fmt(ss), fmt_int(df), fmt(ss / df) if df else "",
                 fmt(f) if f is not None else "", fmt_p(p) if p is not None else ""]
        if effect_size:
            cells.append(fmt(ss / (ss + sse)) if f is not None and ss + sse > 0 else "")
        return cells

    rows = []
    f_model = (ss_model / df_model) / mse if df_model > 0 and mse > 0 else float("nan")
    rows.append(row("Corrected Model", ss_model, df_model, f_model,
                    stats.f.sf(f_model, df_model, df_e)))
    for term, r in table.iterrows():
        if term == "Residual":
            continue
        rows.append(row(label(term), r["sum_sq"], r["df"], r["F"], r["PR(>F)"]))
    rows.append(row("Error", sse, df_e))
    rows.append(row("Total", float((y ** 2).sum()), n))
    rows.append(row("Corrected Total", ss_ct, n - 1))
    rows[0][0] = "Corrected Modelᵃ"
    r2 = ss_model / ss_ct if ss_ct > 0 else float("nan")
    adj = 1 - (1 - r2) * (n - 1) / df_e if df_e > 0 else float("nan")

    if descriptives and fnames:
        drows = []
        cells = frame.groupby(fnames, sort=True)["y"]
        for key, x in cells:
            key = key if isinstance(key, tuple) else (key,)
            labels = [value_text(ds.var(alias[f]), _unrepr(v)) for f, v in zip(fnames, key)]
            drows.append(labels + [fmt(x.mean(), dvar.decimals + 2),
                                   fmt(x.std(ddof=1), dvar.decimals + 3), fmt_int(len(x))])
        drows.append(["Total"] + [""] * (len(fnames) - 1) +
                     [fmt(y.mean(), dvar.decimals + 2), fmt(y.std(ddof=1), dvar.decimals + 3),
                      fmt_int(n)])
        section.table(Table("Descriptive Statistics",
                            [ds.var(f).name for f in factors] + ["Mean", "Std. Deviation", "N"],
                            drows, row_header_cols=len(fnames),
                            footnotes=[f"Dependent Variable: {dvar.title}"]))

    if homogeneity and fnames:
        source = model.resid if cnames else frame["y"]
        cell_key = frame[fnames].astype(str).agg("|".join, axis=1)
        samples = [source[cell_key == c].to_numpy(float) for c in sorted(cell_key.unique())]
        lrows = [[basis, fmt(f), fmt_df(df1), fmt_df(df2), fmt_p(p)]
                 for basis, f, df1, df2, p in levene_rows(samples)]
        section.table(Table("Levene's Test of Equality of Error Variances",
                            ["", "Levene Statistic", "df1", "df2", "Sig."], lrows,
                            footnotes=["Tests the null hypothesis that the error variance of "
                                       "the dependent variable is equal across groups.",
                                       f"Dependent variable: {dvar.title}"]))

    columns = ["Source", "Type III Sum of Squares", "df", "Mean Square", "F", "Sig."]
    if effect_size:
        columns.append("Partial Eta Squared")
    section.table(Table("Tests of Between-Subjects Effects", columns, rows,
                        footnotes=[f"Dependent Variable: {dvar.title}",
                                   f"a. R Squared = {fmt(r2)} (Adjusted R Squared = {fmt(adj)})"]))

    if posthoc_factors and posthoc_methods:
        if cnames:
            section.note("Warning", "Post hoc tests are not performed when covariates are in "
                         "the model.", style="warning")
        else:
            for f in posthoc_factors:
                fn = fnames[factors.index(f)]
                sub = pd.DataFrame({"y": frame["y"], "g": frame[fn].map(_unrepr)})
                groups = groups_of(ds, dependent, f, frame=sub)
                mc = [(dvar.title, groups, m, posthoc(groups, m, mse, df_e, level))
                      for m in POSTHOC_ORDER if m in posthoc_methods]
                if len(groups) > 2 and mc:
                    table_ = multiple_comparisons_table(mc, level, ds.var(f).title)
                    table_.title = f"Multiple Comparisons — {ds.var(f).title}"
                    table_.footnotes.append(f"Based on observed means. The error term is Mean "
                                            f"Square(Error) = {fmt(mse)}.")
                    section.table(table_)
                elif len(groups) <= 2:
                    section.note("", f"Post hoc tests are not performed for {ds.var(f).title} "
                                 "because there are fewer than three groups.")
    return section


def _unrepr(text: str):
    try:
        return float(text)
    except ValueError:
        return text.strip("'\"")

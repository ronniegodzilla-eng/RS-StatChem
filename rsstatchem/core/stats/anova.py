"""One-Way ANOVA, post hoc multiple comparisons and GLM Univariate (factorial ANOVA / ANCOVA).

One-Way ANOVA mirrors SPSS ONEWAY: descriptives (with fixed- and random-effects model rows),
homogeneity of variances, Welch / Brown-Forsythe, polynomial trend partitions of the
between-groups SS (unweighted, weighted, deviation), a priori contrasts with equal / unequal
variance tests, effect sizes with confidence intervals, all SPSS post hoc tests (see
:mod:`posthoc_ext`) with homogeneous subsets, means plot and missing-value handling.
GLM Univariate is implemented in :mod:`glm_output`; :func:`univariate` keeps the older call
signature.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy import optimize, stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats import charts, posthoc_ext
from rsstatchem.core.stats.common import (Group, fmt, fmt_df, fmt_int, fmt_p, groups_of,
                                          levene_rows, numeric, pct, sup, t_interval)

POSTHOC = dict(posthoc_ext.LABELS)
POSTHOC_ORDER = ("tukey", "lsd", "bonferroni", "sidak", "scheffe", "games_howell")
POLY_NAMES = ("Linear", "Quadratic", "Cubic", "4th-order", "5th-order")


# ------------------------------------------------------------------ post hoc
def posthoc(groups: list[Group], method: str, mse: float, df_error: float,
            level: float = 0.95) -> list[tuple[int, int, float, float, float, float, float]]:
    """All ordered pairs (i, j): mean difference, SE, p, CI lower, CI upper."""
    if method not in ("lsd", "bonferroni", "sidak", "scheffe", "tukey", "games_howell"):
        return posthoc_ext.pairwise(groups, method, mse, df_error,
                                    posthoc_ext.PostHocOptions(level))
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
        else:  # tukey
            q = abs(t) * math.sqrt(2)
            p = float(stats.studentized_range.sf(q, k, df_error))
            crit = stats.studentized_range.ppf(level, k, df_error) / math.sqrt(2)
        out.append((i, j, diff, se, float(min(max(p, 0.0), 1.0)), diff - crit * se,
                    diff + crit * se))
    return out


def multiple_comparisons_table(title_rows: list[tuple[str, list[Group], str, list]],
                               level: float, factor_title: str,
                               opts: posthoc_ext.PostHocOptions | None = None) -> Table:
    """title_rows: (dependent title, groups, method key, comparisons)."""
    rows = []
    flagged = False
    dunnett = False
    for dep_title, groups, method, comps in title_rows:
        first_dep = True
        first_method = True
        last_i = None
        label = posthoc_ext.method_label(method, opts)
        if method == "dunnett":
            label += sup("a")
            dunnett = True
        for i, j, diff, se, p, lo, hi in comps:
            if p == p:
                sig = p < 1 - level
            else:  # Dunnett's C: significance from the confidence interval
                sig = lo > 0 or hi < 0
            star = "*" if sig else ""
            flagged |= bool(star)
            rows.append([dep_title if first_dep else "", label if first_method else "",
                         groups[i].label if i != last_i else "", groups[j].label,
                         fmt(diff) + star, fmt(se), fmt_p(p) if p == p else "",
                         fmt(lo) if math.isfinite(lo) else "",
                         fmt(hi) if math.isfinite(hi) else ""])
            first_dep = first_method = False
            last_i = i
    notes = [f"*. The mean difference is significant at the {1 - level:.2f} level."] \
        if flagged else []
    if dunnett:
        notes.append("a. Dunnett t-tests treat one group as a control, and compare all other "
                     "groups against it.")
    return Table("Multiple Comparisons", ["Dependent Variable", "", f"(I) {factor_title}",
                                          f"(J) {factor_title}", "Mean Difference (I-J)",
                                          "Std. Error", "Sig.",
                                          (f"{pct(level)} Confidence Interval", "Lower Bound"),
                                          (f"{pct(level)} Confidence Interval", "Upper Bound")],
                 rows, row_header_cols=4, footnotes=notes)


# ------------------------------------------------------------------ helpers
def eta_interval(F: float, df1: float, df2: float, level: float) -> tuple[float, float]:
    """Confidence limits of eta-squared from the noncentral F (lambda / (lambda + N))."""
    if not np.isfinite(F) or df1 <= 0 or df2 <= 0:
        return float("nan"), float("nan")
    n_tot = df1 + df2 + 1
    a = (1 - level) / 2

    def lam_for(target):
        g = lambda lam: stats.ncf.cdf(F, df1, df2, lam) - target  # noqa: E731
        if g(0.0) < 0:
            return 0.0
        hi = max(10.0, F * df1 * 4)
        while g(hi) > 0 and hi < 1e7:
            hi *= 2
        return float(optimize.brentq(g, 0.0, hi, xtol=1e-9))

    lam_lo = lam_for(1 - a)
    lam_hi = lam_for(a)
    return lam_lo / (lam_lo + n_tot), lam_hi / (lam_hi + n_tot)


def _poly_rows(groups: list[Group], degree: int, ssb: float, msw: float, dfw: int):
    """(term name, kind, SS, df) rows of the SPSS trend partition."""
    k = len(groups)
    n = np.array([g.n for g in groups], float)
    m = np.array([g.data.mean() for g in groups])
    x = np.arange(1, k + 1, dtype=float)
    from rsstatchem.core.stats.glm_model import poly_coefficients

    P = poly_coefficients(k)
    rows = []
    cum_weighted = 0.0
    grand = float(n @ m / n.sum())
    prev_reg = 0.0
    for d in range(1, min(degree, k - 1) + 1):
        c = P[:, d - 1]
        ss_u = float((c @ m) ** 2 / np.sum(c * c / n))
        V = np.vander(x, d + 1, increasing=True)
        W = np.diag(n)
        beta = np.linalg.solve(V.T @ W @ V, V.T @ W @ m)
        reg = float(n @ (V @ beta - grand) ** 2)
        ss_w = reg - prev_reg
        prev_reg = reg
        cum_weighted += ss_w
        dev_df = k - 1 - d
        name = f"{POLY_NAMES[d - 1]} Term"
        rows.append((name, "Unweighted", ss_u, 1))
        rows.append(("", "Weighted", ss_w, 1))
        if dev_df > 0:
            rows.append(("", "Deviation", ssb - cum_weighted, dev_df))
    return rows


# ------------------------------------------------------------------ one-way
def oneway(ds: Dataset, dependents: list[str], factor: str, descriptives: bool = True,
           homogeneity: bool = True, welch: bool = False, brown_forsythe: bool = False,
           effect_size: bool = True, posthoc_methods=(), means_plot: bool = False,
           level: float = 0.95, polynomial: int = 0, contrasts: list[list[float]] | None = None,
           effects: bool = False, listwise: bool = False,
           posthoc_opts: posthoc_ext.PostHocOptions | None = None,
           posthoc_level: float | None = None) -> Section:
    """SPSS ONEWAY. ``level`` is the confidence level for intervals; ``posthoc_level`` the
    (1 - significance level) of the post hoc tests (defaults to ``level``)."""
    fvar = ds.var(factor)
    contrasts = [list(c) for c in (contrasts or []) if c]
    ph_level = posthoc_level if posthoc_level is not None else level
    popts = posthoc_opts or posthoc_ext.PostHocOptions()
    popts = posthoc_ext.PostHocOptions(ph_level, popts.control, popts.dunnett_test,
                                       popts.k_ratio)
    section = Section("Oneway")
    stats_list = [s for s, on in (("DESCRIPTIVES", descriptives), ("EFFECTS", effects),
                                  ("HOMOGENEITY", homogeneity),
                                  ("BROWNFORSYTHE", brown_forsythe), ("WELCH", welch)) if on]
    log = f"ONEWAY {' '.join(dependents)} BY {factor}"
    if effect_size:
        log += "\n  /ES=OVERALL"
    if polynomial:
        log += f"\n  /POLYNOMIAL={polynomial}"
    for c in contrasts:
        log += "\n  /CONTRAST=" + " ".join(f"{v:g}" for v in c)
    if stats_list:
        log += f"\n  /STATISTICS {' '.join(stats_list)}"
    if means_plot:
        log += "\n  /PLOT MEANS"
    log += f"\n  /MISSING {'LISTWISE' if listwise else 'ANALYSIS'}"
    log += f"\n  /CRITERIA=CILEVEL({level:.2f})"
    if posthoc_methods:
        log += (f"\n  /POSTHOC={' '.join(m.upper() for m in posthoc_methods)} "
                f"ALPHA({1 - ph_level:.2f})")
    section.log(log + ".")

    frame_all = None
    if listwise:
        frame_all = pd.DataFrame({d: numeric(ds, d) for d in dependents})
        frame_all["__g"] = ds.analysis_series(factor)
        frame_all = frame_all.dropna()

    desc_rows, lev_rows, anova_rows, eff_rows, robust_rows = [], [], [], [], []
    coef_rows, ctest_rows, mc, subset_tables = [], [], [], []
    warn_negative = False
    for dep in dependents:
        var = ds.var(dep)
        d = var.decimals
        sub = None if frame_all is None else pd.DataFrame({"y": frame_all[dep],
                                                           "g": frame_all["__g"]})
        groups = [g for g in groups_of(ds, dep, factor, frame=sub) if g.n > 0]
        allx = np.concatenate([g.data for g in groups]) if groups else np.array([])
        k, n = len(groups), len(allx)
        grand = allx.mean() if n else float("nan")
        ok = k >= 2 and n - k >= 1
        if ok:
            ssb = sum(g.n * (g.data.mean() - grand) ** 2 for g in groups)
            sst = float(((allx - grand) ** 2).sum())
            ssw = sst - ssb
            dfb, dfw = k - 1, n - k
            msb, msw = ssb / dfb, ssw / dfw
        if descriptives:
            first = True
            for gi, g in enumerate(groups + [Group(None, "Total", allx)]):
                x = g.data
                sd = x.std(ddof=1) if len(x) > 1 else float("nan")
                se = sd / math.sqrt(len(x)) if len(x) else float("nan")
                lo, hi = t_interval(x.mean(), se, len(x) - 1, level) if len(x) > 1 \
                    else (float("nan"), float("nan"))
                row = [var.title if first else "", "", g.label, fmt_int(len(x)),
                       fmt(x.mean(), d + 4), fmt(sd, d + 5), fmt(se, d + 5),
                       fmt(lo, d + 4), fmt(hi, d + 4), fmt(x.min(), d), fmt(x.max(), d)]
                if effects:
                    row.append("")
                desc_rows.append(row)
                first = False
            if effects and ok:
                se_f = math.sqrt(msw / n)
                lo, hi = t_interval(grand, se_f, dfw, level)
                desc_rows.append(["", "Model", "Fixed Effects", "", "", fmt(math.sqrt(msw), d + 5),
                                  fmt(se_f, d + 5), fmt(lo, d + 4), fmt(hi, d + 4), "", "", ""])
                n0 = (n - sum(g.n ** 2 for g in groups) / n) / (k - 1)
                var_b = (msb - msw) / n0
                neg = var_b < 0
                warn_negative |= neg
                se_r = math.sqrt(max(var_b, 0.0) / k + msw / n)
                lo, hi = t_interval(grand, se_r, k - 1, level)
                desc_rows.append(["", "", "Random Effects", "", "", "",
                                  fmt(se_r, d + 5) + (sup("a") if neg else ""),
                                  fmt(lo, d + 4) + (sup("a") if neg else ""),
                                  fmt(hi, d + 4) + (sup("a") if neg else ""), "", "",
                                  fmt(var_b, d + 5)])
        if not ok:
            section.note("Warning", f"ANOVA cannot be computed for {var.title}: fewer than two "
                         "non-empty groups.", style="warning")
            continue
        if homogeneity:
            for li, (basis, f, df1, df2, p) in enumerate(levene_rows([g.data for g in groups])):
                lev_rows.append([var.title if li == 0 else "", basis, fmt(f), fmt_df(df1),
                                 fmt_df(df2), fmt_p(p)])
        f = msb / msw if msw > 0 else float("nan")
        p = float(stats.f.sf(f, dfb, dfw)) if msw > 0 else float("nan")
        if polynomial:
            anova_rows.append([var.title, "Between Groups", "(Combined)", fmt(ssb),
                               fmt_int(dfb), fmt(msb), fmt(f), fmt_p(p)])
            for name, kind, ss, df in _poly_rows(groups, polynomial, ssb, msw, dfw):
                F_ = (ss / df) / msw if msw > 0 else float("nan")
                anova_rows.append(["", name, kind, fmt(ss), fmt_int(df), fmt(ss / df), fmt(F_),
                                   fmt_p(stats.f.sf(F_, df, dfw))])
            anova_rows += [["", "Within Groups", "", fmt(ssw), fmt_int(dfw), fmt(msw), "", ""],
                           ["", "Total", "", fmt(sst), fmt_int(n - 1), "", "", ""]]
        else:
            anova_rows += [[var.title, "Between Groups", fmt(ssb), fmt_int(dfb), fmt(msb),
                            fmt(f), fmt_p(p)],
                           ["", "Within Groups", fmt(ssw), fmt_int(dfw), fmt(msw), "", ""],
                           ["", "Total", fmt(sst), fmt_int(n - 1), "", "", ""]]
        if effect_size:
            eta = ssb / sst if sst > 0 else float("nan")
            eps = (ssb - dfb * msw) / sst if sst > 0 else float("nan")
            omega = (ssb - dfb * msw) / (sst + msw) if sst > 0 else float("nan")
            e_lo, e_hi = eta_interval(f, dfb, dfw, level)

            def to_eps(e):
                return 1 - (1 - e) * (n - 1) / dfw

            def to_omega(e):
                return (e - dfb * (1 - e) / dfw) / (1 + (1 - e) / dfw)

            eff_rows += [[var.title, "Eta-squared", fmt(eta), fmt(e_lo), fmt(e_hi)],
                         ["", "Epsilon-squared", fmt(eps), fmt(to_eps(e_lo)), fmt(to_eps(e_hi))],
                         ["", "Omega-squared Fixed-effect", fmt(omega), fmt(to_omega(e_lo)),
                          fmt(to_omega(e_hi))],
                         ["", "Omega-squared Random-effect"] + _omega_random(
                             f, dfb, dfw, groups, level)]
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
        if contrasts:
            _contrast_rows(var, groups, contrasts, msw, dfw, level, coef_rows, ctest_rows,
                           first_dep=dep == dependents[0])
        methods = [m for m in posthoc_ext.ORDER if m in posthoc_methods]
        pair = [m for m in posthoc_ext.PAIRWISE if m in posthoc_methods]
        for method in pair:
            mc.append((var.title, groups, method,
                       posthoc_ext.pairwise(groups, method, msw, dfw, popts)))
        sub_m = [m for m in methods if m in posthoc_ext.SUBSETS]
        if sub_m:
            t = posthoc_ext.subsets_table(var.title, fvar.title, groups, sub_m, msw, dfw,
                                          popts, f, d)
            if t is not None:
                t.title = f"Homogeneous Subsets: {var.title}"
                subset_tables.append(t)
        if means_plot:
            section.image(f"Means Plot: {var.title}",
                          charts.means_plot([g.label for g in groups],
                                            [g.data.mean() for g in groups],
                                            f"Mean of {var.title}", fvar.title,
                                            f"Mean of {var.title}"))

    if descriptives:
        cols = ["", "", "", "N", "Mean", "Std. Deviation", "Std. Error",
                (f"{pct(level)} Confidence Interval for Mean", "Lower Bound"),
                (f"{pct(level)} Confidence Interval for Mean", "Upper Bound"),
                "Minimum", "Maximum"]
        if effects:
            cols.append("Between-Component Variance")
        if not effects:
            desc_rows = [r[:1] + r[2:] for r in desc_rows]
            cols = cols[:1] + cols[2:]
        foot = []
        if warn_negative:
            foot.append("a. Warning: Between-component variance is negative. It was replaced "
                        "by 0.0 in computing this random effects measure.")
        section.table(Table("Descriptives", cols, desc_rows, row_header_cols=3 if effects else 2,
                            footnotes=foot))
    if lev_rows:
        section.table(Table("Tests of Homogeneity of Variances",
                            ["", "", "Levene Statistic", "df1", "df2", "Sig."], lev_rows,
                            row_header_cols=2))
    if anova_rows:
        if polynomial:
            section.table(Table("ANOVA", ["", "", "", "Sum of Squares", "df", "Mean Square",
                                          "F", "Sig."], anova_rows, row_header_cols=3))
        else:
            section.table(Table("ANOVA", ["", "", "Sum of Squares", "df", "Mean Square", "F",
                                          "Sig."], anova_rows, row_header_cols=2))
    if robust_rows:
        section.table(Table("Robust Tests of Equality of Means",
                            ["", "", "Statisticᵃ", "df1", "df2", "Sig."], robust_rows,
                            row_header_cols=2, footnotes=["a. Asymptotically F distributed."]))
    if eff_rows:
        section.table(Table("ANOVA Effect Sizes" + sup("a") + "," + sup("b"),
                            ["", "", "Point Estimate",
                             (f"{pct(level)} Confidence Interval", "Lower"),
                             (f"{pct(level)} Confidence Interval", "Upper")], eff_rows,
                            row_header_cols=2,
                            footnotes=["a. Eta-squared and Epsilon-squared are estimated based "
                                       "on the fixed-effect model.",
                                       "b. Negative but less biased estimates are retained, "
                                       "not rounded to zero."]))
    if coef_rows:
        groups0 = [g for g in groups_of(ds, dependents[0], factor) if g.n > 0]
        section.table(Table("Contrast Coefficients",
                            ["Contrast"] + [(fvar.title, g.label) for g in groups0],
                            coef_rows))
    if ctest_rows:
        section.table(Table("Contrast Tests",
                            ["", "", "Contrast", "Value of Contrast", "Std. Error", "t", "df",
                             "Sig. (2-tailed)",
                             (f"{pct(level)} Confidence Interval", "Lower"),
                             (f"{pct(level)} Confidence Interval", "Upper")],
                            ctest_rows, row_header_cols=3))
    if mc:
        section.table(multiple_comparisons_table(mc, ph_level, fvar.title, popts))
    for t in subset_tables:
        section.table(t)
    # Charts were appended while looping; keep tables first, as SPSS does.
    section.items.sort(key=lambda it: it.kind == "image")
    return section


def _omega_random(F: float, dfb: int, dfw: int, groups: list[Group], level: float
                  ) -> list[str]:
    """Random-effect omega squared as in the IBM SPSS Statistics Algorithms (ONEWAY, "Effect
    Size for One-way ANOVA", v31 edition): point (BSSM - WSSM)/(BSSM + df_t WSSM) =
    (F - 1)/(F + df_t), df_t = N - 1; limits (df_t eta^2 - df_e)/((df_w - df_t df_e) eta^2 +
    df_t df_e) evaluated at the noncentral-F (Steiger 2004) limits of eta^2. Negative values
    are shown as they are."""
    dft = dfb + dfw
    e_lo, e_hi = eta_interval(F, dfb, dfw, level)

    def lim(e):
        den = (dfw - dft * dfb) * e + dft * dfb
        return (dft * e - dfb) / den if den else float("nan")

    return [fmt((F - 1) / (F + dft)), fmt(lim(e_lo)), fmt(lim(e_hi))]


def _contrast_rows(var, groups: list[Group], contrasts, msw, dfw, level, coef_rows,
                   ctest_rows, first_dep: bool) -> None:
    k = len(groups)
    n = np.array([g.n for g in groups], float)
    m = np.array([g.data.mean() for g in groups])
    v = np.array([g.data.var(ddof=1) if g.n > 1 else np.nan for g in groups])
    eq_block, uneq_block = [], []
    for ci, c in enumerate(contrasts, start=1):
        c = np.array((list(c) + [0.0] * k)[:k], float)
        if first_dep:
            coef_rows.append([str(ci)] + [f"{x:g}" for x in c])
        val = float(c @ m)
        se_e = math.sqrt(msw * np.sum(c * c / n))
        t_e = val / se_e if se_e > 0 else float("nan")
        lo, hi = t_interval(val, se_e, dfw, level)
        eq_block.append([str(ci), fmt(val), fmt(se_e), fmt(t_e), fmt_df(dfw),
                         fmt_p(2 * stats.t.sf(abs(t_e), dfw)), fmt(lo), fmt(hi)])
        terms = c * c * v / n
        se_u = math.sqrt(np.nansum(terms))
        df_u = se_u ** 4 / np.nansum(terms ** 2 / (n - 1)) if se_u > 0 else float("nan")
        t_u = val / se_u if se_u > 0 else float("nan")
        lo, hi = t_interval(val, se_u, df_u, level)
        uneq_block.append([str(ci), fmt(val), fmt(se_u), fmt(t_u), fmt(df_u),
                           fmt_p(2 * stats.t.sf(abs(t_u), df_u)), fmt(lo), fmt(hi)])
    for bi, (name, block) in enumerate((("Assume equal variances", eq_block),
                                        ("Does not assume equal variances", uneq_block))):
        for ri, r in enumerate(block):
            ctest_rows.append([var.title if bi == 0 and ri == 0 else "",
                               name if ri == 0 else ""] + r)


# ------------------------------------------------------------------ GLM univariate
def univariate(ds: Dataset, dependent: str, factors: list[str],
               covariates: list[str] | None = None,
               descriptives: bool = False, effect_size: bool = True, homogeneity: bool = False,
               posthoc_factors: list[str] | None = None, posthoc_methods=(),
               level: float = 0.95, options=None) -> Section:
    """GLM Univariate with the default full-factorial Type III model (older signature);
    pass a :class:`glm_output.GLMOptions` as ``options`` for every other setting."""
    from rsstatchem.core.stats.glm_output import GLMOptions
    from rsstatchem.core.stats.glm_output import univariate as glm_univariate

    opts = options or GLMOptions(descriptives=descriptives, effect_size=effect_size,
                                 homogeneity=homogeneity,
                                 posthoc_factors=list(posthoc_factors or []),
                                 posthoc_methods=list(posthoc_methods), alpha=1 - level)
    return glm_univariate(ds, dependent, factors, covariates or [], opts).section


def _unrepr(text: str):
    try:
        return float(text)
    except ValueError:
        return text.strip("'\"")

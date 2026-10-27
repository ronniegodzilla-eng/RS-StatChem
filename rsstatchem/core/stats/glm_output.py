"""SPSS GLM output built on :mod:`glm_model`: the Univariate procedure (UNIANOVA) and the
tables / charts shared with Multivariate and Repeated Measures — Tests of Between-Subjects
Effects (any SS type, observed power, random-factor error terms), Expected Mean Squares,
Contrast Results (K Matrix), Estimated Marginal Means (pairwise comparisons, univariate tests,
simple main effects), Profile Plots (line/bar, error bars), Parameter Estimates (incl. robust
standard errors), Lack of Fit, heteroskedasticity tests, residual and spread-vs-level plots,
post hoc tests with homogeneous subsets, and saved variables (PRE_1, RES_1, COO_1, LEV_1…).
"""
from __future__ import annotations

import io
import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats import posthoc_ext
from rsstatchem.core.stats.common import (Group, fmt, fmt_df, fmt_int, fmt_p, levene_rows,
                                          numeric, pct, sup)
from rsstatchem.core.stats.glm_model import (CONTRAST_NAMES, GLMFit, TermSpec,
                                             contrast_matrix, ems_table, full_factorial,
                                             heteroskedasticity, lack_of_fit, observed_power_f,
                                             observed_power_t, parameter_estimates,
                                             saved_values)

ADJUST = {"lsd": "Least Significant Difference (equivalent to no adjustments)",
          "bonferroni": "Bonferroni", "sidak": "Sidak"}
SS_NAMES = {1: "Type I", 2: "Type II", 3: "Type III", 4: "Type IV"}
SAVE_LABELS = {
    "PRE": "Predicted Value", "WPR": "Weighted Predicted Value",
    "SEP": "Standard Error of Predicted Value", "RES": "Residual",
    "WRE": "Weighted Residual", "ZRE": "Standardized Residual",
    "SRE": "Studentized Residual", "DRE": "Deleted Residual", "COO": "Cook's Distance",
    "LEV": "Uncentered Leverage Value",
}
SAVE_ORDER = ("PRE", "WPR", "SEP", "RES", "WRE", "ZRE", "SRE", "DRE", "COO", "LEV")


@dataclass
class GLMOptions:
    """Everything the Univariate / Multivariate / Repeated Measures sub-dialogs set."""
    terms: list[TermSpec] | None = None     # None: full factorial
    intercept: bool = True
    ss_type: int = 3
    random: list[str] = field(default_factory=list)
    wls: str | None = None
    contrasts: dict[str, tuple[str, str]] = field(default_factory=dict)  # factor: (kind, ref)
    emm: list[str] = field(default_factory=list)       # "(OVERALL)", "A", "A*B"
    compare: bool = False                              # compare main effects
    simple: bool = False                               # compare simple main effects
    adjust: str = "lsd"
    plots: list[tuple] = field(default_factory=list)   # (horizontal, lines|None, plots|None)
    plot_type: str = "line"
    error_bars: str | None = None                      # None | "ci" | "se"
    se_mult: float = 2.0
    ref_line: bool = False
    y_zero: bool = False
    save: list[str] = field(default_factory=list)
    descriptives: bool = False
    effect_size: bool = False
    power: bool = False
    parameters: bool = False
    contrast_coef: bool = False
    homogeneity: bool = False
    spread_level: bool = False
    residual_plot: bool = False
    lack_of_fit: bool = False
    het: list[str] = field(default_factory=list)       # mbp, f, bp, white
    robust: str | None = None                          # HC0..HC4
    alpha: float = 0.05
    posthoc_factors: list[str] = field(default_factory=list)
    posthoc_methods: list[str] = field(default_factory=list)
    posthoc: posthoc_ext.PostHocOptions = field(default_factory=posthoc_ext.PostHocOptions)

    @property
    def level(self) -> float:
        return 1 - self.alpha


@dataclass
class GLMResult:
    section: Section
    saved: dict = field(default_factory=dict)


# ------------------------------------------------------------------ data
def make_frame(ds: Dataset, numeric_names: list[str], factor_names: list[str],
               extra_numeric: list[str] | None = None) -> pd.DataFrame:
    data = {}
    for n in numeric_names + (extra_numeric or []):
        data[n] = numeric(ds, n)
    for n in factor_names:
        data[n] = ds.analysis_series(n)
    return pd.DataFrame(data).dropna()


def design_syntax(opts: GLMOptions, factors: list[str], covariates: list[str]) -> str:
    terms = opts.terms if opts.terms is not None else full_factorial(factors, covariates)
    return " ".join(t.syntax for t in terms)


# ------------------------------------------------------------------ between-subjects table
def _power_cells(F: float, df1: float, df2: float, opts: GLMOptions) -> list[str]:
    if not opts.power:
        return []
    lam, pw = observed_power_f(F, df1, df2, opts.alpha)
    return [fmt(lam), fmt(pw)]


def between_effects_table(fit: GLMFit, dvars, opts: GLMOptions, wls_note: str | None = None,
                          ems: dict | None = None) -> Table:
    """Tests of Between-Subjects Effects for one or more dependent variables."""
    p = fit.Y.shape[1]
    sse = np.diag(fit.E)
    mse = sse / fit.ve if fit.ve > 0 else sse * np.nan
    ss_model, df_model, sst, df_t = fit.model_ss()
    multi = p > 1
    rows, notes = [], []
    eta = opts.effect_size
    power = opts.power

    def cells(source, j, ss, df, F=None, p_=None, err_ss=None):
        head = [source] + ([dvars[j].title] if multi else [])
        out = head + [fmt(ss), fmt_df(df), fmt(ss / df) if df else "",
                      fmt(F) if F is not None else "",
                      fmt_p(p_) if p_ is not None else ""]
        if eta:
            out.append(fmt(ss / (ss + err_ss)) if F is not None and err_ss is not None
                       and ss + err_ss > 0 else "")
        if power:
            out += _power_cells(F, df, fit.ve, opts) if F is not None else ["", ""]
        return out

    if ems is not None and ems["random"]:
        return _random_effects_table(fit, dvars[0], opts, ems, wls_note)

    letters = iter("abcdefghijklmnop")
    r2_letters = []
    for j in range(p):
        letter = next(letters)
        r2_letters.append(letter)
        F = (ss_model[j] / df_model) / mse[j] if df_model > 0 and mse[j] > 0 else float("nan")
        rows.append(cells("Corrected Model" if fit.intercept else "Model", j, ss_model[j],
                          df_model, F, stats.f.sf(F, df_model, fit.ve), sse[j]))
        rows[-1][2 if multi else 1] += sup(letter)
        if not multi:
            rows[-1][0] = ("Corrected Model" if fit.intercept else "Model")
        if multi and j:
            rows[-1][0] = ""
        r2 = ss_model[j] / sst[j] if sst[j] > 0 else float("nan")
        adj = 1 - (1 - r2) * df_t / fit.ve if fit.ve > 0 else float("nan")
        notes.append(f"{letter}. R Squared = {fmt(r2)} (Adjusted R Squared = {fmt(adj)})")
    for t in fit.terms:
        H = fit.hypothesis(t)
        df = fit.term_df(t)
        for j in range(p):
            ss = float(H[j, j])
            F = (ss / df) / mse[j] if df > 0 and mse[j] > 0 else float("nan")
            rows.append(cells(t.label if j == 0 else "", j, ss, df, F,
                              stats.f.sf(F, df, fit.ve) if df > 0 else float("nan"), sse[j]))
    totals = [("Error", sse, fit.ve)]
    if fit.intercept:
        totals += [("Total", (fit.Yw ** 2).sum(axis=0), fit.n),
                   ("Corrected Total", sst, fit.n - 1)]
    else:
        totals += [("Total", sst, fit.n)]
    for title, vals, df in totals:
        for j in range(p):
            r = cells(title if j == 0 else "", j, vals[j], df)
            if title != "Error":
                r[3 + multi] = ""
            rows.append(r)
    cols = ["Source"] + (["Dependent Variable"] if multi else []) + \
        [f"{SS_NAMES[opts.ss_type]} Sum of Squares", "df", "Mean Square", "F", "Sig."]
    if eta:
        cols.append("Partial Eta Squared")
    if power:
        cols += ["Noncent. Parameter", f"Observed Power{sup(next(letters))}"]
        pl = cols[-1][-1]
    foot = [] if multi else [f"Dependent Variable: {dvars[0].title}"]
    foot += notes
    if power:
        foot.append(f"{_letter_of(pl)}. Computed using alpha = {fmt(opts.alpha, 2)}")
    if wls_note:
        foot.append(wls_note)
    return Table("Tests of Between-Subjects Effects", cols, rows,
                 row_header_cols=1 + multi, footnotes=foot)


def _letter_of(marker: str) -> str:
    from rsstatchem.core.stats.common import _SUPERSCRIPT

    inv = {v: k for k, v in _SUPERSCRIPT.items()}
    return inv.get(marker, marker)


def _random_effects_table(fit: GLMFit, dvar, opts: GLMOptions, ems: dict,
                          wls_note: str | None) -> Table:
    rows, notes = [], []
    marks: dict[str, str] = {}
    letters = iter("abcdefghijklmnopqrstuvwxyz")
    labels = ems["labels"]
    ms, dfm = ems["ms"], ems["df"]

    def describe(weights: dict) -> str:
        parts = []
        for r, w in weights.items():
            name = "MS(Error)" if r == "error" else f"MS({labels[r]})"
            parts.append(name if abs(w - 1) < 1e-9 else f"{fmt(w)} {name}")
        return " + ".join(parts)

    for k in ems["numbers"]:
        t = next(tt for tt in fit.terms if fit.term_number(tt) == k)
        H = float(fit.hypothesis(t)[0, 0])
        df = dfm[k]
        weights, msd, df_d = ems["error"][k]
        F = (H / df) / msd if msd > 0 else float("nan")
        p_ = float(stats.f.sf(F, df, df_d)) if np.isfinite(df_d) else float("nan")
        hyp = [labels[k], "Hypothesis", fmt(H), fmt_df(df), fmt(H / df), fmt(F), fmt_p(p_)]
        if opts.effect_size:
            hyp.append(fmt(H / (H + msd * df_d)) if np.isfinite(df_d) else "")
        if opts.power:
            hyp += _power_cells(F, df, df_d, opts)
        rows.append(hyp)
        desc = describe(weights)
        if desc not in marks:
            marks[desc] = next(letters)
        err = ["", "Error", fmt(msd * df_d), fmt_df(round(df_d, 3)),
               fmt(msd) + sup(marks[desc]), "", ""]
        if opts.effect_size:
            err.append("")
        if opts.power:
            err += ["", ""]
        rows.append(err)
    for desc, letter in marks.items():
        notes.append(f"{letter}. {desc}")
    cols = ["Source", "", f"{SS_NAMES[opts.ss_type]} Sum of Squares", "df", "Mean Square",
            "F", "Sig."]
    if opts.effect_size:
        cols.append("Partial Eta Squared")
    if opts.power:
        letter = next(letters)
        cols += ["Noncent. Parameter", f"Observed Power{sup(letter)}"]
        notes.append(f"{letter}. Computed using alpha = {fmt(opts.alpha, 2)}")
    foot = [f"Dependent Variable: {dvar.title}"] + notes
    if wls_note:
        foot.append(wls_note)
    return Table("Tests of Between-Subjects Effects", cols, rows, row_header_cols=2,
                 footnotes=foot)


def ems_output(ems: dict, opts: GLMOptions, dvar) -> Table:
    comps = ems["random"] + ["error"]
    labels = ems["labels"]
    cols = ["Source"] + [("Variance Component", f"Var({labels[c]})" if c != "error"
                          else "Var(Error)") for c in comps] + \
        [("Variance Component", "Quadratic Term")]
    rows = []
    for k in ems["numbers"]:
        coef = ems["coef"][k]
        rows.append([labels[k]] + [fmt(coef.get(c, 0.0)) for c in comps] +
                    [", ".join(ems["quad"][k])])
    rows.append(["Error"] + [fmt(0.0)] * (len(comps) - 1) + [fmt(1.0), ""])
    return Table("Expected Mean Squares" + sup("a") + "," + sup("b"), cols, rows,
                 footnotes=[f"Dependent Variable: {dvar.title}",
                            "a. For each source, the expected mean square equals the sum of "
                            "the coefficients in the cells times the variance components, plus "
                            "a quadratic term involving effects in the Quadratic Term cell.",
                            f"b. Expected Mean Squares are based on the {SS_NAMES[opts.ss_type]}"
                            " Sums of Squares."])


# ------------------------------------------------------------------ L-matrix tests
def l_test(fit: GLMFit, L: np.ndarray, j: int | None = None, M: np.ndarray | None = None
           ) -> tuple[np.ndarray, int]:
    """Hypothesis SSCP for L B (M) = 0 and its df."""
    B = fit.B if M is None else fit.B @ M
    LB = L @ B
    V = L @ fit.XtX_inv @ L.T
    df = int(np.linalg.matrix_rank(V))
    H = LB.T @ np.linalg.pinv(V) @ LB
    return H, df


def univariate_f_rows(fit: GLMFit, L: np.ndarray, dvars, opts: GLMOptions,
                      prefix: list[str] | None = None) -> list[list[str]]:
    """Contrast / Error rows of an SPSS 'Univariate Tests' / 'Test Results' table."""
    H, df = l_test(fit, L)
    prefix = prefix or []
    multi = len(dvars) > 1
    rows = []
    for source in ("Contrast", "Error"):
        for j, v in enumerate(dvars):
            sse = float(fit.E[j, j])
            mse = sse / fit.ve if fit.ve > 0 else float("nan")
            head = list(prefix) if source == "Contrast" and j == 0 else [""] * len(prefix)
            head += [source if j == 0 else ""] + ([v.title] if multi else [])
            if source == "Contrast":
                ss = float(H[j, j])
                F = (ss / df) / mse if df > 0 and mse > 0 else float("nan")
                r = head + [fmt(ss), fmt_int(df), fmt(ss / df) if df else "", fmt(F),
                            fmt_p(stats.f.sf(F, df, fit.ve)) if df else ""]
                if opts.effect_size:
                    r.append(fmt(ss / (ss + sse)) if ss + sse > 0 else "")
                if opts.power:
                    r += _power_cells(F, df, fit.ve, opts)
            else:
                r = head + [fmt(sse), fmt_int(fit.ve), fmt(mse), "", ""]
                r += [""] * (opts.effect_size + 2 * opts.power)
            rows.append(r)
    if multi:  # SPSS orders by dependent variable then source
        order = []
        nd = len(dvars)
        for j in range(nd):
            order += [rows[j], rows[nd + j]]
        rows = order
    return rows


def f_columns(opts: GLMOptions, multi: bool, letter: str = "a") -> list:
    cols = ["Sum of Squares", "df", "Mean Square", "F", "Sig."]
    if opts.effect_size:
        cols.append("Partial Eta Squared")
    if opts.power:
        cols += ["Noncent. Parameter", f"Observed Power{sup(letter)}"]
    return (["Dependent Variable"] if multi else []) + cols


# ------------------------------------------------------------------ contrasts
def contrast_tables(fit: GLMFit, dvars, opts: GLMOptions, factor_names: list[str]
                    ) -> list[Table]:
    """Contrast Results (K Matrix) and Test Results for each factor with a contrast."""
    out = []
    p = len(dvars)
    multi = p > 1
    level = opts.level
    for fname, (kind, ref) in opts.contrasts.items():
        if kind in (None, "none") or fname not in factor_names:
            continue
        fi = factor_names.index(fname)
        f = fit.factors[fi]
        k = len(f.levels)
        if k < 2:
            continue
        C, labels = contrast_matrix(kind, k, ref)
        rows_l = [fit.average_row({fi: lv}) for lv in range(k)]
        Lmat = C @ np.array(rows_l)
        rows = []
        q = stats.t.ppf(0.5 + level / 2, fit.ve) if fit.ve > 0 else float("nan")
        for ci, (lrow, lab) in enumerate(zip(Lmat, labels)):
            stats_rows = [("Contrast Estimate", []), ("Hypothesized Value", []),
                          ("Difference (Estimate - Hypothesized)", []), ("Std. Error", []),
                          ("Sig.", []), (f"{pct(level)} Confidence Interval for Difference",
                                         "Lower Bound"),
                          (f"{pct(level)} Confidence Interval for Difference",
                           "Upper Bound")]
            vals = []
            for j in range(p):
                est, se = fit.estimate(lrow, np.eye(p)[j])
                t = est / se if se > 0 else float("nan")
                vals.append([fmt(est), "0", fmt(est), fmt(se),
                             fmt_p(2 * stats.t.sf(abs(t), fit.ve)) if se > 0 else "",
                             fmt(est - q * se), fmt(est + q * se)])
            for si, item in enumerate(stats_rows):
                name, sub = item
                head = [f"{CONTRAST_NAMES[kind]}" if ci == 0 and si == 0 else "",
                        lab if si == 0 else ""]
                if sub:
                    head += [name if sub == "Lower Bound" else "", sub]
                else:
                    head += [name, ""]
                rows.append(head + [vals[j][si] for j in range(p)])
        title_factor = f"{fname} {CONTRAST_NAMES[kind]} Contrast"
        foot = []
        if kind in ("simple", "deviation"):
            foot.append(f"a. Reference category = {k if ref == 'last' else 1}")
            title_factor += sup("a")
        cols = [title_factor, "", "", ""] + \
            ([("Dependent Variable", v.title) for v in dvars] if multi else
             [("Dependent Variable", dvars[0].title)])
        out.append(Table(f"Contrast Results (K Matrix) — {fname}", cols, rows,
                         row_header_cols=4, footnotes=foot))
        if multi:
            from rsstatchem.core.stats.manova import multivariate_tests, multivariate_table

            H, df = l_test(fit, Lmat)
            tests = multivariate_tests(H, fit.E, df, fit.ve)
            mt = multivariate_table(f"Multivariate Test Results — {fname}", [("", tests)],
                                    opts.effect_size, fit.design_text)
            out.append(mt)
        rows = univariate_f_rows(fit, Lmat, dvars, opts)
        cols = ["Source"] + f_columns(opts, multi)
        foot = [] if multi else [f"Dependent Variable: {dvars[0].title}"]
        if opts.power:
            foot.append(f"a. Computed using alpha = {fmt(opts.alpha, 2)}")
        out.append(Table(f"{'Univariate ' if multi else ''}Test Results — {fname}", cols, rows,
                         row_header_cols=1 + multi, footnotes=foot))
        if opts.contrast_coef:
            out.append(_k_matrix_coef(fit, Lmat, labels, fname))
    return out


def _k_matrix_coef(fit: GLMFit, L: np.ndarray, labels: list[str], fname: str) -> Table:
    names = [t.label for t in fit.terms for _ in t.idx]
    sub = []
    for t in fit.terms:
        sub += [str(i + 1) if len(t.idx) > 1 else "" for i in range(len(t.idx))]
    rows = [[n, s] + [fmt(L[r, c]) for r in range(L.shape[0])]
            for c, (n, s) in enumerate(zip(names, sub))]
    return Table(f"Contrast Coefficients (L' Matrix) — {fname}", ["Parameter", ""] +
                 [("Contrast", lab) for lab in labels], rows, row_header_cols=2,
                 footnotes=["The default display of this matrix is the transpose of the "
                            "corresponding L matrix (effect-coded parameters)."])


# ------------------------------------------------------------------ EM Means
def _adjusted(p: float, m: float, method: str) -> float:
    if method == "bonferroni":
        return min(1.0, p * m)
    if method == "sidak":
        return min(1.0, 1 - (1 - p) ** m)
    return p


def _adjusted_crit(level: float, m: float, df: float, method: str) -> float:
    alpha = 1 - level
    if method == "bonferroni":
        alpha /= m
    elif method == "sidak":
        alpha = 1 - (1 - alpha) ** (1 / m)
    return float(stats.t.ppf(1 - alpha / 2, df))


def emm_tables(fit: GLMFit, dvars, opts: GLMOptions, factor_names: list[str],
               covariate_note: list[str]) -> list[Table]:
    """Estimated Marginal Means: Grand Mean / Estimates / Pairwise Comparisons / Univariate
    Tests, including simple main effects for interactions (SPSS 28 COMPARE)."""
    out = []
    p = len(dvars)
    multi = p > 1
    level = opts.level
    q = float(stats.t.ppf(0.5 + level / 2, fit.ve)) if fit.ve > 0 else float("nan")
    ci_cols = ["Mean", "Std. Error", (f"{pct(level)} Confidence Interval", "Lower Bound"),
               (f"{pct(level)} Confidence Interval", "Upper Bound")]
    dep_col = ["Dependent Variable"] if multi else []

    def est_row(l, j):
        e, se = fit.estimate(l, np.eye(p)[j])
        return [fmt(e), fmt(se), fmt(e - q * se), fmt(e + q * se)]

    for spec in opts.emm:
        if spec == "(OVERALL)":
            l = fit.average_row({})
            rows = [([v.title] if multi else []) + est_row(l, j) for j, v in enumerate(dvars)]
            out.append(Table("Grand Mean", dep_col + ci_cols, rows, row_header_cols=int(multi),
                             footnotes=([] if multi else [f"Dependent Variable: "
                                                          f"{dvars[0].title}"])
                             + covariate_note))
            continue
        names = [s.strip() for s in spec.split("*")]
        if any(nm not in factor_names for nm in names):
            continue
        idx = [factor_names.index(nm) for nm in names]
        levels = [range(len(fit.factors[i].levels)) for i in idx]
        rows = []
        for j, v in enumerate(dvars):
            for combo in itertools.product(*levels):
                l = fit.average_row(dict(zip(idx, combo)))
                rows.append(([v.title] if multi else []) +
                            [fit.factors[i].labels[lv] for i, lv in zip(idx, combo)] +
                            est_row(l, j))
        from rsstatchem.core.stats.manova import blank_repeats

        rows = blank_repeats(rows, int(multi) + len(idx) - 1) if len(rows) > 1 else rows
        title = f"Estimates — {spec}" if len(idx) == 1 else spec.replace("*", " * ")
        out.append(Table(title, dep_col + names + ci_cols, rows,
                         row_header_cols=int(multi) + len(idx),
                         footnotes=([] if multi else [f"Dependent Variable: {dvars[0].title}"])
                         + covariate_note))
        if len(idx) == 1 and opts.compare:
            out += _comparisons(fit, dvars, opts, idx[0], [], {}, names[0])
        elif len(idx) > 1 and opts.simple:
            for pos, target in enumerate(idx):
                others = [i for i in idx if i != target]
                out += _comparisons(fit, dvars, opts, target, others, None,
                                    fit.factors[target].name)
    return out


def _comparisons(fit: GLMFit, dvars, opts: GLMOptions, target: int, others: list[int],
                 _unused, tname: str) -> list[Table]:
    """Pairwise comparisons (and univariate tests) of factor ``target`` within each level
    combination of ``others`` (empty: main-effect comparisons)."""
    p = len(dvars)
    multi = p > 1
    level = opts.level
    f = fit.factors[target]
    k = len(f.levels)
    m = k * (k - 1) / 2
    method = opts.adjust
    crit = _adjusted_crit(level, m, fit.ve, method) if m else float("nan")
    other_combos = list(itertools.product(*[range(len(fit.factors[i].levels))
                                            for i in others]))
    rows, flagged = [], False
    test_rows = []
    for j, v in enumerate(dvars):
        for oc in other_combos:
            fixed = dict(zip(others, oc))
            ls = [fit.average_row({**fixed, target: lv}) for lv in range(k)]
            olabels = [fit.factors[i].labels[lv] for i, lv in zip(others, oc)]
            first_i = True
            for a in range(k):
                for jj, b in enumerate([x for x in range(k) if x != a]):
                    diff, se = fit.estimate(ls[a] - ls[b], np.eye(p)[j])
                    t = diff / se if se > 0 else float("nan")
                    pv = _adjusted(float(2 * stats.t.sf(abs(t), fit.ve)), m, method) \
                        if se > 0 else float("nan")
                    star = "*" if pv < 1 - level else ""
                    flagged |= bool(star)
                    head = ([v.title] if multi else []) + olabels
                    rows.append(head + [f.labels[a] if jj == 0 else "", f.labels[b],
                                        fmt(diff) + star, fmt(se), fmt_p(pv),
                                        fmt(diff - crit * se), fmt(diff + crit * se)])
                    first_i = False
            del first_i
    from rsstatchem.core.stats.manova import blank_repeats

    nhead = int(multi) + len(others)
    if nhead:
        rows = blank_repeats(rows, nhead)
    other_names = [fit.factors[i].name for i in others]
    cols = (["Dependent Variable"] if multi else []) + other_names + \
        [f"(I) {tname}", f"(J) {tname}", "Mean Difference (I-J)", "Std. Error",
         f"Sig.{sup('b')}",
         (f"{pct(level)} Confidence Interval for Difference{sup('b')}", "Lower Bound"),
         (f"{pct(level)} Confidence Interval for Difference{sup('b')}", "Upper Bound")]
    foot = ([] if multi else [f"Dependent Variable: {dvars[0].title}"]) + \
        ["Based on estimated marginal means"]
    if flagged:
        foot.append(f"*. The mean difference is significant at the {1 - level:.2f} level.")
    foot.append(f"b. Adjustment for multiple comparisons: {ADJUST[method]}.")
    title = f"Pairwise Comparisons — {tname}" + (f" within {' * '.join(other_names)}"
                                                 if others else "")
    out = [Table(title, cols, rows, row_header_cols=nhead + 2, footnotes=foot)]
    # Univariate Tests
    for oc in other_combos:
        fixed = dict(zip(others, oc))
        ls = np.array([fit.average_row({**fixed, target: lv}) for lv in range(k)])
        L = ls[:-1] - ls[1:]
        prefix = [fit.factors[i].labels[lv] for i, lv in zip(others, oc)]
        test_rows += univariate_f_rows(fit, L, dvars, opts, prefix)
    cols = other_names + ["" if others else "Source"] + f_columns(opts, multi)
    if others:
        cols[len(other_names)] = ""
        note = (f"Each F tests the simple effects of {tname} within each level combination "
                "of the other effects shown. These tests are based on the linearly "
                "independent pairwise comparisons among the estimated marginal means.")
    else:
        note = (f"The F tests the effect of {tname}. This test is based on the linearly "
                "independent pairwise comparisons among the estimated marginal means.")
    foot = ([] if multi else [f"Dependent Variable: {dvars[0].title}"]) + [note]
    if opts.power:
        foot.append(f"a. Computed using alpha = {fmt(opts.alpha, 2)}")
    out.append(Table(f"Univariate Tests — {tname}" + (f" within {' * '.join(other_names)}"
                                                      if others else ""),
                     cols, test_rows, row_header_cols=len(others) + 1 + multi,
                     footnotes=foot))
    return out


# ------------------------------------------------------------------ profile plots
COLORS = ["#C62828", "#1565C0", "#2E7D32", "#6A1B9A", "#EF6C00", "#00838F", "#4E342E",
          "#AD1457"]


def profile_plot(title: str, xlabel: str, xlabels: list[str],
                 series: list[tuple[str | None, list[float], list[float]]],
                 legend_title: str | None, opts: GLMOptions, ylabel: str,
                 grand_mean: float | None = None, note: str | None = None) -> bytes:
    """One profile plot; series = (label, means, half-widths of the error bars)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(6.0, 4.2), dpi=120)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    x = np.arange(len(xlabels))
    ns = len(series)
    for si, (lab, ys, half) in enumerate(series):
        color = COLORS[si % len(COLORS)]
        yerr = half if opts.error_bars else None
        if opts.plot_type == "bar":
            width = 0.8 / ns
            pos = x - 0.4 + width * (si + 0.5)
            ax.bar(pos, ys, width * 0.95, color=color, label=lab, yerr=yerr, capsize=3,
                   edgecolor="#333", linewidth=0.5)
        else:
            ax.errorbar(x, ys, yerr=yerr, marker="o", color=color, label=lab, linewidth=1.6,
                        capsize=3 if yerr is not None else 0)
    if opts.ref_line and grand_mean is not None:
        ax.axhline(grand_mean, color="#555", linestyle="--", linewidth=1)
    if opts.y_zero:
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(0, lo), max(hi, 0))
    ax.set_xticks(x, xlabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if legend_title is not None:
        ax.legend(title=legend_title, fontsize=8, title_fontsize=9, frameon=False)
    texts = []
    if opts.error_bars == "ci":
        texts.append(f"Error bars: {pct(opts.level)} CI")
    elif opts.error_bars == "se":
        texts.append(f"Error bars: +/- {opts.se_mult:g} SE")
    if note:
        texts.append(note)
    if texts:
        ax.text(0.0, -0.2, "\n".join(texts), transform=ax.transAxes, fontsize=7, color="#555",
                va="top")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    return buf.getvalue()


def glm_profile_plots(fit: GLMFit, dvars, opts: GLMOptions, factor_names: list[str],
                      measure_title: str | None = None) -> list[tuple[str, bytes]]:
    out = []
    p = len(dvars)
    q = float(stats.t.ppf(0.5 + opts.level / 2, fit.ve)) if fit.ve > 0 else float("nan")
    note = ("Covariates appearing in the model are evaluated at the following values: " +
            ", ".join(f"{k} = {v:.4f}" for k, v in fit.cov_means.items())) \
        if fit.cov_means else None
    for spec in opts.plots:
        h, s, sp = (list(spec) + [None, None])[:3]
        if h not in factor_names or (s and s not in factor_names) or \
                (sp and sp not in factor_names):
            continue
        hi = factor_names.index(h)
        si = factor_names.index(s) if s else None
        pi = factor_names.index(sp) if sp else None
        for j in range(p):
            pages = range(len(fit.factors[pi].levels)) if pi is not None else [None]
            for pg in pages:
                series = []
                lines = range(len(fit.factors[si].levels)) if si is not None else [None]
                for ln in lines:
                    ys, half = [], []
                    for hv in range(len(fit.factors[hi].levels)):
                        fixed = {hi: hv}
                        if ln is not None:
                            fixed[si] = ln
                        if pg is not None:
                            fixed[pi] = pg
                        e, se = fit.estimate(fit.average_row(fixed), np.eye(p)[j])
                        ys.append(e)
                        half.append(q * se if opts.error_bars == "ci" else opts.se_mult * se)
                    series.append((fit.factors[si].labels[ln] if ln is not None else None,
                                   ys, half))
                name = measure_title or dvars[j].title
                title = f"Estimated Marginal Means of {name}"
                if pg is not None:
                    title += f"\nat {sp} = {fit.factors[pi].labels[pg]}"
                gm = fit.estimate(fit.average_row({}), np.eye(p)[j])[0]
                png = profile_plot(title, h, list(fit.factors[hi].labels), series,
                                   s if si is not None else None, opts,
                                   "Estimated Marginal Means", gm, note)
                caption = f"Profile Plot: {h}" + (f" * {s}" if s else "") + \
                    (f" * {sp}" if sp else "") + (f" ({dvars[j].title})" if p > 1 else "")
                out.append((caption, png))
    return out


# ------------------------------------------------------------------ options output
def parameter_table(fit: GLMFit, dvars, opts: GLMOptions) -> list[Table]:
    out = []
    level = opts.level
    multi = len(dvars) > 1
    for robust in ([None] + ([opts.robust] if opts.robust else [])) if opts.parameters \
            else ([opts.robust] if opts.robust else []):
        rows = []
        any_red = False
        df = fit.n - int(np.sum(~fit.redundant))
        q = stats.t.ppf(0.5 + level / 2, df) if df > 0 else float("nan")
        for j, v in enumerate(dvars):
            for i, (label, b, se, t, p, red) in enumerate(parameter_estimates(fit, j, robust)):
                head = ([v.title if i == 0 else ""] if multi else []) + [label]
                if red:
                    any_red = True
                    rows.append(head + ["0" + sup("a")] + [""] * (5 + (robust is None) *
                                                                (opts.effect_size +
                                                                 2 * opts.power)))
                    continue
                r = head + [fmt(b), fmt(se), fmt(t), fmt_p(p), fmt(b - q * se),
                            fmt(b + q * se)]
                if robust is None:
                    if opts.effect_size:
                        r.append(fmt(t * t / (t * t + df)) if np.isfinite(t) else "")
                    if opts.power:
                        nc, pw = observed_power_t(t, df, opts.alpha)
                        r += [fmt(nc), fmt(pw)]
                rows.append(r)
        se_name = "Std. Error" if robust is None else f"Robust Std. Error{sup('b')}"
        cols = (["Dependent Variable"] if multi else []) + \
            ["Parameter", "B", se_name, "t", "Sig.",
             (f"{pct(level)} Confidence Interval", "Lower Bound"),
             (f"{pct(level)} Confidence Interval", "Upper Bound")]
        foot = [] if multi else [f"Dependent Variable: {dvars[0].title}"]
        if any_red:
            foot.append("a. This parameter is set to zero because it is redundant.")
        if robust is None:
            if opts.effect_size:
                cols.append("Partial Eta Squared")
            if opts.power:
                cols += ["Noncent. Parameter", f"Observed Power{sup('b')}"]
                foot.append(f"b. Computed using alpha = {fmt(opts.alpha, 2)}")
            title = "Parameter Estimates"
        else:
            foot.append(f"b. {robust} method")
            title = "Parameter Estimates with Robust Standard Errors"
        out.append(Table(title, cols, rows, row_header_cols=1 + multi, footnotes=foot))
    return out


def lack_of_fit_table(fit: GLMFit, dvars, opts: GLMOptions) -> Table | None:
    rows = []
    multi = len(dvars) > 1
    for j, v in enumerate(dvars):
        res = lack_of_fit(fit, j)
        if res is None:
            return None
        ss_l, df_l, ss_p, df_p = res
        F = (ss_l / df_l) / (ss_p / df_p) if ss_p > 0 else float("nan")
        r = ([v.title] if multi else []) + ["Lack of Fit", fmt(ss_l), fmt_int(df_l),
                                            fmt(ss_l / df_l), fmt(F),
                                            fmt_p(stats.f.sf(F, df_l, df_p))]
        if opts.effect_size:
            r.append(fmt(ss_l / (ss_l + ss_p)) if ss_l + ss_p > 0 else "")
        if opts.power:
            r += _power_cells(F, df_l, df_p, opts)
        rows.append(r)
        rows.append(([""] if multi else []) + ["Pure Error", fmt(ss_p), fmt_int(df_p),
                                               fmt(ss_p / df_p), "", ""] +
                    [""] * (opts.effect_size + 2 * opts.power))
    cols = (["Dependent Variable"] if multi else []) + ["Source"] + f_columns(opts, False)
    foot = [] if multi else [f"Dependent Variable: {dvars[0].title}"]
    if opts.power:
        foot.append(f"a. Computed using alpha = {fmt(opts.alpha, 2)}")
    return Table("Lack of Fit Tests", cols, rows, row_header_cols=1 + multi, footnotes=foot)


def het_tables(fit: GLMFit, dvar, opts: GLMOptions, design: str) -> list[Table]:
    res = heteroskedasticity(fit, 0)
    out = []
    common_notes = [f"a. Dependent variable: {dvar.title}",
                    "b. Tests the null hypothesis that the variance of the errors does not "
                    "depend on the values of the independent variables.",
                    f"c. Predicted values from design: {design}"]
    for key, title in (("mbp", "Modified Breusch-Pagan Test for Heteroskedasticity"),
                       ("bp", "Breusch-Pagan Test for Heteroskedasticity")):
        if key in opts.het:
            stat, df, p = res[key]
            out.append(Table(title + sup("a") + "," + sup("b") + "," + sup("c"),
                             ["Chi-Square", "df", "Sig."],
                             [[fmt(stat), fmt_int(df), fmt_p(p)]], row_header_cols=0,
                             footnotes=common_notes))
    if "f" in opts.het:
        F, d1, d2, p = res["f"]
        out.append(Table("F Test for Heteroskedasticity" + sup("a") + "," + sup("b") + "," +
                         sup("c"), ["F", "df1", "df2", "Sig."],
                         [[fmt(F), fmt_int(d1), fmt_int(d2), fmt_p(p)]], row_header_cols=0,
                         footnotes=common_notes))
    if "white" in opts.het:
        stat, df, p = res["white"]
        out.append(Table("White Test for Heteroskedasticity" + sup("a") + "," + sup("b") + "," +
                         sup("c"), ["Chi-Square", "df", "Sig."],
                         [[fmt(stat), fmt_int(df), fmt_p(p)]], row_header_cols=0,
                         footnotes=common_notes[:2] + [f"c. Design: {design}"]))
    return out


def residual_plot(fit: GLMFit, j: int, title: str) -> bytes:
    """SPSS GLM residual plot: matrix of Observed, Predicted and Std. Residual."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    sv = saved_values(fit, j)
    data = {"Observed": fit.Y[:, j], "Predicted": sv["PRE"], "Std. Residual": sv["ZRE"]}
    names = list(data)
    fig = Figure(figsize=(6.0, 5.6), dpi=120)
    FigureCanvasAgg(fig)
    for r, a in enumerate(names):
        for c, b in enumerate(names):
            ax = fig.add_subplot(3, 3, r * 3 + c + 1)
            if r == c:
                ax.text(0.5, 0.5, a, ha="center", va="center", fontsize=9, fontweight="bold")
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            ax.scatter(data[b], data[a], s=8, color="#C62828", alpha=0.75)
            ax.tick_params(labelsize=6)
    fig.suptitle(f"Dependent Variable: {title}", fontsize=10, fontweight="bold")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    return buf.getvalue()


def spread_level_plots(cells: list[np.ndarray], title: str) -> bytes:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    means = [c.mean() for c in cells if len(c) > 1]
    sds = [c.std(ddof=1) for c in cells if len(c) > 1]
    fig = Figure(figsize=(6.4, 3.0), dpi=120)
    FigureCanvasAgg(fig)
    for k, (vals, lab) in enumerate(((sds, "Spread (Standard Deviation)"),
                                     ([s * s for s in sds], "Spread (Variance)"))):
        ax = fig.add_subplot(1, 2, k + 1)
        ax.scatter(means, vals, color="#C62828", s=18)
        ax.set_xlabel("Level (Mean)", fontsize=8)
        ax.set_ylabel(lab, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(f"Spread vs. Level Plot of {title}", fontsize=8, fontweight="bold")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    return buf.getvalue()


def saved_variables(fit: GLMFit, keys: list[str], index: pd.Index, full_index: pd.Index,
                    dep_names: list[str]) -> dict:
    out = {}
    weighted = not np.allclose(fit.w, 1)
    for j, dep in enumerate(dep_names):
        vals = saved_values(fit, j)
        vals["WPR"] = vals["PRE"]
        for key in SAVE_ORDER:
            if key not in keys or (key in ("WPR", "WRE") and not weighted):
                continue
            label = f"{SAVE_LABELS[key]} for {dep}"
            s = pd.Series(np.nan, index=full_index)
            s.loc[index] = vals[key]
            out[key if len(dep_names) == 1 else f"{key}_{dep}"] = (label, s)
    return out


# ------------------------------------------------------------------ post hoc
def posthoc_tables(groups_by_dep: list[tuple[str, list[Group]]], factor_title: str,
                   methods: list[str], mse_by_dep: list[float], df_error: float,
                   opts: posthoc_ext.PostHocOptions, glm: bool = True,
                   anova_f: list[float] | None = None, decimals: int = 2) -> list[Table]:
    from rsstatchem.core.stats.anova import multiple_comparisons_table

    out = []
    pair = [m for m in posthoc_ext.PAIRWISE if m in methods]
    sub = [m for m in posthoc_ext.SUBSETS if m in methods]
    if pair:
        mc = []
        for (dep, groups), mse in zip(groups_by_dep, mse_by_dep):
            for meth in pair:
                mc.append((dep, groups, meth,
                           posthoc_ext.pairwise(groups, meth, mse, df_error, opts)))
        tab = multiple_comparisons_table(mc, opts.level, factor_title, opts)
        if glm:
            tab.title = f"Multiple Comparisons — {factor_title}"
            tab.footnotes.insert(0, "Based on observed means.")
            tab.footnotes.insert(1, "The error term is Mean Square(Error) = " + ", ".join(
                fmt(m) for m in mse_by_dep) + ".")
        out.append(tab)
    if sub:
        for i, ((dep, groups), mse) in enumerate(zip(groups_by_dep, mse_by_dep)):
            note = ("Based on observed means. The error term is Mean Square(Error) = "
                    f"{fmt(mse)}.") if glm else None
            t = posthoc_ext.subsets_table(dep, factor_title, groups, sub, mse, df_error, opts,
                                          anova_f[i] if anova_f else None, decimals, note)
            if t is not None:
                t.title = f"Homogeneous Subsets — {factor_title}: {dep}"
                out.append(t)
    return out


# ------------------------------------------------------------------ UNIANOVA
def univariate(ds: Dataset, dependent: str, factors: list[str], covariates: list[str],
               opts: GLMOptions) -> GLMResult:
    """SPSS Analyze > General Linear Model > Univariate with all sub-dialog options."""
    from rsstatchem.core.stats.manova import (between_factors_table, make_factor,
                                              nested_descriptive_rows)

    covariates = covariates or []
    random = [r for r in opts.random if r not in factors]
    allf = list(factors) + random
    dvar = ds.var(dependent)
    section = Section("Univariate Analysis of Variance")
    section.log(_syntax(dependent, factors, random, covariates, opts))
    extra = [opts.wls] if opts.wls else []
    frame = make_frame(ds, [dependent] + covariates, allf, extra)
    if opts.wls:
        frame = frame[frame[opts.wls] > 0]
    if len(frame) < 2 or (not allf and not covariates and opts.terms is None):
        section.note("Warning", "Not enough valid cases for the model.", style="warning")
        return GLMResult(section)
    fobjs = [make_factor(ds, f, frame[f]) for f in allf]
    names = [f.name for f in fobjs]
    covs = {ds.var(c).name: frame[c].to_numpy(float) for c in covariates}
    terms = opts.terms
    if terms is None:
        terms = full_factorial(names, list(covs))
    w = frame[opts.wls].to_numpy(float) if opts.wls else None
    y = frame[dependent].to_numpy(float)
    try:
        fit = GLMFit(y, fobjs, covs, terms, opts.intercept, opts.ss_type, w)
    except (ValueError, KeyError) as exc:
        section.note("Warning", f"The model cannot be fitted: {exc}", style="warning")
        return GLMResult(section)
    if fit.ve < 1:
        section.note("Warning", "There are no error degrees of freedom: the model is "
                     "saturated.", style="warning")
        return GLMResult(section)
    design = fit.design_text
    wls_note = f"Weighted Least Squares Regression - Weighted by {ds.var(opts.wls).title}" \
        if opts.wls else None
    if fit.deficient and opts.ss_type in (3, 4):
        section.note("Warning", "Some cells of the design are empty. Type III/IV results use "
                     "a generalized inverse of the effect-coded design and may differ from "
                     "SPSS.", style="warning")

    if fobjs:
        section.table(between_factors_table(fobjs))
    if opts.descriptives and fobjs:
        drows = nested_descriptive_rows(fobjs, y, dvar.decimals)
        section.table(Table("Descriptive Statistics", names + ["Mean", "Std. Deviation", "N"],
                            drows, row_header_cols=len(fobjs),
                            footnotes=[f"Dependent Variable: {dvar.title}"]))
    from rsstatchem.core.stats.manova import cell_index

    cells = cell_index(fobjs) if fobjs else np.zeros(len(y), int)
    if opts.homogeneity and fobjs:
        source = fit.resid[:, 0] if covs else y
        samples = [source[cells == c] for c in np.unique(cells)]
        lrows = [[basis, fmt(F), fmt_df(df1), fmt_df(df2), fmt_p(p)]
                 for basis, F, df1, df2, p in levene_rows(samples)]
        if not lrows or all(r[1] == "" for r in lrows):
            section.note("Warning", "Levene's test cannot be computed: the cells need more "
                         "than one case each.", style="warning")
        else:
            section.table(Table("Levene's Test of Equality of Error Variances" + sup("a") +
                                "," + sup("b"),
                                ["", "Levene Statistic", "df1", "df2", "Sig."], lrows,
                                footnotes=["Tests the null hypothesis that the error variance "
                                           "of the dependent variable is equal across groups.",
                                           f"a. Dependent variable: {dvar.title}",
                                           f"b. Design: {design}"]))
    for t in het_tables(fit, dvar, opts, design) if opts.het else []:
        section.table(t)
    ems = ems_table(fit, random) if random else None
    section.table(between_effects_table(fit, [dvar], opts, wls_note, ems))
    if ems is not None:
        section.table(ems_output(ems, opts, dvar))
    for t in parameter_table(fit, [dvar], opts):
        section.table(t)
    if opts.lack_of_fit:
        t = lack_of_fit_table(fit, [dvar], opts)
        if t is not None:
            section.table(t)
        else:
            section.note("", "Lack of fit cannot be computed: there are no replicated "
                         "combinations of the predictors (no pure error).")
    for t in contrast_tables(fit, [dvar], opts, names):
        section.table(t)
    cov_note = []
    if fit.cov_means:
        vals = ", ".join(f"{k} = {fmt(v, 4)}" for k, v in fit.cov_means.items())
        cov_note = [f"Covariates appearing in the model are evaluated at the following values: "
                    f"{vals}."]
    for t in emm_tables(fit, [dvar], opts, names, cov_note):
        section.table(t)
    # post hoc
    ph = [f for f in opts.posthoc_factors if f in factors]
    if ph and opts.posthoc_methods:
        if covs:
            section.note("Warning", "Post hoc tests are not performed for the factors because "
                         "covariates are in the model.", style="warning")
        else:
            mse = float(fit.E[0, 0]) / fit.ve
            for fname in ph:
                f = fobjs[names.index(ds.var(fname).name)]
                if len(f.levels) < 3:
                    section.note("Warning", f"Post hoc tests are not performed for {f.name} "
                                 "because there are fewer than three groups.", style="warning")
                    continue
                groups = [Group(lv, lab, y[f.codes == i])
                          for i, (lv, lab) in enumerate(zip(f.levels, f.labels))]
                popts = posthoc_ext.PostHocOptions(opts.level, opts.posthoc.control,
                                                   opts.posthoc.dunnett_test,
                                                   opts.posthoc.k_ratio)
                term = next((t for t in fit.terms if t.spec == TermSpec((f.name,))), None)
                f_main = None
                if term is not None:
                    H = float(fit.hypothesis(term)[0, 0])
                    df = fit.term_df(term)
                    f_main = (H / df) / mse if df else None
                for t in posthoc_tables([(dvar.title, groups)], f.title, opts.posthoc_methods,
                                        [mse], fit.ve, popts, True, [f_main],
                                        dvar.decimals):
                    section.table(t)
    if opts.spread_level and fobjs:
        section.image(f"Spread vs. Level Plot: {dvar.title}",
                      spread_level_plots([y[cells == c] for c in np.unique(cells)],
                                         dvar.title))
    for caption, png in glm_profile_plots(fit, [dvar], opts, names):
        section.image(caption, png)
    if opts.residual_plot:
        section.image(f"Residual Plot: {dvar.title}", residual_plot(fit, 0, dvar.title))
    saved = saved_variables(fit, opts.save, frame.index, ds.df.index, [dvar.name]) \
        if opts.save else {}
    return GLMResult(section, saved)


def _syntax(dependent: str, factors: list[str], random: list[str], covariates: list[str],
            opts: GLMOptions) -> str:
    log = f"UNIANOVA {dependent}"
    if factors or random:
        log += f" BY {' '.join(factors + random)}"
    if covariates:
        log += f" WITH {' '.join(covariates)}"
    if random:
        log += f"\n  /RANDOM={' '.join(random)}"
    if opts.wls:
        log += f"\n  /REGWGT={opts.wls}"
    for f, (kind, ref) in opts.contrasts.items():
        if kind and kind != "none":
            extra = f"({1 if ref == 'first' else ''})" if kind in ("simple", "deviation") \
                and ref == "first" else ""
            log += f"\n  /CONTRAST({f})={kind.capitalize()}{extra}"
    log += f"\n  /METHOD=SSTYPE({opts.ss_type})"
    log += f"\n  /INTERCEPT={'INCLUDE' if opts.intercept else 'EXCLUDE'}"
    if opts.save:
        log += "\n  /SAVE=" + " ".join(opts.save)
    for spec in opts.plots:
        h, s, sp = (list(spec) + [None, None])[:3]
        log += f"\n  /PLOT=PROFILE({'*'.join(x for x in (h, s, sp) if x)}) TYPE=" + \
            ("BAR" if opts.plot_type == "bar" else "LINE") + \
            (" ERRORBAR=CI" if opts.error_bars == "ci" else
             f" ERRORBAR=SE({opts.se_mult:g})" if opts.error_bars == "se" else
             " ERRORBAR=NO") + (" MEANREFERENCE=YES" if opts.ref_line else "")
    if opts.posthoc_factors and opts.posthoc_methods:
        log += (f"\n  /POSTHOC={' '.join(opts.posthoc_factors)}"
                f"({' '.join(m.upper() for m in opts.posthoc_methods)})")
    for e in opts.emm:
        log += f"\n  /EMMEANS=TABLES({'OVERALL' if e == '(OVERALL)' else e})"
        if e != "(OVERALL)":
            parts = e.split("*")
            if len(parts) == 1 and opts.compare:
                log += f" COMPARE ADJ({opts.adjust.upper()})"
            elif len(parts) > 1 and opts.simple:
                log += f" COMPARE({parts[0]}) ADJ({opts.adjust.upper()})"
    prints = [k for k, on in (("DESCRIPTIVE", opts.descriptives), ("ETASQ", opts.effect_size),
                              ("OPOWER", opts.power), ("PARAMETER", opts.parameters),
                              ("HOMOGENEITY", opts.homogeneity), ("LOF", opts.lack_of_fit))
              if on]
    if opts.robust:
        prints.append(f"ROBUST({opts.robust})")
    prints += [k.upper() for k in opts.het]
    if prints:
        log += "\n  /PRINT=" + " ".join(prints)
    plots = [k for k, on in (("SPREADLEVEL", opts.spread_level), ("RESIDUALS", opts.residual_plot))
             if on]
    if plots:
        log += "\n  /PLOT=" + " ".join(plots)
    log += f"\n  /CRITERIA=ALPHA({opts.alpha:.2f})"
    terms = opts.terms if opts.terms is not None else full_factorial(factors + random,
                                                                      covariates)
    log += "\n  /DESIGN=" + " ".join(t.syntax for t in terms) + "."
    return log


__all__ = ["GLMOptions", "GLMResult", "univariate", "between_effects_table", "emm_tables",
           "contrast_tables", "glm_profile_plots", "parameter_table", "lack_of_fit_table",
           "posthoc_tables", "saved_variables", "SAVE_LABELS", "SAVE_ORDER", "ADJUST"]

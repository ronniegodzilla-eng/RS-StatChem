"""Analyze > General Linear Model > Multivariate (MANOVA), mirroring SPSS GLM output.

Also holds the multivariate linear-model engine shared with Repeated Measures
(:mod:`repeated`): an effect-coded full-factorial design with covariate main effects, Type III
hypothesis SSCP matrices, the four multivariate tests with SPSS's F approximations
(Pillai, Wilks/Rao, Hotelling-Lawley/Pillai-Samson, Roy upper bound), Box's M and estimated
marginal means (unweighted cell averages with covariates at their means).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.anova import (POSTHOC, POSTHOC_ORDER, multiple_comparisons_table,
                                         posthoc)
from rsstatchem.core.stats.common import (Group, fmt, fmt_df, fmt_int, fmt_p, levene_rows,
                                          numeric, pct, sup, value_text)
from rsstatchem.core.stats.glm_model import (GLMFit, Term, TermSpec, full_factorial,
                                             main_effects)

ADJUST = {"lsd": "Least Significant Difference (equivalent to no adjustments)",
          "bonferroni": "Bonferroni", "sidak": "Sidak"}
MV_NAMES = ("Pillai's Trace", "Wilks' Lambda", "Hotelling's Trace", "Roy's Largest Root")


# ------------------------------------------------------------------ design + fit
@dataclass
class Factor:
    """A categorical between-subjects factor: variable name, title, sorted codes, labels."""
    name: str
    title: str
    levels: list
    labels: list[str]
    codes: np.ndarray = field(default=None, repr=False)  # level index per case


def _level_key(v):
    return (isinstance(v, str), v)


def make_factor(ds: Dataset, name: str, series: pd.Series) -> Factor:
    var = ds.var(name)
    levels = sorted(series.unique(), key=_level_key)
    index = {lv: i for i, lv in enumerate(levels)}
    return Factor(var.name, var.title, levels, [value_text(var, lv) for lv in levels],
                  series.map(index).to_numpy(int))


class LinearModel(GLMFit):
    """Multivariate (W)LS ``Y = X B + E`` with SPSS's coding. By default the full-factorial
    design with covariate main effects and Type III SS; ``terms`` / ``intercept`` /
    ``ss_type`` / ``weights`` give a custom GLM model (see :mod:`glm_model`)."""

    def __init__(self, Y: np.ndarray, factors: list[Factor], covariates: dict[str, np.ndarray],
                 main_effects_only: bool = False, terms: list[TermSpec] | None = None,
                 intercept: bool = True, ss_type: int = 3, weights: np.ndarray | None = None):
        if terms is None:
            names = [f.name for f in factors]
            terms = main_effects(names, list(covariates)) if main_effects_only else \
                full_factorial(names, list(covariates))
        super().__init__(Y, factors, covariates, terms, intercept, ss_type, weights)
        self.main_effects_only = main_effects_only


# ------------------------------------------------------------------ multivariate tests
@dataclass
class MVTest:
    name: str
    value: float
    F: float
    df1: float
    df2: float
    p: float
    eta: float
    exact: bool = False
    upper_bound: bool = False


def multivariate_tests(H: np.ndarray, E: np.ndarray, q: int, ve: int) -> list[MVTest]:
    """Pillai, Wilks, Hotelling-Lawley and Roy with the F approximations SPSS GLM uses."""
    p = H.shape[0]
    nan = float("nan")
    try:
        eig = np.linalg.eigvals(np.linalg.solve(E, H)).real
    except np.linalg.LinAlgError:
        eig = np.full(p, nan)
    eig = np.clip(np.sort(eig)[::-1], 0, None)
    s = min(p, q)
    m = (abs(p - q) - 1) / 2
    nn = (ve - p - 1) / 2
    out = []
    lam = eig[:s]

    def fp(F, d1, d2):
        return float(stats.f.sf(F, d1, d2)) if d2 > 0 and np.isfinite(F) else nan

    # Pillai
    V = float(np.sum(lam / (1 + lam)))
    d1, d2 = s * (2 * m + s + 1), s * (2 * nn + s + 1)
    F = (2 * nn + s + 1) / (2 * m + s + 1) * V / (s - V) if s - V > 0 else nan
    out.append(MVTest(MV_NAMES[0], V, F, d1, d2, fp(F, d1, d2), V / s, s == 1))
    # Wilks (Rao)
    W = float(np.prod(1 / (1 + lam)))
    t = math.sqrt((p * p * q * q - 4) / (p * p + q * q - 5)) if p * p + q * q - 5 > 0 else 1.0
    d1 = p * q
    w = ve - (p - q + 1) / 2
    d2 = w * t - (p * q - 2) / 2
    root = W ** (1 / t) if W > 0 else 0.0
    F = (1 - root) / root * d2 / d1 if root > 0 else nan
    out.append(MVTest(MV_NAMES[1], W, F, d1, d2, fp(F, d1, d2), 1 - W ** (1 / s),
                      min(p, q) <= 2))
    # Hotelling-Lawley (Pillai-Samson approximation)
    T = float(np.sum(lam))
    d1, d2 = s * (2 * m + s + 1), 2 * (s * nn + 1)
    F = d2 * T / (s * s * (2 * m + s + 1)) if d2 > 0 else nan
    out.append(MVTest(MV_NAMES[2], T, F, d1, d2, fp(F, d1, d2), (T / s) / (T / s + 1), s == 1))
    # Roy
    theta = float(lam[0]) if len(lam) else nan
    r = max(p, q)
    d1, d2 = r, ve - r + q
    F = theta * d2 / d1
    out.append(MVTest(MV_NAMES[3], theta, F, d1, d2, fp(F, d1, d2), theta / (1 + theta),
                      s == 1, s > 1))
    return out


def multivariate_table(title: str, blocks: list[tuple[str, list[MVTest]]], effect_size: bool,
                       design: str, extra_notes: list[str] | None = None,
                       first_col: str = "Effect") -> Table:
    rows = []
    exact = bound = False
    for effect, tests in blocks:
        for i, t in enumerate(tests):
            mark = sup("a") if t.exact else sup("b") if t.upper_bound else ""
            exact |= t.exact
            bound |= t.upper_bound
            cells = [effect if i == 0 else "", t.name, fmt(t.value), fmt(t.F) + mark,
                     fmt_df(round(t.df1, 3)), fmt_df(round(t.df2, 3)), fmt_p(t.p)]
            if effect_size:
                cells.append(fmt(t.eta))
            rows.append(cells)
    columns = [first_col, "", "Value", "F", "Hypothesis df", "Error df", "Sig."]
    if effect_size:
        columns.append("Partial Eta Squared")
    notes = []
    notes.append("a. Exact statistic" if exact else "")
    notes.append("b. The statistic is an upper bound on F that yields a lower bound on the "
                 "significance level." if bound else "")
    notes = [n for n in notes if n]
    notes.append(f"Design: {design}")
    notes += extra_notes or []
    return Table(title, columns, rows, row_header_cols=2, footnotes=notes)


# ------------------------------------------------------------------ Box's M
def box_m(samples: list[np.ndarray]) -> tuple[float, float, float, float, float] | None:
    """Box's M with SPSS's F approximation. ``samples``: (n_i, p) arrays per cell.
    Cells whose covariance matrix is singular are left out (as SPSS does)."""
    samples = [np.asarray(s, float) for s in samples]
    p = samples[0].shape[1] if samples else 0
    good = []
    for s in samples:
        if s.shape[0] > p:
            c = np.cov(s, rowvar=False).reshape(p, p)
            if np.linalg.matrix_rank(c) == p:
                good.append((s.shape[0], c))
    g = len(good)
    if g < 2:
        return None
    ns = np.array([n for n, _ in good], float)
    N = ns.sum()
    pooled = sum((n - 1) * c for n, c in good) / (N - g)
    M = (N - g) * math.log(np.linalg.det(pooled)) - \
        sum((n - 1) * math.log(np.linalg.det(c)) for n, c in good)
    c1 = (np.sum(1 / (ns - 1)) - 1 / (N - g)) * (2 * p * p + 3 * p - 1) / (6 * (p + 1) * (g - 1))
    c2 = (np.sum(1 / (ns - 1) ** 2) - 1 / (N - g) ** 2) * (p - 1) * (p + 2) / (6 * (g - 1))
    df1 = (g - 1) * p * (p + 1) / 2
    if c2 - c1 * c1 > 0:
        df2 = (df1 + 2) / (c2 - c1 * c1)
        b = df1 / (1 - c1 - df1 / df2)
        F = M / b
    else:
        df2 = (df1 + 2) / (c1 * c1 - c2)
        b = df2 / (1 - c1 + 2 / df2)
        F = df2 * M / (df1 * (b - M))
    return float(M), float(F), float(df1), float(df2), float(stats.f.sf(F, df1, df2))


def box_table(samples: list[np.ndarray], design: str, extra: str = "") -> Table | None:
    res = box_m(samples)
    if res is None:
        return None
    M, F, df1, df2, p = res
    return Table("Box's Test of Equality of Covariance Matricesᵃ", ["", ""],
                 [["Box's M", fmt(M)], ["F", fmt(F)], ["df1", fmt_df(df1)],
                  ["df2", fmt(df2)], ["Sig.", fmt_p(p)]],
                 footnotes=["Tests the null hypothesis that the observed covariance matrices "
                            f"of the dependent variables are equal across groups.{extra}",
                            f"a. Design: {design}"])


# ------------------------------------------------------------------ shared tables
def cell_index(factors: list[Factor]) -> np.ndarray:
    """One integer per case identifying its between-subjects cell."""
    if not factors:
        return np.zeros(0, int)
    key = np.zeros(len(factors[0].codes), int)
    for f in factors:
        key = key * len(f.levels) + f.codes
    return key


def between_factors_table(factors: list[Factor]) -> Table:
    rows = []
    has_labels = any(lab != value_text_plain(lv) for f in factors
                     for lv, lab in zip(f.levels, f.labels))
    for f in factors:
        for i, (lv, lab) in enumerate(zip(f.levels, f.labels)):
            code = value_text_plain(lv)
            cells = [f.name if i == 0 else "", code]
            if has_labels:
                cells.append(lab if lab != code else "")
            cells.append(fmt_int(int(np.sum(f.codes == i))))
            rows.append(cells)
    cols = ["", ""] + (["Value Label"] if has_labels else []) + ["N"]
    return Table("Between-Subjects Factors", cols, rows, row_header_cols=2)


def value_text_plain(v) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def nested_descriptive_rows(factors: list[Factor], y: np.ndarray, decimals: int,
                            mask: np.ndarray | None = None, depth: int = 0) -> list[list[str]]:
    """SPSS Descriptive Statistics rows: factor levels nested with a Total at every level."""
    if mask is None:
        mask = np.ones(len(y), bool)
    if depth == len(factors):
        x = y[mask]
        sd = x.std(ddof=1) if len(x) > 1 else float("nan")
        return [[fmt(x.mean() if len(x) else float("nan"), decimals + 2), fmt(sd, decimals + 3),
                 fmt_int(len(x))]]
    f = factors[depth]
    out = []
    for lv in list(range(len(f.levels))) + [None]:
        sub = mask & (f.codes == lv) if lv is not None else mask
        if lv is not None and not sub.any():
            continue
        inner = nested_descriptive_rows(factors, y, decimals, sub, depth + 1)
        label = f.labels[lv] if lv is not None else "Total"
        for i, r in enumerate(inner):
            out.append([label if i == 0 else ""] + r)
    return out


def levene_table(dep_titles: list[str], samples_per_dep: list[list[np.ndarray]], design: str,
                 title: str = "Levene's Test of Equality of Error Variancesᵃ") -> Table:
    rows = []
    for dep, samples in zip(dep_titles, samples_per_dep):
        for i, (basis, F, df1, df2, p) in enumerate(levene_rows(samples)):
            rows.append([dep if i == 0 else "", basis, fmt(F), fmt_df(df1), fmt_df(df2),
                         fmt_p(p)])
    return Table(title, ["", "", "Levene Statistic", "df1", "df2", "Sig."], rows,
                 row_header_cols=2,
                 footnotes=["Tests the null hypothesis that the error variance of the dependent "
                            "variable is equal across groups.", f"a. Design: {design}"])


def adjusted(p: float, m: float, method: str) -> float:
    if method == "bonferroni":
        return min(1.0, p * m)
    if method == "sidak":
        return min(1.0, 1 - (1 - p) ** m)
    return p


def adjusted_crit(level: float, m: float, df: float, method: str) -> float:
    alpha = 1 - level
    if method == "bonferroni":
        alpha = alpha / m
    elif method == "sidak":
        alpha = 1 - (1 - alpha) ** (1 / m)
    return float(stats.t.ppf(1 - alpha / 2, df))


def pairwise_table(title: str, head_cols: list[str], factor_title: str, rows: list[list[str]],
                   flagged: bool, method: str, level: float, notes: list[str]) -> Table:
    cols = head_cols + [f"(I) {factor_title}", f"(J) {factor_title}",
                        "Mean Difference (I-J)", "Std. Error", "Sig.ᵇ",
                        (f"{pct(level)} Confidence Interval for Differenceᵇ", "Lower Bound"),
                        (f"{pct(level)} Confidence Interval for Differenceᵇ", "Upper Bound")]
    foot = list(notes) + ["Based on estimated marginal means"]
    if flagged:
        foot.append(f"*. The mean difference is significant at the {1 - level:.2f} level.")
    foot.append(f"b. Adjustment for multiple comparisons: {ADJUST[method]}.")
    return Table(title, cols, rows, row_header_cols=len(head_cols) + 2, footnotes=foot)


def estimates_rows(cells: list[tuple[list[str], np.ndarray, np.ndarray]], model: LinearModel,
                   level: float) -> list[list[str]]:
    q = float(stats.t.ppf(0.5 + level / 2, model.ve)) if model.ve > 0 else float("nan")
    rows = []
    for labels, l, c in cells:
        est, se = model.estimate(l, c)
        rows.append(labels + [fmt(est), fmt(se), fmt(est - q * se), fmt(est + q * se)])
    return rows


def covariate_note(model: LinearModel, letter: str = "a") -> list[str]:
    if not model.covariates:
        return []
    vals = ", ".join(f"{k} = {fmt(v, 4)}" for k, v in model.cov_means.items())
    return [f"{letter}. Covariates appearing in the model are evaluated at the following values: "
            f"{vals}."]


def blank_repeats(rows: list[list[str]], ncols: int) -> list[list[str]]:
    """Blank leading label cells that repeat the row above (SPSS pivot-table nesting)."""
    out, prev = [], [None] * ncols
    for r in rows:
        new = list(r)
        same = True
        for i in range(ncols):
            if same and r[i] == prev[i]:
                new[i] = ""
            else:
                same = False
        prev = r[:ncols]
        out.append(new)
    return out


# ------------------------------------------------------------------ MANOVA
def _frame(ds: Dataset, names: list[str], numeric_names: set[str]) -> pd.DataFrame:
    data = {}
    for n in names:
        data[n] = numeric(ds, n) if n in numeric_names else ds.analysis_series(n)
    return pd.DataFrame(data).dropna()


def manova(ds: Dataset, dependents: list[str], factors: list[str],
           covariates: list[str] | None = None, descriptives: bool = True,
           effect_size: bool = True, homogeneity: bool = True,
           posthoc_factors: list[str] | None = None, posthoc_methods=(),
           emm_factors: list[str] | None = None, compare: bool = False,
           adjustment: str = "bonferroni", level: float = 0.95) -> Section:
    covariates = covariates or []
    posthoc_factors = posthoc_factors or []
    emm_factors = emm_factors or []
    section = Section("Multivariate Analysis of Variance")
    log = f"GLM {' '.join(dependents)} BY {' '.join(factors)}"
    if covariates:
        log += f" WITH {' '.join(covariates)}"
    log += "\n  /METHOD=SSTYPE(3)\n  /INTERCEPT=INCLUDE"
    if posthoc_factors and posthoc_methods:
        log += (f"\n  /POSTHOC={' '.join(posthoc_factors)}"
                f"({' '.join(m.upper() for m in posthoc_methods)})")
    for e in emm_factors:
        log += f"\n  /EMMEANS=TABLES({e})" + (f" COMPARE ADJ({adjustment.upper()})"
                                              if compare and e != "(OVERALL)"
                                              and "*" not in e else "")
    stats_list = [s for s, on in (("DESCRIPTIVE", descriptives), ("ETASQ", effect_size),
                                  ("HOMOGENEITY", homogeneity)) if on]
    if stats_list:
        log += f"\n  /PRINT={' '.join(stats_list)}"
    log += f"\n  /CRITERIA=ALPHA({1 - level:.2f})"
    design_terms = covariates + ["*".join(c) for r in range(1, len(factors) + 1)
                                 for c in itertools.combinations(factors, r)]
    section.log(log + "\n  /DESIGN=" + " ".join(design_terms) + ".")

    frame = _frame(ds, dependents + factors + covariates, set(dependents + covariates))
    if len(dependents) < 2:
        section.note("Warning", "Multivariate analysis needs at least two dependent variables.",
                     style="warning")
        return section
    fobjs = [make_factor(ds, f, frame[f]) for f in factors]
    Y = frame[dependents].to_numpy(float)
    model = LinearModel(Y, fobjs, {ds.var(c).name: frame[c].to_numpy(float)
                                   for c in covariates})
    design = model.design_text
    n, p = Y.shape
    if model.ve < p:
        section.note("Warning", "There are not enough error degrees of freedom for the "
                     "multivariate tests.", style="warning")
        return section
    if model.deficient:
        section.note("Warning", "Some cells of the design are empty; Type III results use a "
                     "generalized inverse and may differ from SPSS.", style="warning")
    dvars = [ds.var(d) for d in dependents]

    if fobjs:
        section.table(between_factors_table(fobjs))
    if descriptives:
        rows = []
        for v, j in zip(dvars, range(p)):
            inner = nested_descriptive_rows(fobjs, Y[:, j], v.decimals)
            for i, r in enumerate(inner):
                rows.append([v.title if i == 0 else ""] + r)
        section.table(Table("Descriptive Statistics",
                            [""] + [f.name for f in fobjs] + ["Mean", "Std. Deviation", "N"],
                            rows, row_header_cols=1 + len(fobjs)))
    cells = cell_index(fobjs)
    if homogeneity and fobjs:
        groups = [Y[cells == c] for c in np.unique(cells)]
        t = box_table(groups, design)
        if t is not None:
            section.table(t)
    blocks = [(t.label, multivariate_tests(model.hypothesis(t), model.E, len(t.idx), model.ve))
              for t in model.terms]
    section.table(multivariate_table("Multivariate Tests", blocks, effect_size, design))
    if homogeneity and fobjs:
        source = model.resid if covariates else Y
        samples = [[source[cells == c, j] for c in np.unique(cells)] for j in range(p)]
        section.table(levene_table([v.title for v in dvars], samples, design))

    # Tests of Between-Subjects Effects
    ss_ct = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    sse = np.diag(model.E)
    mse = sse / model.ve
    rows = []
    r2_notes = []

    def add(source, j, ss, df, test=True):
        f = (ss / df) / mse[j] if test and df > 0 and mse[j] > 0 else float("nan")
        cells_ = [source, dvars[j].title, fmt(ss), fmt_df(df), fmt(ss / df) if df else "",
                  fmt(f) if test else "", fmt_p(stats.f.sf(f, df, model.ve)) if test else ""]
        if effect_size:
            cells_.append(fmt(ss / (ss + sse[j])) if test and ss + sse[j] > 0 else "")
        rows.append(cells_)

    df_model = model.rank - 1
    for j in range(p):
        ss_model = ss_ct[j] - sse[j]
        r2 = ss_model / ss_ct[j] if ss_ct[j] > 0 else float("nan")
        adj = 1 - (1 - r2) * (n - 1) / model.ve
        letter = chr(ord("a") + j)
        add("Corrected Model" if j == 0 else "", j, ss_model, df_model)
        rows[-1][2] += sup(letter)
        r2_notes.append(f"{letter}. R Squared = {fmt(r2)} (Adjusted R Squared = {fmt(adj)})")
    for t in model.terms:
        H = model.hypothesis(t)
        for j in range(p):
            add(t.label if j == 0 else "", j, H[j, j], len(t.idx))
    for title, vals, df, test in (("Error", sse, model.ve, False),
                                  ("Total", (Y ** 2).sum(axis=0), n, False),
                                  ("Corrected Total", ss_ct, n - 1, False)):
        for j in range(p):
            add(title if j == 0 else "", j, vals[j], df, test)
            if title != "Error":
                rows[-1][4] = ""
    columns = ["Source", "Dependent Variable", "Type III Sum of Squares", "df", "Mean Square",
               "F", "Sig."]
    if effect_size:
        columns.append("Partial Eta Squared")
    section.table(Table("Tests of Between-Subjects Effects", columns, rows, row_header_cols=2,
                        footnotes=r2_notes))

    # Estimated marginal means
    if emm_factors:
        _manova_emm(section, model, fobjs, dvars, emm_factors, compare, adjustment, level)

    # Post hoc
    if posthoc_factors and posthoc_methods:
        if covariates:
            section.note("Warning", "Post hoc tests are not performed when covariates are in "
                         "the model.", style="warning")
        else:
            for fname in posthoc_factors:
                f = next(o for o in fobjs if o.name == ds.var(fname).name)
                if len(f.levels) < 3:
                    section.note("", f"Post hoc tests are not performed for {f.name} because "
                                 "there are fewer than three groups.")
                    continue
                mc = []
                for j, v in enumerate(dvars):
                    groups = [Group(lv, lab, Y[f.codes == i, j])
                              for i, (lv, lab) in enumerate(zip(f.levels, f.labels))]
                    for meth in [m for m in POSTHOC_ORDER if m in posthoc_methods]:
                        mc.append((v.title, groups, meth,
                                   posthoc(groups, meth, mse[j], model.ve, level)))
                tab = multiple_comparisons_table(mc, level, f.title)
                tab.title = f"Multiple Comparisons — {f.title}"
                tab.footnotes.append("Based on observed means. The error term is Mean "
                                     "Square(Error) = " +
                                     ", ".join(f"{fmt(mse[j])} ({v.title})"
                                               for j, v in enumerate(dvars)) + ".")
                section.table(tab)
    return section


def _manova_emm(section: Section, model: LinearModel, fobjs: list[Factor], dvars, emm: list[str],
                compare: bool, adjustment: str, level: float) -> None:
    p = model.Y.shape[1]
    cov_notes = covariate_note(model)
    for spec in emm:
        if spec == "(OVERALL)":
            l = model.average_row({})
            rows = estimates_rows([([v.title], l, np.eye(p)[j]) for j, v in enumerate(dvars)],
                                  model, level)
            section.table(Table("Grand Mean", ["Dependent Variable", "Mean", "Std. Error",
                                               (f"{pct(level)} Confidence Interval",
                                                "Lower Bound"),
                                               (f"{pct(level)} Confidence Interval",
                                                "Upper Bound")], rows, footnotes=cov_notes))
            continue
        names = [s.strip() for s in spec.split("*")]
        idx = [next(i for i, f in enumerate(fobjs) if f.name == nm) for nm in names]
        combos = list(itertools.product(*[range(len(fobjs[i].levels)) for i in idx]))
        cells = []
        for j, v in enumerate(dvars):
            for combo in combos:
                labels = [v.title] + [fobjs[i].labels[lv] for i, lv in zip(idx, combo)]
                cells.append((labels, model.average_row(dict(zip(idx, combo))), np.eye(p)[j]))
        rows = blank_repeats(estimates_rows(cells, model, level), 1 + len(idx))
        section.table(Table(f"Estimates — {spec}" if len(idx) == 1 else spec,
                            ["Dependent Variable"] + [fobjs[i].name for i in idx] +
                            ["Mean", "Std. Error",
                             (f"{pct(level)} Confidence Interval", "Lower Bound"),
                             (f"{pct(level)} Confidence Interval", "Upper Bound")],
                            rows, row_header_cols=1 + len(idx), footnotes=cov_notes))
        if compare and len(idx) == 1:
            f = fobjs[idx[0]]
            all_rows, flagged = [], False
            for j, v in enumerate(dvars):
                c = np.eye(p)[j]
                ests = [(model.average_row({idx[0]: lv}), c) for lv in range(len(f.levels))]
                r, fl = pairwise_rows(f.labels, ests, model, adjustment, level, [v.title])
                all_rows += r
                flagged |= fl
            section.table(pairwise_table(f"Pairwise Comparisons — {f.name}",
                                         ["Dependent Variable"], f.name, all_rows, flagged,
                                         adjustment, level, []))


def pairwise_rows(labels: list[str], ests: list[tuple[np.ndarray, np.ndarray]],
                      model: LinearModel, method: str, level: float,
                      prefix: list[str]) -> tuple[list[list[str]], bool]:
    """Rows for (I)-(J) comparisons where level i is ``(l_i, c_i)``; the contrast is taken in
    the component that differs (l for between-subjects, c for within-subjects factors)."""
    k = len(labels)
    m = k * (k - 1) / 2
    df = model.ve
    crit = adjusted_crit(level, m, df, method) if m else float("nan")
    rows, flagged = [], False
    first = True
    for i in range(k):
        for jj, j in enumerate([x for x in range(k) if x != i]):
            li, ci = ests[i]
            lj, cj = ests[j]
            if np.allclose(li, lj):
                l, c = li, ci - cj
            else:
                l, c = li - lj, ci
            diff, se = model.estimate(l, c)
            t = diff / se if se > 0 else float("nan")
            p = adjusted(float(2 * stats.t.sf(abs(t), df)), m, method) if se > 0 \
                else float("nan")
            star = "*" if p < 1 - level else ""
            flagged |= bool(star)
            rows.append((list(prefix) if first else [""] * len(prefix)) +
                        [labels[i] if jj == 0 else "", labels[j], fmt(diff) + star, fmt(se),
                         fmt_p(p), fmt(diff - crit * se), fmt(diff + crit * se)])
            first = False
    return rows, flagged


__all__ = ["Factor", "LinearModel", "MVTest", "multivariate_tests", "box_m", "manova",
           "POSTHOC"]

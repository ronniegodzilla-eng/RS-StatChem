"""Analyze > General Linear Model > Repeated Measures, mirroring SPSS GLM output.

One or two within-subjects factors (wide data: one variable per level combination, the last
factor varying fastest, as in SPSS's Define Factors dialog), optional between-subjects
factors (mixed design) and covariates. Tables: Within-Subjects Factors, Between-Subjects
Factors, Descriptive Statistics, Box's M, Multivariate Tests, Mauchly's Test of Sphericity
(Greenhouse-Geisser, Huynh-Feldt, Lower-bound epsilons), Tests of Within-Subjects Effects,
Tests of Within-Subjects Contrasts (orthonormal polynomial contrasts), Levene's test,
Tests of Between-Subjects Effects (transformed variable "Average"), Estimated Marginal Means
with Bonferroni / Sidak / LSD pairwise comparisons, post hoc tests for between factors and
profile plots.

As in SPSS, covariates are not centred, so within-subjects effects are evaluated at
covariate = 0 when covariates are in the model.
"""
from __future__ import annotations

import io
import itertools
import math

import numpy as np
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import Group, fmt, fmt_df, fmt_int, fmt_p, pct, sup
from rsstatchem.core.stats.glm_model import contrast_matrix
from rsstatchem.core.stats.manova import (LinearModel, _frame, between_factors_table,
                                          blank_repeats, box_table, cell_index, covariate_note,
                                          estimates_rows, levene_table, make_factor,
                                          multivariate_table, multivariate_tests,
                                          nested_descriptive_rows, pairwise_rows, pairwise_table)

CORRECTIONS = ("Sphericity Assumed", "Greenhouse-Geisser", "Huynh-Feldt", "Lower-bound")
SS_NAMES = {1: "Type I", 2: "Type II", 3: "Type III", 4: "Type IV"}


def poly_contrasts(k: int) -> np.ndarray:
    """Orthonormal polynomial contrasts for k equally spaced levels (k x (k-1))."""
    x = np.arange(1, k + 1, dtype=float)
    V = np.vander(x - x.mean(), k, increasing=True)
    Q, R = np.linalg.qr(V)
    Q = Q * np.sign(np.diag(R))
    return Q[:, 1:]


def contrast_names(k: int) -> list[str]:
    base = ["Linear", "Quadratic", "Cubic"]
    return [base[i] if i < 3 else f"Order {i + 1}" for i in range(k - 1)]


def epsilons(E: np.ndarray, ve: int) -> tuple[float, float, float]:
    """Greenhouse-Geisser, Huynh-Feldt and lower-bound epsilon from the error SSCP of the
    orthonormalised transformed variables."""
    d = E.shape[0]
    tr = float(np.trace(E))
    gg = tr * tr / (d * float(np.trace(E @ E))) if tr > 0 else float("nan")
    hf = ((ve + 1) * d * gg - 2) / (d * (ve - d * gg))
    return gg, min(1.0, hf), 1.0 / d


def mauchly(E: np.ndarray, ve: int) -> tuple[float, float, int, float]:
    d = E.shape[0]
    tr = float(np.trace(E))
    W = float(np.linalg.det(E) / (tr / d) ** d) if tr > 0 else float("nan")
    df = d * (d + 1) // 2 - 1
    if df == 0:
        return 1.0, 0.0, 0, float("nan")
    chi = -(ve - (2 * d * d + d + 2) / (6 * d)) * math.log(W) if W > 0 else float("nan")
    return W, chi, df, float(stats.chi2.sf(chi, df))


class WithinDesign:
    """Within-subjects factors and their orthonormal transformation matrices."""

    def __init__(self, factors: list[tuple[str, int]]):
        self.factors = factors
        self.ks = [k for _, k in factors]
        self.K = int(np.prod(self.ks))
        self.effects = [combo for r in range(1, len(factors) + 1)
                        for combo in itertools.combinations(range(len(factors)), r)]

    def M(self, effect: tuple[int, ...]) -> np.ndarray:
        mats = [poly_contrasts(k) if i in effect else np.ones((k, 1)) / math.sqrt(k)
                for i, k in enumerate(self.ks)]
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out

    def average(self) -> np.ndarray:
        return np.ones((self.K, 1)) / math.sqrt(self.K)

    def label(self, effect: tuple[int, ...]) -> str:
        return " * ".join(self.factors[i][0] for i in effect)

    def contrast_labels(self, effect: tuple[int, ...]) -> list[list[str]]:
        """Per column of M(effect): one label per within factor ("" when not in effect)."""
        per = [contrast_names(k) if i in effect else [""] for i, k in enumerate(self.ks)]
        return [list(c) for c in itertools.product(*per)]

    def contrast_basis(self, effect: tuple[int, ...], kinds: dict[str, str]
                       ) -> tuple[np.ndarray, list[list[str]]]:
        """Transformation for the Tests of Within-Subjects Contrasts table: the requested
        contrast type per within factor (polynomial by default; other contrast types are used
        unnormalized, as SPSS prints them)."""
        mats, per = [], []
        for i, (name, k) in enumerate(self.factors):
            if i in effect:
                kind = kinds.get(name, "polynomial")
                if kind == "polynomial":
                    mats.append(poly_contrasts(k))
                    per.append(contrast_names(k))
                else:
                    C, labels = contrast_matrix(kind, k, "last")
                    mats.append(C.T)
                    per.append(labels)
            else:
                mats.append(np.ones((k, 1)) / math.sqrt(k))
                per.append([""])
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out, [list(c) for c in itertools.product(*per)]

    def level_of(self, col: int) -> tuple[int, ...]:
        return tuple(int(v) for v in np.unravel_index(col, self.ks))

    def c_vector(self, fixed: dict[int, int]) -> np.ndarray:
        mask = np.array([all(self.level_of(c)[i] == lv for i, lv in fixed.items())
                         for c in range(self.K)], float)
        return mask / mask.sum()

    @property
    def design_text(self) -> str:
        return " + ".join(self.label(e) for e in self.effects)


def repeated_measures(ds: Dataset, within: list[tuple[str, int]], dependents: list[str],
                      between: list[str] | None = None, covariates: list[str] | None = None,
                      measure: str = "MEASURE_1", descriptives: bool = True,
                      effect_size: bool = True, homogeneity: bool = False,
                      emm: list[str] | None = None, compare: bool = False,
                      adjustment: str = "bonferroni", posthoc_factors: list[str] | None = None,
                      posthoc_methods=(), plots: list[tuple[str, str | None]] | None = None,
                      level: float = 0.95, options=None,
                      within_contrasts: dict[str, str] | None = None):
    """SPSS GLM Repeated Measures. With ``options`` (:class:`glm_output.GLMOptions`: custom
    between-subjects model and SS type, observed power, parameter estimates, extended post hoc
    tests, simple main effects, error bars, Save) a :class:`glm_output.GLMResult` is returned;
    ``within_contrasts`` maps a within factor to deviation / simple / difference / helmert /
    repeated / polynomial for the Tests of Within-Subjects Contrasts table."""
    from rsstatchem.core.stats import glm_output as go

    legacy = options is None
    if options is not None:
        o = options
        effect_size, descriptives, homogeneity = o.effect_size, o.descriptives, o.homogeneity
        emm, compare, adjustment = list(o.emm), o.compare, o.adjust
        posthoc_factors, posthoc_methods = list(o.posthoc_factors), list(o.posthoc_methods)
        plots = list(o.plots)
        level = o.level
    else:
        o = go.GLMOptions(alpha=1 - level)
    power = o.power
    within_contrasts = within_contrasts or {}
    between = between or []
    covariates = covariates or []
    emm = emm or []
    plots = plots or []
    posthoc_factors = posthoc_factors or []
    wd = WithinDesign(within)
    section = Section("General Linear Model")
    if len(dependents) != wd.K:
        section.note("Warning", f"The within-subjects design needs {wd.K} variables; "
                     f"{len(dependents)} were given.", style="warning")
        return section if legacy else go.GLMResult(section)
    log = f"GLM {' '.join(dependents)}"
    if between:
        log += f" BY {' '.join(between)}"
    if covariates:
        log += f" WITH {' '.join(covariates)}"
    log += ("\n  /WSFACTOR=" + " ".join(f"{n} {k} Polynomial" for n, k in within))
    if measure and measure != "MEASURE_1":
        log += f"\n  /MEASURE={measure}"
    for wf, kind in within_contrasts.items():
        log += f"\n  /WSFACTOR CONTRAST({wf})={kind.capitalize()}"
    log += f"\n  /METHOD=SSTYPE({o.ss_type})"
    for spec in plots:
        log += "\n  /PLOT=PROFILE(" + "*".join(x for x in spec[:3] if x) + ")"
    if o.save:
        log += "\n  /SAVE=" + " ".join(o.save)
    for e in emm:
        log += f"\n  /EMMEANS=TABLES({e})"
        if compare and e != "(OVERALL)" and "*" not in e:
            log += f" COMPARE ADJ({adjustment.upper() if adjustment != 'lsd' else 'LSD'})"
    stats_list = [s for s, on in (("DESCRIPTIVE", descriptives), ("ETASQ", effect_size),
                                  ("HOMOGENEITY", homogeneity)) if on]
    if stats_list:
        log += f"\n  /PRINT={' '.join(stats_list)}"
    wsdesign = " ".join(wd.label(e).replace(" ", "") for e in wd.effects)
    log += f"\n  /CRITERIA=ALPHA({1 - level:.2f})\n  /WSDESIGN={wsdesign}"
    design_terms = covariates + ["*".join(c) for r in range(1, len(between) + 1)
                                 for c in itertools.combinations(between, r)]
    section.log(log + "\n  /DESIGN=" + " ".join(design_terms) + ".")

    frame = _frame(ds, dependents + between + covariates, set(dependents + covariates))
    fobjs = [make_factor(ds, f, frame[f]) for f in between]
    Y = frame[dependents].to_numpy(float)
    model = LinearModel(Y, fobjs, {ds.var(c).name: frame[c].to_numpy(float)
                                   for c in covariates}, terms=o.terms,
                        intercept=o.intercept, ss_type=o.ss_type)
    if model.ve < 1:
        section.note("Warning", "Not enough valid cases for the model.", style="warning")
        return section if legacy else go.GLMResult(section)
    design = model.design_text
    wdesign = f"Within Subjects Design: {wd.design_text}"
    mnote = f"Measure: {measure}"

    # Within-Subjects Factors
    rows = []
    for col, dep in enumerate(dependents):
        rows.append([str(lv + 1) for lv in wd.level_of(col)] + [ds.var(dep).name])
    section.table(Table("Within-Subjects Factors", [n_ for n_, _ in within] +
                        ["Dependent Variable"], blank_repeats(rows, len(within) - 1 or 0)
                        if len(within) > 1 else rows, row_header_cols=len(within),
                        footnotes=[mnote]))
    if fobjs:
        section.table(between_factors_table(fobjs))
    dvars = [ds.var(d) for d in dependents]
    if descriptives:
        rows = []
        for j, v in enumerate(dvars):
            inner = nested_descriptive_rows(fobjs, Y[:, j], v.decimals)
            for i, r in enumerate(inner):
                rows.append([v.title if i == 0 else ""] + r)
        section.table(Table("Descriptive Statistics",
                            [""] + [f.name for f in fobjs] + ["Mean", "Std. Deviation", "N"],
                            rows, row_header_cols=1 + len(fobjs)))
    cells = cell_index(fobjs)
    if homogeneity and fobjs:
        t = box_table([Y[cells == c] for c in np.unique(cells)], f"{design}\n{wdesign}")
        if t is not None:
            section.table(t)

    # Multivariate tests
    blocks = []
    for eff in wd.effects:
        M = wd.M(eff)
        E = model.error(M)
        for term in model.terms:
            name = wd.label(eff) if term.label == "Intercept" else \
                f"{wd.label(eff)} * {term.label}"
            blocks.append((name, multivariate_tests(model.hypothesis(term, M), E,
                                                    model.term_df(term), model.ve)))
    section.table(multivariate_table("Multivariate Tests", blocks, effect_size, design,
                                     [wdesign]))

    # Mauchly
    rows = []
    eps_by_effect = {}
    for eff in wd.effects:
        E = model.error(wd.M(eff))
        W, chi, df, p = mauchly(E, model.ve)
        gg, hf, lb = epsilons(E, model.ve)
        eps_by_effect[eff] = (1.0, gg, hf, lb)
        rows.append([wd.label(eff), fmt(W), fmt(chi), fmt_int(df), fmt_p(p) if df else ".",
                     fmt(gg), fmt(hf), fmt(lb)])
    section.table(Table("Mauchly's Test of Sphericityᵃ",
                        ["Within Subjects Effect", "Mauchly's W", "Approx. Chi-Square", "df",
                         "Sig.", ("Epsilonᵇ", "Greenhouse-Geisser"),
                         ("Epsilonᵇ", "Huynh-Feldt"), ("Epsilonᵇ", "Lower-bound")], rows,
                        footnotes=[mnote, "Tests the null hypothesis that the error covariance "
                                   "matrix of the orthonormalized transformed dependent "
                                   "variables is proportional to an identity matrix.",
                                   f"a. Design: {design}", wdesign,
                                   "b. May be used to adjust the degrees of freedom for the "
                                   "averaged tests of significance. Corrected tests are "
                                   "displayed in the Tests of Within-Subjects Effects table."]))

    # Tests of Within-Subjects Effects
    rows = []
    for eff in wd.effects:
        M = wd.M(eff)
        d = M.shape[1]
        E = model.error(M)
        sse = float(np.trace(E))
        dfe = model.ve * d
        for term in model.terms:
            name = wd.label(eff) if term.label == "Intercept" else \
                f"{wd.label(eff)} * {term.label}"
            ss = float(np.trace(model.hypothesis(term, M)))
            dfh = model.term_df(term) * d
            F = (ss / dfh) / (sse / dfe) if sse > 0 else float("nan")
            for ci, (corr, e) in enumerate(zip(CORRECTIONS, eps_by_effect[eff])):
                cells_ = [name if ci == 0 else "", corr, fmt(ss), fmt_df(round(dfh * e, 3)),
                          fmt(ss / (dfh * e)), fmt(F),
                          fmt_p(stats.f.sf(F, dfh * e, dfe * e))]
                if effect_size:
                    cells_.append(fmt(ss / (ss + sse)))
                if power:
                    cells_ += _power(F, dfh * e, dfe * e, o.alpha)
                rows.append(cells_)
        for ci, (corr, e) in enumerate(zip(CORRECTIONS, eps_by_effect[eff])):
            cells_ = [f"Error({wd.label(eff)})" if ci == 0 else "", corr, fmt(sse),
                      fmt_df(round(dfe * e, 3)), fmt(sse / (dfe * e)), "", ""]
            cells_ += [""] * (effect_size + 2 * power)
            rows.append(cells_)
    cols = ["Source", "", f"{SS_NAMES[o.ss_type]} Sum of Squares", "df", "Mean Square", "F",
            "Sig."]
    if effect_size:
        cols.append("Partial Eta Squared")
    pnote = []
    if power:
        cols += ["Noncent. Parameter", "Observed Power" + sup("a")]
        pnote = [f"a. Computed using alpha = {fmt(o.alpha, 2)}"]
    section.table(Table("Tests of Within-Subjects Effects", cols, rows, row_header_cols=2,
                        footnotes=[mnote] + pnote))

    # Tests of Within-Subjects Contrasts
    rows = []
    nw = len(within)
    for eff in wd.effects:
        M, labels = wd.contrast_basis(eff, within_contrasts)
        E = model.error(M)
        for term in [*model.terms, None]:
            if term is None:
                name = f"Error({wd.label(eff)})"
            elif term.label == "Intercept":
                name = wd.label(eff)
            else:
                name = f"{wd.label(eff)} * {term.label}"
            H = model.hypothesis(term, M) if term is not None else None
            for j in range(M.shape[1]):
                if term is None:
                    ss, df = float(E[j, j]), model.ve
                    cells_ = [name if j == 0 else ""] + labels[j] + \
                        [fmt(ss), fmt_int(df), fmt(ss / df), "", ""]
                    cells_ += [""] * (effect_size + 2 * power)
                else:
                    ss, df = float(H[j, j]), model.term_df(term)
                    F = (ss / df) / (E[j, j] / model.ve) if E[j, j] > 0 else float("nan")
                    cells_ = [name if j == 0 else ""] + labels[j] + \
                        [fmt(ss), fmt_int(df), fmt(ss / df), fmt(F),
                         fmt_p(stats.f.sf(F, df, model.ve))]
                    if effect_size:
                        cells_.append(fmt(ss / (ss + E[j, j])))
                    if power:
                        cells_ += _power(F, df, model.ve, o.alpha)
                rows.append(cells_)
    cols = ["Source"] + [n_ for n_, _ in within] + [f"{SS_NAMES[o.ss_type]} Sum of Squares",
                                                    "df", "Mean Square", "F", "Sig."]
    if effect_size:
        cols.append("Partial Eta Squared")
    if power:
        cols += ["Noncent. Parameter", "Observed Power" + sup("a")]
    section.table(Table("Tests of Within-Subjects Contrasts", cols, rows,
                        row_header_cols=1 + nw, footnotes=[mnote] + pnote))

    # Levene
    if homogeneity and fobjs:
        source = model.resid if covariates else Y
        samples = [[source[cells == c, j] for c in np.unique(cells)] for j in range(wd.K)]
        section.table(levene_table([v.title for v in dvars], samples,
                                   f"{design}\n{wdesign}"))

    # Tests of Between-Subjects Effects
    A = wd.average()
    E0 = float(model.error(A)[0, 0])
    mse0 = E0 / model.ve
    rows = []
    for term in model.terms:
        ss = float(model.hypothesis(term, A)[0, 0])
        df = model.term_df(term)
        F = (ss / df) / mse0 if mse0 > 0 else float("nan")
        cells_ = [term.label, fmt(ss), fmt_int(df), fmt(ss / df), fmt(F),
                  fmt_p(stats.f.sf(F, df, model.ve))]
        if effect_size:
            cells_.append(fmt(ss / (ss + E0)))
        if power:
            cells_ += _power(F, df, model.ve, o.alpha)
        rows.append(cells_)
    rows.append(["Error", fmt(E0), fmt_int(model.ve), fmt(mse0), "", ""] +
                [""] * (effect_size + 2 * power))
    cols = ["Source", f"{SS_NAMES[o.ss_type]} Sum of Squares", "df", "Mean Square", "F",
            "Sig."]
    if effect_size:
        cols.append("Partial Eta Squared")
    if power:
        cols += ["Noncent. Parameter", "Observed Power" + sup("a")]
    section.table(Table("Tests of Between-Subjects Effects", cols, rows,
                        footnotes=[mnote, "Transformed Variable: Average"] + pnote))
    if o.parameters:
        for t in go.parameter_table(model, dvars, o):
            section.table(t)

    # Estimated marginal means
    if emm:
        _emm_tables(section, model, wd, fobjs, emm, compare, adjustment, level, measure,
                    o.simple)

    # Post hoc (between factors, observed means of the average over within levels)
    if posthoc_factors and posthoc_methods:
        if covariates:
            section.note("Warning", "Post hoc tests are not performed when covariates are in "
                         "the model.", style="warning")
        else:
            avg = Y.mean(axis=1)
            mse_avg = mse0 / wd.K
            for fname in posthoc_factors:
                f = next(o for o in fobjs if o.name == ds.var(fname).name)
                if len(f.levels) < 3:
                    section.note("", f"Post hoc tests are not performed for {f.name} because "
                                 "there are fewer than three groups.")
                    continue
                groups = [Group(lv, lab, avg[f.codes == i])
                          for i, (lv, lab) in enumerate(zip(f.levels, f.labels))]
                for tab in go.posthoc_tables([(measure, groups)], f.title, posthoc_methods,
                                             [mse_avg], model.ve, o.posthoc, True):
                    section.table(tab)

    for spec in plots:
        h, s = spec[0], (spec[1] if len(spec) > 1 else None)
        png = _profile_plot(model, wd, fobjs, h, s, measure, o)
        if png:
            section.image(f"Profile Plot: {h}" + (f" * {s}" if s else ""), png)
    if legacy:
        return section
    saved = go.saved_variables(model, o.save, frame.index, ds.df.index,
                               [ds.var(d).name for d in dependents]) if o.save else {}
    return go.GLMResult(section, saved)


def _power(F: float, df1: float, df2: float, alpha: float) -> list[str]:
    from rsstatchem.core.stats.glm_model import observed_power_f

    lam, pw = observed_power_f(F, df1, df2, alpha)
    return [fmt(lam), fmt(pw)]


# ------------------------------------------------------------------ marginal means
def _resolve(name: str, wd: WithinDesign, fobjs) -> tuple[str, int]:
    for i, (wn, _) in enumerate(wd.factors):
        if wn.lower() == name.lower():
            return "w", i
    for i, f in enumerate(fobjs):
        if f.name.lower() == name.lower():
            return "b", i
    raise KeyError(name)


def _levels(kind: str, i: int, wd: WithinDesign, fobjs) -> list[str]:
    if kind == "w":
        return [str(j + 1) for j in range(wd.ks[i])]
    return fobjs[i].labels


def _lc(model: LinearModel, wd: WithinDesign, assignment: list[tuple[str, int, int]]):
    bfix = {i: lv for kind, i, lv in assignment if kind == "b"}
    wfix = {i: lv for kind, i, lv in assignment if kind == "w"}
    return model.average_row(bfix), wd.c_vector(wfix)


def _emm_tables(section: Section, model: LinearModel, wd: WithinDesign, fobjs, emm: list[str],
                compare: bool, adjustment: str, level: float, measure: str,
                simple: bool = False) -> None:
    notes = [f"Measure: {measure}"] + covariate_note(model)
    ci_cols = ["Mean", "Std. Error", (f"{pct(level)} Confidence Interval", "Lower Bound"),
               (f"{pct(level)} Confidence Interval", "Upper Bound")]
    for spec in emm:
        if spec == "(OVERALL)":
            l, c = _lc(model, wd, [])
            section.table(Table("Grand Mean", ci_cols, estimates_rows([([], l, c)], model, level),
                                row_header_cols=0, footnotes=notes))
            continue
        names = [s.strip() for s in spec.split("*")]
        resolved = [_resolve(nm, wd, fobjs) for nm in names]
        levels = [_levels(k, i, wd, fobjs) for k, i in resolved]
        cells = []
        for combo in itertools.product(*[range(len(lv)) for lv in levels]):
            assignment = [(k, i, lv) for (k, i), lv in zip(resolved, combo)]
            l, c = _lc(model, wd, assignment)
            cells.append(([levels[j][lv] for j, lv in enumerate(combo)], l, c))
        rows = blank_repeats(estimates_rows(cells, model, level), len(names) - 1) \
            if len(names) > 1 else estimates_rows(cells, model, level)
        title = f"Estimates — {spec}" if len(names) == 1 else spec.replace("*", " * ")
        section.table(Table(title, names + ci_cols, rows, row_header_cols=len(names),
                            footnotes=notes))
        if compare and len(names) == 1:
            kind, i = resolved[0]
            ests = [_lc(model, wd, [(kind, i, lv)]) for lv in range(len(levels[0]))]
            rows, flagged = pairwise_rows(levels[0], ests, model, adjustment, level, [])
            section.table(pairwise_table(f"Pairwise Comparisons — {names[0]}", [], names[0],
                                         rows, flagged, adjustment, level, [f"Measure: {measure}"]))
        if simple and len(names) > 1:
            _simple_effects(section, model, wd, names, resolved, levels, adjustment, level,
                            measure)


def _simple_effects(section, model, wd, names, resolved, levels, adjustment, level,
                    measure) -> None:
    """Pairwise comparisons of each factor within every level combination of the others."""
    for t_pos, tname in enumerate(names):
        others = [k for k in range(len(names)) if k != t_pos]
        all_rows, flagged = [], False
        for combo in itertools.product(*[range(len(levels[k])) for k in others]):
            fixed = [(resolved[k][0], resolved[k][1], lv) for k, lv in zip(others, combo)]
            tk, ti = resolved[t_pos]
            ests = [_lc(model, wd, fixed + [(tk, ti, lv)]) for lv in range(len(levels[t_pos]))]
            prefix = [levels[k][lv] for k, lv in zip(others, combo)]
            r, fl = pairwise_rows(levels[t_pos], ests, model, adjustment, level, prefix)
            all_rows += r
            flagged |= fl
        other_names = [names[k] for k in others]
        section.table(pairwise_table(
            f"Pairwise Comparisons — {tname} within {' * '.join(other_names)}", other_names,
            tname, all_rows, flagged, adjustment, level, [f"Measure: {measure}"]))


def _profile_plot(model: LinearModel, wd: WithinDesign, fobjs, h: str, s: str | None,
                  measure: str, opts=None) -> bytes | None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    try:
        hk, hi = _resolve(h, wd, fobjs)
        sk_si = _resolve(s, wd, fobjs) if s else None
    except KeyError:
        return None
    hlev = _levels(hk, hi, wd, fobjs)
    fig = Figure(figsize=(6.0, 4.2), dpi=120)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    colors = ["#C62828", "#1565C0", "#2E7D32", "#6A1B9A", "#EF6C00", "#00838F", "#4E342E"]
    series = [None] if sk_si is None else list(range(len(_levels(*sk_si, wd, fobjs))))
    bars = getattr(opts, "error_bars", None)
    q = float(stats.t.ppf(0.5 + opts.level / 2, model.ve)) if bars == "ci" else \
        float(getattr(opts, "se_mult", 2.0))
    bar_chart = getattr(opts, "plot_type", "line") == "bar"
    for si, sv in enumerate(series):
        ys, half = [], []
        for hv in range(len(hlev)):
            assignment = [(hk, hi, hv)] + ([(sk_si[0], sk_si[1], sv)] if sk_si else [])
            l, c = _lc(model, wd, assignment)
            e, se = model.estimate(l, c)
            ys.append(e)
            half.append(q * se)
        label = None if sv is None else _levels(*sk_si, wd, fobjs)[sv]
        x = np.arange(len(hlev), dtype=float)
        if bar_chart:
            width = 0.8 / len(series)
            ax.bar(x - 0.4 + width * (si + 0.5), ys, width * 0.95,
                   color=colors[si % len(colors)], label=label,
                   yerr=half if bars else None, capsize=3)
        else:
            ax.errorbar(x, ys, yerr=half if bars else None, marker="o",
                        color=colors[si % len(colors)], label=label, linewidth=1.6,
                        capsize=3 if bars else 0)
    if bars:
        ax.text(1.0, -0.18, f"Error bars: {pct(opts.level)} CI" if bars == "ci" else
                f"Error bars: +/- {q:g} SE", transform=ax.transAxes, fontsize=7, color="#555",
                ha="right")
    ax.set_xticks(range(len(hlev)), hlev)
    ax.set_xlabel(h)
    ax.set_ylabel("Estimated Marginal Means")
    ax.set_title(f"Estimated Marginal Means of {measure}", fontsize=11, fontweight="bold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if sk_si is not None:
        ax.legend(title=s, fontsize=8, title_fontsize=9, frameon=False)
    if model.covariates:
        ax.text(0.0, -0.18, "Covariates appearing in the model are evaluated at: " +
                ", ".join(f"{k} = {v:.4f}" for k, v in model.cov_means.items()),
                transform=ax.transAxes, fontsize=7, color="#555")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    return buf.getvalue()

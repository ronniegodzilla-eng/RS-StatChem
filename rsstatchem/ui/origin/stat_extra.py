"""More Origin Statistics menu analyses, with report sheets (no Qt).

Descriptive: Grubbs and Dixon's Q outlier tests, Distribution Fit, Partial Correlation,
Cross Tabulation with chi-square. Hypothesis testing: one-sample (χ²) and two-sample (F)
tests for variance, equal-variance tests for k samples (Bartlett, Levene, Brown-Forsythe),
one- and two-sample tests for proportions. ANOVA: Three-Way ANOVA, one- and two-way
repeated-measures ANOVA (within × within or within × between) with Mauchly's test,
Greenhouse-Geisser / Huynh-Feldt / lower-bound corrections and the multivariate tests,
one-way MANOVA. Nonparametric: one-sample Wilcoxon signed rank, paired sign test,
two-sample Kolmogorov-Smirnov, Mood's median test.

The analyses register themselves in :mod:`stat_tests` (``st.register``) with their dialog
field specs, so the main window's ``run_stat`` / Recalculate / Change Parameters work for
them as for the built-in tests. Repeated-measures tests use the multivariate (orthonormal
contrast) formulation of SPSS GLM / Origin: for a within effect with contrast matrix M the
transformed data Y·M are fitted with the between-subjects design; the hypothesis and error
SSCP matrices H and E give the univariate F = (tr H / (d·df_h)) / (tr E / (d·df_e)), the
epsilons and Mauchly's W (:mod:`rsstatchem.core.stats.repeated`) and the multivariate tests
(:func:`rsstatchem.core.stats.manova.multivariate_tests`).
"""
from __future__ import annotations

import math
import types
import warnings

import numpy as np
from scipy import stats

from rsstatchem.core.analysis import distfit as dfit
from rsstatchem.core.analysis import outliers as ol
from rsstatchem.core.analysis import vartests as vt
from rsstatchem.core.analysis.common import num, pval
from rsstatchem.core.stats import nptests as npt
from rsstatchem.core.stats import proportions as prop
from rsstatchem.core.stats.manova import box_m, multivariate_tests
from rsstatchem.core.stats.repeated import epsilons, mauchly, poly_contrasts
from rsstatchem.ui.origin import report_graphs as rg
from rsstatchem.ui.origin import stat_tests as st
from rsstatchem.ui.origin.model import ReportSheet, Workbook

StatError = st.StatError
FORMS = {"raw": "Raw", "indexed": "Indexed"}
TAILS = {"two": "Two-tailed (≠)", "upper": "Upper-tailed (>)", "lower": "Lower-tailed (<)"}


# ====================================================================== shared helpers
def alpha_field(default: float = 0.05):
    return ("float", "alpha", "Significance Level", default, (0.0001, 0.5, 4))


def level_field(default: float = 0.95):
    return ("float", "level", "Confidence Level", default, (0.5, 0.9999, 4))


def new_report(first, title: str, base: str, report, key: str, settings: dict, mode: str):
    return st._new_report(first, title, base, report, key, settings, mode)


def target_book(books: dict, settings: dict, report) -> types.SimpleNamespace:
    """Anchor for analyses without input ranges: the workbook named in ``settings["book"]``
    (the active workbook when the dialog ran)."""
    if report is not None:
        return types.SimpleNamespace(book=None)
    wb = books.get(settings.get("book", ""))
    if not isinstance(wb, Workbook):
        wb = next((b for b in books.values() if isinstance(b, Workbook)), None)
    if wb is None:
        raise StatError("There is no workbook for the report.")
    return types.SimpleNamespace(book=wb)


def stamp(book) -> str:
    return st._stamp(book)


def decision(p: float, alpha: float, yes: str, no: str) -> str:
    return st._decision(p, alpha, yes, no)


def lvl(alpha: float) -> str:
    return st._level(alpha)


def aligned(inputs: list) -> tuple[np.ndarray, np.ndarray]:
    """Columns side by side (n × k, NaN-padded) and listwise-complete row mask."""
    n = max(len(r.values) for r in inputs)
    m = np.full((n, len(inputs)), np.nan)
    for j, r in enumerate(inputs):
        m[:len(r.values), j] = r.values
    return m, np.isfinite(m).all(axis=1)


def first_row(r) -> int:
    return r.rows[0] if r.rows else 1


def groups_input(books: dict, settings: dict, minimum: int = 2):
    return st._groups(books, settings, minimum)


def report_text(first_book, name: str, report, lines=()) -> str:
    return f"{stamp(first_book)}\n{name} → [{first_book.name}]{report.name}" + \
        ("\n" + "\n".join(lines) if lines else "")


def index_graph(v: np.ndarray, flagged: list[int], title: str, ytitle: str,
                start: int = 1) -> object:
    """Data vs. row index with the flagged points in red (outlier tests)."""
    g = rg.new_graph(title, 11.0, 7.5)
    x = np.arange(start, start + len(v), dtype=float)
    keep = np.array([i not in flagged for i in range(len(v))])
    rg.add_series(g, x[keep], v[keep], "Data", "scatter", "#1f5fbf", size=5.0)
    if flagged:
        idx = np.array(flagged, int)
        rg.add_series(g, x[idx], v[idx], "Outlier", "scatter", rg.FIT_RED, symbol="D",
                      size=7.0)
    return rg.finish(g, "Row", ytitle, legend=bool(flagged))


# ====================================================================== outliers
GRUBBS_SIDES = {"two": "Two-sided (max |x − mean|)", "max": "Maximum is an outlier",
                "min": "Minimum is an outlier"}


def grubbs_report(books: dict, settings: dict, report: ReportSheet | None = None,
                  mode: str = "manual"):
    inputs = st.read_ranges(books, settings.get("ranges", []))
    alpha = float(settings.get("alpha", 0.05))
    side = settings.get("side", "two")
    iterative = bool(settings.get("iterative", False))
    first = inputs[0]
    report, fresh = new_report(first, f"Grubbs Test of {first.text}", "Grubbs", report,
                               "grubbs", settings, mode)
    st._notes(fresh, "Grubbs' Test for Outliers", inputs,
              [["Alternative", GRUBBS_SIDES.get(side, side)],
               ["Significance Level", num(alpha)],
               ["Mode", "Iterative (remove and repeat)" if iterative else "Single test"],
               ["Method", "G = |x − mean|/SD; critical value from t(α/(2N), N − 2) "
                          "(α/N one-sided); p = Bonferroni bound N·P(T > t_G)"]])
    rows, steps, lines, plots = [], [], [], []
    for r in inputs:
        try:
            res = ol.grubbs_iterative(r.values, alpha, side) if iterative else \
                [ol.grubbs(r.values, alpha, side)]
        except ol.OutlierError as exc:
            raise StatError(f"{r.title}: {exc}") from None
        if not res:
            raise StatError(f"{r.title}: Grubbs' test needs at least 3 values.")
        g0 = res[0]
        rows.append([r.title, str(g0.n), num(g0.mean), num(g0.sd), str(g0.index + first_row(r)),
                     num(g0.value), num(g0.statistic, 5), num(g0.critical, 5), pval(g0.p),
                     "Outlier" if g0.outlier else "Not an outlier"])
        for k, s in enumerate(res):
            steps.append([r.title, str(k + 1), str(s.n), str(s.index + first_row(r)),
                          num(s.value), num(s.statistic, 5), num(s.critical, 5), pval(s.p),
                          "Outlier" if s.outlier else "Not an outlier"])
        flagged = [s.index for s in res if s.outlier]
        lines.append(f"  {r.title}: G = {num(g0.statistic, 5)}, critical = "
                     f"{num(g0.critical, 5)}, p = {pval(g0.p)}"
                     + (f"; outliers at rows {', '.join(str(i + first_row(r)) for i in flagged)}"
                        if flagged else "; no outlier"))
        plots.append((f"Data of {r.title}", index_graph(
            r.values, flagged, f"Grubbs Test of {r.title}", r.column.axis_title,
            first_row(r))))
    fresh.add_table("Grubbs Test", ["", "N", "Mean", "SD", "Row", "Suspect Value",
                                    "G Statistic", "Critical Value", "Prob",
                                    f"Decision at level({lvl(alpha)})"], rows,
                    [f"At the {num(alpha)} level, a value is an outlier when G exceeds the "
                     "critical value."])
    if iterative:
        fresh.add_table("Iterations", ["", "Step", "N", "Row", "Suspect Value", "G Statistic",
                                       "Critical Value", "Prob", "Decision"], steps,
                        ["Each step removes the outlier found in the previous step."])
    for title, g in plots:
        fresh.add_graph(title, g, width=380, group="Plots")
    report = st._finish(report, fresh)
    return report, report_text(first.book, "Grubbs Test", report, lines)


DIXON_KIND_CHOICES = {"auto": "Automatic (Dixon's ranges)", "r10": "r10 (n = 3–7)",
                      "r11": "r11 (n = 8–10)", "r21": "r21 (n = 11–13)",
                      "r22": "r22 (n = 14–25)"}


def dixon_report(books: dict, settings: dict, report: ReportSheet | None = None,
                 mode: str = "manual"):
    inputs = st.read_ranges(books, settings.get("ranges", []))
    alpha = float(settings.get("alpha", 0.05))
    side = settings.get("side", "two")
    kind = settings.get("kind", "auto")
    critical = settings.get("critical", "exact")
    first = inputs[0]
    report, fresh = new_report(first, f"Dixon's Q Test of {first.text}", "DixonQ", report,
                               "dixon", settings, mode)
    st._notes(fresh, "Dixon's Q Test for Outliers", inputs,
              [["Alternative", GRUBBS_SIDES.get(side, side).replace("(max |x − mean|)",
                                                                    "(more extreme end)")],
               ["Significance Level", num(alpha)],
               ["Statistic", DIXON_KIND_CHOICES.get(kind, kind)],
               ["Critical Values", "Rorabacher (1991) table where tabulated, else exact"
                if critical == "table" else "Exact (numerical integration, normal samples)"]])
    rows, ratio_rows, lines, plots = [], [], [], []
    for r in inputs:
        try:
            res = ol.dixon(r.values, alpha, kind, side, critical)
        except ol.OutlierError as exc:
            raise StatError(f"{r.title}: {exc}") from None
        rows.append([r.title, str(res.n), res.kind, "Lowest" if res.end == "low" else "Highest",
                     str(res.index + first_row(r)), num(res.value), num(res.statistic, 5),
                     num(res.critical, 5) + (" (table)" if res.source == "table" else ""),
                     pval(res.p), "Outlier" if res.outlier else "Not an outlier"])
        ratio_rows.append([r.title, num(res.ratios["low"], 5), num(res.ratios["high"], 5)])
        lines.append(f"  {r.title}: {res.kind} = {num(res.statistic, 5)}, critical = "
                     f"{num(res.critical, 5)}, p = {pval(res.p)} → "
                     f"{'outlier' if res.outlier else 'no outlier'}")
        plots.append((f"Data of {r.title}", index_graph(
            r.values, [res.index] if res.outlier else [], f"Dixon's Q Test of {r.title}",
            r.column.axis_title, first_row(r))))
    fresh.add_table("Dixon's Q Test", ["", "N", "Statistic", "Suspect End", "Row",
                                       "Suspect Value", "Q (ratio)", "Critical Value", "Prob",
                                       f"Decision at level({lvl(alpha)})"], rows,
                    ["Two-sided tests use the critical value at α/2 per end and p = 2 × "
                     "one-tailed probability (Rorabacher's two-tailed convention)."])
    fresh.add_table("Ratios of Both Ends", ["", "Lowest Value", "Highest Value"], ratio_rows)
    for title, g in plots:
        fresh.add_graph(title, g, width=380, group="Plots")
    report = st._finish(report, fresh)
    return report, report_text(first.book, "Dixon's Q Test", report, lines)


# ====================================================================== distribution fit
def distfit_report(books: dict, settings: dict, report: ReportSheet | None = None,
                   mode: str = "manual"):
    r = st.read_range(books, settings.get("range", ""))
    keys = [k for k in settings.get("dists", ["normal", "lognormal", "weibull", "gamma"])
            if k in dfit.DISTRIBUTIONS]
    if not keys:
        raise StatError("Select at least one distribution.")
    boot = int(settings.get("bootstrap", 0))
    alpha = float(settings.get("alpha", 0.05))
    v = r.finite
    if len(v) < 3:
        raise StatError(f"{r.title}: distribution fitting needs at least 3 values.")
    skipped = [k for k in keys if k in dfit.POSITIVE and np.any(v <= 0)]
    keys = [k for k in keys if k not in skipped]
    if not keys:
        raise StatError("The data have values ≤ 0: choose Normal, Logistic or Extreme Value.")
    report, fresh = new_report(r, f"Distribution Fit of {r.text}", "DistFit", report,
                               "distfit", settings, mode)
    notes = [["Estimation", "Maximum likelihood"],
             ["GOF p-values", "Parametric bootstrap" + f" ({boot} samples)" if boot else
              "Simple-hypothesis (conservative)"]]
    if skipped:
        notes.append(["Skipped (data ≤ 0)", ", ".join(dfit.DISTRIBUTIONS[k] for k in skipped)])
    st._notes(fresh, "Distribution Fit", [r], notes)
    try:
        fits = [dfit.fit(k, v, boot) for k in keys]
    except dfit.DistFitError as exc:
        raise StatError(str(exc)) from None
    prm = []
    for f in fits:
        for name, est, se in zip(dfit.PARAM_NAMES[f.key], f.params, f.se):
            prm.append([f.name, name, num(est), num(se)])
    fresh.add_table("Parameters", ["Distribution", "Parameter", "Value", "Standard Error"], prm,
                    ["Standard errors from the observed information matrix."])
    order = sorted(range(len(fits)), key=lambda i: fits[i].aic)
    gof = []
    for rank, i in enumerate(order):
        f = fits[i]
        gof.append([f.name, num(-2 * f.loglik), num(f.aic), num(f.bic), num(f.ks, 5),
                    pval(f.ks_p), num(f.ad, 5), pval(f.ad_p), num(f.cvm, 5), pval(f.cvm_p),
                    decision(f.ks_p, alpha, "Reject", "Can't reject"), str(rank + 1)])
    fresh.add_table("Goodness of Fit", ["Distribution", "-2 Log Likelihood", "AIC", "BIC",
                                        "Kolmogorov-Smirnov D", "Prob>D",
                                        "Anderson-Darling A²", "Prob>A²",
                                        "Cramér-von Mises W²", "Prob>W²",
                                        f"KS Decision at level({lvl(alpha)})", "Rank (AIC)"],
                    gof, [f"P-values: {fits[0].p_method}.",
                          "Lower AIC / BIC indicates the better-fitting distribution."])
    fresh.add_table("Fitted Statistics", ["Distribution", "Mean", "SD", "Median"],
                    [[f.name, num(f.extra["mean"]), num(f.extra["sd"]), num(f.extra["median"])]
                     for f in fits])
    # histogram (density) with the fitted densities
    g = rg.new_graph(f"Distribution Fit of {r.title}", 12.0, 8.5)
    from rsstatchem.core.graph.model import Plot
    key = rg._key(g, r.title, v, r.title, kind="num")
    h = Plot(type="histogram", name=r.title, y=key)
    h.fill.color = "#cfd8e8"
    h.fill.edge_color = rg.DATA_BLACK
    h.fill.edge_width = 0.8
    h.options.update({"normal_curve": False, "scale": "density", "bins": "sturges"})
    g.layers[0].plots.append(h)
    lo, hi = float(v.min()), float(v.max())
    pad = 0.1 * (hi - lo)
    x0 = lo - pad
    if any(k in dfit.POSITIVE for k in keys):
        x0 = max(x0, lo / 10)
    xs = np.linspace(x0, hi + pad, 300)
    pal = rg.PALETTES.get("Origin Classic")
    for i, f in enumerate(fits):
        rg.add_series(g, xs, f.frozen().pdf(xs), f.name, "line", pal[(i + 1) % len(pal)],
                      width=1.6)
    fresh.add_graph("Histogram and Fitted Densities", rg.finish(g, r.column.axis_title,
                                                                "Density", legend=True),
                    width=460, group="Plots")
    # P-P plot
    g2 = rg.new_graph("P-P Plot", 10.0, 8.5)
    xs_sorted = np.sort(v)
    emp = (np.arange(1, len(v) + 1) - 0.5) / len(v)
    rg.add_series(g2, [0, 1], [0, 1], "Reference", "line", "#808080", dash="--", width=1.0,
                  legend=False)
    for i, f in enumerate(fits):
        rg.add_series(g2, emp, f.frozen().cdf(xs_sorted), f.name, "scatter",
                      pal[(i + 1) % len(pal)], size=4.0)
    fresh.add_graph("P-P Plot", rg.finish(g2, "Empirical Probability", "Fitted Probability",
                                          legend=True), width=400, group="Plots")
    report = st._finish(report, fresh)
    best = fits[order[0]]
    return report, report_text(r.book, "Distribution Fit", report,
                               [f"  best by AIC: {best.name} (AIC = {num(best.aic)})"])


# ====================================================================== partial correlation
def partial_corr_values(X: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """(partial r matrix, two-sided p matrix, df) of the columns of X controlling for Z."""
    n = X.shape[0]
    D = np.column_stack([np.ones(n), Z]) if Z.size else np.ones((n, 1))
    beta, *_ = np.linalg.lstsq(D, X, rcond=None)
    E = X - D @ beta
    R = np.corrcoef(E, rowvar=False)
    df = n - 2 - (Z.shape[1] if Z.size else 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = R * np.sqrt(df / np.clip(1 - R ** 2, 1e-300, None))
    P = 2 * stats.t.sf(np.abs(t), df)
    np.fill_diagonal(P, np.nan)
    return R, P, df


def partial_corr_report(books: dict, settings: dict, report: ReportSheet | None = None,
                        mode: str = "manual"):
    inputs = st.read_ranges(books, settings.get("ranges", []))
    controls = [st.read_range(books, t) for t in settings.get("covariates", []) if t.strip()]
    if len(inputs) < 2:
        raise StatError("Select two or more variables.")
    if not controls:
        raise StatError("Select one or more control variables.")
    alpha = float(settings.get("alpha", 0.05))
    m, ok = aligned(inputs + controls)
    m = m[ok]
    k = len(inputs)
    if len(m) < len(controls) + 3:
        raise StatError("Not enough complete rows.")
    R, P, df = partial_corr_values(m[:, :k], m[:, k:])
    first = inputs[0]
    report, fresh = new_report(first, f"Partial Correlation of {first.text} …", "PartialCorr",
                               report, "partialcorr", settings, mode)
    st._notes(fresh, "Partial Correlation Coefficient", inputs + controls,
              [["Controlling for", ", ".join(c.title for c in controls)],
               ["Missing Values", "Exclude listwise"], ["N", str(len(m))]])
    names = [r.title for r in inputs]
    rr = [[names[i]] + [num(R[i, j], 5) + ("*" if i != j and P[i, j] < alpha else "")
                        for j in range(k)] for i in range(k)]
    pp = [[names[i]] + [pval(P[i, j]) if i != j else "--" for j in range(k)] for i in range(k)]
    fresh.add_table("Partial Correlation Coefficients", [""] + names, rr,
                    [f"*: significant at the {num(alpha)} level (two-tailed).",
                     f"DF = {df}"])
    fresh.add_table("Partial Correlation P-values", [""] + names, pp)
    report = st._finish(report, fresh)
    return report, report_text(first.book, "Partial Correlation", report)


# ====================================================================== cross tabulation
def crosstab_report(books: dict, settings: dict, report: ReportSheet | None = None,
                    mode: str = "manual"):
    r1 = st.read_range(books, settings.get("range1", ""))
    r2 = st.read_range(books, settings.get("range2", ""))
    alpha = float(settings.get("alpha", 0.05))
    n = min(len(r1.texts), len(r2.texts))
    pairs = [(r1.texts[i], r2.texts[i]) for i in range(n) if r1.texts[i] != "" and
             r2.texts[i] != ""]
    if not pairs:
        raise StatError("The two columns have no complete pairs.")
    rows_l = st._sort_levels([a for a, _ in pairs])
    cols_l = st._sort_levels([b for _, b in pairs])
    if len(rows_l) < 2 or len(cols_l) < 2:
        raise StatError("Each variable needs at least two categories.")
    T = np.zeros((len(rows_l), len(cols_l)))
    for a, b in pairs:
        T[rows_l.index(a), cols_l.index(b)] += 1
    report, fresh = new_report(r1, f"Cross Tabulation of {r1.title} by {r2.title}",
                               "CrossTab", report, "crosstab", settings, mode)
    st._notes(fresh, "Cross Tabulation", [r1, r2], [["Valid Pairs", str(len(pairs))]])
    tot = T.sum()
    body = [[rows_l[i]] + [str(int(v)) for v in T[i]] + [str(int(T[i].sum()))]
            for i in range(len(rows_l))]
    body.append(["Total"] + [str(int(v)) for v in T.sum(0)] + [str(int(tot))])
    fresh.add_table("Counts", [f"{r1.title} \\ {r2.title}"] + cols_l + ["Total"], body)
    chi, p, dof, E = stats.chi2_contingency(T, correction=False)
    fresh.add_table("Expected Counts", [f"{r1.title} \\ {r2.title}"] + cols_l,
                    [[rows_l[i]] + [num(v, 4) for v in E[i]] for i in range(len(rows_l))])
    g2, pg, _, _ = stats.chi2_contingency(T, correction=False, lambda_="log-likelihood")
    tests = [["Pearson Chi-Square", num(chi, 5), str(dof), pval(p)],
             ["Likelihood Ratio", num(g2, 5), str(dof), pval(pg)]]
    if T.shape == (2, 2):
        y, py, _, _ = stats.chi2_contingency(T, correction=True)
        tests.append(["Continuity Correction (Yates)", num(y, 5), "1", pval(py)])
        tests.append(["Fisher's Exact Test (two-sided)", "", "", pval(stats.fisher_exact(T)[1])])
    small = int((E < 5).sum())
    fresh.add_table("Chi-Square Tests", ["", "Value", "DF", "Prob"], tests,
                    [f"{small} cells ({small / E.size * 100:.1f}%) have expected count less "
                     "than 5." if small else "All expected counts are at least 5.",
                     f"At the {num(alpha)} level the variables are "
                     f"{'not ' if p >= alpha else ''}significantly associated (Pearson)."])
    kmin = min(T.shape) - 1
    fresh.add_table("Measures of Association", ["", "Value"],
                    [["Phi", num(math.sqrt(chi / tot), 5)],
                     ["Cramer's V", num(math.sqrt(chi / (tot * kmin)), 5)],
                     ["Contingency Coefficient", num(math.sqrt(chi / (chi + tot)), 5)]])
    report = st._finish(report, fresh)
    return report, report_text(r1.book, "Cross Tabulation", report,
                               [f"  Pearson χ²({dof}) = {num(chi, 5)}, p = {pval(p)}"])


# ====================================================================== variance tests
def var1_report(books: dict, settings: dict, report: ReportSheet | None = None,
                mode: str = "manual"):
    inputs = st.read_ranges(books, settings.get("ranges", []))
    s0 = float(settings.get("variance", 1.0))
    tail = settings.get("tail", "two")
    alpha = float(settings.get("alpha", 0.05))
    level = 1 - alpha
    first = inputs[0]
    report, fresh = new_report(first, f"One-Sample Test for Variance of {first.text}",
                               "OneSampleVar", report, "var1", settings, mode)
    sym = st._SYM[tail]
    st._notes(fresh, "One-Sample Test for Variance (chi-square)", inputs,
              [["Null Hypothesis", f"Variance = {num(s0)}"],
               ["Alternative Hypothesis", f"Variance {sym} {num(s0)}"],
               ["Significance Level", num(alpha)]])
    rows, lines = [], []
    for r in inputs:
        try:
            res = vt.chi2_variance_test(r.values, s0, tail, level)
        except vt.VarTestError as exc:
            raise StatError(f"{r.title}: {exc}") from None
        rows.append([r.title, str(res["n"]), num(res["variance"]), num(res["chi2"], 5),
                     str(res["df"]), pval(res["p"]), num(res["lcl"]), num(res["ucl"]),
                     decision(res["p"], alpha, "Reject H0", "Can't reject H0")])
        lines.append(f"  {r.title}: χ² = {num(res['chi2'], 5)}, DF = {res['df']}, "
                     f"p = {pval(res['p'])}")
    fresh.add_table("Chi-Square Test", ["", "N", "Variance", "Chi-Square", "DF", "Prob",
                                        f"Lower {level * 100:g}% CL of Variance",
                                        f"Upper {level * 100:g}% CL of Variance",
                                        f"Decision at level({lvl(alpha)})"], rows)
    report = st._finish(report, fresh)
    return report, report_text(first.book, "One-Sample Test for Variance", report, lines)


def var2_report(books: dict, settings: dict, report: ReportSheet | None = None,
                mode: str = "manual"):
    inputs, groups = groups_input(books, settings, 2)
    if len(groups) != 2:
        raise StatError("The two-sample variance test needs exactly two groups.")
    (n1, a), (n2, b) = groups
    ratio = float(settings.get("ratio", 1.0))
    tail = settings.get("tail", "two")
    alpha = float(settings.get("alpha", 0.05))
    level = 1 - alpha
    first = inputs[0]
    report, fresh = new_report(first, f"Two-Sample Test for Variance of {n1} and {n2}",
                               "TwoSampleVar", report, "var2", settings, mode)
    sym = st._SYM[tail]
    st._notes(fresh, "Two-Sample Test for Variance (F-test)", inputs,
              [["Null Hypothesis", f"Var1 / Var2 = {num(ratio)}"],
               ["Alternative Hypothesis", f"Var1 / Var2 {sym} {num(ratio)}"],
               ["Significance Level", num(alpha)]])
    try:
        res = vt.f_variance_test(a, b, ratio, tail, level)
        k = vt.k_sample_tests([a, b])
    except vt.VarTestError as exc:
        raise StatError(str(exc)) from None
    fresh.add_table("Descriptive Statistics", ["", "N", "Mean", "SD", "Variance"],
                    [[n1, str(len(a)), num(a.mean()), num(a.std(ddof=1)), num(res["var1"])],
                     [n2, str(len(b)), num(b.mean()), num(b.std(ddof=1)), num(res["var2"])]])
    fresh.add_table("F-Test for Equal Variance",
                    ["", "F Value", "DF1", "DF2", "Prob", "Variance Ratio",
                     f"Lower {level * 100:g}% CL", f"Upper {level * 100:g}% CL",
                     f"Decision at level({lvl(alpha)})"],
                    [[f"{n1} / {n2}", num(res["f"], 5), str(res["df1"]), str(res["df2"]),
                      pval(res["p"]), num(res["ratio"]), num(res["lcl"]), num(res["ucl"]),
                      decision(res["p"], alpha, "Reject H0", "Can't reject H0")]])
    _equal_var_table(fresh, k, alpha)
    fresh.add_graph("Box Chart", rg.box_graph(groups), width=380)
    report = st._finish(report, fresh)
    return report, report_text(first.book, "Two-Sample Test for Variance", report,
                               [f"  F({res['df1']}, {res['df2']}) = {num(res['f'], 5)}, "
                                f"p = {pval(res['p'])}"])


def _equal_var_table(fresh, k: dict, alpha: float) -> None:
    b, lv, bf = k["bartlett"], k["levene"], k["brown_forsythe"]
    fresh.add_table("Tests for Equal Variance", ["", "Statistic", "DF", "Prob",
                                                 f"Decision at level({lvl(alpha)})"],
                    [["Bartlett (chi-square)", num(b[0], 5), str(b[1]), pval(b[2]),
                      decision(b[2], alpha, "Variances differ", "Equal variances")],
                     ["Levene (absolute deviations from mean)", num(lv[0], 5),
                      f"{lv[1][0]}, {lv[1][1]}", pval(lv[2]),
                      decision(lv[2], alpha, "Variances differ", "Equal variances")],
                     ["Brown-Forsythe (deviations from median)", num(bf[0], 5),
                      f"{bf[1][0]}, {bf[1][1]}", pval(bf[2]),
                      decision(bf[2], alpha, "Variances differ", "Equal variances")]],
                    ["Bartlett's test assumes normal data; Levene and Brown-Forsythe are "
                     "robust to non-normality."])


def vark_report(books: dict, settings: dict, report: ReportSheet | None = None,
                mode: str = "manual"):
    inputs, groups = groups_input(books, settings, 2)
    alpha = float(settings.get("alpha", 0.05))
    first = inputs[0]
    report, fresh = new_report(first, f"Equal Variance Tests of {first.text}", "EqualVar",
                               report, "vark", settings, mode)
    st._notes(fresh, "Tests for Equal Variance", inputs, [["Significance Level", num(alpha)]])
    fresh.add_table("Descriptive Statistics", ["", "N", "Mean", "SD", "Variance"],
                    [[nm, str(len(v)), num(v.mean()), num(v.std(ddof=1)), num(v.var(ddof=1))]
                     for nm, v in groups])
    try:
        k = vt.k_sample_tests([v for _, v in groups])
    except vt.VarTestError as exc:
        raise StatError(str(exc)) from None
    _equal_var_table(fresh, k, alpha)
    fresh.add_graph("Box Chart", rg.box_graph(groups), width=400)
    report = st._finish(report, fresh)
    return report, report_text(first.book, "Tests for Equal Variance", report)


# ====================================================================== proportions
P_TESTS1 = {"exact": "Exact Binomial", "score": "Score (normal approximation)",
            "score_cc": "Score with continuity correction", "wald": "Wald"}
P_CI1 = {"clopper_pearson": "Clopper-Pearson (exact)", "wilson": "Wilson Score",
         "wald": "Wald", "agresti_coull": "Agresti-Coull", "jeffreys": "Jeffreys"}
P_TESTS2 = {"wald_h0": "Pooled z (Wald H0)", "wald": "Unpooled z (Wald)",
            "hauck_anderson": "Hauck-Anderson"}
P_CI2 = {"newcombe": "Newcombe (Wilson)", "wald": "Wald", "agresti_caffo": "Agresti-Caffo"}


def _counts(books: dict, settings: dict, suffix: str = "") -> tuple[int, int, list]:
    """(successes, trials, inputs) from summary values or a data column."""
    if settings.get("form", "summary") == "data":
        r = st.read_range(books, settings.get("range" + (suffix or ""), ""))
        texts = [t for t in r.texts if t != ""]
        succ = str(settings.get("success" + suffix, "1")).strip()
        hit = sum(1 for t in texts if t == succ or (np.isfinite(st.to_float(t)) and
                                                    np.isfinite(st.to_float(succ)) and
                                                    st.to_float(t) == st.to_float(succ)))
        return hit, len(texts), [r]
    return int(settings.get("x" + suffix, 0)), int(settings.get("n" + suffix, 0)), []


def _one_sided(z_one: float, two: float, observed: float, null: float, tail: str) -> float:
    if tail == "two":
        return two
    upper = observed > null
    return z_one if (tail == "upper") == upper else 1 - z_one


def prop1_report(books: dict, settings: dict, report: ReportSheet | None = None,
                 mode: str = "manual"):
    x, n, inputs = _counts(books, settings)
    if n <= 0 or not 0 <= x <= n:
        raise StatError("Use 0 ≤ number of successes ≤ sample size (> 0).")
    p0 = float(settings.get("p0", 0.5))
    tail = settings.get("tail", "two")
    method = settings.get("method", "exact")
    ci_m = settings.get("ci", "clopper_pearson")
    alpha = float(settings.get("alpha", 0.05))
    anchor = inputs[0] if inputs else target_book(books, settings, report)
    report, fresh = new_report(anchor, "One-Sample Test for Proportion", "OneSampleProp",
                               report, "prop1", settings, mode)
    sym = st._SYM[tail]
    st._notes(fresh, "One-Sample Test for Proportion", inputs,
              [["Null Hypothesis", f"Proportion = {num(p0)}"],
               ["Alternative Hypothesis", f"Proportion {sym} {num(p0)}"],
               ["Test", P_TESTS1.get(method, method)],
               ["Significance Level", num(alpha)]])
    z, one, two = prop.one_test(x, n, p0, method)
    p = _one_sided(one, two, x / n, p0, tail)
    lo, hi = prop.one_ci(x, n, ci_m, 1 - alpha)
    fresh.add_table("Proportion", ["Successes", "Sample Size", "Sample Proportion",
                                   f"Lower {100 * (1 - alpha):g}% CL",
                                   f"Upper {100 * (1 - alpha):g}% CL"],
                    [[str(x), str(n), num(x / n), num(lo), num(hi)]],
                    [f"Confidence interval: {P_CI1.get(ci_m, ci_m)}."])
    fresh.add_table("Test Statistics", ["", "Z", "Prob", f"Decision at level({lvl(alpha)})"],
                    [[P_TESTS1.get(method, method), num(z, 5), pval(p),
                      decision(p, alpha, "Reject H0", "Can't reject H0")]],
                    ["Two-sided probabilities are twice the one-sided tail probability."])
    report = st._finish(report, fresh)
    book = inputs[0].book if inputs else (anchor.book or _report_book(books, report))
    return report, report_text(book, "One-Sample Test for Proportion", report,
                               [f"  p̂ = {num(x / n)}, p = {pval(p)}"])


def _report_book(books: dict, report) -> Workbook:
    for b in books.values():
        if isinstance(b, Workbook) and report in b.sheets:
            return b
    return next(b for b in books.values() if isinstance(b, Workbook))


def prop2_report(books: dict, settings: dict, report: ReportSheet | None = None,
                 mode: str = "manual"):
    if settings.get("form", "summary") == "data":
        x1, n1, i1 = _counts(books, {**settings, "range": settings.get("range1", "")}, "")
        x2, n2, i2 = _counts(books, {**settings, "range": settings.get("range2", "")}, "")
        inputs = i1 + i2
    else:
        x1, n1, x2, n2 = (int(settings.get(k, 0)) for k in ("x1", "n1", "x2", "n2"))
        inputs = []
    if n1 <= 0 or n2 <= 0 or not 0 <= x1 <= n1 or not 0 <= x2 <= n2:
        raise StatError("Use 0 ≤ successes ≤ sample size (> 0) for both samples.")
    d0 = 0.0
    tail = settings.get("tail", "two")
    method = settings.get("method", "wald_h0")
    ci_m = settings.get("ci", "newcombe")
    alpha = float(settings.get("alpha", 0.05))
    anchor = inputs[0] if inputs else target_book(books, settings, report)
    report, fresh = new_report(anchor, "Two-Sample Test for Proportions", "TwoSampleProp",
                               report, "prop2", settings, mode)
    sym = st._SYM[tail]
    st._notes(fresh, "Two-Sample Test for Proportions", inputs,
              [["Null Hypothesis", "p1 − p2 = 0"],
               ["Alternative Hypothesis", f"p1 − p2 {sym} 0"],
               ["Test", P_TESTS2.get(method, method)], ["Significance Level", num(alpha)]])
    z, one, two = prop.ind_test(x1, n1, x2, n2, method)
    diff = x1 / n1 - x2 / n2
    p = _one_sided(one, two, diff, d0, tail)
    lo, hi = prop.ind_ci(x1, n1, x2, n2, ci_m, 1 - alpha)
    fresh.add_table("Proportions", ["", "Successes", "Sample Size", "Proportion"],
                    [["Sample 1", str(x1), str(n1), num(x1 / n1)],
                     ["Sample 2", str(x2), str(n2), num(x2 / n2)]])
    fresh.add_table("Difference", ["p1 − p2", f"Lower {100 * (1 - alpha):g}% CL",
                                   f"Upper {100 * (1 - alpha):g}% CL"],
                    [[num(diff), num(lo), num(hi)]],
                    [f"Confidence interval: {P_CI2.get(ci_m, ci_m)}."])
    T = np.array([[x1, n1 - x1], [x2, n2 - x2]], float)
    rows = [[P_TESTS2.get(method, method), num(z, 5), pval(p),
             decision(p, alpha, "Reject H0", "Can't reject H0")]]
    alt = {"two": "two-sided", "upper": "greater", "lower": "less"}[tail]
    fe = float(stats.fisher_exact(T, alternative=alt)[1])
    rows.append(["Fisher's Exact Test", "", pval(fe),
                 decision(fe, alpha, "Reject H0", "Can't reject H0")])
    fresh.add_table("Test Statistics", ["", "Z", "Prob", f"Decision at level({lvl(alpha)})"],
                    rows)
    report = st._finish(report, fresh)
    book = inputs[0].book if inputs else (anchor.book or _report_book(books, report))
    return report, report_text(book, "Two-Sample Test for Proportions", report,
                               [f"  p1 − p2 = {num(diff)}, p = {pval(p)}"])


# ====================================================================== three-way ANOVA
def anova3_report(books: dict, settings: dict, report: ReportSheet | None = None,
                  mode: str = "manual"):
    import pandas as pd
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    data = st.read_range(books, settings.get("data", ""))
    facs = [st.read_range(books, settings.get(k, "")) for k in ("factor_a", "factor_b",
                                                                 "factor_c")]
    n = min([len(data.values)] + [len(f.texts) for f in facs])
    rows = [(facs[0].texts[i], facs[1].texts[i], facs[2].texts[i], float(data.values[i]))
            for i in range(n) if all(f.texts[i] != "" for f in facs) and
            np.isfinite(data.values[i])]
    frame = pd.DataFrame(rows, columns=["FA", "FB", "FC", "y"])
    if len(frame) < 8 or any(frame[c].nunique() < 2 for c in ("FA", "FB", "FC")):
        raise StatError("Three-Way ANOVA needs three factors with two or more levels each.")
    alpha = float(settings.get("alpha", 0.05))
    ss_type = str(settings.get("ss", "3"))
    inter = settings.get("interaction", "all")
    main_eff = "C(FA, Sum) + C(FB, Sum) + C(FC, Sum)"
    terms = {"all": "C(FA, Sum) * C(FB, Sum) * C(FC, Sum)", "two": f"({main_eff}) ** 2",
             "none": main_eff}[inter]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.ols(f"y ~ {terms}", data=frame).fit()
        table = anova_lm(model, typ=int(ss_type))
    df_e, ss_e = float(model.df_resid), float(model.ssr)
    if df_e <= 0:
        raise StatError("No degrees of freedom for error: use fewer interaction terms.")
    mse = ss_e / df_e
    names = {"FA": facs[0].title, "FB": facs[1].title, "FC": facs[2].title}

    def label(term: str) -> str:
        return " * ".join(names[t.strip()[2:4]] for t in term.split(":"))

    report, fresh = new_report(data, f"Three-Way ANOVA of {data.text}", "ANOVA3Way", report,
                               "anova3", settings, mode)
    st._notes(fresh, "Three-Way ANOVA", [data] + facs,
              [["Interactions", {"all": "All", "two": "Two-way only", "none": "None"}[inter]],
               ["Sums of Squares", st.SS_TYPES.get(ss_type, ss_type)],
               ["Significance Level", num(alpha)]])
    out, lines = [], []
    for term, row in table.iterrows():
        if term in ("Intercept", "Residual"):
            continue
        ss, df, f, p = (float(row[k]) for k in ("sum_sq", "df", "F", "PR(>F)"))
        out.append([label(term), num(df), num(ss), num(ss / df), num(f, 5), pval(p)])
        lines.append(f"  {label(term)}: F = {num(f, 5)}, p = {pval(p)}")
    ss_tot = float(((frame.y - frame.y.mean()) ** 2).sum())
    out += [["Model", num(model.df_model), num(ss_tot - ss_e),
             num((ss_tot - ss_e) / model.df_model), num(float(model.fvalue), 5),
             pval(float(model.f_pvalue))],
            ["Error", num(df_e), num(ss_e), num(mse), "", ""],
            ["Corrected Total", num(len(frame) - 1), num(ss_tot), "", "", ""]]
    fresh.add_table("Overall ANOVA", ["", "DF", "Sum of Squares", "Mean Square", "F Value",
                                      "Prob>F"], out,
                    [f"At the {num(alpha)} level, a term with Prob>F < {num(alpha)} is "
                     "significant."])
    fresh.add_table("Fit Statistics", ["R-Square", "Coeff Var", "Root MSE", "Data Mean"],
                    [[num(1 - ss_e / ss_tot), num(math.sqrt(mse) / frame.y.mean()),
                      num(math.sqrt(mse)), num(frame.y.mean())]])
    methods = [m for m in settings.get("means", ["tukey"]) if m in st.MEANS_METHODS
               and m != "dunnett"]
    for c in ("FA", "FB", "FC"):
        lv = st._sort_levels(list(frame[c]))
        groups = [(v, frame.y[frame[c] == v].to_numpy(float)) for v in lv]
        st._means_tables(fresh, groups, methods, mse, df_e, alpha, names[c])
    report = st._finish(report, fresh)
    return report, report_text(data.book, "Three-Way ANOVA", report, lines)


# ====================================================================== repeated measures
def _glh(X: np.ndarray, Y: np.ndarray, L: np.ndarray) -> np.ndarray:
    XtXi = np.linalg.pinv(X.T @ X)
    B = XtXi @ X.T @ Y
    LB = L @ B
    return LB.T @ np.linalg.pinv(L @ XtXi @ L.T) @ LB


def rm_glm(Y: np.ndarray, within: list[int], groups: list | None = None,
           within_names=("A", "B"), between_name: str = "Group") -> dict:
    """Repeated-measures GLM: ``Y`` is subjects × K cells (factor A outermost when two
    within factors), ``groups`` the between-subjects level of each subject (or None).
    Returns the within-effects rows (with sphericity data) and between-effects rows."""
    Y = np.asarray(Y, float)
    n, K = Y.shape
    if int(np.prod(within)) != K:
        raise StatError(f"The within design needs {int(np.prod(within))} columns (got {K}).")
    if groups is not None:
        levels = st._sort_levels([str(g) for g in groups])
        gidx = np.array([levels.index(str(g)) for g in groups])
        G = len(levels)
        if G < 2:
            raise StatError("The between-subjects factor needs at least two levels.")
        X = np.column_stack([np.ones(n)] + [np.where(gidx == j, 1.0,
                                                     np.where(gidx == G - 1, -1.0, 0.0))
                                            for j in range(G - 1)])   # effect coding
        Lg = np.hstack([np.zeros((G - 1, 1)), np.eye(G - 1)])
    else:
        levels, G = [], 1
        X = np.ones((n, 1))
        Lg = None
    rank = np.linalg.matrix_rank(X)
    dfe = n - rank
    if dfe < 1:
        raise StatError("Not enough subjects for the error term.")
    L0 = np.zeros((1, X.shape[1]))
    L0[0, 0] = 1.0
    ones = [np.ones((k, 1)) / math.sqrt(k) for k in within]
    contr = [poly_contrasts(k) for k in within]
    if len(within) == 1:
        effects = [(within_names[0], contr[0])]
    else:
        effects = [(within_names[0], np.kron(contr[0], ones[1])),
                   (within_names[1], np.kron(ones[0], contr[1])),
                   (f"{within_names[0]} * {within_names[1]}", np.kron(contr[0], contr[1]))]
    out = {"within": [], "between": [], "dfe": dfe, "levels": levels, "n": n}
    for name, M in effects:
        Yt = Y @ M
        d = M.shape[1]
        XtXi = np.linalg.pinv(X.T @ X)
        B = XtXi @ X.T @ Yt
        R = Yt - X @ B
        E = R.T @ R
        hyps = [(name, L0)]
        if Lg is not None:
            hyps.append((f"{name} * {between_name}", Lg))
        gg, hf, lb = epsilons(E, dfe) if d > 1 else (1.0, 1.0, 1.0)
        W, chi, mdf, mp = mauchly(E, dfe) if d > 1 else (1.0, 0.0, 0, math.nan)
        sse = float(np.trace(E))
        for hname, L in hyps:
            H = _glh(X, Yt, L)
            dfh = np.linalg.matrix_rank(L)
            ssh = float(np.trace(H))
            f = (ssh / (d * dfh)) / (sse / (d * dfe)) if sse > 0 else math.nan
            rows = {}
            for cname, eps in (("Sphericity Assumed", 1.0), ("Greenhouse-Geisser", gg),
                               ("Huynh-Feldt", hf), ("Lower-bound", lb)):
                d1, d2 = d * dfh * eps, d * dfe * eps
                rows[cname] = (d1, d2, float(stats.f.sf(f, d1, d2)))
            mv = multivariate_tests(H, E, dfh, dfe)
            out["within"].append({"effect": hname, "ss": ssh, "df": d * dfh, "f": f,
                                  "ss_error": sse, "df_error": d * dfe, "rows": rows,
                                  "eps": (gg, hf, lb), "mauchly": (W, chi, mdf, mp),
                                  "mv": mv, "base": name, "d": d,
                                  "eta_p": ssh / (ssh + sse) if ssh + sse > 0 else math.nan})
    if Lg is not None:
        yb = Y.sum(1) / math.sqrt(K)
        H = float(_glh(X, yb[:, None], Lg)[0, 0])
        XtXi = np.linalg.pinv(X.T @ X)
        res = yb - X @ (XtXi @ X.T @ yb)
        E = float(res @ res)
        dfh = G - 1
        f = (H / dfh) / (E / dfe)
        out["between"].append({"effect": between_name, "ss": H, "df": dfh, "f": f,
                               "p": float(stats.f.sf(f, dfh, dfe)), "ss_error": E,
                               "df_error": dfe})
    return out


def _rm_tables(fresh, res: dict, alpha: float) -> list[str]:
    lines, sph, rows, mvrows = [], [], [], []
    done_bases = set()
    for e in res["within"]:
        if e["base"] not in done_bases and e["d"] > 1:
            done_bases.add(e["base"])
            W, chi, mdf, mp = e["mauchly"]
            gg, hf, lb = e["eps"]
            sph.append([e["base"], num(W, 5), num(chi, 5), str(mdf), pval(mp), num(gg, 5),
                        num(hf, 5), num(lb, 5)])
        for cname, (d1, d2, p) in e["rows"].items():
            rows.append([e["effect"] if cname == "Sphericity Assumed" else "", cname,
                         num(d1), num(e["ss"]), num(e["ss"] / d1), num(e["f"], 5), pval(p)])
        for cname, (d1, d2, p) in e["rows"].items():
            rows.append([f"Error({e['base']})" if cname == "Sphericity Assumed" else "",
                         cname, num(d2), num(e["ss_error"]), num(e["ss_error"] / d2), "", ""])
        lines.append(f"  {e['effect']}: F({num(e['df'])}, {num(e['df_error'])}) = "
                     f"{num(e['f'], 5)}, p = {pval(e['rows']['Sphericity Assumed'][2])}, "
                     f"p(GG) = {pval(e['rows']['Greenhouse-Geisser'][2])}")
        for t in e["mv"]:
            mvrows.append([e["effect"] if t is e["mv"][0] else "", t.name, num(t.value, 5),
                           num(t.F, 5), num(t.df1), num(t.df2), pval(t.p)])
    if sph:
        fresh.add_table("Mauchly's Test of Sphericity",
                        ["Within Effect", "Mauchly's W", "Approx. Chi-Square", "DF", "Prob",
                         "Epsilon (Greenhouse-Geisser)", "Epsilon (Huynh-Feldt)",
                         "Epsilon (Lower-bound)"], sph,
                        [f"At the {num(alpha)} level, Prob < {num(alpha)} means sphericity is "
                         "violated; use the corrected tests."])
    fresh.add_table("Tests of Within-Subjects Effects",
                    ["Source", "Correction", "DF", "Sum of Squares", "Mean Square", "F Value",
                     "Prob>F"], rows)
    fresh.add_table("Multivariate Tests", ["Effect", "Test", "Value", "F Value", "Num DF",
                                           "Den DF", "Prob>F"], mvrows)
    if res["between"]:
        b = res["between"][0]
        fresh.add_table("Tests of Between-Subjects Effects",
                        ["Source", "DF", "Sum of Squares", "Mean Square", "F Value", "Prob>F"],
                        [[b["effect"], str(b["df"]), num(b["ss"]), num(b["ss"] / b["df"]),
                          num(b["f"], 5), pval(b["p"])],
                         ["Error", str(b["df_error"]), num(b["ss_error"]),
                          num(b["ss_error"] / b["df_error"]), "", ""]])
        lines.append(f"  {b['effect']}: F = {num(b['f'], 5)}, p = {pval(b['p'])}")
    return lines


def _paired_bonferroni(names, M: np.ndarray, alpha: float) -> list[list[str]]:
    k = M.shape[1]
    m = k * (k - 1) // 2
    out = []
    for i in range(k):
        for j in range(i):
            d = M[:, i] - M[:, j]
            n = len(d)
            se = d.std(ddof=1) / math.sqrt(n)
            t = d.mean() / se if se > 0 else math.nan
            p = min(1.0, 2 * stats.t.sf(abs(t), n - 1) * m)
            q = stats.t.ppf(1 - alpha / (2 * m), n - 1)
            out.append([f"{names[i]}  {names[j]}", num(d.mean()), num(se), num(t, 5), pval(p),
                        "1" if p < alpha else "0", num(d.mean() - q * se),
                        num(d.mean() + q * se)])
    return out


def rmanova1_report(books: dict, settings: dict, report: ReportSheet | None = None,
                    mode: str = "manual"):
    inputs = st.read_ranges(books, settings.get("ranges", []))
    if len(inputs) < 2:
        raise StatError("One-Way Repeated Measures ANOVA needs one column per level (2 or "
                        "more).")
    alpha = float(settings.get("alpha", 0.05))
    m, ok = aligned(inputs)
    Y = m[ok]
    if len(Y) < 3:
        raise StatError("Not enough complete subjects (rows).")
    names = [r.title for r in inputs]
    res = rm_glm(Y, [len(inputs)], None, (settings.get("name", "") or "Factor",))
    first = inputs[0]
    report, fresh = new_report(first, f"One-Way Repeated Measures ANOVA of {first.text} …",
                               "RMANOVA1", report, "rmanova1", settings, mode)
    st._notes(fresh, "One-Way Repeated Measures ANOVA", inputs,
              [["Subjects", f"{len(Y)} (complete rows)"], ["Significance Level", num(alpha)]])
    fresh.add_table("Descriptive Statistics", ["", "N", "Mean", "SD", "SEM"],
                    [st._desc_row(nm, Y[:, j]) for j, nm in enumerate(names)])
    lines = _rm_tables(fresh, res, alpha)
    fresh.add_table("Bonferroni Test", ["", "MeanDiff", "SEM", "t Value", "Prob", "Sig", "LCL",
                                        "UCL"], _paired_bonferroni(names, Y, alpha),
                    ["Paired comparisons with Bonferroni-adjusted probabilities and limits."],
                    group="Means Comparisons")
    fresh.add_graph("Means Plot", rg.means_graph([(nm, Y[:, j]) for j, nm in
                                                  enumerate(names)]), width=420)
    report = st._finish(report, fresh)
    return report, report_text(first.book, "One-Way Repeated Measures ANOVA", report, lines)


def rmanova2_report(books: dict, settings: dict, report: ReportSheet | None = None,
                    mode: str = "manual"):
    inputs = st.read_ranges(books, settings.get("ranges", []))
    alpha = float(settings.get("alpha", 0.05))
    design = settings.get("design", "within")
    a_name = (settings.get("name_a") or "Factor A").strip()
    if design == "within":
        na, nb = int(settings.get("levels_a", 2)), int(settings.get("levels_b", 2))
        if na * nb != len(inputs):
            raise StatError(f"Two within factors need {na} × {nb} = {na * nb} columns "
                            f"(A1B1, A1B2, …; got {len(inputs)}).")
        b_name = (settings.get("name_b") or "Factor B").strip()
        m, ok = aligned(inputs)
        Y = m[ok]
        groups = None
        extra_inputs = []
        within = [na, nb]
    else:
        grp = st.read_range(books, settings.get("group", ""))
        m, ok = aligned(inputs)
        n = min(len(m), len(grp.texts))
        ok = ok[:n] & np.array([grp.texts[i] != "" for i in range(n)])
        Y = m[:n][ok]
        groups = [grp.texts[i] for i in range(n) if ok[i]]
        b_name = grp.title
        extra_inputs = [grp]
        within = [len(inputs)]
    if len(Y) < 3:
        raise StatError("Not enough complete subjects (rows).")
    res = rm_glm(Y, within, groups, (a_name, b_name if design == "within" else ""), b_name)
    first = inputs[0]
    report, fresh = new_report(first, f"Two-Way Repeated Measures ANOVA of {first.text} …",
                               "RMANOVA2", report, "rmanova2", settings, mode)
    st._notes(fresh, "Two-Way Repeated Measures ANOVA", inputs + extra_inputs,
              [["Design", "Both factors within subjects" if design == "within" else
                f"{a_name} within, {b_name} between subjects"],
               ["Subjects", str(len(Y))], ["Significance Level", num(alpha)]])
    if design == "within":
        la = st.level_names(settings.get("names_a", ""), within[0], "A")
        lb = st.level_names(settings.get("names_b", ""), within[1], "B")
        desc = [[f"{la[k // within[1]]}  {lb[k % within[1]]}"] + st._desc_row("", Y[:, k])[1:]
                for k in range(Y.shape[1])]
        means = Y.mean(0).reshape(within[0], within[1])
        graph = rg.interaction_graph(la, lb, means, a_name, b_name, "Data")
    else:
        la = [r.title for r in inputs]
        lb = res["levels"]
        desc = []
        means = np.zeros((len(la), len(lb)))
        for j, lv in enumerate(lb):
            sel = np.array([g == lv for g in groups])
            for k in range(len(la)):
                desc.append([f"{la[k]}  {lv}"] + st._desc_row("", Y[sel, k])[1:])
                means[k, j] = Y[sel, k].mean()
        graph = rg.interaction_graph(la, [str(v) for v in lb], means, a_name, b_name, "Data")
    fresh.add_table("Descriptive Statistics", ["Cell", "N", "Mean", "SD", "SEM"], desc)
    lines = _rm_tables(fresh, res, alpha)
    fresh.add_graph("Interaction Plot", graph, width=420)
    report = st._finish(report, fresh)
    return report, report_text(first.book, "Two-Way Repeated Measures ANOVA", report, lines)


# ====================================================================== MANOVA
def manova_report(books: dict, settings: dict, report: ReportSheet | None = None,
                  mode: str = "manual"):
    inputs = st.read_ranges(books, settings.get("ranges", []))
    grp = st.read_range(books, settings.get("group", ""))
    if len(inputs) < 2:
        raise StatError("MANOVA needs two or more dependent variables.")
    alpha = float(settings.get("alpha", 0.05))
    m, ok = aligned(inputs)
    n = min(len(m), len(grp.texts))
    ok = ok[:n] & np.array([grp.texts[i] != "" for i in range(n)])
    Y = m[:n][ok]
    labels = [grp.texts[i] for i in range(n) if ok[i]]
    levels = st._sort_levels(labels)
    if len(levels) < 2:
        raise StatError("The group column needs at least two levels.")
    G = len(levels)
    gi = np.array([levels.index(v) for v in labels])
    X = np.column_stack([np.ones(len(Y))] + [np.where(gi == j, 1.0, np.where(gi == G - 1, -1.0,
                                                                              0.0))
                                             for j in range(G - 1)])
    dfe = len(Y) - G
    if dfe < Y.shape[1]:
        raise StatError("Not enough observations for the number of dependent variables.")
    L = np.hstack([np.zeros((G - 1, 1)), np.eye(G - 1)])
    H = _glh(X, Y, L)
    B = np.linalg.pinv(X) @ Y
    R = Y - X @ B
    E = R.T @ R
    tests = multivariate_tests(H, E, G - 1, dfe)
    first = inputs[0]
    report, fresh = new_report(first, f"One-Way MANOVA of {first.text} …", "MANOVA", report,
                               "manova", settings, mode)
    st._notes(fresh, "One-Way MANOVA (multivariate tests)", inputs + [grp],
              [["Groups", ", ".join(levels)], ["N", str(len(Y))],
               ["Significance Level", num(alpha)]])
    desc = []
    for j, lv in enumerate(levels):
        for k, r in enumerate(inputs):
            v = Y[gi == j, k]
            desc.append([r.title if j == 0 else "", lv] + st._desc_row("", v)[1:])
    fresh.add_table("Descriptive Statistics", ["Variable", "Group", "N", "Mean", "SD", "SEM"],
                    desc)
    fresh.add_table("Multivariate Tests", ["Test", "Value", "F Value", "Num DF", "Den DF",
                                           "Prob>F", f"Decision at level({lvl(alpha)})"],
                    [[t.name, num(t.value, 5), num(t.F, 5), num(t.df1), num(t.df2), pval(t.p),
                      decision(t.p, alpha, "Means differ", "No difference")] for t in tests])
    bm = box_m([Y[gi == j] for j in range(G)])
    if bm is not None:
        fresh.add_table("Box's Test of Equality of Covariance Matrices",
                        ["Box's M", "F", "DF1", "DF2", "Prob"],
                        [[num(bm[0], 5), num(bm[1], 5), num(bm[2]), num(bm[3]), pval(bm[4])]])
    uni = []
    for k, r in enumerate(inputs):
        f, p = stats.f_oneway(*[Y[gi == j, k] for j in range(G)])
        uni.append([r.title, str(G - 1), str(dfe), num(f, 5), pval(p)])
    fresh.add_table("Univariate ANOVA", ["Variable", "DF1", "DF2", "F Value", "Prob>F"], uni)
    report = st._finish(report, fresh)
    w = tests[1]
    return report, report_text(first.book, "One-Way MANOVA", report,
                               [f"  Wilks' Lambda = {num(w.value, 5)}, F = {num(w.F, 5)}, "
                                f"p = {pval(w.p)}"])


# ====================================================================== nonparametric
def wilcoxon1_report(books: dict, settings: dict, report: ReportSheet | None = None,
                     mode: str = "manual"):
    inputs = st.read_ranges(books, settings.get("ranges", []))
    m0 = float(settings.get("median", 0.0))
    tail = settings.get("tail", "two")
    alpha = float(settings.get("alpha", 0.05))
    first = inputs[0]
    report, fresh = new_report(first, f"One-Sample Wilcoxon Signed Rank Test of {first.text}",
                               "Wilcoxon1", report, "wilcoxon1", settings, mode)
    st._notes(fresh, "One-Sample Wilcoxon Signed Rank Test", inputs,
              [["Null Hypothesis", f"Median = {num(m0)}"],
               ["Alternative Hypothesis", f"Median {st._SYM[tail]} {num(m0)}"],
               ["Significance Level", num(alpha)]])
    rows, est, lines = [], [], []
    for r in inputs:
        d = r.finite - m0
        w = npt.wilcoxon_test(d)
        if w["n"] < 1:
            raise StatError(f"{r.title}: all values equal the test median.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = stats.wilcoxon(d[d != 0], alternative=st._ALT[tail],
                                 method="exact" if w["n"] <= 50 and
                                 len(np.unique(np.abs(d[d != 0]))) == w["n"] else "approx")
        zpos = w.get("z_pos", math.nan)
        p_asym = 2 * stats.norm.sf(abs(zpos)) if tail == "two" else \
            (stats.norm.sf(zpos) if tail == "upper" else stats.norm.cdf(zpos))
        rows.append([r.title, str(w["n"]), num(w["r_pos"]), num(w["r_neg"]), num(zpos, 5),
                     pval(p_asym), pval(res.pvalue),
                     decision(float(res.pvalue), alpha, "Reject H0", "Can't reject H0")])
        hl = npt.hodges_lehmann_one(r.finite, 1 - alpha)
        est.append([r.title, num(np.median(r.finite)), num(hl["estimate"]), num(hl["lower"]),
                    num(hl["upper"])])
        lines.append(f"  {r.title}: W+ = {num(w['r_pos'])}, p = {pval(res.pvalue)}")
    fresh.add_table("Test Statistics", ["", "N (non-zero)", "Sum of Positive Ranks (W+)",
                                        "Sum of Negative Ranks", "Z", "Asymptotic Prob",
                                        "Prob (exact when possible)",
                                        f"Decision at level({lvl(alpha)})"], rows,
                    ["Zero differences are dropped; the normal approximation uses the "
                     "tie-corrected variance."])
    fresh.add_table("Location Estimate", ["", "Median", "Hodges-Lehmann Estimate",
                                          f"Lower {100 * (1 - alpha):g}% CL",
                                          f"Upper {100 * (1 - alpha):g}% CL"], est)
    report = st._finish(report, fresh)
    return report, report_text(first.book, "One-Sample Wilcoxon Signed Rank Test", report,
                               lines)


def sign_report(books: dict, settings: dict, report: ReportSheet | None = None,
                mode: str = "manual"):
    r1 = st.read_range(books, settings.get("range1", ""))
    r2 = st.read_range(books, settings.get("range2", ""))
    tail = settings.get("tail", "two")
    alpha = float(settings.get("alpha", 0.05))
    n = min(len(r1.values), len(r2.values))
    a, b = r1.values[:n], r2.values[:n]
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    s = npt.sign_test(d)
    if s["n"] == 0:
        raise StatError("All pairs are tied.")
    alt = {"two": "two-sided", "upper": "greater", "lower": "less"}[tail]
    p_exact = float(stats.binomtest(s["pos"], s["n"], 0.5, alternative=alt).pvalue)
    report, fresh = new_report(r1, f"Paired Sample Sign Test of {r1.title} and {r2.title}",
                               "SignTest", report, "sign", settings, mode)
    st._notes(fresh, "Paired Sample Sign Test", [r1, r2],
              [["Null Hypothesis", "Median of differences = 0"],
               ["Alternative Hypothesis", f"Median of differences {st._SYM[tail]} 0"],
               ["Significance Level", num(alpha)]])
    fresh.add_table("Frequencies", ["", "N"],
                    [["Positive Differences", str(s["pos"])],
                     ["Negative Differences", str(s["neg"])], ["Ties", str(s["ties"])],
                     ["Total", str(s["total"])]])
    fresh.add_table("Test Statistics", ["", "Z (continuity corrected)", "Exact Prob",
                                        f"Decision at level({lvl(alpha)})"],
                    [[f"{r1.title} − {r2.title}", num(s.get("z_np", math.nan), 5),
                      pval(p_exact), decision(p_exact, alpha, "Reject H0",
                                              "Can't reject H0")]],
                    ["Exact binomial probability of the positive differences."])
    report = st._finish(report, fresh)
    return report, report_text(r1.book, "Paired Sample Sign Test", report,
                               [f"  {s['pos']} positive, {s['neg']} negative, p = "
                                f"{pval(p_exact)}"])


def ks2_report(books: dict, settings: dict, report: ReportSheet | None = None,
               mode: str = "manual"):
    inputs, groups = groups_input(books, settings, 2)
    if len(groups) != 2:
        raise StatError("The two-sample Kolmogorov-Smirnov test compares two groups.")
    (n1, a), (n2, b) = groups
    alpha = float(settings.get("alpha", 0.05))
    k = npt.ks_two_sample_test(a, b)
    exact = stats.ks_2samp(a, b, method="exact" if len(a) * len(b) <= 10000 else "asymp")
    first = inputs[0]
    report, fresh = new_report(first, f"Two-Sample Kolmogorov-Smirnov Test of {n1} and {n2}",
                               "KS2Sample", report, "ks2", settings, mode)
    st._notes(fresh, "Two-Sample Kolmogorov-Smirnov Test", inputs,
              [["Null Hypothesis", "The two samples come from the same distribution"],
               ["Significance Level", num(alpha)]])
    fresh.add_table("Descriptive Statistics", ["", "N", "Mean", "SD", "Median"],
                    [[nm, str(len(v)), num(v.mean()), num(v.std(ddof=1)), num(np.median(v))]
                     for nm, v in groups])
    fresh.add_table("Test Statistics", ["", "D", "D+", "D−", "Z", "Asymptotic Prob",
                                        "Exact Prob", f"Decision at level({lvl(alpha)})"],
                    [[f"{n1} − {n2}", num(k["absolute"], 5), num(k["positive"], 5),
                      num(k["negative"], 5), num(k["z"], 5), pval(k["p"]),
                      pval(float(exact.pvalue)),
                      decision(float(exact.pvalue), alpha, "Distributions differ",
                               "Can't reject H0")]])
    g = rg.new_graph("Empirical CDF", 11.0, 8.0)
    for i, (nm, v) in enumerate(groups):
        x = np.sort(v)
        y = np.arange(1, len(x) + 1) / len(x)
        xs = np.repeat(x, 2)[1:]
        ys = np.repeat(y, 2)[:-1]
        rg.add_series(g, np.r_[x[0], xs], np.r_[0.0, ys], nm, "line",
                      rg.PALETTES.get("Origin Classic")[i], width=1.5)
    fresh.add_graph("Empirical CDF", rg.finish(g, "Value", "Cumulative Probability",
                                               legend=True), width=400)
    report = st._finish(report, fresh)
    return report, report_text(first.book, "Two-Sample Kolmogorov-Smirnov Test", report,
                               [f"  D = {num(k['absolute'], 5)}, p = {pval(exact.pvalue)}"])


def median_report(books: dict, settings: dict, report: ReportSheet | None = None,
                  mode: str = "manual"):
    inputs, groups = groups_input(books, settings, 2)
    alpha = float(settings.get("alpha", 0.05))
    res = npt.median_test([v for _, v in groups])
    first = inputs[0]
    report, fresh = new_report(first, f"Mood's Median Test of {first.text}", "MedianTest",
                               report, "median", settings, mode)
    st._notes(fresh, "Mood's Median Test", inputs,
              [["Pooled Median", num(res["median"])], ["Significance Level", num(alpha)]])
    fresh.add_table("Frequencies", ["", "N", "Median", "> Pooled Median", "≤ Pooled Median"],
                    [[nm, str(len(v)), num(np.median(v)), str(int(res["above"][i])),
                      str(int(res["below"][i]))] for i, (nm, v) in enumerate(groups)])
    rows = [["Chi-Square", num(res["chi2"], 5), str(res["df"]), pval(res["p"]),
             decision(res["p"], alpha, "Medians differ", "Can't reject H0")]]
    if "p_exact" in res:
        rows.append(["Fisher's Exact Test", "", "", pval(res["p_exact"]),
                     decision(res["p_exact"], alpha, "Medians differ", "Can't reject H0")])
    fresh.add_table("Test Statistics", ["", "Value", "DF", "Prob",
                                        f"Decision at level({lvl(alpha)})"], rows)
    report = st._finish(report, fresh)
    return report, report_text(first.book, "Mood's Median Test", report,
                               [f"  χ² = {num(res['chi2'], 5)}, p = {pval(res['p'])}"])


# ====================================================================== dialog specs
def _groups_input(sel, label="Data (one column per group)", two=False):
    first = sel[0] if sel else ""
    second = sel[1] if len(sel) > 1 else ""
    return [("choice", "form", "Input Data Form", "raw", FORMS),
            ("ranges", "ranges", label, sel[:2] if two else sel, {"when": ("form", "raw")}),
            ("range", "data", "Data Range", first, {"when": ("form", "indexed")}),
            ("range", "factor", "Group Range", second, {"when": ("form", "indexed")})]


SPECS = {
    "grubbs": lambda sel: ("Grubbs Test", [
        ("Input", [("ranges", "ranges", "Input Data", sel)]),
        ("Options", [("choice", "side", "Alternative", "two", GRUBBS_SIDES), alpha_field(),
                     ("bool", "iterative", "Iterative (remove outliers and repeat)", False)])],
        "Detects one outlier in normally distributed data (repeat to find more).",
        "report sheet"),
    "dixon": lambda sel: ("Dixon's Q Test", [
        ("Input", [("ranges", "ranges", "Input Data (3 ≤ N ≤ 200)", sel)]),
        ("Options", [("choice", "kind", "Statistic", "auto", DIXON_KIND_CHOICES),
                     ("choice", "side", "Alternative", "two",
                      {"two": "Two-sided (more extreme end)", "max": "Highest value",
                       "min": "Lowest value"}), alpha_field(),
                     ("choice", "critical", "Critical Values", "exact",
                      {"exact": "Exact (computed)",
                       "table": "Rorabacher (1991) table where available"})])],
        "Dixon's ratio test for one outlier in a small sample.", "report sheet"),
    "distfit": lambda sel: ("Distribution Fit", [
        ("Input", [("range", "range", "Input Data", sel[0] if sel else "")]),
        ("Distributions", [("checks", "dists", "", ["normal", "lognormal", "weibull", "gamma"],
                            dfit.DISTRIBUTIONS)]),
        ("Goodness of Fit", [alpha_field(),
                             ("int", "bootstrap", "Bootstrap Samples for P-values (0 = none)",
                              0, (0, 10000))])],
        "Maximum-likelihood fit of parametric distributions with goodness-of-fit tests.",
        "report sheet"),
    "partialcorr": lambda sel: ("Partial Correlation", [
        ("Input", [("ranges", "ranges", "Variables", sel[:-1] if len(sel) > 2 else sel),
                   ("ranges", "covariates", "Controlling Variables",
                    sel[-1:] if len(sel) > 2 else [])]),
        ("Options", [alpha_field()])],
        "Correlations between variables after removing the linear effect of the controls.",
        "report sheet"),
    "crosstab": lambda sel: ("Cross Tabulation", [
        ("Input", [("range", "range1", "Row Variable", sel[0] if sel else ""),
                   ("range", "range2", "Column Variable", sel[1] if len(sel) > 1 else "")]),
        ("Options", [alpha_field()])],
        "Contingency table of two categorical columns with chi-square tests.", "report sheet"),
    "var1": lambda sel: ("One-Sample Test for Variance", [
        ("Input", [("ranges", "ranges", "Input Data", sel[:1])]),
        ("Hypotheses", [("float", "variance", "Test Variance", 1.0, (1e-12, 1e12, 6)),
                        ("choice", "tail", "Alternate Hypothesis", "two", TAILS)]),
        ("Options", [alpha_field()])],
        "Chi-square test of the population variance.", "report sheet"),
    "var2": lambda sel: ("Two-Sample Test for Variance", [
        ("Input", _groups_input(sel, "Data (two columns)", True)),
        ("Hypotheses", [("float", "ratio", "Variance Ratio", 1.0, (1e-12, 1e12, 6)),
                        ("choice", "tail", "Alternate Hypothesis", "two", TAILS)]),
        ("Options", [alpha_field()])],
        "F-test of the ratio of two variances, with Levene, Brown-Forsythe and Bartlett.",
        "report sheet"),
    "vark": lambda sel: ("Tests for Equal Variance", [
        ("Input", _groups_input(sel)), ("Options", [alpha_field()])],
        "Bartlett, Levene and Brown-Forsythe tests for k samples.", "report sheet"),
    "prop1": lambda sel: ("One-Sample Test for Proportion", [
        ("Input", [("choice", "form", "Input", "summary",
                    {"summary": "Summary (successes, sample size)",
                     "data": "Data column"}),
                   ("int", "x", "Number of Successes", 10, (0, 10 ** 9),
                    {"when": ("form", "summary")}),
                   ("int", "n", "Sample Size", 20, (1, 10 ** 9), {"when": ("form", "summary")}),
                   ("range", "range", "Data Range", sel[0] if sel else "",
                    {"when": ("form", "data")}),
                   ("text", "success", "Success Value", "1", {"when": ("form", "data")})]),
        ("Hypotheses", [("float", "p0", "Test Proportion", 0.5, (0.0001, 0.9999, 4)),
                        ("choice", "tail", "Alternate Hypothesis", "two", TAILS)]),
        ("Options", [("choice", "method", "Test", "exact", P_TESTS1),
                     ("choice", "ci", "Confidence Interval", "clopper_pearson", P_CI1),
                     alpha_field()])],
        "Tests a population proportion (exact binomial or normal approximation). With "
        "summary input the report goes to the active workbook.", "report sheet"),
    "prop2": lambda sel: ("Two-Sample Test for Proportions", [
        ("Input", [("choice", "form", "Input", "summary",
                    {"summary": "Summary (successes, sample sizes)",
                     "data": "Data columns"}),
                   ("int", "x1", "Successes 1", 10, (0, 10 ** 9), {"when": ("form", "summary")}),
                   ("int", "n1", "Sample Size 1", 20, (1, 10 ** 9),
                    {"when": ("form", "summary")}),
                   ("int", "x2", "Successes 2", 5, (0, 10 ** 9), {"when": ("form", "summary")}),
                   ("int", "n2", "Sample Size 2", 20, (1, 10 ** 9),
                    {"when": ("form", "summary")}),
                   ("range", "range1", "Data Range 1", sel[0] if sel else "",
                    {"when": ("form", "data")}),
                   ("range", "range2", "Data Range 2", sel[1] if len(sel) > 1 else "",
                    {"when": ("form", "data")}),
                   ("text", "success", "Success Value", "1", {"when": ("form", "data")})]),
        ("Hypotheses", [("choice", "tail", "Alternate Hypothesis", "two", TAILS)]),
        ("Options", [("choice", "method", "Test", "wald_h0", P_TESTS2),
                     ("choice", "ci", "Confidence Interval", "newcombe", P_CI2),
                     alpha_field()])],
        "Compares two independent proportions (z test and Fisher's exact test). With "
        "summary input the report goes to the active workbook.", "report sheet"),
    "anova3": lambda sel: ("Three-Way ANOVA", [
        ("Input", [("range", "data", "Data Range", sel[3] if len(sel) > 3 else ""),
                   ("range", "factor_a", "Factor A Range", sel[0] if sel else ""),
                   ("range", "factor_b", "Factor B Range", sel[1] if len(sel) > 1 else ""),
                   ("range", "factor_c", "Factor C Range", sel[2] if len(sel) > 2 else "")]),
        ("Means Comparison", [("checks", "means", "", ["tukey"],
                               {k: v for k, v in st.MEANS_METHODS.items() if k != "dunnett"})]),
        ("Options", [alpha_field(),
                     ("choice", "interaction", "Interactions", "all",
                      {"all": "All (2- and 3-way)", "two": "Two-way only", "none": "None"}),
                     ("choice", "ss", "Type of Sums of Squares", "3", st.SS_TYPES)])],
        "Main effects and interactions of three factors (indexed data).", "report sheet"),
    "rmanova1": lambda sel: ("One-Way Repeated Measures ANOVA", [
        ("Input", [("ranges", "ranges", "Data (one column per level; rows = subjects)", sel),
                   ("text", "name", "Factor Name", "Factor")]),
        ("Options", [alpha_field()])],
        "Within-subjects factor with Mauchly's test and sphericity corrections.",
        "report sheet"),
    "rmanova2": lambda sel: ("Two-Way Repeated Measures ANOVA", [
        ("Input", [("choice", "design", "Design", "within",
                    {"within": "Both factors within subjects",
                     "mixed": "Factor A within, Factor B between (group column)"}),
                   ("ranges", "ranges", "Data (columns A1B1, A1B2, …, or one per A level)",
                    sel),
                   ("text", "name_a", "Factor A Name", "Factor A"),
                   ("int", "levels_a", "Factor A Levels", 2, (2, 50),
                    {"when": ("design", "within")}),
                   ("text", "names_a", "Factor A Level Names", "A1, A2",
                    {"when": ("design", "within")}),
                   ("text", "name_b", "Factor B Name", "Factor B",
                    {"when": ("design", "within")}),
                   ("int", "levels_b", "Factor B Levels", 2, (2, 50),
                    {"when": ("design", "within")}),
                   ("text", "names_b", "Factor B Level Names", "B1, B2",
                    {"when": ("design", "within")}),
                   ("range", "group", "Group Range (between factor)", "",
                    {"when": ("design", "mixed")})]),
        ("Options", [alpha_field()])],
        "Two factors with repeated measures on one or both.", "report sheet"),
    "manova": lambda sel: ("One-Way MANOVA", [
        ("Input", [("ranges", "ranges", "Dependent Variables", sel[:-1] if len(sel) > 2
                    else sel),
                   ("range", "group", "Group Range", sel[-1] if len(sel) > 2 else "")]),
        ("Options", [alpha_field()])],
        "Pillai's trace, Wilks' lambda, Hotelling-Lawley trace and Roy's largest root.",
        "report sheet"),
    "wilcoxon1": lambda sel: ("One-Sample Wilcoxon Signed Rank Test", [
        ("Input", [("ranges", "ranges", "Input Data", sel[:1])]),
        ("Hypotheses", [("float", "median", "Test Median", 0.0, (-1e12, 1e12, 6)),
                        ("choice", "tail", "Alternate Hypothesis", "two", TAILS)]),
        ("Options", [alpha_field()])],
        "Is the population median equal to the test median?", "report sheet"),
    "sign": lambda sel: ("Paired Sample Sign Test", [
        ("Input", [("range", "range1", "1st Data Range", sel[0] if sel else ""),
                   ("range", "range2", "2nd Data Range", sel[1] if len(sel) > 1 else "")]),
        ("Hypotheses", [("choice", "tail", "Alternate Hypothesis", "two", TAILS)]),
        ("Options", [alpha_field()])],
        "Sign test of paired samples (median of the differences = 0).", "report sheet"),
    "ks2": lambda sel: ("Two-Sample Kolmogorov-Smirnov Test", [
        ("Input", _groups_input(sel, "Data (two columns)", True)), ("Options", [alpha_field()])],
        "Do two samples come from the same distribution?", "report sheet"),
    "median": lambda sel: ("Mood's Median Test", [
        ("Input", _groups_input(sel)), ("Options", [alpha_field()])],
        "Tests whether the groups have the same median.", "report sheet"),
}

FUNCS = {"grubbs": grubbs_report, "dixon": dixon_report, "distfit": distfit_report,
         "partialcorr": partial_corr_report, "crosstab": crosstab_report, "var1": var1_report,
         "var2": var2_report, "vark": vark_report, "prop1": prop1_report,
         "prop2": prop2_report, "anova3": anova3_report, "rmanova1": rmanova1_report,
         "rmanova2": rmanova2_report, "manova": manova_report, "wilcoxon1": wilcoxon1_report,
         "sign": sign_report, "ks2": ks2_report, "median": median_report}

for _k, _f in FUNCS.items():
    st.register(_k, _f, SPECS[_k])

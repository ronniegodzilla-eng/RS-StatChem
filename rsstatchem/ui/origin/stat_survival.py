"""Origin's Statistics > Survival Analysis with report sheets (no Qt).

* Kaplan-Meier Estimator (:func:`rsstatchem.core.stats.survival.km_fit`): survival table
  with standard errors (Greenwood) and confidence limits (log-log or linear), quartiles
  (Collett's SE), restricted mean, and the log-rank / Breslow / Tarone-Ware comparison of
  groups (:func:`survival.compare_groups`); survival curves as live step graphs.
* Cox Proportional Hazards Model (:func:`survival.cox_fit`, Breslow ties): coefficients,
  hazard ratios with confidence limits, likelihood-ratio / score / Wald tests, and the
  survival curve at the covariate means (Breslow baseline).
* Weibull Fit Model (:mod:`rsstatchem.core.analysis.weibull_surv`): censored MLE per group,
  likelihood-ratio tests, fitted curves over the Kaplan-Meier estimate and the Weibull
  probability plot.

Censoring follows Origin's dialogs: a *Censor* column and the *Censored Value* that marks
censored observations; every other value of the censor column is an event.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

from rsstatchem.core.analysis import weibull_surv as wb
from rsstatchem.core.analysis.common import num, pval
from rsstatchem.core.graph.themes import PALETTES
from rsstatchem.core.stats import survival as sv
from rsstatchem.ui.origin import report_graphs as rg
from rsstatchem.ui.origin import stat_tests as st
from rsstatchem.ui.origin.model import ReportSheet, to_float
from rsstatchem.ui.origin.stat_extra import aligned, new_report, report_text

StatError = st.StatError
PAL = PALETTES.get("Origin Classic")


def _is_value(text: str, target: str) -> bool:
    a, b = to_float(text), to_float(target)
    if np.isfinite(a) and np.isfinite(b):
        return a == b
    return text.strip() == target.strip()


def survival_data(books: dict, settings: dict, covariates: bool = False):
    """(inputs, t, d, groups or None, X or None) — complete rows only."""
    tr = st.read_range(books, settings.get("time", ""))
    cr = st.read_range(books, settings.get("censor", ""))
    cval = str(settings.get("censor_value", "0"))
    inputs = [tr, cr]
    n = min(len(tr.values), len(cr.texts))
    ok = np.isfinite(tr.values[:n]) & np.array([cr.texts[i] != "" for i in range(n)])
    g = None
    if settings.get("group", "").strip():
        gr = st.read_range(books, settings["group"])
        inputs.append(gr)
        n = min(n, len(gr.texts))
        ok = ok[:n] & np.array([gr.texts[i] != "" for i in range(n)])
    X = None
    if covariates:
        cov = [st.read_range(books, t) for t in settings.get("covariates", []) if t.strip()]
        if not cov:
            raise StatError("Select one or more covariates.")
        inputs += cov
        m, okc = aligned(cov)
        n = min(n, len(m))
        ok = ok[:n] & okc[:n]
        X = m[:n][ok]
    rows = np.flatnonzero(ok)
    t = tr.values[rows]
    d = np.array([0.0 if _is_value(cr.texts[i], cval) else 1.0 for i in rows])
    if g is None and len(inputs) > 2 and not covariates:
        g = [inputs[2].texts[i] for i in rows]
    if len(t) < 2:
        raise StatError("Not enough complete observations.")
    if np.any(t < 0):
        raise StatError("Survival times must not be negative.")
    if d.sum() == 0:
        raise StatError(f"Every observation is censored (censored value “{cval}”).")
    return inputs, t, d, g, X


def _ci(s: np.ndarray, se: np.ndarray, level: float, how: str):
    z = stats.norm.ppf(0.5 + level / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        if how == "linear":
            return np.clip(s - z * se, 0, 1), np.clip(s + z * se, 0, 1)
        ls = np.log(-np.log(s))
        h = z * se / (s * np.abs(np.log(s)))
        lo, hi = np.exp(-np.exp(ls + h)), np.exp(-np.exp(ls - h))
        lo = np.where((s >= 1) | (s <= 0), s, lo)
        hi = np.where((s >= 1) | (s <= 0), s, hi)
    return lo, hi


def _steps(c: sv.KMCurve) -> tuple[np.ndarray, np.ndarray]:
    tmax = c.last_time
    return sv._step_xy(c.times, c.surv, 1.0, tmax)


def km_graph(curves: list[tuple[str, sv.KMCurve]], title: str, xtitle: str,
             ci: list | None = None) -> object:
    g = rg.new_graph(title, 12.0, 8.5)
    for i, (name, c) in enumerate(curves):
        col = PAL[i % len(PAL)]
        x, y = _steps(c)
        rg.add_series(g, x, y, name, "line", col, width=1.6, legend=len(curves) > 1)
        cens = c.t[c.d == 0]
        if len(cens):
            rg.add_series(g, cens, c.at(cens), f"{name} censored", "scatter", col,
                          symbol="+", size=7.0, legend=False)
        if ci is not None:
            lo, hi = ci[i]
            for arr, label in ((lo, "Lower"), (hi, "Upper")):
                xs, ys = sv._step_xy(c.times, arr, 1.0, c.last_time)
                rg.add_series(g, xs, ys, f"{name} {label} CL", "line", col, dash="--",
                              width=0.8, legend=False)
    lay = g.layers[0]
    lay.y.min, lay.y.max = 0.0, 1.05
    return rg.finish(g, xtitle, "Survival Probability", legend=len(curves) > 1)


CI_TYPES = {"loglog": "Log-log transformed", "linear": "Linear"}


def km_report(books: dict, settings: dict, report: ReportSheet | None = None,
              mode: str = "manual"):
    inputs, t, d, g, _ = survival_data(books, settings)
    level = float(settings.get("level", 0.95))
    how = settings.get("ci", "loglog")
    levels = st._sort_levels(g) if g is not None else ["All"]
    garr = np.array(g, dtype=object) if g is not None else np.array(["All"] * len(t),
                                                                    dtype=object)
    curves = [(lv, sv.km_fit(t[garr == lv], d[garr == lv])) for lv in levels]
    first = inputs[0]
    report, fresh = new_report(first, f"Kaplan-Meier Estimator of {first.text}", "KMEstimator",
                               report, "km", settings, mode)
    st._notes(fresh, "Kaplan-Meier Estimator", inputs,
              [["Censored Value", str(settings.get("censor_value", "0"))],
               ["Confidence Limits", f"{level * 100:g}%, {CI_TYPES.get(how, how)}"]])
    summ = []
    for lv, c in curves:
        cen = c.n - c.events
        summ.append([lv, str(c.n), str(c.events), str(cen), num(cen / c.n * 100, 4) + "%"])
    if len(curves) > 1:
        summ.append(["Total", str(len(t)), str(int(d.sum())), str(int(len(t) - d.sum())),
                     num((1 - d.mean()) * 100, 4) + "%"])
    fresh.add_table("Summary of Event and Censored Values", ["", "Total N", "Events",
                                                             "Censored", "Percent Censored"],
                    summ)
    cis = []
    for lv, c in curves:
        lo, hi = _ci(c.surv, c.se, level, how)
        cis.append((lo, hi))
        rows = [[num(c.times[k]), num(c.n_risk[k]), num(c.n_event[k]), num(c.surv[k], 5),
                 num(c.se[k], 5), num(lo[k], 5), num(hi[k], 5)] for k in range(len(c.times))]
        fresh.add_table(f"Survival Estimates ({lv})" if len(curves) > 1 else
                        "Survival Estimates", ["Time", "Number at Risk", "Events",
                                               "Survival Probability", "Standard Error",
                                               f"Lower {level * 100:g}% CL",
                                               f"Upper {level * 100:g}% CL"], rows,
                        group="Survival Estimates")
    z = stats.norm.ppf(0.5 + level / 2)
    qrows, mrows = [], []
    for lv, c in curves:
        for pc, label in ((0.25, "25%"), (0.5, "50% (Median)"), (0.75, "75%")):
            tp, se = sv.km_percentile(c, pc)
            qrows.append([lv, label, num(tp), num(se), num(tp - z * se), num(tp + z * se)])
        mean, se, restricted = sv.km_mean(c)
        mrows.append([lv, num(mean), num(se), num(mean - z * se), num(mean + z * se),
                      "Yes" if restricted else "No"])
    fresh.add_table("Quartile Estimates", ["", "Percent Failed", "Time", "Standard Error",
                                           f"Lower {level * 100:g}% CL",
                                           f"Upper {level * 100:g}% CL"], qrows,
                    ["Time by which the given percentage has failed (survival = 1 − percent); "
                     "SE by Collett's method."])
    fresh.add_table("Mean Survival Time", ["", "Mean", "Standard Error",
                                           f"Lower {level * 100:g}% CL",
                                           f"Upper {level * 100:g}% CL",
                                           "Restricted (largest time censored)"], mrows)
    lines = []
    if len(curves) > 1:
        rows = []
        for w, name in (("logrank", "Log-Rank"), ("breslow", "Breslow (Generalized "
                                                             "Wilcoxon)"),
                        ("tarone", "Tarone-Ware")):
            chi, df, p = sv.compare_groups(t, d, garr, weight=w, levels=levels)
            rows.append([name, num(chi, 5), str(df), pval(p)])
            lines.append(f"  {name}: χ² = {num(chi, 5)}, DF = {df}, p = {pval(p)}")
        fresh.add_table("Test of Equality over Groups", ["", "Chi-Square", "DF",
                                                         "Prob>Chi-Square"], rows)
    fresh.add_graph("Survival Function", km_graph(curves, "Survival Function",
                                                  inputs[0].column.axis_title,
                                                  cis if settings.get("ci_plot") else None),
                    width=460)
    report = st._finish(report, fresh)
    return report, report_text(first.book, "Kaplan-Meier Estimator", report, lines)


# ====================================================================== Cox
def breslow_survival(t, d, X, beta, at) -> tuple[np.ndarray, np.ndarray]:
    """(event times, S(t | x = at)) from the Breslow baseline cumulative hazard."""
    eta = (X - at) @ beta
    w = np.exp(eta)
    times = np.unique(t[d == 1])
    H = np.zeros(len(times))
    acc = 0.0
    for k, tk in enumerate(times):
        risk = w[t >= tk].sum()
        acc += float(((t == tk) & (d == 1)).sum()) / risk
        H[k] = acc
    return times, np.exp(-H)


def cox_report(books: dict, settings: dict, report: ReportSheet | None = None,
               mode: str = "manual"):
    inputs, t, d, _, X = survival_data(books, settings, covariates=True)
    level = float(settings.get("level", 0.95))
    names = [r.title for r in inputs[2:]]
    data = sv.CoxData(t, d, X)
    cols = list(range(X.shape[1]))
    fit = sv.cox_fit(data, cols)
    null = sv.cox_fit(data, [])
    first = inputs[0]
    report, fresh = new_report(first, f"Cox Proportional Hazards Model of {first.text}",
                               "CoxPH", report, "cox", settings, mode)
    st._notes(fresh, "Cox Proportional Hazards Model", inputs,
              [["Censored Value", str(settings.get("censor_value", "0"))],
               ["Ties", "Breslow"],
               ["Convergence", f"{'converged' if fit.converged else 'not converged'} in "
                               f"{fit.iterations} iterations"]])
    fresh.add_table("Summary of Event and Censored Values", ["Total N", "Events", "Censored",
                                                             "Percent Censored"],
                    [[str(len(t)), str(int(d.sum())), str(int(len(t) - d.sum())),
                      num((1 - d.mean()) * 100, 4) + "%"]])
    lr = 2 * (fit.ll - null.ll)
    df = len(cols)
    score = sv.score_statistic(data, cols, np.zeros(df))
    try:
        wald = float(fit.beta @ np.linalg.solve(fit.cov, fit.beta))
    except np.linalg.LinAlgError:
        wald = math.nan
    fresh.add_table("Model Fit Statistics", ["Criterion", "Without Covariates",
                                             "With Covariates"],
                    [["-2 Log Likelihood", num(-2 * null.ll), num(-2 * fit.ll)],
                     ["AIC", num(-2 * null.ll), num(-2 * fit.ll + 2 * df)]])
    fresh.add_table("Testing Global Null Hypothesis: Beta = 0", ["Test", "Chi-Square", "DF",
                                                                 "Prob>Chi-Square"],
                    [["Likelihood Ratio", num(lr, 5), str(df),
                      pval(float(stats.chi2.sf(lr, df)))],
                     ["Score", num(score, 5), str(df), pval(float(stats.chi2.sf(score, df)))],
                     ["Wald", num(wald, 5), str(df), pval(float(stats.chi2.sf(wald, df)))]])
    se = np.sqrt(np.clip(np.diag(fit.cov), 0, None))
    z = stats.norm.ppf(0.5 + level / 2)
    rows, lines = [], []
    for j, nm in enumerate(names):
        b, s = float(fit.beta[j]), float(se[j])
        w = (b / s) ** 2 if s > 0 else math.nan
        p = float(stats.chi2.sf(w, 1))
        rows.append([nm, num(b), num(s), num(w, 5), "1", pval(p), num(math.exp(b)),
                     num(math.exp(b - z * s)), num(math.exp(b + z * s))])
        lines.append(f"  {nm}: B = {num(b)}, HR = {num(math.exp(b))}, p = {pval(p)}")
    fresh.add_table("Parameter Estimates", ["", "Coefficient (B)", "Standard Error",
                                            "Wald Chi-Square", "DF", "Prob>Chi-Square",
                                            "Hazard Ratio",
                                            f"Lower {level * 100:g}% CL of HR",
                                            f"Upper {level * 100:g}% CL of HR"], rows)
    means = X.mean(0)
    times, s_at = breslow_survival(t, d, X, fit.beta, means)
    g = rg.new_graph("Survival at Covariate Means", 12.0, 8.5)
    xs, ys = sv._step_xy(times, s_at, 1.0, float(t.max()))
    rg.add_series(g, xs, ys, "Survival", "line", "#1f5fbf", width=1.6, legend=False)
    g.layers[0].y.min, g.layers[0].y.max = 0.0, 1.05
    fresh.add_graph("Survival Function at Covariate Means",
                    rg.finish(g, inputs[0].column.axis_title, "Survival Probability"),
                    width=440)
    fresh.add_table("Covariate Means", [""] + names, [["Mean"] + [num(v) for v in means]])
    report = st._finish(report, fresh)
    return report, report_text(first.book, "Cox Proportional Hazards Model", report, lines)


# ====================================================================== Weibull
def weibull_report(books: dict, settings: dict, report: ReportSheet | None = None,
                   mode: str = "manual"):
    inputs, t, d, g, _ = survival_data(books, settings)
    level = float(settings.get("level", 0.95))
    expo = bool(settings.get("exponential", False))
    if np.any(t <= 0):
        raise StatError("The Weibull model needs positive survival times.")
    try:
        if g is not None:
            res = wb.weibull_groups(t, d, g, level, expo)
            levels, fits = res["levels"], res["fits"]
        else:
            res = None
            levels, fits = ["All"], {"All": wb.weibull_fit(t, d, level, expo)}
    except wb.WeibullError as exc:
        raise StatError(str(exc)) from None
    first = inputs[0]
    report, fresh = new_report(first, f"Weibull Fit of {first.text}", "WeibullFit", report,
                               "weibull", settings, mode)
    st._notes(fresh, "Weibull Fit Model (maximum likelihood with right censoring)", inputs,
              [["Model", "Exponential (shape = 1)" if expo else
                "Weibull S(t) = exp(−(t/λ)^k)"],
               ["Censored Value", str(settings.get("censor_value", "0"))],
               ["Confidence Limits", f"{level * 100:g}% (Wald, log scale)"]])
    prm, stat_rows, lines = [], [], []
    for lv in levels:
        f = fits[lv]
        (klo, khi), (llo, lhi) = f.shape_ci, f.scale_ci
        prm.append([lv, "Shape (k)", num(f.shape), num(f.se_shape) if not expo else "(fixed)",
                    num(klo), num(khi)])
        prm.append([lv, "Scale (λ)", num(f.scale), num(f.se_scale), num(llo), num(lhi)])
        stat_rows.append([lv, str(f.n), str(f.events), num(f.loglik), num(f.aic),
                          num(f.median), num(f.mean)])
        lines.append(f"  {lv}: k = {num(f.shape)}, λ = {num(f.scale)}")
    fresh.add_table("Parameters", ["Group", "Parameter", "Value", "Standard Error",
                                   f"Lower {level * 100:g}% CL", f"Upper {level * 100:g}% CL"],
                    prm)
    fresh.add_table("Fit Statistics", ["Group", "N", "Events", "Log Likelihood", "AIC",
                                       "Median Survival", "Mean Survival"], stat_rows)
    if res is not None:
        rows = [["Same distribution in all groups", num(res["lr_equal"][0], 5),
                 str(res["lr_equal"][1]), pval(res["lr_equal"][2])]]
        if "lr_shape" in res:
            rows.append(["Common shape (separate scales)", num(res["lr_shape"][0], 5),
                         str(res["lr_shape"][1]), pval(res["lr_shape"][2])])
        fresh.add_table("Likelihood Ratio Tests", ["Null Hypothesis", "Chi-Square", "DF",
                                                   "Prob>Chi-Square"], rows)
    garr = np.array(g, dtype=object) if g is not None else np.array(["All"] * len(t),
                                                                    dtype=object)
    curves = [(lv, sv.km_fit(t[garr == lv], d[garr == lv])) for lv in levels]
    gr = km_graph(curves, "Survival Function", inputs[0].column.axis_title)
    tt = np.linspace(0, float(t.max()) * 1.05, 300)
    for i, lv in enumerate(levels):
        rg.add_series(gr, tt, fits[lv].survival(tt), f"Weibull {lv}", "line",
                      PAL[i % len(PAL)], dash="--", width=1.4, legend=len(levels) > 1)
    fresh.add_graph("Survival Function with Fitted Curves", gr, width=460, group="Plots")
    pp = rg.new_graph("Weibull Probability Plot", 11.0, 8.5)
    for i, (lv, c) in enumerate(curves):
        ok = (c.surv > 0) & (c.surv < 1)
        x = np.log(c.times[ok])
        y = np.log(-np.log(c.surv[ok]))
        rg.add_series(pp, x, y, str(lv), "scatter", PAL[i % len(PAL)], size=5.0,
                      legend=len(curves) > 1)
        if len(x):
            f = fits[lv]
            xs = np.array([x.min(), x.max()])
            rg.add_series(pp, xs, f.shape * (xs - math.log(f.scale)), f"Fit {lv}", "line",
                          PAL[i % len(PAL)], width=1.2, legend=False)
    fresh.add_graph("Weibull Probability Plot", rg.finish(pp, "ln(Time)",
                                                          "ln(−ln S(t))",
                                                          legend=len(curves) > 1),
                    width=400, group="Plots")
    report = st._finish(report, fresh)
    return report, report_text(first.book, "Weibull Fit", report, lines)


# ====================================================================== specs
def _surv_input(sel, group=True, cov=False):
    fields = [("range", "time", "Time Range", sel[0] if sel else ""),
              ("range", "censor", "Censor Range", sel[1] if len(sel) > 1 else ""),
              ("text", "censor_value", "Censored Value", "0")]
    if group:
        fields.append(("range", "group", "Factor (Group) Range (optional)",
                       sel[2] if len(sel) > 2 else ""))
    if cov:
        fields.append(("ranges", "covariates", "Covariates", sel[2:]))
    return fields


SPECS = {
    "km": lambda sel: ("Kaplan-Meier Estimator", [
        ("Input", _surv_input(sel)),
        ("Settings", [("float", "level", "Confidence Level", 0.95, (0.5, 0.9999, 4)),
                      ("choice", "ci", "Confidence Limits", "loglog", CI_TYPES),
                      ("bool", "ci_plot", "Confidence Limits on the Plot", False)])],
        "Product-limit survival estimates and comparison of groups.", "report sheet"),
    "cox": lambda sel: ("Cox Proportional Hazards Model", [
        ("Input", _surv_input(sel, group=False, cov=True)),
        ("Settings", [("float", "level", "Confidence Level", 0.95, (0.5, 0.9999, 4))])],
        "Semi-parametric proportional hazards regression.", "report sheet"),
    "weibull": lambda sel: ("Weibull Fit Model", [
        ("Input", _surv_input(sel)),
        ("Settings", [("float", "level", "Confidence Level", 0.95, (0.5, 0.9999, 4)),
                      ("bool", "exponential", "Exponential Model (shape = 1)", False)])],
        "Parametric Weibull survival model with right censoring.", "report sheet"),
}
FUNCS = {"km": km_report, "cox": cox_report, "weibull": weibull_report}
for _k, _f in FUNCS.items():
    st.register(_k, _f, SPECS[_k])

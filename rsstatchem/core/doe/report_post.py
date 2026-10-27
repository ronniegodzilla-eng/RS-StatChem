"""Report sections for optimisation and post analysis (point prediction with CI / PI / TI and
POE, confirmation with individual runs, coefficients table with model statistics, and the
numerical optimisation report with interval, SE, POE and Cpk goals).

These supersede ``report.prediction_section`` and ``report.optimization_section``; the old
functions stay in ``report.py`` for the SPSS-style front end.
"""
from __future__ import annotations

import math

import numpy as np

from rsstatchem.core.doe import intervals, optimize, plots_opt, poe
from rsstatchem.core.doe.model import Fit, Space
from rsstatchem.core.doe.report import _fmt_level, num, pval
from rsstatchem.core.output import Section, Table

GOAL_TEXT = dict(optimize.GOALS) | dict(optimize.FACTOR_GOALS)


def criterion_name(c: optimize.Criterion, space: Space, names: dict) -> str:
    if c.kind == "factor":
        j = next(i for i, f in enumerate(space.factors) if f.column == c.name)
        return space.factors[j].title
    base = names.get(c.name, c.name)
    if c.kind == "se":
        return f"SE({base})"
    if c.kind == "poe":
        return f"POE({base})"
    return base


def _interval_text(c: optimize.Criterion) -> str:
    if c.kind != "response" or c.interval in (None, "", "none") or c.goal in ("none", "cpk"):
        return ""
    side = c.interval_side()
    return c.interval.upper() + {"two": "", "lower": " (low)", "upper": " (high)"}[side]


# ---------------------------------------------------------------------------- prediction
def prediction_section(fits: dict[str, Fit], space: Space, x: np.ndarray, names: dict,
                       level: float | None = None, proportion: float | None = None,
                       side: str = "two", poe_sd: dict | None = None,
                       second_order: bool = False) -> Section:
    dl, dp = intervals.defaults()
    level = dl if level is None else level
    proportion = dp if proportion is None else proportion
    x = np.asarray(x, float)
    sec = Section("Point Prediction")
    sec.log("DOE PREDICT " + " ".join(f"{f.letter}={_fmt_level(f, x[j])}"
                                      for j, f in enumerate(space.factors)))
    sd = poe.sd_vector(space, poe_sd)
    show_poe = bool(np.any(sd > 0))
    sec.table(Table("Factor Settings", ["Factor", "Name", "Level", "Coded Level"]
                    + (["Std Dev (POE)"] if show_poe else []),
                    [[f.letter, f.name, _fmt_level(f, x[j]),
                      "" if f.categoric else num(space.to_coded(x[None, :])[0, j])]
                     + ([num(sd[j]) if sd[j] > 0 else ""] if show_poe else [])
                     for j, f in enumerate(space.factors)]))
    pct = f"{level * 100:g}%"
    tag = {"two": "", "lower": " (one-sided)", "upper": " (one-sided)"}[side]
    rows = []
    for col, fit in fits.items():
        iv = intervals.predict(fit, x[None, :], level, proportion, side)
        row = [names.get(col, col), num(iv["mean"][0]), num(iv["median"][0]), num(iv["sd"]),
               num(iv["se_mean"][0]), num(iv["ci"][0][0]), num(iv["ci"][1][0]),
               num(iv["se_pred"][0]), num(iv["pi"][0][0]), num(iv["pi"][1][0]),
               num(iv["ti"][0][0]), num(iv["ti"][1][0])]
        if show_poe:
            row.append(num(poe.poe(fit, x[None, :], sd, second_order)[0]))
        rows.append(row)
    cols = ["Response", "Predicted Mean", "Predicted Median", "Std Dev", "SE Mean",
            f"{pct} CI Low{tag}", f"{pct} CI High{tag}", "SE Pred", f"{pct} PI Low{tag}",
            f"{pct} PI High{tag}", f"TI Low{tag}", f"TI High{tag}"] + (["POE"] if show_poe
                                                                     else [])
    notes = ["Std Dev and standard errors are on the analysis (possibly transformed) scale; "
             "intervals are back-transformed.",
             "For a transformed response the predicted mean includes a second-order bias "
             "correction; the median is the simple back-transform.",
             f"TI: tolerance interval expected to contain {proportion * 100:g}% of the "
             f"population with {pct} confidence (k-factor by Howe's approximation"
             f"{'' if side == 'two' else ' / noncentral t for one side'} with the effective "
             "sample size 1/h)."]
    if side != "two":
        notes.append("One-sided intervals: only the " + ("lower" if side == "lower" else
                                                          "upper") + " bound is finite.")
    if show_poe:
        notes.append("POE: propagation of error, the standard deviation of one observation "
                     "including the transmitted factor variation, in original units"
                     + (" (second-order)." if second_order else " (first-order)."))
    sec.table(Table("Prediction", cols, rows, footnotes=notes))
    return sec


# ---------------------------------------------------------------------------- confirmation
def parse_runs(text) -> list[float]:
    """Individual confirmation results from text like '62, 63; 64' (or a list)."""
    if text is None:
        return []
    if isinstance(text, (list, tuple, np.ndarray)):
        vals = list(text)
    else:
        vals = str(text).replace(";", " ").replace(",", " ").split()
    out = []
    for v in vals:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def confirmation_row(fit: Fit, x: np.ndarray, n: int, mean: float | None,
                     level: float | None = None, proportion: float | None = None) -> dict:
    """Prediction interval for the mean of ``n`` confirmation runs and the verdict."""
    iv = intervals.predict(fit, np.asarray(x, float)[None, :], level, proportion,
                           n_runs=max(int(n), 1))
    lo, hi = float(iv["pi"][0][0]), float(iv["pi"][1][0])
    ok = None
    if mean is not None and math.isfinite(mean):
        ok = lo <= mean <= hi
    return {"mean": float(iv["mean"][0]), "median": float(iv["median"][0]), "sd": iv["sd"],
            "n": max(int(n), 1), "se_pred": float(iv["se_pred"][0]), "pi_low": lo,
            "pi_high": hi, "data_mean": mean, "ok": ok, "level": iv["level"],
            "ti": (float(iv["ti"][0][0]), float(iv["ti"][1][0]))}


def confirmation_section(fits: dict[str, Fit], space: Space, x: np.ndarray, names: dict,
                         runs: dict | None = None, summary: dict | None = None,
                         level: float | None = None,
                         proportion: float | None = None) -> Section:
    """``runs``: {column: individual results (list or text)}; ``summary``: {column:
    (n, mean)} for responses entered only as a count and an average. Runs win."""
    runs_in, summary = runs or {}, summary or {}
    dl, dp = intervals.defaults()
    level = dl if level is None else level
    proportion = dp if proportion is None else proportion
    pct = f"{level * 100:g}%"
    sec = Section("Confirmation")
    sec.table(Table("Factor Settings", ["Factor", "Name", "Level"],
                    [[f.letter, f.name, _fmt_level(f, x[j])]
                     for j, f in enumerate(space.factors)]))
    rows, run_rows = [], []
    for col, fit in fits.items():
        runs = parse_runs(runs_in.get(col))
        d = summary.get(col)
        if runs:
            n, mean = len(runs), float(np.mean(runs))
        elif isinstance(d, (list, tuple)) and len(d) == 2:
            n, mean = int(d[0] or 1), (None if d[1] is None else float(d[1]))
        else:
            n, mean = 1, None
        r = confirmation_row(fit, x, n, mean, level, proportion)
        verdict = "" if r["ok"] is None else ("Confirmed (inside PI)" if r["ok"]
                                              else "Not confirmed")
        rows.append([names.get(col, col), num(r["mean"]), num(r["median"]), num(r["sd"]),
                     str(r["n"]), num(r["se_pred"]), num(r["pi_low"]),
                     "" if mean is None else num(mean), num(r["pi_high"]),
                     num(np.std(runs, ddof=1)) if len(runs) > 1 else "", verdict])
        if runs:
            one = intervals.predict(fit, x[None, :], level, proportion)
            plo, phi = float(one["pi"][0][0]), float(one["pi"][1][0])
            tlo, thi = float(one["ti"][0][0]), float(one["ti"][1][0])
            for i, v in enumerate(runs):
                run_rows.append([names.get(col, col), str(i + 1), num(v), num(plo), num(phi),
                                 num(tlo), num(thi),
                                 "yes" if plo <= v <= phi else "no",
                                 "yes" if tlo <= v <= thi else "no"])
    sec.table(Table("Confirmation", ["Response", "Predicted Mean", "Predicted Median",
                                     "Std Dev", "n", "SE Pred", f"{pct} PI Low", "Data Mean",
                                     f"{pct} PI High", "Data Std Dev", "Result"], rows,
                    footnotes=[f"The {pct} prediction interval is for the mean of n "
                               "confirmation runs: ŷ ± t·s·√(1/n + h). The result is confirmed "
                               "when the data mean lies inside it."]))
    if run_rows:
        sec.table(Table("Confirmation Runs", ["Response", "Run", "Value", f"{pct} PI Low",
                                              f"{pct} PI High", "TI Low", "TI High",
                                              "Inside PI", "Inside TI"], run_rows,
                        footnotes=[f"Each run against the {pct} prediction interval of a "
                                   f"single observation and the tolerance interval for "
                                   f"{proportion * 100:g}% of the population."]))
    return sec


# ---------------------------------------------------------------------------- coefficients
def coefficients_section(fits: dict[str, Fit], names: dict, model_test=True) -> Section:
    """Coefficients of every response (coded factors) with p-values, plus fit statistics."""
    sec = Section("Coefficients Table")
    labels: list[str] = []
    data: dict = {}
    for col, fit in fits.items():
        pv = fit.term_p_values()
        by_label = {}
        for t, s in fit.mat.groups:
            for j in range(s.start, s.stop):
                lab = fit.mat.labels[j]
                p = pv.get(t, (None, None))[1] if t else None
                by_label[lab] = (float(fit.beta[j]), p)
                if lab not in labels:
                    labels.append(lab)
        if model_test:
            _, p = fit.model_test()
            by_label["Model p-value"] = (None, p)
        data[col] = by_label
    if model_test:
        labels.append("Model p-value")
    cols = ["Term"]
    for col in fits:
        t = names.get(col, col)
        cols += [t, f"p-value {t.split(':')[0]}"]
    rows = []
    for lab in labels:
        row = [lab]
        for col in fits:
            est, p = data[col].get(lab, (None, None))
            row += ["" if est is None else num(est), pval(p) if p is not None else ""]
        rows.append(row)
    sec.table(Table("Coefficients Table (coded factors)", cols, rows, footnotes=[
        "p-value colours: red < 0.01, green 0.01-0.05, blue 0.05-0.10, black ≥ 0.10 "
        "(not significant). Empty: the term is not in that response's model."]))
    stats_rows = []
    keys = [("Std. Dev.", "std_dev"), ("Mean", "mean"), ("C.V. %", "cv"), ("R²", "r2"),
            ("Adjusted R²", "adj_r2"), ("Predicted R²", "pred_r2"), ("PRESS", "press"),
            ("Adeq Precision", "adeq")]
    fstats = {}
    for col, fit in fits.items():
        try:
            fstats[col] = fit.fit_statistics()
        except (ValueError, np.linalg.LinAlgError):
            fstats[col] = {}
    for title, key in keys:
        stats_rows.append([title] + [num(fstats[c].get(key)) for c in fits])
    stats_rows.append(["Transform"] + [fit.transform.describe() for fit in fits.values()])
    stats_rows.append(["Terms (incl. intercept)"] + [str(fit.p) for fit in fits.values()])
    stats_rows.append(["Residual df"] + [str(fit.df_res) for fit in fits.values()])
    sec.table(Table("Fit Statistics", ["Statistic"] + [names.get(c, c) for c in fits],
                    stats_rows, footnotes=["Statistics on the analysis (transformed) scale."]))
    return sec


# ---------------------------------------------------------------------------- optimisation
def ramp_entries(problem: optimize.Problem, solution: dict, names: dict) -> list[dict]:
    """Ramp descriptions (one per factor and per response/SE/POE goal) at a solution."""
    sp = problem.space
    entries = []
    X = np.asarray(solution["x"], float)[None, :]
    for c in problem.criteria:
        if c.kind == "factor":
            j = problem.factor_index(c.name)
            if j is None:
                continue
            entries.append({"name": sp.factors[j].name, "low": c.lower, "high": c.upper,
                            "value": float(X[0, j]), "goal": c.goal, "target": c.target,
                            "kind": "factor"})
            continue
        if c.name not in problem.fits or c.goal == "none":
            continue
        y, lo, hi = problem.value(c, X)
        e = {"name": criterion_name(c, sp, names), "low": c.lower, "high": c.upper,
             "value": float(y[0]), "goal": c.goal, "target": c.target, "kind": c.kind}
        if c.goal == "cpk":
            e["name"] = f"Cpk({names.get(c.name, c.name)})"
        if lo is not y:
            e["bounds"] = (float(lo[0]), float(hi[0]))
            e["interval"] = _interval_text(c)
        entries.append(e)
    return entries


def evaluate_point(problem: optimize.Problem, x: np.ndarray) -> dict:
    """A solution-like dict at any point of the design space (ramps and bars anywhere)."""
    x = np.asarray(x, float)
    ev = problem.evaluate(x[None, :])
    return {"x": x, "pred": {k: float(v[0]) for k, v in ev["pred"].items()},
            "d": [float(d[0]) for d in ev["d"]], "D": float(ev["D"][0]),
            "extra": {k: float(v[0]) for k, v in ev["extra"].items()}}


def optimization_section(problem: optimize.Problem, solutions: list[dict], names: dict,
                         ramps: bool = True, bars: bool = True, contour: bool = False,
                         starts: int = 0, seed=None) -> Section:
    sp = problem.space
    sec = Section("Optimization")
    sec.log(f"DOE OPTIMIZE RESPONSES={','.join(problem.fits)} STARTS={starts}"
            + (f" SEED={seed}" if seed is not None else ""))
    rows = []
    for c in problem.criteria:
        lower, upper = num(c.lower), num(c.upper)
        target = num(c.target) if c.goal in ("target", "equal") else ""
        spec = ""
        if c.goal == "cpk":
            spec = f"LSL {num(c.lsl) or '-'}, USL {num(c.usl) or '-'}"
        rows.append([criterion_name(c, sp, names), GOAL_TEXT.get(c.goal, c.goal), lower,
                     upper, target, f"{c.w_lower:g}", f"{c.w_upper:g}",
                     str(c.importance) if c.active else "", _interval_text(c), spec])
    lvl, prop = intervals.defaults()
    lvl = problem.level or lvl
    prop = problem.proportion or prop
    foot = [f"Intervals: {lvl * 100:g}% confidence; TI for {prop * 100:g}% of the population. "
            "Maximize uses the one-sided lower bound, minimize the upper bound, target and in "
            "range both bounds of the two-sided interval."]
    if problem.discrete:
        foot.append("Discrete factors: " + "; ".join(
            f"{sp.factors[j].name} ∈ {{{', '.join(num(v) for v in problem.discrete_levels(j))}}}"
            for j in problem.discrete if j < sp.k))
    if poe.has_sd(sp, problem.poe_sd):
        sdv = poe.sd_vector(sp, problem.poe_sd)
        foot.append("Factor standard deviations for POE: " + ", ".join(
            f"{f.name} {num(sdv[j])}" for j, f in enumerate(sp.factors) if sdv[j] > 0))
    sec.table(Table("Constraints", ["Name", "Goal", "Lower Limit", "Upper Limit",
                                    "Target", "Lower Weight", "Upper Weight", "Importance",
                                    "Interval", "Spec Limits"], rows, footnotes=foot))
    if not solutions:
        sec.note("Solutions", "No solutions found.", style="warning")
        return sec
    resp = [c.name for c in problem.criteria if c.kind == "response" and c.name in problem.fits]
    extra_keys = list(solutions[0].get("extra", {}))
    cols = ["Number"] + [f.name for f in sp.factors] + [names.get(r, r) for r in resp] + \
        extra_keys + ["Desirability", ""]
    srows = []
    for i, s in enumerate(solutions):
        row = [str(i + 1)] + [_fmt_level(f, s["x"][j]).split(" ")[0] if f.categoric
                              else num(s["x"][j]) for j, f in enumerate(sp.factors)]
        row += [num(s["pred"].get(r)) for r in resp]
        row += [num(s.get("extra", {}).get(k)) for k in extra_keys]
        row += [f"{s['D']:.3f}", "Selected" if i == 0 else ""]
        srows.append(row)
    sec.table(Table("Solutions", cols, srows, footnotes=[
        f"{len(solutions)} distinct solutions found from {starts} random starting points "
        "plus the design points (Nelder-Mead simplex)."]))
    best = solutions[0]
    if best["D"] == 0:
        sec.note("Solutions", "No point satisfies all criteria (desirability 0). Relax the "
                              "limits; the solution shown minimises the constraint violation.",
                 style="warning")
    if ramps:
        sec.image("Ramps", plots_opt.ramps_plot(ramp_entries(problem, best, names), best["D"],
                                                "Desirability Ramps (Solution 1)"))
    if bars:
        act = problem.active_criteria()
        sec.image("Desirability", plots_opt.desirability_bars(
            [criterion_name(c, sp, names) for c in act], best["d"], best["D"]))
    return sec

"""Origin-style report sheets for the WP-1 fitting, peak and spectroscopy tools (no Qt).

Each ``*_report`` function takes worksheet data (:class:`analysis.XYData`), runs a core
analysis from :mod:`rsstatchem.core.analysis` and puts a report sheet (and, where Origin
does, an output data sheet) into the source workbook, returning ``(report, ..., log text)``.
Peak Analyzer reports store their settings in ``report.op`` (kind ``fit``, fit_kind
``peaks``) so Recalculate re-runs them with the stored settings when the source changes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rsstatchem.core.analysis import baselines as bl
from rsstatchem.core.analysis import calibration as cal
from rsstatchem.core.analysis import fit_compare as fc
from rsstatchem.core.analysis import fit_multi as fm
from rsstatchem.core.analysis import fitting as fit
from rsstatchem.core.analysis import odr
from rsstatchem.core.analysis import peak_analyzer as pa
from rsstatchem.core.analysis import spectroscopy as sp
from rsstatchem.core.analysis import surface_fit as sf
from rsstatchem.core.analysis.common import num, pval
from rsstatchem.core.analysis.fitting import FitResult
from rsstatchem.ui.origin import report_graphs as rg
from rsstatchem.ui.origin.analysis import (AnalysisError, FitOutcome, XYData, _append_column,
                                           _stamp, _table_into, add_report)
from rsstatchem.ui.origin.model import ReportSheet, Workbook, Worksheet, jsonable

PALETTE = ("#1f5fbf", "#2e7d32", "#ef6c00", "#6a1b9a", "#00838f", "#ad1457", "#4e342e",
           "#827717")


def _err(exc) -> AnalysisError:
    return AnalysisError(str(exc))


# ====================================================================== NLFit extras
def add_derived(report: ReportSheet, res: FitResult, yname: str) -> None:
    rows = [[yname if i == 0 else "", name, num(v), num(se)]
            for i, (name, v, se) in enumerate(res.derived())]
    if rows:
        report.add_table("Derived Parameters", ["", "", "Value", "Standard Error"], rows,
                         ["Standard errors by the delta method (first-order propagation)."])


def matrix_rows(names, m, digits: int = 6):
    return [[n] + [num(v, digits) if np.isfinite(v) else "--" for v in row]
            for n, row in zip(names, m)]


def add_matrices(report: ReportSheet, res: FitResult, constraints=None) -> None:
    """Covariance, correlation and dependency tables, information criteria and the
    constraints of a nonlinear fit (Origin's NLFit report tables)."""
    idx = np.flatnonzero(res.free)
    names = [res.names[i] for i in idx]
    if len(idx) == 0:
        return
    if constraints:
        report.add_table("Constraints", ["", "Constraint"],
                         [[str(i + 1), c] for i, c in enumerate(res.constraints)],
                         [f"{res.n_active} constraint(s) active (equalities and binding "
                          "inequalities reduce the degrees of freedom)."])
    cov = res.cov[np.ix_(idx, idx)]
    report.add_table("Covariance", [""] + names, matrix_rows(names, cov), group="Matrices")
    cor = res.correlation()[np.ix_(idx, idx)]
    report.add_table("Correlation", [""] + names, matrix_rows(names, cor, 4),
                     group="Matrices")
    dep = res.dependency()[idx]
    report.add_table("Dependency", ["", "Dependency"],
                     [[n, num(d, 5) if np.isfinite(d) else "--"] for n, d in zip(names, dep)],
                     ["Dependency = 1 − 1/(Cᵢᵢ·(C⁻¹)ᵢᵢ); values close to 1 mean the "
                      "parameter is redundant (over-parametrised model)."], group="Matrices")
    report.add_table("Information Criteria", ["", "Value"],
                     [["Number of Points (N)", str(res.n)],
                      ["Number of Parameters + 1 (K)", str(res.k_eff + 1)],
                      ["AIC", num(res.aic)], ["AICc", num(res.aicc)], ["BIC", num(res.bic)]],
                     ["AIC = N·ln(RSS/N) + 2K; AICc = AIC + 2K(K+1)/(N−K−1); "
                      "BIC = N·ln(RSS/N) + K·ln(N)."])


def add_residual_diagnostics(report: ReportSheet, res: FitResult, yname: str) -> None:
    r = np.asarray(res.residuals, float)
    if len(r) >= 3:
        report.add_graph("Normal Probability Plot of the Residual",
                         rg.qq_graph(r, "Normal Probability Plot of the Residual",
                                     f"Residual of {yname}"), width=460, group="Residual Plots")
    order = np.arange(1, len(r) + 1, dtype=float)
    report.add_graph("Residual vs. Order of the Data",
                     rg.scatter_graph(order, r, "Residual vs. Order of the Data",
                                      "Sequence Order", f"Residual of {yname}"), width=460,
                     group="Residual Plots")
    if len(r) > 2:
        report.add_graph("Residual Lag Plot",
                         rg.scatter_graph(r[:-1], r[1:], "Residual Lag Plot",
                                          "Residual(i−1)", "Residual(i)"), width=460,
                         group="Residual Plots")


# ====================================================================== multi data
@dataclass
class MultiOutcome:
    result: fm.MultiFitResult | FitResult
    report: ReportSheet
    curve: Worksheet
    log: str
    datas: list[XYData]


def _param_rows(res: FitResult, label: str, shared=()) -> list[list[str]]:
    rows = []
    t, p = res.t_values, res.p_values
    for i, name in enumerate(res.names):
        if not res.free[i]:
            rows.append([label if i == 0 else "", name, num(res.values[i]), "0", "--", "--",
                         "Yes" if name in shared else "", "fixed"])
            continue
        rows.append([label if i == 0 else "", name, num(res.values[i]), num(res.errors[i]),
                     num(t[i], 5), pval(p[i]), "Yes" if name in shared else "", ""])
    return rows


def run_multi_fit(datas: list[XYData], model, mode: str = "global", shared=(),
                  params0=None, fixed=None, bounds=None, constraints=None, replicas: int = 2,
                  weighting: str = "none", algorithm: str = "auto",
                  level: float = 0.95) -> MultiOutcome:
    """NLFit with multiple datasets (independent / global / concatenate) or replicas."""
    if not datas:
        raise AnalysisError("Select the input data.")
    book = datas[0].book
    try:
        f = fit.resolve_model(model)
        ws_ = [fit.make_weights(weighting, d.y, d.yerr) if weighting != "none" else None
               for d in datas]
        sets = [fm.Dataset(d.x, d.y, w, d.range_text) for d, w in zip(datas, ws_)]
        if mode == "replica":
            res = fm.replica_fit(datas[0].x, datas[0].y, f, int(replicas), params0=params0,
                                 weights=ws_[0], constraints=constraints, algorithm=algorithm)
            mr = None
        elif mode == "global":
            mr = fm.global_fit(sets, f, shared, params0, fixed, bounds, constraints,
                               algorithm=algorithm)
        elif mode == "concatenate":
            mr = fm.concatenate_fit(sets, f, params0=params0, fixed=fixed, bounds=bounds,
                                    constraints=constraints, algorithm=algorithm)
        elif mode == "independent":
            mr = fm.independent_fit(sets, f, params0, fixed, bounds, constraints,
                                    algorithm=algorithm)
        else:
            raise ValueError(f"Unknown mode '{mode}'")
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        raise _err(exc) from None
    label = fm.MODES.get(mode, mode)
    report = ReportSheet(title=f"{label} ({f.name}) of " + ", ".join(d.range_text
                                                                     for d in datas))
    add_report(book, report, "FitNLGlobal" if mode == "global" else
               ("FitNLReplica" if mode == "replica" else "FitNLMulti"))
    report.add_table("Notes", ["", "Value"], [
        ["Description", f"Nonlinear Curve Fit — {label}"],
        ["Model", f.name], ["Equation", f.formula],
        ["Input Data", "; ".join(d.range_text for d in datas)],
        ["Shared Parameters", ", ".join(shared) if mode == "global" and shared else "--"],
        ["Number of Replicas", str(replicas) if mode == "replica" else "--"],
        ["Weight", fit.WEIGHTINGS.get(weighting, weighting)],
        ["Iteration Algorithm", fit.ALGORITHMS.get(algorithm, algorithm)]])
    cols = ["", "", "Value", "Standard Error", "t-Value", "Prob>|t|", "Shared", ""]
    xx_all = []
    curve = Worksheet(name=book.unique_sheet_name("FitNLCurve"), n_rows=300)
    g = rg.new_graph(f"{label} ({f.name})", 12.5, 8.5)
    if mode == "replica":
        rows = _param_rows(res, datas[0].yname)
        report.add_table("Parameters", cols, rows)
        _table_into(report, fit.statistics_table(res, datas[0].yname))
        add_matrices(report, res, constraints)
        d = datas[0]
        xx = np.linspace(float(d.x.min()), float(d.x.max()), 300)
        rg.add_series(g, d.x, d.y, d.ytitle, "scatter", "#000000", size=5.0)
        meta = res.model.meta
        base_f = fit.resolve_model(meta["replica_of"])
        comps = []
        for i in range(meta["n"]):
            q = [0.0 if p in meta["offsets"] else res.params[f"{p}_{i + 1}"]
                 for p in base_f.params]
            comps.append(base_f(xx, *q) + sum(res.params[o] for o in meta["offsets"]))
            rg.add_series(g, xx, comps[-1], f"Replica {i + 1}", "line",
                          rg.FILLS[i % len(rg.FILLS)], dash="--", width=1.0)
        rg.add_series(g, xx, res.predict(xx), "Fit Curve", "line", rg.FIT_RED, width=1.8)
        _curve_cols(curve, xx, [("Fit", res.predict(xx))] +
                    [(f"Replica {i + 1}", c) for i, c in enumerate(comps)], d.xtitle)
        out_res = res
        log_lines = [f"  R² = {num(res.r2)}  Reduced Chi-Sqr = {num(res.red_chisq)}"]
    else:
        rows = []
        for i, r in enumerate(mr.results):
            rows += _param_rows(r, datas[i].range_text if len(mr.results) > 1 else
                                datas[0].yname, mr.shared)
        report.add_table("Parameters", cols, rows,
                         ["Global fit: shared parameters have one value for all datasets; "
                          "standard errors come from the joint covariance."]
                         if mode == "global" else [])
        st_rows = []
        for i, r in enumerate(mr.results):
            st_rows.append([datas[i].range_text if mode != "concatenate" else "All data",
                            str(r.n), num(r.rss), num(r.r2), num(r.adj_r2) if mode !=
                            "global" else "--", num(r.red_chisq) if mode != "global" else "--"])
        if mr.combined is not None and mode == "global":
            c = mr.combined
            st_rows.append(["Global", str(c.n), num(c.rss), num(c.r2), num(c.adj_r2),
                            num(c.red_chisq)])
        report.add_table("Statistics", ["", "Number of Points", "Residual Sum of Squares",
                                        "R-Square (COD)", "Adj. R-Square", "Reduced Chi-Sqr"],
                         st_rows, [f"Global: degrees of freedom = {mr.combined.dof}."]
                         if mode == "global" else [])
        if mode == "global":
            add_matrices(report, mr.combined, constraints)
        series = []
        for i, d in enumerate(datas):
            col = PALETTE[i % len(PALETTE)]
            rg.add_series(g, d.x, d.y, d.ytitle, "scatter", col, size=5.0)
            r = mr.results[0 if mode == "concatenate" else i]
            xx = np.linspace(float(d.x.min()), float(d.x.max()), 300)
            rg.add_series(g, xx, r.predict(xx), f"Fit of {d.yname}", "line", col, width=1.6)
            series.append((f"Fit of {d.yname}", r.predict(xx), xx))
            xx_all.append(xx)
        for i, (name, yv, xx) in enumerate(series):
            _curve_add(curve, xx, yv, name, datas[i].xtitle, first=i == 0)
        out_res = mr
        log_lines = [f"  {r.names[j]} = {num(r.values[j])} ± {num(r.errors[j])}"
                     for r in mr.results[:1] for j in range(len(r.names))]
    rg.finish(g, datas[0].xtitle, datas[0].ytitle, legend=True)
    report.add_graph("Fitted Curves Plot", g, width=520)
    report.op = {"kind": "fit_multi", "mode": "none",
                 "settings": jsonable({"model": model if isinstance(model, str) else f.name,
                                       "mode": mode, "shared": list(shared),
                                       "replicas": replicas})}
    book.sheets.append(curve)
    log = "\n".join([_stamp(book), f"{label} ({f.name}) of "
                     + ", ".join(d.range_text for d in datas)] + log_lines +
                    [f"  Report: [{book.name}]{report.name}   Curve: [{book.name}]{curve.name}"])
    return MultiOutcome(out_res, report, curve, log, datas)


def _curve_cols(ws: Worksheet, xx, ys: list[tuple[str, np.ndarray]], xtitle: str) -> None:
    ws.n_rows = len(xx)
    c = ws.add_column("X")
    c.long_name, c.comments = "Independent Variable", xtitle
    ws.set_column_values(0, list(map(float, xx)))
    for name, y in ys:
        c = ws.add_column("Y")
        c.long_name = name
        ws.set_column_values(len(ws.columns) - 1, [float(v) if np.isfinite(v) else None
                                                    for v in y])


def _curve_add(ws: Worksheet, xx, y, name, xtitle, first: bool) -> None:
    ws.n_rows = max(ws.n_rows, len(xx))
    c = ws.add_column("X")
    c.long_name, c.comments = "Independent Variable", xtitle
    ws.set_column_values(len(ws.columns) - 1, list(map(float, xx)))
    c = ws.add_column("Y")
    c.long_name = name
    ws.set_column_values(len(ws.columns) - 1, [float(v) if np.isfinite(v) else None
                                                for v in y])


# ====================================================================== comparison
def compare_models_report(data: XYData, models: list, criterion: str = "aicc") \
        -> tuple[ReportSheet, str]:
    try:
        cmp = fc.compare_models(data.x, data.y, models, criterion=criterion)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        raise _err(exc) from None
    book = data.book
    report = ReportSheet(title=f"Compare Models of {data.range_text}")
    add_report(book, report, "FitCompareModel")
    rows = cmp.rows
    report.add_table("Fit Statistics", ["", "Number of Points", "Number of Parameters",
                                        "Residual Sum of Squares", "Reduced Chi-Sqr",
                                        "Adj. R-Square"],
                     [[r["model"], str(r["n"]), str(r["k"]), num(r["rss"]),
                       num(r["red_chisq"]), num(r["adj_r2"])] for r in rows])
    report.add_table("Akaike Information Criterion (AIC) Test",
                     ["", "AIC", "AICc", "Δ AICc", "Akaike Weight"] +
                     (["Evidence Ratio"] if len(rows) == 2 else []),
                     [[r["model"], num(r["aic"]), num(r["aicc"]), num(r["d_aicc"], 5),
                       num(r["w_aicc"], 5)] + ([num(r.get("er_aicc", math.nan), 5)]
                                               if len(rows) == 2 else []) for r in rows],
                     [f"Based on AICc, {min(rows, key=lambda r: r['aicc'])['model']} is "
                      "more likely to be correct."])
    report.add_table("Bayesian Information Criterion (BIC) Test", ["", "BIC", "Δ BIC"],
                     [[r["model"], num(r["bic"]), num(r["d_bic"], 5)] for r in rows],
                     [f"Based on BIC, {min(rows, key=lambda r: r['bic'])['model']} is "
                      "preferred (lower BIC)."])
    if cmp.ftest:
        t = cmp.ftest
        report.add_table("F-test", ["", "DF", "RSS", "F Value", "Prob>F"],
                         [[t["simple"], str(t["dof1"]), num(t["rss1"]), num(t["F"], 5),
                           pval(t["p"])], [t["complex"], str(t["dof2"]), num(t["rss2"]), "",
                                           ""]],
                         ["Assumes the first model is nested in the second. At the 0.05 "
                          "level, " + (f"{t['complex']} is significantly better."
                                       if t["p"] < 0.05 else
                                       f"{t['complex']} is not significantly better.")])
    g = rg.new_graph("Compare Models", 12.5, 8.5)
    rg.add_series(g, data.x, data.y, data.ytitle, "scatter", "#000000", size=5.0)
    xx = np.linspace(float(data.x.min()), float(data.x.max()), 300)
    for i, r in enumerate(rows):
        _table_into(report, fit.parameter_table(r["result"], r["model"],
                                                title=f"Parameters: {r['model']}"))
        rg.add_series(g, xx, r["result"].predict(xx), r["model"], "line",
                      PALETTE[i % len(PALETTE)], width=1.6)
    report.add_graph("Fitted Curves Plot", rg.finish(g, data.xtitle, data.ytitle, True),
                     width=520)
    text = f"{_stamp(book)}\nCompare Models of {data.range_text}: best by " \
           f"{fc.CRITERIA[criterion]} = {cmp.best} → [{book.name}]{report.name}"
    return report, text


def rank_models_report(data: XYData, models=None, category: str | None = None,
                       criterion: str = "aicc") -> tuple[ReportSheet, str]:
    rows = fc.rank_models(data.x, data.y, models, category, criterion=criterion)
    ok = [r for r in rows if r.get("rank")]
    if not ok:
        raise AnalysisError("No model could be fitted to these data.")
    book = data.book
    report = ReportSheet(title=f"Rank Models of {data.range_text}")
    add_report(book, report, "FitRankModels")
    report.add_table("Rank Models", ["Rank", "Model", "AICc", "Δ AICc", "Akaike Weight",
                                     "BIC", "Adj. R-Square", "Reduced Chi-Sqr"],
                     [[str(r["rank"]), r["model"], num(r["aicc"]), num(r["d_aicc"], 5),
                       num(r["w_aicc"], 4), num(r["bic"]), num(r["adj_r2"]),
                       num(r["red_chisq"])] for r in ok],
                     [f"Ranked by {fc.CRITERIA[criterion]}" +
                      (f" over category {category}." if category else ".")])
    bad = [r for r in rows if not r.get("rank")]
    if bad:
        report.add_table("Models Not Fitted", ["Model", "Reason"],
                         [[r["model"], r["error"]] for r in bad])
    g = rg.new_graph("Rank Models", 12.5, 8.5)
    rg.add_series(g, data.x, data.y, data.ytitle, "scatter", "#000000", size=5.0)
    xx = np.linspace(float(data.x.min()), float(data.x.max()), 300)
    for i, r in enumerate(ok[:3]):
        rg.add_series(g, xx, r["result"].predict(xx), f"{r['rank']}. {r['model']}", "line",
                      PALETTE[i], width=1.6)
    report.add_graph("Top Models", rg.finish(g, data.xtitle, data.ytitle, True), width=520)
    return report, f"{_stamp(book)}\nRank Models of {data.range_text}: best = " \
                   f"{ok[0]['model']} → [{book.name}]{report.name}"


def compare_datasets_report(datas: list[XYData], model, criterion: str = "aicc") \
        -> tuple[ReportSheet, str]:
    if len(datas) < 2:
        raise AnalysisError("Select two or more datasets.")
    try:
        out = fc.compare_datasets([(d.x, d.y) for d in datas], model, criterion)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        raise _err(exc) from None
    book = datas[0].book
    f = fit.resolve_model(model)
    report = ReportSheet(title=f"Compare Datasets ({f.name})")
    add_report(book, report, "FitCompareData")
    report.add_table("F-test", ["", "DF", "RSS", "F Value", "Prob>F"],
                     [["Separate fits", str(out["dof_ind"]), num(out["rss_ind"]),
                       num(out["F"], 5), pval(out["p"])],
                      ["One parameter set", str(out["dof_comb"]), num(out["rss_comb"]), "",
                       ""]],
                     ["At the 0.05 level the datasets " + ("can" if out["same"] else
                                                          "cannot") +
                      " be described by one parameter set."])
    report.add_table("Akaike Information Criterion (AIC)", ["", "AIC", "AICc", "BIC"],
                     [["Separate fits", num(out["crit_ind"]["aic"]),
                       num(out["crit_ind"]["aicc"]), num(out["crit_ind"]["bic"])],
                      ["One parameter set", num(out["crit_comb"]["aic"]),
                       num(out["crit_comb"]["aicc"]), num(out["crit_comb"]["bic"])]])
    return report, f"{_stamp(book)}\nCompare Datasets ({f.name}): p = {pval(out['p'])} → " \
                   f"[{book.name}]{report.name}"


# ====================================================================== peak analyzer
def analysis_settings(an: pa.PeakAnalysis) -> dict:
    return jsonable({"goal": an.goal, "baseline_mode": an.baseline_mode,
                     "baseline_opts": {k: v for k, v in an.baseline_opts.items()
                                       if k != "dataset"},
                     "anchors": [list(map(float, an.anchors[0])), list(map(float, an.anchors[1]))]
                     if an.anchors is not None else None,
                     "find_opts": an.find_opts, "limits": an.limits, "fit_opts": an.fit_opts,
                     "manual_peaks": [{k: v for k, v in p.items()} for p in an.peaks]
                     if an.keep_peaks else None})


def analysis_from_settings(x, y, s: dict) -> pa.PeakAnalysis:
    an = pa.PeakAnalysis(x, y, s.get("goal", "fit"), s.get("baseline_mode", "none"),
                         dict(s.get("baseline_opts") or {}),
                         None, dict(s.get("find_opts") or {"method": "local_max"}),
                         s.get("limits"), dict(s.get("fit_opts") or {"shape": "gauss"}))
    if s.get("anchors"):
        an.anchors = (np.asarray(s["anchors"][0], float), np.asarray(s["anchors"][1], float))
    if s.get("manual_peaks"):
        an.peaks = [dict(p) for p in s["manual_peaks"]]
        an.keep_peaks = True
    return an


def peak_report(data: XYData, an: pa.PeakAnalysis, report: ReportSheet | None = None,
                curve: Worksheet | None = None, mode: str = "manual",
                run: bool = True) -> FitOutcome:
    """Run the Peak Analyzer ``an`` on ``data`` and write the PeakAnalyzer report and data
    sheet (in place when ``report`` / ``curve`` are given: Recalculate)."""
    try:
        if run:
            an.run()
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        raise _err(exc) from None
    book = data.book
    goal = pa.GOALS.get(an.goal, an.goal)
    new = ReportSheet(title=f"Peak Analyzer ({goal}) of {data.range_text}")
    rerun = report is not None
    if not rerun:
        report = add_report(book, new, "PeakAnalyzer")
    fo = an.fit_opts
    new.add_table("Notes", ["", "Value"], [
        ["Description", f"Peak Analyzer — {goal}"], ["Input Data", data.range_text],
        ["Baseline Mode", bl.BASELINE_MODES.get(an.baseline_mode, an.baseline_mode)],
        ["Peak Finding Method", pa.FIND_METHODS.get(an.find_opts.get("method", ""), "--")
         if an.goal in ("find", "integrate", "fit") else "--"],
        ["Peak Function", pa.SHAPES[fo.get("shape", "gauss")]["label"]
         if an.goal == "fit" else "--"],
        ["Number of Peaks", str(len(an.peaks)) if an.peaks else "--"]])
    if an.anchors is not None and an.baseline_mode == "anchors":
        new.add_table("Baseline Anchor Points", ["Index", "X", "Y"],
                      [[str(i + 1), num(a), num(b)] for i, (a, b) in
                       enumerate(zip(*an.anchors))])
    if an.peaks:
        new.add_table("Peak Finding", ["Peak Index", "X", "Y", "Height", "FWHM (estimate)",
                                       "Hidden"],
                      [[str(i + 1), num(p["x"]), num(p["y"]), num(p["height"]),
                        num(p["fwhm"]), "Yes" if p.get("hidden") else ""]
                       for i, p in enumerate(an.peaks)],
                      ["Hidden peaks (shoulders) are not local maxima of the data."])
    if an.integration:
        new.add_table("Integration Result", ["Peak Index", "X From", "X To", "Area",
                                             "Area %", "Centroid", "Height", "X at Max",
                                             "FWHM"],
                      [[str(r["peak"]), num(r["x_from"]), num(r["x_to"]), num(r["area"]),
                        num(r["area_pct"], 5), num(r["centroid"]), num(r["height"]),
                        num(r["x_max"]), num(r["fwhm"])] for r in an.integration])
    xx = np.linspace(float(an.x.min()), float(an.x.max()), max(400, len(an.x)))
    g = rg.new_graph(f"Peak Analyzer of {data.yname}", 12.5, 8.5)
    rg.add_series(g, an.x, an.y, data.ytitle, "line", "#000000", width=1.0)
    if an.baseline is not None and an.baseline_mode != "none":
        rg.add_series(g, an.x, an.baseline, "Baseline", "line", "#d40000", dash="--",
                      width=1.2)
    if an.anchors is not None and an.baseline_mode == "anchors":
        rg.add_series(g, an.anchors[0], an.anchors[1], "Anchor Points", "scatter",
                      "#d40000", symbol="o", size=5.0)
    res = None
    if an.fit is not None:
        pf = an.fit
        res = pf.result
        _table_into(new, fit.parameter_table(res, data.yname))
        new.add_table("Peak Properties", ["Peak Index", "Center", "Center SE", "Height",
                                          "Area", "Area SE", "Area %", "FWHM", "Centroid"],
                      [[str(r["peak"]), num(r["center"]), num(r["center_se"]),
                        num(r["height"]), num(r["area"]), num(r["area_se"]),
                        num(r["area_pct"], 5), num(r["fwhm"]), num(r["centroid"])]
                       for r in pf.peaks],
                      ["Height, FWHM and centroid are evaluated on the fitted peak curve."])
        _table_into(new, fit.statistics_table(res, data.yname))
        base = an.baseline if an.baseline is not None else 0.0
        bxx = np.interp(xx, an.x, base) if an.baseline is not None else np.zeros_like(xx)
        comps = pf.components(xx)
        for i, c in enumerate(comps):
            rg.add_series(g, xx, c + bxx + pf.baseline(xx), f"Peak {i + 1}", "line",
                          rg.FILLS[i % len(rg.FILLS)], width=1.0)
        rg.add_series(g, xx, pf.cumulative(xx) + bxx, "Cumulative Fit Peak", "line",
                      rg.FIT_RED, width=1.6)
        del base
    elif an.peaks:
        rg.add_series(g, [p["x"] for p in an.peaks], [p["y"] for p in an.peaks], "Peaks",
                      "scatter", "#d40000", symbol="v", size=7.0)
    new.add_graph("Peak Analyzer Plot", rg.finish(g, data.xtitle, data.ytitle, True),
                  width=540)
    if res is not None:
        new.add_graph("Residual vs. Independent Plot",
                      rg.residual_graph(res, data.xtitle, data.ytitle), width=460,
                      group="Residual Plots")
    src = data.source()
    curve_name = curve.name if curve is not None else book.unique_sheet_name("PeakAnalyzerData")
    new.op = {"kind": "fit", "fit_kind": "peaks", "input": src, "curve": curve_name,
              "mode": report.op.get("mode", mode) if rerun else mode,
              "settings": analysis_settings(an)}
    if rerun:
        report.replace_items(new)
        report.op.update(new.op)
    else:
        report.op = new.op
    report.op["dirty"] = False
    fresh = Worksheet(name=curve_name, n_rows=len(an.x))
    cols = [("Baseline", an.baseline)] if an.baseline is not None else []
    cols.append(("Subtracted Y", an.subtracted))
    if an.fit is not None:
        cols.append(("Cumulative Fit Peak", an.fit.cumulative(an.x)))
        cols += [(f"Peak{i + 1}", c) for i, c in enumerate(an.fit.components(an.x))]
    _curve_cols(fresh, an.x, cols, data.xtitle)
    if curve is None:
        curve = fresh
        book.sheets.append(curve)
    else:
        curve.columns, curve.n_rows = fresh.columns, fresh.n_rows
    lines = [_stamp(book), f"Peak Analyzer ({goal}) of {data.range_text}"]
    if an.fit is not None:
        lines.append(f"  {len(an.fit.peaks)} peak(s), R² = {num(an.fit.result.r2)}")
        for r in an.fit.peaks:
            lines.append(f"  Peak {r['peak']}: center = {num(r['center'])}  area = "
                         f"{num(r['area'])} ({num(r['area_pct'], 4)} %)  FWHM = "
                         f"{num(r['fwhm'])}")
    elif an.integration:
        for r in an.integration:
            lines.append(f"  Peak {r['peak']}: area = {num(r['area'])} "
                         f"({num(r['area_pct'], 4)} %)")
    elif an.peaks:
        lines.append(f"  {len(an.peaks)} peak(s) found")
    lines.append(f"  Report: [{book.name}]{report.name}   Data: [{book.name}]{curve.name}")
    yy = an.fit.cumulative(xx) if an.fit is not None else np.interp(xx, an.x, an.subtracted)
    return FitOutcome(res, report, curve, "\n".join(lines), data, xx, yy,
                      "Cumulative Fit Peak" if an.fit is not None else "Subtracted Y")


def rerun_peaks(data: XYData, report: ReportSheet, book: Workbook) -> FitOutcome:
    an = analysis_from_settings(data.x, data.y, report.op.get("settings") or {})
    curve = book.sheet(report.op.get("curve", ""))
    return peak_report(data, an, report, curve if isinstance(curve, Worksheet) else None)


def batch_peaks_report(datas: list[XYData], template: pa.PeakAnalysis) \
        -> tuple[ReportSheet, str]:
    book = datas[0].book
    rows_out = []
    for d in datas:
        rows_out += pa.batch(d.x, [d.y], [d.range_text], template)
    report = ReportSheet(title="Batch Peak Analysis")
    add_report(book, report, "BatchPeakAnalysis")
    keys = [k for k in ("peak", "center", "x", "area", "area_pct", "height", "fwhm")
            if any(k in r for r in rows_out)]
    report.add_table("Summary", ["Dataset"] + [k.replace("_pct", " %").title() for k in keys]
                     + ["Error"],
                     [[r["dataset"]] + [num(r[k]) if isinstance(r.get(k), float) else
                                        str(r.get(k, "")) for k in keys] + [r.get("error", "")]
                      for r in rows_out])
    return report, f"{_stamp(book)}\nBatch Peak Analysis of {len(datas)} dataset(s) → " \
                   f"[{book.name}]{report.name}"


# ====================================================================== errors in x
def odr_report(data: XYData, model, xerr=None, yerr=None, method: str = "odr") \
        -> tuple[ReportSheet, str]:
    book = data.book
    try:
        if method == "york":
            if xerr is None or yerr is None:
                raise ValueError("York regression needs X and Y standard deviations.")
            yr = odr.york_fit(data.x, data.y, xerr, yerr)
        else:
            r = odr.odr_fit(data.x, data.y, model, sx=xerr, sy=yerr)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        raise _err(exc) from None
    title = "Fit Linear with X Error (York)" if method == "york" else \
        "Orthogonal Distance Regression"
    report = ReportSheet(title=f"{title} of {data.range_text}")
    add_report(book, report, "FitLinearXErr" if method == "york" else "FitODR")
    xx = np.linspace(float(data.x.min()), float(data.x.max()), 300)
    g = rg.new_graph(title, 12.5, 8.5)
    rg.add_series(g, data.x, data.y, data.ytitle, "scatter", "#000000", size=5.0,
                  yerr=None if yerr is None else np.broadcast_to(yerr, data.y.shape))
    if method == "york":
        report.add_table("Parameters", ["", "Value", "Standard Error"],
                         [["Intercept", num(yr.intercept), num(yr.se_intercept)],
                          ["Slope", num(yr.slope), num(yr.se_slope)]],
                         ["York et al. (2004) unified equations; standard errors are not "
                          "scaled by the MSWD."])
        report.add_table("Statistics", ["", "Value"],
                         [["Number of Points", str(yr.n)], ["Chi-Sqr (S)", num(yr.S)],
                          ["Reduced Chi-Sqr (MSWD)", num(yr.mswd)],
                          ["Prob>Chi-Sqr", pval(yr.p_value)],
                          ["Iterations", str(yr.iterations)]])
        rg.add_series(g, xx, yr.intercept + yr.slope * xx, "Fit Curve", "line", rg.FIT_RED,
                      width=1.8)
        text = f"York fit: slope = {num(yr.slope)} ± {num(yr.se_slope)}, intercept = " \
               f"{num(yr.intercept)} ± {num(yr.se_intercept)}"
    else:
        t = np.where(r.free & (r.errors > 0), r.values / r.errors, np.nan)
        from scipy import stats
        p = 2 * stats.t.sf(np.abs(t), r.dof) if r.dof > 0 else np.full(len(t), np.nan)
        report.add_table("Parameters", ["", "Value", "Standard Error", "t-Value", "Prob>|t|"],
                         [[n, num(v), num(e), num(tt, 5), pval(pp)] for n, v, e, tt, pp in
                          zip(r.names, r.values, r.errors, t, p)],
                         ["Standard errors scaled by the residual variance (as ODRPACK)."])
        report.add_table("Statistics", ["", "Value"],
                         [["Number of Points", str(r.n)], ["Degrees of Freedom", str(r.dof)],
                          ["Weighted Sum of Squares", num(r.sum_square)],
                          ["Residual Variance", num(r.res_var)],
                          ["R-Square (COD, vertical)", num(r.r2)],
                          ["Fit Status", r.message]])
        rg.add_series(g, xx, r.predict(xx), "Fit Curve", "line", rg.FIT_RED, width=1.8)
        text = f"ODR ({r.model.name}): " + ", ".join(f"{n} = {num(v)} ± {num(e)}" for n, v, e
                                                     in zip(r.names, r.values, r.errors))
    report.add_graph("Fitted Curves Plot", rg.finish(g, data.xtitle, data.ytitle, True),
                     width=520)
    return report, f"{_stamp(book)}\n{text} → [{book.name}]{report.name}"


def implicit_report(book: Workbook, x, y, xtitle: str, ytitle: str, model,
                    params0=None) -> tuple[ReportSheet, str]:
    try:
        r = odr.implicit_fit(x, y, model, params0)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        raise _err(exc) from None
    report = ReportSheet(title=f"Implicit Fit ({r.model.name})")
    add_report(book, report, "FitImplicit")
    report.add_table("Notes", ["", "Value"], [["Model", r.model.name],
                                              ["Equation", r.model.formula],
                                              ["Fit Status", r.message]])
    report.add_table("Parameters", ["", "Value", "Standard Error"],
                     [[n, num(v), num(e)] for n, v, e in zip(r.names, r.values, r.errors)])
    report.add_table("Statistics", ["", "Value"],
                     [["Number of Points", str(len(r.x))], ["Degrees of Freedom", str(r.dof)],
                      ["Sum of Squared Distances", num(r.sum_square)],
                      ["Mean Orthogonal Distance", num(float(np.mean(r.distances)))]])
    g = rg.new_graph("Implicit Fit", 11.0, 9.5)
    rg.add_series(g, r.x, r.y, ytitle, "scatter", "#000000", size=5.0)
    cx, cy = r.curve()
    if len(cx):
        rg.add_series(g, cx, cy, "Fit Curve", "line", rg.FIT_RED, width=1.6)
    report.add_graph("Fitted Curve Plot", rg.finish(g, xtitle, ytitle, True), width=480)
    return report, f"{_stamp(book)}\nImplicit Fit ({r.model.name}): " + ", ".join(
        f"{n} = {num(v)}" for n, v in zip(r.names, r.values)) + f" → [{book.name}]{report.name}"


def surface_report(book: Workbook, x, y, z, titles: tuple[str, str, str], model,
                   params0=None, source: str = "") -> tuple[ReportSheet, FitResult, str]:
    try:
        res = sf.surface_fit(x, y, z, model, params0=params0)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        raise _err(exc) from None
    report = ReportSheet(title=f"Surface Fit ({res.model.name}) of {source or titles[2]}")
    add_report(book, report, "FitSurface")
    report.add_table("Notes", ["", "Value"], [["Model", res.model.name],
                                              ["Equation", res.model.formula],
                                              ["Input Data", source or ", ".join(titles)],
                                              ["Fit Status", res.message]])
    _table_into(report, fit.parameter_table(res, titles[2]))
    _table_into(report, fit.statistics_table(res, titles[2]))
    if res.dof > 0:
        _table_into(report, fit.anova_table(res, titles[2], lack_of_fit=False))
    add_matrices(report, res)
    report.add_image("Fitted Surface (contour) and Data", _surface_png(res, titles),
                     width=520)
    r = res.residuals
    report.add_graph("Residual vs. Fitted", rg.scatter_graph(res.fitted, r,
                                                             "Residual vs. Fitted Z",
                                                             f"Fitted {titles[2]}",
                                                             "Residual"), width=460)
    return report, res, f"{_stamp(book)}\nSurface Fit ({res.model.name}): R² = " \
                        f"{num(res.r2)} → [{book.name}]{report.name}"


def _surface_png(res: FitResult, titles) -> bytes:
    from rsstatchem.core.analysis.common import figure, png
    X = res.x
    fig = figure(6.0, 4.6)
    ax = fig.add_subplot()
    gx = np.linspace(X[:, 0].min(), X[:, 0].max(), 120)
    gy = np.linspace(X[:, 1].min(), X[:, 1].max(), 120)
    GX, GY = np.meshgrid(gx, gy)
    Z = res.predict(np.stack([GX.ravel(), GY.ravel()], 1)).reshape(GX.shape)
    cs = ax.contourf(GX, GY, Z, 20, cmap="viridis")
    fig.colorbar(cs, ax=ax, label=f"Fitted {titles[2]}")
    ax.scatter(X[:, 0], X[:, 1], s=8, c="k", alpha=0.6)
    ax.set_xlabel(titles[0])
    ax.set_ylabel(titles[1])
    return png(fig)


# ====================================================================== calibration
def calibration_report(data: XYData, weighting: str = "none", sigma: str = "residual",
                       blank_sd: float | None = None, unknowns=(), replicates: int = 1,
                       level: float = 0.95) -> tuple[ReportSheet, str]:
    try:
        c = cal.calibrate(data.x, data.y, weighting, sigma, blank_sd)
    except ValueError as exc:
        raise _err(exc) from None
    book = data.book
    report = ReportSheet(title=f"Calibration Curve of {data.range_text}")
    add_report(book, report, "Calibration")
    report.add_table("Calibration Equation", ["", "Value", "Standard Error"],
                     [["Intercept (a)", num(c.intercept), num(c.se_intercept)],
                      ["Slope (b)", num(c.slope), num(c.se_slope)]],
                     [f"y = a + b·x; weighting: {cal.CAL_WEIGHTS.get(weighting, weighting)}."])
    report.add_table("Statistics", ["", "Value"],
                     [["Number of Standards", str(c.n)], ["R-Square", num(c.r2)],
                      ["Adj. R-Square", num(c.adj_r2)], ["Pearson's r", num(c.r)],
                      ["s(y/x)", num(c.s_yx)]])
    report.add_table("Detection Limits", ["", "Value"],
                     [["σ used", cal.SIGMAS.get(sigma, sigma)], ["σ", num(c.sigma)],
                      ["LOD = 3.3·σ/b", num(c.lod)], ["LOQ = 10·σ/b", num(c.loq)]],
                     ["ICH Q2(R1) definitions."])
    if unknowns:
        rows = []
        for y0 in unknowns:
            x0, s, lo, hi = c.inverse(float(y0), int(replicates), level)
            rows.append([num(y0), num(x0), num(s), num(lo), num(hi),
                         "below LOQ" if x0 < c.loq else ""])
        pc = f"{level * 100:g}%"
        report.add_table("Unknown Samples", ["Response y0", "Concentration x0", "s(x0)",
                                             f"Lower {pc}", f"Upper {pc}", "Note"], rows,
                         [f"Mean of {replicates} replicate(s) per sample; Miller & Miller "
                          "eq. 5.9 / 5.10."])
    xx = np.linspace(float(min(data.x.min(), 0)), float(data.x.max()), 300)
    g = rg.new_graph("Calibration Curve", 12.5, 8.5)
    rg.add_series(g, data.x, data.y, data.ytitle, "scatter", "#000000", size=6.0)
    lo, hi = c.band(xx, level)
    rg.add_series(g, xx, lo, "Lower Confidence Limit", "line", rg.BAND, width=1.0,
                  legend=False)
    rg.add_series(g, xx, hi, f"{level * 100:g}% Confidence Band", "line", rg.BAND, width=1.0)
    rg.add_series(g, xx, c.predict(xx), "Calibration Line", "line", rg.FIT_RED, width=1.8)
    report.add_graph("Calibration Plot", rg.finish(g, data.xtitle, data.ytitle, True),
                     width=520)
    return report, f"{_stamp(book)}\nCalibration of {data.range_text}: y = " \
                   f"{num(c.intercept)} + {num(c.slope)}·x, R² = {num(c.r2)}, LOD = " \
                   f"{num(c.lod)}, LOQ = {num(c.loq)} → [{book.name}]{report.name}"


# ====================================================================== spectroscopy
def convert_units(data: XYData, from_unit: str, to_unit: str) -> tuple[int, str]:
    """Convert the X column of ``data`` into a new X column of the source sheet."""
    if data.xcol is None:
        raise AnalysisError("The data have no X column to convert.")
    ws = data.sheet
    vals = ws.columns[data.xcol].numeric(ws.n_rows)
    out = sp.convert(vals, from_unit, to_unit)
    idx = _append_column(ws, sp.UNITS[to_unit], np.arange(ws.n_rows), out,
                         f"converted from {sp.UNITS[from_unit]}", designation="X")
    ws.columns[idx].units = to_unit.replace("cm-1", "cm^-1")
    return idx, f"{_stamp(data.book)}\nUnit conversion {from_unit} → {to_unit} of column " \
                f"{ws.columns[data.xcol].name} → column {ws.columns[idx].name}"


def kubelka_munk_column(data: XYData) -> tuple[int, str]:
    v = sp.kubelka_munk(data.y)
    idx = _append_column(data.sheet, f"F(R) of {data.yname}", data.rows, v,
                         "Kubelka-Munk (1 − R)²/2R")
    return idx, f"{_stamp(data.book)}\nKubelka-Munk of {data.range_text} → column " \
                f"{data.sheet.columns[idx].name}"


def absorbance_column(data: XYData, to: str = "transmittance") -> tuple[int, str]:
    if to == "transmittance":
        v, name = sp.absorbance_to_transmittance(data.y), f"%T of {data.yname}"
    else:
        v, name = sp.transmittance_to_absorbance(data.y), f"Absorbance of {data.yname}"
    idx = _append_column(data.sheet, name, data.rows, v, "")
    return idx, f"{_stamp(data.book)}\n{name} → column {data.sheet.columns[idx].name}"


def tauc_report(data: XYData, x_unit: str = "nm", transition: str = "direct",
                fit_range=None) -> tuple[ReportSheet, Worksheet, str]:
    try:
        t = sp.tauc(data.x, data.y, x_unit, transition, fit_range)
    except ValueError as exc:
        raise _err(exc) from None
    book = data.book
    report = ReportSheet(title=f"Tauc Plot of {data.range_text}")
    add_report(book, report, "TaucPlot")
    report.add_table("Band Gap", ["", "Value"],
                     [["Transition", sp.TAUC[transition][0]],
                      ["Linear Fit Range (eV)", f"{num(t.fit_range[0])} – {num(t.fit_range[1])}"],
                      ["Slope", num(t.slope)], ["Intercept", num(t.intercept)],
                      ["R-Square", num(t.r2)], ["Band Gap Eg (eV)", num(t.band_gap)]],
                     ["Eg is the energy-axis intercept of the linear part of the Tauc plot."])
    ws = Worksheet(name=book.unique_sheet_name("TaucData"), n_rows=len(t.energy))
    _curve_cols(ws, t.energy, [("(αhν)^(1/n)", t.tauc)], "Energy (eV)")
    ws.columns[0].long_name, ws.columns[0].units = "Photon Energy", "eV"
    book.sheets.append(ws)
    g = rg.new_graph("Tauc Plot", 12.5, 8.5)
    rg.add_series(g, t.energy, t.tauc, "(αhν)^(1/n)", "line_symbol", "#000000", size=3.0)
    e = np.linspace(t.band_gap, t.fit_range[1], 50)
    rg.add_series(g, e, t.slope * e + t.intercept, "Linear Fit", "line", rg.FIT_RED,
                  width=1.6)
    report.add_graph("Tauc Plot", rg.finish(g, "hν (eV)", "(αhν)^(1/n)", True), width=520)
    return report, ws, f"{_stamp(book)}\nTauc plot of {data.range_text}: Eg = " \
                       f"{num(t.band_gap)} eV → [{book.name}]{report.name}"

"""Output Viewer sections for the DOE module, worded like Design-Expert reports.

Each public function returns one :class:`Section`: design summary (with alias structure),
Fit Summary, ANOVA (fit statistics, coefficients, coded and actual equations), effects
(half-normal / Pareto), diagnostics, model graphs, optimisation, point prediction and Taguchi
signal-to-noise analysis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from rsstatchem.core.doe import designs, glm, model, optimize, plots
from rsstatchem.core.doe.glm import is_glm
from rsstatchem.core.doe.model import Analysis, Fit, Space, term_name
from rsstatchem.core.output import Section, Table


# ---------------------------------------------------------------------------- formatting
def num(x, sig: int = 4) -> str:
    """Design-Expert style: about four significant digits, leading zero kept."""
    if x is None:
        return ""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(x):
        return ""
    if x == 0:
        return "0"
    a = abs(x)
    if a < 1e-4 or a >= 1e9:
        return f"{x:.{sig - 1}E}"
    d = max(0, sig - 1 - int(math.floor(math.log10(a))))
    return f"{x:.{d}f}"


def fixed(x, d: int = 4) -> str:
    if x is None or not math.isfinite(float(x)):
        return ""
    return f"{float(x):.{d}f}"


def pval(p) -> str:
    if p is None or not math.isfinite(float(p)):
        return ""
    return "< 0.0001" if p < 0.0001 else f"{p:.4f}"


def fval(f) -> str:
    if f is None or not math.isfinite(float(f)):
        return ""
    return f"{f:.2f}"


def signed(x, sig: int = 4) -> str:
    s = num(x, sig)
    return s if s.startswith("-") else "+" + s


def _p_eq(p) -> str:
    """'= 0.0123' or '< 0.0001' for running text."""
    t = pval(p)
    return t if t.startswith("<") else f"= {t}"


def _sig_text(p, alpha=0.05) -> str:
    if p is None or not math.isfinite(p):
        return ""
    return "significant" if p < alpha else "not significant"


def model_label(space: Space, terms: list) -> str:
    """'Quadratic', 'Reduced Quadratic', ... from the term set."""
    tset = set(terms)
    if not tset:
        return "Mean"
    base = [o for o in model.model_orders(space) if o[1] != "mean"]
    higher = [o for o in model.all_model_orders(space) if o[1] != "mean" and o not in base]
    if space.combined:  # every mixture x process combination, smallest first
        base = sorted(base + higher, key=lambda o: len(model.model_terms(space, o[1])))
        higher = []
    for group in (base, higher):
        for name, order in group:
            if tset == set(model.model_terms(space, order)):
                return name
    for group in (base, higher):
        for name, order in group:
            if tset <= set(model.model_terms(space, order)):
                return f"Reduced {name}"
    return "Custom"


# ---------------------------------------------------------------------------- design summary
def design_section(meta: dict, notes: list[str] | None = None, ds=None) -> Section:
    sec = Section("Design Summary")
    sec.log(f"DESIGN {meta['design'].upper()} FACTORS={len(meta['factors'])} "
            f"RUNS={meta['runs']} BLOCKS={meta['blocks']}"
            + (f" SEED={meta['seed']}" if meta.get("seed") is not None else ""))
    study = {"process": "Response Surface", "mixture": "Mixture",
             "combined": "Combined Mixture-Process"}.get(meta.get("kind"), "Custom")
    if meta["design"] in designs.FACTORIAL_DESIGNS:
        study = "Factorial"
    elif meta["design"] == "taguchi":
        study = "Taguchi"
    rows = [["Study Type", study], ["Design Type", meta["title"]],
            ["Subtype", "Randomized" if meta.get("randomized") else "Not randomized"],
            ["Runs", str(meta["runs"])], ["Blocks", str(meta["blocks"])]]
    if meta.get("center_points"):
        rows.append(["Center Points", str(meta["center_points"])])
    if meta.get("replicates", 1) > 1:
        rows.append(["Replicates", str(meta["replicates"])])
    if "alpha" in meta:
        rows.append(["Alpha", f"{meta['alpha']:.5g} ({meta.get('alpha_type', '')}, "
                              f"{meta.get('ccd_type', '')})"])
    if meta.get("generators"):
        rows.append(["Generators", ", ".join(meta["generators"])])
        rows.append(["Defining Relation", " = ".join(meta.get("defining_relation", []))])
    if meta.get("resolution"):
        rows.append(["Resolution", designs.ROMAN.get(meta["resolution"],
                                                     str(meta["resolution"]))])
    if meta.get("block_generators"):
        rows.append(["Block Generators", ", ".join(meta["block_generators"])])
    if meta.get("taguchi"):
        rows.append(["Orthogonal Array", meta["taguchi"]])
    if meta.get("min_run"):
        rows.append(["Min-Run Type", meta["min_run"]])
    if meta.get("split_plot"):
        sp = meta["split_plot"]
        rows.append(["Split-Plot", f"{sp.get('groups')} groups; hard to change: "
                     + ", ".join(designs.letter(j) for j in sp.get("hard", []))])
    if meta.get("model"):
        rows.append(["Design Model", str(meta["model"]).replace("_", " ").title()])
    if meta.get("mixture"):
        rows.append(["Mixture Total", num(meta["mixture"]["total"])])
    if meta.get("seed") is not None:
        rows.append(["Random Seed", str(meta["seed"])])
    sec.table(Table("Build Information", ["", "Value"], rows))
    frows = []
    for j, f in enumerate(meta["factors"]):
        if f["type"] == "categoric":
            frows.append([designs.letter(j), f["name"], f.get("units", ""), "Categoric",
                          str(len(f["levels"])), ", ".join(map(str, f["labels"])), "", ""])
        elif f["type"] == "mixture":
            frows.append([designs.letter(j), f["name"], f.get("units", ""), "Mixture",
                          num(f["low"]), num(f["high"]), "", ""])
        else:
            frows.append([designs.letter(j), f["name"], f.get("units", ""), "Numeric",
                          num(f["low"]), num(f["high"]), f"-1 ↔ {num(f['low'])}",
                          f"+1 ↔ {num(f['high'])}"])
    cols = ["Factor", "Name", "Units", "Type", "Low / Levels", "High / Labels",
            "Coded Low", "Coded High"]
    sec.table(Table("Factors", cols, frows))
    sec.table(Table("Responses", ["Response", "Name", "Units"],
                    [[f"R{i + 1}", r["name"], r.get("units", "")]
                     for i, r in enumerate(meta["responses"])]))
    if meta.get("aliases"):
        sec.table(Table("Alias Structure", ["Aliases"], [[a] for a in meta["aliases"]],
                        row_header_cols=0,
                        footnotes=["Terms of order 4 and higher are omitted from the "
                                   "alias lists."]))
    elif meta["design"] == "factorial":
        sec.note("Alias Structure", "Full factorial: no effects are aliased.")
    if ds is not None and ds.n_cases <= 200:
        names = ["Std", "Run", "Block", "Space_Type"] + [f["column"] for f in meta["factors"]]
        names = [n for n in names if ds.has_name(n)]
        trows = []
        for i in range(ds.n_cases):
            row = []
            for n in names:
                var = ds.var(n)
                v = ds.df.at[i, var.name]
                lab = var.value_labels.get(v) if var.value_labels else None
                row.append(lab if lab is not None else (num(v) if var.decimals else
                                                        f"{v:.0f}"))
            trows.append(row)
        sec.table(Table("Design Layout", [ds.var(n).title if n in ("Std", "Run") else
                                          ds.var(n).label or n for n in names], trows,
                        row_header_cols=0))
    for n in notes or []:
        sec.note("", n)
    sec.note("", "Enter the measured responses in the Data Editor, then use RSM > Fit "
                 "Summary && Analysis.".replace("&&", "&"))
    return sec


# ---------------------------------------------------------------------------- fit summary
def glm_fit_summary_table(fs) -> Table:
    seq = {r[0].split(" vs ")[0]: r for r in fs.sequential}
    rows = []
    for src, m2ll, aicc, bic, pseudo, remark in fs.extra:
        r = seq.get(src)
        rows.append([src, fval(r[3]) if r else "", str(int(r[2])) if r and r[2] else "",
                     pval(r[4]) if r else "", num(m2ll, 6), num(aicc, 6), num(bic, 6),
                     fixed(pseudo), remark])
    return Table("Fit Summary (Likelihood Ratio Tests)",
                 ["Source", "Chi-Square", "df", "Sequential p-value", "-2 Log Likelihood",
                  "AICc", "BIC", "Pseudo R²", ""], rows,
                 footnotes=["Each order is tested against the previous one by the drop in "
                            "deviance (likelihood-ratio chi-square). Pseudo R² = 1 - "
                            "deviance / null deviance. Prefer the suggested model or the "
                            "lowest AICc."])


def fit_summary_tables(sec: Section, a: Analysis) -> None:
    fs = a.summary
    if fs.kind == "glm":
        sec.table(glm_fit_summary_table(fs))
        sec.note("Suggested Model", f"{fs.suggested} model.")
        return
    rows = []
    for src, ss, df, f, p, remark in fs.sequential:
        ms = ss / df if df else None
        rows.append([src, num(ss), str(int(df)), num(ms), fval(f), pval(p), remark])
    sec.table(Table("Sequential Model Sum of Squares [Type I]",
                    ["Source", "Sum of Squares", "df", "Mean Square", "F-value", "p-value", ""],
                    rows, footnotes=["Select the highest order polynomial where the "
                                     "additional terms are significant and the model is not "
                                     "aliased."]))
    if fs.pure_error[1] > 0:
        rows = []
        for src, ss, df, f, p, remark in fs.lof:
            rows.append([src, num(ss), str(int(df)), num(ss / df) if df else "", fval(f),
                         pval(p), remark])
        ss_pe, df_pe = fs.pure_error
        rows.append(["Pure Error", num(ss_pe), str(df_pe), num(ss_pe / df_pe), "", "", ""])
        sec.table(Table("Lack of Fit Tests", ["Source", "Sum of Squares", "df", "Mean Square",
                                              "F-value", "p-value", ""], rows,
                        footnotes=["Select the model with insignificant lack-of-fit."]))
    else:
        sec.note("Lack of Fit Tests", "No replicated design points: pure error and the "
                                      "lack-of-fit test are not available.")
    rows = []
    for src, sd, r2, adj, pred, press, remark in fs.summary:
        rows.append([src, num(sd), fixed(r2), fixed(adj), fixed(pred), num(press), remark])
    sec.table(Table("Model Summary Statistics", ["Source", "Std. Dev.", "R²", "Adjusted R²",
                                                 "Predicted R²", "PRESS", ""], rows,
                    footnotes=["Focus on the model maximizing the Adjusted R² and the "
                               "Predicted R²."]))
    sec.note("Suggested Model", f"{fs.suggested} model"
             + (" (no model terms are significant)" if fs.suggested == "Mean" else "") + ".")


def fit_summary_section(a: Analysis) -> Section:
    label = a.fit.transform.label(a.fit.response_name)
    sec = Section(f"Fit Summary: {label}")
    sec.log(f"DOE FITSUMMARY RESPONSE={a.spec.response} "
            f"TRANSFORM={a.fit.transform.describe()}")
    fit_summary_tables(sec, a)
    return sec


# ---------------------------------------------------------------------------- ANOVA
def anova_table(fit: Fit, name: str) -> tuple[Table, list[str]]:
    if is_glm(fit):
        return glm_anova_table(fit, name)
    if is_reml(fit):
        return reml_anova_table(fit, name)
    a = fit.anova()
    rows, interp = [], []
    ms_res = fit.ms_res
    if a["block"]:
        ss, df = a["block"]
        rows.append(["Block", num(ss), str(df), num(ss / df), "", "", ""])
    ss_m, df_m = a["model"]
    f_m, p_m = fit.model_test()
    rows.append(["Model", num(ss_m), str(df_m), num(ss_m / df_m) if df_m else "", fval(f_m),
                 pval(p_m), _sig_text(p_m)])
    sig_terms = []
    for src, ss, df, t in a["terms"]:
        f = model.ratio(ss / df, ms_res) if df else float("nan")
        p = float(stats.f.sf(f, df, fit.df_res))
        rows.append([f"  {src}", num(ss), str(df), num(ss / df), fval(f), pval(p), ""])
        if p < 0.05:
            sig_terms.append(src.split("-")[0] if t and len(t) == 1 else src)
    curv = a.get("curvature")
    if curv:
        rows.append(["Curvature", num(curv["ss"]), "1", num(curv["ss"]), fval(curv["f"]),
                     pval(curv["p"]), _sig_text(curv["p"])])
    rows.append(["Residual", num(fit.ss_res), str(fit.df_res), num(ms_res), "", "", ""])
    if a["lof"]:
        ss_l, df_l, ss_pe, df_pe = a["lof"]
        f_l = (ss_l / df_l) / (ss_pe / df_pe) if ss_pe > 0 else float("nan")
        p_l = float(stats.f.sf(f_l, df_l, df_pe)) if ss_pe > 0 else float("nan")
        rows.append(["  Lack of Fit", num(ss_l), str(df_l), num(ss_l / df_l), fval(f_l),
                     pval(p_l), _sig_text(p_l)])
        rows.append(["  Pure Error", num(ss_pe), str(df_pe), num(ss_pe / df_pe), "", "", ""])
        if math.isfinite(p_l):
            interp.append(f"The Lack of Fit F-value of {f_l:.2f} implies the Lack of Fit is "
                          f"{_sig_text(p_l)} relative to the pure error (p = {pval(p_l)}). "
                          + ("Non-significant lack of fit is good: the model fits."
                             if p_l >= 0.05 else "Significant lack of fit is bad."))
    rows.append(["Cor Total", num(a["cor_total"][0]), str(a["cor_total"][1]), "", "", "", ""])
    if curv and math.isfinite(curv["p"]):
        interp.append(f"The curvature F-value of {curv['f']:.2f} implies the curvature (the "
                      "difference between the average of the centre points and the average of "
                      f"the factorial points) is {_sig_text(curv['p'])} (p = "
                      f"{pval(curv['p'])}). " + (
                          "Significant curvature means a quadratic model is needed: augment "
                          "the design with axial points." if curv["p"] < 0.05 else
                          "The factorial model describes the design space adequately."))
    if math.isfinite(p_m):
        if p_m < 0.05:
            chance = "less than 0.01%" if p_m < 0.0001 else f"only a {100 * p_m:.2f}%"
            interp.insert(0, f"The Model F-value of {f_m:.2f} implies the model is significant."
                             f" There is {chance} chance that an F-value this "
                             "large could occur due to noise.")
        else:
            interp.insert(0, f"The Model F-value of {f_m:.2f} implies the model is not "
                             f"significant relative to the noise (a {100 * p_m:.2f}% chance "
                             "that an F-value this large could occur due to noise).")
    if sig_terms:
        interp.insert(1, "P-values less than 0.0500 indicate model terms are significant. "
                         f"In this case {', '.join(sig_terms)} are significant model terms. "
                         "Values greater than 0.1000 indicate the model terms are not "
                         "significant.")
    title = f"ANOVA for {name} model"
    foot = ["Sum of squares Type III (partial)."]
    if curv:
        foot.append("Curvature: centre points vs. factorial points, 1 df, removed from the "
                    "residual; the model graphs and predictions use the factorial model.")
    if fit.space.mixture:
        foot.append("Mixture components coded as L_Pseudo components; the Linear Mixture "
                    "row tests the linear blending terms against the mean.")
    return Table(title, ["Source", "Sum of Squares", "df", "Mean Square", "F-value",
                         "p-value", ""], rows, footnotes=foot), interp


def is_reml(fit) -> bool:
    return isinstance(fit, model.SplitPlotFit)


def glm_anova_table(fit, name: str) -> tuple[Table, list[str]]:
    """Likelihood-ratio (deviance) tests of a logistic / Poisson model."""
    a = fit.anova()
    rows, interp, sig = [], [], []
    chi_m, df_m, p_m = a["model"]
    rows.append(["Model", fval(chi_m), str(df_m), pval(p_m), _sig_text(p_m)])
    for src, chi, df, p, t in a["terms"]:
        rows.append([f"  {src}", fval(chi), str(df), pval(p), ""])
        if math.isfinite(p) and p < 0.05:
            sig.append(src.split("-")[0] if t and len(t) == 1 else src)
    dev, df_r = a["deviance"]
    rows.append(["Residual (Deviance)", num(dev), str(df_r), "", ""])
    if math.isfinite(p_m):
        interp.append(f"The likelihood-ratio chi-square of {chi_m:.2f} on {df_m} df implies the "
                      f"model is {_sig_text(p_m)} (p = {pval(p_m)}).")
    if sig:
        interp.append("P-values less than 0.0500 indicate model terms are significant. In this "
                      f"case {', '.join(sig)} are significant model terms.")
    interp += list(getattr(fit, "warnings", []))
    foot = [f"Distribution: {fit.family_title}; maximum likelihood by iteratively reweighted "
            "least squares.",
            "Each term: likelihood-ratio chi-square of removing it from the model (Type III); "
            "the model row tests all terms against the intercept (and blocks)."]
    return Table(f"Deviance Table for {name} model", ["Source", "Chi-Square", "df",
                                                      "p-value", ""], rows,
                 footnotes=foot), interp


def glm_fit_statistics_table(fit) -> tuple[Table, list[str]]:
    s = fit.fit_statistics()
    rows = [["-2 Log Likelihood", num(s["m2ll"], 6), "Pseudo R²", fixed(s["pseudo_r2"])],
            ["AICc", num(s["aicc"], 6), "McFadden R²", fixed(s["mcfadden_r2"])],
            ["BIC", num(s["bic"], 6), "Dispersion (Pearson / df)", num(s["dispersion"])],
            ["Deviance", num(s["deviance"]), "Null Deviance", num(s["null_deviance"])]]
    interp = ["Pseudo R² = 1 - deviance / null deviance (the null model has the intercept "
              "only). A dispersion well above 1 suggests overdispersion."]
    items = Table("Fit Statistics", ["", "Value", "", "Value"], rows)
    return items, interp


def goodness_of_fit_table(fit) -> Table:
    rows = [[name, num(chi), str(df), pval(p), "poor fit" if p < 0.05 else "adequate"]
            for name, chi, df, p in fit.goodness_of_fit()]
    return Table("Goodness of Fit Tests", ["Test", "Chi-Square", "df", "p-value", ""], rows,
                 footnotes=["Deviance and Pearson tests are reliable for grouped responses "
                            "(several trials per run or larger counts); the Hosmer-Lemeshow "
                            "test groups binary runs by fitted probability."])


def reml_anova_table(fit, name: str) -> tuple[Table, list[str]]:
    """Design-Expert's "ANOVA (REML)": whole-plot and subplot terms tested separately with
    Kenward-Roger F tests (term df, error df, F-value, p-value)."""
    a = fit.anova()
    rows, interp, sig = [], [], []

    def section(label, test, terms):
        if test is not None:
            F, l, m, p = test
            rows.append([label, str(int(l)), num(m, 4), fval(F), pval(p), _sig_text(p)])
        else:
            rows.append([label, "", "", "", "", ""])
        for t, src, F, l, m, p in terms:
            rows.append([f"  {src}", str(l), num(m, 4), fval(F), pval(p), ""])
            if p < 0.05:
                sig.append(src.split("-")[0] if len(t) == 1 else src)

    if a["block"]:
        rows.append(["Block", str(a["block"]), "", "", "", ""])
    section("Whole-plot", a["whole_test"], a["whole"])
    section("Subplot", a["sub_test"], a["sub"])
    for label, test in (("whole-plot", a["whole_test"]), ("subplot", a["sub_test"])):
        if test is not None and math.isfinite(test[3]):
            interp.append(f"The {label} terms together have F = {test[0]:.2f} on {int(test[1])} "
                          f"and {test[2]:.4g} df (p {_p_eq(test[3])}): "
                          + ("significant." if test[3] < 0.05 else "not significant."))
    if sig:
        interp.append("P-values less than 0.0500 indicate model terms are significant. In this "
                      f"case {', '.join(sig)} are significant model terms.")
    foot = ["REML (restricted maximum likelihood) with the groups (whole plots) as a random "
            "effect; F tests and error degrees of freedom by the Kenward-Roger method.",
            "Whole-plot terms involve only hard-to-change factors; they are tested against the "
            "between-group variation, the subplot terms against the within-group (residual) "
            "variation."]
    return Table(f"ANOVA (REML) for {name} model", ["Source", "Term df", "Error df",
                                                    "F-value", "p-value", ""], rows,
                 footnotes=foot), interp


def variance_components_table(fit) -> Table:
    rows = []
    for v in fit.variance_components():
        rows.append([v["source"], num(v["variance"]), num(v["se"]), num(v["low"]),
                     num(v["high"]), f"{v['percent']:.2f}"])
    info = fit.information()
    rows.append(["Total", num(fit.s2 + fit.s2_wp), "", "", "", "100.00"])
    rows.append(["Ratio (Group / Residual)", num(fit.eta), "", "", "", ""])
    rows.append(["-2 Res. Log Likelihood", num(info["m2ll_reml"], 6), "", "", "", ""])
    rows.append(["AICc (ML)", num(info["aicc"], 6), "", "", "", ""])
    rows.append(["BIC (ML)", num(info["bic"], 6), "", "", "", ""])
    return Table("Variance Components", ["Source", "Variance", "Std. Error", "95% CI Low",
                                         "95% CI High", "% of Total"], rows,
                 footnotes=["Group = whole-plot (restriction on randomisation) variance; "
                            "standard errors from the inverse REML information.",
                            "95% CI: profile-likelihood intervals — the variances whose "
                            "restricted likelihood-ratio statistic (the other variance "
                            "re-estimated) stays below the chi-square(1) quantile 3.84; the "
                            "group variance is bounded below by 0, so the interval never "
                            "goes negative.",
                            "AICc / BIC use the maximum likelihood (k = coefficients + 2 "
                            "variances) for comparing models."])


def reml_fit_statistics_table(fit) -> tuple[Table, list[str]]:
    s = fit.fit_statistics()
    info = fit.information()
    rows = [["Std. Dev.", num(s["std_dev"]), "R²", fixed(s["r2"])],
            ["Mean", num(s["mean"]), "Adjusted R²", fixed(s["adj_r2"])],
            ["C.V. %", num(s["cv"]), "AICc", num(info["aicc"], 6)],
            ["-2 Res. LL", num(info["m2ll_reml"], 6), "BIC", num(info["bic"], 6)]]
    interp = ["Std. Dev. is the square root of the residual (subplot) variance. R² compares the "
              "fixed-effect predictions with the total variation; Predicted R² and PRESS are "
              "not defined for REML fits."]
    return Table("Fit Statistics", ["", "Value", "", "Value"], rows), interp


def fit_statistics_table(fit: Fit) -> tuple[Table, list[str]]:
    if is_glm(fit):
        return glm_fit_statistics_table(fit)
    if is_reml(fit):
        return reml_fit_statistics_table(fit)
    s = fit.fit_statistics()
    info = fit.information()
    dw = fit.durbin_watson()
    rows = [["Std. Dev.", num(s["std_dev"]), "R²", fixed(s["r2"])],
            ["Mean", num(s["mean"]), "Adjusted R²", fixed(s["adj_r2"])],
            ["C.V. %", num(s["cv"]), "Predicted R²", fixed(s["pred_r2"])],
            ["PRESS", num(s["press"]), "Adeq Precision", num(s["adeq"])],
            ["-2 Log Likelihood", num(info["m2ll"], 6), "AICc", num(info["aicc"], 6)],
            ["BIC", num(info["bic"], 6), "Durbin-Watson", fixed(dw, 3)]]
    interp = []
    if math.isfinite(s["pred_r2"]) and math.isfinite(s["adj_r2"]):
        diff = s["adj_r2"] - s["pred_r2"]
        if diff < 0.2:
            interp.append(f"The Predicted R² of {s['pred_r2']:.4f} is in reasonable agreement "
                          f"with the Adjusted R² of {s['adj_r2']:.4f}; the difference is less "
                          "than 0.2.")
        else:
            interp.append(f"The Predicted R² of {s['pred_r2']:.4f} is not as close to the "
                          f"Adjusted R² of {s['adj_r2']:.4f} as one might expect (difference "
                          "> 0.2): consider model reduction, a transformation or outliers.")
        if s["pred_r2"] < 0:
            interp.append("A negative Predicted R² implies that the overall mean may be a "
                          "better predictor than the current model.")
    if math.isfinite(s["adeq"]):
        interp.append(f"Adeq Precision measures the signal to noise ratio; a ratio greater "
                      f"than 4 is desirable. The ratio of {s['adeq']:.3f} indicates "
                      + ("an adequate signal." if s["adeq"] > 4 else "an inadequate signal."))
    if math.isfinite(dw) and (dw < 1.0 or dw > 3.0):
        interp.append(f"The Durbin-Watson statistic of {dw:.3f} (residuals in run order) is far "
                      "from 2: look for a time trend or serial correlation on the Residuals "
                      "vs. Run plot.")
    return Table("Fit Statistics", ["", "Value", "", "Value"], rows), interp


def glm_coefficient_table(fit, level: float = 0.95) -> Table:
    pct = f"{level * 100:g}%"
    ratio_name = "Odds Ratio" if fit.family == "binomial" else "Rate Ratio"
    rows = []
    for c in fit.coefficients(level):
        rows.append([c["label"], num(c["estimate"]), num(c["se"]), fval(c["z"]), pval(c["p"]),
                     num(c["low"]), num(c["high"]),
                     num(c["ratio"]) if c["label"] != "Intercept" else "",
                     num(c["vif"]) if c["vif"] is not None else ""])
    return Table("Coefficients in Terms of Coded Factors",
                 ["Factor", "Coefficient Estimate", "Standard Error", "z-value", "p-value",
                  f"{pct} CI Low", f"{pct} CI High", ratio_name, "VIF"], rows,
                 footnotes=[f"Coefficients on the {'logit' if fit.family == 'binomial' else 'log'}"
                            " scale; Wald z tests and intervals. The "
                            f"{ratio_name.lower()} is exp(coefficient): the factor by which "
                            f"the {'odds' if fit.family == 'binomial' else 'expected count'} "
                            "change per coded unit."])


def coefficient_table(fit: Fit, level: float = 0.95) -> Table:
    if is_glm(fit):
        return glm_coefficient_table(fit, level)
    pct = f"{level * 100:g}%"
    rows = []
    reml = is_reml(fit)
    for c in fit.coefficients(level):
        row = [c["label"] if c["label"] != "Intercept" else "Intercept",
               num(c["estimate"]), "1", num(c["se"]), num(c["low"]), num(c["high"]),
               num(c["vif"]) if c["vif"] is not None else ""]
        if reml:
            row[2] = num(c["df"], 4) if c.get("df") is not None else ""
        rows.append(row)
    title = "Coefficients in Terms of Coded Factors"
    if reml:
        return Table(title, ["Factor", "Coefficient Estimate", "Error df", "Standard Error",
                             f"{pct} CI Low", f"{pct} CI High", "VIF"], rows,
                     footnotes=["REML estimates (GLS); standard errors and error df by the "
                                "Kenward-Roger method."])
    if fit.space.mixture or fit.space.combined:
        title = "Coefficients in Terms of L_Pseudo Components" if fit.space.mixture else \
            "Coefficients in Terms of L_Pseudo Components and Coded Factors"
    foot = ["The coefficient estimate is the expected change in response per unit change in "
            "the coded factor value when all remaining factors are held constant."]
    if getattr(fit.mat, "curv_slice", None) is not None:
        foot.append("Curvature = average of the centre points minus the average of the "
                    "factorial points; it is not part of the prediction equation.")
    if fit.space.slack is not None:
        foot.append(f"Slack model: component {fit.space.factors[fit.space.slack].title} is "
                    "left out (it takes up the rest of the mixture).")
    return Table(title, ["Factor", "Coefficient Estimate", "df", "Standard Error",
                         f"{pct} CI Low", f"{pct} CI High", "VIF"], rows, footnotes=foot)


def equation_tables(fit: Fit) -> list[Table]:
    label = fit.transform.label(fit.response_name)
    out = []
    coded = fit.coded_equation()
    title = ("Final Equation in Terms of L_Pseudo Components" if fit.space.mixture else
             "Final Equation in Terms of Coded Factors")
    rows = [[label + " =", ""]] + [[signed(c), ("* " + t) if t else ""] for c, t in coded]
    foot = ["The coded equation identifies the relative impact of the factors by comparing "
            "the factor coefficients (high levels coded +1, low levels -1)."]
    if fit.space.mixture:
        foot = ["Components in L_Pseudo coding (fractions of the constrained range)."]
    out.append(Table(title, ["", ""], rows, row_header_cols=0, footnotes=foot))
    eqs = fit.actual_equations()
    if fit.space.mixture:
        title = "Final Equation in Terms of Actual Components"
    else:
        title = "Final Equation in Terms of Actual Factors"
    if fit.space.mixture and fit.space.slack is None:
        extra = [("real", "Final Equation in Terms of Real Components",
                  "Real components: proportions of the mixture total."),
                 ("upseudo", "Final Equation in Terms of U_Pseudo Components",
                  "U_Pseudo components u = (U - x) / (sum U - total), from the upper limits.")]
        for scale, etitle, note in extra:
            if scale == "real" and abs(fit.space.total - 1) < 1e-12:
                continue
            eq = model.mixture_actual_equation(fit, scale)
            if eq is not None:
                rows = [[label + " =", ""]] + [[signed(c, 6), ("* " + t) if t else ""]
                                                for c, t in eq]
                out.append(Table(etitle, ["", ""], rows, row_header_cols=0, footnotes=[note]))
    if not eqs:
        return out
    if fit.n_blocks:
        foot2 = ["Block effects are averaged out of the equations."]
    else:
        foot2 = []
    foot2.append("Use the actual-factor equation for predictions in original units; it does "
                 "not show the relative importance of factors because coefficients are scaled "
                 "to the units of each factor.")
    for heading, eq in eqs:
        rows = [[label + " =", ""]] + [[signed(c, 6), ("* " + t) if t else ""]
                                        for c, t in eq]
        out.append(Table(title + (f" ({heading})" if heading else ""), ["", ""], rows,
                         row_header_cols=0, footnotes=foot2))
    return out


def anova_section(a: Analysis, show_summary: bool = True, show_anova: bool = True,
                  show_coef: bool = True, show_equations: bool = True,
                  level: float = 0.95) -> Section:
    fit = a.fit
    label = fit.transform.label(fit.response_name)
    name = model_label(a.space, fit.terms)
    sec = Section(f"Analysis: {fit.response_name}")
    terms_txt = " ".join(term_name(t) for t in fit.terms) or "(mean only)"
    sec.log(f"DOE ANALYZE RESPONSE={a.spec.response} MODEL={name.upper()} TERMS={terms_txt}\n"
            f"  TRANSFORM={fit.transform.describe()} REDUCTION={a.spec.reduction.upper()}"
            + (f" ALPHA_OUT={a.spec.alpha_out:g}" if a.spec.reduction == "backward" else ""))
    info = [["Response", f"{fit.response_name}"],
            ["Transform", fit.transform.describe()],
            ["Model", name],
            ["Observations", str(fit.n)]]
    sec.table(Table("Response", ["", "Value"], info))
    if show_summary:
        fit_summary_tables(sec, a)
    if a.dropped:
        sec.note("Aliased Terms", "Removed because they are aliased in this design: "
                 + ", ".join(term_name(t) for t in a.dropped) + ".", style="warning")
    if a.steps:
        sec.table(selection_table(a.spec, a.steps))
    if show_anova:
        t, interp = anova_table(fit, name)
        sec.table(t)
        st, interp2 = fit_statistics_table(fit)
        sec.table(st)
        if is_reml(fit):
            sec.table(variance_components_table(fit))
        if is_glm(fit):
            sec.table(goodness_of_fit_table(fit))
        sec.note("Interpretation", "\n".join(interp + interp2))
    if show_coef:
        sec.table(coefficient_table(fit, level))
        for t in treatment_tables(fit, level):
            sec.table(t)
    if show_equations:
        for t in equation_tables(fit):
            sec.table(t)
    if label != fit.response_name:
        sec.note("", f"Analysis on the transformed scale {label}; predictions are "
                     "back-transformed to original units.")
    return sec


def selection_table(spec, steps: list) -> Table:
    """The Auto Select history: [(action, term, value)]."""
    crit = dict(model.SELECTION_CRITERIA).get(spec.criterion, spec.criterion)
    method = dict(model.SELECTION_METHODS).get(spec.reduction, spec.reduction)
    rows = []
    for i, (action, t, v) in enumerate(steps):
        rows.append([str(i + 1), {"add": "Added", "remove": "Removed"}.get(
            action, "Added for hierarchy"), term_name(t),
            pval(v) if spec.criterion == "p" else num(v)])
    if spec.criterion == "p":
        alphas = []
        if spec.reduction in ("forward", "stepwise"):
            alphas.append(f"Alpha in = {spec.alpha_in:g}")
        if spec.reduction in ("backward", "stepwise"):
            alphas.append(f"Alpha out = {spec.alpha_out:g}")
        title = f"{method} Selection Regression ({', '.join(alphas)})"
    else:
        title = f"{method} Selection by {crit}"
    foot = ["Hierarchy kept: a term is only removed when no higher-order term in the model "
            "contains it; parents of selected terms are added at the end." if spec.hierarchy
            else "Hierarchy not enforced."]
    if spec.criterion != "p":
        foot.append(f"{crit} = -2 ln L + penalty (AICc: 2k + 2k(k+1)/(n-k-1); BIC: k ln n), "
                    "k = coefficients + 1; a step is taken while it lowers the criterion.")
    return Table(title, ["Step", "Action", "Term", "p-value" if spec.criterion == "p"
                         else crit], rows, footnotes=foot)


# ---------------------------------------------------------------------------- effects
def effects_table(eff: list[dict], title: str = "Effects List") -> Table:
    rows = [[e["name"], num(e["effect"]), num(e["ss"]), f"{e['contribution']:.2f} %",
             fval(e.get("t"))] for e in sorted(eff, key=lambda e: -abs(e["effect"]))]
    return Table(title, ["Term", "Standardized Effect", "Sum of Squares", "% Contribution",
                         "t-Value (Lenth)"], rows,
                 footnotes=[f"Lenth's PSE = {num(eff[0]['pse'])}; t-value limit = "
                            f"{eff[0]['t_limit']:.3f}; Bonferroni limit = "
                            f"{eff[0]['bonferroni']:.3f}."] if eff else [])


SPLIT_EFFECTS = (("sub", "Subplot Effects"), ("whole", "Whole-plot Effects"))


def effects_section(a: Analysis, half_normal: bool = True, pareto: bool = True) -> Section:
    if a.space.split_plot and a.obs.groups is not None:
        return split_effects_section(a, half_normal, pareto)
    eff = model.factorial_effects(a.space, a.obs, transform=a.fit.transform)
    sec = Section(f"Effects: {a.fit.response_name}")
    sec.log(f"DOE EFFECTS RESPONSE={a.spec.response}")
    rows = [[e["name"], num(e["effect"]), num(e["ss"]), f"{e['contribution']:.2f} %",
             fval(e.get("t"))] for e in sorted(eff, key=lambda e: -abs(e["effect"]))]
    sec.table(Table("Effects List", ["Term", "Standardized Effect", "Sum of Squares",
                                     "% Contribution", "t-Value (Lenth)"], rows,
                    footnotes=[f"Lenth's PSE = {num(eff[0]['pse'])}; t-value limit = "
                               f"{eff[0]['t_limit']:.3f}; Bonferroni limit = "
                               f"{eff[0]['bonferroni']:.3f}."] if eff else []))
    if half_normal and eff:
        sec.image("Half-Normal Plot", plots.half_normal_plot(eff, f"Half-Normal Plot: "
                                                                  f"{a.fit.response_name}"))
    if pareto and eff:
        sec.image("Pareto Chart", plots.pareto_chart(eff, f"Pareto Chart: "
                                                          f"{a.fit.response_name}"))
    return sec


def split_effects_section(a: Analysis, half_normal: bool = True, pareto: bool = True
                          ) -> Section:
    """Split-plot factorial: subplot and whole-plot effects in separate lists and half-normal
    plots, each with its own error line (Stat-Ease's split-plot tutorial)."""
    eff = model.split_plot_effects(a.space, a.obs, transform=a.fit.transform)
    sec = Section(f"Effects: {a.fit.response_name}")
    sec.log(f"DOE SPLIT-PLOT EFFECTS RESPONSE={a.spec.response}")
    for key, title in SPLIT_EFFECTS:
        sec.table(effects_table(eff[key], title))
        if half_normal and eff[key]:
            sec.image(f"Half-Normal Plot ({title})", plots.half_normal_plot(
                eff[key], f"Half-Normal Plot, {title}: {a.fit.response_name}"))
        if pareto and eff[key]:
            sec.image(f"Pareto Chart ({title})", plots.pareto_chart(
                eff[key], f"Pareto Chart, {title}: {a.fit.response_name}"))
    sec.note("", SPLIT_EFFECTS_NOTE)
    return sec


SPLIT_EFFECTS_NOTE = ("Whole-plot effects (contrasts constant within every group) are estimated "
                      "against the between-group variation and subplot effects against the "
                      "within-group variation, so each set has its own half-normal plot and "
                      "error line; test the chosen model on the ANOVA (REML).")


# ---------------------------------------------------------------------------- diagnostics
DIAGNOSTIC_PLOTS = [("normal", "Normal plot of residuals"),
                    ("resid_pred", "Residuals vs. Predicted"),
                    ("resid_run", "Residuals vs. Run"),
                    ("pred_actual", "Predicted vs. Actual"),
                    ("boxcox", "Box-Cox plot for power transforms"),
                    ("resid_factor", "Residuals vs. Factor"),
                    ("cooks", "Cook's distance"), ("leverage", "Leverage"),
                    ("dffits", "DFFITS"), ("dfbetas", "DFBETAS"),
                    ("covratio", "Covariance ratio")]


RESIDUAL_TYPES = [("external", "Externally Studentized"), ("internal", "Internally Studentized"),
                  ("raw", "Raw (actual - predicted)")]


def color_groups(a: Analysis, by: str | None) -> tuple[np.ndarray | None, dict | None, str]:
    """Colour-by values for the diagnostics plots: None, "block", "group" or "factor:<j>"."""
    if not by or by == "none":
        return None, None, ""
    obs, sp = a.obs, a.space
    if by == "block" and obs.blocks is not None:
        return obs.blocks, {b: f"Block {b:g}" for b in np.unique(obs.blocks)}, "Block"
    if by == "group" and obs.groups is not None:
        return obs.groups, {g: f"Group {g:g}" for g in np.unique(obs.groups)}, "Group"
    if by.startswith("factor:"):
        j = int(by.split(":")[1])
        if 0 <= j < sp.k:
            f = sp.factors[j]
            vals = np.round(obs.actual[:, j], 10)
            if f.categoric:
                lv = np.asarray(f.levels, float)
                labels = {v: f.level_label(int(np.argmin(np.abs(lv - v))))
                          for v in np.unique(vals)}
            else:
                labels = {v: f"{f.letter} = {num(v)}" for v in np.unique(vals)}
            if len(labels) <= len(plots.SERIES):
                return vals, labels, f.title
    return None, None, ""


def diagnostics_section(a: Analysis, plots_wanted=("normal", "resid_pred", "resid_run",
                                                   "pred_actual", "boxcox", "cooks",
                                                   "leverage", "dffits"),
                        table: bool = True, factor_index: int = 0,
                        term_index: int = 1, residual: str = "external",
                        color_by: str | None = None) -> Section:
    fit = a.fit
    d = fit.diagnostics()
    label = fit.transform.label(fit.response_name)
    glm_fit = is_glm(fit)
    if glm_fit:
        label = fit.response_name
    sec = Section(f"Diagnostics: {fit.response_name}")
    sec.log(f"DOE DIAGNOSTICS RESPONSE={a.spec.response}")
    n, p = fit.n, fit.p
    alpha = model.PREFERENCES.alpha
    bonf = stats.t.ppf(1 - alpha / (2 * n), max(fit.df_res - 1, 1))
    if glm_fit:
        bonf = stats.norm.ppf(1 - alpha / (2 * n))
    if table:
        rows = []
        order = np.argsort(d["run"], kind="stable")
        for i in order:
            flag = "¹" if abs(d["ext_stud"][i]) > bonf else ""
            rows.append([f"{d['run'][i]:g}", num(d["actual"][i]), num(d["predicted"][i]),
                         num(d["residual"][i]), fixed(d["leverage"][i], 3),
                         fixed(d["int_stud"][i], 3), fixed(d["ext_stud"][i], 3) + flag,
                         fixed(d["cooks"][i], 3), fixed(d["dffits"][i], 3),
                         fixed(d["covratio"][i], 3) if "covratio" in d else "",
                         f"{d['std'][i]:g}"])
        foot = [f"¹ Outside the outlier limit ±{bonf:.3f} (Bonferroni, alpha {alpha:g})."]
        if label != fit.response_name:
            foot.append(f"Values on the transformed scale {label}.")
        if glm_fit:
            foot.append("Logistic / Poisson fit: internally studentized = standardized Pearson "
                        "residuals, externally studentized = likelihood residuals; Cook's "
                        "distance and DFBETAS by the one-step approximation.")
        else:
            dw = fit.durbin_watson() if hasattr(fit, "durbin_watson") else float("nan")
            if math.isfinite(dw):
                foot.append(f"Durbin-Watson statistic (residuals in run order) = {dw:.3f}; "
                            "values near 2 indicate no serial correlation.")
        sec.table(Table("Diagnostics Case Statistics",
                        ["Run Order", "Actual Value", "Predicted Value", "Residual", "Leverage",
                         "Internally Studentized Residuals",
                         "Externally Studentized Residuals", "Cook's Distance",
                         "Influence on Fitted Value DFFITS", "Covariance Ratio",
                         "Standard Order"], rows, footnotes=foot))
    res, res_label = model.residuals_of_type(d, residual)
    limit = bonf if residual == "external" else (
        stats.norm.ppf(1 - alpha / (2 * n)) * (1 if residual == "internal" else
                                               math.sqrt(max(fit.ms_res, 0)))
        if not glm_fit else None)
    if residual == "raw" and glm_fit:
        limit = None
    colors, clabels, _ctitle = color_groups(a, color_by)
    nm = fit.response_name
    for key in plots_wanted:
        if key == "normal":
            sec.image("Normal Plot of Residuals", plots.normal_plot(
                res, f"Normal Plot of Residuals: {nm}", res_label, colors, clabels))
        elif key == "resid_pred":
            sec.image("Residuals vs. Predicted", plots.residual_plot(
                d["predicted"], res, limit, f"Residuals vs. Predicted: {nm}",
                "Predicted", res_label, colors=colors, color_labels=clabels))
        elif key == "resid_run":
            sec.image("Residuals vs. Run", plots.residual_plot(
                d["run"], res, limit, f"Residuals vs. Run: {nm}", "Run Number", res_label,
                connect=True, colors=colors, color_labels=clabels))
        elif key == "resid_factor":
            j = min(factor_index, fit.space.k - 1)
            f = fit.space.factors[j]
            sec.image(f"Residuals vs. {f.name}", plots.residual_plot(
                a.obs.actual[:, j], res, limit, f"Residuals vs. {f.name}: {nm}",
                f.axis_label, res_label, colors=colors, color_labels=clabels))
        elif key == "pred_actual":
            sec.image("Predicted vs. Actual", plots.predicted_vs_actual(
                d["actual"], d["predicted"], label, colors, clabels))
        elif key == "boxcox":
            if glm_fit:
                sec.note("Box-Cox", "The Box-Cox plot is for least-squares fits; a logistic "
                                    "or Poisson model has no response transformation.",
                         style="warning")
                continue
            try:
                kk = fit.transform.k
                if np.any(a.obs.y + kk <= 0):
                    kk = float(-np.min(a.obs.y) + 0.01 * (np.ptp(a.obs.y) or 1))
                bc = model.box_cox(fit.space, a.obs, fit.terms, kk,
                                   current=fit.transform.power if fit.transform.power
                                   is not None else 1.0)
                sec.image("Box-Cox Plot", plots.box_cox_plot(bc, nm))
                sec.table(Table("Box-Cox Plot for Power Transforms",
                                ["", "Value"],
                                [["Current Lambda", num(bc["current"])],
                                 ["Best Lambda", num(bc["best"])],
                                 ["Low C.I.", num(bc["low"]) if bc["low"] is not None
                                  else "< -3"],
                                 ["High C.I.", num(bc["high"]) if bc["high"] is not None
                                  else "> 3"],
                                 ["Recommended Transform", f"{bc['recommended_name']} "
                                                           f"(Lambda = {bc['recommended']:g})"],
                                 ["Constant k", num(bc["k"]) if bc["k"] else "0"]]))
            except ValueError as exc:
                sec.note("Box-Cox", str(exc), style="warning")
        elif key == "cooks":
            lim = float(stats.f.ppf(0.5, p, max(fit.df_res, 1)))
            sec.image("Cook's Distance", plots.index_plot(
                d["run"], d["cooks"], f"Cook's Distance: {nm}", "Cook's Distance", [lim],
                colors, clabels))
        elif key == "leverage":
            sec.image("Leverage", plots.index_plot(
                d["run"], d["leverage"], f"Leverage vs. Run: {nm}", "Leverage",
                [min(2 * p / n, 1.0)], colors, clabels))
        elif key == "dffits":
            lim = 2 * math.sqrt(p / n)
            sec.image("DFFITS", plots.index_plot(d["run"], d["dffits"], f"DFFITS vs. Run: {nm}",
                                                 "DFFITS", [-lim, lim], colors, clabels))
        elif key == "dfbetas":
            j = min(max(term_index, 0), d["dfbetas"].shape[1] - 1)
            lim = 3 / math.sqrt(n)
            term = d["labels"][j]
            sec.image("DFBETAS", plots.index_plot(
                d["run"], d["dfbetas"][:, j], f"DFBETAS for {term} vs. Run: {nm}",
                f"DFBETAS for {term}", [-lim, lim], colors, clabels))
        elif key == "covratio":
            lims = [1 - 3 * p / n, 1 + 3 * p / n]  # Belsley, Kuh & Welsch
            sec.image("Covariance Ratio", plots.index_plot(
                d["run"], d["covratio"], f"Covariance Ratio vs. Run: {nm}",
                "Covariance Ratio", [v for v in lims if v > 0], colors, clabels))
    return sec


# ---------------------------------------------------------------------------- treatment means
def treatment_tables(fit, level: float = 0.95) -> list[Table]:
    """Treatment means and pairwise comparisons (LSD and Tukey) of each categoric factor in
    the model (least-squares fits)."""
    if is_glm(fit):
        return []
    out = []
    for j in model.categoric_in_model(fit):
        try:
            tm = model.treatment_means(fit, j, level)
        except (ValueError, np.linalg.LinAlgError):
            continue
        f = tm["factor"]
        pct = f"{100 * tm['level']:g}%"
        out.append(Table(f"Treatment Means: {f.title}", ["Level", "Mean", "Standard Error",
                                                          f"{pct} CI Low", f"{pct} CI High"],
                         [[m["level"], num(m["mean"]), num(m["se"]), num(m["low"]),
                           num(m["high"])] for m in tm["means"]],
                         footnotes=["Model predictions on the analysed scale; numeric factors "
                                    "at their centre, other categoric factors averaged over "
                                    "their levels."]))
        out.append(Table(f"Pairwise Comparisons: {f.title}",
                         ["Treatments", "Mean Difference", "Standard Error", "t-value",
                          "LSD p-value", "Tukey p-value", f"LSD ({pct})"],
                         [[f"{c['a']} vs {c['b']}", num(c["diff"]), num(c["se"]),
                           fval(c["t"]), pval(c["p_lsd"]), pval(c["p_tukey"]), num(c["lsd"])]
                          for c in tm["pairs"]],
                         footnotes=[f"Error df = {tm['df']}. LSD: unadjusted t tests (Fisher's "
                                    "least significant difference); Tukey: honestly "
                                    "significant difference adjusted for all pairs."]))
    return out


# ---------------------------------------------------------------------------- analysis summary
def analysis_summary_table(ds, responses: list[str] | None = None) -> Table:
    """Design-Expert's Analysis Summary: one row per analysis of every response with its
    transform, model, the model, lack-of-fit and curvature p-values and fit statistics."""
    rows = []
    meta = ds.meta.get("doe") or {}
    titles = {r["column"]: f"R{i + 1}: {r.get('name') or r['column']}"
              for i, r in enumerate(meta.get("responses", []))}
    for e in model.all_analyses(ds, responses):
        title = titles.get(e["response"], e["response"])
        name = e["name"] + (" *" if e["active"] else "")
        a = e["analysis"]
        if a is None:
            rows.append([title, name, "", "", "", "", "", "", "", "", e["error"][:80]])
            continue
        fit = a.fit
        mdl = model_label(a.space, fit.terms)
        if is_glm(fit):
            chi, p_m = fit.model_test()
            st = fit.fit_statistics()
            rows.append([title, name, fit.family_title, mdl, pval(p_m), "", "",
                         fixed(st["pseudo_r2"]), "", "", "Pseudo R²"])
            continue
        _f, p_m = fit.model_test()
        an = fit.anova() if not is_reml(fit) else {}
        lof = an.get("lof") if isinstance(an, dict) else None
        p_lof = ""
        if lof:
            ss_l, df_l, ss_pe, df_pe = lof
            if ss_pe > 0:
                p_lof = pval(float(stats.f.sf((ss_l / df_l) / (ss_pe / df_pe), df_l, df_pe)))
        curv = an.get("curvature") if isinstance(an, dict) else None
        st = fit.fit_statistics()
        rows.append([title, name, fit.transform.describe(), mdl, pval(p_m), p_lof,
                     pval(curv["p"]) if curv else "", fixed(st["r2"]), fixed(st["adj_r2"]),
                     fixed(st["pred_r2"]), "REML" if is_reml(fit) else ""])
    return Table("Analysis Summary", ["Response", "Analysis", "Transform / Distribution",
                                      "Model", "Model p-value", "Lack of Fit p-value",
                                      "Curvature p-value", "R²", "Adjusted R²",
                                      "Predicted R²", "Note"], rows,
                 footnotes=["* = the active analysis of the response (used by optimization "
                            "and post analysis).",
                            "Look for a significant model, insignificant lack of fit and, "
                            "for factorials with centre points, insignificant curvature."])


def analysis_summary_section(ds) -> Section:
    sec = Section("Analysis Summary")
    sec.log("DOE ANALYSIS SUMMARY")
    sec.table(analysis_summary_table(ds))
    return sec


# ---------------------------------------------------------------------------- effects plots
def effects_normal_figure(eff: list[dict], title: str, selected: set | None = None):
    sw = model.effects_shapiro(eff, selected if selected is not None else
                               {e["name"] for e in eff
                                if e.get("t", 0) > e.get("t_limit", np.inf)})
    return plots.normal_effects_figure(eff, title, selected, sw)


# ---------------------------------------------------------------------------- model graphs
GRAPHS = [("contour", "Contour (2D)"), ("surface", "3D Surface"),
          ("perturbation", "Perturbation"), ("interaction", "Interaction"),
          ("one_factor", "One Factor"), ("cube", "Cube"),
          ("ternary", "Mixture: Contour (ternary)"), ("surface3", "Mixture: 3D Surface"),
          ("trace", "Mixture: Trace (Cox)"), ("se", "Standard Error (contour)"),
          ("se_ternary", "Mixture: Standard Error (ternary)"), ("all_factors", "All Factors")]


def default_point(space: Space) -> np.ndarray:
    """Reference point in actual units: centre of numeric ranges, first categoric level,
    centroid of the mixture region."""
    x = np.empty(space.k)
    for j, f in enumerate(space.factors):
        x[j] = f.levels[0] if f.categoric else f.center
    if space.mixture:
        x = space.lower + space.pseudo_span / space.k
    elif space.combined and space.mix_idx:
        mix = space.mix_idx
        x[mix] = space.lower[mix] + space.pseudo_span / len(mix)
    return x


def _slice_points(fit: Fit, axes: list[int], ref: np.ndarray, tol: float = 1e-6):
    """Design points lying on the displayed slice (other factors at their reference values)."""
    sp = fit.space
    coded_ref = sp.to_coded(ref[None, :])[0]
    keep = np.ones(fit.n, bool)
    for j in range(sp.k):
        if j in axes:
            continue
        if sp.mixture:
            continue
        keep &= np.abs(fit.obs.coded[:, j] - coded_ref[j]) < 1e-3 + tol
    return keep


@dataclass
class GraphFigure:
    """A model graph as a live matplotlib figure: the Model Graphs view shows it on an
    interactive canvas (3D views rotate, contour plots take flags); reports save a PNG."""
    figure: object
    title: str
    rotatable: bool = False
    probe: object = None          # (x, y) in data coordinates -> list of text lines | None
    notes: list = field(default_factory=list)


FIGURE_GRAPHS = ("contour", "surface", "ternary", "surface3", "se", "se_ternary")


def se_of_prediction(fit, X: np.ndarray) -> np.ndarray:
    """Standard error of the predicted mean on the analysed scale, sqrt(x'(X'X)^-1 x s²)."""
    _z, v = fit.predict_coded(fit.space.to_coded(X))
    return np.sqrt(np.clip(v, 0, None) * fit.ms_res)


def _flag_lines(fit: Fit, X: np.ndarray, factors: list[int], nm: str) -> list[str]:
    sp = fit.space
    iv = fit.interval(X[None, :])
    lines = [f"{sp.factors[j].letter}: {sp.factors[j].name} = {_fmt_level(sp.factors[j], X[j])}"
             for j in factors]
    lines.append(f"{nm}: {num(iv['median'][0])}")
    lines.append(f"95% CI: {num(iv['ci'][0][0])} to {num(iv['ci'][1][0])}")
    return lines


def graph_figure(a: Analysis, graph: str, axes: list[int], ref: np.ndarray,
                 points: bool = True, n_grid: int = 50, n_levels: int = 10,
                 full_range: bool = False, levels=None, cmap: str | None = None,
                 grid: bool = False) -> GraphFigure:
    """Contour, 3D surface, mixture ternary contour or mixture 3D surface as a figure;
    ``se`` / ``se_ternary`` draw the standard error of the predicted mean instead of the
    prediction. ``levels``: custom contour values; ``cmap``: colour map name."""
    fit = a.fit
    sp = fit.space
    nm = fit.response_name
    zlabel = nm + (f" ({_units(a)})" if _units(a) else "")
    notes: list[str] = []
    value = fit.predict
    if graph in ("se", "se_ternary"):
        value = lambda X: se_of_prediction(fit, X)  # noqa: E731
        nm = f"Std Error of Design: {fit.response_name}"
        zlabel = "Std Error of Mean"
        notes.append("Standard error of the predicted mean (on the analysed scale): low "
                     "where the design has information, high at the edges.")
        graph = "contour" if graph == "se" else "ternary"
        points = points and True

    def rng(j):
        f = sp.factors[j]
        lo, hi = f.low, f.high
        if full_range:
            lo = min(lo, float(fit.obs.actual[:, j].min()))
            hi = max(hi, float(fit.obs.actual[:, j].max()))
        return lo, hi

    if graph in ("contour", "surface"):
        if sp.mixture:
            raise ValueError("Use the ternary contour for mixture components.")
        i, j = axes[:2]
        if sp.factors[i].categoric or sp.factors[j].categoric:
            raise ValueError("Contour and 3D plots need numeric factors on both axes; choose "
                             "the categoric level in the Factors Tool.")
        if sp.combined and (sp.is_mix(i) or sp.is_mix(j)):
            raise ValueError("Put process factors on the contour axes; the mixture blend is "
                             "set in the Factors Tool (or use the ternary plot for the "
                             "components).")
        (xl, xh), (yl, yh) = rng(i), rng(j)
        gx, gy = np.meshgrid(np.linspace(xl, xh, n_grid), np.linspace(yl, yh, n_grid))
        X = np.tile(ref, (gx.size, 1))
        X[:, i], X[:, j] = gx.ravel(), gy.ravel()
        Z = value(X).reshape(gx.shape)
        keep = _slice_points(fit, [i, j], ref) if points else np.zeros(fit.n, bool)
        tol_x, tol_y = 1e-6 * (xh - xl), 1e-6 * (yh - yl)
        keep &= (fit.obs.actual[:, i] >= xl - tol_x) & (fit.obs.actual[:, i] <= xh + tol_x)
        keep &= (fit.obs.actual[:, j] >= yl - tol_y) & (fit.obs.actual[:, j] <= yh + tol_y)
        pts = fit.obs.actual[keep][:, [i, j]]
        if graph == "contour":
            fig = plots.contour_figure(gx, gy, Z, sp.factors[i].axis_label,
                                       sp.factors[j].axis_label, nm, points=pts,
                                       n_levels=n_levels, zlabel=zlabel, levels=levels,
                                       cmap=cmap, grid=grid)

            def probe(x, y):
                if not (xl <= x <= xh and yl <= y <= yh):
                    return None
                Xp = ref.astype(float).copy()
                Xp[i], Xp[j] = x, y
                return _flag_lines(fit, Xp, [i, j], nm)

            return GraphFigure(fig, "Contour Plot", False, probe, notes)
        p3 = np.column_stack([pts, fit.obs.y[keep]]) if len(pts) else None
        fig = plots.surface_figure(gx, gy, Z, sp.factors[i].axis_label,
                                   sp.factors[j].axis_label, zlabel, nm, points=p3, cmap=cmap)
        return GraphFigure(fig, "3D Surface", True, None, notes)
    if graph in ("ternary", "surface3"):
        if not (sp.mixture or sp.combined):
            raise ValueError("The ternary graphs are for mixture designs.")
        tri = [j for j in axes if sp.is_mix(j)][:3]
        if len(tri) < 3:
            raise ValueError("Choose three components for the ternary plot.")
        rest = [j for j in range(sp.k) if j not in tri]
        rest_mix = [j for j in rest if sp.is_mix(j)]
        lows = sp.lower[tri]
        free = sp.total - ref[rest_mix].sum() - lows.sum()
        if free <= 0:
            raise ValueError("The fixed components leave no room for the plotted ones.")

        def to_full(act3):
            X = np.tile(ref, (len(act3), 1))
            X[:, tri] = act3
            return X

        def fn(act3):
            return value(to_full(act3))

        def feas(act3):
            X = to_full(act3)
            return np.all((X >= sp.lower - 1e-9) & (X <= sp.upper + 1e-9), axis=1)

        keep = np.ones(fit.n, bool)
        for j in rest:
            keep &= np.abs(fit.obs.actual[:, j] - ref[j]) < 1e-6 * max(1, abs(ref[j]))
        names = [sp.factors[j].title for j in tri]
        if sp.lower.sum() > 0:
            notes.append("The triangle shows the L_Pseudo simplex of the three components; "
                         "infeasible areas (upper limits) are left blank.")
        if graph == "surface3":
            pts = np.column_stack([fit.obs.actual[keep][:, tri], fit.obs.y[keep]]) \
                if points and keep.any() else None
            fig = plots.ternary_surface_figure(fn, lows, free, names, nm, zlabel, feas, pts,
                                               cmap=cmap)
            return GraphFigure(fig, "Mixture 3D Surface", True, None, notes)
        pts = fit.obs.actual[keep][:, tri] if points else None
        vv = [f"{lows[m] + free:.4g}" for m in range(3)]
        fig = plots.ternary_figure(fn, lows, free, names, nm, zlabel, feas, pts,
                                   n_levels=n_levels, vertex_values=vv, levels=levels,
                                   cmap=cmap)

        def probe(x, y):
            prop = plots.ternary_point(x, y)
            if prop is None:
                return None
            act = to_full((lows + prop * free)[None, :])[0]
            if not feas(act[tri][None, :])[0]:
                return None
            return _flag_lines(fit, act, list(range(sp.k)), nm)

        return GraphFigure(fig, "Mixture Contour", False, probe, notes)
    raise ValueError(f"{graph!r} is not a figure graph.")


def graph_section(a: Analysis, graph: str, axes: list[int], ref: np.ndarray,
                  points: bool = True, n_grid: int = 50, n_levels: int = 10,
                  full_range: bool = False, levels=None, cmap: str | None = None,
                  pi_bands: bool = False, lsd: bool = False) -> Section:
    """One model graph as a report section. ``pi_bands``: prediction-interval bands on the
    one-factor / interaction graphs; ``lsd``: least-significant-difference bars at the
    levels of a categoric factor (instead of confidence bars)."""
    fit = a.fit
    sp = fit.space
    nm = fit.response_name
    zlabel = nm + (f" ({_units(a)})" if _units(a) else "")
    sec = Section(f"Model Graphs: {nm}")
    sec.log(f"DOE GRAPH RESPONSE={a.spec.response} TYPE={graph.upper()} AXES="
            + ",".join(sp.factors[j].letter for j in axes))
    others = [f"{f.title} = {_fmt_level(f, ref[j])}" for j, f in enumerate(sp.factors)
              if j not in axes]

    def rng(j):
        f = sp.factors[j]
        lo, hi = f.low, f.high
        if full_range:
            lo = min(lo, float(fit.obs.actual[:, j].min()))
            hi = max(hi, float(fit.obs.actual[:, j].max()))
        return lo, hi

    if graph in FIGURE_GRAPHS:
        gf = graph_figure(a, graph, axes, ref, points, n_grid, n_levels, full_range, levels,
                          cmap)
        sec.image(gf.title, plots.figure_png(gf.figure))
        for text in gf.notes:
            sec.note("", text)
    elif graph == "all_factors":
        panels = []
        for j, f in enumerate(sp.factors):
            if sp.mixture or (sp.combined and sp.is_mix(j)):
                continue
            if f.categoric:
                X = np.tile(ref, (len(f.levels), 1))
                X[:, j] = f.levels
                iv = fit.interval(X)
                panels.append({"x": np.arange(len(f.levels), dtype=float), "y": iv["median"],
                               "lo": iv["ci"][0], "hi": iv["ci"][1], "xlabel": f.title,
                               "ticks": [f.level_label(i) for i in range(len(f.levels))]})
            else:
                lo, hi = rng(j)
                xv = np.linspace(lo, hi, 41)
                X = np.tile(ref, (len(xv), 1))
                X[:, j] = xv
                iv = fit.interval(X)
                panels.append({"x": xv, "y": iv["median"], "lo": iv["ci"][0],
                               "hi": iv["ci"][1], "xlabel": f.axis_label})
        if not panels:
            raise ValueError("All Factors shows process factors; use the trace plot for "
                             "mixture components.")
        sec.image("All Factors", plots.all_factors_plot(panels, zlabel, f"All Factors: {nm}"))
        others = []
    elif graph == "perturbation":
        curves = []
        for j, f in enumerate(sp.factors):
            if f.categoric or (sp.combined and sp.is_mix(j)):
                continue
            dev = np.linspace(-1, 1, 41)
            X = np.tile(ref, (len(dev), 1))
            if sp.mixture:
                X = _cox_path(sp, ref, j, dev)
            else:
                X[:, j] = f.center + dev * f.half
            curves.append((f.letter, dev, fit.predict(X)))
        ref_txt = "Reference: " + ", ".join(f"{f.letter} = {_fmt_level(f, ref[j])}"
                                            for j, f in enumerate(sp.factors))
        sec.image("Perturbation", plots.perturbation_plot(curves, zlabel,
                                                          f"Perturbation: {nm}", ref_txt))
    elif graph == "trace":
        if not sp.mixture:
            raise ValueError("The trace plot is for mixture designs.")
        curves = []
        for j, f in enumerate(sp.factors):
            dev = np.linspace(-1, 1, 81)
            X = _cox_path(sp, ref, j, dev)
            ok = np.all((X >= sp.lower - 1e-9) & (X <= sp.upper + 1e-9), axis=1)
            curves.append((f.letter, dev[ok], fit.predict(X[ok])))
        sec.image("Trace (Cox)", plots.perturbation_plot(
            curves, zlabel, f"Trace (Cox): {nm}",
            "Reference blend: " + ", ".join(f"{f.letter} = {ref[j]:.4g}"
                                            for j, f in enumerate(sp.factors))))
    elif graph in ("one_factor", "interaction"):
        i = axes[0]
        f = sp.factors[i]
        pts_all = _slice_points(fit, [i] + (axes[1:2] if graph == "interaction" else []), ref)
        if f.categoric:
            xs = np.arange(len(f.levels), dtype=float)
            xvals = np.asarray(f.levels, float)
        else:
            lo, hi = rng(i)
            xvals = np.linspace(lo, hi, 61)
            xs = xvals
        curves, pts = [], []
        bands = []
        lsd_half = []
        level = model.PREFERENCES.ci_level
        if graph == "one_factor" or len(axes) < 2:
            X = np.tile(ref, (len(xvals), 1))
            X[:, i] = xvals
            iv = fit.interval(X, level)
            curves.append((nm, iv["median"], iv["ci"][0], iv["ci"][1]))
            bands.append(iv["pi"] if pi_bands else None)
            lsd_half.append(_lsd_half(fit, X, level) if lsd and f.categoric else None)
            if points:
                pts.append(np.column_stack([_xpos(f, fit.obs.actual[pts_all, i]),
                                            fit.obs.y[pts_all]]))
            title = f"One Factor: {nm}"
        else:
            j = axes[1]
            g = sp.factors[j]
            lvls = (list(range(len(g.levels))) if g.categoric else [g.low, g.high])
            for lv in lvls:
                X = np.tile(ref, (len(xvals), 1))
                X[:, i] = xvals
                X[:, j] = g.levels[lv] if g.categoric else lv
                iv = fit.interval(X, level)
                lab = f"{g.letter}{'-' if lv == lvls[0] else '+'} {_fmt_level(g, X[0, j])}" \
                    if not g.categoric else f"{g.letter}: {g.level_label(lv)}"
                curves.append((lab, iv["median"], iv["ci"][0], iv["ci"][1]))
                bands.append(iv["pi"] if pi_bands else None)
                lsd_half.append(_lsd_half(fit, X, level) if lsd and f.categoric else None)
                if points:
                    sel = pts_all & (np.abs(fit.obs.actual[:, j] - X[0, j]) < 1e-6 *
                                     max(1, abs(X[0, j])))
                    pts.append(np.column_stack([_xpos(f, fit.obs.actual[sel, i]),
                                                fit.obs.y[sel]]))
            title = f"Interaction: {nm}"
        if f.categoric:
            png = _categoric_line(xs, curves, f, zlabel, title, pts if points else None,
                                  lsd_half if lsd else None)
        else:
            png = plots.line_plot(xs, curves, f.axis_label, zlabel, title,
                                  points=pts if points else None,
                                  legend_title=sp.factors[axes[1]].title if len(axes) > 1
                                  and graph == "interaction" else "",
                                  pi_bands=bands if pi_bands else None)
        sec.image(title, png)
        if pi_bands and not f.categoric:
            sec.note("", f"Dashed: {100 * level:g}% confidence band of the mean; dotted / "
                         f"shaded: {100 * level:g}% prediction band of a single run.")
        if lsd and f.categoric:
            sec.note("", f"Bars: {100 * level:g}% least significant difference (LSD) "
                         "intervals; non-overlapping bars indicate different treatment means.")
    elif graph == "cube":
        ax3 = [j for j in axes if not sp.factors[j].categoric][:3]
        if len(ax3) < 3 or sp.mixture:
            raise ValueError("The cube plot needs three numeric factors.")
        vals = np.empty(8)
        for idx in range(8):
            X = ref.copy()
            for b, j in enumerate(ax3):
                f = sp.factors[j]
                X[j] = f.high if idx >> b & 1 else f.low
            vals[idx] = fit.predict(X[None, :])[0]
        sec.image("Cube", plots.cube_plot(
            vals, [sp.factors[j].axis_label for j in ax3],
            [f"{sp.factors[j].letter}-: {num(sp.factors[j].low)}" for j in ax3],
            [f"{sp.factors[j].letter}+: {num(sp.factors[j].high)}" for j in ax3],
            f"Cube: {nm}"))
    else:
        raise ValueError(f"Unknown graph {graph!r}.")
    if others:
        sec.note("Factor Settings", "Other factors: " + "; ".join(others))
    return sec


def _xpos(f, values):
    if f.categoric:
        lv = np.asarray(f.levels, float)
        return np.array([int(np.argmin(np.abs(lv - v))) for v in values], float)
    return values


def _lsd_half(fit, X: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """LSD bars at each categoric level: the predicted mean -/+ t * SE(mean) * sqrt(2) / 2 on
    the analysed scale (back-transformed), so two bars that do not overlap differ at the
    given level (Design-Expert's LSD bars)."""
    z, v = fit.predict_coded(fit.space.to_coded(X))
    se = np.sqrt(np.clip(v, 0, None) * fit.ms_res)
    df = max(getattr(fit, "df_res", 1), 1)
    q = stats.norm.ppf(0.5 + level / 2) if is_glm(fit) else stats.t.ppf(0.5 + level / 2, df)
    h = q * se * math.sqrt(2) / 2
    lo, hi = fit.transform.inverse(z - h), fit.transform.inverse(z + h)
    return np.minimum(lo, hi), np.maximum(lo, hi)


def _categoric_line(xs, curves, f, zlabel, title, pts, lsd=None):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import io
    fig = Figure(figsize=(6, 4.5), dpi=120)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    for i, (name, y, lo, hi) in enumerate(curves):
        color = plots.SERIES[i % len(plots.SERIES)]
        off = (i - (len(curves) - 1) / 2) * 0.06
        if lsd is not None and i < len(lsd) and lsd[i] is not None:
            lo_l, hi_l = lsd[i]
            yerr = [np.clip(y - lo_l, 0, None), np.clip(hi_l - y, 0, None)]
            ax.errorbar(xs + off, y, yerr=yerr, fmt="-s", color=color, capsize=6,
                        label=name, elinewidth=2.2)
        else:
            ax.errorbar(xs + off, y, yerr=[y - lo, hi - y], fmt="-o", color=color, capsize=4,
                        label=name)
        if pts and i < len(pts) and len(pts[i]):
            ax.scatter(pts[i][:, 0] + off, pts[i][:, 1], s=16, color=color, alpha=0.6)
    ax.set_xticks(xs, [f.level_label(int(i)) for i in xs])
    if len(curves) > 1:
        ax.legend(fontsize=8, frameon=False)
    plots._style(ax, title, f.axis_label, zlabel)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    return buf.getvalue()


def _cox_path(sp: Space, ref: np.ndarray, j: int, dev: np.ndarray) -> np.ndarray:
    """Cox direction through the reference blend (in pseudo components), dev in [-1, 1]."""
    r = sp.to_coded(ref[None, :])[0]
    X = []
    for d in dev:
        lo, hi = -r[j], 1 - r[j]
        delta = d * (hi if d > 0 else -lo)
        x = r.copy()
        x[j] = r[j] + delta
        others = [i for i in range(sp.k) if i != j]
        rest = 1 - r[j]
        for i in others:
            x[i] = r[i] - delta * (r[i] / rest if rest > 0 else 1 / len(others))
        X.append(x)
    return sp.to_actual(np.array(X))


def _fmt_level(f, v) -> str:
    if f.categoric:
        lv = np.asarray(f.levels, float)
        return f.level_label(int(np.argmin(np.abs(lv - v))))
    return num(v) + (f" {f.units}" if f.units else "")


def _units(a: Analysis) -> str:
    return getattr(a.fit, "response_units", "")


# ---------------------------------------------------------------------------- optimisation
def optimization_section(problem: optimize.Problem, solutions: list[dict], names: dict,
                         axes: list[int] | None = None, ramps: bool = True,
                         contour: bool = True, overlay: bool = True, bars: bool = True,
                         seed=None, starts: int = 0) -> Section:
    sp = problem.space
    sec = Section("Optimization")
    sec.log(f"DOE OPTIMIZE RESPONSES={','.join(problem.fits)} STARTS={starts}"
            + (f" SEED={seed}" if seed is not None else ""))
    rows = []
    goal_txt = dict(optimize.GOALS)
    for c in problem.criteria:
        if c.kind == "factor":
            j = next(i for i, f in enumerate(sp.factors) if f.column == c.name)
            nm = sp.factors[j].title
        else:
            nm = names.get(c.name, c.name)
        rows.append([nm, goal_txt.get(c.goal, c.goal), num(c.lower), num(c.upper),
                     num(c.target) if c.goal == "target" else "", f"{c.w_lower:g}",
                     f"{c.w_upper:g}", str(c.importance) if c.active else ""])
    sec.table(Table("Constraints", ["Name", "Goal", "Lower Limit", "Upper Limit", "Target",
                                    "Lower Weight", "Upper Weight", "Importance"], rows))
    if not solutions:
        sec.note("Solutions", "No solutions found.", style="warning")
        return sec
    resp = [c.name for c in problem.criteria if c.kind == "response" and c.name in problem.fits]
    cols = ["Number"] + [f.name for f in sp.factors] + [names.get(r, r) for r in resp] + \
        ["Desirability", ""]
    srows = []
    for i, s in enumerate(solutions):
        row = [str(i + 1)] + [_fmt_level(f, s["x"][j]).split(" ")[0] if f.categoric
                              else num(s["x"][j]) for j, f in enumerate(sp.factors)]
        row += [num(s["pred"][r]) for r in resp] + [f"{s['D']:.3f}",
                                                     "Selected" if i == 0 else ""]
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
        entries = []
        for c in problem.criteria:
            if c.kind == "factor":
                j = next(i for i, f in enumerate(sp.factors) if f.column == c.name)
                entries.append({"name": sp.factors[j].name, "low": c.lower, "high": c.upper,
                                "value": float(best["x"][j]), "goal": c.goal,
                                "target": c.target, "kind": "factor"})
            elif c.name in problem.fits and c.goal != "none":
                entries.append({"name": names.get(c.name, c.name), "low": c.lower,
                                "high": c.upper, "value": best["pred"][c.name],
                                "goal": c.goal, "target": c.target, "kind": "response"})
        sec.image("Ramps", plots.ramps_plot(entries, best["D"], "Desirability Ramps "
                                                                "(Solution 1)"))
    if bars:
        act = [c for c in problem.criteria if c.active]
        labels = [names.get(c.name, c.name) if c.kind == "response" else c.name for c in act]
        sec.image("Desirability", plots.desirability_bars(labels, best["d"], best["D"]))
    num_axes = [j for j in (axes or []) if not sp.factors[j].categoric]
    if not num_axes:
        num_axes = [j for j, f in enumerate(sp.factors) if not f.categoric][:2]
    if not sp.mixture and len(num_axes) >= 2 and (contour or overlay):
        i, j = num_axes[:2]
        fi, fj = sp.factors[i], sp.factors[j]
        bi, bj = problem.bounds_actual()[i], problem.bounds_actual()[j]
        gx, gy = np.meshgrid(np.linspace(bi[0], bi[1], 60), np.linspace(bj[0], bj[1], 60))
        X = np.tile(best["x"], (gx.size, 1))
        X[:, i], X[:, j] = gx.ravel(), gy.ravel()
        ev = problem.evaluate(X)
        sol = (best["x"][i], best["x"][j])
        if contour:
            sec.image("Desirability Contour", plots.contour_plot(
                gx, gy, ev["D"].reshape(gx.shape), fi.axis_label, fj.axis_label,
                "Desirability", solution=sol, zlabel="Desirability"))
        if overlay:
            limits = []
            for c in problem.criteria:
                if c.kind == "response" and c.name in problem.fits and c.goal != "none":
                    limits.append((names.get(c.name, c.name),
                                   ev["pred"][c.name].reshape(gx.shape), c.lower, c.upper))
            if limits:
                sec.image("Overlay Plot", plots.overlay_plot(
                    gx, gy, limits, fi.axis_label, fj.axis_label, "Overlay Plot",
                    solution=sol))
                sec.note("", "Overlay plot (graphical optimisation): the yellow window meets "
                             "every response's lower and upper limits; grey areas do not.")
    elif sp.mixture and len(num_axes) >= 3 and (contour or overlay):
        tri = num_axes[:3]
        rest = [j for j in range(sp.k) if j not in tri]
        lows = sp.lower[tri]
        free = sp.total - best["x"][rest].sum() - lows.sum()

        def to_full(a3):
            X = np.tile(best["x"], (len(a3), 1))
            X[:, tri] = a3
            return X

        def feas(a3):
            X = to_full(a3)
            return np.all((X >= sp.lower - 1e-9) & (X <= sp.upper + 1e-9), axis=1)

        if contour and free > 0:
            sec.image("Desirability Contour", plots.ternary_contour(
                lambda a3: problem.evaluate(to_full(a3))["D"], lows, free,
                [sp.factors[j].title for j in tri], "Desirability", "Desirability", feas,
                points=best["x"][tri][None, :]))
        if overlay and free > 0:
            sec.image("Overlay Plot", plots.ternary_contour(
                lambda a3: optimize.feasible(problem, to_full(a3)).astype(float), lows, free,
                [sp.factors[j].title for j in tri], "Overlay Plot (1 = all limits met)", "",
                feas, n_levels=1))
    return sec


def prediction_section(fits: dict[str, Fit], space: Space, x: np.ndarray, names: dict,
                       level: float = 0.95) -> Section:
    sec = Section("Point Prediction")
    sec.log("DOE PREDICT " + " ".join(f"{f.letter}={_fmt_level(f, x[j])}"
                                      for j, f in enumerate(space.factors)))
    sec.table(Table("Factor Settings", ["Factor", "Name", "Level", "Coded Level"],
                    [[f.letter, f.name, _fmt_level(f, x[j]),
                      "" if f.categoric else num(space.to_coded(x[None, :])[0, j])]
                     for j, f in enumerate(space.factors)]))
    pct = f"{level * 100:g}%"
    rows = []
    for col, fit in fits.items():
        iv = fit.interval(x[None, :], level)
        rows.append([names.get(col, col), num(iv["mean"][0]), num(iv["median"][0]),
                     num(iv["sd"]), num(iv["se_mean"][0]), num(iv["ci"][0][0]),
                     num(iv["ci"][1][0]), num(iv["se_pred"][0]), num(iv["pi"][0][0]),
                     num(iv["pi"][1][0])])
    sec.table(Table("Prediction", ["Response", "Predicted Mean", "Predicted Median",
                                   "Std Dev", "SE Mean", f"{pct} CI Low", f"{pct} CI High",
                                   "SE Pred", f"{pct} PI Low", f"{pct} PI High"], rows,
                    footnotes=["Std Dev and standard errors are on the analysis (possibly "
                               "transformed) scale; intervals are back-transformed.",
                               "For a transformed response the predicted mean includes a "
                               "second-order bias correction; the median is the simple "
                               "back-transform."]))
    return sec


# ---------------------------------------------------------------------------- Taguchi
def taguchi_section(ds, factor_cols: list[str], response_cols: list[str], kind: str,
                    means_plot: bool = True, anova: bool = True) -> Section:
    space = model.space_from_dataset(ds, factor_cols)
    from rsstatchem.core.stats.common import numeric
    Y = np.column_stack([numeric(ds, c).to_numpy(float) for c in response_cols])
    actual = np.column_stack([numeric(ds, c).to_numpy(float) for c in factor_cols])
    ok = np.all(np.isfinite(actual), axis=1) & np.any(np.isfinite(Y), axis=1)
    Y, actual = Y[ok], actual[ok]
    for f in space.factors:
        if not f.categoric:
            vals = sorted(set(actual[:, f.index].tolist()))
            f.type, f.levels = "categoric", vals
            var = ds.var(f.column)
            f.labels = [var.value_labels.get(v, model._num(v)) for v in vals]
    coded = space.to_coded(actual).astype(int)
    sn = optimize.sn_ratio(Y, kind)
    means = np.nanmean(Y, axis=1)
    sec = Section("Taguchi Analysis")
    sec.log(f"DOE TAGUCHI FACTORS={' '.join(factor_cols)} RESPONSES={' '.join(response_cols)} "
            f"SN={kind.upper()}")
    rows = []
    for i in range(len(Y)):
        rows.append([str(i + 1)] + [space.factors[j].level_label(coded[i, j])
                                    for j in range(space.k)]
                    + [num(v) for v in Y[i]] + [num(means[i]), fixed(sn[i], 3)])
    sec.table(Table("Signal-to-Noise Ratios", ["Run"] + [f.title for f in space.factors]
                    + response_cols + ["Mean", "S/N Ratio"], rows,
                    footnotes=[dict(optimize.SN_TYPES)[kind] + ": " + {
                        "larger": "S/N = -10 log10(mean(1/y²))",
                        "smaller": "S/N = -10 log10(mean(y²))",
                        "nominal": "S/N = 10 log10(ȳ²/s²)",
                        "nominal_var": "S/N = -10 log10(s²)"}[kind]]))
    nl = [len(f.levels) for f in space.factors]
    max_l = max(nl)
    for title, vals, unit in (("Response Table for Signal to Noise Ratios", sn, "S/N"),
                              ("Response Table for Means", means, "Mean")):
        rt = optimize.response_table(coded, vals, nl)
        trows = []
        for lv in range(max_l):
            trows.append([str(lv + 1)] + [fixed(rt["means"][j][lv], 3) if lv < nl[j] else ""
                                          for j in range(space.k)])
        trows.append(["Delta"] + [fixed(d, 3) for d in rt["delta"]])
        trows.append(["Rank"] + [str(r) for r in rt["rank"]])
        sec.table(Table(title, ["Level"] + [f.title for f in space.factors], trows))
        if unit == "S/N" or means_plot:
            sec.image(f"Main Effects Plot for {unit}", width=min(160 * space.k + 120, 900),
                      png=plots.main_effects_plot(
                [f.title for f in space.factors],
                [[f.level_label(i) for i in range(len(f.levels))] for f in space.factors],
                rt["means"], f"Main Effects Plot for {'SN ratios' if unit == 'S/N' else 'Means'}",
                f"Mean of {'SN ratios' if unit == 'S/N' else 'Means'}",
                float(np.nanmean(vals))))
        if unit == "S/N":
            best = [int(np.nanargmax(m)) for m in rt["means"]]
            pred = float(np.nanmean(sn)) + sum(rt["means"][j][best[j]] - np.nanmean(sn)
                                               for j in range(space.k))
            sec.table(Table("Optimal Levels (highest S/N)", ["Factor", "Level", "Mean S/N"],
                            [[f.title, f.level_label(best[j]), fixed(rt["means"][j][best[j]],
                                                                     3)]
                             for j, f in enumerate(space.factors)],
                            footnotes=[f"Predicted S/N at the optimal levels (additive model): "
                                       f"{pred:.3f}"]))
    if anova:
        good = np.isfinite(sn)
        obs = model.Observations(actual[good], coded[good].astype(float), sn[good], None,
                                 np.arange(1, good.sum() + 1.0), np.arange(1, good.sum() + 1.0),
                                 np.flatnonzero(good))
        terms = model.process_terms(space.k, "linear", space.categoric)
        try:
            fit = Fit(space, obs, terms, response="S/N", response_name="S/N Ratio")
            t, _ = anova_table(fit, "main effects (S/N)")
            t.title = "Analysis of Variance for S/N Ratios"
            sec.table(t)
        except ValueError:
            sec.note("Analysis of Variance for S/N Ratios", "Not available: the main-effects "
                     "model leaves no degrees of freedom for error (all columns used).")
    return sec

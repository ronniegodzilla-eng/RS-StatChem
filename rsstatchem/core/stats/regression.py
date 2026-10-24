"""Analyze > Regression > Linear, with Enter / Stepwise / Forward / Backward selection and the
classical assumption tests used in theses (normality of residuals, multicollinearity,
heteroscedasticity: Glejser / Breusch-Pagan / White, autocorrelation: Durbin-Watson /
Breusch-Godfrey / Runs test)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats import charts
from rsstatchem.core.stats.common import fmt, fmt_int, fmt_p, listwise, pct, shapiro, sup
from rsstatchem.core.stats.nonparametric import ks_details, ks_table, runs_details, runs_table

METHODS = ("enter", "stepwise", "forward", "backward")
DEFAULT_STATS = ("estimates", "model_fit", "durbin_watson", "collinearity")
ASSUMPTIONS = ("normality", "multicollinearity", "heteroscedasticity", "autocorrelation")


@dataclass
class RegressionResult:
    section: Section
    saved: dict[str, tuple[str, pd.Series]] = field(default_factory=dict)


@dataclass
class _Step:
    cols: list[str]
    entered: list[str]
    removed: list[str]
    fit: object = None


def _fit(y: pd.Series, X: pd.DataFrame, cols: list[str]):
    import statsmodels.api as sm

    design = sm.add_constant(X[cols], has_constant="add") if cols else \
        pd.DataFrame({"const": np.ones(len(y))}, index=y.index)
    return sm.OLS(y, design).fit()


def _select(y, X, names, method, p_enter, p_remove) -> list[_Step]:
    if method == "enter":
        return [_Step(list(names), list(names), [])]
    steps: list[_Step] = []
    current: list[str] = list(names) if method == "backward" else []
    if method == "backward":
        steps.append(_Step(list(current), list(current), []))
    for _ in range(4 * len(names) + 4):
        changed = False
        if method in ("forward", "stepwise"):
            best, best_p = None, 1.0
            for c in names:
                if c in current:
                    continue
                p = _fit(y, X, current + [c]).pvalues[c]
                if p < best_p:
                    best, best_p = c, p
            if best is not None and best_p <= p_enter:
                current = current + [best]
                steps.append(_Step(list(current), [best], []))
                changed = True
        if method in ("backward", "stepwise") and current:
            pvals = _fit(y, X, current).pvalues.drop("const")
            worst = pvals.idxmax()
            if pvals[worst] >= p_remove:
                current = [c for c in current if c != worst]
                steps.append(_Step(list(current), [], [worst]))
                changed = True
        if not changed:
            break
    return steps


def _vif(X: pd.DataFrame, cols: list[str]) -> dict[str, tuple[float, float]]:
    out = {}
    for c in cols:
        others = [o for o in cols if o != c]
        if not others:
            out[c] = (1.0, 1.0)
            continue
        r2 = _fit(X[c], X, others).rsquared
        tol = 1 - r2
        out[c] = (tol, 1 / tol if tol > 0 else float("inf"))
    return out


def linear(ds: Dataset, dependent: str, independents: list[str], method: str = "enter",
           statistics=DEFAULT_STATS, plots=("histogram", "pp", "scatter"), save=(),
           assumptions=ASSUMPTIONS, level: float = 0.95, p_enter: float = 0.05,
           p_remove: float = 0.10) -> RegressionResult:
    from statsmodels.stats.diagnostic import (acorr_breusch_godfrey, het_breuschpagan,
                                              het_white)
    from statsmodels.stats.stattools import durbin_watson

    dvar = ds.var(dependent)
    titles = {n: ds.var(n).title for n in independents}
    section = Section("Regression")
    stat_kw = {"estimates": "COEFF", "model_fit": "R ANOVA", "ci": f"CI({level * 100:g})",
               "collinearity": "COLLIN TOL", "r2_change": "CHANGE",
               "descriptives": "DESCRIPTIVES"}
    section.log("REGRESSION\n  /MISSING LISTWISE\n  /STATISTICS "
                + " ".join(v for k, v in stat_kw.items() if k in statistics)
                + f"\n  /CRITERIA=PIN({p_enter:.2f}) POUT({p_remove:.2f})"
                + f"\n  /DEPENDENT {dependent}\n  /METHOD={method.upper()} "
                + " ".join(independents)
                + ("\n  /RESIDUALS DURBIN" if "durbin_watson" in statistics else "") + ".")
    result = RegressionResult(section)

    frame = listwise(ds, [dependent] + independents)
    n = len(frame)
    if n < len(independents) + 2:
        section.note("Warning", "Not enough valid cases for the regression.", style="warning")
        return result
    y = frame[dependent]
    X = frame[independents]
    steps = _select(y, X, independents, method, p_enter, p_remove)
    if not steps:
        section.note("Warning", "No variables were entered into the equation.", style="warning")
        return result
    for s in steps:
        s.fit = _fit(y, X, s.cols)

    if "descriptives" in statistics:
        rows = [[ds.var(c).title, fmt(frame[c].mean(), ds.var(c).decimals + 2),
                 fmt(frame[c].std(ddof=1), ds.var(c).decimals + 3), fmt_int(n)]
                for c in [dependent] + independents]
        section.table(Table("Descriptive Statistics", ["", "Mean", "Std. Deviation", "N"], rows))

    crit = {"enter": "Enter",
            "stepwise": f"Stepwise (Criteria: Probability-of-F-to-enter <= {fmt(p_enter)}, "
                        f"Probability-of-F-to-remove >= {fmt(p_remove)}).",
            "forward": f"Forward (Criterion: Probability-of-F-to-enter <= {fmt(p_enter)})",
            "backward": f"Backward (criterion: Probability of F-to-remove >= {fmt(p_remove)})."}
    entered_rows = []
    for i, s in enumerate(steps, start=1):
        how = crit[method] if not (method == "backward" and i == 1) else "Enter"
        entered_rows.append([str(i), ", ".join(titles[c] for c in s.entered) or ".",
                             ", ".join(titles[c] for c in s.removed) or ".", how])
    section.table(Table("Variables Entered/Removedᵃ", ["Model", "Variables Entered",
                                                       "Variables Removed", "Method"],
                        entered_rows, footnotes=[f"a. Dependent Variable: {dvar.title}"]))

    letters = "bcdefghijklmnopqrstuvwxyz"
    pred_notes = [f"{letters[i % len(letters)]}. Predictors: (Constant)"
                  + "".join(f", {titles[c]}" for c in reversed(s.cols)) for i, s in
                  enumerate(steps)]
    final = steps[-1].fit
    final_cols = steps[-1].cols

    if "model_fit" in statistics:
        cols = ["Model", "R", "R Square", "Adjusted R Square", "Std. Error of the Estimate"]
        change = "r2_change" in statistics
        if change:
            cols += [("Change Statistics", h) for h in ("R Square Change", "F Change", "df1",
                                                        "df2", "Sig. F Change")]
        dw = "durbin_watson" in statistics
        if dw:
            cols.append("Durbin-Watson")
        rows, prev_r2, prev_k = [], 0.0, 0
        for i, s in enumerate(steps):
            f = s.fit
            r2 = f.rsquared
            row = [str(i + 1), fmt(math.sqrt(max(r2, 0))) + sup(letters[i % len(letters)]),
                   fmt(r2), fmt(f.rsquared_adj), fmt(math.sqrt(f.mse_resid), dvar.decimals + 5)]
            if change:
                df1 = abs(len(s.cols) - prev_k) or 1
                d_r2 = r2 - prev_r2
                f_ch = (d_r2 / df1) / ((1 - r2) / f.df_resid) if r2 < 1 else float("nan")
                p_ch = stats.f.sf(abs(f_ch), df1, f.df_resid)
                row += [fmt(d_r2), fmt(f_ch), fmt_int(df1), fmt_int(f.df_resid), fmt_p(p_ch)]
            if dw:
                row.append(fmt(durbin_watson(f.resid)) if i == len(steps) - 1 else "")
            rows.append(row)
            prev_r2, prev_k = r2, len(s.cols)
        section.table(Table("Model Summary", cols, rows,
                            footnotes=pred_notes + [f"Dependent Variable: {dvar.title}"]))

        arows = []
        for i, s in enumerate(steps):
            f = s.fit
            ess, ssr = f.ess, f.ssr
            dfm, dfr = int(f.df_model), int(f.df_resid)
            arows += [[str(i + 1), "Regression", fmt(ess), fmt_int(dfm),
                       fmt(ess / dfm) if dfm else "", fmt(f.fvalue) if dfm else "",
                       fmt_p(f.f_pvalue) + sup(letters[i % len(letters)]) if dfm else ""],
                      ["", "Residual", fmt(ssr), fmt_int(dfr), fmt(ssr / dfr), "", ""],
                      ["", "Total", fmt(ess + ssr), fmt_int(dfm + dfr), "", "", ""]]
        section.table(Table("ANOVAᵃ", ["Model", "", "Sum of Squares", "df", "Mean Square", "F",
                                       "Sig."], arows, row_header_cols=2,
                            footnotes=[f"a. Dependent Variable: {dvar.title}"] + pred_notes))

    sd_y = y.std(ddof=1)
    ci = "ci" in statistics
    col = "collinearity" in statistics
    cols = ["Model", "", ("Unstandardized Coefficients", "B"),
            ("Unstandardized Coefficients", "Std. Error"),
            ("Standardized Coefficients", "Beta"), "t", "Sig."]
    if ci:
        cols += [(f"{pct(level)} Confidence Interval for B", "Lower Bound"),
                 (f"{pct(level)} Confidence Interval for B", "Upper Bound")]
    if col:
        cols += [("Collinearity Statistics", "Tolerance"), ("Collinearity Statistics", "VIF")]
    crows = []
    for i, s in enumerate(steps):
        f = s.fit
        conf = f.conf_int(alpha=1 - level)
        vif = _vif(X, s.cols) if col else {}
        for j, term in enumerate(["const"] + s.cols):
            beta = "" if term == "const" else fmt(f.params[term] * X[term].std(ddof=1) / sd_y)
            row = [str(i + 1) if j == 0 else "",
                   "(Constant)" if term == "const" else titles[term],
                   fmt(f.params[term]), fmt(f.bse[term]), beta, fmt(f.tvalues[term]),
                   fmt_p(f.pvalues[term])]
            if ci:
                row += [fmt(conf.loc[term, 0]), fmt(conf.loc[term, 1])]
            if col:
                tol, v = vif.get(term, (None, None))
                row += [fmt(tol) if tol is not None else "", fmt(v) if v is not None else ""]
            crows.append(row)
    section.table(Table("Coefficientsᵃ", cols, crows, row_header_cols=2,
                        footnotes=[f"a. Dependent Variable: {dvar.title}"]))

    pred = final.fittedvalues
    resid = final.resid
    mse = final.mse_resid
    zpred = (pred - pred.mean()) / pred.std(ddof=1) if pred.std(ddof=1) > 0 else pred * 0
    zresid = resid / math.sqrt(mse) if mse > 0 else resid * 0
    rrows = []
    for label, s, d in (("Predicted Value", pred, dvar.decimals + 4),
                        ("Residual", resid, dvar.decimals + 5),
                        ("Std. Predicted Value", zpred, 3), ("Std. Residual", zresid, 3)):
        rrows.append([label, fmt(s.min(), d), fmt(s.max(), d), fmt(s.mean(), d),
                      fmt(s.std(ddof=1), d), fmt_int(len(s))])
    section.table(Table("Residuals Statisticsᵃ", ["", "Minimum", "Maximum", "Mean",
                                                  "Std. Deviation", "N"], rrows,
                        footnotes=[f"a. Dependent Variable: {dvar.title}"]))

    # ----------------------------------------------------------- assumption tests
    exog = final.model.exog
    if "normality" in assumptions:
        section.note("Classical Assumption Test 1 — Normality of Residuals",
                     "H0: residuals are normally distributed. Residuals are considered normal "
                     "when Sig. > .05.")
        ks = ks_details(resid.to_numpy())

        column = SimpleNamespace(title="Unstandardized Residual", decimals=3)
        section.table(ks_table([(column, ks)]))
        w, sp = shapiro(resid.to_numpy())
        verdict = "Normal" if (ks.get("p", 0) > 0.05) else "Not normal"
        section.table(Table("Normality of Residuals — Summary",
                            ["Test", "Statistic", "df", "Sig.", "Decision (α = .05)"],
                            [["Kolmogorov-Smirnov (Lilliefors)", fmt(ks.get("statistic")),
                              fmt_int(n), fmt_p(ks.get("p")) + ("*" if ks.get("lower_bound")
                                                                else ""), verdict],
                             ["Shapiro-Wilk", fmt(w), fmt_int(n), fmt_p(sp),
                              "Normal" if sp > 0.05 else "Not normal"]],
                            footnotes=["*. This is a lower bound of the true significance."]
                            if ks.get("lower_bound") else []))
    if "multicollinearity" in assumptions and final_cols:
        vif = _vif(X, final_cols)
        rows = [[titles[c], fmt(t), fmt(v),
                 "No multicollinearity" if t > 0.10 and v < 10 else "Multicollinearity"]
                for c, (t, v) in vif.items()]
        section.note("Classical Assumption Test 2 — Multicollinearity",
                     "No multicollinearity when Tolerance > 0.10 and VIF < 10.")
        section.table(Table("Collinearity Statistics", ["Model", "Tolerance", "VIF",
                                                        "Decision"], rows))
    if "heteroscedasticity" in assumptions and final_cols:
        section.note("Classical Assumption Test 3 — Heteroscedasticity",
                     "H0: residual variance is constant (homoscedastic). No heteroscedasticity "
                     "when Sig. > .05. Glejser test: |residual| regressed on the predictors.")
        abs_res = resid.abs()
        g = _fit(abs_res, X, final_cols)
        grows = []
        for j, term in enumerate(["const"] + final_cols):
            beta = "" if term == "const" else \
                fmt(g.params[term] * X[term].std(ddof=1) / abs_res.std(ddof=1))
            grows.append(["(Constant)" if term == "const" else titles[term],
                          fmt(g.params[term]), fmt(g.bse[term]), beta, fmt(g.tvalues[term]),
                          fmt_p(g.pvalues[term]),
                          "" if term == "const" else
                          ("No heteroscedasticity" if g.pvalues[term] > 0.05
                           else "Heteroscedasticity")])
        section.table(Table("Glejser Test — Coefficientsᵃ", [
            "Model", ("Unstandardized Coefficients", "B"),
            ("Unstandardized Coefficients", "Std. Error"),
            ("Standardized Coefficients", "Beta"), "t", "Sig.", "Decision"], grows,
            footnotes=["a. Dependent Variable: ABS_RES (absolute unstandardized residual)"]))
        hrows = []
        try:
            lm, lm_p, _, _ = het_breuschpagan(resid, exog)
            hrows.append(["Breusch-Pagan (LM)", fmt(lm), fmt_int(len(final_cols)), fmt_p(lm_p),
                          "No heteroscedasticity" if lm_p > 0.05 else "Heteroscedasticity"])
        except (ValueError, np.linalg.LinAlgError):
            pass
        try:
            lm, lm_p, _, _ = het_white(resid, exog)
            k = len(final_cols)
            hrows.append(["White (LM)", fmt(lm), fmt_int(k * (k + 3) // 2), fmt_p(lm_p),
                          "No heteroscedasticity" if lm_p > 0.05 else "Heteroscedasticity"])
        except (ValueError, AssertionError, np.linalg.LinAlgError):
            pass
        if hrows:
            section.table(Table("Heteroscedasticity Tests", ["Test", "Chi-Square", "df", "Sig.",
                                                             "Decision (α = .05)"], hrows))
    if "autocorrelation" in assumptions:
        section.note("Classical Assumption Test 4 — Autocorrelation",
                     "Durbin-Watson: no autocorrelation when dU < d < 4 − dU (see the "
                     f"Durbin-Watson table for n = {n}, k = {len(final_cols)}). "
                     "Breusch-Godfrey and Runs test: no autocorrelation when Sig. > .05.")
        arows = [["Durbin-Watson d", fmt(durbin_watson(resid)), "", "", "Compare with dL / dU"]]
        try:
            bg = acorr_breusch_godfrey(final, nlags=1, result_object=False)
            arows.append(["Breusch-Godfrey LM (lag 1)", fmt(bg[0]), "1", fmt_p(bg[1]),
                          "No autocorrelation" if bg[1] > 0.05 else "Autocorrelation"])
        except (ValueError, np.linalg.LinAlgError):
            pass
        rd = runs_details(resid.to_numpy(), "median")
        arows.append(["Runs Test (Z)", fmt(rd["z"]), "", fmt_p(rd["p"]),
                      "No autocorrelation" if rd["p"] > 0.05 else "Autocorrelation"])
        section.table(Table("Autocorrelation Tests", ["Test", "Statistic", "df", "Sig.",
                                                      "Decision (α = .05)"], arows))
        section.table(runs_table([("Unstandardized Residual", rd)], "median"))

    if "histogram" in plots:
        section.image("Histogram", charts.histogram(
            zresid.to_numpy(), f"Histogram — Dependent Variable: {dvar.title}",
            "Regression Standardized Residual"))
    if "pp" in plots:
        section.image("Normal P-P Plot of Regression Standardized Residual", charts.pp_plot(
            zresid.to_numpy(), f"Normal P-P Plot — Dependent Variable: {dvar.title}"))
    if "scatter" in plots:
        section.image("Scatterplot", charts.scatter(
            zpred.to_numpy(), zresid.to_numpy(), f"Scatterplot — Dependent Variable: {dvar.title}",
            "Regression Standardized Predicted Value", "Regression Standardized Residual",
            zero_lines=True))

    index = ds.df.index
    if "pred" in save:
        result.saved["PRE"] = ("Unstandardized Predicted Value", pred.reindex(index))
    if "resid" in save:
        result.saved["RES"] = ("Unstandardized Residual", resid.reindex(index))
    if "zresid" in save:
        result.saved["ZRE"] = ("Standardized Residual", zresid.reindex(index))
    if "zpred" in save:
        result.saved["ZPR"] = ("Standardized Predicted Value", zpred.reindex(index))
    return result

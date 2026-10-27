"""Analyze > Regression: Weight Estimation (SPSS WLS power search), 2-Stage Least Squares
(SPSS 2SLS), Quantile (SPSS 28 QUANTILE REGRESSION), Linear OLS Alternatives (SPSS 29 Ridge,
Lasso, Elastic Net with K-fold cross-validation and holdout) and Nonlinear (SPSS NLR / CNLR:
model expression, start values, parameter constraints, bootstrap standard errors).

Regularised regression follows the scikit-learn objectives used by SPSS's extension commands:
Ridge minimises ||y - Xb||^2 + alpha ||b||^2; Lasso (1/2n)||y - Xb||^2 + alpha ||b||_1; Elastic
Net (1/2n)||y - Xb||^2 + alpha l1 ||b||_1 + alpha (1 - l1)/2 ||b||^2 (own coordinate descent; no
scikit-learn at run time).
"""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, fmt_p, listwise, numeric, pct, sup
from rsstatchem.core.stats.roc import PALETTE, figure, png, style


@dataclass
class ExtResult:
    section: Section
    saved: dict = field(default_factory=dict)


def _sig(p) -> str:
    return fmt_p(p) if p == p else ""


def _wls(y, X, w):
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    e = y - X @ beta
    return beta, e


# ====================================================================== Weight Estimation
def weight_estimation(ds: Dataset, dependent: str, independents: list[str], source: str,
                      start: float = -2.0, stop: float = 2.0, step: float = 0.5,
                      constant: bool = True, each_power: bool = False,
                      save_weight: bool = False) -> ExtResult:
    """SPSS WLS: weights 1 / source**power for each power; the power with the largest
    log-likelihood is used for the final model."""
    section = Section("Weight Estimation")
    result = ExtResult(section)
    section.log(f"WLS {dependent} WITH {' '.join(independents)}\n  /SOURCE={source}"
                f"\n  /POWER={start:g} TO {stop:g} BY {step:g}"
                + ("\n  /CONSTANT" if constant else "\n  /NOCONSTANT")
                + ("\n  /SAVE WEIGHT" if save_weight else "")
                + f"\n  /PRINT={'ALL' if each_power else 'BEST'}.")
    frame = listwise(ds, list(dict.fromkeys([dependent] + independents + [source])))
    s = frame[source].to_numpy(float)
    if np.any(s <= 0):
        section.note("Warning", "The source variable must be positive for every case.",
                     style="warning")
        return result
    y = frame[dependent].to_numpy(float)
    Xr = frame[independents].to_numpy(float)
    X = np.column_stack([np.ones(len(y)), Xr]) if constant else Xr
    n = len(y)
    powers = np.round(np.arange(start, stop + step / 2, step), 10)
    lls = []
    for pw in powers:
        w = 1 / s ** pw
        _, e = _wls(y, X, w)
        sig2 = np.sum(w * e ** 2) / n
        ll = -n / 2 * (math.log(2 * math.pi) + 1 + math.log(sig2)) + 0.5 * np.sum(np.log(w))
        lls.append(ll)
    best = int(np.argmax(lls))
    rows = [[f"Power", fmt(pw, 3), fmt(ll, 3) + (sup("a") if i == best else "")]
            for i, (pw, ll) in enumerate(zip(powers, lls))]
    for r in rows[1:]:
        r[0] = ""
    section.table(Table("Log-Likelihood Valuesᵇ", ["", "", "Value"], rows, row_header_cols=2,
                        footnotes=["a. The corresponding power is selected for further analysis "
                                   "because it maximizes the log-likelihood function.",
                                   f"b. Dependent variable: {ds.var(dependent).title}, source "
                                   f"variable: {ds.var(source).title}"]))
    show = range(len(powers)) if each_power else [best]
    for i in show:
        pw = powers[i]
        w = 1 / s ** pw
        _wls_report(section, ds, dependent, independents, y, X, w, constant, lls[i],
                    f"{ds.var(source).title}, power {fmt(pw, 3)}", f"MOD_{i + 1}")
    if save_weight:
        wser = pd.Series(np.nan, index=ds.df.index)
        wser[frame.index] = 1 / s ** powers[best]
        result.saved["WGT"] = (f"Weight for {dependent} from WLS, MOD_1 {source}** "
                               f"{-powers[best]:g}", wser)
    return result


def _wls_report(section, ds, dependent, independents, y, X, w, constant, ll, weight_text,
                model_name) -> None:
    n, p = X.shape
    beta, e = _wls(y, X, w)
    W = w.sum()
    ybar = np.sum(w * y) / W if constant else 0.0
    sst = float(np.sum(w * (y - ybar) ** 2))
    sse = float(np.sum(w * e ** 2))
    k = p - (1 if constant else 0)
    dfr = n - p
    r2 = 1 - sse / sst if sst > 0 else float("nan")
    adj = 1 - (1 - r2) * (n - (1 if constant else 0)) / dfr
    mse = sse / dfr
    cov = mse * np.linalg.pinv((X * w[:, None]).T @ X)
    se = np.sqrt(np.diag(cov))
    section.table(Table("Model Description", ["", "", ""], [
        ["Dependent Variable", "", ds.var(dependent).title],
        ["Independent Variables", "", ", ".join(ds.var(c).title for c in independents)],
        ["Weight", "Source and Power", weight_text],
        ["Model Name", "", model_name]], row_header_cols=2))
    section.table(Table("Model Summary", ["", ""], [
        ["Multiple R", fmt(math.sqrt(max(r2, 0)))], ["R Square", fmt(r2)],
        ["Adjusted R Square", fmt(adj)], ["Std. Error of the Estimate", fmt(math.sqrt(mse))],
        ["Log-Likelihood Function Value", fmt(ll, 3)]]))
    f = ((sst - sse) / k) / mse if k else float("nan")
    section.table(Table("ANOVA", ["", "Sum of Squares", "df", "Mean Square", "F", "Sig."], [
        ["Regression", fmt(sst - sse), fmt_int(k), fmt((sst - sse) / k) if k else "", fmt(f),
         _sig(stats.f.sf(f, k, dfr)) if k else ""],
        ["Residual", fmt(sse), fmt_int(dfr), fmt(mse), "", ""],
        ["Total", fmt(sst), fmt_int(n - (1 if constant else 0)), "", "", ""]]))
    sd_y = math.sqrt(np.sum(w * (y - np.sum(w * y) / W) ** 2) / W)
    rows = []
    names = (["(Constant)"] if constant else []) + [ds.var(c).title for c in independents]
    for j, nm in enumerate(names):
        t = beta[j] / se[j]
        if nm == "(Constant)":
            rows.append([nm, fmt(beta[j]), fmt(se[j]), "", "", fmt(t),
                         _sig(2 * stats.t.sf(abs(t), dfr))])
        else:
            xj = X[:, j]
            sd_x = math.sqrt(np.sum(w * (xj - np.sum(w * xj) / W) ** 2) / W)
            rows.append([nm, fmt(beta[j]), fmt(se[j]), fmt(beta[j] * sd_x / sd_y),
                         fmt(se[j] * sd_x / sd_y), fmt(t), _sig(2 * stats.t.sf(abs(t), dfr))])
    section.table(Table("Coefficients", ["", ("Unstandardized Coefficients", "B"),
                                         ("Unstandardized Coefficients", "Std. Error"),
                                         ("Standardized Coefficients", "Beta"),
                                         ("Standardized Coefficients", "Std. Error"), "t",
                                         "Sig."], rows))


# ====================================================================== 2SLS
def two_stage(ds: Dataset, dependent: str, explanatory: list[str], instruments: list[str],
              constant: bool = True, save=(), correlations: bool = True) -> ExtResult:
    """SPSS 2SLS: ``explanatory`` lists all regressors (endogenous and exogenous),
    ``instruments`` all exogenous variables (instruments plus exogenous regressors)."""
    section = Section("Two-Stage Least Squares Analysis")
    result = ExtResult(section)
    section.log(f"2SLS {dependent} WITH {' '.join(explanatory)}\n  /INSTRUMENTS "
                + " ".join(instruments) + ("\n  /CONSTANT" if constant else "\n  /NOCONSTANT")
                + ("\n  /SAVE " + " ".join({"pred": "PRED", "resid": "RESID"}[k]
                                           for k in save) if save else "") + ".")
    frame = listwise(ds, list(dict.fromkeys([dependent] + explanatory + instruments)))
    n = len(frame)
    y = frame[dependent].to_numpy(float)
    X = frame[explanatory].to_numpy(float)
    Z = frame[instruments].to_numpy(float)
    if constant:
        X = np.column_stack([np.ones(n), X])
        Z = np.column_stack([np.ones(n), Z])
    if Z.shape[1] < X.shape[1]:
        section.note("Warning", "The equation is underidentified: there must be at least as "
                     "many instrumental variables as explanatory variables.", style="warning")
        return result
    roles = []
    for v in dict.fromkeys([dependent] + explanatory + instruments):
        if v == dependent:
            role = "dependent"
        elif v in explanatory and v in instruments:
            role = "predictor & instrumental"
        elif v in explanatory:
            role = "predictor"
        else:
            role = "instrumental"
        roles.append([ds.var(v).title, role])
    section.table(Table("Model Description", ["", "Type of Variable"], roles,
                        footnotes=["MOD_1"]))
    Pz = Z @ np.linalg.pinv(Z.T @ Z) @ Z.T
    Xh = Pz @ X
    beta = np.linalg.lstsq(Xh, y, rcond=None)[0]
    e = y - X @ beta
    p = X.shape[1]
    dfr = n - p
    sse = float(e @ e)
    sst = float(np.sum((y - y.mean()) ** 2)) if constant else float(y @ y)
    r2 = 1 - sse / sst
    k = p - (1 if constant else 0)
    adj = 1 - (1 - r2) * (n - (1 if constant else 0)) / dfr
    mse = sse / dfr
    cov = mse * np.linalg.pinv(Xh.T @ Xh)
    se = np.sqrt(np.diag(cov))
    section.table(Table("Model Summary", ["", ""], [
        ["Equation 1", ""], ["Multiple R", fmt(math.sqrt(max(r2, 0)))],
        ["R Square", fmt(r2)], ["Adjusted R Square", fmt(adj)],
        ["Std. Error of the Estimate", fmt(math.sqrt(mse))]]))
    ssr = sst - sse
    f = (ssr / k) / mse if k else float("nan")
    section.table(Table("ANOVA", ["", "", "Sum of Squares", "df", "Mean Square", "F", "Sig."], [
        ["Equation 1", "Regression", fmt(ssr), fmt_int(k), fmt(ssr / k) if k else "", fmt(f),
         _sig(stats.f.sf(f, k, dfr)) if k else ""],
        ["", "Residual", fmt(sse), fmt_int(dfr), fmt(mse), "", ""],
        ["", "Total", fmt(sst), fmt_int(dfr + k), "", "", ""]], row_header_cols=2))
    names = (["(Constant)"] if constant else []) + [ds.var(c).title for c in explanatory]
    sd_y = y.std(ddof=1)
    rows = []
    for j, nm in enumerate(names):
        t = beta[j] / se[j]
        beta_std = "" if nm == "(Constant)" else fmt(beta[j] * X[:, j].std(ddof=1) / sd_y)
        rows.append(["Equation 1" if j == 0 else "", nm, fmt(beta[j]), fmt(se[j]), beta_std,
                     fmt(t), _sig(2 * stats.t.sf(abs(t), dfr))])
    section.table(Table("Coefficients", ["", "", ("Unstandardized Coefficients", "B"),
                                         ("Unstandardized Coefficients", "Std. Error"), "Beta",
                                         "t", "Sig."], rows, row_header_cols=2))
    if correlations:
        idx = list(range(1 if constant else 0, p))
        C = cov[np.ix_(idx, idx)]
        d = np.sqrt(np.diag(C))
        R = C / np.outer(d, d)
        heads = [ds.var(c).title for c in explanatory]
        section.table(Table("Coefficient Correlations", ["", "", ""] + heads,
                            [["Equation 1" if i == 0 else "", "Correlations" if i == 0 else "",
                              heads[i]] + [fmt(v) for v in R[i]] for i in range(len(idx))],
                            row_header_cols=3))
    index = ds.df.index
    if save:
        full = listwise(ds, explanatory)
        Xa = full.to_numpy(float)
        if constant:
            Xa = np.column_stack([np.ones(len(Xa)), Xa])
        pred = pd.Series(np.nan, index=index)
        pred[full.index] = Xa @ beta
        if "pred" in save:
            result.saved["FIT"] = (f"Fit for {dependent} from 2SLS, MOD_1 Equation 1", pred)
        if "resid" in save:
            result.saved["ERR"] = (f"Error for {dependent} from 2SLS, MOD_1 Equation 1",
                                   numeric(ds, dependent) - pred)
    return result


# ====================================================================== Quantile
def quantile_regression(ds: Dataset, dependent: str, factors: list[str] = (),
                        covariates: list[str] = (), quantiles=(0.5,), intercept: bool = True,
                        iid: bool = False, bandwidth: str = "bofinger", level: float = 0.95,
                        plot_estimates: bool = True, save=()) -> ExtResult:
    """SPSS QUANTILE REGRESSION. Standard errors assume iid errors or use the kernel
    (sandwich) estimator with Bofinger or Hall-Sheather bandwidth (statsmodels QuantReg)."""
    import warnings

    from statsmodels.regression.quantile_regression import QuantReg

    section = Section("Quantile Regression")
    result = ExtResult(section)
    factors, covariates = list(factors), list(covariates)
    section.log(f"QUANTILE REGRESSION {dependent}" + (" BY " + " ".join(factors) if factors
                                                      else "")
                + (" WITH " + " ".join(covariates) if covariates else "")
                + f"\n  /CRITERIA METHOD=AUTO SIG={1 - level:g} "
                  f"BANDWIDTH={bandwidth.upper()} {'IID' if iid else 'NOIID'}"
                + "\n  /QUANTILES " + " ".join(f"{q:g}" for q in quantiles)
                + "\n  /PRINT MODEL_QUALITY PARAMETERS"
                + ("\n  /PLOT PARAMETERS" if plot_estimates else "") + ".")
    frame = pd.DataFrame({"__y": numeric(ds, dependent),
                          **{f: ds.analysis_series(f) for f in factors},
                          **{c: numeric(ds, c) for c in covariates}}).dropna()
    y = frame["__y"].to_numpy(float)
    cols, names, redundant = [], [], []
    if intercept:
        cols.append(np.ones(len(y)))
        names.append("(Intercept)")
    for f in factors:
        lv = sorted(frame[f].unique(), key=lambda v: (isinstance(v, str), v))
        for v in lv[:-1]:
            cols.append((frame[f] == v).to_numpy(float))
            names.append(f"[{f}={v:g}]" if isinstance(v, float) else f"[{f}={v}]")
        redundant.append(f"[{f}={lv[-1]:g}]" if isinstance(lv[-1], float) else
                         f"[{f}={lv[-1]}]")
    for c in covariates:
        cols.append(frame[c].to_numpy(float))
        names.append(c)
    X = np.column_stack(cols)
    n, p = X.shape
    z = stats.t.ppf(0.5 + level / 2, n - p)
    estimates = {}
    for q in quantiles:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = QuantReg(y, X).fit(q=q, vcov="iid" if iid else "robust", kernel="epa",
                                     bandwidth="bofinger" if bandwidth == "bofinger"
                                     else "hsheather", max_iter=5000)
        b, se = np.asarray(res.params), np.asarray(res.bse)
        estimates[q] = (b, se)
        resid = y - X @ b
        mae = float(np.mean(np.abs(resid)))
        section.table(Table(f"Model Quality — Quantile = {q:g}", ["", ""], [
            ["Pseudo R Squared", fmt(res.prsquared)], ["Mean Absolute Error (MAE)", fmt(mae)]],
            footnotes=[f"a. Dependent Variable: {ds.var(dependent).title}",
                       "b. Model: " + ", ".join(names)]))
        rows = []
        for j, nm in enumerate(names):
            t = b[j] / se[j] if se[j] > 0 else float("nan")
            rows.append([nm, fmt(b[j]), fmt(se[j]), fmt(t), fmt_int(n - p),
                         _sig(2 * stats.t.sf(abs(t), n - p)), fmt(b[j] - z * se[j]),
                         fmt(b[j] + z * se[j])])
        for nm in redundant:
            rows.append([nm, "0" + sup("c"), "", "", "", "", "", ""])
        g = f"{pct(level)} Confidence Interval"
        section.table(Table(f"Parameter Estimates — Quantile = {q:g}", [
            "Parameter", "Coefficient", "Std. Error", "t", "df", "Sig.", (g, "Lower Bound"),
            (g, "Upper Bound")], rows, footnotes=[
            f"a. Dependent Variable: {ds.var(dependent).title}", "b. Model: " + ", ".join(names)]
            + (["c. Set to zero because this parameter is redundant."] if redundant else [])))
        if "pred" in save:
            pred = pd.Series(np.nan, index=ds.df.index)
            pred[frame.index] = X @ b
            result.saved[f"PRED_Q{int(round(q * 100))}"] = (
                f"Predicted value of {dependent} for quantile {q:g}", pred)
    if plot_estimates and len(quantiles) > 1:
        qs = sorted(estimates)
        k = len(names)
        cols_n = min(3, k)
        rows_n = math.ceil(k / cols_n)
        fig = figure(3.0 * cols_n, 2.6 * rows_n)
        for j, nm in enumerate(names):
            ax = fig.add_subplot(rows_n, cols_n, j + 1)
            bs = np.array([estimates[q][0][j] for q in qs])
            ses = np.array([estimates[q][1][j] for q in qs])
            ax.fill_between(qs, bs - z * ses, bs + z * ses, color=PALETTE[0], alpha=0.18)
            ax.plot(qs, bs, "o-", color=PALETTE[0], ms=4)
            ax.axhline(0, color="#999999", lw=0.8)
            style(ax, nm, "Quantile", "Estimate")
        section.image("Parameter Estimates by Quantile", png(fig), width=560)
    return result


# ====================================================================== regularised
def _prepare_xy(ds, dependent, predictors):
    frame = listwise(ds, [dependent] + list(predictors))
    return frame, frame[dependent].to_numpy(float), frame[list(predictors)].to_numpy(float)


def elastic_net_path(X: np.ndarray, y: np.ndarray, alpha: float, l1_ratio: float,
                     max_iter: int = 10000, tol: float = 1e-10,
                     start: np.ndarray | None = None) -> np.ndarray:
    """Coordinate descent for (1/2n)||y - Xb||^2 + a*l1*||b||_1 + a*(1-l1)/2*||b||^2 on
    centred data (scikit-learn's ElasticNet objective)."""
    n, p = X.shape
    b = np.zeros(p) if start is None else start.copy()
    col_sq = (X ** 2).sum(axis=0) / n
    r = y - X @ b
    l1, l2 = alpha * l1_ratio, alpha * (1 - l1_ratio)
    for _ in range(max_iter):
        max_change = 0.0
        for j in range(p):
            if col_sq[j] == 0:
                continue
            bj = b[j]
            rho = X[:, j] @ r / n + col_sq[j] * bj
            new = np.sign(rho) * max(abs(rho) - l1, 0.0) / (col_sq[j] + l2)
            if new != bj:
                r += X[:, j] * (bj - new)
                b[j] = new
                max_change = max(max_change, abs(new - bj))
        if max_change < tol:
            break
    return b


def _fit_penalised(kind, X, y, alpha, l1_ratio):
    """Return (intercept, coefficients) on the scale of X (X and y are centred inside)."""
    xm, ym = X.mean(axis=0), y.mean()
    Xc, yc = X - xm, y - ym
    if kind == "ridge":
        p = X.shape[1]
        b = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(p), Xc.T @ yc)
    elif kind == "lasso":
        b = elastic_net_path(Xc, yc, alpha, 1.0)
    else:
        b = elastic_net_path(Xc, yc, alpha, l1_ratio)
    return ym - xm @ b, b


def _r2(y, pred):
    sst = np.sum((y - y.mean()) ** 2)
    return 1 - np.sum((y - pred) ** 2) / sst if sst > 0 else float("nan")


def regularized(ds: Dataset, dependent: str, predictors: list[str], kind: str = "ridge",
                alpha: float = 1.0, l1_ratio: float = 0.5, mode: str = "fit",
                alphas=None, l1_ratios=None, folds: int = 5, standardize: bool = True,
                holdout: float = 0.0, seed: int = 2000000, trace: bool = True,
                save=()) -> ExtResult:
    """SPSS 29 Linear OLS Alternatives. ``mode`` "fit" (given alpha) or "cv" (K-fold
    cross-validated grid of ``alphas`` / ``l1_ratios``; the best mean test R^2 wins).
    ``holdout`` = fraction of cases kept out for testing."""
    title = {"ridge": "Linear Ridge Regression", "lasso": "Linear Lasso Regression",
             "enet": "Linear Elastic Net Regression"}[kind]
    section = Section(title)
    result = ExtResult(section)
    cmd = {"ridge": "LINEAR_RIDGE", "lasso": "LINEAR_LASSO", "enet": "LINEAR_ELASTIC_NET"}[kind]
    alphas = list(alphas) if alphas is not None else list(np.logspace(-3, 2, 21))
    l1_ratios = list(l1_ratios) if l1_ratios is not None else [0.1, 0.25, 0.5, 0.75, 0.9]
    section.log(f"{cmd} {dependent} WITH {' '.join(predictors)}\n  /MODE="
                + ("FIT" if mode == "fit" else f"CROSSVALID NFOLDS={folds}")
                + (f"\n  /ALPHAS={alpha:g}" if mode == "fit" else
                   f"\n  /ALPHAS=START={min(alphas):g} STOP={max(alphas):g} N={len(alphas)}")
                + (f"\n  /L1RATIOS={l1_ratio:g}" if kind == "enet" and mode == "fit" else "")
                + (f"\n  /PARTITION TRAINING={100 * (1 - holdout):g} HOLDOUT={100 * holdout:g}"
                   if holdout else "")
                + f"\n  /CRITERIA STANDARDIZE={'TRUE' if standardize else 'FALSE'} "
                  f"SEED={seed}.")
    frame, y, X = _prepare_xy(ds, dependent, predictors)
    n, p = X.shape
    rng = np.random.default_rng(seed)
    test_mask = np.zeros(n, bool)
    if holdout > 0:
        test_mask[rng.permutation(n)[:int(round(holdout * n))]] = True
    tr = ~test_mask
    sd = X[tr].std(axis=0, ddof=0) if standardize else np.ones(p)
    sd[sd == 0] = 1
    Xs = X / sd
    grid = [(a, r) for a in (alphas if mode == "cv" else [alpha])
            for r in ((l1_ratios if mode == "cv" else [l1_ratio]) if kind == "enet" else
                      [1.0 if kind == "lasso" else 0.0])]
    if mode == "cv":
        idx = rng.permutation(np.flatnonzero(tr))
        parts = np.array_split(idx, folds)
        crows, scores = [], []
        for a, r in grid:
            r2s, mses = [], []
            for k in range(folds):
                te = parts[k]
                trn = np.setdiff1d(idx, te)
                b0, b = _fit_penalised(kind, Xs[trn], y[trn], a, r)
                pr = b0 + Xs[te] @ b
                r2s.append(_r2(y[te], pr))
                mses.append(float(np.mean((y[te] - pr) ** 2)))
            scores.append(np.mean(r2s))
            crows.append([fmt(a, 4)] + ([fmt(r, 2)] if kind == "enet" else [])
                         + [fmt(np.mean(r2s)), fmt(np.std(r2s)), fmt(np.mean(mses))])
        best = int(np.argmax(scores))
        crows[best][-1] += sup("a")
        alpha, l1_ratio = grid[best]
        section.table(Table("Cross-Validation Results", ["Alpha"] + (["L1 Ratio"]
                                                                     if kind == "enet" else [])
                            + ["Mean Test R Square", "Std. Deviation", "Mean Test MSE"],
                            crows, footnotes=[f"a. Best model ({folds}-fold cross-validation, "
                                              "largest mean test R square)."]))
    b0, b = _fit_penalised(kind, Xs[tr], y[tr], alpha, l1_ratio)
    coef = b / sd
    pred = b0 + Xs @ b
    r2_tr = _r2(y[tr], pred[tr])
    mse_tr = float(np.mean((y[tr] - pred[tr]) ** 2))
    rows = [["Alpha", fmt(alpha, 4)]] + ([["L1 Ratio", fmt(l1_ratio, 2)]] if kind == "enet"
                                         else []) + [
        ["R Square (training)", fmt(r2_tr)], ["Mean Squared Error (training)", fmt(mse_tr)],
        ["N (training)", fmt_int(tr.sum())]]
    if holdout:
        rows += [["R Square (holdout)", fmt(_r2(y[test_mask], pred[test_mask]))],
                 ["Mean Squared Error (holdout)",
                  fmt(float(np.mean((y[test_mask] - pred[test_mask]) ** 2)))],
                 ["N (holdout)", fmt_int(test_mask.sum())]]
    section.table(Table("Model Summary", ["", "Value"], rows,
                        footnotes=[f"Dependent Variable: {ds.var(dependent).title}"]
                        + (["Predictors were standardized for estimation; coefficients are "
                            "reported in the original metric."] if standardize else [])))
    crow = [["(Intercept)", fmt(b0), ""]]
    for j, c in enumerate(predictors):
        crow.append([ds.var(c).title, fmt(coef[j]), fmt(b[j]) if standardize else ""])
    section.table(Table("Coefficients", ["", "Coefficient", "Standardized Coefficient"]
                        if standardize else ["", "Coefficient", ""], crow))
    if trace:
        path_alphas = np.logspace(math.log10(min(alphas)), math.log10(max(alphas)), 30)
        coefs = []
        for a in path_alphas:
            _, bb = _fit_penalised(kind, Xs[tr], y[tr], a, l1_ratio)
            coefs.append(bb)
        coefs = np.array(coefs)
        fig = figure(6.0, 4.0)
        ax = fig.add_subplot()
        for j, c in enumerate(predictors):
            ax.plot(path_alphas, coefs[:, j], color=PALETTE[j % len(PALETTE)], lw=1.5,
                    label=ds.var(c).title)
        ax.axvline(alpha, color="#555555", ls="--", lw=1)
        ax.set_xscale("log")
        style(ax, "Coefficient Paths", "Alpha", "Standardized coefficient" if standardize
              else "Coefficient")
        ax.legend(fontsize=7, frameon=False)
        section.image("Coefficient Paths", png(fig))
    if "pred" in save:
        s = pd.Series(np.nan, index=ds.df.index)
        s[frame.index] = pred
        result.saved["PRED"] = (f"Predicted {dependent} ({title})", s)
    return result


# ====================================================================== Nonlinear (NLR)
_FUNCS = {"exp": np.exp, "ln": np.log, "log": np.log, "lg10": np.log10, "log10": np.log10,
          "sqrt": np.sqrt, "abs": np.abs, "sin": np.sin, "cos": np.cos, "tan": np.tan,
          "artan": np.arctan, "arctan": np.arctan, "min": np.minimum, "max": np.maximum,
          "cdfnorm": stats.norm.cdf}
_BIN = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide,
        ast.Pow: np.power}
_CMP = {ast.Lt: np.less, ast.LtE: np.less_equal, ast.Gt: np.greater,
        ast.GtE: np.greater_equal, ast.Eq: np.equal, ast.NotEq: np.not_equal}


def compile_model(expr: str):
    """Parse an NLR model expression (SPSS syntax: ** or ^ for powers, EXP/LN/SQRT...,
    comparisons give 1/0 so conditional models such as (x < c)*a + (x >= c)*b work)."""
    text = expr.strip().replace("^", "**").replace("=", "==").replace("<==", "<=").replace(
        ">==", ">=").replace("~==", "!=").replace("!==", "!=")
    tree = ast.parse(text, mode="eval")

    def names_in(node):
        out = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Name):
                out.add(n.id)
        return out

    def ev(node, env):
        if isinstance(node, ast.Expression):
            return ev(node.body, env)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            key = node.id.lower()
            if key in env:
                return env[key]
            raise ValueError(f"Unknown name in model: {node.id}")
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            return _BIN[type(node.op)](ev(node.left, env), ev(node.right, env))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = ev(node.operand, env)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _CMP:
            return _CMP[type(node.ops[0])](ev(node.left, env),
                                           ev(node.comparators[0], env)).astype(float)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and \
                node.func.id.lower() in _FUNCS and not node.keywords:
            return _FUNCS[node.func.id.lower()](*[ev(a, env) for a in node.args])
        raise ValueError("Unsupported element in the model expression.")

    return (lambda env: ev(tree, env)), {n.lower() for n in names_in(tree)} - set(_FUNCS)


def parse_parameters(text: str) -> dict[str, float]:
    """"a=1 b=0.5" or one per line -> {"a": 1.0, "b": 0.5}."""
    out = {}
    for tok in text.replace(",", " ").replace(";", " ").split():
        if "=" not in tok:
            raise ValueError(f"Parameter '{tok}' needs a starting value (name=value).")
        k, v = tok.split("=", 1)
        out[k.strip().lower()] = float(v)
    return out


def parse_constraints(lines: list[str], params: list[str]):
    """Linear constraints such as "a >= 0", "b <= 10", "a + b <= 1" ->
    (bounds dict, list of (coef vector, op, rhs))."""
    bounds = {p: [-np.inf, np.inf] for p in params}
    linear = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        for op in ("<=", ">=", "="):
            if op in line:
                lhs, rhs = line.split(op, 1)
                break
        else:
            raise ValueError(f"Constraint '{line}' needs <=, >= or =.")
        coef = np.zeros(len(params))
        f, _ = compile_model(lhs)
        base = float(f({p: 0.0 for p in params}))
        for i, p in enumerate(params):
            env = {q: 0.0 for q in params}
            env[p] = 1.0
            coef[i] = float(f(env)) - base
        rhs_v = float(compile_model(rhs)[0]({p: 0.0 for p in params})) - base
        nz = np.flatnonzero(coef)
        if len(nz) == 1 and op != "=":
            p = params[nz[0]]
            val = rhs_v / coef[nz[0]]
            if (op == ">=") == (coef[nz[0]] > 0):
                bounds[p][0] = max(bounds[p][0], val)
            else:
                bounds[p][1] = min(bounds[p][1], val)
        else:
            linear.append((coef, op, rhs_v))
    return bounds, linear


def nonlinear(ds: Dataset, dependent: str, model: str, parameters: dict[str, float],
              constraints: list[str] = (), loss: str | None = None, level: float = 0.95,
              bootstrap: int = 0, seed: int = 2000000, save=(),
              max_iter: int = 200) -> ExtResult:
    """SPSS NLR (no constraints: Levenberg-Marquardt) / CNLR (constraints: sequential
    quadratic programming). ``loss`` is an optional expression using RESID_ and PRED_
    (default: sum of squared residuals). ``save`` keys: pred, resid, deriv, loss."""
    section = Section("Nonlinear Regression Analysis")
    result = ExtResult(section)
    pnames = list(parameters)
    f, used = compile_model(model)
    var_names = [v for v in ds.names() if v.lower() in used]
    section.log("* NonLinear Regression.\nMODEL PROGRAM " + " ".join(
        f"{k}={v:g}" for k, v in parameters.items()) + f".\nCOMPUTE PRED_ = {model}.\n"
        + ("CONSTRAINED FUNCTIONS.\n" if loss else "")
        + ("CNLR" if constraints or loss else "NLR") + f" {dependent}"
        + ("\n  /BOUNDS " + "; ".join(constraints) if constraints else "")
        + (f"\n  /BOOTSTRAP={bootstrap}" if bootstrap else "")
        + ("\n  /SAVE " + " ".join(k.upper() for k in save) if save else "")
        + "\n  /CRITERIA SSCONVERGENCE 1E-8 PCON 1E-8.")
    unknown = used - {v.lower() for v in var_names} - set(pnames)
    if unknown:
        section.note("Warning", "Unknown names in the model: " + ", ".join(sorted(unknown)),
                     style="warning")
        return result
    frame = listwise(ds, list(dict.fromkeys([dependent] + var_names)))
    y = frame[dependent].to_numpy(float)
    data_env = {v.lower(): frame[v].to_numpy(float) for v in var_names}
    n, k = len(y), len(pnames)

    def predict(theta, env=data_env):
        e = dict(env)
        e.update({p: float(t) for p, t in zip(pnames, theta)})
        out = np.asarray(f(e), float)
        return np.broadcast_to(out, (len(next(iter(env.values()))) if env else n,)).copy()

    lossf = compile_model(loss)[0] if loss else None

    def objective(theta, yy=y, env=data_env):
        pr = predict(theta, env)
        if lossf is None:
            return float(np.sum((yy - pr) ** 2))
        e = dict(env)
        e.update({p: float(t) for p, t in zip(pnames, theta)})
        e["resid_"], e["pred_"] = yy - pr, pr
        return float(np.sum(lossf(e)))

    bounds, linear = parse_constraints(list(constraints), pnames)
    x0 = np.array([parameters[p] for p in pnames], float)
    history = []
    constrained = bool(linear) or any(np.isfinite(b).any() for b in bounds.values()) or \
        lossf is not None

    def solve(yy, env, start, record=False):
        if not constrained:
            res = optimize.least_squares(lambda t: yy - predict(t, env), start, method="lm",
                                         max_nfev=max_iter * 50, xtol=1e-12, ftol=1e-12)
            return res.x, bool(res.success)
        cons = [{"type": "ineq" if op != "=" else "eq",
                 "fun": (lambda t, c=c, op=op, r=r: (r - c @ t) if op == "<=" else
                         (c @ t - r))} for c, op, r in linear]
        res = optimize.minimize(lambda t: objective(t, yy, env), start, method="SLSQP",
                                bounds=[tuple(bounds[p]) for p in pnames], constraints=cons,
                                callback=(lambda t: history.append((objective(t), t.copy())))
                                if record else None,
                                options={"maxiter": max_iter * 10, "ftol": 1e-14})
        return res.x, bool(res.success)

    history.append((objective(x0), x0.copy()))
    try:
        theta, ok = solve(y, data_env, x0, record=True)
    except (ValueError, FloatingPointError) as exc:
        section.note("Warning", f"The model could not be evaluated: {exc}", style="warning")
        return result
    if not constrained:
        history.append((objective(theta), theta.copy()))
    pred = predict(theta)
    resid = y - pred
    sse = float(resid @ resid)
    eps = 1e-6 * np.maximum(np.abs(theta), 1)
    J = np.column_stack([(predict(theta + eps[i] * np.eye(k)[i]) -
                          predict(theta - eps[i] * np.eye(k)[i])) / (2 * eps[i])
                         for i in range(k)])
    dfr = n - k
    mse = sse / dfr
    cov = mse * np.linalg.pinv(J.T @ J)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    hrows = [[f"{i}.0" if i else "0.1"] + [fmt(h[0], 3)] + [fmt(v, 3) for v in h[1]]
             for i, h in enumerate(history[:50])]
    section.table(Table("Iteration Historyᵇ", ["Iteration Numberᵃ", "Residual Sum of Squares"]
                        + [("Parameter", p) for p in pnames], hrows,
                        footnotes=["Derivatives are calculated numerically.",
                                   "a. Major iteration number is displayed to the left of the "
                                   "decimal, and minor iteration number is to the right of "
                                   "the decimal.",
                                   "b. Run stopped after " + str(len(history) - 1)
                                   + " iterations because the relative reduction between "
                                     "successive residual sums of squares is at most "
                                     "SSCON = 1.000E-8." if ok else
                                   "b. Iterations stopped before convergence."]))
    q = stats.t.ppf(0.5 + level / 2, dfr)
    g = f"{pct(level)} Confidence Interval"
    cols = ["Parameter", "Estimate", "Std. Error", (g, "Lower Bound"), (g, "Upper Bound")]
    rows = [[p, fmt(theta[i]), fmt(se[i]), fmt(theta[i] - q * se[i]),
             fmt(theta[i] + q * se[i])] for i, p in enumerate(pnames)]
    if bootstrap:
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(int(bootstrap)):
            idx = rng.integers(0, n, n)
            env = {kk: vv[idx] for kk, vv in data_env.items()}
            try:
                tb, _ = solve(y[idx], env, theta)
                boots.append(tb)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                continue
        B = np.array(boots)
        bse = B.std(axis=0, ddof=1)
        lo, hi = np.percentile(B, [100 * (1 - level) / 2, 100 * (1 + level) / 2], axis=0)
        cols = ["Parameter", "Estimate", "Std. Error", (g, "Lower Bound"),
                (g, "Upper Bound")]
        rows = [[p, fmt(theta[i]), fmt(bse[i]), fmt(lo[i]), fmt(hi[i])]
                for i, p in enumerate(pnames)]
        section.table(Table("Parameter Estimates", cols, rows, footnotes=[
            f"Bootstrap standard errors and percentile intervals ({len(B)} samples, seed "
            f"{seed})."]))
    else:
        section.table(Table("Parameter Estimates", cols, rows))
    d = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
    R = cov / np.outer(d, d)
    section.table(Table("Correlations of Parameter Estimates", [""] + pnames,
                        [[p] + [fmt(v) for v in R[i]] for i, p in enumerate(pnames)]))
    unc = float(y @ y)
    sst = float(np.sum((y - y.mean()) ** 2))
    section.table(Table("ANOVAᵃ", ["Source", "Sum of Squares", "df", "Mean Squares"], [
        ["Regression", fmt(unc - sse), fmt_int(k), fmt((unc - sse) / k)],
        ["Residual", fmt(sse), fmt_int(dfr), fmt(mse)],
        ["Uncorrected Total", fmt(unc), fmt_int(n), ""],
        ["Corrected Total", fmt(sst), fmt_int(n - 1), ""]],
        footnotes=[f"Dependent variable: {ds.var(dependent).title}",
                   f"a. R squared = 1 - (Residual Sum of Squares) / (Corrected Sum of Squares) "
                   f"= {fmt(1 - sse / sst)}."]))
    if lossf is not None:
        section.note("Loss Function", f"User-defined loss: {loss}; minimum = "
                     f"{fmt(objective(theta), 4)}.")
    idx = ds.df.index
    if save:
        if "pred" in save:
            s = pd.Series(np.nan, index=idx)
            s[frame.index] = pred
            result.saved["PRED"] = ("Predicted Values", s)
        if "resid" in save:
            s = pd.Series(np.nan, index=idx)
            s[frame.index] = resid
            result.saved["RESID"] = ("Residuals", s)
        if "deriv" in save:
            for i, p in enumerate(pnames):
                s = pd.Series(np.nan, index=idx)
                s[frame.index] = J[:, i]
                result.saved[f"D_{p.upper()}"] = (f"Derivative with respect to {p}", s)
        if "loss" in save and lossf is not None:
            e = dict(data_env)
            e.update({p: float(t) for p, t in zip(pnames, theta)})
            e["resid_"], e["pred_"] = resid, pred
            s = pd.Series(np.nan, index=idx)
            s[frame.index] = np.broadcast_to(lossf(e), (n,))
            result.saved["LOSS"] = ("Loss Function Values", s)
    return result


def selftest() -> None:
    rng = np.random.default_rng(5)
    n = 120
    x = rng.uniform(1, 10, n)
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    endo = z + 0.5 * u + rng.normal(size=n)
    y = 1 + 2 * x + rng.normal(0, 0.3 * x)
    ds = Dataset(pd.DataFrame({"y": y, "x": x, "z": z, "endo": endo,
                               "yy": 1 + endo + u, "g": rng.integers(1, 3, n) * 1.0,
                               "c": 5 * np.exp(-0.3 * x) + rng.normal(0, 0.05, n)}))
    weight_estimation(ds, "y", ["x"], "x")
    two_stage(ds, "yy", ["endo"], ["z"], save=("pred",))
    quantile_regression(ds, "y", ["g"], ["x"], quantiles=(0.25, 0.5, 0.75))
    regularized(ds, "y", ["x", "z"], "enet", mode="cv", alphas=[0.01, 0.1], l1_ratios=[0.5],
                folds=3)
    nonlinear(ds, "c", "a*exp(-b*x)", {"a": 1.0, "b": 0.1}, save=("pred",))
    nonlinear(ds, "c", "a*exp(-b*x)", {"a": 1.0, "b": 0.1}, constraints=["b >= 0.31"])

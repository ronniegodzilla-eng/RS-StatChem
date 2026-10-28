"""SPSS Categories procedures: CATREG (Regression > Optimal Scaling), CATPCA, MULTIPLE
CORRESPONDENCE (MCA), OVERALS (Dimension Reduction > Optimal Scaling) and PREFSCAL
(Scale > Multidimensional Unfolding).

This module turns a Dataset plus SPSS-style options into output Sections (tables laid out as
SPSS prints them, plots, saved variables). The numerical work is in
:mod:`rsstatchem.core.stats.optscale_engine` (alternating least squares as documented in IBM
SPSS Statistics Algorithms).

Verification (tests/test_wp10_optscale.py):

* CATREG with all variables numeric equals standardized OLS (statsmodels); ridge equals
  (R + lambda2*I)^-1 r; lasso equals coordinate-descent lasso with alpha = lambda1/2.
* CATPCA with all variables numeric equals PCA of the correlation matrix.
* MCA eigenvalues equal m times the principal inertias of correspondence analysis of the
  indicator matrix (closed form of homogeneity analysis; Gifi 1990, ch. 3).
* CATPCA with single nominal variables in one dimension equals the first MCA dimension.
* OVERALS with two sets of numeric variables: (K*eigenvalue - 1)/(K - 1) equals the canonical
  correlations (statsmodels CanCorr).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from rsstatchem.core.dataset import NUMERIC, Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats import optscale_engine as E
from rsstatchem.core.stats.common import fmt, fmt_int, fmt_p, value_text
from rsstatchem.core.stats.wp4_util import COLORS, Result, figure, png

LEVELS = E.LEVELS
DISCRETIZATION = {"auto": "Unspecified (SPSS default)", "rank": "Ranking",
                  "multiply": "Multiplying", "normal": "Grouping: normal distribution",
                  "uniform": "Grouping: uniform distribution", "equal": "Grouping: equal intervals",
                  "none": "None (use values)"}
NORMALIZATIONS = {"vprincipal": "Variable Principal", "oprincipal": "Object Principal",
                  "symmetrical": "Symmetrical", "independent": "Independent"}


@dataclass
class VarSpec:
    """One analysis variable as set in "Define Scale and Weight" / "Define Scale" plus its
    discretization (SPSS Discretize... sub-dialog)."""
    name: str
    level: str = "spord"
    degree: int = 2
    knots: int = 2
    weight: float = 1.0
    disc: str = "auto"
    ncat: int = 7
    interval: float = 1.0

    @staticmethod
    def of(x) -> "VarSpec":
        if isinstance(x, VarSpec):
            return x
        if isinstance(x, str):
            return VarSpec(x)
        return VarSpec(**x)


def _f(x, d=3):
    return fmt(x, d)


def level_text(spec: VarSpec) -> str:
    t = LEVELS[spec.level]
    if spec.level in ("spord", "spnom"):
        t += f" (degree {spec.degree}, {spec.knots} interior knots)"
    return t


# ===================================================================== data preparation
def _indicators(ds: Dataset, spec: VarSpec) -> tuple[np.ndarray, dict]:
    """Category indicators (NaN = missing) and labels per indicator value."""
    var = ds.var(spec.name)
    s = ds.analysis_series(spec.name)
    if var.type != NUMERIC:
        ind = E.discretize(s.to_numpy(object), strings=True)
        vals = sorted({str(v) for v in s.dropna() if str(v) != ""})
        labels = {float(i + 1): str(value_text(var, v)) for i, v in enumerate(vals)}
        return ind, labels
    x = pd.to_numeric(s, errors="coerce").to_numpy(float)
    method = spec.disc
    if method == "auto":
        v = x[~np.isnan(x)]
        integer = len(v) and np.all(np.abs(v - np.round(v)) < 1e-9)
        if integer and v.min() >= 1:
            method = "none"
        elif integer or len(np.unique(v)) <= spec.ncat:
            method = "rank"
        else:
            method = "normal"
    ind = E.discretize(x, method, spec.ncat, spec.interval)
    labels = {}
    ok = ~np.isnan(ind)
    if method in ("none", "rank"):
        for code in np.unique(ind[ok]):
            orig = x[ok & (ind == code)][0]
            labels[float(code)] = str(value_text(var, orig))
    else:
        for code in np.unique(ind[ok]):
            labels[float(code)] = E._num(code)
    return ind, labels


def prepare_vars(ds: Dataset, specs: list[VarSpec], missing: str = "listwise",
                 weights: str | None = None) -> tuple[list[E.CatVar], np.ndarray, np.ndarray,
                                                      dict]:
    """CatVars on the cases used in the analysis. ``missing``: listwise (exclude objects
    with missing values), mode (impute the modal category), extra (impute an extra category)
    or passive (CATPCA/MCA: the object is ignored for that variable only)."""
    inds, labs = [], []
    for sp in specs:
        a, b = _indicators(ds, sp)
        inds.append(a)
        labs.append(b)
    n = ds.n_cases
    w = np.ones(n)
    if weights:
        w = pd.to_numeric(ds.analysis_series(weights), errors="coerce").to_numpy(float)
    wok = ~np.isnan(w) & (w > 0)
    M = np.column_stack(inds) if inds else np.zeros((n, 0))
    anymiss = np.isnan(M).any(axis=1)
    info = {"valid": int((~anymiss & wok).sum()), "with_missing": int((anymiss & wok).sum()),
            "total": n}
    if missing == "listwise":
        keep = ~anymiss & wok
    else:
        keep = wok.copy()
        if missing == "passive":
            keep &= ~np.isnan(M).all(axis=1)
    info["used"] = int(keep.sum())
    out = []
    for sp, ind, lab in zip(specs, inds, labs):
        x = ind[keep].copy()
        extra = False
        if missing == "mode" and np.isnan(x).any():
            ok = ~np.isnan(x)
            vals, cnt = np.unique(x[ok], return_counts=True)
            wc = np.array([w[keep][ok][x[ok] == v].sum() for v in vals])
            x[~ok] = vals[np.argmax(wc)] if len(vals) else 1
            del cnt
        elif missing == "extra":
            extra = True
        cv = E.make_var(sp.name, x, sp.level, sp.degree, sp.knots, sp.weight, lab,
                        extra_missing=extra)
        out.append(cv)
    return out, keep, w[keep], info


def _case_table(info: dict) -> Table:
    rows = [["Valid Active Cases", fmt_int(info["valid"])],
            ["Active Cases with Missing Values", fmt_int(info["with_missing"])],
            ["Supplementary Cases", "0"],
            ["Total", fmt_int(info["total"])],
            ["Cases Used in Analysis", fmt_int(info["used"])]]
    return Table("Case Processing Summary", ["", "Count"], rows)


def _freq(v: E.CatVar, w: np.ndarray) -> np.ndarray:
    ok = v.codes >= 0
    return np.bincount(v.codes[ok], w[ok], minlength=v.k)


def _transform_plot(name: str, v: E.CatVar, q: np.ndarray, level: str) -> bytes:
    fig = figure(5.2, 3.8)
    ax = fig.add_subplot(111)
    x = np.arange(1, v.k + 1)
    ax.plot(x, q, "-o", color=COLORS[0], ms=4)
    ax.set_xticks(x)
    ax.set_xticklabels([str(lab) for lab in v.labels], rotation=45 if v.k > 6 else 0,
                       fontsize=8)
    ax.set_xlabel("Categories")
    ax.set_ylabel("Quantification")
    ax.set_title(f"{name}\nOptimal Scaling Level: {LEVELS[level]}", fontsize=10)
    ax.axhline(0, color="#999999", lw=0.6)
    return png(fig)


def _scatter(points: list[tuple[np.ndarray, list, str, str]], title: str,
             vectors: list | None = None, xlabel="Dimension 1", ylabel="Dimension 2") -> bytes:
    fig = figure(6.0, 5.0)
    ax = fig.add_subplot(111)
    for k, (xy, labels, name, marker) in enumerate(points):
        col = COLORS[k % len(COLORS)]
        ax.scatter(xy[:, 0], xy[:, 1], s=14, color=col, marker=marker, label=name)
        if labels is not None and len(labels) <= 120:
            for (x, y), t in zip(xy[:, :2], labels):
                ax.annotate(str(t), (x, y), fontsize=7, color=col, xytext=(2, 2),
                            textcoords="offset points")
    for vec, lab in vectors or []:
        ax.annotate("", xy=(vec[0], vec[1]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="#333333", lw=1))
        ax.annotate(lab, (vec[0], vec[1]), fontsize=8, color="#333333")
    ax.axhline(0, color="#999999", lw=0.6)
    ax.axvline(0, color="#999999", lw=0.6)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    if len(points) > 1:
        ax.legend(fontsize=7)
    return png(fig)


# ===================================================================== CATREG
def catreg(ds: Dataset, dependent, independents: list, missing: str = "listwise",
           init: str = "numerical", maxiter: int = 100, eps: float = 1e-5,
           regularization: str | None = None, ridge: tuple = (0.0, 1.0, 0.02),
           lasso: tuple = (0.0, 1.0, 0.02), resampling: str | None = None, folds: int = 10,
           bootstrap_se: int = 1000, seed: int = 2000000, show_history: bool = False,
           correlations: bool = False, quantifications: bool = True,
           anova: bool = True, transformation_plots: tuple = (),
           save_predicted: bool = False, save_residuals: bool = False,
           save_transformed: bool = False, weight: str | None = None) -> Result:
    """Categorical regression (CATREG). ``dependent``/``independents`` are VarSpecs (or names
    / dicts). ``regularization``: None, 'ridge', 'lasso' or 'enet' with the SPSS grids
    (start, stop, increment); ``resampling``: None or 'cv' (``folds``-fold cross-validation
    of the expected prediction error)."""
    dep = VarSpec.of(dependent)
    inds = [VarSpec.of(x) for x in independents]
    sec = Section("Categorical Regression")
    specs = [dep] + inds
    sec.log("CATREG VARIABLES=" + " ".join(s.name for s in specs) + "\n  /ANALYSIS=" +
            f"{dep.name}({dep.level.upper()}) WITH " +
            " ".join(f"{s.name}({s.level.upper()})" for s in inds) +
            f"\n  /MISSING={missing.upper()}\n  /INITIAL={init.upper()}\n"
            f"  /MAXITER={maxiter} /CRITITER={eps:g}" +
            (f"\n  /REGULARIZATION={regularization.upper()}" if regularization else "") +
            (f"\n  /RESAMPLE=CROSSVALIDATION({folds})" if resampling == "cv" else ""))
    vars_, keep, w, info = prepare_vars(ds, specs, missing, weight)
    if info["used"] < len(inds) + 2:
        sec.note("Warning", "Too few valid cases for the analysis.", "warning")
        return Result(sec)
    resp, preds = vars_[0], vars_[1:]
    for v in vars_:
        if (v.codes >= 0).sum() and len(np.unique(v.codes)) < 2:
            sec.note("Warning", f"Variable {v.name} has fewer than two categories.", "warning")
            return Result(sec)
    sec.table(_case_table(info))
    fit = E.catreg_fit(resp, preds, w, maxiter=maxiter, eps=eps, init=init)
    nw = w.sum()
    selected = fit
    reg_rows, path = [], []
    if regularization:
        grid = _penalty_grid(regularization, ridge, lasso)
        b_ols = fit.b
        prev = fit
        best = None
        for lam1, lam2 in grid:
            f2 = E.catreg_fit(resp, preds, w, lam1=lam1, lam2=lam2, maxiter=maxiter, eps=eps,
                              start=prev, signs=fit.signs)
            prev = f2
            epe = se = float("nan")
            if resampling == "cv":
                epe, se = _cv_epe(resp, preds, w, folds, seed, lam1, lam2, maxiter, eps, fit)
            if regularization == "ridge":
                ssc = np.sum(f2.b ** 2) / max(np.sum(b_ols ** 2), 1e-300)
            else:
                ssc = np.sum(np.abs(f2.b)) / max(np.sum(np.abs(b_ols)), 1e-300)
            reg_rows.append((lam1, lam2, f2.ape, epe, se, ssc, f2.b.copy(), f2))
            path.append((lam1, lam2, ssc, f2.b.copy()))
            key = epe if resampling == "cv" else None
            if key is not None and (best is None or key < best[0]):
                best = (key, f2)
        selected = best[1] if best else reg_rows[0][-1]
    # ----- Model Summary
    pdf = [E.spline_df(v, y, np.bincount(v.codes, w, minlength=v.k))
           for v, y in zip(preds, selected.yp)]
    r2 = selected.r2
    sf = sum(pdf)
    adj = 1 - (1 - r2) * (nw - 1) / (nw - 1 - sf) if nw - 1 - sf > 0 else float("nan")
    foot = [f"Dependent Variable: {dep.name}",
            "Predictors: " + " ".join(s.name for s in inds)]
    rows = [[_f(math.sqrt(max(r2, 0))), _f(r2), _f(adj), _f(selected.ape)]]
    cols = ["Multiple R", "R Square", "Adjusted R Square", "Apparent Prediction Error"]
    if regularization:
        cols = ["Multiple R", "R Square", "Regularized R Square (1-Error)",
                "Apparent Prediction Error"]
        rows = [[_f(math.sqrt(max(r2, 0))), _f(r2), _f(1 - selected.ape), _f(selected.ape)]]
    if resampling == "cv":
        epe, se = _cv_epe(resp, preds, w, folds, seed, *(
            _lams_of(selected, reg_rows) if regularization else (0.0, 0.0)), maxiter, eps, fit)
        cols += ["Expected Prediction Error", "Std. Error"]
        rows[0] += [_f(epe), _f(se)]
    sec.table(Table("Model Summary", cols, rows, row_header_cols=0,
                    footnotes=[f"{chr(97 + i)}. {t}" for i, t in enumerate(foot)]))
    if show_history:
        hrows = [[str(it), _f(1 - ape, 5), _f(ape, 5), _f(dec, 5) if it > 1 else ""]
                 for it, r2_, ape, dec in fit.history]
        sec.table(Table("Iteration History", ["Iteration Number", "Multiple R Square",
                                              "Apparent Prediction Error", "Restored Decrease"],
                        hrows, footnotes=[f"Convergence criterion {eps:g}; "
                                          f"{'converged' if fit.converged else 'maximum iterations reached'}."]))
    if anova:
        dfr = sf
        dfe = nw - 1 - sf
        ssr, sse = nw * r2, nw * (1 - r2)
        F = (ssr / dfr) / (sse / dfe) if dfe > 0 and dfr > 0 and sse > 0 else float("nan")
        sec.table(Table("ANOVA", ["", "Sum of Squares", "df", "Mean Square", "F", "Sig."], [
            ["Regression", _f(ssr), fmt_int(dfr), _f(ssr / dfr), _f(F),
             fmt_p(stats.f.sf(F, dfr, dfe)) if np.isfinite(F) else ""],
            ["Residual", _f(sse), fmt_int(dfe), _f(sse / dfe) if dfe > 0 else "", "", ""],
            ["Total", _f(nw), fmt_int(nw - 1), "", "", ""]],
            footnotes=[f"Dependent Variable: {dep.name}"]))
    # ----- Coefficients
    se_b = None
    if bootstrap_se and bootstrap_se > 1:
        se_b = _bootstrap_se(resp, preds, w, selected, bootstrap_se, seed, eps,
                             _lams_of(selected, reg_rows) if regularization else (0.0, 0.0))
    dfe = nw - 1 - sf
    crow = []
    for j, s in enumerate(inds):
        b = selected.b[j]
        row = [s.name, _f(b)]
        if se_b is not None:
            F = (b / se_b[j]) ** 2 if se_b[j] > 0 else float("nan")
            row += [_f(se_b[j]), fmt_int(pdf[j]), _f(F),
                    fmt_p(stats.f.sf(F, pdf[j], max(dfe, 1))) if np.isfinite(F) else ""]
        else:
            row += [fmt_int(pdf[j])]
        crow.append(row)
    ccols = ["", ("Standardized Coefficients", "Beta")]
    if se_b is not None:
        ccols += [("Standardized Coefficients", f"Bootstrap ({bootstrap_se}) Estimate of "
                   "Std. Error"), "df", "F", "Sig."]
    else:
        ccols += ["df"]
    sec.table(Table("Coefficients", ccols, crow, footnotes=[f"Dependent Variable: {dep.name}"]))
    # ----- Correlations and Tolerance
    zr = selected.zr
    Z = selected.Z
    r0 = np.array([np.sum(w * zr * Z[:, j]) / nw for j in range(len(preds))])
    R = (Z * w[:, None]).T @ Z / nw
    active = np.abs(selected.b) > 0
    tol = np.full(len(preds), np.nan)
    if active.any():
        Ra = R[np.ix_(active, active)]
        tol[active] = 1 / np.diag(np.linalg.pinv(Ra))
    beta_ols = selected.b if not regularization else _ols_active(R, r0, active)
    part = beta_ols * np.sqrt(np.where(np.isnan(tol), 0, tol))
    r2o = float(beta_ols @ r0) if regularization else r2
    partial = part / np.sqrt(np.maximum(1 - r2o + part ** 2, 1e-300))
    imp = selected.b * r0 / r2 if r2 > 0 else np.full(len(preds), np.nan)
    Xorig = np.column_stack([v.values[v.codes] for v in preds])
    tol0 = _tolerance(Xorig, w)
    trows = []
    for j, s in enumerate(inds):
        trows.append([s.name, _f(r0[j]), _f(partial[j]), _f(part[j]),
                      "" if regularization else _f(imp[j]), _f(tol[j]), _f(tol0[j])])
    sec.table(Table("Correlations and Tolerance", [
        "", ("Correlations", "Zero-Order"), ("Correlations", "Partial"),
        ("Correlations", "Part"), "Importance", ("Tolerance", "After Transformation"),
        ("Tolerance", "Before Transformation")], trows,
        footnotes=[f"Dependent Variable: {dep.name}"]))
    if correlations:
        Xt = np.column_stack([zr, Z])
        Rt = (Xt * w[:, None]).T @ Xt / nw
        names = [s.name for s in specs]
        Rt_ = np.corrcoef(Xorig.T) if len(preds) > 1 else np.ones((1, 1))
        sec.table(Table("Correlations Original Variables",
                        [""] + [s.name for s in inds],
                        [[s.name] + [_f(x) for x in Rt_[i]] for i, s in enumerate(inds)]))
        sec.table(Table("Correlations Transformed Variables", [""] + names,
                        [[nm] + [_f(x) for x in Rt[i]] for i, nm in enumerate(names)]))
    if regularization:
        rrows = []
        for lam1, lam2, ape, epe, se, ssc, b, _f2 in reg_rows:
            rows_ = []
            if regularization in ("ridge", "enet"):
                rows_.append(_f(lam2, 2))
            if regularization in ("lasso", "enet"):
                rows_.append(_f(lam1, 2))
            rows_ += [_f(ape), _f(epe) if np.isfinite(epe) else "",
                      _f(se) if np.isfinite(se) else "", _f(ssc)] + [_f(x) for x in b]
            rrows.append(rows_)
        pcols = (["Ridge Penalty"] if regularization in ("ridge", "enet") else []) + \
            (["Lasso Penalty"] if regularization in ("lasso", "enet") else [])
        sec.table(Table(f"{ {'ridge': 'Ridge', 'lasso': 'Lasso', 'enet': 'Elastic Net'}[regularization]} "
                        "Regression", pcols + ["Apparent Prediction Error",
                                               "Expected Prediction Error", "Std. Error",
                                               "Standardized Sum of Coefficients"] +
                        [("Standardized Coefficients", s.name) for s in inds], rrows,
                        row_header_cols=0,
                        footnotes=["Results for the selected model (lowest expected prediction "
                                   "error)." if resampling == "cv" else
                                   "Without resampling the first model of the grid is "
                                   "reported in the other tables."]))
        sec.image("Regularization Paths", _paths_plot(path, [s.name for s in inds],
                                                       regularization))
    if quantifications:
        for k, v in enumerate(vars_):
            y = selected.yr if k == 0 else selected.yp[k - 1]
            fr = _freq(v, w)
            qrows = [[str(lab), fmt_int(fr[c]), _f(y[c])] for c, lab in enumerate(v.labels)]
            sec.table(Table(v.name, ["Category", "Frequency", "Quantification"], qrows,
                            footnotes=[f"Optimal Scaling Level: {level_text(specs[k])}."]))
    for k, v in enumerate(vars_):
        if v.name in transformation_plots:
            y = selected.yr if k == 0 else selected.yp[k - 1]
            sec.image(f"Transformation: {v.name}", _transform_plot(v.name, v, y, v.level))
    saved = {}
    idx = ds.df.index
    if save_predicted or save_residuals or save_transformed:
        pred = Z @ selected.b
        if save_predicted:
            s = pd.Series(np.nan, index=idx)
            s[keep] = pred
            saved["PRE"] = ("Predicted value", s, 5)
        if save_residuals:
            s = pd.Series(np.nan, index=idx)
            s[keep] = zr - pred
            saved["RES"] = ("Residual", s, 5)
        if save_transformed:
            for k, v in enumerate(vars_):
                col = zr if k == 0 else Z[:, k - 1]
                s = pd.Series(np.nan, index=idx)
                s[keep] = col
                saved[f"TRA{k + 1}"] = (f"Transformed {v.name}", s, 5)
    return Result(sec, saved)


def _lams_of(sel, reg_rows):
    for lam1, lam2, *_rest, f2 in reg_rows:
        if f2 is sel:
            return lam1, lam2
    return 0.0, 0.0


def _ols_active(R, r, active):
    b = np.zeros(len(r))
    if active.any():
        b[active] = np.linalg.lstsq(R[np.ix_(active, active)], r[active], rcond=None)[0]
    return b


def _tolerance(X, w):
    if X.shape[1] < 2:
        return np.ones(X.shape[1])
    m = (w @ X) / w.sum()
    Xc = X - m
    C = (Xc * w[:, None]).T @ Xc
    sd = np.sqrt(np.diag(C))
    sd[sd == 0] = 1
    R = C / np.outer(sd, sd)
    return 1 / np.diag(np.linalg.pinv(R))


def _penalty_grid(kind, ridge, lasso):
    def seq(t):
        lo, hi, step = t
        n = int(round((hi - lo) / step)) if step > 0 else 0
        return [round(lo + i * step, 10) for i in range(n + 1)]
    if kind == "ridge":
        return [(0.0, l2) for l2 in seq(ridge)]
    if kind == "lasso":
        return [(l1, 0.0) for l1 in seq(lasso)]
    return [(l1, l2) for l2 in seq(ridge) for l1 in seq(lasso)]


def _subset(v: E.CatVar, rows: np.ndarray) -> E.CatVar:
    return E.CatVar(v.name, v.codes[rows], v.values, v.labels, v.level, v.degree, v.knots,
                    v.weight, v.free, v.basis)


def _cv_epe(resp, preds, w, folds, seed, lam1, lam2, maxiter, eps, start):
    n = len(w)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    parts = np.array_split(perm, min(folds, n))
    err = np.full(n, np.nan)
    for test in parts:
        train = np.setdiff1d(np.arange(n), test)
        rs, ps = _subset(resp, train), [_subset(p, train) for p in preds]
        try:
            f = E.catreg_fit(rs, ps, w[train], lam1, lam2, maxiter, eps, "numerical")
        except Exception:  # noqa: BLE001
            continue
        zr = _supp_quant(rs, f.yr, resp.codes[test], w[train])
        pred = np.zeros(len(test))
        for j, p in enumerate(ps):
            pred += f.b[j] * _supp_quant(p, f.yp[j], preds[j].codes[test], w[train])
        err[test] = (zr - pred) ** 2
    ok = ~np.isnan(err)
    if not ok.any():
        return float("nan"), float("nan")
    ww = w[ok]
    epe = float(np.sum(ww * err[ok]) / ww.sum())
    se = float(math.sqrt(np.sum(ww * (err[ok] - epe) ** 2)) / ww.sum())
    return epe, se


def _supp_quant(train_var: E.CatVar, y: np.ndarray, codes: np.ndarray, wtrain) -> np.ndarray:
    """Quantify (test) categories with training quantifications: categories absent in the
    training part are interpolated for numeric/spline levels inside the range (extrapolated for
    numeric), else missing (IBM CATREG "Supplementary objects")."""
    d = np.bincount(train_var.codes, wtrain, minlength=train_var.k)
    present = d > 0
    out = np.full(len(codes), np.nan)
    vals = train_var.values
    for i, c in enumerate(codes):
        if c < 0:
            continue
        if present[c]:
            out[i] = y[c]
        elif train_var.level in ("nume", "spord", "spnom"):
            xv, yv = vals[present], y[present]
            if xv.min() <= vals[c] <= xv.max():
                out[i] = np.interp(vals[c], xv, yv)
            elif train_var.level == "nume" and len(xv) > 1:
                b = np.polyfit(xv, yv, 1)
                out[i] = np.polyval(b, vals[c])
    return out


def _bootstrap_se(resp, preds, w, fit, B, seed, eps, lams):
    rng = np.random.default_rng(seed)
    n = len(w)
    bs = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        rs, ps = _subset(resp, idx), [_subset(p, idx) for p in preds]
        if any(np.bincount(v.codes, minlength=v.k).astype(bool).sum() < 2 for v in [rs] + ps):
            continue
        start = E.CatregFit(fit.yr, fit.yp, fit.b.copy(), fit.yr[rs.codes],
                            np.column_stack([fit.yp[j][p.codes] for j, p in enumerate(ps)]),
                            fit.ape, fit.r2, signs=fit.signs)
        try:
            f = E.catreg_fit(rs, ps, w[idx], lams[0], lams[1], 50, max(eps, 1e-4),
                             start=start, signs=fit.signs)
        except Exception:  # noqa: BLE001
            continue
        bs.append(f.b)
    if len(bs) < 2:
        return None
    return np.std(np.array(bs), axis=0, ddof=1)


def _paths_plot(path, names, kind) -> bytes:
    fig = figure(6.0, 4.2)
    ax = fig.add_subplot(111)
    groups = {}
    for lam1, lam2, ssc, b in path:
        groups.setdefault(lam2 if kind == "enet" else 0, []).append((ssc, b))
    for g, items in groups.items():
        items.sort(key=lambda t: t[0])
        x = np.array([t[0] for t in items])
        B = np.array([t[1] for t in items])
        for j, nm in enumerate(names):
            ax.plot(x, B[:, j], color=COLORS[j % len(COLORS)],
                    label=nm if g == next(iter(groups)) else None,
                    alpha=1.0 if len(groups) == 1 else 0.6)
    ax.axhline(0, color="#999999", lw=0.6)
    ax.set_xlabel("Standardized sum of coefficients")
    ax.set_ylabel("Standardized coefficients")
    ax.set_title({"ridge": "Ridge", "lasso": "Lasso", "enet": "Elastic Net (Lasso paths)"}[kind]
                 + " Paths", fontsize=10)
    ax.legend(fontsize=7)
    return png(fig)


# ===================================================================== CATPCA / MCA
def _normalize_scores(fit: E.PCAFit, norm: str, q: float = 0.0):
    """Object scores and loadings for the requested normalization (variable principal is the
    optimization normalization: scores X, loadings = correlations)."""
    lam = np.sqrt(np.maximum(fit.eig, 1e-300))
    if norm == "vprincipal":
        a, b = 0.0, 1.0
    elif norm == "oprincipal":
        a, b = 1.0, 0.0
    elif norm == "symmetrical":
        a, b = 0.5, 0.5
    elif norm == "independent":
        a, b = 1.0, 1.0
    else:  # custom q
        a, b = (1 + q) / 2, (1 - q) / 2
    X = fit.X * lam ** a
    loads = [None if L is None else L / lam * lam ** b for L in fit.loadings]
    cents = [c * lam ** a for c in fit.centroids]
    return X, loads, cents


def catpca(ds: Dataset, variables: list, ndim: int = 2, missing: str = "passive",
           normalization: str = "vprincipal", q: float = 0.0, maxiter: int = 100,
           eps: float = 1e-5, show_history: bool = False, descriptives: bool = False,
           vaf: bool = True, loadings: bool = True, quantifications: tuple = (),
           correlations_transformed: bool = False, object_scores: bool = False,
           plots: tuple = ("objects", "loadings"), transformation_plots: tuple = (),
           labels: str | None = None, save_scores: bool = False,
           save_transformed: bool = False, title: str = "Categorical Principal Components "
           "Analysis", mca_mode: bool = False) -> Result:
    specs = [VarSpec.of(x) for x in variables]
    if mca_mode:
        for s in specs:
            s.level = "mnom"
    sec = Section("Multiple Correspondence Analysis" if mca_mode else title)
    sec.log(("MULTIPLE CORRESPONDENCE" if mca_mode else "CATPCA") + " VARIABLES=" +
            " ".join(s.name for s in specs) + "\n  /ANALYSIS=" +
            " ".join(f"{s.name}(WEIGHT={s.weight:g},LEVEL={s.level.upper()})" for s in specs) +
            f"\n  /MISSING={missing.upper()}\n  /DIMENSION={ndim}\n"
            f"  /NORMALIZATION={normalization.upper()}\n  /MAXITER={maxiter} /CRITITER={eps:g}")
    vars_, keep, w, info = prepare_vars(ds, specs, missing)
    m = len(vars_)
    kmax = sum(v.k - 1 if v.multiple else 1 for v in vars_)
    if info["used"] < 3 or ndim > min(kmax, info["used"] - 1):
        sec.note("Warning", f"Too few cases or categories for {ndim} dimensions "
                 f"(maximum rank {min(kmax, info['used'] - 1)}).", "warning")
        return Result(sec)
    sec.table(_case_table(info))
    fit = E.gifi_pca(vars_, w, ndim, maxiter, eps)
    nw = w.sum()
    if descriptives:
        for v in vars_:
            fr = _freq(v, w)
            sec.table(Table(v.name, ["", "Frequency"],
                            [[str(lab), fmt_int(fr[c])] for c, lab in enumerate(v.labels)] +
                            [["Missing", fmt_int(info["total"] - fr.sum())],
                             ["Total", fmt_int(info["total"])]]))
    if show_history:
        hrows = [[str(it), _f(tf, 5), _f(m - tf / 1, 5) if not mca_mode else _f(ndim - tf / m, 5)]
                 for it, tf, _d in fit.history]
        sec.table(Table("Iteration History", ["Iteration Number",
                                              "Variance Accounted For: Total", "Loss"], hrows))
    vw = np.array([s.weight for s in specs])
    msum = vw.sum()
    alpha = [m * (lam - 1) / ((m - 1) * lam) if m > 1 and lam > 0 else float("nan")
             for lam in fit.eig]
    tot = fit.eig.sum()
    alpha_t = m * (tot - 1) / ((m - 1) * tot) if m > 1 and tot > 0 else float("nan")
    if mca_mode:
        rows = [[str(s + 1), _f(alpha[s]), _f(fit.eig[s]), _f(fit.eig[s] / m),
                 _f(fit.eig[s] / m * 100, 3)] for s in range(ndim)]
        mean = fit.eig.mean()
        alpha_m = m * (mean - 1) / ((m - 1) * mean) if m > 1 and mean > 0 else float("nan")
        rows.append(["Total", "", _f(tot), _f(tot / m), ""])
        rows.append(["Mean", _f(alpha_m), _f(mean), _f(mean / m), _f(mean / m * 100, 3)])
        sec.table(Table("Model Summary", ["Dimension", "Cronbach's Alpha",
                                          ("Variance Accounted For", "Total (Eigenvalue)"),
                                          ("Variance Accounted For", "Inertia"),
                                          ("Variance Accounted For", "% of Variance")], rows,
                        footnotes=["a. Mean Cronbach's Alpha is based on the mean Eigenvalue."]))
    else:
        rows = [[str(s + 1), _f(alpha[s]), _f(fit.eig[s]), _f(fit.eig[s] / msum * 100, 3)]
                for s in range(ndim)]
        rows.append(["Total", _f(alpha_t), _f(tot), _f(tot / msum * 100, 3)])
        sec.table(Table("Model Summary", ["Dimension", "Cronbach's Alpha",
                                          ("Variance Accounted For", "Total (Eigenvalue)"),
                                          ("Variance Accounted For", "% of Variance")], rows,
                        footnotes=["a. Total Cronbach's Alpha is based on the total "
                                   "Eigenvalue."]))
    X, loads, cents = _normalize_scores(fit, normalization, q)
    D = [np.bincount(v.codes[v.codes >= 0], w[v.codes >= 0], minlength=v.k) for v in vars_]
    if mca_mode:
        disc = np.array([np.einsum("k,ks,ks->s", D[j], fit.centroids[j], fit.centroids[j]) / nw
                         for j in range(m)])
        drows = [[v.name] + [_f(x) for x in disc[j]] + [_f(disc[j].mean())]
                 for j, v in enumerate(vars_)]
        drows.append(["Active Total"] + [_f(x) for x in disc.sum(axis=0)] +
                     [_f(disc.sum(axis=0).mean())])
        drows.append(["% of Variance"] + [_f(x / m * 100, 3) for x in disc.sum(axis=0)] +
                     [_f(disc.sum(axis=0).mean() / m * 100, 3)])
        sec.table(Table("Discrimination Measures", ["", *[("Dimension", str(s + 1))
                                                          for s in range(ndim)], "Mean"],
                        drows))
    elif vaf:
        vrows = []
        for j, v in enumerate(vars_):
            c = np.einsum("k,ks,ks->s", D[j], fit.centroids[j], fit.centroids[j]) / nw * vw[j]
            row = [v.name] + [_f(x) for x in c] + [_f(c.mean())]
            if fit.loadings[j] is not None:
                vec = fit.loadings[j] ** 2 * vw[j]
                row += [_f(x) for x in vec] + [_f(vec.sum())]
            else:
                row += [""] * (ndim + 1)
            vrows.append(row)
        cs = np.array([np.einsum("k,ks,ks->s", D[j], fit.centroids[j], fit.centroids[j]) /
                       nw * vw[j] for j in range(m)])
        vrows.append(["Active Total"] + [_f(x) for x in cs.sum(axis=0)] +
                     [_f(cs.sum(axis=0).mean())] + [_f(x) for x in fit.eig] + [_f(tot)])
        sec.table(Table("Variance Accounted For",
                        [""] + [("Centroid Coordinates: Dimension", str(s + 1))
                                for s in range(ndim)] + ["Mean"] +
                        [("Total (Vector Coordinates): Dimension", str(s + 1))
                         for s in range(ndim)] + ["Total"], vrows))
    if correlations_transformed:
        Q = np.column_stack([(fit.quant[j] if fit.quant[j] is not None else
                              fit.centroids[j][:, 0])[np.maximum(v.codes, 0)] *
                             (v.codes >= 0) for j, v in enumerate(vars_)])
        Qc = Q - (w @ Q) / nw
        C = (Qc * w[:, None]).T @ Qc
        sd = np.sqrt(np.diag(C))
        Rq = C / np.outer(sd, sd)
        ev = np.sort(np.linalg.eigvalsh(Rq))[::-1]
        rows = [[v.name] + [_f(x) for x in Rq[i]] for i, v in enumerate(vars_)]
        rows.append(["Eigenvalue"] + [_f(x) for x in ev])
        sec.table(Table("Correlations Transformed Variables", [""] + [v.name for v in vars_],
                        rows, footnotes=["Missing values imputed with the quantified mode."]))
    if loadings and not mca_mode and any(L is not None for L in loads):
        lrows = [[v.name] + [_f(x) for x in loads[j]] for j, v in enumerate(vars_)
                 if loads[j] is not None]
        sec.table(Table("Component Loadings", [""] + [("Dimension", str(s + 1))
                                                      for s in range(ndim)], lrows,
                        footnotes=[f"{NORMALIZATIONS.get(normalization, 'Custom')} "
                                   "Normalization."]))
    for j, v in enumerate(vars_):
        if v.name not in quantifications and not (mca_mode and quantifications == ("*",)):
            continue
        fr = _freq(v, w)
        if v.multiple:
            qrows = [[str(lab), fmt_int(fr[c])] + [_f(x) for x in cents[j][c]]
                     for c, lab in enumerate(v.labels)]
            sec.table(Table(v.name if not mca_mode else f"{v.name}: Points: Coordinates",
                            ["Category", "Frequency"] +
                            [("Centroid Coordinates: Dimension", str(s + 1))
                             for s in range(ndim)], qrows,
                            footnotes=[f"Optimal Scaling Level: Multiple Nominal."]))
        else:
            vec = np.outer(fit.quant[j], loads[j]) if loads[j] is not None else None
            qrows = [[str(lab), fmt_int(fr[c]), _f(fit.quant[j][c])] +
                     [_f(x) for x in cents[j][c]] + [_f(x) for x in vec[c]]
                     for c, lab in enumerate(v.labels)]
            sec.table(Table(v.name, ["Category", "Frequency", "Quantification"] +
                            [("Centroid Coordinates: Dimension", str(s + 1))
                             for s in range(ndim)] +
                            [("Vector Coordinates: Dimension", str(s + 1))
                             for s in range(ndim)], qrows,
                            footnotes=[f"Optimal Scaling Level: {level_text(specs[j])}."]))
    case_labels = None
    if labels:
        case_labels = [str(x) for x in ds.analysis_series(labels)[keep]]
    elif X.shape[0] <= 60:
        case_labels = [str(i + 1) for i in np.flatnonzero(keep)]
    if object_scores:
        orows = [[(case_labels[i] if case_labels else str(i + 1))] +
                 [_f(x) for x in X[i]] for i in range(min(X.shape[0], 500))]
        sec.table(Table("Object Scores", ["Case"] + [("Dimension", str(s + 1))
                                                     for s in range(ndim)], orows,
                        footnotes=[f"{NORMALIZATIONS.get(normalization, 'Custom')} "
                                   "Normalization."]))
    if ndim >= 2:
        norm_t = f"{NORMALIZATIONS.get(normalization, 'Custom')} Normalization."
        if "objects" in plots:
            sec.image("Object Points", _scatter([(X, case_labels, "Objects", "o")],
                                                f"Object Points\n{norm_t}"))
        if "loadings" in plots and not mca_mode:
            vecs = [(loads[j][:2], v.name) for j, v in enumerate(vars_)
                    if loads[j] is not None]
            if vecs:
                sec.image("Component Loadings", _scatter([(np.zeros((0, 2)), [], "", "o")],
                                                         f"Component Loadings\n{norm_t}",
                                                         vecs))
        if "biplot" in plots and not mca_mode:
            vecs = [(loads[j][:2], v.name) for j, v in enumerate(vars_)
                    if loads[j] is not None]
            sf = _biplot_scale(X, vecs)
            sec.image("Biplot", _scatter([(X * sf, case_labels, "Objects", "o")],
                                         f"Biplot Objects and Loadings\n{norm_t}", vecs))
        if "joint" in plots or (mca_mode and "categories" in plots):
            pts = [(cents[j][:, :2], v.labels, v.name, "^") for j, v in enumerate(vars_)]
            sec.image("Joint Plot of Category Points",
                      _scatter(pts, f"Joint Plot of Category Points\n{norm_t}"))
        if mca_mode and "discrimination" in plots:
            disc = np.array([np.einsum("k,ks,ks->s", D[j], fit.centroids[j],
                                       fit.centroids[j]) / nw for j in range(m)])
            vecs = [(disc[j, :2], v.name) for j, v in enumerate(vars_)]
            sec.image("Discrimination Measures",
                      _scatter([(np.zeros((0, 2)), [], "", "o")], "Discrimination Measures",
                               vecs))
    for j, v in enumerate(vars_):
        if v.name in transformation_plots:
            qv = fit.quant[j] if fit.quant[j] is not None else fit.centroids[j][:, 0]
            sec.image(f"Transformation: {v.name}", _transform_plot(v.name, v, qv, v.level))
    saved = {}
    idx = ds.df.index
    if save_scores:
        for s in range(ndim):
            ser = pd.Series(np.nan, index=idx)
            ser[keep] = X[:, s]
            saved[f"OBSCO{s + 1}"] = (f"Object scores dimension {s + 1}", ser, 5)
    if save_transformed and not mca_mode:
        for j, v in enumerate(vars_):
            if fit.quant[j] is None:
                continue
            ser = pd.Series(np.nan, index=idx)
            vals = np.where(v.codes >= 0, fit.quant[j][np.maximum(v.codes, 0)], np.nan)
            ser[keep] = vals
            saved[f"TRA{j + 1}"] = (f"Transformed {v.name}", ser, 5)
    return Result(sec, saved)


def _biplot_scale(X, vecs):
    if not vecs:
        return 1.0
    lx = np.sqrt((X[:, :2] ** 2).sum(axis=1)).max()
    lv = max(np.sqrt((v ** 2).sum()) for v, _ in vecs)
    return lv / lx if lx > 0 else 1.0


def mca(ds: Dataset, variables: list, ndim: int = 2, missing: str = "passive",
        normalization: str = "vprincipal", maxiter: int = 100, eps: float = 1e-5,
        discrimination: bool = True, category_points: tuple = ("*",),
        object_scores: bool = False, show_history: bool = False,
        plots: tuple = ("objects", "categories", "discrimination"),
        labels: str | None = None, save_scores: bool = False) -> Result:
    """Multiple correspondence analysis (homogeneity analysis): CATPCA with every variable
    multiple nominal; output as SPSS MULTIPLE CORRESPONDENCE prints it."""
    return catpca(ds, variables, ndim, missing, normalization, 0.0, maxiter, eps,
                  show_history, False, False, False, category_points, False, object_scores,
                  plots, (), labels, save_scores, False, mca_mode=True)


# ===================================================================== OVERALS
def overals(ds: Dataset, sets: list[list], ndim: int = 2, maxiter: int = 100,
            eps: float = 1e-5, show_history: bool = False, frequencies: bool = False,
            centroids: bool = False, weights_loadings: bool = True,
            category_quantifications: bool = True, object_scores: bool = False,
            plots: tuple = ("objects", "loadings"), labels: str | None = None,
            save_scores: bool = False) -> Result:
    """Nonlinear canonical correlation analysis. ``sets`` is a list of variable lists (each a
    VarSpec / dict / name); levels: 'ordi', 'nomi' (single nominal), 'mnom', 'nume'."""
    sets_s = [[VarSpec.of(x) for x in st] for st in sets]
    for st in sets_s:
        for s in st:
            if s.level in ("spord", "spnom"):
                s.level = "ordi" if s.level == "spord" else "nomi"
    sec = Section("Nonlinear Canonical Correlation Analysis")
    sec.log("OVERALS VARIABLES=" + " ".join(s.name for st in sets_s for s in st) +
            "\n  /ANALYSIS=" + " ".join(f"{s.name}({s.level.upper()})"
                                       for st in sets_s for s in st) +
            "\n  /SETS=" + f"{len(sets_s)}(" + ",".join(str(len(st)) for st in sets_s) + ")" +
            f"\n  /DIMENSION={ndim}\n  /MAXITER={maxiter} /CONVERGENCE={eps:g}")
    if len(sets_s) < 2:
        sec.note("Warning", "OVERALS needs at least two sets of variables.", "warning")
        return Result(sec)
    flat = [s for st in sets_s for s in st]
    vars_, keep, w, info = prepare_vars(ds, flat, "listwise")
    sec.table(_case_table(info))
    pos = 0
    sets_v = []
    for st in sets_s:
        sets_v.append(vars_[pos:pos + len(st)])
        pos += len(st)
    fit = E.overals(sets_v, w, ndim, maxiter, eps)
    K = len(sets_v)
    nw = w.sum()
    if frequencies:
        for v in vars_:
            fr = _freq(v, w)
            sec.table(Table(v.name, ["", "Marginal Frequency"],
                            [[str(lab), fmt_int(fr[c])] for c, lab in enumerate(v.labels)]))
    if show_history:
        sec.table(Table("Iteration History", ["Iteration", "Loss", "Fit"],
                        [[str(it), _f(loss, 6), _f(ndim - loss, 6)]
                         for it, loss in fit.history]))
    rows = []
    for k in range(K):
        rows.append([f"Loss: Set {k + 1}"] + [_f(x) for x in fit.loss_set[k]] +
                    [_f(fit.loss_set[k].sum())])
    mean_loss = fit.loss_set.mean(axis=0)
    rows.append(["Loss: Mean"] + [_f(x) for x in mean_loss] + [_f(mean_loss.sum())])
    rows.append(["Eigenvalue"] + [_f(x) for x in fit.eig] + [""])
    rows.append(["Fit"] + [""] * ndim + [_f(fit.fit)])
    cancor = (K * fit.eig - 1) / (K - 1)
    rows.append(["Canonical Correlation"] + [_f(x) for x in cancor] + [""])
    sec.table(Table("Summary of Analysis", ["", *[("Dimension", str(s + 1))
                                                  for s in range(ndim)], "Sum"], rows,
                    footnotes=["Canonical correlation = (K*Eigenvalue - 1)/(K - 1), K = "
                               f"{K} sets."]))
    # Fit: multiple fit, single fit, single loss
    frows = []
    for k, vs in enumerate(sets_v):
        for j, v in enumerate(vs):
            Dk = np.bincount(v.codes, w, minlength=v.k)
            Ym = fit.multiple[k][j]
            multiple = np.einsum("k,ks,ks->s", Dk, Ym, Ym) / nw
            row = [f"Set {k + 1}", v.name] + [_f(x) for x in multiple] + [_f(multiple.sum())]
            if fit.weights[k][j] is not None:
                single = fit.weights[k][j] ** 2
                row += [_f(x) for x in single] + [_f(single.sum())]
                row += [_f(x) for x in multiple - single] + [_f((multiple - single).sum())]
            else:
                row += [""] * (2 * (ndim + 1))
            frows.append(row)
    sec.table(Table("Fit", ["Set", "Variable"] +
                    [("Multiple Fit", str(s + 1)) for s in range(ndim)] +
                    [("Multiple Fit", "Sum")] +
                    [("Single Fit", str(s + 1)) for s in range(ndim)] +
                    [("Single Fit", "Sum")] +
                    [("Single Loss", str(s + 1)) for s in range(ndim)] +
                    [("Single Loss", "Sum")], frows, row_header_cols=2))
    if weights_loadings:
        lrows, wrows = [], []
        for k, vs in enumerate(sets_v):
            for j, v in enumerate(vs):
                if fit.weights[k][j] is None:
                    continue
                wrows.append([f"Set {k + 1}", v.name] + [_f(x) for x in fit.weights[k][j]])
                q = fit.quant[k][j][v.codes]
                ld = (q * w) @ fit.X / nw
                lrows.append([f"Set {k + 1}", v.name] + [_f(x) for x in ld])
        if wrows:
            sec.table(Table("Weights", ["Set", "Variable"] + [("Dimension", str(s + 1))
                                                             for s in range(ndim)], wrows,
                            row_header_cols=2))
            sec.table(Table("Component Loadings", ["Set", "Variable"] +
                            [("Dimension", str(s + 1)) for s in range(ndim)], lrows,
                            row_header_cols=2))
    if category_quantifications:
        for k, vs in enumerate(sets_v):
            for j, v in enumerate(vs):
                fr = _freq(v, w)
                if v.multiple:
                    rows = [[str(lab), fmt_int(fr[c])] + [_f(x) for x in fit.Y[k][j][c]]
                            for c, lab in enumerate(v.labels)]
                    sec.table(Table(f"{v.name} (Set {k + 1})", ["Category", "Marginal Frequency"] +
                                    [("Multiple Category Coordinates: Dimension", str(s + 1))
                                     for s in range(ndim)], rows))
                else:
                    sc = fit.Y[k][j]
                    rows = [[str(lab), fmt_int(fr[c]), _f(fit.quant[k][j][c])] +
                            [_f(x) for x in sc[c]] for c, lab in enumerate(v.labels)]
                    sec.table(Table(f"{v.name} (Set {k + 1})",
                                    ["Category", "Marginal Frequency", "Quantification"] +
                                    [("Single Category Coordinates: Dimension", str(s + 1))
                                     for s in range(ndim)], rows,
                                    footnotes=[f"Optimal Scaling Level: {LEVELS[v.level]}."]))
    if centroids:
        for k, vs in enumerate(sets_v):
            for j, v in enumerate(vs):
                rows = [[str(lab)] + [_f(x) for x in fit.centroids[k][j][c]]
                        for c, lab in enumerate(v.labels)]
                sec.table(Table(f"Centroids: {v.name}", ["Category"] +
                                [("Dimension", str(s + 1)) for s in range(ndim)], rows))
    case_labels = [str(x) for x in ds.analysis_series(labels)[keep]] if labels else (
        [str(i + 1) for i in np.flatnonzero(keep)] if fit.X.shape[0] <= 60 else None)
    if object_scores:
        sec.table(Table("Object Scores", ["Case"] + [("Dimension", str(s + 1))
                                                     for s in range(ndim)],
                        [[case_labels[i] if case_labels else str(i + 1)] +
                         [_f(x) for x in fit.X[i]] for i in range(min(len(fit.X), 500))]))
    if ndim >= 2:
        if "objects" in plots:
            sec.image("Object Scores", _scatter([(fit.X, case_labels, "Objects", "o")],
                                                "Object Scores"))
        if "loadings" in plots:
            vecs = []
            for k, vs in enumerate(sets_v):
                for j, v in enumerate(vs):
                    if fit.quant[k][j] is not None:
                        ld = (fit.quant[k][j][v.codes] * w) @ fit.X / nw
                        vecs.append((ld[:2], v.name))
            if vecs:
                sec.image("Component Loadings", _scatter([(np.zeros((0, 2)), [], "", "o")],
                                                         "Component Loadings", vecs))
        if "centroids" in plots:
            pts = [(fit.centroids[k][j][:, :2], v.labels, v.name, "^")
                   for k, vs in enumerate(sets_v) for j, v in enumerate(vs)]
            sec.image("Centroids", _scatter(pts, "Centroids"))
    saved = {}
    if save_scores:
        for s in range(ndim):
            ser = pd.Series(np.nan, index=ds.df.index)
            ser[keep] = fit.X[:, s]
            saved[f"OVE{s + 1}"] = (f"Object scores dimension {s + 1}", ser, 5)
    return Result(sec, saved)


# ===================================================================== PREFSCAL
TRANSFORMS = {"none": "None (ratio)", "linear": "Linear", "spline": "Spline",
              "smooth": "Smooth", "ordinal": "Ordinal"}


def prefscal(ds: Dataset, variables: list[str], ndim: int = 2, similarities: bool = False,
             transform: str = "none", conditional: bool = True, strength: float = 0.5,
             penalty_range: float = 1.0, maxiter: int = 5000, eps: float = 1e-6,
             min_stress: float = 1e-4, row_labels: str | None = None,
             measures: bool = True, coordinates: bool = True, history: bool = False,
             plots: tuple = ("joint",)) -> Result:
    """Multidimensional unfolding of the n x m matrix formed by the cases (rows, e.g.
    respondents) and the selected variables (columns, e.g. products / stimuli)."""
    sec = Section("Multidimensional Unfolding")
    sec.log("PREFSCAL VARIABLES=" + " ".join(variables) +
            f"\n  /INPUT=ROWS(ALL)\n  /PROXIMITIES={'SIMILARITIES' if similarities else 'DISSIMILARITIES'}"
            f"\n  /TRANSFORMATION={transform.upper()}"
            f"({'ROWS' if conditional else 'UNCONDITIONAL'})"
            f"\n  /CRITERIA=DIMENSIONS({ndim}) DIFFSTRESS({eps:g}) MINSTRESS({min_stress:g}) "
            f"MAXITER({maxiter})\n  /PENALTY=LAMBDA({strength:g}) OMEGA({penalty_range:g})")
    M = np.column_stack([pd.to_numeric(ds.analysis_series(v), errors="coerce").to_numpy(float)
                         for v in variables])
    keep = ~np.isnan(M).all(axis=1)
    M = M[keep]
    W = (~np.isnan(M)).astype(float)
    if M.shape[0] < 2 or M.shape[1] < 3:
        sec.note("Warning", "PREFSCAL needs at least 2 rows and 3 column variables.", "warning")
        return Result(sec)
    fit = E.prefscal(M, ndim, transform, conditional, strength, penalty_range, similarities,
                     maxiter, eps, min_stress, W)
    rl = [str(x) for x in ds.analysis_series(row_labels)[keep]] if row_labels else \
        [str(i + 1) for i in np.flatnonzero(keep)]
    if history:
        sec.table(Table("Iteration History", ["Iteration", "Penalized Stress", "Stress Part",
                                              "Penalty Part"],
                        [[str(h[0])] + [_f(x, 6) for x in h[1:]] for h in fit.history[-50:]]))
    if measures:
        mm = E.unfolding_measures(fit, M if not similarities else
                                  np.where(W > 0, np.nanmax(M) + np.nanmin(M) - M, 0.0), W,
                                  conditional)
        rows = [["Iterations", "", str(fit.iterations)],
                ["Final Function Value", "", _f(fit.penalized, 7)],
                ["Function Value Parts", "Stress Part", _f(fit.stress_part, 7)],
                ["", "Penalty Part", _f(fit.penalty_part, 7)],
                ["Badness of Fit", "Normalized Stress", _f(mm["normalized"], 7)],
                ["", "Kruskal's Stress-I", _f(mm["stress1"], 7)],
                ["", "Kruskal's Stress-II", _f(mm["stress2"], 7)],
                ["", "Young's S-Stress-I", _f(mm["sstress1"], 7)],
                ["", "Young's S-Stress-II", _f(mm["sstress2"], 7)],
                ["Goodness of Fit", "Dispersion Accounted For", _f(mm["daf"], 7)],
                ["", "Variance Accounted For", _f(mm["vaf"], 7)],
                ["", "Recovered Preference Orders", _f(mm["rpo"], 7)],
                ["", "Spearman's Rho", _f(mm["spearman"], 7)],
                ["", "Kendall's Tau-b", _f(mm["kendall"], 7)],
                ["Variation Coefficients", "Variation Proximities", _f(mm["cv_prox"], 7)],
                ["", "Variation Transformed Proximities", _f(mm["cv_trans"], 7)],
                ["", "Variation Distances", _f(mm["cv_dist"], 7)],
                ["Degeneracy Indices", "Sum-of-Squares of DeSarbo's Intermixedness Indices",
                 _f(mm["desarbo"], 7)],
                ["", "Shepard's Rough Nondegeneracy Index", _f(mm["shepard"], 7)]]
        sec.table(Table("Measures", ["", "", "Value"], rows, row_header_cols=2,
                        footnotes=[f"Transformation: {TRANSFORMS[transform]}"
                                   f"{' (within each row separately)' if conditional else ''}; "
                                   f"penalty strength {strength:g}, range {penalty_range:g}."]))
    if coordinates:
        sec.table(Table("Final Column Coordinates", ["", *[("Dimension", str(s + 1))
                                                          for s in range(ndim)]],
                        [[v] + [_f(x) for x in fit.cols[j]] for j, v in enumerate(variables)]))
        sec.table(Table("Final Row Coordinates", ["", *[("Dimension", str(s + 1))
                                                       for s in range(ndim)]],
                        [[rl[i]] + [_f(x) for x in fit.rows[i]] for i in range(len(rl))]))
    if ndim >= 2 and "joint" in plots:
        sec.image("Joint Plot", _scatter([(fit.rows, rl if len(rl) <= 60 else None, "Rows", "o"),
                                          (fit.cols, variables, "Columns", "^")],
                                         "Final Common Space"))
    if "transformation" in plots:
        fig = figure(5.4, 4.0)
        ax = fig.add_subplot(111)
        ax.scatter(M[W > 0], fit.gamma[W > 0], s=8, color=COLORS[0])
        ax.set_xlabel("Proximities")
        ax.set_ylabel("Transformed Proximities")
        ax.set_title("Transformation Plot", fontsize=10)
        sec.image("Transformation Plot", png(fig))
    if "shepard" in plots:
        fig = figure(5.4, 4.0)
        ax = fig.add_subplot(111)
        ax.scatter(M[W > 0], fit.dist[W > 0], s=8, color=COLORS[1])
        ax.set_xlabel("Proximities")
        ax.set_ylabel("Distances")
        ax.set_title("Shepard Plot", fontsize=10)
        sec.image("Shepard Plot", png(fig))
    return Result(sec)


# ===================================================================== selftest
def selftest() -> None:
    rng = np.random.default_rng(0)
    n = 60
    X = rng.integers(1, 5, size=(n, 3)).astype(float)
    y = np.clip(np.round(X[:, 0] - 0.5 * X[:, 1] + rng.normal(0, 1, n)) + 3, 1, 9)
    ds = Dataset(pd.DataFrame({"y": y, "a": X[:, 0], "b": X[:, 1], "c": X[:, 2]}))
    r = catreg(ds, {"name": "y", "level": "nume"},
               [{"name": c, "level": "nume"} for c in "abc"], bootstrap_se=0)
    t = next(i for i in r.section.items if getattr(i, "title", "") == "Model Summary")
    import statsmodels.api as sm

    ols = sm.OLS(y, sm.add_constant(X)).fit()
    assert abs(float(t.rows[0][1]) - ols.rsquared) < 1e-3
    r = catpca(ds, [{"name": c, "level": "ordi"} for c in "yabc"], plots=("objects", "biplot"))
    assert any(getattr(i, "title", "") == "Component Loadings" for i in r.section.items)
    r = mca(ds, list("abc"))
    vs = [E.make_var(c, X[:, i], "mnom") for i, c in enumerate("abc")]
    t = next(i for i in r.section.items if getattr(i, "title", "") == "Model Summary")
    assert abs(float(t.rows[0][2]) - E.mca_closed_form(vs, np.ones(n), 2)[0]) < 2e-3
    overals(ds, [[{"name": "a", "level": "nume"}], [{"name": "b", "level": "nume"},
                                                     {"name": "c", "level": "ordi"}]])
    prefscal(ds, ["a", "b", "c"], maxiter=50)

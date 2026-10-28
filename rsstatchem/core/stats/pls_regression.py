"""Analyze > Regression > Partial Least Squares (SPSS PLS), following IBM's "PLS Algorithms".

X (factors as indicator columns, as in a GLM design without intercept, plus covariates) and Y
(scale dependents; categorical dependents dummy coded without the reference category) are
centred and standardised; latent factors come from NIPALS (one iteration for one dependent),
which this module takes from :mod:`rsstatchem.core.analysis.chemometrics` (shared with
Origin's PLS). Output follows SPSS: Proportion of Variance Explained (X variance, cumulative X
variance, Y variance, cumulative Y variance = R², adjusted R²), Parameters (B in original
units), Variable Importance in the Projection (cumulative VIP by latent factor,
VIP_jk = √(n Σ_{a≤k} SSY_a w²_ja / Σ_{a≤k} SSY_a)), Weights and Loadings, the VIP, factor
score and factor weight plots, and saved predicted values, residuals, distances to the model
(DModX) and latent factor scores.
"""
from __future__ import annotations

import numpy as np

from rsstatchem.core.analysis import chemometrics as cm
from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int
from rsstatchem.core.stats.ml_common import (MLError, MLResult, categorical, complete_rows,
                                             full_series, measure_of, raw_column)
from rsstatchem.core.stats.roc import figure, png, style


def design(ds: Dataset, deps: list[str], factors: list[str], covariates: list[str],
           rows: np.ndarray):
    xcols, xnames = [], []
    for f in factors:
        cat = categorical(ds, f, rows)
        for j, lab in enumerate(cat.labels):
            xcols.append((cat.codes == j).astype(float))
            xnames.append(f"[{f}={lab}]")
    for c in covariates:
        xcols.append(raw_column(ds, c).to_numpy(float)[rows])
        xnames.append(c)
    ycols, ynames = [], []
    for d in deps:
        if measure_of(ds, d) != "scale":
            cat = categorical(ds, d, rows)
            for j, lab in enumerate(cat.labels[:-1]):
                ycols.append((cat.codes == j).astype(float))
                ynames.append(f"[{d}={lab}]")
        else:
            ycols.append(raw_column(ds, d).to_numpy(float)[rows])
            ynames.append(d)
    return np.column_stack(xcols), xnames, np.column_stack(ycols), ynames


def cumulative_vip(model: cm.PLSModel, X, Y) -> np.ndarray:
    ssy = cm.explained(model, X, Y)["ssy_comp"]
    W = model.x_weights / np.linalg.norm(model.x_weights, axis=0)
    p = W.shape[0]
    out = np.zeros((p, model.ncomp))
    for k in range(model.ncomp):
        s = ssy[:k + 1]
        out[:, k] = np.sqrt(p * ((W[:, :k + 1] ** 2) @ s) / s.sum())
    return out


def pls_procedure(ds: Dataset, deps: list[str], factors: list[str], covariates: list[str],
                  n_factors: int = 5, id_var: str | None = None, plots: tuple = ("vip",),
                  save: tuple = ()) -> MLResult:
    if not deps:
        raise MLError("Select at least one dependent variable.")
    if not factors and not covariates:
        raise MLError("Select at least one independent variable.")
    rows = complete_rows(ds, list(deps) + list(factors) + list(covariates))
    X, xnames, Y, ynames = design(ds, deps, factors, covariates, rows)
    n, p = X.shape
    if n < 3:
        raise MLError("PLS needs at least 3 complete cases.")
    keep = X.std(0) > 0
    X, xnames = X[:, keep], [nm for nm, k in zip(xnames, keep) if k]
    A = int(max(1, min(n_factors, cm._max_comp(n, X.shape[1]))))
    model = cm.pls_nipals(X, Y, A, scale=True)
    ex = cm.explained(model, X, Y)
    section = Section("Partial Least Squares Regression")
    section.log(f"PLS {' '.join(deps)} BY {' '.join(factors)} WITH {' '.join(covariates)}\n"
                f"  /CRITERIA LATENTFACTORS={A}.")
    rows_v = []
    cx, cy = np.cumsum(ex["x"]), np.cumsum(ex["y"])
    for k in range(A):
        adj = 1 - (1 - cy[k]) * (n - 1) / (n - k - 2) if n - k - 2 > 0 else float("nan")
        rows_v.append([str(k + 1), fmt(ex["x"][k], 3), fmt(cx[k], 3), fmt(ex["y"][k], 3),
                       fmt(cy[k], 3), fmt(adj, 3)])
    section.table(Table("Proportion of Variance Explained", [
        "Latent Factors", ("Statistics", "X Variance"),
        ("Statistics", "Cumulative X Variance"), ("Statistics", "Y Variance"),
        ("Statistics", "Cumulative Y Variance (R-square)"),
        ("Statistics", "Adjusted R-square")], rows_v))
    B, b0 = model.coef, model.intercept
    section.table(Table("Parameters", ["Independent Variables"] +
                        [("Dependent Variables", y) for y in ynames],
                        [["(Constant)"] + [fmt(v, 3) for v in b0]] +
                        [[nm] + [fmt(v, 3) for v in B[i]] for i, nm in enumerate(xnames)]))
    vip = cumulative_vip(model, X, Y)
    section.table(Table("Variable Importance in the Projection", ["Variables"] +
                        [("Latent Factors", str(k + 1)) for k in range(A)],
                        [[nm] + [fmt(v, 3) for v in vip[i]] for i, nm in enumerate(xnames)],
                        footnotes=["Cumulative variable importance."]))
    W = model.x_weights
    section.table(Table("Weights", ["Variables"] + [("Latent Factors", str(k + 1))
                                                     for k in range(A)],
                        [[nm] + [fmt(v, 3) for v in W[i]] for i, nm in enumerate(xnames)] +
                        [[nm] + [fmt(v, 3) for v in model.y_loadings[i]]
                         for i, nm in enumerate(ynames)]))
    section.table(Table("Loadings", ["Variables"] + [("Latent Factors", str(k + 1))
                                                      for k in range(A)],
                        [[nm] + [fmt(v, 3) for v in model.x_loadings[i]]
                         for i, nm in enumerate(xnames)]))
    if "vip" in plots:
        fig = figure(5.6, 3.8)
        ax = fig.add_subplot(111)
        for i, nm in enumerate(xnames):
            ax.plot(range(1, A + 1), vip[i], marker="o", label=nm)
        ax.axhline(0.8, color="#999", ls="--", lw=0.8)
        ax.legend(fontsize=7, ncol=2)
        style(ax, "Cumulative Variable Importance", "Latent Factors", "VIP")
        section.image("Cumulative Variable Importance", png(fig), width=480)
    if "scores" in plots and A >= 2:
        fig = figure(4.8, 4.4)
        ax = fig.add_subplot(111)
        T = model.x_scores
        ax.scatter(T[:, 0], T[:, 1], s=14)
        if id_var:
            ids = ds.analysis_series(id_var).iloc[rows].astype(str).to_numpy()
            for (a, b), t in zip(T[:, :2], ids):
                ax.annotate(t, (a, b), fontsize=6)
        style(ax, "Factor Scores", "Latent Factor 1", "Latent Factor 2")
        section.image("Factor Scores", png(fig), width=440)
    if "weights" in plots and A >= 2:
        fig = figure(4.8, 4.4)
        ax = fig.add_subplot(111)
        ax.scatter(W[:, 0], W[:, 1], s=14)
        for (a, b), t in zip(W[:, :2], xnames):
            ax.annotate(t, (a, b), fontsize=7)
        ax.axhline(0, color="#bbb", lw=0.6)
        ax.axvline(0, color="#bbb", lw=0.6)
        style(ax, "Factor Weights", "Latent Factor 1", "Latent Factor 2")
        section.image("Factor Weights", png(fig), width=440)
    saved = {}
    if save:
        pred = model.predict(X)
        for j, nm in enumerate(ynames):
            if "predicted" in save:
                saved[f"PredictedValue_{j + 1}"] = (f"Predicted value of {nm}",
                                                    full_series(ds, rows, pred[:, j]))
            if "residuals" in save:
                saved[f"Residual_{j + 1}"] = (f"Residual of {nm}",
                                              full_series(ds, rows, Y[:, j] - pred[:, j]))
        if "distance" in save:
            Xs = (X - model.x_mean) / model.x_std
            E = Xs - model.x_scores @ model.x_loadings.T
            saved["DModX"] = ("Distance to model (X)", full_series(
                ds, rows, np.sqrt((E ** 2).sum(1) / max(X.shape[1] - A, 1))))
        if "scores" in save:
            for k in range(A):
                saved[f"LatentFactor_{k + 1}"] = (f"Latent factor score {k + 1}",
                                                  full_series(ds, rows, model.x_scores[:, k]))
    section.table(Table("Case Summary", ["", "N"], [["Valid", fmt_int(n)],
                                                    ["Excluded", fmt_int(ds.n_cases - n)]]))
    return MLResult(section, saved, model)


def selftest() -> None:
    r = np.random.default_rng(4)
    X = r.normal(size=(40, 4))
    y = X @ np.array([1.0, -2.0, 0.5, 0.0]) + 0.1 * r.normal(size=40)
    m = cm.pls_nipals(X, y, 3)
    assert np.corrcoef(m.predict(X)[:, 0], y)[0, 1] > 0.99
    assert cumulative_vip(m, X, y).shape == (4, 3)

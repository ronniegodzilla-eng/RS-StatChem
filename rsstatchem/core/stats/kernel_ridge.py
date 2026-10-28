"""Analyze > Regression > Kernel Ridge (SPSS 28+ KRR extension, which wraps scikit-learn's
``KernelRidge`` and ``GridSearchCV``). Own numpy implementation with the same definitions, so
results equal scikit-learn's:

* dual coefficients α = (K + λI)⁻¹ y (no intercept), predictions ŷ = K(x, X) α;
* kernels: linear x'y; polynomial (γ x'y + c₀)^d; rbf exp(−γ‖x − y‖²); laplacian
  exp(−γ‖x − y‖₁); sigmoid tanh(γ x'y + c₀); cosine x'y/(‖x‖‖y‖); chi2 exp(−γ Σ (x−y)²/(x+y));
  additive_chi2 −Σ (x−y)²/(x+y); γ defaults to 1/p (1 for chi2);
* several values of λ (alpha), γ, c₀ or d (or several kernels) are tuned by k-fold
  cross-validation (contiguous folds, as scikit-learn's KFold without shuffling) with R² as the
  score; the best setting is refitted on all cases.
"""
from __future__ import annotations

import itertools

import numpy as np

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int
from rsstatchem.core.stats.ml_common import (MLError, MLResult, categorical, complete_rows,
                                             full_series, measure_of, raw_column,
                                             scatter_observed_predicted)

KERNELS = {"linear": "Linear", "poly": "Polynomial", "rbf": "Radial basis function (RBF)",
           "laplacian": "Laplacian", "sigmoid": "Sigmoid", "cosine": "Cosine",
           "chi2": "Chi-square", "additive_chi2": "Additive chi-square"}


def kernel(A: np.ndarray, B: np.ndarray, kind: str, gamma: float | None = None,
           degree: float = 3, coef0: float = 1) -> np.ndarray:
    if gamma is None:      # scikit-learn: chi2_kernel defaults to 1, the others to 1/p
        g = 1.0 if kind == "chi2" else 1.0 / A.shape[1]
    else:
        g = gamma
    if kind == "linear":
        return A @ B.T
    if kind == "poly":
        return (g * (A @ B.T) + coef0) ** degree
    if kind == "sigmoid":
        return np.tanh(g * (A @ B.T) + coef0)
    if kind == "rbf":
        d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
        return np.exp(-g * np.maximum(d2, 0))
    if kind == "laplacian":
        return np.exp(-g * np.abs(A[:, None, :] - B[None, :, :]).sum(2))
    if kind == "cosine":
        na = np.linalg.norm(A, axis=1)
        nb = np.linalg.norm(B, axis=1)
        return (A @ B.T) / np.outer(np.where(na > 0, na, 1), np.where(nb > 0, nb, 1))
    if kind in ("chi2", "additive_chi2"):
        if (A < 0).any() or (B < 0).any():
            raise MLError("The chi-square kernels need non-negative data.")
        num = (A[:, None, :] - B[None, :, :]) ** 2
        den = A[:, None, :] + B[None, :, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(den > 0, num / den, 0.0).sum(2)
        return -s if kind == "additive_chi2" else np.exp(-g * s)
    raise MLError(f"Unknown kernel {kind!r}.")


def fit(X, y, alpha=1.0, kind="rbf", gamma=None, degree=3, coef0=1):
    K = kernel(X, X, kind, gamma, degree, coef0)
    K[np.diag_indices_from(K)] += alpha
    try:
        dual = np.linalg.solve(K, y)
    except np.linalg.LinAlgError:
        dual = np.linalg.lstsq(K, y, rcond=None)[0]
    return dual


def r2(y, yh) -> float:
    y, yh = np.asarray(y, float), np.asarray(yh, float)
    if y.ndim == 1:
        y, yh = y[:, None], yh[:, None]
    ss = ((y - y.mean(0)) ** 2).sum(0)
    out = 1 - ((y - yh) ** 2).sum(0) / np.where(ss > 0, ss, np.nan)
    return float(np.nanmean(out))


def kfold(n: int, k: int) -> list[np.ndarray]:
    sizes = np.full(k, n // k)
    sizes[: n % k] += 1
    out, s = [], 0
    for z in sizes:
        out.append(np.arange(s, s + z))
        s += z
    return out


def grid_search(X, y, grid: list[dict], folds: int = 5):
    rows = []
    for params in grid:
        scores = []
        for te in kfold(len(X), folds):
            tr = np.setdiff1d(np.arange(len(X)), te)
            dual = fit(X[tr], y[tr], **params)
            yh = kernel(X[te], X[tr], params["kind"], params.get("gamma"),
                        params.get("degree", 3), params.get("coef0", 1)) @ dual
            scores.append(r2(y[te], yh))
        rows.append((params, float(np.mean(scores)), float(np.std(scores))))
    order = np.argsort([-r[1] for r in rows], kind="mergesort")
    rank = np.empty(len(rows), int)
    rank[order] = np.arange(1, len(rows) + 1)
    return rows, rank


def krr_procedure(ds: Dataset, deps: list[str], predictors: list[str],
                  kernels: list[str] = ("rbf",), alphas=(1.0,), gammas=(None,),
                  degrees=(3,), coef0s=(1.0,), folds: int = 5, standardize: bool = False,
                  plots: tuple = ("observed", "residual"), save: tuple = ()) -> MLResult:
    if not deps or not predictors:
        raise MLError("Select a dependent variable and at least one predictor.")
    rows = complete_rows(ds, list(deps) + list(predictors))
    if len(rows) < 3:
        raise MLError("Too few valid cases.")
    cols, names = [], []
    for nm in predictors:
        if measure_of(ds, nm) == "nominal" or ds.var(nm).type != "numeric":
            cat = categorical(ds, nm, rows)
            for j, lab in enumerate(cat.labels[1:], 1):
                cols.append((cat.codes == j).astype(float))
                names.append(f"{nm}={lab}")
        else:
            cols.append(raw_column(ds, nm).to_numpy(float)[rows])
            names.append(nm)
    X = np.column_stack(cols)
    if standardize:
        sd = X.std(0)
        X = (X - X.mean(0)) / np.where(sd > 0, sd, 1)
    Y = np.column_stack([raw_column(ds, d).to_numpy(float)[rows] for d in deps])
    y = Y[:, 0] if Y.shape[1] == 1 else Y
    grid = []
    for kind in kernels:
        uses_gamma = kind in ("poly", "rbf", "laplacian", "sigmoid", "chi2")
        for a, g, d, c in itertools.product(alphas, gammas if uses_gamma else (None,),
                                            degrees if kind == "poly" else (3,),
                                            coef0s if kind in ("poly", "sigmoid") else (1.0,)):
            grid.append({"alpha": float(a), "kind": kind,
                         "gamma": None if g is None else float(g), "degree": d,
                         "coef0": float(c)})
    section = Section("Kernel Ridge Regression")
    section.log(f"SPSSINC KRR DEPENDENT={' '.join(deps)} ENTER={' '.join(predictors)}\n"
                f"  /OPTIONS KERNEL={' '.join(kernels)} ALPHA={' '.join(map(str, alphas))}"
                f" CROSSVALID={folds}.")
    if len(grid) > 1:
        res, rank = grid_search(X, y, grid, int(folds))
        best = res[int(np.argmin(rank))][0]
        tab = []
        for (params, m, s), rk in sorted(zip(res, rank), key=lambda t: t[1]):
            tab.append([str(rk), KERNELS[params["kind"]], fmt(params["alpha"], 4),
                        "auto" if params["gamma"] is None else fmt(params["gamma"], 4),
                        fmt(params["degree"], 0) if params["kind"] == "poly" else "",
                        fmt(params["coef0"], 3) if params["kind"] in ("poly", "sigmoid")
                        else "", fmt(m, 4), fmt(s, 4)])
        section.table(Table("Cross-Validation Results", ["Rank", "Kernel", "Alpha", "Gamma",
                                                         "Degree", "Coef0", "Mean R Square",
                                                         "Std. Deviation"], tab,
                            footnotes=[f"{int(folds)}-fold cross-validation (contiguous "
                                       "folds); score = R square."]))
    else:
        best = grid[0]
    dual = fit(X, y, **best)
    yh = kernel(X, X, best["kind"], best.get("gamma"), best.get("degree", 3),
                best.get("coef0", 1)) @ dual
    yh2 = yh if yh.ndim == 2 else yh[:, None]
    rows_s = [["Kernel", KERNELS[best["kind"]]], ["Alpha (ridge penalty)", fmt(best["alpha"], 4)]]
    if best["kind"] in ("poly", "rbf", "laplacian", "sigmoid", "chi2"):
        g = best["gamma"] if best["gamma"] is not None else (
            1.0 if best["kind"] == "chi2" else 1 / X.shape[1])
        rows_s.append(["Gamma", fmt(g, 4)])
    if best["kind"] == "poly":
        rows_s.append(["Degree", fmt(best["degree"], 0)])
    if best["kind"] in ("poly", "sigmoid"):
        rows_s.append(["Coef0", fmt(best["coef0"], 3)])
    for j, d in enumerate(deps):
        rows_s.append([f"R Square ({d})", fmt(r2(Y[:, j], yh2[:, j]), 4)])
        rows_s.append([f"Root Mean Squared Error ({d})",
                       fmt(float(np.sqrt(((Y[:, j] - yh2[:, j]) ** 2).mean())), 4)])
    rows_s.append(["N", fmt_int(len(rows))])
    section.table(Table("Model Summary", ["", ""], rows_s,
                        footnotes=["Fit statistics on the estimation (training) data."]))
    for j, d in enumerate(deps):
        if "observed" in plots:
            section.image("Observed by Predicted", scatter_observed_predicted(
                Y[:, j], yh2[:, j], d), width=440)
        if "residual" in plots:
            section.image("Residual by Predicted", scatter_observed_predicted(
                Y[:, j], yh2[:, j], d, residuals=True), width=440)
    saved = {}
    for j, d in enumerate(deps):
        suf = f"_{j + 1}" if len(deps) > 1 else ""
        if "predicted" in save:
            saved[f"KRR_Predicted{suf}"] = (f"Kernel ridge predicted value of {d}",
                                            full_series(ds, rows, yh2[:, j]))
        if "residuals" in save:
            saved[f"KRR_Residual{suf}"] = (f"Kernel ridge residual of {d}",
                                           full_series(ds, rows, Y[:, j] - yh2[:, j]))
    return MLResult(section, saved, {"dual": dual, "params": best})


def selftest() -> None:
    r = np.random.default_rng(1)
    X = r.normal(size=(30, 2))
    y = np.sin(X[:, 0]) + X[:, 1]
    dual = fit(X, y, 0.1, "rbf")
    assert r2(y, kernel(X, X, "rbf") @ dual) > 0.9

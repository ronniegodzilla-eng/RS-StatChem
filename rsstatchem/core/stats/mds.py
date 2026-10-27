"""Analyze > Scale > Multidimensional Scaling, mirroring SPSS PROXSCAL and ALSCAL.

* Proximities: the selected variables form a square matrix (rows = the first k cases, lower
  triangle used, dissimilarities or similarities), or are created from the data as Euclidean,
  squared Euclidean, city-block (Manhattan) or Chebychev distances between variables or between
  cases, optionally after z-score standardization (SPSS "Create distances from data").
* PROXSCAL-like: SMACOF majorization (Guttman transform) with Torgerson (classical) or simplex
  start; ratio, interval or ordinal (Kruskal's monotone regression, primary ties) transformations.
  Output: Iteration History, Stress and Fit Measures (normalized raw stress, Stress-I,
  Stress-II, S-Stress, DAF, Tucker's congruence), Final Coordinates, common-space plot,
  transformation (Shepard) plot and residual plot.
* ALSCAL-like: minimizes Young's S-stress (squared distances) with alternating optimal scaling
  (ratio / interval / ordinal), and prints the iteration history, Young's S-stress formula 1,
  Kruskal's Stress formula 1 and RSQ, stimulus coordinates, the derived stimulus configuration
  and the scatterplot of linear fit.

Classical MDS reproduces a Euclidean configuration exactly (up to rotation); SMACOF agrees with
``sklearn.manifold.smacof`` started from the same configuration.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, numeric
from rsstatchem.core.stats.wp4_util import figure, png

MEASURES = {"euclidean": "Euclidean distance", "seuclidean": "Squared Euclidean distance",
            "cityblock": "Block (city-block)", "chebychev": "Chebychev"}
LEVELS = {"ratio": "Ratio", "interval": "Interval", "ordinal": "Ordinal"}


def _f(x, d: int = 3) -> str:
    return fmt(x, d)


# =========================================================================== proximities
def distances(X: np.ndarray, measure: str = "euclidean") -> np.ndarray:
    diff = X[:, None, :] - X[None, :, :]
    if measure == "cityblock":
        return np.abs(diff).sum(axis=2)
    if measure == "chebychev":
        return np.abs(diff).max(axis=2)
    d2 = (diff ** 2).sum(axis=2)
    return d2 if measure == "seuclidean" else np.sqrt(d2)


def proximities_from_data(ds: Dataset, variables: list[str], measure: str = "euclidean",
                          between: str = "variables", standardize: bool = False,
                          label_var: str | None = None) -> tuple[np.ndarray, list[str]]:
    frame = pd.DataFrame({v: numeric(ds, v) for v in variables}).dropna()
    X = frame.to_numpy(float)
    if standardize:
        if between == "variables":
            X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
        else:
            X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    if between == "variables":
        return distances(X.T, measure), [ds.var(v).title for v in variables]
    if label_var:
        labels = [str(x) for x in ds.analysis_series(label_var).loc[frame.index]]
    else:
        labels = [str(i + 1) for i in frame.index]
    return distances(X, measure), labels


def proximities_from_matrix(ds: Dataset, variables: list[str], similarities: bool = False,
                            shape: str = "lower") -> tuple[np.ndarray, list[str]]:
    """Square proximity matrix held in k variables across the first k cases."""
    k = len(variables)
    M = np.column_stack([numeric(ds, v).to_numpy(float)[:k] for v in variables])
    if M.shape[0] < k:
        raise ValueError(f"The proximity matrix needs {k} cases (one per variable).")
    if shape == "lower":
        L = np.tril(np.nan_to_num(M), -1)
        D = L + L.T
    elif shape == "upper":
        U = np.triu(np.nan_to_num(M), 1)
        D = U + U.T
    else:
        D = (np.nan_to_num(M) + np.nan_to_num(M).T) / 2
    np.fill_diagonal(D, 0.0)
    if similarities:
        off = D[~np.eye(k, dtype=bool)]
        D = off.max() + off.min() - D
        np.fill_diagonal(D, 0.0)
    return D, [ds.var(v).title for v in variables]


# =========================================================================== algorithms
def classical(D: np.ndarray, dims: int) -> np.ndarray:
    """Torgerson classical scaling."""
    n = D.shape[0]
    J = np.eye(n) - 1 / n
    B = -0.5 * J @ (D ** 2) @ J
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(vals)[::-1][:dims]
    return vecs[:, order] * np.sqrt(np.clip(vals[order], 0, None))


def pava(y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Weighted pool-adjacent-violators: non-decreasing least-squares fit to y."""
    y = np.asarray(y, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    vals, wts, sizes = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi)
        wts.append(wi)
        sizes.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, s2 = vals.pop(), wts.pop(), sizes.pop()
            v1, w1, s1 = vals.pop(), wts.pop(), sizes.pop()
            wt = w1 + w2
            vals.append((v1 * w1 + v2 * w2) / wt)
            wts.append(wt)
            sizes.append(s1 + s2)
    return np.repeat(vals, sizes)


def _transform(delta: np.ndarray, d: np.ndarray, level: str) -> np.ndarray:
    """Disparities for the off-diagonal pairs (vectors in a fixed pair order)."""
    if level == "ratio":
        b = float(d @ delta / (delta @ delta)) if delta @ delta > 0 else 0.0
        return b * delta
    if level == "interval":
        A = np.column_stack([np.ones_like(delta), delta])
        coef, *_ = np.linalg.lstsq(A, d, rcond=None)
        out = A @ coef
        return np.clip(out, 0, None)
    # ordinal, primary approach: within ties order by current distance
    order = np.lexsort((d, delta))
    fit = np.empty_like(d)
    fit[order] = pava(d[order])
    return fit


def _normalize(dhat: np.ndarray, m: int) -> np.ndarray:
    s = float((dhat ** 2).sum())
    return dhat * math.sqrt(m / s) if s > 0 else dhat


@dataclass
class MDSFit:
    X: np.ndarray
    delta: np.ndarray       # dissimilarity vector (pairs i<j)
    dhat: np.ndarray        # disparities
    d: np.ndarray           # configuration distances
    history: list = field(default_factory=list)
    converged: bool = True
    level: str = "ratio"
    algorithm: str = "proxscal"

    def measures(self) -> dict:
        dh, d = self.dhat, self.d
        res = dh - d
        nraw = float((res ** 2).sum() / (dh ** 2).sum())
        stress1 = math.sqrt((res ** 2).sum() / (d ** 2).sum())
        stress2 = math.sqrt((res ** 2).sum() / ((d - d.mean()) ** 2).sum())
        sstress = math.sqrt(((dh ** 2 - d ** 2) ** 2).sum() / (dh ** 4).sum())
        tucker = float((dh * d).sum() / math.sqrt((dh ** 2).sum() * (d ** 2).sum()))
        rsq = float(np.corrcoef(dh, d)[0, 1] ** 2) if dh.std() > 0 and d.std() > 0 else 1.0
        return {"nraw": nraw, "stress1": stress1, "stress2": stress2, "sstress": sstress,
                "daf": 1 - nraw, "tucker": tucker, "rsq": rsq}


def _pairs(n):
    return np.triu_indices(n, 1)


def smacof(D: np.ndarray, dims: int = 2, level: str = "ratio", init: str = "torgerson",
           maxiter: int = 100, eps: float = 1e-6, X0: np.ndarray | None = None) -> MDSFit:
    n = D.shape[0]
    iu = _pairs(n)
    delta = D[iu]
    m = len(delta)
    if X0 is not None:
        X = np.asarray(X0, float).copy()
    elif init == "simplex":
        X = np.eye(n)[:, :dims] if dims <= n else np.eye(n)
        X = X - X.mean(axis=0)
    else:
        X = classical(D, dims)
    dhat = _normalize(delta.astype(float), m)
    d0 = distances(X)[iu]
    if d0 @ d0 > 0:   # start from the optimally scaled initial configuration
        X = X * float(dhat @ d0 / (d0 @ d0))
    hist = []
    prev = None
    converged = False
    for it in range(maxiter + 1):
        dist = distances(X)
        dvec = dist[iu]
        if it > 0 or level != "ratio":
            dhat = _normalize(_transform(delta, dvec, level), m)
        stress = float(((dhat - dvec) ** 2).sum() / (dhat ** 2).sum())
        hist.append((it, stress, (prev - stress) if prev is not None else float("nan")))
        if prev is not None and prev - stress < eps:
            converged = True
            break
        prev = stress
        Dh = np.zeros((n, n))
        Dh[iu] = dhat
        Dh = Dh + Dh.T
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(dist > 0, Dh / dist, 0.0)
        Bm = -ratio
        np.fill_diagonal(Bm, 0.0)
        np.fill_diagonal(Bm, -Bm.sum(axis=1))
        X = Bm @ X / n
    d_final = distances(X)[iu]
    return MDSFit(X, delta, dhat, d_final, hist, converged, level, "proxscal")


def alscal(D: np.ndarray, dims: int = 2, level: str = "ordinal", maxiter: int = 30,
           eps: float = 0.001) -> MDSFit:
    """S-stress minimization with alternating optimal scaling of squared disparities."""
    n = D.shape[0]
    iu = _pairs(n)
    delta = D[iu]
    m = len(delta)
    X = classical(D, dims)
    d2hat = _normalize(delta ** 2, m)
    hist = []
    prev = None
    converged = False

    def sstress_obj(flat, target):
        Y = flat.reshape(n, dims)
        diff = Y[:, None, :] - Y[None, :, :]
        d2 = (diff ** 2).sum(axis=2)
        r = np.zeros((n, n))
        r[iu] = d2[iu] - target
        r = r + r.T
        val = float((r[iu] ** 2).sum())
        grad = 8 * (r[:, :, None] * diff).sum(axis=1) / 2
        return val, grad.ravel()

    for it in range(1, maxiter + 1):
        res = optimize.minimize(sstress_obj, X.ravel(), args=(d2hat,), jac=True,
                                method="L-BFGS-B", options={"maxiter": 200})
        X = res.x.reshape(n, dims)
        d2 = (distances(X) ** 2)[iu]
        d2hat = _normalize(_transform(delta, d2, level) if level != "ratio" else
                           _transform(delta ** 2, d2, "ratio"), m)
        s = math.sqrt(((d2hat - d2) ** 2).sum() / (d2hat ** 2).sum())
        hist.append((it, s, (prev - s) if prev is not None else float("nan")))
        if prev is not None and prev - s < eps:
            converged = True
            break
        prev = s
    X = X - X.mean(axis=0)
    # ALSCAL normalizes the configuration so that the sum of squared coordinates = n*dims
    X *= math.sqrt(n * dims / (X ** 2).sum())
    d = distances(X)[iu]
    dhat = np.sqrt(np.clip(d2hat, 0, None))
    dhat *= math.sqrt((d ** 2).sum() / (dhat ** 2).sum())
    return MDSFit(X, delta, dhat, d, hist, converged, level, "alscal")


def _rotate_principal(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Y = Xc @ Vt.T
    for k in range(Y.shape[1]):
        if Y[np.argmax(np.abs(Y[:, k])), k] < 0:
            Y[:, k] *= -1
    return Y


# =========================================================================== plots
def config_png(X: np.ndarray, labels, title: str) -> bytes:
    fig = figure(5.6, 5.0)
    ax = fig.add_subplot(111)
    if X.shape[1] == 1:
        ax.scatter(X[:, 0], np.zeros(len(X)), color="#1f4e9c")
        for x, t in zip(X[:, 0], labels):
            ax.annotate(t, (x, 0), fontsize=8, xytext=(2, 4), textcoords="offset points",
                        rotation=45)
        ax.set_yticks([])
    else:
        ax.scatter(X[:, 0], X[:, 1], color="#1f4e9c")
        for (x, y), t in zip(X[:, :2], labels):
            ax.annotate(t, (x, y), fontsize=8, xytext=(3, 3), textcoords="offset points")
        ax.axhline(0, color="#bbbbbb", lw=0.6)
        ax.axvline(0, color="#bbbbbb", lw=0.6)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_ylabel("Dimension 2")
    ax.set_xlabel("Dimension 1")
    ax.set_title(title, fontsize=10)
    return png(fig)


def shepard_png(fit: MDSFit, title: str, residuals: bool = False) -> bytes:
    fig = figure(5.6, 4.4)
    ax = fig.add_subplot(111)
    if residuals:
        ax.scatter(fit.dhat, fit.d, s=12, color="#1f4e9c")
        lim = max(fit.dhat.max(), fit.d.max()) * 1.05
        ax.plot([0, lim], [0, lim], color="#c8102e", lw=0.8)
        ax.set_xlabel("Transformed Proximities (Disparities)")
        ax.set_ylabel("Distances")
    else:
        order = np.argsort(fit.delta)
        ax.scatter(fit.delta, fit.d, s=12, color="#1f4e9c", label="Distances")
        ax.plot(fit.delta[order], fit.dhat[order], color="#c8102e", lw=1.2,
                label="Disparities", drawstyle="steps-post" if fit.level == "ordinal"
                else "default")
        ax.set_xlabel("Proximities")
        ax.set_ylabel("Distances / Disparities")
        ax.legend(fontsize=7)
    ax.set_title(title, fontsize=10)
    return png(fig)


# =========================================================================== procedures
def _prox(ds, variables, source, measure, between, standardize, similarities, shape, label):
    if source == "matrix":
        return proximities_from_matrix(ds, variables, similarities, shape)
    return proximities_from_data(ds, variables, measure, between, standardize, label)


def proxscal(ds: Dataset, variables: list[str], source: str = "matrix", dims: int = 2,
             level: str = "ratio", init: str = "torgerson", maxiter: int = 100,
             eps: float = 1e-4, measure: str = "euclidean", between: str = "variables",
             standardize: bool = False, similarities: bool = False, shape: str = "lower",
             label: str | None = None, history: bool = True,
             plots: tuple = ("common", "transformation")) -> Section:
    sec = Section("Multidimensional Scaling (PROXSCAL)")
    sec.log(f"PROXSCAL VARIABLES={' '.join(variables)}\n  /SHAPE={shape.upper()}\n"
            f"  /INITIAL={init.upper()}\n  /TRANSFORMATION={level.upper()}\n"
            f"  /PROXIMITIES={'SIMILARITIES' if similarities else 'DISSIMILARITIES'}\n"
            f"  /CRITERIA=DIMENSIONS({dims},{dims}) MAXITER({maxiter}) DIFFSTRESS({eps:g})")
    D, labels = _prox(ds, variables, source, measure, between, standardize, similarities,
                      shape, label)
    n = D.shape[0]
    if n < dims + 2:
        sec.note("Warning", f"At least {dims + 2} objects are needed.", "warning")
        return sec
    fit = smacof(D, dims, level, init, maxiter, eps)
    X = _rotate_principal(fit.X)
    sec.table(Table("Case Processing Summary", ["", ""],
                    [["Objects", fmt_int(n)], ["Proximities: total", fmt_int(n * (n - 1) // 2)],
                     ["Proximities: active", fmt_int(len(fit.delta))]]))
    if history:
        rows = [[str(it), _f(s, 5), _f(imp, 5) if imp == imp else ""]
                for it, s, imp in fit.history]
        sec.table(Table("Iteration History", ["Iteration", "Normalized Raw Stress",
                                              "Improvement"], rows,
                        footnotes=[("a. The iteration process stopped because the "
                                    "convergence test value was reached." if fit.converged
                                    else "a. The iteration process stopped because the "
                                    "maximum number of iterations was reached.")]))
    m = fit.measures()
    sec.table(Table("Stress and Fit Measures", ["", ""],
                    [["Normalized Raw Stress", _f(m["nraw"], 5)],
                     ["Stress-I", _f(m["stress1"], 5) + "ᵃ"],
                     ["Stress-II", _f(m["stress2"], 5) + "ᵃ"],
                     ["S-Stress", _f(m["sstress"], 5) + "ᵇ"],
                     ["Dispersion Accounted For (D.A.F.)", _f(m["daf"], 5)],
                     ["Tucker's Coefficient of Congruence", _f(m["tucker"], 5)]],
                    footnotes=["PROXSCAL minimizes Normalized Raw Stress.",
                               "a. Optimal scaling factor = "
                               f"{1 / (1 - m['nraw']):.3f}." if m["nraw"] < 1 else "",
                               "b. S-Stress computed from the final disparities."]))
    cols = [""] + [("Dimension", str(k + 1)) for k in range(dims)]
    sec.table(Table("Final Coordinates", cols,
                    [[lab] + [_f(X[i, k]) for k in range(dims)] for i, lab in
                     enumerate(labels)]))
    if "common" in plots:
        sec.image("Common Space", config_png(X, labels, "Common Space\nObject Points"))
    if "transformation" in plots:
        sec.image("Transformation Plot", shepard_png(fit, "Transformation Plot"))
    if "residuals" in plots:
        sec.image("Residuals Plot", shepard_png(fit, "Residuals Plot", residuals=True))
    return sec


def alscal_section(ds: Dataset, variables: list[str], source: str = "matrix", dims: int = 2,
                   level: str = "ordinal", maxiter: int = 30, eps: float = 0.001,
                   measure: str = "euclidean", between: str = "variables",
                   standardize: bool = False, similarities: bool = False,
                   shape: str = "lower", label: str | None = None,
                   plots: tuple = ("config", "fit")) -> Section:
    sec = Section("Multidimensional Scaling (ALSCAL)")
    sec.log(f"ALSCAL VARIABLES={' '.join(variables)}\n  /SHAPE=SYMMETRIC\n"
            f"  /LEVEL={level.upper()}\n  /CONDITION=MATRIX\n  /MODEL=EUCLID\n"
            f"  /CRITERIA=CONVERGE({eps:g}) STRESSMIN(.005) ITER({maxiter}) "
            f"DIMENS({dims})")
    D, labels = _prox(ds, variables, source, measure, between, standardize, similarities,
                      shape, label)
    n = D.shape[0]
    if n < dims + 2:
        sec.note("Warning", f"At least {dims + 2} stimuli are needed.", "warning")
        return sec
    fit = alscal(D, dims, level, maxiter, eps)
    X = _rotate_principal(fit.X)
    rows = [[str(it), _f(s, 5), _f(imp, 5) if imp == imp else ""] for it, s, imp in fit.history]
    sec.table(Table("Iteration History for the 2 Dimensional Solution (in squared distances)"
                    .replace("2", str(dims)), ["Iteration", "S-stress", "Improvement"], rows,
                    footnotes=["Young's S-stress formula 1 is used."]))
    m = fit.measures()
    sec.note("Stress and squared correlation (RSQ) in distances",
             f"RSQ values are the proportion of variance of the scaled data (disparities) in the "
             f"partition (row, matrix, or entire data) which is accounted for by their "
             f"corresponding distances.\nStress values are Kruskal's stress formula 1.\n\n"
             f"Stress = {m['stress1']:.5f}      RSQ = {m['rsq']:.5f}")
    sec.table(Table(f"Stimulus Coordinates", ["Stimulus Number", "Stimulus Name"] +
                    [("Dimension", str(k + 1)) for k in range(dims)],
                    [[str(i + 1), lab] + [_f(X[i, k], 4) for k in range(dims)]
                     for i, lab in enumerate(labels)], row_header_cols=2,
                    footnotes=["Configuration derived in " + str(dims) + " dimensions."]))
    if "config" in plots:
        sec.image("Derived Stimulus Configuration",
                  config_png(X, labels, "Derived Stimulus Configuration\nEuclidean distance "
                                        "model"))
    if "fit" in plots:
        sec.image("Scatterplot of Linear Fit", shepard_png(fit, "Scatterplot of Linear Fit\n"
                                                                "Euclidean distance model",
                                                           residuals=True))
    return sec


def selftest() -> None:
    rng = np.random.default_rng(4)
    pts = rng.normal(size=(8, 2))
    D = distances(pts)
    X = classical(D, 2)
    assert np.allclose(distances(X), D, atol=1e-8)
    fit = smacof(D, 2, "ordinal")
    assert fit.measures()["stress1"] < 0.05
    df = pd.DataFrame(D, columns=[f"s{i}" for i in range(8)])
    ds = Dataset(df)
    proxscal(ds, list(df.columns), plots=("common", "transformation", "residuals"))
    alscal_section(ds, list(df.columns))

"""Design evaluation before any response is measured (Design-Expert's Evaluation node).

For a chosen model: degrees of freedom (model, residual, lack of fit, pure error), the
standard error of each coefficient in units of sigma, variance inflation factors, Ri², the
power to detect an effect of a given signal-to-noise ratio, the leverage of each run and the
fraction of design space (FDS) curve of the prediction standard error. Only the design matrix
is used, so this works on an empty design layout. For a split-plot design the coefficient
variances are those of generalised least squares with the whole-plot / residual variance ratio
of the design (V = I + ratio Z Z').
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

from rsstatchem.core.doe.model import (Space, build_matrix, drop_aliased, model_terms,
                                       term_columns)


def _pure_error_df(coded: np.ndarray, blocks: np.ndarray | None) -> int:
    keys: dict = {}
    for i, row in enumerate(np.round(coded, 8)):
        key = tuple(row) + ((float(blocks[i]),) if blocks is not None else ())
        keys[key] = keys.get(key, 0) + 1
    return int(sum(c - 1 for c in keys.values()))


def power(c_jj: float, df_res: int, signal_noise: float = 2.0, alpha: float = 0.05) -> float:
    """Probability of detecting an effect (change from low to high) of ``signal_noise`` sigma
    for a single-df term whose coefficient variance is ``c_jj * sigma²``."""
    if df_res <= 0 or c_jj <= 0:
        return float("nan")
    lam = (signal_noise / 2.0) ** 2 / c_jj
    crit = stats.f.ppf(1 - alpha, 1, df_res)
    return float(stats.ncf.sf(crit, 1, df_res, lam))


def evaluate(space: Space, coded: np.ndarray, order: str | None = None, terms=None,
             blocks: np.ndarray | None = None, signal_noise: float = 2.0,
             alpha: float = 0.05, groups: np.ndarray | None = None,
             variance_ratio: float = 1.0) -> dict:
    """Evaluation of the design ``coded`` (coded factor levels) for a model order or terms."""
    coded = np.asarray(coded, float)
    if blocks is not None and len(np.unique(blocks)) < 2:
        blocks = None
    if terms is None:
        terms = model_terms(space, order or "linear")

    class _Obs:  # drop_aliased only needs the coded matrix and the blocks
        pass

    obs = _Obs()
    obs.coded, obs.blocks = coded, blocks
    kept, aliased = drop_aliased(space, obs, terms)
    mat = build_matrix(space, coded, kept, blocks)
    X = mat.X
    n, p = X.shape
    df_block = (len(np.unique(blocks)) - 1) if blocks is not None else 0
    df_model = p - (1 if mat.intercept else 0) - df_block
    if not mat.intercept:
        df_model -= 1
    df_res = n - p
    df_pe = _pure_error_df(coded, blocks)
    df_lof = max(df_res - df_pe, 0)
    out = {"n": n, "p": p, "terms": kept, "aliased": aliased,
           "df": {"Block": df_block, "Model": df_model, "Residuals": df_res,
                  "Lack of Fit": df_lof, "Pure Error": min(df_pe, max(df_res, 0)),
                  "Corr Total": n - 1},
           "rows": [], "leverage": np.zeros(n), "signal_noise": signal_noise, "alpha": alpha,
           "intercept": mat.intercept}
    if p == 0 or n == 0:
        return out
    W = np.eye(n)
    if groups is not None and len(np.unique(groups)) > 1:
        _, gidx = np.unique(groups, return_inverse=True)
        Z = np.zeros((n, gidx.max() + 1))
        Z[np.arange(n), gidx] = 1.0
        m = Z.sum(axis=0)
        W = np.eye(n) - (Z * (variance_ratio / (1 + variance_ratio * m))) @ Z.T
        out["split_plot"] = True
    try:
        inv = np.linalg.inv(X.T @ W @ X)
    except np.linalg.LinAlgError:
        return out
    out["xtx_inv"] = inv
    out["leverage"] = np.einsum("ij,jk,ik->i", X, inv, X)
    # VIF from the centred correlation matrix (as in Fit.vif)
    cols = [j for j in range(p) if not (mat.intercept and j == 0)
            and not (mat.block_slice is not None
                     and mat.block_slice.start <= j < mat.block_slice.stop)]
    vif: dict = {}
    if cols and space.mixture:
        for j in cols:
            vif[j] = float(inv[j, j] * (X[:, j] @ X[:, j]))
    elif cols:
        Z = X[:, cols] - X[:, cols].mean(axis=0)
        norms = np.sqrt(np.sum(Z ** 2, axis=0))
        if np.all(norms > 0):
            try:
                d = np.diag(np.linalg.inv((Z / norms).T @ (Z / norms)))
                vif = {j: float(v) for j, v in zip(cols, d)}
            except np.linalg.LinAlgError:
                vif = {}
    for t, s in mat.groups:
        for j in range(s.start, s.stop):
            c = float(inv[j, j])
            v = vif.get(j)
            out["rows"].append({
                "term": mat.labels[j], "key": t, "df": 1, "se": math.sqrt(max(c, 0.0)),
                "vif": v, "ri2": max(1 - 1 / v, 0.0) if v and v > 0 else None,
                "power": power(c, df_res, signal_noise, alpha) if t else None})
    return out


def fds_curve(space: Space, ev: dict, n_points: int = 4000, seed: int = 0) -> tuple:
    """(fraction, standard error of the mean in sigma units) for the FDS graph."""
    inv = ev.get("xtx_inv")
    if inv is None:
        return np.array([]), np.array([])
    rng = np.random.default_rng(seed)
    k = space.k
    if space.mixture:
        pts = rng.dirichlet(np.ones(k), size=n_points)
        ok = np.ones(len(pts), bool)
        actual = space.to_actual(pts)
        ok &= np.all((actual >= space.lower - 1e-9) & (actual <= space.upper + 1e-9), axis=1)
        pts = pts[ok]
    else:
        pts = rng.uniform(-1, 1, size=(n_points, k))
        for j, f in enumerate(space.factors):
            if f.categoric:
                pts[:, j] = rng.integers(0, len(f.levels), n_points)
    parts = []
    if ev.get("intercept"):
        parts.append(np.ones((len(pts), 1)))
    if ev["df"]["Block"]:
        parts.append(np.zeros((len(pts), ev["df"]["Block"])))
    for t in ev["terms"]:
        parts.append(term_columns(pts, t, space.categoric, space.nlevels)[0])
    R = np.hstack(parts)
    if R.shape[1] != inv.shape[0]:
        return np.array([]), np.array([])
    se = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", R, inv, R), 0, None))
    se.sort()
    frac = (np.arange(1, len(se) + 1) - 0.5) / len(se)
    return frac, se

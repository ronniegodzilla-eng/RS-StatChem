"""Design evaluation before any response is measured (Design-Expert's Evaluation node).

For a chosen model: degrees of freedom (model, residual, lack of fit, pure error), the
standard error of each coefficient in units of sigma, variance inflation factors, Ri², the
power to detect an effect of a given signal-to-noise ratio, the leverage of each run and the
fraction of design space (FDS) curve of the prediction standard error. Only the design matrix
is used, so this works on an empty design layout. For a split-plot design the coefficient
variances are those of generalised least squares with the whole-plot / residual variance ratio
of the design (V = I + ratio Z Z').

Extras: matrix measures (condition number, D/A/G/I efficiencies, leverage), the alias matrix
of any design for omitted terms, correlation of the coefficients and of the model columns,
FDS for the mean or a prediction with a precision goal (half-width d, sigma, alpha), power for
a change delta given sigma, and the standard error of the design over two factors.
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
    out["leverage"] = np.einsum("ij,ij->i", X @ inv, X)
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
    if getattr(space, "combined", False):
        se = fds_values(space, ev, n_points, seed)
        return (np.arange(1, len(se) + 1) - 0.5) / max(len(se), 1), se
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
    se = np.sqrt(np.clip(np.einsum("ij,ij->i", R @ inv, R), 0, None))
    se.sort()
    frac = (np.arange(1, len(se) + 1) - 0.5) / len(se)
    return frac, se


# ---------------------------------------------------------------------------- extras
def power_delta(c_jj: float, df_res: int, delta: float, sigma: float,
                alpha: float = 0.05) -> float:
    """Power to detect a change ``delta`` (low to high) when the noise SD is ``sigma``."""
    if sigma <= 0:
        return float("nan")
    return power(c_jj, df_res, delta / sigma, alpha)


def region_for(space: Space, constraints=None, process_constraints=None):
    """The design region of a space (box, constrained mixture or combined), coded."""
    from rsstatchem.core.doe import constraints as cons

    if getattr(space, "combined", False):
        q = len(space.mix_idx)
        mix = space.factors[:q]
        proc = space.factors[q:]
        return cons.combined_region(
            [f.low for f in mix], [f.high for f in mix], space.total,
            [f.low if not f.categoric else 0 for f in proc],
            [f.high if not f.categoric else 1 for f in proc], constraints,
            process_constraints, [len(f.levels) if f.categoric else 0 for f in proc])
    if space.mixture:
        return cons.mixture_region(space.lower, np.maximum(space.upper, space.lower),
                                   space.total, constraints)
    return cons.process_region([f.low if not f.categoric else 0 for f in space.factors],
                               [f.high if not f.categoric else 1 for f in space.factors],
                               constraints, space.nlevels)


def model_matrix(space: Space, coded: np.ndarray, terms, blocks=None):
    """(X, labels, intercept) of the model with block columns."""
    mat = build_matrix(space, np.asarray(coded, float), list(terms), blocks)
    return mat


def _term_part(space: Space, terms, mat):
    """Columns of ``mat`` that belong to model terms (not blocks)."""
    cols = []
    for t, s in mat.groups:
        cols += list(range(s.start, s.stop))
    return sorted(cols)


def matrix_measures(space: Space, coded: np.ndarray, terms, blocks=None, constraints=None
                    ) -> dict:
    """Condition number, D/A/G/I efficiencies, scaled D, average prediction variance, max
    and average leverage for the model ``terms``."""
    from rsstatchem.core.doe import optimal

    mat = build_matrix(space, np.asarray(coded, float), list(terms), blocks)
    X = mat.X
    cols = _term_part(space, terms, mat)

    def f(c):
        m = build_matrix(space, np.atleast_2d(c), list(terms), None)
        return m.X

    try:
        region = region_for(space, constraints)
        W = optimal.moment_matrix(f, region, space.k)
        # W is over [model columns]; place it on the model columns of X
        Wx = np.zeros((X.shape[1], X.shape[1]))
        Wx[np.ix_(cols, cols)] = W
    except (ValueError, np.linalg.LinAlgError):
        Wx = None
    out = optimal.matrix_measures(X, None, None)
    if Wx is not None and not out.get("singular"):
        inv = np.linalg.inv(X.T @ X)
        out["i_value"] = float(np.trace(inv @ Wx))
        out["scaled_i"] = out["i_value"] * X.shape[0]
    return out


def default_alias_terms(space: Space, terms) -> list:
    """Terms to check for aliasing: the next order up (2FI for a linear model, 3FI for a 2FI
    model, cubic for a quadratic model), minus the model terms."""
    from rsstatchem.core.doe import model as mdl

    if getattr(space, "combined", False) or space.mixture:
        pool = mdl.mixture_terms(space.k, "special_cubic") if not getattr(
            space, "combined", False) else []
    else:
        deg = max((mdl.term_degree(t) for t in terms), default=1)
        has_sq = any(len(set(t)) < len(t) for t in terms if t and t[0] != -1)
        if has_sq:
            order = "cubic"
        else:
            order = {1: "2fi", 2: "3fi"}.get(deg, "3fi")
        pool = mdl.process_terms(space.k, order, space.categoric)
        if order == "3fi" and deg >= 3:
            pool = [t for t in mdl.candidate_terms(space) if mdl.term_degree(t) <= deg + 1
                    and len(set(t)) == len(t)]
    return [t for t in pool if t not in set(terms)]


def alias_matrix(space: Space, coded: np.ndarray, terms, alias_terms=None, blocks=None
                 ) -> tuple[list[str], list[str], np.ndarray]:
    """Alias matrix (X1'X1)^-1 X1'X2 of the model columns on the omitted terms' columns:
    (row labels, column labels, matrix)."""
    from rsstatchem.core.doe import optimal

    alias_terms = default_alias_terms(space, terms) if alias_terms is None else alias_terms
    m1 = build_matrix(space, np.asarray(coded, float), list(terms), blocks)
    m2 = build_matrix(space, np.asarray(coded, float), list(alias_terms), None)
    if not alias_terms:
        return m1.labels, [], np.zeros((m1.X.shape[1], 0))
    X2 = m2.X[:, 1:] if m2.intercept else m2.X
    labels2 = m2.labels[1:] if m2.intercept else m2.labels
    A = optimal.alias_matrix(m1.X, X2)
    A[np.abs(A) < 1e-10] = 0.0
    return m1.labels, labels2, A


def alias_list(space: Space, coded, terms, alias_terms=None, blocks=None,
               tol: float = 1e-6) -> list[str]:
    """Readable alias list: "[A] = A + 0.333 BC - 0.333 DE ..."."""
    rows, cols, A = alias_matrix(space, coded, terms, alias_terms, blocks)
    out = []
    for i, r in enumerate(rows):
        parts = [f"[{r}] = {r}"]
        for j, c in enumerate(cols):
            v = A[i, j]
            if abs(v) > tol:
                parts.append(f"{'-' if v < 0 else '+'} {abs(v):.3g} {c}")
        out.append(" ".join(parts))
    return out


def correlations(space: Space, coded, terms, blocks=None) -> dict:
    """Correlation of the coefficient estimates and Pearson r of the model columns."""
    from rsstatchem.core.doe import optimal

    mat = build_matrix(space, np.asarray(coded, float), list(terms), blocks)
    cols = _term_part(space, terms, mat)
    X = mat.X
    labels = [mat.labels[j] for j in cols]
    coef = optimal.coefficient_correlation(X)[np.ix_(cols, cols)]
    keep = [j for j in cols if not (mat.intercept and j == 0)]
    r = optimal.column_correlation(X[:, keep]) if keep else np.zeros((0, 0))
    return {"labels": labels, "coef": coef, "r_labels": [mat.labels[j] for j in keep],
            "r": r}


def fds_values(space: Space, ev: dict, n_points: int = 4000, seed: int = 0,
               kind: str = "mean", constraints=None) -> np.ndarray:
    """Sorted scaled standard errors (mean or prediction) at about-uniform points of the
    design region (constraints honoured)."""
    inv = ev.get("xtx_inv")
    if inv is None:
        return np.array([])
    rng = np.random.default_rng(seed)
    try:
        pts = region_for(space, constraints).sample(n_points, rng)
    except ValueError:
        return np.array([])
    parts = []
    if ev.get("intercept"):
        parts.append(np.ones((len(pts), 1)))
    if ev["df"]["Block"]:
        parts.append(np.zeros((len(pts), ev["df"]["Block"])))
    for t in ev["terms"]:
        parts.append(term_columns(pts, t, space.categoric, space.nlevels)[0])
    R = np.hstack(parts)
    if R.shape[1] != inv.shape[0]:
        return np.array([])
    v = np.clip(np.einsum("ij,ij->i", R @ inv, R), 0, None)
    if kind == "prediction":
        v = v + 1.0
    se = np.sqrt(v)
    se.sort()
    return se


def fds_score(se: np.ndarray, d: float, sigma: float, alpha: float, df: int) -> dict:
    """FDS with a precision goal: the fraction of the design space where the interval
    half-width t(1-alpha/2, df) * sigma * SE is at most ``d``."""
    if not len(se) or df <= 0 or sigma <= 0:
        return {"fraction": float("nan"), "limit": float("nan"), "t": float("nan")}
    t = float(stats.t.ppf(1 - alpha / 2, df))
    limit = d / (t * sigma)
    return {"fraction": float(np.mean(se <= limit)), "limit": limit, "t": t}


def se_grid(space: Space, ev: dict, i: int, j: int, fixed: np.ndarray | None = None,
            n: int = 41) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standard error of the mean (sigma units) over factors ``i`` x ``j`` (coded -1..1, or
    pseudo components for a mixture), the others at ``fixed`` (coded; default centre)."""
    inv = ev.get("xtx_inv")
    g = np.linspace(-1, 1, n) if not space.mixture else np.linspace(0, 1, n)
    G1, G2 = np.meshgrid(g, g)
    base = np.zeros(space.k) if fixed is None else np.asarray(fixed, float)
    pts = np.tile(base, (n * n, 1))
    pts[:, i], pts[:, j] = G1.ravel(), G2.ravel()
    parts = []
    if ev.get("intercept"):
        parts.append(np.ones((len(pts), 1)))
    if ev["df"]["Block"]:
        parts.append(np.zeros((len(pts), ev["df"]["Block"])))
    for t in ev["terms"]:
        parts.append(term_columns(pts, t, space.categoric, space.nlevels)[0])
    R = np.hstack(parts)
    se = np.sqrt(np.clip(np.einsum("ij,ij->i", R @ inv, R), 0, None)).reshape(n, n)
    return G1, G2, se

"""Chemometrics: PLS regression (NIPALS and SIMPLS) and PCA helpers, no Qt.

**PLS regression** (Origin's Multivariate Analysis > Partial Least Squares). X (n × p) and
Y (n × m) are centred and, with ``scale=True``, divided by their standard deviations
(ddof = 1), as scikit-learn's ``PLSRegression`` does.

* NIPALS (Wold; the PLS2 algorithm of scikit-learn, "mode A", regression deflation): for each
  component the X weights w (unit length), X scores t = Xw, Y weights c = Y't/(t't), Y scores
  u = Yc/(c'c) iterate until w converges; X loadings p = X't/(t't), Y loadings q = Y't/(t't);
  X and Y are deflated by t p' and t q'. Weights are sign-normalised so the largest |w| is
  positive (scikit-learn's convention).
* SIMPLS (de Jong 1993, *Chemom. Intell. Lab. Syst.* 18, 251–263): weights r are the dominant
  left singular vectors of the deflated cross-product S = X'Y; t = X r normalised to unit
  length, p = X't, q = Y't; S is deflated by projecting out the orthonormalised loadings.
  For one response SIMPLS and NIPALS give the same model.

The coefficients B (p × m, original units) satisfy ŷ = x B + b₀. Rotations R = W (P'W)⁻¹
give the scores of new samples (T = X R). Cross-validation (leave-one-out, k-fold contiguous
blocks or venetian blinds) gives PRESS, RMSECV and Q² = 1 − PRESS/SS(Y) per number of
components. VIP (Wold) uses the Y variance explained per component:
VIP_j = √(p Σ_a SSY_a (w_ja/‖w_a‖)² / Σ_a SSY_a).

**PCA** on the correlation or covariance matrix by SVD: eigenvalues, % of variance,
eigenvectors (coefficients; the sign is chosen so each vector's elements sum to a positive
number), scores (standardised data × eigenvectors) and correlation loadings
(eigenvectors × √eigenvalue, for the biplot).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


class ChemometricsError(ValueError):
    pass


# ====================================================================== PLS
@dataclass
class PLSModel:
    method: str
    ncomp: int
    scale: bool
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    x_weights: np.ndarray        # W (p × A)
    x_loadings: np.ndarray       # P (p × A)
    y_loadings: np.ndarray       # Q (m × A)
    x_scores: np.ndarray         # T (n × A)
    y_scores: np.ndarray         # U (n × A)
    x_rotations: np.ndarray      # R (p × A), T = Xs R
    coef_scaled: np.ndarray      # B for scaled data (p × m)
    extra: dict = field(default_factory=dict)

    @property
    def coef(self) -> np.ndarray:
        """Coefficients in original units (p × m)."""
        return self.coef_scaled * self.y_std[None, :] / self.x_std[:, None]

    @property
    def intercept(self) -> np.ndarray:
        return self.y_mean - self.x_mean @ self.coef

    def predict(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        return X @ self.coef + self.intercept

    def transform(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        return ((X - self.x_mean) / self.x_std) @ self.x_rotations


def _prep(X, Y, scale: bool):
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    if X.ndim != 2 or X.shape[0] != Y.shape[0]:
        raise ChemometricsError("X and Y need the same number of rows.")
    if X.shape[0] < 3:
        raise ChemometricsError("PLS needs at least 3 observations.")
    xm, ym = X.mean(0), Y.mean(0)
    if scale:
        xs, ys = X.std(0, ddof=1), Y.std(0, ddof=1)
        xs[xs == 0] = 1.0
        ys[ys == 0] = 1.0
    else:
        xs, ys = np.ones(X.shape[1]), np.ones(Y.shape[1])
    return X, Y, xm, xs, ym, ys, (X - xm) / xs, (Y - ym) / ys


def _max_comp(n: int, p: int) -> int:
    return max(1, min(p, n - 1))


def pls_nipals(X, Y, ncomp: int, scale: bool = True, tol: float = 1e-13,
               max_iter: int = 1000) -> PLSModel:
    X, Y, xm, xs, ym, ys, Xk, Yk = _prep(X, Y, scale)
    n, p = Xk.shape
    m = Yk.shape[1]
    A = int(ncomp)
    if not 1 <= A <= _max_comp(n, p):
        raise ChemometricsError(f"The number of components must be 1…{_max_comp(n, p)}.")
    W, P, Q, T, U = (np.zeros((p, A)), np.zeros((p, A)), np.zeros((m, A)), np.zeros((n, A)),
                     np.zeros((n, A)))
    eps = np.finfo(float).eps
    for a in range(A):
        cols = [j for j in range(m) if np.any(np.abs(Yk[:, j]) > eps)]
        if not cols:
            raise ChemometricsError("Y is fully explained; use fewer components.")
        u = Yk[:, cols[0]].copy()
        w_old = np.zeros(p)
        for _ in range(max_iter):
            w = Xk.T @ u / (u @ u)
            w /= np.linalg.norm(w) + eps
            t = Xk @ w
            c = Yk.T @ t / (t @ t)
            u = Yk @ c / (c @ c)
            if np.sum((w - w_old) ** 2) < tol or m == 1:
                break
            w_old = w
        k = int(np.argmax(np.abs(w)))
        if w[k] < 0:
            w, c = -w, -c
        t = Xk @ w
        u = Yk @ c / (c @ c)
        pa = Xk.T @ t / (t @ t)
        qa = Yk.T @ t / (t @ t)
        Xk = Xk - np.outer(t, pa)
        Yk = Yk - np.outer(t, qa)
        W[:, a], P[:, a], Q[:, a], T[:, a], U[:, a] = w, pa, qa, t, u
    R = W @ np.linalg.pinv(P.T @ W)
    B = R @ Q.T
    return PLSModel("nipals", A, scale, xm, xs, ym, ys, W, P, Q, T, U, R, B)


def pls_simpls(X, Y, ncomp: int, scale: bool = True) -> PLSModel:
    X, Y, xm, xs, ym, ys, X0, Y0 = _prep(X, Y, scale)
    n, p = X0.shape
    m = Y0.shape[1]
    A = int(ncomp)
    if not 1 <= A <= _max_comp(n, p):
        raise ChemometricsError(f"The number of components must be 1…{_max_comp(n, p)}.")
    S = X0.T @ Y0
    R, T, P, Q, U, V = (np.zeros((p, A)), np.zeros((n, A)), np.zeros((p, A)),
                        np.zeros((m, A)), np.zeros((n, A)), np.zeros((p, A)))
    for a in range(A):
        if m == 1:
            r = S[:, 0].copy()
        else:
            uu, _, _ = np.linalg.svd(S, full_matrices=False)
            r = uu[:, 0]
        t = X0 @ r
        t = t - t.mean()
        nt = np.linalg.norm(t)
        if nt == 0:
            raise ChemometricsError("X is fully explained; use fewer components.")
        t /= nt
        r /= nt
        k = int(np.argmax(np.abs(r)))
        if r[k] < 0:
            r, t = -r, -t
        pa = X0.T @ t
        qa = Y0.T @ t
        u = Y0 @ qa
        v = pa.copy()
        if a > 0:
            v -= V[:, :a] @ (V[:, :a].T @ pa)
            u -= T[:, :a] @ (T[:, :a].T @ u)
        v /= np.linalg.norm(v)
        S = S - np.outer(v, v @ S)
        R[:, a], T[:, a], P[:, a], Q[:, a], U[:, a], V[:, a] = r, t, pa, qa, u, v
    B = R @ Q.T
    W = R / np.linalg.norm(R, axis=0)
    return PLSModel("simpls", A, scale, xm, xs, ym, ys, W, P, Q, T, U, R, B)


def pls_fit(X, Y, ncomp: int, method: str = "nipals", scale: bool = True) -> PLSModel:
    if method == "nipals":
        return pls_nipals(X, Y, ncomp, scale)
    if method == "simpls":
        return pls_simpls(X, Y, ncomp, scale)
    raise ChemometricsError(f"Unknown PLS algorithm {method!r}.")


def cv_folds(n: int, how: str = "loo", k: int = 7) -> list[np.ndarray]:
    """Test-set indices of each fold: "loo", "kfold" (contiguous blocks) or "venetian"."""
    idx = np.arange(n)
    if how == "loo":
        return [np.array([i]) for i in idx]
    k = max(2, min(int(k), n))
    if how == "kfold":
        return [b for b in np.array_split(idx, k) if len(b)]
    if how == "venetian":
        return [idx[j::k] for j in range(k) if len(idx[j::k])]
    raise ChemometricsError(f"Unknown cross-validation scheme {how!r}.")


def pls_cv(X, Y, max_comp: int, method: str = "nipals", scale: bool = True,
           how: str = "loo", k: int = 7) -> dict:
    """PRESS (per response and total), RMSECV and Q² for 1…max_comp components."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    n, p = X.shape
    folds = cv_folds(n, how, k)
    A = int(max_comp)
    press = np.zeros((A, Y.shape[1]))
    for test in folds:
        train = np.setdiff1d(np.arange(n), test)
        amax = min(A, _max_comp(len(train), p))
        for a in range(1, amax + 1):
            mdl = pls_fit(X[train], Y[train], a, method, scale)
            press[a - 1] += ((Y[test] - mdl.predict(X[test])) ** 2).sum(0)
        if amax < A:
            press[amax:] = np.nan
    ss = ((Y - Y.mean(0)) ** 2).sum(0)
    tot = press.sum(1)
    return {"press": tot, "press_y": press, "rmsecv": np.sqrt(tot / (n * Y.shape[1])),
            "q2": 1 - tot / ss.sum(), "q2_y": 1 - press / ss[None, :], "folds": len(folds),
            "scheme": how}


def explained(model: PLSModel, X, Y) -> dict:
    """Fraction of the (scaled) X and Y sums of squares explained by each component."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    Xs = (X - model.x_mean) / model.x_std
    Ys = (Y - model.y_mean) / model.y_std
    ssx, ssy = float((Xs ** 2).sum()), float((Ys ** 2).sum())
    T, P, Q = model.x_scores, model.x_loadings, model.y_loadings
    tt = (T ** 2).sum(0)
    if model.method == "simpls":
        # explained by projection onto the (orthonormal) scores
        x_comp = np.array([float(np.sum((np.outer(T[:, a], Xs.T @ T[:, a])) ** 2))
                           for a in range(model.ncomp)])
        y_comp = np.array([float(np.sum((np.outer(T[:, a], Ys.T @ T[:, a])) ** 2))
                           for a in range(model.ncomp)])
    else:
        x_comp = tt * (P ** 2).sum(0)
        y_comp = tt * (Q ** 2).sum(0)
    return {"x": x_comp / ssx, "y": y_comp / ssy, "ssy_comp": y_comp}


def vip(model: PLSModel, X, Y) -> np.ndarray:
    ssy = explained(model, X, Y)["ssy_comp"]
    W = model.x_weights / np.linalg.norm(model.x_weights, axis=0)
    p = W.shape[0]
    return np.sqrt(p * (W ** 2 @ ssy) / ssy.sum())


# ====================================================================== PCA
@dataclass
class PCAResult:
    matrix: str               # "correlation" | "covariance"
    eigenvalues: np.ndarray
    vectors: np.ndarray       # p × p (columns = components)
    scores: np.ndarray        # n × p
    mean: np.ndarray
    scale: np.ndarray
    n: int

    @property
    def percent(self) -> np.ndarray:
        return self.eigenvalues / self.eigenvalues.sum() * 100

    @property
    def cumulative(self) -> np.ndarray:
        return np.cumsum(self.percent)

    @property
    def loadings(self) -> np.ndarray:
        """Correlation loadings (eigenvector × √eigenvalue) for a correlation PCA; for a
        covariance PCA the same product (covariance of variables with the unit scores)."""
        return self.vectors * np.sqrt(np.clip(self.eigenvalues, 0, None))[None, :]

    def biplot(self, a: int = 0, b: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """(standardised scores n × 2, loading vectors p × 2): Gabriel's column-metric
        preserving biplot (α = 0)."""
        ev = np.sqrt(np.clip(self.eigenvalues[[a, b]], 1e-300, None))
        return self.scores[:, [a, b]] / ev[None, :], self.loadings[:, [a, b]]


def pca(X, matrix: str = "correlation") -> PCAResult:
    X = np.asarray(X, float)
    n, p = X.shape
    if n < 3 or p < 2:
        raise ChemometricsError("PCA needs at least 3 observations of 2 or more variables.")
    mean = X.mean(0)
    sd = X.std(0, ddof=1)
    if matrix == "correlation":
        if np.any(sd == 0):
            raise ChemometricsError("A variable is constant; its correlations are undefined.")
        scale = sd
    elif matrix == "covariance":
        scale = np.ones(p)
    else:
        raise ChemometricsError(f"Unknown matrix {matrix!r}.")
    Z = (X - mean) / scale
    C = Z.T @ Z / (n - 1)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    w, V = np.clip(w[order], 0, None), V[:, order]
    s = np.sign(V.sum(0))
    s[s == 0] = 1
    V = V * s
    return PCAResult(matrix, w, V, Z @ V, mean, scale, n)


def bartlett_sphericity(X) -> tuple[float, int, float]:
    from scipy import stats
    X = np.asarray(X, float)
    n, p = X.shape
    R = np.corrcoef(X, rowvar=False)
    det = float(np.linalg.det(R))
    if det <= 0:
        return math.nan, p * (p - 1) // 2, math.nan
    chi = -(n - 1 - (2 * p + 5) / 6) * math.log(det)
    df = p * (p - 1) // 2
    return float(chi), df, float(stats.chi2.sf(chi, df))

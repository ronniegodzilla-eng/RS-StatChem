"""Fast re-estimation of fitted models on bootstrap samples (case resampling).

Each ``*_model`` function returns a :class:`~rsstatchem.core.stats.resample.ModelInfo` whose
``refit``/``refit_batch`` re-estimates the parameters on resampled analysis cases, starting
from the full-data estimates (warm start). They use small NumPy Newton / IRLS solvers rather
than statsmodels so that SPSS's default 1000 bootstrap samples finish in seconds:

* linear models (OLS/WLS, GLM Univariate parameters): batched normal equations,
* binary logistic: batched Newton-Raphson (SPSS's 20 iterations),
* multinomial logistic: Newton-Raphson on the baseline-category logits,
* ordinal (PLUM) cumulative link models: Newton-Raphson with the analytic Hessian for the
  logit, probit, complementary log-log, negative log-log and Cauchit links,
* generalized linear models: IRLS with the statsmodels family/link objects,
* Cox regression: the Breslow partial likelihood of :mod:`rsstatchem.core.stats.survival`,
* discriminant analysis: Fisher's classification function coefficients.

A sample in which a model cannot be estimated (singular design, a category missing from the
sample, non-finite estimates) returns NaN and is excluded, as SPSS does.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import special

from rsstatchem.core.stats.resample import ModelInfo

_MAX_ELEMS = 2_000_000


def batch_rows(n: int, p: int) -> int:
    return max(1, int(_MAX_ELEMS // max(n * max(p, 1), 1)))


def _solve_batch(H: np.ndarray, g: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(H, g[..., None])[..., 0]
    except np.linalg.LinAlgError:
        out = np.full(g.shape, np.nan)
        for i in range(len(H)):
            try:
                out[i] = np.linalg.solve(H[i], g[i])
            except np.linalg.LinAlgError:
                pass
        return out


# ============================================================================ linear models
def ols_estimates(y, X, w=None):
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    Xw = X * w[:, None]
    XtX = X.T @ Xw
    b = np.linalg.solve(XtX, Xw.T @ y)
    r = y - X @ b
    df = len(y) - X.shape[1]
    mse = float((w * r * r).sum() / df) if df > 0 else float("nan")
    se = np.sqrt(np.clip(np.diag(np.linalg.inv(XtX)) * mse, 0, None))
    return b, se, df


def ols_model(y, X, names, w=None, **kw) -> ModelInfo:
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    b, se, df = ols_estimates(y, X, w)
    p = X.shape[1]

    def refit_batch(idx):
        Xb, yb, wb = X[idx], y[idx], w[idx]
        XtX = np.einsum("bnp,bn,bnq->bpq", Xb, wb, Xb)
        Xty = np.einsum("bnp,bn->bp", Xb, wb * yb)
        out = _solve_batch(XtX, Xty)
        # samples whose design is (numerically) singular, e.g. a dummy that is always 0
        diag = np.einsum("bpp->bp", XtX)
        out[(diag <= 1e-12 * np.maximum(diag.max(axis=1, keepdims=True), 1)).any(axis=1)] = \
            np.nan
        return out

    def refit_batch_se(idx):
        Xb, yb, wb = X[idx], y[idx], w[idx]
        XtX = np.einsum("bnp,bn,bnq->bpq", Xb, wb, Xb)
        bb = refit_batch(idx)
        res_ = yb - np.einsum("bnp,bp->bn", Xb, np.nan_to_num(bb))
        dfb = len(y) - p
        mse = (wb * res_ * res_).sum(axis=1) / dfb if dfb > 0 else np.full(len(idx), np.nan)
        with np.errstate(all="ignore"):
            try:
                inv_diag = np.einsum("bpp->bp", np.linalg.inv(XtX))
            except np.linalg.LinAlgError:
                inv_diag = np.full((len(idx), p), np.nan)
        se_b = np.sqrt(np.clip(inv_diag * mse[:, None], 0, None))
        se_b[~np.isfinite(bb).all(axis=1)] = np.nan
        return np.hstack([bb, se_b])

    return ModelInfo(list(names), b, se, refit_batch=refit_batch, n=len(y), df=df,
                     batch=batch_rows(len(y), p), refit_batch_se=refit_batch_se, **kw)


# ============================================================================ binary logistic
def logit_model(y, X, names, start, max_iter: int = 20, **kw) -> ModelInfo:
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    start = np.asarray(start, float)
    p = X.shape[1]

    def refit_batch(idx):
        Xb, yb = X[idx], y[idx]
        beta = np.tile(start, (len(idx), 1))
        for _ in range(max(max_iter, 1)):
            mu = special.expit(np.einsum("bnp,bp->bn", Xb, beta))
            w = mu * (1 - mu)
            H = np.einsum("bnp,bn,bnq->bpq", Xb, w, Xb)
            g = np.einsum("bnp,bn->bp", Xb, yb - mu)
            step = _solve_batch(H, g)
            beta = beta + step
            if np.all(~np.isfinite(step) | (np.abs(step) < 1e-8)):
                break
        # a sample with only one outcome category cannot be fitted
        one_class = (yb.min(axis=1) == yb.max(axis=1))
        beta[one_class] = np.nan
        return beta

    def refit_batch_se(idx):
        beta = refit_batch(idx)
        Xb = X[idx]
        mu = special.expit(np.einsum("bnp,bp->bn", Xb, np.nan_to_num(beta)))
        H = np.einsum("bnp,bn,bnq->bpq", Xb, mu * (1 - mu), Xb)
        with np.errstate(all="ignore"):
            try:
                se_b = np.sqrt(np.clip(np.einsum("bpp->bp", np.linalg.inv(H)), 0, None))
            except np.linalg.LinAlgError:
                se_b = np.full(beta.shape, np.nan)
        se_b[~np.isfinite(beta).all(axis=1)] = np.nan
        return np.hstack([beta, se_b])

    return ModelInfo(list(names), start, kw.pop("se", None), refit_batch=refit_batch,
                     n=len(y), batch=batch_rows(len(y), p), refit_batch_se=refit_batch_se, **kw)


# ============================================================================ multinomial
def mnlogit_fit(y, X, J: int, start: np.ndarray, max_iter: int = 30) -> np.ndarray:
    """Baseline-category logit (category 0 = reference); returns (p, J-1) coefficients."""
    n, p = X.shape
    B = np.asarray(start, float).copy()
    Y = (y[:, None] == np.arange(1, J)[None, :]).astype(float)
    for _ in range(max_iter):
        eta = np.column_stack([np.zeros(n), X @ B])
        P = special.softmax(eta, axis=1)[:, 1:]
        g = (X.T @ (Y - P)).T.ravel()             # category-major
        m = J - 1
        H = np.zeros((m * p, m * p))
        for j in range(m):
            for k in range(j, m):
                wjk = P[:, j] * ((j == k) - P[:, k])
                blk = (X * wjk[:, None]).T @ X
                H[j * p:(j + 1) * p, k * p:(k + 1) * p] = blk
                H[k * p:(k + 1) * p, j * p:(j + 1) * p] = blk.T
        step = np.linalg.solve(H, g)
        B = B + step.reshape(m, p).T
        if np.max(np.abs(step)) < 1e-8:
            break
    return B


def mnlogit_model(y, X, J: int, names, start, **kw) -> ModelInfo:
    """``start`` (p, J-1) with columns in category order 1..J-1 (statsmodels layout)."""
    y = np.asarray(y, int)
    X = np.asarray(X, float)
    start = np.asarray(start, float)
    est = start.T.ravel()

    def refit(idx):
        yb = y[idx]
        if len(np.unique(yb)) < J:
            return np.full(len(est), np.nan)
        return mnlogit_fit(yb, X[idx], J, start).T.ravel()

    return ModelInfo(list(names), est, kw.pop("se", None), refit=refit, n=len(y), **kw)


# ============================================================================ ordinal (PLUM)
def _link_funcs(link: str):
    if link == "logit":
        def F(x):
            return special.expit(x)

        def f(x):
            e = special.expit(x)
            return e * (1 - e)

        def d(x):
            e = special.expit(x)
            return e * (1 - e) * (1 - 2 * e)
    elif link == "probit":
        F = special.ndtr

        def f(x):
            return np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

        def d(x):
            return -x * np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    elif link == "cloglog":
        def F(x):
            return -np.expm1(-np.exp(x))

        def f(x):
            e = np.exp(x)
            return e * np.exp(-e)

        def d(x):
            e = np.exp(x)
            return e * np.exp(-e) * (1 - e)
    elif link == "nloglog":
        def F(x):
            return np.exp(-np.exp(-x))

        def f(x):
            e = np.exp(-x)
            return e * np.exp(-e)

        def d(x):
            e = np.exp(-x)
            return e * np.exp(-e) * (e - 1)
    else:  # cauchit
        def F(x):
            return 0.5 + np.arctan(x) / math.pi

        def f(x):
            return 1 / (math.pi * (1 + x * x))

        def d(x):
            return -2 * x / (math.pi * (1 + x * x) ** 2)
    return F, f, d


def ordinal_fit(y, X, J: int, link: str, theta0, beta0, max_iter: int = 50):
    """Cumulative link model P(Y <= j) = F(theta_j - x'beta) by Newton-Raphson."""
    F, f, d = _link_funcs(link)
    n, k = X.shape
    m = J - 1
    A = np.zeros((n, m + k))
    Bm = np.zeros((n, m + k))
    top = y < m           # upper bound finite
    bot = y > 0           # lower bound finite
    A[np.flatnonzero(top), y[top]] = 1.0
    Bm[np.flatnonzero(bot), y[bot] - 1] = 1.0
    A[:, m:] = -X
    Bm[:, m:] = -X
    A[~top] = 0.0
    Bm[~bot] = 0.0
    par = np.r_[np.asarray(theta0, float), np.asarray(beta0, float)]

    def parts(par):
        th = np.r_[-np.inf, par[:m], np.inf]
        eta = X @ par[m:] if k else np.zeros(n)
        a = th[y + 1] - eta
        b = th[y] - eta
        fa = np.where(top, f(np.where(top, a, 0.0)), 0.0)
        fb = np.where(bot, f(np.where(bot, b, 0.0)), 0.0)
        pr = np.where(top, F(np.where(top, a, 0.0)), 1.0) - \
            np.where(bot, F(np.where(bot, b, 0.0)), 0.0)
        return a, b, fa, fb, pr

    def loglik(par):
        if m > 1 and np.any(np.diff(par[:m]) <= 0):
            return -np.inf
        pr = parts(par)[4]
        return float(np.sum(np.log(pr))) if np.all(pr > 0) else -np.inf

    ll = loglik(par)
    for _ in range(max_iter):
        a, b, fa, fb, pr = parts(par)
        G = (fa[:, None] * A - fb[:, None] * Bm) / pr[:, None]
        g = G.sum(axis=0)
        da = np.where(top, d(np.where(top, a, 0.0)), 0.0) / pr
        db = np.where(bot, d(np.where(bot, b, 0.0)), 0.0) / pr
        H = (A * da[:, None]).T @ A - (Bm * db[:, None]).T @ Bm - G.T @ G
        step = np.linalg.solve(-H, g)
        t = 1.0
        while True:
            new = par + t * step
            ll_new = loglik(new)
            if ll_new >= ll - 1e-10 or t < 1e-4:
                break
            t /= 2
        par, ll = new, ll_new
        if np.max(np.abs(t * step)) < 1e-8:
            break
    if not np.isfinite(ll):
        raise ValueError("ordinal model could not be fitted")
    return par


def ordinal_model(y, X, J: int, link: str, names, theta, beta, **kw) -> ModelInfo:
    y = np.asarray(y, int)
    X = np.asarray(X, float)
    est = np.r_[theta, beta]

    def refit(idx):
        yb = y[idx]
        if len(np.unique(yb)) < J:
            return np.full(len(est), np.nan)
        return ordinal_fit(yb, X[idx], J, link, theta, beta)

    return ModelInfo(list(names), est, kw.pop("se", None), refit=refit, n=len(y), **kw)


# ============================================================================ GENLIN
def glm_irls(y, X, family, w, offset, start, max_iter: int = 50, tol: float = 1e-9):
    beta = np.asarray(start, float).copy()
    off = 0.0 if offset is None else offset
    link = family.link
    for _ in range(max_iter):
        eta = X @ beta + off
        mu = family.fitted(eta)
        dmu = 1.0 / link.deriv(mu)
        var = family.variance(mu)
        wt = w * dmu * dmu / var
        z = eta - off + (y - mu) / dmu
        Xw = X * wt[:, None]
        new = np.linalg.solve(X.T @ Xw, Xw.T @ z)
        change = np.max(np.abs(new - beta)) if len(beta) else 0.0
        beta = new
        if change < tol * 10 * (1 + np.max(np.abs(beta))):
            break
    if not np.all(np.isfinite(beta)):
        raise ValueError("GLM did not converge")
    return beta


def glm_model(y, X, family, w, offset, names, start, **kw) -> ModelInfo:
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    off = None if offset is None else np.asarray(offset, float)

    def refit(idx):
        return glm_irls(y[idx], X[idx], family, w[idx], None if off is None else off[idx],
                        start)

    return ModelInfo(list(names), np.asarray(start, float), kw.pop("se", None), refit=refit,
                     n=len(y), **kw)


# ============================================================================ Cox
def cox_model(stop, event, X, strata, names, beta, start=None, **kw) -> ModelInfo:
    from rsstatchem.core.stats.survival import CoxData, cox_fit

    stop = np.asarray(stop, float)
    event = np.asarray(event, float)
    X = np.asarray(X, float)
    st = None if strata is None else np.asarray(strata, dtype=object)
    p = X.shape[1]

    def refit(idx):
        if event[idx].sum() == 0:
            return np.full(p, np.nan)
        data = CoxData(stop[idx], event[idx], X[idx], None if st is None else st[idx])
        fit = cox_fit(data, list(range(p)), beta0=beta)
        return fit.beta if fit.converged else np.full(p, np.nan)

    return ModelInfo(list(names), np.asarray(beta, float), kw.pop("se", None), refit=refit,
                     n=len(stop), **kw)


# ============================================================================ discriminant
def classification_functions(X, g, n_groups: int, priors=None) -> np.ndarray:
    """Fisher's linear classification functions; returns (p + 1, groups) with the constant
    in the last row (SPSS "Classification Function Coefficients")."""
    n, p = X.shape
    means = np.array([X[g == k].mean(axis=0) for k in range(n_groups)])
    counts = np.array([(g == k).sum() for k in range(n_groups)], float)
    if np.any(counts < 1):
        raise ValueError("empty group")
    R = X - means[g]
    S = R.T @ R / (n - n_groups)
    Si = np.linalg.inv(S)
    C = Si @ means.T
    pri = counts / n if priors is None else np.asarray(priors, float)
    const = -0.5 * np.einsum("kp,pk->k", means, C) + np.log(pri)
    return np.vstack([C, const])


def discriminant_model(X, g, n_groups: int, names, priors=None, **kw) -> ModelInfo:
    X = np.asarray(X, float)
    g = np.asarray(g, int)
    est = classification_functions(X, g, n_groups, priors).ravel()

    def refit(idx):
        gb = g[idx]
        if len(np.unique(gb)) < n_groups:
            return np.full(len(est), np.nan)
        return classification_functions(X[idx], gb, n_groups, priors).ravel()

    return ModelInfo(list(names), est, None, refit=refit, n=len(X), **kw)

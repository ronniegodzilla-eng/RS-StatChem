"""Weibull survival model with right censoring (Origin's Survival Analysis > Weibull Fit),
no Qt.

S(t) = exp(−(t/λ)^k), shape k > 0, scale λ > 0. With event indicators dᵢ (1 = event,
0 = censored) the log-likelihood is

    ℓ(k, λ) = Σ dᵢ [ln k − k ln λ + (k − 1) ln tᵢ] − Σ (tᵢ/λ)^k.

For fixed k the MLE of λ is λ̂^k = Σ tᵢ^k / D (D = number of events); k̂ solves the profile
score  D/k + Σ dᵢ ln tᵢ − D Σ tᵢ^k ln tᵢ / Σ tᵢ^k = 0  (monotone in k, found by bracketing).
Standard errors come from the observed information on (ln k, ln λ) (analytic second
derivatives) and the delta method; confidence limits are Wald limits on the log scale. With
``exponential=True`` k is fixed at 1: λ̂ = Σ tᵢ / D, SE = λ̂/√D (closed form).

:func:`weibull_groups` fits every group and adds the likelihood-ratio test of one common
Weibull distribution against separate ones (2(g − 1) DF) and of a common shape (g − 1 DF).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.optimize import brentq


class WeibullError(ValueError):
    pass


@dataclass
class WeibullFit:
    shape: float
    scale: float
    se_shape: float
    se_scale: float
    cov_log: np.ndarray     # covariance of (ln k, ln λ)
    loglik: float
    n: int
    events: int
    level: float
    exponential: bool = False

    def _ci(self, value: float, se_log: float) -> tuple[float, float]:
        z = stats.norm.ppf(0.5 + self.level / 2)
        if not np.isfinite(se_log):
            return math.nan, math.nan
        return value * math.exp(-z * se_log), value * math.exp(z * se_log)

    @property
    def shape_ci(self) -> tuple[float, float]:
        if self.exponential:
            return 1.0, 1.0
        return self._ci(self.shape, math.sqrt(self.cov_log[0, 0]))

    @property
    def scale_ci(self) -> tuple[float, float]:
        return self._ci(self.scale, math.sqrt(self.cov_log[1, 1]))

    def survival(self, t) -> np.ndarray:
        t = np.asarray(t, float)
        return np.exp(-(np.clip(t, 0, None) / self.scale) ** self.shape)

    def hazard(self, t) -> np.ndarray:
        t = np.asarray(t, float)
        return self.shape / self.scale * (t / self.scale) ** (self.shape - 1)

    def quantile(self, p: float) -> float:
        """Time by which a fraction ``p`` has failed (median: p = 0.5)."""
        return self.scale * (-math.log(1 - p)) ** (1 / self.shape)

    @property
    def median(self) -> float:
        return self.quantile(0.5)

    @property
    def mean(self) -> float:
        return self.scale * math.gamma(1 + 1 / self.shape)

    @property
    def aic(self) -> float:
        return -2 * self.loglik + 2 * (1 if self.exponential else 2)


def weibull_loglik(k: float, lam: float, t: np.ndarray, d: np.ndarray) -> float:
    z = t / lam
    with np.errstate(divide="ignore", invalid="ignore"):
        ev = np.where(d > 0, np.log(k) - k * np.log(lam) + (k - 1) * np.log(t), 0.0)
    return float(np.sum(ev) - np.sum(z ** k))


def _prepare(t, d) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, float)
    d = np.asarray(d, float)
    ok = np.isfinite(t) & np.isfinite(d)
    t, d = t[ok], (d[ok] != 0).astype(float)
    if np.any(t <= 0):
        raise WeibullError("Survival times must be positive.")
    if d.sum() < 1:
        raise WeibullError("The Weibull fit needs at least one event (uncensored time).")
    return t, d


def _info_log(k: float, lam: float, t: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Observed information of (a, b) = (ln k, ln λ)."""
    D = d.sum()
    z = t / lam
    lz = np.log(z)
    zk = z ** k
    # ℓ = D a − D k b ... in terms of a, b with k = e^a, λ = e^b:
    # ∂ℓ/∂a = D + k Σ d ln z − k Σ z^k ln z ;  ∂ℓ/∂b = −k D + k Σ z^k
    s_dlz = float(np.sum(d * lz))
    s_zl = float(np.sum(zk * lz))
    s_zl2 = float(np.sum(zk * lz * lz))
    s_z = float(np.sum(zk))
    haa = k * s_dlz - k * s_zl - k * k * s_zl2
    hab = -k * D + k * s_z + k * k * s_zl
    hbb = -k * k * s_z
    return -np.array([[haa, hab], [hab, hbb]])


def weibull_fit(t, d, level: float = 0.95, exponential: bool = False) -> WeibullFit:
    t, d = _prepare(t, d)
    D = float(d.sum())
    n = len(t)
    if exponential:
        lam = float(t.sum() / D)
        ll = weibull_loglik(1.0, lam, t, d)
        cov = np.array([[0.0, 0.0], [0.0, 1.0 / D]])
        return WeibullFit(1.0, lam, 0.0, lam / math.sqrt(D), cov, ll, n, int(D), level, True)
    lt = np.log(t)
    s_dlt = float(np.sum(d * lt))
    tmax = float(t.max())
    lts = lt - math.log(tmax)          # scaled for numerical stability

    def score(k):
        w = np.exp(k * lts)
        return D / k + s_dlt - D * float(np.sum(w * lt)) / float(np.sum(w))

    lo, hi = 1e-3, 1.0
    while score(hi) > 0 and hi < 1e4:
        hi *= 2
    while score(lo) < 0 and lo > 1e-8:
        lo /= 2
    if score(lo) * score(hi) > 0:
        raise WeibullError("The Weibull shape could not be estimated (all times equal?).")
    k = float(brentq(score, lo, hi, xtol=1e-14, rtol=1e-14, maxiter=500))
    lam = float(tmax * (np.sum(np.exp(k * lts)) / D) ** (1 / k))
    ll = weibull_loglik(k, lam, t, d)
    info = _info_log(k, lam, t, d)
    try:
        cov = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        cov = np.full((2, 2), math.nan)
    return WeibullFit(k, lam, k * math.sqrt(max(cov[0, 0], 0)), lam * math.sqrt(max(cov[1, 1],
                                                                                    0)),
                      cov, ll, n, int(D), level, False)


def common_shape_loglik(groups: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float]:
    """(k̂, max log L) of Weibull models sharing the shape with separate scales."""
    def prof(k):
        total = 0.0
        for t, d in groups:
            D = d.sum()
            lam = (np.sum(t ** k) / D) ** (1 / k)
            total += weibull_loglik(k, lam, t, d)
        return total

    from scipy.optimize import minimize_scalar
    res = minimize_scalar(lambda a: -prof(math.exp(a)), bounds=(-7, 7), method="bounded",
                          options={"xatol": 1e-10})
    return math.exp(res.x), -float(res.fun)


def weibull_groups(t, d, g, level: float = 0.95, exponential: bool = False) -> dict:
    """Fits per group plus likelihood-ratio tests (see the module docstring)."""
    t = np.asarray(t, float)
    d = np.asarray(d, float)
    g = np.asarray(g, dtype=object)
    levels = []
    for v in g:
        if v not in levels and v not in (None, ""):
            levels.append(v)
    fits = {}
    data = []
    for lv in levels:
        m = g == lv
        tt, dd = _prepare(t[m], d[m])
        fits[lv] = weibull_fit(tt, dd, level, exponential)
        data.append((tt, dd))
    allt, alld = _prepare(np.concatenate([x for x, _ in data]),
                          np.concatenate([y for _, y in data]))
    pooled = weibull_fit(allt, alld, level, exponential)
    sep = sum(f.loglik for f in fits.values())
    gk = len(levels)
    per = 1 if exponential else 2
    lr = 2 * (sep - pooled.loglik)
    out = {"levels": levels, "fits": fits, "pooled": pooled,
           "lr_equal": (lr, per * (gk - 1), float(stats.chi2.sf(lr, per * (gk - 1))))}
    if not exponential and gk > 1:
        _, ll_common = common_shape_loglik(data)
        lr2 = 2 * (sep - ll_common)
        out["lr_shape"] = (lr2, gk - 1, float(stats.chi2.sf(max(lr2, 0.0), gk - 1)))
    return out


"""Distribution fitting (Origin's Statistics > Descriptive Statistics > Distribution Fit),
no Qt.

Maximum-likelihood fits of Normal, Lognormal, Gamma, Weibull, Exponential, Logistic,
Gumbel (largest / smallest extreme value) to one sample, with standard errors from the
observed information (numerical Hessian of the log-likelihood), −2 log L, AIC, BIC and the
goodness-of-fit statistics Kolmogorov-Smirnov D, Anderson-Darling A² and Cramér-von Mises
W². Their p-values come either from the simple-hypothesis distributions (scipy's ``kstest``
/ ``cramervonmises``, conservative because the parameters were estimated; Anderson-Darling
has no simple p-value here) or, with ``bootstrap=B``, from a parametric bootstrap (refit to
each of B simulated samples; Stute, Manteiga & Quindimil 1993), which accounts for the
estimation.

Closed-form MLEs where they exist (normal, lognormal, exponential) and scipy's ``fit`` with
the location fixed at 0 for the positive distributions otherwise.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

DISTRIBUTIONS = {
    "normal": "Normal", "lognormal": "Lognormal", "gamma": "Gamma", "weibull": "Weibull",
    "exponential": "Exponential", "logistic": "Logistic",
    "gumbel_max": "Extreme Value (Gumbel, largest)",
    "gumbel_min": "Extreme Value (Gumbel, smallest)"}
POSITIVE = {"lognormal", "gamma", "weibull", "exponential"}
PARAM_NAMES = {
    "normal": ("Mu (location)", "Sigma (scale)"),
    "lognormal": ("Mu (log location)", "Sigma (log scale)"),
    "gamma": ("Alpha (shape)", "Beta (scale)"),
    "weibull": ("k (shape)", "Lambda (scale)"),
    "exponential": ("Beta (scale = mean)",),
    "logistic": ("Mu (location)", "s (scale)"),
    "gumbel_max": ("Mu (location)", "Beta (scale)"),
    "gumbel_min": ("Mu (location)", "Beta (scale)"),
}


class DistFitError(ValueError):
    pass


@dataclass
class DistFit:
    key: str
    params: np.ndarray
    se: np.ndarray
    loglik: float
    n: int
    ks: float
    ks_p: float
    ad: float
    ad_p: float
    cvm: float
    cvm_p: float
    p_method: str
    extra: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return DISTRIBUTIONS[self.key]

    @property
    def k(self) -> int:
        return len(self.params)

    @property
    def aic(self) -> float:
        return -2 * self.loglik + 2 * self.k

    @property
    def bic(self) -> float:
        return -2 * self.loglik + self.k * math.log(self.n)

    def frozen(self):
        return frozen(self.key, self.params)


def frozen(key: str, params):
    p = list(params)
    if key == "normal":
        return stats.norm(p[0], p[1])
    if key == "lognormal":
        return stats.lognorm(p[1], scale=math.exp(p[0]))
    if key == "gamma":
        return stats.gamma(p[0], scale=p[1])
    if key == "weibull":
        return stats.weibull_min(p[0], scale=p[1])
    if key == "exponential":
        return stats.expon(scale=p[0])
    if key == "logistic":
        return stats.logistic(p[0], p[1])
    if key == "gumbel_max":
        return stats.gumbel_r(p[0], p[1])
    if key == "gumbel_min":
        return stats.gumbel_l(p[0], p[1])
    raise DistFitError(f"Unknown distribution {key!r}.")


def mle(key: str, v: np.ndarray) -> np.ndarray:
    """Maximum-likelihood parameters (in :data:`PARAM_NAMES` order)."""
    if key == "normal":
        return np.array([v.mean(), v.std(ddof=0)])
    if key == "lognormal":
        lv = np.log(v)
        return np.array([lv.mean(), lv.std(ddof=0)])
    if key == "exponential":
        return np.array([v.mean()])
    if key == "gamma":
        a, _, scale = stats.gamma.fit(v, floc=0)
        return np.array([a, scale])
    if key == "weibull":
        k, _, scale = stats.weibull_min.fit(v, floc=0)
        return np.array([k, scale])
    if key == "logistic":
        return np.array(stats.logistic.fit(v))
    if key == "gumbel_max":
        return np.array(stats.gumbel_r.fit(v))
    if key == "gumbel_min":
        return np.array(stats.gumbel_l.fit(v))
    raise DistFitError(f"Unknown distribution {key!r}.")


def loglik(key: str, params, v: np.ndarray) -> float:
    with np.errstate(all="ignore"):
        if any((not np.isfinite(x)) for x in params):
            return -math.inf
        if key != "normal" and key != "lognormal" and key not in ("logistic", "gumbel_max",
                                                                  "gumbel_min"):
            if any(x <= 0 for x in params):
                return -math.inf
        elif params[-1] <= 0:
            return -math.inf
        return float(np.sum(frozen(key, params).logpdf(v)))


def _hessian(f, x: np.ndarray) -> np.ndarray:
    k = len(x)
    h = 1e-4 * np.maximum(np.abs(x), 1e-3)
    H = np.zeros((k, k))
    f0 = f(x)
    for i in range(k):
        for j in range(i, k):
            if i == j:
                e = np.zeros(k)
                e[i] = h[i]
                H[i, i] = (f(x + e) - 2 * f0 + f(x - e)) / h[i] ** 2
            else:
                ei, ej = np.zeros(k), np.zeros(k)
                ei[i], ej[j] = h[i], h[j]
                H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej)
                                     + f(x - ei - ej)) / (4 * h[i] * h[j])
    return H


def anderson_darling(v: np.ndarray, dist) -> float:
    x = np.sort(v)
    n = len(x)
    with np.errstate(divide="ignore"):
        lf = dist.logcdf(x)
        ls = dist.logsf(x[::-1])
    i = np.arange(1, n + 1)
    return float(-n - np.sum((2 * i - 1) * (lf + ls)) / n)


def gof_statistics(v: np.ndarray, dist) -> tuple[float, float, float]:
    ks = float(stats.kstest(v, dist.cdf).statistic)
    ad = anderson_darling(v, dist)
    cvm = float(stats.cramervonmises(v, dist.cdf).statistic)
    return ks, ad, cvm


def fit(key: str, values, bootstrap: int = 0, seed: int = 12345) -> DistFit:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    n = len(v)
    if key not in DISTRIBUTIONS:
        raise DistFitError(f"Unknown distribution {key!r}.")
    if n < 3:
        raise DistFitError("Distribution fitting needs at least 3 values.")
    if key in POSITIVE and np.any(v <= 0):
        raise DistFitError(f"The {DISTRIBUTIONS[key]} distribution needs positive data.")
    if not np.ptp(v) > 0:
        raise DistFitError("All values are equal.")
    params = mle(key, v)
    ll = loglik(key, params, v)
    try:
        H = _hessian(lambda p: loglik(key, p, v), params)
        cov = np.linalg.inv(-H)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except (np.linalg.LinAlgError, ValueError):
        se = np.full(len(params), math.nan)
    dist = frozen(key, params)
    ks, ad, cvm = gof_statistics(v, dist)
    if bootstrap and bootstrap > 0:
        rng = np.random.default_rng(seed)
        exceed = np.zeros(3)
        done = 0
        for _ in range(int(bootstrap)):
            sim = dist.rvs(n, random_state=rng)
            try:
                ps = mle(key, sim)
                s = gof_statistics(sim, frozen(key, ps))
            except (ValueError, FloatingPointError, RuntimeError):
                continue
            done += 1
            exceed += np.array(s) >= np.array([ks, ad, cvm]) - 1e-12
        pv = (exceed + 1) / (done + 1)
        ks_p, ad_p, cvm_p = (float(x) for x in pv)
        method = f"parametric bootstrap ({done} samples)"
    else:
        ks_p = float(stats.kstest(v, dist.cdf).pvalue)
        cvm_p = float(stats.cramervonmises(v, dist.cdf).pvalue)
        ad_p = math.nan
        method = "simple hypothesis (conservative: parameters estimated)"
    return DistFit(key, params, se, ll, n, ks, ks_p, ad, ad_p, cvm, cvm_p, method,
                   {"mean": float(dist.mean()), "sd": float(dist.std()),
                    "median": float(dist.median())})


def fit_all(values, keys=None, bootstrap: int = 0, seed: int = 12345) -> list[DistFit]:
    """Fit every applicable distribution (positive ones skipped for data ≤ 0)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    out = []
    for k in (keys or DISTRIBUTIONS):
        if k in POSITIVE and np.any(v <= 0):
            continue
        out.append(fit(k, v, bootstrap, seed))
    return out

"""Variance tests (Origin's One-Sample / Two-Sample Test for Variance and the equal-variance
tests), no Qt.

* One sample: χ² = (n − 1)s²/σ₀² on n − 1 DF, CI of σ² from the χ² quantiles.
* Two samples: F = s₁²/(s₂² · ratio₀) on (n₁ − 1, n₂ − 1) DF, CI of σ₁²/σ₂².
* k samples: Bartlett, Levene (absolute deviations from the mean) and Brown-Forsythe
  (deviations from the median) through scipy.stats.

``tail``: "two" (≠), "upper" (>), "lower" (<).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


class VarTestError(ValueError):
    pass


def _p(cdf: float, sf: float, tail: str) -> float:
    if tail == "two":
        return float(min(1.0, 2 * min(cdf, sf)))
    return float(sf if tail == "upper" else cdf)


def chi2_variance_test(x, sigma2_0: float, tail: str = "two", level: float = 0.95) -> dict:
    v = np.asarray(x, float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 2:
        raise VarTestError("The variance test needs at least 2 values.")
    if not sigma2_0 > 0:
        raise VarTestError("The hypothesized variance must be positive.")
    s2 = float(v.var(ddof=1))
    df = n - 1
    chi = df * s2 / sigma2_0
    p = _p(stats.chi2.cdf(chi, df), stats.chi2.sf(chi, df), tail)
    a = 1 - level
    if tail == "two":
        lo, hi = df * s2 / stats.chi2.ppf(1 - a / 2, df), df * s2 / stats.chi2.ppf(a / 2, df)
    elif tail == "upper":
        lo, hi = df * s2 / stats.chi2.ppf(1 - a, df), math.inf
    else:
        lo, hi = 0.0, df * s2 / stats.chi2.ppf(a, df)
    return {"n": n, "variance": s2, "sd": math.sqrt(s2), "chi2": chi, "df": df, "p": p,
            "lcl": lo, "ucl": hi}


def f_variance_test(a, b, ratio0: float = 1.0, tail: str = "two",
                    level: float = 0.95) -> dict:
    x = np.asarray(a, float)
    y = np.asarray(b, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        raise VarTestError("Each sample needs at least 2 values.")
    if not ratio0 > 0:
        raise VarTestError("The hypothesized variance ratio must be positive.")
    s1, s2 = float(x.var(ddof=1)), float(y.var(ddof=1))
    if not s2 > 0:
        raise VarTestError("The second sample has zero variance.")
    d1, d2 = len(x) - 1, len(y) - 1
    f = s1 / s2 / ratio0
    p = _p(stats.f.cdf(f, d1, d2), stats.f.sf(f, d1, d2), tail)
    r = s1 / s2
    al = 1 - level
    if tail == "two":
        lo, hi = r / stats.f.ppf(1 - al / 2, d1, d2), r / stats.f.ppf(al / 2, d1, d2)
    elif tail == "upper":
        lo, hi = r / stats.f.ppf(1 - al, d1, d2), math.inf
    else:
        lo, hi = 0.0, r / stats.f.ppf(al, d1, d2)
    return {"n1": len(x), "n2": len(y), "var1": s1, "var2": s2, "ratio": r, "f": f,
            "df1": d1, "df2": d2, "p": p, "lcl": lo, "ucl": hi}


def k_sample_tests(groups: list[np.ndarray]) -> dict:
    """Bartlett, Levene (mean) and Brown-Forsythe (median) of ``k`` ≥ 2 samples."""
    gs = [np.asarray(g, float)[np.isfinite(np.asarray(g, float))] for g in groups]
    if len(gs) < 2 or any(len(g) < 2 for g in gs):
        raise VarTestError("Each of at least two groups needs at least 2 values.")
    k, n = len(gs), sum(len(g) for g in gs)
    b = stats.bartlett(*gs)
    lv = stats.levene(*gs, center="mean")
    bf = stats.levene(*gs, center="median")
    return {"k": k, "n": n,
            "bartlett": (float(b.statistic), k - 1, float(b.pvalue)),
            "levene": (float(lv.statistic), (k - 1, n - k), float(lv.pvalue)),
            "brown_forsythe": (float(bf.statistic), (k - 1, n - k), float(bf.pvalue))}

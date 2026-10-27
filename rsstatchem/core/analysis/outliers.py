"""Outlier tests for analytical data: Grubbs' test and Dixon's Q (ratio) tests (no Qt).

**Grubbs** (Grubbs 1969; NIST/SEMATECH e-Handbook §1.3.5.17): ``G = max|xᵢ − x̄| / s``
(two-sided), ``(x̄ − x_min)/s`` or ``(x_max − x̄)/s`` (one-sided). The critical value is

    G_crit = (n − 1)/√n · √(t² / (n − 2 + t²)),  t = t(α/(2n), n − 2)  (two-sided; α/n one-sided)

and the p-value is the Bonferroni bound ``min(1, n·P(T > t_G))`` (×2 two-sided) with
``t_G = √(n(n − 2)G² / ((n − 1)² − nG²))`` — exact whenever it is below about α, the
region that matters. :func:`grubbs_iterative` repeats the test after removing each outlier.

**Dixon** (Dixon 1950, 1951; Rorabacher 1991, *Anal. Chem.* 63, 139–146): the ratios

    r10 = (x₂ − x₁)/(xₙ − x₁)       r11 = (x₂ − x₁)/(xₙ₋₁ − x₁)
    r21 = (x₃ − x₁)/(xₙ₋₁ − x₁)     r22 = (x₃ − x₁)/(xₙ₋₂ − x₁)

for a suspect low value (mirror image for a high one). The upper-tail probabilities of the
ratios of a normal sample are computed exactly here by conditioning on x₁ and the
denominator's order statistic: with ``i`` the numerator gap (1 or 2), ``j`` the points above
the denominator's top value (0, 1 or 2) and ``m = n − 2 − j`` interior points,

    P(r > q) = n!/(m! j!) ∬ φ(a) φ(c) (1 − Φ(c))ʲ Σ_{k<i} C(m,k) F_lo^k F_hi^(m−k) da dc

with ``t = a + q(c − a)``, ``F_lo = Φ(t) − Φ(a)``, ``F_hi = Φ(c) − Φ(t)``. The double integral
is evaluated by Gauss-Legendre quadrature on ``u = Φ(a)``, ``v = (Φ(c) − u)/(1 − u)``
(converged to < 1e-4; confirmed by Monte Carlo). One-tailed α = 0.05 / 0.025 / 0.005 are
Rorabacher's two-tailed 90 / 95 / 99 % columns; his printed r10 values (kept in
:data:`RORABACHER_R10`, selectable with ``critical="table"``) agree with the exact ones to
within 0.002 except n = 4 at 99 % (0.926 printed vs 0.921 exact), the limited-precision
integration McBane (2006, *J. Stat. Softw.* 16(3)) pointed out. Dixon's recommended ranges: r10 for 3 ≤ n ≤ 7, r11 for
8–10, r21 for 11–13, r22 for 14–25 (``kind="auto"``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy import special, stats
from scipy.optimize import brentq

DIXON_KINDS = {"r10": (1, 0), "r11": (1, 1), "r21": (2, 1), "r22": (2, 2)}
DIXON_NAMES = {"r10": "r10 (Q = gap / range)", "r11": "r11", "r21": "r21", "r22": "r22"}
DIXON_MIN_N = {"r10": 3, "r11": 4, "r21": 5, "r22": 6}
MAX_N = 200
# Rorabacher (1991) Table: r10 critical values, two-tailed 90 % / 95 % / 99 % (= one-tailed
# α 0.05 / 0.025 / 0.005), n = 3…10.
RORABACHER_R10 = {
    3: (0.941, 0.970, 0.994), 4: (0.765, 0.829, 0.926), 5: (0.642, 0.710, 0.821),
    6: (0.560, 0.625, 0.740), 7: (0.507, 0.568, 0.680), 8: (0.468, 0.526, 0.634),
    9: (0.437, 0.493, 0.598), 10: (0.412, 0.466, 0.568)}
_RORA_ALPHA = (0.05, 0.025, 0.005)   # one-tailed


class OutlierError(ValueError):
    pass


def _clean(x) -> np.ndarray:
    v = np.asarray(x, float)
    return v[np.isfinite(v)]


# ====================================================================== Grubbs
@dataclass
class GrubbsResult:
    n: int
    mean: float
    sd: float
    statistic: float
    critical: float
    p: float
    index: int            # position (in the input order) of the suspect value
    value: float
    side: str             # "two" | "max" | "min"
    alpha: float
    outlier: bool


def grubbs_critical(n: int, alpha: float = 0.05, side: str = "two") -> float:
    if n < 3:
        return math.nan
    a = alpha / (2 * n) if side == "two" else alpha / n
    t = stats.t.isf(a, n - 2)
    return (n - 1) / math.sqrt(n) * math.sqrt(t * t / (n - 2 + t * t))


def grubbs_pvalue(g: float, n: int, side: str = "two") -> float:
    if n < 3 or not np.isfinite(g):
        return math.nan
    den = (n - 1) ** 2 - n * g * g
    if den <= 0:
        return 0.0
    t = math.sqrt(n * (n - 2) * g * g / den)
    p = n * stats.t.sf(t, n - 2)
    return float(min(1.0, 2 * p if side == "two" else p))


def grubbs(x, alpha: float = 0.05, side: str = "two") -> GrubbsResult:
    """One Grubbs test. ``x`` may contain NaN (ignored); ``index`` refers to the input."""
    allv = np.asarray(x, float)
    ok = np.flatnonzero(np.isfinite(allv))
    v = allv[ok]
    n = len(v)
    if n < 3:
        raise OutlierError("Grubbs' test needs at least 3 values.")
    mean, sd = float(v.mean()), float(v.std(ddof=1))
    if not sd > 0:
        raise OutlierError("All values are equal: Grubbs' test is undefined.")
    if side == "max":
        k = int(np.argmax(v))
        g = (v[k] - mean) / sd
    elif side == "min":
        k = int(np.argmin(v))
        g = (mean - v[k]) / sd
    elif side == "two":
        k = int(np.argmax(np.abs(v - mean)))
        g = abs(v[k] - mean) / sd
    else:
        raise OutlierError(f"Unknown alternative {side!r}.")
    crit = grubbs_critical(n, alpha, side)
    p = grubbs_pvalue(g, n, side)
    return GrubbsResult(n, mean, sd, float(g), crit, p, int(ok[k]), float(v[k]), side, alpha,
                        bool(g > crit))


def grubbs_iterative(x, alpha: float = 0.05, side: str = "two",
                     max_outliers: int | None = None) -> list[GrubbsResult]:
    """Repeat Grubbs' test, removing the detected outlier each time, until a test is not
    significant (that last test is included) or fewer than 3 values remain."""
    v = np.asarray(x, float).copy()
    out = []
    limit = max_outliers if max_outliers is not None else len(v)
    while np.isfinite(v).sum() >= 3 and len(out) < limit + 1:
        try:
            r = grubbs(v, alpha, side)
        except OutlierError:
            break
        out.append(r)
        if not r.outlier or sum(o.outlier for o in out) >= limit:
            break
        v[r.index] = np.nan
    return out


# ====================================================================== Dixon
_NODES = 160


@lru_cache(maxsize=4)
def _grid(nodes: int):
    z, w = np.polynomial.legendre.leggauss(nodes)
    g = (z + 1) / 2
    wg = w / 2
    u, v = np.meshgrid(g, g, indexing="ij")
    wt = np.outer(wg, wg)
    a = special.ndtri(u)
    wc = u + (1 - u) * v
    c = special.ndtri(wc)
    return u, wc, a, c, wt * (1 - u)


def dixon_sf(q: float, n: int, kind: str = "r10", nodes: int = _NODES) -> float:
    """P(r > q) for Dixon's ratio ``kind`` in a normal sample of size ``n`` (one tail)."""
    if kind not in DIXON_KINDS:
        raise OutlierError(f"Unknown Dixon statistic {kind!r}.")
    i, j = DIXON_KINDS[kind]
    if n < DIXON_MIN_N[kind]:
        raise OutlierError(f"Dixon's {kind} needs at least {DIXON_MIN_N[kind]} values.")
    if q <= 0:
        return 1.0
    if q >= 1:
        return 0.0
    m = n - 2 - j
    u, wc, a, c, wt = _grid(nodes)
    t = a + q * (c - a)
    ft = special.ndtr(t)
    f_lo = np.clip(ft - u, 0, 1)
    f_hi = np.clip(wc - ft, 0, 1)
    inner = np.zeros_like(u)
    for k in range(i):
        if k > m:
            break
        inner += math.comb(m, k) * f_lo ** k * f_hi ** (m - k)
    const = math.exp(math.lgamma(n + 1) - math.lgamma(m + 1) - math.lgamma(j + 1))
    val = const * float(np.sum(wt * (1 - wc) ** j * inner))
    return float(min(1.0, max(0.0, val)))


@lru_cache(maxsize=2048)
def dixon_critical(n: int, alpha: float = 0.05, kind: str = "r10") -> float:
    """Upper one-tailed critical value r_crit with P(r > r_crit) = alpha."""
    f = lambda q: dixon_sf(q, n, kind) - alpha   # noqa: E731
    return float(brentq(f, 1e-9, 1 - 1e-12, xtol=1e-10))


def table_critical(n: int, alpha: float, kind: str) -> float | None:
    """Rorabacher's printed r10 value for one-tailed ``alpha`` (None when not tabulated)."""
    if kind != "r10" or n not in RORABACHER_R10:
        return None
    for k, a in enumerate(_RORA_ALPHA):
        if abs(a - alpha) < 1e-9:
            return RORABACHER_R10[n][k]
    return None


def auto_kind(n: int) -> str:
    if n <= 7:
        return "r10"
    if n <= 10:
        return "r11"
    if n <= 13:
        return "r21"
    return "r22"


def dixon_ratios(v: np.ndarray, kind: str) -> tuple[float, float]:
    """(ratio for the lowest value, ratio for the highest value) of sorted data ``v``."""
    i, j = DIXON_KINDS[kind]
    x = np.sort(v)
    n = len(x)
    lo_den = x[n - 1 - j] - x[0]
    hi_den = x[n - 1] - x[j]
    low = (x[i] - x[0]) / lo_den if lo_den > 0 else math.nan
    high = (x[n - 1] - x[n - 1 - i]) / hi_den if hi_den > 0 else math.nan
    return float(low), float(high)


@dataclass
class DixonResult:
    n: int
    kind: str
    statistic: float
    critical: float
    p: float
    end: str               # "low" | "high"
    index: int
    value: float
    side: str              # "two" | "min" | "max"
    alpha: float
    outlier: bool
    ratios: dict = field(default_factory=dict)
    source: str = "exact"   # where the critical value came from: "exact" | "table"


def dixon(x, alpha: float = 0.05, kind: str = "auto", side: str = "two",
          critical: str = "exact") -> DixonResult:
    """Dixon's Q test. ``side="two"`` tests the more extreme end (critical value at α/2 per
    tail, p = 2 × one-tailed, as in Rorabacher's two-tailed tables); ``"min"`` / ``"max"``
    test one end at α. ``critical="table"`` uses Rorabacher's printed value when it exists
    (r10, n ≤ 10, two-tailed α = 0.10 / 0.05 / 0.01), else the exact one."""
    allv = np.asarray(x, float)
    ok = np.flatnonzero(np.isfinite(allv))
    v = allv[ok]
    n = len(v)
    if n < 3:
        raise OutlierError("Dixon's Q test needs at least 3 values.")
    if n > MAX_N:
        raise OutlierError(f"Dixon's Q test is defined here for n ≤ {MAX_N}.")
    kind = auto_kind(n) if kind == "auto" else kind
    if kind not in DIXON_KINDS:
        raise OutlierError(f"Unknown Dixon statistic {kind!r}.")
    if n < DIXON_MIN_N[kind]:
        raise OutlierError(f"Dixon's {kind} needs at least {DIXON_MIN_N[kind]} values.")
    if not np.ptp(v) > 0:
        raise OutlierError("All values are equal: Dixon's test is undefined.")
    low, high = dixon_ratios(v, kind)
    if side == "min":
        end = "low"
    elif side == "max":
        end = "high"
    elif side == "two":
        end = "low" if (np.nan_to_num(low, nan=-1) >= np.nan_to_num(high, nan=-1)) else "high"
    else:
        raise OutlierError(f"Unknown alternative {side!r}.")
    r = low if end == "low" else high
    tail_alpha = alpha / 2 if side == "two" else alpha
    crit = dixon_critical(n, round(tail_alpha, 12), kind)
    source = "exact"
    if critical == "table":
        tab = table_critical(n, tail_alpha, kind)
        if tab is not None:
            crit, source = tab, "table"
    p1 = dixon_sf(r, n, kind) if np.isfinite(r) else math.nan
    p = min(1.0, 2 * p1) if side == "two" else p1
    k = int(np.argmin(v)) if end == "low" else int(np.argmax(v))
    return DixonResult(n, kind, r, crit, float(p), end, int(ok[k]), float(v[k]), side, alpha,
                       bool(np.isfinite(r) and r > crit), {"low": low, "high": high},
                       source)


def dixon_table(ns, alphas=(0.10, 0.05, 0.025, 0.01, 0.005), kind: str = "r10"):
    """Rows ``[n, crit(α₁), crit(α₂), ...]`` of one-tailed critical values."""
    return [[int(n)] + [dixon_critical(int(n), float(a), kind) for a in alphas] for n in ns]

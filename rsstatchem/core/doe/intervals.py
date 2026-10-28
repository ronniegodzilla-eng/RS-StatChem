"""Confidence, prediction and tolerance intervals of a fitted DOE model at any setting.

Everything is computed from the public attributes of :class:`~rsstatchem.core.doe.model.Fit`
(``predict_coded``, ``ms_res``, ``df_res``, ``transform``, ``space``), so the same code serves
ordinary least squares and REML split-plot fits. Intervals are formed on the analysis
(possibly transformed) scale and back-transformed to the original units.

* CI (mean):        ŷ ± t · s · √h
* PI (n runs):      ŷ ± t · s · √(1/n + h)
* TI (proportion P of the population with confidence γ):  ŷ ± k · s

with h = x'(X'X)⁻¹x and t the Student quantile on the residual degrees of freedom.
The tolerance factor k uses the effective sample size n* = 1/h (Young 2013, as in R's
``tolerance::regtol.int``): two-sided by Howe's (1969) approximation

    k₂ = z_{(1+P)/2} · √( f · (1 + 1/n*) / χ²_{1-γ, f} )

and one-sided exactly through the noncentral t distribution,

    k₁ = t'_{γ}(f, z_P · √n*) / √n*.

For a simple sample (h = 1/n, f = n - 1) these are the classical normal tolerance factors
(n = 10, P = 0.99, γ = 0.95: k₂ = 4.4446, Howe).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

KINDS = [("none", "Predicted value"), ("ci", "CI (mean)"), ("pi", "PI (one run)"),
         ("ti", "TI (population)")]
SIDES = [("two", "Two-sided"), ("lower", "One-sided lower"), ("upper", "One-sided upper")]

DEFAULT_LEVEL = 0.95        # confidence of CI / PI / TI
DEFAULT_PROPORTION = 0.99   # population proportion covered by the TI


def defaults() -> tuple[float, float]:
    """(confidence level, TI proportion): the analysis preferences when the analysis package
    exposes them (``model.PREFERENCES`` / ``model.preferences()``), else 95 % and 99 %."""
    from rsstatchem.core.doe import model

    prefs = getattr(model, "PREFERENCES", None)
    if prefs is None and callable(getattr(model, "preferences", None)):
        try:
            prefs = model.preferences()
        except Exception:  # noqa: BLE001 - a missing / different API falls back to defaults
            prefs = None
    level, prop = DEFAULT_LEVEL, DEFAULT_PROPORTION

    def get(name):
        if prefs is None:
            return None
        return prefs.get(name) if isinstance(prefs, dict) else getattr(prefs, name, None)

    for name in ("ci_level", "confidence", "level"):
        v = get(name)
        if isinstance(v, (int, float)) and 0 < v < 1:
            level = float(v)
            break
    for name in ("ti_proportion", "proportion"):
        v = get(name)
        if isinstance(v, (int, float)) and 0 < v < 1:
            prop = float(v)
            break
    return level, prop


# ---------------------------------------------------------------------------- k-factors
def k_factor(n, df, proportion: float = DEFAULT_PROPORTION, confidence: float = DEFAULT_LEVEL,
             sides: int = 2) -> np.ndarray:
    """Normal tolerance factor for (effective) sample size ``n`` and ``df`` degrees of
    freedom: Howe's approximation for two-sided, the exact noncentral t for one-sided."""
    n = np.asarray(n, float)
    if df <= 0:
        return np.full(n.shape, np.nan)
    if sides == 2:
        z = stats.norm.ppf((1 + proportion) / 2)
        chi = stats.chi2.ppf(1 - confidence, df)
        return z * np.sqrt(df * (1 + 1 / n) / chi)
    z = stats.norm.ppf(proportion)
    rn = np.sqrt(n)
    return stats.nct.ppf(confidence, df, z * rn) / rn


# ---------------------------------------------------------------------------- predictions
def _bias_mean(fit, z: np.ndarray) -> np.ndarray:
    """Predicted mean on the original scale with the second-order bias correction (the
    simple back-transform is the median), as in ``Fit.interval``."""
    tr = fit.transform
    median = tr.inverse(z)
    if tr.kind == "none":
        return median
    eps = 1e-4 * np.maximum(np.abs(z), 1)
    d2 = (tr.inverse(z + eps) - 2 * tr.inverse(z) + tr.inverse(z - eps)) / eps ** 2
    return median + 0.5 * d2 * fit.ms_res


def back_slope(fit, z: np.ndarray) -> np.ndarray:
    """|d y / d z| of the back-transform (1 without a transform), used to express a standard
    deviation on the analysis scale in original units (delta method)."""
    tr = fit.transform
    z = np.asarray(z, float)
    if tr.kind == "none":
        return np.ones_like(z)
    eps = 1e-5 * np.maximum(np.abs(z), 1)
    return np.abs((tr.inverse(z + eps) - tr.inverse(z - eps)) / (2 * eps))


def _quantile(fit, level: float, side: str) -> float:
    if fit.df_res <= 0:
        return float("nan")
    return float(stats.t.ppf(level if side != "two" else 0.5 + level / 2, fit.df_res))


def _back(fit, a, b):
    lo, hi = fit.transform.inverse(a), fit.transform.inverse(b)
    return np.minimum(lo, hi), np.maximum(lo, hi)


def predict(fit, actual: np.ndarray, level: float | None = None,
            proportion: float | None = None, side: str = "two", n_runs: int = 1,
            kinds: tuple = ("ci", "pi", "ti")) -> dict:
    """Prediction with CI, PI (for the mean of ``n_runs`` new runs) and TI at the actual
    settings ``actual`` (rows = points). ``side`` = two | lower | upper; for a one-sided
    interval the opposite bound is infinite. ``kinds`` limits the intervals computed."""
    if level is None or proportion is None:
        dl, dp = defaults()
        level = dl if level is None else level
        proportion = dp if proportion is None else proportion
    level, proportion = float(level), float(proportion)
    actual = np.atleast_2d(np.asarray(actual, float))
    z, h = fit.predict_coded(fit.space.to_coded(actual))
    s2 = fit.ms_res
    s = math.sqrt(s2)
    q = _quantile(fit, level, side)
    se_mean = np.sqrt(h * s2)
    se_pred = np.sqrt((1.0 / max(int(n_runs), 1) + h) * s2)
    k = np.full(np.shape(h), np.nan)
    if "ti" in kinds:
        n_eff = 1.0 / np.maximum(h, 1e-300)
        k = k_factor(n_eff, fit.df_res, proportion, level, 2 if side == "two" else 1)
    out = {"z": z, "h": h, "median": fit.transform.inverse(z), "mean": _bias_mean(fit, z),
           "sd": s, "se_mean": se_mean, "se_pred": se_pred, "k": k, "t": q,
           "level": level, "proportion": proportion, "side": side, "n_runs": int(n_runs)}
    for name, half in (("ci", q * se_mean), ("pi", q * se_pred), ("ti", k * s)):
        if name not in kinds:
            continue
        lo, hi = _back(fit, z - half, z + half)
        if side == "lower":
            hi = np.full_like(hi, np.inf)
        elif side == "upper":
            lo = np.full_like(lo, -np.inf)
        out[name] = (lo, hi)
    return out


def bounds(fit, actual: np.ndarray, kind: str, side: str = "two", level: float | None = None,
           proportion: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(lower, upper) of the CI / PI / TI on the original scale; kind 'none' gives the
    prediction itself for both."""
    if kind in (None, "", "none"):
        y = fit.predict(actual)
        return y, y
    return predict(fit, actual, level, proportion, side, kinds=(kind,))[kind]


def interval_label(kind: str, side: str, level: float, proportion: float) -> str:
    pct = f"{level * 100:g}%"
    side_txt = {"two": "", "lower": " one-sided low", "upper": " one-sided high"}[side]
    if kind == "ti":
        return f"{pct} TI for {proportion * 100:g}% of population{side_txt}"
    return f"{pct} {kind.upper()}{side_txt}"

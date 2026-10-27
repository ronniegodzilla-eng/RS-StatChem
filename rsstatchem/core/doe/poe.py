"""Propagation of error (POE): how variation in the factor settings is transmitted to a
response through the fitted model (robust design; Box & Fung; Myers, Montgomery &
Anderson-Cook, ch. 10).

With independent factor errors e_i ~ N(0, σ_i²) in actual units, a first-order Taylor
expansion of the fitted model f gives

    POE(x) = √( Σ_i (∂f/∂x_i)² σ_i² + σ²_resid ),

the standard deviation of a single observation at x. The optional second-order term
½ Σ_i Σ_j (∂²f/∂x_i ∂x_j)² σ_i² σ_j² makes the variance exact for a quadratic model with
normal factor errors. The model should be hierarchical (POE depends on the coding
otherwise). The derivatives are taken on the analysis (possibly transformed) scale; for a
transformed response the result is converted to original units with the slope of the
back-transform (delta method). Categoric factors carry no POE.

Factor standard deviations are stored with the design in ``doc.optimization["poe_sd"]`` as
{factor column: σ in actual units}.
"""
from __future__ import annotations

import math

import numpy as np

from rsstatchem.core.doe.intervals import back_slope

_STEP = 1e-3   # finite-difference step in coded units (exact enough for polynomials)


def sd_vector(space, poe_sd: dict | None) -> np.ndarray:
    """σ per factor (0 for categoric factors and factors without an entry); ``poe_sd`` may
    be keyed by factor column or by factor index."""
    out = np.zeros(space.k)
    for j, f in enumerate(space.factors):
        if f.categoric or not poe_sd:
            continue
        v = poe_sd.get(f.column, poe_sd.get(j, poe_sd.get(str(j))))
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v) and v > 0:
            out[j] = v
    return out


def has_sd(space, poe_sd: dict | None) -> bool:
    return bool(np.any(sd_vector(space, poe_sd) > 0))


def _coded_scale(space) -> np.ndarray:
    """d(coded)/d(actual) per factor."""
    out = np.zeros(space.k)
    for j, f in enumerate(space.factors):
        if f.categoric:
            continue
        out[j] = 1.0 / space.pseudo_span if space.mixture else 1.0 / f.half
    return out


def derivatives(fit, actual: np.ndarray, second: bool = False):
    """(z, gradient, Hessian or None) of the model on the analysis scale with respect to the
    actual factor settings; gradient is (m, k), Hessian (m, k, k)."""
    sp = fit.space
    actual = np.atleast_2d(np.asarray(actual, float))
    U = sp.to_coded(actual)
    m, k = U.shape
    scale = _coded_scale(sp)
    num = [j for j in range(k) if scale[j] > 0]

    # every perturbed copy of U is stacked and predicted in one call
    shifts: list[dict] = [{}]
    for j in num:
        shifts += [{j: _STEP}, {j: -_STEP}]
    if second:
        for a in num:
            for b in num:
                if b > a:
                    shifts += [{a: sa * _STEP, b: sb * _STEP} for sa in (1, -1)
                               for sb in (1, -1)]
    blocks = []
    for sh in shifts:
        V = U.copy()
        for j, d in sh.items():
            V[:, j] += d
        blocks.append(V)
    Z = fit.predict_coded(np.vstack(blocks))[0].reshape(len(shifts), m)
    z0 = Z[0]
    g = np.zeros((m, k))
    plus, minus = {}, {}
    pos = 1
    for j in num:
        plus[j], minus[j] = Z[pos], Z[pos + 1]
        pos += 2
        g[:, j] = (plus[j] - minus[j]) / (2 * _STEP) * scale[j]
    if not second:
        return z0, g, None
    H = np.zeros((m, k, k))
    for a in num:
        H[:, a, a] = (plus[a] - 2 * z0 + minus[a]) / _STEP ** 2 * scale[a] ** 2
    for a in num:
        for b in num:
            if b <= a:
                continue
            v11, v1m, vm1, vmm = Z[pos], Z[pos + 1], Z[pos + 2], Z[pos + 3]
            pos += 4
            hab = (v11 - v1m - vm1 + vmm) / (4 * _STEP ** 2)
            H[:, a, b] = H[:, b, a] = hab * scale[a] * scale[b]
    return z0, g, H


def poe_variance(fit, actual: np.ndarray, sd, second_order: bool = False,
                 residual: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(z, variance of a single observation on the analysis scale)."""
    sd = np.asarray(sd, float)
    z, g, H = derivatives(fit, actual, second_order)
    s2 = sd ** 2
    var = (g ** 2) @ s2
    if second_order and H is not None:
        var = var + 0.5 * np.einsum("mij,i,j->m", H ** 2, s2, s2)
    if residual:
        var = var + fit.ms_res
    return z, var


def poe(fit, actual: np.ndarray, sd, second_order: bool = False, residual: bool = True,
        original_units: bool = True) -> np.ndarray:
    """POE (a standard deviation) at each row of ``actual``. ``sd`` is a σ vector (one per
    factor, actual units) or the {column: σ} dictionary stored with the design."""
    if isinstance(sd, dict):
        sd = sd_vector(fit.space, sd)
    z, var = poe_variance(fit, actual, sd, second_order, residual)
    out = np.sqrt(np.clip(var, 0, None))
    if original_units:
        out = out * back_slope(fit, z)
    return out


def monte_carlo(fit, x0, sd, n: int = 200_000, seed: int | None = 0,
                residual: bool = True, original_units: bool = True) -> float:
    """Simulated SD of one observation at ``x0``: factor settings drawn from N(x0, σ²)
    pushed through the fitted model, plus N(0, σ²_resid) noise on the analysis scale
    (a check of :func:`poe`)."""
    if isinstance(sd, dict):
        sd = sd_vector(fit.space, sd)
    rng = np.random.default_rng(seed)
    x0 = np.asarray(x0, float)
    X = x0 + rng.standard_normal((n, len(x0))) * np.asarray(sd, float)
    z = fit.predict_coded(fit.space.to_coded(X))[0]
    if residual:
        z = z + rng.standard_normal(n) * math.sqrt(fit.ms_res)
    y = fit.transform.inverse(z) if original_units else z
    return float(np.std(y, ddof=1))


def surface(fit, gx: np.ndarray, gy: np.ndarray, axes: tuple[int, int], ref: np.ndarray, sd,
            second_order: bool = False) -> np.ndarray:
    """POE over a 2D grid of the factors ``axes`` (others at ``ref``): the POE graph."""
    i, j = axes
    X = np.tile(np.asarray(ref, float), (gx.size, 1))
    X[:, i], X[:, j] = gx.ravel(), gy.ravel()
    return poe(fit, X, sd, second_order).reshape(gx.shape)

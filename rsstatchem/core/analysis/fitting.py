"""Curve fitting as in OriginLab's Analysis > Fitting menu: Fit Linear, Fit Polynomial and
Nonlinear Curve Fit (NLFit), with parameter fixing, bounds and weighting (no weighting,
instrumental 1/σ², statistical 1/y, direct weights).

Core API (no Qt; also meant for the Graph Studio)::

    res = fit_curve(x, y, "ExpDec1")            # library name, FitFunction or "y = a*x^b"
    res.params, res.errors, res.predict(xx), res.confidence_band(xx, 0.95)

Linear-in-parameters models (Line, Poly1–9) are solved exactly by statsmodels WLS on a
column-equilibrated design; other models by ``scipy.optimize.least_squares`` (Levenberg-
Marquardt, or trust-region-reflective when bounds are set) started from the grid initial
guess of :mod:`functions`. Standard errors come from (JᵀWJ)⁻¹, scaled by the reduced χ² as
in Origin's default "Use Reduced Chi-Sqr" option. The report reproduces Origin's tables:
Parameters, Statistics, ANOVA (with Lack of Fit when x is replicated), Find X/Y and the
fitted-curve and residual plots.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, stats

from rsstatchem.core.analysis import functions as fl
from rsstatchem.core.analysis.common import (BAND, FIT, PRED, AnalysisResult, column,
                                             expand, figure, intf, num, png, pval, scatter,
                                             style, var_title)
from rsstatchem.core.analysis.functions import FitFunction
from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table

WEIGHTINGS = {"none": "No Weighting", "instrumental": "Instrumental (1/σ²)",
              "statistical": "Statistical (1/y)", "direct": "Direct Weight"}


# ---------------------------------------------------------------------- result
@dataclass
class FitResult:
    model: FitFunction
    x: np.ndarray
    y: np.ndarray
    weights: np.ndarray | None
    values: np.ndarray
    errors: np.ndarray
    cov: np.ndarray
    free: np.ndarray
    initial: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    rss: float
    tss: float
    uss: float
    success: bool = True
    message: str = ""
    nfev: int = 0
    scale_errors: bool = True
    weighting: str = "none"
    notes: list[str] = field(default_factory=list)
    n_active: int = 0                     # active linear constraints (reduce the free count)
    constraints: list[str] = field(default_factory=list)
    algorithm: str = "lm"

    # -- parameters
    @property
    def names(self) -> tuple[str, ...]:
        return self.model.params

    @property
    def params(self) -> dict[str, float]:
        return dict(zip(self.names, map(float, self.values)))

    @property
    def n(self) -> int:
        return len(self.x)

    @property
    def n_free(self) -> int:
        return int(self.free.sum())

    @property
    def dof(self) -> int:
        return self.n - self.n_free + self.n_active

    @property
    def k_eff(self) -> int:
        """Effective number of fitted parameters (free minus active constraints)."""
        return self.n_free - self.n_active

    # -- information criteria (Origin's Fit Comparison: K = parameters + 1)
    @property
    def aic(self) -> float:
        n, k = self.n, self.k_eff + 1
        return n * math.log(self.rss / n) + 2 * k if self.rss > 0 else float("-inf")

    @property
    def aicc(self) -> float:
        n, k = self.n, self.k_eff + 1
        return self.aic + 2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else float("nan")

    @property
    def bic(self) -> float:
        n, k = self.n, self.k_eff + 1
        return n * math.log(self.rss / n) + k * math.log(n) if self.rss > 0 \
            else float("-inf")

    def correlation(self) -> np.ndarray:
        """Parameter correlation matrix (free parameters; NaN rows for fixed ones)."""
        e = np.where(self.free & (self.errors > 0), self.errors, np.nan)
        with np.errstate(all="ignore"):
            return self.cov / np.outer(e, e)

    def dependency(self) -> np.ndarray:
        """Origin's Dependency = 1 − 1/(Cᵢᵢ·(C⁻¹)ᵢᵢ) of each free parameter."""
        idx = np.flatnonzero(self.free)
        out = np.full(len(self.values), np.nan)
        c = self.cov[np.ix_(idx, idx)]
        try:
            ci = np.linalg.pinv(c)
        except np.linalg.LinAlgError:
            return out
        with np.errstate(all="ignore"):
            out[idx] = 1 - 1 / (np.diag(c) * np.diag(ci))
        return out

    def derived(self) -> list[tuple[str, float, float]]:
        """Derived parameters of the model (FWHM, height, half-life, EC50…) with standard
        errors from the delta method."""
        out = []
        idx = np.flatnonzero(self.free)
        c = self.cov[np.ix_(idx, idx)]
        for name, fn in self.model.meta.get("derived", []):
            try:
                v = float(fn(self.params))
                g = np.empty(len(idx))
                for j, i in enumerate(idx):
                    h = 1e-6 * max(abs(self.values[i]), 1e-8)
                    up, dn = self.values.copy(), self.values.copy()
                    up[i] += h
                    dn[i] -= h
                    g[j] = (fn(dict(zip(self.names, up))) - fn(dict(zip(self.names, dn)))) / (2 * h)
                se = float(math.sqrt(max(g @ c @ g, 0.0)))
            except (ValueError, ZeroDivisionError, OverflowError, KeyError):
                continue
            out.append((name, v, se))
        return out

    @property
    def red_chisq(self) -> float:
        return self.rss / self.dof if self.dof > 0 else float("nan")

    @property
    def rmse(self) -> float:
        """Origin's "Root-MSE (SD)" = sqrt(Reduced Chi-Sqr)."""
        return math.sqrt(self.red_chisq) if self.dof > 0 else float("nan")

    @property
    def r2(self) -> float:
        """R-Square (COD) = 1 − RSS/TSS (weighted sums when weights are used)."""
        return 1 - self.rss / self.tss if self.tss > 0 else float("nan")

    @property
    def adj_r2(self) -> float:
        if self.dof <= 0 or self.tss <= 0 or self.n < 2:
            return float("nan")
        return 1 - (self.rss / self.dof) / (self.tss / (self.n - 1))

    @property
    def pearson_r(self) -> float:
        """Pearson's r for a straight line (sign of the slope × sqrt(COD))."""
        if self.model.meta.get("order") != 1:
            return float("nan")
        return math.copysign(math.sqrt(max(self.r2, 0.0)), self.values[1])

    @property
    def t_values(self) -> np.ndarray:
        with np.errstate(all="ignore"):
            return np.where(self.free & (self.errors > 0), self.values / self.errors, np.nan)

    @property
    def p_values(self) -> np.ndarray:
        if self.dof <= 0:
            return np.full(len(self.values), np.nan)
        return 2 * stats.t.sf(np.abs(self.t_values), self.dof)

    def ci(self, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        q = stats.t.ppf(0.5 + level / 2, self.dof) if self.dof > 0 else np.nan
        lo = np.where(self.free, self.values - q * self.errors, np.nan)
        hi = np.where(self.free, self.values + q * self.errors, np.nan)
        return lo, hi

    @property
    def at_bound(self) -> np.ndarray:
        tol = 1e-8 * (1 + np.abs(self.values))
        return self.free & ((np.abs(self.values - self.lower) < tol)
                            | (np.abs(self.values - self.upper) < tol))

    # -- curves
    def predict(self, x) -> np.ndarray:
        return self.model(np.asarray(x, float), *self.values)

    @property
    def fitted(self) -> np.ndarray:
        return self.predict(self.x)

    @property
    def residuals(self) -> np.ndarray:
        return self.y - self.fitted

    def gradient(self, x) -> np.ndarray:
        """∂f/∂p for the free parameters at ``x`` (exact for linear models)."""
        x = np.asarray(x, float)
        idx = np.flatnonzero(self.free)
        if self.model.basis is not None:
            return self.model.basis(x)[:, idx]
        return _num_grad(self.model, x, self.values, idx)

    def _sd_mean(self, x) -> np.ndarray:
        g = self.gradient(x)
        c = self.cov[np.ix_(self.free, self.free)]
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", g, c, g), 0.0))

    def _noise_var(self) -> float:
        base = self.red_chisq if (self.scale_errors or self.weights is None) else 1.0
        if self.weights is None:
            return base
        return base * float(np.mean(1.0 / self.weights))

    def _tq(self, level: float) -> float:
        return float(stats.t.ppf(0.5 + level / 2, self.dof)) if self.dof > 0 else float("nan")

    def confidence_band(self, x, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        yhat = self.predict(x)
        half = self._tq(level) * self._sd_mean(x)
        return yhat - half, yhat + half

    def prediction_band(self, x, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        yhat = self.predict(x)
        half = self._tq(level) * np.sqrt(self._sd_mean(x) ** 2 + self._noise_var())
        return yhat - half, yhat + half

    def find_x(self, y0: float, lo: float | None = None, hi: float | None = None) -> list[float]:
        """All x in [lo, hi] (default: data range) where the fitted curve equals ``y0``."""
        lo = float(np.min(self.x)) if lo is None else lo
        hi = float(np.max(self.x)) if hi is None else hi
        grid = np.linspace(lo, hi, 4001)
        d = self.predict(grid) - y0
        out = []
        for i in range(len(grid) - 1):
            a, b = d[i], d[i + 1]
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            if a == 0:
                out.append(float(grid[i]))
            elif a * b < 0:
                out.append(float(optimize.brentq(lambda t: float(self.predict([t])[0]) - y0,
                                                 grid[i], grid[i + 1], xtol=1e-14)))
        if d[-1] == 0:
            out.append(float(grid[-1]))
        return out

    # -- tables
    def anova(self) -> list[tuple[str, float, float, float, float, float]]:
        """(source, DF, SS, MS, F, Prob>F). Linear models with a free intercept use the
        corrected total (Origin Fit Linear); other models report Origin NLFit's
        Regression / Residual / Uncorrected Total / Corrected Total rows."""
        nan = float("nan")
        mse = self.red_chisq
        has_intercept = (self.model.basis is not None and "Intercept" in self.names
                         and self.free[self.names.index("Intercept")])
        if has_intercept:
            df_m = self.n_free - 1
            ss_m = self.tss - self.rss
            ms_m = ss_m / df_m if df_m > 0 else nan
            f = ms_m / mse if df_m > 0 and mse > 0 else nan
            p = stats.f.sf(f, df_m, self.dof) if np.isfinite(f) else nan
            return [("Model", df_m, ss_m, ms_m, f, p), ("Error", self.dof, self.rss, mse, nan, nan),
                    ("Total", self.n - 1, self.tss, nan, nan, nan)]
        df_m = self.n_free
        ss_m = self.uss - self.rss
        ms_m = ss_m / df_m if df_m else nan
        f = ms_m / mse if mse > 0 else nan
        p = stats.f.sf(f, df_m, self.dof) if np.isfinite(f) else nan
        return [("Regression", df_m, ss_m, ms_m, f, p),
                ("Residual", self.dof, self.rss, mse, nan, nan),
                ("Uncorrected Total", self.n, self.uss, nan, nan, nan),
                ("Corrected Total", self.n - 1, self.tss, nan, nan, nan)]

    def lack_of_fit(self) -> dict | None:
        """Lack-of-fit F test using pure error from replicated x values."""
        w = np.ones(self.n) if self.weights is None else self.weights
        keys, inv = np.unique(self.x, return_inverse=True, axis=0 if self.x.ndim > 1 else None)
        inv = inv.ravel()
        k = len(keys)
        df_pe = self.n - k
        df_lof = k - self.n_free
        if df_pe <= 0 or df_lof <= 0:
            return None
        sw = np.bincount(inv, w)
        mean = np.bincount(inv, w * self.y) / sw
        ss_pe = float(np.sum(w * (self.y - mean[inv]) ** 2))
        ss_lof = max(self.rss - ss_pe, 0.0)
        f = (ss_lof / df_lof) / (ss_pe / df_pe) if ss_pe > 0 else float("nan")
        p = float(stats.f.sf(f, df_lof, df_pe)) if np.isfinite(f) else float("nan")
        return {"df_lof": df_lof, "ss_lof": ss_lof, "df_pe": df_pe, "ss_pe": ss_pe, "F": f,
                "p": p}


def _num_grad(model: FitFunction, x, values, idx) -> np.ndarray:
    """∂f/∂p of the free parameters ``idx`` (analytic Jacobian when the model has one)."""
    return model.jacobian(x, values)[:, idx]


# ---------------------------------------------------------------------- constraints
@dataclass
class Constraint:
    """A general linear constraint ``a·p (= | <=) b`` in the parameters (Origin NLFit's
    Linear Constraints page), parsed from text such as ``A1 + A2 = 1`` or ``xc1 < xc2``."""
    text: str
    a: np.ndarray
    b: float
    equality: bool


def parse_constraints(texts, names: Sequence[str]) -> list[Constraint]:
    """Parse constraint lines (``;`` or newline separated). Each must be linear in the
    parameters: ``expr = expr``, ``expr < expr``, ``expr <= expr``, ``expr > expr`` or
    ``expr >= expr``."""
    import re

    if texts is None:
        return []
    if isinstance(texts, str):
        texts = [t for t in re.split(r"[;\n]", texts)]
    out = []
    for raw in texts:
        t = raw.strip()
        if not t:
            continue
        m = re.match(r"^(.*?)(<=|>=|==|=|<|>)(.*)$", t)
        if not m or not m.group(1).strip() or not m.group(3).strip():
            raise ValueError(f"Constraint '{t}': write e.g. 'A1 + A2 = 1' or 'xc1 < xc2'.")
        lhs, op, rhs = m.group(1), m.group(2), m.group(3)
        try:
            f = fl.parse_function(f"y = ({lhs}) - ({rhs})", ("_unused_x",))
        except fl.FunctionSyntaxError as exc:
            raise ValueError(f"Constraint '{t}': {exc}") from None
        unknown = set(f.params) - set(names)
        if unknown:
            raise ValueError(f"Constraint '{t}': unknown parameter(s) "
                             f"{', '.join(sorted(unknown))}.")
        k = [names.index(p) for p in f.params]
        dummy = np.zeros(1)
        p0 = np.zeros(len(f.params))
        c0 = float(f(dummy, *p0)[0])
        g = f.jacobian(dummy, p0)[0]
        rng = np.random.default_rng(7)
        for _ in range(3):
            pr = rng.normal(0, 3, len(f.params))
            if not np.isclose(float(f(dummy, *pr)[0]), c0 + g @ pr, rtol=1e-9, atol=1e-9):
                raise ValueError(f"Constraint '{t}' is not linear in the parameters.")
        a = np.zeros(len(names))
        a[k] = g
        if op in ("=", "=="):
            out.append(Constraint(t, a, -c0, True))
        elif op in ("<", "<="):
            out.append(Constraint(t, a, -c0, False))
        else:
            out.append(Constraint(t, -a, c0, False))
    return out


def _null_space(A: np.ndarray) -> np.ndarray:
    if A.shape[0] == 0:
        return np.eye(A.shape[1])
    U, s, Vt = np.linalg.svd(A)
    rank = int((s > 1e-12 * (s[0] if len(s) else 1)).sum())
    return Vt[rank:].T


# ---------------------------------------------------------------------- fitting
def resolve_model(model) -> FitFunction:
    if isinstance(model, FitFunction):
        return model
    if isinstance(model, str):
        f = fl.get_function(model)
        return f if f is not None else fl.parse_user_function(model)
    raise TypeError("model must be a FitFunction, a library name or a function text")


def make_weights(kind: str, y, yerr=None) -> np.ndarray | None:
    """Origin weighting methods → weights w_i (None for no weighting)."""
    y = np.asarray(y, float)
    if kind in (None, "none"):
        return None
    if kind == "instrumental":
        if yerr is None:
            raise ValueError("Instrumental weighting needs a Y Error (σ) column.")
        e = np.asarray(yerr, float)
        with np.errstate(divide="ignore"):
            return np.where(e > 0, 1.0 / e ** 2, np.nan)
    if kind == "statistical":
        with np.errstate(divide="ignore"):
            return np.where(y > 0, 1.0 / y, np.nan)
    if kind == "experimental":
        with np.errstate(divide="ignore"):
            return np.where(y != 0, 1.0 / y ** 2, np.nan)
    if kind in ("direct", "arbitrary"):
        if yerr is None:
            raise ValueError("Direct / arbitrary-dataset weighting needs a weight column.")
        return np.asarray(yerr, float)
    raise ValueError(f"Unknown weighting '{kind}'")


def _as_dict(values, names) -> dict:
    if values is None:
        return {}
    if isinstance(values, Mapping):
        unknown = set(values) - set(names)
        if unknown:
            raise ValueError(f"Unknown parameter(s): {', '.join(sorted(unknown))}")
        return {k: v for k, v in values.items() if v is not None}
    seq = list(values)
    if len(seq) != len(names):
        raise ValueError(f"Expected {len(names)} values, got {len(seq)}")
    return {k: v for k, v in zip(names, seq) if v is not None}


ALGORITHMS = {"auto": "Levenberg-Marquardt (Trust Region Reflective with bounds)",
              "lm": "Levenberg-Marquardt", "trf": "Trust Region Reflective",
              "simplex": "Downhill Simplex, then Levenberg-Marquardt"}


def _clean_xy(f: FitFunction, x, y, w):
    x = np.asarray(x, float)
    x = x.ravel() if f.nvar == 1 else x.reshape(-1, f.nvar)
    y = np.asarray(y, float).ravel()
    w = None if w is None else np.asarray(w, float).ravel()
    ok = (np.isfinite(x) if f.nvar == 1 else np.isfinite(x).all(axis=1)) & np.isfinite(y)
    if w is not None:
        ok &= np.isfinite(w) & (w > 0)
    return x[ok], y[ok], (None if w is None else w[ok])


def fit_curve(x, y, model, params0=None, bounds=None, fixed=None, weights=None, *,
              scale_errors: bool = True, weighting: str | None = None,
              max_nfev: int | None = None, constraints=None,
              algorithm: str = "auto") -> FitResult:
    """Fit ``model`` to (x, y).

    ``params0``: initial values (dict or sequence; missing → automatic guess).
    ``bounds``: {name: (lower, upper)} with None for an open side.
    ``fixed``: {name: value}, or names to keep at their initial value. When None, the
    parameters a library function fixes by default (e.g. the time offset x0 of ExpDecay1)
    are fixed at their guess.
    ``weights``: w_i per point (instrumental: 1/σ²). Non-finite / non-positive w drop the point.
    ``constraints``: linear constraints, e.g. ``["A1 + A2 = 1", "xc1 < xc2"]``.
    ``algorithm``: "auto" / "lm" / "trf" / "simplex" (Nelder-Mead pre-iteration, then L-M).
    For two-variable models (``model.nvar == 2``) ``x`` is an (n, 2) array.
    """
    f = resolve_model(model)
    names = f.params
    x, y, w = _clean_xy(f, x, y, weights)
    p0d = _as_dict(params0, names)
    if fixed is None and f.meta.get("fixed"):
        fixed = {k: (v if v is not None and k not in p0d else p0d.get(k))
                 for k, v in f.meta["fixed"].items()}
    if fixed is not None and not isinstance(fixed, Mapping):
        fixed = {k: p0d.get(k) for k in fixed}
    fixed = dict(fixed or {})
    unknown = set(fixed) - set(names)
    if unknown:
        raise ValueError(f"Unknown parameter(s): {', '.join(sorted(unknown))}")
    bnd = _as_dict(bounds, names)
    free = np.array([p not in fixed for p in names])
    n_free = int(free.sum())
    if n_free == 0:
        raise ValueError("All parameters are fixed; nothing to fit.")
    cons = parse_constraints(constraints, names)
    n_eq = sum(c.equality for c in cons)
    if len(x) <= n_free - n_eq:
        raise ValueError(f"Not enough data points: {len(x)} valid point(s) for {n_free} free "
                         f"parameter(s).")
    lower = np.array([(bnd.get(p) or (None, None))[0] for p in names], dtype=object)
    upper = np.array([(bnd.get(p) or (None, None))[1] for p in names], dtype=object)
    lower = np.array([-np.inf if v is None else float(v) for v in lower])
    upper = np.array([np.inf if v is None else float(v) for v in upper])
    if np.any(lower >= upper):
        raise ValueError("Each lower bound must be smaller than its upper bound.")

    if f.basis is not None and not cons:
        return _fit_linear(f, x, y, w, free, fixed, lower, upper, scale_errors, weighting)

    need_guess = any(p not in p0d and p not in fixed for p in names) or any(
        fixed.get(p) is None for p in fixed)
    guess = f.initial_guess(x, y, w) if need_guess else {}
    p0 = np.array([float(fixed[p]) if fixed.get(p) is not None else
                   float(p0d.get(p, guess.get(p, 1.0))) for p in names])
    for i in np.flatnonzero(free):
        if not lower[i] <= p0[i] <= upper[i]:
            lo, hi = lower[i], upper[i]
            p0[i] = (lo + hi) / 2 if np.isfinite(lo) and np.isfinite(hi) else \
                (lo + max(1.0, abs(lo)) * 0.1 if np.isfinite(lo) else hi - max(1.0, abs(hi)) * .1)
    sw = np.ones(len(y)) if w is None else np.sqrt(w)
    idx = np.flatnonzero(free)

    def resid(pf):
        p = p0.copy()
        p[idx] = pf
        r = (f(x, *p) - y) * sw
        return np.where(np.isfinite(r), r, 1e10)

    def jac(pf):
        p = p0.copy()
        p[idx] = pf
        J = f.jacobian(x, p)[:, idx] * sw[:, None]
        return np.where(np.isfinite(J), J, 0.0)

    jac_arg = jac if f.jac is not None else "2-point"
    starts = [p0[idx]]
    if f.user_defined and any(p not in p0d for p in names if p not in fixed):
        starts += _multistart(f, x, y, p0, idx, resid)
    if algorithm == "simplex":
        starts = [_simplex(resid, s) for s in starts]
    bounded = np.any(np.isfinite(lower[idx])) or np.any(np.isfinite(upper[idx]))
    method = "trf" if bounded or algorithm == "trf" or len(y) < n_free else "lm"
    best = None
    for s in starts:
        try:
            r = optimize.least_squares(
                resid, s, jac=jac_arg, method=method,
                bounds=(lower[idx], upper[idx]) if bounded else (-np.inf, np.inf),
                x_scale="jac", ftol=1e-12, xtol=1e-12, gtol=1e-12,
                max_nfev=max_nfev or 1000 * (n_free + 1))
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue
        if best is None or (np.isfinite(r.cost) and r.cost < best.cost):
            best = r
    if best is None:
        raise ValueError("The fit failed to start; check the initial parameter values.")
    if np.all(np.isfinite(best.fun)) and best.cost < 1e19:   # polish (tighter, better J)
        try:
            r2 = optimize.least_squares(
                resid, best.x, jac=jac if f.jac is not None else "3-point", method=method,
                bounds=(lower[idx], upper[idx]) if bounded else (-np.inf, np.inf),
                x_scale="jac", ftol=1e-15, xtol=1e-15, gtol=1e-15,
                max_nfev=max_nfev or 200 * (n_free + 1))
            if np.isfinite(r2.cost) and r2.cost <= best.cost:
                r2.nfev += best.nfev
                r2.status = best.status if r2.status == 0 and best.status > 0 else r2.status
                best = r2
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            pass
    values = p0.copy()
    values[idx] = best.x
    message = _status(best)
    success = bool(best.success) and np.all(np.isfinite(best.fun)) and best.cost < 1e19
    cov_free, n_active, cnotes = None, 0, []
    if cons:
        values, cov_free, n_active, cnotes, ok = _constrained(
            f, x, y, w, values, free, lower, upper, cons, scale_errors)
        success = success and ok
        message = "Fit converged with linear constraints." if ok else \
            "The constrained fit did not converge."
    if f.canonical is not None and not cons:
        canon = f.canonical(dict(zip(names, values)))
        values = np.array([canon[p] for p in names], float)
    res = _finish(f, x, y, w, values, free, p0, lower, upper, scale_errors, weighting,
                  success, message, int(best.nfev), cov_free=cov_free, n_active=n_active)
    res.constraints = [c.text for c in cons]
    res.notes += cnotes
    res.algorithm = algorithm if algorithm != "auto" else ("trf" if bounded else "lm")
    return res


def _simplex(resid, s) -> np.ndarray:
    """Nelder-Mead (Origin's Downhill Simplex) on the sum of squares."""
    def cost(p):
        r = resid(p)
        return float(r @ r)
    try:
        out = optimize.minimize(cost, s, method="Nelder-Mead",
                                options={"maxiter": 400 * (len(s) + 1), "xatol": 1e-10,
                                         "fatol": 1e-14, "adaptive": len(s) > 3})
        return out.x if np.isfinite(out.fun) and out.fun <= cost(s) else s
    except (ValueError, FloatingPointError):
        return s


def _constrained(f, x, y, w, values, free, lower, upper, cons, scale_errors):
    """Minimise χ² subject to the linear constraints (SLSQP), then refine on the active set
    with L-M in the null-space parametrisation; the covariance is projected on that space."""
    idx = np.flatnonzero(free)
    sw = np.ones(len(y)) if w is None else np.sqrt(w)
    fixed_vals = values.copy()
    A = np.array([c.a for c in cons])
    b = np.array([c.b for c in cons]) - A[:, ~free] @ values[~free]
    Af = A[:, idx]
    eq = np.array([c.equality for c in cons])

    def full(pf):
        p = fixed_vals.copy()
        p[idx] = pf
        return p

    def resid(pf):
        r = (f(x, *full(pf)) - y) * sw
        return np.where(np.isfinite(r), r, 1e10)

    def jac(pf):
        J = f.jacobian(x, full(pf))[:, idx] * sw[:, None]
        return np.where(np.isfinite(J), J, 0.0)

    start = values[idx].copy()
    viol = Af @ start - b
    if not (np.all(np.abs(viol[eq]) < 1e-10 * (1 + np.abs(b[eq]))) and
            np.all(viol[~eq] <= 1e-12 * (1 + np.abs(b[~eq])))):
        scons = []
        if eq.any():
            scons.append({"type": "eq", "fun": lambda p: Af[eq] @ p - b[eq],
                          "jac": lambda p: Af[eq]})
        if (~eq).any():
            scons.append({"type": "ineq", "fun": lambda p: b[~eq] - Af[~eq] @ p,
                          "jac": lambda p: -Af[~eq]})
        bnds = list(zip(np.where(np.isfinite(lower[idx]), lower[idx], None),
                        np.where(np.isfinite(upper[idx]), upper[idx], None)))

        def cost(p):
            r = resid(p)
            return 0.5 * float(r @ r)

        def grad(p):
            return jac(p).T @ resid(p)
        try:
            out = optimize.minimize(cost, start, jac=grad, method="SLSQP", bounds=bnds,
                                    constraints=scons,
                                    options={"maxiter": 2000, "ftol": 1e-16})
            start = out.x
        except (ValueError, np.linalg.LinAlgError):
            pass
    viol = Af @ start - b
    tol = 1e-7 * (1 + np.abs(b)) + 1e-9 * np.abs(Af).sum(axis=1) * (1 + np.abs(start).max())
    active = eq | (np.abs(viol) <= tol)
    ok = bool(np.all(np.abs(viol[eq]) <= tol[eq]) and np.all(viol[~eq] <= tol[~eq]))
    N = _null_space(Af[active])
    base = start.copy()
    if active.any():   # move exactly onto the active constraints
        Aa, ba = Af[active], b[active]
        base = base - np.linalg.pinv(Aa) @ (Aa @ base - ba)
    if N.shape[1] > 0:
        try:
            r = optimize.least_squares(lambda z: resid(base + N @ z), np.zeros(N.shape[1]),
                                       jac=lambda z: jac(base + N @ z) @ N,
                                       method="lm" if len(y) >= N.shape[1] else "trf",
                                       x_scale="jac", ftol=1e-15, xtol=1e-15, gtol=1e-15)
            cand = base + N @ r.x
            if np.all(Af[~active] @ cand - b[~active] <= tol[~active]) and \
                    np.all(lower[idx] <= cand) and np.all(cand <= upper[idx]):
                base = cand
        except (ValueError, np.linalg.LinAlgError):
            pass
    values = fixed_vals.copy()
    values[idx] = base
    J = jac(base)
    rr = resid(base)
    n_active = int(np.linalg.matrix_rank(Af[active])) if active.any() else 0
    dof = len(y) - len(idx) + n_active
    if N.shape[1] > 0:
        cov = N @ _cov_from_jac(J @ N) @ N.T
    else:
        cov = np.zeros((len(idx), len(idx)))
    if scale_errors and dof > 0:
        cov = cov * float(rr @ rr) / dof
    notes = []
    act = [c.text for c, a in zip(cons, active) if a and not c.equality]
    if act:
        notes.append("Active inequality constraint(s): " + "; ".join(act) + ".")
    return values, cov, n_active, notes, ok


def _multistart(f, x, y, p0, idx, resid) -> list[np.ndarray]:
    """Extra starting points for user functions (no analytic guess available)."""
    rng = np.random.default_rng(12345)
    xs = x if x.ndim == 1 else x[:, 0]
    scales = np.array([1.0, -1.0, 0.1, 10.0, float(np.ptp(xs) or 1),
                       1 / float(np.ptp(xs) or 1), float(np.mean(np.abs(y)) or 1),
                       float(np.median(y)), float(np.min(y)), float(np.max(y)), 0.5, 2.0])
    cands = [scales[rng.integers(0, len(scales), len(idx))] for _ in range(80)]
    cands += [np.full(len(idx), s) for s in scales]
    scored = []
    for c in cands:
        r = resid(c)
        cost = float(r @ r)
        if np.isfinite(cost) and cost < 1e18:
            scored.append((cost, c))
    scored.sort(key=lambda t: t[0])
    return [c for _, c in scored[:6]]


def _status(r) -> str:
    texts = {1: "Fit converged. Gradient tolerance was reached.",
             2: "Fit converged. Chi-Sqr tolerance value was reached.",
             3: "Fit converged. Parameter tolerance was reached.",
             4: "Fit converged. Chi-Sqr and parameter tolerances were reached.",
             0: "Maximum number of iterations reached; the fit did not converge."}
    return texts.get(r.status, r.message)


def _cov_from_jac(J: np.ndarray) -> np.ndarray:
    J = np.where(np.isfinite(J), J, 0.0)
    try:
        U, s, Vt = np.linalg.svd(J, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.full((J.shape[1], J.shape[1]), np.nan)
    tol = np.finfo(float).eps * max(J.shape) * (s[0] if len(s) else 0)
    inv = np.where(s > tol, 1 / np.where(s > 0, s, 1) ** 2, 0.0)
    return (Vt.T * inv) @ Vt


def _finish(f, x, y, w, values, free, p0, lower, upper, scale_errors, weighting, success,
            message, nfev, cov_free=None, n_active: int = 0) -> FitResult:
    ww = np.ones(len(y)) if w is None else w
    with np.errstate(all="ignore"):
        r = y - f(x, *values)
    rss = float(np.sum(ww * r ** 2))
    ybar = float(np.sum(ww * y) / np.sum(ww))
    tss = float(np.sum(ww * (y - ybar) ** 2))
    uss = float(np.sum(ww * y ** 2))
    idx = np.flatnonzero(free)
    dof = len(y) - len(idx) + n_active
    if cov_free is None:
        sw = np.sqrt(ww)
        g = f.basis(x)[:, idx] if f.basis is not None else _num_grad(f, x, values, idx)
        cov_free = _cov_from_jac(g * sw[:, None])
        if scale_errors and dof > 0:
            cov_free = cov_free * (rss / dof)
    cov = np.zeros((len(values), len(values)))
    cov[np.ix_(idx, idx)] = cov_free
    errors = np.sqrt(np.maximum(np.diag(cov), 0.0))
    errors[~free] = np.nan
    notes = []
    if not success:
        notes.append(message)
    res = FitResult(f, x, y, w, values, errors, cov, free, p0, lower, upper, rss, tss, uss,
                    success, message, nfev, scale_errors, weighting or
                    ("none" if w is None else "direct"), notes, n_active=n_active)
    if np.any(res.at_bound):
        notes.append("Parameter(s) " + ", ".join(np.array(f.params)[res.at_bound])
                     + " reached a bound; their standard errors are not reliable.")
    return res


def _fit_linear(f, x, y, w, free, fixed, lower, upper, scale_errors, weighting) -> FitResult:
    """Linear least squares on a column-equilibrated design via Householder QR (``lstsq``);
    with very ill-conditioned designs (NIST Filip, Wampler5) one step of iterative refinement
    on the residual recovers the lost digits."""
    B = f.basis(x)
    vals = np.array([float(fixed.get(p) or 0.0) if p in fixed else 0.0 for p in f.params])
    yy = y - B[:, ~free] @ vals[~free]
    Bf = B[:, free]
    sw = np.ones(len(y)) if w is None else np.sqrt(w)
    A = Bf * sw[:, None]
    scale = np.sqrt((A ** 2).sum(axis=0))
    if np.any(scale == 0):
        raise ValueError("The X values do not vary enough to fit this model.")
    As = A / scale
    rhs = yy * sw
    Q, R = np.linalg.qr(As)
    coef = _solve_triangular(R, Q.T @ rhs)
    for _ in range(3):   # iterative refinement
        res_ = rhs - As @ coef
        coef = coef + _solve_triangular(R, Q.T @ res_)
    dof = len(y) - int(free.sum())
    rank = np.linalg.matrix_rank(As)
    Rinv = np.linalg.pinv(R) if rank < As.shape[1] else _solve_triangular(R, np.eye(len(R)))
    ncov = (Rinv @ Rinv.T) / np.outer(scale, scale)
    coef = coef / scale
    if rank == As.shape[1] and A.size <= 4000 and A.shape[1] <= 16:
        # small full-rank designs: the exact least-squares solution of the (float) data, so
        # ill-conditioned polynomials (NIST Wampler, Filip) are correct to the last digit on
        # every platform, independent of the BLAS behind numpy
        coef, ncov = _exact_least_squares(A, rhs)
    rr = rhs - A @ coef
    s2 = float(rr @ rr) / dof if dof > 0 else float("nan")
    cov = ncov * s2 if scale_errors and dof > 0 else ncov
    vals[free] = coef
    if np.any(np.isfinite(lower[free]) | np.isfinite(upper[free])):
        clipped = np.clip(vals, lower, upper)
        if not np.allclose(clipped, vals):
            raise ValueError("Bounds are not supported for linear models; fix the parameter "
                             "instead.")
    ok = rank == Bf.shape[1]
    msg = "Linear least squares: exact solution." if ok else \
        "The design is rank deficient; some parameters are not identifiable."
    return _finish(f, x, y, w, vals, free, vals.copy(), lower, upper, scale_errors, weighting,
                   ok, msg, 0, cov_free=cov)


def _exact_least_squares(A, b):
    """Solve the normal equations A'A beta = A'b in exact rational arithmetic (float inputs
    are exactly representable as fractions) and return beta and (A'A)^-1 as floats."""
    from fractions import Fraction

    n, p = A.shape
    Af = [[Fraction(float(v)) for v in row] for row in A]
    bf = [Fraction(float(v)) for v in b]
    M = [[sum((Af[k][i] * Af[k][j] for k in range(n)), Fraction(0)) for j in range(p)]
         for i in range(p)]
    v = [sum((Af[k][i] * bf[k] for k in range(n)), Fraction(0)) for i in range(p)]
    # Gauss-Jordan on [M | v | I]
    aug = [M[i] + [v[i]] + [Fraction(int(i == j)) for j in range(p)] for i in range(p)]
    for c in range(p):
        piv = next(r for r in range(c, p) if aug[r][c] != 0)
        aug[c], aug[piv] = aug[piv], aug[c]
        inv = 1 / aug[c][c]
        aug[c] = [x * inv for x in aug[c]]
        for r in range(p):
            if r != c and aug[r][c] != 0:
                f = aug[r][c]
                aug[r] = [x - f * y for x, y in zip(aug[r], aug[c])]
    beta = np.array([float(aug[i][p]) for i in range(p)])
    cov = np.array([[float(aug[i][p + 1 + j]) for j in range(p)] for i in range(p)])
    return beta, cov


def _solve_triangular(R, b):
    from scipy.linalg import solve_triangular
    return solve_triangular(R, b)


def mlr_function(k: int, intercept: bool = True, names: Sequence[str] | None = None) \
        -> FitFunction:
    """Origin's Multiple Linear Regression model y = b0 + b1·x1 + … + bk·xk (x: (n, k))."""
    params = (("Intercept",) if intercept else ()) + tuple(
        names or [f"b{i}" for i in range(1, k + 1)])

    def basis(x, k=k):
        x = np.asarray(x, float).reshape(-1, k)
        return np.column_stack(([np.ones(len(x))] if intercept else []) + [x[:, i]
                                                                         for i in range(k)])

    def func(x, *p):
        return basis(x) @ np.asarray(p, float)

    terms = " + ".join(f"{n}*x{i}" for i, n in enumerate(params[int(intercept):], 1))
    return FitFunction("MLR", "Multiple Variables",
                       f"y = {'Intercept + ' if intercept else ''}{terms}", params, func,
                       basis=basis, nvar=k,
                       description=f"Multiple linear regression on {k} variable(s).",
                       meta={"order": None, "mlr": k})


def fit_mlr(X, y, intercept: bool = True, weights=None, names=None,
            scale_errors: bool = True) -> FitResult:
    X = np.asarray(X, float)
    X = X.reshape(len(X), -1)
    return fit_curve(X, y, mlr_function(X.shape[1], intercept, names), weights=weights,
                     scale_errors=scale_errors)


# ---------------------------------------------------------------------- report
@dataclass
class FitOptions:
    level: float = 0.95
    conf_band: bool = True
    pred_band: bool = False
    residual_plot: bool = True
    anova: bool = True
    lack_of_fit: bool = True
    find_x: Sequence[float] = ()
    find_y: Sequence[float] = ()
    save_fitted: bool = False
    save_residuals: bool = False
    scale_errors: bool = True


def parameter_table(res: FitResult, yname: str, level: float = 0.95,
                    title: str = "Parameters") -> Table:
    lo, hi = res.ci(level)
    t, p = res.t_values, res.p_values
    rows = []
    bound = res.at_bound
    for i, name in enumerate(res.names):
        if not res.free[i]:
            rows.append([yname if i == 0 else "", name, num(res.values[i]), "0", "--", "--",
                         "--", "--"])
            continue
        rows.append([yname if i == 0 else "", name + (" ᵇ" if bound[i] else ""),
                     num(res.values[i]), num(res.errors[i]), num(t[i], 5), pval(p[i]),
                     num(lo[i]), num(hi[i])])
    notes = []
    if (~res.free).any():
        notes.append("Fixed parameters are shown with Standard Error 0, as in Origin.")
    if bound.any():
        notes.append("b. The parameter reached a bound.")
    if res.scale_errors:
        notes.append("Standard errors are scaled with the square root of Reduced Chi-Sqr.")
    pc = f"{level * 100:g}%"
    return Table(title, ["", "", "Value", "Standard Error", "t-Value", "Prob>|t|",
                         (f"{pc} Confidence Interval", "LCL"),
                         (f"{pc} Confidence Interval", "UCL")], rows, row_header_cols=2,
                 footnotes=notes)


def statistics_table(res: FitResult, yname: str) -> Table:
    rows = [["Number of Points", intf(res.n)], ["Degrees of Freedom", intf(res.dof)],
            ["Reduced Chi-Sqr", num(res.red_chisq)],
            ["Residual Sum of Squares", num(res.rss)], ["R-Square (COD)", num(res.r2)],
            ["Adj. R-Square", num(res.adj_r2)], ["Root-MSE (SD)", num(res.rmse)]]
    if res.model.meta.get("order") == 1:
        rows.insert(4, ["Pearson's r", num(res.pearson_r)])
    if res.nfev:
        rows.append(["Number of Function Evaluations", intf(res.nfev)])
    rows.append(["Fit Status", "Succeeded" if res.success else "Failed"])
    notes = ["R-Square (COD) = 1 − RSS/TSS; Adj. R-Square = 1 − (RSS/dof)/(TSS/(n − 1))."]
    if res.weights is not None:
        notes.append("Sums of squares are weighted.")
    return Table("Statistics", ["", yname], rows, footnotes=notes)


def anova_table(res: FitResult, yname: str, lack_of_fit: bool = True) -> Table:
    rows = []
    for i, (src, df, ss, ms, f, p) in enumerate(res.anova()):
        rows.append([yname if i == 0 else "", src, intf(df), num(ss), num(ms) if
                     np.isfinite(ms) else "", num(f) if np.isfinite(f) else "",
                     pval(p) if np.isfinite(p) else ""])
    lof = res.lack_of_fit() if lack_of_fit else None
    if lof:
        err_row = 2 if rows[1][1] in ("Error", "Residual") else None
        ins = [["", "  Lack of Fit", intf(lof["df_lof"]), num(lof["ss_lof"]),
                num(lof["ss_lof"] / lof["df_lof"]), num(lof["F"]), pval(lof["p"])],
               ["", "  Pure Error", intf(lof["df_pe"]), num(lof["ss_pe"]),
                num(lof["ss_pe"] / lof["df_pe"]), "", ""]]
        rows[err_row:err_row] = ins
    p_model = res.anova()[0][5]
    notes = []
    if np.isfinite(p_model):
        notes.append("At the 0.05 level, the fitting function is "
                     + ("significantly" if p_model < 0.05 else "not significantly")
                     + " better than the function y = constant.")
    if lof:
        notes.append("Lack of Fit uses the pure error from replicated X values.")
    return Table("ANOVA", ["", "", "DF", "Sum of Squares", "Mean Square", "F Value", "Prob>F"],
                 rows, row_header_cols=2, footnotes=notes)


def fit_plot(res: FitResult, xlabel: str, ylabel: str, level: float = 0.95,
             conf: bool = True, pred: bool = False, yerr=None, title: str = "",
             components: list[tuple[str, np.ndarray]] | None = None,
             grid: np.ndarray | None = None) -> bytes:
    fig = figure(6.2, 4.2)
    ax = fig.add_subplot()
    lo, hi = float(np.min(res.x)), float(np.max(res.x))
    log_x = lo > 0 and hi / lo >= 1e3
    if grid is not None:
        xx = grid
    else:
        xx = np.geomspace(lo, hi, 600) if log_x else np.linspace(lo, hi, 600)
    if pred:
        a, b = res.prediction_band(xx, level)
        ax.fill_between(xx, a, b, color=PRED, alpha=0.25, lw=0,
                        label=f"{level * 100:g}% Prediction Band")
    if conf:
        a, b = res.confidence_band(xx, level)
        ax.fill_between(xx, a, b, color=BAND, alpha=0.55, lw=0,
                        label=f"{level * 100:g}% Confidence Band")
    if yerr is not None:
        ax.errorbar(res.x, res.y, yerr=yerr, fmt="none", ecolor="#777", elinewidth=0.8,
                    capsize=2, zorder=2)
    scatter(ax, res.x, res.y, label=ylabel)
    if components:
        from rsstatchem.core.analysis.common import PALETTE
        for i, (label, yc) in enumerate(components):
            ax.plot(xx, yc, color=PALETTE[i % len(PALETTE)], lw=1.0, ls="--", label=label)
    ax.plot(xx, res.predict(xx), color=FIT, lw=1.8, label="Fit Curve", zorder=4)
    style(ax, title or f"{res.model.name} Fit of {ylabel}", xlabel, ylabel)
    if log_x:
        ax.set_xscale("log")
    ax.legend(fontsize=8, frameon=False, loc="best")
    return png(fig)


def residual_plot(res: FitResult, xlabel: str, ylabel: str) -> bytes:
    fig = figure(6.2, 2.8)
    ax = fig.add_subplot()
    r = res.residuals
    ax.axhline(0, color=FIT, lw=1)
    if len(r) <= 200:
        ax.vlines(res.x, 0, r, color="#bbbbbb", lw=0.6)
    scatter(ax, res.x, r, label="Residual")
    style(ax, "Residual vs. Independent", xlabel, f"Residual of {ylabel}")
    lo, hi = float(np.min(res.x)), float(np.max(res.x))
    if lo > 0 and hi / lo >= 1e3:
        ax.set_xscale("log")
    return png(fig)


def notes_table(res: FitResult, xname: str, yname: str, extra: list[tuple[str, str]] = ()):
    m = res.model
    rows = [["Model", m.name], ["Equation", m.formula], ["Description", m.description],
            ["Input Data", f"X = {xname}, Y = {yname}"],
            ["Weight", WEIGHTINGS.get(res.weighting, res.weighting)]]
    if m.basis is None:
        rows.append(["Iteration Algorithm", "Levenberg-Marquardt" if not np.any(
            np.isfinite(res.lower[res.free]) | np.isfinite(res.upper[res.free]))
                     else "Trust Region Reflective (bounded)"])
    rows += [list(r) for r in extra]
    rows.append(["Fit Status", res.message])
    return Table("Notes", ["", "Value"], rows)


def find_tables(res: FitResult, yname: str, find_x: Iterable[float] = (),
                find_y: Iterable[float] = ()) -> list[Table]:
    out = []
    xs = list(find_y)
    if xs:
        rows = [[num(v), num(float(res.predict([v])[0]))] for v in xs]
        out.append(Table("Find Y from X", ["X Value", f"{yname} Value"], rows,
                         row_header_cols=0))
    ys = list(find_x)
    if ys:
        rows = []
        span = float(np.ptp(res.x)) or 1.0
        for v in ys:
            sols = res.find_x(v)
            note = ""
            if not sols:
                sols = res.find_x(v, float(np.min(res.x)) - span, float(np.max(res.x)) + span)
                note = " (extrapolated)" if sols else ""
            rows.append([num(v), ", ".join(num(s) for s in sols) + note if sols else
                         "no solution in range"])
        out.append(Table("Find X from Y", [f"{yname} Value", "X Value"], rows,
                         row_header_cols=0))
    return out


def fit_section(res: FitResult, title: str, xname: str, yname: str, log: str,
                opts: FitOptions, yerr=None, extra_notes: list[tuple[str, str]] = ()) \
        -> Section:
    section = Section(title)
    section.log(log)
    section.table(notes_table(res, xname, yname, extra_notes))
    for note in res.notes:
        section.note("Warning", note, style="warning")
    section.table(parameter_table(res, yname, opts.level))
    section.table(statistics_table(res, yname))
    if opts.anova and res.dof > 0:
        section.table(anova_table(res, yname, opts.lack_of_fit))
    for t in find_tables(res, yname, opts.find_x, opts.find_y):
        section.table(t)
    section.image("Fitted Curves Plot", fit_plot(res, xname, yname, opts.level,
                                                 opts.conf_band, opts.pred_band, yerr))
    if opts.residual_plot:
        section.image("Residual vs. Independent Plot", residual_plot(res, xname, yname),
                      width=560)
    return section


# ---------------------------------------------------------------------- dataset wrappers
def _data(ds: Dataset, xname: str, yname: str, ename: str | None, weighting: str):
    frame = pd.DataFrame({"x": column(ds, xname), "y": column(ds, yname)})
    if ename:
        frame["e"] = column(ds, ename)
    frame = frame.dropna()
    e = frame["e"].to_numpy() if ename else None
    w = make_weights(weighting, frame["y"].to_numpy(), e)
    if w is not None:
        keep = np.isfinite(w) & (w > 0)
        dropped = int((~keep).sum())
        frame, w = frame[keep], w[keep]
        e = e[keep] if e is not None else None
    else:
        dropped = 0
    return frame, w, e, dropped


def run_fit(ds: Dataset, xname: str, yname: str, model, *, ename: str | None = None,
            weighting: str = "none", params0=None, bounds=None, fixed=None,
            opts: FitOptions | None = None, title: str | None = None,
            command: str = "NLFIT") -> AnalysisResult:
    """Fit a model to two Dataset variables and build the Origin-style report."""
    opts = opts or FitOptions()
    f = resolve_model(model)
    frame, w, e, dropped = _data(ds, xname, yname, ename, weighting)
    res = fit_curve(frame["x"].to_numpy(), frame["y"].to_numpy(), f, params0=params0,
                    bounds=bounds, fixed=fixed, weights=w, scale_errors=opts.scale_errors,
                    weighting=weighting)
    xt, yt = var_title(ds, xname), var_title(ds, yname)
    log = (f"{command} X={xname} Y={yname} FUNCTION={f.name}"
           + (f" WEIGHT={weighting.upper()}({ename})" if weighting != "none" else "")
           + (f" FIXED=({', '.join(f'{k}={v:g}' for k, v in (fixed or {}).items())})"
              if fixed else "") + ".")
    if f.user_defined:
        log += f"\n  /EQUATION '{f.meta['source']}'."
    extra = []
    if dropped:
        extra.append(("Excluded Points", f"{dropped} point(s) with invalid weight"))
    section = fit_section(res, title or f"{'Nonlinear Curve Fit' if f.basis is None else 'Fit'} "
                          f"({f.name})", xt, yt, log, opts,
                          yerr=e if weighting == "instrumental" else None, extra_notes=extra)
    out = AnalysisResult(section)
    if opts.save_fitted:
        xall = column(ds, xname)
        out.saved["FIT"] = (f"Fitted {yt} ({f.name})",
                            pd.Series(res.predict(xall.to_numpy()), index=ds.df.index))
    if opts.save_residuals:
        out.saved["RES"] = (f"Residual of {yt} ({f.name})",
                            expand(res.residuals, frame.index, ds))
    return out

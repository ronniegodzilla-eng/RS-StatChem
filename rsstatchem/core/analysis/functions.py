"""Fitting-function library modelled on OriginLab's built-in functions (Exponential,
Growth/Sigmoidal, Peak Functions, Power, Rational, Hyperbola, Chemical Kinetics, adsorption
isotherms) plus polynomials and a safe parser for user-defined functions.

Every function knows which parameters enter linearly. Initial values are found by a coarse
grid over the nonlinear parameters with the linear ones solved by least squares at each grid
point (variable projection); the full nonlinear fit then refines that start. This replaces
Origin's per-function "parameter initialization" code with one robust mechanism.

User-defined functions are typed as text, e.g. ``y = A*exp(-x/t) + y0``. The text is parsed
with :mod:`ast`, every node is checked against a whitelist and compiled into closures — no
``eval``. As in Origin, ``ln()`` is the natural logarithm and ``log()`` is base 10.
"""
from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
from typing import Callable

import numpy as np
from scipy import special

R_GAS = 8.314462618  # J mol^-1 K^-1
_LN2 = math.log(2.0)
_GRID_MAX = 3000


@dataclass
class FitFunction:
    name: str
    category: str
    formula: str
    params: tuple[str, ...]
    func: Callable[..., np.ndarray]
    description: str = ""
    linear: tuple[str, ...] = ()
    grid: Callable[[np.ndarray, np.ndarray], dict] | None = None
    grid_filter: Callable[[dict], bool] | None = None
    basis: Callable[[np.ndarray], np.ndarray] | None = None
    canonical: Callable[[dict], dict] | None = None
    user_defined: bool = False
    meta: dict = field(default_factory=dict)
    jac: Callable[..., np.ndarray] | None = None   # exact ∂f/∂p, shape (n, n_params)
    nvar: int = 1                                   # independent variables (2: z = f(x, y))

    def __call__(self, x, *p) -> np.ndarray:
        x = np.asarray(x, float)
        shape = x.shape if self.nvar == 1 else x.shape[:-1]
        with np.errstate(all="ignore"):
            out = np.asarray(self.func(x, *p), float)
        return np.broadcast_to(out, shape).copy() if out.shape != shape else out

    def jacobian(self, x, p) -> np.ndarray:
        """∂f/∂p for all parameters (analytic when available, else central differences)."""
        x = np.asarray(x, float)
        p = np.asarray(p, float)
        if self.jac is not None:
            with np.errstate(all="ignore"):
                return np.asarray(self.jac(x, *p), float)
        n = x.shape[0]
        g = np.empty((n, len(p)))
        for i in range(len(p)):
            h = 6e-6 * abs(p[i]) if abs(p[i]) > 1e-10 else 6e-6
            up, dn = p.copy(), p.copy()
            up[i] += h
            dn[i] -= h
            g[:, i] = (self(x, *up) - self(x, *dn)) / (2 * h)
        return g

    @property
    def categories(self) -> tuple[str, ...]:
        return (self.category,) + tuple(self.meta.get("also", ()))

    @property
    def is_linear(self) -> bool:
        return self.basis is not None

    def initial_guess(self, x, y, w=None) -> dict[str, float]:
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        ok = (np.isfinite(x) if self.nvar == 1 else np.isfinite(x).all(axis=-1)) \
            & np.isfinite(y)
        w = None if w is None else np.asarray(w, float)
        if w is not None:
            ok &= np.isfinite(w) & (w > 0)
            w = w[ok]
        x, y = x[ok], y[ok]
        if self.basis is not None:
            coef = _wlstsq(self.basis(x), y, w)[0]
            return dict(zip(self.params, map(float, coef)))
        guess = self.meta.get("guess")
        if guess is not None and len(x) >= 2:
            try:
                with np.errstate(all="ignore"):
                    g = guess(x, y)
                if all(np.isfinite(v) for v in g.values()):
                    return {p: float(g.get(p, 1.0)) for p in self.params}
            except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
                pass
        if self.grid is not None and len(x) >= 2:
            g = _grid_guess(self, x, y, w)
            if g is not None:
                return g
        return {p: 1.0 for p in self.params}


def _wlstsq(B: np.ndarray, y: np.ndarray, w=None):
    sw = np.ones(len(y)) if w is None else np.sqrt(w)
    A = B * sw[:, None]
    scale = np.sqrt((A ** 2).sum(axis=0))
    scale[scale == 0] = 1.0
    coef, *_ = np.linalg.lstsq(A / scale, y * sw, rcond=None)
    coef = coef / scale
    r = (y - B @ coef) * sw
    return coef, float(r @ r)


def _grid_guess(f: FitFunction, x, y, w) -> dict | None:
    nonlin = [p for p in f.params if p not in f.linear]
    cands = f.grid(x, y)
    arrays = [np.unique(np.asarray(cands[p], float)) for p in nonlin]
    arrays = [a[np.isfinite(a)] if len(a[np.isfinite(a)]) else np.array([1.0]) for a in arrays]
    while math.prod(len(a) for a in arrays) > _GRID_MAX:
        i = int(np.argmax([len(a) for a in arrays]))
        arrays[i] = arrays[i][::2]
    best_rss, best = np.inf, None
    sw = np.ones(len(y)) if w is None else np.sqrt(w)
    with np.errstate(all="ignore"):
        for combo in product(*arrays):
            vals = dict(zip(nonlin, combo))
            if f.grid_filter is not None and not f.grid_filter(vals):
                continue
            cols = []
            for lp in f.linear:
                args = [vals[p] if p in vals else (1.0 if p == lp else 0.0) for p in f.params]
                cols.append(np.broadcast_to(np.asarray(f.func(x, *args), float), y.shape))
            if cols:
                B = np.column_stack(cols)
                if not np.all(np.isfinite(B)):
                    continue
                A = B * sw[:, None]
                scale = np.sqrt((A ** 2).sum(axis=0))
                if np.any(scale == 0) or not np.all(np.isfinite(scale)):
                    continue
                coef, *_ = np.linalg.lstsq(A / scale, y * sw, rcond=None)
                coef = coef / scale
                r = (y - B @ coef) * sw
            else:
                coef = np.array([])
                r = (y - np.asarray(f.func(x, *[vals[p] for p in f.params]), float)) * sw
            rss = float(r @ r)
            if np.isfinite(rss) and rss < best_rss:
                best_rss = rss
                best = {**vals, **dict(zip(f.linear, coef))}
    if best is None:
        return None
    return {p: float(best[p]) for p in f.params}


# ---------------------------------------------------------------------- grid helpers
def _span(x) -> float:
    s = float(np.ptp(x)) if len(x) else 0.0
    return s if s > 0 else (abs(float(np.mean(x))) if len(x) and np.mean(x) else 1.0)


def _taus(x, n=24):
    s = _span(x)
    return np.geomspace(s / 50, s * 5, n)


def _centres(x, n=21):
    return np.linspace(np.min(x), np.max(x), n)


def _log_centres(x, n=21):
    xp = np.asarray(x)[np.asarray(x) > 0]
    if not len(xp):
        return np.array([1.0])
    lo, hi = float(xp.min()), float(xp.max())
    if lo == hi:
        lo, hi = lo / 2, hi * 2
    return np.geomspace(lo, hi, n)


def _widths(x, n=16):
    s = _span(x)
    return np.geomspace(s / 200, s, n)


def _xmax_abs(x) -> float:
    m = float(np.max(np.abs(x))) if len(x) else 1.0
    return m if m > 0 else 1.0


def _peak_centres(x, y, n=25):
    dev = np.abs(y - np.median(y))
    top = np.asarray(x)[np.argsort(dev)[-3:]]
    return np.concatenate([_centres(x, n), top])


def _ordered(*names):
    def check(v):
        return all(v[a] < v[b] for a, b in zip(names, names[1:]))
    return check


def _abs_keys(*keys):
    def canon(v):
        return {k: (abs(val) if k in keys else val) for k, val in v.items()}
    return canon


# ---------------------------------------------------------------------- model functions
def _gauss_area(x, xc, w, A):
    w = np.abs(w)
    return A / (w * np.sqrt(np.pi / 2)) * np.exp(-2 * ((x - xc) / w) ** 2)


def _lorentz_area(x, xc, w, A):
    w = np.abs(w)
    return (2 * A / np.pi) * (w / (4 * (x - xc) ** 2 + w ** 2))


def _gauss_fwhm(x, xc, w, A):
    sigma = np.abs(w) / (2 * np.sqrt(2 * _LN2))
    return A / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - xc) / sigma) ** 2)


def _voigt(x, xc, A, wG, wL):
    sigma = np.abs(wG) / (2 * np.sqrt(2 * _LN2))
    gamma = np.abs(wL) / 2
    return A * special.voigt_profile(x - xc, max(sigma, 1e-300), gamma)


def _pseudo_voigt(x, xc, A, w, mu):
    w = np.abs(w)
    lor = (2 / np.pi) * w / (4 * (x - xc) ** 2 + w ** 2)
    gau = np.sqrt(4 * _LN2) / (np.sqrt(np.pi) * w) * np.exp(-(4 * _LN2 / w ** 2) * (x - xc) ** 2)
    return A * (mu * lor + (1 - mu) * gau)


def voigt_fwhm(wG: float, wL: float) -> float:
    """Olivero & Longbothum (1977) approximation, accurate to 0.02 %."""
    wG, wL = abs(wG), abs(wL)
    return 0.5346 * wL + math.sqrt(0.2166 * wL ** 2 + wG ** 2)


def _F(name, category, formula, params, func, linear=(), grid=None, **kw) -> FitFunction:
    return FitFunction(name, category, formula, tuple(params.split()), func,
                       linear=tuple(linear.split()) if isinstance(linear, str) else linear,
                       grid=grid, **kw)


_LIB: list[FitFunction] = [
    # Exponential
    _F("Exp", "Exponential", "y = y0 + A*exp(R0*x)", "y0 A R0",
       lambda x, y0, A, R0: y0 + A * np.exp(R0 * x), "y0 A",
       lambda x, y: {"R0": np.concatenate([-1 / _taus(x), 1 / _taus(x)])},
       description="Exponential with rate constant R0."),
    _F("ExpDec1", "Exponential", "y = y0 + A1*exp(-x/t1)", "y0 A1 t1",
       lambda x, y0, A1, t1: y0 + A1 * np.exp(-x / t1), "y0 A1",
       lambda x, y: {"t1": _taus(x, 40)}, description="One-phase exponential decay."),
    _F("ExpDec2", "Exponential", "y = y0 + A1*exp(-x/t1) + A2*exp(-x/t2)", "y0 A1 t1 A2 t2",
       lambda x, y0, A1, t1, A2, t2: y0 + A1 * np.exp(-x / t1) + A2 * np.exp(-x / t2),
       "y0 A1 A2", lambda x, y: {"t1": _taus(x, 22), "t2": _taus(x, 22)},
       grid_filter=_ordered("t1", "t2"), description="Two-phase exponential decay."),
    _F("ExpDec3", "Exponential", "y = y0 + A1*exp(-x/t1) + A2*exp(-x/t2) + A3*exp(-x/t3)",
       "y0 A1 t1 A2 t2 A3 t3",
       lambda x, y0, A1, t1, A2, t2, A3, t3: (y0 + A1 * np.exp(-x / t1)
                                              + A2 * np.exp(-x / t2) + A3 * np.exp(-x / t3)),
       "y0 A1 A2 A3", lambda x, y: {"t1": _taus(x, 14), "t2": _taus(x, 14),
                                    "t3": _taus(x, 14)},
       grid_filter=_ordered("t1", "t2", "t3"), description="Three-phase exponential decay."),
    _F("ExpGrow1", "Exponential", "y = y0 + A1*exp(x/t1)", "y0 A1 t1",
       lambda x, y0, A1, t1: y0 + A1 * np.exp(x / t1), "y0 A1",
       lambda x, y: {"t1": _taus(x, 40)}, description="Exponential growth."),
    _F("ExpGrow2", "Exponential", "y = y0 + A1*exp(x/t1) + A2*exp(x/t2)", "y0 A1 t1 A2 t2",
       lambda x, y0, A1, t1, A2, t2: y0 + A1 * np.exp(x / t1) + A2 * np.exp(x / t2),
       "y0 A1 A2", lambda x, y: {"t1": _taus(x, 22), "t2": _taus(x, 22)},
       grid_filter=_ordered("t1", "t2"), description="Two-phase exponential growth."),
    _F("ExpAssoc", "Exponential", "y = y0 + A1*(1 - exp(-x/t1)) + A2*(1 - exp(-x/t2))",
       "y0 A1 t1 A2 t2",
       lambda x, y0, A1, t1, A2, t2: (y0 + A1 * (1 - np.exp(-x / t1))
                                      + A2 * (1 - np.exp(-x / t2))),
       "y0 A1 A2", lambda x, y: {"t1": _taus(x, 22), "t2": _taus(x, 22)},
       grid_filter=_ordered("t1", "t2"), description="Two-phase exponential association."),
    _F("BoxLucas1", "Exponential", "y = a*(1 - exp(-b*x))", "a b",
       lambda x, a, b: a * (1 - np.exp(-b * x)), "a", lambda x, y: {"b": 1 / _taus(x, 40)},
       description="One-phase association (first-order uptake)."),
    # Growth / Sigmoidal
    _F("Boltzmann", "Growth/Sigmoidal", "y = A2 + (A1 - A2)/(1 + exp((x - x0)/dx))",
       "A1 A2 x0 dx", lambda x, A1, A2, x0, dx: A2 + (A1 - A2) / (1 + np.exp((x - x0) / dx)),
       "A1 A2", lambda x, y: {"x0": _centres(x, 25), "dx": _widths(x, 18)},
       description="Boltzmann sigmoid."),
    _F("Logistic", "Growth/Sigmoidal", "y = A2 + (A1 - A2)/(1 + (x/x0)^p)", "A1 A2 x0 p",
       lambda x, A1, A2, x0, p: A2 + (A1 - A2) / (1 + (x / x0) ** p), "A1 A2",
       lambda x, y: {"x0": _log_centres(x, 25), "p": np.geomspace(0.3, 10, 16)},
       description="Four-parameter logistic (4PL) dose-response curve; x ≥ 0."),
    _F("Logistic5", "Growth/Sigmoidal", "y = A2 + (A1 - A2)/(1 + (x/x0)^h)^s",
       "A1 A2 x0 h s", lambda x, A1, A2, x0, h, s: A2 + (A1 - A2) / (1 + (x / x0) ** h) ** s,
       "A1 A2", lambda x, y: {"x0": _log_centres(x, 21), "h": np.geomspace(0.3, 10, 12),
                              "s": np.array([0.25, 0.5, 1.0, 2.0, 4.0])},
       description="Five-parameter logistic (5PL, asymmetric); x ≥ 0."),
    _F("DoseResp", "Growth/Sigmoidal", "y = A1 + (A2 - A1)/(1 + 10^((LOGx0 - x)*p))",
       "A1 A2 LOGx0 p",
       lambda x, A1, A2, LOGx0, p: A1 + (A2 - A1) / (1 + 10.0 ** ((LOGx0 - x) * p)), "A1 A2",
       lambda x, y: {"LOGx0": _centres(x, 25),
                     "p": np.geomspace(0.5 / _span(x), 50 / _span(x), 16)},
       description="Dose-response with x = log10(dose); LOGx0 = log EC50."),
    _F("Hill", "Growth/Sigmoidal", "y = START + (END - START)*x^n/(k^n + x^n)",
       "START END k n",
       lambda x, START, END, k, n: START + (END - START) * x ** n / (k ** n + x ** n),
       "START END", lambda x, y: {"k": _log_centres(x, 25), "n": np.geomspace(0.3, 10, 16)},
       description="Hill equation; k = half-saturation, n = Hill coefficient."),
    _F("Gompertz", "Growth/Sigmoidal", "y = a*exp(-exp(-k*(x - xc)))", "a xc k",
       lambda x, a, xc, k: a * np.exp(-np.exp(-k * (x - xc))), "a",
       lambda x, y: {"xc": _centres(x, 25),
                     "k": np.concatenate([1 / _widths(x, 16), -1 / _widths(x, 16)])},
       description="Gompertz growth model."),
    _F("Richards", "Growth/Sigmoidal", "y = a*(1 + (d - 1)*exp(-k*(x - xc)))^(1/(1 - d))",
       "a xc d k",
       lambda x, a, xc, d, k: a * (1 + (d - 1) * np.exp(-k * (x - xc))) ** (1 / (1 - d)),
       "a", lambda x, y: {"xc": _centres(x, 15), "d": np.array([0.3, 0.6, 1.5, 2, 3, 5]),
                          "k": 1 / _widths(x, 12)},
       description="Richards generalised logistic (d ≠ 1)."),
    _F("Weibull", "Growth/Sigmoidal", "y = A - (A - B)*exp(-(k*x)^d)", "A B k d",
       lambda x, A, B, k, d: A - (A - B) * np.exp(-(k * x) ** d), "A B",
       lambda x, y: {"k": 1 / _widths(x, 20), "d": np.geomspace(0.3, 6, 14)},
       description="Weibull growth (Origin Weibull1 with xc = 0); x ≥ 0."),
    # Peak functions
    _F("Gauss", "Peak Functions", "y = y0 + A/(w*sqrt(pi/2))*exp(-2*((x - xc)/w)^2)",
       "y0 xc w A", lambda x, y0, xc, w, A: y0 + _gauss_area(x, xc, w, A), "y0 A",
       lambda x, y: {"xc": _peak_centres(x, y), "w": _widths(x, 20)},
       canonical=_abs_keys("w"),
       description="Gaussian, area form: A = area, w = 2σ ≈ 0.849·FWHM."),
    _F("GaussAmp", "Peak Functions", "y = y0 + A*exp(-(x - xc)^2/(2*w^2))", "y0 xc w A",
       lambda x, y0, xc, w, A: y0 + A * np.exp(-(x - xc) ** 2 / (2 * w ** 2)), "y0 A",
       lambda x, y: {"xc": _peak_centres(x, y), "w": _widths(x, 20)},
       canonical=_abs_keys("w"), description="Gaussian, amplitude form: A = height, w = σ."),
    _F("Lorentz", "Peak Functions", "y = y0 + (2*A/pi)*(w/(4*(x - xc)^2 + w^2))",
       "y0 xc w A", lambda x, y0, xc, w, A: y0 + _lorentz_area(x, xc, w, A), "y0 A",
       lambda x, y: {"xc": _peak_centres(x, y), "w": _widths(x, 20)},
       canonical=_abs_keys("w"), description="Lorentzian: A = area, w = FWHM."),
    _F("Voigt", "Peak Functions", "y = y0 + A*Voigt(x - xc; wG, wL)", "y0 xc A wG wL",
       lambda x, y0, xc, A, wG, wL: y0 + _voigt(x, xc, A, wG, wL), "y0 A",
       lambda x, y: {"xc": _peak_centres(x, y, 15), "wG": _widths(x, 11),
                     "wL": _widths(x, 11)},
       canonical=_abs_keys("wG", "wL"),
       description="Voigt profile: A = area, wG / wL = Gaussian / Lorentzian FWHM."),
    _F("PsdVoigt1", "Peak Functions",
       "y = y0 + A*(mu*(2/pi)*w/(4*(x-xc)^2 + w^2) + (1-mu)*sqrt(4*ln2)/(sqrt(pi)*w)"
       "*exp(-4*ln2*(x-xc)^2/w^2))", "y0 xc A w mu",
       lambda x, y0, xc, A, w, mu: y0 + _pseudo_voigt(x, xc, A, w, mu), "y0 A",
       lambda x, y: {"xc": _peak_centres(x, y, 21), "w": _widths(x, 16),
                     "mu": np.array([0.0, 0.25, 0.5, 0.75, 1.0])},
       canonical=_abs_keys("w"),
       description="Pseudo-Voigt: A = area, w = FWHM, mu = Lorentzian fraction."),
    _F("LogNormal", "Peak Functions",
       "y = y0 + A/(sqrt(2*pi)*w*x)*exp(-(ln(x/xc))^2/(2*w^2))", "y0 xc w A",
       lambda x, y0, xc, w, A: (y0 + A / (np.sqrt(2 * np.pi) * np.abs(w) * x)
                                * np.exp(-np.log(x / xc) ** 2 / (2 * w ** 2))), "y0 A",
       lambda x, y: {"xc": _log_centres(x, 25), "w": np.geomspace(0.03, 3, 16)},
       canonical=_abs_keys("w"), description="Log-normal peak (x > 0); A = area."),
    # Power
    _F("Allometric1", "Power", "y = a*x^b", "a b", lambda x, a, b: a * x ** b, "a",
       lambda x, y: {"b": np.linspace(-4, 4, 81)}, description="Power law (x > 0)."),
    _F("Allometric2", "Power", "y = a + b*x^c", "a b c", lambda x, a, b, c: a + b * x ** c,
       "a b", lambda x, y: {"c": np.linspace(-4, 4, 80)},
       description="Power law with offset (x > 0)."),
    _F("Power1", "Power", "y = A*abs(x - xc)^p", "A xc p",
       lambda x, A, xc, p: A * np.abs(x - xc) ** p, "A",
       lambda x, y: {"xc": np.concatenate([np.min(x) - _span(x) * np.array([.01, .1, .5, 1]),
                                           np.max(x) + _span(x) * np.array([.01, .1, .5, 1])]),
                     "p": np.linspace(-4, 4, 41)},
       description="Power law with a shifted origin xc."),
    # Rational / Hyperbola
    _F("Rational0", "Rational", "y = (b + c*x)/(1 + a*x)", "a b c",
       lambda x, a, b, c: (b + c * x) / (1 + a * x), "b c",
       lambda x, y: {"a": np.concatenate([[0.0], np.geomspace(1e-3, 1e3, 25) / _xmax_abs(x),
                                          -np.geomspace(1e-3, 1e3, 25) / _xmax_abs(x)])},
       description="Rational function, first order over first order."),
    _F("MichaelisMenten", "Hyperbola", "y = Vmax*x/(Km + x)", "Vmax Km",
       lambda x, Vmax, Km: Vmax * x / (Km + x), "Vmax",
       lambda x, y: {"Km": np.geomspace(1e-3, 1e2, 50) * _xmax_abs(x)},
       description="Michaelis-Menten enzyme kinetics (Origin Hyperbl)."),
    # Chemistry
    _F("Arrhenius", "Chemical Kinetics", "k = A*exp(-Ea/(R*T)),  R = 8.314462618",
       "A Ea", lambda x, A, Ea: A * np.exp(-Ea / (R_GAS * x)), "A",
       lambda x, y: {"Ea": np.geomspace(1e2, 1e6, 80)},
       description="Arrhenius equation; x = T in kelvin, Ea in J/mol."),
    _F("Langmuir", "Adsorption Isotherm", "q = qmax*KL*C/(1 + KL*C)", "qmax KL",
       lambda x, qmax, KL: qmax * KL * x / (1 + KL * x), "qmax",
       lambda x, y: {"KL": np.geomspace(1e-3, 1e3, 60) / _xmax_abs(x)},
       description="Langmuir isotherm; x = equilibrium concentration Ce."),
    _F("Freundlich", "Adsorption Isotherm", "q = KF*C^(1/n)", "KF n",
       lambda x, KF, n: KF * x ** (1 / n), "KF", lambda x, y: {"n": np.geomspace(0.1, 10, 60)},
       description="Freundlich isotherm (x > 0)."),
    _F("Temkin", "Adsorption Isotherm", "q = B*ln(KT*C)", "B KT",
       lambda x, B, KT: B * np.log(KT * x), "B",
       lambda x, y: {"KT": np.geomspace(1e-4, 1e4, 60) / _xmax_abs(x)},
       description="Temkin isotherm, B = RT/bT (x > 0)."),
    _F("BET", "Adsorption Isotherm", "q = qm*C*x/((1 - x)*(1 + (C - 1)*x)),  x = P/P0",
       "qm C", lambda x, qm, C: qm * C * x / ((1 - x) * (1 + (C - 1) * x)), "qm",
       lambda x, y: {"C": np.geomspace(0.5, 1e4, 60)},
       description="BET multilayer isotherm; x = relative pressure P/P0 (0 < x < 1)."),
]



@lru_cache(maxsize=None)
def polynomial(order: int) -> FitFunction:
    """Origin Polynomial: y = Intercept + B1*x + B2*x^2 + ... (Line for order 1)."""
    if not 1 <= order <= 10:
        raise ValueError("Polynomial order must be between 1 and 10.")
    if order == 1:
        params = ("Intercept", "Slope")
        formula = "y = Intercept + Slope*x"
        name = "Line"
    else:
        params = ("Intercept",) + tuple(f"B{i}" for i in range(1, order + 1))
        formula = "y = Intercept + " + " + ".join(
            "B1*x" if i == 1 else f"B{i}*x^{i}" for i in range(1, order + 1))
        name = f"Poly{order}"

    def basis(x, k=order):
        return np.vander(np.asarray(x, float), k + 1, increasing=True)

    def func(x, *p):
        return np.polynomial.polynomial.polyval(x, np.asarray(p, float))

    return FitFunction(name, "Polynomial", formula, params, func, basis=basis,
                       description=f"Polynomial of order {order}.",
                       meta={"order": order})


def get_function(name: str) -> FitFunction | None:
    key = name.strip().lower()
    key = ALIASES.get(key, key).lower()
    for f in _LIB:
        if f.name.lower() == key:
            return f
    if key in ("line", "linear"):
        return polynomial(1)
    if key.startswith("poly") and key[4:].isdigit() and 1 <= int(key[4:]) <= 10:
        return polynomial(int(key[4:]))
    return None


def functions_in(category: str) -> list[FitFunction]:
    return [f for f in _LIB if category in f.categories]


# ---------------------------------------------------------------------- peak sums
def peak_sum(n_peaks: int, shape: str = "gauss", baseline: str = "constant") -> FitFunction:
    """Sum of ``n_peaks`` peaks (FWHM / area parametrisation) on a none/constant/linear base.

    Parameters: [y0 [, k]] then (xc_i, w_i, A_i) per peak (Voigt: xc_i, wG_i, wL_i, A_i).
    """
    shape = shape.lower()
    base_params = {"none": (), "constant": ("y0",), "linear": ("y0", "k")}[baseline]
    per = ("xc", "wG", "wL", "A") if shape == "voigt" else ("xc", "w", "A")
    params = base_params + tuple(f"{p}{i}" for i in range(1, n_peaks + 1) for p in per)
    nb, npp = len(base_params), len(per)

    def single(x, q):
        if shape == "gauss":
            return _gauss_fwhm(x, q[0], q[1], q[2])
        if shape == "lorentz":
            return _lorentz_area(x, q[0], q[1], q[2])
        return _voigt(x, q[0], q[3], q[1], q[2])

    def func(x, *p):
        y = np.zeros_like(np.asarray(x, float))
        if nb >= 1:
            y = y + p[0]
        if nb == 2:
            y = y + p[1] * x
        for i in range(n_peaks):
            y = y + single(x, p[nb + i * npp: nb + (i + 1) * npp])
        return y

    width_keys = [p for p in params if p[:1] == "w"]
    label = {"gauss": "Gaussian", "lorentz": "Lorentzian", "voigt": "Voigt"}[shape]
    formula = {"none": "y = ", "constant": "y = y0 + ", "linear": "y = y0 + k*x + "}[baseline]
    formula += f"Σ {label}(x; xc_i, FWHM_i, area A_i)"
    return FitFunction(f"{n_peaks}×{label}", "Multiple Peaks", formula, params, func,
                       canonical=_abs_keys(*width_keys),
                       description=f"Sum of {n_peaks} {label} peaks, {baseline} baseline.",
                       meta={"shape": shape, "baseline": baseline, "n_peaks": n_peaks,
                             "per": per, "n_base": nb})


# ---------------------------------------------------------------------- user functions
# Each entry: (value function, number of arguments, derivative). The derivative of a unary
# function gets (a, value) and returns df/da; of a binary one (a, b, value) -> (df/da, df/db).
def _i1e_d(a, v):
    return special.i0e(a) - v / a - np.sign(a) * v


_FUNCS: dict[str, tuple[Callable, int, Callable | None]] = {
    "exp": (np.exp, 1, lambda a, v: v),
    "ln": (np.log, 1, lambda a, v: 1 / a),
    "log": (np.log10, 1, lambda a, v: 1 / (a * math.log(10))),
    "log10": (np.log10, 1, lambda a, v: 1 / (a * math.log(10))),
    "log2": (np.log2, 1, lambda a, v: 1 / (a * _LN2)),
    "sqrt": (np.sqrt, 1, lambda a, v: 0.5 / v),
    "abs": (np.abs, 1, lambda a, v: np.sign(a)),
    "sin": (np.sin, 1, lambda a, v: np.cos(a)),
    "cos": (np.cos, 1, lambda a, v: -np.sin(a)),
    "tan": (np.tan, 1, lambda a, v: 1 + v * v),
    "asin": (np.arcsin, 1, lambda a, v: 1 / np.sqrt(1 - a * a)),
    "acos": (np.arccos, 1, lambda a, v: -1 / np.sqrt(1 - a * a)),
    "atan": (np.arctan, 1, lambda a, v: 1 / (1 + a * a)),
    "sinh": (np.sinh, 1, lambda a, v: np.cosh(a)),
    "cosh": (np.cosh, 1, lambda a, v: np.sinh(a)),
    "tanh": (np.tanh, 1, lambda a, v: 1 - v * v),
    "coth": (lambda a: 1 / np.tanh(a), 1, lambda a, v: 1 - v * v),
    "erf": (special.erf, 1, lambda a, v: 2 / math.sqrt(math.pi) * np.exp(-a * a)),
    "erfc": (special.erfc, 1, lambda a, v: -2 / math.sqrt(math.pi) * np.exp(-a * a)),
    "erfcx": (special.erfcx, 1, lambda a, v: 2 * a * v - 2 / math.sqrt(math.pi)),
    "gamma": (special.gamma, 1, lambda a, v: v * special.digamma(a)),
    "lgamma": (special.gammaln, 1, lambda a, v: special.digamma(a)),
    "i1e": (special.i1e, 1, _i1e_d),
    "sign": (np.sign, 1, lambda a, v: 0.0 * a),
    "step": (lambda a: np.heaviside(a, 1.0), 1, lambda a, v: 0.0 * a),
    "pow": (np.power, 2, None),
    "atan2": (np.arctan2, 2, lambda a, b, v: (b / (a * a + b * b), -a / (a * a + b * b))),
    "min": (np.minimum, 2, lambda a, b, v: (1.0 * (a <= b), 1.0 * (a > b))),
    "max": (np.maximum, 2, lambda a, b, v: (1.0 * (a >= b), 1.0 * (a < b))),
    "iif": (None, 3, None),   # written if(cond, a, b); "if" is renamed before parsing
}
_CONSTS = {"pi": math.pi}
_BINOPS = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
           ast.Div: np.divide, ast.Pow: np.power, ast.Mod: np.mod}
_CMPOPS = {ast.Lt: np.less, ast.LtE: np.less_equal, ast.Gt: np.greater,
           ast.GtE: np.greater_equal, ast.Eq: np.equal, ast.NotEq: np.not_equal}
_UNOPS = {ast.USub: np.negative, ast.UAdd: np.positive}
_MAX_LEN = 2000


class FunctionSyntaxError(ValueError):
    pass


def _gadd(g1: dict, g2: dict, s: float = 1.0) -> dict:
    out = dict(g1)
    for k, v in g2.items():
        out[k] = out[k] + s * v if k in out else s * v
    return out


def _gscale(g: dict, f) -> dict:
    return {k: v * f for k, v in g.items()}


def _pow_grad(a, b, v, ga: dict, gb: dict) -> dict:
    out = _gscale(ga, b * np.power(a, b - 1)) if ga else {}
    if gb:
        with np.errstate(all="ignore"):
            lg = np.where(v == 0, 0.0, v * np.log(np.abs(a)))
        out = _gadd(out, _gscale(gb, lg))
    return out


class _Compiler:
    """Compiles a whitelisted expression into closures returning ``(value, gradient)``,
    where ``gradient`` maps parameter names to ∂value/∂parameter (forward-mode automatic
    differentiation; exact Jacobians make NLFit reach the NIST certified digits)."""

    def __init__(self, src: str, independents: tuple[str, ...], dependent: str | None):
        self.src = src
        self.independents = independents
        self.dependent = dependent
        self.params: list[str] = []

    def comp(self, node):  # noqa: C901 - one branch per node type
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            v = float(node.value)
            return lambda env: (v, {})
        if isinstance(node, ast.Name):
            name = node.id
            if name in self.independents:
                key = "\0" + name
                return lambda env: (env[key], {})
            if name in _CONSTS:
                v = _CONSTS[name]
                return lambda env: (v, {})
            if name in _FUNCS:
                raise FunctionSyntaxError(f"'{name}' is a function; write {name}(...).")
            if name == self.dependent:
                raise FunctionSyntaxError(f"'{name}' cannot appear on the right-hand side.")
            if name.startswith("_"):
                raise FunctionSyntaxError(f"'{name}' is not a valid parameter name.")
            if name not in self.params:
                self.params.append(name)
            return lambda env: (env[name], {name: 1.0})
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            a, b = self.comp(node.left), self.comp(node.right)
            op = type(node.op)
            if op is ast.Add:
                def f(env):
                    (va, ga), (vb, gb) = a(env), b(env)
                    return va + vb, _gadd(ga, gb)
            elif op is ast.Sub:
                def f(env):
                    (va, ga), (vb, gb) = a(env), b(env)
                    return va - vb, _gadd(ga, gb, -1.0)
            elif op is ast.Mult:
                def f(env):
                    (va, ga), (vb, gb) = a(env), b(env)
                    return va * vb, _gadd(_gscale(ga, vb), _gscale(gb, va))
            elif op is ast.Div:
                def f(env):
                    (va, ga), (vb, gb) = a(env), b(env)
                    v = va / vb
                    return v, _gadd(_gscale(ga, 1 / vb), _gscale(gb, -v / vb))
            elif op is ast.Pow:
                def f(env):
                    (va, ga), (vb, gb) = a(env), b(env)
                    v = np.power(va, vb)
                    return v, _pow_grad(va, vb, v, ga, gb)
            else:
                def f(env):
                    (va, ga), (vb, gb) = a(env), b(env)
                    return np.mod(va, vb), _gadd(ga, _gscale(gb, -np.floor(va / vb)))
            return f
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNOPS:
            a = self.comp(node.operand)
            if isinstance(node.op, ast.USub):
                return lambda env: (lambda r: (-r[0], _gscale(r[1], -1.0)))(a(env))
            return a
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and \
                type(node.ops[0]) in _CMPOPS:
            a, b = self.comp(node.left), self.comp(node.comparators[0])
            op = _CMPOPS[type(node.ops[0])]
            return lambda env: (np.asarray(op(a(env)[0], b(env)[0]), float), {})
        if isinstance(node, ast.BoolOp):
            parts = [self.comp(v) for v in node.values]
            fn = np.logical_and if isinstance(node.op, ast.And) else np.logical_or

            def f(env):
                out = parts[0](env)[0] != 0
                for p in parts[1:]:
                    out = fn(out, p(env)[0] != 0)
                return np.asarray(out, float), {}
            return f
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
                fname = getattr(node.func, "id", "?")
                raise FunctionSyntaxError(f"Unknown function '{fname}'. Allowed: "
                                          + ", ".join(sorted(_FUNCS)))
            name = node.func.id
            fn, nargs, d = _FUNCS[name]
            if node.keywords or len(node.args) != nargs:
                raise FunctionSyntaxError(f"{name}() takes {nargs} argument(s).")
            args = [self.comp(a) for a in node.args]
            if name == "iif":
                return self._if(*args)
            if name == "pow":
                a, b = args

                def f(env):
                    (va, ga), (vb, gb) = a(env), b(env)
                    v = np.power(va, vb)
                    return v, _pow_grad(va, vb, v, ga, gb)
                return f
            if nargs == 1:
                (a,) = args

                def f(env):
                    va, ga = a(env)
                    v = fn(va)
                    return v, (_gscale(ga, d(va, v)) if ga else {})
                return f
            a, b = args

            def f(env):
                (va, ga), (vb, gb) = a(env), b(env)
                v = fn(va, vb)
                if not ga and not gb:
                    return v, {}
                da, db = d(va, vb, v)
                return v, _gadd(_gscale(ga, da), _gscale(gb, db))
            return f
        segment = ast.get_source_segment(self.src, node) or type(node).__name__
        raise FunctionSyntaxError(f"'{segment}' is not allowed in a fitting function.")

    @staticmethod
    def _if(c, a, b):
        def f(env):
            cond = np.asarray(c(env)[0]) != 0
            (va, ga), (vb, gb) = a(env), b(env)
            v = np.where(cond, va, vb)
            g = {k: np.where(cond, ga.get(k, 0.0), gb.get(k, 0.0)) for k in set(ga) | set(gb)}
            return v, g
        return f


def parse_function(text: str, independents: tuple[str, ...] = ("x",),
                   params: tuple[str, ...] | None = None, implicit: bool = False,
                   name: str = "User", category: str = "User Defined",
                   description: str = "User-defined function.") -> FitFunction:
    """Compile ``y = <expression>`` (or ``z = f(x, y)`` with two independents, or an
    implicit ``f(x, y) = g(x, y)`` with ``implicit=True``) into a :class:`FitFunction` with
    an exact analytic Jacobian. ``params`` fixes the parameter order (default: order of
    first appearance)."""
    src = (text or "").strip()
    if not src:
        raise FunctionSyntaxError("Enter a function, e.g. y = A*exp(-x/t) + y0")
    if len(src) > _MAX_LEN:
        raise FunctionSyntaxError("The function is too long.")
    default_dep = "y" if len(independents) == 1 else "z"
    dependent = None if implicit else default_dep
    if "=" in src.replace("==", "\0\0").replace("<=", "\0\0").replace(">=", "\0\0") \
            .replace("!=", "\0\0"):
        k = src.replace("==", "\0\0").replace("<=", "\0\0").replace(">=", "\0\0") \
            .replace("!=", "\0\0").index("=")
        lhs, rhs = src[:k].strip(), src[k + 1:].strip()
        if implicit:
            src = f"({lhs}) - ({rhs})"
        else:
            if lhs.endswith(")") and "(" in lhs:
                lhs = lhs[: lhs.index("(")].strip()
            if not lhs.isidentifier():
                raise FunctionSyntaxError(f"'{lhs}' is not a valid name for the dependent "
                                          "variable.")
            dependent, src = lhs, rhs
    expr = re.sub(r"\bif\s*\(", "iif(", src.replace("^", "**"))
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise FunctionSyntaxError(f"Syntax error in the function: {exc.msg}") from None
    c = _Compiler(expr, tuple(independents), dependent)
    try:
        body = c.comp(tree.body)
    except RecursionError:
        raise FunctionSyntaxError("The function is nested too deeply.") from None
    if not c.params:
        raise FunctionSyntaxError("The function has no parameters to fit.")
    names = tuple(c.params)
    if params is not None:
        if set(params) != set(names):
            raise FunctionSyntaxError(f"Parameter list {params} does not match {names}.")
        names = tuple(params)
    nvar = len(independents)
    keys = ["\0" + v for v in independents]

    def env_of(x, p):
        env = dict(zip(names, (float(v) for v in p)))
        x = np.asarray(x, float)
        if nvar == 1:
            env[keys[0]] = x
        else:
            for i, k in enumerate(keys):
                env[k] = x[..., i]
        return env, (x.shape if nvar == 1 else x.shape[:-1])

    def func(x, *p):
        env, _ = env_of(x, p)
        return body(env)[0]

    def jac(x, *p):
        env, shape = env_of(x, p)
        with np.errstate(all="ignore"):
            _, g = body(env)
        cols = [np.broadcast_to(np.asarray(g.get(n, 0.0), float), shape) for n in names]
        return np.stack(cols, axis=-1) if cols else np.zeros(shape + (0,))

    shown = text.strip()
    if not implicit and "=" not in shown:
        shown = f"{default_dep} = {shown}"
    return FitFunction(name, category, shown, names, func, description=description,
                       user_defined=category == "User Defined", jac=jac, nvar=nvar,
                       meta={"dependent": dependent, "source": text.strip(),
                             "independents": tuple(independents), "implicit": implicit})


def parse_user_function(text: str, independent: str = "x") -> FitFunction:
    """Compile ``y = <expression in x and parameters>`` into a :class:`FitFunction`.

    Parameters are all names that are not ``x``, a function or ``pi``; their order is the
    order of first appearance. Raises :class:`FunctionSyntaxError` on anything else.
    """
    return parse_function(text, (independent,))


# ---------------------------------------------------------------------- extended library
def _E(name, category, formula, params, linear="", grid=None, description="", also=(),
       fixed=None, guess=None, grid_filter=None, abs_keys=()) -> FitFunction:
    """A library function defined by its formula (exact Jacobian from the parser)."""
    f = parse_function(formula, params=tuple(params.split()), name=name, category=category,
                       description=description)
    f.user_defined = False
    f.linear = tuple(linear.split())
    f.grid = grid
    f.grid_filter = grid_filter
    if abs_keys:
        f.canonical = _abs_keys(*abs_keys.split())
    f.meta = {"source": formula, "dependent": "y", "independents": ("x",)}
    if also:
        f.meta["also"] = tuple(also)
    if fixed:
        f.meta["fixed"] = dict(fixed)
    if guess is not None:
        f.meta["guess"] = guess
    return f


def _min_minus(x, fr=(0.0, 0.01, 0.1, 0.5)):
    return float(np.min(x)) - _span(x) * np.asarray(fr)


def _rates(x, n=30):
    return np.concatenate([np.exp(-1 / _taus(x, n)), np.exp(1 / _taus(x, n))])


def _recip_guess(order):
    """Guess for y = 1/(polynomial): fit the polynomial to 1/y."""
    def guess(x, y):
        ok = y != 0
        c = np.polynomial.polynomial.polyfit(x[ok], 1 / y[ok], order)
        return dict(zip("abc", map(float, c)))
    return guess


def _sine_core(x, y, freq_mult=1.0, decay=None):
    best = None
    s = _span(x)
    for w in s / np.geomspace(0.25, max(len(x) / 3, 1.0), 80):
        om = np.pi / w * freq_mult
        e = np.ones_like(x) if decay is None else np.exp(-x / decay)
        B = np.column_stack([np.ones_like(x), e * np.sin(om * x), e * np.cos(om * x)])
        coef, rss = _wlstsq(B, y)
        if best is None or rss < best[0]:
            best = (rss, w, coef)
    return best


def _sine_guess(x, y):
    _, w, (c, a, b) = _sine_core(x, y)
    A = math.hypot(a, b)
    xc = -math.atan2(b, a) * w / math.pi
    return {"y0": float(c), "xc": float(xc), "w": float(w), "A": float(A)}


def _sinedamp_guess(x, y):
    best = None
    for t0 in list(_taus(x, 12)) + [1e12]:
        r = _sine_core(x, y, decay=t0)
        if best is None or r[0] < best[0][0]:
            best = (r, t0)
    (_, w, (c, a, b)), t0 = best
    return {"y0": float(c), "xc": float(-math.atan2(b, a) * w / math.pi), "w": float(w),
            "A": float(math.hypot(a, b)), "t0": float(t0)}


def _sinesqr_guess(x, y):
    _, w2, (c, a, b) = _sine_core(x, y)   # y = c + a sin(πx/w2) + b cos(πx/w2), w = 2·w2
    w = 2 * w2
    half = math.hypot(a, b)
    phi = math.atan2(-a, -b)
    return {"y0": float(c - half), "xc": float(phi * w / (2 * math.pi)), "w": float(w),
            "A": float(2 * half)}


def _second_order_guess(x, y):
    ok = y > 0
    k, inv0 = np.polynomial.polynomial.polyfit(x[ok], 1 / y[ok], 1)[::-1]
    return {"c0": float(1 / inv0) if inv0 else float(np.max(y)), "k": float(k)}


def _nth_order_guess(x, y):
    g = _second_order_guess(x, y)
    return {"c0": g["c0"], "k": abs(g["k"]) * abs(g["c0"]) ** 0.5 or 1.0, "n": 1.5}


def _hypmod_guess(x, y):
    ok = y != 0
    p2, p1 = np.polynomial.polynomial.polyfit(x[ok], x[ok] / y[ok], 1)[::-1]
    return {"P1": float(p1), "P2": float(p2)}


_LN = "ln(2)"
_EXT: list[FitFunction] = [
    # --- Origin Basic Functions / Exponential (time-offset forms)
    _E("ExpDecay1", "Exponential", "y = y0 + A1*exp(-(x - x0)/t1)", "y0 x0 A1 t1", "y0 A1",
       lambda x, y: {"x0": [float(np.min(x))], "t1": _taus(x, 40)},
       "Exponential decay with time offset x0 (x0 fixed by default).",
       ("Origin Basic Functions",), {"x0": None}),
    _E("ExpDecay2", "Exponential", "y = y0 + A1*exp(-(x - x0)/t1) + A2*exp(-(x - x0)/t2)",
       "y0 x0 A1 t1 A2 t2", "y0 A1 A2",
       lambda x, y: {"x0": [float(np.min(x))], "t1": _taus(x, 22), "t2": _taus(x, 22)},
       "Two-phase exponential decay with time offset.", ("Origin Basic Functions",),
       {"x0": None}, grid_filter=_ordered("t1", "t2")),
    _E("ExpDecay3", "Exponential",
       "y = y0 + A1*exp(-(x - x0)/t1) + A2*exp(-(x - x0)/t2) + A3*exp(-(x - x0)/t3)",
       "y0 x0 A1 t1 A2 t2 A3 t3", "y0 A1 A2 A3",
       lambda x, y: {"x0": [float(np.min(x))], "t1": _taus(x, 13), "t2": _taus(x, 13),
                     "t3": _taus(x, 13)},
       "Three-phase exponential decay with time offset.", ("Origin Basic Functions",),
       {"x0": None}, grid_filter=_ordered("t1", "t2", "t3")),
    _E("ExpGro1", "Exponential", "y = y0 + A1*exp((x - x0)/t1)", "y0 x0 A1 t1", "y0 A1",
       lambda x, y: {"x0": [float(np.min(x))], "t1": _taus(x, 40)},
       "Exponential growth with offset x0 (x0 fixed by default).",
       ("Origin Basic Functions",), {"x0": None}),
    _E("ExpGro2", "Exponential", "y = y0 + A1*exp((x - x0)/t1) + A2*exp((x - x0)/t2)",
       "y0 x0 A1 t1 A2 t2", "y0 A1 A2",
       lambda x, y: {"x0": [float(np.min(x))], "t1": _taus(x, 22), "t2": _taus(x, 22)},
       "Two-phase exponential growth with offset.", ("Origin Basic Functions",),
       {"x0": None}, grid_filter=_ordered("t1", "t2")),
    _E("Asymptotic1", "Exponential", "y = a - b*c^x", "a b c", "a b",
       lambda x, y: {"c": _rates(x)}, "Asymptotic regression model."),
    _E("BoxLucas2", "Exponential", "y = a1/(a2 - a1)*(exp(-a1*x) - exp(-a2*x))", "a1 a2", "",
       lambda x, y: {"a1": 1 / _taus(x, 25), "a2": 1 / _taus(x, 25)},
       "Intermediate of consecutive first-order reactions (unit amplitude).",
       grid_filter=lambda v: v["a1"] != v["a2"]),
    _E("Chapman", "Exponential", "y = a*(1 - exp(-b*x))^c", "a b c", "a",
       lambda x, y: {"b": 1 / _taus(x, 25), "c": np.geomspace(0.2, 10, 14)},
       "Chapman-Richards growth model."),
    _E("ExpLinear", "Exponential", "y = p1*exp(-x/p2) + p3 + p4*x", "p1 p2 p3 p4",
       "p1 p3 p4", lambda x, y: {"p2": _taus(x, 40)}, "Exponential decay plus a line."),
    _E("MnMolecular", "Exponential", "y = A*(1 - exp(-k*(x - xc)))", "A k xc", "A",
       lambda x, y: {"k": 1 / _taus(x, 30), "xc": _min_minus(x)},
       "Mitscherlich (monomolecular) growth."),
    _E("Stirling", "Exponential", "y = a + b*(exp(k*x) - 1)/k", "a b k", "a b",
       lambda x, y: {"k": np.concatenate([1 / _taus(x, 25), -1 / _taus(x, 25)])},
       "Stirling model."),
    _E("YldFert", "Exponential", "y = a + b*r^x", "a b r", "a b",
       lambda x, y: {"r": _rates(x)}, "Yield-fertilizer model."),
    _E("Langevin", "Exponential", "y = C*(coth((x - xc)/s) - s/(x - xc))", "C xc s", "C",
       lambda x, y: {"xc": [0.0], "s": _widths(x, 30)},
       "Langevin function (paramagnetism); xc fixed at 0 by default.", fixed={"xc": 0.0}),
    # --- Growth / Sigmoidal
    _E("SLogistic1", "Growth/Sigmoidal", "y = a/(1 + exp(-k*(x - xc)))", "a xc k", "a",
       lambda x, y: {"xc": _centres(x, 25),
                     "k": np.concatenate([1 / _widths(x, 16), -1 / _widths(x, 16)])},
       "Symmetric logistic growth."),
    _E("SWeibull1", "Growth/Sigmoidal", "y = a - (a - b)*exp(-(k*(x - xc))^d)",
       "a b xc d k", "a b",
       lambda x, y: {"xc": _min_minus(x, (0.0, 0.05, 0.2)), "k": 1 / _widths(x, 14),
                     "d": np.geomspace(0.3, 6, 10)},
       "Weibull growth with offset xc (x ≥ xc)."),
    _E("BiDoseResp", "Growth/Sigmoidal",
       "y = A1 + (A2 - A1)*(p/(1 + 10^((LOGx01 - x)*h1)) + (1 - p)/(1 + 10^((LOGx02 - x)*h2)))",
       "A1 A2 LOGx01 LOGx02 h1 h2 p", "A1 A2",
       lambda x, y: {"LOGx01": _centres(x, 9), "LOGx02": _centres(x, 9),
                     "h1": np.array([2, 6, 20]) / _span(x),
                     "h2": np.array([2, 6, 20]) / _span(x), "p": [0.3, 0.5, 0.7]},
       "Biphasic dose-response (x = log dose).", ("Pharmacology",),
       grid_filter=_ordered("LOGx01", "LOGx02")),
    _E("DoubleBoltzmann", "Growth/Sigmoidal",
       "y = A1 + (A2 - A1)*(p/(1 + exp((x - x01)/h1)) + (1 - p)/(1 + exp((x - x02)/h2)))",
       "A1 A2 x01 x02 h1 h2 p", "A1 A2",
       lambda x, y: {"x01": _centres(x, 9), "x02": _centres(x, 9),
                     "h1": _span(x) / np.array([5, 15, 50]),
                     "h2": _span(x) / np.array([5, 15, 50]), "p": [0.3, 0.5, 0.7]},
       "Double Boltzmann sigmoid.", ("Electrophysiology",),
       grid_filter=_ordered("x01", "x02")),
    # --- Hyperbola
    _E("Hyperbl", "Hyperbola", "y = P1*x/(P2 + x)", "P1 P2", "P1",
       lambda x, y: {"P2": np.geomspace(1e-3, 1e2, 50) * _xmax_abs(x)},
       "Hyperbola (Michaelis-Menten form).", ("Origin Basic Functions",)),
    _E("Dhyperbl", "Hyperbola", "y = P1*x/(P2 + x) + P3*x/(P4 + x) + P5*x",
       "P1 P2 P3 P4 P5", "P1 P3 P5",
       lambda x, y: {"P2": np.geomspace(1e-3, 1e2, 20) * _xmax_abs(x),
                     "P4": np.geomspace(1e-3, 1e2, 20) * _xmax_abs(x)},
       "Double rectangular hyperbola plus a linear term.", ("Origin Basic Functions",),
       grid_filter=_ordered("P2", "P4")),
    _E("HyperbolaGen", "Hyperbola", "y = a - b/(1 + c*x)^(1/d)", "a b c d", "a b",
       lambda x, y: {"c": np.geomspace(1e-3, 1e3, 15) / _xmax_abs(x),
                     "d": np.geomspace(0.2, 5, 10)}, "Generalized hyperbola."),
    _E("HyperbolaMod", "Hyperbola", "y = x/(P1 + P2*x)", "P1 P2", guess=_hypmod_guess,
       description="Modified hyperbola."),
    _E("RectHyperbola", "Hyperbola", "y = a*b*x/(1 + b*x)", "a b", "a",
       lambda x, y: {"b": np.geomspace(1e-3, 1e3, 50) / _xmax_abs(x)},
       "Rectangular hyperbola."),
    # --- Logarithm
    _E("Log2P1", "Logarithm", "y = a + b*ln(x)", "a b", "a b", lambda x, y: {},
       "Logarithmic function (x > 0)."),
    _E("Log3P1", "Logarithm", "y = a - b*ln(x + c)", "a b c", "a b",
       lambda x, y: {"c": -float(np.min(x)) + _span(x) * np.geomspace(1e-3, 10, 30)},
       "Three-parameter logarithm."),
    _E("Bradley", "Logarithm", "y = a*ln(-b*ln(x))", "a b", "a",
       lambda x, y: {"b": np.geomspace(1e-3, 1e3, 40)}, "Bradley model (0 < x < 1)."),
    # --- Peak functions
    _E("Gaussian", "Peak Functions",
       "y = y0 + A/(w*sqrt(pi/(4*ln(2))))*exp(-4*ln(2)*(x - xc)^2/w^2)", "y0 xc w A", "y0 A",
       lambda x, y: {"xc": _peak_centres(x, y), "w": _widths(x, 20)},
       "Gaussian, FWHM form: A = area, w = FWHM.", abs_keys="w"),
    _E("Asym2Sig", "Peak Functions",
       "y = y0 + A/(1 + exp(-(x - xc + w1/2)/w2))*(1 - 1/(1 + exp(-(x - xc - w1/2)/w3)))",
       "y0 xc A w1 w2 w3", "y0 A",
       lambda x, y: {"xc": _peak_centres(x, y, 15), "w1": _widths(x, 8),
                     "w2": _widths(x, 6), "w3": _widths(x, 6)},
       "Asymmetric double sigmoidal peak.", ("Chromatography",)),
    _E("BiGaussian", "Peak Functions",
       "y = y0 + H*if(x < xc, exp(-0.5*((x - xc)/w1)^2), exp(-0.5*((x - xc)/w2)^2))",
       "y0 xc H w1 w2", "y0 H",
       lambda x, y: {"xc": _peak_centres(x, y, 15), "w1": _widths(x, 12),
                     "w2": _widths(x, 12)},
       "Bi-Gaussian (split Gaussian): H = height, w1/w2 = left/right σ.",
       ("Chromatography",), abs_keys="w1 w2"),
    _E("GaussMod", "Peak Functions",
       "y = y0 + A/(2*t0)*if(w/t0 - (x - xc)/w < 0, exp(0.5*(w/t0)^2 - (x - xc)/t0)"
       "*erfc((w/t0 - (x - xc)/w)/sqrt(2)), exp(-0.5*((x - xc)/w)^2)"
       "*erfcx((w/t0 - (x - xc)/w)/sqrt(2)))",
       "y0 A xc w t0", "y0 A",
       lambda x, y: {"xc": _peak_centres(x, y, 15), "w": _widths(x, 10),
                     "t0": _widths(x, 10)},
       "Exponentially modified Gaussian (EMG): A = area, w = σ, t0 = tail time constant.",
       ("Chromatography",)),
    _E("ECS", "Peak Functions",
       "y = y0 + A/(w*sqrt(2*pi))*exp(-0.5*((x - xc)/w)^2)*(1 + a3/6*(((x - xc)/w)^3 "
       "- 3*((x - xc)/w)) + a4/24*(((x - xc)/w)^4 - 6*((x - xc)/w)^2 + 3) + 10*a3^2/720"
       "*(((x - xc)/w)^6 - 15*((x - xc)/w)^4 + 45*((x - xc)/w)^2 - 15))",
       "y0 A xc w a3 a4", "y0 A",
       lambda x, y: {"xc": _peak_centres(x, y), "w": _widths(x, 20), "a3": [0.0],
                     "a4": [0.0]},
       "Edgeworth-Cramer series peak (chromatography).", ("Chromatography",)),
    _E("GCAS", "Peak Functions",
       "y = y0 + A/(w*sqrt(2*pi))*exp(-0.5*((x - xc)/w)^2)*(1 + a3/6*(((x - xc)/w)^3 "
       "- 3*((x - xc)/w)) + a4/24*(((x - xc)/w)^4 - 6*((x - xc)/w)^2 + 3))",
       "y0 A xc w a3 a4", "y0 A",
       lambda x, y: {"xc": _peak_centres(x, y), "w": _widths(x, 20), "a3": [0.0],
                     "a4": [0.0]},
       "Gram-Charlier A-series peak (chromatography).", ("Chromatography",)),
    _E("Extreme", "Peak Functions",
       "y = y0 + A*exp(-exp(-(x - xc)/w) - (x - xc)/w + 1)", "y0 A xc w", "y0 A",
       lambda x, y: {"xc": _peak_centres(x, y), "w": np.concatenate([_widths(x, 15),
                                                                     -_widths(x, 15)])},
       "Extreme value (Gumbel) peak: A = height."),
    _E("Giddings", "Peak Functions",
       "y = y0 + A/w*sqrt(xc/x)*i1e(2*sqrt(xc*x)/w)*exp(-(sqrt(x) - sqrt(xc))^2/w)",
       "y0 A xc w", "y0 A",
       lambda x, y: {"xc": _log_centres(x, 25), "w": _widths(x, 20)},
       "Giddings chromatography peak (x > 0).", ("Chromatography",)),
    _E("InvsPoly", "Peak Functions",
       "y = y0 + A/(1 + A1*(2*(x - xc)/w)^2 + A2*(2*(x - xc)/w)^4 + A3*(2*(x - xc)/w)^6)",
       "y0 xc w A A1 A2 A3", "y0 A",
       lambda x, y: {"xc": _peak_centres(x, y), "w": _widths(x, 20), "A1": [1.0],
                     "A2": [0.0], "A3": [0.0]}, "Inverse polynomial peak."),
    _E("Laplace", "Peak Functions", "y = y0 + A/(2*w)*exp(-abs(x - xc)/w)", "y0 xc w A",
       "y0 A", lambda x, y: {"xc": _peak_centres(x, y), "w": _widths(x, 20)},
       "Laplace (double exponential) peak: A = area.", abs_keys="w"),
    _E("Logistpk", "Peak Functions", "y = y0 + A/cosh((x - xc)/(2*w))^2", "y0 xc w A",
       "y0 A", lambda x, y: {"xc": _peak_centres(x, y), "w": _widths(x, 20)},
       "Logistic peak, y0 + 4A·e^(−z)/(1 + e^(−z))², z = (x − xc)/w: A = height.",
       abs_keys="w"),
    _E("PearsonVII", "Peak Functions",
       "y = y0 + A*2*gamma(mu)*sqrt(2^(1/mu) - 1)/(sqrt(pi)*gamma(mu - 0.5)*w)"
       "*(1 + 4*(2^(1/mu) - 1)*((x - xc)/w)^2)^(-mu)", "y0 xc A w mu", "y0 A",
       lambda x, y: {"xc": _peak_centres(x, y, 15), "w": _widths(x, 14),
                     "mu": [0.75, 1.0, 1.5, 2.5, 5.0, 12.0]},
       "Pearson VII: A = area, w = FWHM, mu = shape (1: Lorentzian, ∞: Gaussian).",
       ("Spectroscopy",), abs_keys="w"),
    _E("PsdVoigt2", "Peak Functions",
       "y = y0 + A*(mu*(2/pi)*wL/(4*(x - xc)^2 + wL^2) + (1 - mu)*sqrt(4*ln(2))"
       "/(sqrt(pi)*wG)*exp(-4*ln(2)*(x - xc)^2/wG^2))", "y0 xc A wG wL mu", "y0 A",
       lambda x, y: {"xc": _peak_centres(x, y, 15), "wG": _widths(x, 8),
                     "wL": _widths(x, 8), "mu": [0.0, 0.5, 1.0]},
       "Pseudo-Voigt with separate Gaussian / Lorentzian FWHM: A = area.",
       ("Spectroscopy",), abs_keys="wG wL"),
    _E("Poisson", "Peak Functions", "y = y0 + A*r^x*exp(-r)/gamma(x + 1)", "y0 A r", "y0 A",
       lambda x, y: {"r": np.geomspace(0.1, max(float(np.max(x)), 1.0) * 1.5, 40)},
       "Poisson distribution peak (x ≥ 0).", ("Origin Basic Functions",)),
    _E("Pulse", "Peak Functions",
       "y = y0 + if(x > x0, A*(1 - exp(-(x - x0)/t1))^P*exp(-(x - x0)/t2), 0)",
       "y0 x0 A t1 P t2", "y0 A",
       lambda x, y: {"x0": _centres(x, 8) - _span(x) / 20, "t1": _taus(x, 8),
                     "P": [1.0, 2.0, 4.0], "t2": _taus(x, 8)},
       "Pulse: rise with t1 (power P) and decay with t2 after x0.",
       ("Origin Basic Functions",)),
    # --- Piecewise
    _E("PWL2", "Piecewise", "y = if(x < xi, a + k1*x, a + k1*xi + k2*(x - xi))",
       "a k1 xi k2", "a k1 k2",
       lambda x, y: {"xi": np.quantile(x, np.linspace(0.05, 0.95, 40))},
       "Two-segment continuous piecewise linear (break at xi)."),
    _E("PWL3", "Piecewise",
       "y = a + k1*x + if(x > x1, (k2 - k1)*(x - x1), 0) + if(x > x2, (k3 - k2)*(x - x2), 0)",
       "a k1 x1 k2 x2 k3", "a k1 k2 k3",
       lambda x, y: {"x1": np.quantile(x, np.linspace(0.05, 0.95, 22)),
                     "x2": np.quantile(x, np.linspace(0.05, 0.95, 22))},
       "Three-segment continuous piecewise linear (breaks at x1 < x2).",
       grid_filter=_ordered("x1", "x2")),
    # --- Polynomial
    _E("Constant", "Polynomial", "y = y0", "y0", "y0", lambda x, y: {}, "Constant."),
    # --- Power
    _E("Belehradek", "Power", "y = a*(x - b)^c", "a b c", "a",
       lambda x, y: {"b": _min_minus(x, (1e-3, 0.01, 0.1, 0.5, 1.0, 2.0)),
                     "c": np.linspace(-4, 4, 41)}, "Belehradek model (x > b)."),
    _E("FreundlichEXT", "Power", "y = a*x^(b*x^(-c))", "a b c", "a",
       lambda x, y: {"b": np.linspace(-2, 2, 21), "c": np.linspace(-1, 1, 11)},
       "Extended Freundlich isotherm (x > 0).", ("Adsorption Isotherm",)),
    _E("LangmuirEXT1", "Power", "y = a*b*x^(1 - c)/(1 + b*x^(1 - c))", "a b c", "a",
       lambda x, y: {"b": np.geomspace(1e-4, 1e4, 30), "c": np.linspace(-0.9, 0.9, 13)},
       "Extended Langmuir (Sips-type) isotherm (x > 0).", ("Adsorption Isotherm",)),
    _E("Power0", "Power", "y = y0 + A*abs(x - xc)^p", "y0 xc A p", "y0 A",
       lambda x, y: {"xc": np.concatenate([_min_minus(x, (0.01, 0.1, 0.5, 1)),
                                           float(np.max(x)) + _span(x) *
                                           np.array([.01, .1, .5, 1])]),
                     "p": np.linspace(-4, 4, 41)}, "Power law with offset and origin xc."),
    # --- Rational
    _E("Holliday", "Rational", "y = 1/(a + b*x + c*x^2)", "a b c", guess=_recip_guess(2),
       description="Holliday model."),
    _E("Reciprocal", "Rational", "y = 1/(a + b*x)", "a b", guess=_recip_guess(1),
       description="Reciprocal model."),
    # --- Waveform
    _E("Sine", "Waveform", "y = y0 + A*sin(pi*(x - xc)/w)", "y0 xc w A", "y0 A",
       guess=_sine_guess, description="Sine wave: period 2w, phase xc.",
       also=("Origin Basic Functions",)),
    _E("SineDamp", "Waveform", "y = y0 + A*exp(-x/t0)*sin(pi*(x - xc)/w)", "y0 xc w t0 A",
       "y0 A", guess=_sinedamp_guess, description="Damped sine wave."),
    _E("SineSqr", "Waveform", "y = y0 + A*sin(pi*(x - xc)/w)^2", "y0 xc w A", "y0 A",
       guess=_sinesqr_guess, description="Squared sine wave."),
    # --- Electrophysiology
    _E("BoltzIV", "Electrophysiology", "y = (x - Vrev)*Gmax/(1 + exp((V50 - x)/k))",
       "Vrev Gmax V50 k", "Gmax",
       lambda x, y: {"Vrev": _centres(x, 15), "V50": _centres(x, 15),
                     "k": np.concatenate([_widths(x, 6), -_widths(x, 6)])},
       "Boltzmann current-voltage relation."),
    # --- Enzyme kinetics (I = inhibitor concentration, a fixed parameter)
    _E("CompInhib", "Enzyme Kinetics", "y = Vmax*x/(Km*(1 + I/Ki) + x)", "Vmax Km Ki I",
       "Vmax", lambda x, y: {"Km": np.geomspace(1e-3, 1e2, 25) * _xmax_abs(x),
                             "Ki": np.geomspace(1e-3, 1e3, 13), "I": [1.0]},
       "Competitive inhibition; I = inhibitor concentration (fixed).", fixed={"I": 1.0}),
    _E("NoncompInhib", "Enzyme Kinetics", "y = Vmax*x/((Km + x)*(1 + I/Ki))",
       "Vmax Km Ki I", "Vmax",
       lambda x, y: {"Km": np.geomspace(1e-3, 1e2, 25) * _xmax_abs(x),
                     "Ki": np.geomspace(1e-3, 1e3, 13), "I": [1.0]},
       "Noncompetitive inhibition; I = inhibitor concentration (fixed).", fixed={"I": 1.0}),
    _E("UncompInhib", "Enzyme Kinetics", "y = Vmax*x/(Km + x*(1 + I/Ki))", "Vmax Km Ki I",
       "Vmax", lambda x, y: {"Km": np.geomspace(1e-3, 1e2, 25) * _xmax_abs(x),
                             "Ki": np.geomspace(1e-3, 1e3, 13), "I": [1.0]},
       "Uncompetitive inhibition; I = inhibitor concentration (fixed).", fixed={"I": 1.0}),
    _E("MixedModelInhib", "Enzyme Kinetics",
       "y = Vmax*x/(Km*(1 + I/Ki) + x*(1 + I/(alpha*Ki)))", "Vmax Km Ki alpha I", "Vmax",
       lambda x, y: {"Km": np.geomspace(1e-3, 1e2, 20) * _xmax_abs(x),
                     "Ki": np.geomspace(1e-3, 1e3, 9), "alpha": [0.3, 1.0, 3.0], "I": [1.0]},
       "Mixed-model inhibition; I = inhibitor concentration (fixed).", fixed={"I": 1.0}),
    _E("SubstrateInhib", "Enzyme Kinetics", "y = Vmax*x/(Km + x*(1 + x/Ki))", "Vmax Km Ki",
       "Vmax", lambda x, y: {"Km": np.geomspace(1e-3, 1e2, 25) * _xmax_abs(x),
                             "Ki": np.geomspace(1e-2, 1e3, 25) * _xmax_abs(x)},
       "Substrate inhibition."),
    # --- Pharmacology
    _E("OneSiteBind", "Pharmacology", "y = Bmax*x/(k1 + x)", "Bmax k1", "Bmax",
       lambda x, y: {"k1": np.geomspace(1e-3, 1e2, 50) * _xmax_abs(x)},
       "One-site saturation binding."),
    _E("TwoSiteBind", "Pharmacology", "y = Bmax1*x/(k1 + x) + Bmax2*x/(k2 + x)",
       "Bmax1 k1 Bmax2 k2", "Bmax1 Bmax2",
       lambda x, y: {"k1": np.geomspace(1e-3, 1e2, 25) * _xmax_abs(x),
                     "k2": np.geomspace(1e-3, 1e2, 25) * _xmax_abs(x)},
       "Two-site saturation binding.", grid_filter=_ordered("k1", "k2")),
    _E("OneSiteComp", "Pharmacology", "y = A2 + (A1 - A2)/(1 + 10^(x - logx0))",
       "A1 A2 logx0", "A1 A2", lambda x, y: {"logx0": _centres(x, 41)},
       "One-site competition (x = log concentration)."),
    _E("TwoSiteComp", "Pharmacology",
       "y = A2 + (A1 - A2)*(f/(1 + 10^(x - logx0_1)) + (1 - f)/(1 + 10^(x - logx0_2)))",
       "A1 A2 logx0_1 logx0_2 f", "A1 A2",
       lambda x, y: {"logx0_1": _centres(x, 17), "logx0_2": _centres(x, 17),
                     "f": [0.2, 0.4, 0.5, 0.6, 0.8]},
       "Two-site competition (x = log concentration).",
       grid_filter=_ordered("logx0_1", "logx0_2")),
    # --- Rheology
    _E("Bingham", "Rheology", "y = t0 + mu*x", "t0 mu", "t0 mu", lambda x, y: {},
       "Bingham plastic: t0 = yield stress, mu = plastic viscosity."),
    _E("Herschel", "Rheology", "y = t0 + k*x^n", "t0 k n", "t0 k",
       lambda x, y: {"n": np.geomspace(0.1, 3, 40)}, "Herschel-Bulkley (x = shear rate)."),
    _E("Carreau", "Rheology",
       "y = eta_inf + (eta0 - eta_inf)*(1 + (lam*x)^2)^((n - 1)/2)", "eta0 eta_inf lam n",
       "eta0 eta_inf", lambda x, y: {"lam": 1 / _log_centres(x, 25),
                                     "n": np.linspace(0.0, 0.9, 10)},
       "Carreau viscosity model (x = shear rate)."),
    _E("CarreauYasuda", "Rheology",
       "y = eta_inf + (eta0 - eta_inf)*(1 + (lam*x)^a)^((n - 1)/a)", "eta0 eta_inf lam a n",
       "eta0 eta_inf", lambda x, y: {"lam": 1 / _log_centres(x, 20),
                                     "a": [0.5, 1.0, 2.0, 4.0],
                                     "n": np.linspace(0.0, 0.9, 7)},
       "Carreau-Yasuda viscosity model."),
    _E("Cross", "Rheology", "y = eta_inf + (eta0 - eta_inf)/(1 + (lam*x)^m)",
       "eta0 eta_inf lam m", "eta0 eta_inf",
       lambda x, y: {"lam": 1 / _log_centres(x, 25), "m": np.geomspace(0.3, 1.5, 8)},
       "Cross viscosity model."),
    _E("VFT", "Rheology", "y = A*exp(B/(x - T0))", "A B T0", "A",
       lambda x, y: {"B": np.geomspace(1e-2, 1e2, 25) * _span(x),
                     "T0": float(np.min(x)) - _span(x) * np.geomspace(0.01, 10, 20)},
       "Vogel-Fulcher-Tammann (x = temperature > T0)."),
    # --- Statistics (cumulative distributions and densities, scaled by A)
    _E("NormCDF", "Statistics", "y = A*0.5*erfc(-(x - xc)/(s*sqrt(2)))", "A xc s", "A",
       lambda x, y: {"xc": _centres(x, 25), "s": _widths(x, 20)},
       "Normal cumulative distribution × A."),
    _E("LognormalCDF", "Statistics", "y = A*0.5*erfc(-(ln(x) - mu)/(s*sqrt(2)))", "A mu s",
       "A", lambda x, y: {"mu": np.log(_log_centres(x, 25)),
                          "s": np.geomspace(0.05, 3, 15)},
       "Log-normal cumulative distribution × A (x > 0)."),
    _E("WeibullCDF", "Statistics", "y = A*(1 - exp(-(x/b)^a))", "A a b", "A",
       lambda x, y: {"a": np.geomspace(0.3, 8, 14), "b": _log_centres(x, 25)},
       "Weibull cumulative distribution × A (x ≥ 0)."),
    _E("ExponentialCDF", "Statistics", "y = A*(1 - exp(-x/b))", "A b", "A",
       lambda x, y: {"b": _taus(x, 40)}, "Exponential cumulative distribution × A."),
    _E("LogisticCDF", "Statistics", "y = A/(1 + exp(-(x - mu)/s))", "A mu s", "A",
       lambda x, y: {"mu": _centres(x, 25), "s": _widths(x, 20)},
       "Logistic cumulative distribution × A."),
    _E("Rayleigh", "Statistics", "y = A*x/s^2*exp(-x^2/(2*s^2))", "A s", "A",
       lambda x, y: {"s": _widths(x, 40)}, "Rayleigh density × A (x ≥ 0).", abs_keys="s"),
    _E("Gumbel", "Statistics", "y = A/s*exp(-(x - mu)/s - exp(-(x - mu)/s))", "A mu s", "A",
       lambda x, y: {"mu": _centres(x, 25), "s": _widths(x, 20)},
       "Gumbel (extreme value) density × A."),
    # --- Chemical kinetics
    _E("FirstOrder", "Chemical Kinetics", "c = c0*exp(-k*x)", "c0 k", "c0",
       lambda x, y: {"k": 1 / _taus(x, 40)}, "First-order decay: c(t) = c0·e^(−kt)."),
    _E("SecondOrder", "Chemical Kinetics", "c = c0/(1 + c0*k*x)", "c0 k",
       guess=_second_order_guess, description="Second-order decay (equal concentrations)."),
    _E("ZeroOrder", "Chemical Kinetics", "c = c0 - k*x", "c0 k", "c0 k", lambda x, y: {},
       "Zero-order decay."),
    _E("NthOrder", "Chemical Kinetics", "c = (c0^(1 - n) + (n - 1)*k*x)^(1/(1 - n))",
       "c0 k n", guess=_nth_order_guess, description="n-th order decay (n ≠ 1)."),
    _E("Consecutive", "Chemical Kinetics", "c = A0*k1/(k2 - k1)*(exp(-k1*x) - exp(-k2*x))",
       "A0 k1 k2", "A0", lambda x, y: {"k1": 1 / _taus(x, 22), "k2": 1 / _taus(x, 22)},
       "Intermediate B of A → B → C (first order).",
       grid_filter=lambda v: v["k1"] != v["k2"]),
    _E("Eyring", "Chemical Kinetics",
       "k = (1.380649e-23*x/6.62607015e-34)*exp(dS/8.314462618)*exp(-dH/(8.314462618*x))",
       "dH dS", "", lambda x, y: {"dH": np.geomspace(1e3, 3e5, 40),
                                  "dS": np.linspace(-300, 100, 41)},
       "Eyring equation; x = T in kelvin, ΔH in J/mol, ΔS in J/(mol·K)."),
    # --- Adsorption isotherms
    _E("Sips", "Adsorption Isotherm", "q = qm*(K*x)^n/(1 + (K*x)^n)", "qm K n", "qm",
       lambda x, y: {"K": np.geomspace(1e-3, 1e3, 25) / _xmax_abs(x),
                     "n": np.geomspace(0.2, 5, 12)}, "Sips (Langmuir-Freundlich) isotherm."),
    _E("RedlichPeterson", "Adsorption Isotherm", "q = K*x/(1 + a*x^b)", "K a b", "K",
       lambda x, y: {"a": np.geomspace(1e-4, 1e4, 25), "b": np.linspace(0.1, 1.0, 10)},
       "Redlich-Peterson isotherm."),
    _E("Toth", "Adsorption Isotherm", "q = qm*x/(b + x^t)^(1/t)", "qm b t", "qm",
       lambda x, y: {"b": np.geomspace(1e-4, 1e4, 30), "t": np.linspace(0.2, 2.0, 10)},
       "Toth isotherm."),
]
_LIB.extend(_EXT)

# also-categories and derived parameters for the original entries
for _name, _cats in {"Allometric1": ("Origin Basic Functions",),
                     "Boltzmann": ("Origin Basic Functions",),
                     "ExpAssoc": ("Origin Basic Functions",),
                     "Gauss": ("Origin Basic Functions",),
                     "GaussAmp": ("Origin Basic Functions",),
                     "Logistic": ("Origin Basic Functions", "Pharmacology"),
                     "LogNormal": ("Origin Basic Functions",),
                     "Lorentz": ("Origin Basic Functions", "Spectroscopy"),
                     "Rational0": ("Origin Basic Functions",),
                     "Voigt": ("Origin Basic Functions", "Spectroscopy"),
                     "PsdVoigt1": ("Spectroscopy",),
                     "DoseResp": ("Pharmacology",), "Hill": ("Pharmacology",
                                                             "Enzyme Kinetics"),
                     "MichaelisMenten": ("Enzyme Kinetics",),
                     "BoxLucas1": ("Chemical Kinetics",)}.items():
    next(f for f in _LIB if f.name == _name).meta["also"] = _cats

_S2 = math.sqrt(2 * _LN2)
DERIVED: dict[str, list[tuple[str, Callable[[dict], float]]]] = {
    "Gauss": [("FWHM", lambda p: abs(p["w"]) * _S2),
              ("Height", lambda p: p["A"] / (abs(p["w"]) * math.sqrt(math.pi / 2)))],
    "Gaussian": [("Height", lambda p: p["A"] * math.sqrt(4 * _LN2 / math.pi) / abs(p["w"]))],
    "GaussAmp": [("FWHM", lambda p: 2 * _S2 * abs(p["w"])),
                 ("Area", lambda p: p["A"] * abs(p["w"]) * math.sqrt(2 * math.pi))],
    "Lorentz": [("Height", lambda p: 2 * p["A"] / (math.pi * abs(p["w"])))],
    "Voigt": [("FWHM", lambda p: voigt_fwhm(p["wG"], p["wL"]))],
    "PearsonVII": [("Height", lambda p: p["A"] * 2 * math.gamma(p["mu"]) * math.sqrt(
        2 ** (1 / p["mu"]) - 1) / (math.sqrt(math.pi) * math.gamma(p["mu"] - 0.5)
                                   * abs(p["w"])))],
    "Laplace": [("Height", lambda p: p["A"] / (2 * abs(p["w"]))),
                ("FWHM", lambda p: 2 * _LN2 * abs(p["w"]))],
    "Logistpk": [("Area", lambda p: 4 * p["A"] * abs(p["w"]))],
    "BiGaussian": [("FWHM", lambda p: _S2 * (abs(p["w1"]) + abs(p["w2"]))),
                   ("Area", lambda p: p["H"] * math.sqrt(math.pi / 2)
                    * (abs(p["w1"]) + abs(p["w2"])))],
    "ExpDec1": [("Half-life", lambda p: p["t1"] * _LN2), ("k", lambda p: 1 / p["t1"])],
    "ExpDecay1": [("Half-life", lambda p: p["t1"] * _LN2), ("k", lambda p: 1 / p["t1"])],
    "ExpDec2": [("Half-life 1", lambda p: p["t1"] * _LN2),
                ("Half-life 2", lambda p: p["t2"] * _LN2)],
    "FirstOrder": [("Half-life", lambda p: _LN2 / p["k"])],
    "SecondOrder": [("Half-life", lambda p: 1 / (p["k"] * p["c0"]))],
    "BoxLucas1": [("Half-life", lambda p: _LN2 / p["b"])],
    "DoseResp": [("EC50", lambda p: 10 ** p["LOGx0"]),
                 ("Span", lambda p: p["A2"] - p["A1"])],
    "Logistic": [("EC50", lambda p: p["x0"]), ("Span", lambda p: p["A2"] - p["A1"])],
    "OneSiteComp": [("IC50", lambda p: 10 ** p["logx0"])],
    "Arrhenius": [("Ea (kJ/mol)", lambda p: p["Ea"] / 1000)],
    "Langmuir": [("1/KL", lambda p: 1 / p["KL"])],
    "Sine": [("Period", lambda p: 2 * abs(p["w"])), ("Frequency", lambda p: 1 / (2 * abs(
        p["w"])))],
    "SineDamp": [("Period", lambda p: 2 * abs(p["w"]))],
    "Eyring": [("ΔH (kJ/mol)", lambda p: p["dH"] / 1000)],
}
for _f in _LIB:
    if _f.name in DERIVED:
        _f.meta["derived"] = DERIVED[_f.name]

ALIASES = {"sgompertz": "Gompertz", "srichards1": "Richards", "sweibull2": "Weibull",
           "hill1": "Hill", "emg": "GaussMod", "parabola": "Poly2", "quadratic": "Poly2",
           "cubic": "Poly3", "psdvoigt": "PsdVoigt1", "bigauss": "BiGaussian",
           "exponentialdecay": "ExpDecay1"}
_ORDER = ["Origin Basic Functions", "Exponential", "Growth/Sigmoidal", "Hyperbola",
          "Logarithm", "Peak Functions", "Chromatography", "Spectroscopy", "Piecewise",
          "Polynomial", "Power", "Rational", "Waveform", "Electrophysiology",
          "Enzyme Kinetics", "Pharmacology", "Rheology", "Statistics", "Chemical Kinetics",
          "Adsorption Isotherm"]


def _rebuild() -> None:
    LIBRARY.clear()
    LIBRARY.update({f.name: f for f in _LIB})
    cats = list(dict.fromkeys([c for c in _ORDER] + [f.category for f in _LIB]))
    used = {f.category for f in _LIB} | {c for f in _LIB for c in f.meta.get("also", ())}
    CATEGORIES[:] = [c for c in cats if c in used]


LIBRARY: dict[str, FitFunction] = {}
CATEGORIES: list[str] = []
_rebuild()


def register(f: FitFunction) -> None:
    """Add (or replace) a function of the user's library (Fitting Function Organizer)."""
    _LIB[:] = [g for g in _LIB if g.name != f.name or g.category != f.category]
    _LIB.append(f)
    _rebuild()


def unregister(name: str, category: str | None = None) -> None:
    _LIB[:] = [g for g in _LIB if not (g.name == name and g.user_defined and
                                       (category is None or g.category == category))]
    _rebuild()


def library_size() -> int:
    """Built-in functions: the library plus Poly2–Poly9 and Line."""
    return sum(1 for f in _LIB if not f.user_defined) + 9

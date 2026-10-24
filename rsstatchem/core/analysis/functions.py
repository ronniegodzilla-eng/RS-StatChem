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

    def __call__(self, x, *p) -> np.ndarray:
        x = np.asarray(x, float)
        with np.errstate(all="ignore"):
            out = np.asarray(self.func(x, *p), float)
        return np.broadcast_to(out, x.shape).copy() if out.shape != x.shape else out

    @property
    def is_linear(self) -> bool:
        return self.basis is not None

    def initial_guess(self, x, y, w=None) -> dict[str, float]:
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        ok = np.isfinite(x) & np.isfinite(y)
        w = None if w is None else np.asarray(w, float)
        if w is not None:
            ok &= np.isfinite(w) & (w > 0)
            w = w[ok]
        x, y = x[ok], y[ok]
        if self.basis is not None:
            coef = _wlstsq(self.basis(x), y, w)[0]
            return dict(zip(self.params, map(float, coef)))
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
                cols.append(np.broadcast_to(np.asarray(f.func(x, *args), float), x.shape))
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

LIBRARY: dict[str, FitFunction] = {f.name: f for f in _LIB}
CATEGORIES: list[str] = list(dict.fromkeys(f.category for f in _LIB))


@lru_cache(maxsize=None)
def polynomial(order: int) -> FitFunction:
    """Origin Polynomial: y = Intercept + B1*x + B2*x^2 + ... (Line for order 1)."""
    if not 1 <= order <= 9:
        raise ValueError("Polynomial order must be between 1 and 9.")
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
    for f in _LIB:
        if f.name.lower() == key:
            return f
    if key in ("line", "linear"):
        return polynomial(1)
    if key.startswith("poly") and key[4:].isdigit():
        return polynomial(int(key[4:]))
    return None


def functions_in(category: str) -> list[FitFunction]:
    return [f for f in _LIB if f.category == category]


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
_FUNCS: dict[str, tuple[Callable, int]] = {
    "exp": (np.exp, 1), "ln": (np.log, 1), "log": (np.log10, 1), "log10": (np.log10, 1),
    "log2": (np.log2, 1), "sqrt": (np.sqrt, 1), "abs": (np.abs, 1), "sin": (np.sin, 1),
    "cos": (np.cos, 1), "tan": (np.tan, 1), "asin": (np.arcsin, 1), "acos": (np.arccos, 1),
    "atan": (np.arctan, 1), "sinh": (np.sinh, 1), "cosh": (np.cosh, 1), "tanh": (np.tanh, 1),
    "erf": (special.erf, 1), "erfc": (special.erfc, 1), "gamma": (special.gamma, 1),
    "sign": (np.sign, 1), "pow": (np.power, 2), "atan2": (np.arctan2, 2),
}
_CONSTS = {"pi": math.pi}
_BINOPS = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
           ast.Div: np.divide, ast.Pow: np.power, ast.Mod: np.mod}
_UNOPS = {ast.USub: np.negative, ast.UAdd: np.positive}
_MAX_LEN = 1000


class FunctionSyntaxError(ValueError):
    pass


def parse_user_function(text: str, independent: str = "x") -> FitFunction:
    """Compile ``y = <expression in x and parameters>`` into a :class:`FitFunction`.

    Parameters are all names that are not ``x``, a function or ``pi``; their order is the
    order of first appearance. Raises :class:`FunctionSyntaxError` on anything else.
    """
    src = (text or "").strip()
    if not src:
        raise FunctionSyntaxError("Enter a function, e.g. y = A*exp(-x/t) + y0")
    if len(src) > _MAX_LEN:
        raise FunctionSyntaxError("The function is too long.")
    dependent = "y"
    if "=" in src:
        lhs, rhs = src.split("=", 1)
        lhs = lhs.strip()
        if lhs.endswith(f"({independent})"):
            lhs = lhs[: -len(independent) - 2].strip()
        if not lhs.isidentifier():
            raise FunctionSyntaxError(f"'{lhs}' is not a valid name for the dependent variable.")
        dependent, src = lhs, rhs.strip()
    src = src.replace("^", "**")
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as exc:
        raise FunctionSyntaxError(f"Syntax error in the function: {exc.msg}") from None
    params: list[str] = []

    def comp(node) -> Callable[[dict], object]:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            v = float(node.value)
            return lambda env: v
        if isinstance(node, ast.Name):
            name = node.id
            if name == independent:
                return lambda env: env["\0x"]
            if name in _CONSTS:
                v = _CONSTS[name]
                return lambda env: v
            if name in _FUNCS:
                raise FunctionSyntaxError(f"'{name}' is a function; write {name}(...).")
            if name == dependent:
                raise FunctionSyntaxError(f"'{name}' cannot appear on the right-hand side.")
            if name.startswith("_"):
                raise FunctionSyntaxError(f"'{name}' is not a valid parameter name.")
            if name not in params:
                params.append(name)
            return lambda env: env[name]
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            op, a, b = _BINOPS[type(node.op)], comp(node.left), comp(node.right)
            return lambda env: op(a(env), b(env))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNOPS:
            op, a = _UNOPS[type(node.op)], comp(node.operand)
            return lambda env: op(a(env))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
                fname = getattr(node.func, "id", "?")
                raise FunctionSyntaxError(f"Unknown function '{fname}'. Allowed: "
                                          + ", ".join(sorted(_FUNCS)))
            fn, nargs = _FUNCS[node.func.id]
            if node.keywords or len(node.args) != nargs:
                raise FunctionSyntaxError(f"{node.func.id}() takes {nargs} argument(s).")
            args = [comp(a) for a in node.args]
            return lambda env: fn(*(a(env) for a in args))
        segment = ast.get_source_segment(src, node) or type(node).__name__
        raise FunctionSyntaxError(f"'{segment}' is not allowed in a fitting function.")

    try:
        body = comp(tree.body)
    except RecursionError:
        raise FunctionSyntaxError("The function is nested too deeply.") from None
    if not params:
        raise FunctionSyntaxError("The function has no parameters to fit.")
    names = tuple(params)

    def func(x, *p):
        env = dict(zip(names, (float(v) for v in p)))
        env["\0x"] = np.asarray(x, float)
        return body(env)

    return FitFunction("User", "User Defined", f"{dependent} = {text.split('=', 1)[-1].strip()}"
                       if "=" in text else f"y = {text.strip()}", names, func,
                       description="User-defined function.", user_defined=True,
                       meta={"dependent": dependent, "source": text.strip()})

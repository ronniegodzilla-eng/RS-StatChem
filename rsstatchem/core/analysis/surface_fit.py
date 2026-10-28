"""Surface fitting z = f(x, y) on XYZ columns or matrices (Origin's NLFit with Surface Fitting
functions, OriginPro) and Simulate Surface.

The surface functions are :class:`FitFunction` objects of two variables (``nvar = 2``) with
exact analytic Jacobians from the expression compiler, so they use the same solver, weights,
bounds and constraints as a curve fit (:func:`fitting.fit_curve` with an (n, 2) x array).
User surfaces are typed as ``z = a*x + b*y^2 + c``.
"""
from __future__ import annotations

import math

import numpy as np

from rsstatchem.core.analysis import fitting as fit
from rsstatchem.core.analysis import functions as fl
from rsstatchem.core.analysis.fitting import FitResult
from rsstatchem.core.analysis.functions import FitFunction


def _peak_guess(X, z, lorentz: bool = False):
    base = float(np.median(z))
    dev = z - base
    i = int(np.argmax(np.abs(dev)))
    A = float(dev[i])
    xc, yc = float(X[i, 0]), float(X[i, 1])
    half = np.abs(dev) >= abs(A) / 2
    sx = float(np.ptp(X[half, 0])) if half.sum() > 1 else float(np.ptp(X[:, 0])) / 10
    sy = float(np.ptp(X[half, 1])) if half.sum() > 1 else float(np.ptp(X[:, 1])) / 10
    sx = sx or float(np.ptp(X[:, 0])) / 10 or 1.0
    sy = sy or float(np.ptp(X[:, 1])) / 10 or 1.0
    k = 0.5 if lorentz else 1 / 2.3548
    return {"z0": base, "A": A, "xc": xc, "yc": yc, "w1": sx * k, "w2": sy * k,
            "theta": 0.0, "mu": 0.5}


def _S(name, formula, params, linear="", grid=None, guess=None, desc="") -> FitFunction:
    f = fl.parse_function(formula, ("x", "y"), params=tuple(params.split()), name=name,
                          category="Surface Fitting", description=desc)
    f.user_defined = False
    f.linear = tuple(linear.split())
    f.grid = grid
    f.meta.update({"guess": guess} if guess else {})
    return f


def _span_x(X, i):
    s = float(np.ptp(X[:, i]))
    return s if s > 0 else 1.0


SURFACES: dict[str, FitFunction] = {f.name: f for f in [
    _S("Plane", "z = z0 + a*x + b*y", "z0 a b", "z0 a b", lambda X, z: {},
       desc="Plane."),
    _S("Parabola2D", "z = z0 + a*x + b*y + c*x^2 + d*y^2", "z0 a b c d", "z0 a b c d",
       lambda X, z: {}, desc="Paraboloid without the cross term."),
    _S("Poly2D", "z = z0 + a*x + b*y + c*x^2 + d*y^2 + f*x*y", "z0 a b c d f",
       "z0 a b c d f", lambda X, z: {}, desc="Second-order polynomial surface."),
    _S("Gauss2D", "z = z0 + A*exp(-0.5*((x - xc)/w1)^2 - 0.5*((y - yc)/w2)^2)",
       "z0 A xc w1 yc w2", "z0 A", guess=lambda X, z: _peak_guess(X, z),
       desc="2D Gaussian: A = height, w1/w2 = σ in x/y."),
    _S("Gauss2DRot",
       "z = z0 + A*exp(-0.5*(((x - xc)*cos(theta) + (y - yc)*sin(theta))/w1)^2 "
       "- 0.5*(((y - yc)*cos(theta) - (x - xc)*sin(theta))/w2)^2)",
       "z0 A xc yc w1 w2 theta", "z0 A", guess=lambda X, z: _peak_guess(X, z),
       desc="Rotated 2D Gaussian (theta in radians)."),
    _S("Lorentz2D", "z = z0 + A/((1 + ((x - xc)/w1)^2)*(1 + ((y - yc)/w2)^2))",
       "z0 A xc w1 yc w2", "z0 A", guess=lambda X, z: _peak_guess(X, z, True),
       desc="2D Lorentzian (product form): A = height, w1/w2 = HWHM."),
    _S("Voigt2D",
       "z = z0 + A*(mu/(1 + 4*((x - xc)/w1)^2) + (1 - mu)*exp(-4*ln(2)*((x - xc)/w1)^2))"
       "*(mu/(1 + 4*((y - yc)/w2)^2) + (1 - mu)*exp(-4*ln(2)*((y - yc)/w2)^2))",
       "z0 A xc w1 yc w2 mu", "z0 A",
       guess=lambda X, z: {**_peak_guess(X, z), "w1": _peak_guess(X, z)["w1"] * 2.3548,
                           "w2": _peak_guess(X, z)["w2"] * 2.3548},
       desc="2D pseudo-Voigt (product form): A = height, w1/w2 = FWHM, mu = Lorentz fraction."),
    _S("Exponential2D", "z = z0 + A*exp(-x/t1 - y/t2)", "z0 A t1 t2", "z0 A",
       lambda X, z: {"t1": np.concatenate([np.geomspace(.05, 5, 15) * _span_x(X, 0),
                                           -np.geomspace(.05, 5, 15) * _span_x(X, 0)]),
                     "t2": np.concatenate([np.geomspace(.05, 5, 15) * _span_x(X, 1),
                                           -np.geomspace(.05, 5, 15) * _span_x(X, 1)])},
       desc="Exponential surface."),
    _S("Power2D", "z = z0 + A*x^a*y^b", "z0 A a b", "z0 A",
       lambda X, z: {"a": np.linspace(-3, 3, 25), "b": np.linspace(-3, 3, 25)},
       desc="Power surface (x, y > 0)."),
    _S("Extreme2D",
       "z = z0 + A*exp(-exp(-(x - xc)/w1) - (x - xc)/w1 - exp(-(y - yc)/w2) - (y - yc)/w2 + 2)",
       "z0 A xc w1 yc w2", "z0 A", guess=lambda X, z: _peak_guess(X, z),
       desc="2D extreme-value peak: A = height."),
]}


def get_surface(name: str) -> FitFunction | None:
    for k, f in SURFACES.items():
        if k.lower() == name.strip().lower():
            return f
    return None


def resolve_surface(model) -> FitFunction:
    if isinstance(model, FitFunction):
        if model.nvar != 2:
            raise ValueError("A surface fit needs a function of x and y.")
        return model
    f = get_surface(model)
    return f if f is not None else fl.parse_function(model, ("x", "y"))


def surface_fit(x, y, z, model, params0=None, fixed=None, bounds=None, weights=None,
                constraints=None, scale_errors: bool = True) -> FitResult:
    """Fit z = f(x, y) to scattered XYZ data."""
    f = resolve_surface(model)
    X = np.column_stack([np.asarray(x, float).ravel(), np.asarray(y, float).ravel()])
    return fit.fit_curve(X, np.asarray(z, float).ravel(), f, params0=params0, fixed=fixed,
                         bounds=bounds, weights=weights, constraints=constraints,
                         scale_errors=scale_errors)


def matrix_xyz(data, mapping: tuple[float, float, float, float]):
    """Matrix (rows = y, columns = x) with X/Y mapping (x1, x2, y1, y2) → x, y, z vectors."""
    data = np.asarray(data, float)
    ny, nx = data.shape
    x1, x2, y1, y2 = mapping
    gx = np.linspace(x1, x2, nx) if nx > 1 else np.array([x1])
    gy = np.linspace(y1, y2, ny) if ny > 1 else np.array([y1])
    GX, GY = np.meshgrid(gx, gy)
    return GX.ravel(), GY.ravel(), data.ravel()


def matrix_fit(data, mapping, model, **kw) -> FitResult:
    x, y, z = matrix_xyz(data, mapping)
    return surface_fit(x, y, z, model, **kw)


def simulate_surface(model, params, x_range=(0.0, 1.0), y_range=(0.0, 1.0),
                     nx: int = 41, ny: int = 41) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Origin's Simulate Surface: the grid (gx, gy) and the matrix Z[row = y, col = x]."""
    f = resolve_surface(model)
    p = [float(params[k]) for k in f.params] if isinstance(params, dict) else list(params)
    gx = np.linspace(*x_range, nx)
    gy = np.linspace(*y_range, ny)
    GX, GY = np.meshgrid(gx, gy)
    Z = f(np.stack([GX.ravel(), GY.ravel()], axis=1), *p).reshape(GX.shape)
    return gx, gy, Z


def simulate_curve(model, params, x_range=(0.0, 1.0), n: int = 200):
    """Origin's Simulate Curve: x grid and f(x)."""
    f = fit.resolve_model(model)
    p = [float(params[k]) for k in f.params] if isinstance(params, dict) else list(params)
    xx = np.linspace(*x_range, n)
    return xx, f(xx, *p)


def surface_r2(res: FitResult) -> float:
    return res.r2 if math.isfinite(res.r2) else float("nan")

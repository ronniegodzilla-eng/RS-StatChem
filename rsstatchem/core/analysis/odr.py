"""Errors-in-variables fitting (Origin's Fit Linear with X Error, Orthogonal Distance
Regression algorithm and Implicit Fit), implemented in-house because ``scipy.odr`` is
deprecated (removed in SciPy 1.19).

* :func:`odr_fit` – weighted orthogonal distance regression of y = f(x; β) (Boggs & Rogers):
  minimise Σ wyᵢ(f(xᵢ + δᵢ; β) − yᵢ)² + wxᵢδᵢ² over β and the x corrections δ with
  ``least_squares`` on the augmented vector (β, δ) and its exact sparse-structure Jacobian.
  The covariance of β is the β block of (JᵀJ)⁻¹, i.e. (Jβᵀ diag(w_eff) Jβ)⁻¹ with
  w_eff = wy·wx/(wy·f′² + wx), scaled by the residual variance as ODRPACK does.
* :func:`york_fit` – straight line with errors in both x and y (and correlated errors),
  York et al. (2004), *Am. J. Phys.* 72, 367, "Unified equations for the slope, intercept,
  and standard errors of the best straight line".
* :func:`implicit_fit` – f(x, y; β) = 0 (circle, ellipse, user equations) by orthogonal
  distance: each point is projected onto the curve (weighted closest point), the outer
  least-squares problem runs over β.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, sparse

from rsstatchem.core.analysis import fitting as fit
from rsstatchem.core.analysis import functions as fl
from rsstatchem.core.analysis.functions import FitFunction


# ====================================================================== explicit ODR
@dataclass
class ODRResult:
    model: FitFunction
    x: np.ndarray
    y: np.ndarray
    wx: np.ndarray
    wy: np.ndarray
    values: np.ndarray
    errors: np.ndarray
    cov: np.ndarray            # scaled covariance (free parameters in place)
    cov_unscaled: np.ndarray
    free: np.ndarray
    delta: np.ndarray          # x corrections
    eps: np.ndarray            # y corrections (fitted − observed at x + δ)
    sum_square: float
    res_var: float
    success: bool
    message: str
    nfev: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def names(self):
        return self.model.params

    @property
    def params(self) -> dict[str, float]:
        return dict(zip(self.names, map(float, self.values)))

    @property
    def n(self) -> int:
        return len(self.x)

    @property
    def dof(self) -> int:
        return self.n - int(self.free.sum())

    def predict(self, x) -> np.ndarray:
        return self.model(np.asarray(x, float), *self.values)

    @property
    def xfit(self) -> np.ndarray:
        return self.x + self.delta

    @property
    def yfit(self) -> np.ndarray:
        return self.predict(self.xfit)

    @property
    def r2(self) -> float:
        yb = float(np.mean(self.y))
        tss = float(np.sum((self.y - yb) ** 2))
        rss = float(np.sum((self.y - self.predict(self.x)) ** 2))
        return 1 - rss / tss if tss > 0 else float("nan")


def _dfdx(f: FitFunction, x, p) -> np.ndarray:
    h = 1e-6 * (np.abs(x) + 1e-3 * (float(np.ptp(x)) or 1.0))
    return (f(x + h, *p) - f(x - h, *p)) / (2 * h)


def _as_w(s, w, n) -> np.ndarray:
    if w is not None:
        return np.broadcast_to(np.asarray(w, float), (n,)).astype(float)
    if s is not None:
        s = np.broadcast_to(np.asarray(s, float), (n,))
        return 1.0 / s ** 2
    return np.ones(n)


def odr_fit(x, y, model, sx=None, sy=None, wx=None, wy=None, params0=None, fixed=None,
            scale_errors: bool = True) -> ODRResult:
    """Orthogonal distance regression. Give the x / y standard deviations (``sx``, ``sy``)
    or weights (``wx = 1/sx²``, ``wy = 1/sy²``); without them both weights are 1."""
    f = fit.resolve_model(model)
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    n = len(x)
    wx, wy = _as_w(sx, wx, n), _as_w(sy, wy, n)
    ok = np.isfinite(x) & np.isfinite(y) & (wx > 0) & (wy > 0)
    x, y, wx, wy = x[ok], y[ok], wx[ok], wy[ok]
    n = len(x)
    # start: an ordinary (y-only) weighted fit
    start = fit.fit_curve(x, y, f, params0=params0, fixed=fixed, weights=wy)
    names = f.params
    free = start.free.copy()
    idx = np.flatnonzero(free)
    k = len(idx)
    if n <= k:
        raise ValueError("Not enough points for this model.")
    beta0 = start.values.copy()
    swx, swy = np.sqrt(wx), np.sqrt(wy)

    def unpack(z):
        b = beta0.copy()
        b[idx] = z[:k]
        return b, z[k:]

    def resid(z):
        b, d = unpack(z)
        r1 = swy * (f(x + d, *b) - y)
        r2 = swx * d
        out = np.concatenate([r1, r2])
        return np.where(np.isfinite(out), out, 1e10)

    def jac(z):
        b, d = unpack(z)
        Jb = f.jacobian(x + d, b)[:, idx] * swy[:, None]
        fp = _dfdx(f, x + d, b) * swy
        top = sparse.hstack([sparse.csr_matrix(Jb), sparse.diags(fp)])
        bot = sparse.hstack([sparse.csr_matrix((n, k)), sparse.diags(swx)])
        J = sparse.vstack([top, bot]).tocsr()
        return J.toarray() if n <= 1500 else J

    z0 = np.concatenate([beta0[idx], np.zeros(n)])
    method = "lm" if n <= 1500 else "trf"
    r = optimize.least_squares(resid, z0, jac=jac, method=method, x_scale="jac",
                               ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=2000 * (k + 1))
    b, d = unpack(r.x)
    fp = _dfdx(f, x + d, b)
    weff = wy * wx / (wy * fp ** 2 + wx)
    Jb = f.jacobian(x + d, b)[:, idx]
    cu = fit._cov_from_jac(Jb * np.sqrt(weff)[:, None])
    ss = 2 * float(r.cost)
    dof = n - k
    res_var = ss / dof if dof > 0 else float("nan")
    cs = cu * res_var if scale_errors and dof > 0 else cu
    cov = np.zeros((len(names), len(names)))
    cov_u = np.zeros_like(cov)
    cov[np.ix_(idx, idx)] = cs
    cov_u[np.ix_(idx, idx)] = cu
    err = np.sqrt(np.maximum(np.diag(cov), 0))
    err[~free] = np.nan
    eps = f(x + d, *b) - y
    return ODRResult(f, x, y, wx, wy, b, err, cov, cov_u, free, d, eps, ss, res_var,
                     bool(r.success), fit._status(r), int(r.nfev))


# ====================================================================== York line
@dataclass
class YorkResult:
    slope: float
    intercept: float
    se_slope: float
    se_intercept: float
    cov_ab: float
    S: float               # weighted sum of squared residuals (χ²)
    mswd: float            # S/(n − 2), reduced χ²
    n: int
    iterations: int
    x_adjusted: np.ndarray
    W: np.ndarray

    @property
    def p_value(self) -> float:
        from scipy import stats
        return float(stats.chi2.sf(self.S, self.n - 2)) if self.n > 2 else float("nan")


def york_fit(x, y, sx, sy, r=0.0, tol: float = 1e-15, max_iter: int = 1000) -> YorkResult:
    """Best straight line y = a + b·x with standard deviations ``sx``, ``sy`` (arrays or
    scalars) and error correlation ``r`` (York et al. 2004, section III algorithm)."""
    X = np.asarray(x, float)
    Y = np.asarray(y, float)
    n = len(X)
    wX = 1 / np.broadcast_to(np.asarray(sx, float), (n,)) ** 2
    wY = 1 / np.broadcast_to(np.asarray(sy, float), (n,)) ** 2
    r = np.broadcast_to(np.asarray(r, float), (n,))
    alpha = np.sqrt(wX * wY)
    b = np.polyfit(X, Y, 1)[0]
    it = 0
    for it in range(1, max_iter + 1):
        W = wX * wY / (wX + b * b * wY - 2 * b * r * alpha)
        Xb, Yb = np.sum(W * X) / np.sum(W), np.sum(W * Y) / np.sum(W)
        U, V = X - Xb, Y - Yb
        beta = W * (U / wY + b * V / wX - (b * U + V) * r / alpha)
        b_new = np.sum(W * beta * V) / np.sum(W * beta * U)
        if abs(b_new - b) <= tol * max(abs(b_new), 1e-300):
            b = b_new
            break
        b = b_new
    W = wX * wY / (wX + b * b * wY - 2 * b * r * alpha)
    Xb, Yb = np.sum(W * X) / np.sum(W), np.sum(W * Y) / np.sum(W)
    U, V = X - Xb, Y - Yb
    beta = W * (U / wY + b * V / wX - (b * U + V) * r / alpha)
    a = Yb - b * Xb
    xa = Xb + beta
    xbar = np.sum(W * xa) / np.sum(W)
    u = xa - xbar
    var_b = 1 / np.sum(W * u * u)
    var_a = 1 / np.sum(W) + xbar ** 2 * var_b
    S = float(np.sum(W * (Y - b * X - a) ** 2))
    return YorkResult(float(b), float(a), math.sqrt(var_b), math.sqrt(var_a),
                      float(-xbar * var_b), S, S / (n - 2) if n > 2 else float("nan"), n, it,
                      xa, W)


# ====================================================================== implicit fit
IMPLICIT = {
    "Circle": ("(x - xc)^2 + (y - yc)^2 - r^2 = 0", "xc yc r",
               "Circle with centre (xc, yc) and radius r."),
    "Ellipse": ("((x - xc)/a)^2 + ((y - yc)/b)^2 - 1 = 0", "xc yc a b",
                "Axis-aligned ellipse with semi-axes a and b."),
    "EllipseRot": ("(((x - xc)*cos(theta) + (y - yc)*sin(theta))/a)^2 + (((y - yc)*cos(theta)"
                   " - (x - xc)*sin(theta))/b)^2 - 1 = 0", "xc yc a b theta",
                   "Rotated ellipse (theta in radians)."),
    "ParabolaI": ("y - yc - A*(x - xc)^2 = 0", "xc yc A", "Parabola (implicit form)."),
}


def implicit_function(name_or_text: str) -> FitFunction:
    if name_or_text in IMPLICIT:
        text, params, desc = IMPLICIT[name_or_text]
        f = fl.parse_function(text, ("x", "y"), params=tuple(params.split()), implicit=True,
                              name=name_or_text, category="Implicit", description=desc)
        f.user_defined = False
        return f
    return fl.parse_function(name_or_text, ("x", "y"), implicit=True)


def _implicit_guess(f: FitFunction, x, y) -> dict:
    name = f.name
    if name in ("Circle", "Ellipse", "EllipseRot"):
        # algebraic (Kasa) circle: x² + y² + D x + E y + F = 0
        A = np.column_stack([x, y, np.ones_like(x)])
        D, E, F = np.linalg.lstsq(A, -(x ** 2 + y ** 2), rcond=None)[0]
        xc, yc = -D / 2, -E / 2
        rr = math.sqrt(max(xc ** 2 + yc ** 2 - F, 1e-300))
        g = {"xc": xc, "yc": yc, "r": rr,
             "a": float(np.sqrt(2) * np.std(x)) or rr, "b": float(np.sqrt(2) * np.std(y)) or rr,
             "theta": 0.0}
        return {p: float(g[p]) for p in f.params}
    if name == "ParabolaI":
        c2, c1, c0 = np.polyfit(x, y, 2)
        xc = -c1 / (2 * c2)
        return {"xc": xc, "yc": c0 - c2 * xc ** 2, "A": c2}
    return {p: 1.0 for p in f.params}


@dataclass
class ImplicitResult:
    model: FitFunction
    x: np.ndarray
    y: np.ndarray
    values: np.ndarray
    errors: np.ndarray
    cov: np.ndarray
    xfit: np.ndarray
    yfit: np.ndarray
    sum_square: float
    dof: int
    success: bool
    message: str

    @property
    def names(self):
        return self.model.params

    @property
    def params(self):
        return dict(zip(self.names, map(float, self.values)))

    @property
    def distances(self) -> np.ndarray:
        return np.hypot(self.x - self.xfit, self.y - self.yfit)

    def curve(self, n: int = 400, pad: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
        """Points on the fitted curve f = 0 (contour of f over the data box)."""
        from matplotlib.figure import Figure
        sx, sy = float(np.ptp(self.x)) or 1.0, float(np.ptp(self.y)) or 1.0
        gx = np.linspace(self.x.min() - pad * sx, self.x.max() + pad * sx, n)
        gy = np.linspace(self.y.min() - pad * sy, self.y.max() + pad * sy, n)
        GX, GY = np.meshgrid(gx, gy)
        Z = self.model(np.stack([GX.ravel(), GY.ravel()], axis=1), *self.values).reshape(GX.shape)
        fig = Figure()
        cs = fig.add_subplot().contour(GX, GY, Z, levels=[0.0])
        segs = cs.allsegs[0] if cs.allsegs else []
        if not segs:
            return np.zeros(0), np.zeros(0)
        out_x, out_y = [], []
        for s in segs:
            out_x += list(s[:, 0]) + [np.nan]
            out_y += list(s[:, 1]) + [np.nan]
        return np.array(out_x[:-1]), np.array(out_y[:-1])


def _project(f, x, y, p, wx, wy, iters: int = 60):
    """Weighted closest points on f(u, v; p) = 0 to the data (x, y)."""
    u, v = x.copy(), y.copy()
    mx, my = 1 / wx, 1 / wy
    hx = 1e-7 * (np.abs(x).max() + float(np.ptp(x)) + 1e-12)
    hy = 1e-7 * (np.abs(y).max() + float(np.ptp(y)) + 1e-12)
    for _ in range(iters):
        P = np.stack([u, v], axis=1)
        f0 = f(P, *p)
        gx = (f(np.stack([u + hx, v], 1), *p) - f(np.stack([u - hx, v], 1), *p)) / (2 * hx)
        gy = (f(np.stack([u, v + hy], 1), *p) - f(np.stack([u, v - hy], 1), *p)) / (2 * hy)
        den = gx * gx * mx + gy * gy * my
        den = np.where(den > 0, den, np.inf)
        lam = (f0 + gx * (x - u) + gy * (y - v)) / den
        un, vn = x - mx * gx * lam, y - my * gy * lam
        step = np.max(np.abs(un - u)) + np.max(np.abs(vn - v))
        u, v = un, vn
        if step < 1e-15 * (1 + np.abs(x).max() + np.abs(y).max()):
            break
    return u, v


def implicit_fit(x, y, model, params0=None, sx=None, sy=None,
                 scale_errors: bool = True) -> ImplicitResult:
    """Fit f(x, y; β) = 0 by weighted orthogonal distance (Origin's Implicit Fit)."""
    f = implicit_function(model) if isinstance(model, str) else model
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    wx, wy = _as_w(sx, None, n), _as_w(sy, None, n)
    g = _implicit_guess(f, x, y)
    if params0:
        g.update({k: float(v) for k, v in dict(params0).items() if v is not None})
    p0 = np.array([g[p] for p in f.params], float)
    if n <= len(p0):
        raise ValueError("Not enough points for this implicit model.")

    def resid(p):
        u, v = _project(f, x, y, p, wx, wy)
        out = np.concatenate([np.sqrt(wx) * (x - u), np.sqrt(wy) * (y - v)])
        return np.where(np.isfinite(out), out, 1e10)

    r = optimize.least_squares(resid, p0, method="lm", x_scale="jac", ftol=1e-14,
                               xtol=1e-14, gtol=1e-14)
    r2 = optimize.least_squares(resid, r.x, method="lm", jac="3-point", x_scale="jac",
                                ftol=1e-15, xtol=1e-15, gtol=1e-15)
    if r2.cost <= r.cost:
        r = r2
    p = r.x
    if f.name in ("Circle",):
        p[2] = abs(p[2])
    if f.name in ("Ellipse", "EllipseRot"):
        p[2], p[3] = abs(p[2]), abs(p[3])
    ss = 2 * float(r.cost)
    dof = n - len(p)
    cov = fit._cov_from_jac(r.jac)
    if scale_errors and dof > 0:
        cov = cov * ss / dof
    u, v = _project(f, x, y, p, wx, wy)
    return ImplicitResult(f, x, y, p, np.sqrt(np.maximum(np.diag(cov), 0)), cov, u, v, ss,
                          dof, bool(r.success), fit._status(r))

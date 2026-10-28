"""Baselines for spectra and chromatograms (Origin's Peak Analyzer "Baseline Mode" page and
Peaks and Baseline tools).

Modes: constant, straight line between the ends, end-points weighted (line fitted to both
end segments), user-defined anchor points (found automatically by local minima, second
derivative or evenly spaced segments; then connected by line / spline / PCHIP or fitted
with a function), asymmetric least squares (ALS, Eilers & Boelens 2005), arPLS (Baek et al.
2015, *Analyst* 140, 250), airPLS (Zhang et al. 2010, *Analyst* 135, 1138), SNIP (Ryan et
al. 1988 / Morháč 1997, with the LLS operator), Shirley (iterative, Proctor & Sherwood
1982), Tougaard (universal cross-section, Tougaard 1989), rolling ball (morphological
opening), an existing dataset, and the I-ModPoly polynomial of :mod:`signal` (read-only).

All functions take x and y in any order and return the baseline in the original order.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import interpolate, ndimage, sparse
from scipy import signal as ss
from scipy.sparse.linalg import spsolve

from rsstatchem.core.analysis import signal as sig

BASELINE_MODES = {
    "constant": "Constant", "line": "Straight Line (end points)",
    "endpoints": "End Points Weighted", "anchors": "User Defined (anchor points)",
    "als": "Asymmetric Least Squares (ALS)", "arpls": "arPLS", "airpls": "airPLS",
    "snip": "SNIP", "shirley": "Shirley (XPS)", "tougaard": "Tougaard (XPS)",
    "rolling_ball": "Rolling Ball", "poly": "Polynomial (I-ModPoly)",
    "dataset": "Use Existing Dataset", "none": "None"}
ANCHOR_METHODS = {"minima": "Local Minima", "second_derivative": "Second Derivative",
                  "even": "Evenly Spaced (segment minima)"}
INTERPOLATIONS = {"linear": "Line", "spline": "Spline", "pchip": "PCHIP (shape-preserving)",
                  "fit": "Fit (polynomial)"}


def _sorted(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    o = np.argsort(x, kind="stable")
    return o, x[o], y[o]


def _unsort(o, values) -> np.ndarray:
    out = np.empty(len(o))
    out[o] = values
    return out


def _second_diff(n: int):
    return sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))


# ---------------------------------------------------------------------- simple modes
def constant(x, y, mode: str = "min", value: float | None = None) -> np.ndarray:
    y = np.asarray(y, float)
    if value is not None:
        c = float(value)
    else:
        c = {"min": np.nanmin, "mean": np.nanmean, "median": np.nanmedian}[mode](y)
    return np.full(len(y), float(c))


def straight_line(x, y, n_end: int = 1) -> np.ndarray:
    """Line through the means of the first and last ``n_end`` points."""
    o, xs, ys = _sorted(x, y)
    k = max(1, min(int(n_end), len(xs) // 2))
    x1, y1 = xs[:k].mean(), ys[:k].mean()
    x2, y2 = xs[-k:].mean(), ys[-k:].mean()
    b = (y2 - y1) / (x2 - x1) if x2 != x1 else 0.0
    return _unsort(o, y1 + b * (xs - x1))


def endpoints_weighted(x, y, fraction: float = 0.05) -> np.ndarray:
    """Least-squares line through the first and last ``fraction`` of the points."""
    o, xs, ys = _sorted(x, y)
    k = max(2, int(round(fraction * len(xs))))
    sel = np.r_[np.arange(k), np.arange(len(xs) - k, len(xs))]
    c = np.polyfit(xs[sel], ys[sel], 1)
    return _unsort(o, np.polyval(c, xs))


# ---------------------------------------------------------------------- anchor points
def _smooth(ys, window):
    window = int(window)
    if window >= 5 and len(ys) > window:
        window += (window + 1) % 2
        return ss.savgol_filter(ys, window, 2)
    return ys


def find_anchors(x, y, n: int = 8, method: str = "minima", smooth: int = 0,
                 include_ends: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Anchor points for a user-defined baseline (Origin's automatic anchor finding)."""
    o, xs, ys = _sorted(x, y)
    m = len(xs)
    n = max(2, int(n))
    sm = _smooth(ys, smooth or max(5, m // 50))
    if method == "even":
        edges = np.linspace(0, m, n + 1).astype(int)
        idx = [a + int(np.argmin(sm[a:b])) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    elif method == "second_derivative":
        d1 = np.gradient(sm, xs)
        d2 = np.gradient(d1, xs)
        score = np.abs(d1) / (np.abs(d1).max() or 1) + np.abs(d2) / (np.abs(d2).max() or 1) \
            + (sm - sm.min()) / (np.ptp(sm) or 1)
        edges = np.linspace(0, m, n + 1).astype(int)
        idx = [a + int(np.argmin(score[a:b])) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    elif method == "minima":
        cand, _ = ss.find_peaks(-sm)
        cand = list(cand)
        if not cand:
            return find_anchors(x, y, n, "even", smooth, include_ends)
        # keep the lowest minimum of each of n segments
        edges = np.linspace(0, m, n + 1).astype(int)
        idx = []
        for a, b in zip(edges[:-1], edges[1:]):
            inside = [c for c in cand if a <= c < b]
            if inside:
                idx.append(min(inside, key=lambda c: sm[c]))
            elif b > a:
                idx.append(a + int(np.argmin(sm[a:b])))
    else:
        raise ValueError(f"Unknown anchor method '{method}'")
    if include_ends:
        idx = [0] + list(idx) + [m - 1]
    idx = sorted(set(int(i) for i in idx))
    idx = _lower_envelope(xs, sm, idx)
    return xs[idx], sm[idx] if smooth else ys[idx]


def _lower_envelope(xs, v, idx: list[int]) -> list[int]:
    """Drop anchors lying clearly above the line through their neighbours (valleys between
    overlapping peaks are not baseline points)."""
    d = np.diff(v)
    noise = 1.4826 * float(np.median(np.abs(d - np.median(d)))) / math.sqrt(2) if len(d) else 0
    tol = 4 * noise + 1e-12 * (float(np.ptp(v)) or 1.0)
    idx = list(idx)
    changed = True
    while changed and len(idx) > 3:
        changed = False
        worst, k_worst = tol, None
        for k in range(1, len(idx) - 1):
            a, b, c = idx[k - 1], idx[k], idx[k + 1]
            line = v[a] + (v[c] - v[a]) * (xs[b] - xs[a]) / (xs[c] - xs[a])
            if v[b] - line > worst:
                worst, k_worst = v[b] - line, k
        if k_worst is not None:
            del idx[k_worst]
            changed = True
    return idx


def anchors_baseline(x, y, ax, ay, interp: str = "linear", order: int = 3) -> np.ndarray:
    """Baseline through anchor points (ax, ay) by line, spline, PCHIP or polynomial fit."""
    o, xs, _ = _sorted(x, y)
    ax = np.asarray(ax, float)
    ay = np.asarray(ay, float)
    k = np.argsort(ax)
    ax, ay = ax[k], ay[k]
    ax, u = np.unique(ax, return_index=True)
    ay = ay[u]
    if len(ax) < 2:
        return _unsort(o, np.full(len(xs), float(ay[0]) if len(ay) else 0.0))
    if interp == "linear":
        b = np.interp(xs, ax, ay)
    elif interp == "spline":
        if len(ax) < 4:
            b = np.interp(xs, ax, ay)
        else:
            b = interpolate.CubicSpline(ax, ay, bc_type="natural", extrapolate=True)(xs)
    elif interp == "pchip":
        b = interpolate.PchipInterpolator(ax, ay, extrapolate=True)(xs)
    elif interp == "fit":
        deg = min(int(order), len(ax) - 1)
        c = np.polynomial.polynomial.polyfit(ax, ay, deg)
        b = np.polynomial.polynomial.polyval(xs, c)
    else:
        raise ValueError(f"Unknown interpolation '{interp}'")
    return _unsort(o, b)


def fitted_baseline(x, y, ax, ay, model) -> np.ndarray:
    """Baseline from a fitting function (library name or equation) fitted to the anchors."""
    from rsstatchem.core.analysis import fitting as fit
    res = fit.fit_curve(ax, ay, model)
    return res.predict(np.asarray(x, float))


# ---------------------------------------------------------------------- penalized LS
def als(x, y, lam: float = 1e5, p: float = 0.01, n_iter: int = 10) -> np.ndarray:
    o, xs, ys = _sorted(x, y)
    return _unsort(o, sig.baseline_als(ys, lam, p, n_iter))


def arpls(x, y, lam: float = 1e5, ratio: float = 1e-6, max_iter: int = 100) -> np.ndarray:
    """Asymmetrically reweighted penalized least squares (Baek, Park, Ahn & Choo 2015)."""
    o, xs, ys = _sorted(x, y)
    n = len(ys)
    if n < 3:
        raise ValueError("At least 3 points are needed.")
    D = _second_diff(n)
    H = lam * (D.T @ D)
    w = np.ones(n)
    z = ys
    for _ in range(int(max_iter)):
        W = sparse.diags(w)
        z = spsolve(sparse.csc_matrix(W + H), w * ys)
        d = ys - z
        dn = d[d < 0]
        if len(dn) < 2:
            break
        m, s = dn.mean(), dn.std()
        if s == 0:
            break
        with np.errstate(over="ignore"):
            wt = 1.0 / (1.0 + np.exp(np.clip(2 * (d - (2 * s - m)) / s, -700, 700)))
        if np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio:
            w = wt
            break
        w = wt
    return _unsort(o, np.asarray(z))


def airpls(x, y, lam: float = 1e5, max_iter: int = 15, tol: float = 1e-3) -> np.ndarray:
    """Adaptive iteratively reweighted penalized least squares (Zhang, Chen & Liang 2010)."""
    o, xs, ys = _sorted(x, y)
    n = len(ys)
    if n < 3:
        raise ValueError("At least 3 points are needed.")
    D = _second_diff(n)
    H = lam * (D.T @ D)
    w = np.ones(n)
    z = ys
    for t in range(1, int(max_iter) + 1):
        z = spsolve(sparse.csc_matrix(sparse.diags(w) + H), w * ys)
        d = ys - z
        neg = d[d < 0]
        dssn = abs(neg.sum())
        if dssn < tol * np.abs(ys).sum() or len(neg) == 0:
            break
        w = np.where(d >= 0, 0.0, np.exp(np.clip(t * np.abs(d) / dssn, None, 700)))
        w[0] = w[-1] = math.exp(min(t * np.abs(neg).max() / dssn, 700))
    return _unsort(o, np.asarray(z))


# ---------------------------------------------------------------------- SNIP / morphology
def snip(x, y, max_half_window: int = 40, lls: bool = True, decreasing: bool = True,
         smooth_half_window: int = 0) -> np.ndarray:
    """Statistics-sensitive non-linear iterative peak clipping."""
    o, xs, ys = _sorted(x, y)
    v = ys.astype(float).copy()
    if smooth_half_window > 0:
        v = ndimage.uniform_filter1d(v, 2 * smooth_half_window + 1, mode="nearest")
    off = 0.0
    if lls:
        off = min(0.0, float(v.min()))
        v = np.log(np.log(np.sqrt(v - off + 1) + 1) + 1)
    n = len(v)
    m = int(max(1, min(max_half_window, (n - 1) // 2)))
    orders = range(m, 0, -1) if decreasing else range(1, m + 1)
    for p in orders:
        a = v.copy()
        mean = 0.5 * (a[:-2 * p] + a[2 * p:])
        a[p:n - p] = np.minimum(a[p:n - p], mean)
        v = a
    if lls:
        v = (np.exp(np.exp(v) - 1) - 1) ** 2 - 1 + off
    return _unsort(o, v)


def rolling_ball(x, y, radius: int = 50, smooth: int = 0) -> np.ndarray:
    """Morphological opening (a flat 'ball' of half-width ``radius`` points)."""
    o, xs, ys = _sorted(x, y)
    size = 2 * int(max(1, radius)) + 1
    b = ndimage.grey_opening(ys, size=size, mode="nearest")
    if smooth > 0:
        b = np.minimum(ndimage.uniform_filter1d(b, 2 * int(smooth) + 1, mode="nearest"), ys)
    return _unsort(o, b)


# ---------------------------------------------------------------------- XPS
def _cumtrapz(xs, v):
    out = np.zeros(len(xs))
    out[1:] = np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(xs))
    return out


def shirley(x, y, n_end: int = 1, tol: float = 1e-8, max_iter: int = 100) -> np.ndarray:
    """Iterative Shirley background between the two ends: B(x) = I_a + (I_b − I_a)·Q(x)/Q,
    Q(x) = ∫ₐˣ (y − B) dx (the step grows with the cumulative peak area along x)."""
    o, xs, ys = _sorted(x, y)
    k = max(1, int(n_end))
    ia, ib = ys[:k].mean(), ys[-k:].mean()
    B = np.full(len(ys), ia)
    for _ in range(int(max_iter)):
        q = _cumtrapz(xs, ys - B)
        total = q[-1]
        if total == 0:
            break
        new = ia + (ib - ia) * q / total
        if np.max(np.abs(new - B)) <= tol * (abs(ib - ia) + 1e-300):
            B = new
            break
        B = new
    return _unsort(o, B)


def tougaard(x, y, B: float = 2866.0, C: float = 1643.0, fit_end: bool = True) -> np.ndarray:
    """Tougaard universal cross-section background, x = binding energy (eV, ascending =
    increasing energy loss): bg(x) = y_a + B·∫ₐˣ K(x − x′)(y(x′) − y_a) dx′ with
    K(T) = T/(C + T²)². With ``fit_end`` the scale B is chosen so that the background meets
    the data at the high-x end (as Origin's fitted B)."""
    o, xs, ys = _sorted(x, y)
    ya = ys[0]
    sig_ = ys - ya
    n = len(xs)
    integ = np.zeros(n)
    for i in range(1, n):
        T = xs[i] - xs[:i + 1]
        K = T / (C + T * T) ** 2
        integ[i] = np.trapezoid(K * sig_[:i + 1], xs[:i + 1])
    scale = B
    if fit_end and integ[-1] != 0:
        scale = (ys[-1] - ya) / integ[-1]
    return _unsort(o, ya + scale * integ)


# ---------------------------------------------------------------------- dispatcher
def create_baseline(x, y, mode: str, **opts) -> np.ndarray:
    """Baseline by Peak Analyzer mode (see :data:`BASELINE_MODES`)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if mode == "none":
        return np.zeros(len(y))
    if mode == "constant":
        return constant(x, y, opts.get("constant_mode", "min"), opts.get("value"))
    if mode == "line":
        return straight_line(x, y, opts.get("n_end", 1))
    if mode == "endpoints":
        return endpoints_weighted(x, y, opts.get("fraction", 0.05))
    if mode == "anchors":
        ax, ay = opts.get("anchors_x"), opts.get("anchors_y")
        if ax is None:
            ax, ay = find_anchors(x, y, opts.get("n_anchors", 8),
                                  opts.get("anchor_method", "minima"), opts.get("smooth", 0))
        if opts.get("interp") == "function" and opts.get("function"):
            return fitted_baseline(x, y, ax, ay, opts["function"])
        return anchors_baseline(x, y, ax, ay, opts.get("interp", "linear"),
                                opts.get("order", 3))
    if mode == "als":
        return als(x, y, opts.get("lam", 1e5), opts.get("p", 0.01), opts.get("n_iter", 10))
    if mode == "arpls":
        return arpls(x, y, opts.get("lam", 1e5), opts.get("ratio", 1e-6))
    if mode == "airpls":
        return airpls(x, y, opts.get("lam", 1e5), opts.get("max_iter", 15))
    if mode == "snip":
        return snip(x, y, opts.get("half_window", 40), opts.get("lls", True))
    if mode == "shirley":
        return shirley(x, y, opts.get("n_end", 1))
    if mode == "tougaard":
        return tougaard(x, y, opts.get("B", 2866.0), opts.get("C", 1643.0))
    if mode == "rolling_ball":
        return rolling_ball(x, y, opts.get("radius", 50), opts.get("smooth", 0))
    if mode == "poly":
        return sig.baseline_poly(x, y, opts.get("order", 2))
    if mode == "dataset":
        b = np.asarray(opts.get("dataset"), float)
        if b.shape != y.shape:
            raise ValueError("The baseline dataset must have as many rows as the data.")
        return b
    raise ValueError(f"Unknown baseline mode '{mode}'")

"""Numerical core of the SPSS Categories procedures (optimal scaling by alternating least
squares): CATREG, CATPCA, MULTIPLE CORRESPONDENCE (homogeneity analysis), OVERALS and PREFSCAL.

No Qt and no Dataset here: every routine works on integer category codes, case weights and
plain numpy arrays, so it can be tested against closed forms and published values. The output
layer (tables, plots, dialogs) is :mod:`rsstatchem.core.stats.optscale`.

Algorithms follow the IBM SPSS Statistics Algorithms chapters "CATREG Algorithms", "CATPCA
Algorithms", "MULTIPLE CORRESPONDENCE Algorithms" and "OVERALS Algorithms":

* category quantifications are updated by the unconstrained least-squares update
  ``D_j^-1 G_j' W (target)`` followed by the restriction of the optimal scaling level --
  nominal (none), ordinal (weighted monotone regression, up-and-down blocks), numeric
  (weighted linear regression on the category values), spline nominal / spline ordinal
  (weighted regression on an I-spline basis, nonnegative coefficients for spline ordinal;
  Ramsay 1988) -- and the normalization ``y' D_j y = n_w`` with weighted mean zero;
* CATREG: backfitting of predictor quantifications and coefficients with the ridge
  ``b/(1+lambda2)``, lasso ``sign(b)(|b| - lambda1/2)_+`` and elastic-net updates of van der
  Kooij (2007), convergence on the apparent prediction error (APE);
* CATPCA / MCA: object scores from the Procrustes orthonormalization of
  ``Z = M*^-1 sum_j v_j G_j Y_j`` (rank-one restriction for single variables, centroids for
  multiple nominal variables), passive missing values via the per-object count ``M*``, total
  fit TFIT = sum of eigenvalues, principal-axes rotation and IBM's reflection rule;
* OVERALS: the K-set homogeneity loss with listwise deletion per set, eigenvalue = 1 - mean
  loss per dimension, canonical correlation (K*eigenvalue - 1)/(K - 1);
* PREFSCAL: penalized stress of Busing, Groenen & Heiser (2005)
  ``sigma_p = sigma_n^(2*lambda) * (1 + omega / nu^2(gamma))`` minimized by alternating a
  SMACOF majorization step for the joint configuration with a transformation step.

Initial configurations are deterministic (numeric solution or the closed-form eigen solution)
rather than IBM's random start; the converged solutions are the same up to reflection.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize
from scipy.interpolate import BSpline

LEVELS = {"spord": "Spline Ordinal", "spnom": "Spline Nominal", "mnom": "Multiple Nominal",
          "ordi": "Ordinal", "nomi": "Nominal", "nume": "Numeric"}
SINGLE_LEVELS = ("spord", "spnom", "ordi", "nomi", "nume")


# ===================================================================== discretization
def lloyd_max_normal(k: int) -> np.ndarray:
    """Interior thresholds of the minimum-distortion k-level quantizer of N(0, 1) (Max 1960),
    found by Lloyd's fixed-point iteration."""
    from scipy.stats import norm

    if k < 2:
        return np.array([])
    t = norm.ppf(np.arange(1, k) / k)
    for _ in range(500):
        edges = np.concatenate([[-np.inf], t, [np.inf]])
        a, b = edges[:-1], edges[1:]
        mass = norm.cdf(b) - norm.cdf(a)
        cent = (norm.pdf(a) - norm.pdf(b)) / np.maximum(mass, 1e-300)
        new = (cent[:-1] + cent[1:]) / 2
        if np.max(np.abs(new - t)) < 1e-12:
            t = new
            break
        t = new
    return t


def discretize(x: np.ndarray, method: str = "auto", ncat: int = 7,
               interval: float = 1.0, strings: bool = False) -> np.ndarray:
    """Category indicators (positive integers, NaN = missing) as SPSS CATREG/CATPCA
    discretization does: ``rank`` (ascending value), ``multiply`` (standardize, x10, round,
    shift so the lowest is 1), ``normal`` (Max 1960 intervals on the standardized variable),
    ``uniform`` (groups of about n/ncat cases), ``equal`` (intervals of the given size), or
    ``none``. ``auto`` uses the values as they are for positive integers, ranking for strings
    and other integers, and grouping into ``ncat`` normal categories for fractional values."""
    x = np.asarray(x, object if strings else float)
    out = np.full(len(x), np.nan)
    if strings:
        ok = np.array([v is not None and not (isinstance(v, float) and math.isnan(v))
                       and str(v) != "" for v in x])
        vals = sorted({str(v) for v in x[ok]})
        pos = {v: i + 1 for i, v in enumerate(vals)}
        out[ok] = [pos[str(v)] for v in x[ok]]
        return out
    ok = ~np.isnan(x)
    v = x[ok]
    if len(v) == 0:
        return out
    if method == "auto":
        integer = np.all(np.abs(v - np.round(v)) < 1e-9)
        if integer and v.min() >= 1:
            method = "none"
        elif integer:
            method = "rank"
        else:
            method = "normal" if len(np.unique(v)) > ncat else "rank"
    if method == "none":
        out[ok] = np.round(v)
    elif method == "rank":
        u = np.unique(v)
        out[ok] = np.searchsorted(u, v) + 1
    elif method == "multiply":
        sd = v.std(ddof=1) if len(v) > 1 else 1.0
        z = np.round((v - v.mean()) / (sd if sd > 0 else 1.0) * 10)
        out[ok] = z - z.min() + 1
    elif method == "normal":
        sd = v.std(ddof=1) if len(v) > 1 else 1.0
        z = (v - v.mean()) / (sd if sd > 0 else 1.0)
        out[ok] = np.searchsorted(lloyd_max_normal(ncat), z) + 1
    elif method == "uniform":
        order = np.argsort(v, kind="mergesort")
        u, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        target = len(v) / ncat
        cat, acc, groups = 1, 0, np.zeros(len(u))
        for i, c in enumerate(cnt):
            if acc > 0 and abs(acc + c - target) > abs(acc - target) and cat < ncat:
                cat, acc = cat + 1, 0
            groups[i] = cat
            acc += c
        out[ok] = groups[inv]
        del order
    elif method == "equal":
        out[ok] = np.floor((v - v.min()) / max(interval, 1e-12)) + 1
    else:
        raise ValueError(f"unknown discretization {method!r}")
    return out


# ===================================================================== variables
@dataclass
class CatVar:
    """One analysis variable: case codes 0..k-1 (-1 = passive missing), category values used
    for numeric/spline spacing and the optimal scaling level."""
    name: str
    codes: np.ndarray
    values: np.ndarray
    labels: list
    level: str = "spord"
    degree: int = 2
    knots: int = 2
    weight: float = 1.0
    free: np.ndarray | None = None      # categories excluded from the level restriction
    basis: np.ndarray | None = None     # I-spline basis (k x (degree+knots))

    @property
    def k(self) -> int:
        return len(self.values)

    @property
    def multiple(self) -> bool:
        return self.level == "mnom"


def make_var(name: str, indicators: np.ndarray, level: str = "spord", degree: int = 2,
             knots: int = 2, weight: float = 1.0, labels: dict | None = None,
             extra_missing: bool = False) -> CatVar:
    """Build a CatVar from positive category indicators (NaN = missing). With
    ``extra_missing`` missing values form an extra category that is free of the level
    restriction (SPSS "impute extra category")."""
    ind = np.asarray(indicators, float)
    ok = ~np.isnan(ind)
    vals = np.unique(ind[ok])
    codes = np.full(len(ind), -1, int)
    codes[ok] = np.searchsorted(vals, ind[ok])
    labs = [(labels or {}).get(v, _num(v)) for v in vals]
    free = np.zeros(len(vals), bool)
    if extra_missing and (~ok).any():
        codes[~ok] = len(vals)
        vals = np.append(vals, vals.max() + 1 if len(vals) else 1.0)
        labs.append("Missing")
        free = np.append(free, True)
    v = CatVar(name, codes, vals.astype(float), labs, level, degree, knots, weight, free)
    return v


def _num(v) -> str:
    v = float(v)
    return str(int(v)) if v.is_integer() else f"{v:g}"


def ispline_basis(values: np.ndarray, weights: np.ndarray, degree: int = 2,
                  nknots: int = 2) -> np.ndarray:
    """Monotone I-spline basis (Ramsay 1988) of the given degree with ``nknots`` interior
    knots placed at weighted quantiles of the case distribution over the category values.
    Returns a k x (degree + nknots) matrix of nondecreasing columns in [0, 1]."""
    values = np.asarray(values, float)
    lo, hi = values.min(), values.max()
    if hi <= lo:
        return np.zeros((len(values), 0))
    interior = []
    if nknots > 0:
        order = np.argsort(values)
        cw = np.cumsum(weights[order])
        cw = cw / cw[-1]
        for q in np.arange(1, nknots + 1) / (nknots + 1):
            interior.append(values[order][np.searchsorted(cw, q)])
        interior = sorted({t for t in interior if lo < t < hi})
    t = np.concatenate([[lo] * (degree + 1), interior, [hi] * (degree + 1)])
    nb = len(t) - degree - 1
    B = np.zeros((len(values), nb))
    x = np.clip(values, lo, hi)
    for j in range(nb):
        c = np.zeros(nb)
        c[j] = 1.0
        B[:, j] = BSpline(t, c, degree, extrapolate=False)(x)
    B[x >= hi, :] = 0.0
    B[x >= hi, -1] = 1.0
    B = np.nan_to_num(B)
    # I_j = sum_{m >= j} B_m (j = 1..nb-1); the j = 0 column is the constant 1
    S = np.cumsum(B[:, ::-1], axis=1)[:, ::-1][:, 1:]
    return S


def prepare(var: CatVar, w: np.ndarray) -> None:
    if var.level in ("spord", "spnom") and var.basis is None:
        d = np.bincount(var.codes[var.codes >= 0], w[var.codes >= 0], minlength=var.k)
        fixed = ~var.free if var.free is not None else np.ones(var.k, bool)
        S = np.zeros((var.k, 0))
        if fixed.sum() >= 2:
            S = np.zeros((var.k, var.degree + var.knots))
            Sb = ispline_basis(var.values[fixed], d[fixed], var.degree, var.knots)
            S[fixed, :Sb.shape[1]] = Sb
            S = S[:, :Sb.shape[1]]
        var.basis = S


# ===================================================================== restrictions
def pava(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted monotone (nondecreasing) regression, pool-adjacent-violators."""
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi)
        wts.append(wi)
        cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            tw = wts[-2] + wts[-1]
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / tw if tw > 0 else \
                (vals[-2] + vals[-1]) / 2
            c = cnt[-2] + cnt[-1]
            vals[-2:], wts[-2:], cnt[-2:] = [v], [tw], [c]
    return np.repeat(vals, cnt)


def restrict(var: CatVar, yt: np.ndarray, d: np.ndarray, level: str | None = None,
             sign: float = 1.0) -> np.ndarray:
    """Apply the optimal scaling level to the unconstrained update ``yt`` (category
    weights ``d``); ``sign`` = -1 fits a decreasing ordinal transformation (used with a fixed
    negative CATREG coefficient). Free categories keep their unconstrained value."""
    level = level or var.level
    yt = np.asarray(yt, float)
    if level in ("nomi", "mnom"):
        return yt.copy()
    fixed = ~var.free if var.free is not None else np.ones(var.k, bool)
    out = yt.copy()
    yf, df, vf = sign * yt[fixed], d[fixed], var.values[fixed]
    if df.sum() <= 0:
        return out
    if level == "nume":
        xm = np.sum(df * vf) / df.sum()
        ym = np.sum(df * yf) / df.sum()
        sxx = np.sum(df * (vf - xm) ** 2)
        b = np.sum(df * (vf - xm) * (yf - ym)) / sxx if sxx > 0 else 0.0
        if sign < 0:
            b = abs(b)
        r = ym + b * (vf - xm)
    elif level == "ordi":
        order = np.argsort(vf, kind="mergesort")
        r = np.empty_like(yf)
        r[order] = pava(yf[order], df[order])
    elif level in ("spord", "spnom"):
        S = var.basis[fixed] if var.basis is not None else np.zeros((len(vf), 0))
        if S.shape[1] == 0:
            return restrict(var, yt, d, "ordi" if level == "spord" else "nomi", sign)
        sw = np.sqrt(df)
        Sm = S - (df @ S) / df.sum()
        ym = np.sum(df * yf) / df.sum()
        A = Sm * sw[:, None]
        rhs = (yf - ym) * sw
        if level == "spord":
            a, _ = optimize.nnls(A, rhs)
        else:
            a = np.linalg.lstsq(A, rhs, rcond=None)[0]
        r = ym + Sm @ a
    else:
        raise ValueError(level)
    out[fixed] = sign * r if level != "nume" or sign > 0 else sign * r
    return out


def normalize(y: np.ndarray, d: np.ndarray, nw: float) -> np.ndarray:
    y = np.asarray(y, float)
    tot = d.sum()
    if tot <= 0:
        return y * 0
    y = y - np.sum(d * y) / tot
    ss = np.sum(d * y * y)
    return y * math.sqrt(nw / ss) if ss > 1e-14 else y * 0


def spline_df(var: CatVar, y: np.ndarray, d: np.ndarray) -> int:
    """IBM degrees of freedom of a transformation: 1 for numeric; distinct quantified values
    minus 1 for nominal/ordinal; degree + knots minus zero coefficients for splines (here
    estimated from the fitted spline as the rank of the active basis)."""
    if var.level == "nume":
        return 1
    if var.level in ("nomi", "ordi", "mnom"):
        occ = d > 0
        return max(len(np.unique(np.round(y[occ], 8))) - 1, 1)
    return max(min(var.degree + var.knots, int((d > 0).sum()) - 1), 1)


def _gw(codes: np.ndarray, x: np.ndarray, w: np.ndarray, k: int) -> np.ndarray:
    """G' W x for one variable (missing codes -1 contribute nothing)."""
    ok = codes >= 0
    if x.ndim == 1:
        return np.bincount(codes[ok], weights=(w * x)[ok], minlength=k)
    return np.column_stack([np.bincount(codes[ok], weights=(w * x[:, s])[ok], minlength=k)
                            for s in range(x.shape[1])])


def _expand(codes: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.zeros((len(codes),) + y.shape[1:])
    ok = codes >= 0
    out[ok] = y[codes[ok]]
    return out


# ===================================================================== CATREG
@dataclass
class CatregFit:
    yr: np.ndarray
    yp: list
    b: np.ndarray
    zr: np.ndarray
    Z: np.ndarray
    ape: float
    r2: float
    history: list = field(default_factory=list)
    iterations: int = 0
    converged: bool = True
    signs: np.ndarray | None = None


def _penalize(b: float, lam1: float, lam2: float) -> float:
    if lam1 > 0:
        b = math.copysign(max(abs(b) - lam1 / 2.0, 0.0), b)
    return b / (1.0 + lam2)


def catreg_fit(resp: CatVar, preds: list[CatVar], w: np.ndarray, lam1: float = 0.0,
               lam2: float = 0.0, maxiter: int = 100, eps: float = 1e-5,
               init: str = "numerical", start: CatregFit | None = None,
               signs: np.ndarray | None = None) -> CatregFit:
    """CATREG alternating least squares (IBM CATREG Algorithms). All codes must be >= 0
    (listwise deletion / imputation already done)."""
    w = np.asarray(w, float)
    nw = w.sum()
    allv = [resp] + preds
    for v in allv:
        prepare(v, w)
    D = [np.bincount(v.codes, w, minlength=v.k) for v in allv]
    p = len(preds)

    def run(levels: list[str], fit0: CatregFit | None, fixed_signs, max_it):
        if fit0 is None:
            ys = [normalize(v.values.astype(float), D[i], nw) for i, v in enumerate(allv)]
            zr = ys[0][resp.codes]
            Z = np.column_stack([ys[i + 1][preds[i].codes] for i in range(p)]) if p else \
                np.zeros((len(w), 0))
            b = np.array([np.sum(w * zr * Z[:, j]) / nw for j in range(p)])
            yr, yp = ys[0], ys[1:]
        else:
            yr, yp, b = fit0.yr.copy(), [y.copy() for y in fit0.yp], fit0.b.copy()
            zr, Z = fit0.zr.copy(), fit0.Z.copy()
        if fixed_signs is not None:
            sg = np.asarray(fixed_signs, float)
        else:
            sg = np.where(b < 0, -1.0, 1.0)
        v_ = Z @ b
        hist = []
        ape_old = np.sum(w * (zr - v_) ** 2) / nw
        conv = False
        it = 0
        for it in range(1, max_it + 1):
            # response
            yt = _gw(resp.codes, v_, w, resp.k) / np.where(D[0] > 0, D[0], 1)
            yr = normalize(restrict(resp, yt, D[0], levels[0]), D[0], nw)
            zr = yr[resp.codes]
            for j in range(p):
                vj = v_ - b[j] * Z[:, j]
                yt = _gw(preds[j].codes, zr - vj, w, preds[j].k) / \
                    np.where(D[j + 1] > 0, D[j + 1], 1)
                lev = levels[j + 1]
                s = sg[j] if lev in ("ordi", "spord", "nume") else 1.0
                if lev == "nume":
                    yj = normalize(restrict(preds[j], yt, D[j + 1], "nume"), D[j + 1], nw)
                    # numeric: orientation of the category values is kept
                    if np.sum(D[j + 1] * yj * preds[j].values) < 0:
                        yj = -yj
                else:
                    yj = normalize(restrict(preds[j], yt, D[j + 1], lev, s), D[j + 1], nw)
                    if lev in ("ordi", "spord") and s < 0:
                        yj = -yj   # quantification increasing, coefficient carries the sign
                bstar = np.sum(D[j + 1] * yt * yj) / nw
                b[j] = _penalize(bstar, lam1, lam2)
                yp[j] = yj
                Z[:, j] = yj[preds[j].codes]
                v_ = vj + b[j] * Z[:, j]
            ape = np.sum(w * (zr - v_) ** 2) / nw
            r2 = _r2(zr, v_, w)
            hist.append((it, r2, ape, ape_old - ape))
            if abs(ape_old - ape) < eps:
                conv = True
                break
            ape_old = ape
        return CatregFit(yr, yp, b, zr, Z, ape, _r2(zr, v_, w), hist, it, conv, sg)

    levels = [v.level for v in allv]
    if start is not None:
        return run(levels, start, signs if signs is not None else start.signs, maxiter)
    if init == "numerical":
        f0 = run(["nume"] * len(allv), None, None, maxiter)
        return run(levels, f0, signs, maxiter)
    return run(levels, None, signs, maxiter)


def _r2(zr: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
    a = np.sum(w * zr * v)
    bb = np.sum(w * v * v) * np.sum(w * zr * zr)
    return float(a * a / bb) if bb > 0 else 0.0


def catreg_predict(fit: CatregFit, preds: list[CatVar], codes: list[np.ndarray],
                   resp: CatVar | None = None, rcodes: np.ndarray | None = None
                   ) -> tuple[np.ndarray, np.ndarray | None]:
    """Predictions for (possibly new) cases with the fitted quantifications. Categories
    outside the fitted set must be mapped by the caller (-1 = unknown -> NaN)."""
    n = len(codes[0]) if codes else 0
    pred = np.zeros(n)
    bad = np.zeros(n, bool)
    for j, (v, c) in enumerate(zip(preds, codes)):
        ok = c >= 0
        bad |= ~ok
        pred[ok] += fit.b[j] * fit.yp[j][c[ok]]
    pred[bad] = np.nan
    obs = None
    if resp is not None and rcodes is not None:
        obs = np.full(n, np.nan)
        ok = rcodes >= 0
        obs[ok] = fit.yr[rcodes[ok]]
    return pred, obs


# ===================================================================== CATPCA / MCA
@dataclass
class PCAFit:
    X: np.ndarray                 # object scores (variable principal, X' M* W X = m n_w I)
    Y: list                       # per variable: k x p (centroids or vector coordinates)
    centroids: list               # per variable: k x p centroid coordinates
    quant: list                   # per variable: k quantifications (single) or None
    loadings: list                # per variable: p loadings (single) or None
    eig: np.ndarray               # eigenvalue per dimension
    tfit: float
    iterations: int
    converged: bool
    history: list = field(default_factory=list)
    mstar: np.ndarray | None = None


def _procrustes(Z: np.ndarray, metric: np.ndarray, c: float) -> tuple[np.ndarray, np.ndarray,
                                                                      np.ndarray]:
    sw = np.sqrt(metric)
    A = Z * sw[:, None]
    K, lam, Lt = np.linalg.svd(A, full_matrices=False)
    X = math.sqrt(c) * (K @ Lt) / np.where(sw > 0, sw, 1)[:, None]
    return X, lam, Lt.T


def _center(Z: np.ndarray, metric: np.ndarray) -> np.ndarray:
    return Z - (metric @ Z) / metric.sum()


def gifi_pca(vars_: list[CatVar], w: np.ndarray, ndim: int = 2, maxiter: int = 100,
             eps: float = 1e-5, X0: np.ndarray | None = None) -> PCAFit:
    """CATPCA (MCA when every variable is multiple nominal). Codes -1 are passive
    missing values (the object is ignored for that variable)."""
    w = np.asarray(w, float)
    n = len(w)
    nw = w.sum()
    for v in vars_:
        prepare(v, w)
    act = [v.codes >= 0 for v in vars_]
    mstar = np.sum([a for a in act], axis=0).astype(float)
    if np.any(mstar == 0):
        raise ValueError("Every object needs at least one active (non-missing) value.")
    metric = mstar * w
    m = len(vars_)
    D = [np.bincount(v.codes[a], w[a], minlength=v.k) for v, a in zip(vars_, act)]
    vw = np.array([v.weight for v in vars_], float)

    def cycle(levels, X, max_it):
        hist = []
        tfit_old = -np.inf
        Y = [None] * m
        quant = [None] * m
        load = [None] * m
        it, conv = 0, False
        for it in range(1, max_it + 1):
            tfit = 0.0
            for j, v in enumerate(vars_):
                Dj = np.where(D[j] > 0, D[j], 1)
                Yt = _gw(v.codes, X, w, v.k) / Dj[:, None]
                if levels[j] == "mnom":
                    Y[j] = Yt
                    tfit += vw[j] * np.trace(Yt.T @ (D[j][:, None] * Yt)) / nw
                    continue
                a = load[j] if load[j] is not None else \
                    (Yt.T @ (D[j] * normalize(v.values, D[j], nw))) / nw
                if np.linalg.norm(a) < 1e-12:
                    a = np.ones(X.shape[1]) / math.sqrt(X.shape[1])
                q = Yt @ a
                q = normalize(restrict(v, q, D[j], levels[j]), D[j], nw)
                if levels[j] == "nume" and np.sum(D[j] * q * v.values) < 0:
                    q = -q
                a = Yt.T @ (D[j] * q) / nw
                quant[j], load[j] = q, a
                Y[j] = np.outer(q, a)
                tfit += vw[j] * float(a @ a)
            Zs = np.zeros_like(X)
            for j, v in enumerate(vars_):
                Zs += vw[j] * _expand(v.codes, Y[j])
            Z = _center(Zs / mstar[:, None], metric)
            X, lam, L = _procrustes(Z, metric, m * nw)
            hist.append((it, tfit, tfit - tfit_old if np.isfinite(tfit_old) else np.nan))
            if abs(tfit - tfit_old) < eps:
                conv = True
                break
            tfit_old = tfit
        return X, Y, quant, load, tfit, it, conv, hist, L

    levels = [v.level for v in vars_]
    if X0 is None:
        X0 = _initial_scores(vars_, w, metric, ndim, m * nw)
    lv1 = [lv if lv in ("mnom", "nume") else "nume" for lv in levels]
    X, *_ = cycle(lv1, X0, maxiter)
    X, Y, quant, load, tfit, it, conv, hist, L = cycle(levels, X, maxiter)
    # principal axes: rotate with L from the (last) orthonormalization; at convergence the
    # single quantifications are invariant under this rotation
    Zs = sum(vw[j] * _expand(v.codes, Y[j]) for j, v in enumerate(vars_))
    Z = _center(Zs / mstar[:, None], metric)
    X, lam, L = _procrustes(Z, metric, m * nw)
    X = X @ L
    # reflection: dimension reflected when the negative squared loadings dominate
    cents, quant2, load2, Yf = [], [], [], []
    for j, v in enumerate(vars_):
        Dj = np.where(D[j] > 0, D[j], 1)
        Yt = _gw(v.codes, X, w, v.k) / Dj[:, None]
        cents.append(Yt)
    singles = [j for j, lv in enumerate(levels) if lv != "mnom"]
    flip = np.ones(X.shape[1])
    for s in range(X.shape[1]):
        if singles:
            ld = np.array([cents[j].T @ (D[j] * quant[j]) / nw for j in singles])[:, s]
            neg, pos = ld[ld < 0] ** 2, ld[ld > 0] ** 2
            if neg.size and (not pos.size or neg.mean() > pos.mean()):
                flip[s] = -1
        else:
            # all multiple nominal: no loadings; orient so the largest |centroid| is positive
            allc = np.concatenate([c[:, s] for c in cents])
            if allc[np.argmax(np.abs(allc))] < 0:
                flip[s] = -1
    X = X * flip
    cents = [c * flip for c in cents]
    eig = np.zeros(X.shape[1])
    for j, v in enumerate(vars_):
        if levels[j] == "mnom":
            Yf.append(cents[j])
            quant2.append(None)
            load2.append(None)
            eig += vw[j] * np.einsum("k,ks,ks->s", D[j], cents[j], cents[j]) / nw
        else:
            q = quant[j]
            a = cents[j].T @ (D[j] * q) / nw
            quant2.append(q)
            load2.append(a)
            Yf.append(np.outer(q, a))
            eig += vw[j] * a ** 2
    return PCAFit(X, Yf, cents, quant2, load2, eig, float(eig.sum()), it, conv, hist, mstar)


def _initial_scores(vars_, w, metric, ndim, c):
    """Deterministic start: object scores of the closed-form homogeneity solution with the
    single variables entered as numeric (centered, standardized) columns."""
    n = len(w)
    cols = []
    for v in vars_:
        ok = v.codes >= 0
        d = np.bincount(v.codes[ok], w[ok], minlength=v.k)
        if v.level == "mnom":
            for k in range(v.k):
                if d[k] > 0:
                    col = np.zeros(n)
                    col[ok & (v.codes == k)] = 1.0 / math.sqrt(d[k])
                    cols.append(col)
        else:
            q = normalize(v.values, d, w.sum())
            col = np.zeros(n)
            col[ok] = q[v.codes[ok]] / math.sqrt(w.sum())
            cols.append(col)
    H = np.column_stack(cols)
    H = H - (metric @ H) / metric.sum()
    sw = np.sqrt(w)
    U, s, _ = np.linalg.svd(H * sw[:, None], full_matrices=False)
    Z = U[:, :ndim] / sw[:, None]
    if Z.shape[1] < ndim:
        Z = np.column_stack([Z, np.random.default_rng(1).standard_normal((n, ndim - Z.shape[1]))])
    Z = _center(Z, metric)
    return _procrustes(Z, metric, c)[0]


def mca_closed_form(vars_: list[CatVar], w: np.ndarray, ndim: int = 2) -> np.ndarray:
    """Eigenvalues (SPSS MCA 'Total (Eigenvalue)') of homogeneity analysis without missing
    data: m times the principal inertias of correspondence analysis of the indicator matrix."""
    w = np.asarray(w, float)
    m = len(vars_)
    blocks = []
    for v in vars_:
        G = np.zeros((len(w), v.k))
        G[np.arange(len(w)), v.codes] = 1
        blocks.append(G[:, G.T @ w > 0])
    G = np.hstack(blocks)
    P = (G * w[:, None]) / (m * w.sum())
    r = P.sum(axis=1)
    c = P.sum(axis=0)
    S = (P - np.outer(r, c)) / np.sqrt(np.outer(r, c))
    sv = np.linalg.svd(S, compute_uv=False)
    return m * sv[:ndim] ** 2


# ===================================================================== OVERALS
@dataclass
class OveralsFit:
    X: np.ndarray
    Y: list          # per set: list of k x p
    quant: list      # per set: list of single quantifications or None
    weights: list    # per set: list of p-vectors (single) or None
    centroids: list  # per set: list of k x p category centroids
    loss_set: np.ndarray   # K x p
    eig: np.ndarray
    fit: float
    iterations: int
    converged: bool
    history: list = field(default_factory=list)
    multiple: list = field(default_factory=list)   # unconstrained (multiple) coordinates


def overals(sets: list[list[CatVar]], w: np.ndarray, ndim: int = 2, maxiter: int = 100,
            eps: float = 1e-5) -> OveralsFit:
    """Nonlinear canonical correlation analysis (IBM OVERALS Algorithms). Codes -1 mark
    objects that are missing in a set (listwise deletion per set)."""
    w = np.asarray(w, float)
    n = len(w)
    K = len(sets)
    act = []
    for vs in sets:
        a = np.ones(n, bool)
        for v in vs:
            a &= v.codes >= 0
        act.append(a)
    mstar = np.sum(act, axis=0).astype(float)
    if np.any(mstar == 0):
        raise ValueError("Every object must be active in at least one set.")
    metric = mstar * w
    nw = w.sum()
    D = [[np.bincount(v.codes[act[k]], w[act[k]], minlength=v.k) for v in vs]
         for k, vs in enumerate(sets)]
    for vs in sets:
        for v in vs:
            prepare(v, w)

    def codes_k(k, v):
        c = v.codes.copy()
        c[~act[k]] = -1
        return c

    def quantify(levels, X, Y, quant, wts):
        for k, vs in enumerate(sets):
            cods = [codes_k(k, v) for v in vs]
            V = sum(_expand(c, Y[k][j]) for j, c in enumerate(cods)) if vs else 0
            for j, v in enumerate(vs):
                Dj = np.where(D[k][j] > 0, D[k][j], 1)
                Vj = V - _expand(cods[j], Y[k][j])
                Yt = _gw(cods[j], X - Vj, w, v.k) / Dj[:, None]
                lev = levels[k][j]
                if lev == "mnom":
                    Yn = Yt
                else:
                    q = quant[k][j]
                    if q is None:
                        q = normalize(v.values, D[k][j], nw)
                    a = Yt.T @ (D[k][j] * q) / nw
                    if np.linalg.norm(a) < 1e-12:
                        a = np.full(X.shape[1], 1e-6)
                    q = normalize(restrict(v, Yt @ a, D[k][j], lev), D[k][j], nw)
                    if lev == "nume" and np.sum(D[k][j] * q * v.values) < 0:
                        q = -q
                    a = Yt.T @ (D[k][j] * q) / nw
                    quant[k][j], wts[k][j] = q, a
                    Yn = np.outer(q, a)
                V = Vj + _expand(cods[j], Yn)
                Y[k][j] = Yn

    def zsum(Y):
        Zs = np.zeros((n, ndim))
        for k, vs in enumerate(sets):
            for j, v in enumerate(vs):
                Zs += _expand(codes_k(k, v), Y[k][j])
        return _center(Zs / mstar[:, None], metric)

    def cycle(levels, X, max_it):
        Y = [[np.zeros((v.k, X.shape[1])) for v in vs] for vs in sets]
        quant = [[None] * len(vs) for vs in sets]
        wts = [[None] * len(vs) for vs in sets]
        loss_old = np.inf
        hist = []
        it, conv = 0, False
        for it in range(1, max_it + 1):
            quantify(levels, X, Y, quant, wts)
            loss = _overals_loss(sets, Y, X, w, act, codes_k, K, nw)
            X = _procrustes(zsum(Y), metric, K * nw)[0]
            hist.append((it, loss))
            if abs(loss_old - loss) < eps:
                conv = True
                break
            loss_old = loss
        return X, Y, quant, wts, it, conv, hist

    levels = [[v.level for v in vs] for vs in sets]
    X0 = _initial_scores([v for vs in sets for v in vs], w, metric, ndim, K * nw)
    lv1 = [[lv if lv == "mnom" else "nume" for lv in lvs] for lvs in levels]
    X, *_ = cycle(lv1, X0, maxiter)
    X, Y, quant, wts, it, conv, hist = cycle(levels, X, maxiter)
    # rotate to principal axes: make the cross products X' M* W Z diagonal
    C = (X * metric[:, None]).T @ zsum(Y)
    evals, evecs = np.linalg.eigh((C + C.T) / 2)
    L = evecs[:, ::-1]
    X = X @ L
    Y = [[y @ L for y in ys] for ys in Y]
    for _ in range(200):   # quantifications consistent with the rotated scores
        old = _overals_loss(sets, Y, X, w, act, codes_k, K, nw)
        quantify(levels, X, Y, quant, wts)
        if abs(old - _overals_loss(sets, Y, X, w, act, codes_k, K, nw)) < 1e-12:
            break
    loss_set = np.zeros((K, X.shape[1]))
    cents = []
    multi = []
    for k, vs in enumerate(sets):
        cods = [codes_k(k, v) for v in vs]
        V = sum(_expand(c, Y[k][j]) for j, c in enumerate(cods))
        multi.append([_gw(cods[j], X - (V - _expand(cods[j], Y[k][j])), w, v.k) /
                      np.where(D[k][j] > 0, D[k][j], 1)[:, None] for j, v in enumerate(vs)])
    for k, vs in enumerate(sets):
        fitted = sum(_expand(codes_k(k, v), Y[k][j]) for j, v in enumerate(vs))
        R = (X - fitted) * act[k][:, None]
        loss_set[k] = np.einsum("i,is,is->s", w, R, R) / nw
        cents.append([_gw(codes_k(k, v), X, w, v.k) /
                      np.where(D[k][j] > 0, D[k][j], 1)[:, None] for j, v in enumerate(vs)])
    flip = np.where(np.sum(X ** 3 * w[:, None], axis=0) < 0, -1.0, 1.0)
    X = X * flip
    Y = [[y * flip for y in ys] for ys in Y]
    wts = [[(a * flip if a is not None else None) for a in ws] for ws in wts]
    cents = [[c * flip for c in cs] for cs in cents]
    multi = [[c * flip for c in cs] for cs in multi]
    eig = 1.0 - loss_set.mean(axis=0)
    out = OveralsFit(X, Y, quant, wts, cents, loss_set, eig, float(eig.sum()), it, conv, hist)
    out.multiple = multi
    return out


def _overals_loss(sets, Y, X, w, act, codes_k, K, nw):
    tot = 0.0
    for k, vs in enumerate(sets):
        fitted = sum(_expand(codes_k(k, v), Y[k][j]) for j, v in enumerate(vs))
        R = (X - fitted) * act[k][:, None]
        tot += np.sum(w[:, None] * R * R)
    return tot / (K * nw)


# ===================================================================== PREFSCAL
@dataclass
class PrefscalFit:
    rows: np.ndarray       # n x p row (individual) coordinates
    cols: np.ndarray       # m x p column (stimulus) coordinates
    gamma: np.ndarray      # transformed proximities n x m
    dist: np.ndarray       # distances n x m
    penalized: float
    stress_part: float
    penalty_part: float
    iterations: int
    converged: bool
    history: list = field(default_factory=list)


def _dist(R, C):
    return np.sqrt(np.maximum(((R[:, None, :] - C[None, :, :]) ** 2).sum(-1), 0.0))


def _cv2(g: np.ndarray, wt: np.ndarray) -> float:
    s = wt.sum()
    mean = np.sum(wt * g) / s
    var = np.sum(wt * (g - mean) ** 2) / s
    return var / mean ** 2 if mean != 0 else 0.0


def prefscal_parts(gamma, d, W, conditional: bool, lam: float, omega: float):
    """Penalized stress (Busing, Groenen & Heiser 2005). Row-conditional: normalized stress
    averaged over rows, nu^2 = harmonic mean of the row variation coefficients."""
    if conditional:
        num = np.sum(W * (gamma - d) ** 2, axis=1)
        den = np.sum(W * gamma ** 2, axis=1)
        sn = float(np.mean(num / np.where(den > 0, den, 1)))
        cvs = np.array([_cv2(gamma[i], W[i]) for i in range(gamma.shape[0])])
        nu2 = len(cvs) / np.sum(1.0 / np.maximum(cvs, 1e-12))
    else:
        sn = float(np.sum(W * (gamma - d) ** 2) / np.sum(W * gamma ** 2))
        nu2 = _cv2(gamma, W)
    stress = sn ** lam
    pen = 1.0 + omega / max(nu2, 1e-12)
    return stress * pen, stress, pen, sn


def prefscal(delta: np.ndarray, ndim: int = 2, transform: str = "none",
             conditional: bool = True, lam: float = 0.5, omega: float = 1.0,
             similarities: bool = False, maxiter: int = 5000, eps: float = 1e-6,
             min_stress: float = 1e-4, W: np.ndarray | None = None,
             init: tuple | None = None) -> PrefscalFit:
    """Multidimensional unfolding of an n x m rectangular proximity matrix (rows =
    individuals, columns = stimuli). Transformations: ``none`` (ratio, gamma = b*delta),
    ``linear`` (gamma = a + b*delta, a >= 0 kept positive), ``ordinal`` (monotone,
    ties kept tied) and ``smooth``/``spline`` fall back to ordinal/linear. ``conditional``
    applies the transformation within each row (SPSS "Within each row separately")."""
    delta = np.asarray(delta, float)
    n, m = delta.shape
    W = np.ones_like(delta) if W is None else np.asarray(W, float)
    W = W * ~np.isnan(delta)
    dd = np.nan_to_num(delta)
    if similarities:
        dd = np.where(W > 0, np.nanmax(delta) + np.nanmin(delta) - dd, 0.0)
    if transform == "spline":
        transform = "linear"
    if transform == "smooth":
        transform = "ordinal"
    starts = [dd]
    if transform == "ordinal" and init is None:
        # an ordinal transformation only uses the order of delta: also start from ranks
        # (within rows when conditional) and keep the better local minimum
        from scipy.stats import rankdata

        if conditional:
            rk = np.vstack([np.where(W[i] > 0, rankdata(dd[i]), 0.0) for i in range(n)])
        else:
            rk = rankdata(dd.ravel()).reshape(dd.shape) * (W > 0)
        starts.append(rk)
    best = None
    for start in starts:
        fit = _prefscal_run(dd, start, W, n, ndim, transform, conditional, lam, omega,
                            maxiter, eps, min_stress, init)
        if best is None or fit.penalized < best.penalized:
            best = fit
    return best


def _prefscal_run(dd, start, W, n, ndim, transform, conditional, lam, omega, maxiter, eps,
                  min_stress, init):
    # initial configuration: classical scaling of the bipartite (row+column) proximities
    if init is None:
        R0, C0 = _unfold_init(start, W, ndim)
    else:
        R0, C0 = init
    # initial gamma: (ranked) delta scaled to unit mean square per row/overall
    gamma = _scale_gamma(start, W, conditional)
    R, C = R0.copy(), C0.copy()
    hist = []
    old = np.inf
    conv = False
    it = 0
    tstate: dict = {}
    if transform != "none":
        gamma = _project(gamma, dd, W, transform, conditional)
    for it in range(1, maxiter + 1):
        # --- configuration (SMACOF on the bipartite problem, row weights 1/||gamma_i||^2)
        if conditional:
            rw = 1.0 / np.maximum(np.sum(W * gamma ** 2, axis=1), 1e-12)
        else:
            rw = np.full(n, 1.0 / max(np.sum(W * gamma ** 2), 1e-12))
        Wt = W * rw[:, None]
        R, C = _smacof_step(R, C, gamma, Wt)
        d = _dist(R, C)
        # --- transformation
        gamma = _transform_step(dd, d, W, gamma, transform, conditional, lam, omega,
                                tstate)
        val, sp, pp, sn = prefscal_parts(gamma, d, W, conditional, lam, omega)
        hist.append((it, val, sp, pp))
        if old - val < eps * max(val, 1e-12) and it > 2:
            conv = True
            break
        if sn < min_stress ** 2:
            conv = True
            break
        old = val
    d = _dist(R, C)
    val, sp, pp, _ = prefscal_parts(gamma, d, W, conditional, lam, omega)
    return PrefscalFit(R, C, gamma, d, val, sp, pp, it, conv, hist)


def _scale_gamma(dd, W, conditional):
    g = dd.copy()
    if conditional:
        for i in range(g.shape[0]):
            s = np.sqrt(np.sum(W[i] * g[i] ** 2))
            if s > 0:
                g[i] = g[i] / s * math.sqrt(W[i].sum())
    else:
        s = np.sqrt(np.sum(W * g ** 2))
        if s > 0:
            g = g / s * math.sqrt(W.sum())
    return g


def _unfold_init(dd, W, ndim):
    n, m = dd.shape
    N = n + m
    full = np.zeros((N, N))
    full[:n, n:] = dd
    full[n:, :n] = dd.T
    # fill within-set dissimilarities by the triangle-type approximation |d_ik - d_jk| mean
    for blk, M in ((slice(0, n), dd), (slice(n, N), dd.T)):
        # triangle inequality bounds |d_ik - d_jk| <= d_ij <= d_ik + d_jk (midpoint)
        A = (np.abs(M[:, None, :] - M[None, :, :]).max(-1) +
             (M[:, None, :] + M[None, :, :]).min(-1)) / 2
        full[blk, blk] = A
    np.fill_diagonal(full, 0)
    J = np.eye(N) - 1.0 / N
    B = -0.5 * J @ (full ** 2) @ J
    ev, V = np.linalg.eigh(B)
    idx = np.argsort(ev)[::-1][:ndim]
    X = V[:, idx] * np.sqrt(np.maximum(ev[idx], 1e-8))
    return X[:n], X[n:]


def _smacof_step(R, C, gamma, Wt):
    n, m = gamma.shape
    X = np.vstack([R, C])
    N = n + m
    Wf = np.zeros((N, N))
    Wf[:n, n:] = Wt
    Wf[n:, :n] = Wt.T
    Df = np.zeros((N, N))
    d = _dist(R, C)
    Df[:n, n:] = d
    Df[n:, :n] = d.T
    Gf = np.zeros((N, N))
    Gf[:n, n:] = gamma
    Gf[n:, :n] = gamma.T
    V = -Wf.copy()
    np.fill_diagonal(V, Wf.sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        Bm = np.where(Df > 1e-12, -Wf * Gf / Df, 0.0)
    np.fill_diagonal(Bm, -Bm.sum(axis=1))
    Xn = np.linalg.pinv(V) @ Bm @ X
    return Xn[:n], Xn[n:]


def _project(g, dd, W, transform, conditional):
    """Weighted least-squares projection of ``g`` onto the transformation cone (per row when
    conditional): ratio = multiples of delta, linear = a + b*delta, ordinal = monotone in
    delta (primary approach: ties may be untied), always nonnegative."""
    def one(gr, xr, wr):
        ok = wr > 0
        out = np.zeros_like(gr)
        if ok.sum() == 0:
            return out
        x, y, wt = xr[ok], gr[ok], wr[ok]
        if transform == "none":
            b = np.sum(wt * x * y) / max(np.sum(wt * x * x), 1e-12)
            out[ok] = max(b, 0.0) * x
        elif transform == "linear":
            A = np.column_stack([np.ones_like(x), x]) * np.sqrt(wt)[:, None]
            coef = np.linalg.lstsq(A, y * np.sqrt(wt), rcond=None)[0]
            if coef[1] < 0:
                coef = np.array([np.sum(wt * y) / wt.sum(), 0.0])
            out[ok] = np.maximum(coef[0] + coef[1] * x, 0.0)
        else:
            order = np.lexsort((y, x))
            r = np.empty_like(y)
            r[order] = pava(y[order], wt[order])
            out[ok] = np.maximum(r, 0.0)
        return out

    if conditional:
        return np.vstack([one(g[i], dd[i], W[i]) for i in range(g.shape[0])])
    return one(g.ravel(), dd.ravel(), W.ravel()).reshape(g.shape)


def _penalized_grad(g, d, W, conditional, lam, omega):
    """Gradient of the penalized stress with respect to the transformed proximities."""
    if conditional:
        rows = range(g.shape[0])
        n = g.shape[0]
        N = np.sum(W * (g - d) ** 2, axis=1)
        Dn = np.maximum(np.sum(W * g ** 2, axis=1), 1e-12)
        sn = np.mean(N / Dn)
        dsn = (2 * W * (g - d) * Dn[:, None] - 2 * W * g * N[:, None]) / (Dn ** 2)[:, None] / n
        S0 = W.sum(axis=1)
        S1 = np.maximum(np.sum(W * g, axis=1), 1e-12)
        S2 = np.sum(W * g ** 2, axis=1)
        c = np.maximum(S2 * S0 / S1 ** 2 - 1, 1e-12)
        pen = 1 + omega / n * np.sum(1 / c)
        dc = 2 * S0[:, None] * W * (g * S1[:, None] - S2[:, None]) / (S1 ** 3)[:, None]
        dpen = -(omega / n) * dc / (c ** 2)[:, None]
        del rows
    else:
        N = np.sum(W * (g - d) ** 2)
        Dn = max(np.sum(W * g ** 2), 1e-12)
        sn = N / Dn
        dsn = (2 * W * (g - d) * Dn - 2 * W * g * N) / Dn ** 2
        S0, S1, S2 = W.sum(), max(np.sum(W * g), 1e-12), np.sum(W * g ** 2)
        c = max(S2 * S0 / S1 ** 2 - 1, 1e-12)
        pen = 1 + omega / c
        dpen = -omega * (2 * S0 * W * (g * S1 - S2) / S1 ** 3) / c ** 2
    stress = max(sn, 1e-300) ** lam
    dstress = lam * max(sn, 1e-300) ** (lam - 1) * dsn
    return pen * dstress + stress * dpen


def _transform_step(dd, d, W, gamma, transform, conditional, lam, omega, state=None):
    """Update the transformed proximities: the ratio transformation is the least-squares
    fit; the others take projected-gradient steps on the penalized stress (projection onto
    the transformation cone) with a backtracking step size."""
    if transform == "none":
        return _project(d, dd, W, "none", conditional)
    state = state if state is not None else {}
    best = gamma
    best_val = prefscal_parts(gamma, d, W, conditional, lam, omega)[0]
    for _ in range(5):
        grad = _penalized_grad(best, d, W, conditional, lam, omega)
        gn = np.sqrt(np.sum(grad ** 2))
        if gn < 1e-14:
            break
        step = state.get("step") or 0.1 * np.sqrt(np.sum(best ** 2)) / gn
        improved = False
        for _ in range(40):
            cand = _project(best - step * grad, dd, W, transform, conditional)
            val = prefscal_parts(cand, d, W, conditional, lam, omega)[0]
            if val < best_val - 1e-15:
                best, best_val = cand, val
                state["step"] = step * 2
                improved = True
                break
            step /= 2
        if not improved:
            state["step"] = None
            break
    return best


def kendall_tau_b(x, y):
    from scipy.stats import kendalltau

    return float(kendalltau(x, y).statistic)


def rpo(gamma_or_delta: np.ndarray, d: np.ndarray, W: np.ndarray) -> float:
    """Recovered preference orders: share of rows whose preference order (smallest
    proximity first) is recovered by the distances, counting pairs (Busing 2010)."""
    agree, tot = 0.0, 0.0
    for i in range(d.shape[0]):
        ok = W[i] > 0
        a, b = gamma_or_delta[i, ok], d[i, ok]
        da = np.sign(a[:, None] - a[None, :])
        db = np.sign(b[:, None] - b[None, :])
        mask = np.triu(np.ones_like(da, bool), 1) & (da != 0)
        agree += np.sum((da == db) & mask)
        tot += mask.sum()
    return agree / tot if tot else float("nan")


def unfolding_measures(fit: PrefscalFit, delta: np.ndarray, W: np.ndarray,
                       conditional: bool) -> dict:
    g, d = fit.gamma, fit.dist
    Wm = W
    out = {}
    if conditional:
        num = np.sum(Wm * (g - d) ** 2, axis=1)
        out["normalized"] = float(np.mean(num / np.maximum(np.sum(Wm * g ** 2, axis=1), 1e-12)))
    else:
        out["normalized"] = float(np.sum(Wm * (g - d) ** 2) / np.sum(Wm * g ** 2))
    out["stress1"] = math.sqrt(np.sum(Wm * (g - d) ** 2) / np.sum(Wm * d ** 2))
    dm = np.sum(Wm * d) / Wm.sum()
    out["stress2"] = math.sqrt(np.sum(Wm * (g - d) ** 2) / np.sum(Wm * (d - dm) ** 2))
    out["sstress1"] = math.sqrt(np.sum(Wm * (g ** 2 - d ** 2) ** 2) / np.sum(Wm * d ** 4))
    d2m = np.sum(Wm * d ** 2) / Wm.sum()
    out["sstress2"] = math.sqrt(np.sum(Wm * (g ** 2 - d ** 2) ** 2) /
                                np.sum(Wm * (d ** 2 - d2m) ** 2))
    out["daf"] = 1 - out["normalized"]
    ok = Wm > 0
    out["vaf"] = float(np.corrcoef(g[ok], d[ok])[0, 1] ** 2)
    out["rpo"] = rpo(np.nan_to_num(delta), d, Wm)
    from scipy.stats import spearmanr

    rho, tau = [], []
    for i in range(d.shape[0]):
        k = Wm[i] > 0
        if k.sum() > 2 and np.ptp(np.nan_to_num(delta)[i, k]) > 0 and np.ptp(d[i, k]) > 0:
            rho.append(spearmanr(np.nan_to_num(delta)[i, k], d[i, k]).statistic)
            tau.append(kendall_tau_b(np.nan_to_num(delta)[i, k], d[i, k]))
    out["spearman"] = float(np.nanmean(rho)) if rho else float("nan")
    out["kendall"] = float(np.nanmean(tau)) if tau else float("nan")
    out["cv_prox"] = math.sqrt(_cv2(np.nan_to_num(delta), Wm))
    out["cv_trans"] = math.sqrt(_cv2(g, Wm))
    out["cv_dist"] = math.sqrt(_cv2(d, Wm))
    # DeSarbo's intermixedness and Shepard's rough nondegeneracy index
    R, C = fit.rows, fit.cols
    dr = _dist(R, R)
    dc = _dist(C, C)
    dmean = d.mean()
    out["desarbo"] = float(((dr[np.triu_indices(len(R), 1)] / dmean - 1) ** 2).mean() +
                           ((dc[np.triu_indices(len(C), 1)] / dmean - 1) ** 2).mean()) \
        if len(R) > 1 and len(C) > 1 else float("nan")
    ranks = np.round(g[ok], 10)
    out["shepard"] = len(np.unique(ranks)) / ranks.size if ranks.size else float("nan")
    return out

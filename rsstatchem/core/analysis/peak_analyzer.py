"""Peak Analyzer (Origin's Analysis > Peaks and Baseline > Peak Analyzer), no Qt.

Goals (Origin's Goal page): **Create Baseline**, **Subtract Baseline**, **Find Peaks**,
**Integrate Peaks** and **Fit Peaks**. The steps are exposed as functions and combined by
:class:`PeakAnalysis`, which the wizard dialog drives page by page.

* Baselines: :mod:`baselines` (anchor points, ALS, arPLS, airPLS, SNIP, Shirley, …).
* Find peaks (Origin's methods): local maximum, window search, first derivative, second
  derivative (also finds hidden peaks / shoulders, which are not local maxima) and
  "residual after first derivative" (peaks found in the residual of a quick fit of the
  first-derivative peaks = hidden peaks); filtering by height (% of the range) and number.
* Integrate: per-peak limits (automatic: where the subtracted signal returns to zero or at
  the valley between peaks; or user limits), area, % area, centroid, FWHM, height.
* Fit: sums of Gaussian, Lorentzian, Voigt, pseudo-Voigt, Pearson VII, Asym2Sig,
  BiGaussian, EMG (GaussMod) or logistic peaks, optional baseline co-fit (constant /
  linear / quadratic), shared or fixed widths, fixed centres; per-peak table with centre,
  height, area (with standard errors), FWHM and % area; component and cumulative curves.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import signal as ss

from rsstatchem.core.analysis import baselines as bl
from rsstatchem.core.analysis import fitting as fit
from rsstatchem.core.analysis import functions as fl
from rsstatchem.core.analysis.fitting import FitResult
from rsstatchem.core.analysis.functions import FitFunction

GOALS = {"baseline": "Create Baseline", "subtract": "Subtract Baseline",
         "find": "Find Peaks", "integrate": "Integrate Peaks", "fit": "Fit Peaks"}
FIND_METHODS = {"local_max": "Local Maximum", "window": "Window Search",
                "first_derivative": "1st Derivative", "second_derivative":
                "2nd Derivative (hidden peaks)", "residual":
                "Residual after 1st Derivative (hidden peaks)"}
FIT_BASELINES = {"none": "None (fit the subtracted data)", "constant": "Constant",
                 "linear": "Linear", "quadratic": "Quadratic"}
_C = math.sqrt(math.pi / (4 * math.log(2)))   # Gaussian area = height·FWHM·_C
_S2 = 2 * math.sqrt(2 * math.log(2))           # FWHM / σ


def _pv7_height_factor(mu, w):
    return 2 * math.gamma(mu) * math.sqrt(2 ** (1 / mu) - 1) / (
        math.sqrt(math.pi) * math.gamma(mu - 0.5) * w)


# shape key -> (library function, centre param, width params, area param or None, init)
SHAPES: dict[str, dict] = {
    "gauss": {"label": "Gaussian", "func": "Gaussian", "center": "xc", "width": ("w",),
              "area": "A", "init": lambda c, h, f: {"xc": c, "w": f, "A": h * f * _C}},
    "lorentz": {"label": "Lorentzian", "func": "Lorentz", "center": "xc", "width": ("w",),
                "area": "A", "init": lambda c, h, f: {"xc": c, "w": f,
                                                      "A": h * f * math.pi / 2}},
    "voigt": {"label": "Voigt", "func": "Voigt", "center": "xc", "width": ("wG", "wL"),
              "area": "A", "init": lambda c, h, f: {"xc": c, "wG": 0.7 * f, "wL": 0.5 * f,
                                                    "A": h * f * 1.3}},
    "pvoigt": {"label": "Pseudo-Voigt", "func": "PsdVoigt1", "center": "xc", "width": ("w",),
               "area": "A", "init": lambda c, h, f: {"xc": c, "w": f, "mu": 0.5,
                                                     "A": h * f * (0.5 * _C + 0.25 * math.pi)}},
    "pearson7": {"label": "Pearson VII", "func": "PearsonVII", "center": "xc",
                 "width": ("w",), "area": "A",
                 "init": lambda c, h, f: {"xc": c, "w": f, "mu": 1.5,
                                          "A": h / _pv7_height_factor(1.5, f)}},
    "asym2sig": {"label": "Asym2Sig", "func": "Asym2Sig", "center": "xc",
                 "width": ("w1", "w2", "w3"), "area": None,
                 "init": lambda c, h, f: {"xc": c, "w1": f, "w2": f / 6, "w3": f / 6,
                                          "A": h / 0.9074}},
    "bigauss": {"label": "BiGaussian", "func": "BiGaussian", "center": "xc",
                "width": ("w1", "w2"), "area": None,
                "init": lambda c, h, f: {"xc": c, "H": h, "w1": f / _S2, "w2": f / _S2}},
    "emg": {"label": "EMG (GaussMod)", "func": "GaussMod", "center": "xc",
            "width": ("w", "t0"), "area": "A",
            "init": lambda c, h, f: {"xc": c - 0.1 * f, "w": 0.35 * f, "t0": 0.25 * f,
                                     "A": h * f * _C}},
    "logistpk": {"label": "Logistic", "func": "Logistpk", "center": "xc", "width": ("w",),
                 "area": None, "init": lambda c, h, f: {"xc": c, "w": f / 3.5255, "A": h}},
}


def _sorted(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    o = np.argsort(x, kind="stable")
    return x[o], y[o]


def _odd(n: int, m: int) -> int:
    n = max(5, int(n))
    n += (n + 1) % 2
    return min(n, m - 1 if (m - 1) % 2 else m - 2)


# ====================================================================== find peaks
def _fwhm_at(xs, v, i) -> float:
    try:
        w = ss.peak_widths(v, [i], rel_height=0.5)
        pos = np.arange(len(xs))
        return float(np.interp(w[3][0], pos, xs) - np.interp(w[2][0], pos, xs))
    except (ValueError, IndexError):
        return float("nan")


def find_peaks(x, y, method: str = "local_max", direction: str = "positive",
               height_pct: float = 5.0, max_peaks: int | None = None,
               smooth_window: int = 0, window_size: int = 5,
               prominence_pct: float = 1.0) -> list[dict]:
    """Peaks as dicts ``{index, x, y, height, fwhm, hidden}`` sorted by x. ``height_pct``
    keeps peaks whose height is at least this % of the Y range."""
    xs, ys = _sorted(x, y)
    n = len(xs)
    if n < 5:
        raise ValueError("At least 5 points are needed to find peaks.")
    sign = -1.0 if direction == "negative" else 1.0
    v = sign * ys
    vmin, vrng = float(v.min()), float(np.ptp(v)) or 1.0
    thr = vmin + height_pct / 100 * vrng
    win = _odd(smooth_window or max(5, n // 60), n)
    sm = ss.savgol_filter(v, win, 2) if n > win else v
    idx: list[int] = []
    hidden: set[int] = set()
    if method == "local_max":
        idx = list(ss.find_peaks(v, height=thr, prominence=prominence_pct / 100 * vrng)[0])
    elif method == "window":
        idx = list(ss.find_peaks(v, height=thr, distance=max(1, int(window_size)),
                                 prominence=prominence_pct / 100 * vrng)[0])
        h = max(1, int(window_size) // 2)
        idx = [i for i in idx if v[i] >= v[max(0, i - h): i + h + 1].max()]
    elif method == "first_derivative":
        d1 = ss.savgol_filter(v, win, 2, deriv=1) if n > win else np.gradient(v)
        cross = np.flatnonzero((d1[:-1] > 0) & (d1[1:] <= 0))
        for c in cross:
            i = c if v[c] >= v[c + 1] else c + 1
            lo, hi = max(0, i - win // 2), min(n, i + win // 2 + 1)
            i = lo + int(np.argmax(v[lo:hi]))
            if v[i] >= thr:
                idx.append(i)
    elif method == "second_derivative":
        win2 = _odd(max(win, n // 40), n)
        d2 = ss.savgol_filter(v, win2, 3, deriv=2) if n > win2 else \
            np.gradient(np.gradient(v))
        neg = -np.minimum(d2, 0.0)
        cand = ss.find_peaks(neg, height=0.02 * float(neg.max() or 1),
                             prominence=max(prominence_pct, 2.0) / 100 * float(neg.max() or 1),
                             distance=max(1, win2 // 2))[0]
        maxima = list(ss.find_peaks(sm, prominence=prominence_pct / 100 * vrng)[0])
        for i in cand:
            if v[i] >= thr:
                idx.append(int(i))
                if not any(abs(i - m) <= win2 // 2 for m in maxima):
                    hidden.add(int(i))
    elif method == "residual":
        first = find_peaks(xs, ys, "first_derivative", direction, height_pct, None,
                           smooth_window)
        idx = [p["index"] for p in first]
        if first:
            try:
                pf = fit_peaks(xs, ys, first, "gauss", baseline="none")
                r = sign * (ys - pf.result.fitted)
                rs = ss.savgol_filter(r, win, 2) if n > win else r
                extra = ss.find_peaks(rs, height=height_pct / 100 * vrng,
                                      prominence=prominence_pct / 100 * vrng)[0]
                for i in extra:
                    if all(abs(i - j) > win // 2 for j in idx):
                        idx.append(int(i))
                        hidden.add(int(i))
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                pass
    else:
        raise ValueError(f"Unknown peak finding method '{method}'")
    idx = sorted(set(int(i) for i in idx))
    peaks = []
    for i in idx:
        f = _fwhm_at(xs, v, i)
        if not np.isfinite(f) or f <= 0 or i in hidden:
            f = _fwhm_from_d2(xs, sm, i)
        peaks.append({"index": i, "x": float(xs[i]), "y": float(ys[i]),
                      "height": float(sign * (v[i] - vmin)), "fwhm": float(f),
                      "hidden": i in hidden})
    if max_peaks:
        peaks = sorted(peaks, key=lambda p: -abs(p["height"]))[: int(max_peaks)]
        peaks.sort(key=lambda p: p["x"])
    return peaks


def _fwhm_from_d2(xs, sm, i) -> float:
    """FWHM from the inflection points around ``i`` (σ ≈ half their distance)."""
    d2 = np.gradient(np.gradient(sm, xs), xs)
    n = len(xs)
    lo = i
    while lo > 0 and d2[lo] < 0:
        lo -= 1
    hi = i
    while hi < n - 1 and d2[hi] < 0:
        hi += 1
    sigma = 0.5 * (xs[hi] - xs[lo])
    span = float(np.ptp(xs)) or 1.0
    return float(_S2 * sigma) if sigma > 0 else span / 20


# ====================================================================== integrate
def _limits(xs, ys, i, left_stop, right_stop):
    a = i
    while a > left_stop and ys[a - 1] > 0:
        a -= 1
    b = i
    while b < right_stop and ys[b + 1] > 0:
        b += 1
    return a, b


def integrate_peaks(x, y, peaks, baseline=None, limits=None) -> list[dict]:
    """Integrate each peak of ``y − baseline``. ``limits``: list of (x_from, x_to) per peak;
    automatic limits run to where the signal crosses zero or to the valley between peaks."""
    xs, ys = _sorted(x, y)
    b = np.zeros(len(ys)) if baseline is None else np.asarray(baseline, float)
    if b.shape != ys.shape:
        xb, bb = _sorted(x, baseline)
        b = bb
    v = ys - b
    pos = sorted(peaks, key=lambda p: p["x"])
    centers = [int(np.argmin(np.abs(xs - p["x"]))) for p in pos]
    valleys = [c1 + int(np.argmin(v[c1:c2 + 1])) for c1, c2 in zip(centers[:-1], centers[1:])]
    out = []
    for k, c in enumerate(centers):
        sign = 1.0 if v[c] >= 0 else -1.0
        if limits is not None and limits[k] is not None:
            xa, xb_ = sorted(limits[k])
        else:
            ls = valleys[k - 1] if k > 0 else 0
            rs = valleys[k] if k < len(valleys) else len(xs) - 1
            a, bb_ = _limits(xs, sign * v, c, ls, rs)
            xa, xb_ = float(xs[a]), float(xs[bb_])
        inner = (xs > xa) & (xs < xb_)
        xx = np.concatenate([[xa], xs[inner], [xb_]])
        yy = np.concatenate([[np.interp(xa, xs, v)], v[inner], [np.interp(xb_, xs, v)]])
        area = float(np.trapezoid(yy, xx))
        absarea = float(np.trapezoid(np.abs(yy), xx))
        j = int(np.argmax(sign * yy))
        height = float(yy[j])
        half = sign * height / 2
        above = sign * yy >= half
        fwhm = float("nan")
        if above.any():
            ii = np.flatnonzero(above)
            l, r = ii[0], ii[-1]
            xl = xx[l] if l == 0 else np.interp(half, [sign * yy[l - 1], sign * yy[l]],
                                                [xx[l - 1], xx[l]])
            xr = xx[r] if r == len(xx) - 1 else np.interp(half, [sign * yy[r + 1],
                                                                 sign * yy[r]],
                                                          [xx[r + 1], xx[r]])
            fwhm = float(xr - xl)
        centroid = float(np.trapezoid(yy * xx, xx) / area) if area else float("nan")
        out.append({"peak": k + 1, "x_from": float(xa), "x_to": float(xb_), "area": area,
                    "abs_area": absarea, "x_max": float(xx[j]), "height": height,
                    "centroid": centroid, "fwhm": fwhm})
    total = sum(r["area"] for r in out)
    for r in out:
        r["area_pct"] = 100 * r["area"] / total if total else float("nan")
    return out


# ====================================================================== fit
def peak_model(n: int, shape: str = "gauss", baseline: str = "none",
               shared_width: bool = False) -> FitFunction:
    """Sum of ``n`` peaks of ``shape`` (+ optional polynomial baseline co-fit)."""
    spec = SHAPES[shape]
    g = fl.get_function(spec["func"])
    local = [p for p in g.params if p != "y0"]
    base = {"none": [], "constant": ["y0"], "linear": ["y0", "k"],
            "quadratic": ["y0", "k", "q"]}[baseline]
    shared = [p for p in spec["width"] if shared_width]
    names = list(base) + list(shared) + [f"{p}_{i + 1}" for i in range(n) for p in local
                                         if p not in shared]
    gpos = {p: g.params.index(p) for p in g.params}

    def cols(i):
        return [names.index(p if p in shared else f"{p}_{i + 1}") for p in local]

    colmap = [cols(i) for i in range(n)]

    def q_of(P, i):
        q = np.zeros(len(g.params))
        for p, c in zip(local, colmap[i]):
            q[gpos[p]] = P[c]
        return q

    def func(x, *P):
        x = np.asarray(x, float)
        P = np.asarray(P, float)
        y = np.zeros_like(x)
        for j, bname in enumerate(base):
            y = y + P[j] * x ** j
        for i in range(n):
            y = y + g(x, *q_of(P, i))
        return y

    def jac(x, *P):
        x = np.asarray(x, float)
        P = np.asarray(P, float)
        J = np.zeros((len(x), len(names)))
        for j in range(len(base)):
            J[:, j] = x ** j
        for i in range(n):
            Ji = g.jacobian(x, q_of(P, i))
            for p, c in zip(local, colmap[i]):
                J[:, c] += Ji[:, gpos[p]]
        return J

    label = spec["label"]
    formula = {"none": "y = ", "constant": "y = y0 + ", "linear": "y = y0 + k*x + ",
               "quadratic": "y = y0 + k*x + q*x^2 + "}[baseline] + f"Σ {label}_i(x)"
    return FitFunction(f"{n}×{label}", "Multiple Peaks", formula, tuple(names), func,
                       description=f"{n} {label} peak(s), baseline: {baseline}"
                                   + (", shared width" if shared else "") + ".",
                       jac=jac, meta={"shape": shape, "n_peaks": n, "base": tuple(base),
                                      "shared": tuple(shared), "local": tuple(local),
                                      "colmap": colmap, "peak_func": g.name})


@dataclass
class PeakFitResult:
    result: FitResult
    shape: str
    peaks: list[dict]          # per-peak table rows
    notes: list[str] = field(default_factory=list)

    @property
    def model(self) -> FitFunction:
        return self.result.model

    def component(self, i: int, xx) -> np.ndarray:
        """Peak ``i`` (0-based) alone, without the baseline."""
        m = self.model.meta
        g = fl.get_function(m["peak_func"])
        P = self.result.values
        q = np.zeros(len(g.params))
        for p, c in zip(m["local"], m["colmap"][i]):
            q[g.params.index(p)] = P[c]
        return g(np.asarray(xx, float), *q)

    def baseline(self, xx) -> np.ndarray:
        xx = np.asarray(xx, float)
        P = self.result.values
        return sum((P[j] * xx ** j for j in range(len(self.model.meta["base"]))),
                   np.zeros_like(xx))

    def components(self, xx) -> list[np.ndarray]:
        return [self.component(i, xx) for i in range(self.model.meta["n_peaks"])]

    def cumulative(self, xx) -> np.ndarray:
        return self.result.predict(xx)


def _peak_props(pf: PeakFitResult, i: int, span: tuple[float, float]) -> dict:
    spec = SHAPES[pf.shape]
    m = pf.model.meta
    res = pf.result
    colmap = m["colmap"][i]
    local = m["local"]
    cidx = colmap[local.index(spec["center"])]
    xc = float(res.values[cidx])
    lo, hi = span
    width_guess = max(float(np.ptp(res.x)) / 50, 1e-12)
    for wname in spec["width"]:
        if wname in local:
            width_guess = max(width_guess, abs(float(res.values[colmap[local.index(wname)]])))
    a, b = min(lo, xc - 60 * width_guess), max(hi, xc + 60 * width_guess)
    xx = np.linspace(a, b, 20001)
    yc = pf.component(i, xx)
    j = int(np.argmax(np.abs(yc)))
    height = float(yc[j])
    above = np.abs(yc) >= abs(height) / 2
    ii = np.flatnonzero(above)
    fwhm = float(xx[ii[-1]] - xx[ii[0]]) if len(ii) > 1 else float("nan")
    if spec["area"]:
        aidx = colmap[local.index(spec["area"])]
        area, area_se = float(res.values[aidx]), float(res.errors[aidx])
    else:
        area = float(np.trapezoid(yc, xx))
        area_se = _area_se(pf, i, xx)
    centroid = float(np.trapezoid(yc * xx, xx) / area) if area else float("nan")
    return {"peak": i + 1, "center": xc, "center_se": float(res.errors[cidx]),
            "height": height, "area": area, "area_se": area_se, "fwhm": fwhm,
            "x_max": float(xx[j]), "centroid": centroid}


def _area_se(pf: PeakFitResult, i: int, xx) -> float:
    res = pf.result
    cols = [c for c in pf.model.meta["colmap"][i] if res.free[c]]
    base = res.values.copy()
    g = np.zeros(len(res.values))

    def area_with(vals):
        saved = res.values
        res.values = vals
        try:
            return float(np.trapezoid(pf.component(i, xx), xx))
        finally:
            res.values = saved
    for c in cols:
        h = 1e-6 * max(abs(base[c]), 1e-8)
        up, dn = base.copy(), base.copy()
        up[c] += h
        dn[c] -= h
        g[c] = (area_with(up) - area_with(dn)) / (2 * h)
    var = float(g @ res.cov @ g)
    return math.sqrt(var) if var >= 0 else float("nan")


def fit_peaks(x, y, peaks, shape: str = "gauss", baseline: str = "none",
              shared_width: bool = False, fixed_width: float | None = None,
              fix_centers: bool = False, center_range: float = 1.0, weights=None,
              constraints=None, algorithm: str = "auto") -> PeakFitResult:
    """Fit a sum of peaks started from ``peaks`` (dicts with x, y/height, fwhm, or plain
    centre values). ``center_range``: each centre may move ± this many FWHM."""
    xs, ys = _sorted(x, y)
    span = float(np.ptp(xs)) or 1.0
    dx = float(np.min(np.diff(np.unique(xs)))) if len(xs) > 1 else span / 100
    plist = []
    for p in peaks:
        if isinstance(p, dict):
            plist.append(p)
        else:
            c = float(p)
            plist.append({"x": c, "y": float(np.interp(c, xs, ys)), "fwhm": span / 20})
    if not plist:
        raise ValueError("No peaks to fit: find peaks first or enter peak centres.")
    plist.sort(key=lambda p: p["x"])
    model = peak_model(len(plist), shape, baseline, shared_width)
    spec = SHAPES[shape]
    m = model.meta
    base0 = float(np.percentile(ys, 5)) if baseline != "none" else 0.0
    p0, bounds, fixed = {}, {}, {}
    if baseline != "none":
        p0["y0"] = base0
        for extra in m["base"][1:]:
            p0[extra] = 0.0
    for i, pk in enumerate(plist):
        f = pk.get("fwhm") or span / 20
        f = max(float(f) if np.isfinite(f) else span / 20, 2 * dx)
        if fixed_width:
            f = float(fixed_width)
        h = float(pk.get("y", np.interp(pk["x"], xs, ys))) - base0
        init = spec["init"](float(pk["x"]), h, f)
        for p in m["local"]:
            key = p if p in m["shared"] else f"{p}_{i + 1}"
            if key not in p0:
                p0[key] = float(init[p])
            if p == spec["center"]:
                if fix_centers:
                    fixed[key] = float(pk["x"])
                else:
                    bounds[key] = (float(pk["x"]) - center_range * f - dx,
                                   float(pk["x"]) + center_range * f + dx)
            elif p in spec["width"] and p not in ("t0",):
                if fixed_width and p == spec["width"][0]:
                    fixed[key] = p0[key]
                elif key not in bounds:
                    bounds[key] = (dx / 20 if p != "wL" else 0.0, 4 * span)
            elif p == "t0":
                bounds[key] = (dx / 100, 4 * span)
            elif p == "mu":
                bounds[key] = (0.0, 1.0) if shape == "pvoigt" else (0.51, 1e3)
    res = fit.fit_curve(xs, ys, model, params0=p0, bounds=bounds or None,
                        fixed=fixed or {}, weights=weights, constraints=constraints,
                        algorithm=algorithm)
    pf = PeakFitResult(res, shape, [])
    rng = (float(xs.min()), float(xs.max()))
    pf.peaks = [_peak_props(pf, i, rng) for i in range(len(plist))]
    total = sum(r["area"] for r in pf.peaks)
    for r in pf.peaks:
        r["area_pct"] = 100 * r["area"] / total if total else float("nan")
    return pf


# ====================================================================== the analysis
@dataclass
class PeakAnalysis:
    """State of one Peak Analyzer session (what the wizard pages edit)."""
    x: np.ndarray
    y: np.ndarray
    goal: str = "fit"
    baseline_mode: str = "none"
    baseline_opts: dict = field(default_factory=dict)
    anchors: tuple[np.ndarray, np.ndarray] | None = None
    find_opts: dict = field(default_factory=lambda: {"method": "local_max",
                                                     "height_pct": 5.0})
    limits: list | None = None
    fit_opts: dict = field(default_factory=lambda: {"shape": "gauss", "baseline": "none"})
    baseline: np.ndarray | None = None
    peaks: list[dict] = field(default_factory=list)
    integration: list[dict] = field(default_factory=list)
    fit: PeakFitResult | None = None
    keep_peaks: bool = False      # peaks edited by hand: do not find them again on a re-run

    def __post_init__(self):
        self.x, self.y = _sorted(self.x, self.y)

    @property
    def subtracted(self) -> np.ndarray:
        return self.y - (self.baseline if self.baseline is not None else 0.0)

    def find_anchors(self, n: int = 8, method: str = "minima") -> tuple:
        self.anchors = bl.find_anchors(self.x, self.y, n, method,
                                       self.baseline_opts.get("smooth", 0))
        return self.anchors

    def compute_baseline(self) -> np.ndarray:
        opts = dict(self.baseline_opts)
        if self.baseline_mode == "anchors":
            if self.anchors is None:
                self.find_anchors(opts.get("n_anchors", 8), opts.get("anchor_method", "minima"))
            opts["anchors_x"], opts["anchors_y"] = self.anchors
        self.baseline = bl.create_baseline(self.x, self.y, self.baseline_mode, **opts)
        return self.baseline

    def find(self) -> list[dict]:
        o = dict(self.find_opts)
        self.peaks = find_peaks(self.x, self.subtracted, o.pop("method", "local_max"), **o)
        return self.peaks

    def integrate(self) -> list[dict]:
        if not self.peaks:
            self.find()
        self.integration = integrate_peaks(self.x, self.subtracted, self.peaks,
                                           limits=self.limits)
        return self.integration

    def fit_peaks(self) -> PeakFitResult:
        if not self.peaks:
            self.find()
        o = dict(self.fit_opts)
        self.fit = fit_peaks(self.x, self.subtracted, self.peaks, o.pop("shape", "gauss"),
                             **o)
        return self.fit

    def run(self) -> "PeakAnalysis":
        if self.baseline_mode != "none" or self.goal in ("baseline", "subtract"):
            self.compute_baseline()
        if self.goal in ("find", "integrate", "fit") and not (self.keep_peaks and self.peaks):
            self.find()
        if self.goal == "integrate":
            self.integrate()
        if self.goal == "fit":
            self.fit_peaks()
        return self


def batch(x, ys: list, labels: list[str], template: PeakAnalysis) -> list[dict]:
    """Batch peak analysis: the settings of ``template`` applied to every Y (summary rows)."""
    rows = []
    for lab, y in zip(labels, ys):
        pa = PeakAnalysis(x, y, template.goal, template.baseline_mode,
                          dict(template.baseline_opts), None, dict(template.find_opts),
                          template.limits, dict(template.fit_opts))
        try:
            pa.run()
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            rows.append({"dataset": lab, "error": str(exc)})
            continue
        table = pa.fit.peaks if pa.fit is not None else (pa.integration or pa.peaks)
        for r in table:
            rows.append({"dataset": lab, **{k: v for k, v in r.items() if k != "hidden"}})
    return rows

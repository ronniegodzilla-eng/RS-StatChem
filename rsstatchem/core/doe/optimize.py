"""Numerical and graphical optimisation as in Design-Expert's Optimization node.

Derringer-Suich desirability for each response (and optionally each factor) with goals
maximize, minimize, target, in range, Cpk or none; lower/upper limits, weights and importance;
overall desirability D = (prod d_i^r_i)^(1/sum r_i).

Besides the predicted value, a response goal can be applied to a bound of its confidence,
prediction or tolerance interval ("use interval": the one-sided lower bound for maximize, the
upper bound for minimize, both bounds of the two-sided interval for target and in range).
The standard error of the mean (``kind="se"``, to keep the search where the design supports
the model) and the propagation of error (``kind="poe"``, robust design) of each response can
be given goals like responses. Factors may be "equal to" a value (held fixed) and numeric
factors may be discrete (only listed levels are allowed).

The search uses many random starts plus the design points, each refined by Nelder-Mead inside
the factor ranges (mixtures in the constrained simplex; categoric factors and, when there are
few combinations, discrete factors are enumerated; otherwise discrete factors are relaxed,
snapped to the nearest levels and the rest re-optimised). Also: feasibility for the graphical
overlay plot (optionally on interval bounds) and Taguchi signal-to-noise ratios.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, fields

import numpy as np
from scipy import optimize

from rsstatchem.core.doe import intervals, poe
from rsstatchem.core.doe.model import Fit, Space

GOALS = [("none", "none"), ("maximize", "maximize"), ("minimize", "minimize"),
         ("target", "is target"), ("in_range", "is in range"), ("cpk", "Cpk")]
FACTOR_GOALS = [("in_range", "is in range"), ("maximize", "maximize"),
                ("minimize", "minimize"), ("target", "is target"), ("equal", "is equal to")]
DERIVED_GOALS = [g for g in GOALS if g[0] != "cpk"]        # goals for SE and POE rows
CRITERION_KINDS = ("response", "factor", "se", "poe")
INTERVALS = intervals.KINDS
INTERVAL_SIDES = [("auto", "Auto (one-sided for max/min)"), ("two", "Two-sided")]
CPK_LIMITS = (0.0, 1.5)


@dataclass
class Criterion:
    name: str
    kind: str = "response"           # response | factor | se | poe
    goal: str = "none"
    lower: float = 0.0
    upper: float = 1.0
    target: float | None = None
    w_lower: float = 1.0
    w_upper: float = 1.0
    importance: int = 3
    interval: str = "none"           # none | ci | pi | ti  (response goals only)
    side: str = "auto"               # auto | two
    lsl: float | None = None         # Cpk specification limits
    usl: float | None = None

    _NEW = ("interval", "side", "lsl", "usl")

    def to_meta(self) -> dict:
        """JSON dict; the newer keys are written only when used, so files that do not use
        them stay readable by older versions."""
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        defaults = {"interval": "none", "side": "auto", "lsl": None, "usl": None}
        for k in self._NEW:
            if d[k] == defaults[k]:
                del d[k]
        return d

    @classmethod
    def from_meta(cls, d: dict) -> "Criterion":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def key(self) -> tuple[str, str]:
        return self.kind, self.name

    @property
    def active(self) -> bool:
        return self.goal not in ("none",) and not (self.kind == "factor"
                                                   and self.goal in ("in_range", "equal"))

    def interval_side(self) -> str:
        """Side of the interval the goal is applied to."""
        if self.interval in (None, "", "none"):
            return "two"
        if self.side == "two" or self.goal not in ("maximize", "minimize"):
            return "two"
        return "lower" if self.goal == "maximize" else "upper"


def desirability(y, c: Criterion) -> np.ndarray:
    """Individual desirability d (Derringer & Suich, 1980); for the Cpk goal ``y`` is the
    Cpk and d ramps up from the lower to the upper limit (default 0 to 1.5)."""
    y = np.asarray(y, float)
    lo, hi = c.lower, c.upper
    span = hi - lo if hi > lo else 1e-12
    if c.goal in ("maximize", "cpk"):
        d = np.clip((y - lo) / span, 0, 1) ** c.w_lower
    elif c.goal == "minimize":
        d = np.clip((hi - y) / span, 0, 1) ** c.w_upper
    elif c.goal == "target":
        t = c.target if c.target is not None else (lo + hi) / 2
        d = np.where(y <= t, np.clip((y - lo) / max(t - lo, 1e-12), 0, 1) ** c.w_lower,
                     np.clip((hi - y) / max(hi - t, 1e-12), 0, 1) ** c.w_upper)
        d = np.where((y < lo) | (y > hi), 0.0, d)
    elif c.goal == "in_range":
        d = ((y >= lo) & (y <= hi)).astype(float)
    else:
        d = np.ones_like(y)
    return np.where(np.isfinite(y), d, 0.0)


def overall(ds: list[np.ndarray], importances: list[int]) -> np.ndarray:
    if not ds:
        return np.array([1.0])
    r = np.asarray(importances, float)
    d = np.vstack([np.atleast_1d(x) for x in ds])
    with np.errstate(divide="ignore"):
        logd = np.where(d > 0, np.log(np.clip(d, 1e-300, None)), -np.inf)
    out = np.exp((r[:, None] * logd).sum(axis=0) / r.sum())
    return np.where(np.all(d > 0, axis=0), out, 0.0)


def cpk(mean, sigma, lsl: float | None, usl: float | None) -> np.ndarray:
    """Process capability Cpk = min(USL - μ, μ - LSL) / (3σ) (one limit: that side only)."""
    mean = np.asarray(mean, float)
    sigma = np.maximum(np.asarray(sigma, float), 1e-300)
    parts = []
    if usl is not None:
        parts.append((usl - mean) / (3 * sigma))
    if lsl is not None:
        parts.append((mean - lsl) / (3 * sigma))
    if not parts:
        return np.full(mean.shape, np.nan)
    return np.minimum.reduce(parts)


def se_mean(fit: Fit, actual: np.ndarray) -> np.ndarray:
    """Standard error of the predicted mean on the analysis scale, s·√(x'(X'X)⁻¹x)."""
    _, h = fit.predict_coded(fit.space.to_coded(np.atleast_2d(actual)))
    return np.sqrt(h * fit.ms_res)


def extra_label(kind: str, name: str) -> str:
    return {"se": "SE", "poe": "POE", "cpk": "Cpk"}.get(kind, kind) + f"({name})"


@dataclass
class Problem:
    space: Space
    fits: dict                     # response column -> Fit
    criteria: list[Criterion]      # responses, factors, SE and POE rows
    fixed_levels: dict = field(default_factory=dict)  # categoric factor index -> level index
    discrete: dict = field(default_factory=dict)      # numeric factor index -> allowed levels
    poe_sd: dict = field(default_factory=dict)        # factor column -> σ (actual units)
    poe_second_order: bool = False
    level: float | None = None                        # interval confidence (None: default)
    proportion: float | None = None                   # TI population proportion

    def factor_index(self, column: str) -> int | None:
        return next((i for i, f in enumerate(self.space.factors) if f.column == column), None)

    def factor_criterion(self, j: int) -> Criterion:
        f = self.space.factors[j]
        for c in self.criteria:
            if c.kind == "factor" and c.name == f.column:
                return c
        return Criterion(f.column, "factor", "in_range", f.low, f.high)

    def fixed_values(self) -> dict[int, float]:
        """Factors whose goal is "equal to" -> value."""
        out = {}
        for c in self.criteria:
            if c.kind == "factor" and c.goal == "equal":
                j = self.factor_index(c.name)
                if j is not None and not self.space.factors[j].categoric:
                    out[j] = float(c.target if c.target is not None
                                   else (c.lower + c.upper) / 2)
        return out

    def bounds_actual(self) -> list[tuple[float, float]]:
        out = []
        fixed = self.fixed_values()
        for j, f in enumerate(self.space.factors):
            if j in fixed:
                out.append((fixed[j], fixed[j]))
                continue
            c = self.factor_criterion(j)
            out.append((max(c.lower, f.low) if self.space.mixture else c.lower,
                        min(c.upper, f.high) if self.space.mixture else c.upper))
        return out

    def discrete_levels(self, j: int) -> list[float]:
        lv = sorted(float(v) for v in self.discrete.get(j, []))
        lo, hi = self.bounds_actual()[j]
        inside = [v for v in lv if lo - 1e-9 <= v <= hi + 1e-9]
        return inside or lv

    def active_criteria(self) -> list[Criterion]:
        """The criteria that contribute a desirability, in the order of ``evaluate()['d']``."""
        out = []
        for c in self.criteria:
            if not c.active:
                continue
            if c.kind in ("response", "se", "poe") and c.name not in self.fits:
                continue
            if c.kind == "factor" and self.factor_index(c.name) is None:
                continue
            out.append(c)
        return out

    def sigma(self, fit: Fit, actual: np.ndarray) -> np.ndarray:
        """SD of one observation in original units: POE with the factor SDs (residual SD
        alone when none are set)."""
        return poe.poe(fit, actual, poe.sd_vector(self.space, self.poe_sd),
                       self.poe_second_order)

    def value(self, c: Criterion, actual: np.ndarray, cache: dict | None = None):
        """(value, low bound, high bound) the goal of ``c`` is judged on."""
        cache = {} if cache is None else cache
        if c.kind == "factor":
            y = actual[:, self.factor_index(c.name)]
            return y, y, y
        fit = self.fits[c.name]
        if c.kind == "se":
            y = se_mean(fit, actual)
            return y, y, y
        if ("sigma", c.name) not in cache and (c.kind == "poe" or c.goal == "cpk"):
            cache[("sigma", c.name)] = self.sigma(fit, actual)
        if c.kind == "poe":
            y = cache[("sigma", c.name)]
            return y, y, y
        if c.goal == "cpk":
            z, _ = fit.predict_coded(fit.space.to_coded(actual))
            mean = intervals._bias_mean(fit, z)
            y = cpk(mean, cache[("sigma", c.name)], c.lsl, c.usl)
            return y, y, y
        if ("pred", c.name) not in cache:
            cache[("pred", c.name)] = fit.predict(actual)
        y = cache[("pred", c.name)]
        if c.interval in (None, "", "none") or c.goal == "none":
            return y, y, y
        side = c.interval_side()
        key = ("iv", c.name, side)
        if key not in cache:
            cache[key] = intervals.predict(fit, actual, self.level, self.proportion, side,
                                           kinds=(c.interval,))
        lo, hi = cache[key][c.interval]
        return y, lo, hi

    def evaluate(self, actual: np.ndarray) -> dict:
        actual = np.atleast_2d(np.asarray(actual, float))
        preds, extra, ds, imps = {}, {}, [], []
        viol = np.zeros(len(actual))
        cache: dict = {}
        for c in self.criteria:
            if c.kind == "response" and c.name in self.fits:
                if ("pred", c.name) not in cache:
                    cache[("pred", c.name)] = self.fits[c.name].predict(actual)
                preds[c.name] = cache[("pred", c.name)]
        for c in self.active_criteria():
            y, lo, hi = self.value(c, actual, cache)
            if c.kind in ("se", "poe"):
                extra[extra_label(c.kind, c.name)] = y
            elif c.goal == "cpk":
                extra[extra_label("cpk", c.name)] = y
            elif c.kind == "response" and c.interval not in (None, "", "none"):
                side = c.interval_side()
                if side != "upper":
                    extra[f"{c.interval.upper()} low({c.name})"] = lo
                if side != "lower":
                    extra[f"{c.interval.upper()} high({c.name})"] = hi
            if c.kind == "response" and c.goal in ("maximize",) and lo is not y:
                d, judged = desirability(lo, c), (lo, lo)
            elif c.kind == "response" and c.goal == "minimize" and hi is not y:
                d, judged = desirability(hi, c), (hi, hi)
            elif c.kind == "response" and lo is not y:
                d, judged = np.minimum(desirability(lo, c), desirability(hi, c)), (lo, hi)
            else:
                d, judged = desirability(y, c), (y, y)
            ds.append(d)
            imps.append(c.importance)
            span = (c.upper - c.lower) or 1.0
            with np.errstate(invalid="ignore"):
                v = (np.clip(c.lower - judged[0], 0, None) + np.clip(judged[1] - c.upper, 0,
                                                                      None)) / span
            if c.goal in ("maximize", "cpk"):
                v = np.clip(c.lower - judged[0], 0, None) / span
            elif c.goal == "minimize":
                v = np.clip(judged[1] - c.upper, 0, None) / span
            viol += np.where(np.isfinite(v), v, 1e6)
        D = overall(ds, imps) if ds else np.ones(len(actual))
        return {"pred": preds, "d": ds, "D": D, "viol": viol, "extra": extra}


def _mixture_points(lo: np.ndarray, hi: np.ndarray, total: float, n: int, rng) -> np.ndarray:
    k = len(lo)
    if k == 0:
        return np.zeros((n, 0))
    room = total - lo.sum()
    if room < -1e-12:
        raise ValueError("Could not find points inside the constrained mixture region.")
    pts: list = []
    tries = 0
    while len(pts) < n and tries < 200:
        tries += 1
        cand = rng.dirichlet(np.ones(k), size=4 * n) * max(room, 0) + lo
        ok = np.all((cand >= lo - 1e-12) & (cand <= hi + 1e-12), axis=1)
        pts += list(cand[ok])
    if not pts:
        raise ValueError("Could not find points inside the constrained mixture region.")
    return np.array(pts[:n])


def _random_points(problem: Problem, n: int, rng) -> np.ndarray:
    sp = problem.space
    b = np.array(problem.bounds_actual())
    if not sp.mixture:
        return b[:, 0] + rng.random((n, sp.k)) * (b[:, 1] - b[:, 0])
    return _mixture_points(b[:, 0], b[:, 1], sp.total, n, rng)


MAX_ENUMERATED = 64   # discrete-level combinations searched one by one


def _snap(x: np.ndarray, levels: dict[int, list[float]]) -> np.ndarray:
    x = np.array(x, float)
    for j, lv in levels.items():
        a = np.asarray(lv, float)
        x[j] = a[int(np.argmin(np.abs(a - x[j])))]
    return x


def solve(problem: Problem, starts: int = 30, seed: int | None = 0, max_solutions: int = 30,
          design_points: np.ndarray | None = None, duplicate: float = 2e-3) -> list[dict]:
    """Maximise D from random starts; returns distinct solutions sorted by desirability.
    ``duplicate`` is the filter: solutions closer than this (max |Δ| as a fraction of each
    factor range) are dropped."""
    sp = problem.space
    rng = np.random.default_rng(seed)
    starts = max(int(starts), 1)
    cat = [j for j, f in enumerate(sp.factors) if f.categoric]
    fixed = problem.fixed_values()
    disc = {j: problem.discrete_levels(j) for j in problem.discrete
            if j < sp.k and not sp.factors[j].categoric and j not in fixed
            and problem.discrete.get(j)} if not sp.mixture else {}
    n_disc = int(np.prod([len(v) for v in disc.values()])) if disc else 1
    enumerate_disc = bool(disc) and n_disc <= MAX_ENUMERATED
    cat_combos = list(itertools.product(*[[problem.fixed_levels[j]] if j in problem.fixed_levels
                                          else range(len(sp.factors[j].levels)) for j in cat]))
    disc_cols = list(disc)
    disc_combos = list(itertools.product(*[disc[j] for j in disc_cols])) if enumerate_disc \
        else [()]
    b_all = np.array(problem.bounds_actual())
    solutions = []
    for combo in cat_combos or [()]:
        if sp.mixture:
            solutions += _solve_mixture(problem, combo, cat, b_all, fixed, starts, rng,
                                        design_points)
            continue
        for dcombo in disc_combos:
            pinned = dict(fixed)
            pinned.update(dict(zip(disc_cols, dcombo)))
            n_starts = starts if len(disc_combos) == 1 else max(5, starts // len(disc_combos))
            sols = _solve_process(problem, combo, cat, pinned, b_all, n_starts, rng,
                                  design_points)
            if disc and not enumerate_disc:
                # relaxed search: snap the discrete factors and re-optimise the others
                refined = []
                for x in sols:
                    xs = _snap(x, disc)
                    pin2 = dict(pinned)
                    pin2.update({j: xs[j] for j in disc})
                    refined += _solve_process(problem, combo, cat, pin2, b_all, 0, rng,
                                              np.array([xs]))
                sols = refined
            solutions += sols
    if not solutions:
        return []
    X = np.array(solutions)
    ev = problem.evaluate(X)
    D = ev["D"]
    order = np.lexsort((ev["viol"], -D))
    out, seen = [], []
    scale = np.array([(f.high - f.low) or 1.0 for f in sp.factors])
    for i in order:
        key = X[i] / scale
        if any(np.max(np.abs(key - s)) < duplicate for s in seen):
            continue
        seen.append(key)
        out.append({"x": X[i], "pred": {k: float(v[i]) for k, v in ev["pred"].items()},
                    "d": [float(d[i]) for d in ev["d"]], "D": float(D[i]),
                    "extra": {k: float(v[i]) for k, v in ev["extra"].items()}})
        if len(out) >= max_solutions:
            break
    return out


def _objective(problem: Problem, full):
    def f(u):
        ev = problem.evaluate(full(u))
        D = float(ev["D"][0])
        return -D if D > 0 else float(ev["viol"][0])
    return f


def _solve_process(problem, combo, cat, pinned, b_all, n_starts, rng, design_points):
    sp = problem.space
    free = [j for j in range(sp.k) if j not in cat and j not in pinned]

    def full(xf):
        x = np.empty(sp.k)
        x[free] = xf
        for j, v in pinned.items():
            x[j] = v
        for j, lv in zip(cat, combo):
            x[j] = sp.factors[j].levels[lv]
        return x

    if not free:
        return [full(np.array([]))]
    lo, hi = b_all[free, 0], b_all[free, 1]
    pts = list(lo + rng.random((n_starts, len(free))) * (hi - lo))
    if design_points is not None:
        for p in np.atleast_2d(design_points):
            p = np.asarray(p, float)
            if np.all(p[free] >= lo - 1e-9) and np.all(p[free] <= hi + 1e-9):
                pts.append(p[free])
    obj = _objective(problem, full)
    out = []
    for s in pts:
        res = optimize.minimize(obj, np.asarray(s, float), method="Nelder-Mead",
                                bounds=list(zip(lo, hi)),
                                options={"xatol": 1e-7, "fatol": 1e-10,
                                         "maxiter": 300 * max(len(free), 1)})
        out.append(full(np.clip(res.x, lo, hi)))
    return out


def _solve_mixture(problem, combo, cat, b_all, fixed, starts, rng, design_points):
    sp = problem.space
    lo, hi = b_all[:, 0], b_all[:, 1]
    free = [j for j in range(sp.k) if j not in fixed]
    total_free = sp.total - sum(fixed.values())

    def to_x(u):
        x = np.empty(sp.k)
        for j, v in fixed.items():
            x[j] = v
        if free:
            x[free[:-1]] = u
            x[free[-1]] = total_free - float(np.sum(u))
        return x

    if len(free) <= 1:
        return [to_x(np.array([]))]

    def objective(u):
        x = to_x(u)
        pen = float(np.sum(np.clip(lo - x, 0, None) + np.clip(x - hi, 0, None)))
        if pen > 1e-12:
            return 10 + 100 * pen
        ev = problem.evaluate(x)
        D = float(ev["D"][0])
        return -D if D > 0 else float(ev["viol"][0])

    pts = list(_mixture_points(lo[free], hi[free], total_free, starts, rng))
    if design_points is not None and not fixed:
        pts += [np.asarray(p, float) for p in np.atleast_2d(design_points)]
    bounds = list(zip(lo[free[:-1]], hi[free[:-1]]))
    out = []
    for s in pts:
        u0 = np.asarray(s, float)[:-1]
        res = optimize.minimize(objective, u0, method="Nelder-Mead", bounds=bounds,
                                options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 400 * sp.k})
        out.append(to_x(res.x))
    return out


# ---------------------------------------------------------------------------- graphical
def overlay_bounds(fit: Fit, actual: np.ndarray, lower: float | None, upper: float | None,
                   interval: str = "none", level: float | None = None,
                   proportion: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(value compared with the lower limit, value compared with the upper limit): the
    prediction, or the bounds of the interval — one-sided when only one limit is set."""
    if interval in (None, "", "none"):
        y = fit.predict(actual)
        return y, y
    side = "two" if (lower is not None and upper is not None) else \
        ("lower" if lower is not None else "upper")
    return intervals.predict(fit, actual, level, proportion, side, kinds=(interval,))[interval]


def feasible(problem: Problem, actual: np.ndarray) -> np.ndarray:
    """Graphical optimisation: True where every response lies within its limits (on the
    interval bounds for criteria that use an interval)."""
    ok = np.ones(len(np.atleast_2d(actual)), bool)
    for c in problem.criteria:
        if c.kind != "response" or c.name not in problem.fits or c.goal == "cpk":
            continue
        lo, hi = overlay_bounds(problem.fits[c.name], actual, c.lower, c.upper, c.interval,
                                problem.level, problem.proportion)
        ok &= (lo >= c.lower) & (hi <= c.upper)
    return ok


def default_criteria(fits: dict[str, Fit], space: Space) -> list[Criterion]:
    out = []
    for f in space.factors:
        if not f.categoric:
            out.append(Criterion(f.column, "factor", "in_range", f.low, f.high))
    for name, fit in fits.items():
        y = fit.obs.y
        out.append(Criterion(name, "response", "maximize", float(np.min(y)), float(np.max(y))))
    return out


def design_range(values: np.ndarray) -> tuple[float, float]:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return 0.0, 1.0
    lo, hi = float(v.min()), float(v.max())
    if hi - lo <= 1e-7 * max(abs(hi), 1.0):     # constant (e.g. POE of a linear model)
        return lo, lo + max(abs(lo), 1.0) * 0.1
    return lo, hi


def derived_criterion(kind: str, col: str, fit: Fit, design_points: np.ndarray,
                      poe_sd: dict | None = None, second_order: bool = False) -> Criterion:
    """A goal-less SE or POE row whose default limits span the values at the design
    points (an "in range" goal with these limits keeps the search inside the region the
    design supports)."""
    if kind == "se":
        vals = se_mean(fit, design_points)
    else:
        vals = poe.poe(fit, design_points, poe.sd_vector(fit.space, poe_sd), second_order)
    lo, hi = design_range(vals)
    return Criterion(col, kind, "none", lo, hi)


def robust_criteria(criteria: list[Criterion], fits: dict, design_points: np.ndarray,
                    poe_sd: dict | None, second_order: bool = False,
                    importance: int = 3) -> list[Criterion]:
    """Robust design: add (or switch on) a "minimize POE" goal for every response that has a
    goal, with limits from the POE range over the design points."""
    out = list(criteria)
    for c in list(criteria):
        if c.kind != "response" or c.goal == "none" or c.name not in fits:
            continue
        row = next((r for r in out if r.kind == "poe" and r.name == c.name), None)
        new = derived_criterion("poe", c.name, fits[c.name], design_points, poe_sd,
                                second_order)
        if row is None:
            row = new
            out.append(row)
        else:
            row.lower, row.upper = new.lower, new.upper
        row.goal, row.importance = "minimize", importance
    return out


# ---------------------------------------------------------------------------- Taguchi
SN_TYPES = [("larger", "Larger is better"), ("smaller", "Smaller is better"),
            ("nominal", "Nominal is best (mean²/variance)"),
            ("nominal_var", "Nominal is best (variance only)")]


def sn_ratio(Y: np.ndarray, kind: str) -> np.ndarray:
    """Signal-to-noise ratio per run from replicate columns (rows = runs)."""
    Y = np.atleast_2d(np.asarray(Y, float))
    out = np.full(len(Y), np.nan)
    for i, row in enumerate(Y):
        y = row[np.isfinite(row)]
        if not len(y):
            continue
        if kind == "larger":
            if np.any(y == 0):
                continue
            out[i] = -10 * math.log10(np.mean(1 / y ** 2))
        elif kind == "smaller":
            out[i] = -10 * math.log10(np.mean(y ** 2))
        elif kind == "nominal":
            if len(y) < 2 or np.var(y, ddof=1) == 0:
                continue
            out[i] = 10 * math.log10(np.mean(y) ** 2 / np.var(y, ddof=1))
        elif kind == "nominal_var":
            if len(y) < 2 or np.var(y, ddof=1) == 0:
                continue
            out[i] = -10 * math.log10(np.var(y, ddof=1))
        else:
            raise ValueError(kind)
    return out


def response_table(levels: np.ndarray, values: np.ndarray, n_levels: list[int]) -> dict:
    """Mean of ``values`` at each level of each factor, Delta = max - min and Rank."""
    means = []
    for j, nl in enumerate(n_levels):
        means.append([float(np.nanmean(values[levels[:, j] == lv])) if np.any(
            levels[:, j] == lv) else float("nan") for lv in range(nl)])
    deltas = [np.nanmax(m) - np.nanmin(m) for m in means]
    order = np.argsort(-np.array(deltas), kind="stable")
    rank = np.empty(len(deltas), int)
    rank[order] = np.arange(1, len(deltas) + 1)
    return {"means": means, "delta": deltas, "rank": rank.tolist()}

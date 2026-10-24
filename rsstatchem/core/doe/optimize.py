"""Numerical and graphical optimisation as in Design-Expert's Optimization node.

Derringer-Suich desirability for each response (and optionally each factor) with goals
maximize, minimize, target, in range or none, lower/upper limits, weights and importance;
overall desirability D = (prod d_i^r_i)^(1/sum r_i). The search uses many random starts plus
the design points, each refined by Nelder-Mead inside the factor ranges (mixtures in pseudo
components on the constrained simplex; categoric factors are enumerated). Also: feasibility for
the graphical-optimisation overlay plot and Taguchi signal-to-noise ratios with response tables.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize

from rsstatchem.core.doe.model import Fit, Space

GOALS = [("none", "none"), ("maximize", "maximize"), ("minimize", "minimize"),
         ("target", "is target"), ("in_range", "is in range")]
FACTOR_GOALS = [("in_range", "is in range"), ("maximize", "maximize"),
                ("minimize", "minimize"), ("target", "is target")]


@dataclass
class Criterion:
    name: str
    kind: str = "response"           # response | factor
    goal: str = "none"
    lower: float = 0.0
    upper: float = 1.0
    target: float | None = None
    w_lower: float = 1.0
    w_upper: float = 1.0
    importance: int = 3

    def to_meta(self) -> dict:
        return dict(self.__dict__)

    @property
    def active(self) -> bool:
        return self.goal not in ("none",) and not (self.kind == "factor"
                                                   and self.goal == "in_range")


def desirability(y, c: Criterion) -> np.ndarray:
    """Individual desirability d (Derringer & Suich, 1980)."""
    y = np.asarray(y, float)
    lo, hi = c.lower, c.upper
    span = hi - lo if hi > lo else 1e-12
    if c.goal == "maximize":
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


@dataclass
class Problem:
    space: Space
    fits: dict                     # response column -> Fit
    criteria: list[Criterion]      # responses and factors
    fixed_levels: dict = field(default_factory=dict)  # categoric factor index -> level index

    def factor_criterion(self, j: int) -> Criterion:
        f = self.space.factors[j]
        for c in self.criteria:
            if c.kind == "factor" and c.name == f.column:
                return c
        return Criterion(f.column, "factor", "in_range", f.low, f.high)

    def bounds_actual(self) -> list[tuple[float, float]]:
        out = []
        for j, f in enumerate(self.space.factors):
            c = self.factor_criterion(j)
            out.append((max(c.lower, f.low) if self.space.mixture else c.lower,
                        min(c.upper, f.high) if self.space.mixture else c.upper))
        return out

    def evaluate(self, actual: np.ndarray) -> dict:
        actual = np.atleast_2d(actual)
        preds, ds, imps = {}, [], []
        viol = np.zeros(len(actual))
        for c in self.criteria:
            if c.kind == "response":
                fit = self.fits.get(c.name)
                if fit is None:
                    continue
                y = fit.predict(actual)
                preds[c.name] = y
            else:
                j = next((i for i, f in enumerate(self.space.factors) if f.column == c.name),
                         None)
                if j is None:
                    continue
                y = actual[:, j]
            if c.goal == "none" or (c.kind == "factor" and c.goal == "in_range"):
                continue
            ds.append(desirability(y, c))
            imps.append(c.importance)
            span = (c.upper - c.lower) or 1.0
            viol += (np.clip(c.lower - y, 0, None) + np.clip(y - c.upper, 0, None)) / span
        D = overall(ds, imps) if ds else np.ones(len(actual))
        return {"pred": preds, "d": ds, "D": D, "viol": viol}


def _random_points(problem: Problem, n: int, rng) -> np.ndarray:
    sp = problem.space
    b = np.array(problem.bounds_actual())
    if not sp.mixture:
        return b[:, 0] + rng.random((n, sp.k)) * (b[:, 1] - b[:, 0])
    pts = []
    tries = 0
    while len(pts) < n and tries < 200:
        tries += 1
        cand = rng.dirichlet(np.ones(sp.k), size=4 * n) * (sp.total - b[:, 0].sum()) + b[:, 0]
        ok = np.all((cand >= b[:, 0] - 1e-12) & (cand <= b[:, 1] + 1e-12), axis=1)
        pts += list(cand[ok])
    if not pts:
        raise ValueError("Could not find points inside the constrained mixture region.")
    return np.array(pts[:n])


def solve(problem: Problem, starts: int = 30, seed: int | None = 0, max_solutions: int = 30,
          design_points: np.ndarray | None = None) -> list[dict]:
    """Maximise D from random starts; returns distinct solutions sorted by desirability."""
    sp = problem.space
    rng = np.random.default_rng(seed)
    cat = [j for j, f in enumerate(sp.factors) if f.categoric]
    num = [j for j in range(sp.k) if j not in cat]
    combos = list(itertools.product(*[[problem.fixed_levels[j]] if j in problem.fixed_levels
                                      else range(len(sp.factors[j].levels)) for j in cat]))
    b_all = np.array(problem.bounds_actual())
    solutions = []
    for combo in combos or [()]:
        def full(xnum):
            x = np.empty(sp.k)
            x[num] = xnum
            for j, lv in zip(cat, combo):
                x[j] = sp.factors[j].levels[lv]
            return x

        if sp.mixture:
            lo, hi = b_all[:, 0], b_all[:, 1]
            free = list(range(sp.k - 1))

            def to_x(u):
                x = np.empty(sp.k)
                x[free] = u
                x[-1] = sp.total - u.sum()
                return x

            def objective(u):
                x = to_x(u)
                pen = float(np.sum(np.clip(lo - x, 0, None) + np.clip(x - hi, 0, None)))
                if pen > 1e-12:
                    return 10 + 100 * pen
                ev = problem.evaluate(x)
                D = float(ev["D"][0])
                return -D if D > 0 else float(ev["viol"][0])

            starts_pts = list(_random_points(problem, starts, rng))
            if design_points is not None:
                starts_pts += list(design_points)
            bounds = list(zip(lo[:-1], hi[:-1]))
            for s in starts_pts:
                u0 = np.asarray(s, float)[:-1]
                res = optimize.minimize(objective, u0, method="Nelder-Mead", bounds=bounds,
                                        options={"xatol": 1e-6, "fatol": 1e-9,
                                                 "maxiter": 400 * sp.k})
                solutions.append(to_x(res.x))
        else:
            lo, hi = b_all[num, 0], b_all[num, 1]

            def objective(xn):
                ev = problem.evaluate(full(xn))
                D = float(ev["D"][0])
                return -D if D > 0 else float(ev["viol"][0])

            starts_pts = list(lo + rng.random((starts, len(num))) * (hi - lo)) if num else []
            if design_points is not None and num:
                for p in design_points:
                    if np.all(p[num] >= lo - 1e-9) and np.all(p[num] <= hi + 1e-9):
                        starts_pts.append(p[num])
            if not num:
                solutions.append(full(np.array([])))
                continue
            for s in starts_pts:
                res = optimize.minimize(objective, np.asarray(s, float), method="Nelder-Mead",
                                        bounds=list(zip(lo, hi)),
                                        options={"xatol": 1e-7, "fatol": 1e-10,
                                                 "maxiter": 300 * max(len(num), 1)})
                solutions.append(full(np.clip(res.x, lo, hi)))
    if not solutions:
        return []
    X = np.array(solutions)
    ev = problem.evaluate(X)
    D = ev["D"]
    order = np.argsort(-D, kind="stable")
    out, seen = [], []
    scale = np.array([(f.high - f.low) or 1.0 for f in sp.factors])
    for i in order:
        key = X[i] / scale
        if any(np.max(np.abs(key - s)) < 2e-3 for s in seen):
            continue
        seen.append(key)
        out.append({"x": X[i], "pred": {k: float(v[i]) for k, v in ev["pred"].items()},
                    "d": [float(d[i]) for d in ev["d"]], "D": float(D[i])})
        if len(out) >= max_solutions:
            break
    return out


def feasible(problem: Problem, actual: np.ndarray) -> np.ndarray:
    """Graphical optimisation: True where every response lies within its limits."""
    ok = np.ones(len(np.atleast_2d(actual)), bool)
    for c in problem.criteria:
        if c.kind != "response" or c.name not in problem.fits:
            continue
        y = problem.fits[c.name].predict(actual)
        ok &= (y >= c.lower) & (y <= c.upper)
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

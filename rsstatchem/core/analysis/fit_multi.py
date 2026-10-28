"""Fitting several datasets at once, as in Origin's NLFit "Multiple Data" options:

* **Independent** – every dataset gets its own fit (one result per dataset).
* **Global Fit** – one fit of all datasets with parameter sharing: a shared parameter has a
  single value for all datasets, the others one value per dataset (Origin's "Share" column).
  Fixed values may differ per dataset (e.g. the inhibitor concentration of an enzyme
  inhibition model).
* **Concatenate** – all data pooled into one dataset.
* **Replica** – one dataset fitted with a sum of ``n`` copies of a function (Origin's
  "Number of Replicas"); offset parameters such as ``y0`` appear once.

The combined model is an ordinary :class:`~rsstatchem.core.analysis.functions.FitFunction`
of two variables (x, dataset index), so a global fit uses exactly the same solver,
constraints, bounds, weighting and covariance as a single fit (:func:`fitting.fit_curve`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from rsstatchem.core.analysis import fitting as fit
from rsstatchem.core.analysis.fitting import FitResult
from rsstatchem.core.analysis.functions import FitFunction

MODES = {"independent": "Independent Fit", "global": "Global Fit (share parameters)",
         "concatenate": "Concatenate Fit", "replica": "Replica Fit"}


@dataclass
class Dataset:
    x: np.ndarray
    y: np.ndarray
    weights: np.ndarray | None = None
    label: str = ""


def as_datasets(data) -> list[Dataset]:
    out = []
    for i, d in enumerate(data):
        if isinstance(d, Dataset):
            out.append(d)
        else:
            x, y, *rest = d
            out.append(Dataset(np.asarray(x, float), np.asarray(y, float),
                               None if not rest or rest[0] is None else np.asarray(rest[0], float),
                               f"Dataset {i + 1}"))
    return out


@dataclass
class MultiFitResult:
    mode: str
    model: FitFunction
    datasets: list[Dataset]
    results: list[FitResult]                 # one per dataset (global: views of the joint fit)
    combined: FitResult | None = None        # the joint fit (global / concatenate)
    shared: tuple[str, ...] = ()
    names: list[str] = field(default_factory=list)   # joint parameter names
    notes: list[str] = field(default_factory=list)

    def table(self) -> list[dict]:
        """Origin's global-fit Parameters table: one row per dataset and parameter."""
        rows = []
        for i, r in enumerate(self.results):
            for j, p in enumerate(r.names):
                rows.append({"dataset": self.datasets[i].label or f"Dataset {i + 1}",
                             "parameter": p, "value": float(r.values[j]),
                             "error": float(r.errors[j]), "shared": p in self.shared,
                             "fixed": not bool(r.free[j])})
        return rows


def _per_dataset(value, k: int) -> list[dict]:
    if value is None:
        return [{} for _ in range(k)]
    if isinstance(value, Mapping):
        return [dict(value) for _ in range(k)]
    vals = list(value)
    if len(vals) != k:
        raise ValueError(f"Expected {k} per-dataset settings, got {len(vals)}.")
    return [dict(v or {}) for v in vals]


def global_function(f: FitFunction, k: int, shared: Sequence[str]) -> tuple[FitFunction, list]:
    """The joint model of ``k`` datasets: x is (n, 2) = (x, dataset index). Returns the
    function and, per dataset, the indices of its parameters in the joint vector."""
    shared = [p for p in f.params if p in set(shared)]
    names: list[str] = []
    maps: list[list[int]] = []
    for i in range(k):
        m = []
        for p in f.params:
            key = p if p in shared else f"{p}_{i + 1}"
            if key not in names:
                names.append(key)
            m.append(names.index(key))
        maps.append(m)

    def split(x):
        x = np.asarray(x, float).reshape(-1, 2)
        return x[:, 0], np.rint(x[:, 1]).astype(int)

    def func(x, *P):
        xs, ds = split(x)
        P = np.asarray(P, float)
        out = np.full(len(xs), np.nan)
        for i in range(k):
            m = ds == i
            if m.any():
                out[m] = f(xs[m], *P[maps[i]])
        return out

    def jac(x, *P):
        xs, ds = split(x)
        P = np.asarray(P, float)
        J = np.zeros((len(xs), len(names)))
        for i in range(k):
            m = ds == i
            if m.any():
                Ji = f.jacobian(xs[m], P[maps[i]])
                for j, col in enumerate(maps[i]):
                    J[np.flatnonzero(m), col] += Ji[:, j]
        return J

    g = FitFunction(f"Global {f.name}", f.category, f.formula, tuple(names), func,
                    description=f"Global fit of {k} datasets with {f.name}; shared: "
                                f"{', '.join(shared) or 'none'}.",
                    jac=jac, nvar=2, meta={"global_of": f.name, "k": k, "shared": shared})
    return g, maps


def global_fit(data, model, shared: Sequence[str] = (), params0=None, fixed=None,
               bounds=None, constraints=None, *, scale_errors: bool = True,
               algorithm: str = "auto") -> MultiFitResult:
    """Global fit with parameter sharing. ``params0`` / ``fixed`` / ``bounds``: one dict for
    all datasets or a list of dicts (one per dataset); shared parameters take the first
    dataset's setting. ``constraints`` use the joint names (``t1``, ``y0_2``…)."""
    f = fit.resolve_model(model)
    ds = as_datasets(data)
    k = len(ds)
    if k < 2:
        raise ValueError("A global fit needs at least two datasets.")
    unknown = set(shared) - set(f.params)
    if unknown:
        raise ValueError(f"Unknown shared parameter(s): {', '.join(sorted(unknown))}")
    g, maps = global_function(f, k, shared)
    p0s, fxs, bds = (_per_dataset(params0, k), _per_dataset(fixed, k),
                     _per_dataset(bounds, k))
    if fixed is None and f.meta.get("fixed"):
        fxs = [{p: v for p, v in f.meta["fixed"].items()} for _ in range(k)]
    # starting values: per-dataset fits (Origin fits each dataset first when no values given)
    start: dict[str, list[float]] = {}
    for i, d in enumerate(ds):
        guess = {}
        need = any(p not in p0s[i] and p not in fxs[i] for p in f.params)
        if need:
            try:
                r = fit.fit_curve(d.x, d.y, f, params0=p0s[i] or None,
                                  fixed={p: v for p, v in fxs[i].items()} or None,
                                  bounds=bds[i] or None, weights=d.weights)
                guess = r.params
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                guess = f.initial_guess(d.x, d.y, d.weights)
        for j, p in enumerate(f.params):
            key = g.params[maps[i][j]]
            v = fxs[i].get(p) if fxs[i].get(p) is not None else p0s[i].get(p, guess.get(p))
            if v is not None:
                start.setdefault(key, []).append(float(v))
    p0 = {key: float(np.median(v)) for key, v in start.items()}
    fixed_g, bounds_g = {}, {}
    for i in range(k):
        for j, p in enumerate(f.params):
            key = g.params[maps[i][j]]
            if p in fxs[i] and key not in fixed_g:
                v = fxs[i][p]
                fixed_g[key] = float(v) if v is not None else p0.get(key)
            if p in bds[i] and key not in bounds_g:
                bounds_g[key] = bds[i][p]
    X = np.concatenate([np.column_stack([d.x, np.full(len(d.x), i)]) for i, d in enumerate(ds)])
    Y = np.concatenate([d.y for d in ds])
    W = None
    if any(d.weights is not None for d in ds):
        W = np.concatenate([d.weights if d.weights is not None else np.ones(len(d.y))
                            for d in ds])
    res = fit.fit_curve(X, Y, g, params0=p0, fixed=fixed_g or {}, bounds=bounds_g or None,
                        weights=W, constraints=constraints, scale_errors=scale_errors,
                        algorithm=algorithm)
    views = [_view(f, res, maps[i], i) for i in range(k)]
    return MultiFitResult("global", f, ds, views, res, tuple(p for p in f.params
                                                            if p in set(shared)),
                          list(g.params), list(res.notes))


def _view(f: FitFunction, res: FitResult, m: list[int], i: int) -> FitResult:
    """Dataset ``i`` of a joint fit as a FitResult (joint covariance, dataset statistics)."""
    sel = np.rint(res.x[:, 1]).astype(int) == i
    x, y = res.x[sel, 0], res.y[sel]
    w = None if res.weights is None else res.weights[sel]
    values = res.values[m]
    cov = res.cov[np.ix_(m, m)]
    free = res.free[m]
    ww = np.ones(len(y)) if w is None else w
    with np.errstate(all="ignore"):
        r = y - f(x, *values)
    rss = float(np.sum(ww * r ** 2))
    ybar = float(np.sum(ww * y) / np.sum(ww))
    errors = np.sqrt(np.maximum(np.diag(cov), 0.0))
    errors[~free] = np.nan
    return FitResult(f, x, y, w, values, errors, cov, free, res.initial[m], res.lower[m],
                     res.upper[m], rss, float(np.sum(ww * (y - ybar) ** 2)),
                     float(np.sum(ww * y ** 2)), res.success, res.message, res.nfev,
                     res.scale_errors, res.weighting, [])


def independent_fit(data, model, params0=None, fixed=None, bounds=None, constraints=None,
                    **kw) -> MultiFitResult:
    f = fit.resolve_model(model)
    ds = as_datasets(data)
    p0s, fxs, bds = (_per_dataset(params0, len(ds)), _per_dataset(fixed, len(ds)),
                     _per_dataset(bounds, len(ds)))
    out = []
    for i, d in enumerate(ds):
        out.append(fit.fit_curve(d.x, d.y, f, params0=p0s[i] or None,
                                 fixed=fxs[i] or (None if fixed is None else {}),
                                 bounds=bds[i] or None, weights=d.weights,
                                 constraints=constraints, **kw))
    return MultiFitResult("independent", f, ds, out, None, (), list(f.params))


def concatenate_fit(data, model, **kw) -> MultiFitResult:
    f = fit.resolve_model(model)
    ds = as_datasets(data)
    x = np.concatenate([d.x for d in ds])
    y = np.concatenate([d.y for d in ds])
    w = None
    if any(d.weights is not None for d in ds):
        w = np.concatenate([d.weights if d.weights is not None else np.ones(len(d.y))
                            for d in ds])
    res = fit.fit_curve(x, y, f, weights=w, **kw)
    return MultiFitResult("concatenate", f, ds, [res], res, tuple(f.params), list(f.params))


# ---------------------------------------------------------------------- replica
OFFSETS = ("y0", "z0", "A1_base")


def replica_function(f: FitFunction, n: int, offsets: Sequence[str] | None = None) \
        -> FitFunction:
    """y = offset + Σᵢ g(x; qᵢ): ``n`` copies of ``f`` sharing its offset parameter(s)."""
    offsets = [p for p in (offsets if offsets is not None else OFFSETS) if p in f.params]
    local = [p for p in f.params if p not in offsets]
    names = list(offsets) + [f"{p}_{i + 1}" for i in range(n) for p in local]
    pos = {p: f.params.index(p) for p in f.params}

    def unpack(P, i):
        q = np.zeros(len(f.params))
        for j, p in enumerate(local):
            q[pos[p]] = P[len(offsets) + i * len(local) + j]
        return q

    def func(x, *P):
        P = np.asarray(P, float)
        y = np.zeros(np.shape(x), float)
        for j, p in enumerate(offsets):
            y = y + P[j]
        for i in range(n):
            y = y + f(x, *unpack(P, i))
        return y

    def jac(x, *P):
        P = np.asarray(P, float)
        x = np.asarray(x, float)
        J = np.zeros((len(x), len(names)))
        J[:, :len(offsets)] = 1.0
        for i in range(n):
            Ji = f.jacobian(x, unpack(P, i))
            for j, p in enumerate(local):
                J[:, len(offsets) + i * len(local) + j] = Ji[:, pos[p]]
        return J

    abs_keys = [nm for nm in names if nm.split("_")[0] in ("w", "wG", "wL", "w1", "w2")]
    canon = None
    if f.canonical is not None and abs_keys:
        def canon(v):
            return {kk: (abs(val) if kk in abs_keys else val) for kk, val in v.items()}
    return FitFunction(f"{n}×{f.name}", "Multiple Peaks", f"{f.formula}  (replica × {n})",
                       tuple(names), func, description=f"{n} replicas of {f.name}.",
                       jac=jac, canonical=canon,
                       meta={"replica_of": f.name, "n": n, "offsets": tuple(offsets),
                             "local": tuple(local)})


def replica_fit(x, y, model, n: int, params0=None, weights=None, **kw) -> FitResult:
    """Replica fit. Without ``params0`` the start comes from a greedy decomposition: fit one
    copy, subtract it, fit the next copy to the residual, …, then the joint refinement."""
    f = fit.resolve_model(model)
    g = replica_function(f, n)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if params0 is None:
        offsets = g.meta["offsets"]
        local = g.meta["local"]
        p0: dict[str, float] = {}
        rem = y.copy()
        for i in range(n):
            try:
                r = fit.fit_curve(x, rem, f, weights=weights,
                                  fixed={o: 0.0 for o in offsets} if i else None)
                vals = r.params
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                vals = f.initial_guess(x, rem)
            if i == 0:
                for o in offsets:
                    p0[o] = vals.get(o, 0.0)
            for p in local:
                p0[f"{p}_{i + 1}"] = vals[p]
            q = [0.0 if p in offsets else vals[p] for p in f.params]
            rem = rem - f(x, *q) - (sum(vals.get(o, 0.0) for o in offsets) if i == 0 else 0)
        params0 = p0
    return fit.fit_curve(x, y, g, params0=params0, weights=weights, **kw)


def total_rss(mr: MultiFitResult) -> tuple[float, int]:
    """(Σ RSS, Σ degrees of freedom) of a multi-dataset fit."""
    if mr.combined is not None:
        return mr.combined.rss, mr.combined.dof
    return float(sum(r.rss for r in mr.results)), int(sum(r.dof for r in mr.results))


def finite(v: float) -> bool:
    return v is not None and math.isfinite(v)

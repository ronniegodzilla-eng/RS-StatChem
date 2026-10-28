"""Design regions with multiple linear constraints (MLC), for process and mixture designs.

A :class:`Region` is the set of coded factor settings an optimal design may use:

* process factors: the box -1 <= x <= +1 (coded) intersected with linear constraints such as
  ``0.2 <= A + 2B <= 1`` given in actual units;
* mixture components: L-pseudo components (0 <= p_j <= (U_j - L_j) / span, sum p = 1) with
  linear constraints on the actual amounts;
* combined designs: a mixture block and a process block side by side;
* categoric factors: level indices 0 .. n-1 (crossed with the continuous region).

Vertices are enumerated combinatorially (every set of ``dim`` linearly independent active
hyperplanes is solved and kept when feasible, McLean & Anderson 1966; equivalent to Piepel's
CONVRT for the small dimensions of a designed experiment); edge centres, thirds of edges,
constraint-plane (facet) centroids, axial check blends ("interior" points halfway between a
vertex and the centroid) and the overall centroid complete the candidate set. Points can be
sampled uniformly from the region (rejection, falling back to hit-and-run) for the I-optimal
moment matrix, and :meth:`Region.line` gives the feasible moves of one coordinate for
coordinate exchange (a Cox-direction move for a mixture component).

Constraint text uses the factor letters (A, B, C, ... without I): ``A + B <= 0.65``,
``0.2 <= A + 2B <= 1``, ``A - 2*C >= 0``, ``-1 <= 3A - B``.
"""
from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass, field

import numpy as np

TOL = 1e-7
LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ" + "abcdefghjklmnopqrstuvwxyz"


# ---------------------------------------------------------------------------- parsing
_TERM = re.compile(r"([+-])?\s*(\d+(?:\.\d*)?|\.\d+)?\s*\*?\s*([A-HJ-Za-hj-z])")
_NUM = re.compile(r"^[+-]?(\d+(?:\.\d*)?|\.\d+)$")


def _parse_linear(expr: str, k: int) -> np.ndarray | None:
    """Coefficients of a linear expression in factor letters, or None for a pure number."""
    s = expr.replace(" ", "").replace(",", ".")
    if not s:
        raise ValueError("Empty side of a constraint.")
    if _NUM.match(s):
        return None
    coefs = np.zeros(k)
    pos = 0
    while pos < len(s):
        m = _TERM.match(s, pos)
        if not m or m.start() != pos:
            raise ValueError(f"Cannot read the constraint term near '{s[pos:]}'.")
        sign = -1.0 if m.group(1) == "-" else 1.0
        if pos > 0 and not m.group(1):
            raise ValueError(f"Missing + or - before '{s[pos:]}'.")
        c = float(m.group(2)) if m.group(2) else 1.0
        j = LETTERS.index(m.group(3))
        if j >= k:
            raise ValueError(f"There is no factor {m.group(3)} (only {k} factors).")
        coefs[j] += sign * c
        pos = m.end()
    return coefs


def parse_constraint(text: str, k: int) -> dict:
    """``"0.2 <= A + 2B <= 1"`` -> {"coefs": [...], "low": 0.2, "high": 1.0, "text": ...}
    (actual units; ``low`` / ``high`` may be None)."""
    t = text.strip().replace("≤", "<=").replace("≥", ">=").replace("=<", "<=") \
        .replace("=>", ">=")
    parts = re.split(r"(<=|>=|<|>)", t)
    if len(parts) not in (3, 5):
        raise ValueError(f"Write a constraint like '0.2 <= A + 2B <= 1' (got '{text}').")
    sides = [p for i, p in enumerate(parts) if i % 2 == 0]
    ops = [p for i, p in enumerate(parts) if i % 2 == 1]
    if len(ops) == 2 and ops[0][0] != ops[1][0]:
        raise ValueError("A double constraint must use the same direction twice.")
    parsed = [_parse_linear(s, k) for s in sides]
    if len(sides) == 3:
        if parsed[0] is not None or parsed[2] is not None or parsed[1] is None:
            raise ValueError("A double constraint needs numbers on the outside.")
        lo, hi = float(sides[0].replace(",", ".")), float(sides[2].replace(",", "."))
        coefs = parsed[1]
        if ops[0][0] == ">":
            lo, hi = hi, lo
    else:
        left, right = parsed
        if (left is None) == (right is None):
            raise ValueError("A constraint compares the factors with a number.")
        if left is not None:
            coefs, val = left, float(sides[1].replace(",", "."))
            lo, hi = (None, val) if ops[0][0] == "<" else (val, None)
        else:
            coefs, val = right, float(sides[0].replace(",", "."))
            lo, hi = (val, None) if ops[0][0] == "<" else (None, val)
    if lo is not None and hi is not None and lo > hi:
        raise ValueError(f"The lower limit exceeds the upper limit in '{text}'.")
    if not np.any(coefs):
        raise ValueError(f"The constraint '{text}' has no factors.")
    return {"coefs": [float(c) for c in coefs], "low": lo, "high": hi,
            "text": format_constraint(coefs, lo, hi)}


def format_constraint(coefs, lo, hi) -> str:
    terms = []
    for j, c in enumerate(coefs):
        if abs(c) < 1e-15:
            continue
        mag = abs(c)
        t = (f"{mag:g}" if not math.isclose(mag, 1) else "") + LETTERS[j]
        terms.append(("- " if c < 0 else ("+ " if terms else "")) + t)
    expr = " ".join(terms)
    if lo is not None and hi is not None:
        return f"{lo:g} <= {expr} <= {hi:g}"
    if lo is not None:
        return f"{expr} >= {lo:g}"
    return f"{expr} <= {hi:g}"


# ---------------------------------------------------------------------------- region
@dataclass
class Region:
    """Feasible coded settings. ``lo`` / ``hi`` bound every continuous column (categoric
    columns are ignored there); ``A x`` lies between ``clo`` and ``chi`` (coded constraint
    rows); the mixture columns ``mix`` sum to 1 (L-pseudo components)."""
    k: int
    lo: np.ndarray
    hi: np.ndarray
    mix: list[int] = field(default_factory=list)
    A: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    clo: np.ndarray = field(default_factory=lambda: np.zeros(0))
    chi: np.ndarray = field(default_factory=lambda: np.zeros(0))
    nlevels: list[int] = field(default_factory=list)  # 0 = continuous
    _verts: np.ndarray | None = None
    _active: list | None = None

    def __post_init__(self):
        self.lo = np.asarray(self.lo, float)
        self.hi = np.asarray(self.hi, float)
        if not self.nlevels:
            self.nlevels = [0] * self.k
        if self.A.size == 0:
            self.A = np.zeros((0, self.k))
            self.clo = np.zeros(0)
            self.chi = np.zeros(0)

    # ------------------------------------------------------------------ structure
    @property
    def cont(self) -> list[int]:
        return [j for j in range(self.k) if not self.nlevels[j]]

    @property
    def cat(self) -> list[int]:
        return [j for j in range(self.k) if self.nlevels[j]]

    @property
    def proc(self) -> list[int]:
        return [j for j in self.cont if j not in self.mix]

    @property
    def constrained(self) -> bool:
        return len(self.A) > 0

    @property
    def dim(self) -> int:
        return len(self.cont) - (1 if self.mix else 0)

    def _hyperplanes(self):
        """(normal over continuous columns, value) for every bound and constraint side."""
        cont = self.cont
        rows = []
        for i, j in enumerate(cont):
            e = np.zeros(len(cont))
            e[i] = 1
            if math.isfinite(self.lo[j]):
                rows.append((e, self.lo[j], -1))
            if math.isfinite(self.hi[j]):
                rows.append((e, self.hi[j], +1))
        for r in range(len(self.A)):
            a = self.A[r, cont]
            if math.isfinite(self.clo[r]):
                rows.append((a, self.clo[r], -1))
            if math.isfinite(self.chi[r]):
                rows.append((a, self.chi[r], +1))
        return rows

    def feasible(self, X, tol: float = 1e-7) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        ok = np.ones(len(X), bool)
        for j in self.cont:
            ok &= (X[:, j] >= self.lo[j] - tol) & (X[:, j] <= self.hi[j] + tol)
        for j in self.cat:
            ok &= (X[:, j] >= -tol) & (X[:, j] <= self.nlevels[j] - 1 + tol)
        if self.mix:
            ok &= np.abs(X[:, self.mix].sum(axis=1) - 1) < 1e-6
        if len(self.A):
            v = X @ self.A.T
            ok &= np.all((v >= self.clo - tol) & (v <= self.chi + tol), axis=1)
        return ok

    # ------------------------------------------------------------------ vertices
    def vertices(self) -> np.ndarray:
        """Vertices of the continuous part (columns ``cont``; McLean-Anderson / CONVRT by
        enumeration of active hyperplane sets)."""
        if self._verts is None:
            self._enumerate()
        return self._verts

    def _enumerate(self, max_combos: int = 400_000) -> None:
        cont = self.cont
        nc = len(cont)
        if nc == 0:
            self._verts, self._active = np.zeros((1, 0)), [frozenset()]
            return
        planes = self._hyperplanes()
        eq = []
        if self.mix:
            e = np.zeros(nc)
            e[[cont.index(j) for j in self.mix]] = 1
            eq.append((e, 1.0))
        need = nc - len(eq)
        if math.comb(len(planes), need) > max_combos:
            raise ValueError("Too many constraints to enumerate the vertices; remove some.")
        N = np.array([p[0] for p in planes])
        b = np.array([p[1] for p in planes])
        found: list[np.ndarray] = []
        for combo in itertools.combinations(range(len(planes)), need):
            M = np.vstack([e for e, _ in eq] + [N[list(combo)]]) if eq else N[list(combo)]
            rhs = np.concatenate([[v for _, v in eq], b[list(combo)]]) if eq else \
                b[list(combo)]
            if abs(np.linalg.det(M)) < 1e-12:
                continue
            x = np.linalg.solve(M, rhs)
            full = np.zeros(self.k)
            full[cont] = x
            if not self.feasible(full[None], tol=1e-9)[0] and not self._cont_feasible(x):
                continue
            if not any(np.allclose(x, v, atol=1e-8) for v in found):
                found.append(x)
        if not found:
            raise ValueError("The constraints leave no feasible region.")
        V = np.array(found)
        V[np.abs(V) < 1e-12] = 0.0
        act = []
        for x in V:
            act.append(frozenset(i for i, (n, v, _) in enumerate(planes)
                                 if abs(n @ x - v) < 1e-7))
        order = np.lexsort(V.T[::-1])
        self._verts = V[order]
        self._active = [act[i] for i in order]

    def _cont_feasible(self, x: np.ndarray) -> bool:
        cont = self.cont
        full = np.zeros(self.k)
        full[cont] = x
        for j in self.cat:
            full[j] = 0
        return bool(self.feasible(full[None], tol=1e-9)[0])

    def centroid(self) -> np.ndarray:
        return self.vertices().mean(axis=0)

    def edges(self) -> list[tuple[int, int]]:
        V = self.vertices()
        planes = self._hyperplanes()
        d = self.dim
        out = []
        if d < 1:
            return out
        if d == 1:
            return [(0, 1)] if len(V) == 2 else []
        for a, b in itertools.combinations(range(len(V)), 2):
            common = self._active[a] & self._active[b]
            if len(common) < d - 1:
                continue
            rows = [planes[i][0] for i in common]
            if self.mix:
                e = np.zeros(len(self.cont))
                e[[self.cont.index(j) for j in self.mix]] = 1
                rows.append(e)
            if np.linalg.matrix_rank(np.array(rows), tol=1e-9) >= len(self.cont) - 1:
                out.append((a, b))
        return out

    def facets(self) -> list[np.ndarray]:
        """Vertex sets of the facets (constraint planes) with at least ``dim`` vertices."""
        V = self.vertices()
        planes = self._hyperplanes()
        out, seen = [], set()
        for i in range(len(planes)):
            on = tuple(v for v in range(len(V)) if i in self._active[v])
            if len(on) >= max(self.dim, 2) and on not in seen and len(on) < len(V):
                seen.add(on)
                out.append(np.array(on))
        return out

    # ------------------------------------------------------------------ candidates
    def candidates(self, kinds=("vertex", "edge", "plane", "interior", "centroid"),
                   max_points: int = 20000) -> tuple[np.ndarray, np.ndarray]:
        """Candidate points (coded, all k columns, crossed with every categoric level
        combination) and their space types (4 vertex, 3 edge, 5 plane, 6 interior, 7 centroid,
        9 third of edge)."""
        V = self.vertices()
        cont = self.cont
        pts, types = [], []

        def add(x, t):
            if not any(np.allclose(x, p, atol=1e-8) for p in pts):
                pts.append(np.asarray(x, float))
                types.append(t)

        if "vertex" in kinds:
            for v in V:
                add(v, 4)
        edges = self.edges() if ({"edge", "thirds"} & set(kinds)) else []
        if "edge" in kinds:
            for a, b in edges:
                add((V[a] + V[b]) / 2, 3)
        if "thirds" in kinds:
            for a, b in edges:
                add((2 * V[a] + V[b]) / 3, 9)
                add((V[a] + 2 * V[b]) / 3, 9)
        if "plane" in kinds and self.dim >= 3:
            for f in self.facets():
                add(V[f].mean(axis=0), 5)
        c = V.mean(axis=0)
        if "interior" in kinds:
            for v in V:
                add((v + c) / 2, 6)
        if "centroid" in kinds:
            add(c, 7)
        P = np.array(pts) if pts else np.zeros((0, len(cont)))
        T = np.array(types, int)
        if self.cat:
            combos = np.array(list(itertools.product(*[range(self.nlevels[j])
                                                       for j in self.cat])), float)
        else:
            combos = np.zeros((1, 0))
        if len(P) * len(combos) > max_points:
            raise ValueError("The candidate set is too large; use coordinate exchange.")
        out = np.zeros((len(P) * len(combos), self.k))
        tt = np.tile(T, len(combos))
        for i, combo in enumerate(combos):
            out[i * len(P):(i + 1) * len(P)][:, cont] = P
            if self.cat:
                out[i * len(P):(i + 1) * len(P)][:, self.cat] = combo
        return out, tt

    # ------------------------------------------------------------------ sampling
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """About uniform points of the region (all k columns; categoric levels uniform)."""
        cont, k = self.cont, self.k
        out = []
        got = 0
        for _ in range(60):
            m = max(4 * n, 256)
            X = np.zeros((m, k))
            if self.mix:
                span = np.array([self.hi[j] for j in self.mix])
                D = rng.dirichlet(np.ones(len(self.mix)), m)
                X[:, self.mix] = D
                del span
            for j in self.proc:
                X[:, j] = rng.uniform(self.lo[j], self.hi[j], m)
            for j in self.cat:
                X[:, j] = rng.integers(0, self.nlevels[j], m)
            X = X[self.feasible(X)]
            out.append(X)
            got += len(X)
            if got >= n:
                break
        if got >= n:
            return np.vstack(out)[:n]
        return self._hit_and_run(n, rng)

    def _hit_and_run(self, n: int, rng: np.random.Generator, thin: int = 5) -> np.ndarray:
        cont = self.cont
        x = self.centroid().copy()
        nc = len(cont)
        if self.mix:
            e = np.zeros((1, nc))
            e[0, [cont.index(j) for j in self.mix]] = 1
            _, _, vt = np.linalg.svd(e)
            basis = vt[1:].T  # null space of the sum constraint
        else:
            basis = np.eye(nc)
        planes = self._hyperplanes()
        out = np.zeros((n, self.k))
        for i in range(n * thin):
            d = basis @ rng.normal(size=basis.shape[1])
            d /= np.linalg.norm(d) or 1
            tmin, tmax = -np.inf, np.inf
            for nrm, v, side in planes:
                a = nrm @ d
                if abs(a) < 1e-14:
                    continue
                t = (v - nrm @ x) / a
                # side +1: n.x <= v ; side -1: n.x >= v
                if (side > 0) == (a > 0):
                    tmax = min(tmax, t)
                else:
                    tmin = max(tmin, t)
            if not (math.isfinite(tmin) and math.isfinite(tmax)) or tmax <= tmin:
                continue
            x = x + rng.uniform(tmin, tmax) * d
            if i % thin == thin - 1:
                out[i // thin, cont] = x
        for j in self.cat:
            out[:, j] = rng.integers(0, self.nlevels[j], n)
        return out

    # ------------------------------------------------------------------ coordinate moves
    def line(self, x: np.ndarray, j: int, n_grid: int = 11) -> np.ndarray:
        """Feasible points reached from ``x`` by changing coordinate ``j`` (a grid along the
        feasible segment, endpoints included). For a mixture component the other
        components change proportionally (Cox direction), keeping the sum."""
        x = np.asarray(x, float)
        if self.nlevels[j]:
            P = np.repeat(x[None], self.nlevels[j], axis=0)
            P[:, j] = np.arange(self.nlevels[j])
            return P[self.feasible(P)]
        if j in self.mix:
            others = [i for i in self.mix if i != j]
            rest = 1 - x[j]
            a = np.zeros(self.k)
            b = np.zeros(self.k)
            if rest > 1e-12:
                a[others] = x[others] / rest
                b[others] = -x[others] / rest
            else:
                a[others] = 1 / len(others)
                b[others] = -1 / len(others)
            b[j] = 1
            base = x.copy()
            base[self.mix] = a[self.mix]
            direction = b
        else:
            base = x.copy()
            base[j] = 0.0
            direction = np.zeros(self.k)
            direction[j] = 1.0
        # x(t) = base + t * direction ; find the feasible t interval
        tmin, tmax = -np.inf, np.inf
        cols = self.cont
        for i in cols:
            di = direction[i]
            if abs(di) < 1e-14:
                if base[i] < self.lo[i] - 1e-9 or base[i] > self.hi[i] + 1e-9:
                    return np.zeros((0, self.k))
                continue
            t1, t2 = (self.lo[i] - base[i]) / di, (self.hi[i] - base[i]) / di
            tmin, tmax = max(tmin, min(t1, t2)), min(tmax, max(t1, t2))
        for r in range(len(self.A)):
            a_d = self.A[r] @ direction
            a_b = self.A[r] @ base
            if abs(a_d) < 1e-14:
                if a_b < self.clo[r] - 1e-9 or a_b > self.chi[r] + 1e-9:
                    return np.zeros((0, self.k))
                continue
            t1, t2 = (self.clo[r] - a_b) / a_d, (self.chi[r] - a_b) / a_d
            lo_t, hi_t = min(t1, t2), max(t1, t2)
            tmin, tmax = max(tmin, lo_t), min(tmax, hi_t)
        if not (math.isfinite(tmin) and math.isfinite(tmax)) or tmax < tmin - 1e-12:
            return np.zeros((0, self.k))
        ts = np.linspace(tmin, tmax, max(n_grid, 2)) if tmax - tmin > 1e-12 \
            else np.array([tmin])
        P = base[None] + ts[:, None] * direction[None]
        P[np.abs(P) < 1e-13] = 0.0
        return P[self.feasible(P, tol=1e-7)]


# ---------------------------------------------------------------------------- constructors
def _coded_constraints(constraints, k, offset, scale):
    """Constraints on actual values (actual = offset + scale * coded) in coded units."""
    rows, lo, hi = [], [], []
    for c in constraints or []:
        a = np.asarray(c["coefs"], float)
        if len(a) < k:
            a = np.concatenate([a, np.zeros(k - len(a))])
        a = a[:k]
        shift = float(a @ offset)
        rows.append(a * scale)
        lo.append(c["low"] - shift if c.get("low") is not None else -np.inf)
        hi.append(c["high"] - shift if c.get("high") is not None else np.inf)
    if not rows:
        return np.zeros((0, k)), np.zeros(0), np.zeros(0)
    return np.array(rows), np.array(lo), np.array(hi)


def process_region(lows, highs, constraints=None, nlevels=None) -> Region:
    """Coded region of process factors (box -1..1) with constraints on the actual values."""
    lows, highs = np.asarray(lows, float), np.asarray(highs, float)
    k = len(lows)
    nlevels = list(nlevels or [0] * k)
    center, half = (lows + highs) / 2, (highs - lows) / 2
    half = np.where(half == 0, 1.0, half)
    for j in range(k):  # categoric columns are not in the linear constraints
        if nlevels[j]:
            center[j], half[j] = 0.0, 0.0
    A, clo, chi = _coded_constraints(constraints, k, center, half)
    lo = np.where(np.array(nlevels) > 0, 0, -1.0)
    hi = np.where(np.array(nlevels) > 0, np.array(nlevels) - 1, 1.0)
    return Region(k, lo.astype(float), hi.astype(float), [], A, clo, chi, nlevels)


def mixture_region(lower, upper, total: float = 1.0, constraints=None) -> Region:
    """L-pseudo coded mixture region: p = (x - L) / (T - sum L)."""
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    q = len(lower)
    span = total - lower.sum()
    if span <= 0:
        raise ValueError("The lower limits add up to the mixture total or more.")
    A, clo, chi = _coded_constraints(constraints, q, lower, np.full(q, span))
    hi = np.minimum((upper - lower) / span, 1.0)
    return Region(q, np.zeros(q), hi, list(range(q)), A, clo, chi, [0] * q)


def combined_region(lower, upper, total, plows, phighs, constraints=None,
                    proc_constraints=None, nlevels=None) -> Region:
    """Mixture components (first q columns, L-pseudo) beside process factors (coded -1..1).
    ``constraints`` act on the components, ``proc_constraints`` on the process factors."""
    mr = mixture_region(lower, upper, total, constraints)
    pr = process_region(plows, phighs, proc_constraints, nlevels)
    q, kp = mr.k, pr.k
    k = q + kp
    A = np.zeros((len(mr.A) + len(pr.A), k))
    A[:len(mr.A), :q] = mr.A
    A[len(mr.A):, q:] = pr.A
    return Region(k, np.concatenate([mr.lo, pr.lo]), np.concatenate([mr.hi, pr.hi]),
                  list(range(q)), A, np.concatenate([mr.clo, pr.clo]),
                  np.concatenate([mr.chi, pr.chi]), [0] * q + list(pr.nlevels))


def describe(constraints) -> list[str]:
    return [c.get("text") or format_constraint(c["coefs"], c.get("low"), c.get("high"))
            for c in constraints or []]


__all__ = ["Region", "parse_constraint", "format_constraint", "process_region",
           "mixture_region", "combined_region", "describe"]

"""Optimal (custom) design engine: D-, I- and A-optimal exact designs.

Criteria (minimised internally):

* **D**: maximise det(X'X) (the joint confidence region of the coefficients);
* **A**: minimise trace((X'X)^-1) over the model coefficients (average coefficient variance);
* **I** (also called IV or Q): minimise the average prediction variance over the design region,
  trace((X'X)^-1 W) with the moment matrix W = mean of f(x) f(x)' over the region (Gauss-
  Legendre quadrature for an unconstrained box, about-uniform points otherwise). Stat-Ease
  recommends I for response surface designs, D for factorial screening.

Exchange algorithms:

* **point exchange** (Fedorov 1972; Cook & Nachtsheim 1980, modified Fedorov): every design
  row is swapped with the best candidate point;
* **coordinate exchange** (Meyer & Nachtsheim 1995): every coordinate of every row is moved
  along a grid of its feasible segment (a Cox-direction move for a mixture component), so no
  candidate list is needed and constraints are honoured exactly;
* **best**: half of the random starts with each algorithm, keeping the best design.

Every swap is scored with the rank-two update of (X'X)^-1 (Sherman-Morrison-Woodbury), so all
candidates are evaluated at once. Runs can be forced into the design (existing runs when a
design is augmented) and the runs can be split into fixed blocks (block effects in the model
matrix, effect coded). Designs may repeat points (exact designs, e.g. {-1, 0, 0, 1} for a
one-factor quadratic I-optimal design) unless ``allow_repeats`` is False.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from rsstatchem.core.doe.constraints import Region

CRITERIA = {"D": "D-optimal", "I": "I-optimal", "A": "A-optimal"}
METHODS = {"point": "Point exchange", "coordinate": "Coordinate exchange",
           "best": "Best (point + coordinate)"}


# ---------------------------------------------------------------------------- matrices
def model_function(terms, categoric=None, nlevels=None, intercept: bool = True):
    """``f(coded) -> X`` for model terms (tuples of factor indices, see core.doe.model)."""
    from rsstatchem.core.doe import model as mdl

    terms = list(terms)
    cat = list(categoric) if categoric is not None else None
    nl = list(nlevels) if nlevels is not None else None

    def f(coded):
        coded = np.atleast_2d(np.asarray(coded, float))
        return mdl.raw_matrix(coded, terms, intercept=intercept,
                              categoric=cat if cat and any(cat) else None, nlevels=nl)
    return f


def block_columns(blocks) -> np.ndarray:
    """Effect-coded block columns (b - 1 columns, the last block -1)."""
    blocks = np.asarray(blocks)
    levels = sorted(set(blocks.tolist()))
    if len(levels) < 2:
        return np.zeros((len(blocks), 0))
    idx = np.array([levels.index(b) for b in blocks])
    E = np.zeros((len(blocks), len(levels) - 1))
    for j in range(len(levels) - 1):
        E[:, j] = (idx == j).astype(float) - (idx == len(levels) - 1).astype(float)
    return E


def moment_matrix(fmat, region: Region | None, k: int, rng=None, n_points: int = 6000
                  ) -> np.ndarray:
    """Region moments W = mean f(x) f(x)' (I criterion)."""
    rng = rng or np.random.default_rng(12345)
    if region is None or (not region.constrained and not region.mix):
        nlev = region.nlevels if region is not None else [0] * k
        cont = [j for j in range(k) if not nlev[j]]
        npt = 5 if len(cont) <= 4 else (4 if len(cont) <= 6 else 0)
        if npt and (npt ** len(cont)) * max(1, int(np.prod([n for n in nlev if n] or [1]))) \
                <= 200_000:
            g, w = np.polynomial.legendre.leggauss(npt)
            w = w / 2
            axes, weights = [], []
            for j in range(k):
                if nlev[j]:
                    axes.append(np.arange(nlev[j], dtype=float))
                    weights.append(np.full(nlev[j], 1.0 / nlev[j]))
                else:
                    axes.append(g)
                    weights.append(w)
            P = np.array(list(itertools.product(*axes)), float) if k else np.zeros((1, 0))
            Wt = np.prod(np.array(list(itertools.product(*weights)), float), axis=1) if k \
                else np.ones(1)
            F = fmat(P)
            return (F * Wt[:, None]).T @ F
        if region is None:
            P = rng.uniform(-1, 1, (n_points, k))
        else:
            P = region.sample(n_points, rng)
    else:
        P = region.sample(n_points, rng)
    F = fmat(P)
    return F.T @ F / len(F)


def criterion_value(X: np.ndarray, criterion: str, W: np.ndarray | None = None,
                    n_model: int | None = None) -> float:
    """D: log det(X'X); A: trace of the model part of (X'X)^-1; I: trace((X'X)^-1 W)."""
    M = X.T @ X
    if criterion == "D":
        sign, ld = np.linalg.slogdet(M)
        return float(ld) if sign > 0 else -np.inf
    try:
        inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return np.inf
    if np.linalg.cond(M) > 1e12:
        return np.inf
    p = n_model if n_model is not None else M.shape[0]
    if criterion == "A":
        return float(np.trace(inv[:p, :p]))
    Wf = np.zeros_like(M)
    Wf[:W.shape[0], :W.shape[1]] = W
    return float(np.trace(inv @ Wf))


# ---------------------------------------------------------------------------- engine
@dataclass
class OptimalResult:
    coded: np.ndarray            # all rows (forced first, then the new runs)
    blocks: np.ndarray | None
    criterion: str
    method: str
    value: float                 # D: log det, A: trace, I: average prediction variance
    log_det: float
    n_forced: int = 0
    starts: int = 0
    history: list = field(default_factory=list)  # best value of each start


class _State:
    """Design rows, their model rows and (X'X)^-1 with a ridge for singular starts."""

    def __init__(self, F, criterion, Wf, ridge):
        self.F = F
        self.criterion = criterion
        self.Wf = Wf
        self.ridge = ridge
        self.update()

    def update(self):
        p = self.F.shape[1]
        M = self.F.T @ self.F + self.ridge * np.eye(p)
        self.Minv = np.linalg.pinv(M)
        self.G = self.Minv @ self.Wf @ self.Minv if self.criterion != "D" else None

    def gains(self, U: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Improvement (larger is better) of replacing the row ``v`` by each row of U."""
        Minv = self.Minv
        UM = U @ Minv
        a_uu = np.einsum("ij,ij->i", UM, U)
        a_uv = UM @ v
        a_vv = float(v @ Minv @ v)
        ratio = (1 + a_uu) * (1 - a_vv) + a_uv ** 2   # det(M') / det(M)
        if self.criterion == "D":
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(ratio > 1e-12, np.log(np.maximum(ratio, 1e-300)), -np.inf)
        G = self.G
        UG = U @ G
        g_uu = np.einsum("ij,ij->i", UG, U)
        g_uv = UG @ v
        g_vv = float(v @ G @ v)
        s11, s12, s22 = 1 + a_uu, a_uv, -1 + a_vv
        det = s11 * s22 - s12 ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            red = (s22 * g_uu - 2 * s12 * g_uv + s11 * g_vv) / det
        red = np.where(ratio > 1e-10, red, -np.inf)  # reduction of the trace
        return red


def _weights(criterion, W, p_model, p_total):
    Wf = np.zeros((p_total, p_total))
    if criterion == "A":
        Wf[:p_model, :p_model] = np.eye(p_model)
    elif criterion == "I":
        Wf[:p_model, :p_model] = W
    return Wf


def optimal_design(fmat, n: int, region: Region | None = None,
                   candidates: np.ndarray | None = None, criterion: str = "D",
                   method: str = "point", starts: int | None = None, rng=None,
                   forced: np.ndarray | None = None, forced_blocks=None,
                   block_sizes: list[int] | None = None, n_grid: int = 11,
                   allow_repeats: bool = True, W: np.ndarray | None = None,
                   max_pass: int = 60, initial: np.ndarray | None = None,
                   block_start: int | None = None) -> OptimalResult:
    """``n`` new runs that optimise ``criterion`` for the model ``fmat`` (coded -> X).

    ``candidates`` (point exchange) or ``region`` (coordinate exchange; also the source of
    candidates and of the I moment matrix). ``forced`` rows stay in the design;
    ``block_sizes`` splits the new runs into consecutive blocks (``forced_blocks`` labels the
    forced rows; new block labels follow the largest forced label, or ``block_start`` + 1,
    2, ...)."""
    criterion = criterion.upper()
    if criterion not in CRITERIA:
        raise ValueError(f"Unknown optimality criterion {criterion!r}.")
    if method not in METHODS:
        raise ValueError(f"Unknown exchange method {method!r}.")
    rng = rng if rng is not None else np.random.default_rng()
    forced = np.zeros((0, region.k if region is not None else candidates.shape[1])) \
        if forced is None else np.atleast_2d(np.asarray(forced, float))
    k = forced.shape[1]
    if candidates is None and region is not None and method != "coordinate":
        candidates, _ = region.candidates(("vertex", "edge", "thirds", "plane", "interior",
                                           "centroid"))
        if region.cont and not region.constrained and not region.mix and len(region.cont) <= 6:
            grid = np.array(list(itertools.product(*[
                (np.arange(region.nlevels[j], dtype=float) if region.nlevels[j]
                 else np.array([-1.0, 0.0, 1.0])) for j in range(k)])), float)
            candidates = np.unique(np.vstack([candidates, grid]).round(10), axis=0)
    if method in ("coordinate", "best") and region is None:
        method = "point"
    if method == "point" and candidates is None:
        raise ValueError("Point exchange needs a candidate set.")
    # blocks
    nf = len(forced)
    blocks = None
    if block_sizes and (len(block_sizes) > 1 or forced_blocks is not None):
        if sum(block_sizes) != n:
            raise ValueError("The block sizes must add up to the number of runs.")
        start = block_start if block_start is not None else (
            int(np.max(forced_blocks)) if forced_blocks is not None and nf else 0)
        new_b = np.concatenate([np.full(s, start + 1 + i) for i, s in enumerate(block_sizes)])
        fb = np.asarray(forced_blocks, float) if forced_blocks is not None and nf \
            else np.zeros(0)
        blocks = np.concatenate([fb, new_b])
    elif forced_blocks is not None and nf and len(set(np.asarray(forced_blocks).tolist())) > 1:
        blocks = np.concatenate([np.asarray(forced_blocks, float),
                                 np.full(n, float(np.max(forced_blocks)))])
    B = block_columns(blocks) if blocks is not None else np.zeros((nf + n, 0))
    p_model = fmat(np.zeros((1, k))).shape[1]
    p = p_model + B.shape[1]
    if nf + n < p:
        raise ValueError(f"The model has {p} terms (with blocks); at least {p} runs are "
                         "needed.")
    if criterion == "I" and W is None:
        W = moment_matrix(fmat, region, k)
    Wf = _weights(criterion, W, p_model, p)

    def full_rows(X, b_rows):
        F = fmat(X)
        return np.hstack([F, b_rows]) if b_rows.shape[1] else F

    Ff = full_rows(forced, B[:nf]) if nf else np.zeros((0, p))
    Bn = B[nf:]
    if candidates is not None:
        Fc = fmat(candidates)
    n_starts = starts or (8 if method == "point" else 6)
    methods = [method] * n_starts
    if method == "best":
        methods = ["point" if i % 2 == 0 else "coordinate" for i in range(max(n_starts, 2))]
        if candidates is None:
            methods = ["coordinate"] * len(methods)
    best = None
    history = []
    for m in methods:
        D, value = _one_start(m, fmat, n, region, candidates, Fc if candidates is not None
                              else None, Ff, Bn, criterion, Wf, rng, n_grid, allow_repeats,
                              max_pass, initial if initial is not None and best is None
                              else None, p_model)
        history.append(value)
        if D is None:
            continue
        better = best is None or (value > best[1] + 1e-9 if criterion == "D"
                                  else value < best[1] - 1e-12)
        if better:
            best = (D, value)
    if best is None or not np.isfinite(best[1]):
        raise ValueError("No non-singular design found for this model: add runs or relax "
                         "the constraints.")
    D = np.vstack([forced, best[0]]) if nf else best[0]
    X = full_rows(D, B)
    sign, ld = np.linalg.slogdet(X.T @ X)
    return OptimalResult(D, blocks, criterion, method, float(best[1]),
                         float(ld) if sign > 0 else -np.inf, nf, len(methods), history)


def _start_rows(n, region, candidates, rng):
    if candidates is not None and (region is None or rng.random() < 0.5):
        return candidates[rng.integers(0, len(candidates), n)].copy()
    return region.sample(n, rng)


def _one_start(method, fmat, n, region, cand, Fc, Ff, Bn, criterion, Wf, rng, n_grid,
               allow_repeats, max_pass, initial, p_model):
    p = Wf.shape[0]
    if method == "point":
        if initial is not None:
            dists = ((initial[:, None, :] - cand[None, :, :]) ** 2).sum(axis=2)
            idx = np.argmin(dists, axis=1)
        else:
            idx = rng.choice(len(cand), n, replace=allow_repeats or n > len(cand))
        D = cand[idx].copy()
    else:
        D = initial.copy() if initial is not None else _start_rows(n, region, cand, rng)
    Fn = np.hstack([fmat(D), Bn]) if Bn.shape[1] else fmat(D)
    st = _State(np.vstack([Ff, Fn]), criterion, Wf, 1e-6)
    nf = len(Ff)
    for _pass in range(max_pass):
        improved = False
        for i in rng.permutation(n):
            v = st.F[nf + i]
            brow = Bn[i] if Bn.shape[1] else np.zeros(0)
            if method == "point":
                U = np.hstack([Fc, np.repeat(brow[None], len(Fc), axis=0)]) if len(brow) \
                    else Fc
                g = st.gains(U, v)
                if not allow_repeats:
                    used = set(np.flatnonzero(np.all(np.isclose(cand[:, None, :],
                                                                D[None, :, :]), axis=2)
                                              .any(axis=1)))
                    for u in used:
                        g[u] = -np.inf
                j = int(np.argmax(g))
                if g[j] > 1e-9 * max(1.0, abs(g[j])) and g[j] > 1e-10:
                    D[i] = cand[j]
                    st.F[nf + i] = U[j]
                    st.update()
                    improved = True
            else:
                for c in rng.permutation(region.k):
                    P = region.line(D[i], c, n_grid)
                    if not len(P):
                        continue
                    FP = fmat(P)
                    U = np.hstack([FP, np.repeat(brow[None], len(P), axis=0)]) if len(brow) \
                        else FP
                    g = st.gains(U, st.F[nf + i])
                    j = int(np.argmax(g))
                    if g[j] > 1e-10:
                        D[i] = P[j]
                        st.F[nf + i] = U[j]
                        st.update()
                        improved = True
        if not improved:
            break
    X = st.F
    value = criterion_value(X, criterion, Wf[:p_model, :p_model] if criterion == "I" else None,
                            p_model)
    if criterion == "D" and not np.isfinite(value):
        return None, value
    return D, value


# ---------------------------------------------------------------------------- measures
def matrix_measures(X: np.ndarray, W: np.ndarray | None = None, n_model: int | None = None
                    ) -> dict:
    """Design matrix measures: condition number, D/A/G/I efficiencies, scaled D, average
    and maximum leverage (Design-Expert's "Matrix Measures")."""
    n, p = X.shape
    M = X.T @ X
    out = {"n": n, "p": p}
    sign, ld = np.linalg.slogdet(M)
    if sign <= 0:
        out.update(singular=True)
        return out
    inv = np.linalg.inv(M)
    h = np.einsum("ij,ij->i", X @ inv, X)
    # condition number of the correlation form (X scaled to unit column length)
    norms = np.sqrt(np.diag(M))
    R = M / np.outer(norms, norms)
    ev = np.linalg.eigvalsh(R)
    out["condition"] = float(ev.max() / ev.min()) if ev.min() > 0 else math.inf
    out["log_det"] = float(ld)
    out["det"] = float(math.exp(ld)) if ld < 700 else math.inf
    out["scaled_d"] = float(math.exp(ld / p) / n)          # |X'X/n|^(1/p)
    out["d_eff"] = 100 * out["scaled_d"]
    out["trace"] = float(np.trace(inv))
    out["a_eff"] = 100 * p / (n * out["trace"])
    out["max_leverage"] = float(h.max())
    out["avg_leverage"] = float(h.mean())
    out["g_eff"] = 100 * p / (n * h.max())
    if W is not None:
        Wf = np.zeros_like(M)
        Wf[:W.shape[0], :W.shape[1]] = W
        out["i_value"] = float(np.trace(inv @ Wf))           # average prediction variance
        out["scaled_i"] = n * out["i_value"]
    return out


def coefficient_correlation(X: np.ndarray) -> np.ndarray:
    """Correlation matrix of the coefficient estimates, from (X'X)^-1."""
    inv = np.linalg.pinv(X.T @ X)
    d = np.sqrt(np.clip(np.diag(inv), 1e-300, None))
    return inv / np.outer(d, d)


def column_correlation(X: np.ndarray) -> np.ndarray:
    """Pearson r between the model columns (the intercept column gives r = 0)."""
    Z = X - X.mean(axis=0)
    s = np.sqrt((Z ** 2).sum(axis=0))
    s = np.where(s < 1e-12, np.inf, s)
    R = (Z / s).T @ (Z / s)
    np.fill_diagonal(R, 1.0)
    return R


def alias_matrix(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """A = (X1'X1)^-1 X1'X2: E[b1] = beta1 + A beta2 when the omitted terms X2 are active."""
    return np.linalg.pinv(X1.T @ X1) @ X1.T @ X2


__all__ = ["CRITERIA", "METHODS", "OptimalResult", "optimal_design", "model_function",
           "moment_matrix", "criterion_value", "matrix_measures", "coefficient_correlation",
           "column_correlation", "alias_matrix", "block_columns"]

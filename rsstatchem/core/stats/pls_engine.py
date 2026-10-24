"""Numerical core of PLS-SEM (Lohmöller / Wold PLS path modelling, as in SmartPLS 3/4).

Everything after standardising the indicators depends only on their correlation matrix, so
the iterative algorithm works on a *batch* of correlation matrices ``S`` with shape
``(B, p, p)``. One call estimates the original sample (B = 1) or thousands of bootstrap /
jackknife resamples at once, which keeps 5000 bootstrap subsamples to a few seconds.

Conventions: ``p`` indicator slots (an indicator used by two constructs occupies two slots),
``K`` constructs in topological order, ``D[i, j] = 1`` for the structural path i -> j,
outer weights ``W`` (p x K, block diagonal) scaled so every construct score has variance 1,
inner weights ``E`` with ``Z = Y E`` (column j = inner proxy of construct j).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy import linalg, stats

from rsstatchem.core.stats.pls_model import FORMATIVE, INTERACTION, REFLECTIVE, PLSModel

SCHEMES = ("path", "factorial", "centroid")
SCHEME_TEXT = {"path": "Path", "factorial": "Factorial", "centroid": "Centroid"}


@dataclass
class Structure:
    names: list[str]
    modes: list[str]
    blocks: list[np.ndarray]
    indicators: list[str]          # variable name of every slot
    owner: np.ndarray              # construct index of every slot
    D: np.ndarray                  # (K, K) adjacency, D[i, j] = 1 for i -> j
    preds: list[np.ndarray] = field(default_factory=list)
    succs: list[np.ndarray] = field(default_factory=list)

    @property
    def p(self) -> int:
        return len(self.indicators)

    @property
    def K(self) -> int:
        return len(self.names)

    @property
    def mask(self) -> np.ndarray:
        m = np.zeros((self.p, self.K))
        m[np.arange(self.p), self.owner] = 1.0
        return m

    @property
    def endogenous(self) -> list[int]:
        return [j for j in range(self.K) if len(self.preds[j])]

    @property
    def exogenous(self) -> list[int]:
        return [j for j in range(self.K) if not len(self.preds[j])]

    def reflective(self, j: int) -> bool:
        return self.modes[j] == REFLECTIVE

    def is_interaction(self, j: int) -> bool:
        return self.modes[j] == INTERACTION

    def edges(self) -> list[tuple[int, int]]:
        return [(i, j) for j in range(self.K) for i in self.preds[j]]

    def chains(self) -> list[list[int]]:
        """All directed paths with at least two arrows (specific indirect effects)."""
        out: list[list[int]] = []

        def walk(chain: list[int]) -> None:
            for nxt in self.succs[chain[-1]]:
                new = chain + [int(nxt)]
                if len(new) >= 3:
                    out.append(new)
                walk(new)

        for i in range(self.K):
            walk([i])
        return out

    def reach_pairs(self, min_len: int = 1) -> list[tuple[int, int]]:
        """(i, j) connected by a directed path of at least ``min_len`` arrows."""
        pairs = set()
        if min_len <= 1:
            pairs.update(self.edges())
        for c in self.chains():
            pairs.add((c[0], c[-1]))
        return sorted(pairs, key=lambda ij: (ij[1], ij[0]))


def make_structure(model: PLSModel) -> Structure:
    model = model.sorted()
    names = model.names()
    idx = {n: k for k, n in enumerate(names)}
    blocks, indicators, owner = [], [], []
    for k, c in enumerate(model.constructs):
        start = len(indicators)
        indicators += list(c.indicators)
        owner += [k] * len(c.indicators)
        blocks.append(np.arange(start, len(indicators)))
    K = len(names)
    D = np.zeros((K, K))
    for s, t in model.paths:
        D[idx[s], idx[t]] = 1.0
    st = Structure(names, [c.mode for c in model.constructs], blocks, indicators,
                   np.array(owner), D)
    st.preds = [np.flatnonzero(D[:, j]) for j in range(K)]
    st.succs = [np.flatnonzero(D[j, :]) for j in range(K)]
    return st


# ---------------------------------------------------------------------- data helpers
def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """z-scores with the population SD (as SmartPLS: blindfolding SSO = n per indicator)."""
    mean = X.mean(axis=0)
    sd = X.std(axis=0)
    return (X - mean) / sd, mean, sd


def corr_from_freq(X: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Correlation matrices of resamples given as case frequencies ``F`` (B x n)."""
    n, p = X.shape
    tot = F.sum(axis=1, keepdims=True)
    mean = F @ X / tot
    XX = (X[:, :, None] * X[:, None, :]).reshape(n, p * p)
    cov = (F @ XX / tot).reshape(-1, p, p) - mean[:, :, None] * mean[:, None, :]
    sd = np.sqrt(np.clip(np.einsum("bii->bi", cov), 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        S = cov / (sd[:, :, None] * sd[:, None, :])
    idx = np.arange(p)
    S[:, idx, idx] = 1.0
    S[~np.isfinite(S).all(axis=(1, 2))] = np.nan
    return S


def _solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Batched least-squares solve that tolerates singular systems."""
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A) @ b


# ---------------------------------------------------------------------- algorithm
def _normalise(W: np.ndarray, S: np.ndarray) -> np.ndarray:
    var = np.einsum("bpk,bpq,bqk->bk", W, S, W)
    with np.errstate(invalid="ignore", divide="ignore"):
        return W / np.sqrt(var)[:, None, :]


def _inner_weights(R: np.ndarray, st: Structure, scheme: str) -> np.ndarray:
    A = st.D + st.D.T
    if scheme == "centroid":
        return np.sign(R) * A
    if scheme == "factorial":
        return R * A
    E = np.zeros_like(R)
    for j in range(st.K):
        P, Q = st.preds[j], st.succs[j]
        if len(P):
            E[:, P, j] = _solve(R[:, P][:, :, P], R[:, P, j][:, :, None])[:, :, 0]
        if len(Q):
            E[:, Q, j] = R[:, Q, j]
    return E


def pls_weights(S: np.ndarray, st: Structure, scheme: str = "path", max_iter: int = 300,
                tol: float = 1e-7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iterate outer/inner estimation. Returns weights (B,p,K), iterations, converged."""
    Bn = S.shape[0]
    mask = st.mask
    W = _normalise(np.broadcast_to(mask, (Bn, st.p, st.K)).copy(), S)
    inv_b = {}
    for j in range(st.K):
        if st.modes[j] == FORMATIVE and len(st.blocks[j]) > 1:
            blk = st.blocks[j]
            Sjj = S[:, blk][:, :, blk]
            with np.errstate(invalid="ignore"):
                inv_b[j] = np.linalg.pinv(np.nan_to_num(Sjj))
    iters = np.zeros(Bn, int)
    done = np.zeros(Bn, bool)
    conv = np.zeros(Bn, bool)
    for it in range(1, max_iter + 1):
        R = np.einsum("bpk,bpq,bql->bkl", W, S, W)
        E = _inner_weights(R, st, scheme)
        SWE = S @ (W @ E)
        Wn = SWE * mask
        for j, inv in inv_b.items():
            blk = st.blocks[j]
            Wn[:, blk, j] = (inv @ SWE[:, blk, j][:, :, None])[:, :, 0]
        Wn = _normalise(Wn, S)
        diff = np.abs(Wn - W).sum(axis=(1, 2))
        W = np.where(done[:, None, None], W, Wn)
        iters[~done] = it
        newly = ~done & (diff < tol)
        conv |= newly
        done |= newly | ~np.isfinite(diff)
        if done.all():
            break
    return W, iters, conv & np.isfinite(W).all(axis=(1, 2))


def rho_a(W: np.ndarray, st: Structure) -> np.ndarray:
    """Dijkstra-Henseler rho_A (closed form for weights scaled to unit score variance)."""
    out = np.ones((W.shape[0], st.K))
    for j in range(st.K):
        blk = st.blocks[j]
        if len(blk) < 2:
            continue
        w = W[:, blk, j]
        ww = (w ** 2).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, j] = ww ** 2 * (1 - ww) / (ww ** 2 - (w ** 4).sum(axis=1))
    return out


def regress_paths(R: np.ndarray, st: Structure) -> tuple[np.ndarray, np.ndarray]:
    """OLS path coefficients (B,K,K) and R-squared (B,K) from construct correlations."""
    Bn = R.shape[0]
    paths = np.zeros((Bn, st.K, st.K))
    r2 = np.full((Bn, st.K), np.nan)
    for j in st.endogenous:
        P = st.preds[j]
        beta = _solve(R[:, P][:, :, P], R[:, P, j][:, :, None])[:, :, 0]
        paths[:, P, j] = beta
        r2[:, j] = (beta * R[:, P, j]).sum(axis=1)
    return paths, r2


def total_effects(paths: np.ndarray, st: Structure) -> np.ndarray:
    total = paths.copy()
    power = paths.copy()
    for _ in range(st.K - 1):
        power = power @ paths
        if not np.any(power):
            break
        total = total + power
    return total


def estimate(S: np.ndarray, st: Structure, scheme: str = "path", max_iter: int = 300,
             tol: float = 1e-7, consistent: bool = False) -> dict:
    """Full PLS (or PLSc) estimation for a batch of correlation matrices."""
    if S.ndim == 2:
        S = S[None]
    W, iters, conv = pls_weights(S, st, scheme, max_iter, tol)
    C = S @ W                                      # cross-loadings corr(x_k, Y_j)
    R = np.einsum("bpk,bpq,bql->bkl", W, S, W)
    rhoA = rho_a(W, st)
    loadings = C * st.mask
    if consistent:
        refl = np.array([st.reflective(j) and len(st.blocks[j]) > 1 for j in range(st.K)])
        with np.errstate(invalid="ignore", divide="ignore"):
            q = np.where(refl[None, :], np.sqrt(rhoA), 1.0)  # composite/construct correlation
            R = R / (q[:, :, None] * q[:, None, :])
            ww = (W ** 2).sum(axis=1)                         # (B, K) = w'w per construct
            lam_c = W * (np.sqrt(rhoA) / ww)[:, None, :]      # c * w, c = sqrt(rhoA) / w'w
        idx = np.arange(st.K)
        R[:, idx, idx] = 1.0
        loadings = np.where(st.mask[None].astype(bool) & refl[st.owner][None, :, None],
                            lam_c, loadings)
    paths, r2 = regress_paths(R, st)
    return {"W": W, "C": C, "R": R, "loadings": loadings, "paths": paths, "r2": r2,
            "rhoA": rhoA, "iterations": iters, "converged": conv}


# ---------------------------------------------------------------------- quality criteria
def f_squared(R: np.ndarray, r2: np.ndarray, st: Structure) -> np.ndarray:
    f2 = np.full(R.shape[:1] + (st.K, st.K), np.nan)
    for j in st.endogenous:
        P = list(st.preds[j])
        for i in P:
            rest = [k for k in P if k != i]
            if rest:
                beta = _solve(R[:, rest][:, :, rest], R[:, rest, j][:, :, None])[:, :, 0]
                r2x = (beta * R[:, rest, j]).sum(axis=1)
            else:
                r2x = np.zeros(R.shape[0])
            with np.errstate(invalid="ignore", divide="ignore"):
                f2[:, i, j] = (r2[:, j] - r2x) / (1 - r2[:, j])
    return f2


def htmt(S: np.ndarray, st: Structure, pairs: list[tuple[int, int]]) -> np.ndarray:
    """Heterotrait-monotrait ratio with absolute correlations (SmartPLS 4)."""
    A = np.abs(S)
    mono = []
    for j in range(st.K):
        blk = st.blocks[j]
        if len(blk) < 2:
            mono.append(np.ones(S.shape[0]))
            continue
        iu = np.triu_indices(len(blk), 1)
        mono.append(A[:, blk][:, :, blk][:, iu[0], iu[1]].mean(axis=1))
    out = np.zeros((S.shape[0], len(pairs)))
    for k, (i, j) in enumerate(pairs):
        hetero = A[:, st.blocks[i]][:, :, st.blocks[j]].mean(axis=(1, 2))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, k] = hetero / np.sqrt(mono[i] * mono[j])
    return out


def outer_vif(S: np.ndarray, st: Structure) -> np.ndarray:
    vif = np.full(st.p, np.nan)
    for blk in st.blocks:
        if len(blk) == 1:
            vif[blk] = 1.0
            continue
        vif[blk] = np.diag(np.linalg.pinv(S[np.ix_(blk, blk)]))
    return vif


def inner_vif(R: np.ndarray, st: Structure) -> np.ndarray:
    vif = np.full((st.K, st.K), np.nan)
    for j in st.endogenous:
        P = st.preds[j]
        vif[P, j] = 1.0 if len(P) == 1 else np.diag(np.linalg.pinv(R[np.ix_(P, P)]))
    return vif


def reliability(S: np.ndarray, loadings: np.ndarray, rhoA: np.ndarray,
                st: Structure) -> dict[str, np.ndarray]:
    """Cronbach's alpha (standardised), rho_A, rho_C and AVE per construct."""
    K = st.K
    alpha, rhoc, ave = (np.full(K, np.nan) for _ in range(3))
    for j in range(K):
        blk = st.blocks[j]
        lam = loadings[blk, j]
        k = len(blk)
        if k == 1:
            alpha[j] = rhoc[j] = 1.0
        else:
            rbar = S[np.ix_(blk, blk)][np.triu_indices(k, 1)].mean()
            alpha[j] = k * rbar / (1 + (k - 1) * rbar)
            rhoc[j] = lam.sum() ** 2 / (lam.sum() ** 2 + (1 - lam ** 2).sum())
        ave[j] = (lam ** 2).mean()
    return {"alpha": alpha, "rhoA": rhoA, "rhoC": rhoc, "ave": ave}


def implied_corr(S: np.ndarray, loadings: np.ndarray, Phi: np.ndarray,
                 st: Structure) -> np.ndarray:
    lam = loadings.sum(axis=1)
    L = st.mask * lam[:, None]
    sigma = L @ Phi @ L.T
    for j, blk in enumerate(st.blocks):
        ix = np.ix_(blk, blk)
        sigma[ix] = S[ix] if not st.reflective(j) else np.outer(lam[blk], lam[blk])
    np.fill_diagonal(sigma, 1.0)
    return sigma


def structural_implied(R: np.ndarray, paths: np.ndarray, r2: np.ndarray,
                       st: Structure) -> np.ndarray:
    """Construct correlations implied by the recursive structural model."""
    K = st.K
    exo = st.exogenous
    psi = np.zeros((K, K))
    psi[np.ix_(exo, exo)] = R[np.ix_(exo, exo)]
    for j in st.endogenous:
        psi[j, j] = 1 - r2[j]
    A = np.linalg.inv(np.eye(K) - paths.T)
    return A @ psi @ A.T


def fit_indices(S: np.ndarray, sigma: np.ndarray, n: int) -> dict[str, float]:
    p = S.shape[0]
    resid = S - sigma
    low = np.tril_indices(p)
    out = {"SRMR": float(np.sqrt(np.mean(resid[low] ** 2))),
           "d_ULS": float(0.5 * np.sum(resid ** 2)),
           "d_G": np.nan, "Chi-square": np.nan, "NFI": np.nan}
    try:
        ev = linalg.eigh(sigma, S, eigvals_only=True)
        if np.all(ev > 0):
            out["d_G"] = float(0.5 * np.sum(np.log(ev) ** 2))
        sign_s, logdet_s = np.linalg.slogdet(S)
        sign_m, logdet_m = np.linalg.slogdet(sigma)
        if sign_s > 0 and sign_m > 0:
            fml = logdet_m + np.trace(S @ np.linalg.inv(sigma)) - logdet_s - p
            chi = (n - 1) * fml
            chi0 = (n - 1) * (-logdet_s)
            out["Chi-square"] = float(chi)
            out["NFI"] = float(1 - chi / chi0) if chi0 > 0 else np.nan
    except (np.linalg.LinAlgError, linalg.LinAlgError, ValueError):
        pass
    return out


# ---------------------------------------------------------------------- statistics vector
GROUPS = ("paths", "total_indirect", "specific_indirect", "total", "loadings", "weights",
          "r2", "f2", "htmt")


def stat_labels(st: Structure) -> dict[str, list]:
    n = st.names
    refl = [j for j in range(st.K) if st.reflective(j)]
    return {
        "paths": [(i, j, f"{n[i]} -> {n[j]}") for i, j in st.edges()],
        "total_indirect": [(i, j, f"{n[i]} -> {n[j]}") for i, j in st.reach_pairs(2)],
        "specific_indirect": [(c, " -> ".join(n[k] for k in c)) for c in st.chains()],
        "total": [(i, j, f"{n[i]} -> {n[j]}") for i, j in st.reach_pairs(1)],
        "loadings": [(k, f"{st.indicators[k]} <- {n[st.owner[k]]}") for k in range(st.p)],
        "weights": [(k, f"{st.indicators[k]} -> {n[st.owner[k]]}") for k in range(st.p)],
        "r2": [(j, n[j]) for j in st.endogenous],
        "f2": [(i, j, f"{n[i]} -> {n[j]}") for i, j in st.edges()],
        "htmt": [(j, i, f"{n[j]} <-> {n[i]}") for i, j in combinations(refl, 2)],
    }


def stat_vectors(S: np.ndarray, est: dict, st: Structure, labels: dict) -> dict:
    """Every bootstrappable statistic as (B, m) arrays keyed like :func:`stat_labels`."""
    P = est["paths"]
    tot = total_effects(P, st)
    out = {
        "paths": np.stack([P[:, i, j] for i, j, _ in labels["paths"]], 1),
        "total": np.stack([tot[:, i, j] for i, j, _ in labels["total"]], 1),
        "loadings": np.stack([est["loadings"][:, k, st.owner[k]]
                              for k, _ in labels["loadings"]], 1),
        "weights": np.stack([est["W"][:, k, st.owner[k]] for k, _ in labels["weights"]], 1),
        "r2": np.stack([est["r2"][:, j] for j, _ in labels["r2"]], 1),
    }
    ind = tot - P
    out["total_indirect"] = np.stack([ind[:, i, j] for i, j, _ in labels["total_indirect"]],
                                     1) if labels["total_indirect"] else np.zeros((len(P), 0))
    spec = []
    for chain, _ in labels["specific_indirect"]:
        prod = np.ones(len(P))
        for a, b in zip(chain, chain[1:]):
            prod = prod * P[:, a, b]
        spec.append(prod)
    out["specific_indirect"] = np.stack(spec, 1) if spec else np.zeros((len(P), 0))
    f2 = f_squared(est["R"], est["r2"], st)
    out["f2"] = np.stack([f2[:, i, j] for i, j, _ in labels["f2"]], 1)
    pairs = [(i, j) for j, i, _ in labels["htmt"]]
    out["htmt"] = htmt(S, st, pairs) if pairs else np.zeros((len(P), 0))
    return out


# ---------------------------------------------------------------------- bootstrap
@dataclass
class BootResult:
    B: int
    valid: int
    method: str
    tails: int
    alpha: float
    labels: dict
    original: dict
    draws: dict
    table: dict = field(default_factory=dict)  # group -> list of row dicts


@dataclass
class Stage1:
    """Two-stage moderation: the main-effects model whose scores form interaction terms."""
    st: Structure
    columns: np.ndarray                    # X column of every stage-1 slot
    terms: list[tuple[int, int, int]]      # (X column of the product, construct a, b)
    scheme: str = "path"
    max_iter: int = 300
    tol: float = 1e-7

    def products(self, Xr: np.ndarray) -> np.ndarray:
        """Insert the interaction columns into raw data batches ``Xr`` (b, n, p)."""
        Z1 = _zbatch(Xr[:, :, self.columns])
        n = Z1.shape[1]
        S1 = np.einsum("bni,bnj->bij", Z1, Z1) / n
        W = estimate(S1, self.st, self.scheme, self.max_iter, self.tol)["W"]
        Y = _zbatch(Z1 @ W)
        Xr = Xr.copy()
        for col, a, b in self.terms:
            Xr[:, :, col] = Y[:, :, a] * Y[:, :, b]
        return Xr


def _zbatch(X: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)


def _resample_S(X: np.ndarray, Z: np.ndarray, idx: np.ndarray,
                stage1: Stage1 | None) -> np.ndarray:
    """Correlation matrices of the resamples given by row indices ``idx`` (b, m)."""
    b, m = idx.shape
    n = X.shape[0]
    if stage1 is None:
        F = np.zeros((b, n))
        np.add.at(F, (np.repeat(np.arange(b), m), idx.ravel()), 1.0)
        return corr_from_freq(Z, F)
    Zr = _zbatch(stage1.products(X[idx]))
    S = np.einsum("bni,bnj->bij", Zr, Zr) / m
    S[~np.isfinite(S).all(axis=(1, 2))] = np.nan
    return S


def _ci_bounds(alpha: float, tails: int) -> tuple[float, float]:
    return (alpha / 2, 1 - alpha / 2) if tails == 2 else (alpha, 1 - alpha)


def bootstrap(X: np.ndarray, st: Structure, subsamples: int = 5000, seed: int | None = 1,
              scheme: str = "path", max_iter: int = 300, tol: float = 1e-7,
              consistent: bool = False, method: str = "percentile", tails: int = 2,
              alpha: float = 0.05, chunk: int | None = None,
              stage1: Stage1 | None = None) -> BootResult:
    """Nonparametric bootstrap with case resampling (percentile, BC or BCa intervals).

    With ``stage1`` (two-stage moderation) the interaction terms are rebuilt from the
    stage-1 scores of every resample, as SmartPLS does."""
    n, p = X.shape
    Z, _, _ = standardize(X)
    labels = stat_labels(st)
    S0 = (Z.T @ Z / n)[None]
    est0 = estimate(S0, st, scheme, max_iter, tol, consistent)
    orig = {g: v[0] for g, v in stat_vectors(S0, est0, st, labels).items()}
    rng = np.random.default_rng(seed)
    chunk = chunk or max(50, min(1000, int(4e7 / (n * p * p + 1))))
    if stage1 is not None:
        chunk = max(20, min(chunk, int(2e7 / (n * p * 4 + 1))))
    draws = {g: [] for g in GROUPS}
    done = 0
    while done < subsamples:
        b = min(chunk, subsamples - done)
        idx = rng.integers(0, n, size=(b, n))
        S = _resample_S(X, Z, idx, stage1)
        est = estimate(np.nan_to_num(S), st, scheme, max_iter, tol, consistent)
        vec = stat_vectors(S, est, st, labels)
        bad = ~np.isfinite(S).all(axis=(1, 2)) | ~est["converged"]
        for g in GROUPS:
            v = vec[g]
            v[bad] = np.nan
            draws[g].append(v)
        done += b
    draws = {g: np.concatenate(v, 0) for g, v in draws.items()}
    valid = int(np.isfinite(draws["paths"]).all(axis=1).sum())
    res = BootResult(subsamples, valid, method, tails, alpha, labels, orig, draws)
    accel = {}
    if method == "bca":
        jidx = np.array([np.delete(np.arange(n), i) for i in range(n)])
        S = np.concatenate([_resample_S(X, Z, jidx[k:k + 200], stage1)
                            for k in range(0, n, 200)], 0)
        estj = estimate(np.nan_to_num(S), st, scheme, max_iter, tol, consistent)
        jack = stat_vectors(S, estj, st, labels)
        for g in GROUPS:
            d = np.nanmean(jack[g], axis=0) - jack[g]
            with np.errstate(invalid="ignore", divide="ignore"):
                accel[g] = np.nansum(d ** 3, axis=0) / (6 * np.nansum(d ** 2, axis=0) ** 1.5)
    lo_q, hi_q = _ci_bounds(alpha, tails)
    df = max(valid - 1, 1)
    for g in GROUPS:
        rows = []
        d = draws[g]
        for k, lab in enumerate(labels[g]):
            col = d[:, k]
            col = col[np.isfinite(col)]
            o = float(orig[g][k])
            if len(col) < 2:
                rows.append({"label": lab[-1], "O": o})
                continue
            mean, sd = float(col.mean()), float(col.std(ddof=1))
            t = abs(o) / sd if sd > 0 else np.nan
            pval = (2 if tails == 2 else 1) * stats.t.sf(t, df) if np.isfinite(t) else np.nan
            qs = (lo_q, hi_q)
            if method in ("bc", "bca"):
                prop = np.mean(col < o)
                prop = min(max(prop, 0.5 / len(col)), 1 - 0.5 / len(col))
                z0 = stats.norm.ppf(prop)
                a = float(accel[g][k]) if method == "bca" and np.isfinite(accel[g][k]) else 0.0
                zq = stats.norm.ppf(np.array(qs))
                qs = tuple(stats.norm.cdf(z0 + (z0 + zq) / (1 - a * (z0 + zq))))
            lo, hi = np.quantile(col, qs)
            rows.append({"label": lab[-1], "O": o, "M": mean, "STDEV": sd, "T": t, "P": pval,
                         "lo": float(lo), "hi": float(hi), "bias": mean - o})
        res.table[g] = rows
    return res


# ---------------------------------------------------------------------- blindfolding
def blindfolding(Z: np.ndarray, st: Structure, D: int = 7, scheme: str = "path",
                 max_iter: int = 300, tol: float = 1e-7) -> dict[int, tuple[float, float]]:
    """Construct cross-validated redundancy: {construct: (SSO, SSE)} for endogenous ones.

    Every D-th data point of the target construct's indicators (counted case by case) is
    replaced by the mean, the model is re-estimated and the omitted points are predicted
    from the predecessor scores through the path coefficients and the loadings.
    """
    n = Z.shape[0]
    out = {}
    for j in st.endogenous:
        blk = st.blocks[j]
        pj = len(blk)
        pos = (np.arange(n)[:, None] * pj + np.arange(pj)[None, :]) % D
        sso = sse = 0.0
        for d in range(D):
            omit = pos == d
            if not omit.any():
                continue
            Xb = Z.copy()
            sub = Xb[:, blk]
            sub[omit] = 0.0
            Xb[:, blk] = sub
            Zb, mb, sb = standardize(Xb)
            S = Zb.T @ Zb / n
            est = estimate(S[None], st, scheme, max_iter, tol)
            W, P, C = est["W"][0], est["paths"][0], est["C"][0]
            Y = Zb @ W
            eta = Y @ P[:, j]
            pred = eta[:, None] * C[blk, j][None, :]
            pred = pred * sb[blk] + mb[blk]
            sse += float(((Z[:, blk] - pred) ** 2)[omit].sum())
            sso += float((Z[:, blk] ** 2)[omit].sum())
        out[j] = (sso, sse)
    return out


# ---------------------------------------------------------------------- PLSpredict
def pls_predict(X: np.ndarray, st: Structure, folds: int = 10, reps: int = 10,
                seed: int | None = 1, scheme: str = "path", max_iter: int = 300,
                tol: float = 1e-7) -> list[dict]:
    """k-fold PLSpredict (Shmueli et al. 2016/2019) with the linear-model benchmark."""
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    endo = st.endogenous
    targets = [k for j in endo for k in st.blocks[j]]
    exo_cols = sorted({st.indicators[k]: k for j in st.exogenous
                       for k in st.blocks[j]}.values())
    err_pls = np.zeros((reps, n, len(targets)))
    err_lm = np.zeros_like(err_pls)
    naive = np.zeros_like(err_pls)
    for r in range(reps):
        order = rng.permutation(n)
        for hold in np.array_split(order, folds):
            train = np.setdiff1d(np.arange(n), hold)
            Xt = X[train]
            mean, sd = Xt.mean(axis=0), Xt.std(axis=0)
            if np.any(sd == 0):
                sd = np.where(sd == 0, 1.0, sd)
            Zt = (Xt - mean) / sd
            est = estimate((Zt.T @ Zt / len(train))[None], st, scheme, max_iter, tol)
            W, P, C = est["W"][0], est["paths"][0], est["C"][0]
            Zh = (X[hold] - mean) / sd
            Yh = Zh @ W
            tot = total_effects(P[None], st)[0]
            exo = st.exogenous
            A = np.column_stack([np.ones(len(train)), Xt[:, exo_cols]])
            coef = np.linalg.lstsq(A, Xt[:, targets], rcond=None)[0]
            Ah = np.column_stack([np.ones(len(hold)), X[hold][:, exo_cols]])
            err_lm[r, hold, :] = X[hold][:, targets] - Ah @ coef
            for t, k in enumerate(targets):
                j = st.owner[k]
                eta = Yh[:, exo] @ tot[exo, j]
                pred = eta * C[k, j] * sd[k] + mean[k]
                err_pls[r, hold, t] = X[hold, k] - pred
                naive[r, hold, t] = X[hold, k] - mean[k]
    rows = []
    for t, k in enumerate(targets):
        e, lm, nv = err_pls[:, :, t], err_lm[:, :, t], naive[:, :, t]
        rows.append({
            "indicator": st.indicators[k], "construct": st.names[st.owner[k]],
            "Q2predict": 1 - (e ** 2).sum() / (nv ** 2).sum(),
            "PLS_RMSE": float(np.mean(np.sqrt((e ** 2).mean(axis=1)))),
            "PLS_MAE": float(np.mean(np.abs(e).mean(axis=1))),
            "LM_RMSE": float(np.mean(np.sqrt((lm ** 2).mean(axis=1)))),
            "LM_MAE": float(np.mean(np.abs(lm).mean(axis=1))),
        })
    return rows

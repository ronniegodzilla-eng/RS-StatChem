"""Estimation engine for covariance-based SEM (the part of R lavaan / AMOS that fits the model).

RAM formulation (McArdle & McDonald): variables v = (observed, latent), A = directed paths,
S = (residual) variances/covariances, Sigma = F (I-A)^-1 S (I-A)^-T F'. Discrepancy functions:
ML (default), GLS and ULS. Optimised by Fisher scoring with step halving, vectorised over a
batch of covariance matrices so the nonparametric bootstrap refits thousands of resamples at
once. Standard errors from the expected or observed information (ULS: normal-theory
sandwich), robust (MLM) sandwich SEs and the Satorra-Bentler scaled chi-square.
Modification indices are score tests with expected parameter change. No Qt here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from rsstatchem.core.stats.sem_syntax import Param, SEMModel

ESTIMATORS = {"ML": "Maximum likelihood", "GLS": "Generalized least squares",
              "ULS": "Unweighted least squares"}


class SEMError(ValueError):
    """A model that cannot be estimated (not identified, singular data, …)."""


# ------------------------------------------------------------------ specification
class Spec:
    """Numeric RAM representation of a :class:`SEMModel` (or of a list of params)."""

    def __init__(self, params: list[Param], observed: list[str], latent: list[str], q: int):
        self.params = params
        self.observed, self.latent = list(observed), list(latent)
        self.names = self.observed + self.latent
        self.pos = {n: i for i, n in enumerate(self.names)}
        self.p, self.m, self.q = len(self.observed), len(self.names), q
        m = self.m
        self.A0 = np.zeros((m, m))
        self.S0 = np.zeros((m, m))
        a, s = [], []
        for prm in params:
            i, j = self._cell(prm)
            if prm.op == "~~":
                if prm.free:
                    s.append((i, j, prm.index))
                else:
                    self.S0[i, j] = self.S0[j, i] = prm.value
            else:
                if prm.free:
                    a.append((i, j, prm.index))
                else:
                    self.A0[i, j] = prm.value
        self.a_pos = np.array(a, dtype=int).reshape(-1, 3)
        self.s_pos = np.array(s, dtype=int).reshape(-1, 3)
        self.Ia = np.zeros((q, len(a)))
        self.Ia[self.a_pos[:, 2], np.arange(len(a))] = 1
        self.Is = np.zeros((q, len(s)))
        self.Is[self.s_pos[:, 2], np.arange(len(s))] = 1
        self.s_off = self.s_pos[:, 0] != self.s_pos[:, 1]

    def _cell(self, prm: Param) -> tuple[int, int]:
        if prm.op == "=~":
            return self.pos[prm.rhs], self.pos[prm.lhs]
        if prm.op == "~":
            return self.pos[prm.lhs], self.pos[prm.rhs]
        return self.pos[prm.lhs], self.pos[prm.rhs]

    @classmethod
    def from_model(cls, model: SEMModel) -> "Spec":
        return cls(model.params, model.observed, model.latent, model.n_free)

    # --------------------------------------------------------------- matrices
    def matrices(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        theta = np.atleast_2d(theta)
        nb = len(theta)
        A = np.repeat(self.A0[None], nb, axis=0)
        S = np.repeat(self.S0[None], nb, axis=0)
        if len(self.a_pos):
            A[:, self.a_pos[:, 0], self.a_pos[:, 1]] = theta[:, self.a_pos[:, 2]]
        if len(self.s_pos):
            S[:, self.s_pos[:, 0], self.s_pos[:, 1]] = theta[:, self.s_pos[:, 2]]
            S[:, self.s_pos[:, 1], self.s_pos[:, 0]] = theta[:, self.s_pos[:, 2]]
        return A, S

    def implied(self, theta: np.ndarray):
        """(Sigma, B, Omega) for a batch: B = (I-A)^-1, Omega = implied cov of all vars."""
        A, S = self.matrices(theta)
        eye = np.eye(self.m)
        B = np.linalg.inv(eye[None] - A)
        Om = B @ S @ np.swapaxes(B, 1, 2)
        Om = 0.5 * (Om + np.swapaxes(Om, 1, 2))
        return Om[:, :self.p, :self.p], B, Om

    def jacobian(self, B: np.ndarray, Om: np.ndarray) -> np.ndarray:
        """dSigma/dtheta, shape (nb, q, p, p)."""
        p = self.p
        M = B[:, :p, :]                       # F B
        OF = Om[:, :, :p]                     # Omega F'
        nb = len(B)
        out = np.zeros((nb, self.q, p * p))
        if len(self.a_pos):
            ai, aj = self.a_pos[:, 0], self.a_pos[:, 1]
            D = M[:, :, ai].transpose(0, 2, 1)[:, :, :, None] * OF[:, aj, None, :]
            D = D + np.swapaxes(D, 2, 3)
            out += self.Ia @ D.reshape(nb, len(ai), p * p)
        if len(self.s_pos):
            si, sj = self.s_pos[:, 0], self.s_pos[:, 1]
            D = M[:, :, si].transpose(0, 2, 1)[:, :, :, None] * \
                M[:, :, sj].transpose(0, 2, 1)[:, :, None, :]
            D = D + self.s_off[None, :, None, None] * np.swapaxes(D, 2, 3)
            out += self.Is @ D.reshape(nb, len(si), p * p)
        out = out.reshape(nb, self.q, p, p)
        return out


# ------------------------------------------------------------------ discrepancy
def _sym_inv(X: np.ndarray) -> np.ndarray:
    return np.linalg.inv(X)


def _pd(X: np.ndarray) -> np.ndarray:
    """Mask of positive-definite matrices in a batch."""
    try:
        w = np.linalg.eigvalsh(X)
    except np.linalg.LinAlgError:
        return np.zeros(len(X), bool)
    return np.all(np.isfinite(w), axis=1) & (w.min(axis=1) > 1e-12 * np.abs(w).max(axis=1))


class Discrepancy:
    """F(S, Sigma), gradient and expected Hessian for a batch of sample covariances."""

    def __init__(self, estimator: str, S: np.ndarray):
        self.est = estimator
        self.S = S                                   # (nb, p, p)
        self.p = S.shape[-1]
        sign, self.logdetS = np.linalg.slogdet(S)
        if estimator in ("ML", "GLS") and np.any(sign <= 0):
            raise SEMError("The sample covariance matrix is not positive definite (a variable "
                           "is constant or a linear combination of others).")
        self.Sinv = _sym_inv(S) if estimator == "GLS" else None

    def sub(self, idx: np.ndarray) -> "Discrepancy":
        """The same discrepancy restricted to the batch members ``idx``."""
        new = object.__new__(Discrepancy)
        new.est, new.p = self.est, self.p
        new.S, new.logdetS = self.S[idx], self.logdetS[idx]
        new.Sinv = None if self.Sinv is None else self.Sinv[idx]
        return new

    def value(self, Sig: np.ndarray) -> np.ndarray:
        nb = len(Sig)
        out = np.full(nb, np.inf)
        if self.est == "ML":
            ok = _pd(Sig)
            if ok.any():
                sg, ld = np.linalg.slogdet(Sig[ok])
                tr = np.einsum("bij,bji->b", self.S[ok], _sym_inv(Sig[ok]))
                out[ok] = ld + tr - self.logdetS[ok] - self.p
            return out
        R = self.S - Sig
        if self.est == "GLS":
            R = self.Sinv @ R
            return 0.5 * np.einsum("bij,bji->b", R, R)
        return 0.5 * np.einsum("bij,bji->b", R, R)

    def weight(self, Sig: np.ndarray) -> np.ndarray:
        if self.est == "ML":
            return _sym_inv(Sig)
        if self.est == "GLS":
            return self.Sinv
        return np.repeat(np.eye(self.p)[None], len(Sig), axis=0)

    def grad_hess(self, Sig: np.ndarray, dSig: np.ndarray):
        """Gradient g_k = tr(W (Sigma - S) W dS_k); expected Hessian H = tr(W dS_k W dS_l)."""
        W = self.weight(Sig)
        nb, q, p, _ = dSig.shape
        R = W @ (Sig - self.S) @ W
        g = (dSig.reshape(nb, q, p * p) @ R.reshape(nb, p * p, 1))[..., 0]
        G = W[:, None] @ dSig                       # W dS_k
        Gt = np.swapaxes(G, 2, 3).reshape(nb, q, p * p)
        H = G.reshape(nb, q, p * p) @ np.swapaxes(Gt, 1, 2)
        return g, 0.5 * (H + np.swapaxes(H, 1, 2))


# ------------------------------------------------------------------ optimisation
@dataclass
class FitState:
    theta: np.ndarray
    F: np.ndarray
    converged: np.ndarray
    iterations: np.ndarray
    grad: np.ndarray


def fisher_scoring(spec: Spec, disc: Discrepancy, theta0: np.ndarray, max_iter: int = 500,
                   tol: float = 1e-8) -> FitState:
    theta = np.array(np.atleast_2d(theta0), float)
    nb, q = theta.shape
    Sig, B, Om = spec.implied(theta)
    F = disc.value(Sig)
    if nb == 1 and not np.isfinite(F[0]):
        raise SEMError("The start values give a non-positive-definite implied covariance "
                       "matrix.")
    conv = np.zeros(nb, bool)
    iters = np.zeros(nb, int)
    grad = np.zeros((nb, q))
    active = np.isfinite(F)
    for it in range(max_iter):
        idx = np.flatnonzero(active)
        if not len(idx):
            break
        Sig, B, Om = spec.implied(theta[idx])
        dS = spec.jacobian(B, Om)
        g, H = disc.sub(idx).grad_hess(Sig, dS)
        grad[idx] = g
        done = np.max(np.abs(g), axis=1) < tol
        conv[idx[done]] = True
        active[idx[done]] = False
        keep = ~done
        idx, g, H = idx[keep], g[keep], H[keep]
        if not len(idx):
            break
        iters[idx] += 1
        ridge = 1e-10 * np.maximum(np.trace(H, axis1=1, axis2=2) / q, 1e-12)
        try:
            step = -np.linalg.solve(H + ridge[:, None, None] * np.eye(q), g[..., None])[..., 0]
        except np.linalg.LinAlgError:
            step = -np.einsum("bkl,bl->bk", np.linalg.pinv(H), g)
        t = np.ones(len(idx))
        newF = np.full(len(idx), np.inf)
        accepted = np.zeros(len(idx), bool)
        for _ in range(40):
            todo = ~accepted
            if not todo.any():
                break
            cand = theta[idx[todo]] + t[todo, None] * step[todo]
            Fc = disc.sub(idx[todo]).value(spec.implied(cand)[0])
            ok = np.isfinite(Fc) & (Fc <= F[idx[todo]] + 1e-12 * np.abs(F[idx[todo]]) + 1e-15)
            sel = np.flatnonzero(todo)
            newF[sel[ok]] = Fc[ok]
            theta[idx[sel[ok]]] = cand[ok]
            accepted[sel[ok]] = True
            t[sel[~ok]] *= 0.5
        small = np.abs(F[idx] - newF) <= 1e-14 * np.maximum(1, np.abs(F[idx]))
        F[idx[accepted]] = newF[accepted]
        stuck = ~accepted | (small & (np.max(np.abs(t[:, None] * step), axis=1) < 1e-10))
        if stuck.any():
            active[idx[stuck]] = False
            conv[idx[stuck]] = np.max(np.abs(g[stuck]), axis=1) < 1e-5
    if active.any():
        idx = np.flatnonzero(active)
        Sig, B, Om = spec.implied(theta[idx])
        g, _ = disc.sub(idx).grad_hess(Sig, spec.jacobian(B, Om))
        grad[idx] = g
        conv[idx] = np.max(np.abs(g), axis=1) < 1e-5
    return FitState(theta, F, conv, iters, grad)


# ------------------------------------------------------------------ start values
def start_values(model: SEMModel, spec: Spec, S: np.ndarray) -> np.ndarray:
    theta = np.zeros(spec.q)
    seen = np.zeros(spec.q, bool)
    obs = {v: i for i, v in enumerate(spec.observed)}
    lat_var: dict[str, float] = {}
    loads: dict[tuple[str, str], float] = {}
    for f in model.latent:
        items = [x for x in model.indicators[f] if x in obs]
        prms = {p.rhs: p for p in model.params if p.op == "=~" and p.lhs == f}
        if not items:
            lat_var[f] = 0.5
            for x in model.indicators[f]:
                loads[(f, x)] = 1.0
            continue
        ii = [obs[x] for x in items]
        C = S[np.ix_(ii, ii)]
        if len(ii) == 1:
            lvec = np.array([math.sqrt(max(C[0, 0], 1e-6) * 0.8)])
        else:
            w, V = np.linalg.eigh(C)
            lvec = V[:, -1] * math.sqrt(max(w[-1], 1e-6))
            if lvec.sum() < 0:
                lvec = -lvec
        ref = next((k for k, x in enumerate(items) if not prms[x].free), None)
        if ref is not None and abs(lvec[ref]) > 1e-8 and prms[items[ref]].value != 0:
            scale = lvec[ref] / prms[items[ref]].value
        else:
            scale = 1.0
        for k, x in enumerate(items):
            loads[(f, x)] = lvec[k] / scale
        lat_var[f] = scale ** 2
        for x in model.indicators[f]:
            loads.setdefault((f, x), 1.0)
    endo = set(model.endogenous)
    for prm in model.params:
        if not prm.free or seen[prm.index]:
            continue
        if prm.start is not None:
            val = prm.start
        elif prm.op == "=~":
            val = loads.get((prm.lhs, prm.rhs), 1.0)
        elif prm.op == "~":
            val = 0.0
        elif prm.lhs == prm.rhs:
            v = prm.lhs
            if v in obs:
                val = S[obs[v], obs[v]] * (0.5 if v in endo else 1.0)
            else:
                val = lat_var.get(v, 0.5) * (0.6 if v in endo else 1.0)
        elif prm.lhs in obs and prm.rhs in obs and prm.lhs not in endo and prm.rhs not in endo:
            val = S[obs[prm.lhs], obs[prm.rhs]]
        else:
            val = 0.0
        theta[prm.index] = val
        seen[prm.index] = True
    return theta


# ------------------------------------------------------------------ helpers
def vech_index(p: int) -> tuple[np.ndarray, np.ndarray]:
    return np.tril_indices(p)


def duplication(p: int) -> np.ndarray:
    r, c = np.tril_indices(p)
    D = np.zeros((p * p, len(r)))
    for k, (i, j) in enumerate(zip(r, c)):
        D[i * p + j, k] = 1
        D[j * p + i, k] = 1
    return D


def identification_check(spec: Spec, theta: np.ndarray, rng=None) -> list[int]:
    """Indices of free parameters involved in a rank deficiency of dSigma/dtheta at a
    generic point near ``theta`` (empty when locally identified)."""
    rng = rng or np.random.default_rng(12345)
    pt = theta * (1 + 0.05 * rng.standard_normal(theta.shape)) + 0.01 * rng.standard_normal(
        theta.shape)
    Sig, B, Om = spec.implied(pt)
    r, c = vech_index(spec.p)
    J = spec.jacobian(B, Om)[0][:, r, c].T
    if J.shape[1] == 0:
        return []
    col = np.linalg.norm(J, axis=0)
    zero = np.flatnonzero(col < 1e-10)
    if len(zero):
        return zero.tolist()
    Jn = J / col
    u, s, vt = np.linalg.svd(Jn, full_matrices=False)
    if s[-1] > 1e-7 * s[0]:
        return []
    null = np.abs(vt[-1])
    return np.flatnonzero(null > 0.1 * null.max()).tolist()


# ------------------------------------------------------------------ results
@dataclass
class Estimates:
    model: SEMModel
    spec: Spec
    estimator: str
    n: int
    n_mult: float              # N (normal likelihood) or N-1 (Wishart)
    S: np.ndarray              # sample covariance used in F
    S_n: np.ndarray            # covariance with divisor N (log-likelihood)
    theta: np.ndarray
    F: float
    converged: bool
    iterations: int
    max_grad: float
    Sigma: np.ndarray
    B: np.ndarray
    Omega: np.ndarray
    A: np.ndarray
    Smat: np.ndarray
    cov: np.ndarray | None = None
    se_type: str = "expected"
    chi2: float = float("nan")
    df: int = 0
    robust: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def sample_moments(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(X)
    Xc = X - X.mean(axis=0)
    S_n = Xc.T @ Xc / n
    return S_n, S_n * n / (n - 1)


def estimate(model: SEMModel, X: np.ndarray, estimator: str = "ML",
             likelihood: str = "normal", information: str = "expected",
             robust: bool = False, max_iter: int = 1000) -> Estimates:
    """Fit ``model`` to the data matrix ``X`` (columns in ``model.observed`` order)."""
    n, p = X.shape
    if n < 3:
        raise SEMError("Too few cases for SEM after listwise deletion.")
    spec = Spec.from_model(model)
    S_n, S_u = sample_moments(X)
    if np.any(np.diag(S_n) <= 0):
        const = [v for v, s in zip(model.observed, np.diag(S_n)) if s <= 0]
        raise SEMError(f"Variable(s) with zero variance: {', '.join(const)}.")
    wishart = likelihood == "wishart"
    S = S_u if wishart else S_n
    n_mult = n - 1 if wishart else n
    moments = p * (p + 1) // 2
    df = moments - spec.q
    if df < 0:
        raise SEMError(f"The model is not identified: {spec.q} free parameters but only "
                       f"{moments} distinct sample moments (df = {df}). Fix parameters or "
                       f"add indicators.")
    theta0 = start_values(model, spec, S)
    bad = identification_check(spec, theta0)
    if bad:
        names = _names_for(model, bad)
        raise SEMError("The model is not identified (the information matrix is singular). "
                       "Parameters involved: " + ", ".join(names) + ". Give each latent "
                       "variable a scale (marker loading = 1 or variance = 1) and check "
                       "single-indicator constructs.")
    disc = Discrepancy(estimator, S[None])
    Sig0 = spec.implied(theta0)[0]
    if not np.isfinite(disc.value(Sig0))[0]:
        theta0 = _repair_start(model, spec, theta0, S)
    st = fisher_scoring(spec, disc, theta0, max_iter=max_iter)
    if not st.converged[0]:
        st = _polish(spec, disc, st)
    theta = st.theta[0]
    Sig, B, Om = spec.implied(theta)
    A, Sm = spec.matrices(theta)
    est = Estimates(model, spec, estimator, n, n_mult, S, S_n, theta, float(st.F[0]),
                    bool(st.converged[0]), int(st.iterations[0]),
                    float(np.max(np.abs(st.grad[0]))) if spec.q else 0.0,
                    Sig[0], B[0], Om[0], A[0], Sm[0], df=df)
    est.chi2 = n_mult * est.F
    est.cov, est.se_type = covariance(est, information)
    if robust and estimator == "ML":
        est.robust = satorra_bentler(est, X)
        est.cov = est.robust["cov"]
        est.se_type = "robust.sem"
    bad = identification_check(spec, theta)
    if bad:
        est.warnings.append("The model may not be empirically identified at the solution "
                            "(singular information matrix); parameters involved: "
                            + ", ".join(_names_for(model, bad)) + ".")
    return est


def _names_for(model: SEMModel, idx: list[int]) -> list[str]:
    out = []
    for k in idx:
        prm = next((p for p in model.params if p.index == k), None)
        if prm is not None:
            out.append(prm.name)
    return out


def _repair_start(model, spec, theta, S):
    """Shrink covariances / paths until the start values give a PD implied matrix."""
    t = theta.copy()
    off = [p.index for p in model.params if p.free and not (p.op == "~~" and p.lhs == p.rhs)
           and p.op != "=~"]
    for _ in range(30):
        t[off] *= 0.5
        if _pd(spec.implied(t)[0])[0]:
            return t
    for prm in model.params:
        if prm.free and prm.op == "~~" and prm.lhs == prm.rhs:
            t[prm.index] = abs(t[prm.index]) + 0.1
    return t


def _polish(spec: Spec, disc: Discrepancy, st: FitState) -> FitState:
    """Quasi-Newton fallback when Fisher scoring stalls (e.g. near a Heywood case)."""
    from scipy.optimize import minimize

    def fun(t):
        Sig, B, Om = spec.implied(t)
        f = disc.value(Sig)[0]
        if not np.isfinite(f):
            return 1e10, np.zeros_like(t)
        g, _ = disc.grad_hess(Sig, spec.jacobian(B, Om))
        return f, g[0]

    res = minimize(fun, st.theta[0], jac=True, method="BFGS",
                   options={"gtol": 1e-7, "maxiter": 5000})
    if res.fun <= st.F[0]:
        Sig, B, Om = spec.implied(res.x)
        g, _ = disc.grad_hess(Sig, spec.jacobian(B, Om))
        conv = bool(np.max(np.abs(g)) < 1e-5)
        return FitState(res.x[None], np.array([res.fun]), np.array([conv]),
                        st.iterations + int(res.nit), g)
    return st


# ------------------------------------------------------------------ standard errors
def covariance(est: Estimates, information: str = "expected"):
    spec = est.spec
    if spec.q == 0:
        return np.zeros((0, 0)), information
    Sig, B, Om = spec.implied(est.theta)
    dS = spec.jacobian(B, Om)[0]
    N = est.n_mult
    try:
        if est.estimator == "ULS":
            H = np.einsum("kij,lji->kl", dS, dS)
            G = np.einsum("kij,jl->kil", dS, Sig[0])
            M = 2 * np.einsum("kij,lji->kl", G, G)
            Hi = np.linalg.inv(H)
            return Hi @ M @ Hi / N, "sandwich (normal theory)"
        if information == "observed" and est.estimator == "ML":
            H = numerical_hessian(spec, est.S, est.theta, est.estimator)
            return 2 / N * np.linalg.inv(H), "observed"
        disc = Discrepancy(est.estimator, est.S[None])
        _, H = disc.grad_hess(Sig, dS[None])
        return 2 / N * np.linalg.inv(H[0]), "expected"
    except np.linalg.LinAlgError:
        est.warnings.append("The information matrix is singular; standard errors could not be "
                            "computed.")
        return None, information


def numerical_hessian(spec: Spec, S: np.ndarray, theta: np.ndarray, estimator: str):
    q = len(theta)
    h = 1e-5 * np.maximum(1, np.abs(theta))
    pts = np.concatenate([theta + np.diag(h), theta - np.diag(h)])
    Sig, B, Om = spec.implied(pts)
    discb = Discrepancy(estimator, np.repeat(S[None], len(pts), axis=0))
    g, _ = discb.grad_hess(Sig, spec.jacobian(B, Om))
    H = (g[:q] - g[q:]) / (2 * h[:, None])
    return 0.5 * (H + H.T)


def satorra_bentler(est: Estimates, X: np.ndarray) -> dict:
    """MLM: robust sandwich covariance of the estimates and the Satorra-Bentler scaled
    test statistic (Gamma from fourth-order moments, divisor N, as lavaan)."""
    spec = est.spec
    n, p = X.shape
    r, c = vech_index(p)
    Xc = X - X.mean(axis=0)
    Z = Xc[:, r] * Xc[:, c]
    Gamma = np.cov(Z, rowvar=False, bias=True)
    Sig, B, Om = spec.implied(est.theta)
    Delta = spec.jacobian(B, Om)[0][:, r, c].T
    V = _ml_weight_vech(Sig[0])
    info = Delta.T @ V @ Delta
    info_i = np.linalg.inv(info)
    cov = info_i @ (Delta.T @ V @ Gamma @ V @ Delta) @ info_i / n
    U = V - V @ Delta @ info_i @ Delta.T @ V
    df = est.df
    c_sb = float(np.trace(U @ Gamma) / df) if df > 0 else float("nan")
    return {"cov": cov, "scale": c_sb, "chi2": est.chi2 / c_sb if df > 0 else float("nan"),
            "Gamma": Gamma}


def _ml_weight_vech(Sig: np.ndarray) -> np.ndarray:
    p = len(Sig)
    D = duplication(p)
    Si = np.linalg.inv(Sig)
    return 0.5 * D.T @ np.kron(Si, Si) @ D


def sb_scale_baseline(Sig_b: np.ndarray, Gamma: np.ndarray, df_b: int) -> float:
    """Scaling factor of the independence model (no free off-diagonal parameters)."""
    p = len(Sig_b)
    r, c = vech_index(p)
    V = _ml_weight_vech(Sig_b)
    diag = np.flatnonzero(r == c)
    Delta = np.zeros((len(r), len(diag)))
    Delta[diag, np.arange(len(diag))] = 1
    U = V - V @ Delta @ np.linalg.inv(Delta.T @ V @ Delta) @ Delta.T @ V
    return float(np.trace(U @ Gamma) / df_b)


# ------------------------------------------------------------------ fit measures
def baseline_model(model: SEMModel, fixed_x: bool = True) -> SEMModel:
    """Independence model: observed variances only (plus the covariances among observed
    exogenous covariates when ``fixed_x``, as lavaan does)."""
    params = []
    k = 0
    for v in model.observed:
        params.append(Param(v, "~~", v, index=k))
        k += 1
    exo = [v for v in model.observed if v in model.exogenous]
    if fixed_x and len(exo) > 1:
        for i, a in enumerate(exo):
            for b in exo[i + 1:]:
                params.append(Param(a, "~~", b, index=k))
                k += 1
    return SEMModel(params, list(model.observed), [], [], {}, [], list(model.observed), k)


def fit_baseline(est: Estimates, fixed_x: bool = True) -> tuple[float, int, np.ndarray]:
    bm = baseline_model(est.model, fixed_x)
    spec = Spec.from_model(bm)
    theta0 = np.array([est.S[spec.pos[p.lhs], spec.pos[p.rhs]] for p in bm.params])
    disc = Discrepancy(est.estimator, est.S[None])
    st = fisher_scoring(spec, disc, theta0, max_iter=200)
    p = len(est.model.observed)
    return est.n_mult * float(st.F[0]), p * (p + 1) // 2 - spec.q, spec.implied(st.theta)[0][0]


def rmsea_ci(chi2: float, df: int, n: float, level: float = 0.90) -> tuple[float, float]:
    if df <= 0 or not np.isfinite(chi2):
        return float("nan"), float("nan")
    from scipy.optimize import brentq

    def lam(target):
        f = lambda nc: stats.ncx2.cdf(chi2, df, nc) - target  # noqa: E731
        if f(1e-10) < 0:
            return 0.0
        hi = max(chi2, 1.0)
        while f(hi) > 0:
            hi *= 2
        return brentq(f, 1e-10, hi, xtol=1e-10)

    lo = lam(0.5 + level / 2)
    up = lam(0.5 - level / 2)
    return math.sqrt(lo / (df * n)), math.sqrt(up / (df * n))


def fit_measures(est: Estimates, fixed_x: bool = True) -> dict:
    chi2, df, N = est.chi2, est.df, est.n_mult
    p = est.spec.p
    q = est.spec.q
    out: dict = {"chi2": chi2, "df": df, "npar": q, "moments": p * (p + 1) // 2}
    out["p"] = float(stats.chi2.sf(chi2, df)) if df > 0 else float("nan")
    out["cmin_df"] = chi2 / df if df > 0 else float("nan")
    chi_b, df_b, Sig_b = fit_baseline(est, fixed_x)
    out["chi2_base"], out["df_base"] = chi_b, df_b
    out["p_base"] = float(stats.chi2.sf(chi_b, df_b)) if df_b > 0 else float("nan")
    out.update(_incremental(chi2, df, chi_b, df_b))
    if df > 0:
        out["rmsea"] = math.sqrt(max(chi2 - df, 0) / (df * N))
        out["rmsea_lo"], out["rmsea_hi"] = rmsea_ci(chi2, df, N)
        out["pclose"] = float(stats.ncx2.sf(chi2, df, 0.05 ** 2 * df * N))
    else:
        out["rmsea"] = out["rmsea_lo"] = out["rmsea_hi"] = out["pclose"] = float("nan")
    S, Sig = est.S, est.Sigma
    d = np.sqrt(np.diag(S))
    r, c = vech_index(p)
    resid = (S - Sig) / np.outer(d, d)
    out["srmr"] = math.sqrt(np.mean(resid[r, c] ** 2))
    out["rmr"] = math.sqrt(np.mean((S - Sig)[r, c] ** 2))
    if est.estimator == "ML":
        Si = np.linalg.inv(Sig)
        P = Si @ S
        out["gfi"] = 1 - np.trace((P - np.eye(p)) @ (P - np.eye(p))) / np.trace(P @ P)
    elif est.estimator == "GLS":
        Q = np.eye(p) - np.linalg.solve(S, Sig)
        out["gfi"] = 1 - np.trace(Q @ Q) / p
    else:
        R = S - Sig
        out["gfi"] = 1 - np.trace(R @ R) / np.trace(S @ S)
    out["agfi"] = 1 - (p * (p + 1) / (2 * df)) * (1 - out["gfi"]) if df > 0 else float("nan")
    out["pgfi"] = out["gfi"] * df / out["moments"] if df > 0 else float("nan")
    if est.estimator == "ML":
        n = est.n
        Sn = est.S_n
        ld = np.linalg.slogdet(Sig)[1]
        ll = -n / 2 * (p * math.log(2 * math.pi) + ld + np.trace(Sn @ np.linalg.inv(Sig)))
        ll1 = -n / 2 * (p * math.log(2 * math.pi) + np.linalg.slogdet(Sn)[1] + p)
        out["logl"], out["logl_h1"] = ll, ll1
        out["aic"] = -2 * ll + 2 * q
        out["bic"] = -2 * ll + q * math.log(n)
        out["bic2"] = -2 * ll + q * math.log((n + 2) / 24)
        out["caic_amos"] = chi2 + q * (math.log(n) + 1)
        out["aic_amos"] = chi2 + 2 * q
        out["ecvi"] = (chi2 + 2 * q) / (n - 1)
    if est.robust:
        c = est.robust["scale"]
        out["chi2_scaled"] = est.robust["chi2"]
        out["scale"] = c
        out["p_scaled"] = float(stats.chi2.sf(out["chi2_scaled"], df)) if df > 0 else np.nan
        c_b = sb_scale_baseline(Sig_b, est.robust["Gamma"], df_b) if df_b > 0 else 1.0
        sc = _incremental(out["chi2_scaled"], df, chi_b / c_b, df_b)
        out.update({f"{k}_scaled": v for k, v in sc.items()})
        out["chi2_base_scaled"] = chi_b / c_b
        if df > 0:
            x = out["chi2_scaled"]
            out["rmsea_scaled"] = math.sqrt(max(x - df, 0) / (df * N))
            out["rmsea_lo_scaled"], out["rmsea_hi_scaled"] = rmsea_ci(x, df, N)
    return out


def _incremental(chi2, df, chi_b, df_b) -> dict:
    out = {}
    d0, db = max(chi2 - df, 0), max(chi_b - df_b, 0)
    out["cfi"] = 1 - d0 / max(db, d0) if max(db, d0) > 0 else 1.0
    if df > 0 and df_b > 0 and chi_b / df_b != 1:
        out["tli"] = (chi_b / df_b - chi2 / df) / (chi_b / df_b - 1)
        out["rfi"] = (chi_b / df_b - chi2 / df) / (chi_b / df_b)
    else:
        out["tli"] = out["rfi"] = float("nan")
    out["nfi"] = 1 - chi2 / chi_b if chi_b > 0 else float("nan")
    out["ifi"] = (chi_b - chi2) / (chi_b - df) if chi_b != df else float("nan")
    out["pnfi"] = out["nfi"] * df / df_b if df_b > 0 else float("nan")
    return out


# ------------------------------------------------------------------ modification indices
@dataclass
class MI:
    lhs: str
    op: str
    rhs: str
    mi: float
    epc: float
    sepc: float


def modification_indices(est: Estimates) -> list[MI]:
    model, spec = est.model, est.spec
    free = {(p.lhs, p.op, p.rhs) for p in model.params}
    free |= {(p.rhs, p.op, p.lhs) for p in model.params if p.op == "~~"}
    ind_of: dict[str, set] = {}
    for f, xs in model.indicators.items():
        for x in xs:
            ind_of.setdefault(x, set()).add(f)
    obs_ind = [x for x in model.observed if x in ind_of]
    cands: list[tuple[str, str, str]] = []
    for f in model.latent:
        for x in obs_ind:
            if (f, "=~", x) not in free and x not in model.indicators.get(f, []):
                cands.append((f, "=~", x))
    endo_obs = [v for v in model.observed if v in model.endogenous]
    for i, a in enumerate(endo_obs):
        for b in endo_obs[i + 1:]:
            if (a, "~~", b) not in free:
                cands.append((a, "~~", b))
    struct = model.structural()
    lat_like = [v for v in struct if v in model.latent]
    for i, a in enumerate(lat_like):
        for b in lat_like[i + 1:]:
            if (a, "~~", b) not in free and not _has_path(model, a, b):
                cands.append((a, "~~", b))
    for y in struct:
        if y in model.exogenous:
            continue
        for x in struct:
            if x == y or (y, "~", x) in free or (x, "~", y) in free or (x, "~~", y) in free:
                continue
            if y in model.indicators.get(x, []) or x in model.indicators.get(y, []):
                continue
            cands.append((y, "~", x))
    if not cands or est.cov is None or spec.q == 0:
        return []
    extra = [Param(a, op, b, index=spec.q + k) for k, (a, op, b) in enumerate(cands)]
    for prm in extra:
        prm.free = True
    big = Spec(list(model.params) + extra, model.observed, model.latent, spec.q + len(extra))
    theta = np.concatenate([est.theta, np.zeros(len(extra))])
    Sig, B, Om = big.implied(theta)
    dS = big.jacobian(B, Om)
    disc = Discrepancy(est.estimator if est.estimator != "ULS" else "ML", est.S[None])
    g, H = disc.grad_hess(Sig, dS)
    g, H = g[0], H[0]
    q = spec.q
    Htt = H[:q, :q]
    try:
        Hi = np.linalg.inv(Htt)
    except np.linalg.LinAlgError:
        return []
    Hct = H[q:, :q]
    V = np.diag(H[q:, q:]) - np.einsum("ck,kl,cl->c", Hct, Hi, Hct)
    gc = g[q:]
    N = est.n_mult
    out = []
    sd = np.sqrt(np.clip(np.diag(est.Omega), 1e-300, None))
    resid_var = np.diag(est.Smat)
    for k, (a, op, b) in enumerate(cands):
        if V[k] <= 1e-10:
            continue
        mi = N * gc[k] ** 2 / (2 * V[k])
        epc = -gc[k] / V[k]
        ia, ib = spec.pos[a], spec.pos[b]
        if op == "=~":
            sepc = epc * sd[ia] / sd[ib]
        elif op == "~":
            sepc = epc * sd[ib] / sd[ia]
        else:
            den = math.sqrt(abs(resid_var[ia] * resid_var[ib]))
            sepc = epc / den if den > 0 else float("nan")
        out.append(MI(a, op, b, float(mi), float(epc), float(sepc)))
    out.sort(key=lambda m: -m.mi)
    return out


def _has_path(model: SEMModel, a: str, b: str) -> bool:
    return any(p.op == "~" and {p.lhs, p.rhs} == {a, b} for p in model.params)


# ------------------------------------------------------------------ bootstrap
def bootstrap(est: Estimates, X: np.ndarray, n_boot: int, seed: int | None,
              chunk: int = 250, max_iter: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Nonparametric bootstrap: refit to resampled data, vectorised over resamples.
    Returns (theta_boot for converged resamples, converged mask)."""
    rng = np.random.default_rng(seed)
    n = len(X)
    thetas = np.full((n_boot, est.spec.q), np.nan)
    conv = np.zeros(n_boot, bool)
    wishart = est.n_mult != est.n
    for start in range(0, n_boot, chunk):
        nb = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(nb, n))
        Xb = X[idx]
        Xc = Xb - Xb.mean(axis=1, keepdims=True)
        Sb = np.einsum("bni,bnj->bij", Xc, Xc) / (n - 1 if wishart else n)
        sign, _ = np.linalg.slogdet(Sb)
        ok = sign > 0 if est.estimator != "ULS" else np.ones(nb, bool)
        if not ok.any():
            continue
        disc = Discrepancy(est.estimator, Sb[ok])
        th0 = np.repeat(est.theta[None], ok.sum(), axis=0)
        try:
            st = fisher_scoring(est.spec, disc, th0, max_iter=max_iter, tol=1e-6)
        except SEMError:
            continue
        rows = np.flatnonzero(ok) + start
        thetas[rows] = st.theta
        conv[rows] = st.converged
    return thetas, conv


def percentile_ci(boot: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    a = (1 - level) / 2
    return (np.nanquantile(boot, a, axis=0), np.nanquantile(boot, 1 - a, axis=0))


def bc_ci(boot: np.ndarray, est: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Bias-corrected percentile interval (Efron), as AMOS reports."""
    a = (1 - level) / 2
    valid = np.isfinite(boot)
    nvalid = valid.sum(axis=0)
    below = np.where(valid, boot < est[None], False).sum(axis=0)
    ties = np.where(valid, boot == est[None], False).sum(axis=0)
    prop = (below + 0.5 * ties) / np.maximum(nvalid, 1)
    prop = np.clip(prop, 1 / (nvalid + 1), nvalid / (nvalid + 1))
    z0 = stats.norm.ppf(prop)
    za = stats.norm.ppf(1 - a)
    lo_q = stats.norm.cdf(2 * z0 - za)
    hi_q = stats.norm.cdf(2 * z0 + za)
    lo = np.array([np.nanquantile(boot[:, k], lo_q[k]) if nvalid[k] else np.nan
                   for k in range(boot.shape[1])])
    hi = np.array([np.nanquantile(boot[:, k], hi_q[k]) if nvalid[k] else np.nan
                   for k in range(boot.shape[1])])
    return lo, hi


def bc_p(boot: np.ndarray, est: np.ndarray) -> np.ndarray:
    """Two-sided bootstrap p for H0: value = 0 from the percentile distribution."""
    valid = np.isfinite(boot)
    nvalid = np.maximum(valid.sum(axis=0), 1)
    pos = np.where(valid, boot > 0, False).sum(axis=0) / nvalid
    neg = np.where(valid, boot < 0, False).sum(axis=0) / nvalid
    return np.minimum(1.0, 2 * np.minimum(pos, neg))


def delta_cov(fun, theta: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Delta-method SEs of a vector function evaluated on batches of theta."""
    q = len(theta)
    h = 1e-6 * np.maximum(1, np.abs(theta))
    pts = np.concatenate([theta + np.diag(h), theta - np.diag(h)])
    vals = fun(pts)
    J = (vals[:q] - vals[q:]) / (2 * h[:, None])
    return np.sqrt(np.clip(np.einsum("kj,kl,lj->j", J, cov, J), 0, None))


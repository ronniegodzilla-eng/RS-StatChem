"""Analyze > Mixed Models > Generalized Linear (SPSS GENLINMIXED), long-format data.

Targets: normal, binomial (binary target or events/trials), Poisson, negative binomial,
gamma and inverse Gaussian with the GENLINMIXED links (identity, log, logit, probit,
complementary log-log, negative log-log, log-complement, cauchit, power). Fixed effects are
factors and covariates (main effects by default, or a custom model with interactions);
random-effect blocks give random intercepts and/or random slopes (covariates) or factor
effects for a subject combination with a Variance Components, Scaled Identity, Diagonal or
Unstructured covariance; nested subjects (school, school*class) are separate blocks.

Estimation follows the IBM SPSS Statistics Algorithms chapter "Generalized linear mixed models
(GENLINMIXED)": the doubly iterative linearisation method REPL-PQL (residual pseudo-likelihood
with the empirical Bayes random effects in the Taylor expansion, Wolfinger & O'Connell 1993).
Each outer iteration builds the pseudo target ``v = g(mu) - O + (y - mu) g'(mu)`` with weights
``w = omega / (V(mu) g'(mu)^2)`` and fits the weighted linear mixed model by REML; R = phi I
with phi = 1 for the discrete distributions and phi estimated for the continuous ones; the
negative binomial ancillary k is the ML estimate of the model without random effects. Outer
iterations stop when ``max 2|t_j - t_j-1| / (|t_j| + |t_j-1|) < 1e-6`` over the covariance
parameters. Covariance parameter standard errors are ``2 H^-1`` of the -2 REPL objective,
variance confidence intervals are log-transformed, fixed-effect t tests use the residual
degrees of freedom ``N - rank(X)`` (IBM's default), the model-based or robust (sandwich by
subject) covariance of the fixed effects, Type III F tests (with the Corrected Model test),
AICC/BIC from the restricted log pseudo-likelihood, Nakagawa-Schielzeth pseudo-R2 and ICCs
(delta method, trigamma for log links of Poisson / negative binomial / gamma), estimated means
on the link or the original scale (delta method) with LSD / sequential Bonferroni /
sequential Sidak adjusted pairwise, simple or deviation contrasts, PQL-type predictions and
the classification table for binary targets.

An optional **Laplace** approximation (maximum likelihood, as R ``lme4::glmer`` with
``nAGQ = 1``) is offered for comparison; it is not an SPSS estimation method and is labelled
as such in the output.
"""
from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import linalg, optimize, sparse, special, stats

from rsstatchem.core.dataset import NUMERIC, Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, fmt_p, numeric, pct, sup, value_text
from rsstatchem.core.stats.glm_model import TermSpec, main_effects

DISTRIBUTIONS = {"normal": "Normal", "binomial": "Binomial", "poisson": "Poisson",
                 "negbin": "Negative binomial", "gamma": "Gamma",
                 "igauss": "Inverse Gaussian"}
DISCRETE = {"binomial", "poisson", "negbin"}
LINKS = {"identity": "Identity", "log": "Log", "logit": "Logit", "probit": "Probit",
         "cloglog": "Complementary log-log", "nloglog": "Negative log-log",
         "logc": "Log complement", "cauchit": "Cauchit", "power": "Power"}
PRESETS = {  # GENLINMIXED "Target Distribution and Relationship (Link) with the Linear Model"
    "linear": ("normal", "identity", "Linear model"),
    "gamma": ("gamma", "log", "Gamma regression"),
    "loglinear": ("poisson", "log", "Loglinear"),
    "negbin": ("negbin", "log", "Negative binomial regression"),
    "binary_logistic": ("binomial", "logit", "Binary logistic regression"),
    "binary_probit": ("binomial", "probit", "Binary probit"),
    "interval_censored": ("binomial", "cloglog", "Interval censored survival"),
}
CANONICAL = {"normal": "identity", "binomial": "logit", "poisson": "log", "negbin": "log",
             "gamma": "power", "igauss": "power"}
COVTYPES = {"VC": "Variance components", "ID": "Scaled identity", "DIAG": "Diagonal",
            "UN": "Unstructured"}
ADJUST = {"lsd": "Least significant difference", "seqbonferroni": "Sequential Bonferroni",
          "seqsidak": "Sequential Sidak"}
CONTRASTS = {"none": "None", "pairwise": "Pairwise", "simple": "Simple",
             "deviation": "Deviation"}
METHODS = {"pql": "Residual pseudo-likelihood (REPL-PQL)",
           "laplace": "Laplace approximation (maximum likelihood; not an SPSS method)"}
EXP_LINKS = {("binomial", "logit"), ("poisson", "log"), ("negbin", "log")}


# ====================================================================== specifications
@dataclass
class RandomBlock:
    """One GENLINMIXED RANDOM subcommand: effects | SUBJECTS=... COVARIANCE_TYPE=..."""
    effects: list = field(default_factory=list)    # covariate or factor names
    intercept: bool = True
    subjects: list = field(default_factory=list)   # subject combination (outer to inner)
    covtype: str = "VC"
    predictions: bool = False

    def label(self) -> str:
        parts = (["Intercept"] if self.intercept else []) + list(self.effects)
        return " + ".join(parts) or "(none)"


@dataclass
class GlmmResult:
    section: Section
    saved: dict = field(default_factory=dict)
    models: list = field(default_factory=list)
    fit: object = None


# ====================================================================== distributions
def _link(name: str, power: float = 1.0):
    from statsmodels.genmod.families import links as L

    if name == "power":
        return L.Log() if abs(power) < 2.2e-16 else L.Power(power)
    if name == "logc":
        return _LogComplement()
    return {"identity": L.Identity, "log": L.Log, "logit": L.Logit, "probit": L.Probit,
            "cloglog": L.CLogLog, "nloglog": L.LogLog, "cauchit": L.Cauchy}[name]()


class _LogComplement:
    """eta = log(1 - mu) (GENLINMIXED's log-complement link)."""

    def __call__(self, mu):
        return np.log1p(-mu)

    def inverse(self, eta):
        return 1 - np.exp(eta)

    def deriv(self, mu):
        return -1 / (1 - mu)

    def inverse_deriv(self, eta):
        return -np.exp(eta)


def _variance(dist: str, mu, k: float = 0.0):
    if dist == "normal":
        return np.ones_like(mu)
    if dist == "binomial":
        return mu * (1 - mu)
    if dist == "poisson":
        return mu
    if dist == "negbin":
        return mu + k * mu * mu
    if dist == "gamma":
        return mu * mu
    return mu ** 3


def _clip_mu(dist: str, mu):
    if dist == "binomial":
        return np.clip(mu, 1e-10, 1 - 1e-10)
    if dist == "normal":
        return mu
    return np.clip(mu, 1e-10, None)


def _loglik_obs(dist, y, mu, m, phi=1.0, k=0.0, omega=None):
    """Full conditional log-likelihood of each record (y is a proportion for binomial)."""
    omega = np.ones_like(y) if omega is None else omega
    if dist == "binomial":
        r = y * m
        return (special.gammaln(m + 1) - special.gammaln(r + 1) - special.gammaln(m - r + 1)
                + special.xlogy(r, mu) + special.xlogy(m - r, 1 - mu))
    if dist == "poisson":
        return special.xlogy(y, mu) - mu - special.gammaln(y + 1)
    if dist == "negbin":
        if k <= 0:
            return special.xlogy(y, mu) - mu - special.gammaln(y + 1)
        a = 1 / k
        return (special.gammaln(y + a) - special.gammaln(a) - special.gammaln(y + 1)
                + special.xlogy(y, k * mu) - (y + a) * np.log1p(k * mu))
    if dist == "normal":
        s2 = phi / omega
        return -0.5 * (np.log(2 * np.pi * s2) + (y - mu) ** 2 / s2)
    if dist == "gamma":
        a = omega / phi
        return a * np.log(a * y / mu) - a * y / mu - np.log(y) - special.gammaln(a)
    s2 = phi / omega  # inverse Gaussian
    return -0.5 * (np.log(2 * np.pi * s2 * y ** 3) + (y - mu) ** 2 / (s2 * y * mu * mu))


def _nb_k(y, X, offset) -> float:
    """ML negative binomial ancillary k of the model without random effects (IBM)."""
    from statsmodels.discrete.discrete_model import NegativeBinomial

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = NegativeBinomial(y, X, offset=offset).fit(disp=0, maxiter=200)
            return float(max(res.params[-1], 1e-8))
        except Exception:  # noqa: BLE001
            return 1.0


# ====================================================================== random-effects design
class _ZDesign:
    """Random-effects design: blocks with per-subject columns, covariance parameterisation."""

    def __init__(self, blocks: list[dict], n: int):
        self.blocks = blocks          # dict: Zs (n x qb), codes (n,), m, covtype, groups, names
        self.n = n
        mats, self.offsets = [], []
        off = 0
        for b in blocks:
            qb, m = b["Zs"].shape[1], b["m"]
            rows = np.repeat(np.arange(n), qb)
            cols = (b["codes"][:, None] * qb + np.arange(qb)[None, :]).ravel()
            Zb = sparse.csc_matrix((b["Zs"].ravel(), (rows, cols)), shape=(n, m * qb))
            mats.append(Zb)
            self.offsets.append(off)
            off += m * qb
        self.q = off
        self.Z = sparse.hstack(mats).tocsc() if mats else sparse.csc_matrix((n, 0))
        # parameter layout
        self.slices = []
        pos = 0
        for b in blocks:
            k = self._nparam(b)
            self.slices.append(slice(pos, pos + k))
            pos += k
        self.n_theta = pos

    @staticmethod
    def _nparam(b) -> int:
        qb = b["Zs"].shape[1]
        if b["covtype"] == "ID":
            return 1
        if b["covtype"] == "VC":
            return len(b["groups"])
        if b["covtype"] == "DIAG":
            return qb
        return qb * (qb + 1) // 2

    def lower_bounds(self) -> list:
        lb = []
        for b in self.blocks:
            qb = b["Zs"].shape[1]
            if b["covtype"] == "UN":
                for i in range(qb):
                    for j in range(i + 1):
                        lb.append(0.0 if i == j else None)
            else:
                lb += [0.0] * self._nparam(b)
        return lb

    def chol_block(self, k: int, t: np.ndarray) -> np.ndarray:
        """Relative Cholesky factor of block k from its parameter slice."""
        b = self.blocks[k]
        qb = b["Zs"].shape[1]
        if b["covtype"] == "ID":
            return np.eye(qb) * t[0]
        if b["covtype"] == "VC":
            d = np.zeros(qb)
            for g, cols in enumerate(b["groups"]):
                d[cols] = t[g]
            return np.diag(d)
        if b["covtype"] == "DIAG":
            return np.diag(t)
        L = np.zeros((qb, qb))
        L[np.tril_indices(qb)] = t
        return L

    def lam(self, theta) -> sparse.csc_matrix:
        mats = []
        for k, b in enumerate(self.blocks):
            L = self.chol_block(k, theta[self.slices[k]])
            mats.append(sparse.kron(sparse.identity(b["m"]), sparse.csr_matrix(L)))
        if not mats:
            return sparse.csc_matrix((0, 0))
        return sparse.block_diag(mats).tocsc()

    # natural parameters (variances / covariances, IBM order)
    def natural(self, theta, phi: float = 1.0) -> np.ndarray:
        out = []
        for k, b in enumerate(self.blocks):
            L = self.chol_block(k, theta[self.slices[k]])
            G = phi * L @ L.T
            out += self._from_G(b, G)
        return np.array(out, float)

    def _from_G(self, b, G) -> list:
        qb = G.shape[0]
        if b["covtype"] == "ID":
            return [G[0, 0]]
        if b["covtype"] == "VC":
            return [G[cols[0], cols[0]] for cols in b["groups"]]
        if b["covtype"] == "DIAG":
            return list(np.diag(G))
        return [G[i, j] for i in range(qb) for j in range(i + 1)]

    def G_from_natural(self, k: int, nat: np.ndarray) -> np.ndarray:
        b = self.blocks[k]
        qb = b["Zs"].shape[1]
        if b["covtype"] == "ID":
            return np.eye(qb) * nat[0]
        if b["covtype"] == "VC":
            d = np.zeros(qb)
            for g, cols in enumerate(b["groups"]):
                d[cols] = nat[g]
            return np.diag(d)
        if b["covtype"] == "DIAG":
            return np.diag(nat)
        G = np.zeros((qb, qb))
        G[np.tril_indices(qb)] = nat
        return G + np.tril(G, -1).T

    def theta_from_natural(self, nat: np.ndarray, phi: float = 1.0) -> np.ndarray | None:
        out = []
        for k, b in enumerate(self.blocks):
            G = self.G_from_natural(k, nat[self.slices[k]]) / phi
            if b["covtype"] == "UN":
                try:
                    L = np.linalg.cholesky(G)
                except np.linalg.LinAlgError:
                    return None
                out += list(L[np.tril_indices(G.shape[0])])
            else:
                d = self._from_G(b, G)
                if min(d) < 0:
                    return None
                out += list(np.sqrt(d))
        return np.array(out, float)

    def param_names(self) -> list[tuple[int, str, bool]]:
        """(block index, label, is_variance) per natural parameter."""
        out = []
        for k, b in enumerate(self.blocks):
            names = b["names"]
            if b["covtype"] == "ID":
                out.append((k, "Variance", True))
            elif b["covtype"] == "VC":
                for g in b["gnames"]:
                    out.append((k, f"Var({g})", True))
            elif b["covtype"] == "DIAG":
                for nme in names:
                    out.append((k, f"Var({nme})", True))
            else:
                qb = len(names)
                for i in range(qb):
                    for j in range(i + 1):
                        out.append((k, f"Var({names[i]})" if i == j else
                                    f"Cov({names[i]}, {names[j]})", i == j))
        return out


# ====================================================================== weighted LMM (REML)
class _PseudoLMM:
    """Weighted linear mixed model v = X b + Z g + e, Var(e) = phi W^-1, Var(g) = phi LL'
    (continuous) or Var(e) = W^-1, Var(g) = LL' (phi fixed at 1)."""

    def __init__(self, X, zd: _ZDesign, estimate_phi: bool):
        self.X = X
        self.zd = zd
        self.n, self.p = X.shape
        self.estimate_phi = estimate_phi
        self.v = None
        self.w = None

    def set_data(self, v, w):
        self.v = v
        self.w = w

    def _parts(self, theta):
        zd = self.zd
        X, v, w = self.X, self.v, self.w
        if zd.q:
            Zt = (zd.Z @ zd.lam(theta)).tocsc()
            WZt = Zt.multiply(w[:, None]).tocsc()
            A = (Zt.T @ WZt).toarray() + np.eye(zd.q)
            cA = linalg.cho_factor(A, lower=True)
            logdetA = 2 * np.sum(np.log(np.diag(cA[0])))
            ZWX = np.asarray(WZt.T @ X)
            ZWv = np.asarray(WZt.T @ v).ravel()
            AiZWX = linalg.cho_solve(cA, ZWX)
            AiZWv = linalg.cho_solve(cA, ZWv)
        else:
            Zt = None
            logdetA = 0.0
            ZWX = np.zeros((0, self.p))
            ZWv = np.zeros(0)
            AiZWX = ZWX
            AiZWv = ZWv
            cA = None
        XWX = X.T @ (X * w[:, None])
        XWv = X.T @ (w * v)
        XVX = XWX - ZWX.T @ AiZWX
        XVv = XWv - ZWX.T @ AiZWv
        vVv = float(v @ (w * v) - ZWv @ AiZWv)
        logdetV = -float(np.sum(np.log(w))) + logdetA
        return dict(Zt=Zt, cA=cA, XVX=XVX, XVv=XVv, vVv=vVv, logdetV=logdetV, AiZWX=AiZWX,
                    AiZWv=AiZWv, ZWX=ZWX, ZWv=ZWv)

    def objective(self, theta, profile: bool = True, phi: float | None = None) -> float:
        """-2 restricted log pseudo-likelihood (phi profiled out when estimated)."""
        try:
            P = self._parts(theta)
            cX = linalg.cho_factor(P["XVX"], lower=True)
        except (linalg.LinAlgError, ValueError):
            return 1e300
        beta = linalg.cho_solve(cX, P["XVv"])
        rVr = P["vVv"] - float(beta @ P["XVv"])
        logdetX = 2 * np.sum(np.log(np.diag(cX[0])))
        nstar = self.n - self.p
        if not self.estimate_phi:
            return nstar * math.log(2 * math.pi) + P["logdetV"] + rVr + logdetX
        if profile:
            ph = max(rVr / nstar, 1e-300)
            return nstar * (1 + math.log(2 * math.pi * ph)) + P["logdetV"] + logdetX
        return (nstar * math.log(2 * math.pi) + nstar * math.log(phi) + P["logdetV"]
                + rVr / phi + logdetX)

    def solve(self, theta) -> dict:
        P = self._parts(theta)
        cX = linalg.cho_factor(P["XVX"], lower=True)
        beta = linalg.cho_solve(cX, P["XVv"])
        rVr = P["vVv"] - float(beta @ P["XVv"])
        phi = rVr / (self.n - self.p) if self.estimate_phi else 1.0
        C = phi * linalg.cho_solve(cX, np.eye(self.p))
        if self.zd.q:
            u = P["AiZWv"] - P["AiZWX"] @ beta
            gamma = np.asarray(self.zd.lam(theta) @ u).ravel()
        else:
            u = gamma = np.zeros(0)
        return dict(beta=beta, gamma=gamma, u=u, phi=phi, C=C, parts=P)

    def natural_objective(self, nat: np.ndarray) -> float:
        """-2 REPL as a function of the natural covariance parameters (+ phi last)."""
        phi = float(nat[-1]) if self.estimate_phi else 1.0
        if phi <= 0:
            return 1e300
        g = nat[:-1] if self.estimate_phi else nat
        theta = self.zd.theta_from_natural(g, phi)
        if theta is None:
            return 1e300
        return self.objective(theta, profile=False, phi=phi) if self.estimate_phi \
            else self.objective(theta)

    def fit_theta(self, theta0) -> np.ndarray:
        if self.zd.n_theta == 0:
            return np.zeros(0)
        bounds = [(lb, None) for lb in self.zd.lower_bounds()]
        best = None
        starts = [np.asarray(theta0, float)]
        if not np.any(theta0):
            starts.append(np.full(self.zd.n_theta, 0.5))
        for s in starts:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = optimize.minimize(self.objective, s, method="L-BFGS-B", bounds=bounds,
                                      options={"ftol": 1e-14, "gtol": 1e-9, "maxiter": 500})
                r2 = optimize.minimize(self.objective, r.x, method="Nelder-Mead",
                                       options={"xatol": 1e-9, "fatol": 1e-12,
                                                "maxiter": 4000})
            for cand in (r, r2):
                x = np.array([max(xi, lb[0]) if lb[0] is not None else xi
                              for xi, lb in zip(cand.x, bounds)])
                f = self.objective(x)
                if best is None or f < best[0]:
                    best = (f, x)
        return best[1]

    def robust_cov(self, sol, groups) -> np.ndarray:
        """Sandwich covariance of the fixed effects with subjects as independent clusters."""
        P = sol["parts"]
        X, w = self.X, self.w
        e = self.v - X @ sol["beta"]
        VX = X * w[:, None]
        if self.zd.q:
            Zt = P["Zt"]
            WZt = Zt.multiply(w[:, None]).tocsc()
            VX = VX - np.asarray(WZt @ P["AiZWX"])
        VX = VX / sol["phi"]
        scores = np.zeros((groups.max() + 1, self.p))
        np.add.at(scores, groups, VX * e[:, None])
        meat = scores.T @ scores
        return sol["C"] @ meat @ sol["C"]


# ====================================================================== the fit
@dataclass
class GlmmFit:
    beta: np.ndarray
    C: np.ndarray
    gamma: np.ndarray
    theta: np.ndarray
    natural: np.ndarray            # G parameters (+ phi last when estimated)
    cov_nat: np.ndarray | None     # covariance of natural parameters
    active: list
    phi: float
    k: float
    eta: np.ndarray
    mu: np.ndarray
    neg2: float                    # -2 REPL (PQL) or -2 log likelihood (Laplace)
    n_outer: int
    converged: bool
    zd: _ZDesign
    lmm: _PseudoLMM
    method: str
    C22: np.ndarray | None = None
    notes: list = field(default_factory=list)


def _pql(y, m, X, zd, dist, link, offset, omega, k, max_iter=100, tol=1e-6) -> GlmmFit:
    estimate_phi = dist not in DISCRETE
    lmm = _PseudoLMM(X, zd, estimate_phi)
    if dist == "binomial":
        mu = (y * m + 0.5) / (m + 1)
    elif dist == "normal":
        mu = y.astype(float).copy()
    else:
        mu = np.maximum((y + np.mean(y)) / 2, 1e-3)
    mu = _clip_mu(dist, mu)
    eta = link(mu)
    theta = np.zeros(zd.n_theta)
    nat_old = None
    converged = False
    sol = None
    notes = []
    j = 0
    for j in range(max_iter + 1):
        d = link.deriv(mu)
        v = eta - offset + (y - mu) * d
        w = omega * m / (_variance(dist, mu, k) * d * d) if dist == "binomial" else \
            omega / (_variance(dist, mu, k) * d * d)
        w = np.clip(w, 1e-12, 1e12)
        lmm.set_data(v, w)
        theta = lmm.fit_theta(theta if j else np.full(zd.n_theta, 0.5))
        sol = lmm.solve(theta)
        nat = zd.natural(theta, sol["phi"])
        if j > 0 and nat_old is not None:
            num = 2 * np.abs(nat - nat_old)
            den = np.abs(nat) + np.abs(nat_old)
            crit = np.where(den > 1e-12, num / np.where(den > 0, den, 1), 0.0)
            if (crit.max() if len(crit) else 0.0) < tol and \
                    np.max(np.abs(sol["beta"] - beta_old)) < 1e-6 * (
                        1 + np.max(np.abs(beta_old))):
                converged = True
                break
        nat_old = nat
        beta_old = sol["beta"]
        eta_lin = X @ sol["beta"] + (np.asarray(zd.Z @ sol["gamma"]).ravel() if zd.q else 0)
        eta = eta_lin + offset
        mu = _clip_mu(dist, link.inverse(eta))
        eta = link(mu)
    if not converged:
        notes.append("The outer iterations did not converge within the maximum number of "
                     "iterations; the last estimates are displayed.")
    eta = X @ sol["beta"] + (np.asarray(zd.Z @ sol["gamma"]).ravel() if zd.q else 0) + offset
    mu = _clip_mu(dist, link.inverse(eta))
    phi = sol["phi"]
    nat = zd.natural(theta, phi)
    full = np.r_[nat, phi] if estimate_phi else nat
    active = [i for i in range(len(nat)) if not (zd.param_names()[i][2] and nat[i] < 1e-8 *
                                                 max(1.0, phi))]
    if estimate_phi:
        active.append(len(nat))
    cov_nat = None
    if active:
        from rsstatchem.core.stats.mixed import _hessian

        H, _ = _hessian(lmm.natural_objective, full, active)
        sub = H[np.ix_(active, active)]
        if np.all(np.isfinite(sub)):
            try:
                inv = 2 * np.linalg.inv(sub)
            except np.linalg.LinAlgError:
                inv = 2 * np.linalg.pinv(sub)
            cov_nat = np.full((len(full), len(full)), np.nan)
            cov_nat[np.ix_(active, active)] = inv
            if np.any(np.diag(inv) <= 0):
                notes.append("The final Hessian matrix is not positive definite although "
                             "all convergence criteria are satisfied. Validity of subsequent "
                             "results cannot be ascertained.")
    neg2 = lmm.objective(theta)
    return GlmmFit(sol["beta"], sol["C"], sol["gamma"], theta, full, cov_nat, active, phi, k,
                   eta, mu, neg2, j, converged, zd, lmm, "pql", notes=notes)


def _laplace(y, m, X, zd, dist, link, offset, omega, k, start: GlmmFit) -> GlmmFit:
    """ML by the Laplace approximation (lme4 nAGQ = 1): PIRLS for the spherical modes, joint
    optimisation of the relative Cholesky parameters and the fixed effects."""
    p = X.shape[1]
    Z = zd.Z
    state = {"u": np.zeros(zd.q)}

    def pirls(theta, beta):
        lam = zd.lam(theta)
        Zt = (Z @ lam).tocsc()
        u = state["u"].copy()
        base = X @ beta + offset
        obj_old = None
        for _ in range(100):
            eta = base + np.asarray(Zt @ u).ravel()
            mu = _clip_mu(dist, link.inverse(eta))
            obj = -2 * np.sum(omega * _loglik_obs(dist, y, mu, m, 1.0, k)) + u @ u
            if obj_old is not None and obj > obj_old + 1e-10:
                # step halving
                for _h in range(20):
                    u = (u + u_prev) / 2
                    eta = base + np.asarray(Zt @ u).ravel()
                    mu = _clip_mu(dist, link.inverse(eta))
                    obj = -2 * np.sum(omega * _loglik_obs(dist, y, mu, m, 1.0, k)) + u @ u
                    if obj <= obj_old + 1e-10:
                        break
            if obj_old is not None and abs(obj_old - obj) < 1e-10 * (1 + abs(obj)):
                break
            obj_old = obj
            d = link.deriv(mu)
            w = omega * (m if dist == "binomial" else 1) / (_variance(dist, mu, k) * d * d)
            zwork = eta - base + (y - mu) * d
            WZt = Zt.multiply(w[:, None]).tocsc()
            A = (Zt.T @ WZt).toarray() + np.eye(zd.q)
            u_prev = u
            u = linalg.solve(A, np.asarray(WZt.T @ zwork).ravel(), assume_a="pos")
        eta = base + np.asarray(Zt @ u).ravel()
        mu = _clip_mu(dist, link.inverse(eta))
        d = link.deriv(mu)
        w = omega * (m if dist == "binomial" else 1) / (_variance(dist, mu, k) * d * d)
        WZt = Zt.multiply(w[:, None]).tocsc()
        A = (Zt.T @ WZt).toarray() + np.eye(zd.q)
        ll = np.sum(omega * _loglik_obs(dist, y, mu, m, 1.0, k))
        logdet = np.linalg.slogdet(A)[1]
        state["u"] = u
        return dict(dev=-2 * ll + u @ u + logdet, u=u, eta=eta, mu=mu, w=w, Zt=Zt, A=A,
                    lam=lam)

    nt = zd.n_theta

    def dev(par):
        r = pirls(par[:nt], par[nt:])
        return r["dev"]

    phi0 = start.phi if start.phi > 0 else 1.0
    theta0 = start.theta * (math.sqrt(phi0) if start.lmm.estimate_phi else 1.0)
    x0 = np.r_[theta0, start.beta]
    bounds = [(lb, None) for lb in zd.lower_bounds()] + [(None, None)] * p
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = optimize.minimize(dev, x0, method="L-BFGS-B", bounds=bounds,
                              options={"ftol": 1e-15, "gtol": 1e-8, "maxiter": 1000})
        r2 = optimize.minimize(dev, r.x, method="Nelder-Mead",
                               options={"xatol": 1e-8, "fatol": 1e-11, "maxiter": 20000})
    x = r2.x if r2.fun <= r.fun else r.x
    x[:nt] = [max(xi, b[0]) if b[0] is not None else xi for xi, b in zip(x[:nt], bounds)]
    theta, beta = x[:nt], x[nt:]
    fin = pirls(theta, beta)
    nat = zd.natural(theta, 1.0)
    names = zd.param_names()
    active_t = [i for i in range(len(nat)) if not (names[i][2] and nat[i] < 1e-8)]

    def dev_nat(par):
        th = zd.theta_from_natural(par[:len(nat)], 1.0)
        if th is None:
            return 1e300
        state["u"] = fin["u"].copy()
        return pirls(th, par[len(nat):])["dev"]

    from rsstatchem.core.stats.mixed import _hessian

    full = np.r_[nat, beta]
    active = active_t + list(range(len(nat), len(full)))
    H, _ = _hessian(dev_nat, full, active)
    state["u"] = fin["u"]
    sub = H[np.ix_(active, active)]
    notes = []
    try:
        inv = 2 * np.linalg.inv(sub)
    except np.linalg.LinAlgError:
        inv = 2 * np.linalg.pinv(sub)
    cov = np.full((len(full), len(full)), np.nan)
    cov[np.ix_(active, active)] = inv
    # fixed-effect covariance from the Hessian of the Laplace deviance (lme4's default
    # vcov(use.hessian = TRUE)); the Schur complement of the PIRLS system (lme4's RX) is the
    # fallback when the Hessian is not usable
    C = cov[len(nat):, len(nat):]
    if not np.all(np.isfinite(C)) or np.any(np.diag(C) <= 0):
        w = fin["w"]
        XWX = X.T @ (X * w[:, None])
        ZWX = np.asarray(fin["Zt"].multiply(w[:, None]).T @ X)
        C = np.linalg.inv(XWX - ZWX.T @ np.linalg.solve(fin["A"], ZWX))
    gamma = np.asarray(fin["lam"] @ fin["u"]).ravel()
    lmm = start.lmm
    return GlmmFit(beta, C, gamma, theta, nat, cov[:len(nat), :len(nat)], active_t, 1.0, k,
                   fin["eta"], fin["mu"], float(fin["dev"]),
                   int(r.nit), bool(r.success or r2.success), zd, lmm, "laplace",
                   notes=notes)


# ====================================================================== data preparation
def _subject_codes(frame: pd.DataFrame, subjects: list[str]) -> tuple[np.ndarray, int]:
    if not subjects:
        return np.zeros(len(frame), int), 1
    keys = frame[subjects].astype(str).agg("\x1f".join, axis=1)
    codes, uniq = pd.factorize(keys)
    return codes.astype(int), len(uniq)


def _sort_key(v):
    return (isinstance(v, str), v)


def _adjusted(pvals: list[float], method: str) -> list[float]:
    m = len(pvals)
    if method == "lsd" or m <= 1:
        return list(pvals)
    order = np.argsort(pvals)
    adj = np.zeros(m)
    run = 0.0
    for rank, i in enumerate(order):
        p = pvals[i]
        a = (m - rank) * p if method == "seqbonferroni" else 1 - (1 - p) ** (m - rank)
        run = max(run, min(a, 1.0))
        adj[i] = run
    return list(adj)


# ====================================================================== main procedure
def glmm(ds: Dataset, target: str, factors: list[str] | None = None,
         covariates: list[str] | None = None, subjects: list[str] | None = None,
         random_blocks: list[RandomBlock] | None = None, distribution: str = "normal",
         link: str = "identity", power: float = 1.0, trials: str | None = None,
         trials_value: float | None = None, reference: str = "last",
         custom_reference=None, offset: str | None = None, weight: str | None = None,
         fixed_terms: list[TermSpec] | None = None, intercept: bool = True,
         method: str = "pql", max_iter: int = 100, level: float = 0.95,
         robust: bool = False, emm: list[str] | None = None, emm_scale: str = "original",
         contrast: str = "none", contrast_field: str | None = None, adjust: str = "lsd",
         emm_cov_values: dict | None = None, save: list[str] | None = None,
         classification: bool = True, pseudo_r2: bool = True,
         parameter_estimates: bool = True, covariance_estimates: bool = True
         ) -> GlmmResult:
    """SPSS GENLINMIXED. Returns a :class:`GlmmResult` (``saved`` holds PredictedValue,
    PredictedProbability, CI_Lower / CI_Upper and PearsonResidual when requested)."""
    from rsstatchem.core.stats.manova import make_factor
    from rsstatchem.core.stats.mixed import _designs_spec

    factors = list(factors or [])
    covariates = list(covariates or [])
    subjects = list(subjects or [])
    blocks = [b for b in (random_blocks or []) if b.intercept or b.effects]
    emm = list(emm or [])
    save = list(save or [])
    section = Section("Generalized Linear Mixed Models")
    result = GlmmResult(section)
    tvar = ds.var(target)
    lk = _link(link, power)
    link_text = LINKS[link] if link != "power" else f"Power({power:g})"
    if fixed_terms is None:
        fixed_terms = main_effects([ds.var(f).name for f in factors],
                                   [ds.var(c).name for c in covariates])
    section.log(_syntax(target, trials, trials_value, factors, covariates, subjects, blocks,
                        distribution, link, power, fixed_terms, intercept, offset, weight,
                        method, max_iter, level, robust, emm, emm_scale, contrast, adjust,
                        save, reference))

    # ------------------------------------------------------------------ data
    block_vars = [e for b in blocks for e in b.effects] + [s for b in blocks for s in b.subjects]
    names = list(dict.fromkeys([target] + factors + covariates + subjects + block_vars +
                               ([trials] if trials else []) + ([offset] if offset else []) +
                               ([weight] if weight else [])))
    num_names = set(covariates) | {x for x in (trials, offset, weight) if x}
    if distribution != "binomial" or trials or trials_value:
        num_names.add(target)
    for e in block_vars:
        if ds.var(e).type == NUMERIC and e not in factors:
            num_names.add(e)
    data = {}
    for nme in names:
        data[nme] = numeric(ds, nme) if nme in num_names else ds.analysis_series(nme)
    frame = pd.DataFrame(data)
    total = len(frame)
    frame = frame.dropna()
    excl_notes = []
    if weight:
        frame = frame[frame[weight] > 0]
    y_raw = frame[target]
    success_label = failure_label = None
    if distribution == "binomial" and not trials and trials_value is None:
        vals = sorted(y_raw.unique(), key=_sort_key)
        if len(vals) != 2:
            section.note("Warning", "A binomial target without a trials variable must have "
                         f"exactly two values; {tvar.name} has {len(vals)}.", style="warning")
            return result
        if custom_reference is not None and custom_reference in vals:
            ref = custom_reference
        else:
            ref = vals[-1] if reference == "last" else vals[0]
        succ = vals[0] if ref == vals[-1] else vals[-1]
        y = (y_raw == succ).to_numpy(float)
        m = np.ones(len(y))
        success_label = value_text(tvar, succ)
        failure_label = value_text(tvar, ref)
    elif distribution == "binomial":
        r = y_raw.to_numpy(float)
        m = frame[trials].to_numpy(float) if trials else np.full(len(r), float(trials_value))
        ok = (m > 0) & (r >= 0) & (r <= m)
        frame, r, m = frame[ok], r[ok], m[ok]
        y = r / m
    else:
        y = y_raw.to_numpy(float)
        if distribution in ("gamma", "igauss"):
            ok = y > 0
        elif distribution in ("poisson", "negbin"):
            ok = (y >= 0) & (np.abs(y - np.round(y)) < 1e-9)
        else:
            ok = np.ones(len(y), bool)
        if not ok.all():
            excl_notes.append(f"{int((~ok).sum())} records with target values outside the "
                              "range of the distribution were excluded.")
        frame, y = frame[ok], y[ok]
        m = np.ones(len(y))
    n = len(frame)
    if n < 3:
        section.note("Warning", "Not enough valid records for the model.", style="warning")
        return result
    omega = frame[weight].to_numpy(float) if weight else np.ones(n)
    off = frame[offset].to_numpy(float) if offset else np.zeros(n)

    fobjs = [make_factor(ds, f, frame[f]) for f in factors]
    covs = {ds.var(c).name: frame[c].to_numpy(float) for c in covariates}
    try:
        Xe, eff_idx, Xr, ref_info, design = _designs_spec(fobjs, covs, fixed_terms, n,
                                                          intercept)
    except KeyError as exc:
        section.note("Warning", f"Unknown variable in the fixed-effects model: {exc}",
                     style="warning")
        return result
    if emm_cov_values:
        for kname, val in emm_cov_values.items():
            if kname in design.cov_means and val is not None:
                design.cov_means[kname] = float(val)
    p = Xr.shape[1]
    if p == 0:
        section.note("Warning", "The model has no fixed effects.", style="warning")
        return result

    # random effects design
    zblocks = []
    for b in blocks:
        cols, cnames, groups, gnames = [], [], [], []
        if b.intercept:
            cols.append(np.ones(n))
            cnames.append("Intercept")
            groups.append([0])
            gnames.append("Intercept")
        for e in b.effects:
            ename = ds.var(e).name
            if e in covariates or (ds.var(e).type == NUMERIC and e not in factors and
                                   e in num_names):
                start_ = len(cols)
                cols.append(frame[e].to_numpy(float))
                cnames.append(ename)
                groups.append([start_])
                gnames.append(ename)
            else:
                lv = sorted(frame[e].unique(), key=_sort_key)
                start_ = len(cols)
                for v_ in lv:
                    cols.append((frame[e] == v_).to_numpy(float))
                    cnames.append(f"{ename}={value_text(ds.var(e), v_)}")
                groups.append(list(range(start_, len(cols))))
                gnames.append(ename)
        codes, msub = _subject_codes(frame, b.subjects)
        zblocks.append(dict(Zs=np.column_stack(cols), codes=codes, m=msub,
                            covtype=b.covtype if len(cols) > 1 or b.covtype != "UN" else "VC",
                            groups=groups, gnames=gnames, names=cnames,
                            subjects=[ds.var(s).name for s in b.subjects],
                            spec=b))
    zd = _ZDesign(zblocks, n)

    k_nb = _nb_k(y, Xr, off) if distribution == "negbin" else 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = _pql(y, m, Xr, zd, distribution, lk, off, omega, k_nb, max_iter)
        if method == "laplace" and distribution in DISCRETE and zd.q:
            fit = _laplace(y, m, Xr, zd, distribution, lk, off, omega, k_nb, fit)
        elif method == "laplace":
            excl_notes.append("The Laplace approximation is used for discrete targets with "
                              "random effects only; REPL-PQL estimates are shown.")
    result.fit = fit
    beta, C = fit.beta, fit.C
    if robust and fit.method == "pql":
        outer = zblocks[0]["codes"] if zblocks else np.arange(n)
        if subjects:
            outer, _ = _subject_codes(frame, subjects[:1])
        C = fit.lmm.robust_cov(fit.lmm.solve(fit.theta), outer)
    df_res = float(n - np.linalg.matrix_rank(Xr))
    tcrit = stats.t.ppf(0.5 + level / 2, df_res)

    # ------------------------------------------------------------------ Model Summary
    d_cov = len([i for i in fit.active])
    if fit.method == "pql":
        nstar = n - p
        neg2 = fit.neg2
        aicc = neg2 + 2 * d_cov * nstar / max(nstar - d_cov - 1, 1)
        bic = neg2 + d_cov * math.log(nstar)
        crit_note = (f"Information criteria are based on the -2 log pseudo likelihood "
                     f"({fmt(neg2)}) and are used to compare models. Models with smaller "
                     "information criterion values fit better.")
    else:
        dpar = d_cov + p
        neg2 = fit.neg2
        aicc = neg2 + 2 * dpar * n / max(n - dpar - 1, 1)
        bic = neg2 + dpar * math.log(n)
        crit_note = (f"Information criteria are based on the -2 log likelihood ({fmt(neg2)}, "
                     "Laplace approximation) and are used to compare models. Models with "
                     "smaller information criterion values fit better.")
    tdesc = tvar.title
    if distribution == "binomial" and (trials or trials_value is not None):
        tdesc = f"{tvar.title} / {ds.var(trials).title if trials else fmt(trials_value, 0)}"
    rows = [["Target", tdesc], ["Probability Distribution", DISTRIBUTIONS[distribution]],
            ["Link Function", link_text]]
    if distribution == "binomial" and success_label is not None:
        rows.append(["Target Category Modeled", f"{success_label} (reference category: "
                                                f"{failure_label})"])
    if distribution == "negbin":
        rows.append(["Negative Binomial Ancillary Parameter (k)", fmt(k_nb, 4)])
    rows += [["Estimation Method", METHODS[fit.method]],
             ["Information Criterion: Akaike Corrected", fmt(aicc)],
             ["Information Criterion: Bayesian", fmt(bic)]]
    section.table(Table("Model Summary", ["", ""], rows, footnotes=[crit_note]))

    # ------------------------------------------------------------------ Case processing
    rows = [["Included", fmt_int(n), fmt(100 * n / total if total else 0, 1) + "%"],
            ["Excluded", fmt_int(total - n), fmt(100 * (total - n) / total if total else 0, 1)
             + "%"], ["Total", fmt_int(total), "100.0%"]]
    section.table(Table("Case Processing Summary", ["", "N", "Percent"], rows,
                        footnotes=excl_notes))
    if zblocks:
        rows = []
        for kb, zb in enumerate(zblocks):
            rows.append([f"Random Effect {kb + 1}", " * ".join(zb["subjects"]) or "(none)",
                         fmt_int(zb["m"]), zb["spec"].label(), COVTYPES[zb["covtype"]]])
        section.table(Table("Data Structure", ["", "Subject", "Number of Subjects",
                                               "Effects", "Covariance Type"], rows))
    for note in fit.notes:
        section.note("Warning", note, style="warning")

    # ------------------------------------------------------------------ Classification
    pred_mu = fit.mu
    if distribution == "binomial" and success_label is not None and classification:
        pred = (pred_mu >= 0.5).astype(int)
        obs = y.astype(int)
        rows = []
        for o_val, o_lab in ((0, failure_label), (1, success_label)):
            sel = obs == o_val
            c0 = int(np.sum(sel & (pred == 0)))
            c1 = int(np.sum(sel & (pred == 1)))
            tot = c0 + c1
            rows.append([o_lab, fmt_int(c0), fmt_int(c1),
                         fmt(100 * (c1 if o_val else c0) / tot, 1) + "%" if tot else ""])
        correct = float(np.mean(pred == obs) * 100)
        rows.append(["Overall Percent", fmt(100 * np.mean(pred == 0), 1) + "%",
                     fmt(100 * np.mean(pred == 1), 1) + "%", fmt(correct, 1) + "%"])
        section.table(Table("Classification", [("", "Observed"),
                                               ("Predicted", failure_label),
                                               ("Predicted", success_label),
                                               ("Predicted", "Percent Correct")], rows,
                            footnotes=[f"Target: {tvar.title}"]))

    # ------------------------------------------------------------------ Fixed effects
    T = np.linalg.lstsq(Xr, Xe, rcond=None)[0]       # Xe = Xr T
    Ti = np.linalg.pinv(T)
    beta_e = Ti @ beta
    C_e = Ti @ C @ Ti.T
    foot = [f"Target: {tvar.title}", f"Probability distribution: "
                                     f"{DISTRIBUTIONS[distribution]}",
            f"Link function: {link_text}"]
    rows = []
    labels_terms = [(t.label, eff_idx[i + 1]) for i, t in enumerate(fixed_terms)]
    all_idx = [c for _, idx in labels_terms for c in idx]
    if all_idx and intercept:
        rows.append(_ftest_row("Corrected Model", all_idx, beta_e, C_e, df_res))
    for label, idx in labels_terms:
        if idx:
            rows.append(_ftest_row(label, idx, beta_e, C_e, df_res))
    if robust:
        foot.append("Robust covariances are used to compute the tests and standard errors.")
    section.table(Table("Fixed Effects", ["Source", "F", "df1", "df2", "Sig."], rows,
                        footnotes=foot))

    exp_on = (distribution, link) in EXP_LINKS
    pe = []
    if parameter_estimates:
        prow = []
        any_red = False
        it = iter(range(p))
        for label, cols_ in ref_info:
            if cols_ is None:
                any_red = True
                prow.append([label, "0" + sup("b"), "", "", "", "", ""] +
                            (["", "", ""] if exp_on else []))
                continue
            j = next(it)
            b_, se = float(beta[j]), math.sqrt(max(float(C[j, j]), 0.0))
            tt = b_ / se if se > 0 else float("nan")
            sig = 2 * stats.t.sf(abs(tt), df_res)
            lo, hi = b_ - tcrit * se, b_ + tcrit * se
            pe.append((label, b_, se))
            row = [label, fmt(b_), fmt(se), fmt(tt, 3), fmt_p(sig), fmt(lo), fmt(hi)]
            if exp_on:
                row += [fmt(_safe_exp(b_)), fmt(_safe_exp(lo)), fmt(_safe_exp(hi))]
            prow.append(row)
        cols = ["Model Term", "Coefficient", "Std. Error", "t", "Sig.",
                (f"{pct(level)} Confidence Interval", "Lower"),
                (f"{pct(level)} Confidence Interval", "Upper")]
        if exp_on:
            cols += ["Exp(Coefficient)",
                     (f"{pct(level)} Confidence Interval for Exp(Coefficient)", "Lower"),
                     (f"{pct(level)} Confidence Interval for Exp(Coefficient)", "Upper")]
        f2 = list(foot)
        f2.append(f"Degrees of freedom (residual method): {fmt(df_res, 0)}.")
        if any_red:
            f2.append("b. This coefficient is set to zero because it is redundant.")
        if fit.method == "laplace":
            f2.append("Laplace approximation (maximum likelihood): R lme4::glmer-equivalent, "
                      "not an SPSS GENLINMIXED estimation method.")
        section.table(Table("Fixed Coefficients", cols, prow, footnotes=f2))
    if pe:
        from rsstatchem.core.stats.resample import ModelInfo

        result.models = [ModelInfo([r_[0] for r_ in pe], np.array([r_[1] for r_ in pe]),
                                   np.array([r_[2] for r_ in pe]), n=n,
                                   table="Fixed Coefficients", headers=["Model Term"],
                                   est_label="Coefficient", df=np.full(len(pe), df_res),
                                   bootable=False,
                                   note="Bootstrapping is not available for Generalized "
                                        "Linear Mixed Models (as in SPSS).")]

    # ------------------------------------------------------------------ Covariance parameters
    if covariance_estimates and (zblocks or fit.lmm.estimate_phi):
        z = stats.norm.ppf(0.5 + level / 2)
        rows = []
        pnames = zd.param_names()
        nat = fit.natural

        def add(head, label, i, variance):
            est = float(nat[i])
            if i not in fit.active:
                rows.append([head, label, fmt(est) + sup("a"), "", "", "", "", ""])
                return
            se = math.sqrt(max(float(fit.cov_nat[i, i]), 0.0)) \
                if fit.cov_nat is not None and np.isfinite(fit.cov_nat[i, i]) else float("nan")
            wz = est / se if se > 0 else float("nan")
            sig = 2 * stats.norm.sf(abs(wz)) if wz == wz else float("nan")
            if variance and est > 0 and se > 0:
                ratio = min(z * se / est, 700.0)
                lo, hi = est * math.exp(-ratio), est * math.exp(ratio)
            else:
                lo, hi = est - z * se, est + z * se
            rows.append([head, label, fmt(est), fmt(se) if se == se else "",
                         fmt(wz) if wz == wz else "", fmt_p(sig) if sig == sig else "",
                         fmt(lo) if lo == lo else "", fmt(hi) if hi == hi else ""])

        if fit.lmm.estimate_phi:
            add("Residual Effect", "Residual" if distribution == "normal" else "Scale",
                len(nat) - 1, True)
        last_block = -1
        for i, (kb, label, isvar) in enumerate(pnames):
            zb = zblocks[kb]
            head = (f"Random Effect {kb + 1} [subject = {' * '.join(zb['subjects']) or '-'}]"
                    if kb != last_block else "")
            last_block = kb
            add(head, label, i, isvar)
        foot = [f"Covariance Structure: "
                + "; ".join(f"Random Effect {kb + 1}: {COVTYPES[zb['covtype']]}"
                            for kb, zb in enumerate(zblocks)) if zblocks else
                "Covariance Structure: Scaled identity",
                f"Target: {tvar.title}"]
        if len(fit.active) < len(nat):
            foot.append("a. This covariance parameter is redundant (estimated at the "
                        "boundary). The test statistic and confidence interval cannot be "
                        "computed.")
        if not fit.lmm.estimate_phi:
            foot.append("The residual (scale) parameter is fixed at 1 for the "
                        f"{DISTRIBUTIONS[distribution].lower()} distribution.")
        section.table(Table("Covariance Parameter Estimates",
                            ["", "", "Estimate", "Std. Error", "Z", "Sig.",
                             (f"{pct(level)} Confidence Interval", "Lower"),
                             (f"{pct(level)} Confidence Interval", "Upper")],
                            rows, row_header_cols=2, footnotes=foot))

    # random-effect predictions
    for kb, zb in enumerate(zblocks):
        if not zb["spec"].predictions:
            continue
        off_ = zd.offsets[kb]
        qb = zb["Zs"].shape[1]
        rows = []
        keys = {}
        for code, row_ in zip(zb["codes"], frame[zb["spec"].subjects].itertuples(index=False)):
            keys.setdefault(int(code), " * ".join(str(x) for x in row_))
        for code in sorted(keys, key=lambda c: keys[c]):
            for j_, nme in enumerate(zb["names"]):
                rows.append([keys[code] if j_ == 0 else "", nme,
                             fmt(fit.gamma[off_ + code * qb + j_])])
        section.table(Table(f"Random Effect {kb + 1} Predictions (EBLUP)",
                            [" * ".join(zb["subjects"]) or "Subject", "Effect", "Prediction"],
                            rows, row_header_cols=2))

    # ------------------------------------------------------------------ Pseudo-R2 / ICC
    if pseudo_r2 and zblocks and not weight:
        r2 = _pseudo_r2(fit, Xr, off, zd, distribution, link, lk, k_nb, n)
        if r2 is not None:
            rows = [["Marginal R Squared", fmt(r2["Rm"])], ["Conditional R Squared",
                                                           fmt(r2["Rc"])],
                    ["Conditional ICC", fmt(r2["ICCc"])], ["Adjusted ICC", fmt(r2["ICCa"])]]
            section.table(Table("Pseudo R Squared and Intraclass Correlation", ["", "Value"],
                                rows, footnotes=[
                                    "Nakagawa & Schielzeth (2013, 2017); observation-level "
                                    f"variance by the {r2['method']}.",
                                    f"Variance components: fixed {fmt(r2['sf'])}, random "
                                    f"{fmt(r2['sg'])}, observation-level {fmt(r2['se'])}."]))

    # ------------------------------------------------------------------ Estimated means
    if emm:
        _emmeans(section, emm, design, fobjs, beta_e, C_e, df_res, level, lk, link_text,
                 emm_scale, contrast, contrast_field, adjust, distribution, tvar)

    # ------------------------------------------------------------------ Save
    if save:
        result.saved = _save(save, frame, ds, fit, Xr, C, zd, off, lk, distribution, y, m,
                             omega, tcrit, success_label, level)
    return result


def _safe_exp(x):
    return math.exp(x) if x < 700 else float("inf")


def _ftest_row(label, idx, beta_e, C_e, df2):
    L = np.zeros((len(idx), len(beta_e)))
    for r_, c in enumerate(idx):
        L[r_, c] = 1
    b = L @ beta_e
    LCL = L @ C_e @ L.T
    q = int(np.linalg.matrix_rank(LCL))
    F = float(b @ np.linalg.pinv(LCL) @ b) / max(q, 1)
    return [label, fmt(F, 3), fmt_int(q), fmt_int(int(round(df2))),
            fmt_p(stats.f.sf(F, q, df2))]


def _pseudo_r2(fit, X, off, zd, dist, link_name, lk, k, n):
    """IBM GENLINMIXED pseudo-R2 and ICC (Nakagawa, Johnson & Schielzeth 2017)."""
    sf = float(np.var(X @ fit.beta + off, ddof=1))
    if not zd.q:
        return None
    G = []
    for kb, b in enumerate(zd.blocks):
        nat = fit.natural[zd.slices[kb]]
        G.append(zd.G_from_natural(kb, nat))
    sg = 0.0
    for kb, b in enumerate(zd.blocks):
        Zs = b["Zs"]
        sg += float(np.einsum("ij,jk,ik->", Zs, G[kb], Zs)) / n
    mubar = float(np.mean(fit.mu))
    phi = fit.phi
    if dist == "normal":
        h = phi * (lk.deriv(np.array([mubar]))[0]) ** 2
    else:
        var_y = (phi if dist in ("gamma", "igauss") else 1.0) * \
            _variance(dist, np.array([mubar]), k)[0]
        h = var_y * lk.deriv(np.array([mubar]))[0] ** 2
    method = "delta method"
    if link_name == "log" and dist in ("poisson", "negbin", "gamma"):
        se = float(special.polygamma(1, 1 / h))
        method = "trigamma function"
    else:
        se = float(h)
    tot = sf + sg + se
    return dict(Rm=sf / tot, Rc=(sf + sg) / tot, ICCc=sg / tot, ICCa=sg / (sg + se), sf=sf,
                sg=sg, se=se, method=method)


def _emmeans(section, emm, design, fobjs, beta, C, df, level, lk, link_text, scale, contrast,
             contrast_field, adjust, dist, tvar) -> None:
    names = [f.name for f in fobjs]
    tq = stats.t.ppf(0.5 + level / 2, df)
    orig = scale == "original"
    foot = [f"Target: {tvar.title}",
            "Estimated means are displayed in the "
            + ("original target scale." if orig else f"link function ({link_text}) scale.")]
    if design.cov_means:
        foot.append("Continuous predictors are fixed at: " +
                    ", ".join(f"{k_} = {fmt(v, 4)}" for k_, v in design.cov_means.items()) + ".")
    ci = [(f"{pct(level)} Confidence Interval", "Lower"),
          (f"{pct(level)} Confidence Interval", "Upper")]
    for spec in emm:
        parts = [p_.strip() for p_ in spec.split("*")]
        if any(p_ not in names for p_ in parts):
            continue
        idx = [names.index(p_) for p_ in parts]
        Ls, labels = [], []
        for combo in itertools.product(*[range(len(fobjs[i].levels)) for i in idx]):
            Ls.append(design.average_row(dict(zip(idx, combo))))
            labels.append([fobjs[i].labels[lv] for i, lv in zip(idx, combo)])
        L = np.array(Ls)
        eta = L @ beta
        Veta = L @ C @ L.T
        if orig:
            mu = lk.inverse(eta)
            dmu = lk.inverse_deriv(eta)
            est = mu
            V = Veta * np.outer(dmu, dmu)
        else:
            est, V = eta, Veta
        rows = []
        for j, lab in enumerate(labels):
            se_eta = math.sqrt(max(Veta[j, j], 0.0))
            lo_e, hi_e = eta[j] - tq * se_eta, eta[j] + tq * se_eta
            if orig:
                lo, hi = sorted([float(lk.inverse(np.array([lo_e]))[0]),
                                 float(lk.inverse(np.array([hi_e]))[0])])
            else:
                lo, hi = lo_e, hi_e
            se = math.sqrt(max(V[j, j], 0.0))
            tt = eta[j] / se_eta if se_eta > 0 else float("nan")
            rows.append(lab + [fmt(est[j]), fmt(se), fmt(tt, 3), fmt(df, 0),
                               fmt_p(2 * stats.t.sf(abs(tt), df)), fmt(lo), fmt(hi)])
        section.table(Table(f"Estimated Means: {spec}", parts +
                            ["Mean", "Std. Error", "t", "df", "Sig."] + ci, rows,
                            row_header_cols=len(parts), footnotes=foot))
        # contrasts
        if contrast == "none" or (contrast != "pairwise" and len(idx) > 1):
            continue
        cfield = contrast_field if contrast_field in parts else parts[0]
        ci_idx = parts.index(cfield)
        others = [k_ for k_ in range(len(parts)) if k_ != ci_idx]
        groups: dict[tuple, list[int]] = {}
        for j, lab in enumerate(labels):
            groups.setdefault(tuple(lab[k_] for k_ in others), []).append(j)
        crow, pvals, raw = [], [], []
        for okey, js in groups.items():
            pairs = []
            if contrast == "pairwise":
                pairs = [(a, b, f"{labels[a][ci_idx]} - {labels[b][ci_idx]}", None)
                         for a, b in itertools.combinations(js, 2)]
            elif contrast == "simple":
                pairs = [(a, js[-1], f"{labels[a][ci_idx]} - {labels[js[-1]][ci_idx]}", None)
                         for a in js[:-1]]
            else:  # deviation: level - mean of all levels
                kk = len(js)
                for a in js[:-1]:
                    cvec = np.zeros(len(labels))
                    cvec[js] = -1 / kk
                    cvec[a] += 1
                    pairs.append((a, None, f"{labels[a][ci_idx]} - Mean", cvec))
            for a, b, text, cvec in pairs:
                if cvec is None:
                    cvec = np.zeros(len(labels))
                    cvec[a], cvec[b] = 1, -1
                e_ = float(cvec @ est)
                se = math.sqrt(max(float(cvec @ V @ cvec), 0.0))
                tt = e_ / se if se > 0 else float("nan")
                pv = float(2 * stats.t.sf(abs(tt), df))
                pvals.append(pv)
                raw.append((list(okey), text, e_, se, tt))
        adj = _adjusted(pvals, adjust)
        for (okey, text, e_, se, tt), pv, pa in zip(raw, pvals, adj):
            crow.append(okey + [text, fmt(e_), fmt(se), fmt(tt, 3), fmt(df, 0), fmt_p(pv),
                                fmt_p(pa), fmt(e_ - tq * se), fmt(e_ + tq * se)])
        heads = [parts[k_] for k_ in others]
        section.table(Table(f"{CONTRASTS[contrast]} Contrasts: {cfield}"
                            + (f" (within {', '.join(heads)})" if heads else ""),
                            heads + [f"{cfield} Contrast", "Contrast Estimate", "Std. Error",
                                     "t", "df", "Sig.", "Adj. Sig."] + ci, crow,
                            row_header_cols=len(heads) + 1,
                            footnotes=[f"The sequential {ADJUST[adjust].split()[-1]} adjusted "
                                       "significance level is displayed."
                                       if adjust != "lsd" else
                                       "Least significant difference (no adjustment).",
                                       "Confidence intervals are not adjusted."] + foot[:2]))


def _save(keys, frame, ds, fit, X, C, zd, off, lk, dist, y, m, omega, tcrit, success_label,
          level) -> dict:
    n = len(frame)
    eta = fit.eta
    mu = fit.mu
    # PQL-type standard error of the linear predictor: fixed part only plus random part
    se_eta = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", X, C, X), 0))
    lo = lk.inverse(eta - tcrit * se_eta)
    hi = lk.inverse(eta + tcrit * se_eta)
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    var = _variance(dist, mu, fit.k) * (fit.phi if dist not in DISCRETE else 1.0) / omega
    if dist == "binomial":
        var = var / m
    pearson = (y - mu) / np.sqrt(np.maximum(var, 1e-300))
    vals = {}
    if "predicted" in keys:
        if dist == "binomial" and success_label is not None:
            vals["PredictedValue"] = ("Predicted Value (1 = " + success_label + ")",
                                      (mu >= 0.5).astype(float))
        else:
            vals["PredictedValue"] = ("Predicted Value", mu)
    if "probability" in keys and dist == "binomial":
        vals["PredictedProbability"] = (f"Predicted Probability"
                                        + (f" of {success_label}" if success_label else ""), mu)
    if "ci" in keys:
        vals["CI_Lower"] = (f"Lower {pct(level)} CI of the Predicted Mean", lo)
        vals["CI_Upper"] = (f"Upper {pct(level)} CI of the Predicted Mean", hi)
    if "pearson" in keys:
        vals["PearsonResidual"] = ("Pearson Residual", pearson)
    out = {}
    for key, (label, arr) in vals.items():
        s = pd.Series(np.nan, index=ds.df.index)
        s.loc[frame.index] = arr
        out[key] = (label, s)
    del n
    return out


def _syntax(target, trials, trials_value, factors, covariates, subjects, blocks, dist, link,
            power, terms, intercept, offset, weight, method, max_iter, level, robust, emm,
            emm_scale, contrast, adjust, save, reference) -> str:
    log = "GENLINMIXED"
    if subjects:
        log += f"\n  /DATA_STRUCTURE SUBJECTS={'*'.join(subjects)}"
    tgt = f"TARGET={target}"
    if trials:
        tgt += f" TRIALS={trials}"
    elif trials_value is not None:
        tgt += f" TRIALS={trials_value:g}"
    if offset:
        tgt += f" OFFSET={offset}"
    if weight:
        tgt += f" ANALYSISWEIGHT={weight}"
    log += f"\n  /FIELDS {tgt}"
    lk = f"POWER({power:g})" if link == "power" else {
        "identity": "IDENTITY", "log": "LOG", "logit": "LOGIT", "probit": "PROBIT",
        "cloglog": "CLOGLOG", "nloglog": "NLOGLOG", "logc": "LOGC", "cauchit": "CAUCHIT"}[link]
    dn = {"normal": "NORMAL", "binomial": "BINOMIAL", "poisson": "POISSON",
          "negbin": "NEGATIVE_BINOMIAL", "gamma": "GAMMA", "igauss": "IGAUSS"}[dist]
    log += f"\n  /TARGET_OPTIONS DISTRIBUTION={dn} LINK={lk}"
    if reference != "last":
        log += " REFERENCE=FIRST"
    log += (f"\n  /FIXED EFFECTS={' '.join(t.syntax for t in terms) or ''} "
            f"USE_INTERCEPT={'TRUE' if intercept else 'FALSE'}")
    for b in blocks:
        eff = " ".join(b.effects)
        log += (f"\n  /RANDOM EFFECTS={eff} USE_INTERCEPT={'TRUE' if b.intercept else 'FALSE'}"
                f" SUBJECTS={'*'.join(b.subjects)} COVARIANCE_TYPE="
                f"{ {'VC': 'VARIANCE_COMPONENTS', 'ID': 'IDENTITY', 'DIAG': 'DIAGONAL', 'UN': 'UNSTRUCTURED'}[b.covtype]}")
        if b.predictions:
            log += " SOLUTION=TRUE"
    log += (f"\n  /BUILD_OPTIONS TARGET_CATEGORY_ORDER=ASCENDING INPUTS_CATEGORY_ORDER=ASCENDING"
            f" MAX_ITERATIONS={max_iter} CONFIDENCE_LEVEL={level * 100:g} DF_METHOD=RESIDUAL"
            f" COVB={'ROBUST' if robust else 'MODEL'}")
    if method == "laplace":
        log += "\n  * Estimation: Laplace approximation (RS-StatChem option, not SPSS)."
    for e in emm:
        log += (f"\n  /EMMEANS TABLES={e} COMPARE={e.split('*')[0]} CONTRAST="
                f"{contrast.upper()}")
    if emm:
        log += (f"\n  /EMMEANS_OPTIONS SCALE={'ORIGINAL' if emm_scale == 'original' else 'TRANSFORMED'}"
                f" PADJUST={ {'lsd': 'LSD', 'seqbonferroni': 'SEQBONFERRONI', 'seqsidak': 'SEQSIDAK'}[adjust]}")
    if save:
        log += "\n  /SAVE " + " ".join({"predicted": "PREDICTED_VALUES(PredictedValue)",
                                        "probability": "PREDICTED_PROBABILITY",
                                        "ci": "CONFIDENCE_INTERVALS(CI)",
                                        "pearson": "PEARSON_RESIDUALS(PearsonResidual)"}[s]
                                       for s in save)
    return log + "."


def selftest() -> None:
    rng = np.random.default_rng(11)
    g = np.repeat(np.arange(12), 8)
    x = rng.normal(size=len(g))
    u = rng.normal(0, 0.8, 12)[g]
    yb = (rng.random(len(g)) < 1 / (1 + np.exp(-(0.3 + 0.8 * x + u)))).astype(float)
    ds = Dataset(pd.DataFrame({"y": yb, "x": x, "g": g.astype(float),
                               "c": rng.poisson(np.exp(0.5 + 0.3 * x + u)).astype(float)}))
    r = glmm(ds, "y", covariates=["x"], subjects=["g"], distribution="binomial",
             link="logit", random_blocks=[RandomBlock(subjects=["g"])], save=["predicted"])
    assert r.fit is not None and np.all(np.isfinite(r.fit.beta))
    r2 = glmm(ds, "c", covariates=["x"], subjects=["g"], distribution="poisson", link="log",
              random_blocks=[RandomBlock(subjects=["g"])])
    assert r2.fit.natural[0] >= 0

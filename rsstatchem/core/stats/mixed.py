"""Analyze > Mixed Models > Linear, mirroring SPSS MIXED output for long-format data.

Fixed effects: factors and covariates in a full-factorial, main-effects or custom model
(interactions, factor-by-covariate terms, nested terms; intercept optional). Random effects
for a subject variable: intercept and/or random slopes of covariates with a Variance
Components (diagonal) or Unstructured covariance matrix. Repeated effects: the residual
covariance within a subject over the levels of a repeated variable can be Scaled Identity,
Diagonal, Compound Symmetry, Heterogeneous CS, AR(1), Heterogeneous AR(1), Toeplitz or
Unstructured. The model is fitted by REML or ML (own Newton-free likelihood optimisation,
started from statsmodels ``MixedLM`` when only G is estimated) and the SPSS tables are
computed from the fitted covariance parameters: Model Dimension, Information Criteria (-2LL,
AIC, AICC, CAIC, BIC in SPSS's conventions), Type III Tests of Fixed Effects and Estimates of
Fixed Effects (last category redundant, Satterthwaite denominator df), Estimates of Covariance
Parameters (Wald Z, asymptotic SEs from the observed information), Estimated Marginal Means
with pairwise comparisons, an intraclass correlation for random-intercept models, and saved
fixed / EBLUP predicted values and residuals.
"""
from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, fmt_p, pct, sup
from rsstatchem.core.stats.glm_model import (Design, TermSpec, aliased_columns,
                                             full_factorial, main_effects)
from rsstatchem.core.stats.manova import (_frame, make_factor, nested_descriptive_rows,
                                          value_text_plain)

LOG2PI = math.log(2 * math.pi)

REPEATED_TYPES = {
    "ID": "Scaled Identity", "DIAG": "Diagonal", "CS": "Compound Symmetry",
    "CSH": "Heterogeneous Compound Symmetry", "AR1": "First-Order Autoregressive",
    "ARH1": "Heterogeneous First-Order Autoregressive", "TP": "Toeplitz",
    "UN": "Unstructured",
}


@dataclass
class _FTerm:
    label: str
    factors: tuple[int, ...] = ()
    covariate: str | None = None


@dataclass
class MixedResult:
    section: Section
    saved: dict = field(default_factory=dict)


def _designs(factors, covs: dict[str, np.ndarray], terms: list[_FTerm], n: int):
    """Effect-coded (Type III) and last-category-reference (estimates) design matrices."""
    specs = []
    for t in terms:
        if t.covariate is not None:
            specs.append(TermSpec(covariates=(t.covariate,)))
        else:
            specs.append(TermSpec(tuple(factors[fi].name for fi in t.factors)))
    return _designs_spec(factors, covs, specs, n, True)


def _designs_spec(factors, covs: dict[str, np.ndarray], specs: list[TermSpec], n: int,
                  intercept: bool = True):
    """(effect-coded design, column indices of [intercept] + terms, reference design of the
    non-redundant indicator columns, [(label, [col] | None)])."""
    design = Design(factors, covs, specs, intercept, n)
    Xe, idx = design.effect_matrix()
    Xo, labels, _ = design.indicator_matrix()
    red = aliased_columns(Xo) if Xo.shape[1] else np.zeros(0, bool)
    ref_info, cols, pos = [], [], 0
    for j, (lab, r) in enumerate(zip(labels, red)):
        if r:
            ref_info.append((lab, None))
        else:
            ref_info.append((lab, [pos]))
            cols.append(j)
            pos += 1
    Xr = Xo[:, cols]
    return Xe, idx, Xr, ref_info, design


# ------------------------------------------------------------------ residual structures
def _r_names(rtype: str, t: int, rlabels: list[str]) -> list[tuple[str, bool]]:
    """(parameter label, is a variance) for the residual covariance parameters."""
    if rtype == "ID":
        return [("Residual", True)]
    if rtype == "DIAG":
        return [(f"Var: [{lab}]", True) for lab in rlabels]
    if rtype == "CS":
        return [("CS diagonal offset", True), ("CS covariance", False)]
    if rtype == "CSH":
        return [(f"Var: [{lab}]", True) for lab in rlabels] + [("CSH rho", False)]
    if rtype == "AR1":
        return [("AR1 diagonal", True), ("AR1 rho", False)]
    if rtype == "ARH1":
        return [(f"Var: [{lab}]", True) for lab in rlabels] + [("ARH1 rho", False)]
    if rtype == "TP":
        return [("TP diagonal", True)] + [(f"TP rho {k}", False) for k in range(1, t)]
    if rtype == "UN":
        return [(f"UN ({i + 1},{j + 1})", i == j) for i in range(t) for j in range(i + 1)]
    raise ValueError(rtype)


def _r_matrix(rtype: str, p: np.ndarray, t: int) -> np.ndarray:
    lag = np.abs(np.subtract.outer(np.arange(t), np.arange(t)))
    if rtype == "ID":
        return p[0] * np.eye(t)
    if rtype == "DIAG":
        return np.diag(p[:t])
    if rtype == "CS":
        return p[0] * np.eye(t) + p[1] * np.ones((t, t))
    if rtype in ("CSH", "ARH1"):
        sd = np.sqrt(np.clip(p[:t], 0, None))
        rho = p[t]
        corr = np.where(lag == 0, 1.0, rho if rtype == "CSH" else rho ** lag)
        return np.outer(sd, sd) * corr
    if rtype == "AR1":
        return p[0] * p[1] ** lag
    if rtype == "TP":
        rhos = np.r_[1.0, p[1:t]]
        return p[0] * rhos[lag]
    if rtype == "UN":
        R = np.zeros((t, t))
        k = 0
        for i in range(t):
            for j in range(i + 1):
                R[i, j] = R[j, i] = p[k]
                k += 1
        return R
    raise ValueError(rtype)


def _r_free(rtype: str, p: np.ndarray, t: int) -> list[float]:
    def lg(v):
        return math.log(max(v, 1e-12))

    def at(r):
        return math.atanh(min(max(r, -0.999), 0.999))

    if rtype in ("ID", "DIAG"):
        return [0.5 * lg(v) for v in p]
    if rtype == "CS":
        return [0.5 * lg(p[0]), p[1]]
    if rtype in ("CSH", "ARH1"):
        return [0.5 * lg(v) for v in p[:t]] + [at(p[t])]
    if rtype == "AR1":
        return [0.5 * lg(p[0]), at(p[1])]
    if rtype == "TP":
        return [0.5 * lg(p[0])] + [at(r) for r in p[1:]]
    R = _r_matrix("UN", p, t)
    L = np.linalg.cholesky(R + 1e-10 * np.eye(t) * max(np.trace(R) / t, 1e-12))
    return [lg(L[i, j]) if i == j else L[i, j] for i in range(t) for j in range(i + 1)]


def _r_natural(rtype: str, z: np.ndarray, t: int) -> list[float]:
    if rtype in ("ID", "DIAG"):
        return [math.exp(2 * v) for v in z]
    if rtype == "CS":
        return [math.exp(2 * z[0]), z[1]]
    if rtype in ("CSH", "ARH1"):
        return [math.exp(2 * v) for v in z[:t]] + [math.tanh(z[t])]
    if rtype == "AR1":
        return [math.exp(2 * z[0]), math.tanh(z[1])]
    if rtype == "TP":
        return [math.exp(2 * z[0])] + [math.tanh(v) for v in z[1:]]
    L = np.zeros((t, t))
    k = 0
    for i in range(t):
        for j in range(i + 1):
            L[i, j] = math.exp(z[k]) if i == j else z[k]
            k += 1
    R = L @ L.T
    return [R[i, j] for i in range(t) for j in range(i + 1)]


def _r_start(rtype: str, s2: float, t: int) -> np.ndarray:
    if rtype == "ID":
        return np.array([s2])
    if rtype == "DIAG":
        return np.full(t, s2)
    if rtype == "CS":
        return np.array([0.7 * s2, 0.3 * s2])
    if rtype in ("CSH", "ARH1"):
        return np.r_[np.full(t, s2), 0.3]
    if rtype == "AR1":
        return np.array([s2, 0.3])
    if rtype == "TP":
        return np.r_[s2, 0.3 ** np.arange(1, t)]
    return np.array([s2 if i == j else 0.3 * s2 for i in range(t) for j in range(i + 1)])


class _Likelihood:
    """(RE)ML criterion for y = X b + Z u + e, u ~ N(0, G) per subject and
    e ~ N(0, R) per subject with R from a repeated covariance structure."""

    def __init__(self, y, X, Z, groups, reml: bool, structure: str, rtype: str = "ID",
                 rpos: np.ndarray | None = None, t: int = 1):
        self.reml = reml
        self.structure = structure
        self.rtype = rtype
        self.t = t
        self.q = Z.shape[1]
        self.N = len(y)
        self.blocks = []
        self.index = []
        rpos = np.zeros(len(y), int) if rpos is None else rpos
        for g in pd.unique(groups):
            m = np.flatnonzero(groups == g)
            self.index.append(m)
            self.blocks.append((y[m], X[m], Z[m], rpos[m]))
        self.p = int(np.linalg.matrix_rank(X))
        if self.q == 0:
            self.pairs = []
        elif structure == "UN":
            self.pairs = [(i, j) for i in range(self.q) for j in range(i + 1)]
        else:
            self.pairs = [(i, i) for i in range(self.q)]
        self.nr = len(_r_names(rtype, t, [""] * t))

    @property
    def n_theta(self) -> int:
        return len(self.pairs) + self.nr

    def unpack(self, theta):
        G = np.zeros((self.q, self.q))
        for (i, j), v in zip(self.pairs, theta[:len(self.pairs)]):
            G[i, j] = G[j, i] = v
        return G, np.asarray(theta[len(self.pairs):], float)

    def pack(self, G, rparams):
        return np.array([G[i, j] for i, j in self.pairs] + list(np.atleast_1d(rparams)))

    def residual_scale(self, theta) -> float:
        _, rp = self.unpack(theta)
        R = _r_matrix(self.rtype, rp, self.t)
        return float(np.mean(np.diag(R)))

    def V_block(self, theta, k: int) -> np.ndarray:
        G, rp = self.unpack(theta)
        _, _, Z, pos = self.blocks[k]
        return Z @ G @ Z.T + self._r_block(rp, pos)

    def _r_block(self, rp, pos) -> np.ndarray:
        if self.rtype == "ID":
            return rp[0] * np.eye(len(pos))
        R = _r_matrix(self.rtype, rp, self.t)
        return R[np.ix_(pos, pos)]

    def gls(self, theta, X_alt=None):
        """Returns (-2 log-likelihood, beta, C=(X'V^-1X)^-1) for the given parameters."""
        G, rp = self.unpack(theta)
        R = _r_matrix(self.rtype, rp, self.t)
        if np.any(np.diag(R) <= 0):
            return math.inf, None, None
        logdet = 0.0
        p = (X_alt if X_alt is not None else self.blocks[0][1]).shape[1]
        XtVX = np.zeros((p, p))
        XtVy = np.zeros(p)
        ytVy = 0.0
        for (y, X, Z, pos), idx in zip(self.blocks, self.index):
            if X_alt is not None:
                X = X_alt[idx]
            V = Z @ G @ Z.T + (rp[0] * np.eye(len(pos)) if self.rtype == "ID"
                               else R[np.ix_(pos, pos)])
            try:
                cf = linalg.cho_factor(V, lower=True)
            except linalg.LinAlgError:
                return math.inf, None, None
            logdet += 2 * float(np.sum(np.log(np.diag(cf[0]))))
            vx = linalg.cho_solve(cf, X)
            vy = linalg.cho_solve(cf, y)
            XtVX += X.T @ vx
            XtVy += X.T @ vy
            ytVy += float(y @ vy)
        try:
            C = np.linalg.inv(XtVX)
        except np.linalg.LinAlgError:
            C = np.linalg.pinv(XtVX)
        beta = C @ XtVy
        rvr = ytVy - float(beta @ XtVy)
        if self.reml:
            sign, ld = np.linalg.slogdet(XtVX)
            val = logdet + ld + rvr + (self.N - self.p) * LOG2PI
        else:
            val = logdet + rvr + self.N * LOG2PI
        return val, beta, C

    def value(self, theta) -> float:
        return self.gls(theta)[0]

    # optimiser in an unconstrained (Cholesky / log / atanh) parametrisation
    def _from_free(self, z):
        ng = len(self.pairs)
        rp = _r_natural(self.rtype, z[ng:], self.t)
        if self.q == 0:
            return np.array(rp)
        L = np.zeros((self.q, self.q))
        k = 0
        if self.structure == "UN":
            for i in range(self.q):
                for j in range(i + 1):
                    L[i, j] = math.exp(z[k]) if i == j else z[k]
                    k += 1
        else:
            for i in range(self.q):
                L[i, i] = math.exp(z[k])
                k += 1
        return self.pack(L @ L.T, rp)

    def _to_free(self, theta):
        G, rp = self.unpack(theta)
        s2 = max(self.residual_scale(theta), 1e-12)
        z = []
        if self.q:
            if self.structure == "UN":
                L = np.linalg.cholesky(G + 1e-10 * np.eye(self.q) * s2)
                for i in range(self.q):
                    for j in range(i + 1):
                        z.append(math.log(max(L[i, j], 1e-8)) if i == j else L[i, j])
            else:
                z = [0.5 * math.log(max(G[i, i], 1e-12 * s2)) for i in range(self.q)]
        return np.array(z + _r_free(self.rtype, rp, self.t))

    def polish(self, theta0):
        z0 = self._to_free(theta0)

        def f(z):
            try:
                return self.value(self._from_free(z))
            except (OverflowError, ValueError, np.linalg.LinAlgError):
                return math.inf

        res = optimize.minimize(f, z0, method="BFGS", options={"gtol": 1e-7, "maxiter": 1000})
        if not np.isfinite(res.fun):
            res = optimize.minimize(f, z0, method="Nelder-Mead",
                                    options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-10})
        best = self._from_free(res.x)
        return best if self.value(best) <= self.value(theta0) + 1e-9 else theta0


def _hessian(f, x, active, rel=1e-4):
    k = len(x)
    H = np.full((k, k), np.nan)
    h = np.array([rel * max(abs(v), 1e-3) for v in x])
    f0 = f(x)
    for a in active:
        for b in active:
            if b < a:
                continue
            if a == b:
                xp, xm = x.copy(), x.copy()
                xp[a] += h[a]
                xm[a] -= h[a]
                H[a, a] = (f(xp) - 2 * f0 + f(xm)) / h[a] ** 2
            else:
                vals = []
                for sa, sb in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    xx = x.copy()
                    xx[a] += sa * h[a]
                    xx[b] += sb * h[b]
                    vals.append(f(xx))
                H[a, b] = H[b, a] = (vals[0] - vals[1] - vals[2] + vals[3]) / (4 * h[a] * h[b])
    return H, h


def c_gradients(lik: _Likelihood, theta, active, X, steps) -> tuple[np.ndarray, list]:
    """C = (X'V^-1X)^-1 at theta and its numerical derivatives along the active parameters."""
    C0 = lik.gls(theta, X)[2]
    grads = []
    for a in active:
        tp, tm = theta.copy(), theta.copy()
        tp[a] += steps[a]
        tm[a] -= steps[a]
        grads.append((lik.gls(tp, X)[2] - lik.gls(tm, X)[2]) / (2 * steps[a]))
    return C0, grads


def satterthwaite_df(L: np.ndarray, C0: np.ndarray, grads: list, A: np.ndarray) -> \
        tuple[float, np.ndarray]:
    """Denominator df for the rows of L (Satterthwaite; Fai-Cornelius for several rows)
    and L C L'."""
    LCL = L @ C0 @ L.T
    w, P = np.linalg.eigh(LCL)
    nus = []
    for m in range(len(w)):
        lm = P[:, m] @ L
        g = np.array([lm @ dC @ lm for dC in grads])
        var = float(g @ A @ g) if len(g) else 0.0
        nus.append(2 * w[m] ** 2 / var if var > 0 else math.inf)
    nus = np.array(nus)
    q = len(w)
    if q == 1:
        return float(nus[0]), LCL
    big = nus[nus > 2]
    E = float(np.sum(big / (big - 2))) + (q - len(big))
    df = 2 * E / (E - q) if E > q else float(np.min(nus))
    return df, LCL


ADJUST = {"lsd": "Least Significant Difference (equivalent to no adjustments)",
          "bonferroni": "Bonferroni", "sidak": "Sidak"}


def linear_mixed(ds: Dataset, dependent: str, subject: str, factors: list[str] | None = None,
                 covariates: list[str] | None = None, random_intercept: bool = True,
                 random_slopes: list[str] | None = None, structure: str = "VC",
                 method: str = "REML", main_effects_only: bool = False,
                 descriptives: bool = False, parameter_estimates: bool = True,
                 covariance_estimates: bool = True, level: float = 0.95,
                 fixed_terms: list[TermSpec] | None = None, intercept: bool = True,
                 repeated: str | None = None, repeated_type: str = "ID",
                 emm: list[str] | None = None, compare: bool = False, adjust: str = "lsd",
                 save: list[str] | None = None, icc: bool = False):
    """SPSS MIXED. Returns a Section, or a :class:`MixedResult` when ``save`` is given
    (FIXPRED, PRED, RESID, SEFIXP)."""
    factors = factors or []
    covariates = covariates or []
    random_slopes = [s for s in (random_slopes or []) if s in covariates]
    emm = emm or []
    save = save or []
    repeated_type = repeated_type if repeated else "ID"
    reml = method.upper() == "REML"
    section = Section("Mixed Model Analysis")
    dvar = ds.var(dependent)
    svar = ds.var(subject)
    fnames = [ds.var(f).name for f in factors]
    cnames = [ds.var(c).name for c in covariates]
    if fixed_terms is None:
        # SPSS MIXED lists the factor effects before the covariates
        fixed_terms = (main_effects(fnames, []) if main_effects_only else
                       full_factorial(fnames, [])) + [TermSpec(covariates=(c,)) for c in cnames]
    rnames = (["Intercept"] if random_intercept else []) + [ds.var(s).name
                                                            for s in random_slopes]
    log = f"MIXED {dependent}"
    if factors:
        log += f" BY {' '.join(factors)}"
    if covariates:
        log += f" WITH {' '.join(covariates)}"
    log += (f"\n  /CRITERIA=CIN({level * 100:g}) MXITER(100) SCORING(1)"
            f"\n  /FIXED={' '.join(t.syntax for t in fixed_terms)} | SSTYPE(3)"
            + ("" if intercept else " NOINT") +
            f"\n  /METHOD={'REML' if reml else 'ML'}")
    prints = [p for p, on in (("DESCRIPTIVES", descriptives), ("G", False),
                              ("SOLUTION", parameter_estimates),
                              ("TESTCOV", covariance_estimates)) if on]
    if prints:
        log += f"\n  /PRINT={' '.join(prints)}"
    if rnames:
        log += (f"\n  /RANDOM={' '.join(n.upper() if n == 'Intercept' else n for n in rnames)}"
                f" | SUBJECT({svar.name}) COVTYPE({'UN' if structure == 'UN' else 'VC'})")
    if repeated:
        log += (f"\n  /REPEATED={ds.var(repeated).name} | SUBJECT({svar.name}) "
                f"COVTYPE({repeated_type})")
    for e in emm:
        log += f"\n  /EMMEANS=TABLES({'OVERALL' if e == '(OVERALL)' else e})"
        if compare and e != "(OVERALL)" and "*" not in e:
            log += f" COMPARE ADJ({adjust.upper()})"
    if save:
        log += "\n  /SAVE=" + " ".join(save)
    section.log(log + ".")

    def done(saved=None):
        if save:
            out = MixedResult(section, saved or {})
            out.models = getattr(section, "models", [])
            return out
        return section

    names = [dependent, subject] + factors + covariates + ([repeated] if repeated else [])
    frame = _frame(ds, list(dict.fromkeys(names)), {dependent, *covariates})
    if len(frame) < 3:
        section.note("Warning", "Not enough valid cases for the model.", style="warning")
        return done()
    fobjs = [make_factor(ds, f, frame[f]) for f in factors]
    covs = {ds.var(c).name: frame[c].to_numpy(float) for c in covariates}
    y = frame[dependent].to_numpy(float)
    n = len(y)
    groups = frame[subject].map(lambda v: repr(v)).to_numpy()
    try:
        Xe, eff_idx, Xr, ref_info, design = _designs_spec(fobjs, covs, fixed_terms, n,
                                                          intercept)
    except KeyError as exc:
        section.note("Warning", f"Unknown variable in the fixed-effects model: {exc}",
                     style="warning")
        return done()
    zcols = ([np.ones(n)] if random_intercept else []) + [covs[ds.var(s).name]
                                                          for s in random_slopes]
    Z = np.column_stack(zcols) if zcols else np.zeros((n, 0))
    q = Z.shape[1]
    rpos, t, rlabels = None, 1, [""]
    if repeated:
        rvals = frame[repeated]
        levels = sorted(rvals.unique(), key=lambda v: (isinstance(v, str), v))
        index = {lv: i for i, lv in enumerate(levels)}
        rpos = rvals.map(index).to_numpy(int)
        t = len(levels)
        rv = ds.var(repeated)
        rlabels = [f"{rv.name} = {value_text_plain(lv)}" for lv in levels]
        dup = pd.DataFrame({"g": groups, "r": rpos}).duplicated().any()
        if dup:
            section.note("Warning", "The repeated variable has duplicate values within a "
                         "subject; each subject must have at most one case per repeated "
                         "level.", style="warning")
            return done()
    lik = _Likelihood(y, Xr, Z, groups, reml, structure if q > 1 else "VC", repeated_type,
                      rpos, t)

    theta = _fit(y, Xr, Z, groups, lik, reml, n)
    G, rp = lik.unpack(theta)
    scale = lik.residual_scale(theta)
    val, beta_r, C_r = lik.gls(theta)
    _, beta_e, C_e = lik.gls(theta, Xe)

    ng = len(lik.pairs)
    active = [i for i, (a, b) in enumerate(lik.pairs) if a != b or G[a, a] > 1e-6 * scale]
    active += list(range(ng, lik.n_theta))
    redundant = [i for i in range(lik.n_theta) if i not in active]
    H, steps = _hessian(lik.value, theta, active)
    A = np.zeros((lik.n_theta, lik.n_theta))
    sub = H[np.ix_(active, active)]
    if not np.all(np.isfinite(sub)) or np.min(np.linalg.eigvalsh(np.nan_to_num(sub))) <= 0:
        section.note("Warning", "The final Hessian matrix is not positive definite although all "
                     "convergence criteria are satisfied. The MIXED procedure continues despite "
                     "this warning. Validity of subsequent results cannot be ascertained.",
                     style="warning")
        sub = np.nan_to_num(sub)
    try:
        A_act = 2 * np.linalg.inv(sub)
    except np.linalg.LinAlgError:
        A_act = 2 * np.linalg.pinv(sub)
    A[np.ix_(active, active)] = A_act
    A_sub = A[np.ix_(active, active)]

    cache = {}

    def sat(L, X):
        key = id(X)
        if key not in cache:
            cache[key] = c_gradients(lik, theta, active, X, steps)
        return satterthwaite_df(L, *cache[key], A_sub)

    labels_terms = ([("Intercept", eff_idx[0])] if intercept else []) + \
        [(t_.label, eff_idx[i + 1]) for i, t_ in enumerate(fixed_terms)]

    # ---------------------------------------------------------------- Model Dimension
    rows = []
    n_par = 0
    n_lev = 0
    first = True
    for (label, cols), spec in zip(labels_terms, ([None] if intercept else []) + fixed_terms):
        if spec is None or not spec.all_factors:
            levels = 1
        else:
            levels = int(np.prod([len(fobjs[[f.name for f in fobjs].index(fn)].levels)
                                  for fn in list(spec.factors) + list(spec.within)]))
        rows.append(["Fixed Effects" if first else "", label, fmt_int(levels), "",
                     fmt_int(len(cols)), "", ""])
        first = False
        n_par += len(cols)
        n_lev += levels
    if q:
        k = len(lik.pairs)
        rows.append(["Random Effects", " + ".join(rnames) + sup("a"), fmt_int(q),
                     "Unstructured" if lik.structure == "UN" else "Variance Components",
                     fmt_int(k), svar.name, ""])
        n_par += k
        n_lev += q
    if repeated:
        rows.append(["Repeated Effects", ds.var(repeated).name, fmt_int(t),
                     REPEATED_TYPES[repeated_type], fmt_int(lik.nr), svar.name,
                     fmt_int(len(lik.blocks))])
    else:
        rows.append(["Residual", "", "", "", "1", "", ""])
    n_par += lik.nr
    rows.append(["Total", "", fmt_int(n_lev + (t if repeated else 0)), "", fmt_int(n_par), "",
                 ""])
    foot = [f"{'a' if not q else 'b'}. Dependent Variable: {dvar.title}."]
    if q:
        foot = ["a. As of version 11.5, the syntax rules for the RANDOM subcommand have "
                "changed. Your command syntax may yield results that differ from those produced "
                "by prior versions.", f"b. Dependent Variable: {dvar.title}."]
    cols = ["", "", "Number of Levels", "Covariance Structure", "Number of Parameters",
            "Subject Variables"]
    if repeated:
        cols.append("Number of Subjects")
    else:
        rows = [r[:-1] for r in rows]
    section.table(Table("Model Dimension" + sup("b" if q else "a"), cols, rows,
                        row_header_cols=2, footnotes=foot))

    if descriptives and fobjs:
        drows = nested_descriptive_rows(fobjs, y, dvar.decimals)
        section.table(Table("Descriptive Statistics" + sup("a"),
                            [f.name for f in fobjs] + ["Mean", "Std. Deviation", "Count"],
                            drows, row_header_cols=len(fobjs),
                            footnotes=[f"a. Dependent Variable: {dvar.title}."]))

    # ---------------------------------------------------------------- Information Criteria
    d = lik.n_theta - len(redundant) + (0 if reml else lik.p)
    nstar = n - lik.p if reml else n
    crit = [("-2 Restricted Log Likelihood" if reml else "-2 Log Likelihood", val),
            ("Akaike's Information Criterion (AIC)", val + 2 * d),
            ("Hurvich and Tsai's Criterion (AICC)", val + 2 * d * nstar / (nstar - d - 1)),
            ("Bozdogan's Criterion (CAIC)", val + d * (math.log(nstar) + 1)),
            ("Schwarz's Bayesian Criterion (BIC)", val + d * math.log(nstar))]
    section.table(Table("Information Criteria" + sup("a"), ["", ""],
                        [[k, fmt(v)] for k, v in crit],
                        footnotes=["The information criteria are displayed in smaller-is-better "
                                   "form.", f"a. Dependent Variable: {dvar.title}."]))

    # ---------------------------------------------------------------- Type III tests
    rows = []
    for label, idx in labels_terms:
        if not idx:
            continue
        L = np.zeros((len(idx), Xe.shape[1]))
        for r, c in enumerate(idx):
            L[r, c] = 1
        df, LCL = sat(L, Xe)
        b = L @ beta_e
        F = float(b @ np.linalg.pinv(LCL) @ b) / len(idx)
        rows.append([label, fmt_int(len(idx)), fmt(df), fmt(F), fmt_p(stats.f.sf(F, len(idx),
                                                                                  df))])
    section.table(Table("Type III Tests of Fixed Effects" + sup("a"),
                        ["Source", "Numerator df", "Denominator df", "F", "Sig."], rows,
                        footnotes=[f"a. Dependent Variable: {dvar.title}."]))

    # ---------------------------------------------------------------- Estimates
    pe = []
    for label, cols_ in ref_info:
        if cols_ is not None:
            L = np.zeros((1, Xr.shape[1]))
            L[0, cols_[0]] = 1
            df, LCL = sat(L, Xr)
            pe.append((label, float(beta_r[cols_[0]]), math.sqrt(float(LCL[0, 0])), df))
    if pe:
        from rsstatchem.core.stats.resample import ModelInfo
        section.models = [ModelInfo([r[0] for r in pe], np.array([r[1] for r in pe]),
                                    np.array([r[2] for r in pe]), n=len(frame),
                                    table="Estimates of Fixed Effects",
                                    headers=["Parameter"], est_label="Estimate",
                                    df=np.array([r[3] for r in pe], float), bootable=False,
                                    note="Bootstrapping is not available for Linear Mixed "
                                         "Models (as in SPSS).")]
    if parameter_estimates:
        rows = []
        any_red = False
        pe_iter = iter(pe)
        for label, cols_ in ref_info:
            if cols_ is None:
                rows.append([label, "0" + sup("b"), "", "", "", "", "", ""])
                any_red = True
                continue
            _, est, se, df = next(pe_iter)
            tt = est / se if se > 0 else float("nan")
            qt = stats.t.ppf(0.5 + level / 2, df)
            rows.append([label, fmt(est), fmt(se), fmt(df), fmt(tt),
                         fmt_p(2 * stats.t.sf(abs(tt), df)), fmt(est - qt * se),
                         fmt(est + qt * se)])
        foot = [f"a. Dependent Variable: {dvar.title}."]
        if any_red:
            foot.append("b. This parameter is set to zero because it is redundant.")
        section.table(Table("Estimates of Fixed Effects" + sup("a"),
                            ["Parameter", "Estimate", "Std. Error", "df", "t", "Sig.",
                             (f"{pct(level)} Confidence Interval", "Lower Bound"),
                             (f"{pct(level)} Confidence Interval", "Upper Bound")],
                            rows, footnotes=foot))

    # ---------------------------------------------------------------- Covariance parameters
    if covariance_estimates:
        z = stats.norm.ppf(0.5 + level / 2)
        rows = []

        def add(head, sub_, i, variance=True):
            est = float(theta[i])
            if i in redundant:
                rows.append([head, sub_, fmt(0.0) + sup("b"), "", "", "", "", ""])
                return
            se = math.sqrt(max(A[i, i], 0.0))
            wz = est / se if se > 0 else float("nan")
            p = 2 * stats.norm.sf(abs(wz))
            if variance and est > 0 and se > 0:
                ratio = min(z * se / est, 700.0)
                lo, hi = est * math.exp(-ratio), est * math.exp(ratio)
            elif not variance and abs(est) < 1 and "rho" in sub_:
                # correlations: Fisher-z interval, as SPSS does for rho parameters
                zz, sez = math.atanh(est), se / (1 - est * est)
                lo, hi = math.tanh(zz - z * sez), math.tanh(zz + z * sez)
            else:
                lo, hi = est - z * se, est + z * se
            rows.append([head, sub_, fmt(est), fmt(se), fmt(wz), fmt_p(p), fmt(lo), fmt(hi)])

        if repeated:
            for k, (name, var_) in enumerate(_r_names(repeated_type, t, rlabels)):
                head = f"Repeated Measures" if k == 0 else ""
                add(head, name, ng + k, var_)
        else:
            add("Residual", "", lik.n_theta - 1)
        for k, (a, b) in enumerate(lik.pairs):
            if lik.structure == "UN":
                head = f"{' + '.join(rnames)} [subject = {svar.name}]" if k == 0 else ""
                add(head, f"UN ({a + 1},{b + 1})", k, variance=a == b)
            else:
                add(f"{rnames[a]} [subject = {svar.name}]", "Variance", k)
        foot = [f"a. Dependent Variable: {dvar.title}."]
        if redundant:
            foot.append("b. This covariance parameter is redundant. The test statistic and "
                        "confidence interval cannot be computed.")
        section.table(Table("Estimates of Covariance Parameters" + sup("a"),
                            ["Parameter", "", "Estimate", "Std. Error", "Wald Z", "Sig.",
                             (f"{pct(level)} Confidence Interval", "Lower Bound"),
                             (f"{pct(level)} Confidence Interval", "Upper Bound")],
                            rows, row_header_cols=2, footnotes=foot))

    if icc and random_intercept and q == 1 and repeated_type == "ID":
        su, s2 = float(G[0, 0]), float(rp[0])
        section.table(Table("Intraclass Correlation" + sup("a"), ["", "Estimate"],
                            [["Intraclass correlation (subject)", fmt(su / (su + s2))]],
                            footnotes=[f"a. Dependent Variable: {dvar.title}.",
                                       "Var(Intercept) / (Var(Intercept) + Var(Residual)) of "
                                       "the random-intercept model."]))

    # ---------------------------------------------------------------- EM means
    if emm:
        _mixed_emm(section, emm, design, fobjs, beta_e, Xe, sat, compare, adjust, level, dvar,
                   design.cov_means)

    saved = {}
    if save:
        saved = _mixed_save(save, lik, theta, Xr, beta_r, C_r, Z, frame.index, ds.df.index,
                            y, dvar.name)
    return done(saved)


def _mixed_emm(section, emm, design, fobjs, beta, Xe, sat, compare, adjust, level, dvar,
               cov_means) -> None:
    names = [f.name for f in fobjs]
    foot = [f"a. Dependent Variable: {dvar.title}."]
    if cov_means:
        foot.append("b. Covariates appearing in the model are evaluated at the following "
                    "values: " + ", ".join(f"{k} = {fmt(v, 4)}" for k, v in cov_means.items())
                    + ".")
    ci = [(f"{pct(level)} Confidence Interval", "Lower Bound"),
          (f"{pct(level)} Confidence Interval", "Upper Bound")]

    def est(l):
        df, LCL = sat(l[None, :], Xe)
        e = float(l @ beta)
        se = math.sqrt(max(float(LCL[0, 0]), 0.0))
        qt = stats.t.ppf(0.5 + level / 2, df)
        return e, se, df, e - qt * se, e + qt * se

    for spec in emm:
        if spec == "(OVERALL)":
            e, se, df, lo, hi = est(design.average_row({}))
            section.table(Table("Grand Mean" + sup("a"), ["Mean", "Std. Error", "df"] + ci,
                                [[fmt(e), fmt(se), fmt(df), fmt(lo), fmt(hi)]],
                                row_header_cols=0, footnotes=foot))
            continue
        parts = [p.strip() for p in spec.split("*")]
        if any(p not in names for p in parts):
            continue
        idx = [names.index(p) for p in parts]
        rows = []
        for combo in itertools.product(*[range(len(fobjs[i].levels)) for i in idx]):
            e, se, df, lo, hi = est(design.average_row(dict(zip(idx, combo))))
            rows.append([fobjs[i].labels[lv] for i, lv in zip(idx, combo)] +
                        [fmt(e), fmt(se), fmt(df), fmt(lo), fmt(hi)])
        section.table(Table(f"Estimates — {spec}" + sup("a"), parts +
                            ["Mean", "Std. Error", "df"] + ci, rows, row_header_cols=len(parts),
                            footnotes=foot))
        if compare and len(idx) == 1:
            f = fobjs[idx[0]]
            k = len(f.levels)
            m = k * (k - 1) / 2
            ls = [design.average_row({idx[0]: lv}) for lv in range(k)]
            prow, flagged = [], False
            for a in range(k):
                for jj, b in enumerate([x for x in range(k) if x != a]):
                    diff, se, df, _, _ = est(ls[a] - ls[b])
                    tt = diff / se if se > 0 else float("nan")
                    p = float(2 * stats.t.sf(abs(tt), df))
                    p = min(1.0, p * m) if adjust == "bonferroni" else \
                        min(1.0, 1 - (1 - p) ** m) if adjust == "sidak" else p
                    alpha = 1 - level
                    a_adj = alpha / m if adjust == "bonferroni" else \
                        1 - (1 - alpha) ** (1 / m) if adjust == "sidak" else alpha
                    crit = stats.t.ppf(1 - a_adj / 2, df)
                    star = "*" if p < 1 - level else ""
                    flagged |= bool(star)
                    prow.append([f.labels[a] if jj == 0 else "", f.labels[b], fmt(diff) + star,
                                 fmt(se), fmt(df), fmt_p(p), fmt(diff - crit * se),
                                 fmt(diff + crit * se)])
            notes = ["Based on estimated marginal means"]
            if flagged:
                notes.append(f"*. The mean difference is significant at the {1 - level:.2f} "
                             "level.")
            notes += [f"a. Dependent Variable: {dvar.title}.",
                      f"b. Adjustment for multiple comparisons: {ADJUST[adjust]}."]
            section.table(Table(f"Pairwise Comparisons — {f.name}" + sup("a"),
                                [f"(I) {f.name}", f"(J) {f.name}", "Mean Difference (I-J)",
                                 "Std. Error", "df", f"Sig.{sup('b')}",
                                 (f"{pct(level)} Confidence Interval for Difference"
                                  f"{sup('b')}", "Lower Bound"),
                                 (f"{pct(level)} Confidence Interval for Difference"
                                  f"{sup('b')}", "Upper Bound")], prow, row_header_cols=2,
                                footnotes=notes))
            L = np.array(ls[:-1]) - np.array(ls[1:])
            df, LCL = sat(L, Xe)
            b = L @ beta
            F = float(b @ np.linalg.pinv(LCL) @ b) / (k - 1)
            section.table(Table(f"Univariate Tests — {f.name}" + sup("a"),
                                ["Numerator df", "Denominator df", "F", "Sig."],
                                [[fmt_int(k - 1), fmt(df), fmt(F),
                                  fmt_p(stats.f.sf(F, k - 1, df))]], row_header_cols=0,
                                footnotes=[f"The F tests the effect of {f.name}. This test is "
                                           "based on the linearly independent pairwise "
                                           "comparisons among the estimated marginal means.",
                                           f"a. Dependent Variable: {dvar.title}."]))


def _mixed_save(keys, lik: _Likelihood, theta, X, beta, C, Z, index, full_index, y,
                dep: str) -> dict:
    n = len(y)
    fixed = X @ beta
    pred = fixed.copy()
    G, _ = lik.unpack(theta)
    if lik.q:
        for k, idx in enumerate(lik.index):
            V = lik.V_block(theta, k)
            u = G @ Z[idx].T @ np.linalg.solve(V, y[idx] - fixed[idx])
            pred[idx] = fixed[idx] + Z[idx] @ u
    sefix = np.sqrt(np.einsum("ij,jk,ik->i", X, C, X))
    vals = {"FIXPRED": ("Fixed Predicted Values", fixed),
            "SEFIXP": ("Standard Errors of Fixed Predicted Values", sefix),
            "PRED": ("Predicted Values", pred), "RESID": ("Residuals", y - pred)}
    out = {}
    for key in ("FIXPRED", "SEFIXP", "PRED", "RESID"):
        if key in keys:
            s = pd.Series(np.nan, index=full_index)
            s.loc[index] = vals[key][1]
            out[key] = (f"{vals[key][0]} for {dep}", s)
    del n
    return out


def _fit(y, X, Z, groups, lik: _Likelihood, reml: bool, n: int) -> np.ndarray:
    """Covariance parameters: statsmodels MixedLM start (G with scaled-identity residual),
    then polished on the SPSS criterion with the requested residual structure."""
    q = Z.shape[1]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    rss = float(np.sum((y - X @ beta) ** 2))
    s2_ols = rss / (n - lik.p if reml else n)
    if q == 0 and lik.rtype == "ID":
        return np.array([s2_ols])
    theta0 = None
    if q:
        try:
            import statsmodels.api as sm
            from statsmodels.regression.mixed_linear_model import MixedLMParams

            model = sm.MixedLM(y, X, groups=groups, exog_re=Z)
            kw = {}
            if lik.structure != "UN" and q > 1:
                kw["free"] = MixedLMParams.from_components(np.ones(X.shape[1]), np.eye(q))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = model.fit(reml=reml, method=["lbfgs", "bfgs", "powell"], **kw)
            G = np.asarray(res.cov_re, float)
            if lik.structure != "UN":
                G = np.diag(np.diag(G))
            theta0 = lik.pack(G, _r_start(lik.rtype, float(res.scale), lik.t))
            if lik.rtype == "ID":
                theta0 = lik.pack(G, [float(res.scale)])
            if not np.isfinite(lik.value(theta0)):
                theta0 = None
        except Exception:  # noqa: BLE001 - fall back to our own starting values
            theta0 = None
    if theta0 is None:
        frac = 0.5 if q else 1.0
        theta0 = lik.pack(np.eye(q) * s2_ols * 0.5, _r_start(lik.rtype, s2_ols * frac, lik.t))
    return lik.polish(theta0)

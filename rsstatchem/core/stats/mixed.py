"""Analyze > Mixed Models > Linear, mirroring SPSS MIXED output for long-format data.

Fixed effects: factors (full factorial or main effects) and covariates. Random effects for a
subject variable: intercept and/or random slopes of covariates with a Variance Components
(diagonal) or Unstructured covariance matrix; residual covariance is scaled identity. The
model is fitted by REML or ML with statsmodels ``MixedLM`` and the SPSS tables are computed
from the fitted covariance parameters: Model Dimension, Information Criteria (-2LL, AIC, AICC,
CAIC, BIC in SPSS's conventions), Type III Tests of Fixed Effects and Estimates of Fixed
Effects (last category redundant, Satterthwaite denominator df) and Estimates of Covariance
Parameters (Wald Z, asymptotic SEs from the observed information).
"""
from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import dataclass

import numpy as np
from scipy import linalg, optimize, stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, fmt_p, pct, sup
from rsstatchem.core.stats.manova import (_frame, make_factor, nested_descriptive_rows,
                                          value_text_plain)

LOG2PI = math.log(2 * math.pi)


@dataclass
class _FTerm:
    label: str
    factors: tuple[int, ...] = ()
    covariate: str | None = None


def _designs(factors, covs: dict[str, np.ndarray], terms: list[_FTerm], n: int):
    """Effect-coded (Type III) and last-category-reference (estimates) design matrices."""
    eff_cols, ref_cols = [np.ones((n, 1))], [np.ones((n, 1))]
    eff_idx = [[0]]
    ref_info = [("Intercept", [0])]  # (label, column or None for redundant)
    pos_e = pos_r = 1
    for t in terms:
        if t.covariate is not None:
            x = covs[t.covariate][:, None]
            eff_cols.append(x)
            ref_cols.append(x)
            eff_idx.append([pos_e])
            ref_info.append((t.label, [pos_r]))
            pos_e += 1
            pos_r += 1
            continue
        mains_e, mains_r = [], []
        for fi in t.factors:
            f = factors[fi]
            m = len(f.levels)
            mains_e.append(np.column_stack([(f.codes == j).astype(float) -
                                            (f.codes == m - 1).astype(float)
                                            for j in range(m - 1)]))
            mains_r.append(np.column_stack([(f.codes == j).astype(float) for j in range(m)]))
        e = mains_e[0]
        for c in mains_e[1:]:
            e = np.einsum("ni,nj->nij", e, c).reshape(n, -1)
        eff_cols.append(e)
        eff_idx.append(list(range(pos_e, pos_e + e.shape[1])))
        pos_e += e.shape[1]
        combos = itertools.product(*[range(len(factors[fi].levels)) for fi in t.factors])
        for combo in combos:
            label = " * ".join(f"[{factors[fi].name}={value_text_plain(factors[fi].levels[lv])}]"
                               for fi, lv in zip(t.factors, combo))
            redundant = any(lv == len(factors[fi].levels) - 1 for fi, lv in zip(t.factors, combo))
            if redundant:
                ref_info.append((label, None))
                continue
            col = np.ones(n)
            for k, (fi, lv) in enumerate(zip(t.factors, combo)):
                col = col * mains_r[k][:, lv]
            ref_cols.append(col[:, None])
            ref_info.append((label, [pos_r]))
            pos_r += 1
    return np.hstack(eff_cols), eff_idx, np.hstack(ref_cols), ref_info


class _Likelihood:
    """(RE)ML criterion for y = X b + Z u + e, u ~ N(0, G) per subject, e ~ N(0, s2 I)."""

    def __init__(self, y, X, Z, groups, reml: bool, structure: str):
        self.reml = reml
        self.structure = structure
        self.q = Z.shape[1]
        self.N = len(y)
        self.blocks = []
        self.index = []
        for g in np.unique(groups):
            m = np.flatnonzero(groups == g)
            self.index.append(m)
            self.blocks.append((y[m], X[m], Z[m]))
        self.p = int(np.linalg.matrix_rank(X))
        if self.q == 0:
            self.pairs = []
        elif structure == "UN":
            self.pairs = [(i, j) for i in range(self.q) for j in range(i + 1)]
        else:
            self.pairs = [(i, i) for i in range(self.q)]

    @property
    def n_theta(self) -> int:
        return len(self.pairs) + 1

    def unpack(self, theta):
        G = np.zeros((self.q, self.q))
        for (i, j), v in zip(self.pairs, theta[:-1]):
            G[i, j] = G[j, i] = v
        return G, float(theta[-1])

    def pack(self, G, s2):
        return np.array([G[i, j] for i, j in self.pairs] + [s2])

    def gls(self, theta, X_alt=None):
        """Returns (-2 log-likelihood, beta, C=(X'V^-1X)^-1) for the given parameters."""
        G, s2 = self.unpack(theta)
        if s2 <= 0:
            return math.inf, None, None
        logdet = 0.0
        p = (X_alt if X_alt is not None else self.blocks[0][1]).shape[1]
        XtVX = np.zeros((p, p))
        XtVy = np.zeros(p)
        ytVy = 0.0
        for (y, X, Z), idx in zip(self.blocks, self.index):
            if X_alt is not None:
                X = X_alt[idx]
            V = Z @ G @ Z.T + s2 * np.eye(len(y))
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

    # optimiser in an unconstrained (Cholesky / log) parametrisation
    def _from_free(self, z):
        s2 = math.exp(z[-1])
        if self.q == 0:
            return np.array([s2])
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
        return self.pack(L @ L.T, s2)

    def _to_free(self, theta):
        G, s2 = self.unpack(theta)
        z = []
        if self.q:
            if self.structure == "UN":
                L = np.linalg.cholesky(G + 1e-10 * np.eye(self.q) * max(s2, 1e-12))
                for i in range(self.q):
                    for j in range(i + 1):
                        z.append(math.log(max(L[i, j], 1e-8)) if i == j else L[i, j])
            else:
                z = [0.5 * math.log(max(G[i, i], 1e-12 * s2)) for i in range(self.q)]
        return np.array(z + [math.log(s2)])

    def polish(self, theta0):
        z0 = self._to_free(theta0)
        res = optimize.minimize(lambda z: self.value(self._from_free(z)), z0, method="BFGS",
                                options={"gtol": 1e-7, "maxiter": 500})
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


def linear_mixed(ds: Dataset, dependent: str, subject: str, factors: list[str] | None = None,
                 covariates: list[str] | None = None, random_intercept: bool = True,
                 random_slopes: list[str] | None = None, structure: str = "VC",
                 method: str = "REML", main_effects_only: bool = False,
                 descriptives: bool = False, parameter_estimates: bool = True,
                 covariance_estimates: bool = True, level: float = 0.95) -> Section:
    factors = factors or []
    covariates = covariates or []
    random_slopes = [s for s in (random_slopes or []) if s in covariates]
    reml = method.upper() == "REML"
    section = Section("Mixed Model Analysis")
    dvar = ds.var(dependent)
    svar = ds.var(subject)
    fterms = []
    if factors:
        order = 1 if main_effects_only else len(factors)
        for r in range(1, order + 1):
            for combo in itertools.combinations(range(len(factors)), r):
                fterms.append(_FTerm(" * ".join(ds.var(factors[c]).name for c in combo), combo))
    for c in covariates:
        fterms.append(_FTerm(ds.var(c).name, covariate=c))
    rnames = (["Intercept"] if random_intercept else []) + [ds.var(s).name for s in random_slopes]
    log = f"MIXED {dependent}"
    if factors:
        log += f" BY {' '.join(factors)}"
    if covariates:
        log += f" WITH {' '.join(covariates)}"
    log += (f"\n  /CRITERIA=CIN({level * 100:g}) MXITER(100) SCORING(1)"
            f"\n  /FIXED={' '.join(t.label.replace(' ', '') for t in fterms)} | SSTYPE(3)"
            f"\n  /METHOD={'REML' if reml else 'ML'}")
    prints = [p for p, on in (("DESCRIPTIVES", descriptives), ("G", False),
                              ("SOLUTION", parameter_estimates),
                              ("TESTCOV", covariance_estimates)) if on]
    if prints:
        log += f"\n  /PRINT={' '.join(prints)}"
    if rnames:
        log += (f"\n  /RANDOM={' '.join(n.upper() if n == 'Intercept' else n for n in rnames)}"
                f" | SUBJECT({svar.name}) COVTYPE({'UN' if structure == 'UN' else 'VC'})")
    section.log(log + ".")

    frame = _frame(ds, [dependent, subject] + factors + covariates, {dependent, *covariates})
    if len(frame) < 3:
        section.note("Warning", "Not enough valid cases for the model.", style="warning")
        return section
    fobjs = [make_factor(ds, f, frame[f]) for f in factors]
    covs = {c: frame[c].to_numpy(float) for c in covariates}
    y = frame[dependent].to_numpy(float)
    n = len(y)
    groups = frame[subject].map(lambda v: repr(v)).to_numpy()
    Xe, eff_idx, Xr, ref_info = _designs(fobjs, covs, fterms, n)
    zcols = ([np.ones(n)] if random_intercept else []) + [covs[s] for s in random_slopes]
    Z = np.column_stack(zcols) if zcols else np.zeros((n, 0))
    q = Z.shape[1]
    lik = _Likelihood(y, Xr, Z, groups, reml, structure if q > 1 else "VC")
    if np.linalg.matrix_rank(Xr) < Xr.shape[1]:
        section.note("Warning", "The fixed-effects design is rank deficient (empty cells); "
                     "results may differ from SPSS.", style="warning")

    theta = _fit(y, Xr, Z, groups, lik, reml, n)
    G, s2 = lik.unpack(theta)
    val, beta_r, C_r = lik.gls(theta)
    _, beta_e, C_e = lik.gls(theta, Xe)

    active = [i for i, (a, b) in enumerate(lik.pairs) if a != b or G[a, a] > 1e-6 * s2]
    active.append(lik.n_theta - 1)
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

    # ---------------------------------------------------------------- Model Dimension
    rows = []
    n_par = 0
    n_lev = 0
    first = True
    for label, cols in [("Intercept", [0])] + [(t.label, eff_idx[i + 1])
                                               for i, t in enumerate(fterms)]:
        t = next((ft for ft in fterms if ft.label == label), None)
        levels = 1 if t is None or t.covariate else \
            int(np.prod([len(fobjs[fi].levels) for fi in t.factors]))
        rows.append(["Fixed Effects" if first else "", label, fmt_int(levels), "",
                     fmt_int(len(cols)), ""])
        first = False
        n_par += len(cols)
        n_lev += levels
    if q:
        k = len(lik.pairs)
        rows.append(["Random Effects", " + ".join(rnames) + sup("a"), fmt_int(q),
                     "Unstructured" if lik.structure == "UN" else "Variance Components",
                     fmt_int(k), svar.name])
        n_par += k
        n_lev += q
    rows.append(["Residual", "", "", "", "1", ""])
    n_par += 1
    rows.append(["Total", "", fmt_int(n_lev), "", fmt_int(n_par), ""])
    foot = [f"{'a' if not q else 'b'}. Dependent Variable: {dvar.title}."]
    if q:
        foot = ["a. As of version 11.5, the syntax rules for the RANDOM subcommand have "
                "changed. Your command syntax may yield results that differ from those produced "
                "by prior versions.", f"b. Dependent Variable: {dvar.title}."]
    section.table(Table("Model Dimension" + sup("b" if q else "a"),
                        ["", "", "Number of Levels", "Covariance Structure",
                         "Number of Parameters", "Subject Variables"], rows,
                        row_header_cols=2, footnotes=foot))

    if descriptives and fobjs:
        drows = nested_descriptive_rows(fobjs, y, dvar.decimals)
        section.table(Table("Descriptive Statistics" + sup("a"),
                            [f.name for f in fobjs] + ["Mean", "Std. Deviation", "Count"],
                            [r[:-3] + [r[-3], r[-2], r[-1]] for r in drows],
                            row_header_cols=len(fobjs),
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
    for label, idx in [("Intercept", eff_idx[0])] + [(t.label, eff_idx[i + 1])
                                                     for i, t in enumerate(fterms)]:
        L = np.zeros((len(idx), Xe.shape[1]))
        for r, c in enumerate(idx):
            L[r, c] = 1
        df, LCL = sat(L, Xe)
        b = L @ beta_e
        F = float(b @ np.linalg.solve(LCL, b)) / len(idx)
        rows.append([label, fmt_int(len(idx)), fmt(df), fmt(F), fmt_p(stats.f.sf(F, len(idx),
                                                                                  df))])
    section.table(Table("Type III Tests of Fixed Effects" + sup("a"),
                        ["Source", "Numerator df", "Denominator df", "F", "Sig."], rows,
                        footnotes=[f"a. Dependent Variable: {dvar.title}."]))

    # ---------------------------------------------------------------- Estimates
    if parameter_estimates:
        rows = []
        any_red = False
        for label, cols in ref_info:
            if cols is None:
                rows.append([label, "0" + sup("b"), "", "", "", "", "", ""])
                any_red = True
                continue
            L = np.zeros((1, Xr.shape[1]))
            L[0, cols[0]] = 1
            df, LCL = sat(L, Xr)
            est = float(beta_r[cols[0]])
            se = math.sqrt(float(LCL[0, 0]))
            t = est / se if se > 0 else float("nan")
            qt = stats.t.ppf(0.5 + level / 2, df)
            rows.append([label, fmt(est), fmt(se), fmt(df), fmt(t),
                         fmt_p(2 * stats.t.sf(abs(t), df)), fmt(est - qt * se),
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
            else:
                lo, hi = est - z * se, est + z * se
            rows.append([head, sub_, fmt(est), fmt(se), fmt(wz), fmt_p(p), fmt(lo), fmt(hi)])

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
    return section


def _fit(y, X, Z, groups, lik: _Likelihood, reml: bool, n: int) -> np.ndarray:
    """Covariance parameters from statsmodels MixedLM, polished on the SPSS criterion."""
    q = Z.shape[1]
    if q == 0:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        rss = float(np.sum((y - X @ beta) ** 2))
        return np.array([rss / (n - lik.p if reml else n)])
    theta0 = None
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
        theta0 = lik.pack(G, float(res.scale))
        if not np.isfinite(lik.value(theta0)):
            theta0 = None
    except Exception:  # noqa: BLE001 - fall back to our own optimiser
        theta0 = None
    if theta0 is None:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        s2 = float(np.var(y - X @ beta))
        theta0 = lik.pack(np.eye(q) * s2 * 0.5, s2 * 0.5)
    return lik.polish(theta0)

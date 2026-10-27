"""Analyze > General Linear Model > Variance Components, mirroring SPSS VARCOMP.

Model: dependent = fixed effects (intercept, fixed factors, covariates) + random effects (every
model term that involves a random factor) + error, by default the full factorial of all
factors. Estimation methods as in SPSS: ANOVA (Type I or Type III sums of squares with
expected mean squares), MINQUE(0), MINQUE(1), Maximum Likelihood and Restricted Maximum
Likelihood. Output: Factor Level Information, Variance Estimates, and for ANOVA the ANOVA table
and Expected Mean Squares; for ML/REML the asymptotic covariance matrix of the estimates.

Dense n x n matrices are used, which is fine for the few-thousand-case data sets of a thesis.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import optimize

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, sup
from rsstatchem.core.stats.glm_model import GLMFit, TermSpec, ems_table, full_factorial

SS_ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV"}
METHODS = {"anova": "ANOVA", "minque0": "Minimum Norm Quadratic Unbiased Estimation (MINQUE(0))",
           "minque1": "Minimum Norm Quadratic Unbiased Estimation (MINQUE(1))",
           "ml": "Maximum Likelihood Estimation", "reml": "Restricted Maximum Likelihood "
           "Estimation"}


def _split(fit: GLMFit, random: list[str]):
    """Fixed design X and random incidence matrices Z_k (one per random term)."""
    owner = np.array(fit.param_owner)
    rset = set(random)
    specs = fit.specs
    rterms = [k for k, s in enumerate(specs, start=1) if set(s.all_factors) & rset]
    fixed_cols = [j for j, o in enumerate(owner) if o not in rterms]
    X = fit.Xo[:, fixed_cols]
    X = X[:, ~_aliased(X)]
    Zs = [fit.Xo[:, owner == k] for k in rterms]
    return X, Zs, rterms


def _aliased(X: np.ndarray) -> np.ndarray:
    from rsstatchem.core.stats.glm_model import aliased_columns

    return aliased_columns(X) if X.shape[1] else np.zeros(0, bool)


def _projector(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    WX = W @ X
    return W - WX @ np.linalg.pinv(X.T @ WX) @ WX.T


def minque(y: np.ndarray, X: np.ndarray, Zs: list[np.ndarray], prior: list[float]
           ) -> np.ndarray:
    """MINQUE estimates (random components..., error) for the given prior weights."""
    n = len(y)
    Vs = [Z @ Z.T for Z in Zs] + [np.eye(n)]
    V0 = sum(w * V for w, V in zip(prior, Vs))
    W = np.linalg.inv(V0)
    P = _projector(X, W)
    PV = [P @ V for V in Vs]
    S = np.array([[np.sum(a * b.T) for b in PV] for a in PV])
    Py = P @ y
    q = np.array([float(Py @ V @ Py) for V in Vs])
    return np.linalg.solve(S, q)


def likelihood_estimates(y: np.ndarray, X: np.ndarray, Zs: list[np.ndarray], reml: bool
                         ) -> tuple[np.ndarray, np.ndarray, float]:
    """(ML/REML variance components, asymptotic covariance matrix, -2 log likelihood)."""
    n = len(y)
    Vs = [Z @ Z.T for Z in Zs] + [np.eye(n)]
    p = int(np.linalg.matrix_rank(X))

    def neg2ll(theta):
        theta = np.asarray(theta)
        if theta[-1] <= 0 or np.any(theta[:-1] < 0):
            return math.inf
        V = sum(t * Vk for t, Vk in zip(theta, Vs))
        try:
            L = np.linalg.cholesky(V)
        except np.linalg.LinAlgError:
            return math.inf
        logdet = 2 * np.sum(np.log(np.diag(L)))
        Vi = np.linalg.inv(V)
        XtVX = X.T @ Vi @ X
        beta = np.linalg.pinv(XtVX) @ X.T @ Vi @ y
        r = y - X @ beta
        val = logdet + float(r @ Vi @ r)
        if reml:
            val += np.linalg.slogdet(XtVX)[1] + (n - p) * math.log(2 * math.pi)
        else:
            val += n * math.log(2 * math.pi)
        return val

    start = np.clip(minque(y, X, Zs, [1.0] * len(Vs)), 1e-6, None)
    start[-1] = max(start[-1], 1e-4 * np.var(y) + 1e-12)
    z0 = 0.5 * np.log(start)
    res = optimize.minimize(lambda z: neg2ll(np.exp(2 * z)), z0, method="BFGS",
                            options={"gtol": 1e-9, "maxiter": 2000})
    theta = np.exp(2 * res.x)
    theta[theta < 1e-10 * theta[-1]] = 0.0
    # expected information (Fisher scoring form): I_ij = 1/2 tr(P V_i P V_j) (REML) or with
    # V^-1 for ML
    V = sum(t * Vk for t, Vk in zip(theta, Vs))
    Vi = np.linalg.inv(V)
    M = _projector(X, Vi) if reml else Vi
    MV = [M @ Vk for Vk in Vs]
    info = 0.5 * np.array([[np.sum(a * b.T) for b in MV] for a in MV])
    active = [i for i in range(len(theta)) if theta[i] > 0]
    cov = np.full((len(theta), len(theta)), np.nan)
    try:
        cov[np.ix_(active, active)] = np.linalg.inv(info[np.ix_(active, active)])
    except np.linalg.LinAlgError:
        pass
    return theta, cov, neg2ll(theta)


def variance_components(ds: Dataset, dependent: str, factors: list[str], random: list[str],
                        covariates: list[str] | None = None, method: str = "minque1",
                        ss_type: int = 3, terms: list[TermSpec] | None = None,
                        intercept: bool = True, exclude_intercept_prior: bool = False
                        ) -> Section:
    from rsstatchem.core.stats.manova import _frame, between_factors_table, make_factor

    covariates = covariates or []
    section = Section("Variance Components Estimation")
    allf = list(factors) + [r for r in random if r not in factors]
    log = f"VARCOMP {dependent} BY {' '.join(allf)}"
    if covariates:
        log += f" WITH {' '.join(covariates)}"
    log += f"\n  /RANDOM={' '.join(random)}\n  /METHOD=" + \
        {"anova": f"SSTYPE({ss_type})", "minque0": "MINQUE(0)", "minque1": "MINQUE(1)",
         "ml": "ML", "reml": "REML"}[method]
    tlist = terms if terms is not None else full_factorial(allf, covariates)
    log += "\n  /DESIGN=" + " ".join(t.syntax for t in tlist)
    log += f"\n  /INTERCEPT={'INCLUDE' if intercept else 'EXCLUDE'}."
    section.log(log)
    if not random:
        section.note("Warning", "At least one random factor is required.", style="warning")
        return section
    frame = _frame(ds, [dependent] + allf + covariates, {dependent, *covariates})
    fobjs = [make_factor(ds, f, frame[f]) for f in allf]
    covs = {ds.var(c).name: frame[c].to_numpy(float) for c in covariates}
    rnames = [ds.var(r).name for r in random]
    y = frame[dependent].to_numpy(float)
    fit = GLMFit(y, fobjs, covs, terms, intercept, ss_type if method == "anova" else 3)
    if fit.ve < 1:
        section.note("Warning", "There are no error degrees of freedom.", style="warning")
        return section
    info = between_factors_table(fobjs)
    info.title = "Factor Level Information"
    section.table(info)
    X, Zs, rterms = _split(fit, rnames)
    labels = [fit.specs[k - 1].label for k in rterms]
    dvar = ds.var(dependent)
    foot = [f"Dependent Variable: {dvar.title}", f"Method: {METHODS[method]}"]
    if method == "anova":
        ems = ems_table(fit, rnames)
        comps = ems["random"] + ["error"]
        rows_k = ems["random"]
        Mx = np.array([[ems["coef"][k].get(c, 0.0) for c in comps] for k in rows_k] +
                      [[0.0] * (len(comps) - 1) + [1.0]])
        msv = np.array([ems["ms"][k] for k in rows_k] + [ems["ms"]["error"]])
        theta = np.linalg.solve(Mx, msv)
        est_rows = [[f"Var({ems['labels'][k]})", fmt(v)] for k, v in zip(rows_k, theta[:-1])]
        est_rows.append(["Var(Error)", fmt(theta[-1])])
        section.table(Table("Variance Estimates", ["Component", "Estimate"], est_rows,
                            footnotes=foot))
        arows = []
        for k in ems["numbers"]:
            t = next(tt for tt in fit.terms if fit.term_number(tt) == k)
            ss = float(fit.hypothesis(t)[0, 0])
            df = ems["df"][k]
            arows.append([ems["labels"][k], fmt(ss), fmt_int(df), fmt(ss / df)])
        sse = float(fit.E[0, 0])
        arows.append(["Error", fmt(sse), fmt_int(fit.ve), fmt(sse / fit.ve)])
        sst = float(((y - y.mean()) ** 2).sum())
        arows.append(["Total", fmt(float((y ** 2).sum())), fmt_int(fit.n), ""])
        arows.append(["Corrected Total", fmt(sst), fmt_int(fit.n - 1), ""])
        section.table(Table("ANOVA", ["Source", f"Sum of Squares (Type {SS_ROMAN[ss_type]})",
                                      "df", "Mean Square"], arows,
                            footnotes=[f"Dependent Variable: {dvar.title}"]))
        cols = ["Source"] + [("Variance Component", f"Var({ems['labels'][c]})" if c != "error"
                              else "Var(Error)") for c in comps] + \
            [("Variance Component", "Quadratic Term")]
        erows = [[ems["labels"][k]] + [fmt(ems["coef"][k].get(c, 0.0)) for c in comps] +
                 [", ".join(ems["quad"][k])] for k in ems["numbers"]]
        erows.append(["Error"] + [fmt(0.0)] * (len(comps) - 1) + [fmt(1.0), ""])
        section.table(Table("Expected Mean Squares" + sup("a") + "," + sup("b"), cols, erows,
                            footnotes=[f"Dependent Variable: {dvar.title}",
                                       "a. For each source, the expected mean square equals "
                                       "the sum of the coefficients in the cells times the "
                                       "variance components, plus a quadratic term involving "
                                       "effects in the Quadratic Term cell.",
                                       f"b. Expected Mean Squares are based on the Type "
                                       f"{SS_ROMAN[ss_type]} Sums of "
                                       "Squares."]))
        if np.any(theta < 0):
            section.note("", "Negative variance component estimates are reported as obtained "
                         "(SPSS does the same for the ANOVA method).")
        return section
    if method in ("minque0", "minque1"):
        prior = [0.0 if method == "minque0" else 1.0] * len(Zs) + [1.0]
        theta = minque(y, X, Zs, prior)
        rows = [[f"Var({lab})", fmt(v)] for lab, v in zip(labels, theta[:-1])]
        rows.append(["Var(Error)", fmt(theta[-1])])
        section.table(Table("Variance Estimates", ["Component", "Estimate"], rows,
                            footnotes=foot))
        return section
    theta, cov, n2ll = likelihood_estimates(y, X, Zs, method == "reml")
    rows = [[f"Var({lab})", fmt(v)] for lab, v in zip(labels, theta[:-1])]
    rows.append(["Var(Error)", fmt(theta[-1])])
    if np.any(theta[:-1] == 0):
        foot.append("a. This estimate is set to zero because it is redundant.")
        rows = [[r[0], (r[1] + sup("a")) if r[1] in (".000", "0.000") else r[1]] for r in rows]
    section.table(Table("Variance Estimates", ["Component", "Estimate"], rows, footnotes=foot))
    names = [f"Var({lab})" for lab in labels] + ["Var(Error)"]
    crow = [[names[i]] + [fmt(cov[i, j]) if np.isfinite(cov[i, j]) else "."
                          for j in range(len(names))] for i in range(len(names))]
    section.table(Table("Asymptotic Covariance Matrix", [""] + names, crow,
                        footnotes=[f"Dependent Variable: {dvar.title}",
                                   f"Method: {METHODS[method]}",
                                   f"-2 {'Restricted ' if method == 'reml' else ''}Log "
                                   f"Likelihood = {fmt(n2ll)}"]))
    return section


__all__ = ["variance_components", "minque", "likelihood_estimates", "METHODS"]

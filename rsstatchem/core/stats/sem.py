"""Analyze > Structural Equation Modeling > CB-SEM: covariance-based SEM / CFA as in AMOS and
R lavaan (``sem()`` / ``cfa()`` defaults).

Model syntax: :mod:`sem_syntax` (lavaan style). Estimation, fit measures, modification indices
and bootstrap: :mod:`sem_fit`. Path diagram: :mod:`sem_diagram`. This module assembles the
Output Viewer section with AMOS/lavaan wording:

Model Summary (Notes for Model), Assessment of Normality (+ Mahalanobis distances), Model Fit
(chi-square, CMIN/DF, GFI, AGFI, NFI, IFI, TLI, CFI, RMSEA 90% CI + PCLOSE, SRMR, AIC, BIC with
cut-off criteria), Parameter Estimates (Estimate, S.E., C.R., P, Standardized), Squared
Multiple Correlations, Total / Direct / Indirect Effects, specific indirect and user-defined
(:=) effects with delta-method and bootstrap (percentile + bias-corrected) CIs, measurement
quality (standardised loadings, Composite Reliability, AVE, Cronbach's alpha), discriminant
validity (Fornell-Larcker, HTMT), Modification Indices, Standardized Residual Covariances and
the path diagram. Regression-method factor scores can be saved.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, listwise, pct
from rsstatchem.core.stats.reliability import cronbach_alpha
from rsstatchem.core.stats import sem_fit as sf
from rsstatchem.core.stats.sem_syntax import SEMModel, SEMSyntaxError, build_model

OUTPUTS = {"normality": True, "fit": True, "estimates": True, "r2": True, "effects": True,
           "measurement": True, "discriminant": True, "mi": True, "residuals": False,
           "diagram": True, "sample_cov": False, "implied_cov": False}
ARROW = {"=~": "<---", "~": "<---", "~~": "<-->"}


@dataclass
class SEMResult:
    section: Section
    saved: dict = field(default_factory=dict)
    est: sf.Estimates | None = None
    fit: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


# ------------------------------------------------------------------ derived quantities
class Derived:
    """Vectorised functions of theta: standardised estimates, effects, R², defined params.
    Used for delta-method SEs (numerical Jacobian) and for the bootstrap."""

    def __init__(self, est: sf.Estimates):
        self.est = est
        model, spec = est.model, est.spec
        self.model, self.spec = model, spec
        self.rows = list(model.params)
        struct = model.structural()
        self.struct = struct
        edges = np.zeros((len(struct), len(struct)), bool)  # edges[y, x]: x -> y
        loc = {v: k for k, v in enumerate(struct)}
        for p in model.params:
            if p.op == "~" and p.lhs in loc and p.rhs in loc:
                edges[loc[p.lhs], loc[p.rhs]] = True
            elif p.op == "=~" and p.lhs in loc and p.rhs in loc:
                edges[loc[p.rhs], loc[p.lhs]] = True
        reach = edges.copy()
        for _ in range(len(struct)):
            new = reach | ((reach.astype(int) @ edges.astype(int)) > 0)
            if (new == reach).all():
                break
            reach = new
        self.pairs = [(struct[i], struct[j]) for i in range(len(struct))
                      for j in range(len(struct)) if i != j and reach[i, j]]
        self.pair_idx = [(spec.pos[y], spec.pos[x]) for y, x in self.pairs]
        self.has_indirect = [_indirect_exists(edges, loc[y], loc[x])
                             for y, x in self.pairs]
        self.no_indirect = ~np.array(self.has_indirect, bool)
        self.paths = _simple_paths(struct, edges, limit=60)
        self.r2_vars = [v for v in model.endogenous]
        self.defined = model.defined
        label_first: dict[str, object] = {}
        for p in model.params:
            if p.label and p.label not in label_first:
                label_first[p.label] = p
        self.label_first = label_first
        k = 0
        self.sl = {}
        for key, size in (("std", len(self.rows)), ("dir", len(self.pairs)),
                          ("ind", len(self.pairs)), ("tot", len(self.pairs)),
                          ("dir_std", len(self.pairs)), ("ind_std", len(self.pairs)),
                          ("tot_std", len(self.pairs)), ("path", len(self.paths)),
                          ("path_std", len(self.paths)), ("r2", len(self.r2_vars)),
                          ("def", len(self.defined)), ("def_std", len(self.defined))):
            self.sl[key] = slice(k, k + size)
            k += size
        self.size = k

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        theta = np.atleast_2d(theta)
        spec = self.spec
        A, S = spec.matrices(theta)
        B = np.linalg.inv(np.eye(spec.m)[None] - A)
        Om = B @ S @ np.swapaxes(B, 1, 2)
        var = np.einsum("bii->bi", Om)
        sd = np.sqrt(np.clip(var, 1e-300, None))
        nb = len(theta)
        out = np.full((nb, self.size), np.nan)
        std = out[:, self.sl["std"]]
        for k, p in enumerate(self.rows):
            if p.op == "=~":
                i, j = spec.pos[p.rhs], spec.pos[p.lhs]
                std[:, k] = A[:, i, j] * sd[:, j] / sd[:, i]
            elif p.op == "~":
                i, j = spec.pos[p.lhs], spec.pos[p.rhs]
                std[:, k] = A[:, i, j] * sd[:, j] / sd[:, i]
            elif p.lhs == p.rhs:
                i = spec.pos[p.lhs]
                std[:, k] = S[:, i, i] / var[:, i]
            else:
                i, j = spec.pos[p.lhs], spec.pos[p.rhs]
                den = S[:, i, i] * S[:, j, j]
                with np.errstate(invalid="ignore"):
                    std[:, k] = np.where(den > 0, S[:, i, j] / np.sqrt(np.abs(den)), np.nan)
        if self.pairs:
            yi = np.array([a for a, _ in self.pair_idx])
            xi = np.array([b for _, b in self.pair_idx])
            T = B - np.eye(spec.m)[None]
            d = A[:, yi, xi]
            t = T[:, yi, xi]
            t = np.where(self.no_indirect[None], d, t)
            ratio = sd[:, xi] / sd[:, yi]
            out[:, self.sl["dir"]] = d
            out[:, self.sl["tot"]] = t
            out[:, self.sl["ind"]] = t - d
            out[:, self.sl["dir_std"]] = d * ratio
            out[:, self.sl["tot_std"]] = t * ratio
            out[:, self.sl["ind_std"]] = (t - d) * ratio
        for k, path in enumerate(self.paths):
            v = np.ones(nb)
            for a, b in zip(path[:-1], path[1:]):
                v = v * A[:, spec.pos[b], spec.pos[a]]
            out[:, self.sl["path"].start + k] = v
            out[:, self.sl["path_std"].start + k] = v * sd[:, spec.pos[path[0]]] / \
                sd[:, spec.pos[path[-1]]]
        for k, v in enumerate(self.r2_vars):
            i = spec.pos[v]
            out[:, self.sl["r2"].start + k] = 1 - S[:, i, i] / var[:, i]
        if self.defined:
            env, env_std = {}, {}
            for lab, p in self.label_first.items():
                env[lab] = theta[:, p.index] if p.free else np.full(nb, p.value)
                env_std[lab] = std[:, self.rows.index(p)]
            for k, d in enumerate(self.defined):
                with np.errstate(all="ignore"):
                    val = d.evaluate(env)
                    vstd = d.evaluate(env_std)
                env[d.name], env_std[d.name] = val, vstd
                out[:, self.sl["def"].start + k] = val
                out[:, self.sl["def_std"].start + k] = vstd
        return out


def _indirect_exists(edges: np.ndarray, y: int, x: int) -> bool:
    """Is there a directed path x -> ... -> y of length >= 2?"""
    n = len(edges)
    frontier = {m for m in range(n) if edges[m, x]}
    seen = set()
    while frontier:
        m = frontier.pop()
        if m in seen:
            continue
        seen.add(m)
        if edges[y, m] and m != y:
            return True
        frontier |= {k for k in range(n) if edges[k, m]}
    return False


def _simple_paths(struct: list[str], edges: np.ndarray, limit: int = 60) -> list[tuple]:
    """Directed paths with at least one intermediate node (specific indirect effects)."""
    n = len(struct)
    out: list[tuple] = []

    def dfs(path):
        if len(out) >= limit:
            return
        last = path[-1]
        for k in range(n):
            if edges[k, last] and k not in path:
                new = path + [k]
                if len(new) >= 3:
                    out.append(tuple(struct[i] for i in new))
                dfs(new)

    for s in range(n):
        dfs([s])
    return out


# ------------------------------------------------------------------ formatting helpers
def _p_amos(p: float) -> str:
    if p is None or not math.isfinite(p):
        return ""
    return "***" if p < 0.001 else fmt(p, 3)


def _f(x, d=3) -> str:
    return fmt(x, d) if x is not None and np.isfinite(x) else ""


def _crit(key: str, v: float) -> tuple[str, str]:
    if v is None or not np.isfinite(v):
        return "", ""

    def grade(good, ok=None, higher=True):
        if higher:
            return "Good fit" if v >= good else ("Acceptable fit" if ok is not None and v >= ok
                                                  else "Poor fit")
        return "Good fit" if v <= good else ("Acceptable fit" if ok is not None and v <= ok
                                              else "Poor fit")

    if key == "p":
        return "p > .05", ("Good fit (χ² not significant)" if v > 0.05
                           else "Significant (χ² is sensitive to N)")
    if key == "cmin_df":
        return "≤ 3.00 (≤ 5.00 acceptable)", grade(3, 5, higher=False)
    if key in ("gfi", "agfi"):
        return "≥ .90 (≥ .80 marginal)", ("Good fit" if v >= 0.9 else
                                            "Marginal fit" if v >= 0.8 else "Poor fit")
    if key in ("nfi", "ifi", "rfi"):
        return "≥ .95 good, ≥ .90 acceptable", grade(0.95, 0.90)
    if key in ("tli", "cfi"):
        return "≥ .95 good, ≥ .90 acceptable", grade(0.95, 0.90)
    if key == "rmsea":
        txt = ("Good fit" if v <= 0.06 else "Acceptable fit" if v <= 0.08 else
               "Mediocre fit" if v <= 0.10 else "Poor fit")
        return "≤ .06 good, ≤ .08 acceptable, > .10 poor", txt
    if key == "pclose":
        return "> .05", "Close fit" if v > 0.05 else "Close fit rejected"
    if key == "srmr":
        return "≤ .08", grade(0.08, 0.10, higher=False)
    if key in ("aic", "bic"):
        return "Smaller is better", "Model comparison"
    return "", ""


# ------------------------------------------------------------------ analysis
def cb_sem(ds: Dataset, syntax: str, estimator: str = "ML", identification: str = "marker",
           likelihood: str = "normal", information: str = "expected", robust: bool = False,
           boot: int = 0, seed: int | None = None, level: float = 0.95,
           outputs: dict | None = None, mi_min: float = 3.84, mi_top: int = 20,
           save_scores: bool = False, standardized_diagram: bool = True,
           fixed_x_baseline: bool = True, auto_cov_y: bool = True,
           cov_exogenous: bool = True) -> SEMResult:
    out = dict(OUTPUTS)
    out.update(outputs or {})
    estimator = estimator.upper()
    if estimator not in sf.ESTIMATORS:
        raise ValueError(f"Unknown estimator {estimator}")
    section = Section("Structural Equation Modeling (CB-SEM)")
    flat = "; ".join(ln.strip() for ln in syntax.splitlines()
                     if ln.strip() and not ln.strip().startswith("#"))
    section.log(f"SEM /MODEL '{flat}'\n  /ESTIMATOR={estimator}"
                f"{' ROBUST(MLM)' if robust and estimator == 'ML' else ''}"
                f" /IDENTIFICATION={'STD.LV' if identification == 'std.lv' else 'MARKER'}"
                f" /LIKELIHOOD={likelihood.upper()} /INFORMATION={information.upper()}"
                + (f"\n  /BOOTSTRAP={boot} SEED={seed if seed is not None else 'RANDOM'} "
                   f"CI={level * 100:g}" if boot else "") + " /MISSING=LISTWISE.")
    numeric = {v.name for v in ds.variables if v.type == "numeric"}
    try:
        model = build_model(syntax, ds.names(), std_lv=identification == "std.lv",
                            auto_cov_y=auto_cov_y, cov_exogenous=cov_exogenous,
                            numeric=numeric)
    except SEMSyntaxError as exc:
        section.note("Model syntax error", str(exc), style="warning")
        return SEMResult(section)
    frame = listwise(ds, model.observed)
    X = frame.to_numpy(float)
    n_total = ds.n_cases
    section.note("Model Specification", model.syntax.strip())
    if robust and estimator != "ML":
        section.note("Note", "Robust (MLM) standard errors are only available with ML; "
                             "ignored.", style="warning")
        robust = False
    try:
        est = sf.estimate(model, X, estimator, likelihood, information, robust)
    except sf.SEMError as exc:
        section.note("Model could not be estimated", str(exc), style="warning")
        return SEMResult(section)
    except np.linalg.LinAlgError as exc:
        section.note("Model could not be estimated",
                     f"Numerical problem ({exc}). Check for redundant variables or an "
                     f"unidentified model.", style="warning")
        return SEMResult(section)
    res = SEMResult(section, est=est)
    derived = Derived(est)
    dvals = derived(est.theta)[0]
    se_theta = np.sqrt(np.clip(np.diag(est.cov), 0, None)) if est.cov is not None else \
        np.full(est.spec.q, np.nan)
    dse = sf.delta_cov(derived, est.theta, est.cov) if est.cov is not None else \
        np.full(derived.size, np.nan)
    res.extra.update(derived=derived, dvals=dvals, dse=dse, se=se_theta)
    fit = sf.fit_measures(est, fixed_x_baseline)
    res.fit = fit

    _summary_tables(section, model, est, n_total, frame, robust)
    _warnings(section, model, est, derived, dvals)
    if out["normality"]:
        _normality(section, model, frame)
    if out["sample_cov"]:
        section.table(_matrix_table("Sample Covariances", model.observed, est.S))
    if out["fit"]:
        _fit_table(section, est, fit)

    # bootstrap
    boot_info = None
    if boot and boot > 0:
        thetas, conv = sf.bootstrap(est, X, int(boot), seed)
        ok = conv & np.all(np.isfinite(thetas), axis=1)
        tb = thetas[ok]
        if len(tb) >= 10:
            db = np.concatenate([derived(tb[i:i + 500]) for i in range(0, len(tb), 500)])
            boot_info = {"theta": tb, "derived": db, "n_ok": int(ok.sum()), "n": int(boot)}
            res.extra["boot"] = boot_info
        section.note("Bootstrap", f"{int(ok.sum())} of {int(boot)} bootstrap samples converged "
                                  f"and were used (nonparametric resampling of cases"
                                  f"{', seed ' + str(seed) if seed is not None else ''})."
                     + ("" if len(tb) >= 10 else " Too few for bootstrap intervals."),
                     style="note" if ok.mean() > 0.9 else "warning")

    if out["estimates"]:
        _estimate_tables(section, model, est, derived, dvals, se_theta, dse, boot_info, level)
    if out["r2"] and derived.r2_vars:
        rows = [[v, _f(dvals[derived.sl["r2"].start + k])]
                for k, v in enumerate(derived.r2_vars)]
        section.table(Table("Squared Multiple Correlations (R²)", ["", "Estimate"], rows))
    if out["implied_cov"]:
        section.table(_matrix_table("Implied Covariances", model.observed, est.Sigma))
    if out["effects"] and derived.pairs:
        _effect_tables(section, derived, dvals, dse, boot_info, level)
    if model.latent and out["measurement"]:
        res.extra["quality"] = _measurement(section, model, est, derived, dvals, frame)
    if len(model.latent) >= 2 and out["discriminant"]:
        res.extra["discriminant"] = _discriminant(section, model, est, frame,
                                                  res.extra.get("quality"))
    if out["mi"]:
        mis = sf.modification_indices(est)
        res.extra["mi"] = mis
        rows = [[f"{m.lhs} {m.op} {m.rhs}", _f(m.mi), _f(m.epc), _f(m.sepc)]
                for m in mis if m.mi >= mi_min][:mi_top]
        if rows:
            section.table(Table("Modification Indices", ["", "M.I.", "Par Change (EPC)",
                                                         "Std. EPC"], rows,
                                footnotes=[f"Threshold M.I. ≥ {mi_min:g}; top {mi_top} shown. "
                                           "=~ cross-loading, ~~ (residual) covariance, "
                                           "~ regression. Free a parameter only when it is "
                                           "theoretically justified."]))
        else:
            section.note("Modification Indices", f"No modification index exceeds {mi_min:g}.")
    if out["residuals"]:
        _residual_table(section, model, est)
    if out["diagram"]:
        try:
            from rsstatchem.core.stats.sem_diagram import path_diagram
            values = {}
            for k, p in enumerate(derived.rows):
                if standardized_diagram:
                    values[p.name] = dvals[derived.sl["std"].start + k]
                else:
                    values[p.name] = est.theta[p.index] if p.free else p.value
            r2 = {v: dvals[derived.sl["r2"].start + k] for k, v in enumerate(derived.r2_vars)}
            png = path_diagram(model, values, r2, standardized_diagram)
            section.image("Path Diagram", png, width=640)
        except Exception as exc:  # noqa: BLE001 - a drawing problem must not lose results
            section.note("Path Diagram", f"The diagram could not be drawn: {exc}",
                         style="warning")
    if save_scores and model.latent:
        res.saved = _factor_scores(model, est, frame, ds)
    return res


# ------------------------------------------------------------------ tables
def _summary_tables(section, model: SEMModel, est: sf.Estimates, n_total, frame, robust):
    lat_endo = [v for v in model.latent if v in model.endogenous]
    obs_endo = [v for v in model.observed if v in model.endogenous]
    rows = [
        ["Estimator", sf.ESTIMATORS[est.estimator] + f" ({est.estimator}"
         + (", robust MLM" if robust else "") + ")"],
        ["Optimization method", "Fisher scoring (step halving)"],
        ["Model identification", "Standardized latent variances (std.lv)" if model.std_lv
         else "Marker variable (first loading = 1)"],
        ["Number of observations used", fmt_int(est.n)],
        ["Number of cases in data", fmt_int(n_total)],
        ["Missing data", "Listwise deletion"],
        ["Number of distinct sample moments", fmt_int(est.spec.p * (est.spec.p + 1) // 2)],
        ["Number of distinct parameters to be estimated", fmt_int(est.spec.q)],
        ["Degrees of freedom", fmt_int(est.df)],
        ["Iterations", fmt_int(est.iterations)],
        ["Minimum was achieved", "Yes" if est.converged else "No"],
        ["Minimum of discrepancy function (F)", fmt(est.F, 5)],
        ["Standard errors", {"expected": "Expected information",
                             "observed": "Observed information",
                             "robust.sem": "Robust (sandwich, MLM)"}.get(est.se_type,
                                                                        est.se_type.capitalize())],
        ["Chi-square multiplier", "N (normal likelihood, lavaan)" if est.n_mult == est.n
         else "N − 1 (Wishart likelihood, AMOS)"],
    ]
    section.table(Table("Model Summary", ["", "Value"], rows))
    vrows = [["Observed, endogenous", fmt_int(len(obs_endo)), ", ".join(obs_endo)],
             ["Observed, exogenous", fmt_int(len(model.observed) - len(obs_endo)),
              ", ".join(v for v in model.observed if v not in obs_endo)],
             ["Unobserved (latent), endogenous", fmt_int(len(lat_endo)), ", ".join(lat_endo)],
             ["Unobserved (latent), exogenous", fmt_int(len(model.latent) - len(lat_endo)),
              ", ".join(v for v in model.latent if v not in lat_endo)]]
    section.table(Table("Variable Summary", ["", "Number", "Variables"], vrows))


def _warnings(section, model, est: sf.Estimates, derived: Derived, dvals):
    msgs = list(model.warnings) + list(est.warnings)
    if not est.converged:
        msgs.insert(0, f"The estimation did not converge (maximum |gradient| = "
                       f"{est.max_grad:.2g} after {est.iterations} iterations). The results "
                       f"are not trustworthy: check the model specification, start values and "
                       f"identification.")
    neg = [p.name for p in model.params if p.op == "~~" and p.lhs == p.rhs and p.free
           and est.theta[p.index] < 0]
    if neg:
        msgs.append("Heywood case: negative variance estimate(s) for " + ", ".join(neg)
                    + ". The solution is improper (inadmissible).")
    big = [p.name for k, p in enumerate(derived.rows) if p.op in ("=~", "~")
           and abs(dvals[derived.sl["std"].start + k]) > 1.0 and p.op == "=~"]
    if big:
        msgs.append("Standardized loading(s) greater than 1: " + ", ".join(big) + ".")
    if model.latent:
        idx = [est.spec.pos[f] for f in model.latent]
        w = np.linalg.eigvalsh(est.Omega[np.ix_(idx, idx)])
        if w.min() <= 0:
            msgs.append("The covariance matrix of the latent variables is not positive "
                        "definite.")
    if est.n < 100:
        msgs.append(f"Small sample (N = {est.n}). ML chi-square tests and fit indices may be "
                    f"unreliable; N ≥ 200 (or 5–10 cases per parameter) is commonly "
                    f"recommended. Consider bootstrap confidence intervals.")
    for m in msgs:
        section.note("Warning", m, style="warning")


def _normality(section, model: SEMModel, frame: pd.DataFrame):
    X = frame.to_numpy(float)
    n, p = X.shape
    Xc = X - X.mean(axis=0)
    m2 = (Xc ** 2).mean(axis=0)
    skew = (Xc ** 3).mean(axis=0) / m2 ** 1.5
    kurt = (Xc ** 4).mean(axis=0) / m2 ** 2 - 3
    rows = []
    for k, v in enumerate(frame.columns):
        rows.append([v, _f(X[:, k].min()), _f(X[:, k].max()), _f(skew[k]),
                     _f(skew[k] / math.sqrt(6 / n)), _f(kurt[k]),
                     _f(kurt[k] / math.sqrt(24 / n))])
    S = Xc.T @ Xc / n
    try:
        d2 = np.einsum("ij,jk,ik->i", Xc, np.linalg.inv(S), Xc)
        b2 = float(np.mean(d2 ** 2))
        mk = b2 - p * (p + 2)
        rows.append(["Multivariate", "", "", "", "", _f(mk), _f(mk / math.sqrt(8 * p * (p + 2)
                                                                             / n))])
    except np.linalg.LinAlgError:
        d2 = None
    section.table(Table("Assessment of Normality", ["Variable", "min", "max", "skew", "c.r.",
                                                    "kurtosis", "c.r."], rows,
                        footnotes=["Skew and excess kurtosis as in AMOS; |c.r.| > 2.58 "
                                   "suggests non-normality at p < .01. Multivariate = Mardia's "
                                   "kurtosis (c.r. > 1.96 rejects multivariate normality); "
                                   "consider bootstrap or robust (MLM) standard errors."]))
    if d2 is not None:
        order = np.argsort(-d2)[:10]
        out = []
        for rank, i in enumerate(order, 1):
            p1 = float(stats.chi2.sf(d2[i], p))
            p2 = float(stats.binom.sf(rank - 1, n, p1))
            out.append([fmt_int(frame.index[i] + 1), _f(d2[i]), _f(p1), _f(p2)])
        section.table(Table("Observations Farthest from the Centroid (Mahalanobis Distance)",
                            ["Observation number", "Mahalanobis d-squared", "p1", "p2"], out,
                            footnotes=["p1: probability that a case's d² exceeds this value; "
                                       "p2: probability that the largest (2nd, 3rd …) d² "
                                       "exceeds it under normality. Small p1 and p2 suggest "
                                       "outliers."]))


def _fit_table(section, est: sf.Estimates, fit: dict):
    uls = est.estimator == "ULS"
    rows = []

    def add(name, key, value=None, text=None):
        v = fit.get(key) if value is None else value
        crit, interp = _crit(key, v)
        rows.append([name, text if text is not None else _f(v), crit, interp])

    if uls:
        rows.append(["Chi-square (χ²)", "n/a", "", "ULS has no chi-square test"])
        rows.append(["Degrees of freedom (df)", fmt_int(fit["df"]), "", ""])
    else:
        rows.append(["Chi-square (χ²)", _f(fit["chi2"]), "Small relative to df", ""])
        rows.append(["Degrees of freedom (df)", fmt_int(fit["df"]), "", ""])
        add("Probability level (p)", "p")
        add("CMIN/DF (χ²/df)", "cmin_df")
    add("GFI", "gfi")
    add("AGFI", "agfi")
    add("NFI", "nfi")
    add("RFI", "rfi")
    if not uls:
        add("IFI", "ifi")
        add("TLI (NNFI)", "tli")
        add("CFI", "cfi")
        add("RMSEA", "rmsea")
        lo, hi = fit["rmsea_lo"], fit["rmsea_hi"]
        rows.append(["RMSEA 90% CI", f"[{_f(lo)}, {_f(hi)}]" if np.isfinite(lo) else "", "", ""])
        add("PCLOSE", "pclose")
    add("SRMR", "srmr")
    rows.append(["RMR", _f(fit["rmr"]), "Smaller is better", ""])
    if "aic" in fit:
        add("AIC", "aic")
        add("BIC", "bic")
    if "chi2_scaled" in fit:
        rows.append(["Satorra-Bentler scaled χ²", _f(fit["chi2_scaled"]), "", "Robust (MLM)"])
        rows.append(["Scaling correction factor", _f(fit["scale"]), "", ""])
        add("Probability level (scaled)", "p", fit["p_scaled"])
        add("CFI (scaled)", "cfi", fit.get("cfi_scaled"))
        add("TLI (scaled)", "tli", fit.get("tli_scaled"))
        add("RMSEA (scaled)", "rmsea", fit.get("rmsea_scaled"))
    if not uls:
        base = f"{_f(fit['chi2_base'])} ({fmt_int(fit['df_base'])})"
        rows.append(["Baseline model χ² (df)", base,
                     "", "Independence model"])
    notes = ["Cut-offs: Hu & Bentler (1999) for CFI/TLI ≥ .95, RMSEA ≤ .06, SRMR ≤ .08; "
             "Browne & Cudeck (1993) RMSEA ≤ .08 acceptable; GFI/AGFI/NFI/IFI ≥ .90 and "
             "CMIN/DF ≤ 3 are conventional (e.g. Hair et al.). Judge fit from several "
             "indices together."]
    if "aic" in fit:
        notes.append(f"Log-likelihood (H0) = {fit['logl']:.3f}, unrestricted (H1) = "
                     f"{fit['logl_h1']:.3f}. AIC = −2LL + 2q and BIC = −2LL + q·ln N as in "
                     f"lavaan; AMOS reports AIC = χ² + 2q = {fit['aic_amos']:.3f}.")
    if fit["df"] == 0:
        notes.insert(0, "Saturated (just-identified) model, df = 0: the model reproduces the "
                        "sample covariances exactly, so global fit cannot be tested.")
    if uls:
        notes.append("ULS: chi-square-based indices are not reported (as AMOS); NFI/RFI use "
                     "the ULS discrepancy.")
    section.table(Table("Model Fit", ["Fit index", "Value", "Criterion", "Interpretation"],
                        rows, footnotes=notes))


def _est_row(name, p, est, se, std, extra=()):
    if p.free:
        val = est.theta[p.index]
        s = se[p.index]
        cr = val / s if s > 0 else float("nan")
        pv = 2 * stats.norm.sf(abs(cr)) if np.isfinite(cr) else float("nan")
        return [name, _f(val), _f(s), _f(cr, 3), _p_amos(pv), _f(std), p.label or "", *extra]
    return [name, _f(p.value), "", "", "", _f(std),
            f"{p.label} (fixed)" if p.label else "fixed", *extra]


def _estimate_tables(section, model: SEMModel, est, derived: Derived, dvals, se, dse,
                     boot_info, level):
    cols = ["", "Estimate", "S.E.", "C.R.", "P", "Standardized", "Label"]
    stdv = {id(p): dvals[derived.sl["std"].start + k] for k, p in enumerate(derived.rows)}
    foot = ["*** p < .001. C.R. = critical ratio (Estimate / S.E.), two-tailed P from the "
            "standard normal. Standardized = completely standardized solution (std.all)."]
    groups = [("Factor Loadings (Measurement Model)", lambda p: p.op == "=~",
               lambda p: f"{p.rhs} <--- {p.lhs}"),
              ("Regression Weights (Structural Model)", lambda p: p.op == "~",
               lambda p: f"{p.lhs} <--- {p.rhs}"),
              ("Covariances", lambda p: p.op == "~~" and p.lhs != p.rhs,
               lambda p: f"{p.lhs} <--> {p.rhs}")]
    for title, keep, name in groups:
        rows = [_est_row(name(p), p, est, se, stdv[id(p)]) for p in derived.rows if keep(p)]
        if rows:
            f = list(foot)
            if title == "Covariances":
                f.append("Standardized covariances are correlations (between residuals for "
                         "endogenous variables).")
            section.table(Table(title, cols, rows, footnotes=f))
    endo = set(model.endogenous)
    rows = []
    for p in derived.rows:
        if p.op == "~~" and p.lhs == p.rhs:
            kind = "Residual" if p.lhs in endo else "Variance"
            row = _est_row(p.lhs, p, est, se, stdv[id(p)])
            rows.append(row[:1] + [kind] + row[1:])
    section.table(Table("Variances", ["", "Type", "Estimate", "S.E.", "C.R.", "P",
                                      "Standardized", "Label"], rows,
                        footnotes=["Residual = error (disturbance) variance of an endogenous "
                                   "variable; its standardized value is 1 − R²."]))
    if model.defined:
        rows = []
        for k, d in enumerate(model.defined):
            i = derived.sl["def"].start + k
            v, s = dvals[i], dse[i]
            z = v / s if s > 0 else float("nan")
            pv = 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else float("nan")
            rows.append([f"{d.name} := {d.expr}", _f(v), _f(s), _f(z), _p_amos(pv),
                         _f(dvals[derived.sl["def_std"].start + k])])
        section.table(Table("Defined Parameters", ["", "Estimate", "S.E.", "z", "P",
                                                   "Standardized"], rows,
                            footnotes=["S.E. by the delta method. For indirect effects prefer "
                                       "the bootstrap intervals (Preacher & Hayes, 2008)."]))
    if boot_info is not None:
        _boot_table(section, model, est, derived, dvals, boot_info, level)


def _boot_table(section, model, est, derived: Derived, dvals, boot_info, level):
    tb, db = boot_info["theta"], boot_info["derived"]
    names, ests, draws = [], [], []
    seen = set()
    for p in derived.rows:
        if not p.free or p.index in seen or (p.op == "~~" and p.lhs == p.rhs):
            continue
        seen.add(p.index)
        if p.op == "=~":
            nm = f"{p.rhs} <--- {p.lhs}"
        elif p.op == "~":
            nm = f"{p.lhs} <--- {p.rhs}"
        else:
            nm = f"{p.lhs} <--> {p.rhs}"
        names.append(nm)
        ests.append(est.theta[p.index])
        draws.append(tb[:, p.index])
    for k, d in enumerate(model.defined):
        i = derived.sl["def"].start + k
        names.append(f"{d.name} := {d.expr}")
        ests.append(dvals[i])
        draws.append(db[:, i])
    E = np.array(ests)
    D = np.column_stack(draws)
    _ci_table(section, f"Bootstrap Confidence Intervals ({pct(level)}, unstandardized)", names,
              E, D, level, boot_info)


def _ci_table(section, title, names, E, D, level, boot_info, extra_foot=None):
    lo_p, hi_p = sf.percentile_ci(D, level)
    lo_b, hi_b = sf.bc_ci(D, E, level)
    pv = sf.bc_p(D, E)
    sd = np.nanstd(D, axis=0, ddof=1)
    rows = [[n, _f(e), _f(s), _f(a), _f(b), _f(c), _f(d), _f(p)]
            for n, e, s, a, b, c, d, p in zip(names, E, sd, lo_p, hi_p, lo_b, hi_b, pv)]
    cols = ["", "Estimate", "Boot S.E.", ("Percentile", "Lower"), ("Percentile", "Upper"),
            ("Bias-corrected", "Lower"), ("Bias-corrected", "Upper"), "P"]
    foot = [f"{boot_info['n_ok']} bootstrap samples. P = two-tailed bootstrap probability "
            f"(share of resamples on the other side of zero). An interval that excludes 0 "
            f"indicates a significant effect."]
    if extra_foot:
        foot.append(extra_foot)
    section.table(Table(title, cols, rows, footnotes=foot))


def _effect_tables(section, derived: Derived, dvals, dse, boot_info, level):
    sl = derived.sl
    rows = []
    for k, (y, x) in enumerate(derived.pairs):
        g = lambda key: dvals[sl[key].start + k]  # noqa: E731
        rows.append([f"{y} <--- {x}", _f(g("dir")), _f(g("ind")), _f(g("tot")),
                     _f(g("dir_std")), _f(g("ind_std")), _f(g("tot_std"))])
    section.table(Table("Total, Direct and Indirect Effects", [
        "", ("Unstandardized", "Direct"), ("Unstandardized", "Indirect"),
        ("Unstandardized", "Total"), ("Standardized", "Direct"), ("Standardized", "Indirect"),
        ("Standardized", "Total")], rows))
    ind_rows, names, E, cols_idx = [], [], [], []
    for k, (y, x) in enumerate(derived.pairs):
        if not derived.has_indirect[k]:
            continue
        i = sl["ind"].start + k
        v, s = dvals[i], dse[i]
        z = v / s if s > 0 else float("nan")
        pv = 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else float("nan")
        ind_rows.append([f"{y} <--- {x}", _f(v), _f(s), _f(z), _p_amos(pv),
                         _f(dvals[sl["ind_std"].start + k])])
        names.append(f"{y} <--- {x} (total indirect)")
        E.append(v)
        cols_idx.append(i)
    for k, path in enumerate(derived.paths):
        i = sl["path"].start + k
        v, s = dvals[i], dse[i]
        z = v / s if s > 0 else float("nan")
        pv = 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else float("nan")
        ind_rows.append([" → ".join(path), _f(v), _f(s), _f(z), _p_amos(pv),
                         _f(dvals[sl["path_std"].start + k])])
        names.append(" → ".join(path))
        E.append(v)
        cols_idx.append(i)
    if ind_rows:
        section.table(Table("Indirect Effects (Delta Method)", ["", "Estimate", "S.E.", "z",
                                                                "P", "Standardized"], ind_rows,
                            footnotes=["Rows with arrows are specific indirect effects "
                                       "(product of the path coefficients along the path). "
                                       "Sobel-type S.E. by the delta method."]))
        if boot_info is not None:
            _ci_table(section, f"Bootstrap Confidence Intervals for Indirect Effects "
                               f"({pct(level)})", names, np.array(E),
                      boot_info["derived"][:, cols_idx], level, boot_info)
            std_idx = [i - sl["ind"].start + sl["ind_std"].start if i < sl["path"].start
                       else i - sl["path"].start + sl["path_std"].start for i in cols_idx]
            _ci_table(section, f"Bootstrap Confidence Intervals for Standardized Indirect "
                               f"Effects ({pct(level)})", names, dvals[std_idx],
                      boot_info["derived"][:, std_idx], level, boot_info)


def _measurement(section, model: SEMModel, est, derived: Derived, dvals, frame) -> dict:
    sl = derived.sl["std"]
    quality = {}
    rows_l, rows_q = [], []
    for f in model.latent:
        items = model.indicators[f]
        lam, theta = [], []
        for x in items:
            k = next(i for i, p in enumerate(derived.rows) if p.op == "=~" and p.lhs == f
                     and p.rhs == x)
            lv = dvals[sl.start + k]
            vk = next((i for i, p in enumerate(derived.rows) if p.op == "~~" and p.lhs == x
                       and p.rhs == x), None)
            th = dvals[sl.start + vk] if vk is not None else 1 - lv ** 2
            lam.append(lv)
            theta.append(th)
            crit = ("Valid (≥ .70)" if abs(lv) >= 0.7 else "Acceptable (≥ .50)"
                    if abs(lv) >= 0.5 else "Low (< .50)")
            rows_l.append([f, x, _f(lv), _f(lv ** 2), crit])
        lam, theta = np.array(lam), np.array(theta)
        cr = lam.sum() ** 2 / (lam.sum() ** 2 + theta.sum())
        ave = float(np.mean(lam ** 2))
        obs_items = [x for x in items if x in frame.columns]
        alpha = cronbach_alpha(frame[obs_items].to_numpy(float)) \
            if len(obs_items) == len(items) and len(items) >= 2 else float("nan")
        quality[f] = {"loadings": lam, "cr": cr, "ave": ave, "alpha": alpha}
        verdict = []
        verdict.append("CR ✓" if cr >= 0.7 else "CR ✗")
        verdict.append("AVE ✓" if ave >= 0.5 else "AVE ✗")
        if np.isfinite(alpha):
            verdict.append("α ✓" if alpha >= 0.7 else "α ✗")
        rows_q.append([f, fmt_int(len(items)), _f(alpha), _f(cr), _f(ave),
                       _f(math.sqrt(ave)), ", ".join(verdict)])
    section.table(Table("Convergent Validity: Standardized Factor Loadings",
                        ["Construct", "Indicator", "Std. loading", "λ² (item R²)", "Criterion"],
                        rows_l, row_header_cols=2,
                        footnotes=["Loadings ≥ .70 are ideal; ≥ .50 acceptable when CR and AVE "
                                   "are adequate (Hair et al., 2019)."]))
    section.table(Table("Construct Reliability and Validity",
                        ["Construct", "Items", "Cronbach's α", "Composite Reliability (CR)",
                         "AVE", "√AVE", "Assessment"], rows_q,
                        footnotes=["CR = (Σλ)² / ((Σλ)² + Σθ) with standardized loadings λ and "
                                   "standardized residual variances θ; AVE = Σλ² / k "
                                   "(Fornell & Larcker, 1981). Criteria: CR ≥ .70, AVE ≥ .50, "
                                   "α ≥ .70. Residual covariances are ignored in CR."]))
    return quality


def _discriminant(section, model: SEMModel, est, frame, quality) -> dict:
    lat = model.latent
    idx = [est.spec.pos[f] for f in lat]
    Om = est.Omega[np.ix_(idx, idx)]
    d = np.sqrt(np.clip(np.diag(Om), 1e-300, None))
    R = Om / np.outer(d, d)
    k = len(lat)
    rows = []
    fails = []
    for i in range(k):
        row = [lat[i]]
        for j in range(k):
            if j > i:
                row.append("")
            elif i == j:
                ave = quality[lat[i]]["ave"] if quality else float("nan")
                row.append(_f(math.sqrt(ave)) if np.isfinite(ave) else "")
            else:
                row.append(_f(R[i, j]))
                if quality:
                    lim = min(math.sqrt(quality[lat[i]]["ave"]), math.sqrt(quality[lat[j]]["ave"]))
                    if abs(R[i, j]) >= lim:
                        fails.append(f"{lat[i]}–{lat[j]}")
        rows.append(row)
    foot = ["Diagonal: √AVE; below: latent variable correlations. Discriminant validity holds "
            "when each √AVE exceeds the construct's correlations (Fornell & Larcker, 1981)."]
    foot.append("Not met for: " + ", ".join(fails) + "." if fails else
                "The criterion is met for all pairs.")
    section.table(Table("Discriminant Validity: Fornell-Larcker Criterion", [""] + lat, rows,
                        footnotes=foot))
    C = np.corrcoef(frame.to_numpy(float), rowvar=False)
    col = {v: i for i, v in enumerate(frame.columns)}
    H = np.full((k, k), np.nan)
    items = {f: [x for x in model.indicators[f] if x in col] for f in lat}

    def mono(f):
        ii = [col[x] for x in items[f]]
        if len(ii) < 2:
            return float("nan")
        sub = np.abs(C[np.ix_(ii, ii)])
        return (sub.sum() - len(ii)) / (len(ii) * (len(ii) - 1))

    for i in range(k):
        for j in range(i):
            a, b = [col[x] for x in items[lat[i]]], [col[x] for x in items[lat[j]]]
            if not a or not b:
                continue
            het = np.abs(C[np.ix_(a, b)]).mean()
            den = math.sqrt(mono(lat[i]) * mono(lat[j]))
            H[i, j] = H[j, i] = het / den if den > 0 else float("nan")
    rows = []
    high = []
    for i in range(k):
        row = [lat[i]]
        for j in range(k):
            row.append(_f(H[i, j]) if j < i else "")
            if j < i and np.isfinite(H[i, j]) and H[i, j] >= 0.85:
                high.append(f"{lat[i]}–{lat[j]} ({fmt(H[i, j], 3)})")
        rows.append(row)
    foot = ["Heterotrait-monotrait ratio of correlations (Henseler, Ringle & Sarstedt, 2015) "
            "from absolute item correlations. Discriminant validity: HTMT < .85 (strict) or "
            "< .90 (liberal)."]
    foot.append("HTMT ≥ .85 for: " + ", ".join(high) + "." if high else
                "All HTMT values are below .85.")
    section.table(Table("Discriminant Validity: HTMT", [""] + lat, rows, footnotes=foot))
    return {"latent_corr": R, "htmt": H}


def _residual_table(section, model: SEMModel, est: sf.Estimates):
    S, Sig, N = est.S, est.Sigma, est.n_mult
    var = np.diag(Sig)
    se = np.sqrt((np.outer(var, var) + Sig ** 2) / N)
    Z = (S - Sig) / se
    names = model.observed
    rows = []
    for i, a in enumerate(names):
        rows.append([a] + [(_f(Z[i, j]) + ("*" if abs(Z[i, j]) > 2.58 else ""))
                           if j <= i else "" for j in range(len(names))])
    section.table(Table("Standardized Residual Covariances", [""] + names, rows,
                        footnotes=["(s − σ̂) / √((σ̂ᵢᵢσ̂ⱼⱼ + σ̂ᵢⱼ²)/N), the normal-theory "
                                   "approximation used by AMOS. * |value| > 2.58."]))


def _matrix_table(title, names, M):
    rows = [[a] + [_f(M[i, j]) if j <= i else "" for j in range(len(names))]
            for i, a in enumerate(names)]
    return Table(title, [""] + list(names), rows)


def _factor_scores(model: SEMModel, est: sf.Estimates, frame: pd.DataFrame, ds: Dataset):
    """Regression (Thurstone) factor scores: E[eta | x] = Omega_eta,x Sigma^-1 (x - mean)."""
    X = frame.to_numpy(float)
    Xc = X - X.mean(axis=0)
    p = est.spec.p
    lat_idx = [est.spec.pos[f] for f in model.latent]
    W = np.linalg.solve(est.Sigma, est.Omega[:p, lat_idx])
    scores = Xc @ W
    saved = {}
    for k, f in enumerate(model.latent):
        base = "FS_" + "".join(ch for ch in f if ch.isalnum() or ch == "_")[:20]
        s = pd.Series(np.nan, index=ds.df.index)
        s.loc[frame.index] = scores[:, k]
        saved[base] = (f"Factor score (regression) for {f}", s, 5)
    return saved

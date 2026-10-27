"""Analyze > Meta Analysis, mirroring SPSS Statistics 28+ (META ES / META BINARY / META REG).

* Continuous outcomes from raw data (n, mean, SD per group: Cohen's d, Hedges' g, Glass's
  delta, unstandardized mean difference) or pre-calculated effect sizes with their standard
  errors.
* Binary outcomes from raw 2x2 counts (log odds ratio, log risk ratio, risk difference, log
  Peto odds ratio; 0.5 correction of zero cells; Mantel-Haenszel and Peto fixed effects) or
  pre-calculated (log) effect sizes.
* Fixed effects (inverse variance) and random effects with the estimators SPSS lists: REML, ML,
  Empirical Bayes, Hunter-Schmidt, DerSimonian-Laird, Hedges and Sidik-Jonkman, optional Knapp-
  Hartung adjustment (truncated or not), prediction interval.
* Heterogeneity: Q, tau-squared, tau, H-squared, I-squared; subgroup analysis with the test of
  subgroup differences; cumulative and leave-one-out analysis.
* Publication bias: Egger's regression-based test (the SE as a moderator in the same model),
  Begg and Mazumdar's rank correlation and Duval and Tweedie's trim-and-fill (L0 / R0).
* Meta-regression (mixed-effects with covariates and dummy-coded factors): coefficients, QM,
  QE, residual tau-squared, I-squared and R-squared.
* Plots: forest, cumulative forest, funnel (with trim-and-fill imputed studies), Galbraith,
  L'Abbe (binary) and bubble plot (meta-regression).

Formulas follow Borenstein et al. (2009) for the effect sizes and ``metafor`` (Viechtbauer,
2010) for the tau-squared estimators, trim-and-fill and meta-regression; results reproduce
metafor's published BCG-vaccine and Viechtbauer (2005) examples.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, stats

from rsstatchem.core.dataset import NUMERIC, Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, fmt_p, numeric, value_text
from rsstatchem.core.stats.wp4_util import Result, figure, png, sort_key

ESTIMATORS = {"REML": "Restricted maximum likelihood", "ML": "Maximum likelihood",
              "EB": "Empirical Bayes", "HS": "Hunter-Schmidt", "DL": "DerSimonian-Laird",
              "HE": "Hedges", "SJ": "Sidik-Jonkman"}
CONT_MEASURES = {"cohen": "Cohen's d", "hedges": "Hedges' g", "glass": "Glass's delta",
                 "md": "Unstandardized Mean Difference"}
BIN_MEASURES = {"or": "Log Odds Ratio", "rr": "Log Risk Ratio", "rd": "Risk Difference",
                "peto": "Log Peto Odds Ratio"}
LOG_MEASURES = {"or", "rr", "peto", "log"}


def _f(x, d: int = 3) -> str:
    return fmt(x, d)


# =========================================================================== effect sizes
def continuous_effects(n1, m1, s1, n2, m2, s2, measure: str = "hedges"):
    """Per-study effect size and sampling variance (treatment minus control)."""
    n1, m1, s1, n2, m2, s2 = (np.asarray(a, float) for a in (n1, m1, s1, n2, m2, s2))
    diff = m1 - m2
    if measure == "md":
        return diff, s1 ** 2 / n1 + s2 ** 2 / n2
    sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    base = (n1 + n2) / (n1 * n2)
    if measure == "glass":
        delta = diff / s2
        return delta, base + delta ** 2 / (2 * (n2 - 1))
    d = diff / sp
    vd = base + d ** 2 / (2 * (n1 + n2))
    if measure == "cohen":
        return d, vd
    j = 1 - 3 / (4 * (n1 + n2 - 2) - 1)
    return j * d, j ** 2 * vd


def binary_effects(a, b, c, d, measure: str = "or", zero: str = "only", add: float = 0.5):
    """a/b = treatment events/non-events, c/d = control events/non-events.

    ``zero``: "only" adds ``add`` to every cell of studies with a zero cell, "all" to every
    study, "none" never (Peto never needs it)."""
    a, b, c, d = (np.asarray(x, float).copy() for x in (a, b, c, d))
    if measure == "peto":
        n = a + b + c + d
        n1, m1 = a + b, a + c
        e = n1 * m1 / n
        v = n1 * (c + d) * m1 * (b + d) / (n ** 2 * (n - 1))
        return (a - e) / v, 1 / v
    if zero != "none" and measure != "rd":
        has0 = (a == 0) | (b == 0) | (c == 0) | (d == 0)
        mask = np.ones_like(a, bool) if zero == "all" else has0
        for x in (a, b, c, d):
            x[mask] += add
    n1, n2 = a + b, c + d
    if measure == "or":
        return np.log(a * d / (b * c)), 1 / a + 1 / b + 1 / c + 1 / d
    if measure == "rr":
        return np.log((a / n1) / (c / n2)), 1 / a - 1 / n1 + 1 / c - 1 / n2
    p1, p2 = a / n1, c / n2
    return p1 - p2, p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2


def mantel_haenszel(a, b, c, d, measure: str = "or") -> tuple[float, float]:
    """Mantel-Haenszel pooled (log) estimate and its SE (Robins-Breslow-Greenland for the OR,
    Greenland-Robins for the RR and RD)."""
    a, b, c, d = (np.asarray(x, float) for x in (a, b, c, d))
    n = a + b + c + d
    n1, n2 = a + b, c + d
    if measure == "or":
        r, s = a * d / n, b * c / n
        R, S = r.sum(), s.sum()
        p, q = (a + d) / n, (b + c) / n
        var = ((p * r).sum() / (2 * R ** 2) + ((p * s + q * r).sum()) / (2 * R * S) +
               (q * s).sum() / (2 * S ** 2))
        return math.log(R / S), math.sqrt(var)
    if measure == "rr":
        R = (a * n2 / n).sum()
        S = (c * n1 / n).sum()
        P = ((n1 * n2 * (a + c) - a * c * n) / n ** 2).sum()
        return math.log(R / S), math.sqrt(P / (R * S))
    w = n1 * n2 / n
    est = ((a * n2 - c * n1) / n).sum() / w.sum()
    num = ((a * b * n2 ** 3 + c * d * n1 ** 3) / (n1 * n2 * n ** 2)).sum()
    return float(est), math.sqrt(num / w.sum() ** 2)


# =========================================================================== model fitting
@dataclass
class MetaFit:
    k: int
    beta: np.ndarray
    cov: np.ndarray
    tau2: float
    tau2_se: float
    model: str                 # fixed | random
    method: str
    q: float                   # QE (residual heterogeneity statistic)
    q_df: int
    i2: float
    h2: float
    weights: np.ndarray        # 1 / (v + tau2)
    knapp: str = "none"
    df_t: int | None = None    # degrees of freedom of t tests under Knapp-Hartung
    qm: float = float("nan")
    qm_df: int = 0
    names: list = field(default_factory=list)
    r2: float = float("nan")

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    def stat(self) -> np.ndarray:
        return self.beta / self.se

    def pvalues(self) -> np.ndarray:
        z = self.stat()
        if self.df_t:
            return 2 * stats.t.sf(np.abs(z), self.df_t)
        return 2 * stats.norm.sf(np.abs(z))

    def crit(self, level: float = 0.95) -> float:
        if self.df_t:
            return float(stats.t.ppf(0.5 + level / 2, self.df_t))
        return float(stats.norm.ppf(0.5 + level / 2))

    def ci(self, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        c = self.crit(level)
        return self.beta - c * self.se, self.beta + c * self.se

    def prediction(self, level: float = 0.95) -> tuple[float, float]:
        se = math.sqrt(self.tau2 + self.cov[0, 0])
        c = self.crit(level)
        return float(self.beta[0] - c * se), float(self.beta[0] + c * se)

    @property
    def weight_pct(self) -> np.ndarray:
        return 100 * self.weights / self.weights.sum()


def _wls(y, v, X, tau2):
    w = 1 / (v + tau2)
    xtw = X.T * w
    cov = np.linalg.inv(xtw @ X)
    beta = cov @ (xtw @ y)
    return beta, cov, w


def _p_matrix(v, X, tau2):
    w = 1 / (v + tau2)
    W = np.diag(w)
    xtw = X.T * w
    return W - xtw.T @ np.linalg.inv(xtw @ X) @ xtw


def estimate_tau2(y, v, X, method: str = "REML") -> float:
    k, p = X.shape
    if k - p < 1:
        return 0.0
    if method == "DL":
        w = 1 / v
        P = _p_matrix(v, X, 0.0)
        qe = float(y @ P @ y)
        xtw = X.T * w
        trace = w.sum() - np.trace(np.linalg.inv(xtw @ X) @ ((X.T * w ** 2) @ X))
        return max(0.0, (qe - (k - p)) / trace)
    if method == "HE":
        P0 = np.eye(k) - X @ np.linalg.inv(X.T @ X) @ X.T
        return max(0.0, (float(y @ P0 @ y) - float(np.trace(P0 @ np.diag(v)))) / (k - p))
    if method == "HS":
        P = _p_matrix(v, X, 0.0)
        return max(0.0, (float(y @ P @ y) - k) / (1 / v).sum())
    if method == "SJ":
        P0 = np.eye(k) - X @ np.linalg.inv(X.T @ X) @ X.T
        t0 = float(y @ P0 @ y) / k
        if t0 <= 0:
            return 0.0
        P = _p_matrix(v, X, t0)
        return t0 * float(y @ P @ y) / (k - p)
    if method == "EB":
        tau2 = max(0.0, estimate_tau2(y, v, X, "HE"))
        for _ in range(1000):
            P = _p_matrix(v, X, tau2)
            w = 1 / (v + tau2)
            adj = (float(y @ P @ y) * k / (k - p) - k) / w.sum()
            new = max(0.0, tau2 + adj)
            if abs(new - tau2) < 1e-10:
                tau2 = new
                break
            tau2 = new
        return tau2

    def negll(t2):
        w = 1 / (v + t2)
        beta, cov, _ = _wls(y, v, X, t2)
        r = y - X @ beta
        ll = -0.5 * (np.log(v + t2).sum() + (w * r ** 2).sum())
        if method == "REML":
            ll -= 0.5 * np.linalg.slogdet((X.T * w) @ X)[1]
        return -ll

    hi = max(10.0, 10 * float(np.var(y)) + 10 * float(v.max()))
    res = optimize.minimize_scalar(negll, bounds=(0.0, hi), method="bounded",
                                   options={"xatol": 1e-12, "maxiter": 2000})
    tau2 = float(res.x)
    if negll(0.0) <= negll(tau2):
        tau2 = 0.0
    # Polish with Fisher scoring (metafor's algorithm) for full precision.
    for _ in range(100):
        if tau2 <= 0:
            break
        P = _p_matrix(v, X, tau2)
        if method == "REML":
            adj = (float(y @ P @ P @ y) - np.trace(P)) / np.trace(P @ P)
        else:
            w = 1 / (v + tau2)
            adj = (float(y @ P @ P @ y) - w.sum()) / (w ** 2).sum()
        new = max(0.0, tau2 + adj)
        if abs(new - tau2) < 1e-12:
            tau2 = new
            break
        tau2 = new
    return tau2


def _tau2_se(v, X, tau2, method) -> float:
    if method not in ("REML", "ML", "EB") or tau2 <= 0:
        return float("nan")
    P = _p_matrix(v, X, tau2)
    if method == "ML":
        w = 1 / (v + tau2)
        return math.sqrt(2 / (w ** 2).sum())
    return math.sqrt(2 / np.trace(P @ P))


def fit(y, v, X=None, model: str = "random", method: str = "REML", knapp: str = "none",
        names=None) -> MetaFit:
    y = np.asarray(y, float)
    v = np.asarray(v, float)
    k = len(y)
    X = np.ones((k, 1)) if X is None else np.asarray(X, float)
    p = X.shape[1]
    beta0, _, w0 = _wls(y, v, X, 0.0)
    qe = float((w0 * (y - X @ beta0) ** 2).sum())
    tau2 = estimate_tau2(y, v, X, method) if model == "random" else 0.0
    tau2_se = _tau2_se(v, X, tau2, method) if model == "random" else float("nan")
    beta, cov, w = _wls(y, v, X, tau2)
    # I^2 and H^2 from the "typical" within-study variance (Higgins & Thompson; metafor).
    wi = 1 / v
    if k <= p:
        s2 = float("nan")
    elif p == 1:
        s2 = (k - 1) * wi.sum() / (wi.sum() ** 2 - (wi ** 2).sum())
    else:
        P = _p_matrix(v, X, 0.0)
        s2 = (k - p) / np.trace(P)
    if model == "random":
        i2 = 100 * tau2 / (tau2 + s2)
        h2 = (tau2 + s2) / s2
    else:
        i2 = max(0.0, 100 * (qe - (k - p)) / qe) if qe > 0 else 0.0
        h2 = qe / (k - p) if k > p else float("nan")
    df_t = None
    if knapp != "none" and k > p:
        r = y - X @ beta
        q = float((w * r ** 2).sum()) / (k - p)
        if knapp == "truncated":
            q = max(1.0, q)
        cov = cov * q
        df_t = k - p
    fitres = MetaFit(k, beta, cov, tau2, tau2_se, model, method, qe, k - p, i2, h2, w,
                     knapp, df_t, names=list(names or ["Intercept"] + [f"X{j}" for j in
                                                                       range(1, p)]))
    if p > 1:
        b, c = beta[1:], cov[1:, 1:]
        fitres.qm = float(b @ np.linalg.solve(c, b))
        fitres.qm_df = p - 1
        if df_t:
            fitres.qm /= (p - 1)
        if model == "random":
            t_re = estimate_tau2(y, v, np.ones((k, 1)), method)
            fitres.r2 = max(0.0, 100 * (t_re - tau2) / t_re) if t_re > 0 else 0.0
    return fitres


# =========================================================================== bias / sensitivity
def egger_test(y, v, model="random", method="REML", knapp="none"):
    """Egger's regression-based test: the SE as a moderator (returns coefficient, se, stat,
    p, df_t)."""
    se = np.sqrt(v)
    X = np.column_stack([np.ones(len(y)), se])
    f = fit(y, v, X, model, method, knapp)
    return float(f.beta[1]), float(f.se[1]), float(f.stat()[1]), float(f.pvalues()[1]), \
        f.df_t, f


def begg_test(y, v, model="fixed", method="REML"):
    """Begg and Mazumdar's rank correlation (Kendall's tau of standardized effects vs
    variances)."""
    f = fit(y, v, None, model, method)
    b, vb = f.beta[0], f.cov[0, 0]
    vstar = v - vb
    ystar = (y - b) / np.sqrt(vstar)
    tau, p = stats.kendalltau(ystar, v)
    return float(tau), float(p)


@dataclass
class TrimFill:
    k0: int
    side: str
    fit: MetaFit
    y_fill: np.ndarray
    v_fill: np.ndarray


def trim_and_fill(y, v, model="random", method="REML", estimator="L0", side: str | None = None,
                  maxiter: int = 100) -> TrimFill:
    y = np.asarray(y, float)
    v = np.asarray(v, float)
    k = len(y)
    if side is None:
        X = np.column_stack([np.ones(k), np.sqrt(v)])
        b = fit(y, v, X, model, method).beta[1]
        side = "right" if b < 0 else "left"
    sign = -1.0 if side == "right" else 1.0
    yy = sign * y
    order = np.argsort(yy, kind="stable")
    ys, vs = yy[order], v[order]
    k0, prev = 0, -1
    b = fit(ys, vs, None, model, method).beta[0]
    for _ in range(maxiter):
        b = fit(ys[:k - k0], vs[:k - k0], None, model, method).beta[0]
        c = ys - b
        ranks = stats.rankdata(np.abs(c), method="ordinal")
        signed = np.sign(c) * ranks
        if estimator == "R0":
            neg = -signed[signed < 0]
            k0 = int(max(0, round((k - (neg.max() if len(neg) else 0)) - 1)))
        else:
            sr = signed[signed > 0].sum()
            k0 = int(max(0, round((4 * sr - k * (k + 1)) / (2 * k - 1))))
        k0 = min(k0, k - 2)
        if k0 == prev:
            break
        prev = k0
    if k0 > 0:
        y_fill = 2 * b - ys[k - k0:]
        v_fill = vs[k - k0:]
        ya = np.concatenate([ys, y_fill])
        va = np.concatenate([vs, v_fill])
        f = fit(sign * ya, va, None, model, method)
        return TrimFill(k0, side, f, sign * y_fill, v_fill)
    return TrimFill(0, side, fit(y, v, None, model, method), np.array([]), np.array([]))


def cumulative(y, v, model="random", method="REML", knapp="none") -> list[MetaFit]:
    return [fit(y[:i], v[:i], None, model if i > 1 else "fixed", method,
                knapp if i > 1 else "none") for i in range(1, len(y) + 1)]


def leave_one_out(y, v, model="random", method="REML", knapp="none") -> list[MetaFit]:
    idx = np.arange(len(y))
    return [fit(y[idx != i], v[idx != i], None, model, method, knapp) for i in idx]


# =========================================================================== data handling
@dataclass
class StudyData:
    y: np.ndarray
    v: np.ndarray
    labels: list[str]
    rows: np.ndarray                    # dataset row index of each study
    measure: str                        # key of CONT_MEASURES / BIN_MEASURES or "es"/"log"
    measure_label: str
    counts: np.ndarray | None = None    # binary raw: columns a, b, c, d
    n_excluded: int = 0
    group: np.ndarray | None = None
    group_labels: list | None = None
    order: np.ndarray | None = None


def _labels(ds: Dataset, study, label, index) -> list[str]:
    if label:
        s = ds.analysis_series(label)
        return [str(value_text(ds.var(label), s.loc[i])) for i in index]
    if study:
        s = ds.analysis_series(study)
        return [str(value_text(ds.var(study), s.loc[i])) for i in index]
    return [f"Study {i + 1}" for i in index]


def prepare(ds: Dataset, mode: str, variables: dict, measure: str, zero: str = "only",
            se_kind: str = "se", study=None, label=None, group=None, order=None) -> StudyData:
    """mode: cont_raw | cont_es | bin_raw | bin_es. ``variables`` maps role -> variable."""
    label_name = label
    roles = {"cont_raw": ["n1", "m1", "s1", "n2", "m2", "s2"], "cont_es": ["es", "se"],
             "bin_raw": ["a", "b", "c", "d"], "bin_es": ["es", "se"]}[mode]
    cols = {r: numeric(ds, variables[r]) for r in roles}
    frame = pd.DataFrame(cols)
    if group:
        frame["_g"] = ds.analysis_series(group)
    if order:
        frame["_o"] = numeric(ds, order) if ds.var(order).type == NUMERIC else \
            ds.analysis_series(order)
    ok = frame.notna().all(axis=1)
    frame = frame[ok]
    counts = None
    if mode == "cont_raw":
        y, v = continuous_effects(*(frame[r] for r in roles), measure=measure)
        mlabel = CONT_MEASURES[measure]
    elif mode == "bin_raw":
        counts = frame[roles].to_numpy(float)
        y, v = binary_effects(*(counts[:, j] for j in range(4)), measure=measure, zero=zero)
        mlabel = BIN_MEASURES[measure]
    else:
        y = frame["es"].to_numpy(float)
        se = frame["se"].to_numpy(float)
        v = se if se_kind == "var" else se ** 2
        mlabel = {"es": "Effect Size", "log": "Log Effect Size", "or": "Log Odds Ratio",
                 "rr": "Log Risk Ratio", "rd": "Risk Difference",
                 "peto": "Log Peto Odds Ratio"}.get(measure, "Effect Size")
    y, v = np.asarray(y, float), np.asarray(v, float)
    good = np.isfinite(y) & np.isfinite(v) & (v > 0)
    idx = frame.index.to_numpy()[good]
    sd = StudyData(y[good], v[good], _labels(ds, study, label_name, idx), idx, measure,
                   mlabel, counts[good] if counts is not None else None,
                   int((~ok).sum() + (~good).sum()))
    if group:
        g = frame["_g"].to_numpy(dtype=object)[good]
        levels = sorted(pd.unique(g), key=sort_key)
        sd.group = g
        sd.group_labels = [(lv, str(value_text(ds.var(group), lv))) for lv in levels]
    if order:
        sd.order = frame["_o"].to_numpy(dtype=object)[good]
    return sd


# =========================================================================== output helpers
def _model_note(f: MetaFit) -> str:
    if f.model == "fixed":
        return "Fixed-effects model."
    text = f"Random-effects model; tau-squared estimator: {ESTIMATORS[f.method]}."
    if f.knapp != "none":
        text += (" Standard errors use the " + ("truncated " if f.knapp == "truncated" else "")
                 + "Knapp-Hartung adjustment.")
    return text


def _estimate_row(label, f: MetaFit, level, exp=False, pi=True):
    lo, hi = f.ci(level)
    b, se = float(f.beta[0]), float(f.se[0])
    row = [label]
    if exp:
        row += [_f(math.exp(b)), _f(math.exp(lo[0])), _f(math.exp(hi[0]))]
        if pi:
            plo, phi = f.prediction(level)
            row += [_f(math.exp(plo)), _f(math.exp(phi))] if f.model == "random" else ["", ""]
        return row
    row += [_f(b), _f(se), _f(float(f.stat()[0]), 3), fmt_p(float(f.pvalues()[0])),
            _f(lo[0]), _f(hi[0])]
    if pi:
        plo, phi = f.prediction(level)
        row += [_f(plo), _f(phi)] if f.model == "random" else ["", ""]
    return row


def _estimate_cols(f: MetaFit, level, exp=False, pi=True):
    ci = f"{level * 100:g}% Conf. Interval"
    pint = f"{level * 100:g}% Pred. Interval"
    if exp:
        cols = ["", "Exp(Effect Size)", (ci, "Lower"), (ci, "Upper")]
    else:
        cols = ["", "Effect Size", "Std. Error", "t" if f.df_t else "Z", "Sig.", (ci, "Lower"),
                (ci, "Upper")]
    if pi:
        cols += [(pint, "Lower"), (pint, "Upper")]
    return cols


def _hetero_rows(label, f: MetaFit):
    return [label, _f(f.tau2, 4), _f(math.sqrt(f.tau2), 4), _f(f.h2, 3), _f(f.i2, 2),
            _f(f.q, 3), fmt_int(f.q_df), fmt_p(float(stats.chi2.sf(f.q, f.q_df)))
            if f.q_df > 0 else ""]


HETERO_COLS = ["", "Tau-squared", "Tau", "H-squared", "I-squared (%)",
               ("Test of Homogeneity", "Q"), ("Test of Homogeneity", "df"),
               ("Test of Homogeneity", "Sig.")]


# =========================================================================== plots
def forest_png(labels, y, v, f: MetaFit | None, level=0.95, exp=False, title="Forest Plot",
               weights=None, summary_label="Overall", extra=None, sort=None) -> bytes:
    y = np.asarray(y, float)
    se = np.sqrt(np.asarray(v, float))
    z = stats.norm.ppf(0.5 + level / 2)
    lo, hi = y - z * se, y + z * se
    idx = np.arange(len(y))
    if sort == "asc":
        idx = np.argsort(y)
    elif sort == "desc":
        idx = np.argsort(-y)
    k = len(y)
    rows = k + (2 if f is not None else 0) + (len(extra) if extra else 0)
    fig = figure(7.4, max(2.6, 0.32 * rows + 1.4))
    ax = fig.add_subplot(111)
    tr = np.exp if exp else (lambda a: a)
    wts = np.asarray(weights if weights is not None else 1 / np.asarray(v), float)
    size = 20 + 180 * wts / wts.max()
    for pos, i in enumerate(idx):
        yy = rows - pos
        ax.plot([tr(lo[i]), tr(hi[i])], [yy, yy], color="#333333", lw=1)
        ax.scatter([tr(y[i])], [yy], marker="s", s=size[i], color="#1f4e9c", zorder=3)
    ticks = [rows - p for p in range(k)]
    names = [labels[i] for i in idx]
    pos = rows - k - 1
    diamonds = []
    if extra:
        for name, ef in extra:
            diamonds.append((name, ef, pos))
            pos -= 1
    if f is not None:
        diamonds.append((summary_label, f, pos))
    for name, ef, yy in diamonds:
        b = float(ef.beta[0])
        l, h = ef.ci(level)
        xs = [tr(l[0]), tr(b), tr(h[0]), tr(b)]
        ax.fill(xs, [yy, yy + 0.35, yy, yy - 0.35], color="#c8102e", zorder=3)
        ticks.append(yy)
        names.append(name)
    ax.set_yticks(ticks)
    ax.set_yticklabels(names, fontsize=8)
    ax.axvline(1.0 if exp else 0.0, color="#888888", lw=0.8)
    if f is not None:
        ax.axvline(tr(float(f.beta[0])), color="#c8102e", lw=0.6, ls="--")
    if exp:
        ax.set_xscale("log")
    ax.set_ylim(min(ticks) - 1, rows + 1)
    ax.set_xlabel("Effect Size" + (" (exponentiated)" if exp else ""))
    ax.set_title(title, fontsize=10)
    # Right-hand text column: estimate [CI] and weight, as in SPSS's forest plot.
    trans = ax.get_yaxis_transform()
    ax.text(1.02, rows + 0.7, f"Effect Size [{level * 100:g}% CI]", transform=trans,
            fontsize=7, va="center", fontweight="bold")
    if weights is not None:
        ax.text(1.30, rows + 0.7, "Weight", transform=trans, fontsize=7, va="center",
                fontweight="bold")
    for pos_, i in enumerate(idx):
        ax.text(1.02, rows - pos_, f"{tr(y[i]):.2f} [{tr(lo[i]):.2f}, {tr(hi[i]):.2f}]",
                transform=trans, fontsize=7, va="center")
        if weights is not None:
            ax.text(1.30, rows - pos_, f"{100 * wts[i] / wts.sum():.1f}%", transform=trans,
                    fontsize=7, va="center")
    for name, ef, yy in diamonds:
        l, h = ef.ci(level)
        b = float(ef.beta[0])
        ax.text(1.02, yy, f"{tr(b):.2f} [{tr(l[0]):.2f}, {tr(h[0]):.2f}]", transform=trans,
                fontsize=7, va="center", fontweight="bold")
    fig.subplots_adjust(right=0.7)
    ax.grid(axis="x", alpha=0.3)
    return png(fig)


def cumulative_png(labels, fits: list[MetaFit], level=0.95, exp=False) -> bytes:
    k = len(fits)
    fig = figure(7.0, max(2.6, 0.32 * k + 1.4))
    ax = fig.add_subplot(111)
    tr = np.exp if exp else (lambda a: a)
    for i, f in enumerate(fits):
        lo, hi = f.ci(level)
        yy = k - i
        ax.plot([tr(lo[0]), tr(hi[0])], [yy, yy], color="#333333", lw=1)
        ax.plot([tr(float(f.beta[0]))], [yy], "o", color="#1f4e9c")
    ax.set_yticks(range(k, 0, -1))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(1.0 if exp else 0.0, color="#888888", lw=0.8)
    if exp:
        ax.set_xscale("log")
    ax.set_xlabel("Cumulative Effect Size")
    ax.set_title("Cumulative Forest Plot", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    return png(fig)


def funnel_png(y, v, f: MetaFit, level=0.95, filled: TrimFill | None = None,
               exp=False) -> bytes:
    se = np.sqrt(v)
    fig = figure(6.0, 4.6)
    ax = fig.add_subplot(111)
    tr = np.exp if exp else (lambda a: a)
    b = float(f.beta[0])
    smax = float(max(se.max(), np.sqrt(filled.v_fill).max() if filled is not None and
                     filled.k0 else 0)) * 1.05
    z = stats.norm.ppf(0.5 + level / 2)
    s = np.linspace(0, smax, 50)
    ax.fill_betweenx(s, tr(b - z * s), tr(b + z * s), color="#eeeeee")
    ax.plot(tr(b - z * s), s, color="#999999", lw=0.8, ls="--")
    ax.plot(tr(b + z * s), s, color="#999999", lw=0.8, ls="--")
    ax.axvline(tr(b), color="#c8102e", lw=1)
    ax.scatter(tr(y), se, color="#1f4e9c", s=24, zorder=3, label="Observed studies")
    if filled is not None and filled.k0:
        ax.scatter(tr(filled.y_fill), np.sqrt(filled.v_fill), facecolors="white",
                   edgecolors="#c8102e", s=24, zorder=3, label="Imputed studies")
        ax.axvline(tr(float(filled.fit.beta[0])), color="#c8102e", lw=0.8, ls=":")
        ax.legend(fontsize=7, loc="lower right")
    ax.set_ylim(smax, 0)
    if exp:
        ax.set_xscale("log")
    ax.set_xlabel("Effect Size")
    ax.set_ylabel("Standard Error")
    ax.set_title("Funnel Plot", fontsize=10)
    return png(fig)


def galbraith_png(y, v, f: MetaFit) -> bytes:
    se = np.sqrt(v)
    x, zz = 1 / se, y / se
    fig = figure(6.0, 4.4)
    ax = fig.add_subplot(111)
    xs = np.linspace(0, x.max() * 1.05, 20)
    b = float(f.beta[0])
    ax.plot(xs, b * xs, color="#c8102e")
    ax.plot(xs, b * xs + 1.96, color="#999999", ls="--", lw=0.8)
    ax.plot(xs, b * xs - 1.96, color="#999999", ls="--", lw=0.8)
    ax.scatter(x, zz, color="#1f4e9c", s=24, zorder=3)
    ax.axhline(0, color="#888888", lw=0.6)
    ax.set_xlabel("Precision (1 / SE)")
    ax.set_ylabel("Standardized Effect Size (ES / SE)")
    ax.set_title("Galbraith Plot", fontsize=10)
    return png(fig)


def labbe_png(counts: np.ndarray, f: MetaFit | None, measure: str) -> bytes:
    a, b, c, d = (counts[:, j] for j in range(4))
    pt, pc = a / (a + b), c / (c + d)
    n = counts.sum(axis=1)
    fig = figure(5.2, 5.0)
    ax = fig.add_subplot(111)
    ax.scatter(pc, pt, s=20 + 300 * n / n.max(), facecolors="none", edgecolors="#1f4e9c")
    xs = np.linspace(0.001, 0.999, 100)
    ax.plot([0, 1], [0, 1], color="#888888", lw=0.8)
    if f is not None:
        b0 = float(f.beta[0])
        if measure == "rr":
            ys = np.clip(xs * math.exp(b0), 0, 1)
        elif measure == "rd":
            ys = np.clip(xs + b0, 0, 1)
        else:
            o = xs / (1 - xs) * math.exp(b0)
            ys = o / (1 + o)
        ax.plot(xs, ys, color="#c8102e", ls="--", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Event Rate (Control Group)")
    ax.set_ylabel("Event Rate (Treatment Group)")
    ax.set_title("L'Abbé Plot", fontsize=10)
    return png(fig)


def bubble_png(x, y, v, f: MetaFit, j: int, xname: str) -> bytes:
    fig = figure(6.0, 4.4)
    ax = fig.add_subplot(111)
    w = f.weights
    ax.scatter(x, y, s=20 + 400 * w / w.max(), facecolors="none", edgecolors="#1f4e9c")
    xs = np.linspace(np.min(x), np.max(x), 50)
    ax.plot(xs, f.beta[0] + f.beta[j] * xs, color="#c8102e")
    ax.set_xlabel(xname)
    ax.set_ylabel("Effect Size")
    ax.set_title(f"Bubble Plot: Effect Size by {xname}", fontsize=10)
    return png(fig)


# =========================================================================== main analysis
@dataclass
class MetaOptions:
    model: str = "random"
    method: str = "REML"
    knapp: str = "none"
    level: float = 0.95
    mh: bool = False                 # Mantel-Haenszel fixed effects for binary raw data
    individual: bool = True
    homogeneity: bool = True
    prediction: bool = True
    exp: bool = True                 # exponentiated estimates for log measures
    cumulative: bool = False
    leave_one_out: bool = False
    egger: bool = False
    begg: bool = False
    trimfill: bool = False
    tf_estimator: str = "L0"
    tf_side: str = "auto"
    plots: tuple = ("forest",)
    forest_sort: str = "none"
    save: tuple = ()


def meta_analysis(ds: Dataset, mode: str, variables: dict, measure: str,
                  opts: MetaOptions | None = None, study=None, label=None, group=None,
                  order=None, zero: str = "only", se_kind: str = "se") -> Result:
    o = opts or MetaOptions()
    sd = prepare(ds, mode, variables, measure, zero, se_kind, study, label, group, order)
    title = {"cont_raw": "Meta-Analysis Continuous Outcomes",
             "cont_es": "Meta-Analysis Continuous Outcomes",
             "bin_raw": "Meta-Analysis Binary Outcomes",
             "bin_es": "Meta-Analysis Binary Outcomes"}[mode]
    sec = Section(title)
    cmd = "META BINARY" if mode.startswith("bin") else "META ES" if mode.endswith("es") \
        else "META CONTINUOUS"
    sec.log(f"{cmd} {' '.join(str(v) for v in variables.values())}\n"
            f"  /METHOD MODEL={o.model.upper()}" +
            (f" ESTIMATOR={o.method}" if o.model == "random" else "") +
            ("\n  /ADJUST=KH" + ("(TRUNCATE)" if o.knapp == "truncated" else "")
             if o.knapp != "none" else "") +
            f"\n  /CRITERIA CILEVEL={o.level * 100:g}")
    if len(sd.y) < 2:
        sec.note("Warning", "At least two studies with valid effect sizes are needed.",
                 "warning")
        return Result(sec)
    y, v = sd.y, sd.v
    if sd.order is not None:
        perm = np.array(sorted(range(len(y)), key=lambda i: sort_key(sd.order[i])))
    else:
        perm = np.arange(len(y))
    level = o.level
    exp = o.exp and (sd.measure in LOG_MEASURES)
    f = fit(y, v, None, o.model, o.method, o.knapp)
    mh_est = None
    if o.model == "fixed" and o.mh and sd.counts is not None and sd.measure in ("or", "rr",
                                                                                 "rd"):
        cnt = sd.counts
        mh_est = mantel_haenszel(*(cnt[:, j] for j in range(4)), measure=sd.measure)
        f.beta = np.array([mh_est[0]])
        f.cov = np.array([[mh_est[1] ** 2]])

    sec.table(Table("Case Processing Summary", ["", "N", "Percent"],
                    [["Included", fmt_int(len(y)), _f(100 * len(y) / (len(y) + sd.n_excluded),
                                                       1) + "%"],
                     ["Excluded", fmt_int(sd.n_excluded),
                      _f(100 * sd.n_excluded / (len(y) + sd.n_excluded), 1) + "%"],
                     ["Total", fmt_int(len(y) + sd.n_excluded), "100.0%"]]))
    notes = [f"a. Effect size: {sd.measure_label}.", "b. " + _model_note(f)]
    if mh_est is not None:
        notes[1] = "b. Fixed-effects model, Mantel-Haenszel method."
    if o.model == "random" and o.prediction:
        notes.append("c. The prediction interval uses the estimated tau-squared.")
    pi = o.prediction and o.model == "random"
    sec.table(Table("Effect Size Estimatesᵃ ᵇ", _estimate_cols(f, level, pi=pi),
                    [_estimate_row("Overall", f, level, pi=pi)], footnotes=notes))
    if exp:
        sec.table(Table("Effect Size Estimates (Exponentiated)",
                        _estimate_cols(f, level, exp=True, pi=pi),
                        [_estimate_row("Overall", f, level, exp=True, pi=pi)],
                        footnotes=[f"Exponentiated {sd.measure_label.lower()}."]))
    if o.homogeneity:
        fr = f if o.model == "random" else fit(y, v, None, "random", "DL")
        sec.table(Table("Heterogeneity", HETERO_COLS, [_hetero_rows("Overall", fr)],
                        footnotes=[f"Tau-squared estimated by {ESTIMATORS[fr.method]}."]))
    if o.individual:
        z = stats.norm.ppf(0.5 + level / 2)
        se = np.sqrt(v)
        rows = []
        for i in perm:
            r = [sd.labels[i], _f(y[i]), _f(se[i]), _f(y[i] - z * se[i]), _f(y[i] + z * se[i]),
                 _f(f.weight_pct[i], 2) if mh_est is None else ""]
            if exp:
                r += [_f(math.exp(y[i])), _f(math.exp(y[i] - z * se[i])),
                      _f(math.exp(y[i] + z * se[i]))]
            rows.append(r)
        rows.append(["Overall"] + _estimate_row("", f, level, pi=False)[1:3] +
                    [_f(f.ci(level)[0][0]), _f(f.ci(level)[1][0]), "100.00"] +
                    ([_f(math.exp(f.beta[0])), _f(math.exp(f.ci(level)[0][0])),
                      _f(math.exp(f.ci(level)[1][0]))] if exp else []))
        ci = f"{level * 100:g}% Conf. Interval"
        cols = ["Study", "Effect Size", "Std. Error", (ci, "Lower"), (ci, "Upper"),
                "Weight (%)"]
        if exp:
            cols += [("Exponentiated", "Effect Size"), ("Exponentiated", "Lower"),
                     ("Exponentiated", "Upper")]
        sec.table(Table("Effect Size Estimates for Individual Studies", cols, rows))

    # subgroups
    sub_fits = []
    if sd.group is not None:
        srows, hrows = [], []
        for lv, lab in sd.group_labels:
            m = sd.group == lv
            if m.sum() < 1:
                continue
            fg = fit(y[m], v[m], None, o.model if m.sum() > 1 else "fixed", o.method,
                     o.knapp if m.sum() > 2 else "none")
            sub_fits.append((lab, fg, m))
            srows.append(_estimate_row(lab, fg, level, pi=False))
            if m.sum() > 1:
                hrows.append(_hetero_rows(lab, fg))
        srows.append(_estimate_row("Overall", f, level, pi=False))
        sec.table(Table("Subgroup Analysis: Effect Size Estimates",
                        _estimate_cols(f, level, pi=False), srows,
                        footnotes=[_model_note(f) + " Separate estimates per subgroup."]))
        if hrows:
            sec.table(Table("Subgroup Analysis: Heterogeneity", HETERO_COLS, hrows))
        b = np.array([float(x.beta[0]) for _, x, _ in sub_fits])
        w = np.array([1 / float(x.cov[0, 0]) for _, x, _ in sub_fits])
        if len(b) > 1:
            mb = (w * b).sum() / w.sum()
            qb = float((w * (b - mb) ** 2).sum())
            sec.table(Table("Test of Subgroup Differences", ["", "Chi-square", "df", "Sig."],
                            [["Between subgroups", _f(qb), fmt_int(len(b) - 1),
                              fmt_p(float(stats.chi2.sf(qb, len(b) - 1)))]]))

    # bias
    tf = None
    if o.egger and len(y) >= 3:
        coef, se_c, stat, p, df_t, _ = egger_test(y, v, o.model, o.method, o.knapp)
        c = stats.t.ppf(0.5 + level / 2, df_t) if df_t else stats.norm.ppf(0.5 + level / 2)
        ci = f"{level * 100:g}% Conf. Interval"
        sec.table(Table("Egger's Regression-Based Test",
                        ["", "Coefficient", "Std. Error", "t" if df_t else "Z", "Sig.",
                         (ci, "Lower"), (ci, "Upper")],
                        [["Std. Error", _f(coef), _f(se_c), _f(stat), fmt_p(p),
                          _f(coef - c * se_c), _f(coef + c * se_c)]],
                        footnotes=["Meta-regression of the effect size on its standard error. "
                                   "A significant coefficient indicates small-study effects "
                                   "(funnel plot asymmetry).", _model_note(f)]))
    if o.begg and len(y) >= 3:
        tau, p = begg_test(y, v)
        sec.table(Table("Begg's Rank Correlation Test", ["", "Kendall's tau", "Sig."],
                        [["Standardized effect size vs. variance", _f(tau), fmt_p(p)]]))
    if o.trimfill and len(y) >= 3:
        side = None if o.tf_side == "auto" else o.tf_side
        tf = trim_and_fill(y, v, o.model, o.method, o.tf_estimator, side)
        ci = f"{level * 100:g}% Conf. Interval"
        rows = []
        for name, ff, k in (("Observed", f, len(y)), ("Observed + Imputed", tf.fit,
                                                      len(y) + tf.k0)):
            lo, hi = ff.ci(level)
            r = [name, fmt_int(k), _f(ff.beta[0]), _f(ff.se[0]), _f(lo[0]), _f(hi[0])]
            if exp:
                r += [_f(math.exp(ff.beta[0])), _f(math.exp(lo[0])), _f(math.exp(hi[0]))]
            rows.append(r)
        cols = ["", "Number of Studies", "Effect Size", "Std. Error", (ci, "Lower"),
                (ci, "Upper")]
        if exp:
            cols += [("Exponentiated", "Effect Size"), ("Exponentiated", "Lower"),
                     ("Exponentiated", "Upper")]
        sec.table(Table("Trim-and-Fill Analysis", cols, rows,
                        footnotes=[f"{tf.k0} studies imputed on the {tf.side} side "
                                   f"(estimator {o.tf_estimator}).", _model_note(f)]))

    if o.cumulative:
        ys, vs = y[perm], v[perm]
        cf = cumulative(ys, vs, o.model, o.method, o.knapp)
        rows = []
        for i, ff in enumerate(cf):
            lo, hi = ff.ci(level)
            rows.append([sd.labels[perm[i]], _f(ff.beta[0]), _f(ff.se[0]),
                         fmt_p(float(ff.pvalues()[0])), _f(lo[0]), _f(hi[0])])
        ci = f"{level * 100:g}% Conf. Interval"
        sec.table(Table("Cumulative Effect Size Estimates",
                        ["Study Added", "Effect Size", "Std. Error", "Sig.", (ci, "Lower"),
                         (ci, "Upper")], rows,
                        footnotes=["Each row pools the studies up to and including that row."
                                   + (" Ordered by " + ds.var(order).title + "." if order
                                      else "")]))
        if "forest" in o.plots or "cumulative" in o.plots:
            sec.image("Cumulative Forest Plot",
                      cumulative_png([sd.labels[i] for i in perm], cf, level, exp))
    if o.leave_one_out:
        lf = leave_one_out(y, v, o.model, o.method, o.knapp)
        rows = []
        for i in perm:
            ff = lf[i]
            lo, hi = ff.ci(level)
            rows.append([sd.labels[i], _f(ff.beta[0]), _f(ff.se[0]),
                         fmt_p(float(ff.pvalues()[0])), _f(lo[0]), _f(hi[0]), _f(ff.tau2, 4),
                         _f(ff.i2, 2)])
        ci = f"{level * 100:g}% Conf. Interval"
        sec.table(Table("Leave-One-Out Analysis",
                        ["Study Omitted", "Effect Size", "Std. Error", "Sig.", (ci, "Lower"),
                         (ci, "Upper"), "Tau-squared", "I-squared (%)"], rows))

    # plots
    if "forest" in o.plots:
        extra = [(lab, fg) for lab, fg, _ in sub_fits] if sub_fits else None
        if sub_fits:
            order_idx = np.concatenate([np.where(m)[0] for _, _, m in sub_fits])
        else:
            order_idx = perm
        sec.image("Forest Plot", forest_png([sd.labels[i] for i in order_idx], y[order_idx],
                                            v[order_idx], f, level, exp,
                                            weights=f.weights[order_idx]
                                            if mh_est is None else None, extra=extra,
                                            sort=None if o.forest_sort == "none" else
                                            o.forest_sort))
    if "funnel" in o.plots:
        sec.image("Funnel Plot", funnel_png(y, v, f, level, tf, exp))
    if "galbraith" in o.plots:
        sec.image("Galbraith Plot", galbraith_png(y, v, f))
    if "labbe" in o.plots and sd.counts is not None:
        sec.image("L'Abbé Plot", labbe_png(sd.counts, f, sd.measure))

    saved = {}
    if o.save:
        z = stats.norm.ppf(0.5 + level / 2)
        se = np.sqrt(v)
        idx = ds.df.index
        series = {"ES": (f"Effect size ({sd.measure_label})", y),
                  "SE": ("Standard error of effect size", se),
                  "LCL": ("Lower confidence limit of effect size", y - z * se),
                  "UCL": ("Upper confidence limit of effect size", y + z * se),
                  "WT": ("Study weight (%)", f.weight_pct)}
        for key in o.save:
            if key in series:
                lab, vals = series[key]
                s = pd.Series(np.nan, index=idx)
                s.loc[sd.rows] = vals
                saved[f"ES_{key}" if key != "ES" else "ES"] = (lab, s, 3)
    return Result(sec, saved)


def meta_regression(ds: Dataset, es: str, se: str, covariates: list[str],
                    factors: list[str] = (), model: str = "random", method: str = "REML",
                    knapp: str = "none", level: float = 0.95, se_kind: str = "se",
                    bubble: bool = True, study=None, label=None) -> Section:
    """META REG: mixed-effects (or fixed-effects) meta-regression with dummy-coded factors
    (the first category is the reference, as metafor and SPSS print it)."""
    sec = Section("Meta-Regression")
    sec.log(f"META REG ES={es} SE={se}" + (" WITH " + " ".join(covariates) if covariates else "")
            + (" BY " + " ".join(factors) if factors else "") +
            f"\n  /METHOD MODEL={model.upper()}" +
            (f" ESTIMATOR={method}" if model == "random" else ""))
    cols = {"_y": numeric(ds, es), "_s": numeric(ds, se)}
    for c in covariates:
        cols[c] = numeric(ds, c)
    for fct in factors:
        cols[fct] = ds.analysis_series(fct)
    frame = pd.DataFrame(cols).dropna()
    y = frame["_y"].to_numpy(float)
    v = frame["_s"].to_numpy(float)
    if se_kind != "var":
        v = v ** 2
    names, X = ["Intercept"], [np.ones(len(y))]
    for fct in factors:
        levels = sorted(pd.unique(frame[fct]), key=sort_key)
        for lv in levels[1:]:
            names.append(f"{ds.var(fct).title}={value_text(ds.var(fct), lv)}")
            X.append((frame[fct] == lv).to_numpy(float))
    for c in covariates:
        names.append(ds.var(c).title)
        X.append(frame[c].to_numpy(float))
    X = np.column_stack(X)
    if len(y) <= X.shape[1]:
        sec.note("Warning", "Too few studies for the number of predictors.", "warning")
        return sec
    f = fit(y, v, X, model, method, knapp, names)
    lo, hi = f.ci(level)
    stat, p = f.stat(), f.pvalues()
    ci = f"{level * 100:g}% Conf. Interval"
    rows = [[n, _f(b), _f(s), _f(t), fmt_p(pp), _f(l), _f(h)]
            for n, b, s, t, pp, l, h in zip(names, f.beta, f.se, stat, p, lo, hi)]
    sec.table(Table("Parameter Estimates",
                    ["Parameter", "Coefficient", "Std. Error", "t" if f.df_t else "Z", "Sig.",
                     (ci, "Lower"), (ci, "Upper")], rows,
                    footnotes=[_model_note(f), f"Number of studies = {len(y)}."]))
    if f.df_t:
        qm_p = float(stats.f.sf(f.qm, f.qm_df, f.df_t))
        mrow = ["Model", _f(f.qm), f"{f.qm_df}, {f.df_t}", fmt_p(qm_p)]
        mcols = ["", "F", "df", "Sig."]
    else:
        qm_p = float(stats.chi2.sf(f.qm, f.qm_df))
        mrow = ["Model", _f(f.qm), fmt_int(f.qm_df), fmt_p(qm_p)]
        mcols = ["", "Chi-square (QM)", "df", "Sig."]
    sec.table(Table("Test of Model", mcols, [mrow],
                    footnotes=["Omnibus test of all coefficients except the intercept."]))
    sec.table(Table("Residual Heterogeneity", ["", "Tau-squared", "Tau", "I-squared (%)",
                                               "H-squared", "R-squared (%)",
                                               ("Test of Residual Homogeneity", "Q (QE)"),
                                               ("Test of Residual Homogeneity", "df"),
                                               ("Test of Residual Homogeneity", "Sig.")],
                    [["Residual", _f(f.tau2, 4), _f(math.sqrt(f.tau2), 4), _f(f.i2, 2),
                      _f(f.h2, 3), _f(f.r2, 2) if f.r2 == f.r2 else "", _f(f.q), fmt_int(f.q_df),
                      fmt_p(float(stats.chi2.sf(f.q, f.q_df)))]],
                    footnotes=["R-squared: proportion of tau-squared explained by the "
                               "moderators."] if model == "random" else []))
    if bubble and covariates:
        j = names.index(ds.var(covariates[0]).title)
        sec.image("Bubble Plot", bubble_png(frame[covariates[0]].to_numpy(float), y, v, f, j,
                                            ds.var(covariates[0]).title))
    return sec


# =========================================================================== selftest
BCG = pd.DataFrame({
    "trial": range(1, 14),
    "author": ["Aronson", "Ferguson & Simes", "Rosenthal et al", "Hart & Sutherland",
               "Frimodt-Moller et al", "Stein & Aronson", "Vandiviere et al", "TPT Madras",
               "Coetzee & Berjak", "Rosenthal et al", "Comstock et al", "Comstock & Webster",
               "Comstock et al"],
    "year": [1948, 1949, 1960, 1977, 1973, 1953, 1973, 1980, 1968, 1961, 1974, 1969, 1976],
    "tpos": [4, 6, 3, 62, 33, 180, 8, 505, 29, 17, 186, 5, 27],
    "tneg": [119, 300, 228, 13536, 5036, 1361, 2537, 87886, 7470, 1699, 50448, 2493, 16886],
    "cpos": [11, 29, 11, 248, 47, 372, 10, 499, 45, 65, 141, 3, 29],
    "cneg": [128, 274, 209, 12619, 5761, 1079, 619, 87892, 7232, 1600, 27197, 2338, 17825],
    "ablat": [44, 55, 42, 52, 13, 44, 19, 13, 27, 42, 18, 33, 33],
    "alloc": ["random", "random", "random", "random", "alternate", "alternate", "random",
              "random", "random", "systematic", "systematic", "systematic", "systematic"],
})


def selftest() -> None:
    d = BCG
    y, v = binary_effects(d.tpos, d.tneg, d.cpos, d.cneg, "rr")
    f = fit(y, v, None, "random", "REML")
    assert abs(f.beta[0] + 0.7145) < 1e-3 and abs(f.tau2 - 0.3132) < 1e-3
    ds = Dataset(d.drop(columns=["alloc"]))
    res = meta_analysis(ds, "bin_raw", {"a": "tpos", "b": "tneg", "c": "cpos", "d": "cneg"},
                        "rr", MetaOptions(egger=True, trimfill=True, cumulative=True,
                                          plots=("forest", "funnel", "galbraith", "labbe")),
                        study="author", order="year")
    assert any(getattr(i, "title", "") == "Trim-and-Fill Analysis" for i in res.section.items)

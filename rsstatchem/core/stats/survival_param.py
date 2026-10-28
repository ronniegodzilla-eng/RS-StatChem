"""Analyze > Survival > Parametric Accelerated Failure Time (AFT) Models and Parametric Shared
Frailty Models (SPSS 29+ SURVREG AFT / SURVREG RECURRENT).

The procedures follow the IBM SPSS Advanced Statistics manual (v31, pp. 71-87):

* **AFT** for non-recurrent data: ``log T = x'b + sigma * e`` with Weibull (extreme-value e),
  exponential (sigma = 1), log-normal (normal e) or log-logistic (logistic e) survival times.
  Time is one Survival variable or Start/End variables; the Status variable maps records to
  Failure/Event, Right, Left or Interval censoring (string defaults F, R, L, I), unmapped
  values go to a chosen category or are excluded; Start/End records whose status conflicts
  with the times are discarded (default), timed by the status or re-statused by the times; an
  optional Left Truncation (delayed entry) variable conditions on survival to entry.
  Covariates and Fixed Factors (indicator coding, last category or a selected category as
  the baseline) enter as main effects. Exp(B) is the acceleration factor.
* **Shared frailty** for recurrent data: records of a Subject share a frailty ``z`` that
  multiplies the hazard of the AFT baseline (gamma frailty, mean 1 and variance theta; or
  inverse Gaussian); the marginal likelihood of a subject is
  ``prod h_ij^d_ij * (-1)^D L^(D)(sum H_ij)`` with L the frailty Laplace transform. The
  likelihood-ratio test of the frailty component compares with the model without frailty
  (theta = 0 is on the boundary, so p = P(chi2(1) > LR) / 2). Unconditional (population)
  survival ``L(H)`` and hazard are plotted and saved.

Estimation is maximum likelihood by Newton-Raphson with step-halving on (b, log sigma
[, log theta]) using analytic gradients (observed information by differentiating them);
the initial intercept and scale come from the OLS regression of log time on the predictors
(IBM's default "standard error of the corresponding OLS regression"). The L1 regularised
ADMM feature selection of SPSS is not implemented. Output: Model Summary, Case Processing
Summary, factor encoding, initial values, iteration history, model fit with the omnibus
likelihood-ratio test, parameter estimates with acceleration factors and confidence limits,
the scale (and Weibull shape), the frailty component, Cox-Snell residual and survival /
hazard / density function plots (at the covariate means and factor category frequencies, or
separate lines for a factor) and the Predict variables (PredSurvival, PredHazard,
PredCumHazard, PredConditionalSurvival; for frailty models the unconditional versions too).
Log likelihoods are on the time scale (they include -log t for exact times), as in R
``survival::survreg``.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import special, stats

from rsstatchem.core.dataset import NUMERIC, Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, fmt_p, numeric, pct, sup, value_text
from rsstatchem.core.stats.wp4_util import COLORS, figure, png

DISTRIBUTIONS = {"weibull": "Weibull", "exponential": "Exponential",
                 "lognormal": "Log-Normal", "loglogistic": "Log-Logistic"}
FRAILTIES = {"gamma": "Gamma", "invgauss": "Inverse-Gaussian"}
STATUS = {"event": "Failure/Event", "right": "Right Censoring", "left": "Left Censoring",
          "interval": "Interval Censoring"}
STRING_DEFAULTS = {"event": ["F"], "right": ["R"], "left": ["L"], "interval": ["I"]}
EXACT, RIGHT, LEFT, INTERVAL = 0, 1, 2, 3


@dataclass
class SurvParamResult:
    section: Section
    saved: dict = field(default_factory=dict)
    models: list = field(default_factory=list)
    fit: object = None


# ====================================================================== error distributions
def _err(dist: str):
    """(log f0, log S0, log F0, d log f0/dz, d log S0/dz, d log F0/dz) of the standardised
    log-time error."""
    if dist in ("weibull", "exponential"):
        def lf(z):
            return z - np.exp(np.minimum(z, 700))

        def ls(z):
            return -np.exp(np.minimum(z, 700))

        def lF(z):
            e = np.exp(np.minimum(z, 700))
            return np.log(-np.expm1(-e))

        def dlf(z):
            return 1 - np.exp(np.minimum(z, 700))

        def dls(z):
            return -np.exp(np.minimum(z, 700))

        def dlF(z):
            e = np.exp(np.minimum(z, 700))
            return e * np.exp(-e) / -np.expm1(-e)
        return lf, ls, lF, dlf, dls, dlF
    if dist == "lognormal":
        def dls_n(z):
            return -np.exp(stats.norm.logpdf(z) - stats.norm.logsf(z))

        def dlF_n(z):
            return np.exp(stats.norm.logpdf(z) - stats.norm.logcdf(z))
        return (stats.norm.logpdf, stats.norm.logsf, stats.norm.logcdf, lambda z: -z,
                dls_n, dlF_n)

    def lf_l(z):
        return -np.logaddexp(0, -z) - np.logaddexp(0, z)

    def ls_l(z):
        return -np.logaddexp(0, z)

    def lF_l(z):
        return -np.logaddexp(0, -z)
    return (lf_l, ls_l, lF_l, lambda z: 1 - 2 * special.expit(z),
            lambda z: -special.expit(z), lambda z: 1 - special.expit(z))


# ====================================================================== AFT likelihood
@dataclass
class _Data:
    X: np.ndarray            # design incl. intercept
    kind: np.ndarray         # EXACT / RIGHT / LEFT / INTERVAL
    y1: np.ndarray           # log time (exact / right / left), log lower bound (interval)
    y2: np.ndarray           # log upper bound (interval)
    ytr: np.ndarray | None   # log left-truncation time (-inf = none)
    t_end: np.ndarray        # time used for prediction
    t_start: np.ndarray      # start time (recurrent data)
    groups: np.ndarray | None = None


def _aft_parts(par, d: _Data, dist: str):
    """Per-record log-likelihood, d/d eta and d/d log sigma, and log h, H (for frailty)."""
    p = d.X.shape[1]
    b = par[:p]
    ls_ = 0.0 if dist == "exponential" else par[p]
    s = math.exp(ls_)
    eta = d.X @ b
    lf, lS, lF, dlf, dlS, dlF = _err(dist)
    n = len(eta)
    ll = np.zeros(n)
    dz1 = np.zeros(n)        # dl/dz1
    dz2 = np.zeros(n)        # dl/dz2 (interval upper)
    z1 = (d.y1 - eta) / s
    z2 = (d.y2 - eta) / s
    ex = d.kind == EXACT
    ll[ex] = lf(z1[ex]) - ls_ - d.y1[ex]
    dz1[ex] = dlf(z1[ex])
    ri = d.kind == RIGHT
    ll[ri] = lS(z1[ri])
    dz1[ri] = dlS(z1[ri])
    le = d.kind == LEFT
    ll[le] = lF(z1[le])
    dz1[le] = dlF(z1[le])
    iv = d.kind == INTERVAL
    if iv.any():
        a1, a2 = lS(z1[iv]), lS(z2[iv])
        with np.errstate(divide="ignore", invalid="ignore"):
            lw = a1 + np.log(-np.expm1(np.minimum(a2 - a1, -1e-300)))
        ll[iv] = lw
        f1 = np.exp(lf(z1[iv]) - lw)
        f2 = np.exp(lf(z2[iv]) - lw)
        dz1[iv] = -f1
        dz2[iv] = f2
    # left truncation: - log S(ztr)
    dztr = np.zeros(n)
    if d.ytr is not None:
        tr = np.isfinite(d.ytr)
        if tr.any():
            ztr = (d.ytr[tr] - eta[tr]) / s
            ll[tr] -= lS(ztr)
            dztr[tr] = -dlS(ztr)
    # chain rule: dz/deta = -1/s, dz/dlogs = -z
    deta = -(dz1 + dz2 + dztr) / s
    zt = np.zeros(n)
    if d.ytr is not None:
        tr = np.isfinite(d.ytr)
        zt[tr] = (d.ytr[tr] - eta[tr]) / s
    with np.errstate(invalid="ignore"):
        dls = -(dz1 * np.where(np.isfinite(z1), z1, 0) + dz2 * np.where(np.isfinite(z2), z2, 0)
                + dztr * zt)
    dls[ex] -= 1
    return ll, deta, dls


def _aft_loglik(par, d, dist):
    ll, deta, dls = _aft_parts(par, d, dist)
    g = d.X.T @ deta
    if dist != "exponential":
        g = np.r_[g, dls.sum()]
    return float(ll.sum()), g


# ====================================================================== frailty likelihood
def _hazards(par, d: _Data, dist: str):
    """log h(end), H(end) - H(start) per record (H from the AFT baseline)."""
    p = d.X.shape[1]
    b = par[:p]
    ls_ = 0.0 if dist == "exponential" else par[p]
    s = math.exp(ls_)
    eta = d.X @ b
    lf, lS, *_ = _err(dist)
    z = (d.y1 - eta) / s
    logh = lf(z) - lS(z) - ls_ - d.y1
    H = -lS(z)
    st = d.t_start > 0
    if st.any():
        z0 = (np.log(d.t_start[st]) - eta[st]) / s
        H = H.copy()
        H[st] = H[st] + lS(z0)
    return logh, H


def _lt_logderiv(frailty: str, theta: float, s: np.ndarray, D: np.ndarray) -> np.ndarray:
    """log((-1)^D L^(D)(s)) of the frailty Laplace transform (mean 1, variance theta)."""
    if frailty == "gamma":
        a = 1 / theta
        return (special.gammaln(a + D) - special.gammaln(a) + D * math.log(theta)
                - (a + D) * np.log1p(theta * s))
    out = np.zeros(len(s))
    for i, (si, di) in enumerate(zip(s, D)):
        base = 1 + 2 * theta * si
        psi0 = (1 - math.sqrt(base)) / theta
        di = int(di)
        if di == 0:
            out[i] = psi0
            continue
        # psi^(j) = -(1/theta) (2 theta)^j c_j base^(1/2 - j); work with a_j = (-1)^j psi^(j)
        a = np.zeros(di + 1)
        c = 1.0
        for j in range(1, di + 1):
            c *= (0.5 - (j - 1))
            a[j] = (-1) ** j * (-(1 / theta) * (2 * theta) ** j * c * base ** (0.5 - j))
        # r_n = (-1)^n g^(n)/g with g = exp(psi): r_n = sum C(n-1,k) a_{k+1} r_{n-1-k}
        r = np.zeros(di + 1)
        r[0] = 1.0
        for nn in range(1, di + 1):
            r[nn] = sum(math.comb(nn - 1, k) * a[k + 1] * r[nn - 1 - k] for k in range(nn))
        out[i] = psi0 + math.log(max(r[di], 1e-300))
    return out


def _frailty_ll(par, d: _Data, dist: str, frailty: str, ng: int) -> float:
    k = d.X.shape[1] + (0 if dist == "exponential" else 1)
    theta = math.exp(par[k])
    logh, H = _hazards(par[:k], d, dist)
    ev = (d.kind == EXACT).astype(float)
    Di = np.bincount(d.groups, ev, ng)
    Hi = np.bincount(d.groups, H, ng)
    lh = np.bincount(d.groups, ev * logh, ng)
    return float(np.sum(lh + _lt_logderiv(frailty, theta, Hi, Di)))


def _num_grad(f, x, h=1e-6):
    g = np.zeros(len(x))
    for i in range(len(x)):
        e = np.zeros(len(x))
        e[i] = h * max(1.0, abs(x[i]))
        g[i] = (f(x + e) - f(x - e)) / (2 * e[i])
    return g


def _jac(grad, x, h=1e-5):
    k = len(x)
    J = np.zeros((k, k))
    for i in range(k):
        e = np.zeros(k)
        e[i] = h * max(1.0, abs(x[i]))
        J[:, i] = (grad(x + e) - grad(x - e)) / (2 * e[i])
    return (J + J.T) / 2


def _newton(fg, x0, max_iter=200, max_halving=20, tol=1e-9, history=None):
    """Maximise f by Newton-Raphson with step-halving (Hessian from the gradient)."""
    x = np.asarray(x0, float)
    f, g = fg(x)
    it = 0
    if history is not None:
        history.append((0, 0, f, x.copy()))
    for it in range(1, max_iter + 1):
        H = _jac(lambda z: fg(z)[1], x)
        try:
            step = -np.linalg.solve(H, g)
            if g @ step <= 0:   # not an ascent direction
                raise np.linalg.LinAlgError
        except np.linalg.LinAlgError:
            step = g / max(np.linalg.norm(g), 1.0)
        halv = 0
        while True:
            xn = x + step
            fn, gn = fg(xn)
            if np.isfinite(fn) and fn >= f - 1e-12:
                break
            step = step / 2
            halv += 1
            if halv > max_halving:
                xn, fn, gn = x, f, g
                break
        conv = abs(fn - f) < tol * (abs(f) + 1) and np.max(np.abs(xn - x)) < 1e-6
        x, f, g = xn, fn, gn
        if history is not None:
            history.append((it, halv, f, x.copy()))
        if conv or halv > max_halving:
            break
    H = _jac(lambda z: fg(z)[1], x)
    return x, f, H, it


# ====================================================================== data preparation
def _status_kind(values: pd.Series, var, status_map: dict | None, unmapped: str,
                 default: str) -> pd.Series:
    if values is None:
        return None
    is_str = var.type != NUMERIC
    smap = dict(status_map or {})
    if not smap:
        smap = dict(STRING_DEFAULTS) if is_str else {"event": [1.0], "right": [0.0]}
    out = pd.Series(np.nan, index=values.index, dtype=float)
    codes = {"event": EXACT, "right": RIGHT, "left": LEFT, "interval": INTERVAL}
    for key, vals in smap.items():
        if not vals:
            continue
        if is_str:
            hit = values.astype(str).str.strip().isin([str(v).strip() for v in vals])
        else:
            hit = values.isin([float(v) for v in vals])
        out[hit & out.isna()] = codes[key]
    if unmapped != "exclude":
        out[out.isna() & values.notna()] = codes.get(unmapped, codes[default])
    out[values.isna()] = np.nan
    return out


def _factor_columns(frame, ds, factors, reference):
    cols, names, enc, levels_all = [], [], [], {}
    for f in factors:
        var = ds.var(f)
        lv = sorted(frame[f].unique(), key=lambda v: (isinstance(v, str), v))
        levels_all[f] = lv
        ref = (reference or {}).get(f, "last")
        if ref == "first":
            ref_val = lv[0]
        elif ref == "last" or ref not in lv:
            ref_val = lv[-1]
        else:
            ref_val = ref
        for v in lv:
            enc.append([var.name, value_text(var, v), "(baseline)" if v == ref_val else "1"])
            if v == ref_val:
                continue
            cols.append((frame[f] == v).to_numpy(float))
            names.append(f"[{var.name}={value_text(var, v)}]")
        names_ref = f"[{var.name}={value_text(var, ref_val)}]"
        levels_all[f] = (lv, ref_val, names_ref)
    return cols, names, enc, levels_all


def _prepare(ds, time, start, end, status, status_map, unmapped, covariates, factors,
             truncation, subject, reference, status_treatment, recurrent=False,
             interval_var=None):
    """Returns (frame, _Data, names, encoding, counts dict, levels, excluded counts)."""
    two = start is not None and end is not None
    tnames = [start, end] if two else [time]
    names = list(dict.fromkeys(
        [x for x in tnames + [status, truncation, subject, interval_var] if x] +
        list(covariates) + list(factors)))
    data = {}
    for nme in names:
        if nme in factors or nme == subject or (nme == status and
                                                ds.var(nme).type != NUMERIC):
            data[nme] = ds.analysis_series(nme)
        else:
            data[nme] = numeric(ds, nme)
    frame = pd.DataFrame(data)
    total = len(frame)
    excl = {"missing": 0, "invalid": 0, "conflict": 0, "unmapped": 0}
    need = [x for x in names if x not in tnames]
    keep = frame[need].notna().all(axis=1) if need else pd.Series(True, index=frame.index)
    if two:
        keep &= frame[start].notna() | frame[end].notna()
    else:
        keep &= frame[time].notna()
    excl["missing"] = int((~keep).sum())
    frame = frame[keep]
    default = "interval" if two else "event"
    if status:
        kind = _status_kind(frame[status], ds.var(status), status_map, unmapped, default)
        excl["unmapped"] = int(kind.isna().sum())
        frame = frame[kind.notna()]
        kind = kind[kind.notna()].to_numpy(int)
    else:
        kind = np.full(len(frame), INTERVAL if two else EXACT)
    if two:
        lo = frame[start].to_numpy(float)
        hi = frame[end].to_numpy(float)
        lo = np.where(np.isnan(lo), 0.0, lo)
        hi = np.where(np.isnan(hi), np.inf, hi)
        if recurrent:
            # recurrent records: (start, end] with the status at end
            t1, t2 = hi, np.full(len(hi), np.nan)
            ok = (hi > lo) & np.isfinite(hi) & (lo >= 0)
            conflict = ~ok
        else:
            derived = np.where(lo == hi, EXACT, np.where(~np.isfinite(hi), RIGHT,
                                                         np.where(lo <= 0, LEFT, INTERVAL)))
            if status_treatment == "derive" or not status:
                conflict = (hi < lo)
                kind = derived
            else:
                conflict = (kind != derived) & (hi >= lo)
                conflict |= hi < lo
                if status_treatment == "status":
                    conflict = hi < lo
            t1 = np.where(kind == EXACT, np.where(lo == hi, hi, hi),
                          np.where(kind == RIGHT, np.where(np.isfinite(lo) & (lo > 0), lo, hi),
                                   np.where(kind == LEFT, hi, lo)))
            t2 = np.where(kind == INTERVAL, hi, np.nan)
            if status_treatment == "status" and status:
                bad_iv = (kind == INTERVAL) & ~((lo > 0) & np.isfinite(hi) & (hi > lo))
                kind = np.where(bad_iv & (lo == hi), EXACT, kind)
                conflict |= (kind == INTERVAL) & ~(np.isfinite(hi) & (hi > lo) & (lo > 0))
    else:
        t1 = frame[time].to_numpy(float)
        t2 = np.full(len(t1), np.nan)
        lo = np.zeros(len(t1))
        conflict = np.zeros(len(t1), bool)
        if (kind == INTERVAL).any():
            conflict |= kind == INTERVAL   # interval censoring needs Start/End
    pos = (t1 > 0) & np.isfinite(t1) & (np.isnan(t2) | (t2 > 0))
    excl["invalid"] = int((~pos & ~conflict).sum())
    excl["conflict"] = int(conflict.sum())
    ok = pos & ~conflict
    if recurrent and subject and interval_var:
        # discard whole subjects with overlapping records (interval order vs times)
        sub = frame[subject].to_numpy()
        bad = set()
        order = pd.DataFrame({"s": sub, "i": frame[interval_var].to_numpy(float), "lo": lo,
                              "hi": t1}).sort_values(["s", "i"])
        for s_, grp in order.groupby("s", sort=False):
            if len(grp) > 1 and np.any(grp["lo"].to_numpy()[1:] < grp["hi"].to_numpy()[:-1]):
                bad.add(s_)
        if bad:
            badm = np.isin(sub, list(bad))
            excl["conflict"] += int((badm & ok).sum())
            ok &= ~badm
    frame = frame[ok]
    kind, t1, t2, lo = kind[ok], t1[ok], t2[ok], lo[ok]
    n = len(frame)
    cols = [np.ones(n)]
    pnames = ["(Intercept)"]
    for c in covariates:
        cols.append(frame[c].to_numpy(float))
        pnames.append(ds.var(c).name)
    fcols, fnames, enc, levels = _factor_columns(frame, ds, factors, reference)
    cols += fcols
    pnames += fnames
    X = np.column_stack(cols)
    ytr = None
    if truncation and not two:
        tr = frame[truncation].to_numpy(float)
        ytr = np.where(tr > 0, np.log(np.where(tr > 0, tr, 1.0)), -np.inf)
    with np.errstate(divide="ignore"):
        y1 = np.log(t1)
        y2 = np.log(np.where(np.isnan(t2), 1.0, t2))
    y2 = np.where(kind == INTERVAL, y2, np.nan)
    groups = None
    if subject:
        groups = pd.factorize(frame[subject].astype(str))[0].astype(int)
    data = _Data(X, kind.astype(int), y1, y2, ytr, np.where(kind == INTERVAL, t2, t1),
                 lo if (two and recurrent) else np.zeros(n), groups)
    counts = {k: int(np.sum(kind == code)) for k, code in
              (("event", EXACT), ("right", RIGHT), ("left", LEFT), ("interval", INTERVAL))}
    return frame, data, pnames, enc, counts, levels, excl, total


def _initial(d: _Data, dist: str, init_intercept=None, init_scale="ols",
             scale_value: float | None = None) -> np.ndarray:
    y = np.where(d.kind == INTERVAL, (d.y1 + np.nan_to_num(d.y2)) / 2, d.y1)
    y = np.where(np.isfinite(y), y, np.nanmax(y[np.isfinite(y)]))
    b, *_ = np.linalg.lstsq(d.X, y, rcond=None)
    resid = y - d.X @ b
    dof = max(len(y) - d.X.shape[1], 1)
    se = math.sqrt(max(float(resid @ resid) / dof, 1e-4))
    if init_intercept is not None:
        b[0] = float(init_intercept)
    if init_scale == "inverse":
        se = 1 / se
    elif init_scale == "user" and scale_value:
        se = float(scale_value)
    if dist == "exponential":
        return b
    return np.r_[b, math.log(max(se, 1e-3))]


# ====================================================================== curves
def _curves(par, xrow, dist, tgrid):
    p = len(xrow)
    eta = float(xrow @ par[:p])
    ls_ = 0.0 if dist == "exponential" else par[p]
    s = math.exp(ls_)
    lf, lS, *_ = _err(dist)
    with np.errstate(divide="ignore"):
        z = (np.log(tgrid) - eta) / s
    S = np.exp(lS(z))
    dens = np.exp(lf(z) - ls_ - np.log(tgrid))
    haz = np.exp(lf(z) - lS(z) - ls_ - np.log(tgrid))
    return S, haz, dens


def _uncond(S, haz, theta, frailty):
    """Population (unconditional) survival / hazard / density from conditional curves."""
    H = -np.log(np.clip(S, 1e-300, None))
    if frailty == "gamma":
        Su = (1 + theta * H) ** (-1 / theta)
        hu = haz / (1 + theta * H)
    else:
        a = np.sqrt(1 + 2 * theta * H)
        Su = np.exp((1 - a) / theta)
        hu = haz / a
    return Su, hu, Su * hu


def _plot_functions(kind, curves, title, xlabel, cond_text) -> bytes:
    fig = figure(6.0, 4.2)
    ax = fig.add_subplot(111)
    for i, (label, t, vals) in enumerate(curves):
        ax.plot(t, vals, color=COLORS[i % len(COLORS)], lw=1.6, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel({"survival": "Survival", "hazard": "Hazard", "density": "Density"}[kind])
    ax.set_title(title)
    if kind == "survival":
        ax.set_ylim(0, 1.02)
    if len(curves) > 1:
        ax.legend(fontsize=8, frameon=False)
    ax.text(0.99, 0.99, cond_text, transform=ax.transAxes, ha="right", va="top", fontsize=7,
            color="#555")
    ax.grid(alpha=0.25)
    return png(fig)


def _cox_snell_plot(r: np.ndarray, event: np.ndarray) -> bytes:
    order = np.argsort(r)
    rs, es = r[order], event[order]
    n = len(rs)
    at_risk = n - np.arange(n)
    H = np.cumsum(es / at_risk)
    fig = figure(5.4, 4.6)
    ax = fig.add_subplot(111)
    ax.step(rs, H, where="post", color=COLORS[0], lw=1.4, label="Nelson-Aalen")
    lim = max(float(rs.max()), float(H.max()))
    ax.plot([0, lim], [0, lim], color=COLORS[1], lw=1, ls="--", label="45° reference")
    ax.set_xlabel("Cox-Snell residual")
    ax.set_ylabel("Estimated cumulative hazard of the residuals")
    ax.set_title("Cox-Snell Residual Plot")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    return png(fig)


# ====================================================================== main procedures
def parametric_survival(ds: Dataset, time: str | None = None, status: str | None = None,
                        covariates: list[str] | None = None,
                        factors: list[str] | None = None, start: str | None = None,
                        end: str | None = None, truncation: str | None = None,
                        status_map: dict | None = None, unmapped: str = "exclude",
                        distribution: str = "weibull", reference: dict | None = None,
                        level: float = 0.95, status_treatment: str = "discard",
                        init_intercept: float | None = None, init_scale: str = "ols",
                        scale_value: float | None = None, max_iter: int = 20,
                        max_halving: int = 5, print_encoding: bool = False,
                        print_initial: bool = False, print_history: bool = False,
                        history_steps: int = 1, cox_snell: bool = False,
                        function_plots: tuple = (), n_points: int = 100,
                        plot_values: dict | None = None, separate: str | None = None,
                        max_lines: int = 10, predict: tuple = (),
                        predict_times: str = "dependent", interval: float | None = None,
                        periods: int = 2, duration: str | None = None,
                        past_time: str | None = None,
                        subject: str | None = None, frailty: str | None = None,
                        frailty_init: float | None = None, interval_var: str | None = None,
                        alpha_frailty: float = 0.05) -> SurvParamResult:
    """SURVREG AFT (``subject`` None) or SURVREG RECURRENT (``subject`` and ``frailty``)."""
    covariates = list(covariates or [])
    factors = list(factors or [])
    recurrent = subject is not None
    if recurrent and frailty is None:
        frailty = "gamma"
    title = "Parametric Shared Frailty Models" if recurrent else \
        "Parametric Accelerated Failure Time Models"
    section = Section(title)
    out = SurvParamResult(section)
    section.log(_syntax(time, start, end, status, covariates, factors, truncation,
                        status_map, distribution, level, subject, frailty, reference,
                        cox_snell, function_plots, separate, predict, interval_var))
    two = start is not None and end is not None
    if not two and time is None:
        section.note("Warning", "Specify a survival time variable or Start and End times.",
                     style="warning")
        return out
    frame, d, pnames, enc, counts, levels, excl, total = _prepare(
        ds, time, start, end, status, status_map, unmapped, covariates, factors,
        truncation, subject, reference, status_treatment, recurrent, interval_var)
    n = len(frame)
    p = d.X.shape[1]
    if n <= p + 1 or counts["event"] + counts["left"] + counts["interval"] == 0:
        section.note("Warning", "Not enough valid records (or no failures) to fit the model.",
                     style="warning")
        return out
    expo = distribution == "exponential"
    k = p + (0 if expo else 1)

    # ------------------------------------------------------------------ fit without frailty
    x0 = _initial(d, distribution, init_intercept, init_scale, scale_value)
    hist: list = []

    def fg(par):
        try:
            ll, g = _aft_loglik(par, d, distribution)
        except FloatingPointError:
            return -np.inf, np.zeros(len(par))
        return (ll, g) if np.isfinite(ll) else (-np.inf, np.zeros(len(par)))

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        par0, ll0, H0, it0 = _newton(fg, x0, max_iter=max(max_iter, 50),
                                     max_halving=max(max_halving, 20),
                                     history=None if recurrent else hist)
        # intercept-only model for the omnibus test
        d_int = _Data(d.X[:, :1], d.kind, d.y1, d.y2, d.ytr, d.t_end, d.t_start, d.groups)
        xi = _initial(d_int, distribution)
        pari, lli, _, _ = _newton(lambda q: _aft_loglik(q, d_int, distribution), xi,
                                  max_iter=100, max_halving=20)
    par, ll, H, iters = par0, ll0, H0, it0
    theta = None
    ng = 0
    lr_frailty = None
    names_all = list(pnames) + ([] if expo else ["(Log scale)"])
    if recurrent:
        ng = int(d.groups.max()) + 1
        th0 = frailty_init if frailty_init else (1.0 if frailty == "gamma" else 0.1)

        def ffg(q):
            f = _frailty_ll(q, d, distribution, frailty, ng)
            if not np.isfinite(f):
                return -np.inf, np.zeros(len(q))
            return f, _num_grad(lambda z: _frailty_ll(z, d, distribution, frailty, ng), q)

        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best = None
            for start_th in (th0, 0.1, 1.0):
                cand = _newton(ffg, np.r_[par0, math.log(start_th)], max_iter=200,
                               max_halving=40, history=[] if best else hist)
                if best is None or cand[1] > best[1] + 1e-9:
                    best = cand
            parf, llf, Hf, itf = best
        theta = math.exp(parf[k])
        lr_frailty = max(2 * (llf - ll0), 0.0)
        par, ll, H, iters = parf, llf, Hf, itf
        names_all.append("(Log frailty variance)")
    try:
        cov = np.linalg.inv(-H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(-H)
        section.note("Warning", "The Hessian matrix is singular; standard errors are "
                     "approximate.", style="warning")
    z = stats.norm.ppf(0.5 + level / 2)

    # ------------------------------------------------------------------ Model Summary
    rows = [["Procedure", title.rstrip("s").replace("Models", "Model")],
            ["Time", f"Start: {ds.var(start).name}; End: {ds.var(end).name}" if two
             else ds.var(time).name],
            ["Status", ds.var(status).name if status else
             ("(none: all records Interval Censoring)" if two and not recurrent else
              "(none: all records Failure/Event)")],
            ["Distribution of Survival Time", DISTRIBUTIONS[distribution]]]
    if recurrent:
        rows += [["Subject", ds.var(subject).name],
                 ["Frailty Distribution", FRAILTIES[frailty]]]
    if covariates:
        rows.append(["Covariates", ", ".join(ds.var(c).name for c in covariates)])
    if factors:
        rows.append(["Fixed Factors", ", ".join(ds.var(f).name for f in factors)])
    if truncation and not two:
        rows.append(["Left Truncation", ds.var(truncation).name])
    rows += [["Estimation Method", "Maximum likelihood (Newton-Raphson)"],
             ["Iterations", fmt_int(iters)]]
    section.table(Table("Model Summary", ["", ""], rows))

    # ------------------------------------------------------------------ Case processing
    rows = []
    inc = sum(counts.values())
    for key in ("event", "right", "left", "interval"):
        if key in ("left", "interval") and recurrent:
            continue
        rows.append(["Included", STATUS[key], fmt_int(counts[key]),
                     fmt(100 * counts[key] / total, 1) + "%"])
    rows.append(["", "Total", fmt_int(inc), fmt(100 * inc / total, 1) + "%"])
    ex_total = total - inc
    for key, text in (("missing", "Records with missing values"),
                      ("unmapped", "Records with an unmapped status"),
                      ("invalid", "Records with nonpositive or invalid times"),
                      ("conflict", "Records with conflicting status or times")):
        if excl[key]:
            rows.append(["Excluded", text, fmt_int(excl[key]),
                         fmt(100 * excl[key] / total, 1) + "%"])
    rows.append(["Excluded" if ex_total and not any(excl.values()) else "", "Total excluded",
                 fmt_int(ex_total), fmt(100 * ex_total / total, 1) + "%"])
    rows.append(["Total", "", fmt_int(total), "100.0%"])
    foot = [f"Number of subjects: {ng}."] if recurrent else []
    section.table(Table("Case Processing Summary", ["", "", "N", "Percent"], rows,
                        row_header_cols=2, footnotes=foot))

    if print_encoding and enc:
        section.table(Table("Factor Encoding", ["Factor", "Value", "Parameter Coding"],
                            [[r[0] if i == 0 or enc[i - 1][0] != r[0] else "", r[1], r[2]]
                             for i, r in enumerate(enc)], row_header_cols=2))
    if print_initial:
        x_ini = list(x0) + ([math.log(th0)] if recurrent else [])
        section.table(Table("Initial Values", ["Parameter", "Initial Value"],
                            [[nm, fmt(v, 4)] for nm, v in zip(names_all, x_ini)]))
    if print_history and hist:
        rows = [[fmt_int(it), fmt_int(hv), fmt(f, 4)] + [fmt(v, 4) for v in x]
                for it, hv, f, x in hist if it % max(history_steps, 1) == 0 or it == hist[-1][0]]
        section.table(Table("Iteration History",
                            ["Iteration", "Step-halving", "Log Likelihood"] + names_all,
                            rows, row_header_cols=1))

    # ------------------------------------------------------------------ Model fit
    nk = len(par)
    aic = -2 * ll + 2 * nk
    bic = -2 * ll + nk * math.log(n)
    aicc = aic + 2 * nk * (nk + 1) / max(n - nk - 1, 1)
    rows = [["Log Likelihood", fmt(ll)], ["-2 Log Likelihood", fmt(-2 * ll)],
            ["Akaike's Information Criterion (AIC)", fmt(aic)],
            ["Finite Sample Corrected AIC (AICC)", fmt(aicc)],
            ["Bayesian Information Criterion (BIC)", fmt(bic)]]
    section.table(Table("Model Fit", ["", "Value"], rows,
                        footnotes=["The log likelihood is on the time scale (it includes "
                                   "-log t of exact failure times).",
                                   "Information criteria are in smaller-is-better form."]))
    if p > 1:
        lr = max(2 * (ll0 - lli), 0.0)
        section.table(Table("Omnibus Test" + sup("a"),
                            ["Likelihood Ratio Chi-Square", "df", "Sig."],
                            [[fmt(lr), fmt_int(p - 1), fmt_p(stats.chi2.sf(lr, p - 1))]],
                            row_header_cols=0,
                            footnotes=["a. Compares the fitted model (without the frailty "
                                       "component) against the intercept-only model."]))
    if recurrent:
        psig = 0.5 * stats.chi2.sf(lr_frailty, 1) if lr_frailty > 0 else 0.5
        sig_txt = "significant" if psig < alpha_frailty else "not significant"
        section.table(Table("Likelihood Ratio Test for the Frailty Component",
                            ["-2 Log Likelihood (without frailty)",
                             "-2 Log Likelihood (with frailty)", "Chi-Square", "df", "Sig."],
                            [[fmt(-2 * ll0), fmt(-2 * ll), fmt(lr_frailty), "1", fmt_p(psig)]],
                            row_header_cols=0,
                            footnotes=["The frailty variance is tested on the boundary of "
                                       "its parameter space: Sig. = P(Chi-Square(1) > LR) / 2.",
                                       f"The frailty component is {sig_txt} at the "
                                       f"{alpha_frailty:g} level."]))

    # ------------------------------------------------------------------ Parameter estimates
    rows = []
    pe = []
    for j in range(p):
        b, se = float(par[j]), math.sqrt(max(cov[j, j], 0.0))
        wald = (b / se) ** 2 if se > 0 else float("nan")
        lo, hi = b - z * se, b + z * se
        pe.append((pnames[j], b, se))
        rows.append([pnames[j], fmt(b), fmt(se), fmt(wald), "1", fmt_p(stats.chi2.sf(wald, 1)),
                     fmt(lo), fmt(hi), fmt(math.exp(min(b, 700))), fmt(math.exp(min(lo, 700))),
                     fmt(math.exp(min(hi, 700)))])
    for f in factors:
        lv, ref_val, ref_name = levels[f]
        rows.append([ref_name, "0" + sup("a"), "", "", "", "", "", "", "1", "", ""])
    foot = ["a. Set to zero because this parameter is the baseline (redundant).",
            "Exp(B) is the acceleration factor: the ratio of survival times (quantiles) for a "
            "one-unit increase of the covariate or the factor level against its baseline."]
    if not expo:
        ls_, se_ls = float(par[p]), math.sqrt(max(cov[p, p], 0.0))
        sig_ = math.exp(ls_)
        rows.append(["(Scale)", fmt(sig_), fmt(sig_ * se_ls), "", "", "",
                     fmt(math.exp(ls_ - z * se_ls)), fmt(math.exp(ls_ + z * se_ls)), "", "",
                     ""])
        if distribution == "weibull":
            rows.append(["(Weibull shape = 1/Scale)", fmt(1 / sig_), fmt(se_ls / sig_), "", "",
                         "", fmt(math.exp(-ls_ - z * se_ls)), fmt(math.exp(-ls_ + z * se_ls)),
                         "", "", ""])
    else:
        rows.append(["(Scale)", "1" + sup("b"), "", "", "", "", "", "", "", "", ""])
        foot.append("b. Fixed at the displayed value (exponential distribution).")
    section.table(Table("Parameter Estimates",
                        ["Parameter", "B", "Std. Error", "Wald Chi-Square", "df", "Sig.",
                         (f"{pct(level)} Wald Confidence Interval", "Lower"),
                         (f"{pct(level)} Wald Confidence Interval", "Upper"), "Exp(B)",
                         (f"{pct(level)} Wald Confidence Interval for Exp(B)", "Lower"),
                         (f"{pct(level)} Wald Confidence Interval for Exp(B)", "Upper")],
                        rows, footnotes=foot))
    if recurrent:
        lt, se_lt = float(par[k]), math.sqrt(max(cov[k, k], 0.0))
        section.table(Table("Frailty Component",
                            ["Distribution", "Variance", "Std. Error",
                             (f"{pct(level)} Confidence Interval", "Lower"),
                             (f"{pct(level)} Confidence Interval", "Upper")],
                            [[FRAILTIES[frailty], fmt(theta), fmt(theta * se_lt),
                              fmt(math.exp(lt - z * se_lt)), fmt(math.exp(lt + z * se_lt))]],
                            footnotes=["The frailty has mean 1; its variance is estimated. "
                                       "The confidence interval is computed on the log "
                                       "scale."]))
    from rsstatchem.core.stats.resample import ModelInfo

    out.models = [ModelInfo([r[0] for r in pe], np.array([r[1] for r in pe]),
                            np.array([r[2] for r in pe]), n=n, table="Parameter Estimates",
                            headers=["Parameter"], est_label="B", bootable=False,
                            note="Bootstrapping is not available for parametric survival "
                                 "models.")]
    out.fit = dict(par=par, cov=cov, ll=ll, ll0=ll0, theta=theta, names=names_all,
                   lr_frailty=lr_frailty, data=d, n=n)

    # ------------------------------------------------------------------ plots
    tmax = float(np.nanmax(np.where(d.kind == INTERVAL, d.t_end, np.exp(d.y1))))
    if cox_snell:
        lf, lS, *_ = _err(distribution)
        s_ = 1.0 if expo else math.exp(par[p])
        zz = (d.y1 - d.X @ par[:p]) / s_
        r = -lS(zz)
        ev = (d.kind == EXACT).astype(float)
        section.image("Cox-Snell Residual Plot", _cox_snell_plot(r, ev))
    if function_plots:
        tgrid = np.linspace(tmax / max(n_points, 2) / 10, tmax, max(n_points, 2))
        if recurrent:
            # means over subjects (one profile per subject), as SPSS evaluates the plots
            base = pd.DataFrame(d.X).groupby(d.groups).mean().to_numpy().mean(axis=0)
        else:
            base = d.X.mean(axis=0)
        base = base.copy()
        cond = []
        for c_ in covariates:
            nm = ds.var(c_).name
            j = pnames.index(nm)
            if plot_values and plot_values.get(c_) is not None:
                base[j] = float(plot_values[c_])
            cond.append(f"{nm} = {fmt(base[j], 2)}")
        lines = []
        if separate and separate in factors:
            lv, ref_val, ref_name = levels[separate]
            var = ds.var(separate)
            fcols = [j for j, nm in enumerate(pnames) if nm.startswith(f"[{var.name}=")]
            for v in lv[:max_lines]:
                row = base.copy()
                row[fcols] = 0
                nm = f"[{var.name}={value_text(var, v)}]"
                if nm in pnames:
                    row[pnames.index(nm)] = 1
                lines.append((f"{var.name} = {value_text(var, v)}", row))
        else:
            for f in factors:
                if plot_values and plot_values.get(f) is not None:
                    var = ds.var(f)
                    fcols = [j for j, nm in enumerate(pnames) if nm.startswith(f"[{var.name}=")]
                    base[fcols] = 0
                    nm = f"[{var.name}={value_text(var, plot_values[f])}]"
                    if nm in pnames:
                        base[pnames.index(nm)] = 1
            lines.append(("", base))
        cond_text = "; ".join(cond + (["factors at category frequencies"]
                                      if factors and not separate else []))
        xlabel = (ds.var(end).name if two else ds.var(time).name)
        for kind in function_plots:
            curves = []
            for label, row in lines:
                S, haz, dens = _curves(par, row, distribution, tgrid)
                if recurrent:
                    S, haz, dens = _uncond(S, haz, theta, frailty)
                vals = {"survival": S, "hazard": haz, "density": dens}[kind]
                curves.append((label or "Pattern 1", tgrid, vals))
            ptitle = ("Unconditional " if recurrent else "") + \
                {"survival": "Survival Function", "hazard": "Hazard Function",
                 "density": "Density Function"}[kind]
            section.image(ptitle, _plot_functions(kind, curves, ptitle, xlabel, cond_text))

    # ------------------------------------------------------------------ predict / save
    if predict:
        out.saved = _predict(predict, ds, frame, d, par, distribution, p, theta, frailty,
                             predict_times, interval, periods, duration, past_time,
                             recurrent, ng)
    return out


def _predict(keys, ds, frame, d, par, dist, p, theta, frailty, mode, interval, periods,
             duration, past_time, recurrent, ng) -> dict:
    s = 1.0 if dist == "exponential" else math.exp(par[p])
    eta = d.X @ par[:p]
    lf, lS, *_ = _err(dist)
    if mode == "regular" and interval:
        tsets = [(f"_{j}", np.full(len(eta), interval * j), f" at t = {fmt(interval * j, 4)}")
                 for j in range(1, max(int(periods), 2) + 1)]
    elif mode == "duration" and duration:
        tsets = [("", frame[duration].to_numpy(float), f" at {ds.var(duration).name}")]
    else:
        tsets = [("", d.t_end.astype(float), "")]
    post = None
    if recurrent and theta:
        # empirical Bayes frailty of each subject (posterior mean)
        logh, H = _hazards(par, d, dist)
        ev = (d.kind == EXACT).astype(float)
        Di = np.bincount(d.groups, ev, ng)
        Hi = np.bincount(d.groups, H, ng)
        post_i = np.exp(_lt_logderiv(frailty, theta, Hi, Di + 1)
                        - _lt_logderiv(frailty, theta, Hi, Di))
        post = post_i[d.groups]
    out = {}

    def put(base, label, arr):
        ser = pd.Series(np.nan, index=ds.df.index)
        ser.loc[frame.index] = arr
        out[base] = (label, ser)

    for suffix, t, tlabel in tsets:
        with np.errstate(divide="ignore", invalid="ignore"):
            zz = (np.log(t) - eta) / s
            Hc = -lS(zz)
            hc = np.exp(lf(zz) - lS(zz) - math.log(s) - np.log(t))
        mult = post if post is not None else 1.0
        if "survival" in keys:
            put("PredSurvival" + suffix, "Predicted survival" + tlabel, np.exp(-mult * Hc))
        if "hazard" in keys:
            put("PredHazard" + suffix, "Predicted hazard" + tlabel, mult * hc)
        if "cumhazard" in keys:
            put("PredCumHazard" + suffix, "Predicted cumulative hazard" + tlabel, mult * Hc)
        if "conditional" in keys and past_time:
            t0 = frame[past_time].to_numpy(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                H0 = -lS((np.log(t0) - eta) / s)
            put("PredConditionalSurvival" + suffix,
                f"Predicted survival{tlabel} given survival to {ds.var(past_time).name}",
                np.exp(-mult * (Hc - H0)))
        if recurrent and theta:
            S_ = np.exp(-Hc)
            Su, hu, _ = _uncond(S_, hc, theta, frailty)
            if "uncond_survival" in keys:
                put("PredUnCondSurvival" + suffix, "Predicted unconditional survival" + tlabel,
                    Su)
            if "uncond_hazard" in keys:
                put("PredUncondHazard" + suffix, "Predicted unconditional hazard" + tlabel, hu)
            if "uncond_cumhazard" in keys:
                put("PredUncondCumHazard" + suffix,
                    "Predicted unconditional cumulative hazard" + tlabel,
                    -np.log(np.clip(Su, 1e-300, None)))
    return out


def _syntax(time, start, end, status, covariates, factors, truncation, status_map, dist,
            level, subject, frailty, reference, cox_snell, plots, separate, predict,
            interval_var) -> str:
    cmd = "SURVREG RECURRENT" if subject else "SURVREG AFT"
    tv = f"{start} {end}" if start and end else (time or "")
    log = f"{cmd} {tv}"
    if covariates:
        log += f" WITH {' '.join(covariates)}"
    if factors:
        log += " BY " + " ".join(
            f"{f}({reference[f]})" if reference and reference.get(f) not in (None, "last")
            else f for f in factors)
    dn = {"weibull": "WEIBULL", "exponential": "EXPONENTIAL", "lognormal": "LOG_NORMAL",
          "loglogistic": "LOG_LOGISTIC"}[dist]
    model = f"\n  /MODEL DISTRIBUTION={dn}"
    if subject:
        model = (f"\n  /MODEL SUBJECT={subject} FRAILTY="
                 f"{'GAMMA' if frailty == 'gamma' else 'INV_GAUSSIAN'} DISTRIBUTION={dn}")
        if interval_var:
            model += f" INTERVAL={interval_var}"
    log += model
    if truncation and not subject:
        log += f" LEFTTRUNCATION={truncation}"
    if status:
        parts = [f"VARIABLE={status}"]
        for key, kw in (("event", "FAILURE"), ("right", "RIGHT"), ("left", "LEFT"),
                        ("interval", "INTERVAL")):
            vals = (status_map or {}).get(key)
            if vals:
                parts.append(f"{kw}={' '.join(str(v) for v in vals)}")
        log += "\n  /STATUS " + " ".join(parts)
    log += f"\n  /CRITERIA CILEVEL={level * 100:g}"
    if cox_snell:
        log += "\n  /COXSNELLPLOT"
    if plots:
        log += "\n  /FUNCTIONPLOT " + " ".join(p.upper() for p in plots)
        if separate:
            log += f" PLOTBY({separate})"
    if predict:
        log += "\n  /PREDICT " + " ".join(k.upper().replace("_", "") for k in predict)
    return log + "."


def selftest() -> None:
    rng = np.random.default_rng(4)
    n = 120
    x = rng.normal(size=n)
    g = rng.integers(0, 2, n).astype(float)
    t = np.exp(2 + 0.5 * x - 0.4 * g + 0.6 * np.log(rng.exponential(size=n)))
    c = rng.exponential(np.exp(2.6), n)
    ds = Dataset(pd.DataFrame({"t": np.minimum(t, c), "d": (t <= c).astype(float), "x": x,
                               "g": g, "id": np.repeat(np.arange(40), 3).astype(float)}))
    r = parametric_survival(ds, "t", "d", ["x"], ["g"], distribution="weibull",
                            function_plots=("survival",), cox_snell=True,
                            predict=("survival",))
    assert r.fit is not None and abs(r.fit["par"][1] - 0.5) < 0.3 and r.saved
    r2 = parametric_survival(ds, "t", "d", ["x"], [], subject="id", frailty="gamma",
                             distribution="weibull")
    assert r2.fit["theta"] is not None

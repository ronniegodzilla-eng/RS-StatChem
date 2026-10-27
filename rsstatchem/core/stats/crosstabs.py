"""Analyze > Descriptive Statistics > Crosstabs (SPSS CROSSTABS).

Rows x columns tables with optional layers, the SPSS cell displays (counts, expected counts,
percentages, unstandardized / standardized / adjusted standardized residuals, column-proportion
z-tests with Bonferroni subscript letters) and every statistic of the Statistics sub-dialog:
chi-square tests (Pearson, continuity correction, likelihood ratio, Fisher's exact,
linear-by-linear, McNemar / McNemar-Bowker), correlations, phi and Cramér's V, contingency
coefficient, lambda, Goodman and Kruskal tau, uncertainty coefficient, gamma, Somers' d,
Kendall's tau-b and tau-c, eta, kappa, risk (odds ratio, relative risk) and the Cochran /
Mantel-Haenszel statistics (Breslow-Day, Tarone, common odds ratio).

Asymptotic standard errors follow the SPSS CROSSTABS algorithms: ASE1 (not assuming the null
hypothesis) is the delta-method standard error, and the approximate T uses ASE0 (the standard
error under the null hypothesis). :func:`exact_tests` provides exact (full enumeration of the
tables with the observed margins) and Monte Carlo p values for the chi-square family, ready for
an Exact... sub-dialog.
"""
from __future__ import annotations

import io
import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from rsstatchem.core.dataset import NUMERIC, Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, fmt_p, value_text

CELLS = ("observed", "expected", "row", "column", "total", "resid", "sresid", "asresid")
STATISTICS = ("chisq", "corr", "phi", "cc", "lambda", "uc", "gamma", "d", "btau", "ctau",
              "kappa", "eta", "risk", "mcnemar", "cmh")
_CELL_KEYWORDS = {"observed": "COUNT", "expected": "EXPECTED", "row": "ROW",
                  "column": "COLUMN", "total": "TOTAL", "resid": "RESID", "sresid": "SRESID",
                  "asresid": "ASRESID"}
_STAT_KEYWORDS = {"chisq": "CHISQ", "corr": "CORR", "phi": "PHI", "cc": "CC",
                  "lambda": "LAMBDA", "uc": "UC", "gamma": "GAMMA", "d": "D", "btau": "BTAU",
                  "ctau": "CTAU", "kappa": "KAPPA", "eta": "ETA", "risk": "RISK",
                  "mcnemar": "MCNEMAR", "cmh": "CMH"}


# ============================================================================ measures (pure)
@dataclass
class Measure:
    value: float
    ase: float = float("nan")      # ASE1, not assuming the null hypothesis
    t: float = float("nan")        # approximate T (value / ASE0)
    p: float = float("nan")        # approximate significance
    note: str = ""                 # footnote key: "chisq" | "lr" | "zero" | "normal"


def concordance(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """C_ij / D_ij: counts of cells concordant / discordant with cell (i, j)."""
    n = np.asarray(n, float)
    r, c = n.shape
    pad = np.zeros((r + 1, c + 1))
    pad[1:, 1:] = n.cumsum(0).cumsum(1)          # pad[i, j] = sum over k < i, l < j
    total = pad[r, c]
    upper_left = pad[:r, :c]
    lower_right = total - pad[1:, c][:, None] - pad[r, 1:][None, :] + pad[1:, 1:]
    upper_right = pad[:r, c][:, None] - pad[:r, 1:]
    lower_left = pad[r, :c][None, :] - pad[1:, :c]
    return upper_left + lower_right, upper_right + lower_left


def delta_ase(f, n: np.ndarray, h: float = 1e-6) -> float:
    """Delta-method (multinomial) standard error of a statistic f(table of proportions)."""
    n = np.asarray(n, float)
    total = n.sum()
    p = (n / total).ravel()
    grad = np.zeros_like(p)
    for k in np.flatnonzero(p > 0):
        up, down = p.copy(), p.copy()
        step = min(h, p[k] / 2)
        up[k] += step
        down[k] -= step
        grad[k] = (f(up.reshape(n.shape)) - f(down.reshape(n.shape))) / (2 * step)
    var = (p * grad ** 2).sum() - (p * grad).sum() ** 2
    return math.sqrt(max(var, 0.0) / total)


def _entropies(n: np.ndarray) -> tuple[float, float, float]:
    p = n / n.sum()
    r, c = p.sum(1), p.sum(0)

    def h(v):
        v = v[v > 0]
        return float(-(v * np.log(v)).sum())

    return h(r), h(c), h(p.ravel())


def _lambda_value(n: np.ndarray, kind: str) -> float:
    total = n.sum()
    fim, fmj = n.max(1).sum(), n.max(0).sum()
    cm, rm = n.sum(0).max(), n.sum(1).max()
    if kind == "C":
        return (fim - cm) / (total - cm) if total > cm else float("nan")
    if kind == "R":
        return (fmj - rm) / (total - rm) if total > rm else float("nan")
    den = 2 * total - rm - cm
    return (fim + fmj - cm - rm) / den if den > 0 else float("nan")


def _gk_tau(n: np.ndarray, kind: str) -> float:
    if kind == "R":
        n = n.T
    total = n.sum()
    r, c = n.sum(1), n.sum(0)
    den = total ** 2 - (c ** 2).sum()
    if den <= 0:
        return float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        inner = np.where(r[:, None] > 0, n ** 2 / r[:, None], 0.0).sum()
    return (total * inner - (c ** 2).sum()) / den


def _uncertainty(n: np.ndarray, kind: str) -> float:
    hr, hc, hrc = _entropies(n)
    info = hr + hc - hrc
    if kind == "C":
        return info / hc if hc > 0 else float("nan")
    if kind == "R":
        return info / hr if hr > 0 else float("nan")
    return 2 * info / (hr + hc) if hr + hc > 0 else float("nan")


def _weighted_corr(n: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
    total = n.sum()
    r, c = n.sum(1), n.sum(0)
    mu, mv = (r * u).sum() / total, (c * v).sum() / total
    sxy = (n * np.outer(u - mu, v - mv)).sum()
    sxx = (r * (u - mu) ** 2).sum()
    syy = (c * (v - mv) ** 2).sum()
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def _ridits(margin: np.ndarray) -> np.ndarray:
    return np.cumsum(margin) - margin / 2.0     # midranks up to a constant shift


def _spearman(n: np.ndarray) -> float:
    return _weighted_corr(n, _ridits(n.sum(1)), _ridits(n.sum(0)))


def _t_corr(r: float, total: float) -> tuple[float, float]:
    if not np.isfinite(r) or abs(r) >= 1 or total <= 2:
        return float("nan"), float("nan")
    t = r * math.sqrt((total - 2) / (1 - r * r))
    return t, 2 * stats.t.sf(abs(t), total - 2)


def nominal_measures(n: np.ndarray) -> dict[str, Measure]:
    """Phi, Cramér's V, contingency coefficient, lambda, Goodman-Kruskal tau and uncertainty
    coefficient (keys like "lambda_S", "lambda_R", "lambda_C")."""
    n = np.asarray(n, float)
    total = n.sum()
    rr, cc = n.shape
    out: dict[str, Measure] = {}
    chi, p_chi = pearson_chi2(n)
    k = min(rr, cc)
    phi = math.sqrt(chi / total)
    if n.shape == (2, 2):
        a, b, c, d = n.ravel()
        den = math.sqrt(n.sum(1).prod() * n.sum(0).prod())
        phi = (a * d - b * c) / den if den else float("nan")
    out["phi"] = Measure(phi, p=p_chi)
    out["v"] = Measure(math.sqrt(chi / (total * (k - 1))) if k > 1 else float("nan"), p=p_chi)
    out["cc"] = Measure(math.sqrt(chi / (chi + total)), p=p_chi)

    # lambda (ASE1 as in SPSS; approximate T from ASE0)
    fim_idx = n.argmax(1)
    fmj_idx = n.argmax(0)
    cm_idx, rm_idx = int(n.sum(0).argmax()), int(n.sum(1).argmax())
    for kind in ("S", "R", "C"):
        value = _lambda_value(n, kind)
        if kind == "C":
            psi = np.zeros_like(n)
            psi[np.arange(rr), fim_idx] += 1
            psi[:, cm_idx] -= 1
            den = total - n.sum(0).max()
        elif kind == "R":
            psi = np.zeros_like(n)
            psi[fmj_idx, np.arange(cc)] += 1
            psi[rm_idx, :] -= 1
            den = total - n.sum(1).max()
        else:
            psi = np.zeros_like(n)
            psi[np.arange(rr), fim_idx] += 1
            psi[fmj_idx, np.arange(cc)] += 1
            psi[:, cm_idx] -= 1
            psi[rm_idx, :] -= 1
            den = 2 * total - n.sum(0).max() - n.sum(1).max()
        if kind in ("R", "C"):
            fsum = n.max(1).sum() if kind == "C" else n.max(0).sum()
            m = n.sum(0).max() if kind == "C" else n.sum(1).max()
            if kind == "C":
                same = n[np.arange(rr), fim_idx][fim_idx == cm_idx].sum()
            else:
                same = n[fmj_idx, np.arange(cc)][fmj_idx == rm_idx].sum()
            var1 = (total - fsum) * (fsum + m - 2 * same) / (total - m) ** 3 \
                if total > m else float("nan")
            ase1 = math.sqrt(max(var1, 0.0)) if np.isfinite(var1) else float("nan")
        else:
            ase1 = delta_ase(lambda p: _lambda_value(p, "S"), n)
        var0 = (n * psi ** 2).sum() - (n * psi).sum() ** 2 / total
        ase0 = math.sqrt(max(var0, 0.0)) / den if den > 0 else float("nan")
        meas = Measure(value, ase1)
        if ase0 and np.isfinite(ase0) and ase0 > 0:
            meas.t = value / ase0
            meas.p = 2 * stats.norm.sf(abs(meas.t))
        else:
            meas.note = "zero"
        out[f"lambda_{kind}"] = meas

    # Goodman and Kruskal tau: significance from (N-1)(J-1) tau ~ chi2((I-1)(J-1))
    for kind in ("R", "C"):
        value = _gk_tau(n, kind)
        levels = rr if kind == "R" else cc
        stat = (total - 1) * (levels - 1) * value
        out[f"tau_{kind}"] = Measure(value, delta_ase(lambda p, k=kind: _gk_tau(p, k), n),
                                     p=stats.chi2.sf(stat, (rr - 1) * (cc - 1)), note="chisq")

    # uncertainty coefficient: T from the null-variance of the mutual information
    hr, hc, hrc = _entropies(n)
    info = hr + hc - hrc
    p = n / total
    rp, cp = p.sum(1), p.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(p > 0, np.log(p / np.outer(rp, cp)), 0.0)
    var_info = max(((p * g ** 2).sum() - info ** 2) / total, 0.0)
    lr, lr_p = likelihood_ratio(n)
    for kind, den in (("S", (hr + hc) / 2), ("R", hr), ("C", hc)):
        value = _uncertainty(n, kind)
        ase1 = delta_ase(lambda q, k=kind: _uncertainty(q, k), n)
        ase0 = math.sqrt(var_info) / den if den > 0 else float("nan")
        meas = Measure(value, ase1, p=lr_p, note="lr")
        if ase0 and np.isfinite(ase0):
            meas.t = value / ase0
        out[f"uc_{kind}"] = meas
    return out


def ordinal_measures(n: np.ndarray) -> dict[str, Measure]:
    """Gamma, Somers' d (S/R/C), Kendall's tau-b and tau-c."""
    n = np.asarray(n, float)
    total = n.sum()
    rr, cc = n.shape
    r, c = n.sum(1), n.sum(0)
    C, D = concordance(n)
    P, Q = (n * C).sum(), (n * D).sum()
    dr, dc = total ** 2 - (r ** 2).sum(), total ** 2 - (c ** 2).sum()
    s0 = (n * (C - D) ** 2).sum() - (P - Q) ** 2 / total
    s0 = math.sqrt(max(s0, 0.0))
    t = (P - Q) / (2 * s0) if s0 > 0 else float("nan")
    p_t = 2 * stats.norm.sf(abs(t)) if np.isfinite(t) else float("nan")
    out: dict[str, Measure] = {}

    def make(value, ase):
        m = Measure(value, ase, t, p_t)
        if not np.isfinite(t):
            m.note = "zero"
        return m

    if P + Q > 0:
        out["gamma"] = make((P - Q) / (P + Q),
                            4 / (P + Q) ** 2 * math.sqrt((n * (Q * C - P * D) ** 2).sum()))
    else:
        out["gamma"] = Measure(float("nan"), note="zero")
    tau_b = (P - Q) / math.sqrt(dr * dc) if dr > 0 and dc > 0 else float("nan")
    if np.isfinite(tau_b):
        v = np.outer(r, np.ones(cc)) * dc + np.outer(np.ones(rr), c) * dr
        inner = (n * (2 * math.sqrt(dr * dc) * (C - D) + tau_b * v) ** 2).sum() \
            - total ** 3 * tau_b ** 2 * (dr + dc) ** 2
        out["btau"] = make(tau_b, math.sqrt(max(inner, 0.0)) / (dr * dc))
    else:
        out["btau"] = Measure(float("nan"), note="zero")
    m = min(rr, cc)
    if m > 1:
        out["ctau"] = make(m * (P - Q) / (total ** 2 * (m - 1)),
                           2 * m / ((m - 1) * total ** 2) * s0)
    rows_i = np.repeat(r[:, None], cc, 1)
    cols_j = np.repeat(c[None, :], rr, 0)
    if dr + dc > 0:
        ds = 2 * (P - Q) / (dr + dc)
        ase = 4 / (dr + dc) ** 2 * math.sqrt(
            (n * ((dr + dc) * (C - D) - (P - Q) * (2 * total - rows_i - cols_j)) ** 2).sum())
        out["d_S"] = make(ds, ase)
    if dc > 0:   # row variable dependent
        out["d_R"] = make((P - Q) / dc, 2 / dc ** 2 * math.sqrt(
            (n * (dc * (C - D) - (P - Q) * (total - cols_j)) ** 2).sum()))
    if dr > 0:   # column variable dependent
        out["d_C"] = make((P - Q) / dr, 2 / dr ** 2 * math.sqrt(
            (n * (dr * (C - D) - (P - Q) * (total - rows_i)) ** 2).sum()))
    return out


def correlations(n: np.ndarray, row_scores, col_scores) -> dict[str, Measure]:
    """Pearson's R on the category values and Spearman correlation on midranks."""
    n = np.asarray(n, float)
    total = n.sum()
    u, v = np.asarray(row_scores, float), np.asarray(col_scores, float)
    r = _weighted_corr(n, u, v)
    rs = _spearman(n)
    t_r, p_r = _t_corr(r, total)
    t_s, p_s = _t_corr(rs, total)
    return {"pearson": Measure(r, delta_ase(lambda p: _weighted_corr(p, u, v), n), t_r, p_r,
                               "normal"),
            "spearman": Measure(rs, delta_ase(_spearman, n), t_s, p_s, "normal")}


def eta(n: np.ndarray, row_scores, col_scores) -> dict[str, float]:
    """Eta with the row (interval) variable dependent and with the column variable dependent."""
    n = np.asarray(n, float)

    def one(table, y):
        total = table.sum()
        r = table.sum(1)
        mean = (r * y).sum() / total
        ss_total = (r * (y - mean) ** 2).sum()
        col_n = table.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            col_mean = np.where(col_n > 0, (table * y[:, None]).sum(0) / col_n, 0.0)
        ss_within = (table * (y[:, None] - col_mean[None, :]) ** 2).sum()
        return math.sqrt(max(1 - ss_within / ss_total, 0.0)) if ss_total > 0 else float("nan")

    return {"R": one(n, np.asarray(row_scores, float)),
            "C": one(n.T, np.asarray(col_scores, float))}


def pearson_chi2(n: np.ndarray) -> tuple[float, float]:
    n = np.asarray(n, float)
    e = np.outer(n.sum(1), n.sum(0)) / n.sum()
    chi = float(((n - e) ** 2 / np.where(e > 0, e, 1)).sum())
    dof = (n.shape[0] - 1) * (n.shape[1] - 1)
    return chi, float(stats.chi2.sf(chi, dof)) if dof > 0 else float("nan")


def likelihood_ratio(n: np.ndarray) -> tuple[float, float]:
    n = np.asarray(n, float)
    e = np.outer(n.sum(1), n.sum(0)) / n.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        g = 2 * float(np.where(n > 0, n * np.log(n / e), 0.0).sum())
    dof = (n.shape[0] - 1) * (n.shape[1] - 1)
    return g, float(stats.chi2.sf(g, dof)) if dof > 0 else float("nan")


def kappa(n: np.ndarray) -> Measure:
    """Cohen's kappa with ASE1 (Fleiss, Cohen & Everitt 1969) and T = kappa / ASE0."""
    n = np.asarray(n, float)
    total = n.sum()
    p = n / total
    r, c = p.sum(1), p.sum(0)
    po, pe = np.trace(p), (r * c).sum()
    if pe >= 1:
        return Measure(float("nan"), note="zero")
    k = (po - pe) / (1 - pe)
    diag = np.diag(p)
    term1 = (diag * (1 - (r + c) * (1 - k)) ** 2).sum()
    off = p * (1 - k) ** 2 * np.add.outer(c, r) ** 2
    np.fill_diagonal(off, 0.0)
    term3 = (k - pe * (1 - k)) ** 2
    var1 = (term1 + off.sum() - term3) / ((1 - pe) ** 2 * total)
    var0 = (pe + pe ** 2 - (r * c * (r + c)).sum()) / ((1 - pe) ** 2 * total)
    ase1 = math.sqrt(max(var1, 0.0))
    ase0 = math.sqrt(max(var0, 0.0))
    t = k / ase0 if ase0 > 0 else float("nan")
    return Measure(k, ase1, t, 2 * stats.norm.sf(abs(t)) if np.isfinite(t) else float("nan"))


def risk(n: np.ndarray, level: float = 0.95) -> dict[str, tuple[float, float, float]]:
    """Odds ratio and the two cohort relative risks of a 2x2 table (log-scale Wald CIs)."""
    a, b, c, d = np.asarray(n, float).ravel()
    z = stats.norm.ppf(0.5 + level / 2)
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        if min(a, b, c, d) > 0:
            orr = a * d / (b * c)
            se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
            out["or"] = (orr, orr * math.exp(-z * se), orr * math.exp(z * se))
        for key, (x1, n1, x2, n2) in (("rr1", (a, a + b, c, c + d)),
                                     ("rr2", (b, a + b, d, c + d))):
            if x1 > 0 and x2 > 0 and n1 > 0 and n2 > 0:
                rr = (x1 / n1) / (x2 / n2)
                se = math.sqrt(1 / x1 - 1 / n1 + 1 / x2 - 1 / n2)
                out[key] = (rr, rr * math.exp(-z * se), rr * math.exp(z * se))
    return out


def mcnemar(n: np.ndarray) -> dict:
    """McNemar (2x2: exact binomial as SPSS) or McNemar-Bowker symmetry test."""
    n = np.asarray(n, float)
    if n.shape == (2, 2):
        b, c = n[0, 1], n[1, 0]
        p = stats.binomtest(int(min(b, c)), int(b + c), 0.5).pvalue if b + c > 0 else 1.0
        return {"kind": "mcnemar", "p": float(p)}
    k = n.shape[0]
    stat, dof = 0.0, 0
    for i in range(k):
        for j in range(i + 1, k):
            s = n[i, j] + n[j, i]
            if s > 0:
                stat += (n[i, j] - n[j, i]) ** 2 / s
                dof += 1
    return {"kind": "bowker", "value": stat, "df": dof,
            "p": float(stats.chi2.sf(stat, dof)) if dof else float("nan")}


def cmh(tables: list[np.ndarray], test_value: float = 1.0, level: float = 0.95) -> dict:
    """Breslow-Day / Tarone homogeneity, Cochran and Mantel-Haenszel conditional independence
    tests and the Mantel-Haenszel common odds ratio (Robins-Breslow-Greenland SE)."""
    from statsmodels.stats.contingency_tables import StratifiedTable

    tabs = [np.asarray(t, float) for t in tables if np.asarray(t).sum() > 0]
    out: dict = {"strata": len(tabs)}
    a = np.array([t[0, 0] for t in tabs])
    n1 = np.array([t[0].sum() for t in tabs])
    n2 = np.array([t[1].sum() for t in tabs])
    m1 = np.array([t[:, 0].sum() for t in tabs])
    m2 = np.array([t[:, 1].sum() for t in tabs])
    tot = n1 + n2
    expected = n1 * m1 / tot
    diff = (a - expected).sum()
    var_c = (n1 * n2 * m1 * m2 / tot ** 3).sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        var_mh = np.where(tot > 1, n1 * n2 * m1 * m2 / (tot ** 2 * (tot - 1)), 0.0).sum()
    cochran = diff ** 2 / var_c if var_c > 0 else float("nan")
    cc = 0.5 if abs(diff) > 0 else 0.0
    mh = (abs(diff) - cc) ** 2 / var_mh if var_mh > 0 else float("nan")
    out["cochran"] = (cochran, float(stats.chi2.sf(cochran, 1)))
    out["mh"] = (mh, float(stats.chi2.sf(mh, 1)))
    try:
        st = StratifiedTable(tabs)
        if len(tabs) > 1:
            bd = st.test_equal_odds(adjust=False)
            ta = st.test_equal_odds(adjust=True)
            out["breslow"] = (float(bd.statistic), len(tabs) - 1, float(bd.pvalue))
            out["tarone"] = (float(ta.statistic), len(tabs) - 1, float(ta.pvalue))
        est = float(st.oddsratio_pooled)
        se = float(st.logodds_pooled_se)
        z = stats.norm.ppf(0.5 + level / 2)
        ln = math.log(est)
        p = 2 * stats.norm.sf(abs((ln - math.log(test_value)) / se))
        out["or"] = {"estimate": est, "ln": ln, "se": se, "p": float(p),
                     "lo": math.exp(ln - z * se), "hi": math.exp(ln + z * se),
                     "ln_lo": ln - z * se, "ln_hi": ln + z * se}
    except (ValueError, ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError):
        pass
    return out


# ============================================================================ column z-tests
def column_proportion_letters(n: np.ndarray, alpha: float = 0.05,
                              bonferroni: bool = True) -> tuple[list[list[str]], set]:
    """SPSS 'Compare column proportions': per row, pairwise two-proportion z-tests between the
    columns; columns that share a letter do not differ. Returns letters[i][j] and the set of
    (i, j) cells excluded because the column proportion is 0 or 1."""
    n = np.asarray(n, float)
    rr, cc = n.shape
    col = n.sum(0)
    letters = [[""] * cc for _ in range(rr)]
    excluded = set()
    for i in range(rr):
        props = np.where(col > 0, n[i] / np.where(col > 0, col, 1), np.nan)
        usable = [j for j in range(cc) if col[j] > 0 and 0 < props[j] < 1]
        excluded |= {(i, j) for j in range(cc) if j not in usable and col[j] > 0}
        pairs = list(itertools.combinations(usable, 2))
        significant = []
        for j, k in pairs:
            pooled = (n[i, j] + n[i, k]) / (col[j] + col[k])
            se = math.sqrt(pooled * (1 - pooled) * (1 / col[j] + 1 / col[k]))
            if se == 0:
                continue
            p = 2 * stats.norm.sf(abs(props[j] - props[k]) / se)
            if bonferroni:
                p = min(1.0, p * len(pairs))
            if p < alpha:
                significant.append((j, k))
        for j, word in _letter_groups(usable, significant).items():
            letters[i][j] = word
    return letters, excluded


def _letter_groups(items: list[int], significant: list[tuple[int, int]]) -> dict[int, str]:
    """Piepho's (2004) insert-and-absorb algorithm for compact letter displays."""
    if not items:
        return {}
    columns = [set(items)]
    for a, b in significant:
        new = []
        for col in columns:
            if a in col and b in col:
                new += [col - {a}, col - {b}]
            else:
                new.append(col)
        columns = [c for c in new if c and not any(c < o for o in new)]
        unique = []
        for c in columns:
            if c not in unique:
                unique.append(c)
        columns = unique
    columns.sort(key=lambda c: (min(items.index(x) for x in c), len(c)))
    out = {}
    for idx, col in enumerate(columns):
        for x in col:
            out[x] = out.get(x, "") + _letter(idx)
    return out


def _letter(i: int) -> str:
    return chr(ord("a") + i) if i < 26 else f"a{i}"


# ============================================================================ exact / Monte Carlo
class TooManyTables(Exception):
    pass


def _log_fact(x):
    from scipy.special import gammaln
    return gammaln(np.asarray(x, float) + 1)


def table_statistics(t: np.ndarray, row_scores=None, col_scores=None) -> dict[str, np.ndarray]:
    """Pearson chi-square, likelihood ratio, Fisher-Freeman-Halton (-2 ln(gamma P)) and the
    linear-by-linear statistic sum(u_i v_j n_ij) for one table or a stack of tables (..., r, c)."""
    t = np.asarray(t, float)
    r, c = t.sum(-1), t.sum(-2)
    total = r.sum(-1)
    e = r[..., :, None] * c[..., None, :] / total[..., None, None]
    pearson = ((t - e) ** 2 / e).sum((-1, -2))
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = 2 * np.where(t > 0, t * np.log(t / e), 0.0).sum((-1, -2))
    rr, cc = t.shape[-2:]
    log_p = (_log_fact(r).sum(-1) + _log_fact(c).sum(-1) - _log_fact(total)
             - _log_fact(t).sum((-1, -2)))
    log_gamma = ((rr - 1) * (cc - 1) / 2 * math.log(2 * math.pi)
                 - (rr * cc - 1) / 2 * np.log(total)
                 + (cc - 1) / 2 * np.log(r).sum(-1) + (rr - 1) / 2 * np.log(c).sum(-1))
    u = np.arange(rr, dtype=float) if row_scores is None else np.asarray(row_scores, float)
    v = np.arange(cc, dtype=float) if col_scores is None else np.asarray(col_scores, float)
    lbl = (t * np.outer(u, v)).sum((-1, -2))
    return {"pearson": pearson, "lr": lr, "fisher": -2 * (log_gamma + log_p),
            "log_p": log_p, "lbl": lbl}


def enumerate_tables(row: np.ndarray, col: np.ndarray, limit: int = 200_000) -> np.ndarray:
    """All r x c tables with the given margins (raises TooManyTables beyond ``limit``)."""
    row = [int(x) for x in row]
    col = [int(x) for x in col]
    rr, cc = len(row), len(col)
    out: list[np.ndarray] = []
    cur = np.zeros((rr, cc), int)

    def fill_column(j: int, remaining_rows: list[int]):
        if j == cc - 1:
            cur[:, j] = remaining_rows
            out.append(cur.copy())
            if len(out) > limit:
                raise TooManyTables
            return
        need = col[j]
        rest_cols = sum(col[j + 1:])

        def place(i: int, left: int):
            if i == rr - 1:
                if left <= remaining_rows[i]:
                    cur[i, j] = left
                    nxt = [remaining_rows[k] - cur[k, j] for k in range(rr)]
                    fill_column(j + 1, nxt)
                return
            cap_after = sum(remaining_rows[i + 1:])
            lo = max(0, left - cap_after)
            hi = min(left, remaining_rows[i])
            for x in range(lo, hi + 1):
                cur[i, j] = x
                place(i + 1, left - x)

        if sum(remaining_rows) - need != rest_cols:
            return
        place(0, need)

    fill_column(0, row[:])
    return np.array(out, float)


def mc_interval(p: float, samples: int, level: float = 0.99) -> tuple[float, float]:
    z = stats.norm.ppf(0.5 + level / 2)
    half = z * math.sqrt(max(p * (1 - p), 0.0) / samples)
    return max(0.0, p - half), min(1.0, p + half)


def exact_tests(obs: np.ndarray, method: str = "exact", samples: int = 10000,
                level: float = 0.99, seed: int = 2000000, limit: int = 200_000,
                row_scores=None, col_scores=None) -> dict[str, dict]:
    """Exact or Monte Carlo significance for the Pearson chi-square, likelihood ratio,
    Fisher's exact (Fisher-Freeman-Halton for r x c) and linear-by-linear tests.

    Returns {test: {"p": two-sided p, "p1": one-sided p (linear-by-linear, 2x2 Fisher),
    "point": point probability (exact), "lower"/"upper": Monte Carlo CI, "method"}}.
    ``method="exact"`` falls back to Monte Carlo when there are more than ``limit`` tables.
    """
    obs = np.asarray(obs, float)
    keep_r, keep_c = obs.sum(1) > 0, obs.sum(0) > 0
    obs = obs[keep_r][:, keep_c]
    row, col = obs.sum(1), obs.sum(0)
    if row_scores is not None:
        row_scores = np.asarray(row_scores, float)[keep_r]
    if col_scores is not None:
        col_scores = np.asarray(col_scores, float)[keep_c]
    observed = table_statistics(obs, row_scores, col_scores)
    total = obs.sum()
    u = np.arange(len(row), dtype=float) if row_scores is None else row_scores
    v = np.arange(len(col), dtype=float) if col_scores is None else np.asarray(col_scores, float)
    mean_lbl = (row * u).sum() * (col * v).sum() / total
    eps = 1e-7
    tables = None
    used = method
    if method == "exact":
        try:
            tables = enumerate_tables(row, col, limit)
        except TooManyTables:
            used = "mc"
    out: dict[str, dict] = {}
    if used == "exact":
        st = table_statistics(tables, u, v)
        prob = np.exp(st["log_p"])
        prob = prob / prob.sum()
        for key in ("pearson", "lr"):
            ge = st[key] >= observed[key] - eps * max(1.0, abs(float(observed[key])))
            point = prob[np.isclose(st[key], observed[key], rtol=1e-9, atol=1e-9)].sum()
            out[key] = {"p": float(min(1.0, prob[ge].sum())), "point": float(point),
                        "method": "exact"}
        le = st["log_p"] <= observed["log_p"] + 1e-7
        out["fisher"] = {"p": float(min(1.0, prob[le].sum())),
                         "point": float(np.exp(observed["log_p"])), "method": "exact"}
        dev = np.abs(st["lbl"] - mean_lbl)
        d_obs = abs(float(observed["lbl"]) - mean_lbl)
        tail = st["lbl"] >= observed["lbl"] - 1e-9 if observed["lbl"] >= mean_lbl \
            else st["lbl"] <= observed["lbl"] + 1e-9
        out["lbl"] = {"p": float(min(1.0, prob[dev >= d_obs - 1e-9].sum())),
                      "p1": float(min(1.0, prob[tail].sum())),
                      "point": float(prob[np.isclose(st["lbl"], observed["lbl"])].sum()),
                      "method": "exact"}
        if obs.shape == (2, 2):
            a = tables[:, 0, 0]
            a_obs = obs[0, 0]
            one = min(prob[a >= a_obs].sum(), prob[a <= a_obs].sum())
            out["fisher"]["p1"] = float(min(1.0, one))
        return out
    rng = np.random.default_rng(seed)
    sims = stats.random_table(row.astype(int), col.astype(int)).rvs(size=samples,
                                                                  random_state=rng)
    st = table_statistics(sims.astype(float), u, v)
    for key in ("pearson", "lr", "fisher"):
        p = float(np.mean(st[key] >= observed[key] - eps * max(1.0, abs(float(observed[key])))))
        lo, hi = mc_interval(p, samples, level)
        out[key] = {"p": p, "lower": lo, "upper": hi, "method": "mc"}
    d_obs = abs(float(observed["lbl"]) - mean_lbl)
    p = float(np.mean(np.abs(st["lbl"] - mean_lbl) >= d_obs - 1e-9))
    tail = st["lbl"] >= observed["lbl"] - 1e-9 if observed["lbl"] >= mean_lbl \
        else st["lbl"] <= observed["lbl"] + 1e-9
    p1 = float(np.mean(tail))
    lo, hi = mc_interval(p, samples, level)
    lo1, hi1 = mc_interval(p1, samples, level)
    out["lbl"] = {"p": p, "lower": lo, "upper": hi, "p1": p1, "lower1": lo1, "upper1": hi1,
                  "method": "mc"}
    return out


# ============================================================================ procedure
def _categories(series: pd.Series) -> list:
    return sorted(series.unique(), key=lambda v: (isinstance(v, str), v))


def _pct(part, whole) -> str:
    return f"{100.0 * part / whole:.1f}%" if whole else ""


def _scores(var, cats) -> np.ndarray:
    if var.type == NUMERIC:
        try:
            return np.array([float(c) for c in cats])
        except (TypeError, ValueError):
            pass
    return np.arange(1, len(cats) + 1, dtype=float)


def crosstabs(ds: Dataset, rows: list[str], columns: list[str], chisq: bool = True,
              phi: bool = True, contingency: bool = False, cells=("observed",), *,
              layers: list[list[str]] | None = None, statistics=(), hide_small: int | None = None,
              z_test: bool = False, bonferroni: bool = True, row_order: str = "ascending",
              suppress_tables: bool = False, clustered_bar: bool = False,
              cmh_value: float = 1.0, exact: str | None = None, mc_samples: int = 10000,
              mc_level: float = 0.99, mc_seed: int = 2000000) -> Section:
    """CROSSTABS. ``statistics`` holds keys of :data:`STATISTICS`; ``chisq``/``phi``/
    ``contingency`` are kept for the older call style. ``layers`` is a list of layer lists
    (each case of the cartesian product of one variable per layer is a table)."""
    stats_set = set(statistics)
    for key, on in (("chisq", chisq), ("phi", phi), ("cc", contingency)):
        if on:
            stats_set.add(key)
    cells = tuple(c for c in CELLS if c in cells) or ("observed",)
    layers = [list(lay) for lay in (layers or []) if lay]
    section = Section("Crosstabs")
    layer_text = "".join(f" BY {' '.join(lay)}" for lay in layers)
    fmt_line = "  /FORMAT=" + ("AVALUE" if row_order == "ascending" else "DVALUE") + \
        (" NOTABLES" if suppress_tables else " TABLES")
    log = (f"CROSSTABS\n  /TABLES={' '.join(rows)} BY {' '.join(columns)}{layer_text}\n"
           f"{fmt_line}\n  /STATISTICS="
           f"{' '.join(_STAT_KEYWORDS[s] for s in STATISTICS if s in stats_set) or 'NONE'}"
           + (f"\n  /CMH({cmh_value:g})" if "cmh" in stats_set else "")
           + f"\n  /CELLS={' '.join(_CELL_KEYWORDS[c] for c in cells)}"
           + (" COLUMN" if z_test and "column" not in cells else "")
           + (" PROP" if z_test else "") + (" BPROP" if z_test and bonferroni else "")
           + (f"\n  /HIDESMALLCOUNTS COUNT={hide_small}" if hide_small else "")
           + ("\n  /BARCHART" if clustered_bar else "")
           + (("\n  /METHOD=EXACT" if exact == "exact" else
               f"\n  /METHOD=MC CIN({mc_level * 100:g}) SAMPLES({mc_samples})")
              if exact else "") + ".")
    section.log(log)
    combos = list(itertools.product(*layers)) if layers else [()]
    opts = dict(stats=stats_set, cells=cells, hide_small=hide_small, z_test=z_test,
                bonferroni=bonferroni, descending=row_order == "descending",
                suppress=suppress_tables, bar=clustered_bar, cmh_value=cmh_value,
                exact=exact, mc=(mc_samples, mc_level, mc_seed))
    for rname in rows:
        for cname in columns:
            for combo in combos:
                _one(ds, section, rname, cname, list(combo), opts)
    return section


def _one(ds: Dataset, section: Section, rname: str, cname: str, lnames: list[str],
         o: dict) -> None:
    rvar, cvar = ds.var(rname), ds.var(cname)
    lvars = [ds.var(n) for n in lnames]
    raw = pd.DataFrame({"r": ds.analysis_series(rname), "c": ds.analysis_series(cname)}
                       | {f"l{k}": ds.analysis_series(n) for k, n in enumerate(lnames)})
    total = len(raw)
    frame = raw.dropna()
    n_valid = len(frame)
    title = " * ".join([rvar.title, cvar.title] + [v.title for v in lvars])
    section.table(Table("Case Processing Summary", [
        "", ("Valid", "N"), ("Valid", "Percent"),
        ("Missing", "N"), ("Missing", "Percent"), ("Total", "N"), ("Total", "Percent")], [[
            title, fmt_int(n_valid), _pct(n_valid, total), fmt_int(total - n_valid),
            _pct(total - n_valid, total), fmt_int(total), "100.0%" if total else ""]]))
    if n_valid == 0:
        return
    strata: list[tuple[list[str], pd.DataFrame]] = []
    if lvars:
        keys = sorted({tuple(t) for t in frame[[f"l{k}" for k in range(len(lvars))]]
                       .itertuples(index=False)},
                      key=lambda t: tuple((isinstance(v, str), v) for v in t))
        for key in keys:
            mask = np.ones(len(frame), bool)
            for k, v in enumerate(key):
                mask &= (frame[f"l{k}"] == v).to_numpy()
            strata.append(([value_text(lv, v) for lv, v in zip(lvars, key)], frame[mask]))
        if len(lvars) == 1:
            strata.append((["Total"], frame))
    else:
        strata.append(([], frame))

    nl = len(lvars)
    labels = {"observed": "Count", "expected": "Expected Count",
              "row": f"% within {rvar.title}", "column": f"% within {cvar.title}",
              "total": "% of Total", "resid": "Residual", "sresid": "Standardized Residual",
              "asresid": "Adjusted Residual"}
    shown = list(o["cells"])
    all_ccats = _categories(frame["c"])
    table_rows: list[list[str]] = []
    results = []
    z_note = excluded_note = False
    for s_idx, (slabels, sub) in enumerate(strata):
        rcats, ccats = _categories(sub["r"]), _categories(sub["c"])
        if o["descending"]:
            rcats = rcats[::-1]
        obs = np.array([[int(((sub["r"] == r) & (sub["c"] == c)).sum()) for c in ccats]
                        for r in rcats], float)
        results.append((slabels, rcats, ccats, obs))
        if o["suppress"]:
            continue
        row_tot, col_tot, n = obs.sum(1), obs.sum(0), obs.sum()
        letters, excluded = (column_proportion_letters(obs, 0.05, o["bonferroni"])
                             if o["z_test"] and len(ccats) > 1 else (None, set()))
        if letters is not None:
            z_note = True
            excluded_note |= bool(excluded)
        letter_row = "observed" if "observed" in shown else (
            "column" if "column" in shown else shown[0])
        for i in range(len(rcats) + 1):
            for k, key in enumerate(shown):
                cells_ = []
                for j in range(len(ccats) + 1):
                    inner = i < len(rcats) and j < len(ccats)
                    o_val = obs[i, j] if inner else (
                        n if i == len(rcats) and j == len(ccats) else
                        col_tot[j] if i == len(rcats) else row_tot[i])
                    rt = row_tot[i] if i < len(rcats) else n
                    ct = col_tot[j] if j < len(ccats) else n
                    e_val = rt * ct / n
                    if o["hide_small"] and inner and 0 < o_val < o["hide_small"]:
                        cells_.append("")
                        continue
                    text = ""
                    if key == "observed":
                        text = fmt_int(o_val)
                    elif key == "expected":
                        text = fmt(e_val, 1)
                    elif key == "row":
                        text = _pct(o_val, rt)
                    elif key == "column":
                        text = _pct(o_val, ct)
                    elif key == "total":
                        text = _pct(o_val, n)
                    elif inner:
                        res = o_val - e_val
                        if key == "resid":
                            text = fmt(res, 1)
                        elif key == "sresid":
                            text = fmt(res / math.sqrt(e_val), 1) if e_val > 0 else ""
                        else:
                            den = e_val * (1 - rt / n) * (1 - ct / n)
                            text = fmt(res / math.sqrt(den), 1) if den > 0 else ""
                    if letters is not None and key == letter_row and inner:
                        mark = letters[i][j]
                        if (i, j) in excluded:
                            mark = "ᵃ"
                        text += (" " + mark if mark and mark != "ᵃ" else mark)
                    cells_.append(text)
                # columns follow the union of categories so layered tables line up
                full = []
                for c in all_ccats:
                    full.append(cells_[ccats.index(c)] if c in ccats else "")
                full.append(cells_[-1])
                first = i == 0 and k == 0
                head = [slabels[m] if first else "" for m in range(nl)]
                if i < len(rcats):
                    head += [rvar.title if first else "",
                             value_text(rvar, rcats[i]) if k == 0 else ""]
                else:
                    head += ["Total" if k == 0 else "", ""]
                table_rows.append(head + [labels[key]] + full)
    if not o["suppress"]:
        cols = [lv.title for lv in lvars] + [""] * 3 + \
            [(cvar.title, value_text(cvar, c)) for c in all_ccats] + \
            ["Total"]
        notes = []
        if z_note:
            notes.append(f"Each subscript letter denotes a subset of {cvar.title} categories "
                         "whose column proportions do not differ significantly from each other "
                         "at the .05 level." + (" (Bonferroni method)" if o["bonferroni"]
                                                else ""))
        if excluded_note:
            notes.append("a. This category is not used in comparisons because its column "
                         "proportion is equal to zero or one.")
        section.table(Table(f"{title} Crosstabulation", cols, table_rows,
                            row_header_cols=nl + 3, footnotes=notes))
    _statistics_tables(section, rvar, cvar, lvars, results, o)
    if o["bar"]:
        for slabels, rcats, ccats, obs in results:
            if slabels and slabels[-1] == "Total" and len(results) > 1:
                continue
            suffix = "" if not slabels else " (" + ", ".join(
                f"{lv.title} = {s}" for lv, s in zip(lvars, slabels)) + ")"
            section.image(f"Bar Chart{suffix}", clustered_bar_chart(
                [value_text(rvar, r) for r in rcats], [value_text(cvar, c) for c in ccats], obs,
                rvar.title, cvar.title, f"Bar Chart{suffix}"))


def _statistics_tables(section, rvar, cvar, lvars, results, o) -> None:
    s = o["stats"]
    nl = len(lvars)
    lead = [lv.title for lv in lvars]

    def prefix(slabels, first):
        return [slabels[m] if first else "" for m in range(nl)]

    exact = o["exact"]
    if "chisq" in s or "mcnemar" in s:
        crow, notes, late = [], [], []
        exact_cols = []
        if exact == "exact":
            exact_cols = ["Exact Sig. (2-sided)", "Exact Sig. (1-sided)", "Point Probability"]
        elif exact == "mc":
            lvl = f"{o['mc'][1] * 100:g}%"
            exact_cols = [("Monte Carlo Sig. (2-sided)", "Significance"),
                          ("Monte Carlo Sig. (2-sided)", f"{lvl} Confidence Interval: Lower "
                                                         "Bound"),
                          ("Monte Carlo Sig. (2-sided)", f"{lvl} Confidence Interval: Upper "
                                                         "Bound"),
                          ("Monte Carlo Sig. (1-sided)", "Significance")]
        width = 3 + (len(exact_cols) if exact else 2)
        any_two = False
        for slabels, rcats, ccats, obs in results:
            pre = prefix(slabels, True)
            if min(obs.shape) < 2:
                crow.append(pre + ["No statistics are computed because "
                                   f"{rvar.title if obs.shape[0] < 2 else cvar.title} "
                                   "is a constant."] + [""] * width)
                continue
            n = obs.sum()
            dof = (obs.shape[0] - 1) * (obs.shape[1] - 1)
            ex = exact_tests(obs, exact, *o["mc"], row_scores=_scores(rvar, rcats),
                             col_scores=_scores(cvar, ccats)) if exact else {}

            def tail(key, two=None, one=None):
                if not exact:
                    return [two or "", one or ""]
                e = ex.get(key)
                if e is None:
                    return [""] * len(exact_cols)
                if e["method"] == "exact":
                    return [fmt_p(e["p"]), fmt_p(e["p1"]) if "p1" in e else "",
                            fmt(e.get("point"), 3)]
                return [fmt_p(e["p"]) + "ᶜ", fmt_p(e.get("lower")), fmt_p(e.get("upper")),
                        fmt_p(e["p1"]) if "p1" in e else ""]

            first = True
            rowspec = []
            if "chisq" in s:
                chi, p_chi = pearson_chi2(obs)
                expected = np.outer(obs.sum(1), obs.sum(0)) / n
                rowspec.append(["Pearson Chi-Square", fmt(chi) + "ᵃ", fmt_int(dof),
                                fmt_p(p_chi)] + tail("pearson"))
                if obs.shape == (2, 2):
                    any_two = True
                    cc_chi, cc_p, _, _ = stats.chi2_contingency(obs, correction=True)
                    rowspec.append(["Continuity Correctionᵇ", fmt(cc_chi), fmt_int(dof),
                                    fmt_p(cc_p)] + ([""] * len(exact_cols) if exact else
                                                    ["", ""]))
                lr, lr_p = likelihood_ratio(obs)
                rowspec.append(["Likelihood Ratio", fmt(lr), fmt_int(dof), fmt_p(lr_p)]
                               + tail("lr"))
                if obs.shape == (2, 2) or exact:
                    if exact:
                        ffh = float(table_statistics(obs)["fisher"])
                        rowspec.append(["Fisher's Exact Test",
                                        fmt(ffh) if obs.shape != (2, 2) else "", "", ""]
                                       + tail("fisher"))
                    else:
                        _, fisher_two = stats.fisher_exact(obs, alternative="two-sided")
                        one = min(stats.fisher_exact(obs, alternative="less")[1],
                                  stats.fisher_exact(obs, alternative="greater")[1])
                        rowspec.append(["Fisher's Exact Test", "", "", "", fmt_p(fisher_two),
                                        fmt_p(one)])
                u, v = _scores(rvar, rcats), _scores(cvar, ccats)
                r = _weighted_corr(obs, u, v)
                lbl = (n - 1) * r * r if np.isfinite(r) else float("nan")
                rowspec.append(["Linear-by-Linear Association", fmt(lbl), "1",
                                fmt_p(stats.chi2.sf(lbl, 1))] + (
                    tail("lbl") if exact else ["", ""]))
                small = int((expected < 5).sum())
                where = ", ".join(f"{lv.title} = {v}" for lv, v in zip(lvars, slabels)
                                  if v != "Total") or ("Total" if slabels else "")
                where = f"{where}: " if where else ""
                notes.append(f"a. {where}{small} cells ({100 * small / expected.size:.1f}%) "
                             f"have expected count less than 5. The minimum expected count is "
                             f"{expected.min():.2f}.")
            if "mcnemar" in s:
                if obs.shape[0] == obs.shape[1] and list(map(str, rcats)) == \
                        list(map(str, ccats)):
                    res = mcnemar(obs)
                    if res["kind"] == "mcnemar":
                        rowspec.append(["McNemar Test", "", "", ""] + (
                            [fmt_p(res["p"]) + "ᵈ", "", ""][:len(exact_cols)]
                            if exact else [fmt_p(res["p"]) + "ᵈ", ""]))
                        if "d. Binomial distribution used." not in late:
                            late.append("d. Binomial distribution used.")
                    else:
                        rowspec.append(["McNemar-Bowker Test", fmt(res["value"]),
                                        fmt_int(res["df"]), fmt_p(res["p"])]
                                       + [""] * (len(exact_cols) if exact else 2))
                else:
                    rowspec.append(["McNemar Test", "", "", "Not computed: the table must be "
                                    "square with matching categories."]
                                   + [""] * (len(exact_cols) if exact else 2))
            rowspec.append(["N of Valid Cases", fmt_int(n), "", ""]
                           + [""] * (len(exact_cols) if exact else 2))
            for spec in rowspec:
                crow.append(prefix(slabels, first) + spec)
                first = False
        if any_two:
            notes.append("b. Computed only for a 2x2 table")
        if exact == "mc":
            notes.append(f"c. Based on {o['mc'][0]} sampled tables with starting seed "
                         f"{o['mc'][2]}.")
        notes += late
        cols = lead + ["", "Value", "df", "Asymptotic Significance (2-sided)"] + (
            exact_cols if exact else ["Exact Sig. (2-sided)", "Exact Sig. (1-sided)"])
        section.table(Table("Chi-Square Tests", cols, crow, row_header_cols=nl + 1,
                            footnotes=notes))

    want_dir = s & {"lambda", "uc", "d", "eta"}
    if want_dir:
        drows = []
        notes_used = set()
        for slabels, rcats, ccats, obs in results:
            first = True
            if min(obs.shape) < 2:
                drows.append(prefix(slabels, True) + ["No statistics are computed because a "
                                                      "variable is a constant.", "", "", "", "",
                                                      "", ""])
                continue
            nom = nominal_measures(obs) if s & {"lambda", "uc"} else {}
            ordm = ordinal_measures(obs) if "d" in s else {}
            rdep, cdep = f"{rvar.title} Dependent", f"{cvar.title} Dependent"
            spec = []
            if "lambda" in s:
                spec += [("Nominal by Nominal", "Lambda", "Symmetric", nom["lambda_S"]),
                         ("", "", rdep, nom["lambda_R"]), ("", "", cdep, nom["lambda_C"]),
                         ("", "Goodman and Kruskal tau", rdep, nom["tau_R"]),
                         ("", "", cdep, nom["tau_C"])]
            if "uc" in s:
                spec += [("" if spec else "Nominal by Nominal", "Uncertainty Coefficient",
                          "Symmetric", nom["uc_S"]), ("", "", rdep, nom["uc_R"]),
                         ("", "", cdep, nom["uc_C"])]
            if "d" in s:
                spec += [("Ordinal by Ordinal", "Somers' d", "Symmetric", ordm.get("d_S")),
                         ("", "", rdep, ordm.get("d_R")), ("", "", cdep, ordm.get("d_C"))]
            for group, name, sub, m in spec:
                if m is None:
                    continue
                sig_mark = {"chisq": "ᶜ", "lr": "ᵈ"}.get(m.note, "")
                if m.note == "zero":
                    notes_used.add("e")
                if sig_mark:
                    notes_used.add({"ᶜ": "c", "ᵈ": "d"}[sig_mark])
                t_text = "" if m.note == "chisq" else (fmt(m.t) if np.isfinite(m.t)
                                                       else ("ᵉ" if m.note == "zero" else ""))
                drows.append(prefix(slabels, first) + [
                    group, name, sub, fmt(m.value), fmt(m.ase), t_text,
                    (fmt_p(m.p) + sig_mark) if np.isfinite(m.p) else ""])
                first = False
            if "eta" in s:
                e = eta(obs, _scores(rvar, rcats), _scores(cvar, ccats))
                drows.append(prefix(slabels, first) + ["Nominal by Interval", "Eta", rdep,
                                                       fmt(e["R"]), "", "", ""])
                drows.append(prefix(slabels, False) + ["", "", cdep, fmt(e["C"]), "", "", ""])
        notes = ["a. Not assuming the null hypothesis.",
                 "b. Using the asymptotic standard error assuming the null hypothesis."]
        if "c" in notes_used:
            notes.append("c. Based on chi-square approximation")
        if "d" in notes_used:
            notes.append("d. Likelihood ratio chi-square probability.")
        if "e" in notes_used:
            notes.append("e. Cannot be computed because the asymptotic standard error equals "
                         "zero.")
        section.table(Table("Directional Measures", lead + [
            "", "", "", "Value", "Asymptotic Std. Errorᵃ", "Approximate Tᵇ",
            "Approximate Significance"], drows, row_header_cols=nl + 3, footnotes=notes))

    want_sym = s & {"phi", "cc", "btau", "ctau", "gamma", "corr", "kappa"}
    if want_sym:
        srows = []
        kappa_note = normal_note = False
        for slabels, rcats, ccats, obs in results:
            first = True
            if min(obs.shape) < 2:
                srows.append(prefix(slabels, True) + ["No statistics are computed because a "
                                                      "variable is a constant.", "", "", "", "",
                                                      ""])
                continue
            n = obs.sum()
            spec = []
            if s & {"phi", "cc"}:
                nom = nominal_measures(obs)
                group = "Nominal by Nominal"
                if "phi" in s:
                    spec += [(group, "Phi", nom["phi"], False),
                             ("", "Cramer's V", nom["v"], False)]
                    group = ""
                if "cc" in s:
                    spec.append((group, "Contingency Coefficient", nom["cc"], False))
            if s & {"btau", "ctau", "gamma"} or ("corr" in s):
                ordm = ordinal_measures(obs) if s & {"btau", "ctau", "gamma"} else {}
                group = "Ordinal by Ordinal"
                for key, name in (("btau", "Kendall's tau-b"), ("ctau", "Kendall's tau-c"),
                                  ("gamma", "Gamma")):
                    if key in s and key in ordm:
                        spec.append((group, name, ordm[key], True))
                        group = ""
                if "corr" in s:
                    cor = correlations(obs, _scores(rvar, rcats), _scores(cvar, ccats))
                    spec.append((group, "Spearman Correlation", cor["spearman"], True))
                    spec.append(("Interval by Interval", "Pearson's R", cor["pearson"], True))
                    normal_note = True
            if "kappa" in s:
                if obs.shape[0] == obs.shape[1] and list(map(str, rcats)) == \
                        list(map(str, ccats)):
                    spec.append(("Measure of Agreement", "Kappa", kappa(obs), True))
                else:
                    kappa_note = True
            for group, name, m, full in spec:
                mark = "ᶜ" if m.note == "normal" else ""
                srows.append(prefix(slabels, first) + [
                    group, name, fmt(m.value), fmt(m.ase) if full else "",
                    (fmt(m.t) if np.isfinite(m.t) else "") if full else "",
                    (fmt_p(m.p) + mark) if np.isfinite(m.p) else ""])
                first = False
            srows.append(prefix(slabels, first) + ["N of Valid Cases", "", fmt_int(n), "", "",
                                                   ""])
        notes = []
        if s & {"btau", "ctau", "gamma", "corr", "kappa"}:
            notes = ["a. Not assuming the null hypothesis.",
                     "b. Using the asymptotic standard error assuming the null hypothesis."]
        if normal_note:
            notes.append("c. Based on normal approximation.")
        if kappa_note:
            notes.append("Kappa statistics cannot be computed. They require a symmetric 2-way "
                         "table in which the values of the first variable match the values of "
                         "the second variable.")
        full_cols = bool(s & {"btau", "ctau", "gamma", "corr", "kappa"})
        cols = lead + ["", "", "Value"] + (["Asymptotic Std. Errorᵃ", "Approximate Tᵇ"]
                                           if full_cols else ["", ""]) + \
            ["Approximate Significance"]
        section.table(Table("Symmetric Measures", cols, srows, row_header_cols=nl + 2,
                            footnotes=notes))

    if "risk" in s:
        rrows, note = [], []
        for slabels, rcats, ccats, obs in results:
            first = True
            if obs.shape != (2, 2) or (obs == 0).any():
                rrows.append(prefix(slabels, True) + ["Odds Ratio", "ᵃ", "", ""])
                note = ["a. Risk Estimate statistics cannot be computed. They are only computed "
                        "for a 2*2 table without empty cells."]
                continue
            res = risk(obs)
            r1, r2 = value_text(rvar, rcats[0]), value_text(rvar, rcats[1])
            spec = [(f"Odds Ratio for {rvar.title} ({r1} / {r2})", res.get("or")),
                    (f"For cohort {cvar.title} = {value_text(cvar, ccats[0])}", res.get("rr1")),
                    (f"For cohort {cvar.title} = {value_text(cvar, ccats[1])}", res.get("rr2"))]
            for name, val in spec:
                if val is None:
                    continue
                rrows.append(prefix(slabels, first) + [name, fmt(val[0]), fmt(val[1]),
                                                       fmt(val[2])])
                first = False
            rrows.append(prefix(slabels, False) + ["N of Valid Cases", fmt_int(obs.sum()), "",
                                                   ""])
        section.table(Table("Risk Estimate", lead + ["", "Value",
                                                     ("95% Confidence Interval", "Lower"),
                                                     ("95% Confidence Interval", "Upper")],
                            rrows, row_header_cols=nl + 1, footnotes=note))

    if "cmh" in s:
        _cmh_tables(section, rvar, cvar, lvars, results, o)


def _cmh_tables(section, rvar, cvar, lvars, results, o) -> None:
    strata = [r for r in results if not (r[0] and r[0][-1] == "Total" and len(results) > 1)]
    rset = sorted({str(c) for r in strata for c in r[1]})
    cset = sorted({str(c) for r in strata for c in r[2]})
    if len(rset) != 2 or len(cset) != 2:
        section.note("Cochran's and Mantel-Haenszel Statistics",
                     "These statistics are computed only for 2 x 2 tables (dichotomous row and "
                     "column variables).", style="warning")
        return
    rcats_all = sorted({c for r in strata for c in r[1]}, key=lambda v: (isinstance(v, str), v))
    ccats_all = sorted({c for r in strata for c in r[2]}, key=lambda v: (isinstance(v, str), v))
    tables = []
    for _, rcats, ccats, obs in strata:
        t = np.zeros((2, 2))
        for i, rc in enumerate(rcats):
            for j, cc in enumerate(ccats):
                t[rcats_all.index(rc), ccats_all.index(cc)] = obs[i, j]
        tables.append(t)
    res = cmh(tables, o["cmh_value"])
    rows = []
    if "breslow" in res:
        rows = [["Breslow-Day", fmt(res["breslow"][0]), fmt_int(res["breslow"][1]),
                 fmt_p(res["breslow"][2])],
                ["Tarone's", fmt(res["tarone"][0]), fmt_int(res["tarone"][1]),
                 fmt_p(res["tarone"][2])]]
        section.table(Table("Tests of Homogeneity of the Odds Ratio", [
            "", "Chi-Squared", "df", "Asymptotic Significance (2-sided)"], rows))
    section.table(Table("Tests of Conditional Independence", [
        "", "Chi-Squared", "df", "Asymptotic Significance (2-sided)"],
        [["Cochran's", fmt(res["cochran"][0]), "1", fmt_p(res["cochran"][1])],
         ["Mantel-Haenszel", fmt(res["mh"][0]), "1", fmt_p(res["mh"][1])]],
        footnotes=["Under the conditional independence assumption, Cochran's statistic is "
                   "asymptotically distributed as a 1 df chi-squared distribution, only if the "
                   "number of strata is fixed, while the Mantel-Haenszel statistic is always "
                   "asymptotically distributed as a 1 df chi-squared distribution. Note that "
                   "the continuity correction is removed from the Mantel-Haenszel statistic "
                   "when the sum of the differences between the observed and the expected is "
                   "0."]))
    if "or" in res:
        e = res["or"]
        section.table(Table("Mantel-Haenszel Common Odds Ratio Estimate", ["", "", "", ""], [
            ["Estimate", "", "", fmt(e["estimate"])],
            ["ln(Estimate)", "", "", fmt(e["ln"])],
            ["Standard Error of ln(Estimate)", "", "", fmt(e["se"])],
            ["Asymptotic Significance (2-sided)", "", "", fmt_p(e["p"])],
            ["Asymptotic 95% Confidence Interval", "Common Odds Ratio", "Lower Bound",
             fmt(e["lo"])],
            ["", "", "Upper Bound", fmt(e["hi"])],
            ["", "ln(Common Odds Ratio)", "Lower Bound", fmt(e["ln_lo"])],
            ["", "", "Upper Bound", fmt(e["ln_hi"])]], row_header_cols=3,
            footnotes=[f"The Mantel-Haenszel common odds ratio estimate is asymptotically "
                       f"normally distributed under the common odds ratio of "
                       f"{o['cmh_value']:.3f} assumption. So is the natural log of the "
                       "estimate."]))


def clustered_bar_chart(row_labels: list[str], col_labels: list[str], obs: np.ndarray,
                        xlabel: str, legend: str, title: str) -> bytes:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from rsstatchem.core.stats.charts import _style

    fig = Figure(figsize=(max(5.0, 0.9 * len(row_labels) * max(1, len(col_labels)) / 2 + 3),
                          4.2), dpi=120)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    palette = ["#C62828", "#1f4e79", "#7cb342", "#f9a825", "#6a1b9a", "#00838f", "#8d6e63",
               "#ef6c00", "#455a64", "#ad1457"]
    k = max(1, len(col_labels))
    width = 0.8 / k
    x = np.arange(len(row_labels))
    for j, lab in enumerate(col_labels):
        ax.bar(x - 0.4 + width * (j + 0.5), obs[:, j], width, label=lab,
               color=palette[j % len(palette)], edgecolor="#333333", linewidth=0.5)
    ax.set_xticks(x, row_labels)
    ax.legend(title=legend, fontsize=8, title_fontsize=9, frameon=False)
    _style(ax, title, xlabel, "Count")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    return buf.getvalue()

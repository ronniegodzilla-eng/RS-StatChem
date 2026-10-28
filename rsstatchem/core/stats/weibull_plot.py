"""Graphs > Weibull Plot (SPSS 28+). Failure times on Weibull probability paper: ln(t) against
ln(−ln(1 − F)), with F from Benard's median ranks (i − 0.3)/(n + 0.4); right-censored times use
Johnson's adjusted ranks (the rank increment (n + 1 − previous rank)/(1 + number of items
beyond) for each failure). The Weibull line is fitted by rank regression (X on Y, the
reliability-engineering default, or Y on X) or by maximum likelihood (censoring handled),
giving the shape β and scale η of F(t) = 1 − exp(−(t/η)^β), with Wald intervals for the ML
estimates. Optional grouping draws one line per group.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, value_text
from rsstatchem.core.stats.ml_common import MLError, raw_column, sort_levels
from rsstatchem.core.stats.roc import figure, png, style

WeibullError = MLError


def median_ranks(t: np.ndarray, failed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(failure times, Benard median rank F) with Johnson's adjusted ranks for censoring."""
    order = np.lexsort((~failed, t))     # failures before censorings at tied times
    t, failed = t[order], failed[order]
    n = len(t)
    prev = 0.0
    times, F = [], []
    for i in range(n):
        if not failed[i]:
            continue
        beyond = n - i          # items at or beyond this position (reverse rank)
        inc = (n + 1 - prev) / (1 + beyond)
        prev += inc
        times.append(t[i])
        F.append((prev - 0.3) / (n + 0.4))
    return np.array(times), np.array(F)


def rank_regression(times: np.ndarray, F: np.ndarray, on: str = "x") -> tuple[float, float]:
    """Shape and scale from the line through (ln t, ln(−ln(1−F)))."""
    x = np.log(times)
    y = np.log(-np.log(1 - F))
    if on == "x":       # regress ln t on y: ln t = ln η + y / β
        b, a = np.polyfit(y, x, 1)
        beta = 1 / b
        eta = np.exp(a)
    else:               # y = β ln t − β ln η
        b, a = np.polyfit(x, y, 1)
        beta = b
        eta = np.exp(-a / b)
    return float(beta), float(eta)


def weibull_mle(t: np.ndarray, failed: np.ndarray) -> dict:
    """ML estimates of (β, η) for right-censored data with Wald SEs (delta method on logs)."""
    t = np.asarray(t, float)
    d = np.asarray(failed, bool)
    lt = np.log(t)

    def nll(th):
        lb, le = th
        b, e = np.exp(lb), np.exp(le)
        z = (lt - le) * b
        return -((d * (lb + (b - 1) * lt - b * le)).sum() - np.exp(z).sum())

    sd = lt.std() if lt.std() > 0 else 1.0
    x0 = np.array([np.log(1.2 / sd), lt.mean()])
    res = optimize.minimize(nll, x0, method="BFGS")
    res = optimize.minimize(nll, res.x, method="Nelder-Mead",
                            options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000})
    lb, le = res.x
    H = _hessian(nll, res.x)
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.full((2, 2), np.nan)
    return {"beta": float(np.exp(lb)), "eta": float(np.exp(le)), "cov_log": cov,
            "loglik": float(-res.fun)}


def _hessian(f, x, h=1e-4):
    k = len(x)
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            e_i, e_j = np.eye(k)[i] * h, np.eye(k)[j] * h
            H[i, j] = (f(x + e_i + e_j) - f(x + e_i - e_j) - f(x - e_i + e_j)
                       + f(x - e_i - e_j)) / (4 * h * h)
    return H


def weibull_plot(ds: Dataset, variable: str, status: str | None = None,
                 event_values: list | None = None, group: str | None = None,
                 method: str = "rrx", level: float = 0.95, plot: bool = True) -> Section:
    t = raw_column(ds, variable).to_numpy(float)
    ok = np.isfinite(t) & (t > 0)
    if status:
        s = raw_column(ds, status)
        ok &= s.notna().to_numpy()
        ev = s.isin(event_values or [1.0]).to_numpy()
    else:
        ev = np.ones(ds.n_cases, bool)
    if group:
        g = raw_column(ds, group)
        ok &= g.notna().to_numpy()
        levels = sort_levels(g[ok].tolist())
        groups = [(value_text(ds.var(group), lv), ok & (g == lv).to_numpy()) for lv in levels]
    else:
        groups = [("", ok)]
    section = Section("Weibull Plot")
    section.log(f"WEIBULLPLOT VARIABLE={variable}" + (f" STATUS={status}" if status else "")
                + (f" BY {group}" if group else "")
                + f"\n  /METHOD={method.upper()}.")
    z = stats.norm.ppf(0.5 + level / 2)
    rows = []
    fig = figure(6.0, 4.6) if plot else None
    ax = fig.add_subplot(111) if plot else None
    for k, (lab, sel) in enumerate(groups):
        tt, dd = t[sel], ev[sel]
        if dd.sum() < 2:
            raise WeibullError("At least two failures are needed" + (f" in group {lab}."
                                                                      if lab else "."))
        times, F = median_ranks(tt, dd)
        if method == "mle":
            m = weibull_mle(tt, dd)
            beta, eta = m["beta"], m["eta"]
            se_lb, se_le = np.sqrt(np.diag(m["cov_log"]))
            ci_b = (beta * np.exp(-z * se_lb), beta * np.exp(z * se_lb))
            ci_e = (eta * np.exp(-z * se_le), eta * np.exp(z * se_le))
        else:
            beta, eta = rank_regression(times, F, "x" if method == "rrx" else "y")
            ci_b = ci_e = (np.nan, np.nan)
        y = np.log(-np.log(1 - F))
        r = np.corrcoef(np.log(times), y)[0, 1]
        rows.append(([lab] if group else []) + [
            fmt_int(len(tt)), fmt_int(int(dd.sum())), fmt(beta, 4), fmt(ci_b[0], 4),
            fmt(ci_b[1], 4), fmt(eta, 4), fmt(ci_e[0], 4), fmt(ci_e[1], 4), fmt(r ** 2, 4)])
        if plot:
            c = f"C{k}"
            ax.scatter(times, y, s=16, color=c, label=lab or "Observed")
            xs = np.linspace(np.log(times.min()) - 0.3, np.log(times.max()) + 0.3, 50)
            ax.plot(np.exp(xs), beta * (xs - np.log(eta)), color=c, lw=1.2)
    cols = ([ds.var(group).title] if group else []) + [
        "N", "Failures", ("Shape (β)", "Estimate"), ("Shape (β)", "Lower"),
        ("Shape (β)", "Upper"), ("Scale (η)", "Estimate"), ("Scale (η)", "Lower"),
        ("Scale (η)", "Upper"), "R Square (plot)"]
    meth = {"rrx": "rank regression on X (ln t on the Weibull score)",
            "rry": "rank regression on Y", "mle": "maximum likelihood"}[method]
    section.table(Table("Weibull Parameter Estimates", cols, rows,
                        footnotes=[f"Estimated by {meth}; median ranks (Benard)"
                                   + (" with Johnson's adjustment for censored times"
                                      if status else "") + ".",
                                   f"{level * 100:g}% Wald confidence limits (maximum "
                                   "likelihood only)."]))
    if plot:
        probs = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.632, 0.8, 0.9, 0.95, 0.99])
        ax.set_yticks(np.log(-np.log(1 - probs)), [f"{100 * p:g}" for p in probs])
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.3)
        if group:
            ax.legend(fontsize=8, title=ds.var(group).title)
        style(ax, f"Weibull Plot of {ds.var(variable).title}", ds.var(variable).title,
              "Cumulative Percent Failed")
        section.image("Weibull Plot", png(fig), width=520)
    return section


def selftest() -> None:
    t = np.array([10.0, 20, 30, 40, 50])
    times, F = median_ranks(t, np.ones(5, bool))
    assert abs(F[0] - 0.7 / 5.4) < 1e-12
    b, e = rank_regression(times, F)
    assert b > 0 and e > 0

"""Analyze > Forecasting > Vector Autoregression (SPSS 32 VAR). A wrapper around statsmodels'
``VAR`` (OLS per equation): lag-order selection by AIC / BIC (SC) / HQ / FPE, coefficient
tables per equation, residual correlations, the stability check (moduli of the companion
roots), Granger causality and instantaneous causality tests, impulse responses (orthogonalised
by the Cholesky factor in variable order), forecast error variance decomposition and
forecasts with intervals.

IBM's Algorithms manual (v31) has no VAR chapter yet; results equal statsmodels / R ``vars``.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, fmt_p
from rsstatchem.core.stats.ml_common import MLError, MLResult, raw_column
from rsstatchem.core.stats.roc import figure, png, style

VarError = MLError
TRENDS = {"c": "Constant", "n": "None", "ct": "Constant and trend",
          "ctt": "Constant, linear and quadratic trend"}
CRITERIA = {"aic": "AIC", "bic": "BIC (SC)", "hqic": "HQ", "fpe": "FPE"}


def fit_var(Y: np.ndarray, names: list[str], lags: int | None = None, max_lags: int = 8,
            criterion: str = "aic", trend: str = "c", exog: np.ndarray | None = None):
    import pandas as pd
    from statsmodels.tsa.api import VAR

    model = VAR(pd.DataFrame(Y, columns=names), exog=exog)
    sel = None
    if lags is None:
        max_lags = int(min(max_lags, max(1, (len(Y) - 1) // (Y.shape[1] + 1) - 1)))
        sel = model.select_order(maxlags=max_lags, trend=trend)
        lags = int(max(1, sel.selected_orders[criterion]))
    res = model.fit(lags, trend=trend)
    return res, sel


def var_procedure(ds: Dataset, variables: list[str], lags: int | None = None,
                  max_lags: int = 8, criterion: str = "aic", trend: str = "c",
                  exog: list[str] | None = None, granger: bool = True,
                  irf_periods: int = 10, irf_plot: bool = True, fevd: bool = True,
                  forecast: int = 0, level: float = 0.95, stability: bool = True,
                  residual_corr: bool = True, save_forecast: bool = False,
                  save_resid: bool = False) -> MLResult:
    if len(variables) < 2:
        raise VarError("Select at least two series.")
    cols = list(variables) + list(exog or [])
    X = np.column_stack([raw_column(ds, v).to_numpy(float) for v in cols])
    ok = np.isfinite(X).all(1)
    rows = np.where(ok)[0]
    if len(rows) < 3 * len(variables) + 5:
        raise VarError("Too few valid observations for a VAR.")
    if not np.all(np.diff(rows) == 1):
        raise VarError("The series contain missing values inside the estimation span; VAR "
                       "needs consecutive observations.")
    Y = X[rows, :len(variables)]
    E = X[rows, len(variables):] if exog else None
    res, sel = fit_var(Y, list(variables), lags, max_lags, criterion, trend, E)
    p = res.k_ar
    section = Section("Vector Autoregression")
    section.log(f"VAR /VARIABLES={' '.join(variables)}"
                + (f" WITH {' '.join(exog)}" if exog else "")
                + f"\n  /MODEL LAGS={p if lags else f'AUTO({criterion.upper()}, MAX={max_lags})'}"
                f" TREND={trend.upper()}\n  /PRINT=GRANGER IRF({irf_periods}) FEVD"
                + (f"\n  /FORECAST={forecast}" if forecast else "") + ".")
    if sel is not None:
        ics = sel.ics
        rows_s = []
        nl = len(ics["aic"])
        for L in range(nl):
            rows_s.append([str(L)] + [fmt(ics[k][L], 4) + ("*" if sel.selected_orders[k] == L
                                                            else "")
                                      for k in ("aic", "bic", "hqic", "fpe")])
        section.table(Table("Lag Order Selection", ["Lag", "AIC", "BIC (SC)", "HQ", "FPE"],
                            rows_s, footnotes=["* The lag order selected by the criterion.",
                                               f"Selected by {CRITERIA[criterion]}: {p}."]))
    section.table(Table("Model Summary", ["", "Value"], [
        ["Endogenous series", ", ".join(variables)],
        ["Exogenous series", ", ".join(exog) if exog else "(none)"],
        ["Lag order", fmt_int(p)], ["Deterministic terms", TRENDS[trend]],
        ["Observations used", fmt_int(res.nobs)], ["Log-likelihood", fmt(res.llf, 3)],
        ["AIC", fmt(res.aic, 4)], ["BIC (SC)", fmt(res.bic, 4)], ["HQ", fmt(res.hqic, 4)],
        ["FPE", fmt(res.fpe, 6)], ["Determinant of residual covariance",
                                   fmt(np.linalg.det(res.sigma_u_mle), 6)]]))
    pnames = list(res.exog_names)
    params, stderr = np.asarray(res.params), np.asarray(res.stderr)
    resid_all, sigma_u = np.asarray(res.resid), np.asarray(res.sigma_u)
    for j, v in enumerate(variables):
        rows_c = []
        for i, nm in enumerate(pnames):
            b, se = params[i, j], stderr[i, j]
            t = b / se if se > 0 else np.nan
            pv = 2 * stats.t.sf(abs(t), res.df_resid) if np.isfinite(t) else np.nan
            rows_c.append([_pretty(nm, variables), fmt(b, 4), fmt(se, 4), fmt(t, 3),
                           fmt_p(pv)])
        resid = resid_all[:, j]
        yv = Y[p:, j]
        r2 = 1 - (resid ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
        section.table(Table(f"Equation: {v}", ["", "B", "Std. Error", "t", "Sig."], rows_c,
                            footnotes=[f"R Square = {fmt(r2, 3)}; residual SD = "
                                       f"{fmt(np.sqrt(sigma_u[j, j]), 4)}."]))
    if residual_corr:
        C = np.corrcoef(resid_all.T)
        section.table(Table("Residual Correlations", [""] + list(variables),
                            [[v] + [fmt(C[i, k], 3) for k in range(len(variables))]
                             for i, v in enumerate(variables)]))
    if stability:
        roots = np.abs(np.linalg.eigvals(_companion(res)))
        roots = np.sort(roots)[::-1]
        section.table(Table("Stability (Moduli of the Companion Roots)", ["Root", "Modulus"],
                            [[str(i + 1), fmt(m, 4)] for i, m in enumerate(roots)],
                            footnotes=["The VAR is stable when every modulus is below 1."
                                       + (" Stable." if roots.max() < 1 else
                                          " NOT stable.")]))
    if granger:
        rows_g = []
        for j, v in enumerate(variables):
            for i, w in enumerate(variables):
                if i == j:
                    continue
                t = res.test_causality(v, [w], kind="f")
                rows_g.append([v, w, fmt(t.test_statistic, 3), f"{t.df[0]}, {t.df[1]}",
                               fmt_p(t.pvalue)])
            others = [w for w in variables if w != v]
            if len(others) > 1:
                t = res.test_causality(v, others, kind="f")
                rows_g.append([v, "All", fmt(t.test_statistic, 3), f"{t.df[0]}, {t.df[1]}",
                               fmt_p(t.pvalue)])
        section.table(Table("Granger Causality Tests", ["Equation", "Excluded", "F", "df",
                                                        "Sig."], rows_g,
                            footnotes=["H0: the excluded series do not Granger-cause the "
                                       "equation's series."]))
    if irf_periods:
        irf = res.irf(int(irf_periods))
        if irf_plot:
            k = len(variables)
            fig = figure(2.4 * k + 1, 2.0 * k + 0.6)
            for i in range(k):
                for j in range(k):
                    ax = fig.add_subplot(k, k, i * k + j + 1)
                    ax.plot(irf.orth_irfs[:, i, j], color="#1f5fa8")
                    ax.axhline(0, color="#999", lw=0.6)
                    ax.set_title(f"{variables[j]} → {variables[i]}", fontsize=8)
                    ax.tick_params(labelsize=7)
            fig.suptitle("Orthogonalized Impulse Responses", fontsize=10, fontweight="bold")
            section.image("Impulse Responses", png(fig), width=min(160 * k + 120, 720))
        rows_i = []
        for h in range(int(irf_periods) + 1):
            rows_i.append([str(h)] + [fmt(irf.orth_irfs[h, i, j], 4)
                                      for i in range(len(variables))
                                      for j in range(len(variables))])
        section.table(Table("Orthogonalized Impulse Responses",
                            ["Period"] + [(f"Response of {variables[i]}",
                                           f"to {variables[j]}")
                                          for i in range(len(variables))
                                          for j in range(len(variables))], rows_i))
    if fevd:
        fd = res.fevd(max(int(irf_periods), 1)).decomp   # (k, periods, k)
        for i, v in enumerate(variables):
            section.table(Table(f"Variance Decomposition of {v}",
                                ["Period"] + list(variables),
                                [[str(h + 1)] + [fmt(100 * fd[i, h, j], 2)
                                                 for j in range(len(variables))]
                                 for h in range(fd.shape[1])],
                                footnotes=["Percentage of the forecast error variance due to "
                                           "each orthogonalised shock."]))
    saved = {}
    if forecast:
        if exog:
            raise VarError("Forecasts with exogenous series need their future values; remove "
                           "the exogenous series to forecast.")
        f, lo, hi = res.forecast_interval(Y[-p:], int(forecast), alpha=1 - level)
        rows_f = []
        for h in range(int(forecast)):
            rows_f.append([str(h + 1)] + [x for j in range(len(variables)) for x in
                                          (fmt(f[h, j], 4), fmt(lo[h, j], 4),
                                           fmt(hi[h, j], 4))])
        pc = f"{level * 100:g}%"
        section.table(Table("Forecasts", ["Horizon"] + [(v, s) for v in variables
                                                         for s in ("Forecast", f"LCL {pc}",
                                                                   f"UCL {pc}")], rows_f))
        fig = figure(6.0, 2.2 * len(variables))
        for j, v in enumerate(variables):
            ax = fig.add_subplot(len(variables), 1, j + 1)
            n = len(Y)
            ax.plot(np.arange(n), Y[:, j], color="#1f5fa8")
            xs = np.arange(n, n + int(forecast))
            ax.plot(xs, f[:, j], color="#c0392b")
            ax.fill_between(xs, lo[:, j], hi[:, j], color="#c0392b", alpha=0.15)
            style(ax, v, "", "")
        section.image("Forecasts", png(fig), width=560)
        if save_forecast:
            for j, v in enumerate(variables):
                saved[f"Predicted_{v}"[:60]] = (f"VAR fitted value of {v}",
                                                _series(ds, rows[p:], Y[p:, j]
                                                        - resid_all[:, j]))
    if save_resid:
        for j, v in enumerate(variables):
            saved[f"Residual_{v}"[:60]] = (f"VAR residual of {v}",
                                           _series(ds, rows[p:], resid_all[:, j]))
    return MLResult(section, saved, res)


def _series(ds, rows, values):
    import pandas as pd

    out = pd.Series(np.nan, index=ds.df.index, dtype=float)
    out.iloc[rows] = values
    return out


def _companion(res) -> np.ndarray:
    k, p = res.neqs, res.k_ar
    A = np.zeros((k * p, k * p))
    A[:k, :] = np.hstack([res.coefs[i] for i in range(p)])
    if p > 1:
        A[k:, :-k] = np.eye(k * (p - 1))
    return A


def _pretty(name: str, variables: list[str]) -> str:
    if name == "const":
        return "Constant"
    if name in ("trend", "trend**2"):
        return "Trend" if name == "trend" else "Trend²"
    if name.startswith("L") and "." in name:
        lag, _, var = name.partition(".")
        return f"{var} (lag {lag[1:]})"
    return name


def selftest() -> None:
    r = np.random.default_rng(0)
    Y = np.zeros((120, 2))
    for t in range(1, 120):
        Y[t] = 0.5 * Y[t - 1] + r.normal(size=2)
    res, _ = fit_var(Y, ["a", "b"], 1)
    assert abs(np.asarray(res.params)[1, 0] - 0.5) < 0.2

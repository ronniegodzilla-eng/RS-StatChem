"""Fit comparison as in Origin's Analysis > Fitting > Compare Models / Compare Datasets /
Rank Models (OriginPro).

* **Compare Models**: F-test for nested models, Akaike (AIC, AICc) and Bayesian (BIC)
  information criteria with Akaike weights and evidence ratio. Origin's definitions are used:
  AIC = N·ln(RSS/N) + 2K, AICc = AIC + 2K(K + 1)/(N − K − 1), BIC = N·ln(RSS/N) + K·ln(N),
  with K = number of fitted parameters + 1. Differences between models (the only thing
  that matters) equal those of the likelihood-based definition used by statsmodels.
* **Compare Datasets**: do two or more datasets share one parameter set? Separate fits
  (independent) against one fit with all parameters shared (concatenated), by F-test and AIC.
* **Rank Models**: fit a list of functions (e.g. a whole category) and rank them by AICc /
  BIC / Adj. R².
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from rsstatchem.core.analysis import fit_multi as fm
from rsstatchem.core.analysis import fitting as fit
from rsstatchem.core.analysis import functions as fl
from rsstatchem.core.analysis.fitting import FitResult

CRITERIA = {"aicc": "AICc", "aic": "AIC", "bic": "BIC", "adj_r2": "Adj. R-Square",
            "red_chisq": "Reduced Chi-Sqr"}


def f_test(simple: FitResult, complex_: FitResult) -> dict:
    """Extra-sum-of-squares F-test of nested models (``simple`` ⊂ ``complex_``)."""
    df1 = simple.dof - complex_.dof
    df2 = complex_.dof
    if df1 <= 0 or df2 <= 0:
        raise ValueError("The second model must have more parameters than the first and "
                         "positive residual degrees of freedom.")
    num = (simple.rss - complex_.rss) / df1
    den = complex_.rss / df2
    F = num / den if den > 0 else float("inf")
    p = float(stats.f.sf(F, df1, df2)) if math.isfinite(F) else 0.0
    return {"F": float(F), "df1": int(df1), "df2": int(df2), "p": p,
            "rss1": simple.rss, "rss2": complex_.rss, "dof1": simple.dof,
            "dof2": complex_.dof, "preferred": 2 if p < 0.05 else 1}


def information(results: list[FitResult], labels: list[str] | None = None,
                criterion: str = "aicc") -> list[dict]:
    """AIC, AICc, BIC, Δ, Akaike weights and evidence ratios of fits to the same data."""
    labels = labels or [r.model.name for r in results]
    rows = [{"model": lab, "n": r.n, "k": r.k_eff, "rss": r.rss, "aic": r.aic,
             "aicc": r.aicc, "bic": r.bic, "adj_r2": r.adj_r2, "red_chisq": r.red_chisq,
             "result": r} for lab, r in zip(labels, results)]
    for crit in ("aic", "aicc", "bic"):
        vals = np.array([row[crit] for row in rows], float)
        ok = np.isfinite(vals)
        best = np.min(vals[ok]) if ok.any() else np.nan
        delta = vals - best
        w = np.where(ok, np.exp(-0.5 * np.where(ok, delta, 0.0)), 0.0)
        w = w / w.sum() if w.sum() > 0 else w
        for row, d, wi in zip(rows, delta, w):
            row[f"d_{crit}"] = float(d)
            row[f"w_{crit}"] = float(wi)
    if len(rows) == 2:
        a, b = rows
        for crit in ("aic", "aicc", "bic"):
            wa, wb = a[f"w_{crit}"], b[f"w_{crit}"]
            a[f"er_{crit}"] = wa / wb if wb > 0 else float("inf")
            b[f"er_{crit}"] = wb / wa if wa > 0 else float("inf")
    return rows


@dataclass
class Comparison:
    rows: list[dict]
    ftest: dict | None
    criterion: str
    notes: list[str] = field(default_factory=list)

    @property
    def best(self) -> str:
        key = self.criterion
        if key in ("adj_r2",):
            return max(self.rows, key=lambda r: r[key])["model"]
        return min(self.rows, key=lambda r: (r[key] if math.isfinite(r[key]) else 1e300))[
            "model"]


def compare_models(x, y, models, weights=None, criterion: str = "aicc", **kw) -> Comparison:
    """Fit two (or more) models to the same data and compare them. The F-test is reported
    for two models when they have different numbers of parameters (it assumes nesting)."""
    res = [fit.fit_curve(x, y, m, weights=weights, **kw) for m in models]
    labels = [r.model.name if not r.model.user_defined else r.model.formula for r in res]
    rows = information(res, labels, criterion)
    ft = None
    notes = []
    if len(res) == 2 and res[0].k_eff != res[1].k_eff:
        a, b = sorted(res, key=lambda r: r.k_eff)
        try:
            ft = f_test(a, b)
            ft["simple"] = labels[res.index(a)]
            ft["complex"] = labels[res.index(b)]
        except ValueError as exc:
            notes.append(str(exc))
    return Comparison(rows, ft, criterion, notes)


def compare_datasets(data, model, criterion: str = "aicc", **kw) -> dict:
    """Test whether datasets can be described by one parameter set (Origin's Compare
    Datasets): independent fits vs. one fit to the pooled data."""
    ind = fm.independent_fit(data, model, **kw)
    con = fm.concatenate_fit(data, model, **kw)
    rss_i = float(sum(r.rss for r in ind.results))
    dof_i = int(sum(r.dof for r in ind.results))
    n = int(sum(r.n for r in ind.results))
    k_i = int(sum(r.k_eff for r in ind.results))
    rss_c, dof_c, k_c = con.combined.rss, con.combined.dof, con.combined.k_eff
    df1 = dof_c - dof_i
    F = ((rss_c - rss_i) / df1) / (rss_i / dof_i) if df1 > 0 and dof_i > 0 and rss_i > 0 \
        else float("nan")
    p = float(stats.f.sf(F, df1, dof_i)) if math.isfinite(F) else float("nan")

    def crit(rss, k):
        K = k + 1
        aic = n * math.log(rss / n) + 2 * K
        return {"aic": aic, "aicc": aic + 2 * K * (K + 1) / (n - K - 1) if n - K - 1 > 0
                else float("nan"), "bic": n * math.log(rss / n) + K * math.log(n)}
    ci = crit(rss_i, k_i)
    cc = crit(rss_c, k_c)
    return {"independent": ind, "combined": con, "rss_ind": rss_i, "dof_ind": dof_i,
            "rss_comb": rss_c, "dof_comb": dof_c, "F": float(F), "df1": int(df1),
            "df2": int(dof_i), "p": p, "crit_ind": ci, "crit_comb": cc,
            "same": bool(p >= 0.05) if math.isfinite(p) else None,
            "preferred_by_" + criterion: "combined" if cc[criterion] <= ci[criterion]
            else "independent"}


def rank_models(x, y, models=None, category: str | None = None, weights=None,
                criterion: str = "aicc", **kw) -> list[dict]:
    """Fit every model (names / FitFunctions, or all functions of ``category``) and rank by
    ``criterion``. Models that fail are listed last with their error message."""
    if models is None:
        if category is None:
            raise ValueError("Give a list of models or a category.")
        models = [f.name for f in fl.functions_in(category)]
    ok, failed = [], []
    for m in models:
        try:
            r = fit.fit_curve(x, y, m, weights=weights, **kw)
            if not np.isfinite(r.rss) or r.dof <= 0:
                raise ValueError("no residual degrees of freedom")
            ok.append(r)
        except (ValueError, RuntimeError, np.linalg.LinAlgError, OverflowError) as exc:
            failed.append({"model": m if isinstance(m, str) else m.name, "error": str(exc)})
    rows = information(ok, [r.model.name for r in ok], criterion)
    reverse = criterion == "adj_r2"
    rows.sort(key=lambda r: (r[criterion] if math.isfinite(r[criterion]) else
                             (-1e300 if reverse else 1e300)), reverse=reverse)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows + [{**f_, "rank": None} for f_ in failed]

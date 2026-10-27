"""Analyze > Missing Value Analysis (SPSS MVA).

Univariate statistics with extreme counts (outside Q1 - 1.5 IQR / Q3 + 1.5 IQR), listwise,
pairwise, EM and regression estimates of means / covariances / correlations, Little's (1988)
MCAR chi-square test (computed from the EM maximum-likelihood estimates, as SPSS and R
``naniar::mcar_test``), separate-variance (Welch) t tests by missingness indicators,
crosstabulations of categorical variables against missingness indicators, and tabulated
missing-value patterns.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, fmt_p, numeric, value_text
from rsstatchem.core.stats.wp4_util import sort_key


_ADJUST = {"normal": "normal variates", "residual": "random residuals",
           "none": "no adjustment"}


def _f(x, d: int = 2) -> str:
    return fmt(x, d)


# =========================================================================== EM and MCAR
@dataclass
class EMResult:
    mean: np.ndarray
    cov: np.ndarray            # maximum-likelihood covariance (divisor n)
    iterations: int
    converged: bool


def em_normal(data: np.ndarray, max_iter: int = 25, tol: float = 1e-4,
              mean0=None, cov0=None) -> EMResult:
    """EM estimates of the mean vector and covariance of a multivariate normal with values
    missing at random (rows that are completely missing are ignored)."""
    x = np.asarray(data, float)
    x = x[~np.isnan(x).all(axis=1)]
    n, p = x.shape
    obs = ~np.isnan(x)
    mu = np.nanmean(x, axis=0) if mean0 is None else np.asarray(mean0, float)
    if cov0 is None:
        sig = np.diag(np.nanvar(x, axis=0))
        sig[np.isnan(sig)] = 1.0
    else:
        sig = np.asarray(cov0, float)
    patterns: dict[tuple, np.ndarray] = {}
    for i, row in enumerate(obs):
        patterns.setdefault(tuple(row), []).append(i)
    patterns = {k: np.asarray(v) for k, v in patterns.items()}
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        sx = np.zeros(p)
        sxx = np.zeros((p, p))
        for pat, rows in patterns.items():
            o = np.array(pat)
            m = ~o
            xo = x[np.ix_(rows, o)]
            xhat = np.empty((len(rows), p))
            xhat[:, o] = xo
            cmiss = np.zeros((p, p))
            if m.any():
                soo = sig[np.ix_(o, o)]
                smo = sig[np.ix_(m, o)]
                coef = np.linalg.solve(soo, smo.T).T if o.any() else np.zeros((m.sum(), 0))
                xhat[:, m] = mu[m] + (xo - mu[o]) @ coef.T
                cmiss[np.ix_(m, m)] = sig[np.ix_(m, m)] - coef @ smo.T
            sx += xhat.sum(0)
            sxx += xhat.T @ xhat + len(rows) * cmiss
        mu_new = sx / n
        sig_new = sxx / n - np.outer(mu_new, mu_new)
        delta = max(np.max(np.abs(mu_new - mu)), np.max(np.abs(sig_new - sig)))
        mu, sig = mu_new, sig_new
        if delta < tol:
            converged = True
            break
    return EMResult(mu, sig, it, converged)


def little_mcar(data: np.ndarray, em: EMResult | None = None, max_iter: int = 200
                ) -> tuple[float, int, float, int]:
    """Little's MCAR test: returns (chi-square, df, p, number of missing-data patterns)."""
    x = np.asarray(data, float)
    x = x[~np.isnan(x).all(axis=1)]
    if em is None:
        em = em_normal(x, max_iter=max_iter, tol=1e-8)
    obs = ~np.isnan(x)
    p = x.shape[1]
    keys = [tuple(r) for r in obs]
    d2 = 0.0
    df = 0
    pats = sorted(set(keys))
    for pat in pats:
        o = np.array(pat)
        rows = np.array([k == pat for k in keys])
        nj = rows.sum()
        diff = x[np.ix_(rows, o)].mean(axis=0) - em.mean[o]
        d2 += nj * float(diff @ np.linalg.solve(em.cov[np.ix_(o, o)], diff))
        df += int(o.sum())
    df -= p
    pval = float(stats.chi2.sf(d2, df)) if df > 0 else float("nan")
    return float(d2), df, pval, len(pats)


# =========================================================================== MVA
def _extremes(x: np.ndarray) -> tuple[int, int]:
    x = x[~np.isnan(x)]
    if len(x) < 4:
        return 0, 0
    q1, q3 = np.percentile(x, [25, 75], method="weibull")
    iqr = q3 - q1
    return int((x < q1 - 1.5 * iqr).sum()), int((x > q3 + 1.5 * iqr).sum())


def _matrix_table(title, names, mat, decimals=3, footnotes=None, lower=True):
    rows = []
    for i, a in enumerate(names):
        rows.append([a] + [(_f(mat[i, j], decimals) if (j <= i or not lower) else "")
                           for j in range(len(names))])
    return Table(title, [""] + names, rows, footnotes=footnotes or [])


def regression_estimates(x: np.ndarray, residual: str = "normal", seed: int = 2000000
                         ) -> np.ndarray:
    """SPSS MVA regression method: each missing value predicted from the observed variables of
    its case (coefficients from the listwise means/covariance), with an optional random
    residual ("normal", "residual" = random observed residual, "none")."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float).copy()
    complete = ~np.isnan(x).any(axis=1)
    if complete.sum() < x.shape[1] + 2:
        mu = np.nanmean(x, axis=0)
        cov = pd.DataFrame(x).cov().to_numpy()
    else:
        mu = x[complete].mean(0)
        cov = np.cov(x[complete], rowvar=False)
    obs = ~np.isnan(x)
    for i in np.nonzero(~obs.all(axis=1))[0]:
        o, m = obs[i], ~obs[i]
        if o.any():
            coef = np.linalg.lstsq(cov[np.ix_(o, o)], cov[np.ix_(o, m)], rcond=None)[0]
            pred = mu[m] + (x[i, o] - mu[o]) @ coef
            resid_cov = cov[np.ix_(m, m)] - cov[np.ix_(m, o)] @ coef
        else:
            pred = mu[m].copy()
            resid_cov = cov[np.ix_(m, m)]
        if residual == "normal":
            sd = np.sqrt(np.clip(np.diag(resid_cov), 0, None))
            pred = pred + rng.normal(size=len(sd)) * sd
        elif residual == "residual" and complete.sum() > 2:
            j = rng.integers(0, complete.sum())
            donor = x[complete][j]
            sd = np.sqrt(np.clip(np.diag(resid_cov), 0, None))
            full_sd = np.sqrt(np.clip(np.diag(cov[np.ix_(m, m)]), 1e-12, None))
            pred = pred + (donor[m] - mu[m]) / full_sd * sd
        x[i, m] = pred
    return x


def mva(ds: Dataset, quantitative: list[str], categorical: list[str] | None = None,
        estimation: tuple = ("listwise", "em"), univariate: bool = True,
        ttest: bool = False, crosstabs: bool = False, threshold: float = 5.0,
        patterns: bool = True, case_patterns: bool = False, em_iter: int = 25,
        regression_residual: str = "normal", sort_patterns: bool = True,
        max_categories: int = 25) -> Section:
    categorical = list(categorical or [])
    names = list(quantitative)
    sec = Section("Missing Value Analysis")
    syn = "MVA VARIABLES=" + " ".join(names + categorical)
    if categorical:
        syn += f"\n  /CATEGORICAL={' '.join(categorical)}"
    if ttest:
        syn += f"\n  /TTEST PERCENT={threshold:g}"
    if crosstabs:
        syn += f"\n  /CROSSTAB PERCENT={threshold:g}"
    if patterns:
        syn += "\n  /TPATTERN PERCENT=1"
    syn += "\n  " + " ".join(f"/{e.upper()}" for e in estimation)
    sec.log(syn + ".")
    qx = pd.concat({n: numeric(ds, n) for n in names}, axis=1) if names else pd.DataFrame()
    cx = {c: ds.analysis_series(c) for c in categorical}
    n = ds.n_cases
    titles = names

    if univariate:
        rows = []
        for name in names:
            x = qx[name].to_numpy(float)
            valid = x[~np.isnan(x)]
            miss = n - len(valid)
            lo, hi = _extremes(x)
            rows.append([name, fmt_int(len(valid)), _f(valid.mean(), 2) if len(valid) else "",
                         _f(valid.std(ddof=1), 3) if len(valid) > 1 else "", fmt_int(miss),
                         _f(100 * miss / n, 1) if n else "", fmt_int(lo), fmt_int(hi)])
        for c in categorical:
            s = cx[c]
            miss = int(s.isna().sum())
            rows.append([c, fmt_int(n - miss), "", "", fmt_int(miss), _f(100 * miss / n, 1),
                         "", ""])
        sec.table(Table("Univariate Statistics", ["", "N", "Mean", "Std. Deviation",
                                                  ("Missing", "Count"), ("Missing", "Percent"),
                                                  ("No. of Extremesᵃ", "Low"),
                                                  ("No. of Extremesᵃ", "High")], rows,
                        footnotes=["a. Number of cases outside the range (Q1 - 1.5*IQR, Q3 + "
                                   "1.5*IQR)."]))

    xm = qx.to_numpy(float) if names else np.zeros((n, 0))
    summary_means = {}
    summary_sd = {}
    if names and "listwise" in estimation:
        lw = xm[~np.isnan(xm).any(axis=1)]
        if len(lw) > 1:
            mean = lw.mean(0)
            cov = np.cov(lw, rowvar=False).reshape(len(names), len(names))
            sd = np.sqrt(np.diag(cov))
            summary_means["Listwise"] = mean
            summary_sd["Listwise"] = sd
            sec.table(Table("Listwise Means", ["", "N"] + titles,
                            [["Listwise", fmt_int(len(lw))] + [_f(v, 4) for v in mean]]))
            sec.table(_matrix_table("Listwise Covariances", titles, cov, 3))
            sec.table(_matrix_table("Listwise Correlations", titles, np.corrcoef(lw, rowvar=False)
                                    .reshape(len(names), len(names)), 3))
        else:
            sec.note("Listwise", "Too few complete cases for listwise statistics.", "warning")
    if names and "pairwise" in estimation:
        frame = pd.DataFrame(xm, columns=names)
        counts = (~frame.isna()).astype(int).T @ (~frame.isna()).astype(int)
        sec.table(_matrix_table("Pairwise Frequencies", titles, counts.to_numpy(float), 0))
        pm = np.full((len(names), len(names)), np.nan)
        psd = np.full((len(names), len(names)), np.nan)
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                m = frame[list(dict.fromkeys([a, b]))].dropna()
                if len(m):
                    pm[i, j] = m[a].mean()
                    psd[i, j] = m[a].std(ddof=1) if len(m) > 1 else np.nan
        sec.table(Table("Pairwise Means", [""] + titles,
                        [[titles[i]] + [_f(pm[i, j], 4) for j in range(len(names))]
                         for i in range(len(names))],
                        footnotes=["Rows: variable whose mean is shown; columns: the other "
                                   "variable required to be present."]))
        sec.table(Table("Pairwise Standard Deviations", [""] + titles,
                        [[titles[i]] + [_f(psd[i, j], 4) for j in range(len(names))]
                         for i in range(len(names))]))
        sec.table(_matrix_table("Pairwise Covariances", titles, frame.cov().to_numpy(), 3))
        sec.table(_matrix_table("Pairwise Correlations", titles, frame.corr().to_numpy(), 3))
        summary_means["All Values"] = np.nanmean(xm, axis=0)
        summary_sd["All Values"] = np.nanstd(xm, axis=0, ddof=1)
    elif names:
        summary_means["All Values"] = np.nanmean(xm, axis=0)
        summary_sd["All Values"] = np.nanstd(xm, axis=0, ddof=1)
    if names and "em" in estimation:
        em = em_normal(xm, max_iter=em_iter)
        chi, df, p, _ = little_mcar(xm, em_normal(xm, max_iter=max(em_iter, 200), tol=1e-8))
        foot = f"a. Little's MCAR test: Chi-Square = {_f(chi, 3)}, DF = {df}, Sig. = {fmt_p(p)}"
        notes = [foot] + ([] if em.converged else
                          [f"b. The EM algorithm failed to converge in {em_iter} iterations."])
        summary_means["EM"] = em.mean
        summary_sd["EM"] = np.sqrt(np.diag(em.cov))
        sec.table(Table("EM Meansᵃ", [""] + titles, [[""] + [_f(v, 4) for v in em.mean]],
                        footnotes=notes))
        sec.table(_matrix_table("EM Covariancesᵃ", titles, em.cov, 3, notes))
        sd = np.sqrt(np.diag(em.cov))
        sec.table(_matrix_table("EM Correlationsᵃ", titles, em.cov / np.outer(sd, sd), 3, notes))
    if names and "regression" in estimation:
        filled = regression_estimates(xm, regression_residual)
        mean = filled.mean(0)
        cov = np.cov(filled, rowvar=False).reshape(len(names), len(names))
        summary_means["Regression"] = mean
        summary_sd["Regression"] = np.sqrt(np.diag(cov))
        sec.table(Table("Regression Meansᵃ", [""] + titles, [[""] + [_f(v, 4) for v in mean]],
                        footnotes=[f"a. Regression estimates with "
                                   f"{_ADJUST[regression_residual]} added."]))
        sec.table(_matrix_table("Regression Covariances", titles, cov, 3))
        sd = np.sqrt(np.diag(cov))
        sec.table(_matrix_table("Regression Correlations", titles, cov / np.outer(sd, sd), 3))
    if names and len(summary_means) > 1:
        rows = [[k] + [_f(v, 4) for v in m] for k, m in summary_means.items()]
        sec.table(Table("Summary of Estimated Means", [""] + titles, rows))
        rows = [[k] + [_f(v, 4) for v in m] for k, m in summary_sd.items()]
        sec.table(Table("Summary of Estimated Standard Deviations", [""] + titles, rows))

    allnames = names + categorical
    miss_ind = {nm: (qx[nm].isna() if nm in names else cx[nm].isna()) for nm in allnames}
    indicators = [nm for nm in allnames if n and 100 * miss_ind[nm].mean() >= threshold
                  and miss_ind[nm].any() and not miss_ind[nm].all()]
    if ttest and names:
        rows = []
        for ind in indicators:
            mask = miss_ind[ind].to_numpy()
            stat_rows = {k: [] for k in ("t", "df", "P(2-tail)", "# Present", "# Missing",
                                         "Mean(Present)", "Mean(Missing)")}
            for y in names:
                v = qx[y].to_numpy(float)
                a = v[~mask & ~np.isnan(v)]
                b = v[mask & ~np.isnan(v)]
                if len(a) > 1 and len(b) > 1 and y != ind:
                    t, pv = stats.ttest_ind(a, b, equal_var=False)
                    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
                    dfw = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)) \
                        if va + vb > 0 else float("nan")
                    vals = [_f(t, 1), _f(dfw, 1), fmt_p(pv)]
                else:
                    vals = [".", ".", "."]
                stat_rows["t"].append(vals[0])
                stat_rows["df"].append(vals[1])
                stat_rows["P(2-tail)"].append(vals[2])
                stat_rows["# Present"].append(fmt_int(len(a)))
                stat_rows["# Missing"].append(fmt_int(len(b)))
                stat_rows["Mean(Present)"].append(_f(a.mean(), 2) if len(a) else ".")
                stat_rows["Mean(Missing)"].append(_f(b.mean(), 2) if len(b) else ".")
            for i, (k, vals) in enumerate(stat_rows.items()):
                rows.append([ind if i == 0 else "", k] + vals)
        sec.table(Table("Separate Variance t Testsᵃ", ["", ""] + titles, rows,
                        row_header_cols=2,
                        footnotes=["For each quantitative variable, pairs of groups are formed "
                                   "by indicator variables (present, missing).",
                                   f"a. Indicator variables with less than {threshold:g}% "
                                   f"missing are not displayed."]))
    if crosstabs and categorical:
        for c in categorical:
            s = cx[c]
            cats = sorted(s.dropna().unique(), key=sort_key)[:max_categories]
            labels = [value_text(ds.var(c), v) for v in cats]
            rows = []
            for ind in indicators:
                if ind == c:
                    continue
                mask = miss_ind[ind]
                present = ~mask
                total_counts = [int((s == v).sum()) for v in cats]
                rows.append([ind, "Present", "Count", fmt_int(int(present.sum()))] +
                            [fmt_int(int(((s == v) & present).sum())) for v in cats] +
                            [fmt_int(int((s.isna() & present).sum()))])
                rows.append(["", "", "Percent", _f(100 * present.mean(), 1)] +
                            [_f(100 * ((s == v) & present).sum() / tc, 1) if tc else ""
                             for v, tc in zip(cats, total_counts)] +
                            [_f(100 * (s.isna() & present).sum() / max(s.isna().sum(), 1), 1)])
                rows.append(["", "Missing", "% SysMis", _f(100 * mask.mean(), 1)] +
                            [_f(100 * ((s == v) & mask).sum() / tc, 1) if tc else ""
                             for v, tc in zip(cats, total_counts)] +
                            [_f(100 * (s.isna() & mask).sum() / max(s.isna().sum(), 1), 1)])
            if rows:
                sec.table(Table(c, ["", "", "", "Total"] + labels + ["Missing (SysMis)"], rows,
                                row_header_cols=3,
                                footnotes=["Indicator variables with less than "
                                           f"{threshold:g}% missing are not displayed."]))
    if patterns and allnames:
        mask = pd.DataFrame({nm: miss_ind[nm] for nm in allnames})
        keys = mask.apply(lambda r: tuple(r.to_numpy()), axis=1)
        counts = keys.value_counts()
        pats = list(counts.index)
        if sort_patterns:
            pats.sort(key=lambda k: (sum(k), [not v for v in k]))
        rows = []
        for pat in pats:
            k = np.array(pat)
            cnt = int(counts[pat])
            complete_if = int(sum(counts[q] for q in counts.index
                                  if all(not a or b for a, b in zip(q, k))))
            means = []
            for nm in names:
                sel = (keys == pat).to_numpy()
                v = qx[nm].to_numpy(float)[sel]
                v = v[~np.isnan(v)]
                means.append(_f(v.mean(), 2) if len(v) else "")
            rows.append([fmt_int(cnt)] + ["X" if v else "" for v in k] +
                        [fmt_int(complete_if)] + means)
        cols = ["Number of Cases"] + [("Missing Patternsᵃ", nm) for nm in allnames] + \
            ["Complete ifᵇ"] + [("Meansᶜ", nm) for nm in names]
        sec.table(Table("Tabulated Patterns", cols, rows,
                        footnotes=["Patterns with less than 1% cases (0 or fewer) are not "
                                   "displayed." if False else
                                   "X marks a missing value in the pattern.",
                                   "a. Variables are sorted on missing patterns.",
                                   "b. Number of complete cases if variables missing in that "
                                   "pattern (marked with X) are not used.",
                                   "c. Means at each unique pattern."]))
    if case_patterns and allnames:
        rows = []
        lohi = {}
        for nm in names:
            x = qx[nm].to_numpy(float)
            v = x[~np.isnan(x)]
            if len(v) >= 4:
                q1, q3 = np.percentile(v, [25, 75], method="weibull")
                lohi[nm] = (q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))
        for i in range(n):
            flags = []
            for nm in allnames:
                if miss_ind[nm].iloc[i]:
                    flags.append("S")
                elif nm in lohi:
                    v = qx[nm].iloc[i]
                    flags.append("-" if v < lohi[nm][0] else "+" if v > lohi[nm][1] else "")
                else:
                    flags.append("")
            k = sum(f == "S" for f in flags)
            if k:
                rows.append([str(i + 1), fmt_int(k), _f(100 * k / len(allnames), 1)] + flags)
        sec.table(Table("Missing Patterns (cases with missing values)",
                        ["Case", "# Missing", "% Missing"] + allnames, rows,
                        footnotes=["- indicates an extreme low value, and + indicates an "
                                   "extreme high value. The range used is (Q1 - 1.5*IQR, Q3 + "
                                   "1.5*IQR).", "S. System-missing value"]))
    _ = math
    return sec

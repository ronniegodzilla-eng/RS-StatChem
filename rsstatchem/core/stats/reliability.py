"""Analyze > Scale > Reliability Analysis (SPSS RELIABILITY) plus the item validity table
(Pearson item-total r compared with r-table) commonly reported in Indonesian theses.

Models: Alpha (Cronbach), Split-half (Spearman-Brown equal/unequal length, Guttman split-half),
Guttman (lambda 1-6), Parallel and Strict parallel (maximum likelihood under the (strictly)
parallel model, with the goodness-of-fit chi-square) and McDonald's omega (one-factor model
fitted by maximum likelihood to the covariance matrix). Statistics: item / scale /
scale-if-item-deleted descriptives, inter-item correlation and covariance matrices, summary item
statistics, the ANOVA table with F, Friedman or Cochran test, Tukey's test of additivity,
Hotelling's T-squared, the intraclass correlation coefficient and Fleiss' multiple-rater kappa.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import optimize, stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats import agreement
from rsstatchem.core.stats.common import fmt, fmt_int, fmt_p, listwise

MODELS = {"alpha": "ALPHA", "split": "SPLIT", "guttman": "GUTTMAN", "parallel": "PARALLEL",
          "strict": "STRICTPARALLEL", "omega": "OMEGA"}


# ============================================================================ coefficients
def cronbach_alpha(items: np.ndarray) -> float:
    k = items.shape[1]
    if k < 2:
        return float("nan")
    total_var = items.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return float("nan")
    return k / (k - 1) * (1 - items.var(axis=0, ddof=1).sum() / total_var)


def standardized_alpha(items: np.ndarray) -> float:
    k = items.shape[1]
    corr = np.corrcoef(items, rowvar=False)
    rbar = (corr.sum() - k) / (k * (k - 1))
    return k * rbar / (1 + (k - 1) * rbar)


def r_table(n: int, alpha: float = 0.05) -> float:
    """Critical Pearson r (two-tailed) for df = n - 2."""
    df = n - 2
    if df < 1:
        return float("nan")
    t = stats.t.ppf(1 - alpha / 2, df)
    return t / math.sqrt(df + t * t)


def split_half(x: np.ndarray, second: int | None = None) -> dict:
    """SPSS MODEL=SPLIT(n): ``second`` items in part 2 (default: halves, odd item to part 1)."""
    k = x.shape[1]
    k2 = second if second else k // 2
    k1 = k - k2
    a, b = x[:, :k1], x[:, k1:]
    sa, sb = a.sum(1), b.sum(1)
    r = float(np.corrcoef(sa, sb)[0, 1]) if np.ptp(sa) and np.ptp(sb) else float("nan")
    equal = 2 * r / (1 + r)
    if abs(r) < 1:
        frac = k1 * k2 / k ** 2
        unequal = (math.sqrt(r ** 4 + 4 * r ** 2 * (1 - r ** 2) * frac) - r ** 2) / \
            (2 * (1 - r ** 2) * frac)
    else:
        unequal = r
    vt = x.sum(1).var(ddof=1)
    guttman = 2 * (1 - (sa.var(ddof=1) + sb.var(ddof=1)) / vt)
    return {"k1": k1, "k2": k2, "alpha1": cronbach_alpha(a) if k1 > 1 else float("nan"),
            "alpha2": cronbach_alpha(b) if k2 > 1 else float("nan"), "r": r,
            "equal": equal, "unequal": unequal, "guttman": guttman}


def guttman_lambdas(x: np.ndarray) -> list[float]:
    k = x.shape[1]
    s = np.cov(x, rowvar=False)
    vt = s.sum()
    l1 = 1 - np.trace(s) / vt
    off = s - np.diag(np.diag(s))
    l2 = l1 + math.sqrt(k / (k - 1) * (off ** 2).sum()) / vt
    l3 = k / (k - 1) * l1
    l4 = split_half(x)["guttman"]
    l5 = l1 + 2 * math.sqrt((off ** 2).sum(1).max()) / vt
    try:
        smc_err = 1 / np.diag(np.linalg.inv(s))
        l6 = 1 - smc_err.sum() / vt
    except np.linalg.LinAlgError:
        l6 = float("nan")
    return [l1, l2, l3, l4, l5, l6]


def parallel_model(x: np.ndarray, strict: bool = False) -> dict:
    """Parallel / strictly parallel model (Kristof 1963) exactly as in the IBM SPSS Statistics
    Algorithms, RELIABILITY: "Parallel Model" and "Strict Parallel" (common, true and error
    variance, common inter-item correlation, reliability, unbiased reliability and the
    goodness-of-fit chi-square with IBM's small-sample correction multiplier
    1 - k(k+1)^2(2k-3) / (12(k-1) df (W-1)))."""
    n, k = x.shape
    s = np.cov(x, rowvar=False)
    m = x.mean(0)
    var_b = float(np.trace(s) / k)                               # mean item variance
    cov_b = float((s.sum() - np.trace(s)) / (k * (k - 1)))       # mean inter-item covariance
    sign_u, logdet_u = np.linalg.slogdet(s)
    if strict:
        sd2 = float(np.sum((m - m.mean()) ** 2))                 # sum (T_i - G)^2
        common = var_b + sd2 / k
        resid = x - x.mean(1, keepdims=True)
        error = float(np.sum(resid ** 2) / (n * (k - 1)))        # MS within people
        true = common - error
        rho = (cov_b - sd2 / ((k - 1) * k)) / common if common else float("nan")
        rel = k * rho / (1 + (k - 1) * rho)
        unbiased = (3 + (n - 3) * rel) / n
        with np.errstate(divide="ignore", invalid="ignore"):
            logdet_c = math.log(var_b + (k - 1) * cov_b) + (k - 1) * math.log(
                var_b - cov_b + sd2 / k)
        dof = k * (k + 3) // 2 - 3
    else:
        common, true = var_b, cov_b
        error = var_b - cov_b
        rho = cov_b / var_b if var_b else float("nan")
        rel = k * rho / (1 + (k - 1) * rho)
        unbiased = (2 + (n - 3) * rel) / (n - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            logdet_c = (k - 1) * math.log(var_b - cov_b) + math.log(var_b + (k - 1) * cov_b)
        dof = k * (k + 1) // 2 - 2
    log_l = logdet_u - logdet_c
    corr = 1 - k * (k + 1) ** 2 * (2 * k - 3) / (12 * (k - 1) * dof * (n - 1))
    chi = -(n - 1) * corr * log_l
    return {"common": common, "true": true, "error": error, "rho": rho,
            "reliability": rel, "unbiased": unbiased,
            "chi": chi, "df": dof, "p": float(stats.chi2.sf(chi, dof)),
            "logdet_u": logdet_u if sign_u > 0 else float("nan"),
            "logdet_c": logdet_c}


def one_factor_ml(s: np.ndarray, n: int | None = None) -> dict:
    """One-factor model Sigma = l l' + diag(psi) fitted to the covariance matrix by maximum
    likelihood (Joreskog's discrepancy function)."""
    s = np.asarray(s, float)
    k = s.shape[0]
    _, logdet_s = np.linalg.slogdet(s)
    sd = np.sqrt(np.diag(s))

    def unpack(theta):
        return theta[:k], np.exp(theta[k:])

    def f(theta):
        lam, psi = unpack(theta)
        sigma = np.outer(lam, lam) + np.diag(psi)
        sign, logdet = np.linalg.slogdet(sigma)
        if sign <= 0:
            return 1e10
        return logdet + np.trace(np.linalg.solve(sigma, s)) - logdet_s - k

    def grad(theta):
        lam, psi = unpack(theta)
        sigma = np.outer(lam, lam) + np.diag(psi)
        inv = np.linalg.inv(sigma)
        g_sigma = inv - inv @ s @ inv
        return np.concatenate([2 * g_sigma @ lam, np.diag(g_sigma) * psi])

    vals, vecs = np.linalg.eigh(s)
    lam0 = vecs[:, -1] * math.sqrt(max(vals[-1], 1e-8))
    if lam0.sum() < 0:
        lam0 = -lam0
    psi0 = np.clip(np.diag(s) - lam0 ** 2, 0.05 * np.diag(s), None)
    theta0 = np.concatenate([lam0, np.log(psi0)])
    res = optimize.minimize(f, theta0, jac=grad, method="L-BFGS-B",
                            options={"maxiter": 2000, "gtol": 1e-10, "ftol": 1e-14})
    lam, psi = unpack(res.x)
    if lam.sum() < 0:
        lam = -lam
    return {"loadings": lam, "uniquenesses": psi, "std_loadings": lam / sd,
            "converged": bool(res.success), "fmin": float(res.fun)}


def mcdonald_omega(x: np.ndarray) -> float:
    fit = one_factor_ml(np.cov(x, rowvar=False))
    lam, psi = fit["loadings"], fit["uniquenesses"]
    return float(lam.sum() ** 2 / (lam.sum() ** 2 + psi.sum()))


def hotelling_t2(x: np.ndarray) -> dict:
    """Hotelling's T-squared test that all item means are equal."""
    n, k = x.shape
    c = np.eye(k)[:-1] - np.eye(k)[1:]
    d = x @ c.T
    md = d.mean(0)
    sd = np.cov(d, rowvar=False).reshape(k - 1, k - 1)
    t2 = float(n * md @ np.linalg.solve(sd, md))
    df1, df2 = k - 1, n - k + 1
    f = (n - k + 1) / ((n - 1) * (k - 1)) * t2
    return {"t2": t2, "F": f, "df1": df1, "df2": df2,
            "p": float(stats.f.sf(f, df1, df2)) if df2 > 0 else float("nan")}


def tukey_additivity(x: np.ndarray) -> dict:
    n, k = x.shape
    grand = x.mean()
    ri, cj = x.mean(1) - grand, x.mean(0) - grand
    den = (ri ** 2).sum() * (cj ** 2).sum()
    num = (x * np.outer(ri, cj)).sum()
    ss_n = num ** 2 / den if den > 0 else float("nan")
    a = agreement.anova_two_way(x)
    ss_bal = a["ss_err"] - ss_n
    df_bal = (n - 1) * (k - 1) - 1
    f = ss_n / (ss_bal / df_bal) if df_bal > 0 and ss_bal > 0 else float("nan")
    power = 1 - grand * (num / den) if den > 0 else float("nan")
    return {"ss": ss_n, "F": f, "df_bal": df_bal, "ss_bal": ss_bal,
            "p": float(stats.f.sf(f, 1, df_bal)) if np.isfinite(f) else float("nan"),
            "power": power}


# ============================================================================ procedure
def reliability(ds: Dataset, names: list[str], item_stats: bool = False,
                scale_stats: bool = False, item_deleted: bool = True,
                inter_item: bool = False, validity: bool = False,
                alpha_level: float = 0.05, scale_label: str = "ALL VARIABLES", *,
                model: str = "alpha", split: int | None = None, inter_cov: bool = False,
                summaries=(), anova: str = "none", tukey: bool = False,
                hotelling: bool = False, icc: dict | None = None,
                fleiss: dict | None = None) -> Section:
    """RELIABILITY. ``summaries`` ⊂ {"means","variances","covariances","correlations"};
    ``anova`` in {"none","f","friedman","cochran"}; ``icc`` = {"model", "type", "level",
    "test_value"}; ``fleiss`` = {"individual": bool, "level": float}."""
    section = Section("Reliability")
    model = model if model in MODELS else "alpha"
    model_kw = MODELS[model] + (f"({split})" if model == "split" and split else "")
    stat_kw = [kw for kw, on in (("DESCRIPTIVE", item_stats), ("SCALE", scale_stats),
                                 ("CORR", inter_item), ("COV", inter_cov),
                                 ("TUKEY", tukey), ("HOTELLING", hotelling),
                                 ("ANOVA" if anova == "f" else "FRIEDMAN" if anova == "friedman"
                                  else "COCHRAN", anova != "none")) if on]
    log = (f"RELIABILITY\n  /VARIABLES={' '.join(names)}\n  /SCALE('{scale_label}') ALL"
           f"\n  /MODEL={model_kw}")
    if stat_kw:
        log += f"\n  /STATISTICS={' '.join(stat_kw)}"
    summ = [s.upper() for s in ("means", "variances", "covariances", "correlations")
            if s in summaries]
    if item_deleted or summ:
        log += f"\n  /SUMMARY={' '.join((['TOTAL'] if item_deleted else []) + summ)}"
    if icc:
        mdl = {"oneway": "ONEWAY", "twoway_random": "RANDOM",
               "twoway_mixed": "MIXED"}[icc.get("model", "twoway_mixed")]
        log += (f"\n  /ICC=MODEL({mdl}) TYPE({icc.get('type', 'consistency').upper()[:11]}) "
                f"CIN={icc.get('level', 0.95) * 100:g} TESTVAL={icc.get('test_value', 0):g}")
    section.log(log + ".")
    titles = [ds.var(n).title for n in names]
    total = ds.n_cases
    frame = listwise(ds, names)
    n = len(frame)
    section.table(Table("Case Processing Summary", ["", "", "N", "%"], [
        ["Cases", "Valid", fmt_int(n), f"{100 * n / total:.1f}" if total else ""],
        ["", "Excludedᵃ", fmt_int(total - n),
         f"{100 * (total - n) / total:.1f}" if total else ""],
        ["", "Total", fmt_int(total), "100.0"]], row_header_cols=2,
        footnotes=["a. Listwise deletion based on all variables in the procedure."]))
    if n < 2 or len(names) < 2:
        section.note("Warning", "At least two items and two valid cases are needed.",
                     style="warning")
        return section

    x = frame.to_numpy(float)
    k = x.shape[1]
    alpha = cronbach_alpha(x)
    if model == "alpha":
        section.table(Table("Reliability Statistics", [
            "Cronbach's Alpha", "Cronbach's Alpha Based on Standardized Items", "N of Items"],
            [[fmt(alpha), fmt(standardized_alpha(x)), fmt_int(k)]], row_header_cols=0))
        verdict = ("Excellent" if alpha >= 0.9 else "Good" if alpha >= 0.8 else
                   "Acceptable" if alpha >= 0.7 else "Questionable" if alpha >= 0.6 else
                   "Poor" if alpha >= 0.5 else "Unacceptable")
        section.note("", f"Cronbach's alpha = {fmt(alpha)} ({verdict}). The scale is commonly "
                         f"considered reliable when alpha > .60 (or > .70).")
    elif model == "split":
        sh = split_half(x, split)
        part1, part2 = titles[:sh["k1"]], titles[sh["k1"]:]
        section.table(Table("Reliability Statistics", ["", "", "", ""], [
            ["Cronbach's Alpha", "Part 1", "Value", fmt(sh["alpha1"])],
            ["", "", "N of Items", f"{sh['k1']}ᵃ"],
            ["", "Part 2", "Value", fmt(sh["alpha2"])],
            ["", "", "N of Items", f"{sh['k2']}ᵇ"],
            ["", "Total N of Items", "", fmt_int(k)],
            ["Correlation Between Forms", "", "", fmt(sh["r"])],
            ["Spearman-Brown Coefficient", "Equal Length", "", fmt(sh["equal"])],
            ["", "Unequal Length", "", fmt(sh["unequal"])],
            ["Guttman Split-Half Coefficient", "", "", fmt(sh["guttman"])]],
            row_header_cols=3, footnotes=[f"a. The items are: {', '.join(part1)}.",
                                          f"b. The items are: {', '.join(part2)}."]))
    elif model == "guttman":
        lam = guttman_lambdas(x)
        rows = [["Lambda", str(i + 1), fmt(v)] for i, v in enumerate(lam)]
        rows = [[r[0] if i == 0 else "", r[1], r[2]] for i, r in enumerate(rows)]
        rows.append(["N of Items", "", fmt_int(k)])
        section.table(Table("Reliability Statistics", ["", "", ""], rows, row_header_cols=2))
    elif model in ("parallel", "strict"):
        pm = parallel_model(x, strict=model == "strict")
        word = "strictly parallel" if model == "strict" else "parallel"
        section.table(Table("Test for Goodness of Fit of Model", ["", "", ""], [
            ["Chi-Square", "Value", fmt(pm["chi"])], ["", "df", fmt_int(pm["df"])],
            ["", "Sig", fmt_p(pm["p"])],
            ["Log of Determinant of", "Unconstrained Matrix", fmt(pm["logdet_u"])],
            ["", "Constrained Matrix", fmt(pm["logdet_c"])]], row_header_cols=2,
            footnotes=[f"Under the {word} model assumption"]))
        rows = [["Common Mean", fmt(x.mean())]] if model == "strict" else []
        rows += [["Common Variance", fmt(pm["common"])], ["True Variance", fmt(pm["true"])],
                 ["Error Variance", fmt(pm["error"])],
                 ["Common Inter-Item Correlation", fmt(pm["rho"])],
                 ["Reliability of Scale", fmt(pm["reliability"])],
                 ["Reliability of Scale (Unbiased)", fmt(pm["unbiased"])]]
        section.table(Table("Reliability Statistics", ["", ""], rows))
    else:
        omega = mcdonald_omega(x)
        section.table(Table("Reliability Statistics", ["McDonald's Omega", "N of Items"],
                            [[fmt(omega), fmt_int(k)]], row_header_cols=0))
        fit = one_factor_ml(np.cov(x, rowvar=False))
        section.table(Table("Factor Loadings (One-Factor Model)", [
            "", "Loading", "Standardized Loading", "Unique Variance"],
            [[t, fmt(lv), fmt(sl), fmt(u)] for t, lv, sl, u in zip(
                titles, fit["loadings"], fit["std_loadings"], fit["uniquenesses"])],
            footnotes=["Maximum likelihood estimates of a single-factor model without error "
                       "covariances. Omega = (Σ loadings)² / ((Σ loadings)² + Σ unique "
                       "variances)."]))

    if item_stats:
        rows = [[t, fmt(x[:, j].mean(), 4), fmt(x[:, j].std(ddof=1), 5), fmt_int(n)]
                for j, t in enumerate(titles)]
        section.table(Table("Item Statistics", ["", "Mean", "Std. Deviation", "N"], rows))
    corr = np.corrcoef(x, rowvar=False)
    cov = np.cov(x, rowvar=False)
    if inter_item:
        rows = [[t] + [fmt(corr[i, j]) for j in range(k)] for i, t in enumerate(titles)]
        section.table(Table("Inter-Item Correlation Matrix", [""] + titles, rows))
    if inter_cov:
        rows = [[t] + [fmt(cov[i, j]) for j in range(k)] for i, t in enumerate(titles)]
        section.table(Table("Inter-Item Covariance Matrix", [""] + titles, rows))
    if summaries:
        iu = np.triu_indices(k, 1)
        spec = [("means", "Item Means", x.mean(0)),
                ("variances", "Item Variances", x.var(0, ddof=1)),
                ("covariances", "Inter-Item Covariances", cov[iu]),
                ("correlations", "Inter-Item Correlations", corr[iu])]
        rows = []
        for key, label, v in spec:
            if key not in summaries:
                continue
            ratio = v.max() / v.min() if v.min() != 0 else float("nan")
            rows.append([label, fmt(v.mean()), fmt(v.min()), fmt(v.max()),
                         fmt(v.max() - v.min()), fmt(ratio),
                         fmt(v.var(ddof=1)) if len(v) > 1 else "", fmt_int(k)])
        section.table(Table("Summary Item Statistics", [
            "", "Mean", "Minimum", "Maximum", "Range", "Maximum / Minimum", "Variance",
            "N of Items"], rows))

    total_score = x.sum(axis=1)
    if item_deleted:
        rows = []
        smc = None
        if inter_item:
            try:
                smc = 1 - 1 / np.diag(np.linalg.inv(corr))
            except np.linalg.LinAlgError:
                smc = None
        last = "McDonald's Omega if Item Deleted" if model == "omega" else \
            "Cronbach's Alpha if Item Deleted"
        for j, t in enumerate(titles):
            rest = total_score - x[:, j]
            r_corr = np.corrcoef(x[:, j], rest)[0, 1] if np.ptp(x[:, j]) and np.ptp(rest) \
                else float("nan")
            reduced = np.delete(x, j, axis=1)
            if model == "omega":
                a_del = mcdonald_omega(reduced) if k > 2 else float("nan")
            else:
                a_del = cronbach_alpha(reduced) if k > 2 else float("nan")
            row = [t, fmt(rest.mean(), 4), fmt(rest.var(ddof=1), 3), fmt(r_corr)]
            if smc is not None:
                row.append(fmt(smc[j]))
            rows.append(row + [fmt(a_del)])
        cols = ["", "Scale Mean if Item Deleted", "Scale Variance if Item Deleted",
                "Corrected Item-Total Correlation"]
        if smc is not None:
            cols.append("Squared Multiple Correlation")
        section.table(Table("Item-Total Statistics", cols + [last], rows))
    if scale_stats:
        section.table(Table("Scale Statistics", ["Mean", "Variance", "Std. Deviation",
                                                 "N of Items"],
                            [[fmt(total_score.mean(), 4), fmt(total_score.var(ddof=1), 3),
                              fmt(total_score.std(ddof=1), 5), fmt_int(k)]], row_header_cols=0))
    if anova != "none":
        _anova_table(section, x, anova, tukey)
    elif tukey:
        _anova_table(section, x, "f", True)
    if hotelling:
        h = hotelling_t2(x)
        section.table(Table("Hotelling's T-Squared Test", [
            "Hotelling's T-Squared", "F", "df1", "df2", "Sig"],
            [[fmt(h["t2"]), fmt(h["F"]), fmt_int(h["df1"]), fmt_int(h["df2"]), fmt_p(h["p"])]],
            row_header_cols=0, footnotes=[f"Grand Mean = {fmt(x.mean(), 4)}"]))
    if icc:
        section.table(agreement.icc_table(x, icc.get("model", "twoway_mixed"),
                                          icc.get("type", "consistency"),
                                          icc.get("level", 0.95), icc.get("test_value", 0.0)))
    if fleiss is not None:
        raw = np.column_stack([ds.analysis_series(nm).to_numpy(dtype=object) for nm in names])
        keep = np.array([all(v == v and v is not None for v in row) for row in raw])
        raw = raw[keep]
        try:
            cats = sorted({v for v in raw.ravel()}, key=lambda v: (isinstance(v, str), v))
            var0 = ds.var(names[0])
            from rsstatchem.core.stats.common import value_text

            labels = [value_text(var0, c) for c in cats]
            for tab in agreement.fleiss_tables(raw, labels, fleiss.get("level", 0.95),
                                               fleiss.get("individual", True)):
                section.table(tab)
        except ValueError as exc:
            section.note("Fleiss' Kappa", str(exc), style="warning")
    if validity:
        crit = r_table(n, alpha_level)
        rows = []
        for j, t in enumerate(titles):
            if np.ptp(x[:, j]) == 0:
                rows.append([t, "", "", fmt(crit), "", "Constant item"])
                continue
            r, p = stats.pearsonr(x[:, j], total_score)
            rest = total_score - x[:, j]
            rc = np.corrcoef(x[:, j], rest)[0, 1] if np.ptp(rest) else float("nan")
            valid = r > crit and p < alpha_level
            rows.append([t, fmt(r), fmt_p(p), fmt(crit), fmt(rc),
                         "Valid" if valid else "Not valid"])
        section.table(Table("Item Validity (Pearson Item-Total Correlation)", [
            "Item", "r (item-total)", "Sig. (2-tailed)",
            f"r table (α = {alpha_level:g}, df = {n - 2})",
            "Corrected r", "Decision"], rows,
            footnotes=["An item is valid when r (item-total) > r table and Sig. < "
                       f"{alpha_level:g}. Corrected r excludes the item from the total."]))
    return section


def _anova_table(section: Section, x: np.ndarray, test: str, tukey: bool) -> None:
    n, k = x.shape
    a = agreement.anova_two_way(x)
    ms_err = a["mse"]
    rows = [["Between People", "", fmt(a["ss_rows"]), fmt_int(n - 1), fmt(a["msr"]), "", ""]]
    notes = [f"Grand Mean = {fmt(a['grand'], 4)}"]
    if test == "friedman":
        chi, p = stats.friedmanchisquare(*x.T)
        w = chi / (n * (k - 1))
        stat, sig = fmt(chi), fmt_p(p)
        head, title = "Friedman's Chi-Square", "ANOVA with Friedman's Test"
        notes.append(f"Kendall's coefficient of concordance W = {fmt(w)}.")
    elif test == "cochran":
        values = np.unique(x)
        if len(values) > 2:
            section.note("ANOVA with Cochran's Test", "Cochran's Q requires dichotomous items.",
                         style="warning")
            return
        b = (x == values.max()).astype(float)
        c, r = b.sum(0), b.sum(1)
        tot = b.sum()
        den = k * tot - (r ** 2).sum()
        q = (k - 1) * (k * (c ** 2).sum() - tot ** 2) / den if den else float("nan")
        stat, sig = fmt(q), fmt_p(stats.chi2.sf(q, k - 1))
        head, title = "Cochran's Q", "ANOVA with Cochran's Test"
    else:
        f = a["msc"] / ms_err if ms_err > 0 else float("nan")
        stat, sig = fmt(f), fmt_p(stats.f.sf(f, k - 1, (n - 1) * (k - 1)))
        head, title = "F", "ANOVA"
    rows.append(["Within People", "Between Items", fmt(a["ss_cols"]), fmt_int(k - 1),
                 fmt(a["msc"]), stat, sig])
    df_err = (n - 1) * (k - 1)
    if tukey:
        tk = tukey_additivity(x)
        rows = [r[:2] + [""] + r[2:] for r in rows]
        rows += [["", "Residual", "Nonadditivity", fmt(tk["ss"]), "1", fmt(tk["ss"]),
                  fmt(tk["F"]) + "ᵃ", fmt_p(tk["p"])],
                 ["", "", "Balance", fmt(tk["ss_bal"]), fmt_int(tk["df_bal"]),
                  fmt(tk["ss_bal"] / tk["df_bal"]) if tk["df_bal"] > 0 else "", "", ""],
                 ["", "", "Total", fmt(a["ss_err"]), fmt_int(df_err), fmt(ms_err), "", ""]]
        notes.append("a. Tukey's estimate of power to which observations must be raised to "
                     f"achieve additivity = {fmt(tk['power'], 3)}.")
    else:
        rows.append(["", "Residual", fmt(a["ss_err"]), fmt_int(df_err), fmt(ms_err), "", ""])
    tail = [["", "Total", fmt(a["ss_within"]), fmt_int(n * (k - 1)),
             fmt(a["ss_within"] / (n * (k - 1))), "", ""],
            ["Total", "", fmt(a["ss_total"]), fmt_int(n * k - 1),
             fmt(a["ss_total"] / (n * k - 1)), "", ""]]
    rh = 3 if tukey else 2
    rows += [r[:2] + [""] + r[2:] for r in tail] if tukey else tail
    section.table(Table(title, [""] * rh + ["Sum of Squares", "df", "Mean Square", head, "Sig"],
                        rows, row_header_cols=rh, footnotes=notes))

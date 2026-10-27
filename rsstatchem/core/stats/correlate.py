"""Analyze > Correlate: Bivariate (Pearson, Kendall's tau-b, Spearman) and Partial.

Bivariate also has the SPSS 27.0.1 Confidence Interval sub-dialog (Fisher r-to-z, optional
bias adjustment for Pearson; Fieller-Hartley-Pearson, Bonett-Wright or Caruso-Cliff standard
errors for the rank correlations) and the cross-product deviations / covariances option.
Partial adds means and standard deviations and pairwise deletion ("exclude cases pairwise").
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, fmt_p, listwise, numeric, pct, stars

RANK_CI_METHODS = {"fieller": "Fieller, Hartley, and Pearson",
                   "bonett": "Bonett and Wright", "caruso": "Caruso and Cliff"}


def _pair(x: pd.Series, y: pd.Series, method: str) -> tuple[float, float, int]:
    frame = pd.concat([x, y], axis=1).dropna()
    n = len(frame)
    a, b = frame.iloc[:, 0].to_numpy(float), frame.iloc[:, 1].to_numpy(float)
    if n < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan"), float("nan"), n
    if method == "pearson":
        r, p = stats.pearsonr(a, b)
    elif method == "spearman":
        r, p = stats.spearmanr(a, b)
    else:
        r, p = stats.kendalltau(a, b, variant="b")
    return float(r), float(p), n


def correlation_ci(r: float, n: int, method: str = "pearson", level: float = 0.95,
                   bias: bool = False, rank_se: str = "fieller") -> tuple[float, float]:
    """SPSS CORRELATIONS /CI: Fisher z interval with the standard error of each method."""
    if not (abs(r) < 1) or n < 5:
        return float("nan"), float("nan")
    z = math.atanh(r)
    if method == "pearson":
        if bias:
            z -= r / (2 * (n - 1))
        se = 1 / math.sqrt(n - 3)
    elif method == "spearman":
        se = {"fieller": math.sqrt(1.06 / (n - 3)),
              "bonett": math.sqrt((1 + r * r / 2) / (n - 3)),
              "caruso": math.sqrt(1 / (n - 2) + abs(z) / (6 * n + 4 * math.sqrt(n)))}[rank_se]
    else:
        se = math.sqrt(0.437 / (n - 4))
    q = stats.norm.ppf(0.5 + level / 2)
    return math.tanh(z - q * se), math.tanh(z + q * se)


def _footnotes(flagged: set[str], tails: str) -> list[str]:
    notes = []
    if "**" in flagged:
        notes.append(f"**. Correlation is significant at the 0.01 level ({tails}).")
    if "*" in flagged:
        notes.append(f"*. Correlation is significant at the 0.05 level ({tails}).")
    return notes


def bivariate(ds: Dataset, names: list[str], pearson: bool = True, kendall: bool = False,
              spearman: bool = False, two_tailed: bool = True, flag: bool = True,
              descriptives: bool = False, listwise_deletion: bool = False,
              cross_products: bool = False, ci: bool = False, ci_level: float = 0.95,
              bias_adjust: bool = False, rank_se: str = "fieller") -> Section:
    section = Section("Correlations")
    tails = "2-tailed" if two_tailed else "1-tailed"
    methods = [m for m, on in (("pearson", pearson), ("kendall", kendall),
                               ("spearman", spearman)) if on]
    section.log(f"CORRELATIONS\n  /VARIABLES={' '.join(names)}\n"
                f"  /PRINT={'TWOTAIL' if two_tailed else 'ONETAIL'} "
                f"{'SIG' if flag else 'NOSIG'}\n"
                f"  /MISSING={'LISTWISE' if listwise_deletion else 'PAIRWISE'}\n"
                f"  /METHODS={' '.join(m.upper() for m in methods)}"
                + (f"\n  /CI CILEVEL({ci_level * 100:g})"
                   + (" BIASADJUST" if bias_adjust else "") if ci else "")
                + ("\n  /STATISTICS XPROD" if cross_products else "") + ".")
    if listwise_deletion:
        frame = listwise(ds, names)
        data = {n: frame[n] for n in names}
    else:
        data = {n: numeric(ds, n) for n in names}
    titles = {n: ds.var(n).title for n in names}

    if descriptives:
        rows = []
        for n in names:
            x = data[n].dropna()
            d = ds.var(n).decimals
            rows.append([titles[n], fmt(x.mean(), d + 2), fmt(x.std(ddof=1), d + 3),
                         fmt_int(len(x))])
        section.table(Table("Descriptive Statistics", ["", "Mean", "Std. Deviation", "N"], rows))

    def cell(a, b, method):
        if a == b:
            return "1" if method == "pearson" else "1.000", "", fmt_int(data[a].notna().sum())
        r, p, n = _pair(data[a], data[b], method)
        if not two_tailed:
            p = p / 2
        mark = stars(p) if flag else ""
        return fmt(r) + mark, fmt_p(p), fmt_int(n), mark

    if pearson:
        rows, flagged = [], set()
        for a in names:
            vals = [cell(a, b, "pearson") for b in names]
            flagged |= {v[3] for v in vals if len(v) > 3 and v[3]}
            rows.append([titles[a], "Pearson Correlation"] + [v[0] for v in vals])
            rows.append(["", f"Sig. ({tails})"] + [v[1] for v in vals])
            if cross_products:
                ss, cv = [], []
                for b in names:
                    fr = pd.concat([data[a], data[b]], axis=1).dropna()
                    xa, xb = fr.iloc[:, 0].to_numpy(float), fr.iloc[:, 1].to_numpy(float)
                    sp = float(np.sum((xa - xa.mean()) * (xb - xb.mean())))
                    ss.append(fmt(sp))
                    cv.append(fmt(sp / (len(xa) - 1)) if len(xa) > 1 else "")
                rows.append(["", "Sum of Squares and Cross-products"] + ss)
                rows.append(["", "Covariance"] + cv)
            rows.append(["", "N"] + [v[2] for v in vals])
        section.table(Table("Correlations", ["", ""] + [titles[n] for n in names], rows,
                            row_header_cols=2, footnotes=_footnotes(flagged, tails)))

    nonpar = [m for m in ("kendall", "spearman") if m in methods]
    if nonpar:
        rows, flagged = [], set()
        for m in nonpar:
            for ai, a in enumerate(names):
                vals = [cell(a, b, m) for b in names]
                flagged |= {v[3] for v in vals if len(v) > 3 and v[3]}
                head = ("Kendall's tau_b" if m == "kendall" else "Spearman's rho") \
                    if ai == 0 else ""
                rows.append([head, titles[a], "Correlation Coefficient"] + [v[0] for v in vals])
                rows.append(["", "", f"Sig. ({tails})"] + [v[1] for v in vals])
                rows.append(["", "", "N"] + [v[2] for v in vals])
        section.table(Table("Correlations", ["", "", ""] + [titles[n] for n in names], rows,
                            row_header_cols=3, footnotes=_footnotes(flagged, tails)))
    if ci:
        for m in methods:
            _ci_table(section, data, names, titles, m, ci_level, bias_adjust, rank_se)
    return section


def _ci_table(section, data, names, titles, method, level, bias, rank_se) -> None:
    head = {"pearson": "Pearson Correlation", "spearman": "Spearman's rho",
            "kendall": "Kendall's tau_b"}[method]
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            r, p, n = _pair(data[a], data[b], method)
            lo, hi = correlation_ci(r, n, method, level, bias, rank_se)
            rows.append([f"{titles[a]} - {titles[b]}", fmt(r), fmt_p(p), fmt(lo), fmt(hi)])
    g = f"{pct(level)} Confidence Intervals (2-tailed)ᵃ"
    notes = ["a. Estimation is based on Fisher's r-to-z transformation"
             + (" with bias adjustment." if bias and method == "pearson" else ".")]
    if method == "spearman":
        notes.append(f"b. Estimation of standard error is based on the formula proposed by "
                     f"{RANK_CI_METHODS[rank_se]}.")
    elif method == "kendall":
        notes.append("b. Estimation of standard error is based on the formula proposed by "
                     "Fieller, Hartley, and Pearson.")
    title = "Confidence Intervals" if method == "pearson" else \
        f"Confidence Intervals of {head}"
    section.table(Table(title, ["", head, "Sig. (2-tailed)", (g, "Lower"), (g, "Upper")],
                        rows, footnotes=notes))


def partial(ds: Dataset, names: list[str], controls: list[str], two_tailed: bool = True,
            flag: bool = True, zero_order: bool = False, descriptives: bool = False,
            listwise_deletion: bool = True) -> Section:
    section = Section("Partial Corr")
    tails = "2-tailed" if two_tailed else "1-tailed"
    section.log(f"PARTIAL CORR\n  /VARIABLES={' '.join(names)} BY {' '.join(controls)}\n"
                f"  /SIGNIFICANCE={'TWOTAIL' if two_tailed else 'ONETAIL'}\n"
                + ("  /STATISTICS=" + " ".join((["DESCRIPTIVES"] if descriptives else [])
                                               + (["CORR"] if zero_order else [])) + "\n"
                   if descriptives or zero_order else "")
                + f"  /MISSING={'LISTWISE' if listwise_deletion else 'ANALYSIS'}.")
    allv = list(dict.fromkeys(names + controls))
    if listwise_deletion:
        frame = listwise(ds, allv)
    else:
        frame = pd.DataFrame({v: numeric(ds, v) for v in allv})
    if descriptives:
        rows = []
        for v in allv:
            x = frame[v].dropna()
            d = ds.var(v).decimals
            rows.append([ds.var(v).title, fmt(x.mean(), d + 2), fmt(x.std(ddof=1), d + 3),
                         fmt_int(len(x))])
        section.table(Table("Descriptive Statistics", ["", "Mean", "Std. Deviation", "N"],
                            rows))
    R = frame[allv].corr(min_periods=2).to_numpy(float)
    valid = frame[allv].notna().astype(float)
    counts = (valid.T @ valid).to_numpy()
    pos = {v: i for i, v in enumerate(allv)}

    def pcorr(a: str, b: str, ctrl: list[str]) -> tuple[float, int]:
        idx = [pos[a], pos[b]] + [pos[c] for c in ctrl]
        sub = R[np.ix_(idx, idx)]
        n = int(counts[np.ix_(idx, idx)].min())
        if not ctrl:
            return float(sub[0, 1]), n - 2
        try:
            P = np.linalg.inv(sub)
        except np.linalg.LinAlgError:
            return float("nan"), n - 2 - len(ctrl)
        return float(-P[0, 1] / math.sqrt(P[0, 0] * P[1, 1])), n - 2 - len(ctrl)

    rows = []
    flagged_all = set()
    sets = ([("-none-ᵃ", [])] if zero_order else []) + \
        [(" & ".join(ds.var(c).title for c in controls), controls)]
    width = len(allv) if zero_order else len(names)
    for title, ctrl in sets:
        dset = names if ctrl else allv
        for vi, a in enumerate(dset):
            rvals, pvals, dvals = [], [], []
            for b in dset:
                if a == b:
                    rvals.append("1.000")
                    pvals.append("")
                    dvals.append("0")
                    continue
                r, dfree = pcorr(a, b, ctrl)
                t = r * math.sqrt(dfree / (1 - r * r)) if dfree > 0 and abs(r) < 1 \
                    else float("nan")
                p = 2 * stats.t.sf(abs(t), dfree) if dfree > 0 else float("nan")
                if not two_tailed:
                    p /= 2
                mark = stars(p) if flag else ""
                flagged_all.add(mark)
                rvals.append(fmt(r) + mark)
                pvals.append(fmt_p(p))
                dvals.append(fmt_int(dfree))
            pad = [""] * (width - len(dset))
            rows.append([title if vi == 0 else "", ds.var(a).title, "Correlation"] + rvals
                        + pad)
            rows.append(["", "", f"Significance ({tails})"] + pvals + pad)
            rows.append(["", "", "df"] + dvals + pad)
    notes = _footnotes({f for f in flagged_all if f}, tails)
    if zero_order:
        notes.append("a. Cells contain zero-order (Pearson) correlations.")
    heads = [ds.var(v).title for v in (allv if zero_order else names)]
    section.table(Table("Correlations", ["Control Variables", "", ""] + heads, rows,
                        row_header_cols=3, footnotes=notes))
    return section

"""Analyze > Scale > Reliability Analysis (Cronbach's alpha) plus the item validity table
(Pearson item-total r compared with r-table) commonly reported in Indonesian theses."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, fmt_p, listwise


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


def reliability(ds: Dataset, names: list[str], item_stats: bool = False,
                scale_stats: bool = False, item_deleted: bool = True,
                inter_item: bool = False, validity: bool = False,
                alpha_level: float = 0.05, scale_label: str = "ALL VARIABLES") -> Section:
    section = Section("Reliability")
    section.log(f"RELIABILITY\n  /VARIABLES={' '.join(names)}\n  /SCALE('{scale_label}') ALL"
                f"\n  /MODEL=ALPHA" + ("\n  /SUMMARY=TOTAL" if item_deleted else "") + ".")
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
    std_alpha = standardized_alpha(x)
    section.table(Table("Reliability Statistics", ["Cronbach's Alpha",
                                                   "Cronbach's Alpha Based on Standardized Items",
                                                   "N of Items"],
                        [[fmt(alpha), fmt(std_alpha), fmt_int(k)]], row_header_cols=0))
    verdict = ("Excellent" if alpha >= 0.9 else "Good" if alpha >= 0.8 else
               "Acceptable" if alpha >= 0.7 else "Questionable" if alpha >= 0.6 else
               "Poor" if alpha >= 0.5 else "Unacceptable")
    section.note("", f"Cronbach's alpha = {fmt(alpha)} ({verdict}). The scale is commonly "
                     f"considered reliable when alpha > .60 (or > .70).")

    if item_stats:
        rows = [[t, fmt(x[:, j].mean(), 4), fmt(x[:, j].std(ddof=1), 5), fmt_int(n)]
                for j, t in enumerate(titles)]
        section.table(Table("Item Statistics", ["", "Mean", "Std. Deviation", "N"], rows))
    if inter_item:
        corr = np.corrcoef(x, rowvar=False)
        rows = [[t] + [fmt(corr[i, j]) for j in range(k)] for i, t in enumerate(titles)]
        section.table(Table("Inter-Item Correlation Matrix", [""] + titles, rows))

    total_score = x.sum(axis=1)
    if item_deleted:
        rows = []
        for j, t in enumerate(titles):
            rest = total_score - x[:, j]
            r_corr = np.corrcoef(x[:, j], rest)[0, 1] if np.ptp(x[:, j]) and np.ptp(rest) \
                else float("nan")
            a_del = cronbach_alpha(np.delete(x, j, axis=1)) if k > 2 else float("nan")
            rows.append([t, fmt(rest.mean(), 4), fmt(rest.var(ddof=1), 3), fmt(r_corr),
                         fmt(a_del)])
        section.table(Table("Item-Total Statistics", ["", "Scale Mean if Item Deleted",
                                                      "Scale Variance if Item Deleted",
                                                      "Corrected Item-Total Correlation",
                                                      "Cronbach's Alpha if Item Deleted"], rows))
    if scale_stats:
        section.table(Table("Scale Statistics", ["Mean", "Variance", "Std. Deviation",
                                                 "N of Items"],
                            [[fmt(total_score.mean(), 4), fmt(total_score.var(ddof=1), 3),
                              fmt(total_score.std(ddof=1), 5), fmt_int(k)]], row_header_cols=0))
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

"""Analyze > Descriptive Statistics > Crosstabs: contingency table, chi-square tests and
nominal measures of association."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, fmt_p, value_text

CELLS = ("observed", "expected", "row", "column", "total")


def _categories(series: pd.Series) -> list:
    return sorted(series.unique(), key=lambda v: (isinstance(v, str), v))


def crosstabs(ds: Dataset, rows: list[str], columns: list[str], chisq: bool = True,
              phi: bool = True, contingency: bool = False, cells=("observed",)) -> Section:
    section = Section("Crosstabs")
    requested = [s for s, on in (("CHISQ", chisq), ("PHI", phi), ("CC", contingency)) if on]
    section.log(f"CROSSTABS\n  /TABLES={' '.join(rows)} BY {' '.join(columns)}\n"
                f"  /STATISTICS={' '.join(requested) or 'NONE'}\n"
                f"  /CELLS={' '.join(c.upper() for c in cells)}.")
    for rname in rows:
        for cname in columns:
            _one(ds, section, rname, cname, chisq, phi, contingency, cells)
    return section


def _one(ds, section, rname, cname, chisq, phi, contingency, cells) -> None:
    rvar, cvar = ds.var(rname), ds.var(cname)
    raw = pd.DataFrame({"r": ds.analysis_series(rname), "c": ds.analysis_series(cname)})
    total = len(raw)
    frame = raw.dropna()
    n = len(frame)
    section.table(Table("Case Processing Summary", [
        "", ("Valid", "N"), ("Valid", "Percent"), ("Missing", "N"), ("Missing", "Percent"),
        ("Total", "N"), ("Total", "Percent")], [[
            f"{rvar.title} * {cvar.title}", fmt_int(n), _pct(n, total), fmt_int(total - n),
            _pct(total - n, total), fmt_int(total), "100.0%"]]))
    if n == 0:
        return
    rcats, ccats = _categories(frame["r"]), _categories(frame["c"])
    obs = np.array([[int(((frame["r"] == r) & (frame["c"] == c)).sum()) for c in ccats]
                    for r in rcats], float)
    row_tot, col_tot = obs.sum(axis=1), obs.sum(axis=0)
    expected = np.outer(row_tot, col_tot) / n

    labels = {"observed": "Count", "expected": "Expected Count",
              "row": f"% within {rvar.title}", "column": f"% within {cvar.title}",
              "total": "% of Total"}
    shown = [c for c in ("observed", "expected", "row", "column", "total") if c in cells] or \
        ["observed"]

    def values(i, j):
        o = obs[i, j] if i < len(rcats) and j < len(ccats) else None
        rt = row_tot[i] if i < len(rcats) else n
        ct = col_tot[j] if j < len(ccats) else n
        if i == len(rcats) and j == len(ccats):
            o_val = n
        elif i == len(rcats):
            o_val = col_tot[j]
        elif j == len(ccats):
            o_val = row_tot[i]
        else:
            o_val = o
        e_val = rt * ct / n
        return {"observed": fmt_int(o_val), "expected": fmt(e_val, 1),
                "row": _pct(o_val, rt), "column": _pct(o_val, ct), "total": _pct(o_val, n)}

    table_rows = []
    for i in range(len(rcats) + 1):
        head = value_text(rvar, rcats[i]) if i < len(rcats) else "Total"
        for k, key in enumerate(shown):
            cells_ = [values(i, j)[key] for j in range(len(ccats) + 1)]
            first_col = rvar.title if i == 0 and k == 0 else ""
            table_rows.append([first_col if i < len(rcats) else ("Total" if k == 0 else ""),
                               head if k == 0 and i < len(rcats) else "",
                               labels[key]] + cells_)
    cols = ["", "", ""] + [(cvar.title, value_text(cvar, c)) for c in ccats] + ["Total"]
    section.table(Table(f"{rvar.title} * {cvar.title} Crosstabulation", cols, table_rows,
                        row_header_cols=3))

    if min(len(rcats), len(ccats)) < 2:
        section.note("", "No statistics are computed because at least one variable is constant.")
        return
    dof = (len(rcats) - 1) * (len(ccats) - 1)
    chi, p_chi, _, _ = stats.chi2_contingency(obs, correction=False)
    if chisq:
        two_by_two = obs.shape == (2, 2)
        crow = [["Pearson Chi-Square", fmt(chi) + "ᵃ", fmt_int(dof), fmt_p(p_chi), "", ""]]
        if two_by_two:
            cc_chi, cc_p, _, _ = stats.chi2_contingency(obs, correction=True)
            crow.append(["Continuity Correctionᵇ", fmt(cc_chi), fmt_int(dof), fmt_p(cc_p), "",
                         ""])
        lr, lr_p, _, _ = stats.chi2_contingency(obs, correction=False, lambda_="log-likelihood")
        crow.append(["Likelihood Ratio", fmt(lr), fmt_int(dof), fmt_p(lr_p), "", ""])
        if two_by_two:
            _, fisher_two = stats.fisher_exact(obs, alternative="two-sided")
            one = min(stats.fisher_exact(obs, alternative="less")[1],
                      stats.fisher_exact(obs, alternative="greater")[1])
            crow.append(["Fisher's Exact Test", "", "", "", fmt_p(fisher_two), fmt_p(one)])
        rnum = pd.Series([float(rcats.index(v)) for v in frame["r"]])
        cnum = pd.Series([float(ccats.index(v)) for v in frame["c"]])
        if pd.api.types.is_numeric_dtype(frame["r"]) and \
                pd.api.types.is_numeric_dtype(frame["c"]):
            rnum, cnum = frame["r"].astype(float), frame["c"].astype(float)
        r = np.corrcoef(rnum, cnum)[0, 1] if np.ptp(rnum) and np.ptp(cnum) else float("nan")
        lbl = (n - 1) * r * r
        crow.append(["Linear-by-Linear Association", fmt(lbl), "1",
                     fmt_p(stats.chi2.sf(lbl, 1)), "", ""])
        crow.append(["N of Valid Cases", fmt_int(n), "", "", "", ""])
        small = int((expected < 5).sum())
        notes = [f"a. {small} cells ({100 * small / expected.size:.1f}%) have expected count "
                 f"less than 5. The minimum expected count is {expected.min():.2f}."]
        if two_by_two:
            notes.append("b. Computed only for a 2x2 table")
        section.table(Table("Chi-Square Tests", ["", "Value", "df",
                                                 "Asymptotic Significance (2-sided)",
                                                 "Exact Sig. (2-sided)", "Exact Sig. (1-sided)"],
                            crow, footnotes=notes))
    if phi or contingency:
        mrows = []
        k = min(len(rcats), len(ccats))
        if phi:
            phi_v = math.sqrt(chi / n)
            if obs.shape == (2, 2):
                a, b, c, d = obs.ravel()
                denom = math.sqrt(row_tot[0] * row_tot[1] * col_tot[0] * col_tot[1])
                phi_v = (a * d - b * c) / denom if denom else float("nan")
            mrows.append(["Nominal by Nominal", "Phi", fmt(phi_v), fmt_p(p_chi)])
            mrows.append(["", "Cramer's V", fmt(math.sqrt(chi / (n * (k - 1)))), fmt_p(p_chi)])
        if contingency:
            mrows.append(["" if mrows else "Nominal by Nominal", "Contingency Coefficient",
                          fmt(math.sqrt(chi / (chi + n))), fmt_p(p_chi)])
        mrows.append(["N of Valid Cases", "", fmt_int(n), ""])
        section.table(Table("Symmetric Measures", ["", "", "Value", "Approximate Significance"],
                            mrows, row_header_cols=2))


def _pct(part, whole) -> str:
    return f"{100.0 * part / whole:.1f}%" if whole else ""

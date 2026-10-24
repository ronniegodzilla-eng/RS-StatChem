"""Analyze > Correlate: Bivariate (Pearson, Kendall's tau-b, Spearman) and Partial."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, fmt_p, listwise, numeric, stars


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


def _footnotes(flagged: set[str], tails: str) -> list[str]:
    notes = []
    if "**" in flagged:
        notes.append(f"**. Correlation is significant at the 0.01 level ({tails}).")
    if "*" in flagged:
        notes.append(f"*. Correlation is significant at the 0.05 level ({tails}).")
    return notes


def bivariate(ds: Dataset, names: list[str], pearson: bool = True, kendall: bool = False,
              spearman: bool = False, two_tailed: bool = True, flag: bool = True,
              descriptives: bool = False, listwise_deletion: bool = False) -> Section:
    section = Section("Correlations")
    tails = "2-tailed" if two_tailed else "1-tailed"
    methods = [m for m, on in (("pearson", pearson), ("kendall", kendall),
                               ("spearman", spearman)) if on]
    section.log(f"CORRELATIONS\n  /VARIABLES={' '.join(names)}\n"
                f"  /PRINT={'TWOTAIL' if two_tailed else 'ONETAIL'} "
                f"{'SIG' if flag else 'NOSIG'}\n"
                f"  /MISSING={'LISTWISE' if listwise_deletion else 'PAIRWISE'}\n"
                f"  /METHODS={' '.join(m.upper() for m in methods)}.")
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
    return section


def partial(ds: Dataset, names: list[str], controls: list[str], two_tailed: bool = True,
            flag: bool = True, zero_order: bool = False) -> Section:
    section = Section("Partial Corr")
    tails = "2-tailed" if two_tailed else "1-tailed"
    section.log(f"PARTIAL CORR\n  /VARIABLES={' '.join(names)} BY {' '.join(controls)}\n"
                f"  /SIGNIFICANCE={'TWOTAIL' if two_tailed else 'ONETAIL'}\n  /MISSING=LISTWISE.")
    frame = listwise(ds, names + controls)
    n = len(frame)
    k = len(controls)
    df = n - 2 - k
    titles = [ds.var(v).title for v in names]

    def residualize(cols: list[str]) -> pd.DataFrame:
        if not cols:
            return frame[names] - frame[names].mean()
        X = np.column_stack([np.ones(n), frame[cols].to_numpy(float)])
        beta, *_ = np.linalg.lstsq(X, frame[names].to_numpy(float), rcond=None)
        return pd.DataFrame(frame[names].to_numpy(float) - X @ beta, columns=names)

    def block(resid: pd.DataFrame, dfree: int):
        out = []
        flagged = set()
        corr = np.corrcoef(resid.to_numpy(float), rowvar=False) if len(resid) > 1 else None
        for i, a in enumerate(names):
            rvals, pvals, dvals = [], [], []
            for j, _ in enumerate(names):
                if i == j:
                    rvals.append("1.000")
                    pvals.append("")
                    dvals.append("0")
                    continue
                r = float(corr[i, j]) if corr is not None else float("nan")
                t = r * math.sqrt(dfree / (1 - r * r)) if dfree > 0 and abs(r) < 1 \
                    else float("nan")
                p = 2 * stats.t.sf(abs(t), dfree) if dfree > 0 else float("nan")
                if not two_tailed:
                    p /= 2
                mark = stars(p) if flag else ""
                flagged.add(mark)
                rvals.append(fmt(r) + mark)
                pvals.append(fmt_p(p))
                dvals.append(fmt_int(dfree))
            out.append((a, rvals, pvals, dvals))
        return out, flagged

    rows = []
    flagged_all = set()
    sets = ([("-none-ᵃ", [], n - 2)] if zero_order else []) + \
        [(" & ".join(ds.var(c).title for c in controls), controls, df)]
    for title, cols, dfree in sets:
        blk, flagged = block(residualize(cols), dfree)
        flagged_all |= flagged
        for vi, (a, rvals, pvals, dvals) in enumerate(blk):
            rows.append([title if vi == 0 else "", titles[vi], "Correlation"] + rvals)
            rows.append(["", "", f"Significance ({tails})"] + pvals)
            rows.append(["", "", "df"] + dvals)
    notes = _footnotes({f for f in flagged_all if f}, tails)
    if zero_order:
        notes.append("a. Cells contain zero-order (Pearson) correlations.")
    section.table(Table("Correlations", ["Control Variables", "", ""] + titles, rows,
                        row_header_cols=3, footnotes=notes))
    return section

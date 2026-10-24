"""Analyze > ROC Curve (SPSS ROC): area under the curve with its standard error (nonparametric
DeLong or bi-negative exponential Hanley-McNeil), asymptotic significance against AUC = .5,
asymptotic confidence interval, coordinates of the curve and the ROC plot.

Also holds the small matplotlib helpers shared by the logistic/classify modules.
"""
from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy import stats

from rsstatchem.core.dataset import Dataset, format_code
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, fmt_p, numeric, parse_value, pct

PALETTE = ["#C62828", "#1565C0", "#2E7D32", "#EF6C00", "#6A1B9A", "#00838F", "#5D4037",
           "#AD1457", "#455A64", "#9E9D24", "#283593"]


def figure(width: float = 6.0, height: float = 4.2) -> Figure:
    fig = Figure(figsize=(width, height), dpi=120)
    FigureCanvasAgg(fig)
    return fig


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=9)


def png(fig: Figure) -> bytes:
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    return buf.getvalue()


# ---------------------------------------------------------------------------- statistics
def auc_details(pos: np.ndarray, neg: np.ndarray) -> dict:
    """AUC = P(X_pos > X_neg) + .5 P(tie), with DeLong placement values."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    n1, n0 = len(pos), len(neg)
    psi = (pos[:, None] > neg[None, :]).astype(float) + 0.5 * (pos[:, None] == neg[None, :])
    auc = float(psi.mean())
    v10, v01 = psi.mean(axis=1), psi.mean(axis=0)
    var_delong = (v10.var(ddof=1) / n1 if n1 > 1 else 0) + (v01.var(ddof=1) / n0 if n0 > 1
                                                            else 0)
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    var_hm = (auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n0 - 1) * (q2 - auc ** 2)) / \
        (n1 * n0)
    var_null = (n1 + n0 + 1) / (12 * n1 * n0)
    ties = bool(np.isin(pos, neg).any())
    return {"auc": auc, "se_delong": math.sqrt(max(var_delong, 0)),
            "se_hm": math.sqrt(max(var_hm, 0)), "se_null": math.sqrt(var_null), "n1": n1,
            "n0": n0, "ties": ties}


def roc_points(pos: np.ndarray, neg: np.ndarray, larger_positive: bool = True):
    """SPSS cutoffs: min - 1, midpoints of consecutive distinct values, max + 1."""
    allv = np.unique(np.r_[pos, neg])
    cuts = np.r_[allv[0] - 1, (allv[:-1] + allv[1:]) / 2, allv[-1] + 1]
    if larger_positive:
        sens = np.array([(pos >= c).mean() for c in cuts])
        fpr = np.array([(neg >= c).mean() for c in cuts])
    else:
        sens = np.array([(pos <= c).mean() for c in cuts])
        fpr = np.array([(neg <= c).mean() for c in cuts])
    return cuts, sens, fpr


def roc(ds: Dataset, tests: list[str], state: str, positive, larger_positive: bool = True,
        distribution: str = "free", plot: bool = True, reference_line: bool = True,
        se_ci: bool = True, coordinates: bool = True, level: float = 0.95,
        youden: bool = False) -> Section:
    """``distribution`` is "free" (nonparametric, DeLong) or "negexpo" (Hanley-McNeil)."""
    svar = ds.var(state)
    pos_value = parse_value(svar, positive) if isinstance(positive, str) else positive
    section = Section("ROC Curve")
    plot_kw = ("CURVE(REFERENCE)" if reference_line else "CURVE") if plot else "NONE"
    section.log(f"ROC {' '.join(tests)} BY {state} ({format_code(pos_value)})\n  /PLOT="
                + plot_kw
                + "\n  /PRINT=" + " ".join(["SE"] * se_ci + ["COORDINATES"] * coordinates)
                + "\n  /CRITERIA=CUTOFF(INCLUDE) TESTPOS("
                + ("LARGE" if larger_positive else "SMALL") + ")"
                  f" DISTRIBUTION({distribution.upper()}) CI({level * 100:g})"
                + "\n  /MISSING=EXCLUDE.")
    frame = pd.DataFrame({"__s": ds.analysis_series(state), **{t: numeric(ds, t)
                                                               for t in tests}}).dropna()
    is_pos = (frame["__s"] == pos_value).to_numpy()
    n1, n0 = int(is_pos.sum()), int((~is_pos).sum())
    section.table(Table("Case Processing Summary", [svar.title, "Valid N (listwise)"], [
        ["Positiveᵃ", fmt_int(n1)], ["Negative", fmt_int(n0)]], footnotes=[
        ("Larger" if larger_positive else "Smaller") + " values of the test result "
        "variable(s) indicate stronger evidence for a positive actual state.",
        f"a. The positive actual state is {svar.label_for(pos_value) or format_code(pos_value)}."
    ]))
    if n1 == 0 or n0 == 0:
        section.note("Warning", "Both positive and negative cases are needed: check the value "
                     "of the state variable.", style="warning")
        return section
    z = stats.norm.ppf(0.5 + level / 2)
    arows, ties, curves, crows, yrows = [], [], [], [], []
    for t in tests:
        x = frame[t].to_numpy(float)
        pos, neg = x[is_pos], x[~is_pos]
        sp, sn = (pos, neg) if larger_positive else (-pos, -neg)
        d = auc_details(sp, sn)
        se = d["se_delong"] if distribution == "free" else d["se_hm"]
        p = 2 * stats.norm.sf(abs(d["auc"] - 0.5) / d["se_null"])
        lo, hi = max(0.0, d["auc"] - z * se), min(1.0, d["auc"] + z * se)
        title = ds.var(t).title
        row = [title, fmt(d["auc"])]
        if se_ci:
            row += [fmt(se), fmt_p(p), fmt(lo), fmt(hi)]
        arows.append(row)
        if d["ties"]:
            ties.append(title)
        cuts, sens, fpr = roc_points(pos, neg, larger_positive)
        curves.append((title, fpr, sens))
        if coordinates:
            dec = ds.var(t).decimals + 2
            for i in range(len(cuts)):
                j = i
                crows.append([title if j == 0 else "", fmt(cuts[i], dec), fmt(sens[i]),
                              fmt(fpr[i])])
        if youden:
            j = sens - fpr
            best = int(np.argmax(j))
            yrows.append([title, fmt(cuts[best], ds.var(t).decimals + 2), fmt(sens[best]),
                          fmt(1 - fpr[best]), fmt(j[best])])
    cols = ["Test Result Variable(s)", "Area"]
    if se_ci:
        g = f"Asymptotic {pct(level)} Confidence Interval"
        cols += ["Std. Errorᵃ", "Asymptotic Sig.ᵇ", (g, "Lower Bound"), (g, "Upper Bound")]
    notes = [f"The test result variable(s): {', '.join(ties)} has at least one tie between the "
             "positive actual state group and the negative actual state group. Statistics may "
             "be biased."] if ties else []
    if se_ci:
        notes += ["a. Under the nonparametric assumption" if distribution == "free" else
                  "a. Under the bi-negative exponential assumption",
                  "b. Null hypothesis: true area = 0.5"]
    section.table(Table("Area Under the Curve", cols, arows, footnotes=notes))
    if plot:
        fig = figure(5.4, 5.0)
        ax = fig.add_subplot()
        for k, (title, fpr, sens) in enumerate(curves):
            order = np.lexsort((sens, fpr))
            ax.plot(fpr[order], sens[order], color=PALETTE[k % len(PALETTE)], lw=1.6,
                    label=title)
        if reference_line:
            ax.plot([0, 1], [0, 1], color="#888888", lw=1, label="Reference Line")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        style(ax, "ROC Curve", "1 - Specificity", "Sensitivity")
        if len(curves) > 1 or reference_line:
            ax.legend(fontsize=8, loc="lower right", title="Source of the Curve",
                      title_fontsize=8, frameon=False)
        section.image("ROC Curve", png(fig), width=460)
        section.note("", "Diagonal segments are produced by ties.")
    if coordinates:
        section.table(Table("Coordinates of the Curve", [
            "Test Result Variable(s)",
            "Positive if Greater Than or Equal Toᵃ" if larger_positive
            else "Positive if Less Than or Equal Toᵃ", "Sensitivity", "1 - Specificity"],
            crows, footnotes=[
                "a. The smallest cutoff value is the minimum observed test value minus 1, and "
                "the largest cutoff value is the maximum observed test value plus 1. All the "
                "other cutoff values are the averages of two consecutive ordered observed test "
                "values."]))
    if youden:
        section.table(Table("Optimal Cut-off (Youden Index)", [
            "Test Result Variable(s)", "Cut-off", "Sensitivity", "Specificity",
            "Youden's J"], yrows, footnotes=[
                "Youden's J = Sensitivity + Specificity - 1 (not part of SPSS ROC output)."]))
    return section

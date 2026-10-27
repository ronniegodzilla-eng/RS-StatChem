"""Analyze > Classify > ROC Analysis (SPSS 26+ ROC ANALYSIS).

Several test variables against one state variable: areas under the ROC curve (DeLong or
bi-negative exponential standard errors), paired-sample comparison of the AUCs with the
DeLong-DeLong-Clarke-Pearson covariance, independent-group comparison of each AUC across the
two levels of a grouping variable, precision-recall curves with their areas (average
precision), classifier evaluation metrics at the Youden cut-off (sensitivity, specificity,
precision, F1, Gini index, maximum Kolmogorov-Smirnov statistic) and the curves.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from rsstatchem.core.dataset import Dataset, format_code
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, fmt_p, numeric, parse_value, pct, \
    value_text
from rsstatchem.core.stats.roc import PALETTE, auc_details, figure, png, roc_points, style


def placements(pos: np.ndarray, neg: np.ndarray):
    psi = (pos[:, None] > neg[None, :]).astype(float) + 0.5 * (pos[:, None] == neg[None, :])
    return psi.mean(), psi.mean(axis=1), psi.mean(axis=0)


def delong_paired(tests_pos: list[np.ndarray], tests_neg: list[np.ndarray]):
    """AUCs and their DeLong covariance matrix for tests measured on the same cases."""
    aucs, V10, V01 = [], [], []
    for p, ng in zip(tests_pos, tests_neg):
        a, v10, v01 = placements(p, ng)
        aucs.append(a)
        V10.append(v10)
        V01.append(v01)
    V10, V01 = np.array(V10), np.array(V01)
    n1, n0 = V10.shape[1], V01.shape[1]
    S = np.cov(V10) / n1 + np.cov(V01) / n0
    return np.array(aucs), np.atleast_2d(S)


def pr_curve(pos: np.ndarray, neg: np.ndarray):
    """Precision-recall points over descending cut-offs and the average precision."""
    scores = np.r_[pos, neg]
    labels = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    order = np.argsort(-scores, kind="mergesort")
    s, lab = scores[order], labels[order]
    distinct = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(lab)[distinct]
    fp = (distinct + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / len(pos)
    ap = float(np.sum(np.diff(np.r_[0, recall]) * precision))
    return recall, precision, ap


def roc_analysis(ds: Dataset, tests: list[str], state: str, positive,
                 group: str | None = None, larger_positive: bool = True,
                 distribution: str = "free", level: float = 0.95, plot: bool = True,
                 pr: bool = True, paired: bool = True, metrics: bool = True,
                 coordinates: bool = False) -> Section:
    svar = ds.var(state)
    pos_value = parse_value(svar, positive) if isinstance(positive, str) else positive
    section = Section("ROC Analysis")
    section.log(f"ROC ANALYSIS {' '.join(tests)} BY {state} ({format_code(pos_value)})"
                + (f"\n  /GROUPBY={group}" if group else "")
                + "\n  /PLOT=CURVE" + (" PRC" if pr else "")
                + "\n  /PRINT=" + " ".join(["PAIRED"] * paired + ["INDEPENDENT"] * bool(group)
                                          + ["CLASSIFIER"] * metrics
                                          + ["COORDINATES"] * coordinates)
                + f"\n  /CRITERIA=CUTOFF(INCLUDE) TESTPOS({'LARGE' if larger_positive else 'SMALL'})"
                  f" DISTRIBUTION({distribution.upper()}) CI({level * 100:g})"
                  "\n  /MISSING=EXCLUDE.")
    cols = {"__s": ds.analysis_series(state), **{t: numeric(ds, t) for t in tests}}
    if group:
        cols["__g"] = ds.analysis_series(group)
    frame = pd.DataFrame(cols).dropna()
    is_pos = (frame["__s"] == pos_value).to_numpy()
    n1, n0 = int(is_pos.sum()), int((~is_pos).sum())
    section.table(Table("Case Processing Summary", [svar.title, "Valid N (listwise)"], [
        ["Positiveᵃ", fmt_int(n1)], ["Negative", fmt_int(n0)]], footnotes=[
        ("Larger" if larger_positive else "Smaller") + " values of the test result "
        "variable(s) indicate stronger evidence for a positive actual state.",
        f"a. The positive actual state is {svar.label_for(pos_value) or format_code(pos_value)}."
    ]))
    if n1 == 0 or n0 == 0:
        section.note("Warning", "Both positive and negative cases are needed.", style="warning")
        return section
    z = stats.norm.ppf(0.5 + level / 2)
    sign = 1 if larger_positive else -1
    P = [sign * frame.loc[is_pos, t].to_numpy(float) for t in tests]
    N = [sign * frame.loc[~is_pos, t].to_numpy(float) for t in tests]
    rows = []
    for t, p_, n_ in zip(tests, P, N):
        d = auc_details(p_, n_)
        se = d["se_delong"] if distribution == "free" else d["se_hm"]
        sig = 2 * stats.norm.sf(abs(d["auc"] - 0.5) / d["se_null"])
        rows.append([ds.var(t).title, fmt(d["auc"]), fmt(se), fmt_p(sig),
                     fmt(max(0, d["auc"] - z * se)), fmt(min(1, d["auc"] + z * se))])
    g = f"Asymptotic {pct(level)} Confidence Interval"
    section.table(Table("Area Under the ROC Curve", ["Test Result Variable(s)", "Area",
                                                     "Std. Errorᵃ", "Asymptotic Sig.ᵇ",
                                                     (g, "Lower Bound"), (g, "Upper Bound")],
                        rows, footnotes=["a. Under the nonparametric assumption"
                                         if distribution == "free" else
                                         "a. Under the bi-negative exponential assumption",
                                         "b. Null hypothesis: true area = 0.5"]))
    if plot:
        fig = figure(5.4, 5.0)
        ax = fig.add_subplot()
        for k, (t, p_, n_) in enumerate(zip(tests, P, N)):
            _, sens, fpr = roc_points(p_, n_, True)
            o = np.lexsort((sens, fpr))
            ax.plot(fpr[o], sens[o], color=PALETTE[k % len(PALETTE)], lw=1.6,
                    label=ds.var(t).title)
        ax.plot([0, 1], [0, 1], color="#888888", lw=1, label="Reference Line")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        style(ax, "ROC Curve", "1 - Specificity", "Sensitivity")
        ax.legend(fontsize=8, loc="lower right", frameon=False, title="Source of the Curve",
                  title_fontsize=8)
        section.image("ROC Curve", png(fig), width=460)
    if paired and len(tests) > 1:
        aucs, S = delong_paired(P, N)
        prow = []
        for i in range(len(tests)):
            for j in range(i + 1, len(tests)):
                diff = aucs[i] - aucs[j]
                se = math.sqrt(max(S[i, i] + S[j, j] - 2 * S[i, j], 0))
                zz = diff / se if se > 0 else float("nan")
                prow.append([f"{ds.var(tests[i]).title} - {ds.var(tests[j]).title}", fmt(zz),
                             fmt_p(2 * stats.norm.sf(abs(zz))), fmt(diff), fmt(se),
                             fmt(diff - z * se), fmt(diff + z * se)])
        g2 = f"Asymptotic {pct(level)} Confidence Interval"
        section.table(Table("Paired-Sample Area Difference Under the ROC Curves", [
            "Test Result Pair(s)", "Asymptotic z", "Asymptotic Sig. (2-tail)ᵃ",
            "AUC Difference", "Std. Error Differenceᵇ", (g2, "Lower Bound"),
            (g2, "Upper Bound")], prow, footnotes=[
            "a. Null hypothesis: true area difference = 0",
            "b. Under the nonparametric assumption (DeLong, DeLong and Clarke-Pearson, 1988)"]))
    if group:
        levels = sorted(frame["__g"].unique(), key=lambda v: (isinstance(v, str), v))
        if len(levels) >= 2:
            gv = ds.var(group)
            irow = []
            for t in tests:
                vals = []
                for lv in levels[:2]:
                    m = (frame["__g"] == lv).to_numpy()
                    p_ = sign * frame.loc[is_pos & m, t].to_numpy(float)
                    n_ = sign * frame.loc[~is_pos & m, t].to_numpy(float)
                    if len(p_) and len(n_):
                        d = auc_details(p_, n_)
                        vals.append((d["auc"], d["se_delong"] if distribution == "free"
                                     else d["se_hm"]))
                if len(vals) == 2:
                    diff = vals[0][0] - vals[1][0]
                    se = math.sqrt(vals[0][1] ** 2 + vals[1][1] ** 2)
                    zz = diff / se if se > 0 else float("nan")
                    irow.append([ds.var(t).title, f"{value_text(gv, levels[0])} - "
                                                  f"{value_text(gv, levels[1])}", fmt(zz),
                                 fmt_p(2 * stats.norm.sf(abs(zz))), fmt(diff), fmt(se),
                                 fmt(diff - z * se), fmt(diff + z * se)])
            g3 = f"Asymptotic {pct(level)} Confidence Interval"
            section.table(Table("Independent-Group Area Difference Under the ROC Curves", [
                "Test Result Variable(s)", gv.title, "Asymptotic z",
                "Asymptotic Sig. (2-tail)ᵃ", "AUC Difference", "Std. Error Difference",
                (g3, "Lower Bound"), (g3, "Upper Bound")], irow, row_header_cols=2,
                footnotes=["a. Null hypothesis: true area difference = 0"]
                + (["Only the first two levels of the grouping variable are compared."]
                   if len(levels) > 2 else [])))
    if pr:
        fig = figure(5.4, 4.6)
        ax = fig.add_subplot()
        arow = []
        for k, (t, p_, n_) in enumerate(zip(tests, P, N)):
            rec, prec, ap = pr_curve(p_, n_)
            ax.step(np.r_[0, rec], np.r_[prec[0], prec], where="post",
                    color=PALETTE[k % len(PALETTE)], lw=1.5, label=ds.var(t).title)
            arow.append([ds.var(t).title, fmt(ap)])
        ax.axhline(n1 / (n1 + n0), color="#888888", lw=1, ls="--", label="No skill")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        style(ax, "Precision-Recall Curve", "Recall", "Precision")
        ax.legend(fontsize=8, frameon=False)
        section.image("Precision-Recall Curve", png(fig), width=460)
        section.table(Table("Area Under the Precision-Recall Curve",
                            ["Test Result Variable(s)", "Area"], arow,
                            footnotes=["Area computed as the average precision (step "
                                       "interpolation)."]))
    if metrics:
        mrows = []
        for t, p_, n_ in zip(tests, P, N):
            cuts, sens, fpr = roc_points(p_, n_, True)
            j = sens - fpr
            b = int(np.argmax(j))
            tp, fn = sens[b] * len(p_), (1 - sens[b]) * len(p_)
            fp, tn = fpr[b] * len(n_), (1 - fpr[b]) * len(n_)
            prec = tp / (tp + fp) if tp + fp else float("nan")
            f1 = 2 * prec * sens[b] / (prec + sens[b]) if prec + sens[b] else float("nan")
            auc = auc_details(p_, n_)["auc"]
            mrows.append([ds.var(t).title, fmt(sign * cuts[b], ds.var(t).decimals + 2),
                          fmt(sens[b]), fmt(1 - fpr[b]), fmt(prec), fmt(f1),
                          fmt((tp + tn) / (tp + tn + fp + fn)), fmt(2 * auc - 1),
                          fmt(float(np.max(j)))])
        section.table(Table("Classifier Evaluation Metrics", [
            "Test Result Variable(s)", "Cut-offᵃ", "Sensitivity (Recall)", "Specificity",
            "Precision", "F1 Score", "Accuracy", "Gini Index", "Max K-S"], mrows, footnotes=[
            "a. Cut-off maximising Youden's J (sensitivity + specificity - 1); test is "
            "positive when the value is " + (">=" if larger_positive else "<=")
            + " the cut-off.", "Gini Index = 2 x AUC - 1. Max K-S = max(sensitivity - "
                               "(1 - specificity))."]))
    if coordinates:
        crows = []
        for t, p_, n_ in zip(tests, P, N):
            cuts, sens, fpr = roc_points(p_, n_, True)
            for i in range(len(cuts)):
                crows.append([ds.var(t).title if i == 0 else "",
                              fmt(sign * cuts[i], ds.var(t).decimals + 2), fmt(sens[i]),
                              fmt(fpr[i])])
        section.table(Table("Coordinates of the Curve", [
            "Test Result Variable(s)", "Positive if Greater Than or Equal To" if larger_positive
            else "Positive if Less Than or Equal To", "Sensitivity", "1 - Specificity"],
            crows))
    return section


def selftest() -> None:
    rng = np.random.default_rng(4)
    n = 120
    s = rng.integers(0, 2, n).astype(float)
    ds = Dataset(pd.DataFrame({"s": s, "a": s + rng.normal(size=n),
                               "b": 0.5 * s + rng.normal(size=n),
                               "g": rng.integers(1, 3, n).astype(float)}))
    roc_analysis(ds, ["a", "b"], "s", 1.0, group="g", coordinates=True)

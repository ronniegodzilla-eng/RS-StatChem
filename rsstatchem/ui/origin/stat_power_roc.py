"""Origin's Statistics > ROC Curve and Power and Sample Size, with report sheets (no Qt).

**ROC Curve**: one or more test variables against a state column (the positive value marks
the diseased / positive cases). Area under the curve with the DeLong standard error and
confidence limits, the asymptotic test of AUC = 0.5 (null SE of Hanley & McNeil), the
Youden-optimal cut-off, pairwise DeLong comparisons of correlated curves and the curve
coordinates (:mod:`rsstatchem.core.stats.roc` / :mod:`roc_analysis`).

**Power and Sample Size**: Origin's eight procedures — one-sample, two-sample and paired
t-tests, one and two proportions, one and two variances, one-way ANOVA. The t, ANOVA and
proportion designs are those of :mod:`rsstatchem.core.stats.power` (noncentral t and F;
exact binomial or normal approximation); the variance designs use the exact χ² and F
distributions. Compute power for given sample sizes or the sample size reaching given
powers; the report has the power table and a power curve.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
from scipy import stats

from rsstatchem.core.analysis.common import num, pval
from rsstatchem.core.graph.themes import PALETTES
from rsstatchem.core.stats import power as pw
from rsstatchem.core.stats.roc import auc_details, roc_points
from rsstatchem.core.stats.roc_analysis import delong_paired
from rsstatchem.ui.origin import report_graphs as rg
from rsstatchem.ui.origin import stat_tests as st
from rsstatchem.ui.origin.model import ReportSheet, to_float
from rsstatchem.ui.origin.stat_extra import (_report_book, aligned, new_report, report_text,
                                             target_book)

StatError = st.StatError
PAL = PALETTES.get("Origin Classic")


# ====================================================================== ROC
def roc_report(books: dict, settings: dict, report: ReportSheet | None = None,
               mode: str = "manual"):
    tests = st.read_ranges(books, settings.get("ranges", []))
    state = st.read_range(books, settings.get("state", ""))
    positive = str(settings.get("positive", "1")).strip()
    larger = settings.get("direction", "larger") == "larger"
    level = float(settings.get("level", 0.95))
    m, ok = aligned(tests)
    n = min(len(m), len(state.texts))
    ok = ok[:n] & np.array([state.texts[i] != "" for i in range(n)])
    rows = np.flatnonzero(ok)

    def is_pos(t):
        a, b = to_float(t), to_float(positive)
        return a == b if np.isfinite(a) and np.isfinite(b) else t.strip() == positive
    pos = np.array([is_pos(state.texts[i]) for i in rows])
    X = m[rows]
    if pos.sum() == 0 or (~pos).sum() == 0:
        raise StatError(f"The state column needs positive (“{positive}”) and negative cases.")
    first = tests[0]
    report, fresh = new_report(first, f"ROC Curve of {first.text}" +
                               (" …" if len(tests) > 1 else ""), "ROCCurve", report, "roc",
                               settings, mode)
    st._notes(fresh, "ROC Curve", tests + [state],
              [["Positive State Value", positive],
               ["Direction", "Larger test result indicates positive" if larger else
                "Smaller test result indicates positive"]])
    fresh.add_table("Case Processing Summary", ["", "Valid N"],
                    [["Positive", str(int(pos.sum()))], ["Negative", str(int((~pos).sum()))]])
    z = stats.norm.ppf(0.5 + level / 2)
    area, cut_rows, coords = [], [], []
    g = rg.new_graph("ROC Curve", 10.0, 9.5)
    rg.add_series(g, [0, 1], [0, 1], "Reference Line", "line", "#808080", dash="--",
                  width=1.0, legend=False)
    lines, pos_list, neg_list = [], [], []
    for k, r in enumerate(tests):
        v = X[:, k] if larger else -X[:, k]
        a, b = v[pos], v[~pos]
        pos_list.append(a)
        neg_list.append(b)
        det = auc_details(a, b)
        auc, se = det["auc"], det["se_delong"]
        zz = (auc - 0.5) / det["se_null"]
        p = float(2 * stats.norm.sf(abs(zz)))
        area.append([r.title, num(auc, 5), num(se, 5), pval(p), num(max(0, auc - z * se), 5),
                     num(min(1, auc + z * se), 5)])
        cuts, sens, fpr = roc_points(X[pos, k], X[~pos, k], larger)
        j = int(np.argmax(sens - fpr))
        cut_rows.append([r.title, num(cuts[j]), num(sens[j], 5), num(1 - fpr[j], 5),
                         num(sens[j] - fpr[j], 5)])
        for c, s_, f_ in zip(cuts, sens, fpr):
            coords.append([r.title, num(c), num(s_, 5), num(f_, 5)])
        order = np.lexsort((sens, fpr))
        rg.add_series(g, fpr[order], sens[order], r.title, "line", PAL[k % len(PAL)],
                      width=1.6)
        lines.append(f"  {r.title}: AUC = {num(auc, 5)} (SE {num(se, 4)}), p = {pval(p)}")
    fresh.add_table("Area Under the Curve", ["Test Variable", "Area", "Standard Error",
                                             "Asymptotic Prob", f"Lower {level * 100:g}% CL",
                                             f"Upper {level * 100:g}% CL"], area,
                    ["Standard error and limits by DeLong's method; the test of Area = 0.5 "
                     "uses the null standard error."])
    fresh.add_table("Optimal Cutoff (Youden Index)", ["Test Variable", "Cutoff",
                                                      "Sensitivity", "Specificity",
                                                      "Youden J"], cut_rows,
                    ["Positive if the test result is " + ("≥" if larger else "≤") +
                     " the cutoff."])
    if len(tests) > 1:
        aucs, S = delong_paired(pos_list, neg_list)
        comp = []
        for i in range(len(tests)):
            for j2 in range(i + 1, len(tests)):
                d = aucs[i] - aucs[j2]
                se = math.sqrt(max(S[i, i] + S[j2, j2] - 2 * S[i, j2], 0))
                zz = d / se if se > 0 else math.nan
                comp.append([f"{tests[i].title} − {tests[j2].title}", num(d, 5), num(se, 5),
                             num(zz, 5), pval(float(2 * stats.norm.sf(abs(zz))))])
        fresh.add_table("Pairwise Comparison of ROC Curves (DeLong)",
                        ["", "AUC Difference", "Standard Error", "Z", "Prob"], comp)
    fresh.add_table("Coordinates of the Curve", ["Test Variable", "Cutoff", "Sensitivity",
                                                 "1 − Specificity"], coords,
                    group="Coordinates")
    fresh.add_graph("ROC Curve", rg.finish(g, "1 − Specificity", "Sensitivity", legend=True),
                    width=420)
    report = st._finish(report, fresh)
    return report, report_text(first.book, "ROC Curve", report, lines)


# ====================================================================== power designs
@dataclass
class OneVariance(pw.Design):
    var: float = 2.0
    var0: float = 1.0
    title: str = "Test for One Variance"

    def power(self, n: int) -> float:
        df = n - 1
        if df < 1:
            return math.nan
        r = self.var0 / self.var
        if self.sided == "two":
            lo = stats.chi2.ppf(self.alpha / 2, df)
            hi = stats.chi2.ppf(1 - self.alpha / 2, df)
            return float(stats.chi2.cdf(lo * r, df) + stats.chi2.sf(hi * r, df))
        if self.var >= self.var0:
            return float(stats.chi2.sf(stats.chi2.ppf(1 - self.alpha, df) * r, df))
        return float(stats.chi2.cdf(stats.chi2.ppf(self.alpha, df) * r, df))

    def assumptions(self):
        return [("Variance", pw._g(self.var)), ("Null Variance", pw._g(self.var0))]


@dataclass
class TwoVariances(pw.Design):
    ratio_var: float = 2.0      # σ1²/σ2² under the alternative (null 1)
    ratio: float = 1.0          # N2 / N1
    title: str = "Test for Two Variances"

    def n2(self, n1: int) -> int:
        return max(2, int(math.ceil(self.ratio * n1 - 1e-9)))

    def sizes(self, n):
        return [n, self.n2(n)]

    def size_labels(self):
        return ["N1", "N2"]

    def power(self, n: int) -> float:
        d1, d2 = n - 1, self.n2(n) - 1
        if d1 < 1 or d2 < 1:
            return math.nan
        rho = self.ratio_var
        if self.sided == "two":
            lo = stats.f.ppf(self.alpha / 2, d1, d2)
            hi = stats.f.ppf(1 - self.alpha / 2, d1, d2)
            return float(stats.f.cdf(lo / rho, d1, d2) + stats.f.sf(hi / rho, d1, d2))
        if rho >= 1:
            return float(stats.f.sf(stats.f.ppf(1 - self.alpha, d1, d2) / rho, d1, d2))
        return float(stats.f.cdf(stats.f.ppf(self.alpha, d1, d2) / rho, d1, d2))

    def assumptions(self):
        return [("Variance Ratio σ1²/σ2²", pw._g(self.ratio_var))]


PROCEDURES = {"t1": "One-Sample t-Test", "t2": "Two-Sample t-Test",
              "tpair": "Paired-Sample t-Test", "p1": "One Proportion",
              "p2": "Two Proportions", "var1": "One Variance", "var2": "Two Variances",
              "anova1": "One-Way ANOVA"}


def numbers(text: str, kind=float) -> list:
    out = []
    for tok in re.split(r"[,;\s]+", str(text or "").strip()):
        if tok:
            try:
                out.append(kind(float(tok)))
            except ValueError:
                raise StatError(f"“{tok}” is not a number.") from None
    return out


def make_design(settings: dict) -> pw.Design:
    proc = settings.get("procedure", "t2")
    alpha = float(settings.get("alpha", 0.05))
    sided = "two" if settings.get("tail", "two") == "two" else "one"
    s = settings
    if proc == "t1":
        return pw.OneSampleT(alpha, sided, mean=float(s.get("mean", 1)),
                             null=float(s.get("null", 0)), sd=float(s.get("sd", 1)),
                             title=PROCEDURES[proc])
    if proc == "tpair":
        return pw.PairedT(alpha, sided, mean=float(s.get("mean", 1)),
                          null=float(s.get("null", 0)), sd=float(s.get("sd", 1)),
                          title=PROCEDURES[proc])
    if proc == "t2":
        return pw.IndependentT(alpha, sided, mean1=float(s.get("mean1", 1)),
                               mean2=float(s.get("mean2", 0)), sd1=float(s.get("sd", 1)),
                               ratio=float(s.get("ratio_n", 1)), title=PROCEDURES[proc])
    if proc == "p1":
        return pw.OneSampleBinomial(alpha, sided, p=float(s.get("p", 0.6)),
                                    p0=float(s.get("p0", 0.5)),
                                    method=s.get("binomial", "normal"),
                                    title=PROCEDURES[proc])
    if proc == "p2":
        return pw.IndependentBinomial(alpha, sided, p1=float(s.get("p1", 0.6)),
                                      p2=float(s.get("p2", 0.4)),
                                      ratio=float(s.get("ratio_n", 1)),
                                      title=PROCEDURES[proc])
    if proc == "var1":
        return OneVariance(alpha, sided, var=float(s.get("var", 2)),
                           var0=float(s.get("var0", 1)), title=PROCEDURES[proc])
    if proc == "var2":
        return TwoVariances(alpha, sided, ratio_var=float(s.get("ratio_var", 2)),
                            ratio=float(s.get("ratio_n", 1)), title=PROCEDURES[proc])
    if proc == "anova1":
        means = numbers(s.get("means", "0, 0.5, 1"))
        if len(means) < 2:
            raise StatError("Type two or more group means.")
        return pw.OneWayAnova(alpha, "two", means=tuple(means), sd=float(s.get("sd", 1)),
                              title=PROCEDURES[proc])
    raise StatError(f"Unknown procedure {proc!r}.")


def power_report(books: dict, settings: dict, report: ReportSheet | None = None,
                 mode: str = "manual"):
    design = make_design(settings)
    proc = settings.get("procedure", "t2")
    estimate = settings.get("estimate", "n")
    if estimate == "n":
        powers = numbers(settings.get("powers", "0.8, 0.9"))
        if not powers or any(not 0 < v < 1 for v in powers):
            raise StatError("Powers must be between 0 and 1.")
        res = pw.run_design(design, "n", powers=powers)
    else:
        ns = numbers(settings.get("sizes", "10, 20, 50"), int)
        if not ns or any(v < 2 for v in ns):
            raise StatError("Sample sizes must be at least 2.")
        res = pw.run_design(design, "power", ns=ns)
    anchor = target_book(books, settings, report)
    report, fresh = new_report(anchor, f"Power and Sample Size: {PROCEDURES[proc]}",
                               "PowerSS", report, "power", settings, mode)
    tail = "Two-sided" if design.sided == "two" else "One-sided"
    st._notes(fresh, f"Power and Sample Size: {PROCEDURES[proc]}", [],
              [["Significance Level", num(design.alpha)], ["Alternative", tail]] +
              [[k, v] for k, v in design.assumptions()] +
              [["Standardized Effect Size", num(design.effect(), 5)]])
    labels = design.size_labels()
    rows = []
    for target, nunit, power in res.rows:
        sizes = design.sizes(nunit) if nunit else [None] * len(labels)
        rows.append(([num(target)] if estimate == "n" else []) +
                    [str(v) if v is not None else "--" for v in sizes] + [num(power, 5)])
    head = (["Target Power"] if estimate == "n" else []) + labels + ["Actual Power"
                                                                     if estimate == "n"
                                                                     else "Power"]
    notes = list(design.footnotes())
    if proc == "anova1":
        notes.append("N is the total sample size (equal group sizes).")
    fresh.add_table("Sample Size" if estimate == "n" else "Power", head, rows, notes)
    ns_found = [r[1] for r in res.rows if r[1]]
    hi = max(ns_found + [20]) * 2 if ns_found else 100
    lo = max(design.min_n, 2)
    grid = np.unique(np.linspace(lo, max(hi, lo + 5), 60).astype(int))
    g = rg.new_graph("Power Curve", 11.0, 8.0)
    xs = np.array([design.sizes(int(v))[0] if proc != "anova1" else design.sizes(int(v))[0]
                   for v in grid], float)
    rg.add_series(g, xs, [design.power(int(v)) for v in grid], "Power", "line", "#1f5fbf",
                  width=1.6, legend=False)
    fresh.add_graph("Power Curve", rg.finish(g, labels[0] if proc != "anova1" else
                                             "Total Sample Size", "Power"), width=400)
    report = st._finish(report, fresh)
    book = anchor.book or _report_book(books, report)
    return report, report_text(book, f"Power and Sample Size ({PROCEDURES[proc]})", report,
                               ["  " + ", ".join(f"{h} {v}" for h, v in zip(head, r))
                                for r in rows])


# ====================================================================== specs
def _when(*procs):
    return {"when": ("procedure", tuple(procs))}


SPECS = {
    "roc": lambda sel: ("ROC Curve", [
        ("Input", [("ranges", "ranges", "Test Variables", sel[:-1] if len(sel) > 1 else sel),
                   ("range", "state", "State Variable", sel[-1] if len(sel) > 1 else ""),
                   ("text", "positive", "Positive State Value", "1")]),
        ("Settings", [("choice", "direction", "Test Direction", "larger",
                       {"larger": "Larger result indicates positive",
                        "smaller": "Smaller result indicates positive"}),
                      ("float", "level", "Confidence Level", 0.95, (0.5, 0.9999, 4))])],
        "Receiver operating characteristic curves with AUC and the optimal cutoff.",
        "report sheet"),
    "power": lambda sel: ("Power and Sample Size", [
        ("Procedure", [("choice", "procedure", "Test", "t2", PROCEDURES),
                       ("choice", "estimate", "Calculate", "n",
                        {"n": "Sample Size", "power": "Power"}),
                       ("text", "powers", "Power(s)", "0.8, 0.9", {"when": ("estimate", "n")}),
                       ("text", "sizes", "Sample Size(s)", "10, 20, 50",
                        {"when": ("estimate", "power")})]),
        ("Parameters", [
            ("float", "mean", "Mean (alternative)", 1.0, (-1e9, 1e9, 6), _when("t1", "tpair")),
            ("float", "null", "Null Mean", 0.0, (-1e9, 1e9, 6), _when("t1", "tpair")),
            ("float", "mean1", "Mean 1", 1.0, (-1e9, 1e9, 6), _when("t2")),
            ("float", "mean2", "Mean 2", 0.0, (-1e9, 1e9, 6), _when("t2")),
            ("float", "sd", "Standard Deviation", 1.0, (1e-9, 1e9, 6),
             _when("t1", "tpair", "t2", "anova1")),
            ("text", "means", "Group Means", "0, 0.5, 1", _when("anova1")),
            ("float", "p", "Proportion (alternative)", 0.6, (0.0001, 0.9999, 4), _when("p1")),
            ("float", "p0", "Null Proportion", 0.5, (0.0001, 0.9999, 4), _when("p1")),
            ("choice", "binomial", "Power Computation", "normal",
             {"normal": "Normal approximation", "exact": "Exact binomial"}, _when("p1")),
            ("float", "p1", "Proportion 1", 0.6, (0.0001, 0.9999, 4), _when("p2")),
            ("float", "p2", "Proportion 2", 0.4, (0.0001, 0.9999, 4), _when("p2")),
            ("float", "var", "Variance (alternative)", 2.0, (1e-9, 1e9, 6), _when("var1")),
            ("float", "var0", "Null Variance", 1.0, (1e-9, 1e9, 6), _when("var1")),
            ("float", "ratio_var", "Variance Ratio σ1²/σ2² (alternative)", 2.0,
             (1e-9, 1e9, 6), _when("var2")),
            ("float", "ratio_n", "Sample Size Ratio N2/N1", 1.0, (0.01, 100, 4),
             _when("t2", "p2", "var2"))]),
        ("Options", [("float", "alpha", "Significance Level", 0.05, (0.0001, 0.5, 4)),
                     ("choice", "tail", "Alternative", "two",
                      {"two": "Two-sided", "one": "One-sided"})])],
        "Power for given sample sizes, or the sample size for a target power (report in "
        "the active workbook).", "report sheet"),
}
FUNCS = {"roc": roc_report, "power": power_report}
for _k, _f in FUNCS.items():
    st.register(_k, _f, SPECS[_k])

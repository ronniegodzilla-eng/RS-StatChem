"""Agreement statistics (SPSS Reliability ICC, Fleiss' kappa, Weighted Kappa, Bland-Altman).

- :func:`icc` — intraclass correlation coefficients exactly as SPSS RELIABILITY /ICC reports
  them (McGraw & Wong 1996): one-way random, two-way random and two-way mixed models,
  consistency or absolute agreement, single and average measures, confidence intervals and the
  F test against a test value.
- :func:`fleiss_kappa` — Fleiss' multiple-rater kappa with the Fleiss, Nee & Landis (1979)
  standard errors (SPSS 28 Reliability "Inter-Rater Agreement").
- :func:`weighted_kappa` and :func:`weighted_kappa_analysis` — Cohen's (weighted) kappa for
  pairs of raters (SPSS 27 Analyze > Scale > Weighted Kappa) with linear or quadratic weights.
- :func:`bland_altman` — Bland-Altman agreement analysis (SPSS 30): bias, limits of agreement
  with exact confidence intervals (Bland & Altman 1999), proportional-bias regression and plot.
"""
from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, fmt_p, listwise, value_text

ICC_MODELS = {"oneway": "One-Way Random", "twoway_random": "Two-Way Random",
              "twoway_mixed": "Two-Way Mixed"}


# ============================================================================ ICC
def anova_two_way(x: np.ndarray) -> dict:
    """Mean squares of a subjects x raters table without replication."""
    x = np.asarray(x, float)
    n, k = x.shape
    grand = x.mean()
    ss_rows = k * ((x.mean(1) - grand) ** 2).sum()
    ss_cols = n * ((x.mean(0) - grand) ** 2).sum()
    ss_total = ((x - grand) ** 2).sum()
    ss_err = ss_total - ss_rows - ss_cols
    ss_within = ss_total - ss_rows
    return {"n": n, "k": k, "msr": ss_rows / (n - 1), "msc": ss_cols / (k - 1),
            "mse": ss_err / ((n - 1) * (k - 1)), "msw": ss_within / (n * (k - 1)),
            "ss_rows": ss_rows, "ss_cols": ss_cols, "ss_err": ss_err, "ss_total": ss_total,
            "ss_within": ss_within, "grand": grand}


def icc(x: np.ndarray, model: str = "twoway_mixed", kind: str = "consistency",
        level: float = 0.95, test_value: float = 0.0) -> dict:
    """ICC as SPSS: returns {"single": {...}, "average": {...}} each with value, lo, hi, F,
    df1, df2, p. ``model`` in ICC_MODELS, ``kind`` "consistency" | "absolute" (one-way is
    always absolute agreement)."""
    a = anova_two_way(x)
    n, k = a["n"], a["k"]
    msr, msc, mse, msw = a["msr"], a["msc"], a["mse"], a["msw"]
    alpha = 1 - level
    r0 = test_value
    out = {}
    if model == "oneway":
        f_obs = msr / msw
        df1, df2 = n - 1, n * (k - 1)
        fl = f_obs / stats.f.ppf(1 - alpha / 2, df1, df2)
        fu = f_obs * stats.f.ppf(1 - alpha / 2, df2, df1)
        out["single"] = dict(value=(msr - msw) / (msr + (k - 1) * msw),
                             lo=(fl - 1) / (fl + k - 1), hi=(fu - 1) / (fu + k - 1),
                             F=f_obs * (1 - r0) / (1 + (k - 1) * r0), df1=df1, df2=df2)
        out["average"] = dict(value=(msr - msw) / msr, lo=1 - 1 / fl, hi=1 - 1 / fu,
                              F=f_obs * (1 - r0), df1=df1, df2=df2)
    elif kind == "consistency":
        f_obs = msr / mse
        df1, df2 = n - 1, (n - 1) * (k - 1)
        fl = f_obs / stats.f.ppf(1 - alpha / 2, df1, df2)
        fu = f_obs * stats.f.ppf(1 - alpha / 2, df2, df1)
        out["single"] = dict(value=(msr - mse) / (msr + (k - 1) * mse),
                             lo=(fl - 1) / (fl + k - 1), hi=(fu - 1) / (fu + k - 1),
                             F=f_obs * (1 - r0) / (1 + (k - 1) * r0), df1=df1, df2=df2)
        out["average"] = dict(value=(msr - mse) / msr, lo=1 - 1 / fl, hi=1 - 1 / fu,
                              F=f_obs * (1 - r0), df1=df1, df2=df2)
    else:
        rho = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
        aa = k * rho / (n * (1 - rho))
        bb = 1 + k * rho * (n - 1) / (n * (1 - rho))
        v = (aa * msc + bb * mse) ** 2 / ((aa * msc) ** 2 / (k - 1)
                                         + (bb * mse) ** 2 / ((n - 1) * (k - 1)))
        fs = stats.f.ppf(1 - alpha / 2, n - 1, v)
        fsu = stats.f.ppf(1 - alpha / 2, v, n - 1)
        lo = n * (msr - fs * mse) / (fs * (k * msc + (k * n - k - n) * mse) + n * msr)
        hi = n * (fsu * msr - mse) / (k * msc + (k * n - k - n) * mse + n * fsu * msr)

        def f_test(c_avg: bool):
            if r0 == 0:
                return msr / mse, n - 1, (n - 1) * (k - 1)
            if c_avg:
                a0, b0 = r0 / (n * (1 - r0)), 1 + r0 * (n - 1) / (n * (1 - r0))
            else:
                a0, b0 = k * r0 / (n * (1 - r0)), 1 + k * r0 * (n - 1) / (n * (1 - r0))
            v0 = (a0 * msc + b0 * mse) ** 2 / ((a0 * msc) ** 2 / (k - 1)
                                               + (b0 * mse) ** 2 / ((n - 1) * (k - 1)))
            return msr / (a0 * msc + b0 * mse), n - 1, v0

        f1, d11, d12 = f_test(False)
        f2, d21, d22 = f_test(True)
        out["single"] = dict(value=rho, lo=lo, hi=hi, F=f1, df1=d11, df2=d12)
        out["average"] = dict(value=(msr - mse) / (msr + (msc - mse) / n),
                              lo=k * lo / (1 + (k - 1) * lo), hi=k * hi / (1 + (k - 1) * hi),
                              F=f2, df1=d21, df2=d22)
    for res in out.values():
        res["p"] = float(stats.f.sf(res["F"], res["df1"], res["df2"]))
    out["anova"] = a
    return out


def icc_table(x: np.ndarray, model: str, kind: str, level: float, test_value: float) -> Table:
    res = icc(x, model, kind, level, test_value)
    if model == "oneway":
        marks = {"single": "", "average": ""}
        notes = ["One-way random effects model where people effects are random."]
        head_mark = ""
    else:
        marks = {"single": "ᵃ", "average": "ᶜ" if model == "twoway_mixed" else "ᵃ"}
        head_mark = "ᵇ"
        notes = ["Two-way mixed effects model where people effects are random and measures "
                 "effects are fixed." if model == "twoway_mixed" else
                 "Two-way random effects model where both people effects and measures effects "
                 "are random.",
                 "a. The estimator is the same, whether the interaction effect is present or "
                 "not.",
                 "b. Type C intraclass correlation coefficients using a consistency definition. "
                 "The between-measure variance is excluded from the denominator variance."
                 if kind == "consistency" else
                 "b. Type A intraclass correlation coefficients using an absolute agreement "
                 "definition."]
        if model == "twoway_mixed":
            notes.append("c. This estimate is computed assuming the interaction effect is "
                         "absent, because it is not estimable otherwise.")
    rows = []
    for key, label in (("single", "Single Measures"), ("average", "Average Measures")):
        r = res[key]
        rows.append([label, fmt(r["value"]) + marks[key], fmt(r["lo"]), fmt(r["hi"]),
                     fmt(r["F"]), _df(r["df1"]), _df(r["df2"]), fmt_p(r["p"])])
    lvl = f"{level * 100:g}%"
    head = f"F Test with True Value {test_value:g}"
    return Table("Intraclass Correlation Coefficient", [
        "", f"Intraclass Correlation{head_mark}", (f"{lvl} Confidence Interval", "Lower Bound"),
        (f"{lvl} Confidence Interval", "Upper Bound"), (head, "Value"), (head, "df1"),
        (head, "df2"), (head, "Sig")], rows, footnotes=notes)


def _df(v: float) -> str:
    return fmt_int(v) if float(v).is_integer() else fmt(v, 3)


# ============================================================================ Fleiss
def fleiss_counts(ratings: np.ndarray, categories: list | None = None) -> tuple[np.ndarray, list]:
    """subjects x raters ratings -> subjects x categories counts."""
    ratings = np.asarray(ratings, dtype=object)
    cats = categories or sorted({v for v in ratings.ravel() if v == v},
                                key=lambda v: (isinstance(v, str), v))
    counts = np.array([[int(np.sum(row == c)) for c in cats] for row in ratings], float)
    return counts, cats


def fleiss_kappa(counts: np.ndarray, level: float = 0.95) -> dict:
    """Fleiss (1971) kappa with Fleiss, Nee & Landis (1979) null standard errors.

    Returns overall kappa, SE, z, p, CI and per-category kappas (same fields)."""
    x = np.asarray(counts, float)
    n = x.shape[0]
    m = x.sum(1)
    if not np.allclose(m, m[0]):
        raise ValueError("Every subject must be rated by the same number of raters.")
    m = m[0]
    p = x.sum(0) / (n * m)
    q = 1 - p
    pi = ((x * (x - 1)).sum(1)) / (m * (m - 1))
    pbar, pe = pi.mean(), (p ** 2).sum()
    kappa = (pbar - pe) / (1 - pe)
    pq = (p * q).sum()
    var0 = 2 / (n * m * (m - 1)) * (pq ** 2 - (p * q * (q - p)).sum()) / pq ** 2
    se = math.sqrt(max(var0, 0.0))
    z = stats.norm.ppf(0.5 + level / 2)
    out = {"kappa": kappa, "se": se, "z": kappa / se if se else float("nan")}
    out["p"] = 2 * stats.norm.sf(abs(out["z"])) if se else float("nan")
    out["lo"], out["hi"] = kappa - z * se, kappa + z * se
    cats = []
    for j in range(x.shape[1]):
        if p[j] in (0, 1):
            cats.append(None)
            continue
        kj = 1 - (x[:, j] * (m - x[:, j])).sum() / (n * m * (m - 1) * p[j] * q[j])
        sej = math.sqrt(2 / (n * m * (m - 1)))
        zj = kj / sej
        cats.append({"kappa": kj, "se": sej, "z": zj, "p": 2 * stats.norm.sf(abs(zj)),
                     "lo": kj - z * sej, "hi": kj + z * sej})
    out["categories"] = cats
    out["n"], out["m"] = n, int(m)
    return out


def fleiss_tables(ratings: np.ndarray, titles_cats: list[str], level: float = 0.95,
                  individual: bool = True) -> list[Table]:
    counts, cats = fleiss_counts(ratings)
    res = fleiss_kappa(counts, level)
    lvl = f"{level * 100:g}%"
    cols = ["", "Kappa", "Asymptotic Standard Error", "Z", "P Value",
            f"Lower {lvl} Asymptotic CI Bound", f"Upper {lvl} Asymptotic CI Bound"]
    tables = [Table("Overall Kappa", cols, [[
        "Overall", fmt(res["kappa"]), fmt(res["se"]), fmt(res["z"], 3), fmt_p(res["p"]),
        fmt(res["lo"]), fmt(res["hi"])]], footnotes=[
        f"Sample data contains {res['n']} effective subjects and {res['m']} raters."])]
    if individual:
        rows = []
        for label, r in zip(titles_cats if titles_cats else [str(c) for c in cats],
                            res["categories"]):
            if r is None:
                rows.append([label, "", "", "", "", "", ""])
            else:
                rows.append([label, fmt(r["kappa"]), fmt(r["se"]), fmt(r["z"], 3),
                             fmt_p(r["p"]), fmt(r["lo"]), fmt(r["hi"])])
        tables.append(Table("Kappas for Individual Categories", ["Rating Category"] + cols[1:],
                            rows))
    return tables


# ============================================================================ weighted kappa
def weighted_kappa(table: np.ndarray, weighting: str = "linear",
                   level: float = 0.95) -> dict:
    """Cohen's kappa (``weighting="none"``) or weighted kappa with linear/quadratic agreement
    weights; ASE (Fleiss, Cohen & Everitt 1969) for the CI and the null ASE for Z."""
    t = np.asarray(table, float)
    k = t.shape[0]
    total = t.sum()
    p = t / total
    r, c = p.sum(1), p.sum(0)
    i, j = np.indices((k, k))
    if weighting == "none" or k < 2:
        w = (i == j).astype(float)
    elif weighting == "quadratic":
        w = 1 - (i - j) ** 2 / (k - 1) ** 2
    else:
        w = 1 - np.abs(i - j) / (k - 1)
    po = (w * p).sum()
    pe = (w * np.outer(r, c)).sum()
    kw = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    wr = w @ c        # w_i.  (weighted average over columns)
    wc = w.T @ r      # w_.j
    term = (p * (w - (wr[:, None] + wc[None, :]) * (1 - kw)) ** 2).sum()
    var1 = (term - (kw - pe * (1 - kw)) ** 2) / (total * (1 - pe) ** 2)
    var0 = ((np.outer(r, c) * (w - (wr[:, None] + wc[None, :])) ** 2).sum() - pe ** 2) \
        / (total * (1 - pe) ** 2)
    ase, ase0 = math.sqrt(max(var1, 0.0)), math.sqrt(max(var0, 0.0))
    zq = stats.norm.ppf(0.5 + level / 2)
    z = kw / ase0 if ase0 > 0 else float("nan")
    return {"kappa": kw, "ase": ase, "ase0": ase0, "z": z,
            "p": 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else float("nan"),
            "lo": kw - zq * ase, "hi": kw + zq * ase, "n": int(total), "po": po, "pe": pe}


def weighted_kappa_analysis(ds: Dataset, pairs: list[tuple[str, str]],
                            weighting: str = "linear", level: float = 0.95,
                            crosstab: bool = False) -> Section:
    section = Section("Weighted Kappa")
    section.log("WEIGHTED KAPPA\n  /PAIRS=" + " ".join(f"{a} WITH {b}" for a, b in pairs)
                + f"\n  /CRITERIA WEIGHTING={weighting.upper()} CILEVEL={level * 100:g}"
                + ("\n  /PRINT=TABLE" if crosstab else "") + ".")
    rows = []
    lvl = f"{level * 100:g}%"
    for a, b in pairs:
        va, vb = ds.var(a), ds.var(b)
        frame = pd.DataFrame({"a": ds.analysis_series(a), "b": ds.analysis_series(b)}).dropna()
        cats = sorted(set(frame["a"]) | set(frame["b"]), key=lambda v: (isinstance(v, str), v))
        t = np.array([[int(((frame["a"] == x) & (frame["b"] == y)).sum()) for y in cats]
                      for x in cats], float)
        if len(frame) == 0 or len(cats) < 2:
            rows.append([f"{va.title} - {vb.title}", "", "", "", "", "", "", fmt_int(len(frame))])
            continue
        res = weighted_kappa(t, weighting, level)
        rows.append([f"{va.title} - {vb.title}", fmt(res["kappa"]), fmt(res["ase"]),
                     fmt(res["z"]), fmt_p(res["p"]), fmt(res["lo"]), fmt(res["hi"]),
                     fmt_int(res["n"])])
        if crosstab:
            trow = [[value_text(va, x)] + [fmt_int(v) for v in t[i]] + [fmt_int(t[i].sum())]
                    for i, x in enumerate(cats)]
            trow.append(["Total"] + [fmt_int(v) for v in t.sum(0)] + [fmt_int(t.sum())])
            section.table(Table(f"{va.title} * {vb.title} Crosstabulation",
                                [""] + [(vb.title, value_text(vb, y)) for y in cats]
                                + ["Total"], trow))
    w_text = {"linear": "Linear", "quadratic": "Quadratic", "none": "Unweighted"}[weighting]
    section.table(Table("Weighted Kappaᵃ", [
        "", "Weighted Kappa", "Asymptotic Std. Error", "Z", "Asymptotic Sig.",
        (f"Asymptotic {lvl} Confidence Interval", "Lower Bound"),
        (f"Asymptotic {lvl} Confidence Interval", "Upper Bound"), "N"], rows,
        footnotes=[f"a. {w_text} weighting is used. The categories are treated as equally "
                   "spaced ordinal positions.",
                   "Z uses the asymptotic standard error under the null hypothesis; the "
                   "confidence interval uses the standard error not assuming it."]))
    return section


# ============================================================================ Bland-Altman
def bland_altman_stats(a: np.ndarray, b: np.ndarray, level: float = 0.95) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    m = (a + b) / 2
    n = len(d)
    mean, sd = d.mean(), d.std(ddof=1)
    z = stats.norm.ppf(0.5 + level / 2)
    t = stats.t.ppf(0.5 + level / 2, n - 1)
    se_bias = sd / math.sqrt(n)
    se_loa = sd * math.sqrt(1 / n + z ** 2 / (2 * (n - 1)))
    lower, upper = mean - z * sd, mean + z * sd
    tt = mean / se_bias if se_bias > 0 else float("nan")
    slope, intercept, r, p_slope, se_slope = stats.linregress(m, d)
    return {"n": n, "bias": mean, "sd": sd, "se_bias": se_bias,
            "bias_lo": mean - t * se_bias, "bias_hi": mean + t * se_bias,
            "t": tt, "df": n - 1, "p": 2 * stats.t.sf(abs(tt), n - 1),
            "lower": lower, "upper": upper, "se_loa": se_loa,
            "lower_lo": lower - t * se_loa, "lower_hi": lower + t * se_loa,
            "upper_lo": upper - t * se_loa, "upper_hi": upper + t * se_loa,
            "slope": slope, "intercept": intercept, "se_slope": se_slope,
            "t_slope": slope / se_slope if se_slope > 0 else float("nan"),
            "p_slope": p_slope, "r": r, "z": z, "diff": d, "mean_ab": m,
            "outside": int(((d < lower) | (d > upper)).sum())}


def bland_altman(ds: Dataset, pairs: list[tuple[str, str]], level: float = 0.95,
                 proportional: bool = True, plot: bool = True, percent: bool = False) -> Section:
    """``percent`` analyses the difference as a percentage of the mean (100*(A-B)/mean)."""
    section = Section("Bland-Altman")
    section.log("BLANDALTMAN\n  /PAIRS=" + " ".join(f"{a} WITH {b}" for a, b in pairs)
                + f"\n  /CRITERIA CILEVEL={level * 100:g}"
                + (" DIFFERENCE=PERCENT" if percent else "") + ".")
    lvl = f"{level * 100:g}%"
    srows, lrows, prows = [], [], []
    for a, b in pairs:
        va, vb = ds.var(a), ds.var(b)
        frame = listwise(ds, [a, b])
        x, y = frame[a].to_numpy(float), frame[b].to_numpy(float)
        if len(x) < 3:
            continue
        if percent:
            mean_ab = (x + y) / 2
            ok = mean_ab != 0
            x, y = x[ok], y[ok]
            res = bland_altman_stats(100 * (x - y) / ((x + y) / 2), np.zeros(len(x)), level)
            res["mean_ab"] = (x + y) / 2
        else:
            res = bland_altman_stats(x, y, level)
        label = f"{va.title} - {vb.title}"
        srows.append([label, fmt_int(res["n"]), fmt(res["bias"], 4), fmt(res["sd"], 4),
                      fmt(res["se_bias"], 4), fmt(res["bias_lo"], 4), fmt(res["bias_hi"], 4),
                      fmt(res["t"]), fmt_int(res["df"]), fmt_p(res["p"])])
        lrows.append([label, "Lower", fmt(res["lower"], 4), fmt(res["se_loa"], 4),
                      fmt(res["lower_lo"], 4), fmt(res["lower_hi"], 4)])
        lrows.append(["", "Upper", fmt(res["upper"], 4), fmt(res["se_loa"], 4),
                      fmt(res["upper_lo"], 4), fmt(res["upper_hi"], 4)])
        if proportional:
            prows.append([label, "(Constant)", fmt(res["intercept"], 4), "", "", ""])
            prows.append(["", "Mean of measurements", fmt(res["slope"], 4),
                          fmt(res["se_slope"], 4), fmt(res["t_slope"]), fmt_p(res["p_slope"])])
        if plot:
            section.image(f"Bland-Altman Plot: {label}", bland_altman_plot(
                res, f"Bland-Altman Plot: {label}",
                "Mean of " + va.title + " and " + vb.title,
                ("Difference (%)" if percent else f"Difference ({va.title} - {vb.title})"),
                level))
    unit = " (%)" if percent else ""
    section.table(Table("Bias (Mean Difference)", [
        "", "N", f"Mean Difference{unit}", "Std. Deviation", "Std. Error",
        (f"{lvl} Confidence Interval", "Lower"), (f"{lvl} Confidence Interval", "Upper"),
        ("One-Sample Test (Bias = 0)", "t"), ("One-Sample Test (Bias = 0)", "df"),
        ("One-Sample Test (Bias = 0)", "Sig. (2-tailed)")], srows))
    section.table(Table("Limits of Agreement", [
        "", "", "Limit", "Std. Error", (f"{lvl} Confidence Interval", "Lower"),
        (f"{lvl} Confidence Interval", "Upper")], lrows, row_header_cols=2,
        footnotes=[f"Limits of agreement = mean difference ± {stats.norm.ppf(0.5 + level / 2):.2f}"
                   " × SD of the differences. Their confidence intervals use the exact standard "
                   "error of Bland & Altman (1999)."]))
    if proportional:
        section.table(Table("Proportional Bias", [
            "", "", "B", "Std. Error", "t", "Sig."], prows, row_header_cols=2,
            footnotes=["Dependent variable: difference between the measurements. A significant "
                       "slope indicates proportional bias."]))
    return section


def bland_altman_plot(res: dict, title: str, xlabel: str, ylabel: str, level: float) -> bytes:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from rsstatchem.core.stats.charts import BAR, EDGE, _style

    fig = Figure(figsize=(6.0, 4.3), dpi=120)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    m, d = res["mean_ab"], res["diff"]
    ax.scatter(m, d, s=22, color=BAR, edgecolors=EDGE, linewidths=0.6, alpha=0.9)
    lo_x, hi_x = float(np.min(m)), float(np.max(m))
    span = hi_x - lo_x or 1.0
    for y, text, style in ((res["bias"], "Mean", "-"), (res["upper"], f"+{res['z']:.2f} SD",
                                                         "--"),
                           (res["lower"], f"-{res['z']:.2f} SD", "--")):
        ax.axhline(y, color="#1f1f1f", lw=1.2, ls=style)
        ax.text(hi_x + 0.02 * span, y, f"{text}\n{y:.3f}", va="center", fontsize=8)
    ax.axhline(0, color="#999999", lw=0.8, ls=":")
    ax.set_xlim(lo_x - 0.05 * span, hi_x + 0.22 * span)
    _style(ax, title, xlabel, ylabel)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    return buf.getvalue()

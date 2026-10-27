"""Analyze > Descriptive Statistics: stand-alone P-P Plots / Q-Q Plots (SPSS PPLOT),
Percentiles (SPSS 29.0.1) and Normality Analysis (SPSS 30).

PPLOT: 13 test distributions with parameters estimated from the data (SPSS PPLOT estimators:
moments for beta/gamma/Laplace/logistic, logs for the lognormal, min/max for the uniform, MLE
shape for the Pareto) or specified; transforms (natural log, differencing, seasonal
differencing, standardize, applied in that order); Blom / Rankit / Tukey / Van der Waerden
proportion estimates; mean / high / low / arbitrary ranks for ties; probability plot plus its
detrended version.

Percentiles: the five SPSS definitions (HAVERAGE, WAVERAGE, ROUND, EMPIRICAL, AEMPIRICAL),
Tukey's hinges, and distribution-free confidence intervals from binomial order statistics.

Normality Analysis: Shapiro-Wilk, K-S (Lilliefors), Anderson-Darling and D'Agostino tests per
variable; Mardia's multivariate skewness and kurtosis, Henze-Zirkler and Royston tests as in
the R package MVN (Korkmaz, Goksuluk & Zararsiz 2014); normal Q-Q plots and the chi-square
Q-Q plot of squared Mahalanobis distances.
"""
from __future__ import annotations

import io
import math

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import (fmt, fmt_int, fmt_p, lilliefors, numeric, shapiro,
                                          value_text)

BAR = "#C62828"
EDGE = "#7f1010"
LINE = "#1f1f1f"

# key: (SPSS name, parameter names, PPLOT keyword)
DISTRIBUTIONS: dict[str, tuple[str, tuple[str, ...], str]] = {
    "beta": ("Beta", ("Shape", "Shape"), "BETA"),
    "chisq": ("Chi-square", ("df",), "CHI"),
    "exponential": ("Exponential", ("Scale",), "EXPONENTIAL"),
    "gamma": ("Gamma", ("Shape", "Scale"), "GAMMA"),
    "halfnormal": ("Half-normal", ("Location", "Scale"), "HNORMAL"),
    "laplace": ("Laplace", ("Location", "Scale"), "LAPLACE"),
    "logistic": ("Logistic", ("Location", "Scale"), "LOGISTIC"),
    "lognormal": ("Lognormal", ("Scale", "Shape"), "LNORMAL"),
    "normal": ("Normal", ("Location", "Scale"), "NORMAL"),
    "pareto": ("Pareto", ("Threshold", "Shape"), "PARETO"),
    "t": ("Student t", ("df",), "T"),
    "uniform": ("Uniform", ("Minimum", "Maximum"), "UNIFORM"),
    "weibull": ("Weibull", ("Scale", "Shape"), "WEIBULL"),
}

FRACTIONS = {"blom": ("Blom's", 3 / 8, 1 / 4), "rankit": ("Rankit", 1 / 2, 0.0),
             "tukey": ("Tukey's", 1 / 3, 1 / 3), "vw": ("Van der Waerden's", 0.0, 1.0)}
TIES = {"mean": "Mean rank of tied values", "high": "High rank of tied values",
        "low": "Low rank of tied values", "break": "Break ties arbitrarily"}


# ============================================================================ plotting helpers
def _figure(width: float = 5.4, height: float = 4.3) -> Figure:
    fig = Figure(figsize=(width, height), dpi=120)
    FigureCanvasAgg(fig)
    return fig


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=9)


def _png(fig: Figure) -> bytes:
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    return buf.getvalue()


def _points_plot(x, y, title: str, xlabel: str, ylabel: str, line=None,
                 hline: float | None = None) -> bytes:
    fig = _figure()
    ax = fig.add_subplot()
    ax.plot(x, y, "o", color=BAR, markeredgecolor=EDGE, markersize=4.5)
    if line is not None:
        ax.plot(line[0], line[1], color=LINE, lw=1.3)
    if hline is not None:
        ax.axhline(hline, color=LINE, lw=1.1)
    _style(ax, title, xlabel, ylabel)
    return _png(fig)


# ============================================================================ distributions
def frozen(dist: str, params: tuple[float, ...]):
    """scipy frozen distribution with SPSS parameterisation (CDF.<DIST> argument order)."""
    p = params
    if dist == "beta":
        return stats.beta(p[0], p[1])
    if dist == "chisq":
        return stats.chi2(p[0])
    if dist == "exponential":  # CDF.EXP(q, scale) = 1 - exp(-scale * q): scale is a rate
        return stats.expon(scale=1.0 / p[0])
    if dist == "gamma":  # CDF.GAMMA(q, shape, scale) with scale a rate
        return stats.gamma(p[0], scale=1.0 / p[1])
    if dist == "halfnormal":
        return stats.halfnorm(p[0], p[1])
    if dist == "laplace":
        return stats.laplace(p[0], p[1])
    if dist == "logistic":
        return stats.logistic(p[0], p[1])
    if dist == "lognormal":  # CDF.LNORMAL(q, a, b): a = exp(mu), b = sigma
        return stats.lognorm(s=p[1], scale=p[0])
    if dist == "normal":
        return stats.norm(p[0], p[1])
    if dist == "pareto":
        return stats.pareto(b=p[1], scale=p[0])
    if dist == "t":
        return stats.t(p[0])
    if dist == "uniform":
        return stats.uniform(p[0], p[1] - p[0])
    if dist == "weibull":  # CDF.WEIBULL(q, a, b) = 1 - exp(-(q/a)^b)
        return stats.weibull_min(c=p[1], scale=p[0])
    raise ValueError(f"unknown distribution {dist!r}")


def estimate(dist: str, x: np.ndarray, df: float | None = None) -> tuple[float, ...]:
    """Parameters estimated from the data as SPSS PPLOT does (moments unless noted)."""
    x = np.asarray(x, float)
    n = len(x)
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else float("nan")
    var = sd * sd
    if dist == "beta":
        k = mean * (1 - mean) / var - 1
        return mean * k, (1 - mean) * k
    if dist == "chisq":
        return (float(df) if df else mean,)
    if dist == "exponential":
        return (1.0 / mean,)
    if dist == "gamma":
        return mean * mean / var, mean / var
    if dist == "halfnormal":  # maximum likelihood: threshold at the minimum
        loc = float(x.min())
        return loc, float(math.sqrt(np.mean((x - loc) ** 2)))
    if dist == "laplace":
        return mean, sd / math.sqrt(2.0)
    if dist == "logistic":
        return mean, sd * math.sqrt(3.0) / math.pi
    if dist == "lognormal":
        lx = np.log(x)
        return float(math.exp(lx.mean())), float(lx.std(ddof=1))
    if dist == "normal":
        return mean, sd
    if dist == "pareto":
        lo = float(x.min())
        return lo, float(n / np.sum(np.log(x / lo)))
    if dist == "t":
        if df:
            return (float(df),)
        return (2 * var / (var - 1) if var > 1 else 30.0,)
    if dist == "uniform":
        return float(x.min()), float(x.max())
    if dist == "weibull":  # maximum likelihood with the threshold fixed at 0
        c, _, scale = stats.weibull_min.fit(x, floc=0)
        return float(scale), float(c)
    raise ValueError(f"unknown distribution {dist!r}")


def _support_ok(dist: str, x: np.ndarray) -> str | None:
    if dist in ("beta",) and (x.min() <= 0 or x.max() >= 1):
        return "Beta requires all values strictly between 0 and 1."
    if dist in ("chisq", "exponential", "gamma", "lognormal", "pareto", "weibull") and \
            x.min() <= 0:
        return f"{DISTRIBUTIONS[dist][0]} requires positive values."
    return None


# ============================================================================ ranks / fractions
def ranks(x: np.ndarray, ties: str = "mean") -> np.ndarray:
    method = {"mean": "average", "high": "max", "low": "min", "break": "ordinal"}[ties]
    return stats.rankdata(x, method=method)


def fractions(r: np.ndarray, n: int, formula: str = "blom") -> np.ndarray:
    """Proportion estimate (r - a) / (n + b) for Blom, Rankit, Tukey, Van der Waerden."""
    _, a, b = FRACTIONS[formula]
    return (np.asarray(r, float) - a) / (n + b)


def transform(x: np.ndarray, log: bool = False, diff: int = 0, sdiff: int = 0,
              period: int = 1, standardize: bool = False) -> np.ndarray:
    """SPSS PPLOT transforms; non-positive values become missing under the natural log."""
    y = np.asarray(x, float).copy()
    if log:
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.where(y > 0, np.log(np.where(y > 0, y, 1.0)), np.nan)
    for _ in range(int(diff)):
        y = y[1:] - y[:-1]
    for _ in range(int(sdiff)):
        y = y[period:] - y[:-period] if period >= 1 and len(y) > period else y[:0]
    if standardize:
        v = y[~np.isnan(y)]
        if len(v) > 1 and v.std(ddof=1) > 0:
            y = (y - v.mean()) / v.std(ddof=1)
    return y


def probability_points(x: np.ndarray, dist: str, params: tuple, kind: str = "pp",
                       formula: str = "blom", ties: str = "mean") -> dict:
    """Observed/expected coordinates of a P-P or Q-Q plot and its detrended version."""
    x = np.asarray(x, float)
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    frac = fractions(ranks(x, ties)[order], n, formula)
    d = frozen(dist, params)
    if kind == "pp":
        expected = d.cdf(xs)
        return {"x": frac, "y": expected, "observed": xs, "detrended": frac - expected}
    expected = d.ppf(frac)
    return {"x": xs, "y": expected, "observed": xs, "detrended": xs - expected}


# ============================================================================ PPLOT
def pplot(ds: Dataset, names: list[str], kind: str = "pp", dist: str = "normal",
          estimate_params: bool = True, params: tuple | None = None, df: float | None = None,
          log: bool = False, standardize: bool = False, diff: int = 0, sdiff: int = 0,
          period: int = 1, formula: str = "blom", ties: str = "mean") -> Section:
    """P-P Plots… / Q-Q Plots… (SPSS PPLOT)."""
    title, pnames, keyword = DISTRIBUTIONS[dist]
    label = "P-P" if kind == "pp" else "Q-Q"
    section = Section("PPlot")
    dist_arg = keyword if estimate_params or not params else \
        f"{keyword}({','.join(f'{v:g}' for v in params)})"
    if dist in ("chisq", "t") and df and estimate_params:
        dist_arg = f"{keyword}({df:g})"
    section.log("PPLOT\n  /VARIABLES=" + " ".join(names)
                + ("\n  /LN" if log else "\n  /NOLOG")
                + (f"\n  /DIFF={diff}" if diff else "")
                + (f"\n  /SDIFF={sdiff}" if sdiff else "")
                + ("\n  /STANDARDIZE" if standardize else "\n  /NOSTANDARDIZE")
                + f"\n  /TYPE={label}\n  /FRACTION={formula.upper()}\n  /TIES={ties.upper()}"
                + f"\n  /DIST={dist_arg}.")
    if not names:
        return section
    tname = ("Natural logarithm" if log else "None")
    model = [["Model Name", "", "MOD_1"]]
    for i, n in enumerate(names, start=1):
        model.append(["Series or Sequence" if i == 1 else "", str(i), ds.var(n).title])
    model += [["Transformation", "", tname],
              ["Non-Seasonal Differencing", "", str(diff)],
              ["Seasonal Differencing", "", str(sdiff)],
              ["Length of Seasonal Period", "", str(period) if sdiff else "No periodicity"],
              ["Standardization", "", "Applied" if standardize else "Not applied"],
              ["Distribution", "Type", title]]
    for pi, pn in enumerate(pnames):
        if dist in ("chisq", "t") and df:
            shown = f"{df:g}"
        elif not estimate_params and params and pi < len(params):
            shown = f"{params[pi]:g}"
        else:
            shown = "estimated"
        model.append(["", pn, shown])
    model += [["Fractional Rank Estimation Method", "", FRACTIONS[formula][0]],
              ["Rank Assigned to Ties", "", TIES[ties]]]
    section.table(Table("Model Description", ["", "", ""], model, row_header_cols=2,
                        footnotes=["Applying the model specifications from MOD_1"]))

    cps_rows: list[list[str]] = []
    est_cols, est_vals, notes, series = [], [], [], []
    for n in names:
        var = ds.var(n)
        raw = ds.df[n]
        clean = numeric(ds, n)
        user_missing = int((raw.notna() & clean.isna()).sum())
        system_missing = int(raw.isna().sum())
        y = transform(clean.to_numpy(float), log, diff, sdiff, period, standardize)
        lost = int(np.isnan(y).sum() - clean.isna().sum()) if len(y) == len(clean) else 0
        y = y[~np.isnan(y)]
        series.append((var, y))
        cps_rows.append([var.title, fmt_int(len(clean)), fmt_int(user_missing),
                         fmt_int(system_missing + max(lost, 0))])
    section.table(Table("Case Processing Summary", [
        "", "Series or Sequence Length",
        ("Number of Missing Values in the Plot", "User-Missing"),
        ("Number of Missing Values in the Plot", "System-Missing")], cps_rows,
        footnotes=["The cases are unweighted."]))

    plots = []
    for var, y in series:
        if len(y) < 2:
            notes.append(f"{var.title}: not enough valid values.")
            continue
        problem = _support_ok(dist, y)
        if problem:
            notes.append(f"{var.title}: {problem}")
            continue
        if estimate_params or not params:
            p = estimate(dist, y, df)
        else:
            p = tuple(float(v) for v in params)
        est_cols.append(var.title)
        est_vals.append(p)
        plots.append((var, y, p))
    if est_cols:
        rows = [[f"{title} Distribution" if i == 0 else "", pn] +
                [fmt(v[i], 5) for v in est_vals] for i, pn in enumerate(pnames)]
        section.table(Table("Estimated Distribution Parameters", ["", ""] + est_cols, rows,
                            row_header_cols=2, footnotes=["The cases are unweighted."]))
    for msg in notes:
        section.note("Warning", msg, style="warning")
    for var, y, p in plots:
        pts = probability_points(y, dist, p, kind, formula, ties)
        if kind == "pp":
            section.image(f"{title} P-P Plot of {var.title}", _points_plot(
                pts["x"], pts["y"], f"{title} P-P Plot of {var.title}", "Observed Cum Prob",
                "Expected Cum Prob", line=([0, 1], [0, 1])))
            section.image(f"Detrended {title} P-P Plot of {var.title}", _points_plot(
                pts["observed"], pts["detrended"], f"Detrended {title} P-P Plot of "
                f"{var.title}", "Observed Value", f"Deviation from {title}", hline=0.0))
        else:
            finite = np.isfinite(pts["y"])
            lo, hi = float(np.min(pts["x"])), float(np.max(pts["x"]))
            section.image(f"{title} Q-Q Plot of {var.title}", _points_plot(
                pts["x"][finite], pts["y"][finite], f"{title} Q-Q Plot of {var.title}",
                "Observed Value", f"Expected {title} Value", line=([lo, hi], [lo, hi])))
            section.image(f"Detrended {title} Q-Q Plot of {var.title}", _points_plot(
                pts["observed"][finite], pts["detrended"][finite],
                f"Detrended {title} Q-Q Plot of {var.title}", "Observed Value",
                f"Deviation from {title}", hline=0.0))
    return section


# ============================================================================ percentiles
METHODS = {"haverage": "Weighted Average(Definition 1)",
           "waverage": "Weighted Average(Definition 2)",
           "round": "Rounded(Definition 3)",
           "empirical": "Empirical Distribution Function(Definition 4)",
           "aempirical": "Empirical Distribution Function with Averaging(Definition 5)"}
DEFAULT_PERCENTILES = (5, 10, 25, 50, 75, 90, 95)
_EPS = 1e-9


def _split(tc: float) -> tuple[int, float]:
    k = math.floor(tc)
    g = tc - k
    if g > 1 - _EPS * max(1.0, abs(tc)):
        return k + 1, 0.0
    if g < _EPS * max(1.0, abs(tc)):
        g = 0.0
    return int(k), g


def spss_percentile(x, p: float, method: str = "haverage") -> float:
    """One percentile (p in 0–100) by an SPSS definition (FREQUENCIES / EXAMINE)."""
    xs = np.sort(np.asarray(x, float))
    n = len(xs)
    if n == 0:
        return float("nan")
    q = p / 100.0

    def at(i: int) -> float:  # 1-based, clamped
        return float(xs[min(max(i, 1), n) - 1])

    if method == "haverage":
        k, g = _split((n + 1) * q)
        if k < 1:
            return at(1)
        if k >= n:
            return at(n)
        return (1 - g) * at(k) + g * at(k + 1)
    k, g = _split(n * q)
    if method == "waverage":
        if k < 1:
            return at(1)
        if k >= n:
            return at(n)
        return (1 - g) * at(k) + g * at(k + 1)
    if method == "round":
        return at(k if g < 0.5 else k + 1)
    if method == "empirical":
        return at(k if g == 0 else k + 1)
    if method == "aempirical":
        if g == 0:
            return (at(k) + at(k + 1)) / 2 if k < n else at(n)
        return at(k + 1)
    raise ValueError(f"unknown percentile method {method!r}")


def tukey_hinges(x) -> tuple[float, float, float]:
    """Tukey's hinges (EXAMINE): lower hinge, median, upper hinge."""
    xs = np.sort(np.asarray(x, float))
    n = len(xs)
    if n == 0:
        return (float("nan"),) * 3
    depth = (math.floor((n + 1) / 2) + 1) / 2
    lo = (xs[math.floor(depth) - 1] + xs[math.ceil(depth) - 1]) / 2
    hi = (xs[n - math.floor(depth)] + xs[n - math.ceil(depth)]) / 2
    return float(lo), float(np.median(xs)), float(hi)


def percentile_ci(x, p: float, level: float = 0.95) -> tuple[float, float, float]:
    """Distribution-free CI from binomial order statistics: (lower, upper, actual coverage)."""
    xs = np.sort(np.asarray(x, float))
    n = len(xs)
    q = p / 100.0
    if n == 0:
        return (float("nan"),) * 3
    alpha = 1 - level
    lo = int(stats.binom.ppf(alpha / 2, n, q))
    hi = int(stats.binom.ppf(1 - alpha / 2, n, q)) + 1
    lo, hi = max(lo, 1), min(hi, n)
    coverage = float(stats.binom.cdf(hi - 1, n, q) - stats.binom.cdf(lo - 1, n, q))
    return float(xs[lo - 1]), float(xs[hi - 1]), coverage


def percentiles(ds: Dataset, names: list[str], factor: str | None = None,
                points=DEFAULT_PERCENTILES, method: str = "haverage", hinges: bool = True,
                ci: bool = True, level: float = 0.95) -> Section:
    """Percentiles… (SPSS 29.0.1): percentile table plus order-statistic CIs."""
    points = [float(p) for p in points if 0 < float(p) < 100]
    section = Section("Percentiles")
    section.log(f"PERCENTILES VARIABLES={' '.join(names)}"
                + (f" BY {factor}" if factor else "")
                + f"\n  /PERCENTILES={' '.join(f'{p:g}' for p in points)}"
                + f"\n  /METHOD={method.upper()}" + (f"\n  /CI={level * 100:g}" if ci else "")
                + ".")
    if not names or not points:
        return section
    groups: list[tuple[str, pd.Series]] = [("", pd.Series(True, index=ds.df.index))]
    fvar = ds.var(factor) if factor else None
    if factor:
        f = ds.analysis_series(factor)
        cats = sorted(f.dropna().unique(), key=lambda v: (isinstance(v, str), v))
        groups = [(value_text(fvar, c), f == c) for c in cats]
    rh = 3 if factor else 2
    columns = [""] * rh + [("Percentiles", f"{p:g}") for p in points]
    rows, ci_rows = [], []
    cps = []
    for name in names:
        var = ds.var(name)
        y = numeric(ds, name)
        for gi, (glabel, mask) in enumerate(groups):
            x = y[mask].dropna().to_numpy(float)
            n_total = int(mask.sum())
            head = [var.title if gi == 0 else ""] + ([glabel] if factor else [])
            cps.append(head + [fmt_int(len(x)), fmt_int(n_total - len(x)), fmt_int(n_total)])
            if not len(x):
                continue
            d = var.decimals + 2
            rows.append(head + [METHODS[method]] +
                        [fmt(spss_percentile(x, p, method), d) for p in points])
            if hinges:
                lo, med, hi = tukey_hinges(x)
                cells = {25.0: lo, 50.0: med, 75.0: hi}
                rows.append(([""] * len(head)) + ["Tukey's Hinges"] +
                            [fmt(cells[p], d) if p in cells else "" for p in points])
            if ci:
                for pi, p in enumerate(points):
                    l, u, cov = percentile_ci(x, p, level)
                    first = pi == 0
                    ci_rows.append([var.title if gi == 0 and first else ""] +
                                   ([glabel if first else ""] if factor else []) +
                                   [f"{p:g}", fmt(spss_percentile(x, p, method), d),
                                    fmt(l, d), fmt(u, d), f"{100 * cov:.1f}%"])
    section.table(Table("Case Processing Summary",
                        [""] * (rh - 1) + ["Valid N", "Missing N", "Total N"], cps,
                        row_header_cols=rh - 1))
    section.table(Table("Percentiles", columns, rows, row_header_cols=rh))
    if ci and ci_rows:
        lv = f"{level * 100:g}%"
        section.table(Table("Percentile Confidence Intervals", [""] * (rh - 1) + [
            "Percentile", "Estimate", (f"{lv} Confidence Interval", "Lower Bound"),
            (f"{lv} Confidence Interval", "Upper Bound"), "Actual Coverage"], ci_rows,
            row_header_cols=rh - 1, footnotes=[
                "Distribution-free interval between two order statistics (binomial "
                "method); its actual coverage is at least the requested level when the "
                "sample is large enough."]))
    return section


# ============================================================================ normality
def mardia(x: np.ndarray) -> dict:
    """Mardia's multivariate skewness and kurtosis (MVN::mvn(mvnTest = "mardia"))."""
    x = np.asarray(x, float)
    n, p = x.shape
    d = x - x.mean(axis=0)
    s = np.cov(x, rowvar=False, ddof=0)
    dmat = d @ np.linalg.solve(s, d.T)
    b1 = float((dmat ** 3).sum() / n ** 2)
    b2 = float((np.diag(dmat) ** 2).sum() / n)
    df = p * (p + 1) * (p + 2) / 6
    skew = n * b1 / 6
    k = (p + 1) * (n + 1) * (n + 3) / (n * ((n + 1) * (p + 1) - 6))
    skew_small = n * k * b1 / 6
    kurt = (b2 - p * (p + 2)) / math.sqrt(8 * p * (p + 2) / n)
    return {"b1": b1, "b2": b2, "skew": skew, "skew_p": float(stats.chi2.sf(skew, df)),
            "skew_small": skew_small, "skew_small_p": float(stats.chi2.sf(skew_small, df)),
            "kurt": kurt, "kurt_p": float(2 * stats.norm.sf(abs(kurt))), "df": df}


def henze_zirkler(x: np.ndarray) -> dict:
    """Henze-Zirkler test (MVN::mvn(mvnTest = "hz")), lognormal p-value."""
    x = np.asarray(x, float)
    n, p = x.shape
    d = x - x.mean(axis=0)
    s = np.cov(x, rowvar=False, ddof=0)
    y = d @ np.linalg.solve(s, d.T)
    dj = np.diag(y)
    djk = -2 * y + dj[:, None] + dj[None, :]
    b = 1 / math.sqrt(2) * ((2 * p + 1) / 4) ** (1 / (p + 4)) * n ** (1 / (p + 4))
    hz = n * (np.exp(-(b ** 2) / 2 * djk).sum() / n ** 2
              - 2 * (1 + b ** 2) ** (-p / 2) * np.exp(-(b ** 2) / (2 * (1 + b ** 2)) * dj).sum()
              / n + (1 + 2 * b ** 2) ** (-p / 2))
    wb = (1 + b ** 2) * (1 + 3 * b ** 2)
    a = 1 + 2 * b ** 2
    mu = 1 - a ** (-p / 2) * (1 + p * b ** 2 / a + p * (p + 2) * b ** 4 / (2 * a ** 2))
    si2 = (2 * (1 + 4 * b ** 2) ** (-p / 2)
           + 2 * a ** (-p) * (1 + 2 * p * b ** 4 / a ** 2 + 3 * p * (p + 2) * b ** 8
                              / (4 * a ** 4))
           - 4 * wb ** (-p / 2) * (1 + 3 * p * b ** 4 / (2 * wb) + p * (p + 2) * b ** 8
                                   / (2 * wb ** 2)))
    pmu = math.log(math.sqrt(mu ** 4 / (si2 + mu ** 2)))
    psi = math.sqrt(math.log((si2 + mu ** 2) / mu ** 2))
    pval = float(stats.lognorm.sf(hz, s=psi, scale=math.exp(pmu)))
    return {"hz": float(hz), "p": pval}


def shapiro_francia(x: np.ndarray) -> float:
    """Shapiro-Francia W' (nortest::sf.test): squared correlation with Blom scores."""
    xs = np.sort(np.asarray(x, float))
    n = len(xs)
    m = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    return float(np.corrcoef(xs, m)[0, 1] ** 2)


def royston(x: np.ndarray) -> dict:
    """Royston's multivariate test (MVN::mvn(mvnTest = "royston"))."""
    x = np.asarray(x, float)
    n, p = x.shape
    if n < 4 or n > 5000:
        return {"h": float("nan"), "p": float("nan"), "edf": float("nan")}
    z = np.empty(p)
    for i in range(p):
        a = x[:, i]
        m2 = np.mean((a - a.mean()) ** 2)
        kurt = np.mean((a - a.mean()) ** 4) / m2 ** 2
        w = shapiro_francia(a) if kurt > 3 else float(stats.shapiro(a)[0])
        if n <= 11:
            g = -2.273 + 0.459 * n
            m = 0.544 - 0.39978 * n + 0.025054 * n ** 2 - 0.0006714 * n ** 3
            s = math.exp(1.3822 - 0.77857 * n + 0.062767 * n ** 2 - 0.0020322 * n ** 3)
            z[i] = (-math.log(g - math.log(1 - w)) - m) / s
        else:
            lx = math.log(n)
            m = -1.5861 - 0.31082 * lx - 0.083751 * lx ** 2 + 0.0038915 * lx ** 3
            s = math.exp(-0.4803 - 0.082676 * lx + 0.0030302 * lx ** 2)
            z[i] = (math.log(1 - w) - m) / s
    u = 0.715
    v = 0.21364 + 0.015124 * math.log(n) ** 2 - 0.0018034 * math.log(n) ** 3
    c = np.corrcoef(x, rowvar=False)
    nc = (c ** 5) * (1 - u * (1 - c) ** u / v)
    t = nc.sum() - p
    mc = t / (p * p - p) if p > 1 else 0.0
    edf = p / (1 + (p - 1) * mc)
    res = stats.norm.ppf(stats.norm.cdf(-z) / 2) ** 2
    h = edf * res.sum() / p
    return {"h": float(h), "p": float(stats.chi2.sf(h, edf)), "edf": float(edf)}


def univariate_tests(x: np.ndarray) -> dict:
    from statsmodels.stats.diagnostic import normal_ad

    x = np.asarray(x, float)
    n = len(x)
    out = {"n": n}
    out["sw"] = shapiro(x)
    d, p, lb = lilliefors(x)
    out["ks"] = (d, p, lb)
    if n >= 8 and np.ptp(x) > 0:
        ad, adp = normal_ad(x)
        out["ad"] = (float(ad), float(adp))
        s = stats.skewtest(x)
        out["skew"] = (float(s.statistic), float(s.pvalue))
        if n >= 20:
            k = stats.kurtosistest(x)
            out["kurt"] = (float(k.statistic), float(k.pvalue))
            o = stats.normaltest(x)
            out["omnibus"] = (float(o.statistic), float(o.pvalue))
    return out


def mahalanobis_sq(x: np.ndarray) -> np.ndarray:
    d = x - x.mean(axis=0)
    s = np.cov(x, rowvar=False, ddof=1)
    return np.einsum("ij,ij->i", d, np.linalg.solve(s, d.T).T)


def chisq_qq_plot(x: np.ndarray, title: str = "Chi-Square Q-Q Plot") -> bytes:
    d2 = np.sort(mahalanobis_sq(np.asarray(x, float)))
    n, p = x.shape
    q = stats.chi2.ppf((np.arange(1, n + 1) - 0.5) / n, p)
    hi = float(max(d2.max(), q.max()))
    return _points_plot(q, d2, title, f"Chi-Square Quantile (df = {p})",
                        "Squared Mahalanobis Distance", line=([0, hi], [0, hi]))


def normal_qq_plot(x: np.ndarray, title: str) -> bytes:
    xs = np.sort(np.asarray(x, float))
    n = len(xs)
    expected = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    mean, sd = xs.mean(), xs.std(ddof=1)
    lo, hi = xs.min(), xs.max()
    return _points_plot(xs, expected, title, "Observed Value", "Expected Normal",
                        line=([lo, hi], [(lo - mean) / sd, (hi - mean) / sd]) if sd > 0
                        else None)


def normality(ds: Dataset, names: list[str], univariate: bool = True,
              extra_tests: bool = True, mardia_test: bool = True, hz_test: bool = True,
              royston_test: bool = True, qq: bool = True, chisq_qq: bool = True,
              alpha: float = 0.05) -> Section:
    """Normality Analysis (SPSS 30): univariate and multivariate normality."""
    section = Section("Normality Analysis")
    mv = [t for t, on in (("MARDIA", mardia_test), ("HZ", hz_test), ("ROYSTON", royston_test))
          if on]
    section.log(f"NORMALITY VARIABLES={' '.join(names)}"
                + ("\n  /UNIVARIATE" if univariate else "")
                + (f"\n  /MULTIVARIATE={' '.join(mv)}" if mv else "")
                + f"\n  /ALPHA={alpha:g}.")
    if not names:
        return section
    frame = pd.concat({n: numeric(ds, n) for n in names}, axis=1)
    total = len(frame)
    frame = frame.dropna()
    n = len(frame)
    section.table(Table("Case Processing Summary", ["", ("Valid", "N"), ("Missing", "N"),
                                                    ("Total", "N")],
                        [["Cases", fmt_int(n), fmt_int(total - n), fmt_int(total)]],
                        footnotes=["Listwise deletion based on all variables in the "
                                   "procedure."]))
    x = frame.to_numpy(float)
    titles = [ds.var(v).title for v in names]
    if univariate and n:
        rows, lower = [], False
        for j, t in enumerate(titles):
            r = univariate_tests(x[:, j])
            d, p, lb = r["ks"]
            lower |= lb
            w, sp = r["sw"]
            row = [t, fmt(w), fmt_p(sp), fmt(d), fmt_p(p) + ("*" if lb else "")]
            if extra_tests:
                ad = r.get("ad", (np.nan, np.nan))
                sk = r.get("skew", (np.nan, np.nan))
                ku = r.get("kurt", (np.nan, np.nan))
                om = r.get("omnibus", (np.nan, np.nan))
                row += [fmt(ad[0]), fmt_p(ad[1]), fmt(sk[0]), fmt_p(sk[1]), fmt(ku[0]),
                        fmt_p(ku[1]), fmt(om[0]), fmt_p(om[1])]
            sig = [sp, p] + ([r.get("ad", (0, np.nan))[1]] if extra_tests else [])
            sig = [s for s in sig if s == s]
            row.append("Normal" if sig and min(sig) >= alpha else "Not normal")
            rows.append(row)
        cols = ["", ("Shapiro-Wilk", "Statistic"), ("Shapiro-Wilk", "Sig."),
                ("Kolmogorov-Smirnovᵃ", "Statistic"), ("Kolmogorov-Smirnovᵃ", "Sig.")]
        if extra_tests:
            cols += [("Anderson-Darling", "Statistic"), ("Anderson-Darling", "Sig."),
                     ("D'Agostino Skewness", "Z"), ("D'Agostino Skewness", "Sig."),
                     ("D'Agostino Kurtosis", "Z"), ("D'Agostino Kurtosis", "Sig."),
                     ("D'Agostino-Pearson", "K²"), ("D'Agostino-Pearson", "Sig.")]
        cols.append("Decision")
        notes = ["a. Lilliefors Significance Correction"]
        if lower:
            notes.append("*. This is a lower bound of the true significance.")
        notes.append(f"N = {n}. Decision: normal when every displayed Sig. ≥ {alpha:g}. "
                     "D'Agostino tests need N ≥ 8 (kurtosis and omnibus N ≥ 20).")
        section.table(Table("Tests of Univariate Normality", cols, rows, footnotes=notes))
    p = x.shape[1]
    if p >= 2 and n > p + 1 and mv:
        rows = []
        if mardia_test:
            m = mardia(x)
            rows.append(["Mardia Skewness", fmt(m["skew"]), fmt(m["df"], 0),
                         fmt_p(m["skew_p"]), _decision(m["skew_p"], alpha)])
            rows.append(["Mardia Skewness (small-sample corrected)", fmt(m["skew_small"]),
                         fmt(m["df"], 0), fmt_p(m["skew_small_p"]),
                         _decision(m["skew_small_p"], alpha)])
            rows.append(["Mardia Kurtosis (Z)", fmt(m["kurt"]), "",
                         fmt_p(m["kurt_p"]), _decision(m["kurt_p"], alpha)])
        if hz_test:
            h = henze_zirkler(x)
            rows.append(["Henze-Zirkler", fmt(h["hz"]), "", fmt_p(h["p"]),
                         _decision(h["p"], alpha)])
        if royston_test:
            r = royston(x)
            rows.append(["Royston (H)", fmt(r["h"]), fmt(r["edf"]), fmt_p(r["p"]),
                         _decision(r["p"], alpha)])
        section.table(Table("Tests of Multivariate Normality",
                            ["Test", "Statistic", "df", "Sig.", "Decision"], rows,
                            footnotes=[f"Variables: {', '.join(titles)}. N = {n}.",
                                       "Formulas as in the R package MVN (Korkmaz et al., "
                                       "2014). Mardia's skewness is chi-square with "
                                       "p(p+1)(p+2)/6 df; the kurtosis Z is two-sided; "
                                       "Henze-Zirkler uses a lognormal approximation; "
                                       "Royston's H is chi-square with the equivalent df."]))
    elif mv and p >= 2:
        section.note("Warning", "Multivariate tests need more valid cases than variables.",
                     style="warning")
    if qq and n > 2:
        for j, t in enumerate(titles):
            section.image(f"Normal Q-Q Plot of {t}", normal_qq_plot(x[:, j],
                                                                   f"Normal Q-Q Plot of {t}"))
    if chisq_qq and p >= 2 and n > p + 1:
        section.image("Chi-Square Q-Q Plot", chisq_qq_plot(x, "Chi-Square Q-Q Plot of "
                                                           "Squared Mahalanobis Distances"))
    return section


def _decision(p: float, alpha: float) -> str:
    if p != p:
        return ""
    return "Normal" if p >= alpha else "Not normal"

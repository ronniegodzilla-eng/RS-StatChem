"""Analyze > Quality Control: Control Charts, Pareto Charts and process capability (SPSS
SPCHART and the Pareto chart of GRAPH /PARETO), following IBM's "SPCHART Algorithms" chapter.

* **Charts**: X-bar and R, X-bar and s, individuals and moving range (span m), p, np, c and u.
  Centre lines and limits use IBM's formulas, which also cover unequal subgroup sizes:
  σ̂ = mean(R_i / d₂(n_i)) or mean(S_i / c₄(n_i)), limits x̄ ± A σ̂ / √n_i, R chart
  σ̂ d₂(n_i) ± A σ̂ d₃(n_i), s chart σ̂ c₄(n_i) ± A σ̂ √(1 − c₄²), p̄ = Σx / Σn, ū = Σx / Σn,
  lower limits truncated at 0 (and p/np upper limits at 1 / n_i); A = 3 sigma by default.
* **Constants**: d₂(n) = ∫ 1 − (1 − Φ(x))ⁿ − Φ(x)ⁿ dx and d₃(n) from the double integral of the
  range distribution (computed numerically, so every n works, not only tabulated ones);
  c₄(n) = √(2/(n−1)) Γ(n/2)/Γ((n−1)/2). They reproduce the ASTM / Montgomery tables
  (A₂, D₃, D₄, B₃, B₄ ...).
* **Control rules** (SPCHART /RULES): above +3σ / below −3σ, 2 of the last 3 beyond ±2σ,
  4 of the last 5 beyond ±1σ, 8 consecutive points on one side of the centre line, 6 points
  trending up / down and 14 points alternating; σ zones are (UCL − CL)/A per point.
* **Process capability / performance** (SPCHART /STATISTICS): CP, CpL, CpU, K, CpK, CR, CpM,
  Z-lower/upper/min/max and the estimated % outside the specification limits with the
  capability sigma from R̄, S̄, the mean moving range or the within-subgroup variance; the PP
  family with the overall standard deviation, and the observed % outside the limits.
* **Pareto charts**: counts (or sums of a variable) per category in descending order with the
  cumulative percentage line, simple or stacked by a second variable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy import integrate, special, stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, value_text
from rsstatchem.core.stats.ml_common import MLError, raw_column, sort_levels
from rsstatchem.core.stats.roc import figure, png, style

QCError = MLError


# ============================================================================ constants
@lru_cache(maxsize=None)
def d2(n: int) -> float:
    """Expected range of n standard normal observations."""
    n = int(n)
    if n < 2:
        return float("nan")
    f = lambda x: 1 - (1 - stats.norm.cdf(x)) ** n - stats.norm.cdf(x) ** n  # noqa: E731
    return float(integrate.quad(f, -np.inf, np.inf, limit=200)[0])


@lru_cache(maxsize=None)
def d3(n: int) -> float:
    """Standard deviation of the range of n standard normal observations (IBM's double
    integral)."""
    n = int(n)
    if n < 2:
        return float("nan")
    # Gauss-Legendre over x in [-L, L] and y in [-L, x]; the integrand vanishes in the tails
    L = 9.0
    t, w = np.polynomial.legendre.leggauss(240)
    x = L * t
    wx = L * w
    Fx = stats.norm.cdf(x)
    half = (x + L) / 2
    y = -L + half[:, None] * (t[None, :] + 1)          # (240, 240) nodes on [-L, x_i]
    wy = half[:, None] * w[None, :]
    Fy = stats.norm.cdf(y)
    g = 1 - Fx[:, None] ** n - (1 - Fy) ** n + (Fx[:, None] - Fy) ** n
    e = float((wx * (g * wy).sum(1)).sum())
    return float(math.sqrt(max(2 * e - d2(n) ** 2, 0.0)))


def c4(n: int) -> float:
    n = int(n)
    if n < 2:
        return float("nan")
    return float(math.sqrt(2 / (n - 1)) * math.exp(special.gammaln(n / 2)
                                                    - special.gammaln((n - 1) / 2)))


def constants(n: int, A: float = 3.0) -> dict:
    """The tabulated control-chart factors for subgroup size n (Montgomery Appendix VI)."""
    D2, D3, C4 = d2(n), d3(n), c4(n)
    return {"n": n, "A": A / math.sqrt(n), "A2": A / (D2 * math.sqrt(n)),
            "A3": A / (C4 * math.sqrt(n)), "c4": C4, "B3": max(0.0, 1 - A * math.sqrt(1 - C4 ** 2)
                                                             / C4),
            "B4": 1 + A * math.sqrt(1 - C4 ** 2) / C4, "d2": D2, "d3": D3,
            "D1": max(0.0, D2 - A * D3), "D2": D2 + A * D3,
            "D3": max(0.0, 1 - A * D3 / D2), "D4": 1 + A * D3 / D2}


# ============================================================================ charts
CHARTS = {"xbar_r": "X-Bar and R", "xbar_s": "X-Bar and s", "ir": "Individuals and Moving Range",
          "p": "p", "np": "np", "c": "c", "u": "u"}


@dataclass
class Chart:
    title: str
    statistic: str                  # y-axis label
    labels: list[str]
    values: np.ndarray
    cl: np.ndarray
    ucl: np.ndarray
    lcl: np.ndarray
    A: float = 3.0
    zone: np.ndarray | None = None  # one-sigma width per point
    n: np.ndarray | None = None
    violations: dict = field(default_factory=dict)  # point index -> [rule texts]

    def __post_init__(self):
        if self.zone is None:
            self.zone = (self.ucl - self.cl) / self.A


def _arr(v, k):
    return np.broadcast_to(np.asarray(v, float), (k,)).astype(float).copy()


def xbar_r(groups: list[np.ndarray], labels: list[str], A: float = 3.0) -> tuple[Chart, Chart]:
    n = np.array([len(g) for g in groups], float)
    K = len(groups)
    xb = np.array([g.mean() for g in groups])
    R = np.array([g.max() - g.min() for g in groups])
    grand = float((n * xb).sum() / n.sum())
    D2 = np.array([d2(int(k)) for k in n])
    D3 = np.array([d3(int(k)) for k in n])
    sig = float(np.mean(R / D2))
    half = A * sig / np.sqrt(n)
    cx = Chart("X-Bar", "Mean", labels, xb, _arr(grand, K), grand + half, grand - half, A, n=n)
    clr = sig * D2
    cr = Chart("R", "Range", labels, R, clr, clr + A * sig * D3,
               np.maximum(clr - A * sig * D3, 0), A, zone=sig * D3, n=n)
    return cx, cr


def xbar_s(groups: list[np.ndarray], labels: list[str], A: float = 3.0) -> tuple[Chart, Chart]:
    n = np.array([len(g) for g in groups], float)
    K = len(groups)
    xb = np.array([g.mean() for g in groups])
    S = np.array([g.std(ddof=1) for g in groups])
    grand = float((n * xb).sum() / n.sum())
    C4 = np.array([c4(int(k)) for k in n])
    sig = float(np.mean(S / C4))
    half = A * sig / np.sqrt(n)
    cx = Chart("X-Bar", "Mean", labels, xb, _arr(grand, K), grand + half, grand - half, A, n=n)
    cls = sig * C4
    w = sig * np.sqrt(1 - C4 ** 2)
    cs = Chart("s", "Standard Deviation", labels, S, cls, cls + A * w,
               np.maximum(cls - A * w, 0), A, zone=w, n=n)
    return cx, cs


def moving_ranges(y: np.ndarray, span: int = 2) -> np.ndarray:
    y = np.asarray(y, float)
    out = np.full(len(y), np.nan)
    for i in range(span - 1, len(y)):
        w = y[i - span + 1:i + 1]
        out[i] = w.max() - w.min()
    return out


def individuals(y: np.ndarray, labels: list[str], span: int = 2,
                A: float = 3.0) -> tuple[Chart, Chart]:
    y = np.asarray(y, float)
    N = len(y)
    if span < 2 or span > N:
        raise QCError(f"The span ({span}) must be between 2 and the number of units ({N}).")
    MR = moving_ranges(y, span)
    rbar = float(np.nanmean(MR))
    D2, D3 = d2(span), d3(span)
    m = float(y.mean())
    ci = Chart("Individuals", "Value", labels, y, _arr(m, N), _arr(m + A * rbar / D2, N),
               _arr(m - A * rbar / D2, N), A)
    cm = Chart("Moving Range", f"Moving Range (span {span})", labels, MR, _arr(rbar, N),
               _arr(rbar * (1 + A * D3 / D2), N), _arr(max(rbar * (1 - A * D3 / D2), 0), N), A,
               zone=_arr(rbar * D3 / D2, N))
    return ci, cm


def p_chart(x: np.ndarray, n: np.ndarray, labels: list[str], A: float = 3.0,
            np_chart: bool = False) -> Chart:
    x, n = np.asarray(x, float), np.asarray(n, float)
    K = len(x)
    pbar = float(x.sum() / n.sum())
    w = np.sqrt(pbar * (1 - pbar) / n)
    if np_chart:
        cl = n * pbar
        return Chart("np", "Number Nonconforming", labels, x, cl,
                     np.minimum(cl + A * n * w, n), np.maximum(cl - A * n * w, 0), A,
                     zone=n * w, n=n)
    p = x / n
    return Chart("p", "Proportion Nonconforming", labels, p, _arr(pbar, K),
                 np.minimum(pbar + A * w, 1), np.maximum(pbar - A * w, 0), A, zone=w, n=n)


def c_chart(x: np.ndarray, n: np.ndarray, labels: list[str], A: float = 3.0,
            u_chart: bool = False) -> Chart:
    x, n = np.asarray(x, float), np.asarray(n, float)
    K = len(x)
    ubar = float(x.sum() / n.sum())
    if u_chart:
        w = np.sqrt(ubar / n)
        return Chart("u", "Nonconformities per Unit", labels, x / n, _arr(ubar, K),
                     ubar + A * w, np.maximum(ubar - A * w, 0), A, zone=w, n=n)
    cl = n * ubar
    w = np.sqrt(cl)
    return Chart("c", "Number of Nonconformities", labels, x, cl, cl + A * w,
                 np.maximum(cl - A * w, 0), A, zone=w, n=n)


# ============================================================================ rules
RULES = {
    "ucl": "Above +3 sigma", "lcl": "Below -3 sigma",
    "r2upper": "2 points out of the last 3 above +2 sigma",
    "r2lower": "2 points out of the last 3 below -2 sigma",
    "r4upper": "4 points out of the last 5 above +1 sigma",
    "r4lower": "4 points out of the last 5 below -1 sigma",
    "r8upper": "8 consecutive points above the center line",
    "r8lower": "8 consecutive points below the center line",
    "trup": "6 consecutive points trending up", "trdown": "6 consecutive points trending down",
    "alternating": "14 consecutive points alternating up and down",
}


def apply_rules(chart: Chart, rules: list[str]) -> dict:
    """Index of each violating point -> list of rule texts (the point that completes the
    pattern is flagged, as SPCHART does)."""
    v, cl, z = chart.values, chart.cl, chart.zone
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(z > 0, (v - cl) / z, 0.0)
    ok = np.isfinite(v)
    out: dict[int, list[str]] = {}

    def flag(i, key):
        out.setdefault(int(i), []).append(RULES[key])

    idx = np.where(ok)[0]
    for key in rules:
        for pos, i in enumerate(idx):
            if key == "ucl" and v[i] > chart.ucl[i]:
                flag(i, key)
            elif key == "lcl" and v[i] < chart.lcl[i]:
                flag(i, key)
            elif key in ("r2upper", "r2lower", "r4upper", "r4lower"):
                need, last, lim = (2, 3, 2) if key.startswith("r2") else (4, 5, 1)
                if pos + 1 < last:
                    continue
                w = s[idx[pos - last + 1:pos + 1]]
                hit = (w > lim) if key.endswith("upper") else (w < -lim)
                cur = s[i] > lim if key.endswith("upper") else s[i] < -lim
                if cur and hit.sum() >= need:
                    flag(i, key)
            elif key in ("r8upper", "r8lower"):
                if pos + 1 < 8:
                    continue
                w = s[idx[pos - 7:pos + 1]]
                if (w > 0).all() if key == "r8upper" else (w < 0).all():
                    flag(i, key)
            elif key in ("trup", "trdown"):
                if pos + 1 < 6:
                    continue
                d = np.diff(v[idx[pos - 5:pos + 1]])
                if (d > 0).all() if key == "trup" else (d < 0).all():
                    flag(i, key)
            elif key == "alternating":
                if pos + 1 < 14:
                    continue
                d = np.sign(np.diff(v[idx[pos - 13:pos + 1]]))
                if (d != 0).all() and (d[1:] * d[:-1] < 0).all():
                    flag(i, key)
    return out


# ============================================================================ capability
SIGMA = {"rbar": "the mean of the sample group ranges",
         "sbar": "the mean of the sample group standard deviations",
         "mrbar": "the mean of the moving ranges",
         "variance": "the sample within-subgroup variance"}


def capability_sigma(groups: list[np.ndarray] | None, method: str, y: np.ndarray | None = None,
                     span: int = 2) -> float:
    if method == "mrbar":
        MR = moving_ranges(np.asarray(y, float), span)
        return float(np.nanmean(MR) / d2(span))
    if method == "rbar":
        return float(np.mean([(g.max() - g.min()) / d2(len(g)) for g in groups]))
    if method == "sbar":
        return float(np.mean([g.std(ddof=1) / c4(len(g)) for g in groups]))
    if method == "variance":
        ss = sum(((g - g.mean()) ** 2).sum() for g in groups)
        df = sum(len(g) - 1 for g in groups)
        return float(math.sqrt(ss / df))
    raise QCError(f"Unknown capability sigma {method!r}.")


def capability(x: np.ndarray, sigma: float, lsl: float | None = None, usl: float | None = None,
               target: float | None = None) -> dict:
    """IBM's capability (sigma = capability sigma) and performance (s = overall SD) indices."""
    x = np.asarray(x, float)
    m, s = float(x.mean()), float(x.std(ddof=1))
    both = lsl is not None and usl is not None and lsl < usl
    out: dict[str, float | None] = {"mean": m, "sigma": sigma, "s": s, "n": len(x)}

    def family(sd, p):
        r: dict[str, float | None] = {}
        ok = sd is not None and sd > 0 and math.isfinite(sd)
        r[f"{p}P"] = (usl - lsl) / (6 * sd) if ok and both else None
        r[f"{p}pL"] = (m - lsl) / (3 * sd) if ok and lsl is not None else None
        r[f"{p}pU"] = (usl - m) / (3 * sd) if ok and usl is not None else None
        if ok and (lsl is not None or usl is not None):
            r[f"{p}pK"] = min(v for v in (r[f"{p}pL"], r[f"{p}pU"]) if v is not None)
        else:
            r[f"{p}pK"] = None
        r[f"{p}R"] = 1 / r[f"{p}P"] if r[f"{p}P"] else None
        r[f"{p}pM"] = ((usl - lsl) / (6 * math.sqrt(sd ** 2 + (m - target) ** 2))
                       if ok and both and target is not None and lsl <= target <= usl else None)
        zl = (m - lsl) / sd if ok and lsl is not None else None
        zu = (usl - m) / sd if ok and usl is not None else None
        r[f"{p}ZL"], r[f"{p}ZU"] = zl, zu
        r[f"{p}Zmin"] = min(zl, zu) if ok and both else None
        r[f"{p}Zmax"] = max(zl, zu) if ok and both else None
        r[f"{p}Zout"] = ((1 - stats.norm.cdf(zu) + stats.norm.cdf(-zl)) * 100
                         if ok and both else None)
        return r

    out.update(family(sigma, "C"))
    out.update(family(s, "P"))
    out["K"] = 2 * abs((usl + lsl) / 2 - m) / (usl - lsl) if both else None
    if lsl is not None or usl is not None:
        bad = np.zeros(len(x), bool)
        if lsl is not None:
            bad |= x < lsl
        if usl is not None:
            bad |= x > usl
        out["observed_out"] = 100 * bad.mean()
    else:
        out["observed_out"] = None
    return out


CAP_ROWS = [("Capability Indices", [("CP", "CP"), ("CpL", "CpL"), ("CpU", "CpU"), ("K", "K"),
                                     ("CpK", "CpK"), ("CR", "CR"), ("CpM", "CpM"),
                                     ("Z-lower", "CZL"), ("Z-upper", "CZU"),
                                     ("Z-min", "CZmin"), ("Z-max", "CZmax"),
                                     ("Estimated % Outside SL", "CZout")]),
            ("Performance Indices", [("PP", "PP"), ("PpL", "PpL"), ("PpU", "PpU"),
                                      ("PpK", "PpK"), ("PR", "PR"), ("PpM", "PpM"),
                                      ("Z-lower", "PZL"), ("Z-upper", "PZU"),
                                      ("Z-min", "PZmin"), ("Z-max", "PZmax"),
                                      ("Estimated % Outside SL", "PZout")])]


def capability_table(cap: dict, sigma_method: str, lsl, usl, target) -> Table:
    rows = [["Act. % Outside SL", "", fmt(cap["observed_out"], 1)]] \
        if cap.get("observed_out") is not None else []
    for group, items in CAP_ROWS:
        for label, key in items:
            v = cap.get(key)
            rows.append([group, label, fmt(v, 3) if v is not None else ""])
    head = [["Mean", "", fmt(cap["mean"], 4)], ["Std. Deviation", "", fmt(cap["s"], 4)],
            ["N", "", fmt_int(cap["n"])], ["LSL", "", fmt(lsl, 4) if lsl is not None else ""],
            ["USL", "", fmt(usl, 4) if usl is not None else ""],
            ["Target", "", fmt(target, 4) if target is not None else ""],
            ["Capability Sigma", "", fmt(cap["sigma"], 4)]]
    return Table("Process Statistics", ["", "", ""], head + rows, row_header_cols=2,
                 footnotes=["The normal distribution is assumed. LSL and USL are the lower and "
                            "upper specification limits.",
                            f"The capability sigma is based on {SIGMA[sigma_method]}."])


# ============================================================================ plotting
def chart_png(chart: Chart, var: str, sub_label: str) -> bytes:
    k = len(chart.values)
    fig = figure(max(6.0, min(11.0, 0.18 * k + 4)), 3.8)
    ax = fig.add_subplot(111)
    xs = np.arange(1, k + 1)
    ax.plot(xs, chart.values, "-o", color="#1f5fa8", ms=3.5, lw=1)
    step = dict(where="mid", lw=1)
    ax.step(xs, chart.ucl, color="#c0392b", label="UCL", **step)
    ax.step(xs, chart.lcl, color="#c0392b", ls="--", label="LCL", **step)
    ax.step(xs, chart.cl, color="#2e7d32", label="Center", **step)
    bad = sorted(chart.violations)
    if bad:
        ax.plot(xs[bad], chart.values[bad], "s", mfc="none", mec="#c0392b", ms=8,
                label="Violation")
        ax.legend(fontsize=7, loc="best")
    if k <= 30:
        ax.set_xticks(xs, chart.labels, rotation=90 if k > 12 else 0, fontsize=8)
    ax.set_xlim(0.5, k + 0.5 + max(1.0, 0.12 * k))
    for arr, name, col in ((chart.ucl, "UCL", "#c0392b"), (chart.cl, "Center", "#2e7d32"),
                           (chart.lcl, "LCL", "#c0392b")):
        if np.isfinite(arr[-1]):
            same = np.allclose(arr, arr[-1])
            ax.text(k + 0.7, arr[-1], f"{name}" + (f" = {arr[-1]:.4g}" if same else ""),
                    va="center", fontsize=7, color=col)
    style(ax, f"Control Chart: {var}", sub_label, chart.statistic)
    return png(fig)


def capability_png(x: np.ndarray, var: str, lsl, usl, target, sigma: float) -> bytes:
    fig = figure(6.0, 4.0)
    ax = fig.add_subplot(111)
    ax.hist(x, bins="auto", density=True, color="#9ecae1", edgecolor="white")
    m, s = x.mean(), x.std(ddof=1)
    lo = min([x.min()] + [v for v in (lsl, usl) if v is not None])
    hi = max([x.max()] + [v for v in (lsl, usl) if v is not None])
    t = np.linspace(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo), 300)
    ax.plot(t, stats.norm.pdf(t, m, s), color="#1f5fa8", label="Overall (s)")
    if sigma > 0:
        ax.plot(t, stats.norm.pdf(t, m, sigma), color="#2e7d32", ls="--",
                label="Within (capability sigma)")
    for v, lab in ((lsl, "LSL"), (usl, "USL"), (target, "Target")):
        if v is not None:
            ax.axvline(v, color="#c0392b" if lab != "Target" else "#555", lw=1.2)
            ax.text(v, ax.get_ylim()[1] * 0.95, lab, ha="center", fontsize=8,
                    color="#c0392b" if lab != "Target" else "#555")
    ax.legend(fontsize=7)
    style(ax, f"Process Capability: {var}", var, "Density")
    return png(fig)


# ============================================================================ procedure
def _subgroups(ds: Dataset, var: str, subgroup: str):
    x = raw_column(ds, var)
    g = raw_column(ds, subgroup)
    ok = x.notna() & g.notna()
    levels = sort_levels(g[ok].tolist())
    groups = [x[ok & (g == lv)].to_numpy(float) for lv in levels]
    labels = [value_text(ds.var(subgroup), lv) for lv in levels]
    return groups, labels


def _columns(ds: Dataset, samples: list[str], label_var: str | None):
    X = np.column_stack([raw_column(ds, s).to_numpy(float) for s in samples])
    groups, labels = [], []
    lab = raw_column(ds, label_var) if label_var else None
    for i in range(len(X)):
        g = X[i][np.isfinite(X[i])]
        if len(g):
            groups.append(g)
            labels.append(value_text(ds.var(label_var), lab.iloc[i]) if lab is not None
                          else str(i + 1))
    return groups, labels


def control_chart(ds: Dataset, chart: str, variables: list[str] | None = None,
                  subgroup: str | None = None, samples: list[str] | None = None,
                  label_var: str | None = None, span: int = 2, A: float = 3.0,
                  min_size: int = 2, rules: list[str] | None = None,
                  nonconforming_value=None, sample_size: float | str | None = None,
                  capability_stats: bool = False, lsl: float | None = None,
                  usl: float | None = None, target: float | None = None,
                  sigma_method: str | None = None, capability_plot: bool = False,
                  show_limits_table: bool = True, second_chart: bool = True) -> Section:
    """One SPCHART run. ``chart`` in CHARTS. Data organisation: cases are units (``variables``
    + ``subgroup``; I-MR needs no subgroup) or cases are subgroups (``samples`` columns for
    X-bar charts; for p/np/c/u a count variable in ``variables`` with ``sample_size`` = a
    constant or a variable name)."""
    if chart not in CHARTS:
        raise QCError(f"Unknown chart {chart!r}.")
    rules = list(rules or [])
    section = Section("Control Charts")
    var = (variables or samples or ["?"])[0]
    sub_label = ds.var(subgroup).title if subgroup else (
        ds.var(label_var).title if label_var else "Sequence")
    charts: list[Chart] = []
    groups = None
    units = None
    note = []
    if chart in ("xbar_r", "xbar_s"):
        if samples:
            groups, labels = _columns(ds, samples, label_var)
            var = ", ".join(samples) if len(samples) <= 3 else f"{samples[0]} ... {samples[-1]}"
        else:
            if not variables or not subgroup:
                raise QCError("Select a process measurement and a subgroup variable.")
            groups, labels = _subgroups(ds, variables[0], subgroup)
        keep = [i for i, g in enumerate(groups) if len(g) >= max(2, int(min_size))]
        dropped = len(groups) - len(keep)
        groups = [groups[i] for i in keep]
        labels = [labels[i] for i in keep]
        if dropped:
            note.append(f"{dropped} subgroup(s) with fewer than {max(2, int(min_size))} cases "
                        "were dropped.")
        if len(groups) < 2:
            raise QCError("At least two subgroups with enough cases are needed.")
        charts = list(xbar_r(groups, labels, A) if chart == "xbar_r" else
                      xbar_s(groups, labels, A))
        units = np.concatenate(groups)
        syntax = f"SPCHART /{'XR' if chart == 'xbar_r' else 'XS'}={var}" + (
            f" BY {subgroup}" if subgroup and not samples else "")
    elif chart == "ir":
        if not variables:
            raise QCError("Select a process measurement.")
        y = raw_column(ds, variables[0])
        ok = y.notna().to_numpy()
        units = y.to_numpy(float)[ok]
        if label_var:
            lab = raw_column(ds, label_var)
            labels = [value_text(ds.var(label_var), lab.iloc[i]) for i in np.where(ok)[0]]
        else:
            labels = [str(i + 1) for i in np.where(ok)[0]]
        charts = list(individuals(units, labels, int(span), A))
        syntax = f"SPCHART /IR={variables[0]}" + (f" BY {label_var}" if label_var else "") + \
            f"\n  /SPAN={int(span)}"
    else:
        if not variables:
            raise QCError("Select a characteristic variable.")
        if subgroup:   # cases are units
            x = raw_column(ds, variables[0])
            g = raw_column(ds, subgroup)
            ok = x.notna() & g.notna()
            levels = sort_levels(g[ok].tolist())
            labels = [value_text(ds.var(subgroup), lv) for lv in levels]
            if chart in ("p", "np"):
                if nonconforming_value is None:
                    raise QCError("Specify the value that marks a nonconforming unit.")
                bad = (x == nonconforming_value)
                cnt = np.array([float((bad & ok & (g == lv)).sum()) for lv in levels])
            else:
                cnt = np.array([float(x[ok & (g == lv)].sum()) for lv in levels])
            n = np.array([float((ok & (g == lv)).sum()) for lv in levels])
        else:          # cases are subgroups
            x = raw_column(ds, variables[0]).to_numpy(float)
            if isinstance(sample_size, str):
                n = raw_column(ds, sample_size).to_numpy(float)
            elif sample_size is not None:
                n = np.full(len(x), float(sample_size))
            else:
                raise QCError("Specify the sample size (a constant or a variable).")
            ok = np.isfinite(x) & np.isfinite(n) & (n > 0)
            if label_var:
                lab = raw_column(ds, label_var)
                labels = [value_text(ds.var(label_var), lab.iloc[i]) for i in np.where(ok)[0]]
            else:
                labels = [str(i + 1) for i in np.where(ok)[0]]
            cnt, n = x[ok], n[ok]
            if chart in ("p", "np"):
                over = cnt > n
                if over.any():
                    note.append(f"{int(over.sum())} subgroup(s) whose count exceeds the sample "
                                "size were dropped.")
                    labels = [lb for lb, o in zip(labels, over) if not o]
                    cnt, n = cnt[~over], n[~over]
        if len(cnt) < 2:
            raise QCError("At least two subgroups are needed.")
        if chart in ("np", "c") and len(set(n.tolist())) > 1:
            note.append(f"The subgroups have unequal sizes; a {'p' if chart == 'np' else 'u'} "
                        "chart is more appropriate.")
        charts = [p_chart(cnt, n, labels, A, chart == "np") if chart in ("p", "np")
                  else c_chart(cnt, n, labels, A, chart == "u")]
        syntax = f"SPCHART /{chart.upper()}={variables[0]}" + (f" BY {subgroup}" if subgroup
                                                               else "")
    section.log(syntax + f"\n  /SIGMAS={A:g}" + (f"\n  /RULES={' '.join(r.upper() for r in rules)}"
                                                  if rules else "") + ".")
    if not second_chart:
        charts = charts[:1]
    for ch in charts:
        ch.violations = apply_rules(ch, rules) if rules else {}
        section.image(f"{ch.title} Chart: {var}", chart_png(ch, var, sub_label), width=620)
    if show_limits_table:
        rows = []
        for ch in charts:
            same = np.allclose(ch.ucl, ch.ucl[0], equal_nan=True) and \
                np.allclose(ch.lcl, ch.lcl[0], equal_nan=True)
            if same:
                rows.append([ch.title, fmt(ch.ucl[0], 4), fmt(ch.cl[0], 4), fmt(ch.lcl[0], 4)])
            else:
                rows.append([ch.title, "varies", fmt(np.nanmean(ch.cl), 4) if np.allclose(
                    ch.cl, ch.cl[0]) else "varies", "varies"])
        section.table(Table("Control Limits", ["Chart", "UCL", "Center Line", "LCL"], rows,
                            footnotes=[f"{A:g} sigma limits."]))
    if rules:
        rows = []
        for ch in charts:
            for i in sorted(ch.violations):
                rows.append([ch.title, ch.labels[i], fmt(ch.values[i], 4),
                             "; ".join(ch.violations[i])])
        if rows:
            section.table(Table("Rule Violations", ["Chart", sub_label, "Value",
                                                    "Violations for Points"], rows,
                                footnotes=[f"{len(rows)} point(s) violate control rules."]))
        else:
            section.note("Rule Violations", "All control rules are satisfied.")
    if capability_stats and units is not None and chart in ("xbar_r", "xbar_s", "ir"):
        if lsl is None and usl is None:
            raise QCError("Process capability needs at least one specification limit.")
        method = sigma_method or {"xbar_r": "rbar", "xbar_s": "sbar", "ir": "mrbar"}[chart]
        if method == "mrbar" and chart != "ir":
            method = "rbar"
        if method in ("rbar", "sbar", "variance") and chart == "ir":
            method = "mrbar"
        sig = capability_sigma(groups, method, units, int(span))
        cap = capability(units, sig, lsl, usl, target)
        section.table(capability_table(cap, method, lsl, usl, target))
        if capability_plot:
            section.image("Process Capability", capability_png(units, var, lsl, usl, target,
                                                               sig), width=520)
    for t in note:
        section.note("", t, style="warning")
    section.result = charts  # type: ignore[attr-defined]
    return section


def process_capability(ds: Dataset, var: str, subgroup: str | None = None,
                       lsl: float | None = None, usl: float | None = None,
                       target: float | None = None, sigma_method: str = "auto",
                       span: int = 2, plot: bool = True) -> Section:
    """Analyze > Quality Control > Process Capability: the SPCHART capability statistics
    without the charts (subgroups give R̄-based sigma, single units the moving range)."""
    if lsl is None and usl is None:
        raise QCError("Specify at least one specification limit.")
    section = Section("Process Capability")
    if subgroup:
        groups, _ = _subgroups(ds, var, subgroup)
        groups = [g for g in groups if len(g) >= 2]
        units = np.concatenate(groups)
        method = "rbar" if sigma_method == "auto" else sigma_method
    else:
        groups = None
        units = raw_column(ds, var).dropna().to_numpy(float)
        method = "mrbar"
    if len(units) < 3:
        raise QCError("Too few valid cases.")
    section.log(f"SPCHART /{'XR' if subgroup else 'IR'}={var}" + (f" BY {subgroup}" if subgroup
                                                                 else "")
                + f"\n  /STATISTICS=CP CPL CPU K CPK CR CPM CZL CZU CZMIN CZMAX CZOUT PP PPL "
                  f"PPU PPK PR PPM PZL PZU PZMIN PZMAX PZOUT AZOUT\n  /CAPSIGMA={method.upper()}"
                + (f"\n  /LSL={lsl:g}" if lsl is not None else "")
                + (f" /USL={usl:g}" if usl is not None else "")
                + (f" /TARGET={target:g}" if target is not None else "") + ".")
    sig = capability_sigma(groups, method, units, int(span))
    cap = capability(units, sig, lsl, usl, target)
    section.table(capability_table(cap, method, lsl, usl, target))
    if plot:
        section.image("Process Capability", capability_png(units, var, lsl, usl, target, sig),
                      width=520)
    return section


# ============================================================================ Pareto
def pareto_counts(categories, values=None) -> tuple[list, np.ndarray]:
    cats = list(categories)
    vals = np.ones(len(cats)) if values is None else np.asarray(values, float)
    totals: dict = {}
    for c, v in zip(cats, vals):
        if np.isfinite(v):
            totals[c] = totals.get(c, 0.0) + v
    order = sorted(totals, key=lambda c: (-totals[c], sort_levels([c])))
    return order, np.array([totals[c] for c in order])


def pareto(ds: Dataset, category: str, value: str | None = None, stack: str | None = None,
           cumulative: bool = True, table: bool = True) -> Section:
    """Analyze > Quality Control > Pareto Charts (simple or stacked; counts or sums)."""
    section = Section("Pareto Chart")
    cat = raw_column(ds, category)
    ok = cat.notna()
    vals = None
    if value:
        v = raw_column(ds, value)
        ok &= v.notna()
        vals = v[ok].to_numpy(float)
    keys = cat[ok].tolist()
    order, tot = pareto_counts(keys, vals)
    labels = [value_text(ds.var(category), c) for c in order]
    section.log(f"GRAPH /PARETO({'CUM' if cumulative else 'NOCUM'} "
                f"{'STACKED' if stack else 'SIMPLE'})="
                + (f"SUM({value})" if value else "COUNT") + f" BY {category}"
                + (f" BY {stack}" if stack else "") + ".")
    total = tot.sum()
    if total <= 0:
        raise QCError("The Pareto chart needs positive counts or sums.")
    cum = np.cumsum(tot) / total * 100
    fig = figure(max(5.5, 0.5 * len(order) + 3), 4.2)
    ax = fig.add_subplot(111)
    xs = np.arange(len(order))
    if stack:
        st = raw_column(ds, stack)[ok]
        slevels = sort_levels(st.dropna().tolist())
        bottom = np.zeros(len(order))
        vv = vals if vals is not None else np.ones(int(ok.sum()))
        for j, lv in enumerate(slevels):
            h = np.array([vv[(np.array(keys, dtype=object) == c) & (st.to_numpy(object) == lv)]
                          .sum() for c in order])
            ax.bar(xs, h, bottom=bottom, color=f"C{j}", edgecolor="white",
                   label=value_text(ds.var(stack), lv))
            bottom += h
        ax.legend(fontsize=7, title=ds.var(stack).title)
    else:
        ax.bar(xs, tot, color="#1f5fa8", edgecolor="white")
    ax.set_xticks(xs, labels, rotation=45 if len(order) > 6 else 0, ha="right"
                  if len(order) > 6 else "center", fontsize=8)
    style(ax, f"Pareto Chart of {ds.var(category).title}", ds.var(category).title,
          f"Sum of {ds.var(value).title}" if value else "Count")
    if cumulative:
        ax2 = ax.twinx()
        ax2.plot(xs, cum, "-o", color="#c0392b", ms=3.5)
        ax2.set_ylim(0, 105)
        ax2.set_ylabel("Cumulative Percent")
        ax.set_ylim(0, total * 1.05)
    section.image("Pareto Chart", png(fig), width=560)
    if table:
        rows = [[lab, fmt(t, 0 if value is None else 3), fmt(100 * t / total, 1),
                 fmt(c, 1)] for lab, t, c in zip(labels, tot, cum)]
        rows.append(["Total", fmt(total, 0 if value is None else 3), "100.0", ""])
        section.table(Table("Pareto Table", [ds.var(category).title,
                                             "Sum" if value else "Count", "Percent",
                                             "Cumulative Percent"], rows))
    return section


def selftest() -> None:
    assert abs(d2(5) - 2.326) < 1e-3 and abs(d3(5) - 0.864) < 1e-3
    assert abs(c4(5) - 0.9400) < 1e-4
    r = np.random.default_rng(3)
    groups = [r.normal(10, 1, 5) for _ in range(20)]
    cx, cr = xbar_r(groups, [str(i) for i in range(20)])
    assert np.allclose(cx.ucl - cx.cl, constants(5)["A2"] * cr.cl[0])
    cap = capability(np.concatenate(groups), 1.0, 7, 13, 10)
    assert abs(cap["CP"] - 1.0) < 1e-12

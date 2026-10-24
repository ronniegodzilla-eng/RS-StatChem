"""Small matplotlib charts rendered to PNG bytes for the Output Viewer.

Uses the object-oriented Figure API with the Agg canvas, so it never touches a GUI backend.
"""
from __future__ import annotations

import io

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy import stats

BAR = "#C62828"
EDGE = "#7f1010"
LINE = "#1f1f1f"


def _figure(width=6.0, height=4.0) -> Figure:
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


def bar_chart(labels: list[str], counts: list[float], title: str, ylabel="Frequency") -> bytes:
    fig = _figure()
    ax = fig.add_subplot()
    ax.bar(range(len(labels)), counts, color=BAR, edgecolor=EDGE, width=0.65)
    ax.set_xticks(range(len(labels)), labels, rotation=30 if len(labels) > 6 else 0,
                  ha="right" if len(labels) > 6 else "center")
    _style(ax, title, "", ylabel)
    return _png(fig)


def pie_chart(labels: list[str], counts: list[float], title: str) -> bytes:
    fig = _figure(5.5, 4.2)
    ax = fig.add_subplot()
    colors = ["#C62828", "#EF5350", "#8E1B1B", "#FF8A80", "#B71C1C", "#E57373", "#5D1010",
              "#FFCDD2", "#D32F2F", "#9E9E9E"]
    ax.pie(counts, labels=labels, autopct="%1.1f%%", startangle=90, counterclock=False,
           colors=[colors[i % len(colors)] for i in range(len(counts))],
           wedgeprops={"edgecolor": "white", "linewidth": 1}, textprops={"fontsize": 9})
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("equal")
    return _png(fig)


def histogram(values: np.ndarray, title: str, xlabel: str, normal_curve: bool = True) -> bytes:
    values = np.asarray(values, dtype=float)
    fig = _figure()
    ax = fig.add_subplot()
    counts, edges, _ = ax.hist(values, bins="sturges", color=BAR, edgecolor=EDGE, alpha=0.9)
    if normal_curve and len(values) > 1:
        mean, sd = values.mean(), values.std(ddof=1)
        if sd > 0:
            xs = np.linspace(edges[0], edges[-1], 200)
            width = edges[1] - edges[0]
            ax.plot(xs, stats.norm.pdf(xs, mean, sd) * len(values) * width, color=LINE, lw=1.6)
        ax.text(0.98, 0.95, f"Mean = {mean:.2f}\nStd. Dev. = {sd:.3f}\nN = {len(values)}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                bbox={"facecolor": "white", "edgecolor": "#cccccc", "boxstyle": "round"})
    _style(ax, title, xlabel, "Frequency")
    return _png(fig)


def boxplot(samples: list[np.ndarray], labels: list[str], title: str, ylabel: str) -> bytes:
    """SPSS-style boxplot: whiskers at 1.5 IQR, outliers (o) and extremes (*) beyond 3 IQR."""
    fig = _figure(max(4.0, 1.2 * len(samples) + 2.5), 4.2)
    ax = fig.add_subplot()
    ax.boxplot(samples, tick_labels=labels, widths=0.5, patch_artist=True, whis=1.5,
               medianprops={"color": LINE, "linewidth": 2},
               boxprops={"facecolor": "#F6C9C7", "edgecolor": EDGE},
               whiskerprops={"color": EDGE}, capprops={"color": EDGE},
               flierprops={"marker": "o", "markerfacecolor": "white",
                           "markeredgecolor": EDGE, "markersize": 5})
    for i, s in enumerate(samples, start=1):
        s = np.asarray(s, float)
        if len(s) < 4:
            continue
        q1, q3 = np.percentile(s, [25, 75], method="weibull")
        extreme = s[(s < q1 - 3 * (q3 - q1)) | (s > q3 + 3 * (q3 - q1))]
        ax.plot([i] * len(extreme), extreme, linestyle="none", marker="*", color=BAR,
                markersize=9)
    _style(ax, title, "", ylabel)
    return _png(fig)


def qq_plot(values: np.ndarray, title: str) -> bytes:
    """Normal Q-Q plot with Blom plotting positions (as SPSS)."""
    x = np.sort(np.asarray(values, float))
    n = len(x)
    fig = _figure(5.2, 4.2)
    ax = fig.add_subplot()
    if n >= 2:
        mean, sd = x.mean(), x.std(ddof=1)
        expected = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
        ax.plot(x, expected, "o", color=BAR, markeredgecolor=EDGE, markersize=5)
        if sd > 0:
            lo, hi = x.min(), x.max()
            ax.plot([lo, hi], [(lo - mean) / sd, (hi - mean) / sd], color=LINE, lw=1.3)
    _style(ax, title, "Observed Value", "Expected Normal")
    return _png(fig)


def pp_plot(residuals: np.ndarray, title: str) -> bytes:
    """Normal P-P plot of regression standardized residuals."""
    z = np.sort(np.asarray(residuals, float))
    n = len(z)
    fig = _figure(5.2, 4.6)
    ax = fig.add_subplot()
    observed = (np.arange(1, n + 1) - 0.375) / (n + 0.25)
    sd = z.std(ddof=1) if n > 1 else 1.0
    expected = stats.norm.cdf((z - z.mean()) / (sd if sd > 0 else 1.0))
    ax.plot(observed, expected, "o", color=BAR, markeredgecolor=EDGE, markersize=4.5)
    ax.plot([0, 1], [0, 1], color=LINE, lw=1.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _style(ax, title, "Observed Cum Prob", "Expected Cum Prob")
    return _png(fig)


def scatter(x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str,
            zero_lines: bool = False, fit_line: bool = False) -> bytes:
    fig = _figure(5.4, 4.3)
    ax = fig.add_subplot()
    ax.scatter(x, y, s=22, color=BAR, edgecolors=EDGE, linewidths=0.6, alpha=0.9)
    if zero_lines:
        ax.axhline(0, color="#888888", lw=0.9, ls="--")
        ax.axvline(0, color="#888888", lw=0.9, ls="--")
    if fit_line and len(x) > 1 and np.ptp(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.array([np.min(x), np.max(x)])
        ax.plot(xs, intercept + slope * xs, color=LINE, lw=1.4)
    _style(ax, title, xlabel, ylabel)
    return _png(fig)


def means_plot(labels: list[str], means: list[float], title: str, xlabel: str,
               ylabel: str) -> bytes:
    fig = _figure(5.4, 4.0)
    ax = fig.add_subplot()
    ax.plot(range(len(labels)), means, marker="o", color=BAR, markeredgecolor=EDGE, lw=1.8)
    ax.set_xticks(range(len(labels)), labels)
    _style(ax, title, xlabel, ylabel)
    return _png(fig)

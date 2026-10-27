"""SPSS Chart Builder / Legacy Dialogs charts that Graph Studio has no template for.

Each chart is drawn with matplotlib's object API into a PNG for the Output Viewer (SPSS puts
Chart Builder charts in the Viewer too): scatterplot matrix, violin plot, population pyramid,
high-low(-close), range bar, difference area, dual Y axes, dot plot, drop-line, summary point
(error bar) chart, stacked histogram, frequency polygon, 1-D boxplot, regression variable plots
and panelled histograms/scatterplots (rows by a panel variable). No Qt here.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy import stats

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats.common import value_text

PALETTE = ["#1f5fa8", "#d9822b", "#3a923a", "#c23b3b", "#7b52ab", "#8c564b", "#d062a6",
           "#6b6b6b", "#a8a82e", "#25a7b8"]


class ChartError(ValueError):
    pass


@dataclass(frozen=True)
class Slot:
    key: str
    label: str
    numeric: bool = True
    multiple: bool = False
    required: bool = True


@dataclass(frozen=True)
class ChartType:
    key: str
    label: str
    category: str
    slots: tuple
    options: dict = field(default_factory=dict)
    hint: str = ""


_GROUP = Slot("group", "Group / color by (optional):", False, required=False)
_PANEL = Slot("panel", "Panel by rows (optional):", False, required=False)
_CAT = Slot("x", "Category axis:", False)

CHARTS: dict[str, ChartType] = {c.key: c for c in [
    ChartType("splom", "Scatterplot Matrix", "Scatter/Dot",
              (Slot("vars", "Variables (2-8):", multiple=True), _GROUP),
              {"fit": False, "diagonal": "histogram"},
              "One scatterplot per pair of variables; histograms on the diagonal."),
    ChartType("panel_scatter", "Panelled Scatterplot", "Scatter/Dot",
              (Slot("x", "X axis:"), Slot("y", "Y axis:"), _GROUP, _PANEL), {"fit": False}),
    ChartType("dot", "Simple Dot Plot", "Scatter/Dot", (Slot("y", "Variable:"), _GROUP),
              hint="Each case is a dot stacked over its value (dot density)."),
    ChartType("dropline", "Drop-line", "Scatter/Dot",
              (_CAT, Slot("y", "Y (mean):", multiple=True)), {"summary": "mean"},
              "Means of each variable per category joined by vertical drop lines."),
    ChartType("summary_point", "Summary Point (Error Bar)", "Scatter/Dot",
              (_CAT, Slot("y", "Y:"), _GROUP), {"error": "ci", "level": 95.0},
              "Mean with error bars (95% CI, ±2 SE or ±1 SD) by category."),
    ChartType("violin", "Violin Plot", "Boxplot",
              (Slot("y", "Variable:"), Slot("x", "Category axis (optional):", False,
                                            required=False)), {"box": True}),
    ChartType("box1d", "1-D Boxplot", "Boxplot", (Slot("vars", "Variables:", multiple=True),)),
    ChartType("stacked_hist", "Stacked Histogram", "Histogram",
              (Slot("y", "Variable:"), Slot("group", "Stack by:", False)), {"bins": 0}),
    ChartType("freq_polygon", "Frequency Polygon", "Histogram", (Slot("y", "Variable:"), _GROUP),
              {"bins": 0}),
    ChartType("pyramid", "Population Pyramid", "Histogram",
              (Slot("y", "Distribution variable:", False), Slot("split", "Split by (2 groups):",
                                                                 False)),
              {"bins": 0, "percent": False},
              "Back-to-back horizontal histograms of two groups (e.g. age by sex)."),
    ChartType("panel_hist", "Panelled Histogram", "Histogram",
              (Slot("y", "Variable:"), Slot("panel", "Panel by rows:", False)),
              {"bins": 0, "normal": True}),
    ChartType("highlow", "High-Low-Close", "High-Low",
              (Slot("x", "Category / time axis:", False), Slot("high", "High:"),
               Slot("low", "Low:"), Slot("close", "Close (optional):", required=False))),
    ChartType("rangebar", "Simple Range Bar", "High-Low",
              (Slot("x", "Category axis:", False), Slot("high", "High (mean):"),
               Slot("low", "Low (mean):"))),
    ChartType("difference", "Difference Area", "High-Low",
              (Slot("x", "X / time axis:"), Slot("y1", "Series 1:"), Slot("y2", "Series 2:")),
              hint="Two lines with the area between them shaded by which one is higher."),
    ChartType("dualaxis", "Dual Y Axes (bar + line)", "Dual Axes",
              (_CAT, Slot("y1", "Left axis (bars, mean):"),
               Slot("y2", "Right axis (line, mean):")), {"summary": "mean"}),
    ChartType("regplots", "Regression Variable Plots", "Regression",
              (Slot("y", "Dependent:"), Slot("vars", "Independent variables:", multiple=True)),
              {"fit": True}),
]}
CATEGORIES = list(dict.fromkeys(c.category for c in CHARTS.values()))


# ------------------------------------------------------------------ helpers
def _num(ds: Dataset, name: str) -> np.ndarray:
    return pd.to_numeric(ds.analysis_series(name), errors="coerce").to_numpy(float)


def _cats(ds: Dataset, name: str) -> tuple[list, list[str], np.ndarray]:
    s = ds.analysis_series(name)
    codes = [c for c in pd.unique(s) if not pd.isna(c)]
    try:
        codes = sorted(codes)
    except TypeError:
        codes = sorted(codes, key=str)
    labels = [value_text(ds.var(name), c) for c in codes]
    return codes, labels, s.to_numpy(dtype=object)


def _title(ds: Dataset, name: str) -> str:
    return ds.var(name).title


def _figure(w: float = 6.4, h: float = 4.6) -> Figure:
    fig = Figure(figsize=(w, h), dpi=110)
    FigureCanvasAgg(fig)
    return fig


def _png(fig: Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    return buf.getvalue()


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8)


def _bins(x: np.ndarray, requested: int = 0) -> np.ndarray:
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.array([0.0, 1.0])
    if requested and requested > 0:
        return np.linspace(x.min(), x.max(), int(requested) + 1)
    return np.histogram_bin_edges(x, bins="sturges")


def _mean_err(x: np.ndarray, kind: str, level: float) -> tuple[float, float]:
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return float("nan"), float("nan")
    m = float(x.mean())
    if n < 2:
        return m, 0.0
    sd = float(x.std(ddof=1))
    if kind == "sd":
        return m, sd
    if kind == "se":
        return m, 2 * sd / math.sqrt(n)
    return m, float(stats.t.ppf(0.5 + level / 200, n - 1) * sd / math.sqrt(n))


# ------------------------------------------------------------------ charts
def _splom(ds, r, o):
    names = r["vars"]
    if not 2 <= len(names) <= 8:
        raise ChartError("Choose 2 to 8 variables.")
    k = len(names)
    fig = _figure(1.7 * k + 1, 1.7 * k + 0.6)
    data = {n: _num(ds, n) for n in names}
    groups = _cats(ds, r["group"]) if r.get("group") else None
    for i, yi in enumerate(names):
        for j, xj in enumerate(names):
            ax = fig.add_subplot(k, k, i * k + j + 1)
            ax.tick_params(labelsize=6)
            if i == j:
                x = data[xj][~np.isnan(data[xj])]
                ax.hist(x, bins=_bins(x), color="#9fb7d6", edgecolor="white")
            else:
                if groups:
                    codes, labels, arr = groups
                    for g, (code, lab) in enumerate(zip(codes, labels)):
                        m = arr == code
                        ax.scatter(data[xj][m], data[yi][m], s=8, alpha=0.75,
                                   color=PALETTE[g % 10], label=lab)
                else:
                    ax.scatter(data[xj], data[yi], s=8, alpha=0.7, color=PALETTE[0])
                if o.get("fit"):
                    ok = ~np.isnan(data[xj]) & ~np.isnan(data[yi])
                    if ok.sum() > 2:
                        b, a = np.polyfit(data[xj][ok], data[yi][ok], 1)
                        xs = np.linspace(np.nanmin(data[xj]), np.nanmax(data[xj]), 20)
                        ax.plot(xs, a + b * xs, color="#c23b3b", lw=1)
            if i == k - 1:
                ax.set_xlabel(_title(ds, xj), fontsize=7)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(_title(ds, yi), fontsize=7)
            else:
                ax.set_yticklabels([])
    if groups:
        handles, labels = fig.axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", fontsize=7, title=_title(ds, r["group"]),
                   title_fontsize=7)
    fig.suptitle("Scatterplot Matrix", fontsize=10)
    return fig


def _panel_axes(fig, ds, panel, n_default=1):
    if not panel:
        return [(None, "", fig.add_subplot(1, 1, 1))]
    codes, labels, arr = _cats(ds, panel)
    out = []
    for i, (code, lab) in enumerate(zip(codes, labels)):
        ax = fig.add_subplot(len(codes), 1, i + 1)
        out.append((arr == code, f"{_title(ds, panel)} = {lab}", ax))
    return out


def _panel_scatter(ds, r, o):
    x, y = _num(ds, r["x"]), _num(ds, r["y"])
    n_panels = len(_cats(ds, r["panel"])[0]) if r.get("panel") else 1
    fig = _figure(6.0, 2.4 * n_panels + 0.8)
    groups = _cats(ds, r["group"]) if r.get("group") else None
    for mask, label, ax in _panel_axes(fig, ds, r.get("panel")):
        m = np.ones(len(x), dtype=bool) if mask is None else mask
        if groups:
            for g, (code, lab) in enumerate(zip(*groups[:2])):
                mm = m & (groups[2] == code)
                ax.scatter(x[mm], y[mm], s=12, color=PALETTE[g % 10], label=lab, alpha=0.8)
            ax.legend(fontsize=7, title=_title(ds, r["group"]), title_fontsize=7)
        else:
            ax.scatter(x[m], y[m], s=12, color=PALETTE[0], alpha=0.8)
        if o.get("fit"):
            ok = m & ~np.isnan(x) & ~np.isnan(y)
            if ok.sum() > 2:
                b, a = np.polyfit(x[ok], y[ok], 1)
                xs = np.linspace(np.nanmin(x[ok]), np.nanmax(x[ok]), 20)
                ax.plot(xs, a + b * xs, color="#c23b3b", lw=1.2)
        ax.set_ylabel(_title(ds, r["y"]), fontsize=8)
        if label:
            ax.set_title(label, fontsize=8)
        _style(ax)
    fig.axes[-1].set_xlabel(_title(ds, r["x"]), fontsize=8)
    fig.tight_layout()
    return fig


def _dot(ds, r, o):
    y = _num(ds, r["y"])
    fig = _figure(6.4, 3.4)
    ax = fig.add_subplot()
    ok = ~np.isnan(y)
    edges = _bins(y, 30)
    idx = np.clip(np.digitize(y[ok], edges[1:-1]), 0, len(edges) - 2)
    centers = (edges[:-1] + edges[1:]) / 2
    groups = _cats(ds, r["group"]) if r.get("group") else None
    garr = groups[2][ok] if groups else None
    height = np.zeros(len(centers))
    order = np.argsort(y[ok], kind="stable")
    for k in order:
        b = idx[k]
        height[b] += 1
        color = PALETTE[0]
        if groups:
            color = PALETTE[groups[0].index(garr[k]) % 10] if garr[k] in groups[0] else "#999"
        ax.scatter(centers[b], height[b], s=28, color=color, edgecolor="white", lw=0.4)
    if groups:
        for g, lab in enumerate(groups[1]):
            ax.scatter([], [], color=PALETTE[g % 10], label=lab)
        ax.legend(fontsize=7, title=_title(ds, r["group"]), title_fontsize=7)
    ax.set_yticks([])
    ax.set_xlabel(_title(ds, r["y"]), fontsize=8)
    ax.spines["left"].set_visible(False)
    _style(ax)
    return fig


def _summary_by_category(ds, xname, yname, kind="mean"):
    codes, labels, arr = _cats(ds, xname)
    y = _num(ds, yname)
    values = []
    for c in codes:
        v = y[(arr == c) & ~np.isnan(y)]
        values.append(float(np.median(v)) if kind == "median" else
                      float(v.mean()) if len(v) else np.nan)
    return labels, np.array(values)


def _dropline(ds, r, o):
    fig = _figure()
    ax = fig.add_subplot()
    labels = None
    allv = []
    for i, yname in enumerate(r["y"]):
        labels, means = _summary_by_category(ds, r["x"], yname, o.get("summary", "mean"))
        ax.scatter(range(len(labels)), means, s=40, color=PALETTE[i % 10], zorder=3,
                   label=_title(ds, yname))
        allv.append(means)
    allv = np.array(allv)
    for j in range(allv.shape[1]):
        col = allv[:, j][~np.isnan(allv[:, j])]
        if len(col):
            ax.vlines(j, col.min(), col.max(), color="#555", lw=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel(_title(ds, r["x"]), fontsize=8)
    ax.set_ylabel("Mean", fontsize=8)
    ax.legend(fontsize=7)
    _style(ax)
    return fig


def _summary_point(ds, r, o):
    kind, level = o.get("error", "ci"), float(o.get("level", 95.0))
    codes, labels, arr = _cats(ds, r["x"])
    y = _num(ds, r["y"])
    fig = _figure()
    ax = fig.add_subplot()
    groups = _cats(ds, r["group"]) if r.get("group") else ([None], [""], None)
    width = 0.6 / max(len(groups[0]), 1)
    for g, (gcode, glab) in enumerate(zip(groups[0], groups[1])):
        means, errs = [], []
        for c in codes:
            m = (arr == c)
            if groups[2] is not None:
                m &= groups[2] == gcode
            mean, err = _mean_err(y[m], kind, level)
            means.append(mean)
            errs.append(err)
        xs = np.arange(len(codes)) + (g - (len(groups[0]) - 1) / 2) * width
        ax.errorbar(xs, means, yerr=errs, fmt="o", color=PALETTE[g % 10], capsize=4,
                    label=glab or None)
    ax.set_xticks(range(len(codes)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel(_title(ds, r["x"]), fontsize=8)
    what = {"ci": f"{level:g}% CI", "se": "±2 SE", "sd": "±1 SD"}[kind]
    ax.set_ylabel(f"Mean {_title(ds, r['y'])}", fontsize=8)
    ax.set_title(f"Error bars: {what}", fontsize=8)
    if r.get("group"):
        ax.legend(fontsize=7, title=_title(ds, r["group"]), title_fontsize=7)
    _style(ax)
    return fig


def _violin(ds, r, o):
    y = _num(ds, r["y"])
    fig = _figure()
    ax = fig.add_subplot()
    if r.get("x"):
        codes, labels, arr = _cats(ds, r["x"])
        data = [y[(arr == c) & ~np.isnan(y)] for c in codes]
    else:
        labels, data = [_title(ds, r["y"])], [y[~np.isnan(y)]]
    keep = [i for i, d in enumerate(data) if len(d) > 1]
    if not keep:
        raise ChartError("Each violin needs at least two values.")
    parts = ax.violinplot([data[i] for i in keep], positions=keep, showextrema=False,
                          widths=0.8)
    for k, body in enumerate(parts["bodies"]):
        body.set_facecolor(PALETTE[keep[k] % 10])
        body.set_alpha(0.55)
        body.set_edgecolor("#333")
    if o.get("box", True):
        ax.boxplot([data[i] for i in keep], positions=keep, widths=0.12, patch_artist=True,
                   showfliers=False, boxprops={"facecolor": "white"},
                   medianprops={"color": "#c23b3b"})
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    if r.get("x"):
        ax.set_xlabel(_title(ds, r["x"]), fontsize=8)
    ax.set_ylabel(_title(ds, r["y"]), fontsize=8)
    _style(ax)
    return fig


def _box1d(ds, r, o):
    fig = _figure(6.4, 1.2 + 0.7 * len(r["vars"]))
    ax = fig.add_subplot()
    data = [_num(ds, n) for n in r["vars"]]
    ax.boxplot([d[~np.isnan(d)] for d in data], vert=False, patch_artist=True,
               boxprops={"facecolor": "#9fb7d6"}, medianprops={"color": "#c23b3b"})
    ax.set_yticks(range(1, len(data) + 1))
    ax.set_yticklabels([_title(ds, n) for n in r["vars"]], fontsize=8)
    _style(ax)
    return fig


def _stacked_hist(ds, r, o):
    y = _num(ds, r["y"])
    codes, labels, arr = _cats(ds, r["group"])
    edges = _bins(y, int(o.get("bins", 0)))
    fig = _figure()
    ax = fig.add_subplot()
    ax.hist([y[(arr == c) & ~np.isnan(y)] for c in codes], bins=edges, stacked=True,
            color=[PALETTE[i % 10] for i in range(len(codes))], edgecolor="white", label=labels)
    ax.legend(fontsize=7, title=_title(ds, r["group"]), title_fontsize=7)
    ax.set_xlabel(_title(ds, r["y"]), fontsize=8)
    ax.set_ylabel("Frequency", fontsize=8)
    _style(ax)
    return fig


def _freq_polygon(ds, r, o):
    y = _num(ds, r["y"])
    edges = _bins(y, int(o.get("bins", 0)))
    centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0] if len(edges) > 1 else 1.0
    xs = np.concatenate([[centers[0] - width], centers, [centers[-1] + width]])
    fig = _figure()
    ax = fig.add_subplot()
    groups = _cats(ds, r["group"]) if r.get("group") else ([None], [_title(ds, r["y"])], None)
    for g, (code, lab) in enumerate(zip(groups[0], groups[1])):
        m = ~np.isnan(y) if groups[2] is None else (~np.isnan(y) & (groups[2] == code))
        counts, _ = np.histogram(y[m], bins=edges)
        ax.plot(xs, np.concatenate([[0], counts, [0]]), marker="o", ms=3,
                color=PALETTE[g % 10], label=lab)
    ax.set_xlabel(_title(ds, r["y"]), fontsize=8)
    ax.set_ylabel("Frequency", fontsize=8)
    if r.get("group"):
        ax.legend(fontsize=7, title=_title(ds, r["group"]), title_fontsize=7)
    _style(ax)
    return fig


def pyramid_counts(ds: Dataset, yname: str, split: str, bins: int = 0
                   ) -> tuple[list[str], list[str], np.ndarray]:
    """(row labels bottom-to-top, the two group labels, counts [rows x 2])."""
    codes, labels, arr = _cats(ds, split)
    if len(codes) != 2:
        raise ChartError("The split variable must have exactly two categories.")
    var = ds.var(yname)
    if var.measure == "scale" and var.type == "numeric":
        y = _num(ds, yname)
        edges = _bins(y, bins)
        rows = [f"{edges[i]:.4g}–{edges[i + 1]:.4g}" for i in range(len(edges) - 1)]
        counts = np.array([np.histogram(y[(arr == c) & ~np.isnan(y)], bins=edges)[0]
                           for c in codes]).T
    else:
        ycodes, ylabels, yarr = _cats(ds, yname)
        rows = ylabels
        counts = np.array([[int(np.sum((yarr == yc) & (arr == c))) for c in codes]
                           for yc in ycodes])
    return rows, labels, counts


def _pyramid(ds, r, o):
    rows, labels, counts = pyramid_counts(ds, r["y"], r["split"], int(o.get("bins", 0)))
    values = counts.astype(float)
    if o.get("percent"):
        values = 100 * values / max(values.sum(), 1)
    fig = _figure(6.4, max(3.0, 0.32 * len(rows) + 1.2))
    ax = fig.add_subplot()
    pos = np.arange(len(rows))
    ax.barh(pos, -values[:, 0], color=PALETTE[0], label=labels[0], height=0.85)
    ax.barh(pos, values[:, 1], color=PALETTE[1], label=labels[1], height=0.85)
    ax.set_yticks(pos)
    ax.set_yticklabels(rows, fontsize=7)
    ax.axvline(0, color="#333", lw=0.8)
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{abs(t):g}" for t in ticks], fontsize=7)
    ax.set_xlabel("Percent" if o.get("percent") else "Frequency", fontsize=8)
    ax.set_ylabel(_title(ds, r["y"]), fontsize=8)
    ax.legend(fontsize=7, title=_title(ds, r["split"]), title_fontsize=7, loc="upper right")
    _style(ax)
    return fig


def _panel_hist(ds, r, o):
    y = _num(ds, r["y"])
    edges = _bins(y, int(o.get("bins", 0)))
    n_panels = len(_cats(ds, r["panel"])[0])
    fig = _figure(6.0, 1.9 * n_panels + 0.8)
    for mask, label, ax in _panel_axes(fig, ds, r["panel"]):
        v = y[mask & ~np.isnan(y)]
        ax.hist(v, bins=edges, color="#9fb7d6", edgecolor="white")
        if o.get("normal", True) and len(v) > 2:
            xs = np.linspace(edges[0], edges[-1], 100)
            sd = v.std(ddof=1)
            if sd > 0:
                ax.plot(xs, stats.norm.pdf(xs, v.mean(), sd) * len(v) * (edges[1] - edges[0]),
                        color="#c23b3b", lw=1)
        ax.set_title(f"{label}  (Mean = {v.mean():.2f}, SD = {v.std(ddof=1):.2f}, "
                     f"N = {len(v)})" if len(v) > 1 else label, fontsize=8)
        ax.set_ylabel("Frequency", fontsize=7)
        _style(ax)
    fig.axes[-1].set_xlabel(_title(ds, r["y"]), fontsize=8)
    fig.tight_layout()
    return fig


def _xaxis(ds, name):
    var = ds.var(name)
    if var.type == "numeric" and var.measure == "scale":
        x = _num(ds, name)
        return x, None
    codes, labels, arr = _cats(ds, name)
    pos = np.array([codes.index(v) if v in codes else np.nan for v in arr], dtype=float)
    return pos, labels


def _highlow(ds, r, o):
    x, labels = _xaxis(ds, r["x"])
    hi, lo = _num(ds, r["high"]), _num(ds, r["low"])
    fig = _figure()
    ax = fig.add_subplot()
    ax.vlines(x, lo, hi, color="#1f5fa8", lw=1.6)
    if r.get("close"):
        cl = _num(ds, r["close"])
        span = (np.nanmax(x) - np.nanmin(x)) if np.isfinite(x).any() else 1
        tick = 0.012 * (span or 1)
        ax.hlines(cl, x, x + tick * 3, color="#c23b3b", lw=1.6)
    if labels:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel(_title(ds, r["x"]), fontsize=8)
    ax.set_ylabel("Value", fontsize=8)
    _style(ax)
    return fig


def _rangebar(ds, r, o):
    labels, highs = _summary_by_category(ds, r["x"], r["high"])
    _, lows = _summary_by_category(ds, r["x"], r["low"])
    fig = _figure()
    ax = fig.add_subplot()
    ax.bar(range(len(labels)), highs - lows, bottom=lows, color=PALETTE[0], width=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel(_title(ds, r["x"]), fontsize=8)
    ax.set_ylabel(f"Mean {_title(ds, r['low'])} – {_title(ds, r['high'])}", fontsize=8)
    _style(ax)
    return fig


def _difference(ds, r, o):
    x = _num(ds, r["x"])
    a, b = _num(ds, r["y1"]), _num(ds, r["y2"])
    order = np.argsort(x)
    x, a, b = x[order], a[order], b[order]
    fig = _figure()
    ax = fig.add_subplot()
    ax.plot(x, a, color=PALETTE[0], label=_title(ds, r["y1"]))
    ax.plot(x, b, color=PALETTE[1], label=_title(ds, r["y2"]))
    ax.fill_between(x, a, b, where=a >= b, interpolate=True, color=PALETTE[0], alpha=0.3)
    ax.fill_between(x, a, b, where=a < b, interpolate=True, color=PALETTE[1], alpha=0.3)
    ax.set_xlabel(_title(ds, r["x"]), fontsize=8)
    ax.legend(fontsize=7)
    _style(ax)
    return fig


def _dualaxis(ds, r, o):
    labels, m1 = _summary_by_category(ds, r["x"], r["y1"], o.get("summary", "mean"))
    _, m2 = _summary_by_category(ds, r["x"], r["y2"], o.get("summary", "mean"))
    fig = _figure()
    ax = fig.add_subplot()
    pos = np.arange(len(labels))
    ax.bar(pos, m1, color=PALETTE[0], width=0.6, label=_title(ds, r["y1"]))
    ax.set_ylabel(f"Mean {_title(ds, r['y1'])}", fontsize=8, color=PALETTE[0])
    ax2 = ax.twinx()
    ax2.plot(pos, m2, color=PALETTE[1], marker="o", label=_title(ds, r["y2"]))
    ax2.set_ylabel(f"Mean {_title(ds, r['y2'])}", fontsize=8, color=PALETTE[1])
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel(_title(ds, r["x"]), fontsize=8)
    ax.tick_params(labelsize=8)
    ax2.tick_params(labelsize=8)
    return fig


def _regplots(ds, r, o):
    names = r["vars"]
    k = len(names)
    cols = min(k, 3)
    rows = math.ceil(k / cols)
    fig = _figure(3.0 * cols + 0.6, 2.6 * rows + 0.6)
    y = _num(ds, r["y"])
    for i, name in enumerate(names):
        ax = fig.add_subplot(rows, cols, i + 1)
        x = _num(ds, name)
        ax.scatter(x, y, s=10, color=PALETTE[0], alpha=0.7)
        ok = ~np.isnan(x) & ~np.isnan(y)
        if o.get("fit", True) and ok.sum() > 2:
            res = stats.linregress(x[ok], y[ok])
            xs = np.linspace(x[ok].min(), x[ok].max(), 20)
            ax.plot(xs, res.intercept + res.slope * xs, color="#c23b3b", lw=1.2)
            ax.set_title(f"R² linear = {res.rvalue ** 2:.3f}", fontsize=8)
        ax.set_xlabel(_title(ds, name), fontsize=8)
        ax.set_ylabel(_title(ds, r["y"]), fontsize=8)
        _style(ax)
    fig.tight_layout()
    return fig


_DRAW = {"splom": _splom, "panel_scatter": _panel_scatter, "dot": _dot, "dropline": _dropline,
         "summary_point": _summary_point, "violin": _violin, "box1d": _box1d,
         "stacked_hist": _stacked_hist, "freq_polygon": _freq_polygon, "pyramid": _pyramid,
         "panel_hist": _panel_hist, "highlow": _highlow, "rangebar": _rangebar,
         "difference": _difference, "dualaxis": _dualaxis, "regplots": _regplots}


def check_roles(chart: ChartType, roles: dict) -> str | None:
    for slot in chart.slots:
        value = roles.get(slot.key)
        if slot.required and not value:
            return f"{chart.label}: '{slot.label.rstrip(':')}' is required."
    return None


def render(ds: Dataset, key: str, roles: dict, options: dict | None = None
           ) -> tuple[Figure, str]:
    """(figure, GGRAPH-like syntax line) for chart ``key``."""
    chart = CHARTS[key]
    error = check_roles(chart, roles)
    if error:
        raise ChartError(error)
    opts = {**chart.options, **(options or {})}
    fig = _DRAW[key](ds, roles, opts)
    parts = []
    for slot in chart.slots:
        v = roles.get(slot.key)
        if v:
            parts.append(f"{slot.key.upper()}={' '.join(v) if isinstance(v, list) else v}")
    syntax = f"GGRAPH /GRAPHDATASET VARIABLES=... /CHART TYPE={key.upper()} " + " ".join(parts)
    return fig, syntax + "."


def chart_section(ds: Dataset, key: str, roles: dict, options: dict | None = None):
    from rsstatchem.core.output import Section

    fig, syntax = render(ds, key, roles, options)
    png = _png(fig)
    section = Section(f"GGraph: {CHARTS[key].label}")
    section.log(syntax)
    width = int(fig.get_figwidth() * 88)
    section.image(CHARTS[key].label, png, min(width, 760))
    return section

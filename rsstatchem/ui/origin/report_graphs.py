"""Live graphs embedded in report sheets (no Qt).

Origin's report sheets embed real graph objects, not pictures: the Fitted Curves Plot, the
residual plots, histograms, box charts, Q-Q plots and means plots are graphs that are redrawn
when the report is recalculated and open as a graph window on double-click. Here every report
graph is a :class:`~rsstatchem.core.graph.model.Graph` (the Graph Studio model) built from
the analysis results and stored in the report item (``{"kind": "graph", "graph": <json>}``,
see :meth:`ReportSheet.add_graph`); the report view renders it with the graph renderer, so a
recalculation (which rebuilds the items from the new data) redraws it, and the graph window
opened from it stays linked to the report (``graph.meta["report_link"]``).

Snapshot column keys are stable (``"<series> X"`` / ``"<series>"``) so a linked graph window
can take the new data of a recalculated report without losing the user's formatting
(:func:`update_linked`).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

from rsstatchem.core.graph.model import Column, Graph, Layer, Plot
from rsstatchem.core.graph.themes import PALETTES, apply_theme
from rsstatchem.ui.origin.plotting import ORIGIN_THEME

FIT_RED = "#FF0000"
BAND = "#e8a0a0"
PRED = "#8fb3e8"
DATA_BLACK = "#000000"
FILLS = ("#e57373", "#9ec2f0", "#a5d6a7", "#ffcc80", "#ce93d8", "#80cbc4", "#fff59d")


def new_graph(title: str, width_cm: float = 12.0, height_cm: float = 8.5) -> Graph:
    """An empty one-layer graph with Origin's default 2D look."""
    g = Graph(title=title)
    g.layers.append(Layer())
    apply_theme(g, ORIGIN_THEME, resize_page=False)
    g.page.width_cm, g.page.height_cm = width_cm, height_cm
    layer = g.layers[0]
    layer.legend.visible = False
    layer.legend.loc = "best"
    return g


def _key(g: Graph, name: str, values, label: str = "", value_labels=None,
         kind: str | None = None) -> str:
    key = name
    n = 1
    while key in g.data:
        n += 1
        key = f"{name} ({n})"
    g.data[key] = Column.from_array(key, values, label, value_labels, kind=kind)
    return key


def add_series(g: Graph, x, y, name: str, kind: str = "scatter", color: str | None = None,
               yerr=None, symbol: str = "s", size: float = 6.0, dash: str = "-",
               width: float = 1.5, legend: bool = True, x_labels: list[str] | None = None,
               filled: bool = True, layer: int = 0) -> Plot:
    """Append an XY plot of arrays. ``x_labels`` makes X categorical (codes 1..k)."""
    lay = g.layers[layer]
    y = np.asarray(y, float)
    if x_labels is not None:
        x = np.arange(1, len(x_labels) + 1, dtype=float)
        xk = _key(g, f"{name} X", x, "", {float(i + 1): str(t) for i, t in
                                          enumerate(x_labels)}, kind="num")
    else:
        xk = _key(g, f"{name} X", np.asarray(x, float), kind="num")
    yk = _key(g, name, y, name, kind="num")
    p = Plot(type=kind, name=name, x=xk, y=yk)
    p.line.visible = kind in ("line", "line_symbol")
    p.symbol.visible = kind in ("scatter", "line_symbol")
    p.symbol.shape = symbol
    p.symbol.size = size
    p.symbol.filled = filled
    p.line.style = dash
    p.line.width = width
    if color:
        p.line.color = p.symbol.face = p.symbol.edge = p.error.color = color
    if yerr is not None:
        p.yerr = _key(g, f"{name} Error", np.asarray(yerr, float), kind="num")
    p.options["sort_x"] = False
    if not legend:
        lay.legend.hidden.append(p.id)
    lay.plots.append(p)
    return p


def finish(g: Graph, xtitle: str, ytitle: str, legend: bool = False,
           layer: int = 0) -> Graph:
    lay = g.layers[layer]
    lay.x.title, lay.y.title = xtitle, ytitle
    lay.legend.visible = legend
    return g


# ====================================================================== statistics graphs
def histogram_graph(values, title: str, xtitle: str, normal_curve: bool = False,
                    color: str = FILLS[0], ytitle: str = "Counts") -> Graph:
    g = new_graph(title)
    v = np.asarray(values, float)
    p = Plot(type="histogram", name=title, y=_key(g, title, v[np.isfinite(v)], title,
                                                     kind="num"))
    p.fill.color = color
    p.fill.edge_color = DATA_BLACK
    p.fill.edge_width = 0.8
    p.line.color = FIT_RED
    p.options["normal_curve"] = normal_curve
    g.layers[0].plots.append(p)
    return finish(g, xtitle, ytitle)


def box_graph(groups: list[tuple[str, np.ndarray]], title: str = "Box Chart",
              ytitle: str = "Value") -> Graph:
    g = new_graph(title)
    for k, (name, v) in enumerate(groups):
        v = np.asarray(v, float)
        p = Plot(type="box", name=name, y=_key(g, name, v[np.isfinite(v)], name, kind="num"))
        p.fill.color = FILLS[(k + 1) % len(FILLS)]
        p.fill.edge_color = DATA_BLACK
        p.options.update({"position": float(k + 1), "show_mean": True})
        g.layers[0].plots.append(p)
    return finish(g, "", ytitle)


def qq_graph(values, title: str, ytitle: str) -> Graph:
    """Normal Q-Q plot: ordered data against normal quantiles with the reference line."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    (osm, osr), (slope, icpt, _) = stats.probplot(v, dist="norm")
    g = new_graph(title)
    add_series(g, osm, osr, ytitle or "Data", "scatter", "#1f5fbf", size=5.0)
    xs = np.array([float(np.min(osm)), float(np.max(osm))])
    add_series(g, xs, slope * xs + icpt, "Reference Line", "line", FIT_RED, width=1.2)
    return finish(g, "Theoretical Quantiles (Normal)", ytitle)


def scatter_graph(x, y, title: str, xtitle: str, ytitle: str) -> Graph:
    g = new_graph(title)
    add_series(g, x, y, ytitle or "Y", "scatter", "#1f5fbf", size=5.0)
    return finish(g, xtitle, ytitle)


def means_graph(groups: list[tuple[str, np.ndarray]], title: str = "Means Plot",
                ylabel: str = "Mean") -> Graph:
    """Origin's Means Plot (SE as Error): the level means with ± SE error bars."""
    names = [n for n, _ in groups]
    means = [float(np.mean(v)) for _, v in groups]
    se = [float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0
          for _, v in groups]
    g = new_graph(title, 11.0, 8.0)
    add_series(g, None, means, "Mean", "line_symbol", DATA_BLACK, yerr=se, x_labels=names,
               size=7.0, width=1.2)
    g.layers[0].plots[0].symbol.face = FIT_RED
    return finish(g, "", f"{ylabel} (± SE)")


def interaction_graph(levels_a: list[str], levels_b: list[str], means: np.ndarray,
                      a_title: str, b_title: str, ytitle: str) -> Graph:
    """Two-way Interaction Plot: one line per level of factor B across the levels of A
    (``means[i, j]`` = mean of cell (A_i, B_j))."""
    g = new_graph("Interaction Plot", 11.0, 8.0)
    symbols = ("s", "o", "^", "v", "D", "<", ">")
    pal = PALETTES.get("Origin Classic")
    for j, b in enumerate(levels_b):
        add_series(g, None, means[:, j], f"{b_title} = {b}", "line_symbol",
                   pal[j % len(pal)], x_labels=levels_a, symbol=symbols[j % len(symbols)],
                   size=6.0, width=1.2)
    return finish(g, a_title, f"Mean of {ytitle}", legend=True)


# ====================================================================== fit graphs
def fit_graph(res, xtitle: str, ytitle: str, level: float = 0.95, conf: bool = True,
              pred: bool = False, yerr=None, title: str = "",
              components: list[tuple[str, np.ndarray]] | None = None,
              grid: np.ndarray | None = None) -> Graph:
    """Origin's Fitted Curves Plot: the data, the fit curve and the confidence / prediction
    bands (and peak components of a multiple peak fit)."""
    g = new_graph(title or f"{res.model.name} Fit of {ytitle}", 12.5, 8.5)
    lo, hi = float(np.min(res.x)), float(np.max(res.x))
    log_x = lo > 0 and hi / lo >= 1e3
    xx = grid if grid is not None else (np.geomspace(lo, hi, 300) if log_x else
                                        np.linspace(lo, hi, 300))
    add_series(g, res.x, res.y, ytitle, "scatter", DATA_BLACK,
               yerr=None if yerr is None else np.abs(np.asarray(yerr, float)), size=6.0)
    if pred and res.dof > 0:
        a, b = res.prediction_band(xx, level)
        add_series(g, xx, a, f"Lower {level * 100:g}% Prediction Limit", "line", PRED,
                   dash="--", width=1.0, legend=False)
        add_series(g, xx, b, f"{level * 100:g}% Prediction Band", "line", PRED, dash="--",
                   width=1.0)
    if conf and res.dof > 0:
        a, b = res.confidence_band(xx, level)
        add_series(g, xx, a, f"Lower {level * 100:g}% Confidence Limit", "line", BAND,
                   width=1.0, legend=False)
        add_series(g, xx, b, f"{level * 100:g}% Confidence Band", "line", BAND, width=1.0)
    for i, (label, yc) in enumerate(components or []):
        add_series(g, xx, yc, label, "line", FILLS[i % len(FILLS)], dash="--", width=1.0)
    add_series(g, xx, res.predict(xx), "Fit Curve", "line", FIT_RED, width=1.8)
    lay = g.layers[0]
    if log_x:
        lay.x.scale = "log10"
    return finish(g, xtitle, ytitle, legend=True)


def residual_graph(res, xtitle: str, ytitle: str) -> Graph:
    """Residual vs. Independent plot with the zero line."""
    g = new_graph("Residual vs. Independent", 12.5, 6.0)
    r = np.asarray(res.residuals, float)
    lo, hi = float(np.min(res.x)), float(np.max(res.x))
    add_series(g, [lo, hi], [0.0, 0.0], "Zero", "line", FIT_RED, width=1.0, legend=False)
    add_series(g, res.x, r, "Residual", "scatter", DATA_BLACK, size=5.0)
    if lo > 0 and hi / lo >= 1e3:
        g.layers[0].x.scale = "log10"
    return finish(g, xtitle, f"Residual of {ytitle}")


def residual_histogram_graph(res, yname: str) -> Graph:
    r = np.asarray(res.residuals, float)
    g = histogram_graph(r, "Histogram of the Residual", f"Residual of {yname}",
                        normal_curve=True, color="#7fa7e0", ytitle="Count")
    return g


# ====================================================================== linked windows
def update_linked(target: Graph, source: Graph) -> None:
    """Give a graph window opened from a report the data of the recalculated report graph,
    keeping its formatting when the plots still match (else take the new graph)."""
    same = len(target.layers) == len(source.layers) and all(
        [p.y for p in a.plots] == [p.y for p in b.plots]
        for a, b in zip(target.layers, source.layers))
    if same:
        target.data = {k: v for k, v in source.data.items()}
        for a, b in zip(target.layers, source.layers):
            for pa, pb in zip(a.plots, b.plots):
                pa.x, pa.yerr = pb.x, pb.yerr
        return
    link = target.meta.get("report_link")
    target.layers = source.copy().layers
    target.data = dict(source.data)
    if link:
        target.meta["report_link"] = link

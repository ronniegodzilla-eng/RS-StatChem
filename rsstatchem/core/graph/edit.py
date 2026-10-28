"""Structural edits on a graph: add/remove layers, insets, merge plots (no Qt)."""
from __future__ import annotations

import copy

from rsstatchem.core.graph.model import Annotation, Graph, Layer, new_id
from rsstatchem.core.graph.themes import THEMES, apply_theme_to_layer

COMPATIBLE = {"cartesian": {"line", "scatter", "line_symbol", "column", "bar", "area",
                            "histogram", "box", "bubble", "color_scatter", "contour",
                            "heatmap", "violin", "qq", "probability", "kde", "kde2d", "ecdf",
                            "pareto", "meanbar", "bland_altman", "control",
                            "floating_column", "lollipop", "vector"},
              "polar": {"line", "scatter", "line_symbol", "area", "radar", "windrose"},
              "3d": {"surface3d", "scatter3d", "bar3d", "line3d", "waterfall3d"},
              "ternary": {"line", "scatter", "line_symbol"},
              "pie": {"pie", "doughnut"}}


def _styled_layer(graph: Graph, like: Layer | None, kind: str = "cartesian") -> Layer:
    layer = Layer(kind=kind)
    if like is not None:
        for which in ("x", "y", "y2", "z", "x2"):
            ax = copy.deepcopy(like.axis(which))
            ax.title, ax.min, ax.max, ax.major_step = "", None, None, None
            ax.breaks, ax.custom_ticks = [], []
            setattr(layer, which, ax)
        layer.box = like.box
        layer.legend = copy.deepcopy(like.legend)
        layer.legend.hidden = []
        layer.title_font = copy.deepcopy(like.title_font)
    else:
        apply_theme_to_layer(layer, THEMES.get(graph.theme, THEMES["Origin Default"]))
    return layer


def add_panel(graph: Graph, where: str = "below", like: Layer | None = None) -> Layer:
    """New layer in a new grid row ("below") or column ("right"); the page grows to fit."""
    like = like or (graph.layers[0] if graph.layers else None)
    gridded = [l for l in graph.layers if not l.rect]
    rows = max([graph.rows] + [l.grid[0] + l.grid[2] for l in gridded])
    cols = max([graph.cols] + [l.grid[1] + l.grid[3] for l in gridded])
    layer = _styled_layer(graph, like)
    if where == "right":
        layer.grid = [0, cols, rows, 1]
        graph.cols = cols + 1
        graph.page.width_cm = round(graph.page.width_cm * (cols + 1) / cols, 2)
    else:
        layer.grid = [rows, 0, 1, cols]
        graph.rows = rows + 1
        graph.page.height_cm = round(min(40.0, graph.page.height_cm * (rows + 1) / rows), 2)
    graph.layers.append(layer)
    _relabel(graph)
    return layer


def add_inset(graph: Graph, host: Layer | None = None) -> Layer:
    layer = _styled_layer(graph, host or (graph.layers[0] if graph.layers else None))
    for which in ("x", "y"):
        ax = layer.axis(which)
        ax.label_font.size = max(5.0, ax.label_font.size - 2)
        ax.title_font.size = max(5.0, ax.title_font.size - 2)
    layer.rect = [0.22, 0.55, 0.3, 0.3]
    layer.legend.visible = False
    graph.layers.append(layer)
    return layer


def remove_layer(graph: Graph, index: int) -> None:
    if len(graph.layers) <= 1:
        raise ValueError("A graph needs at least one layer.")
    graph.layers.pop(index)
    gridded = [l for l in graph.layers if not l.rect]
    if gridded:
        graph.rows = max(l.grid[0] + l.grid[2] for l in gridded)
        graph.cols = max(l.grid[1] + l.grid[3] for l in gridded)
        # compact empty rows/columns
        used_rows = sorted({r for l in gridded for r in range(l.grid[0], l.grid[0] + l.grid[2])})
        used_cols = sorted({c for l in gridded for c in range(l.grid[1], l.grid[1] + l.grid[3])})
        for l in gridded:
            l.grid[0] = used_rows.index(l.grid[0])
            l.grid[1] = used_cols.index(l.grid[1])
        graph.rows, graph.cols = max(1, len(used_rows)), max(1, len(used_cols))
    graph.prune_data()
    _relabel(graph)


def _relabel(graph: Graph) -> None:
    panels = [l for l in graph.layers if not l.rect]
    labelled = len(panels) > 1
    for k, layer in enumerate(panels):
        if labelled and (not layer.panel_label or layer.panel_label.startswith("(")):
            layer.panel_label = f"({chr(97 + k)})"
        elif not labelled and layer.panel_label.startswith("("):
            layer.panel_label = ""


def merge_plots(target: Graph, layer: Layer, source: Graph) -> list[str]:
    """Append the plots of ``source`` (first layer) to ``layer``; returns new plot ids."""
    if not source.layers:
        return []
    src = source.layers[0]
    allowed = COMPATIBLE.get(layer.kind, set())
    bad = [p.type for p in src.plots if p.type not in allowed]
    if bad:
        raise ValueError(f"{bad[0]} plots cannot be added to a {layer.kind} layer.")
    remap = {}
    for key, col in source.data.items():
        remap[key] = target.add_column(copy.deepcopy(col), key)
    ids = []
    for p in src.plots:
        p = copy.deepcopy(p)
        p.id = new_id("p")
        for attr in ("x", "y", "z", "xerr", "yerr", "labels"):
            value = getattr(p, attr)
            if value:
                setattr(p, attr, remap.get(value, value))
        for k, v in list(p.options.items()):
            if k.endswith("_col") and isinstance(v, str):
                p.options[k] = remap.get(v, v)
        p.line.color = p.fill.color = "auto"
        layer.plots.append(p)
        ids.append(p.id)
    if len(layer.plots) > 1 and layer.kind != "pie":
        layer.legend.visible = True
    if not layer.y.title and src.y.title:
        layer.y.title = src.y.title
    if not layer.x.title and src.x.title:
        layer.x.title = src.x.title
    return ids


def new_annotation(kind: str, x: float, y: float, x2: float | None = None,
                   y2: float | None = None, coords: str = "axes",
                   font_size: float = 10.0) -> Annotation:
    a = Annotation(kind=kind, x=x, y=y, x2=x if x2 is None else x2, y2=y if y2 is None else y2,
                   coords=coords)
    a.font.size = font_size
    if kind == "rect":
        a.fill.visible = False
    return a

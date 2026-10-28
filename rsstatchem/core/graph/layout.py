"""Layer management and page layout (Origin's Layer Management, Merge Graph Windows,
Extract Layers, Exchange X-Y Axes, Copy/Paste Format, panel presets and the Layout page).
No Qt here."""
from __future__ import annotations

import copy
import math

import numpy as np

from rsstatchem.core.graph.model import Column, Graph, Layer, new_id

LINEISH = ("line", "scatter", "line_symbol", "lollipop")
PANEL_PRESETS = {"v2": (2, 1), "h2": (1, 2), "4": (2, 2), "6": (2, 3), "9": (3, 3)}


# ====================================================================== arranging layers
def gridded(graph: Graph) -> list[Layer]:
    return [layer for layer in graph.layers if not layer.rect]


def arrange(graph: Graph, rows: int, cols: int, hspace: float | None = None,
            wspace: float | None = None, resize_page: bool = True) -> None:
    """Put the (non-inset) layers on a rows × cols grid, row by row (Layer Management >
    Arrange). Spacing is a fraction of the panel size (matplotlib's wspace/hspace)."""
    rows, cols = max(1, int(rows)), max(1, int(cols))
    panels = gridded(graph)
    if len(panels) > rows * cols:
        raise ValueError(f"{len(panels)} layers do not fit on a {rows} × {cols} grid.")
    old_rows, old_cols = graph.rows, graph.cols
    for k, layer in enumerate(panels):
        layer.grid = [k // cols, k % cols, 1, 1]
    graph.rows, graph.cols = rows, cols
    if hspace is not None:
        graph.meta["hspace"] = float(hspace)
    if wspace is not None:
        graph.meta["wspace"] = float(wspace)
    if resize_page and (old_rows, old_cols) != (rows, cols):
        cell_w = graph.page.width_cm / max(1, old_cols)
        cell_h = graph.page.height_cm / max(1, old_rows)
        graph.page.width_cm = round(min(60.0, max(8.0, cell_w * cols)), 2)
        graph.page.height_cm = round(min(60.0, max(6.0, cell_h * rows)), 2)
    relabel(graph)


def relabel(graph: Graph) -> None:
    panels = gridded(graph)
    many = len(panels) > 1
    for k, layer in enumerate(panels):
        if many and (not layer.panel_label or layer.panel_label.startswith("(")):
            layer.panel_label = f"({chr(97 + k % 26)})"
        elif not many and layer.panel_label.startswith("("):
            layer.panel_label = ""


def styled_empty_layer(like: Layer) -> Layer:
    layer = Layer(kind=like.kind)
    for which in ("x", "y", "y2", "z", "x2"):
        ax = copy.deepcopy(like.axis(which))
        ax.title, ax.min, ax.max, ax.major_step = "", None, None, None
        ax.breaks, ax.custom_ticks = [], []
        setattr(layer, which, ax)
    layer.box = like.box
    layer.legend = copy.deepcopy(like.legend)
    layer.legend.hidden = []
    layer.title_font = copy.deepcopy(like.title_font)
    return layer


def split_to_panels(graph: Graph, rows: int, cols: int) -> None:
    """Origin's 2/4/9-panel templates: plot k of the first layer goes to panel k."""
    if not graph.layers:
        return
    src = graph.layers[0]
    plots = list(src.plots)
    n = rows * cols
    layers = []
    for k in range(n):
        layer = copy.deepcopy(src) if k == 0 else styled_empty_layer(src)
        layer.id = src.id if k == 0 else new_id("L")
        layer.plots = [p for j, p in enumerate(plots) if j % n == k]
        layer.legend.hidden = []
        layer.legend.visible = len(layer.plots) > 1
        layer.x.title = src.x.title
        layer.y.title = (layer.plots[0].name if len(layer.plots) == 1 else src.y.title)
        layer.refs = []
        layers.append(layer)
    graph.layers = layers + [l for l in graph.layers[1:] if l.rect]
    arrange(graph, rows, cols)


def add_layers(graph: Graph, n: int, like: Layer | None = None) -> list[Layer]:
    like = like or (graph.layers[0] if graph.layers else Layer())
    out = []
    for _ in range(max(0, int(n))):
        layer = styled_empty_layer(like)
        graph.layers.append(layer)
        out.append(layer)
    k = len(gridded(graph))
    cols = math.ceil(math.sqrt(k))
    arrange(graph, math.ceil(k / cols), cols)
    return out


def move_layer(graph: Graph, index: int, step: int) -> int:
    """Swap a layer with its neighbour (order = drawing order and grid order)."""
    j = index + step
    if not (0 <= index < len(graph.layers) and 0 <= j < len(graph.layers)):
        return index
    a, b = graph.layers[index], graph.layers[j]
    a.grid, b.grid = b.grid, a.grid
    a.rect, b.rect = b.rect, a.rect
    graph.layers[index], graph.layers[j] = b, a
    relabel(graph)
    return j


def link_layer(graph: Graph, layer_index: int, master: int | None, x: bool = True,
               y: bool = False) -> None:
    """Link a layer's X and/or Y range to a master layer (None unlinks)."""
    layer = graph.layers[layer_index]
    for which, on in (("x", x), ("y", y)):
        key = f"link_{which}"
        if master is None or not on:
            layer.options.pop(key, None)
        else:
            if master == layer_index or not 0 <= master < len(graph.layers):
                raise ValueError("Choose another layer to link to.")
            layer.options[key] = int(master)


# ====================================================================== merge / extract
def _copy_layer_into(target: Graph, source: Graph, layer: Layer) -> Layer:
    new = copy.deepcopy(layer)
    new.id = new_id("L")
    remap = {}
    for p in new.plots:
        p.id = new_id("p")
        keys = [getattr(p, a) for a in ("x", "y", "z", "xerr", "yerr", "labels")]
        keys += [v for k, v in p.options.items() if k.endswith("_col") and isinstance(v, str)]
        for key in keys:
            if key and key not in remap and key in source.data:
                remap[key] = target.add_column(copy.deepcopy(source.data[key]), key)
        for attr in ("x", "y", "z", "xerr", "yerr", "labels"):
            v = getattr(p, attr)
            if v:
                setattr(p, attr, remap.get(v, v))
        for k, v in list(p.options.items()):
            if k.endswith("_col") and isinstance(v, str):
                p.options[k] = remap.get(v, v)
    hidden = {old.id: newp.id for old, newp in zip(layer.plots, new.plots)}
    new.legend.hidden = [hidden.get(h, h) for h in new.legend.hidden]
    links = source.meta.get("links", {})
    tl = target.meta.setdefault("links", {})
    for old, newk in remap.items():
        if old in links:
            tl[newk] = copy.deepcopy(links[old])
    for k in ("link_x", "link_y"):
        new.options.pop(k, None)
    return new


def merge_graphs(graphs: list[Graph], rows: int | None = None, cols: int | None = None,
                 title: str = "Merged") -> Graph:
    """Origin's Merge Graph Windows: every layer of every graph on one page, arranged on
    a grid (insets keep their position relative to their own graph's cell)."""
    if not graphs:
        raise ValueError("Choose the graphs to merge.")
    out = Graph(title=title)
    first = graphs[0]
    for attr in ("font_family", "math_font", "palette", "fill_palette", "dash_cycle",
                 "hatch_cycle", "theme"):
        setattr(out, attr, getattr(first, attr))
    panels = []
    for g in graphs:
        for layer in g.layers:
            new = _copy_layer_into(out, g, layer)
            if not layer.rect:
                panels.append(new)
            out.layers.append(new)
    n = max(1, len(panels))
    if cols is None:
        cols = math.ceil(math.sqrt(n))
    if rows is None:
        rows = math.ceil(n / cols)
    out.page.width_cm = round(min(40.0, first.page.width_cm * cols * 0.8), 2)
    out.page.height_cm = round(min(40.0, first.page.height_cm * rows * 0.8), 2)
    out.rows, out.cols = 1, 1
    arrange(out, rows, cols, resize_page=False)
    return out


def extract_layers(graph: Graph) -> list[Graph]:
    """Origin's Extract Layers to Graphs: one new graph per layer."""
    out = []
    for k, layer in enumerate(graph.layers):
        g = Graph(title=f"{graph.title}_L{k + 1}")
        for attr in ("font_family", "math_font", "palette", "fill_palette", "dash_cycle",
                     "hatch_cycle", "theme"):
            setattr(g, attr, getattr(graph, attr))
        g.page = copy.deepcopy(graph.page)
        if graph.rows > 1 or graph.cols > 1:
            g.page.width_cm = round(max(8.0, graph.page.width_cm / graph.cols), 2)
            g.page.height_cm = round(max(6.0, graph.page.height_cm / graph.rows), 2)
        new = _copy_layer_into(g, graph, layer)
        new.grid, new.rect, new.panel_label = [0, 0, 1, 1], None, ""
        g.layers.append(new)
        out.append(g)
    return out


# ====================================================================== exchange X-Y
def exchange_xy(graph: Graph, layer: Layer) -> None:
    """Origin's Exchange X-Y Axes: swap X/Y data of line/scatter plots and the axis
    settings; columns become bars and vice versa."""
    if layer.kind != "cartesian":
        raise ValueError("Exchange X-Y works on 2D (cartesian) layers.")
    for p in layer.plots:
        if p.type in LINEISH or p.type in ("bubble", "color_scatter"):
            if p.x is None:
                y = graph.values(p.y)
                n = 0 if y is None else len(y)
                p.x = graph.add_column(Column.from_array("Row", np.arange(1, n + 1,
                                                                         dtype=float)),
                                       f"Row ({p.id})")
            p.x, p.y = p.y, p.x
            p.xerr, p.yerr = p.yerr, p.xerr
            p.options["sort_x"] = False
        elif p.type == "column":
            p.type = "bar"
        elif p.type == "bar":
            p.type = "column"
        elif p.type == "histogram":
            p.options["orientation"] = "vertical" if p.options.get("orientation") == \
                "horizontal" else "horizontal"
    layer.x, layer.y = layer.y, layer.x
    for ref in layer.refs:
        ref.axis = {"x": "y", "y": "x"}.get(ref.axis, ref.axis)


# ====================================================================== copy / paste format
_PLOT_OPTS = ("connect", "skip", "drop_lines", "label_pos", "label_size", "label_format",
              "error_band", "band_alpha", "alpha", "cmap", "size_min", "size_max")
_AXIS_KEEP = ("title", "min", "max", "major_step", "breaks", "custom_ticks")


def format_of(obj) -> dict:
    """Style of a plot or layer for Copy Format (Origin's Copy Format / Paste Format)."""
    from rsstatchem.core.graph.model import Plot, to_dict
    if isinstance(obj, Plot):
        return {"kind": "plot", "line": to_dict(obj.line), "symbol": to_dict(obj.symbol),
                "fill": to_dict(obj.fill), "error": to_dict(obj.error),
                "options": {k: copy.deepcopy(v) for k, v in obj.options.items()
                            if k in _PLOT_OPTS}}
    if isinstance(obj, Layer):
        return {"kind": "layer", "axes": {w: to_dict(obj.axis(w)) for w in
                                          ("x", "y", "y2", "z", "x2")},
                "box": obj.box, "background": obj.background, "legend": to_dict(obj.legend),
                "title_font": to_dict(obj.title_font), "gap": obj.gap, "cmap": obj.cmap,
                "plots": [format_of(p) for p in obj.plots]}
    raise TypeError("Copy Format works on plots and layers.")


def apply_format(obj, fmt: dict) -> None:
    from rsstatchem.core.graph.model import (Axis, ErrorStyle, FillStyle, Font, Legend,
                                             LineStyle, Plot, SymbolStyle, from_dict)
    if isinstance(obj, Plot):
        if fmt.get("kind") == "layer":
            fmt = (fmt.get("plots") or [None])[0]
            if fmt is None:
                return
        obj.line = from_dict(LineStyle, fmt["line"])
        obj.symbol = from_dict(SymbolStyle, fmt["symbol"])
        obj.fill = from_dict(FillStyle, fmt["fill"])
        obj.error = from_dict(ErrorStyle, fmt["error"])
        obj.options.update(copy.deepcopy(fmt.get("options", {})))
        return
    if isinstance(obj, Layer):
        if fmt.get("kind") == "plot":
            for p in obj.plots:
                apply_format(p, fmt)
            return
        for which, data in fmt.get("axes", {}).items():
            old = obj.axis(which)
            new = from_dict(Axis, data)
            for keep in _AXIS_KEEP:
                setattr(new, keep, copy.deepcopy(getattr(old, keep)))
            setattr(obj, which, new)
        obj.box = fmt.get("box", obj.box)
        obj.background = fmt.get("background", obj.background)
        if "legend" in fmt:
            hidden = obj.legend.hidden
            obj.legend = from_dict(Legend, fmt["legend"])
            obj.legend.hidden = hidden
        if "title_font" in fmt:
            obj.title_font = from_dict(Font, fmt["title_font"])
        obj.gap = fmt.get("gap", obj.gap)
        obj.cmap = fmt.get("cmap", obj.cmap)
        styles = fmt.get("plots") or []
        for k, p in enumerate(obj.plots):
            if styles:
                apply_format(p, styles[k % len(styles)])
        return
    raise TypeError("Paste Format works on plots and layers.")


# ====================================================================== layout page
PAGE_SIZES = {"A4 portrait": (21.0, 29.7), "A4 landscape": (29.7, 21.0),
              "Letter portrait": (21.59, 27.94), "Letter landscape": (27.94, 21.59),
              "Slide 16:9": (25.4, 14.29)}


def layout_page(graphs: list[Graph], cols: int = 2, page: str | tuple = "A4 portrait",
                margin_cm: float = 1.5, gap_cm: float = 1.0, title: str = "Layout1",
                heading: str = "") -> Graph:
    """Origin's Layout page: the graphs placed on one publication page (each graph's
    layers keep their arrangement inside the graph's cell). The page is an ordinary graph,
    so it is edited, saved and exported like one."""
    if not graphs:
        raise ValueError("Choose the graphs to place on the layout page.")
    w, h = PAGE_SIZES.get(page, (21.0, 29.7)) if isinstance(page, str) else page
    out = Graph(title=title)
    out.page.width_cm, out.page.height_cm = float(w), float(h)
    first = graphs[0]
    for attr in ("font_family", "math_font", "palette", "fill_palette", "theme"):
        setattr(out, attr, getattr(first, attr))
    cols = max(1, int(cols))
    rows = math.ceil(len(graphs) / cols)
    top = margin_cm + (1.2 if heading else 0.0)
    cell_w = (w - 2 * margin_cm - gap_cm * (cols - 1)) / cols
    cell_h_avail = (h - top - margin_cm - gap_cm * (rows - 1)) / rows
    for k, g in enumerate(graphs):
        r, c = divmod(k, cols)
        aspect = g.page.height_cm / g.page.width_cm if g.page.width_cm else 0.75
        cw = cell_w
        ch = min(cell_h_avail, cw * aspect)
        cw = ch / aspect if ch < cell_w * aspect else cw
        x0 = margin_cm + c * (cell_w + gap_cm) + (cell_w - cw) / 2
        y0_top = top + r * (cell_h_avail + gap_cm)
        cell = (x0 / w, 1 - (y0_top + ch) / h, cw / w, ch / h)
        g_rows = max([g.rows] + [l.grid[0] + l.grid[2] for l in gridded(g)] or [1])
        g_cols = max([g.cols] + [l.grid[1] + l.grid[3] for l in gridded(g)] or [1])
        for layer in g.layers:
            new = _copy_layer_into(out, g, layer)
            if layer.rect:
                lx, ly, lw, lh = layer.rect
            else:
                rr, cc, rs, cs = layer.grid
                pad_l, pad_b, pad_r, pad_t = 0.16, 0.16, 0.05, 0.08
                lx = cc / g_cols + pad_l / g_cols
                lw = cs / g_cols - (pad_l + pad_r) / g_cols
                ly = 1 - (rr + rs) / g_rows + pad_b / g_rows
                lh = rs / g_rows - (pad_b + pad_t) / g_rows
            new.rect = [cell[0] + lx * cell[2], cell[1] + ly * cell[3], lw * cell[2],
                        lh * cell[3]]
            if not layer.rect and len(graphs) > 1 and not new.panel_label:
                new.panel_label = f"({chr(97 + k % 26)})" if layer is g.layers[0] else ""
            out.layers.append(new)
    if heading:
        from rsstatchem.core.graph.edit import new_annotation
        ann = new_annotation("text", margin_cm / w, 1 - (margin_cm + 0.8) / h,
                             coords="figure", font_size=14.0)
        ann.text = heading
        ann.font.bold = True
        out.layers[0].annotations.append(ann)
    out.meta["layout_page"] = {"cols": cols, "page": list((w, h)),
                               "graphs": [g.title for g in graphs]}
    return out

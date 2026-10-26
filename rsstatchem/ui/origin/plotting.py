"""Plot worksheet columns the way Origin does (no Qt).

Origin plots the *selected* columns using their designations: every selected Y column is
drawn against its associated X column (the nearest X to its left, else row numbers), with the
Y Error column to its right as error bars and a Label column for data labels. The graph is
built with the Graph Studio core (:func:`rsstatchem.core.graph.build.build_graph`) on a
temporary dataset made from the sheet, then given Origin's default look: black frame, ticks
out, Origin's colour/symbol increment, legend from the Long Names and axis titles
"Long Name (Units)".

Each plot remembers its source columns (``plot.options["source"]``) and the graph remembers
which snapshot columns come from which worksheet column (``graph.meta["links"]``), so the
graph is refreshed when the worksheet changes (:func:`refresh_links`) and analyses started
from a graph find their data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rsstatchem.core.dataset import NOMINAL, NUMERIC, SCALE, Dataset, Variable
from rsstatchem.core.graph import edit
from rsstatchem.core.graph.build import TEMPLATES, PlotSpec, build_graph
from rsstatchem.core.graph.model import Column, Graph, Plot
from rsstatchem.core.graph.themes import apply_theme
from rsstatchem.ui.origin.model import Workbook, Worksheet

# Origin's default 2D graph: framed layer, ticks pointing out, no legend border.
ORIGIN_THEME = {
    "page": (12.0, 9.0), "font_family": "Arial", "label_size": 10.0, "title_size": 11.0,
    "legend_size": 9.0, "tick_dir": "out", "box": True, "grid": False, "axis_width": 1.0,
    "line_width": 1.5, "symbol_size": 6.0, "major_len": 5.0, "minor_len": 3.0,
    "palette": "Origin Classic", "fill_palette": "Origin Classic", "dash_cycle": False,
    "hatch_cycle": False, "legend_frame": False, "math_font": "stixsans",
}

# Plot menu / 2D Graphs toolbar entries: key -> (menu text, Graph Studio template key).
PLOT_TYPES: dict[str, tuple[str, str]] = {
    "line": ("Line", "line"), "scatter": ("Scatter", "scatter"),
    "line_symbol": ("Line + Symbol", "line_symbol"), "yerror": ("Y Error", "errorbar"),
    "column": ("Column", "column"), "column_stacked": ("Stacked Column", "column_stacked"),
    "column_percent": ("100% Stacked Column", "column_percent"), "bar": ("Bar", "bar"),
    "bar_stacked": ("Stacked Bar", "bar_stacked"), "area": ("Area", "area"),
    "area_stacked": ("Stacked Area", "area_stacked"), "pie": ("2D Pie", "pie"),
    "doughnut": ("Doughnut", "doughnut"), "histogram": ("Histogram", "histogram"),
    "box": ("Box Chart", "box"), "double_y": ("Double-Y", "double_y"),
    "stack": ("Stack (multi-panel)", "multi_panel"), "waterfall": ("Waterfall", "waterfall"),
    "bubble": ("Bubble", "bubble"), "color_scatter": ("Color Mapped", "color_scatter"),
    "contour": ("Contour - Color Fill", "contour"), "heatmap": ("Heatmap", "heatmap"),
    "scatter3d": ("3D Scatter", "scatter3d"), "surface3d": ("3D Colormap Surface",
                                                            "surface3d"),
    "polar": ("Polar θ(X) r(Y)", "polar"), "ternary": ("Ternary", "ternary"),
}
XYZ = ("contour", "heatmap", "scatter3d", "surface3d", "ternary")
XY_SIZE = ("bubble", "color_scatter")
NO_X = ("histogram", "box")


class PlotError(ValueError):
    pass


# ====================================================================== dataset
def plot_dataset(ws: Worksheet) -> Dataset:
    """The sheet as a Dataset for the graph builder. Text columns become category codes in
    worksheet order (so a column chart keeps the row order, as in Origin)."""
    n = max(ws.data_rows(), 1)
    data, variables = {}, []
    for c in ws.columns:
        if c.is_text():
            order: dict[str, float] = {}
            codes = []
            for t in c.texts(n):
                if t == "":
                    codes.append(np.nan)
                    continue
                if t not in order:
                    order[t] = float(len(order) + 1)
                codes.append(order[t])
            data[c.name] = pd.Series(codes, dtype=float)
            variables.append(Variable(c.name, NUMERIC, label=c.title, measure=NOMINAL,
                                      value_labels={v: k for k, v in order.items()}))
        else:
            data[c.name] = pd.Series(c.numeric(n))
            variables.append(Variable(c.name, NUMERIC, label=c.title, measure=SCALE))
    return Dataset(pd.DataFrame(data, index=pd.RangeIndex(n)), variables, name=ws.name)


# ====================================================================== selection -> roles
def _ys_for(ws: Worksheet, cols: list[int], kind: str) -> list[int]:
    cols = sorted(dict.fromkeys(cols))
    if kind in NO_X:
        return [c for c in cols if ws.columns[c].designation not in ("L", "N")] or cols
    ys = [c for c in cols if ws.columns[c].designation == "Y"]
    if not ys:   # Origin plots non-X columns as Y when no Y is selected
        ys = [c for c in cols if ws.columns[c].designation in ("Z", "N")]
    if not ys and cols:   # only X columns selected: plot the Ys that belong to them
        xs = set(cols)
        ys = [i for i, c in enumerate(ws.columns) if c.designation == "Y" and ws.x_of(i) in xs]
    return ys


def _group_key(ws: Worksheet, y: int, kind: str) -> tuple:
    x = ws.x_of(y)
    yerr = ws.yerr_of(y) if kind not in NO_X else None
    label = ws.label_of(y) if kind in ("pie", "doughnut") else None
    xerr = ws.xerr_of(y) if kind in ("scatter", "line_symbol", "line", "yerror") else None
    return x, yerr, label, xerr


def plot_columns(book: Workbook, ws: Worksheet, cols: list[int], kind: str,
                 title: str = "Graph1") -> Graph:
    """Build a graph of the selected columns ``cols`` of sheet ``ws`` (Origin's Plot menu)."""
    if kind not in PLOT_TYPES:
        raise PlotError(f"Unknown plot type {kind!r}.")
    template = PLOT_TYPES[kind][1]
    if not cols:
        raise PlotError("Select one or more columns (click the column headers) to plot.")
    if ws.data_rows() == 0:
        raise PlotError("The selected columns contain no data.")
    ds = plot_dataset(ws)
    name = ws.columns.__getitem__
    if kind in XYZ:
        specs = [_xyz_spec(ws, cols, template)]
    elif kind in XY_SIZE:
        ys = [c for c in sorted(cols) if ws.columns[c].designation != "X"]
        if len(ys) < 2:
            raise PlotError(f"{PLOT_TYPES[kind][0]} needs a Y column and a second column for "
                            "the bubble size / colour.")
        x = ws.x_of(ys[0])
        specs = [PlotSpec(template, x=name(x).name if x is not None else None,
                          ys=[name(ys[0]).name], z=name(ys[1]).name)]
    else:
        ys = _ys_for(ws, cols, kind)
        if not ys:
            raise PlotError("Select at least one Y column to plot (right-click a column "
                            "header > Set As > Y).")
        if kind in ("pie", "doughnut"):
            ys = ys[:1]
        if kind in ("double_y", "stack", "waterfall") and len(ys) < 2 and kind != "waterfall":
            raise PlotError(f"{PLOT_TYPES[kind][0]} needs at least two Y columns.")
        groups: dict[tuple, list[int]] = {}
        if kind in NO_X or kind in ("double_y", "stack"):
            groups[(None if kind in NO_X else ws.x_of(ys[0]), None, None, None)] = ys
        else:
            for y in ys:
                groups.setdefault(_group_key(ws, y, kind), []).append(y)
        specs = []
        for (x, yerr, label, xerr), members in groups.items():
            spec_kind = template
            if kind == "yerror" and yerr is None:
                spec_kind = "scatter"
            specs.append(PlotSpec(
                spec_kind, x=name(x).name if x is not None and kind not in NO_X else None,
                ys=[name(y).name for y in members],
                yerr=name(yerr).name if yerr is not None else None,
                xerr=name(xerr).name if xerr is not None else None,
                labels=name(label).name if label is not None else None))
    if kind in ("pie", "doughnut") and specs[0].x is None and specs[0].labels:
        specs[0].x, specs[0].labels = specs[0].labels, None
    graph = None
    for spec in specs:
        try:
            part = build_graph(ds, spec, title=title)
        except (ValueError, KeyError) as exc:
            raise PlotError(str(exc)) from None
        if graph is None:
            graph = part
        else:
            edit.merge_plots(graph, graph.layers[0], part)
    graph.title = title
    graph.meta.pop("spec", None)
    graph.meta["origin_kind"] = kind
    _origin_look(graph, book, ws, kind)
    return graph


def _xyz_spec(ws: Worksheet, cols: list[int], template: str) -> PlotSpec:
    cols = sorted(cols)
    z = next((c for c in cols if ws.columns[c].designation == "Z"), None)
    if z is None:
        non_x = [c for c in cols if ws.columns[c].designation != "X"]
        if len(non_x) >= 2:
            y, z = non_x[0], non_x[1]
            x = ws.x_of(y)
        else:
            raise PlotError("XYZ plots need a Z column: set a column as Z (Column > Set As > Z) "
                            "next to its X and Y columns, then select it.")
    else:
        y = ws.y_of(z)
        x = ws.x_of(y if y is not None else z)
        if y is None:
            raise PlotError("No Y column to the left of the Z column.")
    if x is None:
        raise PlotError("No X column for this XYZ plot.")
    names = ws.columns
    return PlotSpec(template, x=names[x].name, ys=[names[y].name], z=names[z].name)


# ====================================================================== Origin look & links
def _source_col(ws: Worksheet, key: str | None, graph: Graph):
    col = graph.column(key)
    if col is None:
        return None
    try:
        return ws.column(col.name)
    except KeyError:
        return None


def column_key(book: str, sheet: str, col: str, text: bool = False) -> str:
    """Snapshot key of a worksheet column in a graph, e.g. ``[Book1]Sheet1!B``."""
    return f"[{book}]{sheet}!{col}" + (" (labels)" if text else "")


def rekey(graph: Graph, mapping: dict[str, str]) -> None:
    """Rename snapshot columns (and every plot reference to them)."""
    if not mapping:
        return
    graph.data = {mapping.get(k, k): v for k, v in graph.data.items()}
    for layer in graph.layers:
        for p in layer.plots:
            for attr in ("x", "y", "z", "xerr", "yerr", "labels"):
                value = getattr(p, attr)
                if value in mapping:
                    setattr(p, attr, mapping[value])
            for k, v in list(p.options.items()):
                if k.endswith("_col") and v in mapping:
                    p.options[k] = mapping[v]
    links = graph.meta.get("links", {})
    graph.meta["links"] = {mapping.get(k, k): v for k, v in links.items()}


def _origin_look(graph: Graph, book: Workbook, ws: Worksheet, kind: str) -> None:
    apply_theme(graph, ORIGIN_THEME, resize_page=graph.rows == 1 and graph.cols == 1)
    if kind in ("box", "histogram", "pie", "doughnut"):   # black solid boxes read badly
        graph.fill_palette = "Bright"
    links = graph.meta.setdefault("links", {})
    mapping = {}
    for key, col in graph.data.items():
        try:
            src = ws.column(col.name)
        except KeyError:
            continue
        if key == src.name:
            mapping[key] = column_key(book.name, ws.name, src.name)
            links[mapping[key]] = {"book": book.name, "sheet": ws.name, "col": src.name,
                                   "role": "num"}
        elif key == f"{src.name} (labels)":
            mapping[key] = column_key(book.name, ws.name, src.name, True)
            links[mapping[key]] = {"book": book.name, "sheet": ws.name, "col": src.name,
                                   "role": "text"}
    rekey(graph, mapping)
    for layer in graph.layers:
        if layer.kind == "cartesian" and kind not in ("pie", "doughnut"):
            layer.legend.visible = True
            layer.legend.loc = "best"
        first_y = None
        for p in layer.plots:
            ycol = _source_col(ws, p.y, graph)
            xcol = _source_col(ws, p.x, graph)
            if ycol is not None and kind not in ("box",):
                p.name = ycol.title
                first_y = first_y or ycol
            p.options["source"] = {
                "book": book.name, "sheet": ws.name,
                "x": xcol.name if xcol is not None else None,
                "y": ycol.name if ycol is not None else None,
                "yerr": (_source_col(ws, p.yerr, graph).name
                         if _source_col(ws, p.yerr, graph) is not None else None)}
            if p.type in ("line", "line_symbol"):
                p.options["sort_x"] = False
        if layer.kind == "cartesian" and kind not in NO_X and kind not in ("bar",
                                                                          "bar_stacked"):
            xcol = next((_source_col(ws, p.x, graph) for p in layer.plots
                         if _source_col(ws, p.x, graph) is not None), None)
            if xcol is not None:
                layer.x.title = xcol.axis_title
            if first_y is not None and kind != "double_y" and "percent" not in kind:
                layer.y.title = first_y.axis_title
        elif kind in ("bar", "bar_stacked") and first_y is not None:
            layer.x.title = first_y.axis_title
        elif kind == "histogram" and first_y is not None:
            layer.x.title = first_y.axis_title
        if layer.kind in ("3d", "ternary") and layer.plots:
            p = layer.plots[0]
            for axis, key in ((layer.x, p.x), (layer.y, p.y), (layer.z, p.z)):
                col = _source_col(ws, key, graph)
                if col is not None:
                    axis.title = col.axis_title if layer.kind == "3d" else col.title
        if kind == "double_y":
            ys = [_source_col(ws, p.y, graph) for p in layer.plots]
            left = [c for c, p in zip(ys, layer.plots) if c is not None and p.y_axis == "left"]
            right = [c for c, p in zip(ys, layer.plots) if c is not None and p.y_axis == "right"]
            if left:
                layer.y.title = left[0].axis_title
            if right:
                layer.y2.title = right[0].axis_title
            for p in layer.plots:   # Origin colours each axis like its plot
                if p.y_axis == "right":
                    p.line.color = p.symbol.face = p.symbol.edge = "#FF0000"
                    layer.y2.color = layer.y2.title_font.color = "#FF0000"


def refresh_links(graph: Graph, books: dict[str, Workbook]) -> bool:
    """Re-read linked snapshot columns from their worksheets; True when anything changed."""
    changed = False
    for key, link in graph.meta.get("links", {}).items():
        book = books.get(link.get("book"))
        ws = book.sheet(link.get("sheet", "")) if book is not None else None
        col = graph.data.get(key)
        if not isinstance(ws, Worksheet) or col is None:
            continue
        try:
            src = ws.column(link["col"])
        except KeyError:
            continue
        n = max(ws.data_rows(), 1)
        if link.get("role") == "text":
            new = Column(col.name, col.label, "text", src.texts(n))
        elif src.is_text():
            continue   # category codes are only rebuilt by re-plotting
        else:
            new = Column.from_array(col.name, src.numeric(n), src.title, col.value_labels,
                                    kind="num")
        if new.values != col.values or new.label != col.label:
            graph.data[key] = new
            changed = True
    return changed


def add_xy_plot(graph: Graph, layer_index: int, x, y, name: str, kind: str = "line",
                color: str | None = None, link: dict | None = None,
                source: dict | None = None) -> Plot:
    """Append a plot of arrays (x, y) — e.g. a fitted curve — to a layer."""
    layer = graph.layers[layer_index]
    xk, yk = f"{name} X", name
    if link:
        xk = column_key(link["book"], link["sheet"], link.get("x") or "X")
        yk = column_key(link["book"], link["sheet"], link.get("y") or "Y")
    xkey = graph.add_column(Column.from_array(link.get("x", "X") if link else f"{name} X",
                                              np.asarray(x, float), kind="num"), xk)
    ykey = graph.add_column(Column.from_array(link.get("y", "Y") if link else name,
                                              np.asarray(y, float), name, kind="num"), yk)
    p = Plot(type=kind, name=name, x=xkey, y=ykey)
    p.line.visible = kind in ("line", "line_symbol")
    p.symbol.visible = kind in ("scatter", "line_symbol")
    p.line.width = 1.5
    if color:
        p.line.color = p.symbol.face = p.symbol.edge = color
    if source:
        p.options["source"] = source
    layer.plots.append(p)
    layer.legend.visible = True
    if link:
        links = graph.meta.setdefault("links", {})
        base = {"book": link["book"], "sheet": link["sheet"]}
        if link.get("x"):
            links[xkey] = {**base, "col": link["x"], "role": "num"}
        if link.get("y"):
            links[ykey] = {**base, "col": link["y"], "role": "num"}
    return p


def plot_data(graph: Graph, plot: Plot) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) arrays of a plot as drawn (row numbers when it has no X)."""
    y = graph.values(plot.y)
    if y is None:
        raise PlotError("The plot has no Y data.")
    x = graph.values(plot.x)
    if x is None:
        x = np.arange(1, len(y) + 1, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def template_label(kind: str) -> str:
    return PLOT_TYPES[kind][0] if kind in PLOT_TYPES else TEMPLATES[kind].label

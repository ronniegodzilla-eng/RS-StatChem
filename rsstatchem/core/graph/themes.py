"""Palettes, increment lists and graph themes / user templates (Origin's "Graph Themes").

A theme is a dict of style settings applied to every layer, axis and plot of a graph: page
size, fonts, tick direction, frame, grid, line widths, symbol size and colour lists. User
templates are saved as JSON in ``<user data>/graph_templates``; they capture the same style
settings plus the per-plot styles of the source graph (so a template carries custom colours
and symbols to the next graph, like an Origin template).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from rsstatchem.core.graph.model import (Axis, Font, Graph, Layer, Plot, from_dict,
                                         to_dict)

PALETTES: dict[str, list[str]] = {
    # Origin's classic colour increment list (Black, Red, Blue, Magenta, Olive, Navy, …).
    "Origin Classic": ["#000000", "#FF0000", "#0000FF", "#FF00FF", "#808000", "#000080",
                       "#8000FF", "#800080", "#800000", "#008080", "#008000", "#808080"],
    "Nature": ["#3C5488", "#E64B35", "#00A087", "#F39B7F", "#4DBBD5", "#8491B4",
               "#91D1C2", "#DC0000", "#7E6148", "#B09C85"],
    "Bright": ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B",
               "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"],
    "Colorblind": ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9",
                   "#F0E442", "#000000"],
    "Grayscale": ["#000000", "#555555", "#888888", "#AAAAAA", "#333333", "#777777",
                  "#CCCCCC"],
    "Gray Fill": ["#FFFFFF", "#BFBFBF", "#7F7F7F", "#3F3F3F", "#DDDDDD", "#9F9F9F",
                  "#5F5F5F", "#000000"],
}
SYMBOLS = ["s", "o", "^", "v", "D", "<", ">", "p", "h", "*"]
SYMBOL_NAMES = {"s": "Square", "o": "Circle", "^": "Up triangle", "v": "Down triangle",
                "D": "Diamond", "<": "Left triangle", ">": "Right triangle", "p": "Pentagon",
                "h": "Hexagon", "*": "Star", "+": "Plus", "x": "Cross", ".": "Dot",
                "auto": "Auto (increment)"}
DASHES = ["-", "--", ":", "-.", (0, (5, 1, 1, 1, 1, 1)), (0, (1, 1))]
DASH_NAMES = {"-": "Solid", "--": "Dash", ":": "Dot", "-.": "Dash dot",
              "auto": "Auto (increment)", "none": "None"}
HATCHES = ["", "///", "\\\\\\", "xxx", "...", "---", "|||", "ooo"]
HATCH_NAMES = {"": "None", "///": "Diagonal /", "\\\\\\": "Diagonal \\", "xxx": "Cross",
               "...": "Dots", "---": "Horizontal", "|||": "Vertical", "ooo": "Circles",
               "auto": "Auto (increment)"}
COLORMAPS = ["viridis", "plasma", "inferno", "magma", "cividis", "jet", "rainbow",
             "coolwarm", "RdBu_r", "RdYlBu_r", "Spectral_r", "terrain", "gray", "Greys",
             "Blues", "Reds", "YlGnBu", "hot"]
FONT_FAMILIES = ["Arial", "Helvetica", "Times New Roman", "Calibri", "Cambria", "Georgia",
                 "Verdana", "Tahoma", "Courier New", "DejaVu Sans", "DejaVu Serif",
                 "Liberation Sans", "Liberation Serif"]

_BASE = {
    "page": (12.0, 9.0), "font_family": "Arial", "label_size": 10.0, "title_size": 11.0,
    "legend_size": 9.0, "tick_dir": "in", "box": True, "grid": False, "axis_width": 1.0,
    "line_width": 1.2, "symbol_size": 6.0, "major_len": 5.0, "minor_len": 3.0,
    "palette": "Origin Classic", "fill_palette": "Nature", "dash_cycle": False,
    "hatch_cycle": False, "legend_frame": True, "math_font": "stixsans",
}
THEMES: dict[str, dict] = {
    "Origin Default": dict(_BASE),
    "Journal - Single Column (8.3 cm)": dict(
        _BASE, page=(8.3, 6.4), label_size=8.0, title_size=9.0, legend_size=7.5,
        axis_width=0.75, line_width=1.0, symbol_size=4.0, major_len=3.5, minor_len=2.0,
        legend_frame=False),
    "Journal - Double Column (17.4 cm)": dict(
        _BASE, page=(17.4, 9.5), label_size=8.5, title_size=9.0, legend_size=8.0,
        axis_width=0.75, line_width=1.0, symbol_size=4.5, major_len=3.5, minor_len=2.0,
        legend_frame=False),
    "Grayscale": dict(_BASE, palette="Grayscale", fill_palette="Gray Fill", dash_cycle=True,
                      hatch_cycle=True),
    "Presentation": dict(_BASE, page=(16.0, 11.0), label_size=14.0, title_size=16.0,
                         legend_size=12.0, axis_width=1.5, line_width=2.0, symbol_size=8.0,
                         major_len=6.0, minor_len=3.5),
    "Open Frame (ticks out)": dict(_BASE, tick_dir="out", box=False, palette="Nature"),
    "Grid": dict(_BASE, grid=True, palette="Bright", fill_palette="Bright"),
}
DEFAULT_THEME = "Origin Default"


def page_for(theme: dict, rows: int = 1, cols: int = 1) -> tuple[float, float]:
    width, height = theme["page"]
    if rows > 1 or cols > 1:
        cell_h = height / max(1.0, 0.9 * cols) if cols > 1 else height * 0.62
        height = min(max(height, cell_h * rows + 1.0), 24.0)
    return width, round(height, 2)


def _style_axis(axis: Axis, t: dict, family: str) -> None:
    axis.title_font.size = t["title_size"]
    axis.label_font.size = t["label_size"]
    axis.title_font.family = axis.label_font.family = ""
    axis.tick_dir = t["tick_dir"]
    axis.line_width = axis.tick_width = t["axis_width"]
    axis.major_len, axis.minor_len = t["major_len"], t["minor_len"]
    axis.grid_major = t["grid"]


def apply_theme(graph: Graph, name_or_theme: str | dict, resize_page: bool = True) -> None:
    t = THEMES.get(name_or_theme, None) if isinstance(name_or_theme, str) else name_or_theme
    if t is None:
        raise KeyError(f"Unknown theme {name_or_theme!r}")
    t = dict(_BASE, **t)
    if isinstance(name_or_theme, str):
        graph.theme = name_or_theme
    if resize_page:
        graph.page.width_cm, graph.page.height_cm = page_for(t, graph.rows, graph.cols)
    graph.font_family = t["font_family"]
    graph.math_font = t["math_font"]
    graph.palette, graph.fill_palette = t["palette"], t["fill_palette"]
    graph.dash_cycle, graph.hatch_cycle = t["dash_cycle"], t["hatch_cycle"]
    for layer in graph.layers:
        apply_theme_to_layer(layer, t)


def apply_theme_to_layer(layer: Layer, t: dict) -> None:
    t = dict(_BASE, **t)
    for which in ("x", "y", "y2", "z", "x2"):
        _style_axis(layer.axis(which), t, t["font_family"])
    layer.box = t["box"]
    layer.title_font.size = t["title_size"]
    layer.legend.font.size = t["legend_size"]
    layer.legend.frame = t["legend_frame"]
    for plot in layer.plots:
        plot.line.width = t["line_width"]
        plot.symbol.size = t["symbol_size"]
        plot.fill.edge_width = t["axis_width"]
        plot.error.width = max(0.6, t["line_width"] * 0.8)
        if t["dash_cycle"] and plot.type in ("line", "line_symbol"):
            plot.line.style = "auto"
        elif not t["dash_cycle"] and plot.line.style == "auto":
            plot.line.style = "-"
        if plot.type in ("column", "bar", "area", "histogram", "box", "pie", "doughnut"):
            if t["hatch_cycle"]:
                plot.fill.hatch = "auto"
            elif plot.fill.hatch == "auto":
                plot.fill.hatch = ""
        for key in ("color",):
            for style in (plot.line, plot.fill):
                setattr(style, key, "auto")
        plot.symbol.face = plot.symbol.edge = "auto"


# ---------------------------------------------------------------------- user templates
def templates_dir() -> Path:
    from rsstatchem.paths import user_data_dir
    path = user_data_dir() / "graph_templates"
    path.mkdir(parents=True, exist_ok=True)
    return path


def template_from_graph(graph: Graph) -> dict:
    """Style of ``graph`` without its data (a reusable Origin-like template)."""
    layer = graph.layers[0] if graph.layers else Layer()
    return {
        "format": "rsstatchem-graph-template", "version": 1,
        "page": [graph.page.width_cm, graph.page.height_cm],
        "font_family": graph.font_family, "math_font": graph.math_font,
        "palette": graph.palette, "fill_palette": graph.fill_palette,
        "dash_cycle": graph.dash_cycle, "hatch_cycle": graph.hatch_cycle,
        "axes": {w: to_dict(layer.axis(w)) for w in ("x", "y", "y2", "z", "x2")},
        "box": layer.box, "legend": to_dict(layer.legend),
        "title_font": to_dict(layer.title_font),
        "plots": [{"line": to_dict(p.line), "symbol": to_dict(p.symbol),
                   "fill": to_dict(p.fill), "error": to_dict(p.error)} for p in layer.plots],
    }


_KEEP_AXIS = ("title", "min", "max", "major_step", "show_title", "breaks", "custom_ticks",
              "scale", "scale_offset")


def apply_template(graph: Graph, template: dict) -> None:
    from rsstatchem.core.graph.model import ErrorStyle, FillStyle, LineStyle, SymbolStyle
    graph.page.width_cm, graph.page.height_cm = template["page"]
    for key in ("font_family", "math_font", "palette", "fill_palette", "dash_cycle",
                "hatch_cycle"):
        if key in template:
            setattr(graph, key, template[key])
    graph.theme = "Template"
    styles = template.get("plots", [])
    for layer in graph.layers:
        for which, data in template.get("axes", {}).items():
            old = layer.axis(which)
            new = from_dict(Axis, data)
            for keep in _KEEP_AXIS:
                setattr(new, keep, getattr(old, keep))
            setattr(layer, which, new)
        layer.box = template.get("box", layer.box)
        if "legend" in template:
            hidden = layer.legend.hidden
            layer.legend = from_dict(type(layer.legend), template["legend"])
            layer.legend.hidden = hidden
        if "title_font" in template:
            layer.title_font = from_dict(Font, template["title_font"])
        for i, plot in enumerate(layer.plots):
            if not styles:
                break
            st = styles[i % len(styles)]
            plot.line = from_dict(LineStyle, st["line"])
            plot.symbol = from_dict(SymbolStyle, st["symbol"])
            plot.fill = from_dict(FillStyle, st["fill"])
            plot.error = from_dict(ErrorStyle, st["error"])
            _fit_plot_style(plot)


def _fit_plot_style(plot: Plot) -> None:
    """Keep a template's style sensible for the plot type it lands on."""
    if plot.type == "scatter":
        plot.line.visible, plot.symbol.visible = False, True
    elif plot.type == "line":
        plot.line.visible, plot.symbol.visible = True, False
    elif plot.type == "line_symbol":
        plot.line.visible = plot.symbol.visible = True


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w\- ]+", "_", name).strip() or "template"


def save_user_template(name: str, graph: Graph) -> Path:
    path = templates_dir() / f"{_safe_name(name)}.json"
    data = template_from_graph(graph)
    data["name"] = name
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return path


def list_user_templates() -> dict[str, Path]:
    out = {}
    for path in sorted(templates_dir().glob("*.json")):
        try:
            out[json.loads(path.read_text(encoding="utf-8")).get("name", path.stem)] = path
        except (OSError, ValueError):
            continue
    return out


def load_user_template(name: str) -> dict:
    path = list_user_templates()[name]
    return json.loads(path.read_text(encoding="utf-8"))


def delete_user_template(name: str) -> None:
    list_user_templates()[name].unlink(missing_ok=True)


# ---------------------------------------------------------------------- user themes
def theme_from_graph(graph: Graph) -> dict:
    """A theme (style only: page, fonts, ticks, frame, widths, colour lists) taken from
    ``graph`` — Origin's "Save Format as Theme"."""
    layer = graph.layers[0] if graph.layers else Layer()
    plots = layer.plots
    return {
        "page": [graph.page.width_cm, graph.page.height_cm],
        "font_family": graph.font_family, "math_font": graph.math_font,
        "label_size": layer.x.label_font.size, "title_size": layer.x.title_font.size,
        "legend_size": layer.legend.font.size, "tick_dir": layer.x.tick_dir,
        "box": layer.box, "grid": layer.x.grid_major or layer.y.grid_major,
        "axis_width": layer.x.line_width,
        "line_width": plots[0].line.width if plots else _BASE["line_width"],
        "symbol_size": plots[0].symbol.size if plots else _BASE["symbol_size"],
        "major_len": layer.x.major_len, "minor_len": layer.x.minor_len,
        "palette": graph.palette, "fill_palette": graph.fill_palette,
        "dash_cycle": graph.dash_cycle, "hatch_cycle": graph.hatch_cycle,
        "legend_frame": layer.legend.frame,
    }


def themes_dir() -> Path:
    from rsstatchem.paths import user_data_dir
    path = user_data_dir() / "graph_themes"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_user_theme(name: str, graph: Graph) -> Path:
    data = {"format": "rsstatchem-graph-theme", "version": 1, "name": name,
            "theme": theme_from_graph(graph)}
    path = themes_dir() / f"{_safe_name(name)}.json"
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return path


def list_user_themes() -> dict[str, dict]:
    out = {}
    for path in sorted(themes_dir().glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("format") == "rsstatchem-graph-theme":
                out[data.get("name", path.stem)] = dict(data["theme"])
        except (OSError, ValueError, KeyError):
            continue
    return out


def apply_user_theme(graph: Graph, name: str) -> None:
    theme = list_user_themes()[name]
    theme = dict(theme, page=tuple(theme.get("page", _BASE["page"])))
    apply_theme(graph, theme, resize_page=False)
    graph.theme = name


def delete_user_theme(name: str) -> None:
    (themes_dir() / f"{_safe_name(name)}.json").unlink(missing_ok=True)

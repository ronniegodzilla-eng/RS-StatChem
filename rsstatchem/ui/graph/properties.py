"""Property editor for graph objects (Origin's Plot Details / Axis / Legend dialogs).

Each model object type has a schema: sections of fields addressed by dotted paths
(``line.color``, ``options.bins``). :class:`PropertyForm` builds widgets from a schema and
reports edits through a callback; the graph window applies them and redraws. The same form is
used in the docked Properties panel and in the modal dialog opened by double-clicking.
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QScrollArea, QSpinBox, QVBoxLayout,
                               QWidget)

from rsstatchem.core.graph.model import Annotation, Axis, Graph, Layer, Legend, Plot, get_path
from rsstatchem.core.graph.text import installed_families
from rsstatchem.core.graph.themes import (COLORMAPS, DASH_NAMES, FONT_FAMILIES, HATCH_NAMES,
                                          PALETTES, SYMBOL_NAMES)

_DASH = [(k, v) for k, v in DASH_NAMES.items() if k != "none"]
_DASH_A = [(k, v) for k, v in DASH_NAMES.items() if k not in ("none", "auto")]
_SYM = list(SYMBOL_NAMES.items())
_HATCH = list(HATCH_NAMES.items())
_CMAPS = [(c, c) for c in COLORMAPS]
_LOCS = [("best", "Best"), ("upper right", "Upper right"), ("upper left", "Upper left"),
         ("lower left", "Lower left"), ("lower right", "Lower right"),
         ("center right", "Center right"), ("center left", "Center left"),
         ("upper center", "Upper center"), ("lower center", "Lower center"),
         ("center", "Center"), ("outside right", "Outside right"),
         ("outside top", "Outside top"), ("outside bottom", "Outside bottom"),
         ("custom", "Custom (X, Y below / drag)")]

# Defaults shown for options that are not stored yet (renderer defaults).
OPTION_DEFAULTS = {
    "connect": "straight", "sort_x": False, "label_size": 8.0, "show_values": False,
    "bins": "auto", "bin_width": None, "scale": "count", "normal_curve": True,
    "cumulative": False, "show_mean": True, "outliers": True, "notch": False, "points": False,
    "whisker": 1.5, "width": 0.5, "size_min": 4.0, "size_max": 24.0, "cmap": "viridis",
    "levels": 10, "labels": False, "grid": "auto", "grid_size": 80, "colorbar_title": "",
    "show_points": False, "wireframe": False, "floor_contour": False, "extrapolate": True,
    "color_by_z": True, "drop_lines": False, "show_labels": True, "show_percent": True,
    "percent_decimals": 1, "start_angle": 90.0, "clockwise": True, "hole": 0.45,
    "explode": 0.0, "percent": False, "theta_unit": "deg", "theta_zero": "E",
    "fill_to": "none", "fill_value": 0.0, "error_band": False, "drop_lines": "none",
    "data_labels": "none", "label_pos": "above", "label_format": "{:.3g}", "skip": 1,
    "dist_curve": "normal", "rug": False, "inner": "box", "side": "both", "bw": "scott",
    "stat": "sd", "style": "column", "dist": "norm", "positions": "filliben",
    "ref_line": True, "fill": True, "size": 5, "chart": "xbar", "k": 1.96, "sectors": 16,
    "speed_bins": 5, "mode": "xyam", "arrow_scale": 1.0, "ref": "fit", "baseline": 0.0,
    "orientation": "vertical",
    "trim": True, "depth": 0.0, "projection": False, "raw": False, "fill_two_colors": False,
}
SCALES = [("linear", "Linear"), ("log10", "Log10"), ("log2", "Log2"), ("ln", "Ln"),
          ("probability", "Probability"), ("probit", "Probit"), ("logit", "Logit"),
          ("reciprocal", "Reciprocal"), ("offset_reciprocal", "Offset Reciprocal"),
          ("weibull", "Double Log Reciprocal (Weibull)"), ("symlog", "Symmetric Log")]
TICK_FORMATS = [("auto", "Auto"), ("decimal", "Decimal: 1000"),
                ("scientific", "Scientific: 1E3"), ("engineering", "Engineering: 1k"),
                ("power10", "×10ⁿ: 1×10³"), ("percent", "Percent (×100 %)")]


def breaks_to_text(breaks: list) -> str:
    out = []
    for b in breaks or []:
        pos = b.get("position")
        out.append(f"{b['from']:g}-{b['to']:g}" + (f"@{pos:g}" if pos is not None else ""))
    return "; ".join(out)


def text_to_breaks(text: str) -> list:
    """"12-88@50; 120-180" → [{"from": 12, "to": 88, "position": 50}, …] (position = %
    of the axis length; omitted = automatic)."""
    import re
    out = []
    for part in (text or "").replace(",", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(-?[\d.eE+]+)\s*(?:-|to|\.\.)\s*(-?[\d.eE+]+)\s*"
                         r"(?:@\s*([\d.]+)\s*%?)?", part)
        if not m:
            raise ValueError(f"Cannot read the break “{part}” (use from-to@position%).")
        a, b = float(m.group(1)), float(m.group(2))
        out.append({"from": min(a, b), "to": max(a, b),
                    "position": float(m.group(3)) if m.group(3) else None})
    return out


def ticks_to_text(ticks: list) -> str:
    return "; ".join(f"{v:g}={t}" for v, t in ticks or [])


def text_to_ticks(text: str) -> list:
    out = []
    for part in (text or "").split(";"):
        if "=" not in part:
            continue
        v, t = part.split("=", 1)
        out.append([float(v.strip().replace(",", ".")), t.strip()])
    return out


def _font(prefix: str, title: str) -> tuple:
    return (title, [(f"{prefix}.family", "Font", "font"),
                    (f"{prefix}.size", "Size (pt)", "float", (3, 72, 1)),
                    (f"{prefix}.bold", "Bold", "bool"), (f"{prefix}.italic", "Italic", "bool"),
                    (f"{prefix}.color", "Color", "color")])


def graph_schema(g: Graph) -> list:
    return [
        ("Page", [("title", "Graph name", "text"),
                  ("page.width_cm", "Width (cm)", "float", (2, 60, 2)),
                  ("page.height_cm", "Height (cm)", "float", (2, 60, 2)),
                  ("page.background", "Background", "color")]),
        ("Style", [("font_family", "Default font", "font"),
                   ("math_font", "Math text font", "choice",
                    [("stixsans", "STIX Sans"), ("dejavusans", "DejaVu Sans"),
                     ("stix", "STIX (Times-like)"), ("cm", "Computer Modern")]),
                   ("palette", "Line colors", "choice", [(k, k) for k in PALETTES]),
                   ("fill_palette", "Fill colors", "choice", [(k, k) for k in PALETTES]),
                   ("dash_cycle", "Increment line style", "bool"),
                   ("hatch_cycle", "Increment fill pattern", "bool")]),
    ]


def layer_schema(layer: Layer) -> list:
    general = [("title", "Layer title", "text"), ("panel_label", "Panel label", "text"),
               ("box", "Frame (closed box)", "bool"), ("background", "Background", "color")]
    out = [("Layer", general), _font("title_font", "Title font")]
    if layer.rect:
        out.append(("Position on page (inset)",
                    [("rect.0", "Left (0–1)", "float", (0, 1, 3)),
                     ("rect.1", "Bottom (0–1)", "float", (0, 1, 3)),
                     ("rect.2", "Width (0–1)", "float", (0.05, 1, 3)),
                     ("rect.3", "Height (0–1)", "float", (0.05, 1, 3))]))
    if layer.kind == "cartesian":
        out.append(("Plots in layer", [
            ("stack", "Column/Bar/Area stacking", "choice",
             [("none", "None (grouped)"), ("stacked", "Stacked"), ("percent", "100% stacked")]),
            ("gap", "Gap between groups", "float", (0, 0.95, 2)),
            ("offset_x", "Offset X (waterfall)", "float", (-1e9, 1e9, 4)),
            ("offset_y", "Offset Y (waterfall)", "float", (-1e9, 1e9, 4)),
            ("show_y2", "Show right Y axis", "bool"),
            ("colorbar", "Color scale", "bool"), ("cmap", "Colormap", "choice", _CMAPS)]))
        out.append(("Linked axes", [
            ("show_x2", "Show top X axis", "bool"),
            ("x2_formula", "Top axis = f(x) (e.g. 1000/(x+273.15))", "text"),
            ("y2_formula", "Right axis = f(x) (when no plot uses it)", "text"),
            ("options.link_x", "Link X range to layer # (0-based)", "optint"),
            ("options.link_y", "Link Y range to layer # (0-based)", "optint")]))
    elif layer.kind == "3d":
        out.append(("3D view", [("elev", "Elevation (°)", "float", (-90, 90, 1)),
                                ("azim", "Azimuth (°)", "float", (-360, 360, 1)),
                                ("colorbar", "Color scale", "bool"),
                                ("cmap", "Colormap", "choice", _CMAPS)]))
    elif layer.kind == "polar":
        out.append(("Polar", [("options.theta_unit", "Theta unit", "choice",
                               [("deg", "Degrees"), ("rad", "Radians")]),
                              ("options.theta_zero", "0° at", "choice",
                               [("E", "Right (E)"), ("N", "Top (N)"), ("W", "Left (W)"),
                                ("S", "Bottom (S)")])]))
    return out


def axis_schema(axis: Axis, kind: str = "cartesian") -> list:
    scale = [("min", "From", "optfloat"), ("max", "To", "optfloat")]
    if kind == "cartesian":
        scale += [("scale", "Type", "choice", SCALES),
                  ("scale_offset", "Offset (offset reciprocal)", "float", (-1e6, 1e6, 3)),
                  ("reverse", "Reverse", "bool")]
    elif kind == "polar":
        scale += [("reverse", "Clockwise (theta)", "bool")]
    scale += [("major_step", "Major increment", "optfloat"),
              ("minor_count", "Minor ticks", "int", (0, 20))]
    if kind == "cartesian":
        scale.append(("decimals", "Decimal places (-1 auto)", "int", (-1, 10)))
    labels = []
    if kind == "cartesian":
        labels = [("tick_format", "Display", "choice", TICK_FORMATS),
                  ("prefix", "Prefix", "text"), ("suffix", "Suffix", "text"),
                  ("custom_ticks", "Custom ticks (value=label; …)", "ticks")]
    brk = [("breaks", "Breaks (from-to@pos%; …)", "breaks"),
           ("break_gap", "Break gap (% of axis)", "float", (0.5, 20, 1))]
    ticks = [("tick_dir", "Direction", "choice", [("in", "In"), ("out", "Out"),
                                                  ("both", "In and out"), ("none", "None")]),
             ("major_len", "Major length (pt)", "float", (0, 30, 1)),
             ("minor_len", "Minor length (pt)", "float", (0, 30, 1)),
             ("tick_width", "Tick thickness", "float", (0, 10, 2)),
             ("show_labels", "Show tick labels", "bool"),
             ("label_rotation", "Label rotation (°)", "float", (-90, 90, 0))]
    if kind == "cartesian":
        ticks.append(("mirror_ticks", "Ticks on opposite side", "bool"))
    line = [("visible", "Show axis", "bool"), ("line_width", "Line thickness", "float",
                                               (0, 10, 2)),
            ("color", "Color", "color"), ("grid_major", "Major grid", "bool"),
            ("grid_minor", "Minor grid", "bool"), ("grid_color", "Grid color", "color"),
            ("grid_style", "Grid style", "choice", _DASH_A)]
    out = [("Title", [("title", "Title", "text"), ("show_title", "Show title", "bool")]),
           _font("title_font", "Title font"), ("Scale", scale)]
    if kind == "cartesian":
        out.append(("Breaks", brk))
    out += [("Ticks", ticks)]
    if labels:
        out.append(("Tick labels", labels))
    out += [_font("label_font", "Tick label font"), ("Line && Grid", line)]
    return out


_LINE_TYPES = ("line", "scatter", "line_symbol")
_POSITIONS = [("filliben", "Filliben (scipy)"), ("blom", "Blom"),
              ("benard", "Benard (median rank)"),
              ("hazen", "Hazen"), ("vdw", "Van der Waerden"), ("km", "Kaplan-Meier")]
_SWITCH = [("line", "Line"), ("scatter", "Scatter"), ("line_symbol", "Line + Symbol"),
           ("column", "Column"), ("bar", "Bar"), ("area", "Area")]


def plot_schema(p: Plot, layer_kind: str = "cartesian") -> list:
    general = [("name", "Name (legend)", "text"), ("visible", "Show", "bool")]
    if layer_kind == "cartesian" and p.type in dict(_SWITCH):
        general.append(("type", "Plot type", "choice", _SWITCH))
    if layer_kind == "cartesian":
        general.append(("y_axis", "Y axis", "choice", [("left", "Left"), ("right", "Right")]))
        general.append(("x_axis", "X axis", "choice", [("bottom", "Bottom"),
                                                        ("top", "Top (independent)")]))
    if p.options.get("data_range"):   # Origin's data reference, e.g. [Book1]Sheet1!B[3:10]
        general.insert(0, ("options.data_range", "Data", "label"))
    out = [("Plot", general)]
    t = p.type
    if t in _LINE_TYPES or layer_kind in ("polar", "ternary") or t in (
            "histogram", "area", "contour", "surface3d", "kde", "ecdf", "qq", "probability",
            "control", "lollipop", "vector", "line3d", "waterfall3d", "pareto", "kde2d"):
        title = {"histogram": "Normal curve line", "contour": "Contour lines",
                 "surface3d": "Mesh lines"}.get(t, "Line")
        out.append((title, [("line.visible", "Show", "bool"),
                            ("line.color", "Color", "acolor"),
                            ("line.style", "Style", "choice", _DASH),
                            ("line.width", "Width", "float", (0, 20, 2))]))
    if t in _LINE_TYPES or t in ("scatter3d", "bubble", "color_scatter", "qq",
                                 "probability", "bland_altman", "lollipop", "line3d") or \
            layer_kind in ("polar", "ternary"):
        sym = [("symbol.visible", "Show", "bool"), ("symbol.shape", "Shape", "choice", _SYM),
               ("symbol.size", "Size", "float", (0.5, 60, 1)),
               ("symbol.filled", "Filled", "bool"), ("symbol.face", "Fill color", "acolor"),
               ("symbol.edge", "Edge color", "acolor"),
               ("symbol.edge_width", "Edge width", "float", (0, 10, 2))]
        if t in ("bubble", "color_scatter", "scatter3d"):
            sym = sym[1:]
        out.append(("Symbol", sym))
    if t in ("column", "bar", "area", "histogram", "box", "pie", "doughnut", "bubble",
             "contour", "surface3d", "violin", "meanbar", "pareto", "floating_column", "kde",
             "bar3d", "waterfall3d", "radar", "kde2d"):
        fill = [("fill.visible", "Fill", "bool"), ("fill.color", "Color", "acolor"),
                ("fill.alpha", "Opacity", "float", (0, 1, 2)),
                ("fill.hatch", "Pattern", "choice", _HATCH),
                ("fill.edge_color", "Border color", "color"),
                ("fill.edge_width", "Border width", "float", (0, 10, 2))]
        if t in ("pie", "doughnut"):
            fill = [fill[4], fill[5]]
        elif t in ("contour", "surface3d", "bubble"):
            fill = [fill[0], fill[2]] if t != "bubble" else [fill[1], fill[2]]
        out.append(("Fill && Border", fill))
    if t in _LINE_TYPES or t in ("column", "bar", "meanbar"):
        out.append(("Error bars", [("error.color", "Color", "acolor"),
                                   ("error.width", "Width", "float", (0, 10, 2)),
                                   ("error.cap", "Cap size", "float", (0, 30, 1)),
                                   ("error.direction", "Direction", "choice",
                                    [("both", "Plus and minus"), ("plus", "Plus"),
                                     ("minus", "Minus")])]))
    opts = _options(t, layer_kind)
    if opts:
        out.append(("Options", opts))
    return out


def _options(t: str, kind: str) -> list:
    o = "options."
    if t in _LINE_TYPES and kind == "cartesian":
        return [(o + "connect", "Connect", "choice",
                 [("straight", "Straight"), ("step", "Step (centre)"),
                  ("step_h", "Horizontal step"), ("step_v", "Vertical step"),
                  ("spline", "Spline")]),
                (o + "sort_x", "Sort by X", "bool"),
                (o + "skip", "Show every n-th symbol", "int", (1, 1000)),
                (o + "fill_to", "Fill area under curve", "choice",
                 [("none", "None"), ("zero", "To Y = 0"), ("value", "To Y = value"),
                  ("next", "To next plot")]),
                (o + "fill_value", "Fill to Y =", "float", (-1e12, 1e12, 4)),
                (o + "fill_two_colors", "Two colors (above/below)", "bool"),
                (o + "error_band", "Y error as band", "bool"),
                (o + "drop_lines", "Drop lines", "choice",
                 [("none", "None"), ("x", "Vertical"), ("y", "Horizontal"),
                  ("both", "Both")]),
                (o + "data_labels", "Data labels", "choice",
                 [("none", "None / label column"), ("y", "Y value"), ("x", "X value"),
                  ("xy", "(X, Y)")]),
                (o + "label_format", "Label format", "text"),
                (o + "label_pos", "Label position", "choice",
                 [("above", "Above"), ("below", "Below"), ("left", "Left"),
                  ("right", "Right"), ("center", "Center"), ("above right", "Above right")]),
                (o + "label_size", "Data label size", "float", (3, 30, 1))]
    if kind == "ternary":
        return [(o + "percent", "Axis scale 0–100", "bool")]
    return {
        "column": [(o + "show_values", "Show values", "bool")],
        "bar": [(o + "show_values", "Show values", "bool")],
        "histogram": [(o + "bins", "Bins (number or auto)", "text"),
                      (o + "bin_width", "Bin width", "optfloat"),
                      (o + "scale", "Y scale", "choice", [("count", "Count"),
                                                          ("percent", "Percent"),
                                                          ("density", "Density")]),
                      (o + "normal_curve", "Normal curve", "bool"),
                      (o + "dist_curve", "Distribution curve", "choice",
                       [("normal", "Normal"), ("lognormal", "Lognormal"),
                        ("exponential", "Exponential"), ("gamma", "Gamma"),
                        ("weibull", "Weibull"), ("kde", "Kernel density"),
                        ("none", "None")]),
                      (o + "rug", "Rug", "bool"),
                      (o + "orientation", "Orientation", "choice",
                       [("vertical", "Vertical"), ("horizontal", "Horizontal")]),
                      (o + "cumulative", "Cumulative", "bool")],
        "violin": [(o + "inner", "Inside", "choice", [("box", "Box"),
                                                      ("quartile", "Quartile lines"),
                                                      ("points", "Data points"),
                                                      ("none", "None")]),
                   (o + "side", "Side", "choice", [("both", "Both"), ("left", "Left half"),
                                                   ("right", "Right half")]),
                   (o + "bw", "Bandwidth (scott, silverman or factor)", "text"),
                   (o + "trim", "Trim to data range", "bool"),
                   (o + "show_mean", "Show mean", "bool"),
                   (o + "width", "Width", "float", (0.05, 1, 2))],
        "meanbar": [(o + "stat", "Error", "choice", [("sd", "SD"), ("se", "SE"),
                                                     ("ci", "95% CI")]),
                    (o + "style", "Style", "choice", [("column", "Column"),
                                                      ("point", "Point (interval plot)")]),
                    (o + "points", "Show data points", "bool"),
                    (o + "width", "Width", "float", (0.05, 1, 2))],
        "qq": [(o + "dist", "Distribution", "choice",
                [("norm", "Normal"), ("lognorm", "Lognormal"), ("expon", "Exponential"),
                 ("weibull_min", "Weibull"), ("gamma", "Gamma"), ("logistic", "Logistic"),
                 ("uniform", "Uniform")]),
               (o + "positions", "Plotting positions", "choice", _POSITIONS),
               (o + "ref_line", "Reference line", "bool"),
               (o + "ref", "Reference line through", "choice",
                [("fit", "Least squares fit"), ("quartile", "Quartiles")])],
        "probability": [(o + "positions", "Plotting positions", "choice", _POSITIONS),
                        (o + "ref_line", "Normal reference line", "bool")],
        "kde": [(o + "bw", "Bandwidth (scott, silverman or factor)", "text"),
                (o + "fill", "Fill under curve", "bool"), (o + "rug", "Rug", "bool"),
                (o + "scale", "Scale", "choice", [("density", "Density"),
                                                  ("count", "Count")])],
        "kde2d": [(o + "bw", "Bandwidth (scott, silverman or factor)", "text"),
                  (o + "levels", "Levels", "int", (2, 100)),
                  (o + "cmap", "Colormap", "choice", _CMAPS),
                  (o + "show_points", "Show data points", "bool")],
        "ecdf": [(o + "percent", "Percent", "bool"),
                 (o + "normal_ref", "Normal CDF reference", "bool")],
        "pareto": [(o + "raw", "Count raw values", "bool")],
        "bland_altman": [(o + "k", "Limits of agreement (± k SD)", "float", (0.5, 5, 2))],
        "control": [(o + "chart", "Chart", "choice", [("xbar", "X-bar"), ("r", "Range"),
                                                      ("s", "S"), ("i", "Individuals"),
                                                      ("mr", "Moving range")]),
                    (o + "size", "Subgroup size", "int", (2, 25))],
        "lollipop": [(o + "baseline", "Baseline", "float", (-1e12, 1e12, 4))],
        "vector": [(o + "mode", "Mode", "choice", [("xyam", "XYAM (angle, magnitude)"),
                                                   ("xyxy", "XYXY (start, end)")]),
                   (o + "arrow_scale", "Length scale", "float", (0.001, 1000, 3))],
        "bar3d": [(o + "color_by_z", "Color by Z", "bool"),
                  (o + "cmap", "Colormap", "choice", _CMAPS),
                  (o + "width", "Bar width", "float", (0.05, 2, 2))],
        "line3d": [(o + "drop_lines", "Drop lines", "bool"),
                   (o + "projection", "Projection on floor", "bool")],
        "waterfall3d": [(o + "depth", "Depth (Y position)", "float", (-1e6, 1e6, 3)),
                        (o + "fill", "Fill to floor", "bool")],
        "radar": [(o + "fill", "Fill", "bool")],
        "windrose": [(o + "sectors", "Direction sectors", "int", (4, 72)),
                     (o + "speed_bins", "Speed bins", "int", (1, 12)),
                     (o + "percent", "Percent", "bool"),
                     (o + "cmap", "Colormap", "choice", _CMAPS)],
        "box": [(o + "style", "Style", "choice", [("box", "Box"),
                                                  ("half_points", "Half box + points")]),
                (o + "show_mean", "Show mean", "bool"), (o + "outliers", "Outliers", "bool"),
                (o + "notch", "Notch", "bool"), (o + "points", "Show data points", "bool"),
                (o + "whisker", "Whiskers", "choice",
                 [(1.5, "1.5 IQR"), (3.0, "3 IQR"), ("range", "Min–Max"),
                  ("5-95", "5–95 percentile"), ("10-90", "10–90 percentile")]),
                (o + "width", "Box width", "float", (0.05, 1, 2))],
        "bubble": [(o + "size_min", "Smallest bubble (pt)", "float", (1, 100, 1)),
                   (o + "size_max", "Largest bubble (pt)", "float", (1, 200, 1)),
                   (o + "cmap", "Colormap", "choice", _CMAPS)],
        "color_scatter": [(o + "cmap", "Colormap", "choice", _CMAPS),
                          (o + "colorbar_title", "Color scale title", "text")],
        "contour": [(o + "levels", "Levels", "int", (2, 100)),
                    (o + "labels", "Contour labels", "bool"),
                    (o + "cmap", "Colormap", "choice", _CMAPS),
                    (o + "grid", "Gridding", "choice",
                     [("auto", "Auto (regular → direct)"), ("linear", "Linear"),
                      ("cubic", "Cubic"), ("nearest", "Nearest")]),
                    (o + "grid_size", "Grid points", "int", (10, 400)),
                    (o + "show_points", "Show data points", "bool"),
                    (o + "colorbar_title", "Color scale title", "text")],
        "heatmap": [(o + "cmap", "Colormap", "choice", _CMAPS),
                    (o + "show_values", "Show values", "bool"),
                    (o + "grid", "Gridding", "choice",
                     [("auto", "Auto"), ("linear", "Linear"), ("nearest", "Nearest")]),
                    (o + "colorbar_title", "Color scale title", "text")],
        "surface3d": [(o + "cmap", "Colormap", "choice", _CMAPS),
                      (o + "wireframe", "Wireframe only", "bool"),
                      (o + "floor_contour", "Contour on floor", "bool"),
                      (o + "extrapolate", "Fill outside data hull", "bool"),
                      (o + "colorbar_title", "Color scale title", "text")],
        "scatter3d": [(o + "color_by_z", "Color by Z", "bool"),
                      (o + "drop_lines", "Drop lines", "bool"),
                      (o + "cmap", "Colormap", "choice", _CMAPS)],
        "pie": _pie_opts(o), "doughnut": _pie_opts(o) + [(o + "hole", "Hole size", "float",
                                                            (0.05, 0.95, 2))],
    }.get(t, [])


def _pie_opts(o: str) -> list:
    return [(o + "show_labels", "Category labels", "bool"),
            (o + "show_percent", "Percentages", "bool"),
            (o + "percent_decimals", "Decimals", "int", (0, 4)),
            (o + "start_angle", "Start angle (°)", "float", (-360, 360, 0)),
            (o + "clockwise", "Clockwise", "bool"),
            (o + "explode", "Explode", "float", (0, 0.5, 2))]


def legend_schema(lg: Legend, layer: Layer) -> list:
    out = [("Legend", [("visible", "Show", "bool"), ("loc", "Position", "choice", _LOCS),
                       ("x", "X (custom, 0–1)", "float", (-1, 2, 3)),
                       ("y", "Y (custom, 0–1)", "float", (-1, 2, 3)),
                       ("frame", "Frame", "bool"), ("ncols", "Columns", "int", (1, 10)),
                       ("title", "Title", "text")]),
           _font("font", "Font")]
    entries = [(f"@entry:{p.id}", p.name or p.type, "entry") for p in layer.plots]
    if entries:
        out.append(("Entries", entries))
    return out


def ref_schema(r, layer: Layer) -> list:
    plots = [("", "(first plot)")] + [(p.id, p.name or p.type) for p in layer.plots]
    return [("Reference line", [
        ("axis", "Axis", "choice", [("x", "X (vertical line)"), ("y", "Y (horizontal line)"),
                                    ("y2", "Right Y")]),
        ("kind", "Type", "choice", [("line", "Line"), ("band", "Shaded range")]),
        ("mode", "At", "choice", [("value", "Value(s)"), ("mean", "Mean"),
                                  ("median", "Median"), ("mean_sd", "Mean ± k·SD"),
                                  ("mean_se", "Mean ± k·SE"), ("mean_ci", "Mean ± 95% CI")]),
        ("value", "Value / From", "float", (-1e12, 1e12, 6)),
        ("value2", "To (range)", "float", (-1e12, 1e12, 6)),
        ("k", "k", "float", (0, 10, 2)), ("plot", "Statistic of plot", "choice", plots),
        ("label", "Label", "text"), ("show_label", "Show label", "bool")]),
        ("Line", [("line.visible", "Show", "bool"), ("line.color", "Color", "color"),
                  ("line.style", "Style", "choice", _DASH_A),
                  ("line.width", "Width", "float", (0, 20, 2))]),
        ("Fill (range)", [("fill.color", "Color", "color"),
                          ("fill.alpha", "Opacity", "float", (0, 1, 2))])]


def annotation_schema(a: Annotation) -> list:
    pos = [("coords", "Attach to", "choice", [("axes", "Layer frame (0–1)"),
                                             ("data", "Axis scales (data)"),
                                             ("figure", "Page (0–1)")]),
           ("x", "X", "float", (-1e9, 1e9, 4)), ("y", "Y", "float", (-1e9, 1e9, 4))]
    if a.kind != "text":
        pos += [("x2", "X2", "float", (-1e9, 1e9, 4)), ("y2", "Y2", "float", (-1e9, 1e9, 4))]
    out = []
    if a.kind == "text":
        out.append(("Text", [("text", "Text", "text"), ("frame", "Frame", "bool"),
                             ("fill.visible", "Background", "bool"),
                             ("fill.color", "Background color", "color")]))
        out.append(_font("font", "Font"))
    else:
        line = [("line.color", "Color", "color"), ("line.width", "Width", "float", (0, 20, 2)),
                ("line.style", "Style", "choice", _DASH_A)]
        if a.kind == "arrow":
            line.append(("head_size", "Head size", "float", (1, 60, 1)))
        out.append(("Line", line))
        if a.kind == "rect":
            out.append(("Fill", [("fill.visible", "Fill", "bool"),
                                 ("fill.color", "Color", "color"),
                                 ("fill.alpha", "Opacity", "float", (0, 1, 2))]))
    out.append(("Position", pos))
    return out


# ====================================================================== widgets
class ColorButton(QWidget):
    changed = Signal(str)

    def __init__(self, value: str, allow_auto: bool = False, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.button = QPushButton()
        self.button.setFixedWidth(64)
        self.button.clicked.connect(self._pick)
        lay.addWidget(self.button)
        self.auto = None
        if allow_auto:
            self.auto = QCheckBox("Auto")
            self.auto.toggled.connect(self._auto_toggled)
            lay.addWidget(self.auto)
        lay.addStretch()
        self._value = "#000000"
        self.set_value(value)

    def set_value(self, value: str) -> None:
        is_auto = value == "auto"
        if self.auto is not None:
            self.auto.blockSignals(True)
            self.auto.setChecked(is_auto)
            self.auto.blockSignals(False)
        if not is_auto:
            self._value = value
        shown = "#d0d0d0" if is_auto else (value if value != "none" else "#ffffff")
        self.button.setStyleSheet(f"background:{shown}; border:1px solid #777; "
                                  f"min-height:16px;")
        self.button.setText("auto" if is_auto else "")

    def _pick(self) -> None:
        start = QColor(self._value if self._value.startswith("#") else "#000000")
        color = QColorDialog.getColor(start, self, "Color")
        if color.isValid():
            self.set_value(color.name())
            self.changed.emit(color.name())

    def _auto_toggled(self, on: bool) -> None:
        value = "auto" if on else self._value
        self.set_value(value)
        self.changed.emit(value)


class PropertyForm(QWidget):
    """Widgets for one object's schema; edits call ``on_change(path, value)``."""

    def __init__(self, obj, sections: list, on_change: Callable[[str, object], None],
                 layer: Layer | None = None, parent=None):
        super().__init__(parent)
        self.obj, self.layer, self.on_change = obj, layer, on_change
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        for title, fields in sections:
            box = QGroupBox(title.replace("&&", "&"))
            form = QFormLayout(box)
            form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            for field in fields:
                path, label, kind = field[:3]
                extra = field[3] if len(field) > 3 else None
                widget = self._editor(path, kind, extra)
                if kind in ("bool", "entry"):
                    form.addRow("", widget)
                    widget.setText(label)
                else:
                    form.addRow(label, widget)
            outer.addWidget(box)
        outer.addStretch()

    def _value(self, path: str):
        if path.startswith("@entry:"):
            return path.split(":", 1)[1] not in self.layer.legend.hidden
        try:
            return get_path(self.obj, path)
        except (KeyError, AttributeError):
            return OPTION_DEFAULTS.get(path.split(".")[-1])

    def _editor(self, path: str, kind: str, extra):
        value = self._value(path)
        emit = lambda v, p=path: self.on_change(p, v)  # noqa: E731
        if kind in ("bool", "entry"):
            w = QCheckBox()
            w.setChecked(bool(value))
            w.toggled.connect(emit)
            return w
        if kind == "text":
            w = QLineEdit("" if value is None else str(value))
            w.editingFinished.connect(lambda w=w, p=path: self.on_change(p, w.text()))
            if path.endswith("title") or path == "text" or path.endswith("name"):
                w.setToolTip("Supports H$_2$O, cm$^{-1}$, $\\alpha$ and Origin codes "
                             "\\+(sup) \\-(sub) \\g(a)")
            return w
        if kind in ("breaks", "ticks"):
            to_text = breaks_to_text if kind == "breaks" else ticks_to_text
            parse = text_to_breaks if kind == "breaks" else text_to_ticks
            w = QLineEdit(to_text(value or []))
            w.setPlaceholderText("12-88@50; 150-300" if kind == "breaks" else
                                 "1=Low; 2=Mid; 3=High")

            def done_list(w=w, p=path, parse=parse):
                try:
                    self.on_change(p, parse(w.text()))
                    w.setStyleSheet("")
                except ValueError as exc:
                    w.setStyleSheet("background:#ffe0e0;")
                    w.setToolTip(str(exc))
            w.editingFinished.connect(done_list)
            return w
        if kind == "optint":
            w = QLineEdit("" if value is None else str(int(value)))
            w.setPlaceholderText("None")

            def done_int(w=w, p=path):
                text = w.text().strip()
                try:
                    self.on_change(p, int(text) if text else None)
                except ValueError:
                    w.setText("")
            w.editingFinished.connect(done_int)
            return w
        if kind == "optfloat":
            w = QLineEdit("" if value is None else f"{value:g}")
            w.setPlaceholderText("Auto")

            def done(w=w, p=path):
                text = w.text().strip().replace(",", ".")
                if not text:
                    self.on_change(p, None)
                    return
                try:
                    self.on_change(p, float(text))
                except ValueError:
                    w.setText("")
            w.editingFinished.connect(done)
            return w
        if kind == "float":
            lo, hi, dec = extra or (-1e9, 1e9, 2)
            w = QDoubleSpinBox()
            w.setRange(lo, hi)
            w.setDecimals(dec)
            w.setSingleStep(10 ** -dec if dec else 1)
            if dec <= 2 and hi - lo <= 2:
                w.setSingleStep(0.05)
            w.setValue(float(value or 0))
            w.setKeyboardTracking(False)
            w.valueChanged.connect(emit)
            return w
        if kind == "int":
            lo, hi = extra or (0, 1000)
            w = QSpinBox()
            w.setRange(lo, hi)
            w.setValue(int(value or 0))
            w.setKeyboardTracking(False)
            w.valueChanged.connect(emit)
            return w
        if kind in ("color", "acolor"):
            w = ColorButton(str(value or "#000000"), allow_auto=kind == "acolor")
            w.changed.connect(emit)
            return w
        if kind == "choice":
            w = QComboBox()
            values = [v for v, _ in extra]
            for v, text in extra:
                w.addItem(text, v)
            if value not in values:
                w.addItem(str(value), value)
                values.append(value)
            w.setCurrentIndex(values.index(value))
            w.currentIndexChanged.connect(lambda i, w=w, p=path: self.on_change(p, w.itemData(i)))
            return w
        if kind == "font":
            w = QComboBox()
            w.setEditable(True)
            installed = set(installed_families())
            names = [f for f in FONT_FAMILIES if f in installed]
            if path != "font_family":
                w.addItem("(graph default)", "")
            for name in names:
                w.addItem(name, name)
            idx = w.findData(value)
            if idx < 0:
                w.addItem(str(value), value)
                idx = w.count() - 1
            w.setCurrentIndex(idx)
            w.activated.connect(lambda i, w=w, p=path: self.on_change(p, w.itemData(i)))
            return w
        return QLabel(str(value))


def schema_for(obj, ref: tuple, graph: Graph) -> tuple[list, Layer | None]:
    kind = ref[0]
    layer = graph.layers[ref[1]] if len(ref) > 1 and ref[1] < len(graph.layers) else None
    if kind == "graph":
        return graph_schema(graph), None
    if kind in ("layer", "colorbar", "roi"):
        return layer_schema(layer), layer
    if kind == "ref":
        return ref_schema(obj, layer), layer
    if kind == "axis":
        return axis_schema(obj, layer.kind), layer
    if kind == "plot":
        return plot_schema(obj, layer.kind), layer
    if kind == "legend":
        return legend_schema(obj, layer), layer
    if kind == "annotation":
        return annotation_schema(obj), layer
    return [], layer


class PropertiesDialog(QDialog):
    """Modal editor with live preview; Cancel restores the state when it was opened."""

    def __init__(self, title: str, obj, sections, on_change, layer=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(420, 620)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(PropertyForm(obj, sections, on_change, layer))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(scroll)
        lay.addWidget(buttons)

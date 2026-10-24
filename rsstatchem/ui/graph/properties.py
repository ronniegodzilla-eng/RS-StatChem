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
}


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
        scale += [("scale", "Type", "choice", [("linear", "Linear"), ("log10", "Log10"),
                                               ("log2", "Log2"), ("ln", "Ln")]),
                  ("reverse", "Reverse", "bool")]
    elif kind == "polar":
        scale += [("reverse", "Clockwise (theta)", "bool")]
    scale += [("major_step", "Major increment", "optfloat"),
              ("minor_count", "Minor ticks", "int", (0, 20))]
    if kind == "cartesian":
        scale.append(("decimals", "Decimal places (-1 auto)", "int", (-1, 10)))
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
    return [("Title", [("title", "Title", "text"), ("show_title", "Show title", "bool")]),
            _font("title_font", "Title font"), ("Scale", scale), ("Ticks", ticks),
            _font("label_font", "Tick label font"), ("Line && Grid", line)]


_LINE_TYPES = ("line", "scatter", "line_symbol")
_SWITCH = [("line", "Line"), ("scatter", "Scatter"), ("line_symbol", "Line + Symbol"),
           ("column", "Column"), ("bar", "Bar"), ("area", "Area")]


def plot_schema(p: Plot, layer_kind: str = "cartesian") -> list:
    general = [("name", "Name (legend)", "text"), ("visible", "Show", "bool")]
    if layer_kind == "cartesian" and p.type in dict(_SWITCH):
        general.append(("type", "Plot type", "choice", _SWITCH))
    if layer_kind == "cartesian":
        general.append(("y_axis", "Y axis", "choice", [("left", "Left"), ("right", "Right")]))
    out = [("Plot", general)]
    t = p.type
    if t in _LINE_TYPES or layer_kind in ("polar", "ternary") or t in (
            "histogram", "area", "contour", "surface3d"):
        title = {"histogram": "Normal curve line", "contour": "Contour lines",
                 "surface3d": "Mesh lines"}.get(t, "Line")
        out.append((title, [("line.visible", "Show", "bool"),
                            ("line.color", "Color", "acolor"),
                            ("line.style", "Style", "choice", _DASH),
                            ("line.width", "Width", "float", (0, 20, 2))]))
    if t in _LINE_TYPES or t in ("scatter3d", "bubble", "color_scatter") or \
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
             "contour", "surface3d"):
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
    if t in _LINE_TYPES or t in ("column", "bar"):
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
        return [(o + "connect", "Connect", "choice", [("straight", "Straight"),
                                                      ("step", "Step"), ("spline", "Spline")]),
                (o + "sort_x", "Sort by X", "bool"),
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
                      (o + "cumulative", "Cumulative", "bool")],
        "box": [(o + "show_mean", "Show mean", "bool"), (o + "outliers", "Outliers", "bool"),
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
    if kind in ("layer", "colorbar"):
        return layer_schema(layer), layer
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

"""Graph document model for the Graph Studio (mirrors OriginLab's graph hierarchy).

A :class:`Graph` (Origin's graph window / page) holds :class:`Layer` objects; each layer has
its axes, plots, legend and annotations. Plot data is a snapshot of dataset columns stored in
``Graph.data`` so a saved ``.rsg`` file reopens without the original dataset.

Every class is a plain dataclass, serialised to JSON with :func:`to_dict` / :func:`from_dict`.
Colours may be ``"auto"``: they are then taken from the graph's palette by plot index, like
Origin's colour increment. No Qt or matplotlib imports here.
"""
from __future__ import annotations

import copy
import dataclasses
import itertools
import math
import typing
import uuid
from dataclasses import dataclass, field

import numpy as np

FORMAT = "rsstatchem-graph"
VERSION = 1

# Plot types understood by the renderer.
PLOT_TYPES = ("line", "scatter", "line_symbol", "column", "bar", "area", "pie", "doughnut",
              "histogram", "box", "bubble", "color_scatter", "contour", "heatmap",
              "surface3d", "scatter3d")
LAYER_KINDS = ("cartesian", "polar", "3d", "ternary", "pie")


def new_id(prefix: str = "") -> str:
    return prefix + uuid.uuid4().hex[:8]


@dataclass
class Font:
    family: str = ""          # "" = graph default family
    size: float = 10.0
    bold: bool = False
    italic: bool = False
    color: str = "#000000"


@dataclass
class LineStyle:
    visible: bool = True
    color: str = "auto"
    style: str = "-"          # "-", "--", ":", "-.", "auto"
    width: float = 1.2


@dataclass
class SymbolStyle:
    visible: bool = False
    shape: str = "auto"       # matplotlib marker code or "auto"
    size: float = 6.0
    filled: bool = True
    face: str = "auto"
    edge: str = "auto"
    edge_width: float = 0.8


@dataclass
class FillStyle:
    visible: bool = True
    color: str = "auto"
    alpha: float = 1.0
    hatch: str = ""           # "", "///", "xxx", "...", "auto"
    edge_color: str = "#000000"
    edge_width: float = 0.8


@dataclass
class ErrorStyle:
    color: str = "auto"
    width: float = 1.0
    cap: float = 3.0
    direction: str = "both"   # both | plus | minus


@dataclass
class Axis:
    title: str = ""
    show_title: bool = True
    title_font: Font = field(default_factory=lambda: Font(size=11.0))
    label_font: Font = field(default_factory=lambda: Font(size=10.0))
    min: float | None = None
    max: float | None = None
    scale: str = "linear"     # linear | log10 | log2 | ln
    reverse: bool = False
    major_step: float | None = None
    minor_count: int = 1
    tick_dir: str = "in"      # in | out | both | none
    major_len: float = 5.0
    minor_len: float = 3.0
    tick_width: float = 1.0
    show_labels: bool = True
    label_rotation: float = 0.0
    decimals: int = -1        # -1 = automatic
    visible: bool = True
    line_width: float = 1.0
    color: str = "#000000"
    grid_major: bool = False
    grid_minor: bool = False
    grid_color: str = "#c8c8c8"
    grid_style: str = ":"
    mirror_ticks: bool = False


@dataclass
class Legend:
    visible: bool = True
    loc: str = "best"         # matplotlib location string or "custom"
    x: float = 0.02           # custom position: lower-left corner in axes fraction
    y: float = 0.70
    frame: bool = True
    font: Font = field(default_factory=lambda: Font(size=9.0))
    ncols: int = 1
    title: str = ""
    hidden: list[str] = field(default_factory=list)


@dataclass
class Annotation:
    kind: str = "text"        # text | arrow | line | rect
    id: str = field(default_factory=lambda: new_id("a"))
    text: str = "Text"
    coords: str = "axes"      # axes | data | figure
    x: float = 0.5
    y: float = 0.5
    x2: float = 0.7
    y2: float = 0.7
    font: Font = field(default_factory=lambda: Font(size=10.0))
    line: LineStyle = field(default_factory=lambda: LineStyle(color="#000000", width=1.0))
    fill: FillStyle = field(default_factory=lambda: FillStyle(visible=False,
                                                                color="#ffffff"))
    frame: bool = False
    head_size: float = 10.0


@dataclass
class Plot:
    type: str = "line"
    id: str = field(default_factory=lambda: new_id("p"))
    name: str = ""
    visible: bool = True
    x: str | None = None
    y: str | None = None
    z: str | None = None
    xerr: str | None = None
    yerr: str | None = None
    labels: str | None = None
    y_axis: str = "left"      # left | right
    line: LineStyle = field(default_factory=LineStyle)
    symbol: SymbolStyle = field(default_factory=SymbolStyle)
    fill: FillStyle = field(default_factory=FillStyle)
    error: ErrorStyle = field(default_factory=ErrorStyle)
    options: dict = field(default_factory=dict)


@dataclass
class Layer:
    kind: str = "cartesian"
    id: str = field(default_factory=lambda: new_id("L"))
    title: str = ""
    title_font: Font = field(default_factory=lambda: Font(size=11.0, bold=True))
    x: Axis = field(default_factory=Axis)
    y: Axis = field(default_factory=Axis)
    y2: Axis = field(default_factory=Axis)
    z: Axis = field(default_factory=Axis)
    show_y2: bool = False
    box: bool = True
    background: str = "#ffffff"
    grid: list[int] = field(default_factory=lambda: [0, 0, 1, 1])  # row, col, rowspan, colspan
    rect: list[float] | None = None   # manual position (insets): left, bottom, width, height
    stack: str = "none"       # none | stacked | percent (column, bar, area)
    gap: float = 0.25         # gap between column groups, fraction of a category
    offset_x: float = 0.0     # waterfall offsets between successive line plots
    offset_y: float = 0.0
    colorbar: bool = True
    cmap: str = "viridis"
    elev: float = 25.0
    azim: float = -60.0
    plots: list[Plot] = field(default_factory=list)
    legend: Legend = field(default_factory=Legend)
    annotations: list[Annotation] = field(default_factory=list)
    panel_label: str = ""     # "(a)", "(b)"… drawn at the top-left corner
    options: dict = field(default_factory=dict)   # kind-specific (polar theta unit/zero)

    def axis(self, which: str) -> Axis:
        return getattr(self, which)


@dataclass
class Column:
    name: str
    label: str = ""
    kind: str = "num"         # num | text
    values: list = field(default_factory=list)
    value_labels: dict = field(default_factory=dict)

    @property
    def title(self) -> str:
        return self.label or self.name

    def array(self) -> np.ndarray:
        if self.kind == "text":
            return np.array(["" if v is None else str(v) for v in self.values], dtype=object)
        return np.array([np.nan if v is None else v for v in self.values], dtype=float)

    @property
    def categorical(self) -> bool:
        return bool(self.value_labels)

    @classmethod
    def from_array(cls, name: str, values, label: str = "", value_labels: dict | None = None,
                   kind: str | None = None) -> "Column":
        arr = np.asarray(values)
        if kind is None:
            kind = "num" if arr.dtype.kind in "fiub" else "text"
        if kind == "num":
            vals = [None if not math.isfinite(v) else float(v)
                    for v in np.asarray(arr, dtype=float)]
        else:
            vals = ["" if v is None else str(v) for v in arr]
        return cls(name, label, kind, vals, dict(value_labels or {}))


@dataclass
class Page:
    width_cm: float = 12.0
    height_cm: float = 9.0
    background: str = "#ffffff"


@dataclass
class Graph:
    title: str = "Graph1"
    page: Page = field(default_factory=Page)
    rows: int = 1
    cols: int = 1
    font_family: str = "Arial"
    math_font: str = "stixsans"
    palette: str = "Origin Classic"
    fill_palette: str = "Nature"
    dash_cycle: bool = False
    hatch_cycle: bool = False
    theme: str = "Origin Default"
    layers: list[Layer] = field(default_factory=list)
    data: dict[str, Column] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ data
    def add_column(self, col: Column, key: str | None = None) -> str:
        key = key or col.name
        base, n = key, 1
        while key in self.data and self.data[key] != col:
            n += 1
            key = f"{base} ({n})"
        self.data[key] = col
        return key

    def column(self, key: str | None) -> Column | None:
        return self.data.get(key) if key else None

    def values(self, key: str | None) -> np.ndarray | None:
        col = self.column(key)
        return None if col is None else col.array()

    def prune_data(self) -> None:
        """Drop snapshot columns no plot refers to (after plots were removed)."""
        used = set()
        for layer in self.layers:
            for p in layer.plots:
                used.update(k for k in (p.x, p.y, p.z, p.xerr, p.yerr, p.labels) if k)
                used.update(v for k, v in p.options.items()
                            if k.endswith("_col") and isinstance(v, str))
        self.data = {k: v for k, v in self.data.items() if k in used}

    # ------------------------------------------------------------------ lookup
    def find(self, obj_id: str):
        for layer in self.layers:
            if layer.id == obj_id:
                return layer
            for item in itertools.chain(layer.plots, layer.annotations):
                if item.id == obj_id:
                    return item
        return None

    def layer_of(self, obj_id: str) -> Layer | None:
        for layer in self.layers:
            if layer.id == obj_id or any(i.id == obj_id for i in
                                         itertools.chain(layer.plots, layer.annotations)):
                return layer
        return None

    def copy(self) -> "Graph":
        return copy.deepcopy(self)


# ---------------------------------------------------------------------- serialisation
def _clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (np.floating,)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def to_dict(obj) -> dict:
    if isinstance(obj, Column):
        return {"name": obj.name, "label": obj.label, "kind": obj.kind,
                "values": _clean(list(obj.values)),
                "value_labels": [[_clean(k), str(v)] for k, v in obj.value_labels.items()]}
    out = {}
    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name)
        if dataclasses.is_dataclass(value):
            out[f.name] = to_dict(value)
        elif isinstance(value, list) and value and dataclasses.is_dataclass(value[0]):
            out[f.name] = [to_dict(v) for v in value]
        elif isinstance(value, dict) and value and \
                all(isinstance(v, Column) for v in value.values()):
            out[f.name] = {k: to_dict(v) for k, v in value.items()}
        else:
            out[f.name] = _clean(copy.deepcopy(value))
    return out


_HINTS: dict[type, dict] = {}


def _hints(cls) -> dict:
    if cls not in _HINTS:
        _HINTS[cls] = typing.get_type_hints(cls)
    return _HINTS[cls]


def from_dict(cls, data: dict):
    if cls is Column:
        labels = {}
        for k, v in data.get("value_labels", []):
            labels[float(k) if isinstance(k, (int, float)) else k] = v
        return Column(data["name"], data.get("label", ""), data.get("kind", "num"),
                      list(data.get("values", [])), labels)
    hints = _hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        hint = hints[f.name]
        origin = typing.get_origin(hint)
        args = typing.get_args(hint)
        if dataclasses.is_dataclass(hint) and isinstance(value, dict):
            value = from_dict(hint, value)
        elif origin is list and args and dataclasses.is_dataclass(args[0]):
            value = [from_dict(args[0], v) for v in value or []]
        elif origin is dict and len(args) == 2 and args[1] is Column:
            value = {k: from_dict(Column, v) for k, v in (value or {}).items()}
        kwargs[f.name] = value
    return cls(**kwargs)


def graph_to_json_dict(graph: Graph) -> dict:
    return {"format": FORMAT, "version": VERSION, "graph": to_dict(graph)}


def graph_from_json_dict(data: dict) -> Graph:
    if data.get("format") != FORMAT:
        raise ValueError("Not an RS-StatChem graph file.")
    if int(data.get("version", 1)) > VERSION:
        raise ValueError("This graph was saved by a newer version of RS-StatChem.")
    return from_dict(Graph, data["graph"])


def _step(obj, part: str):
    if isinstance(obj, dict):
        return obj[part]
    if isinstance(obj, list):
        return obj[int(part)]
    return getattr(obj, part)


def get_path(obj, path: str):
    """Value at a dotted path such as ``line.color``, ``options.bins`` or ``rect.0``."""
    for part in path.split("."):
        obj = _step(obj, part)
    return obj


def set_path(obj, path: str, value) -> None:
    parts = path.split(".")
    for part in parts[:-1]:
        obj = _step(obj, part)
    if isinstance(obj, dict):
        obj[parts[-1]] = value
    elif isinstance(obj, list):
        obj[int(parts[-1])] = value
    else:
        setattr(obj, parts[-1], value)

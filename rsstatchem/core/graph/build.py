"""Build a graph from a Dataset — the core of Origin's "Plot Setup" dialog.

:class:`PlotSpec` names the plot template, the X / Y / Z / error / label variables and an
optional group-by variable that splits the rows into series. :func:`build_graph` snapshots the
needed columns (user-missing values honoured through ``common.numeric``) into the graph, so
the graph no longer depends on the dataset. Optional summaries (mean ± SD/SE/95% CI, median,
sum, count by X category) mirror Origin's "Grouped Column / statistics" plots.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from rsstatchem.core.dataset import NUMERIC, STRING, Dataset
from rsstatchem.core.graph.model import Axis, Column, Graph, Layer, Plot
from rsstatchem.core.graph.themes import DEFAULT_THEME, apply_theme
from rsstatchem.core.stats.common import numeric, value_text

ROLES = ("x", "y", "z", "xerr", "yerr", "labels", "group")
ROLE_LABELS = {"x": "X:", "y": "Y:", "z": "Z:", "xerr": "X Error:", "yerr": "Y Error:",
               "labels": "Labels:", "group": "Group by (series):"}
XY = ("x", "y", "xerr", "yerr", "labels", "group")


@dataclass(frozen=True)
class Template:
    key: str
    label: str
    category: str
    layer: str
    plot: str
    roles: tuple = XY
    required: tuple = ("y",)
    stack: str = "none"
    summary: bool = False
    role_text: dict = field(default_factory=dict)
    hint: str = ""
    options: dict = field(default_factory=dict)   # plot options the template presets


_T = [
    Template("line", "Line", "Basic 2D", "cartesian", "line", summary=True),
    Template("scatter", "Scatter", "Basic 2D", "cartesian", "scatter", summary=True),
    Template("line_symbol", "Line + Symbol", "Basic 2D", "cartesian", "line_symbol",
             summary=True),
    Template("errorbar", "Scatter with Error Bars", "Basic 2D", "cartesian", "scatter",
             summary=True, hint="Error bars come from the X/Y Error columns or a summary "
                                "(mean ± SD, SE or 95% CI)."),
    Template("column", "Column (grouped)", "Column / Bar", "cartesian", "column",
             ("x", "y", "yerr", "labels", "group"), summary=True),
    Template("column_stacked", "Stacked Column", "Column / Bar", "cartesian", "column",
             ("x", "y", "labels", "group"), stack="stacked", summary=True),
    Template("column_percent", "100% Stacked Column", "Column / Bar", "cartesian", "column",
             ("x", "y", "group"), stack="percent", summary=True),
    Template("bar", "Bar (grouped)", "Column / Bar", "cartesian", "bar",
             ("x", "y", "yerr", "labels", "group"), summary=True),
    Template("bar_stacked", "Stacked Bar", "Column / Bar", "cartesian", "bar",
             ("x", "y", "group"), stack="stacked", summary=True),
    Template("bar_percent", "100% Stacked Bar", "Column / Bar", "cartesian", "bar",
             ("x", "y", "group"), stack="percent", summary=True),
    Template("area", "Area", "Area", "cartesian", "area", ("x", "y", "group")),
    Template("area_stacked", "Stacked Area", "Area", "cartesian", "area", ("x", "y", "group"),
             stack="stacked"),
    Template("area_percent", "100% Stacked Area", "Area", "cartesian", "area",
             ("x", "y", "group"), stack="percent"),
    Template("pie", "Pie", "Pie", "pie", "pie", ("x", "y", "labels"), required=(),
             role_text={"x": "Categories:", "y": "Values (optional):"},
             hint="Categories only → slice per category count; with Values → sum per "
                  "category; Values only → one slice per row."),
    Template("doughnut", "Doughnut", "Pie", "pie", "doughnut", ("x", "y", "labels"),
             required=(), role_text={"x": "Categories:", "y": "Values (optional):"}),
    Template("histogram", "Histogram (+ normal curve)", "Statistics", "cartesian",
             "histogram", ("y", "group"), role_text={"y": "Variable(s):"}),
    Template("box", "Box Chart", "Statistics", "cartesian", "box", ("x", "y", "group"),
             role_text={"x": "Category axis (optional):", "y": "Variable(s):"}),
    Template("bubble", "Bubble", "Bubble / Color Map", "cartesian", "bubble",
             ("x", "y", "z", "labels", "group"), ("y", "z"), role_text={"z": "Size:"}),
    Template("color_scatter", "Color-Mapped Scatter", "Bubble / Color Map", "cartesian",
             "color_scatter", ("x", "y", "z", "labels"), ("y", "z"),
             role_text={"z": "Color by:"}),
    Template("bubble_color", "Bubble + Color Mapped", "Bubble / Color Map", "cartesian",
             "bubble", ("x", "y", "z", "labels"), ("y", "z"),
             role_text={"z": "Size && color:"}),
    Template("double_y", "Double-Y Axis", "Multi-Axis / Multi-Panel", "cartesian",
             "line_symbol", ("x", "y", "yerr", "group"),
             hint="First Y on the left axis, the others on the right axis."),
    Template("multi_panel", "Multi-Panel (one panel per Y)", "Multi-Axis / Multi-Panel",
             "cartesian", "line_symbol", ("x", "y", "group")),
    Template("waterfall", "Waterfall / Offset Y", "Multi-Axis / Multi-Panel", "cartesian",
             "line", ("x", "y")),
    Template("contour", "Contour (XYZ)", "Contour / 3D", "cartesian", "contour",
             ("x", "y", "z"), ("x", "y", "z")),
    Template("heatmap", "Heatmap (XYZ)", "Contour / 3D", "cartesian", "heatmap",
             ("x", "y", "z"), ("x", "y", "z")),
    Template("surface3d", "3D Surface", "Contour / 3D", "3d", "surface3d", ("x", "y", "z"),
             ("x", "y", "z")),
    Template("scatter3d", "3D Scatter", "Contour / 3D", "3d", "scatter3d",
             ("x", "y", "z", "group"), ("x", "y", "z")),
    Template("polar", "Polar (θ, r)", "Specialized", "polar", "line",
             ("x", "y", "group"), ("x", "y"),
             role_text={"x": "Theta (degrees):", "y": "Radius:"}),
    Template("ternary", "Ternary", "Specialized", "ternary", "scatter",
             ("x", "y", "z", "labels", "group"), ("x", "y", "z"),
             role_text={"x": "Component A:", "y": "Component B:", "z": "Component C:"}),
]
_Y = {"y": "Variable(s):"}
_T += [
    # ---- 2D line/scatter variants
    Template("step", "Step (horizontal)", "Basic 2D", "cartesian", "line",
             options={"connect": "step_h"}),
    Template("spline", "Spline", "Basic 2D", "cartesian", "line",
             options={"connect": "spline"}),
    Template("lollipop", "Lollipop", "Basic 2D", "cartesian", "lollipop",
             ("x", "y", "labels", "group")),
    Template("error_band", "Error Band", "Basic 2D", "cartesian", "line",
             ("x", "y", "yerr", "group"), options={"error_band": True},
             hint="The Y Error column is drawn as a shaded band around the line."),
    Template("fill_area", "Fill Area (between curves)", "Area", "cartesian", "line",
             ("x", "y"), hint="Each curve is filled to the next one; the last to zero."),
    Template("floating_column", "Floating Column", "Column / Bar", "cartesian",
             "floating_column", ("x", "y", "z"), ("y", "z"),
             role_text={"y": "Bottom:", "z": "Top:"}),
    # ---- statistical
    Template("violin", "Violin Plot", "Statistics", "cartesian", "violin",
             ("x", "y", "group"), role_text=dict(_Y, x="Category axis (optional):")),
    Template("box_half", "Half Box + Data Points", "Statistics", "cartesian", "box",
             ("x", "y", "group"), role_text=dict(_Y, x="Category axis (optional):"),
             options={"style": "half_points"}),
    Template("meanbar", "Mean Bar (± SD)", "Statistics", "cartesian", "meanbar",
             ("x", "y", "group"), role_text=_Y, options={"stat": "sd"}),
    Template("interval", "Interval Plot (mean ± 95% CI)", "Statistics", "cartesian",
             "meanbar", ("x", "y", "group"), role_text=_Y,
             options={"stat": "ci", "style": "point"}),
    Template("hist_kde", "Histogram + Kernel Density", "Statistics", "cartesian",
             "histogram", ("y", "group"), role_text=_Y,
             options={"dist_curve": "kde", "scale": "density"}),
    Template("hist_rug", "Histogram + Rug", "Statistics", "cartesian", "histogram",
             ("y", "group"), role_text=_Y, options={"rug": True}),
    Template("kde", "Kernel Density", "Statistics", "cartesian", "kde", ("y", "group"),
             role_text=_Y),
    Template("ecdf", "Cumulative Distribution (ECDF)", "Statistics", "cartesian", "ecdf",
             ("y", "group"), role_text=_Y),
    Template("qq", "Q-Q Plot (normal)", "Statistics", "cartesian", "qq", ("y", "group"),
             role_text=_Y),
    Template("probability", "Probability Plot (normal)", "Statistics", "cartesian",
             "probability", ("y", "group"), role_text=_Y),
    Template("kde2d", "2D Kernel Density", "Statistics", "cartesian", "kde2d", ("x", "y"),
             ("x", "y")),
    Template("pareto", "Pareto Chart", "Statistics", "cartesian", "pareto", ("x", "y"),
             ("x",), role_text={"x": "Categories:", "y": "Values (optional):"},
             hint="Categories only → counts per category; with Values → sum per category."),
    Template("bland_altman", "Bland-Altman Plot", "Statistics", "cartesian",
             "bland_altman", ("x", "y"), ("x", "y"),
             role_text={"x": "Method A:", "y": "Method B:"}),
    Template("control_xbar", "QC Chart (X-bar R)", "Statistics", "cartesian", "control",
             ("y",), role_text={"y": "Data:"}, options={"size": 5}),
    Template("control_imr", "QC Chart (I-MR)", "Statistics", "cartesian", "control",
             ("y",), role_text={"y": "Data:"}),
    Template("scatter_matrix", "Scatter Matrix", "Statistics", "cartesian", "scatter",
             ("x", "y"), role_text={"y": "Variables:"}),
    Template("marginal", "Scatter + Marginal Histograms", "Statistics", "cartesian",
             "scatter", ("x", "y"), ("x", "y")),
    # ---- 3D and specialised
    Template("bar3d", "3D Bars", "Contour / 3D", "3d", "bar3d", ("x", "y", "z"),
             ("x", "y", "z")),
    Template("line3d", "3D Line / Trajectory", "Contour / 3D", "3d", "line3d",
             ("x", "y", "z"), ("x", "y", "z")),
    Template("waterfall3d", "3D Waterfall", "Contour / 3D", "3d", "waterfall3d", ("x", "y")),
    Template("vector", "Vector XYAM", "Specialized", "cartesian", "vector", ("x", "y", "z"),
             ("x", "y", "z"), role_text={"z": "Angle (°):"}),
    Template("radar", "Radar / Spider", "Specialized", "polar", "radar", ("x", "y"),
             role_text={"x": "Categories:"}),
    Template("windrose", "Wind Rose", "Specialized", "polar", "windrose", ("x", "y"),
             ("x", "y"), role_text={"x": "Direction (°):", "y": "Speed:"}),
]
TEMPLATES: dict[str, Template] = {t.key: t for t in _T}
SUMMARIES = {"none": "None (plot the data rows)", "mean": "Mean", "mean_sd": "Mean ± SD",
             "mean_se": "Mean ± SE", "mean_ci": "Mean ± 95% CI", "median": "Median",
             "sum": "Sum", "count": "Count (N)"}
SINGLE_Y = ("contour", "heatmap", "surface3d", "scatter3d", "ternary", "bubble",
            "color_scatter", "bubble_color", "kde2d", "pareto", "bland_altman", "bar3d",
            "line3d", "vector", "windrose", "floating_column", "marginal")


@dataclass
class PlotSpec:
    kind: str = "line"
    x: str | None = None
    ys: list[str] = field(default_factory=list)
    z: str | None = None
    xerr: str | None = None
    yerr: str | None = None
    labels: str | None = None
    group: str | None = None
    summary: str = "none"
    theme: str = DEFAULT_THEME
    title: str = ""
    options: dict = field(default_factory=dict)

    def validate(self, ds: Dataset | None = None) -> str | None:
        t = TEMPLATES.get(self.kind)
        if t is None:
            return f"Unknown plot type {self.kind!r}."
        values = {"x": self.x, "y": self.ys, "z": self.z, "xerr": self.xerr,
                  "yerr": self.yerr, "labels": self.labels, "group": self.group}
        for role in t.required:
            if not values[role]:
                if role == "y" and self.summary == "count" and self.x:
                    continue
                text = t.role_text.get(role, ROLE_LABELS[role]).replace("&&", "&")
                return f"{t.label} needs a variable in “{text.rstrip(':')}”."
        if self.kind == "pie" or self.kind == "doughnut":
            if not self.x and not self.ys:
                return "Pie charts need Categories and/or Values."
        if self.kind == "double_y" and len(self.ys) < 2:
            return "A double-Y graph needs at least two Y variables."
        if ds is not None:
            for role in ("y", "z", "xerr", "yerr"):
                names = values[role] if role == "y" else [values[role]]
                for name in names or []:
                    if name and ds.var(name).type != NUMERIC:
                        return f"{name} must be numeric."
        return None


# ====================================================================== snapshot helpers
class _Snap:
    def __init__(self, ds: Dataset, graph: Graph):
        self.ds, self.g = ds, graph

    def var(self, name):
        return self.ds.var(name)

    def title(self, name) -> str:
        return self.var(name).title if name else ""

    def labels_of(self, name) -> dict:
        var = self.var(name)
        if var.type == STRING:
            return self.codes(name)[1]
        return dict(var.value_labels)

    def codes(self, name) -> tuple[np.ndarray, dict]:
        """Numeric codes for a variable; strings become 1..k with value labels."""
        var = self.var(name)
        if var.type == STRING:
            s = self.ds.analysis_series(name)
            cats = sorted({str(v) for v in s.dropna()})
            index = {c: float(i + 1) for i, c in enumerate(cats)}
            codes = np.array([index.get(str(v), np.nan) if pd.notna(v) else np.nan
                              for v in s], dtype=float)
            return codes, {v: k for k, v in index.items()}
        if var.type == NUMERIC:
            return numeric(self.ds, name).to_numpy(dtype=float), dict(var.value_labels)
        s = self.ds.analysis_series(name)   # dates → days since epoch
        return (pd.to_datetime(s).astype("int64") / 86400e9).to_numpy(dtype=float), {}

    def num(self, name: str | None, mask=None, tag: str = "") -> str | None:
        if not name:
            return None
        values, labels = self.codes(name)
        var = self.var(name)
        if mask is not None:
            values = values[mask]
        keep_labels = labels if (var.type == STRING or var.measure != "scale") else {}
        col = Column.from_array(var.name, values, var.label, keep_labels, kind="num")
        return self.g.add_column(col, f"{var.name} [{tag}]" if tag else var.name)

    def text(self, name: str | None, mask=None, tag: str = "") -> str | None:
        if not name:
            return None
        var = self.var(name)
        s = self.ds.analysis_series(name)
        vals = ["" if pd.isna(v) else value_text(var, v) for v in s]
        if mask is not None:
            vals = [v for v, k in zip(vals, mask) if k]
        col = Column(var.name, var.label, "text", vals)
        return self.g.add_column(col, f"{var.name} (labels) [{tag}]" if tag
                                 else f"{var.name} (labels)")

    def derived(self, key: str, values, label: str = "", value_labels=None) -> str:
        col = Column.from_array(key, np.asarray(values, dtype=float), label, value_labels,
                                kind="num")
        return self.g.add_column(col, key)

    def groups(self, name: str | None) -> list[tuple[str, np.ndarray | None, str]]:
        """(label, row mask, tag) per group; a single all-rows entry without grouping."""
        if not name:
            return [("", None, "")]
        codes, labels = self.codes(name)
        out = []
        for code in sorted(set(codes[np.isfinite(codes)])):
            label = labels.get(code)
            if label is None:
                label = f"{code:g}"
            out.append((str(label), codes == code, f"{self.var(name).name}={label}"))
        return out


def _stat(values: np.ndarray, how: str) -> tuple[float, float]:
    v = values[np.isfinite(values)]
    n = len(v)
    if how == "count":
        return float(n), np.nan
    if n == 0:
        return np.nan, np.nan
    if how == "sum":
        return float(v.sum()), np.nan
    if how == "median":
        return float(np.median(v)), np.nan
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else np.nan
    if how == "mean_sd":
        return mean, sd
    if how == "mean_se":
        return mean, sd / np.sqrt(n) if n > 1 else np.nan
    if how == "mean_ci":
        return mean, stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
    return mean, np.nan


_SUFFIX = {"mean": "mean", "mean_sd": "mean ± SD", "mean_se": "mean ± SE",
           "mean_ci": "mean ± 95% CI", "median": "median", "sum": "sum", "count": "N"}


def _style(plot: Plot, kind: str) -> None:
    t = plot.type
    if t in ("line", "line_symbol") or kind == "polar":
        plot.line.visible = True
    if t in ("scatter", "line_symbol", "scatter3d") or kind == "ternary":
        plot.symbol.visible = True
    if t == "scatter" or kind == "ternary":
        plot.line.visible = False
    if t in ("column", "bar", "histogram", "box", "pie", "doughnut"):
        plot.line.visible = t == "histogram"
    if t == "area":
        plot.fill.alpha = 0.85
    if t == "bubble":
        plot.symbol.shape = "o"
        plot.fill.alpha = 0.75
    if t == "color_scatter":
        plot.symbol.shape = "o"
    if t == "surface3d":
        plot.line.visible = True
        plot.line.width = 0.4


def build_graph(ds: Dataset, spec: PlotSpec, title: str | None = None) -> Graph:
    error = spec.validate(ds)
    if error:
        raise ValueError(error)
    t = TEMPLATES[spec.kind]
    graph = Graph(title=title or spec.title or t.label)
    snap = _Snap(ds, graph)
    layer = Layer(kind=t.layer, stack=t.stack)
    graph.layers.append(layer)
    ys = list(spec.ys)
    if spec.kind in SINGLE_Y:
        ys = ys[:1]
    builder = {"pie": _build_pie, "doughnut": _build_pie, "histogram": _build_hist,
               "box": _build_box, "multi_panel": _build_multi}.get(spec.kind, _build_xy)
    from rsstatchem.core.graph import stat_build
    builder = stat_build.BUILDERS.get(spec.kind, builder)
    builder(snap, spec, t, graph, layer, ys)
    for lay in graph.layers:
        for plot in lay.plots:
            _style(plot, lay.kind)
        lay.legend.visible = len(lay.plots) > 1 and lay.kind != "pie" and \
            not (spec.kind in ("box", "violin", "box_half", "meanbar", "interval") and
                 not any("group_label" in p.options for p in lay.plots))
        if lay.kind == "ternary":
            for axis in (lay.x, lay.y, lay.z):
                axis.grid_major = True
                axis.major_step = 0.2
        if lay.kind == "polar":
            lay.x.grid_major = lay.y.grid_major = True
    apply_theme(graph, spec.theme)
    for lay in graph.layers:   # themes reset grids; specialised layers keep theirs
        if lay.kind in ("ternary", "polar"):
            for axis in (lay.x, lay.y, lay.z):
                axis.grid_major = True
    if spec.kind == "waterfall":
        _auto_offset(graph, graph.layers[0])
    if spec.title:
        graph.layers[0].title = spec.title
    graph.meta["spec"] = {"kind": spec.kind, "x": spec.x, "ys": spec.ys, "z": spec.z,
                          "group": spec.group, "summary": spec.summary,
                          "dataset": getattr(ds, "name", "")}
    return graph


def _series_name(snap: _Snap, y: str, ys: list[str], glabel: str) -> str:
    if glabel and len(ys) <= 1:
        return glabel
    if glabel:
        return f"{snap.title(y)} ({glabel})"
    return snap.title(y)


def _plot(t: Template, spec: PlotSpec, **kw) -> Plot:
    p = Plot(type=t.plot, **kw)
    p.options.update(t.options)
    for key, value in spec.options.items():
        if key in ("bins", "bin_width", "normal_curve", "scale", "show_values", "levels",
                   "grid", "connect", "show_mean", "outliers", "points", "notch", "cmap",
                   "wireframe", "hole", "show_percent", "show_labels", "dist_curve", "rug",
                   "stat", "style", "size", "chart", "bw", "positions", "dist", "inner",
                   "side", "fill_to", "error_band", "data_labels"):
            p.options[key] = value
    return p


def _build_xy(snap: _Snap, spec: PlotSpec, t: Template, graph: Graph, layer: Layer, ys):
    summary = spec.summary if t.summary else "none"
    groups = snap.groups(spec.group) if "group" in t.roles else [("", None, "")]
    if summary != "none" and not spec.x:
        _build_summary_by_variable(snap, spec, t, layer, ys, groups, summary)
        return
    if summary == "count" and not ys:
        ys = [None]
    for glabel, mask, tag in groups:
        for k, y in enumerate(ys):
            name = _series_name(snap, y, ys, glabel) if y else (glabel or "N")
            if summary != "none":
                p = _summary_plot(snap, spec, t, y, mask, tag, summary, name)
            else:
                p = _plot(t, spec, name=name, x=snap.num(spec.x, mask, tag),
                          y=snap.num(y, mask, tag), z=snap.num(spec.z, mask, tag),
                          xerr=snap.num(spec.xerr, mask, tag),
                          yerr=snap.num(spec.yerr, mask, tag),
                          labels=snap.text(spec.labels, mask, tag))
            if spec.kind == "double_y" and k > 0:
                p.y_axis = "right"
            if spec.kind == "bubble_color":
                p.options["color_col"] = p.z
            if t.plot in ("line", "line_symbol") and spec.x and summary == "none":
                p.options.setdefault("sort_x", spec.options.get("sort_x", True))
            layer.plots.append(p)
    first = ys[0] if ys else None
    layer.x.title = snap.title(spec.x) if spec.x else ("Row" if t.layer == "cartesian"
                                                       else "")
    if t.layer in ("3d", "ternary"):
        layer.x.title, layer.y.title, layer.z.title = (snap.title(spec.x), snap.title(first),
                                                       snap.title(spec.z))
    elif spec.kind == "double_y":
        layer.y.title, layer.y2.title = snap.title(ys[0]), ", ".join(
            snap.title(y) for y in ys[1:])
        layer.show_y2 = True
    else:
        if first is None:
            layer.y.title = "Count"
        elif len(ys) == 1:
            layer.y.title = snap.title(first)
        elif len(ys) == 2 and summary == "none":
            layer.y.title = ", ".join(snap.title(y) for y in ys)
        if summary != "none" and first is not None:
            if len(ys) == 1:
                layer.y.title += f" ({_SUFFIX[summary]})"
            else:
                layer.y.title = _SUFFIX[summary][0].upper() + _SUFFIX[summary][1:]
        if t.stack == "percent":
            layer.y.title = "Percent (%)"
    if t.plot == "bar":
        layer.x.title, layer.y.title = layer.y.title, layer.x.title
    if t.layer == "polar":
        layer.y.title = ""


def _summary_plot(snap: _Snap, spec: PlotSpec, t: Template, y, mask, tag, summary, name):
    xcodes, xlabels = snap.codes(spec.x)
    yv = snap.codes(y)[0] if y else np.ones(len(xcodes))
    if mask is not None:
        xcodes, yv = xcodes[mask], yv[mask]
    cats = sorted(set(xcodes[np.isfinite(xcodes)]))
    centers, errs = [], []
    for c in cats:
        center, err = _stat(yv[xcodes == c], summary)
        centers.append(center)
        errs.append(err)
    base = f"{_SUFFIX[summary]}({y or 'N'})" + (f" [{tag}]" if tag else "")
    var = snap.var(spec.x)
    labels = xlabels if (var.type == STRING or var.measure != "scale") else {}
    xkey = snap.derived(f"{var.name} categories" + (f" [{tag}]" if tag else ""), cats,
                        var.label, labels)
    ykey = snap.derived(base, centers, f"{snap.title(y) if y else 'N'} ({_SUFFIX[summary]})")
    ekey = None
    if summary in ("mean_sd", "mean_se", "mean_ci"):
        ekey = snap.derived(f"err {base}", errs, f"{_SUFFIX[summary]} error")
    return _plot(t, spec, name=name, x=xkey, y=ykey, yerr=ekey)


def _build_summary_by_variable(snap, spec, t, layer, ys, groups, summary):
    """No X: one category per Y variable (e.g. mean ± SD of pretest vs posttest)."""
    labels = {float(i + 1): snap.title(y) for i, y in enumerate(ys)}
    for glabel, mask, tag in groups:
        centers, errs = [], []
        for y in ys:
            v = snap.codes(y)[0]
            if mask is not None:
                v = v[mask]
            c, e = _stat(v, summary)
            centers.append(c)
            errs.append(e)
        sfx = f" [{tag}]" if tag else ""
        xkey = snap.derived("Variable" + sfx, list(labels), "Variable", labels)
        ykey = snap.derived(f"{_SUFFIX[summary]}{sfx}", centers, _SUFFIX[summary])
        ekey = snap.derived(f"err {_SUFFIX[summary]}{sfx}", errs) \
            if summary in ("mean_sd", "mean_se", "mean_ci") else None
        layer.plots.append(_plot(t, spec, name=glabel or _SUFFIX[summary].capitalize(),
                                 x=xkey, y=ykey, yerr=ekey))
    layer.x.title = ""
    layer.y.title = _SUFFIX[summary].replace("mean", "Mean").replace("median", "Median") \
        .replace("sum", "Sum")
    if t.plot == "bar":
        layer.x.title, layer.y.title = layer.y.title, layer.x.title


def _build_pie(snap: _Snap, spec: PlotSpec, t: Template, graph, layer, ys):
    y = ys[0] if ys else None
    if spec.x:
        codes, labels = snap.codes(spec.x)
        yv = snap.codes(y)[0] if y else np.ones(len(codes))
        cats = sorted(set(codes[np.isfinite(codes)]))
        how = "sum" if y else "count"
        if spec.summary in ("mean", "median", "sum", "count") and y:
            how = spec.summary
        vals = [_stat(yv[codes == c], how)[0] for c in cats]
        var = snap.var(spec.x)
        xkey = snap.derived(f"{var.name} categories", cats, var.label,
                            {c: labels.get(c, f"{c:g}") for c in cats})
        ykey = snap.derived(f"{how}({y or 'N'})", vals, snap.title(y) if y else "Count")
        p = _plot(t, spec, name=snap.title(y) if y else snap.title(spec.x), x=xkey, y=ykey)
    else:
        p = _plot(t, spec, name=snap.title(y), y=snap.num(y), labels=snap.text(spec.labels))
    layer.plots.append(p)
    layer.x.label_font.size = 9.0


def _build_hist(snap: _Snap, spec: PlotSpec, t: Template, graph, layer, ys):
    for glabel, mask, tag in snap.groups(spec.group):
        for y in ys:
            layer.plots.append(_plot(t, spec, name=_series_name(snap, y, ys, glabel),
                                     y=snap.num(y, mask, tag)))
    layer.x.title = snap.title(ys[0]) if len(ys) == 1 else "Value"
    layer.y.title = {"percent": "Percent", "density": "Density"}.get(
        spec.options.get("scale", "count"), "Count")


def _build_box(snap: _Snap, spec: PlotSpec, t: Template, graph, layer, ys):
    group = spec.group or spec.x
    groups = snap.groups(group)
    grouped = bool(group) and len(ys) > 1
    width = 0.8 / max(1, len(ys))
    for g, (glabel, mask, tag) in enumerate(groups):
        for k, y in enumerate(ys):
            p = _plot(t, spec, name=snap.title(y) if grouped else
                      _series_name(snap, y, ys, glabel), y=snap.num(y, mask, tag))
            if grouped:
                p.options.update(position=g + 1 + (k - (len(ys) - 1) / 2) * width,
                                 width=width * 0.8, group_pos=g + 1, group_label=glabel,
                                 color_index=k)
                if g > 0:
                    layer.legend.hidden.append(p.id)
            layer.plots.append(p)
    layer.y.title = snap.title(ys[0]) if len(ys) == 1 else ""
    layer.x.title = snap.title(group) if group else ""
    layer.x.minor_count = 0


def _build_multi(snap: _Snap, spec: PlotSpec, t: Template, graph, layer, ys):
    n = len(ys)
    cols = 1 if n <= 3 else 2
    rows = int(np.ceil(n / cols))
    graph.rows, graph.cols = rows, cols
    graph.layers.clear()
    for k, y in enumerate(ys):
        lay = Layer(kind="cartesian", grid=[k // cols, k % cols, 1, 1])
        sub = PlotSpec(kind="line_symbol", x=spec.x, ys=[y], group=spec.group,
                       options=spec.options)
        _build_xy(snap, sub, TEMPLATES["line_symbol"], graph, lay, [y])
        lay.panel_label = f"({chr(97 + k)})" if n > 1 else ""
        graph.layers.append(lay)


def _auto_offset(graph: Graph, layer: Layer) -> None:
    spans = []
    for p in layer.plots:
        v = graph.values(p.y)
        if v is not None and np.isfinite(v).any():
            spans.append(np.nanmax(v) - np.nanmin(v))
    if len(spans) > 1:
        layer.offset_y = float(f"{0.5 * max(spans):.2g}")
        layer.legend.visible = True


def axis_titles(layer: Layer) -> dict[str, Axis]:
    return {"x": layer.x, "y": layer.y, "y2": layer.y2, "z": layer.z}

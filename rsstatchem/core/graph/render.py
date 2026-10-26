"""Render a :class:`~rsstatchem.core.graph.model.Graph` onto a matplotlib ``Figure``.

Object-oriented API only (no pyplot), so the same code draws on the Qt canvas of the graph
window and on an Agg canvas for export. :func:`render` returns a :class:`RenderMap` linking
matplotlib artists back to model objects, which the graph window uses for picking
(double-click an axis, plot or legend to edit it) and for the data reader.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from matplotlib import ticker
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
from scipy import stats

from rsstatchem.core.graph import ternary
from rsstatchem.core.graph.gridding import xyz_grid
from rsstatchem.core.graph.model import Axis, Font, Graph, Layer, Plot
from rsstatchem.core.graph.text import resolve_family, rich
from rsstatchem.core.graph.themes import DASHES, HATCHES, PALETTES, SYMBOLS

CM = 1 / 2.54
LINEISH = ("line", "scatter", "line_symbol")
FILLED = ("column", "bar", "area", "histogram", "box", "pie", "doughnut")


@dataclass
class RenderMap:
    """Artists drawn for each model object (for picking) and data for the data reader."""
    axes: dict[int, dict] = field(default_factory=dict)       # layer index -> {"ax", "ax2"}
    artists: list[tuple[object, tuple]] = field(default_factory=list)
    points: list[tuple] = field(default_factory=list)          # (li, plot_id, ax, x, y)
    errors: list[str] = field(default_factory=list)

    def add(self, artist, ref: tuple) -> None:
        if artist is None:
            return
        if isinstance(artist, (list, tuple)):
            for a in artist:
                self.add(a, ref)
            return
        self.artists.append((artist, ref))


class _Ctx:
    def __init__(self, graph: Graph, fig, rmap: RenderMap):
        self.g, self.fig, self.map = graph, fig, rmap

    # ------------------------------------------------------------------ styles
    def font(self, font: Font, size_scale: float = 1.0) -> dict:
        return {"fontfamily": resolve_family(font.family or self.g.font_family),
                "fontsize": font.size * size_scale,
                "fontweight": "bold" if font.bold else "normal",
                "fontstyle": "italic" if font.italic else "normal",
                "color": font.color, "math_fontfamily": self.g.math_font}

    def fontprops(self, font: Font) -> FontProperties:
        return FontProperties(family=resolve_family(font.family or self.g.font_family),
                              size=font.size, weight="bold" if font.bold else "normal",
                              style="italic" if font.italic else "normal",
                              math_fontfamily=self.g.math_font)

    def color(self, spec: str, i: int, fill: bool = False) -> str:
        if spec and spec != "auto":
            return spec
        pal = PALETTES.get(self.g.fill_palette if fill else self.g.palette) or \
            PALETTES["Origin Classic"]
        return pal[i % len(pal)]

    def line_color(self, p: Plot, i: int) -> str:
        return self.color(p.line.color, i, fill=p.type in FILLED)

    def fill_color(self, p: Plot, i: int) -> str:
        return self.color(p.fill.color, i, fill=True)

    def dash(self, p: Plot, i: int):
        if p.line.style == "auto":
            return DASHES[i % len(DASHES)] if self.g.dash_cycle else "-"
        return p.line.style

    def marker(self, p: Plot, i: int) -> str:
        return SYMBOLS[i % len(SYMBOLS)] if p.symbol.shape == "auto" else p.symbol.shape

    def hatch(self, p: Plot, i: int) -> str | None:
        h = HATCHES[i % len(HATCHES)] if p.fill.hatch == "auto" else p.fill.hatch
        return h or None

    def symbol_kw(self, p: Plot, i: int) -> dict:
        col = self.line_color(p, i)
        face = (p.symbol.face if p.symbol.face != "auto" else col) if p.symbol.filled \
            else "none"
        edge = p.symbol.edge if p.symbol.edge != "auto" else col
        return {"marker": self.marker(p, i), "markersize": p.symbol.size,
                "markerfacecolor": face, "markeredgecolor": edge,
                "markeredgewidth": p.symbol.edge_width}

    # ------------------------------------------------------------------ data
    def arr(self, key):
        return self.g.values(key)

    def col(self, key):
        return self.g.column(key)


# ====================================================================== entry point
def render(graph: Graph, fig) -> RenderMap:
    rmap = RenderMap()
    ctx = _Ctx(graph, fig, rmap)
    fig.clear()
    fig.set_size_inches(graph.page.width_cm * CM, graph.page.height_cm * CM, forward=False)
    fig.set_facecolor(graph.page.background)
    gridded = [layer for layer in graph.layers if not layer.rect]
    if gridded:
        fig.set_layout_engine("constrained")
        fig.get_layout_engine().set(w_pad=0.05, h_pad=0.05, wspace=0.04, hspace=0.04)
    else:
        fig.set_layout_engine(None)
    rows = max([graph.rows] + [l.grid[0] + l.grid[2] for l in gridded]) if gridded else 1
    cols = max([graph.cols] + [l.grid[1] + l.grid[3] for l in gridded]) if gridded else 1
    gs = fig.add_gridspec(rows, cols)
    for li, layer in enumerate(graph.layers):
        try:
            _render_layer(ctx, gs, li, layer)
        except Exception as exc:  # noqa: BLE001 - one bad layer must not kill the graph
            rmap.errors.append(f"Layer {li + 1}: {exc}")
    return rmap


def _render_layer(ctx: _Ctx, gs, li: int, layer: Layer) -> None:
    proj = {"polar": "polar", "3d": "3d"}.get(layer.kind)
    if layer.rect:
        ax = ctx.fig.add_axes(layer.rect, projection=proj)
    else:
        r, c, rs, cs = layer.grid
        ax = ctx.fig.add_subplot(gs[r:r + rs, c:c + cs], projection=proj)
    if layer.kind != "3d":
        ax.set_facecolor(layer.background)
    ctx.map.axes[li] = {"ax": ax, "ax2": None, "layer": layer}
    ctx.map.add(ax.patch, ("layer", li))
    draw = {"cartesian": _draw_cartesian, "polar": _draw_polar, "3d": _draw_3d,
            "ternary": _draw_ternary, "pie": _draw_pie}[layer.kind]
    handles = draw(ctx, li, layer, ax)
    if layer.title:
        t = ax.set_title(rich(layer.title), **ctx.font(layer.title_font))
        ctx.map.add(t, ("layer", li))
    if layer.panel_label:
        t = ax.text(0.0, 1.02, rich(layer.panel_label), transform=ax.transAxes,
                    ha="right", va="bottom", **ctx.font(layer.title_font))
        ctx.map.add(t, ("layer", li))
    _legend(ctx, li, layer, ax, handles or [])
    _annotations(ctx, li, layer, ax)


# ====================================================================== cartesian
def _safe(ctx: _Ctx, li: int, plot: Plot, fn, *args):
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001
        ctx.map.errors.append(f"{plot.name or plot.type}: {exc}")
        return None


def _xy(ctx: _Ctx, p: Plot):
    y = ctx.arr(p.y)
    if y is None:
        raise ValueError("no Y column")
    x = ctx.arr(p.x)
    if x is None or x.dtype == object:
        x = np.arange(1, len(y) + 1, dtype=float)
    n = min(len(x), len(y))
    return x[:n], y[:n]


def _categories(ctx: _Ctx, plots: list[Plot]):
    """Category positions for column/bar plots: (code -> position) or None if numeric X."""
    codes, labels, categorical = [], {}, False
    for p in plots:
        col = ctx.col(p.x)
        if col is None or col.categorical or col.kind == "text":
            categorical = True
        if col is not None and col.kind == "text":
            labels.update({k + 1.0: str(v) for k, v in enumerate(col.values)})
        elif col is not None:
            labels.update(col.value_labels)
        x, _ = _xy(ctx, p)
        codes.extend(float(v) for v in x if np.isfinite(v))
    uniq = sorted(set(codes))
    if categorical:
        pos = {c: float(i) for i, c in enumerate(uniq)}
        ticks = ([pos[c] for c in uniq], [labels.get(c, _num(c)) for c in uniq])
        return pos, 1.0, ticks
    spacing = float(np.min(np.diff(uniq))) if len(uniq) > 1 else 1.0
    return None, spacing, None


def _num(v) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def _err_arrays(ctx: _Ctx, p: Plot, key, n: int, mask, order=None):
    e = ctx.arr(key)
    if e is None:
        return None
    e = np.abs(e[:n])
    if order is not None:
        e = e[order]
    e = e[mask]
    if p.error.direction == "plus":
        return np.vstack([np.zeros_like(e), e])
    if p.error.direction == "minus":
        return np.vstack([e, np.zeros_like(e)])
    return e


def _draw_cartesian(ctx: _Ctx, li: int, layer: Layer, ax) -> list:
    plots = [p for p in layer.plots if p.visible]
    need_y2 = layer.show_y2 or any(p.y_axis == "right" for p in plots)
    ax2 = ax.twinx() if need_y2 else None
    if ax2 is not None:
        ctx.map.axes[li]["ax2"] = ax2
    handles: list = []
    state = {"cols": [p for p in plots if p.type == "column"],
             "bars": [p for p in plots if p.type == "bar"],
             "areas": [p for p in plots if p.type == "area"],
             "boxes": [p for p in plots if p.type == "box"],
             "hists": [p for p in plots if p.type == "histogram"],
             "wf": 0, "cbar": None}
    _prepare_stacks(ctx, layer, state)
    xticks = None
    for i, p in enumerate(layer.plots):
        if not p.visible:
            continue
        target = ax2 if (p.y_axis == "right" and ax2 is not None) else ax
        fn = _CART.get(p.type)
        if fn is None:
            ctx.map.errors.append(f"Plot type {p.type!r} needs a different layer type")
            continue
        ci = int(p.options.get("color_index", i))
        out = _safe(ctx, li, p, fn, ctx, li, layer, target, p, ci, state)
        if out:
            handle, ticks = out
            if handle is not None and p.id not in layer.legend.hidden:
                handles.append((handle, p))
            xticks = ticks or xticks
    _style_cartesian(ctx, li, layer, ax, ax2, xticks, plots)
    return handles


def _register(ctx, li, p, ax, artists, x=None, y=None):
    ctx.map.add(artists, ("plot", li, p.id))
    if x is not None and y is not None:
        ctx.map.points.append((li, p.id, ax, np.asarray(x, float), np.asarray(y, float)))


def _labels(ctx, li, ax, p, x, y, mask, i, order=None):
    if not p.labels:
        return
    lab = ctx.col(p.labels)
    if lab is None:
        return
    vals = lab.array()[:len(mask)]
    if order is not None:
        vals = vals[order]
    vals = vals[mask]
    font = Font(size=p.options.get("label_size", 8.0), color=ctx.line_color(p, i))
    for xv, yv, t in zip(x, y, vals):
        if isinstance(t, float):
            t = lab.value_labels.get(t, _num(t)) if np.isfinite(t) else ""
        ann = ax.annotate(rich(str(t)), (xv, yv), xytext=(3, 3), textcoords="offset points",
                          **ctx.font(font))
        ctx.map.add(ann, ("plot", li, p.id))


def _line(ctx: _Ctx, li, layer, ax, p: Plot, i, state):
    x, y = _xy(ctx, p)
    if layer.offset_x or layer.offset_y:
        x = x + state["wf"] * layer.offset_x
        y = y + state["wf"] * layer.offset_y
        state["wf"] += 1
    if p.options.get("sort_x") and p.x:
        order = np.argsort(x, kind="stable")
        x, y = x[order], y[order]
    else:
        order = None
    mask = np.isfinite(x) & np.isfinite(y)
    col = ctx.line_color(p, i)
    kw = {"color": col, "lw": p.line.width, "ls": ctx.dash(p, i), "label": p.name}
    if not p.line.visible:
        kw["ls"] = "none"
    if p.symbol.visible:
        kw.update(ctx.symbol_kw(p, i))
    else:
        kw["marker"] = "none"
    connect = p.options.get("connect", "straight")
    xs, ys = x, y
    artists = []
    if connect == "spline" and p.line.visible and mask.sum() >= 4:
        from scipy.interpolate import make_interp_spline
        order = np.argsort(x[mask])
        xu, yu = x[mask][order], y[mask][order]
        keep = np.concatenate([[True], np.diff(xu) > 0])
        xu, yu = xu[keep], yu[keep]
        if len(xu) >= 4:
            xs = np.linspace(xu[0], xu[-1], 300)
            ys = make_interp_spline(xu, yu, k=3)(xs)
            artists += ax.plot(xs, ys, color=col, lw=p.line.width, ls=ctx.dash(p, i))
            kw["ls"] = "none"
            xs, ys = x, y
    if connect == "step":
        kw["drawstyle"] = "steps-mid"
    (handle,) = ax.plot(xs, ys, **kw)
    artists.append(handle)
    if p.xerr or p.yerr:
        n = len(x)
        eb = ax.errorbar(x[mask], y[mask], yerr=_err_arrays(ctx, p, p.yerr, n, mask, order),
                         xerr=_err_arrays(ctx, p, p.xerr, n, mask, order), fmt="none",
                         ecolor=ctx.color(p.error.color, i) if p.error.color != "auto"
                         else col, elinewidth=p.error.width, capsize=p.error.cap,
                         capthick=p.error.width, zorder=handle.get_zorder() - 0.1)
        artists += list(eb.lines[1]) + list(eb.lines[2])
    if artists[:-1] and connect == "spline":
        handle = Line2D([], [], color=col, lw=p.line.width, ls=ctx.dash(p, i),
                        **(ctx.symbol_kw(p, i) if p.symbol.visible else {}))
    _register(ctx, li, p, ax, artists, x[mask], y[mask])
    _labels(ctx, li, ax, p, x[mask], y[mask], mask, i, order)
    ticks = None
    xc = ctx.col(p.x)
    if xc is not None and xc.categorical:
        codes = sorted(set(v for v in x[mask]))
        if len(codes) <= 20:
            ticks = (codes, [xc.value_labels.get(c, _num(c)) for c in codes])
    return handle, ticks


def _scatter_mapped(ctx: _Ctx, li, layer, ax, p: Plot, i, state):
    """Bubble and colour-mapped scatter: size and/or colour from extra columns."""
    x, y = _xy(ctx, p)
    n = len(x)
    size_col = ctx.arr(p.options.get("size_col")) if p.options.get("size_col") else None
    color_col = ctx.arr(p.options.get("color_col")) if p.options.get("color_col") else None
    if p.type == "bubble" and size_col is None and p.z:
        size_col = ctx.arr(p.z)
    if p.type == "color_scatter" and color_col is None and p.z:
        color_col = ctx.arr(p.z)
    mask = np.isfinite(x) & np.isfinite(y)
    for extra in (size_col, color_col):
        if extra is not None:
            mask &= np.isfinite(extra[:n])
    col = ctx.line_color(p, i)
    kw = {"marker": ctx.marker(p, i), "linewidths": p.symbol.edge_width, "alpha": p.fill.alpha,
          "edgecolors": p.symbol.edge if p.symbol.edge != "auto" else
          ("#333333" if color_col is not None or p.type == "bubble" else col),
          "label": p.name, "zorder": 2.5}
    if size_col is not None:
        s = size_col[:n][mask]
        lo, hi = float(p.options.get("size_min", 4)), float(p.options.get("size_max", 24))
        rng = np.ptp(s) if len(s) else 0
        d = lo + (s - s.min()) / rng * (hi - lo) if rng > 0 else np.full(len(s), (lo + hi) / 2)
        kw["s"] = d ** 2
    else:
        kw["s"] = p.symbol.size ** 2
    if color_col is not None:
        kw["c"] = color_col[:n][mask]
        kw["cmap"] = p.options.get("cmap", layer.cmap)
    else:
        face = p.symbol.face if p.symbol.face != "auto" else col
        kw["c"] = face if p.symbol.filled or p.type == "bubble" else "none"
    sc = ax.scatter(x[mask], y[mask], **kw)
    _register(ctx, li, p, ax, sc, x[mask], y[mask])
    handle = sc
    if size_col is not None or color_col is not None:
        face = kw["c"] if isinstance(kw["c"], str) else "#888888"
        handle = Line2D([], [], ls="none", marker=kw["marker"], markersize=p.symbol.size,
                        markerfacecolor=face, markeredgecolor=kw["edgecolors"],
                        alpha=p.fill.alpha)
    if color_col is not None and layer.colorbar and state["cbar"] is None:
        state["cbar"] = _colorbar(ctx, li, layer, ax, sc,
                                  p.options.get("colorbar_title") or _title(ctx, p.z or
                                                                           p.options.get(
                                                                               "color_col")))
    _labels(ctx, li, ax, p, x[mask], y[mask], mask, i)
    return handle, None


def _title(ctx: _Ctx, key) -> str:
    col = ctx.col(key)
    return col.title if col else ""


def _colorbar(ctx: _Ctx, li, layer: Layer, ax, mappable, title: str):
    cb = ctx.fig.colorbar(mappable, ax=ax, pad=0.03, fraction=0.06, aspect=25)
    font = layer.y.label_font
    cb.ax.tick_params(labelsize=font.size, direction=layer.y.tick_dir if
                      layer.y.tick_dir in ("in", "out") else "inout",
                      width=layer.y.tick_width, length=layer.y.major_len * 0.8,
                      labelfontfamily=resolve_family(font.family or ctx.g.font_family))
    cb.outline.set_linewidth(layer.y.line_width)
    if title:
        cb.set_label(rich(title), **ctx.font(layer.y.title_font))
    ctx.map.add(cb.ax.patch, ("colorbar", li))
    ctx.map.add(cb.ax, ("colorbar", li))
    return cb


def _prepare_stacks(ctx: _Ctx, layer: Layer, state: dict) -> None:
    for key in ("cols", "bars"):
        plots = state[key]
        if not plots:
            continue
        pos, spacing, ticks = _categories(ctx, plots)
        totals: dict = {}
        if layer.stack == "percent":
            for p in plots:
                x, y = _xy(ctx, p)
                for xv, yv in zip(x, y):
                    if np.isfinite(xv) and np.isfinite(yv):
                        totals[xv] = totals.get(xv, 0.0) + abs(yv)
        state[key + "_info"] = {"pos": pos, "spacing": spacing, "ticks": ticks,
                                "totals": totals, "pos_base": {}, "neg_base": {}}
    areas = state["areas"]
    if areas and layer.stack in ("stacked", "percent"):
        xs = [_xy(ctx, p) for p in areas]
        grid = np.unique(np.concatenate([x[np.isfinite(x)] for x, _ in xs]))
        ys = []
        for x, y in xs:
            m = np.isfinite(x) & np.isfinite(y)
            order = np.argsort(x[m])
            ys.append(np.interp(grid, x[m][order], y[m][order], left=0, right=0))
        ys = np.array(ys)
        if layer.stack == "percent":
            tot = np.abs(ys).sum(axis=0)
            ys = np.divide(ys * 100, tot, out=np.zeros_like(ys), where=tot > 0)
        state["area_info"] = {"grid": grid, "ys": ys, "cum": np.vstack([np.zeros(len(grid)),
                                                                       np.cumsum(ys, 0)])}
    if state["hists"]:
        values = [ctx.arr(p.y) for p in state["hists"]]
        allv = np.concatenate([v[np.isfinite(v)] for v in values if v is not None])
        p0 = state["hists"][0]
        bins = p0.options.get("bins", "auto")
        try:
            bins = int(bins)
        except (TypeError, ValueError):
            bins = bins if bins in ("auto", "sturges", "fd", "scott", "sqrt") else "auto"
        if len(allv):
            edges = np.histogram_bin_edges(allv, bins=bins)
            if p0.options.get("bin_width"):
                w = float(p0.options["bin_width"])
                start = math.floor(allv.min() / w) * w
                edges = np.arange(start, allv.max() + w * 1.001, w)
            state["hist_edges"] = edges


def _column(ctx: _Ctx, li, layer, ax, p: Plot, i, state):
    horizontal = p.type == "bar"
    key = "bars" if horizontal else "cols"
    info = state[key + "_info"]
    group = state[key]
    j = group.index(p)
    x, y = _xy(ctx, p)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    pos = np.array([info["pos"][v] for v in x]) if info["pos"] is not None else x.astype(float)
    total_w = info["spacing"] * (1 - layer.gap)
    stacked = layer.stack in ("stacked", "percent")
    if layer.stack == "percent":
        y = np.array([v / info["totals"][xv] * 100 if info["totals"].get(xv) else 0.0
                      for xv, v in zip(x, y)])
    if stacked:
        width, offset = total_w, 0.0
        base = np.array([(info["pos_base"] if v >= 0 else info["neg_base"]).get(xv, 0.0)
                         for xv, v in zip(x, y)])
        for xv, v, b in zip(x, y, base):
            (info["pos_base"] if v >= 0 else info["neg_base"])[xv] = b + v
    else:
        width = total_w / len(group)
        offset = (j - (len(group) - 1) / 2) * width
        base = np.zeros(len(y))
    kw = {"color": ctx.fill_color(p, i) if p.fill.visible else "none",
          "edgecolor": p.fill.edge_color if p.fill.edge_width > 0 else "none",
          "linewidth": p.fill.edge_width, "hatch": ctx.hatch(p, i), "alpha": p.fill.alpha,
          "label": p.name}
    yerr = _err_arrays(ctx, p, p.yerr, len(mask), mask) if p.yerr else None
    if layer.stack == "percent":
        yerr = None
    ecol = ctx.color(p.error.color, i) if p.error.color != "auto" else "#000000"
    ekw = {"elinewidth": p.error.width, "capthick": p.error.width, "ecolor": ecol}
    if horizontal:
        bars = ax.barh(pos + offset, y, height=width * 0.98, left=base, xerr=yerr,
                       capsize=p.error.cap, error_kw=ekw, **kw)
    else:
        bars = ax.bar(pos + offset, y, width=width * 0.98, bottom=base, yerr=yerr,
                      capsize=p.error.cap, error_kw=ekw, **kw)
    artists = list(bars.patches)
    if bars.errorbar is not None:
        artists += list(bars.errorbar.lines[2])
    if horizontal:
        _register(ctx, li, p, ax, artists, base + y, pos + offset)
    else:
        _register(ctx, li, p, ax, artists, pos + offset, base + y)
    if p.options.get("show_values"):
        fmt = p.options.get("value_format", "{:.3g}")
        try:
            ax.bar_label(bars, labels=[fmt.format(v) for v in y], padding=2,
                         **_bar_label_font(ctx, layer))
        except (ValueError, IndexError, KeyError):
            ax.bar_label(bars, padding=2, **_bar_label_font(ctx, layer))
    if p.labels:
        _labels(ctx, li, ax, p, pos + offset, base + y, mask, i)
    return bars, info["ticks"]


def _bar_label_font(ctx: _Ctx, layer: Layer) -> dict:
    f = ctx.font(layer.x.label_font, 0.85)
    f.pop("math_fontfamily", None)
    return {"fontsize": f["fontsize"], "fontfamily": f["fontfamily"], "color": f["color"]}


def _area(ctx: _Ctx, li, layer, ax, p: Plot, i, state):
    col = ctx.fill_color(p, i)
    if "area_info" in state:
        info = state["area_info"]
        j = state["areas"].index(p)
        x, lo, hi = info["grid"], info["cum"][j], info["cum"][j + 1]
    else:
        x, y = _xy(ctx, p)
        m = np.isfinite(x) & np.isfinite(y)
        x, hi = x[m], y[m]
        lo = np.zeros(len(x))
    poly = ax.fill_between(x, lo, hi, color=col if p.fill.visible else "none",
                           alpha=p.fill.alpha if p.fill.alpha < 1 or "area_info" in state
                           else 0.75, hatch=ctx.hatch(p, i), linewidth=0, label=p.name)
    artists = [poly]
    if p.line.visible:
        lc = ctx.color(p.line.color, i, fill=True)
        artists += ax.plot(x, hi, color=lc, lw=p.line.width, ls=ctx.dash(p, i))
    _register(ctx, li, p, ax, artists, x, hi)
    return poly, None


def _histogram(ctx: _Ctx, li, layer, ax, p: Plot, i, state):
    v = ctx.arr(p.y)
    v = v[np.isfinite(v)]
    if not len(v):
        raise ValueError("no valid data")
    edges = state.get("hist_edges")
    mode = p.options.get("scale", "count")     # count | percent | density
    weights = np.full(len(v), 100.0 / len(v)) if mode == "percent" else None
    multi = len(state["hists"]) > 1
    alpha = p.fill.alpha if p.fill.alpha < 1 or not multi else 0.6
    counts, bins, patches = ax.hist(
        v, bins=edges if edges is not None else "auto", weights=weights,
        density=mode == "density", color=ctx.fill_color(p, i) if p.fill.visible else "none",
        edgecolor=p.fill.edge_color, linewidth=p.fill.edge_width, hatch=ctx.hatch(p, i),
        alpha=alpha, label=p.name, cumulative=bool(p.options.get("cumulative")))
    artists = list(patches)
    centers = (bins[:-1] + bins[1:]) / 2
    if p.options.get("normal_curve", True) and len(v) > 1 and np.std(v) > 0 and \
            not p.options.get("cumulative"):
        mean, sd = v.mean(), v.std(ddof=1)
        xs = np.linspace(bins[0] - (bins[1] - bins[0]), bins[-1] + (bins[1] - bins[0]), 300)
        scale = {"count": len(v) * (bins[1] - bins[0]), "percent": 100 * (bins[1] - bins[0]),
                 "density": 1.0}[mode]
        lc = ctx.color(p.line.color, i) if p.line.color != "auto" else "#000000"
        if multi and p.line.color == "auto":
            lc = ctx.fill_color(p, i)
        artists += ax.plot(xs, stats.norm.pdf(xs, mean, sd) * scale, color=lc,
                           lw=p.line.width, ls=ctx.dash(p, i))
    _register(ctx, li, p, ax, artists, centers, counts)
    return Patch(facecolor=ctx.fill_color(p, i), edgecolor=p.fill.edge_color,
                 hatch=ctx.hatch(p, i), alpha=alpha), None


def _box(ctx: _Ctx, li, layer, ax, p: Plot, i, state):
    v = ctx.arr(p.y)
    v = v[np.isfinite(v)]
    if not len(v):
        raise ValueError("no valid data")
    boxes = state["boxes"]
    pos = float(p.options.get("position", boxes.index(p) + 1))
    face = ctx.fill_color(p, i) if p.fill.visible else "none"
    edge = p.fill.edge_color
    lw = p.fill.edge_width
    whis = p.options.get("whisker", 1.5)
    if whis == "range":
        whis = (0, 100)
    elif whis == "5-95":
        whis = (5, 95)
    elif whis == "10-90":
        whis = (10, 90)
    else:
        whis = float(whis)
    bp = ax.boxplot([v], positions=[pos], widths=p.options.get("width", 0.5),
                    patch_artist=True, notch=bool(p.options.get("notch")), whis=whis,
                    showmeans=p.options.get("show_mean", True),
                    showfliers=p.options.get("outliers", True),
                    boxprops={"facecolor": face, "edgecolor": edge, "linewidth": lw,
                              "hatch": ctx.hatch(p, i), "alpha": p.fill.alpha},
                    medianprops={"color": edge, "linewidth": lw * 1.4},
                    whiskerprops={"color": edge, "linewidth": lw},
                    capprops={"color": edge, "linewidth": lw},
                    meanprops={"marker": "s", "markerfacecolor": "white",
                               "markeredgecolor": edge, "markersize": 4},
                    flierprops={"marker": "o", "markersize": 3.5, "markerfacecolor": "none",
                                "markeredgecolor": edge})
    artists = [a for group in bp.values() for a in group]
    if p.options.get("points"):
        rng = np.random.default_rng(12345)
        w = float(p.options.get("width", 0.5))
        jitter = pos + rng.uniform(-0.3, 0.3, len(v)) * w
        artists.append(ax.scatter(jitter, v, s=p.symbol.size ** 2 * 0.6,
                                  color=ctx.line_color(p, i), alpha=0.7, linewidths=0,
                                  zorder=3))
    _register(ctx, li, p, ax, artists, np.full(len(v), pos), v)
    if any("group_label" in q.options for q in boxes):
        seen = {}
        for q in boxes:
            if "group_label" in q.options:
                seen.setdefault(float(q.options["group_pos"]), q.options["group_label"])
        ticks = (list(seen), list(seen.values()))
    else:
        labels = [(q.options.get("position", k + 1), q.name) for k, q in enumerate(boxes)]
        ticks = ([float(a) for a, _ in labels], [b for _, b in labels])
    return Patch(facecolor=face, edgecolor=edge, hatch=ctx.hatch(p, i)), ticks


def _contour(ctx: _Ctx, li, layer, ax, p: Plot, i, state):
    x, y, z = ctx.arr(p.x), ctx.arr(p.y), ctx.arr(p.z)
    if x is None or y is None or z is None:
        raise ValueError("contour needs X, Y and Z columns")
    X, Y, Z, regular = xyz_grid(x, y, z, p.options.get("grid", "auto"),
                                int(p.options.get("grid_size", 80)))
    cmap = p.options.get("cmap", layer.cmap)
    artists = []
    mappable = None
    if p.type == "heatmap":
        mappable = ax.pcolormesh(X, Y, Z, cmap=cmap, shading="nearest" if regular else "auto")
        artists.append(mappable)
        if regular and p.options.get("show_values") and Z.size <= 400:
            for (r, c), val in np.ndenumerate(Z):
                if np.isfinite(val):
                    artists.append(ax.text(X[r, c], Y[r, c], f"{val:.3g}", ha="center",
                                           va="center", fontsize=layer.x.label_font.size * .7))
    else:
        levels = int(p.options.get("levels", 10))
        if p.fill.visible:
            mappable = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=p.fill.alpha)
            artists.append(mappable)
        if p.line.visible:
            lc = p.line.color if p.line.color != "auto" else "#000000"
            cs = ax.contour(X, Y, Z, levels=mappable.levels if mappable is not None else levels,
                            colors=None if lc == "cmap" else lc,
                            cmap=cmap if lc == "cmap" else None,
                            linewidths=p.line.width * 0.6, negative_linestyles="solid")
            artists.append(cs)
            if p.options.get("labels", False):
                artists += cs.clabel(fontsize=layer.x.label_font.size * 0.75, fmt="%.3g")
            mappable = mappable or cs
        if p.options.get("show_points"):
            artists.append(ax.scatter(x, y, s=6, c="k", marker=".", zorder=3))
    _register(ctx, li, p, ax, artists, X.ravel(), Y.ravel())
    if layer.colorbar and mappable is not None and state["cbar"] is None:
        state["cbar"] = _colorbar(ctx, li, layer, ax, mappable,
                                  p.options.get("colorbar_title") or _title(ctx, p.z))
    return None, None


_CART = {"line": _line, "scatter": _line, "line_symbol": _line, "column": _column,
         "bar": _column, "area": _area, "histogram": _histogram, "box": _box,
         "bubble": _scatter_mapped, "color_scatter": _scatter_mapped, "contour": _contour,
         "heatmap": _contour}


# ---------------------------------------------------------------------- axis styling
def _tick_dir(axis: Axis) -> str:
    return {"in": "in", "out": "out", "both": "inout"}.get(axis.tick_dir, "in")


def _apply_ticks(ctx: _Ctx, mpl_axis, axis: Axis, sides: dict) -> None:
    length = 0 if axis.tick_dir == "none" or not axis.visible else 1
    family = resolve_family(axis.label_font.family or ctx.g.font_family)
    common = {"direction": _tick_dir(axis), "width": axis.tick_width, "color": axis.color,
              "labelsize": axis.label_font.size, "labelcolor": axis.label_font.color,
              "labelfontfamily": family}
    mpl_axis.set_tick_params(which="major", length=axis.major_len * length, **common,
                             **sides)
    mpl_axis.set_tick_params(which="minor", length=axis.minor_len * length, **common,
                             **sides)
    for t in mpl_axis.get_ticklabels(which="both"):
        t.set_fontweight("bold" if axis.label_font.bold else "normal")
        t.set_fontstyle("italic" if axis.label_font.italic else "normal")
        if axis.label_rotation:
            t.set_rotation(axis.label_rotation)


def _scale(mpl_ax, which: str, axis: Axis) -> None:
    setter = mpl_ax.set_xscale if which == "x" else mpl_ax.set_yscale
    if axis.scale == "log10":
        setter("log", base=10)
    elif axis.scale == "log2":
        setter("log", base=2)
    elif axis.scale == "ln":
        setter("log", base=math.e)
    else:
        setter("linear")


def _limits(mpl_ax, which: str, axis: Axis) -> None:
    get = mpl_ax.get_xlim if which == "x" else mpl_ax.get_ylim
    lo, hi = sorted(get())
    lo = axis.min if axis.min is not None else lo
    hi = axis.max if axis.max is not None else hi
    if axis.scale != "linear" and lo <= 0:
        lo = min(hi / 10, 1e-3) if hi > 0 else 1e-3
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    (mpl_ax.set_xlim if which == "x" else mpl_ax.set_ylim)((hi, lo) if axis.reverse
                                                          else (lo, hi))


def _locators(mpl_axis, axis: Axis, view: tuple[float, float]) -> None:
    if axis.scale == "linear":
        span = abs(view[1] - view[0])
        if axis.major_step and axis.major_step > 0 and span / axis.major_step <= 200:
            mpl_axis.set_major_locator(ticker.MultipleLocator(axis.major_step))
        if axis.minor_count > 0:
            mpl_axis.set_minor_locator(ticker.AutoMinorLocator(axis.minor_count + 1))
        else:
            mpl_axis.set_minor_locator(ticker.NullLocator())
        if axis.decimals >= 0:
            mpl_axis.set_major_formatter(ticker.FormatStrFormatter(f"%.{axis.decimals}f"))
    elif axis.minor_count == 0:
        mpl_axis.set_minor_locator(ticker.NullLocator())
    if not axis.show_labels:
        mpl_axis.set_major_formatter(ticker.NullFormatter())
        mpl_axis.set_minor_formatter(ticker.NullFormatter())


def _grid(mpl_axis, axis: Axis) -> None:
    if axis.grid_major:
        mpl_axis.grid(True, which="major", color=axis.grid_color, linestyle=axis.grid_style,
                      linewidth=0.6)
    else:
        mpl_axis.grid(False, which="major")
    if axis.grid_minor:
        mpl_axis.grid(True, which="minor", color=axis.grid_color, linestyle=axis.grid_style,
                      linewidth=0.4)


def _axis_title(ctx: _Ctx, setter, axis: Axis, li: int, which: str):
    if axis.show_title and axis.title and axis.visible:
        t = setter(rich(axis.title), **ctx.font(axis.title_font))
        ctx.map.add(t, ("axis", li, which))


def _headroom(layer: Layer, ax, ax2, plots) -> None:
    """Leave space at the top for the legend on filled charts and double-Y graphs."""
    if not layer.legend.visible or layer.legend.loc not in ("best", "upper right",
                                                             "upper left", "upper center"):
        return
    shown = [p for p in plots if p.id not in layer.legend.hidden]
    if len(shown) < 2:
        return
    filled = any(p.type in ("column", "box", "histogram", "area") for p in plots)
    horizontal = any(p.type == "bar" for p in plots) and not filled
    if not (filled or horizontal or ax2 is not None):
        return
    rows = 1 + (len(shown) - 1) // max(1, layer.legend.ncols)
    frac = min(0.45, 0.07 * rows + 0.06)
    targets = [(ax, "x" if horizontal else "y", layer.x if horizontal else layer.y)]
    if ax2 is not None:
        targets.append((ax2, "y", layer.y2))
    for mpl_ax, which, axis in targets:
        if axis.max is not None or axis.reverse or axis.scale != "linear":
            continue
        lo, hi = mpl_ax.get_xlim() if which == "x" else mpl_ax.get_ylim()
        hi = hi + (hi - lo) * frac / (1 - frac)
        (mpl_ax.set_xlim if which == "x" else mpl_ax.set_ylim)(lo, hi)


def _style_cartesian(ctx: _Ctx, li, layer: Layer, ax, ax2, xticks, plots) -> None:
    for which, axis in (("x", layer.x), ("y", layer.y)):
        _scale(ax, which, axis)
    if ax2 is not None:
        _scale(ax2, "y", layer.y2)
    if xticks is not None and layer.x.scale == "linear" and layer.x.major_step is None:
        horizontal = any(p.type == "bar" for p in plots) and \
            not any(p.type == "column" for p in plots)
        mpl_axis = ax.yaxis if horizontal else ax.xaxis
        mpl_axis.set_major_locator(ticker.FixedLocator(xticks[0]))
        mpl_axis.set_major_formatter(ticker.FixedFormatter([rich(str(t)) for t in xticks[1]]))
        mpl_axis.set_minor_locator(ticker.NullLocator())
        cat_axis = "y" if horizontal else "x"
        if not horizontal and not layer.x.label_rotation:
            width_pt = ctx.g.page.width_cm / 2.54 * 72 * layer.grid[3] / max(
                1, ctx.g.cols) * 0.75
            need = sum(len(str(t)) + 2 for t in xticks[1]) * layer.x.label_font.size * 0.55
            if need > width_pt:
                for t in mpl_axis.get_ticklabels():
                    t.set_rotation(30)
                    t.set_ha("right")
                    t.set_rotation_mode("anchor")
    else:
        cat_axis = None
    ax.autoscale_view()
    _headroom(layer, ax, ax2, plots)
    _limits(ax, "x", layer.x)
    _limits(ax, "y", layer.y)
    if ax2 is not None:
        _limits(ax2, "y", layer.y2)
    if cat_axis != "x":
        _locators(ax.xaxis, layer.x, ax.get_xlim())
    if cat_axis != "y":
        _locators(ax.yaxis, layer.y, ax.get_ylim())
    if ax2 is not None:
        _locators(ax2.yaxis, layer.y2, ax2.get_ylim())
    xs = {"bottom": layer.x.visible, "labelbottom": layer.x.visible,
          "top": layer.x.mirror_ticks and layer.box, "labeltop": False}
    ys = {"left": layer.y.visible, "labelleft": layer.y.visible,
          "right": layer.y.mirror_ticks and layer.box and ax2 is None, "labelright": False}
    _apply_ticks(ctx, ax.xaxis, layer.x, xs)
    _apply_ticks(ctx, ax.yaxis, layer.y, ys)
    _grid(ax.xaxis, layer.x)
    _grid(ax.yaxis, layer.y)
    ax.set_axisbelow(True)
    _axis_title(ctx, ax.set_xlabel, layer.x, li, "x")
    _axis_title(ctx, ax.set_ylabel, layer.y, li, "y")
    spines = ax.spines
    spines["bottom"].set_visible(layer.x.visible)
    spines["left"].set_visible(layer.y.visible)
    spines["top"].set_visible(layer.box)
    spines["right"].set_visible(layer.box and ax2 is None)
    for side, axis in (("bottom", layer.x), ("top", layer.x), ("left", layer.y),
                       ("right", layer.y)):
        spines[side].set_linewidth(axis.line_width)
        spines[side].set_color(axis.color)
    ctx.map.add(ax.xaxis, ("axis", li, "x"))
    ctx.map.add(ax.yaxis, ("axis", li, "y"))
    if ax2 is not None:
        y2 = layer.y2
        _apply_ticks(ctx, ax2.yaxis, y2, {"right": y2.visible, "labelright": y2.visible,
                                          "left": False, "labelleft": False})
        _grid(ax2.yaxis, y2)
        _axis_title(ctx, ax2.set_ylabel, y2, li, "y2")
        for side in ("top", "bottom", "left"):
            ax2.spines[side].set_visible(False)
        ax2.spines["right"].set_visible(y2.visible)
        ax2.spines["right"].set_linewidth(y2.line_width)
        ax2.spines["right"].set_color(y2.color)
        ax2.patch.set_visible(False)
        ctx.map.add(ax2.yaxis, ("axis", li, "y2"))


# ====================================================================== other layers
def _draw_polar(ctx: _Ctx, li, layer: Layer, ax) -> list:
    handles = []
    for i, p in enumerate(layer.plots):
        if not p.visible:
            continue

        def draw(p=p, i=i):
            th, r = _xy(ctx, p)
            if layer.options.get("theta_unit", "deg") == "deg":
                th = np.deg2rad(th)
            m = np.isfinite(th) & np.isfinite(r)
            kw = {"color": ctx.line_color(p, i), "lw": p.line.width,
                  "ls": ctx.dash(p, i) if p.line.visible else "none", "label": p.name}
            kw.update(ctx.symbol_kw(p, i) if p.symbol.visible else {"marker": "none"})
            th_m, r_m = th[m], r[m]
            if p.options.get("close", False) and len(th_m):
                th_m, r_m = np.append(th_m, th_m[0]), np.append(r_m, r_m[0])
            (h,) = ax.plot(th_m, r_m, **kw)
            if p.type == "area" or p.options.get("fill"):
                ax.fill(th_m, r_m, color=ctx.fill_color(p, i), alpha=0.3)
            _register(ctx, li, p, ax, h, th[m], r[m])
            return h
        h = _safe(ctx, li, p, draw)
        if h is not None and p.id not in layer.legend.hidden:
            handles.append((h, p))
    ax.set_theta_direction(-1 if layer.x.reverse else 1)
    ax.set_theta_zero_location(layer.options.get("theta_zero", "E"))
    if layer.y.min is not None or layer.y.max is not None:
        lo, hi = ax.get_ylim()
        ax.set_ylim(layer.y.min if layer.y.min is not None else lo,
                    layer.y.max if layer.y.max is not None else hi)
    if layer.y.major_step:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(layer.y.major_step))
    if layer.x.major_step:
        ax.set_thetagrids(np.arange(0, 360, layer.x.major_step))
    for mpl_axis, axis in ((ax.xaxis, layer.x), (ax.yaxis, layer.y)):
        mpl_axis.set_tick_params(labelsize=axis.label_font.size,
                                 labelfontfamily=resolve_family(axis.label_font.family or
                                                                ctx.g.font_family))
        if axis.grid_major:
            mpl_axis.grid(True, color=axis.grid_color, linestyle=axis.grid_style,
                          linewidth=0.6)
        else:
            mpl_axis.grid(False)
        if not axis.show_labels:
            mpl_axis.set_major_formatter(ticker.NullFormatter())
    ax.spines["polar"].set_visible(layer.box)
    ax.spines["polar"].set_linewidth(layer.x.line_width)
    if layer.y.show_title and layer.y.title:
        t = ax.text(np.deg2rad(22.5), ax.get_rmax() * 1.18, rich(layer.y.title),
                    ha="center", va="center", **ctx.font(layer.y.title_font))
        ctx.map.add(t, ("axis", li, "y"))
    ctx.map.add(ax.xaxis, ("axis", li, "x"))
    ctx.map.add(ax.yaxis, ("axis", li, "y"))
    return handles


def _draw_3d(ctx: _Ctx, li, layer: Layer, ax) -> list:
    handles = []
    cbar_done = False
    for i, p in enumerate(layer.plots):
        if not p.visible:
            continue

        def draw(p=p, i=i):
            nonlocal cbar_done
            x, y, z = ctx.arr(p.x), ctx.arr(p.y), ctx.arr(p.z)
            if x is None or y is None or z is None:
                raise ValueError("3D plots need X, Y and Z columns")
            cmap = p.options.get("cmap", layer.cmap)
            if p.type == "surface3d":
                X, Y, Z, _ = xyz_grid(x, y, z, p.options.get("grid", "auto"),
                                      int(p.options.get("grid_size", 50)),
                                      fill_outside=p.options.get("extrapolate", True))
                if p.options.get("wireframe"):
                    art = ax.plot_wireframe(X, Y, Z, color=ctx.line_color(p, i),
                                            linewidth=p.line.width * 0.6)
                    mappable = None
                elif p.options.get("solid"):   # Origin's 3D Surface with a single fill colour
                    edge = ctx.line_color(p, i) if p.line.visible else "none"
                    art = ax.plot_surface(X, Y, np.ma.masked_invalid(Z),
                                          color=ctx.fill_color(p, i), shade=True,
                                          edgecolor=edge, linewidth=p.line.width * 0.3,
                                          alpha=p.fill.alpha, antialiased=True,
                                          rstride=1, cstride=1)
                    mappable = None
                else:
                    edge = ctx.line_color(p, i) if p.line.visible else "none"
                    art = ax.plot_surface(X, Y, np.ma.masked_invalid(Z), cmap=cmap,
                                          edgecolor=edge, linewidth=p.line.width * 0.3,
                                          alpha=p.fill.alpha, antialiased=True,
                                          rstride=1, cstride=1)
                    mappable = art
                if p.options.get("floor_contour"):
                    zmin = np.nanmin(Z)
                    ax.contourf(X, Y, Z, zdir="z", offset=zmin, cmap=cmap, alpha=0.6)
                _register(ctx, li, p, ax, art)
            else:
                m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
                by_z = p.options.get("color_by_z", True)
                kw = {"s": p.symbol.size ** 2, "marker": ctx.marker(p, i),
                      "depthshade": True, "edgecolors": "#333333", "linewidths": 0.3}
                if by_z:
                    kw.update(c=z[m], cmap=cmap)
                else:
                    kw.update(color=ctx.line_color(p, i))
                art = ax.scatter(x[m], y[m], z[m], label=p.name, **kw)
                mappable = art if by_z else None
                if p.options.get("drop_lines"):
                    zmin = np.nanmin(z[m])
                    for xv, yv, zv in zip(x[m], y[m], z[m]):
                        ax.plot([xv, xv], [yv, yv], [zmin, zv], color="#999999", lw=0.5)
                _register(ctx, li, p, ax, art)
            if mappable is not None and layer.colorbar and not cbar_done:
                cb = ctx.fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.08, aspect=18)
                cb.ax.tick_params(labelsize=layer.z.label_font.size)
                if p.options.get("colorbar_title"):
                    cb.set_label(rich(p.options["colorbar_title"]),
                                 **ctx.font(layer.z.title_font))
                ctx.map.add(cb.ax, ("colorbar", li))
                cbar_done = True
            return art
        art = _safe(ctx, li, p, draw)
        if art is not None and p.type == "scatter3d" and p.id not in layer.legend.hidden:
            handles.append((art, p))
    ax.view_init(elev=layer.elev, azim=layer.azim)
    for which, axis in (("x", layer.x), ("y", layer.y), ("z", layer.z)):
        mpl_axis = getattr(ax, f"{which}axis")
        if axis.min is not None or axis.max is not None:
            get = getattr(ax, f"get_{which}lim")
            lo, hi = get()
            getattr(ax, f"set_{which}lim")(axis.min if axis.min is not None else lo,
                                           axis.max if axis.max is not None else hi)
        if axis.major_step:
            mpl_axis.set_major_locator(ticker.MultipleLocator(axis.major_step))
        mpl_axis.set_tick_params(labelsize=axis.label_font.size, pad=1,
                                 labelfontfamily=resolve_family(axis.label_font.family or
                                                                ctx.g.font_family))
        if axis.show_title and axis.title:
            t = getattr(ax, f"set_{which}label")(rich(axis.title),
                                                 labelpad=6 if which == "z" else 2,
                                                 **ctx.font(axis.title_font))
            ctx.map.add(t, ("axis", li, which))
        mpl_axis.pane.set_facecolor((1, 1, 1, 0) if not layer.box else (0.96, 0.96, 0.96,
                                                                        1))
        mpl_axis._axinfo["grid"]["linewidth"] = 0.4 if axis.grid_major else 0.0
    ax.grid(any(a.grid_major for a in (layer.x, layer.y, layer.z)))
    try:
        ax.set_box_aspect(None, zoom=0.88)
    except TypeError:
        pass
    return handles


def _draw_ternary(ctx: _Ctx, li, layer: Layer, ax) -> list:
    scale100 = any(p.options.get("percent") for p in layer.plots)
    arts = ternary.draw_frame(ax, layer, ctx.font, rich, scale100)
    for name, items in arts.items():
        ctx.map.add(items, ("axis", li, name))
    handles = []
    for i, p in enumerate(layer.plots):
        if not p.visible:
            continue

        def draw(p=p, i=i):
            a, b, c = ctx.arr(p.x), ctx.arr(p.y), ctx.arr(p.z)
            if a is None or b is None or c is None:
                raise ValueError("ternary plots need three component columns (X, Y, Z)")
            tx, ty = ternary.to_xy(a, b, c)
            m = np.isfinite(tx) & np.isfinite(ty)
            kw = {"color": ctx.line_color(p, i), "lw": p.line.width,
                  "ls": ctx.dash(p, i) if p.line.visible else "none", "label": p.name,
                  "zorder": 4}
            kw.update(ctx.symbol_kw(p, i) if p.symbol.visible else {"marker": "none"})
            (h,) = ax.plot(tx[m], ty[m], **kw)
            _register(ctx, li, p, ax, h, tx[m], ty[m])
            _labels(ctx, li, ax, p, tx[m], ty[m], m, i)
            return h
        h = _safe(ctx, li, p, draw)
        if h is not None and p.id not in layer.legend.hidden:
            handles.append((h, p))
    return handles


def _draw_pie(ctx: _Ctx, li, layer: Layer, ax) -> list:
    handles = []
    for i, p in enumerate(layer.plots[:1]):
        y = ctx.arr(p.y)
        if y is None:
            raise ValueError("pie needs a Y column")
        labels_col = ctx.col(p.labels) or ctx.col(p.x)
        if labels_col is not None:
            raw = labels_col.array()[:len(y)]
            names = [labels_col.value_labels.get(v, _num(v)) if not isinstance(v, str) else v
                     for v in raw]
        else:
            names = [str(k + 1) for k in range(len(y))]
        m = np.isfinite(y) & (y > 0)
        y = y[m]
        names = [n for n, keep in zip(names, m) if keep]
        colors = [ctx.color("auto", k, fill=True) for k in range(len(y))]
        if p.options.get("colors"):
            colors = [c or colors[k] for k, c in
                      enumerate((p.options["colors"] + [None] * len(y))[:len(y)])]
        hole = float(p.options.get("hole", 0.45)) if p.type == "doughnut" else 0.0
        show_pct = p.options.get("show_percent", True)
        show_lab = p.options.get("show_labels", True)
        decimals = int(p.options.get("percent_decimals", 1))
        explode = [float(p.options.get("explode", 0.0))] * len(y)
        font = ctx.font(layer.x.label_font)
        font.pop("math_fontfamily", None)
        pct_font = dict(font)
        wedges, *texts = ax.pie(
            y, labels=[rich(n) for n in names] if show_lab else None, colors=colors,
            autopct=(f"%.{decimals}f%%" if show_pct else None),
            startangle=float(p.options.get("start_angle", 90)),
            counterclock=not p.options.get("clockwise", True), explode=explode,
            pctdistance=1 - hole / 2 if hole else 0.65, labeldistance=1.1,
            wedgeprops={"width": 1 - hole if hole else None,
                        "edgecolor": p.fill.edge_color, "linewidth": p.fill.edge_width},
            textprops=font)
        if len(texts) > 1:
            for t in texts[1]:
                t.set_fontsize(pct_font["fontsize"] * 0.9)
        ax.set_aspect("equal")
        ctx.map.add(list(wedges), ("plot", li, p.id))
        for group in texts:
            ctx.map.add(list(group), ("plot", li, p.id))
        if p.id not in layer.legend.hidden:
            for w, n in zip(wedges, names):
                handles.append((w, Plot(name=n, id=p.id)))
    return handles


# ====================================================================== legend & annotations
def _legend(ctx: _Ctx, li, layer: Layer, ax, handles: list) -> None:
    lg = layer.legend
    if not lg.visible or not handles:
        return
    hs = [h for h, _ in handles]
    names = [rich(p.name or " ") for _, p in handles]
    kw = {"frameon": lg.frame, "prop": ctx.fontprops(lg.font), "ncols": max(1, lg.ncols),
          "labelcolor": lg.font.color, "fancybox": False, "edgecolor": "#000000",
          "framealpha": 1.0, "handlelength": 1.8, "borderaxespad": 0.6}
    if lg.frame:
        kw["borderpad"] = 0.5
    if lg.title:
        kw["title"] = rich(lg.title)
        kw["title_fontproperties"] = ctx.fontprops(lg.font)
    if lg.loc == "custom":
        kw.update(loc="lower left", bbox_to_anchor=(lg.x, lg.y))
    elif lg.loc.startswith("outside"):
        side = lg.loc.split()[-1]
        anchor = {"right": ("center left", (1.02, 0.5)), "top": ("lower center", (0.5, 1.02)),
                  "bottom": ("upper center", (0.5, -0.18))}[side]
        kw.update(loc=anchor[0], bbox_to_anchor=anchor[1])
    else:
        kw["loc"] = lg.loc
    leg = ax.legend(hs, names, **kw)
    if lg.frame:
        leg.get_frame().set_linewidth(0.8)
    ctx.map.add(leg, ("legend", li))


def _transform(ctx: _Ctx, ax, coords: str):
    return {"axes": ax.transAxes, "data": ax.transData,
            "figure": ctx.fig.transFigure}.get(coords, ax.transAxes)


def _annotations(ctx: _Ctx, li, layer: Layer, ax) -> None:
    if layer.kind == "3d":
        for a in layer.annotations:
            if a.kind == "text":
                t = ax.text2D(a.x, a.y, rich(a.text), transform=ax.transAxes,
                              **ctx.font(a.font))
                ctx.map.add(t, ("annotation", li, a.id))
        return
    for a in layer.annotations:
        tr = _transform(ctx, ax, a.coords)
        try:
            if a.kind == "text":
                box = None
                if a.frame or a.fill.visible:
                    box = {"boxstyle": "square,pad=0.3",
                           "facecolor": a.fill.color if a.fill.visible else "none",
                           "edgecolor": a.line.color if a.frame else "none",
                           "linewidth": a.line.width}
                art = ax.text(a.x, a.y, rich(a.text), transform=tr, bbox=box,
                              ha="left", va="bottom", zorder=6, clip_on=False,
                              **ctx.font(a.font))
            elif a.kind in ("arrow", "line"):
                style = "-|>" if a.kind == "arrow" else "-"
                art = FancyArrowPatch((a.x, a.y), (a.x2, a.y2), transform=tr,
                                      arrowstyle=style, mutation_scale=a.head_size,
                                      color=a.line.color, linewidth=a.line.width,
                                      linestyle=a.line.style if a.line.style != "auto"
                                      else "-", zorder=6, shrinkA=0, shrinkB=0)
                ax.add_patch(art)
                art.set_clip_on(False)
            elif a.kind == "rect":
                x0, y0 = min(a.x, a.x2), min(a.y, a.y2)
                art = Rectangle((x0, y0), abs(a.x2 - a.x), abs(a.y2 - a.y), transform=tr,
                                facecolor=a.fill.color if a.fill.visible else "none",
                                alpha=a.fill.alpha if a.fill.visible else 1.0,
                                edgecolor=a.line.color, linewidth=a.line.width,
                                linestyle=a.line.style if a.line.style != "auto" else "-",
                                zorder=5)
                ax.add_patch(art)
                art.set_clip_on(False)
            else:
                continue
            art.set_in_layout(False)
            ctx.map.add(art, ("annotation", li, a.id))
        except Exception as exc:  # noqa: BLE001
            ctx.map.errors.append(f"Annotation: {exc}")

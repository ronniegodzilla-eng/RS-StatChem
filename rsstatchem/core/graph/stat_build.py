"""Graph builders for the statistical, 3D and specialised templates (used by
:func:`rsstatchem.core.graph.build.build_graph`). No Qt here."""
from __future__ import annotations

from rsstatchem.core.graph.model import Layer


def _b():
    from rsstatchem.core.graph import build
    return build


def _dist(snap, spec, t, graph, layer, ys):
    b = _b()
    for glabel, mask, tag in snap.groups(spec.group):
        for y in ys:
            layer.plots.append(b._plot(t, spec, name=b._series_name(snap, y, ys, glabel),
                                       y=snap.num(y, mask, tag)))
    title = snap.title(ys[0]) if len(ys) == 1 else "Value"
    if spec.kind == "kde":
        layer.x.title, layer.y.title = title, "Density"
    elif spec.kind == "ecdf":
        layer.x.title, layer.y.title = title, "Cumulative Probability"
    elif spec.kind == "qq":
        layer.x.title, layer.y.title = "Expected Normal Quantile", title
    elif spec.kind == "probability":
        layer.x.title, layer.y.title = title, "Percent"
        layer.y.scale = "probability"
        layer.y.grid_major = True


def _boxlike(snap, spec, t, graph, layer, ys):
    b = _b()
    b._build_box(snap, spec, t, graph, layer, ys)
    if spec.kind in ("meanbar", "interval"):
        text = "Mean ± 95% CI" if spec.kind == "interval" else "Mean ± SD"
        short = "mean ± 95% CI" if spec.kind == "interval" else "mean ± SD"
        layer.y.title = f"{layer.y.title} ({short})" if layer.y.title else text


def _hist(snap, spec, t, graph, layer, ys):
    b = _b()
    b._build_hist(snap, spec, t, graph, layer, ys)
    if spec.kind == "hist_kde":
        layer.y.title = "Density"


def _control(snap, spec, t, graph, layer, ys):
    b = _b()
    y = ys[0]
    charts = (("xbar", "X-bar", "Subgroup"), ("r", "Range", "Subgroup")) \
        if spec.kind == "control_xbar" else (("i", "Individual Value", "Observation"),
                                             ("mr", "Moving Range", "Observation"))
    graph.layers.clear()
    graph.rows, graph.cols = 2, 1
    for k, (chart, ytitle, xtitle) in enumerate(charts):
        lay = Layer(kind="cartesian", grid=[k, 0, 1, 1])
        p = b._plot(t, spec, name=f"{snap.title(y)} {ytitle}", y=snap.num(y))
        p.options["chart"] = chart
        lay.plots.append(p)
        lay.x.title, lay.y.title = xtitle, ytitle
        lay.panel_label = f"({chr(97 + k)})"
        graph.layers.append(lay)


def _scatter_matrix(snap, spec, t, graph, layer, ys):
    b = _b()
    names = list(dict.fromkeys(([spec.x] if spec.x else []) + list(ys)))
    n = len(names)
    graph.layers.clear()
    graph.rows = graph.cols = n
    hist_t = b.TEMPLATES["histogram"]
    for r, yv in enumerate(names):
        for c, xv in enumerate(names):
            lay = Layer(kind="cartesian", grid=[r, c, 1, 1])
            if r == c:
                p = b._plot(hist_t, spec, name=snap.title(xv), y=snap.num(xv))
                p.options["normal_curve"] = False
            else:
                p = b._plot(t, spec, name=f"{snap.title(yv)} vs {snap.title(xv)}",
                            x=snap.num(xv), y=snap.num(yv))
            lay.plots.append(p)
            lay.x.title = snap.title(xv) if r == n - 1 else ""
            lay.y.title = snap.title(yv) if c == 0 else ""
            for axis in (lay.x, lay.y):
                axis.label_font.size = 8.0
                axis.title_font.size = 9.0
            graph.layers.append(lay)


def _marginal(snap, spec, t, graph, layer, ys):
    b = _b()
    y = ys[0]
    graph.layers.clear()
    graph.rows = graph.cols = 4
    main = Layer(kind="cartesian", grid=[1, 0, 3, 3])
    main.plots.append(b._plot(t, spec, name=snap.title(y), x=snap.num(spec.x),
                              y=snap.num(y)))
    main.x.title, main.y.title = snap.title(spec.x), snap.title(y)
    hist_t = b.TEMPLATES["histogram"]
    top = Layer(kind="cartesian", grid=[0, 0, 1, 3])
    top.plots.append(b._plot(hist_t, spec, name=snap.title(spec.x), y=snap.num(spec.x)))
    top.x.show_labels = False
    top.y.title = "Count"
    top.options["link_x"] = 0
    right = Layer(kind="cartesian", grid=[1, 3, 3, 1])
    hp = b._plot(hist_t, spec, name=snap.title(y), y=snap.num(y))
    hp.options["orientation"] = "horizontal"
    right.plots.append(hp)
    right.y.show_labels = False
    right.x.title = "Count"
    right.options["link_y"] = 0
    for lay in (top, right):
        for p in lay.plots:
            p.options["normal_curve"] = True
    graph.layers += [main, top, right]


def _xy_then(fix):
    def builder(snap, spec, t, graph, layer, ys):
        _b()._build_xy(snap, spec, t, graph, layer, ys)
        fix(snap, spec, graph, layer, ys)
    return builder


def _fix_fill(snap, spec, graph, layer, ys):
    for k, p in enumerate(layer.plots):
        p.options["fill_to"] = "next" if k < len(layer.plots) - 1 else "zero"


def _fix_waterfall3d(snap, spec, graph, layer, ys):
    for k, p in enumerate(layer.plots):
        p.options["depth"] = float(k)
    layer.z.title = snap.title(ys[0]) if len(ys) == 1 else "Value"
    layer.y.title = "Curve"


def _fix_bland(snap, spec, graph, layer, ys):
    a, c = snap.title(spec.x), snap.title(ys[0])
    layer.x.title = f"Mean of {a} and {c}"
    layer.y.title = f"Difference ({a} − {c})"


def _fix_windrose(snap, spec, graph, layer, ys):
    layer.options["theta_zero"] = "N"
    layer.x.reverse = True
    layer.legend.visible = False


def _fix_radar(snap, spec, graph, layer, ys):
    layer.options["theta_zero"] = "N"
    layer.x.reverse = True


def _pareto(snap, spec, t, graph, layer, ys):
    b = _b()
    if ys:
        b._build_xy(snap, spec, t, graph, layer, ys)
        layer.y.title = snap.title(ys[0])
    else:
        layer.plots.append(b._plot(t, spec, name=snap.title(spec.x), x=snap.num(spec.x)))
        layer.y.title = "Count"
    layer.x.title = snap.title(spec.x)
    layer.y2.title = "Cumulative Percent (%)"
    layer.show_y2 = True


BUILDERS = {
    "kde": _dist, "ecdf": _dist, "qq": _dist, "probability": _dist,
    "violin": _boxlike, "box_half": _boxlike, "meanbar": _boxlike, "interval": _boxlike,
    "hist_kde": _hist, "hist_rug": _hist,
    "control_xbar": _control, "control_imr": _control,
    "scatter_matrix": _scatter_matrix, "marginal": _marginal,
    "fill_area": _xy_then(_fix_fill), "waterfall3d": _xy_then(_fix_waterfall3d),
    "bland_altman": _xy_then(_fix_bland), "windrose": _xy_then(_fix_windrose),
    "radar": _xy_then(_fix_radar), "pareto": _pareto,
}

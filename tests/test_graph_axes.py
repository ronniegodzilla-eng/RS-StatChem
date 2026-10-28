"""WP-4 axes: Origin scale types, axis breaks, linked top/right axes, tick label formats,
reference lines and graph-file compatibility (old and new .rsg / .rsop content)."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy import special, stats

from rsstatchem.core.graph import scales
from rsstatchem.core.graph.export import export_bytes, figure_for, load_graph, save_graph
from rsstatchem.core.graph.model import (Column, Graph, Layer, Plot, RefLine,
                                         graph_from_json_dict, graph_to_json_dict)
from rsstatchem.core.graph.render import ref_value
from rsstatchem.core.graph.themes import apply_theme


def _graph(x, y, kind="line_symbol"):
    g = Graph(layers=[Layer()])
    kx = g.add_column(Column.from_array("x", x))
    ky = g.add_column(Column.from_array("y", y))
    p = Plot(type=kind, x=kx, y=ky, name="data")
    p.symbol.visible = True
    g.layers[0].plots.append(p)
    apply_theme(g, "Origin Default")
    return g, p


# ====================================================================== scale formulas
def test_scale_formulas_match_origin_scale_tab():
    p = np.array([0.0001, 0.1, 2.5, 50, 84.1344746, 99.9, 99.999])
    # Probability: X' = norminv(X/100)
    assert np.allclose(scales.probability(p), stats.norm.ppf(p / 100))
    assert scales.probability([50])[0] == pytest.approx(0.0)
    assert scales.probability([84.1344746])[0] == pytest.approx(1.0, abs=1e-7)
    # Probit: 5 = 50 %, one unit = one SD
    assert scales.probit([50])[0] == pytest.approx(5.0)
    assert scales.probit([84.1344746])[0] == pytest.approx(6.0, abs=1e-7)
    assert np.allclose(scales.probit_tick_positions(), 100 * stats.norm.cdf(np.arange(1, 10)
                                                                              - 5.0))
    # Logit = ln(Y/(100-Y))
    assert np.allclose(scales.logit(p), np.log(p / (100 - p)))
    # Offset reciprocal 1/(X + 273.15) (°C -> 1/K) and reciprocal 1/X
    t = np.array([-40.0, 0.0, 25.0, 100.0])
    assert np.allclose(scales.reciprocal(t, 273.15), 1 / (t + 273.15))
    assert np.allclose(scales.reciprocal([2.0, 4.0]), [0.5, 0.25])
    # Double log reciprocal (Weibull): ln(-ln(1 - X)), X as percent
    assert np.allclose(scales.weibull(p), np.log(-np.log(1 - p / 100)))
    assert scales.weibull([100 * (1 - math.exp(-1))])[0] == pytest.approx(0.0, abs=1e-12)
    # probability range is clipped to Origin's 0.0001–99.999 %
    assert np.isfinite(scales.probability([0.0, 100.0])).all()
    # inverses
    for fwd, inv in ((scales.probability, scales.probability_inv),
                     (scales.probit, scales.probit_inv), (scales.logit, scales.logit_inv),
                     (scales.weibull, scales.weibull_inv)):
        assert np.allclose(inv(fwd(p[1:-1])), p[1:-1])
    assert np.allclose(scales.reciprocal_inv(scales.reciprocal(t, 273.15), 273.15), t)
    assert special.ndtri(0.5) == 0.0


@pytest.mark.parametrize("scale", ["probability", "probit", "logit", "weibull", "reciprocal",
                                   "offset_reciprocal", "symlog", "log10", "log2", "ln"])
def test_every_scale_renders_with_ticks(scale):
    x = np.linspace(1, 99, 25)
    g, _ = _graph(x, x)
    g.layers[0].y.scale = scale
    fig, rmap = figure_for(g)
    assert rmap.errors == []
    ax = rmap.axes[0]["ax"]
    ticks = ax.yaxis.get_majorticklocs()
    lo, hi = sorted(ax.get_ylim())
    inside = [t for t in ticks if lo <= t <= hi]
    assert len(inside) >= 2, (scale, ticks)
    if scale == "probit":
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert {"4", "5", "6"} <= set(labels)
    if scale == "probability":
        assert ax.yaxis.get_scale() == "rs_probability"
        assert 50.0 in inside
    # data at 50 % sits in the middle of a symmetric probability axis
    if scale in ("probability", "logit"):
        ax.set_ylim(1, 99)
        y_disp = ax.transData.transform([(0, 50)])[0][1]
        y0 = ax.transData.transform([(0, 1)])[0][1]
        y1 = ax.transData.transform([(0, 99)])[0][1]
        assert y_disp == pytest.approx((y0 + y1) / 2, rel=1e-6)


def test_reciprocal_axis_positions_follow_one_over_x():
    t = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])
    g, _ = _graph(t, np.exp(-1 / (t + 273.15)))
    g.layers[0].x.scale = "offset_reciprocal"
    g.layers[0].x.min, g.layers[0].x.max = 0.0, 100.0
    fig, rmap = figure_for(g)
    ax = rmap.axes[0]["ax"]
    px = ax.transData.transform(np.c_[t, np.zeros(len(t))])[:, 0]
    inv = 1 / (t + 273.15)
    # display position is linear in 1/(T + 273.15)
    assert np.allclose(np.diff(px) / np.diff(inv), (np.diff(px) / np.diff(inv))[0], rtol=1e-6)


# ====================================================================== breaks
def test_break_layout_maps_segments_to_axis_fractions():
    kt, ku, gaps = scales.break_layout(0, 100, [{"from": 10, "to": 90, "position": 50}],
                                       gap=2.0)
    assert list(kt) == [0, 10, 90, 100]
    assert ku[0] == 0 and ku[-1] == 1
    assert ku[1] == pytest.approx(0.49) and ku[2] == pytest.approx(0.51)
    assert gaps == [(pytest.approx(0.49), pytest.approx(0.51))]
    # automatic positions are proportional to the kept lengths (10 and 10 of 20)
    kt, ku, gaps = scales.break_layout(0, 100, [{"from": 10, "to": 90}], gap=2.0)
    assert ku[1] == pytest.approx(0.49)
    # two breaks
    kt, ku, gaps = scales.break_layout(0, 300, [{"from": 50, "to": 100, "position": 30},
                                                {"from": 150, "to": 250, "position": 70}])
    assert len(gaps) == 2 and gaps[0][1] < gaps[1][0]


def test_axis_break_renders_gap_ticks_and_marks():
    x = np.r_[np.linspace(0, 10, 20), np.linspace(90, 100, 20)]
    g, p = _graph(x, np.sin(x))
    g.layers[0].x.breaks = [{"from": 12, "to": 88, "position": 40}]
    fig, rmap = figure_for(g)
    assert rmap.errors == []
    ax = rmap.axes[0]["ax"]
    assert ax.xaxis.get_scale() == "rs_broken"
    ticks = [t for t in ax.xaxis.get_majorticklocs() if 0 <= t <= 100]
    assert not any(12 < t < 88 for t in ticks)
    assert any(t <= 10 for t in ticks) and any(t >= 90 for t in ticks)
    frac = ax.transAxes.inverted().transform(ax.transData.transform([(12, 0), (88, 0)]))[:, 0]
    assert frac[0] == pytest.approx(0.40 - 0.01, abs=1e-6)
    assert frac[1] == pytest.approx(0.40 + 0.01, abs=1e-6)
    # a log axis can be broken as well
    g.layers[0].x.scale = "log10"
    g.layers[0].x.breaks = [{"from": 20, "to": 80}]
    fig, rmap = figure_for(g)
    assert rmap.errors == []


# ====================================================================== linked axes
def test_linked_top_axis_arrhenius():
    t = np.linspace(20, 120, 6)
    inv = 1000 / (t + 273.15)
    g, _ = _graph(inv, -5000 / (t + 273.15))
    layer = g.layers[0]
    layer.show_x2 = True
    layer.x2_formula = "1000/x - 273.15"
    layer.x2.title = "T (°C)"
    fig, rmap = figure_for(g)
    assert rmap.errors == []
    sec = rmap.axes[0]["sec_x2"]
    ax = rmap.axes[0]["ax"]
    # the top axis value at a bottom position is the formula of the bottom value
    f, finv = scales.link_functions("1000/x - 273.15", *ax.get_xlim())
    assert float(f(np.array([inv[2]]))[0]) == pytest.approx(t[2])
    assert float(finv(np.array([t[3]]))[0]) == pytest.approx(inv[3], rel=1e-4)
    assert sec.get_xlabel() == "T (°C)"
    with pytest.raises(scales.FormulaError):
        scales.compile_formula("__import__('os')")
    with pytest.raises(scales.FormulaError):
        scales.link_functions("sin(x)", -10, 10)   # not monotonic


def test_independent_top_axis_and_right_formula():
    x = np.linspace(1, 10, 10)
    g, p = _graph(x, x ** 2)
    k2 = g.add_column(Column.from_array("x2", x * 100))
    q = Plot(type="line", x=k2, y=p.y, name="top", x_axis="top")
    g.layers[0].plots.append(q)
    g.layers[0].x2.title = "Top"
    g.layers[0].y2_formula = "x/2"
    fig, rmap = figure_for(g)
    assert rmap.errors == []
    assert rmap.axes[0]["axt"].get_xlim()[1] > 500
    assert "sec_y2" in rmap.axes[0]


# ====================================================================== tick formats
def test_tick_label_formats():
    f = scales.tick_formatter("scientific", 1)
    assert f(1500.0) == "1.5E3"
    assert scales.tick_formatter("engineering")(1500.0) == "1.5k"
    assert scales.tick_formatter("power10", 1)(3000.0) == "$3\\times10^{3}$"
    assert scales.tick_formatter("power10")(100.0) == "$10^{2}$"
    assert scales.tick_formatter("decimal", 2, "$", " M")(3.14159) == "$3.14 M"
    assert scales.tick_formatter("percent")(0.25) == "25%"
    g, _ = _graph(np.arange(5.0), np.arange(5.0) * 1000)
    g.layers[0].y.tick_format = "engineering"
    g.layers[0].x.custom_ticks = [[1, "one"], [3, "three"]]
    fig, rmap = figure_for(g)
    fig.canvas.draw()
    ax = rmap.axes[0]["ax"]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["one", "three"]
    assert any(t.get_text().endswith("k") for t in ax.get_yticklabels())


# ====================================================================== reference lines
def test_reference_lines_values_and_statistics():
    rng = np.random.default_rng(4)
    y = rng.normal(10, 2, 30)
    g, p = _graph(np.arange(30.0), y)
    layer = g.layers[0]
    sd = y.std(ddof=1)
    assert ref_value(g, layer, RefLine(mode="mean"))[0] == pytest.approx(y.mean())
    assert ref_value(g, layer, RefLine(mode="median"))[0] == pytest.approx(np.median(y))
    c, lo, hi = ref_value(g, layer, RefLine(mode="mean_sd", k=2))
    assert (lo, hi) == (pytest.approx(y.mean() - 2 * sd), pytest.approx(y.mean() + 2 * sd))
    c, lo, hi = ref_value(g, layer, RefLine(mode="mean_ci"))
    half = stats.t.ppf(0.975, 29) * sd / np.sqrt(30)
    assert hi - c == pytest.approx(half)
    assert ref_value(g, layer, RefLine(kind="band", value=5, value2=2)) == [2.0, 5.0]
    layer.refs += [RefLine(mode="mean_sd", kind="band"), RefLine(axis="x", value=12,
                                                                 label="t₀")]
    fig, rmap = figure_for(g)
    assert rmap.errors == []
    assert sum(1 for _, ref in rmap.artists if ref[0] == "ref") >= 4


def test_plot_details_fill_band_drops_labels():
    x = np.linspace(0, 6, 25)
    g, p = _graph(x, np.sin(x) + 2)
    ke = g.add_column(Column.from_array("e", np.full(25, 0.2)))
    q = Plot(type="line", x=p.x, y=g.add_column(Column.from_array("y2", np.cos(x) + 2)),
             name="cos")
    g.layers[0].plots.append(q)
    p.yerr = ke
    p.options.update(error_band=True, fill_to="next", drop_lines="x", data_labels="y",
                     label_format="{:.1f}", skip=3)
    q.options.update(fill_to="zero", fill_two_colors=True)
    fig, rmap = figure_for(g)
    assert rmap.errors == []
    texts = [a for a, ref in rmap.artists if ref == ("plot", 0, p.id)
             and hasattr(a, "get_text")]
    assert len(texts) == 25 and texts[0].get_text() == f"{np.sin(0) + 2:.1f}"


# ====================================================================== file compatibility
def test_old_graph_files_still_load_and_new_fields_round_trip(tmp_path):
    g, p = _graph(np.arange(5.0), np.arange(5.0) ** 2)
    d = graph_to_json_dict(g)
    # an old (pre-WP-4) file has none of the new keys
    old = json.loads(json.dumps(d))
    for key in ("x2", "show_x2", "x2_formula", "y2_formula", "refs"):
        old["graph"]["layers"][0].pop(key)
    for which in ("x", "y", "y2", "z"):
        for key in ("scale_offset", "breaks", "break_gap", "tick_format", "prefix", "suffix",
                    "custom_ticks"):
            old["graph"]["layers"][0][which].pop(key)
    old["graph"]["layers"][0]["plots"][0].pop("x_axis")
    back = graph_from_json_dict(old)
    assert back.layers[0].refs == [] and back.layers[0].x.breaks == []
    assert back.layers[0].x.scale_offset == 273.15
    assert export_bytes(back, "png", dpi=40)[:4] == b"\x89PNG"
    # new content survives .rsg save/load
    layer = g.layers[0]
    layer.y.scale = "probability"
    layer.x.breaks = [{"from": 1.2, "to": 2.8, "position": 30.0}]
    layer.refs.append(RefLine(mode="mean_sd", kind="band", k=1.5))
    layer.show_x2, layer.x2_formula = True, "x*2"
    layer.x.tick_format, layer.x.suffix = "decimal", " s"
    p.options.update(fill_to="zero", data_labels="xy")
    path = save_graph(g, tmp_path / "new")
    loaded = load_graph(path)
    assert graph_to_json_dict(loaded) == graph_to_json_dict(g)
    assert loaded.layers[0].refs[0].k == 1.5
    fig, rmap = figure_for(loaded)
    assert rmap.errors == []

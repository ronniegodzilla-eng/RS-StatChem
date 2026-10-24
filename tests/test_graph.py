"""Graph Studio: model serialisation, plot building, rendering and export."""
from __future__ import annotations

import io
import json
import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.graph import (TEMPLATES, Graph, PlotSpec, apply_theme, build_graph,
                                   export_bytes, export_graph, figure_for, from_dict,
                                   graph_from_json_dict, graph_to_json_dict, load_graph,
                                   save_graph, to_dict)
from rsstatchem.core.graph import edit, themes
from rsstatchem.core.graph.gridding import xyz_grid
from rsstatchem.core.graph.model import Column
from rsstatchem.core.graph.ternary import to_xy
from rsstatchem.core.graph.text import rich
from rsstatchem.io.files import load_project

EXAMPLE = os.path.join(os.path.dirname(__file__), "..", "examples", "contoh_penelitian.rsd")


@pytest.fixture(scope="module")
def ds() -> Dataset:
    return load_project(EXAMPLE)


@pytest.fixture(scope="module")
def ds_xyz() -> Dataset:
    rng = np.random.default_rng(3)
    n = 60
    xx, yy = np.meshgrid(np.linspace(-2, 2, 6), np.linspace(-1, 1, 5))
    df = pd.DataFrame({"rx": rng.uniform(-2, 2, n), "ry": rng.uniform(-2, 2, n)})
    df["rz"] = np.sin(df.rx) * np.cos(df.ry)
    df["a"], df["b"], df["c"] = rng.uniform(0, 1, (3, n))
    df["theta"] = np.linspace(0, 360, n)
    df["r"] = 1 + 0.3 * np.cos(np.deg2rad(df.theta))
    df["gx"] = np.r_[xx.ravel(), np.full(n - 30, np.nan)]
    df["gy"] = np.r_[yy.ravel(), np.full(n - 30, np.nan)]
    df["gz"] = df.gx ** 2 + df.gy
    return Dataset(df)


def _spec(key: str) -> tuple[str, PlotSpec]:
    """A sensible spec for each template: (which dataset, spec)."""
    xyz = {"contour": ("gx", "gy", "gz"), "heatmap": ("gx", "gy", "gz"),
           "surface3d": ("rx", "ry", "rz"), "scatter3d": ("rx", "ry", "rz"),
           "ternary": ("a", "b", "c")}
    if key in xyz:
        x, y, z = xyz[key]
        return "xyz", PlotSpec(key, x=x, ys=[y], z=z)
    if key == "polar":
        return "xyz", PlotSpec(key, x="theta", ys=["r"])
    if key.startswith("area"):
        return "xyz", PlotSpec(key, x="theta", ys=["a", "b", "c"])
    if key in ("pie", "doughnut"):
        return "ds", PlotSpec(key, x="motivasi")
    if key.startswith(("column", "bar")):
        return "ds", PlotSpec(key, x="motivasi", ys=["posttest"], group="kelompok",
                              summary="mean_se")
    if key in ("histogram", "box"):
        return "ds", PlotSpec(key, ys=["pretest", "posttest"], group="kelompok")
    if key in ("bubble", "color_scatter", "bubble_color"):
        return "ds", PlotSpec(key, x="pretest", ys=["posttest"], z="usia")
    if key in ("double_y", "multi_panel", "waterfall"):
        return "ds", PlotSpec(key, x="pretest", ys=["posttest", "usia"])
    return "ds", PlotSpec(key, x="pretest", ys=["posttest"], group="kelompok")


@pytest.mark.parametrize("key", list(TEMPLATES))
def test_every_plot_type_renders_and_round_trips(key, ds, ds_xyz, tmp_path):
    which, spec = _spec(key)
    data = ds if which == "ds" else ds_xyz
    graph = build_graph(data, spec)
    fig, rmap = figure_for(graph)
    assert rmap.errors == []
    assert rmap.artists, "nothing was drawn"
    png = export_bytes(graph, "png", dpi=80)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    d = graph_to_json_dict(graph)
    text = json.dumps(d, allow_nan=False)            # strictly valid JSON (no NaN)
    back = graph_from_json_dict(json.loads(text))
    assert graph_to_json_dict(back) == d
    path = save_graph(graph, tmp_path / "g")
    assert path.suffix == ".rsg"
    assert graph_to_json_dict(load_graph(path)) == d
    assert export_bytes(back, "png", dpi=80)[:8] == png[:8]


def test_snapshot_survives_without_dataset(ds, tmp_path):
    g = build_graph(ds, PlotSpec("scatter", x="pretest", ys=["posttest"]))
    path = save_graph(g, tmp_path / "snap.rsg")
    g2 = load_graph(path)
    p = g2.layers[0].plots[0]
    np.testing.assert_allclose(g2.values(p.y), ds.df["posttest"].to_numpy())
    np.testing.assert_allclose(g2.values(p.x), ds.df["pretest"].to_numpy())


def test_user_missing_values_are_honoured(ds):
    assert 99.0 in ds.var("usia").missing
    df = ds.df.copy()
    ds2 = Dataset(df, [v for v in ds.variables])
    ds2.df.loc[0, "usia"] = 99.0
    g = build_graph(ds2, PlotSpec("scatter", x="pretest", ys=["usia"]))
    y = g.values(g.layers[0].plots[0].y)
    assert np.isnan(y[0])
    assert not np.any(y == 99.0)


def test_group_by_splits_series(ds):
    g = build_graph(ds, PlotSpec("scatter", x="pretest", ys=["posttest"], group="kelompok"))
    plots = g.layers[0].plots
    assert [p.name for p in plots] == ["Kontrol", "Eksperimen"]
    counts = ds.df["kelompok"].value_counts()
    assert len(g.values(plots[0].y)) == counts[1.0]
    assert len(g.values(plots[1].y)) == counts[2.0]
    assert g.layers[0].legend.visible


@pytest.mark.parametrize("summary", ["mean", "mean_sd", "mean_se", "mean_ci", "median", "sum",
                                     "count"])
def test_summaries_match_pandas(ds, summary):
    g = build_graph(ds, PlotSpec("column", x="motivasi", ys=["posttest"], summary=summary))
    p = g.layers[0].plots[0]
    grouped = ds.df.groupby("motivasi")["posttest"]
    ref_center = {"mean": grouped.mean(), "mean_sd": grouped.mean(), "mean_se": grouped.mean(),
                  "mean_ci": grouped.mean(), "median": grouped.median(), "sum": grouped.sum(),
                  "count": grouped.count()}[summary]
    np.testing.assert_allclose(g.values(p.x), ref_center.index.to_numpy())
    np.testing.assert_allclose(g.values(p.y), ref_center.to_numpy())
    if summary in ("mean_sd", "mean_se", "mean_ci"):
        from scipy import stats
        sd, n = grouped.std(ddof=1), grouped.count()
        ref = {"mean_sd": sd, "mean_se": sd / np.sqrt(n),
               "mean_ci": stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)}[summary]
        np.testing.assert_allclose(g.values(p.yerr), ref.to_numpy())
    labels = g.column(p.x).value_labels
    assert labels[1.0] == "Sangat rendah"


def test_summary_by_variable_without_x(ds):
    g = build_graph(ds, PlotSpec("column", ys=["pretest", "posttest"], summary="mean_sd"))
    p = g.layers[0].plots[0]
    np.testing.assert_allclose(g.values(p.y), [ds.df.pretest.mean(), ds.df.posttest.mean()])
    assert g.column(p.x).value_labels == {1.0: "Nilai pre-test", 2.0: "Nilai post-test"}


def test_percent_stack_sums_to_100(ds):
    g = build_graph(ds, PlotSpec("column_percent", x="motivasi", summary="count",
                                 group="kelompok"))
    _, rmap = figure_for(g)
    tops = [pts[4] for pts in rmap.points]       # cumulative tops of each series
    np.testing.assert_allclose(tops[-1], 100.0)


def test_histogram_counts_match_numpy(ds):
    g = build_graph(ds, PlotSpec("histogram", ys=["posttest"], options={"bins": 8}))
    _, rmap = figure_for(g)
    counts = rmap.points[0][4]
    ref, _ = np.histogram(ds.df.posttest.dropna(), bins=8)
    np.testing.assert_array_equal(counts, ref)


def test_double_y_and_multipanel_structure(ds):
    g = build_graph(ds, PlotSpec("double_y", x="pretest", ys=["posttest", "usia"]))
    layer = g.layers[0]
    assert layer.show_y2 and [p.y_axis for p in layer.plots] == ["left", "right"]
    _, rmap = figure_for(g)
    assert rmap.axes[0]["ax2"] is not None
    g = build_graph(ds, PlotSpec("multi_panel", x="pretest", ys=["posttest", "usia",
                                                                 "motivasi", "pretest"]))
    assert len(g.layers) == 4 and (g.rows, g.cols) == (2, 2)
    assert [l.panel_label for l in g.layers] == ["(a)", "(b)", "(c)", "(d)"]


def test_waterfall_offsets(ds_xyz):
    g = build_graph(ds_xyz, PlotSpec("waterfall", x="theta", ys=["a", "b"]))
    layer = g.layers[0]
    assert layer.offset_y > 0
    _, rmap = figure_for(g)
    raw = g.values(layer.plots[1].y)
    np.testing.assert_allclose(rmap.points[1][4], raw + layer.offset_y)


def test_gridding_regular_and_interpolated():
    x, y = np.meshgrid([0.0, 1.0, 2.0], [10.0, 20.0])
    z = x * 2 + y
    order = np.random.default_rng(0).permutation(6)
    X, Y, Z, regular = xyz_grid(x.ravel()[order], y.ravel()[order], z.ravel()[order])
    assert regular
    np.testing.assert_allclose(Z, z)
    rng = np.random.default_rng(1)
    xs, ys = rng.uniform(0, 1, 200), rng.uniform(0, 1, 200)
    X, Y, Z, regular = xyz_grid(xs, ys, 3 * xs - 2 * ys, n=30)
    assert not regular
    ok = np.isfinite(Z)
    np.testing.assert_allclose(Z[ok], 3 * X[ok] - 2 * Y[ok], atol=1e-9)
    X, Y, Z, _ = xyz_grid(xs, ys, 3 * xs - 2 * ys, n=30, fill_outside=True)
    assert np.isfinite(Z).all()


def test_ternary_transform():
    x, y = to_xy([1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1])
    np.testing.assert_allclose(x, [0, 1, 0.5, 0.5])
    np.testing.assert_allclose(y, [0, 0, np.sqrt(3) / 2, np.sqrt(3) / 6])
    # unnormalised components are normalised
    x2, y2 = to_xy([2], [2], [4])
    x3, y3 = to_xy([0.25], [0.25], [0.5])
    np.testing.assert_allclose([x2[0], y2[0]], [x3[0], y3[0]])


def test_rich_text():
    assert rich("H$_2$O") == "H$_2$O"
    assert rich(r"cm\+(-1)") == "cm$^{-1}$"
    assert rich(r"H\-(2)O") == "H$_{2}$O"
    assert rich(r"\g(a)-helix") == r"$\alpha$-helix"
    assert rich("costs $5") == r"costs \$5"                 # unbalanced → literal
    assert rich(r"$\notacommand{$") == r"\$\notacommand{\$"  # invalid mathtext → literal


@pytest.mark.parametrize("fmt,magic", [("png", b"\x89PNG"), ("jpg", b"\xff\xd8"),
                                       ("tif", b"II*\x00"), ("pdf", b"%PDF"),
                                       ("eps", b"%!PS"), ("svg", b"<?xml")])
def test_export_formats(ds, fmt, magic, tmp_path):
    g = build_graph(ds, PlotSpec("line_symbol", x="pretest", ys=["posttest"],
                                 group="kelompok"))
    g.layers[0].y.title = "Absorbance (cm$^{-1}$)"
    path = export_graph(g, tmp_path / f"out.{fmt}", dpi=300)
    data = path.read_bytes()
    assert data.startswith(magic) or (fmt == "tif" and data.startswith(b"MM\x00*"))


@pytest.mark.parametrize("dpi", [300, 600, 1200])
def test_raster_size_and_dpi(ds, dpi):
    g = build_graph(ds, PlotSpec("scatter", x="pretest", ys=["posttest"],
                                 theme="Journal - Single Column (8.3 cm)"))
    for fmt in ("png", "tif"):
        im = Image.open(io.BytesIO(export_bytes(g, fmt, dpi=dpi)))
        assert abs(im.width - 8.3 / 2.54 * dpi) <= 2
        assert round(im.info["dpi"][0]) == dpi
    im = Image.open(io.BytesIO(export_bytes(g, "png", dpi=300, width_cm=17.4)))
    assert abs(im.width - 17.4 / 2.54 * 300) <= 2


def test_pdf_fonts_are_truetype(ds):
    g = build_graph(ds, PlotSpec("scatter", x="pretest", ys=["posttest"]))
    pdf = export_bytes(g, "pdf")
    assert b"/FontFile2" in pdf and b"/Type3" not in pdf


def test_themes(ds):
    g = build_graph(ds, PlotSpec("line_symbol", x="pretest", ys=["posttest"],
                                 group="kelompok"))
    assert g.layers[0].x.tick_dir == "in" and g.layers[0].box and not g.layers[0].x.grid_major
    apply_theme(g, "Journal - Double Column (17.4 cm)")
    assert g.page.width_cm == 17.4 and g.layers[0].x.label_font.size == 8.5
    apply_theme(g, "Grayscale")
    assert g.palette == "Grayscale" and g.dash_cycle
    assert all(p.line.style == "auto" for p in g.layers[0].plots)
    assert export_bytes(g, "png", dpi=60)


def test_user_templates(ds, tmp_path, monkeypatch):
    monkeypatch.setenv("RSSTATCHEM_HOME", str(tmp_path))
    src = build_graph(ds, PlotSpec("scatter", x="pretest", ys=["posttest"]))
    src.layers[0].plots[0].symbol.shape = "D"
    src.layers[0].plots[0].symbol.face = "#00ff00"
    src.layers[0].x.tick_dir = "out"
    src.page.width_cm = 10.0
    themes.save_user_template("My style", src)
    assert "My style" in themes.list_user_templates()
    assert (tmp_path / "graph_templates").is_dir()
    target = build_graph(ds, PlotSpec("line_symbol", x="usia", ys=["pretest"]))
    title = target.layers[0].x.title
    themes.apply_template(target, themes.load_user_template("My style"))
    assert target.page.width_cm == 10.0
    assert target.layers[0].x.tick_dir == "out" and target.layers[0].x.title == title
    p = target.layers[0].plots[0]
    assert p.symbol.shape == "D" and p.symbol.face == "#00ff00" and p.line.visible
    themes.delete_user_template("My style")
    assert "My style" not in themes.list_user_templates()


def test_layer_editing(ds):
    g = build_graph(ds, PlotSpec("scatter", x="pretest", ys=["posttest"]))
    edit.add_panel(g, "below")
    edit.add_panel(g, "right")
    edit.add_inset(g)
    assert len(g.layers) == 4 and (g.rows, g.cols) == (2, 2)
    other = build_graph(ds, PlotSpec("line", x="pretest", ys=["usia"]))
    ids = edit.merge_plots(g, g.layers[1], other)
    assert len(ids) == 1 and g.layers[1].plots[0].id == ids[0]
    _, rmap = figure_for(g)
    assert rmap.errors == [] and len(rmap.axes) == 4
    edit.remove_layer(g, 2)
    assert len(g.layers) == 3
    with pytest.raises(ValueError):
        edit.merge_plots(g, g.layers[0], build_graph(ds, PlotSpec("pie", x="motivasi")))
    single = build_graph(ds, PlotSpec("scatter", x="pretest", ys=["posttest"]))
    with pytest.raises(ValueError):
        edit.remove_layer(single, 0)


def test_annotations_and_axis_options_render(ds):
    g = build_graph(ds, PlotSpec("scatter", x="pretest", ys=["posttest"]))
    layer = g.layers[0]
    for kind in ("text", "arrow", "line", "rect"):
        layer.annotations.append(edit.new_annotation(kind, 0.1, 0.2, 0.4, 0.6))
    layer.x.scale, layer.y.reverse, layer.x.min, layer.x.max = "log10", True, 10, 100
    layer.x.tick_dir, layer.y.grid_major, layer.y.major_step = "both", True, 10
    layer.legend.loc, layer.legend.x, layer.legend.y = "custom", 0.1, 0.1
    fig, rmap = figure_for(g)
    assert rmap.errors == []
    ax = rmap.axes[0]["ax"]
    assert ax.get_xscale() == "log" and ax.get_xlim() == (10, 100)
    lo, hi = ax.get_ylim()
    assert lo > hi
    assert sum(1 for _, ref in rmap.artists if ref[0] == "annotation") == 4


def test_validation_messages(ds):
    assert "needs" in PlotSpec("contour", x="pretest", ys=["posttest"]).validate(ds)
    assert "two Y" in PlotSpec("double_y", x="pretest", ys=["posttest"]).validate(ds)
    assert "numeric" in PlotSpec("line", ys=["responden"]).validate(ds)
    with pytest.raises(ValueError):
        build_graph(ds, PlotSpec("ternary", x="pretest"))


def test_from_dict_ignores_unknown_and_missing_fields():
    g = from_dict(Graph, {"title": "G", "future_field": 1, "layers": [{"kind": "polar"}]})
    assert g.title == "G" and g.layers[0].kind == "polar" and g.page.width_cm == 12.0
    col = Column.from_array("v", np.array([1.0, np.nan]))
    assert to_dict(col)["values"] == [1.0, None]


def test_module_selftest():
    from rsstatchem.ui.graph import MENUS, selftest
    selftest()
    entries = [e for e in MENUS["Graphs"] if e]
    assert entries[0][0].startswith("Graph Studio") and callable(entries[0][1])
    assert MENUS["File"][0][0] == "Open Graph..."


def test_graph_window_offscreen(ds, tmp_path, monkeypatch, qapp):
    monkeypatch.setenv("RSSTATCHEM_HOME", str(tmp_path))
    from matplotlib.backend_bases import MouseEvent
    app = qapp
    from rsstatchem.ui.graph.setup_dialog import PlotSetupDialog, blocks_visible
    from rsstatchem.ui.graph.window import WINDOWS, GraphWindow

    dlg = PlotSetupDialog(ds, None, kind="contour")
    assert set(blocks_visible(dlg)) == {"x", "y", "z"}
    dlg.reset()
    for role, name in (("x", "pretest"), ("y", "posttest"), ("group", "kelompok")):
        dlg._take(name, next(t for t in dlg._targets if t.key == role))
    dlg.kind_combo.setCurrentIndex(dlg.kind_combo.findData("scatter"))
    assert dlg.validate() is None
    spec = dlg.spec()
    assert (spec.kind, spec.x, spec.ys, spec.group) == ("scatter", "pretest", ["posttest"],
                                                        "kelompok")
    win = GraphWindow(build_graph(ds, spec))
    win.show()
    app.processEvents()
    win.canvas.draw()
    ax = win.rmap.axes[0]["ax"]
    bb = ax.xaxis.label.get_window_extent(win.canvas.get_renderer())
    ev = MouseEvent("button_press_event", win.canvas, (bb.x0 + bb.x1) / 2,
                    (bb.y0 + bb.y1) / 2, button=1)
    assert win.hit_test(ev)[0] == ("axis", 0, "x")
    win.select(("axis", 0, "y"))
    win.apply_change(win.graph.layers[0].y, "min", 0.0)
    win.redraw()
    assert win.rmap.axes[0]["ax"].get_ylim()[0] == 0.0
    win.undo()
    assert win.graph.layers[0].y.min is None
    section = win.output_section()
    assert section.items[-1].png[:4] == b"\x89PNG"
    win.modified = False
    win.close()
    app.processEvents()
    assert win not in WINDOWS


def test_property_forms_for_every_object(ds, ds_xyz, qapp):
    from rsstatchem.ui.graph.properties import PropertyForm, schema_for
    from rsstatchem.ui.graph.window import GraphWindow, REF_ROLE

    for key in TEMPLATES:
        which, spec = _spec(key)
        g = build_graph(ds if which == "ds" else ds_xyz, spec)
        if key == "line":
            edit.add_inset(g)
            g.layers[0].annotations.append(edit.new_annotation("arrow", 0, 0, 1, 1))
        win = GraphWindow(g)
        refs = [tuple(item.data(0, REF_ROLE)) for item in win._iter_items()]
        assert ("graph",) in refs and any(r[0] == "layer" for r in refs)
        for ref in refs:
            obj = win.resolve(ref)
            sections, layer = schema_for(obj, ref, g)
            PropertyForm(obj, sections, lambda p, v: None, layer)
            win.show_properties(ref)
        win.modified = False
        win.close()

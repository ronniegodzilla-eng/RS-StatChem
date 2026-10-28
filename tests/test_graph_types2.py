"""WP-4 graph types and graph management: statistical plots checked against scipy and hand
calculations, 3D/specialised types, layer management, layout page, gadgets, digitizer and
the Origin menus installed by ``graph_menus.install``."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from rsstatchem.core.graph import gadgets as gd
from rsstatchem.core.graph import layout, stat_plots as sp, themes
from rsstatchem.core.graph.export import export_bytes, export_many, figure_for
from rsstatchem.core.graph.model import (Column, Graph, Layer, Plot, graph_from_json_dict,
                                         graph_to_json_dict)
from rsstatchem.ui.origin.model import Workbook
from rsstatchem.ui.origin.plotting import PLOT_TYPES, plot_columns

NEW_KINDS = [k for k in PLOT_TYPES if k not in (
    "line", "scatter", "line_symbol", "yerror", "column", "column_stacked", "column_percent",
    "bar", "bar_stacked", "area", "area_stacked", "pie", "doughnut", "histogram", "box",
    "double_y", "stack", "waterfall", "bubble", "color_scatter", "contour", "heatmap",
    "scatter3d", "surface3d", "polar", "ternary")]


def _book(n: int = 40) -> Workbook:
    rng = np.random.default_rng(1)
    book = Workbook.new("Book1")
    ws = book.sheets[0]
    x = np.linspace(0, 10, n)
    ws.set_column_values(0, list(x))
    ws.set_column_values(1, list(np.sin(x) * 3 + 10 + rng.normal(0, .4, n)))
    ws.columns[1].long_name = "Signal A"
    ws.add_column("yEr")
    ws.set_column_values(2, list(rng.uniform(.2, .6, n)))
    ws.add_column("Y")
    ws.set_column_values(3, list(np.cos(x) * 2 + 9 + rng.normal(0, .4, n)))
    ws.columns[3].long_name = "Signal B"
    ws.add_column("Y")
    ws.set_column_values(4, list(rng.gamma(4, 2, n)))
    return book


# ====================================================================== numeric checks
def test_qq_and_probability_positions_match_scipy_probplot():
    rng = np.random.default_rng(0)
    v = rng.normal(5, 2, 37)
    (osm, osr), (slope, icpt, r) = stats.probplot(v, dist="norm", fit=True)
    q_osm, q_osr, (s2, i2, r2) = sp.qq_data(v)
    assert np.allclose(q_osm, osm) and np.allclose(q_osr, osr)
    assert (s2, i2, r2) == (pytest.approx(slope), pytest.approx(icpt), pytest.approx(r))
    x, pct = sp.probability_data(v)
    assert np.allclose(stats.norm.ppf(pct / 100), osm)
    n = 10
    i = np.arange(1, n + 1)
    assert np.allclose(sp.plotting_positions(n, "blom"), (i - 0.375) / (n + 0.25))
    assert np.allclose(sp.plotting_positions(n, "benard"), (i - 0.3) / (n + 0.4))
    assert np.allclose(sp.plotting_positions(n, "hazen"), (i - 0.5) / n)
    assert np.allclose(sp.plotting_positions(n, "vdw"), i / (n + 1))


def test_kde_matches_gaussian_kde():
    rng = np.random.default_rng(2)
    v = rng.gamma(3, 1.5, 80)
    xs, d = sp.kde_curve(v, "silverman")
    ref = stats.gaussian_kde(v, bw_method="silverman")(xs)
    assert np.allclose(d, ref)
    X, Y, Z = sp.kde2d(v, v * 2 + rng.normal(0, 1, 80), n=20)
    kde = stats.gaussian_kde(np.vstack([v, v * 2 + rng.normal(0, 1, 80)]))
    assert Z.shape == (20, 20) and np.all(Z >= 0)
    assert np.trapezoid(np.trapezoid(Z, X[0]), Y[:, 0]) == pytest.approx(1.0, abs=0.05)
    del kde


def test_bland_altman_limits_by_hand():
    a = np.array([10.1, 12.0, 9.8, 11.5, 13.2, 10.9])
    b = np.array([10.0, 12.4, 9.5, 11.0, 13.0, 11.3])
    ba = sp.bland_altman(a, b)
    diff = a - b
    bias, sd = diff.mean(), diff.std(ddof=1)
    assert ba["bias"] == pytest.approx(bias)
    assert ba["lower"] == pytest.approx(bias - 1.96 * sd)
    assert ba["upper"] == pytest.approx(bias + 1.96 * sd)
    assert np.allclose(ba["mean"], (a + b) / 2)


def test_control_chart_limits():
    rng = np.random.default_rng(3)
    v = rng.normal(50, 2, 100)
    c = sp.control_limits(v, 5, "xbar")
    g = v.reshape(20, 5)
    rbar = np.ptp(g, axis=1).mean()
    assert c["cl"] == pytest.approx(g.mean())
    assert c["ucl"] == pytest.approx(g.mean() + 0.577 * rbar)
    r = sp.control_limits(v, 5, "r")
    assert r["ucl"] == pytest.approx(2.114 * rbar) and r["lcl"] == 0
    s = sp.control_limits(v, 5, "s")
    c4 = 0.9400   # Montgomery Appendix VI, n = 5
    assert sp._c4(5) == pytest.approx(c4, abs=1e-4)
    assert s["cl"] == pytest.approx(g.std(axis=1, ddof=1).mean())
    i = sp.control_limits(v, chart="i")
    mr = np.abs(np.diff(v)).mean()
    assert i["ucl"] == pytest.approx(v.mean() + 2.66 * mr)


def test_pareto_mean_interval_ecdf():
    labels, vals, cum = sp.pareto_data(["a", "b", "c"], [2, 5, 3])
    assert labels == ["b", "c", "a"] and list(cum) == [50.0, 80.0, 100.0]
    m, half, n = sp.mean_interval([1, 2, 3, 4], "ci")
    assert half == pytest.approx(stats.t.ppf(0.975, 3) * np.std([1, 2, 3, 4], ddof=1) / 2)
    x, f = sp.ecdf([3, 1, 2])
    assert list(x) == [1, 2, 3] and f[-1] == 1.0


# ====================================================================== plot types
@pytest.mark.parametrize("kind", NEW_KINDS)
def test_new_plot_types_render_without_errors_and_round_trip(kind):
    book = _book()
    ws = book.sheets[0]
    cols = {"radar": [1, 3, 4], "scatter_matrix": [1, 3, 4], "windrose": [1],
            "pareto": [4]}.get(kind, [1, 3])
    g = plot_columns(book, ws, cols, kind, "G")
    fig, rmap = figure_for(g)
    assert rmap.errors == [], (kind, rmap.errors)
    assert rmap.artists
    back = graph_from_json_dict(graph_to_json_dict(g))
    assert graph_to_json_dict(back) == graph_to_json_dict(g)
    assert export_bytes(back, "png", dpi=40)[:4] == b"\x89PNG"


def test_layouts_of_multi_layer_types():
    book = _book()
    ws = book.sheets[0]
    g = plot_columns(book, ws, [1, 3, 4], "scatter_matrix", "G")
    assert len(g.layers) == 9 and g.rows == g.cols == 3
    assert g.layers[0].plots[0].type == "histogram"
    g = plot_columns(book, ws, [1], "control_xbar", "G")
    assert [p.options["chart"] for layer in g.layers for p in layer.plots] == ["xbar", "r"]
    g = plot_columns(book, ws, [1, 3], "marginal", "G")
    assert g.layers[1].options["link_x"] == 0 and g.layers[2].options["link_y"] == 0
    fig, rmap = figure_for(g)
    assert rmap.axes[1]["ax"].get_xlim() == pytest.approx(rmap.axes[0]["ax"].get_xlim())
    g = plot_columns(book, ws, [1, 3, 4], "panel_4", "G")
    assert len(g.layers) == 4 and [len(l.plots) for l in g.layers] == [1, 1, 1, 0]
    g = plot_columns(book, ws, [1], "probability", "G")
    assert g.layers[0].y.scale == "probability"


def test_violin_and_histogram_options_render():
    g = Graph(layers=[Layer()])
    rng = np.random.default_rng(5)
    k = g.add_column(Column.from_array("v", rng.lognormal(1, .4, 60)))
    for opts in ({"inner": "quartile", "side": "left"}, {"inner": "points", "side": "right"},
                 {"inner": "none", "bw": "0.3"}):
        g.layers[0].plots = [Plot(type="violin", y=k, name="v", options=opts)]
        assert figure_for(g)[1].errors == []
    for curve in ("lognormal", "exponential", "gamma", "weibull", "kde", "none"):
        g.layers[0].plots = [Plot(type="histogram", y=k, options={"dist_curve": curve,
                                                                  "rug": True})]
        assert figure_for(g)[1].errors == [], curve


# ====================================================================== layout
def _two_graphs():
    book = _book()
    ws = book.sheets[0]
    return (plot_columns(book, ws, [1], "line", "A"),
            plot_columns(book, ws, [3], "scatter", "B"))


def test_merge_extract_arrange_and_layout_page():
    a, b = _two_graphs()
    m = layout.merge_graphs([a, b])
    assert len(m.layers) == 2 and m.cols == 2
    assert all(m.values(p.y) is not None for l in m.layers for p in l.plots)
    assert figure_for(m)[1].errors == []
    parts = layout.extract_layers(m)
    assert len(parts) == 2 and parts[1].layers[0].plots[0].type == "scatter"
    layout.arrange(m, 2, 1, hspace=0.1)
    assert [l.grid for l in m.layers] == [[0, 0, 1, 1], [1, 0, 1, 1]]
    with pytest.raises(ValueError):
        layout.arrange(m, 1, 1)
    page = layout.layout_page([a, b], cols=1, page="A4 portrait", heading="Figure 1")
    assert (page.page.width_cm, page.page.height_cm) == (21.0, 29.7)
    assert all(l.rect for l in page.layers)
    assert figure_for(page)[1].errors == []
    layout.link_layer(m, 1, 0, x=True)
    assert m.layers[1].options["link_x"] == 0
    j = layout.move_layer(m, 0, 1)
    assert j == 1 and m.layers[0].plots[0].type == "scatter"


def test_exchange_xy_and_format_copy():
    a, b = _two_graphs()
    layer = a.layers[0]
    p = layer.plots[0]
    x0, y0 = p.x, p.y
    xt, yt = layer.x.title, layer.y.title
    layout.exchange_xy(a, layer)
    assert (p.x, p.y) == (y0, x0) and (layer.x.title, layer.y.title) == (yt, xt)
    b.layers[0].plots[0].symbol.size = 13
    b.layers[0].plots[0].line.color = "#123456"
    fmt = layout.format_of(b.layers[0].plots[0])
    layout.apply_format(p, fmt)
    assert p.symbol.size == 13 and p.line.color == "#123456"
    lf = layout.format_of(b.layers[0])
    b.layers[0].x.tick_dir = "both"
    layout.apply_format(layer, layout.format_of(b.layers[0]))
    assert layer.x.tick_dir == "both" and layer.x.title == yt
    assert lf["kind"] == "layer"


def test_user_theme_and_batch_export(tmp_path, monkeypatch):
    monkeypatch.setenv("RSSTATCHEM_HOME", str(tmp_path / "home"))
    a, b = _two_graphs()
    a.layers[0].x.tick_dir = "both"
    a.palette = "Bright"
    themes.save_user_theme("Mine", a)
    assert "Mine" in themes.list_user_themes()
    themes.apply_user_theme(b, "Mine")
    assert b.palette == "Bright" and b.layers[0].y.tick_dir == "both"
    paths = export_many({"A": a, "B/b": b}, tmp_path / "out", "png", dpi=50)
    assert [p.name for p in paths] == ["A.png", "B_b.png"] and all(p.stat().st_size for p in
                                                                    paths)


# ====================================================================== gadgets / digitizer
def test_gadget_computations():
    x = np.linspace(-5, 5, 2001)
    y = 3 * np.exp(-x ** 2 / 2)
    r = gd.integrate(x, y, -5, 5)
    assert r["Area"] == pytest.approx(3 * np.sqrt(2 * np.pi), rel=1e-5)
    assert r["FWHM"] == pytest.approx(2 * np.sqrt(2 * np.log(2)), rel=1e-4)
    assert r["Centroid"] == pytest.approx(0, abs=1e-9)
    base = gd.integrate(x, y + 1, -2, 2, baseline="line")
    assert base["Area"] < gd.integrate(x, y + 1, -2, 2)["Area"]
    s = gd.statistics(x, y, 0, 5)
    sel = y[(x >= 0) & (x <= 5)]
    assert s["N"] == len(sel) and s["Mean"] == pytest.approx(sel.mean())
    assert s["Max"] == pytest.approx(3.0)
    t = np.linspace(0, 1, 1000, endpoint=False)
    f = gd.fft(t, np.sin(2 * np.pi * 50 * t), 0, 1)
    assert f["Dominant Frequency"] == pytest.approx(50, abs=1.01)
    pts = gd.intersect(x, y, x, np.full(len(x), 1.5))
    assert [p[0] for p in pts] == [pytest.approx(-np.sqrt(2 * np.log(2)), abs=1e-3),
                                   pytest.approx(np.sqrt(2 * np.log(2)), abs=1e-3)]
    step = np.where(t > 0.5, 1.0, 0.0) * (1 - np.exp(-(t - 0.5).clip(0) * 40))
    rt = gd.rise_time(t, step, 0, 1)
    assert rt["Rise Time"] == pytest.approx(np.log(9) / 40, rel=0.05)
    assert gd.cursor(x, y, 0.0)["Y (interpolated)"] == pytest.approx(3.0)
    masked = gd.statistics(np.arange(5.0), np.array([1, 2, 100, 3, 4.0]), 0, 4,
                           mask=[False, False, True, False, False])
    assert masked["Max"] == 4.0


def test_digitizer_calibration_linear_log_and_rotated():
    cal = gd.Calibration(((100, 400), (500, 400)), (0, 10), ((100, 400), (100, 0)), (0, 20))
    assert cal.to_data(300, 200) == (pytest.approx(5.0), pytest.approx(10.0))
    cal = gd.Calibration(((100, 400), (500, 400)), (1, 1000), ((100, 400), (100, 0)),
                         (0, 1), x_log=True)
    assert cal.to_data(300, 400)[0] == pytest.approx(np.sqrt(1000))
    # a scan rotated by 10° gives the same data
    th = np.deg2rad(10)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    rot = lambda p: tuple(R @ np.array(p, float))  # noqa: E731
    cal = gd.Calibration((rot((100, 400)), rot((500, 400))), (0, 10),
                         (rot((100, 400)), rot((100, 0))), (0, 20))
    assert cal.to_data(*rot((300, 200))) == (pytest.approx(5.0), pytest.approx(10.0))
    with pytest.raises(ValueError):
        gd.Calibration(((1, 1), (1, 1)), (0, 1), ((0, 0), (0, 5)), (0, 1))


# ====================================================================== Origin window
@pytest.fixture
def ctx(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("RSSTATCHEM_HOME", str(tmp_path))
    from rsstatchem.app import AppContext

    context = AppContext(qapp)
    monkeypatch.setattr(context.app, "quit", lambda: None)
    return context


def _no_prompts(monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    def boom(*args, **kwargs):
        raise AssertionError("unexpected modal dialog")
    for name in ("question", "warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name, boom)


def test_graph_menus_install_and_actions(ctx, monkeypatch, tmp_path, qapp):
    _no_prompts(monkeypatch)
    from PySide6.QtGui import QColor, QImage, QPainter, QPen

    from rsstatchem.ui.origin import graph_menus, gadgets_ui, open_workspace
    from rsstatchem.ui.origin.digitizer import DigitizerDialog
    from rsstatchem.ui.origin.layer_manager import LayerManagerDialog
    from rsstatchem.ui.origin import layout_view

    win = open_workspace(ctx)
    graph_menus.install(win)
    graph_menus.install(win)   # idempotent (main may already have installed it)
    stat = graph_menus.find_submenu(win.menus["plot"], "Statistical")
    texts = [a.text() for a in stat.actions()]
    assert "Violin Plot" in texts and "Q-Q Plot" in texts
    assert texts.count("Violin Plot") == 1
    assert "plot:bland_altman" in win.wp4_actions and "gadget:integrate" in win.wp4_actions
    book = _book()
    win.new_book(book)
    bw = win.active_book_window()
    bw.select_columns([1, 3])
    gwin = win.plot("line_symbol")
    assert gwin is not None
    # axes
    graph_menus.set_scale(win, "y", "log10")
    assert gwin.graph.layers[0].y.scale == "log10"
    assert graph_menus.breaks_dialog(win, "x", "3-6@40", 3.0)
    assert gwin.graph.layers[0].x.breaks == [{"from": 3.0, "to": 6.0, "position": 40.0}]
    assert graph_menus.set_top_axis(win, "x*60", "Time (s)")
    assert gwin.graph.layers[0].x2_formula == "x*60"
    assert gwin.rmap.errors == []
    ref = graph_menus.add_ref(win, mode="mean_sd", kind="band")
    assert ref in gwin.graph.layers[0].refs
    from rsstatchem.ui.graph.window import REF_ROLE
    gwin.rebuild_tree()
    assert any(tuple(item.data(0, REF_ROLE) or ()) == ("ref", 0, ref.id)
               for item in gwin._iter_items())
    gwin.select(("ref", 0, ref.id))
    gwin.delete_selected()
    assert gwin.graph.layers[0].refs == []
    # data additions
    key = graph_menus.add_error_bars(gwin, "percent", 10)
    assert gwin.graph.layers[0].plots[0].yerr == key
    p = graph_menus.add_function_plot(gwin, "10 + 2*sin(x)", 0, 10)
    assert p.options["function"]["formula"] == "10 + 2*sin(x)"
    # format copy/paste
    gwin.selected = ("plot", 0, gwin.graph.layers[0].plots[0].id)
    assert graph_menus.copy_format(win)["kind"] == "plot"
    gwin.selected = ("plot", 0, p.id)
    assert graph_menus.paste_format(win)
    # gadgets: run the integrate gadget over an X range (no mouse needed)
    gwin.selected = ("layer", 0)
    tool = graph_menus.start_gadget(win, "integrate")
    assert gwin.gadget is tool
    res = tool.run(0, 1.0, 9.0)
    assert res["Area"] > 0 and gwin.graph.layers[0].options["roi"]["gadget"] == "integrate"
    assert "Integrate gadget" in win.results.toPlainText()
    gadgets_ui.GadgetTool(gwin, "fft", win).run(0, 0.0, 10.0)
    assert any(name.startswith("FFT") for name in win.project.windows)
    gadgets_ui.clear_roi(gwin)
    assert "roi" not in gwin.graph.layers[0].options and gwin.gadget is None
    # layers
    graph_menus.panel_preset(win, "v2")
    assert len(gwin.graph.layers) == 2
    dlg = LayerManagerDialog(gwin, win)
    dlg.rows.setValue(1)
    dlg.cols.setValue(2)
    dlg.arrange()
    assert gwin.graph.cols == 2
    dlg.list.setCurrentRow(1)
    dlg.master.setCurrentIndex(1)
    dlg.link_x.setChecked(True)
    dlg.apply_link()
    assert gwin.graph.layers[1].options["link_x"] == 0
    dlg.close()
    graph_menus.exchange_xy(win)
    parts = graph_menus.extract_layers(win)
    assert len(parts) == 2
    # layout page, merge, export, print
    lp = layout_view.new_layout_page(win, [gwin.graph.title, parts[0].graph.title], 1)
    assert lp.graph.meta["layout_page"]["cols"] == 1
    mg = layout_view.merge_graph_windows(win, [parts[0].graph.title, parts[1].graph.title], 2)
    assert len(mg.graph.layers) == 2
    paths = layout_view.export_all_graphs(win, str(tmp_path / "exp"), "png", 40)
    assert len(paths) == len(win.graph_windows())
    from PySide6.QtPrintSupport import QPrinter
    printer = QPrinter()
    printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
    printer.setOutputFileName(str(tmp_path / "print.pdf"))
    assert layout_view.print_graph(win, gwin, printer)
    assert (tmp_path / "print.pdf").stat().st_size > 1000
    # theme saving
    graph_menus.save_theme(win, "WP4 theme")
    graph_menus.apply_user_theme(win, "WP4 theme")
    # digitizer: a synthetic image with a straight dark line from (100,300) to (300,100)
    img = QImage(400, 400, QImage.Format.Format_RGB32)
    img.fill(QColor("white"))
    painter = QPainter(img)
    painter.setPen(QPen(QColor("black"), 3))
    painter.drawLine(100, 300, 300, 100)
    painter.end()
    dig = DigitizerDialog(win)
    dig.load_image(img)
    dig.set_reference("X1", 100, 300, 0)
    dig.set_reference("X2", 300, 300, 10)
    dig.set_reference("Y1", 100, 300, 0)
    dig.set_reference("Y2", 100, 100, 10)
    dig.add_point(200, 200)
    assert dig.data()[0] == pytest.approx([5.0, 5.0])
    dig.auto_trace()
    d = dig.data()
    assert len(d) > 20 and np.allclose(d[:, 0], d[:, 1], atol=0.3)
    book2 = dig.to_workbook()
    assert book2 is not None and book2.name in win.project.windows
    dig.close()
    # the project with the new graph content saves and reopens
    path = tmp_path / "wp4.rsop"
    assert win.save_to(str(path))
    win2 = open_workspace(ctx, str(path))
    g2 = win2.view_of(lp.graph.title)
    assert g2 is not None and g2.graph.meta["layout_page"]
    gw2 = win2.view_of(gwin.graph.title)
    assert gw2.graph.layers[0].x2_formula == "x*60"
    win2.modified = False
    win2.close()
    win.modified = False
    win.close()

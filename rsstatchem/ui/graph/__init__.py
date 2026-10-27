"""Graphs menu: Graph Studio (OriginLab-style publication graphs).

Entries open the Plot Setup dialog (optionally with a plot type preselected) on the active
Data Editor's dataset and show the result in a :class:`~rsstatchem.ui.graph.window.GraphWindow`.
Heavy Qt / matplotlib-canvas imports happen lazily inside the menu callables, and
:func:`selftest` touches the core only.
"""
from __future__ import annotations


def _open_setup(editor, kind: str | None = None):
    from PySide6.QtWidgets import QMessageBox

    from rsstatchem.core.graph.build import build_graph
    from rsstatchem.core.graph.themes import DEFAULT_THEME, apply_template, load_user_template
    from rsstatchem.ui.graph.setup_dialog import PlotSetupDialog
    from rsstatchem.ui.graph.window import next_title, show_graph

    ds = editor.ds
    if ds.n_vars == 0 or ds.n_cases == 0:
        QMessageBox.information(editor, "Graph Studio", "The dataset is empty.")
        return None
    dlg = PlotSetupDialog(ds, editor, kind=kind)
    if not dlg.exec():
        return None
    spec = dlg.spec()
    user = spec.theme[5:] if spec.theme.startswith("user:") else None
    if user:
        spec.theme = DEFAULT_THEME
    try:
        graph = build_graph(ds, spec, title=next_title())
        if user:
            apply_template(graph, load_user_template(user))
    except (ValueError, KeyError) as exc:
        QMessageBox.warning(editor, "Graph Studio", str(exc))
        return None
    return show_graph(graph, editor)


def open_graph_studio(editor):
    return _open_setup(editor)


def open_graph_file(editor):
    from rsstatchem.ui.graph.window import open_graph_dialog
    return open_graph_dialog(editor)


def open_chart_builder(editor):
    """Graphs > Chart Builder (SPSS gallery charts into the Output Viewer)."""
    from rsstatchem.ui.graph.chart_builder import ChartBuilderDialog

    return editor.run_analysis(ChartBuilderDialog)


def _builder(key: str):
    def run(editor):
        from rsstatchem.ui.graph.chart_builder import preset

        return editor.run_analysis(preset(key, key))
    run.__name__ = f"chart_builder_{key}"
    return run


def _kind(kind: str):
    def run(editor):
        return _open_setup(editor, kind)
    run.__name__ = f"open_{kind}"
    return run


MENUS = {
    "Graphs": [
        ("Graph Studio (Plot Setup)...", open_graph_studio),
        ("Chart Builder...", open_chart_builder),
        ("Regression Variable Plots...", _builder("regplots")),
        ("Legacy Dialogs", [("Population Pyramid...", _builder("pyramid")),
                            ("High-Low...", _builder("highlow")),
                            ("Range Bar...", _builder("rangebar")),
                            ("Difference Area...", _builder("difference")),
                            ("Scatterplot Matrix...", _builder("splom")),
                            ("Dot / Drop-line...", _builder("dropline")),
                            ("Error Bar (Summary Point)...", _builder("summary_point")),
                            ("Violin Plot...", _builder("violin")),
                            ("1-D Boxplot...", _builder("box1d")),
                            ("Stacked Histogram...", _builder("stacked_hist")),
                            ("Frequency Polygon...", _builder("freq_polygon")),
                            ("Panelled Histogram...", _builder("panel_hist")),
                            ("Dual Y Axes...", _builder("dualaxis"))]),
        None,
        ("Line...", _kind("line")),
        ("Scatter...", _kind("scatter")),
        ("Line + Symbol...", _kind("line_symbol")),
        ("Error Bar...", _kind("errorbar")),
        ("Column / Bar", [("Column...", _kind("column")),
                          ("Stacked Column...", _kind("column_stacked")),
                          ("100% Stacked Column...", _kind("column_percent")),
                          None,
                          ("Bar...", _kind("bar")),
                          ("Stacked Bar...", _kind("bar_stacked")),
                          ("100% Stacked Bar...", _kind("bar_percent"))]),
        ("Area", [("Area...", _kind("area")), ("Stacked Area...", _kind("area_stacked")),
                  ("100% Stacked Area...", _kind("area_percent"))]),
        ("Pie / Doughnut", [("Pie...", _kind("pie")), ("Doughnut...", _kind("doughnut"))]),
        None,
        ("Histogram...", _kind("histogram")),
        ("Box Chart...", _kind("box")),
        ("Bubble / Color Map", [("Bubble...", _kind("bubble")),
                                ("Color-Mapped Scatter...", _kind("color_scatter")),
                                ("Bubble + Color Mapped...", _kind("bubble_color"))]),
        ("Multi-Axis / Multi-Panel", [("Double-Y Axis...", _kind("double_y")),
                                      ("Multi-Panel (Stacked)...", _kind("multi_panel")),
                                      ("Waterfall / Offset Y...", _kind("waterfall"))]),
        ("Contour / 3D", [("Contour...", _kind("contour")), ("Heatmap...", _kind("heatmap")),
                          None,
                          ("3D Surface...", _kind("surface3d")),
                          ("3D Scatter...", _kind("scatter3d"))]),
        ("Specialized", [("Polar...", _kind("polar")), ("Ternary...", _kind("ternary"))]),
        None,
        ("Open Graph...", open_graph_file),
    ],
    "File": [("Open Graph...", open_graph_file)],
}


def selftest() -> None:
    """Build, render and round-trip every plot type on synthetic data (core only, no Qt)."""
    import io

    import numpy as np
    import pandas as pd

    from rsstatchem.core.dataset import Dataset
    from rsstatchem.core.graph import (TEMPLATES, PlotSpec, apply_theme, build_graph,
                                       export_bytes, graph_from_json_dict, graph_to_json_dict)
    from rsstatchem.core.graph.edit import add_inset, add_panel

    rng = np.random.default_rng(7)
    n = 40
    df = pd.DataFrame({"g": rng.integers(1, 3, n).astype(float), "x": np.arange(n, dtype=float),
                       "y1": rng.normal(10, 2, n), "y2": rng.normal(5, 1, n),
                       "z": rng.uniform(0, 1, n), "c": rng.uniform(0.1, 1, n)})
    df.loc[3, "y1"] = np.nan
    ds = Dataset(df)
    ds.var("g").value_labels = {1.0: "A", 2.0: "B"}
    ds.var("g").measure = "nominal"
    for key, t in TEMPLATES.items():
        ys = ["y1", "y2"] if key in ("double_y", "multi_panel", "waterfall") else ["y1"]
        spec = PlotSpec(key, x="g" if key.startswith(("column", "bar", "pie", "doughnut"))
                        else "x" if "x" in t.roles else None, ys=ys,
                        z="z" if "z" in t.roles else None,
                        group="g" if key in ("scatter", "histogram", "box") else None,
                        summary="mean_sd" if key in ("column", "bar") else "none")
        if key == "ternary":
            spec.x, spec.ys, spec.z = "z", ["c"], "y2"
        graph = build_graph(ds, spec)
        back = graph_from_json_dict(graph_to_json_dict(graph))
        if graph_to_json_dict(back) != graph_to_json_dict(graph):
            raise AssertionError(f"{key}: JSON round-trip changed the graph")
        png = export_bytes(back, "png", dpi=60)
        if png[:8] != b"\x89PNG\r\n\x1a\n":
            raise AssertionError(f"{key}: no PNG produced")
    g = build_graph(ds, PlotSpec("line_symbol", x="x", ys=["y1"]))
    add_panel(g, "below")
    add_inset(g)
    apply_theme(g, "Journal - Single Column (8.3 cm)")
    pdf = export_bytes(g, "pdf")
    if not pdf.startswith(b"%PDF"):
        raise AssertionError("PDF export failed")
    io.BytesIO(export_bytes(g, "svg", dpi=72)).read()

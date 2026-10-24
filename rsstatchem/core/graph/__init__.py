"""Graph Studio core: Origin-style graph model, renderer, themes, export (no Qt).

Typical use::

    from rsstatchem.core.graph import PlotSpec, build_graph, export_graph
    g = build_graph(ds, PlotSpec("scatter", x="pretest", ys=["posttest"], group="kelompok"))
    export_graph(g, "fig1.tif", dpi=600)
"""
from __future__ import annotations

from rsstatchem.core.graph.build import (SUMMARIES, TEMPLATES, PlotSpec, Template,
                                         build_graph)
from rsstatchem.core.graph.export import (DPI_CHOICES, FORMATS, export_bytes, export_graph,
                                          figure_for, load_graph, png_bytes, save_graph)
from rsstatchem.core.graph.model import (Annotation, Axis, Column, Font, Graph, Layer,
                                         Legend, Plot, from_dict, graph_from_json_dict,
                                         graph_to_json_dict, to_dict)
from rsstatchem.core.graph.render import RenderMap, render
from rsstatchem.core.graph.themes import (DEFAULT_THEME, THEMES, apply_template,
                                          apply_theme)

__all__ = ["Annotation", "Axis", "Column", "DEFAULT_THEME", "DPI_CHOICES", "FORMATS", "Font",
           "Graph", "Layer", "Legend", "Plot", "PlotSpec", "RenderMap", "SUMMARIES",
           "TEMPLATES", "THEMES", "Template", "apply_template", "apply_theme", "build_graph",
           "export_bytes", "export_graph", "figure_for", "from_dict", "graph_from_json_dict",
           "graph_to_json_dict", "load_graph", "png_bytes", "render", "save_graph", "to_dict"]

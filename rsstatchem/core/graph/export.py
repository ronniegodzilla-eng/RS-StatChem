"""Export graphs (PNG/JPEG/TIFF at journal DPI, SVG, PDF, EPS) and save/open ``.rsg`` files.

Export re-renders the model on a fresh Agg figure, so the screen zoom never affects the file.
A custom export size re-lays out the graph at that size with fonts kept at their point size
(what journals ask for), instead of scaling the bitmap. Fonts are embedded as TrueType
(Type 42) in PDF/EPS so text stays editable and passes journal checks.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from rsstatchem.core.graph.model import Graph, graph_from_json_dict, graph_to_json_dict
from rsstatchem.core.graph.render import RenderMap, render

FORMATS = {"png": "PNG image", "jpg": "JPEG image", "tif": "TIFF image (LZW)",
           "svg": "SVG vector", "pdf": "PDF vector", "eps": "EPS vector"}
RASTER = ("png", "jpg", "tif")
DPI_CHOICES = (150, 300, 600, 1200)
EXTENSION = ".rsg"
_ALIASES = {"jpeg": "jpg", "tiff": "tif", "ps": "eps"}
_RC = {"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
       "savefig.bbox": None}


def normalize_format(fmt: str) -> str:
    fmt = fmt.lower().lstrip(".")
    fmt = _ALIASES.get(fmt, fmt)
    if fmt not in FORMATS:
        raise ValueError(f"Unsupported export format: {fmt}")
    return fmt


def sized(graph: Graph, width_cm: float | None = None, height_cm: float | None = None,
          keep_aspect: bool = True) -> Graph:
    g = graph.copy()
    w0, h0 = g.page.width_cm, g.page.height_cm
    if width_cm and height_cm:
        g.page.width_cm, g.page.height_cm = width_cm, height_cm
    elif width_cm:
        g.page.width_cm = width_cm
        g.page.height_cm = h0 * width_cm / w0 if keep_aspect else h0
    elif height_cm:
        g.page.height_cm = height_cm
        g.page.width_cm = w0 * height_cm / h0 if keep_aspect else w0
    return g


def figure_for(graph: Graph, dpi: float = 100) -> tuple[Figure, RenderMap]:
    fig = Figure(dpi=dpi)
    FigureCanvasAgg(fig)
    rmap = render(graph, fig)
    return fig, rmap


def export_bytes(graph: Graph, fmt: str = "png", dpi: int = 300,
                 width_cm: float | None = None, height_cm: float | None = None,
                 transparent: bool = False) -> bytes:
    fmt = normalize_format(fmt)
    g = sized(graph, width_cm, height_cm)
    fig, _ = figure_for(g, dpi=100)
    buf = io.BytesIO()
    kwargs: dict = {"format": {"jpg": "jpeg", "tif": "tiff"}.get(fmt, fmt), "dpi": dpi,
                    "facecolor": "none" if transparent else g.page.background,
                    "transparent": transparent and fmt in ("png", "svg", "pdf", "tif")}
    if fmt == "jpg":
        kwargs["pil_kwargs"] = {"quality": 95, "dpi": (dpi, dpi)}
        kwargs["transparent"] = False
        kwargs["facecolor"] = g.page.background if g.page.background != "none" else "white"
    elif fmt == "tif":
        kwargs["pil_kwargs"] = {"compression": "tiff_lzw", "dpi": (dpi, dpi)}
    elif fmt == "png":
        kwargs["pil_kwargs"] = {"dpi": (dpi, dpi)}
    with matplotlib.rc_context(_RC):
        fig.savefig(buf, **kwargs)
    return buf.getvalue()


def export_graph(graph: Graph, path: str | Path, fmt: str | None = None, dpi: int = 300,
                 width_cm: float | None = None, height_cm: float | None = None,
                 transparent: bool = False) -> Path:
    path = Path(path)
    fmt = normalize_format(fmt or path.suffix or "png")
    data = export_bytes(graph, fmt, dpi, width_cm, height_cm, transparent)
    path.write_bytes(data)
    return path


def png_bytes(graph: Graph, dpi: int = 150) -> bytes:
    return export_bytes(graph, "png", dpi)


# ---------------------------------------------------------------------- .rsg files
def save_graph(graph: Graph, path: str | Path) -> Path:
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(EXTENSION)
    text = json.dumps(graph_to_json_dict(graph), ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))
    path.write_text(text, encoding="utf-8")
    return path


def load_graph(path: str | Path) -> Graph:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return graph_from_json_dict(data)

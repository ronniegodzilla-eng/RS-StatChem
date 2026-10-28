"""Gadgets on a graph window (Origin's Gadgets menu): drag a region of interest over a layer
and get Statistics, Integrate, FFT, Differentiate, Rise Time or Intersect results for the
active plot; click for the Vertical Cursor. The ROI stays on the graph (``layer.options
["roi"]``) and the results go to the Results Log (FFT and Differentiate also make a
workbook with the output curve). Masked rows (WP-2's ``masked_rows``) are excluded when the
worksheet provides them.
"""
from __future__ import annotations

import numpy as np

from rsstatchem.core.graph import gadgets as gd

LINEISH = ("line", "scatter", "line_symbol", "lollipop", "area")
LABELS = {**{k: v[0] for k, v in gd.GADGETS.items()}, "intersect": "Intersect",
          "cursor": "Vertical Cursor"}


def target_plot(gwin, li: int):
    """The plot a gadget works on: the selected plot of the layer, else its first
    line/scatter plot."""
    layer = gwin.graph.layers[li]
    sel = gwin.selected
    if sel and sel[0] == "plot" and sel[1] == li:
        p = next((q for q in layer.plots if q.id == sel[2]), None)
        if p is not None:
            return p
    return next((q for q in layer.plots if q.type in LINEISH and q.visible), None) or \
        (layer.plots[0] if layer.plots else None)


def plot_mask(main, plot):
    """Masked rows of the plot's source column (True = excluded) when WP-2's mask API is
    available; None otherwise."""
    src = (plot.options.get("source") or {}) if plot is not None else {}
    if not src.get("book") or not src.get("y") or main is None:
        return None
    try:
        book = main.project.windows.get(src["book"])
        ws = book.sheet(src.get("sheet", ""))
        fn = getattr(ws, "masked_rows", None)
        if fn is None:
            return None
        rows = fn(src["y"])
    except Exception:  # noqa: BLE001 - the mask API is optional
        return None
    if rows is None:
        return None
    lo = (src.get("rows") or [1])[0] - 1
    rows = np.asarray(sorted(rows), int) - lo
    y = None
    try:
        y = len(ws.column(src["y"]).numeric())
    except Exception:  # noqa: BLE001
        pass
    n = max([y or 0] + list(rows + 1))
    mask = np.zeros(n, bool)
    mask[rows[(rows >= 0) & (rows < n)]] = True
    return mask


def compute(gwin, li: int, key: str, x0: float, x1: float, main=None) -> dict:
    """Run gadget ``key`` over [x0, x1] of layer ``li`` (no UI): the result dict."""
    from rsstatchem.ui.origin.plotting import plot_data
    layer = gwin.graph.layers[li]
    p = target_plot(gwin, li)
    if p is None:
        raise ValueError("The layer has no plot.")
    y = gwin.graph.values(p.y)
    x = gwin.graph.values(p.x)
    if y is None:
        raise ValueError("The plot has no Y data.")
    if x is None or x.dtype == object:
        x = np.arange(1, len(y) + 1, dtype=float)
    mask = plot_mask(main, p)
    if key == "cursor":
        out = gd.cursor(x, y, x0)
    elif key == "intersect":
        others = [q for q in layer.plots if q.visible and q.type in LINEISH and q is not p]
        if not others:
            raise ValueError("Intersect needs two curves in the layer.")
        x2, y2 = plot_data(gwin.graph, others[0])
        pts = gd.intersect(x, y, x2, y2, min(x0, x1), max(x0, x1))
        out = {"Curve 1": p.name, "Curve 2": others[0].name, "Intersections": len(pts)}
        for k, (xv, yv) in enumerate(pts, 1):
            out[f"X{k}"], out[f"Y{k}"] = xv, yv
    elif key == "integrate":
        out = gd.integrate(x, y, x0, x1, layer.options.get("roi_baseline", "zero"), mask)
    else:
        if key not in gd.GADGETS:
            raise ValueError(f"Unknown gadget {key!r}.")
        out = gd.GADGETS[key][1](x, y, x0, x1, mask=mask)
    out = {"Plot": p.name or p.type, **out}
    return out


class GadgetTool:
    """ROI interaction on an :class:`OriginGraphWindow` (the window forwards its mouse
    events here while ``gwin.gadget`` is set)."""

    def __init__(self, gwin, key: str, main=None):
        self.gwin, self.key, self.main = gwin, key, main
        self.start = None      # (li, x0)
        self.preview = None

    # ------------------------------------------------------------------ events
    def press(self, event) -> None:
        if event.button != 1 or self.gwin.rmap is None:
            return
        li = self.gwin._layer_at(event)
        info = self.gwin.rmap.axes.get(li)
        if info is None or info["layer"].kind != "cartesian":
            return
        ax = info["ax"]
        x0 = ax.transData.inverted().transform((event.x, event.y))[0]
        self.start = (li, float(x0))
        self.preview = ax.axvspan(x0, x0, color="#3d7fd9", alpha=0.2)
        self.gwin.canvas.draw_idle()

    def motion(self, event) -> None:
        if self.start is None or self.preview is None:
            return
        li, x0 = self.start
        ax = self.gwin.rmap.axes[li]["ax"]
        x1 = ax.transData.inverted().transform((event.x, event.y))[0]
        try:
            self.preview.set_x(min(x0, x1))
            self.preview.set_width(abs(x1 - x0))
        except AttributeError:   # matplotlib returns a Polygon on older versions
            xy = [[min(x0, x1), 0], [min(x0, x1), 1], [max(x0, x1), 1], [max(x0, x1), 0]]
            self.preview.set_xy(xy)
        self.gwin.canvas.draw_idle()

    def release(self, event) -> None:
        if self.start is None:
            return
        li, x0 = self.start
        self.start = None
        ax = self.gwin.rmap.axes[li]["ax"]
        x1 = float(ax.transData.inverted().transform((event.x, event.y))[0])
        if self.key != "cursor" and abs(x1 - x0) < 1e-12:
            lo, hi = sorted(ax.get_xlim())   # a click takes the whole visible range
            x0, x1 = lo, hi
        self.run(li, x0, x1)

    # ------------------------------------------------------------------ results
    def run(self, li: int, x0: float, x1: float) -> dict | None:
        try:
            result = compute(self.gwin, li, self.key, x0, x1, self.main)
        except (ValueError, IndexError) as exc:
            self._report_error(str(exc))
            self.gwin.redraw()
            return None
        title = LABELS.get(self.key, self.key)

        def fn():
            layer = self.gwin.graph.layers[li]
            if self.key == "cursor":
                layer.options["roi"] = {"x0": x0, "x1": x0, "gadget": "cursor",
                                        "title": f"X={x0:.4g}"}
            else:
                layer.options["roi"] = {"x0": min(x0, x1), "x1": max(x0, x1),
                                        "gadget": self.key, "title": title}
        self.gwin.mutate(fn, rebuild_tree=False)
        self._report(title, result)
        return result

    def _report_error(self, text: str) -> None:
        if self.main is not None:
            self.main.message(f"{LABELS.get(self.key, self.key)} gadget: {text}", error=True)

    def _report(self, title: str, result: dict) -> None:
        text = gd.describe(result)
        if self.main is None:
            return
        self.main.result(f"{title} gadget on [{self.gwin.graph.title}]\n{text}")
        self.main.show_status(f"{title}: " + ", ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in list(gd.public(result).items())[1:5]))
        curves = {"fft": ("_frequency", "_amplitude", "Frequency", "Amplitude"),
                  "differentiate": ("_x", "_dydx", "X", "dY/dX")}.get(self.key)
        if curves and curves[0] in result:
            self._to_book(title, result[curves[0]], result[curves[1]], curves[2], curves[3])

    def _to_book(self, title, x, y, xname, yname) -> None:
        from rsstatchem.ui.origin.model import Workbook
        main = self.main
        book = Workbook.new(main.project.unique_name(title.replace(" ", "")), 2, len(x))
        ws = book.sheets[0]
        ws.set_column_values(0, [float(v) for v in x])
        ws.set_column_values(1, [float(v) for v in y])
        ws.columns[0].long_name, ws.columns[1].long_name = xname, yname
        main.new_book(book, show=False)
        main.message(f"{title} gadget output → [{book.name}] (hidden; open it from the "
                     "Project Explorer)")


def start(main, gwin, key: str) -> GadgetTool:
    tool = GadgetTool(gwin, key, main)
    gwin.gadget = tool
    gwin.canvas.setCursor(_cross())
    if main is not None:
        main.show_status(f"{LABELS.get(key, key)} gadget: drag over the X range of interest "
                         "(click for the whole layer); Esc or the Pointer tool ends it.")
    return tool


def stop(gwin) -> None:
    gwin.gadget = None
    from PySide6.QtCore import Qt
    gwin.canvas.setCursor(Qt.CursorShape.ArrowCursor)


def clear_roi(gwin) -> None:
    li = gwin.active_index()

    def fn():
        gwin.graph.layers[li].options.pop("roi", None)
    gwin.mutate(fn, rebuild_tree=False)
    stop(gwin)


def _cross():
    from PySide6.QtCore import Qt
    return Qt.CursorShape.CrossCursor

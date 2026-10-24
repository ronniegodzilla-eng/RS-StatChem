"""Ternary diagram drawn on an ordinary equal-aspect Axes (Origin's Ternary plot).

Components (A, B, C) are normalised to sum 1 and mapped to the triangle with A at the
bottom-left, B at the bottom-right and C at the top: ``P = (b + c/2, c·√3/2)``.
"""
from __future__ import annotations

import numpy as np

from rsstatchem.core.graph.model import Axis, Layer

H = np.sqrt(3) / 2


def to_xy(a, b, c) -> tuple[np.ndarray, np.ndarray]:
    a, b, c = (np.asarray(v, dtype=float) for v in (a, b, c))
    total = a + b + c
    with np.errstate(invalid="ignore", divide="ignore"):
        b, c = b / total, c / total
    return b + c / 2, c * H


def _pt(a, b, c):
    x, y = to_xy([a], [b], [c])
    return float(x[0]), float(y[0])


def draw_frame(ax, layer: Layer, font_kw, rich, scale100: bool = False) -> dict:
    """Triangle, grid, ticks, tick labels and corner titles. Returns axis-name → artists."""
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.12, H + 0.1)
    axes = {"x": layer.x, "y": layer.y, "z": layer.z}
    artists: dict[str, list] = {"x": [], "y": [], "z": []}
    frame_w = max(a.line_width for a in axes.values())
    tri = np.array([_pt(1, 0, 0), _pt(0, 1, 0), _pt(0, 0, 1), _pt(1, 0, 0)])
    ax.plot(tri[:, 0], tri[:, 1], color=layer.x.color, lw=frame_w, zorder=3,
            solid_joinstyle="miter")

    def ticks_of(axis: Axis) -> np.ndarray:
        step = axis.major_step if axis.major_step and 0 < axis.major_step < 1 else 0.2
        if scale100 and axis.major_step and axis.major_step >= 1:
            step = axis.major_step / 100
        return np.arange(step, 1 - 1e-9, step)

    # For each component: the grid line at value t runs between two edge points; the tick
    # sits on the edge where that component's labels are read.
    def seg(name: str, t: float):
        if name == "x":    # A: from bottom edge to left edge
            return _pt(t, 1 - t, 0), _pt(t, 0, 1 - t)
        if name == "y":    # B: from right edge to bottom edge
            return _pt(0, t, 1 - t), _pt(1 - t, t, 0)
        return _pt(1 - t, 0, t), _pt(0, 1 - t, t)  # C: from left edge to right edge

    label_off = {"x": (0, -0.05), "y": (0.045, 0.02), "z": (-0.045, 0.0)}
    align = {"x": ("center", "top"), "y": ("left", "center"), "z": ("right", "center")}
    for name, axis in axes.items():
        if not axis.visible:
            continue
        for t in ticks_of(axis):
            p0, p1 = seg(name, t)
            if axis.grid_major:
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=axis.grid_color,
                        ls=axis.grid_style, lw=0.6, zorder=1)
            d = np.subtract(p1, p0)
            d = d / np.hypot(*d) * 0.025
            if axis.tick_dir != "none":
                start = p0 - d if axis.tick_dir in ("out", "both") else p0
                end = p0 + d if axis.tick_dir in ("in", "both") else p0
                ax.plot([start[0], end[0]], [start[1], end[1]], color=axis.color,
                        lw=axis.tick_width, zorder=3)
            if axis.show_labels:
                value = t * 100 if scale100 else t
                text = f"{value:.0f}" if scale100 else f"{value:.1f}"
                ox, oy = label_off[name]
                ha, va = align[name]
                artists[name].append(ax.text(p0[0] + ox, p0[1] + oy, text, ha=ha, va=va,
                                             **font_kw(axis.label_font)))
        if axis.show_title and axis.title:
            pos = {"x": (_pt(1, 0, 0), ("right", "top"), (-0.03, -0.07)),
                   "y": (_pt(0, 1, 0), ("left", "top"), (0.03, -0.07)),
                   "z": (_pt(0, 0, 1), ("center", "bottom"), (0, 0.03))}[name]
            (px, py), (ha, va), (ox, oy) = pos
            artists[name].append(ax.text(px + ox, py + oy, rich(axis.title), ha=ha, va=va,
                                         **font_kw(axis.title_font)))
    return artists

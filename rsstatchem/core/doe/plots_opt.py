"""Optimisation and post-analysis graphs (PNG bytes / matplotlib figures, Agg canvas):
desirability ramps with interval bounds, desirability bars, desirability and POE contour /
3D surfaces, the graphical overlay with interval bands and the confirmation plot."""
from __future__ import annotations

import math

import numpy as np
from matplotlib.figure import Figure

from rsstatchem.core.doe import plots
from rsstatchem.core.doe.plots import POINT, SERIES, _figure, _png, _style

GOAL_COLOR = {"factor": "#1f3fbf", "response": "#2e7d32", "se": "#6a1b9a", "poe": "#ef6c00"}


# ---------------------------------------------------------------------------- ramps / bars
def ramps_plot(entries: list[dict], D: float, title: str = "Desirability Ramps") -> bytes:
    """entries: {name, low, high, value, goal, target?, kind, bounds?, interval?}. When a goal
    is judged on an interval, its bounds are drawn as a bracket around the dot."""
    n = len(entries)
    cols = min(4, max(n, 1))
    rows = math.ceil(n / cols) + 1
    fig = _figure(1.7 * cols + 0.6, 1.3 * rows + 0.4)
    for i, e in enumerate(entries):
        ax = fig.add_subplot(rows, cols, i + 1)
        lo, hi, v = e["low"], e["high"], e["value"]
        span = hi - lo if hi > lo else 1.0
        goal = e.get("goal")
        xs = [lo, hi]
        if goal in ("maximize", "cpk"):
            ys = [0, 1]
        elif goal == "minimize":
            ys = [1, 0]
        elif goal == "target":
            t = e.get("target")
            t = (lo + hi) / 2 if t is None else t
            xs, ys = [lo, t, hi], [0, 1, 0]
        elif goal == "equal":
            t = e.get("target", v)
            xs, ys = [t, t], [0, 1]
        else:
            ys = [1, 1]
        ax.plot(xs, ys, color=GOAL_COLOR.get(e.get("kind"), "#2e7d32"), linewidth=1.6)
        clip = lambda u: min(max(u, lo - 0.1 * span), hi + 0.1 * span)  # noqa: E731
        if e.get("bounds"):
            b0, b1 = e["bounds"]
            a0 = clip(b0) if np.isfinite(b0) else lo - 0.1 * span
            a1 = clip(b1) if np.isfinite(b1) else hi + 0.1 * span
            ax.plot([a0, a1], [0.5, 0.5], color="#555555", linewidth=1.2)
            for a, fin in ((a0, np.isfinite(b0)), (a1, np.isfinite(b1))):
                if fin:
                    ax.plot([a, a], [0.38, 0.62], color="#555555", linewidth=1.2)
        ax.scatter([clip(v)], [0.5], s=50, color=POINT, zorder=4)
        ax.set_xlim(lo - 0.1 * span, hi + 0.1 * span)
        ax.set_ylim(-0.1, 1.15)
        ax.set_yticks([])
        ax.set_xticks([lo, hi], [f"{lo:.4g}", f"{hi:.4g}"])
        ax.tick_params(labelsize=6)
        label = f"{e['name']} = {v:.4g}"
        if e.get("interval"):
            label += f"\n{e['interval']}"
        ax.set_title(label, fontsize=7.5)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    ax = fig.add_subplot(rows, 1, rows)
    ax.barh([0], [D], color="#2e7d32", height=0.5)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.tick_params(labelsize=7)
    ax.set_title(f"Desirability = {D:.3f}", fontsize=9, fontweight="bold")
    fig.suptitle(title, fontsize=11, fontweight="bold")
    return _png(fig)


def desirability_bars(names: list[str], values: list[float], D: float,
                      title: str = "Desirability") -> bytes:
    fig = _figure(6.0, 0.45 * (len(names) + 1) + 1.2)
    ax = fig.add_subplot()
    labels = list(names) + ["Combined"]
    vals = list(values) + [D]
    colors = ["#1f3fbf"] * len(values) + ["#2e7d32"]
    ax.barh(range(len(vals))[::-1], vals, color=colors, height=0.6)
    ax.set_yticks(range(len(vals))[::-1], labels)
    for i, v in enumerate(vals):
        ax.text(min(v, 0.92) + 0.01, len(vals) - 1 - i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, 1)
    _style(ax, title, "Desirability", "")
    return _png(fig)


# ---------------------------------------------------------------------------- surfaces
def contour(gx, gy, Z, xlabel: str, ylabel: str, title: str, zlabel: str = "",
            solution=None, points=None) -> bytes:
    return plots.contour_plot(gx, gy, Z, xlabel, ylabel, title, points=points,
                              solution=solution, zlabel=zlabel)


def surface_figure(gx, gy, Z, xlabel: str, ylabel: str, zlabel: str, title: str) -> Figure:
    return plots.surface_figure(gx, gy, Z, xlabel, ylabel, zlabel, title)


def poe_figure(gx, gy, Z, xlabel: str, ylabel: str, response: str, kind: str = "contour",
               solution=None) -> Figure:
    """POE graph of one response: contour (default) or 3D surface. Low POE (blue) marks
    settings that are robust to factor variation."""
    title = f"POE ({response})"
    if kind == "3d":
        return plots.surface_figure(gx, gy, Z, xlabel, ylabel, "POE", title)
    return plots.contour_figure(gx, gy, Z, xlabel, ylabel, title, solution=solution,
                                zlabel="POE")


# ---------------------------------------------------------------------------- overlay
def overlay_interval_plot(gx, gy, limits: list[tuple], xlabel: str, ylabel: str, title: str,
                          solution=None, points=None) -> bytes:
    """Graphical optimisation with interval bands. ``limits``: (name, Z, Zlow, Zhigh, lo, hi)
    where Zlow / Zhigh are the interval bounds compared with the lower / upper limit (equal
    to Z when no interval is used). Yellow: every response meets its limits on the bounds;
    solid lines: the prediction at a limit; dashed: the interval bound at that limit."""
    fig = _figure(6.2, 4.8)
    ax = fig.add_subplot()
    bad = np.zeros_like(gx, dtype=bool)
    for _name, _Z, Zl, Zh, lo, hi in limits:
        if lo is not None:
            bad |= ~(Zl >= lo)
        if hi is not None:
            bad |= ~(Zh <= hi)
    ax.contourf(gx, gy, (~bad).astype(float), levels=[-0.5, 0.5, 1.5],
                colors=["#bdbdbd", "#fff176"])
    for i, (name, Z, Zl, Zh, lo, hi) in enumerate(limits):
        color = SERIES[i % len(SERIES)]
        for v, Zb in ((lo, Zl), (hi, Zh)):
            if v is None:
                continue
            if np.nanmin(Z) < v < np.nanmax(Z):
                cs = ax.contour(gx, gy, Z, levels=[v], colors=[color], linewidths=1.4)
                ax.clabel(cs, fmt=lambda val, nm=name: f"{nm}: {val:.4g}", fontsize=7)
            if Zb is not Z:
                Zf = np.where(np.isfinite(Zb), Zb, np.nan)
                if np.nanmin(Zf) < v < np.nanmax(Zf):
                    ax.contour(gx, gy, Zf, levels=[v], colors=[color], linewidths=1.1,
                               linestyles="dashed")
    plots._design_points(ax, points)
    if solution is not None:
        ax.scatter([solution[0]], [solution[1]], marker="X", s=90, color="#ffffff",
                   edgecolor="#000000", zorder=7)
    _style(ax, title, xlabel, ylabel)
    ax.grid(False)
    return _png(fig)


# ---------------------------------------------------------------------------- confirmation
def confirmation_plot(rows: list[dict], title: str = "Confirmation") -> bytes:
    """rows: {name, mean, pi_low, pi_high, runs, data_mean, n} per response; one panel each,
    the PI for the mean of n runs as a band, individual runs as dots."""
    n = max(len(rows), 1)
    fig = _figure(min(2.4 * n + 1.0, 9.0), 3.4)
    for i, r in enumerate(rows):
        ax = fig.add_subplot(1, n, i + 1)
        ax.axhspan(r["pi_low"], r["pi_high"], color="#c8e6c9", alpha=0.8, zorder=1)
        ax.axhline(r["mean"], color="#1f3fbf", linewidth=1.4, zorder=2)
        runs = r.get("runs") or []
        if runs:
            ax.scatter(np.arange(1, len(runs) + 1), runs, color=POINT, s=30, zorder=4)
        if r.get("data_mean") is not None:
            ax.axhline(r["data_mean"], color=POINT, linestyle="dashed", linewidth=1.2)
        ax.set_xlim(0.3, max(len(runs), 1) + 0.7)
        ax.set_xticks(range(1, max(len(runs), 1) + 1))
        vals = [r["pi_low"], r["pi_high"], r["mean"]] + list(runs)
        span = (max(vals) - min(vals)) or 1.0
        ax.set_ylim(min(vals) - 0.15 * span, max(vals) + 0.15 * span)
        _style(ax, r["name"], "Run", "")
        ax.tick_params(labelsize=7)
        ax.title.set_fontsize(9)
    fig.suptitle(title + " (band: PI for the mean of n runs)", fontsize=10, fontweight="bold")
    return _png(fig)

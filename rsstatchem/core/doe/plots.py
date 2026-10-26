"""Matplotlib figures for the DOE module, rendered to PNG bytes (Agg canvas, no pyplot).

Diagnostics (normal plot of residuals, residuals vs predicted / run, predicted vs actual,
Box-Cox, Cook's distance, leverage, DFFITS), model graphs (2D contour with design points, 3D
surface, perturbation, interaction, one factor, cube, ternary contour for mixtures),
optimisation graphics (desirability contour, overlay plot, ramps, desirability bars),
half-normal / Pareto charts of effects and Taguchi main-effect plots. Colours follow
Design-Expert (blue = low, red = high).
"""
from __future__ import annotations

import io
import math

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.tri import Triangulation
from scipy import stats

DE_CMAP = LinearSegmentedColormap.from_list(
    "design_expert", ["#1f3fbf", "#22a7d6", "#2fbf5f", "#e8e23a", "#e2442b"])
POINT = "#C62828"
LINE = "#1f1f1f"
LIMIT = "#d32f2f"
SERIES = ["#1f3fbf", "#C62828", "#2e7d32", "#ef6c00", "#6a1b9a", "#00838f", "#5d4037",
          "#ad1457", "#455a64", "#9e9d24"]


def _figure(width=6.0, height=4.5) -> Figure:
    fig = Figure(figsize=(width, height), dpi=120)
    FigureCanvasAgg(fig)
    return fig


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)


def _png(fig: Figure) -> bytes:
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    return buf.getvalue()


# ---------------------------------------------------------------------------- diagnostics
def normal_plot(resid: np.ndarray, title: str = "Normal Plot of Residuals",
                xlabel: str = "Externally Studentized Residuals") -> bytes:
    fig = _figure()
    ax = fig.add_subplot()
    r = np.sort(np.asarray(resid, float))
    n = len(r)
    q = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    ax.scatter(r, q, s=28, color=POINT, edgecolor="#7f1010", zorder=3)
    if n > 1:
        slope, icpt = np.polyfit(q, r, 1) if np.ptp(r) > 0 else (1, 0)
        qq = np.array([q.min(), q.max()])
        ax.plot(slope * qq + icpt, qq, color=LINE, linewidth=1)
    ticks = [1, 5, 10, 20, 30, 50, 70, 80, 90, 95, 99]
    ax.set_yticks(stats.norm.ppf(np.array(ticks) / 100), [str(t) for t in ticks])
    _style(ax, title, xlabel, "Normal % Probability")
    return _png(fig)


def residual_plot(x: np.ndarray, resid: np.ndarray, limit: float | None, title: str,
                  xlabel: str, ylabel: str = "Externally Studentized Residuals",
                  connect: bool = False) -> bytes:
    fig = _figure()
    ax = fig.add_subplot()
    x = np.asarray(x, float)
    order = np.argsort(x, kind="stable")
    if connect:
        ax.plot(x[order], np.asarray(resid)[order], color="#9e9e9e", linewidth=0.8, zorder=2)
    ax.scatter(x, resid, s=28, color=POINT, edgecolor="#7f1010", zorder=3)
    ax.axhline(0, color=LINE, linewidth=0.8)
    if limit is not None and np.isfinite(limit):
        for s in (-1, 1):
            ax.axhline(s * limit, color=LIMIT, linewidth=1.2)
        m = max(limit * 1.15, np.nanmax(np.abs(resid)) * 1.1)
        ax.set_ylim(-m, m)
        ax.text(1.0, limit, f" {limit:.4g}", transform=ax.get_yaxis_transform(), fontsize=7,
                color=LIMIT, va="bottom", ha="right")
        ax.text(1.0, -limit, f" {-limit:.4g}", transform=ax.get_yaxis_transform(),
                fontsize=7, color=LIMIT, va="top", ha="right")
    _style(ax, title, xlabel, ylabel)
    return _png(fig)


def predicted_vs_actual(actual: np.ndarray, predicted: np.ndarray, label: str) -> bytes:
    fig = _figure()
    ax = fig.add_subplot()
    ax.scatter(actual, predicted, s=28, color=POINT, edgecolor="#7f1010", zorder=3)
    lo = float(min(np.min(actual), np.min(predicted)))
    hi = float(max(np.max(actual), np.max(predicted)))
    pad = 0.05 * (hi - lo or 1)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=LINE, linewidth=1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    _style(ax, f"Predicted vs. Actual: {label}", "Actual", "Predicted")
    return _png(fig)


def box_cox_plot(bc: dict, label: str) -> bytes:
    fig = _figure(7.0, 4.5)
    ax = fig.add_subplot()
    lam, lnss = bc["lambdas"], bc["ln_ss"]
    ax.plot(lam, lnss, color="#1f3fbf", linewidth=1.6)
    ax.axhline(bc["cut"], color=LIMIT, linewidth=1, linestyle="--")
    ax.axvline(bc["best"], color="#2e7d32", linewidth=1.4)
    ax.axvline(bc["current"], color="#1f3fbf", linewidth=1, linestyle=":")
    for v in (bc["low"], bc["high"]):
        if v is not None:
            ax.axvline(v, color=LIMIT, linewidth=1)
    lo = float(lnss.min())
    top = max(float(bc["cut"]), lo + 1e-6)
    span = top - lo
    shown = lnss[lnss <= top + 2.0 * span]
    hi = max(top, float(shown.max()) if len(shown) else top)
    ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))
    txt = (f"Lambda\nCurrent = {bc['current']:g}\nBest = {bc['best']:.3g}\n"
           f"Low C.I. = {bc['low']:.3g}" if bc["low"] is not None else
           f"Lambda\nCurrent = {bc['current']:g}\nBest = {bc['best']:.3g}\nLow C.I. = -")
    txt += f"\nHigh C.I. = {bc['high']:.3g}" if bc["high"] is not None else "\nHigh C.I. = -"
    txt += f"\n\nRecommend\ntransform:\n{bc['recommended_name']}"
    fig.subplots_adjust(right=0.74)
    fig.text(0.76, 0.85, txt, fontsize=8, va="top",
             bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "boxstyle": "round"})
    _style(ax, f"Box-Cox Plot: {label}", "Lambda", "Ln(ResidualSS)")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    return buf.getvalue()


def index_plot(run: np.ndarray, values: np.ndarray, title: str, ylabel: str,
               limits: list[float] | None = None) -> bytes:
    fig = _figure()
    ax = fig.add_subplot()
    order = np.argsort(run, kind="stable")
    ax.plot(np.asarray(run)[order], np.asarray(values)[order], color="#9e9e9e", linewidth=0.8)
    ax.scatter(run, values, s=28, color=POINT, edgecolor="#7f1010", zorder=3)
    for lim in limits or []:
        ax.axhline(lim, color=LIMIT, linewidth=1.2)
    if limits:
        allv = np.concatenate([np.asarray(values, float), np.asarray(limits, float)])
        span = np.ptp(allv) or 1
        ax.set_ylim(allv.min() - 0.08 * span, allv.max() + 0.08 * span)
    _style(ax, title, "Run Number", ylabel)
    return _png(fig)


# ---------------------------------------------------------------------------- model graphs
def _design_points(ax, pts, color=POINT, size=26):
    if pts is None or not len(pts):
        return
    pts = np.asarray(pts, float)
    uniq, counts = np.unique(np.round(pts, 8), axis=0, return_counts=True)
    ax.scatter(uniq[:, 0], uniq[:, 1], s=size, color=color, edgecolor="#5a0b0b", zorder=5,
               clip_on=False)
    for (x, y), c in zip(uniq, counts):
        if c > 1:
            ax.annotate(str(c), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7,
                        color="#5a0b0b", zorder=6)


def contour_plot(gx, gy, Z, xlabel: str, ylabel: str, title: str, points=None,
                 n_levels: int = 10, solution=None, zlabel: str = "", filled: bool = True,
                 mask=None) -> bytes:
    fig = _figure(6.2, 4.8)
    ax = fig.add_subplot()
    Zm = np.ma.masked_invalid(Z if mask is None else np.where(mask, Z, np.nan))
    levels = np.linspace(float(Zm.min()), float(Zm.max()), n_levels + 1) \
        if Zm.count() and Zm.max() > Zm.min() else n_levels
    if filled:
        cf = ax.contourf(gx, gy, Zm, levels=levels, cmap=DE_CMAP, alpha=0.9)
        cbar = fig.colorbar(cf, ax=ax, shrink=0.85)
        cbar.ax.tick_params(labelsize=7)
        if zlabel:
            cbar.set_label(zlabel, fontsize=8)
    cs = ax.contour(gx, gy, Zm, levels=levels, colors="#202020", linewidths=0.7)
    ax.clabel(cs, fmt="%.4g", fontsize=7, inline=True)
    _design_points(ax, points)
    if solution is not None:
        ax.scatter([solution[0]], [solution[1]], marker="X", s=90, color="#ffffff",
                   edgecolor="#000000", zorder=7)
    _style(ax, title, xlabel, ylabel)
    ax.grid(False)
    return _png(fig)


def surface_plot(gx, gy, Z, xlabel: str, ylabel: str, zlabel: str, title: str,
                 points=None) -> bytes:
    fig = _figure(6.4, 5.0)
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(gx, gy, Z, cmap=DE_CMAP, linewidth=0.2, edgecolor="#555555",
                    antialiased=True, alpha=0.95)
    if points is not None and len(points):
        p = np.asarray(points)
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], color=POINT, s=18, depthshade=False)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_zlabel(zlabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.view_init(elev=28, azim=-130)
    return _png(fig)


def perturbation_plot(curves: list[tuple[str, np.ndarray, np.ndarray]], ylabel: str,
                      title: str, reference: str = "") -> bytes:
    fig = _figure()
    ax = fig.add_subplot()
    for i, (name, dev, y) in enumerate(curves):
        color = SERIES[i % len(SERIES)]
        ax.plot(dev, y, color=color, linewidth=1.6)
        j = int(len(dev) * (0.92 if i % 2 else 0.08))
        ax.annotate(name, (dev[j], y[j]), fontsize=9, fontweight="bold", color=color,
                    xytext=(0, 5), textcoords="offset points", ha="center")
    _style(ax, title, "Deviation from Reference Point (Coded Units)", ylabel)
    if reference:
        ax.text(0.99, 0.02, reference, transform=ax.transAxes, fontsize=7, ha="right",
                va="bottom", color="#555555")
    return _png(fig)


def line_plot(x: np.ndarray, curves: list[tuple[str, np.ndarray, np.ndarray | None,
                                                np.ndarray | None]],
              xlabel: str, ylabel: str, title: str, points=None, legend_title: str = ""
              ) -> bytes:
    """One-factor and interaction plots: lines with optional CI bands and design points."""
    fig = _figure()
    ax = fig.add_subplot()
    for i, (name, y, lo, hi) in enumerate(curves):
        color = SERIES[i % len(SERIES)] if len(curves) > 1 else "#1f3fbf"
        ax.plot(x, y, color=color, linewidth=1.8, label=name)
        if lo is not None and hi is not None:
            ax.plot(x, lo, color=color, linewidth=0.8, linestyle="--")
            ax.plot(x, hi, color=color, linewidth=0.8, linestyle="--")
    if points is not None:
        for i, pts in enumerate(points):
            if pts is not None and len(pts):
                p = np.asarray(pts)
                color = SERIES[i % len(SERIES)] if len(curves) > 1 else POINT
                ax.scatter(p[:, 0], p[:, 1], s=22, color=color, edgecolor="#333333", zorder=4)
    if len(curves) > 1:
        ax.legend(fontsize=8, title=legend_title, title_fontsize=8, frameon=False)
    _style(ax, title, xlabel, ylabel)
    return _png(fig)


def cube_plot(values: np.ndarray, labels: list[str], lows: list[str], highs: list[str],
              title: str) -> bytes:
    """Predicted response at the 8 corners (index bits: A = 1, B = 2, C = 4)."""
    fig = _figure(6.0, 5.0)
    ax = fig.add_subplot()
    ax.set_axis_off()
    off = np.array([0.38, 0.30])

    def pos(i):
        a, b, c = i & 1, i >> 1 & 1, i >> 2 & 1
        return np.array([a, b]) + c * off

    for i in range(8):
        for bit in (1, 2, 4):
            j = i | bit
            if j != i:
                p, q = pos(i), pos(j)
                hidden = 4 in (i, j) and not (i & 3 or j & 3) or (i == 4 and j in (5, 6))
                ax.plot([p[0], q[0]], [p[1], q[1]], color="#333333", linewidth=1.2,
                        linestyle="--" if hidden else "-")
    for i in range(8):
        p = pos(i)
        ax.scatter(*p, s=40, color=POINT, zorder=3)
        ax.annotate(f"{values[i]:.4g}", p, xytext=(6, 6), textcoords="offset points",
                    fontsize=9, fontweight="bold", color="#1f3fbf",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1})
    ax.text(0.5, -0.12, f"{labels[0]}", ha="center", fontsize=9)
    ax.text(0.0, -0.06, lows[0], ha="center", fontsize=7)
    ax.text(1.0, -0.06, highs[0], ha="center", fontsize=7)
    ax.text(-0.14, 0.5, f"{labels[1]}", va="center", rotation=90, fontsize=9)
    ax.text(-0.06, 0.0, lows[1], va="center", ha="right", fontsize=7)
    ax.text(-0.06, 1.0, highs[1], va="center", ha="right", fontsize=7)
    ax.text(1.0 + off[0] / 2 + 0.10, off[1] / 2 - 0.10, f"{labels[2]}", fontsize=9,
            rotation=38, ha="left", va="top")
    ax.text(1.02, -0.07, lows[2], fontsize=7, ha="left", va="top")
    ax.text(1.0 + off[0] + 0.04, off[1] - 0.07, highs[2], fontsize=7, ha="left", va="top")
    ax.set_xlim(-0.3, 1.6)
    ax.set_ylim(-0.2, 1.45)
    ax.set_title(title, fontsize=11, fontweight="bold")
    return _png(fig)


def _ternary_xy(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Proportions (a, b, c) -> plane coordinates; A at the top, B left, C right."""
    p = np.atleast_2d(p)
    s = p.sum(axis=1)
    a, b, c = p[:, 0] / s, p[:, 1] / s, p[:, 2] / s
    return 0.5 * a + c, a * math.sqrt(3) / 2


def ternary_contour(fn, lows, span: float, names: list[str], title: str, zlabel: str = "",
                    feasible=None, points=None, n: int = 70, n_levels: int = 10,
                    vertex_values: list[str] | None = None) -> bytes:
    """Contours over the (pseudo-component) simplex of three components.

    ``fn(actual (m x 3)) -> values``; ``lows`` the lower bounds and ``span`` the amount shared
    by the three components above their lower bounds; ``feasible(actual) -> bool mask``.
    """
    fig = _figure(6.0, 5.2)
    ax = fig.add_subplot()
    ax.set_aspect("equal")
    ax.set_axis_off()
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            pts.append((i, j, n - i - j))
    P = np.array(pts, float) / n
    actual = np.asarray(lows, float) + P * span
    Z = np.asarray(fn(actual), float)
    x, y = _ternary_xy(P)
    tri = Triangulation(x, y)
    ok = np.isfinite(Z)
    if feasible is not None:
        ok &= np.asarray(feasible(actual), bool)
    tri.set_mask(~np.all(ok[tri.triangles], axis=1))
    Zf = np.where(ok, Z, np.nan)
    zmin, zmax = np.nanmin(Zf), np.nanmax(Zf)
    if np.isfinite(zmin) and zmax > zmin:
        levels = np.linspace(zmin, zmax, n_levels + 1)
        Zc = np.where(ok, Z, zmin)
        cf = ax.tricontourf(tri, Zc, levels=levels, cmap=DE_CMAP, alpha=0.9)
        cs = ax.tricontour(tri, Zc, levels=levels, colors="#202020", linewidths=0.7)
        ax.clabel(cs, fmt="%.4g", fontsize=7, inline=True)
        cbar = fig.colorbar(cf, ax=ax, shrink=0.75)
        cbar.ax.tick_params(labelsize=7)
        if zlabel:
            cbar.set_label(zlabel, fontsize=8)
    h = math.sqrt(3) / 2
    ax.plot([0, 1, 0.5, 0], [0, 0, h, 0], color="#222222", linewidth=1.3)
    for t in np.linspace(0.2, 0.8, 4):
        for a, b in (((t, 1 - t, 0), (t, 0, 1 - t)), ((1 - t, t, 0), (0, t, 1 - t)),
                     ((1 - t, 0, t), (0, 1 - t, t))):
            xa, ya = _ternary_xy(np.array([a, b]))
            ax.plot(xa, ya, color="#cccccc", linewidth=0.5, zorder=1)
    vv = vertex_values or ["", "", ""]
    ax.text(0.5, h + 0.04, f"{names[0]}\n{vv[0]}", ha="center", va="bottom", fontsize=9)
    ax.text(-0.03, -0.04, f"{names[1]}\n{vv[1]}", ha="right", va="top", fontsize=9)
    ax.text(1.03, -0.04, f"{names[2]}\n{vv[2]}", ha="left", va="top", fontsize=9)
    if points is not None and len(points):
        pp = (np.asarray(points, float) - np.asarray(lows, float)) / span
        px, py = _ternary_xy(np.clip(pp, 0, None) + 1e-12)
        _design_points(ax, np.column_stack([px, py]))
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.2, h + 0.2)
    ax.set_title(title, fontsize=11, fontweight="bold")
    return _png(fig)


# ---------------------------------------------------------------------------- optimisation
def overlay_plot(gx, gy, limits: list[tuple[str, np.ndarray, float | None, float | None]],
                 xlabel: str, ylabel: str, title: str, solution=None, points=None) -> bytes:
    """Graphical optimisation: grey where a response is outside its limits, yellow window."""
    fig = _figure(6.2, 4.8)
    ax = fig.add_subplot()
    bad = np.zeros_like(gx, dtype=bool)
    for _name, Z, lo, hi in limits:
        if lo is not None:
            bad |= Z < lo
        if hi is not None:
            bad |= Z > hi
    ax.contourf(gx, gy, (~bad).astype(float), levels=[-0.5, 0.5, 1.5],
                colors=["#bdbdbd", "#fff176"])
    for i, (name, Z, lo, hi) in enumerate(limits):
        color = SERIES[i % len(SERIES)]
        for v in (lo, hi):
            if v is not None and np.nanmin(Z) < v < np.nanmax(Z):
                cs = ax.contour(gx, gy, Z, levels=[v], colors=[color], linewidths=1.4)
                ax.clabel(cs, fmt=lambda val, nm=name: f"{nm}: {val:.4g}", fontsize=7)
    _design_points(ax, points)
    if solution is not None:
        ax.scatter([solution[0]], [solution[1]], marker="X", s=90, color="#ffffff",
                   edgecolor="#000000", zorder=7)
    _style(ax, title, xlabel, ylabel)
    ax.grid(False)
    return _png(fig)


def ramps_plot(entries: list[dict], D: float, title: str = "Desirability Ramps") -> bytes:
    """entries: {name, low, high, value, goal, target?, kind} per factor/response."""
    n = len(entries)
    cols = min(4, max(n, 1))
    rows = math.ceil(n / cols) + 1
    fig = _figure(1.6 * cols + 0.6, 1.25 * rows + 0.4)
    for i, e in enumerate(entries):
        ax = fig.add_subplot(rows, cols, i + 1)
        lo, hi, v = e["low"], e["high"], e["value"]
        span = hi - lo if hi > lo else 1.0
        xs = [lo, hi]
        goal = e.get("goal")
        if goal == "maximize":
            ys = [0, 1]
        elif goal == "minimize":
            ys = [1, 0]
        elif goal == "target":
            t = e.get("target", (lo + hi) / 2)
            xs, ys = [lo, t, hi], [0, 1, 0]
        else:
            ys = [1, 1]
        ax.plot(xs, ys, color="#1f3fbf" if e.get("kind") == "factor" else "#2e7d32",
                linewidth=1.6)
        ax.scatter([min(max(v, lo - 0.1 * span), hi + 0.1 * span)], [0.5], s=50,
                   color=POINT, zorder=4)
        ax.set_xlim(lo - 0.1 * span, hi + 0.1 * span)
        ax.set_ylim(-0.1, 1.15)
        ax.set_yticks([])
        ax.set_xticks([lo, hi], [f"{lo:.4g}", f"{hi:.4g}"])
        ax.tick_params(labelsize=6)
        ax.set_title(f"{e['name']} = {v:.4g}", fontsize=7.5)
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


def desirability_bars(names: list[str], values: list[float], D: float) -> bytes:
    fig = _figure(6.0, 0.45 * (len(names) + 1) + 1.2)
    ax = fig.add_subplot()
    labels = names + ["Combined"]
    vals = list(values) + [D]
    colors = ["#1f3fbf"] * len(values) + ["#2e7d32"]
    ax.barh(range(len(vals))[::-1], vals, color=colors, height=0.6)
    ax.set_yticks(range(len(vals))[::-1], labels)
    for i, v in enumerate(vals):
        ax.text(min(v, 0.92) + 0.01, len(vals) - 1 - i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, 1)
    _style(ax, "Desirability", "Desirability", "")
    return _png(fig)


# ---------------------------------------------------------------------------- effects
def half_normal_plot(effects: list[dict], title: str = "Half-Normal Plot") -> bytes:
    fig = _figure()
    ax = fig.add_subplot()
    abs_e = np.array([abs(e["effect"]) for e in effects])
    order = np.argsort(abs_e)
    m = len(abs_e)
    q = stats.norm.ppf(0.5 + 0.5 * (np.arange(1, m + 1) - 0.5) / m)
    big = {e["name"] for e in effects if e.get("t", 0) > e.get("t_limit", np.inf)}
    for rank, i in enumerate(order):
        e = effects[i]
        sig = e["name"] in big
        ax.scatter(abs_e[i], q[rank], s=30, color=POINT if sig else "#1f3fbf",
                   edgecolor="#333333", zorder=3)
        if sig or rank >= m - 3:
            ax.annotate(e["name"], (abs_e[i], q[rank]), xytext=(4, 2),
                        textcoords="offset points", fontsize=8)
    small = order[: max(m - len(big), 1)]
    if len(small) > 1 and abs_e[small].max() > 0:
        slope = np.sum(abs_e[small] * q[: len(small)]) / np.sum(q[: len(small)] ** 2)
        qq = np.array([0, q.max()])
        ax.plot(qq * slope, qq, color=LINE, linewidth=1)
    ax.set_ylim(-0.05, q.max() * 1.08 if m else 1)
    ticks = [0, 10, 20, 30, 50, 70, 80, 90, 95, 99]
    ax.set_yticks(stats.norm.ppf(0.5 + np.array(ticks) / 200), [str(t) for t in ticks])
    _style(ax, title, "|Standardized Effect|", "Half-Normal % Probability")
    return _png(fig)


def pareto_chart(effects: list[dict], title: str = "Pareto Chart") -> bytes:
    order = sorted(effects, key=lambda e: -e.get("t", 0))
    fig = _figure(6.0, 4.5)
    ax = fig.add_subplot()
    ts = [e.get("t", 0) for e in order]
    colors = ["#ef6c00" if e["effect"] > 0 else "#1f3fbf" for e in order]
    ax.bar(range(len(ts)), ts, color=colors)
    ax.set_xticks(range(len(ts)), [e["name"] for e in order], rotation=90, fontsize=7)
    if order:
        ax.axhline(order[0]["t_limit"], color=LINE, linewidth=1)
        ax.axhline(order[0]["bonferroni"], color=LIMIT, linewidth=1)
        ax.text(len(ts) - 0.5, order[0]["t_limit"], f"t-Value Limit {order[0]['t_limit']:.3f}",
                ha="right", va="bottom", fontsize=7)
        ax.text(len(ts) - 0.5, order[0]["bonferroni"],
                f"Bonferroni Limit {order[0]['bonferroni']:.3f}", ha="right", va="bottom",
                fontsize=7, color=LIMIT)
    _style(ax, title, "Rank", "t-Value of |Effect|")
    return _png(fig)


def main_effects_plot(names: list[str], labels: list[list[str]], means: list[list[float]],
                      title: str, ylabel: str, overall: float | None = None) -> bytes:
    k = len(names)
    fig = _figure(max(2.0 * k, 5.0), 3.4)
    axes = fig.subplots(1, k, sharey=True, squeeze=False)[0]
    for ax, name, labs, m in zip(axes, names, labels, means):
        ax.plot(range(len(m)), m, color="#1f3fbf", marker="o", linewidth=1.6,
                markerfacecolor=POINT)
        if overall is not None:
            ax.axhline(overall, color="#9e9e9e", linewidth=0.8, linestyle="--")
        ax.set_xticks(range(len(m)), labs, fontsize=7)
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, color="#eeeeee")
        ax.set_xlim(-0.4, len(m) - 0.6)
    axes[0].set_ylabel(ylabel, fontsize=8)
    fig.suptitle(title, fontsize=11, fontweight="bold")
    return _png(fig)


# ---------------------------------------------------------------------------- design tools
def fds_plot(frac: np.ndarray, se: np.ndarray, title: str = "FDS Graph",
             marks=(0.5, 0.8)) -> bytes:
    """Fraction of design space: sorted prediction standard error (sigma units)."""
    fig = _figure()
    ax = fig.add_subplot()
    if len(frac):
        ax.plot(frac, se, color="#1f3fbf", linewidth=1.8)
        for m in marks:
            i = min(int(m * len(se)), len(se) - 1)
            ax.axvline(m, color="#9e9e9e", linewidth=0.8, linestyle="--")
            ax.annotate(f"{se[i]:.3f}", (m, se[i]), xytext=(4, -10), textcoords="offset points",
                        fontsize=7, color="#333333")
        ax.set_xlim(0, 1)
    _style(ax, title, "Fraction of Design Space", "Std Error Mean")
    return _png(fig)


def scatter_plot(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str = "",
                 color=None, color_label: str = "") -> bytes:
    """Graph Columns: one design column against another, optionally coloured by a third."""
    fig = _figure()
    ax = fig.add_subplot()
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if color is not None:
        c = np.asarray(color, float)
        ok &= np.isfinite(c)
        sc = ax.scatter(x[ok], y[ok], c=c[ok], cmap=DE_CMAP, s=36, edgecolor="#333333",
                        linewidth=0.5, zorder=3)
        cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(color_label, fontsize=8)
    else:
        ax.scatter(x[ok], y[ok], s=36, color=POINT, edgecolor="#5a0b0b", zorder=3)
    if ok.sum() >= 3 and np.ptp(x[ok]) > 0 and np.ptp(y[ok]) > 0:
        r = float(np.corrcoef(x[ok], y[ok])[0, 1])
        ax.text(0.99, 0.02, f"Correlation: {r:.3f}", transform=ax.transAxes, fontsize=8,
                ha="right", va="bottom", color="#555555")
    _style(ax, title or f"{ylabel} vs {xlabel}", xlabel, ylabel)
    return _png(fig)

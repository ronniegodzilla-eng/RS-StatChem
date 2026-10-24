"""SmartPLS-style path diagram: blue circles for constructs, yellow rectangles for indicators,
loadings / weights on the outer arrows, path coefficients (with p or t values) on the inner
arrows and R-square inside the endogenous constructs. Drawn with matplotlib's OO API."""
from __future__ import annotations

import io

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

from rsstatchem.core.stats.common import fmt
from rsstatchem.core.stats.pls_engine import Structure

R_CIRCLE = 0.62
BOX_H = 0.42
LV_FILL, LV_EDGE = "#a9cdf0", "#2f6fae"
MV_FILL, MV_EDGE = "#fbe7a1", "#b38f00"
INT_FILL, INT_EDGE = "#c9e8c5", "#3d8b37"


def _layout(st: Structure) -> dict[int, tuple[float, float]]:
    depth = np.zeros(st.K, int)
    for j in range(st.K):  # constructs are in topological order
        if len(st.preds[j]):
            depth[j] = 1 + max(depth[i] for i in st.preds[j])
    nlayer = depth.max() + 1
    dx, dy = 4.2, 3.0
    pos: dict[int, tuple[float, float]] = {}
    for layer in range(nlayer):
        members = [j for j in range(st.K) if depth[j] == layer]
        if layer:
            members.sort(key=lambda j: -np.mean([pos[i][1] for i in st.preds[j]]))
        m = len(members)
        for r, j in enumerate(members):
            pos[j] = (layer * dx, (m - 1) / 2 * dy - r * dy)
    # lift middle constructs that would sit on an arrow skipping over them (mediation)
    for j in range(st.K):
        if depth[j] in (0, nlayer - 1):
            continue
        x, y = pos[j]
        for a, b in st.edges():
            if depth[a] < depth[j] < depth[b]:
                ya = pos[a][1] + (pos[b][1] - pos[a][1]) * (x - pos[a][0]) / \
                    (pos[b][0] - pos[a][0])
                if abs(ya - y) < 0.9:
                    pos[j] = (x, y + 0.75 * dy if y >= ya else y - 0.75 * dy)
                    break
    return pos


def _side(st: Structure, j: int, pos: dict, depth_max: float) -> str:
    x, y = pos[j]
    if not len(st.preds[j]):
        return "left"
    if not len(st.succs[j]):
        return "right"
    ys = [pos[k][1] for k in range(st.K)]
    return "top" if y >= np.mean(ys) else "bottom"


def _edge_point(center, target, kind: str, box_w: float = 0.95) -> tuple[float, float]:
    cx, cy = center
    vx, vy = target[0] - cx, target[1] - cy
    norm = np.hypot(vx, vy) or 1.0
    if kind == "circle":
        return cx + vx / norm * R_CIRCLE, cy + vy / norm * R_CIRCLE
    tx = (box_w / 2) / abs(vx) if vx else np.inf
    ty = (BOX_H / 2) / abs(vy) if vy else np.inf
    t = min(tx, ty)
    return cx + vx * t, cy + vy * t


def _seg_dist(pt, a, b) -> float:
    a, b, pt = np.asarray(a, float), np.asarray(b, float), np.asarray(pt, float)
    ab = b - a
    t = np.clip(np.dot(pt - a, ab) / (np.dot(ab, ab) or 1.0), 0, 1)
    return float(np.hypot(*(pt - a - t * ab)))


def path_diagram(st: Structure, weights: np.ndarray, loadings: np.ndarray,
                 paths: np.ndarray, r2: np.ndarray, inner_extra: dict | None = None,
                 outer_extra: dict | None = None, title: str = "") -> bytes:
    """``inner_extra[(i, j)]`` / ``outer_extra[k]`` = text shown in brackets (p or t)."""
    inner_extra = inner_extra or {}
    outer_extra = outer_extra or {}
    box_w = max(0.95, 0.14 * max(len(n) for n in st.indicators) + 0.25)
    pos = _layout(st)
    depth_max = max(p[0] for p in pos.values())
    boxes: dict[int, tuple[float, float]] = {}
    for j in range(st.K):
        if st.is_interaction(j):
            continue
        blk = st.blocks[j]
        m = len(blk)
        cx, cy = pos[j]
        side = _side(st, j, pos, depth_max)
        for r, k in enumerate(blk):
            if side in ("left", "right"):
                off = (m - 1) / 2 * 0.62 - r * 0.62
                gap = 1.9 + box_w / 2
                bx = cx - gap if side == "left" else cx + gap
                boxes[k] = (bx, cy + off)
            else:
                step = box_w + 0.13
                off = -(m - 1) / 2 * step + r * step
                by = cy + 2.4 if side == "top" else cy - 2.4
                boxes[k] = (cx + off, by)
    xs = [p[0] for p in pos.values()] + [b[0] for b in boxes.values()]
    ys = [p[1] for p in pos.values()] + [b[1] for b in boxes.values()]
    x0, x1 = min(xs) - 1.0, max(xs) + 1.0
    y0, y1 = min(ys) - 0.9, max(ys) + 0.9
    scale = 0.45
    fig = Figure(figsize=(max(5.0, (x1 - x0) * scale), max(3.0, (y1 - y0) * scale)), dpi=120)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.text(x0 + 0.1, y1 - 0.1, title, ha="left", va="top", fontsize=8, color="#555")

    def arrow(p, q, color="#333", lw=1.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, color=color,
                                     lw=lw, shrinkA=0, shrinkB=0, zorder=2))

    def label(x, y, text, size=7.5, bold=False):
        ax.text(x, y, text, ha="center", va="center", fontsize=size, zorder=5,
                fontweight="bold" if bold else "normal",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))

    for k, (bx, by) in boxes.items():
        j = st.owner[k]
        ax.add_patch(Rectangle((bx - box_w / 2, by - BOX_H / 2), box_w, BOX_H, fc=MV_FILL,
                               ec=MV_EDGE, lw=0.8, zorder=3))
        ax.text(bx, by, st.indicators[k], ha="center", va="center", fontsize=8, zorder=4)
        if st.reflective(j):
            p = _edge_point(pos[j], (bx, by), "circle")
            q = _edge_point((bx, by), pos[j], "box", box_w)
            value = loadings[k, j]
        else:
            p = _edge_point((bx, by), pos[j], "box", box_w)
            q = _edge_point(pos[j], (bx, by), "circle")
            value = weights[k, j]
        arrow(p, q, "#666", 0.8)
        mx, my = p[0] + (q[0] - p[0]) * 0.5, p[1] + (q[1] - p[1]) * 0.5
        text = fmt(value)
        if k in outer_extra:
            text += f" ({outer_extra[k]})"
        label(mx, my, text, 7.0)
    placed: list[tuple[float, float]] = []
    for i, j in st.edges():
        p, q = _edge_point(pos[i], pos[j], "circle"), _edge_point(pos[j], pos[i], "circle")
        arrow(p, q, "#222", 1.3)
        others = [(pos[a], pos[b]) for a, b in st.edges() if (a, b) != (i, j)]
        for frac in (0.5, 0.35, 0.65, 0.25, 0.75, 0.5):  # keep labels off crossing arrows
            mx, my = p[0] + (q[0] - p[0]) * frac, p[1] + (q[1] - p[1]) * frac
            if all(abs(mx - a) > 1.1 or abs(my - b) > 0.6 for a, b in placed) and \
                    all(_seg_dist((mx, my), a, b) > 0.45 for a, b in others):
                break
        placed.append((mx, my))
        text = fmt(paths[i, j])
        if (i, j) in inner_extra:
            text += f"\n({inner_extra[(i, j)]})"
        label(mx, my, text, 8.5, bold=True)
    for j in range(st.K):
        cx, cy = pos[j]
        inter = st.is_interaction(j)
        ax.add_patch(Circle((cx, cy), R_CIRCLE, fc=INT_FILL if inter else LV_FILL,
                            ec=INT_EDGE if inter else LV_EDGE, lw=1.2, zorder=3))
        name = st.names[j] if len(st.names[j]) <= 9 else st.names[j][:8] + "…"
        if np.isfinite(r2[j]):
            ax.text(cx, cy + 0.17, name, ha="center", va="center", fontsize=8.5,
                    fontweight="bold", zorder=4)
            ax.text(cx, cy - 0.22, fmt(r2[j]), ha="center", va="center", fontsize=8.5,
                    zorder=4)
        else:
            ax.text(cx, cy, name, ha="center", va="center", fontsize=8.5, fontweight="bold",
                    zorder=4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    return buf.getvalue()

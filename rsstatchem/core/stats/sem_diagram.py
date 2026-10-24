"""AMOS-style path diagram for a fitted CB-SEM model (matplotlib OO API, no pyplot).

Latent variables are ellipses, observed variables rectangles; single-headed arrows carry the
(standardised or unstandardised) estimates, curved double-headed arrows the covariances or
correlations, and R² is printed beside endogenous variables. Layout: structural variables in
columns by causal order (longest path), indicators fanned out to the outer side (left for the
first column, right for the last, below for middle columns). A pure CFA puts the factors in a
column with their indicators on the left.
"""
from __future__ import annotations

import io
import math

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch

from rsstatchem.core.stats.common import fmt
from rsstatchem.core.stats.sem_syntax import SEMModel

LATENT_FILL = "#fde8e8"
OBS_FILL = "#ffffff"
EDGE = "#7a1f1f"
ARROW = "#333333"
COV = "#1f4e79"

IND_GAP = 0.78       # vertical distance between stacked indicators
COL_GAP = 4.2        # distance between structural columns
IND_OFFSET = 2.6     # distance from a latent to its indicator column


def _layers(model: SEMModel, struct: list[str]) -> dict[str, int]:
    preds: dict[str, set] = {v: set() for v in struct}
    for p in model.params:
        if p.op == "~" and p.lhs in preds and p.rhs in preds:
            preds[p.lhs].add(p.rhs)
        elif p.op == "=~" and p.rhs in preds and p.lhs in preds:
            preds[p.rhs].add(p.lhs)
    layer = {v: 0 for v in struct}
    for _ in range(len(struct) + 1):
        changed = False
        for v in struct:
            new = max((layer[u] + 1 for u in preds[v]), default=0)
            new = min(new, len(struct))
            if new != layer[v]:
                layer[v], changed = new, True
        if not changed:
            break
    used = sorted(set(layer.values()))
    remap = {old: k for k, old in enumerate(used)}
    return {v: remap[k] for v, k in layer.items()}


def _box_width(name: str) -> float:
    return max(0.95, 0.13 * len(name) + 0.35)


def layout(model: SEMModel) -> tuple[dict, dict]:
    """Positions {name: (x, y)} and, for indicators, their side ('left'/'right'/'below')."""
    struct = model.structural()
    ind_parent: dict[str, str] = {}
    for f in model.latent:
        for x in model.indicators.get(f, []):
            if x not in model.latent:
                ind_parent.setdefault(x, f)
    own = {f: [x for x in model.indicators.get(f, []) if ind_parent.get(x) == f]
           for f in model.latent}
    layer = _layers(model, struct)
    n_layers = max(layer.values(), default=0) + 1
    pos: dict[str, tuple[float, float]] = {}
    side: dict[str, str] = {}

    def side_of(v):
        if n_layers == 1:
            return "left"
        if layer[v] == 0:
            return "left"
        if layer[v] == n_layers - 1:
            return "right"
        return "below"

    columns: dict[int, list[str]] = {}
    for v in struct:
        columns.setdefault(layer[v], []).append(v)
    for k, nodes in columns.items():
        heights = []
        for v in nodes:
            n_ind = len(own.get(v, []))
            s = side_of(v)
            if s == "below" and n_ind:
                heights.append(2.9)
            else:
                heights.append(max(1.5, n_ind * IND_GAP + 0.4))
        total = sum(heights)
        y = total / 2
        for v, h in zip(nodes, heights):
            cy = y - (h / 2 if side_of(v) != "below" or not own.get(v) else 0.6)
            pos[v] = (k * COL_GAP, cy)
            y -= h
        if len(nodes) == 1:
            pos[nodes[0]] = (k * COL_GAP, 0.0)
    for f, items in own.items():
        if f not in pos or not items:
            continue
        fx, fy = pos[f]
        s = side_of(f)
        if s in ("left", "right"):
            dx = -IND_OFFSET if s == "left" else IND_OFFSET
            top = fy + (len(items) - 1) * IND_GAP / 2
            for i, x in enumerate(items):
                pos[x] = (fx + dx, top - i * IND_GAP)
                side[x] = s
        else:
            widths = [_box_width(x) + 0.25 for x in items]
            total = sum(widths)
            cx = fx - total / 2
            for x, w in zip(items, widths):
                pos[x] = (cx + w / 2, fy - 1.75)
                side[x] = "below"
                cx += w
    return pos, side


def path_diagram(model: SEMModel, values: dict[str, float], r2: dict[str, float],
                 standardized: bool = True, title: str | None = None) -> bytes:
    """PNG bytes. ``values`` maps parameter names (``lhs op rhs`` without spaces) to the
    estimate to print; ``r2`` maps endogenous variable names to R²."""
    pos, side = layout(model)
    xs = [p[0] for p in pos.values()] or [0]
    ys = [p[1] for p in pos.values()] or [0]
    x0, x1 = min(xs) - 1.6, max(xs) + 2.2
    y0, y1 = min(ys) - 1.1, max(ys) + 1.1
    w_units, h_units = x1 - x0, y1 - y0
    scale = min(1.15, 13.0 / w_units, 15.0 / h_units)
    fig = Figure(figsize=(max(w_units * scale, 5.5), max(h_units * scale, 3.6)), dpi=130)
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1 if not title else 0.94))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=10, y=0.985)
    patches = {}
    for name, (x, y) in pos.items():
        if name in model.latent:
            w = max(1.5, 0.14 * len(name) + 0.6)
            patch = Ellipse((x, y), w, 0.95, facecolor=LATENT_FILL, edgecolor=EDGE, lw=1.4,
                            zorder=3)
        else:
            w = _box_width(name)
            patch = FancyBboxPatch((x - w / 2, y - 0.25), w, 0.5,
                                   boxstyle="square,pad=0.0", facecolor=OBS_FILL,
                                   edgecolor=EDGE, lw=1.2, zorder=3)
        ax.add_patch(patch)
        patches[name] = patch
        ax.text(x, y, name, ha="center", va="center", fontsize=8.5, zorder=4,
                fontweight="bold" if name in model.latent else "normal")

    def label(x, y, text, color=ARROW, size=7.5):
        ax.text(x, y, text, ha="center", va="center", fontsize=size, color=color, zorder=6,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))

    def val(name):
        v = values.get(name)
        return "" if v is None else fmt(v, 2)

    half = {n: _half_size(n, model) for n in pos}

    def blocked(a, b) -> bool:
        """Does the straight segment a-b pass through another node?"""
        (x0_, y0_), (x1_, y1_) = pos[a], pos[b]
        dx, dy = x1_ - x0_, y1_ - y0_
        L2 = dx * dx + dy * dy or 1.0
        for n, (x, y) in pos.items():
            if n in (a, b):
                continue
            t = ((x - x0_) * dx + (y - y0_) * dy) / L2
            if 0.05 < t < 0.95:
                px, py = x0_ + t * dx, y0_ + t * dy
                hw, hh = half[n]
                if abs(px - x) < hw + 0.05 and abs(py - y) < hh + 0.05:
                    return True
        return False

    for p in model.params:
        if p.op not in ("=~", "~"):
            continue
        src, dst = (p.lhs, p.rhs) if p.op == "=~" else (p.rhs, p.lhs)
        if src not in pos or dst not in pos:
            continue
        (sx, sy), (tx, ty) = pos[src], pos[dst]
        rad = 0.0
        if dst not in side and blocked(src, dst):
            rad = -0.22 if tx >= sx else 0.22          # bow upwards
        arrow = FancyArrowPatch(pos[src], pos[dst], patchA=patches[src], patchB=patches[dst],
                                arrowstyle="-|>", mutation_scale=11, lw=1.0, color=ARROW,
                                connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=1,
                                zorder=2)
        ax.add_patch(arrow)
        frac = 0.62 if (p.op == "=~" and dst in side) else 0.5
        lx, ly = sx + frac * (tx - sx), sy + frac * (ty - sy)
        if rad:
            dx, dy = tx - sx, ty - sy
            lx, ly = lx + 0.5 * rad * dy, ly - 0.5 * rad * dx
        label(lx, ly, val(p.name))

    for p in model.params:
        if p.op != "~~" or p.lhs == p.rhs or p.lhs not in pos or p.rhs not in pos:
            continue
        a, b = p.lhs, p.rhs
        pa, pb, rad = _cov_geometry(a, b, pos, side, half)
        arrow = FancyArrowPatch(pa, pb, arrowstyle="<|-|>", mutation_scale=9, lw=0.9,
                                color=COV, connectionstyle=f"arc3,rad={rad}", shrinkA=1,
                                shrinkB=1, zorder=1)
        ax.add_patch(arrow)
        dx, dy = pb[0] - pa[0], pb[1] - pa[1]
        mx = (pa[0] + pb[0]) / 2 + 0.5 * rad * dy
        my = (pa[1] + pb[1]) / 2 - 0.5 * rad * dx
        label(mx, my, val(p.name), color=COV)

    for v, r in r2.items():
        if v not in pos or r is None or not math.isfinite(r):
            continue
        x, y = pos[v]
        text = f"R² = {fmt(r, 2)}"
        if v in side:
            s = side[v]
            if s == "left":
                ax.text(x - _box_width(v) / 2 - 0.08, y, fmt(r, 2), ha="right", va="center",
                        fontsize=6.8, color="#666666", style="italic")
            elif s == "right":
                ax.text(x + _box_width(v) / 2 + 0.08, y, fmt(r, 2), ha="left", va="center",
                        fontsize=6.8, color="#666666", style="italic")
            else:
                ax.text(x, y - 0.42, fmt(r, 2), ha="center", va="center", fontsize=6.8,
                        color="#666666", style="italic")
        else:
            ax.text(x, y + 0.66, text, ha="center", va="center", fontsize=7.5,
                    color="#7a1f1f", style="italic")
    note = ("Standardized estimates" if standardized else "Unstandardized estimates") + \
        "; small italic numbers / R² = squared multiple correlations"
    ax.text(x0 + 0.1, y0 + 0.1, note, fontsize=6.5, color="#666666", ha="left", va="bottom")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    return buf.getvalue()


def _half_size(name: str, model: SEMModel) -> tuple[float, float]:
    if name in model.latent:
        return max(1.5, 0.14 * len(name) + 0.6) / 2, 0.475
    return _box_width(name) / 2, 0.25


def _cov_geometry(a: str, b: str, pos: dict, side: dict, half: dict):
    """Anchor points on the outer edges and the arc3 bend for a covariance arrow, so arcs
    bulge away from the diagram and longer arcs clear the nodes in between."""
    (ax_, ay), (bx, by) = pos[a], pos[b]
    s = side.get(a) if side.get(a) == side.get(b) else None
    vertical = abs(bx - ax_) < 1e-6
    horizontal = abs(by - ay) < 1e-6
    if s == "left":
        pa, pb = (ax_ - half[a][0], ay), (bx - half[b][0], by)
        rad = 0.45 if ay > by else -0.45
    elif s == "right" or (s is None and vertical):
        pa, pb = (ax_ + half[a][0], ay), (bx + half[b][0], by)
        rad = -0.45 if ay > by else 0.45
    elif s == "below":
        pa, pb = (ax_, ay - half[a][1]), (bx, by - half[b][1])
        rad = 0.45 if ax_ < bx else -0.45
    elif horizontal:
        pa, pb = (ax_, ay + half[a][1]), (bx, by + half[b][1])
        rad = -0.3 if ax_ < bx else 0.3
    else:
        pa, pb = (ax_, ay), (bx, by)
        rad = 0.25
        dx, dy = bx - ax_, by - ay
        d = math.hypot(dx, dy) or 1.0
        pa = (ax_ + dx / d * half[a][0], ay + dy / d * half[a][1])
        pb = (bx - dx / d * half[b][0], by - dy / d * half[b][1])
    dist = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
    if dist < 1.2:                     # adjacent indicators: a tight loop
        rad = math.copysign(0.9, rad)
    return pa, pb, rad

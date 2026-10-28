"""Design-evaluation graphs (PNG bytes, Agg canvas): correlation and alias heat maps, the
standard error of the design over two factors, the FDS graph with a precision goal and the
design points of a constrained region."""
from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from rsstatchem.core.doe.plots import DE_CMAP, _figure, _png, _style

CORR_CMAP = LinearSegmentedColormap.from_list("corr", ["#1f3fbf", "#ffffff", "#e2442b"])


def heatmap(R: np.ndarray, rows: list[str], cols: list[str] | None = None,
            title: str = "Correlation", vmin: float = -1.0, vmax: float = 1.0,
            annotate: bool = True) -> bytes:
    """A labelled heat map of R (blue negative, red positive)."""
    cols = rows if cols is None else cols
    R = np.asarray(R, float)
    n, m = R.shape if R.ndim == 2 else (0, 0)
    size = max(4.5, min(0.42 * max(n, m) + 2.2, 11))
    fig = _figure(size, max(3.6, min(0.42 * n + 1.8, 10)))
    ax = fig.add_subplot()
    if n and m:
        im = ax.imshow(R, cmap=CORR_CMAP, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(m))
        ax.set_xticklabels(cols, rotation=90 if m > 8 else 0, fontsize=7)
        ax.set_yticks(range(n))
        ax.set_yticklabels(rows, fontsize=7)
        if annotate and n * m <= 400:
            for i in range(n):
                for j in range(m):
                    v = R[i, j]
                    if abs(v) > 1e-9:
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                                color="white" if abs(v) > 0.6 else "#222222")
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        cb.ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=11, fontweight="bold")
    return _png(fig)


def se_contour(G1, G2, se, xlabel: str, ylabel: str, points=None,
               title: str = "Standard Error of Design") -> bytes:
    """Contour of the standard error of the mean (sigma units) with the design points."""
    fig = _figure()
    ax = fig.add_subplot()
    cs = ax.contourf(G1, G2, se, levels=12, cmap=DE_CMAP)
    ax.contour(G1, G2, se, levels=12, colors="#333333", linewidths=0.4)
    cb = fig.colorbar(cs, ax=ax, shrink=0.85)
    cb.set_label("Std Error of Design", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    if points is not None and len(points):
        pts, counts = np.unique(np.round(np.asarray(points, float), 6), axis=0,
                                return_counts=True)
        ax.scatter(pts[:, 0], pts[:, 1], s=26, color="#C62828", edgecolor="white",
                   linewidth=0.6, zorder=4)
        for (x, y), c in zip(pts, counts):
            if c > 1:
                ax.annotate(str(c), (x, y), xytext=(4, 3), textcoords="offset points",
                            fontsize=7)
    _style(ax, title, xlabel, ylabel)
    ax.grid(False)
    return _png(fig)


def fds_plot(se: np.ndarray, title: str = "FDS Graph", limit: float | None = None,
             fraction: float | None = None, ylabel: str = "Std Error Mean") -> bytes:
    """Sorted standard errors against the fraction of design space, with the precision
    limit (horizontal line) and the FDS score."""
    fig = _figure()
    ax = fig.add_subplot()
    if len(se):
        frac = (np.arange(1, len(se) + 1) - 0.5) / len(se)
        ax.plot(frac, se, color="#1f3fbf", linewidth=1.8)
        for m in (0.5, 0.8):
            i = min(int(m * len(se)), len(se) - 1)
            ax.axvline(m, color="#9e9e9e", linewidth=0.8, linestyle="--")
            ax.annotate(f"{se[i]:.3f}", (m, se[i]), xytext=(4, -10),
                        textcoords="offset points", fontsize=7)
        if limit is not None and np.isfinite(limit):
            ax.axhline(limit, color="#d32f2f", linewidth=1.1)
            if fraction is not None and np.isfinite(fraction):
                ax.axvline(fraction, color="#d32f2f", linewidth=0.9, linestyle=":")
                ax.annotate(f"FDS = {100 * fraction:.0f} %", (min(fraction, 0.98), limit),
                            xytext=(-60, 6), textcoords="offset points", fontsize=8,
                            color="#d32f2f")
        ax.set_xlim(0, 1)
    _style(ax, title, "Fraction of Design Space", ylabel)
    return _png(fig)


def region_points(vertices: np.ndarray, points: np.ndarray, labels: tuple[str, str],
                  title: str = "Design Space") -> bytes:
    """Two-factor view of a constrained region (its vertices' hull) and the design points."""
    from scipy.spatial import ConvexHull

    fig = _figure()
    ax = fig.add_subplot()
    V = np.asarray(vertices, float)
    if len(V) >= 3:
        try:
            hull = ConvexHull(V)
            poly = V[hull.vertices]
            ax.fill(poly[:, 0], poly[:, 1], color="#E3EEFB", edgecolor="#1F4E8C")
        except Exception:  # noqa: BLE001 - degenerate hull
            pass
    P = np.asarray(points, float)
    if len(P):
        ax.scatter(P[:, 0], P[:, 1], s=28, color="#C62828", zorder=4)
    _style(ax, title, labels[0], labels[1])
    return _png(fig)


__all__ = ["heatmap", "se_contour", "fds_plot", "region_points"]

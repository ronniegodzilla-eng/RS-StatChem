"""3D and specialised plot types: 3D bars, 3D line/trajectory, 3D waterfall (3D layers),
radar/spider and wind rose (polar layers). No Qt here.

Every function draws one plot and returns its legend handle (or None)."""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Patch


def _R():
    import sys
    return sys.modules["rsstatchem.core.graph.render"]


# ====================================================================== 3D layer
def bar3d(ctx, li, layer, ax, p, i):
    """3D bars at (X, Y) with height Z (Origin's 3D Bars from XYZ)."""
    x, y, z = ctx.arr(p.x), ctx.arr(p.y), ctx.arr(p.z)
    if x is None or y is None or z is None:
        raise ValueError("3D bars need X, Y and Z columns")
    n = min(len(x), len(y), len(z))
    x, y, z = x[:n], y[:n], z[:n]
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]

    def width(v):
        u = np.unique(v)
        return float(np.min(np.diff(u))) * 0.7 if len(u) > 1 else 0.7
    dx, dy = width(x) * float(p.options.get("width", 1.0)), width(y) * float(
        p.options.get("width", 1.0))
    if p.options.get("color_by_z", True):
        import matplotlib
        cmap = matplotlib.colormaps[p.options.get("cmap", layer.cmap)]
        span = np.ptp(z) or 1.0
        colors = [cmap((v - z.min()) / span) for v in z]
    else:
        colors = ctx.fill_color(p, i)
    art = ax.bar3d(x - dx / 2, y - dy / 2, np.zeros(len(z)), dx, dy, z, color=colors,
                   shade=True, edgecolor=p.fill.edge_color if p.fill.edge_width > 0 else None,
                   linewidth=p.fill.edge_width * 0.4, alpha=p.fill.alpha)
    _R()._register(ctx, li, p, ax, art)
    return Patch(facecolor=ctx.fill_color(p, i))


def line3d(ctx, li, layer, ax, p, i):
    """3D line / trajectory through (X, Y, Z) with optional symbols."""
    x, y, z = ctx.arr(p.x), ctx.arr(p.y), ctx.arr(p.z)
    if x is None or y is None or z is None:
        raise ValueError("3D lines need X, Y and Z columns")
    n = min(len(x), len(y), len(z))
    m = np.isfinite(x[:n]) & np.isfinite(y[:n]) & np.isfinite(z[:n])
    kw = {"color": ctx.line_color(p, i), "lw": p.line.width,
          "ls": ctx.dash(p, i) if p.line.visible else "none", "label": p.name}
    kw.update(ctx.symbol_kw(p, i) if p.symbol.visible else {"marker": "none"})
    (h,) = ax.plot(x[:n][m], y[:n][m], z[:n][m], **kw)
    arts = [h]
    if p.options.get("drop_lines"):
        zmin = float(np.nanmin(z[:n][m]))
        for xv, yv, zv in zip(x[:n][m], y[:n][m], z[:n][m]):
            arts += ax.plot([xv, xv], [yv, yv], [zmin, zv], color="#999999", lw=0.5)
    if p.options.get("projection"):
        zmin = float(np.nanmin(z[:n][m]))
        arts += ax.plot(x[:n][m], y[:n][m], np.full(m.sum(), zmin), color="#aaaaaa",
                        lw=p.line.width * 0.7, ls="--")
    _R()._register(ctx, li, p, ax, arts)
    return h


def waterfall3d(ctx, li, layer, ax, p, i):
    """One curve of a 3D waterfall: Y(X) drawn at depth ``options["depth"]`` (Z position),
    optionally filled to the floor."""
    R = _R()
    x, y = R._xy(ctx, p)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    depth = float(p.options.get("depth", i))
    col = ctx.line_color(p, i)
    (h,) = ax.plot(x, np.full(len(x), depth), y, color=col, lw=p.line.width, label=p.name)
    arts = [h]
    if p.options.get("fill", True) and len(x) > 1:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        base = float(p.options.get("base", min(0.0, float(np.min(y)))))
        verts = [list(zip(x, np.full(len(x), depth), y)) +
                 [(x[-1], depth, base), (x[0], depth, base)]]
        poly = Poly3DCollection(verts, facecolor=ctx.fill_color(p, i), edgecolor=col,
                                alpha=p.fill.alpha if p.fill.alpha < 1 else 0.6, lw=0.3)
        ax.add_collection3d(poly)
        arts.append(poly)
    R._register(ctx, li, p, ax, arts)
    return h


D3 = {"bar3d": bar3d, "line3d": line3d, "waterfall3d": waterfall3d}


# ====================================================================== polar layer
def radar(ctx, li, layer, ax, p, i):
    """Radar / spider: Y values spread over equal angles (categories from X or labels),
    closed polygon with an optional fill."""
    R = _R()
    y = ctx.arr(p.y)
    if y is None:
        raise ValueError("radar needs a Y column")
    lab = ctx.col(p.labels) or ctx.col(p.x)
    n = len(y)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    m = np.isfinite(y)
    col = ctx.line_color(p, i)
    tc, yc = np.r_[th[m], th[m][:1]], np.r_[y[m], y[m][:1]]
    kw = {"color": col, "lw": p.line.width, "ls": ctx.dash(p, i), "label": p.name}
    kw.update(ctx.symbol_kw(p, i) if p.symbol.visible else {"marker": "none"})
    (h,) = ax.plot(tc, yc, **kw)
    arts = [h]
    if p.options.get("fill", True):
        arts += ax.fill(tc, yc, color=ctx.fill_color(p, i) if p.fill.color != "auto" else col,
                        alpha=p.fill.alpha if p.fill.alpha < 1 else 0.25)
    if lab is not None:
        vals = lab.array()[:n]
        names = [str(v) if lab.kind == "text" else lab.value_labels.get(v, R._num(v))
                 for v in vals]
    else:
        names = [str(k + 1) for k in range(n)]
    ax.set_xticks(th)
    ax.set_xticklabels([R.rich(t) for t in names])
    R._register(ctx, li, p, ax, arts, th[m], y[m])
    return h


def windrose(ctx, li, layer, ax, p, i):
    """Wind rose: X = direction (degrees), Y = speed; stacked bars of counts per direction
    sector and speed bin (Origin's Wind Rose - Raw Data)."""
    import matplotlib
    R = _R()
    d, s = ctx.arr(p.x), ctx.arr(p.y)
    if d is None or s is None:
        raise ValueError("wind rose needs direction (X) and speed (Y) columns")
    n = min(len(d), len(s))
    d, s = d[:n], s[:n]
    m = np.isfinite(d) & np.isfinite(s)
    d, s = np.mod(d[m], 360.0), s[m]
    sectors = int(p.options.get("sectors", 16))
    nbins = int(p.options.get("speed_bins", 5))
    width = 360.0 / sectors
    sec = np.floor(np.mod(d + width / 2, 360.0) / width).astype(int) % sectors
    edges = np.linspace(s.min(), s.max() if s.max() > s.min() else s.min() + 1, nbins + 1)
    cmap = matplotlib.colormaps[p.options.get("cmap", "viridis")]
    th = np.deg2rad(np.arange(sectors) * width)
    bottom = np.zeros(sectors)
    pct = p.options.get("percent", True)
    arts, handles = [], []
    for k in range(nbins):
        hi_edge = edges[k + 1] + (1e-12 if k == nbins - 1 else 0)
        sel = (s >= edges[k]) & (s < hi_edge)
        counts = np.bincount(sec[sel], minlength=sectors).astype(float)
        if pct and len(s):
            counts = counts / len(s) * 100
        bars = ax.bar(th, counts, width=np.deg2rad(width) * 0.9, bottom=bottom,
                      color=cmap(k / max(1, nbins - 1)), edgecolor="#333333", lw=0.3)
        bottom += counts
        arts += list(bars.patches)
        handles.append(bars)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    R._register(ctx, li, p, ax, arts)
    return handles[0] if handles else None


POLAR = {"radar": radar, "windrose": windrose}

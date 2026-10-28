"""Analyze > Surface Plot...: a 3-D surface of pixel values rendered as an RGB image, in the
style of ImageJ 1.x's SurfacePlotter (polygon multiplier, wireframe, shading, axes, black fill,
smoothing, one polygon per line, source background lighter, perspective-free 3/4 view).

Written for RS-StatChem after ImageJ 1.54 ij/plugin/SurfacePlotter.java (public domain, NIH);
the rendering itself is a numpy/Pillow painter's algorithm, not a line-by-line port.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def surface_plot(image, polygon_multiplier: int = 100, draw_wireframe: bool = False,
                 shade: bool = True, draw_axis: bool = True, source_lighter: bool = False,
                 fill_black: bool = False, one_per_line: bool = False, smooth: bool = True,
                 plot_width: int = 350, angle_x: float = 35.0, angle_z: float = 30.0,
                 roi=None):
    """RGB uint8 array (H, W, 3) of the surface of the selection (or whole plane)."""
    from PIL import Image, ImageDraw

    from rsstatchem.core.imaging.lut import render
    from rsstatchem.core.imaging.profile import value_plane
    from rsstatchem.core.imaging.roi import geometry as G

    vp = value_plane(image, True)
    rgb = render(image)
    roi = image.roi if roi is None else roi
    if roi is not None and roi.is_area():
        x, y, w, h = G.int_bounds(roi)
        x0, y0 = max(x, 0), max(y, 0)
        vp = vp[y0:y0 + h, x0:x0 + w]
        rgb = rgb[y0:y0 + h, x0:x0 + w]
    if smooth:
        vp = ndimage.uniform_filter(vp, 3, mode="nearest")
    h, w = vp.shape
    n_lines = max(2, min(h, int(round(h * polygon_multiplier / 100.0 / 2)) + 1))
    rows = np.linspace(0, h - 1, n_lines).astype(int)
    step = max(1, w // 300)
    cols = np.arange(0, w, step)
    vmin, vmax = float(np.nanmin(vp)), float(np.nanmax(vp))
    rng = vmax - vmin or 1.0
    scale = plot_width / max(w, h)
    zscale = 0.5 * plot_width / rng
    ax, az = np.radians(angle_x), np.radians(angle_z)

    def project(px, py, pz):
        X = (px - w / 2) * scale
        Y = (py - h / 2) * scale
        xr = X * np.cos(az) - Y * np.sin(az)
        yr = X * np.sin(az) + Y * np.cos(az)
        sx = xr
        sy = yr * np.sin(ax) - (pz - vmin) * zscale * np.cos(ax)
        return sx, sy

    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], float)
    cx, cy = project(corners[:, 0], corners[:, 1], np.array([vmin] * 4))
    tx, ty = project(corners[:, 0], corners[:, 1], np.array([vmax] * 4))
    allx = np.concatenate([cx, tx])
    ally = np.concatenate([cy, ty])
    margin = 30
    W = int(allx.max() - allx.min()) + 2 * margin
    H = int(ally.max() - ally.min()) + 2 * margin
    ox, oy = -allx.min() + margin, -ally.min() + margin
    bg = (0, 0, 0) if fill_black else (255, 255, 255)
    canvas = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(canvas)
    if draw_axis:
        base = [(cx[0] + ox, cy[0] + oy), (cx[1] + ox, cy[1] + oy), (cx[3] + ox, cy[3] + oy),
                (cx[2] + ox, cy[2] + oy)]
        d.polygon(base, outline=(128, 128, 128))
        d.line([(cx[0] + ox, cy[0] + oy), (tx[0] + ox, ty[0] + oy)], fill=(128, 128, 128))
    order = rows if np.cos(az) >= 0 else rows[::-1]
    for r in order:
        z = vp[r, cols]
        sx, sy = project(cols.astype(float), np.full(len(cols), float(r)), z)
        bx, by = project(cols.astype(float), np.full(len(cols), float(r)),
                         np.full(len(cols), vmin))
        pts = [(float(a + ox), float(b + oy)) for a, b in zip(sx, sy)]
        if not one_per_line:
            poly = pts + [(float(a + ox), float(b + oy)) for a, b in zip(bx[::-1], by[::-1])]
            d.polygon(poly, fill=bg)
        colors = rgb[r, cols].astype(float)
        if shade:
            grad = np.gradient(z) if len(z) > 1 else np.zeros_like(z)
            f = np.clip(0.75 + 0.25 * np.tanh(-grad / (rng / 50.0)), 0.3, 1.0)[:, None]
            colors = colors * f
        if source_lighter:
            colors = 255 - (255 - colors) * 0.6
        if fill_black and not draw_wireframe:
            colors = np.maximum(colors, 40)
        for i in range(len(pts) - 1):
            col = (0, 0, 0) if draw_wireframe and not fill_black else \
                tuple(int(c) for c in colors[i])
            d.line([pts[i], pts[i + 1]], fill=col)
    return np.asarray(canvas, dtype=np.uint8)

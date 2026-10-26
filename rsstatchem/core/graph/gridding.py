"""XYZ → matrix conversion for contour, heatmap and 3D surface plots.

Mirrors Origin's two routes: data already on a regular X×Y grid is pivoted directly (Origin's
"XYZ Gridding" with regular data); scattered data is interpolated onto a regular grid with
:func:`scipy.interpolate.griddata` (Delaunay triangulation, like Origin's random-XYZ contour).
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import griddata


def is_regular(x: np.ndarray, y: np.ndarray) -> bool:
    ux, uy = np.unique(x), np.unique(y)
    if len(ux) < 2 or len(uy) < 2 or len(ux) * len(uy) != len(x):
        return False
    pairs = np.unique(np.column_stack([x, y]), axis=0)
    return len(pairs) == len(x)


def xyz_grid(x, y, z, method: str = "auto", n: int = 80, fill_outside: bool = False,
             nx: int | None = None, ny: int | None = None):
    """Return ``(X, Y, Z, regular)`` meshes; ``Z`` may contain NaN outside the convex hull.

    ``method``: "auto" (pivot if regular, else linear), "linear", "cubic", "nearest".
    ``nx`` / ``ny`` set the number of grid columns / rows for interpolated data (default
    ``n`` each).
    """
    x, y, z = (np.asarray(a, dtype=float) for a in (x, y, z))
    if np.isfinite(z).sum() < 3:
        raise ValueError("Need at least 3 complete (X, Y, Z) points.")
    has_xy = np.isfinite(x) & np.isfinite(y)
    if method == "auto" and not np.isfinite(z[has_xy]).all() and \
            is_regular(x[has_xy], y[has_xy]):
        # a regular grid with missing Z (e.g. a matrix with empty cells): keep the holes
        x, y, z = x[has_xy], y[has_xy], z[has_xy]
        ux, uy = np.unique(x), np.unique(y)
        Z = np.full((len(uy), len(ux)), np.nan)
        Z[np.searchsorted(uy, y), np.searchsorted(ux, x)] = z
        X, Y = np.meshgrid(ux, uy)
        return X, Y, Z, True
    ok = has_xy & np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]
    if len(x) < 3:
        raise ValueError("Need at least 3 complete (X, Y, Z) points.")
    if method == "auto" and is_regular(x, y):
        ux, uy = np.unique(x), np.unique(y)
        Z = np.full((len(uy), len(ux)), np.nan)
        Z[np.searchsorted(uy, y), np.searchsorted(ux, x)] = z
        X, Y = np.meshgrid(ux, uy)
        return X, Y, Z, True
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        raise ValueError("X and Y must both vary to build a surface.")
    kind = "linear" if method == "auto" else method
    xi = np.linspace(x.min(), x.max(), max(2, int(nx or n)))
    yi = np.linspace(y.min(), y.max(), max(2, int(ny or n)))
    X, Y = np.meshgrid(xi, yi)
    try:
        Z = griddata((x, y), z, (X, Y), method=kind)
    except Exception:  # noqa: BLE001 - degenerate triangulation (collinear points)
        Z = griddata((x, y), z, (X, Y), method="nearest")
    if fill_outside and np.isnan(Z).any():
        near = griddata((x, y), z, (X, Y), method="nearest")
        Z = np.where(np.isnan(Z), near, Z)
    return X, Y, Z, False

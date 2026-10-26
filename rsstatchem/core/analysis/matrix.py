"""Matrix operations of the Graph & Analysis module (Origin's Matrix menu), without Qt.

A matrix is a 2-D array of Z values (row 0 = first row = ``y1``) with a linear X/Y
coordinate mapping ``(x1, x2, y1, y2)``. Functions return new arrays / mappings:

* :func:`evaluate` — Set Matrix Values: ``Mat = f(x, y, i, j)`` in Origin's syntax;
* :func:`transpose`, :func:`flip`, :func:`rotate` — the geometry commands (Rotate90 remaps
  each value's X/Y coordinates, see :func:`cell_coordinates`);
* :func:`expand` (bilinear interpolation) and :func:`shrink` (block averages);
* :func:`from_columns` (Worksheet > Convert to Matrix > Direct) and :func:`from_xyz`
  (XYZ Gridding, through :mod:`rsstatchem.core.graph.gridding`);
* :func:`to_xyz` — the matrix as X, Y, Z columns (for plots and Convert to Worksheet).
"""
from __future__ import annotations

import ast
import math
import re

import numpy as np

from rsstatchem.core.graph.gridding import xyz_grid

Mapping = tuple[float, float, float, float]   # x1, x2, y1, y2

GRIDDING = {"auto": "Regular / Linear (Delaunay)", "linear": "Triangulation (Linear)",
            "cubic": "Triangulation (Cubic)", "nearest": "Nearest Neighbor"}


class MatrixError(ValueError):
    pass


# ---------------------------------------------------------------------- formulas
_FUNCS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "asin": np.arcsin, "acos": np.arccos,
    "atan": np.arctan, "atan2": np.arctan2, "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "exp": np.exp, "ln": np.log, "log": np.log10, "log10": np.log10, "sqrt": np.sqrt,
    "abs": np.abs, "int": np.trunc, "round": np.round, "mod": np.mod, "sign": np.sign,
    "min": np.minimum, "max": np.maximum, "hypot": np.hypot,
}
_NAMES = {"x", "y", "i", "j", "pi", "e"}
_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Constant,
          ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd,
          ast.FloorDiv, ast.Compare, ast.Gt, ast.Lt, ast.GtE, ast.LtE, ast.Eq, ast.NotEq,
          ast.IfExp)


def evaluate(formula: str, shape: tuple[int, int], mapping: Mapping) -> np.ndarray:
    """Origin's Set Matrix Values: ``formula`` may use ``x`` and ``y`` (the mapped
    coordinates), ``i`` (row, 1-based), ``j`` (column, 1-based), ``pi``, ``^`` for powers and
    the usual functions (sin, cos, exp, ln, log, sqrt, abs, …)."""
    text = re.sub(r"(?i)^\s*mat(\(\s*\d*\s*\))?\s*=", "", formula or "").strip()
    text = text.lstrip("=").strip()
    if not text:
        raise MatrixError("The formula is empty.")
    text = text.replace("^", "**")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        raise MatrixError(f"Syntax error in “{formula}”.") from None
    for node in ast.walk(tree):
        if not isinstance(node, _NODES):
            raise MatrixError(f"“{formula}” uses something a matrix formula cannot contain.")
        if isinstance(node, ast.Name) and node.id not in _NAMES and node.id not in _FUNCS:
            raise MatrixError(f"Unknown name “{node.id}” (use x, y, i, j and functions such "
                              "as sin, exp, sqrt).")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and
                                               node.func.id in _FUNCS):
            raise MatrixError("Only functions such as sin(), exp(), sqrt() can be called.")
    rows, cols = shape
    x1, x2, y1, y2 = mapping
    jj, ii = np.meshgrid(np.arange(1, cols + 1, dtype=float),
                         np.arange(1, rows + 1, dtype=float))
    xs = np.linspace(x1, x2, cols) if cols > 1 else np.array([x1], float)
    ys = np.linspace(y1, y2, rows) if rows > 1 else np.array([y1], float)
    xx, yy = np.meshgrid(xs, ys)
    env = {"x": xx, "y": yy, "i": ii, "j": jj, "pi": math.pi, "e": math.e, **_FUNCS}
    code = compile(tree, "<matrix formula>", "eval")
    with np.errstate(all="ignore"):
        try:
            value = eval(code, {"__builtins__": {}}, env)  # noqa: S307 - AST-checked above
        except (ZeroDivisionError, TypeError, ValueError) as exc:
            raise MatrixError(str(exc)) from None
    out = np.broadcast_to(np.asarray(value, dtype=float), (rows, cols)).astype(float)
    out[~np.isfinite(out)] = np.nan
    return out


# ---------------------------------------------------------------------- geometry
def transpose(data: np.ndarray, mapping: Mapping) -> tuple[np.ndarray, Mapping]:
    x1, x2, y1, y2 = mapping
    return np.asarray(data).T.copy(), (y1, y2, x1, x2)


def flip(data: np.ndarray, mapping: Mapping, direction: str = "horizontal") \
        -> tuple[np.ndarray, Mapping]:
    """Origin's Flip: horizontal mirrors the columns (left↔right), vertical the rows. The
    coordinate mapping stays, so the values move to new X / Y positions."""
    data = np.asarray(data)
    return (data[:, ::-1] if direction == "horizontal" else data[::-1, :]).copy(), mapping


def rotate(data: np.ndarray, mapping: Mapping, angle: int = 90) -> tuple[np.ndarray, Mapping]:
    """Origin's Rotate90: 90 (counter-clockwise, ``ccw90``), 270 / -90 (clockwise, ``cw90``)
    or 180 degrees.

    The grid turns as it is shown in the matrix sheet (row 1 at the top, so cell (i, j) of a
    counter-clockwise turn comes from (j, n_cols-1-i)), and the X/Y mapping is remapped so
    that every value keeps a rotated coordinate: a value at (x, y) moves to the point
    obtained by turning (x, y) by the same angle about the centre of the matrix
    ((x1+x2)/2, (y1+y2)/2) in the sheet's frame. The new X range has the length of the old
    Y range and vice versa, both centred on the same point, so the rotated matrix covers
    the rotated rectangle — not merely the old ranges swapped (:func:`cell_coordinates`
    gives the coordinates of every cell)."""
    angle = int(angle) % 360
    if angle not in (0, 90, 180, 270):
        raise MatrixError("Rotation must be a multiple of 90 degrees.")
    x1, x2, y1, y2 = (float(v) for v in mapping)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    out = np.rot90(np.asarray(data), k=angle // 90)
    if angle == 90:     # (x, y) -> (cx + (y - cy), cy - (x - cx))
        new_map = (cx + y1 - cy, cx + y2 - cy, cy + cx - x2, cy + cx - x1)
    elif angle == 270:  # (x, y) -> (cx - (y - cy), cy + (x - cx))
        new_map = (cx + cy - y2, cx + cy - y1, cy - cx + x1, cy - cx + x2)
    else:               # 180: (x, y) -> (2cx - x, 2cy - y) keeps the ranges
        new_map = (x1, x2, y1, y2)
    return out.copy(), tuple(_tidy(v) for v in new_map)


def _tidy(v: float) -> float:
    """Round away binary noise (2.9999999999999996) left by the centre arithmetic."""
    r = round(float(v), 12)
    return 0.0 if r == 0 else r


def cell_coordinates(shape: tuple[int, int], mapping: Mapping) \
        -> tuple[np.ndarray, np.ndarray]:
    """(X, Y) coordinate grids of every cell of a matrix of ``shape`` with ``mapping``."""
    rows, cols = shape
    x1, x2, y1, y2 = mapping
    xs = np.linspace(x1, x2, cols) if cols > 1 else np.array([x1], float)
    ys = np.linspace(y1, y2, rows) if rows > 1 else np.array([y1], float)
    return tuple(np.meshgrid(xs, ys))


def expand(data: np.ndarray, col_factor: int = 2, row_factor: int = 2) -> np.ndarray:
    """Origin's Expand: (n-1)*factor+1 points per dimension by bilinear interpolation, so the
    original values stay on the new grid and the X/Y mapping is unchanged."""
    data = np.asarray(data, float)
    cf, rf = max(1, int(col_factor)), max(1, int(row_factor))
    rows, cols = data.shape
    new_r = (rows - 1) * rf + 1 if rows > 1 else 1
    new_c = (cols - 1) * cf + 1 if cols > 1 else 1
    r = np.linspace(0, rows - 1, new_r)
    c = np.linspace(0, cols - 1, new_c)
    r0 = np.clip(np.floor(r).astype(int), 0, max(rows - 2, 0))
    c0 = np.clip(np.floor(c).astype(int), 0, max(cols - 2, 0))
    r1, c1 = np.minimum(r0 + 1, rows - 1), np.minimum(c0 + 1, cols - 1)
    fr, fc = (r - r0)[:, None], (c - c0)[None, :]
    a = data[np.ix_(r0, c0)]
    b = data[np.ix_(r0, c1)]
    cc = data[np.ix_(r1, c0)]
    d = data[np.ix_(r1, c1)]
    return a * (1 - fr) * (1 - fc) + b * (1 - fr) * fc + cc * fr * (1 - fc) + d * fr * fc


def shrink(data: np.ndarray, col_factor: int = 2, row_factor: int = 2) -> np.ndarray:
    """Origin's Shrink: average non-overlapping blocks of ``row_factor × col_factor`` cells
    (a partial last block is averaged over the cells it has)."""
    data = np.asarray(data, float)
    cf, rf = max(1, int(col_factor)), max(1, int(row_factor))
    rows, cols = data.shape
    out = np.full((math.ceil(rows / rf), math.ceil(cols / cf)), np.nan)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                out[i, j] = np.nanmean(data[i * rf:(i + 1) * rf, j * cf:(j + 1) * cf])
    return out


def shrink_mapping(mapping: Mapping, shape: tuple[int, int], new_shape: tuple[int, int],
                   col_factor: int, row_factor: int) -> Mapping:
    """X/Y mapping of a shrunk matrix: each block sits at the mean of its cells."""
    x1, x2, y1, y2 = mapping
    rows, cols = shape

    def centers(lo, hi, n, f, m):
        pos = np.linspace(lo, hi, n) if n > 1 else np.array([lo], float)
        cs = [pos[k * f:(k + 1) * f].mean() for k in range(m)]
        return float(cs[0]), float(cs[-1])
    nx1, nx2 = centers(x1, x2, cols, max(1, col_factor), new_shape[1])
    ny1, ny2 = centers(y1, y2, rows, max(1, row_factor), new_shape[0])
    return nx1, nx2, ny1, ny2


# ---------------------------------------------------------------------- conversions
def from_columns(columns: list[np.ndarray], y_values: np.ndarray | None = None,
                 x_values: list[float] | None = None) -> tuple[np.ndarray, Mapping]:
    """Worksheet > Convert to Matrix > Direct: worksheet rows become matrix rows and the
    selected columns matrix columns. ``y_values`` / ``x_values`` (evenly spaced) set the
    mapping, else rows and columns are numbered from 1."""
    if not columns:
        raise MatrixError("Select the columns to convert.")
    n = max(len(c) for c in columns)
    data = np.full((n, len(columns)), np.nan)
    for j, c in enumerate(columns):
        data[:len(c), j] = np.asarray(c, float)
    keep = np.isfinite(data).any(axis=1)
    if keep.any():
        last = int(np.flatnonzero(keep)[-1]) + 1
        data = data[:last]
        if y_values is not None:
            y_values = np.asarray(y_values, float)[:last]
    rows, cols = data.shape
    x1, x2 = (1.0, float(cols))
    if x_values is not None and len(x_values) == cols and np.all(np.isfinite(x_values)):
        x1, x2 = float(x_values[0]), float(x_values[-1])
    y1, y2 = 1.0, float(rows)
    if y_values is not None:
        yv = np.asarray(y_values, float)
        ok = np.isfinite(yv)
        if ok.sum() >= 1:
            y1, y2 = float(yv[ok][0]), float(yv[ok][-1])
    return data, (x1, x2, y1, y2)


def from_xyz(x, y, z, cols: int = 20, rows: int = 20, method: str = "auto") \
        -> tuple[np.ndarray, Mapping, bool]:
    """XYZ Gridding: regular XYZ data are pivoted directly; scattered data are interpolated
    on a ``rows × cols`` grid. Returns ``(data, mapping, regular)``."""
    try:
        X, Y, Z, regular = xyz_grid(x, y, z, method, nx=cols, ny=rows)
    except ValueError as exc:
        raise MatrixError(str(exc)) from None
    return Z, (float(X[0, 0]), float(X[0, -1]), float(Y[0, 0]), float(Y[-1, 0])), regular


def to_xyz(data: np.ndarray, mapping: Mapping, drop_missing: bool = False) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The matrix as X, Y, Z columns (row by row: Y varies slowest)."""
    data = np.asarray(data, float)
    rows, cols = data.shape
    x1, x2, y1, y2 = mapping
    xs = np.linspace(x1, x2, cols) if cols > 1 else np.array([x1], float)
    ys = np.linspace(y1, y2, rows) if rows > 1 else np.array([y1], float)
    xx, yy = np.meshgrid(xs, ys)
    x, y, z = xx.ravel(), yy.ravel(), data.ravel()
    if drop_missing:
        ok = np.isfinite(z)
        x, y, z = x[ok], y[ok], z[ok]
    return x, y, z


def number_labels(text_values: list[str]) -> list[float] | None:
    """Numbers from column Long Names (e.g. "10", "20"…) used as X values, or None."""
    out = []
    for t in text_values:
        m = re.fullmatch(r"\s*[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?\s*", t or "")
        if not m:
            return None
        out.append(float(t))
    return out

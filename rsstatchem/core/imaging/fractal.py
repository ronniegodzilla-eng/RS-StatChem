"""Analyze > Tools > Fractal Box Count..., Save XY Coordinates... and Analyze Line Graph.

Box counting: for each box size the number of boxes containing foreground (binary image,
foreground per "black background"), and the fractal dimension D = -slope of
log(count) vs log(size). ImageJ's default sizes are 2,3,4,6,8,12,16,32,64.

Written after ImageJ 1.54 ij/plugin/filter/FractalBoxCounter.java, ij/plugin/XYCoordinates.java
and LineGraphAnalyzer.java (public domain, NIH).
"""
from __future__ import annotations

import numpy as np

from rsstatchem.core.imaging.results import ResultsTable


def foreground(image, black_background: bool = False) -> np.ndarray:
    p = image.plane()
    if p.ndim == 3:
        p = p.mean(axis=2)
    if image.threshold is not None:
        with np.errstate(invalid="ignore"):
            return (p >= image.threshold[0]) & (p <= image.threshold[1])
    if image.is_inverted_lut():
        return p == 255 if not black_background else p == 0
    return p == 0 if not black_background else p == 255


def box_count(mask: np.ndarray, sizes=(2, 3, 4, 6, 8, 12, 16, 32, 64)):
    """[(size, count)] of boxes containing at least one foreground pixel."""
    h, w = mask.shape
    out = []
    ys, xs = np.nonzero(mask)
    for s in sizes:
        s = int(s)
        if s <= 0:
            continue
        cells = np.unique((ys // s) * ((w + s - 1) // s) + xs // s)
        out.append((s, int(len(cells))))
    return out


def fractal_dimension(counts) -> tuple[float, float]:
    """(D, intercept) from the least-squares fit of log(count) on log(size)."""
    s = np.array([c[0] for c in counts if c[1] > 0], float)
    n = np.array([c[1] for c in counts if c[1] > 0], float)
    if len(s) < 2:
        return float("nan"), float("nan")
    slope, icpt = np.polyfit(np.log(s), np.log(n), 1)
    return float(-slope), float(icpt)


def box_count_table(image, sizes, black_background=False) -> tuple[ResultsTable, float]:
    counts = box_count(foreground(image, black_background), sizes)
    rt = ResultsTable("Results")
    for s, c in counts:
        rt.add_row({"size": s, "count": c})
    d, _ = fractal_dimension(counts)
    return rt, d


def xy_coordinates(image, background: float | None = None, invert_y: bool = False):
    """``XYCoordinates``: rows (x, y, value) of every pixel != background (default: the value
    of the top-left corner pixel, as ImageJ assumes)."""
    p = image.plane()
    if p.ndim == 3:
        packed = (p[..., 0].astype(np.int64) << 16) | (p[..., 1].astype(np.int64) << 8) | \
            p[..., 2].astype(np.int64)
        v = packed
    else:
        v = p
    if background is None:
        background = float(v[0, 0])
    ys, xs = np.nonzero(v != background)
    vals = v[ys, xs]
    if invert_y:
        ys = image.height - 1 - ys
    return xs, ys, vals


def save_xy_coordinates(path, image, background=None, invert_y=False) -> int:
    xs, ys, vals = xy_coordinates(image, background, invert_y)
    with open(path, "w", encoding="utf-8") as fh:
        for x, y, v in zip(xs, ys, vals):
            fh.write(f"{x}\t{y}\t{v:g}\n" if isinstance(v, float) else f"{x}\t{y}\t{v}\n")
    return len(xs)


def analyze_line_graph(image, black_background: bool = False) -> ResultsTable:
    """Recover (X, Y) of a line graph from a binary image: for each column the mean row of
    the foreground pixels, y measured upwards in calibrated units."""
    m = foreground(image, black_background)
    cal = image.calibration
    rt = ResultsTable("Results")
    for x in range(m.shape[1]):
        ys = np.nonzero(m[:, x])[0]
        if len(ys):
            y = m.shape[0] - 1 - ys.mean()
            rt.add_row({"X": cal.get_x(x), "Y": y * cal.pixel_height})
    return rt

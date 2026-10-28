"""Line and rectangle profiles (Analyze > Plot Profile) and the straightening of wide lines.

* straight lines: ``ImageProcessor.getLine`` (n = round(length) steps, bilinear
  ``getInterpolatedValue`` with ImageJ's edge handling, or nearest pixel);
* segmented/freehand lines: ``ProfilePlot.getIrregularProfile`` (0.01-pixel stepping, one value
  per pixel of path length);
* wide lines: ``Straightener`` (perpendicular sampling along the 1-pixel resampled path) and
  the column average;
* rectangles: column average (``getColumnAverageProfile``) or row average (vertical profile).

Values are calibrated (density calibration table) unless ``calibrated=False``.
Ported from ImageJ 1.54 ij/gui/ProfilePlot.java, ij/process/ImageProcessor.java (getLine,
getInterpolatedValue) and ij/plugin/Straightener.java (public domain, NIH).
"""
from __future__ import annotations

import math

import numpy as np

from rsstatchem.core.imaging.calibration import NONE
from rsstatchem.core.imaging.roi import shapes as S
from rsstatchem.core.imaging.roi import geometry as G


def value_plane(image, calibrated: bool = True, plane: np.ndarray | None = None,
                weighted_rgb: bool = False) -> np.ndarray:
    """``getPixelValue`` for the whole plane as float64 (calibrated, RGB as gray)."""
    p = image.plane() if plane is None else plane
    if p.ndim == 3:
        w = (0.299, 0.587, 0.114) if weighted_rgb else (1 / 3, 1 / 3, 1 / 3)
        return (p[..., 0] * w[0] + p[..., 1] * w[1] + p[..., 2] * w[2]).astype(np.float32) \
            .astype(np.float64)
    cal = image.calibration
    if calibrated and cal.function != NONE and image.bit_depth in (8, 16):
        table = np.asarray(cal.ctable(image.bit_depth), dtype=np.float32)
        return table[p.astype(np.int64)].astype(np.float64)
    return p.astype(np.float64)


def interpolated_values(vp: np.ndarray, x, y) -> np.ndarray:
    """``ImageProcessor.getInterpolatedValue`` (bilinear; edge pixels repeated within one
    pixel outside the image, 0 further out) for arrays of coordinates."""
    h, w = vp.shape
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    out = np.zeros(np.broadcast(x, y).shape)
    ok = (x >= -1.0) & (x < w) & (y >= -1.0) & (y < h)
    xb = np.trunc(x).astype(np.int64)
    yb = np.trunc(y).astype(np.int64)
    xf = np.maximum(x - xb, 0.0)
    yf = np.maximum(y - yb, 0.0)
    x0 = np.clip(xb, 0, w - 1)
    x1 = np.clip(xb + 1, 0, w - 1)
    y0 = np.clip(yb, 0, h - 1)
    y1 = np.clip(yb + 1, 0, h - 1)
    ll = vp[y0, x0]
    lr = vp[y0, x1]
    ur = vp[y1, x1]
    ul = vp[y1, x0]
    upper = ul + xf * (ur - ul)
    lower = ll + xf * (lr - ll)
    val = lower + yf * (upper - lower)
    out[ok] = val[ok]
    return out


def pixel_values(vp: np.ndarray, x, y) -> np.ndarray:
    """``getPixelValue((int)Math.round(x), (int)Math.round(y))`` (NaN outside)."""
    h, w = vp.shape
    xi = np.floor(np.asarray(x, dtype=float) + 0.5).astype(np.int64)
    yi = np.floor(np.asarray(y, dtype=float) + 0.5).astype(np.int64)
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    out = np.full(xi.shape, np.nan)
    out[ok] = vp[yi[ok], xi[ok]]
    return out


def get_line(vp: np.ndarray, x1, y1, x2, y2, interpolate: bool = True) -> np.ndarray:
    """``ImageProcessor.getLine``: round(length)+1 samples from (x1, y1) to (x2, y2)."""
    dx, dy = x2 - x1, y2 - y1
    n = int(math.floor(math.sqrt(dx * dx + dy * dy) + 0.5))
    xinc = dx / n if n > 0 else 0.0
    yinc = dy / n if n > 0 else 0.0
    rx = np.add.accumulate(np.concatenate([[x1], np.full(n, xinc)]))
    ry = np.add.accumulate(np.concatenate([[y1], np.full(n, yinc)]))
    if interpolate:
        return interpolated_values(vp, rx, ry)
    return pixel_values(vp, rx, ry)


def irregular_profile(vp: np.ndarray, xs, ys, interpolate: bool = True) -> np.ndarray:
    """``ProfilePlot.getIrregularProfile`` for segmented and freehand lines."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n = len(xs)
    inc = 0.01
    values = []
    lastx = lasty = 0.0
    x2, y2 = xs[0], ys[0]
    px_list, py_list = [], []
    for i in range(1, n):
        x1, y1 = x2, y2
        x, y = x1, y1
        x2, y2 = xs[i], ys[i]
        dx, dy = x2 - x1, y2 - y1
        distance = math.sqrt(dx * dx + dy * dy)
        if distance == 0:
            continue
        xinc = dx * inc / distance
        yinc = dy * inc / distance
        n2 = int(distance / inc)
        if n == 2:
            n2 += 1
        while True:
            ddx, ddy = x - lastx, y - lasty
            if math.sqrt(ddx * ddx + ddy * ddy) >= 1.0 - inc / 2.0:
                px_list.append(x)
                py_list.append(y)
                lastx, lasty = x, y
            x += xinc
            y += yinc
            n2 -= 1
            if n2 <= 0:
                break
    if not px_list:
        return np.zeros(0)
    if interpolate:
        values = interpolated_values(vp, px_list, py_list)
    else:
        values = pixel_values(vp, px_list, py_list)
    return np.asarray(values, dtype=float)


def straighten(vp: np.ndarray, xs, ys, width: int, spline: bool = True) -> np.ndarray:
    """``Straightener.straightenLine``: (width, n) samples perpendicular to the path, which
    is resampled at 1-pixel steps (spline fitted for segmented lines)."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) > 2 and spline:
        sx, sy = G.spline_fit(xs, ys, closed=False)
        px, py = G.interpolated_polygon(sx, sy, 1.0, closed=False)
    else:
        px, py = G.interpolated_polygon(xs, ys, 1.0, closed=False)
        if len(px) and (px[-1] != xs[-1] or py[-1] != ys[-1]):
            if math.hypot(px[-1] - xs[-1], py[-1] - ys[-1]) > 0.5:
                px = np.append(px, xs[-1])
                py = np.append(py, ys[-1])
    n = len(px)
    if n < 2:
        return np.zeros((max(width, 1), n))
    out = np.zeros((max(width, 1), n))
    x2 = px[0] - (px[1] - px[0])
    y2 = py[0] - (py[1] - py[0])
    for i in range(n):
        x1, y1 = x2, y2
        x2, y2 = px[i], py[i]
        if width <= 1:
            out[0, i] = interpolated_values(vp, x2, y2)
            continue
        dx = x2 - x1
        dy = y1 - y2
        length = float(np.float32(math.sqrt(dx * dx + dy * dy))) or 1.0
        dx /= length
        dy /= length
        x = x2 - dy * (width - 1) / 2.0
        y = y2 - dx * (width - 1) / 2.0
        jj = np.arange(width)
        out[:, i] = interpolated_values(vp, x + dy * jj, y + dx * jj)
    return out


def line_profile(image, roi=None, interpolate: bool = True, calibrated: bool = True,
                 vertical: bool = False, line_width: int | None = None,
                 weighted_rgb: bool = False) -> np.ndarray:
    """Profile of a line or rectangle selection of ``image`` (ImageJ ``ProfilePlot``)."""
    roi = image.roi if roi is None else roi
    if roi is None:
        raise ValueError("Line or rectangular selection required")
    vp = value_plane(image, calibrated, weighted_rgb=weighted_rgb)
    if isinstance(roi, S.RotatedRectRoi):
        roi = S.LineRoi(roi.x1, roi.y1, roi.x2, roi.y2, width=roi.rect_width)
    if line_width is None:
        line_width = int(round(max(getattr(roi, "width", 1.0) if isinstance(roi, S.LineRoi)
                                   else 1.0, roi.stroke_width or 0.0, 1.0)))
    if isinstance(roi, S.LineRoi):
        x1, y1 = float(np.float32(roi.x1)), float(np.float32(roi.y1))
        x2, y2 = float(np.float32(roi.x2)), float(np.float32(roi.y2))
        if line_width <= 1:
            return get_line(vp, roi.x1, roi.y1, roi.x2, roi.y2, interpolate)
        s = straighten(vp, [x1, x2], [y1, y2], line_width, spline=False)
        return s.mean(axis=0)
    if roi.type in (S.POLYLINE, S.FREELINE):
        xs, ys = G.float_polygon(roi)
        if line_width <= 1:
            return irregular_profile(vp, xs, ys, interpolate)
        return straighten(vp, xs, ys, line_width).mean(axis=0)
    if isinstance(roi, S.RectRoi):
        x, y, w, h = G.int_bounds(roi)
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, image.width), min(y + h, image.height)
        block = np.full((h, w), np.nan)
        block[y0 - y:y1 - y, x0 - x:x1 - x] = vp[y0:y1, x0:x1]
        with np.errstate(invalid="ignore"):
            if vertical:
                return np.nanmean(block, axis=1)
            return np.nanmean(block, axis=0)
    raise ValueError("Line or rectangular selection required")


def x_values(image, roi, n: int, vertical: bool = False) -> np.ndarray:
    """Distances for the profile samples in calibrated units (ProfilePlot.xInc)."""
    cal = image.calibration
    inc = cal.pixel_height if vertical else cal.pixel_width
    if isinstance(roi, S.LineRoi) and cal.pixel_width != cal.pixel_height:
        dx, dy = roi.x2 - roi.x1, roi.y2 - roi.y1
        pl = math.hypot(dx, dy)
        if pl > 0:
            inc = math.hypot(dx * cal.pixel_width, dy * cal.pixel_height) / pl
    return (np.arange(n) * inc).astype(np.float32).astype(np.float64)


# ---------------------------------------------------------------------------------- plot data
class PlotData:
    """Data of an ImageJ ``Plot`` window (profile plots, lane plots, fits, distributions):
    series of (x, y) with a style ("line", "connected circle", "circle", "bar",
    "separated bar"), labels, optional fixed ranges and log axes. ``source`` describes how a
    Live plot is recomputed (e.g. {"kind": "profile", "image_id": ...})."""

    def __init__(self, title: str, x_label: str = "", y_label: str = "",
                 source: dict | None = None):
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.series: list[dict] = []
        self.x_range: tuple | None = None
        self.y_range: tuple | None = None
        self.log_x = False
        self.log_y = False
        self.source = source or {}
        self.legend = False
        self.labels: list[tuple[float, float, str]] = []   # text annotations (x, y, text)

    def add(self, x, y, label: str = "", style: str = "line", color: str = "#000000"):
        self.series.append({"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float),
                            "label": label, "style": style, "color": color})
        return self

    def limits(self) -> tuple[float, float, float, float]:
        xs = [s["x"] for s in self.series if len(s["x"])]
        ys = [s["y"] for s in self.series if len(s["y"])]
        if not xs:
            return 0.0, 1.0, 0.0, 1.0
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]
        xmin, xmax = (float(x.min()), float(x.max())) if x.size else (0.0, 1.0)
        ymin, ymax = (float(y.min()), float(y.max())) if y.size else (0.0, 1.0)
        if any(s["style"] in ("bar", "separated bar") for s in self.series):
            ymin = min(ymin, 0.0)
        if self.x_range is not None:
            xmin, xmax = self.x_range
        if self.y_range is not None:
            ymin, ymax = self.y_range
        if xmax == xmin:
            xmax = xmin + 1.0
        if ymax == ymin:
            ymax = ymin + 1.0
        return xmin, xmax, ymin, ymax

    def table(self, title: str | None = None):
        """The plot's "List" table (X, Y columns per series as ImageJ)."""
        from rsstatchem.core.imaging.results import ResultsTable

        rt = ResultsTable(title or "Plot Values")
        n = max((len(s["x"]) for s in self.series), default=0)
        for i in range(n):
            row = {}
            for k, s in enumerate(self.series):
                sx = "X" if k == 0 else f"X{k}"
                sy = "Y" if k == 0 else f"Y{k}"
                row[sx] = float(s["x"][i]) if i < len(s["x"]) else float("nan")
                row[sy] = float(s["y"][i]) if i < len(s["y"]) else float("nan")
            rt.add_row(row)
        return rt


def profile_plot(image, roi=None, vertical: bool = False, interpolate: bool = True) -> PlotData:
    """Analyze > Plot Profile data for ``image`` (ImageJ's titles and axis labels)."""
    roi = image.roi if roi is None else roi
    values = line_profile(image, roi, interpolate=interpolate, vertical=vertical)
    xv = x_values(image, roi, len(values), vertical)
    cal = image.calibration
    title = image.title
    i = title.rfind(".")
    if i > 0 and len(title) - i <= 5:
        title = title[:i]
    plot = PlotData(f"Plot of {title}", f"Distance ({cal.units})", cal.value_unit,
                    {"kind": "profile", "image_id": image.id, "vertical": vertical})
    plot.add(xv, np.asarray(values).astype(np.float32).astype(float))
    return plot

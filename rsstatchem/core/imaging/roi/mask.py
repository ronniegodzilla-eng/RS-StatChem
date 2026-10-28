"""ROI rasterisation: ``rasterize(roi, shape) -> bool (H, W)``. **Owned by WP-C.**

Exact ports of ImageJ's area masks:

* rectangles: ImageJ's integer bounds ``(int)x, (int)y, ceil(w), ceil(h)``
  (``Roi(double, double, double, double)``); rounded rectangles via their flattened outline;
* ovals: ``OvalRoi.getMask`` (ellipse test at pixel centres of the integer bounds, with ImageJ's
  ``a - 0.5`` offsets);
* polygons, freehand, traced, rotated rectangles, ellipses: ``PolygonFiller`` (scan-line fill
  with ImageJ's half-pixel convention: a point exactly on a left border is outside, on a right
  border inside; float vertices are rounded per edge as in ``setPolygon(float[], ...)``;
  several polygons separated by NaN are filled with the even-odd rule);
* composite ROIs: the parts combined in order with or/and/xor/not (ImageJ's ShapeRoi works on
  java.awt.geom.Area and fills the flattened result with PolygonFiller, which gives the same
  pixels for the usual shapes).

Ported from ImageJ 1.54 ij/process/PolygonFiller.java and ij/gui/OvalRoi.java (public domain,
NIH; W. Rasband, M. Schmid et al.).
"""
from __future__ import annotations

import math

import numpy as np

from rsstatchem.core.imaging.roi import shapes as S


def rect_bounds(x: float, y: float, w: float, h: float) -> tuple[int, int, int, int]:
    """ImageJ's integer bounds of a rectangle/oval created from doubles."""
    return int(x), int(y), int(math.ceil(w)), int(math.ceil(h))


def rasterize(roi, shape: tuple[int, int]) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    mask = np.zeros((h, w), dtype=bool)
    if isinstance(roi, S.CompositeRoi):
        for i, (op, part) in enumerate(roi.parts):
            m = rasterize(part, shape)
            if i == 0 or op == "or":
                mask |= m
            elif op == "and":
                mask &= m
            elif op == "xor":
                mask ^= m
            elif op == "not":
                mask &= ~m
        return mask
    if isinstance(roi, S.TextRoi) or not roi.is_area():
        return mask
    if isinstance(roi, S.RectRoi):
        x0, y0, rw, rh = rect_bounds(roi.x, roi.y, roi.width, roi.height)
        if roi.corner_diameter and roi.corner_diameter > 0:
            from rsstatchem.core.imaging.roi.geometry import rounded_rect_polygon
            xs, ys = rounded_rect_polygon(roi.x, roi.y, roi.width, roi.height,
                                          roi.corner_diameter, flatness=0.01)
            return polygon_mask(xs, ys, shape)
        mask[max(y0, 0):max(min(y0 + rh, h), 0), max(x0, 0):max(min(x0 + rw, w), 0)] = True
        return mask
    if isinstance(roi, S.OvalRoi):
        x0, y0, ow, oh = rect_bounds(roi.x, roi.y, roi.width, roi.height)
        local = oval_mask(ow, oh)
        return paste(mask, local, x0, y0)
    xs, ys = roi.polygon()
    return polygon_mask(xs, ys, shape)


def paste(mask: np.ndarray, local: np.ndarray, x0: int, y0: int) -> np.ndarray:
    """OR a local mask whose top-left pixel is (x0, y0) into ``mask`` (clipped)."""
    h, w = mask.shape
    lh, lw = local.shape
    xa, ya = max(x0, 0), max(y0, 0)
    xb, yb = min(x0 + lw, w), min(y0 + lh, h)
    if xb > xa and yb > ya:
        mask[ya:yb, xa:xb] |= local[ya - y0:yb - y0, xa - x0:xb - x0]
    return mask


def oval_mask(width: int, height: int) -> np.ndarray:
    """``OvalRoi.getMask()`` of an oval with integer bounds (local coordinates)."""
    if width <= 0 or height <= 0:
        return np.zeros((max(height, 0), max(width, 0)), dtype=bool)
    a, b = width / 2.0, height / 2.0
    a2, b2 = a * a, b * b
    a -= 0.5
    b -= 0.5
    yy, xx = np.mgrid[0:height, 0:width].astype(float)
    xx -= a
    yy -= b
    return (xx * xx / a2 + yy * yy / b2) <= 1.0


def polygon_mask(xs, ys, shape: tuple[int, int], x_offset: float = 0.0,
                 y_offset: float = 0.0) -> np.ndarray:
    """ImageJ ``PolygonFiller`` fill of one polygon, or several separated by NaN, into an
    image of ``shape`` (vertex coordinates in image space plus the offsets)."""
    h, w = int(shape[0]), int(shape[1])
    mask = np.zeros((h, w), dtype=bool)
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n = len(xs)
    if n < 3 or h == 0 or w == 0:
        return mask
    # integer polygons use the int path of PolygonFiller (identical result for integers)
    ex, ey1, ey2, slope = [], [], [], []
    poly_start = 0
    for i in range(n):
        ip1 = poly_start if i == n - 1 else i + 1
        if math.isnan(xs[ip1]):
            ip1 = poly_start
        if math.isnan(xs[i]):
            poly_start = i + 1
            continue
        y1f, y2f = ys[i] + y_offset, ys[ip1] + y_offset
        x1f, x2f = xs[i] + x_offset, xs[ip1] + x_offset
        y1, y2 = _jround(y1f), _jround(y2f)
        if y1 == y2 or (y1 <= 0 and y2 <= 0):
            continue
        if y1 > y2:
            y1, y2 = y2, y1
            y1f, y2f = y2f, y1f
            x1f, x2f = x2f, x1f
        s = (x2f - x1f) / (y2f - y1f)
        ex.append(x1f + (y1 - y1f + 0.5) * s + 1e-8)
        ey1.append(y1)
        ey2.append(y2)
        slope.append(s)
    if not ex:
        return mask
    ex = np.asarray(ex)
    ey1 = np.asarray(ey1, dtype=np.int64)
    ey2 = np.asarray(ey2, dtype=np.int64)
    slope = np.asarray(slope)
    first = np.maximum(ey1, 0)
    last = np.minimum(ey2, h)          # exclusive
    counts = np.maximum(last - first, 0)
    keep = counts > 0
    if not keep.any():
        return mask
    ex, ey1, slope, first, counts = ex[keep], ey1[keep], slope[keep], first[keep], counts[keep]
    edge = np.repeat(np.arange(len(ex)), counts)
    starts = np.cumsum(counts) - counts
    k = np.arange(counts.sum()) - np.repeat(starts, counts)
    rows = first[edge] + k
    xv = ex[edge] + slope[edge] * (rows - ey1[edge])
    order = np.lexsort((xv, rows))
    rows, xv = rows[order], xv[order]
    xi = np.floor(xv + 0.5).astype(np.int64)
    np.clip(xi, 0, w, out=xi)
    r1, x1 = rows[0::2], xi[0::2]
    x2 = xi[1::2]
    diff = np.zeros((h, w + 1), dtype=np.int32)
    good = x2 > x1
    np.add.at(diff, (r1[good], x1[good]), 1)
    np.add.at(diff, (r1[good], x2[good]), -1)
    return np.cumsum(diff[:, :w], axis=1) > 0


def _jround(v: float) -> int:
    """Java ``Math.round`` (floor(v + 0.5))."""
    return int(math.floor(v + 0.5))

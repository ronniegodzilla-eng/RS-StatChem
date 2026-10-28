"""ROI rasterisation: ``rasterize(roi, shape) -> bool (H, W)``. **Owned by WP-C after Phase 0.**

Phase 0 stub with the contract semantics (see :mod:`.shapes`): rectangles cover their integer
pixel block (non-integer bounds are rounded like ImageJ's ``Roi.getBounds``), polygons and
freehand/traced ROIs include a pixel whose centre is inside (even-odd rule, as ImageJ's
PolygonFiller), ovals use the ellipse equation at pixel centres. WP-C replaces the oval and
polygon code with exact ports of ImageJ's OvalRoi.getPolygon/getMask and PolygonFiller behind
this same function; lines, points and text have no area and give an empty mask.
"""
from __future__ import annotations

import numpy as np

from rsstatchem.core.imaging.roi import shapes as S


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
    if isinstance(roi, S.RectRoi) and not roi.corner_diameter:
        x0, y0 = int(round(roi.x)), int(round(roi.y))
        x1, y1 = x0 + int(round(roi.width)), y0 + int(round(roi.height))
        mask[max(y0, 0):max(min(y1, h), 0), max(x0, 0):max(min(x1, w), 0)] = True
        return mask
    if isinstance(roi, S.OvalRoi):
        if roi.width <= 0 or roi.height <= 0:
            return mask
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = roi.x + roi.width / 2.0, roi.y + roi.height / 2.0
        rx, ry = roi.width / 2.0, roi.height / 2.0
        return ((xx + 0.5 - cx) / rx) ** 2 + ((yy + 0.5 - cy) / ry) ** 2 <= 1.0
    xs, ys = roi.polygon()
    return polygon_mask(xs, ys, shape)


def polygon_mask(xs, ys, shape: tuple[int, int]) -> np.ndarray:
    """Even-odd fill of a closed polygon sampled at pixel centres."""
    h, w = int(shape[0]), int(shape[1])
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mask = np.zeros((h, w), dtype=bool)
    n = len(xs)
    if n < 3:
        return mask
    x2, y2 = np.roll(xs, -1), np.roll(ys, -1)
    for row in range(max(int(np.floor(ys.min())), 0), min(int(np.ceil(ys.max())) + 1, h)):
        yc = row + 0.5
        crosses = (ys <= yc) != (y2 <= yc)
        if not crosses.any():
            continue
        xi = xs[crosses] + (yc - ys[crosses]) * (x2[crosses] - xs[crosses]) / \
            (y2[crosses] - ys[crosses])
        xi.sort()
        for a, b in zip(xi[0::2], xi[1::2]):
            # pixel i is inside when a <= i + 0.5 < b
            i0 = max(int(np.ceil(a - 0.5)), 0)
            i1 = min(int(np.ceil(b - 0.5)), w)
            if i1 > i0:
                mask[row, i0:i1] = True
    return mask

"""ROI geometry with ImageJ semantics: integer bounds, outlines as ImageJ returns them
(``getFloatPolygon``), convex hull, Feret values, perimeters/lengths, flattening of curved
outlines (Java's ``FlatteningPathIterator``), spline fit, interpolation, circle/ellipse/
rectangle fits, rotation, scaling and boundary tracing of masks.

All coordinates are image coordinates (pixel corners at integers). Float outlines are rounded
to float32 where ImageJ stores ``float[]`` coordinates, so derived numbers (Feret, convex hull
areas) agree with ImageJ.

Ported from ImageJ 1.54 ij/gui/Roi.java, PolygonRoi.java, OvalRoi.java, EllipseRoi.java,
ShapeRoi.java, ij/process/FloatPolygon.java and ij/plugin/Selection.java (public domain, NIH;
W. Rasband, M. Schmid et al.); the Bezier flattening follows the documented behaviour of
java.awt.geom.FlatteningPathIterator / CubicCurve2D (recursive midpoint subdivision until the
control points are within ``flatness`` of the chord, at most 10 levels).
"""
from __future__ import annotations

import math

import numpy as np

from rsstatchem.core.imaging.roi import shapes as S

FLATNESS = 0.1          # ShapeRoi.FLATNESS (outline listing, convex hull)
FILL_FLATNESS = 0.01    # ShapeRoi.FILL_FLATNESS (masks)
FERET_ARRAYSIZE = 16
FERET_ARRAY_POINTOFFSET = 8


def f32(a) -> np.ndarray:
    """Round to float32 (ImageJ ``float[]`` coordinates) and return float64."""
    return np.asarray(a, dtype=np.float32).astype(np.float64)


# ------------------------------------------------------------------------------ bounds
def int_bounds(roi) -> tuple[int, int, int, int]:
    """ImageJ ``Roi.getBounds()``: integer (x, y, width, height)."""
    if isinstance(roi, (S.RectRoi, S.OvalRoi)):
        return int(roi.x), int(roi.y), int(math.ceil(roi.width)), int(math.ceil(roi.height))
    if isinstance(roi, S.TextRoi):
        x, y, w, h = roi.bounds()
        return int(x), int(y), int(math.ceil(w)), int(math.ceil(h))
    if isinstance(roi, S.CompositeRoi):
        boxes = [int_bounds(r) for _, r in roi.parts]
        if not boxes:
            return 0, 0, 0, 0
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[0] + b[2] for b in boxes)
        y1 = max(b[1] + b[3] for b in boxes)
        return x0, y0, x1 - x0, y1 - y0
    xs, ys = float_polygon(roi)
    if len(xs) == 0:
        return 0, 0, 0, 0
    bx, by = float(xs.min()), float(ys.min())
    bw, bh = float(xs.max()) - bx, float(ys.max()) - by
    if line_convention(roi):
        x = math.floor(bx + 0.5)
        y = math.floor(by + 0.5)
        return x, y, math.floor(bx + bw + 1.5) - x, math.floor(by + bh + 1.5) - y
    x, y = math.floor(bx), math.floor(by)
    return x, y, math.ceil(bx + bw) - x, math.ceil(by + bh) - y


def line_convention(roi) -> bool:
    """Open lines and points: integer coordinates are pixel centres (``setIntBounds``)."""
    return isinstance(roi, (S.LineRoi, S.PointRoi)) or roi.type in (S.POLYLINE, S.FREELINE,
                                                                       S.ANGLE, S.POINT)


# ------------------------------------------------------------------------------ outlines
def float_polygon(roi) -> tuple[np.ndarray, np.ndarray]:
    """ImageJ ``getFloatPolygon("")``: outline vertices without a duplicate closing point."""
    if isinstance(roi, S.OvalRoi):
        return ellipse_polygon(roi.x, roi.y, roi.width, roi.height, FLATNESS)
    if isinstance(roi, S.RectRoi):
        if roi.corner_diameter and roi.corner_diameter > 0:
            return rounded_rect_polygon(roi.x, roi.y, roi.width, roi.height,
                                        roi.corner_diameter, FLATNESS)
        x, y, w, h = float(roi.x), float(roi.y), float(roi.width), float(roi.height)
        return f32([x, x + w, x + w, x]), f32([y, y, y + h, y + h])
    if isinstance(roi, S.LineRoi):
        return f32([roi.x1, roi.x2]), f32([roi.y1, roi.y2])
    if isinstance(roi, S.CompositeRoi):
        xs, ys = [], []
        for _, part in roi.parts:
            px, py = float_polygon(part)
            xs.append(px)
            ys.append(py)
        if not xs:
            return np.zeros(0), np.zeros(0)
        return np.concatenate(xs), np.concatenate(ys)
    xs, ys = roi.polygon()
    xs, ys = f32(xs), f32(ys)
    if roi.is_area() and len(xs) > 1 and xs[0] == xs[-1] and ys[0] == ys[-1]:
        xs, ys = xs[:-1], ys[:-1]
    return xs, ys


def ellipse_polygon(x: float, y: float, w: float, h: float, flatness: float = FLATNESS):
    """``OvalRoi.getFloatPolygon()``: Java's Ellipse2D (4 cubic Beziers, enlarged by 0.0004
    as ImageJ does) flattened through a ShapeRoi."""
    ex, ey, ew, eh = x - 0.0004, y - 0.0004, w + 0.0008, h + 0.0008
    ctrl = 0.5522847498307933
    pcv, ncv = 0.5 + ctrl * 0.5, 0.5 - ctrl * 0.5
    pts = ((1.0, pcv, pcv, 1.0, 0.5, 1.0), (ncv, 1.0, 0.0, pcv, 0.0, 0.5),
           (0.0, ncv, ncv, 0.0, 0.5, 0.0), (pcv, 0.0, 1.0, ncv, 1.0, 0.5))
    bx, by = math.floor(ex), math.floor(ey)   # ShapeRoi(Shape): translate by -bounds
    start = (ex + 1.0 * ew - bx, ey + 0.5 * eh - by)
    segs = []
    for c in pts:
        segs.append(tuple(v for k in range(3) for v in (ex + c[2 * k] * ew - bx,
                                                        ey + c[2 * k + 1] * eh - by)))
    xs, ys = flatten_path(start, segs, flatness)
    return f32(np.float32(xs) + np.float32(bx)), f32(np.float32(ys) + np.float32(by))


def rounded_rect_polygon(x: float, y: float, w: float, h: float, diameter: float,
                         flatness: float = FLATNESS):
    """Outline of ImageJ's rounded rectangle (Java RoundRectangle2D flattened)."""
    aw, ah = min(w, abs(diameter)), min(h, abs(diameter))
    angle = math.pi / 4.0
    a = 1.0 - math.cos(angle)
    b = math.tan(angle)
    c = math.sqrt(1.0 + b * b) - 1 + a
    cv = 4.0 / 3.0 * a * b / c
    acv = (1.0 - cv) / 2.0
    table = [
        ("M", (0.0, 0.0, 0.0, 0.5)),
        ("L", (0.0, 0.0, 1.0, -0.5)),
        ("C", (0.0, 0.0, 1.0, -acv, 0.0, acv, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0)),
        ("L", (1.0, -0.5, 1.0, 0.0)),
        ("C", (1.0, -acv, 1.0, 0.0, 1.0, 0.0, 1.0, -acv, 1.0, 0.0, 1.0, -0.5)),
        ("L", (1.0, 0.0, 0.0, 0.5)),
        ("C", (1.0, 0.0, 0.0, acv, 1.0, -acv, 0.0, 0.0, 1.0, -0.5, 0.0, 0.0)),
        ("L", (0.0, 0.5, 0.0, 0.0)),
        ("C", (0.0, acv, 0.0, 0.0, 0.0, 0.0, 0.0, acv, 0.0, 0.0, 0.0, 0.5)),
    ]
    xs: list[float] = []
    ys: list[float] = []

    def pt(v, k):
        return x + v[4 * k] * w + v[4 * k + 1] * aw, y + v[4 * k + 2] * h + v[4 * k + 3] * ah

    cur = None
    for kind, v in table:
        if kind == "M":
            cur = pt(v, 0)
            xs.append(cur[0])
            ys.append(cur[1])
        elif kind == "L":
            cur = pt(v, 0)
            xs.append(cur[0])
            ys.append(cur[1])
        else:
            p1, p2, p3 = pt(v, 0), pt(v, 1), pt(v, 2)
            out: list[tuple[float, float]] = []
            _flatten_cubic([cur[0], cur[1], *p1, *p2, *p3], 0, flatness * flatness, out)
            for px, py in out:
                xs.append(px)
                ys.append(py)
            cur = p3
    if len(xs) > 1 and xs[0] == xs[-1] and ys[0] == ys[-1]:
        xs.pop()
        ys.pop()
    return f32(xs), f32(ys)


def flatten_path(start, cubics, flatness: float):
    """Flatten a closed path made of cubic segments (each a 6-tuple of control/end points)
    like ``FlatteningPathIterator`` on a float path; the duplicate closing point is dropped
    (ShapeRoi.getFloatPolygon). Returns float32-rounded lists."""
    sq = flatness * flatness
    cx, cy = float(np.float32(start[0])), float(np.float32(start[1]))
    xs, ys = [cx], [cy]
    for seg in cubics:
        seg = [float(np.float32(v)) for v in seg]
        out: list[tuple[float, float]] = []
        _flatten_cubic([cx, cy, *seg], 0, sq, out)
        for px, py in out:
            xs.append(float(np.float32(px)))
            ys.append(float(np.float32(py)))
        cx, cy = seg[4], seg[5]
    if len(xs) > 1 and xs[-1] == xs[0] and ys[-1] == ys[0]:
        xs.pop()
        ys.pop()
    return xs, ys


def _pt_seg_dist_sq(x1, y1, x2, y2, px, py) -> float:
    x2 -= x1
    y2 -= y1
    px -= x1
    py -= y1
    dot = px * x2 + py * y2
    if dot <= 0.0:
        proj = 0.0
    else:
        px = x2 - px
        py = y2 - py
        dot = px * x2 + py * y2
        proj = 0.0 if dot <= 0.0 else dot * dot / (x2 * x2 + y2 * y2)
    d = px * px + py * py - proj
    return d if d > 0 else 0.0


def _flatten_cubic(c, level, sq, out, limit=10):
    x1, y1, cx1, cy1, cx2, cy2, x2, y2 = c
    if level >= limit or max(_pt_seg_dist_sq(x1, y1, x2, y2, cx1, cy1),
                             _pt_seg_dist_sq(x1, y1, x2, y2, cx2, cy2)) < sq:
        out.append((x2, y2))
        return
    centerx = (cx1 + cx2) / 2.0
    centery = (cy1 + cy2) / 2.0
    cx1 = (x1 + cx1) / 2.0
    cy1 = (y1 + cy1) / 2.0
    cx2 = (x2 + cx2) / 2.0
    cy2 = (y2 + cy2) / 2.0
    cx12 = (cx1 + centerx) / 2.0
    cy12 = (cy1 + centery) / 2.0
    cx21 = (cx2 + centerx) / 2.0
    cy21 = (cy2 + centery) / 2.0
    centerx = (cx12 + cx21) / 2.0
    centery = (cy12 + cy21) / 2.0
    _flatten_cubic([x1, y1, cx1, cy1, cx12, cy12, centerx, centery], level + 1, sq, out, limit)
    _flatten_cubic([centerx, centery, cx21, cy21, cx2, cy2, x2, y2], level + 1, sq, out, limit)


# ------------------------------------------------------------------------------ hull & Feret
def convex_hull(xs, ys) -> tuple[np.ndarray, np.ndarray]:
    """``FloatPolygon.getConvexHull()`` (gift wrapping, counter-clockwise in screen
    coordinates, no collinear points), float32 like ImageJ."""
    xp = np.asarray(xs, dtype=np.float32)
    yp = np.asarray(ys, dtype=np.float32)
    n = len(xp)
    if n == 0:
        return np.zeros(0), np.zeros(0)
    xl = [float(v) for v in xp]
    yl = [float(v) for v in yp]
    smallest_y = min(yl)
    smallest_x = math.inf
    p1 = 0
    for i in range(n):
        if yl[i] == smallest_y and xl[i] < smallest_x:
            smallest_x = xl[i]
            p1 = i
    pstart = p1
    hx: list[float] = []
    hy: list[float] = []
    while True:
        x1, y1 = xl[p1], yl[p1]
        p2 = p1
        x2 = y2 = 0.0
        while True:
            p2 += 1
            if p2 == n:
                p2 = 0
            if p2 == p1:
                break
            x2, y2 = xl[p2], yl[p2]
            if not (x2 == x1 and y2 == y1):
                break
        p3 = p2 + 1
        if p3 == n:
            p3 = 0
        if p2 != p1:
            while True:
                x3, y3 = xl[p3], yl[p3]
                det = x1 * (y2 - y3) - y1 * (x2 - x3) + (y3 * x2 - y2 * x3)
                further = False
                if det == 0 and p3 != p2:
                    further = (x3 - x1) ** 2 + (y3 - y1) ** 2 > (x2 - x1) ** 2 + (y2 - y1) ** 2
                if det > 0 or further:
                    x2, y2, p2 = x3, y3, p3
                p3 += 1
                if p3 == n:
                    p3 = 0
                if p3 == p1:
                    break
        hx.append(x1)
        hy.append(y1)
        if p2 == p1 or len(hx) > n:
            break
        p1 = p2
        if len(hx) > 1 and xl[p1] == hx[0] and yl[p1] == hy[0]:
            break
        if p1 == pstart:
            break
    return f32(hx), f32(hy)


def float_convex_hull(roi):
    xs, ys = float_polygon(roi)
    return convex_hull(xs, ys)


def int_polygon_area(xs, ys) -> float:
    """``Analyzer.getArea(Polygon)`` of the convex hull rounded to integers."""
    xi = np.floor(np.asarray(xs, dtype=float) + 0.5).astype(np.int64)
    yi = np.floor(np.asarray(ys, dtype=float) + 0.5).astype(np.int64)
    if len(xi) == 0:
        return float("nan")
    carea = int(np.sum((xi + np.roll(xi, 1)) * (yi - np.roll(yi, 1))))
    return abs(carea / 2.0)


def polygon_area(xs, ys) -> float:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) < 3:
        return 0.0
    return abs(float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))) / 2.0


def feret_values(roi, pw: float = 1.0, ph: float = 1.0) -> np.ndarray | None:
    """``Roi.getFeretValues()``: [Feret, FeretAngle, MinFeret, FeretX, FeretY, ..., 8-11 max
    Feret end points, 12-15 min Feret end points]. Ovals use OvalRoi's exact version."""
    a = np.zeros(FERET_ARRAYSIZE)
    if isinstance(roi, S.OvalRoi):
        x, y, width, height = int_bounds(roi)
        high = ph * height > pw * width
        a[0] = height * ph if high else width * pw
        a[1] = 90.0 if high else 0.0
        a[2] = width * pw if high else height * ph
        a[3] = (x + (0.5 * width if high else 0)) * pw
        a[4] = (y + (height if high else 0.5 * height)) * ph
        i = FERET_ARRAY_POINTOFFSET
        a[i:i + 8] = [x + (0.5 * width if high else 0), y + (height if high else 0.5 * height),
                      x + (0.5 * width if high else width), y + (0 if high else 0.5 * height),
                      x + (0 if high else 0.5 * width), y + (0.5 * height if high else height),
                      x + (width if high else 0.5 * width), y + (0.5 * height if high else 0)]
        return a
    hx, hy = float_convex_hull(roi)
    n = len(hx)
    if n == 0:
        return None
    dx = (hx[:, None] - hx[None, :]) * pw
    dy = (hy[:, None] - hy[None, :]) * ph
    d2 = dx * dx + dy * dy
    iu = np.triu_indices(n, 1)
    if len(iu[0]):
        vals = d2[iu]
        k = int(np.argmax(vals))          # first maximum in (i, j) order, as ImageJ
        if vals[k] > 0:
            p1, p2 = int(iu[0][k]), int(iu[1][k])
            dmax = float(vals[k])
        else:
            p1 = p2 = 0
            dmax = 0.0
    else:
        p1 = p2 = 0
        dmax = 0.0
    if hx[p1] > hx[p2]:
        p1, p2 = p2, p1
    xf1, yf1, xf2, yf2 = hx[p1], hy[p1], hx[p2], hy[p2]
    angle = math.degrees(math.atan2((yf1 - yf2) * ph, (xf2 - xf1) * pw))
    if angle < 0.0:
        angle += 180.0
    a[0] = math.sqrt(dmax)
    a[1] = angle
    a[3], a[4] = xf1, yf1
    i = FERET_ARRAY_POINTOFFSET
    a[i:i + 4] = [hx[p1], hy[p1], hx[p2], hy[p2]]
    x0, y0 = hx[-1], hy[-1]
    min_feret = math.inf
    xy_end = [0.0] * 4
    for i in range(n):
        xprev, yprev = x0, y0
        x0, y0 = hx[i], hy[i]
        xnorm = (y0 - yprev) * ph
        ynorm = (xprev - x0) * pw
        norm = math.sqrt(xnorm * xnorm + ynorm * ynorm)
        if norm == 0:
            fac = math.inf
        else:
            fac = 1.0 / norm
        xnorm *= fac * pw
        ynorm *= fac * ph
        max_dist = 0.0
        xy_ei = [0.0] * 4
        with np.errstate(invalid="ignore"):
            dist = (hx - x0) * xnorm + (hy - y0) * ynorm
        for j in range(n):
            if dist[j] > max_dist:
                max_dist = float(dist[j])
                xy_ei = [hx[j], hy[j], hx[j] - (xnorm / pw * max_dist) / pw,
                         hy[j] - (ynorm / ph * max_dist) / ph]
        if max_dist < min_feret:
            min_feret = max_dist
            xy_end = xy_ei
    a[2] = min_feret if min_feret != math.inf else 0.0
    a[FERET_ARRAY_POINTOFFSET + 4:] = xy_end
    return a


# ------------------------------------------------------------------------------ lengths
def is_traced(roi) -> bool:
    """``PolygonRoi.isTraced()``: wand/particle outlines (and integer polygons whose edges are
    all horizontal or vertical) use the traced-perimeter formula."""
    if not isinstance(roi, S.PolygonRoi) or isinstance(roi, (S.RotatedRectRoi,)):
        return False
    if roi.type == S.TRACED_ROI:
        return True
    if roi.type != S.POLYGON:
        return False
    xs, ys = roi.xs, roi.ys
    if len(xs) == 0 or np.any(xs != np.floor(xs)) or np.any(ys != np.floor(ys)):
        return False
    dx = np.roll(xs, -1) - xs
    dy = np.roll(ys, -1) - ys
    return bool(np.all((dx == 0) | (dy == 0)))


def traced_perimeter(xs, ys, pw: float = 1.0, ph: float = 1.0) -> float:
    """``PolygonRoi.getTracedPerimeter``: edges count 1, corners sqrt(2) (non-adjacent)."""
    xp = [int(v) for v in xs]
    yp = [int(v) for v in ys]
    n = len(xp)
    if n < 4:
        return 0.0
    sumdx = sumdy = ncorners = 0
    dx1 = xp[0] - xp[n - 1]
    dy1 = yp[0] - yp[n - 1]
    side1 = abs(dx1) + abs(dy1)
    corner = False
    for i in range(n):
        nexti = 0 if i + 1 == n else i + 1
        dx2 = xp[nexti] - xp[i]
        dy2 = yp[nexti] - yp[i]
        sumdx += abs(dx1)
        sumdy += abs(dy1)
        side2 = abs(dx2) + abs(dy2)
        if side1 > 1 or not corner:
            corner = True
            ncorners += 1
        else:
            corner = False
        dx1, dy1, side1 = dx2, dy2, side2
    return sumdx * pw + sumdy * ph - (ncorners * ((pw + ph) - math.sqrt(pw * pw + ph * ph)))


def polyline_length(xs, ys, closed: bool, pw: float = 1.0, ph: float = 1.0) -> float:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) < 2:
        return 0.0
    length = 0.0
    for i in range(len(xs) - 1):
        length += math.sqrt(((xs[i + 1] - xs[i]) * pw) ** 2 + ((ys[i + 1] - ys[i]) * ph) ** 2)
    if closed:
        length += math.sqrt(((xs[0] - xs[-1]) * pw) ** 2 + ((ys[0] - ys[-1]) * ph) ** 2)
    return length


def smoothed_line_length(xs, ys, pw: float = 1.0, ph: float = 1.0) -> float:
    """``getSmoothedLineLength`` (3-point running average)."""
    xp = np.asarray(xs, dtype=float)
    yp = np.asarray(ys, dtype=float)
    n = len(xp)
    w2, h2 = pw * pw, ph * ph
    dx = (xp[0] + xp[1] + xp[2]) / 3.0 - xp[0]
    dy = (yp[0] + yp[1] + yp[2]) / 3.0 - yp[0]
    length = math.sqrt(dx * dx * w2 + dy * dy * h2)
    for i in range(1, n - 2):
        dx = (xp[i + 2] - xp[i - 1]) / 3.0
        dy = (yp[i + 2] - yp[i - 1]) / 3.0
        length += math.sqrt(dx * dx * w2 + dy * dy * h2)
    dx = xp[n - 1] - (xp[n - 3] + xp[n - 2] + xp[n - 1]) / 3.0
    dy = yp[n - 1] - (yp[n - 3] + yp[n - 2] + yp[n - 1]) / 3.0
    length += math.sqrt(dx * dx * w2 + dy * dy * h2)
    return length


def roi_length(roi, pw: float = 1.0, ph: float = 1.0) -> float:
    """ImageJ ``Roi.getLength()``: perimeter of area ROIs, length of lines."""
    if isinstance(roi, S.LineRoi):
        return math.sqrt(((roi.x2 - roi.x1) * pw) ** 2 + ((roi.y2 - roi.y1) * ph) ** 2)
    if isinstance(roi, S.OvalRoi):
        _, _, w, h = int_bounds(roi)
        a, b = 0.5 * w * pw, 0.5 * h * ph
        return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
    if isinstance(roi, S.RectRoi):
        _, _, w, h = int_bounds(roi)
        perimeter = 2.0 * w * pw + 2.0 * h * ph
        cd = roi.corner_diameter
        if cd and cd > 0:
            a = 0.5 * min(cd, w) * pw
            b = 0.5 * min(cd, h) * ph
            perimeter += math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b))) \
                - 4 * (a + b)
        return perimeter
    if isinstance(roi, S.CompositeRoi):
        return sum(roi_length(r, pw, ph) for _, r in roi.parts)
    if isinstance(roi, S.PointRoi):
        return 0.0
    if isinstance(roi, (S.RotatedRectRoi, S.EllipseRoi)):
        xs, ys = float_polygon(roi)
        if isinstance(roi, S.EllipseRoi) and len(xs) > 2:
            return smoothed_line_length(xs, ys, pw, ph) + math.sqrt(
                ((xs[-1] - xs[0]) * pw) ** 2 + ((ys[-1] - ys[0]) * ph) ** 2)
        return polyline_length(xs, ys, True, pw, ph)
    xs, ys = float_polygon(roi)
    if is_traced(roi):
        return traced_perimeter(xs, ys, pw, ph)
    n = len(xs)
    if n > 2:
        if roi.type == S.FREEROI:
            return smoothed_line_length(xs, ys, pw, ph) + math.sqrt(
                ((xs[-1] - xs[0]) * pw) ** 2 + ((ys[-1] - ys[0]) * ph) ** 2)
        if roi.type == S.FREELINE and np.ptp(xs) != 0 and np.ptp(ys) != 0:
            return smoothed_line_length(xs, ys, pw, ph)
    return polyline_length(xs, ys, roi.is_area(), pw, ph)


def line_angle(x1, y1, x2, y2) -> float:
    """``Roi.getFloatAngle``: degrees, counter-clockwise, y up."""
    return math.degrees(math.atan2(y1 - y2, x2 - x1))


def polygon_angle(roi) -> float:
    """Angle ROI (3 points): the angle at the middle vertex, 0-180 degrees."""
    xs, ys = float_polygon(roi)
    if len(xs) < 3:
        return float("nan")
    a1 = line_angle(xs[0], ys[0], xs[1], ys[1])
    a2 = line_angle(xs[1], ys[1], xs[2], ys[2])
    degrees = abs(180 - abs(a1 - a2))
    if degrees > 180.0:
        degrees = 360.0 - degrees
    return degrees


# ------------------------------------------------------------------------------ interpolation
def interpolated_polygon(xs, ys, interval: float = 1.0, smooth: bool = False,
                         closed: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Points spaced ``interval`` apart along the outline (``getInterpolatedPolygon``);
    a negative interval may be shortened so the last segment ends on the end point."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    adjust = interval < 0
    interval = abs(interval) or 1.0
    if smooth and len(xs) > 3:
        if closed:
            xs = (np.roll(xs, 1) + xs + np.roll(xs, -1)) / 3.0
            ys = (np.roll(ys, 1) + ys + np.roll(ys, -1)) / 3.0
        else:
            xs = np.concatenate([[xs[0]], (xs[:-2] + xs[1:-1] + xs[2:]) / 3.0, [xs[-1]]])
            ys = np.concatenate([[ys[0]], (ys[:-2] + ys[1:-1] + ys[2:]) / 3.0, [ys[-1]]])
    if closed and len(xs) > 1:
        xs = np.append(xs, xs[0])
        ys = np.append(ys, ys[0])
    if len(xs) < 2:
        return xs.copy(), ys.copy()
    seg = np.hypot(np.diff(xs), np.diff(ys))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total == 0:
        return xs[:1].copy(), ys[:1].copy()
    if adjust:
        nseg = max(1, int(round(total / interval)))
        interval = total / nseg
    d = np.arange(0.0, total + 1e-9, interval)
    if closed and len(d) > 1 and total - d[-1] < interval * 0.5:
        d = d[:-1]
    if not closed and total - d[-1] > 1e-9 and adjust:
        d = np.append(d, total)
    px = np.interp(d, cum, xs)
    py = np.interp(d, cum, ys)
    return px, py


def spline_fit(xs, ys, closed: bool, n_points: int | None = None):
    """Cubic spline through the vertices (``PolygonRoi.fitSpline``; ImageJ's SplineFitter
    evaluates about 200 points or 2 per pixel of length)."""
    from scipy.interpolate import CubicSpline

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) < 3:
        return xs.copy(), ys.copy()
    if closed:
        px = np.append(xs, xs[0])
        py = np.append(ys, ys[0])
    else:
        px, py = xs, ys
    t = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(px), np.diff(py)))])
    keep = np.concatenate([[True], np.diff(t) > 0])
    px, py, t = px[keep], py[keep], t[keep]
    if len(t) < 3:
        return xs.copy(), ys.copy()
    bc = "periodic" if closed else "not-a-knot"
    if closed and (px[0] != px[-1] or py[0] != py[-1]):
        bc = "not-a-knot"
    sx = CubicSpline(t, px, bc_type=bc)
    sy = CubicSpline(t, py, bc_type=bc)
    if n_points is None:
        n_points = max(200, int(t[-1] / 2))
    tt = np.linspace(0, t[-1], n_points, endpoint=not closed)
    return sx(tt), sy(tt)


def fit_circle_points(xs, ys):
    """``Selection.fitCircle`` for points/lines (Newton's method on the characteristic
    polynomial, Chernov & Lesort); returns (cx, cy, r) or None when collinear."""
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    n = len(x)
    if n < 3:
        return None
    meanx, meany = x.sum() / n, y.sum() / n
    X, Y = x - meanx, y - meany
    Z = X * X + Y * Y
    Mxy, Mxx, Myy = (X * Y).sum() / n, (X * X).sum() / n, (Y * Y).sum() / n
    Mxz, Myz, Mzz = (X * Z).sum() / n, (Y * Z).sum() / n, (Z * Z).sum() / n
    Mz = Mxx + Myy
    cov = Mxx * Myy - Mxy * Mxy
    Mxz2, Myz2 = Mxz * Mxz, Myz * Myz
    A2 = 4 * cov - 3 * Mz * Mz - Mzz
    A1 = Mzz * Mz + 4 * cov * Mz - Mxz2 - Myz2 - Mz * Mz * Mz
    A0 = Mxz2 * Myy + Myz2 * Mxx - Mzz * cov - 2 * Mxz * Myz * Mxy + Mz * Mz * cov
    A22 = A2 + A2
    ynew, xnew = 1e20, 0.0
    for it in range(1, 21):
        yold = ynew
        ynew = A0 + xnew * (A1 + xnew * (A2 + 4.0 * xnew * xnew))
        if abs(ynew) > abs(yold):
            xnew = 0.0
            break
        Dy = A1 + xnew * (A22 + 16 * xnew * xnew)
        xold = xnew
        xnew = xold - ynew / Dy if Dy != 0 else xold
        if xnew != 0 and abs((xnew - xold) / xnew) < 1e-12:
            break
        if it >= 20:
            xnew = 0.0
        if xnew < 0:
            xnew = 0.0
    det = xnew * xnew - xnew * Mz + cov
    if det == 0:
        return None
    cx = (Mxz * (Myy - xnew) - Myz * Mxy) / (2 * det)
    cy = (Myz * (Mxx - xnew) - Mxz * Mxy) / (2 * det)
    r2 = cx * cx + cy * cy + Mz + 2 * xnew
    if not np.isfinite(r2) or r2 < 0:
        return None
    return cx + meanx, cy + meany, math.sqrt(r2)


def rotate_points(xs, ys, angle_deg: float, xc: float, yc: float):
    """Rotate counter-clockwise (screen: y down) by ``angle_deg`` about (xc, yc) as
    ``RoiRotator.rotate``."""
    theta = -angle_deg * math.pi / 180.0
    c, s = math.cos(theta), math.sin(theta)
    xs = np.asarray(xs, dtype=float) - xc
    ys = np.asarray(ys, dtype=float) - yc
    return xs * c - ys * s + xc, xs * s + ys * c + yc


def min_area_rect(xs, ys):
    """Smallest enclosing rotated rectangle of the convex hull (``Selection.fitRectangle``):
    returns (x1, y1, x2, y2, width) for a RotatedRectRoi."""
    hx, hy = convex_hull(xs, ys)
    n = len(hx)
    if n < 3:
        return None
    best = None
    for i in range(n):
        j = (i + 1) % n
        ex, ey = hx[j] - hx[i], hy[j] - hy[i]
        ln = math.hypot(ex, ey)
        if ln == 0:
            continue
        ux, uy = ex / ln, ey / ln
        proj_u = (hx - hx[i]) * ux + (hy - hy[i]) * uy
        proj_v = -(hx - hx[i]) * uy + (hy - hy[i]) * ux
        umin, umax = proj_u.min(), proj_u.max()
        vmin, vmax = proj_v.min(), proj_v.max()
        area = (umax - umin) * (vmax - vmin)
        if best is None or area < best[0]:
            best = (area, i, ux, uy, umin, umax, vmin, vmax)
    _, i, ux, uy, umin, umax, vmin, vmax = best
    ox, oy = hx[i], hy[i]
    vm = (vmin + vmax) / 2
    x1 = ox + umin * ux - vm * uy
    y1 = oy + umin * uy + vm * ux
    x2 = ox + umax * ux - vm * uy
    y2 = oy + umax * uy + vm * ux
    return x1, y1, x2, y2, vmax - vmin


# ------------------------------------------------------------------------------ tracing
def trace_mask_outlines(mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """All boundary loops (outer outlines and holes) of a boolean mask as integer corner
    polygons, collinear points removed; foreground is 8-connected at diagonal contacts, like
    ImageJ's ThresholdToSelection. Filling the loops with the even-odd rule gives ``mask``."""
    m = np.pad(np.asarray(mask, dtype=bool), 1)
    h, w = m.shape
    # directed edges with the foreground on the right (screen coordinates, y down):
    # top edge of a pixel whose upper neighbour is background runs left->right, etc.
    nxt: dict[tuple[int, int], list[tuple[int, int]]] = {}
    ys, xs = np.nonzero(m[1:, :] & ~m[:-1, :])      # pixel (y+1) fg, above bg: top edge
    for y, x in zip(ys.tolist(), xs.tolist()):
        nxt.setdefault((x, y + 1), []).append((x + 1, y + 1))
    ys, xs = np.nonzero(m[:-1, :] & ~m[1:, :])      # bottom edge: right->left
    for y, x in zip(ys.tolist(), xs.tolist()):
        nxt.setdefault((x + 1, y + 1), []).append((x, y + 1))
    ys, xs = np.nonzero(m[:, :-1] & ~m[:, 1:])      # right edge: top->bottom
    for y, x in zip(ys.tolist(), xs.tolist()):
        nxt.setdefault((x + 1, y), []).append((x + 1, y + 1))
    ys, xs = np.nonzero(m[:, 1:] & ~m[:, :-1])      # left edge: bottom->top
    for y, x in zip(ys.tolist(), xs.tolist()):
        nxt.setdefault((x + 1, y + 1), []).append((x + 1, y))
    loops = []
    while nxt:
        start = min(nxt)
        pts = [start]
        cur = start
        prev_dir = None
        while True:
            outs = nxt.get(cur)
            if not outs:
                break
            if len(outs) == 1:
                nb = outs.pop()
            else:  # saddle: turn left (keeps diagonal foreground pixels connected)
                nb = _pick_turn(cur, prev_dir, outs)
                outs.remove(nb)
            if not outs:
                del nxt[cur]
            prev_dir = (nb[0] - cur[0], nb[1] - cur[1])
            cur = nb
            if cur == start:
                break
            pts.append(cur)
        arr = np.asarray(pts, dtype=float) - 1.0
        px, py = _simplify(arr[:, 0], arr[:, 1])
        loops.append((px, py))
    return loops


def _pick_turn(cur, prev_dir, outs):
    if prev_dir is None:
        return outs[0]
    dx, dy = prev_dir
    left = (dy, -dx)             # screen coordinates (y down): turning left
    for nb in outs:
        if (nb[0] - cur[0], nb[1] - cur[1]) == left:
            return nb
    return outs[0]


def _simplify(xs, ys):
    n = len(xs)
    if n < 3:
        return xs, ys
    px, py = np.roll(xs, 1), np.roll(ys, 1)
    nx, ny = np.roll(xs, -1), np.roll(ys, -1)
    cross = (xs - px) * (ny - ys) - (ys - py) * (nx - xs)
    keep = cross != 0
    if not keep.any():
        return xs, ys
    return xs[keep], ys[keep]

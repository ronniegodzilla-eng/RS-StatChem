"""Edit > Selection operations (``ij.plugin.Selection``, ``RoiEnlarger``, ``RoiRotator``,
``RoiScaler``, ``ThresholdToSelection``, ``Straightener``), as pure functions of ROIs and
images. Commands in ``ui/imagej/commands_analyze.py`` call them.

Ported from ImageJ 1.54 ij/plugin/Selection.java, RoiEnlarger.java, RoiRotator.java,
RoiScaler.java, Straightener.java and ij/plugin/filter/ThresholdToSelection.java (public
domain, NIH; W. Rasband, M. Schmid et al.).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from rsstatchem.core.imaging import measure as M
from rsstatchem.core.imaging.image import GRAY8, ImageDoc
from rsstatchem.core.imaging.lut import LUT
from rsstatchem.core.imaging.roi import geometry as G
from rsstatchem.core.imaging.roi import shapes as S
from rsstatchem.core.imaging.roi.mask import polygon_mask


class SelectionError(ValueError):
    """User-facing error (the command layer turns it into a CommandError)."""


def _transfer(src, dst):
    if src is None or dst is None:
        return dst
    dst.name = src.name
    dst.stroke_color = src.stroke_color
    dst.fill_color = src.fill_color
    dst.stroke_width = src.stroke_width
    dst.position = src.position
    dst.properties = dict(src.properties)
    return dst


def select_all(image) -> S.RectRoi:
    return S.RectRoi(0, 0, image.width, image.height)


# ---------------------------------------------------------------------------------- fits
def fit_spline(roi):
    if not isinstance(roi, S.PolygonRoi) or roi.type not in (S.POLYGON, S.POLYLINE, S.FREEROI,
                                                             S.TRACED_ROI, S.FREELINE):
        raise SelectionError("Polygon or polyline selection required")
    closed = roi.is_area()
    xs, ys = roi.xs, roi.ys
    if roi.type in (S.FREEROI, S.TRACED_ROI, S.FREELINE) and len(xs) > 3:
        # trimPolygon: keep about one vertex per 4-8 pixels of length before fitting
        length = G.roi_length(roi)
        n = max(4, int(length / 8))
        step = max(1, len(xs) // n)
        xs, ys = xs[::step], ys[::step]
    sx, sy = G.spline_fit(xs, ys, closed)
    typ = roi.type
    if typ == S.TRACED_ROI:
        typ = S.POLYGON
    out = S.PolygonRoi(sx, sy, type=typ, spline_fit=True)
    return _transfer(roi, out)


def fit_circle(image, roi):
    if roi is None:
        raise SelectionError("Fit Circle requires a selection")
    if roi.is_area():
        rect, mask = M.roi_rect_and_mask(roi, image.width, image.height)
        st = M.get_statistics(np.zeros((image.height, image.width), np.uint8), GRAY8, rect,
                              mask, M.AREA | M.CENTROID, None)
        r = math.sqrt(st.pixel_count / math.pi)
        d = int(math.floor(2.0 * r + 0.5))
        out = S.OvalRoi(int(math.floor(st.x_centroid - r + 0.5)),
                        int(math.floor(st.y_centroid - r + 0.5)), d, d)
        return _transfer(roi, out)
    xs, ys = G.float_polygon(roi)
    xs = np.floor(xs + 0.5)
    ys = np.floor(ys + 0.5)
    if len(xs) < 3:
        raise SelectionError("At least 3 points are required to fit a circle.")
    fit = G.fit_circle_points(xs, ys)
    if fit is None:
        raise SelectionError("Points are collinear.")
    cx, cy, r = fit
    jr = lambda v: int(math.floor(v + 0.5))  # noqa: E731
    return S.OvalRoi(jr(cx - r), jr(cy - r), jr(2 * r), jr(2 * r))


def fit_ellipse(image, roi):
    if roi is None:
        raise SelectionError("Fit Ellipse requires a selection")
    if roi.is_line():
        raise SelectionError('"Fit Ellipse" does not work with line selections')
    rect, mask = M.roi_rect_and_mask(roi, image.width, image.height)
    st = M.get_statistics(np.zeros((image.height, image.width), np.uint8), GRAY8, rect, mask,
                          M.CENTROID | M.ELLIPSE, None)
    dx = st.major * math.cos(st.angle / 180.0 * math.pi) / 2.0
    dy = -st.major * math.sin(st.angle / 180.0 * math.pi) / 2.0
    out = S.EllipseRoi(st.x_centroid - dx, st.y_centroid - dy, st.x_centroid + dx,
                       st.y_centroid + dy, st.minor / st.major if st.major else 1.0)
    return _transfer(roi, out)


def fit_rectangle(roi):
    if roi is None or not roi.is_area():
        raise SelectionError("Area selection required")
    xs, ys = G.float_polygon(roi)
    r = G.min_area_rect(xs, ys)
    if r is None:
        raise SelectionError("Fit Rectangle needs at least 3 points")
    return _transfer(roi, S.RotatedRectRoi(*r[:4], rect_width=r[4]))


def interpolate(roi, interval: float = 1.0, smooth: bool = False, adjust: bool = False):
    if roi is None:
        raise SelectionError("Interpolate requires a selection")
    if isinstance(roi, S.PointRoi):
        return roi
    if isinstance(roi, S.CompositeRoi):
        parts = []
        for op, part in roi.parts:
            xs, ys = G.float_polygon(part)
            px, py = G.interpolated_polygon(xs, ys, interval, smooth, closed=True)
            parts.append((op, S.PolygonRoi(px, py, type=S.POLYGON)))
        return _transfer(roi, S.CompositeRoi(parts))
    xs, ys = G.float_polygon(roi)
    closed = roi.is_area()
    px, py = G.interpolated_polygon(xs, ys, -interval if adjust else interval, smooth, closed)
    t = roi.type
    typ = S.FREELINE if roi.is_line() else S.FREEROI
    if t == S.POLYGON and interval > 1.0:
        typ = S.POLYGON
    if t in (S.RECTANGLE, S.OVAL, S.FREEROI) and interval >= 8.0:
        typ = S.POLYGON
    if t in (S.LINE, S.FREELINE, S.POLYLINE) and interval >= 8.0:
        typ = S.POLYLINE
    out = S.PolygonRoi(px, py, type=typ)
    return _transfer(roi, out)


def convex_hull(roi):
    if roi is None:
        raise SelectionError("Selection required")
    if isinstance(roi, S.LineRoi):
        raise SelectionError("Area selection, point selection, or segmented or free line "
                             "required")
    hx, hy = G.float_convex_hull(roi)
    return _transfer(roi, S.PolygonRoi(hx, hy, type=S.POLYGON))


# ---------------------------------------------------------------------------------- masks
def make_inverse(image, roi):
    if roi is None:
        return select_all(image)
    if not roi.is_area():
        raise SelectionError("Area selection required")
    return S.CompositeRoi([("or", S.RectRoi(0, 0, image.width, image.height)),
                           ("not", roi.copy())])


def roi_mask(image, roi) -> np.ndarray:
    return roi.get_mask((image.height, image.width))


def create_mask(image, roi=None, black_background: bool = False, existing=None) -> ImageDoc:
    """Edit > Selection > Create Mask: 255 inside the selection (or the thresholded pixels
    when there is no selection), inverting LUT unless black background."""
    if roi is None or (isinstance(roi, S.RectRoi) and G.int_bounds(roi) ==
                       (0, 0, image.width, image.height) and image.threshold is not None):
        if image.threshold is None:
            if image.overlay is not None and len(image.overlay):
                m = np.zeros((image.height, image.width), bool)
                for r in image.overlay:
                    if r.is_area():
                        m |= roi_mask(image, r)
                title = "Mask"
            else:
                raise SelectionError("Area selection, overlay or thresholded image required")
        else:
            p = image.plane()
            lo, hi = image.threshold[0], image.threshold[1]
            with np.errstate(invalid="ignore"):
                m = (p >= lo) & (p <= hi) if p.ndim == 2 else np.zeros(p.shape[:2], bool)
            title = "mask"
    else:
        if not roi.is_area():
            m = np.zeros((image.height, image.width), bool)
            if isinstance(roi, (S.LineRoi,)) or roi.is_line():
                m = roi_mask(image, line_to_area(roi, max(1, int(round(
                    getattr(roi, "width", 1) or 1)))))
        else:
            m = roi_mask(image, roi)
        title = "Mask"
    arr = np.where(m, 255, 0).astype(np.uint8)
    if existing is not None and existing.type == GRAY8 and title == "Mask" and \
            existing.width == image.width and existing.height == image.height:
        existing.set_plane(np.maximum(existing.plane(), arr))
        return existing
    doc = ImageDoc.from_array(arr, "YX", title)
    if not black_background:
        doc.luts = [LUT.grays().inverted()]
    cal = image.calibration
    if cal.scaled():
        doc.calibration.pixel_width = cal.pixel_width
        doc.calibration.pixel_height = cal.pixel_height
        doc.calibration.unit = cal.unit
    from rsstatchem.core.imaging.image import Threshold
    doc.threshold = Threshold(255, 255, "none")
    return doc


def mask_to_roi(mask: np.ndarray):
    """ThresholdToSelection of a boolean mask: a traced polygon, or a composite ROI of all
    outlines (even-odd) when there are several; None for an empty mask."""
    loops = G.trace_mask_outlines(mask)
    if not loops:
        return None
    if len(loops) == 1:
        xs, ys = loops[0]
        return S.PolygonRoi(xs, ys, type=S.TRACED_ROI)
    return S.CompositeRoi([("or" if i == 0 else "xor", S.PolygonRoi(xs, ys, type=S.TRACED_ROI))
                           for i, (xs, ys) in enumerate(loops)])


def create_selection(image, black_background: bool = False):
    """Edit > Selection > Create Selection from the threshold or a binary mask."""
    p = image.plane()
    if image.threshold is not None and p.ndim == 2:
        lo, hi = image.threshold[0], image.threshold[1]
        with np.errstate(invalid="ignore"):
            m = (p >= lo) & (p <= hi)
    else:
        if p.ndim != 2 or image.bit_depth != 8 or not np.all(np.isin(np.unique(p), (0, 255))):
            raise SelectionError("This command creates a composite selection from\n"
                                 "a mask (8-bit binary image) or from an image\n"
                                 "thresholded using Image>Adjust>Threshold\n"
                                 "The current image is not a mask and has not\n"
                                 "been thresholded.")
        level = 255 if image.is_inverted_lut() else 0
        if black_background:
            level = 0 if level == 255 else 255
        m = p == level
    return mask_to_roi(m)


# ---------------------------------------------------------------------------------- transforms
def _as_polygon(roi):
    if isinstance(roi, S.LineRoi):
        return roi
    xs, ys = G.float_polygon(roi)
    typ = roi.type if isinstance(roi, S.PolygonRoi) and not isinstance(
        roi, (S.EllipseRoi, S.RotatedRectRoi)) else (S.POLYGON if roi.is_area() else S.POLYLINE)
    if typ == S.TRACED_ROI:
        typ = S.POLYGON
    return S.PolygonRoi(xs, ys, type=typ)


def rotate(roi, angle: float, xc: float | None = None, yc: float | None = None):
    """``RoiRotator.rotate``: about the bounds centre (or (xc, yc))."""
    if roi is None:
        raise SelectionError("Selection required")
    if xc is None or yc is None:
        x, y, w, h = roi.bounds()
        xc, yc = x + w / 2.0, y + h / 2.0
    if isinstance(roi, S.CompositeRoi):
        return _transfer(roi, S.CompositeRoi([(op, rotate(p, angle, xc, yc))
                                              for op, p in roi.parts]))
    if isinstance(roi, S.LineRoi):
        (x1, x2), (y1, y2) = G.rotate_points([roi.x1, roi.x2], [roi.y1, roi.y2], angle, xc, yc)
        out = S.LineRoi(x1, y1, x2, y2, width=roi.width, arrow=roi.arrow)
        return _transfer(roi, out)
    if isinstance(roi, S.PointRoi):
        xs, ys = G.rotate_points(roi.xs, roi.ys, angle, xc, yc)
        out = S.PointRoi(xs, ys, counters=roi.counters, positions=roi.positions)
        return _transfer(roi, out)
    if isinstance(roi, S.EllipseRoi):
        (x1, x2), (y1, y2) = G.rotate_points([roi.x1, roi.x2], [roi.y1, roi.y2], angle, xc, yc)
        return _transfer(roi, S.EllipseRoi(x1, y1, x2, y2, roi.aspect_ratio))
    if isinstance(roi, S.RotatedRectRoi):
        (x1, x2), (y1, y2) = G.rotate_points([roi.x1, roi.x2], [roi.y1, roi.y2], angle, xc, yc)
        return _transfer(roi, S.RotatedRectRoi(x1, y1, x2, y2, rect_width=roi.rect_width))
    p = _as_polygon(roi)
    xs, ys = G.rotate_points(p.xs, p.ys, angle, xc, yc)
    return _transfer(roi, S.PolygonRoi(xs, ys, type=p.type))


def scale(roi, sx: float, sy: float, centered: bool = False):
    """``RoiScaler.scale`` about the origin, or about the bounds centre."""
    if roi is None:
        raise SelectionError("Selection required")
    x, y, w, h = roi.bounds()
    cx, cy = (x + w / 2.0, y + h / 2.0) if centered else (0.0, 0.0)

    def f(px, py):
        return (np.asarray(px, float) - cx) * sx + cx, (np.asarray(py, float) - cy) * sy + cy
    if isinstance(roi, (S.RectRoi, S.OvalRoi)) and not getattr(roi, "corner_diameter", 0):
        (x0,), (y0,) = f([x], [y])
        out = type(roi)(float(x0), float(y0), w * sx, h * sy)
        return _transfer(roi, out)
    if isinstance(roi, S.CompositeRoi):
        return _transfer(roi, S.CompositeRoi([(op, scale(p, sx, sy, False)) for op, p in
                                              roi.parts]) if not centered else
                         S.CompositeRoi([(op, _scale_about(p, sx, sy, cx, cy))
                                         for op, p in roi.parts]))
    return _transfer(roi, _scale_about(roi, sx, sy, cx, cy))


def _scale_about(roi, sx, sy, cx, cy):
    if isinstance(roi, S.LineRoi):
        return S.LineRoi((roi.x1 - cx) * sx + cx, (roi.y1 - cy) * sy + cy,
                         (roi.x2 - cx) * sx + cx, (roi.y2 - cy) * sy + cy, width=roi.width)
    if isinstance(roi, S.PointRoi):
        return S.PointRoi((roi.xs - cx) * sx + cx, (roi.ys - cy) * sy + cy,
                          counters=roi.counters, positions=roi.positions)
    p = _as_polygon(roi)
    return S.PolygonRoi((p.xs - cx) * sx + cx, (p.ys - cy) * sy + cy, type=p.type)


def translate(roi, dx: float, dy: float):
    out = roi.copy()
    out.translate(dx, dy)
    return out


def enlarge(image, roi, n: float, pixel_units: bool = True):
    """``RoiEnlarger.enlarge255``: n is rounded to whole pixels; rectangles and ovals stay
    rectangles/ovals; other shapes grow (or shrink, n < 0) by thresholding ImageJ's 8-bit
    distance map (distances rounded half up): grow keeps round(d) <= n, shrink keeps
    round(d) >= |n| + 1."""
    if roi is None or not roi.is_area():
        raise SelectionError("Area selection required")
    n = int(math.floor(n + 0.5))
    if n == 0:
        return roi
    if isinstance(roi, S.RectRoi) and not roi.corner_diameter:
        x, y, w, h = G.int_bounds(roi)
        return _transfer(roi, S.RectRoi(x - n, y - n, w + 2 * n, h + 2 * n))
    if isinstance(roi, S.OvalRoi):
        x, y, w, h = G.int_bounds(roi)
        return _transfer(roi, S.OvalRoi(x - n, y - n, w + 2 * n, h + 2 * n))
    x, y, w, h = G.int_bounds(roi)
    pad = int(math.ceil(abs(n))) + 2
    ox, oy = x - pad, y - pad
    local = roi.copy()
    local.translate(-ox, -oy)
    m = local.get_mask((h + 2 * pad, w + 2 * pad))
    if n > 0:
        d = np.floor(ndimage.distance_transform_edt(~m) + 0.5)
        m2 = d <= n
    else:
        d = np.floor(ndimage.distance_transform_edt(m) + 0.5)
        m2 = d >= -n + 1
    out = mask_to_roi(m2)
    if out is None:
        raise SelectionError("Selection vanished")
    out.translate(ox, oy)
    return _transfer(roi, out)


def make_band(image, roi, band: float):
    """Edit > Selection > Make Band (``Selection.makeBand``): the 8-bit distance map of the
    image outside the selection (rounded distances) thresholded at 0..n, its outer outline
    (wand) minus the selection; clipped to the image."""
    if roi is None or not roi.is_area():
        raise SelectionError("Area selection required")
    n = int(math.floor(band + 0.5))
    if n > 255:
        raise SelectionError("Cannot make bands wider that 255 pixels")
    m = roi.get_mask((image.height, image.width))
    if not m.any():
        raise SelectionError("Unable to make band")
    d = np.floor(ndimage.distance_transform_edt(~m) + 0.5)
    region = d <= n
    lab, _ = ndimage.label(region, structure=np.ones((3, 3), int))
    ys, xs = np.nonzero(m)
    region = ndimage.binary_fill_holes(lab == lab[ys[0], xs[0]])
    ring = region & ~m
    out = mask_to_roi(ring)
    if out is None:
        raise SelectionError("Band is empty")
    return _transfer(roi, out)


def to_bounding_box(roi):
    if roi is None:
        raise SelectionError("To Bounding Box requires a selection")
    x, y, w, h = G.int_bounds(roi)
    return _transfer(roi, S.RectRoi(x, y, w, h))


def line_to_area(roi, width: float | None = None):
    """``Roi.convertLineToArea``: straight lines become rotated rectangles; segmented and
    freehand lines the union of their segment rectangles and joints."""
    if roi is None or not roi.is_line():
        raise SelectionError("Line selection required")
    lw = width if width is not None else max(getattr(roi, "width", 1.0) or 1.0,
                                             roi.stroke_width or 1.0)
    lw = max(lw, 1.0)
    if isinstance(roi, S.LineRoi):
        return _transfer(roi, S.RotatedRectRoi(roi.x1, roi.y1, roi.x2, roi.y2, rect_width=lw))
    xs, ys = G.float_polygon(roi)
    x0 = int(math.floor(min(xs) - lw)) - 1
    y0 = int(math.floor(min(ys) - lw)) - 1
    W = int(math.ceil(max(xs) + lw)) - x0 + 2
    H = int(math.ceil(max(ys) + lw)) - y0 + 2
    m = np.zeros((H, W), bool)
    r = lw / 2.0
    n = len(xs)
    for i in range(n - 1):
        dx, dy = xs[i + 1] - xs[i], ys[i + 1] - ys[i]
        ln = math.hypot(dx, dy) or 1.0
        ux, uy = dx / ln, dy / ln
        ext0 = 0.5 if i == 0 else 0.0
        ext1 = 0.5 if i == n - 2 else 0.0
        ax, ay = xs[i] - ux * ext0 - x0, ys[i] - uy * ext0 - y0
        bx, by = xs[i + 1] + ux * ext1 - x0, ys[i + 1] + uy * ext1 - y0
        px, py = -uy * r, ux * r
        qx = [ax + px, bx + px, bx - px, ax - px]
        qy = [ay + py, by + py, by - py, ay - py]
        m |= polygon_mask(np.asarray(qx) + 0.5, np.asarray(qy) + 0.5, (H, W))
        if 0 < i:
            yy, xx = np.mgrid[0:H, 0:W]
            m |= (xx - (xs[i] - x0)) ** 2 + (yy - (ys[i] - y0)) ** 2 <= r * r
    out = mask_to_roi(m)
    if out is None:
        raise SelectionError("Line to Area failed")
    out.translate(x0, y0)
    return _transfer(roi, out)


def area_to_line(roi):
    if roi is None or not roi.is_area():
        raise SelectionError("Area selection required")
    if isinstance(roi, S.CompositeRoi):
        raise SelectionError("Composite selections cannot be converted to lines.")
    xs, ys = G.float_polygon(roi)
    if isinstance(roi, S.OvalRoi):
        from rsstatchem.core.imaging.roi.mask import oval_mask
        x, y, w, h = G.int_bounds(roi)
        loops = G.trace_mask_outlines(oval_mask(w, h))
        xs, ys = loops[0][0] + x, loops[0][1] + y
    typ = S.POLYLINE
    if roi.type in (S.OVAL, S.FREEROI, S.TRACED_ROI) or getattr(roi, "spline_fit", False):
        typ = S.FREELINE
    xs = np.append(xs, xs[0]) - 0.5
    ys = np.append(ys, ys[0]) - 0.5
    return _transfer(roi, S.PolygonRoi(xs, ys, type=typ))


def specify(image, width: float, height: float, x: float, y: float, oval: bool = False,
            centered: bool = False, scaled: bool = False):
    cal = image.calibration
    if scaled and cal.scaled():
        width, height = width / cal.pixel_width, height / cal.pixel_height
        x, y = cal.raw_x(x), cal.raw_y(y)
    if centered:
        x, y = x - width / 2.0, y - height / 2.0
    cls = S.OvalRoi if oval else S.RectRoi
    return cls(x, y, width, height)


def straighten(image, roi, width: int = 20, all_slices: bool = False) -> ImageDoc:
    """Edit > Selection > Straighten... (line selections); a 32-bit (or RGB) image."""
    from rsstatchem.core.imaging import profile as Pf

    if roi is None or not roi.is_line():
        raise SelectionError("Line selection required")
    if isinstance(roi, S.LineRoi):
        xs, ys = [roi.x1, roi.x2], [roi.y1, roi.y2]
    else:
        xs, ys = G.float_polygon(roi)
    planes = range(image.stack_size) if all_slices else [image.current_index]
    out = []
    for index in planes:
        plane = image.plane_at(index)
        if plane.ndim == 3:
            chans = [Pf.straighten(plane[..., k].astype(float), xs, ys, width,
                                   spline=len(xs) > 2) for k in range(3)]
            out.append(np.clip(np.floor(np.stack(chans, -1) + 0.5), 0, 255).astype(np.uint8))
        else:
            vp = Pf.value_plane(image, False, plane=plane)
            out.append(Pf.straighten(vp, xs, ys, width, spline=len(xs) > 2).astype(np.float32))
    arr = np.stack(out)
    title = image.title.rsplit(".", 1)[0] + "-1"
    if arr.ndim == 4:
        doc = ImageDoc.from_array(arr if len(out) > 1 else arr[0],
                                  "ZYXS" if len(out) > 1 else "YXS", title)
    else:
        doc = ImageDoc.from_array(arr if len(out) > 1 else arr[0],
                                  "ZYX" if len(out) > 1 else "YX", title)
    doc.calibration.pixel_width = image.calibration.pixel_width
    doc.calibration.pixel_height = image.calibration.pixel_width
    doc.calibration.unit = image.calibration.unit
    return doc


def _rgb(color, default=(255, 255, 0)):
    if not color:
        return default
    s = str(color).lstrip("#")
    if len(s) == 8:
        s = s[2:]
    try:
        return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return default


def flatten(image, rois=None, show_labels: bool = False, default_color: str = "#ffff00",
            label_color=(255, 255, 255)) -> np.ndarray:
    """Render the displayed image with ROI outlines/fills (and labels) burned in: an RGB
    array, as Image > Overlay > Flatten / ROI Manager "Flatten" produce."""
    from rsstatchem.core.imaging.lut import render
    from rsstatchem.core.imaging.particles import draw_polygon
    from rsstatchem.core.imaging.scalebar import burn_text

    rgb = np.ascontiguousarray(render(image)).copy()
    if rois is None:
        rois = list(image.overlay) if image.overlay is not None else []
        if image.roi is not None:
            rois = rois + [image.roi]
    h, w = rgb.shape[:2]
    for k, roi in enumerate(rois):
        stroke = np.array(_rgb(roi.stroke_color or default_color), np.uint8)
        if roi.fill_color and roi.is_area():
            rgb[roi.get_mask((h, w))] = np.array(_rgb(roi.fill_color), np.uint8)
        if isinstance(roi, S.TextRoi):
            x, y, _, _ = roi.bounds()
            burn_text(rgb, roi.text, int(x), int(y + roi.font_size), int(roi.font_size),
                      tuple(int(v) for v in stroke), center=False)
            continue
        parts = roi.parts if isinstance(roi, S.CompositeRoi) else [("or", roi)]
        for _, p in parts:
            if isinstance(p, S.PointRoi):
                for px, py in zip(p.xs, p.ys):
                    xi, yi = int(px), int(py)
                    rgb[max(yi - 1, 0):yi + 2, max(xi - 1, 0):xi + 2] = stroke
                continue
            if isinstance(p, S.LineRoi):
                draw_polygon(rgb, [p.x1, p.x2], [p.y1, p.y2], stroke, closed=False)
                continue
            if isinstance(p, S.OvalRoi):
                from rsstatchem.core.imaging.roi.mask import oval_mask
                x0, y0, ow, oh = G.int_bounds(p)
                loops = G.trace_mask_outlines(oval_mask(ow, oh))
                for lx, ly in loops:
                    draw_polygon(rgb, np.minimum(lx + x0, w - 1), np.minimum(ly + y0, h - 1),
                                 stroke)
                continue
            xs, ys = G.float_polygon(p)
            if len(xs):
                draw_polygon(rgb, np.minimum(np.floor(xs), w - 1), np.minimum(np.floor(ys), h - 1),
                             stroke, closed=p.is_area())
        if show_labels:
            x, y, bw, bh = G.int_bounds(roi)
            text = roi.name if roi.name and not roi.name[:1].isdigit() else str(k + 1)
            burn_text(rgb, text, x + bw // 2, y + bh // 2 + 5, 9, label_color)
    return rgb


def roi_from_options(roi, name=None, stroke=None, width=None, fill=None, position=None):
    out = roi
    if name is not None:
        out.name = name
    if stroke is not None:
        out.stroke_color = stroke or None
    if width is not None:
        out.stroke_width = float(width)
    if fill is not None:
        out.fill_color = fill or None
    if position is not None:
        out.position = position
    return out

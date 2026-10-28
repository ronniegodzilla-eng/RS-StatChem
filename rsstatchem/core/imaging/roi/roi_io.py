"""ImageJ ROI files: single ``.roi`` files and ``RoiSet.zip`` sets (File > Open, Save As >
Selection..., ROI Manager Open/Save), via **roifile** (BSD-3) which implements the format
documented in ImageJ's RoiDecoder/RoiEncoder.

:func:`to_ijroi` / :func:`from_ijroi` convert between :mod:`.shapes` ROIs and
``roifile.ImagejRoi``: rectangles (rounded, sub-pixel), ovals, polygons, freehand, traced,
lines (width, arrow), polylines, freelines, angles, multi-points (counters, positions),
ellipses and rotated rectangles (ImageJ subtypes), text, and composite ROIs (path); names,
stroke/fill colours, stroke width, stack position (c, z, t) and properties.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from rsstatchem.core.imaging.roi import shapes as S


def _rf():
    import roifile
    return roifile


def _color_bytes(c: str | None) -> bytes:
    if not c:
        return b"\x00\x00\x00\x00"
    s = c.lstrip("#")
    if len(s) == 6:
        s = "ff" + s
    try:
        return bytes.fromhex(s[:8])
    except ValueError:
        return b"\x00\x00\x00\x00"


def _color_str(b: bytes | None) -> str | None:
    if not b or b == b"\x00\x00\x00\x00":
        return None
    a, r, g, bl = b[0], b[1], b[2], b[3]
    return f"#{r:02x}{g:02x}{bl:02x}" if a == 255 else f"#{a:02x}{r:02x}{g:02x}{bl:02x}"


def _float_param(value: float):
    b = struct.pack(">f", float(value))
    return b[0], b[1], struct.unpack(">h", b[2:4])[0]


def _read_float_param(ij) -> float:
    b = bytes([ij.arrow_style_or_aspect_ratio & 0xFF, ij.arrow_head_size & 0xFF]) + \
        struct.pack(">h", ij.rounded_rect_arc_size)
    return struct.unpack(">f", b)[0]


def _set_bounds(ij, x0, y0, x1, y1):
    ij.left, ij.top = int(np.floor(x0)), int(np.floor(y0))
    ij.right, ij.bottom = int(np.ceil(x1)), int(np.ceil(y1))


def _set_points(ij, xs, ys, subpixel: bool | None = None):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    rf = _rf()
    if subpixel is None:
        subpixel = bool(np.any(xs != np.round(xs)) or np.any(ys != np.round(ys)))
    if len(xs):
        _set_bounds(ij, xs.min(), ys.min(), xs.max(), ys.max())
    ij.n_coordinates = len(xs)
    ij.integer_coordinates = np.column_stack(
        [np.round(xs) - ij.left, np.round(ys) - ij.top]).astype(np.int32)
    if subpixel:
        ij.options |= rf.ROI_OPTIONS.SUB_PIXEL_RESOLUTION
        ij.subpixel_coordinates = np.column_stack([xs, ys]).astype(np.float32)


def to_ijroi(roi):
    """A ``roifile.ImagejRoi`` for ``roi``."""
    rf = _rf()
    T = rf.ROI_TYPE
    ij = rf.ImagejRoi()
    ij.version = 228
    if isinstance(roi, S.CompositeRoi):
        from rsstatchem.core.imaging.roi.geometry import trace_mask_outlines
        from rsstatchem.core.imaging.roi.geometry import int_bounds
        x, y, w, h = int_bounds(roi)
        m = roi.get_mask((y + h + 1, x + w + 1)) if (x >= 0 and y >= 0) else None
        path = []
        if m is not None:
            for lx, ly in trace_mask_outlines(m):
                for i, (px, py) in enumerate(zip(lx, ly)):
                    path += [0.0 if i == 0 else 1.0, float(px), float(py)]
                path.append(4.0)
        ij.roitype = T.RECT
        ij.shape_roi_size = len(path)
        ij.multi_coordinates = np.asarray(path, dtype=np.float32)
        _set_bounds(ij, x, y, x + w, y + h)
    elif isinstance(roi, S.TextRoi):
        ij.roitype = T.RECT
        ij.subtype = rf.ROI_SUBTYPE.TEXT
        x, y, w, h = roi.bounds()
        _set_bounds(ij, x, y, x + w, y + h)
        ij.text = roi.text
        ij.text_name = roi.font
        ij.text_size = int(roi.font_size)
        ij.text_angle = float(roi.angle)
    elif isinstance(roi, S.RectRoi):
        ij.roitype = T.RECT
        _set_bounds(ij, roi.x, roi.y, roi.x + roi.width, roi.y + roi.height)
        if any(float(v) != int(v) for v in (roi.x, roi.y, roi.width, roi.height)):
            ij.options |= rf.ROI_OPTIONS.SUB_PIXEL_RESOLUTION
            ij.xd, ij.yd, ij.widthd, ij.heightd = (float(roi.x), float(roi.y),
                                                   float(roi.width), float(roi.height))
        ij.rounded_rect_arc_size = int(roi.corner_diameter or 0)
    elif isinstance(roi, S.OvalRoi):
        ij.roitype = T.OVAL
        _set_bounds(ij, roi.x, roi.y, roi.x + roi.width, roi.y + roi.height)
        if any(float(v) != int(v) for v in (roi.x, roi.y, roi.width, roi.height)):
            ij.options |= rf.ROI_OPTIONS.SUB_PIXEL_RESOLUTION
            ij.xd, ij.yd, ij.widthd, ij.heightd = (float(roi.x), float(roi.y),
                                                   float(roi.width), float(roi.height))
    elif isinstance(roi, S.LineRoi):
        ij.roitype = T.LINE
        ij.x1, ij.y1, ij.x2, ij.y2 = (float(roi.x1), float(roi.y1), float(roi.x2),
                                      float(roi.y2))
        _set_bounds(ij, min(roi.x1, roi.x2), min(roi.y1, roi.y2), max(roi.x1, roi.x2),
                    max(roi.y1, roi.y2))
        if roi.arrow:
            ij.subtype = rf.ROI_SUBTYPE.ARROW
        if roi.width and roi.width > 1 and not roi.stroke_width:
            ij.stroke_width = int(round(roi.width))
            ij.float_stroke_width = float(roi.width)
    elif isinstance(roi, S.EllipseRoi):
        ij.roitype = T.FREEHAND
        ij.subtype = rf.ROI_SUBTYPE.ELLIPSE
        ij.x1, ij.y1, ij.x2, ij.y2 = (float(roi.x1), float(roi.y1), float(roi.x2),
                                      float(roi.y2))
        a, b, c = _float_param(roi.aspect_ratio)
        ij.arrow_style_or_aspect_ratio, ij.arrow_head_size, ij.rounded_rect_arc_size = a, b, c
        xs, ys = roi.polygon()
        _set_points(ij, xs, ys, subpixel=True)
    elif isinstance(roi, S.RotatedRectRoi):
        ij.roitype = T.FREEHAND
        ij.subtype = rf.ROI_SUBTYPE.ROTATED_RECT
        ij.x1, ij.y1, ij.x2, ij.y2 = (float(roi.x1), float(roi.y1), float(roi.x2),
                                      float(roi.y2))
        a, b, c = _float_param(roi.rect_width)
        ij.arrow_style_or_aspect_ratio, ij.arrow_head_size, ij.rounded_rect_arc_size = a, b, c
        xs, ys = roi.polygon()
        _set_points(ij, xs, ys, subpixel=True)
    elif isinstance(roi, S.PointRoi):
        ij.roitype = T.POINT
        _set_points(ij, roi.xs, roi.ys)
        if roi.counters is not None or roi.positions is not None:
            n = len(roi.xs)
            ij.counters = np.asarray(roi.counters if roi.counters is not None
                                     else np.zeros(n), dtype=np.uint8)
            ij.counter_positions = np.asarray(roi.positions if roi.positions is not None
                                              else np.zeros(n), dtype=np.uint32)
    elif isinstance(roi, S.PolygonRoi):
        ij.roitype = {S.POLYGON: T.POLYGON, S.FREEROI: T.FREEHAND, S.TRACED_ROI: T.TRACED,
                      S.POLYLINE: T.POLYLINE, S.FREELINE: T.FREELINE,
                      S.ANGLE: T.ANGLE}[roi.type]
        _set_points(ij, roi.xs, roi.ys)
        if roi.spline_fit:
            ij.options |= rf.ROI_OPTIONS.SPLINE_FIT
    else:
        raise ValueError(f"cannot save {type(roi).__name__}")
    ij.name = roi.name or ""
    ij.stroke_color = _color_bytes(roi.stroke_color)
    ij.fill_color = _color_bytes(roi.fill_color)
    if roi.stroke_width:
        ij.stroke_width = int(round(roi.stroke_width))
        ij.float_stroke_width = float(roi.stroke_width)
    c, z, t = roi.position
    if c or t:
        ij.c_position, ij.z_position, ij.t_position = int(c), int(z), int(t)
    elif z:
        ij.position = int(z)
    if roi.properties:
        ij.props = "\n".join(f"{k}: {v}" for k, v in roi.properties.items())
    return ij


def from_ijroi(ij):
    """A :mod:`.shapes` ROI from a ``roifile.ImagejRoi``."""
    rf = _rf()
    T = rf.ROI_TYPE
    rt, sub = ij.roitype, ij.subtype
    sub_px = bool(ij.options & rf.ROI_OPTIONS.SUB_PIXEL_RESOLUTION)

    def pts():
        c = ij.coordinates()
        c = np.asarray(c, dtype=float)
        if c.ndim != 2 or len(c) == 0:
            return np.zeros(0), np.zeros(0)
        return c[:, 0], c[:, 1]

    if ij.multi_coordinates is not None and ij.shape_roi_size:
        parts = []
        for k, arr in enumerate(ij.path2coords(ij.multi_coordinates)):
            arr = np.asarray(arr, dtype=float)
            if len(arr) > 1 and np.all(arr[0] == arr[-1]):
                arr = arr[:-1]
            if len(arr) < 3:
                continue
            parts.append(("or" if k == 0 else "xor", S.PolygonRoi(arr[:, 0], arr[:, 1])))
        roi = S.CompositeRoi(parts)
    elif sub == rf.ROI_SUBTYPE.TEXT:
        roi = S.TextRoi(float(ij.left), float(ij.top), ij.text or "",
                        font=ij.text_name or "SansSerif", font_size=float(ij.text_size or 18),
                        angle=float(ij.text_angle or 0))
    elif rt == T.RECT:
        if sub_px and ij.subpixelrect:
            roi = S.RectRoi(ij.xd, ij.yd, ij.widthd, ij.heightd)
        else:
            roi = S.RectRoi(ij.left, ij.top, ij.right - ij.left, ij.bottom - ij.top)
        roi.corner_diameter = float(ij.rounded_rect_arc_size or 0)
    elif rt == T.OVAL:
        if sub_px and ij.subpixelrect:
            roi = S.OvalRoi(ij.xd, ij.yd, ij.widthd, ij.heightd)
        else:
            roi = S.OvalRoi(ij.left, ij.top, ij.right - ij.left, ij.bottom - ij.top)
    elif rt == T.LINE:
        roi = S.LineRoi(ij.x1, ij.y1, ij.x2, ij.y2, arrow=sub == rf.ROI_SUBTYPE.ARROW)
        if ij.float_stroke_width > 1 or ij.stroke_width > 1:
            roi.width = float(ij.float_stroke_width or ij.stroke_width)
    elif rt == T.FREEHAND and sub == rf.ROI_SUBTYPE.ELLIPSE:
        roi = S.EllipseRoi(ij.x1, ij.y1, ij.x2, ij.y2, _read_float_param(ij))
    elif rt == T.FREEHAND and sub == rf.ROI_SUBTYPE.ROTATED_RECT:
        roi = S.RotatedRectRoi(ij.x1, ij.y1, ij.x2, ij.y2, _read_float_param(ij))
    elif rt == T.POINT:
        xs, ys = pts()
        roi = S.PointRoi(xs, ys)
        if ij.counters is not None:
            roi.counters = np.asarray(ij.counters, dtype=int)
        if ij.counter_positions is not None:
            roi.positions = np.asarray(ij.counter_positions, dtype=int)
    else:
        xs, ys = pts()
        typ = {T.POLYGON: S.POLYGON, T.FREEHAND: S.FREEROI, T.TRACED: S.TRACED_ROI,
               T.POLYLINE: S.POLYLINE, T.FREELINE: S.FREELINE, T.ANGLE: S.ANGLE}.get(rt)
        if typ is None:
            raise ValueError(f"unsupported ROI type {rt!r}")
        if len(xs) > 1 and typ in (S.POLYGON, S.FREEROI, S.TRACED_ROI) and \
                xs[0] == xs[-1] and ys[0] == ys[-1] and len(xs) > 3:
            xs, ys = xs[:-1], ys[:-1]
        roi = S.PolygonRoi(xs, ys, type=typ)
        roi.spline_fit = bool(ij.options & rf.ROI_OPTIONS.SPLINE_FIT)
    roi.name = ij.name or ""
    roi.stroke_color = _color_str(ij.stroke_color)
    roi.fill_color = _color_str(ij.fill_color)
    sw = float(ij.float_stroke_width or ij.stroke_width or 0)
    if not isinstance(roi, S.LineRoi):
        roi.stroke_width = sw
    if ij.c_position or ij.t_position or ij.z_position:
        roi.position = (int(ij.c_position), int(ij.z_position), int(ij.t_position))
    elif ij.position:
        roi.position = (0, int(ij.position), 0)
    if ij.props:
        for line in str(ij.props).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                roi.properties[k.strip()] = v.strip()
    return roi


def read_rois(path) -> list:
    """All ROIs of a ``.roi`` file or a RoiSet ``.zip``."""
    out = _rf().roiread(str(path))
    if not isinstance(out, list):
        out = [out]
    return [from_ijroi(r) for r in out]


def read_roi(path):
    rois = read_rois(path)
    if not rois:
        raise ValueError(f"no ROI in {path}")
    return rois[0]


def roi_file_name(roi, index: int = 0) -> str:
    name = roi.name or ""
    if not name:
        from rsstatchem.core.imaging.roi.geometry import int_bounds
        x, y, w, h = int_bounds(roi)
        name = f"{y + h // 2:04d}-{x + w // 2:04d}"
    return name if name.endswith(".roi") else name + ".roi"


def write_roi(path, roi) -> None:
    """Save one ROI as a ``.roi`` file (Save As > Selection...)."""
    to_ijroi(roi).tofile(str(path))


def write_rois(path, rois) -> None:
    """Save ROIs as a RoiSet ``.zip`` (or a single ``.roi`` for one ROI and a .roi path)."""
    path = Path(path)
    if path.suffix.lower() == ".roi":
        if len(rois) != 1:
            raise ValueError("a .roi file holds one selection")
        write_roi(path, rois[0])
        return
    if path.exists():
        path.unlink()
    names, seen = [], set()
    for i, r in enumerate(rois):
        n = roi_file_name(r, i)
        base = n[:-4]
        k = 1
        while n in seen:
            n = f"{base}-{k}.roi"
            k += 1
        seen.add(n)
        names.append(n[:-4])
    _rf().roiwrite(str(path), [to_ijroi(r) for r in rois], name=names, mode="w")

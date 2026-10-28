"""ImageJ ``.roi`` encoding/decoding between ImageJ's binary ROI format and the ROI data classes
of :mod:`rsstatchem.core.imaging.roi` (used for .roi files, RoiSet.zip, the ROI and overlay
blobs inside ImageJ TIFFs and ``.rsij`` projects).

The byte layout is ImageJ's (``ij/io/RoiDecoder.java`` / ``RoiEncoder.java``, public domain,
NIH); the low-level parsing is done by roifile (BSD-3) and the ImageJ-specific fields roifile
does not interpret (the float parameter at offset 52 used by ellipses and rotated rectangles,
ImageJ's path format for composite "shape" ROIs, counters, positions) are handled here.

WP-C owns ``core/imaging/roi/roi_io.py`` for the ROI Manager; it can use :func:`decode_roi`,
:func:`encode_roi`, :func:`read_roi_zip` and :func:`write_roi_zip` from here.
"""
from __future__ import annotations

import io
import math
import struct
import zipfile
from pathlib import Path

import numpy as np

from rsstatchem.core.imaging.roi import shapes as S

FLOAT_PARAM = 52
VERSION = 228


# ---------------------------------------------------------------------------------- colours
def _argb_to_hex(b: bytes) -> str | None:
    if b == b"\0\0\0\0" or len(b) != 4:
        return None
    a, r, g, bb = b[0], b[1], b[2], b[3]
    if a == 255:
        return f"#{r:02x}{g:02x}{bb:02x}"
    return f"#{a:02x}{r:02x}{g:02x}{bb:02x}"


_NAMED = {"red": "#ff0000", "green": "#00ff00", "blue": "#0000ff", "yellow": "#ffff00",
          "cyan": "#00ffff", "magenta": "#ff00ff", "white": "#ffffff", "black": "#000000",
          "orange": "#ffc800", "pink": "#ffafaf", "gray": "#808080", "grey": "#808080",
          "lightgray": "#c0c0c0", "darkgray": "#404040"}


def color_to_argb(color: str | None) -> bytes:
    """"#rrggbb" / "#aarrggbb" / ImageJ colour name -> 4 ARGB bytes (zeros = none)."""
    if not color:
        return b"\0\0\0\0"
    c = _NAMED.get(color.lower(), color).lstrip("#")
    if len(c) == 6:
        c = "ff" + c
    if len(c) != 8:
        return b"\0\0\0\0"
    return bytes.fromhex(c)


# ---------------------------------------------------------------------------------- decode
def decode_roi(data: bytes, name: str | None = None) -> S.Roi:
    """ImageJ ``.roi`` bytes -> Roi (RectRoi, OvalRoi, PolygonRoi, LineRoi, PointRoi,
    RotatedRectRoi, EllipseRoi, TextRoi or CompositeRoi)."""
    import roifile

    data = bytes(data)
    r = roifile.ImagejRoi.frombytes(data)
    T, ST = roifile.ROI_TYPE, roifile.ROI_SUBTYPE
    opts = int(r.options)
    subpixel = bool(opts & 128) and r.version >= 222
    left, top = r.left, r.top
    width, height = r.right - r.left, r.bottom - r.top
    param = struct.unpack_from(">f", data, FLOAT_PARAM)[0]
    roi: S.Roi
    if r.shape_roi_size > 0 and r.multi_coordinates is not None:
        polys = _path_to_polygons(r.multi_coordinates)
        parts = [("xor" if i else "or", S.PolygonRoi(p[:, 0], p[:, 1], type=S.POLYGON))
                 for i, p in enumerate(polys)]
        roi = S.CompositeRoi(parts=parts)
    elif r.roitype == T.RECT:
        if subpixel and r.version >= 223:
            x, y, w, h = r.xd, r.yd, r.widthd, r.heightd
        else:
            x, y, w, h = left, top, width, height
        if r.subtype == ST.TEXT:
            roi = S.TextRoi(x=float(x), y=float(y), text=r.text, font=r.text_name or "SansSerif",
                            font_size=float(r.text_size or 18), angle=float(r.text_angle))
            roi.properties["_text_style"] = str(r.text_style)
            roi.properties["_text_bounds"] = f"{w} {h}"
        else:
            roi = S.RectRoi(float(x), float(y), float(w), float(h),
                            corner_diameter=float(r.rounded_rect_arc_size))
    elif r.roitype == T.OVAL:
        if subpixel and r.version >= 223:
            roi = S.OvalRoi(float(r.xd), float(r.yd), float(r.widthd), float(r.heightd))
        else:
            roi = S.OvalRoi(float(left), float(top), float(width), float(height))
    elif r.roitype == T.LINE:
        roi = S.LineRoi(float(r.x1), float(r.y1), float(r.x2), float(r.y2),
                        arrow=r.subtype == ST.ARROW)
        if roi.arrow:
            roi.properties["_arrow_style"] = str(r.arrow_style_or_aspect_ratio)
            roi.properties["_arrow_head"] = str(r.arrow_head_size)
            if opts & 2:
                roi.properties["_arrow_double"] = "1"
    elif r.roitype == T.FREEHAND and r.subtype == ST.ELLIPSE:
        roi = S.EllipseRoi(float(r.x1), float(r.y1), float(r.x2), float(r.y2),
                           aspect_ratio=float(param))
    elif r.roitype in (T.FREEHAND, T.POLYGON) and r.subtype == ST.ROTATED_RECT:
        x1, y1, x2, y2 = struct.unpack_from(">ffff", data, 18)
        roi = S.RotatedRectRoi(float(x1), float(y1), float(x2), float(y2),
                               rect_width=float(param))
    elif r.roitype in (T.POLYGON, T.FREEHAND, T.TRACED, T.POLYLINE, T.FREELINE, T.ANGLE,
                       T.POINT):
        n = r.n_coordinates
        if subpixel and r.subpixel_coordinates is not None:
            xs = r.subpixel_coordinates[:, 0].astype(float)
            ys = r.subpixel_coordinates[:, 1].astype(float)
        elif r.integer_coordinates is not None:
            xs = (np.maximum(r.integer_coordinates[:, 0], 0) + left).astype(float)
            ys = (np.maximum(r.integer_coordinates[:, 1], 0) + top).astype(float)
        else:
            xs = ys = np.zeros(0)
        xs, ys = xs[:n], ys[:n]
        if r.roitype == T.POINT:
            counters = positions = None
            if r.counters is not None and np.any(r.counters):
                counters = np.asarray(r.counters, dtype=int)
            if r.counter_positions is not None and np.any(r.counter_positions):
                positions = np.asarray(r.counter_positions, dtype=int)
            roi = S.PointRoi(xs, ys, counters=counters, positions=positions)
            if r.version >= 226:
                roi.properties["_point_type"] = str(r.arrow_style_or_aspect_ratio)
                roi.properties["_point_size"] = str(r.stroke_width)
        else:
            kind = {T.POLYGON: S.POLYGON, T.FREEHAND: S.FREEROI, T.TRACED: S.TRACED_ROI,
                    T.POLYLINE: S.POLYLINE, T.FREELINE: S.FREELINE, T.ANGLE: S.ANGLE}[r.roitype]
            roi = S.PolygonRoi(xs, ys, type=kind, spline_fit=bool(opts & 1))
    else:
        raise ValueError(f"Unsupported ROI type: {r.roitype!r}")

    roi.name = r.name or (name or "")
    if r.version >= 218 and r.roitype != T.POINT:
        width_ = float(r.stroke_width)
        if r.float_stroke_width and r.float_stroke_width > 0:
            width_ = float(r.float_stroke_width)
        roi.stroke_width = width_
        if isinstance(roi, S.LineRoi):
            roi.width = max(1.0, width_)
    roi.stroke_color = _argb_to_hex(bytes(r.stroke_color))
    roi.fill_color = _argb_to_hex(bytes(r.fill_color))
    c, z, t = r.c_position, r.z_position, r.t_position
    if c or z or t:
        roi.position = (int(c), int(z), int(t))
    elif r.position:
        roi.position = (0, int(r.position), 0)
    if r.props:
        roi.properties.update(_parse_props(r.props))
    if r.group:
        roi.properties["_group"] = str(r.group)
    roi.properties["_overlay_options"] = str(opts & (8 | 16 | 32 | 64))
    if r.overlay_font_size:
        roi.properties["_overlay_font_size"] = str(r.overlay_font_size)
    return roi


def _parse_props(text: str) -> dict:
    out = {}
    for line in text.split("\n"):
        if ": " in line:
            k, v = line.split(": ", 1)
            out[k] = v
    return out


def _path_to_polygons(path) -> list[np.ndarray]:
    """ImageJ ShapeRoi path (SEG_MOVETO=0 x y, SEG_LINETO=1 x y, SEG_QUADTO=2, SEG_CUBICTO=3,
    SEG_CLOSE=4) -> list of closed polygons (curves are flattened to their end points)."""
    vals = [float(v) for v in np.asarray(path).ravel()]
    polys, cur = [], []
    i = 0
    while i < len(vals):
        op = int(vals[i])
        if op == 0:
            if len(cur) >= 3:
                polys.append(np.array(cur))
            cur = [(vals[i + 1], vals[i + 2])]
            i += 3
        elif op == 1:
            cur.append((vals[i + 1], vals[i + 2]))
            i += 3
        elif op == 2:
            cur.append((vals[i + 3], vals[i + 4]))
            i += 5
        elif op == 3:
            cur.append((vals[i + 5], vals[i + 6]))
            i += 7
        elif op == 4:
            if len(cur) >= 3:
                polys.append(np.array(cur))
            cur = []
            i += 1
        else:
            break
    if len(cur) >= 3:
        polys.append(np.array(cur))
    return polys


# ---------------------------------------------------------------------------------- encode
def encode_roi(roi: S.Roi, overlay=None) -> bytes:
    """Roi -> ImageJ ``.roi`` bytes (``overlay``: the Overlay whose label options are stored in
    its first ROI, like ImageJ's ``setPrototypeOverlay``)."""
    import roifile

    T, ST, OPT = roifile.ROI_TYPE, roifile.ROI_SUBTYPE, roifile.ROI_OPTIONS
    r = roifile.ImagejRoi()
    r.version = VERSION
    options = 0
    param = None
    if isinstance(roi, S.CompositeRoi):
        path = _composite_path(roi)
        r.roitype = T.RECT
        r.multi_coordinates = np.asarray(path, dtype=np.float32)
        r.shape_roi_size = len(path)
        x, y, w, h = roi.bounds()
        _set_bounds(r, x, y, w, h)
    elif isinstance(roi, S.TextRoi):
        r.roitype, r.subtype = T.RECT, ST.TEXT
        xs, ys = roi.polygon()
        bw = roi.properties.get("_text_bounds")
        w, h = (float(v) for v in bw.split()) if bw else (xs.max() - xs.min(), ys.max() - ys.min())
        _set_bounds(r, roi.x, roi.y, w, h)
        r.text, r.text_name, r.text_size = roi.text, roi.font, int(round(roi.font_size))
        r.text_style = int(roi.properties.get("_text_style", 0))
        r.text_angle = float(roi.angle)
    elif isinstance(roi, (S.RectRoi, S.OvalRoi)):
        r.roitype = T.RECT if isinstance(roi, S.RectRoi) else T.OVAL
        _set_bounds(r, roi.x, roi.y, roi.width, roi.height)
        options |= 128       # ImageJ 1.54 always stores the double-precision rectangle
        r.xd, r.yd, r.widthd, r.heightd = roi.x, roi.y, roi.width, roi.height
        if isinstance(roi, S.RectRoi) and roi.corner_diameter:
            r.rounded_rect_arc_size = int(round(roi.corner_diameter))
    elif isinstance(roi, S.LineRoi):
        r.roitype = T.LINE
        r.x1, r.y1, r.x2, r.y2 = roi.x1, roi.y1, roi.x2, roi.y2
        _fill_coordinates(r, [roi.x1, roi.x2], [roi.y1, roi.y2], line=True)
        r.n_coordinates = 0
        r.integer_coordinates = r.subpixel_coordinates = None
        if roi.arrow:
            r.subtype = ST.ARROW
            r.arrow_style_or_aspect_ratio = int(roi.properties.get("_arrow_style", 0))
            r.arrow_head_size = int(roi.properties.get("_arrow_head", 10))
            if roi.properties.get("_arrow_double"):
                options |= 2
    elif isinstance(roi, (S.EllipseRoi, S.RotatedRectRoi)):
        r.roitype = T.FREEHAND if isinstance(roi, S.EllipseRoi) else T.POLYGON
        r.subtype = ST.ELLIPSE if isinstance(roi, S.EllipseRoi) else ST.ROTATED_RECT
        r.x1, r.y1, r.x2, r.y2 = roi.x1, roi.y1, roi.x2, roi.y2
        param = roi.aspect_ratio if isinstance(roi, S.EllipseRoi) else roi.rect_width
        xs, ys = roi.polygon()
        _fill_coordinates(r, xs, ys, floor_max=isinstance(roi, S.EllipseRoi))
        options |= 128
    elif isinstance(roi, (S.PolygonRoi, S.PointRoi)):
        xs, ys = roi.polygon()
        if isinstance(roi, S.PointRoi):
            r.roitype = T.POINT
            r.arrow_style_or_aspect_ratio = int(roi.properties.get("_point_type", 0))
        else:
            r.roitype = {S.POLYGON: T.POLYGON, S.FREEROI: T.FREEHAND, S.TRACED_ROI: T.TRACED,
                         S.POLYLINE: T.POLYLINE, S.FREELINE: T.FREELINE,
                         S.ANGLE: T.ANGLE}[roi.type]
            if roi.spline_fit:
                options |= 1
        _fill_coordinates(r, xs, ys, line=isinstance(roi, S.PointRoi) or not roi.is_area())
        integer = np.all(np.equal(np.mod(xs, 1), 0)) and np.all(np.equal(np.mod(ys, 1), 0))
        if not integer or isinstance(roi, S.PointRoi):
            options |= 128
        else:
            r.subpixel_coordinates = None
        if isinstance(roi, S.PointRoi):
            if True:     # ImageJ 1.54 always writes the counters block of point ROIs
                n = len(xs)
                counters = np.zeros(n, np.uint8) if roi.counters is None else \
                    np.asarray(roi.counters, dtype=np.uint8)
                positions = np.zeros(n, np.uint32) if roi.positions is None else \
                    np.asarray(roi.positions, dtype=np.uint32)
                r.counters, r.counter_positions = counters, positions
    else:
        raise ValueError(f"cannot encode {type(roi).__name__}")

    r.name = roi.name or ""
    sw = float(roi.stroke_width or 0)
    if isinstance(roi, S.LineRoi) and roi.width > 1 and not sw:
        sw = float(roi.width)
    if isinstance(roi, S.PointRoi):
        r.stroke_width = int(roi.properties.get("_point_size", 0))
    else:
        r.stroke_width = int(sw)
        if sw:
            r.float_stroke_width = sw
            options |= 8192          # SCALE_STROKE_WIDTH
    r.stroke_color = color_to_argb(roi.stroke_color)
    r.fill_color = color_to_argb(roi.fill_color)
    c, z, t = (tuple(roi.position) + (0, 0, 0))[:3]
    if c or t:
        r.c_position, r.z_position, r.t_position = int(c), int(z), int(t)
    elif z:
        r.position = int(z)
    props = {k: v for k, v in roi.properties.items() if not str(k).startswith("_")}
    if props:
        r.props = "\n".join(f"{k}: {v}" for k, v in props.items()) + "\n"
    if roi.properties.get("_group"):
        r.group = int(roi.properties["_group"])
    if overlay is not None:
        if overlay.show_labels:
            options |= 8
        if overlay.draw_names:
            options |= 16
        if overlay.draw_backgrounds:
            options |= 32
        if overlay.show_labels or overlay.draw_names:
            r.overlay_font_size = int(overlay.label_font_size or 0)
        if overlay.label_color:
            r.overlay_label_color = color_to_argb(overlay.label_color)
    r.options = OPT(options)
    data = bytearray(r.tobytes())
    if param is not None:
        data[FLOAT_PARAM:FLOAT_PARAM + 4] = struct.pack(">f", float(param))
        data[18:34] = struct.pack(">ffff", roi.x1, roi.y1, roi.x2, roi.y2)
    return bytes(data)


def _set_bounds(r, x, y, w, h, ceil: bool = False) -> None:
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    if ceil:
        x1, y1 = int(math.ceil(x + w)), int(math.ceil(y + h))
    else:
        x1, y1 = x0 + int(round(w)), y0 + int(round(h))
    r.left, r.top, r.right, r.bottom = x0, y0, x1, y1


def _fill_coordinates(r, xs, ys, line: bool = False, floor_max: bool = False) -> None:
    """Coordinates relative to the bounds (ImageJ Roi.getBounds: floor of the minimum,
    width = int(max - left), one more for lines and points)."""
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    n = len(xs)
    left = int(math.floor(xs.min())) if n else 0
    top = int(math.floor(ys.min())) if n else 0
    r.left, r.top = left, top
    if line:
        r.right = left + int(xs.max() - left) + 1 if n else 0
        r.bottom = top + int(ys.max() - top) + 1 if n else 0
    else:
        f = int if floor_max else math.ceil
        r.right = left + int(f(xs.max() - left)) if n else 0
        r.bottom = top + int(f(ys.max() - top)) if n else 0
    r.n_coordinates = n
    r.integer_coordinates = np.stack([np.floor(xs) - left, np.floor(ys) - top], axis=1
                                     ).astype(np.int32)
    r.subpixel_coordinates = np.stack([xs, ys], axis=1).astype(np.float32)


def _composite_path(roi: S.CompositeRoi) -> list[float]:
    """The area of a composite ROI as ImageJ path segments (outlines of its mask on the
    pixel grid; parts that are all "or"-ed polygons/rectangles keep their exact vertices)."""
    simple = all(op == "or" or i == 0 for i, (op, _) in enumerate(roi.parts)) and \
        all(isinstance(p, (S.RectRoi, S.PolygonRoi)) and p.is_area() for _, p in roi.parts)
    path: list[float] = []
    if simple and len(roi.parts) == 1:
        loops = [np.column_stack(roi.parts[0][1].polygon())]
    else:
        x, y, w, h = roi.bounds()
        x0, y0 = int(math.floor(x)), int(math.floor(y))
        W, H = int(math.ceil(x + w)) - x0 + 2, int(math.ceil(y + h)) - y0 + 2
        moved = roi.copy()
        moved.translate(-x0 + 1, -y0 + 1)
        mask = moved.get_mask((H, W))
        loops = [lp + [x0 - 1, y0 - 1] for lp in mask_outlines(mask)]
    for lp in loops:
        path += [0, float(lp[0][0]), float(lp[0][1])]
        for px, py in lp[1:]:
            path += [1, float(px), float(py)]
        path.append(4)
    return path


def mask_outlines(mask: np.ndarray) -> list[np.ndarray]:
    """Closed outlines (pixel-corner vertices, collinear points removed) of every boundary of a
    boolean mask: outer boundaries and holes, suitable for an even-odd filled path."""
    m = np.pad(np.asarray(mask, bool), 1)
    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    ys, xs = np.nonzero(m[1:-1, 1:-1])
    xs, ys = xs + 1, ys + 1
    for x, y in zip(xs.tolist(), ys.tolist()):
        # directed edges with the inside on the right (clockwise in image coordinates)
        if not m[y - 1, x]:
            edges.setdefault((x, y), []).append((x + 1, y))
        if not m[y, x + 1]:
            edges.setdefault((x + 1, y), []).append((x + 1, y + 1))
        if not m[y + 1, x]:
            edges.setdefault((x + 1, y + 1), []).append((x, y + 1))
        if not m[y, x - 1]:
            edges.setdefault((x, y + 1), []).append((x, y))
    loops = []
    while edges:
        start = next(iter(edges))
        loop = [start]
        cur = start
        prev_dir = None
        while True:
            nxts = edges.get(cur)
            if not nxts:
                break
            if len(nxts) > 1 and prev_dir is not None:
                # at a diagonal touch keep turning right (8-connected inside)
                def turn(n, c=cur, d=prev_dir):
                    nd = (n[0] - c[0], n[1] - c[1])
                    return -(d[0] * nd[1] - d[1] * nd[0])
                nxts.sort(key=turn)
            nxt = nxts.pop(0)
            if not nxts:
                del edges[cur]
            prev_dir = (nxt[0] - cur[0], nxt[1] - cur[1])
            cur = nxt
            if cur == start:
                break
            loop.append(cur)
        pts = np.array(loop, dtype=float) - 1.0
        if len(pts) >= 3:
            keep = []
            n = len(pts)
            for i in range(n):
                a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
                if (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) != 0:
                    keep.append(b)
            loops.append(np.array(keep))
    return loops


# ---------------------------------------------------------------------------------- files
def read_roi(path) -> S.Roi:
    path = Path(path)
    return decode_roi(path.read_bytes(), name=path.stem)


def write_roi(roi: S.Roi, path) -> None:
    Path(path).write_bytes(encode_roi(roi))


def read_roi_zip(path_or_bytes) -> list[S.Roi]:
    """All ROIs of an ImageJ RoiSet.zip (ROI Manager format), in archive order."""
    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else \
        path_or_bytes
    rois = []
    with zipfile.ZipFile(src) as zf:
        for info in zf.infolist():
            if info.filename.lower().endswith(".roi"):
                rois.append(decode_roi(zf.read(info), name=Path(info.filename).stem))
    return rois


def roi_entry_name(roi: S.Roi, index: int, used: set) -> str:
    """ROI Manager entry name: the ROI name or ImageJ's "yyyy-xxxx" label."""
    base = roi.name
    if not base:
        x, y, w, h = roi.bounds()
        base = f"{int(y + h / 2):04d}-{int(x + w / 2):04d}"
    name = base
    k = 1
    while name in used:
        name = f"{base}-{k}"
        k += 1
    used.add(name)
    return name


def write_roi_zip(rois, path_or_buffer) -> None:
    used: set = set()
    with zipfile.ZipFile(path_or_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, roi in enumerate(rois):
            zf.writestr(roi_entry_name(roi, i, used) + ".roi", encode_roi(roi))


def is_roi_zip(path) -> bool:
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
    except (zipfile.BadZipFile, OSError):
        return False
    return bool(names) and all(n.lower().endswith(".roi") for n in names if not n.endswith("/"))

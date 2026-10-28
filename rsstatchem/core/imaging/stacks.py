"""Image > Stacks and Image > Hyperstacks operations on :class:`ImageDoc`.

Ported from ImageJ 1.54 ij/plugin/{ZProjector,GroupedZProjector,Slicer,MontageMaker,StackMaker,
StackReducer,StackReverser,SubstackMaker,StackCombiner,Concatenator,StackInserter,ImagesToStack,
StackEditor,HyperStackConverter,HyperStackReducer,SubHyperstackMaker,Stack_Statistics}.java and
ij/process/StackStatistics.java (public domain, NIH; W. Rasband, P. Kelly et al.). Projection
arithmetic follows ZProjector exactly (float32 sums, n-1 standard deviation, float median of
``Arrays.sort``; RGB projections per channel with SD/Sum rescaled to 255), verified against the
ImageJ goldens in ``tests/data_imagej/golden/image_stacks``.
"""
from __future__ import annotations

import math

import numpy as np

from rsstatchem.core.imaging.convert import float_to_byte, java_int
from rsstatchem.core.imaging.image import (COLOR_256, COLOR_RGB, GRAY8, GRAY16, GRAY32, ImageDoc)
from rsstatchem.core.imaging.lut import LUT

PROJECTIONS = ("Average Intensity", "Max Intensity", "Min Intensity", "Sum Slices",
               "Standard Deviation", "Median")
PREFIX = {"Average Intensity": "AVG_", "Max Intensity": "MAX_", "Min Intensity": "MIN_",
          "Sum Slices": "SUM_", "Standard Deviation": "STD_", "Median": "MED_"}
ORDERS = ("xyczt(default)", "xyctz", "xyzct", "xyztc", "xytcz", "xytzc")


class StackError(ValueError):
    """User error (shown as an ImageJ message)."""


def _flat(doc: ImageDoc) -> np.ndarray:
    """All planes in ImageJ stack order: (N, Y, X[, 3])."""
    p = doc.pixels
    t, z, c = p.shape[:3]
    return p.reshape((t * z * c,) + p.shape[3:])


def from_planes(planes, type_: int, title: str, dims=None, like: ImageDoc | None = None,
                labels=None) -> ImageDoc:
    """New ImageDoc from planes in stack order with (c, z, t) ``dims``."""
    planes = np.asarray(planes)
    n = planes.shape[0]
    c, z, t = dims or (1, n, 1)
    arr = planes.reshape((t, z, c) + planes.shape[1:])
    doc = ImageDoc(np.ascontiguousarray(arr), type_, title)
    if like is not None:
        doc.calibration = like.calibration.copy()
        if type_ != COLOR_RGB and like.type != COLOR_RGB:
            if len(like.luts) == doc.n_channels:
                doc.luts = [lut.copy() for lut in like.luts]
            elif doc.n_channels == 1:
                doc.luts = [like.lut.copy()]
        doc.properties = dict(like.properties)
    if labels is not None and len(labels) == doc.stack_size:
        doc.slice_labels = list(labels)
    return doc


def type_kind(doc: ImageDoc) -> str:
    return {GRAY8: "byte", COLOR_256: "byte", GRAY16: "short", GRAY32: "float",
            COLOR_RGB: "rgb"}[doc.type]


# ---------------------------------------------------------------------------------- Z project
def _project_gray(planes: np.ndarray, method: str, bit_depth: int, signed16: bool = False
                  ) -> tuple[np.ndarray, int]:
    """ZProjector.doProjection on gray planes (N, Y, X); returns (pixels, output bit depth)."""
    n = planes.shape[0]
    if bit_depth == 32 and method == "Average Intensity":
        f = planes.astype(np.float64)
        valid = ~np.isnan(f)
        s = np.where(valid, f, 0.0).sum(axis=0)
        cnt = valid.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (s / cnt).astype(np.float32), 32
    if method == "Median":
        f = np.sort(planes.astype(np.float32), axis=0)
        mid = n // 2
        if n % 2 == 0:
            med = (f[mid - 1] + f[mid]) / np.float32(2.0)
        else:
            med = f[mid]
        if bit_depth == 8:
            return float_to_byte(med, 0, 255, scale=False), 8
        return med.astype(np.float32), 32 if bit_depth != 16 else 32
    acc = np.zeros(planes.shape[1:], dtype=np.float32)
    if method in ("Average Intensity", "Sum Slices"):
        for k in range(n):
            acc = acc + planes[k].astype(np.float32)
        if method == "Sum Slices":
            if signed16:
                acc = (acc.astype(np.float64) - n * 32768.0).astype(np.float32)
            return acc, 32
        acc = acc / np.float32(n)
    elif method == "Max Intensity":
        acc[...] = -np.finfo(np.float32).max
        for k in range(n):
            v = planes[k].astype(np.float32)
            acc = np.where(~np.isnan(v) & (v > acc), v, acc)
    elif method == "Min Intensity":
        acc[...] = np.finfo(np.float32).max
        for k in range(n):
            v = planes[k].astype(np.float32)
            acc = np.where(v < acc, v, acc)
    elif method == "Standard Deviation":
        s = np.zeros(planes.shape[1:])
        s2 = np.zeros(planes.shape[1:])
        for k in range(n):
            v = planes[k].astype(np.float64)
            s += v
            s2 += v * v
        if n > 1:
            sd = (n * s2 - s * s) / n
            with np.errstate(invalid="ignore"):
                out = np.where(sd > 0.0, np.sqrt(np.maximum(sd, 0) / (n - 1.0)), 0.0)
            return out.astype(np.float32), 32
        return np.zeros(planes.shape[1:], np.float32), 32
    else:
        raise StackError(f"Unknown projection method {method!r}")
    if bit_depth == 8:
        return (java_int((acc + np.float32(0.5)).astype(np.float32)) & 0xFF).astype(np.uint8), 8
    if bit_depth == 16:
        return (java_int((acc + np.float32(0.5)).astype(np.float32)) & 0xFFFF
                ).astype(np.uint16), 16
    return acc.astype(np.float32), 32


def _project_rgb(planes: np.ndarray, method: str, clip: bool = False) -> np.ndarray:
    """ZProjector.doRGBProjection: per channel, SD/Sum rescaled by 255/max over channels and
    converted with the channel's own min..max (RGBStackMerge)."""
    res = [_project_gray(np.ascontiguousarray(planes[..., k]), method, 8)[0] for k in range(3)]
    if method == "Standard Deviation" or (method == "Sum Slices" and not clip):
        mx = max(float(np.nanmax(r)) for r in res)
        scale = np.float32(255.0 / mx) if mx != 0 else np.float32(np.inf)
        res = [(r * scale).astype(np.float32) for r in res]
    out = []
    for r in res:
        if r.dtype == np.uint8:
            out.append(r)
        else:
            out.append(float_to_byte(r, float(np.nanmin(r)), float(np.nanmax(r)), True))
    return np.stack(out, axis=-1)


def z_project(doc: ImageDoc, method: str = "Max Intensity", start: int = 1,
              stop: int | None = None, all_frames: bool = True) -> ImageDoc:
    """Image > Stacks > Z Project. ``start``/``stop`` are 1-based slices (z of a hyperstack,
    frames when it has one slice per frame)."""
    if doc.stack_size == 1:
        raise StackError("Stack required")
    if method not in PROJECTIONS:
        raise StackError(f"Unknown projection method {method!r}")
    c_n, z_n, t_n = doc.dims
    hyper = (doc.hyperstack or (c_n > 1 and c_n < doc.stack_size) or
             (t_n > 1 and z_n > 1)) and not (c_n == doc.stack_size)
    signed16 = doc.calibration.is_signed16()
    title = PREFIX[method] + doc.title

    def proj(planes):
        if doc.type == COLOR_RGB:
            return _project_rgb(planes, method), COLOR_RGB
        pix, bits = _project_gray(planes, method, 8 if doc.type == COLOR_256 else doc.bit_depth,
                                  signed16)
        return pix, {8: GRAY8, 16: GRAY16, 32: GRAY32}[bits]

    if not hyper:
        flat = _flat(doc)
        stop = stop or doc.stack_size
        start = min(max(1, start), doc.stack_size)
        stop = min(max(start, stop), doc.stack_size)
        pix, t = proj(flat[start - 1:stop])
        out = from_planes(pix[None], t, title, like=doc)
        if doc.type == COLOR_256:
            out.type = COLOR_256 if t == GRAY8 else t
        if t != COLOR_RGB and doc.is_composite:
            out.luts = [doc.luts[0].copy()]
        out.reset_display_range()
        return out
    depth_axis_is_t = z_n == 1
    depth = t_n if depth_axis_is_t else z_n
    stop = stop or depth
    start = min(max(1, start), depth)
    stop = min(max(start, stop), depth)
    frames = range(t_n) if (all_frames and not depth_axis_is_t) else \
        ([0] if depth_axis_is_t else [doc.position[2]])
    planes_out = []
    out_t = None
    for t in frames:
        for c in range(c_n):
            if depth_axis_is_t:
                stackp = doc.pixels[start - 1:stop, 0, c]
            else:
                stackp = doc.pixels[t, start - 1:stop, c]
            pix, out_t = proj(stackp)
            planes_out.append(pix)
    out = from_planes(np.stack(planes_out), out_t, title, dims=(c_n, 1, len(frames)), like=doc)
    out.hyperstack = len(frames) > 1
    out.composite_mode = doc.composite_mode
    if out_t == doc.type and method not in ("Sum Slices", "Standard Deviation"):
        out.display_ranges = list(doc.display_ranges)
    else:
        out.reset_display_range()
    return out


def grouped_z_project(doc: ImageDoc, method: str, group: int) -> ImageDoc:
    """Image > Stacks > Tools > Grouped Z Project."""
    n = doc.stack_size
    if group < 1 or n % group != 0:
        raise StackError(f"Group size must divide evenly into the stack size ({n}).")
    flat = _flat(doc)
    planes = []
    t = None
    for g in range(n // group):
        sub = from_planes(flat[g * group:(g + 1) * group], doc.type, "tmp", like=doc)
        r = z_project(sub, method)
        planes.append(r.plane())
        t = r.type
    out = from_planes(np.stack(planes), t, PREFIX[method] + doc.title, like=doc)
    out.reset_display_range()
    return out


# ---------------------------------------------------------------------------------- reslice
def reslice(doc: ImageDoc, output_spacing: float = 1.0, start: str = "Top", flip: bool = False,
            rotate: bool = False, avoid_interpolation: bool = False,
            rect: tuple | None = None, line: tuple | None = None, slice_count: int = 1
            ) -> ImageDoc:
    """Image > Stacks > Reslice (rectangle/whole image, or a straight ``line`` x1,y1,x2,y2)."""
    from rsstatchem.core.imaging import transform as T
    cal = doc.calibration
    pw = cal.pixel_width or 1.0
    input_z = abs(cal.pixel_depth)
    out_z = output_spacing / pw
    if avoid_interpolation:
        input_z = 1.0
        if line is None:
            out_z = 1.0
        pw = 1.0
    z_spacing = input_z / pw
    kind = type_kind(doc)
    flat = _flat(doc)
    n = flat.shape[0]
    order = range(n - 1, -1, -1) if flip else range(n)
    h, w = doc.height, doc.width

    def get_line(p, x1, y1, x2, y2):
        ortho = ((int(x1) == x1 and int(y1) == y1 and x1 == x2) or y1 == y2)
        if ortho:
            x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
            dx, dy = x2i - x1i, y2i - y1i
            m = max(abs(dx), abs(dy))
            xinc, yinc = int(dx / m), int(dy / m)
            xs = x1i + xinc * np.arange(m)
            ys = y1i + yinc * np.arange(m)
            check = x1i < 0 or x1i > w or y1i < 0 or y1i > h or x2i < 0 or x2i > w or \
                y2i < 0 or y2i > h
            if check and kind != "rgb":
                ok = (xs > 0) & (xs < w) & (ys > 0) & (ys < h)
                vals = np.where(ok, p[np.clip(ys, 0, h - 1), np.clip(xs, 0, w - 1)], 0)
                return vals
            return p[np.clip(ys, 0, h - 1), np.clip(xs, 0, w - 1)]
        dx, dy = x2 - x1, y2 - y1
        m = int(round(math.sqrt(dx * dx + dy * dy)))
        xs = x1 + np.arange(m) * (dx / m)
        ys = y1 + np.arange(m) * (dy / m)
        if kind == "rgb":
            return np.stack([T._clamped_bilinear(np.ascontiguousarray(p[..., k]), xs, ys,
                                                 "byte") for k in range(3)], axis=-1)
        return T._clamped_bilinear(p, xs, ys, "float" if kind == "float" else kind)

    def slice_at(x1, y1, x2, y2):
        rows = [get_line(flat[i], x1, y1, x2, y2) for i in order]
        arr = np.stack(rows)                       # (n, len[, 3])
        if kind in ("byte", "short") and arr.dtype.kind == "f":
            arr = T._from_float(arr.astype(np.float32), kind)
        arr = arr.astype(flat.dtype)
        if rotate:
            arr = np.swapaxes(arr, 0, 1)
        if z_spacing != 1.0:
            if rotate:
                arr = T.resize_plane(arr, kind, int(n * z_spacing), arr.shape[0], T.BILINEAR)
            else:
                arr = T.resize_plane(arr, kind, arr.shape[1], int(n * z_spacing), T.BILINEAR)
        return arr

    out = []
    if line is None:
        rx, ry, rw, rh = rect if rect is not None else (0, 0, w, h)
        if start == "Top":
            x1, y1, x2, y2, xi, yi, cnt = rx, ry, rx + rw, ry, 0.0, out_z, int(rh / out_z)
        elif start == "Left":
            x1, y1, x2, y2, xi, yi, cnt = rx, ry, rx, ry + rh, out_z, 0.0, int(rw / out_z)
        elif start == "Bottom":
            x1, y1, x2, y2, xi, yi, cnt = (rx, ry + rh - 1, rx + rw, ry + rh - 1, 0.0, -out_z,
                                           int(rh / out_z))
        else:
            x1, y1, x2, y2, xi, yi, cnt = (rx + rw - 1, ry, rx + rw - 1, ry + rh, -out_z, 0.0,
                                           int(rw / out_z))
    else:
        x1, y1, x2, y2 = line
        dx, dy = x2 - x1, y2 - y1
        nrm = math.sqrt(dx * dx + dy * dy) / out_z
        xi, yi = -(dy / nrm), dx / nrm
        cnt = max(int(slice_count), 1)
    if cnt == 0:
        raise StackError(f"Output Z spacing ({out_z:.0f} pixels) is too large.")
    for _ in range(cnt):
        out.append(slice_at(x1, y1, x2, y2))
        x1 += xi
        x2 += xi
        y1 += yi
        y2 += yi
    res = from_planes(np.stack(out), doc.type if doc.type != COLOR_256 else COLOR_256,
                      "Reslice of " + doc.title.rsplit(".", 1)[0], like=doc)
    if doc.type != COLOR_RGB:
        res.display_ranges = [doc.display_range] * res.n_channels
    c2 = res.calibration
    horizontal = line is None and start in ("Top", "Bottom")
    if line is not None:
        horizontal = y2 - y1 == 0 and False
    if line is None:
        if horizontal:
            c2.pixel_width, c2.pixel_height = cal.pixel_width, cal.pixel_depth / z_spacing
            c2.pixel_depth = cal.pixel_height * out_z
        else:
            c2.pixel_width, c2.pixel_height = cal.pixel_height, cal.pixel_depth / z_spacing
            c2.pixel_depth = cal.pixel_width * out_z
        if rotate:
            c2.pixel_width, c2.pixel_height = c2.pixel_height, c2.pixel_width
    return res


# ---------------------------------------------------------------------------------- montage
def _fill_value(kind: str, white: bool, display=(0.0, 255.0)):
    if kind == "rgb":
        return (255, 255, 255) if white else (0, 0, 0)
    if kind == "byte":
        return 255 if white else 0
    lo, hi = display                            # ShortProcessor/FloatProcessor.setColor
    if white:
        return int(hi) if kind == "short" else hi
    return 0 if lo > 0 else (int(lo) if kind == "short" else lo)


def make_montage(doc: ImageDoc, columns: int, rows: int, scale: float = 1.0, first: int = 1,
                 last: int | None = None, increment: int = 1, border: int = 0,
                 font_size: int = 12, label: bool = False, use_foreground: bool = False,
                 foreground="#ffffff", background="#000000") -> ImageDoc:
    """Image > Stacks > Make Montage (MontageMaker.makeMontage2; labels are drawn by the UI)."""
    from rsstatchem.core.imaging import transform as T
    flat = _flat(doc)
    n = flat.shape[0]
    last = n if last is None else min(last, n)
    kind = type_kind(doc)
    width = int(doc.width * scale)
    height = int(doc.height * scale)
    mw = width * columns + border * (columns - 1)
    mh = height * rows + border * (rows - 1)
    white_bg = False
    if kind in ("byte", "rgb") and not use_foreground:
        strip = doc.plane()[max(doc.height - 12, 0):]
        if kind == "rgb":
            from rsstatchem.core.imaging.convert import rgb_to_byte
            strip = rgb_to_byte(strip)
        hist = np.bincount(strip.ravel().astype(np.int64), minlength=256)
        white_bg = int(np.argmax(hist)) >= 200
        if doc.is_inverted_lut():
            white_bg = not white_bg
    shape = (mh, mw) + ((3,) if kind == "rgb" else ())
    montage = np.zeros(shape, dtype=flat.dtype)
    disp = doc.display_range if kind != "rgb" else (0.0, 255.0)
    montage[...] = _fill_value(kind, white_bg, disp)
    x = y = 0
    s = first
    while s <= last:
        p = flat[s - 1]
        if scale != 1.0:
            p = T.resize_plane(p, kind, width, height, T.BILINEAR, None, width < 200)
        T.insert(montage, p, x, y)
        x += width + border
        if x >= mw:
            x = 0
            y += height + border
            if y >= mh:
                break
        s += increment
    if border > 0:
        fg = _fill_value(kind, not white_bg, disp)
        for xx in range(width, mw, width + border):
            montage[:, xx:xx + border] = fg
        for yy in range(height, mh, height + border):
            montage[yy:yy + border, :] = fg
    out = from_planes(montage[None], doc.type, "Montage", like=doc)
    if out.calibration.scaled():
        out.calibration.pixel_width /= scale
        out.calibration.pixel_height /= scale
    out.info = f"xMontage={columns}\nyMontage={rows}\n"
    out.properties["xMontage"] = str(columns)
    out.properties["yMontage"] = str(rows)
    out.reset_display_range()
    if doc.type != COLOR_RGB and doc.bit_depth != 8:
        out.display_ranges = [doc.display_range]
    return out


def montage_to_stack(doc: ImageDoc, columns: int, rows: int, border: int = 0) -> ImageDoc:
    """Image > Stacks > Tools > Montage to Stack (StackMaker.makeStack)."""
    p = doc.plane()
    w = doc.width // columns
    h = doc.height // rows
    tiles = [p[y * h:(y + 1) * h, x * w:(x + 1) * w] for y in range(rows) for x in range(columns)]
    arr = np.stack(tiles)
    if border > 0:
        cw = w - border - border // 2
        ch = h - border - border // 2
        arr = arr[:, border:border + ch, border:border + cw]
    out = from_planes(np.ascontiguousarray(arr), doc.type, "Stack", like=doc)
    out.reset_display_range()
    return out


# ---------------------------------------------------------------------------------- tools
def reduce_stack(doc: ImageDoc, factor: int, reduce_z: bool = True) -> None:
    """Image > Stacks > Tools > Reduce (keeps slices 1, 1+f, 1+2f ...)."""
    if factor < 1:
        return
    c, z, t = doc.dims
    if doc.hyperstack and (z > 1 or t > 1) and (c > 1 or (z > 1 and t > 1)):
        if reduce_z and z > 1:
            pix = doc.pixels[:, ::factor]
        else:
            pix = doc.pixels[::factor]
        _replace_keep(doc, np.ascontiguousarray(pix))
        return
    flat = _flat(doc)
    keep = list(range(0, flat.shape[0], factor))
    labels = [doc.slice_labels[i] for i in keep]
    _replace_keep(doc, from_planes(flat[keep], doc.type, "x").pixels)
    doc.slice_labels = labels
    if doc.calibration.scaled():
        doc.calibration.pixel_depth *= factor


def _replace_keep(doc: ImageDoc, pixels: np.ndarray) -> None:
    luts = [lut.copy() for lut in doc.luts]
    ranges = list(doc.display_ranges)
    mode = doc.composite_mode
    doc.replace_pixels(pixels, doc.type)
    if len(luts) == doc.n_channels:
        doc.luts = luts
        doc.display_ranges = ranges
        doc.composite_mode = mode


def reverse(doc: ImageDoc) -> None:
    """Image > Stacks > Tools > Reverse (StackReverser: frames of a time series, else slices)."""
    c, z, t = doc.dims
    if t > 1 and (z == 1 or c > 1):
        doc.pixels = np.ascontiguousarray(doc.pixels[::-1])
        return
    if z > 1 and (c > 1 or t > 1):
        doc.pixels = np.ascontiguousarray(doc.pixels[:, ::-1])
        return
    labels = doc.slice_labels[::-1]
    flat = _flat(doc)[::-1]
    doc.pixels = np.ascontiguousarray(flat.reshape(doc.pixels.shape))
    doc.slice_labels = labels


def parse_range(text: str, n: int) -> list[int]:
    """SubstackMaker ranges: "2-14", "1-100-2" or "7,9,25,27" -> 1-based list."""
    text = text.strip().replace(" ", "")
    out: list[int] = []
    if not text:
        return out
    for part in text.split(","):
        if "-" in part[1:]:
            bits = part.split("-")
            a, b = int(float(bits[0])), int(float(bits[1]))
            inc = int(float(bits[2])) if len(bits) > 2 else 1
            out.extend(range(a, b + 1, max(inc, 1)))
        else:
            out.append(int(float(part)))
    bad = [i for i in out if i < 1 or i > n]
    if bad:
        raise StackError(f"Slice {bad[0]} is out of range (1-{n})")
    return out


def substack(doc: ImageDoc, slices: str, delete: bool = False) -> ImageDoc:
    """Image > Stacks > Tools > Make Substack."""
    idx = parse_range(slices, doc.stack_size)
    flat = _flat(doc)
    labels = [doc.slice_labels[i - 1] for i in idx]
    out = from_planes(flat[[i - 1 for i in idx]], doc.type, "Substack (" + slices + ")",
                      like=doc, labels=labels)
    out.display_ranges = [doc.display_range]
    if delete:
        keep = [i for i in range(doc.stack_size) if i + 1 not in set(idx)]
        if keep:
            lab = [doc.slice_labels[i] for i in keep]
            _replace_keep(doc, from_planes(flat[keep], doc.type, "x").pixels)
            doc.slice_labels = lab
    return out


def combine(a: ImageDoc, b: ImageDoc, vertical: bool = False, background=0) -> ImageDoc:
    """Image > Stacks > Tools > Combine (side by side or one above the other)."""
    if a.type != b.type:
        raise StackError("The stacks must have the same type")
    fa, fb = _flat(a), _flat(b)
    d = max(fa.shape[0], fb.shape[0])
    if vertical:
        w3, h3 = max(a.width, b.width), a.height + b.height
        fill = background if a.width != b.width else 0
    else:
        w3, h3 = a.width + b.width, max(a.height, b.height)
        fill = background if a.height != b.height else 0
    shape = (d, h3, w3) + fa.shape[3:]
    out = np.zeros(shape, dtype=fa.dtype)
    out[...] = fill
    for i in range(d):
        if i < fa.shape[0]:
            out[i, :a.height, :a.width] = fa[i]
        if i < fb.shape[0]:
            if vertical:
                out[i, a.height:, :b.width] = fb[i]
            else:
                out[i, :b.height, a.width:] = fb[i]
    res = from_planes(out, a.type, "Combined Stacks", like=a)
    res.reset_display_range()
    return res


def concatenate(images: list[ImageDoc], title: str = "Untitled", as_4d: bool = False
                ) -> ImageDoc:
    """Image > Stacks > Tools > Concatenate (same size and type)."""
    if len(images) < 2:
        raise StackError("At least two images are required")
    first = images[0]
    for im in images[1:]:
        if (im.width, im.height, im.type) != (first.width, first.height, first.type):
            raise StackError("Images do not all have the same dimensions or type")
    planes = np.concatenate([_flat(im) for im in images])
    labels = sum((list(im.slice_labels) for im in images), [])
    if as_4d and all(im.stack_size == first.stack_size for im in images):
        out = from_planes(planes, first.type, title,
                          dims=(1, first.stack_size, len(images)), like=first, labels=labels)
        out.hyperstack = True
    else:
        out = from_planes(planes, first.type, title, like=first, labels=labels)
    out.display_ranges = [first.display_range] * out.n_channels
    return out


def insert_stack(src: ImageDoc, dst: ImageDoc, x: int, y: int) -> None:
    """Image > Stacks > Tools > Insert (StackInserter: source slices into destination)."""
    from rsstatchem.core.imaging import transform as T
    if src.type != dst.type:
        raise StackError("Source and destination must have the same type")
    fs, fd = _flat(src), _flat(dst)
    for i in range(fd.shape[0]):
        T.insert(fd[i], fs[min(i, fs.shape[0] - 1)] if fs.shape[0] > 1 else fs[0], x, y)


def add_slice(doc: ImageDoc, after: bool = True, fill=0) -> None:
    """Image > Stacks > Add Slice (after the current one)."""
    flat = _flat(doc)
    i = doc.current_index + (1 if after else 0)
    blank = np.zeros_like(flat[0])
    blank[...] = fill
    new = np.concatenate([flat[:i], blank[None], flat[i:]])
    labels = list(doc.slice_labels)
    labels.insert(i, None)
    if doc.n_channels > 1 or doc.n_frames > 1:
        c, z, t = doc.dims
        if c > 1:       # add a z-slice to every channel
            pix = np.insert(doc.pixels, doc.position[1] + 1, 0, axis=1)
            _replace_keep(doc, pix)
            return
    _replace_keep(doc, from_planes(new, doc.type, "x").pixels)
    doc.slice_labels = labels
    doc.set_position(0, i, 0)


def delete_slice(doc: ImageDoc) -> None:
    if doc.stack_size < 2:
        raise StackError('"Delete Slice" requires a stack')
    c, z, t = doc.dims
    if c > 1 and doc.is_composite:
        if z > 1:
            pix = np.delete(doc.pixels, doc.position[1], axis=1)
        elif t > 1:
            pix = np.delete(doc.pixels, doc.position[2], axis=0)
        else:
            pix = np.delete(doc.pixels, doc.position[0], axis=2)
        _replace_keep(doc, pix) if pix.shape[2] == c else doc.replace_pixels(pix, doc.type)
        return
    flat = _flat(doc)
    i = doc.current_index
    labels = list(doc.slice_labels)
    del labels[i]
    _replace_keep(doc, from_planes(np.delete(flat, i, axis=0), doc.type, "x").pixels)
    doc.slice_labels = labels
    doc.set_position(0, min(i, doc.stack_size - 1), 0)


def images_to_stack(images: list[ImageDoc], name: str = "Stack", use_titles: bool = True,
                    method: str = "Copy (center)", bicubic: bool = False,
                    keep: bool = False) -> ImageDoc:
    """Image > Stacks > Images to Stack (ImagesToStack: "Copy (center)", "Copy (top-left)" or
    "Scale (smallest)"/"Scale (largest)")."""
    from rsstatchem.core.imaging import transform as T
    from rsstatchem.core.imaging.convert import to_gray32, to_rgb
    if not images:
        raise StackError("There must be at least one open image.")
    types = {im.type for im in images}
    work = [im.duplicate() if len(types) > 1 else im for im in images]
    if len(types) > 1:
        target = COLOR_RGB if COLOR_RGB in types else GRAY32
        for im in work:
            if im.type != target:
                (to_rgb if target == COLOR_RGB else to_gray32)(im)
    t = work[0].type
    if method.startswith("Scale"):
        sizes = [(im.width * im.height, im.width, im.height) for im in work]
        _, W, H = (min if "smallest" in method else max)(sizes)
    else:
        W = max(im.width for im in work)
        H = max(im.height for im in work)
    kind = type_kind(work[0])
    planes = []
    labels = []
    for im in work:
        p = im.plane()
        if (im.width, im.height) != (W, H):
            if method.startswith("Scale"):
                p = T.resize_plane(p, kind, W, H, T.BICUBIC if bicubic else T.BILINEAR)
            else:
                canvas = np.zeros((H, W) + p.shape[2:], dtype=p.dtype)
                xo = (W - im.width) // 2 if "center" in method else 0
                yo = (H - im.height) // 2 if "center" in method else 0
                T.insert(canvas, p, xo, yo)
                p = canvas
        planes.append(p)
        labels.append(im.title if use_titles else None)
    out = from_planes(np.stack(planes), t, name, like=work[0], labels=labels)
    if t in (GRAY16, GRAY32):
        lo = min(im.display_range[0] for im in work)
        hi = max(im.display_range[1] for im in work)
        out.display_ranges = [(lo, hi)]
    return out


def stack_to_images(doc: ImageDoc) -> list[ImageDoc]:
    """Image > Stacks > Stack to Images (titles from slice labels or "<title>-0001")."""
    flat = _flat(doc)
    stem = doc.title.rsplit(".", 1)[0] if "." in doc.title else doc.title
    digits = max(4, len(str(doc.stack_size)))
    out = []
    for i in range(flat.shape[0]):
        label = doc.slice_labels[i]
        title = label.splitlines()[0] if label else f"{stem}-{i + 1:0{digits}d}"
        im = from_planes(flat[i][None], doc.type, title, like=doc)
        c = doc.position_of(i)[0]
        if doc.type != COLOR_RGB:
            im.luts = [doc.luts[c].copy()]
            im.display_ranges = [doc.display_ranges[c]]
        out.append(im)
    return out


# ---------------------------------------------------------------------------------- hyperstacks
def stack_to_hyperstack(doc: ImageDoc, channels: int, slices: int, frames: int,
                        order: str = "xyczt(default)", mode: str = "Color") -> None:
    """Image > Hyperstacks > Stack to Hyperstack (HyperStackConverter.shuffle)."""
    n = doc.stack_size
    if channels * slices * frames != n:
        raise StackError(f"channels x slices x frames <> stack size ({n})")
    order = "xyczt(default)" if order in ("default", "xyczt") else order
    if order not in ORDERS:
        raise StackError(f"Unknown order {order!r}")
    o = ORDERS.index(order)
    nc, nz, nt = channels, slices, frames
    first, middle, last = {0: (0, 1, 2), 1: (0, 2, 1), 2: (1, 0, 2), 3: (1, 2, 0),
                           4: (2, 0, 1), 5: (2, 1, 0)}[o]
    sizes = (nc, nz, nt)
    flat = _flat(doc)
    labels = list(doc.slice_labels)
    new = np.empty_like(flat)
    new_labels = [None] * n
    for t in range(nt):
        for z in range(nz):
            for c in range(nc):
                idx = (c, z, t)
                dst = c + z * nc + t * nc * nz
                src = idx[first] + idx[middle] * sizes[first] + \
                    idx[last] * sizes[first] * sizes[middle]
                new[dst] = flat[src]
                new_labels[dst] = labels[src]
    luts = [lut.copy() for lut in doc.luts]
    rng = doc.display_range
    rgb = doc.type == COLOR_RGB
    arr = new.reshape((nt, nz, nc) + flat.shape[1:])
    doc.replace_pixels(np.ascontiguousarray(arr), doc.type)
    doc.slice_labels = new_labels
    doc.hyperstack = True
    if nc > 1 and not rgb:
        m = mode.lower()
        doc.composite_mode = m if m in ("composite", "color", "grayscale") else "color"
        if len(luts) >= nc:
            doc.luts = luts[:nc]
        doc.reset_display_range()
    elif not rgb:
        doc.luts = [luts[0]]
        doc.display_ranges = [rng]


def hyperstack_to_stack(doc: ImageDoc) -> None:
    """Image > Hyperstacks > Hyperstack to Stack (all planes as one z series)."""
    flat = _flat(doc)
    labels = list(doc.slice_labels)
    lut = doc.lut.copy() if doc.type != COLOR_RGB else None
    rng = doc.display_range
    doc.replace_pixels(np.ascontiguousarray(flat[None, :, None]), doc.type)
    doc.slice_labels = labels
    doc.hyperstack = False
    if lut is not None:
        doc.luts = [lut]
        doc.display_ranges = [rng]


def reduce_dimensionality(doc: ImageDoc, keep_channels: bool, keep_slices: bool,
                          keep_frames: bool) -> ImageDoc:
    """Image > Hyperstacks > Reduce Dimensionality: dimensions not kept are taken at the
    current position."""
    c0, z0, t0 = doc.position
    p = doc.pixels
    p = p if keep_frames else p[t0:t0 + 1]
    p = p if keep_slices else p[:, z0:z0 + 1]
    p = p if keep_channels else p[:, :, c0:c0 + 1]
    out = ImageDoc(np.ascontiguousarray(p), doc.type, doc.title.rsplit(".", 1)[0] + "-1")
    out.calibration = doc.calibration.copy()
    if doc.type != COLOR_RGB:
        chans = range(doc.n_channels) if keep_channels else [c0]
        out.luts = [doc.luts[c].copy() for c in chans]
        out.display_ranges = [doc.display_ranges[c] for c in chans]
        out.composite_mode = doc.composite_mode
    out.hyperstack = out.n_channels > 1 or (out.n_slices > 1 and out.n_frames > 1)
    return out


def make_subset(doc: ImageDoc, channels: str = "", slices: str = "", frames: str = ""
                ) -> ImageDoc:
    """Image > Hyperstacks > Make Subset (SubHyperstackMaker) with range strings."""
    cs = [i - 1 for i in parse_range(channels, doc.n_channels)] if channels else \
        list(range(doc.n_channels))
    zs = [i - 1 for i in parse_range(slices, doc.n_slices)] if slices else \
        list(range(doc.n_slices))
    ts = [i - 1 for i in parse_range(frames, doc.n_frames)] if frames else \
        list(range(doc.n_frames))
    p = doc.pixels[np.ix_(ts, zs, cs)] if doc.type != COLOR_RGB else \
        doc.pixels[ts][:, zs][:, :, cs]
    out = ImageDoc(np.ascontiguousarray(p), doc.type, doc.title.rsplit(".", 1)[0] + "-1")
    out.calibration = doc.calibration.copy()
    if doc.type != COLOR_RGB:
        out.luts = [doc.luts[c].copy() for c in cs]
        out.display_ranges = [doc.display_ranges[c] for c in cs]
        out.composite_mode = doc.composite_mode
    out.hyperstack = True
    return out


def new_hyperstack(title: str, type_: int, width: int, height: int, channels: int,
                   slices: int, frames: int, mode: str = "Composite", label: bool = True
                   ) -> ImageDoc:
    """Image > Hyperstacks > New Hyperstack (black; optional "c:1/3 z:2/5 t:1/1" labels)."""
    doc = ImageDoc.new(width, height, type_, channels, slices, frames, 0, title)
    doc.hyperstack = True
    if channels > 1 and type_ != COLOR_RGB:
        doc.composite_mode = mode.lower() if mode.lower() in ("composite", "color",
                                                               "grayscale") else "composite"
    if label:
        for i in range(doc.stack_size):
            c, z, t = doc.position_of(i)
            doc.slice_labels[i] = f"c:{c + 1}/{channels} z:{z + 1}/{slices} t:{t + 1}/{frames}"
    if type_ in (GRAY16, GRAY32):
        doc.display_ranges = [(0.0, 255.0 if type_ == GRAY32 else 65535.0)] * doc.n_channels
    return doc


# ---------------------------------------------------------------------------------- statistics
def stack_statistics(doc: ImageDoc, mask: np.ndarray | None = None):
    """Image > Stacks > Statistics: dict with Voxels, %Volume, Mean, StdDev, Min, Max, Mode,
    Median (StackStatistics, 8/16-bit via full histograms; 32-bit with 256 bins)."""
    flat = _flat(doc)
    if doc.type == COLOR_RGB:
        from rsstatchem.core.imaging.convert import rgb_to_byte
        flat = np.stack([rgb_to_byte(p) for p in flat])
    vals = flat[:, mask] if mask is not None else flat.reshape(flat.shape[0], -1)
    vals = vals.ravel()
    total = flat.shape[0] * (int(mask.sum()) if mask is not None else doc.width * doc.height)
    cal = doc.calibration
    if doc.bit_depth in (8, 24):
        hist = np.bincount(vals.astype(np.int64), minlength=256).astype(np.int64)
        i = np.arange(256, dtype=np.float64)
        n = int(hist.sum())
        s = float((i * hist).sum())
        s2 = float((i * i * hist).sum())
        mode = float(np.argmax(hist))
        nz = np.nonzero(hist)[0]
        vmin, vmax = float(nz[0]), float(nz[-1])
        median = _median_hist(hist, 0, 255, n)
    elif doc.bit_depth == 16:
        hist16 = np.bincount(vals.astype(np.int64), minlength=65536).astype(np.int64)
        nz = np.nonzero(hist16)[0]
        lo, hi = int(nz[0]), int(nz[-1])
        i = np.arange(lo, hi + 1, dtype=np.float64)
        h = hist16[lo:hi + 1]
        n = int(h.sum())
        s = float((i * h).sum())
        s2 = float((i * i * h).sum())
        bin_size = (hi - lo + 1) / 256.0
        idx = np.minimum(java_int((1.0 / bin_size) * (i - lo)), 255)
        h256 = np.bincount(idx, weights=h, minlength=256)
        mode = lo + int(np.argmax(h256)) * bin_size
        vmin, vmax = float(lo), float(hi)
        median = _median_hist(hist16, 0, 65535, n)
    else:
        f = vals[np.isfinite(vals)].astype(np.float64)
        n = f.size
        s, s2 = float(f.sum()), float((f * f).sum())
        vmin, vmax = float(f.min()), float(f.max())
        bin_size = (vmax - vmin) / 256.0 if vmax > vmin else 1.0
        idx = np.minimum(java_int((256 / (vmax - vmin) if vmax > vmin else 0) * (f - vmin)),
                         255)
        h256 = np.bincount(idx, minlength=256)
        mode = vmin + int(np.argmax(h256)) * bin_size
        median = float(np.median(f))
    mean = s / n if n else float("nan")
    sd = 0.0
    if n > 1:
        v = (n * s2 - s * s) / n
        sd = math.sqrt(v / (n - 1.0)) if v > 0 else 0.0
    out = {"Voxels": float(n)}
    scale = cal.pixel_width * cal.pixel_height * cal.pixel_depth
    if scale != 1.0:
        out[f"Volume({cal.units}^3)"] = n * scale
    out.update({"%Volume": n * 100.0 / total, "Mean": float(cal.cvalue(mean)),
                "StdDev": sd, "Min": float(cal.cvalue(vmin)), "Max": float(cal.cvalue(vmax)),
                "Mode": float(cal.cvalue(mode)), "Median": float(cal.cvalue(median))})
    return out


def _median_hist(hist, first, last, count):
    if count == 0:
        return float("nan")
    half = count / 2.0
    cs = np.cumsum(hist[first:last + 1].astype(np.float64))
    i = int(np.searchsorted(cs, half, side="right"))
    return float(first + min(i, last - first))


def z_profile(doc: ImageDoc, mask: np.ndarray | None = None, rect=None) -> tuple:
    """Image > Stacks > Plot Z-axis Profile: (positions, means) along z (or t for time
    series) of the selection; the current channel of a hyperstack."""
    from rsstatchem.core.imaging.convert import rgb_to_byte
    c0, z0, t0 = doc.position
    axis_t = doc.n_frames > 1 and doc.n_slices == 1
    n = doc.n_frames if axis_t else doc.n_slices
    if doc.n_channels == 1 and doc.n_frames == 1:
        planes = _flat(doc)
    elif axis_t:
        planes = doc.pixels[:, z0, c0]
    else:
        planes = doc.pixels[t0, :, c0]
    means = []
    for p in planes[:n]:
        if doc.type == COLOR_RGB:
            p = rgb_to_byte(p)
        if rect is not None:
            x, y, w, h = rect
            p = p[y:y + h, x:x + w]
            m = mask[y:y + h, x:x + w] if mask is not None else None
        else:
            m = mask
        v = p[m] if m is not None else p
        v = doc.calibration.cvalue(v.astype(np.float64)) if doc.type != COLOR_RGB else v
        means.append(float(np.nanmean(v)))
    if axis_t and doc.calibration.frame_interval > 0:
        xs = np.arange(1, len(means) + 1) * doc.calibration.frame_interval
    else:
        xs = np.arange(1, len(means) + 1, dtype=float)
    return xs, np.asarray(means)


def label_text(fmt: str, index: int, start: float, interval: float, decimals: int = 0,
               text: str = "", label: str | None = None) -> str:
    """StackLabeler formats: "0", "0000", "00:00", "00:00:00", "Text", "Label"."""
    value = start + index * interval
    if fmt == "Text":
        return text
    if fmt == "Label":
        return label or ""
    if fmt == "0000":
        return f"{int(value):04d}" + (f" {text}" if text else "")
    if fmt == "00:00":
        v = int(round(value))
        return f"{v // 60:02d}:{v % 60:02d}"
    if fmt == "00:00:00":
        v = int(round(value))
        return f"{v // 3600:02d}:{(v // 60) % 60:02d}:{v % 60:02d}"
    s = f"{value:.{decimals}f}"
    return s + (f" {text}" if text else "")


def blank_lut() -> LUT:
    return LUT.grays()

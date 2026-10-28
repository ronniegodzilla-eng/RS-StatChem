"""Image Analysis commands: Edit menu (Undo, clipboard, Clear/Fill/Draw/Invert) and
Edit > Options. **Owned by WP-A.** (Edit > Selection belongs to WP-C.)

Pixel operations follow ImageJ 1.54 (public domain, NIH): ``ij/plugin/filter/Filler.java``
(fill with the foreground colour, clear with the background colour, Clear Outside, Draw of
the selection outline with the line width), ``ImageProcessor.lineTo/drawDot/drawDot2`` and
``Roi.drawPixels`` (rectangles are drawn one pixel wider and higher with width 1),
``ij/plugin/Clipboard.java`` and ``frame/PasteController.java`` (paste transfer modes),
``ShortProcessor/FloatProcessor.invert`` (16-bit: max - (v - min) of the pixel range, or the
full range with "Full range 16-bit inversions"; 32-bit: of the display range).

The helpers :func:`color_value`, :func:`fill_mask`, :func:`draw_roi_outline` and
:func:`invert_plane` are pure numpy (no Qt) and are also used by the macro interpreter.
"""
from __future__ import annotations

import math

import numpy as np

from rsstatchem.core.imaging import commands as C
from rsstatchem.core.imaging.commands import CommandError
from rsstatchem.core.imaging.image import COLOR_RGB, GRAY8, GRAY16, GRAY32, ImageDoc

COLOR_NAMES = ("white", "black", "red", "green", "blue", "yellow", "cyan", "magenta",
               "orange", "pink", "gray", "lightGray", "darkGray")
COLOR_HEX = {"white": "#ffffff", "black": "#000000", "red": "#ff0000", "green": "#00ff00",
             "blue": "#0000ff", "yellow": "#ffff00", "cyan": "#00ffff", "magenta": "#ff00ff",
             "orange": "#ffc800", "pink": "#ffafaf", "gray": "#808080",
             "lightgray": "#c0c0c0", "darkgray": "#404040"}
PASTE_MODES = ("Copy", "Blend", "Average", "Difference", "Transparent-white",
               "Transparent-zero", "AND", "OR", "XOR", "Add", "Subtract", "Multiply", "Divide",
               "Min", "Max")


def hex_rgb(color: str) -> tuple[int, int, int]:
    c = COLOR_HEX.get(str(color).lower(), str(color)).lstrip("#")
    c = c[-6:] if len(c) >= 6 else "ffffff"
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def color_name(hexcolor: str) -> str:
    for name, h in COLOR_HEX.items():
        if h == hexcolor.lower():
            return next(n for n in COLOR_NAMES if n.lower() == name)
    return hexcolor


# ---------------------------------------------------------------------------------- pixels
def color_value(im: ImageDoc, color: str):
    """Pixel value ImageJ uses for a colour (``ImageProcessor.setColor(Color)``): RGB as is;
    otherwise the best-matching LUT index, scaled into the display range for 16/32-bit."""
    r, g, b = hex_rgb(color)
    if im.is_rgb:
        return (r, g, b)
    lut = im.lut.as_array().astype(int)
    d = (lut[:, 0] - r) ** 2 + (lut[:, 1] - g) ** 2 + (lut[:, 2] - b) ** 2
    idx = int(np.argmin(d))
    if im.bit_depth == 8:
        return idx
    lo, hi = im.display_range
    v = lo + (hi - lo) * (idx / 255.0)
    return int(v) if im.type == GRAY16 else float(v)


def _planes(im: ImageDoc, stack: bool):
    return list(im.iter_planes(stack))


def fill_mask(im: ImageDoc, mask: np.ndarray, value, stack: bool = False) -> None:
    for c, z, t in _planes(im, stack):
        p = im.plane(c, z, t)
        p[mask] = value


def _dot(p: np.ndarray, x: int, y: int, width: int, value) -> None:
    h, w = p.shape[:2]
    if width == 1:
        if 0 <= x < w and 0 <= y < h:
            p[y, x] = value
        return
    if width == 2:
        for xx, yy in ((x, y), (x - 1, y), (x, y - 1), (x - 1, y - 1)):
            if 0 <= xx < w and 0 <= yy < h:
                p[yy, xx] = value
        return
    r = width / 2.0
    xmin, ymin = int(x - r + 0.5), int(y - r + 0.5)
    r2 = r * r
    rr = r - 0.5
    for yy in range(ymin, ymin + width):
        for xx in range(xmin, xmin + width):
            dx, dy = xx - (xmin + rr), yy - (ymin + rr)
            if dx * dx + dy * dy <= r2 and 0 <= xx < w and 0 <= yy < h:
                p[yy, xx] = value


def line_points(x1: int, y1: int, x2: int, y2: int) -> list[tuple[int, int]]:
    """``ImageProcessor.lineTo``'s DDA points (rounded half up like Math.round)."""
    dx, dy = x2 - x1, y2 - y1
    n = max(abs(dx), abs(dy))
    if n == 0:
        return [(x1, y1)]
    xinc, yinc = dx / n, dy / n
    pts = []
    x, y = float(x1), float(y1)
    for _ in range(n + 1):
        pts.append((int(math.floor(x + 0.5)), int(math.floor(y + 0.5))))
        x += xinc
        y += yinc
    return pts


def draw_polyline(p: np.ndarray, pts, value, width: int = 1) -> None:
    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
        for x, y in line_points(x1, y1, x2, y2):
            _dot(p, x, y, width, value)
    if len(pts) == 1:
        _dot(p, pts[0][0], pts[0][1], width, value)


def roi_outline_points(roi, width: int = 1) -> list[tuple[int, int]]:
    """Integer vertices ``Roi.drawPixels`` walks for ``roi``."""
    from rsstatchem.core.imaging.roi import shapes as S

    if isinstance(roi, S.RectRoi) and not roi.corner_diameter:
        x, y = int(round(roi.x)), int(round(roi.y))
        w, h = int(round(roi.width)), int(round(roi.height))
        if width == 1:
            w, h = w + 1, h + 1
            return [(x, y), (x + w - 1, y), (x + w - 1, y + h - 1), (x, y + h - 1), (x, y)]
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
    xs, ys = roi.polygon()
    pts = [(int(math.floor(a + 0.5)), int(math.floor(b + 0.5))) for a, b in zip(xs, ys)]
    if roi.is_area() and pts:
        pts.append(pts[0])
    return pts


def draw_roi_outline(im: ImageDoc, roi, value, width: int = 1, stack: bool = False) -> None:
    from rsstatchem.core.imaging.roi import shapes as S

    if isinstance(roi, S.LineRoi) and max(roi.width, roi.stroke_width or 0, width) > 1:
        # Line.drawPixels: a wide line is drawn as the outline of its 4-point area
        sw = max(roi.width, roi.stroke_width or 0, width)
        pts = wide_line_polygon(roi, sw)
        pts.append(pts[0])
        for c, z, t in _planes(im, stack):
            draw_polyline(im.plane(c, z, t), pts, value, 1)
        return
    if roi.stroke_width and roi.stroke_width > 1:
        width = int(round(roi.stroke_width))
    if isinstance(roi, S.CompositeRoi):
        for _, part in roi.parts:
            draw_roi_outline(im, part, value, width, stack)
        return
    if isinstance(roi, S.PointRoi):
        for c, z, t in _planes(im, stack):
            p = im.plane(c, z, t)
            for x, y in zip(*roi.polygon()):
                _dot(p, int(x), int(y), max(width, 1), value)
        return
    pts = roi_outline_points(roi, width)
    for c, z, t in _planes(im, stack):
        draw_polyline(im.plane(c, z, t), pts, value, width)


def wide_line_polygon(roi, stroke_width: float) -> list[tuple[int, int]]:
    """``Line.getPolygon`` of a wide line (rounded corners of its rotated rectangle)."""
    x1, y1, x2, y2 = roi.x1, roi.y1, roi.x2, roi.y2
    dx, dy = x2 - x1, y2 - y1
    n = math.hypot(dx, dy) or 1.0
    dx *= 0.5 / n
    dy *= 0.5 / n
    w = stroke_width
    pts = [(x1 + 0.5 - dx + dy * w, y1 + 0.5 - dy - dx * w),
           (x1 + 0.5 - dx - dy * w, y1 + 0.5 - dy + dx * w),
           (x2 + 0.5 + dx - dy * w, y2 + 0.5 + dy + dx * w),
           (x2 + 0.5 + dx + dy * w, y2 + 0.5 + dy - dx * w)]
    return [(int(math.floor(np.float32(a) + 0.5)), int(math.floor(np.float32(b) + 0.5)))
            for a, b in pts]


def invert_plane(im: ImageDoc, p: np.ndarray, mask: np.ndarray | None, full_range: bool,
                 display_range: tuple[float, float]) -> np.ndarray:
    """Inverted copy of plane ``p`` restricted to ``mask`` (ImageJ Edit > Invert)."""
    out = p.copy()
    sel = np.ones(p.shape[:2], bool) if mask is None else mask
    if im.is_rgb or im.bit_depth == 8:
        out[sel] = 255 - p[sel]
        return out
    if im.type == GRAY16:
        if full_range:
            lo, hi = 0, 65535
        else:
            lo, hi = int(p.min()), int(p.max())
        v = hi - (p[sel].astype(np.int64) - lo)
        out[sel] = np.clip(v, 0, 65535).astype(np.uint16)
        return out
    lo, hi = np.float32(display_range[0]), np.float32(display_range[1])
    out[sel] = (hi - (p[sel].astype(np.float32) - lo)).astype(np.float32)
    return out


def _mask(im: ImageDoc) -> np.ndarray | None:
    roi = im.roi
    if roi is None or not roi.is_area():
        return None
    return roi.get_mask((im.height, im.width))


# ---------------------------------------------------------------------------------- commands
def _stack_field(session, image):
    return []


def undo(session, image):
    if image is None or not image.undo():
        session.status("Undo: nothing to undo")
    else:
        session.image_updated(image)


def fill(session, image, stack=False):
    from rsstatchem.core.imaging.roi import shapes as S
    roi = image.roi
    if roi is None:
        raise CommandError("Area selection required")
    if isinstance(roi, S.PointRoi):
        raise CommandError("Area selection required")
    value = color_value(image, session.options.foreground)
    if roi.is_line():
        lw = max(1, int(round(roi.stroke_width or session.options.line_width)))
        draw_roi_outline(image, roi, value, lw, stack)
        return
    fill_mask(image, roi.get_mask((image.height, image.width)), value, stack)


def clear(session, image, stack=False):
    roi = image.roi
    if roi is None:
        raise CommandError("Area selection required")
    value = color_value(image, session.options.background)
    if roi.is_line():
        draw_roi_outline(image, roi, value, max(1, int(round(roi.stroke_width or 1))), stack)
        return
    if not roi.is_area():
        raise CommandError("Area selection required")
    fill_mask(image, roi.get_mask((image.height, image.width)), value, stack)


def clear_outside(session, image, stack=False):
    roi = image.roi
    if roi is None or not roi.is_area():
        raise CommandError("Area selection required")
    value = color_value(image, session.options.background)
    fill_mask(image, ~roi.get_mask((image.height, image.width)), value, stack)


def draw(session, image, stack=False):
    roi = image.roi
    if roi is None:
        raise CommandError("Selection required")
    value = color_value(image, session.options.foreground)
    width = max(1, int(round(session.options.line_width)))
    draw_roi_outline(image, roi, value, width, stack)


def invert(session, image, stack=False):
    mask = _mask(image)
    full = session.options.full_range_16bit_inversion
    for c, z, t in image.iter_planes(stack):
        p = image.plane(c, z, t)
        image.set_plane(invert_plane(image, p, mask, full, image.display_ranges[c]), c, z, t)


# ---- clipboard
def copy(session, image, cut=False):
    roi = image.roi
    if roi is not None and roi.is_area():
        x, y, w, h = roi.bounds()
        x0, y0 = max(int(math.floor(x)), 0), max(int(math.floor(y)), 0)
        x1 = min(int(math.ceil(x + w)), image.width)
        y1 = min(int(math.ceil(y + h)), image.height)
    else:
        roi = None
        x0, y0, x1, y1 = 0, 0, image.width, image.height
    c, z, t = image.position
    data = image.pixels[t:t + 1, z:z + 1, c:c + 1, y0:y1, x0:x1].copy()
    clip = ImageDoc(data, image.type, "Clipboard", image.calibration.copy())
    clip.luts = [image.lut.copy()]
    clip.display_ranges = [image.display_range]
    if roi is not None:
        r2 = roi.copy()
        r2.translate(-x0, -y0)
        clip.roi = r2
    session.clipboard = clip
    session.status(f"{x1 - x0}x{y1 - y0} image copied to internal clipboard")


def cut(session, image):
    copy(session, image)
    if image.roi is not None and image.roi.is_area():
        clear(session, image)
    else:
        fill_mask(image, np.ones((image.height, image.width), bool),
                  color_value(image, session.options.background))


def _convert_clip(clip: ImageDoc, target: ImageDoc) -> np.ndarray:
    p = clip.plane()
    if clip.type == target.type:
        return p
    if target.is_rgb:
        from rsstatchem.core.imaging.lut import render
        return render(clip)
    if clip.is_rgb:
        g = p.astype(np.float64).mean(axis=2)
    else:
        g = p.astype(np.float64)
    if target.type == GRAY8:
        lo, hi = clip.display_range
        if clip.bit_depth != 8:
            from rsstatchem.core.imaging.lut import to_8bit
            return to_8bit(p, lo, hi) if not clip.is_rgb else np.clip(g + 0.5, 0, 255).astype(
                np.uint8)
        return np.clip(g, 0, 255).astype(np.uint8)
    if target.type == GRAY16:
        return np.clip(np.round(g), 0, 65535).astype(np.uint16)
    return g.astype(np.float32)


def blend(dst: np.ndarray, src: np.ndarray, mode: str, kind: int) -> np.ndarray:
    """Paste transfer mode (PasteController / Blitter): ``dst`` op ``src``."""
    if mode == "Copy":
        return src.copy()
    rgb = dst.ndim == 3
    d = dst.astype(np.float64)
    s = src.astype(np.float64)
    if mode == "Blend" or mode == "Average":
        r = (d + s) / 2
    elif mode == "Difference":
        r = np.abs(d - s)
    elif mode in ("Transparent-white", "Transparent-zero"):
        key = 255 if mode == "Transparent-white" else 0
        if kind == GRAY16 and mode == "Transparent-white":
            key = 65535
        hit = np.all(src == key, axis=-1, keepdims=True) if rgb else src == key
        r = np.where(hit, d, s)
    elif mode in ("AND", "OR", "XOR"):
        if kind == GRAY32:
            return dst.copy()
        a, b = dst.astype(np.int64), src.astype(np.int64)
        r = (a & b) if mode == "AND" else (a | b) if mode == "OR" else (a ^ b)
    elif mode == "Add":
        r = d + s
    elif mode == "Subtract":
        r = d - s
    elif mode == "Multiply":
        r = d * s
    elif mode == "Divide":
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(s == 0, d if kind != GRAY32 else np.inf, d / np.where(s == 0, 1, s))
    elif mode == "Min":
        r = np.minimum(d, s)
    elif mode == "Max":
        r = np.maximum(d, s)
    else:
        r = s
    if kind == GRAY32:
        return np.asarray(r, np.float32)
    top = 65535 if kind == GRAY16 else 255
    if mode in ("Blend", "Average"):
        r = np.floor(r)
    return np.clip(np.round(r) if mode not in ("Blend", "Average") else r, 0, top).astype(
        dst.dtype)


def paste(session, image):
    from rsstatchem.core.imaging.roi import RectRoi
    clip = session.clipboard
    if clip is None:
        raise CommandError("The internal clipboard is empty.")
    src = _convert_clip(clip, image)
    h, w = src.shape[:2]
    roi = image.roi
    if roi is not None and roi.is_area():
        bx, by, bw, bh = roi.bounds()
        if int(round(bw)) == w and int(round(bh)) == h:
            x0, y0 = int(round(bx)), int(round(by))
        else:
            x0 = int(round(bx + (bw - w) / 2))
            y0 = int(round(by + (bh - h) / 2))
    else:
        x0, y0 = (image.width - w) // 2, (image.height - h) // 2
    mode = session.options.extra.get("paste_mode", "Copy")
    p = image.plane()
    xa, ya = max(x0, 0), max(y0, 0)
    xb, yb = min(x0 + w, image.width), min(y0 + h, image.height)
    if xa >= xb or ya >= yb:
        return
    region = src[ya - y0:yb - y0, xa - x0:xb - x0]
    res = blend(p[ya:yb, xa:xb], region, mode, image.type)
    if clip.roi is not None:
        m = clip.roi.get_mask((h, w))[ya - y0:yb - y0, xa - x0:xb - x0]
        cur = p[ya:yb, xa:xb]
        res = np.where(m[..., None] if res.ndim == 3 else m, res, cur)
    p[ya:yb, xa:xb] = res
    if clip.roi is not None:
        r2 = clip.roi.copy()
        r2.translate(x0, y0)
        image.roi = r2
    else:
        image.roi = RectRoi(x0, y0, w, h)


def internal_clipboard(session, image):
    clip = session.clipboard
    if clip is None:
        raise CommandError("The internal clipboard is empty.")
    doc = clip.duplicate("Clipboard")
    doc.roi = None
    return doc


def system_clipboard(session, image):
    """File > New > System Clipboard (Qt clipboard image)."""
    from PySide6.QtGui import QGuiApplication, QImage

    from rsstatchem.core.imaging.io.pillow_io import read_pil_image

    app = QGuiApplication.instance()
    qimg = app.clipboard().image() if app is not None else QImage()
    if qimg is None or qimg.isNull():
        raise CommandError("The system clipboard does not contain an image.")
    qimg = qimg.convertToFormat(QImage.Format.Format_RGB888)
    w, h = qimg.width(), qimg.height()
    ptr = qimg.constBits()
    arr = np.frombuffer(ptr, np.uint8, qimg.bytesPerLine() * h).reshape(h, -1)[:, :w * 3]
    from PIL import Image
    return read_pil_image(Image.fromarray(arr.reshape(h, w, 3).copy()), "Clipboard")


def copy_to_system(session, image):
    from PySide6.QtGui import QGuiApplication, QImage

    from rsstatchem.core.imaging.lut import render
    rgb = np.ascontiguousarray(render(image))
    roi = image.roi
    if roi is not None and roi.is_area():
        x, y, w, h = (int(round(v)) for v in roi.bounds())
        rgb = np.ascontiguousarray(rgb[max(y, 0):y + h, max(x, 0):x + w])
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    app = QGuiApplication.instance()
    if app is not None:
        app.clipboard().setImage(qimg)
    session.status(f"{w}x{h} image copied to the system clipboard")


def paste_control(session, image, transfer="Copy"):
    session.options.extra["paste_mode"] = transfer


# ---- Edit > Options
def _opt_fields(builder):
    return lambda session, image: builder(session.options)


def _colors_fields(o):
    return [C.choice("Foreground:", COLOR_NAMES + ((color_name(o.foreground),)
                                                   if color_name(o.foreground) not in
                                                   COLOR_NAMES else ()),
                     color_name(o.foreground)),
            C.choice("Background:", COLOR_NAMES + ((color_name(o.background),)
                                                   if color_name(o.background) not in
                                                   COLOR_NAMES else ()),
                     color_name(o.background)),
            C.choice("Selection:", COLOR_NAMES + ((color_name(o.selection),)
                                                  if color_name(o.selection) not in
                                                  COLOR_NAMES else ()),
                     color_name(o.selection))]


def colors(session, image, foreground, background, selection):
    o = session.options
    o.foreground = COLOR_HEX.get(foreground.lower(), foreground)
    o.background = COLOR_HEX.get(background.lower(), background)
    o.selection = COLOR_HEX.get(selection.lower(), selection)
    session.emit("options_changed", "colors")


def _appearance_fields(o):
    e = o.extra
    return [C.checkbox("Interpolate zoomed images", o.interpolate_zoomed),
            C.checkbox("Black canvas", e.get("black_canvas", False)),
            C.checkbox("No image border", e.get("no_border", False)),
            C.checkbox("Use inverting lookup table", e.get("inverting_lut", False)),
            C.checkbox("Auto contrast stacks", e.get("auto_contrast", False)),
            C.checkbox("IJ window always on top", e.get("on_top", False)),
            C.choice("16-bit range:", ("Automatic", "8-bit (0-255)", "10-bit (0-1023)",
                                       "12-bit (0-4095)", "15-bit (0-32767)",
                                       "16-bit (0-65535)"), e.get("range16", "Automatic"),
                     key="16-bit"),
            C.number("Menu font size:", e.get("menu_font", 0), 0, "points", key="menu")]


def appearance(session, image, interpolate, black, no, use, auto, ij, **kw):
    o = session.options
    o.interpolate_zoomed = interpolate
    o.extra.update(black_canvas=black, no_border=no, inverting_lut=use, auto_contrast=auto,
                   on_top=ij, range16=kw.get("16-bit", "Automatic"), menu_font=kw.get("menu",
                                                                                      0))
    session.emit("options_changed", "appearance")


def _conversions_fields(o):
    return [C.checkbox("Scale when converting", o.scale_conversions),
            C.checkbox("Weighted RGB conversions", o.weighted_rgb),
            C.checkbox("Full range 16-bit inversions", o.full_range_16bit_inversion,
                       key="full")]


def conversions(session, image, scale, weighted, full):
    o = session.options
    o.scale_conversions, o.weighted_rgb, o.full_range_16bit_inversion = scale, weighted, full


def _io_fields(o):
    e = o.extra
    return [C.number("JPEG quality (0-100):", o.jpeg_quality, 0, key="jpeg", integer=True),
            C.number("GIF and PNG transparent index:", e.get("transparent_index", -1), 0,
                     key="gif", integer=True),
            C.string("File extension for tables:", o.results_extension, key="file"),
            C.checkbox("Use JFileChooser to open/save", e.get("jfilechooser", False),
                       key="use_jfilechooser"),
            C.checkbox("Use_file chooser to import sequences", e.get("seq_chooser", False),
                       key="use_file"),
            C.checkbox("Save TIFF and raw in Intel byte order", o.intel_byte_order,
                       key="intel"),
            C.checkbox("Skip dialog when opening .raw files", e.get("skip_raw", False),
                       key="skip"),
            C.message("Results Table Options"),
            C.checkbox("Copy_column headers", e.get("copy_headers", False), key="copy_column"),
            C.checkbox("Copy_row numbers", e.get("copy_rows", True), key="copy_row"),
            C.checkbox("Save_column headers", o.save_column_headers, key="save_column"),
            C.checkbox("Save_row numbers", o.save_row_numbers, key="save_row")]


def input_output(session, image, jpeg, gif, file, use_jfilechooser, use_file, intel, skip,
                 copy_column, copy_row, save_column, save_row):
    o = session.options
    o.jpeg_quality = max(0, min(100, int(jpeg)))
    ext = file if str(file).startswith(".") else "." + str(file)
    o.results_extension = ext
    o.intel_byte_order = intel
    o.save_column_headers, o.save_row_numbers = save_column, save_row
    o.extra.update(transparent_index=int(gif), jfilechooser=use_jfilechooser,
                   seq_chooser=use_file, skip_raw=skip, copy_headers=copy_column,
                   copy_rows=copy_row)


def _line_width_fields(o):
    return [C.number("Line width:", o.line_width, 0, "pixels", key="line", minimum=1,
                     maximum=500)]


def line_width(session, image, line):
    session.options.line_width = max(1.0, float(line))
    roi = image.roi if image is not None else None
    if roi is not None and roi.is_line():
        roi.stroke_width = session.options.line_width if line > 1 else 0.0
        if hasattr(roi, "width"):
            roi.width = session.options.line_width
        session.image_updated(image)
    session.emit("options_changed", "line_width")


def _misc_fields(o):
    e = o.extra
    return [C.string("Divide by zero value:", _inf_text(o.divide_by_zero), key="divide"),
            C.checkbox("Use pointer cursor", e.get("pointer_cursor", False), key="use"),
            C.checkbox("Hide \"Process Stack?\" dialog", e.get("hide_stack_dialog", False),
                       key="hide"),
            C.checkbox("Require control/command key for shortcuts", o.require_control_key,
                       key="require"),
            C.checkbox("Move isolated plugins to Misc. menu", o.move_isolated, key="move"),
            C.checkbox("Run single instance listener", e.get("single_instance", False),
                       key="run"),
            C.checkbox("Debug mode", e.get("debug", False), key="debug")]


def _inf_text(v: float) -> str:
    if math.isinf(v):
        return "Infinity" if v > 0 else "-Infinity"
    if v != v:
        return "NaN"
    return C.format_number(v, 4)


def misc(session, image, divide, use, hide, require, move, run, debug):
    o = session.options
    try:
        o.divide_by_zero = float(str(divide).replace("Infinity", "inf").replace("max",
                                                                                  "3.4e38"))
    except ValueError:
        raise CommandError(f'Invalid divide by zero value: "{divide}"') from None
    o.require_control_key, o.move_isolated = require, move
    o.extra.update(pointer_cursor=use, hide_stack_dialog=hide, single_instance=run,
                   debug=debug)
    session.emit("options_changed", "misc")


ARROW_STYLES = ("Filled", "Notched", "Open", "Headless", "Bar")


def _arrow_fields(o):
    e = o.extra
    return [C.slider("Width:", 1, 50, e.get("arrow_width", 2)),
            C.slider("Size:", 0, 30, e.get("arrow_size", 10)),
            C.choice("Color:", COLOR_NAMES, e.get("arrow_color", "yellow")),
            C.choice("Style:", ARROW_STYLES, e.get("arrow_style", "Filled")),
            C.checkbox("Outline", e.get("arrow_outline", False)),
            C.checkbox("Double head", e.get("arrow_double", False), key="double"),
            C.checkbox("Keep after adding to overlay", e.get("arrow_keep", False),
                       key="keep")]


def arrow_tool(session, image, width, size, color, style, outline, double, keep):
    session.options.extra.update(arrow_width=width, arrow_size=size, arrow_color=color,
                                 arrow_style=style, arrow_outline=outline,
                                 arrow_double=double, arrow_keep=keep)


POINT_TYPES = ("Hybrid", "Cross", "Dot", "Circle")
POINT_SIZES = ("Tiny", "Small", "Medium", "Large", "Extra Large", "XXL", "XXXL")


def _point_fields(o):
    return [C.choice("Type:", POINT_TYPES, o.point_type),
            C.choice("Color:", COLOR_NAMES, o.point_color.lower() if o.point_color.lower() in
                     COLOR_NAMES else "yellow"),
            C.choice("Size:", POINT_SIZES, o.point_size),
            C.checkbox("Label points", o.point_label, key="label"),
            C.checkbox("Show on all slices", o.extra.get("point_all_slices", False),
                       key="show"),
            C.checkbox("Auto-measure", o.point_auto_measure, key="auto-measure"),
            C.checkbox("Add to ROI Manager", o.point_add_to_manager, key="add"),
            C.number("Counter:", o.extra.get("point_counter", 0), 0, key="counter",
                     integer=True)]


def point_tool(session, image, type, color, size, label, show, add, counter, **kw):
    o = session.options
    o.point_type, o.point_color, o.point_size = type, color, size
    o.point_label, o.point_add_to_manager = label, add
    o.point_auto_measure = bool(kw.get("auto-measure", False))
    o.extra.update(point_all_slices=show, point_counter=int(counter))
    session.emit("options_changed", "point")


def _roi_defaults_fields(o):
    e = o.extra
    return [C.choice("Color:", COLOR_NAMES, color_name(o.selection) if color_name(o.selection)
                     in COLOR_NAMES else "yellow"),
            C.number("Stroke width:", e.get("roi_stroke_width", 0), 0, key="stroke"),
            C.number("Group:", e.get("roi_group", 0), 0, integer=True),
            C.string("Name:", e.get("roi_name", ""))]


def roi_defaults(session, image, color, stroke, group, name):
    o = session.options
    o.selection = COLOR_HEX.get(color.lower(), color)
    o.extra.update(roi_stroke_width=stroke, roi_group=group, roi_name=name)
    session.emit("options_changed", "colors")


def _rounded_fields(o):
    e = o.extra
    return [C.number("Stroke width:", e.get("rrect_stroke", 2), 0, key="stroke"),
            C.slider("Corner diameter:", 0, 200, o.rounded_rect_diameter, key="corner"),
            C.choice("Color:", COLOR_NAMES, e.get("rrect_color", "yellow")),
            C.checkbox("Fill", e.get("rrect_fill", False))]


def rounded_rect(session, image, stroke, corner, color, fill):
    o = session.options
    o.rounded_rect_diameter = int(corner)
    o.extra.update(rrect_stroke=stroke, rrect_color=color, rrect_fill=fill)
    roi = image.roi if image is not None else None
    from rsstatchem.core.imaging.roi import RectRoi
    if isinstance(roi, RectRoi):
        roi.corner_diameter = float(corner)
        session.image_updated(image)


WAND_MODES = ("Legacy", "4-connected", "8-connected")


def _wand_fields(o):
    return [C.radio("Mode:", WAND_MODES, o.wand_mode, columns=3),
            C.number("Tolerance:", o.wand_tolerance, 1),
            C.checkbox("Smooth if thresholded", o.wand_smooth, key="smooth")]


def wand_tool(session, image, mode, tolerance, smooth):
    o = session.options
    o.wand_mode, o.wand_tolerance, o.wand_smooth = mode, float(tolerance), smooth


def _fonts_fields(o):
    e = o.extra
    return [C.choice("Font:", ("SansSerif", "Serif", "Monospaced", "Arial", "Helvetica",
                               "Times New Roman", "Courier New"), e.get("font.name",
                                                                        "SansSerif")),
            C.number("Size:", e.get("font.size", 18), 0, integer=True),
            C.choice("Style:", ("Plain", "Bold", "Italic", "Bold+Italic"),
                     e.get("font.style", "Plain")),
            C.choice("Justification:", ("Left", "Center", "Right"),
                     e.get("font.justification", "Left")),
            C.checkbox("Antialiased text", e.get("font.antialiased", True), key="antialiased")]


def fonts(session, image, font, size, style, justification, antialiased):
    session.options.extra.update({"font.name": font, "font.size": int(size),
                                  "font.style": style, "font.justification": justification,
                                  "font.antialiased": antialiased})
    session.emit("options_changed", "fonts")


def _plots_fields(o):
    e = o.extra
    return [C.number("Width:", e.get("plot.width", 530), 0, "pixels", integer=True),
            C.number("Height:", e.get("plot.height", 300), 0, "pixels", integer=True),
            C.number("Font size:", e.get("plot.font", 12), 0, key="font", integer=True),
            C.checkbox("Draw grid lines", e.get("plot.grid", True), key="draw"),
            C.checkbox("Draw_ticks", e.get("plot.ticks", True), key="draw_ticks"),
            C.checkbox("Auto-close", e.get("plot.autoclose", False), key="auto-close"),
            C.checkbox("List values", e.get("plot.list", False), key="list"),
            C.checkbox("Interpolate line profiles", e.get("plot.interpolate", True),
                       key="interpolate"),
            C.checkbox("Sub-pixel resolution", e.get("plot.subpixel", False), key="sub-pixel")]


def plots(session, image, width, height, font, draw, draw_ticks, list, interpolate, **kw):
    session.options.extra.update({"plot.width": int(width), "plot.height": int(height),
                                  "plot.font": int(font), "plot.grid": draw,
                                  "plot.ticks": draw_ticks,
                                  "plot.autoclose": bool(kw.get("auto-close")),
                                  "plot.list": list, "plot.interpolate": interpolate,
                                  "plot.subpixel": bool(kw.get("sub-pixel"))})


def _dicom_fields(o):
    e = o.extra
    return [C.checkbox("Open as 32-bit float", e.get("dicom.float", False), key="open"),
            C.checkbox("Ignore Rescale Slope", e.get("dicom.ignore_slope", False),
                       key="ignore"),
            C.checkbox("Fixed DICOM scaling", e.get("dicom.fixed", False), key="fixed"),
            C.checkbox("Rotate DICOM images", e.get("dicom.rotate", False), key="rotate")]


def dicom_options(session, image, open, ignore, fixed, rotate):
    session.options.extra.update({"dicom.float": open, "dicom.ignore_slope": ignore,
                                  "dicom.fixed": fixed, "dicom.rotate": rotate})


def _startup_fields(o):
    return [C.string("Startup macro:", o.extra.get("startup_macro", ""), key="code",
                     columns=60)]


def startup(session, image, code):
    session.options.extra["startup_macro"] = code


def fresh_start(session, image):
    """ImageJ's Fresh Start: default options (black background on, as 1.54's Fresh Start),
    empty ROI Manager and Results."""
    from rsstatchem.core.imaging.options import Options
    keep = {k: v for k, v in session.options.extra.items() if k == "startup_macro"}
    new = Options()
    new.black_background = True
    new.extra.update(keep)
    _replace_options(session, new)
    session.roi_manager.clear()
    session.results.reset()
    session.emit("options_changed", "all")


def reset_options(session, image):
    from rsstatchem.core.imaging.options import Options
    _replace_options(session, Options())
    session.emit("options_changed", "all")


def _replace_options(session, new) -> None:
    for name, value in vars(new).items():
        setattr(session.options, name, value)


def register(registry) -> None:
    add = registry.add
    add("Undo", undo, needs_image=False, undoable=False, recordable=False)
    add("Cut", cut, roi="optional")
    add("Copy", lambda s, im: copy(s, im), undoable=False)
    add("Copy to System", copy_to_system, undoable=False)
    add("Paste", paste)
    add("Paste Control...", paste_control,
        lambda s, im: [C.choice("Transfer Mode:", PASTE_MODES,
                                s.options.extra.get("paste_mode", "Copy"), key="transfer")],
        needs_image=False, undoable=False)
    add("Clear", clear, roi="required", supports_stack=True)
    add("Clear Outside", clear_outside, roi="area", supports_stack=True)
    add("Fill", fill, roi="required", supports_stack=True)
    add("Draw", draw, roi="required", supports_stack=True)
    add("Invert", invert, supports_stack=True)
    opt = {"needs_image": False, "undoable": False}
    add("Appearance...", appearance, _opt_fields(_appearance_fields), **opt)
    add("Arrow Tool...", arrow_tool, _opt_fields(_arrow_fields), **opt)
    add("Colors...", colors, _opt_fields(_colors_fields), **opt)
    add("Conversions...", conversions, _opt_fields(_conversions_fields), **opt)
    add("DICOM...", dicom_options, _opt_fields(_dicom_fields), **opt)
    add("Fonts...", fonts, _opt_fields(_fonts_fields), **opt)
    add("Input/Output...", input_output, _opt_fields(_io_fields), **opt)
    add("Line Width...", line_width, _opt_fields(_line_width_fields), **opt)
    add("Misc...", misc, _opt_fields(_misc_fields), **opt)
    add("Plots...", plots, _opt_fields(_plots_fields), **opt)
    add("Point Tool...", point_tool, _opt_fields(_point_fields), **opt)
    add("Roi Defaults...", roi_defaults, _opt_fields(_roi_defaults_fields), **opt)
    add("Rounded Rect Tool...", rounded_rect, _opt_fields(_rounded_fields), **opt)
    add("Startup...", startup, _opt_fields(_startup_fields), **opt)
    add("Wand Tool...", wand_tool, _opt_fields(_wand_fields), **opt)
    add("Fresh Start", fresh_start, **opt)
    add("Reset... ", reset_options, **opt)


def selftest() -> None:
    from rsstatchem.core.imaging.roi import RectRoi
    from rsstatchem.core.imaging.session import Session

    s = Session()
    register(s.registry)
    im = ImageDoc.from_array(np.zeros((10, 10), np.uint8))
    s.add_image(im)
    im.roi = RectRoi(2, 2, 3, 3)
    s.run("Fill", "slice")
    assert im.pixels.sum() == 9 * 255
    s.run("Invert")
    assert im.raw_value(3, 3) == 0 and im.raw_value(0, 0) == 0
    s.run("Undo")
    assert im.raw_value(3, 3) == 255


__all__ = ["register", "selftest", "color_value", "fill_mask", "draw_roi_outline",
           "invert_plane", "blend", "line_points"]

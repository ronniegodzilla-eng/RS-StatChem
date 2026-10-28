"""Image Analysis commands: Image menu (types, adjust, color, stacks, hyperstacks,
crop/duplicate/scale, transform, zoom, overlay, lookup tables) except the thresholds. **Owned by
WP-D.**

``register(registry)`` adds this area's commands (labels exactly as in
``rsstatchem.core.imaging.menu_layout.labels_of_area("image")``) plus the Fiji-style extras
"Bleach Correction", "Colour Deconvolution" and "KymographBuilder" (not in the ImageJ menus;
reachable through Find Commands and macros). Field labels follow ImageJ's dialogs so that macro
option strings recorded by ImageJ run unchanged (``run("Scale...", "x=0.5 y=0.5
interpolation=Bilinear average create")``). The numerical work lives in ``core/imaging``
(convert, transform, stacks, project3d, adjust, color, lut_tables, fiji/bleach, fiji/kymograph);
windows (B&C, Channels Tool, Orthogonal Views) in ``ui/imagej/frames``. Optional ``selftest()``
(no Qt widgets) is run by ``RS-StatChem --selftest``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rsstatchem.core.imaging import commands as C
from rsstatchem.core.imaging import lut_tables  # noqa: F401 - registers the named LUTs
from rsstatchem.core.imaging.commands import Command, CommandError
from rsstatchem.core.imaging.image import (COLOR_256, COLOR_RGB, GRAY8, GRAY16, GRAY32,
                                           ImageDoc)
from rsstatchem.core.imaging.lut import LUT, named_lut
from rsstatchem.core.imaging.roi.shapes import RECTANGLE, Roi

GRAY = (GRAY8, GRAY16, GRAY32)
ALL = (GRAY8, GRAY16, GRAY32, COLOR_256, COLOR_RGB)
NONE_ITEM = "*None*"


# ---------------------------------------------------------------------------------- helpers
def _area_rect(image: ImageDoc):
    """Integer bounds (x, y, w, h) of the area selection clipped to the image, or None."""
    roi = image.roi
    if roi is None or not roi.is_area():
        return None
    x, y, w, h = roi.bounds()
    x0, y0 = max(int(math.floor(x)), 0), max(int(math.floor(y)), 0)
    x1, y1 = min(int(math.ceil(x + w)), image.width), min(int(math.ceil(y + h)), image.height)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1 - x0, y1 - y0


def _nonrect_mask(image: ImageDoc):
    from rsstatchem.core.imaging.roi import RectRoi
    roi = image.roi
    if roi is None or not roi.is_area() or (isinstance(roi, RectRoi) and
                                            not roi.corner_diameter):
        return None
    return roi.get_mask((image.height, image.width))


def _kind(image: ImageDoc) -> str:
    from rsstatchem.core.imaging.transform import kind_of
    return kind_of(image)


def _process_planes(image: ImageDoc, fn, stack: bool, masked: bool = True) -> None:
    """Apply ``fn(plane) -> plane`` (same shape) to the current or all planes, restoring the
    pixels outside a non-rectangular selection (ImageJ SUPPORTS_MASKING)."""
    mask = _nonrect_mask(image) if masked else None
    for c, z, t in image.iter_planes(stack):
        p = image.plane(c, z, t)
        new = fn(p)
        if mask is not None:
            m = mask[..., None] if p.ndim == 3 else mask
            new = np.where(m, new, p)
        image.set_plane(new, c, z, t)


def _stack_field(image) -> list:
    if image is not None and image.stack_size > 1:
        return [C.checkbox(f"Process all {image.stack_size} slices", False, key="stack")]
    return []


def _titles(session, exclude=None, none=False) -> list[str]:
    titles = [im.title for im in session.images if im is not exclude]
    return ([NONE_ITEM] if none else []) + titles


def _find(session, title: str) -> ImageDoc:
    im = session.find_image(title)
    if im is None:
        raise CommandError(f'Image not found: "{title}"')
    return im


def _parse(text, default=0.0) -> float:
    try:
        return float(str(text).strip().replace("Infinity", "inf"))
    except ValueError:
        return default


def _new_title(title: str, suffix: str = "-1") -> str:
    stem, dot, ext = title.rpartition(".")
    if dot and 0 < len(ext) <= 5:
        return f"{stem}{suffix}.{ext}"
    return title + suffix


def _bg_color(session, image, fill: bool):
    from rsstatchem.core.imaging.transform import background_value
    return background_value(image, session.options.background, fill)


def _window(image):
    try:
        from rsstatchem.ui.imagej.windows import window_for
        return window_for(image)
    except Exception:  # noqa: BLE001 - headless
        return None


def _noop(session, image, **_):
    return None


def _ui_only(fn):
    """Command callable from the menu (ui_action) that does nothing in batch macros."""
    return fn


# ---------------------------------------------------------------------------------- Type
def _type_command(item: str):
    def run(session, image, number=256, **_):
        from rsstatchem.core.imaging import convert as CV
        opts = session.options
        try:
            result = CV.convert(image, item, opts.scale_conversions, CV.rgb_weights(opts),
                                int(number))
        except CV.ConversionError as exc:
            raise CommandError(str(exc)) from None
        image.roi = image.roi
        if result is not None:              # Lab Stack: a new window replaces the source
            result.title = image.title
            session.close_image(image)
            return result
        return None
    return run


def _color_count_fields(session, image):
    if image is not None and image.type == COLOR_RGB:
        return [C.number("Number of Colors (2-256):", 256, 0, key="number", integer=True)]
    return []


# ---------------------------------------------------------------------------------- Adjust
def _size_fields(session, image):
    w = image.width if image else 100
    h = image.height if image else 100
    fields = [C.number("Width (pixels):", w, 0, key="width", integer=True),
              C.number("Height (pixels):", h, 0, key="height", integer=True)]
    if image is not None and image.stack_size > 1:
        if image.n_frames > 1 and image.n_slices == 1:
            fields.append(C.number("Time (frames):", image.n_frames, 0, key="time",
                                   integer=True))
        else:
            fields.append(C.number("Depth (images):", image.n_slices, 0, key="depth",
                                   integer=True))
    fields += [C.checkbox("Constrain aspect ratio", True, key="constrain"),
               C.checkbox("Average when downsizing", True, key="average"),
               C.choice("Interpolation:", ("None", "Bilinear", "Bicubic"), "Bilinear")]
    return fields


def size_cmd(session, image, width, height, constrain=False, average=True,
             interpolation="Bilinear", depth=None, time=None, **_):
    """Image > Adjust > Size (Resizer)."""
    from rsstatchem.core.imaging import transform as T
    rect = _area_rect(image)
    ow, oh = (rect[2], rect[3]) if rect else (image.width, image.height)
    w, h = int(width), int(height)
    if constrain:
        h = int(round(w * (oh / ow)))
    if w <= 0:
        w = 50
    if h <= 0:
        h = 50
    method = T.method_index(interpolation)
    kind = _kind(image)
    if rect is not None or w != ow or h != oh:
        def fn(p):
            return T.resize_plane(p, kind, w, h, method, rect, average)
        _replace_all_planes(image, fn)
        cal = image.calibration
        if cal.scaled():
            cal.pixel_width *= ow / w
            cal.pixel_height *= oh / h
        if image.overlay is not None:
            _scale_overlay(image.overlay, w / ow, h / oh)
        image.roi = None
    target = depth if depth is not None else time
    current = image.n_slices if depth is not None else image.n_frames
    if target is not None and int(target) != current and int(target) > 0:
        _resize_z(image, int(target), method, average, axis_t=depth is None)


def _replace_all_planes(image: ImageDoc, fn) -> None:
    from rsstatchem.core.imaging.transform import map_planes
    luts = [lut.copy() for lut in image.luts]
    ranges = list(image.display_ranges)
    labels = list(image.slice_labels)
    mode = image.composite_mode
    pix = map_planes(image, fn)
    image.replace_pixels(pix, image.type)
    if len(luts) == image.n_channels:
        image.luts = luts
        image.composite_mode = mode
    image.slice_labels = labels
    if image.bit_depth in (16, 32):
        image.display_ranges = ranges


def _resize_z(image: ImageDoc, depth2: int, method: int, average: bool, axis_t=False) -> None:
    """Resizer.zScale/resizeZ: resize the xz planes (per channel/frame) to the new depth."""
    from rsstatchem.core.imaging import transform as T
    kind = _kind(image)
    p = image.pixels
    if axis_t:
        p = np.swapaxes(p, 0, 1)                    # (Z, T, C, Y, X)
    nz = p.shape[1]
    out_shape = (p.shape[0], depth2) + p.shape[2:]
    out = np.zeros(out_shape, dtype=p.dtype)
    for t in range(p.shape[0]):
        for c in range(p.shape[2]):
            vol = p[t, :, c]                         # (Z, Y, X[,3])
            for y in range(vol.shape[1]):
                xz = np.ascontiguousarray(vol[:, y])
                out[t, :, c, y] = T.resize_plane(xz, kind, xz.shape[1], depth2, method, None,
                                                 average)
    if axis_t:
        out = np.swapaxes(out, 0, 1)
    luts = [lut.copy() for lut in image.luts]
    ranges = list(image.display_ranges)
    image.replace_pixels(np.ascontiguousarray(out), image.type)
    image.luts = luts if len(luts) == image.n_channels else image.luts
    image.display_ranges = ranges if len(ranges) == image.n_channels else image.display_ranges
    if image.calibration.scaled() and not axis_t:
        image.calibration.pixel_depth *= nz / depth2


def _scale_overlay(overlay, sx: float, sy: float) -> None:
    from rsstatchem.core.imaging.roi import shapes as S
    for roi in overlay:
        if isinstance(roi, (S.RectRoi, S.OvalRoi)):
            roi.x *= sx
            roi.y *= sy
            roi.width *= sx
            roi.height *= sy
        elif isinstance(roi, (S.PolygonRoi, S.PointRoi)):
            roi.xs = roi.xs * sx
            roi.ys = roi.ys * sy
        elif isinstance(roi, S.LineRoi):
            roi.x1 *= sx
            roi.x2 *= sx
            roi.y1 *= sy
            roi.y2 *= sy
        elif isinstance(roi, S.TextRoi):
            roi.x *= sx
            roi.y *= sy


def _canvas_fields(session, image):
    from rsstatchem.core.imaging.transform import CANVAS_POSITIONS
    w = image.width if image else 100
    h = image.height if image else 100
    return [C.number("Width:", w, 0, units="pixels", integer=True),
            C.number("Height:", h, 0, units="pixels", integer=True),
            C.choice("Position:", CANVAS_POSITIONS, "Center"),
            C.checkbox("Zero Fill", False, key="zero")]


def canvas_cmd(session, image, width, height, position="Center", zero=False, **_):
    from rsstatchem.core.imaging import transform as T
    w, h = int(width), int(height)
    xo, yo = T.canvas_offsets(image.width, image.height, w, h, position)
    fill = 0 if zero else _bg_color(session, image, True)
    if isinstance(fill, tuple):
        fill = np.array(fill, dtype=np.uint8)
    _replace_all_planes(image, lambda p: T.canvas_plane(p, w, h, xo, yo, fill))
    if image.overlay is not None:
        for roi in image.overlay:
            roi.translate(xo, yo)
    image.roi = None


def line_width_cmd(session, image, line=1.0, **_):
    session.options.line_width = float(line)
    roi = image.roi if image is not None else None
    if roi is not None and roi.is_line():
        if hasattr(roi, "width"):
            roi.width = float(line)
        roi.stroke_width = float(line)


def _coord_fields(session, image):
    cal = image.calibration if image else None
    w, h = (image.width, image.height) if image else (1, 1)
    left = -cal.x_origin * cal.pixel_width if cal else 0.0
    top = -cal.y_origin * cal.pixel_height if cal else 0.0
    fields = [C.number("Left:", left, 4), C.number("Right:", left + w * cal.pixel_width
                                                    if cal else w, 4),
              C.number("Top:", top, 4), C.number("Bottom:", top + h * cal.pixel_height
                                                  if cal else h, 4)]
    if image is not None and image.n_slices > 1:
        front = -cal.z_origin * cal.pixel_depth
        fields += [C.number("Front:", front, 4),
                   C.number("Back:", front + image.n_slices * cal.pixel_depth, 4)]
    fields += [C.string("Unit:", cal.unit if cal else "pixel"),
               C.string("Y_unit:", cal.y_unit or "" if cal else ""),
               C.string("Z_unit:", cal.z_unit or "" if cal else "")]
    return fields


def coordinates_cmd(session, image, left=0.0, right=None, top=0.0, bottom=None, front=None,
                    back=None, unit="pixel", y_unit="", z_unit="", **_):
    """Image > Adjust > Coordinates: calibrate so the image spans left..right, top..bottom."""
    cal = image.calibration
    right = image.width if right is None else right
    bottom = image.height if bottom is None else bottom
    pw = (right - left) / image.width
    ph = (bottom - top) / image.height
    cal.invert_y = ph < 0
    cal.pixel_width, cal.pixel_height = abs(pw) or 1.0, abs(ph) or 1.0
    cal.x_origin = -left / pw if pw else 0.0
    cal.y_origin = -top / ph if ph else 0.0
    if front is not None and back is not None and image.n_slices > 0:
        pd = (back - front) / image.n_slices
        cal.pixel_depth = abs(pd) or 1.0
        cal.z_origin = -front / pd if pd else 0.0
    cal.unit = unit or "pixel"
    cal.y_unit = y_unit or None
    cal.z_unit = z_unit or None


# ---------------------------------------------------------------------------------- Info/props
def image_info(image: ImageDoc, session=None) -> str:
    """Image > Show Info text (ImageInfo layout, abbreviated)."""
    cal = image.calibration
    lines = []
    if image.info:
        lines += [image.info.rstrip(), "-" * 20]
    lines.append(f"Title: {image.title}")
    if cal.scaled():
        lines.append(f"Width:  {image.width * cal.pixel_width:.4g} {cal.units} ({image.width})")
        lines.append(f"Height:  {image.height * cal.pixel_height:.4g} {cal.units} "
                     f"({image.height})")
        if image.n_slices > 1:
            lines.append(f"Depth:  {image.n_slices * cal.pixel_depth:.4g} {cal.units} "
                         f"({image.n_slices})")
        lines.append(f"Resolution:  {1.0 / cal.pixel_width:.4g} pixels per {cal.unit}")
        lines.append(f"Voxel size: {cal.pixel_width:.4g}x{cal.pixel_height:.4g}x"
                     f"{cal.pixel_depth:.4g} {cal.unit}^3")
    else:
        lines.append(f"Width:  {image.width} pixels")
        lines.append(f"Height:  {image.height} pixels")
        if image.n_slices > 1:
            lines.append(f"Depth:  {image.n_slices} pixels")
    lines.append(f"Size:  {image.size_bytes() // 1024}K")
    lines.append(f"ID: {image.id}")
    kind = {GRAY8: "8-bit", GRAY16: "16-bit (unsigned)", GRAY32: "32-bit (float)",
            COLOR_256: "8-bit (color LUT)", COLOR_RGB: "RGB"}[image.type]
    lines.append(f"Bits per pixel: {kind}")
    if image.type != COLOR_RGB:
        lo, hi = image.display_range
        lines.append(f"Display range: {lo:g}-{hi:g}")
        lut = image.lut
        lines.append(f"LUT: {lut.name or ('inverting' if lut.is_inverted() else 'grays')}")
    c, z, t = image.dims
    if image.stack_size > 1:
        if c > 1:
            lines.append(f"  Channels: {c}")
        if z > 1:
            lines.append(f"  Slices: {z}")
        if t > 1:
            lines.append(f"  Frames: {t}")
        lines.append(f"  Image: {image.current_index + 1}/{image.stack_size}")
        label = image.slice_label()
        if label:
            lines.append(f"  Label: {label.splitlines()[0]}")
    if cal.frame_interval:
        lines.append(f"Frame interval: {cal.frame_interval:g} {cal.time_unit}")
    if cal.calibrated():
        lines.append(f"Calibration function: {cal.function} {tuple(cal.coefficients)}")
        lines.append(f"  Unit: \"{cal.value_unit}\"")
    else:
        lines.append("Uncalibrated")
    if cal.x_origin or cal.y_origin:
        lines.append(f"Coordinate origin:  {cal.x_origin:g},{cal.y_origin:g}")
    if image.path:
        lines.append(f"Path: {image.path}")
    if image.roi is not None:
        x, y, w, h = image.roi.bounds()
        lines.append(f"{image.roi.type_name} Selection")
        lines.append(f"  X: {x:g}\n  Y: {y:g}\n  Width: {w:g}\n  Height: {h:g}")
    else:
        lines.append("No selection")
    if image.overlay is not None and len(image.overlay):
        lines.append(f"Overlay: {len(image.overlay)} elements"
                     + (" (hidden)" if image.overlay.hidden else ""))
    for k, v in image.properties.items():
        lines.append(f"{k}: {v}")
    return "\n".join(lines) + "\n"


def show_info_cmd(session, image, **_):
    session.log(image_info(image, session))


def _show_info_ui(bar, cmd) -> None:
    image = bar.session.active
    if image is None:
        bar.error("There are no images open")
        return
    from rsstatchem.ui.imagej import windows as W
    win = W.TextWindow(f"Info for {image.title}", image_info(image, bar.session))
    W.register(win)
    win.show()


def _properties_fields(session, image):
    cal = image.calibration
    c, z, t = image.dims
    return [C.number("Channels (c):", c, 0, key="channels", integer=True),
            C.number("Slices (z):", z, 0, key="slices", integer=True),
            C.number("Frames (t):", t, 0, key="frames", integer=True),
            C.message("Note: c*z*t must equal " + str(image.stack_size)),
            C.string("Unit of length:", cal.unit, key="unit"),
            C.number("Pixel_width:", cal.pixel_width, 7, key="pixel_width"),
            C.number("Pixel_height:", cal.pixel_height, 7, key="pixel_height"),
            C.number("Voxel_depth:", cal.pixel_depth, 7, key="voxel_depth"),
            C.number("Frame interval:", cal.frame_interval, 4, key="frame"),
            C.string("Time unit:", cal.time_unit, key="time"),
            C.string("Origin (pixels):", f"{cal.x_origin:g},{cal.y_origin:g}", key="origin"),
            C.checkbox("Invert Y coordinates", cal.invert_y, key="invert"),
            C.checkbox("Global", False, key="global")]


def properties_cmd(session, image, channels, slices, frames, unit="pixel", pixel_width=1.0,
                   pixel_height=1.0, voxel_depth=1.0, frame=0.0, time="sec", origin="0,0",
                   invert=False, **_):
    """Image > Properties: dimensions (c*z*t must equal the stack size) and calibration."""
    c, z, t = int(channels), int(slices), int(frames)
    if (c, z, t) != image.dims:
        if c * z * t != image.stack_size:
            raise CommandError(f"channels x slices x frames <> stack size "
                               f"({image.stack_size})")
        from rsstatchem.core.imaging.stacks import stack_to_hyperstack
        stack_to_hyperstack(image, c, z, t, "xyczt(default)",
                            image.composite_mode if image.n_channels > 1 else "Color")
        image.hyperstack = (c > 1) + (z > 1) + (t > 1) > 1 or c > 1
    cal = image.calibration
    cal.unit = unit or "pixel"
    if cal.unit in ("pixel", "pixels"):
        cal.unit = "pixel"
    cal.pixel_width = float(pixel_width) or 1.0
    cal.pixel_height = float(pixel_height) or 1.0
    cal.pixel_depth = float(voxel_depth) or 1.0
    cal.frame_interval = float(frame)
    cal.time_unit = time or "sec"
    parts = [p for p in str(origin).replace(" ", "").split(",") if p]
    if parts:
        cal.x_origin = _parse(parts[0])
        cal.y_origin = _parse(parts[1]) if len(parts) > 1 else cal.x_origin
        if len(parts) > 2:
            cal.z_origin = _parse(parts[2])
    cal.invert_y = bool(invert)


# ---------------------------------------------------------------------------------- Color
def split_channels_cmd(session, image, **_):
    from rsstatchem.core.imaging.color import split_channels
    try:
        parts = split_channels(image)
    except ValueError as exc:
        raise CommandError(str(exc)) from None
    session.close_image(image)
    return parts


def _merge_fields(session, image):
    from rsstatchem.core.imaging.color import SLOT_LABELS
    titles = _titles(session, none=True)
    fields = []
    for i, label in enumerate(SLOT_LABELS):
        fields.append(C.choice(label, titles, NONE_ITEM, key=f"c{i + 1}"))
    fields += [C.checkbox("Create composite", True, key="create"),
               C.checkbox("Keep source images", False, key="keep"),
               C.checkbox("Ignore source LUTs", False, key="ignore")]
    return fields


def merge_channels_cmd(session, image, create=False, keep=False, ignore=False, **values):
    from rsstatchem.core.imaging.color import merge_channels
    slots = []
    for i in range(7):
        t = values.get(f"c{i + 1}", NONE_ITEM)
        slots.append(None if t in (NONE_ITEM, "", None) else _find(session, t))
    sources = [im for im in slots if im is not None]
    if len(sources) < 1:
        raise CommandError("No source images")
    try:
        out = merge_channels(slots, create_composite=create, ignore_luts=ignore,
                             title="Composite" if create else "RGB")
    except ValueError as exc:
        raise CommandError(str(exc)) from None
    if not keep:
        for im in sources:
            session.close_image(im)
    return out


def arrange_cmd(session, image, new="", **_):
    from rsstatchem.core.imaging.color import arrange_channels
    if not new:
        return None
    try:
        arrange_channels(image, new)
    except ValueError as exc:
        raise CommandError(str(exc)) from None


def invert_luts_cmd(session, image, **_):
    from rsstatchem.core.imaging.color import invert_luts
    invert_luts(image)


def _stack_to_rgb_fields(session, image):
    fields = []
    if image is not None and image.n_slices > 1:
        fields.append(C.checkbox("Slices", True))
    if image is not None and image.n_frames > 1:
        fields.append(C.checkbox("Frames", True))
    if fields:          # ImageJ asks only for stacks; a single composite is always kept
        fields.append(C.checkbox("Keep source", True, key="keep"))
    return fields


def stack_to_rgb_cmd(session, image, slices=True, frames=True, keep=True, **_):
    from rsstatchem.core.imaging.color import stack_to_rgb
    if image.type == COLOR_RGB:
        raise CommandError("Composite image or 2-3 slice stack required")
    if not image.is_composite and image.stack_size not in (2, 3):
        raise CommandError("Composite image or 2-3 slice stack required")
    out = stack_to_rgb(image, slices, frames)
    if not keep:
        session.close_image(image)
    return out


def make_composite_cmd(session, image, display="Composite", **_):
    from rsstatchem.core.imaging.color import make_composite
    try:
        make_composite(image, display.lower())
    except ValueError as exc:
        raise CommandError(str(exc)) from None


def show_lut_cmd(session, image, **_):
    from rsstatchem.core.imaging.color import show_lut_image
    if image.type == COLOR_RGB:
        raise CommandError("RGB images do not have a LUT")
    return show_lut_image(image.lut, f"{image.title} LUT")


def display_luts_cmd(session, image, **_):
    from rsstatchem.core.imaging.color import display_luts
    return display_luts()


def _edit_lut_ui(bar, cmd) -> None:
    image = bar.session.active
    if image is None or image.type == COLOR_RGB:
        bar.error("8-bit, 16-bit or 32-bit image with a LUT required", "Edit LUT")
        return
    from rsstatchem.ui.imagej.dialogs.image_lut_editor import LutEditor
    dlg = LutEditor(image.lut, bar)
    if dlg.exec():
        image.snapshot("Edit LUT...")
        luts = list(image.luts)
        luts[image.position[0]] = dlg.lut()
        image.luts = luts
        bar.session.image_updated(image)


def _color_picker_ui(bar, cmd) -> None:
    try:
        from rsstatchem.ui.imagej.frames import color_picker
        factory = getattr(color_picker, "ColorPicker", None) or \
            getattr(color_picker, "show", None)
    except ImportError:
        factory = None
    if factory is not None:
        win = factory(bar) if callable(factory) else None
        if win is not None and hasattr(win, "show"):
            from rsstatchem.ui.imagej import windows as W
            W.register(win)
            win.show()
            return
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QColorDialog
    color = QColorDialog.getColor(QColor(bar.session.options.foreground), bar,
                                  "Foreground color")
    if color.isValid():
        bar.session.options.foreground = color.name()


# ---------------------------------------------------------------------------------- Stacks
def add_slice_cmd(session, image, **_):
    from rsstatchem.core.imaging import stacks as S
    fill = 0
    if image.type == GRAY8 and not image.is_inverted_lut():
        fill = 0
    elif image.type == GRAY8:
        fill = 255
    S.add_slice(image, True, fill)


def delete_slice_cmd(session, image, **_):
    from rsstatchem.core.imaging import stacks as S
    try:
        S.delete_slice(image)
    except S.StackError as exc:
        raise CommandError(str(exc)) from None


def _step_slice(delta: int):
    def run(session, image, **_):
        c, z, t = image.position
        index = image.current_index
        if image.n_channels > 1 or image.n_frames > 1:
            if image.n_slices > 1:
                image.set_position(c, (z + delta) % image.n_slices, t)
            elif image.n_frames > 1:
                image.set_position(c, z, (t + delta) % image.n_frames)
            else:
                image.set_position((c + delta) % image.n_channels, z, t)
        else:
            image.set_position(*image.position_of((index + delta) % image.stack_size))
    return run


def set_slice_cmd(session, image, slice=1, **_):
    n = int(slice)
    if not 1 <= n <= image.stack_size:
        raise CommandError(f'"Set Slice..." argument must be 1-{image.stack_size}')
    image.set_position(*image.position_of(n - 1))


def _images_to_stack_fields(session, image):
    return [C.choice("Method:", ("Copy (center)", "Copy (top-left)", "Scale (smallest)",
                                 "Scale (largest)"), "Copy (center)"),
            C.string("Name:", "Stack"),
            C.string("Title contains:", "", key="title"),
            C.checkbox("Bicubic interpolation", False, key="bicubic"),
            C.checkbox("Use titles as labels", True, key="use"),
            C.checkbox("Keep source images", False, key="keep")]


def images_to_stack_cmd(session, image, method="Copy (center)", name="Stack", title="",
                        bicubic=False, use=False, keep=False, **_):
    from rsstatchem.core.imaging import stacks as S
    images = [im for im in session.images if im.stack_size == 1 and
              (not title or title in im.title)]
    if len(images) < 1:
        raise CommandError("There must be at least two open images.")
    out = S.images_to_stack(images, name or "Stack", use, method, bicubic)
    if not keep:
        for im in images:
            session.close_image(im)
    return out


def stack_to_images_cmd(session, image, **_):
    from rsstatchem.core.imaging import stacks as S
    if image.stack_size < 2:
        raise CommandError('"Convert Stack to Images" requires a stack')
    parts = S.stack_to_images(image)
    session.close_image(image)
    return parts


def _montage_fields(session, image):
    n = image.stack_size if image else 1
    cols = int(math.sqrt(n) + 0.5) if n else 1
    while cols * max(1, math.ceil(n / cols)) < n:
        cols += 1
    rows = max(1, math.ceil(n / max(cols, 1)))
    scale = 1.0 if (image is None or image.width * cols <= 1200) else 0.5
    fields = [C.number("Columns:", cols, 0, integer=True),
              C.number("Rows:", rows, 0, integer=True),
              C.number("Scale factor:", scale, 2, key="scale")]
    if image is not None and image.stack_size > 1:
        fields += [C.number("First slice:", 1, 0, key="first", integer=True),
                   C.number("Last slice:", n, 0, key="last", integer=True)]
    fields += [C.number("Increment:", 1, 0, integer=True),
               C.number("Border width:", 0, 0, key="border", integer=True),
               C.number("Font size:", 12, 0, key="font", integer=True),
               C.checkbox("Label slices", False, key="label"),
               C.checkbox("Use foreground color", False, key="use")]
    return fields


def montage_cmd(session, image, columns, rows, scale=1.0, first=1, last=None, increment=1,
                border=0, font=12, label=False, use=False, **_):
    from rsstatchem.core.imaging import stacks as S
    if image.stack_size < 2:
        raise CommandError("Stack required")
    if image.is_composite and (image.n_slices > 1 or image.n_frames > 1):
        # hyperstack montage: a montage of every channel's z (or t) series, as a composite
        from rsstatchem.core.imaging.color import split_channels
        parts = []
        for chan in split_channels(image):
            sub = S.from_planes(S._flat(chan), chan.type, chan.title, like=chan)
            parts.append(S.make_montage(sub, int(columns), int(rows), float(scale), 1, None,
                                        int(increment), int(border)))
        planes = np.stack([p.plane() for p in parts])
        out = S.from_planes(planes, image.type, "Montage", dims=(len(parts), 1, 1), like=image)
        out.composite_mode = image.composite_mode
        out.display_ranges = list(image.display_ranges)
        return out
    last = image.stack_size if last is None else int(last)
    out = S.make_montage(image, int(columns), int(rows), float(scale), int(first), last,
                         int(increment), int(border), int(font), bool(label), bool(use),
                         session.options.foreground, session.options.background)
    if label:
        _draw_montage_labels(out, image, int(columns), int(rows), float(scale), int(first),
                             last, int(increment), int(border), int(font))
    return out


def _draw_montage_labels(out, image, columns, rows, scale, first, last, inc, border, font):
    """MontageMaker.drawLabel: the slice label (or number) centred at the bottom of each tile,
    burnt in with Pillow (white on dark, black on white backgrounds)."""
    from PIL import Image, ImageDraw, ImageFont
    w = int(image.width * scale)
    h = int(image.height * scale)
    p = out.plane()
    rgb = p if p.ndim == 3 else None
    base = Image.fromarray(rgb if rgb is not None else
                           (p if p.dtype == np.uint8 else np.zeros(p.shape, np.uint8)))
    draw = ImageDraw.Draw(base)
    try:
        fnt = ImageFont.load_default(size=font)
    except TypeError:
        fnt = ImageFont.load_default()
    x = y = 0
    s = first
    fg = 255 if p.dtype != np.uint8 or p.mean() < 128 else 0
    labels_mask = Image.new("L", base.size, 0)
    mdraw = ImageDraw.Draw(labels_mask)
    while s <= last:
        text = (image.slice_labels[s - 1] or "").splitlines()[0][:40] if \
            image.slice_labels[s - 1] else str(s)
        tw = draw.textlength(text, font=fnt)
        tx = x + w / 2 - tw / 2
        ty = y + h - border / 2 - font
        mdraw.text((tx, ty), text, fill=255, font=fnt)
        x += w + border
        if x >= out.width:
            x = 0
            y += h + border
            if y >= out.height:
                break
        s += inc
    m = np.asarray(labels_mask) > 127
    if p.ndim == 3:
        p[m] = 255 if fg else 0
    elif p.dtype == np.uint8:
        p[m] = fg
    else:
        p[m] = out.display_range[1]


def _reslice_fields(session, image):
    from rsstatchem.core.imaging.roi import LineRoi
    cal = image.calibration if image else None
    fields = [C.number(f"Output spacing ({cal.units if cal else 'pixels'}):",
                       cal.pixel_depth if cal else 1.0, 3, key="output")]
    if image is not None and isinstance(image.roi, LineRoi):
        fields.append(C.number("Slice_count:", 1, 0, key="slice_count", integer=True))
    else:
        fields.append(C.choice("Start at:", ("Top", "Left", "Bottom", "Right"), "Top",
                               key="start"))
    fields += [C.checkbox("Flip vertically", False, key="flip"),
               C.checkbox("Rotate 90 degrees", False, key="rotate"),
               C.checkbox("Avoid interpolation", False, key="avoid")]
    return fields


def reslice_cmd(session, image, output=1.0, start="Top", flip=False, rotate=False,
                avoid=False, slice_count=1, **_):
    from rsstatchem.core.imaging import stacks as S
    from rsstatchem.core.imaging.roi import LineRoi, RectRoi
    if image.stack_size < 2:
        raise CommandError("Stack required")
    roi = image.roi
    rect = line = None
    if isinstance(roi, LineRoi):
        line = (roi.x1, roi.y1, roi.x2, roi.y2)
    elif isinstance(roi, RectRoi):
        rect = _area_rect(image)
    elif roi is not None and not roi.is_line():
        raise CommandError("Line or rectangular selection required")
    try:
        if image.n_channels > 1 or (image.n_frames > 1 and image.n_slices > 1):
            return _reslice_hyper(image, output, start, flip, rotate, avoid, rect, line,
                                  slice_count)
        return S.reslice(image, float(output), start, flip, rotate, avoid, rect, line,
                         int(slice_count))
    except S.StackError as exc:
        raise CommandError(str(exc)) from None


def _reslice_hyper(image, output, start, flip, rotate, avoid, rect, line, slice_count):
    from rsstatchem.core.imaging import stacks as S
    planes = []
    nz2 = None
    for t in range(image.n_frames):
        for c in range(image.n_channels):
            sub = S.from_planes(image.pixels[t, :, c], image.type, "tmp", like=image)
            sub.calibration = image.calibration.copy()
            r = S.reslice(sub, float(output), start, flip, rotate, avoid, rect, line,
                          int(slice_count))
            planes.append(S._flat(r))
            nz2 = r.stack_size
    # (T, C, Z2, ...) -> (T, Z2, C, ...)
    arr = np.stack(planes).reshape((image.n_frames, image.n_channels, nz2) +
                                   planes[0].shape[1:])
    arr = np.swapaxes(arr, 1, 2)
    out = ImageDoc(np.ascontiguousarray(arr), image.type, "Reslice of " + image.title)
    out.luts = [lut.copy() for lut in image.luts]
    out.display_ranges = list(image.display_ranges)
    out.composite_mode = image.composite_mode
    out.hyperstack = True
    return out


def _zproject_fields(session, image):
    from rsstatchem.core.imaging.stacks import PROJECTIONS
    n = image.stack_size if image else 1
    if image is not None and (image.n_channels > 1 or image.hyperstack):
        n = image.n_slices if image.n_slices > 1 else image.n_frames
    fields = [C.number("Start slice:", 1, 0, key="start", integer=True),
              C.number("Stop slice:", n, 0, key="stop", integer=True),
              C.choice("Projection type", PROJECTIONS, "Max Intensity", key="projection")]
    if image is not None and image.n_frames > 1 and image.n_slices > 1:
        fields.append(C.checkbox("All time frames", True, key="all"))
    return fields


def zproject_cmd(session, image, start=1, stop=None, projection="Max Intensity", all=False,
                 **_):
    from rsstatchem.core.imaging import stacks as S
    try:
        return S.z_project(image, projection, int(start), int(stop) if stop else None,
                           bool(all))
    except S.StackError as exc:
        raise CommandError(str(exc)) from None


def _project3d_fields(session, image):
    from rsstatchem.core.imaging import project3d as P
    cal = image.calibration if image else None
    lower, upper = 1, 255
    if image is not None and image.threshold is not None:
        lower, upper = int(image.threshold.lower), int(image.threshold.upper)
    fields = [C.choice("Projection method:", P.METHODS, "Brightest Point", key="projection"),
              C.choice("Axis of rotation:", P.AXES, "Y-Axis", key="axis"),
              C.number(f"Slice spacing ({cal.units if cal else 'pixels'}):",
                       cal.pixel_depth if cal else 1.0, 2, key="slice"),
              C.number("Initial angle (0-359 degrees):", 0, 0, key="initial", integer=True),
              C.number("Total rotation (0-359 degrees):", 360, 0, key="total", integer=True),
              C.number("Rotation angle increment:", 10, 0, key="rotation", integer=True),
              C.number("Lower transparency bound:", lower, 0, key="lower", integer=True),
              C.number("Upper transparency bound:", upper, 0, key="upper", integer=True),
              C.number("Opacity (0-100%):", 0, 0, key="opacity", integer=True),
              C.number("Surface depth-cueing (0-100%):", 100, 0, key="surface", integer=True),
              C.number("Interior depth-cueing (0-100%):", 50, 0, key="interior", integer=True),
              C.checkbox("Interpolate", False)]
    if image is not None and image.n_frames > 1 and image.n_slices > 1:
        fields.append(C.checkbox("All time points", False, key="all"))
    return fields


def project3d_cmd(session, image, projection="Brightest Point", axis="Y-Axis", slice=1.0,
                  initial=0, total=360, rotation=10, lower=1, upper=255, opacity=0,
                  surface=100, interior=50, interpolate=False, all=False, **_):
    from rsstatchem.core.imaging import project3d as P
    if image.n_slices < 2:
        raise CommandError("Stack required" if image.n_frames == 1 else
                           "Hyperstack Z dimension must be greater than 1")
    image.calibration.pixel_depth = float(slice)
    rect = _area_rect(image)
    try:
        return P.project(image, projection, axis, float(slice), int(initial), int(total),
                         int(rotation), int(lower), int(upper), int(opacity),
                         int(surface), int(interior), bool(interpolate), rect,
                         bool(all), session.progress)
    except ValueError as exc:
        raise CommandError(str(exc)) from None


def zprofile_cmd(session, image, **_):
    from rsstatchem.core.imaging import stacks as S
    from rsstatchem.core.imaging.results import ResultsTable
    if image.stack_size < 2:
        raise CommandError("Stack required")
    rect = _area_rect(image)
    mask = image.roi.get_mask((image.height, image.width)) if (
        image.roi is not None and image.roi.is_area()) else None
    xs, means = S.z_profile(image, mask, None if mask is not None else rect)
    rt = ResultsTable(f"{image.title}-0-0")
    axis = "Frame" if (image.n_frames > 1 and image.n_slices == 1) else "Slice"
    for x, m in zip(xs, means):
        rt.add_row({axis: float(x), "Mean": float(m)})
    return rt


def measure_stack_cmd(session, image, **_):
    """Image > Stacks > Measure Stack: Measure of every slice (WP-C's measure when present)."""
    from rsstatchem.core.imaging.results import ResultsTable
    rt = session.results
    try:
        from rsstatchem.core.imaging import measure as M
        fn = getattr(M, "measure", None)
    except ImportError:
        fn = None
    pos = image.position
    for i in range(image.stack_size):
        image.set_position(*image.position_of(i))
        if fn is not None:
            try:
                res = fn(session, image)
                if isinstance(res, ResultsTable):
                    continue
            except Exception:  # noqa: BLE001 - fall back to the basic statistics
                pass
        p = image.plane().astype(np.float64)
        if p.ndim == 3:
            p = p.mean(axis=-1)
        m = image.roi.get_mask(p.shape) if image.roi is not None and image.roi.is_area() \
            else np.ones(p.shape, bool)
        v = image.calibration.cvalue(p[m]) if image.type != COLOR_RGB else p[m]
        cal = image.calibration
        rt.add_row({"Area": float(m.sum()) * cal.pixel_width * cal.pixel_height,
                    "Mean": float(np.mean(v)), "Min": float(np.min(v)),
                    "Max": float(np.max(v)), "Slice": float(i + 1)}, label=image.title)
    image.set_position(*pos)
    return rt


def _label_fields(session, image):
    n = image.stack_size if image else 1
    return [C.choice("Format:", ("0", "0000", "00:00", "00:00:00", "Text", "Label"), "0"),
            C.string("Starting value:", "0", key="starting"),
            C.string("Interval:", "1", key="interval"),
            C.number("X location:", 5, 0, key="x", integer=True),
            C.number("Y location:", 20, 0, key="y", integer=True),
            C.number("Font size:", 18, 0, key="font", integer=True),
            C.string("Text:", "", key="text"),
            C.string("Range:", f"1-{n}", key="range"),
            C.checkbox(" Use overlay", False, key="use"),
            C.checkbox(" Use_text tool font", False, key="use_text")]


def label_cmd(session, image, format="0", starting="0", interval="1", x=5, y=20, font=18,
              text="", range="", use=False, **_):
    """Image > Stacks > Label: stamp numbers/times/text on every slice (burnt in, or as
    overlay text with "Use overlay")."""
    from rsstatchem.core.imaging import stacks as S
    from rsstatchem.core.imaging.roi import Overlay, TextRoi
    start = _parse(starting)
    inc = _parse(interval, 1.0)
    decimals = max(len(starting.split(".")[1]) if "." in starting else 0,
                   len(interval.split(".")[1]) if "." in interval else 0)
    idx = S.parse_range(range, image.stack_size) if range else \
        list(np.arange(1, image.stack_size + 1))
    if use:
        if image.overlay is None:
            image.overlay = Overlay()
        for k, i in enumerate(idx):
            label = S.label_text(format, k, start, inc, decimals, text,
                                 image.slice_labels[i - 1])
            c, z, t = image.position_of(i - 1)
            pos = (0, z + 1, t + 1) if image.n_channels > 1 else (0, i, 0)
            image.overlay.add(TextRoi(x=float(x), y=float(y) - font, text=label,
                                      font_size=float(font), stroke_color="#ffffff",
                                      position=pos))
        return None
    from PIL import Image, ImageDraw, ImageFont
    try:
        fnt = ImageFont.load_default(size=int(font))
    except TypeError:
        fnt = ImageFont.load_default()
    color = session.options.foreground
    for k, i in enumerate(idx):
        label = S.label_text(format, k, start, inc, decimals, text, image.slice_labels[i - 1])
        c, z, t = image.position_of(i - 1)
        p = image.plane(c, z, t)
        mask_img = Image.new("L", (image.width, image.height), 0)
        ImageDraw.Draw(mask_img).text((x, y - font), label, fill=255, font=fnt)
        m = np.asarray(mask_img) > 127
        if p.ndim == 3:
            rgb = tuple(int(color.lstrip("#")[-6:][j:j + 2], 16) for j in (0, 2, 4))
            p[m] = rgb
        else:
            p[m] = _fg_value(session, image)


def _fg_value(session, image):
    from rsstatchem.core.imaging.transform import background_value
    return background_value(image, session.options.foreground, True)


def statistics_cmd(session, image, **_):
    from rsstatchem.core.imaging import stacks as S
    mask = image.roi.get_mask((image.height, image.width)) if (
        image.roi is not None and image.roi.is_area()) else None
    stats = S.stack_statistics(image, mask)
    rt = session.results
    rt.add_row(stats)
    return rt


def animation_options_cmd(session, image, speed=7.0, first=1, last=None, loop=False, **_):
    image.calibration.fps = float(speed)
    image.properties["animation.first"] = str(int(first))
    image.properties["animation.last"] = str(int(last if last else image.stack_size))
    image.properties["animation.loop"] = "true" if loop else "false"


def _animation_fields(session, image):
    fps = image.calibration.fps if image and image.calibration.fps else 7.0
    n = image.stack_size if image else 1
    return [C.number("Speed (0.1-1000 fps):", fps, 1, key="speed"),
            C.number("First frame:", 1, 0, key="first", integer=True),
            C.number("Last frame:", n, 0, key="last", integer=True),
            C.checkbox("Loop back and forth", False, key="loop")]


def _start_animation_ui(bar, cmd) -> None:
    from PySide6.QtCore import QTimer
    image = bar.session.active
    if image is None or image.stack_size < 2:
        bar.error("Stack required", "Animation")
        return
    win = _window(image)
    if win is None:
        return
    timer = getattr(win, "_ij_d_animation", None)
    if timer is None:
        timer = QTimer(win)
        win._ij_d_animation = timer
        state = {"dir": 1}

        def tick():
            doc = win.doc
            first = int(doc.properties.get("animation.first", 1)) - 1
            last = int(doc.properties.get("animation.last", doc.stack_size)) - 1
            loop = doc.properties.get("animation.loop") == "true"
            i = doc.current_index + state["dir"]
            if i > last:
                if loop:
                    state["dir"] = -1
                    i = max(last - 1, first)
                else:
                    i = first
            elif i < first:
                state["dir"] = 1
                i = min(first + 1, last)
            doc.set_position(*doc.position_of(i))
            win.refresh()
        timer.timeout.connect(tick)
    fps = image.calibration.fps or 7.0
    timer.start(max(1, int(1000 / fps)))


def _stop_animation_ui(bar, cmd) -> None:
    from rsstatchem.ui.imagej.windows import image_windows
    for win in image_windows():
        timer = getattr(win, "_ij_d_animation", None)
        if timer is not None:
            timer.stop()


# ---------------------------------------------------------------------------------- Tools
def _combine_fields(session, image):
    titles = _titles(session) or ["-"]
    return [C.choice("Stack1:", titles, titles[0], key="stack1"),
            C.choice("Stack2:", titles, titles[1] if len(titles) > 1 else titles[0],
                     key="stack2"),
            C.checkbox("Combine vertically", False, key="combine")]


def combine_cmd(session, image, stack1, stack2, combine=False, **_):
    from rsstatchem.core.imaging import stacks as S
    a, b = _find(session, stack1), _find(session, stack2)
    if a is b:
        raise CommandError("Two different stacks are required")
    bg = _bg_color(session, a, True)
    try:
        out = S.combine(a, b, combine, np.array(bg, dtype=np.uint8) if isinstance(bg, tuple)
                        else bg)
    except S.StackError as exc:
        raise CommandError(str(exc)) from None
    session.close_image(a)
    session.close_image(b)
    return out


def _concat_fields(session, image):
    titles = _titles(session) or ["-"]
    fields = [C.checkbox("All_open windows", False, key="all_open"),
              C.choice("Image1:", titles, titles[0], key="image1"),
              C.choice("Image2:", titles, titles[1] if len(titles) > 1 else titles[0],
                       key="image2")]
    none = [NONE_ITEM] + titles
    for i in range(3, max(len(titles), 2) + 2):
        fields.append(C.choice(f"Image{i}:", none, NONE_ITEM, key=f"image{i}"))
    fields += [C.string("Title:", "Untitled"),
               C.checkbox("Keep original images", False, key="keep"),
               C.checkbox("Open as 4D_image", False, key="open")]
    return fields


def concatenate_cmd(session, image, all_open=False, title="Untitled", keep=False, open=False,
                    **values):
    from rsstatchem.core.imaging import stacks as S
    if all_open:
        images = list(session.images)
    else:
        names = [values[k] for k in sorted((k for k in values if k.startswith("image")),
                                           key=lambda k: int(k[5:]))
                 if values[k] not in (NONE_ITEM, "", None)]
        images = [_find(session, n) for n in names]
    try:
        out = S.concatenate(images, title or "Untitled", open)
    except S.StackError as exc:
        raise CommandError(str(exc)) from None
    if not keep:
        for im in images:
            session.close_image(im)
    return out


def _grouped_fields(session, image):
    from rsstatchem.core.imaging.stacks import PROJECTIONS
    n = image.stack_size if image else 1
    return [C.choice("Projection method:", PROJECTIONS, "Average Intensity",
                     key="projection"),
            C.number("Group size:", n, 0, key="group", integer=True)]


def grouped_cmd(session, image, projection="Average Intensity", group=1, **_):
    from rsstatchem.core.imaging import stacks as S
    try:
        return S.grouped_z_project(image, projection, int(group))
    except S.StackError as exc:
        raise CommandError(str(exc)) from None


def _insert_fields(session, image):
    titles = _titles(session) or ["-"]
    return [C.choice("Source: ", titles, titles[0], key="source"),
            C.choice("Destination: ", titles, titles[-1], key="destination"),
            C.number("X Location: ", 0, 0, key="x", integer=True),
            C.number("Y Location: ", 0, 0, key="y", integer=True)]


def insert_cmd(session, image, source, destination, x=0, y=0, **_):
    from rsstatchem.core.imaging import stacks as S
    src, dst = _find(session, source), _find(session, destination)
    dst.snapshot("Insert...")
    try:
        S.insert_stack(src, dst, int(x), int(y))
    except S.StackError as exc:
        raise CommandError(str(exc)) from None
    dst.changed = True
    session.image_updated(dst)


def _magic_montage_ui(bar, cmd) -> None:
    from PySide6.QtWidgets import QMessageBox
    image = bar.session.active
    box = QMessageBox(bar)
    box.setWindowTitle("Magic Montage Tools")
    box.setText("Magic Montage: make a montage of the stack, or turn a montage back into a "
                "stack.")
    make = box.addButton("Make Montage...", QMessageBox.ButtonRole.ActionRole)
    back = box.addButton("Montage to Stack...", QMessageBox.ButtonRole.ActionRole)
    box.addButton(QMessageBox.StandardButton.Close)
    if bar.message_hook is not None:
        bar.message_hook("Magic Montage Tools", box.text())
        return
    box.exec()
    if box.clickedButton() is make and image is not None:
        bar.run_command("Make Montage...")
    elif box.clickedButton() is back and image is not None:
        bar.run_command("Montage to Stack...")


def substack_cmd(session, image, slices="", delete=False, do_not=False, **_):
    from rsstatchem.core.imaging import stacks as S
    if not slices:
        raise CommandError("Range or list required")
    try:
        if image.hyperstack and (image.n_channels > 1 or image.n_frames > 1):
            return S.make_subset(image, slices=slices if image.n_slices > 1 else "",
                                 frames=slices if image.n_slices == 1 else "")
        return S.substack(image, slices, delete)
    except S.StackError as exc:
        raise CommandError(str(exc)) from None


def _m2s_fields(session, image):
    cols = int(image.properties.get("xMontage", 2)) if image else 2
    rows = int(image.properties.get("yMontage", 2)) if image else 2
    return [C.number("Columns: ", cols, 0, key="columns", integer=True),
            C.number("Rows: ", rows, 0, key="rows", integer=True),
            C.number("Border width: ", 0, 0, key="border", integer=True)]


def m2s_cmd(session, image, columns=2, rows=2, border=0, **_):
    from rsstatchem.core.imaging import stacks as S
    if int(columns) < 1 or int(rows) < 1:
        return None
    return S.montage_to_stack(image, int(columns), int(rows), int(border))


def xy_profile_cmd(session, image, **_):
    """Image > Stacks > Tools > Plot XY Profile: the line profile of every slice (table)."""
    from rsstatchem.core.imaging.results import ResultsTable
    from rsstatchem.core.imaging.roi import LineRoi
    from rsstatchem.core.imaging.transform import _clamped_bilinear
    roi = image.roi
    if not isinstance(roi, LineRoi):
        raise CommandError("Straight line selection required")
    n = int(round(math.hypot(roi.x2 - roi.x1, roi.y2 - roi.y1)))
    xs = roi.x1 + np.arange(n + 1) * ((roi.x2 - roi.x1) / max(n, 1))
    ys = roi.y1 + np.arange(n + 1) * ((roi.y2 - roi.y1) / max(n, 1))
    rt = ResultsTable("Plot Values")
    for i in range(n + 1):
        rt.increment_counter()
        rt.add_value("X", float(i * image.calibration.pixel_width))
    for s in range(image.stack_size):
        p = image.plane_at(s)
        if p.ndim == 3:
            from rsstatchem.core.imaging.convert import rgb_to_byte
            p = rgb_to_byte(p)
        kind = {np.dtype(np.uint8): "byte", np.dtype(np.uint16): "short"}.get(p.dtype, "float")
        vals = _clamped_bilinear(p, xs, ys, kind)
        for i, v in enumerate(vals):
            rt.set_value(f"Y{s + 1}", i, float(v))
    return rt


def reduce_cmd(session, image, reduction=2, reduce=False, **_):
    from rsstatchem.core.imaging import stacks as S
    S.reduce_stack(image, int(reduction), bool(reduce) or image.n_frames == 1)


def remove_labels_cmd(session, image, **_):
    image.slice_labels = [None] * image.stack_size


def reverse_cmd(session, image, **_):
    from rsstatchem.core.imaging import stacks as S
    S.reverse(image)


def set_label_cmd(session, image, label="", **_):
    image.set_slice_label(label or None)


# ---------------------------------------------------------------------------------- Hyperstacks
def _new_hyper_fields(session, image):
    return [C.string("Title:", "HyperStack"),
            C.choice("Type:", ("8-bit", "16-bit", "32-bit", "RGB"), "8-bit"),
            C.choice("Display Mode:", ("Composite", "Color", "Grayscale"), "Composite",
                     key="display"),
            C.number("Width:", 400, 0, integer=True),
            C.number("Height:", 300, 0, integer=True),
            C.number("Channels (c):", 3, 0, key="channels", integer=True),
            C.number("Slices (z):", 4, 0, key="slices", integer=True),
            C.number("Frames (t):", 5, 0, key="frames", integer=True),
            C.checkbox("Label images", True, key="label")]


def new_hyper_cmd(session, image, title="HyperStack", type="8-bit", display="Composite",
                  width=400, height=300, channels=3, slices=4, frames=5, label=False, **_):
    from rsstatchem.core.imaging import stacks as S
    t = {"8-bit": GRAY8, "16-bit": GRAY16, "32-bit": GRAY32, "RGB": COLOR_RGB}[type]
    return S.new_hyperstack(title, t, int(width), int(height), int(channels), int(slices),
                            int(frames), display, bool(label))


def _s2h_fields(session, image):
    from rsstatchem.core.imaging.stacks import ORDERS
    n = image.stack_size if image else 1
    return [C.choice("Order:", ORDERS, ORDERS[0], key="order"),
            C.number("Channels (c):", 1, 0, key="channels", integer=True),
            C.number("Slices (z):", n, 0, key="slices", integer=True),
            C.number("Frames (t):", 1, 0, key="frames", integer=True),
            C.choice("Display Mode:", ("Composite", "Color", "Grayscale"), "Color",
                     key="display")]


def s2h_cmd(session, image, order="xyczt(default)", channels=1, slices=1, frames=1,
            display="Color", **_):
    from rsstatchem.core.imaging import stacks as S
    if order == "xyczt":
        order = "xyczt(default)"
    try:
        S.stack_to_hyperstack(image, int(channels), int(slices), int(frames), order, display)
    except S.StackError as exc:
        raise CommandError(str(exc)) from None


def h2s_cmd(session, image, **_):
    from rsstatchem.core.imaging import stacks as S
    S.hyperstack_to_stack(image)


def _reduce_dim_fields(session, image):
    fields = [C.message("Create image with:")]
    if image is not None:
        c, z, t = image.dims
        if c != 1:
            fields.append(C.checkbox(f"Channels ({c})", True, key="channels"))
        if z != 1:
            fields.append(C.checkbox(f"Slices ({z})", True, key="slices"))
        if t != 1:
            fields.append(C.checkbox(f"Frames ({t})", True, key="frames"))
    fields.append(C.checkbox("Keep source", True, key="keep"))
    return fields


def reduce_dim_cmd(session, image, channels=False, slices=False, frames=False, keep=False,
                   **_):
    from rsstatchem.core.imaging import stacks as S
    out = S.reduce_dimensionality(image, channels or image.n_channels == 1,
                                  slices or image.n_slices == 1, frames or image.n_frames == 1)
    if not keep:
        session.close_image(image)
    return out


def _subset_fields(session, image):
    c, z, t = image.dims if image else (1, 1, 1)
    return [C.string("Channels:", f"1-{c}", key="channels"),
            C.string("Slices:", f"1-{z}", key="slices"),
            C.string("Frames:", f"1-{t}", key="frames")]


def subset_cmd(session, image, channels="", slices="", frames="", **_):
    from rsstatchem.core.imaging import stacks as S
    try:
        return S.make_subset(image, channels, slices, frames)
    except S.StackError as exc:
        raise CommandError(str(exc)) from None


# ---------------------------------------------------------------------------------- Crop etc.
def crop_cmd(session, image, **_):
    from rsstatchem.core.imaging import transform as T
    from rsstatchem.core.imaging.roi import RectRoi
    rect = _area_rect(image)
    if rect is None:
        raise CommandError("Area selection required")
    roi = image.roi
    T.crop_doc(image, rect)
    if isinstance(roi, RectRoi) and not roi.corner_diameter:
        image.roi = None
    else:
        roi = roi.copy()
        roi.translate(-rect[0], -rect[1])
        image.roi = roi


def _duplicate_fields(session, image):
    title = session.unique_title(_new_title(image.title)) if image else "Untitled"
    fields = [C.string("Title:", title, columns=15)]
    if image is not None and image.roi is not None and image.roi.is_area():
        fields.append(C.checkbox("Ignore selection", False, key="ignore"))
    if image is not None and image.stack_size > 1:
        if image.n_channels > 1 or image.hyperstack:
            c, z, t = image.dims
            fields.append(C.checkbox("Duplicate hyperstack", True, key="duplicate"))
            if c > 1:
                fields.append(C.string("Channels (c):", f"1-{c}", key="channels"))
            if z > 1:
                fields.append(C.string("Slices (z):", f"1-{z}", key="slices"))
            if t > 1:
                fields.append(C.string("Frames (t):", f"1-{t}", key="frames"))
        else:
            fields.append(C.checkbox("Duplicate stack", False, key="duplicate"))
            fields.append(C.string("Range:", f"1-{image.stack_size}", key="range"))
    return fields


def duplicate_cmd(session, image, title="", ignore=False, duplicate=False, range="",
                  channels="", slices="", frames="", **_):
    """Image > Duplicate (current slice, a range, or a hyperstack subset; cropped to the
    selection bounds unless "Ignore selection")."""
    from rsstatchem.core.imaging import stacks as S
    from rsstatchem.core.imaging import transform as T
    if image.stack_size > 1 and duplicate:
        if image.n_channels > 1 or image.hyperstack:
            out = S.make_subset(image, channels, slices, frames)
        elif range:
            idx = S.parse_range(range.replace(" ", ""), image.stack_size)
            flat = S._flat(image)
            out = S.from_planes(flat[[i - 1 for i in idx]], image.type, title, like=image,
                                labels=[image.slice_labels[i - 1] for i in idx])
            out.display_ranges = [image.display_range]
        else:
            out = image.duplicate(title)
    elif image.stack_size > 1:
        c, z, t = image.position
        pix = image.pixels[t:t + 1, z:z + 1, c:c + 1].copy()
        out = ImageDoc(pix, image.type, title)
        out.calibration = image.calibration.copy()
        if image.type != COLOR_RGB:
            out.luts = [image.lut.copy()]
            out.display_ranges = [image.display_range]
        out.slice_labels = [image.slice_label()]
    else:
        out = image.duplicate(title)
    out.title = title or _new_title(image.title)
    out.info = image.info
    out.properties = dict(image.properties)
    if image.overlay is not None:
        out.overlay = image.overlay.copy()
    rect = _area_rect(image)
    if rect is not None and not ignore:
        roi = image.roi.copy()
        T.crop_doc(out, rect)
        if roi is not None and _nonrect_mask(image) is not None:
            roi.translate(-rect[0], -rect[1])
            out.roi = roi
        else:
            out.roi = None
    else:
        out.roi = image.roi.copy() if image.roi is not None else None
    return out


def rename_cmd(session, image, title="", **_):
    if title:
        image.title = session.unique_title(title, exclude=image)
        win = _window(image)
        if win is not None:
            win.update_title()


def _scale_fields(session, image):
    stack = image is not None and image.stack_size > 1
    w = image.width if image else 100
    h = image.height if image else 100
    fields = [C.string("X Scale:", "-", key="x"), C.string("Y Scale:", "-", key="y")]
    if stack:
        fields.append(C.string("Z Scale:", "-", key="z"))
    fields += [C.string("Width (pixels):", str(max(1, w // 2)), key="width"),
               C.string("Height (pixels):", str(max(1, h // 2)), key="height")]
    if stack:
        label = "Depth (images):"
        if image.hyperstack or image.n_channels > 1:
            label = "Depth (frames):" if (image.n_slices == 1 and image.n_frames > 1) \
                else "Depth (slices):"
        depth = image.n_frames if label == "Depth (frames):" else (
            image.n_slices if label == "Depth (slices):" else image.stack_size)
        fields.append(C.string(label, str(depth), key="depth"))
    fields.append(C.choice("Interpolation:", ("None", "Bilinear", "Bicubic"), "Bilinear"))
    if image is not None and image.bit_depth in (8, 24):
        fields.append(C.checkbox("Fill with background color", False, key="fill"))
    fields.append(C.checkbox("Average when downsizing", True, key="average"))
    if stack and not (image.hyperstack or image.n_channels > 1):
        fields.append(C.checkbox("Process entire stack", True, key="process"))
    fields += [C.checkbox("Create new window", True, key="create"),
               C.string("Title:", _new_title(image.title) if image else "Untitled",
                        key="title")]
    return fields


def scale_cmd(session, image, x="-", y="-", z="-", width="", height="", depth="",
              interpolation="Bilinear", fill=False, average=False, process=False,
              create=False, title="", **_):
    """Image > Scale (Scaler): scale factors or a new size; new window or in place."""
    from rsstatchem.core.imaging import transform as T
    rect = _area_rect(image)
    r = rect if rect is not None else (0, 0, image.width, image.height)
    xs, ys = _parse(x), _parse(y)
    if xs > 0 and ys <= 0:
        ys = xs
    nw = int(round(_parse(width))) if str(width).strip() not in ("", "-") else 0
    nh = int(round(_parse(height))) if str(height).strip() not in ("", "-") else 0
    if nh and not nw:
        nw = int(round(nh * r[2] / r[3]))
    elif nw and not nh:
        nh = int(round(nw * r[3] / r[2]))
    if xs > 0 and ys > 0:
        nw, nh = int(round(r[2] * xs)), int(round(r[3] * ys))
    if nw <= 0 or nh <= 0:
        raise CommandError("Width or height is 0")
    if xs <= 0:
        xs, ys = nw / r[2], nh / r[3]
    method = T.method_index(interpolation)
    if image.width == 1 or image.height == 1:
        method = T.NONE if not create else method
    bg = _bg_color(session, image, bool(fill))
    kind = _kind(image)
    stack = image.stack_size > 1
    hyper = image.hyperstack or image.n_channels > 1
    z_depth = None
    if stack and str(depth).strip() not in ("", "-"):
        z_depth = int(round(_parse(depth)))
        zs = _parse(z)
        if zs > 0:
            base = image.n_frames if (hyper and image.n_slices == 1) else (
                image.n_slices if hyper else image.stack_size)
            z_depth = int(round(base * zs))
    old_depth = image.n_frames if (hyper and image.n_slices == 1) else (
        image.n_slices if hyper else image.stack_size)
    do_z = z_depth is not None and z_depth > 0 and z_depth != old_depth
    if do_z:
        create = True
        process = True
    if hyper:
        process = True
    cal = image.calibration
    if create:
        if stack and process:
            out = image.duplicate(title or _new_title(image.title))

            def fn(p):
                return T.resize_plane(p, kind, nw, nh, method, rect, average, bg=bg)
            _replace_all_planes(out, fn)
        else:
            p = T.resize_plane(image.plane(), kind, nw, nh, method, rect, average, bg=bg)
            out = ImageDoc(p[None, None, None], image.type, title or _new_title(image.title))
            out.calibration = cal.copy()
            if image.type != COLOR_RGB:
                out.luts = [image.lut.copy()]
                if image.bit_depth != 8:
                    out.display_ranges = [image.display_range]
        out.title = title or _new_title(image.title)
        c2 = out.calibration
        if c2.scaled():
            c2.pixel_width *= 1.0 / xs
            c2.pixel_height *= 1.0 / ys
        c2.x_origin *= xs
        c2.y_origin *= ys
        out.roi = None
        if image.overlay is not None and not image.overlay.hidden and not do_z:
            ov = image.overlay.copy()
            if rect is not None:
                for roi in ov:
                    roi.translate(-rect[0], -rect[1])
            _scale_overlay(ov, xs, ys)
            out.overlay = ov
        if do_z:
            _resize_z(out, z_depth, method, average,
                      axis_t=hyper and image.n_slices == 1 and image.n_frames > 1)
        return out
    lo = image.display_range[0] if image.type != COLOR_RGB else 0.0
    _process_planes(image, lambda p: T.scale_plane(p, kind, xs, ys, method, rect, bg, lo),
                    stack and process, masked=False)
    image.roi = None
    return None


# ---------------------------------------------------------------------------------- Transform
def _flip(direction: str):
    def run(session, image, stack=False, **_):
        from rsstatchem.core.imaging import transform as T
        rect = _area_rect(image)
        fn = T.flip_horizontal if direction == "h" else T.flip_vertical
        _process_planes(image, lambda p: fn(p, rect), stack)
    return run


def flip_z_cmd(session, image, **_):
    from rsstatchem.core.imaging import stacks as S
    if image.stack_size < 2:
        raise CommandError("Stack required")
    if image.n_slices > 1:
        image.pixels = np.ascontiguousarray(image.pixels[:, ::-1])
        image.slice_labels = [image.slice_labels[image.stack_index(c, image.n_slices - 1 - z,
                                                                   t)]
                              for t in range(image.n_frames) for z in range(image.n_slices)
                              for c in range(image.n_channels)]
    else:
        S.reverse(image)


def _rotate90(direction: str):
    def run(session, image, **_):
        from rsstatchem.core.imaging import transform as T
        fn = T.rotate_right if direction == "r" else T.rotate_left
        _replace_all_planes(image, fn)
        cal = image.calibration
        cal.pixel_width, cal.pixel_height = cal.pixel_height, cal.pixel_width
        image.roi = None
        if image.overlay is not None:
            image.overlay = None
    return run


def _rotate_fields(session, image):
    fields = [C.slider("Angle:", -90, 90, 15.0, 1, key="angle"),
              C.number("Grid lines:", 1, 0, key="grid", integer=True),
              C.choice("Interpolation:", ("None", "Bilinear", "Bicubic"), "Bilinear"),
              C.checkbox("Fill with background color", False, key="fill")]
    rect = _area_rect(image) if image else None
    if rect is None or rect == (0, 0, image.width, image.height):
        fields.append(C.checkbox("Enlarge image", False, key="enlarge"))
    return fields + _stack_field(image)


def rotate_cmd(session, image, angle=15.0, grid=1, interpolation="None", fill=False,
               enlarge=False, stack=False, **_):
    """Image > Transform > Rotate (Rotator: clockwise about the selection centre)."""
    from rsstatchem.core.imaging import transform as T
    method = T.method_index(interpolation)
    bg = _bg_color(session, image, bool(fill))
    kind = _kind(image)
    angle = float(angle)
    if enlarge:
        w2, h2 = T.rotated_bounds(image.width, image.height, angle)
        if (w2, h2) != (image.width, image.height):
            xo, yo = T.canvas_offsets(image.width, image.height, w2, h2, "Center")
            fill_v = np.array(bg, dtype=np.uint8) if isinstance(bg, tuple) else (
                bg if fill else 0)
            _replace_all_planes(image, lambda p: T.canvas_plane(p, w2, h2, xo, yo, fill_v))
            image.roi = None
            stack = True
    rect = _area_rect(image)
    _process_planes(image, lambda p: T.rotate_plane(p, kind, angle, method, rect, bg),
                    stack or enlarge)


def _translate_fields(session, image):
    fields = [C.slider("X offset:", -100, 100, 15.0, 1, key="x"),
              C.slider("Y offset:", -100, 100, 15.0, 1, key="y"),
              C.choice("Interpolation:", ("None", "Bilinear", "Bicubic"), "None")]
    if image is not None and image.overlay is not None:
        fields.append(C.checkbox("Overlay only", False, key="overlay"))
    return fields + _stack_field(image)


def translate_cmd(session, image, x=0.0, y=0.0, interpolation="None", overlay=False,
                  stack=False, **_):
    from rsstatchem.core.imaging import transform as T
    method = T.method_index(interpolation)
    kind = _kind(image)
    if not overlay or image.overlay is None:
        rect = _area_rect(image)
        _process_planes(image, lambda p: T.translate_plane(p, kind, float(x), float(y),
                                                           method, rect), stack)
    if image.overlay is not None:
        for roi in image.overlay:
            roi.translate(float(x), float(y))


def _bin_fields(session, image):
    from rsstatchem.core.imaging.transform import BIN_METHODS
    fields = [C.number("X shrink factor:", 2, 0, key="x", integer=True),
              C.number("Y shrink factor:", 2, 0, key="y", integer=True)]
    if image is not None and image.stack_size > 1 and not (
            image.is_composite and image.n_channels == image.stack_size):
        fields.append(C.number("Z shrink factor:", 1, 0, key="z", integer=True))
    fields.append(C.choice("Bin Method: ", BIN_METHODS, "Average", key="bin"))
    return fields


def bin_cmd(session, image, x=2, y=2, z=1, bin="Average", **_):
    """Image > Transform > Bin (Binner.shrink)."""
    from rsstatchem.core.imaging import transform as T
    xs, ys, zs = max(int(x), 1), max(int(y), 1), max(int(z), 1)
    kind = _kind(image)
    inv = image.is_inverted_lut()
    _replace_all_planes(image, lambda p: T.bin_plane(p, kind, xs, ys, bin, inv))
    if bin == "Sum" and kind == "rgb":
        from rsstatchem.core.imaging.adjust import apply_rgb_range
        image.pixels = apply_rgb_range(image.pixels, 0, 255 * xs * ys * zs)
    if zs > 1 and image.n_slices > 1:
        p = image.pixels
        out = np.stack([np.stack([T.bin_z(p[t, :, c], kind, zs, bin)
                                  for c in range(p.shape[2])], axis=1)
                        for t in range(p.shape[0])])
        luts = [lut.copy() for lut in image.luts]
        image.replace_pixels(np.ascontiguousarray(out), image.type)
        image.luts = luts
    cal = image.calibration
    if cal.scaled():
        cal.pixel_width *= xs
        cal.pixel_height *= ys
        cal.pixel_depth *= zs
    if bin == "Sum" and image.bit_depth in (16, 32):
        lo, hi = image.plane_min_max()
        image.set_display_range(lo, hi)
    image.roi = None


def image_to_results_cmd(session, image, **_):
    from rsstatchem.core.imaging.transform import image_to_results
    rect = _area_rect(image)
    p = image.plane()
    if image.type != COLOR_RGB and image.calibration.calibrated():
        p = image.calibration.cvalue(p.astype(np.float64))
    rt = image_to_results(p, rect)
    session.results = rt
    return rt


def results_to_image_cmd(session, image, **_):
    from rsstatchem.core.imaging.transform import results_to_image
    rt = session.results
    if rt is None or len(rt.headings) == 0:
        raise CommandError("The Results table is empty")
    try:
        return results_to_image(rt)
    except ValueError as exc:
        raise CommandError(str(exc)) from None


# ---------------------------------------------------------------------------------- Zoom
def _zoom_ui(kind: str):
    def action(bar, cmd) -> None:
        image = bar.session.active
        win = _window(image) if image is not None else None
        if win is None:
            bar.error("There are no images open")
            return
        cv = win.canvas
        if kind == "in":
            cv.zoom_in()
        elif kind == "out":
            cv.zoom_out()
        elif kind in ("original", "100"):
            cv.set_magnification(1.0, (image.width / 2, image.height / 2))
            if kind == "100" or kind == "original":
                hint = cv.sizeHint()
                win.resize(max(hint.width() + 8, 220), hint.height() + 50)
        elif kind == "selection":
            rect = _area_rect(image) or (image.roi and tuple(int(v) for v in image.roi.bounds()))
            if not rect:
                bar.error("Selection required", "Zoom")
                return
            from rsstatchem.ui.imagej.canvas import ZOOM_LEVELS
            x, y, w, h = rect
            avail = max(cv.width(), 200), max(cv.height(), 200)
            m = min(avail[0] / max(w, 1), avail[1] / max(h, 1))
            m = max([z for z in ZOOM_LEVELS if z <= m] or [ZOOM_LEVELS[0]])
            cv.set_magnification(m, (x + w / 2, y + h / 2))
        elif kind in ("fit", "max"):
            screen = win.screen().availableGeometry() if win.screen() else None
            if kind == "max" and screen is not None:
                win.setGeometry(screen)
            avail_w = (screen.width() - 40) if (screen and kind == "max") else max(cv.width(),
                                                                                  1)
            avail_h = (screen.height() - 120) if (screen and kind == "max") else \
                max(cv.height(), 1)
            m = min(avail_w / image.width, avail_h / image.height)
            cv.set_magnification(m, (image.width / 2, image.height / 2))
        elif kind == "set":
            from rsstatchem.ui.imagej.dialogs.generic import GenericDialog
            fields = [C.number("Zoom (%):", cv.magnification * 100, 1, key="zoom"),
                      C.number("X center:", image.width / 2, 0, key="x"),
                      C.number("Y center:", image.height / 2, 0, key="y")]
            vals = GenericDialog("Set Zoom", fields, bar).get()
            if vals is None:
                return
            cv.set_magnification(float(vals["zoom"]) / 100.0, (vals["x"], vals["y"]))
        win.update_title()
    return action


# ---------------------------------------------------------------------------------- Overlay
def _overlay(image):
    from rsstatchem.core.imaging.roi import Overlay
    if image.overlay is None:
        image.overlay = Overlay()
    return image.overlay


def add_selection_cmd(session, image, stroke="", width=0.0, fill="", **_):
    roi = image.roi
    if roi is None:
        raise CommandError("Selection required")
    r = roi.copy()
    defaults = session.options.extra.get("overlay", {})
    r.stroke_color = r.stroke_color or stroke or defaults.get("stroke") or None
    if width:
        r.stroke_width = float(width)
    elif defaults.get("width"):
        r.stroke_width = float(defaults["width"])
    if fill or defaults.get("fill"):
        r.fill_color = fill or defaults.get("fill")
    if image.stack_size > 1 and defaults.get("position", True):
        c, z, t = image.position
        if image.n_channels > 1 or image.n_frames > 1:
            r.position = (c + 1 if image.n_channels > 1 else 0, z + 1, t + 1)
        else:
            r.position = (0, image.current_index + 1, 0)
    _overlay(image).add(r)
    image.roi = None


def _add_image_fields(session, image):
    titles = _titles(session, exclude=image) or ["-"]
    return [C.choice("Image to add:", titles, titles[0], key="image"),
            C.number("X location:", 0, 0, key="x", integer=True),
            C.number("Y location:", 0, 0, key="y", integer=True),
            C.number("Opacity (0-100%):", 100, 0, key="opacity", integer=True),
            C.checkbox("Zero transparent", False, key="zero")]


def add_image_cmd(session, target, x=0, y=0, opacity=100, zero=False, **values):
    """Image > Overlay > Add Image (the field key "image" is the source title, so the target
    document is the second positional argument)."""
    from rsstatchem.core.imaging.lut import render
    image = target
    src = _find(session, values.get("image", ""))
    rgb = render(src)
    roi = ImageRoi(x=float(x), y=float(y), width=float(src.width), height=float(src.height),
                   pixels=rgb, opacity=int(opacity) / 100.0, zero_transparent=bool(zero),
                   name=src.title)
    _overlay(image).add(roi)


def _overlay_flag(attr: str, value):
    def run(session, image, **_):
        if image.overlay is None:
            if attr == "hidden" and value is None:
                return None
            raise CommandError("Overlay required")
        image.overlay.hidden = (not image.overlay.hidden) if value is None else value
    return run


def from_manager_cmd(session, image, **_):
    if not session.roi_manager:
        raise CommandError("ROI Manager is empty")
    ov = _overlay(image)
    ov.clear()
    for r in session.roi_manager:
        ov.add(r.copy())
    ov.hidden = False


def to_manager_cmd(session, image, **_):
    if image.overlay is None or not len(image.overlay):
        raise CommandError("Overlay required")
    session.roi_manager[:] = [r.copy() for r in image.overlay]
    image.overlay = None


def remove_overlay_cmd(session, image, **_):
    image.overlay = None


def list_elements_cmd(session, image, **_):
    from rsstatchem.core.imaging.results import ResultsTable
    if image.overlay is None or not len(image.overlay):
        raise CommandError("Overlay required")
    rt = ResultsTable(f"Overlay Elements of {image.title}")
    for i, roi in enumerate(image.overlay):
        x, y, w, h = roi.bounds()
        c, z, t = roi.position
        rt.add_row({"Index": i, "X": x, "Y": y, "Width": w, "Height": h, "C": c, "Z": z,
                    "T": t, "Stroke Width": roi.stroke_width}, label=roi.name or roi.type_name)
        rt.set_value("Name", i, roi.name or "")
        rt.set_value("Type", i, roi.type_name)
        rt.set_value("Color", i, roi.stroke_color or "")
        rt.set_value("Fill", i, roi.fill_color or "")
    return rt


def flatten_cmd(session, image, **_):
    """Image > Overlay > Flatten: an RGB image with the overlay (and selection) burnt in."""
    rgb = flatten(image, session.options.line_width)
    out = ImageDoc(rgb[None, None, None], COLOR_RGB, _new_title(image.title, "-1"))
    out.calibration = image.calibration.copy()
    return out


def _labels_fields(session, image):
    ov = image.overlay if image else None
    return [C.choice("Color:", ("white", "black", "red", "green", "blue", "yellow", "cyan",
                                "magenta"), (ov.label_color if ov and ov.label_color
                                             else "white"), key="color"),
            C.number("Font size:", ov.label_font_size if ov else 9, 0, key="font",
                     integer=True),
            C.checkbox("Show labels", ov.show_labels if ov else True, key="show"),
            C.checkbox("Use names as labels", ov.draw_names if ov else False, key="use"),
            C.checkbox("Draw backgrounds", ov.draw_backgrounds if ov else False,
                       key="draw"),
            C.checkbox("Bold", False, key="bold")]


def labels_cmd(session, image, color="white", font=9, show=False, use=False, draw=False,
               **_):
    ov = _overlay(image)
    ov.label_color = color
    ov.label_font_size = int(font)
    ov.show_labels = bool(show)
    ov.draw_names = bool(use)
    ov.draw_backgrounds = bool(draw)


def measure_overlay_cmd(session, image, **_):
    from rsstatchem.core.imaging.results import ResultsTable
    if image.overlay is None or not len(image.overlay):
        raise CommandError("Overlay required")
    saved = image.roi
    rt = session.results
    try:
        from rsstatchem.core.imaging import measure as M
        fn = getattr(M, "measure", None)
    except ImportError:
        fn = None
    for roi in image.overlay:
        image.roi = roi
        if fn is not None:
            try:
                fn(session, image)
                continue
            except Exception:  # noqa: BLE001
                pass
        if not roi.is_area():
            continue
        m = roi.get_mask((image.height, image.width))
        p = image.plane().astype(np.float64)
        if p.ndim == 3:
            p = p.mean(axis=-1)
        v = p[m]
        if v.size:
            cal = image.calibration
            rt.add_row({"Area": float(m.sum()) * cal.pixel_width * cal.pixel_height,
                        "Mean": float(v.mean()), "Min": float(v.min()),
                        "Max": float(v.max())}, label=roi.name or image.title)
    image.roi = saved
    return rt if isinstance(rt, ResultsTable) else None


def _overlay_options_fields(session, image):
    d = session.options.extra.get("overlay", {})
    return [C.string("Stroke color:", d.get("stroke", "cyan"), key="stroke"),
            C.number("Width:", d.get("width", 0), 1, key="width"),
            C.string("Fill color:", d.get("fill", "none"), key="fill"),
            C.checkbox("Set stack positions", d.get("position", True), key="set"),
            C.checkbox("Apply to current overlay", False, key="apply")]


_NAMED = {"white": "#ffffff", "black": "#000000", "red": "#ff0000", "green": "#00ff00",
          "blue": "#0000ff", "yellow": "#ffff00", "cyan": "#00ffff", "magenta": "#ff00ff",
          "orange": "#ffc800", "pink": "#ffafaf", "gray": "#808080", "lightgray": "#c0c0c0",
          "darkgray": "#404040"}


def _color_value(text: str) -> str | None:
    text = (text or "").strip().lower()
    if not text or text == "none":
        return None
    if text in _NAMED:
        return _NAMED[text]
    if text.startswith("#"):
        return text
    return None


def overlay_options_cmd(session, image, stroke="cyan", width=0.0, fill="none", set=False,
                        apply=False, **_):
    d = {"stroke": _color_value(stroke) or "#00ffff", "width": float(width),
         "fill": _color_value(fill), "position": bool(set)}
    session.options.extra["overlay"] = d
    if apply and image is not None and image.overlay is not None:
        for roi in image.overlay:
            roi.stroke_color = d["stroke"]
            roi.stroke_width = d["width"]
            roi.fill_color = d["fill"]


# ---------------------------------------------------------------------------------- overlay data

@dataclass
class ImageRoi(Roi):
    """An image overlay element (ImageJ ImageRoi): RGB pixels drawn at (x, y) with opacity."""
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    pixels: np.ndarray = field(default_factory=lambda: np.zeros((1, 1, 3), np.uint8))
    opacity: float = 1.0
    zero_transparent: bool = False
    type: int = field(default=RECTANGLE, kw_only=True)

    def bounds(self):
        return float(self.x), float(self.y), float(self.width), float(self.height)

    def polygon(self):
        x, y, w, h = self.bounds()
        return np.array([x, x + w, x + w, x]), np.array([y, y, y + h, y + h])

    def is_area(self) -> bool:
        return False

    def translate(self, dx, dy):
        self.x += dx
        self.y += dy


def _rgb_of(color: str | None, default=(0, 255, 255)) -> tuple[int, int, int]:
    if not color:
        return default
    c = color.lstrip("#")
    if len(c) == 8:
        c = c[2:]
    try:
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    except ValueError:
        return default


def flatten(image: ImageDoc, line_width: float = 1.0, include_roi: bool = False) -> np.ndarray:
    """The displayed image (current position) with its overlay drawn, as (H, W, 3) uint8.
    Drawn with Pillow so it works headless (ImageJ's Java2D anti-aliasing is not
    reproduced)."""
    from PIL import Image, ImageDraw, ImageFont
    from rsstatchem.core.imaging.lut import render
    from rsstatchem.core.imaging.roi import shapes as S
    rgb = np.ascontiguousarray(render(image))
    base = Image.fromarray(rgb)
    ov = image.overlay
    rois = list(ov) if (ov is not None and not ov.hidden) else []
    if include_roi and image.roi is not None:
        rois.append(image.roi)
    c0, z0, t0 = image.position
    draw = ImageDraw.Draw(base)
    for k, roi in enumerate(rois):
        pc, pz, pt = roi.position
        if image.stack_size > 1:
            if image.n_channels == 1 and image.n_frames == 1 and pz and \
                    pz != image.current_index + 1:
                continue
            if (pc and pc != c0 + 1 and image.n_channels > 1) or \
                    (pt and pt != t0 + 1 and image.n_frames > 1) or \
                    (pz and image.n_slices > 1 and pz != z0 + 1 and
                     (image.n_channels > 1 or image.n_frames > 1)):
                continue
        color = _rgb_of(roi.stroke_color)
        w = max(1, int(round(roi.stroke_width or line_width)))
        fill = _rgb_of(roi.fill_color) if roi.fill_color else None
        if isinstance(roi, ImageRoi):
            src = Image.fromarray(np.ascontiguousarray(roi.pixels))
            alpha = int(round(255 * roi.opacity))
            mask = Image.new("L", src.size, alpha)
            if roi.zero_transparent:
                z = (np.asarray(roi.pixels).sum(axis=-1) == 0)
                m = np.where(z, 0, alpha).astype(np.uint8)
                mask = Image.fromarray(m)
            base.paste(src, (int(roi.x), int(roi.y)), mask)
            continue
        if isinstance(roi, S.TextRoi):
            try:
                fnt = ImageFont.load_default(size=int(roi.font_size))
            except TypeError:
                fnt = ImageFont.load_default()
            draw.text((roi.x, roi.y), roi.text, fill=_rgb_of(roi.stroke_color, (255, 255, 255)),
                      font=fnt)
            continue
        if isinstance(roi, S.OvalRoi):
            box = [roi.x, roi.y, roi.x + roi.width - 1, roi.y + roi.height - 1]
            draw.ellipse(box, outline=None if fill else color, fill=fill, width=w)
        elif isinstance(roi, S.RectRoi):
            box = [roi.x, roi.y, roi.x + roi.width - 1, roi.y + roi.height - 1]
            draw.rectangle(box, outline=None if fill else color, fill=fill, width=w)
        elif isinstance(roi, S.PointRoi):
            for x, y in zip(roi.xs, roi.ys):
                draw.rectangle([x - 1, y - 1, x + 1, y + 1], fill=color)
        elif isinstance(roi, S.CompositeRoi):
            for _, part in roi.parts:
                xs, ys = part.polygon()
                pts = list(zip(xs.tolist(), ys.tolist()))
                if len(pts) > 1:
                    draw.polygon(pts, outline=color)
        else:
            xs, ys = roi.polygon()
            pts = list(zip(xs.tolist(), ys.tolist()))
            if len(pts) > 1:
                if roi.is_area():
                    draw.polygon(pts, outline=None if fill else color, fill=fill, width=w)
                else:
                    draw.line(pts, fill=color, width=w)
        if ov is not None and ov.show_labels and roi in ov.rois:
            label = roi.name if (ov.draw_names and roi.name) else str(k + 1)
            x, y, bw, bh = roi.bounds()
            draw.text((x + bw / 2 - 3, y + bh / 2 - 5), label,
                      fill=_rgb_of(_color_value(ov.label_color or "white"), (255, 255, 255)))
    return np.asarray(base, dtype=np.uint8).copy()


# ---------------------------------------------------------------------------------- LUTs
def _lut_command(name: str):
    def run(session, image, **_):
        lut = named_lut(name)
        if image.type == COLOR_RGB:
            raise CommandError("LUTs cannot be assiged to RGB Images.")
        luts = list(image.luts)
        if image.is_composite and image.composite_mode == "grayscale":
            image.composite_mode = "color"
            image.luts = [lut.copy() for _ in luts]
            return None
        luts[image.position[0]] = lut
        image.luts = luts
        if image.type == COLOR_256 and name in ("Grays",):
            image.type = GRAY8
        if name == "Grays":
            image.properties.pop("LUT", None)
        else:
            image.properties["LUT"] = name
    return run


def invert_lut_cmd(session, image, **_):
    if image.type == COLOR_RGB:
        raise CommandError("RGB images do not use LUTs")
    luts = list(image.luts)
    c = image.position[0]
    luts[c] = luts[c].inverted()
    image.luts = luts


def apply_lut_cmd(session, image, stack=False, **_):
    from rsstatchem.core.imaging import adjust as A
    if image.type == GRAY32:
        raise CommandError('"Apply LUT" requires an image of type:\n \n8-bit, 16-bit or RGB')
    if image.threshold is not None:
        raise CommandError("Thresholded image: use Process > Binary > Convert to Mask")
    mask = _nonrect_mask(image)
    A.apply_lut(image, all_slices=bool(stack) or image.stack_size == 1, mask=mask)


# ---------------------------------------------------------------------------------- B&C etc.
def _bc_ui(mode: str):
    def action(bar, cmd) -> None:
        from rsstatchem.ui.imagej.frames.bc_adjuster import show_adjuster
        show_adjuster(bar, mode)
    return action


def _channels_ui(bar, cmd) -> None:
    from rsstatchem.ui.imagej.frames.channels_tool import show_channels_tool
    show_channels_tool(bar)


def _ortho_ui(bar, cmd) -> None:
    image = bar.session.active
    if image is None or image.n_slices < 2:
        bar.error("Stack required", "Orthogonal Views")
        return
    from rsstatchem.ui.imagej.frames.ortho_views import show_ortho_views
    show_ortho_views(bar, image)


def set_display_range_cmd(session, image, min=0.0, max=255.0, **_):
    """Headless B&C "Set" (macro setMinAndMax equivalent used by the B&C window)."""
    if image.type == COLOR_RGB:
        image.set_display_range(float(min), float(max))
        return
    cal = image.calibration
    lo = cal.raw_value(float(min), image.bit_depth) if cal.calibrated() else float(min)
    hi = cal.raw_value(float(max), image.bit_depth) if cal.calibrated() else float(max)
    image.set_display_range(lo, hi)


# ---------------------------------------------------------------------------------- Fiji extras
def _bleach_fields(session, image):
    from rsstatchem.core.imaging.fiji.bleach import METHODS
    return [C.choice("Correction Method", METHODS, "Simple Ratio", key="correction"),
            C.number("Background Intensity", 0.0, 1, key="background")]


def bleach_cmd(session, image, correction="Simple Ratio", background=0.0, **_):
    """Bleach Correction (simple ratio, exponential fit, histogram matching) of the frames (or
    slices) of every channel; returns a corrected copy "DUP_<title>"."""
    from rsstatchem.core.imaging.fiji import bleach as B
    from rsstatchem.core.imaging.results import ResultsTable
    if image.stack_size < 2 or image.type == COLOR_RGB:
        raise CommandError("A grayscale time series (stack) is required")
    out = image.duplicate("DUP_" + image.title)
    mask = image.roi.get_mask((image.height, image.width)) if (
        image.roi is not None and image.roi.is_area()) else None
    over_t = image.n_frames > 1
    rt = None
    for c in range(image.n_channels):
        if over_t:
            for z in range(image.n_slices):
                frames = image.pixels[:, z, c]
                res, info = B.correct(frames, correction, float(background), mask)
                out.pixels[:, z, c] = res
        else:
            for t in range(image.n_frames):
                frames = image.pixels[t, :, c]
                res, info = B.correct(frames, correction, float(background), mask)
                out.pixels[t, :, c] = res
        if info:
            rt = rt or ResultsTable("Bleach Correction fit")
            rt.add_row({"Channel": c + 1, **info})
    out.reset_display_range()
    return [out, rt] if rt is not None else out


def _deconv_fields(session, image):
    from rsstatchem.core.imaging.color import STAINS
    return [C.choice("Vectors", tuple(STAINS), "H&E", key="vectors"),
            C.choice("Output", ("8bit_Transmittance", "32bit_Absorbance"),
                     "8bit_Transmittance", key="output"),
            C.checkbox("Show matrix", False, key="show"),
            C.checkbox("Hide legend", True, key="hide")]


def deconvolution_cmd(session, image, vectors="H&E", output="8bit_Transmittance", show=False,
                      **_):
    """Colour Deconvolution (Ruifrok & Johnston 2001) of an RGB image: three stain images
    "<title>-(Colour_1..3)"; optionally the stain matrix as a table."""
    from rsstatchem.core.imaging import color as CL
    from rsstatchem.core.imaging.results import ResultsTable
    if image.type != COLOR_RGB:
        raise CommandError("RGB image required")
    planes, m = CL.colour_deconvolution(image.plane(), vectors,
                                        "8bit_transmittance" if output.startswith("8")
                                        else "32bit_OD")
    outs = []
    for k, p in enumerate(planes):
        im = ImageDoc(p[None, None, None], GRAY8 if p.dtype == np.uint8 else GRAY32,
                      f"{image.title}-(Colour_{k + 1})")
        im.calibration = image.calibration.copy()
        if p.dtype == np.uint8:
            im.luts = [CL.stain_lut(m[k])]
        outs.append(im)
    if show:
        rt = ResultsTable(f"Colour Deconvolution matrix ({vectors})")
        for k in range(3):
            rt.add_row({"R": m[k, 0], "G": m[k, 1], "B": m[k, 2]}, label=f"Colour_{k + 1}")
        outs.append(rt)
    return outs


def _kymo_fields(session, image):
    return [C.number("Line width:", max(1, int(session.options.line_width)), 0, key="line",
                     integer=True)]


def kymograph_cmd(session, image, line=1, **_):
    """KymographBuilder: distance along the line/polyline (x) against time (y), one kymograph
    per channel."""
    from rsstatchem.core.imaging.fiji.kymograph import kymograph, to_dtype
    roi = image.roi
    if roi is None or not roi.is_line():
        raise CommandError("Line or polyline selection required")
    xs, ys = roi.polygon()
    width = int(getattr(roi, "width", 0) or 0) or int(line)
    over_t = image.n_frames > 1
    rows = []
    for c in range(image.n_channels):
        frames = image.pixels[:, image.position[1], c] if over_t else \
            image.pixels[image.position[2], :, c]
        k = kymograph(frames, xs, ys, max(width, 1))
        rows.append(to_dtype(k, image.pixels.dtype))
    arr = np.stack(rows)                       # (C, T, L[, 3])
    out = ImageDoc(np.ascontiguousarray(arr[None, None]), image.type,
                   f"Kymograph of {image.title}")
    if image.type != COLOR_RGB:
        out.luts = [lut.copy() for lut in image.luts]
        out.composite_mode = image.composite_mode
        out.reset_display_range()
    cal = out.calibration
    cal.pixel_width = image.calibration.pixel_width
    cal.unit = image.calibration.unit
    cal.pixel_height = image.calibration.frame_interval or 1.0
    return out


# ---------------------------------------------------------------------------------- register
def _cmd(registry, label, func=None, fields=(), **kw):
    registry.register(Command(label, func, fields if callable(fields) else list(fields), **kw),
                      replace=True)


def register(registry) -> None:
    U = dict(undoable=False)
    # Type
    for item in ("8-bit", "16-bit", "32-bit", "RGB Color", "RGB Stack", "HSB Stack",
                 "HSB (32-bit)", "Lab Stack"):
        _cmd(registry, item, _type_command(item), roi="optional")
    _cmd(registry, "8-bit Color", _type_command("8-bit Color"), _color_count_fields,
         types=(COLOR_RGB,))
    # Adjust
    _cmd(registry, "Brightness/Contrast...", None, ui_action=_bc_ui("bc"), **U)
    _cmd(registry, "Window/Level...", None, ui_action=_bc_ui("wl"), **U)
    _cmd(registry, "Color Balance...", None, ui_action=_bc_ui("balance"), **U)
    _cmd(registry, "Size...", size_cmd, _size_fields)
    _cmd(registry, "Canvas Size...", canvas_cmd, _canvas_fields)
    _cmd(registry, "Line Width... ", line_width_cmd,
         [C.number("Line width:", 1, 0, key="line")], needs_image=False, undoable=False)
    _cmd(registry, "Coordinates...", coordinates_cmd, _coord_fields)
    _cmd(registry, "Show Info...", show_info_cmd, ui_action=_show_info_ui, **U)
    _cmd(registry, "Properties...", properties_cmd, _properties_fields)
    # Color
    _cmd(registry, "Split Channels", split_channels_cmd, **U)
    _cmd(registry, "Merge Channels...", merge_channels_cmd, _merge_fields, needs_image=True,
         undoable=False)
    _cmd(registry, "Arrange Channels...", arrange_cmd,
         [C.string("New channel order:", "123", key="new")])
    _cmd(registry, "Invert LUTs", invert_luts_cmd, types=GRAY + (COLOR_256,))
    _cmd(registry, "Channels Tool...", None, ui_action=_channels_ui, needs_image=False, **U)
    _cmd(registry, "Stack to RGB", stack_to_rgb_cmd, _stack_to_rgb_fields, **U)
    _cmd(registry, "Make Composite", make_composite_cmd,
         [C.choice("Display Mode:", ("Composite", "Color", "Grayscale"), "Composite",
                   key="display")])
    _cmd(registry, "Show LUT", show_lut_cmd, **U)
    _cmd(registry, "Display LUTs", display_luts_cmd, needs_image=False, **U)
    _cmd(registry, "Edit LUT...", None, ui_action=_edit_lut_ui, **U)
    _cmd(registry, "Color Picker...", None, ui_action=_color_picker_ui, needs_image=False,
         **U)
    # Stacks
    _cmd(registry, "Add Slice", add_slice_cmd)
    _cmd(registry, "Delete Slice", delete_slice_cmd)
    _cmd(registry, "Next Slice [>]", _step_slice(1), undoable=False, recordable=False)
    _cmd(registry, "Previous Slice [<]", _step_slice(-1), undoable=False, recordable=False)
    _cmd(registry, "Set Slice...", set_slice_cmd,
         [C.number("Slice Number:", 1, 0, key="slice", integer=True)], **U)
    _cmd(registry, "Images to Stack", images_to_stack_cmd, _images_to_stack_fields, **U)
    _cmd(registry, "Stack to Images", stack_to_images_cmd, **U)
    _cmd(registry, "Make Montage...", montage_cmd, _montage_fields, **U)
    _cmd(registry, "Reslice [/]...", reslice_cmd, _reslice_fields, **U)
    _cmd(registry, "Orthogonal Views", None, ui_action=_ortho_ui, **U)
    _cmd(registry, "Z Project...", zproject_cmd, _zproject_fields, **U)
    _cmd(registry, "3D Project...", project3d_cmd, _project3d_fields,
         types=ALL, **U)
    _cmd(registry, "Plot Z-axis Profile", zprofile_cmd, **U)
    _cmd(registry, "Measure Stack...", measure_stack_cmd, **U)
    _cmd(registry, "Label...", label_cmd, _label_fields)
    _cmd(registry, "Statistics", statistics_cmd, **U)
    _cmd(registry, "Start Animation [\\]", _noop, ui_action=_start_animation_ui,
         recordable=False, **U)
    _cmd(registry, "Stop Animation", _noop, ui_action=_stop_animation_ui, recordable=False,
         **U)
    _cmd(registry, "Animation Options...", animation_options_cmd, _animation_fields, **U)
    _cmd(registry, "Combine...", combine_cmd, _combine_fields, **U)
    _cmd(registry, "Concatenate...", concatenate_cmd, _concat_fields, **U)
    _cmd(registry, "Grouped Z Project...", grouped_cmd, _grouped_fields, **U)
    _cmd(registry, "Insert...", insert_cmd, _insert_fields, **U)
    _cmd(registry, "Magic Montage Tools", None, ui_action=_magic_montage_ui, needs_image=False,
         **U)
    _cmd(registry, "Make Substack...", substack_cmd,
         [C.message("Enter a range (e.g. 2-14), a range with increment\n(e.g. 1-100-2) or a "
                    "list (e.g. 7,9,25,27)"),
          C.string("Slices:", "", key="slices", columns=40),
          C.checkbox("Delete slices from original stack", False, key="delete"),
          C.checkbox("Do_not create substack", False, key="do_not")])
    _cmd(registry, "Montage to Stack...", m2s_cmd, _m2s_fields, **U)
    _cmd(registry, "Plot XY Profile", xy_profile_cmd, roi="line", **U)
    _cmd(registry, "Reduce...", reduce_cmd,
         [C.number("Reduction Factor:", 2, 0, key="reduction", integer=True),
          C.checkbox("Reduce in Z-Dimension", False, key="reduce")])
    _cmd(registry, "Remove Slice Labels", remove_labels_cmd)
    _cmd(registry, "Reverse", reverse_cmd)
    _cmd(registry, "Set Label...", set_label_cmd, lambda s, im: [
        C.string("Label:", (im.slice_label() or "") if im else "", key="label", columns=30)])
    # Hyperstacks
    _cmd(registry, "New Hyperstack...", new_hyper_cmd, _new_hyper_fields, needs_image=False,
         **U)
    _cmd(registry, "Stack to Hyperstack...", s2h_cmd, _s2h_fields)
    _cmd(registry, "Hyperstack to Stack", h2s_cmd)
    _cmd(registry, "Reduce Dimensionality...", reduce_dim_cmd, _reduce_dim_fields, **U)
    _cmd(registry, "Make Subset...", subset_cmd, _subset_fields, **U)
    # Crop, Duplicate, Rename, Scale
    _cmd(registry, "Crop", crop_cmd, roi="area")
    _cmd(registry, "Duplicate...", duplicate_cmd, _duplicate_fields, **U)
    _cmd(registry, "Rename...", rename_cmd, lambda s, im: [
        C.string("Title:", im.title if im else "", key="title", columns=20)], **U)
    _cmd(registry, "Scale...", scale_cmd, _scale_fields)
    # Transform
    _cmd(registry, "Flip Horizontally", _flip("h"), supports_stack=True)
    _cmd(registry, "Flip Vertically", _flip("v"), supports_stack=True)
    _cmd(registry, "Flip Z", flip_z_cmd)
    _cmd(registry, "Rotate 90 Degrees Right", _rotate90("r"))
    _cmd(registry, "Rotate 90 Degrees Left", _rotate90("l"))
    _cmd(registry, "Rotate... ", rotate_cmd, _rotate_fields)
    _cmd(registry, "Translate...", translate_cmd, _translate_fields)
    _cmd(registry, "Bin...", bin_cmd, _bin_fields)
    _cmd(registry, "Image to Results", image_to_results_cmd, **U)
    _cmd(registry, "Results to Image", results_to_image_cmd, needs_image=False, **U)
    # Zoom (window commands; no-ops in batch macros, as in ImageJ's batch mode)
    for label, kind in (("In [+]", "in"), ("Out [-]", "out"), ("Original Scale", "original"),
                        ("View 100%", "100"), ("To Selection", "selection"),
                        ("Scale to Fit", "fit"), ("Set... ", "set"), ("Maximize", "max")):
        _cmd(registry, label, _noop, ui_action=_zoom_ui(kind), recordable=False, **U)
    # Overlay
    _cmd(registry, "Add Selection...", add_selection_cmd, roi="required")
    _cmd(registry, "Add Image...", add_image_cmd, _add_image_fields)
    _cmd(registry, "Hide Overlay", _overlay_flag("hidden", True), **U)
    _cmd(registry, "Show Overlay", _overlay_flag("hidden", False), **U)
    _cmd(registry, "From ROI Manager", from_manager_cmd)
    _cmd(registry, "To ROI Manager", to_manager_cmd)
    _cmd(registry, "Remove Overlay", remove_overlay_cmd)
    _cmd(registry, "Toggle Overlay", _overlay_flag("hidden", None), **U)
    _cmd(registry, "List Elements", list_elements_cmd, **U)
    _cmd(registry, "Flatten", flatten_cmd, **U)
    _cmd(registry, "Labels...", labels_cmd, _labels_fields)
    _cmd(registry, "Measure Overlay", measure_overlay_cmd, **U)
    _cmd(registry, "Overlay Options...", overlay_options_cmd, _overlay_options_fields,
         needs_image=False, **U)
    # Lookup tables
    _cmd(registry, "Invert LUT", invert_lut_cmd)
    _cmd(registry, "Apply LUT", apply_lut_cmd, types=(GRAY8, GRAY16, COLOR_256, COLOR_RGB),
         supports_stack=True)
    for name in lut_tables.IMAGEJ_LUTS:
        _cmd(registry, name, _lut_command(name), types=GRAY + (COLOR_256,))
    # extras (Fiji-style, not in the ImageJ 1.54 menus)
    _cmd(registry, "Bleach Correction", bleach_cmd, _bleach_fields, **U)
    _cmd(registry, "Colour Deconvolution", deconvolution_cmd, _deconv_fields,
         types=(COLOR_RGB,), **U)
    _cmd(registry, "KymographBuilder", kymograph_cmd, _kymo_fields, roi="line", **U)
    _cmd(registry, "Set Display Range", set_display_range_cmd,
         [C.number("Minimum displayed value:", 0, 3, key="min"),
          C.number("Maximum displayed value:", 255, 3, key="max")], recordable=True)


# ---------------------------------------------------------------------------------- selftest
def selftest() -> None:
    """Headless checks of the Image menu ports (no Qt widgets)."""
    from rsstatchem.core.imaging import adjust as A
    from rsstatchem.core.imaging import convert as CV
    from rsstatchem.core.imaging import project3d as P
    from rsstatchem.core.imaging import stacks as S
    from rsstatchem.core.imaging import transform as T
    from rsstatchem.core.imaging.session import Session

    fire = named_lut("Fire").as_array()
    assert fire.shape == (256, 3) and tuple(fire[255]) == (255, 255, 255)
    a = np.arange(64, dtype=np.uint16).reshape(8, 8) * 1000
    doc = ImageDoc.from_array(a, "YX", "t16")
    CV.convert(doc, "8-bit")
    assert doc.type == GRAY8 and doc.pixels.max() == 255 and doc.pixels.min() == 0
    p = np.arange(100, dtype=np.uint8).reshape(10, 10)
    up = T.resize_plane(p, "byte", 20, 20, T.BICUBIC)
    assert up.shape == (20, 20)
    assert T.rotate_plane(p, "byte", 90, T.NONE).shape == (10, 10)
    st = ImageDoc.from_array(np.stack([p, p + 1, p + 2]), "ZYX", "s")
    assert S.z_project(st, "Max Intensity").pixels.max() == 101
    assert P.project(st, total=0, increment=0).stack_size == 1
    assert A.auto_range(p) is not None
    session = Session()
    register(session.registry)
    session.add_image(ImageDoc.from_array(p, "YX", "x"))
    session.run("Scale...", "x=0.5 y=0.5 interpolation=Bilinear average create title=half")
    assert session.active.width == 5

"""Process > Math, Process > Image Calculator... and Process > Enhance Contrast... **WP-B.**

Headless (no Qt). The algorithms are the exact ImageJ ports in
:mod:`rsstatchem.core.imaging.math_ops` and :mod:`rsstatchem.core.imaging.contrast`.
Macro keys are ImageJ's: ``run("Add...", "value=25 stack")``,
``run("AND...", "value=11110000")`` (binary string), ``run("Macro...", "code=v=v*2")``,
``run("Image Calculator...", "image1=a.tif operation=Add image2=b.tif create 32-bit")``,
``run("Enhance Contrast...", "saturated=0.35 normalize equalize process_all use")``.
"""
from __future__ import annotations

import math

import numpy as np

from rsstatchem.core.imaging import commands as C
from rsstatchem.core.imaging import contrast as CE
from rsstatchem.core.imaging import math_ops as M
from rsstatchem.core.imaging.commands import CommandError
from rsstatchem.core.imaging.image import COLOR_RGB, GRAY8, GRAY16, GRAY32, ImageDoc
from rsstatchem.ui.imagej.dialogs.process_common import (ALL_TYPES, apply_filter, roi_mask,
                                                         roi_rect)

DEFAULT_MACRO = "v=v+50*sin(d/10)"

# label -> (op, field factory)
_VALUE_OPS = {
    "Add...": ("add", lambda: [C.number("Value:", 25.0, 0)]),
    "Subtract...": ("subtract", lambda: [C.number("Value:", 25.0, 0)]),
    "Multiply...": ("multiply", lambda: [C.number("Value:", 1.25, 2)]),
    "Divide...": ("divide", lambda: [C.number("Value:", 1.25, 2)]),
    "AND...": ("and", lambda: [C.string("Value (binary):", "11110000")]),
    "OR...": ("or", lambda: [C.string("Value (binary):", "11110000")]),
    "XOR...": ("xor", lambda: [C.string("Value (binary):", "11110000")]),
    "Min...": ("min", lambda: [C.number("Value:", 0.0, 0)]),
    "Max...": ("max", lambda: [C.number("Value:", 255.0, 0)]),
    "Gamma...": ("gamma", lambda: [C.slider("Value:", 0.0, 5.0, 0.5, 2)]),
    "Set...": ("set", lambda: [C.number("Value:", 25.0, 0)]),
}
_PLAIN_OPS = {"Log": "log", "Exp": "exp", "Square": "square", "Square Root": "sqrt",
              "Reciprocal": "reciprocal", "NaN Background": "nan_background", "Abs": "abs"}


# ---------------------------------------------------------------------------------- Math
def run_math(session, image: ImageDoc, op: str, value=None, stack: bool = False) -> None:
    """Apply one Process > Math operation like ImageMath (current plane or all planes, inside
    an area selection only)."""
    signed = image.calibration.is_signed16()
    if op == "gamma" and (value is None or not 0.05 <= float(value) <= 5.0):
        raise CommandError("Gamma must be between 0.05 and 5.0")
    if op in ("and", "or", "xor"):
        try:
            value = M.parse_binary(value)
        except ValueError:
            raise CommandError("Binary number required") from None
    if op == "reciprocal" and image.type != GRAY32:
        session.status("Reciprocal: 32-bit image required") if session else None
        return
    if op == "abs" and not (image.type == GRAY32 or signed):
        session.status("Abs: 32-bit or signed 16-bit image required") if session else None
        return
    threshold = None
    if op == "nan_background":
        if image.type != GRAY32 or image.threshold is None:
            raise CommandError(f'Thresholded 32-bit float image required:\n"{image.title}"')
        threshold = (image.threshold.lower, image.threshold.upper)
    display = image.display_range if image.type == GRAY16 else None
    apply_filter(image, lambda p: M.math_op(p, op, value, display, signed, threshold),
                 bool(stack), rgb="whole", progress=_progress(session))
    if op in ("min", "max", "abs", "nan_background") and image.type in (GRAY16, GRAY32):
        image.reset_display_range()


def _progress(session):
    return getattr(session, "progress", None) if session is not None else None


def _value_command(op):
    def func(session, image, value=None, stack=False):
        run_math(session, image, op, value, stack)
    func.__name__ = f"math_{op}"
    return func


def _plain_command(op):
    def func(session, image, stack=False):
        run_math(session, image, op, None, stack)
    func.__name__ = f"math_{op}"
    return func


def math_macro(session, image: ImageDoc, code: str = DEFAULT_MACRO, stack: bool = False
               ) -> None:
    """Process > Math > Macro... (``ImageMath.applyMacro``)."""
    code = str(code).strip() or DEFAULT_MACRO
    try:
        M.parse_macro(code)
    except M.MacroError as exc:
        raise CommandError(f"Macro error:\n{exc}") from None
    rect = roi_rect(image)
    mask = None if M.macro_uses_getpixel(code) else roi_mask(image)
    signed = image.calibration.is_signed16()
    planes = list(image.iter_planes(bool(stack)))
    for i, (c, z, t) in enumerate(planes):
        plane = image.plane(c, z, t)
        try:
            out = M.macro_expression(plane, code, image.stack_index(c, z, t), rect, signed)
        except M.MacroError as exc:
            raise CommandError(f"Macro error:\n{exc}") from None
        if mask is not None:
            out = np.where(mask[..., None] if plane.ndim == 3 else mask, out, plane)
        image.set_plane(out, c, z, t)
        if session is not None and len(planes) > 1:
            session.progress((i + 1) / len(planes))
    if image.type in (GRAY16, GRAY32):
        image.reset_display_range()


# ---------------------------------------------------------------------------------- calculator
def _calc_fields(session, image):
    titles = [im.title for im in session.images] if session is not None else []
    if not titles and image is not None:
        titles = [image.title]
    default = image.title if image is not None and image.title in titles else \
        (titles[0] if titles else "")
    fields = [C.choice("Image1:", titles or [""], default),
              C.choice("Operation:", M.CALC_OPERATORS, "Add"),
              C.choice("Image2:", titles or [""], default),
              C.checkbox("Create new window", True),
              C.checkbox("32-bit (float) result", False, key="32-bit")]
    if any(im.stack_size > 1 for im in (session.images if session else [])):
        fields.append(C.checkbox("Process all slices", False, key="stack"))
    return fields


def _find(session, title):
    im = session.find_image(title) if session is not None else None
    if im is None:
        raise CommandError(f'Image not found: "{title}"')
    return im


def image_calculator(session, image, image1=None, operation="Add", image2=None,
                     create=False, stack=False, **kw):
    """Process > Image Calculator (``ImageCalculator.calculate``). Returns the new image
    with "create" (or a 32-bit result of non-32-bit inputs), else changes image1."""
    float_result = bool(kw.get("32-bit", False) or kw.get("float_result", False))
    img1 = _find(session, image1) if image1 else image
    img2 = _find(session, image2) if image2 else image
    if img1 is None or img2 is None:
        raise CommandError("There are no images open")
    op = str(operation)
    try:
        M.calculator_mode(op)
    except ValueError:
        raise CommandError("No valid operator") from None
    dbz = session.options.divide_by_zero if session is not None else math.inf
    return calculate(img1, img2, op, bool(create), float_result, bool(stack), dbz,
                     session=session)


def calculate(img1: ImageDoc, img2: ImageDoc, operation: str, create: bool = False,
              float_result: bool = False, stack: bool = False,
              divide_by_zero: float = math.inf, session=None) -> ImageDoc | None:
    """Headless Image Calculator: returns the new ImageDoc, or None after changing img1."""
    if img1.calibration.is_signed16() or img2.calibration.is_signed16():
        float_result = True
    if float_result and not (img1.type == GRAY32 and img2.type == GRAY32):
        create = True
    size1, size2 = img1.stack_size, img2.stack_size
    stack_op = stack and (size1 > 1 or size2 > 1)
    if stack_op and size1 > 1 and size2 > 1 and size1 != size2:
        raise CommandError("'Image1' and 'image2' must be stacks with the same\nnumber of "
                           "slices, or 'image2' must be a single image.")
    if stack_op and size1 == 1:
        raise CommandError("'Image1' must be a stack when 'image2' is a stack")

    def args_for(i1, i2):
        return dict(float_result=float_result, divide_by_zero=divide_by_zero,
                    range1=img1.display_ranges[img1.position_of(i1)[0]] if not img1.is_rgb
                    else None,
                    range2=img2.display_ranges[img2.position_of(i2)[0]] if not img2.is_rgb
                    else None,
                    lut1=None, lut2=None if img2.is_rgb else
                    img2.luts[img2.position_of(i2)[0]].as_array(),
                    ctable1=img1.calibration.ctable(img1.bit_depth),
                    ctable2=img2.calibration.ctable(img2.bit_depth))

    if not stack_op:
        i1, i2 = img1.current_index, img2.current_index
        p1, p2 = img1.plane_at(i1), img2.plane_at(i2)
        res = M.calculate_planes(p1, p2, operation, create=create, **args_for(i1, i2))
        if create:
            if res.ndim == 3:
                pix = res[None, None, None]
            else:
                pix = res[None, None, None]
            out = ImageDoc(pix, None, f"Result of {img1.title}", img1.calibration.copy())
            if out.type == GRAY8 and not img1.is_rgb:
                out.luts = [img1.luts[img1.position[0]].copy()]
            if out.type not in (GRAY8, COLOR_RGB):
                out.reset_display_range()
            return out
        if session is not None:
            img1.snapshot("Image Calculator...")
        img1.set_plane(res, *img1.position)
        _touch(session, img1)
        return None
    # stack operation: duplicate (with create) then copyBits slice by slice
    if create:
        target = img1.duplicate(f"Result of {img1.title}")
        target.roi = None
        if float_result and target.type != GRAY32:
            ct = img1.calibration.ctable(img1.bit_depth)
            target.replace_pixels(np.stack([M.to_float(target.plane_at(k), ct)
                                            for k in range(target.stack_size)]
                                           ).reshape(target.pixels.shape[:5]), GRAY32)
            target.calibration = img1.calibration.copy()
    else:
        target = img1
        if session is not None:
            img1.snapshot("Image Calculator...")
    for k in range(target.stack_size):
        k2 = k if size2 > 1 else img2.current_index
        p1 = target.plane_at(k)
        p2 = img2.plane_at(k2)
        a = args_for(k, k2)
        a["float_result"] = False   # target already converted
        res = M.calculate_planes(p1, p2, operation, **a)
        target.set_plane(res, *target.position_of(k))
        if session is not None and target.stack_size > 1:
            session.progress((k + 1) / target.stack_size)
    if target.type != GRAY8:
        target.reset_display_range()
    if create:
        return target
    _touch(session, img1)
    return None


def _touch(session, img):
    img.changed = True
    if session is not None and img in session.images:
        session.image_updated(img)


# ---------------------------------------------------------------------------------- contrast
def _contrast_fields(session, image):
    fields = [C.number("Saturated pixels:", 0.35, 2, "%", key="saturated")]
    if image is None:
        return fields + [C.checkbox("Normalize"), C.checkbox("Equalize histogram")]
    composite = image.is_composite
    rgb = image.is_rgb
    if not rgb and not composite:
        fields.append(C.checkbox("Normalize", False))
    roi = image.roi
    if roi is not None and roi.is_area() and not composite:
        fields.append(C.checkbox("Update entire image" if rgb else
                                 "Update all when normalizing", False, key="update"))
    fields.append(C.checkbox("Equalize histogram", False))
    stack_size = 1 if composite else image.stack_size
    if stack_size > 1:
        fields.append(C.checkbox(f"Process_all {stack_size} slices", False, key="process_all"))
        fields.append(C.checkbox("Use stack histogram", False, key="use"))
    fields.append(C.checkbox("Classic equalization", False, key="classic"))
    return fields


def enhance_contrast(session, image: ImageDoc, saturated=0.35, normalize=False,
                     update=False, equalize=False, process_all=False, use=False,
                     classic=False) -> None:
    """Process > Enhance Contrast (``ContrastEnhancer.run``)."""
    composite = image.is_composite
    rgb = image.is_rgb
    stack_size = 1 if composite else image.stack_size
    sat = min(max(float(saturated), 0.0), 100.0)
    if rgb or composite:
        normalize = False
    roi = image.roi
    area_roi = roi is not None and roi.is_area() and not composite
    mask = roi.get_mask((image.height, image.width)) if area_roi else None
    update_selection_only = False
    if area_roi:
        update_selection_only = not update
        if not normalize and not rgb:
            update_selection_only = False
    process_stack = bool(process_all) if stack_size > 1 else False
    use_stack = bool(use) if stack_size > 1 else False
    if process_stack and not equalize:
        normalize = True
    if equalize:
        _equalize(session, image, process_stack, use_stack, bool(classic))
    else:
        _stretch(session, image, sat, bool(normalize), mask, update_selection_only,
                 process_stack, use_stack)
    if normalize:
        image.set_display_range(0.0, 1.0 if image.type == GRAY32 else
                                (65535.0 if image.type == GRAY16 else 255.0))


def _weights(session):
    if session is not None and session.options.weighted_rgb:
        return (0.299, 0.587, 0.114)
    return (1 / 3, 1 / 3, 1 / 3)


def _stretch(session, image, sat, normalize, mask, selection_only, process_stack, use_stack):
    w = _weights(session)
    stack_stats = None
    if use_stack:
        stack_stats = CE.histogram_stats([image.plane_at(k) for k in range(image.stack_size)],
                                         mask, w)
    planes = list(image.iter_planes(process_stack))
    for i, (c, z, t) in enumerate(planes):
        plane = image.plane(c, z, t)
        stats = stack_stats or CE.histogram_stats(plane, mask, w)
        rng = CE.stretch_range(stats, sat)
        if rng is None:
            continue
        vmin, vmax = rng
        if normalize:
            new = CE.normalize(plane, vmin, vmax)
        elif image.is_rgb:
            new = CE.rgb_set_min_max(plane, vmin, vmax)
        else:
            image.set_display_range(vmin, vmax)
            continue
        if selection_only and mask is not None:
            plane[mask] = new[mask]
        else:
            plane[...] = new
        if session is not None and len(planes) > 1:
            session.progress((i + 1) / len(planes))


def _equalize(session, image, process_stack, use_stack, classic):
    if image.type == GRAY32:
        raise CommandError("Equalization of 32-bit images not supported.")
    w = _weights(session)
    histogram = None
    if use_stack:
        histogram = CE.histogram_stats([image.plane_at(k) for k in range(image.stack_size)],
                                       None, w).histogram
    planes = list(image.iter_planes(process_stack))
    for c, z, t in planes:
        plane = image.plane(c, z, t)
        if histogram is None:
            histogram = CE.histogram_stats(plane, None, w).histogram
        plane[...] = CE.equalize(plane, histogram, classic)
    if image.type == GRAY16 and process_stack and image.stack_size > 1:
        mid = image.plane_at(image.stack_size // 2 - 1)
        image.set_display_range(float(mid.min()), float(mid.max()))
    else:
        image.reset_display_range(image.position[0])


# ---------------------------------------------------------------------------------- register
def register(registry) -> None:
    for label, (op, fields) in _VALUE_OPS.items():
        registry.add(label, _value_command(op), fields(), supports_stack=True,
                     types=ALL_TYPES)
    for label, op in _PLAIN_OPS.items():
        registry.add(label, _plain_command(op), [], supports_stack=True, types=ALL_TYPES)
    registry.add("Macro...", math_macro, [C.string("Code:", DEFAULT_MACRO, columns=42),
                                          C.message("v=pixel value, x,y&z=pixel coordinates, "
                                                    "w=image width,\nh=image height, a=angle, "
                                                    "d=distance from center")],
                 supports_stack=True, types=ALL_TYPES)
    registry.add("Image Calculator...", image_calculator, _calc_fields, undoable=False,
                 types=ALL_TYPES)
    registry.add("Enhance Contrast...", enhance_contrast, _contrast_fields, types=ALL_TYPES)


def selftest() -> None:
    from rsstatchem.core.imaging.session import Session

    s = Session()
    register(s.registry)
    a = ImageDoc.from_array(np.arange(64, dtype=np.uint8).reshape(8, 8), "YX", "a")
    s.add_image(a)
    s.run("Add...", "value=25")
    assert a.raw_value(0, 0) == 25 and a.raw_value(7, 7) == 88
    s.run("Macro...", "code=v=v*2")
    assert a.raw_value(7, 7) == 176
    f = ImageDoc.from_array(np.full((4, 4), 2.0, np.float32), "YX", "f")
    s.add_image(f)
    s.run("Reciprocal")
    assert f.raw_value(0, 0) == 0.5
    b = ImageDoc.from_array(np.full((8, 8), 10, np.uint8), "YX", "b")
    s.add_image(b)
    res = s.run("Image Calculator...", "image1=a operation=Subtract image2=b create")
    assert isinstance(res, ImageDoc) and res.raw_value(7, 7) == 166
    s.set_active(a)
    s.run("Enhance Contrast...", "saturated=0.35 normalize")
    assert a.plane().max() == 255

"""Thresholding commands: Image > Adjust > Threshold... / Color Threshold..., Process > Binary >
Make Binary / Convert to Mask, and the Fiji-style "Auto Threshold". **WP-B.**

Headless: the command functions only use :mod:`rsstatchem.core.imaging.threshold`; the windows
(``frames/threshold_adjuster.py``, ``frames/color_threshold.py``) are imported lazily by the
``ui_action`` callables. Macro forms::

    run("Threshold...", "method=Otsu dark");            // setAutoThreshold("Otsu dark")
    run("Threshold...", "lower=50 upper=200");          // setThreshold(50, 200)
    run("Convert to Mask", "method=Otsu background=Dark calculate black");   // stacks
    run("Color Threshold...", "space=HSB min3=128 max3=255 select");
    run("Auto Threshold", "method=[Try all]");          // Fiji behaviour (montage)

"Auto Threshold" re-implements the *behaviour* of Fiji's Auto_Threshold plugin (GPL) on top of
the public-domain AutoThresholder port; no Fiji code is used.
"""
from __future__ import annotations

import math

import numpy as np

from rsstatchem.core.imaging import commands as C
from rsstatchem.core.imaging import threshold as T
from rsstatchem.core.imaging.commands import CommandError
from rsstatchem.core.imaging.image import (COLOR_256, COLOR_RGB, GRAY8, GRAY16, GRAY32,
                                           ImageDoc, Threshold)
from rsstatchem.ui.imagej.dialogs.process_common import roi_mask

GRAY = (GRAY8, GRAY16, GRAY32)
ALL = (GRAY8, GRAY16, GRAY32, COLOR_256, COLOR_RGB)


# ---------------------------------------------------------------------------- Threshold...
def threshold_fields(session, image):
    method = session.options.threshold_method if session is not None else "Default"
    if method not in T.METHODS:
        method = "Default"
    return [
        C.choice("Method", T.METHODS, method, key="method"),
        C.choice("Display", T.DISPLAY_MODES, "Red", key="display"),
        C.checkbox("Dark background", False, key="dark"),
        C.checkbox("Stack histogram", False, key="stack"),
        C.checkbox("Don't reset range", False, key="no-reset"),
        C.checkbox("16-bit histogram", False, key="16-bit"),
        C.number("Lower threshold level", float("nan"), 4, key="lower"),
        C.number("Upper threshold level", float("nan"), 4, key="upper"),
        C.checkbox("Reset", False, key="reset"),
    ]


def threshold_command(session, image, method="Default", display="Red", dark=False,
                      stack=False, reset=False, lower=float("nan"), upper=float("nan"),
                      **kw):
    """Headless Threshold...: setAutoThreshold (method/dark/...), setThreshold (lower/upper)
    or resetThreshold (reset)."""
    mode = T.MODE_KEYS.get(display, "red")
    if reset:
        image.threshold = None
        return None
    if lower == lower and upper == upper:          # explicit levels (not NaN)
        lo, hi = float(lower), float(min(max(upper, lower), upper))
        if hi < lo:
            hi = lo
        lo, hi = T.java_threshold_levels(lo, hi, image.bit_depth)
        image.threshold = Threshold(lo, hi, mode)
        return None
    if method not in T.METHODS:
        raise CommandError(f'Invalid method ("{method}")')
    T.set_auto_threshold(image, method, bool(dark), stack_histogram=bool(stack),
                         no_reset=bool(kw.get("no-reset")), sixteen_bit=bool(kw.get("16-bit")),
                         mode=mode)
    if session is not None:
        session.options.threshold_method = method
    return None


def open_threshold_window(bar, command):
    from rsstatchem.ui.imagej.frames.threshold_adjuster import ThresholdAdjuster
    return ThresholdAdjuster.show_for(bar)


# ---------------------------------------------------------------------------- Convert to Mask
def mask_fields(session, image):
    if image is None or image.stack_size <= 1:
        return []
    bb = session.options.black_background if session is not None else False
    return [
        C.choice("Method:", T.METHODS, "Default", key="method"),
        C.choice("Background:", T.BACKGROUNDS, "Default", key="background"),
        C.checkbox("Calculate threshold for each image", False),
        C.checkbox("Only convert current image", False),
        C.checkbox("Black background (of binary masks)", bb),
        C.checkbox("List thresholds", False),
        C.checkbox("Create new stack", False),
    ]


def convert_to_mask_command(session, image, method="Default", background="Default",
                            calculate=False, only=False, black=None, list=False,
                            create=False, **kw):
    if image.stack_size > 1:
        bb = bool(black)             # the stack dialog's checkbox (macro: absent = off)
    else:
        bb = session.options.black_background if session is not None else False
    log = session.log if session is not None else None
    try:
        out = T.convert_to_mask(image, bb, method, background, bool(calculate), bool(only),
                                bool(list), bool(create), log=log,
                                weighted_rgb=getattr(session.options, "weighted_rgb", False)
                                if session is not None else False)
    except ValueError as exc:
        raise CommandError(str(exc)) from None
    return out if out is not image else None


# ---------------------------------------------------------------------------- Color Threshold
def color_fields(session, image):
    f = [C.choice("Color space", T.COLOR_SPACES, "HSB", key="space")]
    for k in (1, 2, 3):
        f += [C.number(f"Min{k}", 0, 0, minimum=0, maximum=255, key=f"min{k}", integer=True),
              C.number(f"Max{k}", 255, 0, minimum=0, maximum=255, key=f"max{k}", integer=True),
              C.choice(f"Filter{k}", ("pass", "stop"), "pass", key=f"filter{k}")]
    f += [C.checkbox("Auto (thresholding method)", False, key="auto"),
          C.choice("Thresholding method", T.METHODS, "Default", key="method"),
          C.checkbox("Dark background", False, key="dark"),
          C.choice("Threshold color", T.COLOR_MODES, "Red", key="color"),
          C.checkbox("Select", False, key="select"),
          C.checkbox("Filtered", False, key="filtered"),
          C.checkbox("Mask", False, key="mask"),
          C.checkbox("Stack", False, key="stack")]
    return f


def color_settings(values: dict) -> tuple[str, list]:
    space = values.get("space", "HSB")
    ranges = [(int(values.get(f"min{k}", 0)), int(values.get(f"max{k}", 255)),
               values.get(f"filter{k}", "pass") == "pass") for k in (1, 2, 3)]
    return space, ranges


def color_threshold_command(session, image, **values):
    """Headless Color Threshold...: apply the colour-space band filters to the current RGB
    plane and make a selection (default), a "Filtered" image (in place) or a new mask."""
    space, ranges = color_settings(values)
    planes = list(image.iter_planes(bool(values.get("stack")) and values.get("filtered")))
    results = []
    for c, z, t in planes:
        rgb = image.plane(c, z, t)
        chans = T.color_channels(rgb, space)
        if values.get("auto"):
            auto = T.color_auto_ranges(chans, space, values.get("method", "Default"),
                                       bool(values.get("dark")))
            ranges = [(a[0], a[1], r[2]) for a, r in zip(auto, ranges)]
        mask = T.color_threshold(chans, space, ranges)
        bb = session.options.black_background if session is not None else False
        if values.get("filtered"):
            image.set_plane(T.apply_color_mode(rgb, mask, values.get("color", "Red"), bb),
                            c, z, t)
        results.append(mask)
    mask = results[0] if results else None
    if values.get("mask") and mask is not None:
        doc = ImageDoc.from_array(mask.astype(np.uint8) * 255, "YX", f"Mask of {image.title}")
        if not (session.options.black_background if session is not None else False):
            doc.luts = [doc.luts[0].inverted()]
        return doc
    if values.get("select") or not (values.get("filtered") or values.get("mask")):
        image.roi = T.mask_to_roi(mask) if mask is not None else None
    return None


def open_color_window(bar, command):
    from rsstatchem.ui.imagej.frames.color_threshold import ColorThresholdWindow
    return ColorThresholdWindow.show_for(bar)


# ---------------------------------------------------------------------------- Auto Threshold
AUTO_METHODS = T.FIJI_METHODS + ("Try all",)


def auto_fields(session, image):
    return [C.choice("Method", AUTO_METHODS, "Default", key="method"),
            C.checkbox("Ignore_black", False, key="ignore_black"),
            C.checkbox("Ignore_white", False, key="ignore_white"),
            C.checkbox("White objects on black background", False, key="white"),
            C.checkbox("Set threshold instead of Threshold (single images)", False, key="set"),
            C.checkbox("Show threshold values in log window", False, key="show"),
            C.checkbox("Use stack histogram", False, key="use_stack_histogram"),
            C.checkbox("16-bit histogram", True, key="16-bit")]


def _auto_level(plane, method, ignore_black, ignore_white, sixteen, mask=None):
    hist, off, width = T.fiji_histogram(plane, ignore_black, ignore_white, sixteen, mask)
    t = T.fiji_threshold_level(hist, method)
    return off + t * width, hist, off, width


def _binarize(plane, level, white, raw_level_is_bin=False):
    a = np.asarray(plane)
    obj = a > level if white else a <= level
    return obj.astype(np.uint8) * 255


def auto_threshold_command(session, image, method="Default", ignore_black=False,
                           ignore_white=False, white=False, set=False, show=False,
                           use_stack_histogram=False, stack=False, **kw):
    if image.type not in (GRAY8, GRAY16, GRAY32):
        raise CommandError("8-bit or 16-bit image required")
    sixteen = bool(kw.get("16-bit", True))
    mask = roi_mask(image)
    log = session.log if (session is not None and show) else None
    if method == "Try all":
        return _try_all(image, ignore_black, ignore_white, white, sixteen, mask)
    planes = list(image.iter_planes(bool(stack) or bool(use_stack_histogram)))
    if use_stack_histogram and image.stack_size > 1:
        allv = np.concatenate([image.plane(*p).ravel() for p in planes])
        level, *_ = _auto_level(allv.reshape(1, -1), method, ignore_black, ignore_white,
                                sixteen)
        levels = [level] * len(planes)
    else:
        levels = [_auto_level(image.plane(*p), method, ignore_black, ignore_white, sixteen,
                              mask)[0] for p in planes]
    if log:
        for (c, z, t), lv in zip(planes, levels):
            log(f"{method}: {lv:g}" + (f" (slice {image.stack_index(c, z, t) + 1})"
                                        if image.stack_size > 1 else ""))
    if set and image.stack_size == 1:
        lv = levels[0]
        vmin, vmax = T.float_min_max(image.plane())
        if image.type == GRAY32:
            lo, hi = (lv, vmax) if white else (vmin, lv)
        else:
            lo, hi = (math.floor(lv) + 1, vmax) if white else (vmin, math.floor(lv))
        image.threshold = Threshold(float(lo), float(hi), "red")
        return None
    if image.type == GRAY8:
        for (c, z, t), lv in zip(planes, levels):
            image.set_plane(_binarize(image.plane(c, z, t), lv, white), c, z, t)
        image.luts = [type(image.luts[0]).grays() for _ in image.luts]
    else:
        out = np.zeros(image.pixels.shape, np.uint8)
        done = {p: lv for p, lv in zip(planes, levels)}
        for idx in range(image.stack_size):
            p = image.position_of(idx)
            c, z, t = p
            if p in done:
                out[t, z, c] = _binarize(image.plane(c, z, t), done[p], white)
        image.replace_pixels(out, GRAY8)
    image.threshold = None
    image.display_ranges = [(0.0, 255.0)] * image.n_channels
    return None


def _try_all(image, ignore_black, ignore_white, white, sixteen, mask) -> ImageDoc:
    """Montage of the result of every method (Fiji's "Try all"): panels in method order,
    5 columns, the method name in an overlay label and in the image info."""
    from rsstatchem.core.imaging.roi import Overlay, TextRoi

    plane = image.plane()
    methods = T.FIJI_METHODS
    panels = []
    for m in methods:
        try:
            lv = _auto_level(plane, m, ignore_black, ignore_white, sixteen, mask)[0]
            panels.append(_binarize(plane, lv, white))
        except Exception:  # noqa: BLE001 - a method that fails shows an empty panel
            panels.append(np.zeros(plane.shape, np.uint8))
    cols = 5
    rows = -(-len(panels) // cols)
    h, w = plane.shape
    mont = np.zeros((rows * h, cols * w), np.uint8)
    overlay = Overlay()
    for i, (m, p) in enumerate(zip(methods, panels)):
        r, c = divmod(i, cols)
        mont[r * h:(r + 1) * h, c * w:(c + 1) * w] = p
        overlay.add(TextRoi(c * w + 2, r * h + 2, m, font_size=max(9.0, min(18.0, w / 10)),
                            stroke_color="#ffff00"))
    doc = ImageDoc.from_array(mont, "YX", f"Montage of {image.title}")
    doc.overlay = overlay
    doc.info = "Auto Threshold, Try all: " + ", ".join(methods)
    return doc


# ---------------------------------------------------------------------------- registration
def register(registry) -> None:
    registry.add("Threshold...", threshold_command, threshold_fields, types=GRAY,
                 ui_action=open_threshold_window)
    registry.add("Color Threshold...", color_threshold_command, color_fields,
                 types=(COLOR_RGB,), ui_action=open_color_window)
    registry.add("Make Binary", convert_to_mask_command, mask_fields, types=ALL)
    registry.add("Convert to Mask", convert_to_mask_command, mask_fields, types=ALL)
    registry.add("Auto Threshold", auto_threshold_command, auto_fields, types=GRAY,
                 supports_stack=True)


def selftest() -> None:
    from rsstatchem.core.imaging.session import Session

    s = Session()
    register(s.registry)
    yy, xx = np.mgrid[0:32, 0:32]
    plane = (40 + 150 * (((xx - 16) ** 2 + (yy - 16) ** 2) < 64)).astype(np.uint8)
    doc = s.add_image(ImageDoc.from_array(plane, "YX", "t"))
    s.run("Threshold...", "method=Otsu dark")
    assert doc.threshold is not None and doc.threshold.lower > 40
    s.run("Convert to Mask")
    assert set(np.unique(doc.plane()).tolist()) == {0, 255}
    assert T.auto_threshold([0, 10, 0, 0, 30, 20, 0, 5], "Huang") >= 0
    rgb = np.zeros((8, 8, 3), np.uint8)
    rgb[2:5, 2:5] = (255, 0, 0)
    m = T.color_threshold(rgb, "HSB", [(0, 255, True), (128, 255, True), (0, 255, True)])
    assert int(m.sum()) == 9

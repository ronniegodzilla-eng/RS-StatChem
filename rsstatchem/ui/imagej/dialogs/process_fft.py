"""Process > FFT commands: FFT, Inverse FFT, Redisplay Power Spectrum, FFT Options...,
Bandpass Filter..., Custom Filter..., FD Math..., Swap Quadrants, Make Circular Selection....
**WP-B.** Headless (no Qt); the maths is in :mod:`rsstatchem.core.imaging.fft`.

An "FFT of ..." image keeps ImageJ's "FHT" property as the Python attribute ``doc.fht``
(:class:`~rsstatchem.core.imaging.fft.FHTData`); "Inverse FFT" uses it together with the
black (filter) / white (pass) pixels painted on the power spectrum, like ImageJ.
"""
from __future__ import annotations

import numpy as np

from rsstatchem.core.imaging import commands as C
from rsstatchem.core.imaging import fft as F
from rsstatchem.core.imaging.commands import CommandError
from rsstatchem.core.imaging.image import COLOR_RGB, GRAY8, GRAY16, GRAY32, ImageDoc
from rsstatchem.ui.imagej.dialogs.process_common import ALL_TYPES, new_image_like, roi_rect

OPTIONS_KEY = "fft_options"
_OPTION_FIELDS = (("fft", 'FFT ("FFT of...") window', True), ("raw", "Raw power spectrum", False),
                  ("fast", "Fast Hartley Transform", False),
                  ("complex", "Complex Fourier Transform", False),
                  ("reuse", 'Reuse "FFT of..." window', False), ("do", "Do forward transform",
                                                                 False))


def fft_options(session) -> dict:
    opts = session.options.extra.get(OPTIONS_KEY) if session is not None else None
    base = {key: default for key, _, default in _OPTION_FIELDS}
    if isinstance(opts, dict):
        base.update({k: bool(v) for k, v in opts.items() if k in base})
    return base


def get_fht(image) -> F.FHTData | None:
    fht = getattr(image, "fht", None)
    return fht if isinstance(fht, F.FHTData) else None


def _bit_depth(image) -> int:
    return image.bit_depth


def _input_plane(image) -> np.ndarray:
    plane = image.plane()
    roi = image.roi
    if roi is not None and roi.is_area():
        x, y, w, h = roi_rect(image)
        plane = plane[y:y + h, x:x + w]
    return np.array(plane, copy=True)


def _display_stretch(doc: ImageDoc, saturated: float = 0.35) -> None:
    """``IJ.run(imp, "Enhance Contrast", "saturated=0.35")`` for 32-bit FFT outputs."""
    for c, z, t in doc.iter_planes(True):
        lo, hi = F.float_stretch_range(doc.plane(c, z, t), saturated)
        doc.set_display_range(lo, hi, channel=c)
        break


# ---------------------------------------------------------------------------------- FFT
def forward(session, image) -> list[ImageDoc]:
    """The forward transform with the FFT Options outputs (power spectrum by default)."""
    opts = fft_options(session)
    plane = _input_plane(image)
    bd = _bit_depth(image)
    lut = None if image.is_rgb else image.lut.copy()
    ps, info = F.forward_fft(plane, bd, lut)
    info.calibration = image.calibration.copy()
    name = image.title
    out = []
    if opts["raw"]:
        raw = F.swap_quadrants(F.raw_power_spectrum(info.data))
        doc = new_image_like(image, raw[None, None, None], f"PS of {name}", GRAY32)
        _display_stretch(doc)
        out.append(doc)
    if opts["fast"]:
        doc = new_image_like(image, F.swap_quadrants(info.data)[None, None, None],
                             f"FHT of {name}", GRAY32)
        doc.properties.update({" ": "Fast Hartley Transform",
                               "Original width": str(plane.shape[1]),
                               "Original height": str(plane.shape[0])})
        _display_stretch(doc)
        out.append(doc)
    if opts["complex"]:
        re, im = F.complex_transform(info.data)
        doc = new_image_like(image, np.stack([re, im])[None, :, None], f"Complex of {name}",
                             GRAY32)
        doc.slice_labels = ["Real", "Imaginary"]
        doc.properties.update({" ": "Complex Fourier Transform",
                               "Original width": str(plane.shape[1]),
                               "Original height": str(plane.shape[0])})
        _display_stretch(doc)
        out.append(doc)
    if opts["fft"] or not (opts["raw"] or opts["fast"] or opts["complex"]):
        title = f"FFT of {name}"
        reuse = session.find_image(title) if (opts["reuse"] and session is not None) else None
        if reuse is not None and get_fht(reuse) is not None:
            reuse.replace_pixels(ps[None, None, None])
            reuse.fht = info
            session.image_updated(reuse)
        else:
            doc = new_image_like(image, ps[None, None, None], title, GRAY8)
            doc.fht = info
            doc.info = (f"Fast Hartley Transform\nwidth: {info.original_width}\n"
                        f"height: {info.original_height}\nbitdepth: {info.original_bit_depth}\n")
            out.append(doc)
    return out


def inverse(session, image) -> ImageDoc:
    """Inverse FFT of an "FFT of", "FHT of" or "Complex of" image."""
    title = image.title
    if title.startswith("FHT of") and image.type == GRAY32:
        data = F.fht_inverse_of_fht_image(image.plane())
        data = _unpad(image, data)
        doc = new_image_like(image, data[None, None, None], title[7:], GRAY32)
        doc.reset_display_range()
        return doc
    if image.stack_size == 2 and image.slice_labels[0] == "Real" and image.type == GRAY32:
        re, im = F.complex_inverse(image.plane_at(0), image.plane_at(1))
        re, im = _unpad(image, re), _unpad(image, im)
        doc = new_image_like(image, np.stack([re, im])[None, :, None], title[10:], GRAY32)
        doc.slice_labels = ["Real", "Imaginary"]
        doc.reset_display_range()
        return doc
    info = get_fht(image)
    if info is None:
        raise CommandError("Frequency domain image required")
    image.roi = None
    pixels, bd = F.inverse_fft(info, image.plane(), image.bit_depth, image.stack_size)
    if title.startswith("FFT of "):
        title = title[7:]
    type_ = {8: GRAY8, 16: GRAY16, 24: COLOR_RGB, 32: GRAY32}[bd]
    doc = ImageDoc(pixels[None, None, None], type_, f"Inverse FFT of {title}",
                   (info.calibration or image.calibration).copy())
    if bd != 24 and info.original_lut is not None:
        doc.luts = [info.original_lut.copy()]
    doc.reset_display_range()
    return doc


def _unpad(image, data: np.ndarray) -> np.ndarray:
    try:
        w = int(float(image.properties.get("Original width", 0)))
        h = int(float(image.properties.get("Original height", 0)))
    except ValueError:
        return data
    if w == 0 or h == 0 or (w == data.shape[1] and h == data.shape[0]):
        return data
    if F.is_power_of_2(w) and w == h:
        return data
    return np.ascontiguousarray(data[:h, :w])


def fft_command(session, image):
    if get_fht(image) is not None:
        return inverse(session, image)
    return forward(session, image)


def inverse_command(session, image):
    return inverse(session, image)


def redisplay(session, image):
    info = get_fht(image)
    if info is None:
        raise CommandError("Frequency domain image required")
    image.replace_pixels(F.power_spectrum(info.data)[None, None, None], GRAY8)
    image.fht = info


def options_command(session, image, **values):
    session.options.extra[OPTIONS_KEY] = {k: bool(values.get(k)) for k, _, _ in _OPTION_FIELDS}
    if values.get("do"):
        if image is None:
            raise CommandError("There are no images open")
        return forward(session, image)
    return None


def _options_fields(session, image):
    opts = fft_options(session)
    return [C.message("Display:")] + [C.checkbox(label, opts[key], key=key)
                                      for key, label, _ in _OPTION_FIELDS]


# ---------------------------------------------------------------------------------- bandpass
def _bandpass_fields(session, image):
    fields = [C.number("Filter_large structures down to", 40.0, 0, "pixels"),
              C.number("Filter_small structures up to", 3.0, 0, "pixels"),
              C.choice("Suppress stripes:", F.STRIPES, "None"),
              C.number("Tolerance of direction:", 5.0, 0, "%"),
              C.checkbox("Autoscale after filtering", True),
              C.checkbox("Saturate image when autoscaling", True),
              C.checkbox("Display filter", False)]
    if image is not None and image.stack_size > 1:
        fields.append(C.checkbox("Process entire stack", False))
    return fields


def bandpass(session, image, filter_large=40.0, filter_small=3.0, suppress="None",
             tolerance=5.0, autoscale=True, saturate=True, display=False, process=False):
    if get_fht(image) is not None:
        raise CommandError("Spatial domain image required")
    rect = roi_rect(image)
    bd = image.bit_depth
    filter_doc = None
    planes = list(image.iter_planes(bool(process)))
    for i, (c, z, t) in enumerate(planes):
        out, filt = F.bandpass_filter(image.plane(c, z, t), bd, filter_large, filter_small,
                                      suppress, tolerance, autoscale, saturate, rect,
                                      want_filter=bool(display) and i == 0)
        image.set_plane(out, c, z, t)
        if filt is not None:
            filter_doc = ImageDoc(filt[None, None, None], GRAY32, "Filter")
        if len(planes) > 1:
            session.progress((i + 1) / len(planes))
    image.reset_display_range()
    return filter_doc


# ---------------------------------------------------------------------------------- custom
def _other_titles(session, image):
    return [im.title for im in session.images if im is not image] if session else []


def _custom_fields(session, image):
    titles = _other_titles(session, image) or [""]
    fields = [C.choice("Filter:", titles, titles[0])]
    if image is not None and image.stack_size > 1:
        fields.append(C.checkbox("Process entire stack", False))
    return fields


def custom(session, image, filter="", process=False):
    if get_fht(image) is not None:
        raise CommandError("Spatial domain (non-FFT) image required")
    fdoc = session.find_image(filter) if filter else None
    if fdoc is None:
        others = [im for im in session.images if im is not image]
        if not others:
            raise CommandError("A filter (as an open image) is required.")
        fdoc = others[0]
    if fdoc is image:
        raise CommandError("The filter cannot be the same as the image being filtered.")
    if fdoc.stack_size > 1:
        raise CommandError("The filter cannot be a stack.")
    filt = F.filter_to_8bit_or_float(fdoc.plane(), fdoc.bit_depth, fdoc.display_range)
    bd = image.bit_depth
    planes = list(image.iter_planes(bool(process)))
    for i, (c, z, t) in enumerate(planes):
        image.set_plane(F.custom_filter(image.plane(c, z, t), bd, filt), c, z, t)
        if len(planes) > 1:
            session.progress((i + 1) / len(planes))
    if image.type in (GRAY16, GRAY32):
        lo, hi = F.float_stretch_range(image.plane().astype(np.float32), 0.0)
        image.set_display_range(lo, hi)
    elif image.type == GRAY8:
        p = image.plane()
        image.set_display_range(float(p.min()), float(p.max()))


# ---------------------------------------------------------------------------------- FD math
def _pow2_titles(session):
    return [im.title for im in session.images
            if im.width == im.height and F.is_power_of_2(im.width)] if session else []


def _fdmath_fields(session, image):
    titles = _pow2_titles(session) or [""]
    return [C.choice("Image1: ", titles, titles[0]),
            C.choice("Operation:", F.FD_OPERATIONS, "Correlate"),
            C.choice("Image2: ", titles, titles[0]),
            C.string("Result:", "Result"),
            C.checkbox("Do inverse transform", True)]


def fd_math(session, image, image1="", operation="Correlate", image2="", result="Result",
            do=True):
    im1, im2 = session.find_image(image1), session.find_image(image2)
    if im1 is None or im2 is None:
        raise CommandError("Images must be a power of 2 size (256x256, 512x512, etc.)")
    for im in (im1, im2):
        if im.width != im.height or not F.is_power_of_2(im.width):
            raise CommandError("Images must be a power of 2 size (256x256, 512x512, etc.)")
    if im1.width != im2.width:
        raise CommandError("Images must be the same size")

    def operand(im):
        info = get_fht(im)
        if info is not None:
            return info
        p = im.plane()
        return F.get_brightness(p) if im.is_rgb else p.astype(np.float32)
    a = operand(im1)
    b = a if im2 is im1 else operand(im2)
    cal1, cal2 = im1.calibration, im2.calibration
    cal = cal1 if cal1.scaled() else cal2
    out = F.fd_math(a, b, operation, bool(do))
    if do:
        doc = ImageDoc(out[None, None, None], GRAY32, result or "Result", cal.copy())
        doc.reset_display_range()
        return doc
    ps, info = out
    doc = ImageDoc(ps[None, None, None], GRAY32, result or "Result", cal.copy())
    doc.fht = info
    doc.reset_display_range()
    return doc


# ---------------------------------------------------------------------------------- misc
def swap(session, image):
    if image.width != image.height:
        raise CommandError("Image must be square")
    for c, z, t in image.iter_planes(True):
        image.set_plane(F.swap_quadrants(image.plane(c, z, t)), c, z, t)


def _circle_fields(session, image):
    w = image.width if image is not None else 256
    h = image.height if image is not None else 256
    radius = min(w // 4, w // 2, h // 2)
    return [C.slider("Radius:", 0, w // 2, float(radius))]


def circular_selection(session, image, radius=None):
    from rsstatchem.core.imaging.roi import OvalRoi

    w, h = image.width, image.height
    if radius is None:
        radius = min(w // 4, w // 2, h // 2)
    xc, yc = w // 2, h // 2
    image.roi = OvalRoi(xc - radius, yc - radius, radius * 2.0, radius * 2.0)
    if session is not None:
        if get_fht(image) is not None:
            s = "Infinity/c" if radius < 1.0 else f"{w / radius:.2f} p/c"
        else:
            s = f"{radius * image.calibration.pixel_width:.0f} {image.calibration.units}"
        session.status(" radius = " + s)


def register(registry) -> None:
    registry.add("FFT", fft_command, [], types=ALL_TYPES, undoable=False, roi="optional")
    registry.add("Inverse FFT", inverse_command, [], undoable=False)
    registry.add("Redisplay Power Spectrum", redisplay, [])
    registry.add("FFT Options...", options_command, _options_fields, needs_image=False,
                 undoable=False)
    registry.add("Bandpass Filter...", bandpass, _bandpass_fields, types=ALL_TYPES)
    registry.add("Custom Filter...", custom, _custom_fields, types=ALL_TYPES)
    registry.add("FD Math...", fd_math, _fdmath_fields, needs_image=False, undoable=False)
    registry.add("Swap Quadrants", swap, [])
    registry.add("Make Circular Selection...", circular_selection, _circle_fields,
                 undoable=False)


def selftest() -> None:
    from rsstatchem.core.imaging.session import Session

    s = Session()
    register(s.registry)
    yy, xx = np.mgrid[0:24, 0:30]
    img = ImageDoc.from_array(((xx * 7 + yy * 3) % 50 + 20).astype(np.uint8), "YX", "s")
    s.add_image(img)
    spec = s.run("FFT")[0]
    assert spec.width == 32 and get_fht(spec) is not None
    back = s.run("Inverse FFT", image=spec)
    assert np.array_equal(back.plane(), img.plane())
    s.set_active(img)
    s.run("Bandpass Filter...", "filter_large=10 filter_small=2 suppress=None tolerance=5 "
                                "autoscale saturate")
    assert img.plane().dtype == np.uint8

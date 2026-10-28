"""Process > Binary commands (Erode ... Voronoi, Options...) and Process > Find Maxima....
**WP-B.** Headless (no Qt); registered by ``commands_process.register``.

The core algorithms are ports of ImageJ 1.54p (``core/imaging/binary.py``, ``maxima.py``).
This module reproduces ImageJ's plug-in behaviour around them: the "8-bit binary image (0 and
255) required" check, foreground/background from "Black background" and the inverting LUT,
"Process all N slices", the area-selection mask (Binary supports masking; EDM commands and
Skeletonize ignore selections), the invisible binary threshold set afterwards
(``setBinaryThreshold``), new output windows ("EDM of ...", "<title> Maxima") with an inverting
LUT unless black background is on, and the Results output of Find Maxima.
"""
from __future__ import annotations

import numpy as np

from rsstatchem.core.imaging import binary as B
from rsstatchem.core.imaging import commands as C
from rsstatchem.core.imaging import maxima as M
from rsstatchem.core.imaging.commands import CommandError
from rsstatchem.core.imaging.image import (COLOR_256, COLOR_RGB, GRAY8, GRAY16, GRAY32,
                                           ImageDoc, Threshold)
from rsstatchem.core.imaging.roi import PointRoi
from rsstatchem.ui.imagej.dialogs.process_common import apply_filter, roi_mask, roi_rect

BINARY_TYPES = (GRAY8, COLOR_256)
_STATE = {"median_radius": 3.0, "maxima": {
    "prominence": 10.0, "strict": False, "exclude": False, "above": False, "light": False,
    "output": "Single Points"}}


# ---------------------------------------------------------------------------------- helpers
def _check_binary(image: ImageDoc) -> None:
    if image.type not in BINARY_TYPES or not B.is_binary(image.plane()):
        raise CommandError("8-bit binary (0 and 255 only) image required.")


def short_title(title: str) -> str:
    """ImagePlus.getShortTitle: up to the first space, without the file extension."""
    t = title.strip().split(" ")[0] if title.strip() else title
    dot = t.rfind(".")
    if dot > 0 and len(t) - dot <= 6:
        t = t[:dot]
    return t


def _set_binary_threshold(session, image: ImageDoc) -> None:
    """ImageProcessor.setBinaryThreshold: invisible threshold on the objects."""
    t = B.binary_threshold(image.is_inverted_lut(), session.options.black_background)
    image.threshold = Threshold(t, t, "none")


def _fg_bg(session, image: ImageDoc) -> tuple[int, int]:
    fg = B.foreground_value(image.is_inverted_lut(), session.options.black_background)
    return fg, 255 - fg


def _binary_op(session, image, op: str, stack: bool = False) -> None:
    """Binary.run for ``op`` in erode|dilate|open|close|outline|fill|skel|median."""
    _check_binary(image)
    o = session.options
    fg, bg = _fg_bg(session, image)
    it, cnt, pad = int(o.binary_iterations), int(o.binary_count), bool(o.pad_edges)
    inv, bb = image.is_inverted_lut(), bool(o.black_background)
    fns = {
        "erode": lambda p: B.erode(p, cnt, bg, it, pad),
        "dilate": lambda p: B.dilate(p, cnt, bg, it),
        "open": lambda p: B.binary_open(p, cnt, bg, it, pad),
        "close": lambda p: B.binary_close(p, cnt, bg, it, pad),
        "outline": lambda p: B.outline(p, inv, bb),
        "fill": lambda p: B.fill_holes(p, fg, bg),
        "skel": lambda p: B.skeletonize(p, fg),
        "median": lambda p: B.binary_median(p, _STATE["median_radius"]),
    }
    apply_filter(image, fns[op], stack=stack, use_roi=op != "skel",
                 progress=getattr(session, "progress", None))
    _set_binary_threshold(session, image)


def _command(op):
    def run(session, image, stack=False, **_):
        _binary_op(session, image, op, stack)
    run.__name__ = f"binary_{op}"
    return run


def binary_median(session, image, radius=3.0, stack=False):
    _STATE["median_radius"] = float(radius)
    _binary_op(session, image, "median", stack)


# ---------------------------------------------------------------------------------- EDM family
_EDM_PREFIX = {"edm": "EDM of ", "uep": "UEPs of ", "voronoi": "Voronoi of "}


def _edm_command(kind: str):
    def run(session, image, stack=False, **_):
        return edm_command(session, image, kind, stack)
    run.__name__ = f"edm_{kind}"
    return run


def edm_command(session, image: ImageDoc, kind: str, stack: bool = False):
    """EDM.run for Distance Map ("edm"), Ultimate Points ("uep"), Voronoi ("voronoi") and
    Watershed ("watershed"); returns the new output image for non-overwrite EDM output."""
    _check_binary(image)
    o = session.options
    inv, bb = image.is_inverted_lut(), bool(o.black_background)
    output = "Overwrite" if kind == "watershed" else o.edm_output
    if output not in B.OUTPUT_TYPES:
        output = "Overwrite"
    planes = list(image.iter_planes(stack))
    results = []
    for i, (c, z, t) in enumerate(planes):
        plane = image.plane(c, z, t)
        if kind == "watershed":
            image.set_plane(B.watershed(plane, inv, bb), c, z, t)
        else:
            if kind == "edm":
                bg255 = B.edm_background_is_255(inv, bb)
                fedm = B.float_edm(plane, 255 if bg255 else 0)
            elif kind == "uep":
                fedm = B.ultimate_points_edm(plane, inv, bb)
            else:
                fedm = B.voronoi_edm(plane, inv, bb)
            if output == "Overwrite":
                image.set_plane(B.convert_edm(fedm, "Overwrite"), c, z, t)
                if (c, z, t) == image.position:
                    lo, hi = float(fedm.min()), float(fedm.max())
                    if hi > 255.0:
                        image.reset_display_range()
                    else:
                        image.set_display_range(lo, hi, channel=c)
            else:
                results.append(B.convert_edm(fedm, output))
        if len(planes) > 1 and hasattr(session, "progress"):
            session.progress((i + 1) / len(planes))
    if kind == "watershed":
        _set_binary_threshold(session, image)
        return None
    if output == "Overwrite":
        return None
    title = _EDM_PREFIX[kind] + short_title(image.title)
    if len(results) == 1:
        arr = results[0][None, None, None]
    else:
        nc, nz, nt = image.dims
        arr = np.stack(results).reshape((nt, nz, nc) + results[0].shape)
    doc = ImageDoc(arr, None, title, image.calibration.copy())
    if len(results) > 1:
        doc.slice_labels = list(image.slice_labels)
    if not bb:
        doc.luts = [lut.inverted() for lut in doc.luts]
    doc.reset_display_range()
    return doc


# ---------------------------------------------------------------------------------- options
def _options_fields(session, image):
    o = session.options if session is not None else None
    it = o.binary_iterations if o else 1
    cnt = o.binary_count if o else 1
    fields = [
        C.number("Iterations (1-100):", it, 0, integer=True, minimum=1, maximum=100),
        C.number("Count (1-8):", cnt, 0, integer=True, minimum=1, maximum=8),
        C.checkbox("Black background", o.black_background if o else False),
        C.checkbox("Pad edges when eroding", o.pad_edges if o else False),
        C.choice("EDM output:", B.OUTPUT_TYPES, o.edm_output if o else "Overwrite"),
    ]
    if image is not None and image.type in BINARY_TYPES and B.is_binary(image.plane()):
        fields.append(C.choice("Do:", B.OPERATIONS, "Nothing"))   # Binary.operation
        if image.stack_size > 1:
            fields.append(C.checkbox(f"Process all {image.stack_size} slices", key="stack"))
    return fields


_DO_OPS = {"Erode": "erode", "Dilate": "dilate", "Open": "open", "Close": "close",
           "Outline": "outline", "Fill Holes": "fill", "Skeletonize": "skel"}


def binary_options(session, image, iterations=1, count=1, black=False, pad=False,
                   edm="Overwrite", do="Nothing", stack=False):
    """Process > Binary > Options...: stores the options; "Do" runs an operation."""
    o = session.options
    o.binary_iterations = int(min(max(int(iterations), 1), 100))
    o.binary_count = int(min(max(int(count), 1), 8))
    o.black_background = bool(black)
    o.pad_edges = bool(pad)
    o.edm_output = edm if edm in B.OUTPUT_TYPES else "Overwrite"
    if do != "Nothing" and image is not None and do in _DO_OPS:
        image.snapshot("Options...")
        _binary_op(session, image, _DO_OPS[do], stack)
        image.changed = True
        session.image_updated(image)


# ---------------------------------------------------------------------------------- maxima
def _thresholded(image: ImageDoc) -> bool:
    th = image.threshold
    if th is None:
        return False
    return not (th.mode == "none" and th.lower == th.upper)   # resetBinaryThreshold


def _maxima_fields(session, image):
    s = _STATE["maxima"]
    digits = 2 if image is not None and (image.type == GRAY32 or
                                         image.calibration.calibrated()) else 0
    fields = [C.number("Prominence >", s["prominence"], digits),
              C.checkbox("Strict", s["strict"]),
              C.checkbox("Exclude edge maxima", s["exclude"])]
    if image is not None and _thresholded(image):
        fields.append(C.checkbox("Above lower threshold", s["above"]))
    fields += [C.checkbox("Light background", s["light"]),
               C.choice("Output type:", M.OUTPUT_TYPE_NAMES, s["output"])]
    return fields


def _ctable(image: ImageDoc):
    cal = image.calibration
    if image.type in (GRAY8, GRAY16, COLOR_256) and cal.calibrated():
        try:
            return np.asarray(cal.ctable(image.bit_depth), dtype=np.float32)
        except Exception:  # noqa: BLE001 - no table for this function
            return None
    return None


def find_maxima_command(session, image: ImageDoc, prominence=10.0, strict=False,
                        exclude=False, above=False, light=False, output="Single Points"):
    """MaximumFinder.run on the current plane."""
    _STATE["maxima"].update(prominence=float(prominence), strict=bool(strict),
                            exclude=bool(exclude), above=bool(above), light=bool(light),
                            output=output)
    tolerance = max(float(prominence), 0.0)
    out_type = M.OUTPUT_TYPE_NAMES.index(output)
    roi = image.roi
    if roi is not None and (not roi.is_area() or out_type == M.SEGMENTED):
        image.roi = None
        roi = None
    inverted = image.is_inverted_lut()
    use_min = bool(above) and _thresholded(image)
    minima = (inverted and not light) or (not inverted and light)
    if use_min and minima:
        raise CommandError('"Above Lower Threshold" option cannot be used\nwhen finding '
                           'minima (image with light background\nor image with dark '
                           'background and inverting LUT).')
    threshold = float(image.threshold.lower) if use_min else None
    ctable = _ctable(image)
    if threshold is not None and ctable is not None and 0 < threshold < len(ctable):
        threshold = float(ctable[int(threshold)])
    values = M.values_for_maxima(image.plane(), minima, image.display_range, ctable)
    rect = roi_rect(image) if roi is not None else None
    mask = roi_mask(image) if roi is not None else None
    res = M.find_maxima(values, tolerance, bool(strict), threshold, out_type, bool(exclude),
                        is_edm=image.type == GRAY32 and image.title.startswith("EDM of "),
                        roi_rect=rect, roi_mask=mask, float_image=image.type == GRAY32)
    if out_type == M.POINT_SELECTION:
        if res.points:
            xs, ys = zip(*res.points)
            image.roi = PointRoi(np.array(xs, float), np.array(ys, float))
        return None
    if out_type == M.LIST:
        rt = session.results
        rt.reset()
        for x, y in res.points:
            rt.increment_counter()
            rt.add_value("X", x)
            rt.add_value("Y", y)
        return rt
    if out_type == M.COUNT:
        rt = session.results
        row = rt.increment_counter()
        rt.set_value("Count", row, len(res.points))
        if session.options.measure("display"):
            label = image.title
            if roi is not None and getattr(roi, "name", ""):
                label += ":" + roi.name
            if image.is_stack:
                sl = image.slice_label()
                label += ":" + (sl.splitlines()[0] if sl else str(image.current_index + 1))
            rt.set_label(label, row)
        return rt
    name = image.title
    if image.n_slices > 1:
        name += f"({image.position[1] + 1})"
    name += " Segmented" if out_type == M.SEGMENTED else " Maxima"
    cal = image.calibration.copy()
    cal.disable_density()
    doc = ImageDoc(res.image[None, None, None], GRAY8, name, cal)
    if not session.options.black_background:
        doc.luts = [doc.luts[0].inverted()]
    return doc


# ---------------------------------------------------------------------------------- register
def register(registry) -> None:
    stackable = dict(supports_stack=True, types=BINARY_TYPES)
    for label, op in (("Erode", "erode"), ("Dilate", "dilate"), ("Open", "open"),
                      ("Close-", "close"), ("Outline", "outline"), ("Fill Holes", "fill"),
                      ("Skeletonize", "skel")):
        registry.add(label, _command(op), [], **stackable)
    registry.add("Median", binary_median,
                 lambda s, im: [C.number("Radius:", _STATE["median_radius"], 1)], **stackable)
    for label, kind in (("Distance Map", "edm"), ("Ultimate Points", "uep"),
                        ("Watershed", "watershed"), ("Voronoi", "voronoi")):
        registry.add(label, _edm_command(kind), [], roi="none", **stackable)
    registry.add("Options...", binary_options, _options_fields, needs_image=False)
    registry.add("Find Maxima...", find_maxima_command, _maxima_fields, undoable=False,
                 types=(GRAY8, GRAY16, GRAY32, COLOR_256, COLOR_RGB))


def selftest() -> None:
    from rsstatchem.core.imaging.session import Session

    a = np.zeros((24, 32), np.uint8)
    a[4:12, 4:12] = 255
    a[6:18, 14:26] = 255
    a[8:10, 8:10] = 0
    s = Session()
    s.options.black_background = True
    register(s.registry)
    total = int(a.astype(int).sum())
    doc = s.add_image(ImageDoc.from_array(a.copy(), "YX", "selftest.tif"))
    s.run("Fill Holes")
    assert doc.plane()[9, 9] == 255
    s.run("Erode")
    assert int(doc.plane().astype(int).sum()) < total
    s.run("Options...", "iterations=1 count=1 black edm=32-bit")
    edm = s.run("Distance Map")
    assert edm.type == GRAY32 and edm.title == "EDM of selftest"
    s.set_active(doc)
    s.run("Find Maxima...", "prominence=0.5 output=List")
    assert len(s.results) >= 1
    s.run("Options...", "iterations=1 count=1 edm=Overwrite")

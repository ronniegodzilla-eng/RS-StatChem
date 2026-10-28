"""Fiji-style extras: Auto Local Threshold, Enhance Local Contrast (CLAHE) and a
MorphoLibJ-like toolbox. **WP-B.** Headless (no Qt); registered by
``commands_process.register``.

The algorithms are clean-room rewrites from the papers (``core/imaging/fiji/*``; no Fiji or
MorphoLibJ code). The labels are not in ImageJ 1.54's menu tree (``menu_layout``), so the
commands are reached through Find Commands and macros, with Fiji's macro keys, e.g.
``run("Auto Local Threshold", "method=Bernsen radius=15 parameter_1=0 parameter_2=0 white")``.

Binary outputs follow ImageJ's mask convention: 255 = selected/object, shown with an
inverting LUT unless Process > Binary > Options "Black background" is on.
"""
from __future__ import annotations

import numpy as np

from rsstatchem.core.imaging import commands as C
from rsstatchem.core.imaging.commands import CommandError
from rsstatchem.core.imaging.fiji import clahe as CL
from rsstatchem.core.imaging.fiji import local_threshold as LT
from rsstatchem.core.imaging.fiji import morpho as MO
from rsstatchem.core.imaging.image import COLOR_RGB, GRAY8, GRAY16, GRAY32, ImageDoc
from rsstatchem.core.imaging.lut import LUT
from rsstatchem.ui.imagej.dialogs.process_common import apply_filter

GRAY = (GRAY8, GRAY16, GRAY32)
CONNECTIVITY = ("4", "8")
NONE = "*None*"


# ---------------------------------------------------------------------------------- helpers
def short_title(title: str) -> str:
    """ImageJ's short title: up to the first space, without a file extension."""
    t = title.strip().split(" ")[0] if title.strip() else title
    dot = t.rfind(".")
    if dot > 0 and len(t) - dot <= 6:
        t = t[:dot]
    return t


def _titles(session) -> list[str]:
    return [im.title for im in session.images] if session is not None else []


def _image_named(session, title: str, what: str) -> ImageDoc:
    doc = session.find_image(title) if session is not None else None
    if doc is None:
        raise CommandError(f'{what} image "{title}" not found')
    return doc


def _mask_lut(session) -> LUT:
    bb = bool(session.options.black_background) if session is not None else False
    return LUT.grays() if bb else LUT.grays().inverted()


def _derived(image: ImageDoc, planes: list[np.ndarray], stack: bool, title: str) -> ImageDoc:
    """New image from processed planes: the whole stack (``stack``) or the current plane."""
    first = planes[0]
    if stack and image.stack_size > 1:
        t, z, c = image.n_frames, image.n_slices, image.n_channels
        shape = (t, z, c) + first.shape
        pix = np.empty(shape, first.dtype)
        for index, p in enumerate(planes):
            cc, zz, tt = image.position_of(index)
            pix[tt, zz, cc] = p
        doc = ImageDoc(pix, COLOR_RGB if first.ndim == 3 else None, title,
                       image.calibration.copy())
        doc.slice_labels = list(image.slice_labels)
    else:
        pix = first.reshape((1, 1, 1) + first.shape)
        doc = ImageDoc(pix, COLOR_RGB if first.ndim == 3 else None, title,
                       image.calibration.copy())
    return doc


def _planes(image: ImageDoc, stack: bool) -> list[np.ndarray]:
    if stack and image.stack_size > 1:
        return [image.plane_at(i) for i in range(image.stack_size)]
    return [image.plane()]


def _binary_doc(session, image, planes, stack, title) -> ImageDoc:
    doc = _derived(image, planes, stack, title)
    doc.luts = [_mask_lut(session) for _ in range(doc.n_channels)]
    doc.display_ranges = [(0.0, 255.0)] * doc.n_channels
    return doc


def _label_doc(image, planes, stack, title) -> ImageDoc:
    doc = _derived(image, planes, stack, title)
    if doc.type != GRAY8:
        doc.reset_display_range()
    else:
        hi = max(float(max(p.max() for p in planes)), 1.0)
        doc.display_ranges = [(0.0, hi)] * doc.n_channels
    return doc


def _connectivity(value) -> int:
    return 8 if str(value).strip().startswith("8") else 4


def _gray_planes(image: ImageDoc) -> None:
    if image.type == COLOR_RGB:
        raise CommandError("Grayscale image required (8, 16 or 32-bit)")


# ---------------------------------------------------------------------------------- local threshold
def _alt_fields(session, image):
    return [C.choice("Method", ("Try all",) + LT.METHODS, "Bernsen"),
            C.number("Radius", 15, 0), C.number("Parameter_1", 0, 2),
            C.number("Parameter_2", 0, 2),
            C.checkbox("White objects on black background", True),
            C.checkbox("Stack", False)]


def auto_local_threshold(session, image, method="Bernsen", radius=15.0, parameter_1=0.0,
                         parameter_2=0.0, white=False, stack=False, **_):
    if image.type != GRAY8:
        raise CommandError("Only 8-bit images are supported")
    radius = float(radius)
    if radius < 1:
        raise CommandError("Radius must be >= 1")
    if method == "Try all":
        plane = image.plane()
        results = LT.try_all(plane, radius, parameter_1, parameter_2, white)
        pix = np.stack([m for _, m in results])[None, :, None]
        doc = ImageDoc(pix, GRAY8, f"{short_title(image.title)} Auto Local Threshold (Try all)",
                       image.calibration.copy())
        doc.slice_labels = [name for name, _ in results]
        doc.luts = [_mask_lut(session)]
        return doc
    if method not in LT.METHODS:
        raise CommandError(f'Unknown method "{method}"')
    apply_filter(image, lambda p: LT.to_mask(LT.local_threshold(p, method, radius, parameter_1,
                                                                parameter_2, white)),
                 stack=stack, use_roi=False, progress=getattr(session, "progress", None))
    image.luts = [_mask_lut(session) for _ in range(image.n_channels)]
    image.display_ranges = [(0.0, 255.0)] * image.n_channels
    return None


# ---------------------------------------------------------------------------------- CLAHE
def _clahe_fields(session, image):
    masks = [NONE] + [t for t in _titles(session) if image is None or t != image.title]
    return [C.number("blocksize", 127, 0), C.number("histogram bins", 256, 0),
            C.number("maximum slope", 3.0, 2), C.choice("mask", masks, NONE),
            C.checkbox("fast_(less_accurate)", False)]


def enhance_local_contrast(session, image, blocksize=127, histogram=256, maximum=3.0,
                           mask=NONE, stack=False, **values):
    fast = bool(values.get("fast_(less_accurate)", False))
    mplane = None
    if mask and mask != NONE:
        mdoc = _image_named(session, mask, "Mask")
        if (mdoc.width, mdoc.height) != (image.width, image.height):
            raise CommandError("The mask must have the same size as the image")
        mp = mdoc.plane()
        mplane = (mp.any(axis=-1) if mp.ndim == 3 else mp != 0)
    if float(maximum) < 1:
        raise CommandError("The maximum slope must be >= 1")
    apply_filter(image, lambda p: CL.clahe(p, int(blocksize), int(histogram), float(maximum),
                                           mplane, fast),
                 stack=stack, rgb="whole", progress=getattr(session, "progress", None))
    return None


# ---------------------------------------------------------------------------------- morphology
def _morph_fields(session, image):
    return [C.choice("Operation", MO.OPERATIONS, "Erosion"),
            C.choice("Element", MO.SHAPES, "Disk"),
            C.number("Radius (in pixels)", 2, 0, integer=True)]


_SUFFIX = {"Erosion": "Erosion", "Dilation": "Dilation", "Opening": "Opening",
           "Closing": "Closing", "White Top Hat": "WTH", "Black Top Hat": "BTH",
           "Gradient": "Gradient", "Laplacian": "Laplacian", "Internal Gradient": "IntGrad",
           "External Gradient": "ExtGrad"}


def morphological_filters(session, image, operation="Erosion", element="Disk", radius=2,
                          stack=False, **_):
    if operation not in MO.OPERATIONS:
        raise CommandError(f'Unknown operation "{operation}"')
    if element not in MO.SHAPES:
        raise CommandError(f'Unknown structuring element "{element}"')
    r = int(round(float(radius)))

    def fn(p):
        if p.ndim == 3:
            return np.stack([MO.morphological_filter(np.ascontiguousarray(p[..., k]), operation,
                                                     element, r) for k in range(3)], axis=-1)
        return MO.morphological_filter(p, operation, element, r)

    planes = [fn(p) for p in _planes(image, stack)]
    doc = _derived(image, planes, stack, f"{short_title(image.title)}-{_SUFFIX[operation]}")
    if not image.is_rgb:
        doc.luts = [lut.copy() for lut in image.luts]
        if doc.type == GRAY8:
            doc.display_ranges = list(image.display_ranges)
    return doc


def _recon_fields(session, image):
    titles = _titles(session) or [""]
    default = image.title if image is not None else titles[0]
    return [C.choice("Marker Image", titles, titles[0], key="marker"),
            C.choice("Mask Image", titles, default, key="mask"),
            C.choice("Type of Reconstruction", ("By Dilation", "By Erosion"), "By Dilation",
                     key="type"),
            C.choice("Connectivity", CONNECTIVITY, "4")]


def morphological_reconstruction(session, image, marker="", mask="", type="By Dilation",
                                 connectivity="4", **_):
    mk = _image_named(session, marker, "Marker")
    ms = _image_named(session, mask, "Mask")
    if (mk.width, mk.height) != (ms.width, ms.height):
        raise CommandError("Marker and mask images must have the same size")
    if mk.is_rgb or ms.is_rgb:
        raise CommandError("Grayscale images required")
    by = "erosion" if "Erosion" in type else "dilation"
    out = MO.reconstruction(mk.plane(), ms.plane(), by, _connectivity(connectivity))
    doc = ImageDoc.from_array(out, "YX", f"{short_title(ms.title)}-rec")
    doc.calibration = ms.calibration.copy()
    doc.luts = [ms.lut.copy()]
    if doc.type == GRAY8:
        doc.display_ranges = [ms.display_range]
    return doc


def _regional_fields(session, image):
    return [C.choice("Operation", ("Regional Minima", "Regional Maxima"), "Regional Minima"),
            C.choice("Connectivity", CONNECTIVITY, "4")]


def regional_min_max(session, image, operation="Regional Minima", connectivity="4",
                     stack=False, **_):
    _gray_planes(image)
    conn = _connectivity(connectivity)
    fn = MO.regional_maxima if "Max" in operation else MO.regional_minima
    planes = [MO.to_binary(fn(p, conn)) for p in _planes(image, stack)]
    suffix = "rmax" if "Max" in operation else "rmin"
    return _binary_doc(session, image, planes, stack, f"{short_title(image.title)}-{suffix}")


def _extended_fields(session, image):
    return [C.choice("Operation", ("Extended Minima", "Extended Maxima"), "Extended Minima"),
            C.number("Dynamic", 10, 2), C.choice("Connectivity", CONNECTIVITY, "4")]


def extended_min_max(session, image, operation="Extended Minima", dynamic=10.0,
                     connectivity="4", stack=False, **_):
    _gray_planes(image)
    conn = _connectivity(connectivity)
    fn = MO.extended_maxima if "Max" in operation else MO.extended_minima
    planes = [MO.to_binary(fn(p, float(dynamic), conn)) for p in _planes(image, stack)]
    suffix = "emax" if "Max" in operation else "emin"
    return _binary_doc(session, image, planes, stack, f"{short_title(image.title)}-{suffix}")


def fill_holes_gray(session, image, stack=False, **_):
    _gray_planes(image)
    planes = [MO.fill_holes(p) for p in _planes(image, stack)]
    doc = _derived(image, planes, stack, f"{short_title(image.title)}-fillHoles")
    doc.luts = [lut.copy() for lut in image.luts]
    doc.display_ranges = list(image.display_ranges)
    return doc


def kill_borders(session, image, stack=False, **_):
    _gray_planes(image)
    planes = [MO.kill_borders(p) for p in _planes(image, stack)]
    doc = _derived(image, planes, stack, f"{short_title(image.title)}-killBorders")
    doc.luts = [lut.copy() for lut in image.luts]
    doc.display_ranges = list(image.display_ranges)
    return doc


def _distance_fields(session, image):
    return [C.choice("Distances", tuple(MO.CHAMFER_WEIGHTS), "Borgefors (3,4)"),
            C.choice("Output Type", ("16 bits", "32 bits"), "16 bits", key="output"),
            C.checkbox("Normalize weights", True)]


def chamfer_distance_map(session, image, distances="Borgefors (3,4)", output="16 bits",
                         normalize=False, stack=False, **_):
    _gray_planes(image)
    if distances not in MO.CHAMFER_WEIGHTS:
        raise CommandError(f'Unknown distances "{distances}"')
    planes = [MO.convert_distance(MO.chamfer_distance(p, distances, normalize), output)
              for p in _planes(image, stack)]
    doc = _derived(image, planes, stack, f"{short_title(image.title)}-dist")
    doc.reset_display_range()
    return doc


def _mcw_fields(session, image):
    titles = _titles(session) or [""]
    default = image.title if image is not None else titles[0]
    return [C.choice("Input", titles, default),
            C.choice("Marker", titles, titles[-1]),
            C.choice("Mask", ["None"] + titles, "None"),
            C.checkbox("Binary markers", True),
            C.checkbox("Calculate dams", True),
            C.checkbox("Use diagonal connectivity", False)]


def marker_controlled_watershed(session, image, input="", marker="", mask="None",
                                binary=False, calculate=False, use=False, **_):
    src = _image_named(session, input, "Input")
    mk = _image_named(session, marker, "Marker")
    size = (src.width, src.height)
    if (mk.width, mk.height) != size:
        raise CommandError("Input and marker images must have the same size")
    if src.is_rgb:
        raise CommandError("Grayscale input image required")
    conn = 8 if use else 4
    if binary:
        markers, _ = MO.label_components(mk.plane(), conn)
    else:
        markers = np.asarray(mk.plane(), np.float64).astype(np.int64)
    mplane = None
    if mask and mask != "None":
        md = _image_named(session, mask, "Mask")
        if (md.width, md.height) != size:
            raise CommandError("The mask must have the same size as the input")
        mplane = md.plane() != 0
    labels = MO.watershed(src.plane(), markers, mplane, conn, bool(calculate))
    doc = ImageDoc.from_array(MO.labels_as(labels, "float"), "YX",
                              f"{short_title(src.title)}-watershed")
    doc.calibration = src.calibration.copy()
    doc.reset_display_range()
    return doc


def _dtw_fields(session, image):
    return [C.choice("Distances", tuple(MO.CHAMFER_WEIGHTS), "Borgefors (3,4)"),
            C.choice("Output Type", ("16 bits", "32 bits"), "32 bits", key="output"),
            C.checkbox("Normalize weights", True), C.number("Dynamic", 1, 2),
            C.choice("Connectivity", CONNECTIVITY, "4")]


def distance_transform_watershed(session, image, distances="Borgefors (3,4)",
                                 output="32 bits", normalize=False, dynamic=1.0,
                                 connectivity="4", stack=False, **_):
    _gray_planes(image)
    if distances not in MO.CHAMFER_WEIGHTS:
        raise CommandError(f'Unknown distances "{distances}"')
    conn = _connectivity(connectivity)
    kind = "16 bits" if "16" in output else "float"
    try:
        planes = [MO.labels_as(MO.distance_transform_watershed(p, distances, normalize,
                                                               float(dynamic), conn), kind)
                  for p in _planes(image, stack)]
    except ValueError as exc:
        raise CommandError(str(exc)) from None
    return _label_doc(image, planes, stack, f"{short_title(image.title)}-dist-watershed")


def _ccl_fields(session, image):
    return [C.choice("Connectivity", CONNECTIVITY, "4"),
            C.choice("Type", ("8 bits", "16 bits", "float"), "16 bits")]


def connected_components_labeling(session, image, connectivity="4", type="16 bits",
                                  stack=False, **_):
    _gray_planes(image)
    conn = _connectivity(connectivity)
    try:
        planes = [MO.labels_as(MO.label_components(p, conn)[0], type)
                  for p in _planes(image, stack)]
    except ValueError as exc:
        raise CommandError(str(exc)) from None
    return _label_doc(image, planes, stack, f"{short_title(image.title)}-lbl")


# ---------------------------------------------------------------------------------- register
def register(registry) -> None:
    new = dict(undoable=False, roi="none")
    registry.add("Auto Local Threshold", auto_local_threshold, _alt_fields, types=(GRAY8,))
    registry.add("Enhance Local Contrast (CLAHE)", enhance_local_contrast, _clahe_fields,
                 types=(GRAY8, GRAY16, GRAY32, COLOR_RGB), supports_stack=True)
    registry.add("Morphological Filters", morphological_filters, _morph_fields,
                 types=(GRAY8, GRAY16, GRAY32, COLOR_RGB), supports_stack=True, **new)
    registry.add("Morphological Reconstruction", morphological_reconstruction, _recon_fields,
                 **new)
    registry.add("Regional Min & Max", regional_min_max, _regional_fields, types=GRAY,
                 supports_stack=True, **new)
    registry.add("Extended Min & Max", extended_min_max, _extended_fields, types=GRAY,
                 supports_stack=True, **new)
    registry.add("Fill Holes (Binary/Gray)", fill_holes_gray, [], types=GRAY,
                 supports_stack=True, **new)
    registry.add("Kill Borders", kill_borders, [], types=GRAY, supports_stack=True, **new)
    registry.add("Chamfer Distance Map", chamfer_distance_map, _distance_fields, types=GRAY,
                 supports_stack=True, **new)
    registry.add("Marker-controlled Watershed", marker_controlled_watershed, _mcw_fields,
                 **new)
    registry.add("Distance Transform Watershed", distance_transform_watershed, _dtw_fields,
                 types=GRAY, supports_stack=True, **new)
    registry.add("Connected Components Labeling", connected_components_labeling, _ccl_fields,
                 types=GRAY, supports_stack=True, **new)


def selftest() -> None:
    from rsstatchem.core.imaging.session import Session

    yy, xx = np.mgrid[0:40, 0:60]
    a = ((((xx - 18) ** 2 + (yy - 20) ** 2) <= 100) |
         (((xx - 35) ** 2 + (yy - 20) ** 2) <= 81)).astype(np.uint8) * 255
    s = Session()
    register(s.registry)
    doc = s.add_image(ImageDoc.from_array(a.copy(), "YX", "two.tif"))
    lbl = s.run("Connected Components Labeling", "connectivity=4 type=[16 bits]")
    assert lbl.title == "two-lbl" and int(lbl.plane().max()) == 1
    s.set_active(doc)
    ws = s.run("Distance Transform Watershed",
               "distances=[Borgefors (3,4)] output=[32 bits] normalize dynamic=1 connectivity=4")
    assert int(ws.plane().max()) == 2
    s.set_active(doc)
    ero = s.run("Morphological Filters", "operation=Erosion element=Disk radius=2")
    assert int(ero.plane().astype(int).sum()) < int(a.astype(int).sum())
    g = (np.add.outer(np.arange(40), np.arange(60)) % 256).astype(np.uint8)
    gd = s.add_image(ImageDoc.from_array(g, "YX", "ramp.tif"))
    s.run("Auto Local Threshold", "method=Mean radius=5 parameter_1=0 parameter_2=0 white")
    assert set(np.unique(gd.plane()).tolist()) <= {0, 255}
    cd = s.add_image(ImageDoc.from_array(g.copy(), "YX", "ramp2.tif"))
    s.run("Enhance Local Contrast (CLAHE)", "blocksize=15 histogram=256 maximum=3 mask=*None*")
    assert cd.plane().dtype == np.uint8

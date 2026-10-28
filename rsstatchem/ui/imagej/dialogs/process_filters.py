"""Process > Smooth / Sharpen / Find Edges, Process > Noise, Process > Shadows and Process >
Filters (2-D and 3-D). **WP-B.** Headless (no Qt); the algorithms are the exact ports in
:mod:`rsstatchem.core.imaging.filters`.

Macro keys are ImageJ's: ``sigma=2 scaled stack`` (Gaussian Blur), ``radius=2 stack`` (rank
filters), ``radius=2 threshold=50 which=Bright`` (Remove Outliers), ``radius=5 light don't``
(Top Hat), ``radius=1 mask=0.60`` (Unsharp Mask), ``text1=[...] normalize`` (Convolve),
``x=2 y=2 z=2`` (3-D filters), ``standard=25`` (Add Specified Noise). Like ImageJ's
PlugInFilterRunner, 2-D filters read the whole plane and write only inside an area selection;
RGB images are filtered channel by channel (Sharpen/Shadows on RGB keep the edge pixels, as
``ColorProcessor.convolve3x3``). Noise uses ``java.util.Random`` semantics when a seed was set
(``session.options.extra["random_seed"]``, as the macro ``random("seed", n)``), otherwise a
fast numpy generator with the same statistics.
"""
from __future__ import annotations

import numpy as np

from rsstatchem.core.imaging import commands as C
from rsstatchem.core.imaging import filters as F
from rsstatchem.core.imaging.commands import CommandError
from rsstatchem.core.imaging.image import (COLOR_256, COLOR_RGB, GRAY8, GRAY16, GRAY32,
                                           ImageDoc)
from rsstatchem.ui.imagej.dialogs.process_common import (ALL_TYPES, roi_mask, roi_rect,
                                                         write_back)

RANK_TYPES = ALL_TYPES + (COLOR_256,)
DEFAULT_KERNEL = ("-1 -1 -1 -1 -1\n-1 -1 -1 -1 -1\n-1 -1 24 -1 -1\n-1 -1 -1 -1 -1\n"
                  "-1 -1 -1 -1 -1\n")
SHADOW_LABELS = ("North", "Northeast", "East", "Southeast", "South", "Southwest", "West",
                 "Northwest")
RANK_3D = {"Median 3D...": "median", "Mean 3D...": "mean", "Minimum 3D...": "min",
           "Maximum 3D...": "max", "Variance 3D...": "variance"}
RANK_2D = {"Median...": F.MEDIAN, "Mean...": F.MEAN, "Minimum...": F.MIN,
           "Maximum...": F.MAX, "Variance...": F.VARIANCE}
LABELS = ("Smooth", "Sharpen", "Find Edges", "Add Noise", "Add Specified Noise...",
          "Salt and Pepper", "Despeckle", "Remove Outliers...", "Remove NaNs...") + \
    SHADOW_LABELS + ("Shadows Demo", "Convolve...", "Gaussian Blur...") + tuple(RANK_2D) + \
    ("Unsharp Mask...", "Top Hat...", "Gaussian Blur 3D...") + tuple(RANK_3D) + \
    ("Show Circular Masks...",)

_RNG = F.JavaRandom()      # ImageProcessor's static Random


# ---------------------------------------------------------------------------------- plumbing
def _process(session, image: ImageDoc, fn, stack: bool = False, rgb_whole: bool = False):
    """``fn(plane, rect) -> plane`` on the current plane (or every plane), written back
    inside the area selection; RGB per channel unless ``rgb_whole``."""
    mask = roi_mask(image)
    rect = roi_rect(image)
    planes = list(image.iter_planes(bool(stack)))
    for i, (c, z, t) in enumerate(planes):
        plane = image.plane(c, z, t)
        if image.type == COLOR_RGB and not rgb_whole:
            out = np.empty_like(plane)
            for k in range(3):
                out[..., k] = fn(np.ascontiguousarray(plane[..., k]), rect)
        else:
            out = fn(plane.copy(), rect)
        write_back(image, out, c, z, t, mask)
        if session is not None and len(planes) > 1:
            session.progress((i + 1) / len(planes))


def _reset_range_if_not_8bit(image: ImageDoc) -> None:
    if image.type in (GRAY16, GRAY32):
        image.reset_display_range()


def _rect_or_none(image: ImageDoc, rect):
    return None if rect == (0, 0, image.width, image.height) else rect


# ---------------------------------------------------------------------------------- 3x3
def smooth(session, image, stack=False):
    _process(session, image, lambda p, r: F.smooth(p), stack)


def sharpen(session, image, stack=False):
    _process(session, image, lambda p, r: F.sharpen(p), stack, rgb_whole=True)


def find_edges(session, image, stack=False):
    _process(session, image, lambda p, r: F.find_edges(p), stack)


def _shadow_command(direction: str):
    def run(session, image, stack=False):
        _process(session, image, lambda p, r: F.shadow(p, direction), stack, rgb_whole=True)
    run.__name__ = f"shadow_{direction}"
    return run


def shadows_demo(session, image):
    if image.is_stack:
        raise CommandError("Shadows Demo does not work with stacks.")
    plane = image.plane()
    out = [F.shadow(plane, d) for d in F.SHADOW_ORDER]
    axes = "ZYXS" if image.is_rgb else "ZYX"
    doc = ImageDoc.from_array(np.stack(out), axes, f"Shadows of {image.title}")
    doc.calibration = image.calibration.copy()
    doc.luts = [lut.copy() for lut in image.luts[:1]]
    doc.slice_labels = [d.capitalize() for d in F.SHADOW_ORDER]
    return doc


# ---------------------------------------------------------------------------------- noise
def _seeded_rng(session):
    """ImageJ: a seed set by ``random("seed", n)`` is applied to the next noise call only."""
    extra = getattr(getattr(session, "options", None), "extra", None)
    if extra is not None and extra.get("random_seed") is not None:
        _RNG.set_seed(int(float(extra.pop("random_seed"))))
        return _RNG
    return None


def _noise(session, image, sd, stack):
    rng = _seeded_rng(session)
    _process(session, image, lambda p, r: F.add_noise(p, sd, rng, r), stack, rgb_whole=True)
    _reset_range_if_not_8bit(image)


def add_noise(session, image, stack=False):
    _noise(session, image, 25.0, stack)


def add_specified_noise(session, image, standard=25.0, stack=False):
    _noise(session, image, float(standard), stack)


def salt_and_pepper(session, image, stack=False):
    rng = _seeded_rng(session) or _RNG
    _process(session, image, lambda p, r: F.salt_and_pepper(p, 0.05, rng, r), stack)


# ---------------------------------------------------------------------------------- rank
def _rank_command(kind: str):
    def run(session, image, radius=2.0, stack=False):
        radius = float(radius)
        if radius < 0 or radius > (100 if kind == F.MEDIAN else 1000):
            raise CommandError("Radius out of range")
        _process(session, image, lambda p, r: F.rank_filter(p, radius, kind), stack)
        if kind == F.VARIANCE:
            _reset_range_if_not_8bit(image)
    run.__name__ = f"rank_{kind}"
    return run


def despeckle(session, image, stack=False):
    _process(session, image, lambda p, r: F.rank_filter(p, 1.0, F.MEDIAN), stack)


def remove_outliers(session, image, radius=2.0, threshold=50.0, which="Bright", stack=False):
    inv = image.is_inverted_lut()
    _process(session, image, lambda p, r: F.rank_filter(
        p, float(radius), F.OUTLIERS, threshold=float(threshold), which_outliers=which,
        inverted_lut=inv), stack)


def remove_nans(session, image, radius=2.0, stack=False):
    if image.type != GRAY32:
        raise CommandError('"Remove NaNs" requires a 32-bit image')
    _process(session, image, lambda p, r: F.rank_filter(p, float(radius), F.REMOVE_NAN,
                                                        rect=r), stack)


def top_hat(session, image, radius=2.0, light=False, stack=False, **kw):
    dont = bool(kw.get("don't", False))
    inv = image.is_inverted_lut()
    _process(session, image, lambda p, r: F.rank_filter(
        p, float(radius), F.TOP_HAT, light_background=bool(light), dont_subtract=bool(dont),
        inverted_lut=inv), stack)
    if not dont:
        _reset_range_if_not_8bit(image)


# ---------------------------------------------------------------------------------- linear
def gaussian_blur(session, image, sigma=2.0, scaled=False, stack=False):
    sigma = abs(float(sigma))
    cal = image.calibration
    sx = sigma / cal.pixel_width if scaled and cal.scaled() else sigma
    sy = sigma / cal.pixel_height if scaled and cal.scaled() else sigma
    acc = 0.002 if image.type in (GRAY8, COLOR_RGB, COLOR_256) else 0.0002
    _process(session, image, lambda p, r: F.gaussian_blur(
        p, sx, sy, acc, _rect_or_none(image, r)), stack)


def unsharp_mask(session, image, radius=1.0, mask=0.6, stack=False):
    sigma, weight = float(radius), float(mask)
    if sigma < 0 or weight < 0 or weight > 0.99:
        raise CommandError("Radius must be >= 0 and Mask Weight between 0 and 0.99")
    _process(session, image, lambda p, r: F.unsharp_mask(
        p, sigma, weight, _rect_or_none(image, r)), stack)


def convolve(session, image, text1=DEFAULT_KERNEL, normalize=False, stack=False):
    try:
        kernel, _, _ = F.parse_kernel(str(text1))
    except ValueError as exc:
        raise CommandError(str(exc)) from None
    _process(session, image, lambda p, r: F.convolve(p, kernel, bool(normalize)), stack)
    if image.roi is None:
        _reset_range_if_not_8bit(image)


# ---------------------------------------------------------------------------------- 3-D
def _volumes_rank(image: ImageDoc):
    """Filters3D: a single-channel stack is one volume (all planes in stack order); a
    multi-channel image is filtered per channel."""
    nc = image.n_channels
    if nc == 1:
        yield [image.position_of(i) for i in range(image.stack_size)]
    else:
        for c in range(nc):
            yield [(c, z, t) for t in range(image.n_frames) for z in range(image.n_slices)]


def _volumes_z(image: ImageDoc):
    """GaussianBlur3D: hyperstacks are z-blurred per (channel, frame) (over frames when
    there is a single slice); a plain stack is one volume."""
    if image.hyperstack or image.n_channels > 1:
        if image.n_slices == 1:
            for c in range(image.n_channels):
                yield [(c, 0, t) for t in range(image.n_frames)]
        else:
            for c in range(image.n_channels):
                for t in range(image.n_frames):
                    yield [(c, z, t) for z in range(image.n_slices)]
    else:
        yield [image.position_of(i) for i in range(image.stack_size)]


def _rank3d_command(kind: str):
    def run(session, image, x=2.0, y=2.0, z=2.0):
        if image.n_channels > 1 and image.n_channels == image.stack_size:
            raise CommandError("Composite color images not supported")
        vols = list(_volumes_rank(image))
        results = []
        for vol in vols:
            data = np.stack([image.plane(c, zz, t) for c, zz, t in vol])
            res = F.filter_3d(data, kind, float(x), float(y), float(z))
            if image.is_rgb and res.dtype != np.uint8:
                res = np.clip(np.floor(res + 0.5), 0, 255).astype(np.uint8)
            results.append((vol, res))
        if kind == "variance" and not image.is_rgb:
            pixels = np.zeros(image.pixels.shape, np.float32)
            for vol, res in results:
                for (c, zz, t), plane in zip(vol, res):
                    pixels[t, zz, c] = plane
            image.replace_pixels(pixels, GRAY32)
        else:
            for vol, res in results:
                for (c, zz, t), plane in zip(vol, res):
                    image.set_plane(plane, c, zz, t)
            _reset_range_if_not_8bit(image)
    run.__name__ = f"filter3d_{kind}"
    return run


def gaussian_blur_3d(session, image, x=2.0, y=2.0, z=2.0):
    if image.n_channels > 1 and image.n_channels == image.stack_size:
        raise CommandError("Composite color images not supported")
    sx, sy, sz = float(x), float(y), float(z)
    image.roi = None
    for vol in _volumes_z(image):
        data = np.stack([image.plane(c, zz, t) for c, zz, t in vol])
        res = F.gaussian_blur_3d(data, sx, sy, sz)
        for (c, zz, t), plane in zip(vol, res):
            image.set_plane(plane, c, zz, t)


def show_circular_masks(session, image):
    stack, labels = F.circular_masks_image()
    doc = ImageDoc.from_array(stack, "ZYX", "Masks")
    doc.slice_labels = labels
    doc.set_display_range(0.0, 1.0)
    return doc


# ---------------------------------------------------------------------------------- dialogs
def _gaussian_fields(session, image):
    fields = [C.number("Sigma (Radius):", 2.0, 2)]
    if image is not None and image.calibration.scaled():
        fields.append(C.checkbox(f"Scaled Units ({image.calibration.units})", False,
                                 key="scaled"))
    return fields


def _outlier_fields(session, image):
    digits = 2 if image is not None and image.type == GRAY32 else 0
    max_value = 100.0
    if image is not None:
        sd = float(np.nanstd(image.plane().astype(np.float64)))
        if sd > max_value:
            max_value = sd
    return [C.slider("Radius:", 0.5, 25, 2.0, 1),
            C.slider("Threshold:", 0, max_value, 50.0, digits),
            C.choice("Which outliers", ("Bright", "Dark"), "Bright")]


def _radius():
    return [C.number("Radius", 2.0, 1, "pixels")]


def _xyz(word: str):
    return [C.number(f"X {word}:", 2.0, 1), C.number(f"Y {word}:", 2.0, 1),
            C.number(f"Z {word}:", 2.0, 1)]


def register(registry) -> None:
    add = registry.add
    add("Smooth", smooth, supports_stack=True, types=ALL_TYPES)
    add("Sharpen", sharpen, supports_stack=True, types=ALL_TYPES)
    add("Find Edges", find_edges, supports_stack=True, types=ALL_TYPES)
    add("Add Noise", add_noise, supports_stack=True, types=ALL_TYPES)
    add("Add Specified Noise...", add_specified_noise,
        [C.number("Standard Deviation:", 25.0, 2)], supports_stack=True, types=ALL_TYPES)
    add("Salt and Pepper", salt_and_pepper, supports_stack=True, types=(GRAY8, COLOR_256))
    add("Despeckle", despeckle, supports_stack=True, types=RANK_TYPES)
    add("Remove Outliers...", remove_outliers, _outlier_fields, supports_stack=True,
        types=RANK_TYPES)
    add("Remove NaNs...", remove_nans, _radius(), supports_stack=True, types=(GRAY32,))
    for label in SHADOW_LABELS:
        add(label, _shadow_command(label.lower()), supports_stack=True, types=ALL_TYPES)
    add("Shadows Demo", shadows_demo, types=ALL_TYPES, undoable=False)
    add("Convolve...", convolve,
        [C.string("Kernel:", DEFAULT_KERNEL, key="text1", columns=30),
         C.checkbox("Normalize Kernel", True, key="normalize")],
        supports_stack=True, types=ALL_TYPES)
    add("Gaussian Blur...", gaussian_blur, _gaussian_fields, supports_stack=True,
        types=RANK_TYPES)
    for label, kind in RANK_2D.items():
        add(label, _rank_command(kind), _radius(), supports_stack=True, types=RANK_TYPES)
    add("Unsharp Mask...", unsharp_mask,
        [C.number("Radius (Sigma)", 1.0, 1, "pixels"),
         C.number("Mask Weight (0.1-0.9)", 0.6, 2)], supports_stack=True, types=ALL_TYPES)
    add("Top Hat...", top_hat,
        _radius() + [C.checkbox("Light Background", False),
                     C.checkbox("Don't subtract (grayscale open)", False)],
        supports_stack=True, types=RANK_TYPES)
    add("Gaussian Blur 3D...", gaussian_blur_3d, _xyz("sigma"), types=ALL_TYPES, roi="none")
    for label, kind in RANK_3D.items():
        add(label, _rank3d_command(kind), _xyz("radius"), types=ALL_TYPES, roi="none")
    add("Show Circular Masks...", show_circular_masks, needs_image=False, undoable=False)


def selftest() -> None:
    from rsstatchem.core.imaging.session import Session

    s = Session()
    register(s.registry)
    yy, xx = np.mgrid[0:24, 0:24]
    data = ((xx * 7 + yy * 3) % 50 + 100 * (((xx - 12) ** 2 + (yy - 12) ** 2) < 20)).astype(
        np.uint8)
    img = ImageDoc.from_array(data, "YX", "f")
    s.add_image(img)
    s.run("Smooth")
    assert img.plane()[12, 12] > 90
    img.undo()
    s.run("Gaussian Blur...", "sigma=2")
    s.run("Median...", "radius=2")
    s.run("Convolve...", "text1=[1 1 1\n1 1 1\n1 1 1\n] normalize")
    s.run("Top Hat...", "radius=3")
    assert img.plane().dtype == np.uint8
    # 3x3 mean = Mean r=1, exact integer arithmetic
    assert np.array_equal(F.smooth(data)[5:-5, 5:-5],
                          ((np.add.reduce([np.roll(np.roll(data.astype(int), dy, 0), dx, 1)
                                           for dy in (-1, 0, 1) for dx in (-1, 0, 1)]) + 4)
                           // 9)[5:-5, 5:-5])
    assert F.JavaRandom(42).next(32) == -1170105035
    stk = ImageDoc.from_array(np.stack([data] * 3), "ZYX", "s")
    s.add_image(stk)
    s.run("Mean 3D...", "x=1 y=1 z=1")
    assert stk.n_slices == 3

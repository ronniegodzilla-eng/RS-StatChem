"""Image > Color: Split / Merge / Arrange Channels, Invert LUTs, Stack to RGB, Make Composite,
Show LUT, Display LUTs, Edit LUT helpers, and Colour Deconvolution.

Split/merge/arrange follow ImageJ 1.54 ij/plugin/{ChannelSplitter,RGBStackMerge,ChannelArranger,
CompositeConverter,RGBStackConverter}.java (public domain, NIH; W. Rasband). Colour deconvolution
is written from the paper (A. C. Ruifrok, D. A. Johnston, "Quantification of histochemical
staining by color deconvolution", Anal. Quant. Cytol. Histol. 23:291-299, 2001): optical density
OD = -log10((I + 1) / 256) per RGB channel, unit stain OD vectors as the rows of M (a missing
third vector is the normalised complement), concentrations C = OD * M^-1, and each stain shown as
the transmitted light 255 * 10^-C (Fiji's GPL plugin was not used).
"""
from __future__ import annotations

import numpy as np

from rsstatchem.core.imaging.image import COLOR_256, COLOR_RGB, GRAY8, GRAY16, GRAY32, ImageDoc
from rsstatchem.core.imaging.lut import LUT, named_lut

SLOT_COLORS = ("Red", "Green", "Blue", "Grays", "Cyan", "Magenta", "Yellow")
SLOT_LABELS = ("C1 (red):", "C2 (green):", "C3 (blue):", "C4 (gray):", "C5 (cyan):",
               "C6 (magenta):", "C7 (yellow):")


# ---------------------------------------------------------------------------------- split/merge
def split_channels(doc: ImageDoc) -> list[ImageDoc]:
    """Image > Color > Split Channels: RGB -> "<title> (red|green|blue)" 8-bit images;
    composite/hyperstack -> "C<n>-<title>" per channel (keeping z/t)."""
    if doc.type == COLOR_RGB:
        out = []
        for k, name in enumerate(("red", "green", "blue")):
            pix = np.ascontiguousarray(doc.pixels[..., k])
            im = ImageDoc(pix, GRAY8, f"{doc.title} ({name})")
            im.calibration = doc.calibration.copy()
            im.slice_labels = list(doc.slice_labels)
            out.append(im)
        return out
    if doc.n_channels == 1:
        raise ValueError("Multichannel image required")
    out = []
    for c in range(doc.n_channels):
        pix = np.ascontiguousarray(doc.pixels[:, :, c:c + 1])
        im = ImageDoc(pix, doc.type, f"C{c + 1}-{doc.title}")
        im.calibration = doc.calibration.copy()
        im.luts = [doc.luts[c].copy()]
        im.display_ranges = [doc.display_ranges[c]]
        im.slice_labels = [doc.slice_labels[doc.stack_index(c, z, t)]
                           for t in range(doc.n_frames) for z in range(doc.n_slices)]
        im.hyperstack = doc.n_slices > 1 and doc.n_frames > 1
        out.append(im)
    return out


def merge_channels(images: list[ImageDoc | None], create_composite: bool = True,
                   ignore_luts: bool = False, title: str | None = None) -> ImageDoc:
    """Image > Color > Merge Channels. ``images`` holds up to 7 entries (C1 red .. C7 yellow,
    None = "*None*"). Returns a composite (``create_composite``) or an RGB image."""
    used = [(i, im) for i, im in enumerate(images) if im is not None]
    if not used:
        raise ValueError("No source images")
    first = used[0][1]
    for _, im in used:
        if (im.width, im.height, im.stack_size) != (first.width, first.height,
                                                     first.stack_size):
            raise ValueError("The source images or stacks must have the same width, height "
                             "and depth.")
        if im.type == COLOR_RGB and len(used) > 1:
            raise ValueError("Source images must be grayscale")
    types = {im.type for _, im in used}
    if len(types) > 1:
        raise ValueError("The source images must have the same bit depth.")
    t = first.type
    simple_rgb = (not create_composite and t in (GRAY8, COLOR_256) and
                  all(i < 3 for i, _ in used) and
                  all(im.lut.is_grayscale() and not im.is_inverted_lut() for _, im in used))
    if simple_rgb:
        rgb = np.zeros(first.pixels.shape[:3] + (first.height, first.width, 3), np.uint8)
        for i, im in used:
            rgb[..., i] = im.pixels
        out = ImageDoc(rgb[:, :, :1] if rgb.shape[2] == 1 else rgb, COLOR_RGB, title or "RGB")
        out.calibration = first.calibration.copy()
        return out
    planes = np.stack([im.pixels[:, :, 0] for _, im in used], axis=2)  # (T, Z, C, Y, X)
    comp = ImageDoc(np.ascontiguousarray(planes), t, title or ("Composite" if create_composite
                                                              else "RGB"))
    comp.calibration = first.calibration.copy()
    luts, ranges = [], []
    for i, im in used:
        lut = im.lut
        if ignore_luts or lut.is_grayscale():
            lut = named_lut(SLOT_COLORS[i])
        luts.append(lut.copy())
        ranges.append(im.display_range)
    comp.luts = luts
    comp.display_ranges = ranges
    comp.composite_mode = "composite"
    comp.active_channels = [True] * comp.n_channels
    comp.hyperstack = True
    if create_composite:
        return comp
    from rsstatchem.core.imaging.convert import to_rgb
    to_rgb(comp)
    comp.title = title or "RGB"
    return comp


def arrange_channels(doc: ImageDoc, order: str) -> None:
    """Image > Color > Arrange Channels: ``order`` like "312" (channels may be dropped)."""
    idx = [int(ch) - 1 for ch in order.strip() if ch.isdigit()]
    if not idx or any(i < 0 or i >= doc.n_channels for i in idx):
        raise ValueError(f'Invalid channel order "{order}"')
    luts = [doc.luts[i].copy() for i in idx]
    ranges = [doc.display_ranges[i] for i in idx]
    active = [doc.active_channels[i] for i in idx]
    labels = [[doc.slice_labels[doc.stack_index(i, z, t)] for i in idx]
              for t in range(doc.n_frames) for z in range(doc.n_slices)]
    pix = np.ascontiguousarray(doc.pixels[:, :, idx])
    mode = doc.composite_mode
    doc.replace_pixels(pix, doc.type)
    doc.luts, doc.display_ranges, doc.active_channels = luts, ranges, active
    doc.composite_mode = mode if doc.n_channels > 1 else "color"
    doc.slice_labels = [lab for row in labels for lab in row]


def invert_luts(doc: ImageDoc) -> None:
    """Image > Color > Invert LUTs (every channel)."""
    doc.luts = [lut.inverted() for lut in doc.luts]


def make_composite(doc: ImageDoc, mode: str = "composite") -> None:
    """Image > Color > Make Composite: RGB -> 3-channel composite; a stack with 2-7 slices ->
    channels."""
    if doc.type == COLOR_RGB:
        arr = np.moveaxis(doc.pixels[:, :, 0], -1, 2)          # (T, Z, 3, Y, X)
        labels = list(doc.slice_labels)
        doc.replace_pixels(np.ascontiguousarray(arr), GRAY8)
        doc.luts = [named_lut(c) for c in ("Red", "Green", "Blue")]
        doc.slice_labels = [labels[i // 3] for i in range(doc.stack_size)]
    elif doc.n_channels == 1:
        n = doc.n_slices * doc.n_frames
        if not 2 <= n <= 7:
            raise ValueError("Stack with 2-7 slices, or RGB image, required")
        luts = doc.luts[0]
        rng = doc.display_range
        arr = doc.pixels.reshape((1, 1, n) + doc.pixels.shape[3:])
        doc.replace_pixels(np.ascontiguousarray(arr), doc.type)
        doc.luts = [named_lut(SLOT_COLORS[i]) for i in range(n)]
        if doc.bit_depth != 8:
            doc.display_ranges = [rng] * n
        del luts
    doc.composite_mode = mode
    doc.active_channels = [True] * doc.n_channels
    doc.hyperstack = True


def stack_to_rgb(doc: ImageDoc, slices: bool = True, frames: bool = True,
                 keep: bool = True) -> ImageDoc:
    """Image > Color > Stack to RGB: render a composite (or a 2/3-slice stack) as RGB, a new
    image (all slices/frames when requested, else the current position)."""
    from rsstatchem.core.imaging.convert import convert
    src = doc.duplicate(doc.title.rsplit(".", 1)[0] + " (RGB)")
    if src.n_channels == 1 and src.stack_size in (2, 3) and src.type in (GRAY8, GRAY16, GRAY32):
        make_composite(src)
    if not slices and src.n_slices > 1:
        z = doc.position[1]
        src.replace_pixels(np.ascontiguousarray(src.pixels[:, z:z + 1]), src.type)
        src.luts = [lut.copy() for lut in doc.luts]
        src.display_ranges = list(doc.display_ranges)
        src.composite_mode = doc.composite_mode
    if not frames and src.n_frames > 1:
        t = doc.position[2]
        src.replace_pixels(np.ascontiguousarray(src.pixels[t:t + 1]), src.type)
        src.luts = [lut.copy() for lut in doc.luts]
        src.display_ranges = list(doc.display_ranges)
        src.composite_mode = doc.composite_mode
    if src.type != COLOR_RGB:
        convert(src, "RGB Color")
    return src


# ---------------------------------------------------------------------------------- LUT images
def show_lut_image(lut: LUT, title: str = "LUT") -> ImageDoc:
    """Image > Color > Show LUT as an RGB image (a 256 x 32 ramp with the LUT applied, plus the
    red/green/blue curves are drawn by the UI)."""
    ramp = np.repeat(np.arange(256, dtype=np.uint8)[None, :], 32, axis=0)
    return ImageDoc.from_array(lut.as_array()[ramp], "YXS", title=title)


def display_luts(names=None) -> ImageDoc:
    """Image > Color > Display LUTs: a montage of ramps of every registered LUT (RGB)."""
    from rsstatchem.core.imaging import lut_tables as LT
    from rsstatchem.core.imaging.lut import NAMED_LUTS
    names = list(names) if names else [n for n in LT.IMAGEJ_LUTS] + \
        [n for n in NAMED_LUTS if n not in LT.IMAGEJ_LUTS]
    rows = []
    for n in names:
        lut = NAMED_LUTS[n]()
        ramp = np.repeat(np.arange(256, dtype=np.uint8)[None, :], 32, axis=0)
        rows.append(lut.as_array()[ramp])
        rows.append(np.full((4, 256, 3), 255, np.uint8))
    img = ImageDoc.from_array(np.concatenate(rows), "YXS", title="LUTs")
    img.info = "\n".join(names)
    img.properties["LUT names"] = ",".join(names)
    return img


def lut_from_points(points: list[tuple[int, int, int, int]]) -> LUT:
    """Edit LUT helper: piecewise-linear LUT through (index, r, g, b) control points."""
    pts = sorted(points)
    idx = [p[0] for p in pts]
    arr = np.stack([np.interp(np.arange(256), idx, [p[k] for p in pts]) for k in (1, 2, 3)],
                   axis=1)
    return LUT.from_array(np.round(arr).astype(np.uint8), "Edited")


# ---------------------------------------------------------------------------------- deconvolution
# Stain OD vectors (R, G, B) as published (Ruifrok & Johnston 2001 and standard follow-ups).
STAINS: dict[str, tuple] = {
    "H&E": ((0.644211, 0.716556, 0.266844), (0.092789, 0.954111, 0.283111), None),
    "H&E 2": ((0.49015734, 0.76897085, 0.41040173), (0.04615336, 0.8420684, 0.5373925), None),
    "H DAB": ((0.650, 0.704, 0.286), (0.268, 0.570, 0.776), None),
    "H&E DAB": ((0.650, 0.704, 0.286), (0.072, 0.990, 0.105), (0.268, 0.570, 0.776)),
    "H AEC": ((0.650, 0.704, 0.286), (0.2743, 0.6796, 0.6803), None),
    "H PAS": ((0.644211, 0.716556, 0.266844), (0.175411, 0.972178, 0.154589), None),
    "FastRed FastBlue DAB": ((0.21393921, 0.85112669, 0.47794022),
                             (0.74890292, 0.60624161, 0.26731082), (0.268, 0.570, 0.776)),
    "Methyl Green DAB": ((0.98003, 0.144316, 0.133146), (0.268, 0.570, 0.776), None),
    "Azan-Mallory": ((0.853033, 0.508733, 0.112656), (0.09289875, 0.8662008, 0.49098468),
                     (0.10732849, 0.36765403, 0.9237484)),
    "Alcian blue & H": ((0.874622, 0.457711, 0.158256), (0.552556, 0.7544, 0.353744), None),
    "Feulgen Light Green": ((0.46420921, 0.83008335, 0.30827187),
                            (0.94705542, 0.25373821, 0.19650764), None),
    "Giemsa": ((0.834750233, 0.513556283, 0.196330403), (0.092789, 0.954111, 0.283111), None),
    "Masson Trichrome": ((0.7995107, 0.5913521, 0.10528667), (0.09997159, 0.73738605,
                                                              0.6680326), None),
    "RGB": ((0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)),
    "CMY": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
}


def stain_matrix(vectors) -> np.ndarray:
    """3x3 matrix of unit stain OD vectors (rows); a missing (None/zero) third row is the
    normalised complement: sqrt(max(0, 1 - a^2 - b^2)) per channel (Ruifrok & Johnston)."""
    rows = []
    for v in vectors[:3]:
        if v is None or not np.any(v):
            rows.append(None)
            continue
        a = np.asarray(v, dtype=float)
        rows.append(a / np.linalg.norm(a))
    if rows[1] is None:
        rows[1] = np.array([rows[0][2], rows[0][0], rows[0][1]])
    if len(rows) < 3 or rows[2] is None:
        a, b = rows[0], rows[1]
        third = np.sqrt(np.maximum(0.0, 1.0 - a * a - b * b))
        n = np.linalg.norm(third)
        third = third / n if n > 0 else np.cross(a, b) / np.linalg.norm(np.cross(a, b))
        if len(rows) < 3:
            rows.append(third)
        else:
            rows[2] = third
    return np.vstack(rows)


def colour_deconvolution(rgb: np.ndarray, stains="H&E", output: str = "8bit_transmittance"
                         ) -> tuple[list[np.ndarray], np.ndarray]:
    """Unmix an RGB image (Y, X, 3) into 3 stain images.

    ``output``: "8bit_transmittance" (uint8 255*10^-C, the usual display), "32bit_OD"
    (float32 concentrations C, i.e. optical density per stain) or "32bit_Absorbance" (same).
    Returns (images, matrix)."""
    vectors = STAINS[stains] if isinstance(stains, str) else stains
    m = stain_matrix(vectors)
    od = -np.log10((rgb.astype(np.float64) + 1.0) / 256.0)
    conc = od.reshape(-1, 3) @ np.linalg.inv(m)
    conc = conc.reshape(rgb.shape[:2] + (3,))
    out = []
    for k in range(3):
        c = conc[..., k]
        if output.startswith("8bit"):
            t = 255.0 * np.power(10.0, -c)
            out.append(np.clip(np.floor(t + 0.5), 0, 255).astype(np.uint8))
        else:
            out.append(c.astype(np.float32))
    return out, m


def stain_lut(vector) -> LUT:
    """LUT for an 8-bit transmittance stain image: index i (= 255*10^-C) shows the colour of
    concentration C of this stain, 255*(i/255)^v per channel (white for 255)."""
    v = np.asarray(vector, dtype=float)
    v = v / (np.linalg.norm(v) or 1.0)
    i = np.arange(256) / 255.0
    rgb = np.stack([255.0 * np.power(i, v[k]) for k in range(3)], axis=1)
    return LUT.from_array(np.clip(np.round(rgb), 0, 255).astype(np.uint8), "stain")


def recompose(conc: list[np.ndarray], m: np.ndarray) -> np.ndarray:
    """Inverse of :func:`colour_deconvolution` with 32-bit concentrations: RGB uint8."""
    c = np.stack(conc, axis=-1).reshape(-1, 3).astype(np.float64)
    od = c @ m
    rgb = 256.0 * np.power(10.0, -od) - 1.0
    return np.clip(np.floor(rgb + 0.5), 0, 255).astype(np.uint8).reshape(conc[0].shape + (3,))


def value_types():
    return GRAY8, GRAY16, GRAY32

"""Image file I/O. **Owned by WP-A after Phase 0** (tiff.py, pillow_io.py, dicom.py, fits.py,
raw.py, avi.py, text_image.py, sequence.py, lut_io.py, results_io.py, opener.py, writer.py).

Phase 0 contract: ``open_image(path) -> ImageDoc`` dispatching on the extension through
:data:`READERS` (extension -> ``reader(path) -> ImageDoc``), and :func:`register_reader` so
format modules plug in without editing the dispatcher. The Phase 0 readers are minimal: TIFF
via tifffile (ImageJ hyperstack axes and calibration when present) and the Pillow formats with
the palette kept as the LUT (blobs.gif has an inverting LUT).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from rsstatchem.core.imaging.image import COLOR_256, GRAY8, ImageDoc
from rsstatchem.core.imaging.lut import LUT

READERS: dict[str, Callable[[str], ImageDoc]] = {}


def register_reader(extensions, reader: Callable[[str], ImageDoc]) -> None:
    for ext in ([extensions] if isinstance(extensions, str) else extensions):
        READERS[ext.lower()] = reader


def open_image(path) -> ImageDoc:
    path = str(path)
    reader = READERS.get(Path(path).suffix.lower())
    if reader is None:
        raise ValueError(f"File is not in a supported format:\n{path}")
    doc = reader(path)
    doc.path = path
    doc.changed = False
    doc.forget_undo()
    return doc


def read_pillow(path: str) -> ImageDoc:
    """GIF/PNG/JPEG/BMP/PGM (first frame); palette images keep their palette as the LUT."""
    from PIL import Image

    title = Path(path).name
    with Image.open(path) as im:
        if im.mode == "P":
            pal = np.asarray(im.getpalette() or [], dtype=np.uint8)
            lut = LUT.from_array(pal.reshape(-1, 3) if pal.size else np.zeros((256, 3)))
            doc = ImageDoc.from_array(np.asarray(im), "YX", title)
            doc.luts = [lut]
            if not (lut.is_grayscale() or lut.is_inverted()):
                doc.type = COLOR_256
            return doc
        if im.mode in ("L", "1"):
            return ImageDoc.from_array(np.asarray(im.convert("L")), "YX", title)
        if im.mode in ("I;16", "I;16B", "I;16L"):
            return ImageDoc.from_array(np.asarray(im).astype(np.uint16), "YX", title)
        if im.mode in ("I", "F"):
            return ImageDoc.from_array(np.asarray(im, dtype=np.float32), "YX", title)
        return ImageDoc.from_array(np.asarray(im.convert("RGB")), "YXS", title)


def read_tiff(path: str) -> ImageDoc:
    """TIFF via tifffile: ImageJ hyperstacks (axes, spacing/unit, ranges) when present."""
    import tifffile

    title = Path(path).name
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        data = series.asarray()
        axes = series.axes.upper().replace("Q", "Z").replace("I", "Z")
        ij = tif.imagej_metadata or {}
        page = tif.pages[0]
        if page.photometric == tifffile.PHOTOMETRIC.RGB or axes.endswith("S"):
            axes = axes[:-1] + "S" if axes.endswith("S") else axes
        doc = ImageDoc.from_array(data, axes, title)
        res = page.tags.get("XResolution")
        if res is not None and ij.get("unit"):
            num, den = res.value
            if num:
                doc.calibration.pixel_width = den / num
                yres = page.tags.get("YResolution")
                doc.calibration.pixel_height = (yres.value[1] / yres.value[0]) if yres else \
                    doc.calibration.pixel_width
                doc.calibration.unit = str(ij.get("unit")).replace("\\u00B5", "µ")
                doc.calibration.pixel_depth = float(ij.get("spacing", 1.0))
        if "min" in ij and "max" in ij and doc.type != GRAY8:
            doc.display_ranges = [(float(ij["min"]), float(ij["max"]))] * doc.n_channels
        if ij.get("Labels"):
            labels = list(ij["Labels"])[:doc.stack_size]
            doc.slice_labels = labels + [None] * (doc.stack_size - len(labels))
        if ij.get("mode") in ("composite", "color", "grayscale") and doc.n_channels > 1:
            doc.composite_mode = ij["mode"]
        if ij.get("Info"):
            doc.info = str(ij["Info"])
        if page.photometric == tifffile.PHOTOMETRIC.PALETTE and page.colormap is not None:
            cmap = np.asarray(page.colormap)
            doc.luts = [LUT.from_array((cmap.T >> 8).astype(np.uint8))]
        elif page.photometric == tifffile.PHOTOMETRIC.MINISWHITE and doc.type == GRAY8:
            doc.luts = [LUT.grays().inverted()]
        return doc


register_reader((".gif", ".png", ".jpg", ".jpeg", ".bmp", ".pgm", ".pbm", ".ppm"), read_pillow)
register_reader((".tif", ".tiff"), read_tiff)

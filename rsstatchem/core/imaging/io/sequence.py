"""Image sequences: File > Import > Image Sequence... (``ij/plugin/FolderOpener.java``) and
File > Save As > Image Sequence... (``ij/plugin/StackWriter.java``); ported behaviour from
ImageJ 1.54 (public domain, NIH).

Opening lists the image files of a folder, optionally filtered by a substring or a regular
expression in parentheses, sorted "numerically" (embedded numbers compared by value, ImageJ's
``StringSorter.sortNumerically``), takes ``count`` files from ``start`` with ``step``, scales
them by ``scale`` percent and stacks images of the first image's size (other sizes are
skipped; mixed types are converted to the widest type). Slice labels are the file names.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from rsstatchem.core.imaging.image import COLOR_RGB, GRAY8, GRAY16, GRAY32, ImageDoc

IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".pgm", ".ppm",
                    ".pbm", ".dcm", ".fits", ".fit", ".fts", ".zip")


def numeric_key(name: str):
    """Sort key comparing digit runs by value ("img2" < "img10")."""
    parts = re.split(r"(\d+)", name.lower())
    return [(0, int(p), len(p)) if p.isdigit() else (1, p, 0) for p in parts]


def list_images(directory, file_filter: str = "", sort_numerically: bool = True) -> list[Path]:
    d = Path(directory)
    files = [p for p in d.iterdir() if p.is_file() and not p.name.startswith(".")
             and p.suffix.lower() in IMAGE_EXTENSIONS]
    if file_filter:
        if file_filter.startswith("(") and file_filter.endswith(")"):
            rx = re.compile(file_filter[1:-1])
            files = [p for p in files if rx.search(p.name)]
        else:
            files = [p for p in files if file_filter in p.name]
    files.sort(key=(lambda p: numeric_key(p.name)) if sort_numerically else
               (lambda p: p.name))
    return files


_RANK = {GRAY8: 0, GRAY16: 1, GRAY32: 2, COLOR_RGB: 3}


def _convert(doc: ImageDoc, target: int) -> np.ndarray:
    """Planes of ``doc`` converted (unscaled) to ``target`` type as (N, Y, X[, 3])."""
    planes = doc.pixels.reshape((-1,) + doc.pixels.shape[3:])
    if doc.type == target:
        return planes
    if target == COLOR_RGB:
        from rsstatchem.core.imaging.lut import render
        return np.stack([render(doc, *doc.position_of(i)) for i in range(doc.stack_size)])
    if target == GRAY32:
        return planes.astype(np.float32)
    if target == GRAY16:
        return planes.astype(np.uint16)
    return planes


def open_sequence(directory, count: int = 0, start: int = 1, step: int = 1,
                  scale: float = 100.0, file_filter: str = "", sort_numerically: bool = True,
                  convert_rgb: bool = False, progress=None) -> ImageDoc:
    from rsstatchem.core.imaging.io import open_image

    files = list_images(directory, file_filter, sort_numerically)
    if not files:
        raise ValueError(f"No images found in\n{directory}")
    start = max(1, int(start))
    files = files[start - 1::max(1, int(step))]
    if count and count > 0:
        files = files[:int(count)]
    docs, labels = [], []
    first = None
    for i, f in enumerate(files):
        if progress is not None:
            progress(i / len(files))
        try:
            doc = open_image(f)
        except Exception:  # noqa: BLE001 - unreadable files are skipped like ImageJ
            continue
        if scale != 100.0:
            doc = _scaled(doc, scale / 100.0)
        if first is None:
            first = doc
        if (doc.width, doc.height) != (first.width, first.height):
            continue
        docs.append(doc)
        for k in range(doc.stack_size):
            lab = doc.slice_labels[k]
            labels.append(f.name if doc.stack_size == 1 else (lab or f"{f.name}:{k + 1}"))
    if not docs:
        raise ValueError("No images of a common size were found")
    target = max((d.type if d.type in _RANK else GRAY8 for d in docs), key=_RANK.get)
    if convert_rgb:
        target = COLOR_RGB
    planes = np.concatenate([_convert(d, target) for d in docs])
    axes = "ZYXS" if target == COLOR_RGB else "ZYX"
    out = ImageDoc.from_array(planes, axes, Path(directory).name or "Stack")
    out.calibration = first.calibration.copy()
    if target != COLOR_RGB and len({d.type for d in docs}) == 1:
        out.luts = [first.lut.copy()]
    out.slice_labels = labels[:out.stack_size]
    out.info = first.info
    out.reset_display_range()
    if progress is not None:
        progress(1.0)
    return out


def _scaled(doc: ImageDoc, f: float) -> ImageDoc:
    from scipy import ndimage

    h, w = max(1, int(doc.height * f)), max(1, int(doc.width * f))
    planes = doc.pixels.reshape((-1,) + doc.pixels.shape[3:])
    out = []
    for p in planes:
        zoom = (h / p.shape[0], w / p.shape[1]) + ((1,) if p.ndim == 3 else ())
        out.append(ndimage.zoom(p.astype(np.float32), zoom, order=1)[:h, :w])
    arr = np.stack(out)
    if doc.pixels.dtype != np.float32:
        arr = np.clip(np.rint(arr), 0, np.iinfo(doc.pixels.dtype).max).astype(doc.pixels.dtype)
    new = ImageDoc.from_array(arr, "ZYXS" if doc.is_rgb else "ZYX", doc.title)
    new.luts = [lt.copy() for lt in doc.luts[:1]] if not doc.is_rgb else new.luts
    new.calibration = doc.calibration.copy()
    new.calibration.pixel_width /= f
    new.calibration.pixel_height /= f
    return new


FORMATS = {"TIFF": ".tif", "GIF": ".gif", "JPEG": ".jpg", "BMP": ".bmp", "PNG": ".png",
           "PGM": ".pgm", "FITS": ".fits", "Text Image": ".txt", "ZIP": ".zip", "Raw": ".raw"}


def save_sequence(doc: ImageDoc, directory, fmt: str = "TIFF", name: str = "",
                  start: int = 0, digits: int = 4, use_labels: bool = False,
                  writer=None) -> list[Path]:
    """Write every plane as its own file "name0000.tif"... (StackWriter)."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    ext = FORMATS.get(fmt, "." + fmt.lower())
    stem = name or Path(doc.title).stem
    written = []
    for i in range(doc.stack_size):
        c, z, t = doc.position_of(i)
        plane = ImageDoc(doc.pixels[t:t + 1, z:z + 1, c:c + 1].copy(), doc.type,
                         doc.slice_labels[i] or stem, doc.calibration.copy())
        plane.luts = [doc.luts[c].copy()]
        plane.display_ranges = [doc.display_ranges[c]]
        label = doc.slice_labels[i]
        if use_labels and label:
            fname = re.sub(r'[\\/:*?"<>|\n]', "_", label.splitlines()[0])
            if not Path(fname).suffix:
                fname += ext
        else:
            fname = f"{stem}{start + i:0{max(digits, 1)}d}{ext}"
        path = d / fname
        if writer is not None:
            writer(plane, path, fmt)
        else:
            from rsstatchem.core.imaging.io import save_image
            save_image(plane, path, fmt)
        written.append(path)
    return written

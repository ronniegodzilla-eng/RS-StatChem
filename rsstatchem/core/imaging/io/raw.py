"""File > Import > Raw... and File > Save As > Raw Data....

Follows ImageJ 1.54 ``ij/io/ImportDialog.java``, ``ImageReader.java`` and ``ImageWriter.java``
(public domain, NIH): the 14 raw types, offset, number of images, gap between images, white is
zero, little-endian, "open all files in folder"; 16-bit signed data get ImageJ's signed
calibration (raw = v + 32768), 32-bit integers become 32-bit float, 24-bit integer and ARGB as
ImageJ reads them. Raw Data writes the planes in stack order, big-endian unless "Intel byte
order" is set (Edit > Options > Input/Output).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rsstatchem.core.imaging.image import ImageDoc

RAW_TYPES = ("8-bit", "16-bit Signed", "16-bit Unsigned", "32-bit Signed", "32-bit Unsigned",
             "32-bit Real", "64-bit Real", "24-bit RGB", "24-bit RGB Planar", "24-bit BGR",
             "24-bit Integer", "32-bit ARGB", "32-bit ABGR", "1-bit Bitmap")
BYTES = {"8-bit": 1, "16-bit Signed": 2, "16-bit Unsigned": 2, "32-bit Signed": 4,
         "32-bit Unsigned": 4, "32-bit Real": 4, "64-bit Real": 8, "24-bit RGB": 3,
         "24-bit RGB Planar": 3, "24-bit BGR": 3, "24-bit Integer": 3, "32-bit ARGB": 4,
         "32-bit ABGR": 4}


@dataclass
class RawOptions:
    image_type: str = "8-bit"
    width: int = 64
    height: int = 64
    offset: int = 0
    n_images: int = 1
    gap: int = 0
    white_is_zero: bool = False
    little_endian: bool = False


def plane_bytes(opts: RawOptions) -> int:
    if opts.image_type == "1-bit Bitmap":
        return ((opts.width + 7) // 8) * opts.height
    return opts.width * opts.height * BYTES[opts.image_type]


def _decode(buf: bytes, opts: RawOptions) -> np.ndarray:
    w, h, t = opts.width, opts.height, opts.image_type
    e = "<" if opts.little_endian else ">"
    if t == "8-bit":
        return np.frombuffer(buf, np.uint8, w * h).reshape(h, w)
    if t in ("16-bit Signed", "16-bit Unsigned"):
        dt = e + ("i2" if t == "16-bit Signed" else "u2")
        return np.frombuffer(buf, dt, w * h).reshape(h, w)
    if t in ("32-bit Signed", "32-bit Unsigned", "32-bit Real", "64-bit Real"):
        dt = e + {"32-bit Signed": "i4", "32-bit Unsigned": "u4", "32-bit Real": "f4",
                  "64-bit Real": "f8"}[t]
        return np.frombuffer(buf, dt, w * h).reshape(h, w).astype(np.float32)
    if t in ("24-bit RGB", "24-bit BGR"):
        a = np.frombuffer(buf, np.uint8, w * h * 3).reshape(h, w, 3)
        return a[..., ::-1].copy() if t == "24-bit BGR" else a.copy()
    if t == "24-bit RGB Planar":
        a = np.frombuffer(buf, np.uint8, w * h * 3).reshape(3, h, w)
        return np.moveaxis(a, 0, -1).copy()
    if t == "24-bit Integer":
        a = np.frombuffer(buf, np.uint8, w * h * 3).reshape(h, w, 3).astype(np.uint32)
        v = (a[..., 0] << 16 | a[..., 1] << 8 | a[..., 2]) if not opts.little_endian else \
            (a[..., 2] << 16 | a[..., 1] << 8 | a[..., 0])
        return v.astype(np.float32)
    if t in ("32-bit ARGB", "32-bit ABGR"):
        a = np.frombuffer(buf, np.uint8, w * h * 4).reshape(h, w, 4)
        if opts.little_endian:
            a = a[..., ::-1]
        rgb = a[..., 1:4] if t == "32-bit ARGB" else a[..., 1:4][..., ::-1]
        return rgb.copy()
    if t == "1-bit Bitmap":
        rows = (w + 7) // 8
        a = np.frombuffer(buf, np.uint8, rows * h).reshape(h, rows)
        bits = np.unpackbits(a, axis=1)[:, :w]
        return (bits * 255).astype(np.uint8)
    raise ValueError(f"unknown raw type {t!r}")


def read_raw(path, opts: RawOptions) -> ImageDoc:
    data = Path(path).read_bytes()
    size = plane_bytes(opts)
    planes = []
    pos = opts.offset
    n = opts.n_images
    if n <= 0:
        n = max(1, (len(data) - opts.offset + opts.gap) // (size + opts.gap))
    for _ in range(n):
        if pos + size > len(data):
            break
        planes.append(_decode(data[pos:pos + size], opts))
        pos += size + opts.gap
    if not planes:
        raise ValueError("File is too short for the raw image description")
    arr = np.stack(planes)
    title = Path(path).name
    if arr.ndim == 4:
        doc = ImageDoc.from_array(arr, "ZYXS", title)
    else:
        doc = ImageDoc.from_array(arr, "ZYX", title)
    if opts.white_is_zero and not doc.is_rgb:
        from rsstatchem.core.imaging.lut import LUT
        if doc.bit_depth == 8:
            doc.luts = [LUT.grays().inverted()]
        else:
            doc.pixels = (doc.pixels.max() - doc.pixels).astype(doc.pixels.dtype)
    doc.reset_display_range()
    return doc


def raw_bytes(doc: ImageDoc, little_endian: bool = False) -> bytes:
    """Raw Data: every plane in stack order (RGB interleaved)."""
    e = "<" if little_endian else ">"
    if doc.is_rgb:
        return np.ascontiguousarray(doc.pixels).tobytes()
    dt = doc.pixels.dtype
    if dt.itemsize > 1:
        dt = dt.newbyteorder(e)
    return np.ascontiguousarray(doc.pixels, dtype=dt).tobytes()


def write_raw(doc: ImageDoc, path, little_endian: bool = False) -> None:
    Path(path).write_bytes(raw_bytes(doc, little_endian))

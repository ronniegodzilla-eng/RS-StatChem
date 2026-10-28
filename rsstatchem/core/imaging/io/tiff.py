"""TIFF reading (tifffile; ImageJ hyperstacks and metadata) and ImageJ-compatible writing.

The writer is a port of ImageJ 1.54 ``ij/io/TiffEncoder.java``, ``FileSaver.getDescriptionString``
and ``ImagePlus.getFileInfo`` (public domain, NIH; W. Rasband et al.): uncompressed strips, one
IFD per stack plane (czt order), the ImageJ description (images/channels/slices/frames,
hyperstack, mode, unit, value calibration, finterval, spacing, loop, min/max, origins) and the
IJMetadata tags (Info, slice labels, display ranges and LUTs of composites, the ROI, overlay and
properties), in big-endian ("MM") or Intel byte order. Files written here are byte-identical to
ImageJ's for the same image (checked against goldens from ImageJ 1.54p).

Reading uses tifffile for all TIFF flavours (ImageJ, OME, LSM, STK, ...); LZW strips decode
through :mod:`.lzw` when imagecodecs is not installed.
"""
from __future__ import annotations

import math
import struct
from decimal import Decimal
from pathlib import Path

import numpy as np

from rsstatchem.core.imaging.calibration import NONE, Calibration
from rsstatchem.core.imaging.image import (COLOR_256, COLOR_RGB, GRAY8, GRAY16, GRAY32,
                                           ImageDoc)
from rsstatchem.core.imaging.lut import LUT

IJ_VERSION = "1.54p"     # format version written in the description (ImageJ checks "ImageJ=")

# TIFF tags
NEW_SUBFILE_TYPE, IMAGE_WIDTH, IMAGE_LENGTH, BITS_PER_SAMPLE = 254, 256, 257, 258
COMPRESSION, PHOTO_INTERP, IMAGE_DESCRIPTION, STRIP_OFFSETS = 259, 262, 270, 273
SAMPLES_PER_PIXEL, ROWS_PER_STRIP, STRIP_BYTE_COUNT = 277, 278, 279
X_RESOLUTION, Y_RESOLUTION, RESOLUTION_UNIT = 282, 283, 296
COLOR_MAP, SAMPLE_FORMAT = 320, 339
META_DATA_BYTE_COUNTS, META_DATA = 50838, 50839
MAGIC_NUMBER, INFO, LABELS, RANGES, LUTS, PLOT, ROI, OVERLAY, PROPERTIES = (
    0x494a494a, 0x696e666f, 0x6c61626c, 0x72616e67, 0x6c757473, 0x706c6f74, 0x726f6920,
    0x6f766572, 0x70726f70)


# ---------------------------------------------------------------------------------- helpers
def java_double(v: float) -> str:
    """``Double.toString`` of Java (shortest digits; "1.0E7" style outside [1e-3, 1e7))."""
    v = float(v)
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "Infinity" if v > 0 else "-Infinity"
    if v == 0:
        return "-0.0" if math.copysign(1, v) < 0 else "0.0"
    a = abs(v)
    if 1e-3 <= a < 1e7:
        s = repr(v)
        if "e" in s or "E" in s:
            s = format(Decimal(s), "f")
        if "." not in s:
            s += ".0"
        return s
    sign, digits, exp = Decimal(repr(v)).as_tuple()
    digits = list(digits)
    while len(digits) > 1 and digits[-1] == 0:
        digits.pop()
        exp += 1
    mant = str(digits[0]) + "." + ("".join(map(str, digits[1:])) or "0")
    e = len(digits) - 1 + exp
    return ("-" if sign else "") + f"{mant}E{e}"


def _escaped(s: str) -> str:
    """FileSaver.appendEscapedLine: non-ASCII and backslash as \\uXXXX."""
    out = []
    for ch in s:
        c = ord(ch)
        if 0x20 <= c < 0x7f and ch != "\\":
            out.append(ch)
        elif c <= 0xFFFF:
            out.append(f"\\u{c:04X}")
    return "".join(out)


def _unescape(s: str) -> str:
    import re
    return re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), s)


def _is_default_gray(lut: LUT) -> bool:
    return np.array_equal(lut.as_array(), LUT.grays().as_array())


# ---------------------------------------------------------------------------------- writer
def description_string(doc: ImageDoc) -> str:
    """ImageJ's TIFF ImageDescription for ``doc`` (FileSaver.getDescriptionString)."""
    cal = doc.calibration
    n_images = doc.stack_size
    sb = [f"ImageJ={IJ_VERSION}\n"]
    if n_images > 1:
        sb.append(f"images={n_images}\n")
    c, z, t = doc.dims
    if c > 1:
        sb.append(f"channels={c}\n")
    if z > 1:
        sb.append(f"slices={z}\n")
    if t > 1:
        sb.append(f"frames={t}\n")
    if is_hyperstack(doc):
        sb.append("hyperstack=true\n")
    if doc.is_composite:
        sb.append(f"mode={doc.composite_mode}\n")
    if cal.scaled():
        sb.append(_escaped("unit=" + cal.unit) + "\n")
    bit_depth = doc.bit_depth
    value_unit = None
    if cal.calibrated():
        value_unit = cal.value_unit
        if bit_depth != 32:
            sb.append(f"cf={cal.function}\n")
            for i, coef in enumerate(cal.coefficients or ()):
                sb.append(f"c{i}={java_double(coef)}\n")
    elif cal.value_unit != "Gray Value":
        value_unit = cal.value_unit
    if value_unit is not None:
        sb.append(_escaped("vunit=" + value_unit) + "\n")
    if cal.frame_interval != 0.0:
        fi = cal.frame_interval
        sb.append(f"finterval={int(fi) if int(fi) == fi else java_double(fi)}\n")
    if cal.time_unit != "sec":
        sb.append(_escaped("tunit=" + cal.time_unit) + "\n")
    if cal.y_unit and cal.y_unit != cal.unit:
        sb.append(_escaped("yunit=" + cal.y_unit) + "\n")
    if cal.z_unit and cal.z_unit != cal.unit:
        sb.append(_escaped("zunit=" + cal.z_unit) + "\n")
    if n_images > 1:
        if cal.pixel_depth != 1.0:
            sb.append(f"spacing={java_double(cal.pixel_depth)}\n")
        if cal.fps != 0.0:
            sb.append(f"fps={int(cal.fps) if int(cal.fps) == cal.fps else java_double(cal.fps)}\n")
        loop = str(cal.info.get("loop", "false")).lower() == "true"
        sb.append(f"loop={'true' if loop else 'false'}\n")
    vmin, vmax = doc.display_range
    if doc.type in (GRAY8, COLOR_256):
        if (vmin, vmax) != (0.0, 255.0):
            sb.append(f"min={java_double(vmin)}\nmax={java_double(vmax)}\n")
    elif doc.type in (GRAY16, GRAY32):
        sb.append(f"min={java_double(vmin)}\nmax={java_double(vmax)}\n")
    if cal.x_origin != 0.0:
        sb.append(f"xorigin={java_double(cal.x_origin)}\n")
    if cal.y_origin != 0.0:
        sb.append(f"yorigin={java_double(cal.y_origin)}\n")
    if cal.z_origin != 0.0:
        sb.append(f"zorigin={java_double(cal.z_origin)}\n")
    info = cal.info.get("info") if isinstance(cal.info, dict) else None
    if info and len(info) <= 64 and "=" not in info and "\n" not in info:
        sb.append(_escaped("info=" + info) + "\n")
    if cal.invert_y:
        sb.append("inverty=true\n")
    if doc.type == COLOR_256:
        sb.append("8bitcolor=true\n")
    sb.append("\0")
    return "".join(sb)


def is_hyperstack(doc: ImageDoc) -> bool:
    c, z, t = doc.dims
    if doc.is_rgb and c == 1:
        return doc.hyperstack and (z > 1 and t > 1)
    return bool(doc.hyperstack and doc.stack_size > 1) or doc.is_composite or \
        sum(d > 1 for d in (c, z, t)) >= 2


def tiff_bytes(doc: ImageDoc, little_endian: bool = False, save_roi: bool = True) -> bytes:
    """The image as an ImageJ TIFF (TiffEncoder port)."""
    from rsstatchem.core.imaging.io.roi_codec import encode_roi

    bo = "<" if little_endian else ">"
    width, height, n_images = doc.width, doc.height, doc.stack_size
    lut = doc.lut
    composite = doc.is_composite
    color_map = None
    bits, spp, bpp, bps_size = 8, 1, 1, 0
    inverted = doc.is_inverted_lut()
    if doc.type in (GRAY8, COLOR_256):
        custom = doc.type == COLOR_256 or not _is_default_gray(lut)
        photo = 3 if custom else 1
        if custom:
            color_map = lut
    elif doc.type == GRAY16:
        bits, bpp = 16, 2
        photo = 0 if inverted else 1
        if not composite and not _is_default_gray(lut):
            color_map = lut
    elif doc.type == GRAY32:
        bits, bpp = 32, 4
        photo = 0 if inverted else 1
        if not composite and not _is_default_gray(lut):
            color_map = lut
    else:
        photo, spp, bpp, bps_size = 2, 3, 3, 6
    cal = doc.calibration
    scaled = cal.scaled() and cal.pixel_width != 0 and cal.pixel_height != 0
    n_entries = 10 + (1 if color_map is not None else 0) + (3 if scaled else 0) + \
        (1 if doc.type == GRAY32 else 0)
    description = description_string(doc).encode("latin-1", "replace")
    n_entries += 1
    image_size = width * height * bpp

    # ---- metadata (IJMetadata tags)
    labels = list(doc.slice_labels[:n_images])
    if not any(lb for lb in labels):
        labels = []
    if n_images == 1 and labels and labels[0] is None:
        labels = []
    entries: list[tuple[int, list[bytes]]] = []
    info = doc.info or ""
    if info:
        entries.append((INFO, [_chars(info, bo)]))
    if labels:
        entries.append((LABELS, [_chars(lb or "", bo) for lb in labels]))
    if composite:
        ranges = b"".join(struct.pack(bo + "d", float(v)) for r in doc.display_ranges
                          for v in r)
        entries.append((RANGES, [ranges]))
        if doc.composite_mode != "grayscale":
            entries.append((LUTS, [lt.as_array().T.astype(np.uint8).tobytes()
                                   for lt in doc.luts]))
    if save_roi and doc.roi is not None:
        try:
            entries.append((ROI, [encode_roi(doc.roi)]))
        except ValueError:
            pass
    overlay = doc.overlay
    if overlay is not None and not overlay.hidden and len(overlay):
        blobs = []
        for i, r in enumerate(overlay):
            try:
                blobs.append(encode_roi(r, overlay if i == 0 else None))
            except ValueError:
                continue
        if blobs:
            entries.append((OVERLAY, blobs))
    if doc.properties:
        props = []
        for k, v in doc.properties.items():
            props += [_chars(str(k), bo), _chars(str(v), bo)]
        entries.append((PROPERTIES, props))
    n_meta_entries = sum(len(b) for _, b in entries)
    meta_size = sum(len(x) for _, b in entries for x in b)
    if n_meta_entries:
        n_meta_entries += 1
    hdr_size = 4 + len(entries) * 8
    if meta_size > 0:
        meta_size += hdr_size
    if meta_size > 0:
        n_entries += 2
    ifd_size = 2 + n_entries * 12 + 4
    scale_size = 16 if scaled else 0
    cmap_size = 768 * 2 if color_map is not None else 0
    image_offset = 8 + ifd_size + bps_size + len(description) + scale_size + cmap_size + \
        n_meta_entries * 4 + meta_size

    out = bytearray()
    out += (b"II*\0" if little_endian else b"MM\0*") + struct.pack(bo + "I", 8)

    def entry(buf, tag, ftype, count, value):
        buf += struct.pack(bo + "HHI", tag, ftype, count)
        if count == 1 and ftype == 3:
            buf += struct.pack(bo + "HH", value, 0)
        else:
            buf += struct.pack(bo + "I", value & 0xFFFFFFFF)

    def ifd(buf, offset, next_ifd, with_meta):
        tag_data = 8 + ifd_size
        ne = n_entries if with_meta or meta_size == 0 else n_entries - 2
        buf += struct.pack(bo + "H", ne)
        entry(buf, NEW_SUBFILE_TYPE, 4, 1, 0)
        entry(buf, IMAGE_WIDTH, 4, 1, width)
        entry(buf, IMAGE_LENGTH, 4, 1, height)
        if doc.type == COLOR_RGB:
            entry(buf, BITS_PER_SAMPLE, 3, 3, tag_data)
            tag_data += 6
        else:
            entry(buf, BITS_PER_SAMPLE, 3, 1, bits)
        entry(buf, COMPRESSION, 3, 1, 1)
        entry(buf, PHOTO_INTERP, 3, 1, photo)
        entry(buf, IMAGE_DESCRIPTION, 2, len(description), tag_data)
        tag_data += len(description)
        entry(buf, STRIP_OFFSETS, 4, 1, offset)
        entry(buf, SAMPLES_PER_PIXEL, 3, 1, spp)
        entry(buf, ROWS_PER_STRIP, 4, 1, height)
        entry(buf, STRIP_BYTE_COUNT, 4, 1, image_size)
        if scaled:
            entry(buf, X_RESOLUTION, 5, 1, tag_data)
            entry(buf, Y_RESOLUTION, 5, 1, tag_data + 8)
            tag_data += 16
            unit = 2 if cal.unit == "inch" else 3 if cal.unit == "cm" else 1
            entry(buf, RESOLUTION_UNIT, 3, 1, unit)
        if doc.type == GRAY32:
            entry(buf, SAMPLE_FORMAT, 3, 1, 3)
        if color_map is not None:
            entry(buf, COLOR_MAP, 3, 768, tag_data)
            tag_data += 1536
        if with_meta and meta_size > 0:
            entry(buf, META_DATA_BYTE_COUNTS, 4, n_meta_entries, tag_data)
            entry(buf, META_DATA, 1, meta_size, tag_data + 4 * n_meta_entries)
        buf += struct.pack(bo + "I", next_ifd)

    stack_size = image_size * n_images
    next_ifd = image_offset + stack_size if n_images > 1 else 0
    ifd(out, image_offset, next_ifd, True)
    if doc.type == COLOR_RGB:
        out += struct.pack(bo + "HHH", 8, 8, 8)
    out += description
    if scaled:
        xscale, yscale = 1.0 / cal.pixel_width, 1.0 / cal.pixel_height
        scale = 1000000.0
        imax = 2147483647
        if xscale * scale > imax or yscale * scale > imax:
            scale = float(int(imax / max(xscale, yscale)))
        out += struct.pack(bo + "iiii", int(xscale * scale), int(scale), int(yscale * scale),
                           int(scale))
    if color_map is not None:
        rgb = color_map.as_array().T.astype(np.uint16)
        table = (rgb << 8).astype(bo + "u2")    # 8-bit entries in the high byte, as ImageJ
        out += table.tobytes()
    if meta_size > 0:
        out += struct.pack(bo + "I", hdr_size)
        for _, blobs in entries:
            for b in blobs:
                out += struct.pack(bo + "I", len(b))
        out += struct.pack(bo + "I", MAGIC_NUMBER)
        for kind, blobs in entries:
            out += struct.pack(bo + "II", kind, len(blobs) if kind not in (INFO, RANGES, ROI)
                               else 1)
        for _, blobs in entries:
            for b in blobs:
                out += b
    assert len(out) == image_offset, (len(out), image_offset)
    out += _pixel_bytes(doc, bo)
    if n_images > 1:
        nxt = next_ifd
        offset = image_offset
        ifd2_size = ifd_size - (24 if meta_size > 0 else 0)
        for i in range(2, n_images + 1):
            nxt = 0 if i == n_images else nxt + ifd2_size
            offset += image_size
            ifd(out, offset, nxt, False)
    return bytes(out)


def _chars(s: str, bo: str) -> bytes:
    return s.encode("utf-16-be" if bo == ">" else "utf-16-le")


def _pixel_bytes(doc: ImageDoc, bo: str) -> bytes:
    if doc.is_rgb:
        planes = doc.pixels.reshape((-1, doc.height, doc.width, 3))
        return np.ascontiguousarray(planes).tobytes()
    planes = doc.pixels.reshape((-1, doc.height, doc.width))
    dt = np.dtype(doc.pixels.dtype).newbyteorder(bo) if doc.pixels.dtype.itemsize > 1 else \
        doc.pixels.dtype
    return np.ascontiguousarray(planes, dtype=dt).tobytes()


def write_tiff(doc: ImageDoc, path, little_endian: bool = False) -> None:
    Path(path).write_bytes(tiff_bytes(doc, little_endian))


# ---------------------------------------------------------------------------------- reader
def parse_description(text: str) -> dict[str, str]:
    """ImageJ description "key=value" lines (values still escaped strings)."""
    out = {}
    for line in text.replace("\r", "\n").split("\n"):
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().rstrip("\0")
    return out


def read_tiff(path) -> ImageDoc:
    """Open a TIFF like ImageJ: ImageJ hyperstacks with calibration, display ranges, LUTs,
    labels, Info, properties, ROI and overlay; other TIFFs (OME, multi-page, planar RGB,
    palette, LZW) as stacks."""
    from rsstatchem.core.imaging.io import lzw

    import tifffile

    lzw.install()
    title = Path(path).name
    with tifffile.TiffFile(str(path)) as tif:
        return _doc_from_tifffile(tif, title)


def read_tiff_bytes(data: bytes, title: str = "Untitled") -> ImageDoc:
    import io

    from rsstatchem.core.imaging.io import lzw

    import tifffile

    lzw.install()
    with tifffile.TiffFile(io.BytesIO(data)) as tif:
        return _doc_from_tifffile(tif, title)


def _doc_from_tifffile(tif, title: str) -> ImageDoc:
    import tifffile

    page = tif.pages[0]
    desc_tag = page.tags.get(IMAGE_DESCRIPTION)
    desc = parse_description(desc_tag.value) if desc_tag is not None and \
        str(desc_tag.value).startswith("ImageJ") else {}
    ij = tif.imagej_metadata if desc else None
    series = tif.series[0]
    data = series.asarray()
    axes = series.axes.upper()
    photometric = page.photometric
    rgb = photometric == tifffile.PHOTOMETRIC.RGB or "S" in axes
    if ij is not None and desc:
        c = int(desc.get("channels", 1))
        z = int(desc.get("slices", 1))
        t = int(desc.get("frames", 1))
        n = int(desc.get("images", 1))
        if rgb:
            data = data.reshape((-1,) + data.shape[-3:])
            if data.shape[-1] != 3 and data.shape[1] == 3:     # planar
                data = np.moveaxis(data, 1, -1)
            data = data[..., :3]
            if data.shape[0] == c * z * t:
                data = data.reshape((t, z, c) + data.shape[1:])
            else:
                data = data.reshape((1, data.shape[0], 1) + data.shape[1:])
            doc = ImageDoc.from_array(data, "TZCYXS", title)
        else:
            data = data.reshape((-1,) + data.shape[-2:])
            if c * z * t != data.shape[0]:
                c, z, t = 1, data.shape[0], 1
            if n != data.shape[0] and data.shape[0] != c * z * t:
                c, z, t = 1, data.shape[0], 1
            data = data.reshape((t, z, c) + data.shape[1:])
            doc = ImageDoc.from_array(data, "TZCYX", title)
        _apply_imagej_metadata(doc, page, desc, ij)
    else:
        doc = _generic_doc(data, axes, rgb, title)
        res = page.tags.get(X_RESOLUTION)
        unit_tag = page.tags.get(RESOLUTION_UNIT)
        if res is not None and unit_tag is not None and int(unit_tag.value) in (2, 3):
            num, den = res.value
            if num and den:
                xres = num / den
                if not (int(unit_tag.value) == 2 and xres == 72.0):
                    doc.calibration.pixel_width = 1.0 / xres
                    yres = page.tags.get(Y_RESOLUTION)
                    doc.calibration.pixel_height = (yres.value[1] / yres.value[0]) if \
                        yres is not None and yres.value[0] else doc.calibration.pixel_width
                    doc.calibration.unit = "inch" if int(unit_tag.value) == 2 else "cm"
        if desc_tag is not None and not str(desc_tag.value).startswith("ImageJ"):
            text = str(desc_tag.value).strip()
            if text and len(text) < 8192 and not text.startswith("<?xml"):
                doc.info = text
        if doc.type in (GRAY16, GRAY32):
            doc.reset_display_range()
    if photometric == tifffile.PHOTOMETRIC.PALETTE and page.colormap is not None and \
            not doc.is_rgb:
        cmap = np.asarray(page.colormap)
        if cmap.max() > 255:
            cmap = cmap >> 8
        lut = LUT.from_array(cmap.T.astype(np.uint8))
        doc.luts = [lut.copy() for _ in range(doc.n_channels)] if doc.n_channels == 1 \
            else doc.luts
        if doc.type == GRAY8 and not (lut.is_grayscale() or lut.is_inverted()) and \
                str(_desc_get(desc, "8bitcolor")).lower() == "true":
            doc.type = COLOR_256
    elif photometric == tifffile.PHOTOMETRIC.MINISWHITE and not doc.is_rgb and \
            doc.n_channels == 1:
        doc.luts = [LUT.grays().inverted()]
    elif page.colormap is not None and not doc.is_rgb and doc.n_channels == 1:
        cmap = np.asarray(page.colormap)
        if cmap.max() > 255:
            cmap = cmap >> 8
        doc.luts = [LUT.from_array(cmap.T.astype(np.uint8))]
    doc.position = (0, 0, 0)
    return doc


def _desc_get(desc, key, default=None):
    return desc.get(key, default) if desc else default


def _generic_doc(data: np.ndarray, axes: str, rgb: bool, title: str) -> ImageDoc:
    """Non-ImageJ TIFF series -> ImageDoc (unknown axes become Z, ImageJ's stack)."""
    a = data
    ax = list(axes)
    if rgb and "S" in ax:
        si = ax.index("S")
        if a.shape[si] >= 3:
            a = np.take(a, [0, 1, 2], axis=si)
        else:
            rgb = False
            a = np.take(a, 0, axis=si)
            ax.pop(si)
    mapped = []
    for i, letter in enumerate(ax):
        if letter in "TZCYXS" and letter not in mapped:
            mapped.append(letter)
        else:
            mapped.append("?")
    # merge unknown/duplicate axes into Z (or fold into an existing one)
    order = [i for i, m in enumerate(mapped) if m == "?"]
    if order:
        keep = [i for i, m in enumerate(mapped) if m != "?"]
        a = np.transpose(a, order + keep)
        extra = int(np.prod([a.shape[i] for i in range(len(order))]))
        a = a.reshape((extra,) + a.shape[len(order):])
        letters = [mapped[i] for i in keep]
        if "Z" in letters:
            zi = letters.index("Z") + 1
            a = np.moveaxis(a, 0, zi - 1)
            shp = list(a.shape)
            a = a.reshape(shp[:zi - 1] + [shp[zi - 1] * shp[zi]] + shp[zi + 1:])
        else:
            letters = ["Z"] + letters
        mapped = letters
    if a.dtype == np.uint8 and rgb:
        pass
    elif rgb:
        # 16-bit RGB (48-bit) -> 3-channel composite like ImageJ
        si = mapped.index("S")
        a = np.moveaxis(a, si, 0)
        mapped = ["C"] + [m for m in mapped if m != "S"]
        if mapped.count("C") > 1:
            mapped = ["C"] + [m if m != "C" else "Z" for m in mapped[1:]]
    if a.dtype == np.bool_:
        a = a.astype(np.uint8) * 255
    doc = ImageDoc.from_array(a, "".join(mapped), title)
    return doc


def _apply_imagej_metadata(doc: ImageDoc, page, desc: dict, ij: dict) -> None:
    from rsstatchem.core.imaging.io.roi_codec import decode_roi
    from rsstatchem.core.imaging.roi import Overlay

    cal = doc.calibration
    unit = desc.get("unit")
    res = page.tags.get(X_RESOLUTION)
    if unit:
        cal.unit = _unescape(unit)
        if res is not None:
            num, den = res.value
            if num:
                cal.pixel_width = den / num
                yres = page.tags.get(Y_RESOLUTION)
                cal.pixel_height = (yres.value[1] / yres.value[0]) if yres is not None and \
                    yres.value[0] else cal.pixel_width
    if desc.get("yunit"):
        cal.y_unit = _unescape(desc["yunit"])
    if desc.get("zunit"):
        cal.z_unit = _unescape(desc["zunit"])
    cal.pixel_depth = abs(float(desc["spacing"])) if desc.get("spacing") else 1.0
    if cal.unit == "um":
        cal.unit = "µm"
    if desc.get("finterval"):
        cal.frame_interval = float(desc["finterval"])
    if desc.get("tunit"):
        cal.time_unit = _unescape(desc["tunit"])
    if desc.get("fps"):
        cal.fps = float(desc["fps"])
    if desc.get("loop"):
        cal.info["loop"] = desc["loop"].lower()
    for key, attr in (("xorigin", "x_origin"), ("yorigin", "y_origin"),
                      ("zorigin", "z_origin")):
        if desc.get(key):
            setattr(cal, attr, float(desc[key]))
    if desc.get("inverty") == "true":
        cal.invert_y = True
    if desc.get("info"):
        cal.info["info"] = _unescape(desc["info"])
    if desc.get("cf") is not None and doc.bit_depth != 32:
        cf = int(desc["cf"])
        coeffs = []
        i = 0
        while f"c{i}" in desc:
            coeffs.append(float(desc[f"c{i}"]))
            i += 1
        cal.function = cf
        cal.coefficients = tuple(coeffs)
        cal.value_unit = _unescape(desc.get("vunit", "Gray Value"))
    elif desc.get("vunit"):
        cal.value_unit = _unescape(desc["vunit"])
    if doc.type == GRAY16 and cal.function == 0 and tuple(cal.coefficients) == (0.0, 1.0):
        cal.function, cal.coefficients = NONE, ()
    if doc.stack_size > 1:   # FileOpener: setOpenAsHyperStack only for stacks
        doc.hyperstack = desc.get("hyperstack") == "true" or sum(d > 1 for d in doc.dims) >= 2
    mode = desc.get("mode")
    if doc.is_composite:
        doc.composite_mode = mode if mode in ("composite", "color", "grayscale") else "color"
    if "min" in desc and "max" in desc and not doc.is_rgb:
        rng = (float(desc["min"]), float(desc["max"]))
        doc.display_ranges = [rng] * doc.n_channels
    elif doc.type in (GRAY16, GRAY32):
        doc.reset_display_range()
    if ij.get("Ranges") is not None and doc.is_composite:
        r = list(ij["Ranges"])
        if len(r) >= 2 * doc.n_channels:
            doc.display_ranges = [(float(r[2 * i]), float(r[2 * i + 1]))
                                  for i in range(doc.n_channels)]
    luts = ij.get("LUTs")
    if luts and not doc.is_rgb:
        ll = [LUT.from_array(np.asarray(lt).reshape(3, 256).T) for lt in luts]
        if len(ll) == doc.n_channels:
            doc.luts = ll
    labels = ij.get("Labels")
    if labels:
        if isinstance(labels, str):
            labels = [labels]
        labels = [lb if lb else None for lb in list(labels)[:doc.stack_size]]
        doc.slice_labels = labels + [None] * (doc.stack_size - len(labels))
    if ij.get("Info"):
        doc.info = str(ij["Info"])
    props = ij.get("Properties")
    if props:
        doc.properties = {str(k): str(v) for k, v in dict(props).items()}
    if ij.get("ROI"):
        try:
            doc.roi = decode_roi(bytes(ij["ROI"]))
        except (ValueError, struct.error):
            doc.roi = None
    overlays = ij.get("Overlays")
    if overlays:
        if isinstance(overlays, (bytes, bytearray)):
            overlays = [overlays]
        ov = Overlay()
        for k, blob in enumerate(overlays):
            try:
                roi = decode_roi(bytes(blob))
            except (ValueError, struct.error):
                continue
            if k == 0:
                opts = int(roi.properties.get("_overlay_options", 0))
                ov.show_labels = bool(opts & 8)
                ov.draw_names = bool(opts & 16)
                ov.draw_backgrounds = bool(opts & 32)
                if roi.properties.get("_overlay_font_size"):
                    ov.label_font_size = int(roi.properties["_overlay_font_size"])
            ov.add(roi)
        if len(ov):
            doc.overlay = ov

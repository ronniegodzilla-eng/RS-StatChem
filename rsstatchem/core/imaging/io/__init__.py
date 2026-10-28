"""Image file I/O of the Image Analysis module (owned by WP-A).

Contract (Phase 0): ``open_image(path) -> ImageDoc`` dispatching on the extension through
:data:`READERS` (extension -> ``reader(path) -> ImageDoc``) and :func:`register_reader`, so
other packages can plug in formats. WP-A adds:

* :func:`open_file` - what File > Open does with any file: images (TIFF incl. ImageJ hyperstacks
  and LZW, GIF/PNG/JPEG/BMP/PNM, DICOM, FITS, AVI, ImageJ ZIP), ``.roi`` selections, RoiSet
  ``.zip`` archives, ``.lut`` tables, results/tables (``.csv``/``.xls``/``.tsv``), text files,
  macros and ``.rsij`` projects; the type is recognised by magic bytes first (ImageJ
  ``Opener.getFileType``), then by extension;
* :func:`save_image` / :data:`WRITERS` - File > Save As formats under ImageJ's ``saveAs`` format
  names ("tiff", "gif", "jpeg", "png", "bmp", "pgm", "fits", "raw", "text image", "zip",
  "avi", "lut");
* modules: :mod:`.tiff` (ImageJ TIFF writer port, byte-identical to ImageJ), :mod:`.lzw`,
  :mod:`.roi_codec`, :mod:`.pillow_io`, :mod:`.dicom`, :mod:`.fits`, :mod:`.raw`,
  :mod:`.text_image`, :mod:`.lut_io`, :mod:`.sequence`, :mod:`.avi`, :mod:`.zip_io`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from rsstatchem.core.imaging.image import ImageDoc
from rsstatchem.core.imaging.io.pillow_io import read_pillow
from rsstatchem.core.imaging.io.tiff import read_tiff

READERS: dict[str, Callable[[str], ImageDoc]] = {}


def register_reader(extensions, reader: Callable[[str], ImageDoc]) -> None:
    for ext in ([extensions] if isinstance(extensions, str) else extensions):
        READERS[ext.lower()] = reader


def _read_dicom(path):
    from rsstatchem.core.imaging.io.dicom import read_dicom
    return read_dicom(path)


def _read_fits(path):
    from rsstatchem.core.imaging.io.fits import read_fits
    return read_fits(path)


def _read_avi(path):
    from rsstatchem.core.imaging.io.avi import read_avi
    return read_avi(path)


def _read_zip(path):
    from rsstatchem.core.imaging.io.zip_io import read_zip_image
    return read_zip_image(path)


def _read_text_image(path):
    from rsstatchem.core.imaging.io.text_image import read_text_image
    return read_text_image(path)


register_reader((".gif", ".png", ".jpg", ".jpeg", ".bmp", ".pgm", ".pbm", ".ppm", ".pnm"),
                read_pillow)
register_reader((".tif", ".tiff"), read_tiff)
register_reader((".dcm", ".dicom", ".ima"), _read_dicom)
register_reader((".fits", ".fit", ".fts"), _read_fits)
register_reader(".avi", _read_avi)
register_reader(".zip", _read_zip)


def sniff(path) -> str | None:
    """File type from the first bytes (ImageJ ``Opener.getFileType``), or None."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(132)
    except OSError:
        return None
    if head[:4] in (b"II*\0", b"MM\0*") or head[:4] in (b"II+\0", b"MM\0+"):
        return ".tif"
    if len(head) >= 132 and head[128:132] == b"DICM":
        return ".dcm"
    if head[:6] == b"SIMPLE":
        return ".fits"
    if head[:4] == b"GIF8":
        return ".gif"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if head[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if head[:2] == b"BM" and len(head) > 14:
        return ".bmp"
    if head[:4] == b"PK\x03\x04":
        return ".zip"
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
        return ".avi"
    if head[:4] == b"Iout":
        return ".roi"
    if head[:2] in (b"P1", b"P2", b"P3", b"P4", b"P5", b"P6") and len(head) > 2 and \
            head[2:3] in (b"\n", b"\r", b" ", b"\t"):
        return ".pgm"
    return None


def open_image(path) -> ImageDoc:
    """Open an image file (contract). Dispatches on the magic bytes, then the extension."""
    path = str(path)
    ext = Path(path).suffix.lower()
    kind = sniff(path)
    reader = READERS.get(kind) if kind is not None else None
    if reader is None:
        reader = READERS.get(ext)
    if reader is None:
        if ext == ".txt":
            reader = _read_text_image
        else:
            raise ValueError(f"File is not in a supported format:\n{path}")
    doc = reader(path)
    doc.path = path
    doc.changed = False
    doc.forget_undo()
    return doc


@dataclass
class Opened:
    """Result of :func:`open_file`: ``kind`` is "image", "roi", "rois", "lut", "table",
    "text", "macro" or "project"; ``value`` the object (ImageDoc, Roi, list of Roi, LUT,
    ResultsTable, str or the project dict)."""
    kind: str
    value: Any
    path: str = ""


TABLE_EXTENSIONS = (".csv", ".xls", ".tsv")
TEXT_EXTENSIONS = (".txt", ".java", ".js", ".py", ".bsh", ".md", ".ijm", ".macro")


def open_file(path) -> Opened:
    """What File > Open does with ``path`` (ImageJ ``Opener.open``)."""
    p = Path(path)
    ext = p.suffix.lower()
    kind = sniff(p)
    if ext == ".rsij":
        from rsstatchem.core.imaging.project import read_project
        return Opened("project", read_project(p), str(p))
    if kind == ".roi" or ext == ".roi":
        from rsstatchem.core.imaging.io.roi_codec import read_roi
        return Opened("roi", read_roi(p), str(p))
    if kind == ".zip" or ext == ".zip":
        from rsstatchem.core.imaging.io.roi_codec import is_roi_zip, read_roi_zip
        if is_roi_zip(p):
            return Opened("rois", read_roi_zip(p), str(p))
        return Opened("image", open_image(p), str(p))
    if ext == ".lut" and kind is None:
        from rsstatchem.core.imaging.io.lut_io import read_lut
        return Opened("lut", read_lut(p), str(p))
    if ext in TABLE_EXTENSIONS and kind is None:
        from rsstatchem.core.imaging.results import ResultsTable
        table = ResultsTable.read(p)
        table.title = p.name if p.stem != "Results" else "Results"
        return Opened("table", table, str(p))
    if ext in (".ijm", ".macro"):
        return Opened("macro", p.read_text(errors="replace"), str(p))
    if ext in TEXT_EXTENSIONS and kind is None:
        text = p.read_text(errors="replace")
        if ext == ".txt" and _looks_like_text_image(text):
            from rsstatchem.core.imaging.io.text_image import read_text_image
            try:
                doc = read_text_image(p)
                doc.path = str(p)
                return Opened("image", doc, str(p))
            except ValueError:
                pass
        return Opened("text", text, str(p))
    return Opened("image", open_image(p), str(p))


def _looks_like_text_image(text: str) -> bool:
    import re
    lines = [ln for ln in text.splitlines()[:5] if ln.strip()]
    if not lines:
        return False
    num = re.compile(r"^[\s,]*([-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?|NaN)([\s,]+"
                     r"([-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?|NaN))*[\s,]*$")
    body = lines[1:] if len(lines) > 1 else lines
    return all(num.match(ln) for ln in body)


# ---------------------------------------------------------------------------------- writing
def _w_tiff(doc, path, opts):
    from rsstatchem.core.imaging.io.tiff import write_tiff
    write_tiff(doc, path, little_endian=bool(opts.get("intel")))


def _w_zip(doc, path, opts):
    from rsstatchem.core.imaging.io.zip_io import write_zip
    write_zip(doc, path, little_endian=bool(opts.get("intel")))


def _w_gif(doc, path, opts):
    from rsstatchem.core.imaging.io.pillow_io import write_gif
    write_gif(doc, path)


def _w_jpeg(doc, path, opts):
    from rsstatchem.core.imaging.io.pillow_io import write_jpeg
    write_jpeg(doc, path, int(opts.get("quality", 85)))


def _w_png(doc, path, opts):
    from rsstatchem.core.imaging.io.pillow_io import write_png
    write_png(doc, path)


def _w_bmp(doc, path, opts):
    from rsstatchem.core.imaging.io.pillow_io import write_bmp
    write_bmp(doc, path)


def _w_pgm(doc, path, opts):
    from rsstatchem.core.imaging.io.pillow_io import write_pnm
    write_pnm(doc, path)


def _w_fits(doc, path, opts):
    from rsstatchem.core.imaging.io.fits import write_fits
    write_fits(doc, path)


def _w_raw(doc, path, opts):
    from rsstatchem.core.imaging.io.raw import write_raw
    write_raw(doc, path, little_endian=bool(opts.get("intel")))


def _w_text(doc, path, opts):
    from rsstatchem.core.imaging.io.text_image import write_text_image
    write_text_image(doc, path, int(opts.get("precision", 3)))


def _w_avi(doc, path, opts):
    from rsstatchem.core.imaging.io.avi import write_avi
    write_avi(doc, path, opts.get("compression", "Uncompressed"),
              float(opts.get("fps", doc.calibration.fps or 7.0)), int(opts.get("quality", 90)))


def _w_lut(doc, path, opts):
    from rsstatchem.core.imaging.io.lut_io import write_lut
    write_lut(doc.lut, path)


WRITERS: dict[str, tuple[str, Callable]] = {
    "tiff": (".tif", _w_tiff), "zip": (".zip", _w_zip), "gif": (".gif", _w_gif),
    "jpeg": (".jpg", _w_jpeg), "png": (".png", _w_png), "bmp": (".bmp", _w_bmp),
    "pgm": (".pgm", _w_pgm), "fits": (".fits", _w_fits), "raw": (".raw", _w_raw),
    "text image": (".txt", _w_text), "avi": (".avi", _w_avi), "lut": (".lut", _w_lut),
}
_ALIASES = {"tif": "tiff", "jpg": "jpeg", "text": "text image", "raw data": "raw",
            "fit": "fits", "pnm": "pgm", "txt": "text image"}


def format_key(fmt: str | None, path=None) -> str:
    """Normalise an ImageJ ``saveAs`` format name ("Tiff", "Jpeg", "Text Image"...) or derive
    it from the extension of ``path``."""
    if fmt:
        key = fmt.strip().lower().rstrip(". ")
        key = _ALIASES.get(key, key)
        if key in WRITERS:
            return key
    if path is not None:
        ext = Path(path).suffix.lower().lstrip(".")
        key = _ALIASES.get(ext, ext)
        if key in WRITERS:
            return key
        if ext == "tiff":
            return "tiff"
    raise ValueError(f"Unsupported save() or saveAs() file format: \"{fmt}\"")


def _as_8bit(doc: ImageDoc) -> ImageDoc:
    """8-bit copy scaled with the display range ("8-bit TIFF" of Process > Batch)."""
    import numpy as np

    from rsstatchem.core.imaging.lut import render, to_8bit
    if doc.bit_depth == 8 and not doc.is_rgb:
        return doc
    if doc.is_rgb:
        planes = np.stack([render(doc, *doc.position_of(i)).mean(axis=2)
                           for i in range(doc.stack_size)]).astype(np.uint8)
    else:
        planes = np.stack([to_8bit(doc.plane(*doc.position_of(i)),
                                   *doc.display_ranges[doc.position_of(i)[0]])
                           for i in range(doc.stack_size)])
    out = ImageDoc.from_array(planes, "ZYX", doc.title)
    out.calibration = doc.calibration.copy()
    return out


def save_image(doc: ImageDoc, path, fmt: str | None = None, **options) -> Path:
    """Write ``doc`` as ``fmt`` (ImageJ name or extension). Options: ``intel`` (byte order),
    ``quality`` (JPEG), ``precision`` (text image), ``compression``/``fps`` (AVI)."""
    if fmt and fmt.strip().lower() == "8-bit tiff":
        doc = _as_8bit(doc)
        fmt = "tiff"
    key = format_key(fmt, path)
    ext, writer = WRITERS[key]
    path = Path(path)
    if not path.suffix and key != "raw":
        path = path.with_suffix(ext)
    writer(doc, path, options)
    return path


def lut_image(lut, title: str = "LUT") -> ImageDoc:
    """The 256x32 ramp ImageJ shows when a LUT is opened without an image."""
    import numpy as np

    ramp = np.tile(np.arange(256, dtype=np.uint8), (32, 1))
    doc = ImageDoc.from_array(ramp, "YX", title)
    doc.luts = [lut.copy()]
    if not (lut.is_grayscale() or lut.is_inverted()):
        from rsstatchem.core.imaging.image import COLOR_256
        doc.type = COLOR_256
    return doc


def open_in_session(session, path) -> Opened:
    """File > Open into a session: images are added, a ``.roi`` becomes the active image's
    selection, RoiSet archives fill the ROI Manager, LUTs apply to the active image (or open
    as a LUT ramp), tables are shown, projects are loaded; text files and macros are announced
    with the "text_opened" / "macro_opened" events (the UI opens an editor window)."""
    opened = open_file(path)
    kind, value = opened.kind, opened.value
    if kind == "image":
        session.add_image(value)
    elif kind == "roi":
        im = session.active
        if im is None:
            raise ValueError("There are no images open")
        im.roi = value
        session.image_updated(im)
    elif kind == "rois":
        session.roi_manager.extend(value)
        session.emit("roi_manager_updated", session.roi_manager)
    elif kind == "lut":
        im = session.active
        if im is None or im.is_rgb:
            session.add_image(lut_image(value, Path(path).name))
        else:
            luts = list(im.luts)
            luts[im.position[0]] = value.copy()
            im.luts = luts
            session.image_updated(im)
    elif kind == "table":
        session.show_table(value)
    elif kind == "project":
        from rsstatchem.core.imaging.project import load_project
        load_project(session, path)
    elif kind == "text":
        session.emit("text_opened", (Path(path).name, value, str(path)))
    elif kind == "macro":
        session.emit("macro_opened", (Path(path).name, value, str(path)))
    return opened


IMAGE_TITLE_FORMATS = ("tiff", "zip", "gif", "jpeg", "png", "bmp", "pgm", "fits", "text image",
                       "raw")


def save_in_session(session, fmt: str, path, image: ImageDoc | None = None) -> Path:
    """``saveAs(format, path)`` of ImageJ: image formats save the active image (renamed to the
    file name, marked unchanged); "results"/"measurements", "selection", "xy coordinates",
    "lut" and "text" (the Log) save those objects."""
    key = fmt.strip().lower().rstrip(". ")
    path = Path(path)
    im = image if image is not None else session.active
    opts = session.options
    if key in ("results", "measurements", "table"):
        if not path.suffix:
            path = path.with_suffix(opts.results_extension or ".csv")
        session.results.save(path, headers=opts.save_column_headers,
                             row_numbers=opts.save_row_numbers if session.results is not None
                             else None)
        return path
    if key == "text":
        if not path.suffix:
            path = path.with_suffix(".txt")
        path.write_text("\n".join(session.log_lines) + "\n", encoding="utf-8")
        return path
    if im is None:
        raise ValueError("There are no images open")
    if key in ("selection", "roi"):
        from rsstatchem.core.imaging.io.roi_codec import write_roi
        if im.roi is None:
            raise ValueError("Selection required")
        if not path.suffix:
            path = path.with_suffix(".roi")
        write_roi(im.roi, path)
        return path
    if key in ("xy coordinates", "xy"):
        from rsstatchem.core.imaging.io.text_image import xy_text
        if im.roi is None:
            raise ValueError("Selection required")
        if not path.suffix:
            path = path.with_suffix(".txt")
        path.write_text(xy_text(im.roi, im.calibration, im.height))
        return path
    k = format_key(fmt, path)
    if k == "lut":
        from rsstatchem.core.imaging.io.lut_io import write_lut
        if not path.suffix:
            path = path.with_suffix(".lut")
        write_lut(im.lut, path)
        return path
    out = save_image(im, path, k, intel=opts.intel_byte_order, quality=opts.jpeg_quality,
                     precision=opts.decimal_places)
    if k in IMAGE_TITLE_FORMATS:
        title = out.name
        if k == "zip":
            title = out.stem + ".tif"
        im.title = session.unique_title(title, exclude=im)
        im.path = str(out)
        if k in ("tiff", "zip"):
            im.changed = False
        session.image_updated(im)
    return out


__all__ = ["READERS", "register_reader", "open_image", "open_file", "Opened", "save_image",
           "WRITERS", "format_key", "sniff", "read_pillow", "read_tiff", "open_in_session",
           "save_in_session", "lut_image"]

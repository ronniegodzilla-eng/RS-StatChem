"""ImageJ ZIP files: File > Save As > ZIP... stores the image as one ImageJ TIFF inside a
zip archive; File > Open of a .zip opens the first TIFF (or DICOM) in it, or loads the ROI
Manager when the archive holds .roi files (``ij/io/Opener.openZip``, ``ij/plugin/filter/
Writer``; ImageJ 1.54, public domain).
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

from rsstatchem.core.imaging.image import ImageDoc


def write_zip(doc: ImageDoc, path, little_endian: bool = False) -> None:
    from rsstatchem.core.imaging.io.tiff import tiff_bytes

    name = Path(doc.title).stem or "image"
    with zipfile.ZipFile(str(path), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name + ".tif", tiff_bytes(doc, little_endian))


def zip_entries(path) -> list[str]:
    with zipfile.ZipFile(str(path)) as zf:
        return [n for n in zf.namelist() if not n.endswith("/")]


def read_zip_image(path) -> ImageDoc:
    from rsstatchem.core.imaging.io.tiff import read_tiff_bytes

    with zipfile.ZipFile(str(path)) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        for n in names:
            low = n.lower()
            if low.endswith((".tif", ".tiff")):
                doc = read_tiff_bytes(zf.read(n), Path(n).name)
                doc.title = Path(path).stem + ".tif" if len(names) == 1 else Path(n).name
                return doc
        for n in names:
            if n.lower().endswith(".dcm"):
                import tempfile

                from rsstatchem.core.imaging.io.dicom import read_dicom
                with tempfile.TemporaryDirectory() as tmp:
                    p = Path(tmp) / Path(n).name
                    p.write_bytes(zf.read(n))
                    return read_dicom(p)
        for n in names:
            low = n.lower()
            if low.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                from PIL import Image

                from rsstatchem.core.imaging.io.pillow_io import read_pil_image
                with Image.open(io.BytesIO(zf.read(n))) as im:
                    return read_pil_image(im, Path(n).name)
    raise ValueError(f"This ZIP archive does not appear to contain a\nTIFF or DICOM file:\n{path}")

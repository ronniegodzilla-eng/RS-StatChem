"""FITS reading and writing (no astropy).

Ported from ImageJ 1.54 ``ij/plugin/FITS_Reader.java`` and ``FITS_Writer.java`` (public domain,
NIH; FITS_Writer after J. West, K. A. Collins, F. V. Hessman): 2880-byte blocks, BITPIX 8, 16,
32, -32 and -64, NAXIS3 stacks, the image flipped vertically (FITS has its origin at the bottom
left), the header kept as the Info property, CDELT1/2/3 and CTYPE1 as spatial calibration.
16-bit data are signed in FITS: ImageJ reads them as 16-bit signed (raw = v + 32768) and drops
the calibration again when BZERO = 32768 and BSCALE = 1 (unsigned data); the writer stores
unsigned 16-bit images that way.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from rsstatchem.core.imaging.calibration import NONE
from rsstatchem.core.imaging.image import GRAY8, GRAY16, ImageDoc
from rsstatchem.core.imaging.io.tiff import java_double

BLOCK = 2880


def read_fits(path) -> ImageDoc:
    raw = Path(path).read_bytes()
    if path and str(path).lower().endswith(".gz"):
        import gzip
        raw = gzip.decompress(raw)
    if not raw.startswith(b"SIMPLE"):
        raise ValueError("This does not appear to be a FITS file.")
    cards: dict[str, str] = {}
    info_lines = []
    count = 0
    pos = 0
    while pos + 80 <= len(raw):
        line = raw[pos:pos + 80].decode("latin-1")
        pos += 80
        count += 1
        info_lines.append(line)
        idx = line.find("=")
        if idx >= 0:
            cidx = line.find("/", idx)
            if cidx < 0:
                cidx = len(line)
            key, value = line[:idx].strip(), line[idx + 1:cidx].strip()
        else:
            key, value = line.strip(), ""
        if key == "END":
            break
        cards.setdefault(key, value)
        if count > 360 and "NAXIS1" not in cards:
            raise ValueError("This does not appear to be a FITS file.")
    bitpix = int(cards.get("BITPIX", "0"))
    width = int(cards.get("NAXIS1", "0"))
    height = int(cards.get("NAXIS2", "0"))
    n = int(cards.get("NAXIS3", "1")) if int(cards.get("NAXIS", "2")) >= 3 else 1
    if width <= 0 or height <= 0:
        raise ValueError("This does not appear to be a FITS file.")
    dtypes = {8: ">u1", 16: ">i2", 32: ">i4", -32: ">f4", -64: ">f8"}
    if bitpix not in dtypes:
        raise ValueError("BITPIX must be 8, 16, 32, -32 (float) or -64 (double).")
    offset = BLOCK + BLOCK * ((count * 80 - 1) // BLOCK)
    dt = np.dtype(dtypes[bitpix])
    data = np.frombuffer(raw, dtype=dt, count=width * height * n, offset=offset)
    data = data.reshape(n, height, width)[:, ::-1, :]          # flip vertically
    if bitpix == 16:
        doc = ImageDoc.from_array(data.astype(np.int16), "ZYX", Path(path).name)
        bscale = float(cards.get("BSCALE", "0") or 0)
        bzero = float(cards.get("BZERO", "0") or 0)
        if bscale == 1.0 and bzero == 32768.0:
            doc.calibration.function, doc.calibration.coefficients = NONE, ()
    elif bitpix == 8:
        doc = ImageDoc.from_array(data.astype(np.uint8), "ZYX", Path(path).name)
    else:
        doc = ImageDoc.from_array(data.astype(np.float32), "ZYX", Path(path).name)
    cal = doc.calibration
    pw = float(cards["CDELT1"]) if "CDELT1" in cards else 1.0
    ph = float(cards["CDELT2"]) if "CDELT2" in cards else 1.0
    pd = float(cards["CDELT3"]) if "CDELT3" in cards else 1.0
    unit = cards.get("CTYPE1")
    if unit is not None:
        unit = unit.strip("'").strip()
        if pw == 1.0 and pd == 1.0:
            unit = "pixel"
        cal.pixel_width, cal.pixel_height, cal.pixel_depth, cal.unit = pw, ph, pd, unit or "pixel"
    doc.info = "\n".join(info_lines) + "\n"
    doc.reset_display_range()
    return doc


def _card(title: str, value: str, comment: str) -> str:
    card = [" "] * 80
    for i, ch in enumerate(title[:80]):
        card[i] = ch
    card[8] = "="
    for i, ch in enumerate(value):
        if 10 + i < 80:
            card[10 + i] = ch
    card[31] = "/"
    card[32] = " "
    for i, ch in enumerate(comment):
        if 33 + i < 80:
            card[33 + i] = ch
    return "".join(card)


def fits_bytes(doc: ImageDoc) -> bytes:
    """The current plane as FITS, byte-identical to ImageJ's FITS_Writer."""
    if doc.is_rgb:
        raise ValueError("RGB images are not supported")
    plane = doc.plane()
    cal = doc.calibration
    w, h = doc.width, doc.height
    bzero, bscale = 0.0, 1.0
    if doc.bit_depth == 8:
        nbytes = 1
        if cal.function != NONE and cal.coefficients:
            bzero = cal.coefficients[0]
            if cal.coefficients[1] != 0:
                bscale = cal.coefficients[1]
    elif doc.type == GRAY16:
        nbytes = 2
        if cal.function == NONE or not cal.coefficients:
            bzero, bscale = 32768.0, 1.0
        else:
            if cal.coefficients[1] != 0:
                bscale = cal.coefficients[1]
            bzero = cal.coefficients[0] + 32768.0 * bscale
    else:
        nbytes = 4
    cards = []
    bitpix = {1: "                   8", 2: "                  16", 4: "                 -32"}
    cards.append(_card("SIMPLE", "                   T", "Created by ImageJ FITS_Writer"))
    cards.append(_card("BITPIX", bitpix[nbytes], "number of bits per data pixel"))
    cards.append(_card("NAXIS", "                   2", "number of data axes"))
    cards.append(_card("NAXIS1", f"{w:>20d}", "length of data axis 1"))
    cards.append(_card("NAXIS2", f"{h:>20d}", "length of data axis 2"))
    if bzero != 0 or bscale != 1.0:
        cards.append(_card("BZERO", java_double(bzero), "data range offset"))
        cards.append(_card("BSCALE", java_double(bscale), "scaling factor"))
    for line in _original_header(doc):
        if not line.startswith(("SIMPLE", "BITPIX", "NAXIS", "BZERO", "BSCALE", "END")) and \
                len(line.strip()) > 1:
            cards.append(line[:80].ljust(80))
    header = "".join(cards) + "END"
    header += " " * (BLOCK - ((len(cards) * 80 + 3) % BLOCK))
    flipped = plane[::-1]
    if nbytes == 1:
        data = np.ascontiguousarray(flipped, np.uint8).tobytes()
    elif nbytes == 2:
        data = np.ascontiguousarray(flipped.astype(np.uint16) ^ 0x8000, ">u2").tobytes()
    else:
        data = np.ascontiguousarray(flipped, ">f4").tobytes()
    filler = b"\0" * (BLOCK - (nbytes * w * h) % BLOCK)
    return header.encode("latin-1") + data + filler


def _original_header(doc: ImageDoc) -> list[str]:
    content = doc.slice_label() if doc.stack_size > 1 else None
    content = content or doc.info
    if not content:
        return []
    lines = content.split("\n")
    try:
        start = next(i for i, ln in enumerate(lines) if ln.startswith("SIMPLE"))
    except StopIteration:
        return []
    end = next((i for i in range(start + 1, len(lines))
                if lines[i].strip() == "END" or lines[i].strip().startswith("END ")), None)
    if end is None:
        return []
    return lines[start:end + 1]


def write_fits(doc: ImageDoc, path) -> None:
    Path(path).write_bytes(fits_bytes(doc))


__all__ = ["read_fits", "write_fits", "fits_bytes", "GRAY8"]

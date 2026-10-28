"""Text images and XY coordinate files.

* :func:`read_text_image` - File > Import > Text Image (``ij/plugin/TextReader.java``): numbers
  separated by whitespace, commas or tabs, every non-empty line the same length, opened as a
  32-bit image; a first row that is all non-numeric is taken as a header and dropped.
* :func:`text_image_bytes` - File > Save As > Text Image (``ij/io/TextEncoder.java``): tab
  separated; uncalibrated 8/16-bit as integers, otherwise calibrated values with the
  measurement precision (``IJ.d2s``, 3 decimals by default).
* :func:`read_xy` / :func:`xy_text` - File > Import > XY Coordinates (``XY_Reader``) and
  File > Save As > XY Coordinates (``filter/XYWriter``) of a selection.

Ported from ImageJ 1.54 (public domain, NIH).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from rsstatchem.core.imaging.image import GRAY8, GRAY16, ImageDoc

_SPLIT = re.compile(r"[\x00-\x2a\x2c\x80-\xff]+")


def _tokens(line: str) -> list[str]:
    return [t for t in _SPLIT.split(line) if t]


def _parse(tok: str) -> float:
    try:
        return float(tok.replace("Infinity", "inf"))
    except ValueError:
        return float("nan")


def read_text_image(path) -> ImageDoc:
    text = Path(path).read_text(errors="replace")
    rows = [_tokens(line) for line in text.splitlines()]
    rows = [r for r in rows if r]
    if not rows:
        raise ValueError("No numbers found in the text file")
    width = len(rows[0])
    for i, r in enumerate(rows[1:], start=2):
        if len(r) != width:
            raise ValueError(f"Line {i} is not the same length as the first line.")
    values = np.array([[_parse(t) for t in r] for r in rows], dtype=np.float32)
    first = rows[0][0]
    if np.all(np.isnan(values[0])) and first not in ("NaN", "nan") and len(rows) > 1:
        values = values[1:]
    doc = ImageDoc.from_array(values, "YX", Path(path).name)
    doc.reset_display_range()
    return doc


def d2s(value: float, decimals: int) -> str:
    """ImageJ ``IJ.d2s`` (HALF_EVEN DecimalFormat; NaN/Infinity as Java prints them)."""
    v = float(value)
    if v != v:
        return "NaN"
    if v in (float("inf"), float("-inf")):
        return "Infinity" if v > 0 else "-Infinity"
    if v == 3.4028234663852886e38:
        return "3.4e38"
    if decimals < 0:
        d = min(-decimals, 9)
        mant, exp = f"{v:.{d}E}".split("E")
        return f"{mant}E{int(exp)}"
    return f"{v:.{min(decimals, 9)}f}"


def text_image_bytes(doc: ImageDoc, precision: int = 3, delimiter: str = "\t") -> bytes:
    plane = doc.plane()
    cal = doc.calibration
    if doc.is_rgb:
        plane = plane.astype(np.float64).mean(axis=2)  # ImageJ: getPixelValue of RGB
        int_data = False
    else:
        int_data = not cal.calibrated() and doc.type in (GRAY8, GRAY16)
    lines = []
    if int_data:
        for row in plane.astype(np.int64):
            lines.append(delimiter.join(map(str, row.tolist())))
    else:
        vals = cal.cvalue(plane.astype(np.float64)) if cal.calibrated() and not doc.is_rgb \
            else plane.astype(np.float64)
        for row in np.asarray(vals):
            lines.append(delimiter.join(d2s(v, precision) for v in row.tolist()))
    return ("\n".join(lines) + "\n").encode()


def write_text_image(doc: ImageDoc, path, precision: int = 3) -> None:
    Path(path).write_bytes(text_image_bytes(doc, precision))


# ---------------------------------------------------------------------------------- XY
def read_xy(path) -> tuple[np.ndarray, np.ndarray]:
    """Two-column (x y) text file -> coordinates (header lines ignored)."""
    xs, ys = [], []
    for line in Path(path).read_text(errors="replace").splitlines():
        toks = _tokens(line)
        if len(toks) < 2:
            continue
        x, y = _parse(toks[0]), _parse(toks[1])
        if x != x or y != y:
            continue
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def xy_text(roi, calibration=None, image_height: int = 0) -> str:
    """XYWriter: one "x<TAB>y" line per vertex; integer polygons as integers, others with 4
    decimals (scaled units when the image is spatially calibrated)."""
    xs, ys = roi.polygon()
    scaled = calibration is not None and calibration.scaled()
    integer = not scaled and np.all(np.mod(xs, 1) == 0) and np.all(np.mod(ys, 1) == 0)
    lines = []
    for x, y in zip(xs.tolist(), ys.tolist()):
        if scaled:
            x = calibration.get_x(x)
            y = calibration.get_y(y, image_height)
        if integer:
            lines.append(f"{int(x)}\t{int(y)}")
        else:
            lines.append(f"{d2s(x, 4)}\t{d2s(y, 4)}")
    return "\n".join(lines) + "\n"

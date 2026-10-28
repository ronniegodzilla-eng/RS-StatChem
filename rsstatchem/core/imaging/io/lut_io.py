"""LUT files (File > Import > LUT..., File > Save As > LUT...).

Ported from ImageJ 1.54 ``ij/plugin/LutLoader.java`` (public domain, NIH): binary 768-byte
tables (256 reds, greens, blues), 800-byte NIH Image tables (32-byte header), tables with
fewer than 256 entries (interpolated to 256), and text tables with 3 or 4 columns (an index
column first). Saving writes the 768-byte format, as ImageJ does.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from rsstatchem.core.imaging.lut import LUT


def _interpolate(r, g, b) -> LUT:
    n = len(r)
    if n == 256:
        return LUT(r, g, b)
    scale = n / 256.0
    out = []
    for ch in (r, g, b):
        ch = np.asarray(ch, float)
        idx = np.arange(256) * scale
        i1 = np.floor(idx).astype(int)
        i2 = np.minimum(i1 + 1, n - 1)
        frac = idx - i1
        out.append(np.round((1 - frac) * ch[i1] + frac * ch[i2]).astype(np.uint8))
    return LUT(*out)


def read_lut(path) -> LUT:
    data = Path(path).read_bytes()
    name = Path(path).stem
    if len(data) == 768:
        a = np.frombuffer(data, np.uint8)
        return LUT(a[:256], a[256:512], a[512:768], name)
    if len(data) == 800 and data[:4] == b"ICOL":
        a = np.frombuffer(data[32:], np.uint8)
        return LUT(a[:256], a[256:512], a[512:768], name)
    if len(data) > 768 and len(data) - 32 >= 768 and data[:4] == b"ICOL":
        a = np.frombuffer(data[32:32 + 768], np.uint8)
        return LUT(a[:256], a[256:512], a[512:768], name)
    # text LUT
    rows = []
    for line in data.decode("latin-1").splitlines():
        toks = line.replace(",", " ").replace("\t", " ").split()
        try:
            nums = [float(t) for t in toks]
        except ValueError:
            continue
        if len(nums) in (3, 4):
            rows.append(nums[-3:])
    if len(rows) >= 2:
        a = np.clip(np.array(rows), 0, 255).astype(np.uint8)
        lut = _interpolate(a[:, 0], a[:, 1], a[:, 2])
        lut.name = name
        return lut
    if len(data) < 768 and len(data) % 3 == 0 and len(data) >= 6:
        n = len(data) // 3
        a = np.frombuffer(data, np.uint8)
        lut = _interpolate(a[:n], a[n:2 * n], a[2 * n:])
        lut.name = name
        return lut
    raise ValueError("This is not an ImageJ LUT file")


def lut_bytes(lut: LUT) -> bytes:
    return lut.red.tobytes() + lut.green.tobytes() + lut.blue.tobytes()


def write_lut(lut: LUT, path) -> None:
    Path(path).write_bytes(lut_bytes(lut))

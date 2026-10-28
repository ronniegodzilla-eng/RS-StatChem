"""Thermo/Galactic SPC reader, written from the published SPC file format description
(Galactic Industries "SPC file format", GSPCIO / spc.h layout). No GPL code is used.

Supported layouts:

* New format, little-endian (``fversn = 0x4B``): 512-byte main header, optional global X
  array (``TXVALS``), one or more subfiles (``TMULTI``) each with a 32-byte subheader, Y as
  IEEE float (exponent 0x80) or scaled 16-bit (``TSPREC``) / 32-bit integers
  (``y = int · 2^(exp − 16 or 32)``), XY-XY multifiles with a separate X per subfile
  (``TXYXYS``), Z values from ``subtime`` / ``fzinc``, axis labels from ``fcatxt``
  (``TALABS``) and the log block (``key=value`` text) at ``flogoff``.
* Old format (``fversn = 0x4D``): 256-byte header with the first subheader inside it,
  32-bit integers stored with swapped 16-bit words.

The big-endian new format (``0x4C``) is read with the same layout in big-endian order.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from rsstatchem.core.analysis.importers import ImportedTable, ImportFailed, numeric_column

TSPREC, TCGRAM, TMULTI, TRANDM, TORDRD, TALABS, TXYXYS, TXVALS = \
    0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80

XTYPES = {0: ("X", ""), 1: ("Wavenumber", "cm⁻¹"), 2: ("Wavelength", "µm"),
          3: ("Wavelength", "nm"), 4: ("Time", "s"), 5: ("Time", "min"),
          6: ("Frequency", "Hz"), 7: ("Frequency", "kHz"), 8: ("Frequency", "MHz"),
          9: ("Mass", "m/z"), 10: ("Chemical Shift", "ppm"), 11: ("Time", "days"),
          12: ("Time", "years"), 13: ("Raman Shift", "cm⁻¹"), 14: ("Energy", "eV"),
          16: ("Diode Number", ""), 17: ("Channel", ""), 18: ("Angle", "°"),
          19: ("Temperature", "°F"), 20: ("Temperature", "°C"),
          21: ("Temperature", "K"), 22: ("Data Points", ""), 23: ("Time", "ms"),
          24: ("Time", "µs"), 25: ("Time", "ns"), 26: ("Frequency", "GHz"),
          27: ("Length", "cm"), 28: ("Length", "m"), 29: ("Length", "mm"),
          30: ("Time", "h"), 255: ("Interferogram", "")}
YTYPES = {0: "Intensity", 1: "Interferogram", 2: "Absorbance", 3: "Kubelka-Munk",
          4: "Counts", 5: "Volts", 6: "Degrees", 7: "Milliamps", 8: "Millimeters",
          9: "Millivolts", 10: "Log(1/R)", 11: "Percent", 12: "Intensity",
          13: "Relative Intensity", 14: "Energy", 16: "Decibel", 19: "Temperature (F)",
          20: "Temperature (C)", 21: "Temperature (K)", 22: "Index of Refraction",
          23: "Extinction Coeff.", 24: "Real", 25: "Imaginary", 26: "Complex",
          128: "Transmittance", 129: "Reflectance", 130: "Single Beam", 131: "Emission"}

_MAIN_NEW = "<BBBbIddIBBBBI9s9sH32s130s30sIIBBHf48sfIfB187s"
_SUB = "<BbHfffIIf4s"


def _text(b: bytes) -> str:
    return b.split(b"\x00")[0].decode("latin-1", "replace").strip()


def _y_values(raw: bytes, n: int, exp: int, flags: int, endian: str, old: bool) \
        -> tuple[np.ndarray, int]:
    """Decode ``n`` Y values; returns (values, bytes used)."""
    if exp == -128:
        size = 4 * n
        return np.frombuffer(raw[:size], dtype=endian + "f4").astype(float), size
    if flags & TSPREC:
        size = 2 * n
        ints = np.frombuffer(raw[:size], dtype=endian + "i2").astype(float)
        return ints * 2.0 ** (exp - 16), size
    size = 4 * n
    if old:   # old format: the two 16-bit words of each 32-bit integer are swapped
        words = np.frombuffer(raw[:size], dtype="<u2").reshape(-1, 2).astype(np.uint32)
        ints = ((words[:, 0] << 16) | words[:, 1]).astype(np.int32).astype(float)
    else:
        ints = np.frombuffer(raw[:size], dtype=endian + "i4").astype(float)
    return ints * 2.0 ** (exp - 32), size


def _log_block(data: bytes, off: int) -> dict:
    if not off or off + 64 > len(data):
        return {}
    sizd, sizm, txto = struct.unpack_from("<III", data, off)
    start = off + txto
    if txto <= 0 or start >= len(data):
        return {}
    text = data[start:off + sizd if sizd else len(data)].split(b"\x00")[0]
    out = {}
    for line in text.decode("latin-1", "replace").replace("\r", "\n").split("\n"):
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def read_bytes(data: bytes) -> dict:
    """Parse SPC bytes: ``{"x": [array per subfile], "y": [...], "z": [...], "header",
    "flags", "xlabel", "xunits", "ylabel", "zlabel", "comment", "log"}``."""
    if len(data) < 256:
        raise ImportFailed("The file is too short to be an SPC file.")
    flags, version = data[0], data[1]
    if version in (0x4B, 0x4C):
        endian = "<" if version == 0x4B else ">"
        fmt = endian + _MAIN_NEW[1:]
        if len(data) < 512:
            raise ImportFailed("Truncated SPC header.")
        h = struct.unpack_from(fmt, data, 0)
        (_, _, fexper, fexp, fnpts, ffirst, flast, fnsub, fxtype, fytype, fztype, _post,
         _date, _res, _src, _peak, _spare, fcmnt, fcatxt, flogoff, _mods, _procs, _level,
         _sampin, _factor, _method, fzinc, _wplanes, _winc, _wtype, _resv) = h
        pos = 512
        header = {"version": hex(version), "experiment": fexper, "npoints": fnpts,
                  "first": ffirst, "last": flast, "nsub": fnsub, "xtype": fxtype,
                  "ytype": fytype, "ztype": fztype, "zinc": fzinc}
        old = False
    elif version == 0x4D:
        endian = "<"
        oexp, onpts, ofirst, olast, oxtype, oytype = struct.unpack_from("<hfffBB", data, 2)
        fexp = int(oexp)
        fnpts, ffirst, flast = int(onpts), float(ofirst), float(olast)
        fxtype, fytype, fztype = oxtype, oytype, 0
        fcmnt = data[64:194]
        fcatxt = data[194:224]
        fnsub = 1 if not flags & TMULTI else 0     # unknown: read until the data ends
        flogoff, fzinc = 0, 0.0
        pos = 224                                  # first subheader is inside the header
        header = {"version": "0x4d", "npoints": fnpts, "first": ffirst, "last": flast,
                  "xtype": fxtype, "ytype": fytype}
        old = True
        if fexp == -128 or fexp == 128:
            fexp = -128
    else:
        raise ImportFailed(f"Unknown SPC version byte 0x{version:02X} (expected 0x4B, 0x4C "
                           "or 0x4D).")
    xs_all, ys_all, zs = [], [], []
    global_x = None
    if flags & TXVALS and not flags & TXYXYS:
        size = 4 * fnpts
        global_x = np.frombuffer(data[pos:pos + size], dtype=endian + "f4").astype(float)
        pos += size
    k = 0
    subfmt = endian + _SUB[1:]
    while pos + 32 <= len(data) and (k < fnsub or (old and flags & TMULTI)):
        (_sflg, subexp, _idx, subtime, subnext, _noise, subnpts, _scan, _wlevel,
         _r) = struct.unpack_from(subfmt, data, pos)
        pos += 32
        n = int(subnpts) if flags & TXYXYS and subnpts else int(fnpts)
        if flags & TXYXYS:
            size = 4 * n
            x = np.frombuffer(data[pos:pos + size], dtype=endian + "f4").astype(float)
            pos += size
        elif global_x is not None:
            x = global_x
        else:
            x = np.linspace(ffirst, flast, n) if n > 1 else np.array([ffirst], float)
        # float everywhere when fexp = 0x80; a multifile scales each subfile by its own
        # subexp, a single file by fexp
        exp = -128 if fexp == -128 else (subexp if flags & TMULTI and not old else fexp)
        need = n * (2 if flags & TSPREC and exp != -128 else 4)
        if pos + need > len(data):
            if k == 0:
                raise ImportFailed("The SPC data block is truncated.")
            break
        y, used = _y_values(data[pos:], n, exp, flags, endian, old)
        pos += used
        xs_all.append(x)
        ys_all.append(y)
        zs.append(float(subtime) if subtime or not fzinc else k * float(fzinc))
        del subnext
        k += 1
    if not ys_all:
        raise ImportFailed("The SPC file contains no subfiles.")
    xl, xu = XTYPES.get(fxtype, ("X", ""))
    yl = YTYPES.get(fytype, "Intensity")
    zl = XTYPES.get(fztype, ("Z", ""))[0] if fztype else "Z"
    if flags & TALABS:
        labels = [t.decode("latin-1", "replace").strip()
                  for t in bytes(fcatxt).split(b"\x00")]
        if len(labels) > 0 and labels[0]:
            xl, xu = labels[0], ""
        if len(labels) > 1 and labels[1]:
            yl = labels[1]
        if len(labels) > 2 and labels[2]:
            zl = labels[2]
    return {"x": xs_all, "y": ys_all, "z": zs, "header": header, "flags": flags,
            "xlabel": xl, "xunits": xu, "ylabel": yl, "zlabel": zl,
            "comment": _text(bytes(fcmnt)), "log": _log_block(data, flogoff)}


def read(path: str | Path) -> dict:
    return read_bytes(Path(path).read_bytes())


def tables(spc: dict, name: str = "SPC") -> list[ImportedTable]:
    """One sheet: X, Y1, Y2… (shared X) or X1, Y1, X2, Y2… (XY-XY multifile); the Z value
    of each subfile goes into the Comments row."""
    cols = []
    shared = all(len(x) == len(spc["x"][0]) and np.allclose(x, spc["x"][0])
                 for x in spc["x"])
    multi = len(spc["y"]) > 1
    for i, (x, y, z) in enumerate(zip(spc["x"], spc["y"], spc["z"])):
        if i == 0 or not shared:
            cols.append(numeric_column(spc["xlabel"], x, spc["xunits"], "X", spc["comment"]
                                       if i == 0 else ""))
        cols.append(numeric_column(spc["ylabel"], y, "", "Y",
                                   f"{spc['zlabel']} = {z:g}" if multi else spc["comment"]))
    meta = {"source_format": "SPC", "header": spc["header"], "log": spc["log"],
            "comment": spc["comment"], "z": spc["z"]}
    return [ImportedTable(name, cols, meta)]

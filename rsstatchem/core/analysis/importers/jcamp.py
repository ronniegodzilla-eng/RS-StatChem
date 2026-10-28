"""JCAMP-DX reader and writer (IR, Raman, UV-Vis, NMR, MS), written from the IUPAC
specifications (McDonald & Wilks, Appl. Spectrosc. 42 (1988) 151; JCAMP-DX 4.24, 5.01 and the
NMR / MS extensions). No third-party code is used.

Supported:

* Labelled data records ``##LABEL=value`` (labels compared without spaces, ``-``, ``/``,
  ``_``; ``$$`` comments removed), several blocks per file and ``##BLOCKS=`` link blocks.
* ``##XYDATA=(X++(Y..Y))`` in every ASDF form: AFFN (free format, with exponents), PAC
  (``+``/``-`` as separators), SQZ (``@A-Ia-i``), DIF (``%J-Rj-r``) and DUP (``S-Zs``),
  including the DIF Y-check value repeated at the start of the next line; X is rebuilt from
  ``FIRSTX``/``LASTX``/``NPOINTS`` and the factors ``XFACTOR``/``YFACTOR`` are applied.
* ``##XYPOINTS=(XY..XY)`` and ``##PEAK TABLE=(XY..XY)`` / ``(XYW..XYW)``.
* ``##NTUPLES=`` with ``##PAGE=`` / ``##DATA TABLE=(X++(R..R)), XYDATA`` (NMR real and
  imaginary pages) and ``(XY..XY), PEAKS`` (MS) — per-variable ``FACTOR``, ``FIRST``, ``LAST``,
  ``VAR_DIM`` and ``UNITS``.

:func:`write` produces JCAMP-DX 4.24 files in AFFN or DIF/DUP compression (round-trip
tests and export).
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from rsstatchem.core.analysis.importers import ImportedTable, ImportFailed, numeric_column

SQZ = {"@": 0, **{c: i + 1 for i, c in enumerate("ABCDEFGHI")},
       **{c: -(i + 1) for i, c in enumerate("abcdefghi")}}
DIF = {"%": 0, **{c: i + 1 for i, c in enumerate("JKLMNOPQR")},
       **{c: -(i + 1) for i, c in enumerate("jklmnopqr")}}
DUP = {**{c: i + 1 for i, c in enumerate("STUVWXYZ")}, "s": 9}
_AFFN = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_LDR = re.compile(r"^##([^=]*)=(.*)$")


def norm_label(label: str) -> str:
    """Label as compared by the specification: upper case, no spaces, ``-``, ``/``, ``_``."""
    return re.sub(r"[\s\-/_]", "", label).upper()


# ====================================================================== ASDF decoding
def decode_line(line: str) -> tuple[list[float], bool]:
    """Numbers on one data line (abscissa first) and whether the line ended in DIF form."""
    s = line.split("$$")[0].strip()
    if not s:
        return [], False
    rest = _AFFN.sub(" ", s)
    if not rest.replace(",", " ").replace(";", " ").strip():   # AFFN / PAC
        return [float(t) for t in _AFFN.findall(s)], False
    out: list[float] = []
    kind, text = None, ""
    last_kind, last_dif = None, 0.0

    def flush():
        nonlocal kind, text, last_kind, last_dif
        if kind is None:
            return
        if kind == "dup":
            count = int(text)
            for _ in range(count - 1):
                if last_kind == "dif":
                    out.append(out[-1] + last_dif)
                else:
                    out.append(out[-1])
        elif kind == "dif":
            if not out:
                raise ValueError("A DIF value cannot start a line.")
            last_dif = float(text)
            out.append(out[-1] + last_dif)
            last_kind = "dif"
        else:
            out.append(float(text))
            last_kind = "abs"
        kind, text = None, ""

    for ch in s:
        if ch in SQZ:
            flush()
            kind, text = "abs", str(SQZ[ch])
        elif ch in DIF:
            flush()
            kind, text = "dif", str(DIF[ch])
        elif ch in DUP:
            flush()
            kind, text = "dup", str(DUP[ch])
        elif ch in "+-":
            flush()
            kind, text = "abs", ch
        elif ch.isdigit() or ch == ".":
            if kind is None:
                kind, text = "abs", ""
            text += ch
        elif ch in " \t,;":
            flush()
        else:
            raise ValueError(f"Unexpected character {ch!r} in JCAMP data line: {line!r}")
    flush()
    # a line "ends in DIF form" when its last value came from a DIF (or DUP of a DIF)
    return out, last_kind == "dif"


def decode_xydata(lines: list[str]) -> tuple[list[float], list[float]]:
    """Lines of an ``(X++(Y..Y))`` table → ``([(line abscissa, index of its first y)], ys)``
    with the DIF Y-check duplicates removed."""
    xs: list[tuple[float, int]] = []
    ys: list[float] = []
    prev_dif = False
    for ln in lines:
        vals, is_dif = decode_line(ln)
        if not vals:
            continue
        x, yv = vals[0], vals[1:]
        if prev_dif and ys and yv:
            # Y-check: the first ordinate repeats the last one of the previous line
            if not math.isclose(yv[0], ys[-1], rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"JCAMP Y-check failed at X = {x}: {yv[0]} ≠ {ys[-1]}")
            ys.pop()
        xs.append((x, len(ys)))
        ys.extend(yv)
        prev_dif = is_dif
    return xs, ys


def _numbers(text: str) -> list[float]:
    return [float(t) for t in _AFFN.findall(text.split("$$")[0])]


# ====================================================================== parsing
def _blocks(text: str) -> list[list[tuple[str, str, list[str]]]]:
    """Split a file into blocks of ``(raw label, value, continuation lines)``."""
    blocks: list[list] = []
    cur: list | None = None
    depth = 0
    stack: list[list] = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        m = _LDR.match(raw.strip())
        if m:
            label, value = m.group(1), m.group(2)
            key = norm_label(label)
            if key == "TITLE":
                if cur is not None:
                    stack.append(cur)
                cur = []
                depth += 1
            if cur is None:
                cur = []
            cur.append((label.strip(), value.split("$$")[0].strip(), []))
            if key == "END":
                blocks.append(cur)
                cur = stack.pop() if stack else None
                depth = max(0, depth - 1)
        elif cur is not None and cur:
            cur[-1][2].append(raw)
    if cur:
        blocks.append(cur)
    return blocks


def _float(ldr: dict, key: str, default=None):
    v = ldr.get(key)
    if v is None or v == "":
        return default
    try:
        return float(v.split(",")[0])
    except ValueError:
        nums = _numbers(v)
        return nums[0] if nums else default


def _xy_block(ldr: dict, data_lines: list[str]) -> tuple[np.ndarray, np.ndarray]:
    xf = _float(ldr, "XFACTOR", 1.0) or 1.0
    yf = _float(ldr, "YFACTOR", 1.0) or 1.0
    line_x, ys = decode_xydata(data_lines)
    y = np.asarray(ys, float) * yf
    n = int(_float(ldr, "NPOINTS", len(y)) or len(y))
    first, last = _float(ldr, "FIRSTX"), _float(ldr, "LASTX")
    if n != len(y):
        raise ValueError(f"NPOINTS = {n} but {len(y)} ordinates were decoded.")
    if first is not None and last is not None and n > 1:
        x = np.linspace(first, last, n)
    elif line_x:
        if len(line_x) > 1:
            (x0, i0), (x1, i1) = line_x[0], line_x[1]
            dx = (x1 - x0) * xf / max(i1 - i0, 1)
        else:
            dx = _float(ldr, "DELTAX", 1.0)
        x = line_x[0][0] * xf + dx * np.arange(n)
    else:
        x = np.arange(n, dtype=float)
    return x, y


def _pairs(data_lines: list[str], width: int = 2) -> np.ndarray:
    nums: list[float] = []
    for ln in data_lines:
        nums.extend(_numbers(ln))
    k = len(nums) // width
    return np.asarray(nums[:k * width], float).reshape(k, width)


def _table_kind(value: str) -> str:
    v = value.replace(" ", "").upper()
    if "X++(Y..Y)" in v or "X++(R..R)" in v or "X++(I..I)" in v or "++(" in v:
        return "xydata"
    if "XYW" in v:
        return "xyw"
    return "xy"


def _parse_block(entries) -> dict:
    ldr: dict[str, str] = {}
    labels: dict[str, str] = {}
    out = {"ldr": ldr, "labels": labels, "kind": None, "x": None, "ys": [], "pages": []}
    nt: dict[str, list[str]] = {}
    page = None
    for label, value, cont in entries:
        key = norm_label(label)
        labels[key] = label
        if key in ("XYDATA", "XYPOINTS", "PEAKTABLE", "DATATABLE"):
            kind = _table_kind(value)
            if key == "DATATABLE":
                sym = re.search(r"\(\s*X\+\+\(\s*(\w)", value.replace(" ", ""), re.I)
                page_ldr = dict(ldr)
                if nt:
                    names = [s.strip() for s in nt.get("SYMBOL", [])]
                    idx = {s.upper(): i for i, s in enumerate(names)}

                    def var(field, s, default=None):
                        vals = nt.get(field, [])
                        i = idx.get(s.upper(), -1)
                        return vals[i].strip() if 0 <= i < len(vals) and vals[i].strip() \
                            else default
                    ysym = sym.group(1) if sym else "Y"
                    page_ldr.update({k: v for k, v in (
                        ("FIRSTX", var("FIRST", "X")), ("LASTX", var("LAST", "X")),
                        ("XFACTOR", var("FACTOR", "X")), ("YFACTOR", var("FACTOR", ysym)),
                        ("NPOINTS", var("VARDIM", ysym) or var("VARDIM", "X")),
                        ("XUNITS", var("UNITS", "X")), ("YUNITS", var("UNITS", ysym)))
                        if v is not None})
                    name = var("VARNAME", ysym, ysym) or ysym
                else:
                    name = sym.group(1) if sym else "Y"
                if kind == "xydata":
                    x, y = _xy_block(page_ldr, cont)
                else:
                    arr = _pairs(cont, 3 if kind == "xyw" else 2)
                    xf = _float(page_ldr, "XFACTOR", 1.0) or 1.0
                    yf = _float(page_ldr, "YFACTOR", 1.0) or 1.0
                    x, y = arr[:, 0] * xf, arr[:, 1] * yf
                out["pages"].append({"page": page, "name": name, "x": x, "y": y,
                                     "units": page_ldr.get("YUNITS", ""),
                                     "xunits": page_ldr.get("XUNITS", "")})
                out["kind"] = "ntuples"
                continue
            if kind == "xydata":
                x, y = _xy_block(ldr, cont)
                out.update(kind="xydata", x=x, ys=[("Y", y)])
            else:
                arr = _pairs(cont, 3 if kind == "xyw" else 2)
                xf = _float(ldr, "XFACTOR", 1.0) or 1.0
                yf = _float(ldr, "YFACTOR", 1.0) or 1.0
                ys = [("Y", arr[:, 1] * yf)]
                if kind == "xyw":
                    ys.append(("W", arr[:, 2]))
                out.update(kind="peaks" if key == "PEAKTABLE" else "xypoints",
                           x=arr[:, 0] * xf, ys=ys)
            continue
        if key == "PAGE":
            page = value
            continue
        full = " ".join([value] + [c.split("$$")[0].strip() for c in cont]).strip()
        if key in ("VARNAME", "SYMBOL", "VARTYPE", "VARFORM", "VARDIM", "UNITS", "FIRST",
                   "LAST", "MIN", "MAX", "FACTOR"):
            nt[key] = [t for t in full.split(",")]
        ldr[key] = full
    return out


def read_text(text: str) -> list[dict]:
    """Parse JCAMP-DX text into data blocks: ``{"ldr", "kind", "x", "ys", "pages"}``
    (``kind`` xydata / xypoints / peaks / ntuples; link blocks without data are skipped)."""
    blocks = []
    for entries in _blocks(text):
        try:
            b = _parse_block(entries)
        except ValueError as exc:
            title = next((v for k, v, _ in entries if norm_label(k) == "TITLE"), "")
            raise ImportFailed(f"JCAMP-DX block “{title}”: {exc}") from None
        if b["kind"] is not None:
            blocks.append(b)
    if not blocks:
        raise ImportFailed("No JCAMP-DX data (XYDATA, XYPOINTS, PEAK TABLE or NTUPLES) "
                           "was found.")
    return blocks


def read(path: str | Path) -> list[dict]:
    data = Path(path).read_bytes()
    for enc in ("utf-8", "latin-1"):
        try:
            return read_text(data.decode(enc))
        except UnicodeDecodeError:
            continue
    return read_text(data.decode("latin-1", "replace"))


# ====================================================================== tables
_XUNITS = {"1/CM": ("Wavenumber", "cm⁻¹"), "CM-1": ("Wavenumber", "cm⁻¹"),
           "NANOMETERS": ("Wavelength", "nm"), "NM": ("Wavelength", "nm"),
           "MICROMETERS": ("Wavelength", "µm"), "HZ": ("Frequency", "Hz"),
           "PPM": ("Chemical Shift", "ppm"), "SECONDS": ("Time", "s"),
           "M/Z": ("m/z", ""), "MINUTES": ("Time", "min")}


def _axis(units: str, default: str) -> tuple[str, str]:
    u = (units or "").strip()
    return _XUNITS.get(u.upper(), (default, u if u.upper() != "ARBITRARY UNITS" else ""))


def tables(blocks: list[dict], name: str = "JCAMP") -> list[ImportedTable]:
    out = []
    for k, b in enumerate(blocks):
        ldr = b["ldr"]
        title = ldr.get("TITLE", "") or name
        meta = {"source_format": "JCAMP-DX", "ldr": dict(ldr), "kind": b["kind"]}
        sheet = (title[:28] or name) if len(blocks) > 1 else name
        if b["kind"] == "ntuples":
            pages = b["pages"]
            xl, xu = _axis(pages[0]["xunits"] if pages else ldr.get("XUNITS", ""), "X")
            cols = [numeric_column(xl, pages[0]["x"], xu, "X", title)]
            for pg in pages:
                if len(pg["x"]) != len(pages[0]["x"]) or not np.allclose(pg["x"],
                                                                         pages[0]["x"]):
                    cols.append(numeric_column(xl, pg["x"], xu, "X", pg["page"] or ""))
                cols.append(numeric_column(pg["name"], pg["y"], pg["units"], "Y",
                                           pg["page"] or ""))
        else:
            xl, xu = _axis(ldr.get("XUNITS", ""), "X")
            yname = (ldr.get("YUNITS", "") or "Y").title()
            cols = [numeric_column(xl, b["x"], xu, "X", title)]
            for nm, y in b["ys"]:
                cols.append(numeric_column(yname if nm == "Y" else nm, y, "", "Y", title))
        out.append(ImportedTable(sheet if k == 0 or len(blocks) > 1 else f"{name}{k + 1}",
                                 cols, meta))
    return out


# ====================================================================== writer
def _sqz(v: int) -> str:
    s = str(abs(v))
    first = "@" if v == 0 else ("ABCDEFGHI"[int(s[0]) - 1] if v > 0
                                else "abcdefghi"[int(s[0]) - 1])
    return first + s[1:]


def _dif(v: int) -> str:
    s = str(abs(v))
    first = "%" if v == 0 else ("JKLMNOPQR"[int(s[0]) - 1] if v > 0
                                else "jklmnopqr"[int(s[0]) - 1])
    return first + s[1:]


def _dup(count: int) -> str:
    s = str(count)
    return ("STUVWXYZs"[int(s[0]) - 1]) + s[1:]


def encode_difdup(values: list[int]) -> str:
    """One line of DIF/DUP-compressed integers (first value in SQZ form)."""
    if not values:
        return ""
    tokens = [_sqz(values[0])]
    difs = [b - a for a, b in zip(values, values[1:])]
    i = 0
    while i < len(difs):
        j = i
        while j + 1 < len(difs) and difs[j + 1] == difs[i]:
            j += 1
        tokens.append(_dif(difs[i]))
        if j > i:
            tokens.append(_dup(j - i + 1))
        i = j + 1
    return "".join(tokens)


def write(path: str | Path, x, y, title: str = "Spectrum", xunits: str = "1/CM",
          yunits: str = "ABSORBANCE", compression: str = "difdup", per_line: int = 10,
          extra: dict | None = None) -> Path:
    """Write evenly spaced (x, y) as a JCAMP-DX 4.24 ``XYDATA`` block."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(y)
    if n < 2 or len(x) != n:
        raise ValueError("x and y must have the same length (≥ 2).")
    dx = (x[-1] - x[0]) / (n - 1)
    big = float(np.max(np.abs(y))) or 1.0
    yfactor = 10 ** math.floor(math.log10(big / 1e7)) if compression != "affn" else 1.0
    xfactor = 1.0     # X written in AFFN on each line (X check), Y compressed
    lines = [f"##TITLE={title}", "##JCAMP-DX=4.24", "##DATA TYPE=SPECTRUM",
             f"##XUNITS={xunits}", f"##YUNITS={yunits}", f"##XFACTOR={xfactor:.12g}",
             f"##YFACTOR={yfactor:.12g}", f"##FIRSTX={x[0]:.12g}", f"##LASTX={x[-1]:.12g}",
             f"##DELTAX={dx:.12g}", f"##NPOINTS={n}", f"##FIRSTY={y[0]:.12g}",
             f"##MAXY={y.max():.12g}", f"##MINY={y.min():.12g}"]
    for k, v in (extra or {}).items():
        lines.append(f"##{k}={v}")
    lines.append("##XYDATA=(X++(Y..Y))")
    if compression == "affn":
        for i in range(0, n, per_line):
            seg = y[i:i + per_line]
            lines.append(f"{x[i] / xfactor:.10g} " + " ".join(f"{v:.10g}" for v in seg))
    else:
        ints = [int(round(v / yfactor)) for v in y]
        i = 0
        while i < n:
            seg = ints[i:i + per_line + 1]       # + Y-check (first value of the next line)
            lines.append(f"{x[i]:.10g}" + encode_difdup(seg))
            i += per_line
            if i == n - 1:                       # last point alone: already the Y-check
                break
    lines.append("##END=")
    p = Path(path)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p

"""JCAMP-DX spectra (``.jdx`` / ``.dx``): reading measured NMR spectra to overlay on a
prediction, and writing a predicted spectrum.

Follows the JCAMP-DX standards for spectra (R. S. McDonald, P. A. Wilks, Appl. Spectrosc. 42
(1988) 151) and NMR (A. N. Davies, P. Lampen, Appl. Spectrosc. 47 (1993) 1093):

- labelled data records ``##LABEL= value`` (labels compared without spaces, dashes, slashes
  and underscores, as the standard prescribes);
- ``##XYDATA= (X++(Y..Y))`` in AFFN (plain numbers) or the compressed ASDF forms
  (SQZ ``@A-Ia-i``, DIF ``%J-Rj-r``, DUP ``S-Zs``, with the DIF Y-check value at line starts);
- ``##PEAK TABLE= (XY..XY)`` / ``##XYPOINTS= (XY..XY)`` (peak lists);
- NMR NTUPLES files (``##DATA TABLE= (X++(R..R))``): the first (real) page;
- X in HZ is converted to ppm with ``##.OBSERVE FREQUENCY`` (MHz); ``##$OFFSET`` / first X
  give the ppm origin when present.

Compound files (``##BLOCKS``) use the first block that holds data. No Qt here.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

__all__ = ["JcampSpectrum", "JcampError", "read_jcamp", "parse_jcamp", "write_jcamp",
           "decode_asdf_line", "encode_difdup"]


class JcampError(ValueError):
    pass


@dataclass
class JcampSpectrum:
    x: np.ndarray                     # ppm (or the file's unit when it cannot be converted)
    y: np.ndarray
    title: str = ""
    nucleus: str = ""                 # "1H", "13C" ... ('' when unknown)
    frequency: float | None = None    # MHz
    x_unit: str = "PPM"
    kind: str = "continuous"          # "continuous" or "peaks"
    solvent: str = ""
    labels: dict = field(default_factory=dict)


def _norm(label: str) -> str:
    return re.sub(r"[\s\-/_]", "", label).upper()


def _records(text: str) -> list[tuple[str, str]]:
    """[(normalised label, raw value incl. continuation lines)]; $$ comments removed."""
    out: list[list[str]] = []
    for raw in text.splitlines():
        line = raw.split("$$", 1)[0].rstrip()
        if line.startswith("##"):
            label, _, value = line[2:].partition("=")
            out.append([_norm(label), value.strip()])
        elif out and line.strip():
            out[-1][1] += "\n" + line
    return [(k, v) for k, v in out]


_SQZ = {c: (i + 1) for i, c in enumerate("ABCDEFGHI")} | \
    {c: -(i + 1) for i, c in enumerate("abcdefghi")} | {"@": 0}
_DIF = {c: (i + 1) for i, c in enumerate("JKLMNOPQR")} | \
    {c: -(i + 1) for i, c in enumerate("jklmnopqr")} | {"%": 0}
_DUP = {c: (i + 1) for i, c in enumerate("STUVWXYZ")} | {"s": 9}


def _tokens(line: str) -> list[tuple[str, str]]:
    """[(kind, text)] with kind in 'affn', 'sqz', 'dif', 'dup', 'miss'."""
    toks: list[tuple[str, str]] = []
    i, n = 0, len(line)
    while i < n:
        c = line[i]
        if c in " \t,;":
            i += 1
            continue
        if c == "?":
            toks.append(("miss", "?"))
            i += 1
            continue
        if c in _SQZ or c in _DIF or c in _DUP:
            kind = "sqz" if c in _SQZ else "dif" if c in _DIF else "dup"
            j = i + 1
            while j < n and (line[j].isdigit() or line[j] == "."):
                j += 1
            toks.append((kind, line[i:j]))
            i = j
            continue
        if c in "+-." or c.isdigit():
            j = i + 1
            while j < n and (line[j].isdigit() or line[j] in ".eE" or
                             (line[j] in "+-" and line[j - 1] in "eE")):
                j += 1
            toks.append(("affn", line[i:j]))
            i = j
            continue
        raise JcampError(f"unexpected character {c!r} in data line")
    return toks


def _num(kind: str, text: str) -> float:
    if kind == "affn":
        return float(text)
    table = _SQZ if kind == "sqz" else _DIF if kind == "dif" else _DUP
    lead = table[text[0]]
    rest = text[1:]
    if kind == "dup":
        return float(int(str(lead) + rest))
    mag = float(str(abs(lead)) + rest) if rest else float(abs(lead))
    return -mag if lead < 0 else mag


def decode_asdf_line(line: str) -> tuple[float, list[float], bool]:
    """(x, [y values], ended_in_dif) of one (X++(Y..Y)) data line."""
    toks = _tokens(line)
    if not toks:
        raise JcampError("empty data line")
    x = _num(*toks[0]) if toks[0][0] in ("affn", "sqz") else math.nan
    ys: list[float] = []
    last_kind = ""
    last_dif = 0.0
    for kind, text in toks[1:]:
        if kind == "miss":
            ys.append(math.nan)
            last_kind = "affn"
            continue
        if kind == "dup":
            reps = int(_num(kind, text)) - 1
            for _ in range(reps):
                if last_kind == "dif":
                    ys.append(ys[-1] + last_dif)
                else:
                    ys.append(ys[-1])
            continue
        if kind == "dif":
            if not ys:
                raise JcampError("DIF value without a preceding ordinate")
            last_dif = _num(kind, text)
            ys.append(ys[-1] + last_dif)
            last_kind = "dif"
        else:
            ys.append(_num(kind, text))
            last_kind = "affn"
    return x, ys, last_kind == "dif"


def _xy_table(value: str, first: float | None, last: float | None, npoints: int | None,
              xfactor: float, yfactor: float, deltax_sign: float = 1.0
              ) -> tuple[np.ndarray, np.ndarray]:
    lines = [ln for ln in value.split("\n")[1:] if ln.strip()]
    xs_line, ys = [], []
    prev_dif = False
    for ln in lines:
        x, vals, ended_dif = decode_asdf_line(ln)
        if prev_dif and ys and vals:
            vals = vals[1:]                        # Y-check value repeats the last ordinate
        xs_line.append((x, len(ys)))
        ys.extend(vals)
        prev_dif = ended_dif
    y = np.array(ys, dtype=float) * yfactor
    n = len(y)
    if npoints and n != npoints:
        if n > npoints:
            y = y[:npoints]
            n = npoints
    if first is not None and last is not None and n > 1:
        x = np.linspace(first, last, n)
    elif xs_line and n > 1:
        x0 = xs_line[0][0] * xfactor
        if len(xs_line) > 1 and xs_line[1][1] > 0:
            dx = (xs_line[1][0] - xs_line[0][0]) * xfactor / xs_line[1][1]
        else:
            dx = deltax_sign
        x = x0 + dx * np.arange(n)
    else:
        raise JcampError("cannot determine the X axis")
    return x, y


def _pairs(value: str, xfactor: float, yfactor: float) -> tuple[np.ndarray, np.ndarray]:
    nums = [float(t) for t in re.split(r"[\s,;]+", value.split("\n", 1)[1] if "\n" in value
                                       else "") if t and t != "?"]
    if len(nums) < 2:
        raise JcampError("empty peak table")
    if len(nums) % 2:
        nums = nums[:-1]
    arr = np.array(nums).reshape(-1, 2)
    return arr[:, 0] * xfactor, arr[:, 1] * yfactor


def _f(d: dict, key: str, default=None):
    v = d.get(key)
    if v is None or v == "":
        return default
    try:
        return float(str(v).split()[0].rstrip(","))
    except ValueError:
        return default


def _list(v: str | None) -> list[str]:
    return [t.strip() for t in (v or "").split(",")]


def _nucleus(text: str) -> str:
    m = re.search(r"(\d+)\s*([A-Za-z]+)", text or "")
    return f"{m.group(1)}{m.group(2).capitalize()}" if m else ""


def parse_jcamp(text: str) -> JcampSpectrum:
    recs = _records(text)
    if not recs or recs[0][0] != "TITLE":
        raise JcampError("not a JCAMP-DX file (no ##TITLE=)")
    # first block that contains data
    blocks, cur = [], []
    for k, v in recs:
        cur.append((k, v))
        if k == "END":
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)
    data_keys = ("XYDATA", "PEAKTABLE", "XYPOINTS", "DATATABLE")
    block = next((b for b in blocks if any(k in data_keys for k, _ in b)), None)
    if block is None:
        raise JcampError("the file holds no spectrum data")
    labels: dict = {}
    for k, v in block:
        labels.setdefault(k, v)
    title = labels.get("TITLE", "").strip()
    freq = _f(labels, ".OBSERVEFREQUENCY")
    nucleus = _nucleus(labels.get(".OBSERVENUCLEUS", ""))
    solvent = labels.get(".SOLVENTNAME", "").strip()
    xunit = labels.get("XUNITS", "").strip().upper()
    kind = "continuous"
    if "XYDATA" in labels:
        x, y = _xy_table(labels["XYDATA"], _f(labels, "FIRSTX"), _f(labels, "LASTX"),
                         int(_f(labels, "NPOINTS", 0)) or None, _f(labels, "XFACTOR", 1.0),
                         _f(labels, "YFACTOR", 1.0))
    elif "DATATABLE" in labels:                      # NTUPLES, first page = real part
        names = [s.upper() for s in _list(labels.get("UNITS"))]
        first = [float(s) if s else math.nan for s in _list(labels.get("FIRST"))]
        last = [float(s) if s else math.nan for s in _list(labels.get("LAST"))]
        factor = [float(s) if s else 1.0 for s in _list(labels.get("FACTOR"))]
        vardim = [int(float(s)) if s else 0 for s in _list(labels.get("VARDIM"))]
        xunit = names[0] if names else xunit
        x, y = _xy_table(labels["DATATABLE"], first[0] if first else None,
                         last[0] if last else None, vardim[0] if vardim else None,
                         factor[0] if factor else 1.0, factor[1] if len(factor) > 1 else 1.0)
    else:
        kind = "peaks"
        x, y = _pairs(labels.get("PEAKTABLE") or labels.get("XYPOINTS"),
                      _f(labels, "XFACTOR", 1.0), _f(labels, "YFACTOR", 1.0))
    if xunit in ("HZ", "HERTZ") and freq:
        offset = _f(labels, "$OFFSET")
        if offset is not None and kind == "continuous" and len(x) > 1:
            x = offset + (x - x[0]) / freq            # Bruker: $OFFSET = ppm of the first point
        else:
            x = x / freq
        xunit = "PPM"
    return JcampSpectrum(np.asarray(x, float), np.asarray(y, float), title, nucleus, freq,
                         xunit or "PPM", kind, solvent, labels)


def read_jcamp(path) -> JcampSpectrum:
    with open(path, "rb") as fh:
        raw = fh.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    return parse_jcamp(text)


# --- writing -----------------------------------------------------------------------------------

def _sqz(v: int) -> str:
    s = str(abs(v))
    if v == 0:
        return "@"
    return ("ABCDEFGHI" if v > 0 else "abcdefghi")[int(s[0]) - 1] + s[1:]


def _dif(v: int) -> str:
    s = str(abs(v))
    if v == 0:
        return "%"
    return ("JKLMNOPQR" if v > 0 else "jklmnopqr")[int(s[0]) - 1] + s[1:]


def _dup(n: int) -> str:
    s = str(n)
    return "STUVWXYZs"[int(s[0]) - 1] + s[1:]


def encode_difdup(ys: list[int]) -> str:
    """DIFDUP text of integer ordinates (first value SQZ, then differences, runs as DUP)."""
    if not ys:
        return ""
    out = [_sqz(ys[0])]
    difs = [b - a for a, b in zip(ys, ys[1:])]
    i = 0
    while i < len(difs):
        j = i
        while j + 1 < len(difs) and difs[j + 1] == difs[i]:
            j += 1
        out.append(_dif(difs[i]))
        if j > i:
            out.append(_dup(j - i + 1))
        i = j + 1
    return "".join(out)


def write_jcamp(path, x, y, title: str = "", nucleus: str = "1H", frequency: float = 300.0,
                origin: str = "RS-StatChem prediction", per_line: int = 10) -> None:
    """Write a spectrum (x in ppm, decreasing or increasing) as JCAMP-DX 5.01 with DIFDUP
    compressed (X++(Y..Y)) data."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    ymax = float(np.max(np.abs(y))) or 1.0
    yfactor = ymax / 1e6
    iy = [int(round(v / yfactor)) for v in y]
    dx = (x[-1] - x[0]) / (n - 1) if n > 1 else 1.0
    lines = [f"##TITLE= {title or 'spectrum'}", "##JCAMP-DX= 5.01", "##DATA TYPE= NMR SPECTRUM",
             "##DATA CLASS= XYDATA", f"##ORIGIN= {origin}", "##OWNER= public domain",
             f"##.OBSERVE FREQUENCY= {frequency:.4f}", f"##.OBSERVE NUCLEUS= ^{nucleus}",
             "##XUNITS= PPM", "##YUNITS= ARBITRARY UNITS", "##XFACTOR= 1.0",
             f"##YFACTOR= {yfactor:.10g}", f"##FIRSTX= {x[0]:.6f}", f"##LASTX= {x[-1]:.6f}",
             f"##DELTAX= {dx:.8f}", f"##NPOINTS= {n}", f"##FIRSTY= {y[0]:.10g}",
             "##XYDATA= (X++(Y..Y))"]
    for start in range(0, n, per_line):
        chunk = iy[start:start + per_line + 1]       # one extra value: the DIF Y-check
        if start + per_line >= n:
            chunk = iy[start:]
        lines.append(f"{x[start]:.6f} {encode_difdup(chunk)}")
    lines.append("##END=")
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")

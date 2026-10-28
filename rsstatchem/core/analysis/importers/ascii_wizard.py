"""Delimited text import with Origin's Import Wizard / ASCII options, without Qt.

* Main header lines (skipped, kept in the metadata; ``key: value`` / ``key = value`` lines are
  parsed into ``meta["params"]``) and subheader lines mapped to Long Name / Units / Comments.
  ``header_lines=-1`` detects them like Origin: the non-numeric lines above the first data
  row (1 line → Long Name, 2 → + Units, 3 or more → + Comments).
* Column delimiter (auto, comma, tab, semicolon, space, whitespace, ``|`` or any string) and
  decimal separator (auto, ``.`` or ``,``); comment lines.
* Partial import: data rows ``first_row..last_row`` (1-based), every ``read_every``-th row and a
  column list such as ``"1:3,5"``.
* Column designations by pattern (``XY``, ``XYY``, ``XYXY``… repeated like Origin's
  "Assign designations"), date/time text to numbers (Julian day or Excel serial day).
* Import filters: options saved as JSON (``*.json`` in the user folder) with a file-name
  pattern, applied automatically when a file name matches (Origin's import filters).
"""
from __future__ import annotations

import csv
import fnmatch
import json
import math
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from rsstatchem.core.analysis.importers import ImportColumn, ImportedTable, ImportFailed

DELIMITERS = {"auto": "Auto detect", ",": "Comma", "\t": "Tab", ";": "Semicolon",
              " ": "Space", "whitespace": "Whitespace (any)", "|": "Pipe"}
DECIMALS = {"auto": "Auto detect", ".": "Dot (1.5)", ",": "Comma (1,5)"}
DATES = {"keep": "Keep as text", "julian": "Julian day number", "excel": "Excel serial day"}
_KV = re.compile(r"^\s*[#;%!]*\s*([A-Za-z][\w .()/-]{0,40}?)\s*[:=]\s*(.+?)\s*$")


@dataclass
class AsciiOptions:
    delimiter: str = "auto"
    decimal: str = "auto"
    encoding: str = "auto"
    skip_lines: int = 0          # main header lines
    header_lines: int = -1       # subheader lines (-1: auto detect)
    long_name_line: int = 1      # 1-based within the subheader, 0 = none
    units_line: int = 2          # 0 = none
    comments_from: int = 3       # 0 = none
    comments_to: int = 0         # 0 = to the last subheader line
    comment_char: str = ""       # lines starting with this are ignored
    first_row: int = 1           # partial import (1-based data rows, 0 = to the end)
    last_row: int = 0
    read_every: int = 1
    columns: str = ""            # e.g. "1:3,5" (1-based), "" = all
    designations: str = "XY"     # pattern repeated over the columns ("" = all Y)
    dates: str = "keep"
    trim_empty: bool = True      # drop columns that are completely empty

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AsciiOptions":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})


# ---------------------------------------------------------------------- reading lines
def detect_encoding(data: bytes) -> str:
    if data.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    for enc in ("utf-8", "cp1252"):
        try:
            data.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


def read_lines(path: str | Path, encoding: str = "auto", limit: int | None = None) \
        -> tuple[list[str], str]:
    data = Path(path).read_bytes()
    enc = detect_encoding(data[:200_000]) if encoding in ("auto", "") else encoding
    text = data.decode(enc, errors="replace")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    return (lines[:limit] if limit else lines), enc


# ---------------------------------------------------------------------- tokens
def split_line(line: str, delimiter: str) -> list[str]:
    if delimiter in ("whitespace", " "):
        return line.split() if delimiter == "whitespace" else \
            [t for t in line.strip().split(" ") if t != ""]
    if len(delimiter) == 1:
        return [t.strip() for t in next(csv.reader([line], delimiter=delimiter))] \
            if line else []
    return [t.strip() for t in line.split(delimiter)]


_NUM = re.compile(r"^[+-]?(\d+([.,]\d*)?|[.,]\d+)([eEdD][+-]?\d+)?$")


def to_number(token: str, decimal: str = "."):
    """``token`` as float, or None when it is not a number (``1,5`` with ``decimal=","``;
    Fortran ``1.0D+03`` exponents; ``nan``/``inf``)."""
    t = token.strip().strip('"').strip()
    if not t:
        return None
    low = t.lower()
    if low in ("nan", "-nan", "inf", "+inf", "-inf", "infinity", "-infinity", "--"):
        return math.nan if "nan" in low or low == "--" else float(low.replace("infinity", "inf"))
    if decimal == ",":
        if "." in t and "," in t:          # 1.234,5 -> thousands dots
            t = t.replace(".", "")
        t = t.replace(",", ".")
    elif "," in t:
        return None
    if not _NUM.match(t):
        return None
    try:
        return float(t.replace("d", "e").replace("D", "e"))
    except ValueError:
        return None


def _is_numeric_row(tokens: list[str], decimal: str) -> bool:
    toks = [t for t in tokens if t.strip()]
    if not toks:
        return False
    nums = sum(1 for t in toks if to_number(t, decimal) is not None)
    return nums >= max(1, math.ceil(0.5 * len(toks)))


# ---------------------------------------------------------------------- detection
def detect_delimiter(lines: list[str]) -> str:
    """The delimiter that splits the sample lines into the most consistent number (> 1) of
    fields; whitespace when nothing else does."""
    sample = [ln for ln in lines if ln.strip()][-200:]
    if not sample:
        return ","
    best, best_score = "whitespace", 0.0
    for d in ("\t", ";", ",", "|"):
        counts = [len(split_line(ln, d)) for ln in sample]
        if max(counts) < 2:
            continue
        common = max(set(counts), key=counts.count)
        share = counts.count(common) / len(counts)
        score = share * (1 + math.log(common)) * (1.2 if d in "\t;" else 1.0)
        if common > 1 and score > best_score:
            best, best_score = d, score
    ws_counts = [len(ln.split()) for ln in sample]
    common = max(set(ws_counts), key=ws_counts.count)
    if common > 1:
        share = ws_counts.count(common) / len(ws_counts)
        if share * (1 + math.log(common)) > best_score * 1.05:
            return "whitespace"
    return best


def detect_decimal(lines: list[str], delimiter: str) -> str:
    text = "\n".join(lines[-300:])
    if delimiter == ",":
        return "."
    comma = len(re.findall(r"(?<![\w.,])-?\d+,\d+(?![\w.,])", text))
    dot = len(re.findall(r"(?<![\w.,])-?\d+\.\d+(?![\w.,])", text))
    return "," if comma > dot else "."


def detect_header(lines: list[str], delimiter: str, decimal: str) -> int:
    """Number of subheader lines: non-numeric lines directly above the first numeric row."""
    for i, ln in enumerate(lines):
        if ln.strip() and _is_numeric_row(split_line(ln, delimiter), decimal):
            return i
    return min(1, len(lines))


def parse_column_list(text: str, n: int) -> list[int]:
    """``"1:3,5"`` → 0-based [0, 1, 2, 4] (clipped to ``n`` columns); "" = all."""
    if not text or not text.strip():
        return list(range(n))
    out: list[int] = []
    for part in re.split(r"[,; ]+", text.strip()):
        if not part:
            continue
        if ":" in part or "-" in part[1:]:
            a, b = re.split(r"[:-]", part, maxsplit=1)
            lo = int(a) if a else 1
            hi = int(b) if b and b != "*" else n
            out.extend(range(lo - 1, min(hi, n)))
        else:
            k = int(part) - 1
            if 0 <= k < n:
                out.append(k)
    return sorted(dict.fromkeys(i for i in out if 0 <= i < n))


def designation_pattern(pattern: str, n: int) -> list[str]:
    """Origin's designation pattern repeated over ``n`` columns: ``"XY"`` → X Y Y Y…
    (only the first X unless the pattern holds several), ``"XYXY"`` → X Y X Y…"""
    p = (pattern or "").upper().replace(" ", "")
    if not p:
        return ["Y"] * n
    if p == "XY":
        return (["X"] + ["Y"] * (n - 1))[:n]
    codes = {"X": "X", "Y": "Y", "Z": "Z", "E": "yEr", "L": "L", "N": "N"}
    seq = [codes.get(ch, "Y") for ch in p]
    return [seq[i % len(seq)] for i in range(n)]


# ---------------------------------------------------------------------- parse
def _date_number(text: str, how: str):
    import pandas as pd
    try:
        ts = pd.to_datetime(text)
    except (ValueError, TypeError, OverflowError):
        return None
    if ts is pd.NaT:
        return None
    days = (ts - pd.Timestamp("1970-01-01")) / pd.Timedelta(days=1)
    return days + 2440587.5 if how == "julian" else days + 25569.0


def parse_lines(lines: list[str], opts: AsciiOptions | None = None, name: str = "Data") \
        -> ImportedTable:
    """Parse text lines with the Import Wizard options."""
    o = opts or AsciiOptions()
    if o.comment_char:
        lines = [ln for ln in lines if not ln.lstrip().startswith(o.comment_char)]
    main = lines[:max(0, o.skip_lines)]
    rest = lines[max(0, o.skip_lines):]
    while rest and not rest[0].strip():
        rest = rest[1:]
    delim = detect_delimiter(rest) if o.delimiter == "auto" else o.delimiter
    dec = detect_decimal(rest, delim) if o.decimal == "auto" else o.decimal
    nh = detect_header(rest, delim, dec) if o.header_lines < 0 else o.header_lines
    sub = [split_line(ln, delim) for ln in rest[:nh]]
    body = [ln for ln in rest[nh:] if ln.strip()]
    rows = [split_line(ln, delim) for ln in body]
    lo = max(o.first_row, 1) - 1
    hi = o.last_row if o.last_row and o.last_row > 0 else len(rows)
    rows = rows[lo:hi:max(1, int(o.read_every))]
    width = max([len(r) for r in rows] + [len(s) for s in sub] + [0])
    keep = parse_column_list(o.columns, width)

    def sub_line(k: int) -> list[str]:
        return sub[k - 1] if 0 < k <= len(sub) else []

    # Origin's auto subheaders: 1 line -> Long Name, 2 -> + Units, 3+ -> + Comments
    ln_line, un_line = o.long_name_line, o.units_line
    c_from, c_to = o.comments_from, o.comments_to
    if o.header_lines < 0:
        ln_line, un_line = (1 if nh >= 1 else 0), (2 if nh >= 2 else 0)
        c_from, c_to = (3 if nh >= 3 else 0), nh
    c_to = c_to or len(sub)
    desig = designation_pattern(o.designations, len(keep))
    cols: list[ImportColumn] = []
    for k, j in enumerate(keep):
        def at(line: list[str], jj=j) -> str:
            return line[jj].strip().strip('"') if jj < len(line) else ""
        values = []
        for r in rows:
            tok = r[j] if j < len(r) else ""
            v = to_number(tok, dec)
            if v is None:
                t = tok.strip().strip('"')
                values.append(t if t else None)
            else:
                values.append(None if math.isnan(v) else v)
        if o.dates in ("julian", "excel"):
            texts = [v for v in values if isinstance(v, str)]
            if texts:
                conv = [_date_number(t, o.dates) for t in texts]
                if all(c is not None for c in conv):
                    it = iter(conv)
                    values = [next(it) if isinstance(v, str) else v for v in values]
        comments = " ".join(t for t in (at(sub_line(i)) for i in range(c_from, c_to + 1))
                            if t) if c_from else ""
        cols.append(ImportColumn(at(sub_line(ln_line)) if ln_line else "", values,
                                 at(sub_line(un_line)) if un_line else "", comments,
                                 desig[k]))
    if o.trim_empty:
        cols = [c for c in cols if any(v is not None for v in c.values) or c.long_name]
    params = {}
    for ln in main:
        m = _KV.match(ln)
        if m:
            params[m.group(1).strip()] = m.group(2).strip()
    meta = {"header": main, "delimiter": delim, "decimal": dec, "subheader_lines": nh,
            "params": params}
    return ImportedTable(name, cols, meta)


def read(path: str | Path, opts: AsciiOptions | None = None) -> ImportedTable:
    """Read a text file (with a matching saved import filter when ``opts`` is None)."""
    p = Path(path)
    if opts is None:
        found = find_filter(p)
        opts = found[1] if found else AsciiOptions()
    try:
        lines, enc = read_lines(p, opts.encoding)
    except OSError as exc:
        raise ImportFailed(f"Cannot read {p.name}: {exc}") from None
    t = parse_lines(lines, opts, p.stem)
    t.meta.update({"source": str(p), "encoding": enc})
    if not t.columns:
        raise ImportFailed(f"{p.name} contains no data columns.")
    return t


def preview(path: str | Path, opts: AsciiOptions | None = None, n_lines: int = 400) \
        -> tuple[list[str], ImportedTable]:
    """The first raw lines and their parsed table (for the wizard's preview)."""
    o = opts or AsciiOptions()
    lines, _ = read_lines(path, o.encoding, n_lines)
    return lines, parse_lines(lines, o, Path(path).stem)


# ---------------------------------------------------------------------- import filters
def filter_dir(folder: str | Path | None = None) -> Path:
    if folder is None:
        from rsstatchem.paths import user_data_dir
        folder = user_data_dir() / "origin_import_filters"
    p = Path(folder)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name.strip()) or "filter"


def save_filter(name: str, opts: AsciiOptions, pattern: str = "",
                folder: str | Path | None = None) -> Path:
    """Save the options as import filter ``name`` (``pattern`` e.g. ``"*.dat"``)."""
    path = filter_dir(folder) / f"{_safe(name)}.json"
    path.write_text(json.dumps({"name": name, "pattern": pattern,
                                "options": opts.to_dict()}, indent=1), encoding="utf-8")
    return path


def list_filters(folder: str | Path | None = None) -> list[dict]:
    out = []
    for f in sorted(filter_dir(folder).glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append({"name": d.get("name", f.stem), "pattern": d.get("pattern", ""),
                    "options": AsciiOptions.from_dict(d.get("options")), "path": str(f)})
    return out


def load_filter(name: str, folder: str | Path | None = None) -> AsciiOptions | None:
    return next((d["options"] for d in list_filters(folder) if d["name"] == name), None)


def delete_filter(name: str, folder: str | Path | None = None) -> bool:
    for d in list_filters(folder):
        if d["name"] == name:
            Path(d["path"]).unlink(missing_ok=True)
            return True
    return False


def find_filter(path: str | Path, folder: str | Path | None = None) \
        -> tuple[str, AsciiOptions] | None:
    """The first saved filter whose file-name pattern matches ``path``."""
    try:
        filters = list_filters(folder)
    except OSError:
        return None
    name = Path(path).name
    for d in filters:
        pats = [p for p in re.split(r"[;, ]+", d["pattern"]) if p]
        if any(fnmatch.fnmatch(name.lower(), p.lower()) for p in pats):
            return d["name"], d["options"]
    return None

"""Project document model of the Graph & Analysis module (OriginLab-style), without Qt.

Mirrors Origin's hierarchy: a :class:`Project` holds a folder tree and named windows —
:class:`Workbook` (sheets: :class:`Worksheet` data sheets and :class:`ReportSheet` analysis
reports), :class:`MatrixBook` (sheets: :class:`MatrixSheet` — a Z grid with X/Y coordinate
mapping), :class:`GraphDoc` (a :class:`~rsstatchem.core.graph.model.Graph`) and :class:`Notes`.

Data are referenced with Origin's range notation ``[Book1]Sheet1!B[3:10]`` (book, sheet,
column, optional 1-based row range): :class:`RangeRef` / :func:`parse_range`.

A worksheet column (:class:`WColumn`) has Origin's header rows — short name, Long Name,
Units, Comments, F(x)= formula — and a designation (X, Y, Z, X Error, Y Error, Label,
Disregard). Cells hold ``float``, ``str`` or ``None`` (empty), like Origin's default
"Text & Numeric" columns. Cells can be *masked* (Origin's Mask): they keep their value but are
left out of plots, fits and analyses — see :meth:`Worksheet.masked_rows`,
:meth:`Worksheet.mask_array`, :meth:`Worksheet.point_mask` and :meth:`Worksheet.xy_arrays`. Everything round-trips through :func:`project_to_dict` /
:func:`project_from_dict`; :func:`save_project` writes the ``.rsop`` zip.
"""
from __future__ import annotations

import base64
import copy
import json
import math
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rsstatchem.core.graph.model import Graph, graph_from_json_dict, graph_to_json_dict

FORMAT = "rsstatchem-origin-project"
VERSION = 1
EXTENSION = ".rsop"
DEFAULT_ROWS = 32

# Designation code -> (menu text, header suffix as Origin prints it in "A(X)").
DESIGNATIONS: dict[str, tuple[str, str]] = {
    "X": ("X", "X"), "Y": ("Y", "Y"), "Z": ("Z", "Z"), "xEr": ("X Error", "xEr±"),
    "yEr": ("Y Error", "yEr±"), "L": ("Label", "L"), "N": ("Disregard", "N"),
}
LABEL_ROWS: tuple[tuple[str, str], ...] = (("long_name", "Long Name"), ("units", "Units"),
                                           ("comments", "Comments"), ("formula", "F(x)="))
_SHORT_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_RANGE_RE = re.compile(r"""^\s*(?:\[(?P<book>[^\]]+)\])?(?:(?P<sheet>[^!\[\]"]+)!)?
                           (?:col\(\s*)?(?P<col>"[^"]*"|[A-Za-z][A-Za-z0-9_]*)(?:\s*\))?
                           (?:"[^"]*")?
                           (?:\[\s*(?P<r1>\d+)\s*(?::\s*(?P<r2>\d+|\*)\s*)?\])?\s*$""",
                       re.VERBOSE)


# ====================================================================== values
def parse_cell(text) -> float | str | None:
    """Text typed into a cell -> float (decimal comma accepted), str, or None when empty."""
    if text is None:
        return None
    if isinstance(text, (int, float, np.integer, np.floating)) and not isinstance(text, bool):
        v = float(text)
        return None if math.isnan(v) else v
    s = str(text).strip()
    if s == "" or s == "--":
        return None
    candidate = s.replace(",", ".") if ("," in s and "." not in s and s.count(",") == 1) else s
    try:
        v = float(candidate)
    except ValueError:
        return s
    return None if math.isnan(v) else v


def format_cell(value) -> str:
    """How a cell value is shown in the grid (Origin's default ``.10g``-like display)."""
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "--"
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.10g}"
    return str(value)


def to_float(value) -> float:
    """A cell value as float (NaN for empty or non-numeric text)."""
    if value is None:
        return math.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass
    if text.count(",") == 1 and "." not in text:
        try:
            return float(text.replace(",", "."))
        except ValueError:
            pass
    return math.nan


# ====================================================================== ranges
@dataclass
class RangeRef:
    """A worksheet column range in Origin's notation, e.g. ``[Book1]Sheet1!B[3:10]``.
    ``r1`` / ``r2`` are 1-based and inclusive (None = from the first / to the last row)."""
    book: str
    sheet: str
    col: str
    r1: int | None = None
    r2: int | None = None

    @property
    def text(self) -> str:
        return range_text(self.book, self.sheet, self.col, self.r1, self.r2)

    @property
    def rows(self) -> tuple[int, int] | None:
        """``(r1, r2)`` or None for the whole column."""
        if self.r1 is None and self.r2 is None:
            return None
        return (self.r1 or 1, self.r2 or 0)


def rows_text(r1: int | None, r2: int | None) -> str:
    """``[3:10]`` (1-based, inclusive); ``""`` for a whole column."""
    if not r1 and not r2:
        return ""
    return f"[{r1 or 1}:{r2 if r2 else '*'}]"


def range_text(book: str, sheet: str, col: str, r1: int | None = None,
               r2: int | None = None) -> str:
    """Origin's range notation: ``[Book1]Sheet1!B`` or ``[Book1]Sheet1!B[3:10]``."""
    return f"[{book}]{sheet}!{col}{rows_text(r1, r2)}"


def parse_range(text: str, book: str = "", sheet: str = "") -> RangeRef:
    """Parse ``[Book1]Sheet1!B[3:10]``, ``Sheet1!B``, ``B[3:10]``, ``col(B)`` or
    ``"Long Name"`` (book / sheet default to the given ones). Raises ValueError."""
    m = _RANGE_RE.match(text or "")
    if not m:
        raise ValueError(f"“{text}” is not a range such as [Book1]Sheet1!B or "
                         "[Book1]Sheet1!B[3:10].")
    r1 = int(m.group("r1")) if m.group("r1") else None
    r2 = m.group("r2")
    r2 = int(r2) if r2 and r2 != "*" else (r1 if r1 and r2 is None else None)
    if r1 is not None and r2 is not None and r2 < r1:
        r1, r2 = r2, r1
    return RangeRef(m.group("book") or book, (m.group("sheet") or sheet).strip(),
                    m.group("col"), r1, r2)


def row_window(n: int, rows: tuple[int, int] | None) -> tuple[int, int]:
    """0-based ``(start, stop)`` of a 1-based inclusive row range within ``n`` rows
    (``rows[1] == 0`` means "to the last row")."""
    if not rows:
        return 0, n
    lo = max(int(rows[0]) - 1, 0)
    hi = int(rows[1]) if rows[1] else n
    return min(lo, n), max(min(hi, n), min(lo, n))


# ====================================================================== columns & sheets
@dataclass
class WColumn:
    name: str                       # short name (A, B, C...)
    designation: str = "Y"
    long_name: str = ""
    units: str = ""
    comments: str = ""
    formula: str = ""               # F(x)= row, Origin syntax (col(A)*2, sin(i)...)
    values: list = field(default_factory=list)
    width: int = 0                  # display width in pixels, 0 = default
    # Origin's masked cells: 0-based row indices excluded from plots, fits and analyses
    # (shown in red in the grid). Use the Worksheet mask API rather than this set directly.
    mask: set = field(default_factory=set)

    @property
    def title(self) -> str:
        """Legend text: Long Name when set, otherwise the short name (Origin's %(1))."""
        return self.long_name or self.name

    @property
    def axis_title(self) -> str:
        """Axis title: "Long Name (Units)" like Origin's default %(?X) / %(?Y) titles."""
        base = self.long_name or self.name
        return f"{base} ({self.units})" if self.units else base

    def length(self) -> int:
        """Rows up to the last non-empty cell."""
        for i in range(len(self.values) - 1, -1, -1):
            if self.values[i] is not None:
                return i + 1
        return 0

    def is_text(self) -> bool:
        return any(isinstance(v, str) and math.isnan(to_float(v)) for v in self.values)

    def numeric(self, n: int | None = None) -> np.ndarray:
        vals = self.values if n is None else (self.values + [None] * n)[:n]
        return np.array([to_float(v) for v in vals], dtype=float)

    def texts(self, n: int | None = None) -> list[str]:
        vals = self.values if n is None else (self.values + [None] * n)[:n]
        return [format_cell(v) for v in vals]


@dataclass
class Worksheet:
    name: str = "Sheet1"
    columns: list[WColumn] = field(default_factory=list)
    n_rows: int = DEFAULT_ROWS
    kind: str = "data"
    sparklines: bool = False        # Origin's Sparklines header row (tiny plot per column)
    # operations whose output is in this sheet (Statistics on Rows columns, a Frequency
    # Counts sheet): {"kind", "settings", "outputs": [column names] | None (whole sheet),
    # "mode", "hash", "dirty"} — Origin's lock on the output columns
    ops: list = field(default_factory=list)

    def op_of(self, col: int) -> dict | None:
        """The operation that produced column ``col`` (None for ordinary data)."""
        if not 0 <= col < len(self.columns):
            return None
        name = self.columns[col].name
        for op in self.ops:
            outs = op.get("outputs")
            if outs is None or name in outs:
                return op
        return None

    # ------------------------------------------------------------------ structure
    @classmethod
    def new(cls, name: str = "Sheet1", n_cols: int = 2, n_rows: int = DEFAULT_ROWS) \
            -> "Worksheet":
        ws = cls(name=name, n_rows=n_rows)
        for i in range(n_cols):
            ws.add_column("X" if i == 0 else "Y")
        return ws

    def names(self) -> list[str]:
        return [c.name for c in self.columns]

    def index_of(self, name: str) -> int:
        low = name.lower()
        for i, c in enumerate(self.columns):
            if c.name.lower() == low:
                return i
        raise KeyError(name)

    def column(self, name: str) -> WColumn:
        return self.columns[self.index_of(name)]

    def find_long_name(self, long_name: str) -> WColumn | None:
        return next((c for c in self.columns if c.long_name == long_name), None)

    def find(self, ref: str) -> int:
        """Column index from a short name or a quoted ``"Long Name"`` (KeyError if absent)."""
        if len(ref) > 1 and ref.startswith('"') and ref.endswith('"'):
            col = self.find_long_name(ref[1:-1])
            if col is None:
                raise KeyError(ref)
            return self.columns.index(col)
        return self.index_of(ref)

    def next_short_name(self) -> str:
        used = {c.name.upper() for c in self.columns}
        k = 0
        while True:
            name = _letters(k)
            if name not in used:
                return name
            k += 1

    def add_column(self, designation: str = "Y", at: int | None = None,
                   name: str | None = None) -> WColumn:
        col = WColumn(name or self.next_short_name(), designation,
                      values=[None] * self.n_rows)
        if at is None or at >= len(self.columns):
            self.columns.append(col)
        else:
            self.columns.insert(max(0, at), col)
        return col

    def delete_columns(self, indices) -> None:
        for i in sorted(set(indices), reverse=True):
            if 0 <= i < len(self.columns):
                del self.columns[i]

    def rename_column(self, index: int, name: str) -> None:
        name = name.strip()
        if not _SHORT_RE.match(name):
            raise ValueError("A short name must start with a letter and contain only letters, "
                             "digits and underscores.")
        if any(c.name.lower() == name.lower() for i, c in enumerate(self.columns)
               if i != index):
            raise ValueError(f"The short name {name} is already used in this sheet.")
        old = self.columns[index].name
        self.columns[index].name = name
        for c in self.columns:   # keep formulas pointing at the renamed column
            if c.formula:
                c.formula = re.sub(rf"(?i)col\(\s*{re.escape(old)}\s*\)", f"col({name})",
                                   c.formula)

    def set_rows(self, n: int) -> None:
        n = max(1, int(n))
        self.n_rows = n
        for c in self.columns:
            if len(c.values) < n:
                c.values.extend([None] * (n - len(c.values)))
            elif len(c.values) > n:
                del c.values[n:]
            if c.mask:
                c.mask = {r for r in c.mask if r < n}

    def ensure_rows(self, n: int) -> None:
        if n > self.n_rows:
            self.set_rows(n)
        for c in self.columns:
            if len(c.values) < self.n_rows:
                c.values.extend([None] * (self.n_rows - len(c.values)))

    def insert_rows(self, at: int, count: int = 1) -> None:
        for c in self.columns:
            c.values[at:at] = [None] * count
            if c.mask:
                c.mask = {r + count if r >= at else r for r in c.mask}
        self.n_rows += count

    def delete_rows(self, rows) -> None:
        rows = sorted({r for r in rows if 0 <= r < self.n_rows}, reverse=True)
        gone = set(rows)
        for c in self.columns:
            for r in rows:
                if r < len(c.values):
                    del c.values[r]
            if c.mask:
                c.mask = {m - sum(1 for r in gone if r < m) for m in c.mask if m not in gone}
        self.n_rows = max(1, self.n_rows - len(rows))
        self.ensure_rows(self.n_rows)

    def permute_rows(self, order, n: int | None = None) -> None:
        """Reorder the first ``n`` rows (default ``len(order)``) so that new row ``k`` is old
        row ``order[k]``, moving cell masks with their values (use this for sorting)."""
        order = [int(r) for r in order]
        n = len(order) if n is None else int(n)
        self.ensure_rows(n)
        new_of = {old: k for k, old in enumerate(order)}
        for c in self.columns:
            vals = (c.values + [None] * n)[:n]
            c.values[:n] = [vals[r] for r in order]
            if c.mask:
                c.mask = {new_of.get(m, m) if m < n else m for m in c.mask}

    # ------------------------------------------------------------------ cells
    def cell(self, row: int, col: int):
        values = self.columns[col].values
        return values[row] if row < len(values) else None

    def set_cell(self, row: int, col: int, value) -> None:
        self.ensure_rows(row + 1)
        self.columns[col].values[row] = parse_cell(value) if isinstance(value, str) else \
            (None if value is None or (isinstance(value, float) and math.isnan(value))
             else value)

    def set_column_values(self, col: int, values) -> None:
        vals = [parse_cell(v) if isinstance(v, str) else
                (None if v is None or (isinstance(v, float) and math.isnan(v)) else
                 float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v)
                for v in values]
        self.ensure_rows(len(vals))
        target = self.columns[col].values
        for r in range(self.n_rows):
            target[r] = vals[r] if r < len(vals) else None

    def data_rows(self) -> int:
        return max([c.length() for c in self.columns] + [0])

    # ------------------------------------------------------------------ designations
    def label(self, index: int) -> str:
        """Header text "A(X)", "B(Y)"… with X1/Y1, X2/Y2 numbering when there are several
        X columns, as Origin does."""
        col = self.columns[index]
        suffix = DESIGNATIONS.get(col.designation, ("", ""))[1]
        n_x = sum(1 for c in self.columns if c.designation == "X")
        if n_x > 1 and col.designation in ("X", "Y"):
            group = sum(1 for c in self.columns[:index + 1] if c.designation == "X")
            if col.designation == "Y" and group == 0:
                group = 1
            suffix += str(max(group, 1))
        return f"{col.name}({suffix})"

    def x_of(self, index: int) -> int | None:
        """The X column associated with column ``index``: the nearest X to its left, else
        the first X of the sheet; None means row numbers."""
        for i in range(index - 1, -1, -1):
            if self.columns[i].designation == "X":
                return i
        return next((i for i, c in enumerate(self.columns) if c.designation == "X"
                     and i != index), None)

    def _right_of(self, index: int, designation: str) -> int | None:
        for i in range(index + 1, len(self.columns)):
            d = self.columns[i].designation
            if d == designation:
                return i
            if d in ("X", "Y", "Z"):
                return None
        return None

    def yerr_of(self, index: int) -> int | None:
        return self._right_of(index, "yEr")

    def xerr_of(self, index: int) -> int | None:
        return self._right_of(index, "xEr")

    def label_of(self, index: int) -> int | None:
        found = self._right_of(index, "L")
        if found is not None:
            return found
        return next((i for i, c in enumerate(self.columns) if c.designation == "L"), None)

    def y_of(self, index: int) -> int | None:
        """For a Z column: the nearest Y column to its left (XYZ triplets)."""
        for i in range(index - 1, -1, -1):
            if self.columns[i].designation == "Y":
                return i
        return None

    # ------------------------------------------------------------------ masks
    # Origin's Mask: masked cells keep their value but are left out of plots, fits and
    # analyses (and drawn red). A point (x, y) is masked when its X or its Y cell is masked.
    # This is the stable API other packages use; rows are 0-based.
    def _col_index(self, col) -> int:
        return col if isinstance(col, (int, np.integer)) else self.find(str(col))

    def masked_rows(self, col) -> list[int]:
        """Sorted 0-based rows of column ``col`` (index or short name) that are masked."""
        return sorted(self.columns[self._col_index(col)].mask)

    def is_masked(self, row: int, col) -> bool:
        return int(row) in self.columns[self._col_index(col)].mask

    def has_masks(self, cols=None) -> bool:
        cols = range(len(self.columns)) if cols is None else cols
        return any(self.columns[self._col_index(c)].mask for c in cols)

    def mask_array(self, col, n: int | None = None) -> np.ndarray:
        """Boolean array of length ``n`` (default ``n_rows``): True where ``col`` is masked."""
        n = self.n_rows if n is None else int(n)
        out = np.zeros(n, dtype=bool)
        if col is None:
            return out
        rows = [r for r in self.columns[self._col_index(col)].mask if 0 <= r < n]
        out[rows] = True
        return out

    def point_mask(self, cols, n: int | None = None) -> np.ndarray:
        """True for rows where any of ``cols`` (None entries ignored) is masked."""
        n = self.n_rows if n is None else int(n)
        out = np.zeros(n, dtype=bool)
        for c in cols:
            if c is not None:
                out |= self.mask_array(c, n)
        return out

    def set_mask(self, col, rows, on: bool = True) -> None:
        """Mask (``on``) or unmask the 0-based ``rows`` of column ``col``."""
        c = self.columns[self._col_index(col)]
        rows = {int(r) for r in rows if 0 <= int(r) < max(self.n_rows, 1)}
        c.mask = (c.mask | rows) if on else (c.mask - rows)

    def mask_cells(self, col, rows) -> None:
        self.set_mask(col, rows, True)

    def unmask_cells(self, col, rows) -> None:
        self.set_mask(col, rows, False)

    def swap_mask(self, col, rows) -> None:
        """Origin's Swap Mask: masked cells of ``rows`` become unmasked and vice versa."""
        c = self.columns[self._col_index(col)]
        rows = {int(r) for r in rows if 0 <= int(r) < self.n_rows}
        c.mask = c.mask ^ rows

    def clear_mask(self, cols=None) -> None:
        """Unmask every cell of ``cols`` (default: the whole sheet)."""
        for c in (range(len(self.columns)) if cols is None else cols):
            self.columns[self._col_index(c)].mask = set()

    def mask_where(self, col, predicate) -> int:
        """Mask the cells of ``col`` whose numeric value satisfies ``predicate(v)`` (Origin's
        Mask by condition); returns how many cells were masked."""
        ci = self._col_index(col)
        vals = self.columns[ci].numeric(self.n_rows)
        hit = [r for r, v in enumerate(vals) if math.isfinite(v) and predicate(v)]
        self.set_mask(ci, hit, True)
        return len(hit)

    def xy_arrays(self, ycol, xcol=-1, rows: tuple[int, int] | None = None,
                  honor_mask: bool = True, with_error: bool = False):
        """The (x, y) data of column ``ycol`` against ``xcol`` (-1: its designated X;
        None: row numbers) over the 1-based inclusive ``rows`` range, without empty /
        non-numeric cells and, with ``honor_mask``, without masked points.
        Returns ``(idx, x, y, yerr)``: 0-based rows used and their values (``yerr`` is the
        Y Error column when ``with_error`` and complete, else None)."""
        y_i = self._col_index(ycol)
        x_i = self.x_of(y_i) if xcol == -1 else (None if xcol is None
                                                  else self._col_index(xcol))
        n = self.n_rows
        lo, hi = row_window(n, rows) if rows else (0, n)
        y = self.columns[y_i].numeric(n)
        x = self.columns[x_i].numeric(n) if x_i is not None else \
            np.arange(1, n + 1, dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        ok[:lo] = False
        ok[hi:] = False
        if honor_mask:
            ok &= ~self.point_mask([y_i, x_i], n)
        e = None
        e_i = self.yerr_of(y_i) if with_error else None
        if e_i is not None:
            ev = self.columns[e_i].numeric(n)[ok]
            e = ev if np.isfinite(ev).all() else None
        idx = np.flatnonzero(ok)
        return idx, x[ok], y[ok], e

    def copy(self) -> "Worksheet":
        return copy.deepcopy(self)


def _letters(k: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA… (Origin's short-name sequence)."""
    out = ""
    k += 1
    while k:
        k, r = divmod(k - 1, 26)
        out = chr(65 + r) + out
    return out


RECALC_MODES = {"auto": "Auto", "manual": "Manual", "none": "None"}


@dataclass
class ReportSheet:
    """An analysis report (Origin's report sheet): titled tables, notes and plot images,
    shown as collapsible sections. Consecutive items with the same ``group`` are nested under
    one parent section (e.g. "Residual Plots"); an item's ``collapsed`` flag (and
    ``group_collapsed`` on the first item of a group) remembers the state of a section.

    ``op`` describes the operation that produced the report so it can be re-run when its
    source data change (Origin's lock icon / Recalculate): ``{"kind": ..., "settings": ...,
    "mode": "auto" | "manual" | "none", "hash": <source data hash>, "dirty": bool}``."""
    name: str = "Report1"
    title: str = ""
    items: list[dict] = field(default_factory=list)
    kind: str = "report"
    op: dict = field(default_factory=dict)

    def add_table(self, title: str, columns: list, rows: list[list], notes=(),
                  group: str = "") -> None:
        self.items.append({"kind": "table", "title": title,
                           "columns": [list(c) if isinstance(c, tuple) else c for c in columns],
                           "rows": [[("" if v is None else str(v)) for v in r] for r in rows],
                           "notes": list(notes), "group": group})

    def add_text(self, title: str, text: str, style: str = "note", group: str = "") -> None:
        self.items.append({"kind": "text", "title": title, "text": text, "style": style,
                           "group": group})

    def add_image(self, title: str, png: bytes, width: int = 520, group: str = "",
                  graph: dict | None = None) -> None:
        item = {"kind": "image", "title": title, "width": width, "group": group,
                "png": base64.b64encode(png).decode("ascii")}
        if graph:
            item["graph"] = graph   # how to rebuild the picture as a graph window
        self.items.append(item)

    def add_graph(self, title: str, graph: Graph, width: int = 460, group: str = "",
                  open_spec: dict | None = None) -> None:
        """Embed a live graph (Origin's embedded graph object): the graph model is stored
        and rendered by the report view, so a recalculation redraws it. ``open_spec``
        tells how to open it as a graph window linked to worksheet data (e.g. a fit's
        source data and fitted curve); without it the window is linked to this report."""
        item = {"kind": "graph", "title": title, "width": width, "group": group,
                "graph": graph_to_json_dict(graph)}
        if open_spec:
            item["open"] = open_spec
        self.items.append(item)

    def graphs(self) -> list[dict]:
        return [i for i in self.items if i["kind"] == "graph"]

    @property
    def mode(self) -> str:
        return self.op.get("mode", "none") if self.op else "none"

    @property
    def dirty(self) -> bool:
        return bool(self.op.get("dirty")) if self.op else False

    def sections(self) -> list[tuple[str, list[int]]]:
        """Top-level sections: ``(group title or "", [item indices])`` in report order."""
        out: list[tuple[str, list[int]]] = []
        for k, item in enumerate(self.items):
            group = item.get("group", "")
            if group and out and out[-1][0] == group:
                out[-1][1].append(k)
            else:
                out.append((group, [k]))
        return out

    def replace_items(self, other: "ReportSheet") -> None:
        """Take the content of a re-run, keeping which sections were collapsed."""
        collapsed = {(i.get("group", ""), i.get("title")) for i in self.items
                     if i.get("collapsed")}
        groups = {i.get("group") for i in self.items if i.get("group_collapsed")}
        self.title = other.title
        self.items = other.items
        seen = set()
        for i in self.items:
            if (i.get("group", ""), i.get("title")) in collapsed:
                i["collapsed"] = True
            g = i.get("group")
            if g and g not in seen:
                seen.add(g)
                if g in groups:
                    i["group_collapsed"] = True

    def tables(self) -> list[dict]:
        return [i for i in self.items if i["kind"] == "table"]

    def table(self, title: str) -> dict | None:
        return next((t for t in self.tables() if t["title"] == title), None)

    def copy(self) -> "ReportSheet":
        return copy.deepcopy(self)


class SheetOp:
    """A recalculable operation whose output lives in a worksheet (Statistics on Rows adds
    columns to its source sheet, Frequency Counts fills its own sheet). It offers the same
    ``op`` / ``mode`` / ``dirty`` / ``name`` as a :class:`ReportSheet`, so the lock and
    Recalculate code treat both alike."""

    def __init__(self, sheet: Worksheet, op: dict):
        self.sheet = sheet
        self.op = op

    @property
    def name(self) -> str:
        return self.sheet.name

    @property
    def mode(self) -> str:
        return self.op.get("mode", "none")

    @property
    def dirty(self) -> bool:
        return bool(self.op.get("dirty"))

    def output_columns(self) -> list[int]:
        outs = self.op.get("outputs")
        return [i for i, c in enumerate(self.sheet.columns) if outs is None or c.name in outs]

    def __eq__(self, other) -> bool:
        return isinstance(other, SheetOp) and other.sheet is self.sheet and \
            other.op is self.op

    def __hash__(self) -> int:
        return hash((id(self.sheet), id(self.op)))


Sheet = Worksheet | ReportSheet


# ====================================================================== matrices
def _empty_grid(rows: int, cols: int) -> np.ndarray:
    return np.full((max(1, int(rows)), max(1, int(cols))), np.nan)


@dataclass
class MatrixSheet:
    """Origin's matrix sheet: a grid of Z values; columns map linearly to X from ``x1`` to
    ``x2`` and rows to Y from ``y1`` to ``y2`` (row 1 = ``y1``)."""
    name: str = "MSheet1"
    data: np.ndarray = field(default_factory=lambda: _empty_grid(32, 32))
    x1: float = 1.0
    x2: float = 32.0
    y1: float = 1.0
    y2: float = 32.0
    long_name: str = ""             # Z long name / units / comments
    units: str = ""
    comments: str = ""
    x_name: str = ""
    x_units: str = ""
    y_name: str = ""
    y_units: str = ""
    formula: str = ""               # last Set Matrix Values formula
    kind: str = "matrix"

    @classmethod
    def new(cls, name: str = "MSheet1", rows: int = 32, cols: int = 32) -> "MatrixSheet":
        return cls(name, _empty_grid(rows, cols), 1.0, float(cols), 1.0, float(rows))

    @property
    def n_rows(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_cols(self) -> int:
        return int(self.data.shape[1])

    def xs(self) -> np.ndarray:
        return np.linspace(self.x1, self.x2, self.n_cols) if self.n_cols > 1 else \
            np.array([self.x1], float)

    def ys(self) -> np.ndarray:
        return np.linspace(self.y1, self.y2, self.n_rows) if self.n_rows > 1 else \
            np.array([self.y1], float)

    def set_dimensions(self, rows: int, cols: int) -> None:
        """Origin's Set Dimensions: keep the top-left values, pad with empty cells."""
        new = _empty_grid(rows, cols)
        r, c = min(rows, self.n_rows), min(cols, self.n_cols)
        new[:r, :c] = self.data[:r, :c]
        self.data = new

    def is_empty(self) -> bool:
        return not np.isfinite(self.data).any()

    @staticmethod
    def _title(name: str, units: str, default: str) -> str:
        base = name or default
        return f"{base} ({units})" if units else base

    @property
    def z_title(self) -> str:
        return self._title(self.long_name, self.units, "Z")

    @property
    def x_title(self) -> str:
        return self._title(self.x_name, self.x_units, "X")

    @property
    def y_title(self) -> str:
        return self._title(self.y_name, self.y_units, "Y")

    def copy(self) -> "MatrixSheet":
        return copy.deepcopy(self)


@dataclass
class MatrixBook:
    """Origin's matrix book (MBook) window."""
    name: str = "MBook1"
    long_name: str = ""
    sheets: list = field(default_factory=list)
    active: int = 0
    kind: str = "matrix"

    @classmethod
    def new(cls, name: str = "MBook1", rows: int = 32, cols: int = 32) -> "MatrixBook":
        return cls(name=name, sheets=[MatrixSheet.new("MSheet1", rows, cols)])

    @property
    def title(self) -> str:
        return f"{self.name} - {self.long_name}" if self.long_name else self.name

    def sheet(self, name: str):
        return next((s for s in self.sheets if s.name.lower() == name.lower()), None)

    def active_sheet(self) -> MatrixSheet | None:
        if not self.sheets:
            return None
        return self.sheets[max(0, min(self.active, len(self.sheets) - 1))]

    def unique_sheet_name(self, base: str = "MSheet") -> str:
        k = 1
        while self.sheet(f"{base}{k}") is not None:
            k += 1
        return f"{base}{k}"


@dataclass
class Workbook:
    name: str = "Book1"
    long_name: str = ""
    sheets: list = field(default_factory=list)
    active: int = 0
    kind: str = "book"

    @classmethod
    def new(cls, name: str = "Book1", n_cols: int = 2, n_rows: int = DEFAULT_ROWS) \
            -> "Workbook":
        return cls(name=name, sheets=[Worksheet.new("Sheet1", n_cols, n_rows)])

    @property
    def title(self) -> str:
        return f"{self.name} - {self.long_name}" if self.long_name else self.name

    def sheet(self, name: str):
        return next((s for s in self.sheets if s.name.lower() == name.lower()), None)

    def active_sheet(self):
        if not self.sheets:
            return None
        return self.sheets[max(0, min(self.active, len(self.sheets) - 1))]

    def unique_sheet_name(self, base: str = "Sheet") -> str:
        k = 1
        while self.sheet(f"{base}{k}") is not None:
            k += 1
        return f"{base}{k}"

    def add_sheet(self, sheet=None, at: int | None = None):
        sheet = sheet or Worksheet.new(self.unique_sheet_name(), 2)
        if at is None:
            self.sheets.append(sheet)
        else:
            self.sheets.insert(at, sheet)
        return sheet

    def is_empty(self) -> bool:
        return all(isinstance(s, Worksheet) and s.data_rows() == 0 and
                   not any(c.long_name or c.units or c.comments or c.formula
                           for c in s.columns) for s in self.sheets) and len(self.sheets) <= 1


@dataclass
class GraphDoc:
    name: str
    graph: Graph
    kind: str = "graph"

    @property
    def title(self) -> str:
        return self.name


@dataclass
class Notes:
    name: str = "Notes1"
    text: str = ""
    kind: str = "notes"

    @property
    def title(self) -> str:
        return self.name


@dataclass
class Folder:
    name: str = "Folder1"
    folders: list["Folder"] = field(default_factory=list)
    windows: list[str] = field(default_factory=list)   # window names, in creation order

    def walk(self):
        yield self
        for f in self.folders:
            yield from f.walk()

    def find_window(self, name: str) -> "Folder | None":
        return next((f for f in self.walk() if name in f.windows), None)

    def parent_of(self, child: "Folder") -> "Folder | None":
        for f in self.walk():
            if child in f.folders:
                return f
        return None


# ====================================================================== project
@dataclass
class Project:
    name: str = "UNTITLED"
    root: Folder = field(default_factory=lambda: Folder("UNTITLED"))
    windows: dict = field(default_factory=dict)        # name -> Workbook | GraphDoc | Notes
    hidden: set = field(default_factory=set)
    results_log: list[str] = field(default_factory=list)
    path: str | None = None

    def unique_name(self, base: str) -> str:
        k = 1
        while any(n.lower() == f"{base}{k}".lower() for n in self.windows):
            k += 1
        return f"{base}{k}"

    def add(self, obj, folder: Folder | None = None):
        if obj.name in self.windows:
            obj.name = self.unique_name(re.sub(r"\d+$", "", obj.name) or "Window")
        self.windows[obj.name] = obj
        (folder or self.root).windows.append(obj.name)
        return obj

    def remove(self, name: str) -> None:
        self.windows.pop(name, None)
        self.hidden.discard(name)
        for f in self.root.walk():
            if name in f.windows:
                f.windows.remove(name)

    def rename(self, old: str, new: str) -> None:
        new = new.strip()
        if not new:
            raise ValueError("The name cannot be empty.")
        if new != old and any(n.lower() == new.lower() for n in self.windows):
            raise ValueError(f"A window named {new} already exists.")
        obj = self.windows.pop(old)
        obj.name = new
        if isinstance(obj, GraphDoc):
            obj.graph.title = new
        self.windows[new] = obj
        for f in self.root.walk():
            f.windows = [new if n == old else n for n in f.windows]
        if old in self.hidden:
            self.hidden.discard(old)
            self.hidden.add(new)
        for g in self.graphs():   # graph links follow the renamed book
            links = g.graph.meta.get("links", {})
            for link in links.values():
                if link.get("book") == old:
                    link["book"] = new
        for book in self.books():   # so do the inputs of recalculable reports
            for s in book.sheets:
                if isinstance(s, ReportSheet) and s.op:
                    s.op = rename_refs(s.op, old, new)

    def books(self) -> list[Workbook]:
        return [w for w in self.windows.values() if isinstance(w, Workbook)]

    def matrices(self) -> list[MatrixBook]:
        return [w for w in self.windows.values() if isinstance(w, MatrixBook)]

    def data_windows(self) -> dict:
        """Workbooks and matrix books by name (the sources of graph links)."""
        return {w.name: w for w in self.windows.values()
                if isinstance(w, (Workbook, MatrixBook))}

    def graphs(self) -> list[GraphDoc]:
        return [w for w in self.windows.values() if isinstance(w, GraphDoc)]

    def is_blank(self) -> bool:
        """An untouched new project (one empty Book1): closing it needs no save prompt."""
        books = self.books()
        return (not self.graphs() and not self.matrices()
                and not any(isinstance(w, Notes) and w.text
                                              for w in self.windows.values())
                and len(books) <= 1 and all(b.is_empty() for b in books)
                and len(list(self.root.walk())) == 1)


def rename_refs(value, old: str, new: str, key: str = "book"):
    """Copy of a settings structure pointing at a renamed window: with ``key="book"``,
    ``[old]`` in range strings and ``{"book": old}`` entries become ``new``; with any other
    ``key``, strings have the substring ``old`` replaced (e.g. ``"[Book1]Sheet1!"``)."""
    if isinstance(value, dict):
        return {k: (new if k == key and v == old else rename_refs(v, old, new, key))
                for k, v in value.items()}
    if isinstance(value, list):
        return [rename_refs(v, old, new, key) for v in value]
    if isinstance(value, str):
        return value.replace(f"[{old}]", f"[{new}]") if key == "book" else \
            value.replace(old, new)
    return value


# ====================================================================== serialisation
def _column_to_dict(c: WColumn) -> dict:
    d = {"name": c.name, "designation": c.designation, "long_name": c.long_name,
         "units": c.units, "comments": c.comments, "formula": c.formula, "width": c.width,
         "values": [None if v is None or (isinstance(v, float) and not math.isfinite(v))
                    else v for v in c.values]}
    if c.mask:   # only written when used, so older readers see an unchanged column
        d["mask"] = sorted(int(r) for r in c.mask)
    return d


def _column_from_dict(d: dict) -> WColumn:
    return WColumn(d["name"], d.get("designation", "Y"), d.get("long_name", ""),
                   d.get("units", ""), d.get("comments", ""), d.get("formula", ""),
                   list(d.get("values", [])), int(d.get("width", 0) or 0),
                   {int(r) for r in d.get("mask") or [] if int(r) >= 0})


_MATRIX_ATTRS = ("x1", "x2", "y1", "y2", "long_name", "units", "comments", "x_name",
                 "x_units", "y_name", "y_units", "formula")


def jsonable(value):
    """Copy of a settings / op dict with JSON-safe values (no numpy scalars, no NaN)."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    return value


def sheet_to_dict(s) -> dict:
    if isinstance(s, ReportSheet):
        return {"kind": "report", "name": s.name, "title": s.title,
                "items": copy.deepcopy(s.items), "op": jsonable(s.op)}
    if isinstance(s, MatrixSheet):
        d = {"kind": "matrix", "name": s.name,
             "data": [[float(v) if math.isfinite(v) else None for v in row]
                      for row in np.asarray(s.data, float)]}
        d.update({k: getattr(s, k) for k in _MATRIX_ATTRS})
        return d
    d = {"kind": "data", "name": s.name, "n_rows": s.n_rows, "sparklines": s.sparklines,
         "columns": [_column_to_dict(c) for c in s.columns]}
    if s.ops:
        d["ops"] = jsonable(s.ops)
    return d


def sheet_from_dict(d: dict):
    if d.get("kind") == "report":
        return ReportSheet(d["name"], d.get("title", ""), list(d.get("items", [])),
                           op=dict(d.get("op") or {}))
    if d.get("kind") == "matrix":
        data = np.array([[np.nan if v is None else float(v) for v in row]
                         for row in (d.get("data") or [[None]])], dtype=float)
        if data.ndim != 2 or data.size == 0:
            data = _empty_grid(1, 1)
        ms = MatrixSheet(d["name"], data)
        for k in _MATRIX_ATTRS:
            if d.get(k) is not None:
                setattr(ms, k, type(getattr(ms, k))(d[k]))
        return ms
    ws = Worksheet(d["name"], [_column_from_dict(c) for c in d.get("columns", [])],
                   int(d.get("n_rows", DEFAULT_ROWS)), sparklines=bool(d.get("sparklines")),
                   ops=[dict(o) for o in d.get("ops") or []])
    ws.ensure_rows(ws.n_rows)
    return ws


def window_to_dict(w) -> dict:
    if isinstance(w, (Workbook, MatrixBook)):
        return {"kind": w.kind, "name": w.name, "long_name": w.long_name, "active": w.active,
                "sheets": [sheet_to_dict(s) for s in w.sheets]}
    if isinstance(w, GraphDoc):
        return {"kind": "graph", "name": w.name, "graph": graph_to_json_dict(w.graph)}
    return {"kind": "notes", "name": w.name, "text": w.text}


def window_from_dict(d: dict):
    kind = d.get("kind")
    if kind == "book":
        return Workbook(d["name"], d.get("long_name", ""),
                        [sheet_from_dict(s) for s in d.get("sheets", [])],
                        int(d.get("active", 0)))
    if kind == "matrix":
        return MatrixBook(d["name"], d.get("long_name", ""),
                          [sheet_from_dict(s) for s in d.get("sheets", [])] or
                          [MatrixSheet.new()], int(d.get("active", 0)))
    if kind == "graph":
        return GraphDoc(d["name"], graph_from_json_dict(d["graph"]))
    return Notes(d["name"], d.get("text", ""))


def _folder_to_dict(f: Folder) -> dict:
    return {"name": f.name, "windows": list(f.windows),
            "folders": [_folder_to_dict(c) for c in f.folders]}


def _folder_from_dict(d: dict) -> Folder:
    return Folder(d.get("name", "Folder"), [_folder_from_dict(c) for c in d.get("folders", [])],
                  list(d.get("windows", [])))


def project_to_dict(p: Project, extra: dict | None = None) -> dict:
    return {"format": FORMAT, "version": VERSION, "name": p.name,
            "folders": _folder_to_dict(p.root),
            "windows": [window_to_dict(w) for w in p.windows.values()],
            "hidden": sorted(p.hidden), "results_log": list(p.results_log),
            "view": dict(extra or {})}


def project_from_dict(d: dict) -> Project:
    if d.get("format") != FORMAT:
        raise ValueError("This file is not an RS-StatChem Graph & Analysis project.")
    if int(d.get("version", 1)) > VERSION:
        raise ValueError("This project was saved by a newer version of RS-StatChem.")
    p = Project(d.get("name", "UNTITLED"), _folder_from_dict(d.get("folders", {})))
    for wd in d.get("windows", []):
        w = window_from_dict(wd)
        p.windows[w.name] = w
    known = set(p.windows)
    for f in p.root.walk():   # drop dangling references, file unfiled windows at the root
        f.windows = [n for n in f.windows if n in known]
    filed = {n for f in p.root.walk() for n in f.windows}
    p.root.windows.extend(n for n in p.windows if n not in filed)
    p.hidden = set(d.get("hidden", [])) & known
    p.results_log = list(d.get("results_log", []))
    return p


def save_project(p: Project, path: str | Path, extra: dict | None = None) -> None:
    data = json.dumps(project_to_dict(p, extra), ensure_ascii=False)
    tmp = Path(str(path) + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("project.json", data)
    tmp.replace(path)


def load_project(path: str | Path) -> tuple[Project, dict]:
    with zipfile.ZipFile(path) as zf:
        d = json.loads(zf.read("project.json").decode("utf-8"))
    p = project_from_dict(d)
    p.path = str(path)
    p.name = Path(path).stem
    p.root.name = p.name
    return p, d.get("view", {})

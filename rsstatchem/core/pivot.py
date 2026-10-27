"""Pivot-table editing for output tables (SPSS Pivot Table Editor, without Qt).

Any :class:`~rsstatchem.core.output.Table` is read as a cube: one dimension per row-header
column (outer labels are filled down), and one or two column dimensions (the two header
levels). :func:`pivot` rebuilds the table with the dimensions placed in the row, column and
layer trays, like SPSS's pivoting trays. The original table is kept in ``table.pivot`` so
repeated pivots never lose cells.

Also here: hiding rows/columns (:func:`visible` applies ``hidden_rows`` / ``hidden_cols``),
changing the decimals of numeric cells, and the table <-> dict conversion used by the
``.rso`` output documents.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field

from rsstatchem.core.output import Table

_NUM = re.compile(r"^(?P<pre>[<>≤≥]?\s*)(?P<sign>-?)(?P<int>\d*)(?:\.(?P<dec>\d+))?"
                  r"(?P<post>%?\s*[*ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ]*)$")


# ------------------------------------------------------------------ dict conversion
def table_to_dict(table: Table) -> dict:
    return {"kind": "table", "title": table.title,
            "columns": [list(c) if isinstance(c, tuple) else c for c in table.columns],
            "rows": [["" if v is None else str(v) for v in row] for row in table.rows],
            "row_header_cols": table.row_header_cols, "footnotes": list(table.footnotes),
            "hidden_rows": list(getattr(table, "hidden_rows", []) or []),
            "hidden_cols": list(getattr(table, "hidden_cols", []) or []),
            "pivot": copy.deepcopy(getattr(table, "pivot", None))}


def table_from_dict(d: dict) -> Table:
    t = Table(d.get("title", ""),
              [tuple(c) if isinstance(c, list) else c for c in d.get("columns", [])],
              [list(r) for r in d.get("rows", [])], int(d.get("row_header_cols", 1)),
              list(d.get("footnotes", [])))
    t.hidden_rows = list(d.get("hidden_rows", []))
    t.hidden_cols = list(d.get("hidden_cols", []))
    t.pivot = copy.deepcopy(d.get("pivot"))
    return t


def copy_table(table: Table, **changes) -> Table:
    t = table_from_dict(table_to_dict(table))
    for k, v in changes.items():
        setattr(t, k, v)
    return t


# ------------------------------------------------------------------ cube
@dataclass
class Cube:
    names: list[str]                 # dimension names
    values: list[list[str]]          # category order per dimension
    n_row_dims: int
    cells: dict = field(default_factory=dict)  # (value per dim) -> text
    title: str = ""
    footnotes: list = field(default_factory=list)
    headers: list = field(default_factory=list)  # original row-header labels (may be "")


def _label(col) -> str:
    return " ".join(p for p in col if p) if isinstance(col, tuple) else str(col)


def _filled_row_keys(table: Table) -> list[tuple]:
    rh = table.row_header_cols
    keys, last = [], [""] * rh
    for row in table.rows:
        heads = [("" if row[i] is None else str(row[i])) for i in range(rh)]
        out = []
        for i, h in enumerate(heads):
            if not h and any(heads[j] for j in range(i + 1, rh)):
                h = last[i]
            out.append(h)
        last = [o if o else prev for o, prev in zip(out, last)]
        keys.append(tuple(out))
    return keys


def cube(table: Table) -> Cube:
    """Read a table as a cube of dimensions (rows first, then columns)."""
    rh = table.row_header_cols
    names = []
    for i in range(rh):
        name = _label(table.columns[i]) if i < len(table.columns) else ""
        names.append(name or ("Rows" if rh == 1 else f"Rows {i + 1}"))
    grouped = table.has_groups
    col_keys = []
    for c in table.columns[rh:]:
        if grouped:
            col_keys.append(tuple(c) if isinstance(c, tuple) else (str(c), ""))
        else:
            col_keys.append((str(c),))
    names += ["Column groups", "Columns"] if grouped else ["Columns"]
    row_keys = _filled_row_keys(table) if rh else [()] * len(table.rows)
    n_dims = len(names)
    values: list[list[str]] = [[] for _ in range(n_dims)]
    cells = {}
    for rk, row in zip(row_keys, table.rows):
        for ck, cell in zip(col_keys, row[rh:]):
            key = tuple(rk) + tuple(ck)
            cells[key] = "" if cell is None else str(cell)
            for d, v in enumerate(key):
                if v not in values[d]:
                    values[d].append(v)
    if not table.rows:  # header-only table: keep its columns
        for ck in col_keys:
            for d, v in enumerate(ck):
                if v not in values[rh + d]:
                    values[rh + d].append(v)
    headers = [_label(table.columns[i]) if i < len(table.columns) else "" for i in range(rh)]
    return Cube(names, values, rh, cells, table.title, list(table.footnotes), headers)


def default_layout(table: Table) -> dict:
    if getattr(table, "pivot", None):
        return {"rows": list(table.pivot["rows"]), "cols": list(table.pivot["cols"]),
                "layers": {int(k): v for k, v in table.pivot.get("layers", {}).items()}}
    c = cube(table)
    return {"rows": list(range(c.n_row_dims)),
            "cols": list(range(c.n_row_dims, len(c.names))), "layers": {}}


def source_cube(table: Table) -> Cube:
    if getattr(table, "pivot", None):
        return cube(table_from_dict(table.pivot["source"]))
    return cube(table)


def compose(c: Cube, rows: list[int], cols: list[int], layers: dict | None = None) -> Table:
    """Build a table from a cube with the given dimensions in rows, columns and layers
    (``layers`` maps a dimension to the category shown)."""
    layers = dict(layers or {})
    used = set(rows) | set(cols) | set(layers)
    if used != set(range(len(c.names))) or len(used) != len(rows) + len(cols) + len(layers):
        raise ValueError("Every dimension must be in exactly one tray.")
    order = [{v: i for i, v in enumerate(vals)} for vals in c.values]
    entries = {}
    for key, text in c.cells.items():
        if any(key[d] != v for d, v in layers.items()):
            continue
        rk = tuple(key[d] for d in rows)
        ck = tuple(key[d] for d in cols)
        if (rk, ck) in entries and not text:
            continue
        entries[(rk, ck)] = text

    def sort_keys(keys, dims):
        return sorted(set(keys), key=lambda k: [order[d].get(v, 0) for d, v in zip(dims, k)])

    row_keys = sort_keys([rk for rk, _ in entries], rows)
    col_keys = sort_keys([ck for _, ck in entries], cols)
    if not cols:
        col_keys = [()]
    header = [(c.headers[d] if d < len(c.headers) else c.names[d]) for d in rows] or [""]
    columns: list = list(header)
    for ck in col_keys:
        if len(ck) >= 2:
            group, sub = " / ".join(p for p in ck[:-1] if p), ck[-1]
            columns.append((group, sub) if group and sub else (group or sub))
        elif ck:
            columns.append(ck[0])
        else:
            columns.append(" / ".join(c.names[d] for d in cols) or "Value")
    out_rows = []
    previous: tuple = ()
    for rk in row_keys or [()]:
        heads = list(rk) if rows else [""]
        if rows:  # SPSS prints an outer label once per block
            shown = []
            for i, v in enumerate(heads):
                same = len(previous) > i and all(previous[j] == rk[j] for j in range(i + 1))
                shown.append("" if same and i < len(heads) - 1 else v)
            heads = shown
        previous = rk
        out_rows.append(heads + [entries.get((rk, ck), "") for ck in col_keys])
    title = c.title
    if layers:
        title += " (" + ", ".join(f"{c.names[d]}: {v}" for d, v in layers.items()) + ")"
    return Table(title, columns, out_rows, row_header_cols=len(header),
                 footnotes=list(c.footnotes))


def pivot(table: Table, rows: list[int], cols: list[int], layers: dict | None = None) -> Table:
    """Pivoted copy of ``table``; the original stays in ``.pivot['source']``."""
    source = table.pivot["source"] if getattr(table, "pivot", None) else table_to_dict(table)
    source = dict(source)
    source["pivot"] = None
    c = cube(table_from_dict(source))
    new = compose(c, rows, cols, layers)
    new.pivot = {"source": source, "rows": list(rows), "cols": list(cols),
                 "layers": {str(k): v for k, v in (layers or {}).items()}}
    new.hidden_rows, new.hidden_cols = [], []
    return new


def reset_pivot(table: Table) -> Table:
    if not getattr(table, "pivot", None):
        return table
    return table_from_dict(table.pivot["source"])


# ------------------------------------------------------------------ hiding / formatting
def visible(table: Table) -> Table:
    """The table as displayed: hidden rows and columns removed (outer row labels of a hidden
    row move down to the next shown row)."""
    hr = set(getattr(table, "hidden_rows", []) or [])
    hc = set(getattr(table, "hidden_cols", []) or [])
    if not hr and not hc:
        return table
    rh = table.row_header_cols
    keep_cols = [i for i in range(len(table.columns)) if i not in hc or i < rh]
    rows, carry = [], None
    for r, row in enumerate(table.rows):
        row = list(row)
        if r in hr:
            if carry is None:
                carry = row[:rh]
            else:
                carry = [c if c else row[i] for i, c in enumerate(carry)]
            continue
        if carry is not None:
            for i in range(rh):
                deeper = any(str(row[j] or "") for j in range(i + 1, rh))
                if not row[i] and carry[i] and deeper:
                    row[i] = carry[i]
            carry = None
        rows.append([row[i] if i < len(row) else "" for i in keep_cols])
    t = Table(table.title, [table.columns[i] for i in keep_cols], rows,
              row_header_cols=rh, footnotes=list(table.footnotes))
    t.id = table.id
    return t


def format_cell(text: str, decimals: int) -> str:
    """Re-round a formatted number (".045", "12.50%", "<.001", "3.2*") to ``decimals``."""
    if text is None:
        return text
    s = str(text).strip()
    m = _NUM.match(s)
    if not m or not (m.group("int") or m.group("dec")):
        return text
    if m.group("pre").strip():  # "<.001" is a bound, not a value
        return text
    value = float((m.group("sign") or "") + (m.group("int") or "0") + "." +
                  (m.group("dec") or "0"))
    out = f"{abs(value):.{max(decimals, 0)}f}"
    if not m.group("int") and out.startswith("0."):  # keep SPSS's ".045" style
        out = out[1:]
    neg = value < 0 and float(out or 0) != 0
    return ("-" if neg else "") + out + m.group("post")


def set_decimals(table: Table, decimals: int, rows: list[int] | None = None,
                 cols: list[int] | None = None) -> int:
    """Change the decimals of numeric cells (all, or the given rows/columns); returns how many
    cells changed. Values were already rounded by the analysis, so extra decimals are zeros."""
    rh = table.row_header_cols
    changed = 0
    for r, row in enumerate(table.rows):
        if rows is not None and r not in rows:
            continue
        for c in range(rh, len(row)):
            if cols is not None and c not in cols:
                continue
            new = format_cell(row[c], decimals)
            if new != row[c]:
                row[c] = new
                changed += 1
    return changed

"""Worksheet operations of the Origin-style module (no Qt): conversions between worksheets,
pandas frames and imported tables, and the Worksheet-menu manipulations built on
:mod:`rsstatchem.core.analysis.datamanip` (Stack / Unstack, Pivot, Reduce, Remove
duplicates, Query, multi-key Sort, Split, Paste Transpose, Mask by condition).

Operations that produce a new sheet return a :class:`Worksheet` (the caller adds it to a
book inside an undo step); in-place operations take the sheet and change it.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from rsstatchem.core.analysis import datamanip as dm
from rsstatchem.core.analysis.importers import ImportedTable
from rsstatchem.ui.origin.model import MatrixSheet, Workbook, Worksheet, to_float

IMPORT_MODES = {"books": "New book for each file", "sheets": "New sheet for each file",
                "columns": "New columns in one sheet"}


# ---------------------------------------------------------------------- conversions
def column_series(ws: Worksheet, col: int, n: int | None = None,
                  honor_mask: bool = False) -> pd.Series:
    """A column as a Series: float when every non-empty cell is a number, else object."""
    n = ws.data_rows() if n is None else n
    c = ws.columns[col]
    vals = (c.values + [None] * n)[:n]
    if honor_mask and c.mask:
        vals = [None if r in c.mask else v for r, v in enumerate(vals)]
    if c.is_text():
        return pd.Series([v if v is not None else None for v in vals], dtype=object)
    return pd.Series([to_float(v) for v in vals], dtype=float)


def sheet_frame(ws: Worksheet, cols=None, honor_mask: bool = False) -> pd.DataFrame:
    """The data rows of ``cols`` (default all) as a DataFrame keyed by short name."""
    cols = range(len(ws.columns)) if cols is None else cols
    n = ws.data_rows()
    return pd.DataFrame({ws.columns[c].name: column_series(ws, c, n, honor_mask)
                         for c in cols})


def _cell(v):
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return float(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        f = float(v)
        return None if math.isnan(f) else f
    if v is pd.NA or v is pd.NaT:
        return None
    s = str(v)
    return s if s != "" else None


def frame_to_worksheet(df: pd.DataFrame, name: str = "Sheet1", long_names: dict | None = None,
                       designations: list[str] | None = None, units: dict | None = None) \
        -> Worksheet:
    """A DataFrame as a new worksheet (column labels → Long Names, or ``long_names``)."""
    ws = Worksheet(name=name[:31] or "Sheet1", n_rows=max(len(df), 1))
    for j, col in enumerate(df.columns):
        c = ws.add_column(designations[j] if designations and j < len(designations) else
                          ("X" if j == 0 else "Y"))
        c.long_name = str((long_names or {}).get(col, col))
        c.units = str((units or {}).get(col, ""))
        ws.set_column_values(j, [_cell(v) for v in df[col].tolist()])
    if not ws.columns:
        ws.add_column("X")
        ws.add_column("Y")
    return ws


def table_to_worksheet(t: ImportedTable, name: str | None = None) -> Worksheet:
    n = max(t.n_rows, 1)
    ws = Worksheet(name=(name or t.name or "Sheet1")[:31], n_rows=n)
    for ic in t.columns:
        c = ws.add_column(ic.designation if ic.designation in
                          ("X", "Y", "Z", "xEr", "yEr", "L", "N") else "Y")
        c.long_name, c.units, c.comments = ic.long_name, ic.units, ic.comments
        ws.set_column_values(len(ws.columns) - 1, ic.values)
    if not ws.columns:
        ws.add_column("X")
    return ws


def table_to_matrix(t: ImportedTable, name: str = "MSheet1") -> MatrixSheet:
    data = np.asarray(t.matrix, float)
    x1, x2, y1, y2 = t.mapping or (1.0, float(data.shape[1]), 1.0, float(data.shape[0]))
    return MatrixSheet(name=name, data=data, x1=x1, x2=x2, y1=y1, y2=y2,
                       long_name=str(t.meta.get("variable", "")) or t.name)


def append_columns(ws: Worksheet, t: ImportedTable, suffix: str = "") -> list[int]:
    """Origin's "New columns" import mode: the table's columns appended to ``ws``."""
    start = len(ws.columns)
    ws.ensure_rows(max(t.n_rows, ws.n_rows))
    for ic in t.columns:
        c = ws.add_column(ic.designation)
        c.long_name = ic.long_name
        c.units = ic.units
        c.comments = (ic.comments + " " + suffix).strip() if suffix else ic.comments
        ws.set_column_values(len(ws.columns) - 1, ic.values)
    return list(range(start, len(ws.columns)))


def tables_to_books(tables_per_file: list[tuple[str, list[ImportedTable]]], mode: str,
                    unique_name, rows: int = 32) -> list:
    """Place imported tables into new Workbooks / MatrixBooks (model objects only).
    ``tables_per_file``: ``[(file stem, tables)]``; ``unique_name(base)`` gives free window
    names. ``mode`` is one of :data:`IMPORT_MODES` and applies to sheet tables; matrices
    always go to a matrix book per file."""
    from rsstatchem.ui.origin.model import MatrixBook
    out: list = []
    shared: Workbook | None = None
    for stem, tables in tables_per_file:
        sheets = [t for t in tables if t.kind == "sheet"]
        mats = [t for t in tables if t.kind == "matrix"]
        if sheets:
            if mode == "books" or shared is None:
                book = Workbook(name=unique_name("Book"), long_name=stem if mode == "books"
                                else "Imported")
                book.sheets = []
                out.append(book)
                if mode != "books":
                    shared = book
            else:
                book = shared
            for t in sheets:
                if mode == "columns":
                    if not book.sheets:
                        book.sheets.append(Worksheet(name="Sheet1", n_rows=max(t.n_rows, 1)))
                    append_columns(book.sheets[0], t, stem)
                    continue
                label = t.name if mode == "books" else (stem if len(sheets) == 1 else
                                                         f"{stem}_{t.name}")
                ws = table_to_worksheet(t, book.unique_sheet_name(label[:28]) if
                                        book.sheet(label[:31]) else label[:31])
                ws.ops = []
                book.sheets.append(ws)
        if mats:
            mb = MatrixBook(name=unique_name("MBook"), long_name=stem)
            mb.sheets = [table_to_matrix(t, f"MSheet{k + 1}") for k, t in enumerate(mats)]
            for k, t in enumerate(mats):
                mb.sheets[k].long_name = t.name
            out.append(mb)
    return out


# ---------------------------------------------------------------------- manipulations
def _long(ws: Worksheet, col: int) -> str:
    return ws.columns[col].long_name or ws.columns[col].name


def stack(ws: Worksheet, cols: list[int], id_cols: list[int] = (), name: str = "Stacked",
          drop_missing: bool = True) -> Worksheet:
    df = sheet_frame(ws, list(id_cols) + list(cols), honor_mask=True)
    rename = {ws.columns[c].name: _long(ws, c) for c in list(id_cols) + list(cols)}
    df = df.rename(columns=rename)
    out = dm.stack_columns(df, [rename[ws.columns[c].name] for c in cols],
                           [rename[ws.columns[c].name] for c in id_cols], "Group", "Value",
                           drop_missing)
    desig = ["Y"] * len(id_cols) + ["L", "Y"]
    return frame_to_worksheet(out, name, designations=desig)


def unstack(ws: Worksheet, value_col: int, group_col: int, key_col: int | None = None,
            name: str = "Unstacked") -> Worksheet:
    df = sheet_frame(ws, honor_mask=True)
    v, g = ws.columns[value_col].name, ws.columns[group_col].name
    k = ws.columns[key_col].name if key_col is not None else None
    out = dm.unstack_columns(df[df[g].notna()], v, g, k)
    ws2 = frame_to_worksheet(out, name, designations=(["X"] if k else []) +
                             ["Y"] * (out.shape[1] - (1 if k else 0)))
    for c in ws2.columns[(1 if k else 0):]:
        c.comments = f"{_long(ws, group_col)} = {c.long_name}"
        c.long_name = f"{_long(ws, value_col)} ({c.long_name})"
    return ws2


def pivot_table(ws: Worksheet, row_col: int, col_col: int, value_col: int,
                aggfunc: str = "mean", name: str = "Pivot") -> Worksheet:
    df = sheet_frame(ws, honor_mask=True)
    out = dm.pivot(df, ws.columns[row_col].name, ws.columns[col_col].name,
                   ws.columns[value_col].name, aggfunc)
    ws2 = frame_to_worksheet(out, name)
    ws2.columns[0].long_name = _long(ws, row_col)
    for c in ws2.columns[1:]:
        c.comments = f"{dm.AGGREGATES[aggfunc]} of {_long(ws, value_col)}"
    return ws2


def reduce_rows(ws: Worksheet, n: int, method: str = "first", start: int = 0,
                name: str = "Reduced") -> Worksheet:
    df = sheet_frame(ws)
    out = dm.reduce_rows(df, n, method, start)
    return _like(ws, out, name)


def _like(ws: Worksheet, df: pd.DataFrame, name: str) -> Worksheet:
    """A result frame (columns keyed by short names of ``ws``) as a worksheet keeping the
    source Long Names, Units and designations."""
    out = Worksheet(name=name[:31], n_rows=max(len(df), 1))
    for key in df.columns:
        try:
            src = ws.column(str(key))
        except KeyError:
            src = None
        c = out.add_column(src.designation if src else "Y")
        if src:
            c.long_name, c.units, c.comments = src.long_name, src.units, src.comments
        else:
            c.long_name = str(key)
        out.set_column_values(len(out.columns) - 1, [_cell(v) for v in df[key].tolist()])
    return out


def reduce_duplicate_x(ws: Worksheet, xcol: int, ycols: list[int], method: str = "mean",
                       name: str = "ReducedX") -> Worksheet:
    n = ws.data_rows()
    keep = ~ws.point_mask([xcol] + list(ycols), n)
    x = ws.columns[xcol].numeric(n)[keep]
    ys = {ws.columns[c].name: ws.columns[c].numeric(n)[keep] for c in ycols}
    out = dm.reduce_duplicate_x(x, ys, method).rename(columns={"X": ws.columns[xcol].name})
    return _like(ws, out, name)


def reduce_xy(ws: Worksheet, xcol: int, ycols: list[int], step: float | None = None,
              n_points: int | None = None, method: str = "mean",
              name: str = "ReducedXY") -> Worksheet:
    n = ws.data_rows()
    keep = ~ws.point_mask([xcol] + list(ycols), n)
    x = ws.columns[xcol].numeric(n)[keep]
    ys = {ws.columns[c].name: ws.columns[c].numeric(n)[keep] for c in ycols}
    out = dm.reduce_evenly_spaced(x, ys, step, n_points, None, method)
    return _like(ws, out.rename(columns={"X": ws.columns[xcol].name}), name)


def remove_duplicates(ws: Worksheet, key_cols: list[int] | None = None,
                      name: str = "Unique") -> Worksheet:
    df = sheet_frame(ws)
    subset = [ws.columns[c].name for c in key_cols] if key_cols else None
    return _like(ws, dm.remove_duplicates(df, subset), name)


def query_rows(ws: Worksheet, expr: str) -> np.ndarray:
    """0-based data rows where the condition holds. Columns: short names, ``col(A)`` or
    ``col("Long Name")``."""
    df = sheet_frame(ws)
    names = {c.long_name: c.name for c in ws.columns if c.long_name}
    expr2 = expr
    for ln, short in names.items():
        expr2 = expr2.replace(f'col("{ln}")', f"col({short})")
    return np.flatnonzero(dm.query_mask(df, expr2, names))


def extract_rows(ws: Worksheet, rows, name: str = "Extracted") -> Worksheet:
    out = ws.copy()
    out.name = name[:31]
    out.ops = []
    keep = set(int(r) for r in rows)
    n = ws.data_rows()
    out.delete_rows([r for r in range(n) if r not in keep])
    out.set_rows(max(len(keep), 1))
    return out


def sort_rows(ws: Worksheet, keys: list[tuple[int, bool]]) -> None:
    """Nested sort of the whole worksheet in place by ``[(column, ascending)]`` (first key
    most significant); masks move with their rows."""
    n = ws.data_rows()
    order = dm.sort_order([(ws.columns[c].values + [None] * n)[:n] for c, _ in keys],
                          [asc for _, asc in keys])
    ws.permute_rows(order, n)


def split(ws: Worksheet, by_col: int | None = None, n_rows: int | None = None) \
        -> list[Worksheet]:
    """Split into several sheets by the levels of a column or every ``n_rows`` rows."""
    n = ws.data_rows()
    if by_col is not None:
        levels: dict = {}
        for r, v in enumerate((ws.columns[by_col].values + [None] * n)[:n]):
            levels.setdefault(v, []).append(r)
        out = []
        for lv, rows in levels.items():
            label = "missing" if lv is None else (str(int(lv)) if isinstance(lv, float) and
                                                  lv.is_integer() else str(lv))
            out.append(extract_rows(ws, rows, f"{ws.name}_{label}"[:31]))
        return out
    step = max(1, int(n_rows or 1))
    return [extract_rows(ws, range(i, min(i + step, n)), f"{ws.name}_{k + 1}")
            for k, i in enumerate(range(0, n, step))]


def trim_missing(ws: Worksheet, cols: list[int] | None = None, how: str = "any",
                 name: str = "Trimmed") -> Worksheet:
    cols = list(range(len(ws.columns))) if cols is None else cols
    n = ws.data_rows()
    rows = []
    for r in range(n):
        empty = [ws.cell(r, c) is None for c in cols]
        if not (any(empty) if how == "any" else all(empty)):
            rows.append(r)
    return extract_rows(ws, rows, name)


def paste_transpose(ws: Worksheet, block: list[list[str]], top: int, left: int) -> None:
    """Origin's Paste Transpose: the clipboard block with rows and columns swapped, pasted
    at data row ``top`` / column ``left`` (0-based)."""
    width = max((len(r) for r in block), default=0)
    t = [[(r[j] if j < len(r) else "") for r in block] for j in range(width)]
    while len(ws.columns) < left + len(t[0] if t else []):
        ws.add_column("Y")
    ws.ensure_rows(top + len(t))
    for i, row in enumerate(t):
        for j, v in enumerate(row):
            ws.set_cell(top + i, left + j, v)
            ws.columns[left + j].formula = ""


def mask_condition(ws: Worksheet, col: int, expr: str, mode: str = "mask") -> int:
    """Mask (or unmask) the cells of ``col`` in the rows where ``expr`` holds; returns the
    number of rows matched."""
    rows = query_rows(ws, expr)
    ws.set_mask(col, rows.tolist(), mode == "mask")
    return len(rows)

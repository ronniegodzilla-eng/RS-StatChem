"""Results tables with ImageJ semantics (``ij.measure.ResultsTable``).

* Column order: a "Results" table (:meth:`ResultsTable.results`) keeps ImageJ's standard
  headings (:data:`DEFAULT_HEADINGS`, Measure order) first in their fixed order, then other
  columns in creation order; plain tables keep creation order. A "Label" column (row labels) is
  shown first.
* Values: floats or strings per cell. A column created after rows exist is 0 in earlier rows
  (NaN with ``nan_empty_cells``), as in ImageJ.
* Formatting (:func:`d2s`, :meth:`ResultsTable.format_value`): ImageJ's AUTO_FORMAT shows
  integers without decimals and other values with ``precision`` decimals (default 3); tiny or
  huge numbers switch to scientific notation; Java DecimalFormat rounding (HALF_EVEN, HALF_UP
  for 0 decimals).
* CSV: :meth:`ResultsTable.save` writes ImageJ's format (" " row-number column when row numbers
  are shown, comma for .csv / tab otherwise, quotes around values containing commas);
  :meth:`ResultsTable.read` reads ImageJ CSV/TSV files back.
Ported in part from ImageJ 1.54 ij/measure/ResultsTable.java (public domain, NIH).
"""
from __future__ import annotations

import csv
import io
import math
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

DEFAULT_HEADINGS = (
    "Area", "Mean", "StdDev", "Mode", "Min", "Max", "X", "Y", "XM", "YM", "Perim.", "BX", "BY",
    "Width", "Height", "Major", "Minor", "Angle", "Circ.", "Feret", "IntDen", "Median", "Skew",
    "Kurt", "%Area", "RawIntDen", "Ch", "Slice", "Frame", "FeretX", "FeretY", "FeretAngle",
    "MinFeret", "AR", "Round", "Solidity", "MinThr", "MaxThr")
LABEL = "Label"
AUTO_FORMAT = None


def d2s(value: float, decimals: int) -> str:
    """ImageJ ``ResultsTable.d2s``: fixed decimals, scientific for tiny/huge numbers or
    negative ``decimals`` (= significant digits in scientific notation)."""
    if value is None:
        return ""
    v = float(value)
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "Infinity" if v > 0 else "-Infinity"
    if v == 3.4028234663852886e38:
        return "3.4e38"
    av = abs(v)
    if (av != 0.0 and av < 1.0 / 10 ** decimals) or av > 999999999999 or decimals < 0:
        digits = min(-decimals, 9) if decimals < 0 else 3
        return _scientific(v, digits)
    decimals = min(max(decimals, 0), 9)
    q = Decimal(1).scaleb(-decimals)
    rounding = ROUND_HALF_UP if decimals == 0 else ROUND_HALF_EVEN
    text = str(Decimal(v).quantize(q, rounding=rounding))
    if text.startswith("-") and float(text) == 0:
        text = text[1:]   # negative zero
    return text


def _scientific(v: float, digits: int) -> str:
    """Java DecimalFormat("0.000E0") style: mantissa with ``digits`` decimals, "E-5"."""
    if v == 0:
        return "0." + "0" * digits + "E0"
    d = Decimal(v)
    exp = d.adjusted()
    mant = (d.scaleb(-exp)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_EVEN)
    if abs(mant) >= 10:
        exp += 1
        mant = (d.scaleb(-exp)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_EVEN)
    return f"{mant}E{exp}"


class ResultsTable:
    """An ordered table of named columns (float or str cells) and optional row labels."""

    def __init__(self, title: str = "Results", default_headings: bool = False):
        self.title = title
        self.precision = 3
        self.decimal_places: dict[str, int | None] = {}   # per column; None = AUTO_FORMAT
        self.show_row_numbers = default_headings
        self.nan_empty_cells = False
        self.base_row_number = 1
        self._standard = default_headings
        self._columns: dict[str, list] = {}
        self._labels: list[str | None] | None = None
        self._n = 0

    @classmethod
    def results(cls, title: str = "Results") -> "ResultsTable":
        """A table laid out like ImageJ's Results window (standard headings first)."""
        return cls(title, default_headings=True)

    # ------------------------------------------------------------------ size and columns
    def __len__(self) -> int:
        return self._n

    @property
    def size(self) -> int:
        return self._n

    @property
    def headings(self) -> list[str]:
        """Column headings in display order (including "Label" when present)."""
        names = list(self._columns)
        if self._standard:
            std = [h for h in DEFAULT_HEADINGS if h in self._columns]
            names = std + [h for h in names if h not in DEFAULT_HEADINGS]
        return ([LABEL] if self._labels is not None else []) + names

    def column_exists(self, name: str) -> bool:
        return name in self._columns or (name == LABEL and self._labels is not None)

    def _ensure_column(self, name: str) -> list:
        if name not in self._columns:
            fill = float("nan") if self.nan_empty_cells else 0.0
            self._columns[name] = [fill] * self._n
        return self._columns[name]

    # ------------------------------------------------------------------ editing
    def increment_counter(self) -> int:
        """Add an empty row (ImageJ ``incrementCounter``); returns its index."""
        fill = float("nan") if self.nan_empty_cells else 0.0
        for col in self._columns.values():
            col.append(fill)
        if self._labels is not None:
            self._labels.append(None)
        self._n += 1
        return self._n - 1

    def add_value(self, column: str, value) -> None:
        """Set ``column`` in the last row (ImageJ ``addValue``)."""
        if self._n == 0:
            self.increment_counter()
        self.set_value(column, self._n - 1, value)

    def add_label(self, label: str) -> None:
        if self._n == 0:
            self.increment_counter()
        self.set_label(label, self._n - 1)

    def add_row(self, values: dict | None = None, label: str | None = None, **kw) -> int:
        """Append a row from a mapping (keys in column-creation order)."""
        row = self.increment_counter()
        if label is not None:
            self.set_label(label, row)
        for k, v in {**(values or {}), **kw}.items():
            self.set_value(k, row, v)
        return row

    def set_value(self, column: str, row: int, value) -> None:
        if column == LABEL:
            self.set_label(None if value is None else str(value), row)
            return
        while row >= self._n:
            self.increment_counter()
        col = self._ensure_column(column)
        col[row] = value if isinstance(value, str) else float(value)

    def set_label(self, label: str | None, row: int) -> None:
        while row >= self._n:
            self.increment_counter()
        if self._labels is None:
            self._labels = [None] * self._n
        self._labels[row] = label

    def get_value(self, column: str, row: int):
        if column == LABEL:
            return self.get_label(row)
        if column not in self._columns:
            raise KeyError(f'"{column}" column not found')
        return self._columns[column][row]

    def get_label(self, row: int) -> str | None:
        return None if self._labels is None else self._labels[row]

    def column(self, name: str) -> np.ndarray:
        """Numeric column as float array (strings become NaN); "Label" as object array."""
        if name == LABEL:
            return np.array(self._labels or [None] * self._n, dtype=object)
        return np.array([v if not isinstance(v, str) else np.nan for v in self._columns[name]],
                        dtype=float)

    def row(self, index: int) -> dict:
        return {h: self.get_value(h, index) for h in self.headings}

    def delete_row(self, index: int) -> None:
        for col in self._columns.values():
            del col[index]
        if self._labels is not None:
            del self._labels[index]
        self._n -= 1

    def delete_column(self, name: str) -> None:
        if name == LABEL:
            self._labels = None
        else:
            self._columns.pop(name, None)

    def rename_column(self, old: str, new: str) -> None:
        self._columns = {(new if k == old else k): v for k, v in self._columns.items()}

    def reset(self) -> None:
        """Clear all rows and columns (Analyze > Clear Results)."""
        self._columns.clear()
        self._labels = None
        self._n = 0

    def sort(self, column: str, ascending: bool = True) -> None:
        keys = list(self.column(column))
        order = sorted(range(self._n), key=lambda i: (math.isnan(keys[i]), keys[i]),
                       reverse=not ascending)
        for name, col in self._columns.items():
            self._columns[name] = [col[i] for i in order]
        if self._labels is not None:
            self._labels = [self._labels[i] for i in order]

    def copy(self, title: str | None = None) -> "ResultsTable":
        rt = ResultsTable(title or self.title, self._standard)
        rt.precision, rt.show_row_numbers = self.precision, self.show_row_numbers
        rt.nan_empty_cells, rt.base_row_number = self.nan_empty_cells, self.base_row_number
        rt.decimal_places = dict(self.decimal_places)
        rt._columns = {k: list(v) for k, v in self._columns.items()}
        rt._labels = None if self._labels is None else list(self._labels)
        rt._n = self._n
        return rt

    # ------------------------------------------------------------------ statistics
    def summarize(self) -> "ResultsTable":
        """Mean, SD, Min, Max rows of every numeric column (Analyze > Summarize; SD n-1)."""
        out = ResultsTable(self.title, self._standard)
        out.precision = self.precision
        names = [h for h in self.headings if h != LABEL]
        for stat in ("Mean", "SD", "Min", "Max"):
            row = out.increment_counter()
            out.set_label(stat, row)
            for name in names:
                v = self.column(name)
                v = v[~np.isnan(v)]
                if v.size == 0:
                    value = float("nan")
                elif stat == "Mean":
                    value = float(v.mean())
                elif stat == "SD":
                    value = float(v.std(ddof=1)) if v.size > 1 else float("nan")
                elif stat == "Min":
                    value = float(v.min())
                else:
                    value = float(v.max())
                out.set_value(name, row, value)
        return out

    # ------------------------------------------------------------------ formatting / I/O
    def format_value(self, column: str, row: int) -> str:
        value = self.get_value(column, row)
        if column == LABEL:
            return value or ""
        if isinstance(value, str):
            return value.replace("\n", "\\n")
        if self.nan_empty_cells and math.isnan(value):
            return ""
        places = self.decimal_places.get(column, AUTO_FORMAT)
        if places is AUTO_FORMAT:
            if not math.isnan(value) and not math.isinf(value) and value == int(value) \
                    and abs(value) < 2 ** 31:
                return d2s(value, 0)
            return d2s(value, self.precision)
        return d2s(value, places)

    def to_rows(self) -> list[list[str]]:
        """Formatted cells (no header, no row numbers)."""
        heads = self.headings
        return [[self.format_value(h, r) for h in heads] for r in range(self._n)]

    def to_text(self, delimiter: str = ",", headers: bool = True,
                row_numbers: bool | None = None) -> str:
        rows_on = self.show_row_numbers if row_numbers is None else row_numbers
        out = io.StringIO()
        heads = self.headings
        if headers and (self._n or heads):
            out.write(delimiter.join(([" "] if rows_on else []) + heads) + "\n")
        quote = delimiter == ","
        for r in range(self._n):
            cells = []
            for h in heads:
                s = self.format_value(h, r)
                if h == LABEL and quote:
                    s = s.replace(",", ";")
                elif quote and ("," in s or s == ""):
                    s = f'"{s}"'
                cells.append(s)
            if rows_on:
                cells.insert(0, str(r + self.base_row_number))
            out.write(delimiter.join(cells) + "\n")
        return out.getvalue()

    def save(self, path, headers: bool = True, row_numbers: bool | None = None) -> None:
        """Save like ImageJ: CSV for ".csv", tab-delimited otherwise."""
        path = Path(path)
        delim = "," if path.suffix.lower() == ".csv" else "\t"
        path.write_text(self.to_text(delim, headers, row_numbers), encoding="utf-8")

    @classmethod
    def read(cls, path, title: str | None = None) -> "ResultsTable":
        """Read an ImageJ results/table file (CSV or TSV, with or without row numbers)."""
        path = Path(path)
        text = path.read_text(encoding="utf-8-sig")
        delim = "," if path.suffix.lower() == ".csv" else "\t"
        rows = list(csv.reader(io.StringIO(text), delimiter=delim))
        rt = cls(title or path.stem)
        if not rows:
            return rt
        heads = rows[0]
        skip_first = bool(heads) and heads[0].strip() == ""
        rt.show_row_numbers = skip_first
        for cells in rows[1:]:
            if not cells:
                continue
            r = rt.increment_counter()
            for h, cell in zip(heads, cells):
                if skip_first and h is heads[0]:
                    continue
                if h == LABEL:
                    rt.set_label(cell, r)
                    continue
                rt.set_value(h, r, _parse_cell(cell))
        return rt

    def to_dataframe(self):
        import pandas as pd

        data = {}
        for h in self.headings:
            if h == LABEL:
                data[h] = list(self._labels or [])
            else:
                data[h] = self._columns[h]
        return pd.DataFrame(data)

    def __repr__(self) -> str:
        return f"ResultsTable({self.title!r}, {self._n} rows, {self.headings})"


def _parse_cell(cell: str):
    s = cell.strip()
    if s == "":
        return float("nan")
    if s in ("NaN", "Infinity", "-Infinity"):
        return float(s.replace("Infinity", "inf"))
    try:
        return float(s)
    except ValueError:
        return cell

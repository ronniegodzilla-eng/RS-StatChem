"""File importers of the Graph & Analysis module (Origin's File > Import), without Qt.

Every reader returns a list of :class:`ImportedTable` — a neutral column table (Long Name,
Units, Comments, designation and values per column, plus file metadata) or a matrix — that the
UI turns into worksheets / matrix sheets. :func:`read_file` picks the reader from the file
extension; :data:`FILTERS` lists them for file dialogs.

Readers: delimited text with Origin's Import Wizard options (:mod:`.ascii_wizard`), Excel
(all sheets), JCAMP-DX 4.24/5/6 (:mod:`.jcamp`, own code from the IUPAC specification),
Thermo/Galactic SPC (:mod:`.spc`, own code from the published format description), MATLAB
``.mat`` (:mod:`.matlab`), HDF5 / NetCDF (:mod:`.hdf`; HDF5 needs the optional ``h5py``),
WAV audio and images to a matrix (:mod:`.media`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class ImportFailed(ValueError):
    """A file could not be read (message suitable for the user)."""


@dataclass
class ImportColumn:
    long_name: str = ""
    values: list = field(default_factory=list)      # float | str | None
    units: str = ""
    comments: str = ""
    designation: str = "Y"


@dataclass
class ImportedTable:
    """One sheet of imported data (``kind="sheet"``) or one matrix (``kind="matrix"``:
    ``matrix`` is the Z grid, ``mapping`` = (x1, x2, y1, y2))."""
    name: str
    columns: list[ImportColumn] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    kind: str = "sheet"
    matrix: np.ndarray | None = None
    mapping: tuple[float, float, float, float] | None = None

    @property
    def n_rows(self) -> int:
        return max((len(c.values) for c in self.columns), default=0)

    def column(self, long_name: str) -> ImportColumn | None:
        return next((c for c in self.columns if c.long_name == long_name), None)


def clean_value(v):
    """A value as a worksheet cell: float, str or None."""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return float(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        f = float(v)
        return None if math.isnan(f) else f
    if isinstance(v, bytes):
        v = v.decode("utf-8", "replace")
    s = str(v)
    return s if s.strip() else None


def numeric_column(name: str, values, units: str = "", designation: str = "Y",
                   comments: str = "") -> ImportColumn:
    arr = np.asarray(values, dtype=float).ravel()
    return ImportColumn(name, [None if not math.isfinite(v) else float(v) for v in arr],
                        units, comments, designation)


def frame_table(df, name: str, first_x: bool = True) -> ImportedTable:
    """A pandas DataFrame as an :class:`ImportedTable` (column names → Long Names)."""
    cols = []
    for j, c in enumerate(df.columns):
        cols.append(ImportColumn(str(c), [clean_value(v) for v in df[c].tolist()],
                                 designation="X" if (j == 0 and first_x) else "Y"))
    return ImportedTable(name, cols)


# ---------------------------------------------------------------------- dispatch
FILTERS: dict[str, tuple[str, tuple[str, ...]]] = {
    "ascii": ("ASCII / CSV", (".csv", ".txt", ".dat", ".tsv", ".asc", ".prn")),
    "excel": ("Excel", (".xlsx", ".xlsm", ".xls")),
    "jcamp": ("JCAMP-DX", (".jdx", ".dx", ".jcm", ".jcamp")),
    "spc": ("Thermo Galactic SPC", (".spc",)),
    "matlab": ("MATLAB", (".mat",)),
    "hdf": ("HDF5 / NetCDF", (".h5", ".hdf5", ".hdf", ".he5", ".nc", ".cdf", ".nc4")),
    "wav": ("WAV audio", (".wav",)),
    "image": ("Image to matrix", (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif")),
}


def kind_of(path: str | Path) -> str:
    ext = Path(path).suffix.lower()
    for kind, (_, exts) in FILTERS.items():
        if ext in exts:
            return kind
    return "ascii"


def dialog_filter(kinds=None) -> str:
    """Qt file-dialog filter string for the given reader kinds (default: all)."""
    kinds = list(FILTERS) if kinds is None else kinds
    parts = []
    every = []
    for k in kinds:
        text, exts = FILTERS[k]
        pat = " ".join(f"*{e}" for e in exts)
        every.append(pat)
        parts.append(f"{text} ({pat})")
    return ";;".join([f"All supported ({' '.join(every)})"] + parts + ["All files (*)"])


def read_excel_all(path: str | Path, header: bool = True) -> list[ImportedTable]:
    """Every sheet of an Excel workbook (Origin's Import Excel, all sheets)."""
    import pandas as pd
    try:
        sheets = pd.read_excel(path, sheet_name=None, header=0 if header else None)
    except Exception as exc:  # noqa: BLE001 - reader errors become user messages
        raise ImportFailed(f"Cannot read the Excel file: {exc}") from None
    out = []
    for name, df in sheets.items():
        df = df.dropna(how="all").dropna(axis=1, how="all")
        if not header:
            df.columns = [f"Col{j + 1}" for j in range(df.shape[1])]
        out.append(frame_table(df, str(name)))
    return out


def read_file(path: str | Path, options=None, kind: str | None = None) -> list[ImportedTable]:
    """Read ``path`` with the reader for its extension (or ``kind``). ``options`` is an
    :class:`~.ascii_wizard.AsciiOptions` for text files."""
    kind = kind or kind_of(path)
    p = Path(path)
    if not p.exists():
        raise ImportFailed(f"{p} does not exist.")
    if kind == "ascii":
        from rsstatchem.core.analysis.importers import ascii_wizard
        return [ascii_wizard.read(p, options)]
    if kind == "excel":
        return read_excel_all(p)
    if kind == "jcamp":
        from rsstatchem.core.analysis.importers import jcamp
        return jcamp.tables(jcamp.read(p), p.stem)
    if kind == "spc":
        from rsstatchem.core.analysis.importers import spc
        return spc.tables(spc.read(p), p.stem)
    if kind == "matlab":
        from rsstatchem.core.analysis.importers import matlab
        return matlab.read(p)
    if kind == "hdf":
        from rsstatchem.core.analysis.importers import hdf
        return hdf.read(p)
    if kind == "wav":
        from rsstatchem.core.analysis.importers import media
        return [media.read_wav(p)]
    if kind == "image":
        from rsstatchem.core.analysis.importers import media
        return [media.read_image(p)]
    raise ImportFailed(f"No importer for {p.suffix} files.")


def arrays_tables(items, base: str, max_cols: int = 64) -> list[ImportedTable]:
    """Named arrays → tables, as Origin imports MATLAB / HDF5 variables: 1D arrays of the
    same length share a sheet (grouped by length), a 2D array with few columns becomes a sheet
    of its own, a large 2D array a matrix. ``items``: ``(name, array, units, long_name)``."""
    by_len: dict[int, list] = {}
    out: list[ImportedTable] = []
    for name, arr, units, long_name in items:
        a = np.asarray(arr)
        if a.dtype.kind not in "biufc" or a.size == 0:
            continue
        if a.dtype.kind == "c":
            a = np.abs(a)
        a = np.squeeze(a.astype(float))
        if a.ndim == 0:
            a = a.reshape(1)
        if a.ndim == 1:
            by_len.setdefault(len(a), []).append((name, a, units, long_name))
        elif a.ndim == 2:
            r, c = a.shape
            if min(r, c) <= max_cols and max(r, c) > min(r, c):
                if c > r:
                    a = a.T
                cols = [numeric_column(f"{long_name or name}[{j + 1}]", a[:, j], units)
                        for j in range(a.shape[1])]
                out.append(ImportedTable(str(name)[:31], cols, {"variable": name}))
            else:
                out.append(ImportedTable(str(name)[:31], [], {"variable": name},
                                         kind="matrix", matrix=a,
                                         mapping=(1.0, float(c), 1.0, float(r))))
        else:   # higher dimensions: one matrix per leading index (first 16)
            flat = a.reshape(-1, a.shape[-2], a.shape[-1])
            for k in range(min(len(flat), 16)):
                r, c = flat[k].shape
                out.append(ImportedTable(f"{name}_{k + 1}"[:31], [], {"variable": name},
                                         kind="matrix", matrix=flat[k],
                                         mapping=(1.0, float(c), 1.0, float(r))))
    for n, group in sorted(by_len.items(), key=lambda kv: -len(kv[1])):
        cols = [numeric_column(long_name or name, a, units) for name, a, units, long_name in
                group]
        cols[0].designation = "X" if len(cols) > 1 else "Y"
        label = base if len(by_len) == 1 else f"{base}_{n}"
        out.insert(0, ImportedTable(label[:31], cols, {"variables": [g[0] for g in group]}))
    return out

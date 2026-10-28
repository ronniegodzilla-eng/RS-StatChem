"""MATLAB ``.mat`` import: v4–v7.2 with ``scipy.io.loadmat``; v7.3 files are HDF5 and are
read through :mod:`.hdf` (needs the optional ``h5py``). Numeric variables become worksheet
columns (vectors of the same length share a sheet) or matrices; struct fields are flattened
to ``struct.field`` names."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from rsstatchem.core.analysis.importers import ImportedTable, ImportFailed, arrays_tables


def _flatten(name: str, value, out: list) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{name}.{k}", v, out)
        return
    if hasattr(value, "_fieldnames"):                       # mat_struct
        for k in value._fieldnames:
            _flatten(f"{name}.{k}", getattr(value, k), out)
        return
    a = np.asarray(value)
    if a.dtype == object:
        for i, v in enumerate(a.ravel()[:64]):
            _flatten(f"{name}{{{i + 1}}}", v, out)
        return
    if a.dtype.kind in "biufc":
        out.append((name, a, "", ""))


def read(path: str | Path) -> list[ImportedTable]:
    from scipy import io as sio
    p = Path(path)
    head = p.read_bytes()[:128]
    if b"MATLAB 7.3" in head or head[:8] == b"\x89HDF\r\n\x1a\n":
        from rsstatchem.core.analysis.importers import hdf
        return hdf.read(p)
    try:
        data = sio.loadmat(p, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        from rsstatchem.core.analysis.importers import hdf
        return hdf.read(p)
    except Exception as exc:  # noqa: BLE001 - reader errors become user messages
        raise ImportFailed(f"Cannot read the MATLAB file: {exc}") from None
    items: list = []
    for k, v in data.items():
        if not k.startswith("__"):
            _flatten(k, v, items)
    tables = arrays_tables(items, p.stem)
    if not tables:
        raise ImportFailed("The MATLAB file contains no numeric variables.")
    return tables

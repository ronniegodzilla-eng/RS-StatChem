"""HDF5 and NetCDF import. NetCDF 3 (classic / 64-bit offset) is read with
``scipy.io.netcdf_file``; HDF5 — including NetCDF 4 and MATLAB v7.3 files — needs the optional
``h5py`` package (:func:`have_h5py`). Numeric datasets become worksheet columns or matrices;
``units`` / ``long_name`` attributes fill the column's Units / Long Name, and NetCDF
coordinate variables become the X column."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from rsstatchem.core.analysis.importers import ImportedTable, ImportFailed, arrays_tables

HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


def have_h5py() -> bool:
    try:
        import h5py  # noqa: F401
    except ImportError:
        return False
    return True


def _attr(attrs, key: str) -> str:
    v = attrs.get(key, "") if hasattr(attrs, "get") else ""
    if isinstance(v, bytes):
        v = v.decode("utf-8", "replace")
    if isinstance(v, np.ndarray):
        v = v.tolist()
        v = v[0] if isinstance(v, list) and v else v
        if isinstance(v, bytes):
            v = v.decode("utf-8", "replace")
    return str(v) if v is not None else ""


def read_netcdf3(path: str | Path) -> list[ImportedTable]:
    from scipy.io import netcdf_file
    p = Path(path)
    try:
        f = netcdf_file(str(p), "r", mmap=False)
    except Exception as exc:  # noqa: BLE001
        raise ImportFailed(f"Cannot read the NetCDF file: {exc}") from None
    try:
        items = []
        coords = [n for n in f.variables if n in f.dimensions]
        for name in coords + [n for n in f.variables if n not in coords]:
            v = f.variables[name]
            a = np.array(v.data, copy=True)
            fill = getattr(v, "_FillValue", None)
            if fill is not None and a.dtype.kind in "fiu":
                a = np.where(a == fill, np.nan, a.astype(float))
            scale = getattr(v, "scale_factor", None)
            offset = getattr(v, "add_offset", None)
            if scale is not None or offset is not None:
                a = a.astype(float) * (scale if scale is not None else 1.0) + \
                    (offset if offset is not None else 0.0)
            units = getattr(v, "units", b"")
            long_name = getattr(v, "long_name", b"")
            items.append((name, a, units.decode() if isinstance(units, bytes) else str(units),
                          long_name.decode() if isinstance(long_name, bytes)
                          else str(long_name)))
        meta = {k: (v.decode() if isinstance(v, bytes) else v)
                for k, v in f._attributes.items()}
    finally:
        f.close()
    tables = arrays_tables(items, p.stem)
    for t in tables:
        t.meta["global_attributes"] = {k: str(v) for k, v in meta.items()}
    if not tables:
        raise ImportFailed("The NetCDF file contains no numeric variables.")
    return tables


def read_hdf5(path: str | Path, max_items: int = 200) -> list[ImportedTable]:
    if not have_h5py():
        raise ImportFailed("Reading HDF5 / NetCDF-4 / MATLAB v7.3 files needs the optional "
                           "h5py package, which is not installed.")
    import h5py
    p = Path(path)
    items = []
    try:
        with h5py.File(p, "r") as f:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset) and len(items) < max_items and \
                        obj.dtype.kind in "biufc" and obj.size <= 50_000_000:
                    a = obj[()]
                    if "MATLAB_class" in obj.attrs and np.ndim(a) == 2:
                        a = np.asarray(a).T     # MATLAB stores column-major
                    items.append((name.replace("/", "."), a, _attr(obj.attrs, "units"),
                                  _attr(obj.attrs, "long_name")))
            f.visititems(visit)
    except OSError as exc:
        raise ImportFailed(f"Cannot read the HDF5 file: {exc}") from None
    tables = arrays_tables(items, p.stem)
    if not tables:
        raise ImportFailed("The HDF5 file contains no numeric datasets.")
    return tables


def read(path: str | Path) -> list[ImportedTable]:
    head = Path(path).read_bytes()[:520]
    if head[:3] == b"CDF":
        return read_netcdf3(path)
    if head[:8] == HDF5_MAGIC or head[512:520] == HDF5_MAGIC:
        return read_hdf5(path)
    raise ImportFailed(f"{Path(path).name} is neither an HDF5 nor a NetCDF file.")

"""RS-StatChem Image Analysis projects (``.rsij``).

A project is a zip archive whose parts stay readable by ImageJ itself::

    manifest.json        format/version, images (file, title, position, zoom/window state),
                         active image, tables, ROI Manager, options snapshot
    images/NN.tif        each open image as an ImageJ TIFF (calibration, LUTs, display ranges,
                         labels, Info, properties, selection and overlay)
    rois/RoiSet.zip      the ROI Manager in ImageJ's RoiSet format
    results/*.csv        Results and other tables (full precision, ImageJ CSV layout)
    log.txt              the Log window

:func:`save_project` / :func:`read_project` work on a :class:`~rsstatchem.core.imaging.session.
Session`; :func:`load_project` adds what was read to a session. The UI adds window geometry
and zoom through ``window_state`` (``{image id: {...}}``) and gets it back per image.
"""
from __future__ import annotations

import datetime as _dt
import io
import json
import math
import re
import zipfile
from pathlib import Path

from rsstatchem.core.imaging.results import ResultsTable

FORMAT = "rsij"
VERSION = 1
EXTENSION = ".rsij"


def _safe(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip() or "image"
    return name[:60]


def _table_csv(table: ResultsTable) -> str:
    """CSV of a table with every value at full precision (ImageJ-readable)."""
    heads = table.headings
    out = io.StringIO()
    out.write(",".join(heads) + "\n")
    for r in range(len(table)):
        cells = []
        for h in heads:
            v = table.get_value(h, r) if h != "Label" else table.get_label(r)
            if v is None:
                s = ""
            elif isinstance(v, str):
                s = v.replace(",", ";") if h == "Label" else (f'"{v}"' if "," in v else v)
            elif math.isnan(v):
                s = "NaN"
            elif math.isinf(v):
                s = "Infinity" if v > 0 else "-Infinity"
            elif float(v).is_integer() and abs(v) < 2 ** 53:
                s = str(int(v))
            else:
                s = repr(float(v))
            cells.append(s)
        out.write(",".join(cells) + "\n")
    return out.getvalue()


def save_project(session, path, window_state: dict | None = None,
                 little_endian: bool = False) -> Path:
    from rsstatchem.core.imaging.io.roi_codec import write_roi_zip
    from rsstatchem.core.imaging.io.tiff import tiff_bytes

    path = Path(path)
    if path.suffix.lower() != EXTENSION:
        path = path.with_suffix(EXTENSION)
    window_state = window_state or {}
    manifest = {"format": FORMAT, "version": VERSION, "app": "RS-StatChem Image Analysis",
                "created": _dt.datetime.now().isoformat(timespec="seconds"),
                "images": [], "active": None, "tables": [], "roi_manager": None,
                "log": None, "options": session.options.to_dict()}
    tmp = path.with_name(path.name + ".part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, doc in enumerate(session.images, start=1):
            name = f"images/{i:02d}_{_safe(Path(doc.title).stem)}.tif"
            zf.writestr(name, tiff_bytes(doc, little_endian))
            entry = {"file": name, "title": doc.title, "position": list(doc.position),
                     "path": doc.path, "changed": bool(doc.changed),
                     "composite_mode": doc.composite_mode,
                     "active_channels": list(doc.active_channels)}
            if doc.id in window_state:
                entry["window"] = window_state[doc.id]
            manifest["images"].append(entry)
            if doc is session.active:
                manifest["active"] = i - 1
        if session.roi_manager:
            buf = io.BytesIO()
            write_roi_zip(session.roi_manager, buf)
            zf.writestr("rois/RoiSet.zip", buf.getvalue())
            manifest["roi_manager"] = "rois/RoiSet.zip"
        used = set()
        for title, table in session.tables.items():
            if len(table) == 0 and not table.headings:
                continue
            base = _safe(title)
            name = base
            k = 1
            while name in used:
                name = f"{base}-{k}"
                k += 1
            used.add(name)
            fname = f"results/{name}.csv"
            zf.writestr(fname, _table_csv(table))
            manifest["tables"].append({"title": title, "file": fname,
                                       "results": table is session.results or
                                       title == "Results",
                                       "precision": table.precision})
        if session.log_lines:
            zf.writestr("log.txt", "\n".join(session.log_lines) + "\n")
            manifest["log"] = "log.txt"
        zf.writestr("manifest.json", json.dumps(manifest, indent=1))
    tmp.replace(path)
    return path


def read_project(path) -> dict:
    """{"manifest", "images" [ImageDoc], "window_state" [dict|None], "active" (index|None),
    "rois" [Roi], "tables" [ResultsTable], "log" [str], "options" dict}."""
    from rsstatchem.core.imaging.io.roi_codec import read_roi_zip
    from rsstatchem.core.imaging.io.tiff import read_tiff_bytes

    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        try:
            manifest = json.loads(zf.read("manifest.json"))
        except KeyError:
            raise ValueError(f"{path.name} is not an RS-StatChem image project") from None
        if manifest.get("format") != FORMAT:
            raise ValueError(f"{path.name} is not an RS-StatChem image project")
        if int(manifest.get("version", 1)) > VERSION:
            raise ValueError(f"{path.name} was written by a newer RS-StatChem "
                             f"(project version {manifest['version']})")
        images, states = [], []
        for entry in manifest.get("images", []):
            doc = read_tiff_bytes(zf.read(entry["file"]), entry.get("title", "Untitled"))
            doc.title = entry.get("title", doc.title)
            doc.path = entry.get("path")
            pos = entry.get("position") or [0, 0, 0]
            doc.set_position(*pos)
            if entry.get("composite_mode") in ("composite", "color", "grayscale") and \
                    doc.is_composite:
                doc.composite_mode = entry["composite_mode"]
            ac = entry.get("active_channels")
            if ac and len(ac) == doc.n_channels:
                doc.active_channels = [bool(v) for v in ac]
            doc.changed = bool(entry.get("changed", False))
            images.append(doc)
            states.append(entry.get("window"))
        rois = read_roi_zip(zf.read(manifest["roi_manager"])) if manifest.get("roi_manager") \
            else []
        tables = []
        for t in manifest.get("tables", []):
            text = zf.read(t["file"]).decode("utf-8")
            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / "t.csv"
                p.write_text(text, encoding="utf-8")
                table = ResultsTable.read(p, title=t["title"])
            if t.get("results"):
                rt = ResultsTable.results(t["title"])
                for r in range(len(table)):
                    rt.increment_counter()
                    for h in table.headings:
                        if h == "Label":
                            rt.set_label(table.get_label(r), r)
                        else:
                            rt.set_value(h, r, table.get_value(h, r))
                table = rt
            table.precision = int(t.get("precision", 3))
            tables.append(table)
        log = zf.read(manifest["log"]).decode("utf-8").splitlines() if manifest.get("log") \
            else []
    return {"manifest": manifest, "images": images, "window_state": states,
            "active": manifest.get("active"), "rois": rois, "tables": tables, "log": log,
            "options": manifest.get("options") or {}}


def load_project(session, path, apply_options: bool = False) -> dict:
    """Open a project into ``session`` (images added, ROI Manager/tables/log replaced)."""
    data = read_project(path)
    if apply_options and data["options"]:
        from rsstatchem.core.imaging.options import Options
        session.options = Options.from_dict(data["options"])
    for doc in data["images"]:
        session.add_image(doc, activate=False)
    if data["rois"]:
        session.roi_manager[:] = data["rois"]
    for table in data["tables"]:
        session.show_table(table)
    for line in data["log"]:
        session.log(line)
    act = data["active"]
    if act is not None and 0 <= act < len(data["images"]):
        session.set_active(data["images"][act])
    elif data["images"]:
        session.set_active(data["images"][-1])
    return data

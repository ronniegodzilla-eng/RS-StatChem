"""Save and reopen Output Viewer documents (SPSS ``.spv`` equivalent): ``.rso`` files.

An ``.rso`` file is a zip archive with ``output.json`` (sections, tables, notes, logs and
image references, plus the viewer's TableLook) and one PNG per chart under ``images/``. It is
plain JSON + PNG so other tools can read it. Item ids are reassigned on load so a reopened
document can be appended to an existing viewer.
"""
from __future__ import annotations

import json
import zipfile
from datetime import datetime
from pathlib import Path

from rsstatchem import __version__
from rsstatchem.core.output import Image, OutputDocument, Section, Table, Text
from rsstatchem.core.pivot import table_from_dict, table_to_dict

OUTPUT_EXT = ".rso"
OUTPUT_FORMAT = "rs-statchem-output"
OUTPUT_FILTER = "RS-StatChem output (*.rso)"


def _item_dict(item, images: dict) -> dict:
    if isinstance(item, Table):
        return table_to_dict(item)
    if isinstance(item, Text):
        return {"kind": "text", "title": item.title, "text": item.text, "style": item.style}
    if isinstance(item, Image):
        name = f"images/{len(images) + 1:04d}.png"
        images[name] = item.png
        return {"kind": "image", "title": item.title, "width": item.width, "png": name}
    raise TypeError(f"Unknown output item {type(item).__name__}")


def to_dict(doc: OutputDocument, look: str | None = None) -> tuple[dict, dict]:
    """(JSON document, {archive name: png bytes})."""
    images: dict[str, bytes] = {}
    sections = [{"title": s.title, "created": s.created.isoformat(timespec="seconds"),
                 "items": [_item_dict(i, images) for i in s.items]} for s in doc.sections]
    data = {"format": OUTPUT_FORMAT, "version": 1, "app_version": __version__,
            "look": look, "sections": sections}
    return data, images


def from_dict(data: dict, images: dict) -> OutputDocument:
    if data.get("format") != OUTPUT_FORMAT:
        raise ValueError("This file is not an RS-StatChem output document.")
    doc = OutputDocument()
    for s in data.get("sections", []):
        try:
            created = datetime.fromisoformat(s.get("created", ""))
        except ValueError:
            created = datetime.now()
        section = Section(s.get("title", ""), created=created)
        for d in s.get("items", []):
            kind = d.get("kind")
            if kind == "table":
                section.items.append(table_from_dict(d))
            elif kind == "text":
                section.items.append(Text(d.get("title", ""), d.get("text", ""),
                                          d.get("style", "note")))
            elif kind == "image":
                png = images.get(d.get("png", ""))
                if png:
                    section.items.append(Image(d.get("title", ""), png,
                                               int(d.get("width", 560))))
        doc.add(section)
    return doc


def save_output(doc: OutputDocument, path: str | Path, look: str | None = None) -> None:
    data, images = to_dict(doc, look)
    tmp = Path(str(path) + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("output.json", json.dumps(data, ensure_ascii=False, indent=1))
        for name, png in images.items():
            zf.writestr(name, png, compress_type=zipfile.ZIP_STORED)
    tmp.replace(path)


def load_output(path: str | Path) -> tuple[OutputDocument, str | None]:
    """(document, TableLook stored with it)."""
    with zipfile.ZipFile(path) as zf:
        try:
            data = json.loads(zf.read("output.json").decode("utf-8"))
        except KeyError:
            raise ValueError("This file is not an RS-StatChem output document.") from None
        images = {n: zf.read(n) for n in zf.namelist() if n.startswith("images/")}
    return from_dict(data, images), data.get("look")

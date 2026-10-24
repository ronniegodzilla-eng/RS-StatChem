"""Exports of a drawing: SVG, PNG (any dpi, transparent), PDF, clipboard (image + SMILES +
MOL + SVG + native drawing) and the "Send to Output Viewer" section.

Everything renders through :class:`~rsstatchem.ui.chem_editor.render.Builder` into a fresh
scene without selection/hover marks, so exports match the canvas exactly.
"""
from __future__ import annotations

from PySide6.QtCore import (QBuffer, QByteArray, QIODevice, QMarginsF, QMimeData, QRectF,
                            QSize, QSizeF, Qt)
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPageSize, QPainter, QPdfWriter
from PySide6.QtWidgets import QGraphicsScene

from rsstatchem.core.chem_draw import convert
from rsstatchem.core.chem_draw.model import Document
from rsstatchem.ui.chem_editor.render import Builder

__all__ = ["render_scene", "content_rect", "svg_text", "export_svg", "png_bytes",
           "export_png", "export_pdf", "mime_data", "copy_to_clipboard", "output_section",
           "NATIVE_MIME"]

NATIVE_MIME = "application/x-rsstatchem-drawing"
MOL_MIME = "chemical/x-mdl-molfile"


def render_scene(doc: Document) -> tuple[QGraphicsScene, QRectF]:
    scene = QGraphicsScene()
    Builder(doc.style).build(scene, doc)
    return scene, content_rect(scene, doc)


def content_rect(scene: QGraphicsScene, doc: Document) -> QRectF:
    r = scene.itemsBoundingRect()
    m = max(4.0, doc.style.bond_length * 0.25)
    if r.isNull() or r.isEmpty():
        r = QRectF(0, 0, 10, 10)
    return r.adjusted(-m, -m, m, m)


def _render(doc: Document, painter: QPainter, target: QRectF) -> None:
    scene, rect = render_scene(doc)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    scene.render(painter, target, rect, Qt.AspectRatioMode.KeepAspectRatio)


def svg_text(doc: Document, title: str = "Chemical drawing") -> bytes:
    from PySide6.QtSvg import QSvgGenerator
    scene, rect = render_scene(doc)
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    gen = QSvgGenerator()
    gen.setOutputDevice(buf)
    gen.setSize(QSize(int(round(rect.width())), int(round(rect.height()))))
    gen.setViewBox(QRectF(0, 0, rect.width(), rect.height()))
    gen.setResolution(72)
    gen.setTitle(title)
    gen.setDescription("RS-StatChem structure drawing")
    p = QPainter(gen)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    scene.render(p, QRectF(0, 0, rect.width(), rect.height()), rect)
    p.end()
    buf.close()
    return bytes(buf.data())


def export_svg(doc: Document, path: str) -> None:
    with open(path, "wb") as fh:
        fh.write(svg_text(doc))


def image(doc: Document, dpi: float = 300, transparent: bool = True) -> QImage:
    scene, rect = render_scene(doc)
    k = dpi / 72.0
    img = QImage(max(1, int(round(rect.width() * k))), max(1, int(round(rect.height() * k))),
                 QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(QColor(0, 0, 0, 0) if transparent else QColor("#FFFFFF"))
    dpm = int(round(dpi / 0.0254))
    img.setDotsPerMeterX(dpm)
    img.setDotsPerMeterY(dpm)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    scene.render(p, QRectF(0, 0, img.width(), img.height()), rect)
    p.end()
    return img


def png_bytes(doc: Document, dpi: float = 300, transparent: bool = True) -> bytes:
    img = image(doc, dpi, transparent)
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG")
    buf.close()
    return bytes(ba)


def export_png(doc: Document, path: str, dpi: float = 300, transparent: bool = True) -> None:
    if not image(doc, dpi, transparent).save(path, "PNG"):
        raise OSError(f"Could not write {path}")


def export_pdf(doc: Document, path: str) -> None:
    scene, rect = render_scene(doc)
    writer = QPdfWriter(path)
    writer.setPageSize(QPageSize(QSizeF(rect.width(), rect.height()), QPageSize.Unit.Point,
                                 "Drawing"))
    writer.setPageMargins(QMarginsF(0, 0, 0, 0))
    writer.setResolution(1200)
    writer.setTitle("Chemical drawing")
    writer.setCreator("RS-StatChem")
    p = QPainter(writer)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    k = 1200 / 72.0
    scene.render(p, QRectF(0, 0, rect.width() * k, rect.height() * k), rect)
    p.end()


def mime_data(doc: Document) -> QMimeData:
    md = QMimeData()
    md.setImageData(image(doc, 300, transparent=False))
    md.setData("image/svg+xml", QByteArray(svg_text(doc)))
    md.setData(NATIVE_MIME, QByteArray(doc.to_json().encode("utf-8")))
    text = ""
    try:
        if doc.arrows and convert.split_reaction(doc).ok:
            text = convert.reaction_smiles(doc)
        elif doc.atoms:
            text = convert.to_smiles(doc)
            md.setData(MOL_MIME, QByteArray(convert.to_molblock(doc).encode()))
    except Exception:  # noqa: BLE001 - an invalid structure still copies as a picture
        text = ""
    if text:
        md.setText(text)
    return md


def copy_to_clipboard(doc: Document) -> None:
    QGuiApplication.clipboard().setMimeData(mime_data(doc))


def output_section(doc: Document, title: str = "Chemical Drawing"):
    """Output Viewer section: the drawing plus identifiers/properties (molecules) or the
    reaction SMILES and balance table (reactions)."""
    from rsstatchem.core.chem import sections
    from rsstatchem.core.chem_draw import analysis
    from rsstatchem.core.output import Section, Table

    section = Section(title)
    split = convert.split_reaction(doc) if doc.arrows else None
    if split is not None and split.ok:
        try:
            rsmi = convert.reaction_smiles(doc, split)
        except Exception as exc:  # noqa: BLE001
            rsmi = f"(invalid: {exc})"
        section.log(f"CHEM DRAW REACTION '{rsmi}'.")
    else:
        try:
            smi = convert.to_smiles(doc) if doc.atoms else ""
        except Exception:  # noqa: BLE001
            smi = ""
        section.log(f"CHEM DRAW '{smi}'.")
    scene, rect = render_scene(doc)
    width = int(min(900, max(120, rect.width() * 96 / 72)))
    section.image(title, png_bytes(doc, dpi=192, transparent=False), width=width)
    if split is not None and split.ok:
        section.table(Table("Reaction", ["", "Value"], [["Reaction SMILES", rsmi]]))
        bc = analysis.balance_check(doc)
        if bc.ok:
            rows = [[e, str(lft), str(rgt), "Yes" if lft == rgt else "No"]
                    for e, lft, rgt in bc.rows]
            section.table(Table("Atom Balance", ["Element", "Reactants", "Products",
                                                 "Balanced"], rows, footnotes=[bc.message]))
        return section
    items = []
    for i, (comp, m) in enumerate(convert.fragment_mols(doc)):
        if isinstance(m, Exception):
            section.note("Structure problem", str(m), style="warning")
            continue
        items.append((f"Molecule {i + 1}" if len(doc.fragments()) > 1 else "", m))
    if items:
        props = sections.molecule_section(items, image=False, rules=["Lipinski"])
        section.items.extend(it for it in props.items
                             if getattr(it, "style", "") != "log")
    return section

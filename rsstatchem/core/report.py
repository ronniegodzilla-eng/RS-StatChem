"""Render an :class:`OutputDocument` to HTML (viewer, HTML export) and to Word (.docx)."""
from __future__ import annotations

import base64
import html
import io
from typing import Callable

from rsstatchem import APP_NAME, __version__
from rsstatchem.core.output import Image, OutputDocument, Section, Table, Text

ACCENT = "#B71C1C"

CSS = f"""
body {{ font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif; font-size: 10pt;
        color: #1b1b1b; background: #ffffff; }}
h1.section {{ font-size: 14pt; color: {ACCENT}; margin: 18px 0 6px 0; font-weight: 600; }}
p.caption {{ font-size: 10.5pt; font-weight: 600; margin: 12px 0 4px 0; }}
table.rs {{ border-collapse: collapse; margin: 2px 0 4px 0; }}
table.rs th, table.rs td {{ padding: 3px 10px; font-size: 9.5pt; }}
table.rs th {{ font-weight: 600; background: #fbf3f3; white-space: nowrap; }}
table.rs td {{ white-space: nowrap; }}
table.rs td.rh {{ text-align: left; font-weight: 500; }}
p.foot {{ font-size: 8.5pt; color: #444; margin: 1px 0; }}
pre.log {{ font-family: Menlo, Consolas, monospace; font-size: 8.5pt; color: #555;
           background: #f7f7f7; border-left: 3px solid {ACCENT}; padding: 4px 8px; }}
p.note {{ margin: 4px 0; }}
p.warning {{ margin: 4px 0; color: #8a5a00; background: #fff7e0; padding: 4px 8px; }}
"""


def _esc(value) -> str:
    return html.escape(str(value))


def _looks_numeric(text: str) -> bool:
    t = text.strip().replace(",", "").replace("%", "")
    if t in ("", ".", "-"):
        return False
    try:
        float(t.lstrip("<>"))
        return True
    except ValueError:
        return False


# Borders are inline styles: Qt's rich-text engine ignores :first-child / :last-child.
_TOP = "border-top: 2px solid #333;"
_UNDER_HEADER = "border-bottom: 1px solid #555;"
_BOTTOM = "border-bottom: 2px solid #333;"


def table_html(table: Table, anchor: str = "") -> str:
    out = [f'<a name="{anchor}"></a>' if anchor else "",
           f'<p class="caption">{_esc(table.title)}</p>',
           '<table class="rs" cellspacing="0">', "<thead>"]
    rh = table.row_header_cols
    if table.has_groups:
        top, bottom = [], []
        cols = list(table.columns)
        i = 0
        while i < len(cols):
            c = cols[i]
            if isinstance(c, tuple):
                group = c[0]
                span = 1
                while i + span < len(cols) and isinstance(cols[i + span], tuple) \
                        and cols[i + span][0] == group:
                    span += 1
                top.append(f'<th colspan="{span}" align="center" style="{_TOP} white-space: normal;">'
                           f'{_esc(group)}</th>')
                bottom.extend(f'<th align="center" style="{_UNDER_HEADER}">'
                              f'{_esc(cols[j][1])}</th>' for j in range(i, i + span))
                i += span
            else:
                top.append(f'<th rowspan="2" align="{"left" if i < rh else "center"}" '
                           f'style="{_TOP}{_UNDER_HEADER}">{_esc(c)}</th>')
                i += 1
        out.append("<tr>" + "".join(top) + "</tr>")
        out.append("<tr>" + "".join(bottom) + "</tr>")
    else:
        out.append("<tr>" + "".join(
            f'<th align="{"left" if i < rh else "center"}" style="{_TOP}{_UNDER_HEADER}">'
            f'{_esc(c)}</th>' for i, c in enumerate(table.columns)) + "</tr>")
    out.append("</thead><tbody>")
    for r, row in enumerate(table.rows):
        style = f' style="{_BOTTOM}"' if r == len(table.rows) - 1 else ""
        cells = []
        for i, cell in enumerate(row):
            text = "" if cell is None else str(cell)
            if i < rh:
                cells.append(f'<td class="rh" align="left"{style}>{_esc(text)}</td>')
            else:
                align = "right" if _looks_numeric(text) else "center"
                cells.append(f'<td class="num" align="{align}"{style}>{_esc(text)}</td>')
        out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</tbody></table>")
    for note in table.footnotes:
        out.append(f'<p class="foot">{_esc(note)}</p>')
    return "\n".join(out)


def _png_data_uri(img: Image) -> str:
    return "data:image/png;base64," + base64.b64encode(img.png).decode("ascii")


def section_html(section: Section, image_src: Callable[[Image], str] = _png_data_uri) -> str:
    parts = [f'<a name="s{section.id}"></a><h1 class="section">{_esc(section.title)}</h1>']
    for item in section.items:
        anchor = f"i{item.id}"
        if isinstance(item, Table):
            parts.append(table_html(item, anchor))
        elif isinstance(item, Text):
            if item.style == "log":
                parts.append(f'<a name="{anchor}"></a><pre class="log">{_esc(item.text)}</pre>')
            else:
                title = f'<p class="caption">{_esc(item.title)}</p>' if item.title else ""
                parts.append(f'<a name="{anchor}"></a>{title}'
                             f'<p class="{item.style}">'
                             f'{_esc(item.text).replace(chr(10), "<br>")}</p>')
        elif isinstance(item, Image):
            parts.append(f'<a name="{anchor}"></a><p class="caption">{_esc(item.title)}</p>'
                         f'<p><img src="{image_src(item)}" width="{item.width}"></p>')
    return "\n".join(parts)


def document_html(doc: OutputDocument, image_src: Callable[[Image], str] = _png_data_uri,
                  standalone: bool = True) -> str:
    body = "\n<br>\n".join(section_html(s, image_src) for s in doc.sections)
    if not standalone:
        return f"<html><head><style>{CSS}</style></head><body>{body}</body></html>"
    return ("<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{APP_NAME} Output</title><style>{CSS}</style></head>"
            f"<body>{body}</body></html>")


# ---------------------------------------------------------------- Word export
def _set_cell_border(cell, **edges) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge, size in edges.items():
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single" if size else "nil")
        el.set(qn("w:sz"), str(size))
        el.set(qn("w:color"), "000000")
        borders.append(el)


def _docx_table(document, table: Table) -> None:
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt

    cap = document.add_paragraph()
    run = cap.add_run(table.title)
    run.bold = True
    run.font.size = Pt(10.5)

    header_rows = 2 if table.has_groups else 1
    n_cols = len(table.columns)
    t = document.add_table(rows=header_rows + len(table.rows), cols=n_cols)
    t.autofit = True

    if table.has_groups:
        i = 0
        while i < n_cols:
            c = table.columns[i]
            if isinstance(c, tuple):
                span = 1
                while i + span < n_cols and isinstance(table.columns[i + span], tuple) \
                        and table.columns[i + span][0] == c[0]:
                    span += 1
                top = t.cell(0, i).merge(t.cell(0, i + span - 1)) if span > 1 else t.cell(0, i)
                top.text = c[0]
                for j in range(i, i + span):
                    t.cell(1, j).text = table.columns[j][1]
                i += span
            else:
                merged = t.cell(0, i).merge(t.cell(1, i))
                merged.text = str(c)
                i += 1
    else:
        for i, c in enumerate(table.columns):
            t.cell(0, i).text = str(c)

    for r, row in enumerate(table.rows):
        for c, value in enumerate(row):
            t.cell(header_rows + r, c).text = "" if value is None else str(value)

    last = header_rows + len(table.rows) - 1
    for r, row in enumerate(t.rows):
        for c, cell in enumerate(row.cells):
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(0)
                if c >= table.row_header_cols:
                    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if r >= header_rows \
                        else WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.font.size = Pt(9)
                    run.bold = r < header_rows
            edges = {"left": 0, "right": 0}
            if r == 0:
                edges["top"] = 12
            if r == header_rows - 1:
                edges["bottom"] = 6
            if r == last:
                edges["bottom"] = 12
            _set_cell_border(cell, **edges)

    for note in table.footnotes:
        p = document.add_paragraph()
        run = p.add_run(note)
        run.font.size = Pt(8)
        run.italic = True
    document.add_paragraph()


def export_docx(doc: OutputDocument, path: str) -> None:
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor

    document = Document()
    style = document.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10)
    for section in doc.sections:
        heading = document.add_heading(section.title, level=1)
        for run in heading.runs:
            run.font.color.rgb = RGBColor(0xB7, 0x1C, 0x1C)
        for item in section.items:
            if isinstance(item, Table):
                _docx_table(document, item)
            elif isinstance(item, Text):
                if item.style == "log":
                    continue
                if item.title:
                    document.add_paragraph().add_run(item.title).bold = True
                document.add_paragraph(item.text)
            elif isinstance(item, Image):
                document.add_paragraph().add_run(item.title).bold = True
                document.add_picture(io.BytesIO(item.png), width=Inches(min(item.width / 96, 6.0)))
    core = document.core_properties
    core.comments = f"Generated by {APP_NAME} {__version__}"
    document.save(path)

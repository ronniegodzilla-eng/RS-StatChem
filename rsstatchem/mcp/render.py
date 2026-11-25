"""Output sections as text for an MCP client: Markdown (read by the model) or JSON.

Charts are not sent; a line names each one that was left out. Table cells stay the formatted
strings the Output Viewer shows, so numbers match the app digit for digit.
"""
from __future__ import annotations

import json

from rsstatchem.core.output import Image, Section, Table, Text

FORMATS = ("markdown", "json")


def column_names(table: Table) -> list[str]:
    """One name per column; a two-level header becomes "Group / Sub"."""
    names = []
    for col in table.columns:
        if isinstance(col, tuple):
            parts = [str(p).strip() for p in col if str(p).strip()]
            names.append(" / ".join(dict.fromkeys(parts)))
        else:
            names.append(str(col).strip())
    return names


def _cell(value) -> str:
    text = "" if value is None else str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\r", "").replace("\n", "<br>")


def table_markdown(table: Table) -> str:
    names = column_names(table)
    width = max([len(names)] + [len(r) for r in table.rows])
    names += [""] * (width - len(names))
    lines = []
    if table.title:
        lines += [f"**{table.title}**", ""]
    lines.append("| " + " | ".join(_cell(n) for n in names) + " |")
    lines.append("|" + "|".join(["---"] * width) + "|")
    for row in table.rows:
        cells = [_cell(c) for c in row] + [""] * (width - len(row))
        lines.append("| " + " | ".join(cells) + " |")
    if table.footnotes:
        lines.append("")
        lines += [str(f) + "  " for f in table.footnotes]
    return "\n".join(lines).rstrip()


def section_markdown(section: Section) -> str:
    parts = [f"## {section.title}"] if section.title else []
    for item in section.items:
        if isinstance(item, Table):
            parts.append(table_markdown(item))
        elif isinstance(item, Image):
            parts.append(f"*[Chart not sent: {item.title}]*")
        elif isinstance(item, Text):
            if item.style == "log":
                parts.append("```\n" + item.text.strip() + "\n```")
            else:
                head = f"**{item.title}**" if item.title else ""
                mark = "Warning: " if item.style == "warning" else ""
                parts.append("\n\n".join(p for p in (head, mark + item.text.strip()) if p.strip()))
    return "\n\n".join(p for p in parts if p)


def section_data(section: Section) -> dict:
    items = []
    for item in section.items:
        if isinstance(item, Table):
            items.append({"kind": "table", "title": item.title, "columns": column_names(item),
                          "row_header_columns": item.row_header_cols,
                          "rows": [[str(c) for c in row] for row in item.rows],
                          "footnotes": [str(f) for f in item.footnotes]})
        elif isinstance(item, Image):
            items.append({"kind": "chart", "title": item.title, "sent": False})
        elif isinstance(item, Text):
            items.append({"kind": "syntax" if item.style == "log" else item.style,
                          "title": item.title, "text": item.text})
    return {"title": section.title, "items": items}


def render(section: Section, output_format: str = "markdown") -> str:
    if output_format not in FORMATS:
        raise ValueError(f"output_format must be one of {', '.join(FORMATS)}; "
                         f"got {output_format!r}.")
    if output_format == "json":
        return json.dumps(section_data(section), ensure_ascii=False)
    return section_markdown(section)

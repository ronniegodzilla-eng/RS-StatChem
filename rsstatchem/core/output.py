"""Output document model shown in the Output Viewer (SPSS-style).

An analysis produces one :class:`Section` (e.g. "Descriptives") that holds an ordered list of
items: notes/logs, tables and images. Cells in tables are already-formatted strings so each
analysis controls its own number formatting.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime

_ids = itertools.count(1)


def _next_id() -> int:
    return next(_ids)


@dataclass
class Table:
    title: str
    columns: list  # str, or (group, sub) tuple for a two-level header
    rows: list[list[str]]
    row_header_cols: int = 1
    footnotes: list[str] = field(default_factory=list)
    id: int = field(default_factory=_next_id)
    kind: str = "table"

    @property
    def has_groups(self) -> bool:
        return any(isinstance(c, tuple) for c in self.columns)


@dataclass
class Text:
    title: str
    text: str
    style: str = "note"  # note | log | warning
    id: int = field(default_factory=_next_id)
    kind: str = "text"


@dataclass
class Image:
    title: str
    png: bytes
    width: int = 560
    id: int = field(default_factory=_next_id)
    kind: str = "image"


@dataclass
class Section:
    title: str
    items: list = field(default_factory=list)
    created: datetime = field(default_factory=datetime.now)
    id: int = field(default_factory=_next_id)

    def log(self, text: str) -> "Section":
        self.items.append(Text("Log", text, style="log"))
        return self

    def note(self, title: str, text: str, style: str = "note") -> "Section":
        self.items.append(Text(title, text, style=style))
        return self

    def table(self, table: Table) -> "Section":
        self.items.append(table)
        return self

    def image(self, title: str, png: bytes, width: int = 560) -> "Section":
        self.items.append(Image(title, png, width))
        return self


@dataclass
class OutputDocument:
    sections: list[Section] = field(default_factory=list)

    def add(self, section: Section) -> None:
        self.sections.append(section)

    def remove(self, section_id: int) -> None:
        self.sections = [s for s in self.sections if s.id != section_id]

    def remove_item(self, item_id: int) -> None:
        for s in self.sections:
            s.items = [i for i in s.items if i.id != item_id]

    def clear(self) -> None:
        self.sections.clear()

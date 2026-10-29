"""The small text format of the guide's chapters (``docs/user_guide/content/*.md``).

Blocks (separated by blank lines)::

    # Chapter title                 new chapter (new page, chapter opener)
    ## Section / ### Subsection
    Plain paragraph text, lines are joined.
    - bullet item                   (indent two spaces to continue an item)
    1. numbered step                (a list may start at any number, e.g. 4.)
    > [!TIP] text                   callout: TIP (Tips), NOTE (Catatan), WARNING (Perhatian)
    ![stats.data_view]              screenshot scene (caption from the scene)
    ![stats.data_view](Caption)     ... with its own caption
    !features stats                 the generated feature reference of a module
    | A | B |                       table; the first row is the header,
    |---|---|                       followed by a separator row
    ---pagebreak---

Inline: ``**bold**``, ``*italic*``, ```code```, ``[[File > Open]]`` (menu path),
``++Ctrl+S++`` (keyboard keys).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from html import escape
from pathlib import Path


@dataclass
class Block:
    kind: str                       # h1 h2 h3 p ul ol callout figure features table pagebreak
    text: str = ""
    items: list = field(default_factory=list)
    arg: str = ""                   # callout type, figure key, features module
    source: str = ""                # file:line, for error messages


CALLOUTS = {"TIP": "Tips", "NOTE": "Catatan", "WARNING": "Perhatian"}


def parse(text: str, source: str = "") -> list[Block]:
    lines = text.splitlines()
    blocks: list[Block] = []
    i = 0
    para: list[str] = []
    para_line = 0

    def flush():
        nonlocal para
        if para:
            blocks.append(Block("p", " ".join(s.strip() for s in para),
                                source=f"{source}:{para_line}"))
            para = []

    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        where = f"{source}:{i + 1}"
        if not line.strip():
            flush()
            i += 1
            continue
        if line.startswith("<!--"):
            flush()
            while i < len(lines) and "-->" not in lines[i]:
                i += 1
            i += 1
            continue
        m = re.match(r"^(#{1,3})\s+(.*)$", line)
        if m:
            flush()
            blocks.append(Block(f"h{len(m.group(1))}", m.group(2).strip(), source=where))
            i += 1
            continue
        if line.strip() == "---pagebreak---":
            flush()
            blocks.append(Block("pagebreak", source=where))
            i += 1
            continue
        m = re.match(r"^!\[([\w.\-]+)\](?:\((.*)\))?\s*$", line.strip())
        if m:
            flush()
            blocks.append(Block("figure", m.group(2) or "", arg=m.group(1), source=where))
            i += 1
            continue
        m = re.match(r"^!features\s+(\w+)\s*$", line.strip())
        if m:
            flush()
            blocks.append(Block("features", arg=m.group(1), source=where))
            i += 1
            continue
        m = re.match(r"^>\s*\[!(\w+)\]\s*(.*)$", line)
        if m:
            flush()
            kind = m.group(1).upper()
            if kind not in CALLOUTS:
                raise ValueError(f"{where}: unknown callout [!{kind}]")
            body = [m.group(2)]
            i += 1
            while i < len(lines) and lines[i].startswith(">"):
                body.append(lines[i][1:].strip())
                i += 1
            blocks.append(Block("callout", " ".join(b for b in body if b), arg=kind,
                                source=where))
            continue
        if line.lstrip().startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            blocks.append(Block("table", items=rows, source=where))
            continue
        m = re.match(r"^(-|\d+\.)\s+(.*)$", line)
        if m:
            flush()
            kind = "ul" if m.group(1) == "-" else "ol"
            items = []
            while i < len(lines):
                cur = lines[i].rstrip()
                mm = re.match(r"^(-|\d+\.)\s+(.*)$", cur)
                if mm and (("ul" if mm.group(1) == "-" else "ol") == kind):
                    items.append(mm.group(2).strip())
                elif cur.startswith("  ") and cur.strip() and items:
                    items[-1] += " " + cur.strip()
                else:
                    break
                i += 1
            start = m.group(1)[:-1] if kind == "ol" else ""
            blocks.append(Block(kind, items=items, arg=start, source=where))
            continue
        if not para:
            para_line = i + 1
        para.append(line)
        i += 1
    flush()
    return blocks


def parse_file(path: Path) -> list[Block]:
    return parse(path.read_text(encoding="utf-8"), path.name)


# ---------------------------------------------------------------------------------- inline
MENU_STYLE = ('<font backColor="#FBE3E3" color="#8E1B1B" face="DejaVuSans-Bold">'
              '&#160;{}&#160;</font>')
KEY_STYLE = ('<font backColor="#EEEEEE" color="#333333" face="DejaVuSansMono" size="8.6">'
             '&#160;{}&#160;</font>')
CODE_STYLE = '<font face="DejaVuSansMono" color="#8E1B1B" size="8.8">{}</font>'


def inline(text: str) -> str:
    """Inline markup to reportlab paragraph XML."""
    out = []
    pos = 0
    token = re.compile(r"\[\[(.+?)\]\]|\+\+(.+?)\+\+|`([^`]+)`")
    for m in token.finditer(text):
        out.append(_emphasis(escape(text[pos:m.start()], quote=False)))
        if m.group(1) is not None:
            parts = [escape(p.strip(), quote=False) for p in m.group(1).split(">")]
            out.append(MENU_STYLE.format(" &#9656; ".join(parts)))
        elif m.group(2) is not None:
            out.append(KEY_STYLE.format(escape(m.group(2), quote=False)))
        else:
            out.append(CODE_STYLE.format(escape(m.group(3), quote=False)))
        pos = m.end()
    out.append(_emphasis(escape(text[pos:], quote=False)))
    return "".join(out)


def _emphasis(s: str) -> str:
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", s)
    return s

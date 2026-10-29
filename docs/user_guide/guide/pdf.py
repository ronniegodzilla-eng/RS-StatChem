"""PDF layout of the user guide (reportlab), in the app's red-and-white theme."""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Flowable, Frame, Image,
                                KeepTogether, NextPageTemplate, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

from . import markup
from .features import Row

RED = colors.HexColor("#C62828")
RED_DARK = colors.HexColor("#8E1B1B")
RED_LIGHT = colors.HexColor("#FBE3E3")
RED_PALE = colors.HexColor("#FDF3F3")
TEXT = colors.HexColor("#1B1B1B")
MUTED = colors.HexColor("#6B6B6B")
GRID = colors.HexColor("#E3E3E3")
AMBER = colors.HexColor("#E65100")
AMBER_LIGHT = colors.HexColor("#FFF4E5")
GREY_LIGHT = colors.HexColor("#F4F4F4")

PAGE_W, PAGE_H = A4
MARGIN_L = MARGIN_R = 20 * mm
MARGIN_T = 24 * mm
MARGIN_B = 20 * mm
FRAME_W = PAGE_W - MARGIN_L - MARGIN_R

MONTHS = ("Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus",
          "September", "Oktober", "November", "Desember")


# ---------------------------------------------------------------------------------- fonts
def register_fonts() -> None:
    import matplotlib

    ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    for name, file in (("DejaVuSans", "DejaVuSans.ttf"),
                       ("DejaVuSans-Bold", "DejaVuSans-Bold.ttf"),
                       ("DejaVuSans-Oblique", "DejaVuSans-Oblique.ttf"),
                       ("DejaVuSans-BoldOblique", "DejaVuSans-BoldOblique.ttf"),
                       ("DejaVuSansMono", "DejaVuSansMono.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(ttf / file)))
    pdfmetrics.registerFontFamily("DejaVuSans", normal="DejaVuSans", bold="DejaVuSans-Bold",
                                  italic="DejaVuSans-Oblique",
                                  boldItalic="DejaVuSans-BoldOblique")


def styles() -> dict[str, ParagraphStyle]:
    base = ParagraphStyle("body", fontName="DejaVuSans", fontSize=9.6, leading=14.2,
                          textColor=TEXT, spaceAfter=6)
    s = {
        "body": base,
        "h1": ParagraphStyle("h1", base, fontName="DejaVuSans-Bold", fontSize=24, leading=29,
                             textColor=TEXT, spaceAfter=4),
        "h2": ParagraphStyle("h2", base, fontName="DejaVuSans-Bold", fontSize=14, leading=18,
                             textColor=RED_DARK, spaceBefore=12, spaceAfter=6),
        "h3": ParagraphStyle("h3", base, fontName="DejaVuSans-Bold", fontSize=11, leading=15,
                             textColor=TEXT, spaceBefore=8, spaceAfter=4),
        "bullet": ParagraphStyle("bullet", base, leftIndent=14, bulletIndent=3, spaceAfter=3),
        "caption": ParagraphStyle("caption", base, fontName="DejaVuSans-Oblique", fontSize=8.4,
                                  leading=11, textColor=MUTED, alignment=TA_CENTER,
                                  spaceBefore=3, spaceAfter=2),
        "legend": ParagraphStyle("legend", base, fontSize=8.6, leading=11.6, spaceAfter=0),
        "cell": ParagraphStyle("cell", base, fontSize=8.4, leading=11.2, spaceAfter=0),
        "cellb": ParagraphStyle("cellb", base, fontName="DejaVuSans-Bold", fontSize=8.4,
                                leading=11.2, spaceAfter=0),
        "cellh": ParagraphStyle("cellh", base, fontName="DejaVuSans-Bold", fontSize=8.4,
                                leading=11.2, textColor=RED_DARK, spaceAfter=0),
        "cellm": ParagraphStyle("cellm", base, fontSize=7.4, leading=9.8, textColor=MUTED,
                                spaceAfter=0),
        "callout": ParagraphStyle("callout", base, fontSize=9, leading=13, spaceAfter=0),
        "toc1": ParagraphStyle("toc1", base, fontName="DejaVuSans-Bold", fontSize=10.5,
                               leading=15, spaceBefore=7, textColor=TEXT),
        "toc2": ParagraphStyle("toc2", base, fontSize=9.2, leading=12.5, leftIndent=16,
                               textColor=colors.HexColor("#3A3A3A")),
        "menuhead": ParagraphStyle("menuhead", base, fontName="DejaVuSans-Bold", fontSize=10.5,
                                   leading=14, textColor=colors.white),
    }
    return s


# ---------------------------------------------------------------------------------- flowables
class ChapterOpener(Flowable):
    """Big chapter number, title and a red rule; also feeds the TOC and bookmarks."""

    def __init__(self, number: int, title: str, intro: str = ""):
        super().__init__()
        self.number, self.title, self.intro = number, title, intro
        self.height = 62 * mm

    def wrap(self, aw, ah):
        return aw, self.height

    def draw(self):
        c = self.canv
        c.saveState()
        c.setFillColor(RED_LIGHT)
        c.setFont("DejaVuSans-Bold", 96)
        c.drawString(-2, self.height - 30 * mm, f"{self.number:02d}")
        c.setFillColor(RED)
        c.setFont("DejaVuSans-Bold", 10)
        c.drawString(0, self.height - 6 * mm, f"BAB {self.number}")
        c.setFillColor(TEXT)
        c.setFont("DejaVuSans-Bold", 25)
        c.drawString(0, self.height - 44 * mm, self.title)
        c.setStrokeColor(RED)
        c.setLineWidth(3)
        c.line(0, self.height - 50 * mm, 34 * mm, self.height - 50 * mm)
        c.setStrokeColor(GRID)
        c.setLineWidth(0.6)
        c.line(34 * mm, self.height - 50 * mm, FRAME_W, self.height - 50 * mm)
        c.restoreState()


class Badge(Flowable):
    def __init__(self, n: int, d: float = 12):
        super().__init__()
        self.n, self.d = n, d

    def wrap(self, aw, ah):
        return self.d, self.d

    def draw(self):
        c = self.canv
        c.setFillColor(RED)
        c.circle(self.d / 2, self.d / 2, self.d / 2, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.setFont("DejaVuSans-Bold", 7.2)
        c.drawCentredString(self.d / 2, self.d / 2 - 2.5, str(self.n))


class SectionMark(Flowable):
    """Zero-size marker that records a heading for the TOC and the PDF outline."""

    def __init__(self, level: int, text: str, key: str):
        super().__init__()
        self.level, self.text, self.key = level, text, key

    def wrap(self, aw, ah):
        return 0, 0

    def draw(self):
        pass


@dataclass
class Figure:
    path: Path
    caption: str
    legend: list[tuple[int, str]]


# ---------------------------------------------------------------------------------- document
class GuideDoc(BaseDocTemplate):
    def __init__(self, filename, version: str, date_text: str, cover_image: Path | None,
                 **kw):
        super().__init__(filename, pagesize=A4, leftMargin=MARGIN_L, rightMargin=MARGIN_R,
                         topMargin=MARGIN_T, bottomMargin=MARGIN_B,
                         title="Panduan Pengguna RS-StatChem", author="Roni Saputra, M.Si",
                         subject=f"RS-StatChem {version}", creator="RS-StatChem guide builder",
                         **kw)
        self.version, self.date_text, self.cover_image = version, date_text, cover_image
        self.chapter = ""
        frame = Frame(MARGIN_L, MARGIN_B, FRAME_W, PAGE_H - MARGIN_T - MARGIN_B, id="body",
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([
            PageTemplate("cover", [Frame(0, 0, PAGE_W, PAGE_H, id="c")], onPage=self._cover),
            PageTemplate("front", [frame], onPage=self._front),
            PageTemplate("body", [frame], onPageEnd=self._body),
        ])
        self._seq = 0

    # -- page decorations -------------------------------------------------------------
    def _cover(self, c, doc):
        from rsstatchem.paths import resource

        c.saveState()
        band = PAGE_H * 0.50
        c.setFillColor(RED)
        c.rect(0, PAGE_H - band, PAGE_W, band, stroke=0, fill=1)
        c.setFillColor(RED_DARK)
        c.rect(0, PAGE_H - band, PAGE_W, 9 * mm, stroke=0, fill=1)
        # subtle diagonal stripes
        c.setStrokeColor(colors.Color(1, 1, 1, alpha=0.06))
        c.setLineWidth(14)
        for k in range(-6, 14):
            x = k * 34 * mm
            c.line(x, PAGE_H, x + band * 0.9, PAGE_H - band + 9 * mm)
        logo = resource("logo.png")
        c.setFillColor(colors.white)
        c.circle(MARGIN_L + 16 * mm, PAGE_H - 40 * mm, 16 * mm, stroke=0, fill=1)
        c.drawImage(str(logo), MARGIN_L + 3 * mm, PAGE_H - 53 * mm, 26 * mm, 26 * mm,
                    mask="auto", preserveAspectRatio=True)
        c.setFillColor(colors.white)
        c.setFont("DejaVuSans", 13)
        c.drawString(MARGIN_L, PAGE_H - 78 * mm, "PANDUAN PENGGUNA")
        c.setFont("DejaVuSans-Bold", 42)
        c.drawString(MARGIN_L, PAGE_H - 95 * mm, "RS-StatChem")
        c.setFont("DejaVuSans", 12)
        c.drawString(MARGIN_L, PAGE_H - 106 * mm,
                     "Statistik  ·  RSM / DOE  ·  Grafik & Analisis  ·  "
                     "Gambar Kimia  ·  Analisis Citra")
        c.setFont("DejaVuSans-Bold", 11)
        c.drawString(MARGIN_L, PAGE_H - band + 15 * mm, f"Versi {self.version}")
        c.setFont("DejaVuSans", 11)
        c.drawString(MARGIN_L + 30 * mm, PAGE_H - band + 15 * mm, self.date_text)
        if self.cover_image and self.cover_image.exists():
            from reportlab.lib.utils import ImageReader

            img = ImageReader(str(self.cover_image))
            iw, ih = img.getSize()
            w = PAGE_W - 2 * MARGIN_L
            h = w * ih / iw
            max_h = PAGE_H - band - 46 * mm
            if h > max_h:
                h = max_h
                w = h * iw / ih
            x = (PAGE_W - w) / 2
            y = PAGE_H - band - 12 * mm - h
            c.drawImage(img, x, y, w, h, mask="auto")
        c.setFillColor(MUTED)
        c.setFont("DejaVuSans", 8.5)
        c.drawString(MARGIN_L, 14 * mm, "© 2026 Roni Saputra, M.Si")
        c.drawRightString(PAGE_W - MARGIN_R, 14 * mm,
                          "Untuk penggunaan pribadi dan non-komersial "
                          "(PolyForm Noncommercial 1.0.0)")
        c.restoreState()

    def _front(self, c, doc):
        self._footer(c, roman=True)

    def _body(self, c, doc):
        c.saveState()
        c.setStrokeColor(RED)
        c.setLineWidth(1.2)
        y = PAGE_H - 14 * mm
        c.line(MARGIN_L, y, PAGE_W - MARGIN_R, y)
        c.setFont("DejaVuSans", 7.8)
        c.setFillColor(MUTED)
        c.drawString(MARGIN_L, y + 2.4 * mm, self.chapter.upper())
        c.drawRightString(PAGE_W - MARGIN_R, y + 2.4 * mm,
                          f"Panduan Pengguna RS-StatChem {self.version}")
        c.restoreState()
        self._footer(c)

    def _footer(self, c, roman: bool = False):
        c.saveState()
        n = c.getPageNumber() - 1
        label = _roman(n) if roman else str(n - self._front_pages)
        c.setFont("DejaVuSans-Bold", 8.5)
        c.setFillColor(RED)
        c.drawRightString(PAGE_W - MARGIN_R, 11 * mm, label)
        c.setFont("DejaVuSans", 7.6)
        c.setFillColor(MUTED)
        c.drawString(MARGIN_L, 11 * mm, "© 2026 Roni Saputra, M.Si")
        c.restoreState()

    _front_pages = 0

    def handle_documentBegin(self):
        self.chapter = ""
        super().handle_documentBegin()

    def afterFlowable(self, flowable):
        if isinstance(flowable, ChapterOpener):
            self.chapter = flowable.title
            key = f"ch{flowable.number}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(f"{flowable.number}. {flowable.title}", key, 0)
            self.notify("TOCEntry", (0, f"{flowable.number}.&#160;&#160;{flowable.title}",
                                     self.page - self._front_pages - 1, key))
        elif isinstance(flowable, SectionMark):
            self.canv.bookmarkPage(flowable.key)
            self.canv.addOutlineEntry(flowable.text, flowable.key, flowable.level)
            if flowable.level == 1:
                self.notify("TOCEntry", (1, flowable.text, self.page - self._front_pages - 1,
                                         flowable.key))
        elif getattr(flowable, "_front_end", False):
            self._front_pages = self.page - 1


class FrontEnd(Flowable):
    """Marks the last front-matter page (body page numbers start after it)."""
    _front_end = True

    def wrap(self, aw, ah):
        return 0, 0

    def draw(self):
        pass


def _roman(n: int) -> str:
    vals = ((10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i"))
    out = ""
    for v, s in vals:
        while n >= v:
            out += s
            n -= v
    return out


# ---------------------------------------------------------------------------------- builder
class Builder:
    def __init__(self, figures: dict[str, Figure], references: dict[str, tuple[str, list[Row]]],
                 version: str, date_text: str, cover_image: Path | None = None):
        register_fonts()
        self.st = styles()
        self.figures = figures
        self.references = references
        self.version = version
        self.date_text = date_text
        self.cover_image = cover_image
        self.chapter_no = 0
        self.figure_no = 0
        self.mark_no = 0

    def build(self, chapters: list[list[markup.Block]], out: Path) -> None:
        story: list = [NextPageTemplate("front"), PageBreak()]
        story += self._toc()
        story += [FrontEnd(), NextPageTemplate("body")]
        for blocks in chapters:
            story += self._chapter(blocks)
        doc = GuideDoc(str(out), self.version, self.date_text, self.cover_image)
        doc.multiBuild(story)

    def _toc(self) -> list:
        toc = TableOfContents(dotsMinLevel=0)
        toc.levelStyles = [self.st["toc1"], self.st["toc2"]]
        title = Paragraph("Daftar Isi", self.st["h1"])
        return [Spacer(1, 6 * mm), title,
                _Rule(), Spacer(1, 4 * mm), toc]

    def _chapter(self, blocks: list[markup.Block]) -> list:
        out: list = []
        for b in blocks:
            out += self._block(b)
        return out

    def _mark(self, level: int, text: str) -> SectionMark:
        self.mark_no += 1
        return SectionMark(level, text, f"s{self.mark_no}")

    def _block(self, b: markup.Block) -> list:
        st = self.st
        if b.kind == "h1":
            self.chapter_no += 1
            self.figure_no = 0
            return [PageBreak(), ChapterOpener(self.chapter_no, b.text)]
        if b.kind == "h2":
            plain = _plain(b.text)
            return [CondPageBreak(42 * mm), self._mark(1, plain),
                    Paragraph(markup.inline(b.text), st["h2"])]
        if b.kind == "h3":
            return [CondPageBreak(28 * mm), self._mark(2, _plain(b.text)),
                    Paragraph(markup.inline(b.text), st["h3"])]
        if b.kind == "p":
            return [Paragraph(markup.inline(b.text), st["body"])]
        if b.kind == "ul":
            return [Paragraph(markup.inline(t), st["bullet"],
                              bulletText="•") for t in b.items] + [Spacer(1, 3)]
        if b.kind == "ol":
            first = int(b.arg or 1)
            return [self._step(i, t) for i, t in enumerate(b.items, first)] + [Spacer(1, 4)]
        if b.kind == "callout":
            return [self._callout(b.arg, b.text)]
        if b.kind == "figure":
            return self._figure(b)
        if b.kind == "table":
            return [self._table(b.items), Spacer(1, 6)]
        if b.kind == "features":
            return self._reference(b.arg)
        if b.kind == "pagebreak":
            return [PageBreak()]
        raise ValueError(f"{b.source}: unknown block {b.kind}")

    # -- pieces -----------------------------------------------------------------------
    def _step(self, n: int, text: str):
        body = Paragraph(markup.inline(text), ParagraphStyle("s", self.st["body"],
                                                              spaceAfter=0))
        t = Table([[Badge(n, 13), body]], colWidths=[7.5 * mm, FRAME_W - 7.5 * mm])
        t.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (0, 0), 0), ("RIGHTPADDING", (0, 0), (0, 0), 0),
            ("TOPPADDING", (0, 0), (0, 0), 0.8), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (1, 0), (1, 0), 3), ("TOPPADDING", (1, 0), (1, 0), 0),
        ]))
        return t

    def _callout(self, kind: str, text: str):
        title = markup.CALLOUTS[kind]
        bar, bg = {"TIP": (RED, RED_PALE), "NOTE": (colors.HexColor("#616161"), GREY_LIGHT),
                   "WARNING": (AMBER, AMBER_LIGHT)}[kind]
        hexbar = bar.hexval()[2:]
        p = Paragraph(f'<font color="#{hexbar}"><b>{title}.</b></font>&#160; '
                      + markup.inline(text), self.st["callout"])
        t = Table([[p]], colWidths=[FRAME_W])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), bg),
            ("LINEBEFORE", (0, 0), (0, -1), 3.2, bar),
            ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        return KeepTogether([Spacer(1, 3), t, Spacer(1, 8)])

    def _figure(self, b: markup.Block) -> list:
        fig = self.figures.get(b.arg)
        if fig is None:
            raise ValueError(f"{b.source}: no screenshot for scene {b.arg!r}")
        self.figure_no += 1
        img = _image(fig.path, FRAME_W, PAGE_H * 0.52)
        caption = b.text or fig.caption
        parts = [img, Paragraph(f"<b>Gambar {self.chapter_no}.{self.figure_no}</b>&#160;&#160;"
                                + markup.inline(caption), self.st["caption"])]
        if fig.legend:
            parts.append(Spacer(1, 3))
            parts.append(self._legend(fig.legend))
        return [Spacer(1, 4), KeepTogether(parts), Spacer(1, 8)]

    def _legend(self, legend: list[tuple[int, str]]):
        cells = [[Badge(n), Paragraph(markup.inline(t), self.st["legend"])] for n, t in legend]
        if len(cells) >= 4:
            half = (len(cells) + 1) // 2
            left, right = cells[:half], cells[half:]
            rows = [(left[i] + (right[i] if i < len(right) else ["", ""]))
                    for i in range(half)]
            col = (FRAME_W - 2 * 7 * mm) / 2
            widths = [7 * mm, col, 7 * mm, col]
        else:
            rows = cells
            widths = [7 * mm, FRAME_W - 7 * mm]
        t = Table(rows, colWidths=widths)
        t.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ("BACKGROUND", (0, 0), (-1, -1), RED_PALE),
            ("BOX", (0, 0), (-1, -1), 0.5, RED_LIGHT),
        ]))
        return t

    def _table(self, rows: list[list[str]]):
        st = self.st
        n = max(len(r) for r in rows)
        data = []
        for i, r in enumerate(rows):
            r = r + [""] * (n - len(r))
            style = st["cellh"] if i == 0 else st["cell"]
            data.append([Paragraph(markup.inline(c), style) for c in r])
        t = Table(data, colWidths=_widths(rows, n), repeatRows=1)
        t.setStyle(_grid_style(len(data)))
        return t

    def _reference(self, module: str) -> list:
        if module not in self.references:
            raise ValueError(f"no menu reference for module {module!r}")
        title, sections = self.references[module]
        st = self.st
        out: list = []
        for label, rows in sections:
            if label:
                out += [CondPageBreak(50 * mm), self._mark(2, label),
                        Paragraph(f"Perintah tambahan: {label}", st["h3"])]
            out += self._menu_tables(rows, label)
        return out

    def _menu_tables(self, rows: list[Row], label: str | None) -> list:
        st = self.st
        out: list = []
        by_menu: dict[str, list[Row]] = {}
        for r in rows:
            by_menu.setdefault(r.menu, []).append(r)
        for menu, items in by_menu.items():
            title = f"Menu {menu}" + (f"  \u00b7  {label}" if label else "")
            head = Table([[Paragraph(title, st["menuhead"])]], colWidths=[FRAME_W])
            head.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), RED),
                ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            shortcuts = any(r.shortcut for r in items)
            data = [[Paragraph("Perintah", st["cellh"]), Paragraph("Fungsi", st["cellh"])]
                    + ([Paragraph("Pintasan", st["cellh"])] if shortcuts else [])]
            for r in items:
                cmd = Paragraph(_label_xml(r.label), st["cellb"])
                desc = [Paragraph(markup.inline(r.text) if r.text else
                                  '<font color="#9E9E9E">—</font>', st["cell"])]
                if r.children:
                    desc.append(Paragraph("Isi: " + ", ".join(
                        markup.inline(c) for c in r.children), st["cellm"]))
                sc = Paragraph(markup.KEY_STYLE.format(r.shortcut) if r.shortcut else "",
                               st["cell"])
                data.append([cmd, desc] + ([sc] if shortcuts else []))
            widths = ([FRAME_W * 0.32, FRAME_W * 0.53, FRAME_W * 0.15] if shortcuts
                      else [FRAME_W * 0.32, FRAME_W * 0.68])
            t = Table(data, colWidths=widths, repeatRows=1)
            t.setStyle(_grid_style(len(data)))
            out += [CondPageBreak(40 * mm)]
            if not label:
                out.append(self._mark(2, f"Menu {menu}"))
            out += [head, t, Spacer(1, 10)]
        return out


def _widths(rows: list[list[str]], n: int) -> list[float]:
    """Column widths roughly proportional to their longest text (within limits)."""
    import re

    longest = []
    for j in range(n):
        lens = [len(re.sub(r"\*\*|`|\[\[|\]\]|\+\+", "", r[j])) if j < len(r) else 0
                for r in rows]
        longest.append(max(8, min(60, max(lens) if lens else 8)))
    total = sum(longest)
    w = [max(0.14, x / total) for x in longest]
    s = sum(w)
    return [FRAME_W * x / s for x in w]


def _label_xml(label: str) -> str:
    from html import escape

    parts = label.split(" ▸ ")
    if len(parts) == 1:
        return escape(parts[0])
    prefix = " ▸ ".join(escape(p) for p in parts[:-1])
    return f'<font color="#8A8A8A" face="DejaVuSans">{prefix} ▸ </font>{escape(parts[-1])}'


def _grid_style(n_rows: int) -> TableStyle:
    cmds = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, 0), RED_LIGHT),
        ("LINEBELOW", (0, 0), (-1, 0), 0.9, RED),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, GRID),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.6),
    ]
    for i in range(2, n_rows, 2):
        cmds.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#FAFAFA")))
    return TableStyle(cmds)


class _Rule(Flowable):
    def wrap(self, aw, ah):
        self.w = aw
        return aw, 4

    def draw(self):
        self.canv.setStrokeColor(RED)
        self.canv.setLineWidth(3)
        self.canv.line(0, 2, 34 * mm, 2)
        self.canv.setStrokeColor(GRID)
        self.canv.setLineWidth(0.6)
        self.canv.line(34 * mm, 2, self.w, 2)


def _plain(text: str) -> str:
    import re

    return re.sub(r"\[\[|\]\]|\*\*|\+\+|`", "", text).replace(" > ", " ▸ ")


def _image(path: Path, max_w: float, max_h: float) -> Image:
    """A screenshot, reduced to print resolution and palette-compressed to keep the PDF
    small."""
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        # about 220 dpi at the printed size
        target_w = int(min(w, max_w / 72 * 220))
        if target_w < w:
            im = im.resize((target_w, int(h * target_w / w)), PILImage.LANCZOS)
        q = im.quantize(colors=256, method=PILImage.Quantize.MEDIANCUT, dither=PILImage.Dither.NONE)
        buf = io.BytesIO()
        q.save(buf, format="PNG", optimize=True)
        buf.seek(0)
        pw, ph = im.size
    # one print scale for every picture (a window about 1200 screen pixels wide fills the
    # text width), so a small dialog stays small instead of being blown up
    logical_w = w / 2.0           # the PNGs are drawn at twice the screen density
    width = min(max_w, logical_w * FRAME_W / 1200.0)
    scale = min(width / pw, max_h / ph)
    flow = Image(buf, width=pw * scale, height=ph * scale)
    flow.hAlign = "CENTER"
    return flow


def date_text(d) -> str:
    return f"{MONTHS[d.month - 1]} {d.year}"

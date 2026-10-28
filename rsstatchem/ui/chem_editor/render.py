"""Builds QGraphicsItems for a drawing document (canvas and all exports share it).

Text is drawn as glyph outlines (``QPainterPath.addText``) from a 100 px reference font
scaled to the style's point size, so labels look identical on screen, in SVG/PDF and in
PNGs of any resolution. Bonds stop at the label boxes (plus the style margin) instead of
being masked, as in ChemDraw.

Also draws the teaching / lab-report objects of :mod:`rsstatchem.core.chem_draw.objects`
(drawing elements, chemical symbols, TLC and gel plates), clipware, circled charges,
attachment points and the stereo indicators of Object > Show Stereochemistry.
"""
from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass, field

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QBrush, QColor, QFont, QFontMetricsF, QLinearGradient,
                           QPainterPath, QPen, QPolygonF, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QGraphicsLineItem, QGraphicsPathItem, QGraphicsPolygonItem,
                               QGraphicsScene)

from rsstatchem.core.chem.formula import format_formula
from rsstatchem.core.chem_draw import geometry as geo
from rsstatchem.core.chem_draw import clipart as clip
from rsstatchem.core.chem_draw import labels as lab
from rsstatchem.core.chem_draw import objects as obj
from rsstatchem.core.chem_draw.model import Document
from rsstatchem.core.chem_draw.style import Style, element_color

__all__ = ["Builder", "BuildInfo", "chem_text_lines", "SELECT_COLOR"]

REF_PX = 100.0
SUB = 0.72
SELECT_COLOR = QColor(47, 125, 225)
_FONTS: dict = {}


def _font(family: str, bold: bool = False, italic: bool = False
          ) -> tuple[QFont, QFontMetricsF]:
    key = (family, bold, italic)
    if key not in _FONTS:
        f = QFont(family)
        f.setPixelSize(int(REF_PX))
        f.setBold(bold)
        f.setItalic(italic)
        f.setStyleStrategy(QFont.StyleStrategy.PreferOutline)
        _FONTS[key] = (f, QFontMetricsF(f))
    return _FONTS[key]


_TAG = re.compile(r"<(sub|sup)>(.*?)</\1>")


def _html_runs(text: str) -> list[tuple[str, str]]:
    runs, pos = [], 0
    for m in _TAG.finditer(text):
        if m.start() > pos:
            runs.append((html.unescape(text[pos:m.start()]), "n"))
        runs.append((html.unescape(m.group(2)).replace("&minus;", "−"), m.group(1)))
        pos = m.end()
    if pos < len(text):
        runs.append((html.unescape(text[pos:]), "n"))
    return [(t.replace("&minus;", "−"), k) for t, k in runs]


def _word_runs(word: str) -> list[tuple[str, str]]:
    core = word.rstrip(",;:")
    tail = word[len(core):]
    out = None
    if core:
        h = format_formula(core, "html")
        if h != core and "<" in h:
            out = _html_runs(h)
        elif (core[0].isupper() or core[0] == "(") and any(c.isdigit() for c in core):
            out = lab.split_digits(core)
    out = out or [(core, "n")]
    if tail:
        out.append((tail, "n"))
    return out


def chem_text_lines(text: str, chemical: bool = True) -> list[list[tuple[str, str]]]:
    """Runs per line; with ``chemical`` formulas get subscripts and charges superscripts."""
    lines = []
    for line in (text or "").split("\n"):
        if "<sub>" in line or "<sup>" in line:      # explicit sub/superscript markup
            lines.append(_html_runs(line) or [("", "n")])
            continue
        if not chemical:
            lines.append([(line, "n")])
            continue
        runs: list[tuple[str, str]] = []
        for tok in re.split(r"(\s+)", line):
            if not tok:
                continue
            runs += [(tok, "n")] if tok.isspace() else _word_runs(tok)
        lines.append(runs or [("", "n")])
    return lines


@dataclass
class BuildInfo:
    label_boxes: dict = field(default_factory=dict)    # atom id -> QRectF
    text_boxes: dict = field(default_factory=dict)     # text id -> QRectF
    plus_boxes: dict = field(default_factory=dict)
    bracket_boxes: dict = field(default_factory=dict)
    arrow_text: dict = field(default_factory=dict)     # arrow id -> [QRectF]
    curve_boxes: dict = field(default_factory=dict)    # curve id -> QRectF
    orbital_boxes: dict = field(default_factory=dict)  # orbital id -> QRectF
    object_boxes: dict = field(default_factory=dict)   # (kind, id) -> QRectF (new objects)


class Builder:
    def __init__(self, style: Style):
        self.style = style

    # --- text ----------------------------------------------------------------------------
    def runs_path(self, runs, size: float, x: float, baseline: float, bold: bool = False,
                  italic: bool = False, family: str = "") -> tuple[QPainterPath, list[float]]:
        font, fm = _font(family or self.style.font_family, bold, italic)
        path = QPainterPath()
        widths = []
        cursor = x
        for text, kind in runs:
            s = size * (SUB if kind in ("sub", "sup") else 1.0)
            shift = size * 0.30 if kind == "sub" else (-size * 0.42 if kind == "sup" else 0)
            if text:
                p = QPainterPath()
                p.addText(0, 0, font, text)
                k = s / REF_PX
                t = QTransform()
                t.translate(cursor, baseline + shift)
                t.scale(k, k)
                path.addPath(t.map(p))
            w = fm.horizontalAdvance(text) * s / REF_PX
            widths.append(w)
            cursor += w
        return path, widths

    def run_widths(self, runs, size: float, bold: bool = False, italic: bool = False,
                   family: str = "") -> list[float]:
        _, fm = _font(family or self.style.font_family, bold, italic)
        return [fm.horizontalAdvance(t) * size * (SUB if k in ("sub", "sup") else 1.0)
                / REF_PX for t, k in runs]

    def cap_height(self, size: float, family: str = "") -> float:
        _, fm = _font(family or self.style.font_family)
        return fm.capHeight() * size / REF_PX

    def text_block(self, lines, size: float, x: float, y: float, bold=False, center=False,
                   italic: bool = False, family: str = "", underline: bool = False
                   ) -> tuple[QPainterPath, QRectF]:
        """Multi-line text with top-left (x, y) (or centred on x)."""
        path = QPainterPath()
        line_h = size * 1.25
        cap = self.cap_height(size, family)
        width = 0.0
        for i, runs in enumerate(lines):
            w = sum(self.run_widths(runs, size, bold, italic, family))
            width = max(width, w)
            x0 = x - w / 2 if center else x
            base = y + size * 0.2 + cap + i * line_h
            p, _ = self.runs_path(runs, size, x0, base, bold, italic, family)
            path.addPath(p)
            if underline and w > 0:
                path.setFillRule(Qt.FillRule.WindingFill)   # glyph outlines are non-zero
                t = max(0.5, size * 0.06)
                path.addRect(QRectF(x0, base + size * 0.12, w, t))
        h = line_h * len(lines)
        rect = QRectF(x - width / 2 if center else x, y, width, h)
        return path, rect

    # --- labels --------------------------------------------------------------------------
    def label(self, doc: Document, atom) -> tuple[QPainterPath | None, QRectF | None, str]:
        lay = lab.layout(doc, atom)
        if not lay.visible:
            return None, None, "#000000"
        st = self.style
        size = st.font_size
        widths = self.run_widths(lay.runs, size)
        text = lay.text
        # character offsets -> run index for the anchor range
        starts, pos = [], 0
        for t, _ in lay.runs:
            starts.append(pos)
            pos += len(t)

        def char_x(ci: int) -> float:
            x = 0.0
            for (t, k), s0, w in zip(lay.runs, starts, widths):
                if ci <= s0:
                    break
                if ci >= s0 + len(t):
                    x += w
                else:
                    x += w * (ci - s0) / max(1, len(t))
                    break
            return x
        a0, a1 = lay.anchor
        anchor_c = (char_x(a0) + char_x(a1)) / 2 if text else 0.0
        cap = self.cap_height(size)
        x0 = atom.x - anchor_c
        base = atom.y + cap / 2
        path, _ = self.runs_path(lay.runs, size, x0, base)
        total = sum(widths)
        m = st.margin
        rect = QRectF(x0 - m, base - cap - m, total + 2 * m, cap + 2 * m)
        if any(k == "sub" for _, k in lay.runs):
            rect.setBottom(rect.bottom() + size * 0.25)
        if any(k == "sup" for _, k in lay.runs):
            rect.setTop(rect.top() - size * 0.2)
        if lay.stacked:
            sw = sum(self.run_widths(lay.stacked, size))
            hx = atom.x - self.run_widths([lay.stacked[0]], size)[0] / 2
            gap = cap * 1.45
            sb = base - gap if lay.stacked_pos == "up" else base + gap
            p2, _ = self.runs_path(lay.stacked, size, hx, sb)
            path.addPath(p2)
            r2 = QRectF(hx - m, sb - cap - m, sw + 2 * m, cap + 2 * m + size * 0.25)
            rect = rect.united(r2)
        anchor_sym = atom.symbol if not atom.label else ""
        color = element_color(anchor_sym, st.color_by_element) if anchor_sym else "#000000"
        return path, rect, getattr(atom, "color", "") or color

    # --- scene ---------------------------------------------------------------------------
    def build(self, scene: QGraphicsScene, doc: Document, z0: float = 0.0) -> BuildInfo:
        st = self.style
        info = BuildInfo()
        pen_color = QColor("#000000")
        boxes: dict[int, QRectF] = {}
        labelled = set()
        for a in doc.atoms.values():
            path, rect, color = self.label(doc, a)
            if path is None:
                if a.radicals:
                    self._radicals(scene, a.x, a.y - st.font_size * 0.35, a.radicals, z0)
                continue
            item = QGraphicsPathItem(path)
            item.setBrush(QBrush(QColor(color)))
            item.setPen(Qt.PenStyle.NoPen)
            item.setZValue(z0 + 2)
            scene.addItem(item)
            boxes[a.id] = rect
            labelled.add(a.id)
            if a.radicals:
                self._radicals(scene, a.x, rect.top() - st.line_width * 1.2, a.radicals, z0)
        for a in doc.atoms.values():
            if a.map_number:
                size = st.font_size * 0.6
                ref = boxes.get(a.id)
                x = (ref.right() if ref else a.x + st.font_size * 0.25)
                y = (ref.bottom() if ref else a.y + st.font_size * 0.2)
                path, _ = self.runs_path([(str(a.map_number), "n")], size, x, y + size * 0.6)
                item = QGraphicsPathItem(path)
                item.setBrush(QBrush(QColor("#1565C0")))
                item.setPen(Qt.PenStyle.NoPen)
                item.setZValue(z0 + 2)
                scene.addItem(item)
            if a.number:
                self._atom_number(scene, doc, a, boxes.get(a.id), z0)
            if a.charge and getattr(a, "charge_circle", False):
                self._circled_charge(scene, doc, a, boxes.get(a.id), z0)
            if getattr(a, "attachment", 0):
                self._attachment(scene, doc, a, boxes.get(a.id), z0)
            if st.show_stereo and a.stereo and a.show_stereo:
                self._stereo_atom(scene, doc, a, boxes.get(a.id), z0)
        info.label_boxes = boxes
        rings = doc.rings()
        for b in doc.bonds.values():
            pa, pb = doc.atoms.get(b.a), doc.atoms.get(b.b)
            if pa is None or pb is None:
                continue
            dx, dy = pb.x - pa.x, pb.y - pa.y
            L = math.hypot(dx, dy) or 1.0
            ux, uy = dx / L, dy / L
            ta = geo.ray_box_exit(pa.x, pa.y, ux, uy, _box(boxes[b.a])) if b.a in boxes else 0
            tb = geo.ray_box_exit(pb.x, pb.y, -ux, -uy, _box(boxes[b.b])) \
                if b.b in boxes else 0
            side = None
            if b.order in (2, 1.5):
                side = geo.double_bond_side(doc, b, labelled, rings)
            color = _color(b, pen_color)
            for prim in geo.bond_primitives(doc, b, st, ta, tb, side, labelled):
                self._prim(scene, prim, color, z0 + 1)
            if st.show_stereo and getattr(b, "stereo", ""):
                self._stereo_bond(scene, doc, b, side, z0)
        for ar in doc.arrows.values():
            z = z0 + 3 + ar.z
            for prim in geo.arrow_primitives(ar, st):
                self._prim(scene, prim, _color(ar, pen_color), z)
            info.arrow_text[ar.id] = self._arrow_text(scene, ar, z)
        for c in doc.curves.values():
            z = z0 + 3 + c.z
            items = [self._prim(scene, prim, _color(c, pen_color), z)
                     for prim in geo.curve_primitives(c, st)]
            r = QRectF()
            for it in items:
                r = r.united(it.sceneBoundingRect())
            info.curve_boxes[c.id] = r
        for o in doc.orbitals.values():
            z = z0 + 0.5 + o.z                       # behind bonds and labels
            items = [self._prim(scene, prim, _color(o, pen_color), z)
                     for prim in geo.orbital_primitives(o, st)]
            r = QRectF()
            for it in items:
                r = r.united(it.sceneBoundingRect())
            info.orbital_boxes[o.id] = r
        for p in doc.pluses.values():
            h = st.caption_size * 0.36
            w = max(st.line_width, 0.6)
            for x1, y1, x2, y2 in ((p.x - h, p.y, p.x + h, p.y), (p.x, p.y - h, p.x, p.y + h)):
                self._prim(scene, ("line", x1, y1, x2, y2, w, False), _color(p, pen_color),
                           z0 + 3 + p.z, cap=Qt.PenCapStyle.FlatCap)
            info.plus_boxes[p.id] = QRectF(p.x - h, p.y - h, 2 * h, 2 * h)
        for t in doc.texts.values():
            size = t.size or st.caption_size
            path, rect = self.text_block(chem_text_lines(t.text, t.chemical), size, t.x, t.y,
                                         t.bold, italic=getattr(t, "italic", False),
                                         family=getattr(t, "font", ""),
                                         underline=getattr(t, "underline", False))
            item = QGraphicsPathItem(path)
            item.setBrush(QBrush(_color(t, pen_color)))
            item.setPen(Qt.PenStyle.NoPen)
            item.setZValue(z0 + 3 + t.z)
            scene.addItem(item)
            info.text_boxes[t.id] = rect
        for br in doc.brackets.values():
            z = z0 + 3 + br.z
            sides = getattr(br, "sides", "both")
            for prim in geo.bracket_primitives(br.x1, br.y1, br.x2, br.y2, br.kind, st, sides):
                self._prim(scene, prim, _color(br, pen_color), z)
            size = st.caption_size * 0.85
            lx = (br.x1 - 1.5) if sides == "left" else br.x2 + 1.5
            anchor = "r" if sides == "left" else "l"
            if br.label:
                self._prim(scene, ("text", lx, br.y2 + size * 0.15, br.label, size, anchor),
                           _color(br, pen_color), z)
            top = " ".join(t for t in (POLYMER_SHORT.get(getattr(br, "polymer", ""), ""),
                                       getattr(br, "repeat", "")) if t)
            if top:
                self._prim(scene, ("text", lx, br.y1 + size * 0.1, top, size * 0.85, anchor),
                           _color(br, pen_color), z)
            info.bracket_boxes[br.id] = QRectF(br.x1, br.y1, br.x2 - br.x1, br.y2 - br.y1)
        self._objects(scene, doc, info, z0, pen_color)
        return info

    def _objects(self, scene, doc, info, z0, pen_color) -> None:
        """Drawing elements, chemical symbols, TLC / gel plates and clipware."""
        st = self.style
        groups = (
            ("shape", doc.shapes, lambda o: obj.shape_primitives(o, st), 3),
            ("tlc", doc.tlcs, lambda o: obj.tlc_primitives(o, st), 0.2),
            ("gel", doc.gels, lambda o: obj.gel_primitives(o, st), 0.2),
            ("clipart", doc.cliparts,
             lambda o: clip.clipart_primitives(o, st.line_width), 0.3),
            ("symbol", doc.symbols, lambda o: obj.symbol_primitives(o, st), 3.2))
        for kind, store, prims, dz in groups:
            for o in store.values():
                z = z0 + dz + o.z
                r = QRectF()
                for prim in prims(o):
                    it = self._prim(scene, prim, _color(o, pen_color), z)
                    if it is not None:
                        r = r.united(it.sceneBoundingRect())
                box = obj.object_box(doc, (kind, o.id))
                if box is not None:
                    r = r.united(QRectF(QPointF(box[0], box[1]), QPointF(box[2], box[3])))
                info.object_boxes[(kind, o.id)] = r

    def _arrow_text(self, scene, ar, z) -> list:
        rects = []
        st = self.style
        dx, dy = ar.x2 - ar.x1, ar.y2 - ar.y1
        L = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / L, dx / L
        if ny > 0 or (abs(ny) < 1e-9 and nx > 0):
            nx, ny = -nx, -ny           # (nx, ny) points to the screen-up side
        mx, my = (ar.x1 + ar.x2) / 2, (ar.y1 + ar.y2) / 2
        _, hw = geo.head_size(st)
        gap = hw * 1.2 + st.caption_size * 0.25
        size = st.caption_size
        for text, sign in ((ar.above, 1), (ar.below, -1)):
            if not text:
                continue
            lines = chem_text_lines(text)
            h = size * 1.25 * len(lines)
            cx = mx + nx * sign * (gap + h / 2)
            cy = my + ny * sign * (gap + h / 2)
            path, rect = self.text_block(lines, size, cx, cy - h / 2, center=True)
            item = QGraphicsPathItem(path)
            item.setBrush(QBrush(_color(ar, QColor("#000000"))))
            item.setPen(Qt.PenStyle.NoPen)
            item.setZValue(z)
            scene.addItem(item)
            rects.append(rect)
        return rects

    def _atom_number(self, scene, doc, atom, box, z0) -> None:
        """Small atom number beside the atom, on the side away from its bonds."""
        size = self.style.font_size * 0.62
        ang = geo.away_direction(doc, atom.id) if doc.bonds_of(atom.id) else -math.pi / 4
        r = (max(box.width(), box.height()) / 2 if box is not None else 0) + size * 0.9
        cx, cy = atom.x + r * math.cos(ang), atom.y + r * math.sin(ang)
        runs = [(str(atom.number), "n")]
        w = sum(self.run_widths(runs, size))
        path, _ = self.runs_path(runs, size, cx - w / 2, cy + self.cap_height(size) / 2)
        item = QGraphicsPathItem(path)
        item.setBrush(QBrush(QColor(atom.color or "#5A5A5A")))
        item.setPen(Qt.PenStyle.NoPen)
        item.setZValue(z0 + 2)
        scene.addItem(item)

    def _side_point(self, doc, atom, box, dist: float, avoid=()) -> tuple[float, float]:
        ang = free_angle(doc, atom.id, avoid)
        r = (max(box.width(), box.height()) / 2 if box is not None else 0) + dist
        return atom.x + r * math.cos(ang), atom.y + r * math.sin(ang)

    def _circled_charge(self, scene, doc, atom, box, z0) -> None:
        st = self.style
        r = obj.symbol_radius(st) * 0.8
        x, y = self._side_point(doc, atom, box, r * 1.4)
        kind = "circleplus" if atom.charge > 0 else "circleminus"
        from rsstatchem.core.chem_draw.model import Symbol
        s2 = st.updated(font_size=st.font_size * 0.8)
        for prim in obj.symbol_primitives(Symbol(0, x, y, kind), s2):
            self._prim(scene, prim, QColor(atom.color or "#000000"), z0 + 2)
        if abs(atom.charge) > 1:
            self._prim(scene, ("text", x - r * 1.3, y, str(abs(atom.charge)),
                               st.font_size * 0.7, "r"), QColor("#000000"), z0 + 2)

    def _attachment(self, scene, doc, atom, box, z0) -> None:
        st = self.style
        size = st.font_size * 0.6
        x, y = self._side_point(doc, atom, box, size * 1.1, avoid=(-math.pi / 2,))
        r = size * 0.75
        path = QPainterPath()
        path.addEllipse(QPointF(x, y), r, r)
        item = QGraphicsPathItem(path)
        item.setPen(QPen(QColor("#000000"), max(0.4, st.line_width * 0.7)))
        item.setBrush(Qt.BrushStyle.NoBrush)
        item.setZValue(z0 + 2)
        scene.addItem(item)
        self._prim(scene, ("text", x, y, str(atom.attachment), size, "c"), QColor("#000000"),
                   z0 + 2)

    def _stereo_text(self, label: str) -> str:
        return f"({label})" if self.style.stereo_parens else label

    def _stereo_atom(self, scene, doc, atom, box, z0) -> None:
        size = self.style.font_size * 0.8
        x, y = self._side_point(doc, atom, box, size * 1.15)
        self._prim(scene, ("text", x, y, self._stereo_text(atom.stereo), size, "c",
                           True), QColor(STEREO_COLOR), z0 + 2.5)

    def _stereo_bond(self, scene, doc, b, side, z0) -> None:
        pa, pb = doc.atoms[b.a], doc.atoms[b.b]
        dx, dy = pb.x - pa.x, pb.y - pa.y
        L = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / L, dx / L
        s = -(side or 1)                        # opposite the second line
        size = self.style.font_size * 0.8
        d = self.style.spacing + size * 0.75
        x, y = (pa.x + pb.x) / 2 + nx * s * d, (pa.y + pb.y) / 2 + ny * s * d
        self._prim(scene, ("text", x, y, self._stereo_text(b.stereo), size, "c", True),
                   QColor(STEREO_COLOR), z0 + 2.5)

    def _radicals(self, scene, x, y, n, z0):
        r = max(0.8, self.style.line_width * 1.1)
        xs = [x] if n == 1 else [x - r * 1.6, x + r * 1.6]
        for cx in xs:
            path = QPainterPath()
            path.addEllipse(QPointF(cx, y - r), r, r)
            item = QGraphicsPathItem(path)
            item.setBrush(QBrush(QColor("#000000")))
            item.setPen(Qt.PenStyle.NoPen)
            item.setZValue(z0 + 2)
            scene.addItem(item)

    def _prim(self, scene, prim, color, z, cap=Qt.PenCapStyle.RoundCap):
        kind = prim[0]
        if kind == "text":
            _, x, y, text, size, anchor = prim[:6]
            italic = len(prim) > 6 and prim[6]
            runs = chem_text_lines(text, False)[0]
            w = sum(self.run_widths(runs, size, italic=italic))
            x0 = x - w / 2 if anchor == "c" else (x - w if anchor == "r" else x)
            path, _ = self.runs_path(runs, size, x0, y + self.cap_height(size) / 2,
                                     italic=italic)
            item = QGraphicsPathItem(path)
            item.setBrush(QBrush(color))
            item.setPen(Qt.PenStyle.NoPen)
            item.setZValue(z)
            scene.addItem(item)
            return item
        if kind == "line":
            _, x1, y1, x2, y2, w, dashed = prim
            pen = QPen(color, w)
            pen.setCapStyle(Qt.PenCapStyle.FlatCap if dashed else cap)
            if dashed:
                pen.setDashPattern([2.2, 2.0])
            item = QGraphicsLineItem(x1, y1, x2, y2)
            item.setPen(pen)
        elif kind == "shape":
            _, pts, w, dashed, fill, closed = prim
            path = QPainterPath(QPointF(*pts[0]))
            for x, y in pts[1:]:
                path.lineTo(x, y)
            if closed:
                path.closeSubpath()
            item = QGraphicsPathItem(path)
            pen = QPen(color, w)
            pen.setCapStyle(Qt.PenCapStyle.FlatCap if dashed else Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            if dashed:
                pen.setDashPattern([3.0, 2.4])
            item.setPen(pen if w > 0 else QPen(Qt.PenStyle.NoPen))
            if fill and fill.startswith("#"):
                item.setBrush(QBrush(QColor(fill)))
            elif fill == "fade":
                r = path.boundingRect()
                g = QLinearGradient(r.topLeft(), r.bottomLeft())
                base = QColor(color)
                if base.lightness() < 40:
                    base = QColor("#7A7A7A")
                g.setColorAt(0.0, base)
                g.setColorAt(1.0, QColor("#FFFFFF"))
                item.setBrush(QBrush(g))
            elif fill == "solid":
                item.setBrush(QBrush(color))
            elif fill == "white":
                item.setBrush(QBrush(QColor("#FFFFFF")))
            elif fill == "shade":
                r = path.boundingRect()
                g = QRadialGradient(r.center().x() - r.width() * 0.18,
                                    r.center().y() - r.height() * 0.18,
                                    max(r.width(), r.height()) * 0.75)
                base = QColor(color)
                if base.lightness() < 40:            # black: a gray shading, as ChemDraw
                    base = QColor("#8A8A8A")
                g.setColorAt(0.0, QColor("#FFFFFF"))
                g.setColorAt(1.0, base)
                item.setBrush(QBrush(g))
            else:
                item.setBrush(Qt.BrushStyle.NoBrush)
        elif kind == "poly":
            item = QGraphicsPolygonItem(QPolygonF([QPointF(x, y) for x, y in prim[1]]))
            item.setBrush(QBrush(color))
            pen = QPen(color, 0.15)
            pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
            item.setPen(pen)
        else:
            pts = prim[1]
            path = QPainterPath(QPointF(*pts[0]))
            for x, y in pts[1:]:
                path.lineTo(x, y)
            item = QGraphicsPathItem(path)
            pen = QPen(color, prim[2])
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            item.setPen(pen)
            item.setBrush(Qt.BrushStyle.NoBrush)
        item.setZValue(z)
        scene.addItem(item)
        return item


STEREO_COLOR = "#000000"
POLYMER_SHORT = {"copolymer": "co", "alternating": "alt", "random": "ran", "block": "blk",
                 "graft": "grf", "mer": "mer", "monomer": "mon", "crosslink": "xl",
                 "mixture": "mix", "component": "c"}


def free_angle(doc: Document, aid: int, avoid=()) -> float:
    """Direction around an atom farthest from its bonds, symbols and ``avoid`` angles
    (prefers up-right, where ChemDraw puts charges)."""
    a = doc.atoms[aid]
    taken = [math.atan2(doc.atoms[n].y - a.y, doc.atoms[n].x - a.x)
             for n in doc.neighbors(aid)]
    taken += [s.angle for s in doc.symbols_of(aid)] + list(avoid)
    cands = [-math.pi / 4 + k * math.pi / 4 for k in range(8)]

    def gap(c):
        return min((abs(math.remainder(c - t, 2 * math.pi)) for t in taken),
                   default=math.pi)
    # any direction at least 60 deg from everything is fine; then prefer up-right
    return max(cands, key=lambda c: (min(gap(c), math.pi / 3 + 1e-6),
                                     -abs(math.remainder(c + math.pi / 4, 2 * math.pi))))


def _color(obj, default: QColor) -> QColor:
    c = getattr(obj, "color", "")
    return QColor(c) if c else default


def _box(r: QRectF) -> tuple[float, float, float, float]:
    return r.left(), r.top(), r.right(), r.bottom()

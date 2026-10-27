"""Geometry and editing operations of the drawing objects added for teaching and lab reports
(no Qt): drawing elements (circle, ellipse, boxes, arcs), chemical symbols (lone pairs,
delta charges, circled charges, radical ions, daggers), TLC plates (lanes, spots, Rf) and
gel electrophoresis plates (lanes, bands, mass scale), plus Autonumber Reaction and
ChemDraw 17.1's Ctrl+arrow "duplicate the selection across a new arrow".

Primitives follow :mod:`.geometry`: ``("shape", pts, width, dashed, fill, closed)`` with
fill "", "white", "solid", "shade", "fade" or a ``#RRGGBB`` colour, ``("line", ...)`` and
``("text", x, y, text, size, anchor)`` (y = vertical centre, anchor "l" / "c" / "r").
Behaviour follows the ChemDraw 17.1 User Guide (TLC and gel tools pp. 212-216, Autonumber
Reaction p. 12, Ctrl+arrow p. 60).
"""
from __future__ import annotations

import math

from rsstatchem.core.chem_draw.model import Document, GelPlate, Shape, Symbol, TLCPlate
from rsstatchem.core.chem_draw.style import Style, autonumber_text

__all__ = ["shape_primitives", "symbol_primitives", "symbol_radius", "place_symbol",
           "tlc_layout", "tlc_primitives", "tlc_spot_at", "tlc_rf_at", "tlc_lane_at",
           "add_tlc_spot", "move_tlc_spot", "gel_layout", "gel_primitives", "gel_band_at",
           "gel_mass_at", "gel_lane_at", "add_gel_band", "autonumber", "clear_autonumbers",
           "duplicate_across_arrow", "SYMBOL_TEXT", "SYMBOL_NAMES", "SHAPE_NAMES",
           "SPOT_STYLES", "BAND_STYLES", "shape_box", "object_box"]

SYMBOL_TEXT = {"deltaplus": "δ+", "deltaminus": "δ−", "dagger": "†", "ddagger": "‡"}
SYMBOL_NAMES = {"lonepair": "Lone Pair", "electron": "Electron (radical)",
                "plus": "Plus", "minus": "Minus", "circleplus": "Circled Plus",
                "circleminus": "Circled Minus", "radicalcation": "Radical Cation",
                "radicalanion": "Radical Anion", "deltaplus": "Delta Plus (δ+)",
                "deltaminus": "Delta Minus (δ−)", "dagger": "Dagger (†)",
                "ddagger": "Double Dagger (‡, transition state)"}
SHAPE_NAMES = {"circle": "Circle", "ellipse": "Oval", "rect": "Rectangle",
               "roundrect": "Rounded Rectangle", "shadowrect": "Shadowed Rectangle",
               "arc": "Arc"}
SPOT_STYLES = ("filled", "shaded", "hollow")
BAND_STYLES = ("filled", "hollow", "dashed", "doubled")


def _ellipse(cx, cy, rx, ry, a0=0.0, a1=2 * math.pi, n=48, closed=True):
    m = n if closed else n + 1
    return [(cx + rx * math.cos(a0 + (a1 - a0) * k / n),
             cy + ry * math.sin(a0 + (a1 - a0) * k / n)) for k in range(m)]


def _fill_of(o) -> str:
    if getattr(o, "shaded", False):
        return "shade"
    if getattr(o, "filled", False):
        return "solid"
    if getattr(o, "faded", False):
        return "fade"
    return ""


# --- drawing elements ----------------------------------------------------------------------

def shape_primitives(s: Shape, style: Style) -> list[tuple]:
    lw = style.bold_width if s.bold else style.line_width
    x1, y1, x2, y2 = s.x1, s.y1, s.x2, s.y2
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    fill = _fill_of(s)
    if s.kind in ("circle", "ellipse"):
        rx, ry = w / 2, h / 2
        if s.kind == "circle":
            rx = ry = min(w, h) / 2
        return [("shape", _ellipse(cx, cy, rx, ry), lw, s.dashed, fill, True)]
    if s.kind == "arc":
        a0 = math.radians(s.start)
        pts = _ellipse(cx, cy, w / 2, h / 2, a0, a0 + math.radians(s.sweep), 48, False)
        return [("shape", pts, lw, s.dashed, "", False)]
    rect = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    if s.kind == "roundrect":
        r = min(w, h) * 0.18
        pts = []
        for (ccx, ccy), a in (((x2 - r, y1 + r), -90), ((x2 - r, y2 - r), 0),
                              ((x1 + r, y2 - r), 90), ((x1 + r, y1 + r), 180)):
            pts += _ellipse(ccx, ccy, r, r, math.radians(a), math.radians(a + 90), 8, False)
        rect = pts
    out = []
    if s.kind == "shadowrect":
        d = max(2.0, lw * 3)
        out.append(("shape", [(x + d, y + d) for x, y in rect], 0.0, False, "solid", True))
        fill = fill or "white"
    out.append(("shape", rect, lw, s.dashed, fill, True))
    return out


def shape_box(s: Shape) -> tuple[float, float, float, float]:
    return s.x1, s.y1, s.x2, s.y2


# --- chemical symbols ----------------------------------------------------------------------

def symbol_radius(style: Style) -> float:
    """Radius of circled charges (and the hit radius of symbols)."""
    return style.font_size * 0.42


def symbol_primitives(sym: Symbol, style: Style) -> list[tuple]:
    lw = style.line_width
    x, y = sym.x, sym.y
    dot = max(0.8, lw * 1.15)
    k = sym.kind
    out: list[tuple] = []

    def dot_at(px, py):
        out.append(("shape", _ellipse(px, py, dot, dot, n=16), 0.0, False, "solid", True))

    def sign(px, py, plus, r):
        out.append(("line", px - r, py, px + r, py, lw, False))
        if plus:
            out.append(("line", px, py - r, px, py + r, lw, False))
    if k == "lonepair":
        a = sym.angle + math.pi / 2               # the two dots lie across the direction
        d = dot * 1.7
        for s in (1, -1):
            dot_at(x + s * d * math.cos(a), y + s * d * math.sin(a))
    elif k == "electron":
        dot_at(x, y)
    elif k in ("plus", "minus"):
        sign(x, y, k == "plus", style.font_size * 0.3)
    elif k in ("circleplus", "circleminus"):
        r = symbol_radius(style)
        out.append(("shape", _ellipse(x, y, r, r), lw, False, "white", True))
        sign(x, y, k == "circleplus", r * 0.6)
    elif k in ("radicalcation", "radicalanion"):
        r = style.font_size * 0.26
        dot_at(x - r * 1.4, y)
        sign(x + r * 0.9, y, k == "radicalcation", r)
    elif k in SYMBOL_TEXT:
        out.append(("text", x, y, SYMBOL_TEXT[k], style.font_size * 0.9, "c"))
    return out


def place_symbol(doc: Document, aid: int, kind: str) -> Symbol:
    """Add a chemical symbol associated with atom ``aid`` in the free direction around it
    (away from its bonds and the symbols it already has)."""
    a = doc.atoms[aid]
    taken = [math.atan2(doc.atoms[n].y - a.y, doc.atoms[n].x - a.x)
             for n in doc.neighbors(aid)]
    taken += [s.angle for s in doc.symbols_of(aid)]
    cands = [-math.pi / 2 + k * math.pi / 4 for k in range(8)]

    def gap(c):
        if not taken:
            return math.pi
        return min(abs(math.remainder(c - t, 2 * math.pi)) for t in taken)
    ang = max(cands, key=lambda c: (round(gap(c), 4), -abs(math.remainder(c + math.pi / 2,
                                                                          2 * math.pi))))
    st = doc.style
    labelled = bool(a.label or a.symbol != "C" or a.show_carbon)
    r = st.font_size * (0.85 if labelled else 0.55)
    if kind in SYMBOL_TEXT or kind in ("circleplus", "circleminus", "radicalcation",
                                       "radicalanion"):
        r += st.font_size * 0.35
    return doc.add_symbol(a.x + r * math.cos(ang), a.y + r * math.sin(ang), kind, atom=aid,
                          angle=ang)


# --- TLC plates ----------------------------------------------------------------------------

def tlc_layout(t: TLCPlate, bond_length: float) -> dict:
    """Lane x positions, origin / front y and the spot ellipses of a plate."""
    n = max(1, len(t.lanes))
    w = t.x2 - t.x1
    h = t.y2 - t.y1
    lane_w = w / n
    oy = t.y2 - h * t.origin
    fy = t.y2 - h * t.front
    xs = [t.x1 + (k + 0.5) * lane_w for k in range(n)]
    dw = max(3.0, min(lane_w * 0.42, bond_length * 0.5))
    spots = []
    for k, lane in enumerate(t.lanes):
        for j, sp in enumerate(lane):
            sw = sp.get("w") or dw
            sh = sp.get("h") or sw * 0.7
            cy = oy - sp.get("rf", 0.5) * (oy - fy)
            spots.append((k, j, xs[k], cy, sw, sh, sp.get("tail", 0.0)))
    return {"xs": xs, "origin": oy, "front": fy, "lane_w": lane_w, "spots": spots}


def _spot_outline(cx, cy, w, h, tail) -> list[tuple[float, float]]:
    """Ellipse, stretched downward by ``tail`` (a smear / crescent)."""
    top = _ellipse(cx, cy, w / 2, h / 2, math.pi, 2 * math.pi, 16, False)
    bottom = _ellipse(cx, cy + max(0.0, tail), w / 2, h / 2, 0.0, math.pi, 16, False)
    return top + bottom


def tlc_primitives(t: TLCPlate, style: Style) -> list[tuple]:
    lw = style.line_width
    lay = tlc_layout(t, style.bond_length)
    out: list[tuple] = []
    rect = [(t.x1, t.y1), (t.x2, t.y1), (t.x2, t.y2), (t.x1, t.y2)]
    if t.show_border:
        out.append(("shape", rect, lw, False, "" if t.transparent else "white", True))
    elif not t.transparent:
        out.append(("shape", rect, 0.0, False, "white", True))
    tick = max(2.0, style.bond_length * 0.12)
    if t.show_origin:
        out.append(("line", t.x1, lay["origin"], t.x2, lay["origin"], lw * 0.8, False))
    if t.show_front:
        out.append(("line", t.x1, lay["front"], t.x2, lay["front"], lw * 0.8, True))
    if t.show_ticks:
        for x in lay["xs"]:
            out.append(("line", x, lay["origin"] - tick / 2, x, lay["origin"] + tick / 2,
                        lw * 0.8, False))
        for y in (lay["origin"], lay["front"]):
            out.append(("line", t.x1 - tick, y, t.x1, y, lw * 0.8, False))
    for k, j, cx, cy, w, h, tail in lay["spots"]:
        sp = t.lanes[k][j]
        fill = {"shaded": "shade", "hollow": "white"}.get(sp.get("style", "filled"), "solid")
        if sp.get("color") and fill == "solid":
            fill = sp["color"]
        out.append(("shape", _spot_outline(cx, cy, w, h, tail), lw * 0.8, False, fill, True))
        if t.show_rf or sp.get("show_rf"):
            out.append(("text", cx + w / 2 + 2.0, cy, f"{sp.get('rf', 0.5):.2f}",
                        style.caption_size * 0.75, "l"))
    return out


def tlc_lane_at(t: TLCPlate, x: float) -> int:
    n = max(1, len(t.lanes))
    k = int((x - t.x1) / ((t.x2 - t.x1) / n))
    return max(0, min(n - 1, k))


def tlc_rf_at(t: TLCPlate, y: float, bond_length: float = 30.0) -> float:
    lay = tlc_layout(t, bond_length)
    span = (lay["origin"] - lay["front"]) or 1.0
    return round((lay["origin"] - y) / span, 2)


def tlc_spot_at(t: TLCPlate, x: float, y: float, bond_length: float, tol: float = 2.0):
    """(lane, spot index) of the spot under (x, y), or None."""
    for k, j, cx, cy, w, h, tail in reversed(tlc_layout(t, bond_length)["spots"]):
        if abs(x - cx) <= w / 2 + tol and cy - h / 2 - tol <= y <= cy + h / 2 + tail + tol:
            return k, j
    return None


def add_tlc_spot(t: TLCPlate, x: float, y: float, bond_length: float, **kw) -> tuple:
    k = tlc_lane_at(t, x)
    sp = {"rf": max(0.0, min(1.2, tlc_rf_at(t, y, bond_length)))}
    sp.update(kw)
    t.lanes[k].append(sp)
    return k, len(t.lanes[k]) - 1


def move_tlc_spot(t: TLCPlate, lane: int, j: int, x: float, y: float,
                  bond_length: float) -> tuple[int, int]:
    """Drag a spot: its Rf follows y; dropped over another lane it moves there."""
    sp = t.lanes[lane][j]
    sp["rf"] = max(-0.2, min(1.2, tlc_rf_at(t, y, bond_length)))
    k = tlc_lane_at(t, x)
    if k != lane:
        del t.lanes[lane][j]
        t.lanes[k].append(sp)
        return k, len(t.lanes[k]) - 1
    return lane, j


# --- gel electrophoresis --------------------------------------------------------------------

def _gel_t(g: GelPlate, mass: float) -> float | None:
    lo, hi = max(g.lo, 1e-9), max(g.hi, g.lo * 1.0001)
    if mass <= 0 or mass < lo * 0.999 or mass > hi * 1.001:
        return None
    return (math.log10(hi) - math.log10(mass)) / (math.log10(hi) - math.log10(lo))


def gel_layout(g: GelPlate, bond_length: float) -> dict:
    n = max(1, len(g.lanes))
    lane_w = (g.x2 - g.x1) / n
    h = g.y2 - g.y1
    top, span = g.y1 + h * 0.08, h * 0.86
    xs = [g.x1 + (k + 0.5) * lane_w for k in range(n)]
    bh = max(1.6, bond_length * 0.1)
    bands = []
    for k, lane in enumerate(g.lanes):
        for j, b in enumerate(lane.get("bands", [])):
            t = _gel_t(g, b.get("mass", 0))
            if t is None:
                continue                         # out of range: hidden, as in ChemDraw
            bands.append((k, j, xs[k], top + span * t, lane_w * 0.7, bh))
    return {"xs": xs, "lane_w": lane_w, "top": top, "span": span, "bands": bands}


def gel_mass_at(g: GelPlate, y: float, bond_length: float = 30.0) -> float:
    lay = gel_layout(g, bond_length)
    t = max(0.0, min(1.0, (y - lay["top"]) / (lay["span"] or 1.0)))
    lo, hi = math.log10(max(g.lo, 1e-9)), math.log10(max(g.hi, g.lo * 1.0001))
    return float(f"{10 ** (hi - t * (hi - lo)):.3g}")


def gel_lane_at(g: GelPlate, x: float) -> int:
    n = max(1, len(g.lanes))
    return max(0, min(n - 1, int((x - g.x1) / ((g.x2 - g.x1) / n))))


def gel_band_at(g: GelPlate, x: float, y: float, bond_length: float, tol: float = 2.0):
    for k, j, cx, cy, w, h in reversed(gel_layout(g, bond_length)["bands"]):
        if abs(x - cx) <= w / 2 + tol and abs(y - cy) <= h / 2 + tol:
            return k, j
    return None


def add_gel_band(g: GelPlate, lane: int, mass: float, **kw) -> int:
    b = {"mass": float(mass)}
    b.update(kw)
    g.lanes[lane].setdefault("bands", []).append(b)
    return len(g.lanes[lane]["bands"]) - 1


def _fmt_mass(m: float) -> str:
    return f"{m:g}"


def gel_primitives(g: GelPlate, style: Style) -> list[tuple]:
    lw = style.line_width
    lay = gel_layout(g, style.bond_length)
    rect = [(g.x1, g.y1), (g.x2, g.y1), (g.x2, g.y2), (g.x1, g.y2)]
    out: list[tuple] = [("shape", rect, lw, False, "#F4F6F8", True)]
    wh = max(2.0, (g.y2 - g.y1) * 0.03)
    for x in lay["xs"]:                                   # wells
        ww = lay["lane_w"] * 0.7
        out.append(("shape", [(x - ww / 2, g.y1 + wh * 0.6), (x + ww / 2, g.y1 + wh * 0.6),
                              (x + ww / 2, g.y1 + wh * 1.6), (x - ww / 2, g.y1 + wh * 1.6)],
                    lw * 0.7, False, "white", True))
    for k, j, cx, cy, w, h in lay["bands"]:
        b = g.lanes[k]["bands"][j]
        st = b.get("style", "filled")
        pts = [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2), (cx + w / 2, cy + h / 2),
               (cx - w / 2, cy + h / 2)]
        fill = b.get("color") or "solid"
        if st == "hollow":
            out.append(("shape", pts, lw * 0.8, False, "white", True))
        elif st == "dashed":
            out.append(("shape", pts, lw * 1.4, True, "", True))
        elif st == "doubled":
            for d in (-h * 0.8, h * 0.8):
                out.append(("shape", [(px, py + d) for px, py in pts], 0.0, False, fill,
                            True))
        else:
            out.append(("shape", pts, 0.0, False, fill, True))
        if b.get("show"):
            out.append(("text", cx, cy - h - style.caption_size * 0.4,
                        _fmt_mass(b.get("mass", 0)), style.caption_size * 0.7, "c"))
    size = style.caption_size * 0.7
    if g.show_scale:
        x = g.x2 + 3.0
        out.append(("line", x, lay["top"], x, lay["top"] + lay["span"], lw * 0.8, False))
        lo, hi = math.log10(max(g.lo, 1e-9)), math.log10(max(g.hi, g.lo * 1.0001))
        ticks = [10 ** e for e in range(math.ceil(lo - 1e-9), math.floor(hi + 1e-9) + 1)]
        if len(ticks) < 2:
            ticks = [10 ** lo, 10 ** hi]
        for m in ticks:
            t = _gel_t(g, m)
            if t is None:
                continue
            y = lay["top"] + lay["span"] * t
            out.append(("line", x, y, x + 3.0, y, lw * 0.8, False))
            if g.units == "log":
                text = f"{math.log10(m):g}"
            elif g.units == "distance":
                text = f"{t:.2f}"
            else:
                text = _fmt_mass(m)
            out.append(("text", x + 4.5, y, text, size, "l"))
        unit = {"log": "log(kDa)", "distance": "dist."}.get(g.units, "kDa")
        out.append(("text", x + 4.5, g.y1 + size * 0.2, unit, size, "l"))
    if g.show_labels:
        for k, x in enumerate(lay["xs"]):
            label = g.lanes[k].get("label", "")
            if label:
                out.append(("text", x, g.y1 - size * 0.9, label, size, "c"))
    return out


def object_box(doc: Document, key) -> tuple[float, float, float, float] | None:
    """Bounding box of a shape / TLC / gel / clipart / symbol object (model units)."""
    kind, i = key
    o = doc.get(key)
    if o is None:
        return None
    if kind in ("shape", "tlc", "gel"):
        return o.x1, o.y1, o.x2, o.y2
    if kind == "clipart":
        return o.x, o.y, o.x + o.w, o.y + o.h
    if kind == "symbol":
        r = symbol_radius(doc.style) * (1.6 if o.kind in SYMBOL_TEXT or
                                        o.kind.startswith("radical") else 1.0)
        return o.x - r, o.y - r, o.x + r, o.y + r
    return None


# --- Autonumber Reaction --------------------------------------------------------------------

def _rows(doc: Document, comps):
    """Fragments in reading order: rows (by vertical overlap) top to bottom, then x."""
    boxes = []
    for comp in comps:
        xs = [doc.atoms[a].x for a in comp]
        ys = [doc.atoms[a].y for a in comp]
        boxes.append((comp, min(xs), min(ys), max(xs), max(ys)))
    boxes.sort(key=lambda b: (b[2] + b[4]) / 2)
    rows: list[list] = []
    for b in boxes:
        for row in rows:
            top = min(r[2] for r in row)
            bottom = max(r[4] for r in row)
            if b[2] <= bottom and b[4] >= top:
                row.append(b)
                break
        else:
            rows.append([b])
    out = []
    for row in rows:
        out += sorted(row, key=lambda b: b[1])
    return out


def clear_autonumbers(doc: Document, atom_ids=None) -> int:
    """Structure > Clear Reaction Numbers (for the fragments of ``atom_ids`` or all)."""
    keys = []
    for t in doc.texts.values():
        if t.autonumber:
            if atom_ids is None or t.id in getattr(doc, "_autonum_scope", ()) or \
                    _caption_owner(doc, t, atom_ids):
                keys.append(("text", t.id))
    doc.remove(keys)
    return len(keys)


def _caption_owner(doc: Document, t, atom_ids) -> bool:
    for comp in doc.fragments():
        if not set(comp) & set(atom_ids):
            continue
        xs = [doc.atoms[a].x for a in comp]
        ys = [doc.atoms[a].y for a in comp]
        if min(xs) - 20 <= t.x <= max(xs) + 20 and max(ys) <= t.y <= max(ys) + 60:
            return True
    return False


def autonumber(doc: Document, atom_ids=None) -> list[int]:
    """Structure > Autonumber Reaction: bold compound numbers centred under every molecule
    (of the selection, or all) in reading order, in the document's numeric style (Arabic,
    Roman or alphabetic, from the starting number). Earlier numbers are replaced. Returns
    the ids of the new captions."""
    comps = doc.fragments()
    if atom_ids is not None:
        sel = set(atom_ids)
        comps = [c for c in comps if set(c) & sel]
    clear_autonumbers(doc, None if atom_ids is None else
                      [a for c in comps for a in c])
    st = doc.style
    size = st.caption_size
    ids = []
    n = st.autonumber_start
    for comp, x1, _y1, x2, y2 in _rows(doc, comps):
        text = autonumber_text(n, st.autonumber_style)
        # caption top-left so that the text is centred under the molecule
        w = len(text) * size * 0.56
        t = doc.add_text((x1 + x2) / 2 - w / 2, y2 + st.bond_length * 0.45 + size * 0.2,
                         text, chemical=False, bold=True, autonumber=n)
        ids.append(t.id)
        n += 1
    return ids


# --- Ctrl+arrow: duplicate across a new arrow -----------------------------------------------

def duplicate_across_arrow(doc: Document, keys, direction: str) -> list:
    """ChemDraw 17.1 (p. 60): Ctrl+Right/Left/Up/Down adds a reaction arrow next to the
    selection and a copy of the selection on the other side of it. Returns the new keys
    (arrow first)."""
    keys = doc.with_bonds(keys)
    bb = doc.bbox(keys)
    if bb is None:
        return []
    x1, y1, x2, y2 = bb
    bl = doc.style.bond_length
    gap, length = bl * 0.8, bl * 3.0
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if direction == "right":
        ax1, ay1, ax2, ay2 = x2 + gap, cy, x2 + gap + length, cy
        dx, dy = (x2 - x1) + 2 * gap + length, 0.0
    elif direction == "left":
        ax1, ay1, ax2, ay2 = x1 - gap, cy, x1 - gap - length, cy
        dx, dy = -((x2 - x1) + 2 * gap + length), 0.0
    elif direction == "down":
        ax1, ay1, ax2, ay2 = cx, y2 + gap, cx, y2 + gap + length
        dx, dy = 0.0, (y2 - y1) + 2 * gap + length
    else:
        ax1, ay1, ax2, ay2 = cx, y1 - gap, cx, y1 - gap - length
        dx, dy = 0.0, -((y2 - y1) + 2 * gap + length)
    ar = doc.add_arrow(ax1, ay1, ax2, ay2, "forward")
    sub = doc.subset(keys)
    new = doc.paste(sub, dx, dy)
    return [("arrow", ar.id)] + new

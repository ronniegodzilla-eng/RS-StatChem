"""ChemDraw CDXML (XML) import and export for drawing documents (no Qt).

Supported subset: fragments with nodes (element, charge, isotope, explicit H, radicals,
labels / nicknames) and bonds (order 1 / 1.5 / 2 / 3, wedge, hashed wedge, bold, dashed,
wavy, hollow wedge, hashed, dative, double-bond position), captions (``<t>``), reaction
arrows (``<arrow>``, and legacy ``<graphic>`` lines with an arrow type), "+" signs, bracket
pairs, Pen-tool curves (``<curve>`` with ``CurvePoints`` / ``CurveType`` / arrowheads; curved
electron-pushing arrows are written as two-point curves) and orbitals (``<graphic
GraphicType="Orbital">``). CDXML and our documents both use points with y pointing down, so
coordinates are copied unchanged.

Nicknames are written as labelled generic nodes (their text survives, their expansion is
recomputed by RS-StatChem on import).

The newer objects are written as their nearest ChemDraw element - drawing elements as
``<graphic GraphicType="Oval" / "Rectangle" / "Arc">``, chemical symbols as ``<graphic
GraphicType="Symbol">`` (delta charges as captions), TLC plates as ``<tlcplate>`` with
``<tlclane>`` / ``<tlcspot>``, gel plates and clipware as graphics / curve groups, the extra
arrow kinds as ``<arrow>`` with ``NoGo`` / ``AngularSize`` / ``LineType`` - and carry an
``<objecttag Name="RS-StatChem">`` with the exact object (JSON), so a CDXML round trip keeps
everything (mechanism-arrow anchors, polymer bracket properties, circled charges, attachment
points, autonumber captions, stereo display settings). Other programs ignore the tags.
"""
from __future__ import annotations

import json
import math
import statistics
import xml.etree.ElementTree as ET
from dataclasses import asdict

from rsstatchem.core.chem_draw import clipart as clip
from rsstatchem.core.chem_draw import objects as obj
from rsstatchem.core.chem_draw.geometry import arrow_path_points
from rsstatchem.core.chem_draw.model import (Arrow, Bracket, Clipart, GelPlate, Shape, Symbol,
                                             TLCPlate, Document, _from_dict)
from rsstatchem.core.chem_draw.style import preset

__all__ = ["to_cdxml", "from_cdxml", "save_cdxml", "load_cdxml"]

_DISPLAY_OUT = {"wedge": "WedgeBegin", "hash": "WedgedHashBegin", "bold": "Bold",
                "dashed": "Dash", "wavy": "Wavy", "hollow": "HollowWedgeBegin",
                "hashed": "Hash"}
_DISPLAY_IN = {"WedgeBegin": ("wedge", False), "WedgeEnd": ("wedge", True),
               "WedgedHashBegin": ("hash", False), "WedgedHashEnd": ("hash", True),
               "Hash": ("hashed", False), "Dash": ("dashed", False), "Bold": ("bold", False),
               "Wavy": ("wavy", False), "HollowWedgeBegin": ("hollow", False),
               "HollowWedgeEnd": ("hollow", True)}
_DOUBLE_OUT = {"center": "Center", "left": "Left", "right": "Right"}
_RADICAL_OUT = {1: "Doublet", 2: "Singlet"}
_RADICAL_IN = {"Doublet": 1, "Singlet": 2, "Triplet": 2}
_BRACKET_OUT = {"square": "SquarePair", "round": "RoundPair", "curly": "CurlyPair"}
_FACE_FORMULA = 96
# CDX Curve_Type bits
_CT = {"closed": 1, "dashed": 2, "bold": 4, "arrow_end": 8, "arrow_start": 16,
       "half_end": 32, "half_start": 64, "filled": 128, "shaded": 256, "doubled": 512}
_ORB_OUT = {"s": "s", "p": "p", "lobe": "lobe", "hybrid": "hybridPlus", "dxy": "dxy",
            "dz2": "dz2Plus"}
_ORB_IN = {"s": "s", "oval": "s", "p": "p", "lobe": "lobe", "hybridPlus": "hybrid",
           "hybridMinus": "hybrid", "dxy": "dxy", "dz2Plus": "dz2", "dz2Minus": "dz2"}
_FONT_ID = "3"
TAG = "RS-StatChem"
_SYMBOL_OUT = {"lonepair": "LonePair", "electron": "Electron", "plus": "Plus",
               "minus": "Minus", "circleplus": "CirclePlus", "circleminus": "CircleMinus",
               "radicalcation": "RadicalCation", "radicalanion": "RadicalAnion",
               "dagger": "Dagger", "ddagger": "DoubleDagger"}
_SYMBOL_IN = {v: k for k, v in _SYMBOL_OUT.items()}
_OLD_ARROWS = ("forward", "equilibrium", "retro", "resonance", "line")
_STYLE_EXTRA = ("name", "show_stereo", "stereo_parens", "autonumber_style",
                "autonumber_start", "chain_angle", "page_width", "page_height",
                "color_by_element", "font_family")


def _tag(el, data: dict) -> None:
    """Attach the exact RS-StatChem object as an ``<objecttag>``."""
    ET.SubElement(el, "objecttag", {"Name": TAG, "TagType": "String",
                                    "Value": json.dumps(data, separators=(",", ":"),
                                                        ensure_ascii=False)})


def _tag_of(el) -> dict | None:
    for c in el.findall("objecttag"):
        if c.get("Name") == TAG:
            try:
                data = json.loads(c.get("Value", ""))
            except ValueError:
                return None
            return data if isinstance(data, dict) else None
    return None


def _box(x1, y1, x2, y2) -> str:
    return " ".join(_f(v) for v in (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))


def _symbol(z: int) -> str:
    from rdkit import Chem
    return Chem.GetPeriodicTable().GetElementSymbol(int(z))


def _number(sym: str) -> int:
    from rdkit import Chem
    return Chem.GetPeriodicTable().GetAtomicNumber(sym)


def _f(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


# --- export -------------------------------------------------------------------------------

def to_cdxml(doc: Document) -> str:
    st = doc.style
    root = ET.Element("CDXML", {
        "CreationProgram": "RS-StatChem", "BondLength": _f(st.bond_length),
        "LineWidth": _f(st.line_width), "BoldWidth": _f(st.bold_width),
        "HashSpacing": _f(st.hash_spacing), "MarginWidth": _f(st.margin),
        "BondSpacing": _f(st.bond_spacing), "LabelFont": _FONT_ID,
        "LabelSize": _f(st.font_size), "LabelFace": str(_FACE_FORMULA),
        "CaptionFont": _FONT_ID, "CaptionSize": _f(st.caption_size),
        "ChainAngle": _f(st.chain_angle),
        "ShowAtomStereo": "yes" if st.show_stereo else "no",
        "ShowBondStereo": "yes" if st.show_stereo else "no"})
    _tag(root, {"style": {k: getattr(st, k) for k in _STYLE_EXTRA}})
    colors: list[str] = []

    def color_index(c: str) -> str | None:
        if not c:
            return None
        if c.upper() not in colors:
            colors.append(c.upper())
        return str(colors.index(c.upper()) + 2)        # 0 = black, 1 = white

    fonts = ET.SubElement(root, "fonttable")
    ET.SubElement(fonts, "font", {"id": _FONT_ID, "charset": "iso-8859-1",
                                  "name": st.font_family})
    ctable = ET.SubElement(root, "colortable")
    page = ET.SubElement(root, "page", {"id": "1", "HeightPages": "1", "WidthPages": "1"})
    ids = iter(range(10000, 10 ** 9))

    for comp in doc.fragments():
        frag = ET.SubElement(page, "fragment", {"id": str(next(ids))})
        for aid in comp:
            a = doc.atoms[aid]
            attrs = {"id": str(aid), "p": f"{_f(a.x)} {_f(a.y)}"}
            if a.symbol != "C":
                try:
                    attrs["Element"] = str(_number(a.symbol))
                except Exception:  # noqa: BLE001 - generic symbols (R, X...)
                    attrs["NodeType"] = "GenericNickname"
                    attrs["GenericNickname"] = a.symbol
            if a.charge:
                attrs["Charge"] = str(a.charge)
            if a.isotope:
                attrs["Isotope"] = str(a.isotope)
            if a.explicit_h is not None:
                attrs["NumHydrogens"] = str(a.explicit_h)
            if a.radicals:
                attrs["Radical"] = _RADICAL_OUT.get(a.radicals, "Doublet")
            if a.label:
                attrs["NodeType"] = "GenericNickname"
                attrs["GenericNickname"] = a.label
                attrs.pop("Element", None)
            ci = color_index(a.color)
            if ci:
                attrs["color"] = ci
            n = ET.SubElement(frag, "n", attrs)
            extra = {k: getattr(a, k) for k in ("charge_circle", "attachment")
                     if getattr(a, k)}
            if not a.show_stereo:
                extra["show_stereo"] = False
            if a.label:
                extra["label"] = a.label
            if extra:
                _tag(n, {"atom": extra})
            text = a.label or (a.symbol if a.symbol != "C" or a.show_carbon else "")
            if text:
                h = a.explicit_h
                if not a.label and a.symbol != "C" and h:
                    text += "H" + (str(h) if h > 1 else "")
                t = ET.SubElement(n, "t", {"p": f"{_f(a.x - st.font_size * 0.3)} "
                                                f"{_f(a.y + st.font_size * 0.35)}",
                                           "LabelJustification": "Left"})
                s = ET.SubElement(t, "s", {"font": _FONT_ID, "size": _f(st.font_size),
                                           "face": str(_FACE_FORMULA)})
                s.text = text
        for b in doc.bonds.values():
            if b.a not in comp or b.b not in comp:
                continue
            attrs = {"id": str(b.id), "B": str(b.a), "E": str(b.b)}
            if b.order != 1:
                attrs["Order"] = {1.5: "1.5", 2: "2", 3: "3"}.get(b.order, "1")
            if b.style == "dative":
                attrs["Order"] = "dative"
            if b.style in _DISPLAY_OUT:
                attrs["Display"] = _DISPLAY_OUT[b.style]
            if b.order == 2 and b.double_pos in _DOUBLE_OUT:
                attrs["DoublePosition"] = _DOUBLE_OUT[b.double_pos]
            ci = color_index(b.color)
            if ci:
                attrs["color"] = ci
            ET.SubElement(frag, "b", attrs)

    def caption(x, y, text, size, face=0, color="", bold=False, italic=False,
                underline=False):
        face |= (1 if bold else 0) | (2 if italic else 0) | (4 if underline else 0)
        attrs = {"id": str(next(ids)), "p": f"{_f(x)} {_f(y)}"}
        ci = color_index(color)
        if ci:
            attrs["color"] = ci
        t = ET.SubElement(page, "t", attrs)
        s = ET.SubElement(t, "s", {"font": _FONT_ID, "size": _f(size), "face": str(face)})
        s.text = text
        return t

    for t in doc.texts.values():
        size = t.size or st.caption_size
        el = caption(t.x, t.y + size * 0.95, t.text, size, _FACE_FORMULA if t.chemical else 0,
                     t.color, t.bold, t.italic, t.underline)
        if t.autonumber or t.font:
            _tag(el, {"text": {"autonumber": t.autonumber, "font": t.font}})
    for p in doc.pluses.values():
        size = st.caption_size * 1.2
        caption(p.x - size * 0.3, p.y + size * 0.35, "+", size, color=p.color)
    for ar in doc.arrows.values():
        if ar.kind in ("curved", "fishhook", "double"):     # written as <curve> below
            continue
        attrs = {"id": str(next(ids)),
                 "BoundingBox": " ".join(_f(v) for v in (min(ar.x1, ar.x2), min(ar.y1, ar.y2),
                                                         max(ar.x1, ar.x2), max(ar.y1, ar.y2))),
                 "Head3D": f"{_f(ar.x2)} {_f(ar.y2)} 0", "Tail3D": f"{_f(ar.x1)} {_f(ar.y1)} 0",
                 "ArrowheadType": "Solid", "ArrowheadHead": "Full", "FillType": "None"}
        if ar.kind == "equilibrium":
            attrs.update(ArrowheadHead="HalfLeft", ArrowheadTail="HalfLeft",
                         ArrowShaftSpacing="400")
        elif ar.kind == "resonance":
            attrs.update(ArrowheadTail="Full")
        elif ar.kind == "retro":
            attrs.update(ArrowheadType="Hollow", ArrowShaftSpacing="400")
        elif ar.kind == "fishhook":
            attrs.update(ArrowheadHead="HalfLeft")
        elif ar.kind == "line":
            attrs.pop("ArrowheadHead")
        elif ar.kind == "nogo":
            attrs["NoGo"] = "Cross"
        elif ar.kind == "unbalanced":
            attrs.update(ArrowheadHead="HalfLeft", ArrowheadTail="HalfLeft",
                         ArrowShaftSpacing="400", ArrowEquilibriumRatio="200")
        elif ar.kind == "block":
            attrs.update(ArrowheadType="Hollow", ArrowShaftSpacing="600",
                         FillType={"filled": "Solid", "shaded": "Shaded"}.get(ar.fill,
                                                                               "None"))
        elif ar.kind == "dipole":
            attrs["Dipole"] = "yes"
        elif ar.kind == "arc":
            pts = arrow_path_points(ar)
            mid = pts[len(pts) // 2]
            attrs.update(AngularSize=_f(ar.sweep), Center3D=f"{_f(mid[0])} {_f(mid[1])} 0")
        if ar.dashed:
            attrs["LineType"] = "Dashed"
        elif ar.bold:
            attrs["LineType"] = "Bold"
        ci = color_index(ar.color)
        if ci:
            attrs["color"] = ci
        el = ET.SubElement(page, "arrow", attrs)
        if ar.kind not in _OLD_ARROWS or ar.dashed or ar.bold or ar.tail_ref or ar.head_ref:
            _tag(el, {"arrow": asdict(ar)})
        mx, my = (ar.x1 + ar.x2) / 2, (ar.y1 + ar.y2) / 2
        for text, dy in ((ar.above, -st.caption_size * 0.6), (ar.below, st.caption_size * 1.6)):
            if text:
                w = len(text) * st.caption_size * 0.25
                caption(mx - w, my + dy, text, st.caption_size, _FACE_FORMULA)
    for c in doc.curves.values():
        el = _curve_out(page, next(ids), c.points, c, color_index(c.color))
        if c.faded:
            _tag(el, {"curve": {"faded": True}})
    for ar in doc.arrows.values():
        if ar.kind not in ("curved", "fishhook", "double"):
            continue
        pts = arrow_path_points(ar, 2)
        (x0, y0), (qx, qy), (x1, y1) = pts[0], _quad_control(ar), pts[-1]
        c0 = (x0 + 2 / 3 * (qx - x0), y0 + 2 / 3 * (qy - y0))
        c1 = (x1 + 2 / 3 * (qx - x1), y1 + 2 / 3 * (qy - y1))
        flat = [(x0, y0), (x0, y0), c0, c1, (x1, y1), (x1, y1)]
        el = _curve_out(page, next(ids), flat, None, color_index(ar.color),
                        head="half" if ar.kind == "fishhook" else "full",
                        tail="full" if ar.kind == "double" else "")
        _tag(el, {"arrow": asdict(ar)})
    for o in doc.orbitals.values():
        L = o.size or st.bond_length
        cx, cy = o.x, o.y
        mx, my = cx + L * math.cos(o.angle), cy + L * math.sin(o.angle)
        w = L * 0.45
        nx, ny = cx - w * math.sin(o.angle), cy + w * math.cos(o.angle)
        kind = _ORB_OUT.get(o.kind, "p")
        if o.fill == "shaded":
            kind += "Shaded"
        elif o.fill in ("filled", "phase"):
            kind += "Filled"
        attrs = {"id": str(next(ids)), "GraphicType": "Orbital", "OrbitalType": kind,
                 "BoundingBox": f"{_f(cx - L)} {_f(cy - L)} {_f(cx + L)} {_f(cy + L)}",
                 "Center3D": f"{_f(cx)} {_f(cy)} 0",
                 "MajorAxisEnd3D": f"{_f(mx)} {_f(my)} 0",
                 "MinorAxisEnd3D": f"{_f(nx)} {_f(ny)} 0"}
        ci = color_index(o.color)
        if ci:
            attrs["color"] = ci
        ET.SubElement(page, "graphic", attrs)
    for br in doc.brackets.values():
        btype = _BRACKET_OUT.get(br.kind, "SquarePair")
        if br.sides != "both":
            btype = btype.replace("Pair", "")
        el = ET.SubElement(page, "graphic", {
            "id": str(next(ids)), "GraphicType": "Bracket", "BracketType": btype,
            "BoundingBox": f"{_f(br.x1)} {_f(br.y1)} {_f(br.x2)} {_f(br.y2)}"})
        _tag(el, {"bracket": asdict(br)})
    _objects_out(doc, page, ids, color_index, caption)
    for c in colors:
        r, g, b = (int(c[i:i + 2], 16) / 255 for i in (1, 3, 5))
        ET.SubElement(ctable, "color", {"r": _f(r), "g": _f(g), "b": _f(b)})
    ET.indent(root, " ")
    body = ET.tostring(root, encoding="unicode")
    return ('<?xml version="1.0" encoding="UTF-8" ?>\n'
            '<!DOCTYPE CDXML SYSTEM "http://www.cambridgesoft.com/xml/cdxml.dtd" >\n'
            + body + "\n")


def _quad_control(ar) -> tuple[float, float]:
    dx, dy = ar.x2 - ar.x1, ar.y2 - ar.y1
    L = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / L, dx / L
    return (ar.x1 + ar.x2) / 2 + nx * 2 * ar.bend, (ar.y1 + ar.y2) / 2 + ny * 2 * ar.bend


def _objects_out(doc: Document, page, ids, color_index, caption) -> None:
    """Drawing elements, symbols, TLC / gel plates and clipware."""
    for sh in doc.shapes.values():
        cx, cy = (sh.x1 + sh.x2) / 2, (sh.y1 + sh.y2) / 2
        attrs = {"id": str(next(ids)), "BoundingBox": _box(sh.x1, sh.y1, sh.x2, sh.y2)}
        if sh.kind in ("circle", "ellipse", "arc"):
            rx, ry = (sh.x2 - sh.x1) / 2, (sh.y2 - sh.y1) / 2
            if sh.kind == "circle":
                rx = ry = min(rx, ry)
            attrs.update(GraphicType="Arc" if sh.kind == "arc" else "Oval",
                         Center3D=f"{_f(cx)} {_f(cy)} 0",
                         MajorAxisEnd3D=f"{_f(cx + rx)} {_f(cy)} 0",
                         MinorAxisEnd3D=f"{_f(cx)} {_f(cy + ry)} 0")
            if sh.kind == "arc":
                attrs.update(AngularSize=_f(sh.sweep), ArcType="Plain")
            else:
                flags = [n for n, on in (("Circle", sh.kind == "circle"),
                                         ("Shaded", sh.shaded), ("Filled", sh.filled),
                                         ("Dashed", sh.dashed), ("Bold", sh.bold)) if on]
                attrs["OvalType"] = " ".join(flags) or "Plain"
        else:
            flags = [n for n, on in (("RoundEdge", sh.kind == "roundrect"),
                                     ("Shadow", sh.kind == "shadowrect"),
                                     ("Shaded", sh.shaded), ("Filled", sh.filled),
                                     ("Dashed", sh.dashed), ("Bold", sh.bold)) if on]
            attrs.update(GraphicType="Rectangle", RectangleType=" ".join(flags) or "Plain")
        ci = color_index(sh.color)
        if ci:
            attrs["color"] = ci
        _tag(ET.SubElement(page, "graphic", attrs), {"shape": asdict(sh)})
    r = obj.symbol_radius(doc.style)
    for sy in doc.symbols.values():
        if sy.kind in _SYMBOL_OUT:
            attrs = {"id": str(next(ids)), "GraphicType": "Symbol",
                     "SymbolType": _SYMBOL_OUT[sy.kind],
                     "BoundingBox": _box(sy.x - r, sy.y - r, sy.x + r, sy.y + r)}
            ci = color_index(sy.color)
            if ci:
                attrs["color"] = ci
            el = ET.SubElement(page, "graphic", attrs)
        else:
            size = doc.style.font_size * 0.9
            el = caption(sy.x - size * 0.4, sy.y + size * 0.35, obj.SYMBOL_TEXT[sy.kind], size,
                         color=sy.color)
        _tag(el, {"symbol": asdict(sy)})
    for t in doc.tlcs.values():
        el = ET.SubElement(page, "tlcplate", {
            "id": str(next(ids)), "BoundingBox": _box(t.x1, t.y1, t.x2, t.y2),
            "TopLeft": f"{_f(t.x1)} {_f(t.y1)}", "TopRight": f"{_f(t.x2)} {_f(t.y1)}",
            "BottomRight": f"{_f(t.x2)} {_f(t.y2)}", "BottomLeft": f"{_f(t.x1)} {_f(t.y2)}",
            "OriginFraction": _f(t.origin), "SolventFrontFraction": _f(1 - t.front),
            "ShowOrigin": "yes" if t.show_origin else "no",
            "ShowSolventFront": "yes" if t.show_front else "no",
            "ShowBorders": "yes" if t.show_border else "no",
            "ShowSideTicks": "yes" if t.show_ticks else "no",
            "Transparent": "yes" if t.transparent else "no"})
        for lane in t.lanes:
            ln = ET.SubElement(el, "tlclane", {"id": str(next(ids))})
            for sp in lane:
                ET.SubElement(ln, "tlcspot", {
                    "id": str(next(ids)), "Rf": _f(sp.get("rf", 0.5)),
                    "Width": _f(sp.get("w", 0)), "Height": _f(sp.get("h", 0)),
                    "CurveType": str(_CT["filled"] if sp.get("style", "filled") == "filled"
                                     else (_CT["shaded"] if sp.get("style") == "shaded"
                                           else 0)),
                    "Tail": _f(sp.get("tail", 0)),
                    "ShowRf": "yes" if sp.get("show_rf") else "no"})
        _tag(el, {"tlc": asdict(t)})
    for g in doc.gels.values():
        el = ET.SubElement(page, "graphic", {
            "id": str(next(ids)), "GraphicType": "Rectangle", "RectangleType": "Plain",
            "BoundingBox": _box(g.x1, g.y1, g.x2, g.y2)})
        _tag(el, {"gel": asdict(g)})
    for c in doc.cliparts.values():
        grp = ET.SubElement(page, "group", {"id": str(next(ids))})
        for prim in clip.clipart_primitives(c, doc.style.line_width):
            if prim[0] != "shape" or prim[2] <= 0:
                continue
            pts = prim[1] + ([prim[1][0]] if prim[5] else [])
            flat = []
            for x, y in pts:
                flat += [(x, y)] * 3
            _curve_out(grp, next(ids), flat, None, color_index(c.color))
        _tag(grp, {"clipart": asdict(c)})


def _curve_out(page, cid: int, points, c, color: str | None, head: str = "",
               tail: str = ""):
    """A ``<curve>``: ``c`` is a Curve (its style flags) or None for a curved arrow."""
    bits = 0
    attrs = {"id": str(cid),
             "CurvePoints": " ".join(f"{_f(x)} {_f(y)}" for x, y in points)}
    end, start = (c.arrow_end, c.arrow_start) if c is not None else (head, tail)
    if c is not None:
        for flag in ("closed", "dashed", "bold", "filled", "shaded", "doubled"):
            if getattr(c, flag):
                bits |= _CT[flag]
    if end:
        bits |= _CT["arrow_end" if end == "full" else "half_end"]
        attrs["ArrowheadHead"] = "Full" if end == "full" else "HalfLeft"
        attrs["ArrowheadType"] = "Solid"
    if start:
        bits |= _CT["arrow_start" if start == "full" else "half_start"]
        attrs["ArrowheadTail"] = "Full" if start == "full" else "HalfLeft"
        attrs["ArrowheadType"] = "Solid"
    if bits:
        attrs["CurveType"] = str(bits)
    if color:
        attrs["color"] = color
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    attrs["BoundingBox"] = f"{_f(min(xs))} {_f(min(ys))} {_f(max(xs))} {_f(max(ys))}"
    return ET.SubElement(page, "curve", attrs)


def save_cdxml(doc: Document, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(to_cdxml(doc))


# --- import -------------------------------------------------------------------------------

def _xy(text: str | None) -> tuple[float, float] | None:
    if not text:
        return None
    parts = text.split()
    try:
        return float(parts[0]), float(parts[1])
    except (IndexError, ValueError):
        return None


def _text_of(t) -> tuple[str, float, int]:
    parts, size, face = [], 0.0, 0
    for s in t.iter("s"):
        parts.append(s.text or "")
        try:
            size = size or float(s.get("size", 0))
        except ValueError:
            pass
        try:
            face |= int(s.get("face", "0"))
        except ValueError:
            pass
    return "".join(parts), size, face


def from_cdxml(text: str) -> Document:
    from rsstatchem.core.chem_draw import convert

    try:
        root = ET.fromstring(text.encode("utf-8") if isinstance(text, str) else text)
    except ET.ParseError as exc:
        raise ValueError(f"Not a CDXML file ({exc})") from None
    if root.tag != "CDXML":
        raise ValueError("Not a CDXML file")
    doc = Document(preset("ACS Document 1996"))
    palette = ["#000000", "#FFFFFF"]
    for c in root.iter("color"):
        try:
            palette.append("#%02X%02X%02X" % tuple(round(float(c.get(k, 0)) * 255)
                                                    for k in "rgb"))
        except ValueError:
            palette.append("")

    def color_of(el) -> str:
        try:
            i = int(el.get("color", ""))
        except ValueError:
            return ""
        c = palette[i] if 0 <= i < len(palette) else ""
        return "" if c == "#000000" else c

    try:
        doc.style.caption_size = float(root.get("CaptionSize", doc.style.caption_size))
        doc.style.font_size = float(root.get("LabelSize", doc.style.font_size))
    except ValueError:
        pass
    node_map: dict[str, int] = {}
    bond_map: dict[str, int] = {}
    labels: list[tuple[int, str]] = []
    lengths: list[float] = []
    atom_extra: list[tuple[int, dict]] = []
    tagged: list = []           # (kind, object) restored from RS-StatChem tags
    root_tag = _tag_of(root) or {}
    for k, v in (root_tag.get("style") or {}).items():
        if k in _STYLE_EXTRA:
            try:
                setattr(doc.style, k, type(getattr(doc.style, k))(v))
            except (TypeError, ValueError):
                pass
    if root.get("ShowAtomStereo") == "yes":
        doc.style.show_stereo = True
    try:
        doc.style.chain_angle = float(root.get("ChainAngle", doc.style.chain_angle))
    except ValueError:
        pass

    def read_fragment(frag) -> None:
        for n in frag.findall("n"):
            p = _xy(n.get("p"))
            if p is None:
                continue
            ntype = n.get("NodeType", "Element")
            label = ""
            t = n.find("t")
            if t is not None:
                label = _text_of(t)[0].strip()
            sym = "C"
            if n.get("Element"):
                try:
                    sym = _symbol(int(n.get("Element")))
                except (ValueError, RuntimeError):
                    sym = "C"
            a = doc.add_atom(p[0], p[1], sym)
            a.color = color_of(n)
            for attr, field, conv in (("Charge", "charge", int), ("Isotope", "isotope", int),
                                      ("NumHydrogens", "explicit_h", int)):
                if n.get(attr):
                    try:
                        setattr(a, field, conv(n.get(attr)))
                    except ValueError:
                        pass
            if n.get("Radical") in _RADICAL_IN:
                a.radicals = _RADICAL_IN[n.get("Radical")]
            if ntype in ("Nickname", "Fragment", "GenericNickname", "Unspecified",
                         "Generic", "Unknown"):
                label = n.get("GenericNickname") or label
                if label:
                    labels.append((a.id, label))
            elif sym == "C" and label == "C":
                a.show_carbon = True
            node_map[n.get("id", "")] = a.id
            extra = (_tag_of(n) or {}).get("atom")
            if extra:
                atom_extra.append((a.id, extra))
        for b in frag.findall("b"):
            ia, ib = node_map.get(b.get("B", "")), node_map.get(b.get("E", ""))
            if ia is None or ib is None or ia == ib:
                continue
            dative = b.get("Order", "") == "dative"
            try:
                order = float(b.get("Order", "1"))
            except ValueError:
                order = 1.0
            order = {1.0: 1, 2.0: 2, 3.0: 3}.get(order, 1.5 if order == 1.5 else 1)
            style, rev = _DISPLAY_IN.get(b.get("Display", ""), ("", False))
            if dative:
                style = "dative"
            if rev:
                ia, ib = ib, ia
            bond = doc.add_bond(ia, ib, order, style)
            bond.color = color_of(b)
            bond_map[b.get("id", "")] = bond.id
            pos = b.get("DoublePosition", "")
            if pos.lower() in ("center", "left", "right"):
                bond.double_pos = pos.lower()
            pa, pb = doc.atoms[ia], doc.atoms[ib]
            lengths.append(math.hypot(pa.x - pb.x, pa.y - pb.y))

    def read_curve(el) -> None:
        try:
            vals = [float(v) for v in el.get("CurvePoints", "").split()]
        except ValueError:
            return
        pts = [[vals[i], vals[i + 1]] for i in range(0, len(vals) - 1, 2)]
        if len(pts) < 4:
            return
        while len(pts) % 3:                         # tolerate a short or long point list
            pts.append(list(pts[-1]))
        try:
            bits = int(el.get("CurveType", "0"))
        except ValueError:
            bits = 0
        kw = {flag: bool(bits & _CT[flag]) for flag in
              ("closed", "dashed", "bold", "filled", "shaded", "doubled")}
        head, tail = el.get("ArrowheadHead", ""), el.get("ArrowheadTail", "")
        end = "full" if head == "Full" or bits & _CT["arrow_end"] else (
            "half" if head.startswith("Half") or bits & _CT["half_end"] else "")
        start = "full" if tail == "Full" or bits & _CT["arrow_start"] else (
            "half" if tail.startswith("Half") or bits & _CT["half_start"] else "")
        c = doc.add_curve(pts, arrow_end=end, arrow_start=start, **kw)
        c.color = color_of(el)

    def read_orbital(el) -> None:
        c, m = _xy(el.get("Center3D")), _xy(el.get("MajorAxisEnd3D"))
        if c is None:
            box = el.get("BoundingBox", "").split()
            if len(box) != 4:
                return
            try:
                x1, y1, x2, y2 = (float(v) for v in box)
            except ValueError:
                return
            c = ((x1 + x2) / 2, (y1 + y2) / 2)
            m = (c[0], min(y1, y2))
        if m is None:
            m = (c[0], c[1] - doc.style.bond_length)
        otype = el.get("OrbitalType", "p")
        fill = "plain"
        base = otype
        for suffix, f in (("Shaded", "shaded"), ("Filled", "filled")):
            if otype.endswith(suffix):
                base, fill = otype[:-len(suffix)], f
        kind = _ORB_IN.get(base, "p")
        if fill == "filled" and kind not in ("s", "lobe"):
            fill = "phase"                          # ChemDraw's filled p / d / hybrid
        o = doc.add_orbital(c[0], c[1], kind, fill=fill,
                            size=math.hypot(m[0] - c[0], m[1] - c[1]),
                            angle=math.atan2(m[1] - c[1], m[0] - c[0]))
        o.color = color_of(el)

    def restore(data: dict) -> bool:
        """Re-create an object from its RS-StatChem tag (ids are remapped later)."""
        for kind, cls in (("arrow", Arrow), ("shape", Shape), ("symbol", Symbol),
                          ("tlc", TLCPlate), ("gel", GelPlate), ("clipart", Clipart),
                          ("bracket", Bracket)):
            if kind in data and isinstance(data[kind], dict):
                d = dict(data[kind])
                orig = d.get("id")
                d["id"] = doc.new_id()
                d["group"] = 0
                o = _from_dict(cls, d)
                doc.store(kind)[o.id] = o
                tagged.append((kind, o, orig))
                return True
        return False

    def walk(el) -> None:
        for child in el:
            tag = child.tag
            data = _tag_of(child) if tag != "fragment" else None
            if data and restore(data):
                continue
            if tag == "fragment":
                read_fragment(child)
            elif tag in ("group", "page"):
                walk(child)
            elif tag == "t":
                p = _xy(child.get("p"))
                txt, size, face = _text_of(child)
                if p is None or not txt.strip():
                    continue
                size = size or doc.style.caption_size
                if txt.strip() == "+":
                    pl = doc.add_plus(p[0] + size * 0.3, p[1] - size * 0.35)
                    pl.color = color_of(child)
                    continue
                tb = doc.add_text(p[0], p[1] - size * 0.95, txt.replace("\r", "\n"),
                                  chemical=(face & _FACE_FORMULA) == _FACE_FORMULA,
                                  size=0.0 if abs(size - doc.style.caption_size) < 0.01
                                  else size)
                tb.bold, tb.italic, tb.underline = bool(face & 1), bool(face & 2), \
                    bool(face & 4)
                tb.color = color_of(child)
                extra = (data or {}).get("text") or {}
                tb.autonumber = int(extra.get("autonumber", 0) or 0)
                tb.font = str(extra.get("font", "") or "")
            elif tag == "arrow":
                head, tail = _xy(child.get("Head3D")), _xy(child.get("Tail3D"))
                if head is None or tail is None:
                    continue
                kind = "forward"
                if child.get("ArrowheadType") == "Hollow":
                    kind = "retro"
                elif child.get("ArrowheadHead", "").startswith("Half") and \
                        child.get("ArrowheadTail", "").startswith("Half"):
                    kind = "equilibrium"
                elif child.get("ArrowheadTail") == "Full":
                    kind = "resonance"
                elif not child.get("ArrowheadHead"):
                    kind = "line"
                ar = doc.add_arrow(tail[0], tail[1], head[0], head[1], kind)
                ar.color = color_of(child)
            elif tag == "curve":
                read_curve(child)
                if (data or {}).get("curve", {}).get("faded") and doc.curves:
                    doc.curves[max(doc.curves)].faded = True
            elif tag == "tlcplate":
                read_tlc(child)
            elif tag == "graphic":
                if child.get("SupersededBy"):
                    continue
                if child.get("GraphicType") == "Orbital":
                    read_orbital(child)
                    continue
                box = child.get("BoundingBox", "").split()
                if len(box) != 4:
                    continue
                try:
                    x1, y1, x2, y2 = (float(v) for v in box)
                except ValueError:
                    continue
                gtype = child.get("GraphicType")
                if gtype == "Bracket":
                    kind = {"RoundPair": "round", "CurlyPair": "curly"}.get(
                        child.get("BracketType", ""), "square")
                    doc.add_bracket(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), kind)
                elif gtype == "Line" and child.get("ArrowType"):
                    at = child.get("ArrowType", "")
                    kind = {"Equilibrium": "equilibrium", "Resonance": "resonance",
                            "RetroSynthetic": "retro"}.get(at, "forward")
                    doc.add_arrow(x1, y1, x2, y2, kind)
                elif gtype == "Symbol" and child.get("SymbolType") == "Plus":
                    doc.add_plus((x1 + x2) / 2, (y1 + y2) / 2)
                elif gtype == "Symbol" and child.get("SymbolType") in _SYMBOL_IN:
                    doc.add_symbol((x1 + x2) / 2, (y1 + y2) / 2,
                                   _SYMBOL_IN[child.get("SymbolType")])
                elif gtype in ("Oval", "Rectangle"):
                    flags = (child.get("OvalType", "") + " " +
                             child.get("RectangleType", "")).split()
                    if gtype == "Oval":
                        kind = "circle" if "Circle" in flags else "ellipse"
                    else:
                        kind = "roundrect" if "RoundEdge" in flags else (
                            "shadowrect" if "Shadow" in flags else "rect")
                    doc.add_shape(x1, y1, x2, y2, kind, shaded="Shaded" in flags,
                                  filled="Filled" in flags, dashed="Dashed" in flags,
                                  bold="Bold" in flags)

    def read_tlc(el) -> None:
        box = el.get("BoundingBox", "").split()
        tl, br = _xy(el.get("TopLeft")), _xy(el.get("BottomRight"))
        if tl is None or br is None:
            if len(box) != 4:
                return
            try:
                tl, br = (float(box[0]), float(box[1])), (float(box[2]), float(box[3]))
            except ValueError:
                return
        lanes = []
        for ln in el.findall("tlclane"):
            spots = []
            for sp in ln.findall("tlcspot"):
                try:
                    d = {"rf": float(sp.get("Rf", "0.5"))}
                    for attr, key in (("Width", "w"), ("Height", "h"), ("Tail", "tail")):
                        if float(sp.get(attr, "0") or 0):
                            d[key] = float(sp.get(attr))
                except ValueError:
                    continue
                if sp.get("ShowRf") == "yes":
                    d["show_rf"] = True
                spots.append(d)
            lanes.append(spots)
        try:
            origin = float(el.get("OriginFraction", "0.12"))
            front = 1 - float(el.get("SolventFrontFraction", "0.1"))
        except ValueError:
            origin, front = 0.12, 0.9
        t = doc.add_tlc(tl[0], tl[1], br[0], br[1], lanes=len(lanes) or 1, origin=origin,
                        front=front,
                        show_origin=el.get("ShowOrigin", "yes") != "no",
                        show_front=el.get("ShowSolventFront", "yes") != "no",
                        show_border=el.get("ShowBorders", "yes") != "no",
                        show_ticks=el.get("ShowSideTicks", "yes") != "no",
                        transparent=el.get("Transparent") == "yes")
        if lanes:
            t.lanes = lanes

    walk(root)
    for aid, extra in atom_extra:
        a = doc.atoms[aid]
        a.charge_circle = bool(extra.get("charge_circle", False))
        a.attachment = int(extra.get("attachment", 0) or 0)
        a.show_stereo = bool(extra.get("show_stereo", True))
        if extra.get("label"):
            labels = [(i, lab) for i, lab in labels if i != aid] + [(aid, extra["label"])]
    maps = {"atom": node_map, "bond": bond_map}
    maps["symbol"] = {str(orig): o.id for kind, o, orig in tagged if kind == "symbol"}
    for kind, o, _orig in tagged:
        if kind == "symbol":
            o.atom = node_map.get(str(o.atom), 0) if o.atom else 0
        elif kind == "arrow":
            for end in ("tail", "head"):
                ref = getattr(o, end + "_ref")
                k, _, rid = ref.partition(":")
                new = maps.get(k, {}).get(rid) if ref else None
                setattr(o, end + "_ref", f"{k}:{new}" if new else "")
    for aid, label in labels:
        try:
            convert.apply_label_text(doc, aid, label)
        except Exception:  # noqa: BLE001 - keep the label as typed
            doc.atoms[aid].label = label
    if lengths:
        doc.style.bond_length = statistics.median(lengths)
    return doc


def load_cdxml(path: str) -> Document:
    with open(path, "rb") as fh:
        data = fh.read()
    return from_cdxml(data.decode("utf-8", errors="replace"))

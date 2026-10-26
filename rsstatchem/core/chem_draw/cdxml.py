"""ChemDraw CDXML (XML) import and export for drawing documents (no Qt).

Supported subset: fragments with nodes (element, charge, isotope, explicit H, radicals,
labels / nicknames) and bonds (order 1 / 1.5 / 2 / 3, wedge, hashed wedge, bold, dashed,
wavy, double-bond position), captions (``<t>``), reaction arrows (``<arrow>``, and legacy
``<graphic>`` lines with an arrow type), "+" signs and bracket pairs. CDXML and our
documents both use points with y pointing down, so coordinates are copied unchanged.

Nicknames are written as labelled generic nodes (their text survives, their expansion is
recomputed by RS-StatChem on import); curved arrows are written straight.
"""
from __future__ import annotations

import math
import statistics
import xml.etree.ElementTree as ET

from rsstatchem.core.chem_draw.model import Document
from rsstatchem.core.chem_draw.style import preset

__all__ = ["to_cdxml", "from_cdxml", "save_cdxml", "load_cdxml"]

_DISPLAY_OUT = {"wedge": "WedgeBegin", "hash": "WedgedHashBegin", "bold": "Bold",
                "dashed": "Dash", "wavy": "Wavy"}
_DISPLAY_IN = {"WedgeBegin": ("wedge", False), "WedgeEnd": ("wedge", True),
               "WedgedHashBegin": ("hash", False), "WedgedHashEnd": ("hash", True),
               "Hash": ("dashed", False), "Dash": ("dashed", False), "Bold": ("bold", False),
               "Wavy": ("wavy", False)}
_DOUBLE_OUT = {"center": "Center", "left": "Left", "right": "Right"}
_RADICAL_OUT = {1: "Doublet", 2: "Singlet"}
_RADICAL_IN = {"Doublet": 1, "Singlet": 2, "Triplet": 2}
_BRACKET_OUT = {"square": "SquarePair", "round": "RoundPair", "curly": "CurlyPair"}
_FACE_FORMULA = 96
_FONT_ID = "3"


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
        "CaptionFont": _FONT_ID, "CaptionSize": _f(st.caption_size)})
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
        caption(t.x, t.y + size * 0.95, t.text, size, _FACE_FORMULA if t.chemical else 0,
                t.color, t.bold, t.italic, t.underline)
    for p in doc.pluses.values():
        size = st.caption_size * 1.2
        caption(p.x - size * 0.3, p.y + size * 0.35, "+", size, color=p.color)
    for ar in doc.arrows.values():
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
        ci = color_index(ar.color)
        if ci:
            attrs["color"] = ci
        ET.SubElement(page, "arrow", attrs)
        mx, my = (ar.x1 + ar.x2) / 2, (ar.y1 + ar.y2) / 2
        for text, dy in ((ar.above, -st.caption_size * 0.6), (ar.below, st.caption_size * 1.6)):
            if text:
                w = len(text) * st.caption_size * 0.25
                caption(mx - w, my + dy, text, st.caption_size, _FACE_FORMULA)
    for br in doc.brackets.values():
        ET.SubElement(page, "graphic", {
            "id": str(next(ids)), "GraphicType": "Bracket",
            "BracketType": _BRACKET_OUT.get(br.kind, "SquarePair"),
            "BoundingBox": f"{_f(br.x1)} {_f(br.y1)} {_f(br.x2)} {_f(br.y2)}"})
    for c in colors:
        r, g, b = (int(c[i:i + 2], 16) / 255 for i in (1, 3, 5))
        ET.SubElement(ctable, "color", {"r": _f(r), "g": _f(g), "b": _f(b)})
    ET.indent(root, " ")
    body = ET.tostring(root, encoding="unicode")
    return ('<?xml version="1.0" encoding="UTF-8" ?>\n'
            '<!DOCTYPE CDXML SYSTEM "http://www.cambridgesoft.com/xml/cdxml.dtd" >\n'
            + body + "\n")


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
    labels: list[tuple[int, str]] = []
    lengths: list[float] = []

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
        for b in frag.findall("b"):
            ia, ib = node_map.get(b.get("B", "")), node_map.get(b.get("E", ""))
            if ia is None or ib is None or ia == ib:
                continue
            try:
                order = float(b.get("Order", "1"))
            except ValueError:
                order = 1.0
            order = {1.0: 1, 2.0: 2, 3.0: 3}.get(order, 1.5 if order == 1.5 else 1)
            style, rev = _DISPLAY_IN.get(b.get("Display", ""), ("", False))
            if rev:
                ia, ib = ib, ia
            bond = doc.add_bond(ia, ib, order, style)
            bond.color = color_of(b)
            pos = b.get("DoublePosition", "")
            if pos.lower() in ("center", "left", "right"):
                bond.double_pos = pos.lower()
            pa, pb = doc.atoms[ia], doc.atoms[ib]
            lengths.append(math.hypot(pa.x - pb.x, pa.y - pb.y))

    def walk(el) -> None:
        for child in el:
            tag = child.tag
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
            elif tag == "graphic":
                if child.get("SupersededBy"):
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

    walk(root)
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

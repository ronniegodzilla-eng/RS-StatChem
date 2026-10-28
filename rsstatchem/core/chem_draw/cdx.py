"""ChemDraw binary CDX import and export for drawing documents (pure Python, no Qt).

CDX is the binary twin of CDXML: the same object tree (document, page, fragment, node, bond,
text, graphic, curve, arrow ...) and the same properties, stored as little-endian tagged
records. This module converts between the two representations, so a CDX file is read and
written through :mod:`rsstatchem.core.chem_draw.cdxml` and keeps everything that module
understands (structures, captions, arrows, "+" signs, brackets, curves, orbitals, colours).

- :func:`cdx_to_cdxml` decodes CDX into CDXML text; :func:`cdxml_to_cdx` encodes CDXML text.
- :func:`from_cdx` / :func:`load_cdx` give a :class:`~rsstatchem.core.chem_draw.model.Document`;
  if the document reader finds no structure in a file that has one, the chemistry is taken from
  RDKit's ChemDraw reader (``rdChemDraw``), which is built on Revvity's open-sourced ChemDraw
  format code.
- :func:`to_cdx` / :func:`save_cdx` write a document. A drawn reaction also gets a CDX
  reaction scheme (reactants, products, arrow, plus signs, conditions) so ChemDraw, RDKit and
  databases read it as a reaction.

The tag numbers, record layouts (Point2D is stored y then x, Point3D x/y/z, Rectangle
top/left/bottom/right, 10-byte style runs) and enumerations follow the public CDX format
specification and Revvity's BSD-3 "Open Source ChemDraw File Format" code
(github.com/Glysade/chemdraw: ``CDXConstants.h``, ``CDXIO.h``). Only properties with a known
layout are converted; anything else is skipped on reading and never written.
"""
from __future__ import annotations

import struct
import xml.etree.ElementTree as ET

__all__ = ["cdx_to_cdxml", "cdxml_to_cdx", "from_cdx", "load_cdx", "to_cdx", "save_cdx",
           "is_cdx", "rdkit_cdx_support", "MAGIC"]

MAGIC = b"VjCD0100"
_HEADER = MAGIC + struct.pack("<I", 0x01020304) + bytes(16)     # 28 bytes, as ChemDraw writes
_COORD = 65536.0                                                # CDXCoordinate: 1/65536 pt

OBJECTS = {
    0x8000: "CDXML", 0x8001: "page", 0x8002: "group", 0x8003: "fragment", 0x8004: "n",
    0x8005: "b", 0x8006: "t", 0x8007: "graphic", 0x8008: "curve",
    0x8009: "embeddedobject", 0x800A: "altgroup", 0x800B: "templategrid", 0x800C: "regnum",
    0x800D: "scheme", 0x800E: "step", 0x800F: "objectdefinition", 0x8010: "spectrum",
    0x8011: "objecttag", 0x8013: "sequence", 0x8014: "crossreference", 0x8015: "splitter",
    0x8016: "table", 0x8017: "bracketedgroup", 0x8018: "bracketattachment",
    0x8019: "crossingbond", 0x801A: "border", 0x801B: "geometry", 0x801C: "constraint",
    0x801D: "tlcplate", 0x801E: "tlclane", 0x801F: "tlcspot", 0x8020: "chemicalproperty",
    0x8021: "arrow", 0x802B: "annotation",
}
_OBJECT_TAGS = {v: k for k, v in OBJECTS.items()}

_NODE_TYPES = ["Unspecified", "Element", "ElementList", "ElementListNickname", "Nickname",
               "Fragment", "Formula", "GenericNickname", "AnonymousAlternativeGroup",
               "NamedAlternativeGroup", "MultiAttachment", "VariableAttachment",
               "ExternalConnectionPoint", "LinkNode", "Monomer"]
_RADICALS = ["None", "Singlet", "Doublet", "Triplet"]
_DISPLAY = ["Solid", "Dash", "Hash", "WedgedHashBegin", "WedgedHashEnd", "Bold", "WedgeBegin",
            "WedgeEnd", "Wavy", "HollowWedgeBegin", "HollowWedgeEnd", "WavyWedgeBegin",
            "WavyWedgeEnd", "Dot", "DashDot"]
_GRAPHIC_TYPES = ["Undefined", "Line", "Arc", "Rectangle", "Oval", "Orbital", "Bracket",
                  "Symbol"]
_BRACKETS = ["RoundPair", "SquarePair", "CurlyPair", "Square", "Curly", "Round"]
_SYMBOLS = ["LonePair", "Electron", "RadicalCation", "RadicalAnion", "CirclePlus",
            "CircleMinus", "Dagger", "DoubleDagger", "Plus", "Minus", "Racemic", "Absolute",
            "Relative", "LonePairBar"]
_HEAD_TYPES = ["Unspecified", "Solid", "Hollow", "Angle"]
_HEAD_POS = ["Unspecified", "None", "Full", "HalfLeft", "HalfRight"]
_ORDERS = {0x0001: "1", 0x0002: "2", 0x0004: "3", 0x0008: "4", 0x0010: "5", 0x0020: "6",
           0x0040: "0.5", 0x0080: "1.5", 0x0100: "2.5", 0x0200: "3.5", 0x0400: "4.5",
           0x0800: "5.5", 0x1000: "dative", 0x2000: "ionic", 0x4000: "hydrogen",
           0x8000: "threecenter"}
_ORDER_BITS = {v: k for k, v in _ORDERS.items()}
_DOUBLE_POS = ["Center", "Right", "Left"]
_ORBITALS = {0: "s", 1: "oval", 2: "lobe", 3: "p", 4: "hybridPlus", 5: "hybridMinus",
             6: "dz2Plus", 7: "dz2Minus", 8: "dxy"}
_FILL = {0x0001: "None", 0x0002: "Solid", 0x0004: "Shaded", 0x0008: "Gradient",
         0x0010: "Pattern"}
_ARROW_TYPES = {0: "NoHead", 1: "HalfHead", 2: "FullHead", 4: "Resonance", 8: "Equilibrium",
                16: "Hollow", 32: "RetroSynthetic", 64: "NoGo", 128: "Dipole"}

# tag -> (CDXML attribute, codec, codec argument)
PROPS: dict[int, tuple] = {
    0x0003: ("CreationProgram", "str", None),
    0x0006: ("ModificationProgram", "str", None),
    0x0008: ("Name", "str", None),
    0x0009: ("Comment", "str", None),
    0x000A: ("Z", "i16", None),
    0x0012: ("SupersededBy", "u32", None),
    0x0200: ("p", "pt2", None),
    0x0201: ("xyz", "pt3", None),
    0x0202: ("extent", "pt2", None),
    0x0204: ("BoundingBox", "rect", None),
    0x0207: ("Head3D", "pt3", None),
    0x0208: ("Tail3D", "pt3", None),
    0x0209: ("TopLeft", "pt2", None),
    0x020A: ("TopRight", "pt2", None),
    0x020B: ("BottomRight", "pt2", None),
    0x020C: ("BottomLeft", "pt2", None),
    0x020D: ("Center3D", "pt3", None),
    0x020E: ("MajorAxisEnd3D", "pt3", None),
    0x020F: ("MinorAxisEnd3D", "pt3", None),
    0x0301: ("color", "u16", None),
    0x0302: ("bgcolor", "u16", None),
    0x0400: ("NodeType", "enum16", _NODE_TYPES),
    0x0402: ("Element", "i16", None),
    0x0420: ("Isotope", "i16", None),
    0x0421: ("Charge", "charge", None),
    0x0422: ("Radical", "enum8", _RADICALS),
    0x042B: ("NumHydrogens", "u16", None),
    0x0433: ("GenericNickname", "str", None),
    0x0439: ("AtomNumber", "str", None),
    0x0446: ("EnhancedStereoType", "enum8", ["Unspecified", "None", "Absolute", "Or",
                                             "And"]),
    0x0447: ("EnhancedStereoGroupNum", "i16", None),
    0x0600: ("Order", "order", None),
    0x0601: ("Display", "enum16", _DISPLAY),
    0x0602: ("Display2", "enum16", _DISPLAY),
    0x0603: ("DoublePosition", "dpos", None),
    0x0604: ("B", "u32", None),
    0x0605: ("E", "u32", None),
    0x0803: ("ChainAngle", "angle", None),
    0x0804: ("BondSpacing", "tenth", None),
    0x0805: ("BondLength", "coord", None),
    0x0806: ("BoldWidth", "coord", None),
    0x0807: ("LineWidth", "coord", None),
    0x0808: ("MarginWidth", "coord", None),
    0x0809: ("HashSpacing", "coord", None),
    0x080F: ("WidthPages", "i16", None),
    0x0810: ("HeightPages", "i16", None),
    0x081A: ("LabelFont", "u16", None),
    0x081B: ("CaptionFont", "u16", None),
    0x081C: ("LabelSize", "size", None),
    0x081D: ("CaptionSize", "size", None),
    0x081E: ("LabelFace", "i16", None),
    0x081F: ("CaptionFace", "i16", None),
    0x0A00: ("GraphicType", "enum16", _GRAPHIC_TYPES),
    0x0A02: ("ArrowType", "arrowtype", None),
    0x0A05: ("OrbitalType", "orbital", None),
    0x0A06: ("BracketType", "enum16", _BRACKETS),
    0x0A07: ("SymbolType", "enum16", _SYMBOLS),
    0x0A08: ("CurveType", "i16", None),
    0x0A20: ("HeadSize", "i16", None),
    0x0A23: ("CurvePoints", "curve", None),
    0x0A2F: ("ArrowheadType", "enum16", _HEAD_TYPES),
    0x0A30: ("ArrowheadCenterSize", "i16", None),
    0x0A31: ("ArrowheadWidth", "i16", None),
    0x0A33: ("ArrowShaftSpacing", "i16", None),
    0x0A35: ("ArrowheadHead", "enum16", _HEAD_POS),
    0x0A36: ("ArrowheadTail", "enum16", _HEAD_POS),
    0x0A37: ("FillType", "fill", None),
    0x0C01: ("ReactionStepReactants", "ids", None),
    0x0C02: ("ReactionStepProducts", "ids", None),
    0x0C03: ("ReactionStepPlusses", "ids", None),
    0x0C04: ("ReactionStepArrows", "ids", None),
    0x0C05: ("ReactionStepObjectsAboveArrow", "ids", None),
    0x0C06: ("ReactionStepObjectsBelowArrow", "ids", None),
}
_ATTR_TAGS = {name: tag for tag, (name, _c, _a) in PROPS.items()}
_FONT_TABLE, _COLOR_TABLE, _TEXT = 0x0100, 0x0300, 0x0700
_CHARSETS = {"iso-8859-1": 1252, "windows-1252": 1252, "utf-8": 65001,
             "x-mac-roman": 10000, "macintosh": 10000}
_CHARSET_NAMES = {1252: "iso-8859-1", 65001: "utf-8", 10000: "x-mac-roman"}


def is_cdx(data: bytes) -> bool:
    return data[:8] == MAGIC


def rdkit_cdx_support() -> bool:
    """True when the RDKit build can read CDX (``Chem.HasChemDrawCDXSupport``)."""
    try:
        from rdkit import Chem
        return bool(Chem.HasChemDrawCDXSupport())
    except Exception:  # noqa: BLE001 - old RDKit builds
        return False


def _f(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".") or "0"


def _codec(charset: int) -> str:
    return {65001: "utf-8", 10000: "mac_roman"}.get(charset, "cp1252")


# --- decoding -----------------------------------------------------------------------------

class _Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.fonts: dict[int, int] = {}          # font id -> charset

    def u16(self, pos: int) -> int:
        return struct.unpack_from("<H", self.data, pos)[0]

    def parse(self) -> ET.Element:
        if not is_cdx(self.data):
            raise ValueError("Not a ChemDraw CDX file")
        pos = 28
        root = ET.Element("CDXML")
        if pos + 2 <= len(self.data) and self.u16(pos) == 0x8000:      # explicit document
            pos += 6
        self._object(root, pos)
        return root

    def _object(self, el: ET.Element, pos: int) -> int:
        data = self.data
        while pos + 2 <= len(data):
            tag = self.u16(pos)
            pos += 2
            if tag == 0:
                return pos
            if tag & 0x8000:
                oid = struct.unpack_from("<I", data, pos)[0]
                pos += 4
                child = ET.SubElement(el, OBJECTS.get(tag, "unknown"), {"id": str(oid)})
                pos = self._object(child, pos)
                if child.tag == "unknown":
                    el.remove(child)
                continue
            n = self.u16(pos)
            pos += 2
            if n == 0xFFFF:
                n = struct.unpack_from("<I", data, pos)[0]
                pos += 4
            value = data[pos:pos + n]
            pos += n
            try:
                self._property(el, tag, value)
            except (struct.error, IndexError, ValueError, UnicodeDecodeError):
                continue                # a damaged property never stops the file
        return pos

    def _string(self, value: bytes):
        nruns = struct.unpack_from("<H", value, 0)[0]
        runs = [struct.unpack_from("<5H", value, 2 + 10 * i) for i in range(nruns)]
        raw = value[2 + 10 * nruns:]
        return runs, raw

    def _decode(self, raw: bytes, font: int) -> str:
        return raw.decode(_codec(self.fonts.get(font, 1252)), errors="replace")

    def _property(self, el: ET.Element, tag: int, value: bytes) -> None:
        if tag == _FONT_TABLE:
            count = struct.unpack_from("<H", value, 2)[0]
            table = ET.SubElement(el, "fonttable")
            pos = 4
            for _ in range(count):
                fid, charset, ln = struct.unpack_from("<3H", value, pos)
                pos += 6
                name = value[pos:pos + ln].decode("latin-1")
                pos += ln
                self.fonts[fid] = charset
                ET.SubElement(table, "font", {"id": str(fid), "charset": _CHARSET_NAMES.get(
                    charset, "iso-8859-1"), "name": name})
            return
        if tag == _COLOR_TABLE:
            count = struct.unpack_from("<H", value, 0)[0]
            table = ET.SubElement(el, "colortable")
            for i in range(count):
                r, g, b = struct.unpack_from("<3H", value, 2 + 6 * i)
                ET.SubElement(table, "color", {"r": _f(r / 65535), "g": _f(g / 65535),
                                               "b": _f(b / 65535)})
            return
        if tag == _TEXT:
            runs, raw = self._string(value)
            if not runs:
                runs = [(0, 0, 0, 0, 0)]
            for k, (start, font, face, size, color) in enumerate(runs):
                end = runs[k + 1][0] if k + 1 < len(runs) else len(raw)
                attrs = {"font": str(font), "face": str(face)}
                if size:
                    attrs["size"] = _f(size / 20.0)
                if color:
                    attrs["color"] = str(color)
                s = ET.SubElement(el, "s", attrs)
                s.text = self._decode(raw[start:end], font)
            return
        spec = PROPS.get(tag)
        if spec is None:
            return
        name, codec, arg = spec
        text = self._value(codec, arg, value)
        if text is not None:
            el.set(name, text)

    def _value(self, codec: str, arg, v: bytes) -> str | None:
        if codec == "str":
            runs, raw = self._string(v)
            return self._decode(raw, runs[0][1] if runs else 0)
        if codec == "i16":
            return str(struct.unpack_from("<h", v)[0] if len(v) >= 2 else
                       struct.unpack_from("<b", v)[0])
        if codec == "u16":
            return str(struct.unpack_from("<H", v)[0] if len(v) >= 2 else v[0])
        if codec == "u32":
            return str(struct.unpack_from("<I", v)[0])
        if codec == "charge":
            if len(v) >= 4:
                q = struct.unpack_from("<i", v)[0]
                return str(q // 0x1000000 if abs(q) >= 0x1000000 else q)
            return str(struct.unpack_from("<b", v)[0])
        if codec in ("enum8", "enum16"):
            i = v[0] if codec == "enum8" or len(v) == 1 else struct.unpack_from("<h", v)[0]
            return arg[i] if 0 <= i < len(arg) else None
        if codec == "pt2":
            y, x = struct.unpack_from("<2i", v)
            return f"{_f(x / _COORD)} {_f(y / _COORD)}"
        if codec == "pt3":
            x, y, z = struct.unpack_from("<3i", v)
            return f"{_f(x / _COORD)} {_f(y / _COORD)} {_f(z / _COORD)}"
        if codec == "rect":
            top, left, bottom, right = struct.unpack_from("<4i", v)
            return " ".join(_f(c / _COORD) for c in (left, top, right, bottom))
        if codec == "coord":
            return _f(struct.unpack_from("<i", v)[0] / _COORD)
        if codec == "angle":
            return _f(struct.unpack_from("<i", v)[0] / _COORD)
        if codec == "tenth":
            return _f(struct.unpack_from("<h", v)[0] / 10.0)
        if codec == "size":
            return _f(struct.unpack_from("<h", v)[0] / 20.0)
        if codec == "order":
            bits = struct.unpack_from("<H", v)[0]
            return _ORDERS.get(bits, " ".join(o for b, o in _ORDERS.items() if bits & b) or
                               "1")
        if codec == "dpos":
            i = struct.unpack_from("<H", v)[0] & 0xFF
            return _DOUBLE_POS[i] if i < len(_DOUBLE_POS) else None
        if codec == "orbital":
            i = struct.unpack_from("<H", v)[0]
            base = _ORBITALS.get(i & 0xFF)
            if base is None:
                return None
            return base + ("Shaded" if i & 0x0100 else "Filled" if i & 0x0200 else "")
        if codec == "fill":
            return _FILL.get(struct.unpack_from("<H", v)[0])
        if codec == "arrowtype":
            return _ARROW_TYPES.get(struct.unpack_from("<H", v)[0])
        if codec == "curve":
            n = struct.unpack_from("<H", v)[0]
            pts = [struct.unpack_from("<2i", v, 2 + 8 * i) for i in range(n)]
            return " ".join(f"{_f(x / _COORD)} {_f(y / _COORD)}" for y, x in pts)
        if codec == "ids":
            n = len(v) // 4
            return " ".join(str(i) for i in struct.unpack_from(f"<{n}I", v))
        return None


def cdx_to_cdxml(data: bytes) -> str:
    """CDXML text of a CDX file's bytes (raises ``ValueError`` for non-CDX data)."""
    root = _Reader(bytes(data)).parse()
    for name in ("colortable", "fonttable"):          # tables first, as in CDXML files
        el = root.find(name)
        if el is not None:
            root.remove(el)
            root.insert(0, el)
    ET.indent(root, " ")
    return ('<?xml version="1.0" encoding="UTF-8" ?>\n'
            + ET.tostring(root, encoding="unicode") + "\n")


# --- encoding -----------------------------------------------------------------------------

class _Writer:
    def __init__(self):
        self.out = bytearray(_HEADER)
        self.fonts: dict[int, int] = {}

    def prop(self, tag: int, value: bytes) -> None:
        if len(value) >= 0xFFFF:
            self.out += struct.pack("<HHI", tag, 0xFFFF, len(value))
        else:
            self.out += struct.pack("<HH", tag, len(value))
        self.out += value

    def encode_string(self, text: str, runs=None) -> bytes:
        font = runs[0][1] if runs else 0
        raw = text.encode(_codec(self.fonts.get(font, 1252)), errors="replace")
        head = struct.pack("<H", len(runs or []))
        for r in runs or []:
            head += struct.pack("<5H", *r)
        return head + raw

    def _assign_ids(self, root: ET.Element) -> None:
        """Object ids: nodes and bonds keep theirs (bonds refer to node ids); any other object
        whose id is missing or already taken gets a fresh one."""
        self.ids: dict[int, int] = {}
        used: set[int] = set()
        order = [el for el in root.iter() if el.tag in ("n", "b")] + \
            [el for el in root.iter() if el.tag not in ("n", "b")]
        pending = []
        for el in order:
            try:
                i = int(el.get("id", ""))
            except ValueError:
                i = 0
            if i > 0 and i not in used:
                used.add(i)
                self.ids[id(el)] = i
            else:
                pending.append(el)
        nxt = max(used | {0}) + 1
        for el in pending:
            self.ids[id(el)] = nxt
            nxt += 1

    def _id(self, el: ET.Element) -> int:
        return self.ids.get(id(el), 0)

    def document(self, root: ET.Element) -> bytes:
        self._assign_ids(root)
        fonts = root.find("fonttable")
        if fonts is not None:
            fdata = bytearray(struct.pack("<HH", 1, len(fonts)))      # Windows platform
            for f in fonts:
                fid = int(f.get("id", "0") or 0)
                cs = _CHARSETS.get(f.get("charset", "iso-8859-1").lower(), 1252)
                name = f.get("name", "Arial").encode("latin-1", errors="replace")
                self.fonts[fid] = cs
                fdata += struct.pack("<3H", fid, cs, len(name)) + name
            self.prop(_FONT_TABLE, bytes(fdata))
        colors = root.find("colortable")
        if colors is not None:
            cdata = bytearray(struct.pack("<H", len(colors)))
            for c in colors:
                cdata += struct.pack("<3H", *(max(0, min(65535, round(float(
                    c.get(k, "0") or 0) * 65535))) for k in "rgb"))
            self.prop(_COLOR_TABLE, bytes(cdata))
        self.attributes(root)
        for child in root:
            if child.tag not in ("fonttable", "colortable"):
                self.element(child)
        self.out += b"\0\0"
        return bytes(self.out)

    def element(self, el: ET.Element) -> None:
        tag = _OBJECT_TAGS.get(el.tag)
        if tag is None or tag == 0x8000:
            return                       # "s" runs are part of their text; unknown tags dropped
        self.out += struct.pack("<HI", tag, self._id(el))
        self.attributes(el)
        if el.tag == "t":
            self.text(el)
        for child in el:
            if child.tag != "s":
                self.element(child)
        self.out += b"\0\0"

    def text(self, t: ET.Element) -> None:
        runs, parts, start = [], [], 0
        for s in t.findall("s"):
            chunk = s.text or ""
            try:
                size = round(float(s.get("size", "10")) * 20)
            except ValueError:
                size = 200
            runs.append((start, int(s.get("font", "0") or 0), int(s.get("face", "0") or 0),
                         size, int(s.get("color", "0") or 0)))
            enc = chunk.encode(_codec(self.fonts.get(runs[-1][1], 1252)), errors="replace")
            parts.append(enc)
            start += len(enc)
        if not runs:
            return
        head = struct.pack("<H", len(runs)) + b"".join(struct.pack("<5H", *r) for r in runs)
        self.prop(_TEXT, head + b"".join(parts))

    def attributes(self, el: ET.Element) -> None:
        for name, text in el.attrib.items():
            tag = _ATTR_TAGS.get(name)
            if tag is None or text is None:
                continue
            _n, codec, arg = PROPS[tag]
            try:
                value = self._encode(codec, arg, text)
            except (ValueError, IndexError, struct.error):
                continue
            if value is not None:
                self.prop(tag, value)

    def _encode(self, codec: str, arg, text: str) -> bytes | None:
        def nums():
            return [float(v) for v in text.replace(",", " ").split()]

        def c(v):
            return round(v * _COORD)
        if codec == "str":
            return self.encode_string(text)
        if codec in ("i16",):
            return struct.pack("<h", int(float(text)))
        if codec == "u16":
            return struct.pack("<H", int(float(text)))
        if codec == "u32":
            return struct.pack("<I", int(float(text)))
        if codec == "charge":
            return struct.pack("<b", int(float(text)))
        if codec == "enum8":
            return struct.pack("<B", arg.index(text)) if text in arg else None
        if codec == "enum16":
            return struct.pack("<h", arg.index(text)) if text in arg else None
        if codec == "pt2":
            x, y = nums()[:2]
            return struct.pack("<2i", c(y), c(x))
        if codec == "pt3":
            v = nums() + [0.0]
            return struct.pack("<3i", c(v[0]), c(v[1]), c(v[2]))
        if codec == "rect":
            left, top, right, bottom = nums()[:4]
            return struct.pack("<4i", c(top), c(left), c(bottom), c(right))
        if codec in ("coord", "angle"):
            return struct.pack("<i", c(nums()[0]))
        if codec == "tenth":
            return struct.pack("<h", round(nums()[0] * 10))
        if codec == "size":
            return struct.pack("<h", round(nums()[0] * 20))
        if codec == "order":
            bits = 0
            for part in text.split():
                bits |= _ORDER_BITS.get(part, 0)
            return struct.pack("<H", bits or 1)
        if codec == "dpos":
            return struct.pack("<H", 0x0100 | _DOUBLE_POS.index(text)) \
                if text in _DOUBLE_POS else None
        if codec == "orbital":
            flags = 0
            base = text
            for suffix, bit in (("Shaded", 0x0100), ("Filled", 0x0200)):
                if text.endswith(suffix):
                    base, flags = text[:-len(suffix)], bit
            code = {v: k for k, v in _ORBITALS.items()}.get(base)
            return None if code is None else struct.pack("<H", code | flags)
        if codec == "fill":
            code = {v: k for k, v in _FILL.items()}.get(text)
            return None if code is None else struct.pack("<H", code)
        if codec == "arrowtype":
            code = {v: k for k, v in _ARROW_TYPES.items()}.get(text)
            return None if code is None else struct.pack("<H", code)
        if codec == "curve":
            v = nums()
            pts = [(v[i], v[i + 1]) for i in range(0, len(v) - 1, 2)]
            return struct.pack("<H", len(pts)) + b"".join(
                struct.pack("<2i", c(y), c(x)) for x, y in pts)
        if codec == "ids":
            ids = [int(i) for i in text.split()]
            return struct.pack(f"<{len(ids)}I", *ids)
        return None


def cdxml_to_cdx(text: str | bytes) -> bytes:
    """CDX bytes of CDXML text."""
    try:
        root = ET.fromstring(text.encode("utf-8") if isinstance(text, str) else text)
    except ET.ParseError as exc:
        raise ValueError(f"Not a CDXML document ({exc})") from None
    if root.tag != "CDXML":
        raise ValueError("Not a CDXML document")
    return _Writer().document(root)


# --- documents ----------------------------------------------------------------------------

def _add_scheme(root: ET.Element, doc) -> None:
    """Reaction scheme / step objects for a drawn single-step reaction."""
    from rsstatchem.core.chem_draw import convert
    if not doc.arrows:
        return
    try:
        split = convert.split_reaction(doc)
    except Exception:  # noqa: BLE001
        return
    if not split.ok:
        return
    page = root.find("page")
    if page is None:
        return
    frag_of = {}
    for frag in page.iter("fragment"):
        for n in frag.findall("n"):
            frag_of[n.get("id")] = frag.get("id")
    arrows = [el.get("id") for el in page if el.tag == "arrow"]
    pluses = [el.get("id") for el in page if el.tag == "t" and
              "".join(s.text or "" for s in el.iter("s")).strip() == "+"]

    def ids(comps):
        out = []
        for comp in comps:
            fid = frag_of.get(str(comp[0]))
            if fid and fid not in out:
                out.append(fid)
        return " ".join(out)
    reactants, products = (split.products, split.reactants) if split.retro else \
        (split.reactants, split.products)
    scheme = ET.SubElement(page, "scheme", {"id": "9000001"})
    attrs = {"id": "9000002", "ReactionStepReactants": ids(reactants),
             "ReactionStepProducts": ids(products)}
    if arrows:
        attrs["ReactionStepArrows"] = arrows[0]
    if pluses:
        attrs["ReactionStepPlusses"] = " ".join(pluses)
    if split.agents:
        attrs["ReactionStepObjectsAboveArrow"] = ids(split.agents)
    ET.SubElement(scheme, "step", {k: v for k, v in attrs.items() if v})


def to_cdx(doc) -> bytes:
    """CDX bytes of a drawing document (through its CDXML form)."""
    from rsstatchem.core.chem_draw import cdxml
    root = ET.fromstring(cdxml.to_cdxml(doc).encode("utf-8"))
    _add_scheme(root, doc)
    return _Writer().document(root)


def save_cdx(doc, path: str) -> None:
    with open(path, "wb") as fh:
        fh.write(to_cdx(doc))


def from_cdx(data: bytes, path: str | None = None):
    """Document of CDX bytes. Falls back to RDKit's ChemDraw reader for the chemistry when
    the object tree holds no structure that RDKit can see (``path`` is used for that)."""
    from rsstatchem.core.chem_draw import cdxml
    try:
        doc = cdxml.from_cdxml(cdx_to_cdxml(data))
    except (ValueError, struct.error, IndexError) as exc:
        doc, err = None, exc
    else:
        err = None
    if doc is not None and doc.atoms:
        return doc
    mols = _rdkit_mols(data, path)
    if not mols:
        if doc is not None:
            return doc
        raise ValueError(f"Could not read the CDX file ({err})")
    from rsstatchem.core.chem_draw import convert
    from rsstatchem.core.chem_draw.model import Document
    from rsstatchem.core.chem_draw.style import preset
    out = doc if doc is not None else Document(preset("ACS Document 1996"))
    x = 0.0
    for m in mols:
        keys = convert.insert_mol(out, m, x, 0.0)
        bb = out.bbox(keys)
        x = (bb[2] if bb else x) + out.style.bond_length * 3
    return out


def _rdkit_mols(data: bytes, path: str | None) -> list:
    if not rdkit_cdx_support():
        return []
    import os
    import tempfile

    from rdkit.Chem import rdChemDraw
    tmp = None
    try:
        if path is None or not os.path.exists(path):
            fd, tmp = tempfile.mkstemp(suffix=".cdx")
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            path = tmp
        return [m for m in rdChemDraw.MolsFromChemDrawFile(path) if m is not None]
    except Exception:  # noqa: BLE001 - RDKit raises on files it cannot parse
        return []
    finally:
        if tmp:
            os.unlink(tmp)


def load_cdx(path: str):
    with open(path, "rb") as fh:
        data = fh.read()
    return from_cdx(data, path)

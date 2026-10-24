"""Document style presets, mirroring ChemDraw's *Document Settings* (File > Apply Document
Settings from): fixed bond length, line/bold widths, bond spacing, hash spacing, label
margin and label font.

All lengths are in points (1/72 inch); the drawing coordinates of a document are in points
too, so a style change rescales the drawing to the new bond length (as ChemDraw does).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace

__all__ = ["Style", "ACS_1996", "NEW_DOCUMENT", "PRESETS", "preset", "ELEMENT_COLORS",
           "element_color"]


@dataclass
class Style:
    name: str = "New Document"
    bond_length: float = 30.0       # pt
    line_width: float = 1.0         # pt
    bold_width: float = 4.0         # pt (bold bonds, wide end of wedges)
    bond_spacing: float = 18.0      # % of bond length between the lines of a double bond
    hash_spacing: float = 2.7       # pt between the strokes of hashed wedges
    margin: float = 2.0             # pt gap between a label and the bond ends
    font_family: str = "Arial"
    font_size: float = 10.0         # pt, atom labels
    caption_size: float = 10.0      # pt, free text and arrow conditions
    color_by_element: bool = True

    @property
    def spacing(self) -> float:
        """Distance between the two lines of a double bond (pt)."""
        return self.bond_length * self.bond_spacing / 100.0

    @property
    def wedge_width(self) -> float:
        """Width of the wide end of wedge / hashed wedge bonds (pt)."""
        return max(self.bold_width * 1.25, self.bond_length * 0.17)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "Style":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (data or {}).items() if k in names})

    def updated(self, **kw) -> "Style":
        return replace(self, **kw)


ACS_1996 = Style("ACS Document 1996", bond_length=14.4, line_width=0.6, bold_width=2.0,
                 bond_spacing=18.0, hash_spacing=2.5, margin=1.6, font_family="Arial",
                 font_size=10.0, caption_size=10.0, color_by_element=False)
NEW_DOCUMENT = Style("New Document", bond_length=30.0, line_width=1.0, bold_width=4.0,
                     bond_spacing=18.0, hash_spacing=2.7, margin=2.0, font_family="Arial",
                     font_size=14.0, caption_size=12.0, color_by_element=True)
PRESETS = {s.name: s for s in (NEW_DOCUMENT, ACS_1996)}


def preset(name: str) -> Style:
    return replace(PRESETS[name])


# ChemDraw-like "Color by element" palette (labels only; carbon stays black).
ELEMENT_COLORS = {
    "N": "#2B50D8", "O": "#E0201B", "S": "#B8960B", "P": "#E07B00", "F": "#1F9E3A",
    "Cl": "#1F9E3A", "Br": "#9B3A1E", "I": "#7B1FA2", "B": "#C06E5E", "Si": "#8C7A5B",
    "Se": "#C47A00", "Li": "#7B3FBF", "Na": "#7B3FBF", "K": "#7B3FBF", "Mg": "#2E8B57",
    "Fe": "#C25E00", "Cu": "#B05A2E", "Zn": "#6D6DA8", "Pd": "#006985", "Pt": "#6F6F8F",
    "Sn": "#5F7F80", "H": "#555555",
}


def element_color(symbol: str, enabled: bool) -> str:
    return ELEMENT_COLORS.get(symbol, "#000000") if enabled else "#000000"

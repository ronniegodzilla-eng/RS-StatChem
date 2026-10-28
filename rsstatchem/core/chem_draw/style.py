"""Document style presets, mirroring ChemDraw's *Document Settings* (File > Apply Document
Settings from): fixed bond length, line/bold widths, bond spacing, hash spacing, label
margin and label font.

All lengths are in points (1/72 inch); the drawing coordinates of a document are in points
too, so a style change rescales the drawing to the new bond length (as ChemDraw does).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace

__all__ = ["Style", "ACS_1996", "NEW_DOCUMENT", "NEW_SLIDE", "RSC", "WILEY", "PRESETS",
           "preset", "ELEMENT_COLORS", "element_color", "CM", "AUTONUMBER_STYLES",
           "autonumber_text"]

CM = 72.0 / 2.54                # points per centimetre
AUTONUMBER_STYLES = ("arabic", "roman", "alpha")


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
    chain_angle: float = 120.0      # deg (all ChemDraw style sheets use 120)
    page_width: float = 540.0       # pt, drawing area of the style sheet
    page_height: float = 720.0
    show_stereo: bool = False       # Object > Show Stereochemistry (R/S, E/Z indicators)
    stereo_parens: bool = True      # Preferences: "Include parentheses" around the labels
    autonumber_style: str = "arabic"    # Reaction Display: Arabic, Roman or Alphabetic
    autonumber_start: int = 1

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


def _sheet(name, length, bold, line, margin, hash_, spacing, label, caption, width, height,
           font="Arial") -> Style:
    return Style(name, bond_length=length, line_width=line, bold_width=bold,
                 bond_spacing=spacing, hash_spacing=hash_, margin=margin, font_family=font,
                 font_size=label, caption_size=caption, color_by_element=False,
                 page_width=width, page_height=height)


# The style sheets shipped with ChemDraw, with the values listed in the ChemDraw 17.1 User
# Guide, "Document Settings" (pp. 48-53): fixed length, bold width, line width, margin
# width, hash spacing, bond spacing (% of length), label / caption size and drawing area.
# Every sheet uses a 120 deg chain angle. Values given in cm (RSC, page sizes) are converted
# to points. The guide does not list the values of the newer "Nature Research" sheet.
ACS_1996 = _sheet("ACS Document 1996", 14.4, 2.0, 0.6, 1.6, 2.5, 18, 10, 10, 540, 720)
ADV_SYNTH = _sheet("Adv. Synth. Catal.", 17.0, 2.0, 1.0, 1.6, 2.5, 18, 10, 12, 540, 720)
J_CHIN = _sheet("J. Chin. Chem. Soc.", 18.0, 2.5, 1.0, 2.0, 2.5, 20, 12, 12, 693, 918)
J_MOL_MOD = _sheet("J. Mol. Mod. (1 column)", 14.4, 2.0, 0.6, 1.6, 2.5, 18, 10, 10,
                   8.5 * CM, 25.4 * CM, font="Times New Roman")
NEW_DOCUMENT = _sheet("New Document", 30.0, 2.0, 1.0, 2.0, 2.7, 12, 10, 12, 540, 720)
NEW_DOCUMENT.color_by_element = True
NEW_SLIDE = _sheet("New Slide", 30.0, 4.0, 1.6, 2.0, 2.7, 15, 12, 16, 540, 720)
PHYTOMEDICINE = _sheet("Phytomedicine", 20.0, 1.33, 1.0, 1.25, 3.0, 8, 12, 12, 540, 720)
RSC = _sheet("RSC", 0.43 * CM, 0.056 * CM, 0.016 * CM, 0.044 * CM, 0.062 * CM, 20, 9, 9,
             8.9 * CM, 25.4 * CM)
SCIENCE_OF_SYNTHESIS = _sheet("Science of Synthesis", 17.0, 2.0, 0.8, 2.25, 2.5, 18, 10, 10,
                              19.79 * CM, 27.15 * CM)
SYNTHESIS = _sheet("Synthesis, Synlett", 17.0, 2.0, 0.8, 1.3, 2.5, 18, 10, 10, 12 * CM,
                   26.7 * CM)
HELV_CHIM = _sheet("Verlag Helvetica Chimica Acta", 17.0, 2.9, 0.54, 2.0, 2.0, 14, 10, 12,
                   368, 720)
WILEY = _sheet("Wiley Document", 17.0, 2.6, 0.75, 2.0, 2.6, 18, 12, 12, 19.79 * CM,
               27.15 * CM)
PRESETS = {s.name: s for s in (NEW_DOCUMENT, ACS_1996, ADV_SYNTH, J_CHIN, J_MOL_MOD, NEW_SLIDE,
                               PHYTOMEDICINE, RSC, SCIENCE_OF_SYNTHESIS, SYNTHESIS, HELV_CHIM,
                               WILEY)}


def preset(name: str) -> Style:
    return replace(PRESETS[name])


def _roman(n: int) -> str:
    vals = ((1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"),
            (50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i"))
    out = ""
    for v, s in vals:
        while n >= v:
            out += s
            n -= v
    return out.upper()


def autonumber_text(n: int, style: str = "arabic") -> str:
    """Compound number ``n`` (1-based) in the Reaction Display numeric style."""
    if style == "roman":
        return _roman(max(1, n))
    if style == "alpha":
        s, k = "", max(1, n)
        while k:
            k, r = divmod(k - 1, 26)
            s = chr(97 + r) + s
        return s
    return str(n)


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

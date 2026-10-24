"""Nicknames / abbreviations (ChemDraw "nicknames"): Me, Et, Ph, Boc, CO2H, NO2...

A nickname is drawn as one labelled atom but expands to its full fragment for every
chemical calculation. Each entry gives the fragment as SMILES with ``*`` at the attachment
point, the label written when the bond comes from the left (``right``, e.g. ``CO2H``) and
from the right (``left``, e.g. ``HO2C``), and the characters of the attachment atom
(``anchor``) that sit on the atom position. An empty anchor centres the whole label.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["Abbrev", "ABBREVIATIONS", "lookup", "is_abbreviation", "label_text",
           "element_with_h", "COMMON"]


@dataclass(frozen=True)
class Abbrev:
    name: str
    smiles: str
    right: str
    left: str
    anchor: str = ""
    title: str = ""


_DATA = [
    # name, smiles, right form, left form, anchor, title
    ("Me", "*C", "Me", "Me", "", "methyl"),
    ("Et", "*CC", "Et", "Et", "", "ethyl"),
    ("Pr", "*CCC", "Pr", "Pr", "", "n-propyl"),
    ("nPr", "*CCC", "nPr", "nPr", "", "n-propyl"),
    ("iPr", "*C(C)C", "iPr", "iPr", "", "isopropyl"),
    ("Bu", "*CCCC", "Bu", "Bu", "", "n-butyl"),
    ("nBu", "*CCCC", "nBu", "nBu", "", "n-butyl"),
    ("iBu", "*CC(C)C", "iBu", "iBu", "", "isobutyl"),
    ("sBu", "*C(C)CC", "sBu", "sBu", "", "sec-butyl"),
    ("tBu", "*C(C)(C)C", "tBu", "tBu", "", "tert-butyl"),
    ("Cy", "*C1CCCCC1", "Cy", "Cy", "", "cyclohexyl"),
    ("Ph", "*c1ccccc1", "Ph", "Ph", "", "phenyl"),
    ("Bn", "*Cc1ccccc1", "Bn", "Bn", "", "benzyl"),
    ("Bz", "*C(=O)c1ccccc1", "Bz", "Bz", "", "benzoyl"),
    ("Ac", "*C(C)=O", "Ac", "Ac", "", "acetyl"),
    ("Piv", "*C(=O)C(C)(C)C", "Piv", "Piv", "", "pivaloyl"),
    ("Boc", "*C(=O)OC(C)(C)C", "Boc", "Boc", "", "tert-butoxycarbonyl"),
    ("Cbz", "*C(=O)OCc1ccccc1", "Cbz", "Cbz", "", "benzyloxycarbonyl"),
    ("Fmoc", "*C(=O)OCC1c2ccccc2-c2ccccc12", "Fmoc", "Fmoc", "",
     "9-fluorenylmethoxycarbonyl"),
    ("Ts", "*S(=O)(=O)c1ccc(C)cc1", "Ts", "Ts", "", "tosyl"),
    ("Ms", "*S(C)(=O)=O", "Ms", "Ms", "", "mesyl"),
    ("Tf", "*S(=O)(=O)C(F)(F)F", "Tf", "Tf", "", "triflyl"),
    ("TMS", "*[Si](C)(C)C", "TMS", "TMS", "", "trimethylsilyl"),
    ("TBS", "*[Si](C)(C)C(C)(C)C", "TBS", "TBS", "", "tert-butyldimethylsilyl"),
    ("TBDMS", "*[Si](C)(C)C(C)(C)C", "TBDMS", "TBDMS", "", "tert-butyldimethylsilyl"),
    ("TIPS", "*[Si](C(C)C)(C(C)C)C(C)C", "TIPS", "TIPS", "", "triisopropylsilyl"),
    ("THP", "*C1CCCCO1", "THP", "THP", "", "tetrahydropyranyl"),
    ("OMe", "*OC", "OMe", "MeO", "O", "methoxy"),
    ("OEt", "*OCC", "OEt", "EtO", "O", "ethoxy"),
    ("OPh", "*Oc1ccccc1", "OPh", "PhO", "O", "phenoxy"),
    ("OBn", "*OCc1ccccc1", "OBn", "BnO", "O", "benzyloxy"),
    ("OAc", "*OC(C)=O", "OAc", "AcO", "O", "acetoxy"),
    ("OTs", "*OS(=O)(=O)c1ccc(C)cc1", "OTs", "TsO", "O", "tosyloxy"),
    ("OTf", "*OS(=O)(=O)C(F)(F)F", "OTf", "TfO", "O", "triflate"),
    ("OtBu", "*OC(C)(C)C", "OtBu", "tBuO", "O", "tert-butoxy"),
    ("OTMS", "*O[Si](C)(C)C", "OTMS", "TMSO", "O", "trimethylsilyloxy"),
    ("OTBS", "*O[Si](C)(C)C(C)(C)C", "OTBS", "TBSO", "O", "TBS ether"),
    ("SMe", "*SC", "SMe", "MeS", "S", "methylthio"),
    ("NMe2", "*N(C)C", "NMe2", "Me2N", "N", "dimethylamino"),
    ("NEt2", "*N(CC)CC", "NEt2", "Et2N", "N", "diethylamino"),
    ("NHAc", "*NC(C)=O", "NHAc", "AcHN", "N", "acetamido"),
    ("NHBoc", "*NC(=O)OC(C)(C)C", "NHBoc", "BocHN", "N", "Boc-amino"),
    ("NHMe", "*NC", "NHMe", "MeHN", "N", "methylamino"),
    ("COOH", "*C(=O)O", "COOH", "HOOC", "C", "carboxylic acid"),
    ("CO2H", "*C(=O)O", "CO2H", "HO2C", "C", "carboxylic acid"),
    ("COOMe", "*C(=O)OC", "COOMe", "MeOOC", "C", "methyl ester"),
    ("CO2Me", "*C(=O)OC", "CO2Me", "MeO2C", "C", "methyl ester"),
    ("CO2Et", "*C(=O)OCC", "CO2Et", "EtO2C", "C", "ethyl ester"),
    ("COOEt", "*C(=O)OCC", "COOEt", "EtOOC", "C", "ethyl ester"),
    ("CO2tBu", "*C(=O)OC(C)(C)C", "CO2tBu", "tBuO2C", "C", "tert-butyl ester"),
    ("COCl", "*C(=O)Cl", "COCl", "ClOC", "C", "acyl chloride"),
    ("CONH2", "*C(N)=O", "CONH2", "H2NOC", "C", "carboxamide"),
    ("CHO", "*C=O", "CHO", "OHC", "C", "formyl"),
    ("CN", "*C#N", "CN", "NC", "C", "cyano"),
    ("NC", "*[N+]#[C-]", "NC", "CN", "N", "isocyano"),
    ("NCO", "*N=C=O", "NCO", "OCN", "N", "isocyanate"),
    ("NCS", "*N=C=S", "NCS", "SCN", "N", "isothiocyanate"),
    ("SCN", "*SC#N", "SCN", "NCS", "S", "thiocyanate"),
    ("N3", "*N=[N+]=[N-]", "N3", "N3", "N", "azido"),
    ("NO2", "*[N+](=O)[O-]", "NO2", "O2N", "N", "nitro"),
    ("NO", "*N=O", "NO", "ON", "N", "nitroso"),
    ("CF3", "*C(F)(F)F", "CF3", "F3C", "C", "trifluoromethyl"),
    ("CCl3", "*C(Cl)(Cl)Cl", "CCl3", "Cl3C", "C", "trichloromethyl"),
    ("OCF3", "*OC(F)(F)F", "OCF3", "F3CO", "O", "trifluoromethoxy"),
    ("SO3H", "*S(=O)(=O)O", "SO3H", "HO3S", "S", "sulfonic acid"),
    ("SO2Cl", "*S(=O)(=O)Cl", "SO2Cl", "ClO2S", "S", "sulfonyl chloride"),
    ("SO2NH2", "*S(N)(=O)=O", "SO2NH2", "H2NO2S", "S", "sulfonamide"),
    ("PO3H2", "*P(=O)(O)O", "PO3H2", "H2O3P", "P", "phosphonic acid"),
    ("SiMe3", "*[Si](C)(C)C", "SiMe3", "Me3Si", "Si", "trimethylsilyl"),
    ("CH2OH", "*CO", "CH2OH", "HOH2C", "C", "hydroxymethyl"),
    ("OH", "*O", "OH", "HO", "O", "hydroxy"),
    ("SH", "*S", "SH", "HS", "S", "sulfanyl"),
    ("NH2", "*N", "NH2", "H2N", "N", "amino"),
    ("CH3", "*C", "CH3", "H3C", "C", "methyl"),
    ("C2H5", "*CC", "C2H5", "H5C2", "C", "ethyl"),
]
ABBREVIATIONS: dict[str, Abbrev] = {d[0]: Abbrev(*d) for d in _DATA}
_LOWER = {k.lower(): v for k, v in ABBREVIATIONS.items()}
# Labels that are plain elements with implicit hydrogens (typing "OH" on an atom makes an O).
_ELEMENT_H = re.compile(r"^([A-Z][a-z]?)(H(\d*))?$")
COMMON = ["Me", "Et", "Pr", "iPr", "tBu", "Ph", "Bn", "Ac", "OMe", "OH", "COOH", "CHO", "CN",
          "NO2", "NH2", "CF3", "Boc", "Ts", "TMS"]


def lookup(text: str) -> Abbrev | None:
    """The nickname for a typed label (exact case first, then case-insensitive)."""
    t = (text or "").strip()
    return ABBREVIATIONS.get(t) or _LOWER.get(t.lower())


def is_abbreviation(text: str) -> bool:
    return lookup(text) is not None


def element_with_h(text: str, is_symbol) -> tuple[str, int | None] | None:
    """``("O", 1)`` for "OH", ``("N", 2)`` for "NH2", ``("Cl", None)`` for "Cl"; None if the
    text is not an element optionally followed by hydrogens. ``is_symbol`` validates."""
    m = _ELEMENT_H.match((text or "").strip())
    if not m or not is_symbol(m.group(1)):
        return None
    if m.group(2) is None:
        return m.group(1), None
    return m.group(1), int(m.group(3) or 1)


def label_text(label: str, from_right: bool) -> tuple[str, str]:
    """(text, anchor) to draw a label whose bond arrives from the right (``from_right``)
    or the left. Non-nickname labels are drawn as typed and centred."""
    ab = lookup(label)
    if ab is None:
        return label, ""
    if ab.name.lower() == label.lower() or label in (ab.right, ab.left):
        return (ab.left if from_right else ab.right), ab.anchor
    return label, ""

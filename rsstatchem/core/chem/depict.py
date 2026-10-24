"""2D structure depiction (SVG / PNG) of molecules and reactions with RDKit ``rdMolDraw2D``.

The pictures are what ChemDraw pastes into reports: black/coloured atom labels, wedge bonds,
optional atom indices, CIP labels, highlighted substructures, transparent background. PNGs
are rendered at ``scale`` × the logical size so they stay sharp in the Output Viewer and in
Word export (pass the logical width to ``Section.image``).
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from rsstatchem.core.chem.molecule import (ChemError, compute_2d, detect_format, has_2d,
                                           mol_from_text, reaction_from_text,
                                           substructure_matches)

__all__ = ["DrawOptions", "depict_svg", "depict_png", "depict_grid_png", "depict_grid_svg",
           "depict_reaction_svg", "depict_reaction_png", "depict_text_png", "parse_color"]


@dataclass
class DrawOptions:
    width: int = 320
    height: int = 240
    transparent: bool = False
    background: str = "#FFFFFF"
    bond_width: float = 2.0
    font_size: float | None = None        # base font size (RDKit default ≈ 0.6 bond length)
    min_font_size: int = 9
    atom_indices: bool = False
    bond_indices: bool = False
    stereo_labels: bool = False           # CIP (R/S, E/Z) annotations
    explicit_methyl: bool = False
    black_white: bool = False
    show_hydrogens: bool = False          # draw all H atoms explicitly
    padding: float = 0.05
    rotate: float = 0.0
    fixed_bond_length: float | None = None
    comic: bool = False
    legend: str = ""
    highlight_atoms: list = field(default_factory=list)
    highlight_bonds: list = field(default_factory=list)
    highlight_smarts: str = ""
    highlight_color: str = "#FF9E9E"
    kekulize: bool = True
    clean: bool = False                    # regenerate 2D coordinates before drawing

    def updated(self, **kw) -> "DrawOptions":
        names = {f.name for f in fields(self)}
        bad = set(kw) - names
        if bad:
            raise TypeError(f"Unknown drawing option(s): {', '.join(sorted(bad))}")
        return replace(self, **kw)


def parse_color(c, alpha: float = 1.0) -> tuple:
    if isinstance(c, (tuple, list)):
        vals = tuple(float(x) for x in c)
        vals = tuple(v / 255.0 for v in vals) if any(v > 1 for v in vals) else vals
        return vals if len(vals) == 4 else vals + (alpha,)
    s = str(c).lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) not in (6, 8):
        raise ValueError(f"Invalid colour '{c}'")
    vals = [int(s[i:i + 2], 16) / 255.0 for i in range(0, len(s), 2)]
    return tuple(vals) if len(vals) == 4 else tuple(vals) + (alpha,)


def _options(options: DrawOptions | None, kw: dict) -> DrawOptions:
    opts = options or DrawOptions()
    return opts.updated(**kw) if kw else opts


def _configure(drawer, o: DrawOptions, scale: float) -> None:
    d = drawer.drawOptions()
    d.bondLineWidth = max(1, int(round(o.bond_width * scale)))
    d.scaleBondWidth = False
    d.minFontSize = int(o.min_font_size * scale)
    if o.font_size:
        d.baseFontSize = o.font_size
    d.addAtomIndices = o.atom_indices
    d.addBondIndices = o.bond_indices
    d.addStereoAnnotation = o.stereo_labels
    d.explicitMethyl = o.explicit_methyl
    d.padding = o.padding
    d.rotate = o.rotate
    d.comicMode = o.comic
    if o.fixed_bond_length:
        d.fixedBondLength = o.fixed_bond_length * scale
    if o.black_white:
        d.useBWAtomPalette()
    if o.transparent:
        d.clearBackground = False
    else:
        d.setBackgroundColour(parse_color(o.background))
    d.setHighlightColour(parse_color(o.highlight_color))
    d.legendFontSize = int(14 * scale)


def _prepare(mol: Chem.Mol, o: DrawOptions) -> Chem.Mol:
    if mol is None:
        raise ChemError("No molecule to draw")
    m = Chem.Mol(mol)
    if o.show_hydrogens:
        m = Chem.AddHs(m, addCoords=has_2d(m))
    if o.clean or not has_2d(m):
        compute_2d(m)
    try:
        return rdMolDraw2D.PrepareMolForDrawing(m, kekulize=o.kekulize)
    except Exception:  # noqa: BLE001 - e.g. non-kekulizable query molecules
        return rdMolDraw2D.PrepareMolForDrawing(m, kekulize=False)


def _highlights(mol: Chem.Mol, o: DrawOptions) -> tuple[list, list]:
    atoms, bonds = list(o.highlight_atoms), list(o.highlight_bonds)
    if o.highlight_smarts:
        a, b = substructure_matches(mol, o.highlight_smarts)
        atoms += a
        bonds += b
    return sorted(set(atoms)), sorted(set(bonds))


def _draw(drawer, mol: Chem.Mol, o: DrawOptions, scale: float) -> None:
    _configure(drawer, o, scale)
    m = _prepare(mol, o)
    atoms, bonds = _highlights(m, o)
    drawer.DrawMolecule(m, highlightAtoms=atoms or None, highlightBonds=bonds or None,
                        legend=o.legend)
    drawer.FinishDrawing()


def depict_svg(mol: Chem.Mol, width: int | None = None, height: int | None = None,
               options: DrawOptions | None = None, **kw) -> str:
    """SVG text of a 2D depiction. Keyword arguments override :class:`DrawOptions`."""
    o = _options(options, kw)
    drawer = rdMolDraw2D.MolDraw2DSVG(width or o.width, height or o.height)
    _draw(drawer, mol, o, 1.0)
    return drawer.GetDrawingText()


def depict_png(mol: Chem.Mol, width: int | None = None, height: int | None = None,
               options: DrawOptions | None = None, scale: float = 2.0, **kw) -> bytes:
    """PNG bytes of ``scale`` × (width × height) pixels."""
    o = _options(options, kw)
    w, h = int((width or o.width) * scale), int((height or o.height) * scale)
    drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
    _draw(drawer, mol, o, scale)
    return drawer.GetDrawingText()


def _grid(drawer_cls, mols, legends, cols, o, scale):
    n = len(mols)
    cols = max(1, min(cols, n))
    rows = (n + cols - 1) // cols
    cw, ch = int(o.width * scale), int(o.height * scale)
    drawer = drawer_cls(cw * cols, ch * rows, cw, ch)
    _configure(drawer, o, scale)
    prepared = [_prepare(m, o) for m in mols]
    drawer.DrawMolecules(prepared, legends=[str(x) for x in (legends or [""] * n)])
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def depict_grid_png(mols: list, legends: list | None = None, cols: int = 4,
                    options: DrawOptions | None = None, scale: float = 2.0, **kw) -> bytes:
    """Grid image (like ChemDraw's table of structures); cell size = options width/height."""
    if not mols:
        raise ChemError("No molecules to draw")
    return _grid(rdMolDraw2D.MolDraw2DCairo, mols, legends, cols, _options(options, kw), scale)


def depict_grid_svg(mols: list, legends: list | None = None, cols: int = 4,
                    options: DrawOptions | None = None, **kw) -> str:
    if not mols:
        raise ChemError("No molecules to draw")
    return _grid(rdMolDraw2D.MolDraw2DSVG, mols, legends, cols, _options(options, kw), 1.0)


def _draw_reaction(drawer, rxn, o: DrawOptions, scale: float) -> None:
    _configure(drawer, o, scale)
    drawer.DrawReaction(rxn)
    drawer.FinishDrawing()


def depict_reaction_svg(rxn, width: int = 600, height: int = 200,
                        options: DrawOptions | None = None, **kw) -> str:
    """Reactants → products drawing of an RDKit ``ChemicalReaction`` (or reaction text)."""
    if isinstance(rxn, str):
        rxn = reaction_from_text(rxn)
    o = _options(options, kw)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    _draw_reaction(drawer, rxn, o, 1.0)
    return drawer.GetDrawingText()


def depict_reaction_png(rxn, width: int = 600, height: int = 200,
                        options: DrawOptions | None = None, scale: float = 2.0, **kw) -> bytes:
    if isinstance(rxn, str):
        rxn = reaction_from_text(rxn)
    o = _options(options, kw)
    drawer = rdMolDraw2D.MolDraw2DCairo(int(width * scale), int(height * scale))
    _draw_reaction(drawer, rxn, o, scale)
    return drawer.GetDrawingText()


def depict_text_png(text: str, options: DrawOptions | None = None, scale: float = 2.0,
                    **kw) -> bytes:
    """PNG for SMILES / InChI / MOL text or a reaction SMILES (detected automatically)."""
    o = _options(options, kw)
    if detect_format(text) == "reaction":
        return depict_reaction_png(text, o.width, o.height, o, scale)
    return depict_png(mol_from_text(text), options=o, scale=scale)

"""Chemical drawing core (no Qt) for the ChemDraw-style structure and reaction editor.

- :mod:`model` - the drawing document (atoms, bonds, arrows, "+" signs, text, brackets),
  editing operations and the ``.rsc`` JSON format.
- :mod:`style` - document style presets (ACS Document 1996, New Document).
- :mod:`geometry` - bond angles, zig-zag chains, ring templates, snapping, double-bond
  placement and bond/arrow drawing primitives.
- :mod:`abbrev` - nicknames (Me, Ph, Boc, CO2H...) and how they expand chemically.
- :mod:`labels` - which atom labels are shown, implicit hydrogens and label alignment.
- :mod:`convert` - drawing <-> RDKit (via ``core.chem.molecule``), reactions, SMILES/MOL/RXN.
- :mod:`analysis` - ChemDraw "Analysis" window values, balance check and stoichiometry grid.
- :mod:`templates` - template library (amino acids, nucleobases, sugars, groups).
"""

"""Chemistry menu, stage 2: a native ChemDraw-style structure and reaction drawing editor.

The editor (:class:`~rsstatchem.ui.chem_editor.window.ChemEditorWindow`) draws molecules
with bond / chain / ring / atom-label tools, keeps them chemically aware through the stage-1
RDKit core (live valence marks, Clean Structure, the Analysis panel), draws reactions
(arrows, "+", conditions; balance check and stoichiometry grid), and saves ``.rsc`` drawings
or exports SMILES / MOL / SDF / InChI / RXN / SVG / PNG / PDF. Documents and editing
operations live in :mod:`rsstatchem.core.chem_draw` (no Qt).

Other modules can open the editor with content, e.g.
``open_structure_editor(editor, smiles="c1ccccc1")``.
"""
from __future__ import annotations

__all__ = ["MENUS", "selftest", "open_structure_editor", "open_reaction_editor",
           "open_drawing_file"]

_WINDOWS: list = []     # strong references to open editor windows


def open_structure_editor(editor=None, smiles: str | None = None, text: str | None = None,
                          path: str | None = None, reaction: bool = False, show: bool = True):
    """Open a new editor window; optional initial content (SMILES / MOL / InChI / reaction
    SMILES / RXN text) or a drawing / structure file. Returns the window."""
    from rsstatchem.core.chem_draw.model import Document
    from rsstatchem.core.chem_draw.style import preset
    from rsstatchem.ui.chem_editor.window import ChemEditorWindow

    win = ChemEditorWindow(editor, Document(preset("New Document")), reaction=reaction)
    _WINDOWS.append(win)
    content = smiles or text
    if content:
        try:
            win.load_text(content)
        except Exception as exc:  # noqa: BLE001 - open the window anyway, report the error
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(win, "Structure Editor", f"Could not read the structure:\n{exc}")
    if path:
        win.open_file(path)
        win.undo_stack.clear()
    if show:
        win.show()
        win.raise_()
        win.activateWindow()
    return win


def open_reaction_editor(editor=None, **kw):
    """The editor with the reaction-arrow tool active."""
    return open_structure_editor(editor, reaction=True, **kw)


def open_drawing_file(editor=None, path: str | None = None):
    if path is None:
        from PySide6.QtWidgets import QFileDialog

        from rsstatchem.ui.chem_editor.window import OPEN_FILTER
        start = ""
        ctx = getattr(editor, "ctx", None)
        if ctx is not None and getattr(ctx, "settings", None) is not None:
            start = str(ctx.settings.value("last_dir", ""))
        path, _ = QFileDialog.getOpenFileName(editor, "Open Chemical Drawing", start,
                                              OPEN_FILTER)
        if not path:
            return None
    return open_structure_editor(editor, path=path)


MENUS = {
    "Chemistry": [
        ("Structure Editor...", open_structure_editor),
        ("Reaction Editor...", open_reaction_editor),
        ("Open Chemical Drawing...", open_drawing_file),
        None,
    ],
}


def selftest() -> None:
    """Headless check of the drawing core (no Qt widgets): editing operations, nicknames,
    stereo, serialisation, reactions and the analysis values."""
    from rsstatchem.core.chem_draw import analysis, convert, ops
    from rsstatchem.core.chem_draw.model import Document

    def check(cond: bool, what: str) -> None:
        if not cond:
            raise AssertionError(f"structure editor selftest failed: {what}")

    doc = Document()
    ring = ops.stamp_ring(doc, "benzene", point=(0.0, 0.0))
    check(convert.to_smiles(doc) == "c1ccccc1", "benzene template")
    a, _ = ops.add_bond_at_atom(doc, ring[1])
    convert.apply_label_text(doc, a, "COOH")
    b, _ = ops.add_bond_at_atom(doc, ring[2])
    convert.apply_label_text(doc, b, "OAc")
    check(convert.to_inchikey(doc) == "BSYNRYMUTXBXSQ-UHFFFAOYSA-N", "aspirin via nicknames")
    again = Document.from_json(doc.to_json())
    check(convert.to_smiles(again) == convert.to_smiles(doc), ".rsc round trip")
    res = analysis.analyze(doc)
    check(res.formula == "C9H8O4" and abs(res.mw - 180.159) < 0.01, "analysis")
    d2 = Document()
    ids = ops.add_chain(d2, (0.0, 0.0), 0.0, 3)
    d2.bond_between(ids[1], ids[2]).order = 2
    check(convert.to_smiles(d2) == "C/C=C/C", "E-2-butene from geometry")
    rx = Document()
    convert.insert_text(rx, "CC(=O)O.OCC>>CC(=O)OCC.O", 0.0, 0.0)
    check(analysis.balance_check(rx).balanced, "esterification balance")
    check(convert.reaction_smiles(rx) == "CC(=O)O.CCO>>CCOC(C)=O.O", "reaction split")

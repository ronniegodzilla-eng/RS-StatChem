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
    _selftest_names(check)
    _selftest_properties(check)


def _selftest_properties(check) -> None:
    """Chemical Properties data: the experimental database, the numpy ML models, the group
    contributions and - in a packaged build with RSSTATCHEM_SELFTEST_CDK=bundled (CI), or
    whenever a bundled Java runtime is present - the CDK logP helper on the bundled Java."""
    import os

    from rsstatchem.core.chem import molecule as mo
    from rsstatchem.core.chem_draw import cdk, exp_props, ml_props, names
    from rsstatchem.core.chem_draw.properties import chemical_properties, marrero_gani

    caffeine = mo.mol_from_text("Cn1cnc2c1c(=O)n(C)c(=O)n2C")
    check(exp_props.db_size() > 20000, "experimental property database missing")
    mp = exp_props.lookup(caffeine).get("mp")
    check(mp is not None and abs(mp.value - 510.6) < 3 and "B" in mp.sources,
          "experimental melting point of caffeine")
    check(set(ml_props.models()) == {"mp", "bp", "logp"}, "ML property models missing")
    novel = mo.mol_from_text("O=C(NC1CC2(C1)OCCO2)c1ccc(F)c(Cl)c1")
    pred = ml_props.predict(novel, "mp")
    check(pred.value is not None and 250 < pred.value < 600 and pred.expected_rmse,
          f"ML melting point estimate ({pred.error or pred.value})")
    tol = marrero_gani(mo.mol_from_text("Cc1ccccc1"))
    check(tol.ok and abs(tol.values["tc"] - 600) < 20, "Marrero-Gani estimate")
    from rsstatchem.core.chem_draw import nannoolal
    tmb = nannoolal.nannoolal(mo.mol_from_text("CC(C)(C)C(C)(C)C"), tb=379.6)
    check(tmb["tc"] is not None and abs(tmb["tc"] - 566.8) < 0.1, "Nannoolal Tc estimate")
    third = marrero_gani(mo.mol_from_text("c1cc2ccc3cccc4ccc(c1)c2c34"), third_order=True)
    check(abs(third.values["tb"] - 673.96) < 0.01, "Marrero-Gani third-order estimate")
    rows = {p.name: p for p in chemical_properties(caffeine)}
    check(rows["Melting Point"].kind == "experimental", "properties rows")
    vp = {p.name: p for p in chemical_properties(mo.mol_from_text("CCCCCCC(C)OC(=O)CC"))}
    check(vp["Vapor Pres."].value is not None and vp["Vapor Pres."].short == "Nannoolal",
          "vapour-pressure estimate")
    print(f"selftest ok: experimental database ({exp_props.db_size()} compounds), "
          f"ML models, group contributions")
    mode = os.environ.get("RSSTATCHEM_SELFTEST_CDK", "").lower()
    if not mode and not cdk.cdk_available(allow_system=False):
        return
    check(cdk.cdk_jars() is not None, "CDK jars / helper are not bundled")
    if mode == "bundled":
        check(names.find_java(allow_system=False) is not None, "bundled Java runtime missing")
    helper = cdk.CdkHelper()
    try:
        res = helper.compute("CC(=O)Oc1ccccc1C(=O)O", timeout=120)
    finally:
        helper.close()
    check(res.get("jplogp") is not None and abs(res["jplogp"] - 1.24) < 0.3 and
          res.get("xlogp") is not None and res.get("alogp") is not None,
          f"CDK logP of aspirin ({res})")
    print(f"selftest ok: CDK {helper.version} JPlogP/XLogP/ALogP ({names.find_java()})")


def _selftest_names(check) -> None:
    """Offline structure <-> name: the bundled name database and OPSIN. A packaged build sets
    RSSTATCHEM_SELFTEST_OPSIN=bundled (CI) so the bundled Java runtime must be used; else
    OPSIN is checked whenever a bundled runtime is present."""
    import os

    from rsstatchem.core.chem import molecule as mo
    from rsstatchem.core.chem_draw import names

    check(names.name_db_size() > 10000, "offline name database missing")
    hit = names.db_name(mo.mol_from_text("Cn1cnc2c1c(=O)n(C)c(=O)n2C"))
    check(hit is not None and hit.common == "Caffeine", "name database lookup")
    mode = os.environ.get("RSSTATCHEM_SELFTEST_OPSIN", "").lower()
    if not mode and not names.opsin_available(allow_system=False):
        return
    check(names.opsin_jar() is not None, "OPSIN jar is not bundled")
    if mode == "bundled":
        check(names.find_java(allow_system=False) is not None, "bundled Java runtime missing")
    res = names.opsin_structure("2-acetoxybenzoic acid", timeout=120)
    check(mo.to_inchikey(mo.mol_from_text(res.smiles)) == "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
          "OPSIN name -> structure (2-acetoxybenzoic acid -> aspirin)")
    print(f"selftest ok: OPSIN offline name -> structure ({names.find_java()})")

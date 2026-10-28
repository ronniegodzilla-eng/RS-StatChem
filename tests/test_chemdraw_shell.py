"""Chemical Drawing, WP-C shell: ChemDraw 17.1 menu paths and shortcuts, Paste Special / Copy
As, CDX files and clipboard, Join, Repeat, Rotate 180° (3D), reaction tools (Clean Up
Reaction, interpretation overlay, mapping greyed out without Indigo), rulers / crosshair,
Document Annotations, image exports and the install() hooks of the other work packages."""
from __future__ import annotations

import os
import sys
import types

import pytest
from PySide6.QtGui import QAction, QGuiApplication, QKeySequence

from rsstatchem.core.chem import molecule as mo
from rsstatchem.core.chem_draw import cdx, convert, mapping
from rsstatchem.core.chem_draw.model import Document, load_rsc

SCREENSHOTS = os.environ.get("RSSTATCHEM_SHOTS", "")
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


@pytest.fixture
def ctx(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("RSSTATCHEM_HOME", str(tmp_path))
    from rsstatchem.app import AppContext

    context = AppContext(qapp)
    monkeypatch.setattr(context.app, "quit", lambda: None)
    return context


@pytest.fixture
def win(ctx, qapp):
    from rsstatchem.ui.modules import open_module

    w = open_module(ctx, "chem")
    w.resize(1200, 820)
    qapp.processEvents()
    yield w
    w.confirm_close = False
    w.close()


def _menu(win, title):
    for act in win.menuBar().actions():
        if act.text().replace("&", "") == title:
            return act.menu()
    raise KeyError(title)


def _find(menu, path: str):
    """Action at 'Sub > Item' below ``menu`` (ampersands ignored)."""
    parts = [p.strip() for p in path.split(">")]

    def norm(t):
        return t.replace("&", "").removesuffix("...").strip()
    for act in menu.actions():
        if norm(act.text()) == norm(parts[0]):
            if len(parts) == 1:
                return act
            return _find(act.menu(), " > ".join(parts[1:]))
    raise KeyError(path)


def _all_actions(win):
    out = []

    def walk(menu):
        for a in menu.actions():
            if a.menu() is not None:
                walk(a.menu())
            elif not a.isSeparator():
                out.append(a)
    for top in win.menuBar().actions():
        walk(top.menu())
    return out


def _shot(win, name):
    if SCREENSHOTS:
        win.grab().save(os.path.join(SCREENSHOTS, name))


def _load(win, text):
    win.load_text(text)
    win.canvas.select_keys(set())


# --- menus and shortcuts --------------------------------------------------------------------

CHEMDRAW_PATHS = [  # ChemDraw 17.1 User Guide menu paths that the window backs
    "File > New Document", "File > Open Style Sheets > New Document", "File > Open",
    "File > Close", "File > Save", "File > Save As", "File > Revert",
    "File > Document Annotations...", "File > Document Settings...",
    "File > Apply Document Settings from > ACS Document 1996", "File > List Nicknames...",
    "File > Page Setup...", "File > Print...",
    "Edit > Undo", "Edit > Redo", "Edit > Repeat", "Edit > Copy As > CDXML Text",
    "Edit > Copy As > HELM", "Edit > Copy As > InChI", "Edit > Copy As > InChIKey",
    "Edit > Copy As > SMILES", "Edit > Copy As > MOL V3000 Text", "Edit > Paste",
    "Edit > Paste Special > FASTA DNA", "Edit > Paste Special > FASTA Peptide",
    "Edit > Paste Special > FASTA RNA", "Edit > Paste Special > Name as Structure",
    "Edit > Paste Special > HELM", "Edit > Paste Special > InChI",
    "Edit > Paste Special > MOL / CDXML", "Edit > Paste Special > SMILES", "Edit > Clear",
    "Edit > Select All", "Edit > Insert File...",
    "View > Show Main Toolbar", "View > Show Analysis Window",
    "View > Show Chemical Properties Window", "View > Show Reaction Interpretation",
    "View > Show Rulers", "View > Show Crosshair", "View > Magnify", "View > Reduce",
    "View > Actual Size", "View > Fit to Window",
    "Object > Fixed Lengths", "Object > Fixed Angles", "Object > Bring to Front",
    "Object > Send to Back", "Object > Group", "Object > Ungroup", "Object > Join",
    "Object > Align > Lefts", "Object > Distribute > Horizontally", "Object > Flip Horizontal",
    "Object > Rotate 180° Horizontal", "Object > Rotate...", "Object > Scale...",
    "Object > Center on Page",
    "Structure > Atom Properties...", "Structure > Expand Label", "Structure > Check Structure",
    "Structure > Clean Up Structure", "Structure > Clean Up Reaction",
    "Structure > Convert Name to Structure...", "Structure > Convert Structure to Name",
    "Structure > Map Reaction Atoms", "Structure > Clear Reaction Map",
    "Colors > Color by Element", "Help > Shortcuts and Hotkeys",
]


def test_menu_paths_and_bar(win):
    titles = [a.text().replace("&", "") for a in win.menuBar().actions()]
    assert titles == ["File", "Edit", "View", "Object", "Structure", "Text", "Curves",
                      "Colors", "Window", "Help"]
    for path in CHEMDRAW_PATHS:
        menu, rest = path.split(" > ", 1)
        assert _find(_menu(win, menu), rest) is not None, path
    assert set(win.menus) >= {"File", "Edit", "View", "Object", "Structure", "Text", "Curves",
                              "Colors", "Window", "Help"}


def test_chemdraw_shortcut_table(win):
    from rsstatchem.ui.chemdraw.window import MENU_SHORTCUTS
    by_key: dict[str, list[str]] = {}
    for a in win.findChildren(QAction):
        for k in a.shortcuts():
            if not k.isEmpty():
                by_key.setdefault(k.toString(), []).append(a.text())
    clashes = {k: v for k, v in by_key.items() if len(v) > 1}
    assert not clashes, clashes                      # Qt ignores ambiguous shortcuts
    for menu, command, keys in MENU_SHORTCUTS:
        k = QKeySequence(keys).toString()
        assert k in by_key, (menu, command, keys)
    assert win.act_check.shortcut().isEmpty()       # F5 is Actual Size, as in ChemDraw
    assert QKeySequence("F5") in win.act_actual.shortcuts()
    assert win.act_redo.shortcut() == QKeySequence("Ctrl+Shift+Z")
    assert win.act_repeat.shortcut() == QKeySequence("Ctrl+Y")
    dlg = win.show_hotkeys()
    html = dlg.findChildren(__import__("PySide6.QtWidgets", fromlist=["QTextBrowser"])
                            .QTextBrowser)[0].toHtml()
    assert "Menu shortcuts" in html and "Join" in html


def test_view_keys_zoom(win, qapp):
    win.act_actual.trigger()
    assert abs(win.canvas.zoom_percent() - 100) < 0.5
    win.act_zoom_in.trigger()
    assert win.canvas.zoom_percent() > 100


# --- Paste Special / Copy As ---------------------------------------------------------------

def test_paste_special_kinds(win):
    keys = win.paste_special("smiles", "CCO ethanol")
    assert keys and convert.to_smiles(win.doc) == "CCO"
    win.canvas.set_state(Document(win.doc.style).to_dict())
    win.paste_special("fasta_peptide", ">p\nAGC\n")
    assert convert.to_smiles(win.doc) == "C[C@H](N)C(=O)NCC(=O)N[C@@H](CS)C(=O)O"
    win.canvas.set_state(Document(win.doc.style).to_dict())
    win.paste_special("helm", "PEPTIDE1{G.G}$$$$")
    assert convert.to_smiles(win.doc) == "NCC(=O)NCC(=O)O"
    win.canvas.set_state(Document(win.doc.style).to_dict())
    win.paste_special("fasta_dna", "ACG")
    assert len(win.doc.atoms) > 40
    win.canvas.set_state(Document(win.doc.style).to_dict())
    xml = convert.Chem.MolToMolBlock(convert.Chem.MolFromSmiles("CCN"))
    win.paste_special("molcdxml", xml)
    assert convert.to_smiles(win.doc) == "CCN"
    from rsstatchem.core.chem_draw import cdxml
    d = Document(win.doc.style)
    convert.insert_text(d, "c1ccccc1", 0, 0)
    win.paste_special("molcdxml", cdxml.to_cdxml(d))
    assert "c1ccccc1" in convert.to_smiles(win.doc)


def test_paste_special_name_offline(win):
    win.use_opsin = False
    win.online_lookup = False
    win.paste_special("name", "caffeine")
    assert convert.to_inchikey(win.doc) == "RYYVLZVUVIJVGH-UHFFFAOYSA-N"


def test_copy_as_helm_mol3_cdxml_and_smiles_keys(win):
    _load(win, "PEPTIDE1{A.G.C}$$$$")
    cb = QGuiApplication.clipboard()
    win.copy_as("helm")
    assert cb.text() == "PEPTIDE1{A.G.C}$$$$"
    win.copy_as("fasta")
    assert cb.text().splitlines()[1] == "AGC"
    win.copy_as("mol3")
    assert "V3000" in cb.text()
    _find(_menu(win, "Edit"), "Copy As > CDXML Text").trigger()
    assert cb.text().lstrip().startswith("<?xml") and "<CDXML" in cb.text()
    _find(_menu(win, "Edit"), "Copy As > SMILES").trigger()
    assert cb.text() == "C[C@H](N)C(=O)NCC(=O)N[C@@H](CS)C(=O)O"


def test_clipboard_carries_cdx_and_pastes_back(win):
    from rsstatchem.ui.chem_editor import export
    _load(win, ASPIRIN)
    md = export.mime_data(win.doc)
    assert md.hasFormat("chemical/x-cdx") and md.hasFormat("chemical/x-cdxml")
    assert bytes(md.data("chemical/x-cdx"))[:8] == b"VjCD0100"
    from PySide6.QtCore import QByteArray, QMimeData
    only = QMimeData()                       # as if copied from ChemDraw: CDX only
    only.setData("chemical/x-cdx", QByteArray(bytes(md.data("chemical/x-cdx"))))
    QGuiApplication.clipboard().setMimeData(only)
    n = len(win.doc.atoms)
    win.paste()
    assert len(win.doc.atoms) == 2 * n


# --- files -----------------------------------------------------------------------------------

def test_open_save_cdx_and_exports(win, ctx, tmp_path):
    _load(win, ASPIRIN)
    path = str(tmp_path / "aspirin.cdx")
    assert win.save_as(path)
    assert open(path, "rb").read(8) == b"VjCD0100"
    assert convert.to_smiles(cdx.load_cdx(path)) == ASPIRIN
    from rsstatchem.ui.chemdraw.window import ChemDrawWindow
    w2 = ChemDrawWindow(ctx)
    assert w2.open_file(path)
    assert convert.to_smiles(w2.doc) == ASPIRIN and w2.path == path
    win.canvas.edit("t", lambda: win.doc.add_text(10, 10, "x"))
    win.confirm_close = False                 # File > Revert without the question
    assert win.revert() and not win.doc.texts and win.undo_stack.isClean()
    for fmt in ("tiff", "jpg", "bmp", "png"):
        out = str(tmp_path / f"a.{fmt}")
        assert win.export_as(fmt, out, dpi=150)
        assert os.path.getsize(out) > 500
    assert win.export_as("mol3", str(tmp_path / "a3.mol"))
    assert "V3000" in (tmp_path / "a3.mol").read_text()
    assert win.export_as("cdx", str(tmp_path / "e.cdx"))
    assert (tmp_path / "e.cdx").read_bytes()[:8] == b"VjCD0100"
    w2.confirm_close = False
    w2.close()


def test_insert_file_and_helm_file(win, tmp_path):
    p = tmp_path / "p.helm"
    p.write_text("PEPTIDE1{G.A}$$$$", encoding="utf-8")
    keys = win.insert_file(str(p))
    assert keys and convert.to_smiles(win.doc) == "C[C@H](NC(=O)CN)C(=O)O"
    q = tmp_path / "b.cdx"
    d = Document(win.doc.style)
    convert.insert_text(d, "c1ccncc1", 0, 0)
    cdx.save_cdx(d, str(q))
    win.insert_file(str(q))
    assert "c1ccncc1" in convert.to_smiles(win.doc)


def test_document_annotations_saved(win, tmp_path):
    win.document_annotations({"Name": "Aspirin", "Course": "Organic I"})
    assert win.doc.meta["annotations"]["Course"] == "Organic I"
    path = str(tmp_path / "a.rsc")
    win.save_as(path)
    assert load_rsc(path).meta["annotations"] == {"Name": "Aspirin", "Course": "Organic I"}
    win.undo_stack.undo()
    assert "annotations" not in win.doc.meta


def test_open_style_sheet_and_nicknames(win, ctx):
    assert win.is_pristine()
    win.open_style_sheet("New Document")
    assert win.doc.style.name == "New Document"
    assert ctx.settings.value("chemdraw/default_style") == "New Document"
    from rsstatchem.ui.chemdraw.window import ChemDrawWindow
    w = ChemDrawWindow(ctx)
    assert w.doc.style.name == "New Document"
    ctx.settings.setValue("chemdraw/default_style", "ACS Document 1996")
    dlg = win.list_nicknames()
    from PySide6.QtWidgets import QTableWidget
    table = dlg.findChild(QTableWidget)
    names = {table.item(r, 0).text() for r in range(table.rowCount())}
    assert {"Ph", "Boc", "OMe"} <= names
    w.confirm_close = False
    w.close()


# --- Object menu -----------------------------------------------------------------------------

def test_join_atoms_and_bonds(win):
    d = win.doc
    a1 = d.add_atom(100, 100)
    a2 = d.add_atom(114, 100)
    d.add_bond(a1.id, a2.id)
    b1 = d.add_atom(200, 150, "O")
    b2 = d.add_atom(214, 150)
    d.add_bond(b1.id, b2.id)
    win.canvas.rebuild()
    win.canvas.select_keys({("atom", a2.id), ("atom", b1.id)})
    assert win.join_selection()
    assert convert.to_smiles(d) == "COC" and len(d.fragments()) == 1   # the O label wins
    # two rings sharing a bond after Join (naphthalene skeleton from two benzenes)
    win.canvas.set_state(Document(d.style).to_dict())
    convert.insert_text(win.doc, "C1CCCCC1", 100, 100)
    convert.insert_text(win.doc, "C1CCCCC1", 300, 100)
    win.canvas.rebuild()
    f1, f2 = win.doc.fragments()
    bonds = [next(b.id for b in win.doc.bonds.values() if b.a in f and b.b in f)
             for f in (f1, f2)]
    win.canvas.select_keys({("bond", bonds[0]), ("bond", bonds[1])})
    assert win.join_selection()
    assert convert.to_smiles(win.doc) == "C1CCC2CCCCC2C1"
    win.canvas.select_keys(set())
    assert not win.join_selection()


def test_repeat_and_rotate_180_3d_keeps_stereo(win):
    _load(win, "C[C@H](N)C(=O)O")
    key0 = convert.to_inchikey(win.doc)
    flip_key = None
    win.canvas.select_all()
    _find(_menu(win, "Object"), "Flip Horizontal").trigger()
    flip_key = convert.to_inchikey(win.doc)
    assert flip_key != key0                                   # a mirror image
    assert win.act_repeat.isEnabled() and "Flip Horizontal" in win.act_repeat.text()
    win.act_repeat.trigger()                                  # flips back
    assert convert.to_inchikey(win.doc) == key0
    win.rotate_180_3d(True)
    assert convert.to_inchikey(win.doc) == key0               # 3D turn keeps R/S
    win.rotate_dialog(30)
    assert win.act_repeat.text() == "Repeat Rotate 30°"
    xs = sorted(round(a.x, 3) for a in win.doc.atoms.values())
    win.act_repeat.trigger()
    assert sorted(round(a.x, 3) for a in win.doc.atoms.values()) != xs
    win.scale_dialog(200)
    assert win.act_repeat.text() == "Repeat Scale 200%"


def test_object_dialog_menu_items_ask_first(win, monkeypatch):
    """Rotate... / Scale... / Document Annotations... from the menu open their dialogs (the
    triggered(bool) argument must not be taken for a value)."""
    from PySide6.QtWidgets import QInputDialog
    asked = []
    monkeypatch.setattr(QInputDialog, "getDouble",
                        lambda *a, **k: (asked.append(a[1]), (45.0, True))[1])
    _load(win, "CCO")
    win.canvas.select_all()
    _find(_menu(win, "Object"), "Rotate...").trigger()
    _find(_menu(win, "Object"), "Scale...").trigger()
    assert asked == ["Rotate Objects", "Scale Objects"]
    assert win.act_repeat.text() == "Repeat Scale 45%"
    from rsstatchem.ui.chemdraw import window as wmod
    monkeypatch.setattr(wmod._KeyValueDialog, "exec", lambda self: 0)
    _find(_menu(win, "File"), "Document Annotations...").trigger()
    assert "annotations" not in win.doc.meta


# --- reactions ------------------------------------------------------------------------------

def test_clean_up_reaction_and_interpretation(win, tmp_path):
    _load(win, "CC(=O)O.OCC>>CC(=O)OCC.O")
    first = win.doc.fragments()[0]
    win.canvas.edit("mess", lambda: win.doc.move({("atom", a) for a in first}, -40, 50))
    assert win.clean_up_reaction()
    assert convert.reaction_smiles(win.doc) == "CC(=O)O.CCO>>CCOC(C)=O.O"
    win.act_interpret.trigger()
    assert win.canvas.show_interpretation
    roles = [r for r, _ in win.canvas.interpretation_boxes()]
    assert roles.count("reactants") == 2 and roles.count("products") == 2 and "arrow" in roles
    win.act_rulers.trigger()
    win.act_crosshair.trigger()
    assert win.canvas.show_rulers and win.canvas.show_crosshair
    win.canvas.fit()
    img = win.canvas.viewport().grab().toImage()
    assert not img.isNull()
    _shot(win, "reaction_interpretation_rulers.png")
    win.canvas.set_state(Document(win.doc.style).to_dict())
    win._notify = lambda title, text: setattr(win, "last_message", text)
    assert not win.clean_up_reaction()


def test_mapping_menu_greys_out_without_indigo(win, monkeypatch):
    monkeypatch.setattr(mapping, "_STATE", {"indigo": None, "error": "No module named "
                                            "'indigo'"})
    win._update_mapping_action()
    assert not win.act_map.isEnabled()
    assert "Indigo" in win.act_map.toolTip()
    _load(win, "CBr.[OH-]>>CO.[Br-]")
    for a in win.doc.atoms.values():
        a.map_number = 1
    win.canvas.select_keys(set())
    assert win.clear_reaction_map() == len(win.doc.atoms)
    win._notify = lambda title, text: setattr(win, "last_message", text)
    assert win.map_reaction_atoms() == {}
    assert "Indigo" in win.last_message


@pytest.mark.skipif(not mapping.available(), reason="Indigo (epam.indigo) is not installed")
def test_mapping_menu_with_indigo(win):
    win._update_mapping_action()
    assert win.act_map.isEnabled()
    _load(win, "CCBr.[OH-]>>CCO.[Br-]")
    assert win.map_reaction_atoms()
    assert any(a.map_number for a in win.doc.atoms.values())
    win.undo_stack.undo()
    assert not any(a.map_number for a in win.doc.atoms.values())


# --- install hooks ---------------------------------------------------------------------------

def test_install_hooks_called_and_guarded(ctx, monkeypatch, capsys):
    from rsstatchem.ui.chemdraw import window as wmod
    calls = []
    good = types.ModuleType("fake_intel_actions")
    good.install = lambda w: (calls.append(w), w.menus["Structure"].addAction("Fake Item"))
    bad = types.ModuleType("fake_objects_actions")

    def boom(w):
        raise RuntimeError("broken")
    bad.install = boom
    monkeypatch.setitem(sys.modules, "fake_intel_actions", good)
    monkeypatch.setitem(sys.modules, "fake_objects_actions", bad)
    monkeypatch.setattr(wmod, "INSTALL_MODULES", ("fake_intel_actions", "fake_objects_actions",
                                                  "rsstatchem.ui.chemdraw.no_such_module"))
    w = wmod.ChemDrawWindow(ctx)
    assert calls == [w]
    assert "Fake Item" in [a.text() for a in w.menus["Structure"].actions()]
    assert w.installed["fake_intel_actions"] is None
    assert "broken" in w.installed["fake_objects_actions"]
    assert "rsstatchem.ui.chemdraw.no_such_module" not in w.installed
    assert "install failed" in capsys.readouterr().err
    w.confirm_close = False
    w.close()


def test_real_install_modules_load_if_present(win):
    """Whatever the other work packages ship must install without errors."""
    assert all(err is None for err in win.installed.values()), win.installed


def test_structure_errors_do_not_crash_paste(win, monkeypatch):
    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.warning", lambda *a, **k: None)
    assert win.paste_special("inchi", "not an inchi") == []
    assert win.paste_special("helm", "XYZ{") == []
    with pytest.raises(mo.ChemError):
        win._sequence_text(win.doc, "helm")

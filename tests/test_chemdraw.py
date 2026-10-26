"""Chemical Drawing module (ChemDraw-style workspace): window contract, Main Toolbar, hotkeys,
Analysis / Chemical Properties windows, name <-> structure (offline, network mocked), files
(.rsc, CDXML, MOL), clipboard, printing, text styles and colors."""
from __future__ import annotations

import math
import urllib.error
import urllib.request

import pytest

from rsstatchem.core.chem import molecule as mo
from rsstatchem.core.chem_draw import cdxml, convert, names, ops
from rsstatchem.core.chem_draw.model import Document
from rsstatchem.core.chem_draw.properties import chemical_properties, joback
from rsstatchem.core.chem_draw.style import preset

CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"


@pytest.fixture
def ctx(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("RSSTATCHEM_HOME", str(tmp_path))
    from rsstatchem.app import AppContext

    context = AppContext(qapp)
    quits = []
    monkeypatch.setattr(context.app, "quit", lambda: quits.append(True))
    context.quits = quits
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


def _mouse(canvas, kind, pt):
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication
    vp = canvas.viewport()
    pos = canvas.mapFromScene(QPointF(*pt))
    typ = {"press": QMouseEvent.Type.MouseButtonPress, "move": QMouseEvent.Type.MouseMove,
           "release": QMouseEvent.Type.MouseButtonRelease}[kind]
    btn = Qt.MouseButton.NoButton if kind == "move" else Qt.MouseButton.LeftButton
    buttons = Qt.MouseButton.LeftButton if kind in ("press", "move") else \
        Qt.MouseButton.NoButton
    ev = QMouseEvent(typ, QPointF(pos), QPointF(vp.mapToGlobal(pos)), btn, buttons,
                     Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(vp, ev)


def click(canvas, pt):
    _mouse(canvas, "press", pt)
    _mouse(canvas, "release", pt)


def drag(canvas, a, b, steps=6):
    _mouse(canvas, "press", a)
    for k in range(1, steps + 1):
        t = k / steps
        _mouse(canvas, "move", (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    _mouse(canvas, "release", b)


def key(canvas, text):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QApplication
    ev = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_unknown, Qt.KeyboardModifier.NoModifier,
                   text)
    QApplication.sendEvent(canvas, ev)


def _menu(win, title):
    for act in win.menuBar().actions():
        if act.text().replace("&", "") == title:
            return act.menu()
    raise KeyError(title)


def _texts(menu):
    return [a.text().replace("&", "") for a in menu.actions() if not a.isSeparator()]


# --- window contract -------------------------------------------------------------------

def test_open_module_contract_and_close(ctx, qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from rsstatchem.ui.chemdraw.window import ChemDrawWindow
    from rsstatchem.ui.modules import open_module

    w = open_module(ctx, "chem")
    assert isinstance(w, ChemDrawWindow) and w.isVisible() and w.ctx is ctx
    assert w in ctx.windows and ctx.settings.value("last_module") == "chem"
    titles = [a.text().replace("&", "") for a in w.menuBar().actions()]
    assert titles == ["File", "Edit", "View", "Object", "Structure", "Text", "Curves", "Colors",
                      "Window", "Help"]
    assert w.doc.style.name == "ACS Document 1996" and w.doc.style.bond_length == 14.4
    structure = _texts(_menu(w, "Structure"))
    for item in ("Clean Up Structure", "Check Structure", "Expand Label", "Add Atom Numbers",
                 "Convert Name to Structure...", "Convert Structure to Name"):
        assert item in structure
    view = _texts(_menu(w, "View"))
    for item in ("Show Main Toolbar", "Show Style Toolbar", "Show General Toolbar",
                 "Show Analysis Window", "Show Chemical Properties Window"):
        assert item in view
    obj = _texts(_menu(w, "Object"))
    for item in ("Flip Horizontal", "Flip Vertical", "Group", "Bring to Front", "Align"):
        assert item in obj
    # Help > Check for Updates... / About go to the app context
    calls = []
    monkeypatch.setattr(ctx, "check_for_updates", lambda manual=False: calls.append(manual))
    monkeypatch.setattr(ctx, "about", lambda: calls.append("about"))
    help_menu = _menu(w, "Help")
    next(a for a in help_menu.actions() if a.text() == "Check for &Updates...").trigger()
    next(a for a in help_menu.actions() if "About" in a.text()).trigger()
    assert calls == [True, "about"]
    # Window > Modules submenu
    w._fill_window_menu()
    assert "Modules" in _texts(w.windows_menu)
    # an untouched document closes without asking; ctx.window_closed is called
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: pytest.fail("must not ask to save"))
    w.close()
    assert w not in ctx.windows and ctx.quits == [True]


def test_close_asks_to_save_when_modified(win, ctx, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    win.load_text("CCO")
    win.canvas.edit("Move", lambda: win.doc.move(set(win.doc.keys()), 5, 0))
    asked = []

    def answer(*a, **k):
        asked.append(True)
        return QMessageBox.StandardButton.Cancel
    monkeypatch.setattr(QMessageBox, "question", answer)
    win.close()
    assert asked and win.isVisible() and win in ctx.windows
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.Discard)
    win.close()
    assert win not in ctx.windows


# --- Main Toolbar and drawing ------------------------------------------------------------

def test_main_toolbar_tools_and_flyouts(win):
    from rsstatchem.ui.chemdraw.palette import PALETTE
    pal, cv = win.palette, win.canvas
    assert pal.orientation().name == "Vertical"
    for k, slot in enumerate(PALETTE):
        tid = slot[0][0]
        b = pal.buttons[k]
        if not b.isCheckable():
            continue
        b.click()
        assert win.current_tool == tid and b.isChecked()
    # bond fly-out: picking Triple puts it on the button and selects it
    k = pal.slot_of("bond:double")
    assert pal.buttons[k].menu() is not None
    pal.pick(k, "bond:triple")
    assert cv.tool == "bond:triple" and pal.current[k] == "bond:triple"
    win.set_tool("bracket:round")
    assert cv.tool == "bracket" and cv.bracket_kind == "round"
    # draw: click empty page with the solid bond tool, then a chair
    win.set_tool("bond:single")
    click(cv, (200, 200))
    assert len(win.doc.bonds) == 1
    b = next(iter(win.doc.bonds.values()))
    a1, a2 = win.doc.atoms[b.a], win.doc.atoms[b.b]
    assert math.hypot(a1.x - a2.x, a1.y - a2.y) == pytest.approx(14.4)
    win.set_tool("ring:chair1")
    click(cv, (320, 320))
    assert convert.to_smiles(win.doc, win.doc.fragments()[-1]) == "C1CCCCC1"
    win.set_tool("text")                        # ChemDraw's text tool labels atoms
    click(cv, (a1.x, a1.y))
    assert cv._editor is not None
    cv._editor.setText("OH")
    cv._finish_editor()
    assert convert.to_smiles(win.doc, win.doc.fragments()[0]) == "CO"


def test_drag_bond_fixed_angles_and_lengths(win):
    cv, doc = win.canvas, win.doc
    win.set_tool("bond:single")
    drag(cv, (200, 300), (240, 311))             # ~15.4 deg -> snapped to 15
    b = next(iter(doc.bonds.values()))
    p, q = doc.atoms[b.a], doc.atoms[b.b]
    ang = math.degrees(math.atan2(q.y - p.y, q.x - p.x))
    assert ang == pytest.approx(15, abs=1e-6)
    assert math.hypot(q.x - p.x, q.y - p.y) == pytest.approx(14.4)
    # click the end atom: the next bond keeps ChemDraw's 120 deg zig-zag
    click(cv, (q.x, q.y))
    assert len(doc.bonds) == 2
    win.act_fixed_lengths.setChecked(False)
    win.act_fixed_angles.setChecked(False)
    win._toggle_fixed()
    drag(cv, (100, 500), (100 + 40, 500 + 7))
    nb = max(doc.bonds.values(), key=lambda x: x.id)
    p, q = doc.atoms[nb.a], doc.atoms[nb.b]
    assert math.hypot(q.x - p.x, q.y - p.y) == pytest.approx(math.hypot(40, 7), abs=0.5)


def test_element_number_and_bond_hotkeys(win):
    cv, doc = win.canvas, win.doc
    ids = ops.add_chain(doc, (200.0, 200.0), 0.0, 3)
    cv.rebuild()
    cv.hover = ("atom", ids[-1])
    key(cv, "n")
    assert doc.atoms[ids[-1]].symbol == "N"
    key(cv, "l")
    assert doc.atoms[ids[-1]].symbol == "Cl"
    cv.hover = ("atom", ids[0])
    key(cv, "o")
    assert convert.to_smiles(doc) == "OCCCl"
    bond = doc.bond_between(ids[1], ids[2])
    cv.hover = ("bond", bond.id)
    key(cv, "2")
    assert bond.order == 2
    key(cv, "c")
    assert bond.double_pos == "center"
    key(cv, "b")
    assert bond.order == 1 and bond.style == "bold"
    n = len(doc.atoms)
    cv.hover = ("atom", ids[1])
    key(cv, "6")                                # ring hotkey: cyclohexane on the atom
    assert len(doc.atoms) == n + 5
    win.undo_stack.undo()
    assert len(doc.atoms) == n


# --- Analysis and Chemical Properties windows ----------------------------------------------

def test_analysis_window_caffeine_and_paste(win):
    win.load_text(CAFFEINE)
    win.analysis.refresh()
    v = win.analysis.values()
    assert v["Chemical Formula"] == "C8H10N4O2"
    assert v["Exact Mass"] == "194.08"
    assert v["Molecular Weight"] == "194.19"
    assert v["m/z"].startswith("194.08 (100.0%)")
    assert v["Elemental Analysis"] == "C, 49.48; H, 5.19; N, 28.85; O, 16.48"
    win.analysis.decimals.setValue(4)
    assert win.analysis.values()["Exact Mass"] == "194.0804"
    win.analysis.checks["m/z"].setChecked(False)
    n = len(win.doc.texts)
    win.analysis.paste_btn.click()
    assert len(win.doc.texts) == n + 1
    caption = list(win.doc.texts.values())[-1].text
    assert "Chemical Formula: C8H10N4O2" in caption and "m/z" not in caption
    win.properties.refresh()
    props = win.properties.values()
    # experimental values first (open databases, sources named), else labelled estimates
    assert props["Melting Point"].startswith("510.60 K (237.5 °C) [experimental: Bradley")
    assert props["LogP"].startswith("-0.07 [experimental:")
    assert "[ML: ±" in props["Boiling Point"]                 # caffeine sublimes: no exp. b.p.
    assert "[Abdulelah-Gani]" in props["Critical Temp."]      # Marrero-Gani lacks a group
    assert props["tPSA"] == "61.82 Å² [Ertl]"
    assert "Experimental values" in win.properties.note.text()
    assert "not BioByte" in win.properties.note.text()


def test_chemical_properties_estimates():
    from rdkit import Chem
    acetone = joback(Chem.MolFromSmiles("CC(C)=O"))
    assert acetone.groups == {"-CH3": 2, ">C=O": 1}
    assert acetone.tb == pytest.approx(322.11) and acetone.tc == pytest.approx(500.56, abs=0.01)
    assert not joback(Chem.MolFromSmiles(CAFFEINE)).ok
    assert joback(Chem.MolFromSmiles("CCO")).tb == pytest.approx(337.54)
    rows = {p.name: p for p in chemical_properties(Chem.MolFromSmiles("CCO"))}
    # ethanol is in the experimental database (b.p. 78.2 degC); Joback stays in the details
    assert rows["Boiling Point"].kind == "experimental"
    assert rows["Boiling Point"].value == pytest.approx(351.4, abs=0.5)
    jb = [a for a in rows["Boiling Point"].alternatives if a.key == "joback"][0]
    assert jb.value == pytest.approx(337.54) and "RMSE" in jb.benchmark


# --- names <-> structures --------------------------------------------------------------------

def test_offline_name_lookup_both_ways(win):
    assert win.convert_name_to_structure("Caffeine")
    assert convert.to_inchikey(win.doc) == mo.to_inchikey(mo.mol_from_text(CAFFEINE))
    win.canvas.select_keys(set())
    found = win.convert_structure_to_name()
    assert found == ["1,3,7-trimethyl-3,7-dihydro-1H-purine-2,6-dione"]
    assert any(t.text == found[0] for t in win.doc.texts.values())
    # a selected caption is replaced by its structure
    t = win.doc.add_text(300, 600, "acetaminophen")
    win.canvas.rebuild()
    win.canvas.select_keys({("text", t.id)})
    assert win.convert_name_to_structure()
    assert t.id not in win.doc.texts
    assert names.offline_structure("ACETIC ACID").smiles == "CC(=O)O"


def test_name_lookup_offline_network(win, monkeypatch):
    from PySide6.QtCore import QThreadPool
    from PySide6.QtWidgets import QApplication

    def no_network(*a, **k):
        raise urllib.error.URLError("network is unreachable")
    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    win.use_opsin = False          # this test is about the PubChem path (OPSIN: own test)
    win.online_lookup = False
    assert not win.convert_name_to_structure("some unknown compound")
    assert "offline name list" in win.last_message
    win.online_lookup = True
    assert win.convert_name_to_structure("some unknown compound")   # started, not blocking
    QThreadPool.globalInstance().waitForDone(5000)
    for _ in range(5):
        QApplication.processEvents()
    assert "Could not reach PubChem" in win.last_message
    assert win.doc.is_empty()
    # a structure that is not in the offline list gets its name from (fake) PubChem
    win.lookup.fetch = lambda url: (
        b'{"PropertyTable":{"Properties":[{"CID":1,"IUPACName":"decan-1-ol","Title":"x"}]}}')
    win.load_text("CC(F)(Cl)C(Br)CCCC#N")       # in neither offline list nor database
    assert win.convert_structure_to_name() == []          # not offline: asked in background
    QThreadPool.globalInstance().waitForDone(5000)
    for _ in range(5):
        QApplication.processEvents()
    assert [t.text for t in win.doc.texts.values()] == ["decan-1-ol"]


def test_pubchem_parsing_with_fake_network():
    data = b'{"PropertyTable":{"Properties":[{"CID":2519,"IUPACName":"x-name","Title":"y"}]}}'
    res = names.pubchem_name("RYYVLZVUVIJVGH-UHFFFAOYSA-N", fetch=lambda url: data)
    assert res.name == "x-name" and res.source == "PubChem"
    block = mo.to_molblock(mo.mol_from_text("CCO")).encode() + b"\n$$$$\n"
    res = names.pubchem_structure("ethanol", fetch=lambda url: block)
    assert mo.to_smiles(mo.mol_from_text(res.molblock)) == "CCO"


# --- files, clipboard, print ------------------------------------------------------------------

def test_save_open_round_trip_rsc_cdxml_mol(win, ctx, tmp_path):
    from rsstatchem.ui.modules import open_module
    win.load_text(CAFFEINE)
    smi = convert.to_smiles(win.doc)
    path = str(tmp_path / "caffeine.rsc")
    assert win.save_as(path) and win.undo_stack.isClean()
    assert "caffeine.rsc" in win.windowTitle()
    w2 = open_module(ctx, "chem", path)
    assert convert.to_smiles(w2.doc) == smi and w2.path == path
    assert any(p.endswith("caffeine.rsc") for p in ctx.recent_files())
    cpath = str(tmp_path / "caffeine.cdxml")
    assert w2.save_as(cpath)
    w3 = open_module(ctx, "chem", cpath)
    assert convert.to_smiles(w3.doc) == smi
    mpath = tmp_path / "eth.mol"
    mpath.write_text(mo.to_molblock(mo.mol_from_text("CCO")))
    w4 = open_module(ctx, "chem", str(mpath))
    assert convert.to_smiles(w4.doc) == "CCO" and w4.undo_stack.isClean()
    page = w4.canvas.page
    bb = w4.doc.bbox()
    assert page.left() <= bb[0] and bb[2] <= page.right()
    for w in (w2, w3, w4):
        w.close()


def test_cdxml_round_trip_core():
    d = Document(preset("ACS Document 1996"))
    convert.insert_text(d, "CC(=O)O.OCC>>CC(=O)OCC.O", 200, 200)
    t = d.add_text(10, 10, "conditions")
    t.bold = True
    d.atoms[next(iter(d.atoms))].color = "#E00000"
    xml = cdxml.to_cdxml(d)
    assert "<CDXML" in xml and 'Element="8"' in xml
    d2 = cdxml.from_cdxml(xml)
    assert convert.reaction_smiles(d2) == convert.reaction_smiles(d)
    assert [x.text for x in d2.texts.values()] == ["conditions"]
    assert list(d2.texts.values())[0].bold
    assert any(a.color == "#E00000" for a in d2.atoms.values())
    with pytest.raises(ValueError):
        cdxml.from_cdxml("<html/>")


def test_copy_puts_picture_and_text_on_clipboard(win):
    from PySide6.QtGui import QGuiApplication
    win.load_text("CC(=O)Oc1ccccc1C(=O)O")
    win.canvas.select_all()
    win.copy()
    md = QGuiApplication.clipboard().mimeData()
    assert md.hasImage() and md.text() == "CC(=O)Oc1ccccc1C(=O)O"
    assert md.hasFormat("chemical/x-mdl-molfile")


def test_print_to_pdf(win, tmp_path):
    from PySide6.QtPrintSupport import QPrinter
    win.load_text(CAFFEINE)
    printer = QPrinter(QPrinter.PrinterMode.HighResolution)
    printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
    out = tmp_path / "print.pdf"
    printer.setOutputFileName(str(out))
    assert win.print_document(printer)
    assert out.read_bytes()[:4] == b"%PDF"


# --- text, colors, structure tools ---------------------------------------------------------

def test_text_style_colors_and_atom_numbers(win):
    doc, cv = win.doc, win.canvas
    t = doc.add_text(100, 100, "H2SO4")
    cv.rebuild()
    cv.select_keys({("text", t.id)})
    assert win.apply_text_style(bold=True, italic=True, font="Times New Roman", size=14.0)
    assert (t.bold, t.italic, t.font, t.size) == (True, True, "Times New Roman", 14.0)
    win.style_actions["underline"].trigger()
    assert t.underline
    win.style_actions["superscript"].trigger()
    assert t.text == "<sup>H2SO4</sup>"
    win.apply_color("#E00000")
    assert t.color == "#E00000"
    win.load_text("CCO")
    cv.select_keys(set())
    win.atom_numbers(True)
    assert sorted(a.number for a in doc.atoms.values()) == [1, 2, 3]
    win.atom_numbers(False)
    assert all(a.number == 0 for a in doc.atoms.values())
    calc = next(a.menu() for a in _menu(win, "Structure").actions()
                if a.menu() is not None and "Calculators" in a.text())
    assert "Formula && Molar Mass..." in [a.text() for a in calc.actions()]


def test_zoom_percent_in_status_bar(win):
    cv = win.canvas
    assert cv.zoom_percent() == pytest.approx(100)
    win.act_zoom_in.trigger()
    assert cv.zoom_percent() == pytest.approx(125)
    assert win.zoom_box.currentText() == "125%"
    win.zoom_box.setEditText("200%")
    win._zoom_box_changed()
    assert cv.zoom_percent() == pytest.approx(200)
    win.act_actual.trigger()
    assert cv.zoom_percent() == pytest.approx(100)

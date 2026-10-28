"""Chemical Drawing, work package B: drawing objects and tools (mechanism symbols, the full
arrow palette and anchored curved arrows, drawing elements, TLC and gel plates, clipware,
polymer brackets, Autonumber Reaction, stereo indicators), the ChemDraw 17.1 hotkeys (one
test per key of the User Guide tables, pp. 54-59), the twelve style sheets (pp. 48-53),
drawn templates (Newman, Fischer, Haworth...), .rsc / CDXML round trips and
``objects_actions.install``."""
from __future__ import annotations

import math
import re

import pytest

from rsstatchem.core.chem_draw import cdxml, convert, ops, templates
from rsstatchem.core.chem_draw import geometry as geo
from rsstatchem.core.chem_draw import objects as obj
from rsstatchem.core.chem_draw.clipart import CLIPART, clipart_primitives, default_size
from rsstatchem.core.chem_draw.model import (ARROW_KINDS, KINDS, SHAPE_KINDS, SYMBOL_KINDS,
                                             Arrow, Document)
from rsstatchem.core.chem_draw.style import CM, PRESETS, autonumber_text, preset


def canon(smi: str) -> str:
    from rdkit import Chem
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi))


def acs() -> Document:
    return Document(preset("ACS Document 1996"))


# --- Qt helpers ---------------------------------------------------------------------------

@pytest.fixture
def ctx(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("RSSTATCHEM_HOME", str(tmp_path))
    from rsstatchem.app import AppContext
    context = AppContext(qapp)
    monkeypatch.setattr(context.app, "quit", lambda: None)
    return context


@pytest.fixture
def win(ctx, qapp):
    from rsstatchem.ui.chemdraw import objects_actions
    from rsstatchem.ui.modules import open_module
    w = open_module(ctx, "chem")
    w.resize(1200, 820)
    objects_actions.install(w)
    qapp.processEvents()
    yield w
    w.confirm_close = False
    w.close()


def _mouse(canvas, kind, pt, mods=None):
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
                     mods or Qt.KeyboardModifier.NoModifier)
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


def key(canvas, text, qtkey=None, mods=None):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QApplication
    ev = QKeyEvent(QEvent.Type.KeyPress, qtkey or Qt.Key.Key_unknown,
                   mods or Qt.KeyboardModifier.NoModifier, text)
    QApplication.sendEvent(canvas, ev)


# --- style sheets (17.1 pp. 48-53) ----------------------------------------------------------

SHEETS = {   # name: (length, bold, line, margin, hash, spacing %, label pt, caption pt)
    "ACS Document 1996": (14.4, 2, 0.6, 1.6, 2.5, 18, 10, 10),
    "Adv. Synth. Catal.": (17, 2, 1, 1.6, 2.5, 18, 10, 12),
    "J. Chin. Chem. Soc.": (18, 2.5, 1, 2, 2.5, 20, 12, 12),
    "J. Mol. Mod. (1 column)": (14.4, 2, 0.6, 1.6, 2.5, 18, 10, 10),
    "New Document": (30, 2, 1, 2, 2.7, 12, 10, 12),
    "New Slide": (30, 4, 1.6, 2, 2.7, 15, 12, 16),
    "Phytomedicine": (20, 1.33, 1, 1.25, 3, 8, 12, 12),
    "RSC": (0.43 * CM, 0.056 * CM, 0.016 * CM, 0.044 * CM, 0.062 * CM, 20, 9, 9),
    "Science of Synthesis": (17, 2, 0.8, 2.25, 2.5, 18, 10, 10),
    "Synthesis, Synlett": (17, 2, 0.8, 1.3, 2.5, 18, 10, 10),
    "Verlag Helvetica Chimica Acta": (17, 2.9, 0.54, 2, 2, 14, 10, 12),
    "Wiley Document": (17, 2.6, 0.75, 2, 2.6, 18, 12, 12),
}


@pytest.mark.parametrize("name", sorted(SHEETS))
def test_style_sheet_values(name):
    st = preset(name)
    got = (st.bond_length, st.bold_width, st.line_width, st.margin, st.hash_spacing,
           st.bond_spacing, st.font_size, st.caption_size)
    assert got == pytest.approx(SHEETS[name])
    assert st.chain_angle == 120


def test_style_sheet_list_and_extras():
    assert set(PRESETS) == set(SHEETS)
    assert preset("RSC").bond_length == pytest.approx(12.19, abs=0.01)    # 0.43 cm
    assert preset("J. Mol. Mod. (1 column)").font_family == "Times New Roman"
    assert autonumber_text(4) == "4" and autonumber_text(4, "roman") == "IV"
    assert autonumber_text(28, "alpha") == "ab"
    st = preset("RSC").updated(show_stereo=True)
    assert st.to_dict()["show_stereo"] and not preset("RSC").show_stereo


# --- model ---------------------------------------------------------------------------------

def _all_objects() -> Document:
    d = acs()
    convert.insert_text(d, "CC(=O)O", 0, 0)
    a = next(x for x in d.atoms.values() if x.symbol == "O")
    lp = obj.place_symbol(d, a.id, "lonepair")
    b = next(iter(d.bonds.values()))
    ar = d.add_arrow(lp.x, lp.y, 0, 0, "curved", bend=-6)
    d.anchor(ar, "tail", f"symbol:{lp.id}")
    d.anchor(ar, "head", f"bond:{b.id}")
    for k in SHAPE_KINDS:
        d.add_shape(0, 100, 30, 120, k, shaded=k == "ellipse")
    d.add_symbol(200, 0, "deltaplus")
    t = d.add_tlc(100, 0, 160, 70)
    t.lanes[0].append({"rf": 0.75, "style": "shaded", "show_rf": True})
    g = d.add_gel(200, 50, 260, 140)
    obj.add_gel_band(g, 0, 50, show=True)
    d.add_clipart(0, 200, "erlenmeyer", w=40, h=50, fill=0.5)
    d.add_bracket(-10, -10, 40, 40, "round", "n", polymer="SRU", repeat="ht", sides="left")
    a.charge, a.charge_circle, a.attachment = 1, True, 1
    return d


def test_kinds_and_rsc_round_trip(tmp_path):
    assert {"shape", "symbol", "tlc", "gel", "clipart"} <= set(KINDS)
    d = _all_objects()
    path = tmp_path / "x.rsc"
    from rsstatchem.core.chem_draw.model import load_rsc, save_rsc
    save_rsc(d, str(path))
    assert load_rsc(str(path)).to_dict() == d.to_dict()


def test_symbols_follow_atoms_and_anchored_arrows_follow_objects():
    d = acs()
    a = d.add_atom(0, 0, "O")
    c = d.add_atom(30, 0)
    b = d.add_bond(a.id, c.id)
    lp = obj.place_symbol(d, a.id, "lonepair")
    assert lp.atom == a.id and math.hypot(lp.x, lp.y) > 3
    ar = d.add_arrow(lp.x, lp.y, 15, 0, "curved", bend=-5)
    d.anchor(ar, "tail", f"symbol:{lp.id}")
    d.anchor(ar, "head", f"bond:{b.id}")
    old = (lp.x, lp.y)
    d.move(d.with_bonds({("atom", a.id), ("atom", c.id)}), 10, 5)
    assert (lp.x, lp.y) == pytest.approx((old[0] + 10, old[1] + 5))
    assert (ar.x1, ar.y1) == pytest.approx((lp.x, lp.y))
    assert (ar.x2, ar.y2) == pytest.approx((25, 5))
    d.move([("atom", c.id)], 10, 0)                      # the bond mid-point moves 5
    assert ar.x2 == pytest.approx(30)
    d.move([("arrow", ar.id)], 0, 50)                    # moved alone: it comes loose
    assert not ar.tail_ref and not ar.head_ref
    # copy / paste keeps the association and re-points anchors
    ar2 = d.add_arrow(lp.x, lp.y, 30, 5, "curved")
    d.anchor(ar2, "tail", f"symbol:{lp.id}")
    keys = d.with_bonds({("atom", a.id), ("atom", c.id), ("arrow", ar2.id)})
    assert ("symbol", lp.id) in keys
    new = d.paste(d.subset(keys), 100, 0)
    (nsym,) = [d.symbols[i] for k, i in new if k == "symbol"]
    (nar,) = [d.arrows[i] for k, i in new if k == "arrow"]
    assert nsym.atom in d.atoms and nsym.atom != a.id
    assert nar.tail_ref == f"symbol:{nsym.id}"
    d.remove([("atom", a.id)])                           # symbols go with their atom
    assert lp.id not in d.symbols and not ar2.tail_ref


def test_geometry_of_new_arrows_and_bonds():
    st = preset("ACS Document 1996")
    for k in ARROW_KINDS:
        ar = Arrow(1, 0, 0, 50, 0, k, bend=-8)
        assert geo.arrow_primitives(ar, st), k
    nogo = geo.arrow_primitives(Arrow(1, 0, 0, 50, 0, "nogo"), st)
    assert sum(1 for p in nogo if p[0] == "line") == 3          # shaft + cross
    dashed = geo.arrow_primitives(Arrow(1, 0, 0, 50, 0, "forward", dashed=True), st)
    assert dashed[0][6] is True
    block = geo.arrow_primitives(Arrow(1, 0, 0, 50, 0, "block", fill="filled"), st)
    assert block[0][4] == "solid" and len(block[0][1]) == 7
    for sweep in (90, 180, 270):
        pts = geo.arc_points(0, 0, 40, 0, sweep, 1)
        assert pts[0] == pytest.approx((0, 0), abs=1e-6) and pts[-1] == (40, 0)
        mid = pts[len(pts) // 2]
        r = 40 / (2 * math.sin(math.radians(sweep) / 2))
        assert mid[1] > 0 and abs(mid[1]) <= 2 * r + 1e-6
    d = acs()
    a, b = d.add_atom(0, 0), d.add_atom(14.4, 0)
    q = d.add_bond(a.id, b.id, 4)
    assert len(geo.bond_primitives(d, q, st)) == 4
    q.order, q.style = 2, "bold"
    prims = geo.bond_primitives(d, q, st)
    assert [p[0] for p in prims].count("poly") == 1 and len(prims) == 2
    assert convert.to_smiles(d) == "C=C"


def test_shapes_symbols_and_clipart_primitives():
    st = preset("ACS Document 1996")
    d = acs()
    for k in SHAPE_KINDS:
        s = d.add_shape(0, 0, 40, 20, k)
        prims = obj.shape_primitives(s, st)
        assert prims and prims[-1][0] == "shape"
    assert not obj.shape_primitives(d.add_shape(0, 0, 40, 40, "arc", sweep=90), st)[0][5]
    for k in SYMBOL_KINDS:
        assert obj.symbol_primitives(d.add_symbol(0, 0, k), st), k
    assert obj.symbol_primitives(d.add_symbol(0, 0, "ddagger"), st)[0][3] == "‡"
    for name in CLIPART:
        w, h = default_size(name, st.bond_length)
        c = d.add_clipart(0, 0, name, w=w, h=h, fill=0.5)
        prims = clipart_primitives(c, st.line_width)
        xs = [x for p in prims for x, _ in p[1]]
        assert prims and min(xs) >= -1e-6 and max(xs) <= w + 1e-6, name


# --- TLC and gel ---------------------------------------------------------------------------

def test_tlc_rf_layout_and_editing():
    d = acs()
    t = d.add_tlc(0, 0, 60, 100, lanes=2)
    assert len(t.lanes) == 2 and t.lanes[0][0]["rf"] == 0.5
    lay = obj.tlc_layout(t, 14.4)
    assert lay["origin"] == pytest.approx(88) and lay["front"] == pytest.approx(10)
    assert obj.tlc_rf_at(t, 49) == pytest.approx(0.5)
    k, j = obj.add_tlc_spot(t, 45, lay["origin"] - 0.3 * 78, 14.4)
    assert (k, j) == (1, 1) and t.lanes[1][1]["rf"] == pytest.approx(0.3)
    assert obj.tlc_spot_at(t, 45, lay["origin"] - 0.3 * 78, 14.4) == (1, 1)
    k, j = obj.move_tlc_spot(t, 1, 1, 15, lay["origin"] - 0.8 * 78, 14.4)
    assert (k, j) == (0, 1) and t.lanes[0][1]["rf"] == pytest.approx(0.8)
    assert any(p[0] == "text" for p in obj.tlc_primitives(t, d.style)) is False
    t.show_rf = True
    texts = [p[3] for p in obj.tlc_primitives(t, d.style) if p[0] == "text"]
    assert "0.80" in texts and "0.50" in texts
    lanes = len(d.add_tlc(0, 0, 14.4 * 1.2 * 4, 60).lanes)
    assert lanes == 4                                      # width sets the lane count


def test_gel_log_scale_and_bands():
    d = acs()
    g = d.add_gel(0, 0, 60, 100, lanes=3, lo=10, hi=1000)
    lay = obj.gel_layout(g, 14.4)
    assert obj.gel_mass_at(g, lay["top"]) == pytest.approx(1000)
    assert obj.gel_mass_at(g, lay["top"] + lay["span"]) == pytest.approx(10)
    assert obj.gel_mass_at(g, lay["top"] + lay["span"] / 2) == pytest.approx(100)
    obj.add_gel_band(g, 1, 100)
    obj.add_gel_band(g, 1, 5000)                           # out of range: hidden
    bands = obj.gel_layout(g, 14.4)["bands"]
    assert len(bands) == 1 and bands[0][3] == pytest.approx(lay["top"] + lay["span"] / 2)
    assert obj.gel_band_at(g, bands[0][2], bands[0][3], 14.4) == (1, 0)
    texts = [p[3] for p in obj.gel_primitives(g, d.style) if p[0] == "text"]
    assert {"10", "100", "1000", "kDa", "1", "2", "3"} <= set(texts)


# --- Autonumber Reaction and Ctrl+arrow -------------------------------------------------------

def test_autonumber_reaction_and_clear():
    d = acs()
    for x in (0, 120, 260):
        convert.insert_text(d, "CCO", x, 0)
    ids = obj.autonumber(d)
    assert [d.texts[i].text for i in ids] == ["1", "2", "3"]
    assert all(d.texts[i].bold and d.texts[i].autonumber for i in ids)
    left = min(d.fragments()[0], key=lambda a: d.atoms[a].x)
    assert d.texts[ids[0]].y > max(d.atoms[a].y for a in d.fragments()[0])
    assert d.texts[ids[0]].x < d.atoms[left].x + 40
    d.style.autonumber_style = "roman"
    ids = obj.autonumber(d)                              # replaces the old numbers
    assert sorted(t.text for t in d.texts.values()) == ["I", "II", "III"]
    assert obj.clear_autonumbers(d) == 3 and not d.texts


def test_duplicate_across_arrow():
    d = acs()
    keys = convert.insert_text(d, "c1ccccc1", 0, 0)
    new = obj.duplicate_across_arrow(d, keys, "right")
    assert new[0][0] == "arrow" and len(d.fragments()) == 2
    ar = d.arrows[new[0][1]]
    bb1, bb2 = d.bbox(d.with_bonds(keys)), d.bbox([k for k in new if k[0] != "arrow"])
    assert bb1[2] < ar.x1 < ar.x2 < bb2[0]
    assert convert.reaction_smiles(d) == "c1ccccc1>>c1ccccc1"


# --- ChemDraw 17.1 hotkeys: one test per key ---------------------------------------------------

ATOM_LABELS = [("f", "F"), ("p", "P"), ("P", "Ph"), ("A", "Ac"), ("h", "H"), ("b", "Br"),
               ("B", "B"), ("i", "I"), ("r", "R"), ("k", "K"), ("s", "S"), ("S", "Si"),
               ("m", "Me"), ("n", "N"), ("w", "N"), ("N", "Na"), ("c", "C"), ("l", "Cl"),
               ("C", "Cl"), ("x", "X"), ("o", "O"), ("q", "O"), ("d", "D"), ("e", "Et"),
               ("E", "CO2Me")]


@pytest.mark.parametrize("text,label", ATOM_LABELS)
def test_atom_label_hotkey(win, text, label):
    cv, doc = win.canvas, win.doc
    ids = ops.add_chain(doc, (200.0, 200.0), 0.0, 2)
    cv.rebuild()
    cv.hover = ("atom", ids[-1])
    key(cv, text)
    a = doc.atoms[ids[-1]]
    assert (a.label or a.symbol) == label


SPROUTS = [  # key, hotspot (primary / secondary), expected SMILES
    ("1", 0, "CCCC"), ("0", 0, "CCCC"), ("2", 0, "CCC(C)=O"), ("2", 1, "CC(C)=O"),
    ("3", 0, "CCc1ccccc1"), ("a", 1, "CC(C)c1ccccc1"), ("4", 0, "CCCC"),
    ("5", 1, "CC(C)C"), ("6", 0, "CCC1CCCCC1"), ("6", 1, "CC1(C)CCCCC1"),
    ("7", 1, "CC1(C)CCCC1"), ("8", 1, "C=C(C)C"), ("9", 0, "CCC(C)C"), ("9", 1, "CC(C)(C)C"),
    ("z", 0, "C#CCCC")]


@pytest.mark.parametrize("text,which,smiles", SPROUTS)
def test_atom_sprout_hotkey(win, text, which, smiles):
    cv, doc = win.canvas, win.doc
    ids = ops.add_chain(doc, (200.0, 200.0), 0.0, 2)
    cv.rebuild()
    cv.hover = ("atom", ids[0] if which == 0 else ids[1])
    key(cv, text)
    assert convert.to_smiles(doc) == canon(smiles)


def test_sprout_details(win):
    cv, doc = win.canvas, win.doc
    ids = ops.add_chain(doc, (200.0, 200.0), 0.0, 2)
    cv.rebuild()
    cv.hover = ("atom", ids[0])
    key(cv, "4")
    assert any(b.style == "wedge" for b in doc.bonds.values())
    cv.hover = ("atom", ids[2])
    key(cv, "5")
    assert any(b.style == "hash" for b in doc.bonds.values())
    # 0 (cyclic mode) turns the zig-zag back, 1 continues it
    d = acs()
    c = ops.add_chain(d, (0.0, 0.0), 0.0, 2)
    lin = ops.sprout(d.copy(), c[-1], "1")
    e = d.copy()
    cyc = ops.sprout(e, c[-1], "0")
    assert lin and cyc and e.atoms[cyc[0]].x < d.atoms[c[-1]].x + 1
    # 2 on an aromatic carbon: acetophenone; z gives a linear alkyne
    d = acs()
    convert.insert_text(d, "c1ccccc1", 0, 0)
    ops.sprout(d, next(iter(d.atoms)), "2")
    assert convert.to_smiles(d) == canon("CC(=O)c1ccccc1")


def test_atom_misc_hotkeys(win):
    from PySide6.QtCore import Qt
    cv, doc = win.canvas, win.doc
    ids = ops.add_chain(doc, (200.0, 200.0), 0.0, 2)
    cv.rebuild()
    a = doc.atoms[ids[-1]]
    cv.hover = ("atom", a.id)
    key(cv, "+")
    assert a.charge == 1
    key(cv, "-")
    key(cv, "-")
    assert a.charge == -1
    key(cv, ".")
    assert a.attachment == 1
    key(cv, "'")
    assert a.number == 1
    key(cv, "o")
    assert a.symbol == "O"
    key(cv, " ")                                    # space removes the label
    assert a.symbol == "C" and not a.label
    key(cv, "o")
    key(cv, "", Qt.Key.Key_Backspace)               # so does Backspace on a hotspot
    assert a.symbol == "C" and a.id in doc.atoms
    key(cv, "", Qt.Key.Key_Return)                  # Enter opens the label editor
    assert cv._editor is not None
    cv._cancel_editor()


BOND_CASES = [("1", (1, "")), ("2", (2, "")), ("3", (3, "")), ("d", (1, "dashed")),
              ("B", (2, "bold")), ("y", (1, "wavy")), ("b", (1, "bold")),
              ("w", (1, "wedge")), ("h", (1, "hash")), ("H", (1, "hashed")),
              ("o", (1, "hollow")), ("D", (1, "dative")), ("A", (1.5, ""))]


@pytest.mark.parametrize("text,expect", BOND_CASES)
def test_bond_type_hotkey(win, text, expect):
    cv, doc = win.canvas, win.doc
    ids = ops.add_chain(doc, (200.0, 200.0), 0.0, 2)
    b = doc.bond_between(ids[0], ids[1])
    if text == "1":
        b.order = 2
    cv.rebuild()
    cv.hover = ("bond", b.id)
    key(cv, text)
    assert (b.order, b.style) == expect


@pytest.mark.parametrize("text,pos", [("l", "left"), ("c", "center"), ("r", "right")])
def test_bond_position_hotkey(win, text, pos):
    cv, doc = win.canvas, win.doc
    ids = ops.add_chain(doc, (200.0, 200.0), 0.0, 2)
    b = doc.bond_between(ids[0], ids[1])
    b.order = 2
    cv.rebuild()
    cv.hover = ("bond", b.id)
    key(cv, text)
    assert b.double_pos == pos


FUSIONS = [("a", "c1ccccc1"), ("z", "C1=CCC=C1"), ("v", "C1CC1"), ("4", "C1CCC1"),
           ("5", "C1CCCC1"), ("6", "C1CCCCC1"), ("7", "C1CCCCCC1"), ("8", "C1CCCCCCC1"),
           ("9", "C1CCCCC1"), ("0", "C1CCCCC1")]


@pytest.mark.parametrize("text,smiles", FUSIONS)
def test_bond_fusion_hotkey(win, text, smiles):
    cv, doc = win.canvas, win.doc
    a, b = doc.add_atom(200, 200), doc.add_atom(214.4, 200)
    bond = doc.add_bond(a.id, b.id)
    cv.rebuild()
    cv.hover = ("bond", bond.id)
    key(cv, text)
    assert convert.to_smiles(doc) == canon(smiles)
    if text in "90":                                  # chairs: not a flat hexagon
        ys = sorted(round(at.y, 1) for at in doc.atoms.values())
        assert len(set(ys)) > 2


def test_bond_front_hotkey(win):
    cv, doc = win.canvas, win.doc
    ids = ops.add_chain(doc, (200.0, 200.0), 0.0, 3)
    first = doc.bond_between(ids[0], ids[1])
    cv.rebuild()
    cv.hover = ("bond", first.id)
    key(cv, "f")
    assert list(doc.bonds)[-1] == first.id


GENERIC = [(" ", "select"), ("x", "bond:single"), ("X", "chain"), ("t", "text"),
           ("T", "bracket"), ("e", "arrow:forward"), ("E", "charge+"),
           ("j", "ring:benzene"), ("J", "ring:cyclopentadiene"), ("u", "ring:cyclohexane"),
           ("U", "ring:cyclopentane"), ("g", "tlc"), ("G", "orbital:p:shaded"), ("l", "pen"),
           ("L", "shape:rect"), ("k", "fragment")]


@pytest.mark.parametrize("text,tool", GENERIC)
def test_generic_tool_hotkey(win, text, tool):
    cv = win.canvas
    cv.hover = None
    key(cv, text)
    assert cv.tool == tool


def test_generic_keys_follow_flyout_and_query(win):
    cv, pal = win.canvas, win.palette
    pal.pick(pal.slot_of("arrow:nogo"), "arrow:nogo")
    win.set_tool("select")
    cv.hover = None
    key(cv, "e")
    assert cv.tool == "arrow:nogo"
    key(cv, "K")                                   # query tools: not available, no change
    assert cv.tool == "arrow:nogo"


def test_ctrl_arrow_duplicates_selection(win):
    from PySide6.QtCore import Qt
    cv, doc = win.canvas, win.doc
    keys = cv.insert_text("CCO", 200, 200)
    cv.select_keys(set(keys))
    key(cv, "", Qt.Key.Key_Right, Qt.KeyboardModifier.ControlModifier)
    assert len(doc.arrows) == 1 and len(doc.fragments()) == 2
    win.undo_stack.undo()
    assert not doc.arrows and len(doc.fragments()) == 1


def test_function_keys_and_check_structure(win):
    from PySide6.QtGui import QKeySequence
    assert win.act_actual.shortcut() == QKeySequence("F5")
    assert win.act_fit.shortcut() == QKeySequence("F6")
    assert win.act_zoom_in.shortcut() == QKeySequence("F7")
    assert win.act_zoom_out.shortcut() == QKeySequence("F8")
    assert QKeySequence("Ctrl+]") in win.act_zoom_in.shortcuts()
    assert win.act_check.shortcut() != QKeySequence("F5")


# --- canvas tools ----------------------------------------------------------------------------

def test_symbol_tool_on_atoms_and_in_space(win):
    cv, doc = win.canvas, win.doc
    win.load_text("CO")
    o = next(a for a in doc.atoms.values() if a.symbol == "O")
    win.set_tool("symbol:lonepair")
    click(cv, (o.x, o.y))
    click(cv, (o.x, o.y))
    lps = doc.symbols_of(o.id)
    assert len(lps) == 2 and abs(math.remainder(lps[0].angle - lps[1].angle, 2 * math.pi)) > 0.5
    win.set_tool("symbol:circleplus")
    click(cv, (o.x, o.y))
    assert o.charge == 1 and o.charge_circle
    win.set_tool("symbol:deltaminus")
    click(cv, (400, 400))
    assert any(s.kind == "deltaminus" and not s.atom for s in doc.symbols.values())
    win.set_tool("eraser")
    click(cv, (o.x, o.y))
    assert not doc.symbols_of(o.id)


def test_curved_arrow_anchors_from_lone_pair_to_bond(win):
    cv, doc = win.canvas, win.doc
    win.load_text("C=CO")
    o = next(a for a in doc.atoms.values() if a.symbol == "O")
    lp = obj.place_symbol(doc, o.id, "lonepair")
    cv.rebuild()
    single = next(b for b in doc.bonds.values() if o.id in (b.a, b.b))
    pa, pb = doc.atoms[single.a], doc.atoms[single.b]
    mid = ((pa.x + pb.x) / 2, (pa.y + pb.y) / 2)
    win.set_tool("arrow:curved")
    drag(cv, (lp.x, lp.y), mid)
    (ar,) = doc.arrows.values()
    assert ar.tail_ref == f"symbol:{lp.id}" and ar.head_ref == f"bond:{single.id}"
    assert ar.bend != 0
    cv.select_keys(cv._fragment_keys(o.id))
    key(cv, "", __import__("PySide6.QtCore").QtCore.Qt.Key.Key_Down)
    assert (ar.x1, ar.y1) == pytest.approx((lp.x, lp.y))


def test_shape_tlc_gel_and_clipart_tools(win):
    cv, doc = win.canvas, win.doc
    win.set_tool("shape:circle")
    drag(cv, (200, 200), (230, 200))
    (s,) = doc.shapes.values()
    assert s.kind == "circle" and s.x2 - s.x1 == pytest.approx(60, abs=1)
    win.set_tool("shape:roundrect")
    click(cv, (400, 100))
    assert len(doc.shapes) == 2
    win.set_tool("tlc")
    drag(cv, (100, 400), (180, 520))
    (t,) = doc.tlcs.values()
    assert len(t.lanes) == round(80 / (doc.style.bond_length * 1.2))
    lay = obj.tlc_layout(t, cv.bl)
    click(cv, (lay["xs"][0], lay["origin"] - 0.4 * (lay["origin"] - lay["front"])))
    assert any(abs(sp["rf"] - 0.4) < 0.03 for sp in t.lanes[0])
    win.set_tool("select")
    cx, cy = lay["xs"][1], lay["origin"] - 0.5 * (lay["origin"] - lay["front"])
    drag(cv, (cx, cy), (cx, lay["origin"] - 0.7 * (lay["origin"] - lay["front"])))
    assert t.lanes[1][0]["rf"] == pytest.approx(0.7, abs=0.03)
    cv.edit_rf(t.id, 1, 0, 0.25)
    assert t.lanes[1][0]["rf"] == 0.25
    win.set_tool("eraser")
    lay = obj.tlc_layout(t, cv.bl)
    n = sum(len(x) for x in t.lanes)
    k, j, sx, sy = lay["spots"][-1][:4]
    click(cv, (sx, sy))
    assert sum(len(x) for x in t.lanes) == n - 1 and t.id in doc.tlcs
    win.set_tool("gel")
    drag(cv, (300, 400), (380, 520))
    (g,) = doc.gels.values()
    gl = obj.gel_layout(g, cv.bl)
    click(cv, (gl["xs"][0], gl["top"] + gl["span"] / 2))
    assert g.lanes[0]["bands"][0]["mass"] == pytest.approx(100, rel=0.1)
    win.set_tool("clipart:beaker")
    click(cv, (500, 300))
    (c,) = doc.cliparts.values()
    assert c.name == "beaker" and c.x + c.w / 2 == pytest.approx(500, abs=2)
    for k in (("shape", s.id), ("tlc", t.id), ("gel", g.id), ("clipart", c.id)):
        assert k in cv.info.object_boxes
    win.set_tool("select")
    drag(cv, (0, 0), (700, 700))                           # marquee selects them
    assert {("tlc", t.id), ("gel", g.id), ("clipart", c.id)} <= cv.selection


def test_brackets_single_and_polymer_properties(win):
    from rsstatchem.ui.chem_editor.dialogs import BracketDialog
    cv, doc = win.canvas, win.doc
    win.set_tool("bracket:square:left")
    drag(cv, (100, 100), (160, 150))
    (br,) = doc.brackets.values()
    assert (br.kind, br.sides) == ("square", "left")
    assert len(geo.bracket_primitives(br.x1, br.y1, br.x2, br.y2, br.kind, doc.style,
                                      br.sides)) == 1
    dlg = BracketDialog(br, cv)
    dlg.sides.setCurrentIndex(dlg.sides.findData("both"))
    dlg.polymer.setCurrentIndex(dlg.polymer.findData("copolymer"))
    dlg.repeat.setCurrentIndex(dlg.repeat.findData("ht"))
    dlg.apply(br)
    assert (br.sides, br.polymer, br.repeat, br.label) == ("both", "copolymer", "ht", "n")


def test_arrow_and_bond_dialogs(win):
    from rsstatchem.ui.chem_editor.dialogs import ArrowDialog, BondDialog
    doc = win.doc
    ar = doc.add_arrow(0, 0, 50, 0, "arc")
    dlg = ArrowDialog(ar, win)
    dlg.dashed.setChecked(True)
    dlg.sweep.setCurrentIndex(dlg.sweep.findData(270.0))
    dlg.apply(ar)
    assert ar.dashed and ar.sweep == 270
    ids = ops.add_chain(doc, (0.0, 0.0), 0.0, 1)
    b = doc.bond_between(*ids)
    dlg = BondDialog(b, win)
    dlg.order.setCurrentIndex(dlg.order.findData(4))
    dlg.apply(b)
    assert b.order == 4


def test_palette_has_new_tools_with_icons(win):
    from rsstatchem.ui.chemdraw.icons import svg_markup
    from rsstatchem.ui.chemdraw.palette import PALETTE
    tools = {t for slot in PALETTE for t, _ in slot}
    for t in ("arrow:nogo", "arrow:unbalanced", "arrow:block", "arrow:double", "arrow:arc:90",
              "arrow:arc:270", "arrow:dipole", "symbol:lonepair", "symbol:deltaplus",
              "symbol:circleplus", "symbol:radicalcation", "shape:circle", "shape:arc:120",
              "tlc", "gel", "clipart:beaker", "bracket:curly:right", "bond:quadruple",
              "bond:bolddouble"):
        assert t in tools
        assert ">?<" not in svg_markup(t)
    assert win.palette.current_of("tlc") == "tlc"


# --- stereo indicators ------------------------------------------------------------------------

def test_stereo_labels_and_toggle(win):
    d = acs()
    convert.insert_text(d, "C[C@H](N)C(=O)O", 0, 0)        # L-alanine
    ops.refresh_stereo(d)
    assert [a.stereo for a in d.atoms.values() if a.stereo] == ["S"]
    d = acs()
    convert.insert_text(d, "C/C=C/C", 0, 0)
    ops.refresh_stereo(d)
    assert [b.stereo for b in d.bonds.values() if b.stereo] == ["E"]
    win.load_text("C[C@@H](N)C(=O)O")
    act = win.act_show_stereo
    act.trigger()
    assert win.doc.style.show_stereo
    assert [a.stereo for a in win.doc.atoms.values() if a.stereo] == ["R"]
    # labels follow edits while the indicators are on
    cv = win.canvas
    n = next(a for a in win.doc.atoms.values() if a.symbol == "N")
    cv.edit("x", lambda: setattr(n, "symbol", "C"))      # no stereocentre any more
    assert [a.stereo for a in win.doc.atoms.values() if a.stereo] == []
    from rsstatchem.ui.chemdraw import objects_actions
    objects_actions._stereo_parens(win, False)
    assert not win.doc.style.stereo_parens
    act.trigger()
    assert not win.doc.style.show_stereo


# --- templates --------------------------------------------------------------------------------

@pytest.mark.parametrize("key_", sorted(templates.DRAWN))
def test_drawn_templates(key_):
    d = templates.build_drawn(f"@tpl:{key_}", preset("ACS Document 1996"))
    assert d.atoms
    if key_.startswith("fischer") or key_.startswith("haworth"):
        convert.to_smiles(d)                               # chemically valid drawings
    if key_.startswith("newman"):
        assert len(d.shapes) == 1 and len(d.arrows) == 3


def test_projection_chemistry():
    d = templates.build_drawn("@tpl:fischer_glucose", preset("ACS Document 1996"))
    assert convert.to_smiles(d) == canon("OCC(O)C(O)C(O)C(O)C=O")
    d = templates.build_drawn("@tpl:haworth_pyranose", preset("ACS Document 1996"))
    assert convert.to_smiles(d) == canon("OCC1OC(O)C(O)C(O)C1O")
    assert sum(1 for b in d.bonds.values() if b.style == "bold") == 3
    d = templates.build_drawn("@tpl:chair_axial", preset("ACS Document 1996"))
    assert len(d.atoms) == 18 and convert.to_smiles(d) == canon(
        "CC1(C)C(C)(C)C(C)(C)C(C)(C)C(C)(C)C1(C)C")


def test_template_categories_and_insertion(win):
    cats = templates.categories()
    for c in ("Conformations & projections", "Clipware (glassware)", "Aromatics",
              "Bicyclics & cages"):
        assert c in cats
    for cat in cats:
        for name, smi in templates.TEMPLATES[cat]:
            if not templates.is_drawn(smi):
                from rdkit import Chem
                assert Chem.MolFromSmiles(smi.replace("*", "[*]")) is not None, name
    win._template_chosen("Newman", "@tpl:newman_staggered")
    click(win.canvas, (300, 300))
    assert len(win.doc.shapes) == 1 and len(win.doc.arrows) == 3
    win.canvas.insert_text("@clip:sepfunnel", 100, 100)
    assert [c.name for c in win.doc.cliparts.values()] == ["sepfunnel"]
    win.templates.category.setCurrentText("Clipware (glassware)")
    assert win.templates.list.count() == len(CLIPART)


# --- CDXML ------------------------------------------------------------------------------------

def test_cdxml_round_trip_of_new_objects(tmp_path):
    d = _all_objects()
    d.add_arrow(0, 300, 50, 300, "nogo", dashed=True)
    d.add_arrow(0, 320, 50, 320, "arc", sweep=270)
    d.add_arrow(0, 340, 50, 340, "double", bend=-8)
    d.style.show_stereo = True
    obj.autonumber(d)
    xml = cdxml.to_cdxml(d)
    for tag in ("<tlcplate", "<tlclane", "<tlcspot", 'GraphicType="Oval"',
                'GraphicType="Rectangle"', 'SymbolType="LonePair"', 'NoGo="Cross"',
                'AngularSize="270"', 'ShowAtomStereo="yes"', "<objecttag"):
        assert tag in xml, tag
    e = cdxml.from_cdxml(xml)
    for k in ("atom", "bond", "arrow", "shape", "symbol", "tlc", "gel", "clipart",
              "bracket", "text"):
        assert len(e.store(k)) == len(d.store(k)), k

    def strip(o):
        x = dict(vars(o))
        x.pop("id")
        return x
    for k in ("shape", "tlc", "gel", "clipart", "bracket"):
        assert sorted(map(str, map(strip, e.store(k).values()))) == \
            sorted(map(str, map(strip, d.store(k).values()))), k
    curved = next(a for a in e.arrows.values() if a.kind == "curved")
    lp = e.get(("symbol", int(curved.tail_ref.split(":")[1])))
    o = e.atoms[lp.atom]
    assert lp.kind == "lonepair" and o.symbol == "O" and o.charge_circle and o.attachment == 1
    assert e.bonds.get(int(curved.head_ref.split(":")[1])) is not None
    assert e.style.show_stereo and any(t.autonumber for t in e.texts.values())
    # without the tags (another program) the standard elements still come back
    plain = cdxml.from_cdxml(re.sub(r"<objecttag [^>]*/>", "", xml))
    assert plain.tlcs and plain.shapes and plain.symbols
    assert len(plain.tlcs[min(plain.tlcs)].lanes) == len(d.tlcs[min(d.tlcs)].lanes)
    # RDKit still reads the chemistry
    from rdkit import Chem
    path = tmp_path / "x.cdxml"
    path.write_text(xml, encoding="utf-8")
    mols = [m for m in Chem.MolsFromCDXMLFile(str(path)) if m is not None]
    assert [Chem.MolToSmiles(m) for m in mols] == [canon("CC(=O)[OH+]")] or \
        Chem.MolToSmiles(mols[0]).count("C") == 2


# --- rendering, menus and screenshots ------------------------------------------------------------

def test_render_all_objects_to_image(win):
    from rsstatchem.ui.chem_editor.export import image
    d = _all_objects()
    d.style.show_stereo = True
    img = image(d, dpi=96, transparent=False)
    assert img.width() > 100 and img.height() > 100
    win.canvas.paste_document(d)
    info = win.canvas.info
    assert len(info.object_boxes) >= len(d.shapes) + len(d.symbols) + 3


def test_install_menus_and_autonumber_action(win):
    from rsstatchem.ui.chemdraw import objects_actions
    objects_actions.install(win)                          # second call: no duplicates
    texts = [a.text().replace("&", "") for act in win.menuBar().actions() if act.menu()
             for a in act.menu().actions()]
    assert texts.count("Show Stereochemistry") == 1
    for t in ("Autonumber Reaction", "Clear Reaction Numbers", "Bond Properties...",
              "Bracket Properties...", "Duplicate Across Arrow"):
        assert t in texts, t
    for x in (0, 150):
        win.canvas.insert_text("CCO", x, 0)
    win.canvas.select_keys(set())
    win.act_autonumber.trigger()
    assert sorted(t.text for t in win.doc.texts.values()) == ["1", "2"]
    assert objects_actions.clear_numbers(win) == 2
    dlg = objects_actions.show_hotkeys(win)
    assert "Sprouting" in dlg.findChild(__import__("PySide6.QtWidgets").QtWidgets
                                         .QTextBrowser).toPlainText()
    presets = [a.text() for act in win.menuBar().actions() if act.menu()
               for a in act.menu().actions() if a.menu() and "Apply Document" in a.text()
               for a2 in [a] for a in a2.menu().actions()]
    assert {"RSC", "Wiley Document", "New Slide"} <= set(presets)

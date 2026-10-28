"""Chemical Drawing, second round (ChemDraw gaps): Pen tool and Curves menu, hollow wedge /
hashed / dative bonds, orbitals, Structure Perspective (3D), group-contribution fallbacks
(Stein-Brown, Marrero-Gani) against literature values, the offline name database, OPSIN
(runs only when Java + the OPSIN jar are present) and the Mass Fragmentation tool."""
from __future__ import annotations

import math

import numpy as np
import pytest
from rdkit import Chem

from rsstatchem.core.chem import molecule as mo
from rsstatchem.core.chem_draw import cdxml, convert, fragment, names, ops, threed
from rsstatchem.core.chem_draw import geometry as geo
from rsstatchem.core.chem_draw.model import Document
from rsstatchem.core.chem_draw.properties import (chemical_properties, marrero_gani,
                                                  stein_brown)
from rsstatchem.core.chem_draw.style import preset

CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
ASPIRIN_KEY = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
HAS_OPSIN = names.opsin_available()


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


def _wait_pool(qapp, ms=60000):
    from PySide6.QtCore import QThreadPool
    QThreadPool.globalInstance().waitForDone(ms)
    for _ in range(10):
        qapp.processEvents()


# --- curves (Pen tool, Curves menu) -----------------------------------------------------------

def test_curve_model_geometry_and_round_trips(tmp_path):
    d = Document(preset("ACS Document 1996"))
    pts = geo.smooth_curve_points([(10, 50), (60, 10), (110, 50)])
    c = d.add_curve(pts, arrow_end="full", bold=True)
    assert len(c.anchors()) == 3 and c.anchors()[1] == [60.0, 10.0]
    path = geo.curve_path_points(c)
    assert path[0] == (10.0, 50.0) and path[-1] == pytest.approx((110.0, 50.0))
    prims = geo.curve_primitives(c, d.style)
    assert [p[0] for p in prims] == ["shape", "poly"]           # shaft + arrowhead
    assert prims[0][2] == d.style.bold_width
    half = d.add_curve(pts, arrow_start="half", arrow_end="half")
    assert [p[0] for p in geo.curve_primitives(half, d.style)].count("poly") == 2
    shaded = d.add_curve(geo.smooth_curve_points([(0, 0), (40, 0), (40, 40)], True),
                         closed=True, shaded=True)
    assert geo.curve_primitives(shaded, d.style)[0][4] == "shade"
    # .rsc keeps everything; move / rotate / flip act on all curve points
    d2 = Document.from_json(d.to_json())
    assert d2.curves[c.id].points == c.points and d2.curves[c.id].bold
    d2.move([("curve", c.id)], 5, 5)
    assert d2.curves[c.id].anchors()[0] == [15.0, 55.0]
    d2.flip([("curve", c.id)])
    # CDXML <curve> with CurvePoints / CurveType / arrowheads
    xml = cdxml.to_cdxml(d)
    assert "<curve" in xml and 'ArrowheadHead="Full"' in xml and "CurvePoints" in xml
    d3 = cdxml.from_cdxml(xml)
    flags = sorted((x.bold, x.arrow_end, x.arrow_start, x.closed, x.shaded)
                   for x in d3.curves.values())
    assert flags == sorted([(True, "full", "", False, False), (False, "half", "half", False,
                                                                 False),
                            (False, "", "", True, True)])
    assert d3.curves[min(d3.curves)].points[1] == pytest.approx([10.0, 50.0])
    # curved electron-pushing arrows are written as two-point curves now
    d4 = Document()
    d4.add_arrow(0, 0, 40, 0, "curved", bend=-12)
    xml4 = cdxml.to_cdxml(d4)
    (ar5,) = cdxml.from_cdxml(xml4).arrows.values()      # RS-StatChem tag: exact arrow back
    assert ar5.kind == "curved" and ar5.bend == pytest.approx(-12)
    import re
    d5 = cdxml.from_cdxml(re.sub(r"<objecttag [^>]*/>", "", xml4))   # as other programs
    (cv,) = d5.curves.values()
    assert cv.arrow_end == "full" and len(cv.anchors()) == 2
    mid = geo.bezier_point(*geo.curve_segments(cv)[0], 0.5)
    assert mid[1] == pytest.approx(geo.arrow_path_points(d4.arrows[1], 2)[1][1], abs=0.01)


def test_curves_render_and_export(qapp, tmp_path):
    from rsstatchem.ui.chem_editor import export
    d = Document(preset("ACS Document 1996"))
    d.add_curve(geo.smooth_curve_points([(0, 0), (30, -20), (60, 0)]), arrow_end="full",
                dashed=True)
    d.add_orbital(100, 0, "p", fill="shaded")
    svg = export.svg_text(d).decode()
    assert "<path" in svg and "stroke-dasharray" in svg and "Gradient" in svg
    export.export_pdf(d, str(tmp_path / "c.pdf"))
    assert (tmp_path / "c.pdf").read_bytes()[:4] == b"%PDF"


def test_pen_tool_draw_close_and_edit_handles(win, qapp):
    cv = win.canvas
    win.set_tool("pen")
    assert cv.tool == "pen" and win.palette.button_for("pen").isChecked()
    click(cv, (200, 200))                        # corner point
    drag(cv, (260, 160), (290, 160))             # smooth point: handles follow the drag
    click(cv, (320, 220))                        # corner point
    click(cv, (320, 220))                        # same point again = double-click: finish
    (c,) = win.doc.curves.values()
    assert len(c.anchors()) == 3 and not c.closed and cv._pen is None
    assert c.points[5] == pytest.approx([290, 160], abs=0.6)       # out handle of point 2
    assert c.points[3] == pytest.approx([230, 160], abs=0.6)       # mirrored in handle
    assert ("curve", c.id) in cv.selection                         # handles shown for editing
    # drag the out handle of the middle point: smooth point stays smooth
    drag(cv, (290, 160), (290, 130))
    c = win.doc.curves[c.id]
    ax, ay = c.anchors()[1]
    v1 = (c.points[3][0] - ax, c.points[3][1] - ay)
    v2 = (c.points[5][0] - ax, c.points[5][1] - ay)
    assert c.points[5] == pytest.approx([290, 130], abs=0.6)
    assert abs(v1[0] * v2[1] - v1[1] * v2[0]) < 1.0                # still collinear
    # drag an anchor: its handles move with it
    drag(cv, (200, 200), (190, 210))
    assert win.doc.curves[c.id].anchors()[0] == pytest.approx([190, 210], abs=0.6)
    win.undo_stack.undo()
    assert win.doc.curves[c.id].anchors()[0] == pytest.approx([200, 200], abs=0.6)
    # a closed curve: click back on the first point
    win.set_tool("pen")
    for pt in ((400, 400), (460, 400), (430, 450), (400, 400)):
        click(cv, pt)
    closed = [x for x in win.doc.curves.values() if x.closed]
    assert len(closed) == 1 and len(closed[0].anchors()) == 3
    # Esc finishes; a single point is not kept
    click(cv, (500, 500))
    key(cv, "")
    cv.finish_curve()
    assert len(win.doc.curves) == 2


def test_curves_menu_styles(win):
    cv = win.canvas
    titles = [a.text().replace("&", "") for a in win.menuBar().actions()]
    assert titles.index("Curves") == titles.index("Text") + 1          # as in ChemDraw
    c = win.doc.add_curve(geo.smooth_curve_points([(100, 100), (150, 60), (200, 100)]))
    cv.rebuild()
    cv.select_keys({("curve", c.id)})
    win._fill_curves_menu()
    acts = {a.text().replace("&", ""): a for a in win.curves_menu.actions() if a.text()}
    for name in ("Plain", "Arrow at Start", "Arrow at End", "Half Arrow at End", "Dashed",
                 "Bold", "Doubled", "Closed", "Filled", "Shaded", "Pen Tool"):
        assert name in acts
    acts["Arrow at End"].trigger()
    assert win.doc.curves[c.id].arrow_end == "full"
    acts["Bold"].trigger()
    assert win.doc.curves[c.id].bold
    win._fill_curves_menu()
    acts = {a.text().replace("&", ""): a for a in win.curves_menu.actions() if a.text()}
    assert acts["Arrow at End"].isChecked() and acts["Bold"].isChecked()
    acts["Filled"].trigger()                    # fill excludes arrowheads (ChemDraw's rule)
    c2 = win.doc.curves[c.id]
    assert c2.filled and c2.closed and not c2.arrow_end
    acts["Plain"].trigger()
    c3 = win.doc.curves[c.id]
    assert not (c3.filled or c3.closed or c3.bold or c3.arrow_end)
    # the last style chosen applies to new Pen curves
    cv.select_keys(set())
    cv.set_curve_style([], arrow_end="half")
    win.set_tool("pen")
    click(cv, (300, 300))
    click(cv, (360, 300))
    cv.finish_curve()
    newest = win.doc.curves[max(win.doc.curves)]
    assert newest.arrow_end == "half"


# --- bonds ------------------------------------------------------------------------------------

def test_hollow_hashed_and_dative_bonds():
    d = Document(preset("ACS Document 1996"))
    a, b, c, e = (d.add_atom(x, 0).id for x in (0, 14.4, 28.8, 43.2))
    hollow = d.add_bond(a, b, 1, "hollow")
    hashed = d.add_bond(b, c, 1, "hashed")
    st = d.style
    p = geo.bond_primitives(d, hollow, st)
    assert p[0][0] == "shape" and p[0][4] == "white" and len(p[0][1]) == 3
    strokes = geo.bond_primitives(d, hashed, st)
    widths = {round(math.hypot(s[3] - s[1], s[4] - s[2]), 3) for s in strokes}
    assert len(strokes) >= 4 and len(widths) == 1          # parallel, equal strokes
    d.atoms[e].symbol = "Pt"
    d.atoms[c].symbol = "N"
    dat = d.add_bond(c, e, 1, "dative")
    kinds = [x[0] for x in geo.bond_primitives(d, dat, st)]
    assert kinds == ["line", "poly"]
    # chemistry: dative -> RDKit dative bond; the donor N gains no valence from it
    m = convert.to_mol(d, [c, e])
    assert m.GetBondWithIdx(0).GetBondType() == Chem.BondType.DATIVE
    assert convert.to_smiles(d, [c, e]) == "[NH3]->[Pt]"
    from rsstatchem.core.chem_draw.labels import implicit_h
    assert implicit_h(d, d.atoms[c]) == 2      # one C-N bond; the dative bond adds nothing
    # hollow wedge = stereo "up" like a wedge
    d2 = Document()
    convert.insert_text(d2, "C[C@H](N)O", 0, 0)
    for bd in d2.bonds.values():
        if bd.style == "wedge":
            bd.style = "hollow"
    assert convert.to_smiles(d2) == "C[C@H](N)O"
    # CDXML keeps the styles (Display / Order="dative")
    xml = cdxml.to_cdxml(d)
    assert 'Display="HollowWedgeBegin"' in xml and 'Display="Hash"' in xml and \
        'Order="dative"' in xml
    styles = sorted(x.style for x in cdxml.from_cdxml(xml).bonds.values())
    assert styles == ["dative", "hashed", "hollow"]
    # tool behaviour: hollow wedge reverses on a second click, like the wedge
    ops.apply_bond_tool(d, hollow.id, "hollow", near=b)
    assert (hollow.a, hollow.b) == (b, a)
    ops.apply_bond_tool(d, hollow.id, "hollow")
    assert (hollow.a, hollow.b) == (a, b)


def test_new_bond_tools_in_palette_and_hotkeys(win):
    pal, cv = win.palette, win.canvas
    for tid in ("bond:hollow", "bond:hashed", "bond:dative"):
        k = pal.slot_of(tid)
        assert k is not None
        pal.pick(k, tid)
        assert cv.tool == tid
    win.load_text("CCO")
    b = next(iter(win.doc.bonds.values()))
    pa, pb = win.doc.atoms[b.a], win.doc.atoms[b.b]
    mid = ((pa.x + pb.x) / 2, (pa.y + pb.y) / 2)
    _mouse(cv, "move", mid)
    key(cv, "o")
    assert win.doc.bonds[b.id].style == "hollow"
    key(cv, "D")                                # ChemDraw 17.1 uses v to fuse a cyclopropane
    assert win.doc.bonds[b.id].style == "dative"


# --- orbitals ---------------------------------------------------------------------------------

def test_orbitals_model_tool_and_cdxml(win):
    cv, doc = win.canvas, win.doc
    win.load_text("C=C")
    a = next(iter(doc.atoms.values()))
    win.set_tool("orbital:p:phase")
    click(cv, (a.x, a.y))                        # click an atom: centred on it
    (o,) = doc.orbitals.values()
    assert (o.x, o.y, o.kind, o.fill) == (a.x, a.y, "p", "phase")
    win.set_tool("orbital:dxy:shaded")
    drag(cv, (300, 300), (330, 300))             # drag: size and direction
    o2 = doc.orbitals[max(doc.orbitals)]
    assert o2.kind == "dxy" and o2.size == pytest.approx(30, abs=1)
    assert o2.angle == pytest.approx(0.0, abs=1e-6)
    lobes = geo.orbital_outline(o2, doc.style.bond_length)
    assert len(lobes) == 4 and {f for _, f in lobes} == {"shade"}
    assert [f for _, f in geo.orbital_outline(o, doc.style.bond_length)] == ["solid", "white"]
    from PySide6.QtCore import QPointF
    assert cv.hit(QPointF(o2.x + 15, o2.y + 5)) == ("orbital", o2.id)
    doc.rotate([("orbital", o2.id)], math.pi / 2)
    assert doc.orbitals[o2.id].angle == pytest.approx(math.pi / 2)
    xml = cdxml.to_cdxml(doc)
    assert 'GraphicType="Orbital"' in xml and 'OrbitalType="dxyShaded"' in xml
    back = cdxml.from_cdxml(xml)
    kinds = sorted((x.kind, x.fill) for x in back.orbitals.values())
    assert kinds == [("dxy", "shaded"), ("p", "phase")]
    big = [x for x in back.orbitals.values() if x.kind == "dxy"][0]
    assert big.size == pytest.approx(30, abs=1) and big.angle == pytest.approx(math.pi / 2)
    # the orbital flyout holds every kind
    from rsstatchem.ui.chemdraw.palette import PALETTE
    k = win.palette.slot_of("orbital:p:shaded")
    kinds = {t.split(":")[1] for t, _ in PALETTE[k]}
    assert kinds == {"s", "p", "lobe", "hybrid", "dxy", "dz2"}


# --- Structure Perspective (3D) ---------------------------------------------------------------

def test_structure_perspective_keeps_stereo(win, qapp):
    win.load_text("C[C@H](N)C(=O)O")
    key0 = convert.to_inchikey(win.doc)
    dlg = win.structure_perspective()
    assert dlg is not None and dlg.isVisible()
    m = dlg.view.model
    assert "ETKDG" in m.method and m.symbols.count("H") == 7
    assert len(m.coords) == len(m.symbols) and abs(m.coords.mean(axis=0)).max() < 1e-6
    for yaw, pitch in ((0.8, 0.3), (2.6, -0.9)):
        dlg.view.set_angles(yaw, pitch)
        assert dlg.apply()
        assert convert.to_inchikey(win.doc) == key0          # stereo survives
    win.undo_stack.undo()
    assert convert.to_inchikey(win.doc) == key0
    for style in ("Space Filling", "Stick", "Wire Frame"):
        dlg.style_box.setCurrentText(style)
        assert not dlg.view.grab().isNull()
    dlg.close()


def test_threed_core_on_caffeine():
    model = threed.embed(mo.mol_from_text(CAFFEINE))
    assert len(model.symbols) == 24 and len(model.heavy()) == 14
    r = threed.rotation(0.3, 0.2, 0.1)
    assert abs(float(np.linalg.det(r)) - 1) < 1e-9
    pts = threed.project(model, r)
    d0 = np.linalg.norm(model.coords[0] - model.coords[1])
    assert np.linalg.norm(pts[0] - pts[1]) == pytest.approx(d0)


# --- properties: published group-contribution fallbacks ------------------------------------

LIT_TB = {   # K, NIST WebBook / PubChem (see core/chem_draw/gc_tables.py)
    "CN1CCCC1=O": 475.2,              # N-methyl-2-pyrrolidone
    "CN1CCC[C@H]1c1cccnc1": 520.15,   # nicotine
    "c1ccncc1": 388.5,                # pyridine
    "Cc1ccccc1": 383.8,               # toluene
    "[O-][N+](=O)c1ccccc1": 483.95,   # nitrobenzene
}


@pytest.mark.parametrize("smiles,tb", LIT_TB.items())
def test_stein_brown_boiling_points_against_literature(smiles, tb):
    res = stein_brown(Chem.MolFromSmiles(smiles))
    assert res.ok, res.missing
    assert res.values["tb"] == pytest.approx(tb, abs=16)     # S&B report AAD ~15-20 K


def test_marrero_gani_against_literature():
    tol = marrero_gani(Chem.MolFromSmiles("Cc1ccccc1")).values
    assert tol["tc"] == pytest.approx(591.75, abs=15)      # NIST Tc toluene
    assert tol["pc"] == pytest.approx(41.08, abs=2.5)      # bar
    assert tol["tb"] == pytest.approx(383.8, abs=10)
    pyr = marrero_gani(Chem.MolFromSmiles("c1ccncc1")).values
    assert pyr["tc"] == pytest.approx(620.0, abs=15) and pyr["pc"] == pytest.approx(56.7, abs=3)
    assert pyr["tm"] == pytest.approx(231.5, abs=15)
    asp = marrero_gani(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")).values
    assert asp["tm"] == pytest.approx(408, abs=30)          # NIST 405 +- 10 K
    assert not marrero_gani(Chem.MolFromSmiles("C[Si](C)(C)Cl")).ok   # no guessing


def test_caffeine_gets_labelled_fallback_values():
    rows = {p.name: p for p in chemical_properties(Chem.MolFromSmiles(CAFFEINE))}
    bp = {a.key: a for a in rows["Boiling Point"].alternatives}
    assert bp["sb"].value == pytest.approx(674.0, abs=1)    # EPI Suite MPBPWIN: ~689 K
    assert bp["joback"].value is None and "ring N" in bp["joback"].note   # why: no group
    assert rows["Boiling Point"].short == "ML"               # lowest benchmark error
    assert rows["Melting Point"].kind == "experimental"
    mp = {a.key: a for a in rows["Melting Point"].alternatives}
    assert mp["mg1"].value and mp["mg"].value                # still listed in the details
    tc = {a.key: a for a in rows["Critical Temp."].alternatives}
    assert tc["mg"].value is None and tc["joback"].value is None
    assert rows["Critical Temp."].short == "Abdulelah-Gani" and rows["Critical Temp."].value
    assert rows["LogP"].kind == "experimental"
    novel = {p.name: p for p in chemical_properties(
        Chem.MolFromSmiles("O=C(NC1CC2(C1)OCCO2)c1ccc(F)c(Cl)c1"))}
    assert "not BioByte" in novel["LogP"].method


# --- names: offline database, OPSIN --------------------------------------------------------

def test_offline_name_database_both_ways(win):
    assert names.name_db_size() > 50000
    hit = names.db_name(mo.mol_from_text("CN1CCC[C@H]1c1cccnc1"))
    assert hit.name == "3-[(2S)-1-methylpyrrolidin-2-yl]pyridine" and hit.common == "Nicotine"
    assert names.db_name(mo.mol_from_text("CC(F)(Cl)C(Br)CCCC#N")) is None
    assert mo.to_inchikey(mo.mol_from_text(names.db_structure("ibuprofen").smiles)) == \
        "HEFNNWSXXWATRW-UHFFFAOYSA-N"            # PubChem title "Ibuprofen, (+-)-"
    # the window uses it before going online
    win.online_lookup = False
    win.use_opsin = False
    assert win.convert_name_to_structure("nicotine")
    assert convert.to_inchikey(win.doc) == "SNICXCGAKADSCV-JTQLQIEISA-N"
    win.canvas.select_keys(set())
    assert win.convert_structure_to_name() == ["3-[(2S)-1-methylpyrrolidin-2-yl]pyridine"]
    assert "offline database" in win.statusBar().currentMessage()


def test_opsin_path_with_fake_runtime(win, qapp, monkeypatch):
    """The OPSIN step runs in the thread pool, inserts its structure, and falls back to the
    PubChem message when OPSIN cannot parse the name (no Java needed for this test)."""
    calls = []

    def fake(name):
        calls.append(name)
        if name == "2-acetoxybenzoic acid":
            return names.NameResult(name, name, names.OPSIN_SOURCE,
                                    smiles="CC(=O)Oc1ccccc1C(=O)O")
        raise names.NameLookupError(f"OPSIN could not interpret '{name}'.")
    monkeypatch.setattr(names, "opsin_available", lambda allow_system=True: True)
    win.lookup.opsin = fake
    win.online_lookup = False
    assert win.convert_name_to_structure("2-acetoxybenzoic acid")
    _wait_pool(qapp)
    assert calls == ["2-acetoxybenzoic acid"]
    assert convert.to_inchikey(win.doc) == ASPIRIN_KEY
    assert win.convert_name_to_structure("blorbic acid")
    _wait_pool(qapp)
    assert "could not interpret" in win.last_message and "PubChem" in win.last_message


@pytest.mark.skipif(not HAS_OPSIN, reason="needs Java and resources/opsin/opsin-cli.jar "
                                          "(scripts/fetch_opsin.py)")
def test_opsin_real_name_to_structure(win, qapp):
    res = names.opsin_structure("2-acetoxybenzoic acid")
    assert res.source == names.OPSIN_SOURCE
    assert mo.to_inchikey(mo.mol_from_text(res.smiles)) == ASPIRIN_KEY
    out = names.opsin_structures(["(2S)-2-aminopropanoic acid", "not a chemical name xyz",
                                  "1,3,7-trimethylpurine-2,6-dione"])
    assert out[1] is None
    assert mo.to_smiles(mo.mol_from_text(out[0])) == "C[C@H](N)C(=O)O"
    assert mo.to_inchikey(mo.mol_from_text(out[2])) == "RYYVLZVUVIJVGH-UHFFFAOYSA-N"
    win.online_lookup = False
    assert win.convert_name_to_structure("4-hydroxy-3-methoxybenzaldehyde oxime")
    _wait_pool(qapp)
    assert convert.to_smiles(win.doc) == "COc1cc(/C=N/O)ccc1O"
    assert "OPSIN" in win.statusBar().currentMessage()


def test_opsin_absent_is_reported(monkeypatch):
    monkeypatch.setattr(names, "opsin_jar", lambda: None)
    assert not names.opsin_available()
    with pytest.raises(names.NameLookupError):
        names.opsin_structure("ethanol")


# --- Mass Fragmentation tool ------------------------------------------------------------------

def _cut_line(doc, bond, k=1.6):
    pa, pb = doc.atoms[bond.a], doc.atoms[bond.b]
    mx, my = (pa.x + pb.x) / 2, (pa.y + pb.y) / 2
    dx, dy = pb.x - pa.x, pb.y - pa.y
    return (mx - dy * k, my + dx * k), (mx + dy * k, my - dx * k)


def test_mass_fragmentation_core_and_tool(win):
    d = Document(preset("ACS Document 1996"))
    convert.insert_text(d, "Cc1ccccc1", 100, 100)
    methyl = next(b for b in d.bonds.values()
                  if 1 in (len(d.neighbors(b.a)), len(d.neighbors(b.b))))
    (x1, y1), (x2, y2) = _cut_line(d, methyl)
    frags = fragment.cleave(d, x1, y1, x2, y2)
    assert sorted((f.formula, round(f.mass, 4)) for f in frags) == \
        [("C6H5", 77.0391), ("CH3", 15.0235)]
    assert sum(f.mass for f in frags) == pytest.approx(92.0626, abs=1e-4)
    assert fragment.cleave(d, 0, 0, 1, 1) == []
    # the palette tool: drag a line across acetophenone's C(O)-Ph bond
    win.load_text("CC(=O)c1ccccc1")
    doc = win.doc
    b = next(b for b in doc.bonds.values() if b.order == 1 and
             len(doc.neighbors(b.a)) == 3 and len(doc.neighbors(b.b)) == 3)
    a, e = _cut_line(doc, b)
    win.set_tool("fragment")
    drag(win.canvas, a, e)
    texts = sorted(t.text for t in doc.texts.values())
    assert texts == ["C2H3O\n43.0184", "C6H5\n77.0391"]
    assert any(c.dashed for c in doc.curves.values())

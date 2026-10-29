"""Scenes for the Chemical Drawing chapter (``content/40_chemdraw.md``).

Every structure is built through the app's own document API or its "insert structure text"
path (SMILES), and every name comes from the offline name list, so nothing here needs the
network (or OPSIN).
"""
from PySide6.QtCore import QPointF, QRect, QRectF
from PySide6.QtWidgets import QPushButton

from guide.studio import scene

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
S_IBUPROFEN = "CC(C)Cc1ccc(cc1)[C@H](C)C(=O)O"
SALICYLIC = "OC(=O)c1ccccc1O"
ANHYDRIDE = "CC(=O)OC(C)=O"
RXN = f"{SALICYLIC}.{ANHYDRIDE}>>{ASPIRIN}.CC(=O)O"


# ------------------------------------------------------------------------------ helpers
def _window(st, smiles=None, size=(1240, 800)):
    from rsstatchem.ui.chemdraw.window import ChemDrawWindow
    ChemDrawWindow._counter = 0              # every picture shows "Untitled Document-1"
    w = st.module("chem", size=size)
    if smiles:
        w.load_text(smiles)
    st.settle(10)
    _refresh(st, w)
    return w


def _refresh(st, w):
    """Fill the Analysis / Chemical Properties windows now (they normally wait 120 ms)."""
    w.analysis.refresh()
    w.properties.refresh()
    w.properties.wait_cdk(30)
    st.settle(6)


def _hide_docks(w):
    w.analysis_dock.hide()
    w.properties_dock.hide()
    w.template_dock.hide()


def _rect(w, x1, y1, x2, y2) -> QRect:
    """Window rectangle of a rectangle in document (scene) coordinates."""
    cv = w.canvas
    a = cv.viewport().mapTo(w, cv.mapFromScene(QPointF(x1, y1)))
    b = cv.viewport().mapTo(w, cv.mapFromScene(QPointF(x2, y2)))
    return QRect(a, b).normalized()


def _atom(w, aid, r=None) -> QRect:
    """Window rectangle of an atom: its drawn label, or a square around the atom."""
    a = w.doc.atoms[aid]
    box = w.canvas.info.label_boxes.get(aid)
    if box is not None and r is None:
        m = 2.0
        return _rect(w, box.left() - m, box.top() - m, box.right() + m, box.bottom() + m)
    r = r or w.doc.style.bond_length * 0.45
    return _rect(w, a.x - r, a.y - r, a.x + r, a.y + r)


def _bond(w, bid) -> QRect:
    b = w.doc.bonds[bid]
    p, q = w.doc.atoms[b.a], w.doc.atoms[b.b]
    m = w.doc.style.bond_length * 0.2
    return _rect(w, min(p.x, q.x) - m, min(p.y, q.y) - m, max(p.x, q.x) + m, max(p.y, q.y) + m)


def _text_rect(w, tid, m=3.0) -> QRect:
    r = w.canvas.info.text_boxes[tid]
    return _rect(w, r.left() - m, r.top() - m, r.right() + m, r.bottom() + m)


def _keys_rect(w, keys, pad=0.0) -> QRect:
    bb = w.doc.bbox(keys)
    return _rect(w, bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad)


def _viewport_rect(w) -> QRect:
    vp = w.canvas.viewport()
    return QRect(vp.mapTo(w, vp.rect().topLeft()), vp.size())


def _drawing_crop(w, pad=1.2) -> QRect:
    """Window rectangle of the whole drawing (plus a margin), inside the canvas."""
    bl = w.doc.style.bond_length
    r = w.canvas.scene().itemsBoundingRect().adjusted(-bl * pad, -bl * pad, bl * pad, bl * pad)
    return _rect(w, r.left(), r.top(), r.right(), r.bottom()).intersected(_viewport_rect(w))


def _insert(w, smiles, x, y):
    """Insert a structure (SMILES) centred at page point (x, y); returns its keys."""
    keys = w.canvas.insert_text(smiles, x, y, "Insert Structure")
    w.canvas.select_keys(set())
    return keys


def _show(w, zoom=None):
    w.canvas.select_keys(set())
    if zoom:
        w.canvas.set_zoom_percent(zoom)
        w.canvas.show_content()
    else:
        w.canvas.fit()


def _reaction(st, size=(1240, 760), conditions=True):
    """Aspirin synthesis drawn as a reaction, with the conditions on the arrow, cleaned up
    (Structure > Clean Up Reaction) and centred on the page."""
    w = _window(st, RXN, size)
    if conditions:
        ar = next(iter(w.doc.arrows.values()))
        w.canvas.edit("Conditions", lambda: (setattr(ar, "above", "H2SO4"),
                                             setattr(ar, "below", "85 °C")))
    w.clean_up_reaction()
    w.canvas.select_keys(set())
    w.center_on_page()
    w.canvas.fit()
    st.settle(10)
    _refresh(st, w)
    return w


# ------------------------------------------------------------------------------ the window
@scene("chem.window", "Jendela Chemical Drawing dengan struktur aspirin")
def window(st):
    w = _window(st, ASPIRIN)
    _show(w, 150)
    st.settle(10)
    shot = st.shot(w)
    shot.box(w.menuBar(), "Menu bar: semua perintah modul")
    shot.box(w.general_bar, "General Toolbar: berkas, clipboard, undo, zoom, Clean Up, "
                            "Check Structure, nama ↔ struktur, Output Viewer")
    shot.box(w.style_bar, "Style Toolbar: huruf, ukuran dan gaya teks")
    shot.box(w.palette, "Main Toolbar: semua alat menggambar")
    shot.box(w.canvas, "Halaman dokumen tempat menggambar")
    shot.box(w.analysis_dock, "Analysis: rumus, massa, m/z, analisis unsur")
    shot.box(w.properties_dock, "Chemical Properties: sifat fisika-kimia")
    shot.box(w.zoom_box, "Pembesaran (Zoom); di kiri status bar tampil petunjuk alat aktif")
    return shot


@scene("chem.palette", "Main Toolbar: alat menggambar (segitiga kecil = ada pilihan lain)")
def palette(st):
    w = _window(st, None, (1000, 1000))
    shot = st.shot(w)
    p = w.palette
    area = QRect()
    for b in p.buttons:
        area = area.united(QRect(b.mapTo(w, b.rect().topLeft()), b.size()))
    shot.crop(area, pad=8)
    # (tool, text); the side follows the button's column (even slots are on the left)
    labels = [("lasso", "Lasso: pilih bebas"), ("select", "Marquee: pilih kotak"),
              ("rotate", "Rotate: memutar"), ("eraser", "Eraser: menghapus"),
              ("bond:single", "Solid Bond: ikatan"), ("text", "Text: label dan teks"),
              ("bond:double", "Ikatan rangkap ▸"), ("arrow:forward", "Panah ▸"),
              ("bond:dashed", "Putus-putus, datif ▸"), ("plus", "Tanda plus reaksi"),
              ("bond:hash", "Hashed wedge"), ("bracket:square", "Kurung ▸"),
              ("bond:bold", "Ikatan tebal"), ("charge+", "Muatan, radikal ▸"),
              ("bond:wedge", "Wedge ▸"), ("periodic", "Tabel periodik"),
              ("bond:wavy", "Ikatan bergelombang"), ("template", "Templates ▸"),
              ("chain", "Acyclic Chain"), ("atom:C", "Atom (unsur) ▸"),
              ("ring:benzene", "Benzena"), ("ring:cyclopentadiene", "Siklopentadiena"),
              ("ring:chair1", "Kursi sikloheksana"), ("pen", "Pen: kurva"),
              ("orbital:p:shaded", "Orbital ▸"), ("fragment", "Mass Fragmentation"),
              ("perspective", "Perspektif 3D"), ("shape:rect", "Bentuk ▸"),
              ("tlc", "Pelat KLT dan gel ▸"), ("clipart:beaker", "Alat gelas ▸")]
    for tid, text in labels:
        k = p.slot_of(tid)
        shot.arrow(p.buttons[k], text, side="left" if k % 2 == 0 else "right", pad=0)
    rings = QRect()
    for tid in ("ring:cyclopropane", "ring:cyclooctane"):
        b = p.buttons[p.slot_of(tid)]
        rings = rings.united(QRect(b.mapTo(w, b.rect().topLeft()), b.size()))
    shot.arrow(rings, "Cincin 3 sampai 8 atom", side="left", pad=1)
    return shot


# ------------------------------------------------------------------------------ drawing
@scene("chem.drawing", "Label atom, wedge, singkatan (nickname) dan indikator stereokimia")
def drawing(st):
    from rsstatchem.core.chem_draw import convert
    from rsstatchem.ui.chemdraw import objects_actions

    w = _window(st, None, (1180, 760))
    _hide_docks(w)
    caf = _insert(w, CAFFEINE, 200, 170)
    _insert(w, S_IBUPROFEN, 330, 170)
    ester = _insert(w, "Cc1ccc(C)cc1", 265, 245)
    doc = w.doc
    # the two methyl groups of p-xylene become nicknames: MeO and CO2Me
    methyls = sorted((a for k, a in ester if k == "atom" and len(doc.neighbors(a)) == 1),
                     key=lambda a: doc.atoms[a].x)
    w.canvas.edit("Atom Label", lambda: (convert.apply_label_text(doc, methyls[0], "MeO"),
                                         convert.apply_label_text(doc, methyls[1], "CO2Me")))
    objects_actions.set_show_stereo(w, True)
    _show(w, 175)
    st.settle(10)
    shot = st.shot(w)
    shot.crop(_drawing_crop(w, 1.2), pad=0)
    n_caf = next(a for k, a in caf if k == "atom" and doc.atoms[a].symbol == "N"
                 and len(doc.neighbors(a)) == 3)
    shot.arrow(_atom(w, n_caf), "Label atom (N, O ...): ketik saat menunjuk atom",
               side="left")
    wedge = next(b for b in doc.bonds.values() if b.style in ("wedge", "hash"))
    shot.arrow(_bond(w, wedge.id), "Ikatan wedge / hashed wedge", side="right")
    stereo = next((a for a in doc.atoms.values() if a.stereo), None)
    if stereo is not None:
        bl = doc.style.bond_length
        sx, sy = stereo.x, stereo.y + bl * 0.62
        shot.arrow(_rect(w, sx - bl * 0.38, sy - bl * 0.3, sx + bl * 0.38, sy + bl * 0.3),
                   f"Indikator stereo ({stereo.stereo}) dari "
                                        "Show Stereochemistry", side="right")
    lab = next(a for a in doc.atoms.values() if a.label == "CO2Me")
    shot.arrow(_atom(w, lab.id), "Nickname (singkatan): CO2Me", side="right")
    lab2 = next(a for a in doc.atoms.values() if a.label and a.label != "CO2Me")
    shot.arrow(_atom(w, lab2.id), "Nickname: OMe (ditulis MeO di kiri)", side="left")
    return shot


@scene("chem.templates", "Templates Window dan beberapa templat yang sudah ditempatkan")
def templates(st):
    from rsstatchem.core.chem_draw import templates as tpl

    w = _window(st, None, (1180, 760))
    w.analysis_dock.hide()
    w.properties_dock.hide()
    w.template_dock.show()
    cat = "Conformations & projections"
    w.templates.category.setCurrentText(cat)
    smiles = dict(tpl.TEMPLATES[cat])
    for name, x, y in (("Haworth, β-D-glucopyranose", 210, 150),
                       ("Newman projection, staggered", 330, 150),
                       ("Fischer projection, D-glucose", 210, 255),
                       ("Chair with axial / equatorial bonds", 330, 255)):
        _insert(w, smiles[name], x, y)
    _show(w)
    st.settle(10)
    shot = st.shot(w)
    shot.box(w.templates.category, "Kategori templat")
    shot.box(w.templates.list, "Klik templat, lalu klik di halaman (klik dua kali: di tengah)")
    shot.box(w.palette.button_for("template"), "Tombol Templates di Main Toolbar")
    return shot


# ------------------------------------------------------------------------------ names, properties, NMR
@scene("chem.names", "Convert Structure to Name: nama IUPAC ditulis di bawah struktur")
def names_(st):
    from rsstatchem.ui.chemdraw import intel_actions

    w = _window(st, None, (1240, 700))
    _hide_docks(w)
    _insert(w, CAFFEINE, 440, 200)
    intel_actions.convert_structure_to_name(w)
    _show(w, 150)
    st.settle(6)
    # put the drawing in the right part of the view, clear of the opened menu
    cv = w.canvas
    r = cv.scene().itemsBoundingRect()
    vw = cv.viewport().width() / cv.zoom()
    cv.centerOn(r.right() + 30 - vw / 2, r.center().y())
    st.settle(6)
    shot = st.shot(w)
    shot.menu("Structure", highlight="Convert Structure to Name")
    tid = next(iter(w.doc.texts))
    shot.arrow(_text_rect(w, tid), "Nama dari daftar nama offline", side="bottom")
    return shot


@scene("chem.properties", "Analysis dan Chemical Properties untuk kafein yang dipilih; "
                          "tiga nilai pertama sudah ditempel dengan Paste")
def properties(st):
    from PySide6.QtCore import Qt

    w = _window(st, None, (1240, 900))
    w.resizeDocks([w.analysis_dock, w.properties_dock], [300, 560], Qt.Orientation.Vertical)
    keys = _insert(w, CAFFEINE, 306, 260)
    _show(w, 150)
    w.canvas.select_keys(set(keys))
    _refresh(st, w)
    w._paste_analysis("\n".join(w.analysis.text().splitlines()[:3]))
    w.canvas.select_keys(set(keys))
    st.settle(4)
    # centre the drawing in the part of the canvas that the picture keeps
    tps = w.properties.labels["tPSA"]
    top = w.analysis_dock.mapTo(w, w.analysis_dock.rect().topLeft()).y()
    bottom = tps.mapTo(w, tps.rect().bottomLeft()).y() + 8
    cv = w.canvas
    r = cv.scene().itemsBoundingRect()
    delta = _viewport_rect(w).center().y() - (top + bottom) / 2
    cv.centerOn(r.center().x(), r.center().y() + delta / cv.zoom())
    _refresh(st, w)
    shot = st.shot(w)
    tid = next(iter(w.doc.texts))
    cap = _text_rect(w, tid)
    mol = _keys_rect(w, keys)
    left = min(cap.left(), mol.left()) - 40
    shot.crop(QRect(left, top, w.width() - left, bottom - top), pad=0)
    shot.box(w.analysis.scope, "Lingkup: Selection atau Whole document")
    shot.box(w.analysis.decimals, "Jumlah desimal")
    shot.box(w.analysis.paste_btn, "Paste: tempel nilai yang dicentang ke dokumen")
    shot.box(cap, "Hasil Paste di bawah struktur")
    ct = w.properties.labels["Critical Temp."]
    cb = w.properties.checks["Critical Temp."]
    shot.box(QRect(cb.mapTo(w, cb.rect().topLeft()), cb.size()).united(
        QRect(ct.mapTo(w, ct.rect().topLeft()), ct.size())), "Label dalam kurung siku: sumber data "
                                                    "atau metode perkiraan")
    return shot


@scene("chem.nmr", "Prediksi spektrum ¹H-NMR etil benzoat (Structure ▸ Predict ¹H-NMR Shifts)")
def nmr(st):
    from rsstatchem.ui.chemdraw import intel_actions

    w = _window(st, "CCOC(=O)c1ccccc1", (1100, 700))
    nw = intel_actions.show_nmr(w, "1H")
    nw.resize(1000, 760)
    from PySide6.QtWidgets import QSplitter
    nw.findChild(QSplitter).setSizes([420, 190])
    st.settle(10)
    shot = st.shot(nw)
    shot.arrow(nw.nucleus_box, "Inti: ¹H atau ¹³C", side="top")
    shot.arrow(nw.freq_box, "Frekuensi spektrometer", side="top")
    shot.arrow(nw.solvent_box, "Pelarut (tanda sinyal sisa)", side="top")
    shot.arrow(nw.open_btn, "Tumpangkan spektrum terukur", side="top")
    shot.box(nw.view, "Spektrum prediksi; klik puncak untuk menandai atomnya di gambar")
    shot.box(nw.table, "Daftar puncak: δ, multiplisitas, J, jumlah inti, atom, dasar prediksi")
    return shot


# ------------------------------------------------------------------------------ worked example
@scene("chem.ex_arrow", "Fly-out panah di Main Toolbar dan jendela Arrow Properties")
def ex_arrow(st):
    from rsstatchem.ui.chem_editor.dialogs import ArrowDialog

    w = _reaction(st, (1180, 720), conditions=False)
    _hide_docks(w)
    _show(w, 150)
    ar = next(iter(w.doc.arrows.values()))
    ar_copy = type(ar)(**{**ar.__dict__, "above": "H2SO4", "below": "85 °C"})
    st.settle(6)
    shot = st.shot(w)
    b = w.palette.button_for("arrow:forward")
    pos = QPointF(shot.rect_of(b).topRight()) + QPointF(4, 0)
    img = st.grab_menu(b.menu())
    shot.overlays.append((img, pos))
    shot.outline(b, pad=1)
    shot.arrow(QRectF(pos.x(), pos.y(), img.width() / 2, img.height() / 2),
               "Fly-out panah (tahan atau klik kanan tombolnya)",
               side="top", pad=0)
    dlg = ArrowDialog(ar_copy, w)
    dlg.resize(420, 10)
    bl = w.doc.style.bond_length
    arect = _rect(w, ar.x1, ar.y1 - bl * 0.5, ar.x2, ar.y1 + bl * 0.5)
    shot.overlay(dlg, QPointF(arect.left() - 40, arect.bottom() + 90))
    shot.arrow(arect, "Klik dua kali panah: Arrow Properties", side="top")
    return shot


@scene("chem.ex_reaction", "Show Reaction Interpretation: reaktan merah, produk magenta, "
                           "panah biru")
def ex_reaction(st):
    w = _reaction(st, (1180, 720))
    _hide_docks(w)
    w.act_interpret.setChecked(True)
    w._toggle_interpretation(True)
    _show(w)
    st.settle(10)
    shot = st.shot(w)
    shot.crop(_drawing_crop(w, 1.5), pad=0)
    return shot


@scene("chem.ex_balance", "Check Reaction Balance: jumlah atom di kedua ruas")
def ex_balance(st):
    from rsstatchem.ui.chem_editor.dialogs import BalanceDialog

    w = _reaction(st)
    dlg = BalanceDialog(w.doc, w, send=w._send_section)
    dlg.resize(520, 300)
    shot = st.shot(dlg)
    shot.box(dlg.table, "Jumlah setiap unsur di reaktan dan produk")
    send = next(b for b in dlg.findChildren(QPushButton) if "Output" in b.text())
    shot.box(send, "Kirim tabel ke Output Viewer")
    return shot


@scene("chem.ex_stoich", "Reaction Stoichiometry: 1,38 g asam salisilat dan 3,06 g "
                         "anhidrida asetat")
def ex_stoich(st):
    from rsstatchem.ui.chem_editor.dialogs import StoichiometryDialog

    w = _reaction(st)
    dlg = StoichiometryDialog(w.doc, w, send=w._send_section)
    for r, s in enumerate(dlg.species):
        if s.role == "Reactant" and "C7H6O3" in s.formula:
            dlg.set_amount(r, 1.38, "g")
        elif s.role == "Reactant" and "C4H6O3" in s.formula:
            dlg.set_amount(r, 3.06, "g")
    dlg.resize(900, 280)
    shot = st.shot(dlg)
    t = dlg.table
    h = t.horizontalHeader()
    top = t.mapTo(dlg, t.rect().topLeft())

    def col(c0, c1):
        x0 = h.sectionViewportPosition(c0) + t.viewport().x()
        x1 = h.sectionViewportPosition(c1) + h.sectionSize(c1) + t.viewport().x()
        rows = h.height() + sum(t.rowHeight(r) for r in range(t.rowCount()))
        return QRect(top.x() + x0 + 2, top.y(), x1 - x0 - 4, rows + 2)
    shot.box(col(5, 6), "Jumlah reaktan dan satuannya")
    shot.box(col(7, 10), "Ekuivalen, mmol, massa, pereaksi pembatas, hasil teoretis")
    return shot


@scene("chem.ex_export", "File ▸ Export ▸ PNG Image dan pilihan resolusinya")
def ex_export(st):
    from rsstatchem.ui.chem_editor.dialogs import ImageOptionsDialog

    w = _reaction(st, (1180, 720))
    _hide_docks(w)
    _show(w, 100)
    st.settle(6)
    shot = st.shot(w)
    shot.menu("File > Export", highlight="PNG Image")
    dlg = ImageOptionsDialog(w, 600, False)
    item = shot.menu_item_rect("File > Export > PNG Image")
    shot.overlay(dlg, QPointF(item.right() + 40, item.top() + 40))
    return shot


# ------------------------------------------------------------------------------ objects and styles
@scene("chem.objects", "Objek gambar: pelat KLT, alat gelas, kurung polimer, bentuk dan panah")
def objects(st):
    w = _window(st, None, (1180, 760))
    _hide_docks(w)
    doc = w.doc
    bl = doc.style.bond_length

    def build():
        tlc = doc.add_tlc(60, 70, 60 + bl * 4.2, 70 + bl * 6.2, lanes=3)
        tlc.lanes = [[{"rf": 0.72}], [{"rf": 0.72}, {"rf": 0.31}], [{"rf": 0.31}]]
        doc.add_clipart(150, 100, w=50, h=60, name="roundbottom", fill=0.4)
        doc.add_clipart(215, 70, w=26, h=90, name="condenser")
        doc.add_clipart(258, 105, w=46, h=55, name="beaker", fill=0.5)
        doc.add_clipart(318, 95, w=42, h=65, name="erlenmeyer", fill=0.3)
        doc.add_shape(390, 100, 450, 155, kind="shadowrect")
        doc.add_shape(462, 100, 517, 155, kind="ellipse", shaded=True)
        y = 230
        doc.add_arrow(175, y, 230, y, kind="equilibrium")
        doc.add_arrow(265, y, 320, y, kind="retro")
        doc.add_arrow(355, y, 410, y, kind="block", fill="shaded")
        doc.add_arrow(445, y - 12, 495, y - 12, kind="arc", sweep=180)
        for x, yy, text in ((55, 172, "Pelat KLT"), (205, 172, "Alat gelas (Clipware)"),
                            (420, 172, "Bentuk (Shapes)"), (68, 255, "Kurung polimer"),
                            (168, 255, "Kesetimbangan"), (262, 255, "Retrosintesis"),
                            (355, 255, "Block arrow"), (447, 255, "Arc arrow")):
            doc.add_text(x, yy, text, chemical=False)
    w.canvas.edit("Objects", build)
    keys = _insert(w, "CCCCC", 98, 230)
    atoms = sorted((i for k, i in keys if k == "atom"), key=lambda a: doc.atoms[a].x)
    a1, a2 = doc.atoms[atoms[1]], doc.atoms[atoms[3]]
    w.canvas.edit("Bracket", lambda: doc.add_bracket(
        a1.x - bl * 0.45, min(a1.y, a2.y) - bl * 0.75, a2.x - bl * 0.45,
        max(a1.y, a2.y) + bl * 0.75, "square", "n"))
    _show(w)
    st.settle(10)
    shot = st.shot(w)
    shot.crop(_drawing_crop(w, 1.0), pad=0)
    return shot


@scene("chem.styles", "File ▸ Open Style Sheets dan jendela File ▸ Document Settings")
def styles(st):
    from rsstatchem.ui.chem_editor.dialogs import DocumentSettingsDialog

    w = _window(st, ASPIRIN, (1180, 760))
    _hide_docks(w)
    _show(w, 150)
    st.settle(6)
    shot = st.shot(w)
    shot.menu("File > Open Style Sheets", highlight="ACS Document 1996")
    dlg = DocumentSettingsDialog(w.doc.style, w)
    dlg.adjustSize()
    vp = _viewport_rect(w)
    at = QPointF(vp.right() - dlg.sizeHint().width() - 30, vp.bottom() - dlg.sizeHint().height()
                 - 20)
    shot.overlay(dlg, at)

    from PySide6.QtWidgets import QFormLayout
    form = dlg.findChild(QFormLayout)

    def row(field):
        """Shot rectangle of a form row (its label and field) of the overlaid dialog."""
        r = QRect(field.mapTo(dlg, field.rect().topLeft()), field.size())
        lab = form.labelForField(field)
        if lab is not None:
            r = r.united(QRect(lab.mapTo(dlg, lab.rect().topLeft()), lab.size()))
        return QRectF(r.adjusted(-12, 0, 0, 0)).translated(at)
    shot.box(row(dlg.preset), "Ambil semua nilai dari salah satu style sheet")
    shot.box(row(dlg.bond_length), "Panjang ikatan tetap")
    shot.box(row(dlg.line_width), "Tebal garis ikatan")
    shot.box(row(dlg.font_size), "Ukuran label atom")
    shot.box(row(dlg.caption), "Ukuran teks keterangan")
    return shot

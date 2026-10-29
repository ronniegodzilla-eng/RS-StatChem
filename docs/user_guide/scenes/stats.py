"""Scenes for the Statistics module chapter (``content/10_statistics.md``).

All pictures use one small, deterministic study dataset: 45 students taught with three
learning methods, with pre- and post-test scores and a satisfaction rating.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtCore import QRect, QRectF
from PySide6.QtWidgets import QLabel, QPushButton, QSplitter

from guide.studio import scene

SIZE = (920, 560)
CROP_TOP = 52          # output_posthoc: cut the end of the table above


# ------------------------------------------------------------------------------ demo data
def demo_dataset(seed: int = 32):
    from rsstatchem.core.dataset import NOMINAL, ORDINAL, SCALE, Dataset, Variable

    rng = np.random.default_rng(seed)
    n_per = 15
    metode = np.repeat([1.0, 2.0, 3.0], n_per)
    rng.shuffle(metode)
    n = len(metode)
    jk = rng.integers(1, 3, n).astype(float)
    usia = rng.integers(19, 24, n).astype(float)
    pre = np.round(rng.normal(62, 7, n), 1)
    gain = np.select([metode == 1, metode == 2, metode == 3], [3.0, 9.0, 15.0])
    post = np.round(np.clip(pre + gain + rng.normal(0, 4.5, n), 0, 100), 1)
    puas = np.clip(np.round(2.2 + 0.6 * metode + rng.normal(0, 0.8, n)), 1, 5)
    puas[[6, 29]] = 9.0                       # "tidak menjawab" (user-missing code)
    df = pd.DataFrame({"id": np.arange(1, n + 1, dtype=float), "jk": jk, "metode": metode,
                       "usia": usia, "pretest": pre, "posttest": post, "kepuasan": puas})
    variables = [
        Variable("id", decimals=0, label="Nomor responden", measure=NOMINAL, columns=6),
        Variable("jk", decimals=0, label="Jenis kelamin", measure=NOMINAL,
                 value_labels={1.0: "Laki-laki", 2.0: "Perempuan"}, columns=9),
        Variable("metode", decimals=0, label="Metode pembelajaran", measure=NOMINAL,
                 value_labels={1.0: "Konvensional", 2.0: "Daring", 3.0: "Campuran"},
                 columns=12),
        Variable("usia", decimals=0, label="Usia (tahun)", measure=SCALE, columns=6),
        Variable("pretest", decimals=1, label="Nilai pretest", measure=SCALE),
        Variable("posttest", decimals=1, label="Nilai posttest", measure=SCALE),
        Variable("kepuasan", decimals=0, label="Tingkat kepuasan", measure=ORDINAL,
                 value_labels={1.0: "Sangat tidak puas", 2.0: "Tidak puas", 3.0: "Netral",
                               4.0: "Puas", 5.0: "Sangat puas", 9.0: "Tidak menjawab"},
                 missing=[9.0], columns=14),
    ]
    return Dataset(df, variables)


def editor(st, size=SIZE, labels: bool = False, tab: int = 0):
    """A Data Editor showing the demo data (as if opened from studi_metode.rsd)."""
    for old in [e for e in st.ctx.editors if e.ds.path == "studi_metode.rsd"]:
        old.doc.mark_saved()          # close the previous scene's editor: one demo window
        old.close()
    w = st.module("stats", size=size)
    w.load_dataset(demo_dataset())
    w.ds.name = "DataSet1"
    w.ds.path = "studi_metode.rsd"
    w.doc.mark_saved()
    w.act_labels.setChecked(labels)
    w.data_model.set_show_labels(labels)
    w.tabs.setCurrentIndex(tab)
    w._update_title()
    w._fit_columns()
    w.data_view.clearSelection()
    w.data_view.setCurrentIndex(w.data_model.index(0, 0))
    st.settle()
    return w


def fresh_output(st):
    """An empty Output Viewer (it is shared by all Data Editors of the session)."""
    viewer = st.ctx.output()
    viewer.dirty = False
    viewer.new_document()
    viewer.set_zoom(1.0)
    viewer.set_look("spss")
    return viewer


def run(st, w, dlg):
    """What Data Editor's run_analysis does after OK: compute on the active data, show it."""
    from rsstatchem.ui.data_editor import _compute_with

    result = w._compute_on_views(lambda view_ds: _compute_with(dlg, view_ds))
    dlg.accept()          # remember the selections, like a real OK
    dlg.hide()
    st.ctx.show_output(result.section if hasattr(result, "section") else result)
    st.settle()


def move(dlg, key: str, *names: str):
    """Select variables in the source list and press the arrow of target ``key``."""
    from rsstatchem.ui.dialogs.base import NAME_ROLE

    dlg.source.clearSelection()
    for i in range(dlg.source.count()):
        item = dlg.source.item(i)
        if item.data(NAME_ROLE) in names:
            item.setSelected(True)
    target = next(t for t in dlg._targets if t.key == key)
    dlg._arrow_clicked(target)
    dlg.source.clearSelection()


def capture_subdialog(fn):
    """Run code that would ``exec()`` a SubDialog; show it instead and return it."""
    from rsstatchem.ui.dialogs.base import SubDialog

    shown = []
    original = SubDialog.exec

    def fake_exec(self):
        shown.append(self)
        self.show()
        return 0

    SubDialog.exec = fake_exec
    try:
        fn()
    finally:
        SubDialog.exec = original
    return shown[-1] if shown else None


def tidy(st, dlg):
    """After show(): no current item in the variable list, OK drawn as the default button."""
    st.settle()
    for b in dlg.findChildren(QPushButton):
        b.setAutoDefault(False)
    ok = getattr(dlg, "ok_button", None) or getattr(dlg, "ok", None)
    if ok is not None:
        ok.setDefault(True)
        ok.setFocus()
    if hasattr(dlg, "source"):
        dlg.source.setCurrentRow(-1)
        dlg.source.clearSelection()
    st.settle()


def anova_dialog(st, w):
    from rsstatchem.ui.dialogs.compare import OneWayAnovaDialog

    dlg = OneWayAnovaDialog(w.ds, w)
    dlg.reset()
    move(dlg, "deps", "posttest")
    move(dlg, "factor", "metode")
    dlg.opts.update({"descriptives": True, "homogeneity": True, "welch": True,
                     "posthoc": ["tukey"], "means_plot": True, "effect": True})
    dlg.resize(720, 430)
    dlg.show()
    tidy(st, dlg)
    return dlg


# ------------------------------------------------------------------------------ scenes
@scene("stats.window", "Jendela Data Editor modul Statistics dengan data contoh")
def window(st):
    w = editor(st)
    shot = st.shot(w)
    shot.box(menu_rect(w), "**Menu bar**: File, Edit, View, Data, Transform, Analyze, Graphs, "
                           "Utilities, Window, Help")
    shot.box(toolbar_rect(w.toolbar), "**Toolbar**: tombol pintas untuk perintah yang sering "
                                      "dipakai")
    header = w.data_view.horizontalHeader()
    n = w.ds.n_vars
    right = header.sectionViewportPosition(n - 1) + header.sectionSize(n - 1)
    shot.box(QRect(header.mapTo(w, _pt(0, 0)), _sz(right, header.height())),
             "**Nama variabel** di kepala kolom; klik dua kali untuk membuka Variable View",
             pad=0)
    rows = w.data_view.verticalHeader()
    vp = w.data_view.viewport()
    model = w.data_model
    k = 14
    first = w.data_view.visualRect(model.index(0, 0))
    last = w.data_view.visualRect(model.index(k - 1, n - 1))
    cells = first.united(last)
    shot.box(QRect(rows.mapTo(w, _pt(0, 0)), _sz(rows.width(), cells.bottom())),
             "**Nomor kasus**: satu baris = satu responden/sampel", pad=0)
    shot.box(QRect(vp.mapTo(w, cells.topLeft()), cells.size()).adjusted(3, 3, 0, 0),
             "**Data View**: satu sel = nilai satu variabel untuk satu kasus", pad=0)
    shot.box(st.tab(w.tabs.tabBar(), "Data View").united(
        st.tab(w.tabs.tabBar(), "Variable View")),
        "Tab **Data View / Variable View** (++Ctrl+T++ untuk berpindah)")
    shot.arrow(w.counts, "Jumlah kasus dan variabel; status Filter/Weight/Split", side="bottom")
    return shot


def menu_rect(w) -> QRect:
    bar = w.menuBar()
    r = QRect()
    for act in bar.actions():
        if act.isVisible():
            r = r.united(bar.actionGeometry(act))
    return QRect(bar.mapTo(w, r.topLeft()), r.size())


def toolbar_rect(tb) -> QRect:
    r = QRect()
    for act in tb.actions():
        wid = tb.widgetForAction(act)
        if wid is not None and wid.isVisible():
            r = r.united(wid.geometry())
    return QRect(tb.mapTo(tb.window(), r.topLeft()), r.size()).adjusted(-2, 0, 2, 0)


@scene("stats.toolbar", "Tombol-tombol toolbar Data Editor")
def toolbar(st):
    w = editor(st)
    shot = st.shot(w)
    tb = w.toolbar

    def btn(*names):
        r = QRect()
        for name in names:
            b = st.toolbar_button(tb, name)
            r = r.united(QRect(b.mapTo(w, _pt(0, 0)), b.size()))
        return r

    shot.box(btn("Open data document"), "**Open Data**: membuka berkas data", pad=0)
    shot.box(btn("Save this document"), "**Save**: menyimpan data", pad=0)
    shot.box(btn("Import text or CSV data"), "**Import**: impor berkas teks/CSV", pad=0)
    shot.box(btn("Undo", "Redo"), "**Undo / Redo**: membatalkan / mengulang perubahan data",
             pad=0)
    shot.box(btn("Show the Output Viewer"), "Membuka **Output Viewer**", pad=0)
    shot.box(btn("Go to case"), "**Go to Case**: lompat ke kasus nomor tertentu", pad=0)
    shot.box(btn("Insert variable", "Insert cases"),
             "**Insert Variable / Insert Cases**: menyisipkan kolom atau baris", pad=0)
    shot.box(btn("Sort cases"), "**Sort Cases**: mengurutkan kasus", pad=0)
    shot.box(btn("Find data"), "**Find**: mencari nilai di data", pad=0)
    shot.box(btn("Show value labels"), "**Value Labels**: menampilkan label nilai, bukan kode",
             pad=0)
    shot.box(btn("Frequencies", "Descriptives"), "**Frequencies / Descriptives**: statistik "
                                                 "deskriptif cepat", pad=0)
    r = toolbar_rect(tb)
    shot.crop(QRect(0, r.top() - 2, r.right() + 24, r.height() + 6), pad=0)
    return shot


@scene("stats.variable_view", "Variable View: satu baris untuk setiap variabel")
def variable_view(st):
    w = editor(st, size=(920, 430), tab=1)
    v = w.var_view
    for c, width in enumerate((92, 78, 52, 70, 150, 150, 70, 64, 52, 92)):
        v.setColumnWidth(c, width)
    v.setColumnHidden(10, True)
    v.setCurrentIndex(w.var_model.index(2, 5))
    st.settle()
    shot = st.shot(w)
    header = v.horizontalHeader()
    vp = v.viewport()

    def col(c, rows=7):
        top = v.visualRect(w.var_model.index(0, c))
        bottom = v.visualRect(w.var_model.index(rows - 1, c))
        head = QRect(header.mapTo(w, _pt(header.sectionViewportPosition(c), 0)),
                     _sz(header.sectionSize(c), header.height()))
        body = QRect(vp.mapTo(w, top.topLeft()), top.united(bottom).size())
        return head.united(body)

    shot.box(col(0), "**Name**: nama singkat tanpa spasi (dipakai di rumus)", pad=-3)
    shot.box(col(1), "**Type**: Numeric, String, Date ...", pad=-3)
    shot.box(col(2).united(col(3)), "**Width / Decimals**: lebar dan jumlah desimal", pad=-3)
    shot.box(col(4), "**Label**: keterangan lengkap yang muncul di tabel output", pad=-3)
    shot.box(col(5), "**Values**: label nilai, mis. 1 = Laki-laki (klik dua kali sel)", pad=-3)
    shot.box(col(6), "**Missing**: kode nilai hilang, mis. 9 = tidak menjawab", pad=-3)
    shot.box(col(9), "**Measure**: Scale, Ordinal, atau Nominal", pad=-3)
    return shot


def _pt(x, y):
    from PySide6.QtCore import QPoint
    return QPoint(x, y)


def _sz(w, h):
    from PySide6.QtCore import QSize
    return QSize(w, h)


@scene("stats.value_labels", "Mengisi label nilai (Value Labels) untuk variabel metode")
def value_labels(st):
    import copy

    from rsstatchem.ui.dialogs.data import ValueLabelsDialog

    w = editor(st, tab=1)
    var = copy.deepcopy(w.ds.var("metode"))
    var.value_labels = {1.0: "Konvensional", 2.0: "Daring"}
    dlg = ValueLabelsDialog(var, w)
    dlg.list.setCurrentRow(-1)
    dlg.value.setText("3")
    dlg.label.setText("Campuran")
    dlg.resize(460, 330)
    dlg.show()
    st.settle()
    shot = st.shot(dlg)
    buttons = {b.text(): b for b in dlg.findChildren(QPushButton)}
    shot.arrow(dlg.value, "Ketik nilai (kode angka)", side="right")
    shot.arrow(dlg.label, "Ketik labelnya", side="right")
    shot.arrow(dlg.list, "Daftar label nilai yang sudah dibuat", side="right")
    shot.box(buttons["Add"], "**Add** menambahkan pasangan nilai dan label ke daftar")
    shot.box(buttons["Change"].geometry().united(buttons["Remove"].geometry()).translated(
        buttons["Change"].parentWidget().mapTo(dlg, _pt(0, 0))),
        "**Change / Remove** mengubah atau menghapus label yang dipilih di daftar")
    return shot


@scene("stats.labels_on", "Data View dengan label nilai ditampilkan (View ▸ Value Labels)")
def labels_on(st):
    w = editor(st, labels=True)
    shot = st.shot(w)
    shot.arrow(st.toolbar_button(w.toolbar, "Show value labels"),
               "Tombol Value Labels: kode angka ⇄ label", side="top")
    model = w.data_model
    vp = w.data_view.viewport()
    a = w.data_view.visualRect(model.index(0, 1))
    b = w.data_view.visualRect(model.index(9, 2))
    shot.arrow(QRect(vp.mapTo(w, a.topLeft()), a.united(b).size()),
               "Sel jk dan metode kini menampilkan label", side="bottom")
    c = w.data_view.visualRect(model.index(6, 6))
    shot.arrow(QRect(vp.mapTo(w, c.topLeft()), c.size()),
               "Kode 9 = Tidak menjawab (missing)", side="bottom")
    n = w.ds.n_vars
    last = w.data_view.visualRect(model.index(10, n - 1))
    corner = vp.mapTo(w, last.bottomRight())
    shot.crop(QRect(0, 0, corner.x() + 30, corner.y() + 2), pad=0)
    return shot


@scene("stats.import_text", "Impor berkas CSV: pengaturan pemisah dan pratinjau")
def import_text(st):
    import tempfile
    from pathlib import Path

    from rsstatchem.ui.dialogs.data import TextImportDialog

    ds = demo_dataset()
    folder = Path(tempfile.mkdtemp(prefix="rsguide-csv-"))
    path = folder / "hasil_survei.csv"
    df = ds.df.copy()
    # an Indonesian-style CSV: semicolons, decimal comma
    text = df.to_csv(sep=";", index=False, decimal=",", float_format="%g")
    path.write_text(text, encoding="utf-8")
    w = editor(st)
    dlg = TextImportDialog(str(path), w)
    dlg.resize(580, 440)
    dlg.show()
    st.settle()
    shot = st.shot(dlg)
    shot.arrow(dlg.encoding, "Encoding (biasanya terdeteksi otomatis)", side="right")
    shot.arrow(dlg.delimiter, "Pemisah kolom: koma, titik koma, Tab ...", side="right")
    shot.arrow(dlg.decimal, "Simbol desimal: titik atau koma", side="right")
    shot.arrow(dlg.header, "Baris pertama = nama variabel", side="right")
    shot.box(dlg.preview, "Pratinjau: periksa kolom terbaca dengan benar")
    shot.box(dlg.ok, "**Import** memasukkan data ke Data Editor")
    return shot


@scene("stats.menu_compare", "Menu Analyze ▸ Compare Means")
def menu_compare(st):
    w = editor(st)
    shot = st.shot(w)
    shot.menu("Analyze > Compare Means", highlight="One-Way ANOVA")
    return shot


@scene("stats.anova_dialog", "Dialog One-Way ANOVA yang sudah diisi")
def anova(st):
    w = editor(st)
    dlg = anova_dialog(st, w)
    shot = st.shot(dlg)
    shot.box(dlg.source, "Daftar variabel dataset (label [nama])")
    deps = next(t for t in dlg._targets if t.key == "deps")
    factor = next(t for t in dlg._targets if t.key == "factor")
    shot.box(deps.widget, "**Dependent List**: variabel yang dibandingkan rata-ratanya")
    shot.box(factor.widget, "**Factor**: variabel pengelompok")
    shot.box(deps.arrow, "Tombol panah memindahkan variabel terpilih")
    buttons = {b.text(): b for b in dlg.findChildren(QPushButton)}
    shot.box(buttons["Post Hoc..."].geometry().united(buttons["Options..."].geometry())
             .translated(buttons["Post Hoc..."].parentWidget().mapTo(dlg, _pt(0, 0))),
             "Sub-dialog: Contrasts, **Post Hoc**, **Options**, Bootstrap")
    shot.box(dlg.ok_button, "**OK** menjalankan analisis")
    return shot


@scene("stats.posthoc", "Sub-dialog Post Hoc: memilih uji Tukey")
def posthoc(st):
    from rsstatchem.ui.dialogs import glm_model_ui as gm
    from PySide6.QtWidgets import QCheckBox

    w = editor(st)
    dlg = anova_dialog(st, w)
    sub = capture_subdialog(lambda: gm.posthoc_dialog(dlg, "One-Way ANOVA", None,
                                                      significance=True))
    st.settle()
    shot = st.shot(sub)
    tukey = next(c for c in sub.findChildren(QCheckBox) if c.text() == "Tukey")
    games = next(c for c in sub.findChildren(QCheckBox) if c.text() == "Games-Howell")
    shot.box(tukey, "**Tukey**: pilihan umum bila varians antarkelompok homogen (uji Levene "
                    "tidak signifikan)")
    shot.box(games, "**Games-Howell**: dipakai bila varians antarkelompok tidak homogen")
    sub.hide()
    dlg.hide()
    return shot


@scene("stats.output", "Hasil One-Way ANOVA di Output Viewer")
def output(st):
    w = editor(st)
    viewer = fresh_output(st)
    dlg = anova_dialog(st, w)
    run(st, w, dlg)
    viewer.resize(960, 680)
    st.settle()
    # select the ANOVA table, as after a click
    section = viewer.doc.sections[-1]
    from rsstatchem.core.output import Table
    tables = [i for i in section.items if isinstance(i, Table)]
    target = next((t for t in tables if t.title.upper().startswith("ANOVA")), tables[0])
    viewer.findChild(QSplitter).setSizes([235, 725])
    viewer.select(f"i{target.id}")
    st.settle()
    viewer.view.scrollToAnchor(f"s{section.id}")
    st.settle()
    shot = st.shot(viewer)
    shot.box(viewer.tree, "**Navigasi** (outline): klik judul untuk melompat ke tabel/grafik")
    shot.box(viewer.view, "**Isi output**: tabel bergaya SPSS; klik untuk memilih, "
                          "klik dua kali untuk mengedit")
    tb = viewer.findChild(type(w.toolbar))
    shot.box(st.toolbar_button(tb, "Export to Word (.docx)"), "Ekspor seluruh output ke Word")
    shot.box(st.toolbar_button(tb, "Copy"), "**Copy**: salin objek terpilih, tempel di Word "
                                            "sebagai tabel")
    shot.box(st.toolbar_button(tb, "Zoom Out").geometry().united(
        st.toolbar_button(tb, "Zoom In").geometry()).translated(tb.mapTo(viewer, _pt(0, 0))),
        "**Zoom Out / Zoom In**: memperkecil atau memperbesar tampilan")
    look_label = next(lb for lb in tb.findChildren(QLabel) if "TableLook" in lb.text())
    shot.box(look_label.geometry().united(viewer.look_box.geometry()).translated(
        tb.mapTo(viewer, _pt(0, 0))), "**TableLook**: SPSS Default atau APA (7th edition)")
    return shot


@scene("stats.compute", "Transform ▸ Compute Variable: membuat variabel baru dari rumus")
def compute(st):
    from rsstatchem.ui.dialogs.transform import ComputeDialog

    w = editor(st)
    dlg = ComputeDialog(w.ds, w)
    dlg.target.setText("selisih")
    dlg.builder.set_text("posttest - pretest")
    groups = [dlg.builder.groups.item(i).text() for i in range(dlg.builder.groups.count())]
    if "Arithmetic" in groups:
        dlg.builder.groups.setCurrentRow(groups.index("Arithmetic"))
    dlg.builder.validate()
    dlg.resize(840, 450)
    dlg.show()
    st.settle()
    dlg.builder.validate()
    for b in dlg.findChildren(QPushButton):
        b.setAutoDefault(False)
    dlg.ok.setDefault(True)
    st.settle()
    shot = st.shot(dlg)
    shot.box(dlg.target, "**Target Variable**: nama variabel baru (atau yang sudah ada)")
    shot.box(dlg.builder.variables, "Daftar variabel: klik dua kali untuk menyisipkan ke rumus")
    shot.box(dlg.builder.editor, "**Numeric Expression**: rumus perhitungan")
    shot.box(dlg.builder.functions.parentWidget() and QRect(
        dlg.builder.groups.mapTo(dlg, _pt(0, 0)),
        dlg.builder.functions.geometry().united(dlg.builder.groups.geometry()).size()),
        "Kelompok fungsi dan daftar fungsi (MEAN, SQRT, LN, RV.NORMAL ...)")
    shot.box(dlg.if_button, "**If...**: hitung hanya untuk kasus yang memenuhi syarat")
    return shot


@scene("stats.output_posthoc", "Lanjutan output: perbandingan ganda Tukey dan Means Plot")
def output_posthoc(st):
    from rsstatchem.core.output import Table

    w = editor(st)
    viewer = fresh_output(st)
    dlg = anova_dialog(st, w)
    run(st, w, dlg)
    viewer.resize(1150, 960)
    viewer.findChild(QSplitter).setSizes([300, 850])
    st.settle()
    section = viewer.doc.sections[-1]
    tables = [i for i in section.items if isinstance(i, Table)]
    target = next(t for t in tables if t.title.startswith("Multiple Comparisons"))
    viewer.view.scrollToAnchor(f"i{target.id}")
    st.settle()
    shot = st.shot(viewer)
    bar = viewer.view.verticalScrollBar()
    r = shot.rect_of(viewer.view)
    sb = viewer.view.verticalScrollBar().width() if bar.isVisible() else 0
    shot.crop(QRectF(r.left(), r.top() + CROP_TOP, r.width() - sb - 4, r.height() - CROP_TOP),
              pad=0)
    return shot


@scene("stats.ttest", "Independent-Samples T Test dengan sub-dialog Define Groups")
def ttest(st):
    from rsstatchem.ui.dialogs.base import DefineGroupsDialog
    from rsstatchem.ui.dialogs.compare import IndependentTDialog

    w = editor(st)
    dlg = IndependentTDialog(w.ds, w)
    dlg.reset()
    move(dlg, "vars", "posttest")
    move(dlg, "group", "jk")
    dlg.opts["group_def"] = {"var": "jk", "groups": (1.0, 2.0), "cut": None}
    dlg.targets_changed()
    dlg.resize(700, 400)
    dlg.show()
    tidy(st, dlg)
    groups = DefineGroupsDialog(w.ds.var("jk"), dlg.opts["group_def"], True, dlg)
    groups.resize(300, 10)
    shot = st.shot(dlg)
    vars_t = next(t for t in dlg._targets if t.key == "vars")
    group_t = next(t for t in dlg._targets if t.key == "group")
    shot.box(vars_t.widget, "**Test Variable(s)**: variabel yang dibandingkan rata-ratanya")
    shot.box(group_t.widget, "**Grouping Variable**: jk(1 2) artinya kelompok kode 1 vs 2")
    shot.box(dlg.define_button, "**Define Groups...** membuka sub-dialog Define Groups (bawah)")
    btn = shot.rect_of(dlg.define_button)
    at = _pt(dlg.width() - groups.width() - 10, dlg.height() + 24)
    shot.overlay(groups, at)
    shot.box(QRect(at, groups.size()),
             "Isi kode **Group 1** dan **Group 2** (atau Cut point), lalu Continue")
    return shot


@scene("stats.output_copy", "TableLook APA dan menu Edit ▸ Copy Special di Output Viewer")
def output_copy(st):
    from rsstatchem.core.output import Table

    w = editor(st)
    viewer = fresh_output(st)
    dlg = anova_dialog(st, w)
    run(st, w, dlg)
    viewer.set_look("apa")
    viewer.resize(960, 600)
    viewer.findChild(QSplitter).setSizes([200, 760])
    st.settle()
    section = viewer.doc.sections[-1]
    tables = [i for i in section.items if isinstance(i, Table)]
    target = next(t for t in tables if t.title.upper().startswith("ANOVA"))
    viewer.select(f"i{target.id}")
    st.settle()
    before = tables[tables.index(target) - 1]
    viewer.view.scrollToAnchor(f"i{before.id}")
    st.settle()
    bar = viewer.view.verticalScrollBar()
    bar.setValue(bar.value() - 70)        # the ANOVA table below the opened menu
    st.settle()
    shot = st.shot(viewer)
    shot.menu("Edit > Copy Special", highlight="Copy as HTML (Word table)")
    viewer.set_look("spss")
    return shot

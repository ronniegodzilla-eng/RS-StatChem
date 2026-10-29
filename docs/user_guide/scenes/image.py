"""Scenes for the Image Analysis chapter (module ``image``, ImageJ style).

Only local or synthetic images are used: ``tests/data_imagej/inputs/blobs.gif`` (ImageJ's
public-domain "Blobs" sample) and numpy-generated stacks. Commands run through the module's
own code (the ImageJ bar's macro interpreter and command registry), so every number in the
pictures is a real result.

The module has one ImageJ bar per app and separate top-level windows (images, Results, ROI
Manager, ...). Each scene starts by closing what the previous one left open, and ends the same
way, so the Window menu read for the feature reference lists no stale windows.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRect, QRectF
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QPushButton,
                               QWidget)

from guide.studio import scene

# image windows get their own (macOS-style) menu bar unless this is set; the pictures show
# the window as it looks on Windows and inside the macOS global-menu setup
os.environ["RSSTATCHEM_NO_WINDOW_MENUBAR"] = "1"

ROOT = Path(__file__).resolve().parents[3]
BLOBS = ROOT / "tests" / "data_imagej" / "inputs" / "blobs.gif"
GAP = 18          # space between composed windows (logical px)


# ------------------------------------------------------------------------------ helpers
def _windows():
    from rsstatchem.ui.imagej import windows as W
    return W


def _clear(st, bar) -> None:
    """Close every image/auxiliary window and empty Results, tables and the ROI Manager."""
    from rsstatchem.core.imaging.results import ResultsTable

    W = _windows()
    for w in list(W.WINDOWS):
        if hasattr(w, "confirm_close"):
            w.confirm_close = False
        W.unregister(w)
        try:
            w.close()
        except RuntimeError:
            pass
    s = bar.session
    for doc in list(s.images):
        s.close_image(doc)
    s.roi_manager.clear()
    s.results = ResultsTable.results("Results")
    s.tables = {"Results": s.results}
    bar.window_manager.tables.clear()
    bar.toolbar.set_tool(0)
    bar.show_status(" ")
    st.settle()


def _bar(st):
    bar = st.module("image")
    _clear(st, bar)
    bar.resize(560, 92)
    st.settle()
    return bar


def _done(st, bar, shot):
    _clear(st, bar)
    return shot


def _open_blobs(st, bar, zoom: float = 2.0):
    from rsstatchem.core.imaging.io import open_image

    doc = open_image(str(BLOBS))
    bar.session.add_image(doc)
    st.settle()
    win = _windows().window_for(doc)
    _fit(st, win, zoom)
    return doc, win


def _add(st, bar, doc, zoom: float = 2.0):
    bar.session.add_image(doc)
    st.settle()
    win = _windows().window_for(doc)
    _fit(st, win, zoom)
    return win


def _fit(st, win, zoom: float = 1.0) -> None:
    """Window exactly around the image at ``zoom`` (no empty margin below the image)."""
    if abs(win.canvas.magnification - zoom) > 1e-9:
        win.canvas.zoom_to(zoom)
    st.settle()
    doc = win.doc
    cw, ch = round(doc.width * zoom), round(doc.height * zoom)
    sliders = sum(s.sizeHint().height() + 2 for s in win.sliders.values()
                  if s.isVisibleTo(win))
    win.resize(cw + 4, ch + win.info.sizeHint().height() + 6 + sliders)
    st.settle()
    win.canvas.refresh()
    win.update_title()
    st.settle()


def _macro(st, bar, code: str):
    out = bar.run_macro_text(code)
    st.settle()
    return out


def _union(*widgets_or_rects, shot) -> QRectF:
    r = QRectF()
    for t in widgets_or_rects:
        r = r.united(shot.rect_of(t))
    return r


def _button(widget, text: str) -> QPushButton:
    return next(b for b in widget.findChildren(QPushButton) if b.text() == text)


def _check(widget, text: str) -> QCheckBox:
    return next(b for b in widget.findChildren(QCheckBox) if b.text() == text)


def _text_rect(cb, origin) -> QRectF:
    """Window rectangle of a check box's text only (its tick stays visible next to a badge)."""
    fm = cb.fontMetrics()
    tw = fm.horizontalAdvance(cb.text().replace("&", ""))
    left = max(0, cb.sizeHint().width() - tw - 6)
    top_left = cb.mapTo(origin, cb.rect().topLeft())
    return QRectF(top_left.x() + left, top_left.y(), tw + 8, cb.height())


def _dialog(bar, label: str):
    """The GenericDialog of a registry command, shown (not exec'd) for a picture."""
    from rsstatchem.ui.imagej.dialogs.generic import GenericDialog

    cmd = bar.session.registry.get(label)
    fields = cmd.dialog_fields(bar.session, bar.session.active)
    dlg = GenericDialog(label.rstrip(". ").removesuffix("..."), fields, bar)
    dlg.adjustSize()
    return dlg


def _below(shot, widget, gap: float = GAP) -> QPointF:
    r = shot.rect_of(widget)
    return QPointF(r.left(), r.bottom() + gap)


def _right_of(shot, x: float, y: float = 0) -> QPointF:
    return QPointF(x + GAP, y)


def _stack_doc():
    """A synthetic 3-D stack: bright spheres in a 220 x 180 x 24 volume."""
    from rsstatchem.core.imaging.image import ImageDoc

    rng = np.random.default_rng(7)
    z, y, x = np.mgrid[0:24, 0:180, 0:220].astype(np.float32)
    vol = np.full(z.shape, 18.0, np.float32)
    for _ in range(26):
        cz, cy, cx = rng.uniform(2, 22), rng.uniform(15, 165), rng.uniform(15, 205)
        r = rng.uniform(9, 20)
        d2 = ((x - cx) ** 2 + (y - cy) ** 2) / r ** 2 + ((z - cz) * 2.2) ** 2 / r ** 2
        vol += 210 * np.exp(-2.5 * d2)
    vol += rng.normal(0, 6, vol.shape)
    arr = np.clip(vol, 0, 255).astype(np.uint8)
    doc = ImageDoc.from_array(arr, "ZYX", "sel_3D.tif")
    doc.calibration.pixel_width = doc.calibration.pixel_height = 0.2
    doc.calibration.pixel_depth = 0.5
    doc.calibration.unit = "µm"
    return doc


# ------------------------------------------------------------------------------ scenes
@scene("image.bar", "Bar utama ImageJ: menu, alat seleksi, dan baris status")
def bar_scene(st):
    bar = _bar(st)
    doc, win = _open_blobs(st, bar, 1.0)
    bar.show_status(win.status_text(118, 86))
    st.settle()
    shot = st.shot(bar)
    tb = bar.toolbar.buttons
    from rsstatchem.ui.imagej import toolbar as T
    shot.box(bar.menuBar(), "Menu: File, Edit, Image, Process, Analyze, Plugins, Window, Help")
    shot.box(_union(tb[T.RECTANGLE], tb[T.FREEROI], shot=shot),
             "Seleksi area: Rectangle, Oval, Polygon, Freehand")
    shot.box(_union(tb[T.LINE], tb[T.ANGLE], shot=shot), "Garis dan sudut (Straight Line, Angle)")
    shot.box(_union(tb[T.POINT], tb[T.TEXT], shot=shot),
             "Point (titik/hitung), Wand (seleksi otomatis), Text")
    shot.box(_union(tb[T.MAGNIFIER], tb[T.HAND], shot=shot),
             "Zoom (kaca pembesar) dan Scrolling (geser tampilan)")
    shot.box(tb[T.DROPPER], "Color picker: warna depan/latar")
    shot.arrow(bar.status, "Baris status: posisi kursor dan nilai piksel", side="bottom")
    return _done(st, bar, shot)


@scene("image.window", "Jendela citra: judul, baris info, citra, dan seleksi")
def window_scene(st):
    from rsstatchem.core.imaging.roi import shapes as S

    bar = _bar(st)
    doc, win = _open_blobs(st, bar, 1.5)
    doc.roi = S.OvalRoi(150, 100, 40, 36)
    bar.session.image_updated(doc)
    st.settle()
    shot = st.shot(win)
    w = win.width()
    shot.arrow((w * 0.3, -27, w * 0.4, 24), "Judul: nama citra dan zoom (150%)", side="top",
               pad=0)
    shot.arrow(win.info, "Baris info: ukuran, tipe, LUT, memori", side="right")
    zoom = win.canvas.magnification
    roi_rect = QRectF(150 * zoom, 100 * zoom, 40 * zoom, 36 * zoom).translated(
        shot.rect_of(win.canvas).topLeft())
    shot.arrow(roi_rect, "Seleksi (ROI) oval: yang diukur oleh Measure", side="right", pad=4)
    return _done(st, bar, shot)


@scene("image.open_samples", "File ▸ Open Samples: citra contoh ImageJ (diunduh dari internet)")
def open_samples(st):
    bar = _bar(st)
    shot = st.shot(bar)
    shot.menu("File > Open Samples", highlight="Blobs")
    return _done(st, bar, shot)


@scene("image.set_scale", "Kalibrasi jarak dengan Analyze ▸ Set Scale")
def set_scale(st):
    from rsstatchem.core.imaging.roi import shapes as S

    bar = _bar(st)
    doc, win = _open_blobs(st, bar, 1.0)
    doc.roi = S.LineRoi(28, 232, 128, 232)            # 100 pixels
    bar.session.image_updated(doc)
    st.settle()
    dlg = _dialog(bar, "Set Scale...")
    dlg.findChild(QDoubleSpinBox, "known").setValue(10)
    dlg.findChild(QLineEdit, "unit").setText("µm")
    dlg.adjustSize()
    shot = st.shot(dlg)
    at = QPointF(-win.width() - 2 * GAP, -30)
    shot.overlay(win, at)
    zoom = win.canvas.magnification
    top = QPointF(win.canvas.mapTo(win, win.canvas.rect().topLeft())) + at
    line = QRectF(28 * zoom, 232 * zoom - 6, 100 * zoom, 12).translated(top)
    shot.box(line, "Garis (Straight Line) sepanjang benda yang ukurannya diketahui", pad=2)
    shot.arrow(dlg.findChild(QDoubleSpinBox, "distance"),
               "Panjang garis dalam piksel (terisi otomatis)", side="right")
    shot.arrow(dlg.findChild(QDoubleSpinBox, "known"), "Jarak sebenarnya", side="right")
    shot.arrow(dlg.findChild(QLineEdit, "unit"), "Satuan panjang", side="right")
    shot.arrow(_text_rect(dlg.findChild(QCheckBox, "global"), dlg),
               "Berlaku untuk semua citra yang terbuka", side="right")
    return _done(st, bar, shot)


@scene("image.roi_manager", "ROI Manager: banyak seleksi, diberi nomor dan diukur sekaligus")
def roi_manager(st):
    from rsstatchem.core.imaging.roi import shapes as S
    from rsstatchem.ui.imagej.frames import roi_manager as RM

    bar = _bar(st)
    doc, win = _open_blobs(st, bar, 1.5)
    s = bar.session
    rois = [S.OvalRoi(98, 22, 30, 30), S.OvalRoi(150, 100, 40, 36),
            S.PolygonRoi([12, 52, 60, 30, 8], [150, 140, 175, 200, 182]),
            S.RectRoi(180, 180, 50, 44)]
    for r in rois:
        RM.add(s, doc, r)
    mgr = bar.run_command("ROI Manager...")
    mgr.refresh()
    mgr.show_all_box.setChecked(True)
    mgr.labels_box.setChecked(True)
    mgr.list.item(1).setSelected(True)
    mgr.resize(300, 300)
    st.settle()
    # the overlay and the manager's label painter both write the names (the overlay's are
    # clipped to each ROI); keep the manager's
    doc.overlay.show_labels = doc.overlay.draw_names = False
    win.canvas.refresh()
    st.settle()
    shot = st.shot(win)
    at = _right_of(shot, win.width(), 30)
    shot.overlay(mgr, at)

    def in_mgr(widget):
        r = QRectF(widget.mapTo(mgr, widget.rect().topLeft()), widget.size().toSizeF())
        return r.translated(at)

    shot.box(in_mgr(mgr.list), "Daftar ROI; klik untuk memilihnya di citra")
    shot.box(in_mgr(_button(mgr, "Add [t]")), "Add: simpan seleksi aktif (tombol T)")
    shot.box(in_mgr(_button(mgr, "Measure")), "Measure: ukur ROI terpilih (atau semua)")
    shot.box(in_mgr(_button(mgr, "More >>")), "Simpan/buka RoiSet, gabung, Multi Measure, ...")
    shot.box(_union(in_mgr(mgr.show_all_box), in_mgr(mgr.labels_box), shot=shot),
             "Tampilkan semua ROI beserta nomornya")
    return _done(st, bar, shot)


@scene("image.set_measurements", "Analyze ▸ Set Measurements: memilih besaran yang diukur")
def set_measurements(st):
    bar = _bar(st)
    _open_blobs(st, bar, 1.0)
    bar.session.run("Set Measurements...", "area mean min centroid perimeter shape display "
                                           "redirect=None decimal=3")
    dlg = _dialog(bar, "Set Measurements...")
    shot = st.shot(dlg)
    boxes = {b.text(): b for b in dlg.findChildren(QCheckBox)}
    want = [("Area", "Luas", "left"), ("Mean gray value", "Rata-rata intensitas", "right"),
            ("Shape descriptors", "Circularity, aspect ratio, roundness, solidity", "left"),
            ("Display label", "Nama citra/ROI di kolom Label", "right")]
    for text, note, side in want:
        if text in boxes:
            target = boxes[text] if side == "left" else _text_rect(boxes[text], dlg)
            shot.arrow(target, note, side=side)
    combo = dlg.findChild(QComboBox, "redirect")
    if combo is not None:
        shot.arrow(combo, "Ukur pada citra lain dengan seleksi yang sama", side="right")
    spin = dlg.findChild(QDoubleSpinBox, "decimal")
    if spin is not None:
        shot.arrow(spin, "Jumlah angka desimal di tabel", side="right")
    return _done(st, bar, shot)


@scene("image.results", "Tabel Results setelah mengukur keempat ROI")
def results(st):
    from rsstatchem.core.imaging.roi import shapes as S
    from rsstatchem.ui.imagej.frames import roi_manager as RM

    bar = _bar(st)
    doc, win = _open_blobs(st, bar, 1.0)
    s = bar.session
    s.run("Set Measurements...", "area mean perimeter shape display redirect=None decimal=3")
    for r in (S.OvalRoi(98, 22, 30, 30), S.OvalRoi(150, 100, 40, 36),
              S.PolygonRoi([12, 52, 60, 30, 8], [150, 140, 175, 200, 182]),
              S.RectRoi(180, 180, 50, 44)):
        RM.add(s, doc, r)
    s.show_table(RM.measure(s, doc))
    st.settle()
    rw = bar.window_manager.tables["Results"]
    rw.set_font_size(14)
    rw.view.resizeColumnsToContents()
    rw.resize(min(rw.view.horizontalHeader().length() + 46, 1000),
              rw.view.verticalHeader().length() + rw.view.horizontalHeader().height()
              + rw.menuBar().sizeHint().height() + 8)
    rw.view.scrollToTop()
    st.settle()
    shot = st.shot(rw)
    hh = rw.view.horizontalHeader()

    def col(name):
        for i in range(rw.model.columnCount()):
            if str(rw.model.headerData(i, hh.orientation())) == name:
                x = hh.sectionViewportPosition(i)
                top = hh.mapTo(rw, hh.rect().topLeft())
                return QRectF(top.x() + x, top.y(), hh.sectionSize(i),
                              hh.height() + rw.view.verticalHeader().length())
        return None

    rlabel = col("Label")
    if rlabel is not None:
        shot.arrow(rlabel, "Label: nama citra dan ROI", side="bottom", pad=1)
    rarea = col("Area")
    if rarea is not None:
        shot.arrow(rarea, "Area dalam piksel² (atau satuan kalibrasi)", side="bottom", pad=1)
    rcirc = col("Circ.")
    if rcirc is not None:
        shot.arrow(rcirc, "Circularity: 1 = lingkaran sempurna", side="bottom", pad=1)
    return _done(st, bar, shot)


@scene("image.threshold", "Image ▸ Adjust ▸ Threshold: piksel objek ditandai merah")
def threshold(st):
    bar = _bar(st)
    doc, win = _open_blobs(st, bar, 1.5)
    _macro(st, bar, 'run("Set Scale...", "distance=100 known=10 unit=um"); '
                    'setAutoThreshold("Default");')
    th = bar.run_command("Threshold...")
    st.settle()
    shot = st.shot(win)
    at = _right_of(shot, win.width(), -30)
    shot.overlay(th, at)

    def in_th(widget):
        r = QRectF(widget.mapTo(th, widget.rect().topLeft()), widget.size().toSizeF())
        return r.translated(at)

    shot.box(in_th(th.plot), "Histogram; rentang terpilih berwarna merah")
    shot.box(_union(in_th(th.min_slider), in_th(th.max_field), shot=shot),
             "Batas bawah dan atas (slider atau ketik)")
    shot.box(in_th(th.method), "Metode otomatis (Default, Otsu, Li, ...)")
    shot.box(_text_rect(th.dark, th).translated(at), "Centang bila objek terang di latar gelap")
    shot.box(in_th(_button(th, "Apply")), "Apply: jadikan citra biner (mask)")
    return _done(st, bar, shot)


@scene("image.analyze_particles", "Dialog Analyze ▸ Analyze Particles")
def analyze_particles(st):
    bar = _bar(st)
    _open_blobs(st, bar, 1.0)
    _macro(st, bar, 'run("Set Scale...", "distance=100 known=10 unit=um"); '
                    'setAutoThreshold("Default");')
    dlg = _dialog(bar, "Analyze Particles...")
    dlg.findChild(QComboBox, "show").setCurrentText("Outlines")
    for text in ("Display results", "Clear results", "Summarize", "Exclude on edges"):
        _check(dlg, text).setChecked(text != "Exclude on edges")
    dlg.adjustSize()
    st.settle()
    shot = st.shot(dlg)
    shot.arrow(dlg.findChild(QLineEdit, "size"), "Rentang luas partikel yang dihitung (µm²)",
               side="right")
    shot.arrow(dlg.findChild(QLineEdit, "circularity"), "Rentang circularity (0 sampai 1)",
               side="right")
    shot.arrow(dlg.findChild(QComboBox, "show"), "Gambar hasil: Outlines, Masks, Overlay, ...",
               side="right")
    shot.arrow(_text_rect(_check(dlg, "Exclude on edges"), dlg),
               "Abaikan partikel yang terpotong tepi citra", side="right")
    shot.arrow(_check(dlg, "Display results"), "Tabel Results: satu baris per partikel",
               side="left")
    shot.arrow(_check(dlg, "Summarize"), "Tabel Summary: jumlah, total luas, %Area",
               side="left")
    return _done(st, bar, shot)


@scene("image.particles_result", "Hasil Analyze Particles: outline bernomor, Summary, Results")
def particles_result(st):
    bar = _bar(st)
    doc, win = _open_blobs(st, bar, 1.0)
    bar.session.run("Set Measurements...", "area perimeter redirect=None decimal=3")
    _macro(st, bar, 'run("Set Scale...", "distance=100 known=10 unit=um"); '
                    'setAutoThreshold("Default"); '
                    'run("Analyze Particles...", "size=0-Infinity show=Outlines display '
                    'clear summarize");')
    W = _windows()
    outl = next(w for w in W.image_windows() if w.doc is not doc)
    _fit(st, outl, 1.5)
    tables = bar.window_manager.tables
    summ, res = tables["Summary"], tables["Results"]

    def width(t, extra=8):
        return t.view.horizontalHeader().length() + t.view.verticalHeader().width() + extra

    for t in (summ, res):
        t.set_font_size(14)
        t.view.setMinimumHeight(0)
        t.setMinimumHeight(0)
    summ.resize(width(summ), summ.view.horizontalHeader().height() +
                summ.view.verticalHeader().length() + summ.menuBar().sizeHint().height() + 6)
    res.resize(width(res, 26), outl.height() + 30)     # level with the title bar
    res.view.scrollToTop()
    st.settle()
    shot = st.shot(outl)
    p_res = QPointF(outl.width() + GAP, -30)
    shot.overlay(res, p_res)
    p_sum = QPointF(0, outl.height() + GAP)
    # the table window does not shrink below a few empty rows; picture only the used part
    used = summ.view.mapTo(summ, summ.view.rect().topLeft()).y() + \
        summ.view.horizontalHeader().height() + summ.view.verticalHeader().length() + 4
    st.grab(summ)
    shot.overlays.append((summ.grab(QRect(0, 0, summ.width(), used)).toImage(), p_sum))
    shot.box(shot.rect_of(outl.canvas), "Drawing of blobs.gif: outline setiap partikel dan "
                                        "nomornya")
    shot.box(QRectF(p_res, res.size().toSizeF()), "Results: ukuran setiap partikel (nomor "
                                                  "sama dengan di gambar)")
    shot.box(QRectF(p_sum.x(), p_sum.y(), summ.width(), used), "Summary: Count = 64 partikel, luas "
                                                   "total, ukuran rata-rata, %Area")
    return _done(st, bar, shot)


@scene("image.filters", "Process ▸ Filters: filter-filter penghalus dan penajam")
def filters(st):
    bar = _bar(st)
    _open_blobs(st, bar, 1.0)
    shot = st.shot(bar)
    shot.menu("Process > Filters", highlight="Gaussian Blur")
    return _done(st, bar, shot)


@scene("image.stack", "Jendela stack: slider untuk berpindah slice")
def stack(st):
    bar = _bar(st)
    doc = _stack_doc()
    win = _add(st, bar, doc, 2.0)
    doc.set_position(0, 11, 0)
    win.refresh()
    st.settle()
    _fit(st, win, 2.0)
    shot = st.shot(win)
    shot.arrow(win.info, "Baris info: slice 12 dari 24, ukuran dalam µm", side="right")
    sl = win.sliders["stack"]
    if sl.play is not None:
        shot.arrow(sl.play, "Putar/henti animasi", side="bottom")
    shot.arrow(sl.bar, "Slider slice (juga roda mouse, tombol < dan >)", side="bottom")
    return _done(st, bar, shot)


@scene("image.macro", "Editor makro: menjalankan alur hitung partikel dengan satu klik")
def macro(st):
    from rsstatchem.ui.imagej.frames.text_window import TextWindow

    bar = _bar(st)
    code = ('// Hitung partikel pada citra yang sedang aktif\n'
            'run("Set Scale...", "distance=100 known=10 unit=um");\n'
            'setAutoThreshold("Default");\n'
            'run("Set Measurements...", "area mean shape decimal=3");\n'
            'run("Analyze Particles...", "size=0.5-Infinity display summarize");\n'
            'n = nResults;\n'
            'print("Jumlah partikel: " + n);\n')
    ed = TextWindow("Hitung_partikel.ijm", code, bar, macro=True)
    _windows().register(ed)
    ed.resize(720, 190)
    ed.show()
    st.settle()
    shot = st.shot(ed)

    mb = ed.menuBar()
    macros = next(a for a in mb.actions() if a.text() == "Macros")
    g = QRectF(mb.actionGeometry(macros)).translated(shot.rect_of(mb).topLeft())
    shot.arrow(g, "Macros ▸ Run Macro (Ctrl+R) menjalankan seluruh makro", side="top")
    return _done(st, bar, shot)


@scene("image.fiji", "Plugins ▸ Fiji: plugin populer Fiji yang sudah terpasang")
def fiji(st):
    bar = _bar(st)
    shot = st.shot(bar)
    shot.menu("Plugins > Fiji > MorphoLibJ", highlight="Distance Transform Watershed")
    return _done(st, bar, shot)

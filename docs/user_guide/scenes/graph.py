"""Scenes for the Graph & Analysis chapter (OriginLab-style workspace).

The demo data are a seven-point calibration series (absorbance against concentration, a known
straight line plus seeded noise) and an IR-like spectrum with three overlapping bands.
"""
import numpy as np
from PySide6.QtCore import QPoint, QRect
from PySide6.QtWidgets import QGroupBox, QPushButton, QToolBar

from guide.studio import scene

CONC = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]


# ------------------------------------------------------------------------------ helpers
def _main(st, size=(1280, 760), docks=True):
    w = st.module("graph", size=size)
    w.quiet = True
    if not docks:
        for d in (w.pe_dock, w.om_dock, w.results_dock, w.messages_dock):
            d.hide()
    st.settle()
    return w


def _fill(w, bw, columns, rows=None, long_name=None):
    """Put ``columns`` = [(designation, long name, units, comments, values)] in the
    active sheet of ``bw`` as one undo step."""
    ws = bw.worksheet()

    def fn():
        n = rows or max(len(c[4]) for c in columns)
        ws.set_rows(n)
        while len(ws.columns) < len(columns):
            ws.add_column("Y")
        for c, (des, ln, un, cm, vals) in zip(ws.columns, columns):
            c.designation, c.long_name, c.units, c.comments = des, ln, un, cm
            c.values = [None if v is None else float(v) for v in vals] + \
                [None] * (ws.n_rows - len(vals))
    bw.change("Demo data", fn, recalc=False)
    if long_name:
        bw.book.long_name = long_name
        w.subs[bw.book.name].setWindowTitle(bw.book.title)
    return ws


def _calibration(w, bw=None, rows=12):
    """Calibration series: concentration (X), absorbance (Y), SD (yEr)."""
    rng = np.random.default_rng(7)
    absorb = [round(0.012 + 0.0985 * c + rng.normal(0, 0.006), 4) for c in CONC]
    sd = [round(0.012 + 0.0025 * c + abs(rng.normal(0, 0.003)), 4) for c in CONC]
    bw = bw or w.active_book_window()
    _fill(w, bw, [("X", "Konsentrasi", "mg/L", "larutan standar", CONC),
                  ("Y", "Absorbansi", "", "λ = 540 nm", absorb),
                  ("yEr", "SD", "", "n = 3", sd)], rows=rows, long_name="Kalibrasi")
    return bw


def _spectrum(w, bw):
    """An IR-like spectrum with three overlapping bands plus noise."""
    rng = np.random.default_rng(11)
    x = np.linspace(1500, 1800, 151)
    y = (0.05 + 0.82 * np.exp(-0.5 * ((x - 1652) / 11) ** 2)
         + 0.55 * np.exp(-0.5 * ((x - 1690) / 13) ** 2)
         + 0.30 * np.exp(-0.5 * ((x - 1610) / 10) ** 2)
         + rng.normal(0, 0.008, x.size))
    _fill(w, bw, [("X", "Bilangan gelombang", "cm⁻¹", "", np.round(x, 1).tolist()),
                  ("Y", "Absorbansi", "", "sampel A", np.round(y, 4).tolist())],
          long_name="Spektrum")
    return bw


def _place(w, name, x, y, width, height):
    w.subs[name].setGeometry(x, y, width, height)


def _in(w, widget, rect=None):
    """Rectangle of ``widget`` (or of ``rect`` = (x, y, w, h) inside it) in the
    coordinates of window ``w``."""
    if rect is None:
        return QRect(widget.mapTo(w, QPoint(0, 0)), widget.size())
    x, y, rw, rh = rect
    return QRect(widget.mapTo(w, QPoint(int(x), int(y))), QRect(0, 0, int(rw), int(rh)).size())


def _artist(w, gwin, ref):
    """Main-window rectangle of a drawn graph object (axis title, legend...)."""
    gwin.canvas.draw()
    r = gwin._artist_rect(ref)
    return _in(w, gwin.canvas, r) if r else None


def _point(w, gwin, x, y, size=16, layer=0):
    """Main-window rectangle around data point (x, y) of a graph."""
    gwin.canvas.draw()
    ax = gwin.figure.axes[layer]
    px, py = ax.transData.transform((x, y))
    ratio = gwin.canvas.device_pixel_ratio or 1
    h = gwin.figure.bbox.height
    cx, cy = px / ratio, (h - py) / ratio
    return _in(w, gwin.canvas, (cx - size / 2, cy - size / 2, size, size))


def _header_row(w, bw, row, cols):
    """Main-window rectangle of a header row of the grid (0 = Long Name...)."""
    view = bw.sheet_view()
    m = view.model()
    r = view.visualRect(m.index(row, 0)).united(view.visualRect(m.index(row, cols - 1)))
    r.setLeft(-view.verticalHeader().width())
    return _in(w, view.viewport(), (r.x(), r.y(), r.width(), r.height()))


def _calib_graph(w, bw, kind="scatter", cols=(1,)):
    bw.select_columns(list(cols))
    return w.plot(kind)


def _fit(w, gwin, kind="linear", model="Line", **extra):
    choices, current, _ = w.analysis_inputs()
    params = {"input": choices[current], "model": model, "mode": "manual",
              "weighting": "none", "level": 0.95, "conf_band": True, "pred_band": False,
              "residual_plot": True, "output_fitted": False, "to_graph": True,
              "params0": None, "fixed": None, "bounds": None}
    params.update(extra)
    return w.run_fit(kind, params, gwin)


def _row(dlg, field):
    """A form row (label and field) of a dialog, in dialog coordinates."""
    from PySide6.QtWidgets import QFormLayout

    r = _in(dlg, field)
    lay = field.parentWidget().layout()
    forms = [lay] if isinstance(lay, QFormLayout) else \
        field.parentWidget().findChildren(QFormLayout)
    for form in forms:
        label = form.labelForField(field)
        if label is not None:
            return r.united(_in(dlg, label))
    return r


def _toolbar(w, name):
    return w.findChild(QToolBar, name)


def _group(dlg, title):
    return next(g for g in dlg.findChildren(QGroupBox)
                if g.title().replace("&&", "&") == title)


# ------------------------------------------------------------------------------ window
@scene("graph.window", "Jendela Graph & Analysis: workbook, grafik, dan panel-panel pendukung")
def window(st):
    w = _main(st)
    bw = _calibration(w)
    gwin = _calib_graph(w, bw)
    st.settle()
    mw, mh = w.mdi.width(), w.mdi.height()
    _place(w, "Book1", 4, 4, 318, mh - 30)
    _place(w, gwin.graph.title, 326, 4, mw - 330, mh - 8)
    st.settle()
    gwin.fit_page()
    st.settle()
    shot = st.shot(w)
    shot.box(w.menuBar(), "Menu bar; isinya berganti mengikuti jenis jendela yang aktif")
    shot.box(_toolbar(w, "standardToolbar"), "Toolbar Standard: proyek dan jendela baru, "
             "buka/simpan, impor, undo, panel")
    shot.box(_toolbar(w, "columnToolbar"), "Toolbar Column: designation X, Y, Z, yEr, L, "
             "N, Set Column Values")
    shot.box(w.format_bar, "Toolbar Format: huruf dan gaya teks pada grafik")
    shot.box(_toolbar(w, "toolsToolbar"), "Toolbar Tools: Pointer, Zoom, Pan, Data Reader, "
             "teks, panah, garis")
    shot.box(w.pe_dock, "Project Explorer: folder (atas) dan daftar jendela (bawah)")
    shot.box(w.subs["Book1"], "Workbook berisi worksheet data")
    shot.box(w.subs[gwin.graph.title], "Jendela grafik (graph window)")
    shot.box(w.om_dock, "Object Manager: isi jendela aktif (sheet dan kolom, atau layer "
             "dan plot)")
    shot.box(w.results_dock, "Results Log dan Messages Log")
    shot.box(_toolbar(w, "graphs2dToolbar"), "Toolbar 2D Graphs: membuat grafik dengan "
             "satu klik")
    shot.box(w.statusBar(), "Status bar: statistik cepat sel terpilih dan rentang aktif")
    return shot


# ------------------------------------------------------------------------------ workbook
@scene("graph.workbook", "Worksheet: nama kolom, designation, dan baris label")
def workbook(st):
    w = _main(st, size=(1080, 700), docks=False)
    bw = _calibration(w)
    ws = bw.worksheet()
    bw.change("Add", lambda: ws.add_column("Y"), recalc=False)
    ws.columns[3].long_name = "Absorbansi"
    ws.columns[3].units = "mAU"
    w.set_column_values(bw, 3, "col(B)*1000")
    st.settle()
    _place(w, "Book1", 10, 10, 560, 440)
    st.settle()
    view = bw.sheet_view()
    for c, width in enumerate((112, 104, 90, 120)):
        view.horizontalHeader().resizeSection(c, width)
    st.settle()
    shot = st.shot(w)
    shot.crop(_in(w, w.subs["Book1"]), pad=4)
    hh = view.horizontalHeader()
    shot.arrow(_in(w, hh, (hh.sectionPosition(0), 0, 112, hh.height())),
               "Nama pendek dan designation: A(X)", side="left")
    shot.arrow(_in(w, hh, (hh.sectionPosition(2), 0, 90, hh.height())),
               "C(yEr±): galat untuk Y di kirinya", side="top")
    shot.arrow(_header_row(w, bw, 0, 4), "Long Name: judul sumbu dan legenda", side="left")
    shot.arrow(_header_row(w, bw, 1, 4), "Units: satuan, ditulis dalam kurung pada judul "
               "sumbu", side="right")
    shot.arrow(_header_row(w, bw, 2, 4), "Comments: catatan bebas", side="left")
    shot.arrow(_header_row(w, bw, 3, 4), "F(x)=: rumus kolom, dihitung ulang otomatis",
               side="right")
    tb = bw.tabs.tabBar()
    shot.arrow(_in(w, tb, tb.tabRect(0).getRect()), "Tab sheet", side="left")
    return shot


@scene("graph.set_as", "Column ▸ Set As: mengubah designation kolom yang dipilih")
def set_as(st):
    w = _main(st, size=(1000, 470))
    w.results_dock.hide()
    w.messages_dock.hide()
    bw = _calibration(w)
    _place(w, "Book1", 4, 4, 420, 330)
    bw.select_columns([2])
    st.settle()
    shot = st.shot(w)
    shot.menu("Column > Set As", highlight="Y Error")
    return shot


@scene("graph.set_values", "Dialog Set Values untuk kolom D")
def set_values(st):
    from rsstatchem.ui.origin.dialogs import SetValuesDialog

    w = _main(st)
    bw = _calibration(w)
    ws = bw.worksheet()
    bw.change("Add", lambda: ws.add_column("Y"), recalc=False)
    ws.columns[3].long_name = "Absorbansi"
    ws.columns[3].units = "mAU"
    dlg = SetValuesDialog(ws, 3, lambda *a: None, w)
    dlg.edit.setPlainText("col(B)*1000")
    dlg.resize(600, 0)
    dlg.show()
    st.settle()
    shot = st.shot(dlg)
    btn = {b.text().replace("&", ""): b for b in dlg.findChildren(QPushButton)}
    # labels in the margins keep the small dialog at the scale of the other pictures
    shot.arrow(_in(dlg, dlg.first).united(_in(dlg, dlg.last)),
               "Rentang baris; kosong berarti semua baris", side="top")
    shot.arrow(_in(dlg, btn["<<"]), "<< dan >>: pindah ke kolom lain", side="left")
    shot.arrow(dlg.edit, "Rumus: col(B) = isi kolom B, i = nomor baris", side="left")
    shot.arrow(_in(dlg, btn["Function"]).united(_in(dlg, btn["Examples"])),
               "Menu fungsi, daftar kolom, dan contoh rumus", side="right")
    shot.arrow(btn["Apply"], "Apply: hitung tanpa menutup dialog", side="right")
    return shot


@scene("graph.import", "Import Wizard: pengaturan dan pratinjau impor berkas teks")
def import_wizard(st):
    import tempfile
    from pathlib import Path

    from rsstatchem.ui.origin.import_wizard import ImportWizard

    w = _main(st, size=(1000, 600))
    path = Path(tempfile.mkdtemp(prefix="rsguide-csv-")) / "kalibrasi.csv"
    rng = np.random.default_rng(3)
    lines = ["Konsentrasi;Absorbansi;SD", "mg/L;;"]
    for c in CONC:
        a = 0.012 + 0.0985 * c + rng.normal(0, 0.006)
        s = 0.012 + 0.0025 * c + abs(rng.normal(0, 0.003))
        lines.append(f"{c:g};{a:.4f};{s:.4f}".replace(".", ","))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    dlg = ImportWizard(w, [str(path)])
    dlg.resize(900, 600)
    dlg.show()
    st.settle()
    item = dlg.files.item(0)          # a short name instead of the temporary folder
    dlg.files.blockSignals(True)
    item.setText(path.name)
    dlg.files.blockSignals(False)
    st.settle()
    shot = st.shot(dlg)
    shot.box(dlg.files, "Berkas yang akan diimpor (Add Files... untuk menambah)")
    shot.box(_row(dlg, dlg.w["delimiter"]).united(_row(dlg, dlg.w["decimal"])),
             "Pemisah kolom dan desimal (deteksi otomatis)")
    shot.box(_row(dlg, dlg.w["long_name_line"]).united(_row(dlg, dlg.w["comments_to"])),
             "Baris judul untuk Long Name, Units, Comments")
    shot.box(dlg.table, "Pratinjau worksheet hasil impor")
    return shot


# ------------------------------------------------------------------------------ graphs
@scene("graph.plot_menu", "Plot ▸ Basic 2D: memilih jenis grafik untuk kolom yang dipilih")
def plot_menu(st):
    w = _main(st, size=(1080, 540))
    w.results_dock.hide()
    w.messages_dock.hide()
    bw = _calibration(w)
    _place(w, "Book1", 4, 4, 420, 330)
    bw.select_columns([1, 2])
    st.settle()
    shot = st.shot(w)
    shot.menu("Plot > Basic 2D", highlight="Scatter")
    return shot


@scene("graph.graph_parts", "Bagian-bagian grafik yang bisa diatur")
def graph_parts(st):
    w = _main(st, size=(1100, 680))
    w.pe_dock.hide()
    w.results_dock.hide()
    w.messages_dock.hide()
    bw = _calibration(w)
    gwin = _calib_graph(w, bw, "yerror")
    st.settle()
    _place(w, gwin.graph.title, 0, 0, w.mdi.width(), w.mdi.height())
    w.hide_window("Book1")
    st.settle()
    gwin.fit_page()
    st.settle()
    shot = st.shot(w)
    shot.arrow(_artist(w, gwin, ("axis", 0, "y")), "Judul sumbu Y (dari Long Name); klik "
               "dua kali untuk mengedit", side="left")
    shot.arrow(_artist(w, gwin, ("axis", 0, "x")), "Judul sumbu X: Long Name (Units)",
               side="bottom")
    shot.arrow(_artist(w, gwin, ("legend", 0)), "Legenda", side="top")
    shot.arrow(_point(w, gwin, 6.0, 0.6003, 34), "Titik data dengan error bar dari kolom "
               "yEr", side="top")
    shot.box(_toolbar(w, "toolsToolbar"), "Toolbar Tools: zoom, Data Reader, teks, panah")
    shot.box(w.om_dock, "Object Manager: centang untuk menampilkan atau menyembunyikan")
    shot.box(w.menuBar(), "Menu mode grafik: Graph, Format, Insert")
    return shot


@scene("graph.plot_details", "Dialog properti plot (klik dua kali sebuah plot)")
def plot_details(st):
    from rsstatchem.ui.graph.properties import PropertiesDialog, schema_for

    w = _main(st)
    bw = _calibration(w)
    gwin = _calib_graph(w, bw, "line_symbol")
    st.settle()
    ref = ("plot", 0, gwin.graph.layers[0].plots[0].id)
    obj = gwin.resolve(ref)
    sections, lay = schema_for(obj, ref, gwin.graph)
    dlg = PropertiesDialog(gwin._ref_title(ref), obj, sections, lambda p, v: None, lay, w)
    dlg.resize(440, 640)
    dlg.show()
    st.settle()
    shot = st.shot(dlg)
    for title, text in (("Plot", "Nama di legenda, jenis plot, sumbu"),
                        ("Line", "Garis: warna, gaya, tebal"),
                        ("Symbol", "Simbol: bentuk, ukuran, isi")):
        try:
            shot.box(_group(dlg, title), text)
        except StopIteration:
            pass
    return shot


@scene("graph.export", "Export Graph: format, resolusi, dan ukuran untuk publikasi")
def export(st):
    from rsstatchem.ui.graph.dialogs import ExportDialog

    w = _main(st)
    bw = _calibration(w)
    gwin = _calib_graph(w, bw)
    st.settle()
    dlg = ExportDialog(gwin.graph, w)
    dlg.setWindowTitle(dlg.windowTitle() or "Export Graph")
    dlg.show()
    st.settle()
    shot = st.shot(dlg)
    shot.arrow(dlg.fmt, "Format: PNG, TIFF, SVG, PDF, EPS", side="right")
    shot.arrow(dlg.dpi, "Resolusi (DPI) untuk PNG dan TIFF", side="right")
    shot.arrow(_in(dlg, dlg.custom).united(_in(dlg, dlg.aspect)), "Ukuran cetak, mis. "
               "lebar satu kolom jurnal", side="left")
    shot.arrow(dlg.transparent, "Latar belakang transparan", side="left")
    shot.arrow(dlg.info, "Ukuran hasil dalam piksel", side="left")
    return shot


# ------------------------------------------------------------------------------ fitting
@scene("graph.ex_scatter", "Langkah 2: scatter plot data kalibrasi dengan error bar")
def ex_scatter(st):
    w = _main(st)
    bw = _calibration(w)
    gwin = _calib_graph(w, bw, "yerror")
    st.settle()
    mw, mh = w.mdi.width(), w.mdi.height()
    _place(w, "Book1", 4, 4, 318, mh - 30)
    _place(w, gwin.graph.title, 326, 4, mw - 330, mh - 8)
    st.settle()
    gwin.fit_page()
    st.settle()
    shot = st.shot(w)
    view = bw.sheet_view()
    hh = view.horizontalHeader()
    shot.box(_in(w, hh, (hh.sectionPosition(1), 0, hh.sectionSize(1), hh.height())),
             "Kolom B(Y) dipilih (klik judul kolom)")
    shot.box(_in(w, w.subs[gwin.graph.title]), "Grafik baru Graph1")
    shot.box(_toolbar(w, "graphs2dToolbar"), "Alternatif: tombol di toolbar 2D Graphs")
    return shot


@scene("graph.fit_dialog", "Dialog Linear Fit")
def fit_dialog(st):
    from rsstatchem.ui.origin.dialogs import FitDialog

    w = _main(st)
    bw = _calibration(w)
    _calib_graph(w, bw)
    st.settle()
    choices, current, _ = w.analysis_inputs()
    dlg = FitDialog("linear", choices, w, current, graph_available=True)
    dlg.resize(500, 0)
    dlg.show()
    st.settle()
    shot = st.shot(dlg)
    shot.arrow(dlg.mode, "Recalculate: Auto, Manual, atau None", side="top")
    shot.arrow(dlg.data, "Data yang di-fit (plot di grafik aktif)", side="right")
    shot.arrow(dlg.weight, "Pembobotan, mis. Instrumental dari kolom yEr", side="right")
    shot.arrow(dlg.level, "Tingkat kepercayaan", side="right")
    shot.arrow(_in(dlg, dlg.conf), "Pita kepercayaan dan prediksi", side="left")
    shot.arrow(dlg.resid, "Plot residual di report", side="left")
    shot.arrow(dlg.to_graph, "Tambahkan garis fit ke grafik", side="left")
    return shot


@scene("graph.fit_result", "Hasil Linear Fit: garis fit dan ringkasan di grafik, "
       "report di workbook")
def fit_result(st):
    w = _main(st, size=(1100, 700))
    for d in (w.pe_dock, w.results_dock, w.messages_dock):
        d.hide()
    bw = _calibration(w)
    gwin = _calib_graph(w, bw)
    _fit(w, gwin)
    st.settle()
    mw, mh = w.mdi.width(), w.mdi.height()
    _place(w, "Book1", 4, 4, 300, mh - 30)
    _place(w, gwin.graph.title, 308, 4, mw - 312, mh - 8)
    w.show_window(gwin.graph.title)
    st.settle()
    gwin.fit_page()
    st.settle()
    shot = st.shot(w)
    # only the workspace and the Object Manager: the picture fits on the page with its text
    shot.crop(_in(w, w.mdi).united(_in(w, w.om_dock)), pad=2)
    tb = bw.tabs.tabBar()
    shot.box(_in(w, tb, tb.tabRect(1).united(tb.tabRect(2)).getRect()),
             "Sheet baru: report FitLinear1 dan data kurva FitLinearCurve1")
    shot.box(_point(w, gwin, 8.0, 0.0985 * 8 + 0.012, 30), "Garis fit (merah)")
    note = gwin.graph.layers[0].annotations[-1]
    ref = ("annotation", 0, note.id)
    r = _artist(w, gwin, ref)
    if r is not None:
        shot.box(r, "Ringkasan: persamaan, parameter ± galat, R²", pad=10)
    shot.box(w.om_dock, "Plot fit dan anotasi masuk ke Object Manager")
    return shot


@scene("graph.fit_report", "Report sheet FitLinear1 (bagian atas)")
def fit_report(st):
    w = _main(st, size=(1100, 860), docks=False)
    bw = _calibration(w)
    gwin = _calib_graph(w, bw)
    out = _fit(w, gwin, to_graph=False)
    w.show_window("Book1")
    bw.show_sheet(out.report)
    st.settle()
    w.subs["Book1"].showMaximized()
    st.settle()
    shot = st.shot(w)
    shot.crop(_in(w, w.mdi), pad=0)
    return shot


# ------------------------------------------------------------------------------ others
@scene("graph.peaks", "Peak Analyzer: tiga pita yang tumpang tindih dipisahkan dengan fit "
       "Gaussian")
def peaks(st):
    from rsstatchem.ui.origin.peak_analyzer_dialog import PeakAnalyzerDialog

    w = _main(st, size=(1000, 600))
    bw = _spectrum(w, w.active_book_window())
    bw.select_columns([1])
    choices, current, _ = w.analysis_inputs()
    dlg = PeakAnalyzerDialog(choices, w, current, w.resolve)
    dlg.resize(1040, 560)
    dlg.page = "fit"
    dlg.go(0)
    dlg.show()
    st.settle()
    dlg.preview()
    st.settle()
    shot = st.shot(dlg)
    shot.box(dlg.page_title, "Langkah wizard (Prev / Next)")
    shot.box(dlg.shape, "Fungsi puncak: Gaussian, Lorentzian, Voigt...")
    shot.box(dlg.fit_table, "Pusat, tinggi, luas, dan FWHM tiap puncak")
    shot.box(dlg.canvas, "Pratinjau: data, puncak, dan kurva fit kumulatif")
    return shot


@scene("graph.stats_menu", "Menu Statistics ▸ Hypothesis Testing")
def stats_menu(st):
    w = _main(st, size=(1100, 500))
    w.results_dock.hide()
    w.messages_dock.hide()
    bw = _calibration(w)
    _place(w, "Book1", 4, 4, 420, 330)
    st.settle()
    shot = st.shot(w)
    shot.menu("Statistics > Hypothesis Testing", highlight="Two-Sample t-Test")
    return shot


@scene("graph.matrix", "Matriks hasil Set Values dan grafik 3D Colormap Surface")
def matrix(st):
    w = _main(st, size=(1280, 640))
    for d in (w.pe_dock, w.results_dock, w.messages_dock):
        d.hide()
    mw_ = w.new_matrix()
    w.set_matrix_values(mw_, "exp(-((x-16)^2+(y-16)^2)/60)*cos((x-16)/3)")
    st.settle()
    gwin = w.plot_matrix("surface3d_colormap")
    w.hide_window("Book1")
    st.settle()
    mw, mh = w.mdi.width(), w.mdi.height()
    _place(w, mw_.book.name, 4, 4, 300, mh - 30)
    _place(w, gwin.graph.title, 308, 4, mw - 312, mh - 8)
    st.settle()
    gwin.fit_page()
    w.show_window(mw_.book.name)
    st.settle()
    shot = st.shot(w)
    shot.box(w.menuBar(), "Menu mode matriks: Plot dan Matrix")
    shot.box(w.subs[mw_.book.name], "Matrix book: nilai Z pada kisi X–Y")
    shot.box(w.subs[gwin.graph.title], "Grafik 3D; seret untuk memutar")
    return shot

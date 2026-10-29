"""Scenes for the RSM / DOE chapter (``content/20_rsm.md``).

The worked example is a Box-Behnken design in three factors (temperature, time, ethanol
concentration) for a phenolic extraction, with two responses (yield and total phenolics).
The measured values are computed from a known quadratic model plus seeded noise, so every
picture is the same on every build. After the Model step of the chapter the BC term of the
yield model is removed (it is not significant); later scenes use that reduced model.
"""
import numpy as np
from PySide6.QtCore import QRect
from PySide6.QtGui import QTextCursor, QTextDocument
from PySide6.QtWidgets import QPushButton

from guide.studio import scene

FACTORS = [{"name": "Suhu", "units": "°C", "low": 40, "high": 60},
           {"name": "Waktu", "units": "min", "low": 30, "high": 90},
           {"name": "Etanol", "units": "%", "low": 50, "high": 90}]
RESPONSES = [("Rendemen", "%"), ("Fenolik", "mg GAE/g")]
SEED = 2024
BC = (1, 2)


def demo_doc(with_data: bool = True, missing: int = 0, reduced: bool = True):
    """The example design; ``missing`` leaves the last runs (in run order) without data;
    ``reduced`` drops the BC term from the yield model (as done in the chapter)."""
    from rsstatchem.ui.dx.document import DesignDocument

    spec = {"design": "bbd", "name": "Ekstraksi fenolik", "factors": FACTORS,
            "responses": [{"name": n, "units": u} for n, u in RESPONSES],
            "options": {"center_points": 5}, "randomize": True, "seed": SEED}
    doc = DesignDocument.from_spec(spec)
    if with_data:
        X = doc.coded_matrix()
        a, b, c = X.T
        rng = np.random.default_rng(7)
        y1 = (18.5 + 1.6 * a + 2.1 * b + 0.9 * c + 0.8 * a * b - 0.3 * a * c + 0.2 * b * c
              - 1.4 * a ** 2 - 1.9 * b ** 2 - 2.3 * c ** 2 + rng.normal(0, 0.25, len(a)))
        y2 = (42 + 3.0 * a + 2.2 * b - 1.5 * c + 0.5 * a * b + 1.2 * a * c - 0.4 * b * c
              - 2.8 * a ** 2 - 1.6 * b ** 2 - 3.1 * c ** 2 + rng.normal(0, 0.6, len(a)))
        run = doc.ds.df["Run"].to_numpy(int)
        keep = run <= doc.n_runs - missing
        r1, r2 = doc.response_cols
        doc.set_values([(i, r1, round(float(v), 2)) for i, v in enumerate(y1) if keep[i]]
                       + [(i, r2, round(float(v), 1)) for i, v in enumerate(y2) if keep[i]])
        if reduced and not missing:
            from rsstatchem.core.doe.model import term_key

            terms = [t for t in doc.analyze(r1).fit.terms if tuple(t) != BC]
            s = doc.spec(r1)
            s.model, s.terms, s.reduction = "manual", [term_key(t) for t in terms], "none"
            doc.set_spec(s)
    doc.modified = False
    return doc


def window(st, doc=None, size=(1180, 740), nav=True):
    from rsstatchem.ui.dx.window import DXMainWindow

    w = DXMainWindow(st.ctx, doc)  # not registered: keeps Window > list of the reference clean
    w.resize(*size)
    if not nav:
        w.a_nav.setChecked(False)
    w.show()
    st.settle()
    return w


def span(w, first, last):
    """Window rectangle from the top-left of one widget to the bottom-right of another."""
    a = first.mapTo(w, first.rect().topLeft())
    b = last.mapTo(w, last.rect().bottomRight())
    return QRect(a, b)


def nav_item_rect(w, item):
    r = w.nav.visualItemRect(item)
    return QRect(w.nav.viewport().mapTo(w, r.topLeft()), r.size())


def nav_folder(w, i):
    """Rectangle of top-level navigation item ``i`` with all its children."""
    top = w.nav.topLevelItem(i)
    last = top.child(top.childCount() - 1) if top.childCount() else top
    a, b = nav_item_rect(w, top), nav_item_rect(w, last)
    return QRect(a.topLeft(), b.bottomRight()).adjusted(4, 3, -4, -3)


def text_rect(w, browser, text, nth=0, extend_left=None):
    """Window rectangle of the ``nth`` occurrence of ``text`` in a report (QTextBrowser)."""
    doc = browser.document()
    cur = QTextCursor(doc)
    for _ in range(nth + 1):
        cur = doc.find(text, cur, QTextDocument.FindFlag.FindCaseSensitively)
        if cur.isNull():
            raise KeyError(text)
    c1, c2 = QTextCursor(doc), QTextCursor(doc)
    c1.setPosition(cur.selectionStart())
    c2.setPosition(cur.selectionEnd())
    r1, r2 = browser.cursorRect(c1), browser.cursorRect(c2)
    vp = browser.viewport()
    r = QRect(vp.mapTo(w, r1.topLeft()), vp.mapTo(w, r2.bottomRight()))
    if extend_left is not None:
        r.setLeft(extend_left)
    return r.adjusted(-3, -1, 3, 1)


def analysis(st, size=(940, 640), tab="ANOVA", reduced=True):
    w = window(st, demo_doc(reduced=reduced), size, nav=False)
    col = w.doc.response_cols[0]
    view = w.show_node(f"analysis:{col}")
    st.settle()
    view.set_tab(tab)
    st.settle()
    return w, view


# ------------------------------------------------------------------------------ start
@scene("rsm.start", "Halaman awal modul RSM / DOE saat belum ada desain yang dibuka")
def start(st):
    w = window(st, None, (1000, 600))
    page = w.start_page
    shot = st.shot(w)
    btns = {b.text().strip(): b for b in page.findChildren(QPushButton)}
    shot.box(btns["New Design..."], "Membuat desain baru (wizard New Design)")
    shot.box(btns["Open Design..."], "Membuka berkas desain `.rsx`")
    shot.box(btns["Import Data Set..."], "Membuat desain dari data lama (CSV/Excel)")
    tut = [b for t, b in btns.items()
           if t not in ("New Design...", "Open Design...", "Import Data Set...")]
    shot.box(span(w, tut[0], tut[-1]), "Data tutorial: contoh desain lengkap dengan datanya")
    shot.box(page.recent, "Desain yang terakhir dibuka")
    return shot


# ------------------------------------------------------------------------------ wizard
def _wizard(st, size=(1260, 600)):
    from rsstatchem.ui.dx.wizard import NewDesignWizard

    w = window(st, None, (1060, 620))
    wiz = NewDesignWizard(w, family="rsm", design="bbd")
    wiz.seed = SEED
    wiz.set_factors(FACTORS)
    wiz.resize(*size)
    wiz.show()
    st.settle()
    wiz._update_summary()
    st.settle()
    return w, wiz


@scene("rsm.wizard", "Wizard New Design, halaman 1: jenis desain, faktor, dan opsi")
def wizard(st):
    _, wiz = _wizard(st)
    shot = st.shot(wiz)
    shot.box(wiz.tree, "Keluarga desain (Factorial, Response Surface, Mixture, ...)")
    shot.box(wiz.list, "Desain dalam keluarga yang dipilih")
    shot.box(wiz.k_spin, "Jumlah faktor")
    shot.box(wiz.table, "Tabel faktor: nama, satuan, tipe, minimum, maksimum")
    shot.box(wiz.options_box, "Opsi desain (titik pusat, blok, replikasi)")
    shot.box(wiz.summary, "Ringkasan: jumlah run yang akan dibuat")
    shot.box(wiz.next_btn, "Lanjut ke halaman respons")
    return shot


@scene("rsm.wizard_responses", "Wizard New Design, halaman 2: nama dan satuan respons")
def wizard_responses(st):
    _, wiz = _wizard(st)
    wiz.go(1)
    wiz.set_responses(RESPONSES)
    wiz.name_edit.setText("Ekstraksi fenolik")
    wiz.resize(1060, 360)
    st.settle()
    shot = st.shot(wiz)
    shot.box(wiz.r_spin, "Jumlah respons")
    shot.box(wiz.name_edit, "Nama desain")
    shot.box(wiz.responses, "Nama dan satuan setiap respons")
    shot.box(wiz.finish_btn, "Membuat desain")
    return shot


# ------------------------------------------------------------------------------ window
@scene("rsm.window", "Jendela kerja RSM / DOE dengan desain Box-Behnken yang sudah terisi data")
def main_window(st):
    w = window(st, demo_doc(), (1060, 640))
    page = w.page("design")
    shot = st.shot(w)
    shot.box(w.menuBar(), "Menu bar", pad=0)
    shot.box(w.toolbar, "Toolbar: perintah yang paling sering dipakai", pad=0)
    shot.box(nav_folder(w, 0), "Node **Design**: layout, ringkasan, evaluasi, catatan",
             pad=0)
    shot.box(nav_folder(w, 1), "Node **Analysis**: satu node per respons", pad=0)
    shot.box(nav_folder(w, 2), "Node **Optimization**: numerik, grafis, POE", pad=0)
    shot.box(nav_folder(w, 3), "Node **Post Analysis**: prediksi, konfirmasi, koefisien",
             pad=0)
    shot.box(page.hint, "Petunjuk untuk layar yang sedang dibuka")
    shot.box(page.view, "Design layout: tabel run percobaan dan kolom respons")
    shot.box(w.statusBar(), "Status bar", pad=0)
    return shot


@scene("rsm.layout", "Design layout: kolom faktor dan sel respons (kuning) yang diisi")
def layout(st):
    w = window(st, demo_doc(missing=4), (900, 620))
    page = w.page("design")
    view = page.view
    shot = st.shot(w)
    shot.crop(w.stack)
    m = view.layout_model
    hh = view.horizontalHeader()
    vp = view.viewport()
    last = m.rowCount() - 1
    bottom = vp.mapTo(w, vp.rect().topLeft()).y() + view.rowViewportPosition(last) \
        + view.rowHeight(last)

    def cols_rect(c0, c1):
        top = hh.mapTo(w, hh.rect().topLeft())
        x0 = top.x() + hh.sectionViewportPosition(c0)
        x1 = top.x() + hh.sectionViewportPosition(c1) + hh.sectionSize(c1)
        return QRect(x0 + 2, top.y() + 2, x1 - x0 - 4, bottom - top.y() - 4)

    kinds = [k for k, _ in m.columns]
    shot.box(cols_rect(kinds.index("std"), kinds.index("run")),
             "**Std** (urutan standar) dan **Run** (urutan pengerjaan acak)", pad=0)
    f = [i for i, k in enumerate(kinds) if k == "factor"]
    shot.box(cols_rect(f[0], f[-1]), "Faktor: pengaturan setiap run", pad=0)
    r = [i for i, k in enumerate(kinds) if k == "response"]
    shot.box(cols_rect(r[0], r[-1]), "Respons: ketik atau tempel hasil pengukuran", pad=0)
    shot.arrow(page.hint, "Respons yang belum lengkap disebutkan di sini", side="top")
    return shot


# ------------------------------------------------------------------------------ analysis
@scene("rsm.fit_summary", "Tab Fit Summary: perbandingan model dan model yang disarankan")
def fit_summary(st):
    w, view = analysis(st, tab="Fit Summary", reduced=False)
    page = view.fit_page
    shot = st.shot(w)
    shot.crop(w.stack)
    shot.box(span(w, view.strip.buttons[0], view.strip.buttons[-1]),
             "Tab analisis, dikerjakan dari kiri ke kanan")
    shot.box(span(w, view.analysis_box, view.remove_analysis_btn),
             "Beberapa analisis untuk satu respons (+ / −)")
    shot.box(page.label, "Model yang disarankan")
    shot.box(page.use, "Memakai model yang disarankan")
    shot.box(text_rect(w, page.report, "Suggested", extend_left=page.report.mapTo(
        w, page.report.rect().topLeft()).x() + 6), "Baris model yang disarankan (disorot)")
    return shot


@scene("rsm.model", "Tab Model: suku BC dikeluarkan dari model (M = masuk model, e = galat)")
def model_tab(st):
    w, view = analysis(st, tab="Model", reduced=False)
    page = view.model_page
    page._clicked(page.candidates.index(BC), 0)
    st.settle()
    shot = st.shot(w)
    shot.crop(w.stack)
    shot.box(page.order, "Orde model (Linear, 2FI, Quadratic, ...)")
    shot.box(span(w, page.criterion, page.alpha), "Pengaturan Auto Select")
    shot.box(page.table, "Daftar suku; klik kolom pertama untuk mengganti M / e")
    row = page.candidates.index(BC)
    vp = page.table.viewport()
    top = vp.mapTo(w, vp.rect().topLeft())
    shot.box(QRect(top.x(), top.y() + page.table.rowViewportPosition(row),
                   sum(page.table.columnWidth(c) for c in range(3)), page.table.rowHeight(row)),
             "BC tidak signifikan (p > 0,10), dikeluarkan dari model (e)")
    shot.box(page.status, "Model saat ini")
    return shot


@scene("rsm.anova", "Tab ANOVA untuk model kuadratik tereduksi")
def anova(st):
    w, view = analysis(st, size=(940, 720), tab="ANOVA")
    page = view.anova_page
    rep = page.report
    left = rep.mapTo(w, rep.rect().topLeft()).x() + 8
    shot = st.shot(w)
    shot.crop(w.stack)
    shot.box(text_rect(w, rep, "significant", extend_left=left),
             "Model signifikan (p < 0,05)")
    shot.box(text_rect(w, rep, "not significant", extend_left=left),
             "Lack of Fit tidak signifikan: model cocok")
    stats = text_rect(w, rep, "Std. Dev.").united(text_rect(w, rep, "BIC"))
    stats.setRight(text_rect(w, rep, "Durbin-Watson").right() + 72)
    shot.box(stats, "Fit Statistics: R², Adjusted/Predicted R², Adeq Precision")
    return shot


@scene("rsm.diagnostics", "Tab Diagnostics: plot normal residual")
def diagnostics(st):
    w, view = analysis(st, tab="Diagnostics")
    page = view.diag_page
    shot = st.shot(w)
    shot.crop(w.stack)
    shot.box(page.list, "Pilihan plot diagnostik")
    shot.box(page.residual, "Jenis residual")
    shot.box(page.graph, "Plot yang dipilih")
    return shot


@scene("rsm.model_graphs", "Tab Model Graphs: plot kontur dengan bendera prediksi")
def model_graphs(st):
    w, view = analysis(st, tab="Model Graphs")
    page = view.graph_page
    st.settle()
    page.graph.add_flag(55.0, 82.0)
    st.settle()
    shot = st.shot(w)
    shot.crop(w.stack)
    btns = list(page.buttons.values())
    shot.box(span(w, btns[0], btns[-1]), "Jenis grafik")
    shot.box(page.tool, "Factors Tool: faktor pada sumbu X1/X2 dan nilai faktor lain")
    shot.box(page.graph.canvas, "Grafik; klik plot kontur untuk memasang bendera prediksi")
    page.set_graph("surface")
    st.settle()
    page.graph.set_view(28, -130)
    st.settle()
    shot2 = st.shot(w)
    shot2.crop(page.graph.canvas, pad=0)
    return [shot, shot2]


# ------------------------------------------------------------------------------ optimization
def numerical(st, size=(980, 640)):
    w = window(st, demo_doc(), size, nav=False)
    page = w.show_node("numerical")
    st.settle()
    r1, r2 = w.doc.response_cols
    page.set_goal(r1, "maximize", lower=15, upper=19.5)
    page.set_goal(r2, "maximize", lower=38, upper=44)
    st.settle()
    return w, page


@scene("rsm.numerical", "Optimization ▸ Numerical, tab Criteria: tujuan setiap respons")
def numerical_criteria(st):
    w, page = numerical(st, (1100, 720))
    shot = st.shot(w)
    shot.crop(w.stack)
    shot.box(span(w, page.strip.buttons[0], page.strip.buttons[-1]),
             "Criteria ▸ Solutions ▸ Graphs")
    shot.box(page.table, "Tujuan (Goal), batas bawah/atas, bobot, dan Importance")
    shot.box(page.table.parentWidget().widget(1),
             "Opsi pencarian dan variasi faktor (POE)")
    return shot


@scene("rsm.solutions", "Tab Solutions: solusi dengan desirability tertinggi dan ramps")
def numerical_solutions(st):
    w, page = numerical(st)
    page.strip.set_current(1)
    st.settle()
    shot = st.shot(w)
    shot.crop(w.stack)
    shot.box(page.solve_btn, "Mencari ulang solusi")
    shot.box(span(w, page.view_buttons[0], page.view_buttons[-1]),
             "Tampilan: Ramps, Bar Graph, Report")
    shot.box(page.sol_table, "Daftar solusi, diurutkan menurut desirability")
    shot.box(page.sol_graph, "Ramps: posisi setiap faktor dan respons pada solusi terpilih")
    return shot


@scene("rsm.graphical", "Optimization ▸ Graphical: overlay plot, daerah kuning memenuhi semua "
       "batas")
def graphical(st):
    w, num = numerical(st, (960, 560))
    best = num.solve()[0]["x"]
    page = w.show_node("graphical")
    r1, r2 = w.doc.response_cols
    page.set_limits(r1, 17.5, None)
    page.set_limits(r2, 40, None)
    page.strip.set_current(1)
    page.tool.set_point(best)
    st.settle()
    shot = st.shot(w)
    shot.crop(w.stack)
    shot.box(page.tool, "Factors Tool: sumbu dan nilai faktor ketiga")
    shot.box(page.graph, "Daerah kuning: semua batas respons terpenuhi; tanda × = titik "
             "Factors Tool")
    return shot


@scene("rsm.prediction", "Post Analysis ▸ Point Prediction pada solusi optimasi")
def prediction(st):
    w, page = numerical(st, (1180, 580))
    page.solve()
    pp = w.show_node("prediction")
    pp.use_optimum()
    st.settle()
    shot = st.shot(w)
    shot.crop(w.stack)
    shot.box(pp.tool, "Factors Tool: atur nilai faktor")
    shot.box(pp.opt_btn, "Pakai solusi optimasi nomor 1")
    shot.box(pp.report, "Prediksi, SE, CI, PI, dan TI setiap respons")
    return shot


# ------------------------------------------------------------------------------ menus
@scene("rsm.augment_menu", "Menu Design Tools ▸ Augment Design")
def augment_menu(st):
    w = window(st, demo_doc(), (1000, 560))
    shot = st.shot(w)
    shot.menu("Design Tools > Augment Design")
    return shot

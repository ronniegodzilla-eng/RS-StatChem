"""Scenes for the introduction and installation chapters."""
from PySide6.QtWidgets import QPushButton

from guide.studio import scene


@scene("intro.launcher_plain", "Layar pemilih modul (Module Launcher)")
def launcher_plain(st):
    w = st.ctx.show_launcher()
    w.resize(1080, 640)
    return st.shot(w, title="RS-StatChem")


@scene("intro.launcher", "Module Launcher: pilih modul atau buka berkas terakhir")
def launcher(st):
    w = st.ctx.show_launcher()
    w.resize(1080, 640)
    st.settle()
    shot = st.shot(w, title="RS-StatChem")
    tiles = w.tiles
    shot.box(tiles["stats"], "**Statistics**: analisis data gaya SPSS")
    shot.box(tiles["rsm"], "**RSM / DOE**: rancangan percobaan gaya Design-Expert")
    shot.box(tiles["graph"], "**Graph & Analysis**: grafik dan fitting gaya OriginLab")
    shot.box(tiles["chem"], "**Chemical Drawing**: struktur dan reaksi gaya ChemDraw")
    shot.box(tiles["image"], "**Image Analysis**: pengolahan citra gaya ImageJ")
    shot.box(w.recent, "Berkas terakhir: klik dua kali untuk membuka lagi")
    open_btn = next(b for b in w.findChildren(QPushButton) if "Open File" in b.text())
    shot.arrow(open_btn, "Membuka berkas apa pun; modulnya dipilih otomatis", side="right")
    return shot


@scene("intro.window_modules", "Menu Window ▸ Modules: berpindah atau membuka modul lain")
def window_modules(st):
    w = st.module("stats", size=(980, 560))
    shot = st.shot(w)
    shot.menu("Window > Modules")
    return shot


@scene("intro.update", "Pemberitahuan versi baru (Software Update)")
def update(st):
    from rsstatchem import __version__
    from rsstatchem.ui.update_dialog import UpdateDialog
    from rsstatchem.updates import UpdateInfo

    major, minor, patch = (__version__.split(".") + ["0", "0"])[:3]
    newer = f"{major}.{minor}.{int(patch) + 1}"
    info = UpdateInfo(newer, f"v{newer}",
                      "Perbaikan dan fitur baru.\n\n- Contoh catatan rilis\n- Daftar perubahan",
                      "https://github.com/ronniegodzilla-eng/RS-StatChem-releases/releases",
                      None, "")
    dlg = UpdateDialog(info, st.ctx.settings)
    dlg.resize(560, 420)
    shot = st.shot(dlg)
    buttons = {b.text(): b for b in dlg.findChildren(QPushButton)}
    shot.box(buttons["Download"], "Mengunduh versi baru")
    shot.box(buttons["Remind Me Later"], "Ingatkan lagi nanti")
    shot.box(buttons["Skip This Version"], "Lewati versi ini")
    shot.box(dlg.auto, "Pemeriksaan otomatis (setiap ±20 jam saat aplikasi dibuka)")
    return shot


@scene("intro.about", "Help ▸ About: versi, hak cipta dan lisensi")
def about(st):
    from rsstatchem.ui.dialogs.about import AboutDialog

    dlg = AboutDialog()
    return st.shot(dlg)

"""Window and Help menu commands of the ImageJ bar (Phase 0; WP-A may extend them).

They are UI-only (``ui_action``), not macro commands: Show All, Main Window, Put Behind,
Cascade, Tile, ImageJ Website..., Macro Functions..., Check for Updates..., About
RS-StatChem....
"""
from __future__ import annotations

from rsstatchem.core.imaging.commands import Command

IMAGEJ_WEBSITE = "https://imagej.net/ij/"
MACRO_FUNCTIONS = "https://imagej.net/ij/developer/macro/functions.html"


def _windows():
    from rsstatchem.ui.imagej.windows import WINDOWS
    return [w for w in WINDOWS if w.isVisible()]


def _show_all(bar, _cmd) -> None:
    for w in _windows():
        w.raise_()
    bar.raise_()


def _main_window(bar, _cmd) -> None:
    bar.show()
    bar.raise_()
    bar.activateWindow()


def _put_behind(bar, _cmd) -> None:
    wins = _windows()
    if wins:
        wins[0].lower()
        wins[1 % len(wins)].raise_()
        wins[1 % len(wins)].activateWindow()


def _cascade(bar, _cmd) -> None:
    x0, y0 = bar.x(), bar.frameGeometry().bottom() + 10
    for i, w in enumerate(_windows()):
        w.move(x0 + 22 * i, y0 + 22 * i)
        w.raise_()


def _tile(bar, _cmd) -> None:
    wins = _windows()
    if not wins:
        return
    screen = bar.screen().availableGeometry()
    top = bar.frameGeometry().bottom() + 6
    cols = max(1, int(len(wins) ** 0.5 + 0.999))
    rows = (len(wins) + cols - 1) // cols
    w_each = screen.width() // cols
    h_each = max(120, (screen.bottom() - top) // rows)
    for i, w in enumerate(wins):
        w.setGeometry(screen.left() + (i % cols) * w_each, top + (i // cols) * h_each,
                      w_each - 8, h_each - 30)


def _open_url(url: str):
    def action(bar, _cmd) -> None:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl(url))
    return action


def _updates(bar, _cmd) -> None:
    if bar.ctx is not None:
        bar.ctx.check_for_updates(manual=True)


def _about(bar, _cmd) -> None:
    if bar.ctx is not None:
        bar.ctx.about()


def register(registry) -> None:
    for label, action in (("Show All", _show_all), ("Main Window [return]", _main_window),
                          ("Put Behind [tab]", _put_behind), ("Cascade", _cascade),
                          ("Tile", _tile), ("ImageJ Website...", _open_url(IMAGEJ_WEBSITE)),
                          ("Macro Functions...", _open_url(MACRO_FUNCTIONS)),
                          ("Check for Updates...", _updates),
                          ("About RS-StatChem...", _about)):
        registry.register(Command(label, needs_image=False, undoable=False, recordable=False,
                                  ui_action=action), replace=True)

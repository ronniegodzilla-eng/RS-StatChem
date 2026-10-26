"""Graph & Analysis module: an OriginLab-style workspace (workbooks, graphs, fitting).

See :mod:`rsstatchem.ui.origin.window` for the main window and :mod:`rsstatchem.ui.modules`
for the module contract.
"""
from __future__ import annotations


def open_workspace(ctx, path: str | None = None):
    """Open a new Graph & Analysis main window (with Book1, or the given ``.rsop`` / ``.rsg``
    / data file) and return it. Never runs a modal loop."""
    from rsstatchem.ui.origin.window import OriginMainWindow

    win = OriginMainWindow(ctx)
    win.new_book()
    win.set_modified(False)
    win.show()
    if path:
        win.quiet = True   # problems with the file go to the Messages Log
        if win.open_path(path) and not path.lower().endswith(".rsop"):
            win.set_modified(False)
        win.quiet = False
    return win

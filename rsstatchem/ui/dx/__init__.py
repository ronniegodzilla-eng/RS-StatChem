"""RSM / DOE module: a Design-Expert-style workspace (see :mod:`rsstatchem.ui.modules`).

``open_workspace(ctx, path=None)`` returns the shown main window at once (no modal loop):
with a ``.rsx`` path it opens that design, otherwise it shows the start page and, once the
event loop runs, the New Design wizard (non-modal).
"""
from __future__ import annotations


def open_workspace(ctx, path: str | None = None, wizard: bool = True):
    from rsstatchem.ui.dx.document import DesignDocument
    from rsstatchem.ui.dx.window import DXMainWindow, open_wizard_later

    doc, error = None, None
    if path:
        try:
            doc = DesignDocument.load(path)
            ctx.add_recent(path)
        except (OSError, ValueError, KeyError) as exc:
            error = f"Could not open {path}:\n{exc}"
    win = DXMainWindow(ctx, doc)
    win.show()
    if error:
        win.error(error)
    elif doc is None and wizard:
        open_wizard_later(win)
    return win

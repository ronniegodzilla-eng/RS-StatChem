"""Chemical Drawing module (ChemDraw-style workspace); see :mod:`rsstatchem.ui.modules`.

:func:`open_workspace` opens a ChemDraw-like document window
(:class:`~rsstatchem.ui.chemdraw.window.ChemDrawWindow`): Main Toolbar palette, General and
Style toolbars, a page view of the document, the Analysis and Chemical Properties windows,
the Pen tool and Curves menu, orbitals, the Mass Fragmentation tool, Structure Perspective (3D),
name <-> structure conversion (offline list, offline name database, OPSIN, optional PubChem),
and .rsc / CDXML / MOL / SDF / RXN files. The drawing engine is :mod:`rsstatchem.ui.chem_editor` on top of
:mod:`rsstatchem.core.chem_draw`.
"""
from __future__ import annotations

__all__ = ["open_workspace"]


def open_workspace(ctx, path: str | None = None):
    """Open a new Chemical Drawing window (optionally with a file) and return it, shown.
    Never runs a modal loop: problems reading ``path`` are reported once the event loop
    runs."""
    from rsstatchem.ui.chem_editor import _WINDOWS
    from rsstatchem.ui.chemdraw.window import ChemDrawWindow

    win = ChemDrawWindow(ctx)
    _WINDOWS.append(win)        # cleared with the other editor windows on shutdown
    if path:
        win.open_file(path, deferred_errors=True)
    win.show()
    win.raise_()
    win.activateWindow()
    return win

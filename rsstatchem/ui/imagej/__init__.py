"""Image Analysis module (ImageJ 1.54 / Fiji style); see :mod:`rsstatchem.ui.modules` and
``docs/IMAGEJ_CONTRACTS.md``.

:func:`open_workspace` returns the single ImageJ bar (menus, tools, status line), creating it on
first use; a ``path`` opens in it. Images, Results, Log, ROI Manager and plot windows are
separate top-level windows kept in :data:`rsstatchem.ui.imagej.windows.WINDOWS`. The core
(no Qt) is :mod:`rsstatchem.core.imaging`.
"""
from __future__ import annotations

__all__ = ["open_workspace", "selftest"]


def _alive(widget) -> bool:
    try:
        import shiboken6
        return shiboken6.isValid(widget)
    except ImportError:  # pragma: no cover
        return True


def open_workspace(ctx, path: str | None = None):
    """Show the ImageJ bar (creating it once per app) and return it. Never runs a modal loop:
    problems reading ``path`` are reported once the event loop runs."""
    from rsstatchem.ui.imagej.bar import ImageJBar
    from rsstatchem.ui.imagej.windows import BARS

    BARS[:] = [b for b in BARS if _alive(b)]
    bar = BARS[0] if BARS else None
    if bar is None or bar.ctx is not ctx:
        bar = ImageJBar(ctx)
        BARS[:] = [bar]
    bar.show()
    bar.raise_()
    bar.activateWindow()
    if path:
        bar.open_path(path, deferred_errors=True)
    return bar


def selftest() -> None:
    """Headless checks of the core contracts and the area modules' own selftests."""
    from rsstatchem.core.imaging.selftest import run
    run()

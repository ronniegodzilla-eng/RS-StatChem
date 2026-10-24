"""Deterministic disposal of Qt windows and of the QApplication itself.

Why this exists: a widget that is still alive when the interpreter exits is deleted by
PySide6's atexit hook (``PySide::destroyQCoreApplication``), which walks every Python-owned
QObject in hash-table order and calls its C++ destructor. Many of our widgets are added to a
layout before that layout is installed on a window, so shiboken still believes it owns them
although Qt has reparented them; the hook then deletes children and parents in arbitrary order
and the process crashes after the work is done (SIGSEGV / access violation, "no Python frame").

So nothing Qt may be left to that hook: the app (``rsstatchem.app.main``) and the test suite
(``tests/conftest.py``) delete their windows and the QApplication explicitly, while Python and Qt
are fully alive. No Qt import at module level, so importing this module is free.
"""
from __future__ import annotations

import gc
import sys

# Module-level lists that keep Python references to open windows (module, attribute).
_WINDOW_REGISTRIES = (
    ("rsstatchem.ui.graph.window", "WINDOWS"),
    ("rsstatchem.ui.chem_editor", "_WINDOWS"),
)


def _clear_registries() -> None:
    for module_name, attr in _WINDOW_REGISTRIES:
        registry = getattr(sys.modules.get(module_name), attr, None)
        if registry:
            registry.clear()


def _clear_icon_caches() -> None:
    """Cached QIcons/QPixmaps must not outlive the QGuiApplication."""
    for module_name in ("rsstatchem.ui.icons", "rsstatchem.ui.graph.gicons"):
        fn = getattr(sys.modules.get(module_name), "icon", None)
        if hasattr(fn, "cache_clear"):
            fn.cache_clear()
    cache = getattr(sys.modules.get("rsstatchem.ui.chem_editor.icons"), "_CACHE", None)
    if cache:
        cache.clear()


def flush_deferred_deletes() -> None:
    """Run pending ``deleteLater()`` calls now.

    Qt only honours them when control returns to an event loop, which never happens in tests or
    after ``app.exec()`` has returned; ``processEvents()`` does not run them either.
    """
    from PySide6.QtCore import QCoreApplication, QEvent

    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def dispose_windows(app=None) -> int:
    """Delete every parentless top-level widget (open, hidden or merely leaked) without asking
    to save; children, popups and child dialogs go with their parent. Returns how many."""
    from PySide6.QtWidgets import QApplication

    app = app or QApplication.instance()
    if app is None:
        return 0
    _clear_registries()
    count = 0
    for _ in range(5):  # deleting a window can, rarely, create or release another top level
        windows = [w for w in app.topLevelWidgets() if w.parentWidget() is None]
        if not windows:
            break
        for w in windows:
            w.deleteLater()
        count += len(windows)
        flush_deferred_deletes()
    return count


def shutdown_application(app=None) -> None:
    """Delete all windows, then the QApplication, while the interpreter is still intact."""
    from PySide6.QtCore import QThreadPool
    from PySide6.QtWidgets import QApplication

    app = app or QApplication.instance()
    if app is None:
        return
    dispose_windows(app)
    if app.platformName() == "offscreen":
        # The offscreen platform (tests, CI) crashes on teardown while its clipboard holds
        # data. Real platforms hand the data to the OS clipboard, so it survives quitting.
        app.clipboard().clear()
    _clear_icon_caches()
    gc.collect()  # Python-owned QObjects (QSettings, documents...) held only by dead cycles
    flush_deferred_deletes()
    QThreadPool.globalInstance().waitForDone()  # joins Qt's idle pooled worker threads
    shutdown = getattr(app, "shutdown", None)
    if shutdown is not None:
        shutdown()  # PySide6: deletes the C++ QApplication and invalidates the wrapper
    else:  # pragma: no cover - very old PySide6
        import shiboken6

        shiboken6.delete(app)

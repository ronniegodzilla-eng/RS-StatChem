"""Window registries and the bridge from :class:`~rsstatchem.core.imaging.session.Session`
events to windows. **Owned by WP-A after Phase 0.**

* :data:`BARS` holds the single ImageJ bar (the window registered with the app context).
* :data:`WINDOWS` holds every other top-level window of the module (images, Results, Log,
  ROI Manager, plots, ...). Both lists are cleared by ``ui/qt_lifecycle`` on shutdown.
* :data:`TABLE_WINDOW_FACTORY` / :data:`LOG_WINDOW_FACTORY`: callables ``(bar, table|None)``
  returning a window; WP-C sets the table factory from ``frames/results_window.py`` and WP-A
  the log factory from ``frames/log_window.py``. Phase 0 falls back to a plain text window.
"""
from __future__ import annotations

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QMainWindow, QPlainTextEdit

BARS: list = []
WINDOWS: list = []
TABLE_WINDOW_FACTORY = None
LOG_WINDOW_FACTORY = None


class TextWindow(QMainWindow):
    """Minimal read-only text window (fallback for Results and Log)."""

    def __init__(self, title: str, text: str = ""):
        super().__init__()
        self.setWindowTitle(title)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setFont(QFont("Menlo", 11))
        self.text.setPlainText(text)
        self.setCentralWidget(self.text)
        self.resize(520, 300)
        self.table = None

    def set_table(self, table) -> None:
        self.table = table
        self.text.setPlainText(table.to_text("\t"))

    def append(self, line: str) -> None:
        self.text.appendPlainText(line)

    def closeEvent(self, event) -> None:  # noqa: N802
        event.accept()
        unregister(self)


def register(window) -> None:
    if window not in WINDOWS:
        WINDOWS.append(window)


def unregister(window) -> None:
    if window in WINDOWS:
        WINDOWS.remove(window)


def image_windows() -> list:
    from rsstatchem.ui.imagej.image_window import ImageWindow
    return [w for w in WINDOWS if isinstance(w, ImageWindow)]


def window_for(doc):
    return next((w for w in image_windows() if w.doc is doc), None)


class WindowManager:
    """Shows/updates windows for session events (one per bar)."""

    def __init__(self, bar):
        self.bar = bar
        s = bar.session
        s.on("image_added", self.image_added)
        s.on("image_closed", self.image_closed)
        s.on("image_updated", self.image_updated)
        s.on("active_changed", self.active_changed)
        s.on("table_updated", self.table_updated)
        s.on("log", self.log)
        s.on("status", bar.show_status)
        s.on("progress", bar.show_progress)
        self.tables: dict[str, object] = {}
        self.log_window = None

    def image_added(self, doc) -> None:
        from rsstatchem.ui.imagej.image_window import ImageWindow

        win = ImageWindow(doc, self.bar)
        register(win)
        win.show()

    def image_closed(self, doc) -> None:
        win = window_for(doc)
        if win is not None:
            unregister(win)
            win.bar = None
            win.close()

    def image_updated(self, doc) -> None:
        win = window_for(doc)
        if win is not None:
            win.refresh()

    def active_changed(self, doc) -> None:
        win = window_for(doc) if doc is not None else None
        if win is not None and not win.isActiveWindow():
            win.raise_()

    def table_updated(self, table) -> None:
        win = self.tables.get(table.title)
        if win is None or win not in WINDOWS:
            factory = TABLE_WINDOW_FACTORY
            win = factory(self.bar, table) if factory else TextWindow(table.title)
            self.tables[table.title] = win
            register(win)
        win.set_table(table)
        win.show()

    def log(self, line: str) -> None:
        if self.log_window is None or self.log_window not in WINDOWS:
            factory = LOG_WINDOW_FACTORY
            self.log_window = factory(self.bar, None) if factory else TextWindow("Log")
            register(self.log_window)
        self.log_window.append(line)
        self.log_window.show()

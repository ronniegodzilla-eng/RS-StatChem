"""Window registries and the bridge from :class:`~rsstatchem.core.imaging.session.Session`
events to windows. **Owned by WP-A.**

* :data:`BARS` holds the single ImageJ bar (the window registered with the app context).
* :data:`WINDOWS` holds every other top-level window of the module (images, Results, Log,
  ROI Manager, plots, ...). Both lists are cleared by ``ui/qt_lifecycle`` on shutdown.
* :data:`TABLE_WINDOW_FACTORY` / :data:`LOG_WINDOW_FACTORY`: callables ``(bar, table|None)``
  returning a window; WP-C sets the table factory from ``frames/results_window.py``; the log
  factory is ``frames/log_window.make_log_window``. A plain :class:`TextWindow` is the
  fallback for tables.

Session events handled here: image_added/closed/updated, active_changed, table_updated, log,
status, progress, plus "text_opened" and "macro_opened" (File > Open of a text file or
macro: a text window / macro editor) and "position_changed".
"""
from __future__ import annotations

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QMainWindow, QPlainTextEdit

BARS: list = []
WINDOWS: list = []
TABLE_WINDOW_FACTORY = None
LOG_WINDOW_FACTORY = None


class TextWindow(QMainWindow):
    """Minimal read-only text window (fallback for Results)."""

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
        s.on("text_opened", self.text_opened)
        s.on("macro_opened", self.macro_opened)
        s.on("options_changed", self.options_changed)
        self.tables: dict[str, object] = {}
        self.log_window = None
        self.cascade_index = 0

    def image_added(self, doc) -> None:
        from rsstatchem.ui.imagej.image_window import ImageWindow

        win = ImageWindow(doc, self.bar)
        register(win)
        self._place(win)
        win.show()

    def _place(self, win) -> None:
        """New image windows go below the bar, cascading like ImageJ's ImageWindow."""
        bar = self.bar
        try:
            x0, y0 = bar.x(), bar.frameGeometry().bottom() + 8
        except RuntimeError:
            return
        n = len(image_windows()) - 1
        win.move(x0 + 24 * (n % 12), y0 + 24 * (n % 12))

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
            if factory is None:
                from rsstatchem.ui.imagej.frames.log_window import make_log_window
                factory = make_log_window
            self.log_window = factory(self.bar, None)
            register(self.log_window)
        self.log_window.append(line)
        self.log_window.show()

    def text_opened(self, arg) -> None:
        from rsstatchem.ui.imagej.frames.text_window import TextWindow as TW
        title, text, path = arg
        win = TW(title, text, self.bar, editable=True, path=path)
        register(win)
        win.show()

    def macro_opened(self, arg) -> None:
        from rsstatchem.ui.imagej.frames.text_window import TextWindow as TW
        title, text, path = arg
        win = TW(title, text, self.bar, macro=True, path=path)
        register(win)
        win.show()

    def options_changed(self, what) -> None:
        o = self.bar.session.options
        self.bar.toolbar.set_colors(o.foreground, o.background)
        for win in image_windows():
            win.canvas.selection_color = o.selection
            win.canvas.interpolate = o.interpolate_zoomed
            win.canvas.repaint_overlay()

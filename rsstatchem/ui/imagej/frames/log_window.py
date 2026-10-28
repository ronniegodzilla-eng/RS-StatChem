"""The Log window (``IJ.log``): appended lines, "\\Clear" empties it, File > Save As, Edit >
Copy/Clear. ``windows.LOG_WINDOW_FACTORY`` is set to :func:`make_log_window`. **Owned by
WP-A.**"""
from __future__ import annotations

from rsstatchem.ui.imagej.frames.text_window import TextWindow


class LogWindow(TextWindow):
    def __init__(self, bar=None):
        super().__init__("Log", "", bar, editable=False)
        self.resize(480, 300)

    def append(self, line: str) -> None:
        if line.startswith("\\Update:"):
            doc = self.text.document()
            last = doc.lastBlock()
            from PySide6.QtGui import QTextCursor
            cur = QTextCursor(last)
            cur.select(QTextCursor.SelectionType.BlockUnderCursor)
            cur.removeSelectedText()
            line = line[len("\\Update:"):]
        super().append(line)


def make_log_window(bar, _unused=None) -> LogWindow:
    win = LogWindow(bar)
    if bar is not None:
        for line in bar.session.log_lines[:-1]:
            win.append(line)
    return win

"""Plugins > Macros > Record... (``ij/plugin/frame/Recorder.java``): while open, every
command, ``open()``/``saveAs()`` and tool selection is written in ImageJ macro or Python
syntax; "Create" opens the recording in a macro editor. **Owned by WP-A.**"""
from __future__ import annotations

from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
                               QPlainTextEdit, QPushButton, QVBoxLayout, QWidget)

from rsstatchem.core.imaging.commands import Recorder


class RecorderWindow(QMainWindow):
    def __init__(self, bar):
        super().__init__()
        self.bar = bar
        self.setWindowTitle("Recorder")
        central = QWidget()
        lay = QVBoxLayout(central)
        row = QHBoxLayout()
        row.addWidget(QLabel("Record:"))
        self.language = QComboBox()
        self.language.addItems(["Macro", "Python"])
        row.addWidget(self.language)
        row.addWidget(QLabel("Name:"))
        self.name = QLineEdit("Macro.ijm")
        row.addWidget(self.name, 1)
        self.create = QPushButton("Create")
        row.addWidget(self.create)
        self.clear_btn = QPushButton("Clear")
        row.addWidget(self.clear_btn)
        lay.addLayout(row)
        self.text = QPlainTextEdit()
        self.text.setFont(QFont("Menlo", 12))
        lay.addWidget(self.text, 1)
        self.setCentralWidget(central)
        self.resize(620, 280)
        self.recorder = Recorder("Macro")
        self.recorder.listeners.append(self._append)
        bar.session.recorder = self.recorder
        self.language.currentTextChanged.connect(self._language_changed)
        self.create.clicked.connect(self.create_macro)
        self.clear_btn.clicked.connect(self.clear)

    def _append(self, line: str) -> None:
        self.text.appendPlainText(line)

    def _language_changed(self, lang: str) -> None:
        self.recorder.language = lang
        stem = self.name.text().rsplit(".", 1)[0] or "Macro"
        self.name.setText(stem + (".py" if lang == "Python" else ".ijm"))

    def clear(self) -> None:
        self.recorder.clear()
        self.text.clear()

    def create_macro(self):
        from rsstatchem.ui.imagej import windows as W
        from rsstatchem.ui.imagej.frames.text_window import TextWindow
        win = TextWindow(self.name.text(), self.text.toPlainText(), self.bar, macro=True,
                         language=self.recorder.language)
        W.register(win)
        win.show()
        return win

    def closeEvent(self, event) -> None:  # noqa: N802
        from rsstatchem.ui.imagej import windows as W
        if self.bar is not None and self.bar.session.recorder is self.recorder:
            self.bar.session.recorder = None
        event.accept()
        W.unregister(self)


def show_recorder(bar) -> RecorderWindow:
    from rsstatchem.ui.imagej import windows as W
    for w in W.WINDOWS:
        if isinstance(w, RecorderWindow):
            w.show()
            w.raise_()
            return w
    win = RecorderWindow(bar)
    W.register(win)
    win.show()
    return win

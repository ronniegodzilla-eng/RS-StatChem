"""Text windows of the Image Analysis module (``ij/text/TextWindow`` and the macro
``Editor``). **Owned by WP-A.**

:class:`TextWindow` is a plain or editable text window with File (Save, Save As), Edit and
Font menus; with ``macro=True`` it is the macro editor of Plugins > New > Macro / Macros >
Edit... / the Recorder's "Create" button: Macros > Run Macro (Ctrl+R), Evaluate Line
(Ctrl+Y), Install Macros (Ctrl+I) and the language (ImageJ macro or Python).
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QAction, QFont, QKeySequence, QTextCursor
from PySide6.QtWidgets import QFileDialog, QMainWindow, QPlainTextEdit


class TextWindow(QMainWindow):
    def __init__(self, title: str, text: str = "", bar=None, editable: bool = False,
                 macro: bool = False, path: str | None = None, language: str | None = None):
        super().__init__()
        self.bar = bar
        self.path = path
        self.macro = macro
        self.language = language or ("Python" if path and path.endswith(".py") else "Macro")
        self.setWindowTitle(title)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(not (editable or macro))
        self.text.setFont(QFont("Menlo", 12))
        self.text.setPlainText(text)
        self.setCentralWidget(self.text)
        self.table = None
        self._build_menus()
        self.resize(560, 380)

    def _act(self, menu, text, slot, shortcut: str | None = None):
        act = QAction(text, self)
        if shortcut:
            act.setShortcut(QKeySequence(shortcut))
        act.triggered.connect(lambda _=False: slot())
        menu.addAction(act)
        return act

    def _build_menus(self) -> None:
        mb = self.menuBar()
        f = mb.addMenu("File")
        self._act(f, "Save", self.save, "Ctrl+S")
        self._act(f, "Save As...", self.save_as)
        f.addSeparator()
        self._act(f, "Close", self.close, "Ctrl+W")
        e = mb.addMenu("Edit")
        self._act(e, "Copy", self.text.copy, "Ctrl+C")
        if not self.text.isReadOnly():
            self._act(e, "Cut", self.text.cut, "Ctrl+X")
            self._act(e, "Paste", self.text.paste, "Ctrl+V")
        self._act(e, "Select All", self.text.selectAll, "Ctrl+A")
        self._act(e, "Clear", self.text.clear)
        fo = mb.addMenu("Font")
        self._act(fo, "Make Text Larger", lambda: self._font(+1))
        self._act(fo, "Make Text Smaller", lambda: self._font(-1))
        if self.macro:
            m = mb.addMenu("Macros")
            self._act(m, "Run Macro", self.run_macro, "Ctrl+R")
            self._act(m, "Evaluate Line", self.evaluate_line, "Ctrl+Y")
            self._act(m, "Install Macros", self.install, "Ctrl+I")
            lang = mb.addMenu("Language")
            for name in ("Macro", "Python"):
                a = self._act(lang, "IJ1 Macro" if name == "Macro" else "Python",
                              lambda n=name: setattr(self, "language", n))
                a.setCheckable(True)
                a.setChecked(self.language == name)

    def _font(self, delta: int) -> None:
        f = self.text.font()
        f.setPointSize(max(6, f.pointSize() + delta))
        self.text.setFont(f)

    # ------------------------------------------------------------------ contract helpers
    def set_table(self, table) -> None:
        self.table = table
        self.text.setPlainText(table.to_text("\t"))

    def append(self, line: str) -> None:
        if line == "\\Clear":
            self.text.clear()
            return
        self.text.appendPlainText(line)
        self.text.moveCursor(QTextCursor.MoveOperation.End)

    def contents(self) -> str:
        return self.text.toPlainText()

    # ------------------------------------------------------------------ actions
    def save(self) -> None:
        if not self.path:
            self.save_as()
            return
        Path(self.path).write_text(self.contents(), encoding="utf-8")

    def save_as(self, path: str | None = None) -> None:
        if path is None:
            ext = ".py" if self.language == "Python" else (".ijm" if self.macro else ".txt")
            path, _ = QFileDialog.getSaveFileName(self, "Save As", self.windowTitle()
                                                  if "." in self.windowTitle()
                                                  else self.windowTitle() + ext)
            if not path:
                return
        self.path = path
        self.setWindowTitle(Path(path).name)
        self.save()

    def run_macro(self) -> None:
        if self.bar is not None:
            cur = self.text.textCursor()
            code = cur.selectedText().replace(" ", "\n") if cur.hasSelection() else \
                self.contents()
            self.bar.run_macro_text(code, self.language, self.path)

    def evaluate_line(self) -> None:
        if self.bar is not None:
            line = self.text.textCursor().block().text()
            self.bar.run_macro_text(line, self.language, self.path)

    def install(self) -> None:
        if self.bar is not None:
            self.bar.install_macros(self.contents(), self.windowTitle())

    def closeEvent(self, event) -> None:  # noqa: N802
        from rsstatchem.ui.imagej import windows as W
        event.accept()
        W.unregister(self)

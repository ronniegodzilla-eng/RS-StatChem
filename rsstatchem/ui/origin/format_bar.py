"""Origin's Format toolbar for graph text: font, size, bold, italic, underline, superscript,
subscript and Greek.

It acts on the text selected in the active graph window — a text annotation, an axis (its
title), the legend or the layer title — and, while a text is being edited in place
(double-click a text or an axis title on the graph), on the selected characters: bold,
italic, underline, super/subscript and Greek wrap them in Origin's escape codes ``\\b( )``,
``\\i( )``, ``\\u( )``, ``\\+( )``, ``\\-( )`` and ``\\g( )``, which the graph renders.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QComboBox, QFontComboBox, QLineEdit, QToolBar

from rsstatchem.ui.origin.icons import icon

SIZES = ("6", "7", "8", "9", "10", "11", "12", "14", "16", "18", "20", "22", "24", "28", "32",
         "36", "48")
ESCAPES = {"bold": "b", "italic": "i", "underline": "u", "sup": "+", "sub": "-", "greek": "g"}


def text_target(graph, ref: tuple):
    """``(owner, font path, text path)`` of the graph object ``ref`` that carries text, or
    None. The paths are dotted attribute paths on the owner (e.g. ``"title_font"``)."""
    kind = ref[0]
    if kind == "graph" or len(ref) < 2 or ref[1] >= len(graph.layers):
        return None
    layer = graph.layers[ref[1]]
    if kind == "annotation":
        ann = next((a for a in layer.annotations if a.id == ref[2]), None)
        if ann is not None and ann.kind == "text":
            return ann, "font", "text"
        return None
    if kind == "axis":
        return layer.axis(ref[2]), "title_font", "title"
    if kind == "legend":
        return layer.legend, "font", None
    if kind == "layer":
        return layer, "title_font", "title"
    return None


def wrap_escape(text: str, start: int, end: int, code: str) -> tuple[str, int, int]:
    """Wrap ``text[start:end]`` in Origin's escape ``\\code(...)``; returns the new text and
    the new selection (inside the parentheses)."""
    start, end = sorted((max(0, start), max(0, end)))
    inner = text[start:end]
    head = f"\\{code}("
    new = text[:start] + head + inner + ")" + text[end:]
    return new, start + len(head), start + len(head) + len(inner)


class FormatToolbar(QToolBar):
    def __init__(self, main):
        super().__init__("Format", main)
        self.main = main
        self.setIconSize(main.iconSize() if hasattr(main, "iconSize") else self.iconSize())
        self.font_box = QFontComboBox()
        self.font_box.setMaximumWidth(170)
        self.font_box.setToolTip("Font")
        self.font_box.currentFontChanged.connect(lambda f: self.set_font("family", f.family()))
        self.size_box = QComboBox()
        self.size_box.setEditable(True)
        self.size_box.addItems(SIZES)
        self.size_box.setMaximumWidth(58)
        self.size_box.setToolTip("Font Size")
        self.size_box.lineEdit().editingFinished.connect(self._size_entered)
        self.size_box.activated.connect(lambda _i: self._size_entered())
        self.addWidget(self.font_box)
        self.addWidget(self.size_box)
        self.addSeparator()
        self.acts = {}
        for key, text, ic in (("bold", "Bold", "f_bold"), ("italic", "Italic", "f_italic"),
                              ("underline", "Underline", "f_underline")):
            act = self.addAction(icon(ic), text, lambda k=key: self.style_clicked(k))
            act.setCheckable(key != "underline")
            self.acts[key] = act
        self.addSeparator()
        for key, text, ic in (("sup", "Superscript", "f_sup"), ("sub", "Subscript", "f_sub"),
                              ("greek", "Greek", "f_greek")):
            self.acts[key] = self.addAction(icon(ic), text, lambda k=key: self.escape(k))
        self._syncing = True
        self.font_box.setCurrentFont(QFont("Arial"))
        self.size_box.setCurrentText("10")
        self._syncing = False
        self.sync()

    # ------------------------------------------------------------------ state
    def graph_window(self):
        return self.main.active_graph_window() if self.main is not None else None

    def editor(self) -> QLineEdit | None:
        gwin = self.graph_window()
        ed = getattr(gwin, "text_editor", None) if gwin is not None else None
        return ed if ed is not None and ed.isVisible() else None

    def target(self):
        gwin = self.graph_window()
        if gwin is None:
            return None, None
        return gwin, text_target(gwin.graph, gwin.selected)

    def sync(self) -> None:
        """Enable the toolbar for text-carrying selections and show their font."""
        gwin, tgt = self.target()
        editing = self.editor() is not None
        on = tgt is not None or editing
        for w in (self.font_box, self.size_box):
            w.setEnabled(on)
        for key, act in self.acts.items():
            act.setEnabled(on if key in ("bold", "italic") else
                           editing or (tgt is not None and tgt[2] is not None))
        if tgt is None:
            return
        owner, font_path, _ = tgt
        font = getattr(owner, font_path)
        self._syncing = True
        try:
            family = font.family or gwin.graph.font_family
            self.font_box.setCurrentFont(QFont(family))
            self.size_box.setCurrentText(f"{font.size:g}")
            self.acts["bold"].setChecked(font.bold)
            self.acts["italic"].setChecked(font.italic)
        finally:
            self._syncing = False

    # ------------------------------------------------------------------ commands
    def _size_entered(self) -> None:
        try:
            size = float(self.size_box.currentText().replace(",", "."))
        except ValueError:
            return
        if 1 <= size <= 400:
            self.set_font("size", size)

    def set_font(self, attr: str, value) -> None:
        """Font family / size / bold / italic of the selected text object."""
        if self._syncing:
            return
        gwin, tgt = self.target()
        if tgt is None:
            return
        owner, font_path, _ = tgt
        if getattr(getattr(owner, font_path), attr) == value:
            return
        gwin.apply_change(owner, f"{font_path}.{attr}", value)
        ed = self.editor()
        if ed is not None:
            gwin.style_editor()

    def style_clicked(self, key: str) -> None:
        """Bold / italic / underline: wrap the edited selection, else toggle the object's
        font (underline needs a text selection)."""
        ed = self.editor()
        if ed is not None and ed.hasSelectedText():
            self.escape(key)
            if key in self.acts and self.acts[key].isCheckable():
                self.sync()
            return
        if key in ("bold", "italic"):
            self.set_font(key, self.acts[key].isChecked())
        elif key == "underline":
            self.escape(key)

    def escape(self, key: str) -> None:
        """Superscript / subscript / Greek (and bold / italic / underline) for the selected
        characters of the text being edited, or for the whole text of the selected object."""
        code = ESCAPES[key]
        ed = self.editor()
        if ed is not None:
            text = ed.text()
            start = ed.selectionStart() if ed.hasSelectedText() else ed.cursorPosition()
            end = start + len(ed.selectedText()) if ed.hasSelectedText() else start
            new, s, e = wrap_escape(text, start, end, code)
            ed.setText(new)
            ed.setSelection(s, e - s) if e > s else ed.setCursorPosition(s)
            ed.setFocus(Qt.FocusReason.OtherFocusReason)
            return
        gwin, tgt = self.target()
        if tgt is None or tgt[2] is None:
            return
        owner, _, text_path = tgt
        text = getattr(owner, text_path) or ""
        if not text:
            return
        new, _, _ = wrap_escape(text, 0, len(text), code)
        gwin.apply_change(owner, text_path, new)

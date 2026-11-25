"""Operating the app's own windows for the connector: every menu command, dialog and table.

Two parts:

* :class:`GuiThread` runs functions on the Qt thread for a caller on another thread (the MCP
  server's). A command that opens a modal dialog does not return until the dialog is closed, so
  :meth:`GuiThread.interact` also reports "a dialog is waiting": jobs posted later run inside the
  dialog's own event loop, exactly where a user's clicks would be handled.
* plain functions that describe widgets as JSON-able data and change them the way a user does
  (signals included). They must be called on the Qt thread.
"""
from __future__ import annotations

import re
import threading
import time
import weakref
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout

import shiboken6
from PySide6.QtCore import QBuffer, QObject, QPoint, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QAbstractButton, QAbstractItemView, QAbstractSpinBox, QApplication,
                               QComboBox, QDateTimeEdit, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHeaderView,
                               QLabel, QLineEdit, QListView, QMainWindow, QMenu, QMessageBox,
                               QPlainTextEdit, QProgressDialog, QPushButton, QRadioButton,
                               QSlider, QSpinBox, QTabBar, QTabWidget, QTextEdit, QToolButton,
                               QTreeView, QWidget)

SKIP_COMMANDS = ("exit", "quit")      # would end the program, not a window
# the system's print and page panels are native windows nothing here can operate
NATIVE_COMMANDS = re.compile(r"^(quick )?print(?! preview)( .*)?$|^page setup( .*)?$", re.I)
MAX_CHOICES = 60
MAX_TEXT = 4000


# ---------------------------------------------------------------------------- Qt thread
class GuiThread(QObject):
    _post = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._post.connect(self._run, Qt.ConnectionType.QueuedConnection)
        self._pending: list[Future] = []      # commands still inside a dialog they opened
        self._lock = threading.Lock()

    def _run(self, job) -> None:
        future, fn = job
        if not future.set_running_or_notify_cancel():
            return
        try:
            result = fn()
        except BaseException as exc:  # noqa: BLE001 - handed to the caller
            future.set_exception(exc)
        else:
            future.set_result(result)

    def on_gui_thread(self) -> bool:
        from PySide6.QtCore import QThread

        return QThread.currentThread() is self.thread()

    def submit(self, fn) -> Future:
        future: Future = Future()
        self._post.emit((future, fn))
        return future

    def call(self, fn, timeout: float = 600):
        """Run ``fn`` on the Qt thread and return its result (no dialog expected)."""
        if self.on_gui_thread():
            return fn()
        return self.submit(fn).result(timeout)

    def interact(self, fn, timeout: float = 120) -> str:
        """Run ``fn``, which may open a dialog or end one that an earlier command is waiting
        in. Returns "done", "dialog" (one is waiting for input) or "busy" (still computing)."""
        with self._lock:
            end = time.monotonic() + timeout
            before = set(self.call(lambda: [key_of(d) for d in open_dialogs()], timeout))
            future = self.submit(fn)
            self._pending.append(future)
            while True:
                # a probe runs only when an event loop is turning: the computation is over or
                # a dialog is waiting
                try:
                    self.submit(_settled).result(max(0.05, end - time.monotonic()))
                    time.sleep(0.03)            # zero timers, queued signals, show events
                    dialogs = self.call(open_dialogs, max(0.05, end - time.monotonic()))
                except FutureTimeout:
                    return "busy"
                finished = [f for f in self._pending if f.done()]
                self._pending = [f for f in self._pending if not f.done()]
                for f in finished:
                    if f.exception() is not None:
                        raise f.exception()
                modal = any(d.isModal() for d in dialogs)
                fresh = any(key_of(d) not in before for d in dialogs)
                if future.done() and not self._pending:
                    return "dialog" if modal or fresh else "done"
                if modal or (fresh and not future.done()) or (future.done() and dialogs):
                    return "dialog"
                if time.monotonic() > end:
                    return "busy"
                time.sleep(0.02)

    def wait(self, timeout: float = 120) -> str:
        """Wait for a command reported as "busy"."""
        return self.interact(lambda: None, timeout)


def _settled() -> None:
    QApplication.sendPostedEvents()


def key_of(widget: QWidget) -> int:
    """Identity of a widget: Python wrappers of Qt-created widgets come and go."""
    return shiboken6.getCppPointer(widget)[0]


def release_static_dialogs() -> None:
    """``QFontDialog.getFont`` and ``QMessageBox.about`` hold Python's interpreter lock while
    their dialog is open, so no other thread could answer it: use equivalents that do not."""
    from PySide6.QtWidgets import QFontDialog

    if getattr(QFontDialog, "_rs_released", False):
        return

    def get_font(*args, **kwargs):
        font = args[0] if args and not isinstance(args[0], QWidget) else None
        parent = next((a for a in args if isinstance(a, QWidget)), None)
        title = next((a for a in args if isinstance(a, str)), kwargs.get("title", ""))
        dialog = QFontDialog(parent)
        if title:
            dialog.setWindowTitle(title)
        if font is not None:
            dialog.setCurrentFont(font)
        ok = bool(dialog.exec())
        return ok, dialog.selectedFont() if ok else (font or dialog.currentFont())

    def about(parent, title, text):
        box = QMessageBox(QMessageBox.Icon.NoIcon, title, text,
                          QMessageBox.StandardButton.Ok, parent)
        box.exec()

    QFontDialog.getFont = staticmethod(get_font)
    QMessageBox.about = staticmethod(about)
    QFontDialog._rs_released = True


# ---------------------------------------------------------------------------- windows
def open_dialogs() -> list[QDialog]:
    """Visible dialogs, the one that has the user's attention last."""
    app = QApplication.instance()
    out = [w for w in app.topLevelWidgets()
           if isinstance(w, QDialog) and w.isVisible() and not isinstance(w, QProgressDialog)]
    modal = app.activeModalWidget()
    out.sort(key=lambda w: (w is modal, w.isModal()))
    return out


def open_windows() -> list[QWidget]:
    """Visible top-level windows that are not dialogs, menus or tooltips."""
    out = []
    for w in QApplication.instance().topLevelWidgets():
        if not w.isVisible() or isinstance(w, (QDialog, QMenu, QProgressDialog)):
            continue
        if w.windowType() in (Qt.WindowType.ToolTip, Qt.WindowType.Popup,
                              Qt.WindowType.SplashScreen):
            continue
        if isinstance(w, QMainWindow) or w.windowTitle():
            out.append(w)
    return out


def title_of(w: QWidget) -> str:
    return (w.windowTitle() or type(w).__name__).replace("[*]", "").strip()


def window_kind(w: QWidget) -> str:
    """The module a window belongs to ("stats", "rsm", "graph", "chem", "image", …)."""
    name = type(w).__module__
    for part, key in (("output_viewer", "output"), ("data_editor", "stats"), (".dx.", "rsm"),
                      (".origin.", "graph"), (".chemdraw.", "chem"), (".imagej.", "image"),
                      ("launcher", "launcher")):
        if part in name:
            return key
    return "other"


# ---------------------------------------------------------------------------- element ids
class Elements:
    """Stable short ids ("e12") for widgets, so a client can refer to what it was shown."""

    def __init__(self) -> None:
        self._widgets: dict[str, weakref.ref] = {}
        self._n = 0

    def id_of(self, widget: QWidget) -> str:
        known = widget.property("_rs_element")
        if known and known in self._widgets:
            return known
        self._n += 1
        key = f"e{self._n}"
        widget.setProperty("_rs_element", key)
        self._widgets[key] = weakref.ref(widget)
        return key

    def get(self, key: str) -> QWidget | None:
        ref = self._widgets.get(str(key).strip())
        widget = ref() if ref is not None else None
        if widget is None and ref is not None:     # the wrapper is gone, maybe not the widget
            widget = next((w for w in QApplication.allWidgets()
                           if w.property("_rs_element") == str(key).strip()), None)
        return widget if widget is not None and shiboken6.isValid(widget) else None


# ---------------------------------------------------------------------------- menus
def clean(text: str) -> str:
    text = (text or "").split("\t")[0]
    return text.replace("&&", "\0").replace("&", "").replace("\0", "&").rstrip(".…").strip()


def commands(window: QWidget) -> list[tuple[str, QAction]]:
    """Every menu command of a window: ("Analyze > Compare Means > …", action)."""
    bar = window.menuBar() if isinstance(window, QMainWindow) else None
    if bar is None:
        return []
    out: list[tuple[str, QAction]] = []

    def walk(menu: QMenu, path: list[str]) -> None:
        menu.aboutToShow.emit()
        for act in menu.actions():
            if act.isSeparator() or not act.isVisible():
                continue
            text = clean(act.text())
            if not text:
                continue
            if act.menu() is not None:
                walk(act.menu(), path + [text])
            else:
                out.append((" > ".join(path + [text]), act))

    for top in bar.actions():
        if top.menu() is not None and top.isVisible():
            walk(top.menu(), [clean(top.text())])
        elif top.isVisible() and clean(top.text()):
            out.append((clean(top.text()), top))
    return out


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9>+/]+", " ", clean(text).lower().replace("▸", ">")).strip()


def find_command(window: QWidget, text: str) -> tuple[str, QAction]:
    """The command named by its full menu path, or by a unique ending of it."""
    parts = [_norm(p) for p in re.split(r"\s*(?:>|▸|→)\s*", text) if p.strip()]
    if not parts:
        raise ValueError("Give the menu command, for example 'Analyze > Correlate > Bivariate'.")
    entries = commands(window)
    split = [(path, act, [_norm(p) for p in path.split(" > ")]) for path, act in entries]
    exact = [(p, a) for p, a, words in split if words == parts]
    tail = [(p, a) for p, a, words in split if words[-len(parts):] == parts]
    for hits in (exact, tail):
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            if len({id(a) for _, a in hits}) == 1:
                return hits[0]
            raise ValueError(f"'{text}' matches several commands: "
                             + "; ".join(p for p, _ in hits[:8]) + ". Give the full path.")
    words = set(" ".join(parts).split())
    near = [p for p, _, w in split if words & set(" ".join(w).split())]
    near.sort(key=lambda p: -len(words & set(_norm(p).split())))
    hint = " Similar: " + "; ".join(near[:8]) + "." if near else ""
    raise ValueError(f"No menu command '{text}' in the window '{title_of(window)}'.{hint}")


# ---------------------------------------------------------------------------- labels
def _form_label(widget: QWidget) -> str:
    parent = widget.parentWidget()
    if parent is None or parent.layout() is None:
        return ""

    def search(layout) -> str:
        if isinstance(layout, QFormLayout):
            label = layout.labelForField(widget)
            if isinstance(label, QLabel):
                return label.text()
        for i in range(layout.count()):
            sub = layout.itemAt(i).layout()
            if sub is not None:
                found = search(sub)
                if found:
                    return found
        return ""

    return search(parent.layout())


def _near_label(widget: QWidget) -> str:
    """The label to the left of a widget, else directly above it."""
    parent = widget.parentWidget()
    if parent is None:
        return ""
    r = widget.geometry()
    best, score = "", 10 ** 9
    for lab in parent.findChildren(QLabel, options=Qt.FindChildOption.FindDirectChildrenOnly):
        if not lab.isVisibleTo(parent) or not lab.text().strip() or len(lab.text()) > 120:
            continue
        g = lab.geometry()
        if abs(g.center().y() - r.center().y()) < 14 and 0 <= r.left() - g.right() + 4 < 260:
            s = r.left() - g.right()
        elif -6 <= r.top() - g.bottom() < 30 and g.left() < r.right() and g.right() > r.left():
            s = 300 + r.top() - g.bottom()
        else:
            continue
        if s < score:
            best, score = lab.text(), s
    return best


def _plain(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", text.replace("&nbsp;", " ")).strip()


def label_of(widget: QWidget) -> str:
    if isinstance(widget, QAbstractButton):
        text = clean(widget.text())
        if text:
            return text
    if widget.accessibleName():
        return _plain(widget.accessibleName())
    top = widget.window()
    for lab in top.findChildren(QLabel):
        if lab.buddy() is widget:
            return clean(_plain(lab.text()))
    for text in (_form_label(widget), _near_label(widget)):
        if text:
            return clean(_plain(text)).rstrip(":")
    if isinstance(widget, (QLineEdit, QPlainTextEdit, QTextEdit)) and widget.placeholderText():
        return widget.placeholderText()
    return _plain(widget.toolTip())[:80]


def _group_of(widget: QWidget, root: QWidget) -> str:
    p = widget.parentWidget()
    while p is not None and p is not root:
        if isinstance(p, QGroupBox) and p.title():
            return clean(p.title())
        p = p.parentWidget()
    return ""


def _tab_of(widget: QWidget, root: QWidget) -> str:
    p, child = widget.parentWidget(), widget
    while p is not None:
        if isinstance(p, QTabWidget):
            for i in range(p.count()):
                page = p.widget(i)
                if page is child or page.isAncestorOf(widget):
                    return clean(p.tabText(i))
        if p is root:
            break
        child, p = p, p.parentWidget()
    return ""


# ---------------------------------------------------------------------------- describing
def _index_text(model, index) -> str:
    value = model.data(index, Qt.ItemDataRole.DisplayRole)
    return "" if value is None else str(value)


def table_size(view: QAbstractItemView) -> tuple[int, int]:
    model = view.model()
    return (0, 0) if model is None else (model.rowCount(), model.columnCount())


def read_cells(view: QAbstractItemView, row: int = 0, rows: int = 20, col: int = 0,
               cols: int = 20) -> dict:
    """Part of a table, list or tree as texts (trees are flattened with indentation)."""
    model = view.model()
    if model is None:
        return {"rows": 0, "columns": 0, "header": [], "cells": []}
    if isinstance(view, QTreeView):
        out: list[list[str]] = []

        def walk(parent, depth: int) -> None:
            for r in range(model.rowCount(parent)):
                idx = model.index(r, 0, parent)
                cells = [_index_text(model, model.index(r, c, parent))
                         for c in range(model.columnCount(parent))]
                cells[0] = "  " * depth + cells[0]
                out.append(cells)
                walk(idx, depth + 1)

        walk(view.rootIndex(), 0)
        header = [str(model.headerData(c, Qt.Orientation.Horizontal) or "")
                  for c in range(model.columnCount())]
        return {"rows": len(out), "columns": model.columnCount(), "header": header,
                "cells": out[row:row + rows]}
    n, k = model.rowCount(), model.columnCount()
    r_range, c_range = range(row, min(n, row + rows)), range(col, min(k, col + cols))

    def head(i, orientation):
        v = model.headerData(i, orientation)
        return "" if v is None else str(v)

    return {"rows": n, "columns": k,
            "header": [head(c, Qt.Orientation.Horizontal) for c in c_range],
            "row_header": [head(r, Qt.Orientation.Vertical) for r in r_range],
            "cells": [[_index_text(model, model.index(r, c)) for c in c_range]
                      for r in r_range]}


def write_cells(view: QAbstractItemView, row: int, col: int, values: list[list]) -> int:
    """Type a block of values into an editable table; returns the number of cells set."""
    model = view.model()
    if model is None:
        raise ValueError("This element is not a table.")
    done = 0
    refused = []
    for i, line in enumerate(values):
        for j, value in enumerate(line if isinstance(line, (list, tuple)) else [line]):
            idx = model.index(row + i, col + j)
            if not idx.isValid():
                raise ValueError(f"The table has {model.rowCount()} rows and "
                                 f"{model.columnCount()} columns; row {row + i + 1}, column "
                                 f"{col + j + 1} does not exist ({done} cells were set).")
            if not (model.flags(idx) & Qt.ItemFlag.ItemIsEditable):
                combo = view.indexWidget(idx)
                if combo is not None:
                    set_value(combo, value)
                    done += 1
                    continue
                refused.append((row + i + 1, col + j + 1))
                continue
            text = "" if value is None else value if isinstance(value, bool) else str(value)
            if not model.setData(idx, text, Qt.ItemDataRole.EditRole):
                refused.append((row + i + 1, col + j + 1))
            else:
                done += 1
    if refused:
        raise ValueError(f"{done} cells were set; not accepted (row, column): "
                         + ", ".join(map(str, refused[:10])))
    return done


def _list_state(view: QListView) -> dict:
    model = view.model()
    out = {"items": [], "selected": []}
    if model is None:
        return out
    chosen = {i.row() for i in view.selectionModel().selectedIndexes()} \
        if view.selectionModel() else set()
    checked = []
    for r in range(min(model.rowCount(), 300)):
        idx = model.index(r, 0)
        text = _index_text(model, idx)
        out["items"].append(text)
        if r in chosen:
            out["selected"].append(text)
        if model.flags(idx) & Qt.ItemFlag.ItemIsUserCheckable:
            state = model.data(idx, Qt.ItemDataRole.CheckStateRole)
            if state in (Qt.CheckState.Checked, Qt.CheckState.Checked.value):
                checked.append(text)
            out.setdefault("checked", checked)
    return out


def describe(widget: QWidget, elements: Elements, root: QWidget) -> dict | None:
    """One interactive widget as data; None for decoration."""
    d: dict = {}
    if isinstance(widget, QComboBox):
        d = {"kind": "choice", "value": widget.currentText(),
             "choices": [widget.itemText(i) for i in range(min(widget.count(), MAX_CHOICES))]}
        if widget.isEditable():
            d["kind"] = "choice or text"
    elif isinstance(widget, QDateTimeEdit):
        d = {"kind": "date", "value": widget.text()}
    elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        d = {"kind": "number", "value": widget.value(), "min": widget.minimum(),
             "max": widget.maximum()}
    elif isinstance(widget, QAbstractSpinBox):
        d = {"kind": "text", "value": widget.text()}
    elif isinstance(widget, QLineEdit):
        d = {"kind": "text", "value": widget.text()}
        if widget.isReadOnly():
            d["read_only"] = True
    elif isinstance(widget, (QPlainTextEdit, QTextEdit)):
        d = {"kind": "long text", "value": widget.toPlainText()[:MAX_TEXT]}
        if widget.isReadOnly():
            d["read_only"] = True
    elif isinstance(widget, QSlider):
        d = {"kind": "slider", "value": widget.value(), "min": widget.minimum(),
             "max": widget.maximum()}
    elif isinstance(widget, QTabBar):
        if widget.count() < 2:
            return None
        d = {"kind": "tabs", "value": clean(widget.tabText(widget.currentIndex())),
             "choices": [clean(widget.tabText(i)) for i in range(widget.count())
                         if widget.isTabVisible(i)]}
    elif isinstance(widget, QRadioButton):
        d = {"kind": "radio", "value": widget.isChecked()}
    elif isinstance(widget, QAbstractButton):
        if isinstance(widget, QToolButton) and not label_of(widget):
            return None
        d = {"kind": "check", "value": widget.isChecked()} if widget.isCheckable() \
            else {"kind": "button"}
    elif isinstance(widget, QGroupBox) and widget.isCheckable():
        d = {"kind": "check", "value": widget.isChecked(), "label": clean(widget.title())}
    elif isinstance(widget, QListView):
        d = {"kind": "list", **_list_state(widget)}
    elif isinstance(widget, QHeaderView):
        return None
    elif isinstance(widget, QAbstractItemView):
        rows, cols = table_size(widget)
        d = {"kind": "tree" if isinstance(widget, QTreeView) else "table", "rows": rows,
             "columns": cols}
    else:
        return None
    out = {"id": elements.id_of(widget), "label": d.pop("label", None) or label_of(widget)}
    out.update(d)
    if not widget.isEnabled():
        out["enabled"] = False
    group = _group_of(widget, root)
    if group:
        out["group"] = group
    tab = _tab_of(widget, root)
    if tab:
        out["tab"] = tab
    return out


def _inside_composite(widget: QWidget, root: QWidget) -> bool:
    """Parts of a combo box, spin box or table (its line edit, its headers' buttons)."""
    p = widget.parentWidget()
    while p is not None and p is not root:
        if isinstance(p, (QComboBox, QAbstractSpinBox, QHeaderView)) or \
                (isinstance(p, QAbstractItemView) and not isinstance(widget, QComboBox)):
            return True
        p = p.parentWidget()
    return False


def _shown(widget: QWidget, root: QWidget) -> bool:
    """Visible, or on another page of a tab widget (those fields can be set too)."""
    p = widget
    while p is not None and p is not root:
        parent = p.parentWidget()
        if p.isHidden():
            holder = parent.parentWidget() if parent is not None else None
            if not (isinstance(holder, QTabWidget) and holder.indexOf(p) >= 0):
                return False
        p = parent
    return True


def picker_state(dialog: QWidget) -> dict | None:
    """The variable boxes of an analysis dialog (VariablePickerDialog)."""
    targets = getattr(dialog, "_targets", None)
    if targets is None or not hasattr(dialog, "_take"):
        return None
    boxes = []
    for t in targets:
        # (the "isHidden" test leaves out boxes of a mode that is not selected)
        boxes.append({"box": t.key, "label": label_of(t.widget),
                      "holds": "one variable" if t.single else "several variables",
                      "required": bool(t.min_items), "variables": dialog.selected(t.key)})
    for t in getattr(dialog, "_pairs", []):
        boxes.append({"box": t.key, "label": label_of(t.widget), "holds": "pairs of variables",
                      "required": True, "variables": [list(p) for p in dialog.pairs(t.key)]})
    return {"available": dialog.ds.names(), "boxes": boxes}


def describe_window(window: QWidget, elements: Elements) -> dict:
    """What a window or dialog shows and offers."""
    out: dict = {"window": elements.id_of(window), "title": title_of(window),
                 "type": "dialog" if isinstance(window, QDialog) else window_kind(window)}
    if isinstance(window, QMessageBox):
        out["message"] = _plain(window.text())
        if window.informativeText():
            out["message"] += " " + _plain(window.informativeText())
        out["buttons"] = [clean(b.text()) for b in window.buttons()]
        return out
    if isinstance(window, QFileDialog):
        out["message"] = ("File dialog (" + ("save" if window.acceptMode()
                          == QFileDialog.AcceptMode.AcceptSave else "open")
                          + "): give the full path with file_path.")
        out["file_types"] = window.nameFilters()
        out["buttons"] = ["Cancel"]
        return out
    picker = picker_state(window)
    skip: set = set()
    if picker is not None:
        out["variables"] = picker
        skip = {window.source, *(t.widget for t in window._targets),
                *(t.widget for t in getattr(window, "_pairs", []))}
    fields, buttons, tables = [], [], []
    for w in window.findChildren(QWidget):
        if w in skip or not shiboken6.isValid(w) or w.window() is not window:
            continue
        if not _shown(w, window) or _inside_composite(w, window):
            continue
        d = describe(w, elements, window)
        if d is None:
            continue
        if d["kind"] == "button":
            if picker is not None and not d["label"].strip("◀▶<>►◄ "):
                continue            # the arrow buttons between the variable boxes
            entry = {k: d[k] for k in ("id", "label", "enabled") if k in d}
            if entry["label"]:
                buttons.append(entry)
        elif d["kind"] in ("table", "tree"):
            tables.append(d)
        else:
            fields.append(d)
    texts = []
    if isinstance(window, QDialog):
        used = {f["label"] for f in fields}
        for lab in window.findChildren(QLabel):
            text = _plain(lab.text())
            if text and lab.isVisible() and clean(text).rstrip(":") not in used \
                    and text not in texts:
                texts.append(text[:400])
    if texts:
        out["texts"] = texts[:40]
    if fields:
        out["fields"] = fields
    if tables:
        out["tables"] = tables
    if buttons:
        out["buttons"] = buttons
    return out


# ---------------------------------------------------------------------------- changing
def _truth(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on", "ya", "checked")
    return bool(value)


def _pick(texts: list[str], value, what: str) -> int:
    want = str(value).strip().lower()
    for test in (lambda t: t.strip().lower() == want, lambda t: clean(t).lower() == want,
                 lambda t: want and t.strip().lower().startswith(want)):
        hits = [i for i, t in enumerate(texts) if test(t)]
        if len(hits) >= 1:
            return hits[0]
    raise ValueError(f"'{value}' is not a choice of {what or 'this field'}. Choices: "
                     + ", ".join(texts[:MAX_CHOICES]) + ".")


def set_value(widget: QWidget, value) -> None:
    """Set a field the way a user would (change signals included)."""
    name = label_of(widget)
    if not widget.isEnabled():
        raise ValueError(f"The field '{name}' is disabled at the moment.")
    if isinstance(widget, QComboBox):
        texts = [widget.itemText(i) for i in range(widget.count())]
        try:
            i = _pick(texts, value, name)
        except ValueError:
            if not widget.isEditable():
                raise
            widget.setEditText(str(value))
            if widget.lineEdit() is not None:
                widget.lineEdit().editingFinished.emit()
            return
        widget.setCurrentIndex(i)
        widget.activated.emit(i)
        widget.textActivated.emit(widget.itemText(i))
    elif isinstance(widget, QSpinBox):
        number = int(round(float(value)))
        _check_range(widget, number, name)
        widget.setValue(number)
        widget.editingFinished.emit()
    elif isinstance(widget, QDoubleSpinBox):
        number = float(str(value).replace(",", ".")) if isinstance(value, str) else float(value)
        _check_range(widget, number, name)
        widget.setValue(number)
        widget.editingFinished.emit()
    elif isinstance(widget, QDateTimeEdit):
        from PySide6.QtCore import QDateTime

        stamp = QDateTime.fromString(str(value), Qt.DateFormat.ISODate)
        if not stamp.isValid():
            raise ValueError(f"'{value}' is not a date (YYYY-MM-DD).")
        widget.setDateTime(stamp)
    elif isinstance(widget, QLineEdit):
        widget.setText("" if value is None else str(value))
        widget.textEdited.emit(widget.text())
        widget.editingFinished.emit()
    elif isinstance(widget, (QPlainTextEdit, QTextEdit)):
        widget.setPlainText(str(value))
    elif isinstance(widget, QSlider):
        widget.setValue(int(value))
    elif isinstance(widget, QTabBar):
        widget.setCurrentIndex(_pick([widget.tabText(i) for i in range(widget.count())], value,
                                     "the tabs"))
    elif isinstance(widget, QGroupBox):
        if widget.isChecked() != _truth(value):
            widget.setChecked(_truth(value))
            widget.clicked.emit(widget.isChecked())
    elif isinstance(widget, QAbstractButton) and widget.isCheckable():
        if isinstance(widget, QRadioButton) and not _truth(value):
            return
        if widget.isChecked() != _truth(value):
            widget.click()
    elif isinstance(widget, QListView):
        _set_list(widget, value)
    else:
        raise ValueError(f"The element '{name}' cannot be set; press it or write its cells.")


def _check_range(widget, number, name: str) -> None:
    if not widget.minimum() <= number <= widget.maximum():
        raise ValueError(f"'{name}' must be between {widget.minimum():g} and "
                         f"{widget.maximum():g}.")


def _set_list(view: QListView, value) -> None:
    from PySide6.QtCore import QItemSelectionModel

    model = view.model()
    wanted = value if isinstance(value, (list, tuple)) else [value]
    texts = [_index_text(model, model.index(r, 0)) for r in range(model.rowCount())]
    rows = [_pick(texts, v, label_of(view)) for v in wanted if v not in (None, "")]
    flags = model.flags(model.index(rows[0] if rows else 0, 0)) if texts else 0
    if texts and flags & Qt.ItemFlag.ItemIsUserCheckable:
        for r in range(len(texts)):
            state = Qt.CheckState.Checked if r in rows else Qt.CheckState.Unchecked
            model.setData(model.index(r, 0), state, Qt.ItemDataRole.CheckStateRole)
        return
    view.clearSelection()
    for r in rows:
        idx = model.index(r, 0)
        view.selectionModel().setCurrentIndex(idx, QItemSelectionModel.SelectionFlag.Select)


def select_cells(view: QAbstractItemView, rows: list[int] | None, cols: list[int] | None) -> None:
    """Select whole rows, whole columns, or the cells in both (0-based)."""
    from PySide6.QtCore import QItemSelection, QItemSelectionModel

    model = view.model()
    if model is None:
        raise ValueError("This element is not a table.")
    n, k = model.rowCount(), model.columnCount()
    for r in rows or []:
        if not 0 <= r < n:
            raise ValueError(f"The table has {n} rows.")
    for c in cols or []:
        if not 0 <= c < k:
            raise ValueError(f"The table has {k} columns.")
    sel = QItemSelection()
    if rows is None and cols is not None:
        for c in cols:
            sel.select(model.index(0, c), model.index(n - 1, c))
    elif cols is None and rows is not None:
        for r in rows:
            sel.select(model.index(r, 0), model.index(r, k - 1))
    else:
        for r in rows or []:
            for c in cols or []:
                sel.select(model.index(r, c), model.index(r, c))
    first = model.index((rows or [0])[0], (cols or [0])[0])
    view.setCurrentIndex(first)
    view.selectionModel().select(sel, QItemSelectionModel.SelectionFlag.ClearAndSelect)


def fill_variables(dialog: QWidget, variables: dict) -> None:
    """Move variables into the boxes of an analysis dialog: ``{box or its label: [names]}``.
    Each box named is emptied first."""
    if picker_state(dialog) is None:
        raise ValueError(f"The dialog '{title_of(dialog)}' has no variable boxes; use values.")
    targets = {t.key: t for t in dialog._targets}
    pairs = {t.key: t for t in getattr(dialog, "_pairs", [])}

    def box(name: str):
        for group in (targets, pairs):
            for key, t in group.items():
                if name.strip().lower() in (key.lower(), clean(label_of(t.widget)).lower(),
                                            clean(label_of(t.widget)).lower().rstrip(":")):
                    return key, t
        known = [f"{k} ({label_of(t.widget)})" for k, t in {**targets, **pairs}.items()]
        raise ValueError(f"The dialog has no variable box '{name}'. Boxes: "
                         + ", ".join(known) + ".")

    for name, wanted in variables.items():
        key, target = box(name)
        if key in targets:
            names = [wanted] if isinstance(wanted, str) else list(wanted or [])
            if target.single and len(names) > 1:
                raise ValueError(f"The box '{key}' holds one variable.")
            dialog._move_back(target, all_items=True)
            for raw in names:
                if not dialog.ds.has_name(str(raw)):
                    raise ValueError(f"There is no variable '{raw}'. Variables: "
                                     + ", ".join(dialog.ds.names()[:60]) + ".")
                var = dialog.ds.var(str(raw))
                for other in targets.values():      # already used in another box
                    if other is not target and var.name in dialog.selected(other.key):
                        other.widget.clearSelection()
                        row = dialog.selected(other.key).index(var.name)
                        other.widget.item(row).setSelected(True)
                        dialog._move_back(other)
                if not dialog._take(var.name, target):
                    raise ValueError(f"The box '{key}' does not accept '{var.name}' "
                                     f"({var.type}, {var.measure}).")
        else:
            target.widget.clear()
            for pair in wanted or []:
                a, b = pair
                dialog._add_pair(target, dialog.ds.var(str(a)).name, dialog.ds.var(str(b)).name)
    dialog._update_arrows()
    dialog.targets_changed()


def find_field(window: QWidget, name: str, elements: Elements) -> QWidget:
    """A field by its id or label ("Group: Label" narrows it down)."""
    direct = elements.get(name)
    if direct is not None:
        return direct
    want = clean(str(name)).lower().rstrip(":")
    hits = []
    for w in window.findChildren(QWidget):
        if w.window() is not window or not _shown(w, window) or _inside_composite(w, window):
            continue
        d = describe(w, elements, window)
        if d is None:
            continue
        label = clean(d["label"]).lower().rstrip(":")
        full = f"{d.get('group', '')}: {label}".lower()
        if want in (label, full) or want == f"{d.get('tab', '')}: {label}".lower():
            hits.append((w, d))
    if len(hits) > 1:        # a button and a field of the same name: prefer the field
        fields = [h for h in hits if h[1]["kind"] != "button"]
        if len(fields) == 1:
            hits = fields
    if len(hits) == 1:
        return hits[0][0]
    if hits:
        raise ValueError(f"Several elements are called '{name}': " + ", ".join(
            f"{d['id']} ({d.get('tab', '')} {d.get('group', '')} {d['kind']})".replace("  ", " ")
            for _, d in hits) + ". Use an id.")
    raise ValueError(f"There is no element '{name}' in '{title_of(window)}'.")


def find_button(window: QWidget, name: str, elements: Elements) -> QAbstractButton:
    direct = elements.get(name)
    if isinstance(direct, QAbstractButton):
        return direct
    want = clean(str(name)).lower()
    box_names = {"ok": ("ok", "continue", "finish"), "continue": ("continue", "ok")}
    buttons = [b for b in window.findChildren(QAbstractButton)
               if b.window() is window and b.isVisible()]
    for wanted in dict.fromkeys((want, *box_names.get(want, ()))):
        hits = [b for b in buttons if clean(b.text()).lower() == wanted]
        boxed = [b for b in hits if isinstance(b.parentWidget(), QDialogButtonBox)]
        push = [b for b in hits if not b.isCheckable()]   # "Plots..." next to a radio "Plots"
        for group in (boxed, push, hits):
            if len(group) == 1:
                return group[0]
        if len(hits) > 1:
            raise ValueError(f"Several buttons are called '{name}': "
                             + ", ".join(elements.id_of(b) for b in hits) + ". Use an id.")
    tips = [b for b in buttons if clean(b.toolTip()).lower().startswith(want) and want]
    if len(tips) == 1:
        return tips[0]
    raise ValueError(f"There is no button '{name}' in '{title_of(window)}'. Buttons: "
                     + ", ".join(sorted({clean(b.text()) for b in buttons if clean(b.text())}))
                     + ".")


def choose_file(dialog: QFileDialog, path: str) -> None:
    edit = dialog.findChild(QLineEdit, "fileNameEdit") or dialog.findChild(QLineEdit)
    if edit is not None:
        edit.setText(str(path))
    else:
        dialog.selectFile(str(path))
    dialog.accept()


# ---------------------------------------------------------------------------- pixels and input
def screenshot(widget: QWidget, max_width: int = 1400) -> bytes:
    pixmap = widget.grab()
    if pixmap.width() > max_width:
        pixmap = pixmap.scaledToWidth(max_width, Qt.TransformationMode.SmoothTransformation)
    buffer = QBuffer()
    buffer.open(QBuffer.OpenModeFlag.WriteOnly)
    pixmap.save(buffer, "PNG")
    return bytes(buffer.data())


def mouse(window: QWidget, points: list[tuple[float, float]], action: str = "click",
          button: str = "left", modifiers: list[str] | None = None) -> None:
    """Mouse input at window coordinates (logical pixels from the top left corner)."""
    from PySide6.QtTest import QTest

    buttons = {"left": Qt.MouseButton.LeftButton, "right": Qt.MouseButton.RightButton,
               "middle": Qt.MouseButton.MiddleButton}
    keys = Qt.KeyboardModifier.NoModifier
    for m in modifiers or []:
        keys |= {"shift": Qt.KeyboardModifier.ShiftModifier,
                 "ctrl": Qt.KeyboardModifier.ControlModifier,
                 "cmd": Qt.KeyboardModifier.ControlModifier,
                 "alt": Qt.KeyboardModifier.AltModifier}[m.lower()]
    b = buttons[button]
    spots = [QPoint(round(x), round(y)) for x, y in points]
    if not spots:
        raise ValueError("Give at least one point.")
    for p in spots:
        if not window.rect().contains(p):
            raise ValueError(f"({p.x()}, {p.y()}) is outside the window "
                             f"({window.width()} x {window.height()}).")
    target = window.childAt(spots[0]) or window

    def at(p: QPoint) -> QPoint:
        return target.mapFrom(window, p)

    if action == "click":
        QTest.mouseMove(target, at(spots[0]))
        QTest.mouseClick(target, b, keys, at(spots[0]))
    elif action == "double_click":
        QTest.mouseDClick(target, b, keys, at(spots[0]))
    elif action == "move":
        QTest.mouseMove(target, at(spots[0]))
    elif action == "drag":
        if len(spots) < 2:
            raise ValueError("A drag needs at least two points.")
        _drag(target, [at(p) for p in spots], b, keys)
    else:
        raise ValueError("action must be click, double_click, move or drag.")


def _drag(target: QWidget, path: list[QPoint], button, keys) -> None:
    from PySide6.QtCore import QEvent, QPointF
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtTest import QTest

    QTest.mousePress(target, button, keys, path[0])
    last = path[0]
    for p in path[1:]:
        for step in range(1, 9):     # moves with the button held (QTest.mouseMove drops it)
            f = step / 8
            pos = QPointF(last.x() + (p.x() - last.x()) * f, last.y() + (p.y() - last.y()) * f)
            QApplication.sendEvent(target, QMouseEvent(
                QEvent.Type.MouseMove, pos, QPointF(target.mapToGlobal(pos.toPoint())),
                Qt.MouseButton.NoButton, button, keys))
        last = p
    QTest.mouseRelease(target, button, keys, last)


def type_keys(window: QWidget, text: str = "", keys: str = "") -> None:
    """Type text, then press key combinations such as "Ctrl+A Delete"."""
    from PySide6.QtGui import QKeySequence
    from PySide6.QtTest import QTest

    target = window.focusWidget() or window
    if text:
        QTest.keyClicks(target, text)
    for combo in keys.split():
        seq = QKeySequence.fromString(combo)
        if seq.count() == 0:
            raise ValueError(f"'{combo}' is not a key combination.")
        part = seq[0]
        QTest.keyClick(target, part.key(), part.keyboardModifiers())

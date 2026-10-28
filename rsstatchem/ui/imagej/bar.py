"""The ImageJ bar: the small main window with the menus, the tool icons and the status line.
It is the module window registered with the app context (``self.ctx``; ``ctx.window_closed``
when it closes; Window > Modules). **Owned by WP-A.**

Command plumbing (contract): every menu action calls :meth:`ImageJBar.run_command` with the
command label. A command with ``ui_action`` gets ``ui_action(bar, command)``; otherwise the
bar shows the :class:`~rsstatchem.ui.imagej.dialogs.generic.GenericDialog` of its fields (if
any) and runs it through the registry (recorded, undoable, errors shown as ImageJ messages).

WP-A additions: File > Open Recent, ImageJ's single-key shortcuts (letters without ctrl,
``+ - < > \\ / ]``; "Require control key" in Edit > Options > Misc), tool options on
double-click, the image popup menu, macOS menu bars for image windows that share these
actions, "Save changes to ...?" on closing, installed macros under Plugins > Macros, the
startup macro, macro/Python running with dialogs, and the recorder hooks for ``open()``,
``saveAs()`` and tool selections. Tests can set ``message_hook(title, text)``,
``question_hook(title, text) -> "yes"|"no"|"cancel"`` and ``file_hook(kind, title, default)
-> path | None`` instead of real dialogs.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QInputDialog, QLabel,
                               QMainWindow, QMenu, QMenuBar, QMessageBox, QProgressBar,
                               QVBoxLayout, QWidget)

from rsstatchem.core.imaging.commands import Cancelled, CommandError
from rsstatchem.core.imaging.menu_layout import AREAS, MENU_LAYOUT, PLAIN_KEYS, iter_items
from rsstatchem.core.imaging.options import Options
from rsstatchem.core.imaging.session import Session
from rsstatchem.ui.imagej import windows as W
from rsstatchem.ui.imagej.menus import build_menus, menu_text, update_enabled
from rsstatchem.ui.imagej.toolbar import (DROPPER, FREEROI, LINE, MAGNIFIER, OVAL, POINT,
                                          POLYGON, RECTANGLE, TEXT, WAND, ToolBar)

SETTINGS_KEY = "imagej/options"
RECENT_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".dcm",
                     ".fits", ".fit", ".fts", ".roi", ".zip", ".rsij", ".avi", ".lut", ".pgm",
                     ".ijm", ".txt", ".csv"}
POPUP_COMMANDS = ("Show Info...", "Properties...", "Rename...", None, "Measure", "Histogram",
                  "Duplicate...", "Original Scale", None, "Record...", "Find Commands...",
                  "Capture Screen")


def register_area_commands(registry) -> dict[str, str]:
    """Import every ``commands_<area>`` module and call its ``register(registry)``. A module
    that is missing is skipped; one that fails is reported (stderr) and skipped, so a broken
    work package never takes the bar down. Returns {module: traceback} of failures."""
    failures = {}
    for area in AREAS.values():
        try:
            module = importlib.import_module(area.module)
        except ModuleNotFoundError as exc:
            if exc.name == area.module:
                continue
            failures[area.module] = traceback.format_exc()
            continue
        except Exception:  # noqa: BLE001
            failures[area.module] = traceback.format_exc()
            continue
        try:
            module.register(registry)
        except Exception:  # noqa: BLE001
            failures[area.module] = traceback.format_exc()
    for name, tb in failures.items():
        print(f"[RS-StatChem] Image Analysis commands {name} failed:\n{tb}", file=sys.stderr)
    return failures


def _offscreen() -> bool:
    app = QApplication.instance()
    return app is not None and app.platformName() in ("offscreen", "minimal")


class ImageJBar(QMainWindow):
    def __init__(self, ctx=None, session: Session | None = None):
        super().__init__()
        self.ctx = ctx
        self.session = session or Session(self._load_options())
        self.message_hook = None      # tests: callable(title, text) instead of a message box
        self.question_hook = None     # tests: callable(title, text) -> "yes"|"no"|"cancel"
        self.file_hook = None         # tests: callable(kind, title, default) -> path | None
        self.installed_macros: dict[str, tuple[str, str]] = {}
        self.failed_modules = register_area_commands(self.session.registry)
        self.setWindowTitle("ImageJ")
        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(4, 2, 4, 4)
        lay.setSpacing(2)
        self.toolbar = ToolBar()
        o = self.session.options
        self.toolbar.set_colors(o.foreground, o.background)
        self.toolbar.options_requested.connect(self.tool_options)
        lay.addWidget(self.toolbar)
        row = QHBoxLayout()
        self.status = QLabel(" ")
        self.status.setMinimumWidth(300)
        row.addWidget(self.status, 1)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(120)
        self.progress.setMaximumHeight(12)
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 1000)
        self.progress.hide()
        row.addWidget(self.progress)
        lay.addLayout(row)
        self.setCentralWidget(central)
        self.actions_by_label, self.menus = build_menus(self.menuBar(), self.session.registry,
                                                        self.run_command)
        self.menus["Window"].aboutToShow.connect(self._fill_window_menu)
        self._window_fixed = list(self.menus["Window"].actions())
        self._fill_window_menu()
        recent = self.menus.get("File>Open Recent")
        if recent is not None:
            recent.aboutToShow.connect(self._fill_recent_menu)
            self._fill_recent_menu()
        self._macros_fixed = list(self.menus["Plugins>Macros"].actions()) \
            if "Plugins>Macros" in self.menus else []
        self._apply_custom_shortcuts()
        self.window_manager = W.WindowManager(self)
        self.setAcceptDrops(True)
        self.resize(560, 100)
        startup = o.extra.get("startup_macro")
        if startup:
            QTimer.singleShot(0, self, lambda: self.run_macro_text(startup, "Macro"))

    # ------------------------------------------------------------------ settings
    def _load_options(self) -> Options:
        settings = getattr(self.ctx, "settings", None)
        raw = settings.value(SETTINGS_KEY) if settings is not None else None
        try:
            return Options.from_dict(json.loads(raw)) if raw else Options()
        except (TypeError, ValueError):
            return Options()

    def _save_options(self) -> None:
        settings = getattr(self.ctx, "settings", None)
        if settings is not None:
            settings.setValue(SETTINGS_KEY, json.dumps(self.session.options.to_dict()))

    # ------------------------------------------------------------------ commands
    def refresh_menus(self) -> None:
        """Re-evaluate which menu entries are enabled (after registering commands)."""
        update_enabled(self.actions_by_label, self.session.registry)

    def run_command(self, label: str, options: str | None = None):
        """Run a command as the menu does (``options`` given = macro mode, no dialog)."""
        registry = self.session.registry
        cmd = registry.get(label)
        if cmd is None:
            self.error(f'Unrecognized command: "{label}"')
            return None
        history = self.session.__dict__.setdefault("command_history", [])
        history.append(label)
        del history[:-50]
        if cmd.ui_action is not None and options is None:
            try:
                return cmd.ui_action(self, cmd)
            except CommandError as exc:
                self.error(str(exc), label)
                return None
        image = self.session.active
        try:
            registry.check(self.session, cmd, image)
            values = None
            if options is None:
                fields = cmd.dialog_fields(self.session, image)
                if any(f.has_value or f.kind == "message" for f in fields):
                    from rsstatchem.ui.imagej.dialogs.generic import GenericDialog
                    title = label.rstrip(". ").removesuffix("...")
                    values = GenericDialog(title, fields, self).get()
                    if values is None:
                        return None
                else:
                    values = {}
            start = time.perf_counter()
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                result = registry.run(self.session, label, options or "", values=values)
            finally:
                QApplication.restoreOverrideCursor()
            self.show_status(f"{time.perf_counter() - start:.2f} seconds")
            return result
        except Cancelled:
            self.show_status("Cancelled")
        except CommandError as exc:
            self.error(str(exc), label)
        return None

    def error(self, text: str, title: str = "ImageJ") -> None:
        if self.message_hook is not None:
            self.message_hook(title, text)
        elif _offscreen():
            print(f"[ImageJ] {title}: {text}", file=sys.stderr)
        else:
            QMessageBox.warning(self, title, text)

    def ask(self, title: str, text: str) -> str:
        """"yes" / "no" / "cancel"."""
        if self.question_hook is not None:
            return self.question_hook(title, text)
        if _offscreen():
            return "no"
        b = QMessageBox.question(self, title, text, QMessageBox.StandardButton.Yes |
                                 QMessageBox.StandardButton.No |
                                 QMessageBox.StandardButton.Cancel)
        return {QMessageBox.StandardButton.Yes: "yes",
                QMessageBox.StandardButton.No: "no"}.get(b, "cancel")

    def ask_save_changes(self, doc) -> str:
        return self.ask("ImageJ", f'Save changes to "{doc.title}"?')

    def abort(self) -> None:
        """Esc: stop a running macro/command at its next progress check."""
        self.session.cancel_requested = True
        self.show_status("Aborted")

    # ------------------------------------------------------------------ dialogs
    def get_path(self, kind: str, title: str, default: str = "", filter: str = "") -> str | None:
        """File dialogs ("open", "save", "dir"); tests answer through ``file_hook``."""
        if self.file_hook is not None:
            return self.file_hook(kind, title, default)
        if _offscreen():
            return None
        if kind == "dir":
            path = QFileDialog.getExistingDirectory(self, title, default)
        elif kind == "save":
            path, _ = QFileDialog.getSaveFileName(self, title, default, filter)
        else:
            path, _ = QFileDialog.getOpenFileName(self, title, default, filter)
        return path or None

    def last_directory(self) -> str:
        im = self.session.active
        if im is not None and im.path:
            return str(Path(im.path).parent)
        return str(self.session.options.extra.get("last_dir", Path.home()))

    # ------------------------------------------------------------------ files
    def open_path(self, path: str, deferred_errors: bool = False):
        """File > Open of any supported file (images, ROIs, RoiSets, LUTs, tables, text,
        macros, .rsij projects)."""
        from rsstatchem.core.imaging.io import open_in_session

        try:
            opened = open_in_session(self.session, path)
        except Exception as exc:  # noqa: BLE001 - any reader failure is a user message
            msg = f'File is not in a supported format, a reader\nplugin is not available, ' \
                  f'or it was not found.\n \n{path}\n \n{exc}'
            if deferred_errors:
                QTimer.singleShot(0, self, lambda: self.error(msg, "Opener"))
            else:
                self.error(msg, "Opener")
            return None
        self.session.options.extra["last_dir"] = str(Path(path).parent)
        if self.ctx is not None and hasattr(self.ctx, "add_recent"):
            self.ctx.add_recent(path)
        self.record_call("open", Path(path).as_posix())
        return opened.value

    def save_image(self, doc, path: str | None = None, fmt: str = "tiff") -> bool:
        """File > Save (TIFF to the image's path or a chosen one); records ``saveAs``."""
        from rsstatchem.core.imaging.io import save_in_session

        if path is None:
            if doc.path and Path(doc.path).suffix.lower() in (".tif", ".tiff") and \
                    fmt == "tiff":
                path = doc.path
            else:
                stem = Path(doc.title).stem if "." in doc.title else doc.title
                path = self.get_path("save", "Save as TIFF",
                                     str(Path(self.last_directory()) / (stem + ".tif")),
                                     "TIFF (*.tif *.tiff)")
                if not path:
                    return False
        try:
            out = save_in_session(self.session, fmt, path, image=doc)
        except (OSError, ValueError) as exc:
            self.error(str(exc), "Save")
            return False
        self.record_call("saveAs", fmt.title() if fmt != "tiff" else "Tiff",
                         Path(out).as_posix())
        if self.ctx is not None and hasattr(self.ctx, "add_recent"):
            self.ctx.add_recent(str(out))
        return True

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        for url in event.mimeData().urls():
            if url.isLocalFile():
                p = url.toLocalFile()
                if Path(p).is_dir():
                    from rsstatchem.core.imaging.io.sequence import open_sequence
                    try:
                        self.session.add_image(open_sequence(p))
                    except ValueError as exc:
                        self.error(str(exc), "Image Sequence")
                else:
                    self.open_path(p)

    # ------------------------------------------------------------------ recording
    def record_call(self, name: str, *args) -> None:
        rec = self.session.recorder
        if rec is not None:
            from rsstatchem.core.imaging.macro import call_line
            rec.record_line(call_line(rec.language, name, *args))

    def record_roi(self, roi) -> None:
        rec = self.session.recorder
        if rec is not None:
            from rsstatchem.core.imaging.macro import roi_line
            line = roi_line(rec.language, roi)
            if line:
                rec.record_line(line)

    # ------------------------------------------------------------------ macros
    def macro_hooks(self):
        from rsstatchem.core.imaging.macro import UIHooks
        if _offscreen() and self.message_hook is None:
            return UIHooks(open_path=self.open_path, set_tool=self.toolbar.set_tool)

        def number(prompt, default):
            v, ok = QInputDialog.getDouble(self, "Macro", prompt, default, -1e12, 1e12, 4)
            return v if ok else None

        def string(prompt, default):
            v, ok = QInputDialog.getText(self, "Macro", prompt, text=default)
            return v if ok else None

        def boolean(prompt):
            a = self.ask("Macro", prompt)
            return None if a == "cancel" else a == "yes"

        return UIHooks(get_number=number, get_string=string, get_boolean=boolean,
                       show_message=lambda t, m: self.error(m, t), open_path=self.open_path,
                       set_tool=self.toolbar.set_tool)

    def run_macro_text(self, code: str, language: str = "Macro", path: str | None = None,
                       macro_name: str | None = None, argument: str = ""):
        from rsstatchem.core.imaging.macro import MacroError, run_macro, run_python
        start = time.perf_counter()
        try:
            if language == "Python":
                result = run_python(self.session, code, self.macro_hooks(), path)
            else:
                result = run_macro(self.session, code, argument, self.macro_hooks(), path,
                                   macro_name)
        except MacroError as exc:
            self.error(str(exc), "Macro Error")
            return None
        except CommandError as exc:
            self.error(str(exc), "Macro Error")
            return None
        except Exception:  # noqa: BLE001 - script errors are shown, not raised
            self.error(traceback.format_exc(limit=4), "Script Error")
            return None
        self.show_status(f"{time.perf_counter() - start:.2f} seconds")
        return result

    def run_macro_file(self, path: str, argument: str = ""):
        p = Path(path)
        code = p.read_text(errors="replace")
        lang = "Python" if p.suffix.lower() == ".py" else "Macro"
        if lang == "Macro":
            self.record_call("runMacro", p.as_posix())
        return self.run_macro_text(code, lang, str(p), argument=argument)

    def install_macros(self, code: str, title: str = "") -> list[str]:
        """Plugins > Macros > Install...: ``macro "Name" {...}`` blocks become entries at the
        end of Plugins > Macros."""
        from rsstatchem.core.imaging.macro import MacroError, macro_names
        try:
            names = macro_names(code)
        except MacroError as exc:
            self.error(str(exc), "Install")
            return []
        self.installed_macros = {n: (code, title) for n in names}
        menu = self.menus.get("Plugins>Macros")
        if menu is not None:
            for act in list(menu.actions()):
                if act not in self._macros_fixed:
                    menu.removeAction(act)
            if names:
                menu.addSeparator()
            for n in names:
                act = menu.addAction(menu_text(n))
                act.triggered.connect(lambda _=False, name=n: self.run_installed(name))
        self.show_status(f"{len(names)} macros installed")
        return names

    def run_installed(self, name: str):
        code, title = self.installed_macros[name]
        return self.run_macro_text(code, "Macro", title or None, macro_name=name)

    # ------------------------------------------------------------------ keys & tools
    def handle_key(self, event) -> bool:
        """ImageJ's unmodified single-key shortcuts (``IJ.KeyListener`` semantics)."""
        mods = event.modifiers()
        if mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier |
                   Qt.KeyboardModifier.AltModifier):
            return False
        if self.session.options.require_control_key:
            return False
        text = event.text()
        if not text or len(text) != 1:
            return False
        label = PLAIN_KEYS.get(text)
        if label is None and text.isalnum():
            for _, item in iter_items():
                if item.shortcut == text:
                    label = item.label
                    break
        custom = self.session.options.extra.get("shortcuts", {})
        label = custom.get(text, label)
        if label is None:
            return False
        cmd = self.session.registry.get(label)
        if cmd is None or not cmd.available:
            return False
        self.run_command(label)
        return True

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if not self.handle_key(event):
            super().keyPressEvent(event)

    def tool_options(self, tool: int) -> None:
        """Double-click on a tool icon."""
        variant = self.toolbar.variant.get(tool, "")
        label = {RECTANGLE: "Rounded Rect Tool..." if variant == "Rounded Rectangle"
                 else "Roi Defaults...", OVAL: "Roi Defaults...", POLYGON: "Roi Defaults...",
                 FREEROI: "Roi Defaults...", LINE: "Arrow Tool..." if variant == "Arrow tool"
                 else "Line Width...", POINT: "Point Tool...", WAND: "Wand Tool...",
                 TEXT: "Fonts..."}.get(tool)
        if tool == OVAL and variant == "Selection Brush Tool":
            o = self.session.options
            if _offscreen():
                return
            v, ok = QInputDialog.getInt(self, "Selection Brush", "Brush size (pixels):",
                                        int(o.extra.get("brush_size", 15)), 1, 500)
            if ok:
                o.extra["brush_size"] = v
            return
        if tool == DROPPER:
            from rsstatchem.ui.imagej.frames.color_picker import show_color_picker
            show_color_picker(self)
            return
        if tool == MAGNIFIER:
            win = W.window_for(self.session.active) if self.session.active else None
            if win is not None:
                win.canvas.original_scale()
            return
        if label is not None and label in self.session.registry:
            self.run_command(label)

    def set_color(self, color: str, background: bool = False) -> None:
        o = self.session.options
        if background:
            o.background = color
        else:
            o.foreground = color
        self.toolbar.set_colors(o.foreground, o.background)
        self.session.emit("options_changed", "colors")
        rgb = tuple(int(color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        self.record_call("setBackgroundColor" if background else "setForegroundColor", *rgb)

    def image_popup(self, window, global_pos) -> QMenu:
        menu = QMenu(window)
        for label in POPUP_COMMANDS:
            if label is None:
                menu.addSeparator()
            elif label in self.actions_by_label:
                menu.addAction(self.actions_by_label[label])
        if not _offscreen():
            menu.popup(global_pos)
        return menu

    def _apply_custom_shortcuts(self) -> None:
        from PySide6.QtGui import QKeySequence
        for key, label in self.session.options.extra.get("shortcuts", {}).items():
            act = self.actions_by_label.get(label)
            if act is not None and len(key) > 1:     # F-keys etc.; single keys via handle_key
                act.setShortcut(QKeySequence(key))

    # ------------------------------------------------------------------ status & windows
    def show_status(self, text: str) -> None:
        self.status.setText(text)

    def show_progress(self, fraction) -> None:
        if isinstance(fraction, tuple):
            fraction = fraction[0]
        fraction = float(fraction)
        if fraction >= 1.0 or fraction < 0:
            self.progress.hide()
        else:
            self.progress.show()
            self.progress.setValue(int(fraction * 1000))
        QApplication.processEvents()

    def shortcut_actions(self) -> list:
        """Actions with keyboard shortcuts (added to image windows so keys work there)."""
        return [a for a in self.actions_by_label.values() if not a.shortcut().isEmpty()]

    def image_window_menubar(self, window):
        """On macOS an image window gets a QMenuBar holding the bar's menus (the same QMenus
        and QActions), so the global menu bar keeps ImageJ's menus while it is active."""
        if sys.platform != "darwin" or os.environ.get("RSSTATCHEM_NO_WINDOW_MENUBAR"):
            return None
        mb = QMenuBar(window)
        for top in MENU_LAYOUT:
            menu = self.menus.get(top.title)
            if menu is not None:
                mb.addMenu(menu)
        return mb

    def image_window_closed(self, window) -> None:
        W.unregister(window)
        if window.doc in self.session.images:
            self.session.close_image(window.doc)

    def _fill_window_menu(self) -> None:
        from rsstatchem.ui.modules import add_module_menu

        m = self.menus["Window"]
        for act in list(m.actions()):
            if act not in self._window_fixed:
                m.removeAction(act)
        m.addSeparator()
        act = m.addAction("ImageJ")
        act.triggered.connect(lambda: (self.show(), self.raise_(), self.activateWindow()))
        for win in list(W.WINDOWS):
            doc = getattr(win, "doc", None)
            title = doc.title if doc is not None else win.windowTitle()
            act = m.addAction(menu_text(title))
            act.setCheckable(True)
            act.setChecked(doc is not None and doc is self.session.active)
            act.triggered.connect(lambda _=False, w=win: (w.show(), w.raise_(),
                                                          w.activateWindow()))
        if self.ctx is not None:
            m.addSeparator()
            add_module_menu(m, self.ctx)

    def recent_files(self) -> list[str]:
        if self.ctx is None or not hasattr(self.ctx, "recent_files"):
            return []
        return [p for p in self.ctx.recent_files() if Path(p).suffix.lower() in
                RECENT_EXTENSIONS]

    def _fill_recent_menu(self) -> None:
        menu = self.menus.get("File>Open Recent")
        if menu is None:
            return
        menu.clear()
        files = self.recent_files()
        for p in files[:15]:
            act = QAction(menu_text(Path(p).name), menu)
            act.setToolTip(p)
            act.triggered.connect(lambda _=False, path=p: self.open_path(path))
            menu.addAction(act)
        if not files:
            act = menu.addAction("(no recent files)")
            act.setEnabled(False)

    def closeEvent(self, event) -> None:  # noqa: N802
        """Closing the bar closes every image and auxiliary window of the module, asking
        "Save changes to ...?" for changed images (Cancel keeps everything open)."""
        for doc in list(self.session.images):
            if doc.changed:
                answer = self.ask_save_changes(doc)
                if answer == "cancel" or (answer == "yes" and not self.save_image(doc)):
                    event.ignore()
                    return
        for win in list(W.WINDOWS):
            W.unregister(win)
            if hasattr(win, "bar"):
                win.bar = None
            win.close()
        self.session.images.clear()
        self.session.recorder = None
        self._save_options()
        if self in W.BARS:
            W.BARS.remove(self)
        event.accept()
        if self.ctx is not None and hasattr(self.ctx, "window_closed"):
            self.ctx.window_closed(self)

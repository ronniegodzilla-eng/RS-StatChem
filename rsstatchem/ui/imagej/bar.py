"""The ImageJ bar: the small main window with the menus, the tool icons and the status line.
It is the module window registered with the app context (``self.ctx``; ``ctx.window_closed``
when it closes; Window > Modules). **Owned by WP-A after Phase 0.**

Command plumbing (contract): every menu action calls :meth:`ImageJBar.run_command` with the
command label. A command with ``ui_action`` gets ``ui_action(bar, command)``; otherwise the
bar shows the :class:`~rsstatchem.ui.imagej.dialogs.generic.GenericDialog` of its fields (if
any) and runs it through the registry (recorded, undoable, errors shown as ImageJ messages).
"""
from __future__ import annotations

import importlib
import json
import sys
import time
import traceback
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
                               QProgressBar, QVBoxLayout, QWidget)

from rsstatchem.core.imaging.commands import Cancelled, CommandError
from rsstatchem.core.imaging.menu_layout import AREAS
from rsstatchem.core.imaging.options import Options
from rsstatchem.core.imaging.session import Session
from rsstatchem.ui.imagej import windows as W
from rsstatchem.ui.imagej.menus import build_menus, menu_text, update_enabled
from rsstatchem.ui.imagej.toolbar import ToolBar

SETTINGS_KEY = "imagej/options"


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


class ImageJBar(QMainWindow):
    def __init__(self, ctx=None, session: Session | None = None):
        super().__init__()
        self.ctx = ctx
        self.session = session or Session(self._load_options())
        self.message_hook = None      # tests: callable(title, text) instead of a message box
        self.failed_modules = register_area_commands(self.session.registry)
        self.setWindowTitle("ImageJ")
        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(4, 2, 4, 4)
        lay.setSpacing(2)
        self.toolbar = ToolBar()
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
        self.window_manager = W.WindowManager(self)
        self.setAcceptDrops(True)
        self.resize(560, 100)

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
        if cmd.ui_action is not None and options is None:
            return cmd.ui_action(self, cmd)
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
        else:
            QMessageBox.warning(self, title, text)

    # ------------------------------------------------------------------ files
    def open_path(self, path: str, deferred_errors: bool = False):
        """Open an image file (WP-A extends to ROIs, tables, LUTs, projects)."""
        from rsstatchem.core.imaging.io import open_image

        try:
            doc = open_image(path)
        except Exception as exc:  # noqa: BLE001 - any reader failure is a user message
            msg = f'File is not in a supported format, a reader\nplugin is not available, ' \
                  f'or it was not found.\n \n{path}\n \n{exc}'
            if deferred_errors:
                QTimer.singleShot(0, self, lambda: self.error(msg, "Opener"))
            else:
                self.error(msg, "Opener")
            return None
        self.session.add_image(doc)
        if self.ctx is not None and hasattr(self.ctx, "add_recent"):
            self.ctx.add_recent(path)
        if self.session.recorder is not None:
            self.session.recorder.record_line(f'open("{Path(path).as_posix()}");')
        return doc

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        for url in event.mimeData().urls():
            if url.isLocalFile():
                self.open_path(url.toLocalFile())

    # ------------------------------------------------------------------ status & windows
    def show_status(self, text: str) -> None:
        self.status.setText(text)

    def show_progress(self, fraction: float) -> None:
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
        """Menu bar for an image window: None in Phase 0 (WP-A: on macOS give image windows
        a QMenuBar sharing these QActions so the global menu shows ImageJ's menus)."""
        return None

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
            act = m.addAction(menu_text(doc.title if doc is not None else win.windowTitle()))
            act.setCheckable(True)
            act.setChecked(doc is not None and doc is self.session.active)
            act.triggered.connect(lambda _=False, w=win: (w.show(), w.raise_(),
                                                          w.activateWindow()))
        if self.ctx is not None:
            m.addSeparator()
            add_module_menu(m, self.ctx)

    def closeEvent(self, event) -> None:  # noqa: N802
        """Closing the bar closes every image and auxiliary window of the module (WP-A adds
        "Save changes to ...?" for changed images; cancelling must ``event.ignore()``)."""
        for win in list(W.WINDOWS):
            W.unregister(win)
            if hasattr(win, "bar"):
                win.bar = None
            win.close()
        self.session.images.clear()
        self._save_options()
        if self in W.BARS:
            W.BARS.remove(self)
        event.accept()
        if self.ctx is not None and hasattr(self.ctx, "window_closed"):
            self.ctx.window_closed(self)

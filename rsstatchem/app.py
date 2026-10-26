"""Application bootstrap and the shared context (settings, Output Viewer, open editors)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from rsstatchem.paths import is_frozen, resource, settings_file, user_data_dir

# Keep matplotlib's font cache inside our data folder: portable mode writes nothing to HOME, and
# PyInstaller's runtime hook would otherwise point it at a fresh temp dir (slow rebuild each run).
if is_frozen() or "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(user_data_dir() / "matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from PySide6.QtCore import QEvent, QObject, QPoint, QSettings, Qt, QTimer  # noqa: E402
from PySide6.QtGui import QIcon, QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication, QSplashScreen  # noqa: E402

from rsstatchem import APP_ID, APP_NAME, APP_ORG, COPYRIGHT, __version__  # noqa: E402
from rsstatchem.core.dataset import Dataset  # noqa: E402
from rsstatchem.core.output import Section  # noqa: E402

MAX_RECENT = 9


class AppContext:
    def __init__(self, app: QApplication):
        self.app = app
        self.settings = QSettings(str(settings_file()), QSettings.Format.IniFormat)
        self.editors: list = []   # Statistics data editors
        self.windows: list = []   # main windows of every module (editors included)
        self._launcher = None
        self._output = None
        self._updates = None
        self._counter = 0

    # ------------------------------------------------------------------ editors
    def new_editor(self, dataset: Dataset | None = None):
        from rsstatchem.ui.data_editor import DataEditorWindow

        self._counter += 1
        editor = DataEditorWindow(self, dataset, self._counter)
        if dataset is not None:
            editor.load_dataset(dataset)
        self.editors.append(editor)
        self.register_window(editor)
        if len(self.editors) > 1:
            last = self.editors[-2]
            editor.move(last.pos() + QPoint(28, 28))
        editor.show()
        return editor

    def show_dataset(self, dataset: Dataset, reuse=None):
        if reuse is not None:
            reuse.load_dataset(dataset)
            reuse.raise_()
            return reuse
        return self.new_editor(dataset)

    def editor_closed(self, editor) -> None:
        if editor in self.editors:
            self.editors.remove(editor)
        if not self.editors and self._output is not None:
            self._output.close()
        self.window_closed(editor)

    # ------------------------------------------------------------------ module windows
    def register_window(self, window) -> None:
        if window not in self.windows:
            self.windows.append(window)

    def window_closed(self, window) -> None:
        """Called by a module window once its close is accepted; the app quits with the last
        window (the launcher counts as one while it is open)."""
        if window in self.windows:
            self.windows.remove(window)
        if not self.windows and not (self._launcher is not None and self._launcher.isVisible()):
            if self._output is not None:
                self._output.close()
            self.app.quit()

    def show_launcher(self):
        if self._launcher is None:
            from rsstatchem.ui.launcher import LauncherWindow

            self._launcher = LauncherWindow(self)
        self._launcher.show()
        self._launcher.raise_()
        self._launcher.activateWindow()
        return self._launcher

    def launcher_closed(self) -> None:
        if not self.windows:
            if self._output is not None:
                self._output.close()
            self.app.quit()

    def open_module(self, key: str, path: str | None = None):
        from rsstatchem.ui.modules import open_module

        return open_module(self, key, path)

    # ------------------------------------------------------------------ output
    def output(self):
        if self._output is None:
            from rsstatchem.ui.output_viewer import OutputViewer

            self._output = OutputViewer(self.settings)
            self._output.window_menu.aboutToShow.connect(self._fill_output_window_menu)
        return self._output

    def _fill_output_window_menu(self) -> None:
        menu = self._output.window_menu
        menu.clear()
        for editor in self.editors:
            act = menu.addAction(editor.windowTitle().replace("[*]", ""))
            act.triggered.connect(lambda _=False, e=editor: (e.show(), e.raise_(),
                                                             e.activateWindow()))

    def show_output_window(self) -> None:
        viewer = self.output()
        viewer.show()
        viewer.raise_()
        viewer.activateWindow()

    def show_output(self, section: Section) -> None:
        self.output().add_section(section)
        self.show_output_window()

    # ------------------------------------------------------------------ misc
    def recent_files(self) -> list[str]:
        value = self.settings.value("recent_files", [])
        if isinstance(value, str):
            value = [value]
        return [p for p in (value or []) if Path(p).exists()]

    def add_recent(self, path: str) -> None:
        path = str(Path(path).resolve())
        recent = [p for p in self.recent_files() if p != path]
        self.settings.setValue("recent_files", [path] + recent[:MAX_RECENT - 1])

    def updates(self):
        if self._updates is None:
            from rsstatchem.ui.update_dialog import UpdateChecker

            self._updates = UpdateChecker(self.settings, self._dialog_parent)
        return self._updates

    def _dialog_parent(self):
        active = self.app.activeWindow()
        return active if active is not None else (self.windows[-1] if self.windows else None)

    def check_for_updates(self, manual: bool = False) -> None:
        if manual or self.updates().due():
            self.updates().check(manual)

    def about(self) -> None:
        from rsstatchem.ui.dialogs.about import AboutDialog

        parent = self.app.activeWindow()
        AboutDialog(parent).exec()

    def open_file(self, path: str) -> None:
        from rsstatchem.ui.modules import BY_KEY, module_for_path, open_path

        info = module_for_path(path)
        if info is None or info is BY_KEY["stats"]:
            active = self.app.activeWindow()
            if active in self.editors:  # a data file dropped on an open editor goes there
                active.open_path(path)
                return
        open_path(self, path)


class _FileOpenFilter(QObject):
    """macOS delivers Finder double-clicks / drops on the Dock icon as QFileOpenEvent."""

    def __init__(self, ctx: AppContext):
        super().__init__()
        self.ctx = ctx

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Type.FileOpen:
            self.ctx.open_file(event.file())
            return True
        return False


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    if "--check-update" in argv:  # diagnostics: does the (packaged) app reach GitHub?
        from rsstatchem.updates import UpdateError, fetch_latest

        url = next((a.split("=", 1)[1] for a in argv if a.startswith("--update-url=")), None)
        try:
            info = fetch_latest(url) if url else fetch_latest()
        except UpdateError as exc:
            print(f"update check: {exc}")
            return 2
        print(f"update check: latest {info.version} (this is {__version__}); "
              f"newer={info.newer}; download={info.download_url}")
        return 0
    if "--selftest" in argv:
        from rsstatchem.selftest import run

        return run()
    if sys.platform == "win32":
        try:  # own taskbar icon instead of python.exe's
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
        except (AttributeError, OSError):
            pass
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_ORG)
    app.setApplicationVersion(__version__)
    app.setWindowIcon(QIcon(str(resource("icon.png"))))
    app.setQuitOnLastWindowClosed(False)

    from rsstatchem.ui.theme import apply_theme

    apply_theme(app)

    splash = None
    if not os.environ.get("RSSTATCHEM_NO_SPLASH"):
        pixmap = QPixmap(str(resource("logo_small.png")))
        splash = QSplashScreen(pixmap)
        splash.showMessage(f"Version {__version__}  ·  {COPYRIGHT}  ·  loading…",
                           Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight,
                           Qt.GlobalColor.darkGray)
        splash.show()
        app.processEvents()

    ctx = AppContext(app)
    file_filter = _FileOpenFilter(ctx)
    app.installEventFilter(file_filter)
    files = [arg for arg in argv[1:] if Path(arg).is_file()]
    for path in files:  # opened from Finder/Explorer: straight into the file's module
        ctx.open_file(path)
    first = ctx.windows[0] if ctx.windows else ctx.show_launcher()
    if splash is not None:
        # `splash` as context object: the callback is dropped if the splash is already gone
        QTimer.singleShot(700, splash, lambda: splash.finish(first))
    if not os.environ.get("RSSTATCHEM_NO_UPDATE_CHECK"):
        QTimer.singleShot(4000, first, lambda: ctx.check_for_updates(manual=False))
    app._rs_context = ctx  # keep a strong reference for the lifetime of the app
    try:
        return app.exec()
    finally:
        # Tear everything Qt down while the interpreter is intact. Windows left alive (closed
        # editors are only hidden) would otherwise be deleted by PySide6's atexit hook in an
        # arbitrary order, which crashes on quit.
        from rsstatchem.ui.qt_lifecycle import shutdown_application

        app.removeEventFilter(file_filter)
        ctx.editors.clear()
        ctx.windows.clear()
        ctx._launcher = None
        ctx._output = None
        ctx._updates = None
        del app._rs_context
        del splash, first, ctx, file_filter
        shutdown_application(app)

"""Help > Check for Updates… and the silent check at startup.

The network request runs on Qt's global thread pool (which ``qt_lifecycle.shutdown_application``
waits for at exit) and reports back to the GUI thread through a queued signal.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta

from PySide6.QtCore import QObject, QThreadPool, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QCheckBox, QDialog, QHBoxLayout, QLabel, QMessageBox,
                               QPushButton, QTextBrowser, QVBoxLayout)

from rsstatchem import APP_NAME, __version__
from rsstatchem.updates import UpdateError, UpdateInfo, fetch_latest

AUTO_KEY = "updates/auto_check"
LAST_KEY = "updates/last_check"
SKIP_KEY = "updates/skip_version"
CHECK_EVERY = timedelta(hours=20)


def _install_hint() -> str:
    if sys.platform == "darwin":
        return ("Unzip the download and replace RS-StatChem in your Applications folder. "
                "Your settings are kept.")
    if sys.platform == "win32":
        return ("Unzip the download and replace the RS-StatChem folder (keep or copy its "
                "UserData folder to keep your settings).")
    return "Unzip the download and replace your current copy."


class UpdateDialog(QDialog):
    def __init__(self, info: UpdateInfo, settings, parent=None):
        super().__init__(parent)
        self.info, self.settings = info, settings
        self.setWindowTitle("Software Update")
        self.resize(560, 440)
        title = QLabel(f"<h3 style='color:#B71C1C;margin:0'>{APP_NAME} {info.version} is "
                       f"available</h3>You have version {__version__}.")
        notes = QTextBrowser()
        notes.setOpenExternalLinks(True)
        notes.setMarkdown(info.notes or "_No release notes._")
        hint = QLabel(_install_hint())
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#555;")
        self.auto = QCheckBox("Check for updates automatically")
        self.auto.setChecked(settings.value(AUTO_KEY, True, type=bool))

        download = QPushButton("Download")
        download.setDefault(True)
        download.clicked.connect(self._download)
        skip = QPushButton("Skip This Version")
        skip.clicked.connect(self._skip)
        later = QPushButton("Remind Me Later")
        later.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addWidget(skip)
        buttons.addStretch()
        buttons.addWidget(later)
        buttons.addWidget(download)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(QLabel("Release notes:"))
        layout.addWidget(notes, 1)
        layout.addWidget(hint)
        layout.addWidget(self.auto)
        layout.addLayout(buttons)

    def done(self, result: int) -> None:
        self.settings.setValue(AUTO_KEY, self.auto.isChecked())
        super().done(result)

    def _download(self) -> None:
        QDesktopServices.openUrl(QUrl(self.info.download_url or self.info.page_url))
        self.accept()

    def _skip(self) -> None:
        self.settings.setValue(SKIP_KEY, self.info.version)
        self.reject()


class UpdateChecker(QObject):
    finished = Signal(object, object)  # UpdateInfo | None, error message | None

    def __init__(self, settings, parent_window=None, fetch=fetch_latest):
        super().__init__()
        self.settings = settings
        self.parent_window = parent_window  # callable returning the window to parent dialogs
        self._fetch = fetch
        self._busy = False
        self._manual = False
        self.dialog: UpdateDialog | None = None
        self.finished.connect(self._done)

    def due(self, now: datetime | None = None) -> bool:
        if not self.settings.value(AUTO_KEY, True, type=bool):
            return False
        last = self.settings.value(LAST_KEY, "")
        try:
            return (now or datetime.now()) - datetime.fromisoformat(str(last)) >= CHECK_EVERY
        except ValueError:
            return True

    def check(self, manual: bool = False) -> None:
        if self._busy:
            return
        self._busy, self._manual = True, manual
        QThreadPool.globalInstance().start(self._work)

    def _work(self) -> None:  # thread pool
        try:
            info, error = self._fetch(), None
        except UpdateError as exc:
            info, error = None, str(exc)
        except Exception as exc:  # noqa: BLE001 - never let a check crash the pool thread
            info, error = None, f"Unexpected error: {exc}"
        try:
            self.finished.emit(info, error)
        except RuntimeError:  # the checker was deleted while the request was running
            pass

    def _parent(self):
        return self.parent_window() if callable(self.parent_window) else None

    def _done(self, info: UpdateInfo | None, error: str | None) -> None:
        self._busy = False
        manual = self._manual
        if error:
            if manual:
                QMessageBox.warning(self._parent(), "Software Update",
                                    f"Could not check for updates.\n\n{error}")
            return
        self.settings.setValue(LAST_KEY, datetime.now().isoformat(timespec="seconds"))
        if not info.newer:
            if manual:
                QMessageBox.information(self._parent(), "Software Update",
                                        f"{APP_NAME} {__version__} is the latest version.")
            return
        if not manual and str(self.settings.value(SKIP_KEY, "")) == info.version:
            return
        self.dialog = UpdateDialog(info, self.settings, self._parent())
        self.dialog.show()
        self.dialog.raise_()

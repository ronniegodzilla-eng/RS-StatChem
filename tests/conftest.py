"""Shared pytest setup: one QApplication for the session and deterministic Qt teardown.

Widgets that are still alive when the interpreter exits are deleted by PySide6's atexit hook in
an arbitrary order, which crashes the process after every test has passed (SIGSEGV / exit 139,
"Windows fatal exception: access violation" with no Python frame). So every test's leftover
windows are deleted right after it, and the QApplication is shut down at the end of the session
while Python is still intact. See ``rsstatchem/ui/qt_lifecycle.py``.
"""
from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _running_app():
    if "PySide6.QtWidgets" not in sys.modules:
        return None
    from PySide6.QtWidgets import QApplication

    return QApplication.instance()


@pytest.fixture(scope="session")
def qapp():
    """The session's QApplication (Qt tests take this instead of creating their own)."""
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _dispose_qt_windows():
    """Delete the windows and dialogs a test left behind (closed ones too: a closed window is
    only hidden, and several of ours are kept alive by signal connections to lambdas)."""
    yield
    app = _running_app()
    if app is not None:
        from rsstatchem.ui.qt_lifecycle import dispose_windows

        dispose_windows(app)


def pytest_sessionfinish(session, exitstatus) -> None:
    app = _running_app()
    if app is not None:
        from rsstatchem.ui.qt_lifecycle import shutdown_application

        shutdown_application(app)

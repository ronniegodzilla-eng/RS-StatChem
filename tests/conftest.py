"""Shared pytest setup.

Several tests create Qt windows offscreen. On some Python builds (seen with the python.org
framework build on the macOS CI runner) PySide6 segfaults while the interpreter destroys the
remaining C++ objects at exit — after every test has passed — which turns a green run into
exit code 139. We close the windows ourselves and then leave without that teardown; pytest
has already printed its summary and we keep its exit status.
"""
from __future__ import annotations

import os
import sys

import pytest

_exit_status: int | None = None


def pytest_sessionfinish(session, exitstatus) -> None:
    global _exit_status
    if sys.platform == "win32":  # the crash is not seen there, and forcing the exit made
        return                   # the Windows runner report exit code 1 on a passing run
    _exit_status = int(exitstatus)
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        return
    app = QApplication.instance()
    if app is None:
        return
    for widget in app.topLevelWidgets():
        widget.close()
        widget.deleteLater()
    app.processEvents()


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config) -> None:
    if _exit_status is None:
        return
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_exit_status)

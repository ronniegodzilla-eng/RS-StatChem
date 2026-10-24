"""Shared pytest setup.

Several tests create Qt windows offscreen. On the CI runners PySide6 can crash while the
interpreter destroys the remaining C++ objects at exit — after every test has passed — which
turns a green run into a failure (SIGSEGV / exit 139 on macOS, exit 1 on Windows). Closing the
windows first also crashed on Windows. pytest has already printed its summary by the time
``pytest_unconfigure`` runs, so we leave right there with pytest's own exit status and skip the
interpreter teardown entirely.
"""
from __future__ import annotations

import os
import sys

import pytest

_exit_status: int | None = None


def pytest_sessionfinish(session, exitstatus) -> None:
    global _exit_status
    _exit_status = int(exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config) -> None:
    if _exit_status is None:
        return
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_exit_status)

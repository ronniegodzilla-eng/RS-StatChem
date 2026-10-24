"""Shared pytest setup.

Several tests create Qt windows offscreen. On the CI runners PySide6 can crash while the
interpreter destroys the remaining C++ objects at exit — after every test has passed — which
turns a green run into a failure (SIGSEGV / exit 139 on both macOS and Windows). Closing the
windows first also crashed on Windows. pytest has already printed its summary by the time
``pytest_unconfigure`` runs, so we leave right there with pytest's own exit status and skip the
interpreter (and, on Windows, DLL) teardown entirely.
"""
from __future__ import annotations

import os
import sys

import pytest

_exit_status: int | None = None


def pytest_sessionfinish(session, exitstatus) -> None:
    global _exit_status
    if os.environ.get("RS_TEST_NATURAL_EXIT"):  # diagnostics: let the interpreter tear down
        return
    _exit_status = int(exitstatus)


@pytest.hookimpl(tryfirst=True)  # before pytest's own cleanup, which runs gc.collect()
def pytest_unconfigure(config) -> None:
    if _exit_status is None:
        return
    print(f"\n[conftest] leaving with pytest exit status {_exit_status}", flush=True)
    sys.stderr.flush()
    if sys.platform == "win32":
        # os._exit still runs every DLL's detach routine, and Qt's crashes there (seen as
        # SIGSEGV / exit 139 on the runner). TerminateProcess skips DLL teardown entirely.
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.TerminateProcess(kernel32.GetCurrentProcess(), _exit_status)
    os._exit(_exit_status)

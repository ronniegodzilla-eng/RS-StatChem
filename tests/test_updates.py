import json
import threading
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from rsstatchem import __version__, updates
from rsstatchem.updates import UpdateError, info_from_release, parse_version

RELEASE = {
    "tag_name": "v99.0.0",
    "html_url": "https://github.com/example/releases/tag/v99.0.0",
    "body": "## What's new\n- Faster plots",
    "published_at": "2026-10-25T08:00:00Z",
    "assets": [
        {"name": "RS-StatChem-macOS-arm64.zip", "browser_download_url": "https://x/mac.zip"},
        {"name": "RS-StatChem-Windows-x64-portable.zip",
         "browser_download_url": "https://x/win.zip"},
    ],
}


def test_parse_and_compare_versions():
    assert parse_version("v0.4.2") == (0, 4, 2)
    assert parse_version("0.5") == parse_version("0.5.0")
    assert parse_version("0.10.0") > parse_version("0.9.9")
    assert parse_version("1.0.0-rc1") == (1,)
    assert parse_version("garbage") == (0,)


@pytest.mark.parametrize("keyword, url", [("macOS-arm64", "https://x/mac.zip"),
                                          ("Windows-x64", "https://x/win.zip"),
                                          ("Linux", None)])
def test_asset_for_this_platform(monkeypatch, keyword, url):
    monkeypatch.setattr(updates, "asset_keyword", lambda: keyword)
    info = info_from_release(RELEASE)
    assert info.version == "99.0.0" and info.newer
    assert info.download_url == url
    assert "Faster plots" in info.notes


def test_current_version_is_not_newer():
    info = info_from_release({**RELEASE, "tag_name": f"v{__version__}"})
    assert not info.newer


class _Handler(BaseHTTPRequestHandler):
    status, payload = 200, json.dumps(RELEASE).encode()

    def do_GET(self):  # noqa: N802
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}/latest"
    httpd.shutdown()
    httpd.server_close()


def test_fetch_latest_success_and_errors(server):
    assert updates.fetch_latest(server).version == "99.0.0"
    _Handler.status = 404
    try:
        with pytest.raises(UpdateError, match="No published release"):
            updates.fetch_latest(server)
    finally:
        _Handler.status = 200
    with pytest.raises(UpdateError, match="Could not reach GitHub"):
        updates.fetch_latest("http://127.0.0.1:9/nothing", timeout=2)


# ------------------------------------------------------------------ Qt flow
@pytest.fixture
def checker_env(qapp, tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings

    from rsstatchem.ui import update_dialog

    messages = []

    class FakeBox:
        @staticmethod
        def information(parent, title, text):
            messages.append(("info", text))

        @staticmethod
        def warning(parent, title, text):
            messages.append(("warning", text))

    monkeypatch.setattr(update_dialog, "QMessageBox", FakeBox)
    settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    return update_dialog, settings, messages


def _run(checker, manual):
    from PySide6.QtCore import QCoreApplication, QThreadPool

    checker.check(manual)
    QThreadPool.globalInstance().waitForDone()
    QCoreApplication.processEvents()  # deliver the queued "finished" signal


def test_newer_release_opens_dialog_and_skip_is_respected(checker_env):
    update_dialog, settings, messages = checker_env
    checker = update_dialog.UpdateChecker(settings, fetch=lambda: info_from_release(RELEASE))
    _run(checker, manual=False)
    assert checker.dialog is not None and "99.0.0" in checker.dialog.windowTitle() + \
        checker.dialog.findChildren(update_dialog.QLabel)[0].text()
    checker.dialog._skip()
    assert settings.value(update_dialog.SKIP_KEY) == "99.0.0"

    checker.dialog = None
    _run(checker, manual=False)  # skipped version: silent
    assert checker.dialog is None
    _run(checker, manual=True)  # an explicit check still shows it
    assert checker.dialog is not None


def test_up_to_date_and_error_messages_only_when_manual(checker_env):
    update_dialog, settings, messages = checker_env
    current = info_from_release({**RELEASE, "tag_name": f"v{__version__}"})
    checker = update_dialog.UpdateChecker(settings, fetch=lambda: current)
    _run(checker, manual=False)
    assert messages == [] and checker.dialog is None
    _run(checker, manual=True)
    assert messages[-1][0] == "info" and "latest version" in messages[-1][1]

    def broken():
        raise UpdateError("Could not reach GitHub (offline).")

    checker = update_dialog.UpdateChecker(settings, fetch=broken)
    _run(checker, manual=False)
    assert len(messages) == 1
    _run(checker, manual=True)
    assert messages[-1] == ("warning", "Could not check for updates.\n\n"
                                       "Could not reach GitHub (offline).")


def test_automatic_check_schedule(checker_env):
    update_dialog, settings, _ = checker_env
    checker = update_dialog.UpdateChecker(settings, fetch=lambda: None)
    assert checker.due()  # never checked
    settings.setValue(update_dialog.LAST_KEY, datetime.now().isoformat())
    assert not checker.due()
    settings.setValue(update_dialog.LAST_KEY, (datetime.now() - timedelta(days=2)).isoformat())
    assert checker.due()
    settings.setValue(update_dialog.AUTO_KEY, False)
    assert not checker.due()

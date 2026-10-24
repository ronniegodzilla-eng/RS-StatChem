"""Check GitHub Releases for a newer RS-StatChem (no Qt here).

The source repository is private, so releases are mirrored to a public repository that the
app can query without credentials. Only the release metadata is read; nothing is downloaded or
installed automatically — the user is sent to the download page.
"""
from __future__ import annotations

import json
import platform
import re
import ssl
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass

from rsstatchem import APP_NAME, __version__

RELEASES_REPO = "ronniegodzilla-eng/RS-StatChem-releases"
LATEST_URL = f"https://api.github.com/repos/{RELEASES_REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{RELEASES_REPO}/releases/latest"
TIMEOUT = 8


class UpdateError(Exception):
    pass


@dataclass
class UpdateInfo:
    version: str
    tag: str
    notes: str
    page_url: str
    download_url: str | None
    published: str

    @property
    def newer(self) -> bool:
        return parse_version(self.version) > parse_version(__version__)


def parse_version(text: str) -> tuple[int, ...]:
    """'v0.4.2' / '0.10.0-rc1' -> (0, 4, 2) / (0, 10, 0); anything unparsable sorts lowest."""
    m = re.match(r"^\s*v?(\d+(?:\.\d+)*)", text or "")
    if not m:
        return (0,)
    parts = [int(p) for p in m.group(1).split(".")]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()  # 0.5 == 0.5.0
    return tuple(parts)


def asset_keyword() -> str:
    """Which release asset belongs to this machine."""
    if sys.platform == "darwin":
        return "macOS-arm64" if platform.machine() == "arm64" else "macOS"
    if sys.platform == "win32":
        return "Windows-x64"
    return "Linux"


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:  # pragma: no cover - certifi ships with the app
        return ssl.create_default_context()


def info_from_release(data: dict) -> UpdateInfo:
    tag = str(data.get("tag_name") or "")
    keyword = asset_keyword().lower()
    download = next((a.get("browser_download_url") for a in data.get("assets") or []
                     if keyword in str(a.get("name", "")).lower()), None)
    notes = str(data.get("body") or "").strip()
    return UpdateInfo(version=tag.lstrip("vV"), tag=tag, notes=notes,
                      page_url=str(data.get("html_url") or RELEASES_PAGE),
                      download_url=download, published=str(data.get("published_at") or ""))


def fetch_latest(url: str = LATEST_URL, timeout: float = TIMEOUT) -> UpdateInfo:
    request = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": f"{APP_NAME}/{__version__}",
    })
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=_ssl_context()) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise UpdateError("No published release was found.") from None
        raise UpdateError(f"GitHub answered HTTP {exc.code}.") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise UpdateError(f"Could not reach GitHub ({reason}).") from None
    except ValueError:
        raise UpdateError("GitHub sent an unreadable answer.") from None
    return info_from_release(data)

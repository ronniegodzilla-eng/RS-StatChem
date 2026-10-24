"""Locations of bundled resources and of the per-user data folder.

On Windows the packaged app runs in *portable* mode: settings and temporary files live in a
``UserData`` folder next to ``RS-StatChem.exe`` so nothing is written to the registry or AppData.
If that folder is not writable (e.g. the app was copied into Program Files) it falls back to
AppData. Set ``RSSTATCHEM_HOME`` to force a location.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from rsstatchem import APP_NAME


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def resource_dir() -> Path:
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS")) / "rsstatchem" / "resources"
    return Path(__file__).resolve().parent / "resources"


def resource(name: str) -> Path:
    return resource_dir() / name


def _writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-test"
        probe.write_text("ok")
        probe.unlink()
        return True
    except OSError:
        return False


def _default_user_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", Path.home())) / APP_NAME
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / APP_NAME


def is_portable() -> bool:
    return is_frozen() and sys.platform == "win32"


def user_data_dir() -> Path:
    override = os.environ.get("RSSTATCHEM_HOME")
    if override:
        path = Path(override)
    elif is_portable() and _writable(Path(sys.executable).parent / "UserData"):
        path = Path(sys.executable).parent / "UserData"
    else:
        path = _default_user_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def settings_file() -> Path:
    return user_data_dir() / "settings.ini"

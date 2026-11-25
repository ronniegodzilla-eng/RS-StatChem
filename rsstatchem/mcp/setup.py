"""The connector's entry in the Claude desktop app's configuration file."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

NAME = "rs-statchem"


def config_path() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/Claude/claude_desktop_config.json"
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home()))
        return base / "Claude" / "claude_desktop_config.json"
    return Path.home() / ".config/Claude/claude_desktop_config.json"


def entry() -> dict:
    """How this copy of RS-StatChem is started as a connector."""
    if getattr(sys, "frozen", False):
        return {"command": sys.executable, "args": ["--mcp"]}
    root = Path(__file__).resolve().parents[2]
    return {"command": sys.executable, "args": [str(root / "run.py"), "--mcp"]}


def config_text() -> str:
    return json.dumps({"mcpServers": {NAME: entry()}}, indent=2)


def _read(path: Path) -> dict:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("mcpServers", {}), dict):
        raise ValueError("The configuration file of the Claude app has an unexpected form.")
    return data


def state(path: Path | None = None) -> str:
    """added | other (another command) | missing | broken"""
    try:
        found = _read(path or config_path()).get("mcpServers", {}).get(NAME)
    except (ValueError, OSError):
        return "broken"
    if found is None:
        return "missing"
    mine = entry()
    same = found.get("command") == mine["command"] and found.get("args") == mine["args"]
    return "added" if same else "other"


def _write(path: Path, data: dict) -> Path | None:
    backup = None
    if path.exists():
        backup = path.with_name(f"{path.stem}.backup-{time.strftime('%Y%m%d-%H%M%S')}.json")
        backup.write_bytes(path.read_bytes())
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".rs-tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, path)
    return backup


def add(path: Path | None = None) -> Path | None:
    """Add or update our entry, leaving everything else alone; returns the backup's path."""
    path = path or config_path()
    if not path.parent.is_dir():
        raise ValueError("The Claude desktop app was not found on this computer. Install it and "
                         "open it once, then try again.")
    data = _read(path)
    data.setdefault("mcpServers", {})[NAME] = entry()
    return _write(path, data)


def remove(path: Path | None = None) -> Path | None:
    path = path or config_path()
    data = _read(path)
    if NAME not in data.get("mcpServers", {}):
        return None
    del data["mcpServers"][NAME]
    return _write(path, data)

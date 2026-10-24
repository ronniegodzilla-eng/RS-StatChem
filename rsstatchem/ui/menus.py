"""Merges the feature modules' ``MENUS`` into the Data Editor's menu tree.

Each extension module exports ``MENUS = {"Analyze/Regression": [(text, DialogClass |
callable), None, (submenu, [...])]}``. Entries replace a disabled placeholder with the same
text, or are appended. A module that fails to import is skipped (and reported on stderr) so
one broken feature never takes the whole app down.
"""
from __future__ import annotations

import importlib
import sys
import traceback

EXTENSIONS = (
    "rsstatchem.ui.dialogs.data_ops",
    "rsstatchem.ui.dialogs.logistic",
    "rsstatchem.ui.dialogs.glm",
    "rsstatchem.ui.dialogs.fitting",
    "rsstatchem.ui.dialogs.doe",
    "rsstatchem.ui.graph",
    "rsstatchem.ui.dialogs.sem",
    "rsstatchem.ui.dialogs.pls",
    "rsstatchem.ui.chem_editor",  # before ui.chem so the editors head the Chemistry menu
    "rsstatchem.ui.chem",
)
# Cross-links: an entry of one module that should also appear in another menu.
# (module, attribute, menu path, menu text)
ALIASES = (
    ("rsstatchem.ui.dialogs.fitting", "NonlinearFitDialog", "Analyze/Regression", "Nonlinear..."),
)
# Top-level menus whose placeholder items are dropped once a module provides real entries.
REPLACEABLE = {"Graphs", "RSM", "Chemistry"}
OBSOLETE_PLACEHOLDERS = {"process-style model"}

_loaded: list = []
_failed: dict[str, str] = {}


def _key(text: str) -> str:
    return text.replace("&&", "&").replace("&", "").rstrip(".").strip().lower()


def extension_modules() -> list:
    if not _loaded and not _failed:
        for name in EXTENSIONS:
            try:
                _loaded.append(importlib.import_module(name))
            except ModuleNotFoundError as exc:
                if exc.name != name:
                    _failed[name] = traceback.format_exc()
            except Exception:  # noqa: BLE001 - keep the app usable if one module breaks
                _failed[name] = traceback.format_exc()
        for name, tb in _failed.items():
            print(f"[RS-StatChem] extension {name} failed to load:\n{tb}", file=sys.stderr)
    return _loaded


def failed_extensions() -> dict[str, str]:
    return dict(_failed)


def _merge_list(target: list, entries: list) -> None:
    for entry in entries:
        if entry is None:
            target.append(None)
            continue
        text, payload = entry
        if isinstance(payload, list):
            sub = next((e[1] for e in target if isinstance(e, tuple) and isinstance(e[1], list)
                        and _key(e[0]) == _key(text)), None)
            if sub is None:
                sub = []
                target.append((text, sub))
            _merge_list(sub, payload)
            continue
        for i, existing in enumerate(target):
            label = existing if isinstance(existing, str) else (
                existing[0] if isinstance(existing, tuple) else None)
            if label is not None and label not in ("FREQ", "DESC") and \
                    _key(label) == _key(text):
                target[i] = entry
                break
        else:
            target.append(entry)
    target[:] = [e for e in target
                 if not (isinstance(e, str) and _key(e) in OBSOLETE_PLACEHOLDERS)]


def build_tree(base: dict[str, list]) -> dict[str, list]:
    tree = {k: _copy(v) for k, v in base.items()}
    replaced: set[str] = set()
    for module in extension_modules():
        for path, entries in getattr(module, "MENUS", {}).items():
            parts = path.split("/")
            top = parts[0]
            target = tree.setdefault(top, [])
            if top in REPLACEABLE and top not in replaced:
                target.clear()  # the built-in lists there are placeholders only
                replaced.add(top)
            for title in parts[1:]:
                sub = next((e[1] for e in target if isinstance(e, tuple)
                            and isinstance(e[1], list) and _key(e[0]) == _key(title)), None)
                if sub is None:
                    sub = []
                    target.append((title, sub))
                target = sub
            _merge_list(target, entries)
    loaded = {m.__name__: m for m in extension_modules()}
    for module, attr, path, text in ALIASES:
        target = getattr(loaded.get(module), attr, None)
        if target is None:
            continue
        node = tree.setdefault(path.split("/")[0], [])
        for title in path.split("/")[1:]:
            node = next((e[1] for e in node if isinstance(e, tuple) and isinstance(e[1], list)
                         and _key(e[0]) == _key(title)), node)
        _merge_list(node, [(text, target)])
    for entries in tree.values():
        _tidy(entries)
    return tree


def _tidy(entries: list) -> None:
    """Drop repeated and trailing separators left behind by replaced placeholders."""
    out = []
    for e in entries:
        if e is None and (not out or out[-1] is None) and out:
            continue
        if isinstance(e, tuple) and isinstance(e[1], list):
            _tidy(e[1])
        out.append(e)
    while out and out[-1] is None:
        out.pop()
    entries[:] = out


def _copy(entries: list) -> list:
    out = []
    for e in entries:
        if isinstance(e, tuple) and isinstance(e[1], list):
            out.append((e[0], _copy(e[1])))
        else:
            out.append(e)
    return out


def selftests() -> list[tuple[str, str | None]]:
    """Run every extension's ``selftest()``; returns (module, error or None)."""
    results = []
    for module in extension_modules():
        fn = getattr(module, "selftest", None)
        if fn is None:
            continue
        try:
            fn()
            results.append((module.__name__, None))
        except Exception:  # noqa: BLE001
            results.append((module.__name__, traceback.format_exc()))
    results += [(name, tb) for name, tb in _failed.items()]
    return results

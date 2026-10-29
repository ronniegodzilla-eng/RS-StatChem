"""The feature reference: every menu command of every module, read from the running app.

The command list comes from the live menus, so a new feature appears in the guide by itself;
its explanation comes from ``docs/user_guide/features/<module>.tsv``::

    # comment
    File > New > Data<TAB>Membuat dataset kosong baru.
    File > Open Samples > *<TAB>Membuka contoh citra (satu baris untuk seluruh submenu).

A ``*`` line describes a whole submenu in one row (for lists such as samples, recent files or
open windows). Path parts are separated by `` > `` with spaces, so a label may itself contain
``>``. A key may start with a state label (``Output Viewer: File > Save``) when the same path
means something else in that state. :func:`missing` lists commands that have no explanation
yet; the test suite requires there are none.

Some commands only exist in another window or mode of a module (the Output Viewer, a graph
window being active...). ``docs/user_guide/menu_states.py`` registers those states with
:func:`menu_state`; their extra commands get their own sections in the reference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent.parent
FEATURES_DIR = HERE / "features"


@dataclass
class Command:
    path: tuple[str, ...]          # ("File", "New", "Data")
    shortcut: str = ""
    tip: str = ""                  # the app's own (English) status tip

    @property
    def key(self) -> str:
        return " > ".join(self.path)


@dataclass
class ModuleMenus:
    module: str
    title: str
    menus: list[str] = field(default_factory=list)       # top-level menu titles, in order
    commands: list[Command] = field(default_factory=list)


def _clean(text: str) -> str:
    text = text.split("\t")[0].replace("&&", "\0").replace("&", "").replace("\0", "&")
    return text.rstrip(".").replace("…", "").strip()


def _walk(menu, path, out):
    from PySide6.QtGui import QKeySequence

    for act in menu.actions():
        if act.isSeparator() or not act.isVisible():
            continue
        text = _clean(act.text())
        if not text:
            continue
        if act.menu() is not None:
            # populate menus that are filled on demand
            act.menu().aboutToShow.emit()
            _walk(act.menu(), path + (text,), out)
        else:
            # "Ctrl+S" on every platform (the guide says Ctrl means Cmd on macOS)
            sc = act.shortcut().toString(QKeySequence.SequenceFormat.PortableText)
            tip = act.statusTip() or (act.toolTip() if _clean(act.toolTip()) != text else "")
            out.append(Command(path + (text,), sc, tip))


def read_menus(window, module: str, title: str) -> ModuleMenus:
    """All commands in the menu bar of a module's main window."""
    mm = ModuleMenus(module, title)
    for top in window.menuBar().actions():
        if top.menu() is None or not top.isVisible():
            continue
        name = _clean(top.text())
        mm.menus.append(name)
        top.menu().aboutToShow.emit()
        _walk(top.menu(), (name,), mm.commands)
    # identical paths (the same command reachable twice) appear once
    seen, unique = set(), []
    for c in mm.commands:
        if c.key not in seen:
            seen.add(c.key)
            unique.append(c)
    mm.commands = unique
    return mm


# ---------------------------------------------------------------------------------- texts
def load_texts(module: str) -> dict[str, str]:
    path = FEATURES_DIR / f"{module}.tsv"
    texts: dict[str, str] = {}
    if not path.exists():
        return texts
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if "\t" not in line:
            raise ValueError(f"{path.name}:{n}: expected 'Menu > Path<TAB>explanation'")
        key, text = line.split("\t", 1)
        label, sep, rest = key.partition(": ")
        if sep and " > " in rest and " > " not in label:
            key = f"{label.strip()}: " + _norm(rest)
        else:
            key = _norm(key)
        texts[key] = text.strip()
    return texts


def _norm(key: str) -> str:
    return " > ".join(p.strip() for p in key.split(" > "))


def describe(cmd: Command, texts: dict[str, str], label: str | None = None
             ) -> tuple[str, str | None]:
    """(explanation, wildcard group key or None) for a command."""
    prefixes = ([f"{label}: "] if label else []) + [""]
    for pre in prefixes:
        if pre + cmd.key in texts:
            return texts[pre + cmd.key], None
    for k in range(len(cmd.path) - 1, 0, -1):
        group = " > ".join(cmd.path[:k]) + " > *"
        for pre in prefixes:
            if pre + group in texts:
                return texts[pre + group], pre + group
    return "", None


# ---------------------------------------------------------------------------------- states
MENU_STATES: dict[str, list[tuple[str, Callable]]] = {}


def menu_state(module: str, label: str):
    """Register a function ``fn(window) -> window`` that puts a module into another state (or
    opens another of its windows) whose menus hold more commands; ``label`` names that state
    in the reference, e.g. "Output Viewer" or "Graph window active"."""
    def deco(fn):
        MENU_STATES.setdefault(module, []).append((label, fn))
        return fn
    return deco


@dataclass
class Section:
    label: str | None             # None: the module's main window as it opens
    menus: ModuleMenus


def read_all(window, module: str, title: str) -> list[Section]:
    """The main window's commands, then each registered state's commands not seen before."""
    sections = [Section(None, read_menus(window, module, title))]
    seen = {c.key for c in sections[0].menus.commands}
    for label, fn in MENU_STATES.get(module, []):
        other = fn(window)
        mm = read_menus(other, module, title)
        mm.commands = [c for c in mm.commands if c.key not in seen]
        seen |= {c.key for c in mm.commands}
        if mm.commands:
            sections.append(Section(label, mm))
    return sections


def missing(sections: list[Section]) -> list[str]:
    texts = load_texts(sections[0].menus.module)
    return [(f"{s.label}: " if s.label else "") + c.key for s in sections
            for c in s.menus.commands if not describe(c, texts, s.label)[0]]


def unused(sections: list[Section]) -> list[str]:
    """Explanations whose command no longer exists (renamed or removed)."""
    texts = load_texts(sections[0].menus.module)
    used = set()
    for s in sections:
        for c in s.menus.commands:
            _, group = describe(c, texts, s.label)
            used.add(group)
            used.add(c.key)
            if s.label:
                used.add(f"{s.label}: {c.key}")
    # a "*" row explains a list that may be empty right now (open windows, recorded macros)
    return [k for k in texts if k not in used and not k.endswith(" > *")]


@dataclass
class Row:
    menu: str                     # top-level menu
    label: str                    # "New ▸ Data"
    text: str
    shortcut: str = ""
    children: list[str] = field(default_factory=list)   # for a wildcard group


def rows(mm: ModuleMenus, label: str | None = None) -> list[Row]:
    """Reference rows in menu order; a wildcard submenu becomes one row listing its items."""
    texts = load_texts(mm.module)
    out: list[Row] = []
    groups: dict[str, Row] = {}
    for c in mm.commands:
        text, group = describe(c, texts, label)
        if group:
            if group not in groups:
                parts = group.split(": ", 1)[-1][:-4].split(" > ")
                row = Row(parts[0], " \u25b8 ".join(parts[1:]) or "(daftar)", text)
                groups[group] = row
                out.append(row)
            depth = len(group.split(": ", 1)[-1][:-4].split(" > "))
            groups[group].children.append(" \u25b8 ".join(c.path[depth:]))
            continue
        out.append(Row(c.path[0], " \u25b8 ".join(c.path[1:]), text or c.tip, c.shortcut))
    return out

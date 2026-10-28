"""Builds the ImageJ bar's QMenus from :data:`rsstatchem.core.imaging.menu_layout.MENU_LAYOUT`.

Every leaf becomes a QAction (text = command label, shortcut from the layout) that calls
``trigger(label)``. Actions of labels without an available command in the registry are disabled,
so the work packages' commands appear as soon as their ``commands_<area>.register`` adds them;
call :func:`update_enabled` after registering more commands.
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QMenu, QMenuBar

from rsstatchem.core.imaging.menu_layout import MENU_LAYOUT, Menu, qt_shortcut


def menu_text(label: str) -> str:
    return label.replace("&", "&&")


def build_menus(menubar: QMenuBar, registry, trigger: Callable[[str], None]
                ) -> tuple[dict[str, QAction], dict[str, QMenu]]:
    """Fill ``menubar``; returns (actions by command label, menus by "File>Import" path)."""
    actions: dict[str, QAction] = {}
    menus: dict[str, QMenu] = {}

    def fill(qmenu: QMenu, items, path: str) -> None:
        for node in items:
            if node is None:
                qmenu.addSeparator()
            elif isinstance(node, Menu):
                sub = qmenu.addMenu(menu_text(node.title))
                sub_path = f"{path}>{node.title}"
                menus[sub_path] = sub
                fill(sub, node.items, sub_path)
            else:
                act = QAction(menu_text(node.label), qmenu)
                sc = qt_shortcut(node)
                if sc:
                    act.setShortcut(QKeySequence(sc))
                act.setData(node.label)
                act.triggered.connect(lambda _=False, lb=node.label: trigger(lb))
                qmenu.addAction(act)
                actions[node.label] = act

    for top in MENU_LAYOUT:
        qmenu = menubar.addMenu(menu_text(top.title))
        menus[top.title] = qmenu
        fill(qmenu, top.items, top.title)
    update_enabled(actions, registry)
    return actions, menus


def update_enabled(actions: dict[str, QAction], registry) -> None:
    for label, act in actions.items():
        cmd = registry.get(label)
        act.setEnabled(cmd is not None and cmd.available)

"""ChemDraw's *Main Toolbar*: a vertical, two-column tool palette on the left of the document.

Buttons with a small corner triangle open a fly-out palette (press and hold, or right-click),
like ChemDraw's bond, arrow, bracket and chemical-symbol palettes; picking a tool from a
fly-out puts it on the button. The Templates button opens the template library menu.
"""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (QButtonGroup, QGridLayout, QHBoxLayout, QMenu, QToolBar,
                               QToolButton, QWidget, QWidgetAction)

from rsstatchem.core.chem_draw import templates
from rsstatchem.ui.chemdraw.icons import cd_icon

__all__ = ["MainToolbar", "PALETTE", "TOOL_TIPS"]

# Each slot: list of (tool id, tooltip); the first entry is the default, more = fly-out.
PALETTE: list[list[tuple[str, str]]] = [
    [("lasso", "Lasso: select by drawing around objects")],
    [("select", "Marquee: select with a rectangle; drag selected objects to move them")],
    [("rotate", "Rotate: drag to rotate the selection (or the structure under the pointer)")],
    [("eraser", "Eraser: click or drag over objects to delete them")],
    [("bond:single", "Solid Bond: click to add a bond, click a bond to cycle "
                     "single / double / triple, drag to draw at fixed angles")],
    [("text", "Text: click an atom to type a label (OH, Ph, CO2Me...), click elsewhere "
              "for a caption")],
    [("bond:double", "Double Bond"), ("bond:triple", "Triple Bond"),
     ("bond:aromatic", "Aromatic (dashed double) Bond")],
    [("arrow:forward", "Reaction Arrow"), ("arrow:equilibrium", "Equilibrium Arrow"),
     ("arrow:retro", "Retrosynthetic Arrow"), ("arrow:resonance", "Resonance Arrow"),
     ("arrow:curved", "Curved Arrow (electron pair)"),
     ("arrow:fishhook", "Fishhook Arrow (single electron)"), ("arrow:line", "Line")],
    [("bond:dashed", "Dashed Bond")],
    [("plus", "Plus sign (reaction)")],
    [("bond:hash", "Hashed Wedged Bond (stereo, down)")],
    [("bracket:square", "Square Brackets (drag, or click with a selection)"),
     ("bracket:round", "Parentheses"), ("bracket:curly", "Braces")],
    [("bond:bold", "Bold Bond")],
    [("charge+", "Positive Charge"), ("charge-", "Negative Charge"),
     ("radical", "Radical"), ("map", "Atom-to-Atom Map Number")],
    [("bond:wedge", "Wedged Bond (stereo, up)")],
    [("periodic", "Periodic Table...")],
    [("bond:wavy", "Wavy Bond")],
    [("template", "Templates: choose a template, then click in the document")],
    [("chain", "Acyclic Chain: drag to draw a zig-zag chain")],
    [("atom:C", "Atom: click an atom to make it this element"), ("atom:N", "Nitrogen"),
     ("atom:O", "Oxygen"), ("atom:S", "Sulfur"), ("atom:P", "Phosphorus"),
     ("atom:F", "Fluorine"), ("atom:Cl", "Chlorine"), ("atom:Br", "Bromine"),
     ("atom:I", "Iodine"), ("atom:H", "Hydrogen")],
    [("ring:cyclopropane", "Cyclopropane Ring")],
    [("ring:cyclobutane", "Cyclobutane Ring")],
    [("ring:cyclopentane", "Cyclopentane Ring")],
    [("ring:cyclohexane", "Cyclohexane Ring")],
    [("ring:cycloheptane", "Cycloheptane Ring")],
    [("ring:cyclooctane", "Cyclooctane Ring")],
    [("ring:benzene", "Benzene Ring")],
    [("ring:cyclopentadiene", "Cyclopentadiene Ring")],
    [("ring:chair1", "Cyclohexane Chair 1")],
    [("ring:chair2", "Cyclohexane Chair 2")],
]
TOOL_TIPS = {tid: tip for slot in PALETTE for tid, tip in slot}
_ACTIONS = {"periodic"}                 # buttons that run a command instead of a tool


class MainToolbar(QToolBar):
    """The palette; emits ``tool_selected(tool id)`` and ``command(name)``."""

    tool_selected = Signal(str)
    command = Signal(str)
    template_chosen = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__("Main Toolbar", parent)
        self.setObjectName("chemdraw_main_toolbar")
        self.setOrientation(Qt.Orientation.Vertical)
        self.setMovable(True)
        self.setFloatable(True)
        self.setAllowedAreas(Qt.ToolBarArea.LeftToolBarArea | Qt.ToolBarArea.RightToolBarArea)
        w = QWidget()
        w.setObjectName("chemdraw_palette")
        grid = QGridLayout(w)
        grid.setContentsMargins(3, 3, 3, 3)
        grid.setHorizontalSpacing(1)
        grid.setVerticalSpacing(1)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons: list[QToolButton] = []
        self.current: list[str] = []          # tool shown on each slot's button
        for k, slot in enumerate(PALETTE):
            b = QToolButton()
            b.setObjectName("chemdraw_tool")
            b.setIconSize(QSize(24, 24))
            b.setFixedSize(QSize(32, 30))
            b.setAutoRaise(True)
            tid, tip = slot[0]
            b.setIcon(cd_icon(tid, flyout=len(slot) > 1 or tid == "template"))
            b.setToolTip(tip)
            b.setProperty("tool", tid)
            if tid in _ACTIONS:
                b.clicked.connect(lambda _=False, t=tid: self.command.emit(t))
            else:
                b.setCheckable(True)
                self.group.addButton(b)
                b.clicked.connect(lambda _=False, i=k: self._clicked(i))
            if len(slot) > 1:
                b.setMenu(self._flyout(k, slot))
                b.setPopupMode(QToolButton.ToolButtonPopupMode.DelayedPopup)
            elif tid == "template":
                b.setMenu(self._template_menu())
                b.setPopupMode(QToolButton.ToolButtonPopupMode.DelayedPopup)
            b.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            b.customContextMenuRequested.connect(
                lambda _p, btn=b: btn.showMenu() if btn.menu() is not None else None)
            grid.addWidget(b, k // 2, k % 2)
            self.buttons.append(b)
            self.current.append(tid)
        grid.setRowStretch(len(PALETTE) // 2 + 1, 1)
        self.addWidget(w)

    def _flyout(self, k: int, slot) -> QMenu:
        """A row of tool icons (ChemDraw's fly-out palettes are icon strips)."""
        menu = QMenu(self)
        menu.setObjectName("chemdraw_flyout")
        strip = QWidget()
        strip.setObjectName("chemdraw_flyout_strip")
        row = QHBoxLayout(strip)
        row.setContentsMargins(3, 3, 3, 3)
        row.setSpacing(1)
        for tid, tip in slot:
            b = QToolButton()
            b.setObjectName("chemdraw_tool")
            b.setIcon(cd_icon(tid))
            b.setIconSize(QSize(24, 24))
            b.setFixedSize(QSize(32, 30))
            b.setAutoRaise(True)
            b.setToolTip(tip)
            b.setProperty("tool", tid)
            b.clicked.connect(lambda _=False, i=k, t=tid, m=menu: (m.close(), self.pick(i, t)))
            row.addWidget(b)
        wa = QWidgetAction(menu)
        wa.setDefaultWidget(strip)
        menu.addAction(wa)
        return menu

    def _template_menu(self) -> QMenu:
        menu = QMenu("Templates", self)
        menu.setObjectName("chemdraw_flyout")
        for cat in templates.categories():
            sub = menu.addMenu(cat)
            for name, smi in templates.TEMPLATES[cat]:
                sub.addAction(name, lambda n=name, s=smi: self.template_chosen.emit(n, s))
        return menu

    def _clicked(self, k: int) -> None:
        self.tool_selected.emit(self.current[k])

    def pick(self, k: int, tid: str) -> None:
        """Put ``tid`` on slot ``k`` (from its fly-out) and select it."""
        self.current[k] = tid
        b = self.buttons[k]
        b.setIcon(cd_icon(tid, flyout=True))
        b.setToolTip(TOOL_TIPS.get(tid, tid))
        b.setProperty("tool", tid)
        b.setChecked(True)
        self.tool_selected.emit(tid)

    def slot_of(self, tid: str) -> int | None:
        for k, slot in enumerate(PALETTE):
            if any(t == tid for t, _ in slot):
                return k
        return None

    def show_tool(self, tid: str) -> None:
        """Check the button of ``tid`` (switching its fly-out slot) without emitting."""
        k = self.slot_of(tid)
        if k is None:
            checked = self.group.checkedButton()
            if checked is not None:
                self.group.setExclusive(False)
                checked.setChecked(False)
                self.group.setExclusive(True)
            return
        if self.current[k] != tid:
            self.current[k] = tid
            self.buttons[k].setIcon(cd_icon(tid, flyout=True))
            self.buttons[k].setToolTip(TOOL_TIPS.get(tid, tid))
            self.buttons[k].setProperty("tool", tid)
        if self.buttons[k].isCheckable():
            self.buttons[k].setChecked(True)

    def button_for(self, tid: str) -> QToolButton | None:
        k = self.slot_of(tid)
        return None if k is None else self.buttons[k]

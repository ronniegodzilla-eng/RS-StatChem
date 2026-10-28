"""Menu commands of the drawing objects (work package B), added to a
:class:`~rsstatchem.ui.chemdraw.window.ChemDrawWindow` by :func:`install`.

``install(window)`` is called once by the window (after its menus exist) and adds:

- Object > Show Stereochemistry (R/S, r/s, M/P, E/Z indicators; Object > Stereo Indicator
  Options: parentheses, hide / show the indicator of the selected atoms);
- Structure > Autonumber Reaction, Clear Reaction Numbers, Reaction Numbering Style
  (Arabic, Roman, alphabetic), Bond Properties..., Bracket Properties...;
- Edit > Duplicate Across Arrow (Ctrl+arrow keys, ChemDraw 17.1 p. 60);
- the ChemDraw view keys F5 Actual Size, F6 Fit to Window, F7 Magnify, F8 Reduce (17.1
  p. 62). Check Structure gives up F5, as in ChemDraw, where it has no shortcut;
- Help > Shortcuts and Hotkeys shows :data:`HOTKEYS_HTML` (the 17.1 hotkey tables).

Installing twice is harmless (the second call does nothing).
"""
from __future__ import annotations

from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import QDialog, QTextBrowser, QVBoxLayout

from rsstatchem.core.chem_draw import objects as obj
from rsstatchem.core.chem_draw import ops

__all__ = ["install", "HOTKEYS_HTML", "show_hotkeys", "set_show_stereo", "autonumber",
           "clear_numbers", "F_KEYS"]

F_KEYS = {"act_actual": "F5", "act_fit": "F6", "act_zoom_in": "F7", "act_zoom_out": "F8"}

HOTKEYS_HTML = """
<h3>Hotkeys (ChemDraw 17.1 "enhanced hotkeys")</h3>
<p>Point at an atom or a bond and type; with the pointer over nothing the keys switch
tools. Upper-case letters mean Shift+key.</p>
<table cellpadding='3'>
<tr><td colspan='2'><b>Atom labels</b></td></tr>
<tr><td>c n/w o/q s p f i h k</td><td>C, N, O, S, P, F, I, H, K</td></tr>
<tr><td>l or C / b / B</td><td>Cl / Br / B</td></tr>
<tr><td>S / N / d</td><td>Si / Na / D</td></tr>
<tr><td>m / e / P / A / E</td><td>Me / Et / Ph / Ac / COOCH<sub>3</sub></td></tr>
<tr><td>r / x</td><td>R / X (generic labels)</td></tr>
<tr><td>+ / -</td><td>positive / negative charge</td></tr>
<tr><td>. / '</td><td>attachment point / atom number</td></tr>
<tr><td>Enter / =</td><td>type a label / choose a nickname</td></tr>
<tr><td>Space, Backspace, Delete</td><td>remove the label</td></tr>
<tr><td>/ or ?</td><td>Atom Properties</td></tr>
<tr><td colspan='2'><b>Sprouting on an atom</b></td></tr>
<tr><td>1 / 0</td><td>bond in linear / cyclic mode</td></tr>
<tr><td>2</td><td>carbonyl (secondary C) or acetyl</td></tr>
<tr><td>3 or a</td><td>phenyl</td></tr>
<tr><td>4 / 5</td><td>wedged / hashed wedged bond (+ methyl)</td></tr>
<tr><td>6 / 7</td><td>cyclohexyl / cyclopentyl (spiro on a secondary C)</td></tr>
<tr><td>8 / 9</td><td>methylidene / dimethyl (gem-dimethyl, isopropyl)</td></tr>
<tr><td>z</td><td>alkyne</td></tr>
<tr><td colspan='2'><b>Bonds</b></td></tr>
<tr><td>1 / 2 / 3</td><td>single / double / triple</td></tr>
<tr><td>d / y / b / B</td><td>dashed / wavy / bold / bold double</td></tr>
<tr><td>w / h / H</td><td>wedged / hashed wedged / hashed</td></tr>
<tr><td>l / c / r</td><td>double bond left / centred / right</td></tr>
<tr><td>a / z / v</td><td>fuse benzene / cyclopentadiene / cyclopropane</td></tr>
<tr><td>4 ... 8</td><td>fuse a ring of that size</td></tr>
<tr><td>9 / 0</td><td>fuse a cyclohexane chair (two orientations)</td></tr>
<tr><td>f / (/ or ?)</td><td>bring to front / Bond Properties</td></tr>
<tr><td>o / D / A</td><td>hollow wedge / dative / aromatic (RS-StatChem additions)</td></tr>
<tr><td colspan='2'><b>Tools (pointer over nothing)</b></td></tr>
<tr><td>Space / x / X</td><td>Marquee / Solid Bond / Acyclic Chain</td></tr>
<tr><td>t / T</td><td>Text / Brackets</td></tr>
<tr><td>e / E</td><td>Arrows / Chemical Symbols</td></tr>
<tr><td>j / J / u / U</td><td>benzene / cyclopentadiene / cyclohexane / cyclopentane</td></tr>
<tr><td>g / G</td><td>Chromatography (TLC, gel) / Orbitals</td></tr>
<tr><td>l / L / k</td><td>Pen / Drawing Elements / Mass Fragmentation</td></tr>
<tr><td colspan='2'><b>Anywhere</b></td></tr>
<tr><td>Ctrl+arrow keys</td><td>add a reaction arrow and a copy of the selection beyond it</td></tr>
<tr><td>Arrow keys</td><td>nudge the selection (Shift: 10 pt)</td></tr>
<tr><td>F5 / F6 / F7 / F8</td><td>actual size / fit to window / magnify / reduce</td></tr>
<tr><td>Shift while dragging</td><td>free angle (boxes: square)</td></tr>
<tr><td colspan='2'><b>TLC and gel plates</b></td></tr>
<tr><td>click a lane</td><td>add a spot / band (TLC or gel tool)</td></tr>
<tr><td>drag a spot / band</td><td>set Rf / size (Shift+drag a spot: size and smear)</td></tr>
<tr><td>double-click a spot / band</td><td>type the Rf / size</td></tr>
<tr><td colspan='2'><b>Pen tool</b></td></tr>
<tr><td>click / drag</td><td>corner point / smooth point</td></tr>
<tr><td>click the first point</td><td>close the curve; double-click, Enter, Esc finish</td></tr>
</table>
"""


def _menu(window, title: str):
    for act in window.menuBar().actions():
        m = act.menu()
        if m is not None and act.text().replace("&", "") == title:
            return m
    return None


def _has_action(window, text: str) -> bool:
    for act in window.menuBar().actions():
        m = act.menu()
        if m is None:
            continue
        for a in m.actions():
            if a.text().replace("&", "") == text:
                return True
    return False


def set_show_stereo(window, on: bool) -> None:
    """Object > Show Stereochemistry for the whole document."""
    cv = window.canvas

    def fn():
        cv.doc.style.show_stereo = bool(on)
        if on:
            ops.refresh_stereo(cv.doc)
    cv.edit("Show Stereochemistry", fn)


def _stereo_parens(window, on: bool) -> None:
    window.canvas.edit("Stereo Indicators",
                       lambda: setattr(window.canvas.doc.style, "stereo_parens", bool(on)))


def _stereo_selected(window, show: bool) -> None:
    cv = window.canvas
    atoms = cv.selected_atoms() or set(cv.doc.atoms)

    def fn():
        for a in atoms:
            cv.doc.atoms[a].show_stereo = show
    cv.edit("Stereo Indicators", fn)


def autonumber(window) -> list[int]:
    """Structure > Autonumber Reaction (selection, or the whole document)."""
    cv = window.canvas
    atoms = cv.selected_atoms() or None
    return cv.edit("Autonumber Reaction", lambda: obj.autonumber(cv.doc, atoms)) or []


def clear_numbers(window) -> int:
    cv = window.canvas
    atoms = cv.selected_atoms() or None
    return cv.edit("Clear Reaction Numbers",
                   lambda: obj.clear_autonumbers(cv.doc, atoms)) or 0


def _numbering(window, style: str) -> None:
    window.canvas.edit("Reaction Display",
                       lambda: setattr(window.canvas.doc.style, "autonumber_style", style))


def _selected(window, kind: str) -> int | None:
    ids = [i for k, i in window.canvas.selection if k == kind]
    hover = window.canvas.hover
    if not ids and hover is not None and hover[0] == kind:
        ids = [hover[1]]
    return ids[0] if ids else None


def show_hotkeys(window) -> QDialog:
    dlg = QDialog(window)
    dlg.setWindowTitle("Shortcuts and Hotkeys")
    view = QTextBrowser()
    view.setHtml(HOTKEYS_HTML)
    lay = QVBoxLayout(dlg)
    lay.addWidget(view)
    dlg.resize(520, 640)
    dlg.show()
    return dlg


def _function_keys(window) -> None:
    """ChemDraw's view keys; Check Structure has no shortcut in ChemDraw."""
    check = getattr(window, "act_check", None)
    if isinstance(check, QAction) and check.shortcut() == QKeySequence("F5"):
        check.setShortcut(QKeySequence())
    for name, key in F_KEYS.items():
        act = getattr(window, name, None)
        if not isinstance(act, QAction):
            continue
        seqs = [QKeySequence(key)] + [s for s in act.shortcuts()
                                      if s != QKeySequence(key)]
        act.setShortcuts(seqs)


def install(window) -> None:
    """Add the WP-B commands to ``window``'s menus (idempotent)."""
    if getattr(window, "_objects_actions_installed", False):
        return
    window._objects_actions_installed = True
    window.hotkeys_html = HOTKEYS_HTML
    _function_keys(window)
    cv = window.canvas
    m = _menu(window, "Object")
    if m is not None:
        m.addSeparator()
        if not _has_action(window, "Show Stereochemistry"):
            act = m.addAction("Show Stereochemistry")
            act.setObjectName("act_show_stereo")
            act.setCheckable(True)
            act.setChecked(cv.doc.style.show_stereo)
            act.triggered.connect(lambda on: set_show_stereo(window, on))
            m.aboutToShow.connect(lambda: act.setChecked(cv.doc.style.show_stereo))
            window.act_show_stereo = act
        sub = m.addMenu("Stereo Indicator Options")
        par = sub.addAction("Include Parentheses")
        par.setCheckable(True)
        par.setChecked(cv.doc.style.stereo_parens)
        par.triggered.connect(lambda on: _stereo_parens(window, on))
        sub.aboutToShow.connect(lambda: par.setChecked(cv.doc.style.stereo_parens))
        sub.addAction("Hide Indicators of Selected Atoms",
                      lambda: _stereo_selected(window, False))
        sub.addAction("Show Indicators of Selected Atoms",
                      lambda: _stereo_selected(window, True))
    m = _menu(window, "Structure")
    if m is not None:
        m.addSeparator()
        window.act_autonumber = m.addAction("Autonumber Reaction", lambda: autonumber(window))
        m.addAction("Clear Reaction Numbers", lambda: clear_numbers(window))
        sub = m.addMenu("Reaction Numbering Style")
        group = QActionGroup(window)
        for text, style in (("Arabic (1, 2, 3)", "arabic"), ("Roman (I, II, III)", "roman"),
                            ("Alphabetic (a, b, c)", "alpha")):
            act = sub.addAction(text, lambda s=style: _numbering(window, s))
            act.setCheckable(True)
            act.setChecked(cv.doc.style.autonumber_style == style)
            group.addAction(act)
        m.addSeparator()

        def bond_props():
            bid = _selected(window, "bond")
            if bid is not None:
                cv.bond_properties(bid)

        def bracket_props():
            bid = _selected(window, "bracket")
            if bid is not None:
                cv.bracket_properties(bid)
        m.addAction("&Bond Properties...", bond_props)
        m.addAction("Brac&ket Properties...", bracket_props)
    m = _menu(window, "Edit")
    if m is not None:
        sub = m.addMenu("Duplicate Across Arrow")
        for text, d in (("Right (Ctrl+Right)", "right"), ("Left (Ctrl+Left)", "left"),
                        ("Down (Ctrl+Down)", "down"), ("Up (Ctrl+Up)", "up")):
            sub.addAction(text, lambda d=d: cv.duplicate_across_arrow(d)
                          if cv.selection else None)
    m = _menu(window, "Help")
    if m is not None:
        for act in m.actions():
            if act.text().replace("&", "") == "Shortcuts and Hotkeys":
                try:
                    act.triggered.disconnect()
                except (RuntimeError, TypeError):
                    pass
                act.triggered.connect(lambda: show_hotkeys(window))

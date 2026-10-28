"""Chemistry-intelligence menu items of the Chemical Drawing window (WP-A).

``install(window)`` is called once by :class:`ChemDrawWindow` (WP-C owns ``window.py``) after
its menus exist. It adds, to the existing menus found by title:

- Structure: *Predict ¹H-NMR Shifts*, *Predict ¹³C-NMR Shifts* (the NMR window), *Show
  Stereochemistry* (R/S, r/s, E/Z labels; drawn by the renderer when the document style has
  ``show_stereo`` (WP-B), otherwise listed in the status bar), and upgrades *Convert Structure
  to Name* so that the verified rule-based IUPAC name (:mod:`iupac_namer`, OPSIN round trip)
  is tried after the offline list and before the offline database / PubChem;
- File: *Open Spectrum (JCAMP-DX)...* (the NMR window with the measured spectrum).

Every step checks that what it touches exists, so the module also works against an older
window. No other module edits are needed.
"""
from __future__ import annotations

from PySide6.QtGui import QAction, QGuiApplication
from PySide6.QtCore import Qt

from rsstatchem.core.chem import molecule as mo
from rsstatchem.core.chem_draw import convert, names, stereo

__all__ = ["install", "show_nmr", "open_spectrum", "toggle_stereo", "stereo_summary",
           "convert_structure_to_name"]


def _menu(window, title: str):
    for act in window.menuBar().actions():
        m = act.menu()
        if m is not None and act.text().replace("&", "").lower() == title.lower():
            return m
    return None


def install(window) -> dict:
    """Add the WP-A actions to the window's menus; returns {name: QAction}."""
    acts: dict = {}
    structure = _menu(window, "Structure")
    if structure is not None:
        structure.addSeparator()
        acts["nmr_1h"] = structure.addAction("Predict ¹H-NMR Shifts",
                                             lambda: show_nmr(window, "1H"))
        acts["nmr_13c"] = structure.addAction("Predict ¹³C-NMR Shifts",
                                              lambda: show_nmr(window, "13C"))
        stereo_act = QAction("Show Stereochemistry", window)
        stereo_act.setCheckable(True)
        style = getattr(getattr(window, "doc", None), "style", None)
        stereo_act.setChecked(bool(getattr(style, "show_stereo", False)))
        stereo_act.setToolTip("R/S, r/s and E/Z (CIP) labels on the structure")
        stereo_act.toggled.connect(lambda on: toggle_stereo(window, on))
        structure.addAction(stereo_act)
        acts["stereo"] = stereo_act
    act = getattr(window, "act_struct2name", None)
    if act is not None:
        try:
            act.triggered.disconnect()
        except (RuntimeError, TypeError):
            pass
        act.triggered.connect(lambda *_a: convert_structure_to_name(window))
        acts["struct2name"] = act
    file_menu = _menu(window, "File")
    if file_menu is not None:
        open_act = QAction("Open Spectrum (JCAMP-DX)...", window)
        open_act.triggered.connect(lambda: open_spectrum(window))
        before = next((a for a in file_menu.actions() if a.menu() is not None and
                       a.text().replace("&", "") == "Open Recent"), None)
        if before is not None:
            idx = file_menu.actions().index(before)
            nxt = file_menu.actions()[idx + 1] if idx + 1 < len(file_menu.actions()) else None
            file_menu.insertAction(nxt, open_act) if nxt is not None else \
                file_menu.addAction(open_act)
        else:
            file_menu.addAction(open_act)
        acts["open_spectrum"] = open_act
    window.intel_actions = acts
    return acts


# --- NMR -----------------------------------------------------------------------------------------

def show_nmr(window, nucleus: str = "1H"):
    """Open (or raise) the NMR window for the window's canvas."""
    from rsstatchem.ui.chemdraw.nmr_window import NmrWindow
    w = getattr(window, "nmr_window", None)
    if w is None:
        w = NmrWindow(window.canvas, nucleus, window)
        window.nmr_window = w
    elif w.nucleus != nucleus:
        w.nucleus_box.setCurrentText(nucleus)
    w.refresh(force=True)
    w.show()
    w.raise_()
    return w


def open_spectrum(window, path: str | None = None):
    w = show_nmr(window, getattr(getattr(window, "nmr_window", None), "nucleus", "1H"))
    w.open_spectrum(path)
    return w


# --- stereo -------------------------------------------------------------------------------------

def stereo_summary(doc) -> str:
    """'C2 (S), C3=C4 (E)' with atom numbers in document order (status-bar fallback)."""
    labs = stereo.labels(doc)
    if not labs:
        return ""
    order = {aid: k + 1 for k, aid in enumerate(sorted(doc.atoms))}
    parts = []
    for (kind, i), lab in sorted(labs.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if kind == "atom":
            parts.append((order[i], f"{doc.atoms[i].symbol}{order[i]} ({lab})"))
        else:
            b = doc.bonds[i]
            a1, a2 = sorted((order[b.a], order[b.b]))
            parts.append((a1, f"{doc.atoms[b.a].symbol}{a1}={doc.atoms[b.b].symbol}{a2} "
                              f"({lab})"))
    return ", ".join(t for _k, t in sorted(parts))


def toggle_stereo(window, on: bool) -> dict:
    """Turn the stereo indicators on/off. Returns the labels (empty when off)."""
    doc = window.doc
    labs = stereo.labels(doc) if on else {}
    if hasattr(doc.style, "show_stereo"):
        doc.style.show_stereo = bool(on)
        try:
            window.canvas.rebuild()
        except Exception:  # noqa: BLE001 - drawing must not break on a toggle
            pass
    if on:
        text = stereo_summary(doc)
        window.statusBar().showMessage(
            ("Stereocentres: " + text) if text else "No defined stereocentres or "
            "stereogenic double bonds", 10000)
    return labs


# --- names --------------------------------------------------------------------------------------

def convert_structure_to_name(window) -> list[str]:
    """Convert Structure to Name with the verified rule-based namer as an extra offline tier:
    offline list -> rule-based name (OPSIN-checked) -> the window's own lookup (offline
    database, then PubChem) for the structures still without a name."""
    frags_fn = getattr(window, "_fragments_in_scope", None)
    caption = getattr(window, "_caption_under", None)
    original = getattr(window, "convert_structure_to_name", None)
    if frags_fn is None or caption is None:
        return original() if original else []
    frags = frags_fn()
    if not frags:
        return original() if original else []
    mols, comps = [], []
    for comp in frags[:12]:
        try:
            mols.append(convert.to_mol(window.doc, comp))
            comps.append(comp)
        except mo.ChemError:
            continue
    found: list[str] = []
    rest: list = []
    QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    try:
        listed = [names.offline_name(m) for m in mols]
        need = [m for m, h in zip(mols, listed) if h is None]
        generated = iter(names.generated_names(need)) if need else iter(())
        for comp, hit in zip(comps, listed):
            if hit is None:
                hit = next(generated)
            if hit is None:
                rest.append(comp)
                continue
            caption(comp, hit.name)
            found.append(hit.name)
            window.statusBar().showMessage(f"Name from the {hit.source}", 6000)
    finally:
        QGuiApplication.restoreOverrideCursor()
    if rest and original is not None:
        canvas = window.canvas
        before = set(canvas.selection)
        canvas.select_keys({("atom", a) for comp in rest for a in comp})
        try:
            found += original()
        finally:
            canvas.select_keys({k for k in before if window.doc.get(k) is not None})
    return found

"""Main window of the Chemical Drawing module, laid out like a ChemDraw Professional document
window: menu bar (File, Edit, View, Object, Structure, Text, Curves, Colors, Window, Help), the
General and Style toolbars on top, the vertical Main Toolbar on the left, the document page in
the middle, the Analysis / Chemical Properties windows on the right and the zoom percentage in
the status bar.

The drawing itself (canvas, tools, undo, file formats, exports) is the structure editor of
:mod:`rsstatchem.ui.chem_editor`; this module is the ChemDraw shell around it.

Menus and shortcuts follow the ChemDraw 17.1 User Guide (menu paths throughout the guide,
shortcut tables pp. 60-64): F5 / F6 / F7 / F8 actual size / fit / magnify / reduce, F11
rulers, F2 / F3 bring to front / send to back, Ctrl+L / Ctrl+E fixed lengths / angles,
Ctrl+J join, Ctrl+K scale, Ctrl+R rotate, Ctrl+Y repeat, Ctrl+D copy as CDXML,
Alt+Ctrl+C / Alt+Ctrl+P copy / paste SMILES, Alt+Ctrl+N structure to name ... Commands the
app cannot back are left out rather than shown disabled; the automatic reaction mapping (needs
the optional Indigo toolkit) is greyed out with an explanation when Indigo is missing.

Other work packages extend the menus without editing this file: if
``rsstatchem.ui.chemdraw.intel_actions`` or ``rsstatchem.ui.chemdraw.objects_actions`` exists,
its ``install(window)`` is called once the menus and windows are built (``window.menus`` maps
menu titles to the menus).
"""
from __future__ import annotations

import importlib
import math
import os
import sys
import traceback

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import (QAction, QActionGroup, QColor, QFont, QKeySequence, QPainter,
                           QPen)
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QDialogButtonBox,
                               QDockWidget, QFileDialog, QFontComboBox, QInputDialog, QLabel,
                               QMessageBox, QTableWidget, QTableWidgetItem, QTextBrowser,
                               QToolBar, QVBoxLayout)

from rsstatchem.core.chem import molecule as mo
from rsstatchem.core.chem_draw import cdx, cdxml, convert, mapping, names
from rsstatchem.core.chem_draw.model import Document, save_rsc
from rsstatchem.core.chem_draw.style import PRESETS, preset
from rsstatchem.ui.chem_editor.panels import TemplatePanel
from rsstatchem.ui.chem_editor.window import ChemEditorWindow
from rsstatchem.ui.chemdraw.canvas import ZOOM_STEPS, PageCanvas
from rsstatchem.ui.chemdraw.docks import AnalysisWindow, ChemicalPropertiesWindow
from rsstatchem.ui.chemdraw.icons import cd_icon, color_icon
from rsstatchem.ui.chemdraw.lookup import NameLookup
from rsstatchem.ui.chemdraw.palette import TOOL_TIPS, MainToolbar
from rsstatchem.ui.chemdraw.style import STYLESHEET

__all__ = ["ChemDrawWindow", "OPEN_FILTER", "DEFAULT_STYLE", "COLORS", "HOTKEYS_HTML",
           "MENU_SHORTCUTS", "ShellCanvas", "INSTALL_MODULES"]

DEFAULT_STYLE = "ACS Document 1996"
OPEN_FILTER = ("Chemical documents (*.rsc *.cdx *.cdxml *.mol *.sdf *.sd *.rxn *.smi *.helm "
               "*.fasta *.fa);;RS-StatChem drawing (*.rsc);;ChemDraw document (*.cdx);;"
               "ChemDraw XML (*.cdxml);;MDL Molfile (*.mol);;SD file (*.sdf *.sd);;"
               "MDL Rxnfile (*.rxn);;SMILES (*.smi *.txt);;HELM / FASTA (*.helm *.fasta *.fa);;"
               "All files (*)")
SAVE_FILTERS = {"RS-StatChem drawing (*.rsc)": ".rsc", "ChemDraw document (*.cdx)": ".cdx",
                "ChemDraw XML (*.cdxml)": ".cdxml"}
_DOC_EXT = (".rsc", ".cdx", ".cdxml")
# modules of other work packages that add their own menu items (see the module docstring)
INSTALL_MODULES = ("rsstatchem.ui.chemdraw.intel_actions",
                   "rsstatchem.ui.chemdraw.objects_actions")
# reaction interpretation colours (ChemDraw 17.1 p. 173: reactants red, products magenta,
# conditions cyan / blue)
INTERPRETATION_COLORS = {"reactants": "#E00000", "products": "#D000A0", "agents": "#00A8C8",
                         "arrow": "#0000E0"}
COLORS = [("Black", "#000000"), ("Red", "#E00000"), ("Orange", "#F07800"),
          ("Yellow", "#E8C800"), ("Green", "#009A00"), ("Cyan", "#00A8C8"),
          ("Blue", "#0000E0"), ("Violet", "#8000C0"), ("Magenta", "#D000A0"),
          ("Brown", "#8B4513"), ("Gray", "#808080")]
FONTS = ["Arial", "Helvetica", "Times New Roman", "Courier New", "Calibri", "Symbol"]
SIZES = [6, 7, 8, 9, 10, 11, 12, 14, 18, 24, 36]
_CHEM_EXT = (".rsc", ".cdx", ".cdxml", ".mol", ".sdf", ".sd", ".rxn", ".smi", ".helm",
             ".fasta", ".fa")

HOTKEYS_HTML = """
<h3>Hotkeys (point at an atom or bond and type)</h3>
<table cellpadding='3'>
<tr><td colspan='2'><b>Atoms</b></td></tr>
<tr><td>c n o s p f i h k</td><td>C, N, O, S, P, F, I, H, K</td></tr>
<tr><td>l / b / B</td><td>Cl / Br / B</td></tr>
<tr><td>+ / -</td><td>increase / decrease the charge</td></tr>
<tr><td>.</td><td>radical</td></tr>
<tr><td>1</td><td>add a bond</td></tr>
<tr><td>3 ... 8</td><td>attach a ring of that size</td></tr>
<tr><td>Enter</td><td>type a label (OH, Ph, CO2Me, Boc...)</td></tr>
<tr><td>/</td><td>Atom Properties</td></tr>
<tr><td colspan='2'><b>Bonds</b></td></tr>
<tr><td>1 / 2 / 3</td><td>single / double / triple</td></tr>
<tr><td>4 ... 8</td><td>fuse a ring of that size</td></tr>
<tr><td>a</td><td>aromatic (dashed double)</td></tr>
<tr><td>w / h / o</td><td>wedged / hashed wedged / hollow wedged</td></tr>
<tr><td>v</td><td>dative (arrow from the first atom)</td></tr>
<tr><td>b / d / y</td><td>bold / dashed / wavy</td></tr>
<tr><td>l / r / c</td><td>double bond left / right / centred</td></tr>
<tr><td colspan='2'><b>Anywhere</b></td></tr>
<tr><td>Delete</td><td>delete the selection or the object under the pointer</td></tr>
<tr><td>Arrow keys</td><td>nudge the selection (Shift: 10 pt)</td></tr>
<tr><td>Shift while dragging</td><td>free angle</td></tr>
<tr><td>Ctrl+wheel, F7 / F8</td><td>magnify / reduce</td></tr>
<tr><td colspan='2'><b>Pen tool</b></td></tr>
<tr><td>click / drag</td><td>corner point / smooth point (drag sets the handles)</td></tr>
<tr><td>click the first point</td><td>close the curve</td></tr>
<tr><td>double-click, Enter, Esc</td><td>finish the curve</td></tr>
<tr><td>drag a point or handle</td><td>edit the selected curve (Shift: break the
smooth point)</td></tr>
</table>
"""

# (menu, command, keys) - the ChemDraw 17.1 shortcut tables (User Guide pp. 60-64) as backed
# by this window; shown under Help > Shortcuts and Hotkeys and checked by the tests
MENU_SHORTCUTS = [
    ("File", "New Document", "Ctrl+N"), ("File", "Open", "Ctrl+O"),
    ("File", "Save", "Ctrl+S"), ("File", "Save As", "Shift+Ctrl+S"),
    ("File", "Page Setup", "Shift+Ctrl+P"), ("File", "Print", "Ctrl+P"),
    ("File", "Close", "Ctrl+W"),
    ("Edit", "Undo", "Ctrl+Z"), ("Edit", "Redo", "Shift+Ctrl+Z"),
    ("Edit", "Repeat", "Ctrl+Y"), ("Edit", "Copy As CDXML Text", "Ctrl+D"),
    ("Edit", "Copy As SMILES", "Alt+Ctrl+C"), ("Edit", "Copy As MOL Text", "Alt+Shift+Ctrl+O"),
    ("Edit", "Paste Special SMILES", "Alt+Ctrl+P"),
    ("Edit", "Paste Special MOL / CDXML", "Alt+Shift+Ctrl+P"),
    ("View", "Actual Size", "F5"), ("View", "Fit to Window", "F6"),
    ("View", "Magnify", "F7"), ("View", "Reduce", "F8"), ("View", "Show Rulers", "F11"),
    ("Object", "Fixed Lengths", "Ctrl+L"), ("Object", "Fixed Angles", "Ctrl+E"),
    ("Object", "Group", "Ctrl+G"), ("Object", "Ungroup", "Shift+Ctrl+G"),
    ("Object", "Join", "Ctrl+J"), ("Object", "Bring to Front", "F2"),
    ("Object", "Send to Back", "F3"), ("Object", "Flip Horizontal", "Shift+Ctrl+H"),
    ("Object", "Flip Vertical", "Shift+Ctrl+V"),
    ("Object", "Rotate 180° Horizontal", "Alt+Shift+Ctrl+H"),
    ("Object", "Rotate 180° Vertical", "Alt+Shift+Ctrl+V"),
    ("Object", "Rotate...", "Ctrl+R"), ("Object", "Scale...", "Ctrl+K"),
    ("Object", "Align Lefts", "Alt+Shift+Ctrl+L"),
    ("Object", "Align Horizontal Centers", "Alt+Shift+Ctrl+C"),
    ("Object", "Align Rights", "Alt+Shift+Ctrl+R"), ("Object", "Align Tops", "Alt+Shift+Ctrl+T"),
    ("Object", "Align Vertical Centers", "Alt+Shift+Ctrl+M"),
    ("Object", "Align Bottoms", "Alt+Shift+Ctrl+B"),
    ("Structure", "Clean Up Structure", "Shift+Ctrl+K"),
    ("Structure", "Convert Name to Structure", "Shift+Ctrl+N"),
    ("Structure", "Convert Structure to Name", "Alt+Ctrl+N"),
    ("Text", "Plain", "Ctrl+T"), ("Text", "Bold", "Ctrl+B"), ("Text", "Italic", "Ctrl+I"),
    ("Text", "Underline", "Ctrl+U"), ("Text", "Formula", "Ctrl+F"),
    ("Text", "Subscript", "F9"), ("Text", "Superscript", "F10"),
]


def _shortcuts_html() -> str:
    rows, last = [], None
    for menu, cmd, keys in MENU_SHORTCUTS:
        if menu != last:
            rows.append(f"<tr><td colspan='2'><b>{menu}</b></td></tr>")
            last = menu
        shown = QKeySequence(keys).toString(QKeySequence.SequenceFormat.NativeText)
        rows.append(f"<tr><td>{shown}</td><td>{cmd}</td></tr>")
    return ("<h3>Menu shortcuts (as in ChemDraw)</h3><table cellpadding='3'>" +
            "".join(rows) + "</table>")


class ShellCanvas(PageCanvas):
    """The page canvas plus the View menu's overlays: rulers (F11), the crosshair and the
    reaction interpretation (reactants red, products magenta, conditions cyan, arrow blue),
    painted over the drawing so the document itself is never changed."""
    show_rulers = False
    show_crosshair = False
    show_interpretation = False
    _cursor_pos = None

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        super().mouseMoveEvent(e)
        if self.show_rulers or self.show_crosshair:
            self._cursor_pos = self.mapToScene(e.position().toPoint())
            self.viewport().update()

    def interpretation_boxes(self) -> list[tuple[str, QRectF]]:
        """[(role, scene rectangle)] of the interpreted reaction (empty when there is none)."""
        if not self.doc.arrows:
            return []
        try:
            split = convert.split_reaction(self.doc)
            comps = convert.reaction_components(self.doc, split)
        except Exception:  # noqa: BLE001 - interpretation is only a display aid
            return []
        if not split.ok:
            return []
        pad = self.bl * 0.35
        out = []
        for role in ("reactants", "agents", "products"):
            for keys in comps[role]:
                bb = self.doc.bbox(keys)
                if bb is None:
                    continue
                r = QRectF(bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1])
                for k, i in keys:                       # include the atom labels
                    box = self.info.label_boxes.get(i) if k == "atom" else None
                    if box is not None:
                        r = r.united(box)
                out.append((role, r.adjusted(-pad, -pad, pad, pad)))
        ar = split.arrow
        out.append(("arrow", QRectF(min(ar.x1, ar.x2) - pad, min(ar.y1, ar.y2) - pad,
                                    abs(ar.x2 - ar.x1) + 2 * pad,
                                    abs(ar.y2 - ar.y1) + 2 * pad)))
        return out

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802
        super().drawForeground(painter, rect)
        z = max(self.zoom(), 1e-6)
        if self.show_interpretation:
            for role, r in self.interpretation_boxes():
                c = QColor(INTERPRETATION_COLORS[role])
                painter.setPen(QPen(c, 1.2 / z, Qt.PenStyle.DashLine))
                c.setAlpha(28)
                painter.setBrush(c)
                painter.drawRoundedRect(r, 3 / z, 3 / z)
            painter.setBrush(Qt.BrushStyle.NoBrush)
        pos = self._cursor_pos
        if self.show_crosshair and pos is not None:
            painter.setPen(QPen(QColor(0, 120, 215, 150), 0, Qt.PenStyle.DashLine))
            page = self.page
            painter.drawLine(QPointF(page.left(), pos.y()), QPointF(page.right(), pos.y()))
            painter.drawLine(QPointF(pos.x(), page.top()), QPointF(pos.x(), page.bottom()))
        if self.show_rulers:
            self._draw_rulers(painter)

    def _draw_rulers(self, painter: QPainter) -> None:
        """Rulers along the top and left edges in centimetres from the page corner, with the
        pointer position and the selection extent marked."""
        painter.save()
        painter.resetTransform()
        vp = self.viewport().rect()
        t = 18
        painter.fillRect(QRectF(0, 0, vp.width(), t), QColor("#F4F4F4"))
        painter.fillRect(QRectF(0, 0, t, vp.height()), QColor("#F4F4F4"))
        painter.setPen(QPen(QColor("#707070"), 1))
        painter.drawLine(0, t, vp.width(), t)
        painter.drawLine(t, 0, t, vp.height())
        origin = self.mapFromScene(self.page.topLeft())
        cm = 72 / 2.54 * self.zoom()                     # one centimetre in pixels
        step = 1 if cm >= 25 else 2 if cm >= 12 else 5
        font = painter.font()
        font.setPointSizeF(7)
        painter.setFont(font)
        for axis in (0, 1):
            length = vp.width() if axis == 0 else vp.height()
            o = origin.x() if axis == 0 else origin.y()
            k0 = int(math.floor((t - o) / cm)) - 1
            k1 = int(math.ceil((length - o) / cm)) + 1
            for k in range(k0, k1):
                for sub in range(10 if cm >= 40 else 2):
                    v = o + (k + sub / (10 if cm >= 40 else 2)) * cm
                    if v < t or v > length:
                        continue
                    h = 8 if sub == 0 else 4
                    if axis == 0:
                        painter.drawLine(QPointF(v, t - h), QPointF(v, t))
                    else:
                        painter.drawLine(QPointF(t - h, v), QPointF(t, v))
                    if sub == 0 and k % step == 0:
                        if axis == 0:
                            painter.drawText(QPointF(v + 2, 9), str(k))
                        else:
                            painter.drawText(QPointF(2, v - 2), str(k))
        painter.setPen(QPen(QColor("#0078D7"), 1))
        marks = []
        if self._cursor_pos is not None:
            marks.append(self.mapFromScene(self._cursor_pos))
        r = self._selection_rect() if self.selection else None
        if r is not None:
            marks += [self.mapFromScene(r.topLeft()), self.mapFromScene(r.bottomRight())]
        for m in marks:
            painter.drawLine(QPointF(m.x(), 0), QPointF(m.x(), t))
            painter.drawLine(QPointF(0, m.y()), QPointF(t, m.y()))
        painter.restore()


class _KeyValueDialog(QDialog):
    """File > Document Annotations: keywords (Name, Description, Type, Other Info and
    your own) with their contents (ChemDraw 17.1 p. 6)."""
    DEFAULT_KEYS = ("Name", "Description", "Type", "Other Info")

    def __init__(self, values: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Document Annotations")
        keys = list(self.DEFAULT_KEYS) + [k for k in values if k not in self.DEFAULT_KEYS]
        self.table = QTableWidget(len(keys) + 3, 2)
        self.table.setHorizontalHeaderLabels(["Keyword", "Content"])
        self.table.horizontalHeader().setStretchLastSection(True)
        for r, k in enumerate(keys):
            self.table.setItem(r, 0, QTableWidgetItem(k))
            self.table.setItem(r, 1, QTableWidgetItem(str(values.get(k, ""))))
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Keywords and notes stored with the document (.rsc):"))
        lay.addWidget(self.table)
        lay.addWidget(bb)
        self.resize(460, 320)

    def values(self) -> dict:
        out = {}
        for r in range(self.table.rowCount()):
            k, v = self.table.item(r, 0), self.table.item(r, 1)
            key = k.text().strip() if k else ""
            val = v.text().strip() if v else ""
            if key and val:
                out[key] = val
        return out


def _calculators():
    from rsstatchem.ui.chem import (BalanceEquationDialog, EmpiricalFormulaDialog,
                                    FormulaMassDialog, InsertStructureDialog,
                                    MoleculePropertiesDialog, SolutionCalculatorDialog)
    return [("Formula && Molar Mass...", FormulaMassDialog),
            ("Empirical Formula...", EmpiricalFormulaDialog),
            ("Molecule Properties...", MoleculePropertiesDialog), None,
            ("Balance Equation && Stoichiometry...", BalanceEquationDialog),
            ("Solution Calculator...", SolutionCalculatorDialog), None,
            ("Insert Structure into Output...", InsertStructureDialog)]


class ChemDrawWindow(ChemEditorWindow):
    canvas_class = ShellCanvas
    _counter = 0

    def __init__(self, ctx, doc: Document | None = None):
        ChemDrawWindow._counter += 1
        self._number = ChemDrawWindow._counter
        self.ctx = ctx
        self.menus: dict = {}
        self.installed: dict = {}         # install() module name -> None or error text
        self._last_command = None         # (name, callable) for Edit > Repeat
        style = self._default_style_name(ctx)
        super().__init__(None, doc or Document(preset(style)))
        self.setObjectName("chemdraw_window")
        self.setStyleSheet(STYLESHEET)
        self.display_name = ""
        self.online_lookup = self._setting("chemdraw/pubchem", True)
        self.act_pubchem.setChecked(self.online_lookup)
        self.lookup = NameLookup(self)
        self.lookup.finished.connect(self._lookup_done)
        self.use_opsin = True       # offline IUPAC name parsing when Java + OPSIN are there
        self._pending: dict = {}
        self.last_message = ""
        self.zoom_box = QComboBox()
        self.zoom_box.setEditable(True)
        self.zoom_box.setFixedWidth(78)
        self.zoom_box.setToolTip("Magnification (100% = actual size)")
        self.zoom_box.addItems([f"{s:g}%" for s in ZOOM_STEPS] + ["Fit"])
        self.zoom_box.activated.connect(self._zoom_box_changed)
        self.zoom_box.lineEdit().returnPressed.connect(self._zoom_box_changed)
        self.statusBar().addPermanentWidget(QLabel("Zoom:"))
        self.statusBar().addPermanentWidget(self.zoom_box)
        self.canvas.zoom_changed.connect(self._show_zoom)
        self.canvas.selection_changed.connect(self._sync_text_style)
        self._show_zoom(self.canvas.zoom_percent())
        self.resize(1280, 860)
        self._restore_ui()
        self._update_title()

    # --- settings ------------------------------------------------------------------------
    @staticmethod
    def _default_style_name(ctx) -> str:
        """File > Open Style Sheets remembers the style sheet for new documents."""
        s = getattr(ctx, "settings", None)
        name = str(s.value("chemdraw/default_style", DEFAULT_STYLE)) if s is not None else \
            DEFAULT_STYLE
        return name if name in PRESETS else DEFAULT_STYLE

    def _settings(self):
        return getattr(self.ctx, "settings", None)

    def _setting(self, key: str, default):
        s = self._settings()
        if s is None:
            return default
        return s.value(key, default, type=type(default))

    def _restore_ui(self) -> None:
        for tb, key in ((self.palette, "main"), (self.general_bar, "general"),
                        (self.style_bar, "style")):
            tb.setVisible(self._setting(f"chemdraw/show_{key}_toolbar", True))
        self.analysis_dock.setVisible(self._setting("chemdraw/show_analysis", True))
        self.properties_dock.setVisible(self._setting("chemdraw/show_properties", True))

    def _save_ui(self) -> None:
        s = self._settings()
        if s is None:
            return
        for tb, key in ((self.palette, "main"), (self.general_bar, "general"),
                        (self.style_bar, "style")):
            s.setValue(f"chemdraw/show_{key}_toolbar", not tb.isHidden())
        s.setValue("chemdraw/show_analysis", not self.analysis_dock.isHidden())
        s.setValue("chemdraw/show_properties", not self.properties_dock.isHidden())
        s.setValue("chemdraw/pubchem", self.online_lookup)

    # --- construction --------------------------------------------------------------------
    def _build_actions(self) -> None:
        super()._build_actions()
        self.act_new.setText("&New Document")
        self.act_new.setIcon(cd_icon("new"))
        self.act_open.setIcon(cd_icon("open"))
        self.act_save.setIcon(cd_icon("save"))
        self.act_cut.setIcon(cd_icon("cut"))
        self.act_copy.setIcon(cd_icon("copy"))
        self.act_paste.setIcon(cd_icon("paste"))
        self.act_undo.setIcon(cd_icon("undo"))
        self.act_redo.setIcon(cd_icon("redo"))
        self.act_import.setText("Insert &Structure Text (SMILES, MOL, InChI)...")
        self.act_import.setShortcut(QKeySequence("Ctrl+Alt+V"))
        self.act_close.setShortcut(QKeySequence("Ctrl+W"))
        self.act_delete.setText("Clea&r")
        self.act_delete.setShortcut(QKeySequence())
        self.act_clean.setText("Clean &Up Structure")
        self.act_clean.setIcon(cd_icon("clean"))
        self.act_check.setText("&Check Structure")
        self.act_check.setIcon(cd_icon("check"))
        self.act_expand.setText("&Expand Label")
        self.act_balance.setText("Check Reaction &Balance...")
        self.act_stoich.setText("Reaction S&toichiometry...")
        self.act_settings.setText("&Document Settings...")
        self.act_color.setText("Color by &Element")
        self.act_output.setIcon(cd_icon("output"))
        for act, slot in ((self.act_zoom_in, lambda: self.canvas.step_zoom(True)),
                          (self.act_zoom_out, lambda: self.canvas.step_zoom(False)),
                          (self.act_actual, lambda: self.canvas.actual_size()),
                          (self.act_fit, lambda: self.canvas.fit())):
            act.triggered.disconnect()
            act.triggered.connect(slot)
        # ChemDraw 17.1 keys first (User Guide p. 61), the app's earlier keys kept as alternates
        self.act_zoom_in.setText("&Magnify")
        self.act_zoom_in.setShortcuts([QKeySequence("F7"), QKeySequence("Ctrl+]"),
                                       QKeySequence(QKeySequence.StandardKey.ZoomIn)])
        self.act_zoom_in.setIcon(cd_icon("zoom_in"))
        self.act_zoom_out.setText("&Reduce")
        # (Ctrl+- is Subscript, so the platform's Zoom Out key is not added)
        self.act_zoom_out.setShortcuts([QKeySequence("F8"), QKeySequence("Ctrl+[")])
        self.act_zoom_out.setIcon(cd_icon("zoom_out"))
        self.act_actual.setText("&Actual Size")
        self.act_actual.setShortcuts([QKeySequence("F5"), QKeySequence("Ctrl+0")])
        self.act_actual.setIcon(cd_icon("actual"))
        self.act_fit.setText("&Fit to Window")
        self.act_fit.setShortcuts([QKeySequence("F6"), QKeySequence("Ctrl+9")])
        self.act_check.setShortcut(QKeySequence())          # F5 is Actual Size in ChemDraw
        self.act_redo.setShortcut(QKeySequence("Ctrl+Shift+Z"))  # Ctrl+Y is Repeat
        A = self._act
        self.act_print = A("&Print...", self.print_document, QKeySequence.StandardKey.Print)
        self.act_print.setIcon(cd_icon("print"))
        self.act_name2struct = A("Convert &Name to Structure...",
                                 lambda: self.convert_name_to_structure(), "Ctrl+Shift+N")
        self.act_name2struct.setIcon(cd_icon("name2struct"))
        self.act_struct2name = A("Convert Structure to N&ame", self.convert_structure_to_name,
                                 "Ctrl+Alt+N")
        self.act_struct2name.setIcon(cd_icon("struct2name"))
        self.act_pubchem = A("Use PubChem for Names (online)", self._toggle_pubchem,
                             checkable=True,
                             tip="Look up names and structures not in the offline list at "
                                 "PubChem (sends the InChIKey or the name)")
        self.act_fixed_lengths = A("Fixed &Lengths", self._toggle_fixed, "Ctrl+L",
                                   checkable=True)
        self.act_fixed_lengths.setChecked(True)
        self.act_fixed_angles = A("Fixed An&gles", self._toggle_fixed, "Ctrl+E",
                                  checkable=True)
        self.act_fixed_angles.setChecked(True)
        self.act_repeat = A("Repeat", self.repeat_last, "Ctrl+Y",
                            tip="Repeat the last Object command on the selection")
        self.act_repeat.setEnabled(False)
        self.act_join = A("&Join", self.join_selection, "Ctrl+J",
                          tip="Fuse two selected atoms, or two selected bonds, of different "
                              "structures")
        self.act_page_setup = A("Page Set&up...", self.page_setup, "Ctrl+Shift+P")
        self.act_revert = A("Re&vert", self.revert)
        self.act_map = A("&Map Reaction Atoms", self.map_reaction_atoms,
                         tip="Automatic atom-to-atom mapping of the reaction (Indigo)")
        self.act_clear_map = A("Clear Reaction Ma&p", self.clear_reaction_map)
        self.act_clean_rxn = A("Clean Up &Reaction", self.clean_up_reaction,
                               tip="Even spacing, plus signs and arrow length of a "
                                   "single-step reaction")
        self.act_interpret = A("Show Reaction &Interpretation", self._toggle_interpretation,
                               checkable=True,
                               tip="Color the reactants (red), products (magenta) and "
                                   "conditions (cyan) as the reaction is read")
        self.act_rulers = A("Show &Rulers", self._toggle_rulers, "F11", checkable=True)
        self.act_crosshair = A("Show Cross&hair", self._toggle_crosshair, checkable=True)
        self._update_mapping_action()

    def _repeatable(self, name: str, fn):
        """A menu slot that also becomes Edit > Repeat <name> (ChemDraw's Ctrl+Y)."""
        def run(*_args):
            result = fn()
            if result is not False:
                self._remember(name, fn)
            return result
        return run

    def _remember(self, name: str, fn) -> None:
        self._last_command = (name, fn)
        self.act_repeat.setText(f"Repeat {name}")
        self.act_repeat.setEnabled(True)

    def repeat_last(self) -> bool:
        if self._last_command is None:
            return False
        self._last_command[1]()
        return True

    def _build_menus(self) -> None:
        mb = self.menuBar()
        R = self._repeatable
        # File
        m = mb.addMenu("&File")
        self.menus["File"] = m
        m.addAction(self.act_new)
        ss = m.addMenu("Open St&yle Sheets")
        for name in PRESETS:
            ss.addAction(name, lambda n=name: self.open_style_sheet(n))
        m.addAction(self.act_open)
        self.recent_menu = m.addMenu("Open &Recent")
        self.recent_menu.aboutToShow.connect(self._fill_recent)
        m.addSeparator()
        m.addAction(self.act_close)
        m.addAction(self.act_save)
        m.addAction(self.act_save_as)
        m.addAction(self.act_revert)
        m.addSeparator()
        imp = m.addMenu("&Import")
        imp.addAction(self.act_import)
        imp.addAction("Structure &File (CDX, CDXML, MOL, SDF, RXN, HELM, FASTA)...",
                      lambda: self.open_file())
        ex = m.addMenu("&Export")
        for text, fmt in (("ChemDraw Document (.cdx)...", "cdx"),
                          ("ChemDraw XML (.cdxml)...", "cdxml"), (None, None),
                          ("MDL Molfile (.mol)...", "mol"), ("MDL Molfile V3000 (.mol)...", "mol3"),
                          ("SD File (.sdf)...", "sdf"), ("MDL Rxnfile (.rxn)...", "rxn"),
                          ("MDL Rxnfile V3000 (.rxn)...", "rxn3"),
                          ("SMILES...", "smi"), ("Reaction SMILES...", "rsmi"),
                          ("InChI / InChIKey...", "inchi"), ("HELM...", "helm"),
                          ("FASTA...", "fasta"), (None, None),
                          ("PNG Image...", "png"), ("TIFF Image...", "tiff"),
                          ("JPEG Image...", "jpg"), ("BMP Image...", "bmp"),
                          ("SVG Image...", "svg"), ("PDF...", "pdf")):
            if text is None:
                ex.addSeparator()
            else:
                ex.addAction(text, lambda f=fmt: self.export_as(f))
        m.addSeparator()
        m.addAction("Document &Annotations...", lambda: self.document_annotations())
        m.addAction(self.act_settings)
        ap = m.addMenu("Apply Document Settings &from")
        for name in PRESETS:
            ap.addAction(name, lambda n=name: self.apply_style(preset(n)))
        m.addAction("List &Nicknames...", self.list_nicknames)
        m.addSeparator()
        m.addAction(self.act_page_setup)
        m.addAction(self.act_print)
        m.addSeparator()
        m.addAction(self.act_output)
        # Edit
        m = mb.addMenu("&Edit")
        self.menus["Edit"] = m
        m.addAction(self.act_undo)
        m.addAction(self.act_redo)
        m.addAction(self.act_repeat)
        m.addSeparator()
        m.addAction(self.act_cut)
        m.addAction(self.act_copy)
        ca = m.addMenu("Copy &As")
        for text, what, sc in (("MOL Text", "mol", "Ctrl+Alt+Shift+O"),
                               ("MOL V3000 Text", "mol3", None),
                               ("SMILES", "smiles", "Ctrl+Alt+C"), ("InChI", "inchi", None),
                               ("InChIKey", "inchikey", None), ("HELM", "helm", None),
                               ("FASTA", "fasta", None),
                               ("Reaction SMILES", "rsmi", None),
                               ("CDXML Text", "cdxml", "Ctrl+D"),
                               ("Picture (PNG, 300 dpi)", "image", None), ("SVG", "svg", None)):
            act = ca.addAction(text, lambda w=what: self.copy_as(w))
            if sc:
                act.setShortcut(QKeySequence(sc))
        m.addAction(self.act_paste)
        ps = m.addMenu("Paste &Special")
        for text, kind, sc in (("SMILES", "smiles", "Ctrl+Alt+P"),
                               ("MOL / CDXML", "molcdxml", "Ctrl+Alt+Shift+P"),
                               ("InChI", "inchi", None), ("HELM", "helm", None),
                               ("FASTA Peptide", "fasta_peptide", None),
                               ("FASTA DNA", "fasta_dna", None),
                               ("FASTA RNA", "fasta_rna", None),
                               ("Name as Structure", "name", None)):
            act = ps.addAction(text, lambda k=kind: self.paste_special(k))
            if sc:
                act.setShortcut(QKeySequence(sc))
        ps.addSeparator()
        ps.addAction(self.act_import)
        m.addAction(self.act_delete)
        m.addSeparator()
        m.addAction(self.act_select_all)
        m.addSeparator()
        m.addAction("Insert &File...", lambda: self.insert_file())
        # View (filled once the toolbars and windows exist, see _build_docks)
        self.view_menu = mb.addMenu("&View")
        self.menus["View"] = self.view_menu
        # Object
        m = mb.addMenu("&Object")
        self.menus["Object"] = m
        m.addAction(self.act_fixed_lengths)
        m.addAction(self.act_fixed_angles)
        m.addSeparator()
        act = m.addAction("Bring to &Front", R("Bring to Front", lambda: self.z_order(True)))
        act.setShortcuts([QKeySequence("F2"), QKeySequence("Ctrl+Shift+]")])
        act = m.addAction("Send to &Back", R("Send to Back", lambda: self.z_order(False)))
        act.setShortcuts([QKeySequence("F3"), QKeySequence("Ctrl+Shift+[")])
        m.addSeparator()
        m.addAction("&Group", self.group, "Ctrl+G")
        m.addAction("&Ungroup", self.ungroup, "Ctrl+Shift+G")
        m.addAction(self.act_join)
        m.addSeparator()
        al = m.addMenu("&Align")
        for text, how, key in (("Lefts", "left", "L"), ("Horizontal Centers", "hcenter", "C"),
                               ("Rights", "right", "R"), ("Tops", "top", "T"),
                               ("Vertical Centers", "vcenter", "M"),
                               ("Bottoms", "bottom", "B")):
            al.addAction(text, R(f"Align {text}", lambda h=how: self.align(h)),
                         f"Ctrl+Alt+Shift+{key}")
        di = m.addMenu("&Distribute")
        di.addAction("Horizontally", R("Distribute", lambda: self.distribute(True)))
        di.addAction("Vertically", R("Distribute", lambda: self.distribute(False)))
        m.addSeparator()
        m.addAction("Flip &Horizontal", R("Flip Horizontal", lambda: self.transform("flip_h")),
                    "Ctrl+Shift+H")
        m.addAction("Flip &Vertical", R("Flip Vertical", lambda: self.transform("flip_v")),
                    "Ctrl+Shift+V")
        m.addAction("Rotate 180° Horizontal",
                    R("Rotate 180° Horizontal", lambda: self.rotate_180_3d(True)),
                    "Ctrl+Alt+Shift+H")
        m.addAction("Rotate 180° Vertical",
                    R("Rotate 180° Vertical", lambda: self.rotate_180_3d(False)),
                    "Ctrl+Alt+Shift+V")
        m.addAction("Rotate 90° Clockwise", R("Rotate 90°", lambda: self.transform("rot_cw")))
        m.addAction("Rotate 90° Counterclockwise",
                    R("Rotate -90°", lambda: self.transform("rot_ccw")))
        m.addAction("Rotate 180°", R("Rotate 180°", self.rotate_180))
        m.addAction("&Rotate...", lambda: self.rotate_dialog(), "Ctrl+R")
        m.addAction("&Scale...", lambda: self.scale_dialog(), "Ctrl+K")
        m.addSeparator()
        m.addAction("&Center on Page", R("Center on Page", self.center_on_page))
        # Structure
        m = mb.addMenu("&Structure")
        self.menus["Structure"] = m
        m.addAction("&Atom Properties...", self._atom_properties_of_selection)
        m.addAction(self.act_expand)
        m.addSeparator()
        m.addAction("Add Atom &Numbers", lambda: self.atom_numbers(True))
        m.addAction("Remove Atom Numbers", lambda: self.atom_numbers(False))
        m.addSeparator()
        m.addAction(self.act_check)
        m.addAction(self.act_clean)
        m.addAction(self.act_clean_rxn)
        m.addSeparator()
        m.addAction("Structure &Perspective (3D)...", self.structure_perspective)
        m.addAction("&Mass Fragmentation Tool", lambda: self.set_tool("fragment"))
        m.addSeparator()
        m.addAction(self.act_name2struct)
        m.addAction(self.act_struct2name)
        m.addAction(self.act_pubchem)
        m.addSeparator()
        m.addAction(self.act_map)
        m.addAction(self.act_clear_map)
        m.addAction(self.act_balance)
        m.addAction(self.act_stoich)
        m.addSeparator()
        calc = m.addMenu("Chemistry &Calculators")
        for item in _calculators():
            if item is None:
                calc.addSeparator()
            else:
                calc.addAction(item[0], lambda _=False, c=item[1]: self.run_calculator(c))
        m.addAction(self.act_import)
        # Text
        self.text_menu = mb.addMenu("&Text")
        self.menus["Text"] = self.text_menu
        # Curves (ChemDraw: style of the selected curves and of new Pen curves)
        self.curves_menu = mb.addMenu("C&urves")
        self.menus["Curves"] = self.curves_menu
        self.curves_menu.aboutToShow.connect(self._fill_curves_menu)
        self._fill_curves_menu()
        # Colors
        m = mb.addMenu("&Colors")
        self.menus["Colors"] = m
        for name, value in COLORS:
            m.addAction(color_icon(value), name, lambda v=value: self.apply_color(
                "" if v == "#000000" else v))
        m.addAction("&Other...", self.choose_color)
        m.addSeparator()
        m.addAction(self.act_color)
        # Window
        self.windows_menu = mb.addMenu("&Window")
        self.menus["Window"] = self.windows_menu
        self.windows_menu.aboutToShow.connect(self._fill_window_menu)
        self._fill_window_menu()
        # Help
        m = mb.addMenu("&Help")
        self.menus["Help"] = m
        m.addAction("Shortcuts and &Hotkeys", self.show_hotkeys)
        m.addSeparator()
        m.addAction("Check for &Updates...", self._check_updates)
        m.addAction("&About RS-StatChem", self._about)

    def _build_toolbars(self) -> None:
        g = QToolBar("General Toolbar", self)
        g.setObjectName("chemdraw_general_toolbar")
        g.setIconSize(QSize(20, 20))
        for a in (self.act_new, self.act_open, self.act_save, self.act_print, None,
                  self.act_cut, self.act_copy, self.act_paste, None, self.act_undo,
                  self.act_redo, None, self.act_zoom_in, self.act_zoom_out, self.act_actual,
                  None, self.act_clean, self.act_check, None, self.act_name2struct,
                  self.act_struct2name, None, self.act_output):
            g.addSeparator() if a is None else g.addAction(a)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, g)
        self.general_bar = g
        self.addToolBarBreak(Qt.ToolBarArea.TopToolBarArea)
        self._build_style_bar()
        self.palette = MainToolbar(self)
        self.palette.tool_selected.connect(self.set_tool)
        self.palette.command.connect(self._palette_command)
        self.palette.template_chosen.connect(self._template_chosen)
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, self.palette)
        self._tool_buttons = {}

    def _build_style_bar(self) -> None:
        s = QToolBar("Style Toolbar", self)
        s.setObjectName("chemdraw_style_toolbar")
        s.setIconSize(QSize(18, 18))
        self.font_box = QFontComboBox()
        self.font_box.setCurrentFont(QFont("Arial"))
        self.font_box.setFixedWidth(170)
        self.font_box.setToolTip("Font of the selected captions")
        self.font_box.activated.connect(
            lambda _i: self.apply_text_style(font=self.font_box.currentFont().family()))
        self.size_box = QComboBox()
        self.size_box.setEditable(True)
        self.size_box.addItems([str(v) for v in SIZES])
        self.size_box.setCurrentText("10")
        self.size_box.setFixedWidth(56)
        self.size_box.setToolTip("Size (pt) of the selected captions")
        self.size_box.activated.connect(lambda _i: self._size_chosen())
        self.size_box.lineEdit().returnPressed.connect(self._size_chosen)
        s.addWidget(self.font_box)
        s.addWidget(self.size_box)
        s.addSeparator()
        self.style_actions: dict[str, QAction] = {}
        for key, text, sc in (("bold", "&Bold", "Ctrl+B"), ("italic", "&Italic", "Ctrl+I"),
                              ("underline", "&Underline", "Ctrl+U"),
                              ("formula", "&Formula", "Ctrl+F"),
                              ("subscript", "Su&bscript", "F9"),
                              ("superscript", "Su&perscript", "F10"),
                              ("plain", "&Plain", "Ctrl+T")):
            act = QAction(cd_icon(key), text, self)
            act.setShortcuts([QKeySequence(sc)] + {
                "subscript": [QKeySequence("Ctrl+-")],
                "superscript": [QKeySequence("Ctrl+=")]}.get(key, []))
            act.setCheckable(key in ("bold", "italic", "underline", "formula"))
            act.triggered.connect(lambda checked=False, k=key: self._style_action(k, checked))
            self.style_actions[key] = act
            if key == "plain":
                continue
            s.addAction(act)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, s)
        self.style_bar = s
        # Text menu
        m = self.text_menu
        fm = m.addMenu("&Font")
        for f in FONTS:
            fm.addAction(f, lambda f=f: self.apply_text_style(font=f))
        sm = m.addMenu("&Size")
        for v in SIZES:
            sm.addAction(f"{v} pt", lambda v=v: self.apply_text_style(size=float(v)))
        st = m.addMenu("S&tyle")
        st.addAction(self.style_actions["plain"])
        st.addSeparator()
        for key in ("bold", "italic", "underline", "formula", "subscript", "superscript"):
            st.addAction(self.style_actions[key])

    def _build_docks(self) -> None:
        self.analysis = AnalysisWindow(self.canvas)
        self.analysis.paste_requested.connect(self._paste_analysis)
        self.analysis_dock = self._dock("Analysis", "chemdraw_analysis_dock", self.analysis)
        self.properties = ChemicalPropertiesWindow(self.canvas)
        self.properties.paste_requested.connect(self._paste_analysis)
        self.properties_dock = self._dock("Chemical Properties", "chemdraw_properties_dock",
                                          self.properties)
        self.templates = TemplatePanel()
        self.templates.template_chosen.connect(self._template_chosen)
        self.templates.insert_requested.connect(
            lambda smi: self.canvas.insert_text(smi, title="Insert Template"))
        self.template_dock = self._dock("Templates", "chemdraw_template_dock", self.templates)
        self.tabifyDockWidget(self.properties_dock, self.template_dock)
        self.template_dock.hide()
        self.resizeDocks([self.analysis_dock], [300], Qt.Orientation.Horizontal)
        self.resizeDocks([self.analysis_dock, self.properties_dock], [230, 360],
                         Qt.Orientation.Vertical)
        # View menu
        v = self.view_menu
        for tb, text in ((self.palette, "Show &Main Toolbar"),
                         (self.general_bar, "Show &General Toolbar"),
                         (self.style_bar, "Show &Style Toolbar")):
            act = tb.toggleViewAction()
            act.setText(text)
            v.addAction(act)
        v.addSeparator()
        for d, text in ((self.analysis_dock, "Show &Analysis Window"),
                        (self.properties_dock, "Show &Chemical Properties Window"),
                        (self.template_dock, "Show &Templates Window")):
            act = d.toggleViewAction()
            act.setText(text)
            v.addAction(act)
        v.addAction("Show &Periodic Table Window...", self.periodic_table)
        v.addSeparator()
        v.addAction(self.act_interpret)
        v.addAction(self.act_rulers)
        v.addAction(self.act_crosshair)
        v.addAction(self.act_grid)
        self.act_grid.setText("Show &Grid")
        v.addSeparator()
        for a in (self.act_actual, self.act_fit, self.act_zoom_in, self.act_zoom_out):
            v.addAction(a)
        pg = v.addMenu("Page Si&ze")
        group = QActionGroup(self)
        for name in ("Letter", "A4"):
            act = pg.addAction(name, lambda n=name: self.canvas.set_page_size(n))
            act.setCheckable(True)
            act.setChecked(name == "Letter")
            group.addAction(act)
        self.page_size_actions = {a.text(): a for a in group.actions()}
        self._install_extensions()

    def _install_extensions(self) -> None:
        """Call ``install(window)`` of the other work packages' action modules when they
        exist; a missing module is skipped, a failing one is reported and skipped."""
        for name in INSTALL_MODULES:
            try:
                module = importlib.import_module(name)
            except ModuleNotFoundError as exc:
                if exc.name == name:
                    continue                    # not part of this build
                self.installed[name] = traceback.format_exc()
            except Exception:  # noqa: BLE001 - keep the window usable
                self.installed[name] = traceback.format_exc()
            else:
                install = getattr(module, "install", None)
                if install is None:
                    continue
                try:
                    install(self)
                    self.installed[name] = None
                except Exception:  # noqa: BLE001
                    self.installed[name] = traceback.format_exc()
            if self.installed.get(name):
                print(f"[RS-StatChem] {name}.install failed:\n{self.installed[name]}",
                      file=sys.stderr)

    def _dock(self, title: str, name: str, widget) -> QDockWidget:
        d = QDockWidget(title, self)
        d.setObjectName(name)
        d.setWidget(widget)
        d.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable |
                      QDockWidget.DockWidgetFeature.DockWidgetMovable |
                      QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, d)
        return d

    # --- tools ---------------------------------------------------------------------------
    def set_tool(self, tool: str) -> None:
        canvas_tool = tool
        if tool.startswith("bracket:"):
            self.canvas.bracket_kind = tool.split(":", 1)[1]
            canvas_tool = "bracket"
        self.canvas.set_tool(canvas_tool)
        self.current_tool = tool
        palette = getattr(self, "palette", None)
        if palette is not None:
            palette.show_tool(tool)
        if hasattr(self, "status_tool"):
            self.status_tool.setText(TOOL_TIPS.get(tool, tool))

    def _palette_command(self, name: str) -> None:
        if name == "periodic":
            self.periodic_table()
        elif name == "perspective":
            self.structure_perspective()

    def _fill_curves_menu(self) -> None:
        m = self.curves_menu
        m.clear()
        self.canvas.add_curve_menu(m)
        m.addSeparator()
        m.addAction("&Pen Tool", lambda: self.set_tool("pen"))

    def structure_perspective(self):
        """Structure Perspective: rotatable 3D view of the selected (or only) molecule."""
        from rsstatchem.core.chem_draw import threed
        from rsstatchem.ui.chemdraw.perspective import StructurePerspectiveDialog
        frags = self._fragments_in_scope()
        frags = [f for f in frags if len(f) >= 1]
        if len(frags) != 1:
            self._notify("Structure Perspective",
                         "Select one molecule first." if frags else "Draw a molecule first.")
            return None
        comp = self.doc.fragment_of(frags[0][0])
        try:
            model = threed.embed_doc(self.doc, comp)
        except mo.ChemError as exc:
            self._notify("Structure Perspective", str(exc))
            return None
        dlg = StructurePerspectiveDialog(self.canvas, model, self)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dlg.show()
        self.perspective = dlg
        return dlg

    def _template_chosen(self, name: str, smiles: str) -> None:
        self.canvas.template_smiles = smiles
        self.set_tool("template")
        self.status_tool.setText(f"Template: {name} - click in the document to place it")

    def _toggle_fixed(self) -> None:
        self.canvas.fixed_lengths = self.act_fixed_lengths.isChecked()
        self.canvas.fixed_angles = self.act_fixed_angles.isChecked()

    def _toggle_pubchem(self, on: bool) -> None:
        self.online_lookup = bool(on)
        s = self._settings()
        if s is not None:
            s.setValue("chemdraw/pubchem", self.online_lookup)

    # --- zoom ----------------------------------------------------------------------------
    def _show_zoom(self, pct: float) -> None:
        if hasattr(self, "zoom_box"):
            self.zoom_box.setEditText(f"{pct:.0f}%")

    def _zoom_box_changed(self, *_args) -> None:
        text = self.zoom_box.currentText().strip().rstrip("%").strip()
        if text.lower() == "fit":
            self.canvas.fit()
            return
        try:
            self.canvas.set_zoom_percent(float(text.replace(",", ".")))
        except ValueError:
            self._show_zoom(self.canvas.zoom_percent())
        self.canvas.setFocus()

    # --- document state ------------------------------------------------------------------
    def _doc_changed(self) -> None:
        n = len(self.canvas.problems)
        self.status_problems.setText(f"{n} atom(s) with valence problems" if n else "")
        self.status_problems.setStyleSheet("color: #C62828;")
        self.act_color.blockSignals(True)
        self.act_color.setChecked(self.doc.style.color_by_element)
        self.act_color.blockSignals(False)
        self._update_title()

    def document_name(self) -> str:
        if self.path:
            return os.path.basename(self.path)
        if getattr(self, "display_name", ""):
            return self.display_name
        return f"Untitled Document-{getattr(self, '_number', 1)}"

    def _update_title(self) -> None:
        dirty = not self.undo_stack.isClean()
        self.setWindowModified(dirty)
        self.setWindowTitle(f"Chemical Drawing - [{self.document_name()}[*]] — RS-StatChem")

    def is_pristine(self) -> bool:
        return self.doc.is_empty() and self.undo_stack.count() == 0

    # --- files ---------------------------------------------------------------------------
    def new_window(self):
        from rsstatchem.ui.modules import open_module
        return open_module(self.ctx, "chem")

    def _report(self, title: str, text: str, deferred: bool = False) -> None:
        self.last_message = text
        self.statusBar().showMessage(text, 8000)
        if deferred:
            QTimer.singleShot(0, self, lambda: QMessageBox.warning(self, title, text))
        else:
            QMessageBox.warning(self, title, text)

    def open_file(self, path: str | None = None, deferred_errors: bool = False) -> bool:
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "Open", self._start_dir(), OPEN_FILTER)
            if not path:
                return False
        ext = os.path.splitext(path)[1].lower()
        if not self.is_pristine():          # ChemDraw opens a file in a new document window
            from rsstatchem.ui.modules import open_module
            return open_module(self.ctx, "chem", path) is not None
        try:
            if ext in _DOC_EXT:
                doc = self._load_document(path)
                self.canvas.set_state(doc.to_dict())
                self.path = path
            elif ext in (".helm", ".fasta", ".fa"):
                with open(path, encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
                c = self.canvas.page_center()
                self._insert_sequence_text(text, "helm" if ext == ".helm" else "fasta_peptide",
                                           c.x(), self.canvas.page.top() + 200)
                self.canvas.rebuild()
                self.display_name = os.path.basename(path)
            else:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
                if ext in (".smi", ".txt") and ("\t" in text or " " in text.strip()):
                    text = "\n".join(line.split()[0] for line in text.splitlines()
                                     if line.strip())
                c = self.canvas.page_center()
                convert.insert_text(self.doc, text, c.x(), self.canvas.page.top() + 200)
                self.canvas.rebuild()
                self.display_name = os.path.basename(path)
        except (mo.ChemError, ValueError, OSError, KeyError) as exc:
            self._report("Open", f"Could not read {os.path.basename(path)}:\n{exc}",
                         deferred_errors)
            return False
        self._place_on_page()
        self.canvas.select_keys(set())
        self.undo_stack.clear()
        self._remember_dir(path)
        if self.ctx is not None and hasattr(self.ctx, "add_recent"):
            self.ctx.add_recent(path)
        self._update_title()
        self.canvas.doc_changed.emit()
        self.canvas.fit_later()
        return True

    def _place_on_page(self) -> None:
        """Move a drawing whose coordinates lie off the page (old files, imports) onto it."""
        bb = self.doc.bbox()
        if bb is None:
            return
        page = self.canvas.page
        if page.left() <= bb[0] and bb[2] <= page.right() and page.top() <= bb[1] and \
                bb[3] <= page.bottom():
            return
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        top = page.top() + 72 if h < page.height() - 144 else page.top() + 18
        dx = page.center().x() - (bb[0] + bb[2]) / 2
        dy = top - bb[1]
        if w > page.width():
            dx = page.left() + 18 - bb[0]
        self.doc.move(set(self.doc.keys()), dx, dy)
        self.canvas.rebuild()

    def load_text(self, text: str) -> list:
        c = self.canvas.page_center()
        keys = self.canvas.insert_text(text, c.x(), self.canvas.page.top() + 200,
                                       "Insert Structure")
        self.canvas.select_keys(set())
        self.undo_stack.clear()
        self._update_title()
        self.canvas.fit_later()
        return keys

    def save(self) -> bool:
        if not self.path or os.path.splitext(self.path)[1].lower() not in _DOC_EXT:
            return self.save_as()
        try:
            if self.path.lower().endswith(".cdxml"):
                cdxml.save_cdxml(self.doc, self.path)
            elif self.path.lower().endswith(".cdx"):
                cdx.save_cdx(self.doc, self.path)
            else:
                save_rsc(self.doc, self.path)
        except OSError as exc:
            QMessageBox.warning(self, "Save", str(exc))
            return False
        self.undo_stack.setClean()
        if self.ctx is not None and hasattr(self.ctx, "add_recent"):
            self.ctx.add_recent(self.path)
        self._update_title()
        return True

    def save_as(self, path: str | None = None) -> bool:
        if path is None:
            base = os.path.splitext(self.document_name())[0]
            start = self.path or os.path.join(self._start_dir(), base + ".rsc")
            path, chosen = QFileDialog.getSaveFileName(self, "Save As", start,
                                                       ";;".join(SAVE_FILTERS))
            if not path:
                return False
            if os.path.splitext(path)[1].lower() not in _DOC_EXT:
                path += SAVE_FILTERS.get(chosen, ".rsc")
        self.path = path
        self.display_name = ""
        self._remember_dir(path)
        return self.save()

    def export_as(self, fmt: str, path: str | None = None, dpi: int | None = None,
                  transparent: bool | None = None) -> bool:
        if fmt != "cdxml":
            return super().export_as(fmt, path, dpi, transparent)
        if path is None:
            base = os.path.splitext(self.document_name())[0]
            path, _ = QFileDialog.getSaveFileName(self, "Export", os.path.join(
                self._start_dir(), base + ".cdxml"), "ChemDraw XML (*.cdxml)")
            if not path:
                return False
        if not os.path.splitext(path)[1]:
            path += ".cdxml"
        try:
            cdxml.save_cdxml(self._export_doc(), path)
        except OSError as exc:
            QMessageBox.warning(self, "Export", str(exc))
            return False
        self._remember_dir(path)
        self.statusBar().showMessage(f"Exported {os.path.basename(path)}", 4000)
        return True

    def copy_as(self, what: str) -> None:
        if what == "cdxml":
            from PySide6.QtGui import QGuiApplication
            QGuiApplication.clipboard().setText(cdxml.to_cdxml(self._export_doc()))
            self.statusBar().showMessage("Copied as CDXML text", 4000)
            return
        super().copy_as(what)

    def _fill_recent(self) -> None:
        self.recent_menu.clear()
        files = []
        if self.ctx is not None and hasattr(self.ctx, "recent_files"):
            files = [p for p in self.ctx.recent_files()
                     if os.path.splitext(p)[1].lower() in _CHEM_EXT]
        if not files:
            act = self.recent_menu.addAction("(no recent documents)")
            act.setEnabled(False)
        for p in files:
            self.recent_menu.addAction(os.path.basename(p), lambda p=p: self.open_file(p))

    def print_document(self, printer=None) -> bool:
        from PySide6.QtCore import QRectF
        from PySide6.QtGui import QPainter
        from PySide6.QtPrintSupport import QPrintDialog, QPrinter
        from PySide6.QtWidgets import QGraphicsScene

        from rsstatchem.ui.chem_editor.render import Builder
        if printer is None:
            printer = QPrinter(QPrinter.PrinterMode.HighResolution)
            dlg = QPrintDialog(printer, self)
            dlg.setWindowTitle("Print")
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return False
        scene = QGraphicsScene()
        Builder(self.doc.style).build(scene, self.doc)
        painter = QPainter()
        if not painter.begin(printer):
            QMessageBox.warning(self, "Print", "The printer could not be started.")
            return False
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        target = QRectF(0, 0, printer.width(), printer.height())
        scene.render(painter, target, self.canvas.page, Qt.AspectRatioMode.KeepAspectRatio)
        painter.end()
        scene.clear()
        return True

    def _paste_analysis(self, text: str) -> None:
        """Analysis / Chemical Properties > Paste: a caption under the analysed structure."""
        if not text:
            return
        keys = self.doc.with_bonds(self.canvas.selection) if self.canvas.selection else \
            {("atom", a) for a in self.doc.atoms}
        bb = self.doc.bbox(keys) or self.doc.bbox()
        st = self.doc.style
        x = bb[0] if bb else self.canvas.page_center().x()
        y = bb[3] + max(st.font_size, st.bond_length * 0.6) if bb else \
            self.canvas.page.top() + 72
        self.canvas.edit("Paste Analysis", lambda: self.doc.add_text(x, y, text))

    # --- integration ---------------------------------------------------------------------
    def _send_section(self, section) -> bool:
        ctx = self.ctx
        if ctx is None or not hasattr(ctx, "show_output"):
            QMessageBox.information(self, "Output Viewer", "No Output Viewer is available.")
            return False
        ctx.show_output(section)
        return True

    def run_calculator(self, cls):
        """Open a Chemistry calculator; OK sends its result to the Output Viewer."""
        dlg = cls(None, self)
        if hasattr(dlg, "input") and self.doc.atoms:
            try:
                dlg.input.setPlainText(convert.to_smiles(self._export_doc()))
            except mo.ChemError:
                pass
        if dlg.exec():
            try:
                section = dlg.compute()
            except (ValueError, KeyError, ZeroDivisionError) as exc:
                QMessageBox.warning(self, dlg.windowTitle(), str(exc))
                return None
            if section is not None:
                self._send_section(getattr(section, "section", section))
            return section
        return None

    def _fill_window_menu(self) -> None:
        from rsstatchem.ui.modules import add_module_menu
        m = self.windows_menu
        m.clear()
        wins = [w for w in getattr(self.ctx, "windows", []) if isinstance(w, ChemDrawWindow)]
        if self not in wins:
            wins.insert(0, self)
        for w in wins:
            act = m.addAction(w.document_name())
            act.setCheckable(True)
            act.setChecked(w is self)
            act.triggered.connect(lambda _=False, w=w: (w.show(), w.raise_(),
                                                        w.activateWindow()))
        m.addSeparator()
        m.addAction("Minimize", self.showMinimized, "Ctrl+M")
        if self.ctx is not None:
            m.addSeparator()
            add_module_menu(m, self.ctx)

    def _check_updates(self) -> None:
        if self.ctx is not None:
            self.ctx.check_for_updates(manual=True)

    def _about(self) -> None:
        if self.ctx is not None:
            self.ctx.about()

    def show_hotkeys(self) -> QDialog:
        dlg = QDialog(self)
        dlg.setWindowTitle("Shortcuts and Hotkeys")
        view = QTextBrowser()
        # objects_actions.install() may set window.hotkeys_html (the drawing hotkeys)
        view.setHtml(getattr(self, "hotkeys_html", HOTKEYS_HTML) + _shortcuts_html())
        lay = QVBoxLayout(dlg)
        lay.addWidget(view)
        dlg.resize(480, 560)
        dlg.show()          # modeless, like ChemDraw's cheat sheet
        return dlg

    # --- Object menu ---------------------------------------------------------------------
    def rotate_180(self) -> None:
        keys = self.canvas.target_keys()
        self.canvas.edit("Rotate", lambda: self.doc.rotate(keys, math.pi))

    def rotate_by(self, degrees: float) -> None:
        keys = self.canvas.target_keys()
        self.canvas.edit("Rotate", lambda: self.doc.rotate(keys, math.radians(degrees)))

    def rotate_dialog(self, degrees: float | None = None) -> None:
        if degrees is None:
            degrees, ok = QInputDialog.getDouble(self, "Rotate Objects",
                                                 "Angle (degrees, clockwise):", 15.0,
                                                 -360.0, 360.0, 1)
            if not ok:
                return
        self.rotate_by(degrees)
        self._remember(f"Rotate {degrees:g}°", lambda d=degrees: self.rotate_by(d))

    def scale_dialog(self, pct: float | None = None) -> None:
        if pct is None:
            pct, ok = QInputDialog.getDouble(self, "Scale Objects", "Scale (%):", 100.0, 5.0,
                                             2000.0, 1)
            if not ok:
                return
        if abs(pct - 100.0) > 1e-9:
            self.scale_selection(pct)
            self._remember(f"Scale {pct:g}%", lambda p=pct: self.scale_selection(p))

    def rotate_180_3d(self, horizontal: bool) -> None:
        """Object > Rotate 180° Horizontal / Vertical: turn the selection over about the
        vertical (horizontal) axis in 3D - the mirror image in the plane, with wedges and
        hashes exchanged so every stereocentre keeps its configuration."""
        keys = self.canvas.target_keys()
        if not keys:
            return

        def fn():
            self.doc.flip(keys, horizontal)
            for k, i in self.doc.with_bonds(keys):
                if k == "bond" and i in self.doc.bonds:
                    b = self.doc.bonds[i]
                    if b.style in ("wedge", "hollow"):
                        b.style = "hash"
                    elif b.style == "hash":
                        b.style = "wedge"
        self.canvas.edit("Rotate 180°", fn)

    def join_selection(self) -> bool:
        """Object > Join (ChemDraw 17.1 p. 100): two selected atoms of different structures
        are fused into one (the second structure moves onto the first); two selected bonds are
        fused so that the structures share that bond."""
        sel = self.canvas.selection
        atoms = sorted(a for k, a in sel if k == "atom" and a in self.doc.atoms)
        bonds = sorted(b for k, b in sel if k == "bond" and b in self.doc.bonds)
        if len(bonds) == 2:
            b1, b2 = self.doc.bonds[bonds[0]], self.doc.bonds[bonds[1]]
            pair1, pair2 = (b1.a, b1.b), (b2.a, b2.b)
        elif len(atoms) == 2:
            pair1, pair2 = (atoms[0],), (atoms[1],)
        else:
            self.statusBar().showMessage("Select two atoms or two bonds to join", 5000)
            return False
        frag1 = set(self.doc.fragment_of(pair1[0]))
        if any(a in frag1 for a in pair2):
            self.statusBar().showMessage("Join works on two separate structures", 5000)
            return False
        moving = set(self.doc.fragment_of(pair2[0]))
        P = self.doc.atoms

        def fn():
            keys = self.doc.with_bonds({("atom", a) for a in moving})
            if len(pair1) == 1:
                k, d = pair1[0], pair2[0]
                self.doc.move(keys, P[k].x - P[d].x, P[k].y - P[d].y)
                self.doc.merge_atoms(d, k)
                return
            # rotate + move bond 2 onto bond 1, choosing the orientation that keeps the second
            # structure on the far side of the shared bond
            best = None
            for d1, d2 in (pair2, pair2[::-1]):
                ang = math.atan2(P[pair1[1]].y - P[pair1[0]].y, P[pair1[1]].x - P[pair1[0]].x) - \
                    math.atan2(P[d2].y - P[d1].y, P[d2].x - P[d1].x)
                c, s_ = math.cos(ang), math.sin(ang)
                pts = {}
                for a in moving:
                    x, y = P[a].x - P[d1].x, P[a].y - P[d1].y
                    pts[a] = (P[pair1[0]].x + c * x - s_ * y, P[pair1[0]].y + s_ * x + c * y)
                cx = sum(P[a].x for a in frag1) / len(frag1)
                cy = sum(P[a].y for a in frag1) / len(frag1)
                mx = sum(p[0] for p in pts.values()) / len(pts)
                my = sum(p[1] for p in pts.values()) / len(pts)
                dist = math.hypot(mx - cx, my - cy)
                if best is None or dist > best[0]:
                    best = (dist, pts, (d1, d2))
            _, pts, (d1, d2) = best
            for a, (x, y) in pts.items():
                P[a].x, P[a].y = x, y
            self.doc.merge_atoms(d1, pair1[0])
            self.doc.merge_atoms(d2, pair1[1])
        self.canvas.edit("Join", fn)
        return True

    def scale_selection(self, pct: float) -> None:
        keys = self.canvas.target_keys()
        if not keys:
            return
        cx, cy = self.doc.center(keys)
        k = pct / 100.0

        def fn():
            atoms = self.doc.atoms_of(keys)
            for a in atoms:
                o = self.doc.atoms[a]
                o.x, o.y = cx + (o.x - cx) * k, cy + (o.y - cy) * k
            for kind, i in keys:
                o = self.doc.get((kind, i))
                if kind in ("text", "plus", "orbital"):
                    o.x, o.y = cx + (o.x - cx) * k, cy + (o.y - cy) * k
                    if kind == "orbital":
                        o.size = (o.size or self.doc.style.bond_length) * k
                elif kind == "curve":
                    o.points = [[cx + (x - cx) * k, cy + (y - cy) * k] for x, y in o.points]
                elif kind in ("arrow", "bracket"):
                    o.x1, o.y1 = cx + (o.x1 - cx) * k, cy + (o.y1 - cy) * k
                    o.x2, o.y2 = cx + (o.x2 - cx) * k, cy + (o.y2 - cy) * k
        self.canvas.edit("Scale", fn)

    def center_on_page(self) -> None:
        keys = self.canvas.target_keys()
        bb = self.doc.bbox(keys)
        if bb is None:
            return
        c = self.canvas.page_center()
        dx, dy = c.x() - (bb[0] + bb[2]) / 2, c.y() - (bb[1] + bb[3]) / 2
        self.canvas.edit("Center on Page", lambda: self.doc.move(keys, dx, dy))

    # --- Structure menu: reactions -------------------------------------------------------
    def _update_mapping_action(self) -> None:
        ok = mapping.available()
        self.act_map.setEnabled(ok)
        tip = "Automatic atom-to-atom mapping of the reaction (Indigo " + \
            mapping.indigo_version() + ")" if ok else mapping.why_unavailable()
        self.act_map.setToolTip(tip)
        self.act_map.setStatusTip(tip)

    def map_reaction_atoms(self) -> dict:
        """Structure > Map Reaction Atoms: automatic mapping (Indigo); map numbers appear on
        the mapped atoms."""
        if not mapping.available():
            self._notify("Map Reaction Atoms", mapping.why_unavailable())
            return {}
        try:
            result = self.canvas.edit("Map Reaction Atoms",
                                      lambda: mapping.map_reaction(self.doc))
        except mapping.MappingError as exc:
            self._notify("Map Reaction Atoms", str(exc))
            return {}
        self.statusBar().showMessage(f"{len(set(result.values()))} atom(s) mapped", 6000)
        return result

    def clear_reaction_map(self) -> int:
        atoms = self.canvas.selected_atoms() or None
        return self.canvas.edit("Clear Reaction Map",
                                lambda: mapping.clear_map(self.doc, atoms)) or 0

    def clean_up_reaction(self) -> bool:
        ok = self.canvas.edit("Clean Up Reaction", lambda: convert.clean_up_reaction(self.doc))
        if not ok:
            self._notify("Clean Up Reaction",
                         "Clean Up Reaction needs a single-step reaction with reactants, a "
                         "straight reaction arrow and products.")
        return bool(ok)

    def _toggle_interpretation(self, on: bool) -> None:
        self.canvas.show_interpretation = bool(on)
        self.canvas.viewport().update()

    def _toggle_rulers(self, on: bool) -> None:
        self.canvas.show_rulers = bool(on)
        self.canvas.viewport().update()

    def _toggle_crosshair(self, on: bool) -> None:
        self.canvas.show_crosshair = bool(on)
        self.canvas.viewport().update()

    # --- File / Edit extras ---------------------------------------------------------------
    def open_style_sheet(self, name: str):
        """File > Open Style Sheets: a new document with that style sheet, which also
        becomes the style of later new documents."""
        s = self._settings()
        if s is not None:
            s.setValue("chemdraw/default_style", name)
        if self.is_pristine():
            self.apply_style(preset(name))
            self.undo_stack.clear()
            return self
        win = ChemDrawWindow(self.ctx, Document(preset(name)))
        from rsstatchem.ui.chem_editor import _WINDOWS
        _WINDOWS.append(win)
        if self.ctx is not None and hasattr(self.ctx, "windows") and \
                isinstance(self.ctx.windows, list):
            self.ctx.windows.append(win)
        win.show()
        return win

    def revert(self) -> bool:
        """File > Revert: reload the last saved version."""
        if not self.path or not os.path.exists(self.path):
            self.statusBar().showMessage("The document has not been saved yet", 5000)
            return False
        if not self.undo_stack.isClean() and self.confirm_close:
            r = QMessageBox.question(self, "Revert",
                                     f"Discard the changes to '{self.document_name()}'?")
            if r != QMessageBox.StandardButton.Yes:
                return False
        try:
            doc = self._load_document(self.path)
        except (ValueError, OSError, KeyError) as exc:
            self._report("Revert", str(exc))
            return False
        self.canvas.set_state(doc.to_dict())
        self.undo_stack.clear()
        self._update_title()
        return True

    def page_setup(self) -> bool:
        """File > Page Setup: paper size (Letter / A4) of the document page."""
        from PySide6.QtGui import QPageSize
        from PySide6.QtPrintSupport import QPageSetupDialog, QPrinter
        printer = QPrinter()
        size = QPageSize(QPageSize.PageSizeId.A4 if self.canvas.page.width() < 600 else
                         QPageSize.PageSizeId.Letter)
        printer.setPageSize(size)
        dlg = QPageSetupDialog(printer, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return False
        pid = printer.pageLayout().pageSize().id()
        name = "A4" if pid == QPageSize.PageSizeId.A4 else "Letter"
        self.set_page_size(name)
        return True

    def set_page_size(self, name: str) -> None:
        self.canvas.set_page_size(name)
        act = getattr(self, "page_size_actions", {}).get(name)
        if act is not None:
            act.setChecked(True)

    def document_annotations(self, values: dict | None = None) -> dict:
        """File > Document Annotations: keyword / content notes kept in the document."""
        if values is None:
            dlg = _KeyValueDialog(dict(self.doc.meta.get("annotations") or {}), self)
            if not dlg.exec():
                return dict(self.doc.meta.get("annotations") or {})
            values = dlg.values()
        self.canvas.edit("Document Annotations",
                         lambda: self.doc.meta.__setitem__("annotations", dict(values)))
        return dict(values)

    def list_nicknames(self) -> QDialog:
        """File > List Nicknames: the nicknames (abbreviations) the labels understand."""
        from rsstatchem.core.chem_draw import abbrev
        dlg = QDialog(self)
        dlg.setWindowTitle("List Nicknames")
        rows = sorted(abbrev.ABBREVIATIONS.values(), key=lambda a: a.name.lower())
        table = QTableWidget(len(rows), 3)
        table.setHorizontalHeaderLabels(["Nickname", "Name", "Structure (SMILES)"])
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        for r, ab in enumerate(rows):
            for c, text in enumerate((ab.name, ab.title, ab.smiles.replace("*", "-", 1)
                                      if ab.smiles.startswith("*") else ab.smiles)):
                table.setItem(r, c, QTableWidgetItem(text))
        table.horizontalHeader().setStretchLastSection(True)
        table.resizeColumnsToContents()
        lay = QVBoxLayout(dlg)
        lay.addWidget(table)
        dlg.resize(520, 560)
        dlg.show()
        self.nicknames_dialog = dlg
        return dlg

    def insert_file(self, path: str | None = None) -> list:
        """Edit > Insert File: add the contents of a drawing or structure file to this
        document (File > Open makes a new document)."""
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "Insert File", self._start_dir(),
                                                  OPEN_FILTER)
            if not path:
                return []
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext in _DOC_EXT:
                keys = self.canvas.paste_document(self._load_document(path))
            else:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
                x, y = self._free_spot()
                if ext in (".helm", ".fasta", ".fa"):
                    kind = "helm" if ext == ".helm" else "fasta_peptide"
                    keys = self.canvas.edit("Insert File", lambda: self._insert_sequence_text(
                        text, kind, x, y))
                else:
                    keys = self.canvas.insert_text(text, x, y, "Insert File")
        except (mo.ChemError, ValueError, OSError, KeyError) as exc:
            self._report("Insert File", f"Could not read {os.path.basename(path)}:\n{exc}")
            return []
        self._remember_dir(path)
        return keys or []

    def paste_special(self, kind: str, text: str | None = None) -> list:
        """Edit > Paste Special, plus ChemDraw's "MOL / CDXML" (Alt+Shift+Ctrl+P) and
        "Name as Structure" (the clipboard name converted like Convert Name to Structure)."""
        from PySide6.QtGui import QGuiApplication
        if text is None:
            text = QGuiApplication.clipboard().text()
        if kind == "molcdxml":
            kind = "cdxml" if "<CDXML" in (text or "") else "mol"
        if kind == "name":
            name = " ".join((text or "").split())
            if not name:
                self.statusBar().showMessage("The clipboard holds no name", 4000)
                return []
            self.convert_name_to_structure(name)
            return []
        return super().paste_special(kind, text)

    # --- Structure menu ------------------------------------------------------------------
    def _atom_properties_of_selection(self) -> None:
        atoms = sorted(self.canvas.selected_atoms())
        target = atoms[0] if atoms else (self.canvas.hover[1] if self.canvas.hover and
                                         self.canvas.hover[0] == "atom" else None)
        if target is None:
            self.statusBar().showMessage("Select an atom first", 4000)
            return
        self.canvas.atom_properties(target)

    def atom_numbers(self, show: bool) -> None:
        atoms = self.canvas.selected_atoms() or set(self.doc.atoms)
        if not atoms:
            return

        def fn():
            for comp in self.doc.fragments():
                part = sorted(a for a in comp if a in atoms)
                for n, a in enumerate(part, 1):
                    self.doc.atoms[a].number = n if show else 0
        self.canvas.edit("Atom Numbers", fn)

    def _fragments_in_scope(self) -> list[list[int]]:
        atoms = self.canvas.selected_atoms() or set(self.doc.atoms)
        out = []
        for comp in self.doc.fragments():
            part = [a for a in comp if a in atoms]
            if part:
                out.append(part)
        return out

    def _caption_under(self, atoms, text: str) -> int | None:
        if not atoms or not all(a in self.doc.atoms for a in atoms):
            return None
        keys = self.doc.with_bonds({("atom", a) for a in atoms})
        bb = self.doc.bbox(keys)
        st = self.doc.style
        cx = (bb[0] + bb[2]) / 2
        width = len(text) * st.caption_size * 0.5
        x, y = cx - width / 2, bb[3] + max(st.font_size, st.bond_length * 0.6)
        return self.canvas.edit("Convert Structure to Name",
                                lambda: self.doc.add_text(x, y, text, chemical=False).id)

    def convert_structure_to_name(self) -> list[str]:
        """ChemDraw's Convert Structure to Name: a caption under each selected structure from
        the offline list, then the offline name database, else (when allowed) from PubChem
        in the background."""
        frags = self._fragments_in_scope()
        if not frags:
            self.statusBar().showMessage("Draw or select a structure first", 4000)
            return []
        found, unknown = [], 0
        for comp in frags[:12]:
            try:
                mol = convert.to_mol(self.doc, comp)
            except mo.ChemError as exc:
                self._notify("Convert Structure to Name", f"Invalid structure: {exc}")
                continue
            hit = names.offline_name(mol) or names.db_name(mol)
            if hit is not None:
                self._caption_under(comp, hit.name)
                found.append(hit.name)
                self.statusBar().showMessage(f"Name from the {hit.source}", 6000)
                continue
            if self.online_lookup:
                try:
                    key = mo.to_inchikey(mol)
                except Exception:  # noqa: BLE001
                    key = ""
                if key:
                    self._pending[("name", key)] = comp
                    self.lookup.start("name", key)
                    self.statusBar().showMessage("Looking up the name at PubChem...", 8000)
                    continue
            unknown += 1
        if unknown:
            self._notify("Convert Structure to Name",
                         "No name is known offline for this structure. Turn on Structure > "
                         "Use PubChem for Names (online) to look it up at PubChem."
                         if not self.online_lookup else "No name was found.")
        return found

    def convert_name_to_structure(self, name: str | None = None) -> bool:
        """ChemDraw's Convert Name to Structure: a selected caption is replaced by the
        structure; otherwise the name is asked for and the structure inserted. Sources in
        order: the offline list, SMILES / InChI typed directly, the offline name database,
        OPSIN (offline, in the background) and finally PubChem (when allowed)."""
        replace = None
        if name is None:
            texts = [i for k, i in self.canvas.selection if k == "text"]
            if len(texts) == 1:
                replace, name = texts[0], self.doc.texts[texts[0]].text
            else:
                name, ok = QInputDialog.getText(self, "Convert Name to Structure",
                                                "Chemical name:")
                if not ok:
                    return False
        name = " ".join((name or "").split())
        if not name:
            return False
        hit = names.offline_structure(name)
        if hit is not None:
            self._insert_named(hit.smiles, replace, f"{hit.name} ({hit.source})")
            return True
        if " " not in name:
            try:
                if convert.mols_from_text(name):    # SMILES / InChI typed directly
                    self._insert_named(name, replace, "structure text")
                    return True
            except Exception:  # noqa: BLE001 - not a SMILES either
                pass
        hit = names.db_structure(name)
        if hit is not None:
            self._insert_named(hit.smiles, replace, f"{hit.name} ({hit.source})")
            return True
        if self.use_opsin and names.opsin_available():
            self._pending[("opsin", name)] = replace
            self.lookup.start("opsin", name)
            self.statusBar().showMessage(f"Interpreting '{name}' with OPSIN...", 8000)
            return True
        return self._pubchem_structure(name, replace)

    def _pubchem_structure(self, name: str, replace, why: str = "") -> bool:
        if not self.online_lookup:
            self._notify("Convert Name to Structure",
                         (why + " " if why else "") +
                         f"'{name}' is not in the offline name list or name database"
                         + ("" if self.use_opsin and names.opsin_available() else
                            ", and OPSIN (offline name parser, needs Java) is not available")
                         + ". Turn on Structure > Use PubChem for Names (online) to look it "
                           "up at PubChem.")
            return False
        self._pending[("structure", name)] = replace
        self.lookup.start("structure", name)
        self.statusBar().showMessage(f"Looking up '{name}' at PubChem...", 8000)
        return True

    def _insert_named(self, text: str, replace, what: str) -> list:
        x = y = None
        if replace is not None and replace in self.doc.texts:
            t = self.doc.texts[replace]
            x, y = t.x + self.doc.style.bond_length * 2, t.y + self.doc.style.bond_length
        elif self.doc.atoms or self.doc.texts:
            x, y = self._free_spot()
        else:
            c = self.canvas.page_center()
            x, y = c.x(), self.canvas.page.top() + 200

        def fn():
            if replace is not None and replace in self.doc.texts:
                self.doc.remove([("text", replace)])
            return convert.insert_text(self.doc, text, x, y)
        keys = self.canvas.edit("Convert Name to Structure", fn)
        self.canvas.select_keys(set(keys or []))
        self.statusBar().showMessage(f"Structure from {what}", 6000)
        return keys

    def _lookup_done(self, kind: str, query: str, result, error) -> None:
        target = self._pending.pop((kind, query), None)
        if kind == "opsin":
            if error or result is None:
                self._pubchem_structure(query, target, error or "")
                return
            try:
                self._insert_named(result.smiles, target, result.source)
            except (mo.ChemError, ValueError) as exc:
                self._pubchem_structure(query, target, f"OPSIN's structure is invalid ({exc}).")
            return
        if error:
            self._notify("PubChem", error)
            return
        if kind == "name":
            self._caption_under(target, result.name)
            self.statusBar().showMessage(f"Name from PubChem: {result.name}", 6000)
        else:
            try:
                self._insert_named(result.molblock, target, "PubChem")
            except (mo.ChemError, ValueError) as exc:
                self._notify("PubChem", f"The structure from PubChem could not be read: {exc}")

    def _notify(self, title: str, text: str) -> None:
        """Non-blocking message (never runs a nested event loop)."""
        self.last_message = text
        self.statusBar().showMessage(text, 10000)
        box = QMessageBox(QMessageBox.Icon.Information, title, text,
                          QMessageBox.StandardButton.Ok, self)
        box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        box.open()

    # --- Text / Colors -------------------------------------------------------------------
    def selected_texts(self) -> list[int]:
        return [i for k, i in self.canvas.selection if k == "text" and i in self.doc.texts]

    def _size_chosen(self) -> None:
        try:
            v = float(self.size_box.currentText().replace(",", "."))
        except ValueError:
            return
        if 2 <= v <= 200:
            self.apply_text_style(size=v)

    def _style_action(self, key: str, checked: bool) -> None:
        if key in ("subscript", "superscript"):
            self.apply_text_style(script="sub" if key == "subscript" else "sup")
        elif key == "plain":
            self.apply_text_style(bold=False, italic=False, underline=False, chemical=False)
        elif key == "formula":
            self.apply_text_style(chemical=checked)
        else:
            self.apply_text_style(**{key: checked})

    def apply_text_style(self, **kw) -> bool:
        """Font / size / bold / italic / underline / formula / sub- and superscript of the
        selected captions (or of the selected characters while a caption is being typed)."""
        script = kw.pop("script", None)
        ed = self.canvas._editor
        if script and ed is not None and hasattr(ed, "textCursor"):
            cur = ed.textCursor()
            if cur.hasSelection():
                cur.insertText(f"<{script}>{cur.selectedText()}</{script}>")
                return True
        tids = self.selected_texts()
        if not tids:
            atoms = self.canvas.selected_atoms()
            if atoms and ("font" in kw or "size" in kw):
                st = self.doc.style.updated(
                    font_family=kw.get("font", self.doc.style.font_family),
                    font_size=kw.get("size", self.doc.style.font_size))
                self.canvas.edit("Label Font", lambda: setattr(self.doc, "style", st))
                self.statusBar().showMessage("Atom label font changed for the document", 4000)
                return True
            self.statusBar().showMessage("Select a caption first", 4000)
            return False

        def fn():
            for i in tids:
                t = self.doc.texts[i]
                for k, v in kw.items():
                    setattr(t, k, v)
                if script:
                    t.text = f"<{script}>{t.text}</{script}>"
        self.canvas.edit("Text Style", fn)
        return True

    def _sync_text_style(self) -> None:
        tids = self.selected_texts()
        t = self.doc.texts[tids[0]] if len(tids) == 1 else None
        for key, attr in (("bold", "bold"), ("italic", "italic"), ("underline", "underline"),
                          ("formula", "chemical")):
            act = self.style_actions[key]
            act.blockSignals(True)
            act.setChecked(bool(getattr(t, attr, False)) if t is not None else False)
            act.blockSignals(False)
        if t is not None:
            self.size_box.setCurrentText(f"{(t.size or self.doc.style.caption_size):g}")

    def apply_color(self, color: str) -> None:
        keys = self.doc.with_bonds(self.canvas.selection)
        if not keys:
            self.statusBar().showMessage("Select the objects to color first", 4000)
            return

        def fn():
            for k in keys:
                o = self.doc.get(k)
                if o is not None and hasattr(o, "color"):
                    o.color = color
        self.canvas.edit("Color", fn)

    def choose_color(self) -> None:
        from PySide6.QtWidgets import QColorDialog
        c = QColorDialog.getColor(parent=self, title="Other Color")
        if c.isValid():
            self.apply_color(c.name().upper())

    # --- close ---------------------------------------------------------------------------
    def closeEvent(self, e) -> None:  # noqa: N802
        if self.confirm_close and not self.undo_stack.isClean() and not self.doc.is_empty():
            r = QMessageBox.question(
                self, "Chemical Drawing",
                f"Save changes to '{self.document_name()}' before closing?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard |
                QMessageBox.StandardButton.Cancel)
            if r == QMessageBox.StandardButton.Cancel or \
                    (r == QMessageBox.StandardButton.Save and not self.save()):
                e.ignore()
                return
        self._save_ui()
        from rsstatchem.ui.chem_editor import _WINDOWS
        if self in _WINDOWS:
            _WINDOWS.remove(self)
        e.accept()
        ctx = self.ctx
        if ctx is not None and hasattr(ctx, "window_closed"):
            ctx.window_closed(self)

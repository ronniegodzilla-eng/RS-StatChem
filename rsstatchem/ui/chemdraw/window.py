"""Main window of the Chemical Drawing module, laid out like a ChemDraw Professional document
window: menu bar (File, Edit, View, Object, Structure, Text, Colors, Window, Help), the
General and Style toolbars on top, the vertical Main Toolbar on the left, the document page in
the middle, the Analysis / Chemical Properties windows on the right and the zoom percentage in
the status bar.

The drawing itself (canvas, tools, undo, file formats, exports) is the structure editor of
:mod:`rsstatchem.ui.chem_editor`; this module is the ChemDraw shell around it.
"""
from __future__ import annotations

import math
import os

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QFont, QKeySequence
from PySide6.QtWidgets import (QComboBox, QDialog, QDockWidget, QFileDialog, QFontComboBox,
                               QInputDialog, QLabel, QMessageBox, QTextBrowser, QToolBar,
                               QVBoxLayout)

from rsstatchem.core.chem import molecule as mo
from rsstatchem.core.chem_draw import cdxml, convert, names
from rsstatchem.core.chem_draw.model import Document, load_rsc, save_rsc
from rsstatchem.core.chem_draw.style import PRESETS, preset
from rsstatchem.ui.chem_editor.panels import TemplatePanel
from rsstatchem.ui.chem_editor.window import ChemEditorWindow
from rsstatchem.ui.chemdraw.canvas import ZOOM_STEPS, PageCanvas
from rsstatchem.ui.chemdraw.docks import AnalysisWindow, ChemicalPropertiesWindow
from rsstatchem.ui.chemdraw.icons import cd_icon, color_icon
from rsstatchem.ui.chemdraw.lookup import NameLookup
from rsstatchem.ui.chemdraw.palette import TOOL_TIPS, MainToolbar
from rsstatchem.ui.chemdraw.style import STYLESHEET

__all__ = ["ChemDrawWindow", "OPEN_FILTER", "DEFAULT_STYLE", "COLORS", "HOTKEYS_HTML"]

DEFAULT_STYLE = "ACS Document 1996"
OPEN_FILTER = ("Chemical documents (*.rsc *.cdxml *.mol *.sdf *.sd *.rxn *.smi);;"
               "RS-StatChem drawing (*.rsc);;ChemDraw XML (*.cdxml);;MDL Molfile (*.mol);;"
               "SD file (*.sdf *.sd);;MDL Rxnfile (*.rxn);;SMILES (*.smi *.txt);;All files (*)")
SAVE_FILTERS = {"RS-StatChem drawing (*.rsc)": ".rsc", "ChemDraw XML (*.cdxml)": ".cdxml"}
COLORS = [("Black", "#000000"), ("Red", "#E00000"), ("Orange", "#F07800"),
          ("Yellow", "#E8C800"), ("Green", "#009A00"), ("Cyan", "#00A8C8"),
          ("Blue", "#0000E0"), ("Violet", "#8000C0"), ("Magenta", "#D000A0"),
          ("Brown", "#8B4513"), ("Gray", "#808080")]
FONTS = ["Arial", "Helvetica", "Times New Roman", "Courier New", "Calibri", "Symbol"]
SIZES = [6, 7, 8, 9, 10, 11, 12, 14, 18, 24, 36]
_CHEM_EXT = (".rsc", ".cdxml", ".mol", ".sdf", ".sd", ".rxn", ".smi")

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
<tr><td>w / h</td><td>wedged / hashed wedged</td></tr>
<tr><td>b / d / y</td><td>bold / dashed / wavy</td></tr>
<tr><td>l / r / c</td><td>double bond left / right / centred</td></tr>
<tr><td colspan='2'><b>Anywhere</b></td></tr>
<tr><td>Delete</td><td>delete the selection or the object under the pointer</td></tr>
<tr><td>Arrow keys</td><td>nudge the selection (Shift: 10 pt)</td></tr>
<tr><td>Shift while dragging</td><td>free angle</td></tr>
<tr><td>Ctrl+wheel, Ctrl+] / Ctrl+[</td><td>magnify / reduce</td></tr>
</table>
"""


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
    canvas_class = PageCanvas
    _counter = 0

    def __init__(self, ctx, doc: Document | None = None):
        ChemDrawWindow._counter += 1
        self._number = ChemDrawWindow._counter
        self.ctx = ctx
        super().__init__(None, doc or Document(preset(DEFAULT_STYLE)))
        self.setObjectName("chemdraw_window")
        self.setStyleSheet(STYLESHEET)
        self.display_name = ""
        self.online_lookup = self._setting("chemdraw/pubchem", True)
        self.act_pubchem.setChecked(self.online_lookup)
        self.lookup = NameLookup(self)
        self.lookup.finished.connect(self._lookup_done)
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
        self.act_zoom_in.setText("&Magnify")
        self.act_zoom_in.setShortcuts([QKeySequence("Ctrl+]"),
                                       QKeySequence(QKeySequence.StandardKey.ZoomIn)])
        self.act_zoom_in.setIcon(cd_icon("zoom_in"))
        self.act_zoom_out.setText("&Reduce")
        self.act_zoom_out.setShortcuts([QKeySequence("Ctrl+["),
                                        QKeySequence(QKeySequence.StandardKey.ZoomOut)])
        self.act_zoom_out.setIcon(cd_icon("zoom_out"))
        self.act_actual.setText("&Actual Size")
        self.act_actual.setShortcut(QKeySequence("Ctrl+0"))
        self.act_actual.setIcon(cd_icon("actual"))
        self.act_fit.setText("&Fit to Window")
        self.act_fit.setShortcut(QKeySequence("Ctrl+9"))
        A = self._act
        self.act_print = A("&Print...", self.print_document, QKeySequence.StandardKey.Print)
        self.act_print.setIcon(cd_icon("print"))
        self.act_name2struct = A("Convert &Name to Structure...",
                                 lambda: self.convert_name_to_structure(), "Ctrl+Shift+N")
        self.act_name2struct.setIcon(cd_icon("name2struct"))
        self.act_struct2name = A("Convert Structure to N&ame", self.convert_structure_to_name,
                                 "Ctrl+Alt+Shift+N")
        self.act_struct2name.setIcon(cd_icon("struct2name"))
        self.act_pubchem = A("Use PubChem for Names (online)", self._toggle_pubchem,
                             checkable=True,
                             tip="Look up names and structures not in the offline list at "
                                 "PubChem (sends the InChIKey or the name)")
        self.act_fixed_lengths = A("Fixed &Lengths", self._toggle_fixed, checkable=True)
        self.act_fixed_lengths.setChecked(True)
        self.act_fixed_angles = A("Fixed An&gles", self._toggle_fixed, checkable=True)
        self.act_fixed_angles.setChecked(True)

    def _build_menus(self) -> None:
        mb = self.menuBar()
        # File
        m = mb.addMenu("&File")
        m.addAction(self.act_new)
        m.addAction(self.act_open)
        self.recent_menu = m.addMenu("Open &Recent")
        self.recent_menu.aboutToShow.connect(self._fill_recent)
        m.addSeparator()
        m.addAction(self.act_close)
        m.addAction(self.act_save)
        m.addAction(self.act_save_as)
        m.addSeparator()
        imp = m.addMenu("&Import")
        imp.addAction(self.act_import)
        imp.addAction("Structure &File (MOL, SDF, RXN, CDXML)...", lambda: self.open_file())
        ex = m.addMenu("&Export")
        for text, fmt in (("MDL Molfile (.mol)...", "mol"), ("SD File (.sdf)...", "sdf"),
                          ("MDL Rxnfile (.rxn)...", "rxn"), ("ChemDraw XML (.cdxml)...", "cdxml"),
                          ("SMILES...", "smi"), ("Reaction SMILES...", "rsmi"),
                          ("InChI / InChIKey...", "inchi"), (None, None),
                          ("PNG Image...", "png"), ("SVG Image...", "svg"), ("PDF...", "pdf")):
            if text is None:
                ex.addSeparator()
            else:
                ex.addAction(text, lambda f=fmt: self.export_as(f))
        m.addSeparator()
        m.addAction(self.act_settings)
        ap = m.addMenu("Apply Document Settings &from")
        for name in PRESETS:
            ap.addAction(name, lambda n=name: self.apply_style(preset(n)))
        m.addSeparator()
        m.addAction(self.act_print)
        m.addSeparator()
        m.addAction(self.act_output)
        # Edit
        m = mb.addMenu("&Edit")
        m.addAction(self.act_undo)
        m.addAction(self.act_redo)
        m.addSeparator()
        m.addAction(self.act_cut)
        m.addAction(self.act_copy)
        ca = m.addMenu("Copy &As")
        for text, what in (("MOL Text", "mol"), ("SMILES", "smiles"), ("InChI", "inchi"),
                           ("InChIKey", "inchikey"), ("Reaction SMILES", "rsmi"),
                           ("CDXML Text", "cdxml"), ("Picture (PNG, 300 dpi)", "image"),
                           ("SVG", "svg")):
            ca.addAction(text, lambda w=what: self.copy_as(w))
        m.addAction(self.act_paste)
        ps = m.addMenu("Paste &Special")
        ps.addAction(self.act_import)
        m.addAction(self.act_delete)
        m.addSeparator()
        m.addAction(self.act_select_all)
        # View (filled once the toolbars and windows exist, see _build_docks)
        self.view_menu = mb.addMenu("&View")
        # Object
        m = mb.addMenu("&Object")
        m.addAction(self.act_fixed_lengths)
        m.addAction(self.act_fixed_angles)
        m.addSeparator()
        m.addAction("Bring to &Front", lambda: self.z_order(True), "Ctrl+Shift+]")
        m.addAction("Send to &Back", lambda: self.z_order(False), "Ctrl+Shift+[")
        m.addSeparator()
        m.addAction("&Group", self.group, "Ctrl+G")
        m.addAction("&Ungroup", self.ungroup, "Ctrl+Shift+G")
        m.addSeparator()
        al = m.addMenu("&Align")
        for text, how in (("Lefts", "left"), ("Horizontal Centers", "hcenter"),
                          ("Rights", "right"), ("Tops", "top"), ("Vertical Centers", "vcenter"),
                          ("Bottoms", "bottom")):
            al.addAction(text, lambda h=how: self.align(h))
        di = m.addMenu("&Distribute")
        di.addAction("Horizontally", lambda: self.distribute(True))
        di.addAction("Vertically", lambda: self.distribute(False))
        m.addSeparator()
        m.addAction("Flip &Horizontal", lambda: self.transform("flip_h"), "Ctrl+Shift+H")
        m.addAction("Flip &Vertical", lambda: self.transform("flip_v"), "Ctrl+Shift+V")
        m.addAction("Rotate 90° Clockwise", lambda: self.transform("rot_cw"), "Ctrl+R")
        m.addAction("Rotate 90° Counterclockwise", lambda: self.transform("rot_ccw"),
                    "Ctrl+Shift+R")
        m.addAction("Rotate 180°", self.rotate_180)
        m.addAction("Rotate...", self.rotate_dialog)
        m.addAction("&Scale...", self.scale_dialog)
        m.addSeparator()
        m.addAction("&Center on Page", self.center_on_page)
        # Structure
        m = mb.addMenu("&Structure")
        m.addAction("&Atom Properties...", self._atom_properties_of_selection)
        m.addAction(self.act_expand)
        m.addSeparator()
        m.addAction("Add Atom &Numbers", lambda: self.atom_numbers(True))
        m.addAction("Remove Atom Numbers", lambda: self.atom_numbers(False))
        m.addSeparator()
        m.addAction(self.act_check)
        m.addAction(self.act_clean)
        m.addSeparator()
        m.addAction(self.act_name2struct)
        m.addAction(self.act_struct2name)
        m.addAction(self.act_pubchem)
        m.addSeparator()
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
        # Colors
        m = mb.addMenu("&Colors")
        for name, value in COLORS:
            m.addAction(color_icon(value), name, lambda v=value: self.apply_color(
                "" if v == "#000000" else v))
        m.addAction("&Other...", self.choose_color)
        m.addSeparator()
        m.addAction(self.act_color)
        # Window
        self.windows_menu = mb.addMenu("&Window")
        self.windows_menu.aboutToShow.connect(self._fill_window_menu)
        self._fill_window_menu()
        # Help
        m = mb.addMenu("&Help")
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
                              ("formula", "&Formula", "Ctrl+Alt+F"),
                              ("subscript", "Su&bscript", "Ctrl+-"),
                              ("superscript", "Su&perscript", "Ctrl+="),
                              ("plain", "&Plain", "Ctrl+T")):
            act = QAction(cd_icon(key), text, self)
            act.setShortcut(QKeySequence(sc))
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
            if ext in (".rsc", ".cdxml"):
                doc = load_rsc(path) if ext == ".rsc" else cdxml.load_cdxml(path)
                self.canvas.set_state(doc.to_dict())
                self.path = path
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
        if not self.path or os.path.splitext(self.path)[1].lower() not in (".rsc", ".cdxml"):
            return self.save_as()
        try:
            if self.path.lower().endswith(".cdxml"):
                cdxml.save_cdxml(self.doc, self.path)
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
            if os.path.splitext(path)[1].lower() not in (".rsc", ".cdxml"):
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
        view.setHtml(HOTKEYS_HTML)
        lay = QVBoxLayout(dlg)
        lay.addWidget(view)
        dlg.resize(480, 560)
        dlg.show()          # modeless, like ChemDraw's cheat sheet
        return dlg

    # --- Object menu ---------------------------------------------------------------------
    def rotate_180(self) -> None:
        keys = self.canvas.target_keys()
        self.canvas.edit("Rotate", lambda: self.doc.rotate(keys, math.pi))

    def rotate_dialog(self) -> None:
        ang, ok = QInputDialog.getDouble(self, "Rotate", "Angle (degrees, clockwise):", 15.0,
                                         -360.0, 360.0, 1)
        if ok:
            keys = self.canvas.target_keys()
            self.canvas.edit("Rotate", lambda: self.doc.rotate(keys, math.radians(ang)))

    def scale_dialog(self) -> None:
        pct, ok = QInputDialog.getDouble(self, "Scale", "Scale (%):", 100.0, 5.0, 2000.0, 1)
        if ok and abs(pct - 100.0) > 1e-9:
            self.scale_selection(pct)

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
                if kind in ("text", "plus"):
                    o.x, o.y = cx + (o.x - cx) * k, cy + (o.y - cy) * k
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
        the offline list, else (when allowed) from PubChem in the background."""
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
            hit = names.offline_name(mol)
            if hit is not None:
                self._caption_under(comp, hit.name)
                found.append(hit.name)
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
        structure; otherwise the name is asked for and the structure inserted."""
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
        if not self.online_lookup:
            self._notify("Convert Name to Structure",
                         f"'{name}' is not in the offline name list. Turn on Structure > Use "
                         "PubChem for Names (online) to look it up at PubChem.")
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

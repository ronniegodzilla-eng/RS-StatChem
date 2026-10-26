"""Structure / reaction editor main window (ChemDraw-style): tool palette, canvas, live
Analysis panel, template library, file import/export and the reaction tools."""
from __future__ import annotations

import os

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QAction, QGuiApplication, QKeySequence, QUndoStack
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QDockWidget, QFileDialog,
                               QGridLayout, QLabel, QMainWindow, QMessageBox, QToolBar,
                               QToolButton, QVBoxLayout, QWidget)

from rsstatchem.core.chem import molecule as mo
from rsstatchem.core.chem_draw import abbrev, convert
from rsstatchem.core.chem_draw.model import OBJECT_KINDS, Document, load_rsc, save_rsc
from rsstatchem.core.chem_draw.style import PRESETS, Style, preset
from rsstatchem.ui.chem_editor import export
from rsstatchem.ui.chem_editor.canvas import ChemCanvas
from rsstatchem.ui.chem_editor.icons import tool_icon
from rsstatchem.ui.chem_editor.panels import AnalysisPanel, TemplatePanel

__all__ = ["ChemEditorWindow", "TOOL_GROUPS", "OPEN_FILTER"]

OPEN_FILTER = ("Chemical drawings and structures (*.rsc *.mol *.sdf *.sd *.rxn *.smi *.txt "
               "*.inchi *.mol2 *.pdb);;RS-StatChem drawing (*.rsc);;MOL / SDF (*.mol *.sdf "
               "*.sd);;Reaction (*.rxn);;SMILES (*.smi *.txt);;All files (*)")

TOOL_GROUPS = [
    ("Select", [("select", "Marquee select / move (V)"), ("lasso", "Lasso select"),
                ("rotate", "Rotate selection"), ("eraser", "Eraser (E)")]),
    ("Bonds", [("bond:single", "Single bond: click atom to add, click bond to cycle order"),
               ("bond:double", "Double bond (click again to move the second line)"),
               ("bond:triple", "Triple bond"), ("bond:wedge", "Wedge (stereo, up)"),
               ("bond:hash", "Hashed wedge (stereo, down)"), ("bond:wavy", "Wavy bond"),
               ("bond:bold", "Bold bond"), ("bond:dashed", "Dashed bond"),
               ("bond:aromatic", "Aromatic / delocalised (dashed double) bond"),
               ("chain", "Chain: drag to draw a zig-zag chain")]),
    ("Rings", [("ring:cyclopropane", "Cyclopropane"), ("ring:cyclobutane", "Cyclobutane"),
               ("ring:cyclopentane", "Cyclopentane"), ("ring:cyclohexane", "Cyclohexane"),
               ("ring:cycloheptane", "Cycloheptane"), ("ring:cyclooctane", "Cyclooctane"),
               ("ring:benzene", "Benzene"), ("ring:cyclopentadiene", "Cyclopentadiene")]),
    ("Atoms", [("label", "Atom label: type an element or nickname (OH, Ph, OMe, Boc...)"),
               ("periodic", "Periodic table..."), ("charge+", "Positive charge"),
               ("charge-", "Negative charge"), ("radical", "Radical"),
               ("map", "Atom-to-atom map number")]),
    ("Reaction", [("arrow:forward", "Reaction arrow"), ("arrow:equilibrium", "Equilibrium"),
                  ("arrow:retro", "Retrosynthetic arrow"), ("arrow:resonance", "Resonance"),
                  ("arrow:curved", "Curved arrow (electron pair)"),
                  ("arrow:fishhook", "Fishhook arrow (single electron)"),
                  ("plus", "Plus sign"), ("text", "Text / caption"),
                  ("bracket", "Brackets (drag, or click with a selection)"),
                  ("template", "Insert template (choose one in the Templates panel)")]),
]
QUICK_ELEMENTS = ["C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"]


class ChemEditorWindow(QMainWindow):
    canvas_class = ChemCanvas       # subclasses (the Chemical Drawing workspace) swap it

    def __init__(self, editor=None, doc: Document | None = None, path: str | None = None,
                 reaction: bool = False):
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.editor = editor
        self.doc = doc or Document(preset("New Document"))
        self.path = path
        self.confirm_close = True
        self.reaction_mode = reaction
        self.undo_stack = QUndoStack(self)
        self.canvas = self.canvas_class(self.doc, self.undo_stack, self)
        self.setCentralWidget(self.canvas)
        self._tool_buttons: dict[str, QToolButton] = {}
        self._build_actions()
        self._build_menus()
        self._build_toolbars()
        self._build_docks()
        self.status_tool = QLabel("")
        self.status_problems = QLabel("")
        self.statusBar().addWidget(self.status_tool, 1)
        self.statusBar().addPermanentWidget(self.status_problems)
        self.canvas.status_message.connect(self.status_tool.setText)
        self.canvas.doc_changed.connect(self._doc_changed)
        self.undo_stack.cleanChanged.connect(self._clean_changed)
        self.set_tool("arrow:forward" if reaction else "bond:single")
        self.resize(1180, 780)
        self._update_title()
        self._doc_changed()

    # --- construction --------------------------------------------------------------------
    def _act(self, text, slot, shortcut=None, tip=None, checkable=False) -> QAction:
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        if tip:
            a.setStatusTip(tip)
        a.setCheckable(checkable)
        a.triggered.connect(slot)
        return a

    def _build_actions(self) -> None:
        A = self._act
        self.act_new = A("&New", self.new_window, QKeySequence.StandardKey.New)
        self.act_open = A("&Open...", lambda: self.open_file(), QKeySequence.StandardKey.Open)
        self.act_save = A("&Save", self.save, QKeySequence.StandardKey.Save)
        self.act_save_as = A("Save &As...", self.save_as, QKeySequence.StandardKey.SaveAs)
        self.act_import = A("&Import Structure (SMILES, MOL, InChI, RXN)...",
                            self.import_dialog, "Ctrl+Shift+V")
        self.act_output = A("Send to &Output Viewer", self.send_to_output, "Ctrl+Shift+O",
                            "Structure image and properties to the Output Viewer")
        self.act_data = A("Insert SMILES into &Data Editor...", self.insert_into_data)
        self.act_close = A("&Close", self.close, QKeySequence.StandardKey.Close)
        self.act_undo = self.undo_stack.createUndoAction(self, "&Undo")
        self.act_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.act_redo = self.undo_stack.createRedoAction(self, "&Redo")
        self.act_redo.setShortcut(QKeySequence.StandardKey.Redo)
        self.act_cut = A("Cu&t", self.cut, QKeySequence.StandardKey.Cut)
        self.act_copy = A("&Copy", self.copy, QKeySequence.StandardKey.Copy)
        self.act_paste = A("&Paste", self.paste, QKeySequence.StandardKey.Paste)
        self.act_delete = A("&Delete", self.delete_selection)
        self.act_select_all = A("Select &All", self.canvas.select_all,
                                QKeySequence.StandardKey.SelectAll)
        self.act_zoom_in = A("Zoom &In", lambda: self.canvas.set_zoom(self.canvas.zoom() * 1.25),
                             QKeySequence.StandardKey.ZoomIn)
        self.act_zoom_out = A("Zoom &Out",
                              lambda: self.canvas.set_zoom(self.canvas.zoom() / 1.25),
                              QKeySequence.StandardKey.ZoomOut)
        self.act_fit = A("&Fit to Window", self.canvas.fit, "Ctrl+0")
        self.act_actual = A("&Actual Size", lambda: self.canvas.set_zoom(96 / 72), "Ctrl+1")
        self.act_grid = A("Show &Grid", self._toggle_grid, checkable=True)
        self.act_clean = A("&Clean Structure", self.clean, "Ctrl+Shift+K",
                           "Re-layout the selected (or all) structures")
        self.act_check = A("Check &Structure", self.check_structure, "F5")
        self.act_expand = A("&Expand Labels", self.expand_labels,
                            tip="Replace nicknames (Ph, OMe, Boc...) by their atoms")
        self.act_balance = A("Check Reaction &Balance...", self.check_balance)
        self.act_stoich = A("Reaction &Stoichiometry Table...", self.stoichiometry)
        self.act_settings = A("&Document Settings...", self.document_settings)
        self.act_color = A("&Color by Element", self._toggle_color, checkable=True)

    def _build_menus(self) -> None:
        mb = self.menuBar()
        m = mb.addMenu("&File")
        for a in (self.act_new, self.act_open, None, self.act_save, self.act_save_as, None,
                  self.act_import):
            m.addSeparator() if a is None else m.addAction(a)
        ex = m.addMenu("&Export")
        for text, fmt in (("SMILES...", "smi"), ("MOL File...", "mol"), ("SDF...", "sdf"),
                          ("InChI / InChIKey...", "inchi"), ("Reaction RXN...", "rxn"),
                          ("Reaction SMILES...", "rsmi"), (None, None), ("SVG Image...", "svg"),
                          ("PNG Image...", "png"), ("PDF...", "pdf")):
            if text is None:
                ex.addSeparator()
                continue
            ex.addAction(text, lambda f=fmt: self.export_as(f))
        m.addSeparator()
        m.addAction(self.act_output)
        m.addAction(self.act_data)
        m.addSeparator()
        m.addAction(self.act_close)

        m = mb.addMenu("&Edit")
        for a in (self.act_undo, self.act_redo, None, self.act_cut, self.act_copy):
            m.addSeparator() if a is None else m.addAction(a)
        ca = m.addMenu("Copy &As")
        for text, what in (("SMILES", "smiles"), ("MOL File", "mol"), ("InChI", "inchi"),
                           ("InChIKey", "inchikey"), ("Reaction SMILES", "rsmi"),
                           ("Image (PNG, 300 dpi)", "image"), ("SVG", "svg")):
            ca.addAction(text, lambda w=what: self.copy_as(w))
        for a in (self.act_paste, self.act_import, None, self.act_delete, self.act_select_all):
            m.addSeparator() if a is None else m.addAction(a)

        m = mb.addMenu("&View")
        for a in (self.act_zoom_in, self.act_zoom_out, self.act_fit, self.act_actual, None,
                  self.act_grid):
            m.addSeparator() if a is None else m.addAction(a)
        self.view_menu = m

        m = mb.addMenu("&Object")
        m.addAction("Flip &Horizontal", lambda: self.transform("flip_h"))
        m.addAction("Flip &Vertical", lambda: self.transform("flip_v"))
        m.addAction("Rotate 90° Clockwise", lambda: self.transform("rot_cw"), "Ctrl+R")
        m.addAction("Rotate 90° Counter-clockwise", lambda: self.transform("rot_ccw"),
                    "Ctrl+Shift+R")
        m.addSeparator()
        al = m.addMenu("&Align")
        for text, how in (("Lefts", "left"), ("Horizontal Centers", "hcenter"),
                          ("Rights", "right"), ("Tops", "top"),
                          ("Vertical Centers", "vcenter"), ("Bottoms", "bottom")):
            al.addAction(text, lambda h=how: self.align(h))
        di = m.addMenu("&Distribute")
        di.addAction("Horizontally", lambda: self.distribute(True))
        di.addAction("Vertically", lambda: self.distribute(False))
        m.addSeparator()
        m.addAction("&Group", self.group, "Ctrl+G")
        m.addAction("&Ungroup", self.ungroup, "Ctrl+Shift+G")
        m.addSeparator()
        m.addAction("Bring to &Front", lambda: self.z_order(True))
        m.addAction("Send to &Back", lambda: self.z_order(False))

        m = mb.addMenu("&Structure")
        for a in (self.act_clean, self.act_check, self.act_expand):
            m.addAction(a)
        m.addSeparator()
        m.addAction(self.act_balance)
        m.addAction(self.act_stoich)
        m.addAction("Copy Reaction SMILES", lambda: self.copy_as("rsmi"))

        m = mb.addMenu("&Document")
        m.addAction(self.act_settings)
        for name in PRESETS:
            m.addAction(f"Apply {name} Settings", lambda n=name: self.apply_style(preset(n)))
        m.addSeparator()
        m.addAction(self.act_color)

    def _build_toolbars(self) -> None:
        from rsstatchem.ui.icons import icon
        for act, name in ((self.act_new, "new"), (self.act_open, "open"),
                          (self.act_save, "save"), (self.act_undo, "undo"),
                          (self.act_redo, "redo"), (self.act_output, "output"),
                          (self.act_import, "import")):
            act.setIcon(icon(name))
        tb = QToolBar("Main", self)
        tb.setObjectName("chem_main_toolbar")
        tb.setIconSize(QSize(18, 18))
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        zoom_in = self._act("+", self.act_zoom_in.trigger, tip="Zoom in")
        zoom_out = self._act("−", self.act_zoom_out.trigger, tip="Zoom out")
        fit = self._act("Fit", self.canvas.fit, tip="Fit drawing to window")
        for a in (self.act_new, self.act_open, self.act_save, None, self.act_undo,
                  self.act_redo, None, self.act_cut, self.act_copy, self.act_paste, None,
                  zoom_out, zoom_in, fit, None, self.act_clean, self.act_check):
            tb.addSeparator() if a is None else tb.addAction(a)
        for a in (self.act_new, self.act_open, self.act_save, self.act_undo, self.act_redo):
            tb.widgetForAction(a).setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.act_clean.setIconText("Clean")
        self.act_check.setIconText("Check")
        tb.addSeparator()
        tb.addWidget(QLabel(" Style: "))
        self.style_box = QComboBox()
        self.style_box.addItems(list(PRESETS) + ["Custom"])
        self.style_box.activated.connect(self._style_box_changed)
        tb.addWidget(self.style_box)
        tb.addAction(self.act_color)
        tb.addSeparator()
        tb.addAction(self.act_output)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)

        palette = QToolBar("Tools", self)
        palette.setObjectName("chem_tool_palette")
        palette.setMovable(False)
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(4)
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        for title, tools in TOOL_GROUPS:
            grid = QGridLayout()
            grid.setSpacing(1)
            for k, (tid, tip) in enumerate(tools):
                b = QToolButton()
                b.setIcon(tool_icon(tid))
                b.setIconSize(QSize(26, 26))
                b.setToolTip(tip)
                b.setAutoRaise(True)
                if tid == "periodic":
                    b.clicked.connect(self.periodic_table)
                else:
                    b.setCheckable(True)
                    self.tool_group.addButton(b)
                    b.clicked.connect(lambda _=False, t=tid: self.set_tool(t))
                    self._tool_buttons[tid] = b
                grid.addWidget(b, k // 3, k % 3)
            lay.addLayout(grid)
            line = QLabel()
            line.setFixedHeight(1)
            line.setStyleSheet("background: #D0D0D0;")
            lay.addWidget(line)
        grid = QGridLayout()
        grid.setSpacing(1)
        for k, sym in enumerate(QUICK_ELEMENTS):
            b = QToolButton()
            b.setText(sym)
            b.setFixedSize(QSize(28, 24))
            b.setCheckable(True)
            b.setAutoRaise(True)
            b.setToolTip(f"Atom: click an atom to make it {sym} (hover + key works too)")
            self.tool_group.addButton(b)
            b.clicked.connect(lambda _=False, s=sym: self.set_tool(f"atom:{s}"))
            self._tool_buttons[f"atom:{sym}"] = b
            grid.addWidget(b, k // 3, k % 3)
        lay.addLayout(grid)
        lay.addStretch(1)
        palette.addWidget(w)
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, palette)

    def _build_docks(self) -> None:
        self.analysis = AnalysisPanel(self.canvas)
        self.analysis.paste_requested.connect(self._paste_analysis)
        d = QDockWidget("Analysis", self)
        d.setObjectName("chem_analysis_dock")
        d.setWidget(self.analysis)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, d)
        self.analysis_dock = d
        self.templates = TemplatePanel()
        self.templates.template_chosen.connect(self._template_chosen)
        self.templates.insert_requested.connect(
            lambda smi: self.canvas.insert_text(smi, title="Insert Template"))
        t = QDockWidget("Templates", self)
        t.setObjectName("chem_template_dock")
        t.setWidget(self.templates)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, t)
        self.template_dock = t
        self.resizeDocks([d, t], [330, 330], Qt.Orientation.Horizontal)
        self.resizeDocks([d, t], [380, 300], Qt.Orientation.Vertical)
        self.view_menu.addSeparator()
        self.view_menu.addAction(d.toggleViewAction())
        self.view_menu.addAction(t.toggleViewAction())

    # --- tools ---------------------------------------------------------------------------
    def set_tool(self, tool: str) -> None:
        self.canvas.set_tool(tool)
        b = self._tool_buttons.get(tool)
        if b is not None and not b.isChecked():
            b.setChecked(True)
        tips = {t: tip for _, tools in TOOL_GROUPS for t, tip in tools}
        self.status_tool.setText(tips.get(tool, tool))

    def periodic_table(self) -> None:
        from rsstatchem.ui.chem_editor.dialogs import PeriodicTableDialog
        dlg = PeriodicTableDialog(self)
        if dlg.exec():
            sym = dlg.symbol
            atoms = self.canvas.selected_atoms()
            if atoms:
                from rsstatchem.core.chem_draw import ops
                self.canvas.edit("Change Atom", lambda: [ops.set_element(self.doc, a, sym)
                                                         for a in atoms])
            self.set_tool(f"atom:{sym}")

    def _template_chosen(self, name: str, smiles: str) -> None:
        self.canvas.template_smiles = smiles
        self.set_tool("template")
        self.status_tool.setText(f"Template: {name} — click in the drawing to place it")

    # --- document state ------------------------------------------------------------------
    def _doc_changed(self) -> None:
        n = len(self.canvas.problems)
        self.status_problems.setText(f"⚠ {n} atom(s) with valence problems" if n else "")
        self.status_problems.setStyleSheet("color: #C62828;")
        self.act_color.blockSignals(True)
        self.act_color.setChecked(self.doc.style.color_by_element)
        self.act_color.blockSignals(False)
        name = self.doc.style.name
        idx = self.style_box.findText(name)
        self.style_box.setCurrentIndex(idx if idx >= 0 else self.style_box.count() - 1)
        self._update_title()

    def _clean_changed(self, _clean: bool) -> None:
        try:
            self._update_title()
        except RuntimeError:        # emitted while the window is being destroyed
            pass

    def _update_title(self) -> None:
        name = os.path.basename(self.path) if self.path else "Untitled"
        kind = "Reaction Editor" if self.reaction_mode else "Structure Editor"
        dirty = "" if self.undo_stack.isClean() else " *"
        self.setWindowTitle(f"{name}{dirty} — {kind}")

    def apply_style(self, style: Style) -> None:
        def fn():
            old = self.doc.style.bond_length or style.bond_length
            bb = self.doc.bbox()
            cx, cy = ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2) if bb else (0, 0)
            self.doc.style = style.updated()
            if abs(style.bond_length - old) > 1e-9:
                self.doc.scale(style.bond_length / old, cx, cy)
        self.canvas.edit("Document Settings", fn)

    def _style_box_changed(self, idx: int) -> None:
        name = self.style_box.itemText(idx)
        if name in PRESETS:
            self.apply_style(preset(name))

    def _toggle_color(self, on: bool) -> None:
        self.canvas.edit("Color by Element",
                         lambda: setattr(self.doc.style, "color_by_element", bool(on)))

    def _toggle_grid(self, on: bool) -> None:
        self.canvas.show_grid = bool(on)
        self.canvas.viewport().update()

    def document_settings(self) -> None:
        from rsstatchem.ui.chem_editor.dialogs import DocumentSettingsDialog
        dlg = DocumentSettingsDialog(self.doc.style, self)
        if dlg.exec():
            self.apply_style(dlg.style())

    # --- files ---------------------------------------------------------------------------
    def _settings(self):
        ctx = getattr(self.editor, "ctx", None)
        return getattr(ctx, "settings", None)

    def _start_dir(self) -> str:
        s = self._settings()
        return str(s.value("last_dir", os.path.expanduser("~"))) if s is not None else \
            os.path.expanduser("~")

    def _remember_dir(self, path: str) -> None:
        s = self._settings()
        if s is not None:
            s.setValue("last_dir", os.path.dirname(path))

    def new_window(self) -> None:
        from rsstatchem.ui.chem_editor import open_structure_editor
        open_structure_editor(self.editor)

    def open_file(self, path: str | None = None) -> bool:
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "Open Chemical Drawing",
                                                  self._start_dir(), OPEN_FILTER)
            if not path:
                return False
        self._remember_dir(path)
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".rsc":
                doc = load_rsc(path)
                if self.doc.is_empty() and self.undo_stack.count() == 0:
                    self.canvas.set_state(doc.to_dict())
                    self.path = path
                    self.undo_stack.clear()
                    self._update_title()
                    self.canvas.fit_later()
                else:
                    from rsstatchem.ui.chem_editor import open_structure_editor
                    open_structure_editor(self.editor, path=path)
                return True
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            if ext in (".mol2", ".pdb"):
                mols = mo.read_file(path)
                self.canvas.edit("Import", lambda: [convert.insert_mol(
                    self.doc, m, *self._free_spot()) for m in mols])
            else:
                if ext in (".smi", ".txt") and "\t" in text:
                    text = "\n".join(line.split()[0] for line in text.splitlines()
                                     if line.strip())
                x, y = self._free_spot()
                self.canvas.insert_text(text, x, y, "Import")
            self.canvas.fit_later()
            return True
        except (mo.ChemError, ValueError, OSError) as exc:
            QMessageBox.warning(self, "Open", f"Could not read {os.path.basename(path)}:\n{exc}")
            return False

    def _free_spot(self) -> tuple[float, float]:
        bb = self.doc.bbox()
        if bb is None:
            c = self.canvas.view_center()
            return c.x(), c.y()
        return bb[2] + self.doc.style.bond_length * 4, (bb[1] + bb[3]) / 2

    def save(self) -> bool:
        if not self.path:
            return self.save_as()
        try:
            save_rsc(self.doc, self.path)
        except OSError as exc:
            QMessageBox.warning(self, "Save", str(exc))
            return False
        self.undo_stack.setClean()
        self._update_title()
        return True

    def save_as(self) -> bool:
        start = self.path or os.path.join(self._start_dir(), "drawing.rsc")
        path, _ = QFileDialog.getSaveFileName(self, "Save Chemical Drawing", start,
                                              "RS-StatChem drawing (*.rsc)")
        if not path:
            return False
        if not path.lower().endswith(".rsc"):
            path += ".rsc"
        self.path = path
        self._remember_dir(path)
        return self.save()

    def export_as(self, fmt: str, path: str | None = None, dpi: int | None = None,
                  transparent: bool | None = None) -> bool:
        filters = {"smi": "SMILES (*.smi)", "mol": "MOL file (*.mol)", "sdf": "SDF (*.sdf)",
                   "inchi": "InChI (*.txt)", "rxn": "RXN (*.rxn)",
                   "rsmi": "Reaction SMILES (*.smi)", "svg": "SVG image (*.svg)",
                   "png": "PNG image (*.png)", "pdf": "PDF (*.pdf)"}
        ext = {"inchi": "txt", "rsmi": "smi"}.get(fmt, fmt)
        if path is None:
            base = os.path.splitext(os.path.basename(self.path or "drawing"))[0]
            path, _ = QFileDialog.getSaveFileName(self, "Export",
                                                  os.path.join(self._start_dir(),
                                                               f"{base}.{ext}"), filters[fmt])
            if not path:
                return False
            if fmt == "png" and dpi is None:
                from rsstatchem.ui.chem_editor.dialogs import ImageOptionsDialog
                dlg = ImageOptionsDialog(self)
                if not dlg.exec():
                    return False
                dpi, transparent = dlg.values()
        if not os.path.splitext(path)[1]:
            path += "." + ext
        self._remember_dir(path)
        doc = self._export_doc()
        try:
            if fmt == "svg":
                export.export_svg(doc, path)
            elif fmt == "png":
                export.export_png(doc, path, dpi or 300,
                                  True if transparent is None else transparent)
            elif fmt == "pdf":
                export.export_pdf(doc, path)
            else:
                text = self._text_export(fmt, doc)
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(text)
        except (mo.ChemError, ValueError, OSError) as exc:
            QMessageBox.warning(self, "Export", str(exc))
            return False
        self.statusBar().showMessage(f"Exported {os.path.basename(path)}", 4000)
        return True

    def _export_doc(self) -> Document:
        """Selection only when something is selected, else the whole drawing."""
        if self.canvas.selection:
            return self.doc.subset(self.canvas.selection)
        return self.doc

    def _text_export(self, fmt: str, doc: Document) -> str:
        if fmt == "smi":
            return convert.to_smiles(doc) + "\n"
        if fmt == "mol":
            return convert.to_molblock(doc)
        if fmt == "sdf":
            return convert.to_sdf(doc)
        if fmt == "inchi":
            return convert.to_inchi(doc) + "\n" + convert.to_inchikey(doc) + "\n"
        if fmt == "rxn":
            return convert.reaction_rxn(doc)
        if fmt == "rsmi":
            return convert.reaction_smiles(doc) + "\n"
        raise ValueError(fmt)

    def import_dialog(self) -> None:
        from rsstatchem.ui.chem_editor.dialogs import ImportDialog
        clip = QGuiApplication.clipboard().text()
        dlg = ImportDialog(self, clip if clip and len(clip) < 20000 else "")
        if dlg.exec():
            try:
                x, y = self._free_spot()
                self.canvas.insert_text(dlg.text(), x, y, "Import")
                self.canvas.fit()
            except (mo.ChemError, ValueError) as exc:
                QMessageBox.warning(self, "Import Structure", str(exc))

    def load_text(self, text: str) -> list:
        """Insert SMILES / MOL / InChI / reaction text (used when opened with content)."""
        keys = self.canvas.insert_text(text, 0.0, 0.0, "Insert Structure")
        self.canvas.select_keys(set())
        self.undo_stack.clear()
        self.canvas.fit_later()
        return keys

    # --- clipboard -----------------------------------------------------------------------
    def copy(self) -> None:
        export.copy_to_clipboard(self._export_doc())

    def cut(self) -> None:
        if not self.canvas.selection:
            return
        self.copy()
        self.delete_selection()

    def paste(self) -> None:
        md = QGuiApplication.clipboard().mimeData()
        try:
            if md is not None and md.hasFormat(export.NATIVE_MIME):
                other = Document.from_json(bytes(md.data(export.NATIVE_MIME)).decode("utf-8"))
                self.canvas.paste_document(other)
            elif md is not None and md.hasFormat(export.MOL_MIME):
                self.canvas.insert_text(bytes(md.data(export.MOL_MIME)).decode(), title="Paste")
            elif md is not None and md.hasText() and md.text().strip():
                self.canvas.insert_text(md.text(), title="Paste")
            else:
                self.statusBar().showMessage("Nothing to paste (copy a structure, SMILES or "
                                             "MOL text first)", 4000)
        except (mo.ChemError, ValueError) as exc:
            QMessageBox.warning(self, "Paste", f"The clipboard does not hold a structure:\n{exc}")

    def copy_as(self, what: str) -> None:
        doc = self._export_doc()
        cb = QGuiApplication.clipboard()
        try:
            if what == "image":
                cb.setImage(export.image(doc, 300, transparent=False))
                return
            if what == "svg":
                from PySide6.QtCore import QByteArray, QMimeData
                md = QMimeData()
                svg = export.svg_text(doc)
                md.setData("image/svg+xml", QByteArray(svg))
                md.setText(svg.decode("utf-8"))
                cb.setMimeData(md)
                return
            text = {"smiles": lambda: convert.to_smiles(doc),
                    "mol": lambda: convert.to_molblock(doc),
                    "inchi": lambda: convert.to_inchi(doc),
                    "inchikey": lambda: convert.to_inchikey(doc),
                    "rsmi": lambda: convert.reaction_smiles(doc)}[what]()
            cb.setText(text)
            self.statusBar().showMessage("Copied " + (text if len(text) < 80 else what), 4000)
        except (mo.ChemError, ValueError) as exc:
            QMessageBox.warning(self, "Copy As", str(exc))

    # --- edit / object operations --------------------------------------------------------
    def delete_selection(self) -> None:
        if self.canvas.selection:
            self.canvas.edit("Delete", lambda: self.canvas.erase(
                self.doc.with_bonds(self.canvas.selection)))

    def transform(self, how: str) -> None:
        import math
        keys = self.canvas.target_keys()
        fns = {"flip_h": lambda: self.doc.flip(keys, True),
               "flip_v": lambda: self.doc.flip(keys, False),
               "rot_cw": lambda: self.doc.rotate(keys, math.pi / 2),
               "rot_ccw": lambda: self.doc.rotate(keys, -math.pi / 2)}
        self.canvas.edit({"flip_h": "Flip Horizontal", "flip_v": "Flip Vertical"}.get(
            how, "Rotate"), fns[how])

    def _units(self) -> list[tuple[set, tuple]]:
        """Selected objects as units (each molecule or object) with their boxes."""
        sel = self.doc.with_bonds(self.canvas.selection)
        atoms = self.doc.atoms_of(sel)
        units = []
        for comp in self.doc.fragments():
            part = [a for a in comp if a in atoms]
            if part:
                keys = self.doc.with_bonds({("atom", a) for a in part})
                units.append(keys)
        for k in sel:
            if k[0] in OBJECT_KINDS:
                units.append({k})
        out = []
        for keys in units:
            boxes = []
            for k, i in keys:
                if k == "atom":
                    r = self.canvas.info.label_boxes.get(i)
                    a = self.doc.atoms[i]
                    boxes.append((r.left(), r.top(), r.right(), r.bottom()) if r else
                                 (a.x, a.y, a.x, a.y))
                elif k == "text" and i in self.canvas.info.text_boxes:
                    r = self.canvas.info.text_boxes[i]
                    boxes.append((r.left(), r.top(), r.right(), r.bottom()))
                else:
                    bb = self.doc.bbox([(k, i)])
                    if bb:
                        boxes.append(bb)
            if boxes:
                out.append((keys, (min(b[0] for b in boxes), min(b[1] for b in boxes),
                                   max(b[2] for b in boxes), max(b[3] for b in boxes))))
        return out

    def align(self, how: str) -> None:
        units = self._units()
        if len(units) < 2:
            self.statusBar().showMessage("Select two or more objects to align", 3000)
            return
        boxes = [b for _, b in units]
        ref = {"left": min(b[0] for b in boxes), "right": max(b[2] for b in boxes),
               "top": min(b[1] for b in boxes), "bottom": max(b[3] for b in boxes),
               "hcenter": sum((b[0] + b[2]) / 2 for b in boxes) / len(boxes),
               "vcenter": sum((b[1] + b[3]) / 2 for b in boxes) / len(boxes)}[how]

        def fn():
            for keys, b in units:
                dx = dy = 0.0
                if how == "left":
                    dx = ref - b[0]
                elif how == "right":
                    dx = ref - b[2]
                elif how == "hcenter":
                    dx = ref - (b[0] + b[2]) / 2
                elif how == "top":
                    dy = ref - b[1]
                elif how == "bottom":
                    dy = ref - b[3]
                else:
                    dy = ref - (b[1] + b[3]) / 2
                self.doc.move(keys, dx, dy)
        self.canvas.edit("Align", fn)

    def distribute(self, horizontal: bool) -> None:
        units = self._units()
        if len(units) < 3:
            self.statusBar().showMessage("Select three or more objects to distribute", 3000)
            return
        idx = 0 if horizontal else 1
        units.sort(key=lambda u: (u[1][idx] + u[1][idx + 2]) / 2)
        c0 = (units[0][1][idx] + units[0][1][idx + 2]) / 2
        c1 = (units[-1][1][idx] + units[-1][1][idx + 2]) / 2
        step = (c1 - c0) / (len(units) - 1)

        def fn():
            for k, (keys, b) in enumerate(units):
                d = c0 + k * step - (b[idx] + b[idx + 2]) / 2
                self.doc.move(keys, d if horizontal else 0.0, 0.0 if horizontal else d)
        self.canvas.edit("Distribute", fn)

    def group(self) -> None:
        sel = self.doc.with_bonds(self.canvas.selection)
        if len(sel) < 2:
            return

        def fn():
            g = self.doc.next_group()
            for k in sel:
                self.doc.get(k).group = g
        self.canvas.edit("Group", fn)

    def ungroup(self) -> None:
        sel = self.doc.with_bonds(self.canvas.selection)
        self.canvas.edit("Ungroup", lambda: [setattr(self.doc.get(k), "group", 0)
                                             for k in sel])

    def z_order(self, front: bool) -> None:
        sel = [k for k in self.canvas.selection if k[0] in OBJECT_KINDS]
        if not sel:
            return

        def fn():
            objs = [o for kind in OBJECT_KINDS for o in self.doc.store(kind).values()]
            z = max(o.z for o in objs) + 1 if front else min(o.z for o in objs) - 1
            for k in sel:
                self.doc.get(k).z = z
        self.canvas.edit("Bring to Front" if front else "Send to Back", fn)

    # --- chemistry -----------------------------------------------------------------------
    def clean(self) -> None:
        if not self.canvas.clean():
            self.statusBar().showMessage("Nothing to clean", 3000)

    def check_structure(self) -> None:
        probs = self.canvas.problems
        if not probs:
            QMessageBox.information(self, "Check Structure",
                                    "No valence problems were found.")
            return
        lines = []
        for aid, msg in probs.items():
            a = self.doc.atoms.get(aid)
            lines.append(f"{a.label or a.symbol if a else '?'}: {msg}")
        self.canvas.select_keys({("atom", a) for a in probs if a in self.doc.atoms})
        QMessageBox.warning(self, "Check Structure", "\n".join(lines[:20]))

    def expand_labels(self) -> None:
        atoms = self.canvas.selected_atoms() or set(self.doc.atoms)
        targets = [a for a in atoms if self.doc.atoms[a].label and
                   abbrev.is_abbreviation(self.doc.atoms[a].label)]
        if targets:
            self.canvas.edit("Expand Labels",
                             lambda: [convert.expand_label(self.doc, a) for a in targets])

    def check_balance(self) -> None:
        from rsstatchem.ui.chem_editor.dialogs import BalanceDialog
        BalanceDialog(self.doc, self, send=self._send_section).exec()

    def stoichiometry(self) -> None:
        from rsstatchem.ui.chem_editor.dialogs import StoichiometryDialog
        try:
            dlg = StoichiometryDialog(self.doc, self, send=self._send_section)
        except (ValueError, mo.ChemError) as exc:
            QMessageBox.information(self, "Reaction Stoichiometry", str(exc))
            return
        dlg.exec()

    def _paste_analysis(self, text: str) -> None:
        if not text:
            return
        bb = self.doc.bbox()
        x = bb[0] if bb else 0.0
        y = (bb[3] + self.doc.style.bond_length) if bb else 0.0
        self.canvas.edit("Paste Analysis", lambda: self.doc.add_text(x, y, text,
                                                                     chemical=False))

    # --- integration ---------------------------------------------------------------------
    def _send_section(self, section) -> bool:
        ctx = getattr(self.editor, "ctx", None)
        if ctx is None or not hasattr(ctx, "show_output"):
            QMessageBox.information(self, "Output Viewer",
                                    "Open the editor from the Data Editor's Chemistry menu "
                                    "to send results to the Output Viewer.")
            return False
        ctx.show_output(section)
        return True

    def output_section(self):
        return export.output_section(self._export_doc())

    def send_to_output(self) -> bool:
        if self.doc.is_empty():
            QMessageBox.information(self, "Output Viewer", "The drawing is empty.")
            return False
        return self._send_section(self.output_section())

    def insert_into_data(self) -> bool:
        from rsstatchem.ui.chem_editor.dialogs import DataInsertDialog
        ed = self.editor
        if ed is None or not hasattr(ed, "doc"):
            QMessageBox.information(self, "Data Editor", "No Data Editor is attached.")
            return False
        try:
            smiles = convert.to_smiles(self._export_doc())
        except mo.ChemError as exc:
            QMessageBox.warning(self, "Data Editor", str(exc))
            return False
        row = 0
        view = getattr(ed, "data_view", None)
        if view is not None and view.currentIndex().isValid():
            row = view.currentIndex().row()
        dlg = DataInsertDialog(ed.ds, smiles, row, self)
        if not dlg.exec():
            return False
        return self.write_smiles_to_data(*dlg.target())

    def write_smiles_to_data(self, name: str, new: bool, row: int, smiles: str) -> bool:
        ed = self.editor

        def change(ds):
            if new and not ds.has_name(name):
                ds.add_variable(name, "string")
            ds.set_value(row, ds.index_of(name), smiles)
        try:
            ed.doc.mutate(change, "Insert SMILES")
        except Exception as exc:  # noqa: BLE001 - report invalid names etc.
            QMessageBox.warning(self, "Data Editor", str(exc))
            return False
        return True

    def closeEvent(self, e) -> None:  # noqa: N802
        if self.confirm_close and not self.undo_stack.isClean() and not self.doc.is_empty():
            r = QMessageBox.question(
                self, "Structure Editor", "Save changes to the drawing?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard |
                QMessageBox.StandardButton.Cancel)
            if r == QMessageBox.StandardButton.Cancel or \
                    (r == QMessageBox.StandardButton.Save and not self.save()):
                e.ignore()
                return
        from rsstatchem.ui.chem_editor import _WINDOWS
        if self in _WINDOWS:
            _WINDOWS.remove(self)
        e.accept()

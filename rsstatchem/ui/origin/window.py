"""The Graph & Analysis main window, laid out like OriginLab Origin.

Menu bar (context-sensitive like Origin's: worksheet menus when a workbook is active, Graph /
Insert when a graph is active, Matrix when a matrix book is active), the Standard, Column and
Format toolbars on top, the Tools toolbar on the left, the 2D Graphs toolbar at the bottom,
Project Explorer (left), Object Manager (right), Results Log and Messages Log (bottom) and an
MDI workspace with Workbook, Matrix, Graph and Notes windows. A project
(:class:`~rsstatchem.ui.origin.model.Project`) is saved as one ``.rsop`` file.

Selecting a block of cells (e.g. rows 3–10 of B:C) makes plots and fits use only those rows
(``[Book1]Sheet1!B[3:10]``). Analysis reports remember how they were made; when their source
data change they are recalculated (Auto) or flagged with a yellow lock (Manual); the same
holds for the output columns of Statistics on Rows and Frequency Counts (a lock in their
column headers, Recalculate in the column menu). Report graphs are live: double-clicking
one opens a graph window that follows the report's recalculations.
"""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QBrush, QColor, QKeySequence
from PySide6.QtWidgets import (QApplication, QDockWidget, QFileDialog, QInputDialog, QLabel,
                               QMainWindow, QMdiArea, QMdiSubWindow, QMenu, QMessageBox,
                               QPlainTextEdit, QStackedWidget, QToolBar, QToolButton,
                               QTreeWidget, QTreeWidgetItem)

from rsstatchem import __version__
from rsstatchem.core.analysis import matrix as mx
from rsstatchem.core.graph.export import EXTENSION as GRAPH_EXT
from rsstatchem.core.graph.export import load_graph
from rsstatchem.core.graph.model import Annotation, Graph, Layer
from rsstatchem.core.graph.themes import THEMES, apply_theme
from rsstatchem.ui.modules import add_module_menu
from rsstatchem.ui.origin import analysis as an
from rsstatchem.ui.origin import formula as fx
from rsstatchem.ui.origin import recalc
from rsstatchem.ui.origin import report_graphs as rg
from rsstatchem.ui.origin import stat_dialogs as sd
from rsstatchem.ui.origin import stat_tests as st
from rsstatchem.ui.origin import style
from rsstatchem.ui.origin.dialogs import (AddPlotDialog, ColumnPropertiesDialog, FitDialog,
                                          InputChoice, MatrixDimensionsDialog,
                                          MatrixValuesDialog, ParamDialog, SetValuesDialog)
from rsstatchem.ui.origin.explorer import ProjectExplorer
from rsstatchem.ui.origin.format_bar import FormatToolbar
from rsstatchem.ui.origin.graph_view import OriginGraphWindow
from rsstatchem.ui.origin.icons import icon
from rsstatchem.ui.origin.matrix_view import MatrixWindow
from rsstatchem.ui.origin.model import (DEFAULT_ROWS, DESIGNATIONS, EXTENSION, Folder, GraphDoc,
                                        MatrixBook, MatrixSheet, Notes, Project, ReportSheet,
                                        SheetOp, Workbook, Worksheet, load_project, range_text,
                                        rename_refs, rows_text, save_project)
from rsstatchem.ui.origin.plotting import (MATRIX_PLOTS, ORIGIN_THEME, PLOT_TYPES, PlotError,
                                           add_xy_plot, column_key, plot_columns, plot_matrix,
                                           refresh_links, rekey, update_data_ranges)
from rsstatchem.ui.origin.sheet_view import BookWindow

APP_TITLE = "RS-StatChem Graph & Analysis"
PROJECT_FILTER = f"Origin-style Project (*{EXTENSION});;Graph (*{GRAPH_EXT});;All files (*)"
FIT_COLOR = "#FF0000"


class _SubWindow(QMdiSubWindow):
    """MDI child; closing asks Delete / Hide like Origin (unless the project is closing)."""

    def __init__(self, main: "OriginMainWindow", obj):
        super().__init__()
        self.main = main
        self.obj = obj
        self._styled = False
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if not self._styled:   # after polishing, or the app palette wins
            self._styled = True
            self.setPalette(self.main._sub_palette)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.main.closing_all:
            event.accept()
            return
        event.ignore()
        QTimer.singleShot(0, lambda: self.main.ask_close_window(self.obj.name))


class _NotesEdit(QPlainTextEdit):
    def __init__(self, notes: Notes, main):
        super().__init__(notes.text)
        self.notes = notes
        self.main = main
        self.textChanged.connect(self._changed)

    def _changed(self) -> None:
        self.notes.text = self.toPlainText()
        if self.main is not None:
            self.main.set_modified(True)


class OriginMainWindow(QMainWindow):
    def __init__(self, ctx, project: Project | None = None):
        super().__init__()
        self.ctx = ctx
        self.project = project or Project()
        self.modified = False
        self.closing_all = False
        self.quiet = False   # report problems in the Messages Log only (no message boxes)
        self.subs: dict[str, _SubWindow] = {}      # window name -> MDI subwindow
        self._graph_trees: dict[int, QTreeWidget] = {}   # id(graph window) -> object tree
        self.setObjectName("originMain")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setStyleSheet(style.STYLESHEET)
        self.setPalette(style.palette(self.palette()))
        self.setDockOptions(QMainWindow.DockOption.AllowTabbedDocks
                            | QMainWindow.DockOption.AnimatedDocks)
        self.resize(1360, 860)
        self.setWindowIcon(icon("e_graph"))
        self.setAcceptDrops(True)

        # MDI title bars: Fusion paints the active one in the Highlight colour
        self._sub_palette = self.palette()
        self._sub_palette.setColor(self._sub_palette.ColorRole.Highlight, QColor("#4a7cc0"))
        self._sub_palette.setColor(self._sub_palette.ColorRole.HighlightedText, QColor("#ffffff"))
        self.mdi = QMdiArea()
        self.mdi.setObjectName("originWorkspace")
        self.mdi.setBackground(QBrush(QColor(style.MDI_BACKGROUND)))
        self.mdi.setActivationOrder(QMdiArea.WindowOrder.ActivationHistoryOrder)
        self.mdi.subWindowActivated.connect(self._activated)
        self.setCentralWidget(self.mdi)

        self._build_docks()
        self._build_actions()
        self._build_menus()
        self._build_toolbars()
        self._build_status()
        self.explorer.set_project(self.project)
        self._activated(None)
        self._update_title()
        state = self._setting("window_state")
        if state is not None:   # docks / toolbars as the user left them
            try:
                self.restoreState(state)
            except TypeError:
                pass

    # ================================================================== docks
    def _build_docks(self) -> None:
        self.explorer = ProjectExplorer(self)
        self.explorer.folder_changed.connect(self._folder_changed)
        self.explorer.window_activated.connect(self.show_window)
        self.pe_dock = self._dock("Project Explorer", "projectExplorer", self.explorer,
                                  Qt.DockWidgetArea.LeftDockWidgetArea)
        self.pe_dock.setMinimumWidth(230)

        self.obj_stack = QStackedWidget()
        self.book_tree = QTreeWidget()
        self.book_tree.setObjectName("objectTree")
        self.book_tree.setHeaderHidden(True)
        self.book_tree.itemClicked.connect(self._book_tree_clicked)
        self.obj_stack.addWidget(self.book_tree)
        self.om_dock = self._dock("Object Manager", "objectManager", self.obj_stack,
                                  Qt.DockWidgetArea.RightDockWidgetArea)
        self.om_dock.setMinimumWidth(200)

        self.results = QPlainTextEdit()
        self.results.setObjectName("resultsLog")
        self.results.setReadOnly(True)
        self.results.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.results_dock = self._dock("Results Log", "resultsLog", self.results,
                                       Qt.DockWidgetArea.BottomDockWidgetArea)
        self.messages = QPlainTextEdit()
        self.messages.setObjectName("messagesLog")
        self.messages.setReadOnly(True)
        self.messages_dock = self._dock("Messages Log", "messagesLog", self.messages,
                                        Qt.DockWidgetArea.BottomDockWidgetArea)
        self.tabifyDockWidget(self.results_dock, self.messages_dock)
        self.results_dock.raise_()
        self.resizeDocks([self.results_dock], [150], Qt.Orientation.Vertical)
        self.resizeDocks([self.pe_dock, self.om_dock], [250, 210], Qt.Orientation.Horizontal)

    def _dock(self, title, name, widget, area) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(name)
        dock.setWidget(widget)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable
                         | QDockWidget.DockWidgetFeature.DockWidgetMovable
                         | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.addDockWidget(area, dock)
        return dock

    # ================================================================== actions
    def _act(self, text, slot, shortcut=None, icon_name=None, tip=None,
             checkable=False) -> QAction:
        act = QAction(text, self)
        if icon_name:
            act.setIcon(icon(icon_name))
        if shortcut:
            act.setShortcut(QKeySequence(shortcut))
            act.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            self.addAction(act)   # shortcuts work while the menu holding it is not shown
        act.setCheckable(checkable)
        if tip:
            act.setStatusTip(tip)
            act.setToolTip(tip)
        act.triggered.connect(lambda checked=False: slot())
        return act

    def _build_actions(self) -> None:
        a = self._act
        self.act_new_project = a("&Project", self.new_project, "Ctrl+N", "o_new_project",
                                 "New Project")
        self.act_new_book = a("&Workbook", self.new_book, None, "o_new_book", "New Workbook")
        self.act_new_graph = a("&Graph", self.new_graph, None, "o_new_graph", "New Graph")
        self.act_new_notes = a("&Notes", self.new_notes, None, "o_new_notes", "New Notes")
        self.act_new_matrix = a("&Matrix", self.new_matrix, None, "o_new_matrix",
                                "New Matrix")
        self.act_new_folder = a("&Folder", self.new_folder, None, "o_new_folder", "New Folder")
        self.act_open = a("&Open...", self.open_dialog, "Ctrl+O", "o_open",
                          "Open a project (.rsop) or graph (.rsg)")
        self.act_save = a("&Save Project", self.save_project, "Ctrl+S", "o_save",
                          "Save Project")
        self.act_save_as = a("Save Project &As...", self.save_project_as, None, None)
        self.act_import_csv = a("Single &ASCII / CSV...", self.import_csv_dialog, "Ctrl+K",
                                "o_import", "Import Single ASCII / CSV")
        self.act_import_excel = a("&Excel (XLS, XLSX)...", self.import_excel_dialog, None,
                                  "o_import_excel", "Import Excel")
        self.act_export_graph = a("&Graphs...", self.export_graph, "Ctrl+G", "o_export",
                                  "Export Graph (PNG, TIFF, SVG, PDF, EPS)")
        self.act_export_ascii = a("&ASCII / CSV...", self.export_ascii, None, None)
        self.act_close_project = a("&Close Project", self.close_project, None, None)
        self.act_exit = a("E&xit", self.close, None, None)

        self.act_undo = a("&Undo", self.undo, "Ctrl+Z", "o_undo", "Undo")
        self.act_redo = a("&Redo", self.redo, "Ctrl+Y", "o_redo", "Redo")
        self.act_cut = a("Cu&t", self.cut, "Ctrl+X", None)
        self.act_copy = a("&Copy", self.copy, "Ctrl+C", None)
        self.act_copy_headers = a("Copy (include &Long Name, Units, Comments)",
                                  lambda: self.copy(with_headers=True), None, None)
        self.act_paste = a("&Paste", self.paste, "Ctrl+V", None)
        self.act_clear = a("Clear", self.clear_cells, None, None)
        self.act_delete = a("&Delete", self.delete_selected, "Del", None)
        self.act_select_all = a("Select &All", self.select_all, "Ctrl+A", None)
        self.act_copy_page = a("Copy &Page", self.copy_page, "Ctrl+J", "o_copy_page",
                               "Copy Graph Page (image)")
        self.act_rename = a("Re&name Window...", lambda: self.rename_window(None), "F2", None)
        self.act_duplicate = a("D&uplicate", lambda: self.duplicate_window(None), None,
                               "o_duplicate", "Duplicate the active window")
        self.act_refresh = a("Refres&h", self.refresh_all, "F5", "o_refresh",
                             "Refresh formulas and graphs")

        self.act_view_pe = self.pe_dock.toggleViewAction()
        self.act_view_pe.setText("&Project Explorer")
        self.act_view_pe.setShortcut(QKeySequence("Alt+1"))
        self.act_view_pe.setIcon(icon("o_explorer"))
        self.act_view_results = self.results_dock.toggleViewAction()
        self.act_view_results.setText("&Results Log")
        self.act_view_results.setShortcut(QKeySequence("Alt+2"))
        self.act_view_results.setIcon(icon("o_results"))
        self.act_view_messages = self.messages_dock.toggleViewAction()
        self.act_view_messages.setText("&Messages Log")
        self.act_view_messages.setShortcut(QKeySequence("Alt+6"))
        self.act_view_messages.setIcon(icon("o_messages"))
        self.act_view_om = self.om_dock.toggleViewAction()
        self.act_view_om.setText("&Object Manager")
        self.act_view_om.setShortcut(QKeySequence("Alt+8"))
        self.act_view_om.setIcon(icon("o_objects"))
        for act in (self.act_view_pe, self.act_view_results, self.act_view_messages,
                    self.act_view_om):
            self.addAction(act)
        self.act_folder_only = a("Show Windows in Active &Folder Only", self._folder_filter,
                                 None, None, checkable=True)
        self.act_folder_only.setChecked(self._setting("folder_only", "true") == "true")
        self.act_full_screen = a("&Full Screen", self._full_screen, "Ctrl+Shift+F", None)
        self.act_sparklines = a("&Sparklines", self.toggle_sparklines, None, None,
                                "Show a tiny line plot of each column in the header",
                                checkable=True)
        self.act_image_mode = a("&Image Mode", self.toggle_image_mode, None, "m_image",
                                "Show the matrix as an image", checkable=True)
        self.act_show_xy = a("Show &X/Y", self.toggle_show_xy, None, None,
                             "Show the X/Y coordinates in the matrix headers", checkable=True)

        # Column menu
        self.act_desig = {}
        names = {"X": ("o_set_x", None), "Y": ("o_set_y", None),
                 "Z": ("o_set_z", None), "xEr": (None, None),
                 "yEr": ("o_set_yerr", None), "L": ("o_set_label", None),
                 "N": ("o_set_disregard", None)}
        for code, (text, _) in DESIGNATIONS.items():
            ic, sc = names[code]
            self.act_desig[code] = a(text, lambda c=code: self.set_designation(c), sc, ic,
                                     f"Set As {text}")
        self.act_set_values = a("Set Column &Values...", self.set_column_values_dialog,
                                "Ctrl+Q", "o_set_values", "Set Column Values (formula)")
        self.act_add_column = a("&Add New Columns", self.add_columns, "Ctrl+D",
                                "o_add_column", "Add New Column")
        self.act_add_columns_n = a("Add New Columns (&N)...", self.add_columns_dialog, None,
                                   None)
        self.act_insert_column = a("&Insert", self.insert_column, None, None)
        self.act_delete_column = a("Delete Columns", self.delete_columns, None, None)
        self.act_column_props = a("Column &Properties...", lambda: self.column_properties(None),
                                  "Ctrl+Shift+O", None)
        self.act_fill_rows = a("&Row Numbers", lambda: self.fill_columns("rows"))
        self.act_fill_uniform = a("&Uniform Random Numbers", lambda: self.fill_columns("uniform"))
        self.act_fill_normal = a("&Normal Random Numbers", lambda: self.fill_columns("normal"))
        self.act_sort_asc = a("&Ascending", lambda: self.sort_columns(False), None, "o_sort")
        self.act_sort_desc = a("&Descending", lambda: self.sort_columns(True))
        self.act_move_left = a("Move &Left", lambda: self.move_column(-1))
        self.act_move_right = a("Move &Right", lambda: self.move_column(1))

        # Worksheet menu
        self.act_clear_ws = a("&Clear Worksheet", self.clear_worksheet)
        self.act_sort_ws_asc = a("&Ascending", lambda: self.sort_worksheet(False))
        self.act_sort_ws_desc = a("&Descending", lambda: self.sort_worksheet(True))
        self.act_transpose = a("&Transpose", self.transpose_worksheet)
        self.act_add_sheet = a("&Add Sheet", self.add_sheet)
        self.act_rename_sheet = a("&Rename Sheet...", self.rename_sheet)
        self.act_delete_sheet = a("&Delete Sheet", self.delete_sheet)
        self.act_duplicate_sheet = a("D&uplicate Sheet", self.duplicate_sheet)
        self.act_set_rows = a("Set Number of &Rows...", self.set_rows_dialog)
        self.act_to_matrix_direct = a("&Direct...", self.convert_to_matrix_direct)
        self.act_to_matrix_xyz = a("&XYZ Gridding...", self.convert_to_matrix_xyz)

        # Matrix menu
        self.act_m_values = a("Set &Values...", self.matrix_values_dialog, None,
                              "o_set_values", "Set Matrix Values")
        self.act_m_dims = a("Set &Dimensions/Labels...", self.matrix_dimensions_dialog)
        self.act_m_transpose = a("&Transpose", lambda: self.matrix_geometry("transpose"),
                                 None, "m_transpose", "Transpose")
        self.act_m_flip_h = a("Flip &Horizontal", lambda: self.matrix_geometry("flip_h"))
        self.act_m_flip_v = a("Flip &Vertical", lambda: self.matrix_geometry("flip_v"))
        self.act_m_rot90 = a("Rotate &90 (counter-clockwise)",
                             lambda: self.matrix_geometry("rot90"))
        self.act_m_rot270 = a("Rotate -9&0 (clockwise)",
                              lambda: self.matrix_geometry("rot270"))
        self.act_m_rot180 = a("Rotate &180", lambda: self.matrix_geometry("rot180"))
        self.act_m_expand = a("&Expand...", lambda: self.matrix_resample_dialog("expand"))
        self.act_m_shrink = a("&Shrink...", lambda: self.matrix_resample_dialog("shrink"))
        self.act_m_to_ws = a("Convert to &Worksheet (XYZ)", self.matrix_to_worksheet)
        self.act_m_add_sheet = a("&Add Sheet", self.matrix_add_sheet)

        # Graph menu / Tools toolbar
        self.act_add_plot = a("&Add Plot to Layer...", self.add_plot_active, None, "add_plot",
                              "Add Plot to Layer")
        self.act_rescale = a("&Rescale to Show All", self.rescale, "Ctrl+R", "rescale",
                             "Rescale to Show All")
        self.act_panel_below = a("New Layer &Below (stack)", lambda: self._gw_call(
            "add_layer", "below"))
        self.act_panel_right = a("New Layer &Right (side by side)", lambda: self._gw_call(
            "add_layer", "right"))
        self.act_inset = a("&Inset Layer", lambda: self._gw_call("add_layer", "inset"))
        self.act_remove_layer = a("Remove &Layer", lambda: self._gw_call("remove_layer"))
        self.act_right_y = a("Show &Right Y Axis", lambda: self._gw_call("toggle_right_y"))
        self.act_page = a("&Page Properties...", lambda: self._gw_call("page_setup"))
        self.act_legend = a("&Legend", self.toggle_legend)

        group = QActionGroup(self)
        self.tool_acts = {}
        for key, text, ic in (("pointer", "Pointer", "pointer"), ("zoom", "Zoom", "zoom"),
                              ("pan", "Scroll / Pan", "pan"),
                              ("reader", "Data Reader", "t_reader"),
                              ("text", "Text Tool", "text"), ("arrow", "Arrow Tool", "arrow"),
                              ("line", "Line Tool", "line"), ("rect", "Rectangle Tool", "rect")):
            act = a(text, lambda k=key: self.set_tool(k), None, ic, text, checkable=True)
            group.addAction(act)
            self.tool_acts[key] = act
        self.tool_acts["pointer"].setChecked(True)

        # Analysis / Statistics
        self.act_fit_linear = a("&Linear Fit...", lambda: self.fit_dialog("linear"), None,
                                "o_fit", "Linear Fit")
        self.act_fit_poly = a("&Polynomial Fit...", lambda: self.fit_dialog("poly"))
        self.act_fit_nl = a("&Nonlinear Curve Fit...", lambda: self.fit_dialog("nlfit"))
        self.act_stats_cols = a("Statistics on &Columns...", lambda: self.stat_dialog("desc"),
                                None, "o_stats", "Statistics on Columns")

    def _setting(self, key: str, default=None):
        s = getattr(self.ctx, "settings", None)
        return default if s is None else s.value(f"origin/{key}", default)

    def _set_setting(self, key: str, value) -> None:
        s = getattr(self.ctx, "settings", None)
        if s is not None:
            s.setValue(f"origin/{key}", value)

    # ================================================================== menus
    def _build_menus(self) -> None:
        self.menus: dict[str, QMenu] = {}

        def menu(key, title):
            m = QMenu(title, self)
            self.menus[key] = m
            return m

        f = menu("file", "&File")
        new = f.addMenu(icon("o_new_project"), "&New")
        new.addActions([self.act_new_project, self.act_new_book, self.act_new_matrix,
                        self.act_new_graph, self.act_new_notes])
        new.addSeparator()
        new.addAction(self.act_new_folder)
        f.addAction(self.act_open)
        self.recent_menu = f.addMenu("Recent &Projects")
        self.recent_menu.aboutToShow.connect(self._fill_recent)
        f.addAction(self.act_close_project)
        f.addSeparator()
        f.addActions([self.act_save, self.act_save_as])
        f.addSeparator()
        imp = f.addMenu(icon("o_import"), "&Import")
        imp.addActions([self.act_import_csv, self.act_import_excel])
        exp = f.addMenu("&Export")
        exp.addActions([self.act_export_ascii, self.act_export_graph])
        f.addSeparator()
        f.addAction(self.act_exit)

        e = menu("edit", "&Edit")
        e.addActions([self.act_undo, self.act_redo])
        e.addSeparator()
        e.addActions([self.act_cut, self.act_copy, self.act_copy_headers, self.act_paste,
                      self.act_clear, self.act_delete])
        e.addSeparator()
        e.addAction(self.act_select_all)
        e.addSeparator()
        e.addActions([self.act_copy_page, self.act_duplicate, self.act_rename,
                      self.act_refresh])

        v = menu("view", "&View")
        tb_menu = v.addMenu("&Toolbars")
        self.toolbar_menu = tb_menu
        v.addSeparator()
        v.addActions([self.act_view_pe, self.act_view_results, self.act_view_messages,
                      self.act_view_om])
        v.addSeparator()
        v.addAction(self.act_folder_only)
        v.addAction(self.act_full_screen)
        v.addSeparator()
        v.addActions([self.act_sparklines, self.act_image_mode, self.act_show_xy])
        v.addSeparator()
        v.addAction(self._act("Zoom &In", lambda: self._gw_zoom(1.25), "Ctrl+="))
        v.addAction(self._act("Zoom &Out", lambda: self._gw_zoom(0.8), "Ctrl+-"))
        v.addAction(self._act("&Whole Page", lambda: self._gw_call("fit_page"), "Ctrl+W"))

        p = menu("plot", "&Plot")
        basic = p.addMenu(icon("g_line"), "&Basic 2D")
        for k in ("line", "scatter", "line_symbol", "yerror", "bubble", "color_scatter"):
            basic.addAction(self._plot_action(k))
        bars = p.addMenu(icon("g_column"), "Bar, &Pie, Area")
        for k in ("column", "column_stacked", "column_percent", None, "bar", "bar_stacked",
                  None, "pie", "doughnut", None, "area", "area_stacked"):
            bars.addSeparator() if k is None else bars.addAction(self._plot_action(k))
        multi = p.addMenu(icon("g_double_y"), "&Multi-Panel/Axis")
        for k in ("double_y", "stack", "waterfall"):
            multi.addAction(self._plot_action(k))
        stat = p.addMenu(icon("g_histogram"), "&Statistical")
        for k in ("box", "histogram"):
            stat.addAction(self._plot_action(k))
        cont = p.addMenu(icon("g_contour"), "&Contour")
        for k in ("contour", "heatmap"):
            cont.addAction(self._plot_action(k))
        three = p.addMenu(icon("g_3d"), "3&D")
        for k in ("scatter3d", "surface3d"):
            three.addAction(self._plot_action(k))
        spec = p.addMenu(icon("g_polar"), "Specia&lized")
        for k in ("polar", "ternary"):
            spec.addAction(self._plot_action(k))

        c = menu("column", "&Column")
        set_as = c.addMenu("&Set As")
        set_as.addActions(list(self.act_desig.values()))
        c.addSeparator()
        c.addAction(self.act_set_values)
        fill = c.addMenu("&Fill Column With")
        fill.addActions([self.act_fill_rows, self.act_fill_uniform, self.act_fill_normal])
        c.addSeparator()
        c.addActions([self.act_add_column, self.act_add_columns_n, self.act_insert_column,
                      self.act_delete_column])
        move = c.addMenu("&Move Column")
        move.addActions([self.act_move_left, self.act_move_right])
        sort = c.addMenu("S&ort Column")
        sort.addActions([self.act_sort_asc, self.act_sort_desc])
        c.addSeparator()
        c.addAction(self.act_column_props)

        w = menu("worksheet", "&Worksheet")
        w.addActions([self.act_clear_ws, self.act_set_rows])
        sort_ws = w.addMenu("&Sort Worksheet")
        sort_ws.addActions([self.act_sort_ws_asc, self.act_sort_ws_desc])
        w.addAction(self.act_transpose)
        conv = w.addMenu(icon("e_matrix"), "Convert to &Matrix")
        conv.addActions([self.act_to_matrix_direct, self.act_to_matrix_xyz])
        w.addSeparator()
        w.addActions([self.act_add_sheet, self.act_duplicate_sheet, self.act_rename_sheet,
                      self.act_delete_sheet])

        mp = menu("mplot", "&Plot")
        cont_m = mp.addMenu(icon("g_contour"), "&Contour")
        for k in ("contour", "contour_lines", "heatmap", "image"):
            cont_m.addAction(self._matrix_plot_action(k))
        three_m = mp.addMenu(icon("g_3d"), "3&D")
        for k in ("surface3d_colormap", "surface3d", "wireframe"):
            three_m.addAction(self._matrix_plot_action(k))

        mm = menu("matrix", "&Matrix")
        mm.addActions([self.act_m_values, self.act_m_dims])
        mm.addSeparator()
        mm.addAction(self.act_m_transpose)
        flip = mm.addMenu("&Flip")
        flip.addActions([self.act_m_flip_h, self.act_m_flip_v])
        rot = mm.addMenu("&Rotate")
        rot.addActions([self.act_m_rot90, self.act_m_rot270, self.act_m_rot180])
        mm.addActions([self.act_m_expand, self.act_m_shrink])
        mm.addSeparator()
        mm.addAction(self.act_m_to_ws)
        mm.addSeparator()
        mm.addActions([self.act_image_mode, self.act_show_xy, self.act_m_add_sheet])

        g = menu("graph", "&Graph")
        g.addActions([self.act_add_plot, self.act_rescale])
        g.addSeparator()
        lay = g.addMenu("New &Layer (Axes)")
        lay.addActions([self.act_panel_below, self.act_panel_right, self.act_inset])
        g.addActions([self.act_right_y, self.act_remove_layer])
        g.addSeparator()
        g.addAction(self.act_legend)

        ins = menu("insert", "&Insert")
        for key in ("text", "arrow", "line", "rect"):
            ins.addAction(self.tool_acts[key])

        an_menu = menu("analysis", "&Analysis")
        m = an_menu.addMenu("&Mathematics")
        m.addAction(self._act("&Differentiate...", self.differentiate_dialog))
        m.addAction(self._act("&Integrate...", self.integrate_dialog))
        m.addAction(self._act("&Normalize Columns...", self.normalize_dialog))
        fit_m = an_menu.addMenu(icon("o_fit"), "&Fitting")
        fit_m.addActions([self.act_fit_linear, self.act_fit_poly, self.act_fit_nl])
        s = an_menu.addMenu("&Signal Processing")
        s.addAction(self._act("&Smoothing...", self.smooth_dialog))
        s.addAction(self._act("&FFT...", self.fft_dialog))
        pk = an_menu.addMenu("&Peaks and Baseline")
        pk.addAction(self._act("&Baseline (create and subtract)...", self.baseline_dialog))
        pk.addAction(self._act("&Find Peaks...", self.peaks_dialog))
        pk.addAction(self._act("&Multiple Peak Fit...", self.multi_peak_dialog))

        stm = menu("statistics", "&Statistics")
        d = stm.addMenu("&Descriptive Statistics")
        d.addAction(self.act_stats_cols)
        sdlg = self.stat_dialog
        d.addAction(self._act("Statistics on &Rows...", lambda: sdlg("rows")))
        d.addAction(self._act("&Frequency Counts...", lambda: sdlg("freq")))
        d.addSeparator()
        d.addAction(self._act("&Correlation Coefficient...", lambda: sdlg("correlation")))
        d.addAction(self._act("&Normality Test...", lambda: sdlg("normality")))
        h = stm.addMenu("&Hypothesis Testing")
        h.addAction(self._act("&One-Sample t-Test...", lambda: sdlg("ttest1")))
        h.addAction(self._act("&Two-Sample t-Test...", lambda: sdlg("ttest2")))
        h.addAction(self._act("&Pair-Sample t-Test...", lambda: sdlg("ttest_pair")))
        av = stm.addMenu("&ANOVA")
        av.addAction(self._act("&One-Way ANOVA...", lambda: sdlg("anova1")))
        av.addAction(self._act("&Two-Way ANOVA...", lambda: sdlg("anova2")))
        npm = stm.addMenu("&Nonparametric Tests")
        for key, text in (("mann_whitney", "&Mann-Whitney Test..."),
                          ("wilcoxon", "Paired Sample &Wilcoxon Signed Rank Test..."),
                          ("kruskal", "&Kruskal-Wallis ANOVA..."),
                          ("friedman", "&Friedman ANOVA...")):
            npm.addAction(self._act(text, lambda k=key: sdlg("nonparam", {"test": k})))

        t = menu("tools", "&Tools")
        t.addAction(self._act("&Options...", self.options_dialog))
        t.addSeparator()
        t.addAction(self._act("&Linear Fit...", lambda: self.fit_dialog("linear")))
        t.addAction(self._act("&Polynomial Fit...", lambda: self.fit_dialog("poly")))
        t.addAction(self._act("&Smoothing...", self.smooth_dialog))

        fm = menu("format", "F&ormat")
        fm.aboutToShow.connect(self._fill_format)

        win = menu("window", "&Window")
        win.aboutToShow.connect(self._fill_window_menu)

        hp = menu("help", "&Help")
        hp.addAction(self._act("&Getting Started", self.getting_started, "F1"))
        hp.addSeparator()
        hp.addAction(self._act("Check for &Updates...", self._check_updates))
        hp.addAction(self._act("&About RS-StatChem...", self._about))
        self._menu_mode = None
        self._set_menu_mode("book")

    def _check_updates(self) -> None:
        if self.ctx is not None:
            self.ctx.check_for_updates(manual=True)

    def _about(self) -> None:
        if self.ctx is not None:
            self.ctx.about()

    def _set_menu_mode(self, mode: str) -> None:
        """Origin swaps menus with the active window type."""
        if mode == self._menu_mode:
            return
        self._menu_mode = mode
        order = {
            "book": ["file", "edit", "view", "plot", "column", "worksheet", "analysis",
                     "statistics", "tools", "format", "window", "help"],
            "graph": ["file", "edit", "view", "graph", "format", "insert", "analysis", "tools",
                      "window", "help"],
            "matrix": ["file", "edit", "view", "mplot", "matrix", "tools", "format", "window",
                       "help"]}[mode]
        bar = self.menuBar()
        bar.clear()
        for key in order:
            bar.addMenu(self.menus[key])

    def _plot_action(self, kind: str) -> QAction:
        text, _ = PLOT_TYPES[kind]
        ic = {"line": "g_line", "scatter": "g_scatter", "line_symbol": "g_line_symbol",
              "yerror": "g_yerror", "column": "g_column", "column_stacked": "g_column_stacked",
              "column_percent": "g_column_stacked", "bar": "g_bar", "bar_stacked": "g_bar",
              "area": "g_area", "area_stacked": "g_area_stacked", "pie": "g_pie",
              "doughnut": "g_doughnut", "histogram": "g_histogram", "box": "g_box",
              "double_y": "g_double_y", "stack": "g_stack", "waterfall": "g_waterfall",
              "bubble": "g_bubble", "color_scatter": "g_color_scatter",
              "contour": "g_contour", "heatmap": "g_heatmap", "scatter3d": "g_3d",
              "surface3d": "g_3d", "polar": "g_polar", "ternary": "g_ternary"}[kind]
        return self._act(text, lambda k=kind: self.plot(k), None, ic, text)

    def _matrix_plot_action(self, kind: str) -> QAction:
        text = MATRIX_PLOTS[kind][0]
        ic = {"contour": "g_contour", "contour_lines": "g_contour", "heatmap": "g_heatmap",
              "image": "m_image"}.get(kind, "g_3d")
        return self._act(text, lambda k=kind: self.plot_matrix(k), None, ic, text)

    def _fill_format(self) -> None:
        fm = self.menus["format"]
        fm.clear()
        gwin = self.active_graph_window()
        if gwin is not None:
            fm.addAction(self.act_page)
            fm.addAction("&Layer Properties...", lambda: self._graph_props(("layer",)))
            axes = fm.addMenu("&Axes")
            axes.addAction("&X Axis...", lambda: self._graph_props(("axis", "x")))
            axes.addAction("&Y Axis...", lambda: self._graph_props(("axis", "y")))
            fm.addAction("&Plot Properties...", lambda: self._graph_props(("plot",)))
            fm.addAction("Le&gend Properties...", lambda: self._graph_props(("legend",)))
            fm.addSeparator()
            themes = fm.addMenu("&Theme")
            for name in THEMES:
                themes.addAction(name, lambda n=name: gwin.apply_theme(n))
            themes.addAction("Origin Default (ticks out)", lambda: gwin.mutate(
                lambda: apply_theme(gwin.graph, ORIGIN_THEME), rebuild_tree=False))
            fm.addSeparator()
            fm.addAction("Save as &Template...", gwin.save_template)
            if any(layer.kind == "3d" for layer in gwin.graph.layers):
                fm.addSeparator()
                rot = fm.addMenu("&Rotate 3D")
                rot.addAction("Azimuth +15°", lambda: gwin.rotate_3d(15, 0))
                rot.addAction("Azimuth −15°", lambda: gwin.rotate_3d(-15, 0))
                rot.addAction("Elevation +10°", lambda: gwin.rotate_3d(0, 10))
                rot.addAction("Elevation −10°", lambda: gwin.rotate_3d(0, -10))
            return
        if self.active_matrix_window() is not None:
            fm.addActions([self.act_m_dims, self.act_image_mode, self.act_show_xy])
            return
        fm.addAction(self.act_column_props)
        fm.addAction("&Worksheet Rows...", self.set_rows_dialog)
        fm.addAction("&Autofit Column Widths", self.autofit_columns)

    def _fill_window_menu(self) -> None:
        win = self.menus["window"]
        win.clear()
        win.addAction("&Cascade", self.mdi.cascadeSubWindows)
        win.addAction("&Tile", self.mdi.tileSubWindows)
        win.addAction("Arrange &Icons", self._arrange_minimized)
        win.addSeparator()
        win.addAction("&Refresh", self.refresh_all)
        win.addAction("&Duplicate", lambda: self.duplicate_window(None))
        win.addSeparator()
        active = self.active_name()
        for k, (name, sub) in enumerate(self.subs.items()):
            act = win.addAction(f"&{k + 1} {name}" if k < 9 else name,
                                lambda n=name: self.show_window(n))
            act.setCheckable(True)
            act.setChecked(name == active)
        win.addSeparator()
        add_module_menu(win, self.ctx)

    def _fill_recent(self) -> None:
        self.recent_menu.clear()
        recent = [p for p in (self.ctx.recent_files() if self.ctx is not None else [])
                  if p.lower().endswith((EXTENSION, GRAPH_EXT))]
        if not recent:
            act = self.recent_menu.addAction("(none)")
            act.setEnabled(False)
        for p in recent:
            self.recent_menu.addAction(Path(p).name, lambda path=p: self.open_path(path))

    # ================================================================== toolbars
    def _toolbar(self, title: str, name: str, area=Qt.ToolBarArea.TopToolBarArea,
                 size: int = 18) -> QToolBar:
        tb = QToolBar(title, self)
        tb.setObjectName(name)
        tb.setIconSize(QSize(size, size))
        self.addToolBar(area, tb)
        self.toolbar_menu.addAction(tb.toggleViewAction())
        return tb

    def _build_toolbars(self) -> None:
        std = self._toolbar("Standard", "standardToolbar")
        std.addActions([self.act_new_project, self.act_new_book, self.act_new_matrix,
                        self.act_new_graph, self.act_new_notes, self.act_new_folder])
        std.addSeparator()
        std.addActions([self.act_open, self.act_save])
        std.addSeparator()
        std.addActions([self.act_import_csv, self.act_import_excel])
        std.addSeparator()
        std.addActions([self.act_undo, self.act_redo])
        std.addSeparator()
        std.addActions([self.act_duplicate, self.act_refresh])
        std.addSeparator()
        std.addActions([self.act_view_pe, self.act_view_results, self.act_view_messages,
                        self.act_view_om])
        std.addSeparator()
        std.addActions([self.act_export_graph, self.act_copy_page])

        col = self._toolbar("Column", "columnToolbar")
        col.addActions([self.act_desig[k] for k in ("X", "Y", "Z", "yEr", "L", "N")])
        col.addSeparator()
        col.addActions([self.act_set_values, self.act_add_column, self.act_sort_asc])
        col.addSeparator()
        col.addActions([self.act_stats_cols, self.act_fit_linear])

        self.format_bar = FormatToolbar(self)
        self.format_bar.setObjectName("formatToolbar")
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.format_bar)
        self.toolbar_menu.addAction(self.format_bar.toggleViewAction())

        tools = self._toolbar("Tools", "toolsToolbar", Qt.ToolBarArea.LeftToolBarArea)
        tools.setOrientation(Qt.Orientation.Vertical)
        for key in ("pointer", "zoom", "pan", "reader"):
            tools.addAction(self.tool_acts[key])
        tools.addSeparator()
        for key in ("text", "arrow", "line", "rect"):
            tools.addAction(self.tool_acts[key])
        tools.addSeparator()
        tools.addAction(self.act_rescale)
        tools.addAction(self.act_add_plot)

        g2 = self._toolbar("2D Graphs", "graphs2dToolbar", Qt.ToolBarArea.BottomToolBarArea)
        for group in (("line", "line_symbol"), ("scatter", "yerror", "bubble", "color_scatter"),
                      ("column", "column_stacked", "column_percent", "bar", "bar_stacked"),
                      ("area", "area_stacked"), ("pie", "doughnut"), ("histogram",),
                      ("box",), ("double_y", "stack", "waterfall"), ("contour", "heatmap"),
                      ("scatter3d", "surface3d"), ("polar", "ternary")):
            self._gallery_button(g2, group)
        self.graphs_toolbar = g2

    def _gallery_button(self, tb: QToolBar, kinds: tuple) -> None:
        """Origin's 2D Graphs toolbar button: click plots the first type, the arrow opens
        the group's gallery."""
        first = self._plot_action(kinds[0])
        if len(kinds) == 1:
            tb.addAction(first)
            return
        button = QToolButton(tb)
        button.setDefaultAction(first)
        button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        menu = QMenu(button)
        for k in kinds:
            menu.addAction(self._plot_action(k))
        button.setMenu(menu)
        tb.addWidget(button)

    def _build_status(self) -> None:
        bar = self.statusBar()
        self.range_label = QLabel("")
        self.range_label.setObjectName("statusBox")
        self.au_label = QLabel("AU : ON")
        self.au_label.setObjectName("statusBox")
        self.au_label.setToolTip("Auto update of column formulas and linked graphs")
        self.angle_label = QLabel("Radian")
        self.angle_label.setObjectName("statusBox")
        bar.addPermanentWidget(self.au_label)
        bar.addPermanentWidget(self.range_label)
        bar.addPermanentWidget(self.angle_label)
        bar.showMessage("For Help, press F1", 0)

    # ================================================================== logs
    def message(self, text: str, error: bool = False) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.messages.appendPlainText(f"[{stamp}] {'Error: ' if error else ''}{text}")
        if error:
            self.messages_dock.show()
            self.messages_dock.raise_()
            self.statusBar().showMessage(text, 8000)

    def result(self, text: str) -> None:
        self.project.results_log.append(text)
        self.results.appendPlainText(text + "\n")
        self.results_dock.show()
        self.results_dock.raise_()

    def show_status(self, text: str) -> None:
        if text:
            self.statusBar().showMessage(text, 6000)

    def show_reader(self, text: str) -> None:
        self.statusBar().showMessage(text)

    def _warn(self, text: str, title: str = "Graph & Analysis") -> None:
        self.message(text, error=True)
        if self.isVisible() and not self.quiet and QApplication.platformName() != "offscreen":
            QMessageBox.warning(self, title, text)

    # ================================================================== window registry
    def view_of(self, name: str):
        sub = self.subs.get(name)
        return sub.widget() if sub is not None else None

    def book_window(self, book: Workbook) -> BookWindow | None:
        v = self.view_of(book.name)
        return v if isinstance(v, BookWindow) else None

    def graph_windows(self) -> list[OriginGraphWindow]:
        return [s.widget() for s in self.subs.values() if isinstance(s.widget(),
                                                                     OriginGraphWindow)]

    def active_sub(self) -> _SubWindow | None:
        sub = self.mdi.activeSubWindow() or self.mdi.currentSubWindow()
        return sub if isinstance(sub, _SubWindow) else None

    def active_view(self):
        sub = self.active_sub()
        return sub.widget() if sub is not None else None

    def active_name(self) -> str | None:
        sub = self.active_sub()
        return sub.obj.name if sub is not None else None

    def active_book_window(self) -> BookWindow | None:
        v = self.active_view()
        return v if isinstance(v, BookWindow) else None

    def active_graph_window(self) -> OriginGraphWindow | None:
        v = self.active_view()
        return v if isinstance(v, OriginGraphWindow) else None

    def active_matrix_window(self) -> MatrixWindow | None:
        v = self.active_view()
        return v if isinstance(v, MatrixWindow) else None

    def matrix_window(self, book: MatrixBook) -> MatrixWindow | None:
        v = self.view_of(book.name)
        return v if isinstance(v, MatrixWindow) else None

    def _add_sub(self, obj, widget, size: tuple[int, int]) -> _SubWindow:
        sub = _SubWindow(self, obj)
        sub.setWidget(widget)
        sub.setWindowTitle(obj.title)
        sub.setWindowIcon(icon({"book": "e_book", "graph": "e_graph", "matrix": "e_matrix",
                                "notes": "e_notes"}[obj.kind]))
        self.mdi.addSubWindow(sub)
        sub.resize(*size)
        n = len(self.subs)
        sub.move(12 + 26 * (n % 8), 10 + 24 * (n % 8))
        self.subs[obj.name] = sub
        return sub

    def _register(self, obj, folder: Folder | None = None) -> None:
        # a new object whose name is taken is renamed (Book1 → Book2) by Project.add; objects
        # of the project itself (opening a project) are already registered
        existing = self.project.windows.get(obj.name)
        if existing is obj or (isinstance(existing, GraphDoc) and isinstance(obj, GraphDoc)
                               and existing.graph is obj.graph):
            return
        self.project.add(obj, folder or self.explorer.current or self.project.root)

    def default_rows(self) -> int:
        try:
            return max(1, int(self._setting("rows", DEFAULT_ROWS)))
        except (TypeError, ValueError):
            return DEFAULT_ROWS

    # ================================================================== new windows
    def new_book(self, book: Workbook | None = None, show: bool = True,
                 folder: Folder | None = None) -> BookWindow:
        if book is None:
            book = Workbook.new(self.project.unique_name("Book"), 2, self.default_rows())
        self._register(book, folder)
        bw = BookWindow(book, self)
        bw.changed.connect(self._book_changed)
        sub = self._add_sub(book, bw, (560, 420))
        if show and book.name not in self.project.hidden:
            sub.show()
            self.mdi.setActiveSubWindow(sub)
        else:
            self.project.hidden.add(book.name)
            sub.hide()
        self.explorer.fill_windows()
        if self.project.windows and len(self.project.windows) > 1:
            self.set_modified(True)
        return bw

    def new_matrix(self, book: MatrixBook | None = None, show: bool = True,
                   folder: Folder | None = None) -> MatrixWindow:
        """File > New > Matrix (an MBook with one 32 × 32 matrix sheet)."""
        if book is None:
            book = MatrixBook.new(self.project.unique_name("MBook"))
        self._register(book, folder)
        mw = MatrixWindow(book, self)
        mw.changed.connect(self._matrix_changed)
        sub = self._add_sub(book, mw, (520, 400))
        if show and book.name not in self.project.hidden:
            sub.show()
            self.mdi.setActiveSubWindow(sub)
        else:
            self.project.hidden.add(book.name)
            sub.hide()
        self.explorer.fill_windows()
        self.set_modified(True)
        return mw

    def new_graph(self, graph: Graph | None = None, show: bool = True,
                  folder: Folder | None = None, name: str | None = None) -> OriginGraphWindow:
        name = name or (graph.title if graph is not None and graph.title and
                        graph.title not in self.project.windows else
                        self.project.unique_name("Graph"))
        if graph is None:
            graph = Graph(title=name, layers=[Layer()])
            apply_theme(graph, ORIGIN_THEME)
            for axis in (graph.layers[0].x, graph.layers[0].y):
                axis.min, axis.max = 0.0, 10.0
        graph.title = name
        doc = GraphDoc(name, graph)
        self._register(doc, folder)
        gwin = OriginGraphWindow(graph, self)
        self.obj_stack.addWidget(gwin.tree)
        self._graph_trees[id(gwin)] = gwin.tree
        sub = self._add_sub(doc, gwin, (600, 470))
        if show and name not in self.project.hidden:
            sub.show()
            self.mdi.setActiveSubWindow(sub)
            QTimer.singleShot(0, gwin.fit_page)
        else:
            self.project.hidden.add(name)
            sub.hide()
        self.explorer.fill_windows()
        self.set_modified(True)
        return gwin

    def new_notes(self, notes: Notes | None = None, show: bool = True,
                  folder: Folder | None = None) -> _NotesEdit:
        notes = notes or Notes(self.project.unique_name("Notes"))
        self._register(notes, folder)
        edit = _NotesEdit(notes, self)
        sub = self._add_sub(notes, edit, (420, 300))
        if show and notes.name not in self.project.hidden:
            sub.show()
            self.mdi.setActiveSubWindow(sub)
        else:
            self.project.hidden.add(notes.name)
            sub.hide()
        self.explorer.fill_windows()
        self.set_modified(True)
        return edit

    def new_folder(self, name: str | None = None) -> Folder:
        parent = self.explorer.current or self.project.root
        base = "Folder"
        used = {f.name for f in self.project.root.walk()}
        k = 1
        while f"{base}{k}" in used:
            k += 1
        folder = Folder(name or f"{base}{k}")
        parent.folders.append(folder)
        self.explorer.current = folder
        self.explorer.rebuild()
        self._folder_changed(folder)
        self.set_modified(True)
        return folder

    def delete_folder(self, folder: Folder) -> None:
        parent = self.project.root.parent_of(folder)
        if parent is None:
            return
        for f in list(folder.walk()):
            for name in list(f.windows):
                self._delete_window(name)
        parent.folders.remove(folder)
        self.explorer.current = parent
        self.explorer.rebuild()
        self._folder_changed(parent)
        self.set_modified(True)

    # ================================================================== window commands
    def show_window(self, name: str) -> None:
        sub = self.subs.get(name)
        if sub is None:
            return
        self.project.hidden.discard(name)
        folder = self.project.root.find_window(name)
        if folder is not None and folder is not self.explorer.current and \
                self.act_folder_only.isChecked():
            self.explorer.current = folder
            self.explorer.rebuild()
            self._folder_changed(folder)
        sub.show()
        if sub.isMinimized():
            sub.showNormal()
        self.mdi.setActiveSubWindow(sub)
        self.explorer.fill_windows()

    def hide_window(self, name: str) -> None:
        sub = self.subs.get(name)
        if sub is None:
            return
        self.project.hidden.add(name)
        sub.hide()
        self.explorer.fill_windows()
        self.set_modified(True)

    def ask_close_window(self, name: str) -> None:
        if name not in self.subs:
            return
        box = QMessageBox(self)
        box.setWindowTitle("Attention!")
        box.setIcon(QMessageBox.Icon.Question)
        box.setText(f"Do you want to delete the window {name} permanently, or hide it?")
        delete = box.addButton("Delete", QMessageBox.ButtonRole.DestructiveRole)
        hide = box.addButton("Hide", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(hide)
        box.exec()
        if box.clickedButton() is delete:
            self.delete_windows([name])
        elif box.clickedButton() is hide:
            self.hide_window(name)

    def delete_windows(self, names: list[str], ask: bool = False) -> None:
        if ask and QMessageBox.question(
                self, "Delete", f"Delete {', '.join(names)} permanently?") != \
                QMessageBox.StandardButton.Yes:
            return
        for name in names:
            self._delete_window(name)
        self.explorer.fill_windows()
        self.set_modified(True)

    def _delete_window(self, name: str) -> None:
        sub = self.subs.pop(name, None)
        self.project.remove(name)
        if sub is None:
            return
        widget = sub.widget()
        if isinstance(widget, OriginGraphWindow):
            tree = self._graph_trees.pop(id(widget), None)
            if tree is not None:
                self.obj_stack.removeWidget(tree)
                tree.deleteLater()
        self.closing_all, prev = True, self.closing_all
        sub.close()
        self.closing_all = prev
        self.mdi.removeSubWindow(sub)
        sub.deleteLater()

    def rename_window(self, name: str | None, new: str | None = None) -> bool:
        name = name or self.active_name()
        if name is None:
            return False
        if new is None:
            new, ok = QInputDialog.getText(self, "Rename", "Window name:", text=name)
            if not ok:
                return False
        try:
            self.project.rename(name, new)
        except ValueError as exc:
            self._warn(str(exc))
            return False
        sub = self.subs.pop(name)
        self.subs[new.strip()] = sub
        widget = sub.widget()
        if isinstance(widget, OriginGraphWindow):
            widget.graph.title = new.strip()
            widget.rebuild_tree()
        for gwin in self.graph_windows():   # links / sources follow the renamed book
            for layer in gwin.graph.layers:
                for p in layer.plots:
                    src = p.options.get("source")
                    if src and src.get("book") == name:
                        src["book"] = new.strip()
            for link in gwin.graph.meta.get("links", {}).values():
                if link.get("book") == name:
                    link["book"] = new.strip()
            update_data_ranges(gwin.graph)
            gwin.rebuild_tree()
        sub.setWindowTitle(sub.obj.title)
        self.explorer.fill_windows()
        self.set_modified(True)
        return True

    def sheet_renamed(self, book: Workbook, old: str, new: str) -> None:
        for wb, rep in recalc.reports(self.project):   # report inputs follow the sheet
            src = rep.op.get("input")
            if src and src.get("book") == book.name and src.get("sheet") == old:
                src["sheet"] = new
            if rep.op.get("settings"):
                rep.op["settings"] = rename_refs(rep.op["settings"], f"[{book.name}]{old}!",
                                                 f"[{book.name}]{new}!", key="text")
        for gwin in self.graph_windows():
            for link in gwin.graph.meta.get("links", {}).values():
                if link.get("book") == book.name and link.get("sheet") == old:
                    link["sheet"] = new
            for layer in gwin.graph.layers:
                for p in layer.plots:
                    src = p.options.get("source")
                    if src and src.get("book") == book.name and src.get("sheet") == old:
                        src["sheet"] = new
            update_data_ranges(gwin.graph)

    def duplicate_window(self, name: str | None) -> None:
        name = name or self.active_name()
        obj = self.project.windows.get(name)
        if obj is None:
            return
        folder = self.project.root.find_window(name)
        if isinstance(obj, Workbook):
            from rsstatchem.ui.origin.model import window_from_dict, window_to_dict
            copy_ = window_from_dict(window_to_dict(obj))
            copy_.name = self.project.unique_name("Book")
            self.new_book(copy_, folder=folder)
        elif isinstance(obj, MatrixBook):
            from rsstatchem.ui.origin.model import window_from_dict, window_to_dict
            copy_ = window_from_dict(window_to_dict(obj))
            copy_.name = self.project.unique_name("MBook")
            self.new_matrix(copy_, folder=folder)
        elif isinstance(obj, GraphDoc):
            gwin = self.view_of(name)
            self.new_graph(gwin.graph.copy(), folder=folder,
                           name=self.project.unique_name("Graph"))
        elif isinstance(obj, Notes):
            self.new_notes(Notes(self.project.unique_name("Notes"), obj.text), folder=folder)

    def move_windows(self, names: list[str], folder: Folder) -> None:
        for name in names:
            for f in self.project.root.walk():
                if name in f.windows:
                    f.windows.remove(name)
            folder.windows.append(name)
        self._folder_changed(self.explorer.current)
        self.explorer.fill_windows()
        self.set_modified(True)

    def _folder_changed(self, folder: Folder) -> None:
        if not self.act_folder_only.isChecked():
            return
        for name, sub in self.subs.items():
            visible = name in folder.windows and name not in self.project.hidden
            sub.setVisible(visible)

    def _folder_filter(self) -> None:
        self._set_setting("folder_only", "true" if self.act_folder_only.isChecked() else
                          "false")
        if self.act_folder_only.isChecked():
            self._folder_changed(self.explorer.current or self.project.root)
        else:
            for name, sub in self.subs.items():
                sub.setVisible(name not in self.project.hidden)

    def _arrange_minimized(self) -> None:
        for sub in self.subs.values():
            if sub.isMinimized():
                sub.showMinimized()

    def _full_screen(self) -> None:
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    # ================================================================== activation
    def _activated(self, sub) -> None:
        widget = sub.widget() if sub is not None else None
        if isinstance(widget, OriginGraphWindow):
            self._set_menu_mode("graph")
            tree = self._graph_trees.get(id(widget))
            if tree is not None:
                self.obj_stack.setCurrentWidget(tree)
            self.om_dock.setWindowTitle("Object Manager - " + widget.graph.title)
            self.range_label.setText(f"[{widget.graph.title}] Layer 1")
            self.set_tool(self._current_tool(), quiet=True)
        elif isinstance(widget, MatrixWindow):
            self._set_menu_mode("matrix")
            self.obj_stack.setCurrentWidget(self.book_tree)
            self.matrix_view_changed(widget)
        else:
            self._set_menu_mode("book")
            self.obj_stack.setCurrentWidget(self.book_tree)
            if isinstance(widget, BookWindow):
                self.book_view_changed(widget)
            else:
                self.book_tree.clear()
                self.om_dock.setWindowTitle("Object Manager")
                self.range_label.setText("")
        enable_graph = isinstance(widget, OriginGraphWindow)
        for act in (self.act_add_plot, self.act_rescale, self.act_copy_page,
                    self.act_export_graph, *self.tool_acts.values()):
            act.setEnabled(enable_graph)
        is_matrix = isinstance(widget, MatrixWindow)
        self.act_image_mode.setEnabled(is_matrix)
        self.act_show_xy.setEnabled(is_matrix)
        self.act_image_mode.setChecked(is_matrix and widget.image_mode)
        self.act_show_xy.setChecked(is_matrix and widget.show_xy)
        ws = widget.worksheet() if isinstance(widget, BookWindow) else None
        self.act_sparklines.setEnabled(ws is not None)
        self.act_sparklines.setChecked(bool(ws is not None and ws.sparklines))
        self.format_bar.sync()
        self.explorer.fill_windows()

    def _current_tool(self) -> str:
        return next((k for k, a in self.tool_acts.items() if a.isChecked()), "pointer")

    def book_view_changed(self, bw: BookWindow) -> None:
        if bw is not self.active_book_window():
            return
        self._fill_book_tree(bw)
        ws = bw.worksheet()
        sheet = bw.sheet()
        if ws is not None:
            cols = bw.selected_columns()
            n = ws.data_rows()
            rows = bw.selected_row_range()
            if cols:
                names = ",".join(ws.columns[c].name for c in cols[:4]) + \
                    ("…" if len(cols) > 4 else "")
                rng = f"[{rows[0]}:{rows[1]}]" if rows else f"[1:{n}]"
                self.range_label.setText(f"[{bw.book.name}]{ws.name}!Col({names}){rng}")
            else:
                self.range_label.setText(f"[{bw.book.name}]{ws.name}")
            self.act_sparklines.setEnabled(True)
            self.act_sparklines.setChecked(ws.sparklines)
        elif sheet is not None:
            self.range_label.setText(f"[{bw.book.name}]{sheet.name}")
            self.act_sparklines.setEnabled(False)
        view = bw.sheet_view()
        if view is not None and not getattr(view, "_origin_hooked", False):
            view.selectionModel().selectionChanged.connect(lambda *_: self.book_view_changed(bw))
            view._origin_hooked = True

    def _fill_book_tree(self, bw: BookWindow) -> None:
        self.book_tree.clear()
        book = bw.book
        self.om_dock.setWindowTitle("Object Manager - " + book.name)
        root = QTreeWidgetItem([book.title])
        root.setIcon(0, icon("e_book"))
        self.book_tree.addTopLevelItem(root)
        for si, sheet in enumerate(book.sheets):
            item = QTreeWidgetItem([sheet.name])
            item.setIcon(0, icon("e_report" if isinstance(sheet, ReportSheet) else "e_book"))
            item.setData(0, Qt.ItemDataRole.UserRole, (si, -1))
            root.addChild(item)
            if isinstance(sheet, Worksheet):
                for ci, c in enumerate(sheet.columns):
                    ch = QTreeWidgetItem([sheet.label(ci) + (f"  {c.long_name}" if c.long_name
                                                             else "")])
                    ch.setIcon(0, icon("e_column"))
                    ch.setData(0, Qt.ItemDataRole.UserRole, (si, ci))
                    item.addChild(ch)
            item.setExpanded(si == book.active)
        root.setExpanded(True)

    def _book_tree_clicked(self, item, _col) -> None:
        data = item.data(0, Qt.ItemDataRole.UserRole)
        mw = self.active_matrix_window()
        if mw is not None and data:
            mw.tabs.setCurrentIndex(data[0])
            return
        bw = self.active_book_window()
        if not data or bw is None:
            return
        si, ci = data
        bw.tabs.setCurrentIndex(si)
        if ci >= 0:
            bw.select_columns([ci])

    # ================================================================== modified / title
    def set_modified(self, on: bool) -> None:
        self.modified = on
        self._update_title()

    def _update_title(self) -> None:
        name = Path(self.project.path).name if self.project.path else self.project.name
        self.setWindowTitle(f"{APP_TITLE} - {name}{' *' if self.modified else ''}")

    def _book_changed(self, book: Workbook) -> None:
        self.set_modified(True)
        self.check_reports(book.name)
        self.refresh_graphs()
        bw = self.book_window(book)
        if bw is not None:
            self.book_view_changed(bw)
        self.explorer.fill_windows()

    def _matrix_changed(self, book: MatrixBook) -> None:
        self.set_modified(True)
        self.refresh_graphs()
        mw = self.matrix_window(book)
        if mw is not None:
            self.matrix_view_changed(mw)
        self.explorer.fill_windows()

    def graph_changed(self, gwin: OriginGraphWindow) -> None:
        self.explorer.fill_windows()

    def refresh_graphs(self) -> None:
        """Origin keeps graphs linked to their worksheet and matrix data."""
        books = self.project.data_windows()
        for gwin in self.graph_windows():
            if refresh_links(gwin.graph, books):
                gwin.redraw()

    # ================================================================== recalculation
    def check_reports(self, changed: str | None = None) -> None:
        """After a data change: re-run Auto reports, flag Manual ones (Origin's lock)."""
        if getattr(self, "_recalculating", False):
            return
        for book, rep in recalc.stale_reports(self.project, changed):
            if rep.mode == "auto":
                self.recalculate_report(book, rep, quiet=True)
            elif not rep.dirty:
                rep.op["dirty"] = True
                bw = self.book_window(book)
                if bw is not None:
                    bw.refresh()
                self.message(f"[{book.name}]{rep.name}: the source data changed — "
                             "Recalculate to update (Manual mode).")

    def recalculate_report(self, book: Workbook, report: ReportSheet,
                           quiet: bool = False) -> bool:
        """Re-run the analysis of ``report`` in place (one undo step in its workbook)."""
        self._recalculating = True
        try:
            text = self._run_in_book(book, "Recalculate",
                                     lambda: recalc.recalculate(self.project, book, report))
        except (ValueError, KeyError) as exc:
            self._warn(f"Recalculate [{book.name}]{report.name}: {exc}")
            return False
        finally:
            self._recalculating = False
        self.result(text)
        self.refresh_graphs()
        self.update_report_graphs(book, report)
        bw = self.book_window(book)
        if bw is not None:
            bw.refresh()
        if not quiet:
            self.message(f"Recalculated [{book.name}]{report.name}")
        return True

    def update_report_graphs(self, book: Workbook, report) -> None:
        """Graph windows opened from a report's live graphs follow its recalculation."""
        if not isinstance(report, ReportSheet):
            return
        for gwin in self.graph_windows():
            link = gwin.graph.meta.get("report_link") or {}
            if link.get("book") != book.name or link.get("report") != report.name:
                continue
            item = next((i for i in report.graphs() if i.get("title") == link.get("item")),
                        None)
            if item is None:
                continue
            from rsstatchem.core.graph.model import graph_from_json_dict
            rg.update_linked(gwin.graph, graph_from_json_dict(item["graph"]))
            gwin.redraw(rebuild_tree=True)

    def set_report_mode(self, book: Workbook, report: ReportSheet, mode: str) -> None:
        report.op["mode"] = mode
        self.set_modified(True)
        if mode == "auto" and (report.dirty or recalc.fingerprint(
                self.project.data_windows(), report.op) != report.op.get("hash")):
            self.recalculate_report(book, report)
        elif mode == "none":
            report.op["dirty"] = False
        bw = self.book_window(book)
        if bw is not None:
            bw.refresh()

    def change_report_parameters(self, book: Workbook, report: ReportSheet) -> None:
        """Reopen the dialog of a report (or of locked output columns) with its settings;
        OK recalculates it."""
        op = report.op
        if op.get("kind") in ("stats", "rows", "freq"):
            settings = dict(op.get("settings") or {})
            settings["mode"] = op.get("mode", "manual")
            dlg = sd.make_dialog(op["test"] if op["kind"] == "stats" else op["kind"], self,
                                 settings)
            if not dlg.exec():
                return
            values = dlg.values()
            report.op["settings"] = {k: v for k, v in values.items() if k != "mode"}
            report.op["mode"] = values["mode"]
            self.recalculate_report(book, report)
        elif op.get("kind") == "fit":
            src = op["input"]
            label = range_text(src["book"], src["sheet"], src["y"],
                               *(src.get("rows") or (None, None)))
            choice = InputChoice(label, src["book"], src["sheet"], src["y"], src.get("x"),
                                 None, tuple(src["rows"]) if src.get("rows") else None)
            dlg = FitDialog(op.get("fit_kind", "nlfit"), [choice], self)
            dlg.preset(op.get("settings", {}))
            if not dlg.exec():
                return
            try:
                params = dlg.values()
            except ValueError as exc:
                self._warn(str(exc))
                return
            s = report.op["settings"]
            for k in ("model", "weighting", "params0", "fixed", "bounds", "level", "conf_band",
                      "pred_band", "residual_plot", "output_fitted"):
                s[k] = params.get(k, s.get(k))
            self.recalculate_report(book, report)

    def open_report_graph(self, book: Workbook, report: ReportSheet, item: dict):
        """Double-click on a report graph: a fit's Fitted Curves Plot opens as a graph of the
        source data and the fitted curve (both linked to the workbook); any other live graph
        opens as a copy linked to the report, updated when the report is recalculated."""
        spec = item.get("open") or (item.get("graph") if item["kind"] == "image" else None) \
            or {}
        if spec.get("kind") != "fit":
            if item["kind"] != "graph":
                return None
            from rsstatchem.core.graph.model import graph_from_json_dict
            graph = graph_from_json_dict(item["graph"])
            graph.meta["report_link"] = {"book": book.name, "report": report.name,
                                         "item": item.get("title", "")}
            graph.title = self.project.unique_name("Graph")
            return self.new_graph(graph, folder=self.project.root.find_window(book.name))
        src_book = self.project.windows.get(spec.get("book"))
        ws = src_book.sheet(spec.get("sheet", "")) if isinstance(src_book, Workbook) else None
        curve = book.sheet(spec.get("curve", ""))
        if not isinstance(ws, Worksheet) or not isinstance(curve, Worksheet):
            self._warn("The data of this plot no longer exist.")
            return
        try:
            y = ws.index_of(spec["y"])
            graph = plot_columns(src_book, ws, [y], "scatter", self.project.unique_name("Graph"),
                                 rows=tuple(spec["rows"]) if spec.get("rows") else None)
        except (KeyError, PlotError) as exc:
            self._warn(str(exc))
            return
        add_xy_plot(graph, 0, curve.columns[0].numeric(), curve.columns[1].numeric(),
                    curve.columns[1].long_name, "line", FIT_COLOR,
                    link={"book": book.name, "sheet": curve.name, "x": curve.columns[0].name,
                          "y": curve.columns[1].name},
                    source={"book": book.name, "sheet": curve.name,
                            "x": curve.columns[0].name, "y": curve.columns[1].name})
        return self.new_graph(graph, folder=self.project.root.find_window(book.name))

    def refresh_all(self) -> None:
        for book in self.project.books():
            bw = self.book_window(book)
            for s in book.sheets:
                if isinstance(s, Worksheet):
                    fx.recalculate(s)
            if bw is not None:
                bw.refresh()
        self.refresh_graphs()
        for gwin in self.graph_windows():
            gwin.redraw(rebuild_tree=True)

    # ================================================================== edit commands
    def undo(self) -> None:
        v = self.active_view()
        if isinstance(v, (BookWindow, MatrixWindow)):
            v.undo_stack.undo()
            self.refresh_graphs()
        elif isinstance(v, OriginGraphWindow):
            v.undo()
        elif isinstance(v, QPlainTextEdit):
            v.undo()

    def redo(self) -> None:
        v = self.active_view()
        if isinstance(v, (BookWindow, MatrixWindow)):
            v.undo_stack.redo()
            self.refresh_graphs()
        elif isinstance(v, OriginGraphWindow):
            v.redo()
        elif isinstance(v, QPlainTextEdit):
            v.redo()

    def copy(self, with_headers: bool = False) -> None:
        v = self.active_view()
        if isinstance(v, BookWindow):
            v.copy(with_headers)
        elif isinstance(v, MatrixWindow):
            QApplication.clipboard().setText(v.copy_text())
        elif isinstance(v, OriginGraphWindow):
            v.copy_to_clipboard()
        elif isinstance(v, QPlainTextEdit):
            v.copy()

    def cut(self) -> None:
        v = self.active_view()
        if isinstance(v, BookWindow):
            v.copy()
            v.clear_selection()
        elif isinstance(v, QPlainTextEdit):
            v.cut()

    def paste(self) -> None:
        v = self.active_view()
        if isinstance(v, BookWindow):
            v.paste()
        elif isinstance(v, MatrixWindow):
            block = v.view().selected_block() if v.view() is not None else None
            v.paste_text(QApplication.clipboard().text(), *(block[:2] if block else (0, 0)))
        elif isinstance(v, QPlainTextEdit):
            v.paste()

    def clear_cells(self) -> None:
        v = self.active_view()
        if isinstance(v, BookWindow):
            v.clear_selection()

    def delete_selected(self) -> None:
        v = self.active_view()
        if isinstance(v, BookWindow):
            v.clear_selection()
        elif isinstance(v, OriginGraphWindow):
            v.delete_selected()

    def select_all(self) -> None:
        v = self.active_view()
        if isinstance(v, BookWindow) and v.sheet_view() is not None:
            v.sheet_view().selectAll()
        elif isinstance(v, QPlainTextEdit):
            v.selectAll()

    def copy_page(self) -> None:
        gwin = self.active_graph_window()
        if gwin is not None:
            gwin.copy_to_clipboard()

    # ================================================================== column commands
    def _sheet_target(self) -> tuple[BookWindow, Worksheet, list[int]] | None:
        bw = self.active_book_window()
        ws = bw.worksheet() if bw is not None else None
        if ws is None:
            return None
        return bw, ws, bw.selected_columns()

    def set_designation(self, code: str, cols: list[int] | None = None) -> None:
        t = self._sheet_target()
        if t is None:
            return
        bw, ws, sel = t
        cols = sel if cols is None else cols
        if not cols:
            return

        def fn():
            for c in cols:
                ws.columns[c].designation = code
        bw.change(f"Set As {DESIGNATIONS[code][0]}", fn, recalc=False)
        bw.select_columns(cols)

    def add_columns(self, n: int = 1) -> None:
        t = self._sheet_target()
        if t is None:
            return
        bw, ws, _ = t
        bw.change("Add New Columns", lambda: [ws.add_column("Y") for _ in range(n)],
                  recalc=False)

    def add_columns_dialog(self) -> None:
        n, ok = QInputDialog.getInt(self, "Add New Columns", "Number of columns to add:", 1, 1,
                                    500)
        if ok:
            self.add_columns(n)

    def insert_column(self) -> None:
        t = self._sheet_target()
        if t is None:
            return
        bw, ws, cols = t
        at = cols[0] if cols else len(ws.columns)
        bw.change("Insert Column", lambda: ws.add_column("Y", at=at), recalc=False)

    def delete_columns(self, cols: list[int] | None = None) -> None:
        t = self._sheet_target()
        if t is None:
            return
        bw, ws, sel = t
        cols = sel if cols is None else cols
        if cols:
            bw.change("Delete Columns", lambda: ws.delete_columns(cols))

    def clear_columns(self) -> None:
        t = self._sheet_target()
        if t is None:
            return
        bw, ws, cols = t

        def fn():
            for c in cols:
                ws.set_column_values(c, [])
                ws.columns[c].formula = ""
        bw.change("Clear Columns", fn)

    def move_column(self, step: int) -> None:
        t = self._sheet_target()
        if t is None or len(t[2]) != 1:
            return
        bw, ws, (c,) = t
        to = c + step
        if not 0 <= to < len(ws.columns):
            return
        bw.change("Move Column", lambda: ws.columns.insert(to, ws.columns.pop(c)), recalc=False)
        bw.select_columns([to])

    def fill_columns(self, how: str, cols: list[int] | None = None) -> None:
        t = self._sheet_target()
        if t is None:
            return
        bw, ws, sel = t
        cols = sel if cols is None else cols
        if not cols:
            return
        rng = np.random.default_rng()
        n = ws.n_rows

        def fn():
            for c in cols:
                if how == "rows":
                    values = list(np.arange(1, n + 1, dtype=float))
                elif how == "uniform":
                    values = list(rng.uniform(0, 1, n))
                else:
                    values = list(rng.normal(0, 1, n))
                ws.set_column_values(c, values)
                ws.columns[c].formula = ""
        bw.change("Fill Column", fn)

    def sort_columns(self, descending: bool) -> None:
        t = self._sheet_target()
        if t is None:
            return
        bw, ws, cols = t

        def fn():
            for c in cols:
                vals = [v for v in ws.columns[c].values if v is not None]
                nums = [v for v in vals if isinstance(v, float)]
                texts = [v for v in vals if not isinstance(v, float)]
                ordered = sorted(nums, reverse=descending) + sorted(texts, key=str,
                                                                     reverse=descending)
                ws.set_column_values(c, ordered)
        bw.change("Sort Column", fn)

    def sort_worksheet(self, descending: bool) -> None:
        t = self._sheet_target()
        if t is None:
            return
        bw, ws, cols = t
        key_col = cols[0] if cols else next((i for i, c in enumerate(ws.columns)
                                             if c.designation == "X"), 0)
        n = ws.data_rows()

        def fn():
            keys = ws.columns[key_col].numeric(n)
            order = sorted(range(n), key=lambda r: (math.isnan(keys[r]),
                                                    -keys[r] if descending else keys[r]))
            for c in ws.columns:
                vals = (c.values + [None] * n)[:n]
                c.values[:n] = [vals[r] for r in order]
        bw.change("Sort Worksheet", fn)

    def transpose_worksheet(self) -> None:
        t = self._sheet_target()
        if t is None:
            return
        bw, ws, _ = t
        n = ws.data_rows()

        def fn():
            grid = [[(c.values + [None] * n)[r] for c in ws.columns] for r in range(n)]
            old_cols = len(ws.columns)
            ws.columns.clear()
            for r in range(n):
                ws.add_column("Y" if r else "X")
            ws.set_rows(max(old_cols, 1))
            for ci, row in enumerate(grid):
                for ri, v in enumerate(row):
                    ws.columns[ci].values[ri] = v
        bw.change("Transpose", fn)

    def clear_worksheet(self) -> None:
        t = self._sheet_target()
        if t is None:
            return
        bw, ws, _ = t

        def fn():
            for c in ws.columns:
                c.values = [None] * ws.n_rows
                c.formula = ""
        bw.change("Clear Worksheet", fn, recalc=False)

    def set_rows_dialog(self) -> None:
        t = self._sheet_target()
        if t is None:
            return
        bw, ws, _ = t
        n, ok = QInputDialog.getInt(self, "Worksheet Rows", "Number of rows:", ws.n_rows,
                                    max(1, ws.data_rows()), 10_000_000)
        if ok:
            bw.change("Set Rows", lambda: ws.set_rows(n), recalc=False)

    def autofit_columns(self) -> None:
        bw = self.active_book_window()
        view = bw.sheet_view() if bw is not None else None
        if view is not None:
            view.resizeColumnsToContents()

    def add_sheet(self) -> None:
        bw = self.active_book_window()
        if bw is not None:
            bw.add_sheet()

    def rename_sheet(self) -> None:
        bw = self.active_book_window()
        if bw is not None:
            bw.rename_sheet(bw.tabs.currentIndex())

    def delete_sheet(self) -> None:
        bw = self.active_book_window()
        if bw is not None:
            bw.delete_sheet(bw.tabs.currentIndex())

    def duplicate_sheet(self) -> None:
        bw = self.active_book_window()
        if bw is not None:
            bw.duplicate_sheet(bw.tabs.currentIndex())

    def column_properties(self, col: int | None) -> None:
        t = self._sheet_target()
        if t is None:
            return
        bw, ws, cols = t
        col = col if col is not None else (cols[0] if cols else None)
        if col is None:
            return
        dlg = ColumnPropertiesDialog(ws, col, self)
        if dlg.exec():
            self.apply_column_properties(bw, col, dlg.values())

    def apply_column_properties(self, bw: BookWindow, col: int, values: dict) -> bool:
        ws = bw.worksheet()
        c = ws.columns[col]
        if values["name"] != c.name:
            try:
                bw.change("Rename Column", lambda: ws.rename_column(col, values["name"]),
                          recalc=False)
            except ValueError as exc:
                self._warn(str(exc))
                return False

        def fn():
            c.long_name, c.units, c.comments = (values["long_name"], values["units"],
                                                values["comments"])
            c.designation, c.width = values["designation"], values["width"]
        bw.change("Column Properties", fn, recalc=False)
        if values.get("formula", c.formula) != c.formula:
            bw.set_formula(col, values["formula"])
        return True

    def set_column_values_dialog(self) -> None:
        t = self._sheet_target()
        if t is None:
            return
        bw, ws, cols = t
        if not cols:
            bw.change("Add New Columns", lambda: ws.add_column("Y"), recalc=False)
            cols = [len(ws.columns) - 1]
        SetValuesDialog(ws, cols[0], lambda c, f, a, b: self.set_column_values(bw, c, f, a, b),
                        self).exec()

    def set_column_values(self, bw: BookWindow, col: int, formula: str,
                          first: int | None = None, last: int | None = None) -> None:
        """Set Column Values (raises ValueError with Origin-like messages on errors)."""
        ws = bw.worksheet()
        values = fx.evaluate_formula(ws, formula)
        lo = (first or 1) - 1
        hi = last or len(values)

        def fn():
            current = (ws.columns[col].values + [None] * ws.n_rows)[:ws.n_rows]
            new = [values[r] if lo <= r < hi else current[r] for r in range(ws.n_rows)]
            ws.set_column_values(col, new)
            ws.columns[col].formula = formula if (first is None and last is None) else ""
        bw.change("Set Column Values", fn, skip_col=col)
        self.message(f"Col({ws.columns[col].name}) = {formula}")

    # ================================================================== context menus
    def column_lock_menu(self, bw: BookWindow, col: int, parent: QMenu | None = None) \
            -> QMenu | None:
        """Origin's lock menu of an output column (Statistics on Rows, Frequency Counts):
        Recalculate mode, Recalculate, Change Parameters."""
        ws = bw.worksheet()
        target = recalc.sheet_op(bw.book, ws, col) if ws is not None else None
        if target is None:
            return None
        menu = QMenu("Recalculate", parent or self)
        menu.setIcon(icon(bw.lock_icon(target)))
        group = QActionGroup(menu)
        from rsstatchem.ui.origin.model import RECALC_MODES
        for key, text in RECALC_MODES.items():
            act = QAction(f"Mode: {text}", menu, checkable=True)
            act.setChecked(target.mode == key)
            act.triggered.connect(lambda _=False, k=key: self.set_report_mode(bw.book, target,
                                                                              k))
            group.addAction(act)
            menu.addAction(act)
        menu.addSeparator()
        menu.addAction("&Recalculate", lambda: self.recalculate_report(bw.book, target))
        menu.addAction("&Change Parameters...",
                       lambda: self.change_report_parameters(bw.book, target))
        return menu

    def column_menu(self, bw: BookWindow, pos) -> None:
        menu = QMenu(self)
        cols = bw.selected_columns()
        lock = self.column_lock_menu(bw, cols[0], menu) if cols else None
        if lock is not None:
            menu.addMenu(lock)
            menu.addSeparator()
        menu.addActions([self.act_cut, self.act_copy, self.act_paste])
        menu.addSeparator()
        menu.addActions([self.act_insert_column, self.act_delete_column])
        menu.addAction("Clear", self.clear_columns)
        menu.addSeparator()
        menu.addAction(self.act_add_column)
        set_as = menu.addMenu("Set As")
        set_as.addActions(list(self.act_desig.values()))
        menu.addAction(self.act_set_values)
        fill = menu.addMenu("Fill Column With")
        fill.addActions([self.act_fill_rows, self.act_fill_uniform, self.act_fill_normal])
        menu.addSeparator()
        plot = menu.addMenu(icon("g_line"), "Plot")
        for k in ("line", "scatter", "line_symbol", "yerror", "column", "bar", "area", "pie",
                  "histogram", "box", "double_y"):
            plot.addAction(self._plot_action(k))
        sort = menu.addMenu("Sort Column")
        sort.addActions([self.act_sort_asc, self.act_sort_desc])
        move = menu.addMenu("Move Column")
        move.addActions([self.act_move_left, self.act_move_right])
        menu.addSeparator()
        stats = menu.addMenu("Statistics")
        stats.addAction(self.act_stats_cols)
        stats.addAction("Frequency Counts...", lambda: self.stat_dialog("freq"))
        stats.addAction("Normality Test...", lambda: self.stat_dialog("normality"))
        menu.addSeparator()
        ws = bw.worksheet()
        spark = menu.addAction("Sparklines", lambda: self.toggle_sparklines(
            not (ws is not None and ws.sparklines)))
        spark.setCheckable(True)
        spark.setChecked(bool(ws is not None and ws.sparklines))
        menu.addAction(self.act_column_props)
        menu.exec(pos)

    def cell_menu(self, bw: BookWindow, pos) -> None:
        menu = QMenu(self)
        menu.addActions([self.act_cut, self.act_copy, self.act_copy_headers, self.act_paste])
        menu.addAction("Clear", self.clear_cells)
        menu.addSeparator()
        menu.addAction("Insert Rows", bw.insert_rows)
        menu.addAction("Delete Rows", bw.delete_rows)
        menu.addSeparator()
        plot = menu.addMenu(icon("g_line"), "Plot")
        for k in ("line", "scatter", "line_symbol", "column"):
            plot.addAction(self._plot_action(k))
        menu.addAction(self.act_set_values)
        menu.exec(pos)

    # ================================================================== plotting
    def plot(self, kind: str) -> OriginGraphWindow | None:
        """Plot the selected columns of the active worksheet in a new graph window. A
        selected block of cells (e.g. rows 3–10) plots only those rows."""
        mw = self.active_matrix_window()
        if mw is not None:   # the 2D Graphs toolbar also plots matrices
            mkind = {"contour": "contour", "heatmap": "heatmap",
                     "surface3d": "surface3d_colormap"}.get(kind)
            if mkind is None:
                self._warn(f"{PLOT_TYPES[kind][0]} needs worksheet columns; for a matrix use "
                           "Contour, Heatmap or 3D plots.", "Plot")
                return None
            return self.plot_matrix(mkind)
        bw = self.active_book_window()
        ws = bw.worksheet() if bw is not None else None
        if ws is None:
            self._warn("Activate a workbook and select the columns to plot.")
            return None
        cols = bw.selected_columns()
        rows = bw.selected_row_range()
        try:
            graph = plot_columns(bw.book, ws, cols, kind, self.project.unique_name("Graph"),
                                 rows=rows)
        except PlotError as exc:
            self._warn(str(exc), "Plot")
            return None
        folder = self.project.root.find_window(bw.book.name)
        gwin = self.new_graph(graph, folder=folder)
        self.message(f"Plot {PLOT_TYPES[kind][0]} of [{bw.book.name}]{ws.name}!"
                     f"({', '.join(ws.columns[c].name for c in cols)})"
                     f"{rows_text(*(rows or (None, None)))} → {graph.title}")
        return gwin

    def plot_matrix(self, kind: str) -> OriginGraphWindow | None:
        """Contour / heatmap / image / 3D surface of the active matrix sheet."""
        mw = self.active_matrix_window()
        ms = mw.sheet() if mw is not None else None
        if ms is None:
            self._warn("Activate a matrix window to plot it.")
            return None
        try:
            graph = plot_matrix(mw.book, ms, kind, self.project.unique_name("Graph"))
        except PlotError as exc:
            self._warn(str(exc), "Plot")
            return None
        gwin = self.new_graph(graph, folder=self.project.root.find_window(mw.book.name))
        self.message(f"Plot {MATRIX_PLOTS[kind][0]} of [{mw.book.name}]{ms.name} → "
                     f"{graph.title}")
        return gwin

    def add_plot_active(self) -> None:
        gwin = self.active_graph_window()
        if gwin is not None:
            self.add_plot_to_layer(gwin)

    def add_plot_to_layer(self, gwin: OriginGraphWindow) -> None:
        dlg = AddPlotDialog(self.project.books(), self)
        if not dlg.exec():
            return
        values = dlg.values()
        if values:
            self.add_columns_to_graph(gwin, values["book"], values["sheet"], values["cols"],
                                      values["kind"], values.get("rows"))

    def add_columns_to_graph(self, gwin: OriginGraphWindow, book: Workbook, ws: Worksheet,
                             cols: list[int], kind: str = "line_symbol",
                             rows: tuple[int, int] | None = None) -> bool:
        try:
            src = plot_columns(book, ws, cols, kind, gwin.graph.title, rows=rows)
        except PlotError as exc:
            self._warn(str(exc), "Add Plot to Layer")
            return False
        from rsstatchem.core.graph import edit
        layer = gwin.active_layer()

        def fn():
            empty = not any(layer.plots for layer in gwin.graph.layers)
            ids = edit.merge_plots(gwin.graph, layer, src)
            links = gwin.graph.meta.setdefault("links", {})
            for key, link in src.meta.get("links", {}).items():
                if key in gwin.graph.data:
                    links[key] = link
            new_ids = set(ids)
            for p, sp in zip([p for p in layer.plots if p.id in new_ids], src.layers[0].plots):
                p.options["source"] = sp.options.get("source")
            if empty:   # first plot in a new graph: take titles and auto scale
                layer.x.title, layer.y.title = src.layers[0].x.title, src.layers[0].y.title
                for axis in (layer.x, layer.y):
                    axis.min = axis.max = None
            update_data_ranges(gwin.graph)
        gwin.mutate(fn)
        return True

    # ================================================================== graph commands
    def _gw_call(self, method: str, *args) -> None:
        gwin = self.active_graph_window()
        if gwin is not None:
            getattr(gwin, method)(*args)

    def _gw_zoom(self, factor: float) -> None:
        gwin = self.active_graph_window()
        if gwin is not None:
            gwin.set_zoom(gwin.zoom * factor)

    def rescale(self) -> None:
        self._gw_call("rescale")

    def toggle_legend(self) -> None:
        gwin = self.active_graph_window()
        if gwin is None:
            return
        layer = gwin.active_layer()
        gwin.mutate(lambda: setattr(layer.legend, "visible", not layer.legend.visible), False)

    def _graph_props(self, what: tuple) -> None:
        gwin = self.active_graph_window()
        if gwin is None:
            return
        li = gwin.active_index()
        layer = gwin.active_layer()
        if what[0] == "layer":
            ref = ("layer", li)
        elif what[0] == "axis":
            ref = ("axis", li, what[1])
        elif what[0] == "legend":
            ref = ("legend", li)
        else:
            if not layer.plots:
                return
            ref = gwin.selected if gwin.selected[0] == "plot" else ("plot", li,
                                                                     layer.plots[0].id)
        gwin.open_properties(ref)

    def set_tool(self, tool: str, quiet: bool = False) -> None:
        if tool in self.tool_acts:
            self.tool_acts[tool].setChecked(True)
        gwin = self.active_graph_window()
        if gwin is None:
            return
        gwin.set_tool(None if tool in ("pointer", "reader") else tool)
        if tool == "reader":
            gwin.canvas.setCursor(Qt.CursorShape.CrossCursor)
            if not quiet:
                self.statusBar().showMessage("Data Reader: move over a data point to read "
                                             "its X and Y.", 5000)

    # ================================================================== analysis plumbing
    def analysis_inputs(self) -> tuple[list[InputChoice], int, OriginGraphWindow | None]:
        """Data sets for the analysis dialogs: the active graph layer's plots, or the Y
        columns of the active sheet (the selected ones first)."""
        gwin = self.active_graph_window()
        choices: list[InputChoice] = []
        if gwin is not None:
            layer = gwin.active_layer()
            for p in layer.plots if layer is not None else []:
                src = p.options.get("source") or {}
                if src.get("book") in self.project.windows and src.get("y") and \
                        not src.get("matrix"):
                    rows = tuple(src["rows"]) if src.get("rows") else None
                    choices.append(InputChoice(
                        f"[{src['book']}]{src['sheet']}!{src['y']}"
                        f"{rows_text(*(rows or (None, None)))}  ({p.name})", src["book"],
                        src["sheet"], src["y"], src.get("x"), p.id, rows))
            current = 0
            if gwin.selected[0] == "plot":
                current = next((i for i, c in enumerate(choices)
                                if c.plot_id == gwin.selected[2]), 0)
            return choices, current, gwin
        bw = self.active_book_window()
        ws = bw.worksheet() if bw is not None else None
        if ws is None:
            return [], 0, None
        sel = bw.selected_columns()
        rows = bw.selected_row_range()
        cols = [i for i, c in enumerate(ws.columns) if c.designation in ("Y", "Z", "N")
                and not c.is_text()]
        cols = [c for c in sel if c in cols] + [c for c in cols if c not in sel]
        for c in cols:
            x = ws.x_of(c)
            col = ws.columns[c]
            r = rows if c in sel else None   # the selected row range applies to the selection
            choices.append(InputChoice(
                f"[{bw.book.name}]{ws.name}!{col.name}" +
                (f'"{col.long_name}"' if col.long_name else "") +
                rows_text(*(r or (None, None))) +
                (f"  vs  {ws.columns[x].name}" if x is not None else "  vs  row"),
                bw.book.name, ws.name, col.name, ws.columns[x].name if x is not None else None,
                None, r))
        return choices, 0, None

    def resolve(self, choice: InputChoice) -> an.XYData:
        book = self.project.windows[choice.book]
        ws = book.sheet(choice.sheet)
        if not isinstance(ws, Worksheet):
            raise an.AnalysisError(f"Sheet {choice.sheet} not found in {choice.book}.")
        y = ws.index_of(choice.ycol)
        x = ws.index_of(choice.xcol) if choice.xcol else None
        return an.xy_data(book, ws, y, x, with_error=True, rows=choice.rows)

    def _need_inputs(self):
        choices, current, gwin = self.analysis_inputs()
        if not choices:
            self._warn("Select a Y column of a worksheet (or activate a graph with data plots) "
                       "first.")
            return None
        return choices, current, gwin

    def _run_in_book(self, book: Workbook, text: str, fn):
        bw = self.book_window(book)
        if bw is None:
            return fn()
        return bw.change(text, fn, recalc=False)

    # ================================================================== fitting
    def fit_dialog(self, kind: str) -> None:
        got = self._need_inputs()
        if got is None:
            return
        choices, current, gwin = got
        dlg = FitDialog(kind, choices, self, current, graph_available=gwin is not None)
        if not dlg.exec():
            return
        try:
            params = dlg.values()
        except ValueError as exc:
            self._warn(str(exc))
            return
        self.run_fit(kind, params, gwin)

    def run_fit(self, kind: str, params: dict, gwin: OriginGraphWindow | None = None) \
            -> an.FitOutcome | None:
        choice: InputChoice = params["input"]
        try:
            data = self.resolve(choice)
            outcome = self._run_in_book(data.book, "Fit", lambda: an.run_fit(
                data, params["model"], kind, weighting=params.get("weighting", "none"),
                params0=params.get("params0"), fixed=params.get("fixed"),
                bounds=params.get("bounds"), level=params.get("level", 0.95),
                conf_band=params.get("conf_band", True),
                pred_band=params.get("pred_band", False),
                residual_plot=params.get("residual_plot", True),
                output_fitted=params.get("output_fitted", False),
                mode=params.get("mode", "manual")))
        except (an.AnalysisError, ValueError, KeyError) as exc:
            self._warn(f"Fit failed: {exc}", "Fitting")
            return None
        recalc.stamp(self.project, outcome.report)
        self.result(outcome.log)
        if gwin is not None and params.get("to_graph", True):
            self._add_fit_to_graph(gwin, outcome, choice)
        else:
            bw = self.book_window(data.book)
            if bw is not None:
                bw.show_sheet(outcome.report)
        return outcome

    def _add_fit_to_graph(self, gwin: OriginGraphWindow, outcome: an.FitOutcome,
                          choice: InputChoice) -> None:
        book, curve, res = outcome.data.book, outcome.curve, outcome.result
        li = gwin.active_index()
        for i, layer in enumerate(gwin.graph.layers):
            if any(p.id == choice.plot_id for p in layer.plots):
                li = i

        def fn():
            add_xy_plot(gwin.graph, li, outcome.curve_x, outcome.curve_y, outcome.curve_name,
                        "line", FIT_COLOR,
                        link={"book": book.name, "sheet": curve.name, "x": curve.columns[0].name,
                              "y": curve.columns[1].name},
                        source={"book": book.name, "sheet": curve.name,
                                "x": curve.columns[0].name, "y": curve.columns[1].name})
            lines = [f"Equation  {res.model.formula}"]
            for i, name in enumerate(res.names):
                err = f" ± {an.num(res.errors[i])}" if res.free[i] else " (fixed)"
                lines.append(f"{name}  {an.num(res.values[i])}{err}")
            lines.append(f"R-Square (COD)  {an.num(res.r2)}")
            lines.append(f"Adj. R-Square  {an.num(res.adj_r2)}")
            note = Annotation(kind="text", text="\n".join(lines), x=0.50, y=0.05,
                              frame=True)
            note.font.size = 8.0
            note.fill.visible = True
            note.fill.color = "#ffffff"
            gwin.graph.layers[li].annotations.append(note)
        gwin.mutate(fn)

    # ================================================================== signal / maths
    def _input_dialog(self, title: str, fields: list[tuple], note: str = ""):
        got = self._need_inputs()
        if got is None:
            return None
        choices, current, gwin = got
        options = {i: c.label for i, c in enumerate(choices)}
        dlg = ParamDialog(title, [("input", "Input:", "choice", current, options)] + fields,
                          self, note)
        if not dlg.exec():
            return None
        values = dlg.values()
        values["input"] = choices[values["input"]]
        return values, gwin

    def smooth_dialog(self) -> None:
        got = self._input_dialog("Smooth", [
            ("method", "Method:", "choice", "savgol", an.sig.SMOOTHERS),
            ("window", "Points of Window:", "int", 7, (2, 10001)),
            ("polyorder", "Polynomial Order:", "int", 2, (0, 10))])
        if got:
            self.run_column_op("smooth", got[0], got[1])

    def differentiate_dialog(self) -> None:
        got = self._input_dialog("Differentiate", [("order", "Derivative Order:", "int", 1,
                                                    (1, 3))])
        if got:
            self.run_column_op("differentiate", got[0], got[1])

    def integrate_dialog(self) -> None:
        got = self._input_dialog("Integrate", [("baseline", "Baseline:", "choice", "zero",
                                                an.maths.BASELINES)])
        if got:
            self.run_column_op("integrate", got[0], got[1])

    def normalize_dialog(self) -> None:
        got = self._input_dialog("Normalize", [("method", "Normalize Method:", "choice", "max",
                                                an.maths.NORMALIZE)])
        if got:
            self.run_column_op("normalize", got[0], got[1])

    def baseline_dialog(self) -> None:
        got = self._input_dialog("Baseline", [
            ("method", "Method:", "choice", "als", an.sig.BASELINE_METHODS),
            ("order", "Polynomial Order:", "int", 2, (0, 9))])
        if got:
            self.run_column_op("baseline", got[0], got[1])

    def run_column_op(self, op: str, values: dict, gwin: OriginGraphWindow | None = None) \
            -> int | None:
        """Signal processing / mathematics that output a new column in the source sheet
        (plotted into the graph when started from one)."""
        try:
            data = self.resolve(values["input"])
            fn = {"smooth": lambda: an.smooth_column(data, values.get("method", "savgol"),
                                                     values.get("window", 7),
                                                     values.get("polyorder", 2)),
                  "differentiate": lambda: an.differentiate_column(data, values.get("order", 1)),
                  "integrate": lambda: an.integrate_column(data, values.get("baseline", "zero")),
                  "normalize": lambda: an.normalize_column(data, values.get("method", "max")),
                  "baseline": lambda: an.baseline_columns(data, values.get("method", "als"),
                                                          values.get("order", 2))}[op]
            idx, text = self._run_in_book(data.book, op.capitalize(), fn)
        except (an.AnalysisError, ValueError, KeyError) as exc:
            self._warn(str(exc))
            return None
        self.result(text)
        if gwin is not None:
            colors = {"smooth": "#FF0000", "baseline": "#0000FF"}
            ws = data.sheet
            key_y = column_key(data.book.name, ws.name, ws.columns[idx].name)
            if op in ("smooth", "baseline"):
                self._plot_new_column(gwin, data, idx, colors.get(op, "#FF0000"), key_y)
        return idx

    def _plot_new_column(self, gwin, data: an.XYData, idx: int, color: str, key: str) -> None:
        ws = data.sheet
        x = data.xcol
        li = gwin.active_index()

        def fn():
            p = add_xy_plot(gwin.graph, li, ws.columns[x].numeric() if x is not None else
                            np.arange(1, ws.n_rows + 1, dtype=float),
                            ws.columns[idx].numeric(), ws.columns[idx].title, "line", color,
                            link={"book": data.book.name, "sheet": ws.name,
                                  "x": ws.columns[x].name if x is not None else None,
                                  "y": ws.columns[idx].name},
                            source={"book": data.book.name, "sheet": ws.name,
                                    "x": ws.columns[x].name if x is not None else None,
                                    "y": ws.columns[idx].name})
            del p
        gwin.mutate(fn)

    def fft_dialog(self) -> None:
        got = self._input_dialog("FFT", [("window", "Window:", "choice", "hann",
                                          an.sig.WINDOWS)])
        if not got:
            return
        values, _ = got
        try:
            data = self.resolve(values["input"])
            sheet, text = self._run_in_book(data.book, "FFT", lambda: an.fft_sheet(
                data, values["window"]))
        except (an.AnalysisError, ValueError) as exc:
            self._warn(str(exc))
            return
        self.result(text)
        bw = self.book_window(data.book)
        if bw is not None:
            bw.show_sheet(sheet)
            if sheet is bw.worksheet():
                bw.select_columns([1])
                self.plot("line")

    def peaks_dialog(self) -> None:
        got = self._input_dialog("Find Peaks", [
            ("direction", "Direction:", "choice", "positive",
             {"positive": "Positive", "negative": "Negative"}),
            ("prominence", "Minimum Prominence (% of Y range):", "float", 5.0,
             (0.0, 100.0, 1))])
        if not got:
            return
        values, gwin = got
        try:
            data = self.resolve(values["input"])
            report, peaks, text = self._run_in_book(data.book, "Find Peaks", lambda:
                                                    an.peaks_report(data, values["direction"],
                                                                    values["prominence"] / 100))
        except (an.AnalysisError, ValueError) as exc:
            self._warn(str(exc))
            return
        self.result(text)
        if gwin is not None and peaks:
            li = gwin.active_index()

            def fn():
                p = add_xy_plot(gwin.graph, li, [pk["x"] for pk in peaks],
                                [pk["y"] for pk in peaks], "Peaks", "scatter", "#FF0000")
                p.symbol.shape, p.symbol.size = "v", 7.0
            gwin.mutate(fn)
        else:
            bw = self.book_window(data.book)
            if bw is not None:
                bw.show_sheet(report)

    def multi_peak_dialog(self) -> None:
        got = self._input_dialog("Multiple Peak Fit", [
            ("n", "Number of Peaks (0 = all found):", "int", 0, (0, 30)),
            ("shape", "Peak Function:", "choice", "gauss", an.sig.SHAPES),
            ("baseline", "Baseline:", "choice", "constant",
             {"none": "None", "constant": "Constant", "linear": "Linear"})])
        if not got:
            return
        values, gwin = got
        try:
            data = self.resolve(values["input"])
            outcome = self._run_in_book(data.book, "Multiple Peak Fit", lambda: an.multi_peak_fit(
                data, values["n"] or None, values["shape"], values["baseline"]))
        except (an.AnalysisError, ValueError) as exc:
            self._warn(str(exc))
            return
        self.result(outcome.log)
        if gwin is not None:
            self._add_curve_only(gwin, outcome)
        else:
            bw = self.book_window(data.book)
            if bw is not None:
                bw.show_sheet(outcome.report)

    def _add_curve_only(self, gwin, outcome: an.FitOutcome) -> None:
        book, curve = outcome.data.book, outcome.curve
        li = gwin.active_index()
        gwin.mutate(lambda: add_xy_plot(
            gwin.graph, li, outcome.curve_x, outcome.curve_y, outcome.curve_name, "line",
            FIT_COLOR, link={"book": book.name, "sheet": curve.name,
                             "x": curve.columns[0].name, "y": curve.columns[1].name}))

    # ================================================================== statistics
    def selection_ranges(self, fallback: bool = True) -> list[str]:
        """The selected columns of the active worksheet as ranges (with the selected row
        range), e.g. ``["[Book1]Sheet1!B[3:10]", "[Book1]Sheet1!C[3:10]"]``; the Y columns
        when nothing is selected (unless ``fallback`` is False)."""
        bw = self.active_book_window()
        ws = bw.worksheet() if bw is not None else None
        if ws is None:
            return []
        cols = bw.selected_columns() or ([i for i, c in enumerate(ws.columns)
                                          if c.designation == "Y"] if fallback else [])
        rows = bw.selected_row_range()
        return [range_text(bw.book.name, ws.name, ws.columns[c].name,
                           *(rows or (None, None))) for c in cols]

    def range_sources(self) -> list[tuple[str, list[tuple[str, str]]]]:
        """Every worksheet column for the dialogs' range menus: (sheet, [(range, label)])."""
        out = []
        for book in self.project.books():
            for s in book.sheets:
                if isinstance(s, Worksheet):
                    out.append((f"[{book.name}]{s.name}",
                                [(range_text(book.name, s.name, c.name),
                                  s.label(i) + (f"  {c.long_name}" if c.long_name else ""))
                                 for i, c in enumerate(s.columns)]))
        return out

    def stat_dialog(self, key: str, settings: dict | None = None):
        """Open the Statistics dialog ``key`` and run it on OK."""
        dlg = sd.make_dialog(key, self, settings, self.selection_ranges())
        if not dlg.exec():
            return None
        return self.run_stat(key, dlg.values())

    def run_stat(self, key: str, settings: dict):
        """Run a Statistics analysis with dialog ``settings``; returns the report (or the
        output worksheet / new column indices) or None after an error."""
        books = self.project.data_windows()
        mode = settings.get("mode", "manual")
        values = {k: v for k, v in settings.items() if k != "mode"}
        try:
            if key == "rows":
                first = st.read_ranges(books, values.get("ranges", []))[0]
                ws, added, text = self._run_in_book(
                    first.book, "Statistics on Rows",
                    lambda: st.stats_on_rows(books, values, mode=mode))
                recalc.stamp(self.project, SheetOp(ws, ws.ops[-1]))
                self.result(text)
                bw = self.book_window(first.book)
                if bw is not None:
                    bw.refresh()
                return added
            if key == "freq":
                first = st.read_range(books, values.get("range", ""))
                out, text = self._run_in_book(
                    first.book, "Frequency Counts",
                    lambda: st.frequency_counts(books, values, mode=mode))
                recalc.stamp(self.project, SheetOp(out, out.ops[0]))
                self.result(text)
                bw = self.book_window(first.book)
                if bw is not None:
                    bw.show_sheet(out)
                return out
            first = st.read_range(books, (st.range_strings(values) or [""])[0])
            report, text = self._run_in_book(first.book, sd.spec(key, [])[0],
                                             lambda: st.run(key, books, values, mode=mode))
        except (st.StatError, ValueError) as exc:
            self._warn(str(exc), "Statistics")
            return None
        recalc.stamp(self.project, report)
        self.result(text)
        bw = self.book_window(first.book)
        if bw is not None:
            bw.refresh()
            bw.show_sheet(report)
        return report

    # direct commands (toolbar / context menus / scripts) with the default settings
    def statistics_on_columns(self) -> None:
        self.run_stat("desc", {"ranges": self.selection_ranges()})

    def statistics_on_rows(self) -> None:
        self.run_stat("rows", {"ranges": self.selection_ranges()})

    def correlation(self) -> None:
        self.run_stat("correlation", {"ranges": self.selection_ranges()})

    def normality(self) -> None:
        self.run_stat("normality", {"ranges": self.selection_ranges()})

    # ================================================================== sparklines
    def toggle_sparklines(self, on: bool | None = None) -> None:
        bw = self.active_book_window()
        if bw is None or bw.worksheet() is None:
            return
        on = self.act_sparklines.isChecked() if on is None else on
        bw.set_sparklines(on)
        self.act_sparklines.setChecked(on)

    # ================================================================== matrices
    def matrix_view_changed(self, mw: MatrixWindow) -> None:
        if mw is not self.active_matrix_window():
            return
        self.book_tree.clear()
        book = mw.book
        self.om_dock.setWindowTitle("Object Manager - " + book.name)
        root = QTreeWidgetItem([book.title])
        root.setIcon(0, icon("e_matrix"))
        self.book_tree.addTopLevelItem(root)
        for si, ms in enumerate(book.sheets):
            item = QTreeWidgetItem([f"{ms.name}  ({ms.n_rows}×{ms.n_cols})"])
            item.setIcon(0, icon("e_matrix"))
            item.setData(0, Qt.ItemDataRole.UserRole, (si, -1))
            root.addChild(item)
        root.setExpanded(True)
        ms = mw.sheet()
        if ms is not None:
            self.range_label.setText(f"[{book.name}]{ms.name}  {ms.n_rows}×{ms.n_cols}  "
                                     f"X {ms.x1:g}…{ms.x2:g}  Y {ms.y1:g}…{ms.y2:g}")

    def matrix_menu(self, mw: MatrixWindow, pos) -> None:
        menu = QMenu(self)
        menu.addActions([self.act_copy, self.act_paste])
        menu.addSeparator()
        menu.addActions([self.act_m_values, self.act_m_dims])
        menu.addSeparator()
        plot = menu.addMenu(icon("g_contour"), "Plot")
        for k in MATRIX_PLOTS:
            plot.addAction(self._matrix_plot_action(k))
        menu.addSeparator()
        menu.addActions([self.act_image_mode, self.act_show_xy])
        menu.exec(pos)

    def _matrix_target(self) -> tuple[MatrixWindow, MatrixSheet] | None:
        mw = self.active_matrix_window()
        ms = mw.sheet() if mw is not None else None
        if ms is None:
            self._warn("Activate a matrix window first.")
            return None
        return mw, ms

    def set_matrix_values(self, mw: MatrixWindow, formula: str) -> None:
        """Set Matrix Values (raises ValueError with a message on errors)."""
        ms = mw.sheet()
        values = mx.evaluate(formula, ms.data.shape, (ms.x1, ms.x2, ms.y1, ms.y2))

        def fn():
            ms.data = values
            ms.formula = formula
        mw.change("Set Matrix Values", fn)
        self.message(f"[{mw.book.name}]{ms.name}: Mat = {formula}")

    def matrix_values_dialog(self) -> None:
        t = self._matrix_target()
        if t is not None:
            mw, ms = t
            MatrixValuesDialog(ms, lambda f: self.set_matrix_values(mw, f), self).exec()

    def matrix_dimensions_dialog(self) -> None:
        t = self._matrix_target()
        if t is None:
            return
        mw, ms = t
        dlg = MatrixDimensionsDialog(ms, self)
        if dlg.exec():
            self.set_matrix_dimensions(mw, dlg.values())

    def set_matrix_dimensions(self, mw: MatrixWindow, v: dict) -> None:
        ms = mw.sheet()

        def fn():
            if (v["rows"], v["cols"]) != (ms.n_rows, ms.n_cols):
                ms.set_dimensions(v["rows"], v["cols"])
            for k in ("x1", "x2", "y1", "y2", "x_name", "x_units", "y_name", "y_units",
                      "long_name", "units"):
                if k in v:
                    setattr(ms, k, v[k])
        mw.change("Set Dimensions/Labels", fn)

    def matrix_geometry(self, op: str) -> None:
        t = self._matrix_target()
        if t is None:
            return
        mw, ms = t

        def fn():
            mapping = (ms.x1, ms.x2, ms.y1, ms.y2)
            if op == "transpose":
                data, mapping = mx.transpose(ms.data, mapping)
                ms.x_name, ms.y_name = ms.y_name, ms.x_name
                ms.x_units, ms.y_units = ms.y_units, ms.x_units
            elif op in ("flip_h", "flip_v"):
                data, mapping = mx.flip(ms.data, mapping,
                                        "horizontal" if op == "flip_h" else "vertical")
            else:
                data, mapping = mx.rotate(ms.data, mapping, {"rot90": 90, "rot270": 270,
                                                             "rot180": 180}[op])
                if op != "rot180":
                    ms.x_name, ms.y_name = ms.y_name, ms.x_name
                    ms.x_units, ms.y_units = ms.y_units, ms.x_units
            ms.data = data
            ms.x1, ms.x2, ms.y1, ms.y2 = mapping
        mw.change({"transpose": "Transpose", "flip_h": "Flip Horizontal",
                   "flip_v": "Flip Vertical"}.get(op, "Rotate"), fn)

    def matrix_resample_dialog(self, op: str) -> None:
        t = self._matrix_target()
        if t is None:
            return
        dlg = ParamDialog("Expand" if op == "expand" else "Shrink", [
            ("cols", "Column Factor:", "int", 2, (1, 100)),
            ("rows", "Row Factor:", "int", 2, (1, 100))], self,
            "Expand interpolates (bilinear) between the cells; Shrink averages blocks of "
            "cells." if op == "expand" else "Shrink averages blocks of row × column cells.")
        if dlg.exec():
            v = dlg.values()
            self.matrix_resample(t[0], op, v["cols"], v["rows"])

    def matrix_resample(self, mw: MatrixWindow, op: str, cf: int, rf: int) -> None:
        ms = mw.sheet()

        def fn():
            if op == "expand":
                ms.data = mx.expand(ms.data, cf, rf)
            else:
                shape = ms.data.shape
                ms.data = mx.shrink(ms.data, cf, rf)
                ms.x1, ms.x2, ms.y1, ms.y2 = mx.shrink_mapping(
                    (ms.x1, ms.x2, ms.y1, ms.y2), shape, ms.data.shape, cf, rf)
        mw.change("Expand" if op == "expand" else "Shrink", fn)

    def matrix_add_sheet(self) -> None:
        t = self._matrix_target()
        if t is not None:
            t[0].add_sheet()

    def toggle_image_mode(self) -> None:
        mw = self.active_matrix_window()
        if mw is not None:
            mw.set_image_mode(self.act_image_mode.isChecked())

    def toggle_show_xy(self) -> None:
        mw = self.active_matrix_window()
        if mw is not None:
            mw.set_show_xy(self.act_show_xy.isChecked())

    def matrix_to_worksheet(self) -> BookWindow | None:
        """Matrix > Convert to Worksheet: X, Y, Z columns in a new workbook."""
        t = self._matrix_target()
        if t is None:
            return None
        mw, ms = t
        x, y, z = mx.to_xyz(ms.data, (ms.x1, ms.x2, ms.y1, ms.y2), drop_missing=True)
        book = Workbook.new(self.project.unique_name("Book"), 0, max(len(z), 1))
        ws = book.sheets[0]
        for title, desig, vals in ((ms.x_title, "X", x), (ms.y_title, "Y", y),
                                   (ms.z_title, "Z", z)):
            c = ws.add_column(desig)
            c.long_name = title
            ws.set_column_values(len(ws.columns) - 1, list(vals))
        book.long_name = f"{mw.book.name} {ms.name} (XYZ)"
        bw = self.new_book(book, folder=self.project.root.find_window(mw.book.name))
        self.message(f"Converted [{mw.book.name}]{ms.name} to XYZ columns in {book.name}")
        return bw

    def convert_to_matrix_direct(self, values: dict | None = None) -> MatrixWindow | None:
        """Worksheet > Convert to Matrix > Direct: worksheet rows → matrix rows, selected
        columns → matrix columns."""
        bw = self.active_book_window()
        ws = bw.worksheet() if bw is not None else None
        if ws is None:
            self._warn("Activate a worksheet and select the columns to convert.")
            return None
        cols = [c for c in bw.selected_columns() if not ws.columns[c].is_text()] or \
            [i for i, c in enumerate(ws.columns) if not c.is_text()]
        if values is None:
            dlg = ParamDialog("Convert to Matrix - Direct", [
                ("first_col_y", "First column contains Y values", "bool",
                 ws.columns[cols[0]].designation == "X" if cols else False),
                ("long_names_x", "Long Names contain X values", "bool", False)], self,
                "Each selected column becomes a matrix column; each worksheet row a matrix "
                "row.")
            if not dlg.exec():
                return None
            values = dlg.values()
        rows = bw.selected_row_range()
        lo, hi = (rows[0] - 1, rows[1]) if rows else (0, ws.data_rows())
        yv = None
        if values.get("first_col_y") and len(cols) > 1:
            yv = ws.columns[cols[0]].numeric(hi)[lo:hi]
            cols = cols[1:]
        xv = mx.number_labels([ws.columns[c].long_name for c in cols]) \
            if values.get("long_names_x") else None
        try:
            data, mapping = mx.from_columns([ws.columns[c].numeric(hi)[lo:hi] for c in cols],
                                            yv, xv)
        except mx.MatrixError as exc:
            self._warn(str(exc))
            return None
        return self._new_matrix_from(data, mapping, f"{bw.book.name} {ws.name} (direct)",
                                     ws, cols, bw.book)

    def convert_to_matrix_xyz(self, values: dict | None = None) -> MatrixWindow | None:
        """Worksheet > Convert to Matrix > XYZ Gridding (regular data pivoted, scattered data
        interpolated)."""
        bw = self.active_book_window()
        ws = bw.worksheet() if bw is not None else None
        if ws is None:
            self._warn("Activate a worksheet with X, Y and Z columns.")
            return None
        sel = bw.selected_columns()
        z = next((c for c in sel if ws.columns[c].designation == "Z"), None)
        if z is None:
            z = next((i for i, c in enumerate(ws.columns) if c.designation == "Z"), None)
        if z is None and len(sel) >= 3:
            z = sel[2]
        if z is None:
            self._warn("Select X, Y and Z columns (set the Z column as Z).")
            return None
        y = ws.y_of(z) if ws.y_of(z) is not None else (sel[1] if len(sel) >= 3 else None)
        x = ws.x_of(y) if y is not None else None
        if x is None or y is None:
            self._warn("No X and Y columns to the left of the Z column.")
            return None
        if values is None:
            dlg = ParamDialog("Convert to Matrix - XYZ Gridding", [
                ("method", "Gridding Method:", "choice", "auto", mx.GRIDDING),
                ("cols", "Columns (for scattered data):", "int", 20, (2, 2000)),
                ("rows", "Rows (for scattered data):", "int", 20, (2, 2000))], self,
                f"X = {ws.columns[x].name}, Y = {ws.columns[y].name}, Z = "
                f"{ws.columns[z].name}. Data on a regular X×Y grid are converted directly.")
            if not dlg.exec():
                return None
            values = dlg.values()
        rows = bw.selected_row_range()
        lo, hi = (rows[0] - 1, rows[1]) if rows else (0, ws.data_rows())
        try:
            data, mapping, regular = mx.from_xyz(
                ws.columns[x].numeric(hi)[lo:hi], ws.columns[y].numeric(hi)[lo:hi],
                ws.columns[z].numeric(hi)[lo:hi], values.get("cols", 20),
                values.get("rows", 20), values.get("method", "auto"))
        except mx.MatrixError as exc:
            self._warn(str(exc))
            return None
        mwin = self._new_matrix_from(data, mapping, f"{bw.book.name} {ws.name} (XYZ)", ws,
                                     [z], bw.book, (ws.columns[x], ws.columns[y]))
        self.message(f"XYZ Gridding ({'regular' if regular else mx.GRIDDING.get(values.get('method', 'auto'))}) "
                     f"→ {mwin.book.name}: {data.shape[0]}×{data.shape[1]}")
        return mwin

    def _new_matrix_from(self, data, mapping, long_name: str, ws: Worksheet, cols, src_book,
                         xy_cols=None) -> MatrixWindow:
        mb = MatrixBook(self.project.unique_name("MBook"), long_name,
                        [MatrixSheet("MSheet1", np.asarray(data, float), *mapping)])
        ms = mb.sheets[0]
        zc = ws.columns[cols[0]] if len(cols) == 1 else None
        if zc is not None:
            ms.long_name, ms.units = zc.long_name or zc.name, zc.units
        if xy_cols is not None:
            ms.x_name, ms.x_units = xy_cols[0].long_name or xy_cols[0].name, xy_cols[0].units
            ms.y_name, ms.y_units = xy_cols[1].long_name or xy_cols[1].name, xy_cols[1].units
        return self.new_matrix(mb, folder=self.project.root.find_window(src_book.name))

    # ================================================================== import / export
    def _start_dir(self) -> str:
        return str(self._setting("last_dir", "") or
                   (self.ctx.settings.value("last_dir", str(Path.home()))
                    if self.ctx is not None else str(Path.home())))

    def _remember_dir(self, path: str) -> None:
        self._set_setting("last_dir", str(Path(path).parent))

    def import_csv_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import Single ASCII", self._start_dir(),
                                              "Text / CSV (*.csv *.txt *.dat *.tsv *.asc);;"
                                              "All files (*)")
        if path:
            self.import_file(path)

    def import_excel_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import Excel", self._start_dir(),
                                              "Excel (*.xlsx *.xlsm *.xls);;All files (*)")
        if not path:
            return
        from rsstatchem.io import files
        sheet = 0
        try:
            names = files.excel_sheets(path)
        except Exception:  # noqa: BLE001 - let read_excel report the problem
            names = []
        if len(names) > 1:
            name, ok = QInputDialog.getItem(self, "Import Excel", "Sheet:", names, 0, False)
            if not ok:
                return
            sheet = name
        self.import_file(path, sheet)

    def import_file(self, path: str, excel_sheet=0) -> BookWindow | None:
        """Import CSV/TXT or Excel into the active workbook when it is empty (Origin's
        default), else into a new workbook; header → Long Name."""
        from rsstatchem.io import files
        try:
            if Path(path).suffix.lower() in (".xlsx", ".xlsm", ".xls"):
                df = files.read_excel(path, excel_sheet)
            else:
                enc = files.detect_encoding(path)
                delim, dec = files.sniff_text(path, enc)
                df = files.read_text(path, delim, dec, True, enc)
        except Exception as exc:  # noqa: BLE001 - any reader error is reported
            self._warn(f"Could not import {Path(path).name}:\n{exc}", "Import")
            return None
        self._remember_dir(path)
        bw = self.import_frame(df, Path(path).stem)
        self.message(f"Imported {Path(path).name}: {df.shape[1]} columns × {df.shape[0]} rows "
                     f"→ {bw.book.name}")
        return bw

    def import_frame(self, df, name: str) -> BookWindow:
        bw = self.active_book_window()
        if bw is None or not bw.book.is_empty():
            bw = self.new_book()
        book = bw.book
        ws = bw.worksheet() or book.sheets[0]

        def fn():
            ws.columns.clear()
            ws.set_rows(max(len(df), 1))
            for j, col in enumerate(df.columns):
                c = ws.add_column("X" if j == 0 else "Y")
                c.long_name = str(col)
                ws.set_column_values(j, [None if v is None or (isinstance(v, float) and
                                                               math.isnan(v)) else
                                         (float(v) if isinstance(v, (int, float, np.integer,
                                                                     np.floating))
                                          and not isinstance(v, bool) else str(v))
                                         for v in df[col].tolist()])
            ws.name = name[:31] or ws.name
            book.long_name = name
        bw.change("Import", fn)
        self.subs[book.name].setWindowTitle(book.title)
        self.explorer.fill_windows()
        return bw

    def export_ascii(self) -> None:
        bw = self.active_book_window()
        ws = bw.worksheet() if bw is not None else None
        if ws is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export ASCII",
                                              str(Path(self._start_dir()) / f"{ws.name}.csv"),
                                              "CSV (*.csv);;Text (*.txt)")
        if path:
            self.write_ascii(ws, path)

    def write_ascii(self, ws: Worksheet, path: str) -> None:
        import csv
        n = ws.data_rows()
        delim = "\t" if path.lower().endswith(".txt") else ","
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, delimiter=delim)
            w.writerow([c.long_name or c.name for c in ws.columns])
            if any(c.units for c in ws.columns):
                w.writerow([c.units for c in ws.columns])
            for r in range(n):
                w.writerow(ws.columns[c].texts(n)[r] for c in range(len(ws.columns)))
        self._remember_dir(path)
        self.message(f"Exported {ws.name} to {path}")

    def export_graph(self) -> None:
        gwin = self.active_graph_window()
        if gwin is None:
            self._warn("Activate a graph window to export it.")
            return
        gwin.export()

    # ================================================================== projects
    def sync_models(self) -> None:
        """Copy live window state (graphs after undo, notes) into the project model."""
        for name, sub in self.subs.items():
            w = sub.widget()
            obj = self.project.windows.get(name)
            if isinstance(w, OriginGraphWindow) and isinstance(obj, GraphDoc):
                obj.graph = w.graph

    def view_state(self) -> dict:
        state = {"active": self.active_name(),
                 "current_folder": self.explorer.current.name if self.explorer.current else ""}
        geo = {}
        for name, sub in self.subs.items():
            g = sub.geometry()
            geo[name] = [g.x(), g.y(), g.width(), g.height(), int(sub.isMaximized())]
        state["geometry"] = geo
        return state

    def save_project(self) -> bool:
        if not self.project.path:
            return self.save_project_as()
        return self.save_to(self.project.path)

    def save_project_as(self) -> bool:
        start = str(Path(self._start_dir()) / f"{self.project.name}{EXTENSION}")
        path, _ = QFileDialog.getSaveFileName(self, "Save Project As", start,
                                              f"Project (*{EXTENSION})")
        if not path:
            return False
        if not path.lower().endswith(EXTENSION):
            path += EXTENSION
        return self.save_to(path)

    def save_to(self, path: str) -> bool:
        self.sync_models()
        try:
            save_project(self.project, path, self.view_state())
        except OSError as exc:
            self._warn(f"Could not save the project:\n{exc}", "Save Project")
            return False
        self.project.path = str(path)
        self.project.name = Path(path).stem
        self.project.root.name = self.project.name
        self.explorer.rebuild()
        self._remember_dir(path)
        if self.ctx is not None:
            self.ctx.add_recent(path)
        self.set_modified(False)
        self.message(f"Project saved: {path}")
        return True

    def open_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open", self._start_dir(), PROJECT_FILTER)
        if path:
            self.open_path(path)

    def open_path(self, path: str) -> bool:
        if path.lower().endswith(GRAPH_EXT):
            try:
                graph = load_graph(path)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self._warn(f"Could not open graph:\n{exc}", "Open")
                return False
            graph.title = Path(path).stem if Path(path).stem not in self.project.windows \
                else self.project.unique_name("Graph")
            self.new_graph(graph)
            self._remember_dir(path)
            if self.ctx is not None:
                self.ctx.add_recent(path)
            self.message(f"Opened graph {path}")
            return True
        if path.lower().endswith((".csv", ".txt", ".dat", ".tsv", ".xlsx", ".xls", ".xlsm")):
            return self.import_file(path) is not None
        try:
            project, view = load_project(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self._warn(f"Could not open the project:\n{exc}", "Open Project")
            return False
        if not self.confirm_discard():
            return False
        self.load(project, view)
        self._remember_dir(path)
        if self.ctx is not None:
            self.ctx.add_recent(path)
        self.message(f"Project opened: {path}")
        return True

    def confirm_discard(self) -> bool:
        """Ask to save a modified project; True when it is fine to replace/close it."""
        if not self.modified or self.project.is_blank():
            return True
        answer = QMessageBox.question(
            self, APP_TITLE, f"Save changes to the project {self.project.name}?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel)
        if answer == QMessageBox.StandardButton.Cancel:
            return False
        if answer == QMessageBox.StandardButton.Save:
            return self.save_project()
        return True

    def _clear_windows(self) -> None:
        self.closing_all = True
        for name in list(self.subs):
            self._delete_window(name)
        self.closing_all = False
        self.results.clear()
        self.messages.clear()

    def load(self, project: Project, view: dict | None = None) -> None:
        self._clear_windows()
        self.project = project
        view = view or {}
        self.explorer.set_project(project)
        for folder in list(project.root.walk()):
            for name in list(folder.windows):
                obj = project.windows[name]
                hidden = name in project.hidden
                if isinstance(obj, Workbook):
                    self.new_book(obj, show=not hidden, folder=folder)
                elif isinstance(obj, MatrixBook):
                    self.new_matrix(obj, show=not hidden, folder=folder)
                elif isinstance(obj, GraphDoc):
                    self.new_graph(obj.graph, show=not hidden, folder=folder, name=name)
                else:
                    self.new_notes(obj, show=not hidden, folder=folder)
        for name, g in (view.get("geometry") or {}).items():
            sub = self.subs.get(name)
            if sub is not None and len(g) >= 4:
                sub.setGeometry(g[0], g[1], g[2], g[3])
        for text in project.results_log:
            self.results.appendPlainText(text + "\n")
        folder = next((f for f in project.root.walk()
                       if f.name == view.get("current_folder")), project.root)
        self.explorer.current = folder
        self.explorer.rebuild()
        self._folder_changed(folder)
        active = view.get("active")
        if active in self.subs and active not in project.hidden:
            self.mdi.setActiveSubWindow(self.subs[active])
        self.set_modified(False)

    def new_project(self) -> None:
        if not self.confirm_discard():
            return
        self.load(Project())
        self.new_book()
        self.set_modified(False)

    def close_project(self) -> None:
        self.new_project()

    # ================================================================== options / help
    def options_dialog(self) -> None:
        dlg = ParamDialog("Options", [
            ("rows", "Rows in a new worksheet:", "int", self.default_rows(), (1, 1_000_000)),
            ("folder_only", "Show windows in the active folder only", "bool",
             self.act_folder_only.isChecked())], self)
        if dlg.exec():
            v = dlg.values()
            self._set_setting("rows", v["rows"])
            self.act_folder_only.setChecked(v["folder_only"])
            self._folder_filter()

    def getting_started(self) -> None:
        text = (
            f"RS-StatChem {__version__} — Graph & Analysis (OriginLab-style workflow)\n\n"
            "1. Type or paste data into a workbook (Book1). Paste from Excel with a header "
            "row: the header goes into Long Name.\n"
            "2. Fill the header rows: Long Name, Units (used for axis titles and legends) and "
            "F(x)= for a column formula, e.g. col(A)^2 or sin(i/10).\n"
            "3. Right-click a column header > Set As: X, Y, Y Error, Label… Each Y is plotted "
            "against the X column on its left.\n"
            "4. Select the Y column(s), then use the Plot menu or the 2D Graphs toolbar at the "
            "bottom. Double-click an axis, plot or legend in the graph to edit it.\n"
            "5. Analysis > Fitting > Linear Fit (from the graph, the fitted line is added to "
            "it; the report sheet FitLinear1 and the curve data go into the workbook).\n"
            "6. Statistics > Descriptive Statistics > Statistics on Columns… (dialogs for "
            "normality, correlation, t-tests, ANOVA with Tukey/Bonferroni…, nonparametric "
            "tests). Reports are collapsible; their lock button sets Recalculate Auto / "
            "Manual, so a report follows its source data.\n"
            "7. Select a block of cells (e.g. rows 3–10) before plotting or fitting to use "
            "only those rows: the plot keeps the range, e.g. [Book1]Sheet1!B[3:10].\n"
            "8. View > Sparklines shows a tiny plot of each column in the header.\n"
            "9. File > New > Matrix, Matrix > Set Values (e.g. sin(x)*cos(y)), then Plot > "
            "Contour / 3D; Worksheet > Convert to Matrix turns XYZ columns into a matrix. "
            "Drag a 3D graph to rotate it.\n"
            "10. Double-click a text or axis title in a graph to edit it; the Format toolbar "
            "sets font, size, bold, italic, underline, super/subscript and Greek.\n"
            "11. File > Save Project (.rsop) keeps workbooks, matrices, graphs, notes and "
            "folders (drag windows between folders in the Project Explorer).\n\n"
            "Shortcuts: Ctrl+D add column, Ctrl+Q set column values, Ctrl+R rescale, "
            "Ctrl+G export graph, Ctrl+J copy page, Ctrl+K import ASCII, F2 rename, "
            "Alt+1 Project Explorer, Alt+2 Results Log, Alt+6 Messages Log, "
            "Alt+8 Object Manager.")
        existing = next((n for n, o in self.project.windows.items()
                         if isinstance(o, Notes) and o.name.startswith("GettingStarted")), None)
        if existing:
            self.show_window(existing)
            return
        notes = Notes(self.project.unique_name("GettingStarted"), text)
        self.new_notes(notes)

    # ================================================================== drag & drop
    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        """Drop .rsop / .rsg files to open them, data files (CSV, TXT, Excel) to import."""
        for url in event.mimeData().urls():
            if url.isLocalFile():
                self.open_path(url.toLocalFile())
        event.acceptProposedAction()

    # ================================================================== close
    def closeEvent(self, event) -> None:  # noqa: N802
        if not self.confirm_discard():
            event.ignore()
            return
        self.closing_all = True
        self._set_setting("window_state", self.saveState())
        event.accept()
        super().closeEvent(event)
        if self.ctx is not None:
            self.ctx.window_closed(self)

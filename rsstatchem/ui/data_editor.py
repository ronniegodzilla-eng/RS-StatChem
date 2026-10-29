"""Data Editor window: Data View / Variable View tabs and the SPSS-style menu structure.

The File, Edit and View menus are built here; Data, Transform, Analyze, Graphs and Utilities
come from :func:`menu_tree` (built-in entries merged with the feature modules' ``MENUS`` and
put in SPSS order by :data:`rsstatchem.ui.menus.MENU_ORDER`).
"""
from __future__ import annotations

import json
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd
from PySide6.QtCore import QMimeData, Qt
from PySide6.QtGui import QAction, QActionGroup, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QFileDialog, QFormLayout, QInputDialog, QLabel, QMainWindow,
                               QMenu, QMessageBox, QTabWidget, QVBoxLayout)

from rsstatchem import APP_NAME
from rsstatchem.core import selection
from rsstatchem.core.dataset import Dataset
from rsstatchem.core.stats import descriptive
from rsstatchem.io import files
from rsstatchem.ui import icons
from rsstatchem.ui.dialogs.compare import (IndependentTDialog, OneSampleTDialog,
                                           OneWayAnovaDialog, PairedTDialog, UnivariateDialog)
from rsstatchem.ui.dialogs.data import SortCasesDialog, TextImportDialog, ValueLabelsDialog
from rsstatchem.ui.dialogs.descriptive import DescriptivesDialog, FrequenciesDialog
from rsstatchem.ui.dialogs.explore import CrosstabsDialog, ExploreDialog, ReliabilityDialog
from rsstatchem.ui.dialogs.nonparametric import (BinomialDialog, ChiSquareDialog,
                                                 KIndependentDialog, KRelatedDialog,
                                                 OneSampleKSDialog, RunsDialog,
                                                 TwoIndependentDialog, TwoRelatedDialog)
from rsstatchem.ui.dialogs.regression import (BivariateDialog, LinearRegressionDialog,
                                              PartialDialog)
from rsstatchem.ui.dialogs.selection import (SelectCasesDialog, SplitFileDialog,
                                             WeightCasesDialog)
from rsstatchem.ui.dialogs.transform import (ComputeDialog, RecodeDifferentDialog,
                                             RecodeSameDialog)
from rsstatchem.ui.document import DatasetDocument
from rsstatchem.ui.menus import build_tree
from rsstatchem.ui.modules import add_module_menu
from rsstatchem.ui.models import DataTableModel, VariableTableModel
from rsstatchem.ui.views import ChoiceDelegate, DataGridView, SheetView, SpinDelegate

if TYPE_CHECKING:
    from rsstatchem.app import AppContext

# Analyze menu: (text, DialogClass) runs a dialog, a plain string is a planned (disabled) item,
# "FREQ"/"DESC" are the phase-1 actions, (title, [...]) is a submenu.
ANALYZE = [
    ("Descriptive Statistics", ["FREQ", "DESC", ("Explore...", ExploreDialog),
                                ("Crosstabs...", CrosstabsDialog)]),
    ("Compare Means", ["Means...", ("One-Sample T Test...", OneSampleTDialog),
                       ("Independent-Samples T Test...", IndependentTDialog),
                       ("Paired-Samples T Test...", PairedTDialog),
                       ("One-Way ANOVA...", OneWayAnovaDialog)]),
    ("General Linear Model", [("Univariate...", UnivariateDialog), "Multivariate...",
                              "Repeated Measures..."]),
    ("Mixed Models", ["Linear..."]),
    ("Correlate", [("Bivariate...", BivariateDialog), ("Partial...", PartialDialog)]),
    ("Regression", [("Linear...", LinearRegressionDialog), "Curve Estimation...",
                    "Binary Logistic...", "Multinomial Logistic...", "Ordinal..."]),
    ("Mediation && Moderation", ["PROCESS-style Model..."]),
    ("Dimension Reduction", ["Factor..."]),
    ("Scale", [("Reliability Analysis...", ReliabilityDialog)]),
    ("Structural Equation Modeling", ["CB-SEM...", "PLS-SEM..."]),
    ("Nonparametric Tests", [("Legacy Dialogs", [
        ("Chi-square...", ChiSquareDialog), ("Binomial...", BinomialDialog),
        ("Runs...", RunsDialog), ("1-Sample K-S...", OneSampleKSDialog),
        ("2 Independent Samples...", TwoIndependentDialog),
        ("K Independent Samples...", KIndependentDialog),
        ("2 Related Samples...", TwoRelatedDialog),
        ("K Related Samples...", KRelatedDialog)])]),
]

# Menu entries that exist in the roadmap but are not implemented yet (shown disabled).
PLANNED = {
    "Transform": [("Compute Variable...", ComputeDialog), None,
                  ("Recode into Same Variables...", RecodeSameDialog),
                  ("Recode into Different Variables...", RecodeDifferentDialog),
                  "Automatic Recode...", "Visual Binning...", None, "Rank Cases...",
                  "Replace Missing Values...", None, "Run Pending Transforms"],
    "RSM": [
        ("Create Design", ["Central Composite (CCD)...", "Box-Behnken...",
                           "Full / Fractional Factorial...", "Mixture Design..."]),
        "Analyze Design...", "Contour && Surface Plots...", "Optimization (Desirability)...",
    ],
    "Graphs": ["Graph Studio...", None, "Line / Scatter...", "Bar...", "Histogram...",
               "Box Plot...", "Curve Fitting..."],
    "Chemistry": ["Structure Editor...", "Reaction Editor...", "Formula && Molar Mass...",
                  "Insert Structure into Output..."],
}


_TREE: dict | None = None


def _sort_cases(editor) -> None:
    editor.sort_cases()


def menu_tree() -> dict[str, list]:
    """Built-in menus merged with the feature modules' MENUS (computed once)."""
    global _TREE
    if _TREE is None:
        _TREE = build_tree({
            "File": [],
            "Data": [("Sort Cases...", _sort_cases), None,
                     ("Split File...", SplitFileDialog),
                     ("Select Cases...", SelectCasesDialog),
                     ("Weight Cases...", WeightCasesDialog)],
            "Transform": PLANNED["Transform"],
            "Analyze": ANALYZE,
            "Graphs": PLANNED["Graphs"],
            "Utilities": [],
            "Analysis": [],
            "RSM": PLANNED["RSM"],
            "Chemistry": PLANNED["Chemistry"],
        })
    return _TREE


# top-level menus in SPSS order after File / Edit / View
MAIN_MENUS = ("Data", "Transform", "Analyze", "Graphs", "Utilities")


def _compute_with(dlg, ds: Dataset):
    """Run a dialog's compute() against another dataset (a filtered/split view)."""
    original = dlg.ds
    dlg.ds = ds
    try:
        return dlg.compute()
    finally:
        dlg.ds = original


class DataEditorWindow(QMainWindow):
    def __init__(self, ctx: "AppContext", dataset: Dataset | None = None, number: int = 1):
        super().__init__()
        self.ctx = ctx
        self.number = number
        self.doc = DatasetDocument(dataset or Dataset(name=f"DataSet{number}"), self)
        self.setWindowIcon(icons.icon("data_editor"))
        self.resize(1180, 760)
        self.setAcceptDrops(True)

        self.data_model = DataTableModel(self.doc, self)
        self.var_model = VariableTableModel(self.doc, self)
        self.data_view = DataGridView()
        self.data_view.setModel(self.data_model)
        self.var_view = SheetView()
        self.var_view.setModel(self.var_model)
        self._setup_variable_view()

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.South)
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self.data_view, "Data View")
        self.tabs.addTab(self.var_view, "Variable View")
        self.setCentralWidget(self.tabs)

        self.data_model.error.connect(self._edit_error)
        self.var_model.error.connect(self._edit_error)
        self.data_view.header_double_clicked.connect(self._go_to_variable)
        self.var_view.verticalHeader().sectionDoubleClicked.connect(self._go_to_data_column)
        self.var_view.doubleClicked.connect(self._variable_cell_activated)
        self.doc.structure_changed.connect(self._update_counts)
        self.doc.structure_changed.connect(self._fit_columns)
        self.doc.structure_changed.connect(self.apply_variable_sets)
        self.doc.modified_changed.connect(self.setWindowModified)

        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self._build_context_menus()

        self.state_label = QLabel()
        self.state_label.setStyleSheet("color:#8E1B1B; font-weight:600;")
        self.statusBar().addPermanentWidget(self.state_label)
        self.counts = QLabel()
        self.statusBar().addPermanentWidget(self.counts)
        self.statusBar().showMessage(f"{APP_NAME} Processor is ready")
        self._apply_settings()
        self._update_counts()
        self._fit_columns()
        self._update_title()
        self.apply_variable_sets()

    # ------------------------------------------------------------------ setup
    def _apply_settings(self) -> None:
        """Edit > Options saved earlier: leading zero and value labels in Data View."""
        from rsstatchem.core.stats import common

        settings = getattr(self.ctx, "settings", None)
        if settings is None:
            return
        common.LEADING_ZERO = settings.value("options/leading_zero", False) in (True, "true")
        if settings.value("data/show_labels", False) in (True, "true"):
            self.act_labels.setChecked(True)
            self.data_model.set_show_labels(True)

    @property
    def ds(self) -> Dataset:
        return self.doc.ds

    def _setup_variable_view(self) -> None:
        v = self.var_view
        cols = VariableTableModel.ATTR
        for attr, choices in VariableTableModel.CHOICES.items():
            v.setItemDelegateForColumn(cols.index(attr), ChoiceDelegate(choices, v))
        v.setItemDelegateForColumn(cols.index("width"), SpinDelegate(1, 32767, v))
        v.setItemDelegateForColumn(cols.index("decimals"), SpinDelegate(0, 16, v))
        v.setItemDelegateForColumn(cols.index("columns"), SpinDelegate(1, 255, v))
        widths = (120, 90, 60, 70, 220, 160, 100, 70, 80, 95, 80)
        for i, w in enumerate(widths):
            v.setColumnWidth(i, w)

    def _action(self, text: str, slot: Callable | None = None, shortcut=None, icon: str | None = None,
                tip: str | None = None, checkable: bool = False) -> QAction:
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        if icon:
            action.setIcon(icons.icon(icon))
        if tip:
            action.setToolTip(tip)
            action.setStatusTip(tip)
        action.setCheckable(checkable)
        if slot is not None:
            action.triggered.connect(slot)
        return action

    def _build_actions(self) -> None:
        a = self._action
        self.act_new = a("&Data", lambda: self.ctx.new_editor(), QKeySequence.StandardKey.New,
                         "new", "New dataset")
        self.act_open = a("&Open Data...", self.open_dialog, QKeySequence.StandardKey.Open,
                          "open", "Open data document")
        self.act_import_text = a("&Text / CSV Data...", lambda: self.import_dialog("text"),
                                 icon="import", tip="Import text or CSV data")
        self.act_import_excel = a("&Excel...", lambda: self.import_dialog("excel"))
        self.act_import_sav = a("&SPSS Statistics (.sav)...", lambda: self.import_dialog("sav"))
        self.act_import_por = a("SPSS &Portable (.por)...", lambda: self.import_dialog("por"))
        self.act_import_stata = a("S&tata (.dta)...", lambda: self.import_dialog("stata"))
        self.act_import_sas = a("S&AS (.sas7bdat, .xpt)...", lambda: self.import_dialog("sas"))
        self.act_new_output = a("&Output", self._new_output)
        self.act_open_output = a("&Output...", self._open_output)
        self.act_close = a("&Close", self.close, QKeySequence.StandardKey.Close)
        self.act_rename = a("Rena&me Dataset...", self.rename_dataset)
        self.act_info_working = a("&Working File", self.display_file_info)
        self.act_info_external = a("&External File...", self.display_external_info)
        self.act_save = a("&Save", self.save, QKeySequence.StandardKey.Save, "save",
                          "Save this document")
        self.act_save_as = a("Save &As...", self.save_as, QKeySequence.StandardKey.SaveAs)
        self.act_export_csv = a("&CSV / Text...", lambda: self.export("csv"), icon="export")
        self.act_export_excel = a("&Excel...", lambda: self.export("excel"))
        self.act_export_sav = a("&SPSS Statistics (.sav)...", lambda: self.export("sav"))
        self.act_export_stata = a("S&tata (.dta)...", lambda: self.export("stata"))
        self.act_export_sas = a("SAS &Transport (.xpt)...", lambda: self.export("sas"))
        self.act_exit = a("E&xit", lambda: QApplication.closeAllWindows(),
                          QKeySequence.StandardKey.Quit)

        self.act_undo = self.doc.undo_stack.createUndoAction(self, "&Undo")
        self.act_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.act_undo.setIcon(icons.icon("undo"))
        self.act_redo = self.doc.undo_stack.createRedoAction(self, "&Redo")
        self.act_redo.setShortcuts([QKeySequence.StandardKey.Redo, QKeySequence("Ctrl+Y")])
        self.act_redo.setIcon(icons.icon("redo"))
        self.act_cut = a("Cu&t", self._cut, QKeySequence.StandardKey.Cut)
        self.act_copy = a("&Copy", self._copy, QKeySequence.StandardKey.Copy)
        self.act_paste = a("&Paste", self._paste, QKeySequence.StandardKey.Paste)
        self.act_clear = a("Cle&ar", self._clear, QKeySequence.StandardKey.Delete)
        for act in (self.act_cut, self.act_copy, self.act_paste, self.act_clear):
            act.setShortcutContext(Qt.ShortcutContext.WidgetShortcut)
        self.act_insert_var = a("Insert &Variable", self.insert_variable, icon="insert_var",
                                tip="Insert variable")
        self.act_insert_case = a("Insert Ca&ses", self.insert_case, icon="insert_case",
                                 tip="Insert cases")
        self.act_delete_var = a("Delete Variable(s)", self.delete_variables)
        self.act_delete_case = a("Delete Case(s)", self.delete_cases)
        self.act_goto = a("&Go to Case...", self.go_to_case, "Ctrl+G", "goto", "Go to case")
        self.act_goto_var = a("Go to &Variable...", self.go_to_variable, "Ctrl+Shift+G")
        self.act_copy_names = a("Copy with Variable &Names", lambda: self.copy_with_header("name"))
        self.act_copy_labels = a("Copy with Variable &Labels",
                                 lambda: self.copy_with_header("label"))
        self.act_find = a("&Find...", lambda: self.find_dialog(False), QKeySequence.StandardKey.Find,
                          tip="Find data")
        self.act_find_next = a("Find Ne&xt", self.find_next, QKeySequence.StandardKey.FindNext)
        self.act_replace = a("&Replace...", lambda: self.find_dialog(True), "Ctrl+H")
        self.act_options = a("&Options...", self.options_dialog)
        self.act_status_bar = a("&Status Bar", lambda on: self.statusBar().setVisible(on),
                                checkable=True)
        self.act_status_bar.setChecked(True)
        self.act_toolbar = a("&Toolbar", lambda on: self.toolbar.setVisible(on), checkable=True)
        self.act_toolbar.setChecked(True)
        self.act_grid = a("&Grid Lines", lambda on: (self.data_view.setShowGrid(on),
                                                      self.var_view.setShowGrid(on)),
                          checkable=True)
        self.act_grid.setChecked(True)
        self.act_fonts = a("&Fonts...", self.choose_font)

        self.act_labels = a("&Value Labels", self.data_model.set_show_labels, icon="labels",
                            tip="Show value labels", checkable=True)
        self.act_toggle_view = a("Variables", self._toggle_view, "Ctrl+T")

        self.act_sort = a("&Sort Cases...", self.sort_cases, icon="sort", tip="Sort cases")
        self.act_freq = a("&Frequencies...", self.run_frequencies, icon="frequencies",
                          tip="Frequencies")
        self.act_desc = a("&Descriptives...", self.run_descriptives, icon="descriptives",
                          tip="Descriptives")
        self.act_output = a("Output Viewer", self.ctx.show_output_window, icon="output",
                            tip="Show the Output Viewer")
        self.act_about = a(f"&About {APP_NAME}", self.ctx.about)

    def _planned(self, menu: QMenu, entries) -> None:
        special = {"FREQ": self.act_freq, "DESC": self.act_desc}
        for entry in entries:
            if entry is None:
                menu.addSeparator()
            elif isinstance(entry, tuple) and isinstance(entry[1], list):
                title, children = entry
                self._planned(menu.addMenu(title), children)
            elif isinstance(entry, tuple):
                text, target = entry
                act = menu.addAction(text)
                if isinstance(target, type) and hasattr(target, "compute"):
                    act.triggered.connect(lambda _=False, c=target: self.run_analysis(c))
                else:  # a plain callable taking the editor, e.g. opening a graph window
                    act.triggered.connect(lambda _=False, f=target: self._run_action(f))
            elif entry in special:
                menu.addAction(special[entry])
            else:
                act = menu.addAction(entry)
                act.setEnabled(False)

    def _build_menus(self) -> None:
        mb = self.menuBar()

        file_menu = mb.addMenu("&File")
        new_menu = file_menu.addMenu("&New")
        new_menu.addActions([self.act_new, self.act_new_output])
        open_menu = file_menu.addMenu("&Open")
        open_menu.addActions([self.act_open, self.act_open_output])
        import_menu = file_menu.addMenu("&Import Data")
        import_menu.addActions([self.act_import_text, self.act_import_excel, self.act_import_sav,
                                self.act_import_por, self.act_import_stata,
                                self.act_import_sas])
        file_menu.addAction(self.act_close)
        file_menu.addSeparator()
        file_menu.addActions([self.act_save, self.act_save_as])
        export_menu = file_menu.addMenu("&Export")
        export_menu.addActions([self.act_export_csv, self.act_export_excel, self.act_export_sav,
                                self.act_export_stata, self.act_export_sas])
        file_menu.addAction(self.act_rename)
        info = file_menu.addMenu("Display Data File &Information")
        info.addActions([self.act_info_working, self.act_info_external])
        tree = menu_tree()
        if tree["File"]:
            file_menu.addSeparator()
            self._planned(file_menu, tree["File"])
        file_menu.addSeparator()
        self.recent_menu = file_menu.addMenu("Recently Used &Data")
        self.recent_menu.aboutToShow.connect(self._fill_recent)
        file_menu.addSeparator()
        file_menu.addAction(self.act_exit)

        edit = mb.addMenu("&Edit")
        edit.addActions([self.act_undo, self.act_redo])
        edit.addSeparator()
        edit.addActions([self.act_cut, self.act_copy, self.act_copy_names, self.act_copy_labels,
                         self.act_paste, self.act_clear])
        edit.addSeparator()
        edit.addActions([self.act_insert_var, self.act_insert_case])
        edit.addSeparator()
        edit.addActions([self.act_find, self.act_find_next, self.act_replace])
        edit.addSeparator()
        edit.addActions([self.act_goto, self.act_goto_var])
        edit.addSeparator()
        edit.addAction(self.act_options)

        view = mb.addMenu("&View")
        view.addActions([self.act_status_bar, self.act_toolbar, self.act_fonts, self.act_grid])
        view.addSeparator()
        view.addAction(self.act_labels)
        view.addSeparator()
        view.addAction(self.act_toggle_view)

        for title in MAIN_MENUS:
            if tree.get(title):
                self._planned(mb.addMenu(f"&{title}"), tree[title])
        # Curve fitting / signal analysis, RSM and chemistry are modules of their own now
        # (Graph & Analysis, RSM / DOE, Chemical Drawing), opened from Window > Modules.
        known = {"File", "Analysis", "RSM", "Chemistry", *MAIN_MENUS}
        for title, entries in tree.items():
            if title not in known and entries:
                self._planned(mb.addMenu(title), entries)

        self.window_menu = mb.addMenu("&Window")
        self.window_menu.aboutToShow.connect(self._fill_window_menu)

        help_menu = mb.addMenu("&Help")
        from rsstatchem.ui.update_dialog import open_user_guide

        help_menu.addAction("User &Guide (PDF)", open_user_guide)
        updates = help_menu.addAction("Check for &Updates...")
        updates.triggered.connect(lambda: self.ctx.check_for_updates(manual=True))
        help_menu.addSeparator()
        help_menu.addAction(self.act_about)

    def _build_toolbar(self) -> None:
        tb = self.addToolBar("Main")
        self.toolbar = tb
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        tb.addActions([self.act_open, self.act_save, self.act_import_text])
        tb.addSeparator()
        tb.addActions([self.act_undo, self.act_redo])
        tb.addSeparator()
        tb.addAction(self.act_output)
        tb.addSeparator()
        tb.addActions([self.act_goto, self.act_insert_var, self.act_insert_case, self.act_sort])
        tb.addSeparator()
        tb.addAction(self.act_find)
        tb.addAction(self.act_labels)
        tb.addSeparator()
        tb.addActions([self.act_freq, self.act_desc])
        self.data_view.addActions([self.act_cut, self.act_copy, self.act_paste, self.act_clear])

    def _build_context_menus(self) -> None:
        dv = self.data_view
        dv.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
        sep = QAction(self)
        sep.setSeparator(True)
        dv.addActions([sep, self.act_insert_var, self.act_insert_case])

        header = dv.horizontalHeader()
        header.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
        asc = self._action("Sort Ascending", lambda: self._sort_selected_column(True))
        desc = self._action("Sort Descending", lambda: self._sort_selected_column(False))
        header.addActions([self.act_cut, self.act_copy, self.act_paste, self.act_clear])
        header.addActions([self._separator(), self.act_insert_var, self.act_delete_var,
                           self._separator(), asc, desc])

        rows = dv.verticalHeader()
        rows.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
        rows.addActions([self.act_insert_case, self.act_delete_case])

        vv = self.var_view.verticalHeader()
        vv.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
        ins = self._action("Insert Variable", self._insert_variable_in_var_view)
        dele = self._action("Clear (Delete Variable)", self._delete_in_var_view)
        vv.addActions([ins, dele])

    def _separator(self) -> QAction:
        sep = QAction(self)
        sep.setSeparator(True)
        return sep

    # ------------------------------------------------------------------ helpers
    def _update_counts(self) -> None:
        self.counts.setText(f"  Cases: {self.ds.n_cases}   Variables: {self.ds.n_vars}  ")
        state = []
        if selection.filter_mask(self.ds) is not None:
            state.append("Filter On")
        if self.ds.meta.get("weight"):
            state.append("Weight On")
        if (self.ds.meta.get("split") or {}).get("vars"):
            state.append("Split by " + " ".join(self.ds.meta["split"]["vars"]))
        self.state_label.setText("   ".join(state))

    def _fit_columns(self) -> None:
        """Width follows each variable's Columns attribute but always fits the name + icon."""
        header = self.data_view.horizontalHeader()
        fm = header.fontMetrics()
        char = fm.horizontalAdvance("0")
        for c, var in enumerate(self.ds.variables):
            header.resizeSection(c, max(char * var.columns + 18,
                                        fm.horizontalAdvance(var.name) + 44, 64))

    def _update_title(self) -> None:
        name = Path(self.ds.path).name if self.ds.path else f"Untitled{self.number}"
        self.setWindowTitle(f"{name}[*] [{self.ds.name}] - {APP_NAME} Data Editor")
        self.setWindowModified(self.ds.modified)

    def _edit_error(self, message: str) -> None:
        QApplication.beep()
        self.statusBar().showMessage(message, 6000)

    def _guard(self, title: str, fn: Callable) -> object:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - report any failure in a dialog
            QApplication.restoreOverrideCursor()
            box = QMessageBox(QMessageBox.Icon.Critical, title, str(exc), parent=self)
            box.setDetailedText(traceback.format_exc())
            box.exec()
            return None
        finally:
            if QApplication.overrideCursor():
                QApplication.restoreOverrideCursor()

    def is_pristine(self) -> bool:
        return self.ds.n_vars == 0 and not self.ds.modified and self.ds.path is None

    # ------------------------------------------------------------------ files
    def open_dialog(self) -> None:
        start = self.ctx.settings.value("last_dir", str(Path.home()))
        path, _ = QFileDialog.getOpenFileName(self, "Open Data", start, files.OPEN_FILTER)
        if path:
            self.open_path(path)

    def import_dialog(self, kind: str) -> None:
        patterns = {"text": files.OPEN_FILTER.split(";;")[3],
                    "excel": files.OPEN_FILTER.split(";;")[4], "sav": "SPSS data (*.sav)",
                    "por": files.POR_FILTER, "stata": files.STATA_FILTER,
                    "sas": files.SAS_FILTER}
        pattern = patterns[kind]
        start = self.ctx.settings.value("last_dir", str(Path.home()))
        path, _ = QFileDialog.getOpenFileName(self, "Import Data", start, pattern)
        if path:
            self.open_path(path)

    def open_path(self, path: str) -> None:
        if Path(path).suffix.lower() == ".rsg":  # Graph Studio project
            from rsstatchem.core.graph.export import load_graph
            from rsstatchem.ui.graph.window import show_graph

            graph = self._guard("Open Graph", lambda: load_graph(path))
            if graph is not None:
                show_graph(graph, self, path)
                self.ctx.add_recent(path)
            return
        if Path(path).suffix.lower() == ".rso":  # output document
            viewer = self.ctx.output()
            if viewer.open_path(path):
                self.ctx.show_output_window()
                self.ctx.add_recent(path)
            return
        if Path(path).suffix.lower() == ".rsc":  # chemical drawing
            from rsstatchem.ui.chem_editor import open_drawing_file

            if self._guard("Open Chemical Drawing", lambda: open_drawing_file(self, path)):
                self.ctx.add_recent(path)
            return
        try:
            kind = files.kind_of(path)
        except ValueError as exc:
            QMessageBox.warning(self, APP_NAME, str(exc))
            return
        dataset = None
        if kind == "project":
            dataset = self._guard("Open Data", lambda: files.load_project(path))
        elif kind == "sav":
            dataset = self._guard("Open Data", lambda: files.read_sav(path))
        elif kind == "por":
            dataset = self._guard("Open Data", lambda: files.read_por(path))
        elif kind == "stata":
            dataset = self._guard("Open Data", lambda: files.read_stata(path))
        elif kind == "sas":
            dataset = self._guard("Open Data", lambda: files.read_sas(path))
        elif kind == "text":
            dlg = TextImportDialog(path, self)
            if not dlg.exec():
                return
            params = dlg.params()
            df = self._guard("Import Text", lambda: files.read_text(path, **params))
            dataset = None if df is None else Dataset(df)
        elif kind == "excel":
            sheets = self._guard("Import Excel", lambda: files.excel_sheets(path))
            if not sheets:
                return
            sheet = sheets[0]
            if len(sheets) > 1:
                sheet, ok = QInputDialog.getItem(self, "Import Excel", "Worksheet:", sheets, 0,
                                                 False)
                if not ok:
                    return
            df = self._guard("Import Excel", lambda: files.read_excel(path, sheet))
            dataset = None if df is None else Dataset(df)
        if dataset is None:
            return
        self.ctx.settings.setValue("last_dir", str(Path(path).parent))
        self.ctx.add_recent(path)
        if kind in ("text", "excel", "por", "stata", "sas"):
            dataset.path = None
            dataset.modified = True
        self.ctx.show_dataset(dataset, reuse=self if self.is_pristine() else None)

    def load_dataset(self, dataset: Dataset) -> None:
        dataset.name = f"DataSet{self.number}"
        self.doc.set_dataset(dataset)
        self._update_counts()
        self._update_title()
        self.tabs.setCurrentIndex(0)

    def save(self) -> bool:
        if self.ds.path and Path(self.ds.path).suffix.lower() in (files.PROJECT_EXT, ".sav"):
            return self._save_to(self.ds.path)
        return self.save_as()

    def save_as(self) -> bool:
        start = self.ctx.settings.value("last_dir", str(Path.home()))
        base = Path(self.ds.path).stem if self.ds.path else f"Untitled{self.number}"
        path, chosen = QFileDialog.getSaveFileName(
            self, "Save Data As", str(Path(start) / f"{base}{files.PROJECT_EXT}"),
            "RS-StatChem data (*.rsd);;SPSS data (*.sav)")
        if not path:
            return False
        if Path(path).suffix.lower() not in (files.PROJECT_EXT, ".sav"):
            path += ".sav" if "sav" in chosen else files.PROJECT_EXT
        return self._save_to(path)

    def _save_to(self, path: str) -> bool:
        writer = files.write_sav if path.lower().endswith(".sav") else files.save_project
        if self._guard("Save Data", lambda: writer(self.ds, path) or True) is None:
            return False
        self.ds.path = path
        self.doc.mark_saved()
        self.ctx.settings.setValue("last_dir", str(Path(path).parent))
        self.ctx.add_recent(path)
        self._update_title()
        self.statusBar().showMessage(f"Saved {path}", 5000)
        return True

    def export(self, kind: str) -> None:
        start = self.ctx.settings.value("last_dir", str(Path.home()))
        base = Path(self.ds.path).stem if self.ds.path else f"Untitled{self.number}"
        if kind == "csv":
            path, chosen = QFileDialog.getSaveFileName(
                self, "Export Data", str(Path(start) / f"{base}.csv"),
                "CSV, comma-separated (*.csv);;CSV, semicolon + decimal comma (*.csv);;"
                "Tab-delimited (*.txt)")
            if not path:
                return
            if "semicolon" in chosen:
                fn = lambda: files.write_text(self.ds, path, ";", ",")  # noqa: E731
            elif "Tab" in chosen:
                fn = lambda: files.write_text(self.ds, path, "\t")  # noqa: E731
            else:
                fn = lambda: files.write_text(self.ds, path)  # noqa: E731
        elif kind in ("sav", "stata", "sas"):
            ext, pattern, writer = {"sav": (".sav", "SPSS data (*.sav)", files.write_sav),
                                    "stata": (".dta", files.STATA_FILTER, files.write_stata),
                                    "sas": (".xpt", "SAS transport (*.xpt)", files.write_xport)
                                    }[kind]
            path, _ = QFileDialog.getSaveFileName(self, "Export Data",
                                                  str(Path(start) / f"{base}{ext}"), pattern)
            if not path:
                return
            if not path.lower().endswith(ext):
                path += ext
            fn = lambda: writer(self.ds, path)  # noqa: E731
        else:
            path, _ = QFileDialog.getSaveFileName(self, "Export Data",
                                                  str(Path(start) / f"{base}.xlsx"),
                                                  "Excel workbook (*.xlsx)")
            if not path:
                return
            fn = lambda: files.write_excel(self.ds, path)  # noqa: E731
        if self._guard("Export Data", lambda: fn() or True):
            self.statusBar().showMessage(f"Exported {path}", 5000)

    def _fill_recent(self) -> None:
        self.recent_menu.clear()
        recent = self.ctx.recent_files()
        if not recent:
            self.recent_menu.addAction("(none)").setEnabled(False)
        for i, path in enumerate(recent, 1):
            act = self.recent_menu.addAction(f"&{i} {path}")
            act.triggered.connect(lambda _=False, p=path: self.open_path(p))

    def _fill_window_menu(self) -> None:
        self.window_menu.clear()
        self.window_menu.addAction(self.act_output)
        self.window_menu.addSeparator()
        group = QActionGroup(self.window_menu)
        for editor in self.ctx.editors:
            act = self.window_menu.addAction(editor.windowTitle().replace("[*]", ""))
            act.setCheckable(True)
            act.setChecked(editor is self)
            group.addAction(act)
            act.triggered.connect(lambda _=False, e=editor: (e.show(), e.raise_(),
                                                             e.activateWindow()))
        self.window_menu.addSeparator()
        add_module_menu(self.window_menu, self.ctx)

    # ------------------------------------------------------------------ editing
    def _active_sheet(self) -> SheetView:
        return self.tabs.currentWidget()

    def _cut(self) -> None:
        if self._active_sheet() is self.data_view:
            self.data_view.copy_selection()
            self.data_view.clear_selection_cells()

    def _copy(self) -> None:
        if self._active_sheet() is self.data_view:
            self.data_view.copy_selection()

    def _paste(self) -> None:
        if self._active_sheet() is self.data_view:
            self.data_view.paste()

    def _clear(self) -> None:
        if self._active_sheet() is not self.data_view:
            return
        cols = self.data_view.selected_columns()
        rows = self.data_view.selected_rows()
        if cols and all(c < self.ds.n_vars for c in cols):
            self.delete_variables()
        elif rows and all(r < self.ds.n_cases for r in rows):
            self.delete_cases()
        else:
            self.data_view.clear_selection_cells()

    def insert_variable(self) -> None:
        if self._active_sheet() is self.var_view:
            self._insert_variable_in_var_view()
            return
        col = self.data_view.currentIndex().column()
        col = self.ds.n_vars if col < 0 else min(col, self.ds.n_vars)
        self.doc.mutate(lambda ds: ds.add_variable(index=col), "Insert Variable")
        self.data_view.setCurrentIndex(self.data_model.index(
            max(self.data_view.currentIndex().row(), 0), col))

    def insert_case(self) -> None:
        row = self.data_view.currentIndex().row()
        row = self.ds.n_cases if row < 0 else min(row, self.ds.n_cases)
        def insert(ds):
            if ds.n_vars == 0:
                ds.add_variable()
            ds.insert_cases(row)

        self.doc.mutate(insert, "Insert Cases")
        self.tabs.setCurrentIndex(0)

    def delete_variables(self) -> None:
        cols = [c for c in self.data_view.selected_columns() if c < self.ds.n_vars]
        if not cols:
            idx = self.data_view.currentIndex()
            cols = [idx.column()] if idx.isValid() and idx.column() < self.ds.n_vars else []
        if cols:
            self.doc.mutate(lambda ds: ds.delete_variables(cols), "Delete Variables")

    def delete_cases(self) -> None:
        rows = [r for r in self.data_view.selected_rows() if r < self.ds.n_cases]
        if not rows:
            idx = self.data_view.currentIndex()
            rows = [idx.row()] if idx.isValid() and idx.row() < self.ds.n_cases else []
        if rows:
            self.doc.mutate(lambda ds: ds.delete_cases(rows), "Delete Cases")

    def _insert_variable_in_var_view(self) -> None:
        row = self.var_view.currentIndex().row()
        row = self.ds.n_vars if row < 0 else min(row, self.ds.n_vars)
        self.doc.mutate(lambda ds: ds.add_variable(index=row), "Insert Variable")

    def _delete_in_var_view(self) -> None:
        rows = sorted({i.row() for i in self.var_view.selectionModel().selectedIndexes()
                       if i.row() < self.ds.n_vars})
        if rows:
            self.doc.mutate(lambda ds: ds.delete_variables(rows), "Delete Variables")

    def go_to_case(self) -> None:
        if self.ds.n_cases == 0:
            return
        current = max(self.data_view.currentIndex().row(), 0) + 1
        number, ok = QInputDialog.getInt(self, "Go To", "Go to case number:", current, 1,
                                         self.ds.n_cases)
        if ok:
            self.tabs.setCurrentIndex(0)
            col = max(self.data_view.currentIndex().column(), 0)
            idx = self.data_model.index(number - 1, col)
            self.data_view.setCurrentIndex(idx)
            self.data_view.scrollTo(idx)

    def _toggle_view(self) -> None:
        self.tabs.setCurrentIndex(1 - self.tabs.currentIndex())

    def select_cell(self, row: int, col: int) -> None:
        self.tabs.setCurrentIndex(0)
        idx = self.data_model.index(row, col)
        self.data_view.setCurrentIndex(idx)
        self.data_view.scrollTo(idx)

    def go_to_variable_index(self, col: int) -> None:
        if not 0 <= col < self.ds.n_vars:
            return
        if self.tabs.currentIndex() == 1:
            self.var_view.setCurrentIndex(self.var_model.index(col, 0))
            return
        self.select_cell(max(self.data_view.currentIndex().row(), 0), col)

    def go_to_variable(self) -> None:
        """Edit > Go to Variable (SPSS: Ctrl+Shift+G)."""
        if not self.ds.n_vars:
            return
        items = [f"{v.name}" + (f"  ({v.label})" if v.label else "") for v in self.ds.variables]
        col = max(self.data_view.currentIndex().column(), 0)
        choice, ok = QInputDialog.getItem(self, "Go To", "Go to variable:", items,
                                          min(col, len(items) - 1), False)
        if ok:
            self.go_to_variable_index(items.index(choice))

    def copy_with_header(self, kind: str = "name") -> None:
        """Copy the selected cells with variable names (or labels) as the first row."""
        block = self.data_view.selected_block()
        if not block or not self.ds.n_vars:
            return
        top, left, bottom, right = block
        right = min(right, self.ds.n_vars - 1)
        bottom = min(bottom, self.ds.n_cases - 1)
        header = [(v.title if kind == "label" else v.name)
                  for v in self.ds.variables[left:right + 1]]
        lines = ["\t".join(header)]
        for r in range(top, bottom + 1):
            lines.append("\t".join(self.ds.edit_text(r, c) for c in range(left, right + 1)))
        mime = QMimeData()
        mime.setText("\n".join(lines))
        QGuiApplication.clipboard().setMimeData(mime)
        self.statusBar().showMessage(f"Copied {bottom - top + 1} case(s) with variable "
                                     f"{'labels' if kind == 'label' else 'names'}.", 4000)

    def find_dialog(self, replace: bool = False):
        from rsstatchem.ui.find_replace import show_find

        return show_find(self, replace)

    def find_next(self) -> None:
        from rsstatchem.ui.find_replace import _OPEN, show_find

        dlg = _OPEN.get(id(self))
        if dlg is None or not dlg.find_edit.text():
            show_find(self)
        else:
            dlg.find_next()

    def apply_variable_sets(self) -> None:
        """Utilities > Use Variable Sets: hide the variables outside the active sets."""
        from rsstatchem.core.data_props import visible_variables

        keep = visible_variables(self.ds)
        keep_set = None if keep is None else {n.lower() for n in keep}
        for c, var in enumerate(self.ds.variables):
            hidden = keep_set is not None and var.name.lower() not in keep_set
            self.data_view.setColumnHidden(c, hidden)
            self.var_view.setRowHidden(c, hidden)
        for c in range(self.ds.n_vars, self.data_model.columnCount()):
            self.data_view.setColumnHidden(c, False)
        if hasattr(self, "state_label"):
            self._update_counts()

    def rename_dataset(self) -> None:
        name, ok = QInputDialog.getText(self, "Rename Dataset", "Dataset name:",
                                        text=self.ds.name)
        if ok and name.strip():
            self.ds.name = name.strip()
            self._update_title()

    def display_file_info(self) -> None:
        from rsstatchem.core.stats.reports import file_info

        self.ctx.show_output(file_info(self.ds))

    def display_external_info(self) -> None:
        from rsstatchem.core.stats.reports import file_info
        from rsstatchem.ui.dialogs.data_props import load_any

        start = self.ctx.settings.value("last_dir", str(Path.home()))
        path, _ = QFileDialog.getOpenFileName(self, "Display Data File Information", start,
                                              "Data files (*.rsd *.sav *.por *.dta "
                                              "*.sas7bdat *.xpt)")
        if not path:
            return
        ds = self._guard("Display Data File Information", lambda: load_any(path))
        if ds is not None:
            self.ctx.show_output(file_info(ds, path))

    def _new_output(self) -> None:
        self.ctx.output().new_document()
        self.ctx.show_output_window()

    def _open_output(self) -> None:
        self.ctx.output().open_dialog()
        self.ctx.show_output_window()

    def choose_font(self) -> None:
        from PySide6.QtWidgets import QFontDialog

        result = QFontDialog.getFont(self.data_view.font(), self, "Fonts")
        ok = next((r for r in result if isinstance(r, bool)), False)
        font = next((r for r in result if not isinstance(r, bool)), None)
        if ok and font is not None:
            self.data_view.setFont(font)
            self.var_view.setFont(font)

    def options_dialog(self) -> None:
        """Edit > Options: TableLook, leading zero for numbers below 1, value labels."""
        from rsstatchem.core import report
        from rsstatchem.core.stats import common

        dlg = QDialog(self)
        dlg.setWindowTitle("Options")
        look = QComboBox()
        for key, label in report.LOOKS.items():
            look.addItem(label, key)
        current = self.ctx.settings.value("output/look", report.DEFAULT_LOOK)
        look.setCurrentIndex(max(list(report.LOOKS).index(current)
                                 if current in report.LOOKS else 0, 0))
        zero = QCheckBox("Display a leading zero for numbers below 1 (0.045 instead of .045)")
        zero.setChecked(bool(common.LEADING_ZERO))
        labels = QCheckBox("Show value labels in Data View by default")
        labels.setChecked(self.ctx.settings.value("data/show_labels", "false") in
                          ("true", True))
        form = QFormLayout()
        form.addRow("Output TableLook:", look)
        layout = QVBoxLayout(dlg)
        layout.addLayout(form)
        layout.addWidget(zero)
        layout.addWidget(labels)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)
        if not dlg.exec():
            return
        self.ctx.settings.setValue("output/look", look.currentData())
        self.ctx.settings.setValue("options/leading_zero", zero.isChecked())
        self.ctx.settings.setValue("data/show_labels", labels.isChecked())
        common.LEADING_ZERO = zero.isChecked()
        viewer = getattr(self.ctx, "_output", None)
        if viewer is not None:
            viewer.set_look(look.currentData())

    def _go_to_variable(self, col: int) -> None:
        if col >= self.ds.n_vars:
            return
        self.tabs.setCurrentIndex(1)
        self.var_view.setCurrentIndex(self.var_model.index(col, 0))

    def _go_to_data_column(self, row: int) -> None:
        if row >= self.ds.n_vars:
            return
        self.tabs.setCurrentIndex(0)
        current = max(self.data_view.currentIndex().row(), 0)
        self.data_view.setCurrentIndex(self.data_model.index(current, row))

    def _variable_cell_activated(self, index) -> None:
        if VariableTableModel.ATTR[index.column()] != "value_labels":
            return
        if index.row() >= self.ds.n_vars:
            return
        var = self.ds.variables[index.row()]
        dlg = ValueLabelsDialog(var, self)
        if dlg.exec():
            self.doc.set_variable_attr(index.row(), "value_labels", dlg.labels)

    def _sort_selected_column(self, ascending: bool) -> None:
        cols = [c for c in self.data_view.selected_columns() if c < self.ds.n_vars]
        if cols:
            names = [self.ds.variables[c].name for c in cols]
            self.doc.mutate(lambda ds: ds.sort_cases(names, [ascending] * len(names)),
                            "Sort Cases")

    # ------------------------------------------------------------------ data / analyze
    def _need_data(self) -> bool:
        if self.ds.n_vars and self.ds.n_cases:
            return True
        QMessageBox.information(self, APP_NAME, "There is no data to analyze yet.")
        return False

    def sort_cases(self) -> None:
        if not self._need_data():
            return
        dlg = SortCasesDialog(self.ds, self)
        if dlg.exec():
            p = dlg.params()
            self._guard("Sort Cases",
                        lambda: self.doc.mutate(
                            lambda ds: ds.sort_cases(p["names"], p["ascending"]), "Sort Cases"))

    def run_analysis(self, dialog_cls) -> None:
        """Open an analysis dialog; on OK run its ``compute()`` and show the output."""
        if getattr(dialog_cls, "needs_data", True) and not self._need_data():
            return
        dlg = dialog_cls(self.ds, self)
        if not dlg.exec():
            return
        if getattr(dlg, "modifies_data", False):
            result = self._guard(dlg.windowTitle(), lambda: self.doc.mutate(
                lambda _ds: dlg.compute(), dlg.windowTitle()))
        elif not getattr(dlg, "needs_data", True):  # calculators: run once, no split groups
            result = self._guard(dlg.windowTitle(), dlg.compute)
        else:
            meta_before = json.dumps(self.ds.meta, sort_keys=True, default=str)
            result = self._guard(dlg.windowTitle(), lambda: self._compute_on_views(
                lambda view_ds: _compute_with(dlg, view_ds)))
            if json.dumps(self.ds.meta, sort_keys=True, default=str) != meta_before:
                self.doc.set_modified(True)  # e.g. a DOE analysis stored the chosen model
        if result is None:
            return
        section = result
        if hasattr(result, "section") and hasattr(result, "saved"):
            section = result.section
            if result.saved:
                created = self._guard("Save Variables", lambda: self.doc.mutate(
                    lambda ds: self._save_regression_vars(ds, result.saved),
                    f"{dlg.windowTitle()}: Save Variables"))
                if created:
                    section.note("", "Variables created: " + ", ".join(created))
        self.ctx.show_output(section)

    def _compute_on_views(self, fn):
        """Run ``fn(dataset)`` on the selected/weighted/split view(s) of the active data.

        Returns a Section, or an object with ``.section`` and ``.saved`` whose saved columns
        are mapped back onto the active dataset's rows."""
        source = self._split_source()
        views = selection.analysis_views(source)
        if len(views) == 1 and not selection.describe(source):
            return fn(source)
        results = [(view, fn(view.ds)) for view in views]
        section = selection.combine(
            [(v.label, r.section if hasattr(r, "section") else r) for v, r in results], self.ds)
        saved = {}
        for view, r in results:
            for base, spec in (getattr(r, "saved", None) or {}).items():
                if view.rows is None:
                    continue
                full = saved.setdefault(base, (spec[0], pd.Series(np.nan, index=self.ds.df.index),
                                               *spec[2:]))
                full[1].iloc[view.rows] = spec[1].to_numpy()
        if any(getattr(r, "saved", None) for _, r in results) and \
                any(v.rows is None for v in views):
            section.note("", "New variables are not saved while cases are weighted.",
                         style="warning")
        return SimpleNamespace(section=section, saved=saved) if saved else section

    def _split_source(self) -> Dataset:
        """The dataset whose views the analyses get. When Split File is by the imputation
        variable only (SPSS turns it on after Impute Missing Data Values), analyses receive
        the whole stacked file so they can pool across imputations (Rubin's rules) as SPSS
        does; any other split keeps one analysis per split group."""
        split = [v.lower() for v in (self.ds.meta.get("split") or {}).get("vars") or []]
        if len(split) != 1:
            return self.ds
        try:
            from rsstatchem.core.stats.imputation import imputation_variable, is_imputed
        except ImportError:  # pragma: no cover - imputation module missing
            return self.ds
        name = imputation_variable(self.ds)
        if not name or name.lower() != split[0] or not is_imputed(self.ds):
            return self.ds
        whole = Dataset(self.ds.df, self.ds.variables, name=self.ds.name)
        whole.meta = {k: v for k, v in self.ds.meta.items() if k != "split"}
        return whole

    def _run_action(self, fn) -> None:
        self._guard("Error", lambda: fn(self))

    @staticmethod
    def _save_regression_vars(ds: Dataset, saved: dict) -> list[str]:
        """SPSS naming: PRE_1, RES_1, ZRE_1… with the first suffix free for all of them."""
        k = 1
        while any(ds.has_name(f"{base}_{k}") for base in saved):
            k += 1
        created = []
        for base, spec in saved.items():
            label, series = spec[0], spec[1]
            name = f"{base}_{k}"
            var = ds.add_variable(name, values=series.to_numpy())
            var.label = label
            var.decimals = spec[2] if len(spec) > 2 else 5  # optional (label, series, dec)
            created.append(name)
        return created

    def run_descriptives(self) -> None:
        if not self._need_data():
            return
        dlg = DescriptivesDialog(self.ds, self)
        if not dlg.exec():
            return
        p = dlg.params()
        section = self._guard("Descriptives", lambda: self._compute_on_views(
            lambda d: descriptive.descriptives(d, p["names"], p["stats"])))
        if section is None:
            return
        if p["save_z"]:
            created = self._guard("Descriptives", lambda: self.doc.mutate(
                lambda ds: descriptive.standardize(ds, p["names"]),
                "Save Standardized Values"))
            if created:
                section.note("", "Standardized variables created: " + ", ".join(created))
        self.ctx.show_output(section)

    def run_frequencies(self) -> None:
        if not self._need_data():
            return
        dlg = FrequenciesDialog(self.ds, self)
        if not dlg.exec():
            return
        p = dlg.params()
        section = self._guard("Frequencies", lambda: self._compute_on_views(
            lambda d: descriptive.frequencies(d, p["names"], p["stats"], p["show_tables"],
                                              p["chart"], p["normal_curve"])))
        if section is not None:
            self.ctx.show_output(section)

    # ------------------------------------------------------------------ window events
    def maybe_save(self) -> bool:
        if not self.ds.modified or self.ds.n_vars == 0:
            return True
        name = Path(self.ds.path).name if self.ds.path else f"Untitled{self.number}"
        answer = QMessageBox.question(
            self, APP_NAME, f"Save changes to {name} [{self.ds.name}]?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Save)
        if answer == QMessageBox.StandardButton.Save:
            return self.save()
        return answer == QMessageBox.StandardButton.Discard

    def closeEvent(self, event) -> None:
        if not self.maybe_save():
            event.ignore()
            return
        event.accept()
        self.ctx.editor_closed(self)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        for url in event.mimeData().urls():
            if url.isLocalFile():
                self.open_path(url.toLocalFile())


def open_workspace(ctx, path: str | None = None):
    """Statistics module entry point (see :mod:`rsstatchem.ui.modules`): a Data Editor,
    reusing an untouched empty one."""
    editor = next((e for e in ctx.editors if e.is_pristine()), None) or ctx.new_editor()
    if path:
        editor.open_path(path)
    editor.show()
    editor.raise_()
    editor.activateWindow()
    return editor

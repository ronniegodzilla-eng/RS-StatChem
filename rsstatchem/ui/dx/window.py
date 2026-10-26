"""The RSM / DOE main window, laid out like Stat-Ease Design-Expert.

Menu bar (File, Edit, View, Display Options, Design Tools, Analysis, Optimization, Window,
Help), a toolbar, the navigation pane on the left and the selected node on the right:

    Design (Actual)        the design layout
        Summary / Graph Columns / Evaluation / Notes
    Analysis               one node per response (R1:Yield ...)
    Optimization           Numerical, Graphical
    Post Analysis          Point Prediction, Confirmation, Coefficients Table

A window holds at most one design; New Design / Open in a window that already has one opens a
new window. Without a design the window shows a start page and (from the launcher) the New
Design wizard, shown non-modally.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtCore import QPoint, QSize, Qt, QTimer
from PySide6.QtGui import QActionGroup, QKeySequence, QUndoStack
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
                               QLabel, QLineEdit, QMainWindow, QMessageBox, QSplitter,
                               QStackedWidget, QTextBrowser, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout, QWidget)

from rsstatchem import APP_NAME
from rsstatchem.ui.dx import icons, style
from rsstatchem.ui.dx.document import DesignDocument
from rsstatchem.ui.dx.widgets import GraphView, ReportView, make_action

TIPS = {
    "start": "<h3>Getting started</h3><p>Click <b>New Design</b> to choose a design (Factorial, "
             "Response Surface, Mixture or Optimal), enter the factors with their ranges and "
             "name the responses. Or open a tutorial data set from <b>Help &gt; Tutorial "
             "Data</b>.</p>",
    "design": "<h3>Design layout</h3><p>The runs of the experiment. Perform them in the "
              "<b>Run</b> order (randomised) and type each measured result into the response "
              "columns. You can paste a whole column copied from Excel (Ctrl+V). Right-click a "
              "column header to sort by standard or run order, show the space point type or "
              "switch factors between actual and coded units. Untick <b>Select</b> to ignore "
              "a run.</p>",
    "summary": "<h3>Summary</h3><p>Design type, factor ranges and coding, and a statistical "
               "summary of each response. A max/min ratio above 10 suggests a "
               "transformation.</p>",
    "graphcols": "<h3>Graph Columns</h3><p>Plot any design column against another, e.g. a "
                 "response against a factor, to spot trends and outliers before modelling.</p>",
    "evaluation": "<h3>Evaluation</h3><p>Checks the design before (or after) running it: "
                  "degrees of freedom for the model, lack of fit and pure error, standard "
                  "errors and VIFs of the coefficients (VIF 1 is ideal, &gt;10 is bad), power "
                  "and the leverage of each run. The FDS graph shows the prediction precision "
                  "over the design space.</p>",
    "notes": "<h3>Notes</h3><p>Free text saved with the design.</p>",
    "analysis": "<h3>Analysis</h3><p>Work through the tabs from left to right:</p><ol>"
                "<li><b>Transform</b>: keep None unless the Box-Cox plot or a max/min ratio "
                "&gt; 10 suggests otherwise.</li><li><b>Fit Summary</b>: the suggested model "
                "is highlighted (significant sequential p, insignificant lack of fit, highest "
                "adjusted and predicted R²). Split-plot two-level factorials show "
                "<b>Effects</b> instead: subplot and whole-plot effects on separate half-normal "
                "plots; click an effect to put it in the model.</li><li><b>Model</b>: accept "
                "the suggestion or "
                "pick terms (M = model, e = error); Auto Select chooses terms forward, "
                "backward or stepwise by p-value (alpha in / out), AICc or BIC.</li>"
                "<li><b>ANOVA</b>: model p &lt; 0.05 significant, lack of fit not "
                "significant, Adeq Precision &gt; 4; the coded and actual equations (one per "
                "categoric level). Split-plot designs show the <b>ANOVA (REML)</b>: whole-plot "
                "and subplot terms tested separately (Kenward-Roger) and the variance "
                "components.</li>"
                "<li><b>Diagnostics</b>: residuals should follow a straight line on the normal "
                "plot and scatter randomly against predicted and run order.</li><li><b>Model "
                "Graphs</b>: contour, 3D surface, interaction, perturbation; use the Factors "
                "Tool to choose the axes, move the other factors and pick categoric levels. "
                "Drag a 3D surface to rotate it (wheel = zoom, Reset View); click a contour "
                "plot to place a flag with the settings and the prediction.</li></ol>"
                "<p>p-value colours: <span style='color:#D0021B'>red &lt; 0.01</span>, "
                "<span style='color:#1E8C2E'>green &lt; 0.05</span>, <span style='color:"
                "#1F5FD0'>blue &lt; 0.10</span>, black not significant.</p>",
    "numerical": "<h3>Numerical optimization</h3><p>Give each response a goal (maximize, "
                 "minimize, target or in range) with lower and upper limits; optionally "
                 "constrain factors. Solutions lists settings with the highest overall "
                 "desirability D (0 = unacceptable, 1 = ideal); Ramps shows where each factor "
                 "and response sits; Graphs draws the desirability or a response around the "
                 "selected solution.</p>",
    "graphical": "<h3>Graphical optimization</h3><p>Set limits for the responses; the overlay "
                 "plot shades in yellow the region where all responses meet their limits.</p>",
    "prediction": "<h3>Point Prediction</h3><p>Predicted responses with confidence and "
                  "prediction intervals at any factor setting chosen with the sliders.</p>",
    "confirmation": "<h3>Confirmation</h3><p>Compare the mean of confirmation runs with the "
                    "prediction interval for that number of runs.</p>",
    "coefficients": "<h3>Coefficients Table</h3><p>Coded coefficients of every response's "
                    "model side by side, coloured by p-value.</p>",
}


class InfoDialog(QDialog):
    """Design Tools > Edit Factor and Response Info: names and units."""

    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Factor and Response Info")
        self.doc = doc
        form = QFormLayout()
        self.edits = []
        for j, f in enumerate(doc.factors):
            n, u = QLineEdit(f["name"]), QLineEdit(f.get("units", ""))
            self.edits.append(("f", j, n, u))
            form.addRow(QLabel(f"<b>Factor {j + 1}</b>"))
            form.addRow("Name:", n)
            form.addRow("Units:", u)
        for i, r in enumerate(doc.responses):
            n, u = QLineEdit(r["name"]), QLineEdit(r.get("units", ""))
            self.edits.append(("r", i, n, u))
            form.addRow(QLabel(f"<b>Response {i + 1}</b>"))
            form.addRow("Name:", n)
            form.addRow("Units:", u)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.apply)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(buttons)

    def apply(self) -> None:
        for kind, i, n, u in self.edits:
            if kind == "f":
                f = self.doc.factors[i]
                if (n.text(), u.text()) != (f["name"], f.get("units", "")):
                    self.doc.set_factor_info(i, n.text().strip(), u.text().strip())
            else:
                r = self.doc.responses[i]
                if (n.text(), u.text()) != (r["name"], r.get("units", "")):
                    self.doc.set_response_info(i, n.text().strip(), u.text().strip())
        self.accept()


class FoldoverDialog(QDialog):
    """Design Tools > Augment Design > Foldover: full, single-factor or semifold foldover of a
    two-level design, added as a new block."""

    KINDS = [("full", "Full foldover (reverse every factor)"),
             ("single", "Single-factor foldover"), ("semifold", "Semifold (half the runs)")]

    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Augment Design: Foldover")
        self.doc = doc
        self.added = 0
        self.kind = QComboBox()
        for key, text in self.KINDS:
            self.kind.addItem(text, key)
        self.factor, self.keep = QComboBox(), QComboBox()
        for j in range(len(doc.factors)):
            self.factor.addItem(doc.factor_title(j), j)
            self.keep.addItem(doc.factor_title(j), j)
        if self.keep.count() > 1:
            self.keep.setCurrentIndex(1)
        self.level = QComboBox()
        self.level.addItem("High (+)", 1.0)
        self.level.addItem("Low (-)", -1.0)
        self.info = QLabel()
        self.info.setWordWrap(True)
        form = QFormLayout()
        form.addRow("Foldover:", self.kind)
        form.addRow("Fold on factor:", self.factor)
        form.addRow("Semifold keeps runs where:", self.keep)
        form.addRow("is at level:", self.level)
        for w in (self.kind, self.factor, self.keep, self.level):
            w.currentIndexChanged.connect(self._update)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.apply)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(self.info)
        lay.addWidget(buttons)
        self._update()

    def _update(self) -> None:
        kind = self.kind.currentData()
        self.factor.setEnabled(kind != "full")
        self.keep.setEnabled(kind == "semifold")
        self.level.setEnabled(kind == "semifold")
        self.info.setText({
            "full": "Reverses the signs of every factor. A resolution III fraction becomes "
                    "resolution IV: all main effects clear of two-factor interactions.",
            "single": "Reverses only the chosen factor: it and all its two-factor "
                      "interactions are de-aliased.",
            "semifold": "Half of the single-factor foldover (the runs where the second factor "
                        "is at the chosen level): de-aliases the chosen factor's two-factor "
                        "interactions in half the runs."}[kind]
            + " The new runs form a new block.")

    def apply(self) -> None:
        try:
            self.added = self.doc.foldover(self.kind.currentData(), self.factor.currentData(),
                                           self.keep.currentData(), self.level.currentData())
        except ValueError as exc:
            self.info.setText(f"<span style='color:#B00020'>{exc}</span>")
            return
        self.accept()


class DXMainWindow(QMainWindow):
    def __init__(self, ctx, doc: DesignDocument | None = None):
        super().__init__()
        self.ctx = ctx
        self.doc: DesignDocument | None = None
        self.undo = QUndoStack(self)
        self.pages: dict[str, QWidget] = {}
        self.stale: set[str] = set()
        self.current_key = "start"
        self.wizard = None
        self._closing = False
        self.setStyleSheet(style.STYLESHEET)
        self.setWindowIcon(icons.icon("surface"))
        self.resize(1280, 820)
        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self._build_body()
        self.statusBar().showMessage("Ready")
        if doc is not None:
            self.set_document(doc)
        else:
            self.show_start()

    # ------------------------------------------------------------------ settings
    def setting(self, key: str, default=None):
        v = self.ctx.settings.value(f"dx/{key}", default)
        if isinstance(default, bool):
            return v in (True, "true", "1", 1)
        return v

    def set_setting(self, key: str, value) -> None:
        self.ctx.settings.setValue(f"dx/{key}", value)

    # ------------------------------------------------------------------ actions
    def _build_actions(self) -> None:
        A = make_action
        self.a_new = A(self, "&New Design...", self.new_design, QKeySequence.StandardKey.New,
                       "new", tip="Create a new design")
        self.a_open = A(self, "&Open Design...", self.open_design,
                        QKeySequence.StandardKey.Open, "open", tip="Open a design (.rsx)")
        self.a_save = A(self, "&Save", self.save, QKeySequence.StandardKey.Save, "save",
                        tip="Save the design")
        self.a_save_as = A(self, "Save &As...", self.save_as, QKeySequence("Ctrl+Shift+S"))
        self.a_import = A(self, "&Import Responses (CSV / Excel)...", self.import_responses,
                          icon_name="paste")
        self.a_export_layout = A(self, "Export Design &Layout (CSV / Excel)...",
                                 self.export_layout)
        self.a_export_graph = A(self, "Export &Graph As...", self.export_graph,
                                icon_name="export", tip="Save the current graph as an image")
        self.a_close = A(self, "&Close", self.close, QKeySequence.StandardKey.Close)
        self.a_undo = self.undo.createUndoAction(self, "&Undo")
        self.a_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.a_undo.setIcon(icons.icon("undo"))
        self.a_redo = self.undo.createRedoAction(self, "&Redo")
        self.a_redo.setShortcut(QKeySequence.StandardKey.Redo)
        self.a_redo.setIcon(icons.icon("redo"))
        self.a_copy = A(self, "&Copy", self.copy, QKeySequence.StandardKey.Copy, "copy",
                        tip="Copy cells, tables (for Word / Excel) or the graph")
        self.a_paste = A(self, "&Paste", self.paste, QKeySequence.StandardKey.Paste, "paste",
                         tip="Paste response data into the design layout")
        self.a_copy_graph = A(self, "Copy &Graph", self.copy_graph)
        self.a_nav = A(self, "&Navigation Pane", None, checkable=True)
        self.a_nav.setChecked(True)
        self.a_nav.toggled.connect(lambda on: self.nav_frame.setVisible(on))
        self.a_toolbar = A(self, "&Toolbar", None, checkable=True)
        self.a_toolbar.setChecked(True)
        self.a_annot = A(self, "&Annotated ANOVA", self._toggle_annotate, checkable=True,
                         tip="Explanatory notes under the ANOVA and Fit Summary tables")
        self.a_annot.setChecked(self.setting("annotate", True))
        self.a_actual = A(self, "&Actual", lambda: self.set_coded(False), checkable=True)
        self.a_coded = A(self, "&Coded", lambda: self.set_coded(True), checkable=True,
                         icon_name="coded")
        grp = QActionGroup(self)
        grp.addAction(self.a_actual)
        grp.addAction(self.a_coded)
        self.a_actual.setChecked(True)
        self.a_type = A(self, "Show &Space Point Type", self._toggle_type, checkable=True)
        self.a_points = A(self, "Show &Design Points on Graphs", self._toggle_points,
                          checkable=True)
        self.a_points.setChecked(True)
        self.a_sort_std = A(self, "Sort by &Std Order", lambda: self.sort("std"))
        self.a_sort_run = A(self, "Sort by &Run Order", lambda: self.sort("run"))
        self.a_randomize = A(self, "Randomi&ze Run Order...", self.randomize)
        self.a_info = A(self, "&Edit Factor and Response Info...", self.edit_info)
        self.a_foldover = A(self, "Augment Design: &Foldover...", self.foldover,
                            icon_name="foldover",
                            tip="Add the foldover (full, single-factor or semifold) of a "
                                "two-level design as a new block")
        self.a_eval = A(self, "&Evaluate Design", lambda: self.show_node("evaluation"),
                        icon_name="evaluation")
        self.a_numerical = A(self, "&Numerical", lambda: self.show_node("numerical"),
                             icon_name="numerical")
        self.a_graphical = A(self, "&Graphical", lambda: self.show_node("graphical"),
                             icon_name="graphical")
        self.a_prediction = A(self, "&Point Prediction", lambda: self.show_node("prediction"),
                              icon_name="prediction")
        self.a_confirmation = A(self, "&Confirmation", lambda: self.show_node("confirmation"),
                                icon_name="confirmation")
        self.a_coefficients = A(self, "Coefficients &Table",
                                lambda: self.show_node("coefficients"),
                                icon_name="coefficients")
        self.a_tips = A(self, "Screen &Tips", self.show_tips, QKeySequence("F1"), "tips",
                        tip="Explain the current screen")
        self.a_updates = A(self, "Check for &Updates...",
                           lambda: self.ctx.check_for_updates(manual=True))
        self.a_about = A(self, f"&About {APP_NAME}", lambda: self.ctx.about())
        self.design_actions = [self.a_save, self.a_save_as, self.a_import,
                               self.a_export_layout, self.a_copy, self.a_paste, self.a_coded,
                               self.a_actual, self.a_type, self.a_sort_std, self.a_sort_run,
                               self.a_randomize, self.a_info, self.a_foldover, self.a_eval,
                               self.a_numerical,
                               self.a_graphical, self.a_prediction, self.a_confirmation,
                               self.a_coefficients, self.a_export_graph, self.a_copy_graph]

    def _build_menus(self) -> None:
        mb = self.menuBar()
        m = mb.addMenu("&File")
        m.addActions([self.a_new, self.a_open])
        self.recent_menu = m.addMenu("Open &Recent")
        self.recent_menu.aboutToShow.connect(self._fill_recent)
        m.addSeparator()
        m.addActions([self.a_save, self.a_save_as])
        m.addSeparator()
        m.addActions([self.a_import, self.a_export_layout, self.a_export_graph])
        m.addSeparator()
        m.addAction(self.a_close)
        m = mb.addMenu("&Edit")
        m.addActions([self.a_undo, self.a_redo])
        m.addSeparator()
        m.addActions([self.a_copy, self.a_paste, self.a_copy_graph])
        m = mb.addMenu("&View")
        m.addActions([self.a_nav, self.a_toolbar])
        m.addSeparator()
        m.addAction(self.a_annot)
        m = mb.addMenu("&Display Options")
        pf = m.addMenu("&Process Factors")
        pf.addActions([self.a_actual, self.a_coded])
        m.addAction(self.a_type)
        m.addAction(self.a_points)
        m = mb.addMenu("Design &Tools")
        m.addActions([self.a_sort_std, self.a_sort_run, self.a_randomize])
        m.addSeparator()
        m.addActions([self.a_info, self.a_eval])
        m.addSeparator()
        m.addAction(self.a_foldover)
        m.aboutToShow.connect(lambda: self.a_foldover.setEnabled(
            self.doc is not None and self.doc.two_level()))
        m.addSeparator()
        m.addAction(self.a_import)
        self.analysis_menu = mb.addMenu("&Analysis")
        self.analysis_menu.aboutToShow.connect(self._fill_analysis_menu)
        m = mb.addMenu("&Optimization")
        m.addActions([self.a_numerical, self.a_graphical])
        m.addSeparator()
        sec = m.addAction("Post Analysis")
        sec.setEnabled(False)
        m.addActions([self.a_prediction, self.a_confirmation, self.a_coefficients])
        self.window_menu = mb.addMenu("&Window")
        self.window_menu.aboutToShow.connect(self._fill_window_menu)
        m = mb.addMenu("&Help")
        m.addAction(self.a_tips)
        tut = m.addMenu("Tutorial &Data")
        from rsstatchem.ui.dx.examples import EXAMPLES

        for key, (text, _) in EXAMPLES.items():
            tut.addAction(text, lambda _=False, k=key: self.open_example(k))
        m.addSeparator()
        m.addAction(self.a_updates)
        m.addAction(self.a_about)

    def _build_toolbar(self) -> None:
        tb = self.addToolBar("Standard")
        tb.setObjectName("dxToolbar")
        tb.setMovable(False)
        tb.setIconSize(QSize(24, 24))
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        tb.addActions([self.a_new, self.a_open, self.a_save])
        tb.addSeparator()
        tb.addActions([self.a_undo, self.a_redo, self.a_copy, self.a_paste, self.a_export_graph])
        tb.addSeparator()
        tb.addAction(self.a_coded)
        tb.addSeparator()
        tb.addActions([self.a_eval, self.a_numerical, self.a_graphical, self.a_prediction])
        tb.addSeparator()
        tb.addAction(self.a_tips)
        self.toolbar = tb
        self.a_toolbar.toggled.connect(tb.setVisible)

    def _build_body(self) -> None:
        self.nav_frame = QWidget()
        nl = QVBoxLayout(self.nav_frame)
        nl.setContentsMargins(0, 0, 0, 0)
        nl.setSpacing(0)
        self.nav_title = QLabel("Navigation")
        self.nav_title.setObjectName("navTitle")
        nl.addWidget(self.nav_title)
        self.nav = QTreeWidget()
        self.nav.setObjectName("navTree")
        self.nav.setHeaderHidden(True)
        self.nav.setIconSize(QSize(18, 18))
        self.nav.setIndentation(16)
        self.nav.itemSelectionChanged.connect(self._nav_selected)
        nl.addWidget(self.nav, 1)
        self.stack = QStackedWidget()
        split = QSplitter()
        split.addWidget(self.nav_frame)
        split.addWidget(self.stack)
        split.setStretchFactor(1, 1)
        split.setSizes([230, 1050])
        split.setCollapsible(1, False)
        self.setCentralWidget(split)

    # ------------------------------------------------------------------ document
    def show_start(self) -> None:
        from rsstatchem.ui.dx.start import StartPage

        recent = [p for p in self.ctx.recent_files() if p.lower().endswith(".rsx")][:8]
        page = StartPage(recent)
        page.new_design.connect(self.new_design)
        page.open_design.connect(self.open_design)
        page.open_path.connect(self.open_path)
        page.open_example.connect(self.open_example)
        self.start_page = page
        self.stack.addWidget(page)
        self.stack.setCurrentWidget(page)
        self.nav.clear()
        empty = QTreeWidgetItem(["(no design open)"])
        empty.setFlags(Qt.ItemFlag.NoItemFlags)
        self.nav.addTopLevelItem(empty)
        for a in self.design_actions:
            a.setEnabled(False)
        self._update_title()

    def set_document(self, doc: DesignDocument) -> None:
        if self.doc is not None:
            self.doc.unsubscribe(self._doc_changed)
        self.doc = doc
        doc.subscribe(self._doc_changed)
        self.undo.clear()
        for key, page in list(self.pages.items()):
            self.stack.removeWidget(page)
            page.deleteLater()
        self.pages.clear()
        coded = bool(doc.ui.get("coded", False))
        self.a_coded.setChecked(coded)
        self.a_actual.setChecked(not coded)
        for a in self.design_actions:
            a.setEnabled(True)
        self._build_nav()
        self.show_node("design")
        self._update_title()

    def _build_nav(self) -> None:
        doc = self.doc
        self.nav.blockSignals(True)
        self.nav.clear()
        self.nav_items: dict[str, QTreeWidgetItem] = {}

        def add(parent, key, text, icon_name, bold=False):
            it = QTreeWidgetItem([text])
            it.setData(0, Qt.ItemDataRole.UserRole, key)
            it.setIcon(0, icons.icon(icon_name))
            if bold:
                f = it.font(0)
                f.setBold(True)
                it.setFont(0, f)
            if parent is None:
                self.nav.addTopLevelItem(it)
            else:
                parent.addChild(it)
            if key:
                self.nav_items[key] = it
            return it

        coded = self.a_coded.isChecked()
        design = add(None, "design", "Design (Coded)" if coded else "Design (Actual)", "design",
                     True)
        add(design, "summary", "Summary", "summary")
        add(design, "graphcols", "Graph Columns", "graphcols")
        add(design, "evaluation", "Evaluation", "evaluation")
        add(design, "notes", "Notes", "notes")
        analysis = add(None, "", "Analysis", "folder", True)
        analysis.setFlags(Qt.ItemFlag.ItemIsEnabled)
        for i, col in enumerate(doc.response_cols):
            add(analysis, f"analysis:{col}", self._response_label(i), "response")
        opt = add(None, "", "Optimization", "folder", True)
        opt.setFlags(Qt.ItemFlag.ItemIsEnabled)
        add(opt, "numerical", "Numerical", "numerical")
        add(opt, "graphical", "Graphical", "graphical")
        post = add(None, "", "Post Analysis", "folder", True)
        post.setFlags(Qt.ItemFlag.ItemIsEnabled)
        add(post, "prediction", "Point Prediction", "prediction")
        add(post, "confirmation", "Confirmation", "confirmation")
        add(post, "coefficients", "Coefficients Table", "coefficients")
        self.nav.expandAll()
        self.nav.blockSignals(False)
        self._update_nav_labels()

    def _response_label(self, i: int) -> str:
        doc = self.doc
        col = doc.response_cols[i]
        n = doc.response_count(col)
        label = doc.response_title(i)
        if n == 0:
            return label + " (empty)"
        if n < doc.n_runs - len(doc.ignored):
            return label + f" ({n}/{doc.n_runs - len(doc.ignored)})"
        return label

    def _update_nav_labels(self) -> None:
        if self.doc is None:
            return
        for i, col in enumerate(self.doc.response_cols):
            it = self.nav_items.get(f"analysis:{col}")
            if it is not None:
                it.setText(0, self._response_label(i))
                n = self.doc.response_count(col)
                it.setIcon(0, icons.icon("response" if n else "response_empty"))
        it = self.nav_items.get("design")
        if it is not None:
            it.setText(0, "Design (Coded)" if self.a_coded.isChecked() else "Design (Actual)")

    def _nav_selected(self) -> None:
        items = self.nav.selectedItems()
        if not items:
            return
        key = items[0].data(0, Qt.ItemDataRole.UserRole)
        if key:
            self.show_node(key, from_nav=True)

    def _make_page(self, key: str) -> QWidget:
        from rsstatchem.ui.dx import info, optimization
        from rsstatchem.ui.dx.analysis import AnalysisView

        doc = self.doc
        if key == "design":
            page = info.DesignPage(doc, self.undo)
            page.model.set_coded(self.a_coded.isChecked())
            page.model.set_show_type(self.a_type.isChecked())
            page.view.status.connect(lambda t: self.statusBar().showMessage(t, 6000))
        elif key == "summary":
            page = info.SummaryPage(doc)
        elif key == "graphcols":
            page = info.GraphColumnsPage(doc)
        elif key == "evaluation":
            page = info.EvaluationPage(doc)
        elif key == "notes":
            page = info.NotesPage(doc)
        elif key.startswith("analysis:"):
            page = AnalysisView(doc, key.split(":", 1)[1])
            page.set_annotate(self.a_annot.isChecked())
            page.show_points = self.a_points.isChecked()
            page.go_to_layout.connect(lambda: self.show_node("design"))
        elif key == "numerical":
            page = optimization.NumericalView(doc)
        elif key == "graphical":
            page = optimization.GraphicalView(doc)
        elif key == "prediction":
            page = optimization.PointPredictionView(doc)
        elif key == "confirmation":
            page = optimization.ConfirmationView(doc)
        elif key == "coefficients":
            page = optimization.CoefficientsView(doc)
        else:
            raise KeyError(key)
        return page

    def page(self, key: str) -> QWidget:
        if key not in self.pages:
            page = self._make_page(key)
            self.pages[key] = page
            self.stack.addWidget(page)
            self.stale.discard(key)
        return self.pages[key]

    def show_node(self, key: str, from_nav: bool = False) -> QWidget | None:
        if self.doc is None:
            return None
        page = self.page(key)
        if key in self.stale:
            self.stale.discard(key)
            self._refresh_page(key, page)
        self.stack.setCurrentWidget(page)
        self.current_key = key
        if not from_nav and key in getattr(self, "nav_items", {}):
            self.nav.blockSignals(True)
            self.nav.setCurrentItem(self.nav_items[key])
            self.nav.blockSignals(False)
        tips = {"design": "Type or paste the measured responses into the response columns.",
                "numerical": "Set goals on Criteria, then look at Solutions and Graphs.",
                "graphical": "Set response limits, then look at the overlay plot on Graphs."}
        if key.startswith("analysis:"):
            self.statusBar().showMessage("Work through the tabs from left to right: Transform, "
                                         "Fit Summary, Model, ANOVA, Diagnostics, Model "
                                         "Graphs.  (F1: Screen Tips)")
        else:
            self.statusBar().showMessage(tips.get(key, "F1: Screen Tips for this screen"))
        return page

    def _refresh_page(self, key: str, page) -> None:
        if key.startswith("analysis:"):
            page.invalidate()
        elif hasattr(page, "refresh"):
            page.refresh()

    # ------------------------------------------------------------------ change notification
    def _doc_changed(self, kind: str, detail=None) -> None:
        if self._closing:
            return
        cols = set(detail or []) if kind == "data" else set()
        for key, page in self.pages.items():
            affected = False
            if kind == "data":
                affected = key not in ("notes",) and (not key.startswith("analysis:") or
                                                      key.split(":", 1)[1] in cols or not cols)
            elif kind == "analysis":
                affected = key in ("summary", "numerical", "graphical", "prediction",
                                   "confirmation", "coefficients") or key == f"analysis:{detail}"
            elif kind == "meta":
                affected = key != "notes"
            if not affected:
                continue
            if key == "design" and kind in ("data", "meta"):
                if kind == "meta":
                    page.model.rebuild()
                page.refresh()
            elif page is self.stack.currentWidget():
                self._refresh_page(key, page)
            else:
                self.stale.add(key)
        if kind in ("data", "meta"):
            self._update_nav_labels()
        self._update_title()

    def _update_title(self) -> None:
        if self.doc is None:
            self.setWindowTitle(f"RSM / DOE - {APP_NAME}")
            self.setWindowModified(False)
            return
        name = Path(self.doc.path).name if self.doc.path else f"{self.doc.name} (unsaved)"
        self.setWindowTitle(f"{name}[*] - RSM / DOE - {APP_NAME}")
        self.setWindowModified(self.doc.modified)
        self.nav_title.setText(f"  {self.doc.name}")

    # ------------------------------------------------------------------ file
    def new_design(self) -> None:
        from rsstatchem.ui.dx.wizard import NewDesignWizard

        if self.wizard is not None and self.wizard.isVisible():
            self.wizard.raise_()
            return
        self.wizard = NewDesignWizard(self)
        self.wizard.created.connect(self._design_created)
        self.wizard.open()

    def _design_created(self, doc: DesignDocument) -> None:
        self.adopt(doc)

    def adopt(self, doc: DesignDocument):
        """Show ``doc`` here if this window is empty, else in a new RSM window."""
        if self.doc is None:
            self.set_document(doc)
            return self
        win = DXMainWindow(self.ctx, doc)
        self.ctx.register_window(win)
        win.move(self.pos() + QPoint(28, 28))
        win.show()
        return win

    def _last_dir(self) -> str:
        return str(self.setting("last_dir", str(Path.home())))

    def open_design(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open Design", self._last_dir(),
                                              "Design files (*.rsx);;All files (*)")
        if path:
            self.open_path(path)

    def open_path(self, path: str):
        try:
            doc = DesignDocument.load(path)
        except (OSError, ValueError, KeyError) as exc:
            self.error(f"Could not open {Path(path).name}:\n{exc}")
            return None
        self.set_setting("last_dir", str(Path(path).parent))
        self.ctx.add_recent(path)
        return self.adopt(doc)

    def open_example(self, key: str):
        from rsstatchem.ui.dx.examples import EXAMPLES

        return self.adopt(EXAMPLES[key][1]())

    def save(self) -> bool:
        if self.doc is None:
            return False
        if not self.doc.path:
            return self.save_as()
        return self._write(self.doc.path)

    def save_as(self) -> bool:
        if self.doc is None:
            return False
        start = str(Path(self._last_dir()) / f"{self.doc.name}.rsx")
        path, _ = QFileDialog.getSaveFileName(self, "Save Design As", start,
                                              "Design files (*.rsx)")
        if not path:
            return False
        if not path.lower().endswith(".rsx"):
            path += ".rsx"
        return self._write(path)

    def _write(self, path: str) -> bool:
        notes = self.pages.get("notes")
        if notes is not None:
            notes.commit()
        self.doc.ui["coded"] = self.a_coded.isChecked()
        try:
            self.doc.save(path)
        except OSError as exc:
            self.error(f"Could not save the design:\n{exc}")
            return False
        self.set_setting("last_dir", str(Path(path).parent))
        self.ctx.add_recent(path)
        self.statusBar().showMessage(f"Saved {path}", 5000)
        self._update_title()
        return True

    def import_responses(self) -> None:
        if self.doc is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Import Responses", self._last_dir(),
                                              "Tables (*.csv *.txt *.xlsx *.xls)")
        if path:
            self.import_responses_from(path)

    def import_responses_from(self, path: str) -> list[str]:
        try:
            if path.lower().endswith((".xlsx", ".xls")):
                frame = pd.read_excel(path)
            else:
                frame = pd.read_csv(path, sep=None, engine="python")
        except (OSError, ValueError, ImportError) as exc:
            self.error(f"Could not read {Path(path).name}:\n{exc}")
            return []
        before = {c: self.doc.ds.df[c].copy() for c in self.doc.response_cols}
        filled = self.doc.import_responses(frame)
        if not filled:
            self.error("No column of the file matches a response name (e.g. "
                       f"'{self.doc.responses[0]['name']}' or 'R1').")
            return []
        from rsstatchem.ui.dx.layout import SetValuesCommand

        changes = []
        for c, old in before.items():
            new = self.doc.ds.df[c]
            for row in range(len(new)):
                o, n = float(old.iloc[row]), float(new.iloc[row])
                if not (o == n or (o != o and n != n)):
                    changes.append((row, c, o, n))
        cmd = SetValuesCommand(self.doc, changes, "Import Responses")
        self.undo.push(cmd)  # redo() re-applies the same values: harmless
        self.statusBar().showMessage("Imported " + ", ".join(filled), 6000)
        return filled

    def export_layout(self) -> None:
        if self.doc is None:
            return
        start = str(Path(self._last_dir()) / f"{self.doc.name}.csv")
        path, _ = QFileDialog.getSaveFileName(self, "Export Design Layout", start,
                                              "CSV (*.csv);;Excel (*.xlsx)")
        if path:
            self.export_layout_to(path)

    def export_layout_to(self, path: str) -> None:
        doc = self.doc
        df = doc.ds.df.iloc[doc.row_order("run")].copy()
        df = df.rename(columns={c: doc.column_title(c) for c in doc.factor_cols +
                                doc.response_cols})
        if path.lower().endswith(".xlsx"):
            df.to_excel(path, index=False)
        else:
            df.to_csv(path, index=False)

    # ------------------------------------------------------------------ edit
    def current_page(self):
        return self.stack.currentWidget()

    def current_graph(self) -> GraphView | None:
        page = self.current_page()
        if hasattr(page, "current_graph"):
            return page.current_graph()
        g = getattr(page, "graph", None)
        return g if isinstance(g, GraphView) else None

    def current_report(self) -> ReportView | None:
        page = self.current_page()
        if hasattr(page, "current_report"):
            return page.current_report()
        r = getattr(page, "report", None)
        return r if isinstance(r, ReportView) else None

    def copy(self) -> None:
        page = self.current_page()
        focus = self.focusWidget()
        if self.current_key == "design":
            page.view.copy()
            return
        if hasattr(focus, "copy") and focus.__class__.__name__ in ("QLineEdit",
                                                                   "QPlainTextEdit"):
            focus.copy()
            return
        rep = self.current_report()
        if isinstance(focus, ReportView) and focus.textCursor().hasSelection():
            focus.copy()
            return
        g = self.current_graph()
        if g is not None and g.has_graph and (rep is None or not rep.isVisible()):
            g.copy_graph()
            self.statusBar().showMessage("Graph copied to the clipboard.", 4000)
            return
        if rep is not None:
            rep.copy_all()
            self.statusBar().showMessage("Tables copied (paste into Word or Excel).", 4000)

    def paste(self) -> None:
        focus = self.focusWidget()
        if self.current_key == "design":
            self.page("design").view.paste()
        elif hasattr(focus, "paste"):
            focus.paste()

    def copy_graph(self) -> None:
        g = self.current_graph()
        if g is not None:
            g.copy_graph()

    def export_graph(self) -> None:
        g = self.current_graph()
        if g is None or not g.has_graph:
            self.statusBar().showMessage("No graph on this screen.", 4000)
            return
        g.export_graph()

    # ------------------------------------------------------------------ display options
    def set_coded(self, coded: bool) -> None:
        (self.a_coded if coded else self.a_actual).setChecked(True)
        if self.doc is None:
            return
        self.doc.ui["coded"] = coded
        if "design" in self.pages:
            self.pages["design"].model.set_coded(coded)
        self._update_nav_labels()

    def _toggle_type(self) -> None:
        if "design" in self.pages:
            self.pages["design"].model.set_show_type(self.a_type.isChecked())

    def _toggle_points(self) -> None:
        for key, page in self.pages.items():
            if key.startswith("analysis:"):
                page.set_show_points(self.a_points.isChecked())

    def _toggle_annotate(self) -> None:
        on = self.a_annot.isChecked()
        self.set_setting("annotate", on)
        for key, page in self.pages.items():
            if key.startswith("analysis:"):
                page.set_annotate(on)

    def sort(self, order: str) -> None:
        if self.doc is None:
            return
        self.show_node("design").model.set_order(order)

    def randomize(self) -> None:
        if self.doc is None:
            return
        if any(self.doc.response_count(c) for c in self.doc.response_cols):
            box = QMessageBox(QMessageBox.Icon.Question, "Randomize Run Order",
                              "Responses have been entered. Re-randomizing changes the run "
                              "numbers only (the data stay with their runs). Continue?",
                              QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                              self)
            if box.exec() != QMessageBox.StandardButton.Yes:
                return
        import random

        self.doc.randomize(random.randint(1, 99999))
        self.undo.clear()
        self.sort("run")

    def foldover(self) -> FoldoverDialog | None:
        if self.doc is None:
            return None
        if not self.doc.two_level():
            self.error("Foldover is for two-level designs (factorial, Plackett-Burman, "
                       "min-run) without split-plot groups.")
            return None
        dlg = FoldoverDialog(self.doc, self)
        dlg.setStyleSheet(style.STYLESHEET)

        def done(result):
            if result and dlg.added:
                self.undo.clear()
                self.statusBar().showMessage(f"Foldover: {dlg.added} runs added as block "
                                             f"{self.doc.n_blocks}.", 6000)
                self.sort("std")

        dlg.finished.connect(done)
        dlg.open()
        return dlg

    def edit_info(self) -> None:
        if self.doc is None:
            return
        dlg = InfoDialog(self.doc, self)
        dlg.setStyleSheet(style.STYLESHEET)
        dlg.open()

    # ------------------------------------------------------------------ menus
    def _fill_recent(self) -> None:
        self.recent_menu.clear()
        recent = [p for p in self.ctx.recent_files() if p.lower().endswith(".rsx")]
        for p in recent:
            self.recent_menu.addAction(Path(p).name, lambda _=False, pp=p: self.open_path(pp))
        if not recent:
            a = self.recent_menu.addAction("(none)")
            a.setEnabled(False)

    def _fill_analysis_menu(self) -> None:
        m = self.analysis_menu
        m.clear()
        if self.doc is None:
            a = m.addAction("(create or open a design first)")
            a.setEnabled(False)
            return
        for i, col in enumerate(self.doc.response_cols):
            m.addAction(icons.icon("response"), self._response_label(i),
                        lambda _=False, c=col: self.show_node(f"analysis:{c}"))
        if self.current_key.startswith("analysis:"):
            m.addSeparator()
            view = self.pages[self.current_key]
            for name in ("Transform", "Fit Summary", "Model", "ANOVA", "Diagnostics",
                         "Model Graphs"):
                m.addAction(name, lambda _=False, n=name, v=view: v.set_tab(n))

    def _fill_window_menu(self) -> None:
        from rsstatchem.ui.modules import add_module_menu

        m = self.window_menu
        m.clear()
        add_module_menu(m, self.ctx)
        others = [w for w in self.ctx.windows if isinstance(w, DXMainWindow)]
        if others:
            m.addSeparator()
            for w in others:
                a = m.addAction(w.windowTitle().replace("[*]", ""),
                                lambda _=False, ww=w: (ww.show(), ww.raise_(),
                                                       ww.activateWindow()))
                a.setCheckable(True)
                a.setChecked(w is self)

    # ------------------------------------------------------------------ help
    def show_tips(self) -> None:
        key = self.current_key if self.doc is not None else "start"
        text = TIPS.get("analysis" if key.startswith("analysis:") else key, TIPS["start"])
        dlg = QDialog(self)
        dlg.setWindowTitle("Screen Tips")
        dlg.setStyleSheet(style.STYLESHEET)
        browser = QTextBrowser()
        browser.setHtml(text)
        lay = QVBoxLayout(dlg)
        lay.addWidget(browser)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dlg.close)
        lay.addWidget(buttons)
        dlg.resize(520, 420)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dlg.show()

    def error(self, text: str) -> None:
        self.statusBar().showMessage(text.replace("\n", " "), 10000)
        box = QMessageBox(QMessageBox.Icon.Warning, "RSM / DOE", text,
                          QMessageBox.StandardButton.Ok, self)
        box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        box.open()

    # ------------------------------------------------------------------ closing
    def maybe_save(self) -> bool:
        if self.doc is None or not self.doc.modified:
            return True
        box = QMessageBox(QMessageBox.Icon.Question, "RSM / DOE",
                          f"Save the changes to {self.doc.name}?",
                          QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
                          | QMessageBox.StandardButton.Cancel, self)
        box.setDefaultButton(QMessageBox.StandardButton.Save)
        answer = box.exec()
        if answer == QMessageBox.StandardButton.Save:
            return self.save()
        return answer == QMessageBox.StandardButton.Discard

    def closeEvent(self, event) -> None:
        notes = self.pages.get("notes")
        if notes is not None:
            notes.commit()
        if not self.maybe_save():
            event.ignore()
            return
        event.accept()
        self._closing = True
        if self.doc is not None:
            self.doc.unsubscribe(self._doc_changed)
        if self.wizard is not None:
            self.wizard.close()
        self.ctx.window_closed(self)


def open_wizard_later(win: DXMainWindow) -> None:
    """Show the New Design wizard on an empty workspace once the window is up (non-modal)."""
    def go():
        if win.doc is None and win.isVisible() and not win._closing:
            win.new_design()

    QTimer.singleShot(0, win, go)


__all__ = ["DXMainWindow", "TIPS", "open_wizard_later"]

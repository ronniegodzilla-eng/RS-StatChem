"""The RSM / DOE main window, laid out like Stat-Ease Design-Expert.

Menu bar (File, Edit, View, Display Options, Design Tools, Analysis, Optimization, Window,
Help), a toolbar, the navigation pane on the left and the selected node on the right:

    Design (Actual)        the design layout
        Summary / Graph Columns / Evaluation / Constraints / Notes
    Analysis               one node per response (R1:Yield ...) [+ Analysis Summary]
    Optimization           Numerical, Graphical
    Post Analysis          Point Prediction, Confirmation, Coefficients Table

Design Tools > Augment Design: foldover, centre points, axial points (factorial -> CCD),
replicates, optimal augmentation for a new model, Modify Design Space and the constraints;
Design Tools > Edit Layout: insert / delete runs, add a response, round factor values, rename
blocks (all undoable). File > Import Data Set builds a historical-data design from a table.

Views published by the analysis and optimisation packages are wired in through
:data:`EXTENSION_NODES` when their classes exist (Analysis Summary, Propagation of Error).

A window holds at most one design; New Design / Open in a window that already has one opens a
new window. Without a design the window shows a start page and (from the launcher) the New
Design wizard, shown non-modally.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import QPoint, QSize, Qt, QTimer
from PySide6.QtGui import QActionGroup, QKeySequence, QUndoStack
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QMainWindow, QMessageBox, QPushButton, QSpinBox,
                               QSplitter, QStackedWidget, QTextBrowser, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

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
    "constraints": "<h3>Constraints</h3><p>Multiple linear constraints of the design region "
                   "in actual units, e.g. <b>A + B &lt;= 15</b> or <b>0.2 &lt;= A + 2B &lt;= "
                   "1</b>, with the vertices of the region and the runs that fall outside it. "
                   "Edit the constraints, then augment the design to place runs in the new "
                   "region.</p>",
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


# Views from the analysis (WP-B) and optimisation (WP-C) packages, shown when the class exists:
# (node key, parent folder, text, module, class name, icon)
EXTENSION_NODES = [
    ("analysis_summary", "analysis", "Analysis Summary", "rsstatchem.ui.dx.analysis",
     "AnalysisSummaryView", "analysis_summary"),
    ("poe", "optimization", "Propagation of Error", "rsstatchem.ui.dx.optimization",
     "POEGraphView", "poe"),
]


def extension_class(module: str, name: str):
    import importlib

    try:
        return getattr(importlib.import_module(module), name, None)
    except ImportError:
        return None


class InfoDialog(QDialog):
    """Design Tools > Edit Factor and Response Info (column info sheet): names and units;
    for numeric factors the standard deviation used by propagation of error and the allowed
    discrete levels used by the numerical optimisation."""

    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Factor and Response Info")
        self.doc = doc
        form = QFormLayout()
        self.edits = []
        self.extra = []
        sds = doc.optimization.get("poe_sd", {}) or {}
        disc = doc.optimization.get("discrete", {}) or {}
        for j, f in enumerate(doc.factors):
            n, u = QLineEdit(f["name"]), QLineEdit(f.get("units", ""))
            self.edits.append(("f", j, n, u))
            form.addRow(QLabel(f"<b>Factor {j + 1}</b>"))
            form.addRow("Name:", n)
            form.addRow("Units:", u)
            if f.get("type") == "numeric":
                sd = sds.get(f["column"])
                sd_edit = QLineEdit("" if not sd else f"{sd:g}")
                sd_edit.setPlaceholderText("0 = no variation")
                sd_edit.setToolTip("Standard deviation of the factor setting in actual units "
                                   "(propagation of error).")
                levels = disc.get(f["column"]) or f.get("discrete_levels") or []
                lv_edit = QLineEdit(", ".join(f"{v:g}" for v in levels))
                lv_edit.setPlaceholderText("continuous")
                lv_edit.setToolTip("Allowed settings, separated by commas (optimisation only "
                                   "searches these levels).")
                form.addRow("Std. Dev. (POE):", sd_edit)
                form.addRow("Discrete levels:", lv_edit)
                self.extra.append((j, sd_edit, lv_edit))
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
        for j, sd_edit, lv_edit in self.extra:
            try:
                sd = float(sd_edit.text().replace(",", ".")) if sd_edit.text().strip() else 0.0
                levels = [float(v) for v in lv_edit.text().replace(";", ",").split(",")
                          if v.strip()]
            except ValueError:
                return
            self.doc.set_factor_variation(j, sd, levels)
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


class _FormDialog(QDialog):
    """Small Design Tools dialog: a form, an info line and OK / Cancel; ``action()`` does
    the edit and returns a status message (errors are shown in the dialog)."""

    title = "Design Tools"

    def __init__(self, win, parent=None):
        super().__init__(parent or win)
        self.win = win
        self.doc = win.doc
        self.setWindowTitle(self.title)
        self.setStyleSheet(style.STYLESHEET)
        self.form = QFormLayout()
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.message = ""
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.apply)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(self.form)
        lay.addWidget(self.info)
        lay.addWidget(buttons)

    def spin(self, label, lo, hi, value, tip=""):
        w = QSpinBox()
        w.setRange(lo, hi)
        w.setValue(value)
        if tip:
            w.setToolTip(tip)
        self.form.addRow(label, w)
        return w

    def dspin(self, label, lo, hi, value, decimals=4):
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setDecimals(decimals)
        w.setValue(value)
        self.form.addRow(label, w)
        return w

    def check(self, text, value=True):
        w = QCheckBox(text)
        w.setChecked(value)
        self.form.addRow("", w)
        return w

    def action(self) -> str:  # pragma: no cover - overridden
        return ""

    def apply(self) -> None:
        try:
            self.message = self.action() or ""
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.info.setText(f"<span style='color:#B00020'>{exc}</span>")
            return
        self.accept()


class CenterPointsDialog(_FormDialog):
    title = "Augment Design: Add Center Points"

    def __init__(self, win, parent=None):
        super().__init__(win, parent)
        self.n = self.spin("Center points:", 1, 50, 3)
        self.new_block = self.check("Put them in a new block", False)
        self.info.setText("Centre points estimate pure error and test for curvature "
                          "(numeric factors at the middle of their ranges; one set per "
                          "categoric level combination; the centroid of a mixture).")

    def action(self) -> str:
        n = self.win.augment_center(self.n.value(), self.new_block.isChecked())
        return f"Added {n} centre point(s)."


class AxialDialog(_FormDialog):
    title = "Augment Design: Add Axial Points"

    def __init__(self, win, parent=None):
        super().__init__(win, parent)
        self.alpha = self.dspin("Alpha (coded distance):", 0.1, 10, self.doc.default_alpha())
        self.centers = self.spin("Center points in the axial block:", 0, 20, 2)
        self.info.setText("Turns a two-level factorial into a central composite design: 2 "
                          "axial (star) points per numeric factor at ±alpha plus centre "
                          "points, in a new block. Alpha 1 keeps the axial points at the "
                          "factor limits (face centred); the default is rotatable "
                          "(nF^¼).")

    def action(self) -> str:
        n = self.win.augment_axial(self.alpha.value(), self.centers.value())
        return f"Added {n} axial and centre runs (block {self.doc.n_blocks})."


class ReplicatesDialog(_FormDialog):
    title = "Augment Design: Add Replicates"

    def __init__(self, win, parent=None):
        super().__init__(win, parent)
        self.times = self.spin("Extra replicates of every run:", 1, 10, 1)
        self.new_block = self.check("One new block per replicate", True)
        self.info.setText("Repeats every run of the design (empty responses).")

    def action(self) -> str:
        total = 0
        for _ in range(self.times.value()):
            total += self.win.augment_replicates(self.new_block.isChecked())
        return f"Added {total} replicate run(s)."


class OptimalAugmentDialog(_FormDialog):
    title = "Augment Design: Optimal"

    def __init__(self, win, parent=None):
        super().__init__(win, parent)
        from rsstatchem.ui.dx.info import model_choices

        self.terms = None
        self.order = QComboBox()
        for name, key in model_choices(self.doc):
            self.order.addItem(name, key)
        bigger = {"linear": "2fi", "2fi": "quadratic", "quadratic": "cubic",
                  "special_cubic": "cubic"}
        cur = self.doc.meta.get("model") or "linear"
        i = self.order.findData(bigger.get(cur, "quadratic"))
        self.order.setCurrentIndex(i if i >= 0 else self.order.count() - 1)
        self.form.addRow("New model:", self.order)
        self.edit_btn = QPushButton("Edit Model...")
        self.edit_btn.clicked.connect(self.edit_model)
        self.form.addRow("", self.edit_btn)
        self.runs = self.spin("Runs to add:", 1, 500, 4)
        self.lof = self.spin("Lack-of-fit points:", 0, 50, 0)
        self.criterion = QComboBox()
        for key, text in (("I", "I-optimal"), ("D", "D-optimal"), ("A", "A-optimal")):
            self.criterion.addItem(text, key)
        self.form.addRow("Optimality:", self.criterion)
        self.exchange = QComboBox()
        for key, text in (("point", "Point exchange"), ("coordinate", "Coordinate exchange"),
                          ("best", "Best")):
            self.exchange.addItem(text, key)
        self.form.addRow("Exchange:", self.exchange)
        self.new_block = self.check("Add the runs as a new block (phase)", True)
        self.order.currentIndexChanged.connect(lambda _: self._suggest())
        self._suggest()

    def current_terms(self):
        return self.terms if self.terms is not None else \
            self.doc.model_terms(self.order.currentData())

    def _suggest(self) -> None:
        self.terms = None
        try:
            n = self.doc.augment_default_runs(self.current_terms())
        except (ValueError, np.linalg.LinAlgError):
            n = 4
        self.runs.setValue(n)
        self.info.setText(f"The existing {self.doc.n_runs} runs are kept; {n} new runs (the "
                          "default) let the new model be estimated with some error degrees "
                          "of freedom.")

    def edit_model(self):
        from rsstatchem.core.doe import model as mdl
        from rsstatchem.ui.dx.wizard import TermsDialog

        sp = self.doc.space()
        if self.doc.combined:
            from rsstatchem.core.doe import terms_combined as tc

            cands, n_mix = tc.candidate_terms(sp), sp.n_mix
        else:
            cands, n_mix = mdl.candidate_terms(sp), None
        dlg = TermsDialog(cands, self.current_terms(), self, n_mix=n_mix)

        def done(result):
            if result:
                self.terms = dlg.selected()
                self.runs.setValue(self.doc.augment_default_runs(self.terms))
        dlg.finished.connect(done)
        dlg.open()
        return dlg

    def action(self) -> str:
        n = self.win.augment_optimal(self.current_terms(), self.runs.value(),
                                     self.criterion.currentData(),
                                     self.exchange.currentData(), self.lof.value(),
                                     self.new_block.isChecked())
        return f"Optimal augmentation: {n} run(s) added."


class ModifySpaceDialog(_FormDialog):
    title = "Modify Design Space"

    def __init__(self, win, parent=None):
        super().__init__(win, parent)
        self.rows = []
        for j, f in enumerate(self.doc.factors):
            if f.get("type") == "categoric":
                continue
            lo, hi = QLineEdit(f"{f['low']:g}"), QLineEdit(f"{f['high']:g}")
            box = QHBoxLayout()
            box.addWidget(QLabel("Low"))
            box.addWidget(lo)
            box.addWidget(QLabel("High"))
            box.addWidget(hi)
            self.form.addRow(self.doc.factor_title(j) + ":", box)
            self.rows.append((j, lo, hi))
        self.info.setText("New limits move, grow or shrink the design region. The runs keep "
                          "their actual settings; their coded values change. Augment the "
                          "design afterwards to add runs in the new region.")

    def action(self) -> str:
        changes = {}
        for j, lo, hi in self.rows:
            f = self.doc.factors[j]
            try:
                a, b = float(lo.text().replace(",", ".")), float(hi.text().replace(",", "."))
            except ValueError:
                raise ValueError(f"{f['name']}: enter numbers.") from None
            if (a, b) != (f["low"], f["high"]):
                changes[j] = (a, b)
        if not changes:
            return "Design space unchanged."
        self.win.modify_design_space(changes)
        return f"Design space changed for {len(changes)} factor(s)."


class RoundDialog(_FormDialog):
    title = "Round Factor Values"

    def __init__(self, win, parent=None):
        super().__init__(win, parent)
        self.mode = QComboBox()
        self.mode.addItem("Decimal places", "decimals")
        self.mode.addItem("Significant digits", "sig")
        self.form.addRow("Round to:", self.mode)
        self.digits = self.spin("Digits:", 0, 10, 2)
        self.info.setText("Rounds the numeric factor settings; mixture runs are adjusted so "
                          "the components still add up to the total.")

    def action(self) -> str:
        d = self.digits.value()
        if self.mode.currentData() == "sig":
            n = self.win.round_factors(sig=max(d, 1))
        else:
            n = self.win.round_factors(decimals=d)
        return f"Rounded {n} value(s)."


class AddResponseDialog(_FormDialog):
    title = "Add Response"

    def __init__(self, win, parent=None):
        super().__init__(win, parent)
        self.name = QLineEdit(f"R{len(self.doc.responses) + 1}")
        self.units = QLineEdit("")
        self.form.addRow("Name:", self.name)
        self.form.addRow("Units:", self.units)

    def action(self) -> str:
        col = self.win.add_response(self.name.text().strip(), self.units.text().strip())
        return f"Added response {col}."


class RenameBlocksDialog(_FormDialog):
    title = "Rename Blocks"

    def __init__(self, win, parent=None):
        super().__init__(win, parent)
        self.edits = []
        for b in self.doc.block_values():
            e = QLineEdit(self.doc.block_name(b))
            self.form.addRow(f"Block {b:g}:", e)
            self.edits.append((b, e))

    def action(self) -> str:
        n = 0
        for b, e in self.edits:
            if e.text().strip() and e.text().strip() != self.doc.block_name(b):
                self.win.rename_block(b, e.text().strip())
                n += 1
        return f"Renamed {n} block(s)."


class PreferencesDialog(QDialog):
    """Edit > Preferences: significance level, confidence level of intervals and the
    tolerance-interval settings, shared by analysis, optimisation and post analysis."""

    FIELDS = [("alpha", "Significance level (alpha):", 0.001, 0.5),
              ("ci_level", "Confidence level of intervals:", 0.5, 0.999),
              ("ti_proportion", "Tolerance interval: proportion of population:", 0.5, 0.9999),
              ("ti_confidence", "Tolerance interval: confidence:", 0.5, 0.999)]

    def __init__(self, win):
        super().__init__(win)
        from rsstatchem.core.doe import model as mdl

        self.win = win
        self.setWindowTitle("Preferences")
        self.setStyleSheet(style.STYLESHEET)
        prefs = getattr(mdl, "PREFERENCES", None)
        values = prefs.to_meta() if prefs is not None else {}
        form = QFormLayout()
        self.spins = {}
        for key, label, lo, hi in self.FIELDS:
            w = QDoubleSpinBox()
            w.setRange(lo, hi)
            w.setDecimals(4)
            w.setSingleStep(0.01)
            w.setValue(float(values.get(key, {"alpha": 0.05, "ci_level": 0.95,
                                               "ti_proportion": 0.99,
                                               "ti_confidence": 0.95}[key])))
            form.addRow(label, w)
            self.spins[key] = w
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.apply)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Used by the ANOVA, the confidence / prediction / tolerance "
                             "intervals and the optimisation."))
        lay.addLayout(form)
        lay.addWidget(buttons)

    def apply(self) -> None:
        self.win.set_preferences({k: w.value() for k, w in self.spins.items()})
        self.accept()


class ImportDataDialog(QDialog):
    """File > Import Data Set: choose which columns of a table are factors (numeric or
    categoric) and which are responses; builds a historical-data design."""

    ROLES = [("ignore", "Ignore"), ("factor", "Factor (numeric)"),
             ("categoric", "Factor (categoric)"), ("response", "Response")]

    def __init__(self, frame: pd.DataFrame, name: str = "Imported data", parent=None):
        super().__init__(parent)
        from PySide6.QtWidgets import QTableWidget, QTableWidgetItem

        self.setWindowTitle("Import Data Set")
        self.setStyleSheet(style.STYLESHEET)
        self.frame = frame
        self.document = None
        self.kind = QComboBox()
        self.kind.addItem("Process factors", "process")
        self.kind.addItem("Mixture components", "mixture")
        self.name = QLineEdit(name)
        self.table = QTableWidget(len(frame.columns), 3)
        self.table.setHorizontalHeaderLabels(["Column", "Role", "First values"])
        self.table.verticalHeader().setVisible(False)
        self.roles: dict[str, QComboBox] = {}
        cols = list(frame.columns)
        for i, c in enumerate(cols):
            it = QTableWidgetItem(str(c))
            it.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(i, 0, it)
            box = QComboBox()
            for key, text in self.ROLES:
                box.addItem(text, key)
            nums = pd.to_numeric(frame[c], errors="coerce")
            guess = "response" if i == len(cols) - 1 else (
                "categoric" if nums.isna().all() else "factor")
            box.setCurrentIndex(box.findData(guess))
            self.table.setCellWidget(i, 1, box)
            self.roles[str(c)] = box
            vals = ", ".join(str(v) for v in frame[c].head(4).tolist())
            v = QTableWidgetItem(vals)
            v.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(i, 2, v)
        self.table.resizeColumnsToContents()
        self.info = QLabel(f"{len(frame)} rows. Numeric factors are coded from their data "
                           "range; review the Evaluation before analysing.")
        self.info.setWordWrap(True)
        form = QFormLayout()
        form.addRow("Design name:", self.name)
        form.addRow("Factors are:", self.kind)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.apply)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(self.table, 1)
        lay.addWidget(self.info)
        lay.addWidget(buttons)
        self.resize(560, 420)

    def set_role(self, column: str, role: str) -> None:
        box = self.roles[column]
        box.setCurrentIndex(box.findData(role))

    def build(self):
        factors = [c for c in self.frame.columns
                   if self.roles[str(c)].currentData() in ("factor", "categoric")]
        responses = [c for c in self.frame.columns
                     if self.roles[str(c)].currentData() == "response"]
        if not factors:
            raise ValueError("Mark at least one column as a factor.")
        if not responses:
            raise ValueError("Mark at least one column as a response.")
        info = {c: {"type": "categoric"} for c in factors
                if self.roles[str(c)].currentData() == "categoric"}
        return DesignDocument.from_data(self.frame, factors, responses,
                                        self.kind.currentData(), info,
                                        name=self.name.text().strip() or "Imported data")

    def apply(self) -> None:
        try:
            self.document = self.build()
        except (ValueError, KeyError) as exc:
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
        self._load_preferences()
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
        self.a_prefs = A(self, "&Preferences...", self.edit_preferences,
                         tip="Significance level, confidence level and tolerance intervals")
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
        self.a_center = A(self, "Add &Center Points...", lambda: self._tool(CenterPointsDialog),
                          tip="Add centre points (pure error, curvature test)")
        self.a_axial = A(self, "Add &Axial Points (to CCD)...", lambda: self._tool(AxialDialog),
                         tip="Augment a factorial to a central composite design")
        self.a_reps = A(self, "Add &Replicates...", lambda: self._tool(ReplicatesDialog))
        self.a_optaug = A(self, "&Optimal Augment...", lambda: self._tool(OptimalAugmentDialog),
                          icon_name="augment",
                          tip="Add optimal runs for a bigger model in a new block")
        self.a_space = A(self, "&Modify Design Space...", lambda: self._tool(ModifySpaceDialog))
        self.a_constraints = A(self, "Edit &Constraints...", self.edit_constraints,
                               icon_name="constraints")
        self.a_insert = A(self, "&Insert Run", self.insert_run)
        self.a_delete = A(self, "&Delete Selected Runs", self.delete_selected_runs)
        self.a_add_resp = A(self, "Add &Response...", lambda: self._tool(AddResponseDialog))
        self.a_round = A(self, "R&ound Factor Values...", lambda: self._tool(RoundDialog))
        self.a_blocks = A(self, "Rename &Blocks...", lambda: self._tool(RenameBlocksDialog))
        self.a_import_data = A(self, "Import &Data Set (Historical)...", self.import_data_set,
                               icon_name="import",
                               tip="Build a design from existing data (CSV / Excel)")
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
                               self.a_center, self.a_axial, self.a_reps, self.a_optaug,
                               self.a_space, self.a_constraints, self.a_insert, self.a_delete,
                               self.a_add_resp, self.a_round, self.a_blocks,
                               self.a_numerical,
                               self.a_graphical, self.a_prediction, self.a_confirmation,
                               self.a_coefficients, self.a_export_graph, self.a_copy_graph]

    def _build_menus(self) -> None:
        mb = self.menuBar()
        m = mb.addMenu("&File")
        m.addActions([self.a_new, self.a_open, self.a_import_data])
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
        m.addSeparator()
        m.addAction(self.a_prefs)
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
        aug = m.addMenu(icons.icon("augment"), "&Augment Design")
        aug.addActions([self.a_foldover, self.a_center, self.a_axial, self.a_reps,
                        self.a_optaug])
        aug.addSeparator()
        aug.addActions([self.a_space, self.a_constraints])
        lay = m.addMenu("Edit &Layout")
        lay.addActions([self.a_insert, self.a_delete, self.a_add_resp, self.a_round,
                        self.a_blocks])
        m.aboutToShow.connect(self._update_tool_actions)
        m.addSeparator()
        m.addAction(self.a_import)
        self.analysis_menu = mb.addMenu("&Analysis")
        self.analysis_menu.aboutToShow.connect(self._fill_analysis_menu)
        m = mb.addMenu("&Optimization")
        m.addActions([self.a_numerical, self.a_graphical])
        for key, folder, text, module, cls, icon_name in EXTENSION_NODES:
            if folder == "optimization" and extension_class(module, cls) is not None:
                a = make_action(self, f"&{text}", lambda _=False, k=key: self.show_node(k),
                                icon_name=icon_name)
                m.addAction(a)
                self.design_actions.append(a)
                setattr(self, f"a_{key}", a)
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
        page.import_data.connect(self.import_data_set)
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
        if self._has_constraints_node():
            add(design, "constraints", "Constraints", "constraints")
        add(design, "notes", "Notes", "notes")
        analysis = add(None, "", "Analysis", "folder", True)
        analysis.setFlags(Qt.ItemFlag.ItemIsEnabled)
        for i, col in enumerate(doc.response_cols):
            add(analysis, f"analysis:{col}", self._response_label(i), "response")
        folders = {"analysis": analysis}
        opt = add(None, "", "Optimization", "folder", True)
        opt.setFlags(Qt.ItemFlag.ItemIsEnabled)
        add(opt, "numerical", "Numerical", "numerical")
        add(opt, "graphical", "Graphical", "graphical")
        post = add(None, "", "Post Analysis", "folder", True)
        post.setFlags(Qt.ItemFlag.ItemIsEnabled)
        add(post, "prediction", "Point Prediction", "prediction")
        add(post, "confirmation", "Confirmation", "confirmation")
        add(post, "coefficients", "Coefficients Table", "coefficients")
        folders.update(optimization=opt, post=post)
        for key, folder, text, module, cls, icon_name in EXTENSION_NODES:
            if extension_class(module, cls) is not None:
                add(folders[folder], key, text, icon_name)
        self._nav_responses = list(doc.response_cols)
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
        elif key == "constraints":
            page = info.ConstraintsPage(doc)
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
            ext = next((e for e in EXTENSION_NODES if e[0] == key), None)
            cls = extension_class(ext[3], ext[4]) if ext else None
            if cls is None:
                raise KeyError(key)
            page = cls(doc)
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
            names = page.strip.labels() if hasattr(page, "strip") else [
                "Transform", "Fit Summary", "Model", "ANOVA", "Diagnostics", "Model Graphs"]
            self.statusBar().showMessage("Work through the tabs from left to right: "
                                         + ", ".join(names) + ".  (F1: Screen Tips)")
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
                                   "confirmation", "coefficients", "analysis_summary",
                                   "poe") or key == f"analysis:{detail}"
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
        if kind == "meta" and self.doc is not None and (
                list(self.doc.response_cols) != getattr(self, "_nav_responses", None)
                or self._has_constraints_node() != ("constraints" in self.nav_items)):
            current = self.current_key
            for key in [k for k in self.pages if k.startswith("analysis:")
                        and k.split(":", 1)[1] not in self.doc.response_cols]:
                page = self.pages.pop(key)
                self.stack.removeWidget(page)
                page.deleteLater()
            self._build_nav()
            if current in self.nav_items:
                self.nav.blockSignals(True)
                self.nav.setCurrentItem(self.nav_items[current])
                self.nav.blockSignals(False)
        if kind in ("data", "meta"):
            self._update_nav_labels()
        self._update_title()

    def _has_constraints_node(self) -> bool:
        doc = self.doc
        return doc is not None and (bool(doc.constraints) or doc.meta.get("design") in (
            "dopt", "mixdopt", "combined", "ev", "userdef", "distance", "lhs"))

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

    # ------------------------------------------------------------------ preferences
    def _load_preferences(self) -> None:
        from rsstatchem.core.doe import model as mdl

        prefs = getattr(mdl, "PREFERENCES", None)
        if prefs is None:
            return
        stored = {}
        for key in ("alpha", "ci_level", "ti_proportion", "ti_confidence"):
            v = self.setting(f"pref_{key}")
            try:
                if v not in (None, ""):
                    stored[key] = float(v)
            except (TypeError, ValueError):
                continue
        prefs.update(stored)

    def set_preferences(self, values: dict) -> None:
        """Edit > Preferences: update the shared analysis preferences (saved in the
        settings) and refresh every open RSM window."""
        from rsstatchem.core.doe import model as mdl

        prefs = getattr(mdl, "PREFERENCES", None)
        if prefs is None:
            return
        prefs.update(values)
        for key, v in prefs.to_meta().items():
            self.set_setting(f"pref_{key}", v)
        for w in [w for w in self.ctx.windows if isinstance(w, DXMainWindow)] or [self]:
            if w.doc is not None:
                w.doc._cache.clear()
                w.doc.notify("meta")

    def edit_preferences(self) -> "PreferencesDialog":
        dlg = PreferencesDialog(self)
        dlg.open()
        return dlg

    # ------------------------------------------------------------------ design tools
    def _update_tool_actions(self) -> None:
        doc = self.doc
        ok = doc is not None
        self.a_foldover.setEnabled(ok and doc.two_level())
        process = ok and not doc.mixture and not doc.combined and not doc.split_plot
        self.a_axial.setEnabled(process)
        self.a_center.setEnabled(ok and not doc.split_plot)
        self.a_reps.setEnabled(ok and not doc.split_plot)
        self.a_optaug.setEnabled(ok and not doc.split_plot)
        self.a_blocks.setEnabled(ok)

    def _tool(self, cls):
        """Open a Design Tools dialog (non-modal sheet); returns it (tests drive it)."""
        if self.doc is None:
            return None
        dlg = cls(self)

        def done(result):
            if result and dlg.message:
                self.statusBar().showMessage(dlg.message, 6000)
        dlg.finished.connect(done)
        dlg.open()
        return dlg

    def _structural(self, text: str, fn):
        from rsstatchem.ui.dx.layout import structural

        return structural(self.undo, self.doc, text, fn)

    def augment_center(self, n: int, new_block: bool = False) -> int:
        added = self._structural("Add Center Points",
                                 lambda: self.doc.add_center_points(n, new_block))
        self.sort("std")
        return added

    def augment_axial(self, alpha: float | None = None, center_points: int = 2) -> int:
        added = self._structural("Add Axial Points",
                                 lambda: self.doc.add_axial_points(alpha, center_points))
        self.sort("std")
        return added

    def augment_replicates(self, new_block: bool = True) -> int:
        added = self._structural("Add Replicates",
                                 lambda: self.doc.add_replicates(None, new_block))
        self.sort("std")
        return added

    def augment_optimal(self, terms, n_add: int, criterion: str = "I",
                        exchange: str = "point", lof_points: int = 0,
                        new_block: bool = True) -> int:
        import random

        seed = random.randint(1, 99999)
        added = self._structural("Optimal Augment", lambda: self.doc.optimal_augment(
            terms, n_add, criterion, exchange, lof_points, new_block, seed))
        self.sort("std")
        return added

    def modify_design_space(self, changes: dict) -> None:
        def go():
            for j, (lo, hi) in changes.items():
                self.doc.modify_design_space(j, lo, hi)
        self._structural("Modify Design Space", go)

    def round_factors(self, decimals: int | None = None, sig: int | None = None) -> int:
        return self._structural("Round Factor Values",
                                lambda: self.doc.round_factors(decimals, sig))

    def add_response(self, name: str, units: str = "") -> str:
        return self._structural("Add Response", lambda: self.doc.add_response(name, units))

    def rename_block(self, block: float, name: str) -> None:
        self._structural("Rename Block", lambda: self.doc.rename_block(block, name))

    def insert_run(self) -> int:
        if self.doc is None:
            return 0
        n = self._structural("Insert Run", lambda: self.doc.insert_runs(n=1))
        self.show_node("design")
        return n

    def delete_selected_runs(self) -> int:
        if self.doc is None or "design" not in self.pages:
            return 0
        page = self.pages["design"]
        rows = sorted({page.model.rows[i.row()] for i in page.view.selectedIndexes()})
        if not rows:
            self.statusBar().showMessage("Select the runs to delete in the design layout.",
                                         5000)
            return 0
        return page.view.delete_runs(rows)

    def edit_constraints(self):
        if self.doc is None:
            return None
        page = self.show_node("constraints") if self._has_constraints_node() else None
        if page is None:
            from rsstatchem.ui.dx import info

            page = info.ConstraintsPage(self.doc)
            self.pages["constraints"] = page
            self.stack.addWidget(page)
            self.stack.setCurrentWidget(page)
            self.current_key = "constraints"
        return page.edit()

    def import_data_set(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import Data Set", self._last_dir(),
                                              "Tables (*.csv *.txt *.xlsx *.xls)")
        if path:
            self.import_data_from(path)

    def import_data_from(self, path: str) -> ImportDataDialog | None:
        try:
            if path.lower().endswith((".xlsx", ".xls")):
                frame = pd.read_excel(path)
            else:
                frame = pd.read_csv(path, sep=None, engine="python")
        except (OSError, ValueError, ImportError) as exc:
            self.error(f"Could not read {Path(path).name}:\n{exc}")
            return None
        dlg = ImportDataDialog(frame, Path(path).stem, self)

        def done(result):
            if result and dlg.document is not None:
                self.adopt(dlg.document)
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
            names = view.strip.labels() if hasattr(view, "strip") else [
                "Transform", "Fit Summary", "Model", "ANOVA", "Diagnostics", "Model Graphs"]
            for name in names:
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

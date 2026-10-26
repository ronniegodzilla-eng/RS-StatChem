"""Analysis of one response, with Design-Expert's tab strip across the top:
Transform, Fit Summary, Model, ANOVA, Diagnostics and Model Graphs.

Every tab works on the document's stored analysis choices for the response (transformation,
model terms); changing them on one tab updates the others. All numbers come from
:mod:`rsstatchem.core.doe` (model, report, plots).
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QComboBox,
                               QDoubleSpinBox, QFrame, QGroupBox, QHBoxLayout, QHeaderView,
                               QLabel, QListWidget, QListWidgetItem, QPushButton, QRadioButton,
                               QScrollArea, QSplitter, QStackedWidget, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from rsstatchem.core.doe import model, report
from rsstatchem.core.doe.model import Transform, term_key, term_name
from rsstatchem.core.output import Image, Table, Text
from rsstatchem.ui.dx import icons, style
from rsstatchem.ui.dx.layout import response_stats
from rsstatchem.ui.dx.widgets import (FactorsTool, GraphView, ReportView, TabStrip,
                                      annotation, hint, tool_button)

num, pval, fval, fixed = report.num, report.pval, report.fval, report.fixed

TABS = [("transform", "Transform"), ("fitsummary", "Fit Summary"), ("model", "Model"),
        ("anova", "ANOVA"), ("diagnostics", "Diagnostics"), ("modelgraphs", "Model Graphs")]


class AnalysisView(QWidget):
    """The analysis node of one response."""

    go_to_layout = Signal()

    def __init__(self, doc, col: str, parent=None):
        super().__init__(parent)
        self.doc, self.col = doc, col
        self.annotate = True
        self.show_points = True
        self.strip = TabStrip(TABS)
        sp = doc.space()
        if sp.split_plot:  # Design-Expert labels the REML analysis this way
            self.strip.buttons[3].setText("ANOVA (REML)")
        # split-plot two-level factorials select effects on subplot / whole-plot half-normal
        # plots (Stat-Ease's split-plot tutorial) instead of a Fit Summary
        self.effects_mode = bool(sp.split_plot and doc.meta.get("design") == "factorial"
                                 and not any(sp.categoric))
        if self.effects_mode:
            self.strip.buttons[1].setText("Effects")
            self.strip.buttons[1].setIcon(icons.icon("half_normal")
                                          if "half_normal" in icons.NAMES
                                          else icons.icon("fitsummary"))
        self.title = QLabel()
        self.title.setStyleSheet(f"color:{style.BLUE}; font-weight:600; padding: 0 8px;")
        self.strip.extra.addWidget(self.title)
        self.stack = QStackedWidget()
        self.transform_page = TransformPage(self)
        self.fit_page = EffectsPage(self) if self.effects_mode else FitSummaryPage(self)
        self.model_page = ModelPage(self)
        self.anova_page = AnovaPage(self)
        self.diag_page = DiagnosticsPage(self)
        self.graph_page = ModelGraphsPage(self)
        self.pages = [self.transform_page, self.fit_page, self.model_page, self.anova_page,
                      self.diag_page, self.graph_page]
        for p in self.pages:
            self.stack.addWidget(p)
        self.empty = QWidget()
        el = QVBoxLayout(self.empty)
        self.empty_text = hint("")
        go = QPushButton(icons.icon("design"), "Go to the Design Layout")
        go.clicked.connect(self.go_to_layout.emit)
        el.addWidget(self.empty_text)
        el.addWidget(go, 0, Qt.AlignmentFlag.AlignLeft)
        el.addStretch(1)
        self.outer = QStackedWidget()
        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(0)
        bl.addWidget(self.strip)
        bl.addWidget(self.stack, 1)
        self.outer.addWidget(body)
        self.outer.addWidget(self.empty)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.outer)
        self._dirty = set(range(len(self.pages)))
        self.strip.changed.connect(self._tab_changed)
        start = 1 if not doc.has_spec(col) else 3
        self.strip.set_current(start, emit=False)
        self.stack.setCurrentIndex(start)
        self.refresh()

    # ------------------------------------------------------------------ state
    @property
    def index(self) -> int:
        return self.doc.response_index(self.col)

    @property
    def response_title(self) -> str:
        return self.doc.response_title(self.index)

    def spec(self) -> model.AnalysisSpec:
        return self.doc.spec(self.col)

    def set_spec(self, spec: model.AnalysisSpec) -> None:
        self.doc.set_spec(spec)  # notifies -> invalidate()

    def analysis(self) -> model.Analysis:
        return self.doc.analyze(self.col)

    def needed(self) -> int:
        return max(3, len(self.doc.factor_cols) + 2)

    def ready(self) -> bool:
        return self.doc.response_count(self.col) >= self.needed()

    def invalidate(self) -> None:
        self._dirty = set(range(len(self.pages)))
        self.refresh()

    def set_annotate(self, on: bool) -> None:
        self.annotate = on
        for p in (self.fit_page, self.anova_page):
            if p.report is not None:
                p.report.set_annotate(on)

    def set_show_points(self, on: bool) -> None:
        self.show_points = on
        self.graph_page.render()

    def refresh(self) -> None:
        self.title.setText(self.response_title)
        n = self.doc.response_count(self.col)
        if not self.ready():
            self.outer.setCurrentIndex(1)
            self.empty_text.setText(
                f"<b>{self.response_title}</b> has {n} of {self.doc.n_runs} values. Enter at "
                f"least {self.needed()} response values in the design layout (the pale yellow "
                "cells) to start the analysis; all runs are needed for a complete analysis.")
            return
        self.outer.setCurrentIndex(0)
        i = self.stack.currentIndex()
        if i in self._dirty:
            self._dirty.discard(i)
            self.pages[i].refresh()

    def _tab_changed(self, i: int) -> None:
        self.stack.setCurrentIndex(i)
        self.refresh()

    def set_tab(self, name: str) -> None:
        texts = self.strip.labels()
        i = texts.index(name) if name in texts else [t[1] for t in TABS].index(name)
        self.strip.set_current(i)

    def current_graph(self) -> GraphView | None:
        page = self.pages[self.stack.currentIndex()]
        return getattr(page, "graph", None) if getattr(page, "graph_visible", True) else None

    def current_report(self) -> ReportView | None:
        page = self.pages[self.stack.currentIndex()]
        return getattr(page, "report", None)


def _error_items(exc: Exception) -> list:
    return [Text("The model cannot be fitted", str(exc), style="warning")]


# ---------------------------------------------------------------------------- Transform
TRANSFORM_CHOICES = [("none", "None"), ("sqrt", "Square Root"), ("ln", "Natural Log"),
                     ("log10", "Base 10 Log"), ("inv_sqrt", "Inverse Sqrt"),
                     ("inverse", "Inverse"), ("power", "Power")]


class TransformPage(QWidget):
    def __init__(self, view: AnalysisView):
        super().__init__()
        self.view = view
        self._loading = False
        self.group = QButtonGroup(self)
        box = QGroupBox("Transform")
        bl = QVBoxLayout(box)
        self.radios = {}
        for key, text in TRANSFORM_CHOICES:
            rb = QRadioButton(text)
            self.group.addButton(rb)
            self.radios[key] = rb
            bl.addWidget(rb)
            rb.toggled.connect(lambda on, k=key: on and self._changed())
        self.lam = QDoubleSpinBox()
        self.lam.setRange(-3, 3)
        self.lam.setSingleStep(0.25)
        self.lam.setValue(1.0)
        self.lam.valueChanged.connect(lambda _: self._changed())
        self.k = QDoubleSpinBox()
        self.k.setRange(-1e9, 1e9)
        self.k.setDecimals(4)
        self.k.valueChanged.connect(lambda _: self._changed())
        row = QHBoxLayout()
        row.addWidget(QLabel("Lambda:"))
        row.addWidget(self.lam)
        bl.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Constant k:"))
        row.addWidget(self.k)
        bl.addLayout(row)
        self.use_bc = QPushButton("Use Box-Cox Recommendation")
        self.use_bc.clicked.connect(self._use_boxcox)
        bl.addWidget(self.use_bc)
        bl.addStretch(1)
        box.setFixedWidth(260)
        self.report = ReportView()
        self.graph = GraphView()
        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self.report)
        right.addWidget(self.graph)
        right.setSizes([300, 360])
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(box)
        lay.addWidget(right, 1)
        self._bc = None

    def refresh(self) -> None:
        spec = self.view.spec()
        tr = spec.transform
        self._loading = True
        key = tr.kind if tr.kind in self.radios else "power"
        self.radios[key].setChecked(True)
        self.lam.setValue(tr.lam if tr.kind in ("power", "boxcox") else (tr.power or 1.0))
        self.lam.setEnabled(key == "power")
        self.k.setValue(tr.k)
        self._loading = False
        s = response_stats(self.view.doc, self.view.col)
        ratio = s["max"] / s["min"] if s["min"] and s["min"] > 0 else math.nan
        rows = [["Observations", str(s["n"])], ["Minimum", num(s["min"])],
                ["Maximum", num(s["max"])], ["Ratio max/min", num(ratio)],
                ["Current transform", tr.describe()]]
        items: list = [Table(f"Response {self.view.response_title}", ["", "Value"], rows)]
        advice = []
        if math.isfinite(ratio):
            advice.append(f"The ratio of maximum to minimum is {ratio:.3g}. "
                          + ("A ratio greater than 10 usually indicates that a transformation "
                             "is required." if ratio > 10 else
                             "A ratio below 3 means a power transformation has little "
                             "effect." if ratio < 3 else
                             "A transformation may help; check the Box-Cox plot."))
        self._bc = None
        self.graph_visible = True
        try:
            a = self.view.analysis()
            kk = tr.k
            if np.any(a.obs.y + kk <= 0):
                kk = float(-np.min(a.obs.y) + 0.01 * (np.ptp(a.obs.y) or 1))
            bc = model.box_cox(a.space, a.obs, a.fit.terms, kk,
                               current=tr.power if tr.power is not None else 1.0)
            self._bc = bc
            from rsstatchem.core.doe import plots

            self.graph.set_png(plots.box_cox_plot(bc, a.fit.response_name), "box_cox")
            items.append(Table("Box-Cox Plot for Power Transforms", ["", "Value"], [
                ["Model used", report.model_label(a.space, a.fit.terms)],
                ["Current Lambda", num(bc["current"])], ["Best Lambda", num(bc["best"])],
                ["Low C.I.", num(bc["low"]) if bc["low"] is not None else "< -3"],
                ["High C.I.", num(bc["high"]) if bc["high"] is not None else "> 3"],
                ["Recommended Transform", f"{bc['recommended_name']} (Lambda = "
                                          f"{bc['recommended']:g})"]]))
            advice.append("Box-Cox: " + (
                "the current lambda lies inside the 95 % confidence interval, so no "
                "transformation is needed." if bc["current_in_ci"] else
                f"the current lambda is outside the 95 % confidence interval; the "
                f"recommended transform is {bc['recommended_name']}."))
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.graph.set_message(f"Box-Cox plot not available: {exc}")
        items.append(annotation("\n".join(advice) or "Choose a transformation if the "
                                "residuals show a funnel shape or the Box-Cox plot "
                                "recommends one."))
        self.report.set_items(items, "Transform")

    def _changed(self) -> None:
        if self._loading:
            return
        key = next(k for k, rb in self.radios.items() if rb.isChecked())
        self.lam.setEnabled(key == "power")
        spec = self.view.spec()
        spec.transform = Transform(key, float(self.k.value()), float(self.lam.value()))
        err = spec.transform.check(self.view.doc.ds.df[self.view.col].dropna().to_numpy(float))
        if err:
            self.report.set_items(self.report.items + [Text("", err, style="warning")],
                                  "Transform")
            return
        self.view.set_spec(spec)

    def _use_boxcox(self) -> None:
        if not self._bc:
            return
        rec = self._bc["recommended"]
        spec = self.view.spec()
        key = {1: "none", 0.5: "sqrt", 0: "ln", -0.5: "inv_sqrt", -1: "inverse"}.get(rec, "power")
        spec.transform = Transform(key, float(self._bc["k"] if key != "none" else 0.0),
                                   float(rec))
        self.view.set_spec(spec)


# ---------------------------------------------------------------------------- Fit Summary
def fit_summary_items(a: model.Analysis) -> list:
    fs = a.summary
    seq = {}
    for src, ss, df, f, p, remark in fs.sequential:
        if " vs " in src and not src.startswith(("Mean", "Block")):
            seq[src.split(" vs ")[0]] = (p, remark)
    lof = {r[0]: r for r in fs.lof}
    rows = []
    for src, sd, r2, adj, pred, press, remark in fs.summary:
        sp, rem = seq.get(src, (None, ""))
        lp = lof.get(src, [None] * 6)[4]
        rows.append([src, pval(sp), pval(lp), fixed(adj), fixed(pred), remark or rem])
    items: list = [Table("Fit Summary", ["Source", "Sequential p-value", "Lack of Fit p-value",
                                         "Adjusted R²", "Predicted R²", ""], rows)]
    if a.space.split_plot:
        items.append(Text("", "Split-plot design: the Fit Summary is an ordinary least-squares "
                              "guide that ignores the groups (Stat-Ease documents no "
                              "split-plot Fit Summary: choose the model on the Model tab, e.g. "
                              "Auto Select by AICc, which uses the maximum likelihood); the "
                              "chosen model is analysed by REML on the ANOVA (REML) tab.",
                          style="warning"))
    items.append(annotation(
        f"The {fs.suggested} model is suggested: the highest-order polynomial whose added "
        "terms are significant (sequential p < 0.05), that is not aliased, with insignificant "
        "lack of fit and the best Adjusted and Predicted R². Use the Model tab to accept it "
        "or choose another."))
    srows = []
    for src, ss, df, f, p, remark in fs.sequential:
        ms = ss / df if df else None
        srows.append([src, num(ss), str(int(df)), num(ms), fval(f), pval(p), remark])
    items.append(Table("Sequential Model Sum of Squares [Type I]",
                       ["Source", "Sum of Squares", "df", "Mean Square", "F-value", "p-value",
                        ""], srows,
                       footnotes=["Select the highest order polynomial where the additional "
                                  "terms are significant and the model is not aliased."]))
    if fs.pure_error[1] > 0:
        lrows = []
        for src, ss, df, f, p, remark in fs.lof:
            lrows.append([src, num(ss), str(int(df)), num(ss / df) if df else "", fval(f),
                          pval(p), remark])
        ss_pe, df_pe = fs.pure_error
        lrows.append(["Pure Error", num(ss_pe), str(df_pe), num(ss_pe / df_pe), "", "", ""])
        items.append(Table("Lack of Fit Tests", ["Source", "Sum of Squares", "df",
                                                 "Mean Square", "F-value", "p-value", ""],
                           lrows, footnotes=["Select the model with insignificant "
                                             "lack-of-fit."]))
    else:
        items.append(Text("Lack of Fit Tests", "No replicated design points: pure error and "
                          "the lack-of-fit test are not available."))
    mrows = []
    for src, sd, r2, adj, pred, press, remark in fs.summary:
        mrows.append([src, num(sd), fixed(r2), fixed(adj), fixed(pred), num(press), remark])
    items.append(Table("Model Summary Statistics", ["Source", "Std. Dev.", "R²",
                                                    "Adjusted R²", "Predicted R²", "PRESS",
                                                    ""], mrows,
                       footnotes=["Focus on the model maximizing the Adjusted R² and the "
                                  "Predicted R²."]))
    return items


class FitSummaryPage(QWidget):
    def __init__(self, view: AnalysisView):
        super().__init__()
        self.view = view
        self.report = ReportView()
        self.use = QPushButton("Use Suggested Model")
        self.use.clicked.connect(self._use)
        self.label = QLabel()
        bar = QHBoxLayout()
        bar.addWidget(self.label, 1)
        bar.addWidget(self.use)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addLayout(bar)
        lay.addWidget(self.report, 1)
        self.suggested = "mean"

    def refresh(self) -> None:
        try:
            a = self.view.analysis()
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.report.set_items(_error_items(exc), "Fit Summary")
            return
        self.suggested = a.summary.suggested_order
        label = a.fit.transform.label(a.fit.response_name)
        self.label.setText(f"<b>Response {self.view.index + 1}: {label}</b> — suggested "
                           f"model: <b>{a.summary.suggested}</b>")
        self.report.annotate = self.view.annotate
        self.report.set_items(fit_summary_items(a), "Fit Summary")

    def _use(self) -> None:
        spec = self.view.spec()
        spec.model, spec.terms, spec.reduction = "suggested", None, "none"
        self.view.set_spec(spec)
        self.view.set_tab("ANOVA")


# ---------------------------------------------------------------------------- Effects
EFFECT_SETS = (("sub", "Subplot Effects"), ("whole", "Whole-plot Effects"))


class EffectsPage(QWidget):
    """Effects of a split-plot two-level factorial on two half-normal plots — subplot effects
    and whole-plot effects (contrasts constant within every group), each with its own error
    line, as in Stat-Ease's split-plot tutorial. Click an effect on the plot or in the list
    to select it for the model (red) or return it to error (blue); the selected model is
    tested on the ANOVA (REML) tab."""

    def __init__(self, view: AnalysisView):
        super().__init__()
        self.view = view
        self.report = None
        self.kind = "sub"
        self.effects: dict = {"sub": [], "whole": []}
        self.terms: list = []
        self.rows: list = []
        self.label = QLabel()
        self.label.setWordWrap(True)
        self.buttons = {}
        group = QButtonGroup(self)
        bar = QHBoxLayout()
        bar.addWidget(self.label, 1)
        for key, text in EFFECT_SETS:
            rb = QRadioButton(text)
            group.addButton(rb)
            self.buttons[key] = rb
            bar.addWidget(rb)
        self.buttons["sub"].setChecked(True)
        for key, rb in self.buttons.items():
            rb.toggled.connect(lambda on, k=key: on and self.set_kind(k))
        clear = QPushButton("Clear Selection")
        clear.setToolTip("Return every effect to error (intercept-only model)")
        clear.clicked.connect(self.clear)
        bar.addWidget(clear)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["", "Term", "Effect", "% Contribution",
                                              "t-Value (Lenth)"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setColumnWidth(0, 36)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table.cellClicked.connect(self._clicked)
        self.graph = GraphView()
        split = QSplitter()
        split.addWidget(self.table)
        split.addWidget(self.graph)
        split.setSizes([380, 620])
        self.note = hint("Click an effect — on the half-normal plot or M / e in the list — to "
                         "put it in the model (red, M) or leave it for error (blue, e). "
                         "Whole-plot effects are estimated against the between-group "
                         "variation and subplot effects against the within-group variation, "
                         "so each set is judged on its own plot; check the selected model on "
                         "the ANOVA (REML) tab.")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addLayout(bar)
        lay.addWidget(split, 1)
        lay.addWidget(self.note)

    def set_kind(self, kind: str) -> None:
        self.kind = kind
        if not self.buttons[kind].isChecked():
            self.buttons[kind].setChecked(True)
        self.render()

    def refresh(self) -> None:
        try:
            a = self.view.analysis()
            self.effects = model.split_plot_effects(a.space, a.obs,
                                                    transform=a.fit.transform)
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.effects = {"sub": [], "whole": []}
            self.table.setRowCount(0)
            self.graph.set_message(f"Effects not available: {exc}")
            return
        self.terms = list(a.fit.terms)
        label = a.fit.transform.label(a.fit.response_name)
        nw, ns = len(self.effects["whole"]), len(self.effects["sub"])
        self.label.setText(f"<b>Response {self.view.index + 1}: {label}</b> — {ns} subplot "
                           f"and {nw} whole-plot effects; model: "
                           f"{report.model_label(a.space, self.terms)}")
        self.render()

    def render(self) -> None:
        eff = self.effects.get(self.kind, [])
        tset = set(self.terms)
        selected = {e["name"] for e in eff if e["term"] in tset}
        title = dict(EFFECT_SETS)[self.kind]
        from rsstatchem.core.doe import plots

        name = self.view.response_title
        fig = plots.half_normal_figure(eff, f"Half-Normal Plot, {title}: {name}", selected)
        self.graph.set_figure(fig, f"half_normal_{self.kind}")
        if self.graph.canvas is not None:
            self.graph.canvas.mpl_connect("button_press_event", self._pick)
        self.rows = sorted(eff, key=lambda e: -abs(e["effect"]))
        self.table.setRowCount(len(self.rows))
        for r, e in enumerate(self.rows):
            sel = e["term"] in tset
            it = QTableWidgetItem("M" if sel else "e")
            it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            f = it.font()
            f.setBold(True)
            it.setFont(f)
            if sel:
                it.setBackground(QBrush(QColor("#FFCDD2")))
                it.setForeground(QBrush(QColor("#B71C1C")))
            else:
                it.setForeground(QBrush(QColor("#1f3fbf")))
            self.table.setItem(r, 0, it)
            self.table.setItem(r, 1, QTableWidgetItem(e["name"]))
            self.table.setItem(r, 2, QTableWidgetItem(num(e["effect"])))
            self.table.setItem(r, 3, QTableWidgetItem(f"{e['contribution']:.2f} %"))
            self.table.setItem(r, 4, QTableWidgetItem(fval(e.get("t"))))

    def _pick(self, event) -> None:
        fig = self.graph.figure
        if event.button != 1 or event.inaxes is None or fig is None:
            return
        pts = getattr(fig, "effect_points", [])
        if not pts:
            return
        xy = event.inaxes.transData.transform(np.array([[x, y] for x, y, _ in pts]))
        d = np.hypot(xy[:, 0] - event.x, xy[:, 1] - event.y)
        i = int(np.argmin(d))
        if d[i] <= 12:
            self.toggle_name(pts[i][2])

    def _clicked(self, row: int, col: int) -> None:
        if col in (0, 1) and 0 <= row < len(self.rows):
            self.toggle(self.rows[row]["term"])

    def toggle_name(self, name: str) -> None:
        e = next((e for e in self.effects.get(self.kind, []) if e["name"] == name), None)
        if e is not None:
            self.toggle(e["term"])

    def clear(self) -> None:
        spec = self.view.spec()
        spec.model, spec.reduction, spec.terms = "manual", "none", []
        self.view.set_spec(spec)

    def toggle(self, term) -> None:
        """Add an effect to the model (with its lower-order parents when the model is kept
        hierarchical) or return it to error."""
        terms = list(self.terms)
        if term in terms:
            terms.remove(term)
        else:
            terms.append(term)
            if self.view.spec().hierarchy:
                for r in range(1, len(term)):
                    for sub in itertools.combinations(term, r):
                        if sub not in terms:
                            terms.append(sub)
        spec = self.view.spec()
        spec.model, spec.reduction = "manual", "none"
        spec.terms = [term_key(t) for t in model.sort_terms(terms)]
        self.view.set_spec(spec)


# ---------------------------------------------------------------------------- Model
class ModelPage(QWidget):
    def __init__(self, view: AnalysisView):
        super().__init__()
        self.view = view
        self._loading = False
        self.order = QComboBox()
        self.order.currentIndexChanged.connect(self._order_changed)
        self.alpha = QDoubleSpinBox()          # alpha out
        self.alpha_in = QDoubleSpinBox()
        for w in (self.alpha, self.alpha_in):
            w.setRange(0.001, 0.5)
            w.setDecimals(3)
            w.setSingleStep(0.01)
            w.setValue(0.10)
        self.criterion = QComboBox()
        for key, text in model.SELECTION_CRITERIA:
            self.criterion.addItem(text, key)
        self.method = QComboBox()
        for key, text in model.SELECTION_METHODS + [("manual", "Manual")]:
            self.method.addItem(text, key)
        self.criterion.currentIndexChanged.connect(self._selection_changed)
        self.method.currentIndexChanged.connect(self._selection_changed)
        auto = QPushButton(icons.icon("numerical"), "Auto Select")
        auto.setToolTip("Choose the model terms automatically from the current model")
        auto.clicked.connect(self.auto_select)
        self.hierarchy = QCheckBox("Keep model hierarchical")
        self.hierarchy.setChecked(True)
        top = QHBoxLayout()
        top.addWidget(QLabel("Process Order:"))
        top.addWidget(self.order)
        top.addSpacing(16)
        top.addWidget(self.hierarchy)
        top.addStretch(1)
        sel = QHBoxLayout()
        top2 = sel
        top2.addWidget(QLabel("Auto Select — Criterion:"))
        top2.addWidget(self.criterion)
        top2.addWidget(QLabel("Selection:"))
        top2.addWidget(self.method)
        self.alpha_in_label = QLabel("Alpha in:")
        top2.addWidget(self.alpha_in_label)
        top2.addWidget(self.alpha_in)
        self.alpha_out_label = QLabel("Alpha out:")
        top2.addWidget(self.alpha_out_label)
        top2.addWidget(self.alpha)
        top2.addWidget(auto)
        top2.addStretch(1)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["", "Term", "Type", "p-value (current fit)"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(0, 40)
        self.table.setColumnWidth(1, 110)
        self.table.setColumnWidth(2, 110)
        self.table.cellClicked.connect(self._clicked)
        self.note = hint("Click M / e to move a term in or out of the model: M = in the model, "
                         "e = left for error. Aliased terms cannot be estimated with this "
                         "design. Auto Select starts from the terms marked M: backward removes, "
                         "forward adds and stepwise adds and removes terms by p-value (alpha "
                         "in / out) or by AICc / BIC.")
        self.status = QLabel()
        self.status.setWordWrap(True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addLayout(top)
        lay.addLayout(sel)
        lay.addWidget(self.note)
        lay.addWidget(self.table, 1)
        lay.addWidget(self.status)
        self.terms: list = []
        self.aliased: set = set()
        self.candidates: list = []
        self._selection_changed()

    def _selection_changed(self, _=None) -> None:
        crit, method = self.criterion.currentData(), self.method.currentData()
        if crit != "p" and method == "stepwise":  # Design-Expert: AICc / BIC forward/backward
            self.method.blockSignals(True)
            self.method.setCurrentIndex(self.method.findData("forward"))
            self.method.blockSignals(False)
            method = "forward"
        p = crit == "p"
        self.alpha_in.setEnabled(p and method in ("forward", "stepwise"))
        self.alpha.setEnabled(p and method in ("backward", "stepwise"))
        self.alpha_in_label.setEnabled(self.alpha_in.isEnabled())
        self.alpha_out_label.setEnabled(self.alpha.isEnabled())

    def set_selection(self, method: str, criterion: str = "p", alpha_in: float = 0.1,
                      alpha_out: float = 0.1) -> None:
        self.criterion.setCurrentIndex(self.criterion.findData(criterion))
        self.method.setCurrentIndex(self.method.findData(method))
        self.alpha_in.setValue(alpha_in)
        self.alpha.setValue(alpha_out)
        self._selection_changed()

    def refresh(self) -> None:
        v = self.view
        sp = v.doc.space()
        sel = v.spec().selection or {}
        if sel:
            self.set_selection(sel.get("method", "backward"), sel.get("criterion", "p"),
                               float(sel.get("alpha_in", 0.1)),
                               float(sel.get("alpha_out", 0.1)))
        try:
            a = v.analysis()
        except (ValueError, np.linalg.LinAlgError) as exc:
            a = None
            self.status.setText(f"<span style='color:#B00020'>{exc}</span>")
        self._loading = True
        self.order.clear()
        for name, key in model.model_orders(sp):
            self.order.addItem(name, key)
        self.order.addItem("Manual / Reduced", "manual")
        self._loading = False
        self.candidates = model.candidate_terms(sp)
        if a is not None:
            self.terms = list(a.fit.terms)
            obs = a.obs
            _, dropped = model.drop_aliased(sp, obs, sorted(self.candidates,
                                                            key=model.term_degree))
            self.aliased = set(dropped)
            pv = a.fit.term_p_values()
        else:
            pv = {}
        label = report.model_label(sp, self.terms)
        self._loading = True
        match = next((i for i in range(self.order.count())
                      if self.order.itemText(i) == label), self.order.count() - 1)
        self.order.setCurrentIndex(match)
        self._loading = False
        tset = set(self.terms)
        self.table.setRowCount(len(self.candidates))
        for r, t in enumerate(self.candidates):
            aliased = t in self.aliased and t not in tset
            mark = "M" if t in tset else ("" if aliased else "e")
            it = QTableWidgetItem(mark)
            it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            f = it.font()
            f.setBold(True)
            it.setFont(f)
            if mark == "M":
                it.setBackground(QBrush(QColor("#CFE3FA")))
                it.setForeground(QBrush(QColor(style.BLUE_DARK)))
            elif mark == "e":
                it.setForeground(QBrush(QColor("#9A6B00")))
            self.table.setItem(r, 0, it)
            name = (sp.factors[t[0]].source if len(t) == 1 else term_name(t))
            self.table.setItem(r, 1, QTableWidgetItem(name))
            typ = QTableWidgetItem("Aliased" if aliased else
                                   {1: "Main effect", 2: "Interaction / quadratic",
                                    3: "Third order"}.get(model.term_degree(t), "Higher order"))
            if aliased:
                typ.setForeground(QBrush(QColor("#8A8F96")))
            self.table.setItem(r, 2, typ)
            p = pv.get(t, (None, None))[1] if t in tset else None
            pit = QTableWidgetItem(pval(p) if p is not None else "")
            if p is not None and math.isfinite(p):
                pit.setForeground(QBrush(QColor(style.p_color(p))))
            self.table.setItem(r, 3, pit)
        if a is not None:
            msg = [f"Current model: <b>{label}</b> ({len(self.terms)} terms, "
                   f"{a.fit.df_res} residual df)."]
            if not self._hierarchical():
                msg.append("<span style='color:#9A6B00'>The model is not hierarchical: a "
                           "term is in the model without all of its lower-order parents. "
                           "Design-Expert adds them back when 'Keep model hierarchical' is "
                           "on.</span>")
            self.status.setText(" ".join(msg))

    def _hierarchical(self) -> bool:
        sp = self.view.doc.space()
        if sp.mixture:
            return True
        tset = set(self.terms)
        for t in self.terms:
            for u in self.candidates:
                if model.contains(u, t) and u not in tset:
                    return False
        return True

    def _apply(self, terms: list, selection: dict | None = None) -> None:
        spec = self.view.spec()
        spec.model, spec.terms, spec.reduction = "manual", [term_key(t) for t in terms], "none"
        if selection is not None:
            spec.selection = selection
        elif spec.selection:
            spec.selection = {k: v for k, v in spec.selection.items() if k != "steps"}
        self.view.set_spec(spec)

    def _clicked(self, row: int, col: int) -> None:
        if col not in (0, 1) or row >= len(self.candidates):
            return
        t = self.candidates[row]
        tset = list(self.terms)
        if t in tset:
            if self.view.doc.space().mixture and len(t) == 1:
                self.status.setText("Linear blending terms of a mixture always stay in the "
                                    "model.")
                return
            tset.remove(t)
        elif t in self.aliased:
            self.status.setText(f"{term_name(t)} is aliased in this design and cannot be "
                                "added.")
            return
        else:
            tset.append(t)
            if self.hierarchy.isChecked() and not self.view.doc.space().mixture:
                for u in self.candidates:
                    if model.contains(u, t) and u not in tset and u not in self.aliased:
                        tset.append(u)
        self._apply(tset)

    def _order_changed(self, _=None) -> None:
        if self._loading:
            return
        key = self.order.currentData()
        if key in (None, "manual"):
            return
        self._apply(model.model_terms(self.view.doc.space(), key))

    def auto_select(self) -> list:
        """Design-Expert's Auto Select on the current model's terms. Returns the steps."""
        method, crit = self.method.currentData(), self.criterion.currentData()
        if method == "manual":
            self.status.setText("Manual selection: click M / e in the term list.")
            return []
        v = self.view
        try:
            a = v.analysis()
            fit, steps = model.auto_select(a.space, a.obs, list(a.fit.terms), a.fit.transform,
                                           method, crit, self.alpha_in.value(),
                                           self.alpha.value(), self.hierarchy.isChecked(),
                                           v.col)
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.status.setText(f"<span style='color:#B00020'>{exc}</span>")
            return []
        selection = {"method": method, "criterion": crit, "alpha_in": self.alpha_in.value(),
                     "alpha_out": self.alpha.value(), "hierarchy": self.hierarchy.isChecked(),
                     "candidates": [term_key(t) for t in a.fit.terms],
                     "steps": [[act, term_key(t), float(val)] for act, t, val in steps]}
        self._apply(list(fit.terms), selection)
        self.status.setText(selection_text(selection))
        return steps


def selection_text(sel: dict) -> str:
    p_crit = sel.get("criterion", "p") == "p"
    crit = dict(model.SELECTION_CRITERIA).get(sel.get("criterion", "p"), "p-value")
    method = dict(model.SELECTION_METHODS).get(sel.get("method"), "")
    parts = []
    for act, key, val in sel.get("steps", []):
        t = model.key_term(key)
        v = pval(val) if p_crit else num(val)
        if act == "hierarchy":
            parts.append(f"added {term_name(t)} for hierarchy")
            continue
        parts.append(f"{'added' if act == 'add' else 'removed'} {term_name(t)} "
                     f"({'p' if p_crit else crit} = {v})")
    alphas = ""
    if p_crit:
        if sel.get("method") != "backward":
            alphas += f", alpha in = {sel.get('alpha_in', 0.1):g}"
        if sel.get("method") != "forward":
            alphas += f", alpha out = {sel.get('alpha_out', 0.1):g}"
    return f"Auto Select ({method}, {crit}{alphas}): " + ("; ".join(parts) or "no change") + "."


def selection_items(spec: model.AnalysisSpec) -> list:
    sel = spec.selection or {}
    if not sel.get("steps"):
        return []
    s2 = model.AnalysisSpec(spec.response, reduction=sel.get("method", "backward"),
                            criterion=sel.get("criterion", "p"),
                            alpha_in=float(sel.get("alpha_in", 0.1)),
                            alpha_out=float(sel.get("alpha_out", 0.1)),
                            hierarchy=bool(sel.get("hierarchy", True)))
    steps = [(act, model.key_term(key), val) for act, key, val in sel["steps"]]
    return [report.selection_table(s2, steps)]


# ---------------------------------------------------------------------------- ANOVA
def anova_items(a: model.Analysis) -> list:
    fit = a.fit
    name = report.model_label(a.space, fit.terms)
    items: list = []
    label = fit.transform.label(fit.response_name)
    items.append(Text("", f"Response: {label}    ·    Transform: "
                          f"{fit.transform.describe()}    ·    Model: {name}"))
    if a.dropped:
        items.append(Text("Aliased Terms", "Removed because they are aliased in this design: "
                          + ", ".join(term_name(t) for t in a.dropped) + ".", style="warning"))
    items.extend(selection_items(a.spec))
    t, interp = report.anova_table(fit, name)
    items.append(t)
    if interp:
        items.append(annotation("\n".join(interp)))
    st, interp2 = report.fit_statistics_table(fit)
    items.append(st)
    if interp2:
        items.append(annotation("\n".join(interp2)))
    if report.is_reml(fit):
        items.append(report.variance_components_table(fit))
        items.append(annotation(
            f"The groups (whole plots) contribute a variance of {num(fit.s2_wp)} against a "
            f"residual variance of {num(fit.s2)} (ratio {num(fit.eta)}). A large ratio means "
            "the restricted randomisation matters: whole-plot effects are estimated less "
            "precisely than subplot effects."))
    items.append(report.coefficient_table(fit))
    items.append(annotation("VIFs near 1 mean the coefficients are estimated independently "
                            "(orthogonal factors); VIFs above 10 indicate multicollinearity."))
    items.extend(report.equation_tables(fit))
    if label != fit.response_name:
        items.append(Text("", f"Analysis on the transformed scale {label}; predictions are "
                              "back-transformed to original units."))
    return items


class AnovaPage(QWidget):
    def __init__(self, view: AnalysisView):
        super().__init__()
        self.view = view
        self.report = ReportView()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(self.report)

    def refresh(self) -> None:
        try:
            a = self.view.analysis()
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.report.set_items(_error_items(exc), "ANOVA")
            return
        if not self.view.doc.has_spec(self.view.col):  # remember the model shown
            self.view.doc.meta.setdefault("analysis", {})[self.view.col] = a.spec.to_meta()
        self.report.annotate = self.view.annotate
        self.report.set_items(anova_items(a),
                              f"ANOVA for {report.model_label(a.space, a.fit.terms)} model")


# ---------------------------------------------------------------------------- Diagnostics
DIAGNOSTICS = [("normal", "Normal Plot"), ("resid_pred", "Residuals vs. Predicted"),
               ("resid_run", "Residuals vs. Run"), ("pred_actual", "Predicted vs. Actual"),
               ("boxcox", "Box-Cox"), ("resid_factor", "Residuals vs. Factor"),
               ("cooks", "Cook's Distance"), ("leverage", "Leverage vs. Run"),
               ("dffits", "DFFITS vs. Run"), ("dfbetas", "DFBETAS vs. Run"),
               ("report", "Report")]


class DiagnosticsPage(QWidget):
    def __init__(self, view: AnalysisView):
        super().__init__()
        self.view = view
        panel = QFrame()
        panel.setObjectName("toolPanel")
        panel.setFixedWidth(210)
        pl = QVBoxLayout(panel)
        t = QLabel("Diagnostics Tool")
        t.setStyleSheet(f"color:{style.BLUE}; font-weight:600;")
        pl.addWidget(t)
        self.list = QListWidget()
        for key, text in DIAGNOSTICS:
            it = QListWidgetItem(text)
            it.setData(Qt.ItemDataRole.UserRole, key)
            self.list.addItem(it)
        self.list.currentRowChanged.connect(lambda _: self.render())
        pl.addWidget(self.list, 1)
        pl.addWidget(QLabel("Residuals vs. factor:"))
        self.factor = QComboBox()
        for j, f in enumerate(view.doc.factors):
            self.factor.addItem(view.doc.factor_title(j), j)
        self.factor.currentIndexChanged.connect(lambda _: self.render())
        pl.addWidget(self.factor)
        pl.addWidget(QLabel("DFBETAS for term:"))
        self.term = QComboBox()
        self.term.currentIndexChanged.connect(lambda _: self.render())
        pl.addWidget(self.term)
        lab = QLabel("<small>Residuals: externally studentized.<br>Red lines: limits "
                     "(Bonferroni outlier limits, Cook's distance F(0.5), 2p/n leverage, "
                     "2√(p/n) DFFITS, 3/√n DFBETAS).</small>")
        lab.setWordWrap(True)
        pl.addWidget(lab)
        self.graph = GraphView()
        self.report = ReportView()
        self.right = QStackedWidget()
        self.right.addWidget(self.graph)
        self.right.addWidget(self.report)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(panel)
        lay.addWidget(self.right, 1)
        self.list.setCurrentRow(0)

    @property
    def graph_visible(self) -> bool:
        return self.right.currentIndex() == 0

    def refresh(self) -> None:
        self.render()

    def key(self) -> str:
        it = self.list.currentItem()
        return it.data(Qt.ItemDataRole.UserRole) if it else "normal"

    def render(self) -> None:
        if not self.view.ready():
            return
        try:
            a = self.view.analysis()
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.right.setCurrentIndex(0)
            self.graph.set_message(str(exc))
            return
        key = self.key()
        labels = list(a.fit.mat.labels)
        if [self.term.itemText(i) for i in range(self.term.count())] != labels:
            cur = self.term.currentText()
            self.term.blockSignals(True)
            self.term.clear()
            self.term.addItems(labels)
            i = labels.index(cur) if cur in labels else min(1, len(labels) - 1)
            self.term.setCurrentIndex(i)
            self.term.blockSignals(False)
        self.term.setEnabled(key == "dfbetas")
        self.factor.setEnabled(key == "resid_factor")
        if key == "report":
            sec = report.diagnostics_section(a, plots_wanted=(), table=True)
            self.right.setCurrentIndex(1)
            self.report.set_items([i for i in sec.items if isinstance(i, (Table, Text))],
                                  "Diagnostics Report")
            return
        if a.fit.df_res < 1:
            self.right.setCurrentIndex(0)
            self.graph.set_message("No residual degrees of freedom: diagnostics need a model "
                                   "with fewer terms than runs.")
            return
        try:
            sec = report.diagnostics_section(a, plots_wanted=(key,), table=False,
                                             factor_index=self.factor.currentData() or 0,
                                             term_index=max(self.term.currentIndex(), 0))
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError) as exc:
            self.right.setCurrentIndex(0)
            self.graph.set_message(str(exc))
            return
        img = next((i for i in sec.items if isinstance(i, Image)), None)
        self.right.setCurrentIndex(0)
        if img is None:
            note = next((i.text for i in sec.items if isinstance(i, Text) and i.style != "log"),
                        "Not available.")
            self.graph.set_message(note)
        else:
            self.graph.set_png(img.png, f"{self.view.col}_{key}")


# ---------------------------------------------------------------------------- Model Graphs
PROCESS_GRAPHS = [("perturbation", "perturbation", "Perturbation"),
                  ("one_factor", "onefactor", "One Factor"),
                  ("interaction", "interaction", "Interaction"),
                  ("contour", "contour", "Contour"), ("surface", "surface", "3D Surface"),
                  ("cube", "cube", "Cube")]
GRAPH_AXES = {"contour": 2, "surface": 2, "interaction": 2, "one_factor": 1, "cube": 3,
              "ternary": 3, "perturbation": 0, "trace": 0}
MIXTURE_GRAPHS = [("ternary", "ternary", "Contour (Ternary)"),
                  ("surface3", "surface", "3D Surface"), ("trace", "trace", "Trace (Cox)"),
                  ("one_factor", "onefactor", "One Factor")]
GRAPH_AXES["surface3"] = 3


class ModelGraphsPage(QWidget):
    def __init__(self, view: AnalysisView):
        super().__init__()
        self.view = view
        sp = view.doc.space()
        self.space = sp
        panel = QFrame()
        panel.setObjectName("toolPanel")
        panel.setFixedWidth(250)
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(4, 4, 4, 4)
        t = QLabel("Graphs")
        t.setStyleSheet(f"color:{style.BLUE}; font-weight:600;")
        pl.addWidget(t)
        self.buttons = {}
        graphs = MIXTURE_GRAPHS if sp.mixture else PROCESS_GRAPHS
        numeric = sum(1 for f in sp.factors if not f.categoric)
        self.group = QButtonGroup(self)
        for key, icon_name, text in graphs:
            b = tool_button(icon_name, text)
            b.setAutoExclusive(True)
            b.setMinimumWidth(220)
            ok = True
            if key in ("contour", "surface", "interaction") and (numeric < 2 if key != "interaction"
                                                                 else sp.k < 2):
                ok = False
            if key == "cube" and numeric < 3:
                ok = False
            if key in ("ternary", "surface3") and sp.k < 3:
                ok = False
            b.setEnabled(ok)
            b.clicked.connect(lambda _=False, k=key: self.set_graph(k))
            self.group.addButton(b)
            pl.addWidget(b)
            self.buttons[key] = b
        self.points = QCheckBox("Show design points")
        self.points.setChecked(True)
        self.points.toggled.connect(lambda _: self.render())
        pl.addWidget(self.points)
        self.tool = FactorsTool(sp, n_axes=min(3, sp.k))
        self.tool.changed.connect(self.render)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(self.tool)
        pl.addWidget(scroll, 1)
        self.axes_hint = QLabel()
        self.axes_hint.setWordWrap(True)
        self.axes_hint.setStyleSheet(f"color:{style.MUTED}; font-size:9pt;")
        pl.addWidget(self.axes_hint)
        self.graph = GraphView()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(panel)
        lay.addWidget(self.graph, 1)
        default = "ternary" if sp.mixture and sp.k >= 3 else (
            "contour" if numeric >= 2 else "one_factor")
        if sp.mixture and sp.k < 3:
            default = "one_factor"
        self.graph_key = default
        self.buttons[default].setChecked(True)
        self.tool.set_active_axes(GRAPH_AXES[default])

    def set_graph(self, key: str) -> None:
        self.graph_key = key
        self.buttons[key].setChecked(True)
        self.tool.set_active_axes(GRAPH_AXES[key])
        self.render()

    def refresh(self) -> None:
        self.render()

    def render(self) -> None:
        v = self.view
        if not v.ready():
            return
        try:
            a = v.analysis()
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.graph.set_message(str(exc))
            return
        key = self.graph_key
        axes = self.tool.axes()
        need = GRAPH_AXES[key]
        self.axes_hint.setText({0: "All factors are shown; the sliders set the reference point.",
                                1: "X1 axis: the factor on the horizontal axis.",
                                2: "X1 / X2 axis: the factors on the axes"
                                   + (" (X2 = the lines)." if key == "interaction" else "."),
                                3: "X1, X2, X3: the three factors shown."}[need])
        if len(axes) < need:
            self.graph.set_message(f"Assign {need} factors to axes (X1"
                                   + (", X2" if need > 1 else "")
                                   + (", X3" if need > 2 else "") + ") in the Factors Tool.")
            return
        ref = self.tool.point()
        use = axes[:max(need, 1)] if need else axes or [0]
        show = self.points.isChecked() and v.show_points
        try:
            if key in report.FIGURE_GRAPHS:  # live canvas: rotate 3D views, flag contours
                gf = report.graph_figure(a, key, use, ref, points=show)
                self.graph.set_figure(gf.figure, f"{v.col}_{key}", gf.rotatable, gf.probe)
                return
            sec = report.graph_section(a, key, use, ref, points=show)
        except (ValueError, np.linalg.LinAlgError, IndexError) as exc:
            self.graph.set_message(str(exc))
            return
        img = next((i for i in sec.items if isinstance(i, Image)), None)
        if img is not None:
            self.graph.set_png(img.png, f"{v.col}_{key}")

"""Optimization and Post Analysis nodes: Numerical (Criteria / Solutions / Graphs), Graphical
(overlay plot with the yellow feasible window, optionally on interval bounds), Point Prediction
(CI, PI, TI and POE), Confirmation (individual runs), the Coefficients Table and the POE graph.
The models are the ones chosen on each response's analysis node.

Settings stored in ``doc.optimization`` (JSON, saved in the .rsx file):

* ``numerical``           list of criteria (``optimize.Criterion.to_meta``)
* ``graphical``           {response: [lower, upper]}; ``graphical_interval`` {response: kind}
* ``confirmation``        {response: [n, mean]}; ``confirmation_runs`` {response: [values]}
* ``poe_sd``              {factor column: standard deviation in actual units}
* ``discrete``            {factor column: [allowed levels]}
* ``options``             starts, duplicate, level, proportion, show_se, show_poe,
                          second_order, side (point prediction)
"""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QComboBox,
                               QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout, QHeaderView,
                               QLabel, QPushButton, QScrollArea, QSpinBox, QSplitter,
                               QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from rsstatchem.core.doe import intervals, optimize, plots_opt, poe, report, report_post
from rsstatchem.core.output import Table, Text
from rsstatchem.ui.dx import style
from rsstatchem.ui.dx.layout import parse_number
from rsstatchem.ui.dx.widgets import (FactorsTool, GraphView, ReportView, TabStrip, annotation,
                                      hint, page_title, tool_button)

num = report.num
GOALS = optimize.GOALS
FACTOR_GOALS = optimize.FACTOR_GOALS
IMPORTANCE = ["+", "++", "+++", "++++", "+++++"]
INTERVAL_CHOICES = [("none", "Predicted"), ("ci", "CI"), ("pi", "PI"), ("ti", "TI")]
DEFAULT_OPTIONS = {"starts": 30, "duplicate": 0.002, "level": None, "proportion": None,
                   "show_se": False, "show_poe": False, "second_order": False, "side": "two"}


def no_models_text(doc) -> str:
    return ("No response has a fitted model yet. Enter the response data in the design "
            "layout and open each response under Analysis (Fit Summary > Model > ANOVA); the "
            "chosen models are used here.")


def _names(doc) -> dict:
    return {c: doc.response_title(i) for i, c in enumerate(doc.response_cols)}


def _criterion_label(doc, c: optimize.Criterion) -> str:
    if c.kind == "factor":
        j = doc.factor_cols.index(c.name)
        return doc.factor_title(j)
    base = doc.response_title(doc.response_index(c.name))
    return {"se": f"SE({base})", "poe": f"POE({base})"}.get(c.kind, base)


# ---------------------------------------------------------------------------- settings
def options(doc) -> dict:
    """Optimisation options with defaults (level / proportion from the preferences)."""
    out = dict(DEFAULT_OPTIONS)
    out.update({k: v for k, v in doc.optimization.get("options", {}).items() if v is not None})
    lvl, prop = intervals.defaults()
    out["level"] = float(out["level"] or lvl)
    out["proportion"] = float(out["proportion"] or prop)
    return out


def set_option(doc, key: str, value) -> None:
    doc.optimization.setdefault("options", {})[key] = value
    doc.touch("optimization")


def poe_sd(doc) -> dict:
    return dict(doc.optimization.get("poe_sd", {}) or {})


def discrete_levels(doc) -> dict[int, list[float]]:
    """{factor index: allowed levels} from ``doc.optimization["discrete"]`` and, when the
    design stores them, the factors' own ``discrete`` / ``discrete_levels`` entries."""
    stored = doc.optimization.get("discrete", {}) or {}
    out = {}
    for j, col in enumerate(doc.factor_cols):
        info = doc.factors[j] if j < len(doc.factors) else {}
        lv = stored.get(col) or info.get("discrete_levels") or (
            info.get("discrete") if isinstance(info.get("discrete"), list) else None)
        if lv:
            vals = [float(v) for v in lv if v is not None and math.isfinite(float(v))]
            if vals:
                out[j] = sorted(set(vals))
    return out


def design_points(doc) -> np.ndarray:
    return doc.ds.df[doc.factor_cols].to_numpy(float)


def make_problem(doc, fits: dict, criteria: list[optimize.Criterion]) -> optimize.Problem:
    opt = options(doc)
    return optimize.Problem(doc.space(), fits, list(criteria), discrete=discrete_levels(doc),
                            poe_sd=poe_sd(doc), poe_second_order=bool(opt["second_order"]),
                            level=opt["level"], proportion=opt["proportion"])


def stored_criteria(doc, fits: dict) -> list[optimize.Criterion]:
    """The remembered criteria, completed with defaults for new factors/responses; SE and
    POE rows are listed when they have a goal or are switched on in the options."""
    space = doc.space()
    opt = options(doc)
    saved = {(d.get("kind", "response"), d["name"]): d
             for d in doc.optimization.get("numerical", [])}
    out = []
    for f in space.factors:
        if f.categoric:
            continue
        d = saved.get(("factor", f.column))
        out.append(optimize.Criterion.from_meta(d) if d else
                   optimize.Criterion(f.column, "factor", "in_range", f.low, f.high))
    pts = None
    for col in doc.response_cols:
        if col not in fits:
            continue
        d = saved.get(("response", col))
        if d:
            out.append(optimize.Criterion.from_meta(d))
        else:
            y = fits[col].obs.y
            out.append(optimize.Criterion(col, "response", "none", float(np.min(y)),
                                          float(np.max(y))))
    for kind, flag in (("se", "show_se"), ("poe", "show_poe")):
        for col in doc.response_cols:
            if col not in fits:
                continue
            d = saved.get((kind, col))
            if d and (opt[flag] or d.get("goal", "none") != "none"):
                out.append(optimize.Criterion.from_meta(d))
            elif opt[flag]:
                if pts is None:
                    pts = design_points(doc)
                out.append(optimize.derived_criterion(kind, col, fits[col], pts, poe_sd(doc),
                                                      bool(opt["second_order"])))
    return out


def save_criteria(doc, criteria: list[optimize.Criterion]) -> None:
    keep = {c.key for c in criteria}
    old = [d for d in doc.optimization.get("numerical", [])
           if (d.get("kind", "response"), d["name"]) not in keep]
    doc.optimization["numerical"] = [c.to_meta() for c in criteria] + old
    doc.touch("optimization")


def _scroll(widget: QWidget) -> QScrollArea:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setWidget(widget)
    return scroll


def _drop_cell_widgets(table: QTableWidget) -> None:
    """Hide and release the cell widgets before a table is refilled (Qt deletes replaced
    widgets only later, so they would stay visible until then)."""
    for r in range(table.rowCount()):
        for c in range(table.columnCount()):
            w = table.cellWidget(r, c)
            if w is not None:
                w.hide()
                table.removeCellWidget(r, c)


def _clear(layout) -> None:
    while layout.count():
        w = layout.takeAt(0).widget()
        if w is not None:
            w.setParent(None)


# ---------------------------------------------------------------------------- Numerical
class NumericalView(QWidget):
    COLS = ["Name", "Goal", "Lower Limit", "Upper Limit", "Target", "Lower Weight",
            "Upper Weight", "Importance", "Interval", "LSL", "USL"]

    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.solutions: list[dict] = []
        self.problem: optimize.Problem | None = None
        self.strip = TabStrip([("numerical", "Criteria"), ("ramps", "Solutions"),
                               ("contour", "Graphs")])
        self.stack = QStackedWidget()
        # criteria
        crit = QWidget()
        cl = QVBoxLayout(crit)
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col, width in enumerate((0, 118, 92, 92, 80, 92, 92, 86, 96, 70, 70)):
            if width:
                self.table.setColumnWidth(col, width)
        self.table.verticalHeader().setDefaultSectionSize(26)
        self.table.itemChanged.connect(self._edited)
        cl.addWidget(hint("Set a goal for each response (maximize, minimize, target, in range, "
                          "Cpk) and, if wanted, limits for the factors or 'is equal to' to "
                          "hold one fixed. Importance (+ to +++++) weights a goal in the "
                          "overall desirability D = (Π dᵢ^rᵢ)^(1/Σrᵢ). Interval applies the "
                          "goal to a CI, PI or TI bound instead of the predicted value. Then "
                          "open Solutions."))
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.table)
        split.addWidget(self._options_panel())
        split.setSizes([300, 260])
        cl.addWidget(split, 1)
        self.stack.addWidget(crit)
        # solutions
        sol = QWidget()
        sl = QVBoxLayout(sol)
        bar = QHBoxLayout()
        self.solve_btn = QPushButton("Find Solutions")
        self.solve_btn.clicked.connect(self.solve)
        bar.addWidget(self.solve_btn)
        self.view_group = QButtonGroup(self)
        self.view_buttons = []
        for i, (icon_name, text) in enumerate((("ramps", "Ramps"), ("bars", "Bar Graph"),
                                               ("report", "Report"))):
            b = tool_button(icon_name, text)
            b.setAutoExclusive(True)
            b.clicked.connect(lambda _=False, k=i: self._show_solution_view(k))
            self.view_group.addButton(b)
            bar.addWidget(b)
            self.view_buttons.append(b)
        bar.addStretch(1)
        self.sol_status = QLabel()
        bar.addWidget(self.sol_status)
        sl.addLayout(bar)
        self.sol_table = QTableWidget(0, 0)
        self.sol_table.verticalHeader().setVisible(False)
        self.sol_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.sol_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.sol_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.sol_table.itemSelectionChanged.connect(self._solution_selected)
        self.sol_graph = GraphView()
        self.sol_report = ReportView()
        self.sol_stack = QStackedWidget()
        self.sol_stack.addWidget(self.sol_graph)
        self.sol_stack.addWidget(self.sol_report)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.sol_table)
        split.addWidget(self.sol_stack)
        split.setSizes([200, 420])
        sl.addWidget(split, 1)
        self.stack.addWidget(sol)
        # graphs
        gr = QWidget()
        gl = QHBoxLayout(gr)
        gl.setContentsMargins(0, 0, 0, 0)
        panel = QFrame()
        panel.setObjectName("toolPanel")
        panel.setFixedWidth(250)
        pl = QVBoxLayout(panel)
        pl.addWidget(QLabel("Show:"))
        self.what = QComboBox()
        self.what.currentIndexChanged.connect(lambda _: self.render_graph())
        pl.addWidget(self.what)
        pl.addWidget(QLabel("Graph:"))
        self.gtype = QComboBox()
        self.gtype.addItem("Contour", "contour")
        self.gtype.addItem("3D Surface", "3d")
        self.gtype.currentIndexChanged.connect(lambda _: self.render_graph())
        pl.addWidget(self.gtype)
        self.tool_holder = QVBoxLayout()
        pl.addLayout(self.tool_holder, 1)
        self.tool: FactorsTool | None = None
        self.graph = GraphView()
        gl.addWidget(panel)
        gl.addWidget(self.graph, 1)
        self.stack.addWidget(gr)
        self.message = hint("")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.strip)
        lay.addWidget(self.message)
        lay.addWidget(self.stack, 1)
        self.strip.changed.connect(self._tab)
        self.strip.set_current(0, emit=False)
        self.view_buttons[0].setChecked(True)
        self.criteria: list[optimize.Criterion] = []
        self._loading = False
        self.refresh()

    # ------------------------------------------------------------------ options panel
    def _options_panel(self) -> QWidget:
        box = QFrame()
        box.setObjectName("toolPanel")
        outer = QHBoxLayout(box)
        left = QGridLayout()
        title = QLabel("Options")
        title.setStyleSheet(f"color:{style.BLUE}; font-weight:600;")
        left.addWidget(title, 0, 0, 1, 2)
        self.starts_spin = QSpinBox()
        self.starts_spin.setRange(1, 1000)
        self.dup_spin = QDoubleSpinBox()
        self.dup_spin.setDecimals(4)
        self.dup_spin.setRange(0.0, 0.5)
        self.dup_spin.setSingleStep(0.001)
        self.level_spin = QDoubleSpinBox()
        self.level_spin.setRange(50.0, 99.99)
        self.level_spin.setSuffix(" %")
        self.prop_spin = QDoubleSpinBox()
        self.prop_spin.setRange(50.0, 99.999)
        self.prop_spin.setDecimals(3)
        self.prop_spin.setSuffix(" %")
        rows = [("Random starts", self.starts_spin), ("Duplicate filter (fraction of range)",
                                                      self.dup_spin),
                ("Interval confidence", self.level_spin),
                ("TI: population covered", self.prop_spin)]
        for r, (text, w) in enumerate(rows, start=1):
            left.addWidget(QLabel(text), r, 0)
            left.addWidget(w, r, 1)
        self.show_se = QCheckBox("Standard-error rows (SE of each response)")
        self.show_poe = QCheckBox("POE rows (propagation of error)")
        self.second = QCheckBox("Second-order POE")
        for i, w in enumerate((self.show_se, self.show_poe, self.second)):
            left.addWidget(w, 5 + i, 0, 1, 2)
        self.robust_btn = QPushButton("Robust Design: minimize POE of goal responses")
        self.robust_btn.clicked.connect(self.make_robust)
        left.addWidget(self.robust_btn, 8, 0, 1, 2)
        left.setRowStretch(9, 1)
        outer.addLayout(left, 1)
        right = QVBoxLayout()
        t2 = QLabel("Factor variation (POE) and discrete levels")
        t2.setStyleSheet(f"color:{style.BLUE}; font-weight:600;")
        right.addWidget(t2)
        self.factor_table = QTableWidget(0, 3)
        self.factor_table.setHorizontalHeaderLabels(["Factor", "Std Dev (POE)",
                                                     "Discrete levels (e.g. 2, 2.5, 3)"])
        self.factor_table.setColumnWidth(1, 110)
        self.factor_table.verticalHeader().setVisible(False)
        self.factor_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch)
        self.factor_table.itemChanged.connect(self._factor_edited)
        right.addWidget(self.factor_table, 1)
        outer.addLayout(right, 1)
        self.starts_spin.valueChanged.connect(lambda v: self._opt("starts", int(v)))
        self.dup_spin.valueChanged.connect(lambda v: self._opt("duplicate", float(v)))
        self.level_spin.valueChanged.connect(lambda v: self._opt("level", float(v) / 100))
        self.prop_spin.valueChanged.connect(lambda v: self._opt("proportion", float(v) / 100))
        self.show_se.toggled.connect(lambda v: self._opt("show_se", bool(v), True))
        self.show_poe.toggled.connect(lambda v: self._opt("show_poe", bool(v), True))
        self.second.toggled.connect(lambda v: self._opt("second_order", bool(v)))
        return box

    def _fill_options(self) -> None:
        opt = options(self.doc)
        self._loading = True
        self.starts_spin.setValue(int(opt["starts"]))
        self.dup_spin.setValue(float(opt["duplicate"]))
        self.level_spin.setValue(100 * opt["level"])
        self.prop_spin.setValue(100 * opt["proportion"])
        self.show_se.setChecked(bool(opt["show_se"]))
        self.show_poe.setChecked(bool(opt["show_poe"]))
        self.second.setChecked(bool(opt["second_order"]))
        sp = self.doc.space()
        sds = poe.sd_vector(sp, poe_sd(self.doc))
        disc = discrete_levels(self.doc)
        rows = [j for j, f in enumerate(sp.factors) if not f.categoric]
        self.factor_table.setRowCount(len(rows))
        for r, j in enumerate(rows):
            name = QTableWidgetItem(self.doc.factor_title(j))
            name.setFlags(Qt.ItemFlag.ItemIsEnabled)
            name.setData(Qt.ItemDataRole.UserRole, j)
            self.factor_table.setItem(r, 0, name)
            sd_item = QTableWidgetItem(f"{sds[j]:.6g}" if sds[j] > 0 else "")
            sd_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.factor_table.setItem(r, 1, sd_item)
            lv = disc.get(j)
            disc_item = QTableWidgetItem(", ".join(f"{v:g}" for v in lv) if lv else "")
            if sp.mixture:
                disc_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.factor_table.setItem(r, 2, disc_item)
        self.factor_table.setColumnWidth(0, max(self.factor_table.sizeHintForColumn(0) + 12,
                                                 110))
        self.robust_btn.setEnabled(poe.has_sd(sp, poe_sd(self.doc)))
        self._loading = False

    def _opt(self, key: str, value, rebuild: bool = False) -> None:
        if self._loading:
            return
        set_option(self.doc, key, value)
        if rebuild:
            self.criteria = stored_criteria(self.doc, self.fits)
            self._fill_table()
            self._fill_what()

    def _factor_edited(self, item) -> None:
        if self._loading or item.column() == 0:
            return
        j = self.factor_table.item(item.row(), 0).data(Qt.ItemDataRole.UserRole)
        if item.column() == 1:
            try:
                v = parse_number(item.text()) if item.text().strip() else 0.0
            except ValueError:
                v = float("nan")
            self.set_poe_sd(self.doc.factor_cols[j], v if math.isfinite(v) else 0.0)
        else:
            vals = report_post.parse_runs(item.text())
            self.set_discrete(self.doc.factor_cols[j], vals)

    # programmatic settings (tests, scripts) ------------------------------------------
    def set_option(self, key: str, value) -> None:
        set_option(self.doc, key, value)
        self.refresh()

    def set_poe_sd(self, col: str, sd: float) -> None:
        d = self.doc.optimization.setdefault("poe_sd", {})
        if sd and sd > 0:
            d[col] = float(sd)
        else:
            d.pop(col, None)
        self.doc.touch("optimization")
        self._refresh_derived()

    def set_discrete(self, col: str, levels) -> None:
        d = self.doc.optimization.setdefault("discrete", {})
        vals = [float(v) for v in (levels or [])]
        if vals:
            d[col] = vals
        else:
            d.pop(col, None)
        self.doc.touch("optimization")

    def _refresh_derived(self) -> None:
        """New factor SDs change the POE rows' default limits."""
        pts = design_points(self.doc)
        opt = options(self.doc)
        for c in self.criteria:
            if c.kind == "poe" and c.goal == "none":
                new = optimize.derived_criterion("poe", c.name, self.fits[c.name], pts,
                                                 poe_sd(self.doc), bool(opt["second_order"]))
                c.lower, c.upper = new.lower, new.upper
        self._fill_table()
        self._loading = True
        self.robust_btn.setEnabled(poe.has_sd(self.doc.space(), poe_sd(self.doc)))
        self._loading = False

    def make_robust(self) -> None:
        """Robust design: add 'minimize POE' goals for every response with a goal."""
        opt = options(self.doc)
        self.criteria = optimize.robust_criteria(self.criteria, self.fits,
                                                 design_points(self.doc), poe_sd(self.doc),
                                                 bool(opt["second_order"]))
        save_criteria(self.doc, self.criteria)
        self._fill_table()
        self._fill_what()

    # ------------------------------------------------------------------ state
    @property
    def graph_visible(self) -> bool:
        return self.stack.currentIndex() == 2 or (self.stack.currentIndex() == 1
                                                  and self.sol_stack.currentIndex() == 0)

    def current_graph(self):
        if self.stack.currentIndex() == 2:
            return self.graph
        if self.stack.currentIndex() == 1 and self.sol_stack.currentIndex() == 0:
            return self.sol_graph
        return None

    # ------------------------------------------------------------------ criteria
    def refresh(self) -> None:
        self.fits = self.doc.fits()
        if not self.fits:
            self.message.setText(no_models_text(self.doc))
            self.message.show()
            self.stack.setEnabled(False)
            return
        self.message.hide()
        self.stack.setEnabled(True)
        self.criteria = stored_criteria(self.doc, self.fits)
        self._fill_table()
        self._fill_options()
        self._fill_what()
        if self.tool is not None:
            self.tool.setParent(None)
        self.tool = FactorsTool(self.doc.space(), n_axes=2 if not self.doc.mixture else 3)
        self.tool.changed.connect(self.render_graph)
        _clear(self.tool_holder)
        self.tool_holder.addWidget(_scroll(self.tool))
        if self.stack.currentIndex() == 1:
            self.solve()

    def _fill_what(self) -> None:
        cur = self.what.currentData()
        self.what.blockSignals(True)
        self.what.clear()
        self.what.addItem("Desirability", "desirability")
        for col in self.fits:
            self.what.addItem(self.doc.response_title(self.doc.response_index(col)),
                              f"response:{col}")
        for col in self.fits:
            self.what.addItem(f"SE({self.doc.response_title(self.doc.response_index(col))})",
                              f"se:{col}")
        if poe.has_sd(self.doc.space(), poe_sd(self.doc)) or options(self.doc)["show_poe"]:
            for col in self.fits:
                self.what.addItem(
                    f"POE({self.doc.response_title(self.doc.response_index(col))})",
                    f"poe:{col}")
        self.what.addItem("Ramps at this point", "ramps:")
        self.what.addItem("Bar graph at this point", "bars:")
        i = self.what.findData(cur)
        self.what.setCurrentIndex(max(i, 0))
        self.what.blockSignals(False)

    def _fill_table(self) -> None:
        self._loading = True
        _drop_cell_widgets(self.table)
        self.table.setRowCount(len(self.criteria))
        for r, c in enumerate(self.criteria):
            name = QTableWidgetItem(_criterion_label(self.doc, c))
            name.setFlags(Qt.ItemFlag.ItemIsEnabled)
            color = {"response": style.BLUE_DARK, "se": "#6a1b9a", "poe": "#B35C00"}
            name.setForeground(QBrush(QColor(color.get(c.kind, "#333333"))))
            self.table.setItem(r, 0, name)
            goal = QComboBox()
            choices = FACTOR_GOALS if c.kind == "factor" else (
                GOALS if c.kind == "response" else optimize.DERIVED_GOALS)
            for key, text in choices:
                goal.addItem(text, key)
            goal.setCurrentIndex(max(goal.findData(c.goal), 0))
            goal.currentIndexChanged.connect(lambda _=0, rr=r: self._goal_changed(rr))
            self.table.setCellWidget(r, 1, goal)
            target = c.target if c.target is not None else (c.lower + c.upper) / 2
            for col, v in ((2, c.lower), (3, c.upper), (4, target), (5, c.w_lower),
                           (6, c.w_upper), (9, c.lsl), (10, c.usl)):
                it = QTableWidgetItem("" if v is None else f"{v:.6g}")
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(r, col, it)
            imp = QComboBox()
            imp.addItems(IMPORTANCE)
            imp.setCurrentIndex(min(max(c.importance - 1, 0), 4))
            imp.currentIndexChanged.connect(lambda _=0, rr=r: self._imp_changed(rr))
            self.table.setCellWidget(r, 7, imp)
            if c.kind == "response":
                iv = QComboBox()
                for key, text in INTERVAL_CHOICES:
                    iv.addItem(text, key)
                iv.setCurrentIndex(max(iv.findData(c.interval or "none"), 0))
                iv.currentIndexChanged.connect(lambda _=0, rr=r: self._interval_changed(rr))
                self.table.setCellWidget(r, 8, iv)
            else:
                blank = QTableWidgetItem("")
                blank.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self.table.setItem(r, 8, blank)
            self._enable_row(r)
        self._loading = False

    def _enable_row(self, r: int) -> None:
        c = self.criteria[r]
        for col in (2, 3, 4, 5, 6, 9, 10):
            it = self.table.item(r, col)
            if c.goal == "equal":
                on = col == 4
            elif col in (9, 10):
                on = c.goal == "cpk"
            else:
                on = c.goal != "none" and not (col == 4 and c.goal != "target") and not (
                    col in (5, 6) and c.goal == "in_range") and not (
                    col == 6 and c.goal == "cpk")
            flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            if on:
                flags |= Qt.ItemFlag.ItemIsEditable
            it.setFlags(flags)
            it.setForeground(QBrush(QColor("#000000" if on else "#A0A7B0")))
        self.table.cellWidget(r, 7).setEnabled(c.active)
        w = self.table.cellWidget(r, 8)
        if w is not None:
            w.setEnabled(c.goal not in ("none", "cpk"))

    def _find(self, name: str, kind: str | None):
        for c in self.criteria:
            if c.name == name and (c.kind == kind if kind else c.kind in ("response",
                                                                          "factor")):
                yield c

    def set_goal(self, name: str, goal: str, lower=None, upper=None, target=None,
                 importance: int | None = None, kind: str | None = None,
                 interval: str | None = None, lsl=None, usl=None, side: str | None = None,
                 weights: tuple | None = None) -> None:
        """Programmatic criterion change (tests, scripts). ``kind`` = response / factor
        (default: either), se or poe; SE and POE rows are created when missing."""
        if kind in ("se", "poe") and not list(self._find(name, kind)):
            self.criteria.append(optimize.derived_criterion(
                kind, name, self.fits[name], design_points(self.doc), poe_sd(self.doc),
                bool(options(self.doc)["second_order"])))
        for c in self._find(name, kind):
            if goal == "cpk" and c.goal != "cpk":
                # Cpk limits are on the Cpk scale; the response limits become the specs
                c.lsl = c.lsl if c.lsl is not None else c.lower
                c.usl = c.usl if c.usl is not None else c.upper
                c.lower, c.upper = optimize.CPK_LIMITS
            c.goal = goal
            if lower is not None:
                c.lower = float(lower)
            if upper is not None:
                c.upper = float(upper)
            if target is not None:
                c.target = float(target)
            if importance is not None:
                c.importance = int(importance)
            if interval is not None:
                c.interval = interval
            if side is not None:
                c.side = side
            if lsl is not None:
                c.lsl = float(lsl)
            if usl is not None:
                c.usl = float(usl)
            if weights is not None:
                c.w_lower, c.w_upper = float(weights[0]), float(weights[1])
        save_criteria(self.doc, self.criteria)
        self._fill_table()

    def _goal_changed(self, r: int) -> None:
        if self._loading:
            return
        c = self.criteria[r]
        new = self.table.cellWidget(r, 1).currentData()
        if new == "cpk" and c.goal != "cpk":
            # Cpk limits are on the Cpk scale; the old response limits become the specs
            c.lsl = c.lsl if c.lsl is not None else c.lower
            c.usl = c.usl if c.usl is not None else c.upper
            c.lower, c.upper = optimize.CPK_LIMITS
        elif c.goal == "cpk" and new != "cpk" and c.lsl is not None and c.usl is not None:
            c.lower, c.upper = c.lsl, c.usl
        c.goal = new
        self._fill_table()
        save_criteria(self.doc, self.criteria)

    def _imp_changed(self, r: int) -> None:
        if self._loading:
            return
        self.criteria[r].importance = self.table.cellWidget(r, 7).currentIndex() + 1
        save_criteria(self.doc, self.criteria)

    def _interval_changed(self, r: int) -> None:
        if self._loading:
            return
        self.criteria[r].interval = self.table.cellWidget(r, 8).currentData()
        save_criteria(self.doc, self.criteria)

    def _edited(self, item) -> None:
        if self._loading or item.column() < 2 or item.column() in (7, 8):
            return
        c = self.criteria[item.row()]
        attr = {2: "lower", 3: "upper", 4: "target", 5: "w_lower", 6: "w_upper", 9: "lsl",
                10: "usl"}[item.column()]
        if attr in ("lsl", "usl") and not item.text().strip():
            setattr(c, attr, None)
            save_criteria(self.doc, self.criteria)
            return
        try:
            v = parse_number(item.text())
        except ValueError:
            v = float("nan")
        if not math.isfinite(v):
            self._fill_table()
            return
        setattr(c, attr, v)
        save_criteria(self.doc, self.criteria)

    # ------------------------------------------------------------------ solutions
    def _tab(self, i: int) -> None:
        self.stack.setCurrentIndex(i)
        if i == 1:
            self.solve()
        elif i == 2:
            if not self.solutions:
                self.solve()
            self.render_graph()

    def solve(self) -> list[dict]:
        if not self.fits:
            return []
        active = [c for c in self.criteria if c.kind != "factor" and c.goal != "none"]
        if not active:
            self.sol_status.setText("Set a goal for at least one response on the Criteria tab.")
            self.sol_table.setRowCount(0)
            self.sol_graph.set_message("No response goals: set at least one goal (Criteria).")
            self.solutions = []
            return []
        for c in active:
            if c.goal == "cpk" and c.lsl is None and c.usl is None:
                self.sol_status.setText(f"{_criterion_label(self.doc, c)}: the Cpk goal needs "
                                        "a lower and/or upper specification limit (LSL/USL).")
                self.solutions = []
                return []
        self.problem = make_problem(self.doc, self.fits, self.criteria)
        opt = options(self.doc)
        self.starts = int(opt["starts"])
        try:
            self.solutions = optimize.solve(self.problem, starts=self.starts, seed=0,
                                            design_points=design_points(self.doc),
                                            duplicate=float(opt["duplicate"]))
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.sol_status.setText(str(exc))
            return []
        self.doc.last_solutions = self.solutions
        self._fill_solutions()
        return self.solutions

    def _fill_solutions(self) -> None:
        sp = self.doc.space()
        resp = [c.name for c in self.criteria if c.kind == "response"]
        extra = list(self.solutions[0].get("extra", {})) if self.solutions else []
        names = _names(self.doc)
        extra_titles = []
        for k in extra:
            head, _, col = k.partition("(")
            extra_titles.append(f"{head}({names.get(col[:-1], col[:-1]).split(':')[0]})")
        cols = ["Number"] + [self.doc.factor_title(j) for j in range(sp.k)] + \
            [self.doc.response_title(self.doc.response_index(r)) for r in resp] + \
            extra_titles + ["Desirability", ""]
        self.sol_table.clear()
        self.sol_table.setColumnCount(len(cols))
        self.sol_table.setHorizontalHeaderLabels(cols)
        self.sol_table.setRowCount(len(self.solutions))
        for i, s in enumerate(self.solutions):
            vals = [str(i + 1)]
            for j, f in enumerate(sp.factors):
                vals.append(report._fmt_level(f, s["x"][j]).split(" ")[0] if f.categoric
                            else num(s["x"][j]))
            vals += [num(s["pred"].get(r)) for r in resp]
            vals += [num(s.get("extra", {}).get(k)) for k in extra]
            vals += [f"{s['D']:.3f}", "Selected" if i == 0 else ""]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                if c == len(vals) - 2:
                    it.setBackground(QBrush(QColor("#E7F6EC" if s["D"] > 0 else "#FDECEC")))
                self.sol_table.setItem(i, c, it)
        self.sol_table.resizeColumnsToContents()
        n = len(self.solutions)
        self.sol_status.setText(f"{n} solution{'s' if n != 1 else ''} found "
                                f"({self.starts} random starts + design points, Nelder-Mead).")
        if self.solutions:
            self.sol_table.selectRow(0)
            self._solution_selected()

    def selected(self) -> int:
        rows = self.sol_table.selectionModel().selectedRows() if self.sol_table.selectionModel() \
            else []
        return rows[0].row() if rows else 0

    def _show_solution_view(self, k: int) -> None:
        self.view_buttons[k].setChecked(True)
        self._solution_selected()

    def _bars_png(self, s: dict) -> bytes:
        names = _names(self.doc)
        act = self.problem.active_criteria()
        labels = [report_post.criterion_name(c, self.problem.space, names) for c in act]
        return plots_opt.desirability_bars(labels, s["d"], s["D"])

    def _solution_selected(self) -> None:
        if not self.solutions or self.problem is None:
            return
        s = self.solutions[min(self.selected(), len(self.solutions) - 1)]
        k = next(i for i, b in enumerate(self.view_buttons) if b.isChecked())
        names = _names(self.doc)
        if k == 0:
            self.sol_stack.setCurrentIndex(0)
            self.sol_graph.set_png(plots_opt.ramps_plot(
                report_post.ramp_entries(self.problem, s, names), s["D"],
                f"Desirability Ramps (Solution {self.selected() + 1})"), "ramps")
        elif k == 1:
            self.sol_stack.setCurrentIndex(0)
            self.sol_graph.set_png(self._bars_png(s), "desirability_bars")
        else:
            sec = report_post.optimization_section(self.problem, self.solutions, names,
                                                   ramps=False, bars=False,
                                                   starts=getattr(self, "starts", 30))
            self.sol_stack.setCurrentIndex(1)
            items = [i for i in sec.items if isinstance(i, (Table, Text))]
            self.sol_report.set_items(items, "Optimization Report")
        if self.tool is not None:
            self.tool.set_point(s["x"])

    # ------------------------------------------------------------------ graphs
    @staticmethod
    def _parse_what(data) -> tuple[str, str]:
        """'kind:column' item data of the Show box -> (kind, column)."""
        if not data or data == "desirability":
            return "desirability", ""
        kind, _, col = str(data).partition(":")
        return kind, col

    def _values(self, what, X: np.ndarray) -> tuple[np.ndarray, str]:
        kind, col = self._parse_what(what)
        if kind == "desirability":
            return self.problem.evaluate(X)["D"], "Desirability"
        fit = self.fits[col]
        title = self.what.currentText()
        if kind == "response":
            return fit.predict(X), title
        if kind == "se":
            return optimize.se_mean(fit, X), title
        return self.problem.sigma(fit, X), title

    def render_graph(self) -> None:
        if self.tool is None or not self.fits or self.stack.currentIndex() != 2:
            return
        if self.problem is None:
            self.graph.set_message("Find solutions first (Solutions tab).")
            return
        sp = self.problem.space
        axes = self.tool.axes()
        ref = self.tool.point()
        what = self.what.currentData()
        wkind = self._parse_what(what)[0]
        if wkind in ("ramps", "bars"):
            s = report_post.evaluate_point(self.problem, ref)
            if wkind == "ramps":
                self.graph.set_png(plots_opt.ramps_plot(
                    report_post.ramp_entries(self.problem, s, _names(self.doc)), s["D"],
                    "Desirability Ramps (Factors Tool point)"), "ramps_point")
            else:
                self.graph.set_png(self._bars_png(s), "bars_point")
            return
        best = self.solutions[min(self.selected(), len(self.solutions) - 1)] \
            if self.solutions else None
        if sp.mixture:
            if len(axes) < 3:
                self.graph.set_message("Assign three components to X1, X2 and X3.")
                return
            tri = axes[:3]
            rest = [j for j in range(sp.k) if j not in tri]
            lows = sp.lower[tri]
            free = sp.total - ref[rest].sum() - lows.sum()
            if free <= 0:
                self.graph.set_message("The other components leave no room for the plot.")
                return

            def to_full(a3):
                X = np.tile(ref, (len(a3), 1))
                X[:, tri] = a3
                return X

            def fn(a3):
                return self._values(what, to_full(a3))[0]

            def feas(a3):
                X = to_full(a3)
                return np.all((X >= sp.lower - 1e-9) & (X <= sp.upper + 1e-9), axis=1)

            from rsstatchem.core.doe import plots

            title = "Desirability" if wkind == "desirability" else self.what.currentText()
            self.graph.set_png(plots.ternary_contour(
                fn, lows, free, [sp.factors[j].title for j in tri], title, title, feas,
                points=best["x"][tri][None, :] if best is not None else None), "desirability")
            return
        if len(axes) < 2:
            self.graph.set_message("Assign two factors to the X1 and X2 axes.")
            return
        i, j = axes[:2]
        b = self.problem.bounds_actual()
        bi = b[i] if b[i][1] > b[i][0] else (sp.factors[i].low, sp.factors[i].high)
        bj = b[j] if b[j][1] > b[j][0] else (sp.factors[j].low, sp.factors[j].high)
        gx, gy = np.meshgrid(np.linspace(bi[0], bi[1], 60), np.linspace(bj[0], bj[1], 60))
        X = np.tile(ref, (gx.size, 1))
        X[:, i], X[:, j] = gx.ravel(), gy.ravel()
        Z, title = self._values(what, X)
        Z = Z.reshape(gx.shape)
        xl, yl = sp.factors[i].axis_label, sp.factors[j].axis_label
        if self.gtype.currentData() == "3d":
            self.graph.set_figure(plots_opt.surface_figure(gx, gy, Z, xl, yl, title, title),
                                  "optimization_surface", rotatable=True)
        else:
            self.graph.set_png(plots_opt.contour(gx, gy, Z, xl, yl, title, zlabel=title,
                                                 solution=(ref[i], ref[j])),
                               "optimization_graph")


# ---------------------------------------------------------------------------- Graphical
class GraphicalView(QWidget):
    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.strip = TabStrip([("graphical", "Criteria"), ("contour", "Graphs")])
        self.stack = QStackedWidget()
        crit = QWidget()
        cl = QVBoxLayout(crit)
        cl.addWidget(hint("Enter a lower and/or upper limit for each response that must be "
                          "met. On the Graphs tab the region meeting all limits is shown in "
                          "yellow (the 'sweet spot'); grey areas fail at least one limit. "
                          "With an interval (CI, PI or TI) the limits are applied to its "
                          "bounds — one-sided when only one limit is given — and the interval "
                          "bound is drawn as a dashed line next to the prediction."))
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Response", "Lower Limit", "Upper Limit",
                                              "Interval"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.itemChanged.connect(self._edited)
        cl.addWidget(self.table, 1)
        self.stack.addWidget(crit)
        gr = QWidget()
        gl = QHBoxLayout(gr)
        gl.setContentsMargins(0, 0, 0, 0)
        panel = QFrame()
        panel.setObjectName("toolPanel")
        panel.setFixedWidth(250)
        self.panel_lay = QVBoxLayout(panel)
        self.tool: FactorsTool | None = None
        self.graph = GraphView()
        gl.addWidget(panel)
        gl.addWidget(self.graph, 1)
        self.stack.addWidget(gr)
        self.message = hint("")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.strip)
        lay.addWidget(self.message)
        lay.addWidget(self.stack, 1)
        self.strip.changed.connect(self._tab)
        self.strip.set_current(0, emit=False)
        self._loading = False
        self.refresh()

    graph_visible = True

    def current_graph(self):
        return self.graph if self.stack.currentIndex() == 1 else None

    def limits(self) -> dict:
        return self.doc.optimization.setdefault("graphical", {})

    def interval_of(self, col: str) -> str:
        return self.doc.optimization.get("graphical_interval", {}).get(col, "none")

    def refresh(self) -> None:
        self.fits = self.doc.fits()
        if not self.fits:
            self.message.setText(no_models_text(self.doc))
            self.message.show()
            self.stack.setEnabled(False)
            return
        self.message.hide()
        self.stack.setEnabled(True)
        self._loading = True
        cols = list(self.fits)
        _drop_cell_widgets(self.table)
        self.table.setRowCount(len(cols))
        lim = self.limits()
        for r, col in enumerate(cols):
            name = QTableWidgetItem(self.doc.response_title(self.doc.response_index(col)))
            name.setFlags(Qt.ItemFlag.ItemIsEnabled)
            name.setData(Qt.ItemDataRole.UserRole, col)
            self.table.setItem(r, 0, name)
            lo, hi = (list(lim.get(col, [None, None])) + [None, None])[:2]
            for c, v in ((1, lo), (2, hi)):
                self.table.setItem(r, c, QTableWidgetItem("" if v is None else f"{v:.6g}"))
            box = QComboBox()
            for key, text in INTERVAL_CHOICES:
                box.addItem(text, key)
            box.setCurrentIndex(max(box.findData(self.interval_of(col)), 0))
            box.currentIndexChanged.connect(
                lambda _=0, cc=col, b=box: self.set_interval(cc, b.currentData()))
            self.table.setCellWidget(r, 3, box)
        self._loading = False
        if self.tool is not None:
            self.tool.setParent(None)
        self.tool = FactorsTool(self.doc.space(), n_axes=3 if self.doc.mixture else 2)
        self.tool.changed.connect(self.render)
        _clear(self.panel_lay)
        self.panel_lay.addWidget(_scroll(self.tool))
        self.render()

    def set_limits(self, col: str, lower=None, upper=None) -> None:
        self.limits()[col] = [lower, upper]
        self.doc.touch("optimization")
        self.refresh()

    def set_interval(self, col: str, kind: str) -> None:
        if self._loading:
            return
        self.doc.optimization.setdefault("graphical_interval", {})[col] = kind or "none"
        self.doc.touch("optimization")
        self.render()

    def _edited(self, item) -> None:
        if self._loading or item.column() not in (1, 2):
            return
        col = self.table.item(item.row(), 0).data(Qt.ItemDataRole.UserRole)
        try:
            v = parse_number(item.text())
        except ValueError:
            v = float("nan")
        lim = self.limits().setdefault(col, [None, None])
        lim[item.column() - 1] = v if math.isfinite(v) else None
        self.doc.touch("optimization")

    def _tab(self, i: int) -> None:
        self.stack.setCurrentIndex(i)
        self.render()

    def _bounds(self, col, X, lo, hi):
        opt = options(self.doc)
        return optimize.overlay_bounds(self.fits[col], X, lo, hi, self.interval_of(col),
                                       opt["level"], opt["proportion"])

    def render(self) -> None:
        if self.tool is None or not self.fits or self.stack.currentIndex() != 1:
            return
        lim = self.limits()
        active = [(c, lim[c][0], lim[c][1]) for c in self.fits if c in lim
                  and (lim[c][0] is not None or lim[c][1] is not None)]
        if not active:
            self.graph.set_message("Enter at least one response limit on the Criteria tab.")
            return
        sp = self.doc.space()
        axes = self.tool.axes()
        ref = self.tool.point()
        names = _names(self.doc)
        if sp.mixture:
            if len(axes) < 3:
                self.graph.set_message("Assign three components to X1, X2 and X3.")
                return
            tri = axes[:3]
            rest = [j for j in range(sp.k) if j not in tri]
            lows = sp.lower[tri]
            free = sp.total - ref[rest].sum() - lows.sum()

            def to_full(a3):
                X = np.tile(ref, (len(a3), 1))
                X[:, tri] = a3
                return X

            def ok(a3):
                X = to_full(a3)
                good = np.ones(len(X), bool)
                for c, lo, hi in active:
                    bl, bh = self._bounds(c, X, lo, hi)
                    if lo is not None:
                        good &= bl >= lo
                    if hi is not None:
                        good &= bh <= hi
                return good.astype(float)

            def feas(a3):
                X = to_full(a3)
                return np.all((X >= sp.lower - 1e-9) & (X <= sp.upper + 1e-9), axis=1)

            from rsstatchem.core.doe import plots

            self.graph.set_png(plots.ternary_contour(ok, lows, free,
                                                     [sp.factors[j].title for j in tri],
                                                     "Overlay Plot (1 = all limits met)", "",
                                                     feas, n_levels=1), "overlay")
            return
        if len(axes) < 2:
            self.graph.set_message("Assign two factors to the X1 and X2 axes.")
            return
        i, j = axes[:2]
        fi, fj = sp.factors[i], sp.factors[j]
        gx, gy = np.meshgrid(np.linspace(fi.low, fi.high, 80), np.linspace(fj.low, fj.high,
                                                                           80))
        X = np.tile(ref, (gx.size, 1))
        X[:, i], X[:, j] = gx.ravel(), gy.ravel()
        limits = []
        for c, lo, hi in active:
            Z = self.fits[c].predict(X).reshape(gx.shape)
            if self.interval_of(c) == "none":
                Zl = Zh = Z
            else:
                bl, bh = self._bounds(c, X, lo, hi)
                Zl, Zh = bl.reshape(gx.shape), bh.reshape(gx.shape)
            label = names[c] + ("" if self.interval_of(c) == "none"
                                else f" [{self.interval_of(c).upper()}]")
            limits.append((label, Z, Zl, Zh, lo, hi))
        self.graph.set_png(plots_opt.overlay_interval_plot(
            gx, gy, limits, fi.axis_label, fj.axis_label, "Overlay Plot",
            solution=(ref[i], ref[j])), "overlay_plot")
        good = np.ones(gx.shape, bool)
        for _, _, Zl, Zh, lo, hi in limits:
            if lo is not None:
                good &= Zl >= lo
            if hi is not None:
                good &= Zh <= hi
        self.feasible_fraction = float(np.mean(good))


# ---------------------------------------------------------------------------- Point Prediction
class PointPredictionView(QWidget):
    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.panel = QFrame()
        self.panel.setObjectName("toolPanel")
        self.panel.setFixedWidth(260)
        self.panel_lay = QVBoxLayout(self.panel)
        self.tool: FactorsTool | None = None
        self.report = ReportView()
        self.message = hint("")
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(6, 6, 6, 6)
        bar = QHBoxLayout()
        bar.addWidget(page_title("Point Prediction"))
        bar.addStretch(1)
        bar.addWidget(QLabel("Intervals:"))
        self.side = QComboBox()
        for key, text in intervals.SIDES:
            self.side.addItem(text, key)
        self.side.currentIndexChanged.connect(self._side_changed)
        bar.addWidget(self.side)
        self.opt_btn = QPushButton("Set to Optimization Solution 1")
        self.opt_btn.clicked.connect(self.use_optimum)
        bar.addWidget(self.opt_btn)
        rl.addLayout(bar)
        rl.addWidget(self.message)
        rl.addWidget(self.report, 1)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.panel)
        lay.addWidget(right, 1)
        self.refresh()

    def refresh(self) -> None:
        self.fits = self.doc.fits()
        if self.tool is not None:
            self.tool.setParent(None)
        self.tool = FactorsTool(self.doc.space(), n_axes=0)
        saved = self.doc.ui.get("point")
        if saved and len(saved) == self.doc.space().k:
            self.tool.set_point(saved)
        self.tool.changed.connect(self.update_prediction)
        _clear(self.panel_lay)
        self.panel_lay.addWidget(_scroll(self.tool))
        self.opt_btn.setEnabled(bool(getattr(self.doc, "last_solutions", None)))
        self.side.blockSignals(True)
        self.side.setCurrentIndex(max(self.side.findData(options(self.doc)["side"]), 0))
        self.side.blockSignals(False)
        self.update_prediction()

    def _side_changed(self) -> None:
        set_option(self.doc, "side", self.side.currentData())
        self.update_prediction()

    def set_side(self, side: str) -> None:
        self.side.setCurrentIndex(max(self.side.findData(side), 0))

    def use_optimum(self) -> None:
        sols = getattr(self.doc, "last_solutions", None)
        if sols:
            self.tool.set_point(sols[0]["x"])

    def set_point(self, x) -> None:
        self.tool.set_point(x)

    def update_prediction(self) -> None:
        if not self.fits:
            self.message.setText(no_models_text(self.doc))
            self.message.show()
            self.report.set_items([])
            return
        self.message.setText("Move the sliders (or type values) in the Factors Tool to "
                             "predict the responses at any setting inside the design space.")
        x = self.tool.point()
        self.doc.ui["point"] = [float(v) for v in x]
        opt = options(self.doc)
        sec = report_post.prediction_section(self.fits, self.doc.space(), x, _names(self.doc),
                                             opt["level"], opt["proportion"],
                                             self.side.currentData() or "two",
                                             poe_sd(self.doc), bool(opt["second_order"]))
        items = [i for i in sec.items if isinstance(i, (Table, Text)) and
                 getattr(i, "style", "") != "log"]
        pct = f"{opt['level'] * 100:g}%"
        items.append(annotation(f"Predicted mean: the expected average response at this "
                                f"setting. {pct} CI: where the true mean lies; {pct} PI: where "
                                f"a single new observation is expected to fall; TI: where "
                                f"{opt['proportion'] * 100:g}% of all future observations are "
                                f"expected with {pct} confidence. POE (when factor standard "
                                "deviations are set under Numerical > Criteria) adds the "
                                "variation transmitted from the factor settings. Use "
                                "Confirmation to compare confirmation runs with the "
                                "prediction."))
        self.report.set_items(items, "Point Prediction")
        self.last = items


# ---------------------------------------------------------------------------- Confirmation
class ConfirmationView(QWidget):
    COLS = ["Response", "Predicted Mean", "Std Dev", "n", "SE Pred", "PI Low", "Data Mean",
            "PI High", "Result", "Individual Runs", "Data Std Dev", "Predicted Median"]

    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.table = QTableWidget(0, len(self.COLS))
        self.table.verticalHeader().setVisible(False)
        self.table.itemChanged.connect(self._edited)
        self.settings = QLabel()
        self.settings.setWordWrap(True)
        self.graph = GraphView()
        self.runs_report = ReportView()
        lower = QSplitter(Qt.Orientation.Horizontal)
        lower.addWidget(self.runs_report)
        lower.addWidget(self.graph)
        lower.setSizes([520, 420])
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.table)
        split.addWidget(lower)
        split.setSizes([180, 380])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(page_title("Confirmation"))
        lay.addWidget(hint("Run the experiment a few times at the setting chosen in Point "
                           "Prediction, then type the individual results (Individual Runs, "
                           "e.g. 62, 63, 64) — or only the number of runs (n) and their "
                           "average (Data Mean). The result is confirmed when the data mean "
                           "falls inside the prediction interval for the mean of n runs; each "
                           "run is also listed against the single-run PI and the TI."))
        lay.addWidget(self.settings)
        lay.addWidget(split, 1)
        self._loading = False
        self.refresh()

    graph_visible = True

    def current_graph(self):
        return self.graph

    def _runs(self) -> dict:
        return self.doc.optimization.get("confirmation_runs", {}) or {}

    def refresh(self) -> None:
        self.fits = self.doc.fits()
        sp = self.doc.space()
        opt = options(self.doc)
        pct = f"{opt['level'] * 100:g}%"
        self.table.setHorizontalHeaderLabels([c.replace("PI", f"{pct} PI") for c in self.COLS])
        x = np.array(self.doc.ui.get("point") or report.default_point(sp), float)
        if len(x) != sp.k:
            x = report.default_point(sp)
        self.x = x
        self.settings.setText("Factor settings: " + "; ".join(
            f"{self.doc.factor_title(j)} = {report._fmt_level(f, x[j])}"
            for j, f in enumerate(sp.factors)))
        conf = self.doc.optimization.setdefault("confirmation", {})
        runs_all = self._runs()
        self._loading = True
        self.table.setRowCount(len(self.fits))
        plot_rows = []
        for r, (col, fit) in enumerate(self.fits.items()):
            runs = report_post.parse_runs(runs_all.get(col))
            if runs:
                n, mean = len(runs), float(np.mean(runs))
            else:
                n, mean = conf.get(col, [1, None])
            row = report_post.confirmation_row(fit, x, int(n or 1), mean, opt["level"],
                                               opt["proportion"])
            name = self.doc.response_title(self.doc.response_index(col))
            result = "" if row["ok"] is None else ("Confirmed (inside PI)" if row["ok"]
                                                   else "Not confirmed")
            vals = [name, num(row["mean"]), num(row["sd"]), str(row["n"]),
                    num(row["se_pred"]), num(row["pi_low"]),
                    "" if mean is None else num(mean), num(row["pi_high"]), result,
                    ", ".join(f"{v:g}" for v in runs),
                    num(np.std(runs, ddof=1)) if len(runs) > 1 else "", num(row["median"])]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                editable = c in (3, 6, 9) and not (runs and c in (3, 6))
                if not editable:
                    it.setFlags(Qt.ItemFlag.ItemIsEnabled)
                else:
                    it.setBackground(QBrush(QColor("#FFFBEA")))
                if c == 0:
                    it.setData(Qt.ItemDataRole.UserRole, col)
                if c == 8 and v:
                    it.setForeground(QBrush(QColor("#1E8C2E" if v.startswith("Confirmed")
                                                   else "#B00020")))
                self.table.setItem(r, c, it)
            plot_rows.append({"name": name, "mean": row["mean"], "pi_low": row["pi_low"],
                              "pi_high": row["pi_high"], "runs": runs, "data_mean": mean,
                              "n": row["n"]})
        self.table.resizeColumnsToContents()
        self._loading = False
        sec = report_post.confirmation_section(self.fits, sp, x, _names(self.doc),
                                               runs=runs_all, summary=conf,
                                               level=opt["level"],
                                               proportion=opt["proportion"])
        items = [t for t in sec.items if isinstance(t, Table) and t.title != "Factor Settings"]
        self.runs_report.set_items(items, "Confirmation Report")
        if plot_rows:
            self.graph.set_png(plots_opt.confirmation_plot(plot_rows), "confirmation")
        else:
            self.graph.set_message(no_models_text(self.doc))

    def set_data(self, col: str, n: int, mean: float) -> None:
        self.doc.optimization.setdefault("confirmation", {})[col] = [int(n), float(mean)]
        self.doc.optimization.get("confirmation_runs", {}).pop(col, None)
        self.doc.touch("optimization")
        self.refresh()

    def set_runs(self, col: str, values) -> None:
        vals = report_post.parse_runs(values)
        runs = self.doc.optimization.setdefault("confirmation_runs", {})
        if vals:
            runs[col] = vals
            self.doc.optimization.setdefault("confirmation", {})[col] = [
                len(vals), float(np.mean(vals))]
        else:
            runs.pop(col, None)
        self.doc.touch("optimization")
        self.refresh()

    def _edited(self, item) -> None:
        if self._loading or item.column() not in (3, 6, 9):
            return
        col = self.table.item(item.row(), 0).data(Qt.ItemDataRole.UserRole)
        if item.column() == 9:
            self.set_runs(col, item.text())
            return
        conf = self.doc.optimization.setdefault("confirmation", {})
        n, mean = conf.get(col, [1, None])
        try:
            v = parse_number(item.text())
        except ValueError:
            v = float("nan")
        if item.column() == 3:
            n = int(v) if math.isfinite(v) and v >= 1 else 1
        else:
            mean = v if math.isfinite(v) else None
        conf[col] = [n, mean]
        self.doc.touch("optimization")
        self.refresh()


# ---------------------------------------------------------------------------- Coefficients
def coefficients_items(doc) -> list:
    fits = doc.fits()
    if not fits:
        return [Text("", no_models_text(doc), style="warning")]
    sec = report_post.coefficients_section(fits, _names(doc))
    return [i for i in sec.items if isinstance(i, (Table, Text))]


class CoefficientsView(QWidget):
    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.report = ReportView()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(self.report)
        self.refresh()

    def refresh(self) -> None:
        self.report.set_items(coefficients_items(self.doc), "Coefficients Table")


# ---------------------------------------------------------------------------- POE graph
class POEGraphView(QWidget):
    """Propagation-of-error graph of one response (contour or 3D) over two factors, with the
    factor standard deviations from ``doc.optimization["poe_sd"]``. Published for the
    workspace to place under the analysis graphs."""

    def __init__(self, doc, response: str | None = None, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.response = response
        panel = QFrame()
        panel.setObjectName("toolPanel")
        panel.setFixedWidth(250)
        pl = QVBoxLayout(panel)
        pl.addWidget(QLabel("Response:"))
        self.resp = QComboBox()
        self.resp.currentIndexChanged.connect(lambda _: self.render())
        pl.addWidget(self.resp)
        pl.addWidget(QLabel("Graph:"))
        self.gtype = QComboBox()
        self.gtype.addItem("Contour", "contour")
        self.gtype.addItem("3D Surface", "3d")
        self.gtype.currentIndexChanged.connect(lambda _: self.render())
        pl.addWidget(self.gtype)
        self.tool_holder = QVBoxLayout()
        pl.addLayout(self.tool_holder, 1)
        self.tool: FactorsTool | None = None
        self.graph = GraphView()
        self.message = hint("")
        right = QVBoxLayout()
        right.addWidget(self.message)
        right.addWidget(self.graph, 1)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(panel)
        lay.addLayout(right, 1)
        self.refresh()

    graph_visible = True

    def current_graph(self):
        return self.graph

    def refresh(self) -> None:
        self.fits = self.doc.fits()
        self.resp.blockSignals(True)
        self.resp.clear()
        for col in self.fits:
            self.resp.addItem(self.doc.response_title(self.doc.response_index(col)), col)
        if self.response is not None:
            self.resp.setCurrentIndex(max(self.resp.findData(self.response), 0))
        self.resp.blockSignals(False)
        if self.tool is not None:
            self.tool.setParent(None)
        self.tool = FactorsTool(self.doc.space(), n_axes=2)
        self.tool.changed.connect(self.render)
        _clear(self.tool_holder)
        self.tool_holder.addWidget(_scroll(self.tool))
        self.render()

    def render(self) -> None:
        if not self.fits:
            self.graph.set_message(no_models_text(self.doc))
            return
        sp = self.doc.space()
        sd = poe_sd(self.doc)
        if not poe.has_sd(sp, sd):
            self.message.setText("No factor standard deviations: enter them under "
                                 "Optimization > Numerical > Criteria (Factor variation). The "
                                 "graph then shows only the residual standard deviation.")
            self.message.show()
        else:
            self.message.hide()
        if sp.mixture:
            self.graph.set_message("The POE graph is available for process factors.")
            return
        axes = self.tool.axes()
        if len(axes) < 2:
            self.graph.set_message("Assign two factors to the X1 and X2 axes.")
            return
        col = self.resp.currentData()
        fit = self.fits[col]
        i, j = axes[:2]
        fi, fj = sp.factors[i], sp.factors[j]
        gx, gy = np.meshgrid(np.linspace(fi.low, fi.high, 60), np.linspace(fj.low, fj.high,
                                                                           60))
        Z = poe.surface(fit, gx, gy, (i, j), self.tool.point(), sd,
                        bool(options(self.doc)["second_order"]))
        self.Z = Z
        fig = plots_opt.poe_figure(gx, gy, Z, fi.axis_label, fj.axis_label,
                                   self.resp.currentText(), self.gtype.currentData())
        self.graph.set_figure(fig, "poe_graph", rotatable=self.gtype.currentData() == "3d")

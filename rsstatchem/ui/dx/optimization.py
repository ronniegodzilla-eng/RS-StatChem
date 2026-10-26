"""Optimization and Post Analysis nodes: Numerical (Criteria / Solutions / Graphs), Graphical
(overlay plot with the yellow feasible window), Point Prediction, Confirmation and the
Coefficients Table. The models are the ones chosen on each response's analysis node."""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QComboBox, QFrame,
                               QHBoxLayout, QHeaderView, QLabel, QPushButton, QScrollArea,
                               QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)
from scipy import stats

from rsstatchem.core.doe import optimize, plots, report
from rsstatchem.core.output import Table, Text
from rsstatchem.ui.dx import style
from rsstatchem.ui.dx.layout import parse_number
from rsstatchem.ui.dx.widgets import (FactorsTool, GraphView, ReportView, TabStrip, annotation,
                                      hint, page_title, tool_button)

num = report.num
GOALS = optimize.GOALS
FACTOR_GOALS = optimize.FACTOR_GOALS
IMPORTANCE = ["+", "++", "+++", "++++", "+++++"]


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
    return doc.response_title(doc.response_index(c.name))


def stored_criteria(doc, fits: dict) -> list[optimize.Criterion]:
    """The remembered criteria, completed with defaults for new factors/responses."""
    space = doc.space()
    saved = {(d["kind"], d["name"]): d for d in doc.optimization.get("numerical", [])}
    out = []
    for f in space.factors:
        if f.categoric:
            continue
        d = saved.get(("factor", f.column))
        out.append(optimize.Criterion(**d) if d else
                   optimize.Criterion(f.column, "factor", "in_range", f.low, f.high))
    for col in doc.response_cols:
        if col not in fits:
            continue
        d = saved.get(("response", col))
        if d:
            out.append(optimize.Criterion(**d))
        else:
            y = fits[col].obs.y
            out.append(optimize.Criterion(col, "response", "none", float(np.min(y)),
                                          float(np.max(y))))
    return out


def save_criteria(doc, criteria: list[optimize.Criterion]) -> None:
    doc.optimization["numerical"] = [c.to_meta() for c in criteria]
    doc.touch("optimization")


# ---------------------------------------------------------------------------- Numerical
class NumericalView(QWidget):
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
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["Name", "Goal", "Lower Limit", "Upper Limit",
                                              "Target", "Lower Weight", "Upper Weight",
                                              "Importance"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.itemChanged.connect(self._edited)
        cl.addWidget(hint("Set a goal for each response (maximize, minimize, target, in range) "
                          "and, if wanted, limits for the factors. Importance (+ to +++++) "
                          "weights a goal in the overall desirability D = (Π dᵢ^rᵢ)^(1/Σrᵢ). "
                          "Then open Solutions."))
        cl.addWidget(self.table, 1)
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
        self.what.blockSignals(True)
        self.what.clear()
        self.what.addItem("Desirability", None)
        for col in self.fits:
            self.what.addItem(self.doc.response_title(self.doc.response_index(col)), col)
        self.what.blockSignals(False)
        if self.tool is not None:
            self.tool.setParent(None)
        self.tool = FactorsTool(self.doc.space(), n_axes=2 if not self.doc.mixture else 3)
        self.tool.changed.connect(self.render_graph)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(self.tool)
        while self.tool_holder.count():
            w = self.tool_holder.takeAt(0).widget()
            if w is not None:
                w.setParent(None)
        self.tool_holder.addWidget(scroll)
        if self.stack.currentIndex() == 1:
            self.solve()

    def _fill_table(self) -> None:
        self._loading = True
        self.table.setRowCount(len(self.criteria))
        for r, c in enumerate(self.criteria):
            name = QTableWidgetItem(_criterion_label(self.doc, c))
            name.setFlags(Qt.ItemFlag.ItemIsEnabled)
            name.setForeground(QBrush(QColor(style.BLUE_DARK if c.kind == "response"
                                             else "#333333")))
            self.table.setItem(r, 0, name)
            goal = QComboBox()
            for key, text in (FACTOR_GOALS if c.kind == "factor" else GOALS):
                goal.addItem(text, key)
            goal.setCurrentIndex(max(goal.findData(c.goal), 0))
            goal.currentIndexChanged.connect(lambda _=0, rr=r: self._goal_changed(rr))
            self.table.setCellWidget(r, 1, goal)
            for col, v in ((2, c.lower), (3, c.upper), (4, c.target if c.target is not None
                                                        else (c.lower + c.upper) / 2),
                           (5, c.w_lower), (6, c.w_upper)):
                it = QTableWidgetItem(num(v, 6))
                it.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(r, col, it)
            imp = QComboBox()
            imp.addItems(IMPORTANCE)
            imp.setCurrentIndex(min(max(c.importance - 1, 0), 4))
            imp.currentIndexChanged.connect(lambda _=0, rr=r: self._imp_changed(rr))
            self.table.setCellWidget(r, 7, imp)
            self._enable_row(r)
        self._loading = False

    def _enable_row(self, r: int) -> None:
        c = self.criteria[r]
        for col in range(2, 7):
            it = self.table.item(r, col)
            on = c.goal != "none" and not (col == 4 and c.goal != "target") and not (
                col in (5, 6) and c.goal == "in_range")
            flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            if on:
                flags |= Qt.ItemFlag.ItemIsEditable
            it.setFlags(flags)
            it.setForeground(QBrush(QColor("#000000" if on else "#A0A7B0")))
        self.table.cellWidget(r, 7).setEnabled(c.active)

    def set_goal(self, name: str, goal: str, lower=None, upper=None, target=None,
                 importance: int | None = None) -> None:
        """Programmatic criterion change (tests, scripts)."""
        for r, c in enumerate(self.criteria):
            if c.name == name:
                c.goal = goal
                if lower is not None:
                    c.lower = float(lower)
                if upper is not None:
                    c.upper = float(upper)
                if target is not None:
                    c.target = float(target)
                if importance is not None:
                    c.importance = int(importance)
        save_criteria(self.doc, self.criteria)
        self._fill_table()

    def _goal_changed(self, r: int) -> None:
        if self._loading:
            return
        self.criteria[r].goal = self.table.cellWidget(r, 1).currentData()
        self._loading = True
        self._enable_row(r)
        self._loading = False
        save_criteria(self.doc, self.criteria)

    def _imp_changed(self, r: int) -> None:
        if self._loading:
            return
        self.criteria[r].importance = self.table.cellWidget(r, 7).currentIndex() + 1
        save_criteria(self.doc, self.criteria)

    def _edited(self, item) -> None:
        if self._loading or item.column() < 2:
            return
        c = self.criteria[item.row()]
        try:
            v = parse_number(item.text())
        except ValueError:
            v = float("nan")
        attr = {2: "lower", 3: "upper", 4: "target", 5: "w_lower", 6: "w_upper"}[item.column()]
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
        active = [c for c in self.criteria if c.kind == "response" and c.goal != "none"]
        if not active:
            self.sol_status.setText("Set a goal for at least one response on the Criteria tab.")
            self.sol_table.setRowCount(0)
            self.sol_graph.set_message("No response goals: set at least one goal (Criteria).")
            self.solutions = []
            return []
        space = self.doc.space()
        self.problem = optimize.Problem(space, self.fits, list(self.criteria))
        design_pts = self.doc.ds.df[self.doc.factor_cols].to_numpy(float)
        try:
            self.solutions = optimize.solve(self.problem, starts=30, seed=0,
                                            design_points=design_pts)
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.sol_status.setText(str(exc))
            return []
        self.doc.last_solutions = self.solutions
        self._fill_solutions()
        return self.solutions

    def _fill_solutions(self) -> None:
        sp = self.doc.space()
        resp = [c.name for c in self.criteria if c.kind == "response"]
        cols = ["Number"] + [self.doc.factor_title(j) for j in range(sp.k)] + \
            [self.doc.response_title(self.doc.response_index(r)) for r in resp] + \
            ["Desirability", ""]
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
                                "(30 random starts + design points, Nelder-Mead).")
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

    def _solution_selected(self) -> None:
        if not self.solutions or self.problem is None:
            return
        s = self.solutions[min(self.selected(), len(self.solutions) - 1)]
        k = next(i for i, b in enumerate(self.view_buttons) if b.isChecked())
        sp = self.problem.space
        names = _names(self.doc)
        if k == 0:
            entries = []
            for c in self.problem.criteria:
                if c.kind == "factor":
                    j = self.doc.factor_cols.index(c.name)
                    entries.append({"name": sp.factors[j].name, "low": c.lower,
                                    "high": c.upper, "value": float(s["x"][j]), "goal": c.goal,
                                    "target": c.target, "kind": "factor"})
                elif c.name in self.fits and c.goal != "none":
                    entries.append({"name": self.doc.responses[self.doc.response_index(
                        c.name)]["name"], "low": c.lower, "high": c.upper,
                        "value": s["pred"][c.name], "goal": c.goal, "target": c.target,
                        "kind": "response"})
            self.sol_stack.setCurrentIndex(0)
            self.sol_graph.set_png(plots.ramps_plot(entries, s["D"], f"Desirability Ramps "
                                   f"(Solution {self.selected() + 1})"), "ramps")
        elif k == 1:
            act = [c for c in self.problem.criteria if c.active]
            labels = [names.get(c.name, c.name) if c.kind == "response"
                      else self.doc.factor_title(self.doc.factor_cols.index(c.name))
                      for c in act]
            self.sol_stack.setCurrentIndex(0)
            self.sol_graph.set_png(plots.desirability_bars(labels, s["d"], s["D"]),
                                   "desirability_bars")
        else:
            sec = report.optimization_section(self.problem, self.solutions, names, ramps=False,
                                              contour=False, overlay=False, bars=False,
                                              starts=30)
            self.sol_stack.setCurrentIndex(1)
            items = [i for i in sec.items if isinstance(i, (Table, Text))]
            self.sol_report.set_items(items, "Optimization Report")
        if self.tool is not None:
            self.tool.set_point(s["x"])

    # ------------------------------------------------------------------ graphs
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
                X = to_full(a3)
                return self.problem.evaluate(X)["D"] if what is None else \
                    self.fits[what].predict(X)

            def feas(a3):
                X = to_full(a3)
                return np.all((X >= sp.lower - 1e-9) & (X <= sp.upper + 1e-9), axis=1)

            title = "Desirability" if what is None else self.what.currentText()
            self.graph.set_png(plots.ternary_contour(
                fn, lows, free, [sp.factors[j].title for j in tri], title, title, feas,
                points=best["x"][tri][None, :] if best is not None else None), "desirability")
            return
        if len(axes) < 2:
            self.graph.set_message("Assign two factors to the X1 and X2 axes.")
            return
        i, j = axes[:2]
        b = self.problem.bounds_actual()
        gx, gy = np.meshgrid(np.linspace(b[i][0], b[i][1], 60), np.linspace(b[j][0], b[j][1],
                                                                            60))
        X = np.tile(ref, (gx.size, 1))
        X[:, i], X[:, j] = gx.ravel(), gy.ravel()
        if what is None:
            Z = self.problem.evaluate(X)["D"].reshape(gx.shape)
            title, zl = "Desirability", "Desirability"
        else:
            Z = self.fits[what].predict(X).reshape(gx.shape)
            title = zl = self.what.currentText()
        sol = (ref[i], ref[j])
        self.graph.set_png(plots.contour_plot(gx, gy, Z, sp.factors[i].axis_label,
                                              sp.factors[j].axis_label, title, solution=sol,
                                              zlabel=zl), "optimization_graph")


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
                          "yellow (the 'sweet spot'); grey areas fail at least one limit."))
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Response", "Lower Limit", "Upper Limit"])
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
        self.table.setRowCount(len(cols))
        lim = self.limits()
        for r, col in enumerate(cols):
            name = QTableWidgetItem(self.doc.response_title(self.doc.response_index(col)))
            name.setFlags(Qt.ItemFlag.ItemIsEnabled)
            name.setData(Qt.ItemDataRole.UserRole, col)
            self.table.setItem(r, 0, name)
            lo, hi = lim.get(col, [None, None])
            for c, v in ((1, lo), (2, hi)):
                self.table.setItem(r, c, QTableWidgetItem("" if v is None else num(v, 6)))
        self._loading = False
        if self.tool is not None:
            self.tool.setParent(None)
        self.tool = FactorsTool(self.doc.space(), n_axes=3 if self.doc.mixture else 2)
        self.tool.changed.connect(self.render)
        while self.panel_lay.count():
            w = self.panel_lay.takeAt(0).widget()
            if w is not None:
                w.setParent(None)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(self.tool)
        self.panel_lay.addWidget(scroll)
        self.render()

    def set_limits(self, col: str, lower=None, upper=None) -> None:
        self.limits()[col] = [lower, upper]
        self.doc.touch("optimization")
        self.refresh()

    def _edited(self, item) -> None:
        if self._loading or item.column() == 0:
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
                    y = self.fits[c].predict(X)
                    if lo is not None:
                        good &= y >= lo
                    if hi is not None:
                        good &= y <= hi
                return good.astype(float)

            def feas(a3):
                X = to_full(a3)
                return np.all((X >= sp.lower - 1e-9) & (X <= sp.upper + 1e-9), axis=1)

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
        limits = [(names[c], self.fits[c].predict(X).reshape(gx.shape), lo, hi)
                  for c, lo, hi in active]
        self.graph.set_png(plots.overlay_plot(gx, gy, limits, fi.axis_label, fj.axis_label,
                                              "Overlay Plot", solution=(ref[i], ref[j])),
                           "overlay_plot")
        self.feasible_fraction = float(np.mean(np.all(
            [((Z >= lo) if lo is not None else True) & ((Z <= hi) if hi is not None else True)
             for _, Z, lo, hi in limits], axis=0)))


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
        while self.panel_lay.count():
            w = self.panel_lay.takeAt(0).widget()
            if w is not None:
                w.setParent(None)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(self.tool)
        self.panel_lay.addWidget(scroll)
        self.opt_btn.setEnabled(bool(getattr(self.doc, "last_solutions", None)))
        self.update_prediction()

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
        sec = report.prediction_section(self.fits, self.doc.space(), x, _names(self.doc))
        items = [i for i in sec.items if isinstance(i, (Table, Text)) and
                 getattr(i, "style", "") != "log"]
        items.append(annotation("Predicted mean: the expected average response at this "
                                "setting. 95% CI: where the true mean lies; 95% PI: where a "
                                "single new observation is expected to fall. Use Confirmation "
                                "to compare confirmation runs with the prediction."))
        self.report.set_items(items, "Point Prediction")
        self.last = items


# ---------------------------------------------------------------------------- Confirmation
class ConfirmationView(QWidget):
    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(["Response", "Predicted Mean", "Std Dev", "n",
                                              "SE Pred", "95% PI Low", "Data Mean",
                                              "95% PI High", "Result"])
        self.table.verticalHeader().setVisible(False)
        self.table.itemChanged.connect(self._edited)
        self.settings = QLabel()
        self.settings.setWordWrap(True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(page_title("Confirmation"))
        lay.addWidget(hint("Run the experiment a few times at the setting chosen in Point "
                           "Prediction, then enter the number of confirmation runs (n) and "
                           "their average (Data Mean). The result is confirmed when the data "
                           "mean falls inside the 95% prediction interval for the mean of n "
                           "runs."))
        lay.addWidget(self.settings)
        lay.addWidget(self.table, 1)
        self._loading = False
        self.refresh()

    def refresh(self) -> None:
        self.fits = self.doc.fits()
        sp = self.doc.space()
        from rsstatchem.core.doe.report import default_point

        x = np.array(self.doc.ui.get("point") or default_point(sp), float)
        if len(x) != sp.k:
            x = default_point(sp)
        self.x = x
        self.settings.setText("Factor settings: " + "; ".join(
            f"{self.doc.factor_title(j)} = {report._fmt_level(f, x[j])}"
            for j, f in enumerate(sp.factors)))
        conf = self.doc.optimization.setdefault("confirmation", {})
        self._loading = True
        self.table.setRowCount(len(self.fits))
        for r, (col, fit) in enumerate(self.fits.items()):
            n, mean = conf.get(col, [1, None])
            row = self._compute(fit, x, int(n or 1), mean)
            name = self.doc.response_title(self.doc.response_index(col))
            vals = [name] + row
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c not in (3, 6):
                    it.setFlags(Qt.ItemFlag.ItemIsEnabled)
                else:
                    it.setBackground(QBrush(QColor("#FFFBEA")))
                if c == 0:
                    it.setData(Qt.ItemDataRole.UserRole, col)
                if c == 8 and v:
                    it.setForeground(QBrush(QColor("#1E8C2E" if v.startswith("Confirmed")
                                                   else "#B00020")))
                self.table.setItem(r, c, it)
        self.table.resizeColumnsToContents()
        self._loading = False

    @staticmethod
    def _compute(fit, x, n: int, mean) -> list[str]:
        iv = fit.interval(x[None, :])
        t = stats.t.ppf(0.975, fit.df_res)
        se_n = math.sqrt(iv["sd"] ** 2 / max(n, 1) + float(iv["se_mean"][0]) ** 2)
        z = float(iv["z"][0])
        a, b = fit.transform.inverse(z - t * se_n), fit.transform.inverse(z + t * se_n)
        lo, hi = float(min(a, b)), float(max(a, b))
        result = ""
        if mean is not None and math.isfinite(mean):
            result = "Confirmed (inside PI)" if lo <= mean <= hi else "Not confirmed"
        return [num(iv["mean"][0]), num(iv["sd"]), str(n), num(se_n), num(lo),
                "" if mean is None else num(mean), num(hi), result]

    def set_data(self, col: str, n: int, mean: float) -> None:
        self.doc.optimization.setdefault("confirmation", {})[col] = [int(n), float(mean)]
        self.doc.touch("optimization")
        self.refresh()

    def _edited(self, item) -> None:
        if self._loading or item.column() not in (3, 6):
            return
        col = self.table.item(item.row(), 0).data(Qt.ItemDataRole.UserRole)
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
    labels: list[str] = []
    data: dict = {}
    for col, fit in fits.items():
        pv = fit.term_p_values()
        by_label = {}
        for t, s in fit.mat.groups:
            for j in range(s.start, s.stop):
                lab = fit.mat.labels[j]
                p = pv.get(t, (None, None))[1] if t else None
                by_label[lab] = (float(fit.beta[j]), p)
                if lab not in labels:
                    labels.append(lab)
        f, p = fit.model_test()
        by_label["Model p-value"] = (None, p)
        data[col] = by_label
    labels.append("Model p-value")
    cols = ["Term"]
    for col in fits:
        t = doc.response_title(doc.response_index(col))
        cols += [t, f"p-value {t.split(':')[0]}"]
    rows = []
    for lab in labels:
        row = [lab]
        for col in fits:
            est, p = data[col].get(lab, (None, None))
            row += ["" if est is None else num(est), report.pval(p) if p is not None else ""]
        rows.append(row)
    return [Table("Coefficients Table (coded factors)", cols, rows, footnotes=[
        "p-value colours: red < 0.01, green 0.01-0.05, blue 0.05-0.10, black ≥ 0.10 "
        "(not significant). Empty: the term is not in that response's model."])]


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

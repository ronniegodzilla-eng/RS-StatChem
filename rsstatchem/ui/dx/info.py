"""Views of the Design branch: design layout, Summary, Graph Columns, Evaluation
and Notes."""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel,
                               QPlainTextEdit, QSplitter, QVBoxLayout, QWidget)

from rsstatchem.core.doe import designs, evaluation, model, plots, report
from rsstatchem.core.output import Table, Text
from rsstatchem.ui.dx.layout import LayoutModel, LayoutView, response_stats
from rsstatchem.ui.dx.widgets import GraphView, ReportView, annotation, hint, page_title

num = report.num


# ---------------------------------------------------------------------------- design layout
class DesignPage(QWidget):
    def __init__(self, doc, undo_stack, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.model = LayoutModel(doc, undo_stack, self)
        self.view = LayoutView(self.model)
        self.hint = hint("")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(self.hint)
        lay.addWidget(self.view, 1)
        self.update_hint()

    def update_hint(self) -> None:
        doc = self.doc
        missing = [doc.response_title(i) for i, c in enumerate(doc.response_cols)
                   if doc.response_count(c) < doc.n_runs - len(doc.ignored)]
        if missing:
            self.hint.setText(
                f"<b>{doc.n_runs} runs.</b> Run the experiments in the <b>Run</b> order and "
                "type the results into the pale yellow response cells (or copy a column in "
                "Excel and paste it here with Ctrl+V). Still incomplete: "
                + ", ".join(missing) + ". Then click the response under <b>Analysis</b>.")
        else:
            self.hint.setText(f"<b>{doc.n_runs} runs, all responses entered.</b> Click a "
                              "response under <b>Analysis</b> in the navigation pane to fit "
                              "a model.")

    def refresh(self) -> None:
        self.model.refresh()
        self.update_hint()


# ---------------------------------------------------------------------------- summary
def summary_items(doc) -> list:
    meta = doc.meta
    items: list = []
    study = {"process": "Response Surface", "mixture": "Mixture"}.get(meta.get("kind"),
                                                                     "Response Surface")
    if meta.get("design") in designs.FACTORIAL_DESIGNS:
        study = "Factorial"
    elif meta.get("design") == "taguchi":
        study = "Taguchi"
    rows = [["File Version", "RS-StatChem .rsx 1"], ["Study Type", study],
            ["Design Type", meta.get("title", "")],
            ["Subtype", "Randomized" if meta.get("randomized") else "Not randomized"],
            ["Runs", str(doc.n_runs)], ["Blocks", f"{doc.n_blocks}" + (
                "" if doc.n_blocks > 1 else " (no blocks)")]]
    if meta.get("center_points"):
        rows.append(["Center Points", str(meta["center_points"])])
    if meta.get("replicates", 1) > 1:
        rows.append(["Replicates", str(meta["replicates"])])
    if "alpha" in meta:
        rows.append(["Alpha", f"{meta['alpha']:.5g} ({meta.get('alpha_type', '')}, "
                              f"{meta.get('ccd_type', '')})"])
    if meta.get("generators"):
        rows.append(["Generators", ", ".join(meta["generators"])])
    if meta.get("defining_relation") and len(meta["defining_relation"]) > 1:
        rows.append(["Defining Relation", " = ".join(meta["defining_relation"])])
    if meta.get("resolution"):
        rows.append(["Resolution", designs.ROMAN.get(meta["resolution"],
                                                     str(meta["resolution"]))])
    if meta.get("model"):
        rows.append(["Design Model", str(meta["model"]).replace("_", " ").title()])
    if meta.get("taguchi"):
        rows.append(["Orthogonal Array", meta["taguchi"]])
    if meta.get("mixture"):
        rows.append(["Mixture Total", num(meta["mixture"]["total"])])
    if meta.get("block_generators") and doc.n_blocks > 1:
        rows.append(["Block Generators", ", ".join(meta["block_generators"])
                     + (" (confounded with whole-plot interactions)"
                        if meta.get("split_plot") else "")])
    if meta.get("min_run"):
        rows.append(["Min-Run Type", meta["min_run"]])
    sp = meta.get("split_plot")
    if sp and doc.split_plot:
        hard = [designs.letter(j) for j in sp.get("hard", [])]
        rows.append(["Split-Plot", f"{sp.get('groups')} groups (whole plots); hard to change: "
                                   + ", ".join(hard)])
        if meta.get("whole_plot_words"):
            rows.append(["Whole-Plot Generators", ", ".join(meta["whole_plot_words"])])
    for fo in meta.get("foldovers", []):
        rows.append(["Augmented", f"{fo.get('label')}: {fo.get('runs')} runs in block "
                                  f"{fo.get('block')}"])
    if doc.ignored:
        rows.append(["Ignored Runs", str(len(doc.ignored))])
    items.append(Table("Build Information", ["", "Value"], rows))

    frows = []
    for j, f in enumerate(doc.factors):
        vals = doc.ds.df[f["column"]].to_numpy(float)
        hard = f.get("changes") == "hard"
        if f.get("type") == "categoric":
            frows.append([designs.letter(j), f["name"], f.get("units", ""),
                          "Categoric" + (" (HTC)" if hard else ""),
                          "Nominal", str(len(f.get("levels", []))),
                          ", ".join(map(str, f.get("labels", []))), "", "", "", ""])
            continue
        typ = "Mixture" if f.get("type") == "mixture" else "Numeric"
        clo = f"-1 ↔ {num(f['low'])}" if typ == "Numeric" else ""
        chi = f"+1 ↔ {num(f['high'])}" if typ == "Numeric" else ""
        frows.append([designs.letter(j), f["name"], f.get("units", ""),
                      typ + (" (HTC)" if hard else ""), "Continuous", num(np.nanmin(vals)),
                      num(np.nanmax(vals)), clo, chi,
                      num(np.nanmean(vals)), num(np.nanstd(vals, ddof=1))])
    items.append(Table("Factors", ["Factor", "Name", "Units", "Type", "SubType", "Minimum",
                                   "Maximum", "Coded Low", "Coded High", "Mean", "Std. Dev."],
                       frows))
    rrows = []
    for i, col in enumerate(doc.response_cols):
        r = doc.responses[i]
        s = response_stats(doc, col)
        spec = doc.spec(col)
        model_txt, analysis = "", "Polynomial" if s["n"] else "(empty)"
        if s["n"] >= 2 and doc.has_spec(col):
            try:
                a = doc.analyze(col)
                model_txt = report.model_label(a.space, a.fit.terms)
            except ValueError:
                model_txt = ""
        ratio = s["max"] / s["min"] if s["n"] and s["min"] > 0 else math.nan
        rrows.append([f"R{i + 1}", r["name"], r.get("units", ""), str(s["n"]), analysis,
                      num(s["min"]), num(s["max"]), num(s["mean"]), num(s["sd"]), num(ratio),
                      spec.transform.describe(), model_txt])
    items.append(Table("Responses", ["Response", "Name", "Units", "Observations", "Analysis",
                                     "Minimum", "Maximum", "Mean", "Std. Dev.", "Ratio",
                                     "Transform", "Model"], rrows,
                       footnotes=["Ratio = maximum / minimum. A ratio greater than 10 usually "
                                  "calls for a transformation (see the Transform tab)."]))
    if meta.get("aliases"):
        items.append(Table("Alias Structure", ["Aliases"], [[a] for a in meta["aliases"]],
                           row_header_cols=0))
    if meta.get("alias_note"):
        items.append(Text("Aliasing", meta["alias_note"]))
    if sp and doc.split_plot:
        items.append(Text("", "HTC = hard-to-change (whole-plot) factor. The runs come in "
                              "groups within which the HTC settings stay fixed; analyse with "
                              "REML (ANOVA (REML) tab)."))
    for n in doc.build_notes:
        items.append(Text("", n))
    return items


class SummaryPage(QWidget):
    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.report = ReportView()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(self.report)
        self.refresh()

    def refresh(self) -> None:
        self.report.set_items(summary_items(self.doc),
                              f"Design Summary: {self.doc.name}")


# ---------------------------------------------------------------------------- graph columns
class GraphColumnsPage(QWidget):
    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.x, self.y, self.c = QComboBox(), QComboBox(), QComboBox()
        bar = QHBoxLayout()
        for lab, box in (("X Axis:", self.x), ("Y Axis:", self.y), ("Color By:", self.c)):
            bar.addWidget(QLabel(lab))
            bar.addWidget(box)
            box.currentIndexChanged.connect(lambda _: self.refresh())
        bar.addStretch(1)
        self.graph = GraphView()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(page_title("Graph Columns"))
        lay.addLayout(bar)
        lay.addWidget(self.graph, 1)
        self.fill()

    def fill(self) -> None:
        cols = ["Run", "Std"] + self.doc.factor_cols + self.doc.response_cols
        for box in (self.x, self.y, self.c):
            box.blockSignals(True)
            cur = box.currentData()
            box.clear()
            if box is self.c:
                box.addItem("None", None)
                box.addItem("Space Type", "Space_Type")
            for col in cols:
                box.addItem(self.doc.column_title(col), col)
            if cur is not None and box.findData(cur) >= 0:
                box.setCurrentIndex(box.findData(cur))
            box.blockSignals(False)
        if self.x.currentData() in (None, "Run") and self.doc.factor_cols:
            self.x.setCurrentIndex(self.x.findData(self.doc.factor_cols[0]))
        if self.y.currentData() in (None, "Run"):
            target = self.doc.response_cols[0] if self.doc.response_cols else "Std"
            self.y.setCurrentIndex(self.y.findData(target))
        self.refresh()

    def refresh(self) -> None:
        xc, yc, cc = self.x.currentData(), self.y.currentData(), self.c.currentData()
        if not xc or not yc:
            return
        df = self.doc.ds.df
        keep = np.ones(len(df), bool)
        keep[list(self.doc.ignored)] = False
        x, y = df[xc].to_numpy(float)[keep], df[yc].to_numpy(float)[keep]
        if not np.any(np.isfinite(x) & np.isfinite(y)):
            self.graph.set_message(f"No data yet for {self.doc.column_title(yc)}: enter the "
                                   "responses in the design layout.")
            return
        color = df[cc].to_numpy(float)[keep] if cc else None
        self.graph.set_png(plots.scatter_plot(x, y, self.doc.column_title(xc),
                                              self.doc.column_title(yc),
                                              color=color,
                                              color_label=self.doc.column_title(cc) if cc
                                              else ""), "graph_columns")


# ---------------------------------------------------------------------------- evaluation
def evaluation_items(doc, order: str, signal_noise: float = 2.0) -> tuple[list, dict]:
    sp = doc.space()
    coded = doc.coded_matrix()
    keep = np.ones(len(coded), bool)
    keep[list(doc.ignored)] = False
    blocks = doc.ds.df["Block"].to_numpy(float)[keep] if doc.n_blocks > 1 else None
    groups = doc.ds.df[doc.group_col].to_numpy(float)[keep] if doc.split_plot else None
    ratio = float((doc.meta.get("split_plot") or {}).get("variance_ratio", 1.0))
    ev = evaluation.evaluate(sp, coded[keep], order, blocks=blocks, signal_noise=signal_noise,
                             groups=groups, variance_ratio=ratio)
    name = next((n for n, o in model.model_orders(sp) if o == order), order)
    items: list = []
    if ev["aliased"]:
        items.append(Text("Aliased Terms", "These terms cannot be estimated with this design "
                          "and are left out of the evaluation: "
                          + ", ".join(model.term_name(t) for t in ev["aliased"]) + ".",
                          style="warning"))
    df = ev["df"]
    rows = []
    if df["Block"]:
        rows.append(["Blocks", str(df["Block"])])
    rows += [["Model", str(df["Model"])], ["Residuals", str(df["Residuals"])],
             ["  Lack of Fit", str(df["Lack of Fit"])], ["  Pure Error", str(df["Pure Error"])],
             ["Corr Total", str(df["Corr Total"])]]
    items.append(Table(f"Degrees of Freedom for Evaluation ({name} model)", ["Source", "df"],
                       rows))
    trows = []
    for r in ev["rows"]:
        p = r["power"]
        trows.append([r["term"], num(r["se"]), str(r["df"]),
                      f"{r['vif']:.3f}" if r["vif"] is not None else "",
                      f"{r['ri2']:.4f}" if r["ri2"] is not None else "",
                      f"{100 * p:.1f} %" if p is not None and math.isfinite(p) else ""])
    items.append(Table("Terms", ["Term", "Standard Error*", "df", "VIF", "Ri²", "Power"],
                       trows, footnotes=[
                           "* Standard errors in units of the (unknown) standard deviation σ.",
                           f"Power at the 5 % alpha level to detect a signal/noise ratio of "
                           f"{signal_noise:g} (an effect of {signal_noise:g} σ)."]))
    notes = []
    vifs = [r["vif"] for r in ev["rows"] if r["vif"]]
    if vifs:
        worst = max(vifs)
        notes.append("Ideal VIF is 1.0; VIFs above 10 are cause for alarm (the coefficients "
                     f"are poorly estimated because of multicollinearity). Largest VIF here: "
                     f"{worst:.3g}.")
    if df["Residuals"] < 3:
        notes.append("Fewer than 3 residual degrees of freedom: the error estimate will be "
                     "poor. Add runs or choose a smaller model.")
    if df["Pure Error"] < 3:
        notes.append("At least 3 pure-error degrees of freedom (replicates) are recommended "
                     "for a reliable lack-of-fit test.")
    if ev.get("split_plot"):
        notes.append(f"Split-plot: standard errors and power use generalised least squares "
                     f"with a group / residual variance ratio of {ratio:g}; whole-plot terms "
                     "are estimated less precisely and have fewer error degrees of freedom.")
    if notes:
        items.append(annotation("\n".join(notes)))
    lev = ev["leverage"]
    order_idx = doc.row_order("run")
    pos = np.flatnonzero(keep)
    where = {int(r): i for i, r in enumerate(pos)}
    lrows = []
    for row in order_idx:
        if row not in where:
            continue
        i = where[row]
        lrows.append([f"{doc.value(row, 'Run'):g}", f"{lev[i]:.3f}",
                      designs.SPACE_TYPES.get(int(doc.value(row, 'Space_Type')), "")])
    p = ev["p"]
    items.append(Table("Leverage", ["Run", "Leverage", "Space Type"], lrows,
                       footnotes=[f"Average leverage = p/n = {p}/{ev['n']} = "
                                  f"{p / max(ev['n'], 1):.4f}. Leverage above twice the "
                                  "average marks an influential run; replicating it lowers "
                                  "its leverage."]))
    return items, ev


class EvaluationPage(QWidget):
    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        sp = doc.space()
        self.order = QComboBox()
        for name, key in model.model_orders(sp):
            if key != "mean":
                self.order.addItem(name, key)
        default = doc.meta.get("model") or ("quadratic" if doc.meta.get("design") in
                                            ("ccd", "bbd", "threelevel", "dopt", "lattice",
                                             "mixdopt", "ev", "centroid") else "2fi")
        i = self.order.findData(default)
        self.order.setCurrentIndex(i if i >= 0 else 0)
        self.sn = QDoubleSpinBox()
        self.sn.setRange(0.1, 10)
        self.sn.setValue(2.0)
        self.sn.setSingleStep(0.5)
        self.order.currentIndexChanged.connect(lambda _: self.refresh())
        self.sn.valueChanged.connect(lambda _: self.refresh())
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Model:"))
        bar.addWidget(self.order)
        bar.addSpacing(16)
        bar.addWidget(QLabel("Signal/noise ratio for power:"))
        bar.addWidget(self.sn)
        bar.addStretch(1)
        self.report = ReportView()
        self.graph = GraphView()
        split = QSplitter()
        split.addWidget(self.report)
        split.addWidget(self.graph)
        split.setSizes([560, 440])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(page_title("Design Evaluation"))
        lay.addWidget(QLabel("How well can this design estimate the chosen model? (No "
                             "response data needed.)"))
        lay.addLayout(bar)
        lay.addWidget(split, 1)
        self.refresh()

    def refresh(self) -> None:
        order = self.order.currentData() or "linear"
        try:
            items, ev = evaluation_items(self.doc, order, self.sn.value())
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.report.set_items([Text("Evaluation", str(exc), style="warning")])
            self.graph.set_message(str(exc))
            return
        self.report.set_items(items, "Evaluation")
        frac, se = evaluation.fds_curve(self.doc.space(), ev)
        if len(frac):
            self.graph.set_png(plots.fds_plot(frac, se, f"FDS Graph ({self.order.currentText()}"
                                                        " model)"), "fds_graph")
        else:
            self.graph.set_message("The FDS graph is not available for this model.")


# ---------------------------------------------------------------------------- notes
class NotesPage(QWidget):
    def __init__(self, doc, parent=None):
        super().__init__(parent)
        self.doc = doc
        self.edit = QPlainTextEdit(doc.notes)
        self.edit.setPlaceholderText("Notes for this design: objective, materials, who ran "
                                     "the experiments, observations during the runs...")
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(400)
        self._timer.timeout.connect(self.commit)
        self.edit.textChanged.connect(self._timer.start)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(page_title(f"Notes for {doc.name}"))
        lay.addWidget(self.edit, 1)

    def commit(self) -> None:
        self.doc.set_notes(self.edit.toPlainText())


def separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet("color:#C9D3DF;")
    return line

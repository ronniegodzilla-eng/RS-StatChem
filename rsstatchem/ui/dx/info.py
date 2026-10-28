"""Views of the Design branch: design layout, Summary, Graph Columns, Evaluation (degrees
of freedom, terms with power for a signal/noise ratio or delta and sigma, matrix measures,
alias matrix, leverage; FDS with a precision goal, standard error of design, correlation and
alias heat maps), Constraints and Notes."""
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
    study = {"process": "Response Surface", "mixture": "Mixture",
             "combined": "Combined (Mixture-Process)"}.get(meta.get("kind"),
                                                           "Response Surface")
    if meta.get("design") == "historical":
        study = "Historical Data" + (" (mixture)" if meta.get("kind") == "mixture" else "")
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
    if meta.get("terms"):
        rows.append(["Model Terms", ", ".join(model.term_name(model.key_term(t))
                                              for t in meta["terms"])])
    if meta.get("criterion"):
        from rsstatchem.core.doe.optimal import CRITERIA, METHODS

        rows.append(["Optimality", CRITERIA.get(meta["criterion"], meta["criterion"])
                     + (f", {METHODS.get(meta.get('exchange'), '')}" if meta.get("exchange")
                        else "")])
    if meta.get("model_points"):
        rows.append(["Model / LOF / Replicate Points",
                     f"{meta['model_points']} / {meta.get('lof_points', 0)} / "
                     f"{meta.get('replicate_points', 0)}"])
    if meta.get("conference_order"):
        rows.append(["Conference Matrix Order", str(meta["conference_order"])
                     + (f" ({meta['fake_factors']} fake factors)" if meta.get("fake_factors")
                        else "")])
    if meta.get("space_filling"):
        rows.append(["Space Filling", meta["space_filling"]])
    for c in meta.get("constraints") or []:
        rows.append(["Constraint", c.get("text", "")])
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
    for au in meta.get("augments", []):
        where = f" in block {au['block']}" if au.get("block") is not None else ""
        rows.append(["Augmented", f"{au.get('label')}{where}"])
    for ch in meta.get("design_space_changes", []):
        rows.append(["Design Space", f"{designs.letter(ch['factor'])} limits set to "
                                     f"{num(ch['low'])} .. {num(ch['high'])}"])
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
        if not np.any(np.isfinite(vals)):
            frows.append([designs.letter(j), f["name"], f.get("units", ""), typ, "Continuous",
                          "", "", f"-1 ↔ {num(f['low'])}" if typ == "Numeric" else "",
                          f"+1 ↔ {num(f['high'])}" if typ == "Numeric" else "", "", ""])
            continue
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
def model_choices(doc) -> list[tuple[str, str]]:
    """(label, key) of the models the Evaluation offers."""
    sp = doc.space()
    out: list[tuple[str, str]] = []
    if doc.combined:
        from rsstatchem.core.doe import terms_combined as tc

        for mname, mkey in tc.MIX_ORDERS:
            for pname, pkey in tc.PROC_ORDERS:
                out.append((f"{mname} x {pname}", f"{mkey}*{pkey}"))
    else:
        out = [(n, k) for n, k in model.model_orders(sp) if k != "mean"]
    if doc.meta.get("terms"):
        out.insert(0, ("Design model (edited)", "design"))
    return out


def default_model_key(doc) -> str:
    meta = doc.meta
    if meta.get("terms"):
        return "design"
    if doc.combined:
        mix = meta.get("mixture") or {}
        return f"{mix.get('model', 'quadratic')}*{mix.get('process_model', 'linear')}"
    if meta.get("design") == "dsd":
        return "linear"
    return meta.get("model") or ("quadratic" if meta.get("design") in
                                 ("ccd", "bbd", "threelevel", "dopt", "lattice", "mixdopt",
                                  "ev", "centroid", "onefactor") else "2fi")


def _evaluation_inputs(doc):
    sp = doc.space()
    coded = doc.coded_matrix()
    keep = np.ones(len(coded), bool)
    keep[list(doc.ignored)] = False
    keep &= np.all(np.isfinite(coded), axis=1)
    for j, f in enumerate(sp.factors):
        if f.categoric:
            keep &= coded[:, j] >= 0
    blocks = doc.ds.df["Block"].to_numpy(float)[keep] if doc.n_blocks > 1 else None
    if blocks is not None and len(np.unique(blocks)) < 2:
        blocks = None
    groups = doc.ds.df[doc.group_col].to_numpy(float)[keep] if doc.split_plot else None
    return sp, coded, keep, blocks, groups


def evaluation_items(doc, order: str, signal_noise: float = 2.0, alpha: float = 0.05,
                     delta: float | None = None, sigma: float | None = None
                     ) -> tuple[list, dict]:
    sp, coded, keep, blocks, groups = _evaluation_inputs(doc)
    ratio = float((doc.meta.get("split_plot") or {}).get("variance_ratio", 1.0))
    if delta is not None and sigma:
        signal_noise = delta / sigma
    terms = doc.model_terms(order)
    if keep.sum() == 0:
        raise ValueError("No runs with complete factor settings to evaluate.")
    ev = evaluation.evaluate(sp, coded[keep], terms=terms, blocks=blocks,
                             signal_noise=signal_noise, alpha=alpha, groups=groups,
                             variance_ratio=ratio)
    ev["order"] = order
    ev["coded"] = coded[keep]
    ev["blocks"] = blocks
    name = next((n for n, o in model_choices(doc) if o == order), order)
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
    if delta is not None and sigma:
        power_note = (f"Power at the {100 * alpha:g} % alpha level to detect a change of "
                      f"{delta:g} with a standard deviation of {sigma:g} (signal/noise "
                      f"{signal_noise:.3g}).")
    else:
        power_note = (f"Power at the {100 * alpha:g} % alpha level to detect a signal/noise "
                      f"ratio of {signal_noise:g} (an effect of {signal_noise:g} σ).")
    items.append(Table("Terms", ["Term", "Standard Error*", "df", "VIF", "Ri²", "Power"],
                       trows, footnotes=[
                           "* Standard errors in units of the (unknown) standard deviation σ.",
                           power_note]))
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
    # matrix measures
    if not ev.get("split_plot") and ev.get("xtx_inv") is not None:
        try:
            mm = evaluation.matrix_measures(sp, coded[keep], ev["terms"], blocks,
                                            doc.constraints)
        except (ValueError, np.linalg.LinAlgError):
            mm = {}
        if mm and not mm.get("singular"):
            mrows = [["Condition Number of Coefficient Matrix", f"{mm['condition']:.4g}"],
                     ["Maximum Leverage", f"{mm['max_leverage']:.4f}"],
                     ["Average Leverage", f"{mm['avg_leverage']:.4f}"],
                     ["Scaled D-optimality |X'X/n|^(1/p)", f"{mm['scaled_d']:.4g}"],
                     ["D-efficiency", f"{mm['d_eff']:.2f} %"],
                     ["A-efficiency", f"{mm['a_eff']:.2f} %"],
                     ["G-efficiency", f"{mm['g_eff']:.2f} %"],
                     ["Trace of (X'X)⁻¹", f"{mm['trace']:.4g}"]]
            if "i_value" in mm:
                mrows.append(["Average Prediction Variance (I)", f"{mm['i_value']:.4g}"])
                mrows.append(["Scaled I (n x average variance)", f"{mm['scaled_i']:.4g}"])
            items.append(Table("Matrix Measures", ["Measure", "Value"], mrows, footnotes=[
                "Efficiencies use the coded (or L-pseudo) model matrix; 100 % D- and "
                "A-efficiency is reached by orthogonal two-level designs. The average "
                "prediction variance is over the design region (constraints included)."]))
            ev["measures"] = mm
    # alias list for designs that are not regular fractions
    if not ev.get("split_plot") and not doc.meta.get("aliases") and not sp.mixture:
        try:
            alist = evaluation.alias_list(sp, coded[keep], ev["terms"], None, blocks)
        except (ValueError, np.linalg.LinAlgError):
            alist = []
        partial = [a for a in alist if "+" in a.split("=", 1)[1] or " - " in a]
        if partial:
            items.append(Table("Alias Matrix (next-order terms)", ["Aliases"],
                               [[a] for a in alist], row_header_cols=0,
                               footnotes=["Expected value of each coefficient if the "
                                          "omitted next-order terms are active."]))
    lev = ev["leverage"]
    order_idx = doc.row_order("run")
    pos = np.flatnonzero(keep)
    where = {int(r): i for i, r in enumerate(pos)}
    lrows = []
    for row in order_idx:
        if row not in where:
            continue
        i = where[row]
        st = doc.value(row, 'Space_Type')
        lrows.append([f"{doc.value(row, 'Run'):g}", f"{lev[i]:.3f}",
                      designs.SPACE_TYPES.get(int(st), "") if math.isfinite(st) else ""])
    p = ev["p"]
    items.append(Table("Leverage", ["Run", "Leverage", "Space Type"], lrows,
                       footnotes=[f"Average leverage = p/n = {p}/{ev['n']} = "
                                  f"{p / max(ev['n'], 1):.4f}. Leverage above twice the "
                                  "average marks an influential run; replicating it lowers "
                                  "its leverage."]))
    return items, ev


EVAL_GRAPHS = [("fds", "FDS Graph"), ("se", "Standard Error of Design"),
               ("corr", "Correlation of Coefficients"), ("pearson", "Correlation (Pearson r)"),
               ("alias", "Alias Matrix")]


class EvaluationPage(QWidget):
    def __init__(self, doc, parent=None):
        super().__init__(parent)
        from PySide6.QtWidgets import QCheckBox

        self.doc = doc
        self.order = QComboBox()
        for name, key in model_choices(doc):
            self.order.addItem(name, key)
        i = self.order.findData(default_model_key(doc))
        self.order.setCurrentIndex(i if i >= 0 else 0)
        self.sn = QDoubleSpinBox()
        self.sn.setRange(0.1, 10)
        self.sn.setValue(2.0)
        self.sn.setSingleStep(0.5)
        self.use_delta = QCheckBox("Delta / sigma:")
        self.delta = QDoubleSpinBox()
        self.delta.setRange(1e-6, 1e9)
        self.delta.setDecimals(4)
        self.delta.setValue(2.0)
        self.sigma = QDoubleSpinBox()
        self.sigma.setRange(1e-6, 1e9)
        self.sigma.setDecimals(4)
        self.sigma.setValue(1.0)
        self.alpha = QComboBox()
        for a in (0.01, 0.05, 0.10):
            self.alpha.addItem(f"{a:g}", a)
        self.alpha.setCurrentIndex(1)
        self.graph_kind = QComboBox()
        for key, text in EVAL_GRAPHS:
            self.graph_kind.addItem(text, key)
        self.fds_type = QComboBox()
        self.fds_type.addItem("Mean", "mean")
        self.fds_type.addItem("Prediction", "prediction")
        self.fds_d = QDoubleSpinBox()
        self.fds_d.setRange(0, 1e9)
        self.fds_d.setDecimals(4)
        self.fds_d.setValue(0.0)
        self.fds_d.setToolTip("Acceptable half-width d of the interval (0 = no goal)")
        self.xf, self.yf = QComboBox(), QComboBox()
        sp = doc.space()
        for j, f in enumerate(sp.factors):
            if not f.categoric:
                self.xf.addItem(doc.factor_title(j), j)
                self.yf.addItem(doc.factor_title(j), j)
        if self.yf.count() > 1:
            self.yf.setCurrentIndex(1)
        for w in (self.order, self.alpha, self.graph_kind, self.fds_type, self.xf, self.yf):
            w.currentIndexChanged.connect(lambda _: self.refresh())
        for w in (self.sn, self.delta, self.sigma, self.fds_d):
            w.valueChanged.connect(lambda _: self.refresh())
        self.use_delta.toggled.connect(lambda _: self.refresh())
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Model:"))
        bar.addWidget(self.order)
        bar.addSpacing(12)
        bar.addWidget(QLabel("Signal/noise ratio for power:"))
        bar.addWidget(self.sn)
        bar.addWidget(self.use_delta)
        bar.addWidget(self.delta)
        bar.addWidget(QLabel("σ"))
        bar.addWidget(self.sigma)
        bar.addWidget(QLabel("α"))
        bar.addWidget(self.alpha)
        bar.addStretch(1)
        gbar = QHBoxLayout()
        gbar.addWidget(QLabel("Graph:"))
        gbar.addWidget(self.graph_kind)
        self.fds_box = QWidget()
        fl = QHBoxLayout(self.fds_box)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.addWidget(QLabel("FDS for:"))
        fl.addWidget(self.fds_type)
        fl.addWidget(QLabel("d:"))
        fl.addWidget(self.fds_d)
        self.se_box = QWidget()
        sl = QHBoxLayout(self.se_box)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(QLabel("X:"))
        sl.addWidget(self.xf)
        sl.addWidget(QLabel("Y:"))
        sl.addWidget(self.yf)
        gbar.addWidget(self.fds_box)
        gbar.addWidget(self.se_box)
        gbar.addStretch(1)
        self.report = ReportView()
        self.graph = GraphView()
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addLayout(gbar)
        rl.addWidget(self.graph, 1)
        split = QSplitter()
        split.addWidget(self.report)
        split.addWidget(right)
        split.setSizes([560, 440])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(page_title("Design Evaluation"))
        lay.addWidget(QLabel("How well can this design estimate the chosen model? (No "
                             "response data needed.)"))
        lay.addLayout(bar)
        lay.addWidget(split, 1)
        self.ev = None
        self.refresh()

    def set_graph(self, key: str) -> None:
        self.graph_kind.setCurrentIndex(self.graph_kind.findData(key))

    def refresh(self) -> None:
        order = self.order.currentData() or "linear"
        use_delta = self.use_delta.isChecked()
        self.sn.setEnabled(not use_delta)
        self.delta.setEnabled(use_delta)
        self.sigma.setEnabled(use_delta)
        kind = self.graph_kind.currentData()
        self.fds_box.setVisible(kind == "fds")
        self.se_box.setVisible(kind == "se")
        try:
            items, ev = evaluation_items(
                self.doc, order, self.sn.value(), self.alpha.currentData() or 0.05,
                self.delta.value() if use_delta else None,
                self.sigma.value() if use_delta else None)
        except (ValueError, np.linalg.LinAlgError, KeyError) as exc:
            self.report.set_items([Text("Evaluation", str(exc), style="warning")])
            self.graph.set_message(str(exc))
            return
        self.ev = ev
        extra = self._graph(ev)
        self.report.set_items(items + extra, "Evaluation")

    def _graph(self, ev) -> list:
        from rsstatchem.core.doe import plots_design

        sp = self.doc.space()
        kind = self.graph_kind.currentData()
        title_model = self.order.currentText()
        if ev.get("xtx_inv") is None:
            self.graph.set_message("The model cannot be estimated with this design.")
            return []
        if kind == "fds":
            ftype = self.fds_type.currentData()
            if self.doc.constraints or getattr(sp, "combined", False):
                se = evaluation.fds_values(sp, ev, kind=ftype, constraints=self.doc.constraints)
            else:
                _, se = evaluation.fds_curve(sp, ev)
                if ftype == "prediction":
                    se = np.sqrt(se ** 2 + 1)
            if not len(se):
                self.graph.set_message("The FDS graph is not available for this model.")
                return []
            d = self.fds_d.value()
            out = []
            limit = frac = None
            if d > 0:
                sigma = self.sigma.value() if self.use_delta.isChecked() else 1.0
                sc = evaluation.fds_score(se, d, sigma, self.alpha.currentData() or 0.05,
                                          ev["df"]["Residuals"])
                limit, frac = sc["limit"], sc["fraction"]
                out.append(Table("FDS Score", ["", "Value"], [
                    ["Half-width d", f"{d:g}"], ["σ", f"{sigma:g}"],
                    ["α", f"{self.alpha.currentData():g}"],
                    ["t value", f"{sc['t']:.4f}"],
                    ["Required SE (σ units)", f"{limit:.4f}"],
                    ["Fraction of design space", f"{100 * frac:.1f} %" if math.isfinite(frac)
                     else ""]],
                    footnotes=["Aim for at least 80 % of the design space with the "
                               "half-width at or below d."]))
            self.graph.set_png(plots_design.fds_plot(
                se, f"FDS Graph ({title_model} model)", limit, frac,
                "Std Error Mean" if ftype == "mean" else "Std Error Prediction"), "fds_graph")
            return out
        if kind == "se":
            i, j = self.xf.currentData(), self.yf.currentData()
            if i is None or j is None or i == j:
                self.graph.set_message("Choose two different numeric factors.")
                return []
            fixed = np.zeros(sp.k)
            if sp.mixture:
                reg = self.doc.region()
                fixed[reg.cont] = reg.centroid()
            G1, G2, se = evaluation.se_grid(sp, ev, i, j, fixed)
            pts = ev["coded"][:, [i, j]]
            self.graph.set_png(plots_design.se_contour(
                G1, G2, se, self.doc.factor_title(i) + " (coded)",
                self.doc.factor_title(j) + " (coded)", pts,
                f"Standard Error of Design ({title_model})"), "se_design")
            return []
        if kind in ("corr", "pearson"):
            c = evaluation.correlations(sp, ev["coded"], ev["terms"], ev.get("blocks"))
            if kind == "corr":
                R, labels, title = c["coef"], c["labels"], "Correlation of Coefficients"
            else:
                R, labels, title = c["r"], c["r_labels"], "Correlation of Model Columns (r)"
            self.graph.set_png(plots_design.heatmap(R, labels, None, title), "correlation")
            return []
        rows, cols, A = evaluation.alias_matrix(sp, ev["coded"], ev["terms"], None,
                                                ev.get("blocks"))
        if not cols:
            self.graph.set_message("No omitted terms to check for aliasing.")
            return []
        self.graph.set_png(plots_design.heatmap(A, rows, cols, "Alias Matrix"), "alias")
        return [Table("Alias Matrix", [""] + cols,
                      [[r] + [f"{v:.3g}" if abs(v) > 1e-9 else "" for v in A[i]]
                       for i, r in enumerate(rows)])]


# ---------------------------------------------------------------------------- constraints
class ConstraintsPage(QWidget):
    """Design > Constraints: the multiple linear constraints of the design region, its
    vertices, runs outside the region and a two-factor view of the region."""

    def __init__(self, doc, parent=None):
        super().__init__(parent)
        from PySide6.QtWidgets import QPushButton

        self.doc = doc
        self.edit_btn = QPushButton("Edit Constraints...")
        self.edit_btn.clicked.connect(self.edit)
        self.xf, self.yf = QComboBox(), QComboBox()
        sp = doc.space()
        for j, f in enumerate(sp.factors):
            if not f.categoric:
                self.xf.addItem(doc.factor_title(j), j)
                self.yf.addItem(doc.factor_title(j), j)
        if self.yf.count() > 1:
            self.yf.setCurrentIndex(1)
        for w in (self.xf, self.yf):
            w.currentIndexChanged.connect(lambda _: self.refresh())
        bar = QHBoxLayout()
        bar.addWidget(self.edit_btn)
        bar.addSpacing(16)
        bar.addWidget(QLabel("Plot X:"))
        bar.addWidget(self.xf)
        bar.addWidget(QLabel("Y:"))
        bar.addWidget(self.yf)
        bar.addStretch(1)
        self.report = ReportView()
        self.graph = GraphView()
        split = QSplitter()
        split.addWidget(self.report)
        split.addWidget(self.graph)
        split.setSizes([520, 480])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.addWidget(page_title("Constraints"))
        lay.addLayout(bar)
        lay.addWidget(split, 1)
        self.dialog = None
        self.refresh()

    def edit(self):
        from rsstatchem.ui.dx.wizard import ConstraintsDialog

        doc = self.doc
        names = [f["name"] for f in doc.factors if not doc.combined or
                 f.get("type") == "mixture"]
        dlg = ConstraintsDialog(names, [c.get("text", "") for c in doc.constraints], self)

        def done(result):
            if result:
                doc.set_constraints(dlg.constraints)
                self.refresh()
        dlg.finished.connect(done)
        dlg.open()
        self.dialog = dlg
        return dlg

    def refresh(self) -> None:
        from rsstatchem.core.doe import constraints as cons
        from rsstatchem.core.doe import plots_design

        doc = self.doc
        items: list = []
        texts = cons.describe(doc.constraints)
        items.append(Table("Multiple Linear Constraints", ["Constraint"],
                           [[t] for t in texts] or [["(none: box limits only)"]],
                           row_header_cols=0,
                           footnotes=["Constraints are in actual units with the factor "
                                      "letters. Edit them, then augment the design (or build "
                                      "a new optimal design) to use the new region."]))
        try:
            reg = doc.region()
            V = reg.vertices()
        except ValueError as exc:
            items.append(Text("Region", str(exc), style="warning"))
            self.report.set_items(items, "Constraints")
            self.graph.set_message(str(exc))
            return
        sp = doc.space()
        cont = reg.cont
        full = np.zeros((len(V), sp.k))
        full[:, cont] = V
        actual = sp.to_actual(full)
        heads = [doc.factor_title(j) for j in cont]
        items.append(Table(f"Vertices of the Design Region ({len(V)})", ["Vertex"] + heads,
                           [[str(i + 1)] + [num(actual[i, j]) for j in cont]
                            for i in range(len(V))]))
        bad = doc.infeasible_runs()
        if bad:
            runs = ", ".join(f"{doc.value(r, 'Run'):g}" for r in bad)
            items.append(Text("Runs Outside the Region", f"Runs {runs} violate the "
                              "constraints.", style="warning"))
        self.report.set_items(items, "Constraints")
        i, j = self.xf.currentData(), self.yf.currentData()
        if i is None or j is None or i == j:
            self.graph.set_message("Choose two different factors.")
            return
        pts = doc.ds.df[[doc.factor_cols[i], doc.factor_cols[j]]].to_numpy(float)
        self.graph.set_png(plots_design.region_points(
            actual[:, [i, j]], pts, (doc.factor_title(i), doc.factor_title(j)),
            "Design Region and Runs"), "constraints")


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

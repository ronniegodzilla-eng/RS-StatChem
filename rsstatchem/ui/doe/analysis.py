"""RSM analysis dialogs (Design-Expert analysis / graphs / optimisation nodes).

Every dialog follows the SPSS analysis protocol (``compute()`` returns a Section). Responses and
factors are pre-filled from ``ds.meta["doe"]``; without a stored design the user picks them and
can set the coding (type and low/high) with "Factor Coding...". The model chosen in
"Fit Summary & Analysis" is remembered in ``ds.meta["doe"]["analysis"]`` and reused by
Diagnostics, Model Graphs, Optimization and Point Prediction.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
                               QHeaderView, QLabel, QListWidget, QListWidgetItem, QPushButton,
                               QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout)

from rsstatchem.core.doe import model, optimize, report
from rsstatchem.core.output import Section
from rsstatchem.ui.dialogs.base import SubDialog, VariablePickerDialog, is_numeric


def _merge(sec: Section, other: Section, keep_log: bool = False) -> Section:
    for it in other.items:
        if getattr(it, "style", None) == "log" and not keep_log:
            continue
        sec.items.append(it)
    return sec


class DoeDialog(VariablePickerDialog):
    """Response / factor targets, pre-filled from the stored design."""
    SINGLE_RESPONSE = False
    RESPONSE_LABEL = "Responses:"

    def __init__(self, ds, title: str, parent=None):
        super().__init__(ds, title, parent)
        self.setMinimumSize(760, 520)
        self.add_target("resp", self.RESPONSE_LABEL, accepts=is_numeric,
                        single=self.SINGLE_RESPONSE)
        self.add_target("factors", "Factors:", accepts=is_numeric, min_items=1)

    def finish(self) -> None:
        self._prefilling = True
        super().finish()
        self._prefill()
        self._prefilling = False
        self.targets_changed()

    def _target(self, key):
        return next(t for t in self._targets if t.key == key)

    def _prefill(self) -> None:
        meta = self.ds.meta.get("doe") or {}
        facs = [f["column"] for f in meta.get("factors", []) if self.ds.has_name(f["column"])]
        resps = [r["column"] for r in meta.get("responses", []) if self.ds.has_name(r["column"])]
        if facs and self.selected("factors") != facs:
            self._move_back(self._target("factors"), all_items=True)
            for name in facs:
                self._take(name, self._target("factors"))
        if resps and not self.selected("resp"):
            for name in resps[:1] if self.SINGLE_RESPONSE else resps:
                self._take(name, self._target("resp"))
        self._update_arrows()

    # ------------------------------------------------------------------ coding
    def add_coding_button(self) -> None:
        self.add_option_button("Factor Coding...", self._coding)

    def space(self) -> model.Space:
        mixture = None if self.ds.meta.get("doe") else bool(self.opts.get("mixture", False))
        return model.space_from_dataset(self.ds, self.selected("factors"),
                                        self.opts.get("coding", {}), mixture)

    def _coding(self) -> None:
        if not self.selected("factors"):
            return
        sp = self.space()
        dlg = CodingDialog(sp, bool(self.opts.get("mixture", sp.mixture)), self,
                           locked=bool(self.ds.meta.get("doe")))
        if dlg.exec():
            self.opts["coding"] = {**self.opts.get("coding", {}), **dlg.values()}
            self.opts["mixture"] = dlg.mixture.isChecked()
            self.targets_changed()

    # ------------------------------------------------------------------ analyses
    def analysis(self, response: str) -> model.Analysis:
        spec = model.stored_spec(self.ds, response)
        factors = self.selected("factors")
        if spec.factors is not None and [f.lower() for f in spec.factors] != \
                [f.lower() for f in factors]:
            spec = model.AnalysisSpec(response, factors, transform=spec.transform)
        spec.factors = factors
        return model.analyze(self.ds, spec, self.space())

    def validate(self) -> str | None:
        clash = set(self.selected("resp")) & set(self.selected("factors"))
        if clash:
            return "A variable cannot be both a response and a factor."
        return None


class CodingDialog(QDialog):
    def __init__(self, space: model.Space, mixture: bool, parent=None, locked: bool = False):
        super().__init__(parent)
        self.setWindowTitle("Factor Coding")
        self.space = space
        lay = QVBoxLayout(self)
        if locked:
            lay.addWidget(QLabel("The coding comes from the stored design; changes here "
                                 "override it for this analysis."))
        self.table = QTableWidget(space.k, 4)
        self.table.setHorizontalHeaderLabels(["Factor", "Type", "Low (-1)", "High (+1)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._types = []
        for r, f in enumerate(space.factors):
            it = QTableWidgetItem(f"{f.letter}: {f.column}")
            it.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(r, 0, it)
            box = QComboBox()
            box.addItems(["Numeric", "Categoric"])
            box.setCurrentIndex(1 if f.categoric else 0)
            self.table.setCellWidget(r, 1, box)
            self._types.append(box)
            self.table.setItem(r, 2, QTableWidgetItem("" if f.categoric else f"{f.low:g}"))
            self.table.setItem(r, 3, QTableWidgetItem("" if f.categoric else f"{f.high:g}"))
        lay.addWidget(self.table)
        self.mixture = QCheckBox("Factors are mixture components (Scheffé models; low/high "
                                 "are component limits)")
        self.mixture.setChecked(mixture)
        lay.addWidget(self.mixture)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("Continue")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.resize(560, 140 + 28 * space.k)

    def values(self) -> dict:
        out = {}
        for r, f in enumerate(self.space.factors):
            d = {"type": "categoric" if self._types[r].currentIndex() == 1 else "numeric"}
            if d["type"] == "numeric":
                try:
                    d["low"] = float(self.table.item(r, 2).text())
                    d["high"] = float(self.table.item(r, 3).text())
                except (ValueError, AttributeError):
                    pass
            else:
                d["levels"], d["labels"] = [], []
            out[f.column] = d
        return out


class TermsDialog(QDialog):
    """Manual model selection: tick the terms of the model."""

    def __init__(self, space: model.Space, current: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Terms")
        self.space = space
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Model terms (hierarchy is not enforced here; use Backward "
                             "selection to reduce a model hierarchically):"))
        self.list = QListWidget()
        cur = set(current)
        for t in model.candidate_terms(space):
            it = QListWidgetItem(model.term_name(t))
            it.setData(Qt.ItemDataRole.UserRole, model.term_key(t))
            it.setCheckState(Qt.CheckState.Checked if t in cur else Qt.CheckState.Unchecked)
            self.list.addItem(it)
        lay.addWidget(self.list)
        row = QHBoxLayout()
        for name, order in model.model_orders(space):
            b = QPushButton(name)
            b.clicked.connect(lambda _=False, o=order: self._set(model.model_terms(space, o)))
            row.addWidget(b)
        lay.addLayout(row)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("Continue")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.resize(360, 520)

    def _set(self, terms) -> None:
        keys = {model.term_key(t) for t in terms}
        for i in range(self.list.count()):
            it = self.list.item(i)
            it.setCheckState(Qt.CheckState.Checked if it.data(Qt.ItemDataRole.UserRole) in keys
                             else Qt.CheckState.Unchecked)

    def keys(self) -> list[str]:
        return [self.list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.list.count())
                if self.list.item(i).checkState() == Qt.CheckState.Checked]


def _combo(items, current, on_change) -> QComboBox:
    box = QComboBox()
    for value, text in items:
        box.addItem(text, value)
    box.setCurrentIndex(max(box.findData(current), 0))
    box.currentIndexChanged.connect(lambda _: on_change(box.currentData()))
    return box


# ---------------------------------------------------------------------------- Fit / ANOVA
class FitAnalysisDialog(DoeDialog):
    SINGLE_RESPONSE = True
    RESPONSE_LABEL = "Response:"
    DEFAULTS = {"model": "suggested", "terms": None, "reduction": "none", "alpha_out": 0.10,
                "hierarchy": True, "transform": "none", "k": 0.0, "lam": 1.0, "level": 95.0,
                "out": ["summary", "anova", "coef", "equations", "effects", "diag_table",
                        "normal", "resid_pred", "resid_run", "pred_actual", "boxcox"]}
    OUTPUTS = [("Fit Summary", [("summary", "Fit summary (sequential SS, lack of fit, model "
                                            "statistics)")]),
               ("ANOVA", [("anova", "ANOVA and fit statistics"),
                          ("coef", "Coefficients in coded terms (SE, CI, VIF)"),
                          ("equations", "Final equations (coded and actual)"),
                          ("effects", "Effects: half-normal and Pareto (2-level designs)")]),
               ("Diagnostics", [("diag_table", "Diagnostics case statistics")]
                + list(report.DIAGNOSTIC_PLOTS))]

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Fit Summary & Analysis", parent)
        self.add_option_button("Model Terms...", self._terms)
        self.add_option_button("Output...", self._outputs)
        self.add_coding_button()
        box = QGroupBox("Model")
        form = QFormLayout(box)
        self.model_box = QComboBox()
        self.model_box.currentIndexChanged.connect(self._model_changed)
        form.addRow("Process order / model:", self.model_box)
        sel = QHBoxLayout()
        self.reduction = _combo([("none", "Manual (no selection)"),
                                 ("backward", "Backward (p-value)")], self.opts["reduction"],
                                lambda v: self.opts.__setitem__("reduction", v))
        self.alpha = QDoubleSpinBox()
        self.alpha.setRange(0.001, 0.5)
        self.alpha.setDecimals(3)
        self.alpha.setSingleStep(0.01)
        self.alpha.setValue(self.opts["alpha_out"])
        self.alpha.valueChanged.connect(lambda v: self.opts.__setitem__("alpha_out", v))
        self.hier = QCheckBox("Maintain hierarchy")
        self.hier.setChecked(self.opts["hierarchy"])
        self.hier.toggled.connect(lambda v: self.opts.__setitem__("hierarchy", v))
        sel.addWidget(self.reduction)
        sel.addWidget(QLabel("Alpha out:"))
        sel.addWidget(self.alpha)
        sel.addWidget(self.hier)
        sel.addStretch()
        form.addRow("Auto selection:", sel)
        tr = QHBoxLayout()
        self.transform = _combo(model.TRANSFORMS, self.opts["transform"], self._transform)
        self.k = QDoubleSpinBox()
        self.k.setRange(-1e9, 1e9)
        self.k.setDecimals(4)
        self.k.setValue(self.opts["k"])
        self.k.valueChanged.connect(lambda v: self.opts.__setitem__("k", v))
        self.lam = QDoubleSpinBox()
        self.lam.setRange(-3, 3)
        self.lam.setDecimals(3)
        self.lam.setSingleStep(0.5)
        self.lam.setValue(self.opts["lam"])
        self.lam.valueChanged.connect(lambda v: self.opts.__setitem__("lam", v))
        tr.addWidget(self.transform)
        tr.addWidget(QLabel("k:"))
        tr.addWidget(self.k)
        tr.addWidget(QLabel("Lambda:"))
        tr.addWidget(self.lam)
        tr.addStretch()
        form.addRow("Transform:", tr)
        self.info = QLabel()
        self.info.setStyleSheet("color: #8E1B1B;")
        form.addRow("", self.info)
        self.options_layout.addWidget(box)
        self.finish()
        self._transform(self.opts["transform"])

    def _transform(self, v) -> None:
        self.opts["transform"] = v
        self.lam.setEnabled(v == "power")

    def targets_changed(self) -> None:
        if not hasattr(self, "model_box") or getattr(self, "_prefilling", False):
            return
        try:
            sp = self.space() if self.selected("factors") else None
        except Exception:  # noqa: BLE001 - e.g. a factor with no data yet
            sp = None
        orders = model.model_orders(sp) if sp else model.PROCESS_ORDERS
        cur = self.opts["model"]
        self.model_box.blockSignals(True)
        self.model_box.clear()
        self.model_box.addItem("Suggested (from Fit Summary)", "suggested")
        for name, order in orders:
            self.model_box.addItem(name, order)
        self.model_box.addItem("Manual (Model Terms...)", "manual")
        self.model_box.setCurrentIndex(max(self.model_box.findData(cur), 0))
        self.model_box.blockSignals(False)
        resp = self.selected_one("resp")
        if resp and sp is not None:
            stored = (self.ds.meta.get("doe") or {}).get("analysis", {}).get(
                self.ds.var(resp).name)
            if stored:
                keys = stored.get("terms") or []
                names = ", ".join(model.term_name(model.key_term(k)) for k in keys[:8])
                self.info.setText(f"Stored model for {resp}: {len(keys)} terms ({names}"
                                  f"{'…' if len(keys) > 8 else ''}).")
            else:
                self.info.setText("")

    def _model_changed(self, _=None) -> None:
        v = self.model_box.currentData()
        if v:
            self.opts["model"] = v

    def _terms(self) -> None:
        if not self.selected("factors"):
            return
        sp = self.space()
        if self.opts.get("terms"):
            cur = [model.key_term(k) for k in self.opts["terms"]]
        else:
            order = self.opts["model"] if self.opts["model"] not in ("suggested", "manual") \
                else ("quadratic" if not sp.mixture else "quadratic")
            cur = model.model_terms(sp, order)
        dlg = TermsDialog(sp, cur, self)
        if dlg.exec():
            self.opts["terms"] = dlg.keys()
            self.opts["model"] = "manual"
            self.model_box.setCurrentIndex(self.model_box.findData("manual"))

    def _outputs(self) -> None:
        dlg = SubDialog("Fit Summary & Analysis: Output", self)
        for title, items in self.OUTPUTS:
            box = dlg.group(title)
            for key, text in items:
                dlg.check(box, key, text, key in self.opts["out"])
        dlg.number(dlg.group("Confidence Intervals"), "level", "Level (%):",
                   self.opts["level"], 50, 99.9, 1)
        values = dlg.run()
        if values is None:
            return
        self.opts["level"] = values.pop("level")
        self.opts["out"] = [k for k, v in values.items() if v]

    def validate(self) -> str | None:
        err = super().validate()
        if err:
            return err
        if self.opts["model"] == "manual" and self.opts.get("terms") is None:
            return "Choose the model terms with Model Terms..."
        return None

    def compute(self) -> Section:
        o = self.opts
        resp = self.selected_one("resp")
        sp = self.space()
        tr = model.Transform(o["transform"], o["k"], o["lam"])
        spec = model.AnalysisSpec(resp, self.selected("factors"), o["model"],
                                  o.get("terms") if o["model"] == "manual" else None,
                                  o["reduction"], o["alpha_out"], o["hierarchy"], tr)
        notes = []
        if o["transform"] == "boxcox":
            base = model.analyze(self.ds, model.AnalysisSpec(
                resp, spec.factors, spec.model, spec.terms), sp)
            kk = o["k"]
            if np.any(base.obs.y + kk <= 0):
                kk = float(-np.min(base.obs.y) + 0.01 * (np.ptp(base.obs.y) or 1))
            bc = model.box_cox(sp, base.obs, base.fit.terms, kk)
            spec.transform = model.Transform("power", kk, float(bc["recommended"]))
            if bc["recommended"] == 1:
                spec.transform = model.Transform()
            notes.append(f"Box-Cox: best lambda {bc['best']:.3g}, recommended "
                         f"{bc['recommended_name']} (lambda = {bc['recommended']:g}).")
        a = model.analyze(self.ds, spec, sp)
        model.save_spec(self.ds, spec, a.fit)
        out = set(o["out"])
        sec = report.anova_section(a, "summary" in out, "anova" in out, "coef" in out,
                                   "equations" in out, o["level"] / 100)
        for n in notes:
            sec.note("Transform", n)
        if "effects" in out and model.is_two_level(a.space, a.obs):
            _merge(sec, report.effects_section(a))
        plots_wanted = [k for k, _ in report.DIAGNOSTIC_PLOTS if k in out]
        if plots_wanted or "diag_table" in out:
            _merge(sec, report.diagnostics_section(a, plots_wanted, "diag_table" in out))
        return sec


class DiagnosticsDialog(DoeDialog):
    DEFAULTS = {"plots": ["normal", "resid_pred", "resid_run", "pred_actual", "boxcox",
                          "cooks", "leverage", "dffits"], "table": True}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Diagnostics", parent)
        self.add_coding_button()
        box = QGroupBox("Plots")
        grid = QGridLayout(box)
        for i, (key, text) in enumerate(report.DIAGNOSTIC_PLOTS):
            cb = QCheckBox(text)
            cb.setChecked(key in self.opts["plots"])
            cb.toggled.connect(lambda on, k=key: self._toggle(k, on))
            grid.addWidget(cb, i // 3, i % 3)
        tb = QCheckBox("Diagnostics case statistics table")
        tb.setChecked(self.opts["table"])
        tb.toggled.connect(lambda v: self.opts.__setitem__("table", v))
        grid.addWidget(tb, 3, 0, 1, 3)
        self.options_layout.addWidget(box)
        self.options_layout.addWidget(QLabel("Uses the model stored by Fit Summary & Analysis "
                                             "(else the suggested model)."))
        self.finish()

    def _toggle(self, key, on) -> None:
        s = [k for k in self.opts["plots"] if k != key]
        if on:
            s.append(key)
        self.opts["plots"] = s

    def compute(self) -> Section:
        sec = Section("Diagnostics")
        sec.log("DOE DIAGNOSTICS RESPONSES=" + " ".join(self.selected("resp")))
        plots_wanted = [k for k, _ in report.DIAGNOSTIC_PLOTS if k in self.opts["plots"]]
        for resp in self.selected("resp"):
            a = self.analysis(resp)
            sub = report.diagnostics_section(a, plots_wanted, self.opts["table"])
            sec.note("", f"Response: {a.fit.response_name} — "
                         f"{report.model_label(a.space, a.fit.terms)} model")
            _merge(sec, sub)
        return sec


class LevelsTable(QTableWidget):
    """Factor settings (numeric spin boxes / categoric combo boxes) in actual units."""

    def __init__(self, parent=None):
        super().__init__(0, 2, parent)
        self.setHorizontalHeaderLabels(["Factor", "Setting (actual units)"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(26)
        self._cache: dict[str, float] = {}
        self.space = None

    def set_space(self, space: model.Space, ref: np.ndarray | None = None) -> None:
        for r in range(self.rowCount()):
            if self.space is not None and r < self.space.k:
                self._cache[self.space.factors[r].column] = self._value(r)
        self.space = space
        ref = report.default_point(space) if ref is None else ref
        self.setRowCount(space.k)
        for r, f in enumerate(space.factors):
            it = QTableWidgetItem(f.title + (f" ({f.units})" if f.units else ""))
            it.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.setItem(r, 0, it)
            v = self._cache.get(f.column, ref[r])
            if f.categoric:
                box = QComboBox()
                for i, lv in enumerate(f.levels):
                    box.addItem(f.level_label(i), lv)
                box.setCurrentIndex(max(box.findData(v), 0))
                self.setCellWidget(r, 1, box)
            else:
                spin = QDoubleSpinBox()
                span = max(f.high - f.low, 1e-9)
                spin.setRange(f.low - 10 * span, f.high + 10 * span)
                spin.setDecimals(4)
                spin.setSingleStep(span / 20)
                spin.setValue(float(v))
                self.setCellWidget(r, 1, spin)
        self.setMinimumHeight(min(60 + 26 * space.k, 220))

    def _value(self, r: int) -> float:
        w = self.cellWidget(r, 1)
        if isinstance(w, QComboBox):
            return float(w.currentData())
        return float(w.value()) if w is not None else 0.0

    def values(self) -> np.ndarray:
        return np.array([self._value(r) for r in range(self.rowCount())])


class ModelGraphsDialog(DoeDialog):
    DEFAULTS = {"graph": "contour", "points": True, "levels": 10, "full_range": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Model Graphs", parent)
        self.setMinimumSize(780, 640)
        self.add_coding_button()
        box = QGroupBox("Graph")
        form = QFormLayout(box)
        self.graph = _combo(report.GRAPHS, self.opts["graph"],
                            lambda v: (self.opts.__setitem__("graph", v), self._axes_enable()))
        form.addRow("Graph:", self.graph)
        axes = QHBoxLayout()
        self.axes = [QComboBox() for _ in range(3)]
        for i, (lab, cb) in enumerate(zip(("X1:", "X2:", "X3:"), self.axes)):
            axes.addWidget(QLabel(lab))
            axes.addWidget(cb, 1)
        form.addRow("Axes:", axes)
        self.levels_table = LevelsTable()
        form.addRow("Other factors at:", self.levels_table)
        opts = QHBoxLayout()
        pts = QCheckBox("Show design points")
        pts.setChecked(self.opts["points"])
        pts.toggled.connect(lambda v: self.opts.__setitem__("points", v))
        full = QCheckBox("Extend axes to all design points (axial)")
        full.setChecked(self.opts["full_range"])
        full.toggled.connect(lambda v: self.opts.__setitem__("full_range", v))
        nl = QSpinBox()
        nl.setRange(2, 30)
        nl.setValue(self.opts["levels"])
        nl.valueChanged.connect(lambda v: self.opts.__setitem__("levels", v))
        opts.addWidget(pts)
        opts.addWidget(full)
        opts.addWidget(QLabel("Contour levels:"))
        opts.addWidget(nl)
        opts.addStretch()
        form.addRow("", opts)
        self.options_layout.addWidget(box)
        self.finish()

    def targets_changed(self) -> None:
        if not hasattr(self, "levels_table") or getattr(self, "_prefilling", False):
            return
        factors = self.selected("factors")
        if not factors:
            return
        try:
            sp = self.space()
        except Exception:  # noqa: BLE001
            return
        for i, cb in enumerate(self.axes):
            prev = cb.currentData()
            cb.blockSignals(True)
            cb.clear()
            for j, f in enumerate(sp.factors):
                cb.addItem(f.title, j)
            want = prev if prev is not None and prev < sp.k else min(i, sp.k - 1)
            cb.setCurrentIndex(cb.findData(want))
            cb.blockSignals(False)
        self.levels_table.set_space(sp)
        self._axes_enable()

    def _axes_enable(self) -> None:
        g = self.opts["graph"]
        n = {"contour": 2, "surface": 2, "interaction": 2, "one_factor": 1, "cube": 3,
             "ternary": 3, "perturbation": 0, "trace": 0}.get(g, 2)
        for i, cb in enumerate(self.axes):
            cb.setEnabled(i < n)

    def validate(self) -> str | None:
        err = super().validate()
        if err:
            return err
        g = self.opts["graph"]
        n = {"contour": 2, "surface": 2, "interaction": 2, "one_factor": 1, "cube": 3,
             "ternary": 3}.get(g, 0)
        axes = [cb.currentData() for cb in self.axes[:n]]
        if len(set(axes)) < len(axes):
            return "Choose different factors for the axes."
        return None

    def compute(self) -> Section:
        g = self.opts["graph"]
        axes = [cb.currentData() for cb in self.axes if cb.currentData() is not None]
        sec = Section("Model Graphs")
        sec.log(f"DOE GRAPH TYPE={g.upper()} RESPONSES=" + " ".join(self.selected("resp")))
        ref = self.levels_table.values()
        for resp in self.selected("resp"):
            a = self.analysis(resp)
            _merge(sec, report.graph_section(a, g, axes, ref, self.opts["points"],
                                             n_levels=self.opts["levels"],
                                             full_range=self.opts["full_range"]))
        return sec


GOAL_ITEMS = optimize.GOALS


class OptimizationDialog(DoeDialog):
    DEFAULTS = {"starts": 30, "seed": 1, "solutions": 10, "ramps": True, "bars": True,
                "contour": True, "overlay": True}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Numerical Optimization", parent)
        self.setMinimumSize(900, 680)
        self.add_coding_button()
        box = QGroupBox("Criteria")
        lay = QVBoxLayout(box)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["Name", "Goal", "Lower Limit", "Upper Limit",
                                              "Target", "Lower Weight", "Upper Weight",
                                              "Importance"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(26)
        self.table.setMinimumHeight(170)
        lay.addWidget(self.table)
        row = QHBoxLayout()
        for key, text, lo, hi in (("starts", "Random starts:", 1, 500),
                                  ("seed", "Seed:", 0, 999999),
                                  ("solutions", "Solutions shown:", 1, 100)):
            row.addWidget(QLabel(text))
            s = QSpinBox()
            s.setRange(lo, hi)
            s.setValue(self.opts[key])
            s.valueChanged.connect(lambda v, k=key: self.opts.__setitem__(k, v))
            row.addWidget(s)
        row.addStretch()
        lay.addLayout(row)
        row = QHBoxLayout()
        for key, text in (("ramps", "Ramps"), ("bars", "Desirability bars"),
                          ("contour", "Desirability contour"),
                          ("overlay", "Overlay plot (graphical)")):
            cb = QCheckBox(text)
            cb.setChecked(self.opts[key])
            cb.toggled.connect(lambda v, k=key: self.opts.__setitem__(k, v))
            row.addWidget(cb)
        row.addWidget(QLabel("  Plot axes:"))
        self.axes = [QComboBox() for _ in range(3)]
        for cb in self.axes:
            row.addWidget(cb)
        row.addStretch()
        lay.addLayout(row)
        self.options_layout.addWidget(box)
        self._cache: dict[str, list] = {}
        self.finish()

    def _row_values(self, r: int) -> list:
        goal = self.table.cellWidget(r, 1).currentData()
        vals = [self.table.item(r, c).text() if self.table.item(r, c) else ""
                for c in range(2, 7)]
        imp = self.table.cellWidget(r, 7).value()
        return [goal] + vals + [imp]

    def targets_changed(self) -> None:
        if not hasattr(self, "table") or getattr(self, "_prefilling", False):
            return
        for r in range(self.table.rowCount()):
            key = self.table.item(r, 0).data(Qt.ItemDataRole.UserRole)
            self._cache[key] = self._row_values(r)
        if not self.selected("factors"):
            return
        try:
            sp = self.space()
        except Exception:  # noqa: BLE001
            return
        stored = {(c["kind"], c["name"]): c for c in
                  (self.ds.meta.get("doe") or {}).get("optimization", [])}
        rows = []
        for f in sp.factors:
            if f.categoric:
                continue
            rows.append(("factor", f.column, f.title, ["in_range", f.low, f.high, "", 1, 1, 3]))
        from rsstatchem.core.stats.common import numeric
        for resp in self.selected("resp"):
            y = numeric(self.ds, resp).dropna()
            lo, hi = (float(y.min()), float(y.max())) if len(y) else (0.0, 1.0)
            rows.append(("response", resp, model.response_name(self.ds, resp),
                         ["maximize", lo, hi, "", 1, 1, 3]))
        self.table.setRowCount(len(rows))
        for r, (kind, col, title, default) in enumerate(rows):
            key = f"{kind}:{col}"
            vals = self._cache.get(key)
            if vals is None and (kind, col) in stored:
                c = stored[(kind, col)]
                vals = [c["goal"], c["lower"], c["upper"], c.get("target") or "",
                        c["w_lower"], c["w_upper"], c["importance"]]
            vals = vals or default
            it = QTableWidgetItem(title)
            it.setData(Qt.ItemDataRole.UserRole, key)
            it.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(r, 0, it)
            items = optimize.FACTOR_GOALS if kind == "factor" else GOAL_ITEMS
            box = QComboBox()
            for v, t in items:
                box.addItem(t, v)
            box.setCurrentIndex(max(box.findData(vals[0]), 0))
            self.table.setCellWidget(r, 1, box)
            for c, v in enumerate(vals[1:6]):
                self.table.setItem(r, c + 2, QTableWidgetItem(
                    v if isinstance(v, str) else f"{float(v):.6g}"))
            imp = QSpinBox()
            imp.setRange(1, 5)
            imp.setValue(int(vals[6]))
            self.table.setCellWidget(r, 7, imp)
        for cb in self.axes:
            prev = cb.currentData()
            cb.clear()
            for j, f in enumerate(sp.factors):
                if not f.categoric:
                    cb.addItem(f.letter, j)
            if prev is not None:
                cb.setCurrentIndex(max(cb.findData(prev), 0))
        for i, cb in enumerate(self.axes):
            if cb.count() > i and cb.currentIndex() == 0 and i:
                cb.setCurrentIndex(i)

    def criteria(self) -> list[optimize.Criterion]:
        out = []
        for r in range(self.table.rowCount()):
            kind, col = self.table.item(r, 0).data(Qt.ItemDataRole.UserRole).split(":", 1)
            goal, lo, hi, tgt, wl, wu, imp = self._row_values(r)

            def f(s, default=None):
                try:
                    return float(str(s).replace(",", "."))
                except ValueError:
                    return default
            lo_v, hi_v = f(lo, 0.0), f(hi, 1.0)
            if hi_v < lo_v:
                raise ValueError(f"{self.table.item(r, 0).text()}: the upper limit is below "
                                 "the lower limit.")
            out.append(optimize.Criterion(col, kind, goal, lo_v, hi_v, f(tgt),
                                          f(wl, 1.0) or 1.0, f(wu, 1.0) or 1.0, int(imp)))
        return out

    def validate(self) -> str | None:
        err = super().validate()
        if err:
            return err
        try:
            crit = self.criteria()
        except ValueError as exc:
            return str(exc)
        if not any(c.active for c in crit):
            return "Set a goal (maximize, minimize, target or in range) for at least one " \
                   "response."
        return None

    def compute(self) -> Section:
        crit = self.criteria()
        sp = self.space()
        fits, names, analyses = {}, {}, {}
        for resp in self.selected("resp"):
            a = self.analysis(resp)
            fits[resp] = a.fit
            names[resp] = a.fit.response_name
            analyses[resp] = a
        prob = optimize.Problem(sp, fits, crit)
        first = next(iter(analyses.values()))
        sols = optimize.solve(prob, self.opts["starts"], self.opts["seed"],
                              self.opts["solutions"], design_points=first.obs.actual)
        self.ds.meta.setdefault("doe", {})["optimization"] = [c.to_meta() for c in crit]
        axes = [cb.currentData() for cb in self.axes if cb.currentData() is not None]
        sec = report.optimization_section(prob, sols, names, axes, self.opts["ramps"],
                                          self.opts["contour"], self.opts["overlay"],
                                          self.opts["bars"], self.opts["seed"],
                                          self.opts["starts"])
        return sec


class PointPredictionDialog(DoeDialog):
    DEFAULTS = {"level": 95.0}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Point Prediction", parent)
        self.add_coding_button()
        box = QGroupBox("Factor Settings")
        lay = QVBoxLayout(box)
        self.levels_table = LevelsTable()
        lay.addWidget(self.levels_table)
        row = QHBoxLayout()
        row.addWidget(QLabel("Confidence level (%):"))
        lv = QDoubleSpinBox()
        lv.setRange(50, 99.9)
        lv.setValue(self.opts["level"])
        lv.valueChanged.connect(lambda v: self.opts.__setitem__("level", v))
        row.addWidget(lv)
        row.addStretch()
        lay.addLayout(row)
        self.options_layout.addWidget(box)
        self.finish()

    def targets_changed(self) -> None:
        if not hasattr(self, "levels_table") or getattr(self, "_prefilling", False):
            return
        if self.selected("factors"):
            try:
                self.levels_table.set_space(self.space())
            except Exception:  # noqa: BLE001
                pass

    def compute(self) -> Section:
        sp = self.space()
        x = self.levels_table.values()
        if sp.mixture and abs(x.sum() - sp.total) > 1e-6 * max(1, sp.total):
            raise ValueError(f"The component settings must add up to the mixture total "
                             f"{sp.total:g} (they add up to {x.sum():g}).")
        fits, names = {}, {}
        for resp in self.selected("resp"):
            a = self.analysis(resp)
            fits[resp] = a.fit
            names[resp] = a.fit.response_name
        return report.prediction_section(fits, sp, x, names, self.opts["level"] / 100)


class TaguchiDialog(DoeDialog):
    RESPONSE_LABEL = "Responses (replicates):"
    DEFAULTS = {"sn": "larger", "means": True, "anova": True}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Taguchi Signal-to-Noise Analysis", parent)
        box = QGroupBox("Signal-to-Noise Ratio")
        form = QFormLayout(box)
        form.addRow("Type:", _combo(optimize.SN_TYPES, self.opts["sn"],
                                    lambda v: self.opts.__setitem__("sn", v)))
        for key, text in (("means", "Response table and main effects plot for means"),
                          ("anova", "Analysis of variance for S/N ratios")):
            cb = QCheckBox(text)
            cb.setChecked(self.opts[key])
            cb.toggled.connect(lambda v, k=key: self.opts.__setitem__(k, v))
            form.addRow("", cb)
        self.options_layout.addWidget(box)
        self.finish()

    def compute(self) -> Section:
        return report.taguchi_section(self.ds, self.selected("factors"), self.selected("resp"),
                                      self.opts["sn"], self.opts["means"], self.opts["anova"])


def show_design_summary(editor) -> None:
    from PySide6.QtWidgets import QMessageBox

    meta = editor.ds.meta.get("doe")
    if not meta or "design" not in meta:
        QMessageBox.information(editor, "Design Summary",
                                "This data set has no stored design. Create one with RSM > "
                                "Create Design.")
        return
    editor.ctx.show_output(report.design_section(meta, [], editor.ds))

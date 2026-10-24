"""Analyze > Structural Equation Modeling > CB-SEM… (covariance-based SEM / CFA, mirroring
AMOS and R lavaan).

The dialog has a Model Builder tab (latent constructs with their indicators, structural paths,
covariances and templates: CFA, full SEM, observed path analysis, simple mediation) that
writes lavaan syntax into the Syntax tab, where the model can be edited freely with live
validation. Estimation / Output / Bootstrap / Save sub-dialogs follow the SPSS pattern. The
analysis runs in :mod:`rsstatchem.core.stats.sem`.
"""
from __future__ import annotations

import copy
import re

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QDialogButtonBox,
                               QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QListWidget, QMessageBox, QPlainTextEdit,
                               QPushButton, QTableWidget, QTableWidgetItem, QTabWidget,
                               QVBoxLayout, QWidget)

from rsstatchem.core.dataset import NUMERIC, Dataset
from rsstatchem.core.stats import sem as sem_core
from rsstatchem.core.stats import sem_syntax
from rsstatchem.ui.dialogs.base import NAME_ROLE, SubDialog, variable_item

TEMPLATES = [("cfa", "CFA (all constructs correlated)"),
             ("sem", "Full SEM (constructs + structural paths)"),
             ("path", "Path analysis (observed variables)"),
             ("mediation", "Simple mediation (X → M → Y)")]

HELP = ("lavaan syntax:  F =~ x1 + x2 + x3  (loadings)   Y ~ X + M  (regression)   "
        "x1 ~~ x2  (covariance)   1*x fix · NA*x free · a*x label   ind := a*b  (defined)")


class CBSEMDialog(QDialog):
    """CB-SEM dialog. ``compute()`` returns a :class:`sem_core.SEMResult` (.section, .saved)."""

    modifies_data = False
    DEFAULTS = {"estimator": "ML", "identification": "marker", "likelihood": "normal",
                "information": "expected", "baseline": "lavaan", "auto_cov_y": True,
                "cov_exogenous": True, "boot": False, "boot_n": 2000, "seed": 12345,
                "level": 95.0, "save_scores": False, "mi_min": 3.84, "mi_top": 20,
                "diagram_std": True, **{f"out_{k}": v for k, v in sem_core.OUTPUTS.items()}}
    _state: dict = {}

    def __init__(self, ds: Dataset, parent: QWidget | None = None):
        super().__init__(parent)
        self.ds = ds
        self.setWindowTitle("CB-SEM (Covariance-Based Structural Equation Modeling)")
        self.setMinimumSize(980, 640)
        state = copy.deepcopy(self._state.get(self._key(), {}))
        self.opts = copy.deepcopy(self.DEFAULTS)
        self.opts.update(state.get("opts", {}))

        self.source = QListWidget()
        self.source.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.source.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        for var in ds.variables:
            if var.type == NUMERIC:
                self.source.addItem(variable_item(var))
        self.source.itemDoubleClicked.connect(self._insert_variable)
        left = QVBoxLayout()
        left.addWidget(QLabel("Variables:"))
        left.addWidget(self.source, 1)
        hint = QLabel("Select items, then add them as a construct. Double-click inserts a "
                      "name into the syntax.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#666")
        left.addWidget(hint)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._builder_tab(), "Model Builder")
        self.tabs.addTab(self._syntax_tab(), "Syntax")

        buttons_col = QVBoxLayout()
        buttons_col.setAlignment(Qt.AlignmentFlag.AlignTop)
        for text, fn in (("Estimation...", self._estimation), ("Output...", self._output),
                         ("Bootstrap...", self._bootstrap), ("Save...", self._save)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            buttons_col.addWidget(b)

        body = QHBoxLayout()
        body.addLayout(left, 2)
        body.addWidget(self.tabs, 6)
        body.addLayout(buttons_col)

        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        box = QDialogButtonBox()
        self.ok_button = box.addButton(QDialogButtonBox.StandardButton.Ok)
        reset = box.addButton(QDialogButtonBox.StandardButton.Reset)
        box.addButton(QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        reset.clicked.connect(self.reset)

        layout = QVBoxLayout(self)
        layout.addLayout(body, 1)
        layout.addWidget(self.status)
        layout.addWidget(box)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(350)
        self._timer.timeout.connect(self._validate_live)
        self._restore(state)
        self.editor.textChanged.connect(self._timer.start)
        self._validate_live()

    # ------------------------------------------------------------------ state
    def _key(self) -> str:
        return type(self).__name__

    def _restore(self, state: dict) -> None:
        names = set(self.ds.names())
        for name, items in state.get("constructs", []):
            items = [x for x in items if x in names]
            if items:
                self._add_construct_row(name, items)
        for a, b in state.get("paths", []):
            self.paths.addItem(f"{a} → {b}")
        for a, b in state.get("covs", []):
            self.covs.addItem(f"{a} ↔ {b}")
        self.editor.setPlainText(state.get("syntax", ""))
        if not state.get("syntax") and not state.get("constructs"):
            self._auto_detect(quiet=True)
        self._refresh_combos()
        self.tabs.setCurrentIndex(1 if state.get("syntax") else 0)

    def _store(self) -> None:
        self._state[self._key()] = {"opts": copy.deepcopy(self.opts),
                                    "constructs": self.constructs(),
                                    "paths": self._pairs(self.paths, "→"),
                                    "covs": self._pairs(self.covs, "↔"),
                                    "syntax": self.editor.toPlainText()}

    def reset(self) -> None:
        self.opts = copy.deepcopy(self.DEFAULTS)
        self.table.setRowCount(0)
        self.paths.clear()
        self.covs.clear()
        self.editor.clear()
        self._refresh_combos()
        self.tabs.setCurrentIndex(0)

    # ------------------------------------------------------------------ builder tab
    def _builder_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        grp = QGroupBox("Latent constructs (measurement model)")
        g = QVBoxLayout(grp)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Construct", "Indicators (observed variables)"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self.table.setColumnWidth(0, 130)
        self.table.verticalHeader().setVisible(False)
        self.table.itemChanged.connect(lambda _: self._refresh_combos())
        g.addWidget(self.table, 1)
        row = QHBoxLayout()
        row.addWidget(QLabel("Name:"))
        self.cname = QLineEdit()
        self.cname.setPlaceholderText("e.g. Motivation")
        self.cname.returnPressed.connect(self._add_construct)
        row.addWidget(self.cname, 1)
        for text, fn in (("Add selected as construct", self._add_construct),
                         ("Remove", self._remove_construct),
                         ("Auto-detect from item names", self._auto_detect)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        g.addLayout(row)
        lay.addWidget(grp, 3)

        pc = QHBoxLayout()
        self.paths, self.p_from, self.p_to = QListWidget(), QComboBox(), QComboBox()
        pc.addWidget(self._pair_box("Structural paths (regressions)", self.paths, self.p_from,
                                    self.p_to, "→"))
        self.covs, self.c_a, self.c_b = QListWidget(), QComboBox(), QComboBox()
        pc.addWidget(self._pair_box("Covariances (incl. residual)", self.covs, self.c_a,
                                    self.c_b, "↔"))
        lay.addLayout(pc, 2)

        tr = QHBoxLayout()
        tr.addWidget(QLabel("Template:"))
        self.template = QComboBox()
        for key, text in TEMPLATES:
            self.template.addItem(text, key)
        tr.addWidget(self.template, 1)
        gen = QPushButton("Generate Syntax ▶")
        gen.setDefault(False)
        gen.setAutoDefault(False)
        gen.clicked.connect(self._generate)
        tr.addWidget(gen)
        lay.addLayout(tr)
        note = QLabel("Path analysis uses the paths between observed variables (or: click the "
                      "predictors, then the dependent last). Mediation uses three variables "
                      "clicked in the order X, M, Y — or the first three constructs.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#666")
        lay.addWidget(note)
        return w

    def _pair_box(self, title, lst: QListWidget, a: QComboBox, b: QComboBox, sym: str):
        grp = QGroupBox(title)
        g = QGridLayout(grp)
        lst.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        g.addWidget(QLabel("From:" if sym == "→" else "Between:"), 0, 0)
        g.addWidget(a, 0, 1)
        g.addWidget(QLabel("To:" if sym == "→" else "and:"), 1, 0)
        g.addWidget(b, 1, 1)
        add, rem = QPushButton("Add"), QPushButton("Remove")
        add.setAutoDefault(False)
        rem.setAutoDefault(False)
        add.clicked.connect(lambda: self._add_pair(lst, a, b, sym))
        rem.clicked.connect(lambda: [lst.takeItem(lst.row(i)) for i in lst.selectedItems()])
        g.addWidget(add, 0, 2)
        g.addWidget(rem, 1, 2)
        g.addWidget(lst, 2, 0, 1, 3)
        return grp

    def _add_pair(self, lst, a, b, sym) -> None:
        x, y = a.currentText(), b.currentText()
        if not x or not y or x == y:
            return
        text = f"{x} {sym} {y}"
        if sym == "↔" and lst.findItems(f"{y} {sym} {x}", Qt.MatchFlag.MatchExactly):
            return
        if not lst.findItems(text, Qt.MatchFlag.MatchExactly):
            lst.addItem(text)

    @staticmethod
    def _pairs(lst: QListWidget, sym: str) -> list[tuple[str, str]]:
        out = []
        for i in range(lst.count()):
            a, b = [s.strip() for s in lst.item(i).text().split(sym)]
            out.append((a, b))
        return out

    def _selected_vars(self, list_order: bool = False) -> list[str]:
        """Selected variable names in click order (or in list order)."""
        items = self.source.selectedItems()
        if list_order:
            items = sorted(items, key=self.source.row)
        return [it.data(NAME_ROLE) for it in items]

    def _add_construct_row(self, name: str, items: list[str]) -> None:
        self.table.blockSignals(True)
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(name))
        self.table.setItem(r, 1, QTableWidgetItem(" + ".join(items)))
        self.table.blockSignals(False)

    def _add_construct(self) -> None:
        items = self._selected_vars(list_order=True)
        if not items:
            QMessageBox.information(self, self.windowTitle(),
                                    "Select the indicator variables in the list first.")
            return
        name = self.cname.text().strip() or f"F{self.table.rowCount() + 1}"
        name = re.sub(r"\W", "_", name)
        if self.ds.has_name(name):
            name += "_lv"
        if name in [c for c, _ in self.constructs()]:
            QMessageBox.information(self, self.windowTitle(), f"'{name}' already exists.")
            return
        self._add_construct_row(name, items)
        self.cname.clear()
        self.source.clearSelection()
        self._refresh_combos()

    def _remove_construct(self) -> None:
        for r in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(r)
        self._refresh_combos()

    def _auto_detect(self, quiet: bool = False) -> None:
        """Group numbered items by their prefix (X1…X6 → X, Y1…Y5 → Y), as questionnaires
        are usually coded."""
        groups: dict[str, list[str]] = {}
        for var in self.ds.variables:
            if var.type != NUMERIC:
                continue
            m = re.fullmatch(r"([A-Za-z][A-Za-z_]*?)[_.]?(\d+)", var.name)
            if m:
                groups.setdefault(m.group(1), []).append(var.name)
        found = [(k, v) for k, v in groups.items() if len(v) >= 2]
        if not found:
            if not quiet:
                QMessageBox.information(self, self.windowTitle(),
                                        "No numbered item groups (e.g. X1, X2, …) were found.")
            return
        existing = {c for c, _ in self.constructs()}
        for prefix, items in found:
            name = prefix if not self.ds.has_name(prefix) else f"{prefix}_lv"
            if name not in existing:
                self._add_construct_row(name, items)
        self._refresh_combos()

    def constructs(self) -> list[tuple[str, list[str]]]:
        out = []
        for r in range(self.table.rowCount()):
            n = self.table.item(r, 0)
            i = self.table.item(r, 1)
            name = n.text().strip() if n else ""
            items = [s for s in re.split(r"[\s,+;]+", i.text() if i else "") if s]
            if name and items:
                out.append((name, items))
        return out

    def _refresh_combos(self) -> None:
        names = [c for c, _ in self.constructs()] + [v.name for v in self.ds.variables
                                                     if v.type == NUMERIC]
        for combo in (self.p_from, self.p_to, self.c_a, self.c_b):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            if current in names:
                combo.setCurrentText(current)
            combo.blockSignals(False)

    def _generate(self) -> None:
        kind = self.template.currentData()
        cons = self.constructs()
        paths = self._pairs(self.paths, "→")
        covs = self._pairs(self.covs, "↔")
        sel = self._selected_vars()
        if kind == "cfa":
            if not cons:
                return self._tell("Add at least one construct first.")
            text = sem_syntax.cfa_syntax(cons)
            if covs:
                text += "\n" + "\n".join(f"{a} ~~ {b}" for a, b in covs)
        elif kind == "sem":
            if not paths:
                return self._tell("Add structural paths (From → To) first.")
            used = {x for p in paths for x in p}
            text = sem_syntax.sem_syntax([c for c in cons if c[0] in used] or cons, paths,
                                         covs, labels=True)
        elif kind == "path":
            obs = [p for p in paths if self.ds.has_name(p[0]) and self.ds.has_name(p[1])]
            if obs:
                text = sem_syntax.sem_syntax([], obs, covs, labels=True)
            elif len(sel) >= 2:
                text = sem_syntax.path_syntax(sel[-1], sel[:-1])
            else:
                return self._tell("Add paths between observed variables, or select the "
                                  "predictors and then the dependent variable.")
        else:
            if len(sel) == 3:
                x, m, y = sel
                text = sem_syntax.mediation_syntax(x, m, y)
            elif len(cons) >= 3:
                (x, _), (m, _), (y, _) = cons[:3]
                text = sem_syntax.cfa_syntax(cons[:3]) + "\n" + \
                    sem_syntax.mediation_syntax(x, m, y)
            else:
                return self._tell("Select exactly three variables (X, M, Y in that order) or "
                                  "define three constructs.")
        text = self._add_indirect(text) if kind in ("sem", "path") else text
        self.editor.setPlainText(text)
        self.tabs.setCurrentIndex(1)
        self._validate_live()

    @staticmethod
    def _add_indirect(text: str) -> str:
        """Define every two-step indirect effect for labelled paths (lavaan := lines)."""
        lab: dict[tuple[str, str], str] = {}
        for line in text.splitlines():
            if "~" in line and "=~" not in line and "~~" not in line and ":=" not in line:
                lhs, rhs = line.split("~", 1)
                for term in rhs.split("+"):
                    if "*" in term:
                        l_, v = term.split("*", 1)
                        lab[(v.strip(), lhs.strip())] = l_.strip()
        lines = []
        for (x, m), a in lab.items():
            for (m2, y), b in lab.items():
                if m2 == m and y != x:
                    lines.append(f"ind_{x}_{m}_{y} := {a}*{b}")
        if lines:
            text += "\n# indirect effects\n" + "\n".join(lines)
        return text

    def _tell(self, msg: str) -> None:
        QMessageBox.information(self, self.windowTitle(), msg)

    # ------------------------------------------------------------------ syntax tab
    def _syntax_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.editor = QPlainTextEdit()
        font = QFont()
        font.setFamilies(["Menlo", "Consolas", "Courier New"])
        font.setPointSize(12)
        self.editor.setFont(font)
        self.editor.setPlaceholderText("# Example (lavaan syntax)\nX =~ X1 + X2 + X3\n"
                                       "Y =~ Y1 + Y2 + Y3\nY ~ X")
        lay.addWidget(self.editor, 1)
        help_ = QLabel(HELP)
        help_.setWordWrap(True)
        help_.setStyleSheet("color:#666")
        lay.addWidget(help_)
        row = QHBoxLayout()
        check = QPushButton("Check Syntax")
        check.setAutoDefault(False)
        check.clicked.connect(self._validate_live)
        row.addWidget(check)
        row.addStretch()
        lay.addLayout(row)
        return w

    def _insert_variable(self, item) -> None:
        self.editor.insertPlainText(item.data(NAME_ROLE))
        self.tabs.setCurrentIndex(1)

    def _numeric(self) -> set[str]:
        return {v.name for v in self.ds.variables if v.type == NUMERIC}

    def _validate_live(self) -> None:
        text = self.editor.toPlainText()
        if not text.strip():
            self.status.setText("Build the model in the Model Builder (then Generate Syntax) "
                                "or type lavaan syntax in the Syntax tab.")
            self.status.setStyleSheet("color:#666")
            return
        ok, msg = sem_syntax.validate(text, self.ds.names(), self._numeric())
        self.status.setText(msg)
        self.status.setStyleSheet("color:#1d6b2a" if ok else "color:#b00020")

    # ------------------------------------------------------------------ sub-dialogs
    def _estimation(self) -> None:
        o = self.opts
        dlg = SubDialog("CB-SEM: Estimation", self)
        box = dlg.group("Estimator")
        dlg.radio(box, "estimator", [("ML", "Maximum likelihood (ML)"),
                                     ("GLS", "Generalized least squares (GLS)"),
                                     ("ULS", "Unweighted least squares (ULS)")],
                  o["estimator"])
        box = dlg.group("Identification of latent variables")
        dlg.radio(box, "identification", [("marker", "Marker variable (first loading = 1)"),
                                          ("std.lv", "Standardized latent variances (= 1)")],
                  o["identification"])
        box = dlg.group("Standard errors")
        dlg.radio(box, "information", [("expected", "Expected information"),
                                       ("observed", "Observed information (ML)"),
                                       ("robust", "Robust (MLM): Satorra-Bentler χ²")],
                  o["information"])
        box = dlg.group("Likelihood / chi-square")
        dlg.radio(box, "likelihood", [("normal", "Normal: χ² = N·F (lavaan)"),
                                      ("wishart", "Wishart: χ² = (N−1)·F (AMOS)")],
                  o["likelihood"])
        box = dlg.group("Baseline model")
        dlg.radio(box, "baseline", [("lavaan", "Exogenous covariates saturated (lavaan)"),
                                    ("amos", "Full independence (AMOS)")], o["baseline"])
        box = dlg.group("Automatic parameters")
        dlg.check(box, "cov_exogenous", "Correlate exogenous variables", o["cov_exogenous"])
        dlg.check(box, "auto_cov_y", "Residual covariances among final outcomes (lavaan)",
                  o["auto_cov_y"])
        box.addWidget(QLabel("Missing values: listwise deletion"))
        self._apply(dlg.run())

    def _output(self) -> None:
        o = self.opts
        dlg = SubDialog("CB-SEM: Output", self)
        box = dlg.group("Display")
        for key, text in (("normality", "Assessment of normality (+ Mahalanobis)"),
                          ("fit", "Model fit"), ("estimates", "Parameter estimates"),
                          ("r2", "Squared multiple correlations"),
                          ("effects", "Total, direct and indirect effects"),
                          ("sample_cov", "Sample covariances"),
                          ("implied_cov", "Implied covariances"),
                          ("residuals", "Standardized residual covariances")):
            dlg.check(box, f"out_{key}", text, o[f"out_{key}"])
        box = dlg.group("Measurement model")
        dlg.check(box, "out_measurement", "Loadings, CR, AVE, Cronbach's alpha",
                  o["out_measurement"])
        dlg.check(box, "out_discriminant", "Discriminant validity (Fornell-Larcker, HTMT)",
                  o["out_discriminant"])
        box = dlg.group("Modification indices")
        dlg.check(box, "out_mi", "Modification indices", o["out_mi"])
        dlg.number(box, "mi_min", "Threshold for modification indices:", o["mi_min"], 0, 1000,
                   2)
        dlg.number(box, "mi_top", "Show at most:", o["mi_top"], 1, 500, 0)
        box = dlg.group("Path diagram")
        dlg.check(box, "out_diagram", "Draw path diagram", o["out_diagram"])
        dlg.radio(box, "diagram_std", [(True, "Standardized estimates"),
                                       (False, "Unstandardized estimates")], o["diagram_std"])
        values = dlg.run()
        if values:
            values["mi_top"] = int(values["mi_top"])
        self._apply(values)

    def _bootstrap(self) -> None:
        o = self.opts
        dlg = SubDialog("CB-SEM: Bootstrap", self, columns=1)
        box = dlg.group("Bootstrap")
        dlg.check(box, "boot", "Perform bootstrap", o["boot"])
        dlg.number(box, "boot_n", "Number of bootstrap samples:", o["boot_n"], 100, 100000, 0)
        dlg.number(box, "seed", "Random seed (0 = random):", o["seed"], 0, 2147483646, 0)
        dlg.number(box, "level", "Confidence level (%):", o["level"], 50, 99.9, 1)
        box.addWidget(QLabel("Percentile and bias-corrected percentile intervals for all "
                             "parameters, indirect and defined (:=) effects."))
        values = dlg.run()
        if values:
            values["boot_n"] = int(values["boot_n"])
            values["seed"] = int(values["seed"])
        self._apply(values)

    def _save(self) -> None:
        dlg = SubDialog("CB-SEM: Save", self, columns=1)
        box = dlg.group("Save as variables")
        dlg.check(box, "save_scores", "Factor scores (regression method), FS_<construct>_n",
                  self.opts["save_scores"])
        self._apply(dlg.run())

    def _apply(self, values) -> None:
        if values:
            self.opts.update(values)

    # ------------------------------------------------------------------ run
    def syntax(self) -> str:
        return self.editor.toPlainText()

    def validate(self) -> str | None:
        text = self.syntax()
        if not text.strip():
            if self.constructs():
                self._generate_default()
                text = self.syntax()
            if not text.strip():
                return "Specify a model: build it in the Model Builder or type lavaan syntax."
        ok, msg = sem_syntax.validate(text, self.ds.names(), self._numeric())
        return None if ok else msg

    def _generate_default(self) -> None:
        kind = "sem" if self.paths.count() else "cfa"
        self.template.setCurrentIndex(self.template.findData(kind))
        self._generate()

    def accept(self) -> None:
        error = self.validate()
        if error:
            QMessageBox.warning(self, self.windowTitle(), error)
            return
        self._store()
        super().accept()

    def compute(self):
        o = self.opts
        info = o["information"]
        return sem_core.cb_sem(
            self.ds, self.syntax(), estimator=o["estimator"],
            identification=o["identification"], likelihood=o["likelihood"],
            information="expected" if info == "robust" else info, robust=info == "robust",
            boot=o["boot_n"] if o["boot"] else 0, seed=o["seed"] or None,
            level=o["level"] / 100,
            outputs={k[4:]: v for k, v in o.items() if k.startswith("out_")},
            mi_min=o["mi_min"], mi_top=int(o["mi_top"]), save_scores=o["save_scores"],
            standardized_diagram=o["diagram_std"], fixed_x_baseline=o["baseline"] == "lavaan",
            auto_cov_y=o["auto_cov_y"], cov_exogenous=o["cov_exogenous"])


MENUS = {"Analyze/Structural Equation Modeling": [("CB-SEM...", CBSEMDialog)]}


def _check(cond: bool, what: str) -> None:
    if not cond:
        raise AssertionError(f"CB-SEM selftest failed: {what}")


def selftest() -> None:
    """Headless: simulate a two-factor model with a structural path and recover it."""
    rng = np.random.default_rng(7)
    n = 400
    xi = rng.normal(size=n)
    eta = 0.6 * xi + rng.normal(scale=0.8, size=n)
    data = {}
    for k, lam in enumerate((1.0, 0.8, 0.9), 1):
        data[f"x{k}"] = lam * xi + rng.normal(scale=0.6, size=n)
    for k, lam in enumerate((1.0, 1.1, 0.7), 1):
        data[f"y{k}"] = lam * eta + rng.normal(scale=0.6, size=n)
    ds = Dataset(pd.DataFrame(data))
    syntax = "F1 =~ x1 + x2 + x3\nF2 =~ y1 + y2 + y3\nF2 ~ b*F1\nb2 := b*b"
    res = sem_core.cb_sem(ds, syntax, boot=200, seed=1, save_scores=True)
    est = res.est
    _check(est is not None and est.converged, "convergence")
    _check(abs(res.fit["chi2"]) < 40 and res.fit["df"] == 8, "chi-square / df")
    k = next(p.index for p in est.model.params if p.label == "b")
    _check(abs(est.theta[k] - 0.6) < 0.15, "structural path recovered")
    _check(res.fit["cfi"] > 0.95, "CFI")
    _check("FS_F1" in res.saved and "FS_F2" in res.saved, "factor scores")
    titles = {getattr(i, "title", "") for i in res.section.items}
    for t in ("Model Fit", "Factor Loadings (Measurement Model)", "Construct Reliability and "
              "Validity", "Discriminant Validity: HTMT", "Path Diagram"):
        _check(t in titles, f"output '{t}'")
    ok, _ = sem_syntax.validate("F =~ a + b\nb ~ q", ["a", "b"])
    _check(not ok, "syntax validation of unknown variables")

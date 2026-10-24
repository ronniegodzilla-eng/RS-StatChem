"""Analyze > Structural Equation Modeling > PLS-SEM... (SmartPLS-style model builder).

The model is built either graphically (constructs with their indicators, reflective or
formative, and structural paths) or as text syntax (``X =~ X1 + X2``, ``F <~ F1 + F2``,
``Y ~ X + M``); both views edit the same model. Settings, bootstrapping, quality criteria and
saving latent variable scores are set in SPSS-style sub-dialogs. The numerical work is in
:mod:`rsstatchem.core.stats.pls`.
"""
from __future__ import annotations

import copy

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from rsstatchem.core.dataset import NUMERIC
from rsstatchem.core.stats import pls
from rsstatchem.core.stats.pls_model import (
    EXAMPLE_SYNTAX,
    FORMATIVE,
    MODE_TEXT,
    REFLECTIVE,
    Construct,
    ModelError,
    PLSModel,
    mediation_template,
    parse_syntax,
)
from rsstatchem.ui.dialogs.base import (
    NAME_ROLE,
    SubDialog,
    VariablePickerDialog,
    is_numeric,
    variable_item,
)

SYNTAX_HELP = ("<b>Syntax</b>: <code>C =~ x1 + x2</code> reflective (Mode A) · "
               "<code>C &lt;~ x1 + x2</code> formative (Mode B) · <code>Y ~ X + M</code> "
               "structural paths (X → Y, M → Y) · <code>X -&gt; M -&gt; Y</code> chains · "
               "<code>Y ~ X + W + X:W</code> moderation (two-stage) · <code>#</code> comment")


class MediationTemplateDialog(VariablePickerDialog):
    """Simple mediation X -> M -> Y (plus X -> Y) from three groups of indicators."""

    DEFAULTS = {"x": "X", "m": "M", "y": "Y", "mode": REFLECTIVE}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "PLS-SEM: Simple Mediation Template", parent)
        self.add_target("x", "Independent construct (X) indicators:", accepts=is_numeric)
        self.add_target("m", "Mediator construct (M) indicators:", accepts=is_numeric)
        self.add_target("y", "Dependent construct (Y) indicators:", accepts=is_numeric)
        form = QFormLayout()
        self.names = {}
        for key, text in (("x", "Name of X:"), ("m", "Name of M:"), ("y", "Name of Y:")):
            edit = QLineEdit(self.opts[key])
            edit.textChanged.connect(lambda t, k=key: self.opts.__setitem__(k, t.strip()))
            self.names[key] = edit
            form.addRow(text, edit)
        self.mode = QComboBox()
        self.mode.addItem("Reflective (Mode A)", REFLECTIVE)
        self.mode.addItem("Formative (Mode B)", FORMATIVE)
        self.mode.setCurrentIndex(0 if self.opts["mode"] == REFLECTIVE else 1)
        self.mode.currentIndexChanged.connect(
            lambda _: self.opts.__setitem__("mode", self.mode.currentData()))
        form.addRow("Measurement:", self.mode)
        self.options_layout.addLayout(form)
        self.finish()

    def validate(self):
        names = [self.opts[k] for k in "xmy"]
        if len({n.lower() for n in names}) < 3 or not all(names):
            return "Give the three constructs different names."
        return None

    def model(self) -> PLSModel:
        o = self.opts
        return mediation_template((o["x"], self.selected("x")), (o["m"], self.selected("m")),
                                  (o["y"], self.selected("y")), o["mode"])


class PLSSEMDialog(QDialog):
    DEFAULTS = dict(pls.DEFAULTS)
    modifies_data = False
    _state: dict = {}
    _last_model: PLSModel | None = None

    def __init__(self, ds, parent=None):
        super().__init__(parent)
        self.ds = ds
        self.setWindowTitle("PLS-SEM (Partial Least Squares Structural Equation Modeling)")
        self.setMinimumSize(860, 600)
        self.opts = copy.deepcopy(self.DEFAULTS)
        self.opts.update(copy.deepcopy(self._state))
        last = type(self)._last_model
        self.model = copy.deepcopy(last) if last is not None else PLSModel()
        self.model_syntax = ""

        self.tabs = QTabWidget()
        self.tabs.addTab(self._builder_tab(), "Model Builder")
        self.tabs.addTab(self._syntax_tab(), "Syntax")
        self._current_tab = 0
        self.tabs.currentChanged.connect(self._tab_changed)

        options = QVBoxLayout()
        options.setAlignment(Qt.AlignmentFlag.AlignTop)
        for text, fn in (("Settings...", self._settings), ("Bootstrapping...", self._bootstrap),
                         ("Quality Criteria...", self._quality), ("Save...", self._save),
                         ("Mediation Template...", self._mediation)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            options.addWidget(b)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("color:#555")
        self.summary.setMaximumWidth(170)
        options.addSpacing(10)
        options.addWidget(self.summary)

        buttons = QDialogButtonBox()
        self.ok_button = buttons.addButton(QDialogButtonBox.StandardButton.Ok)
        reset = buttons.addButton(QDialogButtonBox.StandardButton.Reset)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        self.ok_button.setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        reset.clicked.connect(self.reset)

        body = QHBoxLayout()
        body.addWidget(self.tabs, 1)
        body.addLayout(options)
        layout = QVBoxLayout(self)
        layout.addLayout(body, 1)
        layout.addWidget(buttons)
        self._refresh()

    # ------------------------------------------------------------------ builder tab
    def _builder_tab(self) -> QWidget:
        page = QWidget()
        self.source = QListWidget()
        self.source.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        for var in self.ds.variables:
            if var.type == NUMERIC:
                self.source.addItem(variable_item(var))
        self.source.itemDoubleClicked.connect(lambda _: self._add_indicators())

        arrows = QVBoxLayout()
        arrows.addStretch()
        for text, tip, fn in (("New ▶", "Create a construct from the selected variables",
                               self._new_construct),
                              ("Add ▶", "Add the selected variables to the selected construct",
                               self._add_indicators),
                              ("◀ Remove", "Remove the selected indicator or construct",
                               self._remove_selected)):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            arrows.addWidget(b)
        arrows.addStretch()

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Construct / indicator", "Measurement"])
        self.tree.setColumnWidth(0, 190)
        self.tree.itemSelectionChanged.connect(self._tree_selection)
        self.tree.itemDoubleClicked.connect(self._tree_double_click)
        ctrl = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Reflective (Mode A)", REFLECTIVE)
        self.mode_combo.addItem("Formative (Mode B)", FORMATIVE)
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        rename = QPushButton("Rename...")
        rename.clicked.connect(self._rename)
        ctrl.addWidget(QLabel("Mode:"))
        ctrl.addWidget(self.mode_combo, 1)
        ctrl.addWidget(rename)
        cons = QVBoxLayout()
        cons.addWidget(QLabel("Constructs (latent variables):"))
        cons.addWidget(self.tree, 1)
        cons.addLayout(ctrl)

        top = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(QLabel("Variables:"))
        left.addWidget(self.source, 1)
        top.addLayout(left, 1)
        top.addLayout(arrows)
        top.addLayout(cons, 1)

        box = QGroupBox("Structural Model (inner model)")
        inner = QHBoxLayout(box)
        pick = QFormLayout()
        self.from_combo, self.to_combo = QComboBox(), QComboBox()
        self.mod_combo = QComboBox()
        self.mod_combo.setToolTip("Optional moderator: adds the interaction term "
                                  "(two-stage approach) and both main effects")
        pick.addRow("From (predictor):", self.from_combo)
        pick.addRow("To (dependent):", self.to_combo)
        pick.addRow("Moderator (optional):", self.mod_combo)
        add = QPushButton("Add Path ▶")
        add.clicked.connect(self._add_path)
        pick.addRow("", add)
        self.path_list = QListWidget()
        self.path_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.path_list.setMaximumHeight(120)
        rm = QPushButton("◀ Remove Path")
        rm.clicked.connect(self._remove_path)
        right = QVBoxLayout()
        right.addWidget(self.path_list)
        right.addWidget(rm, 0, Qt.AlignmentFlag.AlignRight)
        inner.addLayout(pick, 1)
        inner.addLayout(right, 1)

        layout = QVBoxLayout(page)
        layout.addLayout(top, 1)
        layout.addWidget(box)
        return page

    def _syntax_tab(self) -> QWidget:
        page = QWidget()
        self.editor = QPlainTextEdit()
        font = QFont("Menlo")
        font.setStyleHint(QFont.StyleHint.Monospace)
        self.editor.setFont(font)
        self.editor.setPlaceholderText(EXAMPLE_SYNTAX)
        help_label = QLabel(SYNTAX_HELP)
        help_label.setWordWrap(True)
        row = QHBoxLayout()
        example = QPushButton("Insert Example")
        example.clicked.connect(lambda: self.editor.setPlainText(EXAMPLE_SYNTAX))
        check = QPushButton("Check Model")
        check.clicked.connect(self._check)
        row.addWidget(example)
        row.addStretch()
        row.addWidget(check)
        names = QLabel("Variables: " + ", ".join(v.name for v in self.ds.variables
                                                   if v.type == NUMERIC))
        names.setWordWrap(True)
        names.setStyleSheet("color:#666")
        layout = QVBoxLayout(page)
        layout.addWidget(help_label)
        layout.addWidget(self.editor, 1)
        layout.addWidget(names)
        layout.addLayout(row)
        return page

    # ------------------------------------------------------------------ model <-> widgets
    def _refresh(self) -> None:
        selected = self._selected_construct()
        self.tree.blockSignals(True)
        self.tree.clear()
        for c in self.model.constructs:
            top = QTreeWidgetItem([c.name, MODE_TEXT[c.mode]])
            top.setData(0, NAME_ROLE, ("construct", c.name))
            f = top.font(0)
            f.setBold(True)
            top.setFont(0, f)
            for ind in c.indicators:
                child = QTreeWidgetItem([ind, ""])
                child.setData(0, NAME_ROLE, ("indicator", c.name, ind))
                top.addChild(child)
            self.tree.addTopLevelItem(top)
            top.setExpanded(True)
            if selected == c.name:
                top.setSelected(True)
        self.tree.blockSignals(False)
        names = self.model.names()
        for combo in (self.from_combo, self.to_combo):
            cur = combo.currentText()
            combo.clear()
            combo.addItems(names)
            if cur in names:
                combo.setCurrentText(cur)
        if self.to_combo.count() > 1 and self.to_combo.currentIndex() == \
                self.from_combo.currentIndex():
            self.to_combo.setCurrentIndex(min(self.from_combo.currentIndex() + 1,
                                              self.to_combo.count() - 1))
        cur = self.mod_combo.currentText()
        self.mod_combo.clear()
        self.mod_combo.addItems(["(none)"] + names)
        if cur in names:
            self.mod_combo.setCurrentText(cur)
        self.path_list.clear()
        for s, t in self.model.paths:
            item = QListWidgetItem(f"{s}  →  {t}")
            item.setData(NAME_ROLE, ("path", s, t))
            self.path_list.addItem(item)
        for x, w, t in self.model.interactions:
            item = QListWidgetItem(f"{x} × {w}  →  {t}   (moderation)")
            item.setData(NAME_ROLE, ("int", x, w, t))
            self.path_list.addItem(item)
        self._tree_selection()
        self._update_summary()

    def _update_summary(self) -> None:
        o = self.opts
        parts = [f"{len(self.model.constructs)} constructs, {len(self.model.paths)} paths"
                 + (f", {len(self.model.interactions)} moderation(s)"
                    if self.model.interactions else ""),
                 f"{'PLSc' if o['consistent'] else 'PLS'}, {o['scheme']} weighting"]
        if o["bootstrap"]:
            parts.append(f"Bootstrapping: {int(o['subsamples'])} subsamples, "
                         f"{pls.CI_TEXT[o['ci']].split(' bootstrap')[0].lower()}")
        else:
            parts.append("No bootstrapping")
        self.summary.setText("\n".join(parts))

    def _selected_construct(self) -> str | None:
        items = self.tree.selectedItems() if hasattr(self, "tree") else []
        if not items:
            return None
        data = items[0].data(0, NAME_ROLE)
        return data[1] if data else None

    def _tree_selection(self) -> None:
        name = self._selected_construct()
        self.mode_combo.blockSignals(True)
        if name and self.model.has(name):
            self.mode_combo.setEnabled(True)
            self.mode_combo.setCurrentIndex(0 if self.model.construct(name).reflective else 1)
        else:
            self.mode_combo.setEnabled(False)
        self.mode_combo.blockSignals(False)

    def _tree_double_click(self, item, _col) -> None:
        data = item.data(0, NAME_ROLE)
        if data and data[0] == "indicator":
            self._remove_selected()

    def _next_name(self) -> str:
        k = len(self.model.constructs) + 1
        while self.model.has(f"LV{k}"):
            k += 1
        return f"LV{k}"

    def _selected_vars(self) -> list[str]:
        return [i.data(NAME_ROLE) for i in self.source.selectedItems()]

    def _new_construct(self) -> None:
        names = self._selected_vars()
        default = self._next_name()
        if names:  # e.g. X1, X2, X3 -> "X"
            stem = names[0].rstrip("0123456789_") or default
            if all(n.rstrip("0123456789_") == stem for n in names) and \
                    not self.model.has(stem):
                default = stem
        name, ok = QInputDialog.getText(self, "New Construct", "Construct name:",
                                        text=default)
        name = name.strip()
        if not ok or not name:
            return
        if self.model.has(name):
            QMessageBox.warning(self, self.windowTitle(), f"Construct '{name}' already exists.")
            return
        self.model.constructs.append(Construct(name, names, REFLECTIVE))
        self._refresh()
        self._select_construct(name)

    def _select_construct(self, name: str) -> None:
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            item.setSelected(item.data(0, NAME_ROLE)[1] == name)

    def _add_indicators(self) -> None:
        name = self._selected_construct()
        if name is None:
            self._new_construct()
            return
        c = self.model.construct(name)
        for v in self._selected_vars():
            if v not in c.indicators:
                c.indicators.append(v)
        self._refresh()
        self._select_construct(name)

    def _remove_selected(self) -> None:
        for item in self.tree.selectedItems():
            data = item.data(0, NAME_ROLE)
            if not data:
                continue
            if data[0] == "construct" and self.model.has(data[1]):
                self.model.constructs = [c for c in self.model.constructs if c.name != data[1]]
                self.model.paths = [(s, t) for s, t in self.model.paths
                                    if data[1] not in (s, t)]
                self.model.interactions = [i for i in self.model.interactions
                                           if data[1] not in i]
            elif data[0] == "indicator" and self.model.has(data[1]):
                c = self.model.construct(data[1])
                c.indicators = [i for i in c.indicators if i != data[2]]
        self._refresh()

    def _mode_changed(self) -> None:
        name = self._selected_construct()
        if name and self.model.has(name):
            self.model.construct(name).mode = self.mode_combo.currentData()
            self._refresh()
            self._select_construct(name)

    def _rename(self) -> None:
        name = self._selected_construct()
        if not name:
            return
        new, ok = QInputDialog.getText(self, "Rename Construct", "Construct name:", text=name)
        new = new.strip()
        if not ok or not new or new == name:
            return
        if self.model.has(new):
            QMessageBox.warning(self, self.windowTitle(), f"Construct '{new}' already exists.")
            return
        self.model.construct(name).name = new
        self.model.paths = [(new if s == name else s, new if t == name else t)
                            for s, t in self.model.paths]
        self.model.interactions = [tuple(new if k == name else k for k in i)
                                   for i in self.model.interactions]
        self._refresh()
        self._select_construct(new)

    def _add_path(self) -> None:
        s, t = self.from_combo.currentText(), self.to_combo.currentText()
        if not s or not t:
            return
        if s == t:
            QMessageBox.warning(self, self.windowTitle(), "Choose two different constructs.")
            return
        w = self.mod_combo.currentText() if self.mod_combo.currentIndex() > 0 else None
        if w in (s, t):
            QMessageBox.warning(self, self.windowTitle(),
                                "The moderator must differ from both constructs of the path.")
            return
        new = [p for p in ([(s, t)] + ([(w, t)] if w else [])) if p not in self.model.paths]
        trial = PLSModel(self.model.constructs, self.model.paths + new)
        try:
            trial.topological_order()
        except ModelError as exc:
            QMessageBox.warning(self, self.windowTitle(), str(exc))
            return
        self.model.paths += new
        if w and (s, w, t) not in self.model.interactions:
            self.model.interactions.append((s, w, t))
        self.mod_combo.setCurrentIndex(0)
        self._refresh()

    def _remove_path(self) -> None:
        for item in self.path_list.selectedItems():
            data = item.data(NAME_ROLE)
            if data[0] == "path":
                pair = (data[1], data[2])
                if pair in self.model.paths:
                    self.model.paths.remove(pair)
                self.model.interactions = [i for i in self.model.interactions
                                           if not (i[2] == pair[1] and pair[0] in i[:2])]
            else:
                self.model.interactions = [i for i in self.model.interactions
                                           if i != tuple(data[1:])]
        self._refresh()

    def _tab_changed(self, index: int) -> None:
        if index == 1 and self._current_tab == 0:
            self.editor.setPlainText(self.model.to_syntax())
        elif index == 0 and self._current_tab == 1:
            if not self._apply_syntax():
                self.tabs.blockSignals(True)
                self.tabs.setCurrentIndex(1)
                self.tabs.blockSignals(False)
                return
        self._current_tab = index

    def _apply_syntax(self, show: bool = True) -> bool:
        text = self.editor.toPlainText()
        if not text.strip():
            self.model = PLSModel()
            self._refresh()
            return True
        try:
            self.model = parse_syntax(text)
        except ModelError as exc:
            if show:
                QMessageBox.warning(self, self.windowTitle(), str(exc))
            return False
        self._refresh()
        return True

    def _validated(self) -> tuple[PLSModel, list[str]]:
        if self.tabs.currentIndex() == 1:
            self.model = parse_syntax(self.editor.toPlainText())
            self._refresh()
        model = copy.deepcopy(self.model)
        numeric = {v.name for v in self.ds.variables if v.type == NUMERIC}
        warnings = model.validate(self.ds.names(), numeric)
        return model, warnings

    def _check(self) -> None:
        try:
            model, warnings = self._validated()
        except ModelError as exc:
            QMessageBox.warning(self, self.windowTitle(), str(exc))
            return
        text = (f"The model is valid: {len(model.constructs)} constructs, "
                f"{sum(len(c.indicators) for c in model.constructs)} indicators, "
                f"{len(model.paths)} structural paths.")
        if warnings:
            text += "\n\n" + "\n".join(warnings)
        QMessageBox.information(self, self.windowTitle(), text)

    # ------------------------------------------------------------------ sub-dialogs
    def _apply(self, values) -> None:
        if values:
            for k in ("max_iter", "stop", "subsamples", "seed", "omission", "folds", "reps"):
                if k in values:
                    values[k] = int(round(values[k]))
            self.opts.update(values)
            self._update_summary()

    def _settings(self) -> None:
        o = self.opts
        dlg = SubDialog("PLS-SEM: Settings", self)
        box = dlg.group("Weighting Scheme")
        dlg.radio(box, "scheme", [("path", "Path"), ("factorial", "Factorial"),
                                  ("centroid", "Centroid")], o["scheme"])
        box = dlg.group("Algorithm")
        dlg.number(box, "max_iter", "Maximum iterations:", o["max_iter"], 10, 10000, 0)
        dlg.number(box, "stop", "Stop criterion (10^-x), x:", o["stop"], 3, 12, 0)
        dlg.check(box, "consistent", "Consistent PLS (PLSc) for reflective constructs",
                  o["consistent"])
        box = dlg.group("Missing Values")
        dlg.radio(box, "missing", [("listwise", "Casewise deletion"),
                                   ("mean", "Mean replacement")], o["missing"])
        self._apply(dlg.run())

    def _bootstrap(self) -> None:
        o = self.opts
        dlg = SubDialog("PLS-SEM: Bootstrapping", self)
        box = dlg.group("Bootstrapping")
        dlg.check(box, "bootstrap", "Perform bootstrapping", o["bootstrap"])
        dlg.number(box, "subsamples", "Subsamples:", o["subsamples"], 100, 100000, 0)
        dlg.number(box, "seed", "Random seed (0 = random):", o["seed"], 0, 2 ** 31 - 1, 0)
        box = dlg.group("Confidence Interval Method")
        dlg.radio(box, "ci", [("percentile", "Percentile bootstrap"),
                              ("bc", "Bias-corrected (BC)"),
                              ("bca", "Bias-corrected and accelerated (BCa)")], o["ci"])
        box = dlg.group("Test Type")
        dlg.radio(box, "tails", [(2, "Two-tailed"), (1, "One-tailed")], o["tails"])
        dlg.number(box, "alpha", "Significance level:", o["alpha"], 0.001, 0.2, 3)
        self._apply(dlg.run())

    def _quality(self) -> None:
        o = self.opts
        dlg = SubDialog("PLS-SEM: Quality Criteria", self)
        box = dlg.group("Predictive Relevance")
        dlg.check(box, "blindfold", "Blindfolding (Q², construct cross-validated "
                  "redundancy)", o["blindfold"])
        dlg.number(box, "omission", "Omission distance:", o["omission"], 5, 25, 0)
        dlg.check(box, "predict", "PLSpredict (Q²predict, RMSE, MAE)", o["predict"])
        dlg.number(box, "folds", "Folds:", o["folds"], 2, 50, 0)
        dlg.number(box, "reps", "Repetitions:", o["reps"], 1, 100, 0)
        box = dlg.group("Output")
        dlg.check(box, "crossloadings", "Cross loadings", o["crossloadings"])
        dlg.check(box, "fit", "Model fit (SRMR, d_ULS, d_G, NFI)", o["fit"])
        dlg.check(box, "hypotheses", "Hypothesis testing summary", o["hypotheses"])
        dlg.check(box, "notes", "Interpretation notes (Hair et al.)", o["notes"])
        box = dlg.group("Path Diagram")
        dlg.check(box, "diagram", "Show path diagram", o["diagram"])
        dlg.radio(box, "diagram_value", [("p", "Coefficients with p values"),
                                         ("t", "Coefficients with t values"),
                                         ("none", "Coefficients only")], o["diagram_value"])
        self._apply(dlg.run())

    def _save(self) -> None:
        dlg = SubDialog("PLS-SEM: Save", self, columns=1)
        box = dlg.group("Save as New Variables")
        dlg.check(box, "save_scores", "Latent variable scores (standardized)",
                  self.opts["save_scores"])
        self._apply(dlg.run())

    def _mediation(self) -> None:
        dlg = MediationTemplateDialog(self.ds, self)
        if not dlg.exec():
            return
        model = dlg.model()
        if self.model.constructs and QMessageBox.question(
                self, self.windowTitle(), "Replace the current model with the mediation "
                "template?") != QMessageBox.StandardButton.Yes:
            return
        self.model = model
        self._refresh()
        if self.tabs.currentIndex() == 1:
            self.editor.setPlainText(self.model.to_syntax())

    # ------------------------------------------------------------------ protocol
    def reset(self) -> None:
        self.model = PLSModel()
        self.opts = copy.deepcopy(self.DEFAULTS)
        self.editor.clear()
        self._refresh()

    def accept(self) -> None:
        try:
            model, _ = self._validated()
        except ModelError as exc:
            QMessageBox.warning(self, self.windowTitle(), str(exc))
            return
        self.model_syntax = model.to_syntax()
        type(self)._last_model = copy.deepcopy(self.model)
        type(self)._state = copy.deepcopy(self.opts)
        super().accept()

    def compute(self):
        syntax = self.model_syntax or self.model.to_syntax()
        return pls.pls_sem(self.ds, syntax, **self.opts)


MENUS = {"Analyze/Structural Equation Modeling": [("PLS-SEM...", PLSSEMDialog)]}


def selftest() -> None:
    pls.selftest()

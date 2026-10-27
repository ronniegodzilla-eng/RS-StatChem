"""Analyze > Regression (Binary Logistic, Multinomial Logistic, Ordinal, Curve Estimation),
Analyze > Compare Means > Means and Analyze > Classify > ROC Curve dialogs, laid out like SPSS.

``MENUS`` also carries the Analyze > Classify entries from :mod:`rsstatchem.ui.dialogs.classify`
so the integrator can merge one dictionary; ``selftest()`` exercises all of the core code.
"""
from __future__ import annotations

from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem, QPushButton,
                               QRadioButton, QVBoxLayout, QAbstractItemView)

from rsstatchem.core.dataset import NUMERIC, format_code
from rsstatchem.core.stats import curvefit_spss, logistic, means, roc
from rsstatchem.ui.dialogs.base import (BASE_TEXT_ROLE, NAME_ROLE, SubDialog, VariablePickerDialog,
                                        is_numeric)


def _combo(owner: VariablePickerDialog, key: str, items: list[tuple[str, str]]) -> QComboBox:
    box = QComboBox()
    for value, text in items:
        box.addItem(text, value)
    box.setCurrentIndex(max(box.findData(owner.opts[key]), 0))
    box.currentIndexChanged.connect(lambda _: owner.opts.__setitem__(key, box.currentData()))
    return box


def _check(owner: VariablePickerDialog, key: str, text: str) -> QCheckBox:
    cb = QCheckBox(text)
    cb.setChecked(bool(owner.opts[key]))
    cb.toggled.connect(lambda on: owner.opts.__setitem__(key, on))
    return cb


def _multi(owner: VariablePickerDialog, title: str, key: str,
           groups: list[tuple[str, list[tuple[str, str]]]], extra=None, columns=2) -> None:
    """Checkbox sub-dialog whose ticked keys are stored as a list in ``owner.opts[key]``."""
    dlg = SubDialog(title, owner, columns=columns)
    for group_title, items in groups:
        box = dlg.group(group_title)
        for item_key, text in items:
            dlg.check(box, item_key, text, item_key in owner.opts[key])
    if extra:
        extra(dlg)
    values = dlg.run()
    if values is None:
        return
    keys = [k for _, items in groups for k, _ in items]
    owner.opts[key] = [k for k in keys if values.get(k)]
    for k, v in values.items():
        if k not in keys:
            owner.opts[k] = v


# ============================================================================ binary logistic
class _CategoricalDialog(QDialog):
    """Define Categorical Variables: per-variable contrast and reference category."""

    def __init__(self, ds, covariates: list[str], current: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Logistic Regression: Define Categorical Variables")
        grid = QGridLayout()
        grid.addWidget(QLabel("Covariate"), 0, 0)
        grid.addWidget(QLabel("Categorical"), 0, 1)
        grid.addWidget(QLabel("Contrast"), 0, 2)
        grid.addWidget(QLabel("Reference Category"), 0, 3)
        self.rows = {}
        for i, name in enumerate(covariates, start=1):
            var = ds.var(name)
            cb = QCheckBox()
            cb.setChecked(name in current or var.type != NUMERIC)
            cb.setEnabled(var.type == NUMERIC)
            contrast = QComboBox()
            for c in logistic.CONTRASTS:
                contrast.addItem(c.capitalize(), c)
            ref = QComboBox()
            ref.addItem("Last", "last")
            ref.addItem("First", "first")
            c0, r0 = current.get(name, ("indicator", "last"))
            contrast.setCurrentIndex(max(contrast.findData(c0), 0))
            ref.setCurrentIndex(max(ref.findData(r0), 0))
            grid.addWidget(QLabel(var.title), i, 0)
            grid.addWidget(cb, i, 1)
            grid.addWidget(contrast, i, 2)
            grid.addWidget(ref, i, 3)
            self.rows[name] = (cb, contrast, ref)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Continue")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(grid)
        note = QLabel("Reference Category applies to Indicator, Simple and Deviation contrasts.")
        note.setStyleSheet("color:#666")
        lay.addWidget(note)
        lay.addWidget(buttons)

    def result(self) -> dict:
        return {n: [c.currentData(), r.currentData()] for n, (cb, c, r) in self.rows.items()
                if cb.isChecked()}


class BinaryLogisticDialog(VariablePickerDialog):
    DEFAULTS = {"method": "enter", "blocks": [["enter", []]], "categorical": {},
                "save": [], "hosmer": True, "ci": False, "level": 95.0,
                "iteration_history": False, "each_step": True, "p_enter": 0.05,
                "p_remove": 0.10, "cut": 0.5, "max_iter": 20, "constant": True,
                "class_plot": False, "casewise": False, "casewise_all": False,
                "casewise_sd": 2.0, "correlations": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Logistic Regression", parent)
        self.setMinimumSize(640, 440)
        self.add_target("dep", "Dependent:", single=True)
        header = QHBoxLayout()
        self.block_label = QLabel()
        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.prev_button.clicked.connect(lambda: self._go(-1))
        self.next_button.clicked.connect(lambda: self._go(1))
        header.addWidget(self.block_label)
        header.addStretch()
        header.addWidget(self.prev_button)
        header.addWidget(self.next_button)
        self._targets_column.addLayout(header)
        self.add_target("cov", "Covariates:", required=False)
        inter = QHBoxLayout()
        self.ab_button = QPushButton(">a*b>")
        self.ab_button.setToolTip("Select two or more variables in the source list to add "
                                  "their interaction")
        self.ab_button.clicked.connect(self._add_interaction)
        inter.addWidget(self.ab_button)
        inter.addStretch()
        self._targets_column.addLayout(inter)
        self.method = QComboBox()
        for key, text in logistic.METHODS.items():
            self.method.addItem(text, key)
        form = QFormLayout()
        form.addRow("Method:", self.method)
        self._targets_column.addLayout(form)
        self.add_option_button("Categorical...", self._categorical)
        self.add_option_button("Save...", self._save)
        self.add_option_button("Options...", self._options)
        self._blocks = [[m, [n for n in names if all(self.ds.has_name(p)
                                                     for p in n.split("*"))]]
                        for m, names in self.opts.get("blocks") or [["enter", []]]] \
            or [["enter", []]]
        self._bi = 0
        self.finish()
        self._load_block()

    # ------------------------------------------------------------- covariate items
    def _cov_target(self):
        return next(t for t in self._targets if t.key == "cov")

    def _add_term_item(self, term: str) -> None:
        widget = self._cov_target().widget
        if any(widget.item(i).data(NAME_ROLE) == term for i in range(widget.count())):
            return
        item = QListWidgetItem(term.replace("*", " * "))
        item.setData(NAME_ROLE, term)
        item.setData(BASE_TEXT_ROLE, term.replace("*", " * "))
        widget.addItem(item)

    def _add_interaction(self) -> None:
        names = [i.data(NAME_ROLE) for i in self.source.selectedItems()]
        widget = self._cov_target().widget
        names += [i.data(NAME_ROLE) for i in widget.selectedItems()
                  if "*" not in i.data(NAME_ROLE)]
        names = list(dict.fromkeys(names))
        if len(names) < 2:
            return
        self._add_term_item("*".join(names))
        self.source.clearSelection()
        self._update_arrows()

    def _move_back(self, target, all_items: bool = False) -> None:
        if target.key == "cov":
            items = [target.widget.item(i) for i in range(target.widget.count())] \
                if all_items else target.widget.selectedItems()
            for item in items:
                if "*" in (item.data(NAME_ROLE) or ""):
                    target.widget.takeItem(target.widget.row(item))
        super()._move_back(target, all_items)

    def _cov_terms(self) -> list[str]:
        w = self._cov_target().widget
        return [w.item(i).data(NAME_ROLE) for i in range(w.count())]

    # ------------------------------------------------------------- blocks
    def _store_block(self) -> None:
        self._blocks[self._bi] = [self.method.currentData(), self._cov_terms()]

    def _load_block(self) -> None:
        target = self._cov_target()
        self._move_back(target, all_items=True)
        method, terms = self._blocks[self._bi]
        for t in terms:
            if "*" in t:
                self._add_term_item(t)
            else:
                self._take(t, target)
        self.method.setCurrentIndex(max(self.method.findData(method), 0))
        self._update_arrows()

    def _go(self, step: int) -> None:
        self._store_block()
        target = self._bi + step
        if target < 0:
            return
        if target == len(self._blocks):
            if not self._blocks[self._bi][1]:
                return
            self._blocks.append(["enter", []])
        while len(self._blocks) - 1 > target and not self._blocks[-1][1]:
            self._blocks.pop()
        self._bi = target
        self._load_block()

    def targets_changed(self) -> None:
        if not hasattr(self, "_blocks"):
            return
        total = max(len(self._blocks), self._bi + 1)
        self.block_label.setText(f"Block {self._bi + 1} of {total}")
        self.prev_button.setEnabled(self._bi > 0)
        self.next_button.setEnabled(self._cov_target().widget.count() > 0)
        cats = self.opts["categorical"]
        w = self._cov_target().widget
        for i in range(w.count()):
            item = w.item(i)
            name = item.data(NAME_ROLE)
            if name in cats or (self.ds.has_name(name) and
                                self.ds.var(name).type != NUMERIC):
                c, r = cats.get(name, ("indicator", "last"))
                item.setText(f"{name}(Cat: {c[:3].capitalize()}"
                             f"{'(first)' if r == 'first' else ''})")
            else:
                item.setText(item.data(BASE_TEXT_ROLE) or name)

    def _update_ok(self) -> None:
        if not hasattr(self, "ok_button") or not hasattr(self, "_blocks"):
            return
        has = self._cov_target().widget.count() > 0 or \
            any(t for i, (_, t) in enumerate(self._blocks) if i != self._bi)
        self.ok_button.setEnabled(bool(self.selected("dep")) and has)

    def blocks(self) -> list[tuple[str, list[str]]]:
        self._store_block()
        return [(m, list(t)) for m, t in self._blocks if t]

    def _variables(self) -> list[str]:
        names = []
        for _, terms in self.blocks():
            for t in terms:
                names += t.split("*")
        return list(dict.fromkeys(names))

    # ------------------------------------------------------------- sub-dialogs
    def _categorical(self) -> None:
        names = self._variables()
        if not names:
            return
        dlg = _CategoricalDialog(self.ds, names, self.opts["categorical"], self)
        if dlg.exec():
            self.opts["categorical"] = dlg.result()
            self.targets_changed()

    def _save(self) -> None:
        _multi(self, "Logistic Regression: Save", "save", [
            ("Predicted Values", [("prob", "Probabilities"), ("group", "Group membership")]),
            ("Influence", [("cook", "Cook's"), ("lever", "Leverage values"),
                           ("dfbeta", "DfBeta(s)")]),
            ("Residuals", [("resid", "Unstandardized"), ("lresid", "Logit"),
                           ("sresid", "Studentized"), ("zresid", "Standardized"),
                           ("dev", "Deviance")])], columns=3)

    def _options(self) -> None:
        dlg = SubDialog("Logistic Regression: Options", self)
        box = dlg.group("Statistics and Plots")
        dlg.check(box, "class_plot", "Classification plots", self.opts["class_plot"])
        dlg.check(box, "hosmer", "Hosmer-Lemeshow goodness-of-fit", self.opts["hosmer"])
        dlg.check(box, "casewise", "Casewise listing of residuals", self.opts["casewise"])
        dlg.radio(box, "casewise_all", [(False, "Outliers outside (std. dev.):"),
                                        (True, "All cases")], self.opts["casewise_all"])
        dlg.number(box, "casewise_sd", "std. dev.", self.opts["casewise_sd"], 0.5, 20, 1)
        dlg.check(box, "correlations", "Correlations of estimates", self.opts["correlations"])
        dlg.check(box, "iteration_history", "Iteration history",
                  self.opts["iteration_history"])
        dlg.check(box, "ci", "CI for exp(B):", self.opts["ci"])
        dlg.number(box, "level", "Level (%):", self.opts["level"], 50, 99.99, 2)
        box = dlg.group("Display")
        dlg.radio(box, "each_step", [(True, "At each step"), (False, "At last step")],
                  self.opts["each_step"])
        box = dlg.group("Probability for Stepwise")
        dlg.number(box, "p_enter", "Entry:", self.opts["p_enter"], 0.0001, 0.99, 4)
        dlg.number(box, "p_remove", "Removal:", self.opts["p_remove"], 0.0001, 0.99, 4)
        box = dlg.group("Criteria")
        dlg.number(box, "cut", "Classification cutoff:", self.opts["cut"], 0.01, 0.99, 2)
        dlg.number(box, "max_iter", "Maximum Iterations:", self.opts["max_iter"], 1, 999, 0)
        dlg.check(box, "constant", "Include constant in model", self.opts["constant"])
        values = dlg.run()
        if values:
            values["max_iter"] = int(values["max_iter"])
            if values["p_enter"] >= values["p_remove"]:
                values["p_remove"] = min(0.99, values["p_enter"] + 0.05)
            self.opts.update(values)

    def validate(self):
        blocks = self.blocks()
        if not blocks:
            return "Specify at least one covariate."
        if self.selected_one("dep") in self._variables():
            return "The dependent variable cannot also be a covariate."
        self.opts["blocks"] = [[m, t] for m, t in blocks]
        return None

    def compute(self):
        o = self.opts
        names = self._variables()
        cats = {n: tuple(v) for n, v in o["categorical"].items() if n in names}
        return logistic.binary_logistic(
            self.ds, self.selected_one("dep"), None, cats, cut=o["cut"],
            p_enter=o["p_enter"], p_remove=o["p_remove"], max_iter=int(o["max_iter"]),
            constant=o["constant"], hosmer=o["hosmer"], ci=o["ci"], level=self.level(),
            iteration_history=o["iteration_history"], each_step=o["each_step"],
            save=tuple(o["save"]), blocks=[(m, t) for m, t in o["blocks"]],
            class_plot=o["class_plot"], casewise=o["casewise"],
            casewise_sd=o["casewise_sd"], casewise_all=o["casewise_all"],
            correlations=o["correlations"])


# ============================================================================ term builder
class TermBuilderDialog(QDialog):
    """SPSS "Build Term(s)" pattern: Factors & Covariates list, a term type and target lists
    (e.g. Forced Entry Terms / Stepwise Terms, Location, Scale, GENLIN Model)."""

    TYPES = ("Interaction", "Main effects", "All 2-way", "All 3-way", "All 4-way",
             "All 5-way")

    def __init__(self, title: str, names: list[str], lists: list[tuple[str, str, list[str]]],
                 parent=None, top=None, bottom=None):
        import itertools  # noqa: F401  (used in _build)

        super().__init__(parent)
        self.setWindowTitle(title)
        self.source = QListWidget()
        self.source.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        for n in names:
            self.source.addItem(n)
        self.kind = QComboBox()
        self.kind.addItems(self.TYPES)
        self.targets: dict[str, QListWidget] = {}
        body = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(QLabel("Factors && Covariates:"))
        left.addWidget(self.source)
        body.addLayout(left, 1)
        mid_right = QVBoxLayout()
        for key, label, terms in lists:
            row = QHBoxLayout()
            arrow = QPushButton("▶")
            arrow.setObjectName("arrow")
            lw = QListWidget()
            lw.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            for t in terms:
                lw.addItem(t.replace("*", " * "))
            lw.itemDoubleClicked.connect(lambda item, w=lw: w.takeItem(w.row(item)))
            arrow.clicked.connect(lambda _=False, w=lw: self._add(w))
            col = QVBoxLayout()
            col.addWidget(QLabel(label))
            col.addWidget(lw)
            row.addWidget(arrow)
            row.addLayout(col, 1)
            mid_right.addLayout(row, 1)
            self.targets[key] = lw
        body.addLayout(mid_right, 1)
        lay = QVBoxLayout(self)
        if top is not None:
            lay.addLayout(top)
        form = QFormLayout()
        form.addRow("Build Term(s) — Type:", self.kind)
        lay.addLayout(form)
        lay.addLayout(body, 1)
        hint = QLabel("Double-click a term to remove it.")
        hint.setStyleSheet("color:#666")
        lay.addWidget(hint)
        if bottom is not None:
            lay.addLayout(bottom)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Continue")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self.resize(560, 380)

    def build(self, names: list[str]) -> list[str]:
        import itertools

        kind = self.kind.currentText()
        if not names:
            return []
        if kind == "Interaction":
            return ["*".join(names)]
        if kind == "Main effects":
            return list(names)
        size = int(kind.split()[1][0])
        return ["*".join(c) for c in itertools.combinations(names, size)]

    def _add(self, widget: QListWidget) -> None:
        names = [i.text() for i in self.source.selectedItems()]
        have = {widget.item(i).text() for i in range(widget.count())}
        for t in self.build(names):
            text = t.replace("*", " * ")
            if text not in have:
                widget.addItem(text)

    def terms(self, key: str) -> list[str]:
        w = self.targets[key]
        return [w.item(i).text().replace(" * ", "*") for i in range(w.count())]


# ============================================================================ multinomial
class MultinomialDialog(VariablePickerDialog):
    DEFAULTS = {"reference": "last", "custom": "", "classification": True, "gof": True,
                "lr_tests": True, "level": 95.0, "save": [], "model": "main", "terms": [],
                "stepwise_terms": [], "step_method": "forward_entry", "pseudo": True,
                "step_summary": True, "fitting_info": True, "info_criteria": False,
                "cell_probs": False, "monotonicity": False, "estimates": True,
                "correlations": False, "covariances": False, "max_iter": 100,
                "p_converge": 1e-6, "scale": "none", "p_entry": 0.05, "p_removal": 0.10}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Multinomial Logistic Regression", parent)
        self.add_target("dep", "Dependent:", single=True)
        self.add_target("factors", "Factor(s):", required=False)
        self.add_target("covs", "Covariate(s):", accepts=is_numeric, required=False)
        self.ref_button = self.add_option_button("Reference Category...", self._reference)
        self.add_option_button("Model...", self._model)
        self.add_option_button("Statistics...", self._statistics)
        self.add_option_button("Criteria...", self._criteria)
        self.add_option_button("Options...", self._options)
        self.add_option_button("Save...", self._save)
        self.finish()

    def targets_changed(self) -> None:
        dep = self.selected_one("dep") if self._targets else None
        if dep:
            ref = self.opts["reference"]
            text = {"last": "Last", "first": "First"}.get(ref, f"Custom: {self.opts['custom']}")
            self.set_item_suffix("dep", f"({text})")

    def _reference(self) -> None:
        dlg = SubDialog("Multinomial Logistic Regression: Reference Category", self, columns=1)
        box = dlg.group("Reference Category")
        dlg.radio(box, "reference", [("first", "First Category"), ("last", "Last Category"),
                                     ("custom", "Custom")],
                  self.opts["reference"] if self.opts["reference"] in ("first", "last")
                  else "custom")
        edit = QLineEdit(self.opts["custom"])
        form = QFormLayout()
        form.addRow("Value:", edit)
        box.addLayout(form)
        dep = self.selected_one("dep")
        if dep and self.ds.var(dep).value_labels:
            hint = QLabel(", ".join(f"{format_code(k)} = {v}"
                                    for k, v in self.ds.var(dep).value_labels.items()))
            hint.setWordWrap(True)
            hint.setStyleSheet("color:#666")
            box.addWidget(hint)
        values = dlg.run()
        if values:
            self.opts["reference"] = values["reference"]
            self.opts["custom"] = edit.text().strip()
            self.targets_changed()

    def _model(self) -> None:
        names = self.selected("factors") + self.selected("covs")
        top = QHBoxLayout()
        group = QButtonGroup(self)
        radios = {}
        for key, text in (("main", "Main effects"), ("full", "Full factorial"),
                          ("custom", "Custom/Stepwise")):
            rb = QRadioButton(text)
            rb.setChecked(self.opts["model"] == key)
            group.addButton(rb)
            top.addWidget(rb)
            radios[key] = rb
        bottom = QFormLayout()
        method = QComboBox()
        for key, text in logistic.STEP_METHODS.items():
            method.addItem(text, key)
        method.setCurrentIndex(max(method.findData(self.opts["step_method"]), 0))
        bottom.addRow("Stepwise Method:", method)
        dlg = TermBuilderDialog("Multinomial Logistic Regression: Model", names, [
            ("forced", "Forced Entry Terms:", self.opts["terms"]),
            ("stepwise", "Stepwise Terms:", self.opts["stepwise_terms"])], self, top, bottom)
        if dlg.exec():
            self.opts["model"] = next(k for k, rb in radios.items() if rb.isChecked())
            self.opts["terms"] = dlg.terms("forced")
            self.opts["stepwise_terms"] = dlg.terms("stepwise")
            self.opts["step_method"] = method.currentData()

    def _statistics(self) -> None:
        dlg = SubDialog("Multinomial Logistic Regression: Statistics", self)
        box = dlg.group("Model")
        for key, text in (("pseudo", "Pseudo R-square"), ("step_summary", "Step summary"),
                          ("fitting_info", "Model fitting information"),
                          ("info_criteria", "Information criteria"),
                          ("cell_probs", "Cell probabilities"),
                          ("classification", "Classification table"),
                          ("gof", "Goodness-of-fit"),
                          ("monotonicity", "Monotonicity measures")):
            dlg.check(box, key, text, self.opts[key])
        box = dlg.group("Parameters")
        for key, text in (("estimates", "Estimates"), ("lr_tests", "Likelihood ratio tests"),
                          ("correlations", "Asymptotic correlations"),
                          ("covariances", "Asymptotic covariances")):
            dlg.check(box, key, text, self.opts[key])
        dlg.number(box, "level", "Confidence interval (%):", self.opts["level"], 50, 99.99, 2)
        values = dlg.run()
        if values:
            self.opts.update(values)

    def _criteria(self) -> None:
        dlg = SubDialog("Multinomial Logistic Regression: Convergence Criteria", self,
                        columns=1)
        box = dlg.group("Iterations")
        dlg.number(box, "max_iter", "Maximum iterations:", self.opts["max_iter"], 1, 1000, 0)
        dlg.number(box, "p_converge", "Parameter convergence:", self.opts["p_converge"],
                   1e-10, 0.01, 10)
        values = dlg.run()
        if values:
            values["max_iter"] = int(values["max_iter"])
            self.opts.update(values)

    def _options(self) -> None:
        dlg = SubDialog("Multinomial Logistic Regression: Options", self, columns=1)
        dlg.radio(dlg.group("Dispersion Scale"), "scale", [
            ("none", "None"), ("pearson", "Pearson"), ("deviance", "Deviance")],
            self.opts["scale"])
        box = dlg.group("Stepwise Options")
        dlg.number(box, "p_entry", "Entry Probability:", self.opts["p_entry"], 0.0001, 1, 4)
        dlg.number(box, "p_removal", "Removal Probability:", self.opts["p_removal"], 0.0001,
                   1, 4)
        values = dlg.run()
        if values:
            self.opts.update(values)

    def _save(self) -> None:
        _multi(self, "Multinomial Logistic Regression: Save", "save", [
            ("Saved Variables", [("est", "Estimated response probabilities"),
                                 ("pre", "Predicted category"),
                                 ("max", "Predicted category probability"),
                                 ("acp", "Actual category probability")])], columns=1)

    def validate(self):
        if not self.selected("factors") and not self.selected("covs"):
            return "Specify at least one factor or covariate."
        if self.opts["reference"] == "custom" and not self.opts["custom"]:
            return "Enter the value of the custom reference category."
        names = set(self.selected("factors") + self.selected("covs"))
        for t in self.opts["terms"] + self.opts["stepwise_terms"]:
            if not set(t.split("*")) <= names:
                return f"The model term {t} uses a variable that is no longer selected."
        return None

    def compute(self):
        o = self.opts
        ref = o["custom"] if o["reference"] == "custom" else o["reference"]
        custom = o["model"] == "custom"
        return logistic.multinomial(
            self.ds, self.selected_one("dep"), self.selected("factors"), self.selected("covs"),
            ref, self.level(), o["classification"], o["gof"], o["lr_tests"],
            tuple(o["save"]), model=o["model"], terms=o["terms"] if custom else None,
            stepwise_terms=o["stepwise_terms"] if custom else (),
            step_method=o["step_method"], p_entry=o["p_entry"], p_removal=o["p_removal"],
            pseudo=o["pseudo"], step_summary=o["step_summary"],
            fitting_info=o["fitting_info"], info_criteria=o["info_criteria"],
            cell_probs=o["cell_probs"], monotonicity=o["monotonicity"],
            estimates=o["estimates"], correlations=o["correlations"],
            covariances=o["covariances"], max_iter=int(o["max_iter"]),
            p_converge=o["p_converge"], scale=o["scale"])


# ============================================================================ ordinal
class OrdinalDialog(VariablePickerDialog):
    DEFAULTS = {"link": "logit", "gof": True, "parallel": False, "classification": False,
                "level": 95.0, "save": [], "location": [], "scale_terms": [],
                "summary": True, "estimates": True, "correlations": False,
                "covariances": False, "cell_info": False, "iteration_history": False,
                "max_iter": 100}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Ordinal Regression", parent)
        self.add_target("dep", "Dependent:", single=True)
        self.add_target("factors", "Factor(s):", required=False)
        self.add_target("covs", "Covariate(s):", accepts=is_numeric, required=False)
        self.add_option_button("Options...", self._options)
        self.add_option_button("Output...", self._output)
        self.add_option_button("Location...", lambda: self._terms("location", "Location"))
        self.add_option_button("Scale...", lambda: self._terms("scale_terms", "Scale"))
        self.finish()

    def _terms(self, key: str, what: str) -> None:
        names = self.selected("factors") + self.selected("covs")
        dlg = TermBuilderDialog(f"Ordinal Regression: {what}", names,
                                [(key, f"{what} Model:", self.opts[key])], self)
        if dlg.exec():
            self.opts[key] = dlg.terms(key)

    def _options(self) -> None:
        dlg = SubDialog("Ordinal Regression: Options", self, columns=1)
        box = dlg.group("Iterations")
        dlg.number(box, "max_iter", "Maximum iterations:", self.opts["max_iter"], 1, 1000, 0)
        box = dlg.group("Link")
        dlg.radio(box, "link", list(logistic.LINKS.items()), self.opts["link"])
        dlg.number(dlg.group("Confidence interval"), "level", "Confidence interval (%):",
                   self.opts["level"], 50, 99.99, 2)
        values = dlg.run()
        if values:
            values["max_iter"] = int(values["max_iter"])
            self.opts.update(values)

    def _output(self) -> None:
        dlg = SubDialog("Ordinal Regression: Output", self)
        box = dlg.group("Display")
        for key, text in (("iteration_history", "Print iteration history"),
                          ("gof", "Goodness of fit statistics"),
                          ("summary", "Summary statistics"),
                          ("estimates", "Parameter estimates"),
                          ("correlations", "Asymptotic correlation of parameter estimates"),
                          ("covariances", "Asymptotic covariance of parameter estimates"),
                          ("cell_info", "Cell information"),
                          ("parallel", "Test of parallel lines"),
                          ("classification", "Classification table (predicted vs actual)")):
            dlg.check(box, key, text, self.opts[key])
        box = dlg.group("Saved Variables")
        for key, text in (("est", "Estimated response probabilities"),
                          ("pre", "Predicted category"),
                          ("max", "Predicted category probability"),
                          ("acp", "Actual category probability")):
            dlg.check(box, f"save::{key}", text, key in self.opts["save"])
        values = dlg.run()
        if values:
            self.opts["save"] = [k for k in ("est", "pre", "max", "acp")
                                 if values.pop(f"save::{k}")]
            self.opts.update(values)

    def validate(self):
        if not self.selected("factors") and not self.selected("covs"):
            return "Specify at least one factor or covariate."
        names = set(self.selected("factors") + self.selected("covs"))
        for t in self.opts["location"] + self.opts["scale_terms"]:
            if not set(t.split("*")) <= names:
                return f"The model term {t} uses a variable that is no longer selected."
        return None

    def compute(self):
        o = self.opts
        return logistic.ordinal(
            self.ds, self.selected_one("dep"), self.selected("factors"), self.selected("covs"),
            o["link"], self.level(), o["gof"], o["parallel"], o["classification"],
            tuple(o["save"]), location=o["location"] or None,
            scale_terms=o["scale_terms"], summary=o["summary"], estimates=o["estimates"],
            correlations=o["correlations"], covariances=o["covariances"],
            cell_info=o["cell_info"], iteration_history=o["iteration_history"],
            max_iter=int(o["max_iter"]))


# ============================================================================ curve estimation
class CurveEstimationDialog(VariablePickerDialog):
    DEFAULTS = {"models": ["linear"], "constant": True, "plot": True, "anova": False,
                "use_time": False, "upper": "", "save": [], "save_level": 95.0}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Curve Estimation", parent)
        self.add_target("deps", "Dependent(s):", accepts=is_numeric)
        self.add_target("ind", "Independent Variable:", accepts=is_numeric, single=True,
                        required=False)
        self.add_option_button("Save...", self._save)
        row = QHBoxLayout()
        self.var_radio = QRadioButton("Variable")
        self.time_radio = QRadioButton("Time")
        group = QButtonGroup(self)
        group.addButton(self.var_radio)
        group.addButton(self.time_radio)
        (self.time_radio if self.opts["use_time"] else self.var_radio).setChecked(True)
        self.time_radio.toggled.connect(lambda on: self.opts.__setitem__("use_time", on))
        row.addWidget(self.var_radio)
        row.addWidget(self.time_radio)
        row.addStretch()
        row.addWidget(_check(self, "constant", "Include constant in equation"))
        row.addWidget(_check(self, "plot", "Plot models"))
        self.options_layout.addLayout(row)
        box = QGroupBox("Models")
        grid = QGridLayout(box)
        self._model_checks = {}
        for i, (key, text) in enumerate(curvefit_spss.MODELS.items()):
            cb = QCheckBox(text)
            cb.setChecked(key in self.opts["models"])
            cb.toggled.connect(self._models_changed)
            self._model_checks[key] = cb
            grid.addWidget(cb, i % 4, i // 4)
        self.upper = QLineEdit(self.opts["upper"])
        self.upper.setPlaceholderText("infinity")
        self.upper.setMaximumWidth(110)
        form = QFormLayout()
        form.addRow("Upper bound:", self.upper)
        grid.addLayout(form, 3, 3)
        self.options_layout.addWidget(box)
        self.options_layout.addWidget(_check(self, "anova", "Display ANOVA table"))
        self.finish()

    def _save(self) -> None:
        def extra(dlg):
            dlg.number(dlg.group("Prediction intervals"), "save_level",
                       "Confidence Interval (%):", self.opts["save_level"], 50, 99.99, 2)

        _multi(self, "Curve Estimation: Save", "save", [
            ("Save Variables", [("pred", "Predicted values"), ("resid", "Residuals"),
                                ("cin", "Prediction intervals")])], extra)

    def _models_changed(self) -> None:
        self.opts["models"] = [k for k, cb in self._model_checks.items() if cb.isChecked()]

    def validate(self):
        if not self.opts["models"]:
            return "Select at least one model."
        if not self.opts["use_time"] and not self.selected_one("ind"):
            return "Select an independent variable, or choose Time."
        text = self.upper.text().strip().replace(",", ".")
        if text:
            try:
                float(text)
            except ValueError:
                return "Upper bound must be a number."
        self.opts["upper"] = text
        return None

    def compute(self):
        o = self.opts
        upper = float(o["upper"]) if o["upper"] else None
        return curvefit_spss.curve_estimation(
            self.ds, self.selected("deps"), None if o["use_time"] else self.selected_one("ind"),
            tuple(o["models"]), o["constant"], o["plot"], upper, o["anova"],
            save=tuple(o["save"]), level=o["save_level"] / 100)


# ============================================================================ means
class MeansDialog(VariablePickerDialog):
    """Means with unlimited layers (Layer n of m, Previous / Next) as in SPSS."""

    DEFAULTS = {"statistics": list(means.DEFAULT), "anova": False, "linearity": False,
                "layers": [[]]}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Means", parent)
        self.add_target("deps", "Dependent List:", accepts=is_numeric)
        header = QHBoxLayout()
        self.layer_label = QLabel()
        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.prev_button.clicked.connect(lambda: self._go(-1))
        self.next_button.clicked.connect(lambda: self._go(1))
        header.addWidget(self.layer_label)
        header.addStretch()
        header.addWidget(self.prev_button)
        header.addWidget(self.next_button)
        self._targets_column.addLayout(header)
        self.add_target("layer", "Independent List:", required=False)
        self.add_option_button("Options...", self._options)
        self._layers = [[n for n in names if ds.has_name(n)]
                        for names in self.opts.get("layers") or [[]]] or [[]]
        self._li = 0
        self.finish()
        self._load()

    def _layer_target(self):
        return next(t for t in self._targets if t.key == "layer")

    def _load(self) -> None:
        target = self._layer_target()
        self._move_back(target, all_items=True)
        for name in self._layers[self._li]:
            self._take(name, target)
        self._update_arrows()

    def _go(self, step: int) -> None:
        self._layers[self._li] = self.selected("layer")
        target = self._li + step
        if target < 0:
            return
        if target == len(self._layers):
            if not self._layers[self._li]:
                return
            self._layers.append([])
        while len(self._layers) - 1 > target and not self._layers[-1]:
            self._layers.pop()
        self._li = target
        self._load()

    def targets_changed(self) -> None:
        if not hasattr(self, "_layers"):
            return
        total = max(len(self._layers), self._li + 1)
        self.layer_label.setText(f"Layer {self._li + 1} of {total}")
        self.prev_button.setEnabled(self._li > 0)
        self.next_button.setEnabled(self._layer_target().widget.count() > 0)

    def layers(self) -> list[list[str]]:
        self._layers[self._li] = self.selected("layer")
        return [lay for lay in self._layers if lay]

    def _options(self) -> None:
        dlg = SubDialog("Means: Options", self)
        box = dlg.group("Cell Statistics")
        for key, text in means.STATISTICS.items():
            dlg.check(box, key, text, key in self.opts["statistics"])
        box = dlg.group("Statistics for First Layer")
        dlg.check(box, "anova", "Anova table and eta", self.opts["anova"])
        dlg.check(box, "linearity", "Test for linearity", self.opts["linearity"])
        values = dlg.run()
        if values:
            self.opts["statistics"] = [k for k in means.STATISTICS if values.get(k)]
            self.opts["anova"] = values["anova"] or values["linearity"]
            self.opts["linearity"] = values["linearity"]

    def validate(self):
        self.opts["layers"] = self.layers() or [[]]
        return None

    def compute(self):
        o = self.opts
        return means.means(self.ds, self.selected("deps"), self.layers(),
                           tuple(o["statistics"]), o["anova"], o["linearity"])


# ============================================================================ ROC
class RocDialog(VariablePickerDialog):
    DEFAULTS = {"plot": True, "reference_line": True, "se_ci": True, "coordinates": False,
                "larger_positive": True, "distribution": "free", "level": 95.0,
                "youden": False, "positive": ""}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "ROC Curve", parent)
        self.add_target("tests", "Test Variable:", accepts=is_numeric)
        self.add_target("state", "State Variable:", single=True)
        self.positive = QLineEdit(self.opts["positive"])
        form = QFormLayout()
        form.addRow("Value of State Variable:", self.positive)
        self.options_layout.addLayout(form)
        box = QGroupBox("Display")
        grid = QGridLayout(box)
        grid.addWidget(_check(self, "plot", "ROC Curve"), 0, 0)
        grid.addWidget(_check(self, "reference_line", "With diagonal reference line"), 0, 1)
        grid.addWidget(_check(self, "se_ci", "Standard error and confidence interval"), 1, 0)
        grid.addWidget(_check(self, "coordinates", "Coordinate points of the ROC Curve"), 1, 1)
        self.options_layout.addWidget(box)
        self.add_option_button("Options...", self._options)
        self.finish()

    def targets_changed(self) -> None:
        if not hasattr(self, "positive"):
            return
        state = self.selected_one("state")
        if state and not self.positive.text().strip():
            var = self.ds.var(state)
            codes = list(var.value_labels)
            if len(codes) >= 2:
                self.positive.setText(format_code(max(codes)))

    def _options(self) -> None:
        dlg = SubDialog("ROC Curve: Options", self)
        box = dlg.group("Test Direction")
        dlg.radio(box, "larger_positive", [
            (True, "Larger test result indicates more positive test"),
            (False, "Smaller test result indicates more positive test")],
            self.opts["larger_positive"])
        box = dlg.group("Parameters for Standard Error of Area")
        dlg.radio(box, "distribution", [("free", "Nonparametric (DeLong)"),
                                        ("negexpo", "Bi-negative exponential "
                                                    "(Hanley-McNeil)")],
                  self.opts["distribution"])
        dlg.number(box, "level", "Confidence level (%):", self.opts["level"], 50, 99.99, 2)
        box = dlg.group("Extra")
        dlg.check(box, "youden", "Optimal cut-off (Youden index)", self.opts["youden"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def validate(self):
        text = self.positive.text().strip()
        if not text:
            return "Enter the Value of State Variable that indicates a positive state."
        var = self.ds.var(self.selected_one("state"))
        if var.type == NUMERIC:
            try:
                float(text.replace(",", "."))
            except ValueError:
                return "Value of State Variable must be a number."
        self.opts["positive"] = text
        return None

    def compute(self):
        o = self.opts
        return roc.roc(self.ds, self.selected("tests"), self.selected_one("state"),
                       o["positive"], o["larger_positive"], o["distribution"], o["plot"],
                       o["reference_line"], o["se_ci"], o["coordinates"], self.level(),
                       o["youden"])


# ============================================================================ registry
def _menus() -> dict[str, list]:
    from rsstatchem.ui.dialogs.classify import (DiscriminantDialog, HierarchicalDialog,
                                                KMeansDialog)

    return {
        "Analyze/Compare Means": [("Means...", MeansDialog)],
        "Analyze/Regression": [None, ("Binary Logistic...", BinaryLogisticDialog),
                               ("Multinomial Logistic...", MultinomialDialog),
                               ("Ordinal...", OrdinalDialog),
                               ("Curve Estimation...", CurveEstimationDialog)],
        "Analyze/Classify": [("K-Means Cluster...", KMeansDialog),
                             ("Hierarchical Cluster...", HierarchicalDialog),
                             ("Discriminant...", DiscriminantDialog), None,
                             ("ROC Curve...", RocDialog)],
    }


MENUS = _menus()


def selftest() -> None:
    """Headless check of the logistic, curve estimation, means, ROC and classify cores."""
    import numpy as np
    import pandas as pd

    from rsstatchem.core.dataset import Dataset
    from rsstatchem.core.stats import classify

    rng = np.random.default_rng(7)
    n = 120
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    g = rng.integers(1, 4, n).astype(float)
    eta = 0.8 * x1 - 0.6 * x2 + 0.5 * (g == 2)
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    o = np.digitize(eta + rng.logistic(size=n), [-1.0, 0.0, 1.0]) + 1.0
    ds = Dataset(pd.DataFrame({"y": y, "x1": x1, "x2": x2, "g": g, "o": o,
                               "t": np.arange(1, n + 1, dtype=float),
                               "z": np.exp(0.02 * np.arange(1, n + 1)) + rng.random(n)}))
    res = logistic.binary_logistic(ds, "y", ["x1", "x2", "g"], {"g": ("indicator", "first")},
                                   method="fstep_lr", save=("prob", "group"))
    assert res.saved and any(t.title.startswith("Model Summary") for t in res.section.items
                             if hasattr(t, "rows"))
    logistic.binary_logistic(ds, "y", ["x1", "g"], {"g": ("deviation", "last")},
                             method="bstep_wald", iteration_history=True, ci=True)
    logistic.multinomial(ds, "o", ["g"], ["x1", "x2"], save=("pre",))
    logistic.ordinal(ds, "o", ["g"], ["x1", "x2"], link="cloglog", parallel=True)
    curvefit_spss.curve_estimation(ds, ["z"], "t", tuple(curvefit_spss.MODELS))
    means.means(ds, ["x1", "x2"], [["g"], ["y"]], ("mean", "count", "stddev", "median"),
                anova=True, linearity=True)
    sec = roc.roc(ds, ["x1", "x2"], "y", 1.0, coordinates=True)
    assert any(getattr(i, "kind", "") == "image" for i in sec.items)
    classify.kmeans_cluster(ds, ["x1", "x2"], 3, save=("cluster", "distance"))
    classify.hierarchical(ds, ["x1", "x2"], "ward", "seuclid", "z", solution=(2, 4))
    classify.discriminant(ds, "g", 1, 3, ["x1", "x2"], method="stepwise", f_enter=0.5,
                          f_remove=0.2, fisher=True)

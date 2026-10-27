"""Analyze > Correlate (Bivariate, Partial) and Analyze > Regression > Linear dialogs.

The Linear Regression dialog follows SPSS: Dependent, Block n of m with Previous / Next and a
Method per block, Selection Variable with Rule..., Case Labels, WLS Weight, and the
Statistics / Plots / Save / Options sub-dialogs (plus the classical assumption tests).
"""
from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QPushButton, QVBoxLayout)

from rsstatchem.core.dataset import NUMERIC, format_code
from rsstatchem.core.stats import correlate, regression
from rsstatchem.ui.dialogs.base import SubDialog, VariablePickerDialog, is_numeric


def _check(owner, key, text):
    cb = QCheckBox(text)
    cb.setChecked(bool(owner.opts[key]))
    cb.toggled.connect(lambda on, k=key: owner.opts.__setitem__(k, on))
    return cb


class BivariateDialog(VariablePickerDialog):
    DEFAULTS = {"pearson": True, "kendall": False, "spearman": False, "two_tailed": True,
                "flag": True, "descriptives": False, "listwise": False, "xprod": False,
                "ci": False, "ci_level": 95.0, "bias": False, "rank_se": "fieller"}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Bivariate Correlations", parent)
        self.add_target("vars", "Variables:", accepts=is_numeric, min_items=2)
        self.add_option_button("Options...", self._options)
        self.add_option_button("Confidence interval...", self._ci)
        box = QGroupBox("Correlation Coefficients")
        row = QHBoxLayout(box)
        for key, text in (("pearson", "Pearson"), ("kendall", "Kendall's tau-b"),
                          ("spearman", "Spearman")):
            row.addWidget(_check(self, key, text))
        self.options_layout.addWidget(box)
        box = QGroupBox("Test of Significance")
        row = QHBoxLayout(box)
        self.tails = QComboBox()
        self.tails.addItems(["Two-tailed", "One-tailed"])
        self.tails.setCurrentIndex(0 if self.opts["two_tailed"] else 1)
        self.tails.currentIndexChanged.connect(
            lambda i: self.opts.__setitem__("two_tailed", i == 0))
        row.addWidget(self.tails)
        row.addWidget(_check(self, "flag", "Flag significant correlations"))
        row.addStretch()
        self.options_layout.addWidget(box)
        self.finish()

    def _options(self) -> None:
        dlg = SubDialog("Bivariate Correlations: Options", self, columns=1)
        box = dlg.group("Statistics")
        dlg.check(box, "descriptives", "Means and standard deviations",
                  self.opts["descriptives"])
        dlg.check(box, "xprod", "Cross-product deviations and covariances", self.opts["xprod"])
        dlg.radio(dlg.group("Missing Values"), "listwise",
                  [(False, "Exclude cases pairwise"), (True, "Exclude cases listwise")],
                  self.opts["listwise"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def _ci(self) -> None:
        dlg = SubDialog("Bivariate Correlations: Confidence Interval", self, columns=1)
        box = dlg.group("Pearson and rank correlations")
        dlg.check(box, "ci", "Estimate confidence interval of bivariate correlation parameter",
                  self.opts["ci"])
        dlg.number(box, "ci_level", "Confidence Interval (%):", self.opts["ci_level"], 50,
                   99.99, 2)
        dlg.check(box, "bias", "Apply bias adjustment (Pearson)", self.opts["bias"])
        box = dlg.group("Spearman's rho standard error")
        dlg.radio(box, "rank_se", [(k, v) for k, v in correlate.RANK_CI_METHODS.items()],
                  self.opts["rank_se"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def validate(self):
        if not (self.opts["pearson"] or self.opts["kendall"] or self.opts["spearman"]):
            return "Select at least one correlation coefficient."
        return None

    def compute(self):
        o = self.opts
        return correlate.bivariate(self.ds, self.selected("vars"), o["pearson"], o["kendall"],
                                   o["spearman"], o["two_tailed"], o["flag"], o["descriptives"],
                                   o["listwise"], cross_products=o["xprod"], ci=o["ci"],
                                   ci_level=o["ci_level"] / 100, bias_adjust=o["bias"],
                                   rank_se=o["rank_se"])


class PartialDialog(VariablePickerDialog):
    DEFAULTS = {"two_tailed": True, "flag": True, "zero_order": False, "descriptives": False,
                "listwise": True}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Partial Correlations", parent)
        self.add_target("vars", "Variables:", accepts=is_numeric, min_items=2)
        self.add_target("controls", "Controlling for:", accepts=is_numeric)
        self.add_option_button("Options...", self._options)
        box = QGroupBox("Test of Significance")
        row = QHBoxLayout(box)
        tails = QComboBox()
        tails.addItems(["Two-tailed", "One-tailed"])
        tails.setCurrentIndex(0 if self.opts["two_tailed"] else 1)
        tails.currentIndexChanged.connect(lambda i: self.opts.__setitem__("two_tailed", i == 0))
        row.addWidget(tails)
        row.addWidget(_check(self, "flag", "Display actual significance level"))
        row.addStretch()
        self.options_layout.addWidget(box)
        self.finish()

    def _options(self) -> None:
        dlg = SubDialog("Partial Correlations: Options", self, columns=1)
        box = dlg.group("Statistics")
        dlg.check(box, "descriptives", "Means and standard deviations",
                  self.opts["descriptives"])
        dlg.check(box, "zero_order", "Zero-order correlations", self.opts["zero_order"])
        dlg.radio(dlg.group("Missing Values"), "listwise",
                  [(True, "Exclude cases listwise"), (False, "Exclude cases pairwise")],
                  self.opts["listwise"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def compute(self):
        o = self.opts
        return correlate.partial(self.ds, self.selected("vars"), self.selected("controls"),
                                 o["two_tailed"], o["flag"], o["zero_order"],
                                 o["descriptives"], o["listwise"])


# ============================================================================ Linear
class _RuleDialog(QDialog):
    def __init__(self, var, relation: str, value: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Linear Regression: Set Rule")
        self.relation = QComboBox()
        for key, (_, text) in regression.RELATIONS.items():
            self.relation.addItem(text, key)
        self.relation.setCurrentIndex(max(self.relation.findData(relation), 0))
        self.value = QLineEdit(value)
        form = QFormLayout()
        form.addRow(QLabel(f"Define Selection Rule for {var.name}"))
        form.addRow(self.relation)
        form.addRow("Value:", self.value)
        if var.value_labels:
            hint = QLabel(", ".join(f"{format_code(k)} = {v}"
                                    for k, v in var.value_labels.items()))
            hint.setWordWrap(True)
            hint.setStyleSheet("color:#666")
            form.addRow(hint)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Continue")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(buttons)


class LinearRegressionDialog(VariablePickerDialog):
    DEFAULTS = {"method": "enter", "blocks": [["enter", []]],
                "statistics": list(regression.DEFAULT_STATS),
                "plots": ["histogram", "pp", "scatter"], "pairs": [], "partial_plots": False,
                "save": [], "assumptions": list(regression.ASSUMPTIONS), "level": 95.0,
                "save_level": 95.0, "p_enter": 0.05, "p_remove": 0.10, "use_f": False,
                "f_enter": 3.84, "f_remove": 2.71, "constant": True, "missing": "listwise",
                "casewise_all": False, "casewise_sd": 3.0, "rule": ["eq", ""]}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Linear Regression", parent)
        self.setMinimumSize(660, 520)
        self.add_target("dep", "Dependent:", accepts=is_numeric, single=True)
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
        self.ind_list = self.add_target("ind", "Independent(s):", accepts=is_numeric,
                                        required=False)
        self.method = QComboBox()
        for key in regression.METHODS:
            self.method.addItem(regression.METHOD_NAMES[key], key)
        self.method.currentIndexChanged.connect(self._method_changed)
        form = QFormLayout()
        form.addRow("Method:", self.method)
        self._targets_column.addLayout(form)
        self.add_target("sel", "Selection Variable:", single=True, required=False)
        rule_row = QHBoxLayout()
        rule_row.addStretch()
        self.rule_button = QPushButton("Rule...")
        self.rule_button.clicked.connect(self._rule)
        rule_row.addWidget(self.rule_button)
        self._targets_column.addLayout(rule_row)
        self.add_target("labels", "Case Labels:", single=True, required=False)
        self.add_target("wls", "WLS Weight:", accepts=is_numeric, single=True, required=False)
        self.add_option_button("Statistics...", self._statistics)
        self.add_option_button("Plots...", self._plots)
        self.add_option_button("Save...", self._save)
        self.add_option_button("Options...", self._options)
        self.add_option_button("Assumptions...", self._assumptions)
        blocks = [[m, [n for n in names if ds.has_name(n)]]
                  for m, names in self.opts.get("blocks") or [["enter", []]]]
        self._blocks = blocks or [["enter", []]]
        self._bi = 0
        self.finish()
        self._load_block()

    # ------------------------------------------------------------- blocks
    def _store_block(self) -> None:
        self._blocks[self._bi] = [self.method.currentData(), self.selected("ind")]

    def _load_block(self) -> None:
        self._move_back(self._target("ind"), all_items=True)
        method, names = self._blocks[self._bi]
        target = self._target("ind")
        for name in names:
            self._take(name, target)
        self.method.blockSignals(True)
        self.method.setCurrentIndex(max(self.method.findData(method), 0))
        self.method.blockSignals(False)
        self._update_block_controls()
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
        while len(self._blocks) > 1 and not self._blocks[-1][1] and len(self._blocks) - 1 > \
                target:
            self._blocks.pop()
        self._bi = target
        self._load_block()

    def _target(self, key):
        return next(t for t in self._targets if t.key == key)

    def _method_changed(self) -> None:
        self.opts["method"] = self.method.currentData()

    def _update_block_controls(self) -> None:
        if not hasattr(self, "block_label"):
            return
        total = max(len(self._blocks), self._bi + 1)
        self.block_label.setText(f"Block {self._bi + 1} of {total}")
        self.prev_button.setEnabled(self._bi > 0)
        self.next_button.setEnabled(self._target("ind").widget.count() > 0)

    def targets_changed(self) -> None:
        if hasattr(self, "_blocks"):
            self._update_block_controls()
        if hasattr(self, "rule_button"):
            self.rule_button.setEnabled(bool(self.selected("sel")))

    def _update_ok(self) -> None:
        if not hasattr(self, "ok_button") or not hasattr(self, "_blocks"):
            return
        has_ind = self._target("ind").widget.count() > 0 or \
            any(names for i, (_, names) in enumerate(self._blocks) if i != self._bi)
        self.ok_button.setEnabled(bool(self.selected("dep")) and has_ind)

    def blocks(self) -> list[tuple[str, list[str]]]:
        self._store_block()
        return [(m, list(n)) for m, n in self._blocks if n]

    # ------------------------------------------------------------- sub-dialogs
    def _rule(self) -> None:
        name = self.selected_one("sel")
        if not name:
            return
        rel, val = self.opts["rule"]
        dlg = _RuleDialog(self.ds.var(name), rel, val, self)
        if dlg.exec():
            self.opts["rule"] = [dlg.relation.currentData(), dlg.value.text().strip()]

    def _multi(self, title, key, groups, extra=None, columns=2):
        dlg = SubDialog(title, self, columns=columns)
        for group_title, items in groups:
            box = dlg.group(group_title)
            for item_key, text in items:
                dlg.check(box, item_key, text, item_key in self.opts[key])
        if extra:
            extra(dlg)
        values = dlg.run()
        if values is None:
            return None
        keys = [k for _, items in groups for k, _ in items]
        self.opts[key] = [k for k in keys if values.get(k)]
        for k, v in values.items():
            if k not in keys:
                self.opts[k] = v
        return values

    def _statistics(self) -> None:
        def extra(dlg):
            box = dlg.group("Confidence Intervals / Casewise")
            dlg.number(box, "level", "Level (%):", self.opts["level"], 50, 99.99, 2)
            dlg.radio(box, "casewise_all", [(False, "Outliers outside:"), (True, "All cases")],
                      self.opts["casewise_all"])
            dlg.number(box, "casewise_sd", "standard deviations", self.opts["casewise_sd"],
                       0.5, 20, 1)

        self._multi("Linear Regression: Statistics", "statistics", [
            ("Regression Coefficients", [("estimates", "Estimates"),
                                         ("ci", "Confidence intervals"),
                                         ("bcov", "Covariance matrix")]),
            ("Model", [("model_fit", "Model fit"), ("r2_change", "R squared change"),
                       ("descriptives", "Descriptives"),
                       ("part_partial", "Part and partial correlations"),
                       ("collinearity", "Collinearity diagnostics"),
                       ("press", "PRESS statistic")]),
            ("Residuals", [("durbin_watson", "Durbin-Watson"),
                           ("casewise", "Casewise diagnostics")]),
        ], extra)

    def _plots(self) -> None:
        dlg = SubDialog("Linear Regression: Plots", self)
        box = dlg.group("Scatter")
        ycombo, xcombo = QComboBox(), QComboBox()
        for key in regression.PLOT_VARS:
            ycombo.addItem(key)
            xcombo.addItem(key)
        ycombo.setCurrentText("*ZRESID")
        xcombo.setCurrentText("*ZPRED")
        form = QFormLayout()
        form.addRow("Y:", ycombo)
        form.addRow("X:", xcombo)
        box.addLayout(form)
        pairs = QListWidget()
        pairs.setMaximumHeight(90)
        current = [tuple(p) for p in self.opts["pairs"]]
        if "scatter" in self.opts["plots"] and ("*ZRESID", "*ZPRED") not in current:
            current.insert(0, ("*ZRESID", "*ZPRED"))
        for yk, xk in current:
            pairs.addItem(f"{yk} vs {xk}")
        row = QHBoxLayout()
        add, remove = QPushButton("Next / Add"), QPushButton("Remove")
        add.clicked.connect(lambda: pairs.addItem(f"{ycombo.currentText()} vs "
                                                  f"{xcombo.currentText()}"))
        remove.clicked.connect(lambda: [pairs.takeItem(pairs.row(i))
                                        for i in pairs.selectedItems()])
        row.addWidget(add)
        row.addWidget(remove)
        box.addLayout(row)
        box.addWidget(pairs)
        box = dlg.group("Standardized Residual Plots")
        dlg.check(box, "histogram", "Histogram", "histogram" in self.opts["plots"])
        dlg.check(box, "pp", "Normal probability plot", "pp" in self.opts["plots"])
        dlg.check(box, "partial_plots", "Produce all partial plots", self.opts["partial_plots"])
        values = dlg.run()
        if values is None:
            return
        out = []
        for i in range(pairs.count()):
            yk, xk = pairs.item(i).text().split(" vs ")
            if (yk, xk) not in out:
                out.append((yk, xk))
        self.opts["pairs"] = [list(p) for p in out if p != ("*ZRESID", "*ZPRED")]
        self.opts["plots"] = [k for k in ("histogram", "pp") if values[k]] + \
            (["scatter"] if ("*ZRESID", "*ZPRED") in out else [])
        self.opts["partial_plots"] = values["partial_plots"]

    def _save(self) -> None:
        def extra(dlg):
            dlg.number(dlg.group("Prediction Intervals level"), "save_level",
                       "Confidence Interval (%):", self.opts["save_level"], 50, 99.99, 2)

        self._multi("Linear Regression: Save", "save", [
            ("Predicted Values", [("pred", "Unstandardized"), ("zpred", "Standardized"),
                                  ("adjpred", "Adjusted"),
                                  ("sepred", "S.E. of mean predictions")]),
            ("Residuals", [("resid", "Unstandardized"), ("zresid", "Standardized"),
                           ("sresid", "Studentized"), ("dresid", "Deleted"),
                           ("sdresid", "Studentized deleted")]),
            ("Distances", [("mahal", "Mahalanobis"), ("cook", "Cook's"),
                           ("lever", "Leverage values")]),
            ("Influence Statistics", [("dfbeta", "DfBeta(s)"),
                                      ("sdfbeta", "Standardized DfBeta(s)"),
                                      ("dffit", "DfFit"), ("sdffit", "Standardized DfFit"),
                                      ("covratio", "Covariance ratio")]),
            ("Prediction Intervals", [("mcin", "Mean"), ("icin", "Individual")]),
        ], extra)

    def _assumptions(self) -> None:
        self._multi("Linear Regression: Classical Assumption Tests", "assumptions", [
            ("Tests", [("normality", "Normality of residuals (K-S, Shapiro-Wilk)"),
                       ("multicollinearity", "Multicollinearity (Tolerance, VIF)"),
                       ("heteroscedasticity", "Heteroscedasticity (Glejser, Breusch-Pagan, "
                                              "White)"),
                       ("autocorrelation", "Autocorrelation (Durbin-Watson, Breusch-Godfrey, "
                                           "Runs test)")]),
        ], columns=1)

    def _options(self) -> None:
        dlg = SubDialog("Linear Regression: Options", self, columns=1)
        box = dlg.group("Stepping Method Criteria")
        dlg.radio(box, "use_f", [(False, "Use probability of F"), (True, "Use F value")],
                  self.opts["use_f"])
        dlg.number(box, "p_enter", "Probability Entry:", self.opts["p_enter"], 0.0001, 0.99, 4)
        dlg.number(box, "p_remove", "Probability Removal:", self.opts["p_remove"], 0.0001,
                   0.99, 4)
        dlg.number(box, "f_enter", "F Entry:", self.opts["f_enter"], 0.0001, 1000, 2)
        dlg.number(box, "f_remove", "F Removal:", self.opts["f_remove"], 0.0001, 1000, 2)
        dlg.check(dlg.group("Constant"), "constant", "Include constant in equation",
                  self.opts["constant"])
        dlg.radio(dlg.group("Missing Values"), "missing", [
            ("listwise", "Exclude cases listwise"), ("pairwise", "Exclude cases pairwise"),
            ("mean", "Replace with mean")], self.opts["missing"])
        values = dlg.run()
        if values:
            if values["p_enter"] >= values["p_remove"]:
                values["p_remove"] = min(0.99, values["p_enter"] + 0.05)
            if values["f_enter"] <= values["f_remove"]:
                values["f_remove"] = max(0.0001, values["f_enter"] - 1)
            self.opts.update(values)

    # ------------------------------------------------------------- run
    def validate(self):
        blocks = self.blocks()
        if not blocks:
            return "Specify at least one independent variable."
        dep = self.selected_one("dep")
        if any(dep in names for _, names in blocks):
            return "The dependent variable cannot also be an independent variable."
        if self.selected("sel"):
            rel, val = self.opts["rule"]
            if not val:
                return "Define the selection rule (Rule...)."
            if self.ds.var(self.selected_one("sel")).type == NUMERIC:
                try:
                    float(val.replace(",", "."))
                except ValueError:
                    return "The selection rule value must be a number."
        self.opts["blocks"] = [[m, n] for m, n in blocks]
        return None

    def compute(self):
        o = self.opts
        selection = None
        sel = self.selected_one("sel")
        if sel:
            rel, val = o["rule"]
            value = float(val.replace(",", ".")) if self.ds.var(sel).type == NUMERIC else val
            selection = (sel, rel, value)
        return regression.linear(
            self.ds, self.selected_one("dep"), None, statistics=tuple(o["statistics"]),
            plots=tuple(o["plots"]), save=tuple(o["save"]),
            assumptions=tuple(o["assumptions"]), level=self.level(),
            pi_level=o["save_level"] / 100, p_enter=o["p_enter"], p_remove=o["p_remove"],
            blocks=[(m, n) for m, n in o["blocks"]], selection=selection,
            case_labels=self.selected_one("labels"), wls=self.selected_one("wls"),
            use_f=o["use_f"], f_enter=o["f_enter"], f_remove=o["f_remove"],
            constant=o["constant"], missing=o["missing"], casewise_sd=o["casewise_sd"],
            casewise_all=o["casewise_all"], plot_pairs=[tuple(p) for p in o["pairs"]],
            partial_plots=o["partial_plots"])

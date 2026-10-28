"""Data > Propensity Score Matching, Case Control Matching and Rake Weights (SPSS built-in
extensions). Core code: :mod:`rsstatchem.core.stats.matching`."""
from __future__ import annotations

from PySide6.QtWidgets import QFormLayout, QGridLayout, QLabel, QLineEdit, QPushButton

from rsstatchem.core.dataset import NUMERIC, is_valid_name
from rsstatchem.core.stats import matching
from rsstatchem.core.stats.common import value_text
from rsstatchem.core.stats.ml_common import raw_column, sort_levels
from rsstatchem.ui.dialogs.base import VariablePickerDialog, is_numeric
from rsstatchem.ui.dialogs.ml_tabs import OptionsDialog


def _name_row(dialog, label: str, key: str) -> QLineEdit:
    edit = QLineEdit(str(dialog.opts.get(key) or ""))
    edit.textChanged.connect(lambda t: dialog.opts.__setitem__(key, t.strip()))
    row = QFormLayout()
    row.addRow(label, edit)
    dialog.options_layout.addLayout(row)
    return edit


def _check_names(ds, names: list[str]) -> str | None:
    seen = set()
    for n in names:
        if not n:
            continue
        if not is_valid_name(n):
            return f"'{n}' is not a valid variable name."
        if ds.has_name(n):
            return f"A variable named {n} already exists."
        if n.lower() in seen:
            return "The new variable names must be different."
        seen.add(n.lower())
    return None


class PropensityMatchingDialog(VariablePickerDialog):
    modifies_data = True
    DEFAULTS = {"ps_name": "PropensityScore", "tolerance": 0.02, "match_name": "MatchID",
                "eligible_name": "", "replace": False, "exact_first": True, "order": "data",
                "seed": 2000000, "scale": "ps", "ratio": 1, "logistic_output": True,
                "balance": True}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Propensity Score Matching", parent)
        self.add_target("group", "Group Indicator (1 = treatment, 0 = control):",
                        accepts=is_numeric, single=True)
        self.add_target("pred", "Predictors:")
        self.add_target("id", "Case ID (optional):", single=True, required=False)
        _name_row(self, "Name for Propensity Variable:", "ps_name")
        _name_row(self, "Match ID Variable Name:", "match_name")
        self.add_option_button("Options...", self._options)
        self.finish()

    def _options(self):
        dlg = OptionsDialog("Propensity Score Matching: Options", self.opts, self)
        g = dlg.group("Matching")
        g.number("tolerance", "Match tolerance:", 0, 1e6, 5)
        g.radio("scale", [("ps", "Tolerance on the propensity score"),
                          ("logit", "Tolerance on the logit of the score"),
                          ("sd", "Tolerance in standard deviations of the score")])
        g.number("ratio", "Controls per treated case:", 1, 100)
        g.check("replace", "Sample with replacement")
        g.check("exact_first", "Give priority to exact matches")
        g = dlg.group("Case Order")
        g.radio("order", [("data", "Treated cases in file order"),
                          ("largest", "Largest propensity score first"),
                          ("smallest", "Smallest propensity score first"),
                          ("random", "Randomize case order when drawing matches")])
        g.number("seed", "Random seed:", 1, 2 ** 31 - 1)
        g.text("eligible_name", "Variable for number of eligible cases (optional):")
        g.check("logistic_output", "Display the logistic regression")
        g.check("balance", "Display balance diagnostics")
        dlg.run(self.opts)

    def validate(self):
        o = self.opts
        if self.selected_one("group") in self.selected("pred"):
            return "The group indicator cannot be a predictor."
        if not o["ps_name"] or not o["match_name"]:
            return "Enter names for the propensity and match ID variables."
        return _check_names(self.ds, [o["ps_name"], o["match_name"], o["eligible_name"]])

    def compute(self):
        o = self.opts
        return matching.propensity_matching(
            self.ds, self.selected_one("group"), self.selected("pred"), o["ps_name"],
            float(o["tolerance"]), self.selected_one("id"), o["match_name"],
            o["eligible_name"] or None, bool(o["replace"]), int(o["ratio"]),
            bool(o["exact_first"]), o["order"], int(o["seed"]), o["scale"],
            bool(o["logistic_output"]), bool(o["balance"]))


class CaseControlDialog(VariablePickerDialog):
    modifies_data = True
    DEFAULTS = {"tolerances": "", "match_name": "MatchID", "group_name": "MatchGroup",
                "ratio": 1, "replace": False, "randomize": True, "seed": 2000000}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Case Control Matching", parent)
        self.add_target("group", "Case Indicator (1 = case, 0 = control):", accepts=is_numeric,
                        single=True)
        self.add_target("keys", "Match Keys:")
        self.add_target("id", "Case ID (optional):", single=True, required=False)
        _name_row(self, "Fuzz tolerances (e.g. age=2; income=500):", "tolerances")
        _name_row(self, "Match ID Variable Name:", "match_name")
        _name_row(self, "Match Group Variable Name:", "group_name")
        self.add_option_button("Options...", self._options)
        self.finish()

    def _options(self):
        dlg = OptionsDialog("Case Control Matching: Options", self.opts, self, columns=1)
        g = dlg.group("Options")
        g.number("ratio", "Maximum number of controls per case:", 1, 100)
        g.check("replace", "Sample controls with replacement")
        g.check("randomize", "Draw cases and controls at random")
        g.number("seed", "Random seed:", 1, 2 ** 31 - 1)
        dlg.run(self.opts)

    def _tolerances(self) -> dict:
        return matching.parse_targets(self.opts["tolerances"], numeric=False) \
            if self.opts["tolerances"] else {}

    def validate(self):
        try:
            tol = self._tolerances()
        except ValueError:
            return "Write the tolerances as variable=value pairs separated by semicolons."
        keys = [k.lower() for k in self.selected("keys")]
        for k in tol:
            if k.lower() not in keys:
                return f"{k} is not a match key."
        return _check_names(self.ds, [self.opts["match_name"], self.opts["group_name"]])

    def compute(self):
        o = self.opts
        tol = {self.ds.var(k).name: v for k, v in self._tolerances().items()}
        return matching.case_control_matching(
            self.ds, self.selected_one("group"), self.selected("keys"), tol,
            self.selected_one("id"), o["match_name"], o["group_name"], int(o["ratio"]),
            bool(o["replace"]), bool(o["randomize"]), int(o["seed"]))


class RakeWeightsDialog(VariablePickerDialog):
    modifies_data = True
    DEFAULTS = {"targets": {}, "weight_name": "rakewt", "proportions": True, "total": 0.0,
                "max_iter": 1000, "tol": 1e-6, "trim": False, "trim_lo": 0.3, "trim_hi": 3.0}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Rake Weights", parent)
        self.add_target("vars", "Control Variables:")
        self.add_target("initial", "Initial Weight (optional):", accepts=is_numeric,
                        single=True, required=False)
        grid = QGridLayout()
        self.cur = QLabel("Targets for: (select a control variable)")
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("value=target; value=target  (e.g. 1=0.48; 2=0.52)")
        self.edit.textChanged.connect(self._store)
        fill = QPushButton("Observed")
        fill.setToolTip("Fill in the observed proportions of the categories")
        fill.clicked.connect(self._fill)
        grid.addWidget(self.cur, 0, 0, 1, 2)
        grid.addWidget(self.edit, 1, 0)
        grid.addWidget(fill, 1, 1)
        self.options_layout.addLayout(grid)
        _name_row(self, "Name for Weight Variable:", "weight_name")
        self.add_option_button("Options...", self._options)
        for t in self._targets:
            if t.key == "vars":
                t.widget.currentItemChanged.connect(lambda *_: self._show())
        self.finish()

    def _current(self) -> str | None:
        w = next(t.widget for t in self._targets if t.key == "vars")
        it = w.currentItem() or (w.item(0) if w.count() else None)
        from rsstatchem.ui.dialogs.base import NAME_ROLE

        return it.data(NAME_ROLE) if it else None

    def _show(self):
        v = self._current()
        self.cur.setText(f"Targets for {v}:" if v else "Targets for: (select a control "
                                                       "variable)")
        self.edit.blockSignals(True)
        self.edit.setText(self.opts["targets"].get(v, "") if v else "")
        self.edit.blockSignals(False)

    def targets_changed(self):
        if hasattr(self, "edit"):
            self._show()

    def _store(self, text):
        v = self._current()
        if v:
            self.opts["targets"][v] = text

    def _fill(self):
        v = self._current()
        if not v:
            return
        s = raw_column(self.ds, v)
        vc = s.value_counts(normalize=True)
        levels = sort_levels(s.dropna().tolist())
        self.edit.setText("; ".join(f"{value_text(self.ds.var(v), lv)}={vc[lv]:.4f}"
                                    if self.ds.var(v).type != NUMERIC
                                    else f"{lv:g}={vc[lv]:.4f}" for lv in levels))

    def _options(self):
        dlg = OptionsDialog("Rake Weights: Options", self.opts, self, columns=1)
        g = dlg.group("Targets")
        g.radio("proportions", [(True, "Targets are proportions (normalised)"),
                                (False, "Targets are population counts")])
        g.number("total", "Weighted total of the result (0 = automatic):", 0, 1e12, 2)
        g = dlg.group("Iterations")
        g.number("max_iter", "Maximum iterations:", 1, 100000)
        g.number("tol", "Convergence (relative change):", 1e-12, 1, 10)
        g = dlg.group("Trimming")
        g.check("trim", "Trim weights to a range relative to the mean weight")
        g.number("trim_lo", "Minimum ratio:", 0, 1, 3)
        g.number("trim_hi", "Maximum ratio:", 1, 1000, 3)
        dlg.run(self.opts)

    def _specs(self):
        specs = []
        for v in self.selected("vars"):
            txt = self.opts["targets"].get(v, "")
            numeric = self.ds.var(v).type == NUMERIC
            specs.append((v, matching.parse_targets(txt, numeric)))
        return specs

    def validate(self):
        try:
            specs = self._specs()
        except ValueError:
            return "Write the targets as value=target pairs separated by semicolons."
        for v, t in specs:
            if not t:
                return f"Enter the targets for {v} (select it in the Control Variables list)."
        return _check_names(self.ds, [self.opts["weight_name"]])

    def compute(self):
        o = self.opts
        return matching.rake_weights(
            self.ds, self._specs(), o["weight_name"], self.selected_one("initial"),
            float(o["total"]) or None, bool(o["proportions"]), int(o["max_iter"]),
            float(o["tol"]), (float(o["trim_lo"]), float(o["trim_hi"])) if o["trim"] else None)


MENUS = {"Data": [("Rake Weights...", RakeWeightsDialog),
                  ("Propensity Score Matching...", PropensityMatchingDialog),
                  ("Case Control Matching...", CaseControlDialog)]}


def selftest() -> None:
    matching.selftest()

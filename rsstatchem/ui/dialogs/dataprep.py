"""Data > Validation (Load Predefined Rules, Define Rules, Validate Data), Data > Identify
Unusual Cases, Transform > Optimal Binning and Transform > Prepare Data for Modeling
(SPSS Data Preparation add-on). Core code: :mod:`rsstatchem.core.stats.validation`,
:mod:`~rsstatchem.core.stats.anomaly`, :mod:`~rsstatchem.core.stats.optimal_binning`,
:mod:`~rsstatchem.core.stats.adp`."""
from __future__ import annotations

import copy

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QPushButton, QRadioButton, QTabWidget,
                               QVBoxLayout, QWidget)

from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats import adp, anomaly, optimal_binning, validation
from rsstatchem.core.stats.ml_common import measure_of
from rsstatchem.ui.dialogs.base import is_numeric
from rsstatchem.ui.dialogs.ml_tabs import TabbedPickerDialog
from rsstatchem.ui.dialogs.wp11_util import number_or_none, parse_values


# ============================================================================ Define Rules
class DefineRulesDialog(QDialog):
    """Single-variable and cross-variable validation rules (stored in the dataset)."""
    modifies_data = True

    def __init__(self, ds, parent=None):
        super().__init__(parent)
        self.ds = ds
        self.setWindowTitle("Define Validation Rules")
        self.setMinimumSize(720, 480)
        self.value = validation.rules(ds)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._single_tab(), "Single-Variable Rules")
        self.tabs.addTab(self._cross_tab(), "Cross-Variable Rules")
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(self.tabs, 1)
        lay.addWidget(buttons)
        self._refresh_single()
        self._refresh_cross()

    # ---------------------------------------------------------------- single
    def _single_tab(self) -> QWidget:
        w = QWidget()
        self.s_list = QListWidget()
        self.s_list.currentRowChanged.connect(self._load_single)
        new, dup, dele = QPushButton("New"), QPushButton("Duplicate"), QPushButton("Delete")
        new.clicked.connect(self._new_single)
        dup.clicked.connect(self._dup_single)
        dele.clicked.connect(self._del_single)
        left = QVBoxLayout()
        left.addWidget(QLabel("Rules:"))
        left.addWidget(self.s_list, 1)
        row = QHBoxLayout()
        for b in (new, dup, dele):
            row.addWidget(b)
        left.addLayout(row)
        form = QFormLayout()
        self.s_name = QLineEdit()
        self.s_type = QComboBox()
        for t in ("numeric", "string", "date"):
            self.s_type.addItem(t.capitalize(), t)
        self.s_range = QRadioButton("Within a range")
        self.s_listb = QRadioButton("In a list")
        self.s_min, self.s_max, self.s_values = QLineEdit(), QLineEdit(), QLineEdit()
        self.s_values.setPlaceholderText("values separated by spaces")
        self.s_nonint = QCheckBox("Allow noninteger values")
        self.s_user = QCheckBox("Allow user-missing values")
        self.s_sys = QCheckBox("Allow system-missing values")
        self.s_blank = QCheckBox("Allow blank values (string)")
        form.addRow("Name:", self.s_name)
        form.addRow("Type:", self.s_type)
        form.addRow(self.s_range)
        form.addRow("Minimum:", self.s_min)
        form.addRow("Maximum:", self.s_max)
        form.addRow(self.s_listb)
        form.addRow("Values:", self.s_values)
        for cb in (self.s_nonint, self.s_user, self.s_sys, self.s_blank):
            form.addRow(cb)
        for wdg in (self.s_name, self.s_min, self.s_max, self.s_values):
            wdg.editingFinished.connect(self._store_single)
        for wdg in (self.s_range, self.s_listb, self.s_nonint, self.s_user, self.s_sys,
                    self.s_blank):
            wdg.toggled.connect(lambda _=False: self._store_single())
        self.s_type.currentIndexChanged.connect(lambda _=0: self._store_single())
        box = QGroupBox("Rule Definition")
        box.setLayout(form)
        lay = QHBoxLayout(w)
        lay.addLayout(left, 1)
        lay.addWidget(box, 2)
        return w

    def _refresh_single(self, select: int | None = None):
        self.s_list.blockSignals(True)
        self.s_list.clear()
        for r in self.value["single"]:
            self.s_list.addItem(r["name"])
        self.s_list.blockSignals(False)
        if self.value["single"]:
            self.s_list.setCurrentRow(select if select is not None else 0)
        self._load_single(self.s_list.currentRow())

    def _load_single(self, row: int):
        self._loading = True
        r = self.value["single"][row] if 0 <= row < len(self.value["single"]) else None
        for wdg in (self.s_name, self.s_type, self.s_range, self.s_listb, self.s_min,
                    self.s_max, self.s_values, self.s_nonint, self.s_user, self.s_sys,
                    self.s_blank):
            wdg.setEnabled(r is not None)
        if r is not None:
            self.s_name.setText(r["name"])
            self.s_type.setCurrentIndex(["numeric", "string", "date"].index(
                r.get("type", "numeric")))
            (self.s_listb if r.get("domain") == "list" else self.s_range).setChecked(True)
            self.s_min.setText("" if r.get("min") is None else f"{r['min']:g}"
                               if isinstance(r.get("min"), (int, float)) else str(r["min"]))
            self.s_max.setText("" if r.get("max") is None else f"{r['max']:g}"
                               if isinstance(r.get("max"), (int, float)) else str(r["max"]))
            self.s_values.setText(" ".join(f"{v:g}" if isinstance(v, (int, float)) else str(v)
                                           for v in r.get("values", [])))
            self.s_nonint.setChecked(r.get("noninteger", True))
            self.s_user.setChecked(r.get("user_missing", True))
            self.s_sys.setChecked(r.get("sysmis", True))
            self.s_blank.setChecked(r.get("blank", True))
        self._loading = False

    def _store_single(self):
        if getattr(self, "_loading", False):
            return
        row = self.s_list.currentRow()
        if not 0 <= row < len(self.value["single"]):
            return
        r = self.value["single"][row]
        numeric = self.s_type.currentData() == "numeric"
        conv = (lambda t: number_or_none(t)) if numeric else (lambda t: t.strip() or None)
        try:
            r.update(name=self.s_name.text().strip() or r["name"],
                     type=self.s_type.currentData(),
                     domain="list" if self.s_listb.isChecked() else "range",
                     min=conv(self.s_min.text()), max=conv(self.s_max.text()),
                     values=parse_values(self.s_values.text()) if numeric else
                     self.s_values.text().split(),
                     noninteger=self.s_nonint.isChecked(),
                     user_missing=self.s_user.isChecked(), sysmis=self.s_sys.isChecked(),
                     blank=self.s_blank.isChecked())
        except ValueError:
            return
        self.s_list.item(row).setText(r["name"])

    def _unique(self, base: str, existing) -> str:
        names = {x["name"].lower() for x in existing}
        k, name = 1, base
        while name.lower() in names:
            k += 1
            name = f"{base} {k}"
        return name

    def _new_single(self):
        self.value["single"].append({"name": self._unique("SingleVarRule",
                                                         self.value["single"]),
                                     "type": "numeric", "domain": "range", "min": None,
                                     "max": None, "values": [], "noninteger": True,
                                     "user_missing": True, "sysmis": True, "blank": True})
        self._refresh_single(len(self.value["single"]) - 1)

    def _dup_single(self):
        row = self.s_list.currentRow()
        if 0 <= row < len(self.value["single"]):
            r = copy.deepcopy(self.value["single"][row])
            r["name"] = self._unique(r["name"], self.value["single"])
            self.value["single"].append(r)
            self._refresh_single(len(self.value["single"]) - 1)

    def _del_single(self):
        row = self.s_list.currentRow()
        if 0 <= row < len(self.value["single"]):
            del self.value["single"][row]
            self._refresh_single(max(0, row - 1))

    # ---------------------------------------------------------------- cross
    def _cross_tab(self) -> QWidget:
        w = QWidget()
        self.c_list = QListWidget()
        self.c_list.currentRowChanged.connect(self._load_cross)
        new, dele = QPushButton("New"), QPushButton("Delete")
        new.clicked.connect(self._new_cross)
        dele.clicked.connect(self._del_cross)
        left = QVBoxLayout()
        left.addWidget(QLabel("Rules:"))
        left.addWidget(self.c_list, 1)
        row = QHBoxLayout()
        row.addWidget(new)
        row.addWidget(dele)
        left.addLayout(row)
        form = QFormLayout()
        self.c_name = QLineEdit()
        self.c_expr = QLineEdit()
        self.c_expr.setPlaceholderText("e.g. pregnant = 1 AND gender = 1")
        self.c_status = QLabel()
        self.c_status.setWordWrap(True)
        self.c_status.setStyleSheet("color:#666")
        form.addRow("Name:", self.c_name)
        form.addRow("Logical Expression:", self.c_expr)
        form.addRow(QLabel("The expression evaluates to 1 (true) for cases that violate the "
                           "rule; it uses the Compute Variable language."))
        form.addRow(self.c_status)
        self.c_name.editingFinished.connect(self._store_cross)
        self.c_expr.editingFinished.connect(self._store_cross)
        box = QGroupBox("Rule Definition")
        box.setLayout(form)
        lay = QHBoxLayout(w)
        lay.addLayout(left, 1)
        lay.addWidget(box, 2)
        return w

    def _refresh_cross(self, select: int | None = None):
        self.c_list.blockSignals(True)
        self.c_list.clear()
        for r in self.value["cross"]:
            self.c_list.addItem(r["name"])
        self.c_list.blockSignals(False)
        if self.value["cross"]:
            self.c_list.setCurrentRow(select if select is not None else 0)
        self._load_cross(self.c_list.currentRow())

    def _load_cross(self, row: int):
        r = self.value["cross"][row] if 0 <= row < len(self.value["cross"]) else None
        self.c_name.setEnabled(r is not None)
        self.c_expr.setEnabled(r is not None)
        self.c_name.setText(r["name"] if r else "")
        self.c_expr.setText(r["expression"] if r else "")

    def _store_cross(self):
        row = self.c_list.currentRow()
        if not 0 <= row < len(self.value["cross"]):
            return
        r = self.value["cross"][row]
        r["name"] = self.c_name.text().strip() or r["name"]
        r["expression"] = self.c_expr.text().strip()
        self.c_list.item(row).setText(r["name"])
        from rsstatchem.core.expression import check

        err = check(self.ds, r["expression"]) if r["expression"] else None
        self.c_status.setText(err or "")

    def _new_cross(self):
        self.value["cross"].append({"name": self._unique("CrossVarRule", self.value["cross"]),
                                    "expression": ""})
        self._refresh_cross(len(self.value["cross"]) - 1)

    def _del_cross(self):
        row = self.c_list.currentRow()
        if 0 <= row < len(self.value["cross"]):
            del self.value["cross"][row]
            self._refresh_cross(max(0, row - 1))

    # ---------------------------------------------------------------- result
    def validate(self) -> str | None:
        from rsstatchem.core.expression import check

        names = [r["name"].lower() for r in self.value["single"] + self.value["cross"]]
        if len(set(names)) != len(names):
            return "Rule names must be unique."
        for r in self.value["cross"]:
            if not r["expression"]:
                return f"Rule {r['name']} has no expression."
            err = check(self.ds, r["expression"])
            if err:
                return f"Rule {r['name']}: {err}"
        return None

    def accept(self):
        self._store_single()
        self._store_cross()
        err = self.validate()
        if err:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(self, self.windowTitle(), err)
            return
        super().accept()

    def compute(self):
        validation.set_rules(self.ds, self.value)
        section = Section("Define Validation Rules")
        section.log("* Validation rules updated in the data dictionary.")
        section.table(Table("Single-Variable Rules", ["Rule", "Description"],
                            [[r["name"], validation.describe_rule(r)]
                             for r in self.value["single"]] or [["(none)", ""]]))
        section.table(Table("Cross-Variable Rules", ["Rule", "Expression"],
                            [[r["name"], r["expression"]] for r in self.value["cross"]]
                            or [["(none)", ""]]))
        return section


class LoadPredefinedRulesDialog(QDialog):
    modifies_data = True

    def __init__(self, ds, parent=None):
        super().__init__(parent)
        self.ds = ds
        self.setWindowTitle("Load Predefined Validation Rules")
        self.lst = QListWidget()
        for r in validation.PREDEFINED:
            it = QListWidgetItem(f"{r['name']}  —  {validation.describe_rule(r)}")
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked)
            it.setData(Qt.ItemDataRole.UserRole, r["name"])
            self.lst.addItem(it)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Rules to add to the data dictionary:"))
        lay.addWidget(self.lst, 1)
        lay.addWidget(buttons)
        self.resize(560, 420)

    def compute(self):
        names = [self.lst.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.lst.count())
                 if self.lst.item(i).checkState() == Qt.CheckState.Checked]
        return validation.load_predefined(self.ds, names)


# ============================================================================ Validate Data
class ValidateDataDialog(TabbedPickerDialog):
    modifies_data = True
    DEFAULTS = {"basic": True, **{k: v for k, v in validation.BASIC_DEFAULTS.items()},
                "id_checks": True, "empty": True, "assign": {}, "cross": [],
                "min_violations": 1, "limit": 500, "save_empty": False,
                "save_incomplete": False, "save_duplicate": False, "save_total": False,
                "save_indicators": False, "replace": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Validate Data", parent)
        self.add_target("vars", "Analysis Variables:", required=False)
        self.add_target("ids", "Case Identifier Variables:", required=False)
        self.rules = validation.rules(ds)
        self.make_tabs([("Basic Checks", self._basic), ("Single-Variable Rules", self._single),
                        ("Cross-Variable Rules", self._cross), ("Output", self._output),
                        ("Save", self._save)])
        self.finish()

    def _basic(self, f):
        g = f.group("Analysis Variables")
        g.check("basic", "Flag variables that fail any of the following checks")
        g.number("max_missing", "Maximum percentage of missing values:", 0, 100, 1)
        g.number("max_single", "Maximum percentage of cases in a single category:", 0, 100, 1)
        g.number("max_unique", "Maximum percentage of categories with count of 1:", 0, 100, 1)
        g.number("min_cv", "Minimum coefficient of variation:", 0, 1e6, 4)
        g.number("min_sd", "Minimum standard deviation:", 0, 1e9, 4)
        g = f.group("Case Identifiers")
        g.check("id_checks", "Flag incomplete IDs and duplicate IDs")
        g.check("empty", "Flag empty cases")

    def _single(self, f):
        row = QHBoxLayout()
        self.var_list = QListWidget()
        self.var_list.currentItemChanged.connect(self._show_rules)
        self.rule_list = QListWidget()
        for r in self.rules["single"]:
            it = QListWidgetItem(r["name"])
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Unchecked)
            it.setToolTip(validation.describe_rule(r))
            self.rule_list.addItem(it)
        self.rule_list.itemChanged.connect(self._rule_toggled)
        col1, col2 = QVBoxLayout(), QVBoxLayout()
        col1.addWidget(QLabel("Analysis variables:"))
        col1.addWidget(self.var_list)
        col2.addWidget(QLabel("Rules applied to the selected variable:"))
        col2.addWidget(self.rule_list)
        row.addLayout(col1)
        row.addLayout(col2)
        f.layout.addLayout(row, 10)
        if not self.rules["single"]:
            f.label("No single-variable rules are defined. Use Data > Validation > Define "
                    "Rules or Load Predefined Rules.")

    def targets_changed(self):
        if not hasattr(self, "var_list"):
            return
        names = self.selected("vars")
        cur = self.var_list.currentItem().text() if self.var_list.currentItem() else None
        self.var_list.blockSignals(True)
        self.var_list.clear()
        for n in names:
            self.var_list.addItem(n)
        self.var_list.blockSignals(False)
        if names:
            self.var_list.setCurrentRow(names.index(cur) if cur in names else 0)
        self.opts["assign"] = {k: v for k, v in self.opts["assign"].items() if k in names}

    def _show_rules(self, item, _prev=None):
        var = item.text() if item else None
        chosen = set(self.opts["assign"].get(var, [])) if var else set()
        self.rule_list.blockSignals(True)
        for i in range(self.rule_list.count()):
            it = self.rule_list.item(i)
            it.setCheckState(Qt.CheckState.Checked if it.text() in chosen
                             else Qt.CheckState.Unchecked)
        self.rule_list.blockSignals(False)

    def _rule_toggled(self, _item):
        cur = self.var_list.currentItem()
        if cur is None:
            return
        self.opts["assign"][cur.text()] = [
            self.rule_list.item(i).text() for i in range(self.rule_list.count())
            if self.rule_list.item(i).checkState() == Qt.CheckState.Checked]

    def _cross(self, f):
        self.cross_list = QListWidget()
        for r in self.rules["cross"]:
            it = QListWidgetItem(f"{r['name']}:  {r['expression']}")
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if r["name"] in self.opts["cross"]
                             else Qt.CheckState.Unchecked)
            it.setData(Qt.ItemDataRole.UserRole, r["name"])
            self.cross_list.addItem(it)
        self.cross_list.itemChanged.connect(lambda _i: self.opts.__setitem__("cross", [
            self.cross_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.cross_list.count())
            if self.cross_list.item(i).checkState() == Qt.CheckState.Checked]))
        f.layout.addWidget(self.cross_list, 10)
        if not self.rules["cross"]:
            f.label("No cross-variable rules are defined.")

    def _output(self, f):
        g = f.group("Casewise Report")
        g.number("min_violations", "Minimum number of violations for a case to be included:",
                 1, 1000)
        g.number("limit", "Maximum number of cases in report:", 1, 100000)

    def _save(self, f):
        g = f.group("Summary Variables")
        g.check("save_empty", "Empty case indicator (EmptyCase)")
        g.check("save_duplicate", "Duplicate ID group (DuplicateIDGroup)")
        g.check("save_incomplete", "Incomplete ID indicator (IncompleteID)")
        g.check("save_total", "Validation rule violations, total count "
                              "(ValidationRuleViolations)")
        g.check("save_indicators", "Indicator variables that record all validation rule "
                                   "violations")
        g.check("replace", "Replace existing summary variables")

    def validate(self):
        if not (self.selected("vars") or self.selected("ids") or self.opts["cross"]):
            return "Select analysis variables, case identifiers or cross-variable rules."
        return None

    def compute(self):
        o = self.opts
        basic = {k: o[k] for k in validation.BASIC_DEFAULTS} if o["basic"] else None
        save = {"empty": o["save_empty"], "incomplete": o["save_incomplete"],
                "duplicate": o["save_duplicate"], "total": o["save_total"],
                "indicators": o["save_indicators"], "replace": o["replace"]}
        return validation.validate_data(
            self.ds, self.selected("vars"), self.selected("ids"), basic, o["id_checks"],
            o["empty"], {k: v for k, v in o["assign"].items() if k in self.selected("vars")},
            list(o["cross"]), int(o["limit"]), save if any(save.values()) else None,
            int(o["min_violations"]))


# ============================================================================ Unusual Cases
class UnusualCasesDialog(TabbedPickerDialog):
    modifies_data = True
    DEFAULTS = {"norms": True, "summaries": True, "save_index": False, "save_peers": False,
                "save_reasons": False, "replace": False, "include_missing": False,
                "missing_pct": True, "criterion": "pct", "pct": 5.0, "number": 10,
                "use_cutoff": True, "cutoff": 2.0, "peer_min": 1, "peer_max": 15,
                "reasons": 1, "adjustment": 6.0}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Identify Unusual Cases", parent)
        self.add_target("vars", "Analysis Variables:")
        self.add_target("id", "Case Identifier Variable:", single=True, required=False)
        self.make_tabs([("Output", self._output), ("Save", self._save),
                        ("Missing Values", self._missing), ("Options", self._options)])
        self.finish()

    def _output(self, f):
        g = f.group("Output")
        f.label("The list of unusual cases and the reasons why they are unusual is always "
                "displayed.")
        g.check("norms", "Peer group norms")
        g.check("summaries", "Anomaly indices and reason occurrence summaries")

    def _save(self, f):
        g = f.group("Save Variables")
        g.check("save_index", "Anomaly index (AnomalyIndex)")
        g.check("save_peers", "Peer groups (PeerId, PeerSize, PeerPctSize)")
        g.check("save_reasons", "Reasons (ReasonVar, ReasonMeasure, ReasonValue, ReasonNorm)")
        g.check("replace", "Replace existing variables")

    def _missing(self, f):
        g = f.group("Missing Values")
        g.radio("include_missing", [(False, "Exclude missing values from analysis"),
                                    (True, "Include missing values in analysis")])
        g.check("missing_pct", "Use the proportion of missing values per case as an "
                               "analysis variable")

    def _options(self, f):
        g = f.group("Criteria for Identifying Unusual Cases")
        g.radio("criterion", [("pct", "Percentage of cases with highest anomaly index values"),
                              ("number", "Fixed number of cases with highest anomaly index "
                                         "values")])
        g.number("pct", "Percentage:", 0.01, 100, 2)
        g.number("number", "Number:", 1, 1e6)
        g.check("use_cutoff", "Identify only cases whose anomaly index meets or exceeds a "
                              "minimum value")
        g.number("cutoff", "Cutoff:", 0, 1e6, 2)
        g = f.group("Number of Peer Groups")
        g.number("peer_min", "Minimum:", 1, 100)
        g.number("peer_max", "Maximum:", 1, 100)
        g.number("reasons", "Maximum number of reasons:", 1, 50)
        g.number("adjustment", "Adjustment weight m (Δ = σ²/m):", 0.01, 1000, 2)

    def validate(self):
        if self.opts["peer_min"] > self.opts["peer_max"]:
            return "The minimum number of peer groups exceeds the maximum."
        if self.selected_one("id") in self.selected("vars"):
            return "The case identifier cannot be an analysis variable."
        return None

    def compute(self):
        o = self.opts
        names = self.selected("vars")
        cats = [n for n in names if measure_of(self.ds, n) != "scale"]
        conts = [n for n in names if n not in cats]
        save = {"index": o["save_index"], "peers": o["save_peers"],
                "reasons": o["save_reasons"], "replace": o["replace"]}
        return anomaly.detect_anomaly(
            self.ds, cats, conts, self.selected_one("id"), bool(o["include_missing"]),
            o["missing_pct"], float(o["adjustment"]), int(o["peer_min"]), int(o["peer_max"]),
            float(o["pct"]) if o["criterion"] == "pct" else None,
            int(o["number"]) if o["criterion"] == "number" else None,
            float(o["cutoff"]) if o["use_cutoff"] else None, int(o["reasons"]), o["norms"],
            o["summaries"], save if any(v for k, v in save.items() if k != "replace") else None)


# ============================================================================ Optimal Binning
class OptimalBinningDialog(TabbedPickerDialog):
    modifies_data = True
    DEFAULTS = {"descriptives": True, "entropy": True, "summaries": True, "save": True,
                "suffix": "_bin", "syntax": False, "missing": "pairwise", "prebin": True,
                "prebin_n": 1000, "merge": True, "threshold": 5.0, "lowerlimit": "inclusive"}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Optimal Binning", parent)
        self.add_target("vars", "Variables to Bin:", accepts=is_numeric)
        self.add_target("guide", "Optimize Bins with Respect to:", single=True)
        self.make_tabs([("Output", self._output), ("Save", self._save),
                        ("Missing Values", self._missing), ("Options", self._options)])
        self.finish()

    def _output(self, f):
        g = f.group("Display")
        g.check("summaries", "Endpoints for bins (binning summaries)")
        g.check("descriptives", "Descriptive statistics for variables that are binned")
        g.check("entropy", "Model entropy for variables that are binned")

    def _save(self, f):
        g = f.group("Save")
        g.check("save", "Create variables that contain binned data values")
        g.text("suffix", "Suffix for the new variables:")
        g.check("syntax", "Display the binning rules as syntax")

    def _missing(self, f):
        g = f.group("Missing Values")
        g.radio("missing", [("pairwise", "Pairwise (each variable with the guide)"),
                            ("listwise", "Listwise (all binned variables and the guide)")])

    def _options(self, f):
        g = f.group("Preprocessing")
        g.check("prebin", "Pre-binning (equal frequency) when there are many distinct values")
        g.number("prebin_n", "Maximum number of bins:", 2, 100000)
        g = f.group("Sparsely Populated Bins")
        g.check("merge", "Merge bins with relative case count less than a specified % of "
                         "the neighboring bins")
        g.number("threshold", "Percent threshold:", 0, 100, 2)
        g = f.group("Bin Endpoints")
        g.radio("lowerlimit", [("inclusive", "Lower limit inclusive (<=)"),
                               ("exclusive", "Lower limit exclusive (<)")])

    def validate(self):
        g = self.selected_one("guide")
        if g in self.selected("vars"):
            return "The guide variable cannot also be binned."
        if self.opts["save"]:
            for v in self.selected("vars"):
                if self.ds.has_name(v + self.opts["suffix"]):
                    return f"A variable named {v + self.opts['suffix']} already exists."
        return None

    def compute(self):
        o = self.opts
        save = {v: v + o["suffix"] for v in self.selected("vars")} if o["save"] else None
        return optimal_binning.optimal_binning(
            self.ds, self.selected("vars"), self.selected_one("guide"),
            int(o["prebin_n"]) if o["prebin"] else None, o["merge"], float(o["threshold"]),
            o["lowerlimit"], o["missing"], save, o["syntax"], o["descriptives"],
            o["entropy"], o["summaries"])


# ============================================================================ ADP
class PrepareDataDialog(TabbedPickerDialog):
    modifies_data = True
    DEFAULTS = {**{k: v for k, v in adp.DEFAULTS.items() if k != "reference"},
                "reference": ""}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Automatic Data Preparation", parent)
        self.add_target("target", "Target (optional):", single=True, required=False)
        self.add_target("inputs", "Inputs:")
        self.make_tabs([("Settings", self._settings), ("Transformations", self._transform)],
                       first="Fields")
        self.finish()

    def _settings(self, f):
        g = f.group("Exclude Fields")
        g.number("max_missing", "Maximum percentage of missing values:", 0, 100, 1)
        g.number("max_categories", "Maximum number of categories (nominal):", 2, 100000)
        g.number("max_mode", "Maximum percentage of cases in a single category:", 0, 100, 1)
        g = f.group("Adjust Measurement")
        g.number("cont_to_ord", "Continuous fields with fewer than N values become ordinal:",
                 2, 1000)
        g.number("ord_to_cont", "Ordinal fields with more than N values become continuous:",
                 2, 1000)
        g = f.group("Dates")
        g.check("dates", "Extract date elements and the elapsed time")
        g.text("reference", "Reference date (YYYY-MM-DD, empty = today):")

    def _transform(self, f):
        g = f.group("Improve Data Quality")
        g.radio("outliers", [("trim", "Replace outliers with cutoff values"),
                             ("missing", "Set outliers to missing"),
                             ("none", "Do not handle outliers")])
        g.number("cutoff", "Outlier cutoff (standard deviations):", 0.5, 100, 2)
        g.check("impute", "Replace missing values (mean / median / mode)")
        g.check("reorder_nominal", "Reorder nominal fields (least frequent category first)")
        g = f.group("Rescale and Transform")
        g.radio("rescale", [("zscore", "Continuous inputs: z-score transformation"),
                            ("minmax", "Continuous inputs: min-max (0-100)"),
                            ("none", "Do not rescale inputs")])
        g.check("boxcox", "Continuous target: Box-Cox transformation")
        g.check("merge", "Merge sparse categories (supervised with a target)")
        g.check("select", "Exclude continuous inputs unrelated to a continuous target")
        g.text("suffix", "Suffix for prepared fields:")

    def validate(self):
        if self.selected_one("target") in self.selected("inputs"):
            return "The target cannot also be an input."
        if self.opts["reference"]:
            import datetime as dt

            try:
                dt.date.fromisoformat(self.opts["reference"].strip())
            except ValueError:
                return "The reference date must be written as YYYY-MM-DD."
        return None

    def compute(self):
        import datetime as dt

        o = dict(self.opts)
        o["reference"] = dt.date.fromisoformat(o["reference"].strip()) if o["reference"] \
            else None
        return adp.prepare_data(self.ds, self.selected("inputs"), self.selected_one("target"),
                                o)


MENUS = {
    "Data": [("Validation", [("Load Predefined Rules...", LoadPredefinedRulesDialog),
                             ("Define Rules...", DefineRulesDialog),
                             ("Validate Data...", ValidateDataDialog)]),
             ("Identify Unusual Cases...", UnusualCasesDialog)],
    "Transform": [("Optimal Binning...", OptimalBinningDialog),
                  ("Prepare Data for Modeling", [("Automatic...", PrepareDataDialog)])],
}


def selftest() -> None:
    validation.selftest()
    anomaly.selftest()
    optimal_binning.selftest()
    adp.selftest()


"""Data > Orthogonal Design > Generate / Display and Analyze > Conjoint Analysis (SPSS
ORTHOPLAN, PLANCARDS and CONJOINT). Core code: :mod:`rsstatchem.core.stats.conjoint`."""
from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
                               QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QPushButton, QSpinBox, QCheckBox, QVBoxLayout)

from rsstatchem.core.dataset import NUMERIC, Dataset, is_valid_name
from rsstatchem.core.stats import conjoint
from rsstatchem.ui.dialogs.base import VariablePickerDialog, is_numeric
from rsstatchem.ui.dialogs.ml_tabs import OptionsDialog
from rsstatchem.ui.dialogs.wp11_util import editor_datasets, open_dataset


# ============================================================================ Generate
class DefineValuesDialog(QDialog):
    def __init__(self, name: str, values: list, labels: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Generate Design: Define Values for {name}")
        grid = QGridLayout()
        grid.addWidget(QLabel("Value"), 0, 1)
        grid.addWidget(QLabel("Label"), 0, 2)
        self.vals, self.labs = [], []
        for i in range(9):
            v = QLineEdit(_fmt(values[i]) if i < len(values) else "")
            lab = QLineEdit(labels[i] if i < len(labels) else "")
            grid.addWidget(QLabel(f"{i + 1}"), i + 1, 0)
            grid.addWidget(v, i + 1, 1)
            grid.addWidget(lab, i + 1, 2)
            self.vals.append(v)
            self.labs.append(lab)
        fill = QPushButton("Auto-Fill 1 to")
        self.n = QSpinBox()
        self.n.setRange(2, 9)
        self.n.setValue(max(2, len(values) or 3))
        fill.clicked.connect(self._fill)
        row = QHBoxLayout()
        row.addWidget(fill)
        row.addWidget(self.n)
        row.addStretch()
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Continue")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(grid)
        lay.addLayout(row)
        lay.addWidget(buttons)

    def _fill(self):
        for i, e in enumerate(self.vals):
            e.setText(str(i + 1) if i < self.n.value() else "")

    def result(self) -> tuple[list, list]:
        vals, labs = [], []
        for v, lab in zip(self.vals, self.labs):
            t = v.text().strip()
            if not t:
                continue
            try:
                vals.append(float(t.replace(",", ".")))
            except ValueError:
                vals.append(t)
            labs.append(lab.text().strip())
        return vals, labs


def _fmt(v) -> str:
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


class GenerateDesignDialog(QDialog):
    needs_data = False
    DEFAULTS = {"factors": [], "minimum": 0, "holdout": 0, "mix": False, "seed": 2000000,
                "dataset": "ORTHO"}
    _state: dict = {}

    def __init__(self, ds=None, parent=None):
        super().__init__(parent)
        self.ds = ds
        self.setWindowTitle("Generate Orthogonal Design")
        self.setMinimumSize(620, 460)
        import copy

        self.opts = copy.deepcopy({**self.DEFAULTS, **self._state})
        form = QFormLayout()
        self.name = QLineEdit()
        self.label = QLineEdit()
        form.addRow("Factor Name:", self.name)
        form.addRow("Factor Label:", self.label)
        add, change, remove = QPushButton("Add"), QPushButton("Change"), QPushButton("Remove")
        add.clicked.connect(self._add)
        change.clicked.connect(self._change)
        remove.clicked.connect(self._remove)
        self.lst = QListWidget()
        self.lst.currentRowChanged.connect(self._select)
        values = QPushButton("Define Values...")
        values.clicked.connect(self._values)
        btns = QVBoxLayout()
        for b in (add, change, remove):
            btns.addWidget(b)
        btns.addStretch()
        mid = QHBoxLayout()
        mid.addLayout(btns)
        mid.addWidget(self.lst, 1)
        opt = QFormLayout()
        self.dsname = QLineEdit(self.opts["dataset"])
        self.minimum = QSpinBox()
        self.minimum.setRange(0, 1000)
        self.minimum.setValue(int(self.opts["minimum"]))
        self.holdout = QSpinBox()
        self.holdout.setRange(0, 1000)
        self.holdout.setValue(int(self.opts["holdout"]))
        self.mix = QCheckBox("Randomly mix holdout cases with other cases")
        self.mix.setChecked(bool(self.opts["mix"]))
        self.seed = QSpinBox()
        self.seed.setRange(1, 2 ** 31 - 1)
        self.seed.setValue(int(self.opts["seed"]))
        opt.addRow("New dataset name:", self.dsname)
        opt.addRow("Minimum number of cases to generate:", self.minimum)
        opt.addRow("Number of holdout cases:", self.holdout)
        opt.addRow(self.mix)
        opt.addRow("Random number seed:", self.seed)
        box = QGroupBox("Options")
        box.setLayout(opt)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Reset
                                   | QDialogButtonBox.StandardButton.Cancel)
        self.ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Reset).clicked.connect(self._reset)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addLayout(mid, 1)
        lay.addWidget(values)
        lay.addWidget(box)
        lay.addWidget(buttons)
        self._refresh()

    def _text(self, f: dict) -> str:
        vals = " ".join(f"{_fmt(v)}" + (f" '{lab}'" if lab else "")
                        for v, lab in zip(f["values"], f["labels"] or [""] * len(f["values"])))
        return f"{f['name']}" + (f" '{f['label']}'" if f["label"] else "") + f" ({vals or '?'})"

    def _refresh(self, row: int | None = None):
        self.lst.blockSignals(True)
        self.lst.clear()
        for f in self.opts["factors"]:
            self.lst.addItem(self._text(f))
        self.lst.blockSignals(False)
        if row is not None and 0 <= row < self.lst.count():
            self.lst.setCurrentRow(row)

    def _select(self, row):
        if 0 <= row < len(self.opts["factors"]):
            f = self.opts["factors"][row]
            self.name.setText(f["name"])
            self.label.setText(f["label"])

    def _add(self):
        n = self.name.text().strip()
        if not is_valid_name(n) or any(f["name"].lower() == n.lower()
                                       for f in self.opts["factors"]):
            return
        self.opts["factors"].append({"name": n, "label": self.label.text().strip(),
                                     "values": [], "labels": []})
        self._refresh(len(self.opts["factors"]) - 1)

    def _change(self):
        row = self.lst.currentRow()
        n = self.name.text().strip()
        if 0 <= row < len(self.opts["factors"]) and is_valid_name(n):
            self.opts["factors"][row].update(name=n, label=self.label.text().strip())
            self._refresh(row)

    def _remove(self):
        row = self.lst.currentRow()
        if 0 <= row < len(self.opts["factors"]):
            del self.opts["factors"][row]
            self._refresh(min(row, len(self.opts["factors"]) - 1))

    def _values(self):
        row = self.lst.currentRow()
        if not 0 <= row < len(self.opts["factors"]):
            return
        f = self.opts["factors"][row]
        dlg = DefineValuesDialog(f["name"], f["values"], f["labels"], self)
        if dlg.exec():
            f["values"], f["labels"] = dlg.result()
            self._refresh(row)

    def _reset(self):
        self.opts = dict(self.DEFAULTS, factors=[])
        self._refresh()

    def validate(self) -> str | None:
        if not self.opts["factors"]:
            return "Add at least one factor."
        for f in self.opts["factors"]:
            if len(f["values"]) < 2:
                return f"Define at least two values for factor {f['name']}."
            if len(set(map(str, f["values"]))) != len(f["values"]):
                return f"Factor {f['name']} has duplicate values."
        return None

    def accept(self):
        err = self.validate()
        if err:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(self, self.windowTitle(), err)
            return
        self._collect()
        type(self)._state = dict(self.opts)
        super().accept()

    def _collect(self):
        self.opts.update(minimum=self.minimum.value(), holdout=self.holdout.value(),
                         mix=self.mix.isChecked(), seed=self.seed.value(),
                         dataset=self.dsname.text().strip() or "ORTHO")

    def compute(self):
        self._collect()
        factors = [conjoint.Factor(f["name"], list(f["values"]), list(f["labels"]),
                                   f["label"]) for f in self.opts["factors"]]
        ds, section = conjoint.generate_design(factors, int(self.opts["minimum"]),
                                               int(self.opts["holdout"]),
                                               int(self.opts["seed"]), bool(self.opts["mix"]))
        ds.name = self.opts["dataset"]
        open_dataset(self, ds)
        section.note("", f"The plan was written to the new dataset {ds.name}.")
        return section


class DisplayDesignDialog(VariablePickerDialog):
    DEFAULTS = {"listing": True, "profiles": False, "title": "", "footer": ""}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Display Design", parent)
        self.add_target("factors", "Factors:")
        self.add_checkbox("listing", "Listing for experimenter")
        self.add_checkbox("profiles", "Profiles for subjects")
        for key, label in (("title", "Profile title:"), ("footer", "Profile footer:")):
            edit = QLineEdit(self.opts[key])
            edit.textChanged.connect(lambda t, k=key: self.opts.__setitem__(k, t))
            row = QFormLayout()
            row.addRow(label, edit)
            self.options_layout.addLayout(row)
        self.finish()

    def validate(self):
        if not (self.opts["listing"] or self.opts["profiles"]):
            return "Select at least one format."
        return None

    def compute(self):
        o = self.opts
        return conjoint.display_design(self.ds, self.selected("factors"), o["listing"],
                                       o["profiles"], o["title"], o["footer"])


# ============================================================================ CONJOINT
class ConjointDialog(VariablePickerDialog):
    DEFAULTS = {"mode": "cards", "method": "score", "models": {}, "plan_path": "",
                "plan_name": "", "individual": False, "plots": True, "factors_plan": ""}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Conjoint Analysis", parent)
        top = QFormLayout()
        self.mode = QComboBox()
        self.mode.addItem("Plan and responses in this dataset (cards in rows, one response "
                          "variable per subject)", "cards")
        self.mode.addItem("Subjects in rows; plan in another dataset or file", "subjects")
        self.mode.setCurrentIndex(0 if self.opts["mode"] == "cards" else 1)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        top.addRow("Data layout:", self.mode)
        self.plan = QComboBox()
        self.plan_btn = QPushButton("File...")
        self.plan_btn.clicked.connect(self._browse)
        prow = QHBoxLayout()
        prow.addWidget(self.plan, 1)
        prow.addWidget(self.plan_btn)
        top.addRow("Plan:", prow)
        self.layout().insertLayout(0, top)
        self._fill_plans()
        self.add_target("factors", "Factors (plan variables):", required=False)
        self.add_target("resp", "Responses (in card order):", accepts=is_numeric)
        self.add_target("subject", "Subject ID (optional):", single=True, required=False)
        self.factor_edit = QLineEdit(self.opts["factors_plan"])
        self.factor_edit.setPlaceholderText("factor names in the plan (subjects layout)")
        self.factor_edit.textChanged.connect(lambda t: self.opts.__setitem__("factors_plan", t))
        row = QFormLayout()
        row.addRow("Plan factors:", self.factor_edit)
        self.options_layout.addLayout(row)
        self.add_option_button("Model...", self._model)
        self.add_option_button("Options...", self._options)
        self._mode_changed()
        self.finish()

    # ---------------------------------------------------------------- plan source
    def _fill_plans(self):
        self.plan.clear()
        self._plans = editor_datasets(self)
        for d in self._plans:
            self.plan.addItem(f"Open dataset: {d.name}", ("ds", d))
        if self.opts["plan_path"]:
            self.plan.addItem(f"File: {self.opts['plan_path']}", ("file", self.opts["plan_path"]))
            self.plan.setCurrentIndex(self.plan.count() - 1)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Plan File", "",
                                              "Data files (*.rsd *.sav *.csv *.xlsx)")
        if path:
            self.opts["plan_path"] = path
            self._fill_plans()

    def _mode_changed(self):
        self.opts["mode"] = self.mode.currentData()
        subj = self.opts["mode"] == "subjects"
        self.plan.setEnabled(subj)
        self.plan_btn.setEnabled(subj)
        self.factor_edit.setEnabled(subj)

    def plan_dataset(self) -> Dataset | None:
        data = self.plan.currentData()
        if data is None:
            return None
        kind, obj = data
        if kind == "ds":
            return obj
        from rsstatchem.io import files

        p = str(obj)
        if p.lower().endswith(".rsd"):
            return files.load_project(p)
        if p.lower().endswith(".sav"):
            return files.read_sav(p)
        if p.lower().endswith(".csv"):
            return Dataset(files.read_text(p))
        return Dataset(files.read_excel(p))

    def factor_names(self) -> list[str]:
        if self.opts["mode"] == "cards":
            return self.selected("factors")
        return [t for t in self.opts["factors_plan"].replace(",", " ").split() if t]

    # ---------------------------------------------------------------- sub-dialogs
    def _model(self):
        names = self.factor_names()
        if not names:
            return
        dlg = OptionsDialog("Conjoint Analysis: Model", {"models": dict(self.opts["models"])},
                            self, columns=1)
        g = dlg.group("Relation of each factor to the scores")
        combos = {}
        for nm in names:
            cb = QComboBox()
            for key, text in (("discrete", "Discrete"), ("linear", "Linear"),
                              ("linear_more", "Linear (more is preferred)"),
                              ("linear_less", "Linear (less is preferred)"),
                              ("ideal", "Ideal"), ("antiideal", "Anti-ideal")):
                cb.addItem(text, key)
            cur = self.opts["models"].get(nm, "discrete")
            cb.setCurrentIndex(max(0, [cb.itemData(i) for i in range(cb.count())].index(cur)))
            combos[nm] = cb
            r = QFormLayout()
            r.addRow(nm, cb)
            g.layout.addLayout(r)
        if dlg.exec():
            self.opts["models"] = {nm: cb.currentData() for nm, cb in combos.items()}

    def _options(self):
        dlg = OptionsDialog("Conjoint Analysis: Options", self.opts, self, columns=1)
        g = dlg.group("Responses")
        g.radio("method", [("score", "Scores (higher = preferred)"),
                           ("rank", "Ranks of the cards (1 = most preferred)"),
                           ("sequence", "Sequence (card numbers from most to least preferred)")])
        g = dlg.group("Output")
        g.check("individual", "Results for individual subjects")
        g.check("plots", "Importance and utility charts")
        dlg.run(self.opts)

    def _models(self) -> list:
        out = []
        for nm in self.factor_names():
            m = self.opts["models"].get(nm, "discrete")
            if m.startswith("linear_"):
                out.append(conjoint.FactorModel(nm, "linear", m.split("_")[1]))
            else:
                out.append(conjoint.FactorModel(nm, m))
        return out

    def validate(self):
        if not self.factor_names():
            return "Select the factors (cards layout) or type the plan's factor names."
        if self.opts["mode"] == "subjects":
            try:
                plan = self.plan_dataset()
            except Exception as exc:  # noqa: BLE001
                return f"The plan cannot be read: {exc}"
            if plan is None:
                return "Open the plan dataset or choose a plan file."
            for nm in self.factor_names():
                if not plan.has_name(nm):
                    return f"The plan has no variable {nm}."
        else:
            if set(self.factor_names()) & set(self.selected("resp")):
                return "A factor cannot also be a response variable."
        return None

    def compute(self):
        o = self.opts
        names = self.factor_names()
        models = self._models()
        if o["mode"] == "cards":
            plan = conjoint.plan_frame(self.ds, names)
            status = plan["STATUS_"].to_numpy(float)
            R = np.column_stack([self.ds.df[v].to_numpy(float) for v in self.selected("resp")])
            responses = R[status != 2].T
            ids = list(self.selected("resp"))
        else:
            pds = self.plan_dataset()
            plan = conjoint.plan_frame(pds, names)
            responses = np.column_stack([self.ds.df[v].to_numpy(float)
                                         for v in self.selected("resp")])
            sid = self.selected_one("subject")
            ids = [str(x) for x in self.ds.df[sid]] if sid else None
        return conjoint.conjoint(plan, models, responses, o["method"], ids,
                                 bool(o["individual"]), bool(o["plots"]))


MENUS = {"Data": [("Orthogonal Design", [("Generate...", GenerateDesignDialog),
                                          ("Display...", DisplayDesignDialog)])],
         "Analyze": [("Conjoint Analysis...", ConjointDialog)]}


def selftest() -> None:
    conjoint.selftest()
    assert NUMERIC == "numeric"

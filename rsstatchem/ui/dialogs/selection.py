"""Data > Select Cases, Split File, Weight Cases dialogs."""
from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QButtonGroup, QGroupBox, QHBoxLayout, QLabel, QPushButton,
                               QRadioButton, QVBoxLayout)

from rsstatchem.core import selection
from rsstatchem.ui.dialogs.base import SubDialog, VariablePickerDialog, is_numeric
from rsstatchem.ui.dialogs.transform import IfDialog


def _radios(parent, layout, options: list[tuple[str, str]], current: str) -> dict:
    group = QButtonGroup(parent)
    buttons = {}
    for key, text in options:
        rb = QRadioButton(text)
        rb.setChecked(key == current)
        group.addButton(rb)
        layout.addWidget(rb)
        buttons[key] = rb
    return buttons


def _status(ds) -> str:
    notes = selection.describe(ds)
    return "Current Status: " + (" · ".join(notes) if notes else "Do not filter cases")


class SelectCasesDialog(VariablePickerDialog):
    modifies_data = True
    DEFAULTS = {"mode": "all", "expression": "", "sample": {"percent": 10.0, "exact": False,
                                                            "n": 10, "first": 100},
                "range": {"first": 1, "last": 1}, "output": "filter", "seed": None}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Select Cases", parent)
        self.setMinimumSize(660, 600)
        self.add_target("filter", "Use filter variable:", single=True, required=False)
        box = QGroupBox("Select")
        inner = QVBoxLayout(box)
        rows = [("all", "All cases", None), ("if", "If condition is satisfied", self._if),
                ("sample", "Random sample of cases", self._sample),
                ("range", "Based on time or case range", self._range),
                ("variable", "Use filter variable (list above)", None)]
        group = QButtonGroup(self)
        self.mode_buttons = {}
        for key, text, callback in rows:
            line = QHBoxLayout()
            rb = QRadioButton(text)
            rb.setChecked(self.opts["mode"] == key)
            rb.toggled.connect(lambda on, k=key: on and self.opts.__setitem__("mode", k))
            group.addButton(rb)
            line.addWidget(rb)
            if callback:
                button = QPushButton({"if": "If...", "sample": "Sample...",
                                      "range": "Range..."}[key])
                button.clicked.connect(callback)
                line.addWidget(button)
            line.addStretch()
            inner.addLayout(line)
            self.mode_buttons[key] = rb
        self.detail = QLabel()
        self.detail.setStyleSheet("color:#555;")
        inner.addWidget(self.detail)
        self.options_layout.addWidget(box)
        out = QGroupBox("Output")
        out_layout = QVBoxLayout(out)
        self.output_buttons = _radios(self, out_layout, [
            ("filter", "Filter out unselected cases"),
            ("copy", "Copy selected cases to a new dataset"),
            ("delete", "Delete unselected cases")], self.opts["output"])
        self.options_layout.addWidget(out)
        status = QLabel(_status(ds))
        status.setStyleSheet("color:#8E1B1B;")
        self.options_layout.addWidget(status)
        self.finish()
        self.ok_button.setEnabled(True)
        self._show_detail()

    def _update_ok(self) -> None:
        if hasattr(self, "ok_button"):
            self.ok_button.setEnabled(True)

    def targets_changed(self) -> None:
        if hasattr(self, "mode_buttons") and self.selected_one("filter"):
            self.mode_buttons["variable"].setChecked(True)

    def _show_detail(self) -> None:
        s, r = self.opts["sample"], self.opts["range"]
        sample = (f"Exactly {s['n']} cases from the first {s['first']} cases" if s["exact"]
                  else f"Approximately {s['percent']:g}% of all cases")
        self.detail.setText(f"If: {self.opts['expression'] or '—'}   |   Sample: {sample}"
                            f"   |   Range: {r['first']}–{r['last']}")

    def _if(self) -> None:
        dlg = IfDialog(self.ds, self.opts["expression"], self, "Select Cases")
        dlg.all_cases.setVisible(False)
        dlg.some.setChecked(True)
        if dlg.exec():
            self.opts["expression"] = dlg.condition()
            self.mode_buttons["if"].setChecked(True)
            self._show_detail()

    def _sample(self) -> None:
        s = self.opts["sample"]
        dlg = SubDialog("Select Cases: Random Sample", self, columns=1)
        box = dlg.group("Sample Size")
        dlg.radio(box, "exact", [(False, "Approximately (percent below)"),
                                 (True, "Exactly (cases below)")], s["exact"])
        dlg.number(box, "percent", "% of all cases:", s["percent"], 0.01, 100, 2)
        dlg.number(box, "n", "Exactly … cases:", s["n"], 1, 1e9, 0)
        dlg.number(box, "first", "… from the first … cases:", s["first"] or self.ds.n_cases,
                   1, 1e9, 0)
        values = dlg.run()
        if values:
            self.opts["sample"] = {"percent": values["percent"], "exact": values["exact"],
                                   "n": int(values["n"]), "first": int(values["first"])}
            self.mode_buttons["sample"].setChecked(True)
            self._show_detail()

    def _range(self) -> None:
        r = self.opts["range"]
        dlg = SubDialog("Select Cases: Range", self, columns=1)
        box = dlg.group("Observation")
        dlg.number(box, "first", "First Case:", r["first"], 1, max(1, self.ds.n_cases), 0)
        dlg.number(box, "last", "Last Case:", r["last"] if r["last"] > 1 else self.ds.n_cases,
                   1, max(1, self.ds.n_cases), 0)
        values = dlg.run()
        if values:
            self.opts["range"] = {"first": int(values["first"]), "last": int(values["last"])}
            self.mode_buttons["range"].setChecked(True)
            self._show_detail()

    def validate(self):
        self.opts["output"] = next(k for k, b in self.output_buttons.items() if b.isChecked())
        mode = self.opts["mode"]
        if mode == "if" and not self.opts["expression"]:
            return "Click If… and enter a condition."
        if mode == "variable" and not self.selected_one("filter"):
            return "Move a filter variable into the list."
        if mode == "all" and self.opts["output"] != "filter":
            self.opts["output"] = "filter"
        return None

    def compute(self):
        o, s, r = self.opts, self.opts["sample"], self.opts["range"]
        section, new = selection.select_cases(
            self.ds, o["mode"], expression=o["expression"],
            percent=None if s["exact"] else s["percent"], n_exact=s["n"],
            from_first=s["first"], first=r["first"], last=r["last"],
            variable=self.selected_one("filter"), output=o["output"])
        if new is not None:
            editor = self.parent()
            QTimer.singleShot(0, lambda: editor.ctx.new_editor(new))
        return section


class SplitFileDialog(VariablePickerDialog):
    modifies_data = True
    DEFAULTS = {"mode": "off", "sort": True}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Split File", parent)
        current = ds.meta.get("split") or {}
        if current.get("vars"):
            self.opts["mode"] = current.get("mode", "separate")
        self.add_target("vars", "Groups Based on:", required=False)
        box = QGroupBox("")
        inner = QVBoxLayout(box)
        self.mode_buttons = _radios(self, inner, [
            ("off", "Analyze all cases, do not create groups"),
            ("compare", "Compare groups"), ("separate", "Organize output by groups")],
            self.opts["mode"])
        self.options_layout.addWidget(box)
        sort_box = QGroupBox("")
        sort_inner = QVBoxLayout(sort_box)
        self.sort_buttons = _radios(self, sort_inner, [
            ("sort", "Sort the file by grouping variables"),
            ("sorted", "File is already sorted")], "sort" if self.opts["sort"] else "sorted")
        self.options_layout.addWidget(sort_box)
        status = QLabel(_status(ds))
        status.setStyleSheet("color:#8E1B1B;")
        self.options_layout.addWidget(status)
        self.finish()
        for name in current.get("vars", []):
            if ds.has_name(name) and name not in self.selected("vars"):
                self._take(name, self._targets[0])
        self._update_arrows()

    def targets_changed(self) -> None:
        if hasattr(self, "mode_buttons") and self.selected("vars") and \
                self.mode_buttons["off"].isChecked():
            self.mode_buttons["separate"].setChecked(True)

    def validate(self):
        self.opts["mode"] = next(k for k, b in self.mode_buttons.items() if b.isChecked())
        self.opts["sort"] = self.sort_buttons["sort"].isChecked()
        if self.opts["mode"] != "off" and not self.selected("vars"):
            return "Move at least one grouping variable into 'Groups Based on'."
        return None

    def compute(self):
        return selection.split_file(self.ds, self.selected("vars"), self.opts["mode"],
                                    self.opts["sort"])


class WeightCasesDialog(VariablePickerDialog):
    modifies_data = True

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Weight Cases", parent)
        self.add_target("weight", "Frequency Variable:", accepts=is_numeric, single=True,
                        required=False)
        box = QGroupBox("")
        inner = QVBoxLayout(box)
        self.mode_buttons = _radios(self, inner, [("off", "Do not weight cases"),
                                                  ("on", "Weight cases by")],
                                    "on" if ds.meta.get("weight") else "off")
        self.options_layout.addWidget(box)
        status = QLabel("Current Status: " + (f"Weight cases by {ds.meta['weight']}"
                                              if ds.meta.get("weight")
                                              else "Do not weight cases"))
        status.setStyleSheet("color:#8E1B1B;")
        self.options_layout.addWidget(status)
        self.finish()
        if ds.meta.get("weight") and ds.has_name(ds.meta["weight"]) and \
                not self.selected("weight"):
            self._take(ds.meta["weight"], self._targets[0])
        self._update_arrows()

    def targets_changed(self) -> None:
        if hasattr(self, "mode_buttons") and self.selected_one("weight"):
            self.mode_buttons["on"].setChecked(True)

    def validate(self):
        if self.mode_buttons["on"].isChecked() and not self.selected_one("weight"):
            return "Move a frequency variable into the list."
        return None

    def compute(self):
        var = self.selected_one("weight") if self.mode_buttons["on"].isChecked() else None
        return selection.weight_cases(self.ds, var)

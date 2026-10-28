"""Analyze > Quality Control: Control Charts, Pareto Charts and Process Capability (SPSS
SPCHART / Pareto dialogs). Core code: :mod:`rsstatchem.core.stats.qc`."""
from __future__ import annotations

from PySide6.QtWidgets import QComboBox, QFormLayout, QLabel

from rsstatchem.core.stats import qc
from rsstatchem.ui.dialogs.base import VariablePickerDialog, is_numeric
from rsstatchem.ui.dialogs.ml_tabs import OptionsDialog
from rsstatchem.ui.dialogs.wp11_util import number_or_none, parse_values

CHART_ITEMS = [("xbar_r", "X-Bar, R"), ("xbar_s", "X-Bar, s"),
               ("ir", "Individuals, Moving Range"), ("p", "p, np: p chart"),
               ("np", "p, np: np chart"), ("c", "c, u: c chart"), ("u", "c, u: u chart")]


class ControlChartsDialog(VariablePickerDialog):
    DEFAULTS = {"chart": "xbar_r", "organization": "units", "second": True, "sigmas": 3.0,
                "min_size": 2, "span": 2, "rules": [], "nonconforming": "1",
                "sample_size": 5.0, "capability": False, "lsl": "", "usl": "", "target": "",
                "sigma_method": "auto", "capability_plot": True, "limits_table": True}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Control Charts", parent)
        top = QFormLayout()
        self.kind = QComboBox()
        for key, text in CHART_ITEMS:
            self.kind.addItem(text, key)
        self.kind.setCurrentIndex([k for k, _ in CHART_ITEMS].index(self.opts["chart"]))
        self.org = QComboBox()
        self.org.addItem("Cases are units", "units")
        self.org.addItem("Cases are subgroups", "subgroups")
        self.org.setCurrentIndex(0 if self.opts["organization"] == "units" else 1)
        top.addRow("Chart type:", self.kind)
        top.addRow("Data organization:", self.org)
        self.layout().insertLayout(0, top)
        self.kind.currentIndexChanged.connect(self._kind_changed)
        self.org.currentIndexChanged.connect(self._kind_changed)
        self.add_target("process", "Process Measurement / Characteristic / Samples:",
                        accepts=is_numeric, required=False)
        self.add_target("subgroup", "Subgroups Defined by:", single=True, required=False)
        self.add_target("label", "Label Subgroups by:", single=True, required=False)
        self.add_target("nsize", "Sample Size (variable):", accepts=is_numeric, single=True,
                        required=False)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color:#666")
        self.options_layout.addWidget(self.hint)
        self.add_checkbox("second", "Display second chart (R, s or moving range)")
        self.add_option_button("Statistics...", self._statistics)
        self.add_option_button("Rules...", self._rules)
        self.add_option_button("Options...", self._options)
        self._kind_changed()
        self.finish()

    def _kind_changed(self):
        self.opts["chart"] = self.kind.currentData()
        self.opts["organization"] = self.org.currentData()
        c, org = self.opts["chart"], self.opts["organization"]
        if c in ("xbar_r", "xbar_s"):
            t = ("Process Measurement + Subgroups Defined by." if org == "units" else
                 "Samples: one variable per observation within a subgroup (each case is a "
                 "subgroup).")
        elif c == "ir":
            t = "Process Measurement (each case is one unit); optional point labels."
        elif org == "units":
            t = ("Characteristic + Subgroups Defined by; " + (
                "the value counted as nonconforming is set in Options." if c in ("p", "np")
                else "the characteristic holds the number of defects of each unit."))
        else:
            t = ("Characteristic = number " + ("nonconforming" if c in ("p", "np") else
                                               "of defects") + " per subgroup; the sample "
                 "size is a variable or a constant (Options).")
        self.hint.setText(t)

    def _statistics(self):
        dlg = OptionsDialog("Control Charts: Statistics", self.opts, self)
        g = dlg.group("Specification Limits")
        g.check("capability", "Display process capability and performance indices")
        g.text("usl", "Upper:", "USL")
        g.text("lsl", "Lower:", "LSL")
        g.text("target", "Target:", "for CpM / PpM")
        g = dlg.group("Capability Sigma")
        g.radio("sigma_method", [("auto", "Default for the chart"),
                                 ("rbar", "Estimate using R-bar"),
                                 ("sbar", "Estimate using S-bar"),
                                 ("mrbar", "Estimate using mean moving range"),
                                 ("variance", "Estimate using within-subgroup variation")])
        g.check("capability_plot", "Process capability histogram")
        dlg.run(self.opts)

    def _rules(self):
        dlg = OptionsDialog("Control Charts: Rules", self.opts, self, columns=1)
        dlg.checklist("Control Rules", "rules", list(qc.RULES.items()))
        dlg.run(self.opts)

    def _options(self):
        dlg = OptionsDialog("Control Charts: Options", self.opts, self, columns=1)
        g = dlg.group("Options")
        g.number("sigmas", "Number of sigmas:", 1, 10, 2)
        g.number("min_size", "Minimum subgroup sample size:", 2, 100)
        g.number("span", "Span (moving range):", 2, 100)
        g.text("nonconforming", "Value of nonconforming units (p, np; cases are units):")
        g.number("sample_size", "Constant sample size (cases are subgroups):", 1, 1e9)
        g.check("limits_table", "Display control limits table")
        dlg.run(self.opts)

    def validate(self):
        o = self.opts
        proc = self.selected("process")
        if not proc:
            return "Select a process measurement (or the sample variables)."
        c, org = o["chart"], o["organization"]
        if c in ("xbar_r", "xbar_s"):
            if org == "units" and not self.selected_one("subgroup"):
                return "Select the variable that defines the subgroups."
            if org == "subgroups" and len(proc) < 2:
                return "Select at least two sample variables."
        elif c in ("p", "np", "c", "u") and org == "units" and not \
                self.selected_one("subgroup"):
            return "Select the variable that defines the subgroups."
        if o["capability"]:
            try:
                lsl, usl = number_or_none(o["lsl"]), number_or_none(o["usl"])
                number_or_none(o["target"])
            except ValueError:
                return "The specification limits must be numbers."
            if lsl is None and usl is None:
                return "Specify at least one specification limit."
            if c not in ("xbar_r", "xbar_s", "ir"):
                return "Process capability needs an X-bar or individuals chart."
        return None

    def compute(self):
        o = self.opts
        c, org = o["chart"], o["organization"]
        proc = self.selected("process")
        kw = dict(A=float(o["sigmas"]), min_size=int(o["min_size"]), span=int(o["span"]),
                  rules=list(o["rules"]), second_chart=bool(o["second"]),
                  show_limits_table=bool(o["limits_table"]),
                  capability_stats=bool(o["capability"]),
                  lsl=number_or_none(o["lsl"]), usl=number_or_none(o["usl"]),
                  target=number_or_none(o["target"]),
                  sigma_method=None if o["sigma_method"] == "auto" else o["sigma_method"],
                  capability_plot=bool(o["capability_plot"]),
                  label_var=self.selected_one("label"))
        if c in ("xbar_r", "xbar_s"):
            if org == "subgroups":
                return qc.control_chart(self.ds, c, samples=proc, **kw)
            return qc.control_chart(self.ds, c, [proc[0]], self.selected_one("subgroup"), **kw)
        if c == "ir":
            return qc.control_chart(self.ds, c, [proc[0]], **kw)
        if org == "units":
            vals = parse_values(o["nonconforming"], self.ds, proc[0])
            return qc.control_chart(self.ds, c, [proc[0]], self.selected_one("subgroup"),
                                    nonconforming_value=vals[0] if vals else None, **kw)
        size = self.selected_one("nsize") or float(o["sample_size"])
        return qc.control_chart(self.ds, c, [proc[0]], sample_size=size, **kw)


class ParetoDialog(VariablePickerDialog):
    DEFAULTS = {"cumulative": True, "table": True}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Pareto Charts", parent)
        self.add_target("category", "Category Axis:", single=True)
        self.add_target("value", "Bars Represent Sum of (optional):", accepts=is_numeric,
                        single=True, required=False)
        self.add_target("stack", "Define Stacks by (optional):", single=True, required=False)
        self.add_checkbox("cumulative", "Display cumulative line")
        self.add_checkbox("table", "Display Pareto table")
        self.finish()

    def compute(self):
        return qc.pareto(self.ds, self.selected_one("category"), self.selected_one("value"),
                         self.selected_one("stack"), self.opts["cumulative"],
                         self.opts["table"])


class ProcessCapabilityDialog(VariablePickerDialog):
    DEFAULTS = {"lsl": "", "usl": "", "target": "", "plot": True, "span": 2}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Process Capability", parent)
        self.add_target("var", "Process Measurement:", accepts=is_numeric, single=True)
        self.add_target("subgroup", "Subgroups Defined by (optional):", single=True,
                        required=False)
        self.add_option_button("Limits...", self._limits)
        self.add_checkbox("plot", "Capability histogram")
        self.finish()

    def _limits(self):
        dlg = OptionsDialog("Process Capability: Limits", self.opts, self, columns=1)
        g = dlg.group("Specification Limits")
        g.text("usl", "Upper (USL):")
        g.text("lsl", "Lower (LSL):")
        g.text("target", "Target:")
        g.number("span", "Span (moving range, no subgroups):", 2, 100)
        dlg.run(self.opts)

    def validate(self):
        try:
            lsl, usl = number_or_none(self.opts["lsl"]), number_or_none(self.opts["usl"])
            number_or_none(self.opts["target"])
        except ValueError:
            return "The limits must be numbers."
        if lsl is None and usl is None:
            return "Click Limits… and specify at least one specification limit."
        return None

    def compute(self):
        o = self.opts
        return qc.process_capability(self.ds, self.selected_one("var"),
                                     self.selected_one("subgroup"), number_or_none(o["lsl"]),
                                     number_or_none(o["usl"]), number_or_none(o["target"]),
                                     span=int(o["span"]), plot=bool(o["plot"]))


MENUS = {"Analyze/Quality Control": [("Control Charts...", ControlChartsDialog),
                                     ("Pareto Charts...", ParetoDialog),
                                     ("Process Capability...", ProcessCapabilityDialog)]}


def selftest() -> None:
    qc.selftest()

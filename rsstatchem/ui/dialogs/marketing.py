"""Analyze > Direct Marketing (SPSS Direct Marketing add-on): the Choose Technique window and
the RFM Analysis, Cluster Analysis, Prospect Profiles, Postal Code Response Rates, Propensity
to Purchase and Control Package Test dialogs. Core code:
:mod:`rsstatchem.core.stats.marketing`."""
from __future__ import annotations

import datetime as dt

from PySide6.QtWidgets import (QButtonGroup, QComboBox, QDialog, QDialogButtonBox,
                               QFormLayout, QLabel, QLineEdit, QRadioButton, QVBoxLayout)

from rsstatchem.core.stats import marketing
from rsstatchem.ui.dialogs.base import VariablePickerDialog, is_numeric
from rsstatchem.ui.dialogs.ml_tabs import OptionsDialog
from rsstatchem.ui.dialogs.wp11_util import number_or_none, open_dataset, parse_values


def _text_row(dialog, label: str, key: str, placeholder: str = "") -> QLineEdit:
    edit = QLineEdit(str(dialog.opts.get(key) or ""))
    edit.setPlaceholderText(placeholder)
    edit.textChanged.connect(lambda t: dialog.opts.__setitem__(key, t.strip()))
    row = QFormLayout()
    row.addRow(label, edit)
    dialog.options_layout.addLayout(row)
    return edit


def _reference(text: str):
    return dt.date.fromisoformat(text.strip()) if text and text.strip() else None


class RFMDialog(VariablePickerDialog):
    modifies_data = True
    DEFAULTS = {"format": "transactions", "summary": "total", "r_bins": 5, "f_bins": 5,
                "m_bins": 5, "nested": True, "ties": "same", "seed": 2000000, "reference": ""}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "RFM Analysis", parent)
        top = QFormLayout()
        self.fmt = QComboBox()
        self.fmt.addItem("Transaction data (one row per transaction)", "transactions")
        self.fmt.addItem("Customer data (one row per customer)", "customers")
        self.fmt.setCurrentIndex(0 if self.opts["format"] == "transactions" else 1)
        self.fmt.currentIndexChanged.connect(
            lambda _i: self.opts.__setitem__("format", self.fmt.currentData()))
        top.addRow("Data format:", self.fmt)
        self.layout().insertLayout(0, top)
        self.add_target("id", "Customer Identifier (transactions):", single=True,
                        required=False)
        self.add_target("date", "Transaction Date / Recency (date or days):", single=True,
                        required=False)
        self.add_target("freq", "Number of Transactions (customer data):", accepts=is_numeric,
                        single=True, required=False)
        self.add_target("amount", "Transaction Amount / Monetary:", accepts=is_numeric,
                        single=True, required=False)
        self.add_option_button("Binning...", self._binning)
        self.add_option_button("Options...", self._options)
        self.finish()

    def _binning(self):
        dlg = OptionsDialog("RFM Analysis: Binning", self.opts, self, columns=1)
        g = dlg.group("Binning Method")
        g.radio("nested", [(True, "Nested"), (False, "Independent")])
        g = dlg.group("Number of Bins")
        g.number("r_bins", "Recency:", 2, 9)
        g.number("f_bins", "Frequency:", 2, 9)
        g.number("m_bins", "Monetary:", 2, 9)
        g = dlg.group("Ties")
        g.radio("ties", [("same", "Assign ties to the same bin"),
                         ("random", "Randomly assign ties")])
        g.number("seed", "Random seed:", 1, 2 ** 31 - 1)
        dlg.run(self.opts)

    def _options(self):
        dlg = OptionsDialog("RFM Analysis: Options", self.opts, self, columns=1)
        g = dlg.group("Summary Method (transaction data)")
        g.radio("summary", [("total", "Total"), ("mean", "Mean"), ("median", "Median"),
                            ("maximum", "Maximum")])
        g = dlg.group("Recency")
        g.text("reference", "Reference date (YYYY-MM-DD, empty = today):")
        dlg.run(self.opts)

    def validate(self):
        o = self.opts
        try:
            _reference(o["reference"])
        except ValueError:
            return "The reference date must be written as YYYY-MM-DD."
        need = ["id", "date", "amount"] if o["format"] == "transactions" else \
            ["date", "freq", "amount"]
        for k in need:
            if not self.selected_one(k):
                return "Select " + {"id": "the customer identifier", "date": "the date/recency "
                                    "variable", "freq": "the number of transactions",
                                    "amount": "the amount variable"}[k] + "."
        return None

    def compute(self):
        o = self.opts
        kw = dict(bins=(int(o["r_bins"]), int(o["f_bins"]), int(o["m_bins"])),
                  nested=bool(o["nested"]), ties=o["ties"], seed=int(o["seed"]))
        if o["format"] == "transactions":
            sec, new = marketing.rfm_transactions(
                self.ds, self.selected_one("id"), self.selected_one("date"),
                self.selected_one("amount"), o["summary"], _reference(o["reference"]), **kw)
            open_dataset(self, new)
            sec.note("", "The customer-level RFM scores were written to a new dataset.")
            return sec
        return marketing.rfm_customers(self.ds, self.selected_one("date"),
                                       self.selected_one("freq"), self.selected_one("amount"),
                                       _reference(o["reference"]), True, **kw)


class DMClusterDialog(VariablePickerDialog):
    DEFAULTS = {"auto": True, "max_k": 15, "fixed_k": 3, "save": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Cluster Analysis (Direct Marketing)", parent)
        self.add_target("fields", "Fields to Create Segments:")
        self.add_checkbox("save", "Create a cluster membership field (ClusterGroup)")
        self.add_option_button("Settings...", self._settings)
        self.finish()

    def _settings(self):
        dlg = OptionsDialog("Cluster Analysis: Settings", self.opts, self, columns=1)
        g = dlg.group("Number of Clusters")
        g.radio("auto", [(True, "Determine automatically"), (False, "Specify fixed")])
        g.number("max_k", "Maximum:", 2, 50)
        g.number("fixed_k", "Fixed number:", 1, 50)
        dlg.run(self.opts)

    def compute(self):
        o = self.opts
        return marketing.dm_cluster(self.ds, self.selected("fields"), int(o["max_k"]),
                                    None if o["auto"] else int(o["fixed_k"]), bool(o["save"]))


class ProspectProfilesDialog(VariablePickerDialog):
    DEFAULTS = {"positive": "1", "min_group": 100, "use_rate": False, "min_rate": 7.0,
                "depth": 3}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Prospect Profiles", parent)
        self.add_target("resp", "Response Field:", single=True)
        self.add_target("fields", "Create Profiles with:")
        _text_row(self, "Positive response value(s):", "positive", "e.g. 1")
        self.add_option_button("Settings...", self._settings)
        self.finish()

    def _settings(self):
        dlg = OptionsDialog("Prospect Profiles: Settings", self.opts, self, columns=1)
        g = dlg.group("Profiles")
        g.number("min_group", "Minimum profile group size:", 2, 1e6)
        g.number("depth", "Maximum tree depth:", 1, 10)
        g.check("use_rate", "Include minimum response rate threshold information")
        g.number("min_rate", "Minimum response rate (%):", 0, 100, 2)
        dlg.run(self.opts)

    def validate(self):
        if not parse_values(self.opts["positive"]):
            return "Enter the positive response value(s)."
        return None

    def compute(self):
        o = self.opts
        r = self.selected_one("resp")
        return marketing.prospect_profiles(
            self.ds, r, parse_values(o["positive"], self.ds, r), self.selected("fields"),
            int(o["min_group"]), float(o["min_rate"]) if o["use_rate"] else None,
            int(o["depth"]))


class PostalCodeDialog(VariablePickerDialog):
    DEFAULTS = {"positive": "1", "digits": 0, "use_rate": False, "min_rate": 0.0,
                "profit": False, "cost": 1.0, "revenue": 10.0, "dataset": True}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Postal Code Response Rates", parent)
        self.add_target("resp", "Response Field:", single=True)
        self.add_target("postal", "Postal Code Field:", single=True)
        _text_row(self, "Positive response value(s):", "positive", "e.g. 1")
        self.add_option_button("Settings...", self._settings)
        self.finish()

    def _settings(self):
        dlg = OptionsDialog("Postal Code Response Rates: Settings", self.opts, self, columns=1)
        g = dlg.group("Group Postal Codes")
        g.number("digits", "Use first N digits / characters (0 = all):", 0, 20)
        g = dlg.group("Output")
        g.check("use_rate", "Only postal codes with a minimum response rate")
        g.number("min_rate", "Minimum response rate (%):", 0, 100, 2)
        g.check("profit", "Profit and ROI")
        g.number("cost", "Cost per mailing:", 0, 1e9, 2)
        g.number("revenue", "Revenue per response:", 0, 1e9, 2)
        g.check("dataset", "Write the response rates to a new dataset")
        dlg.run(self.opts)

    def compute(self):
        o = self.opts
        r = self.selected_one("resp")
        sec, new = marketing.postal_code_rates(
            self.ds, r, parse_values(o["positive"], self.ds, r), self.selected_one("postal"),
            int(o["digits"]) or None, float(o["min_rate"]) if o["use_rate"] else None,
            float(o["cost"]) if o["profit"] else None,
            float(o["revenue"]) if o["profit"] else None)
        if o["dataset"]:
            open_dataset(self, new)
        return sec


class PropensityPurchaseDialog(VariablePickerDialog):
    DEFAULTS = {"positive": "1", "validate": True, "train": 50.0, "seed": 2000000,
                "save": True, "cut": 0.5}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Propensity to Purchase", parent)
        self.add_target("resp", "Response Field:", single=True)
        self.add_target("pred", "Predict Propensity with:")
        _text_row(self, "Positive response value(s):", "positive", "e.g. 1")
        self.add_checkbox("save", "Save the propensity scores")
        self.add_option_button("Settings...", self._settings)
        self.finish()

    def _settings(self):
        dlg = OptionsDialog("Propensity to Purchase: Settings", self.opts, self, columns=1)
        g = dlg.group("Model Validation")
        g.check("validate", "Validate the model on a random test partition")
        g.number("train", "Training sample (%):", 1, 99, 1)
        g.number("seed", "Random seed:", 1, 2 ** 31 - 1)
        g.number("cut", "Classification cut value:", 0.01, 0.99, 2)
        dlg.run(self.opts)

    def compute(self):
        o = self.opts
        r = self.selected_one("resp")
        return marketing.propensity_to_purchase(
            self.ds, r, parse_values(o["positive"], self.ds, r), self.selected("pred"),
            bool(o["validate"]), float(o["train"]), int(o["seed"]), bool(o["save"]),
            float(o["cut"]))


class ControlPackageDialog(VariablePickerDialog):
    DEFAULTS = {"control": "1", "kind": "response", "positive": "1", "alpha": 0.05}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Control Package Test", parent)
        self.add_target("campaign", "Campaign Field:", single=True)
        self.add_target("eff", "Effectiveness Response Field:", single=True)
        _text_row(self, "Control package value:", "control", "value of the control group")
        _text_row(self, "Positive response value (response type):", "positive", "e.g. 1")
        self.add_option_button("Options...", self._options)
        self.finish()

    def _options(self):
        dlg = OptionsDialog("Control Package Test: Options", self.opts, self, columns=1)
        g = dlg.group("Effectiveness")
        g.radio("kind", [("response", "Reply (binary response)"),
                         ("spending", "Spending amount")])
        g.number("alpha", "Significance level:", 0.0001, 0.5, 4)
        dlg.run(self.opts)

    def validate(self):
        if not parse_values(self.opts["control"], self.ds, self.selected_one("campaign")):
            return "Enter the value of the control package."
        return None

    def compute(self):
        o = self.opts
        camp, eff = self.selected_one("campaign"), self.selected_one("eff")
        return marketing.control_package_test(
            self.ds, camp, parse_values(o["control"], self.ds, camp)[0], eff, o["kind"],
            parse_values(o["positive"], self.ds, eff), float(o["alpha"]))


TECHNIQUES = [
    ("rfm", "Help identify my best contacts (RFM Analysis)", RFMDialog),
    ("cluster", "Segment my contacts into clusters", DMClusterDialog),
    ("profiles", "Generate profiles of my contacts who responded to an offer",
     ProspectProfilesDialog),
    ("postal", "Identify the top responding postal codes", PostalCodeDialog),
    ("propensity", "Select contacts most likely to purchase", PropensityPurchaseDialog),
    ("package", "Compare effectiveness of campaigns (Control Package Test)",
     ControlPackageDialog),
]


class ChooseTechniqueDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Direct Marketing")
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("What do you want to do?"))
        self.group = QButtonGroup(self)
        for i, (key, text, _) in enumerate(TECHNIQUES):
            rb = QRadioButton(text)
            rb.setChecked(i == 0)
            self.group.addButton(rb, i)
            lay.addWidget(rb)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Continue")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def selected(self):
        return TECHNIQUES[self.group.checkedId()][2]


def choose_technique(editor) -> None:
    dlg = ChooseTechniqueDialog(editor)
    if dlg.exec():
        editor.run_analysis(dlg.selected())


MENUS = {"Analyze/Direct Marketing": [
    ("Choose Technique...", choose_technique), None,
    ("RFM Analysis...", RFMDialog), ("Cluster Analysis...", DMClusterDialog),
    ("Prospect Profiles...", ProspectProfilesDialog),
    ("Postal Code Response Rates...", PostalCodeDialog),
    ("Propensity to Purchase...", PropensityPurchaseDialog),
    ("Control Package Test...", ControlPackageDialog)]}


def selftest() -> None:
    marketing.selftest()
    assert number_or_none("") is None

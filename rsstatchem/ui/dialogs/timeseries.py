"""Analyze > Forecasting dialogs, Transform > Create Time Series and Data > Define Date and Time
(SPSS Forecasting module, CREATE and DATE). Core code: :mod:`rsstatchem.core.stats.timeseries`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel,
                               QListWidget, QSpinBox, QVBoxLayout)

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section
from rsstatchem.core.stats import timeseries as ts
from rsstatchem.ui.dialogs.base import NAME_ROLE, SubDialog, VariablePickerDialog, is_numeric


def _spin(value: int, lo: int, hi: int) -> QSpinBox:
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(int(value))
    return s


def _transform_box(owner: VariablePickerDialog) -> None:
    form = QFormLayout()
    owner.log_cb = owner.add_checkbox("log", "Natural log transform")
    owner.diff_spin = _spin(owner.opts.get("diff", 0), 0, 3)
    owner.sdiff_spin = _spin(owner.opts.get("sdiff", 0), 0, 2)
    form.addRow("Difference:", owner.diff_spin)
    form.addRow("Seasonally difference:", owner.sdiff_spin)
    period = ts.periodicity(owner.ds)
    form.addRow(QLabel(f"Current Periodicity: {period if period > 1 else 'None'}"))
    owner.options_layout.addLayout(form)


def _read_transform(owner) -> None:
    owner.opts["diff"] = owner.diff_spin.value()
    owner.opts["sdiff"] = owner.sdiff_spin.value()


# =========================================================================== Define Dates
class DefineDatesDialog(QDialog):
    """Data > Define Date and Time: creates YEAR_ / MONTH_ ... / DATE_ (modifies data)."""
    modifies_data = True

    def __init__(self, ds: Dataset, parent=None):
        super().__init__(parent)
        self.ds = ds
        self.setWindowTitle("Define Dates")
        self.kinds = QListWidget()
        for key, (title, _, _) in ts.DATE_KINDS.items():
            self.kinds.addItem(title)
            self.kinds.item(self.kinds.count() - 1).setData(NAME_ROLE, key)
        current = ds.meta.get("dates", {}).get("kind", "years_months")
        keys = list(ts.DATE_KINDS)
        self.kinds.setCurrentRow(keys.index(current) if current in keys else 2)
        self.first = _spin(2020, 0, 9999)
        self.second = _spin(1, 0, 999)
        form = QFormLayout()
        form.addRow("First Case Is — first field (e.g. Year):", self.first)
        form.addRow("second field (e.g. Month/Quarter/Day):", self.second)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Cases Are:"))
        lay.addWidget(self.kinds)
        lay.addLayout(form)
        lay.addWidget(buttons)

    def compute(self) -> Section:
        key = self.kinds.currentItem().data(NAME_ROLE)
        return ts.define_dates(self.ds, key, [self.first.value(), self.second.value()])


# =========================================================================== Create Time Series
class CreateTimeSeriesDialog(VariablePickerDialog):
    modifies_data = True
    DEFAULTS = {"func": "DIFF", "order": 1, "span": 3}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Create Time Series", parent)
        self.add_target("vars", "Variable -> New name:", accepts=is_numeric)
        form = QFormLayout()
        self.func = QComboBox()
        for key, text in ts.FUNCTIONS.items():
            self.func.addItem(text, key)
        self.func.setCurrentIndex(max(self.func.findData(self.opts["func"]), 0))
        self.order = _spin(self.opts["order"], 1, 24)
        self.span = _spin(self.opts["span"], 2, 99)
        period = ts.periodicity(ds)
        self.period = _spin(period if period > 1 else 12, 2, 400)
        form.addRow("Function:", self.func)
        form.addRow("Order:", self.order)
        form.addRow("Span:", self.span)
        form.addRow("Periodicity (seasonal difference):", self.period)
        self.options_layout.addLayout(form)
        self.finish()

    def _new_name(self, var: str) -> str:
        k = 1
        while self.ds.has_name(f"{var}_{k}"):
            k += 1
        return f"{var}_{k}"

    def compute(self):
        func = self.func.currentData()
        self.opts.update(func=func, order=self.order.value(), span=self.span.value())
        specs = [(self._new_name(v), func, v, self.order.value(), self.span.value(),
                  self.period.value()) for v in self.selected("vars")]
        return ts.create_time_series(self.ds, specs)


# =========================================================================== charts
class SequenceChartsDialog(VariablePickerDialog):
    DEFAULTS = {"one_chart": True, "log": False, "diff": 0, "sdiff": 0, "mean_line": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Sequence Charts", parent)
        self.add_target("vars", "Variables:", accepts=is_numeric)
        self.add_target("axis", "Time Axis Labels:", single=True, required=False)
        self.add_checkbox("one_chart", "Plot multiple variables in one chart")
        self.add_checkbox("mean_line", "Reference line at the mean of the series")
        _transform_box(self)
        self.finish()

    def compute(self):
        _read_transform(self)
        o = self.opts
        return ts.sequence_chart(self.ds, self.selected("vars"), self.selected_one("axis"),
                                 o["one_chart"], o["log"], o["diff"], o["sdiff"], o["mean_line"])


class AutocorrelationsDialog(VariablePickerDialog):
    DEFAULTS = {"acf": True, "pacf": True, "log": False, "diff": 0, "sdiff": 0,
                "max_lags": 16, "se_method": "independence", "periodic": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Autocorrelations", parent)
        self.add_target("vars", "Variables:", accepts=is_numeric)
        self.add_option_button("Options...", self._options)
        self.add_checkbox("acf", "Autocorrelations")
        self.add_checkbox("pacf", "Partial autocorrelations")
        _transform_box(self)
        self.finish()

    def _options(self):
        dlg = SubDialog("Autocorrelations: Options", self, columns=1)
        box = dlg.group("")
        dlg.number(box, "max_lags", "Maximum Number of Lags:", self.opts["max_lags"], 1, 500, 0)
        box = dlg.group("Standard Error Method")
        dlg.radio(box, "se_method", [("independence", "Independence model"),
                                     ("bartlett", "Bartlett's approximation")],
                  self.opts["se_method"])
        box = dlg.group("")
        dlg.check(box, "periodic", "Display autocorrelations at periodic lags",
                  self.opts["periodic"])
        values = dlg.run()
        if values:
            values["max_lags"] = int(values["max_lags"])
            self.opts.update(values)

    def compute(self):
        _read_transform(self)
        o = self.opts
        return ts.autocorrelations(self.ds, self.selected("vars"), int(o["max_lags"]), o["acf"],
                                   o["pacf"], o["se_method"], o["log"], o["diff"], o["sdiff"],
                                   o["periodic"])


class CrossCorrelationsDialog(VariablePickerDialog):
    DEFAULTS = {"log": False, "diff": 0, "sdiff": 0, "max_lags": 7}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Cross-Correlations", parent)
        self.add_target("vars", "Variables:", accepts=is_numeric, min_items=2)
        self.add_option_button("Options...", self._options)
        _transform_box(self)
        self.finish()

    def _options(self):
        dlg = SubDialog("Cross-Correlations: Options", self, columns=1)
        box = dlg.group("")
        dlg.number(box, "max_lags", "Maximum Number of Lags:", self.opts["max_lags"], 1, 500, 0)
        values = dlg.run()
        if values:
            self.opts["max_lags"] = int(values["max_lags"])

    def compute(self):
        _read_transform(self)
        o = self.opts
        return ts.cross_correlations(self.ds, self.selected("vars"), int(o["max_lags"]),
                                     o["log"], o["diff"], o["sdiff"])


class SeasonalDecompositionDialog(VariablePickerDialog):
    DEFAULTS = {"model": "multiplicative", "endpoints": "half", "save": True}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Seasonal Decomposition", parent)
        self.add_target("vars", "Variable(s):", accepts=is_numeric)
        self.add_option_button("Save...", self._save)
        form = QFormLayout()
        self.model = QComboBox()
        self.model.addItem("Multiplicative", "multiplicative")
        self.model.addItem("Additive", "additive")
        self.model.setCurrentIndex(max(self.model.findData(self.opts["model"]), 0))
        self.weight = QComboBox()
        self.weight.addItem("Endpoints weighted by .5", "half")
        self.weight.addItem("All points equal", "equal")
        self.weight.setCurrentIndex(max(self.weight.findData(self.opts["endpoints"]), 0))
        period = ts.periodicity(ds)
        self.period = _spin(period if period > 1 else 12, 2, 400)
        form.addRow("Model Type:", self.model)
        form.addRow("Moving Average Weight:", self.weight)
        form.addRow("Periodicity:", self.period)
        self.options_layout.addLayout(form)
        self.finish()

    def _save(self):
        dlg = SubDialog("Season: Save", self, columns=1)
        box = dlg.group("Create Variables")
        dlg.radio(box, "save", [(True, "Add to file (ERR, SAS, SAF, STC)"),
                                (False, "Do not create")], self.opts["save"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def compute(self):
        self.opts["model"] = self.model.currentData()
        self.opts["endpoints"] = self.weight.currentData()
        o = self.opts
        return ts.seasonal_decomposition(self.ds, self.selected("vars"), o["model"],
                                         o["endpoints"], self.period.value(), o["save"])


class SpectralDialog(VariablePickerDialog):
    DEFAULTS = {"window": "tukey_hamming", "span": 5, "center": True, "periodogram": True,
                "density": True, "by_period": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Spectral Plots", parent)
        self.add_target("vars", "Variable(s):", accepts=is_numeric)
        form = QFormLayout()
        self.window = QComboBox()
        for key, text in ts.WINDOWS.items():
            self.window.addItem(text, key)
        self.window.setCurrentIndex(max(self.window.findData(self.opts["window"]), 0))
        self.span = _spin(self.opts["span"], 1, 99)
        form.addRow("Spectral Window:", self.window)
        form.addRow("Span:", self.span)
        self.options_layout.addLayout(form)
        self.add_checkbox("center", "Center variables")
        self.add_checkbox("periodogram", "Plot: Periodogram")
        self.add_checkbox("density", "Plot: Spectral density")
        self.add_checkbox("by_period", "Plot by period (instead of frequency)")
        self.finish()

    def compute(self):
        self.opts["window"] = self.window.currentData()
        self.opts["span"] = self.span.value()
        o = self.opts
        return ts.spectral(self.ds, self.selected("vars"), o["window"], o["span"], o["center"],
                           o["periodogram"], o["density"], o["by_period"])


# =========================================================================== modeler
class TimeSeriesModelerDialog(VariablePickerDialog):
    DEFAULTS = {"method": "expert", "ets": "simple", "p": 0, "d": 0, "q": 0, "sp": 0, "sd": 0,
                "sq": 0, "transform": "none", "constant": True, "horizon": 12, "level": 95.0,
                "statistics": True, "params": True, "forecasts": True, "fit_plot": True,
                "residual_acf": False, "save": [], "store": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Time Series Modeler", parent)
        self.add_target("dep", "Dependent Variables:", accepts=is_numeric)
        self.add_target("ind", "Independent Variables:", accepts=is_numeric, required=False)
        self.add_option_button("Criteria...", self._criteria)
        self.add_option_button("Statistics...", self._statistics)
        self.add_option_button("Save...", self._save)
        self.add_option_button("Options...", self._options)
        form = QFormLayout()
        self.method = QComboBox()
        for key, text in (("expert", "Expert Modeler"), ("ets", "Exponential Smoothing"),
                          ("arima", "ARIMA")):
            self.method.addItem(text, key)
        self.method.setCurrentIndex(max(self.method.findData(self.opts["method"]), 0))
        self.method.currentIndexChanged.connect(
            lambda _: self.opts.__setitem__("method", self.method.currentData()))
        form.addRow("Method:", self.method)
        period = ts.periodicity(ds)
        form.addRow(QLabel(f"Current Periodicity: {period if period > 1 else 'None'} "
                           "(Data > Define Date and Time)"))
        self.options_layout.addLayout(form)
        self.finish()

    def _criteria(self):
        m = self.opts["method"]
        if m == "expert":
            dlg = SubDialog("Time Series Modeler: Expert Modeler Criteria", self, columns=1)
            box = dlg.group("Model Type")
            info = QLabel("All models (exponential smoothing and ARIMA) are considered; "
                          "the best fitting model by normalized BIC is selected.")
            info.setWordWrap(True)
            box.addWidget(info)
            dlg.run()
            return
        if m == "ets":
            dlg = SubDialog("Time Series Modeler: Exponential Smoothing Criteria", self)
            box = dlg.group("Nonseasonal")
            dlg.radio(box, "ets", [(k, v) for k, v in ts.ETS_TYPES.items()
                                   if k in ("simple", "holt", "brown", "damped")] +
                      [(k, f"Seasonal: {v}") for k, v in ts.ETS_TYPES.items()
                       if k in ("simple_seasonal", "winters_add", "winters_mult")],
                      self.opts["ets"])
            box = dlg.group("Dependent Variable Transformation")
            dlg.radio(box, "transform", [("none", "None"), ("sqrt", "Square root"),
                                         ("log", "Natural log")], self.opts["transform"])
            values = dlg.run()
            if values:
                self.opts.update(values)
            return
        dlg = SubDialog("Time Series Modeler: ARIMA Criteria", self)
        box = dlg.group("Structure (Nonseasonal / Seasonal)")
        for key, text in (("p", "Autoregressive (p)"), ("d", "Difference (d)"),
                          ("q", "Moving Average (q)"), ("sp", "Seasonal autoregressive (P)"),
                          ("sd", "Seasonal difference (D)"),
                          ("sq", "Seasonal moving average (Q)")):
            dlg.number(box, key, text + ":", self.opts[key], 0, 12, 0)
        box = dlg.group("Transformation")
        dlg.radio(box, "transform", [("none", "None"), ("sqrt", "Square root"),
                                     ("log", "Natural log")], self.opts["transform"])
        dlg.check(box, "constant", "Include constant in model", self.opts["constant"])
        values = dlg.run()
        if values:
            for k in ("p", "d", "q", "sp", "sd", "sq"):
                values[k] = int(values[k])
            self.opts.update(values)

    def _statistics(self):
        dlg = SubDialog("Time Series Modeler: Statistics", self)
        box = dlg.group("Statistics for Individual Models")
        dlg.check(box, "statistics", "Goodness of fit (model statistics)",
                  self.opts["statistics"])
        dlg.check(box, "params", "Parameter estimates", self.opts["params"])
        dlg.check(box, "forecasts", "Display forecasts", self.opts["forecasts"])
        box = dlg.group("Plots")
        dlg.check(box, "fit_plot", "Observed values, fit, forecasts and confidence intervals",
                  self.opts["fit_plot"])
        dlg.check(box, "residual_acf", "Residual autocorrelation function (ACF) and PACF",
                  self.opts["residual_acf"])
        values = dlg.run()
        if values:
            self.opts.update(values)

    def _save(self):
        dlg = SubDialog("Time Series Modeler: Save", self, columns=1)
        box = dlg.group("Save Variables")
        for key, text in (("predicted", "Predicted Values"),
                          ("lcl", "Lower Confidence Limits"),
                          ("ucl", "Upper Confidence Limits"),
                          ("nresidual", "Noise Residuals")):
            dlg.check(box, key, text, key in self.opts["save"])
        box = dlg.group("Export Model")
        dlg.check(box, "store", "Store models with the dataset (for Apply Models)",
                  self.opts["store"])
        values = dlg.run()
        if values is not None:
            self.opts["store"] = values.pop("store")
            self.opts["save"] = [k for k, v in values.items() if v]

    def _options(self):
        dlg = SubDialog("Time Series Modeler: Options", self, columns=1)
        box = dlg.group("Forecast Period")
        dlg.number(box, "horizon", "Number of periods to forecast:", self.opts["horizon"],
                   0, 1000, 0)
        dlg.number(box, "level", "Confidence Interval Width (%):", self.opts["level"],
                   50, 99.99, 2)
        values = dlg.run()
        if values:
            values["horizon"] = int(values["horizon"])
            self.opts.update(values)

    def validate(self):
        overlap = set(self.selected("dep")) & set(self.selected("ind"))
        if overlap:
            return "A variable cannot be both dependent and independent."
        # saving adds variables and forecast cases: run as an undoable data change
        self.modifies_data = bool(self.opts["save"])
        return None

    def _spec(self):
        o = self.opts
        if o["method"] == "ets":
            return ts.ModelSpec("ets", o["ets"], transform=o["transform"])
        return ts.ModelSpec("arima", order=(o["p"], o["d"], o["q"]),
                            seasonal=(o["sp"], o["sd"], o["sq"]), transform=o["transform"],
                            constant=o["constant"])

    def compute(self):
        o = self.opts
        return ts.forecast_models(self.ds, self.selected("dep"), o["method"], self._spec(),
                                  self.selected("ind"), int(o["horizon"]), self.level(),
                                  o["statistics"], o["params"], o["forecasts"], o["fit_plot"],
                                  o["residual_acf"], tuple(o["save"]), o["store"])


class ApplyModelsDialog(QDialog):
    """Apply Traditional Models: forecasts from the models stored with the dataset, either
    re-estimated from the current data or with the stored parameters."""

    def __init__(self, ds, parent=None):
        super().__init__(parent)
        self.ds = ds
        self.setWindowTitle("Apply Time Series Models")
        stored = ds.meta.get("forecast_models", {})
        self.list = QListWidget()
        for dep, info in stored.items():
            spec = ts.ModelSpec.from_dict(info["spec"])
            self.list.addItem(f"{dep}: {spec.label(info.get('period', 1))}")
            self.list.item(self.list.count() - 1).setData(NAME_ROLE, dep)
            self.list.item(self.list.count() - 1).setSelected(True)
        self.list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.mode = QComboBox()
        self.mode.addItem("Reestimate from data", "reestimate")
        self.mode.addItem("Load from model (use stored parameters)", "fixed")
        self.horizon = _spin(12, 1, 1000)
        form = QFormLayout()
        form.addRow("Model Parameters:", self.mode)
        form.addRow("Number of periods to forecast:", self.horizon)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Models stored with the dataset (Time Series Modeler > Save > "
                             "Store models):"))
        lay.addWidget(self.list)
        lay.addLayout(form)
        lay.addWidget(buttons)

    def _accept(self):
        if not self.list.selectedItems():
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(self, self.windowTitle(),
                                "No stored models. Create models with Time Series Modeler and "
                                "tick Save > Store models first.")
            return
        self.accept()

    def compute(self):
        stored = self.ds.meta.get("forecast_models", {})
        deps = [i.data(NAME_ROLE) for i in self.list.selectedItems()]
        fixed = {}
        for dep in deps:
            info = dict(stored[dep])
            if self.mode.currentData() != "fixed":
                info = {"spec": info["spec"], "params": None}
            fixed[dep] = info
        preds = stored[deps[0]].get("predictors", []) if deps else []
        return ts.forecast_models(self.ds, deps, "stored", None, preds, self.horizon.value(),
                                  fixed=fixed, title="Apply Time Series Models")


MENUS = {
    "Analyze/Forecasting": [("Create Traditional Models...", TimeSeriesModelerDialog),
                            ("Apply Traditional Models...", ApplyModelsDialog),
                            None,
                            ("Seasonal Decomposition...", SeasonalDecompositionDialog),
                            ("Spectral Analysis...", SpectralDialog),
                            None,
                            ("Sequence Charts...", SequenceChartsDialog),
                            ("Autocorrelations...", AutocorrelationsDialog),
                            ("Cross-Correlations...", CrossCorrelationsDialog)],
    "Transform": [("Create Time Series...", CreateTimeSeriesDialog)],
    "Data": [("Define Date and Time...", DefineDatesDialog)],
}


def selftest() -> None:
    """Headless check of the time-series core on a synthetic seasonal series."""
    rng = np.random.default_rng(11)
    n = 72
    t = np.arange(n)
    y = 100 + 0.8 * t + 10 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 2, n)
    ds = Dataset(pd.DataFrame({"y": y, "x": rng.normal(size=n)}))
    ts.define_dates(ds, "years_months", [2018, 1])
    assert ts.periodicity(ds) == 12
    ts.create_time_series(ds, [("y_d", "DIFF", "y", 1, 3, 12), ("y_m", "T4253H", "y", 1, 3, 12)])
    ts.sequence_chart(ds, ["y", "y_m"])
    ts.autocorrelations(ds, ["y"], diff=1)
    ts.cross_correlations(ds, ["y", "x"])
    assert ts.seasonal_decomposition(ds, ["y"]).saved
    ts.spectral(ds, ["y"])
    sec = ts.forecast_models(ds, ["y"], "arima", ts.ModelSpec("arima", order=(1, 1, 0),
                                                              seasonal=(0, 1, 1)), horizon=6)
    assert any(getattr(i, "title", "").startswith("Forecast") for i in sec.items)
    ts.forecast_models(ds, ["y"], "ets", ts.ModelSpec("ets", "winters_add"), horizon=6,
                       store=True)
    assert "forecast_models" in ds.meta

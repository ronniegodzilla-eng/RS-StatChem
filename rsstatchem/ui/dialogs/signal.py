"""Analysis > Mathematics, Signal Processing and Peaks and Baseline dialogs (OriginLab-style):
Differentiate, Integrate, Interpolate/Extrapolate, Normalize, Smoothing, FFT, Baseline
Correction, Find Peaks and Multiple Peak Fit. Also holds the small form helpers shared with
``fitting.py``."""
from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
                               QHBoxLayout, QLineEdit, QSpinBox)

from rsstatchem.core.analysis import mathematics as m
from rsstatchem.core.analysis import signal as sg
from rsstatchem.core.analysis.common import parse_numbers
from rsstatchem.ui.dialogs.base import VariablePickerDialog, is_numeric


class FormMixin:
    """Option widgets bound to ``self.opts`` (remembered per dialog class for the session)."""

    def form_group(self, title: str, new_row: bool = False) -> QFormLayout:
        if new_row or not hasattr(self, "_group_row"):
            self._group_row = QHBoxLayout()
            self.options_layout.addLayout(self._group_row)
        box = QGroupBox(title)
        form = QFormLayout(box)
        self._group_row.addWidget(box, 1)
        return form

    def opt_combo(self, form, label: str, key: str, items: dict) -> QComboBox:
        combo = QComboBox()
        for value, text in items.items():
            combo.addItem(text, value)
        combo.setCurrentIndex(max(combo.findData(self.opts[key]), 0))
        self.opts[key] = combo.currentData()
        combo.currentIndexChanged.connect(
            lambda _=0, k=key, c=combo: (self.opts.__setitem__(k, c.currentData()),
                                         self.options_changed()))
        form.addRow(label, combo)
        return combo

    def opt_spin(self, form, label: str, key: str, lo: float, hi: float, decimals: int = 0,
                 step: float | None = None, suffix: str = ""):
        if decimals == 0:
            spin = QSpinBox()
            spin.setRange(int(lo), int(hi))
            spin.setValue(int(self.opts[key]))
        else:
            spin = QDoubleSpinBox()
            spin.setDecimals(decimals)
            spin.setRange(lo, hi)
            spin.setValue(float(self.opts[key]))
        if step:
            spin.setSingleStep(step)
        if suffix:
            spin.setSuffix(suffix)
        spin.valueChanged.connect(lambda v, k=key: self.opts.__setitem__(k, v))
        form.addRow(label, spin)
        return spin

    def opt_check(self, form, key: str, text: str) -> QCheckBox:
        cb = QCheckBox(text)
        cb.setChecked(bool(self.opts[key]))
        cb.toggled.connect(lambda on, k=key: (self.opts.__setitem__(k, on),
                                              self.options_changed()))
        form.addRow(cb)
        return cb

    def opt_line(self, form, label: str, key: str, placeholder: str = "") -> QLineEdit:
        edit = QLineEdit(str(self.opts.get(key) or ""))
        edit.setPlaceholderText(placeholder)
        edit.textChanged.connect(lambda t, k=key: self.opts.__setitem__(k, t))
        form.addRow(label, edit)
        return edit

    def options_changed(self) -> None:
        """Hook: enable/disable widgets that depend on another option."""

    def number_or_none(self, key: str, what: str) -> float | None:
        text = str(self.opts.get(key) or "").strip().replace(",", ".")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            raise ValueError(f"{what}: '{text}' is not a number.") from None

    def check_numbers(self, *keys_and_names) -> str | None:
        for key, what in keys_and_names:
            try:
                self.number_or_none(key, what)
            except ValueError as exc:
                return str(exc)
        return None


class _XYDialog(FormMixin, VariablePickerDialog):
    """Optional X (row number when empty) and one or several Y variables."""

    multi_y = True

    def __init__(self, ds, title, parent=None, x_required=False):
        super().__init__(ds, title, parent)
        self.setMinimumSize(640, 460)
        self.add_target("y", "Y Variable(s):" if self.multi_y else "Y Variable:",
                        accepts=is_numeric, single=not self.multi_y)
        self.add_target("x", "X Variable:" if x_required else "X Variable (optional; row "
                        "number if empty):", accepts=is_numeric, single=True,
                        required=x_required)

    def ys(self) -> list[str]:
        return self.selected("y")

    def x(self) -> str | None:
        return self.selected_one("x")


# ---------------------------------------------------------------------- mathematics
class DifferentiateDialog(_XYDialog):
    DEFAULTS = {"order": 1, "method": "central", "window": 7, "polyorder": 2, "plot": True}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Differentiate", parent)
        form = self.form_group("Derivative")
        self.opt_combo(form, "Derivative Order:", "order", {1: "First", 2: "Second"})
        self.opt_combo(form, "Method:", "method", {"central": "Central difference",
                                                    "savgol": "Savitzky-Golay"})
        form = self.form_group("Savitzky-Golay")
        self.w_window = self.opt_spin(form, "Points of Window:", "window", 3, 999)
        self.w_poly = self.opt_spin(form, "Polynomial Order:", "polyorder", 1, 9)
        self.opt_check(form, "plot", "Plot derivative")
        self.options_changed()
        self.finish()

    def options_changed(self):
        if hasattr(self, "w_poly"):
            on = self.opts["method"] == "savgol"
            self.w_window.setEnabled(on)
            self.w_poly.setEnabled(on)

    def compute(self):
        o = self.opts
        return m.differentiate_report(self.ds, self.x(), self.ys(), int(o["order"]), o["method"],
                                      int(o["window"]), int(o["polyorder"]), o["plot"])


class IntegrateDialog(_XYDialog):
    DEFAULTS = {"baseline": "zero", "value": "", "x_from": "", "x_to": "", "save": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Integrate", parent)
        form = self.form_group("Area")
        self.opt_combo(form, "Baseline:", "baseline", m.BASELINES)
        self.w_value = self.opt_line(form, "Constant Baseline Y:", "value", "0")
        self.opt_check(form, "save", "Save cumulative integral as new variable(s)")
        form = self.form_group("Integration Range (X)")
        self.opt_line(form, "From:", "x_from", "first X")
        self.opt_line(form, "To:", "x_to", "last X")
        self.options_changed()
        self.finish()

    def options_changed(self):
        if hasattr(self, "w_value"):
            self.w_value.setEnabled(self.opts["baseline"] == "constant")

    def validate(self):
        return self.check_numbers(("value", "Baseline"), ("x_from", "From"), ("x_to", "To"))

    def compute(self):
        o = self.opts
        return m.integrate_report(self.ds, self.x(), self.ys(), o["baseline"],
                                  self.number_or_none("value", "Baseline") or 0.0,
                                  self.number_or_none("x_from", "From"),
                                  self.number_or_none("x_to", "To"), o["save"])


class InterpolateDialog(_XYDialog):
    multi_y = False
    DEFAULTS = {"method": "cubic", "points": 100, "x_from": "", "x_to": "",
                "extrapolate": False, "save": True}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Interpolate/Extrapolate", parent, x_required=True)
        self.add_target("at", "Interpolate at X Values (optional):", accepts=is_numeric,
                        single=True, required=False)
        form = self.form_group("Method")
        self.opt_combo(form, "Method:", "method", m.INTERP_METHODS)
        self.opt_check(form, "extrapolate", "Allow extrapolation outside the data range")
        self.opt_check(form, "save", "Save interpolated values as new variable(s)")
        form = self.form_group("Uniform X Grid (when no X values are given)")
        self.opt_spin(form, "Number of Points:", "points", 2, 100000)
        self.opt_line(form, "X From:", "x_from", "minimum X")
        self.opt_line(form, "X To:", "x_to", "maximum X")
        self.finish()

    def validate(self):
        return self.check_numbers(("x_from", "X From"), ("x_to", "X To"))

    def compute(self):
        o = self.opts
        return m.interpolate_report(self.ds, self.x(), self.ys()[0], o["method"],
                                    self.selected_one("at"), int(o["points"]),
                                    self.number_or_none("x_from", "X From"),
                                    self.number_or_none("x_to", "X To"), o["extrapolate"],
                                    o["save"])


class NormalizeDialog(_XYDialog):
    DEFAULTS = {"method": "range", "value": "", "plot": True}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Normalize", parent)
        form = self.form_group("Normalize Method")
        self.opt_combo(form, "Method:", "method", m.NORMALIZE)
        self.w_value = self.opt_line(form, "Divide by Value:", "value", "e.g. 100")
        self.opt_check(form, "plot", "Plot normalized data")
        self.options_changed()
        self.finish()

    def options_changed(self):
        if hasattr(self, "w_value"):
            self.w_value.setEnabled(self.opts["method"] == "value")

    def validate(self):
        if self.opts["method"] == "value" and not self.number_or_none("value", "Value"):
            return "Enter a non-zero value to divide by."
        return self.check_numbers(("value", "Value"))

    def compute(self):
        o = self.opts
        return m.normalize_report(self.ds, self.ys(), o["method"], self.x(),
                                  self.number_or_none("value", "Value"), o["plot"])


# ---------------------------------------------------------------------- signal processing
class SmoothDialog(_XYDialog):
    DEFAULTS = {"method": "savgol", "window": 7, "polyorder": 2, "percentile": 50.0,
                "cutoff": 0.1, "frac": 0.2, "plot": True}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Smoothing", parent)
        form = self.form_group("Method")
        self.opt_combo(form, "Method:", "method", sg.SMOOTHERS)
        self.w_window = self.opt_spin(form, "Points of Window:", "window", 2, 9999)
        self.w_poly = self.opt_spin(form, "Polynomial Order:", "polyorder", 0, 9)
        form = self.form_group("Method Parameters")
        self.w_pct = self.opt_spin(form, "Percentile:", "percentile", 0, 100, 1, 5, " %")
        self.w_cut = self.opt_spin(form, "FFT Cutoff (× Nyquist):", "cutoff", 0.001, 1, 3, 0.01)
        self.w_frac = self.opt_spin(form, "LOWESS Span (fraction):", "frac", 0.01, 1, 3, 0.05)
        self.opt_check(form, "plot", "Plot raw and smoothed data")
        self.options_changed()
        self.finish()

    def options_changed(self):
        if not hasattr(self, "w_frac"):
            return
        meth = self.opts["method"]
        self.w_window.setEnabled(meth in ("savgol", "average", "percentile"))
        self.w_poly.setEnabled(meth == "savgol")
        self.w_pct.setEnabled(meth == "percentile")
        self.w_cut.setEnabled(meth == "fft")
        self.w_frac.setEnabled(meth == "lowess")

    def validate(self):
        if self.opts["method"] == "savgol" and self.opts["polyorder"] >= self.opts["window"]:
            return "The polynomial order must be smaller than the number of points."
        return None

    def compute(self):
        o = self.opts
        return sg.smooth_report(self.ds, self.x(), self.ys(), o["method"], int(o["window"]),
                                int(o["polyorder"]), o["percentile"], o["cutoff"], o["frac"],
                                o["plot"])


class FFTDialog(_XYDialog):
    multi_y = False
    DEFAULTS = {"window": "hann", "detrend": True, "interval": 1.0, "top": 10, "log": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "FFT", parent)
        form = self.form_group("FFT Settings")
        self.opt_combo(form, "Window:", "window", sg.WINDOWS)
        self.opt_check(form, "detrend", "Remove the mean before the FFT")
        self.opt_check(form, "log", "Plot power on a log scale")
        form = self.form_group("Sampling")
        self.w_dt = self.opt_spin(form, "Sampling Interval (no X):", "interval", 1e-9, 1e9, 6)
        self.opt_spin(form, "Dominant Frequencies Listed:", "top", 1, 100)
        self.finish()

    def targets_changed(self):
        if hasattr(self, "w_dt"):
            self.w_dt.setEnabled(not self.selected("x"))

    def compute(self):
        o = self.opts
        return sg.fft_report(self.ds, self.x(), self.ys()[0], o["window"], o["detrend"],
                             o["interval"], int(o["top"]), o["log"])


class BaselineDialog(_XYDialog):
    multi_y = False
    DEFAULTS = {"method": "als", "order": 2, "loglam": 5.0, "p": 0.01, "iter": 10,
                "save": True}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Baseline Correction", parent)
        form = self.form_group("Baseline Method")
        self.opt_combo(form, "Method:", "method", sg.BASELINE_METHODS)
        self.w_order = self.opt_spin(form, "Polynomial Order:", "order", 0, 9)
        self.opt_check(form, "save", "Save baseline and corrected signal")
        form = self.form_group("Asymmetric Least Squares")
        self.w_lam = self.opt_spin(form, "Smoothness log₁₀(λ):", "loglam", 0, 12, 1, 0.5)
        self.w_p = self.opt_spin(form, "Asymmetry p:", "p", 0.0001, 0.5, 4, 0.005)
        self.w_it = self.opt_spin(form, "Iterations:", "iter", 1, 100)
        self.options_changed()
        self.finish()

    def options_changed(self):
        if not hasattr(self, "w_it"):
            return
        als = self.opts["method"] == "als"
        self.w_order.setEnabled(not als)
        for w in (self.w_lam, self.w_p, self.w_it):
            w.setEnabled(als)

    def compute(self):
        o = self.opts
        return sg.baseline_report(self.ds, self.x(), self.ys()[0], o["method"], int(o["order"]),
                                  10 ** o["loglam"], o["p"], int(o["iter"]), o["save"])


class FindPeaksDialog(_XYDialog):
    multi_y = False
    DEFAULTS = {"direction": "positive", "height": "", "prominence": 5.0, "distance": 1,
                "width": 0.0}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Find Peaks", parent)
        form = self.form_group("Peak Filtering")
        self.opt_combo(form, "Direction:", "direction", {"positive": "Positive peaks",
                                                         "negative": "Negative peaks"})
        self.opt_line(form, "Minimum Height:", "height", "none")
        self.opt_spin(form, "Minimum Prominence:", "prominence", 0, 100, 1, 1,
                      " % of Y range")
        form = self.form_group("Separation")
        self.opt_spin(form, "Minimum Distance (points):", "distance", 1, 100000)
        self.opt_spin(form, "Minimum FWHM (X units):", "width", 0, 1e9, 4)
        self.finish()

    def validate(self):
        return self.check_numbers(("height", "Minimum Height"))

    def compute(self):
        o = self.opts
        return sg.peaks_report(self.ds, self.x(), self.ys()[0], o["direction"],
                               self.number_or_none("height", "Minimum Height"),
                               o["prominence"] / 100, int(o["distance"]), o["width"])


class MultiPeakFitDialog(_XYDialog):
    multi_y = False
    DEFAULTS = {"shape": "gauss", "n": 0, "baseline": "constant", "centers": "",
                "prominence": 5.0, "level": 95.0, "save": False}

    def __init__(self, ds, parent=None):
        super().__init__(ds, "Multiple Peak Fit", parent)
        form = self.form_group("Peaks")
        self.opt_combo(form, "Peak Function:", "shape", sg.SHAPES)
        self.opt_spin(form, "Number of Peaks (0 = all found):", "n", 0, 50)
        self.opt_line(form, "Peak Centers (optional):", "centers", "e.g. 12.5, 14")
        self.opt_spin(form, "Minimum Prominence:", "prominence", 0, 100, 1, 1, " % of Y range")
        form = self.form_group("Fit")
        self.opt_combo(form, "Baseline:", "baseline", {"none": "None", "constant": "Constant",
                                                        "linear": "Linear"})
        self.opt_spin(form, "Confidence Level:", "level", 50, 99.99, 2, 1, " %")
        self.opt_check(form, "save", "Save fitted curve as new variable")
        self.finish()

    def validate(self):
        try:
            parse_numbers(self.opts["centers"])
        except ValueError as exc:
            return f"Peak centers: {exc}"
        return None

    def compute(self):
        o = self.opts
        centers = parse_numbers(o["centers"]) or None
        return sg.multi_peak_report(self.ds, self.x(), self.ys()[0], int(o["n"]) or None,
                                    o["shape"], o["baseline"], centers, o["prominence"] / 100,
                                    o["level"] / 100, o["save"])

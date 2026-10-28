"""Peak Analyzer wizard (Origin's Analysis > Peaks and Baseline > Peak Analyzer).

Pages as in Origin: Goal → Baseline Mode → Baseline Anchor Points (user-defined mode) →
Find Peaks → Integrate Peaks / Fit Peaks → Finish, with a live preview of the data, the
baseline, the anchor points, the peaks and the fitted components on the right. Anchor points
and peaks can be found automatically and edited in tables (add / remove / move). The wizard
edits a Qt-free :class:`~rsstatchem.core.analysis.peak_analyzer.PeakAnalysis`; Finish
returns its settings, which :mod:`fitting_menus` turns into the PeakAnalyzer report.
"""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                               QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QPlainTextEdit, QPushButton, QRadioButton,
                               QSpinBox, QSplitter, QStackedWidget, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from rsstatchem.core.analysis import baselines as bl
from rsstatchem.core.analysis import peak_analyzer as pa
from rsstatchem.core.analysis.common import num
from rsstatchem.ui.origin import fit_reports as fr
from rsstatchem.ui.origin.model import RECALC_MODES

PAGES = ("goal", "baseline", "anchors", "find", "integrate", "fit", "finish")
TITLES = {"goal": "Goal", "baseline": "Baseline Mode", "anchors": "Baseline Anchor Points",
          "find": "Find Peaks", "integrate": "Integrate Peaks", "fit": "Fit Peaks",
          "finish": "Finish"}


def _float_box(lo, hi, val, dec=3, step=None):
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(dec)
    w.setValue(val)
    if step:
        w.setSingleStep(step)
    return w


def _int_box(lo, hi, val):
    w = QSpinBox()
    w.setRange(lo, hi)
    w.setValue(val)
    return w


class PeakAnalyzerDialog(QDialog):
    def __init__(self, inputs, parent=None, current: int = 0, resolve=None,
                 settings: dict | None = None, goal: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("Peak Analyzer")
        self.inputs = inputs
        self.resolve = resolve
        self._busy = False
        self.input = QComboBox()
        for c in inputs:
            self.input.addItem(c.label)
        self.input.setCurrentIndex(max(0, min(current, len(inputs) - 1)))
        self.recalc = QComboBox()
        for k, t in RECALC_MODES.items():
            self.recalc.addItem(t, k)
        self.page_title = QLabel()
        self.page_title.setStyleSheet("font-weight: bold; font-size: 11pt; color:#1f3d7a")
        self.stack = QStackedWidget()
        self.pages = {}
        for key in PAGES:
            w = getattr(self, f"_page_{key}")()
            self.pages[key] = w
            self.stack.addWidget(w)
        self.figure = Figure(figsize=(5.2, 3.8), dpi=96)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot()
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#444")
        left = QWidget()
        lv = QVBoxLayout(left)
        f = QFormLayout()
        f.addRow("Input:", self.input)
        f.addRow("Recalculate:", self.recalc)
        lv.addLayout(f)
        lv.addWidget(self.page_title)
        lv.addWidget(self.stack, 1)
        lv.addWidget(self.status)
        split = QSplitter()
        split.addWidget(left)
        split.addWidget(self.canvas)
        split.setSizes([420, 520])
        nav = QHBoxLayout()
        self.prev_btn = QPushButton("< Prev")
        self.next_btn = QPushButton("Next >")
        self.finish_btn = QPushButton("Finish")
        cancel = QPushButton("Cancel")
        self.prev_btn.clicked.connect(lambda: self.go(-1))
        self.next_btn.clicked.connect(lambda: self.go(1))
        self.finish_btn.clicked.connect(self.finish)
        cancel.clicked.connect(self.reject)
        nav.addStretch(1)
        for b in (self.prev_btn, self.next_btn, self.finish_btn, cancel):
            nav.addWidget(b)
        lay = QVBoxLayout(self)
        lay.addWidget(split, 1)
        lay.addLayout(nav)
        self.resize(1000, 600)
        self.input.currentIndexChanged.connect(self._load_data)
        self._load_data()
        if settings:
            self.apply_settings(settings)
        if goal:
            self.set_goal(goal)
        self.page = "goal"
        self._show_page()

    # ------------------------------------------------------------------ pages
    def _page_goal(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self.goal_group = QButtonGroup(self)
        for i, (k, t) in enumerate(pa.GOALS.items()):
            rb = QRadioButton(t)
            rb.setProperty("goal", k)
            self.goal_group.addButton(rb, i)
            v.addWidget(rb)
            if k == "fit":
                rb.setChecked(True)
        v.addStretch(1)
        self.goal_group.buttonClicked.connect(lambda _b: self._settings_changed())
        return w

    def _page_baseline(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.bmode = QComboBox()
        for k, t in bl.BASELINE_MODES.items():
            if k != "dataset":
                self.bmode.addItem(t, k)
        self.bmode.setCurrentIndex(self.bmode.findData("none"))
        self.lam = QLineEdit("1e5")
        self.p_asym = _float_box(1e-4, 0.5, 0.01, 4, 0.005)
        self.half_window = _int_box(1, 10000, 40)
        self.radius = _int_box(1, 10000, 50)
        self.order = _int_box(0, 9, 2)
        self.n_end = _int_box(1, 1000, 1)
        self.n_anchors = _int_box(2, 200, 8)
        self.anchor_method = QComboBox()
        for k, t in bl.ANCHOR_METHODS.items():
            self.anchor_method.addItem(t, k)
        self.interp = QComboBox()
        for k, t in {**bl.INTERPOLATIONS, "function": "Fit a Function"}.items():
            self.interp.addItem(t, k)
        self.bfunc = QLineEdit("Poly3")
        f.addRow("Baseline Mode:", self.bmode)
        f.addRow("λ (ALS / arPLS / airPLS):", self.lam)
        f.addRow("Asymmetry p (ALS):", self.p_asym)
        f.addRow("Half Window (SNIP):", self.half_window)
        f.addRow("Radius (Rolling Ball):", self.radius)
        f.addRow("Polynomial Order:", self.order)
        f.addRow("End Points to Average:", self.n_end)
        f.addRow("Number of Anchor Points:", self.n_anchors)
        f.addRow("Anchor Finding Method:", self.anchor_method)
        f.addRow("Connect Anchors by:", self.interp)
        f.addRow("Baseline Function:", self.bfunc)
        for wd in (self.bmode, self.anchor_method, self.interp):
            wd.currentIndexChanged.connect(self._settings_changed)
        for wd in (self.p_asym, self.half_window, self.radius, self.order, self.n_end,
                   self.n_anchors):
            wd.valueChanged.connect(self._settings_changed)
        self.lam.editingFinished.connect(self._settings_changed)
        self.bfunc.editingFinished.connect(self._settings_changed)
        return w

    def _page_anchors(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self.anchor_table = QTableWidget(0, 2)
        self.anchor_table.setHorizontalHeaderLabels(["X", "Y"])
        self.anchor_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.anchor_table.itemChanged.connect(self._anchors_edited)
        row = QHBoxLayout()
        for text, slot in (("Find", self.find_anchors), ("Add", self.add_anchor),
                           ("Remove", self.remove_anchor), ("Snap to Data", self.snap_anchors)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        v.addLayout(row)
        v.addWidget(self.anchor_table)
        return w

    def _page_find(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        f = QFormLayout()
        self.fmethod = QComboBox()
        for k, t in pa.FIND_METHODS.items():
            self.fmethod.addItem(t, k)
        self.direction = QComboBox()
        self.direction.addItem("Positive", "positive")
        self.direction.addItem("Negative", "negative")
        self.height_pct = _float_box(0, 100, 5.0, 1, 1.0)
        self.max_peaks = _int_box(0, 200, 0)
        self.smooth = _int_box(0, 1001, 0)
        self.window = _int_box(1, 1001, 5)
        f.addRow("Method:", self.fmethod)
        f.addRow("Direction:", self.direction)
        f.addRow("Peak Filter: Height ≥ (% of range):", self.height_pct)
        f.addRow("Number of Peaks (0 = all):", self.max_peaks)
        f.addRow("Smoothing Window (0 = auto):", self.smooth)
        f.addRow("Window Size (Window Search):", self.window)
        v.addLayout(f)
        self.peak_table = QTableWidget(0, 4)
        self.peak_table.setHorizontalHeaderLabels(["X", "Y", "FWHM", "Hidden"])
        self.peak_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        row = QHBoxLayout()
        self.add_x = QLineEdit()
        self.add_x.setPlaceholderText("X of a peak to add")
        for text, slot in (("Find", self.find_peaks), ("Add", self.add_peak),
                           ("Remove", self.remove_peak)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addWidget(self.add_x)
        v.addLayout(row)
        v.addWidget(self.peak_table)
        for wd in (self.fmethod, self.direction):
            wd.currentIndexChanged.connect(self._find_changed)
        for wd in (self.height_pct, self.max_peaks, self.smooth, self.window):
            wd.valueChanged.connect(self._find_changed)
        return w

    def _page_integrate(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self.int_table = QTableWidget(0, 6)
        self.int_table.setHorizontalHeaderLabels(["From", "To", "Area", "Area %", "Centroid",
                                                  "FWHM"])
        self.int_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        row = QHBoxLayout()
        self.auto_limits = QCheckBox("Automatic integration limits")
        self.auto_limits.setChecked(True)
        b = QPushButton("Integrate")
        b.clicked.connect(self.integrate)
        row.addWidget(self.auto_limits)
        row.addStretch(1)
        row.addWidget(b)
        v.addLayout(row)
        v.addWidget(QLabel("Edit From / To to set the limits of a peak, then Integrate."))
        v.addWidget(self.int_table)
        return w

    def _page_fit(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        f = QFormLayout()
        self.shape = QComboBox()
        for k, s in pa.SHAPES.items():
            self.shape.addItem(s["label"], k)
        self.cofit = QComboBox()
        for k, t in pa.FIT_BASELINES.items():
            self.cofit.addItem(t, k)
        self.shared = QCheckBox("Share the width of all peaks")
        self.fixed_width = _float_box(0, 1e9, 0.0, 4)
        self.fix_centers = QCheckBox("Fix peak centres")
        f.addRow("Peak Function:", self.shape)
        f.addRow("Baseline Co-fit:", self.cofit)
        f.addRow("", self.shared)
        f.addRow("Fixed Width (0 = fit):", self.fixed_width)
        f.addRow("", self.fix_centers)
        v.addLayout(f)
        b = QPushButton("Fit")
        b.clicked.connect(self.fit)
        v.addWidget(b)
        self.fit_table = QTableWidget(0, 5)
        self.fit_table.setHorizontalHeaderLabels(["Center", "Height", "Area", "Area %",
                                                  "FWHM"])
        self.fit_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        v.addWidget(self.fit_table)
        return w

    def _page_finish(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        v.addWidget(QLabel("Finish creates the PeakAnalyzer report and data sheet."))
        v.addWidget(self.summary)
        return w

    # ------------------------------------------------------------------ state
    def _load_data(self) -> None:
        if not self.inputs or self.resolve is None:
            self.analysis = pa.PeakAnalysis(np.arange(10.0), np.zeros(10))
            return
        data = self.resolve(self.inputs[self.input.currentIndex()])
        self.data = data
        self.analysis = pa.PeakAnalysis(data.x, data.y)
        self._settings_changed()

    @property
    def goal(self) -> str:
        b = self.goal_group.checkedButton()
        return b.property("goal") if b is not None else "fit"

    def set_goal(self, goal: str) -> None:
        for b in self.goal_group.buttons():
            b.setChecked(b.property("goal") == goal)
        self._settings_changed()

    def sequence(self) -> list[str]:
        seq = ["goal", "baseline"]
        if self.bmode.currentData() == "anchors":
            seq.append("anchors")
        if self.goal in ("find", "integrate", "fit"):
            seq.append("find")
        if self.goal == "integrate":
            seq.append("integrate")
        if self.goal == "fit":
            seq.append("fit")
        return seq + ["finish"]

    def baseline_opts(self) -> dict:
        try:
            lam = float(self.lam.text().replace(",", "."))
        except ValueError:
            lam = 1e5
        o = {"lam": lam, "p": self.p_asym.value(), "half_window": self.half_window.value(),
             "radius": self.radius.value(), "order": self.order.value(),
             "n_end": self.n_end.value(), "n_anchors": self.n_anchors.value(),
             "anchor_method": self.anchor_method.currentData(),
             "interp": self.interp.currentData()}
        if o["interp"] == "function":
            o["function"] = self.bfunc.text().strip() or "Poly3"
        return o

    def find_opts(self) -> dict:
        return {"method": self.fmethod.currentData(), "direction": self.direction.currentData(),
                "height_pct": self.height_pct.value(),
                "max_peaks": self.max_peaks.value() or None,
                "smooth_window": self.smooth.value(), "window_size": self.window.value()}

    def fit_opts(self) -> dict:
        return {"shape": self.shape.currentData(), "baseline": self.cofit.currentData(),
                "shared_width": self.shared.isChecked(),
                "fixed_width": self.fixed_width.value() or None,
                "fix_centers": self.fix_centers.isChecked()}

    def _settings_changed(self) -> None:
        if self._busy or not hasattr(self, "fit_table"):
            return
        an = self.analysis
        an.goal = self.goal
        an.baseline_mode = self.bmode.currentData()
        old = dict(an.baseline_opts)
        an.baseline_opts = self.baseline_opts()
        if an.baseline_mode == "anchors" and (an.anchors is None or any(
                old.get(k) != an.baseline_opts.get(k) for k in ("n_anchors", "anchor_method"))):
            an.anchors = None
        try:
            an.compute_baseline() if an.baseline_mode != "none" else setattr(an, "baseline",
                                                                            None)
            self.status.setText("")
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            an.baseline = None
            self.status.setText(f"Baseline: {exc}")
        self._fill_anchor_table()
        self.preview()

    def _find_changed(self) -> None:
        self.analysis.find_opts = self.find_opts()
        if self.page in ("find", "integrate", "fit"):
            self.find_peaks()

    # anchors
    def _fill_anchor_table(self) -> None:
        an = self.analysis
        self._busy = True
        self.anchor_table.setRowCount(0)
        if an.anchors is not None:
            for i, (a, b) in enumerate(zip(*an.anchors)):
                self.anchor_table.insertRow(i)
                self.anchor_table.setItem(i, 0, QTableWidgetItem(num(a)))
                self.anchor_table.setItem(i, 1, QTableWidgetItem(num(b)))
        self._busy = False

    def _anchors_edited(self) -> None:
        if self._busy:
            return
        xs, ys = [], []
        for i in range(self.anchor_table.rowCount()):
            try:
                xs.append(float(self.anchor_table.item(i, 0).text()))
                ys.append(float(self.anchor_table.item(i, 1).text()))
            except (AttributeError, ValueError):
                continue
        self.set_anchors(xs, ys)

    def set_anchors(self, xs, ys) -> None:
        an = self.analysis
        an.anchors = (np.asarray(xs, float), np.asarray(ys, float))
        try:
            an.compute_baseline()
        except (ValueError, RuntimeError) as exc:
            self.status.setText(f"Baseline: {exc}")
        self._fill_anchor_table()
        self.preview()

    def find_anchors(self) -> None:
        self.analysis.find_anchors(self.n_anchors.value(), self.anchor_method.currentData())
        self.set_anchors(*self.analysis.anchors)

    def add_anchor(self) -> None:
        an = self.analysis
        ax = list(an.anchors[0]) if an.anchors is not None else []
        ay = list(an.anchors[1]) if an.anchors is not None else []
        x = float(np.median(an.x)) if not ax else (ax[-1] + an.x.max()) / 2
        ax.append(x)
        ay.append(float(np.interp(x, an.x, an.y)))
        o = np.argsort(ax)
        self.set_anchors(np.array(ax)[o], np.array(ay)[o])

    def remove_anchor(self) -> None:
        rows = sorted({i.row() for i in self.anchor_table.selectedItems()}, reverse=True)
        an = self.analysis
        if an.anchors is None:
            return
        ax, ay = list(an.anchors[0]), list(an.anchors[1])
        for r in rows or [len(ax) - 1]:
            if 0 <= r < len(ax):
                del ax[r], ay[r]
        self.set_anchors(ax, ay)

    def snap_anchors(self) -> None:
        an = self.analysis
        if an.anchors is not None:
            self.set_anchors(an.anchors[0], np.interp(an.anchors[0], an.x, an.y))

    # peaks
    def find_peaks(self) -> None:
        an = self.analysis
        an.find_opts = self.find_opts()
        try:
            an.find()
            an.keep_peaks = False
            self.status.setText(f"{len(an.peaks)} peak(s) found.")
        except (ValueError, RuntimeError) as exc:
            an.peaks = []
            self.status.setText(str(exc))
        an.fit = None
        an.integration = []
        self._fill_peak_table()
        self.preview()

    def _fill_peak_table(self) -> None:
        t = self.peak_table
        t.setRowCount(0)
        for i, p in enumerate(self.analysis.peaks):
            t.insertRow(i)
            for j, v in enumerate((num(p["x"]), num(p["y"]), num(p["fwhm"]),
                                   "Yes" if p.get("hidden") else "")):
                t.setItem(i, j, QTableWidgetItem(v))

    def add_peak(self, x: float | None = None) -> None:
        an = self.analysis
        if x is None or isinstance(x, bool):
            try:
                x = float(self.add_x.text().replace(",", "."))
            except ValueError:
                self.status.setText("Enter the X of the peak to add.")
                return
        y = float(np.interp(x, an.x, an.subtracted))
        an.peaks.append({"index": int(np.argmin(np.abs(an.x - x))), "x": float(x), "y": y,
                         "height": y, "fwhm": float(np.ptp(an.x)) / 20, "hidden": True})
        an.peaks.sort(key=lambda p: p["x"])
        an.keep_peaks = True
        self._fill_peak_table()
        self.preview()

    def remove_peak(self) -> None:
        rows = sorted({i.row() for i in self.peak_table.selectedItems()}, reverse=True)
        for r in rows:
            if 0 <= r < len(self.analysis.peaks):
                del self.analysis.peaks[r]
        self.analysis.keep_peaks = True
        self._fill_peak_table()
        self.preview()

    # integrate / fit
    def integrate(self) -> None:
        an = self.analysis
        if not an.peaks:
            self.find_peaks()
        limits = None
        if not self.auto_limits.isChecked() and self.int_table.rowCount() == len(an.peaks):
            limits = []
            for i in range(self.int_table.rowCount()):
                try:
                    limits.append((float(self.int_table.item(i, 0).text()),
                                   float(self.int_table.item(i, 1).text())))
                except (AttributeError, ValueError):
                    limits.append(None)
        an.limits = limits
        an.integrate()
        t = self.int_table
        t.setRowCount(0)
        for i, r in enumerate(an.integration):
            t.insertRow(i)
            for j, k in enumerate(("x_from", "x_to", "area", "area_pct", "centroid", "fwhm")):
                t.setItem(i, j, QTableWidgetItem(num(r[k], 6)))
        self.preview()

    def fit(self) -> None:
        an = self.analysis
        if not an.peaks:
            self.find_peaks()
        an.fit_opts = self.fit_opts()
        try:
            an.fit_peaks()
            self.status.setText(f"Fit: R² = {num(an.fit.result.r2)}  {an.fit.result.message}")
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            an.fit = None
            self.status.setText(f"Fit failed: {exc}")
        t = self.fit_table
        t.setRowCount(0)
        if an.fit is not None:
            for i, r in enumerate(an.fit.peaks):
                t.insertRow(i)
                for j, k in enumerate(("center", "height", "area", "area_pct", "fwhm")):
                    t.setItem(i, j, QTableWidgetItem(num(r[k], 6)))
        self.preview()

    # ------------------------------------------------------------------ preview
    def preview(self) -> None:
        an = self.analysis
        ax = self.ax
        ax.clear()
        ax.plot(an.x, an.y, color="k", lw=1.0, label="Data")
        if an.baseline is not None:
            ax.plot(an.x, an.baseline, color="#d40000", ls="--", lw=1.2, label="Baseline")
        if an.anchors is not None and an.baseline_mode == "anchors":
            ax.plot(an.anchors[0], an.anchors[1], "o", ms=5, mfc="none", color="#d40000",
                    label="Anchor Points")
        base = an.baseline if an.baseline is not None else np.zeros_like(an.y)
        if an.peaks:
            px = [p["x"] for p in an.peaks]
            py = np.interp(px, an.x, an.y)
            ax.plot(px, py, "v", color="#1f5fbf", ms=7, label="Peaks")
        if an.integration:
            for r in an.integration:
                m = (an.x >= r["x_from"]) & (an.x <= r["x_to"])
                ax.fill_between(an.x[m], base[m], an.y[m], alpha=0.25)
        if an.fit is not None:
            xx = np.linspace(an.x.min(), an.x.max(), 600)
            bxx = np.interp(xx, an.x, base) + an.fit.baseline(xx)
            for c in an.fit.components(xx):
                ax.plot(xx, c + bxx, lw=0.9)
            ax.plot(xx, an.fit.cumulative(xx) + np.interp(xx, an.x, base), color="#ff0000",
                    lw=1.5, label="Cumulative Fit")
        ax.legend(fontsize=7, frameon=False, loc="best")
        ax.tick_params(labelsize=8)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ navigation
    def _show_page(self) -> None:
        seq = self.sequence()
        if self.page not in seq:
            self.page = seq[0]
        self.stack.setCurrentWidget(self.pages[self.page])
        i = seq.index(self.page)
        self.page_title.setText(f"{i + 1}/{len(seq)}  {TITLES[self.page]}")
        self.prev_btn.setEnabled(i > 0)
        self.next_btn.setEnabled(i < len(seq) - 1)
        if self.page == "finish":
            self.summary.setPlainText(self._summary())

    def go(self, step: int) -> None:
        seq = self.sequence()
        i = max(0, min(len(seq) - 1, seq.index(self.page) + step))
        self.page = seq[i]
        an = self.analysis
        if self.page == "anchors" and an.anchors is None:
            self.find_anchors()
        if self.page in ("find", "integrate", "fit") and not an.peaks:
            self.find_peaks()
        if self.page == "integrate" and not an.integration:
            self.integrate()
        if self.page == "fit" and an.fit is None:
            self.fit()
        self._show_page()

    def _summary(self) -> str:
        an = self.analysis
        lines = [f"Goal: {pa.GOALS[self.goal]}",
                 f"Baseline: {bl.BASELINE_MODES.get(an.baseline_mode, an.baseline_mode)}"]
        if an.peaks:
            lines.append(f"Peaks: {len(an.peaks)} (" + ", ".join(num(p['x'], 5)
                                                                 for p in an.peaks) + ")")
        if an.fit is not None:
            lines.append(f"Fit: {pa.SHAPES[an.fit.shape]['label']}, R² = "
                         f"{num(an.fit.result.r2)}")
        return "\n".join(lines)

    def finish(self) -> None:
        an = self.analysis
        an.goal = self.goal
        an.find_opts = self.find_opts()
        an.fit_opts = self.fit_opts()
        self.accept()

    # ------------------------------------------------------------------ settings
    def apply_settings(self, s: dict) -> None:
        self._busy = True
        try:
            self.set_goal(s.get("goal", "fit"))
            self.bmode.setCurrentIndex(max(0, self.bmode.findData(s.get("baseline_mode",
                                                                        "none"))))
            bo = s.get("baseline_opts") or {}
            if "lam" in bo:
                self.lam.setText(f"{bo['lam']:g}")
            for key, w in (("p", self.p_asym), ("half_window", self.half_window),
                           ("radius", self.radius), ("order", self.order),
                           ("n_end", self.n_end), ("n_anchors", self.n_anchors)):
                if key in bo:
                    w.setValue(bo[key])
            for key, w in (("anchor_method", self.anchor_method), ("interp", self.interp)):
                if key in bo:
                    w.setCurrentIndex(max(0, w.findData(bo[key])))
            fo = s.get("find_opts") or {}
            if "method" in fo:
                self.fmethod.setCurrentIndex(max(0, self.fmethod.findData(fo["method"])))
            if "height_pct" in fo:
                self.height_pct.setValue(fo["height_pct"])
            if fo.get("max_peaks"):
                self.max_peaks.setValue(int(fo["max_peaks"]))
            fi = s.get("fit_opts") or {}
            if "shape" in fi:
                self.shape.setCurrentIndex(max(0, self.shape.findData(fi["shape"])))
            if "baseline" in fi:
                self.cofit.setCurrentIndex(max(0, self.cofit.findData(fi["baseline"])))
            self.shared.setChecked(bool(fi.get("shared_width")))
        finally:
            self._busy = False
        if s.get("anchors"):
            self.analysis.anchors = (np.asarray(s["anchors"][0], float),
                                     np.asarray(s["anchors"][1], float))
        self._settings_changed()

    def values(self) -> dict:
        s = fr.analysis_settings(self.analysis)
        return {"input": self.inputs[self.input.currentIndex()] if self.inputs else None,
                "mode": self.recalc.currentData(), "settings": s}

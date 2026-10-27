"""Reusable SPSS *Bootstrap…* and *Exact…* sub-dialogs, plus the glue that runs a procedure
with bootstrapping, exact tests or automatic multiple-imputation pooling.

``ResampleMixin`` is mixed into a :class:`VariablePickerDialog`:

* ``add_bootstrap_button()`` / ``add_exact_button()`` add the SPSS buttons; the settings live
  in ``self.opts["bootstrap"]`` / ``self.opts["exact"]`` (remembered for the session),
* ``boot_spec()`` / ``exact_spec()`` return the core specifications (None when off),
* ``resampled(compute, boot=..., pool=...)`` runs ``compute()`` — on every imputation and
  pooled when the active dataset is multiply imputed — and then the bootstrap step, with a
  cancellable progress dialog for longer bootstraps.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup, QCheckBox,
                               QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
                               QGridLayout, QGroupBox, QHBoxLayout, QLabel, QListWidget,
                               QProgressDialog, QPushButton, QRadioButton, QSpinBox,
                               QVBoxLayout, QWidget)

from rsstatchem.core.output import Section
from rsstatchem.core.stats import mi_pool
from rsstatchem.core.stats import resample as rs
from rsstatchem.ui.dialogs.base import NAME_ROLE, variable_item

BOOT_DEFAULTS = {"perform": False, "samples": 1000, "use_seed": False, "seed": 2000000,
                 "level": 95.0, "ci": "percentile", "sampling": "simple", "strata": []}
EXACT_DEFAULTS = {"method": "asymptotic", "mc_level": 99.0, "mc_samples": 10000,
                  "use_limit": True, "time_limit": 5.0, "seed": 2000000}


def _buttons(dlg: QDialog, layout) -> None:
    box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                           | QDialogButtonBox.StandardButton.Cancel)
    box.button(QDialogButtonBox.StandardButton.Ok).setText("Continue")
    box.accepted.connect(dlg.accept)
    box.rejected.connect(dlg.reject)
    layout.addWidget(box)


class BootstrapDialog(QDialog):
    """SPSS Bootstrap sub-dialog: Perform bootstrapping, number of samples, seed, confidence
    intervals (percentile / BCa) and simple or stratified sampling."""

    def __init__(self, ds, current: dict | None = None, parent: QWidget | None = None,
                 title: str = "Bootstrap"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.ds = ds
        o = dict(BOOT_DEFAULTS)
        o.update(current or {})
        layout = QVBoxLayout(self)
        self.perform = QCheckBox("Perform bootstrapping")
        self.perform.setChecked(bool(o["perform"]))
        layout.addWidget(self.perform)

        self.body = QWidget()
        body = QVBoxLayout(self.body)
        body.setContentsMargins(18, 0, 0, 0)
        form = QFormLayout()
        self.samples = QSpinBox()
        self.samples.setRange(2, 1_000_000)
        self.samples.setValue(int(o["samples"]))
        form.addRow("Number of samples:", self.samples)
        body.addLayout(form)
        seed_row = QHBoxLayout()
        self.use_seed = QCheckBox("Set seed for Mersenne Twister")
        self.use_seed.setChecked(bool(o["use_seed"]))
        self.seed = QSpinBox()
        self.seed.setRange(1, 2_000_000_000)
        self.seed.setValue(int(o["seed"]))
        seed_row.addWidget(self.use_seed)
        seed_row.addWidget(QLabel("Seed:"))
        seed_row.addWidget(self.seed)
        seed_row.addStretch()
        body.addLayout(seed_row)

        ci = QGroupBox("Confidence Intervals")
        cl = QVBoxLayout(ci)
        lf = QFormLayout()
        self.level = QDoubleSpinBox()
        self.level.setRange(50, 99.9)
        self.level.setDecimals(1)
        self.level.setValue(float(o["level"]))
        lf.addRow("Level(%):", self.level)
        cl.addLayout(lf)
        g1 = QButtonGroup(self)
        self.pct = QRadioButton("Percentile")
        self.bca = QRadioButton("Bias corrected accelerated (BCa)")
        for rb in (self.pct, self.bca):
            g1.addButton(rb)
            cl.addWidget(rb)
        (self.bca if o["ci"] == "bca" else self.pct).setChecked(True)
        body.addWidget(ci)

        smp = QGroupBox("Sampling")
        sl = QVBoxLayout(smp)
        g2 = QButtonGroup(self)
        self.simple = QRadioButton("Simple")
        self.stratified = QRadioButton("Stratified")
        for rb in (self.simple, self.stratified):
            g2.addButton(rb)
            sl.addWidget(rb)
        (self.stratified if o["sampling"] == "stratified" else self.simple).setChecked(True)
        lists = QGridLayout()
        self.source = self._list()
        self.strata = self._list()
        chosen = [s for s in o["strata"] if ds is not None and ds.has_name(s)]
        if ds is not None:
            for var in ds.variables:
                (self.strata if var.name in chosen else self.source).addItem(variable_item(var))
        self.arrow = QPushButton("▶")
        self.arrow.setObjectName("arrow")
        self.arrow.setFixedWidth(34)
        self.arrow.clicked.connect(self._move)
        lists.addWidget(QLabel("Variables:"), 0, 0)
        lists.addWidget(QLabel("Strata Variables:"), 0, 2)
        lists.addWidget(self.source, 1, 0)
        lists.addWidget(self.arrow, 1, 1, Qt.AlignmentFlag.AlignVCenter)
        lists.addWidget(self.strata, 1, 2)
        sl.addLayout(lists)
        body.addWidget(smp)
        layout.addWidget(self.body)
        _buttons(self, layout)
        for w in (self.perform, self.use_seed, self.stratified, self.simple):
            w.toggled.connect(self._sync)
        self.source.itemSelectionChanged.connect(self._sync)
        self.strata.itemSelectionChanged.connect(self._sync)
        self._sync()

    @staticmethod
    def _list() -> QListWidget:
        w = QListWidget()
        w.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        w.setMinimumHeight(110)
        return w

    def _move(self) -> None:
        src, dst = (self.strata, self.source) if self.strata.selectedItems() and \
            not self.source.selectedItems() else (self.source, self.strata)
        for item in src.selectedItems():
            dst.addItem(src.takeItem(src.row(item)))
        self._sync()

    def _sync(self) -> None:
        on = self.perform.isChecked()
        self.body.setEnabled(on)
        self.seed.setEnabled(self.use_seed.isChecked())
        strat = self.stratified.isChecked()
        for w in (self.source, self.strata, self.arrow):
            w.setEnabled(strat)
        back = bool(self.strata.selectedItems()) and not self.source.selectedItems()
        self.arrow.setText("◀" if back else "▶")

    def values(self) -> dict:
        strata = [self.strata.item(i).data(NAME_ROLE) for i in range(self.strata.count())]
        return {"perform": self.perform.isChecked(), "samples": self.samples.value(),
                "use_seed": self.use_seed.isChecked(), "seed": self.seed.value(),
                "level": self.level.value(), "ci": "bca" if self.bca.isChecked()
                else "percentile",
                "sampling": "stratified" if self.stratified.isChecked() and strata
                else "simple", "strata": strata}


class ExactDialog(QDialog):
    """SPSS Exact Tests sub-dialog: Asymptotic only / Monte Carlo / Exact."""

    def __init__(self, current: dict | None = None, parent: QWidget | None = None,
                 title: str = "Exact Tests"):
        super().__init__(parent)
        self.setWindowTitle(title)
        o = dict(EXACT_DEFAULTS)
        o.update(current or {})
        layout = QVBoxLayout(self)
        group = QButtonGroup(self)
        self.asym = QRadioButton("Asymptotic only")
        self.mc = QRadioButton("Monte Carlo")
        self.exact = QRadioButton("Exact")
        for rb in (self.asym, self.mc, self.exact):
            group.addButton(rb)
        {"mc": self.mc, "exact": self.exact}.get(o["method"], self.asym).setChecked(True)
        layout.addWidget(self.asym)
        layout.addWidget(self.mc)
        mcf = QFormLayout()
        mcf.setContentsMargins(24, 0, 0, 0)
        self.mc_level = QDoubleSpinBox()
        self.mc_level.setRange(50, 99.9)
        self.mc_level.setDecimals(1)
        self.mc_level.setValue(float(o["mc_level"]))
        self.mc_level.setSuffix(" %")
        self.mc_samples = QSpinBox()
        self.mc_samples.setRange(100, 1_000_000_000)
        self.mc_samples.setValue(int(o["mc_samples"]))
        mcf.addRow("Confidence level:", self.mc_level)
        mcf.addRow("Number of samples:", self.mc_samples)
        layout.addLayout(mcf)
        layout.addWidget(self.exact)
        row = QHBoxLayout()
        row.setContentsMargins(24, 0, 0, 0)
        self.use_limit = QCheckBox("Time limit per test:")
        self.use_limit.setChecked(bool(o["use_limit"]))
        self.limit = QDoubleSpinBox()
        self.limit.setRange(0.1, 1440)
        self.limit.setDecimals(1)
        self.limit.setValue(float(o["time_limit"]))
        row.addWidget(self.use_limit)
        row.addWidget(self.limit)
        row.addWidget(QLabel("minutes"))
        row.addStretch()
        layout.addLayout(row)
        note = QLabel("Exact method will be used instead of Monte Carlo when computational "
                      "limits allow.\nFor nonasymptotic methods, cell counts are always "
                      "rounded or truncated in computing the test statistics.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#666")
        layout.addWidget(note)
        _buttons(self, layout)
        for rb in (self.asym, self.mc, self.exact, self.use_limit):
            rb.toggled.connect(self._sync)
        self._sync()

    def _sync(self) -> None:
        self.mc_level.setEnabled(self.mc.isChecked())
        self.mc_samples.setEnabled(self.mc.isChecked())
        self.use_limit.setEnabled(self.exact.isChecked())
        self.limit.setEnabled(self.exact.isChecked() and self.use_limit.isChecked())

    def values(self) -> dict:
        method = "mc" if self.mc.isChecked() else "exact" if self.exact.isChecked() \
            else "asymptotic"
        return {"method": method, "mc_level": self.mc_level.value(),
                "mc_samples": self.mc_samples.value(), "use_limit": self.use_limit.isChecked(),
                "time_limit": self.limit.value(), "seed": EXACT_DEFAULTS["seed"]}


@contextmanager
def boot_progress(parent: QWidget | None = None, label: str = "Bootstrapping…"):
    """Install a cancellable progress dialog as the bootstrap progress hook."""
    state = {"dlg": None, "start": time.monotonic()}

    def hook(done: int, total: int) -> None:
        app = QApplication.instance()
        if app is None:
            return
        dlg = state["dlg"]
        if dlg is None and time.monotonic() - state["start"] > 0.4 and done < total:
            dlg = QProgressDialog(label, "Cancel", 0, total, parent)
            dlg.setWindowTitle("Bootstrap")
            dlg.setWindowModality(Qt.WindowModality.WindowModal)
            dlg.setMinimumDuration(0)
            state["dlg"] = dlg
        if dlg is not None:
            dlg.setMaximum(total)
            dlg.setValue(done)
            app.processEvents()
            if dlg.wasCanceled():
                raise rs.Cancelled()

    old = rs.PROGRESS
    rs.PROGRESS = hook
    try:
        yield
    finally:
        rs.PROGRESS = old
        if state["dlg"] is not None:
            state["dlg"].close()


class ResampleMixin:
    """Bootstrap / Exact buttons and automatic MI pooling for a VariablePickerDialog."""

    BOOTSTRAP_TITLE = "Bootstrap"

    def add_bootstrap_button(self) -> QPushButton:
        return self.add_option_button("Bootstrap...", self.bootstrap_dialog)

    def add_exact_button(self) -> QPushButton:
        return self.add_option_button("Exact...", self.exact_dialog)

    def bootstrap_dialog(self) -> None:
        dlg = BootstrapDialog(self.ds, self.opts.get("bootstrap"), self)
        if dlg.exec():
            self.opts["bootstrap"] = dlg.values()

    def exact_dialog(self) -> None:
        dlg = ExactDialog(self.opts.get("exact"), self)
        if dlg.exec():
            self.opts["exact"] = dlg.values()

    def boot_spec(self) -> rs.BootSpec | None:
        return rs.BootSpec.from_options(self.opts.get("bootstrap"))

    def exact_spec(self) -> rs.ExactSpec | None:
        return rs.ExactSpec.from_options(self.opts.get("exact"))

    @staticmethod
    def plain(out):
        """The Section, unless the result carries saved variables."""
        return out if getattr(out, "saved", None) else mi_pool.section_of(out)

    # ------------------------------------------------------------------ running
    def run_views(self, views, compute) -> list[tuple[str, object]]:
        original = self.ds
        out = []
        try:
            for label, view in views:
                self.ds = view
                try:
                    out.append((label, compute()))
                except Exception as exc:  # noqa: BLE001 - the incomplete original data may fail
                    if label != mi_pool.ORIGINAL:
                        raise
                    sec = Section(self.windowTitle())
                    sec.note("Warning", f"The analysis of the original (incomplete) data "
                             f"failed: {exc}", style="warning")
                    out.append((label, sec))
        finally:
            self.ds = original
        return out

    def resampled(self, compute, boot=None, pool=None, title: str | None = None,
                  level: float = 0.95):
        """Run ``compute()`` with automatic pooling and the bootstrap step.

        ``boot(result, spec)`` adds the bootstrap output to the result. ``pool`` is
        ``"models"`` (pool the ModelInfo the result exposes), a callable ``pool(views)``
        returning pooled tables (or a Section), or None (no pooling: per-imputation output)."""
        views = mi_pool.auto_views(self.ds) if pool is not None else None
        if views is not None:
            per_view = self.run_views(views, compute)
            name = title or mi_pool.section_of(per_view[0][1]).title
            if pool == "models":
                section = mi_pool.pool_procedure(name, per_view, level)
            else:
                pooled = pool(views)
                tables = [i for i in pooled.items if getattr(i, "kind", "") == "table"] \
                    if hasattr(pooled, "items") else list(pooled)
                section = mi_pool.combine(name, tables, per_view)
            if self.boot_spec() is not None:
                section.note("Bootstrap", "Bootstrapping is not performed when the output is "
                             "pooled over multiple imputations.", style="warning")
            return section
        result = compute()
        spec = self.boot_spec()
        if spec is not None and boot is not None:
            try:
                with boot_progress(self if isinstance(self, QWidget) else None):
                    boot(result, spec)
            except rs.Cancelled:
                mi_pool.section_of(result).note("Bootstrap", "Bootstrapping was cancelled.",
                                                style="warning")
        return result


def selftest() -> None:
    rs.selftest()
    mi_pool.selftest()

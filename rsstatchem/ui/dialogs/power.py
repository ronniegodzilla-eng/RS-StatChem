"""Analyze > Power Analysis dialogs (SPSS 27+): Means, Proportions, Correlations and Regression.

The dialogs work without data, like SPSS's: choose *Estimate sample size* (single or grid of
power values) or *Estimate power* (one or more sample sizes), enter the test assumptions, the
test direction and the significance level; Plot..., Precision... (t tests) and Contrast...
(one-way ANOVA) open the matching sub-dialogs. Core code: :mod:`rsstatchem.core.stats.power`.
"""
from __future__ import annotations

import copy

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPushButton, QRadioButton, QVBoxLayout,
                               QWidget)

from rsstatchem.core.stats import power as pw
from rsstatchem.ui.dialogs.base import SubDialog


def parse_numbers(text: str) -> list[float]:
    """Numbers separated by spaces or semicolons; a lone comma inside a number is a decimal
    comma ("0,8 0,9"), otherwise commas separate values ("30,40,50" or "0.8,0.9")."""
    out: list[float] = []
    for tok in text.replace(";", " ").split():
        if "," in tok and ("." in tok or tok.count(",") > 1 or tok[0] == "," or
                           tok[-1] == ","):
            out += [float(p) for p in tok.split(",") if p]
        else:
            out.append(float(tok.replace(",", ".")))
    return out


class PowerDialog(QDialog):
    """Common layout of the SPSS Power Analysis dialogs."""

    needs_data = False
    TITLE = "Power Analysis"
    PROC = "one_t"
    PRECISION: str | None = None       # "one" | "two" -> Precision... button
    N_LABEL = "Sample size:"
    _memory: dict[str, dict] = {}

    def __init__(self, ds=None, parent=None):
        super().__init__(parent)
        self.ds = ds
        self.setWindowTitle(f"Power Analysis: {self.TITLE}")
        self.fields: dict[str, QWidget] = {}
        mem = self._memory.get(type(self).__name__, {})
        self.opts = copy.deepcopy(mem.get("__opts__", {"plots": {}, "precision": None,
                                                         "contrasts": []}))
        main = QVBoxLayout(self)
        body = QHBoxLayout()
        left = QVBoxLayout()
        body.addLayout(left, 1)
        right = QVBoxLayout()
        right.setAlignment(Qt.AlignmentFlag.AlignTop)
        body.addLayout(right)
        main.addLayout(body, 1)

        est = QGroupBox("Estimate")
        el = QVBoxLayout(est)
        self.est_n = QRadioButton("Sample size")
        self.est_power = QRadioButton("Power")
        g = QButtonGroup(self)
        g.addButton(self.est_n)
        g.addButton(self.est_power)
        row = QHBoxLayout()
        row.addWidget(self.est_n)
        row.addWidget(self.est_power)
        row.addStretch()
        el.addLayout(row)
        form = QFormLayout()
        self.single = QRadioButton("Single power value:")
        self.grid = QRadioButton("Grid power values:")
        g2 = QButtonGroup(self)
        g2.addButton(self.single)
        g2.addButton(self.grid)
        self.power_spin = self._spin(0.8, 0.01, 0.9999, 4)
        self.power_grid = QLineEdit("0.7 0.8 0.9")
        form.addRow(self.single, self.power_spin)
        form.addRow(self.grid, self.power_grid)
        self.n_edit = QLineEdit("30")
        self.n_edit.setToolTip("One or more sample sizes, separated by spaces.")
        form.addRow(self.N_LABEL, self.n_edit)
        el.addLayout(form)
        left.addWidget(est)

        self.assume_box = QGroupBox("Test Assumptions")
        self.assume = QFormLayout(self.assume_box)
        left.addWidget(self.assume_box)
        self.build_params(self.assume)

        bottom = QHBoxLayout()
        self.dir_box = QGroupBox("Test Direction")
        dl = QVBoxLayout(self.dir_box)
        self.two = QRadioButton("Nondirectional (two-sided) analysis")
        self.one = QRadioButton("Directional (one-sided) analysis")
        g3 = QButtonGroup(self)
        g3.addButton(self.two)
        g3.addButton(self.one)
        dl.addWidget(self.two)
        dl.addWidget(self.one)
        bottom.addWidget(self.dir_box, 1)
        sig = QFormLayout()
        self.alpha = self._spin(0.05, 0.0001, 0.5, 4)
        sig.addRow("Significance level:", self.alpha)
        bottom.addLayout(sig)
        left.addLayout(bottom)

        self.after_build()
        self.extra_buttons(right)
        b = QPushButton("Plot...")
        b.clicked.connect(self._plot)
        right.addWidget(b)
        if self.PRECISION:
            b = QPushButton("Precision...")
            b.clicked.connect(self._precision)
            right.addWidget(b)

        buttons = QDialogButtonBox()
        self.ok_button = buttons.addButton(QDialogButtonBox.StandardButton.Ok)
        reset = buttons.addButton(QDialogButtonBox.StandardButton.Reset)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        reset.clicked.connect(self.reset)
        main.addWidget(buttons)

        self.est_n.toggled.connect(self._sync)
        self.single.toggled.connect(self._sync)
        self._defaults = self._state()
        self._restore(mem)
        self._sync()

    # ------------------------------------------------------------------ widgets
    @staticmethod
    def _spin(value, lo, hi, dec) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(dec)
        s.setValue(value)
        s.setSingleStep(10 ** -min(dec, 2))
        return s

    def num(self, form, key, label, value, lo=-1e9, hi=1e9, dec=4) -> QDoubleSpinBox:
        s = self._spin(value, lo, hi, dec)
        self.fields[key] = s
        form.addRow(label, s)
        return s

    def text(self, form, key, label, value) -> QLineEdit:
        e = QLineEdit(value)
        self.fields[key] = e
        form.addRow(label, e)
        return e

    def choice(self, form, key, label, items: list[tuple[str, str]], current=None) -> QComboBox:
        c = QComboBox()
        for value, txt in items:
            c.addItem(txt, value)
        if current is not None:
            c.setCurrentIndex(max(0, c.findData(current)))
        self.fields[key] = c
        form.addRow(label, c)
        return c

    def check(self, form, key, label, checked=False) -> QCheckBox:
        c = QCheckBox(label)
        c.setChecked(checked)
        self.fields[key] = c
        form.addRow(c)
        return c

    def v(self, key):
        w = self.fields[key]
        if isinstance(w, QDoubleSpinBox):
            return w.value()
        if isinstance(w, QComboBox):
            return w.currentData()
        if isinstance(w, QCheckBox):
            return w.isChecked()
        return w.text()

    # ------------------------------------------------------------------ hooks
    def build_params(self, form: QFormLayout) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def extra_buttons(self, column: QVBoxLayout) -> None:
        pass

    def after_build(self) -> None:
        """Adjust the shared widgets (test direction etc.) once they exist."""

    def designs(self, alpha: float, sided: str) -> list[pw.Design]:  # pragma: no cover
        raise NotImplementedError

    def check_params(self) -> str | None:
        return None

    # ------------------------------------------------------------------ state
    def _state(self) -> dict:
        st = {k: self.v(k) for k in self.fields}
        st.update(__est__="n" if self.est_n.isChecked() or not self.est_power.isChecked()
                  else "power", __single__=not self.grid.isChecked(),
                  __power__=self.power_spin.value(), __grid__=self.power_grid.text(),
                  __n__=self.n_edit.text(), __sided__="one" if self.one.isChecked() else "two",
                  __alpha__=self.alpha.value())
        return st

    def _restore(self, st: dict) -> None:
        for k, w in self.fields.items():
            if k not in st:
                continue
            val = st[k]
            if isinstance(w, QDoubleSpinBox):
                w.setValue(val)
            elif isinstance(w, QComboBox):
                w.setCurrentIndex(max(0, w.findData(val)))
            elif isinstance(w, QCheckBox):
                w.setChecked(bool(val))
            else:
                w.setText(str(val))
        (self.est_power if st.get("__est__") == "power" else self.est_n).setChecked(True)
        (self.single if st.get("__single__", True) else self.grid).setChecked(True)
        self.power_spin.setValue(st.get("__power__", 0.8))
        self.power_grid.setText(st.get("__grid__", "0.7 0.8 0.9"))
        self.n_edit.setText(st.get("__n__", "30"))
        (self.one if st.get("__sided__") == "one" else self.two).setChecked(True)
        self.alpha.setValue(st.get("__alpha__", 0.05))

    def _sync(self, *_):
        n_mode = self.est_n.isChecked()
        for w in (self.single, self.grid):
            w.setEnabled(n_mode)
        self.power_spin.setEnabled(n_mode and self.single.isChecked())
        self.power_grid.setEnabled(n_mode and self.grid.isChecked())
        self.n_edit.setEnabled(not n_mode)

    def reset(self) -> None:
        self._restore(self._defaults)
        self.opts = {"plots": {}, "precision": None, "contrasts": []}
        self._sync()

    # ------------------------------------------------------------------ sub-dialogs
    def _plot(self) -> None:
        p = self.opts.get("plots", {})
        dlg = SubDialog(f"Power Analysis: {self.TITLE}: Plot", self, columns=1)
        box = dlg.group("Two-Dimensional Plot")
        dlg.check(box, "by_n", "Power estimation versus sample size", p.get("by_n", False))
        dlg.number(box, "n_lo", "Lower bound of sample size (0 = automatic):",
                   p.get("n_lo", 0), 0, 1e6, 0)
        dlg.number(box, "n_hi", "Upper bound of sample size (0 = automatic):",
                   p.get("n_hi", 0), 0, 1e6, 0)
        dlg.check(box, "by_effect", "Power estimation versus effect size",
                  p.get("by_effect", False))
        dlg.number(box, "e_lo", "Lower bound of effect size:", p.get("e_lo", 0), -100, 100, 3)
        dlg.number(box, "e_hi", "Upper bound of effect size (0 = automatic):",
                   p.get("e_hi", 0), -100, 100, 3)
        values = dlg.run()
        if values:
            self.opts["plots"] = values

    def _precision(self) -> None:
        cur = self.opts.get("precision") or {}
        dlg = SubDialog(f"Power Analysis: {self.TITLE}: Precision", self, columns=1)
        box = dlg.group("Estimate sample size based on confidence interval")
        dlg.check(box, "on", "Estimate sample size based on confidence interval half-width",
                  cur.get("on", False))
        dlg.number(box, "level", "Confidence level (%):", cur.get("level", 95), 50, 99.99, 2)
        edit = QLineEdit(cur.get("widths", "0.25"))
        row = QFormLayout()
        row.addRow("Half-width value(s):", edit)
        box.addLayout(row)
        values = dlg.run()
        if values:
            values["widths"] = edit.text()
            self.opts["precision"] = values

    # ------------------------------------------------------------------ protocol
    def accept(self) -> None:
        try:
            err = self.check_params()
            if not err:
                if self.est_n.isChecked():
                    powers = [self.power_spin.value()] if self.single.isChecked() else \
                        parse_numbers(self.power_grid.text())
                    if not powers or any(not 0 < p < 1 for p in powers):
                        err = "Power values must lie between 0 and 1."
                else:
                    ns = parse_numbers(self.n_edit.text())
                    if not ns or any(n < 2 for n in ns):
                        err = "Enter one or more sample sizes of at least 2."
            if not err:
                self.designs(self.alpha.value(), "two")
        except (ValueError, ZeroDivisionError) as exc:
            err = str(exc) or "Check the test assumptions."
        if err:
            QMessageBox.warning(self, self.windowTitle(), err)
            return
        st = self._state()
        st["__opts__"] = copy.deepcopy(self.opts)
        self._memory[type(self).__name__] = st
        super().accept()

    def _plot_args(self) -> dict:
        p = self.opts.get("plots") or {}
        out = {"by_n": p.get("by_n", False), "by_effect": p.get("by_effect", False)}
        if p.get("n_hi"):
            out["n_range"] = (int(p.get("n_lo", 0)) or 2, int(p["n_hi"]))
        if p.get("e_hi"):
            out["effect_range"] = (p.get("e_lo", 0.0), p["e_hi"])
        return out

    def _precision_args(self, designs) -> dict | None:
        return None

    def compute(self):
        sided = "one" if self.one.isChecked() else "two"
        designs = self.designs(self.alpha.value(), sided)
        if self.est_n.isChecked():
            powers = [self.power_spin.value()] if self.single.isChecked() else \
                parse_numbers(self.power_grid.text())
            ns = ()
            est = "n"
        else:
            powers = ()
            ns = [int(round(n)) for n in parse_numbers(self.n_edit.text())]
            est = "power"
        syntax = self.syntax(designs[0], est, powers, ns, sided)
        return pw.power_analysis(self.PROC, designs, est, powers, ns, self._plot_args(),
                                 self._precision_args(designs), syntax)

    def syntax(self, d, est, powers, ns, sided) -> str:
        cmd = {"one_t": "POWER MEANS ONESAMPLE", "paired_t": "POWER MEANS RELATED",
               "ind_t": "POWER MEANS INDEPENDENT", "anova": "POWER ONEWAY ANOVA",
               "one_binom": "POWER PROPORTIONS ONESAMPLE",
               "rel_binom": "POWER PROPORTIONS RELATED",
               "ind_binom": "POWER PROPORTIONS INDEPENDENT", "pearson": "POWER PEARSON ONESAMPLE",
               "spearman": "POWER SPEARMAN ONESAMPLE", "partial": "POWER PARTIALCORR",
               "regression": "POWER UNIVARIATE LINEAR"}[self.PROC]
        what = ("POWER=" + " ".join(f"{p:g}" for p in powers)) if est == "n" else \
            ("N=" + " ".join(str(n) for n in ns))
        return (f"{cmd}\n  /PARAMETERS TEST={'NONDIRECTIONAL' if sided == 'two' else 'DIRECTIONAL'}"
                f" SIGNIFICANCE={d.alpha:g} {what}")


# =========================================================================== Means
class OneSampleTPowerDialog(PowerDialog):
    TITLE = "One-Sample T Test"
    PROC = "one_t"
    PRECISION = "one"

    def build_params(self, form):
        self.num(form, "mean", "Population mean:", 0.5)
        self.num(form, "null", "Null value:", 0.0)
        self.num(form, "sd", "Population standard deviation:", 1.0, 1e-9, 1e9)

    def check_params(self):
        if self.v("mean") == self.v("null"):
            return "The population mean must differ from the null value."
        return None

    def designs(self, alpha, sided):
        return [pw.OneSampleT(alpha, sided, mean=self.v("mean"), null=self.v("null"),
                              sd=self.v("sd"))]

    def _precision_args(self, designs):
        p = self.opts.get("precision") or {}
        if not p.get("on"):
            return None
        return {"kind": "one", "sd": designs[0].sd, "half_widths": parse_numbers(p["widths"]),
                "level": p.get("level", 95) / 100}


class PairedTPowerDialog(OneSampleTPowerDialog):
    TITLE = "Paired-Samples T Test"
    PROC = "paired_t"

    def build_params(self, form):
        self.num(form, "mean", "Population mean difference:", 0.5)
        self.choice(form, "sdmode", "Standard deviation:",
                    [("diff", "Standard deviation of the difference"),
                     ("each", "Standard deviations of each group and correlation")])
        self.num(form, "sd", "Std. deviation of the difference:", 1.0, 1e-9, 1e9)
        self.num(form, "sd1", "Std. deviation of group 1:", 1.0, 1e-9, 1e9)
        self.num(form, "sd2", "Std. deviation of group 2:", 1.0, 1e-9, 1e9)
        self.num(form, "rho", "Correlation between the pairs:", 0.5, -0.9999, 0.9999)

    def _sd(self):
        if self.v("sdmode") == "each":
            return pw.paired_sd(self.v("sd1"), self.v("sd2"), self.v("rho"))
        return self.v("sd")

    def check_params(self):
        if self.v("mean") == 0:
            return "The population mean difference must not be 0."
        if self._sd() <= 0:
            return "The standard deviation of the difference must be positive."
        return None

    def designs(self, alpha, sided):
        return [pw.PairedT(alpha, sided, mean=self.v("mean"), null=0.0, sd=self._sd())]


class IndependentTPowerDialog(PowerDialog):
    TITLE = "Independent-Samples T Test"
    PROC = "ind_t"
    PRECISION = "two"
    N_LABEL = "Group 1 sample size(s):"

    def build_params(self, form):
        self.num(form, "ratio", "Group size ratio (N2/N1):", 1.0, 0.01, 100)
        self.num(form, "mean1", "Population mean for group 1:", 0.5)
        self.num(form, "mean2", "Population mean for group 2:", 0.0)
        self.choice(form, "var", "Population standard deviations:",
                    [("pooled", "Are equal for two groups (pooled)"),
                     ("separate", "Are not equal for two groups")])
        self.num(form, "sd1", "Pooled / group 1 standard deviation:", 1.0, 1e-9, 1e9)
        self.num(form, "sd2", "Group 2 standard deviation:", 1.0, 1e-9, 1e9)

    def check_params(self):
        if self.v("mean1") == self.v("mean2"):
            return "The group means must differ."
        return None

    def designs(self, alpha, sided):
        return [pw.IndependentT(alpha, sided, mean1=self.v("mean1"), mean2=self.v("mean2"),
                                sd1=self.v("sd1"),
                                sd2=self.v("sd2") if self.v("var") == "separate" else None,
                                ratio=self.v("ratio"))]

    def _precision_args(self, designs):
        p = self.opts.get("precision") or {}
        if not p.get("on"):
            return None
        d = designs[0]
        return {"kind": "two", "sd": d.sd1, "half_widths": parse_numbers(p["widths"]),
                "level": p.get("level", 95) / 100, "ratio": d.ratio, "sd2": d.sd2}


class AnovaPowerDialog(PowerDialog):
    TITLE = "One-Way ANOVA"
    PROC = "anova"
    N_LABEL = "Multiplier of the group-size weights:"

    def build_params(self, form):
        self.text(form, "means", "Group means:", "0 0.5 1")
        self.text(form, "weights", "Group size weights (blank = equal):", "")
        self.num(form, "sd", "Pooled standard deviation:", 1.0, 1e-9, 1e9)

    def after_build(self):
        self.dir_box.setTitle("Test Direction (contrast tests)")

    def extra_buttons(self, column):
        b = QPushButton("Contrast...")
        b.clicked.connect(self._contrast)
        column.addWidget(b)

    def _contrast(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Power Analysis: One-Way ANOVA: Contrast")
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel("Contrast coefficients (one contrast per line, coefficients "
                             "separated by spaces):"))
        from PySide6.QtWidgets import QPlainTextEdit

        edit = QPlainTextEdit("\n".join(" ".join(f"{c:g}" for c in row)
                                        for row in self.opts.get("contrasts", [])))
        lay.addWidget(edit)
        pair = QCheckBox("Pairwise difference tests")
        pair.setChecked(self.opts.get("pairwise", False))
        lay.addWidget(pair)
        form = QFormLayout()
        adj = QComboBox()
        for value, txt in (("none", "Least Significant Difference (LSD)"),
                           ("bonferroni", "Bonferroni"), ("sidak", "Sidak")):
            adj.addItem(txt, value)
        adj.setCurrentIndex(max(0, adj.findData(self.opts.get("adjust", "none"))))
        form.addRow("Adjustment for pairwise tests:", adj)
        lay.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("Continue")
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        if dlg.exec():
            try:
                rows = [parse_numbers(line) for line in edit.toPlainText().splitlines()
                        if line.strip()]
            except ValueError:
                QMessageBox.warning(self, dlg.windowTitle(), "Contrast coefficients must be "
                                    "numbers.")
                return
            self.opts.update(contrasts=rows, pairwise=pair.isChecked(), adjust=adj.currentData())

    def _means(self):
        m = parse_numbers(self.v("means"))
        w = parse_numbers(self.v("weights")) if self.v("weights").strip() else None
        return m, w

    def check_params(self):
        m, w = self._means()
        if len(m) < 2:
            return "Enter at least two group means."
        if w is not None and (len(w) != len(m) or any(x <= 0 for x in w)):
            return "Give one positive weight per group (or leave the weights blank)."
        if len(set(m)) == 1:
            return "The group means must not all be equal."
        for row in self.opts.get("contrasts", []):
            if len(row) != len(m):
                return "Each contrast needs one coefficient per group."
            if abs(sum(row)) > 1e-8:
                return "Contrast coefficients must sum to 0."
        return None

    def designs(self, alpha, sided):
        m, w = self._means()
        sd = self.v("sd")
        out = [pw.OneWayAnova(alpha, "two", means=tuple(m), sd=sd,
                              weights=tuple(w) if w else None)]
        for i, row in enumerate(self.opts.get("contrasts", [])):
            out.append(pw.AnovaContrast(alpha, sided, means=tuple(m), sd=sd,
                                        weights=tuple(w) if w else None,
                                        coefficients=tuple(row),
                                        title=f"Contrast {i + 1}"))
        if self.opts.get("pairwise"):
            k = len(m)
            npairs = k * (k - 1) // 2
            for i in range(k):
                for j in range(i + 1, k):
                    c = [0.0] * k
                    c[i], c[j] = 1.0, -1.0
                    out.append(pw.AnovaContrast(alpha, sided, means=tuple(m), sd=sd,
                                                weights=tuple(w) if w else None,
                                                coefficients=tuple(c), adjust_k=npairs,
                                                adjust=self.opts.get("adjust", "none"),
                                                title=f"Group {i + 1} vs Group {j + 1}"))
        return out


# =========================================================================== Proportions
_METHOD = [("exact", "Binomial enumeration (exact)"), ("normal", "Normal approximation")]


class OneBinomialPowerDialog(PowerDialog):
    TITLE = "One-Sample Binomial Test"
    PROC = "one_binom"

    def build_params(self, form):
        self.num(form, "p", "Population proportion:", 0.65, 0.0001, 0.9999)
        self.num(form, "p0", "Null value:", 0.5, 0.0001, 0.9999)
        self.choice(form, "method", "Estimation method:", _METHOD)
        self.check(form, "cc", "Apply continuity correction (normal approximation)")

    def check_params(self):
        return "The population proportion must differ from the null value." \
            if self.v("p") == self.v("p0") else None

    def designs(self, alpha, sided):
        return [pw.OneSampleBinomial(alpha, sided, p=self.v("p"), p0=self.v("p0"),
                                     method=self.v("method"), continuity=self.v("cc"))]


class RelatedBinomialPowerDialog(PowerDialog):
    TITLE = "Related-Samples Binomial Test"
    PROC = "rel_binom"
    N_LABEL = "Number of pairs:"

    def build_params(self, form):
        form.addRow(QLabel("Proportions of discordant pairs:"))
        self.num(form, "p12", "Success on first, failure on second:", 0.25, 0.0001, 0.9999)
        self.num(form, "p21", "Failure on first, success on second:", 0.10, 0.0001, 0.9999)
        self.choice(form, "method", "Estimation method:", _METHOD)
        self.check(form, "cc", "Apply continuity correction (normal approximation)")

    def check_params(self):
        if self.v("p12") == self.v("p21"):
            return "The discordant proportions must differ."
        if self.v("p12") + self.v("p21") > 1:
            return "The discordant proportions cannot sum to more than 1."
        return None

    def designs(self, alpha, sided):
        return [pw.RelatedBinomial(alpha, sided, p12=self.v("p12"), p21=self.v("p21"),
                                   method=self.v("method"), continuity=self.v("cc"))]


class IndependentBinomialPowerDialog(PowerDialog):
    TITLE = "Independent-Samples Binomial Test"
    PROC = "ind_binom"
    N_LABEL = "Group 1 sample size(s):"

    def build_params(self, form):
        self.num(form, "p1", "Proportion for group 1:", 0.65, 0.0001, 0.9999)
        self.num(form, "p2", "Proportion for group 2:", 0.85, 0.0001, 0.9999)
        self.num(form, "ratio", "Group size ratio (N2/N1):", 1.0, 0.01, 100)
        self.choice(form, "test", "Test method:", [("chisq", "Chi-squared test (pooled)"),
                                                   ("unpooled", "z test (unpooled variance)"),
                                                   ("fisher", "Fisher's exact test")])
        self.choice(form, "method", "Estimation method:", [("normal", "Normal approximation"),
                                                           ("exact", "Binomial enumeration")])
        self.check(form, "cc", "Apply continuity correction")

    def check_params(self):
        return "The two proportions must differ." if self.v("p1") == self.v("p2") else None

    def designs(self, alpha, sided):
        return [pw.IndependentBinomial(alpha, sided, p1=self.v("p1"), p2=self.v("p2"),
                                       ratio=self.v("ratio"), test=self.v("test"),
                                       method=self.v("method"), continuity=self.v("cc"))]


# =========================================================================== Correlations
class PearsonPowerDialog(PowerDialog):
    TITLE = "Pearson Correlation"
    PROC = "pearson"
    KIND = "pearson"

    def build_params(self, form):
        self.num(form, "rho", "Pearson correlation parameter:", 0.3, -0.9999, 0.9999)
        self.num(form, "rho0", "Null value:", 0.0, -0.9999, 0.9999)
        if self.KIND == "partial":
            self.num(form, "k", "Number of variables partialled out:", 1, 0, 100, 0)
        if self.KIND == "spearman":
            self.choice(form, "svar", "Variance estimate:",
                        [("bonett", "Bonett and Wright"),
                         ("fieller", "Fieller, Hartley and Pearson"),
                         ("caruso", "Caruso and Cliff")])
        self.check(form, "bias", "Use bias-correction formula in the power estimation")

    def check_params(self):
        return "The correlation must differ from the null value." \
            if self.v("rho") == self.v("rho0") else None

    def designs(self, alpha, sided):
        return [pw.Correlation(alpha, sided, rho=self.v("rho"), rho0=self.v("rho0"),
                               kind=self.KIND,
                               partial_k=int(self.v("k")) if self.KIND == "partial" else 0,
                               spearman_var=self.v("svar") if self.KIND == "spearman"
                               else "bonett", bias=self.v("bias"))]


class SpearmanPowerDialog(PearsonPowerDialog):
    TITLE = "Spearman Correlation"
    PROC = "spearman"
    KIND = "spearman"


class PartialPowerDialog(PearsonPowerDialog):
    TITLE = "Partial Correlation"
    PROC = "partial"
    KIND = "partial"


# =========================================================================== Regression
class RegressionPowerDialog(PowerDialog):
    TITLE = "Univariate Linear Regression"
    PROC = "regression"

    def build_params(self, form):
        self.num(form, "p", "Total number of predictors:", 5, 1, 500, 0)
        self.num(form, "q", "Number of test predictors:", 5, 1, 500, 0)
        self.choice(form, "spec", "Specify:", [("partial", "Partial R² (effect of the test "
                                                           "predictors)"),
                                               ("full", "R² for the full and nested models")])
        self.num(form, "pr2", "Partial R²:", 0.1304, 0.0001, 0.9999)
        self.num(form, "r2f", "R² for the full model:", 0.2, 0.0001, 0.9999)
        self.num(form, "r2r", "R² for the nested model:", 0.0, 0.0, 0.9999)

    def after_build(self):
        self.dir_box.setEnabled(False)

    def check_params(self):
        if self.v("q") > self.v("p"):
            return "The test predictors cannot outnumber the total predictors."
        if self.v("spec") == "full" and self.v("r2f") <= self.v("r2r"):
            return "R² of the full model must exceed that of the nested model."
        return None

    def designs(self, alpha, sided):
        if self.v("spec") == "partial":
            r2f, r2r = self.v("pr2"), 0.0
        else:
            r2f, r2r = self.v("r2f"), self.v("r2r")
        return [pw.LinearRegression(alpha, "two", r2_full=r2f, r2_reduced=r2r,
                                    predictors=int(self.v("p")), tested=int(self.v("q")))]


MENUS = {
    "Analyze/Power Analysis": [
        ("Means", [("One-Sample T Test...", OneSampleTPowerDialog),
                   ("Paired-Samples T Test...", PairedTPowerDialog),
                   ("Independent-Samples T Test...", IndependentTPowerDialog),
                   ("One-Way ANOVA...", AnovaPowerDialog)]),
        ("Proportions", [("One-Sample Binomial Test...", OneBinomialPowerDialog),
                         ("Related-Samples Binomial Test...", RelatedBinomialPowerDialog),
                         ("Independent-Samples Binomial Test...",
                          IndependentBinomialPowerDialog)]),
        ("Correlations", [("Pearson Product-Moment...", PearsonPowerDialog),
                          ("Spearman Rank-Order...", SpearmanPowerDialog),
                          ("Partial Correlation...", PartialPowerDialog)]),
        ("Regression", [("Univariate Linear...", RegressionPowerDialog)]),
    ],
}


def selftest() -> None:
    pw.selftest()

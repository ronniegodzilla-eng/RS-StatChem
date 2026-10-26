"""Origin-style dialogs of the Statistics menu (X-Function dialog layout).

Every dialog has Origin's parts: the **Recalculate** mode at the top (Manual / Auto / None),
an **Input** group with range pickers — a range box takes Origin's notation
(``[Book1]Sheet1!B`` or ``[Book1]Sheet1!B[3:10]``) and its ▸ button fills it from the
current worksheet selection or from a list of all columns — the analysis options, and the
output (a report sheet in the input's workbook). The dialog only collects ``settings``; the
main window runs :mod:`rsstatchem.ui.origin.stat_tests`. ``settings`` can be passed back in
(Change Parameters) to reopen the dialog as it was.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QDoubleSpinBox, QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMenu, QPushButton, QSpinBox, QToolButton,
                               QVBoxLayout, QWidget)

from rsstatchem.ui.origin import stat_tests as st
from rsstatchem.ui.origin.model import RECALC_MODES


# ====================================================================== range pickers
def _fill_range_menu(menu: QMenu, main, pick) -> None:
    """Menu of the ▸ button: the active selection, then every column by book / sheet."""
    menu.clear()
    sel = main.selection_ranges() if main is not None else []
    act = menu.addAction("Select Columns in Active Sheet", lambda: pick(sel))
    act.setEnabled(bool(sel))
    menu.addSeparator()
    for title, cols in (main.range_sources() if main is not None else []):
        sub = menu.addMenu(title)
        for text, label in cols:
            sub.addAction(label, lambda t=text: pick([t]))


class RangeEdit(QWidget):
    """One input range: an editable box and Origin's ▸ (select) button."""

    def __init__(self, main, text: str = ""):
        super().__init__()
        self.main = main
        self.edit = QLineEdit(text)
        self.edit.setPlaceholderText("[Book1]Sheet1!B")
        self.button = QToolButton()
        self.button.setText("▸")
        self.button.setToolTip("Select the range from the worksheet")
        self.button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.button)
        menu.aboutToShow.connect(lambda: _fill_range_menu(menu, main, self._pick))
        self.button.setMenu(menu)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(self.edit, 1)
        lay.addWidget(self.button)

    def _pick(self, ranges: list[str]) -> None:
        if ranges:
            self.edit.setText(ranges[0])

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, text: str) -> None:  # noqa: N802
        self.edit.setText(text or "")


class RangeList(QWidget):
    """Several input ranges (one per data set / group) with Add, Remove and ▸."""

    def __init__(self, main, texts: list[str] | None = None):
        super().__init__()
        self.main = main
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list.setMaximumHeight(110)
        self.list.setMinimumHeight(70)
        self.add = QToolButton()
        self.add.setText("▸")
        self.add.setToolTip("Add ranges from the worksheet")
        self.add.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.add)
        menu.aboutToShow.connect(lambda: _fill_range_menu(menu, main, self._pick))
        self.add.setMenu(menu)
        remove = QToolButton()
        remove.setText("−")
        remove.setToolTip("Remove the selected ranges")
        remove.clicked.connect(self._remove)
        new = QToolButton()
        new.setText("+")
        new.setToolTip("Add an empty range to type")
        new.clicked.connect(lambda: self._append("[Book1]Sheet1!A", edit=True))
        side = QVBoxLayout()
        side.setSpacing(2)
        for b in (self.add, new, remove):
            side.addWidget(b)
        side.addStretch(1)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(self.list, 1)
        lay.addLayout(side)
        self.set_texts(texts or [])

    def _append(self, text: str, edit: bool = False) -> None:
        item = QListWidgetItem(text)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        self.list.addItem(item)
        if edit:
            self.list.editItem(item)

    def _pick(self, ranges: list[str]) -> None:
        current = set(self.texts())
        for r in ranges:
            if r not in current:
                self._append(r)

    def _remove(self) -> None:
        for item in self.list.selectedItems():
            self.list.takeItem(self.list.row(item))

    def texts(self) -> list[str]:
        return [self.list.item(i).text().strip() for i in range(self.list.count())
                if self.list.item(i).text().strip()]

    def set_texts(self, texts: list[str]) -> None:
        self.list.clear()
        for t in texts:
            self._append(t)


# ====================================================================== dialog
class StatDialog(QDialog):
    """A Statistics dialog built from a field spec.

    ``groups``: ``[(group title, [field, ...]), ...]`` with fields
    ``(kind, key, label, default, extra)``; kinds: ``ranges``, ``range``, ``choice``
    (extra = {value: text}), ``checks`` (extra = {value: text}, default = selected values),
    ``float`` (extra = (lo, hi, decimals)), ``int`` (extra = (lo, hi)), ``bool``. A field
    may end with ``{"when": (key, value)}`` to show it only for that choice."""

    def __init__(self, title: str, groups: list, main=None, settings: dict | None = None,
                 description: str = "", recalc: bool = True, output: str = "report sheet"):
        super().__init__(main)
        self.setWindowTitle(title)
        self.setObjectName("statDialog")
        self.main = main
        self.widgets: dict[str, QWidget] = {}
        self.kinds: dict[str, str] = {}
        self.rows: list[tuple[QWidget, QWidget | None, tuple | None]] = []
        settings = dict(settings or {})
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        self.mode = QComboBox()
        for key, text in RECALC_MODES.items():
            self.mode.addItem(text, key)
        self.mode.setCurrentIndex(max(0, self.mode.findData(settings.get("mode", "manual"))))
        if recalc:
            top.addWidget(QLabel("Recalculate"))
            top.addWidget(self.mode)
        top.addStretch(1)
        if description:
            desc = QLabel(description)
            desc.setStyleSheet("color:#444")
            desc.setWordWrap(True)
            layout.addWidget(desc)
        layout.addLayout(top)
        for gtitle, fields in groups:
            box = QGroupBox(gtitle)
            form = QFormLayout(box)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            for field in fields:
                self._add_field(form, field, settings)
            layout.addWidget(box)
        out = QLabel(f"Output:  [&lt;input&gt;]&lt;new&gt; — {output} in the workbook of "
                     "the (first) input range; summary in the Results Log.")
        out.setStyleSheet("color:#444")
        out.setWordWrap(True)
        layout.addWidget(out)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                               | QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        layout.addWidget(box)
        for w in self.widgets.values():
            if isinstance(w, QComboBox):
                w.currentIndexChanged.connect(self._update_visibility)
        self._update_visibility()
        self.resize(500, 0)

    def _add_field(self, form: QFormLayout, field: tuple, settings: dict) -> None:
        cond = field[-1].get("when") if isinstance(field[-1], dict) else None
        if cond is not None:
            field = field[:-1]
        kind, key, label, default, *extra = field
        extra = extra[0] if extra else None
        value = settings.get(key, default)
        if kind == "ranges":
            w = RangeList(self.main, list(value or []))
        elif kind == "range":
            w = RangeEdit(self.main, value or "")
        elif kind == "choice":
            w = QComboBox()
            for v, text in extra.items():
                w.addItem(text, v)
            w.setCurrentIndex(max(0, w.findData(value)))
        elif kind == "checks":
            w = QWidget()
            grid = QGridLayout(w)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(14)
            w.boxes = {}
            per_col = 2 if len(extra) > 6 else 1
            for k, (v, text) in enumerate(extra.items()):
                cb = QCheckBox(text)
                cb.setChecked(v in (value or ()))
                grid.addWidget(cb, k // per_col, k % per_col)
                w.boxes[v] = cb
        elif kind == "float":
            w = QDoubleSpinBox()
            lo, hi, dec = extra or (-1e12, 1e12, 4)
            w.setRange(lo, hi)
            w.setDecimals(dec)
            w.setValue(float(value))
        elif kind == "int":
            w = QSpinBox()
            lo, hi = extra or (0, 100000)
            w.setRange(lo, hi)
            w.setValue(int(value))
        elif kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(value))
        else:
            w = QLineEdit(str(value))
        self.widgets[key] = w
        self.kinds[key] = kind
        lab = QLabel(label)
        form.addRow(lab, w)
        self.rows.append((w, lab, cond))

    def _update_visibility(self) -> None:
        for w, lab, cond in self.rows:
            if cond is None:
                continue
            key, val = cond
            src = self.widgets.get(key)
            show = src is None or (src.currentData() if isinstance(src, QComboBox) else
                                   None) == val
            w.setVisible(show)
            if lab is not None:
                lab.setVisible(show)

    def widget(self, key: str):
        return self.widgets[key]

    def values(self) -> dict:
        out = {"mode": self.mode.currentData()}
        for key, w in self.widgets.items():
            kind = self.kinds[key]
            if kind == "ranges":
                out[key] = w.texts()
            elif kind == "range":
                out[key] = w.text()
            elif kind == "choice":
                out[key] = w.currentData()
            elif kind == "checks":
                out[key] = [v for v, cb in w.boxes.items() if cb.isChecked()]
            elif kind in ("float", "int"):
                out[key] = w.value()
            elif kind == "bool":
                out[key] = w.isChecked()
            else:
                out[key] = w.text()
        return out


# ====================================================================== the dialogs
FORMS = {"raw": "Raw", "indexed": "Indexed"}
TAILS = {"two": "Mean <> value (two-tailed)", "upper": "Mean > value (upper-tailed)",
         "lower": "Mean < value (lower-tailed)"}
TAILS2 = {"two": "mean1 − mean2 <> value", "upper": "mean1 − mean2 > value",
          "lower": "mean1 − mean2 < value"}


def _alpha(default: float = 0.05):
    return ("float", "alpha", "Significance Level", default, (0.0001, 0.5, 4))


def spec(key: str, sel: list[str]) -> tuple[str, list, str, str]:
    """(title, groups, description, output) of the dialog ``key``, with the current selection
    ``sel`` (range strings) as the default input."""
    first = sel[0] if sel else ""
    second = sel[1] if len(sel) > 1 else ""
    if key == "desc":
        return ("Statistics on Columns", [
            ("Input", [("ranges", "ranges", "Input Data", sel)]),
            ("Quantities to Compute", [("checks", "quantities", "", list(st.DESC_DEFAULT),
                                        st.DESC_QUANTITIES)]),
            ("Computation Control", [
                ("choice", "moment", "Moment Computation Method", "n-1",
                 {"n-1": "Divisor is N − 1 (sample)", "n": "Divisor is N (population)"}),
                ("float", "level", "Confidence Level for Mean", 0.95, (0.5, 0.9999, 4))]),
            ("Plots", [("bool", "histogram", "Histograms", False),
                       ("bool", "box", "Box Charts", False)])],
            "Descriptive statistics of each input column (Origin's DescStats).", "report sheet")
    if key == "rows":
        return ("Statistics on Rows", [
            ("Input", [("ranges", "ranges", "Input Data (two or more columns)", sel)]),
            ("Quantities to Compute", [("checks", "quantities", "", ["mean", "sd", "n"],
                                        st.ROW_QUANTITIES)])],
            "One value per row across the input columns, added as new columns.",
            "new columns of the source sheet")
    if key == "freq":
        return ("Frequency Counts", [
            ("Input", [("range", "range", "Input", first)]),
            ("Options", [("bool", "auto", "Automatic Binning (Sturges)", True),
                         ("float", "from", "From Minimum", 0.0, (-1e12, 1e12, 6)),
                         ("float", "to", "To Maximum", 10.0, (-1e12, 1e12, 6)),
                         ("float", "step", "Step Size", 1.0, (0.0, 1e12, 6)),
                         ("bool", "discrete", "Discrete Frequency (count each value)",
                          False)])],
            "Bin counts, cumulative counts and relative frequencies (text data are counted "
            "per value).", "worksheet")
    if key == "normality":
        return ("Normality Test", [
            ("Input", [("ranges", "ranges", "Input Data", sel)]),
            ("Tests", [("checks", "tests", "", ["shapiro"], st.NORMALITY_TESTS)]),
            ("Options", [_alpha(), ("bool", "qq", "Q-Q Plots", True)])],
            "Tests whether each data set is drawn from a normal distribution.", "report sheet")
    if key == "correlation":
        return ("Correlation Coefficient", [
            ("Input", [("ranges", "ranges", "Input Data (two or more columns)", sel)]),
            ("Correlation Types", [("checks", "methods", "", ["pearson"],
                                    st.CORR_METHODS)]),
            ("Options", [_alpha(), ("bool", "flag", "Flag Significant Correlations", True),
                         ("bool", "scatter", "Scatter Plot (two columns)", False)])],
            "Pairwise correlation coefficients, p-values and N (missing values excluded "
            "pairwise).", "report sheet")
    if key == "ttest1":
        return ("One-Sample t-Test", [
            ("Input", [("ranges", "ranges", "Input Data", sel[:1])]),
            ("Hypotheses", [("float", "mean", "Test Mean", 0.0, (-1e12, 1e12, 6)),
                            ("choice", "tail", "Alternate Hypothesis", "two", TAILS)]),
            ("Options", [_alpha()])],
            "Is the population mean equal to the test mean?", "report sheet")
    if key == "ttest2":
        return ("Two-Sample t-Test", [
            ("Input", [("choice", "form", "Input Data Form", "raw", FORMS),
                       ("ranges", "ranges", "Data (two columns)", sel[:2],
                        {"when": ("form", "raw")}),
                       ("range", "data", "Data Range", first, {"when": ("form", "indexed")}),
                       ("range", "factor", "Group Range", second,
                        {"when": ("form", "indexed")})]),
            ("Hypotheses", [("float", "diff", "Mean Difference", 0.0, (-1e12, 1e12, 6)),
                            ("choice", "tail", "Alternate Hypothesis", "two", TAILS2)]),
            ("Options", [_alpha(), ("bool", "equal_var_test", "Equality of Variance Test",
                                    True)])],
            "Compares the means of two independent samples (both equal and unequal variance "
            "results).", "report sheet")
    if key == "ttest_pair":
        return ("Pair-Sample t-Test", [
            ("Input", [("range", "range1", "1st Data Range", first),
                       ("range", "range2", "2nd Data Range", second)]),
            ("Hypotheses", [("float", "diff", "Mean Difference", 0.0, (-1e12, 1e12, 6)),
                            ("choice", "tail", "Alternate Hypothesis", "two", TAILS2)]),
            ("Options", [_alpha()])],
            "Compares the means of two paired samples.", "report sheet")
    if key == "anova1":
        return ("One-Way ANOVA", [
            ("Input", [("choice", "form", "Input Data Form", "raw", FORMS),
                       ("ranges", "ranges", "Data (one column per level)", sel,
                        {"when": ("form", "raw")}),
                       ("range", "data", "Data Range", first, {"when": ("form", "indexed")}),
                       ("range", "factor", "Factor Range", second,
                        {"when": ("form", "indexed")})]),
            ("Means Comparison", [("checks", "means", "", ["tukey"], st.MEANS_METHODS)]),
            ("Options", [_alpha(),
                         ("choice", "homogeneity", "Test for Equal Variance", "levene",
                          {"levene": "Levene (absolute deviations)",
                           "brown_forsythe": "Brown-Forsythe", "none": "None"}),
                         ("bool", "means_plot", "Means Plot", True)])],
            "Tests whether the level means are equal, with means comparisons.",
            "report sheet")
    if key == "anova2":
        third = sel[2] if len(sel) > 2 else ""
        return ("Two-Way ANOVA", [
            ("Input (indexed data)", [("range", "data", "Data Range", third or first),
                                      ("range", "factor_a", "Factor A Range",
                                       first if third else second),
                                      ("range", "factor_b", "Factor B Range",
                                       second if third else "")]),
            ("Means Comparison", [("checks", "means", "", ["tukey"], st.MEANS_METHODS)]),
            ("Options", [_alpha(), ("bool", "interaction", "Interactions", True),
                         ("bool", "means_plot", "Interaction Plot", True)])],
            "Main effects of two factors and their interaction (Type III sums of squares).",
            "report sheet")
    if key == "nonparam":
        return ("Nonparametric Tests", [
            ("Test", [("choice", "test", "Test", "mann_whitney", st.NONPARAMETRIC)]),
            ("Input", [("choice", "form", "Input Data Form", "raw", FORMS),
                       ("ranges", "ranges", "Data (one column per group / treatment)", sel,
                        {"when": ("form", "raw")}),
                       ("range", "data", "Data Range", first, {"when": ("form", "indexed")}),
                       ("range", "factor", "Group Range", second,
                        {"when": ("form", "indexed")})]),
            ("Options", [_alpha(),
                         ("choice", "tail", "Alternate Hypothesis (two-group tests)", "two",
                          {"two": "Two-tailed", "upper": "Upper-tailed",
                           "lower": "Lower-tailed"})])],
            "Mann-Whitney and Kruskal-Wallis accept raw or indexed data; Wilcoxon and "
            "Friedman use one column per paired sample / treatment.", "report sheet")
    raise KeyError(key)


def make_dialog(key: str, main=None, settings: dict | None = None,
                selection: list[str] | None = None) -> StatDialog:
    title, groups, description, output = spec(key, selection or [])
    return StatDialog(title, groups, main, settings, description,
                      recalc=key not in ("rows", "freq"), output=output)

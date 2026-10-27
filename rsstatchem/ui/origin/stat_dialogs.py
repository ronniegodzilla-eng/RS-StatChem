"""Origin-style dialogs of the Statistics menu (X-Function dialog layout).

Every dialog has Origin's parts: the **Recalculate** mode at the top (Manual / Auto / None),
an **Input** group with range pickers — a range box takes Origin's notation
(``[Book1]Sheet1!B`` or ``[Book1]Sheet1!B[3:10]``) and its ▸ button rolls the dialog up so
the range can be drag-selected in a worksheet (its arrow fills it from the current selection
or from a list of all columns) — the analysis options, and the
output (a report sheet in the input's workbook). The dialog only collects ``settings``; the
main window runs :mod:`rsstatchem.ui.origin.stat_tests`. ``settings`` can be passed back in
(Change Parameters) to reopen the dialog as it was.
"""
from __future__ import annotations

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QDoubleSpinBox, QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMenu, QPushButton, QSpinBox, QToolButton,
                               QVBoxLayout, QWidget)

from rsstatchem.ui.origin import stat_tests as st
from rsstatchem.ui.origin.model import RECALC_MODES


# ====================================================================== range pickers
def _fill_range_menu(menu: QMenu, main, pick, interactive=None) -> None:
    """Menu of the ▸ button: interactive selection, the active selection, then every column
    by book / sheet."""
    menu.clear()
    if interactive is not None:
        act = menu.addAction("Select Range from Worksheet (drag-select)...", interactive)
        act.setEnabled(main is not None)
    sel = main.selection_ranges(fallback=False) if main is not None else []
    act = menu.addAction("Select Columns in Active Sheet", lambda: pick(sel))
    act.setEnabled(bool(sel))
    menu.addSeparator()
    for title, cols in (main.range_sources() if main is not None else []):
        sub = menu.addMenu(title)
        for text, label in cols:
            sub.addAction(label, lambda t=text: pick([t]))


def _owner_dialog(w: QWidget):
    """The Statistics dialog holding a range picker (None outside one)."""
    while w is not None and not isinstance(w, StatDialog):
        w = w.parentWidget()
    return w


def _picker_button(owner: QWidget, main, pick, tip: str) -> QToolButton:
    """Origin's ▸ button: a click rolls the dialog up to select the range interactively in
    the worksheet; the arrow opens the menu (selection, list of columns)."""
    button = QToolButton()
    button.setText("▸")
    button.setToolTip(tip)
    button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
    menu = QMenu(button)
    menu.aboutToShow.connect(lambda: _fill_range_menu(menu, main, pick, owner.start_pick))
    button.setMenu(menu)
    button.clicked.connect(owner.start_pick)
    return button


class RangeEdit(QWidget):
    """One input range: an editable box and Origin's ▸ (select) button."""

    def __init__(self, main, text: str = ""):
        super().__init__()
        self.main = main
        self.edit = QLineEdit(text)
        self.edit.setPlaceholderText("[Book1]Sheet1!B")
        self.button = _picker_button(self, main, self._pick,
                                     "Select the range from the worksheet (click, then drag "
                                     "over the cells); the arrow lists the columns")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(self.edit, 1)
        lay.addWidget(self.button)

    def start_pick(self) -> None:
        dlg = _owner_dialog(self)
        if dlg is not None and self.main is not None:
            dlg.start_pick(self)

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
        self.add = _picker_button(self, main, self._pick,
                                  "Add ranges from the worksheet (click, then drag over the "
                                  "columns / cells); the arrow lists the columns")
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

    def start_pick(self) -> None:
        dlg = _owner_dialog(self)
        if dlg is not None and self.main is not None:
            dlg.start_pick(self)

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


class LevelCombo(QComboBox):
    """Dropdown of the levels of a factor (Origin's Dunnett control): "(first level)" and
    the levels read from the dialog's current input; the list is refreshed when the input
    changes and whenever it is opened. A control stored as text (older reports) is selected
    when it is one of the levels, and kept as typed while the levels cannot be read."""

    FIRST = "(first level)"

    def __init__(self, value: str = "", fallback: str = ""):
        super().__init__()
        self.refresher = None
        self._fallback = str(fallback or "").strip()
        self.addItem(self.FIRST, "")
        value = str(value or "").strip()
        if value:
            self.addItem(value, value)
            self.setCurrentIndex(1)

    def showPopup(self) -> None:  # noqa: N802 - Qt override
        if self.refresher is not None:
            self.refresher()
        super().showPopup()

    def levels(self) -> list[str]:
        return [self.itemData(i) for i in range(1, self.count())]

    def set_levels(self, levels: list[str]) -> None:
        levels = [str(v) for v in levels]
        current = self.currentData() or ""
        if levels == self.levels():
            return
        if not levels:   # input not readable: keep the stored choice
            keep = [current] if current else []
        else:
            keep = levels
            if current not in levels:
                current = self._fallback if self._fallback in levels else ""
        self.blockSignals(True)
        self.clear()
        self.addItem(self.FIRST, "")
        for v in keep:
            self.addItem(v, v)
        self.setCurrentIndex(max(0, self.findData(current)))
        self.blockSignals(False)


class _PickBar(QWidget):
    """The rolled-up dialog while a range is selected in the worksheet (Origin's dialog
    shrinks to its range box with a ▸ button to come back)."""

    def __init__(self, dialog: "StatDialog"):
        super().__init__()
        self.setObjectName("rangePickBar")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(3)
        self.hint = QLabel("Select the range in the worksheet — drag over the cells or "
                           "click column headers — then click ▸ (or press Enter).")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color:#444")
        row = QHBoxLayout()
        row.setSpacing(2)
        self.edit = QLineEdit()
        self.edit.setReadOnly(True)
        self.edit.setObjectName("rangePickText")
        self.edit.setPlaceholderText("(nothing selected)")
        self.back = QToolButton()
        self.back.setText("▸")
        self.back.setToolTip("Use this range and restore the dialog")
        self.back.clicked.connect(lambda: dialog.finish_pick(True))
        cancel = QToolButton()
        cancel.setText("✕")
        cancel.setToolTip("Restore the dialog without changing the range")
        cancel.clicked.connect(lambda: dialog.finish_pick(False))
        row.addWidget(self.edit, 1)
        row.addWidget(self.back)
        row.addWidget(cancel)
        lay.addWidget(self.hint)
        lay.addLayout(row)


# ====================================================================== dialog
class StatDialog(QDialog):
    """A Statistics dialog built from a field spec.

    ``groups``: ``[(group title, [field, ...]), ...]`` with fields
    ``(kind, key, label, default, extra)``; kinds: ``ranges``, ``range``, ``choice``
    (extra = {value: text}), ``checks`` (extra = {value: text}, default = selected values),
    ``float`` (extra = (lo, hi, decimals)), ``int`` (extra = (lo, hi)), ``bool``, ``text``,
    ``level`` (a :class:`LevelCombo` of factor levels; extra = {"source": key of
    :func:`stat_tests.factor_levels`, "fallback": settings key of a control stored as
    text}). The Dunnett options are enabled only while Dunnett is checked.
    A field may end with ``{"when": (key, value)}`` to show it only for that choice (a tuple
    of values shows it for any of them).

    A range picker's ▸ button rolls the dialog up (:meth:`start_pick`): the dialog shrinks
    to a bar and stops being modal so the user can drag-select in a worksheet; ▸ (or Enter)
    takes the selection and restores the dialog (:meth:`finish_pick`), as in Origin."""

    def __init__(self, title: str, groups: list, main=None, settings: dict | None = None,
                 description: str = "", recalc: bool = True, output: str = "report sheet"):
        super().__init__(main)
        self.setWindowTitle(title)
        self.setObjectName("statDialog")
        self.main = main
        self.widgets: dict[str, QWidget] = {}
        self.kinds: dict[str, str] = {}
        self.rows: list[tuple[QWidget, QWidget | None, tuple | None]] = []
        self.level_sources: dict[str, str] = {}
        self.picking = None          # the range picker being filled from the worksheet
        self._loop = None
        self._geometry = None
        settings = dict(settings or {})
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        self.body = QWidget()
        self.pick_bar = _PickBar(self)
        self.pick_bar.hide()
        outer.addWidget(self.pick_bar)
        outer.addWidget(self.body)
        self._pick_timer = QTimer(self)
        self._pick_timer.setInterval(150)
        self._pick_timer.timeout.connect(self._show_picked)
        layout = QVBoxLayout(self.body)
        layout.setContentsMargins(0, 0, 0, 0)
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
            if isinstance(w, QComboBox) and not isinstance(w, LevelCombo):
                w.currentIndexChanged.connect(self._update_visibility)
        # a control typed in an older report, kept for the factor that has that level
        self.passthrough = {k: settings[k] for k in ("control",)
                            if k in settings and k not in self.widgets}
        self._level_timer = QTimer(self)
        self._level_timer.setSingleShot(True)
        self._level_timer.setInterval(200)
        self._level_timer.timeout.connect(self.refresh_levels)
        if self.level_sources:
            for w in self.widgets.values():
                self._watch_input(w)
            for key in self.level_sources:
                self.widgets[key].refresher = self.refresh_levels
            self.refresh_levels()
        means = self.widgets.get("means")
        if means is not None and "dunnett" in getattr(means, "boxes", {}):
            means.boxes["dunnett"].toggled.connect(self._update_dunnett)
        self._update_dunnett()
        self._update_visibility()
        self.resize(500, 0)

    # ------------------------------------------------------------------ factor levels
    DUNNETT_KEYS = ("control", "control_a", "control_b", "dunnett_tail")

    def _watch_input(self, w: QWidget) -> None:
        """Refresh the level dropdowns (debounced) when an input that defines the levels
        changes."""
        start = self._level_timer.start
        if isinstance(w, LevelCombo):
            return
        if isinstance(w, RangeEdit):
            w.edit.textChanged.connect(start)
        elif isinstance(w, RangeList):
            model = w.list.model()
            model.rowsInserted.connect(start)
            model.rowsRemoved.connect(start)
            model.dataChanged.connect(start)
        elif isinstance(w, QComboBox):
            w.currentIndexChanged.connect(start)
        elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
            w.valueChanged.connect(start)
        elif isinstance(w, QLineEdit):
            w.textChanged.connect(start)

    def _books(self) -> dict:
        try:
            return self.main.project.data_windows()
        except AttributeError:
            return {}

    def refresh_levels(self) -> None:
        """Fill the level dropdowns from the current input ranges."""
        if not self.level_sources:
            return
        values, books = self.values(), self._books()
        for key, source in self.level_sources.items():
            self.widgets[key].set_levels(st.factor_levels(books, values, source))

    def _update_dunnett(self) -> None:
        means = self.widgets.get("means")
        box = getattr(means, "boxes", {}).get("dunnett") if means is not None else None
        on = box is None or box.isChecked()
        for w, lab, _cond in self.rows:
            key = next((k for k, v in self.widgets.items() if v is w), None)
            if key in self.DUNNETT_KEYS:
                w.setEnabled(on)
                if lab is not None:
                    lab.setEnabled(on)

    # ------------------------------------------------------------------ modality / pick
    def exec(self) -> int:  # noqa: A003 - same API as QDialog.exec
        """Modal like QDialog.exec, but in a local event loop so the dialog can turn
        modeless while it is rolled up for a worksheet selection."""
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.show()
        self.raise_()
        loop = QEventLoop()
        self._loop = loop
        self.finished.connect(loop.quit)
        loop.exec()
        self._loop = None
        return self.result()

    def start_pick(self, picker) -> None:
        """Roll the dialog up and let the user select ``picker``'s range in a worksheet."""
        if self.picking is not None or self.main is None:
            return
        self.picking = picker
        self._geometry = self.geometry()
        was_visible = self.isVisible()
        self.hide()
        self.body.hide()
        self.pick_bar.show()
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setMinimumSize(0, 0)
        self._show_picked()
        self._pick_timer.start()
        if was_visible:
            self.show()
        self.adjustSize()
        self.resize(max(self.width(), 420), self.pick_bar.sizeHint().height() + 12)
        top = self.main.geometry().topLeft() if hasattr(self.main, "geometry") else None
        if top is not None and was_visible:
            self.move(top.x() + 80, top.y() + 60)

    def picked_ranges(self) -> list[str]:
        return self.main.selection_ranges(fallback=False) if self.main is not None else []

    def _show_picked(self) -> None:
        sel = self.picked_ranges()
        text = ", ".join(sel)
        if self.pick_bar.edit.text() != text:
            self.pick_bar.edit.setText(text)

    def finish_pick(self, apply: bool = True) -> None:
        """Take the worksheet selection (when ``apply``) and restore the dialog."""
        picker = self.picking
        if picker is None:
            return
        self._pick_timer.stop()
        ranges = self.picked_ranges()
        if apply and ranges:
            picker._pick(ranges)
        self.picking = None
        was_visible = self.isVisible()
        self.hide()
        self.pick_bar.hide()
        self.body.show()
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        if was_visible:
            self.show()
            self.raise_()
            self.activateWindow()
        if self._geometry is not None:
            self.setGeometry(self._geometry)

    def accept(self) -> None:
        if self.picking is not None:   # Enter while rolled up: take the selection
            self.finish_pick(True)
            return
        super().accept()

    def reject(self) -> None:
        if self.picking is not None:   # Esc while rolled up: just come back
            self.finish_pick(False)
            return
        super().reject()

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
        elif kind == "level":
            extra = extra or {}
            w = LevelCombo(value, settings.get(extra.get("fallback", ""), ""))
            self.level_sources[key] = extra.get("source", "")
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
            current = src.currentData() if isinstance(src, QComboBox) else None
            show = src is None or (current in val if isinstance(val, (tuple, list))
                                   else current == val)
            w.setVisible(show)
            if lab is not None:
                lab.setVisible(show)

    def widget(self, key: str):
        return self.widgets[key]

    def values(self) -> dict:
        out = {"mode": self.mode.currentData()}
        out.update(getattr(self, "passthrough", {}))
        for key, w in self.widgets.items():
            kind = self.kinds[key]
            if kind == "ranges":
                out[key] = w.texts()
            elif kind == "range":
                out[key] = w.text()
            elif kind == "choice":
                out[key] = w.currentData()
            elif kind == "level":
                out[key] = w.currentData() or ""
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


DUNNETT_TAILS = {"two": "Mean <> control (two-tailed)",
                 "lower": "Mean < control (lower-tailed)",
                 "upper": "Mean > control (upper-tailed)"}


def _dunnett(sources: list[tuple[str, str, str]]) -> list[tuple]:
    """Origin's Dunnett options: the control level of each factor (a dropdown of its
    levels) and the alternate hypothesis (two-sided, < control, > control)."""
    fields = [("level", key, label, "", {"source": source, "fallback": "control"})
              for key, label, source in sources]
    return fields + [("choice", "dunnett_tail", "Dunnett Alternate Hypothesis", "two",
                      DUNNETT_TAILS)]


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
            ("Means Comparison", [("checks", "means", "", ["tukey"], st.MEANS_METHODS)] +
             _dunnett([("control", "Dunnett Control Level", "anova1")])),
            ("Options", [_alpha(),
                         ("choice", "homogeneity", "Test for Equal Variance", "levene",
                          {"levene": "Levene (absolute deviations)",
                           "brown_forsythe": "Brown-Forsythe", "none": "None"}),
                         ("bool", "means_plot", "Means Plot", True)])],
            "Tests whether the level means are equal, with means comparisons.",
            "report sheet")
    if key == "anova2":
        third = sel[2] if len(sel) > 2 else ""
        raw = {"when": ("form", "raw")}
        indexed = {"when": ("form", "indexed")}
        return ("Two-Way ANOVA", [
            ("Input", [("choice", "form", "Input Data Form", "indexed", FORMS),
                       ("ranges", "ranges", "Data (one column per level combination: "
                        "A1B1, A1B2, …, A2B1, …)", sel, raw),
                       ("text", "name_a", "Factor A Name", "Factor A", raw),
                       ("int", "levels_a", "Factor A Number of Levels", 2, (2, 25), raw),
                       ("text", "names_a", "Factor A Level Names", "A1, A2", raw),
                       ("text", "name_b", "Factor B Name", "Factor B", raw),
                       ("int", "levels_b", "Factor B Number of Levels", 2, (2, 25), raw),
                       ("text", "names_b", "Factor B Level Names", "B1, B2", raw),
                       ("range", "data", "Data Range", third or first, indexed),
                       ("range", "factor_a", "Factor A Range", first if third else second,
                        indexed),
                       ("range", "factor_b", "Factor B Range", second if third else "",
                        indexed)]),
            ("Means Comparison", [("checks", "means", "", ["tukey"], st.MEANS_METHODS)] +
             _dunnett([("control_a", "Dunnett Control Level (Factor A)", "anova2_a"),
                       ("control_b", "Dunnett Control Level (Factor B)", "anova2_b")])),
            ("Options", [_alpha(), ("bool", "interaction", "Interactions", True),
                         ("choice", "ss", "Type of Sums of Squares", "3", st.SS_TYPES),
                         ("bool", "means_plot", "Interaction Plot", True)])],
            "Main effects of two factors and their interaction; raw (one column per level "
            "combination, Factor A outermost) or indexed data, balanced or unbalanced.",
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
    if key not in st.DIALOG_SPECS:
        st.load_extensions()
    if key in st.DIALOG_SPECS:
        return st.DIALOG_SPECS[key](sel)
    raise KeyError(key)


def make_dialog(key: str, main=None, settings: dict | None = None,
                selection: list[str] | None = None) -> StatDialog:
    title, groups, description, output = spec(key, selection or [])
    return StatDialog(title, groups, main, settings, description, output=output)

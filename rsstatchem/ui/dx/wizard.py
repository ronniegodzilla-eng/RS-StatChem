"""New Design wizard, laid out like Design-Expert's.

Page 1: the design families on the left (Standard Designs: Factorial, Response Surface,
Mixture; Custom Designs: Optimal), the designs of the chosen family in the middle and, on the
right, the number of factors, the factor table (name, units, type, minimum/maximum or levels,
coded low/high and the axial points of a CCD) and the design options (alpha, centre points,
blocks, replicates, fraction, ...), with the run count updated live.
Page 2: the responses (number, names, units). Finish builds the design with
:func:`rsstatchem.core.doe.designs.build_design` and emits ``created(DesignDocument)``.

The wizard is never run with ``exec()``; it is shown with ``open()`` (window-modal sheet).
"""
from __future__ import annotations

import math
import random

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog,
                               QDoubleSpinBox, QFrame, QGridLayout, QGroupBox, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QPushButton, QSpinBox, QStackedWidget, QTableWidget,
                               QTableWidgetItem, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
                               QWidget)

from rsstatchem.core.doe import designs
from rsstatchem.ui.doe.wizard import DEFAULT_K, DESCRIPTIONS, LIMITS
from rsstatchem.ui.dx import icons, style
from rsstatchem.ui.dx.document import DesignDocument

# family key -> (tree text, designs)
FAMILIES = {
    "factorial": ("Factorial", [("factorial", "Regular Two-Level"),
                                ("general", "Multilevel Categoric"),
                                ("pb", "Plackett-Burman"), ("taguchi", "Taguchi OA")]),
    "rsm": ("Response Surface", [("ccd", "Central Composite"), ("bbd", "Box-Behnken"),
                                 ("threelevel", "3-Level Factorial"),
                                 ("dopt", "Optimal (Custom)")]),
    "mixture": ("Mixture", [("lattice", "Simplex Lattice"), ("centroid", "Simplex Centroid"),
                            ("ev", "Screening (Extreme Vertices)"),
                            ("mixdopt", "Optimal (Custom)")]),
    "custom": ("Optimal (Custom)", [("dopt", "Optimal (Custom) Process"),
                                    ("mixdopt", "Optimal (Custom) Mixture")]),
}
TREE = [("Standard Designs", ["factorial", "rsm", "mixture"]), ("Custom Designs", ["custom"])]
HELP = {
    "ccd": "Central composite design: a two-level factorial (or fraction), axial (star) points "
           "at ±alpha and centre points. The standard design for a quadratic model.",
    "bbd": "Box-Behnken: every factor at three levels, edge midpoints and centre points, no "
           "extreme corners. 3 to 7 factors.",
    "dopt": "Optimal (custom): runs chosen by D-optimal point exchange for the model you plan "
            "to fit, plus lack-of-fit and replicate points.",
    "mixdopt": "Optimal (custom) mixture: D-optimal blends inside the constrained mixture "
               "region, plus lack-of-fit and replicate points.",
}
ALPHAS = [("rotatable", "Rotatable"), ("spherical", "Spherical"),
          ("orthogonal", "Orthogonal Quadratic"), ("practical", "Practical"),
          ("face", "Face Centered"), ("custom", "Other (enter value)")]
CCD_TYPES = [("circumscribed", "Circumscribed (CCC)"), ("inscribed", "Inscribed (CCI)"),
             ("face", "Face centered (CCF)")]
MODELS = [("linear", "Linear"), ("2fi", "2FI"), ("quadratic", "Quadratic"), ("cubic", "Cubic")]
MIX_MODELS = [("linear", "Linear"), ("quadratic", "Quadratic"),
              ("special_cubic", "Special Cubic"), ("cubic", "Cubic")]
CATEGORIC = ("general", "taguchi")


def _num(text, default: float) -> float:
    try:
        return float(str(text).replace(",", ".").strip())
    except (TypeError, ValueError):
        return default


def _fmt(x: float) -> str:
    return f"{x:.6g}"


class NewDesignWizard(QDialog):
    created = Signal(object)  # DesignDocument

    def __init__(self, parent=None, family: str = "rsm", design: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("New Design")
        self.setStyleSheet(style.STYLESHEET)
        self.resize(1080, 680)
        self.document: DesignDocument | None = None
        self._loading = False
        self._rows: list[dict] = []      # factor entries kept across design changes
        self._opt: dict = {}
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._update_summary)

        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_design_page())
        self.pages.addWidget(self._build_response_page())

        self.back_btn = QPushButton("< Back")
        self.next_btn = QPushButton("Next >")
        self.finish_btn = QPushButton("Finish")
        self.cancel_btn = QPushButton("Cancel")
        self.back_btn.clicked.connect(lambda: self.go(0))
        self.next_btn.clicked.connect(lambda: self.go(1))
        self.finish_btn.clicked.connect(self.finish)
        self.cancel_btn.clicked.connect(self.reject)
        self.error = QLabel()
        self.error.setObjectName("error")
        self.error.setWordWrap(True)
        buttons = QHBoxLayout()
        buttons.addWidget(self.error, 1)
        for b in (self.back_btn, self.next_btn, self.finish_btn, self.cancel_btn):
            buttons.addWidget(b)
        lay = QVBoxLayout(self)
        lay.addWidget(self.pages, 1)
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color:#C9D3DF;")
        lay.addWidget(line)
        lay.addLayout(buttons)
        self.select(family, design)
        self.go(0)

    # ------------------------------------------------------------------ page 1
    def _build_design_page(self) -> QWidget:
        page = QWidget()
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setObjectName("navTree")
        self.tree.setFixedWidth(190)
        self._family_items: dict[str, QTreeWidgetItem] = {}
        for head, fams in TREE:
            top = QTreeWidgetItem([head])
            top.setFlags(Qt.ItemFlag.ItemIsEnabled)
            f = top.font(0)
            f.setBold(True)
            top.setFont(0, f)
            top.setIcon(0, icons.icon("folder"))
            self.tree.addTopLevelItem(top)
            for key in fams:
                it = QTreeWidgetItem([FAMILIES[key][0]])
                it.setData(0, Qt.ItemDataRole.UserRole, key)
                it.setIcon(0, icons.icon({"factorial": "cube", "rsm": "surface",
                                          "mixture": "ternary", "custom": "evaluation"}[key]))
                top.addChild(it)
                self._family_items[key] = it
            top.setExpanded(True)
        self.tree.currentItemChanged.connect(lambda *_: self._family_changed())

        self.list = QListWidget()
        self.list.setFixedWidth(210)
        self.list.currentItemChanged.connect(lambda *_: self._design_changed())

        self.title = QLabel()
        self.title.setObjectName("pageTitle")
        self.desc = QLabel()
        self.desc.setWordWrap(True)
        self.desc.setStyleSheet(f"color:{style.MUTED};")

        self.k_label = QLabel("Numeric factors:")
        self.k_spin = QSpinBox()
        self.k_spin.valueChanged.connect(lambda _: self._resize_factors())
        self.ccd_alpha_entry = QCheckBox("Enter factor ranges in terms of alphas")
        self.ccd_alpha_entry.toggled.connect(lambda _: self._refresh_coded())
        top = QHBoxLayout()
        top.addWidget(self.k_label)
        top.addWidget(self.k_spin)
        top.addSpacing(18)
        top.addWidget(self.ccd_alpha_entry)
        top.addStretch(1)

        self.table = QTableWidget(0, 0)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.itemChanged.connect(self._item_changed)

        self.options_box = QGroupBox("Options")
        self.options_form = QGridLayout(self.options_box)
        self.options_form.setColumnStretch(1, 1)
        self.options_form.setColumnStretch(3, 1)
        self.options_form.setHorizontalSpacing(8)
        self._n_opt = 0
        self._opt_k = None

        self.randomize = QCheckBox("Randomize run order")
        self.randomize.setChecked(True)
        self.seed = random.randint(1, 99999)
        self.summary = QLabel()
        self.summary.setObjectName("hint")
        self.summary.setWordWrap(True)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(10, 4, 4, 4)
        rl.addWidget(self.title)
        rl.addWidget(self.desc)
        rl.addLayout(top)
        rl.addWidget(self.table, 1)
        rl.addWidget(self.options_box)
        rl.addWidget(self.randomize)
        rl.addWidget(self.summary)

        lay = QHBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.tree)
        lay.addWidget(self.list)
        lay.addWidget(right, 1)
        return page

    # ------------------------------------------------------------------ page 2
    def _build_response_page(self) -> QWidget:
        page = QWidget()
        t = QLabel("Responses")
        t.setObjectName("pageTitle")
        self.r_spin = QSpinBox()
        self.r_spin.setRange(1, 30)
        self.r_spin.setValue(1)
        self.r_spin.valueChanged.connect(lambda _: self._resize_responses())
        self.responses = QTableWidget(0, 3)
        self.responses.setHorizontalHeaderLabels(["Response", "Name", "Units"])
        self.responses.verticalHeader().setVisible(False)
        self.responses.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.responses.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.name_edit = QLineEdit("Design")
        row = QHBoxLayout()
        row.addWidget(QLabel("Responses:"))
        row.addWidget(self.r_spin)
        row.addSpacing(24)
        row.addWidget(QLabel("Design name:"))
        row.addWidget(self.name_edit)
        row.addStretch(1)
        self.r_summary = QLabel()
        self.r_summary.setObjectName("hint")
        self.r_summary.setWordWrap(True)
        lay = QVBoxLayout(page)
        lay.addWidget(t)
        lay.addWidget(QLabel("Name each response (the measured result) and give its units. "
                             "You enter the measured values in the design layout after "
                             "Finish."))
        lay.addLayout(row)
        lay.addWidget(self.responses, 1)
        lay.addWidget(self.r_summary)
        self._resize_responses()
        return page

    def _resize_responses(self) -> None:
        n = self.r_spin.value()
        old = self.responses.rowCount()
        self.responses.setRowCount(n)
        for i in range(old, n):
            lab = QTableWidgetItem(f"R{i + 1}")
            lab.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.responses.setItem(i, 0, lab)
            self.responses.setItem(i, 1, QTableWidgetItem(f"R{i + 1}"))
            self.responses.setItem(i, 2, QTableWidgetItem(""))

    def set_responses(self, names: list[tuple[str, str]]) -> None:
        self.r_spin.setValue(len(names))
        for i, (n, u) in enumerate(names):
            self.responses.item(i, 1).setText(n)
            self.responses.item(i, 2).setText(u)

    # ------------------------------------------------------------------ navigation
    def go(self, page: int) -> None:
        if page == 1:
            spec = self.spec()
            count = designs.design_run_count(spec)
            if isinstance(count, str):
                self.error.setText(count)
                return
            self.r_summary.setText(f"{designs.DESIGN_TITLES[self.design]} with "
                                   f"{len(spec['factors'])} factors: {count} runs.")
        self.error.clear()
        self.pages.setCurrentIndex(page)
        self.back_btn.setEnabled(page == 1)
        self.next_btn.setEnabled(page == 0)
        self.finish_btn.setEnabled(page == 1)
        (self.finish_btn if page == 1 else self.next_btn).setDefault(True)

    def select(self, family: str, design: str | None = None) -> None:
        item = self._family_items.get(family) or self._family_items["rsm"]
        self.tree.setCurrentItem(item)
        if design:
            for i in range(self.list.count()):
                if self.list.item(i).data(Qt.ItemDataRole.UserRole) == design:
                    self.list.setCurrentRow(i)
                    break

    @property
    def family(self) -> str:
        it = self.tree.currentItem()
        key = it.data(0, Qt.ItemDataRole.UserRole) if it else None
        return key or "rsm"

    @property
    def design(self) -> str:
        it = self.list.currentItem()
        return it.data(Qt.ItemDataRole.UserRole) if it else "ccd"

    def _family_changed(self) -> None:
        key = self.tree.currentItem().data(0, Qt.ItemDataRole.UserRole) \
            if self.tree.currentItem() else None
        if not key:
            return
        self.list.blockSignals(True)
        self.list.clear()
        for dkey, text in FAMILIES[key][1]:
            it = QListWidgetItem(text)
            it.setData(Qt.ItemDataRole.UserRole, dkey)
            self.list.addItem(it)
        self.list.blockSignals(False)
        self.list.setCurrentRow(0)

    # ------------------------------------------------------------------ design page state
    def kind(self) -> str:
        d = self.design
        if d in designs.MIXTURE_DESIGNS:
            return "mixture"
        if d in CATEGORIC:
            return "categoric"
        return "process"

    def _columns(self) -> list[str]:
        k = self.kind()
        if k == "mixture":
            return ["Component", "Name", "Units", "Type", "Low", "High"]
        if k == "categoric":
            return ["Factor", "Name", "Units", "Type", "Levels", "Level Names"]
        cols = ["Factor", "Name", "Units", "Type", "Minimum", "Maximum", "Coded Low",
                "Coded High"]
        if self.design == "ccd":
            cols += ["-alpha", "+alpha"]
        return cols

    def _design_changed(self) -> None:
        d = self.design
        self._read_table()
        if self.kind() != getattr(self, "_last_kind", None):
            self._rows = []  # numeric, categoric and mixture entries do not carry over
            self._last_kind = self.kind()
        self._loading = True
        self.title.setText(f"{FAMILIES[self.family][0]}: {designs.DESIGN_TITLES[d]}")
        self.desc.setText(HELP.get(d, DESCRIPTIONS.get(d, "")))
        lo, hi = LIMITS[d]
        self.k_spin.setRange(lo, hi)
        if not self._rows:
            self.k_spin.setValue(DEFAULT_K[d])
        else:
            self.k_spin.setValue(min(max(len(self._rows), lo), hi))
        self.k_label.setText({"mixture": "Components:", "categoric": "Categoric factors:"}
                             .get(self.kind(), "Numeric factors:"))
        self.ccd_alpha_entry.setVisible(d == "ccd")
        self._build_options()
        self._loading = False
        self._resize_factors()

    def _default_row(self, j: int) -> dict:
        return {"name": designs.letter(j), "units": "", "low": "-1", "high": "1",
                "mlow": "0", "mhigh": "", "levels": "1, 2, 3" if self.design != "taguchi"
                else ""}

    def _resize_factors(self) -> None:
        if self._loading:
            return
        self._read_table()
        k = self.k_spin.value()
        while len(self._rows) < k:
            self._rows.append(self._default_row(len(self._rows)))
        centers = {"ccd": designs.CCD_CENTERS, "bbd": designs.BBD_CENTERS}.get(self.design)
        w = self._opt.get("center_points")
        if centers and w is not None and w.value() == centers.get(self._opt_k, -1):
            w.setValue(centers.get(k, w.value()))  # untouched default follows k
        self._opt_k = k
        self._fill_table(k)
        self._changed()
        self._update_summary()

    def _read_table(self) -> None:
        for r in range(min(self.table.rowCount(), len(self._rows))):
            row = self._rows[r]
            for c, name in enumerate(self.table_columns):
                it = self.table.item(r, c)
                if it is None:
                    continue
                key = {"Name": "name", "Units": "units", "Minimum": "low", "Maximum": "high",
                       "Low": "mlow", "High": "mhigh", "Level Names": "levels"}.get(name)
                if key:
                    row[key] = it.text()

    table_columns: list[str] = []

    def _fill_table(self, k: int) -> None:
        cols = self._columns()
        self.table_columns = cols
        self._loading = True
        self.table.clear()
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(k)
        ro = Qt.ItemFlag.ItemIsEnabled
        kind = self.kind()
        taguchi_levels = designs.TAGUCHI_ARRAYS.get(self._opt_value("array", "L9"),
                                                    (0, []))[1] if self.design == "taguchi" \
            else []
        for r in range(k):
            row = self._rows[r]
            for c, name in enumerate(cols):
                if name in ("Factor", "Component"):
                    it = QTableWidgetItem(designs.letter(r) + (f" [X{r + 1}]"
                                                               if kind == "process" else ""))
                    it.setFlags(ro)
                    it.setBackground(Qt.GlobalColor.white)
                elif name == "Name":
                    it = QTableWidgetItem(row["name"])
                elif name == "Units":
                    it = QTableWidgetItem(row["units"])
                elif name == "Type":
                    it = QTableWidgetItem({"mixture": "Mixture", "categoric": "Categoric"}
                                          .get(kind, "Numeric"))
                    it.setFlags(ro)
                    it.setToolTip("The factor type is set by the design: numeric for factorial "
                                  "and response surface designs, categoric for multilevel and "
                                  "Taguchi designs.")
                elif name == "Minimum":
                    it = QTableWidgetItem(row["low"])
                elif name == "Maximum":
                    it = QTableWidgetItem(row["high"])
                elif name == "Low":
                    it = QTableWidgetItem(row["mlow"])
                elif name == "High":
                    it = QTableWidgetItem(row["mhigh"] or _fmt(self._total()))
                elif name == "Level Names":
                    text = row["levels"]
                    if self.design == "taguchi":
                        nl = taguchi_levels[r] if r < len(taguchi_levels) else 2
                        parts = [p.strip() for p in text.split(",") if p.strip()]
                        if len(parts) != nl:
                            text = ", ".join(str(i + 1) for i in range(nl))
                            row["levels"] = text
                    it = QTableWidgetItem(text)
                    it.setToolTip("Level names or values, separated by commas.")
                elif name == "Levels":
                    it = QTableWidgetItem(str(len([p for p in row["levels"].split(",")
                                                   if p.strip()])))
                    it.setFlags(ro)
                else:  # computed: coded low/high, -alpha/+alpha
                    it = QTableWidgetItem("")
                    it.setFlags(ro)
                    it.setForeground(Qt.GlobalColor.darkGray)
                self.table.setItem(r, c, it)
        hh = self.table.horizontalHeader()
        for c, name in enumerate(cols):
            mode = QHeaderView.ResizeMode.Stretch if name in ("Name", "Level Names") else \
                QHeaderView.ResizeMode.ResizeToContents
            hh.setSectionResizeMode(c, mode)
        self._loading = False
        self._refresh_coded()

    def _refresh_coded(self) -> None:
        if self.kind() != "process" or not self.table_columns:
            return
        self._loading = True
        cols = self.table_columns
        alpha = self._ccd_alpha() if self.design == "ccd" else None
        for r in range(self.table.rowCount()):
            lo = _num(self._cell(r, "Minimum"), -1)
            hi = _num(self._cell(r, "Maximum"), 1)
            if alpha and self.ccd_alpha_entry.isChecked():
                c, h = (lo + hi) / 2, (hi - lo) / 2 / alpha
                clo, chi, alo, ahi = c - h, c + h, lo, hi
            else:
                clo, chi = lo, hi
                h = (hi - lo) / 2
                alo, ahi = ((lo + hi) / 2 - h * alpha, (lo + hi) / 2 + h * alpha) if alpha \
                    else (None, None)
            values = {"Coded Low": f"-1 ↔ {_fmt(clo)}", "Coded High": f"+1 ↔ {_fmt(chi)}"}
            if alpha:
                values["-alpha"] = f"{_fmt(alo)}"
                values["+alpha"] = f"{_fmt(ahi)}"
            for name, text in values.items():
                if name in cols:
                    self.table.item(r, cols.index(name)).setText(text)
        self._loading = False

    def _cell(self, r: int, name: str) -> str:
        if name not in self.table_columns:
            return ""
        it = self.table.item(r, self.table_columns.index(name))
        return it.text() if it else ""

    def _item_changed(self, item) -> None:
        if self._loading:
            return
        name = self.table_columns[item.column()] if item.column() < len(self.table_columns) \
            else ""
        if name == "Level Names":
            self._loading = True
            lv = self.table_columns.index("Levels")
            n = len([p for p in item.text().split(",") if p.strip()])
            self.table.item(item.row(), lv).setText(str(n))
            self._loading = False
        if name in ("Minimum", "Maximum"):
            self._refresh_coded()
        self._changed()

    # ------------------------------------------------------------------ options
    def _clear_options(self) -> None:
        while self.options_form.count():
            w = self.options_form.takeAt(0).widget()
            if w is not None:
                w.hide()
                w.deleteLater()
        self._opt = {}
        self._n_opt = 0

    def _add_row(self, label: str, w: QWidget) -> None:
        """Options in two columns of label / field pairs."""
        r, c = divmod(self._n_opt, 2)
        if label:
            lab = QLabel(label)
            lab.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.options_form.addWidget(lab, r, 2 * c)
        self.options_form.addWidget(w, r, 2 * c + 1)
        self._n_opt += 1

    def _spin(self, key, label, lo, hi, value, tip=""):
        w = QSpinBox()
        w.setRange(lo, hi)
        w.setValue(value)
        w.valueChanged.connect(self._changed)
        if tip:
            w.setToolTip(tip)
        self._opt[key] = w
        self._add_row(label, w)
        return w

    def _combo(self, key, label, items, current=None):
        w = QComboBox()
        for data, text in items:
            w.addItem(text, data)
        if current is not None:
            i = w.findData(current)
            if i >= 0:
                w.setCurrentIndex(i)
        w.currentIndexChanged.connect(self._changed)
        self._opt[key] = w
        self._add_row(label, w)
        return w

    def _check(self, key, label, value=False):
        w = QCheckBox(label)
        w.setChecked(value)
        w.toggled.connect(self._changed)
        self._opt[key] = w
        self._add_row("", w)
        return w

    def _dspin(self, key, label, lo, hi, value, decimals=4):
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setDecimals(decimals)
        w.setValue(value)
        w.valueChanged.connect(self._changed)
        self._opt[key] = w
        self._add_row(label, w)
        return w

    def _opt_value(self, key, default=None):
        w = self._opt.get(key)
        if w is None:
            return default
        if isinstance(w, QComboBox):
            return w.currentData()
        if isinstance(w, QCheckBox):
            return w.isChecked()
        return w.value()

    def _build_options(self) -> None:
        self._clear_options()
        d = self.design
        k = self.k_spin.value()
        self._opt_k = k
        self._frac_k = None
        if d == "factorial":
            self._combo("fraction", "Design:", [(0, "Full factorial")])
            self._combo("blocks", "Blocks:", [(1, "1"), (2, "2"), (4, "4"), (8, "8")])
            self._spin("center_points", "Center points per block:", 0, 50, 0)
            self._spin("replicates", "Replicates:", 1, 20, 1)
        elif d == "pb":
            self._spin("center_points", "Center points:", 0, 50, 0)
            self._spin("replicates", "Replicates:", 1, 20, 1)
        elif d == "general":
            self._spin("replicates", "Replicates:", 1, 20, 1)
        elif d == "taguchi":
            self._combo("array", "Orthogonal array:", [(a, f"{a} ({n} runs)") for a, (n, _)
                                                       in designs.TAGUCHI_ARRAYS.items()], "L9")
            self._opt["array"].currentIndexChanged.connect(lambda _: self._resize_factors())
            self._spin("replicates", "Replicates:", 1, 20, 1)
        elif d == "ccd":
            self._combo("alpha_type", "Alpha:", ALPHAS, "rotatable")
            self._dspin("alpha", "Alpha value (Other):", 0.1, 10, 1.414, 4)
            self._combo("ccd_type", "Type:", CCD_TYPES, "circumscribed")
            self._combo("fraction", "Factorial portion:", [(0, "Full")])
            self._spin("center_points", "Center points:", 0, 50, designs.CCD_CENTERS.get(k, 6))
            self._combo("blocks", "Blocks:", [(1, "1"), (2, "2"), (3, "3"), (5, "5")])
            self._spin("center_axial", "Center points in axial block:", 0, 20, 2)
            self._spin("replicates", "Replicates:", 1, 20, 1)
            for key in ("alpha_type", "ccd_type", "fraction", "center_points", "blocks"):
                w = self._opt[key]
                sig = w.currentIndexChanged if isinstance(w, QComboBox) else w.valueChanged
                sig.connect(lambda *_: self._refresh_coded())
            self._opt["alpha"].valueChanged.connect(lambda *_: self._refresh_coded())
        elif d in ("bbd", "threelevel"):
            self._spin("center_points", "Center points:", 0, 50,
                       designs.BBD_CENTERS.get(k, 5) if d == "bbd" else 0)
            self._spin("replicates", "Replicates:", 1, 20, 1)
        elif d == "dopt":
            self._combo("model", "Model:", MODELS, "quadratic")
            self._spin("model_points", "Model points (0 = number of terms):", 0, 500, 0)
            self._spin("lof_points", "Lack-of-fit points:", 0, 50, 5)
            self._spin("replicate_points", "Replicate points:", 0, 50, 5)
        elif d in designs.MIXTURE_DESIGNS:
            self._dspin("total", "Mixture total:", 0.0001, 1e6, 1.0, 4)
            self._opt["total"].valueChanged.connect(lambda *_: self._resize_factors())
            if d == "lattice":
                self._spin("degree", "Lattice degree:", 1, 6, 2)
            if d in ("lattice", "centroid"):
                self._check("augment", "Augment with axial check blends and the overall "
                                       "centroid", True)
            if d == "ev":
                self._check("vertex", "Vertices", True)
                self._check("edge", "Edge centers", True)
                self._check("plane", "Plane centroids", False)
                self._check("centroid", "Overall centroid", True)
            if d == "mixdopt":
                self._combo("model", "Model:", MIX_MODELS, "quadratic")
                self._spin("model_points", "Model points (0 = number of terms):", 0, 500, 0)
                self._spin("lof_points", "Lack-of-fit points:", 0, 50, 5)
                self._spin("replicate_points", "Replicate points:", 0, 50, 5)
            self._spin("center_points", "Replicates of the centroid:", 0, 20, 0)
        self._update_fractions()

    def _update_fractions(self) -> None:
        w = self._opt.get("fraction")
        if w is None:
            return
        k = self.k_spin.value()
        current = w.currentData() or 0
        w.blockSignals(True)
        w.clear()
        w.addItem(f"Full ({2 ** k} runs)" if self.design == "factorial" else "Full", 0)
        for p in range(1, k):
            if k - p < 2:
                break
            try:
                gens = designs.default_generators(k, p)
                _, group = designs.fractional_factorial(k, gens)
            except ValueError:
                break
            res = designs.resolution(group) or 0
            if self.design == "ccd" and res < 5:
                break
            if res < 3:
                break
            w.addItem(f"1/{2 ** p} fraction ({2 ** (k - p)} runs, Res "
                      f"{designs.ROMAN.get(res, res)})", p)
        old_k = getattr(self, "_frac_k", None)
        if self.design == "ccd" and (old_k is None or current == designs.CCD_FRACTION.get(
                old_k, 0)):  # still the default: follow the default for the new k
            current = designs.CCD_FRACTION.get(k, 0)
        self._frac_k = k
        i = w.findData(current)
        w.setCurrentIndex(max(i, 0))
        w.blockSignals(False)

    def _ccd_alpha(self) -> float:
        k = self.k_spin.value()
        try:
            spec_opt = self._options()
            res = designs.central_composite(k, spec_opt.get("alpha_type", "rotatable"),
                                            spec_opt.get("ccd_type", "circumscribed"),
                                            spec_opt.get("center_points"),
                                            int(spec_opt.get("blocks", 1)),
                                            int(spec_opt.get("center_axial", 2)),
                                            spec_opt.get("generators", ""),
                                            spec_opt.get("alpha"))
            return float(res["alpha"])
        except (ValueError, ZeroDivisionError):
            return 1.0

    def _total(self) -> float:
        return float(self._opt_value("total", 1.0) or 1.0)

    # ------------------------------------------------------------------ specification
    def _options(self) -> dict:
        d = self.design
        opt: dict = {}
        for key in ("blocks", "center_points", "replicates", "array", "alpha_type", "ccd_type",
                    "center_axial", "model", "lof_points", "replicate_points", "degree",
                    "augment", "total"):
            v = self._opt_value(key)
            if v is not None:
                opt[key] = v
        if self._opt_value("alpha_type") == "custom":
            opt["alpha"] = self._opt_value("alpha")
        mp = self._opt_value("model_points")
        if mp:
            opt["model_points"] = mp
        p = self._opt_value("fraction", 0) or 0
        k = self.k_spin.value()
        if d == "factorial":
            opt["fraction"] = p
        elif d == "ccd" and p:
            opt["generators"] = designs.default_generators(k, p)
        if d == "ev":
            opt["include"] = [n for n in ("vertex", "edge", "plane", "centroid")
                              if self._opt_value(n)]
        return opt

    def spec(self) -> dict:
        self._read_table()
        d = self.design
        k = self.k_spin.value()
        kind = self.kind()
        opt = self._options()
        alpha = self._ccd_alpha() if d == "ccd" and self.ccd_alpha_entry.isChecked() else None
        factors = []
        for r in range(k):
            row = self._rows[r]
            f = {"name": row["name"].strip() or designs.letter(r), "units": row["units"].strip()}
            if kind == "process":
                lo, hi = _num(row["low"], -1), _num(row["high"], 1)
                if alpha:
                    c, h = (lo + hi) / 2, (hi - lo) / 2 / alpha
                    lo, hi = c - h, c + h
                f.update(low=lo, high=hi)
            elif kind == "mixture":
                f.update(low=_num(row["mlow"], 0.0),
                         high=_num(row["mhigh"], self._total()) if row["mhigh"].strip()
                         else self._total())
            else:
                f["levels"] = [p.strip() for p in row["levels"].split(",") if p.strip()]
            factors.append(f)
        responses = []
        for i in range(self.responses.rowCount()):
            n = self.responses.item(i, 1).text().strip() or f"R{i + 1}"
            responses.append({"name": n, "units": self.responses.item(i, 2).text().strip()})
        return {"design": d, "factors": factors, "responses": responses, "options": opt,
                "randomize": self.randomize.isChecked(), "seed": self.seed,
                "name": self.name_edit.text().strip() or "Design"}

    def _changed(self, *_) -> None:
        if self._loading:
            return
        if "fraction" in self._opt and self.sender() is not self._opt.get("fraction"):
            self._update_fractions()
        self._timer.start()

    def _update_summary(self) -> None:
        if "alpha" in self._opt:
            self._opt["alpha"].setEnabled(self._opt_value("alpha_type") == "custom")
        if "center_axial" in self._opt:
            self._opt["center_axial"].setEnabled((self._opt_value("blocks") or 1) > 1)
        spec = self.spec()
        count = designs.design_run_count(spec)
        if isinstance(count, str):
            self.summary.setText(f"⚠ {count}")
            self.summary.setStyleSheet("background:#FDECEC; border:1px solid #E7A1A1; "
                                       "color:#8A1010; padding:5px 8px;")
        else:
            extra = ""
            if self.design == "ccd":
                extra = f", alpha = {self._ccd_alpha():.5g}"
            self.summary.setText(f"{designs.DESIGN_TITLES[self.design]}: "
                                 f"{len(spec['factors'])} factors, {count} runs{extra}.  "
                                 "Click Next to name the responses.")
            self.summary.setStyleSheet("")

    def set_factors(self, rows: list[dict]) -> None:
        """Fill the factor table (used by tests and the tutorial examples)."""
        self._rows = []
        for j, r in enumerate(rows):
            row = self._default_row(j)
            row.update({k: str(v) for k, v in r.items()})
            self._rows.append(row)
        self._loading = True
        self.k_spin.setValue(len(rows))
        self._loading = False
        self._fill_table(len(rows))
        self._changed()

    def set_option(self, key: str, value) -> None:
        w = self._opt[key]
        if isinstance(w, QComboBox):
            w.setCurrentIndex(w.findData(value))
        elif isinstance(w, QCheckBox):
            w.setChecked(bool(value))
        else:
            w.setValue(value)

    # ------------------------------------------------------------------ finish
    def finish(self) -> None:
        spec = self.spec()
        try:
            doc = DesignDocument.from_spec(spec)
        except (ValueError, KeyError) as exc:
            self.error.setText(str(exc))
            return
        if not math.isfinite(doc.n_runs) or doc.n_runs == 0:
            self.error.setText("The design has no runs.")
            return
        self.document = doc
        self.created.emit(doc)
        self.accept()

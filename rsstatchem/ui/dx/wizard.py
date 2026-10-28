"""New Design wizard, laid out like Design-Expert's.

Page 1: the design families on the left (Standard Designs: Factorial, Response Surface,
Mixture, Split-Plot; Custom Designs: Optimal), the designs of the chosen family in the middle
and, on the right, the number of factors, the factor table (name, units, type — numeric or
categoric with named levels —, changes — easy or hard to change, which makes a split-plot —,
minimum/maximum or levels, coded low/high and the axial points of a CCD) and the design
options (alpha, centre points, blocks, replicates, fraction, groups, ...), with the run count
updated live.
Page 2: the responses (number, names, units). Finish builds the design with
:func:`rsstatchem.core.doe.designs.build_design` and emits ``created(DesignDocument)``.

The wizard is never run with ``exec()``; it is shown with ``open()`` (window-modal sheet).
"""
from __future__ import annotations

import math
import random

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog,
                               QDoubleSpinBox, QFrame, QGridLayout, QGroupBox, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QPushButton, QSpinBox, QStackedWidget, QTableWidget,
                               QTableWidgetItem, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
                               QWidget)

from rsstatchem.core.doe import designs
from rsstatchem.ui.doe.wizard import DEFAULT_K as _OLD_DEFAULT_K
from rsstatchem.ui.doe.wizard import DESCRIPTIONS
from rsstatchem.ui.doe.wizard import LIMITS as _OLD_LIMITS
from rsstatchem.ui.dx import icons, style
from rsstatchem.ui.dx.document import DesignDocument

LIMITS = dict(_OLD_LIMITS, dsd=(3, 30), onefactor=(1, 1), simscreen=(3, 10),
              combined=(3, 10), lhs=(1, 20), distance=(1, 10), userdef=(1, 8),
              historical=(1, 20))
DEFAULT_K = dict(_OLD_DEFAULT_K, dsd=6, onefactor=1, simscreen=4, combined=4, lhs=3,
                 distance=3, userdef=3, historical=2)
# family key -> (tree text, designs)
FAMILIES = {
    "factorial": ("Factorial", [("factorial", "Regular Two-Level"),
                                ("mr5", "Min-Run Characterize"), ("mr4", "Min-Run Screen"),
                                ("dsd", "Definitive Screening"),
                                ("general", "Multilevel Categoric"),
                                ("pb", "Plackett-Burman"), ("taguchi", "Taguchi OA")]),
    "rsm": ("Response Surface", [("ccd", "Central Composite"), ("bbd", "Box-Behnken"),
                                 ("threelevel", "3-Level Factorial"),
                                 ("dopt", "Optimal (Custom)"), ("onefactor", "One Factor"),
                                 ("userdef", "User-Defined"),
                                 ("historical", "Historical Data"),
                                 ("lhs", "Space Filling: Latin Hypercube"),
                                 ("distance", "Space Filling: Optimal Distance")]),
    "mixture": ("Mixture", [("lattice", "Simplex Lattice"), ("centroid", "Simplex Centroid"),
                            ("simscreen", "Simplex Screening"),
                            ("ev", "Screening (Extreme Vertices)"),
                            ("mixdopt", "Optimal (Custom)")]),
    "splitplot": ("Split-Plot", [("factorial", "Regular Two-Level"),
                                 ("general", "Multilevel Categoric"),
                                 ("ccd", "Central Composite"), ("dopt", "Optimal (Custom)")]),
    "custom": ("Optimal (Custom)", [("dopt", "Optimal (Custom) Process"),
                                    ("mixdopt", "Optimal (Custom) Mixture"),
                                    ("combined", "Combined Mixture-Process")]),
}
TREE = [("Standard Designs", ["factorial", "rsm", "mixture", "splitplot"]),
        ("Custom Designs", ["custom"])]
RO = Qt.ItemFlag.ItemIsEnabled
HELP = {
    "ccd": "Central composite design: a two-level factorial (or fraction), axial (star) points "
           "at ±alpha and centre points. The standard design for a quadratic model.",
    "bbd": "Box-Behnken: every factor at three levels, edge midpoints and centre points, no "
           "extreme corners. 3 to 7 factors.",
    "mr5": "Min-Run Characterize: a minimum-run equireplicated resolution V design (MR5), 6 "
           "to 50 factors. All main effects and two-factor interactions can be estimated with "
           "about as many runs as model terms (an even number, each factor balanced). Built "
           "by a fast columnwise-pairwise exchange started from runs of a regular resolution V "
           "fraction (about a second for 50 factors); losing a run reduces it to resolution "
           "IV.",
    "dopt": "Optimal (custom): runs chosen for the model you plan to fit by I-, D- or "
            "A-optimal exchange (point, coordinate or best of both), plus lack-of-fit and "
            "replicate points. Optional blocks, an edited model, multiple linear constraints "
            "(e.g. A + B <= 15), forced runs and an imported candidate list.",
    "mixdopt": "Optimal (custom) mixture: I-, D- or A-optimal blends inside the constrained "
               "mixture region (component limits plus multiple linear constraints), plus "
               "lack-of-fit and replicate points.",
    "combined": "Combined mixture-process: mixture components (Type = Mixture, listed first) "
                "and process factors in one optimal design, for the crossed Scheffé x "
                "process model (e.g. quadratic mixture x linear process).",
    "dsd": "Definitive screening (Jones & Nachtsheim): 2m+1 runs from a conference matrix, "
           "three levels per numeric factor. Main effects are clear of two-factor "
           "interactions and quadratic effects; two-level categoric factors allowed.",
    "onefactor": "One-factor response surface: an optimal set of levels for a linear, "
                 "quadratic or cubic curve in one numeric factor, plus lack-of-fit and "
                 "replicate points.",
    "simscreen": "Simplex screening (Snee-Marquardt): vertices, interior axial check blends, "
                 "end-effect blends and the overall centroid, to screen components for "
                 "linear blending.",
    "userdef": "User-defined: pick the point types (vertices, edge centres, thirds of edges, "
               "plane centroids, axial check points, centre) of the possibly constrained "
               "factor region.",
    "historical": "Historical data: a blank layout of the given number of rows; type in (or "
                  "paste) existing factor settings and responses. Use File > Import Data Set "
                  "to build one from a CSV / Excel table.",
    "lhs": "Latin hypercube (space filling): every factor's range cut into as many slices as "
           "runs, each slice used once; good for computer experiments and nonlinear models.",
    "distance": "Optimal distance (space filling): runs spread out to maximise the smallest "
                "distance between them (maximin), honouring constraints.",
    "mr4": "Min-Run Screen: a minimum-run resolution IV design (MR4) in 2 x factors runs, 5 to "
           "50 factors — the foldover of a main-effects design, so main effects are clear of "
           "two-factor interactions. The 2 extra runs (default) protect against a botched "
           "run.",
}
SPLIT_HELP = ("  Split-plot: set Changes = Hard for factors that are hard to change; their "
              "settings stay fixed within each group (whole plot), the groups are run in random "
              "order and the runs within each group in random order. Numeric and categoric "
              "factors can be hard or easy to change; two-level designs can be blocked by "
              "confounding whole-plot interactions. The analysis uses REML.")
ALPHAS = [("rotatable", "Rotatable"), ("spherical", "Spherical"),
          ("orthogonal", "Orthogonal Quadratic"), ("practical", "Practical"),
          ("face", "Face Centered"), ("custom", "Other (enter value)")]
CCD_TYPES = [("circumscribed", "Circumscribed (CCC)"), ("inscribed", "Inscribed (CCI)"),
             ("face", "Face centered (CCF)")]
MODELS = [("linear", "Linear"), ("2fi", "2FI"), ("quadratic", "Quadratic"), ("cubic", "Cubic")]
PROC_MODELS = [("mean", "Mean"), ("linear", "Linear"), ("2fi", "2FI"),
               ("quadratic", "Quadratic"), ("cubic", "Cubic")]
COMB_MIX_MODELS = [("linear", "Linear"), ("quadratic", "Quadratic"),
                   ("special_cubic", "Special Cubic")]
CRITERIA = [("I", "I-optimal (prediction)"), ("D", "D-optimal (coefficients)"),
            ("A", "A-optimal")]
EXCHANGES = [("best", "Best"), ("point", "Point exchange"),
             ("coordinate", "Coordinate exchange")]
POINT_TYPES = [("vertex", "Vertices"), ("edge", "Edge centers"), ("thirds", "Thirds of edges"),
               ("plane", "Plane centroids"), ("interior", "Axial check points"),
               ("centroid", "Overall centroid")]
ENGINE_DESIGNS = ("dopt", "mixdopt", "combined")
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
        # edited model terms, constraints, forced runs and imported candidates (optimal)
        self._custom: dict = {"terms": None, "constraints": [], "forced": [],
                              "candidates": []}
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
                                          "mixture": "ternary", "custom": "evaluation",
                                          "splitplot": "splitplot"}[key]))
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

    @property
    def split_capable(self) -> bool:
        return self.design in designs.SPLIT_PLOT_DESIGNS

    def split(self) -> bool:
        """Split-plot: some factor is hard to change."""
        return self.split_capable and any(r.get("changes") == "Hard" for r in
                                          self._rows[:self.k_spin.value()])

    def _columns(self) -> list[str]:
        k = self.kind()
        if k == "mixture":
            return ["Component", "Name", "Units", "Type", "Low", "High"]
        changes = ["Changes"] if self.split_capable else []
        if k == "categoric":
            return ["Factor", "Name", "Units", "Type"] + changes + ["Levels", "Level Names"]
        cols = ["Factor", "Name", "Units", "Type"] + changes + [
            "Minimum", "Maximum", "Levels", "Level Names", "Coded Low", "Coded High"]
        if self.design == "ccd":
            cols += ["-alpha", "+alpha"]
        return cols

    def _design_changed(self) -> None:
        d = self.design
        self._read_table()
        if self.kind() != getattr(self, "_last_kind", None):
            self._rows = []  # numeric, categoric and mixture entries do not carry over
            self._last_kind = self.kind()
        if d != getattr(self, "_last_design", None):
            self._custom = {"terms": None, "constraints": [], "forced": [], "candidates": []}
            self._last_design = d
        retyped = False
        for j, row in enumerate(self._rows):
            if d == "combined" and j < 3 and sum(r.get("type") == "Mixture"
                                                 for r in self._rows) < 2:
                row.update(type="Mixture", low="0", high="1")
                retyped = True
            elif d != "combined" and row.get("type") == "Mixture":
                row.update(type="Numeric", low="-1", high="1")
                retyped = True
        if retyped:  # the table still shows the old types: do not read it back
            self.table.setRowCount(0)
        self._loading = True
        self.title.setText(f"{FAMILIES[self.family][0]}: {designs.DESIGN_TITLES[d]}")
        self.desc.setText(HELP.get(d, DESCRIPTIONS.get(d, ""))
                          + (SPLIT_HELP if self.family == "splitplot" else ""))
        lo, hi = LIMITS[d]
        self.k_spin.setRange(lo, hi)
        if not self._rows:
            self.k_spin.setValue(DEFAULT_K[d] if self.family != "splitplot" else
                                 max(DEFAULT_K[d], 3 if d != "general" else 2))
        else:
            self.k_spin.setValue(min(max(len(self._rows), lo), hi))
        self.k_label.setText({"mixture": "Components:", "categoric": "Categoric factors:"}
                             .get(self.kind(), "Factors:"))
        self.ccd_alpha_entry.setVisible(d == "ccd")
        self._build_options()
        self._loading = False
        self._resize_factors()
        if self.family == "splitplot" and self._rows and not self.split():
            self._rows[0]["changes"] = "Hard"  # a split-plot needs a hard-to-change factor
            self._fill_table(self.k_spin.value())
            self._changed()

    def _default_row(self, j: int) -> dict:
        row = {"name": designs.letter(j), "units": "", "low": "-1", "high": "1",
               "mlow": "0", "mhigh": "", "type": "Numeric", "changes": "Easy",
               "levels": ("1, 2, 3" if self.kind() == "categoric" else "L1, L2")
               if self.design != "taguchi" else ""}
        if self.design == "combined" and j < 3:
            row.update(type="Mixture", low="0", high="1")
        return row

    def _row_mixture(self, r: int) -> bool:
        return self.design == "combined" and r < len(self._rows) and \
            self._rows[r].get("type") == "Mixture"

    def _resize_factors(self) -> None:
        if self._loading:
            return
        self._read_table()
        k = self.k_spin.value()
        while len(self._rows) < k:
            self._rows.append(self._default_row(len(self._rows)))
        centers = {"ccd": designs.CCD_CENTERS, "bbd": designs.BBD_CENTERS}.get(self.design)
        w = self._opt.get("center_points")
        if centers and w is not None and w.value() == centers.get(self._opt_k, -1) \
                and not self.split():
            w.setValue(centers.get(k, w.value()))  # untouched default follows k
        if self._opt_k != k and getattr(self, "_custom", None):
            self._custom["terms"] = None
            self._custom["forced"] = []
            self._custom["candidates"] = []
        self._opt_k = k
        self._fill_table(k)
        self._changed()
        self._update_summary()

    def _read_table(self) -> None:
        for r in range(min(self.table.rowCount(), len(self._rows))):
            row = self._rows[r]
            for c, name in enumerate(self.table_columns):
                if name in ("Type", "Changes"):
                    w = self.table.cellWidget(r, c)
                    if isinstance(w, QComboBox):
                        row[name.lower()] = w.currentText()
                    continue
                it = self.table.item(r, c)
                if it is None:
                    continue
                key = {"Name": "name", "Units": "units", "Minimum": "low", "Maximum": "high",
                       "Low": "mlow", "High": "mhigh", "Level Names": "levels"}.get(name)
                if key and it.flags() & Qt.ItemFlag.ItemIsEditable:
                    row[key] = it.text()

    table_columns: list[str] = []

    def _row_categoric(self, r: int) -> bool:
        kind = self.kind()
        return kind == "categoric" or (kind == "process"
                                       and self._rows[r].get("type") == "Categoric")

    def _fill_table(self, k: int) -> None:
        cols = self._columns()
        self.table_columns = cols
        self._loading = True
        self.table.clear()
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(k)
        kind = self.kind()
        taguchi_levels = designs.TAGUCHI_ARRAYS.get(self._opt_value("array", "L9"),
                                                    (0, []))[1] if self.design == "taguchi" \
            else []
        for r in range(k):
            row = self._rows[r]
            catrow = self._row_categoric(r)
            for c, name in enumerate(cols):
                if name in ("Factor", "Component"):
                    it = QTableWidgetItem(designs.letter(r) + (f" [X{r + 1}]"
                                                               if kind == "process" else ""))
                    it.setFlags(RO)
                    it.setBackground(Qt.GlobalColor.white)
                elif name == "Name":
                    it = QTableWidgetItem(row["name"])
                elif name == "Units":
                    it = QTableWidgetItem(row["units"])
                elif name == "Type" and kind == "process":
                    self.table.setItem(r, c, QTableWidgetItem(""))
                    self.table.setCellWidget(r, c, self._cell_combo(
                        (["Mixture"] if self.design == "combined" else [])
                        + ["Numeric", "Categoric"], row.get("type", "Numeric"),
                        lambda _=0, rr=r: self._type_changed(rr),
                        "Numeric: a continuous range (minimum to maximum). Categoric: named "
                        "levels (nominal), coded with effect coding in the model."))
                    continue
                elif name == "Type":
                    it = QTableWidgetItem({"mixture": "Mixture"}.get(kind, "Categoric"))
                    it.setFlags(RO)
                    it.setToolTip("The factor type is set by the design: mixture components, "
                                  "or categoric levels for multilevel and Taguchi designs.")
                elif name == "Changes":
                    self.table.setItem(r, c, QTableWidgetItem(""))
                    self.table.setCellWidget(r, c, self._cell_combo(
                        ["Easy", "Hard"], row.get("changes", "Easy"),
                        lambda _=0, rr=r: self._changes_changed(rr),
                        "Hard-to-change factors are reset only between groups of runs (whole "
                        "plots): the design becomes a split-plot with restricted "
                        "randomisation."))
                    continue
                elif name == "Minimum":
                    it = QTableWidgetItem("" if catrow else row["low"])
                elif name == "Maximum":
                    it = QTableWidgetItem("" if catrow else row["high"])
                elif name == "Low":
                    it = QTableWidgetItem(row["mlow"])
                elif name == "High":
                    it = QTableWidgetItem(row["mhigh"] or _fmt(self._total()))
                elif name == "Level Names":
                    text = row["levels"] if catrow else ""
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
                                                   if p.strip()])) if catrow else "")
                    it.setFlags(RO)
                else:  # computed: coded low/high, -alpha/+alpha
                    it = QTableWidgetItem("")
                    it.setFlags(RO)
                    it.setForeground(Qt.GlobalColor.darkGray)
                if catrow and name in ("Minimum", "Maximum"):
                    it.setFlags(RO)
                    it.setBackground(QColor("#EEF1F5"))
                if not catrow and name == "Level Names":
                    it.setFlags(RO)
                    it.setBackground(QColor("#EEF1F5"))
                self.table.setItem(r, c, it)
        hh = self.table.horizontalHeader()
        stretch = "Level Names" if "Level Names" in cols else "Name"
        for c, name in enumerate(cols):
            if name == stretch:
                hh.setSectionResizeMode(c, QHeaderView.ResizeMode.Stretch)
            elif name == "Name":
                hh.setSectionResizeMode(c, QHeaderView.ResizeMode.Interactive)
                self.table.setColumnWidth(c, 120)
            else:
                hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self._loading = False
        self._refresh_coded()

    def _cell_combo(self, items: list[str], current: str, slot, tip: str) -> QComboBox:
        box = QComboBox()
        box.addItems(items)
        box.setCurrentIndex(max(items.index(current) if current in items else 0, 0))
        box.setToolTip(tip)
        box.currentIndexChanged.connect(slot)
        return box

    def _type_changed(self, r: int) -> None:
        if self._loading:
            return
        self._read_table()
        row = self._rows[r]
        if row.get("type") == "Categoric" and not [p for p in row["levels"].split(",")
                                                   if p.strip()]:
            row["levels"] = "L1, L2"
        self._fill_table(self.k_spin.value())
        self._changed()

    def _changes_changed(self, r: int) -> None:
        if self._loading:
            return
        self._read_table()
        self._update_split_options()
        self._changed()

    def _refresh_coded(self) -> None:
        if self.kind() != "process" or not self.table_columns:
            return
        self._loading = True
        cols = self.table_columns
        alpha = self._ccd_alpha() if self.design == "ccd" else None
        for r in range(self.table.rowCount()):
            if self._row_categoric(r) or self._row_mixture(r):
                for name in ("Coded Low", "Coded High", "-alpha", "+alpha"):
                    if name in cols:
                        self.table.item(r, cols.index(name)).setText("")
                continue
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
        if name == "Level Names" and "Levels" in self.table_columns:
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

    def _button(self, text: str, slot, key: str | None = None) -> QPushButton:
        b = QPushButton(text)
        b.clicked.connect(slot)
        self._opt[key or "btn_" + text.strip(".").lower().split()[0]] = b
        self._add_row("", b)
        return b

    def _engine_options(self, forced: bool = True) -> None:
        """Optimality criterion, exchange method, blocks and the Edit Model / Constraints /
        Forced runs / Candidates buttons of the optimal designs."""
        self._combo("criterion", "Optimality:", CRITERIA, "I")
        self._combo("exchange", "Exchange:", EXCHANGES, "best")
        self._spin("opt_blocks", "Blocks:", 1, 20, 1)
        self._button("Edit Model...", self.edit_model, "btn_terms")
        self._button("Constraints...", self.edit_constraints, "btn_constraints")
        if forced:
            self._button("Forced Runs...", lambda: self.edit_runs("forced"), "btn_forced")
            self._button("Candidate Points...", lambda: self.edit_runs("candidates"),
                         "btn_candidates")

    def _opt_value(self, key, default=None):
        w = self._opt.get(key)
        if w is None:
            return default
        if isinstance(w, QComboBox):
            return w.currentData()
        if isinstance(w, QCheckBox):
            return w.isChecked()
        if isinstance(w, QPushButton):
            return None
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
            self._spin("groups", "Groups per replicate (0 = one per hard-to-change setting):",
                       0, 256, 0, "Split-plot: the number of whole plots in each replicate")
            self._check("block_per_replicate", "One block per replicate", False)
        elif d in ("mr5", "mr4"):
            if d == "mr4":
                self._check("extra_runs", "Add 2 runs (MR4+2, recommended)", True)
            self._spin("center_points", "Center points:", 0, 50, 0)
        elif d == "pb":
            self._spin("center_points", "Center points:", 0, 50, 0)
            self._spin("replicates", "Replicates:", 1, 20, 1)
        elif d == "general":
            self._spin("replicates", "Replicates:", 1, 20, 1)
            self._check("block_per_replicate", "One block per replicate", False)
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
            if self.family == "splitplot":
                self._opt["center_points"].setValue(1)
            for key in ("alpha_type", "ccd_type", "fraction", "center_points", "blocks"):
                w = self._opt[key]
                sig = w.currentIndexChanged if isinstance(w, QComboBox) else w.valueChanged
                sig.connect(lambda *_: self._refresh_coded())
            self._opt["alpha"].valueChanged.connect(lambda *_: self._refresh_coded())
        elif d in ("bbd", "threelevel"):
            self._spin("center_points", "Center points:", 0, 50,
                       designs.BBD_CENTERS.get(k, 5) if d == "bbd" else 0)
            self._spin("replicates", "Replicates:", 1, 20, 1)
            if d == "bbd":
                self._combo("blocks", "Blocks (4 factors: 3, 5 factors: 2):",
                            [(1, "1"), (2, "2"), (3, "3")])
        elif d == "dopt":
            self._combo("model", "Model:", MODELS, "quadratic")
            self._spin("model_points", "Model points (0 = number of terms):", 0, 500, 0)
            self._spin("lof_points", "Lack-of-fit points:", 0, 50, 5)
            self._spin("replicate_points", "Replicate points:", 0, 50, 5)
            self._engine_options()
            self._spin("groups", "Groups (whole plots):", 2, 200, 8)
            self._spin("group_size", "Runs per group:", 2, 50, 4)
            self._dspin("variance_ratio", "Variance ratio (group / residual):", 0.01, 100,
                        1.0, 2)
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
                self._check("thirds", "Thirds of edges", False)
                self._check("plane", "Plane centroids", False)
                self._check("interior", "Axial check blends", False)
                self._check("centroid", "Overall centroid", True)
                self._button("Constraints...", self.edit_constraints)
            if d == "mixdopt":
                self._combo("model", "Model:", MIX_MODELS, "quadratic")
                self._spin("model_points", "Model points (0 = number of terms):", 0, 500, 0)
                self._spin("lof_points", "Lack-of-fit points:", 0, 50, 5)
                self._spin("replicate_points", "Replicate points:", 0, 50, 5)
                self._engine_options(forced=False)
            self._spin("center_points", "Replicates of the centroid:", 0, 20, 0)
        elif d == "combined":
            self._dspin("total", "Mixture total:", 0.0001, 1e6, 1.0, 4)
            self._combo("mix_model", "Mixture model:", COMB_MIX_MODELS, "quadratic")
            self._combo("process_model", "Process model:", PROC_MODELS, "linear")
            self._spin("model_points", "Model points (0 = number of terms):", 0, 500, 0)
            self._spin("lof_points", "Lack-of-fit points:", 0, 50, 5)
            self._spin("replicate_points", "Replicate points:", 0, 50, 5)
            self._engine_options(forced=False)
        elif d == "dsd":
            self._spin("center_points", "Extra center points:", 0, 20, 0)
            self._spin("fake_factors", "Extra (fake) factors for more runs:", 0, 6, 0,
                       "Jones & Nachtsheim: two fake factors add runs that improve the "
                       "estimation of second-order effects.")
            self._spin("replicates", "Replicates:", 1, 10, 1)
        elif d == "onefactor":
            self._combo("model", "Model:", MODELS[:1] + MODELS[2:], "quadratic")
            self._combo("criterion", "Optimality:", CRITERIA, "I")
            self._spin("lof_points", "Lack-of-fit points:", 0, 20, 2)
            self._spin("replicate_points", "Replicate points:", 0, 20, 3)
        elif d in ("lhs", "distance"):
            self._spin("runs", "Runs:", 2, 1000, max(10, 5 * k))
            if d == "distance":
                self._spin("levels", "Grid levels per factor:", 3, 21, 5)
            self._button("Constraints...", self.edit_constraints)
        elif d == "userdef":
            for key, text in POINT_TYPES:
                self._check(key, text, key in ("vertex", "edge", "centroid"))
            self._spin("center_points", "Replicates of the center:", 0, 20, 0)
            self._spin("replicates", "Replicates:", 1, 10, 1)
            self._button("Constraints...", self.edit_constraints)
        elif d == "historical":
            self._spin("runs", "Rows:", 1, 10000, 10,
                       "Rows of the blank layout; type or paste the factor settings and "
                       "responses of existing runs.")
        self._update_fractions()
        self._update_split_options()

    def _update_split_options(self) -> None:
        """Options that only apply to split-plot (or only to completely randomised) runs."""
        split = self.split()
        for key in ("groups", "group_size", "variance_ratio"):
            if key in self._opt:
                self._opt[key].setEnabled(split)
        if self.design == "dopt":
            for key in ("model_points", "lof_points", "replicate_points", "criterion",
                        "exchange", "opt_blocks", "btn_terms", "btn_constraints",
                        "btn_forced", "btn_candidates"):
                if key in self._opt:
                    self._opt[key].setEnabled(not split)

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
        # the CCD (and its alpha) is built on the numeric factors only
        k = sum(1 for r in range(min(self.k_spin.value(), len(self._rows)))
                if not self._row_categoric(r)) or self.k_spin.value()
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
        split = self.split()
        for key in ("blocks", "center_points", "replicates", "array", "alpha_type", "ccd_type",
                    "center_axial", "model", "lof_points", "replicate_points", "degree",
                    "augment", "total", "extra_runs", "block_per_replicate", "criterion",
                    "exchange", "mix_model", "process_model", "fake_factors", "runs",
                    "levels"):
            v = self._opt_value(key)
            if v is not None:
                opt[key] = v
        if split:
            if d != "factorial":  # split-plot factorials block by confounding
                opt.pop("blocks", None)
            for key in ("group_size", "variance_ratio"):
                if self._opt_value(key) is not None:
                    opt[key] = self._opt_value(key)
            g = self._opt_value("groups")
            if g:
                opt["groups"] = g
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
            opt["include"] = [n for n in ("vertex", "edge", "thirds", "plane", "interior",
                                          "centroid") if self._opt_value(n)]
        if d == "userdef":
            opt["include"] = [n for n, _ in POINT_TYPES if self._opt_value(n)]
        if d == "bbd" and (opt.get("blocks") or 1) > 1 and \
                opt.get("center_points") == designs.BBD_CENTERS.get(k):
            # untouched default: the blocked designs' own centre points per block
            opt["center_points"] = designs.BBD_BLOCK_CENTERS.get(k, opt["center_points"])
        if d in ENGINE_DESIGNS and not split:
            nb = self._opt_value("opt_blocks", 1) or 1
            if nb > 1:
                opt["blocks"] = nb
            if self._custom.get("terms"):
                opt["terms"] = list(self._custom["terms"])
            if self._custom.get("forced") and d == "dopt":
                opt["forced"] = [list(r) for r in self._custom["forced"]]
            if self._custom.get("candidates") and d == "dopt":
                opt["candidates"] = [list(r) for r in self._custom["candidates"]]
        elif d in ENGINE_DESIGNS:
            for key in ("criterion", "exchange"):
                opt.pop(key, None)
        if self._custom.get("constraints") and d in ENGINE_DESIGNS + ("ev", "lhs", "distance",
                                                                      "userdef"):
            opt["constraints"] = [dict(c) for c in self._custom["constraints"]]
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
            if self.split_capable and row.get("changes") == "Hard":
                f["changes"] = "hard"
            if self._row_mixture(r):
                f.update(type="mixture", low=_num(row["low"], 0.0),
                         high=_num(row["high"], self._total()))
            elif kind == "process" and row.get("type") == "Categoric":
                f.update(type="categoric",
                         levels=[p.strip() for p in row["levels"].split(",") if p.strip()])
            elif kind == "process":
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
        if d == "combined":  # mixture components first, then the process factors
            factors = [f for f in factors if f.get("type") == "mixture"] + \
                [f for f in factors if f.get("type") != "mixture"]
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
            self._opt["center_axial"].setEnabled((self._opt_value("blocks") or 1) > 1
                                                 and not self.split())
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
            if self.split():
                hard = [designs.letter(j) for j, r in enumerate(self._rows[:len(spec['factors'])])
                        if r.get("changes") == "Hard"]
                extra += f"; split-plot, hard to change: {', '.join(hard)}"
            self.summary.setText(f"{designs.DESIGN_TITLES[self.design]}: "
                                 f"{len(spec['factors'])} factors, {count} runs{extra}.  "
                                 "Click Next to name the responses.")
            self.summary.setStyleSheet("")

    def set_factors(self, rows: list[dict]) -> None:
        """Fill the factor table (used by tests and the tutorial examples)."""
        self._rows = []
        for j, r in enumerate(rows):
            row = self._default_row(j)
            for key, v in r.items():
                if key == "levels" and isinstance(v, (list, tuple)):
                    v = ", ".join(map(str, v))
                elif key == "type":
                    v = "Categoric" if str(v).lower().startswith("cat") else "Numeric"
                elif key == "changes":
                    v = "Hard" if str(v).lower().startswith("hard") else "Easy"
                row[key] = str(v)
            self._rows.append(row)
        self._loading = True
        self.k_spin.setValue(len(rows))
        self._loading = False
        self._fill_table(len(rows))
        self._update_split_options()
        self._changed()

    def set_option(self, key: str, value) -> None:
        w = self._opt[key]
        if isinstance(w, QComboBox):
            w.setCurrentIndex(w.findData(value))
        elif isinstance(w, QCheckBox):
            w.setChecked(bool(value))
        else:
            w.setValue(value)

    # ------------------------------------------------------------------ optimal extras
    def candidate_terms(self) -> list:
        """Terms offered by Edit Model for the current optimal design."""
        import itertools

        from rsstatchem.core.doe import model as mdl
        from rsstatchem.core.doe import terms_combined as tc

        spec = self.spec()
        facs = spec["factors"]
        k = len(facs)
        cat = [f.get("type") == "categoric" for f in facs]
        if self.design == "mixdopt":
            return mdl.mixture_terms(k, "cubic")
        if self.design == "combined":
            q = sum(1 for f in facs if f.get("type") == "mixture")
            terms = tc.combined_terms(q, k - q, "special_cubic", "quadratic", cat[q:])
            extra = tc.combined_terms(q, k - q, "linear", "cubic", cat[q:])
            return terms + [t for t in extra if t not in terms]
        terms = set(mdl.process_terms(k, "cubic", cat))
        for r in range(4, min(k, 6) + 1):
            terms |= set(itertools.combinations(range(k), r))
        return mdl.sort_terms(terms)

    def model_terms(self) -> list:
        """The terms of the chosen model order (or the edited model)."""
        from rsstatchem.core.doe import model as mdl
        from rsstatchem.core.doe import terms_combined as tc

        if self._custom.get("terms"):
            return [mdl.key_term(t) for t in self._custom["terms"]]
        spec = self.spec()
        facs = spec["factors"]
        k = len(facs)
        cat = [f.get("type") == "categoric" for f in facs]
        opt = spec["options"]
        if self.design == "mixdopt":
            return mdl.mixture_terms(k, opt.get("model", "quadratic"))
        if self.design == "combined":
            q = sum(1 for f in facs if f.get("type") == "mixture")
            return tc.combined_terms(q, k - q, opt.get("mix_model", "quadratic"),
                                     opt.get("process_model", "linear"), cat[q:])
        return mdl.process_terms(k, opt.get("model", "quadratic"), cat)

    def edit_model(self) -> "TermsDialog":
        from rsstatchem.core.doe import model as mdl

        q = sum(1 for r in self._rows[:self.k_spin.value()] if r.get("type") == "Mixture") \
            if self.design == "combined" else None
        dlg = TermsDialog(self.candidate_terms(), self.model_terms(), self, n_mix=q)

        def done(result):
            if result:
                self._custom["terms"] = [mdl.term_key(t) for t in dlg.selected()]
                self._changed()
        dlg.finished.connect(done)
        dlg.open()
        return dlg

    def edit_constraints(self) -> "ConstraintsDialog":
        rows = self._rows[:self.k_spin.value()]
        if self.design == "combined":
            names = [r["name"] for r in rows if r.get("type") == "Mixture"]
        else:
            names = [r["name"] for r in rows]
        dlg = ConstraintsDialog(names, [c["text"] for c in self._custom["constraints"]], self)

        def done(result):
            if result:
                self._custom["constraints"] = dlg.constraints
                self._changed()
        dlg.finished.connect(done)
        dlg.open()
        return dlg

    def edit_runs(self, which: str) -> "RunsDialog":
        spec = self.spec()
        title = {"forced": "Forced Runs", "candidates": "Candidate Points"}[which]
        dlg = RunsDialog(title, spec["factors"], self._custom.get(which) or [], self)

        def done(result):
            if result:
                self._custom[which] = dlg.rows
                self._changed()
        dlg.finished.connect(done)
        dlg.open()
        return dlg

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


# ---------------------------------------------------------------------------- dialogs
class TermsDialog(QDialog):
    """Edit Model: tick the terms of the model the optimal design is built for."""

    def __init__(self, candidates: list, selected: list, parent=None,
                 n_mix: int | None = None):
        super().__init__(parent)
        from rsstatchem.core.doe import model as mdl
        from rsstatchem.core.doe import terms_combined as tc

        self.setWindowTitle("Edit Model")
        self.setStyleSheet(style.STYLESHEET)
        self.terms = list(candidates)
        chosen = set(selected)
        self.list = QListWidget()
        for t in self.terms:
            label = tc.term_label(t, n_mix) if n_mix else mdl.term_name(t)
            it = QListWidgetItem(label)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if t in chosen else Qt.CheckState.Unchecked)
            self.list.addItem(it)
        self.count = QLabel()
        self.list.itemChanged.connect(lambda _: self._count())
        ok, cancel = QPushButton("OK"), QPushButton("Cancel")
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        row = QHBoxLayout()
        row.addWidget(self.count, 1)
        row.addWidget(ok)
        row.addWidget(cancel)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Terms of the model the design must estimate (the intercept is "
                             "always included for process models):"))
        lay.addWidget(self.list, 1)
        lay.addLayout(row)
        self.resize(360, 480)
        self._count()

    def selected(self) -> list:
        return [t for i, t in enumerate(self.terms)
                if self.list.item(i).checkState() == Qt.CheckState.Checked]

    def set_selected(self, terms) -> None:
        want = set(terms)
        for i, t in enumerate(self.terms):
            self.list.item(i).setCheckState(Qt.CheckState.Checked if t in want
                                            else Qt.CheckState.Unchecked)

    def _count(self) -> None:
        self.count.setText(f"{len(self.selected())} terms selected")


class ConstraintsDialog(QDialog):
    """Multiple linear constraints, one per line, in actual units with factor letters."""

    def __init__(self, names: list[str], texts: list[str], parent=None):
        super().__init__(parent)
        from PySide6.QtWidgets import QPlainTextEdit

        self.setWindowTitle("Constraints")
        self.setStyleSheet(style.STYLESHEET)
        self.k = len(names)
        self.constraints: list[dict] = []
        legend = ", ".join(f"{designs.letter(j)} = {n}" for j, n in enumerate(names))
        self.edit = QPlainTextEdit("\n".join(texts))
        self.edit.setPlaceholderText("e.g.\nA + B <= 15\n0.2 <= A + 2B <= 1")
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.edit.textChanged.connect(self.check)
        ok, cancel = QPushButton("OK"), QPushButton("Cancel")
        ok.clicked.connect(self.apply)
        cancel.clicked.connect(self.reject)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(ok)
        row.addWidget(cancel)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(f"One linear constraint per line, in actual units ({legend}). "
                             "Use <=, >= or a double inequality."))
        lay.addWidget(self.edit, 1)
        lay.addWidget(self.status)
        lay.addLayout(row)
        self.resize(460, 320)
        self.check()

    def set_text(self, text: str) -> None:
        self.edit.setPlainText(text)

    def check(self) -> bool:
        from rsstatchem.core.doe import constraints as cons

        out = []
        for i, line in enumerate(self.edit.toPlainText().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                out.append(cons.parse_constraint(line, self.k))
            except ValueError as exc:
                self.status.setText(f"<span style='color:#B00020'>Line {i}: {exc}</span>")
                return False
        self.constraints = out
        self.status.setText(f"{len(out)} constraint(s)." if out else "No constraints.")
        return True

    def apply(self) -> None:
        if self.check():
            self.accept()


class RunsDialog(QDialog):
    """Forced runs or candidate points typed or pasted as rows of actual settings
    (categoric factors by level name)."""

    def __init__(self, title: str, factors: list[dict], rows: list, parent=None):
        super().__init__(parent)
        from PySide6.QtWidgets import QPlainTextEdit

        self.setWindowTitle(title)
        self.setStyleSheet(style.STYLESHEET)
        self.factors = factors
        self.rows: list[list[float]] = [list(r) for r in rows]
        self.edit = QPlainTextEdit("\n".join("\t".join(self._fmt(j, v)
                                                         for j, v in enumerate(r))
                                              for r in rows))
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.edit.textChanged.connect(self.check)
        ok, cancel = QPushButton("OK"), QPushButton("Cancel")
        ok.clicked.connect(self.apply)
        cancel.clicked.connect(self.reject)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(ok)
        row.addWidget(cancel)
        names = "  ".join(f"{designs.letter(j)}:{f.get('name')}" for j, f in enumerate(factors))
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(f"One run per line (tab, comma or space separated), actual "
                             f"units:  {names}"))
        lay.addWidget(self.edit, 1)
        lay.addWidget(self.status)
        lay.addLayout(row)
        self.resize(520, 340)
        self.check()

    def _levels(self, j):
        f = self.factors[j]
        codes, labels, _ = designs._levels_from_text(f.get("levels") or [])
        return codes, labels

    def _fmt(self, j, v) -> str:
        if self.factors[j].get("type") == "categoric":
            codes, labels = self._levels(j)
            for c, lab in zip(codes, labels):
                if abs(c - v) < 1e-9:
                    return lab
        return f"{v:g}"

    def set_text(self, text: str) -> None:
        self.edit.setPlainText(text)

    def check(self) -> bool:
        import re

        out = []
        k = len(self.factors)
        for i, line in enumerate(self.edit.toPlainText().splitlines(), start=1):
            if not line.strip():
                continue
            cells = [c for c in re.split(r"[\t;, ]+", line.strip()) if c]
            if len(cells) != k:
                self.status.setText(f"<span style='color:#B00020'>Line {i}: {len(cells)} "
                                    f"values for {k} factors.</span>")
                return False
            vals = []
            for j, c in enumerate(cells):
                if self.factors[j].get("type") == "categoric":
                    codes, labels = self._levels(j)
                    if c in labels:
                        vals.append(float(codes[labels.index(c)]))
                        continue
                try:
                    vals.append(float(c))
                except ValueError:
                    self.status.setText(f"<span style='color:#B00020'>Line {i}: cannot read "
                                        f"'{c}'.</span>")
                    return False
            out.append(vals)
        self.rows = out
        self.status.setText(f"{len(out)} run(s).")
        return True

    def apply(self) -> None:
        if self.check():
            self.accept()

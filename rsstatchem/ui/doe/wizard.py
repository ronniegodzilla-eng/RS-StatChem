"""RSM > Create Design: one window like Design-Expert's "New Design" builder.

Left: the design families (factorial, response surface, mixture). Right: number of factors, the
factor table (name, units, low, high / levels / mixture limits), design options, responses and
randomisation. A live summary shows the number of runs. OK builds the design Dataset (opened in
a new Data Editor) and prints the design summary with the alias structure to the Output Viewer.
"""
from __future__ import annotations

import random

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QDoubleSpinBox, QFormLayout, QGroupBox,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMessageBox, QPushButton, QSpinBox,
                               QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from rsstatchem.core.doe import designs, report

CATALOG = [
    ("Factorial", [("factorial", "Regular Two-Level"), ("general", "Multilevel Categoric"),
                   ("pb", "Plackett-Burman"), ("taguchi", "Taguchi OA")]),
    ("Response Surface", [("ccd", "Central Composite"), ("bbd", "Box-Behnken"),
                          ("threelevel", "3-Level Factorial"), ("dopt", "Optimal (D-optimal)")]),
    ("Mixture", [("lattice", "Simplex Lattice"), ("centroid", "Simplex Centroid"),
                 ("ev", "Extreme Vertices"), ("mixdopt", "Optimal (D-optimal)")]),
]
DESCRIPTIONS = {
    "factorial": "Two-level full or fractional factorial (2^k or 2^(k-p)) for screening and "
                 "characterisation. Choose a fraction; the generators and alias structure "
                 "are shown in the output.",
    "general": "Full factorial for categoric factors with any number of levels.",
    "pb": "Plackett-Burman screening design (runs a multiple of 4) for main effects only.",
    "taguchi": "Taguchi orthogonal array; analyse with RSM > Taguchi S/N Analysis.",
    "ccd": "Central composite design: 2-level factorial (or fraction) + axial points + "
           "centre points. The most popular design for fitting quadratic models.",
    "bbd": "Box-Behnken: three levels per factor, no corner points (3 to 7 factors).",
    "threelevel": "Full 3^k factorial with levels -1, 0, +1.",
    "dopt": "D-optimal design chosen by point exchange from a candidate set for a chosen "
            "model, plus lack-of-fit and replicate points.",
    "lattice": "Simplex lattice {q, m}: blends with proportions 0, 1/m, ..., 1.",
    "centroid": "Simplex centroid: pure blends, 50/50 binaries, ..., overall centroid.",
    "ev": "Extreme vertices of a constrained mixture region (lower/upper limits): vertices, "
          "edge centres, plane centroids, axial check blends and the overall centroid.",
    "mixdopt": "D-optimal mixture design chosen from the extreme-vertices candidate set.",
    "mr5": "Minimum-run equireplicated resolution V design (MR5): every main effect and "
           "two-factor interaction estimable in about as many runs as model terms.",
    "mr4": "Minimum-run resolution IV design (MR4): main effects clear of two-factor "
           "interactions in 2k runs (+2 recommended runs).",
}
LIMITS = {"factorial": (2, 15), "general": (1, 6), "pb": (2, 31), "taguchi": (1, 13),
          "ccd": (2, 10), "bbd": (3, 7), "threelevel": (1, 5), "dopt": (1, 8),
          "lattice": (2, 8), "centroid": (2, 8), "ev": (2, 8), "mixdopt": (2, 8),
          "mr5": (6, 50), "mr4": (5, 50)}
DEFAULT_K = {"factorial": 3, "general": 2, "pb": 7, "taguchi": 4, "ccd": 2, "bbd": 3,
             "threelevel": 2, "dopt": 3, "lattice": 3, "centroid": 3, "ev": 3, "mixdopt": 3,
             "mr5": 6, "mr4": 8}
START = {"factorial": "factorial", "rsm": "ccd", "mixture": "lattice", "taguchi": "taguchi",
         "optimal": "dopt"}


class DesignWizard(QDialog):
    def __init__(self, category: str = "rsm", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create Design")
        self.resize(980, 700)
        self.result = None
        self._building = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._update_summary)

        self.catalog = QListWidget()
        self.catalog.setMinimumWidth(190)
        self.catalog.setMaximumWidth(230)
        for group, items in CATALOG:
            head = QListWidgetItem(group)
            head.setFlags(Qt.ItemFlag.NoItemFlags)
            f = head.font()
            f.setBold(True)
            head.setFont(f)
            self.catalog.addItem(head)
            for key, text in items:
                it = QListWidgetItem("    " + text)
                it.setData(Qt.ItemDataRole.UserRole, key)
                self.catalog.addItem(it)
        self.catalog.currentItemChanged.connect(lambda *_: self._design_changed())

        self.title = QLabel()
        self.title.setStyleSheet("font-size: 15px; font-weight: bold; color: #8E1B1B;")
        self.desc = QLabel()
        self.desc.setWordWrap(True)
        self.desc.setStyleSheet("color: #555;")

        self.k_spin = QSpinBox()
        self.k_spin.valueChanged.connect(lambda _: self._resize_factors())
        self.k_label = QLabel("Numeric factors:")
        self.total = QDoubleSpinBox()
        self.total.setRange(0.0001, 1e6)
        self.total.setDecimals(4)
        self.total.setValue(1.0)
        self.total.valueChanged.connect(self._changed)
        self.total_label = QLabel("Mixture total:")
        top = QHBoxLayout()
        top.addWidget(self.k_label)
        top.addWidget(self.k_spin)
        top.addSpacing(16)
        top.addWidget(self.total_label)
        top.addWidget(self.total)
        top.addStretch()

        self.factors = QTableWidget(0, 4)
        self.factors.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.factors.itemChanged.connect(self._changed)
        self.factors.verticalHeader().setDefaultSectionSize(24)

        self.options_box = QGroupBox("Options")
        self.options_form = QFormLayout(self.options_box)

        self.r_spin = QSpinBox()
        self.r_spin.setRange(1, 30)
        self.r_spin.setValue(1)
        self.r_spin.valueChanged.connect(lambda _: self._resize_responses())
        self.responses = QTableWidget(0, 2)
        self.responses.setHorizontalHeaderLabels(["Name", "Units"])
        self.responses.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.responses.setMaximumHeight(84)
        self.responses.verticalHeader().setDefaultSectionSize(24)
        rbox = QGroupBox("Responses")
        rl = QVBoxLayout(rbox)
        row = QHBoxLayout()
        row.addWidget(QLabel("Number of responses:"))
        row.addWidget(self.r_spin)
        row.addStretch()
        rl.addLayout(row)
        rl.addWidget(self.responses)

        self.randomize = QCheckBox("Randomize run order")
        self.randomize.setChecked(True)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_000_000_000)
        self.seed.setValue(random.randint(1, 99999))
        new_seed = QPushButton("New Seed")
        new_seed.clicked.connect(lambda: self.seed.setValue(random.randint(1, 99999)))
        self.randomize.toggled.connect(self.seed.setEnabled)
        self.ds_name = QLineEdit("Design")
        rnd = QHBoxLayout()
        rnd.addWidget(self.randomize)
        rnd.addWidget(QLabel("Seed:"))
        rnd.addWidget(self.seed)
        rnd.addWidget(new_seed)
        rnd.addSpacing(16)
        rnd.addWidget(QLabel("Dataset name:"))
        rnd.addWidget(self.ds_name)
        rnd.addStretch()

        self.summary = QLabel()
        self.summary.setStyleSheet("font-weight: bold; padding: 4px; background: #FBE3E3; "
                                   "border: 1px solid #F0B5B3; border-radius: 4px;")
        self.summary.setWordWrap(True)

        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.addWidget(self.title)
        rlay.addWidget(self.desc)
        rlay.addLayout(top)
        rlay.addWidget(self.factors, 2)
        rlay.addWidget(self.options_box)
        rlay.addWidget(rbox)
        rlay.addLayout(rnd)
        rlay.addWidget(self.summary)

        split = QSplitter()
        split.addWidget(self.catalog)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        split.setCollapsible(0, False)
        split.setSizes([200, 780])
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Create Design")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(split, 1)
        layout.addWidget(buttons)

        self._opt: dict = {}
        self._factor_cache: dict[str, list[list[str]]] = {}
        self._resize_responses()
        start = START.get(category, category)
        for i in range(self.catalog.count()):
            if self.catalog.item(i).data(Qt.ItemDataRole.UserRole) == start:
                self.catalog.setCurrentRow(i)
                break

    # ------------------------------------------------------------------ state
    @property
    def design(self) -> str:
        it = self.catalog.currentItem()
        return it.data(Qt.ItemDataRole.UserRole) if it else "ccd"

    def _columns(self) -> list[str]:
        d = self.design
        if d in ("general", "taguchi"):
            return ["Name", "Units", "Levels (comma separated)"]
        if d in designs.MIXTURE_DESIGNS:
            return ["Name", "Units", "Low limit", "High limit"]
        if d == "ccd" and self._opt.get("ccd_type") == "inscribed":
            return ["Name", "Units", "Low (axial)", "High (axial)"]
        return ["Name", "Units", "Low (-1)", "High (+1)"]

    def _changed(self, *_) -> None:
        if not self._building:
            self._timer.start()

    def _design_changed(self) -> None:
        d = self.design
        if d is None:
            return
        self.title.setText(designs.DESIGN_TITLES[d])
        self.desc.setText(DESCRIPTIONS[d])
        mixture = d in designs.MIXTURE_DESIGNS
        self.k_label.setText("Components:" if mixture else "Factors:")
        self.total.setVisible(mixture)
        self.total_label.setVisible(mixture)
        lo, hi = LIMITS[d]
        self._building = True
        self.k_spin.setRange(lo, hi)
        self.k_spin.setValue(min(max(self.k_spin.value() if self.factors.rowCount() else
                                     DEFAULT_K[d], lo), hi))
        if d == "taguchi":
            self.k_spin.setValue(min(self.k_spin.value(), 4))
        self._building = False
        self._opt = {}
        self._build_options()
        self._resize_factors(force=True)

    def _factor_rows(self) -> list[list[str]]:
        rows = []
        for r in range(self.factors.rowCount()):
            rows.append([(self.factors.item(r, c).text() if self.factors.item(r, c) else "")
                         for c in range(self.factors.columnCount())])
        return rows

    def _default_row(self, r: int) -> list[str]:
        d = self.design
        name = designs.letter(r)
        if d == "general":
            return [name, "", "1, 2, 3" if r == 0 else "1, 2"]
        if d == "taguchi":
            arr = self._opt.get("array", "L9")
            nl = designs.TAGUCHI_ARRAYS[arr][1][r] if r < len(designs.TAGUCHI_ARRAYS[arr][1]) \
                else 2
            return [name, "", ", ".join(str(i + 1) for i in range(nl))]
        if d in designs.MIXTURE_DESIGNS:
            return [name, "", "0", f"{self.total.value():g}"]
        return [name, "", "-1", "1"]

    def _resize_factors(self, force: bool = False) -> None:
        d = self.design
        cols = self._columns()
        old = self._factor_rows()
        kind = "cat" if d in ("general", "taguchi") else "mix" if d in \
            designs.MIXTURE_DESIGNS else "num"
        if old and self.factors.property("kind"):
            self._factor_cache[self.factors.property("kind")] = old
        cache = self._factor_cache.get(kind, [])
        self._building = True
        self.factors.setColumnCount(len(cols))
        self.factors.setHorizontalHeaderLabels(cols)
        self.factors.setProperty("kind", kind)
        k = self.k_spin.value()
        self.factors.setRowCount(k)
        self.factors.setVerticalHeaderLabels([designs.letter(r) for r in range(k)])
        for r in range(k):
            values = cache[r] if r < len(cache) and len(cache[r]) == len(cols) else \
                self._default_row(r)
            if d == "taguchi":
                values = values[:2] + [self._default_row(r)[2]] if not (
                    r < len(cache) and len(cache[r]) == 3 and
                    len(cache[r][2].split(",")) == len(self._default_row(r)[2].split(","))) \
                    else values
            for c, v in enumerate(values):
                self.factors.setItem(r, c, QTableWidgetItem(v))
        hh = self.factors.horizontalHeader()
        for c in range(len(cols)):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.Stretch if c in (0, 2) and
                                    len(cols) == 3 or c == 0 else
                                    QHeaderView.ResizeMode.Interactive)
        self._building = False
        if self.design == "factorial" or self.design == "ccd":
            if self._opt.get("_k") != k:
                self._opt.pop("generators", None)
                self._opt.pop("ccd_fraction", None)
                self._opt["_k"] = k
            self._build_options()
        self._timer.start()

    def _resize_responses(self) -> None:
        n = self.r_spin.value()
        old = [[(self.responses.item(r, c).text() if self.responses.item(r, c) else "")
                for c in range(2)] for r in range(self.responses.rowCount())]
        self.responses.setRowCount(n)
        self.responses.setVerticalHeaderLabels([f"R{i + 1}" for i in range(n)])
        for r in range(n):
            vals = old[r] if r < len(old) else [f"R{r + 1}", ""]
            for c, v in enumerate(vals):
                self.responses.setItem(r, c, QTableWidgetItem(v))

    # ------------------------------------------------------------------ options
    def _clear_options(self) -> None:
        while self.options_form.rowCount():
            self.options_form.removeRow(0)

    def _combo(self, key: str, items: list[tuple[object, str]], default) -> QComboBox:
        box = QComboBox()
        for value, text in items:
            box.addItem(text, value)
        cur = self._opt.get(key, default)
        idx = max(box.findData(cur), 0)
        box.setCurrentIndex(idx)
        self._opt[key] = box.currentData()

        def changed(_):
            self._opt[key] = box.currentData()
            if key in ("fraction", "ccd_fraction", "array", "ccd_type", "alpha_type"):
                QTimer.singleShot(0, self._rebuild_after(key))
            self._changed()
        box.currentIndexChanged.connect(changed)
        return box

    def _rebuild_after(self, key):
        def run():
            if key in ("fraction", "ccd_fraction"):
                self._opt.pop("generators", None)
            self._build_options()
            if key in ("array", "ccd_type"):
                self._resize_factors()
        return run

    def _spin(self, key: str, default: int, lo: int = 0, hi: int = 1000) -> QSpinBox:
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setValue(int(self._opt.get(key, default)))
        self._opt[key] = s.value()
        s.valueChanged.connect(lambda v: (self._opt.__setitem__(key, v), self._changed()))
        return s

    def _dspin(self, key: str, default: float, lo: float, hi: float, dec: int = 4):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(dec)
        s.setValue(float(self._opt.get(key, default)))
        self._opt[key] = s.value()
        s.valueChanged.connect(lambda v: (self._opt.__setitem__(key, v), self._changed()))
        return s

    def _check(self, key: str, text: str, default: bool) -> QCheckBox:
        c = QCheckBox(text)
        c.setChecked(bool(self._opt.get(key, default)))
        self._opt[key] = c.isChecked()
        c.toggled.connect(lambda v: (self._opt.__setitem__(key, v), self._changed()))
        return c

    def _generators_edit(self, k: int, p: int) -> QLineEdit:
        edit = QLineEdit()
        try:
            default = designs.default_generators(k, p) if p else ""
        except ValueError:
            default = ""
        edit.setText(self._opt.get("generators") or default)
        self._opt["generators"] = edit.text()
        edit.setPlaceholderText("e.g. D=ABC, E=ABD (blank = full factorial)")
        edit.textChanged.connect(lambda t: (self._opt.__setitem__("generators", t),
                                            self._changed()))
        return edit

    def _fraction_items(self, k: int, min_res: int = 0) -> list[tuple[int, str]]:
        items = [(0, f"Full factorial (2^{k} = {2 ** k} runs)")]
        for p in range(1, k):
            if k - p < 2 or 2 ** (k - p) < k + 1:
                break
            try:
                gens = designs.default_generators(k, p)
                _, group = designs.fractional_factorial(k, gens)
            except ValueError:
                break
            res = designs.resolution(group)
            if res is not None and res < max(min_res, 3):
                break
            frac = f"1/{2 ** p}"
            items.append((p, f"{frac} fraction (2^({k}-{p}) = {2 ** (k - p)} runs, "
                             f"Res {designs.ROMAN.get(res, res)})"))
        return items

    def _build_options(self) -> None:
        self._building = True
        self._clear_options()
        d = self.design
        k = self.k_spin.value()
        f = self.options_form
        if d == "factorial":
            items = self._fraction_items(k)
            if self._opt.get("fraction", 0) not in [i for i, _ in items]:
                self._opt["fraction"] = 0
            f.addRow("Fraction:", self._combo("fraction", items, 0))
            f.addRow("Generators:", self._generators_edit(k, self._opt["fraction"]))
            nb = [1]
            base = k - self._opt["fraction"]
            while nb[-1] * 2 <= 2 ** (base - 1) and nb[-1] < 32:
                nb.append(nb[-1] * 2)
            f.addRow("Blocks:", self._combo("blocks", [(b, str(b)) for b in nb], 1))
            f.addRow("Center points per block:", self._spin("center_points", 0, 0, 50))
            f.addRow("Replicates:", self._spin("replicates", 1, 1, 20))
        elif d == "pb":
            f.addRow("Runs:", QLabel(f"{designs.pb_runs(k)} (smallest available multiple of 4 "
                                     f"for {k} factors)"))
            f.addRow("Center points:", self._spin("center_points", 0, 0, 50))
            f.addRow("Replicates:", self._spin("replicates", 1, 1, 20))
        elif d == "ccd":
            f.addRow("Type:", self._combo("ccd_type", [("circumscribed", "Circumscribed (CCC)"),
                                                        ("inscribed", "Inscribed (CCI)"),
                                                        ("face", "Face centered (CCF)")],
                                          "circumscribed"))
            if self._opt.get("ccd_type") != "face":
                f.addRow("Alpha:", self._combo("alpha_type", [
                    ("rotatable", "Rotatable (nF^¼)"), ("orthogonal", "Orthogonal quadratic"),
                    ("blocking", "Orthogonal blocking"), ("practical", "Practical (k^¼)"),
                    ("spherical", "Spherical (√k)"), ("custom", "Custom value")],
                    "rotatable"))
                if self._opt.get("alpha_type") == "custom":
                    f.addRow("Custom alpha:", self._dspin("alpha", 1.5, 0.1, 10, 4))
            items = self._fraction_items(k, min_res=5)
            if self._opt.get("ccd_fraction") not in [i for i, _ in items]:
                self._opt["ccd_fraction"] = designs.CCD_FRACTION.get(k, 0) \
                    if designs.CCD_FRACTION.get(k, 0) in [i for i, _ in items] else 0
            f.addRow("Factorial portion:", self._combo("ccd_fraction", items, 0))
            f.addRow("Generators:", self._generators_edit(k, self._opt["ccd_fraction"]))
            f.addRow("Blocks:", self._combo("blocks", [(1, "1"), (2, "2 (factorial | axial)"),
                                                       (3, "3 (2 factorial + axial)"),
                                                       (5, "5 (4 factorial + axial)")], 1))
            f.addRow("Center points (factorial / per factorial block):",
                     self._spin("center_points", designs.CCD_CENTERS.get(k, 6), 0, 50))
            f.addRow("Center points in axial block (blocked only):",
                     self._spin("center_axial", 2, 0, 50))
            f.addRow("Replicates:", self._spin("replicates", 1, 1, 10))
        elif d == "bbd":
            f.addRow("Center points:", self._spin("center_points", designs.BBD_CENTERS.get(k, 5),
                                                  0, 50))
            f.addRow("Replicates:", self._spin("replicates", 1, 1, 10))
        elif d == "threelevel":
            f.addRow("Extra center points:", self._spin("center_points", 0, 0, 50))
            f.addRow("Replicates:", self._spin("replicates", 1, 1, 10))
        elif d == "dopt":
            f.addRow("Model:", self._combo("model", [("linear", "Linear"), ("2fi", "2FI"),
                                                      ("quadratic", "Quadratic"),
                                                      ("cubic", "Cubic")], "quadratic"))
            f.addRow("Model points (0 = number of terms):", self._spin("model_points", 0, 0, 500))
            f.addRow("Lack-of-fit points:", self._spin("lof_points", 5, 0, 100))
            f.addRow("Replicate points:", self._spin("replicate_points", 5, 0, 100))
            f.addRow("Random starts:", self._spin("starts", 8, 1, 100))
        elif d in ("lattice", "centroid"):
            if d == "lattice":
                f.addRow("Degree (m):", self._spin("degree", 2, 1, 4))
            f.addRow("", self._check("augment", "Augment with axial check blends and the "
                                                "overall centroid", False))
            f.addRow("Replicates of the centroid:", self._spin("center_points", 0, 0, 20))
            f.addRow("", QLabel("Lower limits are honoured with L-pseudo components."))
        elif d == "ev":
            row = QHBoxLayout()
            inc = self._opt.setdefault("include", ["vertex", "edge", "centroid"])
            for key, text in (("vertex", "Vertices"), ("edge", "Edge centers"),
                              ("plane", "Plane centroids"), ("interior", "Axial check blends"),
                              ("centroid", "Overall centroid")):
                cb = QCheckBox(text)
                cb.setChecked(key in inc)
                cb.toggled.connect(lambda on, kk=key: self._toggle_include(kk, on))
                row.addWidget(cb)
            w = QWidget()
            w.setLayout(row)
            f.addRow("Points:", w)
            f.addRow("Replicates of the centroid:", self._spin("center_points", 0, 0, 20))
        elif d == "mixdopt":
            f.addRow("Model:", self._combo("model", [("linear", "Linear"),
                                                      ("quadratic", "Quadratic"),
                                                      ("special_cubic", "Special Cubic"),
                                                      ("cubic", "Cubic")], "quadratic"))
            f.addRow("Model points (0 = number of terms):", self._spin("model_points", 0, 0, 500))
            f.addRow("Lack-of-fit points:", self._spin("lof_points", 5, 0, 100))
            f.addRow("Replicate points:", self._spin("replicate_points", 5, 0, 100))
        elif d == "taguchi":
            items = []
            for name, (n, levels) in designs.TAGUCHI_ARRAYS.items():
                two, three = levels.count(2), levels.count(3)
                desc = " + ".join(t for t in (f"{two} two-level" if two else "",
                                              f"{three} three-level" if three else "") if t)
                items.append((name, f"{name}: {n} runs, {desc} columns"))
            f.addRow("Orthogonal array:", self._combo("array", items, "L9"))
            f.addRow("Replicates:", self._spin("replicates", 1, 1, 10))
            n_cols = len(designs.TAGUCHI_ARRAYS[self._opt["array"]][1])
            self.k_spin.setMaximum(n_cols)
        self._building = False
        self._timer.start()

    def _toggle_include(self, key: str, on: bool) -> None:
        inc = set(self._opt.get("include", []))
        inc.add(key) if on else inc.discard(key)
        self._opt["include"] = sorted(inc)
        self._changed()

    # ------------------------------------------------------------------ spec
    def spec(self) -> dict:
        d = self.design
        factors = []
        for row in self._factor_rows():
            fac = {"name": row[0].strip(), "units": row[1].strip()}
            if d in ("general", "taguchi"):
                fac["levels"] = [s.strip() for s in row[2].split(",") if s.strip()]
            else:
                try:
                    fac["low"] = float(row[2].replace(",", ".")) if row[2].strip() else 0.0
                    fac["high"] = float(row[3].replace(",", ".")) if row[3].strip() else \
                        (self.total.value() if d in designs.MIXTURE_DESIGNS else 1.0)
                except ValueError as exc:
                    raise ValueError(f"Factor {row[0]}: limits must be numbers.") from exc
            factors.append(fac)
        responses = [{"name": (self.responses.item(r, 0).text() if self.responses.item(r, 0)
                               else f"R{r + 1}").strip(),
                      "units": (self.responses.item(r, 1).text() if self.responses.item(r, 1)
                                else "").strip()}
                     for r in range(self.responses.rowCount())]
        opt = {k: v for k, v in self._opt.items() if not k.startswith("_")}
        if d == "ccd":
            p = opt.pop("ccd_fraction", 0)
            if not p:
                opt["generators"] = ""
            opt.pop("fraction", None)
            opt["center_points"] = opt.get("center_points")
        if d == "factorial" and not opt.get("fraction"):
            opt["generators"] = opt.get("generators", "") if opt.get("generators") else ""
        if d in ("dopt", "mixdopt") and not opt.get("model_points"):
            opt.pop("model_points", None)
        if d in designs.MIXTURE_DESIGNS:
            opt["total"] = self.total.value()
        return {"design": d, "factors": factors, "responses": responses, "options": opt,
                "randomize": self.randomize.isChecked(),
                "seed": self.seed.value() if self.randomize.isChecked() else None,
                "name": self.ds_name.text().strip() or "Design"}

    def _update_summary(self) -> None:
        try:
            spec = self.spec()
        except ValueError as exc:
            self.summary.setText(f"⚠ {exc}")
            return
        n = designs.design_run_count(spec)
        if isinstance(n, int):
            what = "components" if spec["design"] in designs.MIXTURE_DESIGNS else "factors"
            self.summary.setText(f"{designs.DESIGN_TITLES[spec['design']]}: {n} runs, "
                                 f"{len(spec['factors'])} {what}, "
                                 f"{len(spec['responses'])} response(s).")
        else:
            self.summary.setText(f"⚠ {n}")

    def accept(self) -> None:
        try:
            spec = self.spec()
            names = [f["name"] for f in spec["factors"]]
            if any(not n for n in names):
                raise ValueError("Every factor needs a name.")
            self.result = designs.build_design(spec)
        except ValueError as exc:
            QMessageBox.warning(self, self.windowTitle(), str(exc))
            return
        super().accept()


def open_wizard(editor, category: str = "rsm") -> None:
    dlg = DesignWizard(category, editor)
    if not dlg.exec() or dlg.result is None:
        return
    res = dlg.result
    res.dataset.modified = True
    reuse = editor if hasattr(editor, "is_pristine") and editor.is_pristine() else None
    if hasattr(editor.ctx, "show_dataset"):
        editor.ctx.show_dataset(res.dataset, reuse=reuse)
    else:
        editor.ctx.new_editor(res.dataset)
    editor.ctx.show_output(report.design_section(res.meta, res.notes, res.dataset))

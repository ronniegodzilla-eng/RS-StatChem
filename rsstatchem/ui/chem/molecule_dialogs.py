"""Chemistry > Molecule Properties and Chemistry > Insert Structure into Output dialogs.

Both accept SMILES, InChI or MOL/SDF text (or a file) and show a live 2D depiction. RDKit is
imported only when a structure is drawn, so opening the app stays fast.
"""
from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
                               QLabel, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox,
                               QVBoxLayout)

from rsstatchem.core.chem import sections
from rsstatchem.ui.chem.widgets import (EXAMPLE_MOLECULES, CalculatorDialog,
                                        StructurePreview, line)

FILE_FILTER = ("Structure files (*.mol *.sdf *.sd *.smi *.mol2 *.pdb *.txt);;"
               "All files (*)")


def parse_structures(text: str) -> list[tuple[str, object]]:
    """(name, Mol) list from the input box: one MOL/SDF/InChI block, or SMILES lines."""
    from rsstatchem.core.chem import molecule as mo
    t = text.strip()
    if not t:
        raise ValueError("Enter a structure (SMILES, InChI or MOL block).")
    fmt = mo.detect_format(t)
    if fmt == "reaction":
        raise ValueError("This is a reaction SMILES; use Insert Structure into Output.")
    if fmt == "sdf":
        out = []
        for block in t.split("$$$$"):
            if block.strip():
                m = mo.mol_from_text(block + "\n", "molblock")
                out.append((m.GetProp("_Name") if m.HasProp("_Name") else "", m))
        return out
    if fmt in ("molblock", "inchi"):
        m = mo.mol_from_text(t, fmt)
        return [(m.GetProp("_Name") if m.HasProp("_Name") else "", m)]
    out = []
    for i, raw in enumerate(t.splitlines()):
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        try:
            m = mo.mol_from_text(raw, "smiles")
        except mo.ChemError as exc:
            raise ValueError(f"Line {i + 1}: {exc}") from None
        out.append((m.GetProp("_Name") if m.HasProp("_Name") else "", m))
    return out


class _StructureInput:
    """Mixin: input box, example menu, Open File… and live preview."""

    def _build_input(self, label: str, preview_size=(300, 225)) -> QHBoxLayout:
        top = QHBoxLayout()
        left = QVBoxLayout()
        lbl = QLabel(label)
        lbl.setWordWrap(True)
        left.addWidget(lbl)
        self.input = QPlainTextEdit()
        self.input.setProperty("default_value", EXAMPLE_MOLECULES[0][1])
        self.input.setPlainText(EXAMPLE_MOLECULES[0][1])
        self.input.textChanged.connect(self.schedule)
        self.remember("input", self.input)
        left.addWidget(self.input, 1)
        row = QHBoxLayout()
        self.examples = QComboBox()
        self.examples.addItem("Examples…")
        for name, smi in EXAMPLE_MOLECULES:
            self.examples.addItem(name, smi)
        self.examples.activated.connect(self._example)
        open_btn = QPushButton("Open File...")
        open_btn.setAutoDefault(False)
        open_btn.clicked.connect(self._open_file)
        row.addWidget(self.examples)
        row.addWidget(open_btn)
        row.addStretch()
        left.addLayout(row)
        top.addLayout(left, 1)
        self.preview_widget = StructurePreview(*preview_size)
        top.addWidget(self.preview_widget, 0, Qt.AlignmentFlag.AlignTop)
        return top

    def _example(self, index: int) -> None:
        if index > 0:
            self.input.setPlainText(self.examples.itemData(index))
            self.examples.setCurrentIndex(0)

    def _open_file(self) -> None:
        from rsstatchem.core.chem import molecule as mo
        settings = getattr(getattr(self.parent(), "ctx", None), "settings", None)
        start = settings.value("last_dir", "") if settings is not None else ""
        path, _ = QFileDialog.getOpenFileName(self, "Open Structure", start, FILE_FILTER)
        if not path:
            return
        try:
            mols = mo.read_file(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, self.windowTitle(), str(exc))
            return
        if settings is not None:
            settings.setValue("last_dir", os.path.dirname(path))
        if len(mols) == 1 and os.path.splitext(path)[1].lower() in (".mol", ".sdf", ".sd"):
            m = mols[0]
            if not m.HasProp("_Name") or not m.GetProp("_Name").strip():
                m.SetProp("_Name", os.path.splitext(os.path.basename(path))[0])
            self.input.setPlainText(mo.to_molblock(m))
        else:
            lines = []
            for m in mols[:500]:
                name = m.GetProp("_Name") if m.HasProp("_Name") else ""
                lines.append(f"{mo.to_smiles(m)} {name}".rstrip())
            self.input.setPlainText("\n".join(lines))


class MoleculePropertiesDialog(_StructureInput, CalculatorDialog):
    """Identifiers, RDKit descriptors and drug-likeness rules for one or more molecules."""

    def __init__(self, ds=None, parent=None):
        CalculatorDialog.__init__(self, ds, parent, "Molecule Properties")
        self.setMinimumSize(720, 470)
        self.body.addLayout(self._build_input(
            "Structure: SMILES (one per line, optional name after a space), InChI or MOL "
            "block:"))
        opts = QHBoxLayout()
        box = QGroupBox("Display")
        lay = QVBoxLayout(box)
        self.checks = {}
        for key, text, on in (("image", "Structure image", True),
                              ("ids", "Identifiers (SMILES, InChIKey)", True),
                              ("desc", "Properties / descriptors", True)):
            cb = QCheckBox(text)
            cb.setChecked(on)
            cb.setProperty("default_value", on)
            self.remember(key, cb)
            self.checks[key] = cb
            lay.addWidget(cb)
        opts.addWidget(box)
        box = QGroupBox("Drug-likeness Rules")
        lay = QVBoxLayout(box)
        for key, text, on in (("Lipinski", "Lipinski Rule of Five", True),
                              ("Veber", "Veber (rotatable bonds, TPSA)", True),
                              ("Ghose", "Ghose filter", False),
                              ("Egan", "Egan (cLogP, TPSA)", False)):
            cb = QCheckBox(text)
            cb.setChecked(on)
            cb.setProperty("default_value", on)
            self.remember(key, cb)
            self.checks[key] = cb
            lay.addWidget(cb)
        opts.addWidget(box)
        self.summary = QLabel()
        self.summary.setTextFormat(Qt.TextFormat.RichText)
        self.summary.setWordWrap(True)
        self.summary.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.summary.setMinimumWidth(220)
        opts.addWidget(self.summary, 1)
        self.body.addLayout(opts)
        self.preview()

    def preview(self) -> None:
        text = self.input.toPlainText()
        first = text.strip().splitlines()[0] if text.strip() else ""
        from rsstatchem.core.chem import molecule as mo
        fmt = mo.detect_format(text) if text.strip() else "smiles"
        shown = text if fmt in ("molblock", "sdf", "inchi", "reaction") else first
        self.preview_widget.show_text(shown.split("$$$$")[0])
        try:
            items = parse_structures(text)
        except (ValueError, mo.ChemError) as exc:
            self.summary.setText(f"<span style='color:#b00020'>{exc}</span>")
            return
        from rsstatchem.core.chem import formula as fm
        p = mo.properties(items[0][1], ["MW", "cLogP", "TPSA"])
        txt = (f"<b>{fm.format_formula(p['Formula'], 'html')}</b><br>"
               f"MW {p['MW']:.2f} · cLogP {p['cLogP']:.2f} · TPSA {p['TPSA']:.1f}")
        if len(items) > 1:
            txt = f"{len(items)} molecules<br>First: " + txt
        self.summary.setText(txt)

    def build(self):
        from rsstatchem.core.chem import molecule as mo
        try:
            items = parse_structures(self.input.toPlainText())
        except mo.ChemError as exc:
            raise ValueError(str(exc)) from None
        rules = [k for k in ("Lipinski", "Veber", "Ghose", "Egan") if self.checks[k].isChecked()]
        return sections.molecule_section(items, self.checks["ids"].isChecked(),
                                         self.checks["desc"].isChecked(), rules,
                                         self.checks["image"].isChecked())


class InsertStructureDialog(_StructureInput, CalculatorDialog):
    """Draw a molecule or reaction SMILES as an image in the Output Viewer (or save it)."""

    def __init__(self, ds=None, parent=None):
        CalculatorDialog.__init__(self, ds, parent, "Insert Structure into Output")
        self.setMinimumSize(780, 560)
        self.body.addLayout(self._build_input(
            "Structure: SMILES, InChI, MOL block, or reaction SMILES (A.B>>C):", (360, 270)))
        box = QGroupBox("Drawing Options")
        grid = QGridLayout(box)
        self.width_spin = self._spin(100, 2000, 400, "width")
        self.height_spin = self._spin(80, 2000, 300, "height")
        self.bond = QDoubleSpinBox()
        self.bond.setRange(0.5, 8)
        self.bond.setSingleStep(0.5)
        self.bond.setValue(2.0)
        self.bond.setProperty("default_value", 2.0)
        self.remember("bond", self.bond)
        self.bond.valueChanged.connect(self.schedule)
        form = QFormLayout()
        form.addRow("Width (px):", self.width_spin)
        form.addRow("Height (px):", self.height_spin)
        form.addRow("Bond width:", self.bond)
        grid.addLayout(form, 0, 0, 3, 1)
        self.flags = {}
        flags = [("transparent", "Transparent background", False),
                 ("black_white", "Black and white", False),
                 ("atom_indices", "Atom numbers", False),
                 ("stereo_labels", "Stereo labels (R/S, E/Z)", False),
                 ("show_hydrogens", "Show all hydrogens", False),
                 ("explicit_methyl", "Show CH₃ labels", False),
                 ("clean", "Clean 2D coordinates", False)]
        for i, (key, text, on) in enumerate(flags):
            cb = QCheckBox(text)
            cb.setChecked(on)
            cb.setProperty("default_value", on)
            cb.toggled.connect(self.schedule)
            self.remember(key, cb)
            self.flags[key] = cb
            grid.addWidget(cb, i % 4, 1 + i // 4)
        form2 = QFormLayout()
        self.highlight = line("", "SMARTS, e.g. C(=O)O")
        self.highlight.textChanged.connect(self.schedule)
        self.remember("highlight", self.highlight)
        hl_row = QHBoxLayout()
        hl_row.addWidget(self.highlight, 1)
        self.color_btn = QPushButton("Colour...")
        self.color_btn.setAutoDefault(False)
        self.color_btn.clicked.connect(self._pick_color)
        hl_row.addWidget(self.color_btn)
        self.caption = line("", "shown under the structure")
        self.caption.textChanged.connect(self.schedule)
        self.remember("caption", self.caption)
        form2.addRow("Highlight:", hl_row)
        form2.addRow("Caption:", self.caption)
        grid.addLayout(form2, 4, 0, 1, 3)
        self.body.addWidget(box)
        save = self.button_box.addButton("Save Image...", self.button_box.ButtonRole.ActionRole)
        save.setAutoDefault(False)
        save.clicked.connect(self._save_image)
        self._color = "#FF9E9E"
        self.preview()

    def _spin(self, lo, hi, value, key) -> QSpinBox:
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(20)
        s.setValue(value)
        s.setProperty("default_value", value)
        s.valueChanged.connect(self.schedule)
        self.remember(key, s)
        return s

    def _pick_color(self) -> None:
        c = QColorDialog.getColor(QColor(self._color), self, "Highlight Colour")
        if c.isValid():
            self._color = c.name()
            self.schedule()

    def options(self):
        from rsstatchem.core.chem import depict
        kw = {k: cb.isChecked() for k, cb in self.flags.items()}
        return depict.DrawOptions(width=self.width_spin.value(), height=self.height_spin.value(),
                                  bond_width=self.bond.value(),
                                  highlight_smarts=self.highlight.text().strip(),
                                  highlight_color=self._color, legend=self.caption.text(), **kw)

    def preview(self) -> None:
        o = self.options()
        # keep the preview's aspect ratio equal to the output image
        scale = min(self.preview_widget.w / o.width, self.preview_widget.h / o.height, 1.0)
        prev = o.updated(bond_width=max(1.0, o.bond_width * scale))
        text = self.input.toPlainText()
        self.preview_widget.show_text(text, prev)

    def build(self):
        from rsstatchem.core.chem import molecule as mo
        text = self.input.toPlainText().strip()
        if not text:
            raise ValueError("Enter a structure.")
        try:
            return sections.structure_section(text, self.options(),
                                              self.caption.text().strip() or "Structure")
        except mo.ChemError as exc:
            raise ValueError(str(exc)) from None

    def _save_image(self) -> None:
        from rsstatchem.core.chem import depict, molecule as mo
        path, _ = QFileDialog.getSaveFileName(self, "Save Structure Image", "structure.png",
                                              "PNG image (*.png);;SVG image (*.svg)")
        if not path:
            return
        text = self.input.toPlainText().strip()
        o = self.options()
        try:
            is_rxn = mo.detect_format(text) == "reaction"
            if path.lower().endswith(".svg"):
                data = (depict.depict_reaction_svg(text, o.width, o.height, o) if is_rxn else
                        depict.depict_svg(mo.mol_from_text(text), options=o))
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(data)
            else:
                data = depict.depict_text_png(text, o, scale=3.0)
                with open(path, "wb") as fh:
                    fh.write(data)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, self.windowTitle(), str(exc))

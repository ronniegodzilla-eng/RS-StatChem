"""GenericDialog: the Qt dialog for a command's :class:`~rsstatchem.core.imaging.commands.Field`
list (ImageJ ``ij.gui.GenericDialog``). The values it returns use the same keys as macro
option strings, so ``registry.run(session, label, values=dlg.values())`` records exactly what
ImageJ would.

Preview: pass ``preview=callable(values)`` to add ImageJ's "Preview" checkbox; it is called with
the current values whenever a field changes while Preview is on (the command shows the result
on the image and restores it when the dialog is cancelled).
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QRadioButton, QSlider, QVBoxLayout,
                               QWidget)

from rsstatchem.core.imaging.commands import Field, format_number, macro_key


class GenericDialog(QDialog):
    def __init__(self, title: str, fields: list[Field], parent=None,
                 preview: Callable[[dict], None] | None = None, help_text: str = ""):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.fields = list(fields)
        self._getters: dict[str, Callable[[], object]] = {}
        self._preview = preview
        lay = QVBoxLayout(self)
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        lay.addLayout(grid)
        row = 0
        for f in self.fields:
            row = self._add_field(grid, row, f)
        self.preview_box = None
        if preview is not None:
            self.preview_box = QCheckBox("Preview")
            self.preview_box.toggled.connect(lambda _=False: self._changed())
            lay.addWidget(self.preview_box)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                   QDialogButtonBox.StandardButton.Cancel)
        if help_text:
            hb = buttons.addButton("Help", QDialogButtonBox.ButtonRole.HelpRole)
            hb.clicked.connect(lambda: QLabel(help_text).show())
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    # ------------------------------------------------------------------ building
    def _add_field(self, grid: QGridLayout, row: int, f: Field) -> int:
        label = f.label if f.label.endswith(":") or not f.label else f.label + ":"
        if f.kind in ("number", "slider"):
            spin = QDoubleSpinBox()
            spin.setDecimals(max(f.decimals, 0) if not f.integer else 0)
            spin.setRange(f.minimum if f.minimum is not None else -1e12,
                          f.maximum if f.maximum is not None else 1e12)
            spin.setValue(float(f.default))
            spin.setObjectName(f.key)
            spin.valueChanged.connect(lambda _=0: self._changed())
            grid.addWidget(QLabel(label), row, 0, Qt.AlignmentFlag.AlignRight)
            if f.kind == "slider":
                slider = QSlider(Qt.Orientation.Horizontal)
                scale = 10 ** max(f.decimals, 0)
                slider.setRange(int(f.minimum * scale), int(f.maximum * scale))
                slider.setValue(int(float(f.default) * scale))
                slider.valueChanged.connect(lambda v, s=spin, k=scale: s.setValue(v / k))
                spin.valueChanged.connect(lambda v, sl=slider, k=scale: sl.setValue(int(v * k)))
                box = QHBoxLayout()
                box.addWidget(slider, 1)
                box.addWidget(spin)
                grid.addLayout(box, row, 1)
            else:
                box = QHBoxLayout()
                box.addWidget(spin)
                if f.units:
                    box.addWidget(QLabel(f.units))
                box.addStretch(1)
                grid.addLayout(box, row, 1)
            self._getters[f.key] = (lambda s=spin, i=f.integer:
                                    int(round(s.value())) if i else float(s.value()))
            return row + 1
        if f.kind == "checkbox":
            cb = QCheckBox(f.label)
            cb.setObjectName(f.key)
            cb.setChecked(bool(f.default))
            cb.toggled.connect(lambda _=False: self._changed())
            grid.addWidget(cb, row, 1)
            self._getters[f.key] = cb.isChecked
            return row + 1
        if f.kind == "checkbox_group":
            holder = QWidget()
            g = QGridLayout(holder)
            g.setContentsMargins(0, 0, 0, 0)
            cols = f.columns or 1
            for i, (item, default) in enumerate(zip(f.items, f.default)):
                cb = QCheckBox(item)
                cb.setChecked(bool(default))
                cb.toggled.connect(lambda _=False: self._changed())
                g.addWidget(cb, i // cols, i % cols)
                self._getters[macro_key(item)] = cb.isChecked
            grid.addWidget(holder, row, 0, 1, 2)
            return row + 1
        if f.kind == "choice":
            combo = QComboBox()
            combo.setObjectName(f.key)
            combo.addItems(list(f.items))
            if f.default in f.items:
                combo.setCurrentText(f.default)
            combo.currentIndexChanged.connect(lambda _=0: self._changed())
            grid.addWidget(QLabel(label), row, 0, Qt.AlignmentFlag.AlignRight)
            grid.addWidget(combo, row, 1)
            self._getters[f.key] = combo.currentText
            return row + 1
        if f.kind == "radio":
            holder = QWidget()
            box = QGridLayout(holder)
            box.setContentsMargins(0, 0, 0, 0)
            group = QButtonGroup(holder)
            cols = f.columns or len(f.items)
            for i, item in enumerate(f.items):
                rb = QRadioButton(item)
                rb.setChecked(item == f.default)
                rb.toggled.connect(lambda _=False: self._changed())
                group.addButton(rb)
                box.addWidget(rb, i // cols, i % cols)
            grid.addWidget(QLabel(label), row, 0, Qt.AlignmentFlag.AlignRight)
            grid.addWidget(holder, row, 1)
            self._getters[f.key] = (lambda g=group, d=f.default:
                                    g.checkedButton().text() if g.checkedButton() else d)
            return row + 1
        if f.kind in ("string", "file", "directory"):
            edit = QLineEdit(str(f.default))
            edit.setObjectName(f.key)
            edit.textChanged.connect(lambda _="": self._changed())
            grid.addWidget(QLabel(label), row, 0, Qt.AlignmentFlag.AlignRight)
            if f.kind == "string":
                grid.addWidget(edit, row, 1)
            else:
                box = QHBoxLayout()
                box.addWidget(edit, 1)
                browse = QPushButton("Browse...")
                browse.clicked.connect(lambda _=False, e=edit, k=f.kind: self._browse(e, k))
                box.addWidget(browse)
                grid.addLayout(box, row, 1)
            self._getters[f.key] = edit.text
            return row + 1
        # message
        msg = QLabel(f.label)
        msg.setWordWrap(True)
        grid.addWidget(msg, row, 0, 1, 2)
        return row + 1

    def _browse(self, edit: QLineEdit, kind: str) -> None:
        if kind == "directory":
            path = QFileDialog.getExistingDirectory(self, "Choose a Directory", edit.text())
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Choose a File", edit.text())
        if path:
            edit.setText(path)

    def _changed(self) -> None:
        if self._preview is not None and self.preview_box is not None and \
                self.preview_box.isChecked():
            self._preview(self.values())

    # ------------------------------------------------------------------ results
    def values(self) -> dict:
        """Typed values keyed like macro options (checkbox groups: one key per box)."""
        out = {key: get() for key, get in self._getters.items()}
        for f in self.fields:
            if f.kind == "checkbox_group":
                out[f.key or "group"] = tuple(out[macro_key(i)] for i in f.items)
        return out

    def set_value(self, key: str, value) -> None:
        """Set a field programmatically (tests, Repeat Command)."""
        w = self.findChild(QWidget, key)
        if isinstance(w, QDoubleSpinBox):
            w.setValue(float(value))
        elif isinstance(w, QCheckBox):
            w.setChecked(bool(value))
        elif isinstance(w, QComboBox):
            w.setCurrentText(str(value))
        elif isinstance(w, QLineEdit):
            w.setText(format_number(value) if isinstance(value, (int, float)) else str(value))
        else:
            raise KeyError(key)

    def get(self) -> dict | None:
        """Run modally; the values, or None when cancelled."""
        return self.values() if self.exec() == QDialog.DialogCode.Accepted else None

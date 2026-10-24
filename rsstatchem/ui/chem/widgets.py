"""Shared Qt pieces for the Chemistry dialogs: calculator dialog base, structure preview,
number fields and formula fields with live molar mass."""
from __future__ import annotations

import math

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QVBoxLayout, QWidget)

from rsstatchem.core.chem import formula as fm

EXAMPLE_MOLECULES = [
    ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("Caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
    ("Paracetamol", "CC(=O)Nc1ccc(O)cc1"),
    ("(S)-Ibuprofen", "CC(C)Cc1ccc(cc1)[C@H](C)C(=O)O"),
    ("Quercetin", "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12"),
    ("β-D-Glucose", "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O"),
    ("Curcumin", "COc1cc(/C=C/C(=O)CC(=O)/C=C/c2ccc(O)c(OC)c2)ccc1O"),
    ("Esterification (reaction)", "CC(=O)O.OCC>>CC(=O)OCC.O"),
]


def parse_number(text: str) -> float | None:
    """Float from user text ('1,5', '1.8e-5', '1.8×10^-5', '2.5 %'); None when blank."""
    t = (text or "").strip().replace(" ", "").replace("%", "")
    if not t:
        return None
    t = t.replace("×10^", "e").replace("x10^", "e").replace("*10^", "e").replace("−", "-")
    if t.count(",") == 1 and "." not in t:
        t = t.replace(",", ".")
    try:
        return float(t)
    except ValueError:
        raise ValueError(f"'{text}' is not a number") from None


class CalculatorDialog(QDialog):
    """Base for the data-independent calculators: OK builds the Section (errors keep the
    dialog open), Reset restores defaults; field values are remembered for the session."""

    needs_data = False
    _memory: dict[str, dict] = {}

    def __init__(self, ds, parent=None, title: str = ""):
        super().__init__(parent)
        self.ds = ds
        self.setWindowTitle(title)
        self._section = None
        self._remembered: dict[str, QWidget] = {}
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._refresh)
        self.body = QVBoxLayout()
        buttons = QDialogButtonBox()
        self.ok_button = buttons.addButton(QDialogButtonBox.StandardButton.Ok)
        reset = buttons.addButton(QDialogButtonBox.StandardButton.Reset)
        cancel = buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        reset.setAutoDefault(False)
        cancel.setAutoDefault(False)
        self.ok_button.setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        reset.clicked.connect(self.reset)
        self.button_box = buttons
        layout = QVBoxLayout(self)
        layout.addLayout(self.body, 1)
        layout.addWidget(buttons)

    # -- live preview ------------------------------------------------------------------
    def schedule(self, *_args) -> None:
        self._timer.start()

    def _refresh(self) -> None:
        try:
            self.preview()
        except Exception:  # noqa: BLE001 - previews never raise into the event loop
            pass

    def preview(self) -> None:
        """Update live result widgets (override)."""

    # -- memory ------------------------------------------------------------------------
    def remember(self, key: str, widget: QWidget) -> QWidget:
        self._remembered[key] = widget
        saved = self._memory.get(type(self).__name__, {})
        if key in saved:
            _set_value(widget, saved[key])
        return widget

    def _store(self) -> None:
        self._memory[type(self).__name__] = {k: _get_value(w)
                                             for k, w in self._remembered.items()}

    def reset(self) -> None:
        self._memory.pop(type(self).__name__, None)
        for w in self._remembered.values():
            default = w.property("default_value")
            if default is not None:
                _set_value(w, default)
        self.schedule()

    # -- protocol ----------------------------------------------------------------------
    def build(self):
        """Return the Section (override); raise ValueError with a message for bad input."""
        raise NotImplementedError

    def accept(self) -> None:
        try:
            self._section = self.build()
        except (ValueError, KeyError, ZeroDivisionError, OverflowError) as exc:
            QMessageBox.warning(self, self.windowTitle(), str(exc).strip("'\""))
            return
        self._store()
        super().accept()

    def compute(self):
        return self._section if self._section is not None else self.build()


def _get_value(w):
    if isinstance(w, QLineEdit):
        return w.text()
    if isinstance(w, QComboBox):
        return w.currentText()
    if hasattr(w, "toPlainText"):
        return w.toPlainText()
    if hasattr(w, "isChecked"):
        return w.isChecked()
    if hasattr(w, "value"):
        return w.value()
    if hasattr(w, "currentIndex"):
        return w.currentIndex()
    return None


def _set_value(w, v) -> None:
    if isinstance(w, QLineEdit):
        w.setText(str(v))
    elif isinstance(w, QComboBox):
        i = w.findText(str(v))
        if i >= 0:
            w.setCurrentIndex(i)
    elif hasattr(w, "setPlainText"):
        w.setPlainText(str(v))
    elif hasattr(w, "setChecked"):
        w.setChecked(bool(v))
    elif hasattr(w, "setValue"):
        w.setValue(v)
    elif hasattr(w, "setCurrentIndex"):
        w.setCurrentIndex(int(v))


def line(default: str = "", placeholder: str = "", width: int | None = None) -> QLineEdit:
    e = QLineEdit(default)
    e.setProperty("default_value", default)
    if placeholder:
        e.setPlaceholderText(placeholder)
    if width:
        e.setFixedWidth(width)
    return e


def combo(items: list[str], current: str | None = None, width: int | None = None) -> QComboBox:
    c = QComboBox()
    c.addItems(items)
    if current:
        c.setCurrentText(current)
    c.setProperty("default_value", c.currentText())
    if width:
        c.setMinimumWidth(width)
    return c


class FormulaField(QWidget):
    """Formula line edit with live formatted formula and molar mass; ``molar_mass()``
    returns the typed number if the text is a number instead of a formula."""

    changed = Signal()

    def __init__(self, default: str = "", placeholder: str = "e.g. NaCl or CuSO4·5H2O",
                 parent=None):
        super().__init__(parent)
        self.edit = line(default, placeholder)
        self.info = QLabel()
        self.info.setTextFormat(Qt.TextFormat.RichText)
        self.info.setMinimumWidth(170)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.edit, 1)
        lay.addWidget(self.info)
        self.edit.textChanged.connect(self._update)
        self._update()

    def text(self) -> str:
        return self.edit.text().strip()

    def molar_mass(self) -> float | None:
        t = self.text()
        if not t:
            return None
        try:
            v = parse_number(t)
            return v
        except ValueError:
            pass
        return fm.parse_formula(t).molar_mass

    def display(self) -> str:
        t = self.text()
        try:
            parse_number(t)
            return t + " g/mol"
        except ValueError:
            return fm.format_formula(t, "unicode")

    def _update(self) -> None:
        t = self.text()
        if not t:
            self.info.setText("<i>M = –</i>")
        else:
            try:
                m = self.molar_mass()
                try:
                    parse_number(t)
                    self.info.setText(f"M = {m:g} g/mol")
                except ValueError:
                    self.info.setText(f"{fm.format_formula(t, 'html')}: "
                                      f"<b>{m:.3f}</b> g/mol")
            except (fm.FormulaError, ValueError) as exc:
                self.info.setText(f"<span style='color:#b00020'>{exc}</span>")
        self.changed.emit()


class StructurePreview(QLabel):
    """Shows a 2D depiction (PNG at 2× for sharpness) or an error message."""

    def __init__(self, width: int = 280, height: int = 210, parent=None):
        super().__init__(parent)
        self.w, self.h = width, height
        self.setFixedSize(width + 4, height + 4)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setStyleSheet("QLabel { background: white; border: 1px solid #c8c8c8; }")
        self.error = ""

    def show_png(self, png: bytes) -> None:
        pix = QPixmap()
        pix.loadFromData(png, "PNG")
        ratio = max(pix.width() / self.w, 1.0)
        pix.setDevicePixelRatio(ratio)
        self.setPixmap(pix)
        self.error = ""

    def show_message(self, text: str, error: bool = True) -> None:
        self.setPixmap(QPixmap())
        color = "#b00020" if error else "#777777"
        self.setText(f"<span style='color:{color}'>{text}</span>")
        self.error = text if error else ""

    def show_text(self, text: str, options=None) -> bool:
        """Depict SMILES/InChI/MOL/reaction text; returns False on error."""
        if not text.strip():
            self.show_message("Enter a structure (SMILES, InChI or MOL block)", error=False)
            return False
        try:
            from rsstatchem.core.chem import depict
            opts = options or depict.DrawOptions()
            opts = opts.updated(width=self.w, height=self.h)
            self.show_png(depict.depict_text_png(text, opts, scale=2.0))
            return True
        except Exception as exc:  # noqa: BLE001
            self.show_message(str(exc))
            return False


def fmt_value(x: float, digits: int = 4) -> str:
    from rsstatchem.core.chem.sections import sig
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "–"
    return sig(x, digits)

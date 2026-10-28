"""Fitting Function Organizer (Origin's Tools > Fitting Function Organizer / Builder).

User functions are kept per user as JSON in ``<RSSTATCHEM_HOME>/fitting_functions.json``
(the per-user data folder, see :mod:`rsstatchem.paths`): name, category, type (curve
``y = f(x)`` or surface ``z = f(x, y)``), equation, initial values and description. They are
compiled with the safe expression compiler (exact Jacobians, no ``eval``) and registered in
the function library, so they appear in NLFit under their category.
"""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout,
                               QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from rsstatchem.core.analysis import functions as fl

STORE = "fitting_functions.json"
DEFAULT_CATEGORY = "My Functions"


def store_path() -> Path:
    from rsstatchem.paths import user_data_dir
    return Path(user_data_dir()) / STORE


def load_entries() -> list[dict]:
    p = store_path()
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [d for d in data.get("functions", []) if isinstance(d, dict) and d.get("name")]


def save_entries(entries: list[dict]) -> None:
    p = store_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"version": 1, "functions": entries}, indent=2, ensure_ascii=False),
                 encoding="utf-8")


def parse_initial(text: str) -> dict[str, float]:
    """``"a = 1, b = 2.5"`` → {"a": 1.0, "b": 2.5}."""
    out = {}
    for part in (text or "").replace(";", ",").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            k, v = k.strip(), v.strip().replace(" ", "")
            if k:
                try:
                    out[k] = float(v)
                except ValueError:
                    raise ValueError(f"Initial value of {k}: '{v}' is not a number.") from None
    return out


def make_function(entry: dict) -> fl.FitFunction:
    surface = entry.get("type") == "surface"
    f = fl.parse_function(entry["equation"], ("x", "y") if surface else ("x",),
                          name=entry["name"], category=entry.get("category") or DEFAULT_CATEGORY,
                          description=entry.get("description") or "User function.")
    f.user_defined = True
    p0 = {k: v for k, v in (entry.get("initial") or {}).items() if k in f.params}
    unknown = set(entry.get("initial") or {}) - set(f.params)
    if unknown:
        raise ValueError(f"Initial values for unknown parameter(s): {', '.join(sorted(unknown))}")
    if p0:
        f.meta["guess"] = lambda x, y, p0=p0, names=f.params: {k: p0.get(k, 1.0) for k in names}
        f.meta["initial"] = p0
    f.meta["organizer"] = True
    return f


def register_all() -> list[str]:
    """Register every stored user function (curves only go into the library); returns the
    names that failed to compile."""
    bad = []
    for e in load_entries():
        try:
            f = make_function(e)
        except ValueError:
            bad.append(e.get("name", "?"))
            continue
        if f.nvar == 1:
            fl.register(f)
    return bad


def upsert(entry: dict) -> fl.FitFunction:
    """Validate, store (replacing a function of the same name and category) and register."""
    entry = dict(entry)
    entry["name"] = entry["name"].strip()
    if not entry["name"].replace("_", "").isalnum():
        raise ValueError("The function name may contain letters, digits and _ only.")
    existing = fl.get_function(entry["name"])
    if existing is not None and not existing.user_defined:
        raise ValueError(f"'{entry['name']}' is a built-in function; choose another name.")
    entry["category"] = (entry.get("category") or DEFAULT_CATEGORY).strip()
    f = make_function(entry)
    entries = [e for e in load_entries() if not (e["name"] == entry["name"] and
                                                e.get("category") == entry["category"])]
    entries.append(entry)
    save_entries(entries)
    if f.nvar == 1:
        fl.register(f)
    return f


def delete(name: str, category: str) -> None:
    save_entries([e for e in load_entries() if not (e["name"] == name and
                                                   e.get("category") == category)])
    fl.unregister(name, category)


class FunctionOrganizerDialog(QDialog):
    """List, create, edit, test and delete user fitting functions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fitting Function Organizer")
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Function", "Equation"])
        self.tree.itemSelectionChanged.connect(self._show_selected)
        self.name = QLineEdit()
        self.category = QComboBox()
        self.category.setEditable(True)
        self.kind = QComboBox()
        self.kind.addItem("Curve  y = f(x)", "curve")
        self.kind.addItem("Surface  z = f(x, y)", "surface")
        self.equation = QPlainTextEdit()
        self.equation.setFixedHeight(64)
        self.equation.setPlaceholderText("y = A*exp(-x/t) + y0")
        self.initial = QLineEdit()
        self.initial.setPlaceholderText("A = 1, t = 10, y0 = 0")
        self.description = QLineEdit()
        self.status = QLabel("")
        self.status.setWordWrap(True)
        form = QFormLayout()
        form.addRow("Function Name:", self.name)
        form.addRow("Category:", self.category)
        form.addRow("Type:", self.kind)
        form.addRow("Equation:", self.equation)
        form.addRow("Initial Values:", self.initial)
        form.addRow("Description:", self.description)
        btns = QHBoxLayout()
        for text, slot in (("New", self.new), ("Check Syntax", self.check), ("Save", self.save),
                           ("Delete", self.remove)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            btns.addWidget(b)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.addLayout(form)
        rv.addLayout(btns)
        rv.addWidget(self.status)
        rv.addStretch(1)
        top = QHBoxLayout()
        top.addWidget(self.tree, 1)
        top.addWidget(right, 1)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(self.reject)
        close.accepted.connect(self.accept)
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(close)
        self.resize(820, 420)
        self.refresh()

    def refresh(self) -> None:
        self.tree.clear()
        cats: dict[str, QTreeWidgetItem] = {}
        for e in load_entries():
            c = e.get("category") or DEFAULT_CATEGORY
            if c not in cats:
                cats[c] = QTreeWidgetItem([c, ""])
                self.tree.addTopLevelItem(cats[c])
            it = QTreeWidgetItem([e["name"], e["equation"]])
            it.setData(0, Qt.ItemDataRole.UserRole, e)
            cats[c].addChild(it)
        self.tree.expandAll()
        self.category.clear()
        self.category.addItems(sorted(set(cats) | {DEFAULT_CATEGORY}))

    def _show_selected(self) -> None:
        items = self.tree.selectedItems()
        e = items[0].data(0, Qt.ItemDataRole.UserRole) if items else None
        if not e:
            return
        self.name.setText(e["name"])
        self.category.setCurrentText(e.get("category") or DEFAULT_CATEGORY)
        self.kind.setCurrentIndex(1 if e.get("type") == "surface" else 0)
        self.equation.setPlainText(e["equation"])
        self.initial.setText(", ".join(f"{k} = {v:g}" for k, v in (e.get("initial") or {}).items()))
        self.description.setText(e.get("description", ""))

    def entry(self) -> dict:
        return {"name": self.name.text().strip(), "category": self.category.currentText().strip(),
                "type": self.kind.currentData(), "equation": self.equation.toPlainText().strip(),
                "initial": parse_initial(self.initial.text()),
                "description": self.description.text().strip()}

    def new(self) -> None:
        for w in (self.name, self.initial, self.description):
            w.clear()
        self.equation.clear()
        self.status.setText("")

    def check(self) -> bool:
        try:
            f = make_function(self.entry())
        except ValueError as exc:
            self.status.setStyleSheet("color:#b00000")
            self.status.setText(str(exc))
            return False
        self.status.setStyleSheet("color:#1b5e20")
        self.status.setText(f"OK — parameters: {', '.join(f.params)}")
        return True

    def save(self) -> bool:
        try:
            upsert(self.entry())
        except ValueError as exc:
            self.status.setStyleSheet("color:#b00000")
            self.status.setText(str(exc))
            return False
        self.status.setStyleSheet("color:#1b5e20")
        self.status.setText(f"Saved {self.name.text().strip()} to {store_path()}")
        self.refresh()
        return True

    def remove(self) -> None:
        e = self.entry()
        if not e["name"]:
            return
        if self.isVisible() and QMessageBox.question(
                self, "Delete Function", f"Delete {e['name']}?") != QMessageBox.StandardButton.Yes:
            return
        delete(e["name"], e["category"])
        self.refresh()
        self.new()

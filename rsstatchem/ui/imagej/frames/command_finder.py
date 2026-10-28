"""Plugins > Utilities > Find Commands... [l] (``ij/plugin/CommandFinder.java``): type to
filter every command (all words must match the command or its menu path), Enter or
double-click runs the selected one; disabled (not yet available) commands are listed grey.
**Owned by WP-A.**"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QCheckBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
                               QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
                               QWidget)

from rsstatchem.core.imaging.menu_layout import iter_items


def command_rows(registry) -> list[tuple[str, str, str, bool]]:
    """(label, menu path, shortcut, available) for every menu command."""
    from rsstatchem.core.imaging.menu_layout import qt_shortcut
    rows = []
    seen = set()
    for path, item in iter_items():
        if item.label in seen:
            continue
        seen.add(item.label)
        cmd = registry.get(item.label)
        rows.append((item.label, " > ".join(path), qt_shortcut(item),
                     cmd is not None and cmd.available))
    for cmd in registry:
        if cmd.label not in seen:
            rows.append((cmd.label, "", "", cmd.available))
    return rows


def search(rows, text: str):
    words = text.lower().split()
    return [r for r in rows if all(w in (r[0] + " " + r[1]).lower() for w in words)]


class CommandFinder(QMainWindow):
    def __init__(self, bar):
        super().__init__()
        self.bar = bar
        self.setWindowTitle("Command Finder")
        central = QWidget()
        lay = QVBoxLayout(central)
        row = QHBoxLayout()
        row.addWidget(QLabel("Search:"))
        self.query = QLineEdit()
        self.query.setPlaceholderText("Type part of a command name")
        row.addWidget(self.query, 1)
        lay.addLayout(row)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Command", "Menu Path", "Shortcut"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(self.table, 1)
        bottom = QHBoxLayout()
        self.close_when_running = QCheckBox("Close when running")
        self.close_when_running.setChecked(True)
        bottom.addWidget(self.close_when_running)
        bottom.addStretch(1)
        run = QPushButton("Run")
        run.clicked.connect(self.run_selected)
        bottom.addWidget(run)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        bottom.addWidget(close)
        lay.addLayout(bottom)
        self.setCentralWidget(central)
        self.rows = command_rows(bar.session.registry)
        self.query.textChanged.connect(self.refilter)
        self.query.returnPressed.connect(self.run_selected)
        self.table.cellDoubleClicked.connect(lambda r, c: self.run_selected())
        self.resize(640, 420)
        self.refilter("")

    def refilter(self, text: str | None = None) -> None:
        text = self.query.text() if text is None else text
        hits = search(self.rows, text)
        self.table.setRowCount(len(hits))
        for i, (label, path, sc, ok) in enumerate(hits):
            for j, val in enumerate((label, path, sc)):
                item = QTableWidgetItem(val)
                if not ok:
                    item.setForeground(QColor(150, 150, 150))
                self.table.setItem(i, j, item)
        if hits:
            self.table.selectRow(0)
        self.table.resizeColumnToContents(0)

    def selected_label(self) -> str | None:
        r = self.table.currentRow()
        if r < 0:
            return None
        item = self.table.item(r, 0)
        return item.text() if item is not None else None

    def run_selected(self) -> None:
        label = self.selected_label()
        if label is None:
            return
        if self.close_when_running.isChecked():
            self.close()
        self.bar.run_command(label)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in (Qt.Key.Key_Down, Qt.Key.Key_Up):
            r = self.table.currentRow() + (1 if event.key() == Qt.Key.Key_Down else -1)
            if 0 <= r < self.table.rowCount():
                self.table.selectRow(r)
            return
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802
        from rsstatchem.ui.imagej import windows as W
        event.accept()
        W.unregister(self)


def show_command_finder(bar) -> CommandFinder:
    from rsstatchem.ui.imagej import windows as W
    win = CommandFinder(bar)
    W.register(win)
    win.show()
    win.query.setFocus()
    return win

"""Window > Modules > Claude Connector: add the connector to the Claude desktop app, and allow
Claude to work in the RS-StatChem windows that are open (live mode, off by default)."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog, QDialogButtonBox, QGroupBox,
                               QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout)

from rsstatchem.mcp import setup


class ConnectorDialog(QDialog):
    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.setWindowTitle("Claude Connector")
        self.setMinimumWidth(560)

        def text(s: str) -> QLabel:
            label = QLabel(s)
            label.setWordWrap(True)
            label.setTextFormat(Qt.TextFormat.PlainText)
            return label

        box1 = QGroupBox("1. Claude desktop app")
        lay1 = QVBoxLayout(box1)
        lay1.addWidget(text(
            "Lets the Claude app on this computer, signed in with your own account, use every "
            "module of RS-StatChem. RS-StatChem itself never signs in to Claude. After adding "
            "it, quit Claude completely and open it again."))
        self.status = text("")
        lay1.addWidget(self.status)
        row = QHBoxLayout()
        self.add = QPushButton("Add to Claude")
        self.add.clicked.connect(self.add_entry)
        self.remove = QPushButton("Remove from Claude")
        self.remove.clicked.connect(self.remove_entry)
        copy = QPushButton("Copy Configuration")
        copy.clicked.connect(lambda: QApplication.clipboard().setText(setup.config_text()))
        for b in (self.add, self.remove, copy):
            row.addWidget(b)
        row.addStretch(1)
        lay1.addLayout(row)

        box2 = QGroupBox("2. Open windows")
        lay2 = QVBoxLayout(box2)
        self.live = QCheckBox("Allow Claude to work in the windows that are open")
        self.live.setChecked(ctx.connector_listening())
        self.live.clicked.connect(self.toggle_live)
        lay2.addWidget(self.live)
        lay2.addWidget(text(
            "Off: Claude works in the background and saves its results to files. On: Claude "
            "reads the data you have open and its commands run in these windows, where you can "
            "watch and undo them. Save your work first. Names and tables Claude reads are sent "
            "to Anthropic as part of your conversation."))

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(box1)
        layout.addWidget(box2)
        layout.addWidget(buttons)
        self.refresh()

    def refresh(self) -> None:
        state = setup.state()
        self.status.setText({
            "added": "RS-StatChem is in the configuration of the Claude app.",
            "other": "The configuration of the Claude app points to another copy of "
                     "RS-StatChem; Add to Claude changes it to this one.",
            "missing": "RS-StatChem is not yet in the configuration of the Claude app.",
            "broken": "The configuration file of the Claude app cannot be read; use Copy "
                      "Configuration and edit it by hand.",
        }[state] + f"\n{setup.config_path()}")
        self.add.setEnabled(state in ("missing", "other"))
        self.remove.setEnabled(state in ("added", "other"))

    def add_entry(self) -> None:
        self._change(setup.add, "Added. Quit the Claude app completely and open it again; "
                                "rs-statchem then appears under Connectors.")

    def remove_entry(self) -> None:
        self._change(setup.remove, "Removed. Quit the Claude app and open it again.")

    def _change(self, action, done: str) -> None:
        if self.ctx.connector_calls:     # pressed by Claude, not by the user
            return
        try:
            backup: Path | None = action()
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, self.windowTitle(), str(exc))
            return
        self.refresh()
        note = f"\n\nThe previous file was kept as {backup.name}." if backup else ""
        QMessageBox.information(self, self.windowTitle(), done + note)

    def toggle_live(self, on: bool) -> None:
        if self.ctx.connector_calls:     # pressed by Claude, not by the user
            self.live.setChecked(not on)
            return
        if on and not self.ctx.start_connector():
            self.live.setChecked(False)
            QMessageBox.warning(self, self.windowTitle(),
                                "Another RS-StatChem on this computer already allows Claude. "
                                "Switch it off there first.")
            return
        if not on:
            self.ctx.stop_connector()
        self.ctx.settings.setValue("connector/live", "1" if on else "0")

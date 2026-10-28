"""Start page of an empty RSM window, like Design-Expert's start screen (kept free of
matplotlib imports so a new window opens instantly)."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton,
                               QVBoxLayout, QWidget)

from rsstatchem.ui.dx import icons, style


class StartPage(QWidget):
    """Empty workspace: New Design / Open Design / Import Data Set (historical data) /
    tutorial data / recent designs."""

    new_design = Signal()
    open_design = Signal()
    open_path = Signal(str)
    open_example = Signal(str)
    import_data = Signal()

    def __init__(self, recent: list[str], parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(40, 30, 40, 30)
        title = QLabel("RSM / DOE")
        title.setStyleSheet(f"font-size:22pt; font-weight:600; color:{style.BLUE};")
        sub = QLabel("Design of experiments and response surface methods, laid out like "
                     "Stat-Ease Design-Expert: build a design, run the experiments, type the "
                     "results into the design layout, then analyse each response and optimise.")
        sub.setWordWrap(True)
        sub.setStyleSheet(f"color:{style.MUTED}; font-size:11pt;")
        lay.addWidget(title)
        lay.addWidget(sub)
        lay.addSpacing(16)
        row = QHBoxLayout()
        left = QVBoxLayout()
        for icon_name, text, sig in (("new", "  New Design...", self.new_design),
                                     ("open", "  Open Design...", self.open_design),
                                     ("import", "  Import Data Set...", self.import_data)):
            b = QPushButton(icons.icon(icon_name), text)
            b.setObjectName("bigButton")
            b.setIconSize(QSize(28, 28))
            b.setMinimumWidth(260)
            b.clicked.connect(sig.emit)
            left.addWidget(b)
        left.addSpacing(10)
        tut = QLabel("Tutorial data")
        tut.setStyleSheet(f"color:{style.BLUE}; font-weight:600;")
        left.addWidget(tut)
        from rsstatchem.ui.dx.examples import EXAMPLES

        for key, (text, _) in EXAMPLES.items():
            b = QPushButton(icons.icon("start"), "  " + text)
            b.setStyleSheet("text-align:left; padding:6px 12px;")
            b.clicked.connect(lambda _=False, k=key: self.open_example.emit(k))
            left.addWidget(b)
        left.addStretch(1)
        row.addLayout(left)
        row.addSpacing(30)
        right = QVBoxLayout()
        rl = QLabel("Recent designs")
        rl.setStyleSheet(f"color:{style.BLUE}; font-weight:600;")
        right.addWidget(rl)
        self.recent = QListWidget()
        for p in recent:
            it = QListWidgetItem(icons.icon("design"), Path(p).name)
            it.setToolTip(p)
            it.setData(Qt.ItemDataRole.UserRole, p)
            self.recent.addItem(it)
        if not recent:
            it = QListWidgetItem("(none yet)")
            it.setFlags(Qt.ItemFlag.NoItemFlags)
            self.recent.addItem(it)
        self.recent.itemActivated.connect(
            lambda it: it.data(Qt.ItemDataRole.UserRole) and self.open_path.emit(
                it.data(Qt.ItemDataRole.UserRole)))
        right.addWidget(self.recent, 1)
        steps = QLabel(
            "<b>Workflow</b><ol style='margin-left:-20px'>"
            "<li><b>Design</b>: File &gt; New Design, choose the design and the factor "
            "ranges.</li><li><b>Run</b> the experiments in the run order shown and type each "
            "result into the response cells (or paste from Excel).</li>"
            "<li><b>Analysis</b>: click R1 in the navigation pane and work through the tabs "
            "left to right: Transform, Fit Summary, Model, ANOVA, Diagnostics, Model "
            "Graphs.</li><li><b>Optimization</b>: Numerical (desirability), Graphical "
            "(overlay plot) and Point Prediction.</li></ol>")
        steps.setWordWrap(True)
        steps.setObjectName("hint")
        right.addWidget(steps)
        row.addLayout(right, 1)
        lay.addLayout(row, 1)

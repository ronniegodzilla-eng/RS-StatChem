"""Layout page, merge graphs, batch export and printing dialogs for the Origin workspace.

A layout page (Origin's Layout window) is built as an ordinary graph whose layers are copies
of the chosen graphs placed on a publication page (A4, Letter, slide), so it saves with the
project and exports like any graph."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLineEdit,
                               QListWidget, QListWidgetItem, QSpinBox, QVBoxLayout)

from rsstatchem.core.graph import layout


class GraphPickDialog(QDialog):
    """Choose graphs (checkable list) plus page/grid options."""

    def __init__(self, names: list[str], title: str, parent=None, page: bool = True,
                 grid: bool = True):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(420, 460)
        self.list = QListWidget()
        for n in names:
            it = QListWidgetItem(n)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked)
            self.list.addItem(it)
        form = QFormLayout()
        self.cols = QSpinBox()
        self.cols.setRange(1, 6)
        self.cols.setValue(2)
        self.page = QComboBox()
        for name in layout.PAGE_SIZES:
            self.page.addItem(name)
        self.heading = QLineEdit()
        if grid:
            form.addRow("Columns", self.cols)
        if page:
            form.addRow("Page", self.page)
            form.addRow("Heading", self.heading)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(self.list, 1)
        lay.addLayout(form)
        lay.addWidget(buttons)

    def chosen(self) -> list[str]:
        return [self.list.item(k).text() for k in range(self.list.count())
                if self.list.item(k).checkState() == Qt.CheckState.Checked]


def _graphs(main) -> dict:
    """{window name: current Graph} of the project's graph windows."""
    out = {}
    for gwin in main.graph_windows():
        out[gwin.graph.title] = gwin.graph
    return out


def new_layout_page(main, names: list[str] | None = None, cols: int = 2,
                    page: str = "A4 portrait", heading: str = ""):
    graphs = _graphs(main)
    if not graphs:
        main._warn("The project has no graphs to place on a layout page.", "Layout Page")
        return None
    if names is None:
        dlg = GraphPickDialog(list(graphs), "New Layout Page", main)
        if not dlg.exec():
            return None
        names, cols = dlg.chosen(), dlg.cols.value()
        page, heading = dlg.page.currentText(), dlg.heading.text()
    chosen = [graphs[n] for n in names if n in graphs]
    if not chosen:
        return None
    g = layout.layout_page(chosen, cols, page, title=main.project.unique_name("Layout"),
                           heading=heading)
    gwin = main.new_graph(g, name=g.title)
    main.message(f"Layout page {g.title}: {', '.join(names)}")
    return gwin


def merge_graph_windows(main, names: list[str] | None = None, cols: int | None = None):
    graphs = _graphs(main)
    if len(graphs) < 1:
        main._warn("The project has no graphs to merge.", "Merge Graph Windows")
        return None
    if names is None:
        dlg = GraphPickDialog(list(graphs), "Merge Graph Windows", main, page=False)
        if not dlg.exec():
            return None
        names, cols = dlg.chosen(), dlg.cols.value()
    chosen = [graphs[n] for n in names if n in graphs]
    if not chosen:
        return None
    n = sum(len(layout.gridded(g)) for g in chosen)
    cols = cols or None
    rows = None if cols is None else -(-n // cols)
    g = layout.merge_graphs(chosen, rows, cols, title=main.project.unique_name("Merged"))
    gwin = main.new_graph(g, name=g.title)
    main.message(f"Merged {', '.join(names)} → {g.title}")
    return gwin


def export_all_graphs(main, folder: str | None = None, fmt: str | None = None,
                      dpi: int | None = None) -> list:
    from PySide6.QtWidgets import QFileDialog, QInputDialog

    from rsstatchem.core.graph.export import DPI_CHOICES, FORMATS, export_many
    graphs = _graphs(main)
    if not graphs:
        main._warn("The project has no graphs to export.", "Export Graphs")
        return []
    if folder is None:
        folder = QFileDialog.getExistingDirectory(main, "Export All Graphs to Folder")
        if not folder:
            return []
    if fmt is None:
        items = [f"{k} — {v}" for k, v in FORMATS.items()]
        text, ok = QInputDialog.getItem(main, "Export All Graphs", "Format:", items, 0, False)
        if not ok:
            return []
        fmt = text.split(" ")[0]
    if dpi is None:
        dpi = 300
        if fmt in ("png", "jpg", "tif"):
            text, ok = QInputDialog.getItem(main, "Export All Graphs", "Resolution (DPI):",
                                            [str(d) for d in DPI_CHOICES], 1, False)
            if not ok:
                return []
            dpi = int(text)
    paths = export_many(graphs, folder, fmt, dpi)
    main.message(f"Exported {len(paths)} graphs to {folder}")
    return paths


def print_graph(main, gwin, printer=None) -> bool:
    """Print the graph page (File > Print), scaled to the printable area."""
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtPrintSupport import QPrintDialog, QPrinter

    from rsstatchem.core.graph.export import export_bytes
    if printer is None:
        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        dlg = QPrintDialog(printer, main)
        if not dlg.exec():
            return False
    img = QImage.fromData(export_bytes(gwin.graph, "png", dpi=300))
    painter = QPainter(printer)
    try:
        rect = painter.viewport()
        size = img.size()
        size.scale(rect.size(), Qt.AspectRatioMode.KeepAspectRatio)
        painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
        painter.setWindow(img.rect())
        painter.drawImage(0, 0, img)
    finally:
        painter.end()
    return True

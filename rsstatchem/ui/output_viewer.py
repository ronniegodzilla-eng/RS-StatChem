"""Output Viewer window: outline tree on the left, rendered results on the right."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPoint, Qt, QUrl
from PySide6.QtGui import QAction, QImage, QKeySequence, QPageLayout, QPageSize, QTextDocument
from PySide6.QtWidgets import (QFileDialog, QMainWindow, QMenu, QMessageBox, QSplitter,
                               QTextBrowser, QTreeWidget, QTreeWidgetItem)

from rsstatchem import APP_NAME
from rsstatchem.core import report
from rsstatchem.core.output import Image, OutputDocument, Section, Table, Text
from rsstatchem.ui import icons

ID_ROLE = Qt.ItemDataRole.UserRole


class OutputViewer(QMainWindow):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.doc = OutputDocument()
        self.setWindowTitle(f"Output1 - {APP_NAME} Viewer")
        self.setWindowIcon(icons.icon("output"))
        self.resize(1000, 720)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setMinimumWidth(200)
        self.tree.itemClicked.connect(self._jump)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._tree_menu)
        self.root = QTreeWidgetItem(["Output"])
        self.root.setIcon(0, icons.icon("output"))
        self.tree.addTopLevelItem(self.root)

        self.view = QTextBrowser()
        self.view.setOpenLinks(False)
        self.view.document().setDefaultStyleSheet(report.CSS)

        splitter = QSplitter()
        splitter.addWidget(self.tree)
        splitter.addWidget(self.view)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([230, 770])
        self.setCentralWidget(splitter)
        self._build_menus()
        self.statusBar().showMessage("Ready")

    # ------------------------------------------------------------------ menus
    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        for text, slot, shortcut in (
                ("Export to &Word (.docx)...", self.export_docx, "Ctrl+E"),
                ("Export to &HTML...", self.export_html, None),
                ("Export to &PDF...", self.export_pdf, None)):
            action = QAction(text, self)
            if shortcut:
                action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(slot)
            file_menu.addAction(action)
        file_menu.addSeparator()
        close = QAction("&Close", self)
        close.setShortcut(QKeySequence.StandardKey.Close)
        close.triggered.connect(self.close)
        file_menu.addAction(close)

        edit = self.menuBar().addMenu("&Edit")
        copy = QAction("&Copy", self)
        copy.setShortcut(QKeySequence.StandardKey.Copy)
        copy.triggered.connect(self.view.copy)
        select_all = QAction("Select &All", self)
        select_all.setShortcut(QKeySequence.StandardKey.SelectAll)
        select_all.triggered.connect(self.view.selectAll)
        clear = QAction("Clear All Output", self)
        clear.triggered.connect(self.clear)
        edit.addActions([copy, select_all])
        edit.addSeparator()
        edit.addAction(clear)

        self.window_menu = self.menuBar().addMenu("&Window")

    # ------------------------------------------------------------------ content
    def add_section(self, section: Section) -> None:
        self.doc.add(section)
        node = QTreeWidgetItem([section.title])
        node.setData(0, ID_ROLE, f"s{section.id}")
        node.setIcon(0, icons.icon("frequencies"))
        for item in section.items:
            child = QTreeWidgetItem([self._item_title(item)])
            child.setData(0, ID_ROLE, f"i{item.id}")
            child.setIcon(0, icons.icon(self._item_icon(item)))
            node.addChild(child)
        self.root.addChild(node)
        node.setExpanded(True)
        self.root.setExpanded(True)
        self._render()
        self.view.scrollToAnchor(f"s{section.id}")
        self.tree.setCurrentItem(node)

    @staticmethod
    def _item_title(item) -> str:
        if isinstance(item, Text) and item.style == "log":
            return "Log"
        if isinstance(item, Text) and not item.title:
            return "Note" if item.style != "warning" else "Warning"
        return item.title or item.kind.capitalize()

    @staticmethod
    def _item_icon(item) -> str:
        if isinstance(item, Table):
            return "data_editor"
        if isinstance(item, Image):
            return "chart"
        return "labels"

    def _render(self) -> None:
        document = self.view.document()
        for section in self.doc.sections:
            for item in section.items:
                if isinstance(item, Image):
                    document.addResource(QTextDocument.ResourceType.ImageResource,
                                         QUrl(f"img://{item.id}"), QImage.fromData(item.png))
        scroll = self.view.verticalScrollBar().value()
        self.view.setHtml(report.document_html(self.doc, lambda img: f"img://{img.id}",
                                               standalone=False))
        self.view.verticalScrollBar().setValue(scroll)

    def _jump(self, item: QTreeWidgetItem) -> None:
        anchor = item.data(0, ID_ROLE)
        if anchor:
            self.view.scrollToAnchor(anchor)

    def _tree_menu(self, pos: QPoint) -> None:
        item = self.tree.itemAt(pos)
        if item is None or item is self.root:
            return
        menu = QMenu(self)
        delete = menu.addAction("Delete")
        if menu.exec(self.tree.viewport().mapToGlobal(pos)) is delete:
            anchor = item.data(0, ID_ROLE)
            ident = int(anchor[1:])
            if anchor.startswith("s"):
                self.doc.remove(ident)
            else:
                self.doc.remove_item(ident)
            (item.parent() or self.root).removeChild(item)
            self._render()

    def clear(self) -> None:
        if self.doc.sections and QMessageBox.question(
                self, APP_NAME, "Delete all output?") != QMessageBox.StandardButton.Yes:
            return
        self.doc.clear()
        self.root.takeChildren()
        self._render()

    # ------------------------------------------------------------------ export
    def _ask_path(self, title: str, pattern: str, suffix: str) -> str | None:
        start = self.settings.value("last_dir", str(Path.home()))
        path, _ = QFileDialog.getSaveFileName(self, title, str(Path(start) / f"Output{suffix}"),
                                              pattern)
        if not path:
            return None
        if not path.lower().endswith(suffix):
            path += suffix
        self.settings.setValue("last_dir", str(Path(path).parent))
        return path

    def _done(self, path: str) -> None:
        self.statusBar().showMessage(f"Exported to {path}", 8000)

    def export_docx(self) -> None:
        path = self._ask_path("Export Output to Word", "Word document (*.docx)", ".docx")
        if path:
            report.export_docx(self.doc, path)
            self._done(path)

    def export_html(self) -> None:
        path = self._ask_path("Export Output to HTML", "Web page (*.html)", ".html")
        if path:
            Path(path).write_text(report.document_html(self.doc), encoding="utf-8")
            self._done(path)

    def export_pdf(self) -> None:
        from PySide6.QtPrintSupport import QPrinter

        path = self._ask_path("Export Output to PDF", "PDF (*.pdf)", ".pdf")
        if not path:
            return
        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
        printer.setOutputFileName(path)
        printer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
        printer.setPageOrientation(QPageLayout.Orientation.Portrait)
        self.view.document().print_(printer)
        self._done(path)

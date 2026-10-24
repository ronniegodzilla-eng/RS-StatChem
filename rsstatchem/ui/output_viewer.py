"""Output Viewer window: outline tree on the left, rendered results on the right.

Works like SPSS's viewer: click a table, chart or note (or its entry in the outline) to select
it, then Copy puts it on the clipboard as a Word-ready table plus an image and tab-separated
text; Copy Special chooses one format. The view zooms from 50 % to 300 %.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, QMimeData, QPoint, QSize, Qt, QUrl
from PySide6.QtGui import (QAction, QGuiApplication, QImage, QKeySequence, QPageLayout, QPageSize,
                           QPainter, QTextDocument, QTextOption)
from PySide6.QtWidgets import (QComboBox, QFileDialog, QLabel, QMainWindow, QMenu, QMessageBox,
                               QSplitter, QTextBrowser, QToolBar, QTreeWidget, QTreeWidgetItem)

from rsstatchem import APP_NAME
from rsstatchem.core import report
from rsstatchem.core.output import Image, OutputDocument, Section, Table, Text
from rsstatchem.ui import icons

ID_ROLE = Qt.ItemDataRole.UserRole
ZOOM_LEVELS = (0.5, 0.67, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)


def _wrap_words_only(doc: QTextDocument) -> QTextDocument:
    """Qt's default wraps anywhere, which squeezes wide tables into one letter per line;
    wrapping only at spaces makes a too-wide table scroll horizontally instead."""
    option = doc.defaultTextOption()
    option.setWrapMode(QTextOption.WrapMode.WordWrap)
    doc.setDefaultTextOption(option)
    return doc


def render_html_image(html: str, scale: float = 2.0) -> QImage:
    """Rasterise rich text (a table) into a crisp image for the clipboard."""
    doc = _wrap_words_only(QTextDocument())
    doc.setDocumentMargin(6)
    doc.setHtml(html)
    doc.setTextWidth(doc.idealWidth() + 12)
    size = doc.size().toSize()
    image = QImage(QSize(int(size.width() * scale), int(size.height() * scale)),
                   QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.white)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    painter.scale(scale, scale)
    doc.drawContents(painter)
    painter.end()
    image.setDevicePixelRatio(1.0)
    return image


class OutputViewer(QMainWindow):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.doc = OutputDocument()
        self.selected: str | None = None  # "s<id>" section or "i<id>" item
        try:
            self.zoom = float(settings.value("output/zoom", 1.0))
        except (TypeError, ValueError):
            self.zoom = 1.0
        self.setWindowTitle(f"Output1 - {APP_NAME} Viewer")
        self.setWindowIcon(icons.icon("output"))
        self.resize(1050, 740)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setMinimumWidth(200)
        self.tree.currentItemChanged.connect(self._tree_selected)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(
            lambda pos: self._context_menu(self.tree.viewport().mapToGlobal(pos)))
        self.root = QTreeWidgetItem(["Output"])
        self.root.setIcon(0, icons.icon("output"))
        self.tree.addTopLevelItem(self.root)

        self.view = QTextBrowser()
        self.view.setOpenLinks(False)
        self.view.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self._view_menu)
        self.view.viewport().installEventFilter(self)
        self.view.installEventFilter(self)  # Ctrl+C would otherwise only copy dragged text

        splitter = QSplitter()
        splitter.addWidget(self.tree)
        splitter.addWidget(self.view)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([240, 810])
        self.setCentralWidget(splitter)
        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self.zoom_label = QLabel()
        self.statusBar().addPermanentWidget(self.zoom_label)
        self.statusBar().showMessage("Click a table or chart to select it; Ctrl+C copies it "
                                     "for Word.")
        self._update_zoom_widgets()

    # ------------------------------------------------------------------ actions / menus
    def _action(self, text, slot, shortcut=None, icon=None) -> QAction:
        action = QAction(text, self)
        if shortcut:
            action.setShortcuts(shortcut if isinstance(shortcut, list) else [shortcut])
        if icon:
            action.setIcon(icons.icon(icon))
        action.triggered.connect(slot)
        return action

    def _build_actions(self) -> None:
        self.act_copy = self._action("&Copy", lambda: self.copy(),
                                     QKeySequence(QKeySequence.StandardKey.Copy))
        self.act_copy_image = self._action("Copy as &Image", lambda: self.copy("image"))
        self.act_copy_text = self._action("Copy as &Text (Excel)", lambda: self.copy("text"))
        self.act_copy_html = self._action("Copy as &HTML (Word table)", lambda: self.copy("html"))
        self.act_transpose = self._action("&Transpose Table (swap rows and columns)",
                                          self.transpose_selected, QKeySequence("Ctrl+T"))
        self.act_delete = self._action("&Delete", self.delete_selected,
                                       QKeySequence(QKeySequence.StandardKey.Delete))
        self.act_zoom_in = self._action("Zoom &In", lambda: self.step_zoom(1),
                                        [QKeySequence(QKeySequence.StandardKey.ZoomIn),
                                         QKeySequence("Ctrl+=")])
        self.act_zoom_out = self._action("Zoom &Out", lambda: self.step_zoom(-1),
                                         QKeySequence(QKeySequence.StandardKey.ZoomOut))
        self.act_zoom_reset = self._action("&Actual Size (100%)", lambda: self.set_zoom(1.0),
                                           QKeySequence("Ctrl+0"))
        self.act_export_docx = self._action("Export to &Word (.docx)...", self.export_docx,
                                            QKeySequence("Ctrl+E"), "export")

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.act_export_docx)
        for text, slot in (("Export to &HTML...", self.export_html),
                           ("Export to &PDF...", self.export_pdf)):
            file_menu.addAction(self._action(text, slot))
        file_menu.addSeparator()
        file_menu.addAction(self._action("&Close", self.close,
                                         QKeySequence(QKeySequence.StandardKey.Close)))

        edit = self.menuBar().addMenu("&Edit")
        edit.addAction(self.act_copy)
        special = edit.addMenu("Copy &Special")
        special.addActions([self.act_copy_html, self.act_copy_image, self.act_copy_text])
        edit.addSeparator()
        edit.addAction(self.act_transpose)
        edit.addSeparator()
        edit.addAction(self.act_delete)
        edit.addAction(self._action("Clear All Output", self.clear))

        view = self.menuBar().addMenu("&View")
        view.addActions([self.act_zoom_in, self.act_zoom_out, self.act_zoom_reset])

        self.window_menu = self.menuBar().addMenu("&Window")

    def _build_toolbar(self) -> None:
        tb = QToolBar("Output")
        tb.setMovable(False)
        self.addToolBar(tb)
        tb.addAction(self.act_export_docx)
        tb.addSeparator()
        copy = tb.addAction("Copy")
        copy.setToolTip("Copy the selected table/chart (Ctrl+C) — pastes into Word as a table")
        copy.triggered.connect(lambda: self.copy())
        tb.addSeparator()
        tb.addAction(self.act_zoom_out)
        self.zoom_box = QComboBox()
        self.zoom_box.setEditable(False)
        for z in ZOOM_LEVELS:
            self.zoom_box.addItem(f"{z * 100:.0f}%", z)
        self.zoom_box.activated.connect(lambda i: self.set_zoom(self.zoom_box.itemData(i)))
        tb.addWidget(self.zoom_box)
        tb.addAction(self.act_zoom_in)

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
        self.selected = None
        self._render()
        self.view.scrollToAnchor(f"s{section.id}")
        self.tree.blockSignals(True)
        self.tree.setCurrentItem(node)
        self.tree.blockSignals(False)

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

    def _render(self, keep_position: bool = True) -> None:
        document = self.view.document()
        for section in self.doc.sections:
            for item in section.items:
                if isinstance(item, Image):
                    document.addResource(QTextDocument.ResourceType.ImageResource,
                                         QUrl(f"img://{item.id}"), QImage.fromData(item.png))
        bar = self.view.verticalScrollBar()
        ratio = bar.value() / bar.maximum() if bar.maximum() else 0.0
        self.view.setHtml(report.document_html(self.doc, lambda img: f"img://{img.id}",
                                               standalone=False, scale=self.zoom,
                                               selected=self.selected))
        if keep_position:
            bar.setValue(int(ratio * bar.maximum()))

    # ------------------------------------------------------------------ selection
    def _object(self, anchor: str | None):
        if not anchor:
            return None
        ident = int(anchor[1:])
        for section in self.doc.sections:
            if anchor[0] == "s" and section.id == ident:
                return section
            for item in section.items:
                if anchor[0] == "i" and item.id == ident:
                    return item
        return None

    def _tree_node(self, anchor: str) -> QTreeWidgetItem | None:
        for i in range(self.root.childCount()):
            node = self.root.child(i)
            if node.data(0, ID_ROLE) == anchor:
                return node
            for j in range(node.childCount()):
                if node.child(j).data(0, ID_ROLE) == anchor:
                    return node.child(j)
        return None

    def select(self, anchor: str | None, scroll: bool = False) -> None:
        if anchor == self.selected:
            return
        self.selected = anchor
        self._render()
        if scroll and anchor:
            self.view.scrollToAnchor(anchor)
        node = self._tree_node(anchor) if anchor else None
        self.tree.blockSignals(True)
        self.tree.setCurrentItem(node or self.root)
        self.tree.blockSignals(False)

    def _tree_selected(self, current: QTreeWidgetItem | None, _previous=None) -> None:
        anchor = current.data(0, ID_ROLE) if current is not None else None
        self.select(anchor, scroll=True)

    def _anchor_at(self, pos: QPoint) -> str | None:
        """The output item under a point in the view (last anchor before that position)."""
        target = self.view.cursorForPosition(pos).position()
        block = self.view.document().begin()
        found = None
        while block.isValid() and block.position() <= target:
            it = block.begin()
            while not it.atEnd():
                fragment = it.fragment()
                if fragment.position() > target:
                    break
                for name in fragment.charFormat().anchorNames():
                    if name[:1] in ("i", "s") and name[1:].isdigit():
                        found = name
                it += 1
            block = block.next()
        return found

    def eventFilter(self, obj, event) -> bool:
        if obj is self.view and event.type() in (QEvent.Type.ShortcutOverride,
                                                 QEvent.Type.KeyPress) \
                and event.matches(QKeySequence.StandardKey.Copy):
            if event.type() == QEvent.Type.KeyPress:
                self.copy()
            event.accept()
            return True
        if obj is self.view.viewport():
            if event.type() == QEvent.Type.MouseButtonPress and \
                    event.button() == Qt.MouseButton.LeftButton:
                anchor = self._anchor_at(event.position().toPoint())
                if anchor:
                    self.select(anchor)
            elif event.type() == QEvent.Type.Wheel and \
                    event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.step_zoom(1 if event.angleDelta().y() > 0 else -1)
                return True
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------ copy
    def copy(self, mode: str = "all") -> None:
        """Copy the selected output object (or the dragged text selection) like SPSS does."""
        if mode == "all" and self.view.textCursor().hasSelection():
            self.view.copy()
            return
        obj = self._object(self.selected)
        if obj is None:
            self.statusBar().showMessage("Select a table, chart or note first (click it).", 5000)
            return
        mime = QMimeData()
        if mode in ("all", "html"):
            mime.setHtml(report.clipboard_html([obj]))
        if mode in ("all", "text", "html"):
            mime.setText(report.item_text(obj))
        if mode in ("all", "image"):
            image = self._object_image(obj)
            if image is not None:
                mime.setImageData(image)
        QGuiApplication.clipboard().setMimeData(mime)
        kind = {"all": "", "html": " as a Word table", "image": " as an image",
                "text": " as text"}[mode]
        self.statusBar().showMessage(f"Copied “{getattr(obj, 'title', '')}”{kind}.", 5000)

    def _object_image(self, obj) -> QImage | None:
        if isinstance(obj, Image):
            return QImage.fromData(obj.png)
        if isinstance(obj, Table):
            return render_html_image(f"<style>{report.css(1.0)}</style>"
                                     + report.table_html(obj))
        if isinstance(obj, Section):
            html = report.section_html(obj, scale=1.0)
            return render_html_image(f"<style>{report.css(1.0)}</style>{html}")
        return None

    # ------------------------------------------------------------------ zoom
    def set_zoom(self, value: float) -> None:
        value = max(ZOOM_LEVELS[0], min(ZOOM_LEVELS[-1], float(value)))
        if abs(value - self.zoom) < 1e-6:
            self._update_zoom_widgets()
            return
        self.zoom = value
        self.settings.setValue("output/zoom", value)
        self._render()
        self._update_zoom_widgets()

    def step_zoom(self, direction: int) -> None:
        if direction > 0:
            nxt = next((z for z in ZOOM_LEVELS if z > self.zoom + 1e-6), ZOOM_LEVELS[-1])
        else:
            nxt = next((z for z in reversed(ZOOM_LEVELS) if z < self.zoom - 1e-6),
                       ZOOM_LEVELS[0])
        self.set_zoom(nxt)

    def _update_zoom_widgets(self) -> None:
        self.zoom_label.setText(f"  Zoom {self.zoom * 100:.0f}%  ")
        index = min(range(len(ZOOM_LEVELS)), key=lambda i: abs(ZOOM_LEVELS[i] - self.zoom))
        self.zoom_box.blockSignals(True)
        self.zoom_box.setCurrentIndex(index)
        self.zoom_box.blockSignals(False)

    # ------------------------------------------------------------------ menus / delete
    def _view_menu(self, pos: QPoint) -> None:
        anchor = self._anchor_at(pos)
        if anchor:
            self.select(anchor)
        self._context_menu(self.view.viewport().mapToGlobal(pos))

    def _context_menu(self, global_pos: QPoint) -> None:
        menu = QMenu(self)
        has = self._object(self.selected) is not None
        for action in (self.act_copy, self.act_delete):
            action.setEnabled(has)
        menu.addAction(self.act_copy)
        special = menu.addMenu("Copy Special")
        special.setEnabled(has)
        special.addActions([self.act_copy_html, self.act_copy_image, self.act_copy_text])
        menu.addAction(self.act_transpose)
        self.act_transpose.setEnabled(isinstance(self._object(self.selected), Table))
        menu.addSeparator()
        menu.addActions([self.act_zoom_in, self.act_zoom_out, self.act_zoom_reset])
        menu.addSeparator()
        menu.addAction(self.act_delete)
        menu.exec(global_pos)
        for action in (self.act_copy, self.act_delete, self.act_transpose):
            action.setEnabled(True)

    def transpose_selected(self) -> None:
        """Swap rows and columns of the selected table (SPSS: pivot) — a very wide table
        becomes a tall one that fits a portrait page in Word."""
        obj = self._object(self.selected)
        if not isinstance(obj, Table):
            self.statusBar().showMessage("Select a table first.", 4000)
            return
        new = report.transpose(obj)
        for section in self.doc.sections:
            if obj in section.items:
                section.items[section.items.index(obj)] = new
        node = self._tree_node(self.selected)
        if node is not None:
            node.setData(0, ID_ROLE, f"i{new.id}")
        self.selected = f"i{new.id}"
        self._render()
        self.statusBar().showMessage(f"Transposed “{new.title}”.", 4000)

    def delete_selected(self) -> None:
        anchor = self.selected
        if not anchor:
            return
        node = self._tree_node(anchor)
        ident = int(anchor[1:])
        if anchor.startswith("s"):
            self.doc.remove(ident)
        else:
            self.doc.remove_item(ident)
        if node is not None:
            (node.parent() or self.root).removeChild(node)
        self.selected = None
        self._render()

    def clear(self) -> None:
        if self.doc.sections and QMessageBox.question(
                self, APP_NAME, "Delete all output?") != QMessageBox.StandardButton.Yes:
            return
        self.doc.clear()
        self.root.takeChildren()
        self.selected = None
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
        doc = _wrap_words_only(QTextDocument())
        for section in self.doc.sections:
            for item in section.items:
                if isinstance(item, Image):
                    doc.addResource(QTextDocument.ResourceType.ImageResource,
                                    QUrl(f"img://{item.id}"), QImage.fromData(item.png))
        doc.setHtml(report.document_html(self.doc, lambda img: f"img://{img.id}",
                                         standalone=False))
        doc.print_(printer)
        self._done(path)

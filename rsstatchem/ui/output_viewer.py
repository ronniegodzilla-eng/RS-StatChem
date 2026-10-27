"""Output Viewer window: outline tree on the left, rendered results on the right.

Works like SPSS's viewer: click a table, chart or note (or its entry in the outline) to select
it, then Copy puts it on the clipboard as a Word-ready table plus an image and tab-separated
text; Copy Special chooses one format. The view zooms from 50 % to 300 %.

Output documents are saved and reopened as ``.rso`` files (SPSS ``.spv`` equivalent) and
exported to Word, Excel, HTML, PDF or text. Tables can be edited like SPSS pivot tables
(double-click: cell text, footnotes, hidden rows/columns, decimals; Pivoting Trays; transpose),
and every rendering follows the chosen TableLook (SPSS Default or APA).
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, QMimeData, QPoint, QSize, Qt, QUrl
from PySide6.QtGui import (QAction, QActionGroup, QGuiApplication, QImage, QKeySequence,
                           QPageLayout, QPageSize, QPainter, QTextDocument, QTextOption)
from PySide6.QtWidgets import (QComboBox, QFileDialog, QInputDialog, QLabel, QMainWindow,
                               QMenu, QMessageBox, QSplitter, QTextBrowser, QToolBar,
                               QTreeWidget, QTreeWidgetItem)

from rsstatchem import APP_NAME
from rsstatchem.core import pivot as pv
from rsstatchem.core import report
from rsstatchem.core.output import Image, OutputDocument, Section, Table, Text
from rsstatchem.core.output_store import OUTPUT_EXT, OUTPUT_FILTER, load_output, save_output
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
        self.path: str | None = None
        self.dirty = False
        try:
            self.zoom = float(settings.value("output/zoom", 1.0))
        except (TypeError, ValueError):
            self.zoom = 1.0
        look = settings.value("output/look", report.DEFAULT_LOOK)
        self.look = look if look in report.LOOKS else report.DEFAULT_LOOK
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
                                     "for Word. Double-click a table to edit it.")
        self._update_zoom_widgets()
        self._update_title()

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
        a = self._action
        self.act_new = a("&New Output", self.new_document)
        self.act_open = a("&Open Output...", self.open_dialog,
                          QKeySequence(QKeySequence.StandardKey.Open), "open")
        self.act_save = a("&Save Output", self.save, QKeySequence(QKeySequence.StandardKey.Save),
                          "save")
        self.act_save_as = a("Save Output &As...", self.save_as,
                             QKeySequence(QKeySequence.StandardKey.SaveAs))
        self.act_copy = a("&Copy", lambda: self.copy(),
                          QKeySequence(QKeySequence.StandardKey.Copy))
        self.act_copy_image = a("Copy as &Image", lambda: self.copy("image"))
        self.act_copy_text = a("Copy as &Text (Excel)", lambda: self.copy("text"))
        self.act_copy_html = a("Copy as &HTML (Word table)", lambda: self.copy("html"))
        self.act_transpose = a("&Transpose Table (swap rows and columns)",
                               self.transpose_selected, QKeySequence("Ctrl+T"))
        self.act_edit_table = a("&Edit Table (cells, footnotes, hide, decimals)...",
                                self.edit_selected_table, QKeySequence("Ctrl+Shift+E"))
        self.act_pivot = a("&Pivoting Trays...", self.pivot_selected, QKeySequence("Ctrl+P"))
        self.act_reset_pivot = a("&Reset Pivots and Show Hidden", self.reset_selected)
        self.act_insert_title = a("Insert &Title...", lambda: self.insert_text("title"))
        self.act_insert_text = a("Insert Te&xt...", lambda: self.insert_text("note"))
        self.act_delete = a("&Delete", self.delete_selected,
                            QKeySequence(QKeySequence.StandardKey.Delete))
        self.act_zoom_in = a("Zoom &In", lambda: self.step_zoom(1),
                             [QKeySequence(QKeySequence.StandardKey.ZoomIn),
                              QKeySequence("Ctrl+=")])
        self.act_zoom_out = a("Zoom &Out", lambda: self.step_zoom(-1),
                              QKeySequence(QKeySequence.StandardKey.ZoomOut))
        self.act_zoom_reset = a("&Actual Size (100%)", lambda: self.set_zoom(1.0),
                                QKeySequence("Ctrl+0"))
        self.act_export_docx = a("Export to &Word (.docx)...", self.export_docx,
                                 QKeySequence("Ctrl+E"), "export")
        self.act_export_xlsx = a("Export to E&xcel (.xlsx)...", self.export_xlsx)
        self.act_export_html = a("Export to &HTML...", self.export_html)
        self.act_export_pdf = a("Export to &PDF...", self.export_pdf)
        self.act_export_text = a("Export to Te&xt (.txt)...", self.export_text)
        self.look_actions = {}
        group = QActionGroup(self)
        for key, label in report.LOOKS.items():
            act = QAction(label, self, checkable=True)
            act.setChecked(key == self.look)
            act.triggered.connect(lambda _=False, k=key: self.set_look(k))
            group.addAction(act)
            self.look_actions[key] = act

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addActions([self.act_new, self.act_open])
        file_menu.addSeparator()
        file_menu.addActions([self.act_save, self.act_save_as])
        file_menu.addSeparator()
        export = file_menu.addMenu("&Export")
        export.addActions([self.act_export_docx, self.act_export_xlsx, self.act_export_pdf,
                           self.act_export_html, self.act_export_text])
        file_menu.addSeparator()
        file_menu.addAction(self._action("&Close", self.close,
                                         QKeySequence(QKeySequence.StandardKey.Close)))

        edit = self.menuBar().addMenu("&Edit")
        edit.addAction(self.act_copy)
        special = edit.addMenu("Copy &Special")
        special.addActions([self.act_copy_html, self.act_copy_image, self.act_copy_text])
        edit.addSeparator()
        edit.addActions([self.act_edit_table, self.act_pivot, self.act_transpose,
                         self.act_reset_pivot])
        edit.addSeparator()
        edit.addActions([self.act_delete])
        edit.addAction(self._action("Clear All Output", self.clear))

        view = self.menuBar().addMenu("&View")
        view.addActions([self.act_zoom_in, self.act_zoom_out, self.act_zoom_reset])
        view.addSeparator()
        looks = view.addMenu("&TableLook")
        looks.addActions(list(self.look_actions.values()))

        insert = self.menuBar().addMenu("&Insert")
        insert.addActions([self.act_insert_title, self.act_insert_text])

        pivot_menu = self.menuBar().addMenu("&Pivot")
        pivot_menu.addActions([self.act_pivot, self.act_transpose, self.act_reset_pivot])

        self.window_menu = self.menuBar().addMenu("&Window")

    def _build_toolbar(self) -> None:
        tb = QToolBar("Output")
        tb.setMovable(False)
        self.addToolBar(tb)
        tb.addActions([self.act_open, self.act_save])
        tb.addSeparator()
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
        tb.addSeparator()
        tb.addWidget(QLabel(" TableLook: "))
        self.look_box = QComboBox()
        for key, label in report.LOOKS.items():
            self.look_box.addItem(label, key)
        self.look_box.setCurrentIndex(list(report.LOOKS).index(self.look))
        self.look_box.activated.connect(lambda i: self.set_look(self.look_box.itemData(i)))
        tb.addWidget(self.look_box)

    # ------------------------------------------------------------------ state
    def _set_dirty(self, dirty: bool = True) -> None:
        self.dirty = dirty
        self._update_title()

    def _update_title(self) -> None:
        name = Path(self.path).name if self.path else "Output1"
        self.setWindowTitle(f"{name}[*] - {APP_NAME} Viewer")
        self.setWindowModified(self.dirty)

    def set_look(self, look: str) -> None:
        if look not in report.LOOKS:
            return
        self.look = look
        self.settings.setValue("output/look", look)
        self.look_actions[look].setChecked(True)
        self.look_box.setCurrentIndex(list(report.LOOKS).index(look))
        self._render()

    # ------------------------------------------------------------------ content
    def add_section(self, section: Section) -> None:
        self.doc.add(section)
        node = self._section_node(section)
        self.root.addChild(node)
        node.setExpanded(True)
        self.root.setExpanded(True)
        self.selected = None
        self._render()
        self.view.scrollToAnchor(f"s{section.id}")
        self.tree.blockSignals(True)
        self.tree.setCurrentItem(node)
        self.tree.blockSignals(False)
        self._set_dirty()

    def _section_node(self, section: Section) -> QTreeWidgetItem:
        node = QTreeWidgetItem([section.title])
        node.setData(0, ID_ROLE, f"s{section.id}")
        node.setIcon(0, icons.icon("frequencies"))
        for item in section.items:
            child = QTreeWidgetItem([self._item_title(item)])
            child.setData(0, ID_ROLE, f"i{item.id}")
            child.setIcon(0, icons.icon(self._item_icon(item)))
            node.addChild(child)
        return node

    def _rebuild_tree(self) -> None:
        self.tree.blockSignals(True)
        self.root.takeChildren()
        for section in self.doc.sections:
            node = self._section_node(section)
            self.root.addChild(node)
            node.setExpanded(True)
        self.root.setExpanded(True)
        self.tree.blockSignals(False)

    @staticmethod
    def _item_title(item) -> str:
        if isinstance(item, Text) and item.style == "log":
            return "Log"
        if isinstance(item, Text) and item.style == "title":
            return item.text[:40] or "Title"
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
                                               selected=self.selected, look=self.look))
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
            elif event.type() == QEvent.Type.MouseButtonDblClick and \
                    event.button() == Qt.MouseButton.LeftButton:
                anchor = self._anchor_at(event.position().toPoint())
                if anchor and isinstance(self._object(anchor), Table):
                    self.select(anchor)
                    self.edit_selected_table()
                    return True
            elif event.type() == QEvent.Type.Wheel and \
                    event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.step_zoom(1 if event.angleDelta().y() > 0 else -1)
                return True
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------ copy
    def _table_number(self, table) -> int:
        n = 0
        for section in self.doc.sections:
            for item in section.items:
                if isinstance(item, Table):
                    n += 1
                    if item is table:
                        return n
        return 1

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
        first = next((i for i in ([obj] if not isinstance(obj, Section) else obj.items)
                      if isinstance(i, Table)), None)
        number = self._table_number(first) if first is not None else 1
        if mode in ("all", "html"):
            mime.setHtml(report.clipboard_html([obj], self.look, number))
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
        style = f"<style>{report.css(1.0, self.look)}</style>"
        if isinstance(obj, Table):
            return render_html_image(style + report.table_html(
                obj, look=self.look, number=self._table_number(obj)))
        if isinstance(obj, Section):
            html = report.section_html(obj, scale=1.0, look=self.look)
            return render_html_image(style + html)
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
        obj = self._object(self.selected)
        has = obj is not None
        is_table = isinstance(obj, Table)
        table_actions = (self.act_transpose, self.act_edit_table, self.act_pivot,
                         self.act_reset_pivot)
        for action in (self.act_copy, self.act_delete):
            action.setEnabled(has)
        for action in table_actions:
            action.setEnabled(is_table)
        menu.addAction(self.act_copy)
        special = menu.addMenu("Copy Special")
        special.setEnabled(has)
        special.addActions([self.act_copy_html, self.act_copy_image, self.act_copy_text])
        menu.addSeparator()
        menu.addActions(list(table_actions))
        menu.addSeparator()
        menu.addActions([self.act_insert_title, self.act_insert_text])
        menu.addSeparator()
        menu.addActions([self.act_zoom_in, self.act_zoom_out, self.act_zoom_reset])
        menu.addSeparator()
        menu.addAction(self.act_delete)
        menu.exec(global_pos)
        for action in (self.act_copy, self.act_delete, *table_actions):
            action.setEnabled(True)

    def replace_item(self, old, new) -> None:
        """Put ``new`` in place of ``old`` (keeps the outline and the selection in sync)."""
        for section in self.doc.sections:
            if old in section.items:
                section.items[section.items.index(old)] = new
        node = self._tree_node(f"i{old.id}")
        if node is not None:
            node.setData(0, ID_ROLE, f"i{new.id}")
            node.setText(0, self._item_title(new))
        if self.selected == f"i{old.id}":
            self.selected = f"i{new.id}"
        self._render()
        self._set_dirty()

    def _selected_table(self) -> Table | None:
        obj = self._object(self.selected)
        if not isinstance(obj, Table):
            self.statusBar().showMessage("Select a table first.", 4000)
            return None
        return obj

    def transpose_selected(self) -> None:
        """Swap rows and columns of the selected table (SPSS: pivot) — a very wide table
        becomes a tall one that fits a portrait page in Word."""
        obj = self._selected_table()
        if obj is None:
            return
        new = report.transpose(obj)
        self.replace_item(obj, new)
        self.statusBar().showMessage(f"Transposed “{new.title}”.", 4000)

    def edit_selected_table(self) -> None:
        from rsstatchem.ui.pivot_dialogs import TableEditorDialog

        obj = self._selected_table()
        if obj is None:
            return
        dlg = TableEditorDialog(obj, self)
        if dlg.exec():
            self.replace_item(obj, dlg.result_table())

    def pivot_selected(self) -> None:
        from rsstatchem.ui.pivot_dialogs import PivotTraysDialog

        obj = self._selected_table()
        if obj is None:
            return
        dlg = PivotTraysDialog(obj, self)
        if not dlg.exec():
            return
        spec = dlg.layout_spec()
        try:
            new = pv.pivot(obj, spec["rows"], spec["cols"], spec["layers"])
        except ValueError as exc:
            QMessageBox.warning(self, APP_NAME, str(exc))
            return
        self.replace_item(obj, new)

    def reset_selected(self) -> None:
        obj = self._selected_table()
        if obj is None:
            return
        new = pv.reset_pivot(obj)
        if new is obj:
            new = pv.copy_table(obj, hidden_rows=[], hidden_cols=[])
        self.replace_item(obj, new)

    def insert_text(self, style: str = "note") -> None:
        title = "Insert Title" if style == "title" else "Insert Text"
        text, ok = QInputDialog.getMultiLineText(self, title, "Text:")
        if ok and text.strip():
            self.add_text(text.strip(), style)

    def add_text(self, text: str, style: str = "note") -> None:
        """Insert a title/text item after the selected item (or at the end)."""
        item = Text("" if style == "title" else "Text", text, style=style)
        obj = self._object(self.selected)
        target = None
        for section in self.doc.sections:
            if obj is section:
                target = (section, len(section.items))
            elif obj in section.items:
                target = (section, section.items.index(obj) + 1)
        if target is None:
            if not self.doc.sections:
                self.add_section(Section("Text", [item]))
                return
            target = (self.doc.sections[-1], len(self.doc.sections[-1].items))
        target[0].items.insert(target[1], item)
        self._rebuild_tree()
        self.selected = f"i{item.id}"
        self._render()
        self._set_dirty()

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
        self._set_dirty()

    def clear(self) -> None:
        if self.doc.sections and QMessageBox.question(
                self, APP_NAME, "Delete all output?") != QMessageBox.StandardButton.Yes:
            return
        self.doc.clear()
        self.root.takeChildren()
        self.selected = None
        self._render()
        self._set_dirty()

    # ------------------------------------------------------------------ documents
    def _confirm_discard(self) -> bool:
        if not self.dirty or not self.doc.sections:
            return True
        answer = QMessageBox.question(
            self, APP_NAME, "Save the current output document first?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Save)
        if answer == QMessageBox.StandardButton.Save:
            return self.save()
        return answer == QMessageBox.StandardButton.Discard

    def new_document(self) -> None:
        if not self._confirm_discard():
            return
        self.doc = OutputDocument()
        self.path = None
        self.root.takeChildren()
        self.selected = None
        self._render()
        self._set_dirty(False)

    def open_dialog(self) -> None:
        if not self._confirm_discard():
            return
        start = self.settings.value("last_dir", str(Path.home()))
        path, _ = QFileDialog.getOpenFileName(self, "Open Output", start, OUTPUT_FILTER)
        if path:
            self.open_path(path)

    def open_path(self, path: str) -> bool:
        try:
            doc, look = load_output(path)
        except Exception as exc:  # noqa: BLE001 - a damaged or foreign file
            QMessageBox.warning(self, APP_NAME, f"Cannot open {path}:\n{exc}")
            return False
        self.load_document(doc, path, look)
        self.settings.setValue("last_dir", str(Path(path).parent))
        return True

    def load_document(self, doc: OutputDocument, path: str | None = None,
                      look: str | None = None) -> None:
        self.doc = doc
        self.path = path
        self.selected = None
        if look in report.LOOKS:
            self.look = look
            self.look_actions[look].setChecked(True)
            self.look_box.setCurrentIndex(list(report.LOOKS).index(look))
        self._rebuild_tree()
        self._render(keep_position=False)
        self._set_dirty(False)

    def save(self) -> bool:
        if self.path:
            return self.save_to(self.path)
        return self.save_as()

    def save_as(self) -> bool:
        path = self._ask_path("Save Output As", OUTPUT_FILTER, OUTPUT_EXT,
                              Path(self.path).stem if self.path else "Output1")
        return self.save_to(path) if path else False

    def save_to(self, path: str) -> bool:
        try:
            save_output(self.doc, path, self.look)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, APP_NAME, f"Cannot save {path}:\n{exc}")
            return False
        self.path = path
        self._set_dirty(False)
        self.statusBar().showMessage(f"Saved {path}", 6000)
        return True

    def closeEvent(self, event) -> None:
        offscreen = QGuiApplication.platformName() in ("offscreen", "minimal")
        if self.isVisible() and not offscreen and not self._confirm_discard():
            event.ignore()
            return
        event.accept()

    # ------------------------------------------------------------------ export
    def _ask_path(self, title: str, pattern: str, suffix: str,
                  base: str = "Output") -> str | None:
        start = self.settings.value("last_dir", str(Path.home()))
        path, _ = QFileDialog.getSaveFileName(self, title, str(Path(start) / f"{base}{suffix}"),
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
            report.export_docx(self.doc, path, self.look)
            self._done(path)

    def export_xlsx(self) -> None:
        from rsstatchem.core.report_xlsx import export_xlsx

        choices = ["Each table on its own worksheet", "All output on one worksheet"]
        choice, ok = QInputDialog.getItem(self, "Export Output to Excel", "Layout:", choices,
                                          0, False)
        if not ok:
            return
        path = self._ask_path("Export Output to Excel", "Excel workbook (*.xlsx)", ".xlsx")
        if path:
            export_xlsx(self.doc, path, "single" if choice == choices[1] else "sheets",
                        self.look)
            self._done(path)

    def export_html(self) -> None:
        path = self._ask_path("Export Output to HTML", "Web page (*.html)", ".html")
        if path:
            Path(path).write_text(report.document_html(self.doc, look=self.look),
                                  encoding="utf-8")
            self._done(path)

    def export_text(self) -> None:
        path = self._ask_path("Export Output to Text", "Text (*.txt)", ".txt")
        if path:
            text = "\n\n".join(report.item_text(s) for s in self.doc.sections)
            Path(path).write_text(text, encoding="utf-8")
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
                                         standalone=False, look=self.look))
        doc.print_(printer)
        self._done(path)

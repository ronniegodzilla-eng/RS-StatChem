"""Project Explorer (Origin's PE): the folder tree on top, the windows of the active folder
below in a details list (Name, View, Type, Size, Comments). Hidden windows are listed in grey;
double-click shows and activates a window. Drag windows from the list onto a folder to move
them, or a folder onto another folder to nest it. Folder and window commands are on the
context menus and call back into the main window.
"""
from __future__ import annotations

from PySide6.QtCore import QMimeData, Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (QAbstractItemView, QHeaderView, QMenu, QSplitter, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from rsstatchem.ui.origin.icons import icon
from rsstatchem.ui.origin.model import (Folder, GraphDoc, MatrixBook, Notes, Project, Workbook,
                                        Worksheet)

ROLE = Qt.ItemDataRole.UserRole
TYPES = {"book": "Workbook", "graph": "Graph", "notes": "Notes", "matrix": "Matrix"}
ICONS = {"book": "e_book", "graph": "e_graph", "notes": "e_notes", "matrix": "e_matrix"}
MIME_WINDOWS = "application/x-rsstatchem-origin-windows"
MIME_FOLDER = "application/x-rsstatchem-origin-folder"


def window_size(obj) -> str:
    if isinstance(obj, MatrixBook):
        ms = obj.active_sheet()
        return (f"{ms.n_rows}×{ms.n_cols}" if ms is not None else "") + \
            (f", {len(obj.sheets)} sheets" if len(obj.sheets) > 1 else "")
    if isinstance(obj, Workbook):
        ws = next((s for s in obj.sheets if isinstance(s, Worksheet)), None)
        sheets = len(obj.sheets)
        if ws is None:
            return f"{sheets} sheet(s)"
        return f"{len(ws.columns)}×{max(ws.data_rows(), 0)}" + \
            (f", {sheets} sheets" if sheets > 1 else "")
    if isinstance(obj, GraphDoc):
        n = sum(len(layer.plots) for layer in obj.graph.layers)
        return f"{len(obj.graph.layers)} layer(s), {n} plot(s)"
    if isinstance(obj, Notes):
        return f"{len(obj.text)} chars"
    return ""


class _WindowList(QTreeWidget):
    """The window list; its items can be dragged onto a folder of the folder tree."""

    def __init__(self):
        super().__init__()
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)

    def mimeTypes(self) -> list[str]:  # noqa: N802
        return [MIME_WINDOWS]

    def mimeData(self, items) -> QMimeData:  # noqa: N802
        mime = QMimeData()
        names = [i.data(0, ROLE) for i in items]
        mime.setData(MIME_WINDOWS, "\n".join(names).encode("utf-8"))
        mime.setText("\n".join(names))
        return mime


class _FolderTree(QTreeWidget):
    """The folder tree: accepts windows (moved into the folder under the mouse) and folders
    (moved, with their contents, below the target folder) — Origin's PE drag and drop."""

    def __init__(self, explorer: "ProjectExplorer"):
        super().__init__()
        self.explorer = explorer
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.setDropIndicatorShown(True)

    def mimeTypes(self) -> list[str]:  # noqa: N802
        return [MIME_FOLDER, MIME_WINDOWS]

    def mimeData(self, items) -> QMimeData:  # noqa: N802
        mime = QMimeData()
        if items:
            mime.setData(MIME_FOLDER, str(items[0].data(0, ROLE)).encode("ascii"))
        return mime

    def _target(self, event) -> Folder | None:
        item = self.itemAt(event.position().toPoint())
        return self.explorer.folder_by_id(item.data(0, ROLE)) if item is not None else None

    def _accepts(self, event) -> bool:
        target = self._target(event)
        mime = event.mimeData()
        if target is None:
            return False
        if mime.hasFormat(MIME_WINDOWS):
            return True
        if mime.hasFormat(MIME_FOLDER):
            folder = self.explorer.folder_by_id(int(bytes(mime.data(MIME_FOLDER)).decode()))
            return folder is not None and folder is not self.explorer.project.root and \
                target not in list(folder.walk())
        return False

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        mime = event.mimeData()
        if mime.hasFormat(MIME_WINDOWS) or mime.hasFormat(MIME_FOLDER):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if self._accepts(event):
            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802
        if not self._accepts(event):
            event.ignore()
            return
        target = self._target(event)
        mime = event.mimeData()
        if mime.hasFormat(MIME_WINDOWS):
            names = [n for n in bytes(mime.data(MIME_WINDOWS)).decode("utf-8").split("\n") if n]
            self.explorer.drop_windows(names, target)
        else:
            folder = self.explorer.folder_by_id(int(bytes(mime.data(MIME_FOLDER)).decode()))
            self.explorer.drop_folder(folder, target)
        event.setDropAction(Qt.DropAction.MoveAction)
        event.accept()


class ProjectExplorer(QWidget):
    folder_changed = Signal(object)     # Folder
    window_activated = Signal(str)      # window name (double-click)

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.project: Project | None = None
        self.current: Folder | None = None
        self.folders = _FolderTree(self)
        self.folders.setHeaderHidden(True)
        self.folders.setObjectName("peFolders")
        self.folders.itemSelectionChanged.connect(self._folder_selected)
        self.folders.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.folders.customContextMenuRequested.connect(self._folder_menu)
        self.folders.itemChanged.connect(self._folder_renamed)
        self.windows = _WindowList()
        self.windows.setObjectName("peWindows")
        self.windows.setHeaderLabels(["Name", "View", "Type", "Size", "Comments"])
        self.windows.setRootIsDecorated(False)
        self.windows.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.windows.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.windows.header().setStretchLastSection(True)
        self.windows.itemDoubleClicked.connect(self._activate)
        self.windows.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.windows.customContextMenuRequested.connect(self._window_menu)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.folders)
        split.addWidget(self.windows)
        split.setSizes([120, 240])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(split)

    # ------------------------------------------------------------------ build
    def set_project(self, project: Project) -> None:
        self.project = project
        self.current = project.root
        self.rebuild()

    def rebuild(self) -> None:
        if self.project is None:
            return
        self.folders.blockSignals(True)
        self.folders.clear()
        selected = None

        def add(folder: Folder, parent):
            nonlocal selected
            item = QTreeWidgetItem([folder.name])
            item.setIcon(0, icon("e_folder"))
            item.setData(0, ROLE, id(folder))
            if folder is not self.project.root:
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            if parent is None:
                self.folders.addTopLevelItem(item)
            else:
                parent.addChild(item)
            if folder is self.current:
                selected = item
            for child in folder.folders:
                add(child, item)
        add(self.project.root, None)
        self.folders.expandAll()
        if selected is not None:
            self.folders.setCurrentItem(selected)
        self.folders.blockSignals(False)
        self.fill_windows()

    def fill_windows(self) -> None:
        self.windows.clear()
        if self.project is None or self.current is None:
            return
        active = self.main.active_name() if self.main is not None else None
        for name in self.current.windows:
            obj = self.project.windows.get(name)
            if obj is None:
                continue
            hidden = name in self.project.hidden
            comment = getattr(obj, "long_name", "") or ""
            item = QTreeWidgetItem([name, "Hidden" if hidden else "Normal", TYPES[obj.kind],
                                    window_size(obj), comment])
            item.setIcon(0, icon(ICONS[obj.kind]))
            item.setData(0, ROLE, name)
            if hidden:
                for c in range(5):
                    item.setForeground(c, QBrush(QColor("#9a9a9a")))
            if name == active:
                f = item.font(0)
                f.setBold(True)
                item.setFont(0, f)
            self.windows.addTopLevelItem(item)

    def folder_by_id(self, fid) -> Folder | None:
        return next((f for f in self.project.root.walk() if id(f) == fid), None) \
            if self.project else None

    # ------------------------------------------------------------------ drag and drop
    def drop_windows(self, names: list[str], folder: Folder) -> None:
        """Windows dragged from the list onto a folder."""
        names = [n for n in names if self.project is not None and n in self.project.windows]
        if names and self.main is not None:
            self.main.move_windows(names, folder)

    def drop_folder(self, folder: Folder, target: Folder) -> None:
        """A folder dragged onto another folder becomes its subfolder."""
        if self.project is None or folder is self.project.root or \
                target in list(folder.walk()):
            return
        parent = self.project.root.parent_of(folder)
        if parent is None or parent is target:
            return
        parent.folders.remove(folder)
        target.folders.append(folder)
        self.rebuild()
        if self.main is not None:
            self.main.set_modified(True)

    # ------------------------------------------------------------------ events
    def _folder_selected(self) -> None:
        item = self.folders.currentItem()
        folder = self.folder_by_id(item.data(0, ROLE)) if item else None
        if folder is not None and folder is not self.current:
            self.current = folder
            self.fill_windows()
            self.folder_changed.emit(folder)

    def _folder_renamed(self, item, _col) -> None:
        folder = self.folder_by_id(item.data(0, ROLE))
        name = item.text(0).strip()
        if folder is not None and name and name != folder.name:
            folder.name = name
            if self.main is not None:
                self.main.set_modified(True)

    def _activate(self, item) -> None:
        self.window_activated.emit(item.data(0, ROLE))

    def selected_windows(self) -> list[str]:
        return [i.data(0, ROLE) for i in self.windows.selectedItems()]

    def _folder_menu(self, pos) -> None:
        if self.main is None:
            return
        menu = QMenu(self)
        menu.addAction(icon("o_new_folder"), "New Folder", self.main.new_folder)
        item = self.folders.itemAt(pos)
        folder = self.folder_by_id(item.data(0, ROLE)) if item else None
        if folder is not None and folder is not self.project.root:
            menu.addAction("Rename", lambda: self.folders.editItem(item, 0))
            menu.addAction("Delete Folder", lambda: self.main.delete_folder(folder))
        menu.addSeparator()
        new = menu.addMenu("New Window")
        new.addAction(icon("o_new_book"), "Workbook", self.main.new_book)
        new.addAction(icon("o_new_matrix"), "Matrix", self.main.new_matrix)
        new.addAction(icon("o_new_graph"), "Graph", self.main.new_graph)
        new.addAction(icon("o_new_notes"), "Notes", self.main.new_notes)
        menu.exec(self.folders.viewport().mapToGlobal(pos))

    def _window_menu(self, pos) -> None:
        if self.main is None:
            return
        item = self.windows.itemAt(pos)
        menu = QMenu(self)
        if item is not None:
            name = item.data(0, ROLE)
            names = self.selected_windows() or [name]
            menu.addAction("Show", lambda: self.main.show_window(name))
            menu.addAction("Hide", lambda: [self.main.hide_window(n) for n in names])
            menu.addAction("Rename...", lambda: self.main.rename_window(name))
            menu.addAction("Duplicate", lambda: self.main.duplicate_window(name))
            move = menu.addMenu("Move to Folder")
            for f in self.project.root.walk():
                act = move.addAction(f.name, lambda f=f: self.main.move_windows(names, f))
                act.setEnabled(f is not self.current)
            menu.addSeparator()
            menu.addAction("Delete", lambda: self.main.delete_windows(names, ask=True))
        else:
            menu.addAction(icon("o_new_book"), "New Workbook", self.main.new_book)
            menu.addAction(icon("o_new_matrix"), "New Matrix", self.main.new_matrix)
            menu.addAction(icon("o_new_graph"), "New Graph", self.main.new_graph)
            menu.addAction(icon("o_new_notes"), "New Notes", self.main.new_notes)
            menu.addAction(icon("o_new_folder"), "New Folder", self.main.new_folder)
        menu.exec(self.windows.viewport().mapToGlobal(pos))

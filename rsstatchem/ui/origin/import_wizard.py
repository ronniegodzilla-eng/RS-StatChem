"""Origin's Import Wizard (File > Import > Import Wizard / Multiple ASCII) and the import
actions of the Origin-style window.

The wizard takes one or more files, shows the raw text and a live preview of the parsed
columns with their Long Name / Units / Comments rows, and offers the ASCII options of
:class:`~rsstatchem.core.analysis.importers.ascii_wizard.AsciiOptions` (delimiter, decimal
separator, main header and subheader lines, partial import, designations, dates). Settings
can be saved as named import filters with a file-name pattern. Several files go into one
book each, one sheet each, or side by side as new columns (Origin's Multiple ASCII modes).
Non-text files (JCAMP-DX, SPC, MATLAB, HDF5/NetCDF, WAV, images) are imported with their
own readers.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
                               QPlainTextEdit, QPushButton, QSpinBox, QSplitter, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from rsstatchem.core.analysis import importers as imp
from rsstatchem.core.analysis.importers import ascii_wizard as aw
from rsstatchem.ui.origin.model import MatrixBook, Workbook, format_cell
from rsstatchem.ui.origin.worksheet_ops import IMPORT_MODES, tables_to_books

PREVIEW_ROWS = 30


# ====================================================================== actions
def _namer(project):
    taken: set[str] = set()

    def unique(base: str) -> str:
        k = 1
        while any(n.lower() == f"{base}{k}".lower() for n in project.windows) or \
                f"{base}{k}".lower() in taken:
            k += 1
        taken.add(f"{base}{k}".lower())
        return f"{base}{k}"
    return unique


def read_paths(paths, options: aw.AsciiOptions | None = None, kind: str | None = None) \
        -> list[tuple[str, list]]:
    return [(Path(p).stem, imp.read_file(p, options, kind)) for p in paths]


def import_paths(main, paths, options: aw.AsciiOptions | None = None, mode: str = "books",
                 kind: str | None = None, show: bool = True) -> list:
    """Import files into new workbooks / matrix books of the main window; returns the new
    model objects. Raises :class:`ImportFailed` with a user message."""
    paths = [str(p) for p in paths]
    if not paths:
        return []
    per_file = read_paths(paths, options, kind)
    objs = tables_to_books(per_file, mode, _namer(main.project), main.default_rows())
    for obj in objs:
        if isinstance(obj, Workbook):
            if not obj.sheets:
                continue
            obj.active = 0
            main.new_book(obj, show=show)
        elif isinstance(obj, MatrixBook):
            main.new_matrix(obj, show=show)
    if hasattr(main, "_remember_dir"):
        main._remember_dir(paths[0])
    n_cols = sum(len(t.columns) for _, ts in per_file for t in ts)
    main.message(f"Imported {len(paths)} file(s): {n_cols} columns → " +
                 ", ".join(o.name for o in objs))
    return objs


def import_dialog(main, kinds=None, title: str = "Import", multiple: bool = True,
                  kind: str | None = None) -> list:
    """File dialog for one reader family, then import (each file into its own book)."""
    start = main._start_dir() if hasattr(main, "_start_dir") else ""
    flt = imp.dialog_filter(kinds)
    if multiple:
        paths, _ = QFileDialog.getOpenFileNames(main, title, start, flt)
    else:
        path, _ = QFileDialog.getOpenFileName(main, title, start, flt)
        paths = [path] if path else []
    if not paths:
        return []
    try:
        return import_paths(main, paths, None, "books", kind)
    except (imp.ImportFailed, ValueError, OSError) as exc:
        main._warn(f"Import failed:\n{exc}", title)
        return []


def wizard(main, paths=None) -> list:
    dlg = ImportWizard(main, paths)
    if not dlg.exec():
        return []
    files, opts, mode = dlg.values()
    try:
        return import_paths(main, files, opts, mode)
    except (imp.ImportFailed, ValueError, OSError) as exc:
        main._warn(f"Import failed:\n{exc}", "Import Wizard")
        return []


# ====================================================================== dialog
class ImportWizard(QDialog):
    def __init__(self, main=None, paths=None, filter_folder=None):
        super().__init__(main)
        self.main = main
        self.filter_folder = filter_folder
        self.setWindowTitle("Import Wizard")
        self.setObjectName("importWizard")
        self._loading = False

        # files
        self.files = QListWidget()
        self.files.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.files.currentRowChanged.connect(lambda _r: self.update_preview())
        add = QPushButton("Add Files...")
        add.clicked.connect(self._add_files)
        rem = QPushButton("Remove")
        rem.clicked.connect(self._remove_files)
        self.mode = QComboBox()
        for k, t in IMPORT_MODES.items():
            self.mode.addItem(t, k)
        fbox = QGroupBox("Data Source")
        fl = QVBoxLayout(fbox)
        fl.addWidget(self.files)
        row = QHBoxLayout()
        row.addWidget(add)
        row.addWidget(rem)
        row.addStretch(1)
        fl.addLayout(row)
        mrow = QHBoxLayout()
        mrow.addWidget(QLabel("Multiple files:"))
        mrow.addWidget(self.mode, 1)
        fl.addLayout(mrow)

        # filters
        self.filter = QComboBox()
        self.filter.addItem("(none)", None)
        self.filter.activated.connect(self._filter_chosen)
        save = QPushButton("Save Filter...")
        save.clicked.connect(self._save_filter)
        delete = QPushButton("Delete")
        delete.clicked.connect(self._delete_filter)
        frow = QHBoxLayout()
        frow.addWidget(QLabel("Import Filter:"))
        frow.addWidget(self.filter, 1)
        frow.addWidget(save)
        frow.addWidget(delete)

        # options
        self.w: dict[str, QWidget] = {}
        form = QFormLayout()

        def combo(key, items):
            c = QComboBox()
            for k, t in items.items():
                c.addItem(t, k)
            c.currentIndexChanged.connect(self.update_preview)
            self.w[key] = c
            return c

        def spin(key, lo, hi):
            s = QSpinBox()
            s.setRange(lo, hi)
            s.valueChanged.connect(self.update_preview)
            self.w[key] = s
            return s

        def edit(key):
            e = QLineEdit()
            e.textChanged.connect(self.update_preview)
            self.w[key] = e
            return e
        form.addRow("Column delimiter:", combo("delimiter", aw.DELIMITERS))
        form.addRow("Custom delimiter:", edit("custom_delimiter"))
        form.addRow("Decimal separator:", combo("decimal", aw.DECIMALS))
        form.addRow("Main header lines (skip):", spin("skip_lines", 0, 100000))
        hl = spin("header_lines", -1, 1000)
        hl.setSpecialValueText("Auto")
        form.addRow("Subheader lines:", hl)
        form.addRow("Long Name from line:", spin("long_name_line", 0, 1000))
        form.addRow("Units from line:", spin("units_line", 0, 1000))
        form.addRow("Comments from line:", spin("comments_from", 0, 1000))
        form.addRow("Comments to line (0 = last):", spin("comments_to", 0, 1000))
        form.addRow("Comment lines start with:", edit("comment_char"))
        form.addRow("Partial import: first row:", spin("first_row", 1, 100_000_000))
        lr = spin("last_row", 0, 100_000_000)
        lr.setSpecialValueText("End")
        form.addRow("Last row:", lr)
        form.addRow("Read every Nth row:", spin("read_every", 1, 100000))
        cols = edit("columns")
        cols.setPlaceholderText("all (e.g. 1:3,5)")
        form.addRow("Columns:", cols)
        form.addRow("Designations (pattern):", combo("designations", {
            "XY": "XYY…", "XYXY": "XYXY…", "XYE": "XY yEr…", "": "All Y"}))
        form.addRow("Date/time text:", combo("dates", aw.DATES))
        trim = QCheckBox("Drop empty columns")
        trim.toggled.connect(self.update_preview)
        self.w["trim_empty"] = trim
        form.addRow("", trim)
        obox = QGroupBox("ASCII Options")
        ol = QVBoxLayout(obox)
        ol.addLayout(frow)
        ol.addLayout(form)

        # preview
        self.raw = QPlainTextEdit()
        self.raw.setReadOnly(True)
        self.raw.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.raw.setObjectName("rawPreview")
        self.table = QTableWidget()
        self.table.setObjectName("parsedPreview")
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        pbox = QGroupBox("Preview")
        pl = QVBoxLayout(pbox)
        sp = QSplitter(Qt.Orientation.Vertical)
        sp.addWidget(self.raw)
        sp.addWidget(self.table)
        sp.setSizes([160, 260])
        pl.addWidget(sp)
        pl.addWidget(self.status)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(fbox)
        ll.addWidget(obox)
        main_split = QSplitter(Qt.Orientation.Horizontal)
        main_split.addWidget(left)
        main_split.addWidget(pbox)
        main_split.setSizes([380, 560])
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                              QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("Import")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(main_split)
        lay.addWidget(bb)
        self.resize(980, 680)
        self._fill_filters()
        self.set_options(aw.AsciiOptions())
        for p in paths or []:
            self.files.addItem(str(p))
        if self.files.count():
            self.files.setCurrentRow(0)
            found = aw.find_filter(self.files.item(0).text(), self.filter_folder)
            if found:
                self.filter.setCurrentIndex(max(self.filter.findText(found[0]), 0))
                self.set_options(found[1])
        self.update_preview()

    # ------------------------------------------------------------------ options
    def set_options(self, o: aw.AsciiOptions) -> None:
        self._loading = True
        d = o.to_dict()
        delim = d["delimiter"]
        known = self.w["delimiter"].findData(delim)
        self.w["delimiter"].setCurrentIndex(known if known >= 0 else 0)
        self.w["custom_delimiter"].setText("" if known >= 0 else delim)
        for key, w in self.w.items():
            if key in ("delimiter", "custom_delimiter"):
                continue
            val = d.get(key)
            if isinstance(w, QComboBox):
                w.setCurrentIndex(max(w.findData(val), 0))
            elif isinstance(w, QSpinBox):
                w.setValue(int(val))
            elif isinstance(w, QCheckBox):
                w.setChecked(bool(val))
            elif isinstance(w, QLineEdit):
                w.setText(str(val or ""))
        self._loading = False

    def options(self) -> aw.AsciiOptions:
        d = {}
        for key, w in self.w.items():
            if isinstance(w, QComboBox):
                d[key] = w.currentData()
            elif isinstance(w, QSpinBox):
                d[key] = w.value()
            elif isinstance(w, QCheckBox):
                d[key] = w.isChecked()
            else:
                d[key] = w.text()
        custom = d.pop("custom_delimiter", "")
        if custom:
            d["delimiter"] = custom.replace("\\t", "\t")
        return aw.AsciiOptions.from_dict(d)

    def values(self):
        files = [self.files.item(i).text() for i in range(self.files.count())]
        return files, self.options(), self.mode.currentData()

    # ------------------------------------------------------------------ files
    def _add_files(self) -> None:
        start = self.main._start_dir() if self.main is not None and \
            hasattr(self.main, "_start_dir") else ""
        paths, _ = QFileDialog.getOpenFileNames(self, "Add Files", start,
                                                imp.dialog_filter())
        for p in paths:
            self.files.addItem(p)
        if paths and self.files.currentRow() < 0:
            self.files.setCurrentRow(0)
        self.update_preview()

    def _remove_files(self) -> None:
        for it in self.files.selectedItems():
            self.files.takeItem(self.files.row(it))
        self.update_preview()

    def current_path(self) -> str | None:
        it = self.files.currentItem() or (self.files.item(0) if self.files.count() else None)
        return it.text() if it is not None else None

    # ------------------------------------------------------------------ filters
    def _fill_filters(self) -> None:
        self.filter.blockSignals(True)
        while self.filter.count() > 1:
            self.filter.removeItem(1)
        try:
            for f in aw.list_filters(self.filter_folder):
                self.filter.addItem(f["name"], f["name"])
        except OSError:
            pass
        self.filter.blockSignals(False)

    def _filter_chosen(self, index: int) -> None:
        name = self.filter.itemData(index)
        if name:
            o = aw.load_filter(name, self.filter_folder)
            if o is not None:
                self.set_options(o)
                self.update_preview()

    def save_filter(self, name: str, pattern: str = "") -> None:
        aw.save_filter(name, self.options(), pattern, self.filter_folder)
        self._fill_filters()
        self.filter.setCurrentIndex(max(self.filter.findText(name), 0))

    def _save_filter(self) -> None:
        name, ok = QInputDialog.getText(self, "Save Import Filter", "Filter name:")
        if not ok or not name.strip():
            return
        path = self.current_path()
        pattern, ok = QInputDialog.getText(
            self, "Save Import Filter", "Apply automatically to file names matching:",
            text=f"*{Path(path).suffix}" if path else "*.dat")
        if ok:
            self.save_filter(name.strip(), pattern.strip())

    def _delete_filter(self) -> None:
        name = self.filter.currentData()
        if name:
            aw.delete_filter(name, self.filter_folder)
            self._fill_filters()

    # ------------------------------------------------------------------ preview
    def update_preview(self, *_args) -> None:
        if self._loading:
            return
        path = self.current_path()
        self.table.clear()
        if not path:
            self.raw.setPlainText("")
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            self.status.setText("Add one or more files.")
            return
        try:
            if imp.kind_of(path) == "ascii":
                lines, table = aw.preview(path, self.options())
                self.raw.setPlainText("\n".join(f"{i + 1:>4}  {ln}" for i, ln in
                                                enumerate(lines[:60])))
                meta = table.meta
                extra = (f"delimiter {aw.DELIMITERS.get(meta['delimiter'], repr(meta['delimiter']))}"
                         f", decimal “{meta['decimal']}”, {meta['subheader_lines']} subheader "
                         "line(s)")
            else:
                table = imp.read_file(path)[0]
                self.raw.setPlainText(f"{Path(path).name}: "
                                      f"{imp.FILTERS[imp.kind_of(path)][0]} file")
                extra = imp.FILTERS[imp.kind_of(path)][0]
        except (imp.ImportFailed, ValueError, OSError, UnicodeError) as exc:
            self.status.setText(f"Cannot preview: {exc}")
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return
        self.show_table(table)
        self.status.setText(f"{len(table.columns)} columns × {table.n_rows} rows — {extra}")

    def show_table(self, table) -> None:
        if table.kind == "matrix":
            data = table.matrix
            r, c = min(PREVIEW_ROWS, data.shape[0]), min(30, data.shape[1])
            self.table.setRowCount(r)
            self.table.setColumnCount(c)
            for i in range(r):
                for j in range(c):
                    self.table.setItem(i, j, QTableWidgetItem(format_cell(float(data[i, j]))))
            return
        cols = table.columns
        head = ["Long Name", "Units", "Comments"]
        n = min(PREVIEW_ROWS, table.n_rows)
        self.table.setColumnCount(len(cols))
        self.table.setRowCount(len(head) + n)
        self.table.setHorizontalHeaderLabels(
            [f"{chr(65 + j) if j < 26 else j + 1}({c.designation})" for j, c in
             enumerate(cols)])
        self.table.setVerticalHeaderLabels(head + [str(i + 1) for i in range(n)])
        for j, c in enumerate(cols):
            for i, text in enumerate((c.long_name, c.units, c.comments)):
                it = QTableWidgetItem(text)
                it.setBackground(Qt.GlobalColor.lightGray)
                self.table.setItem(i, j, it)
            for i in range(n):
                v = c.values[i] if i < len(c.values) else None
                self.table.setItem(len(head) + i, j, QTableWidgetItem(format_cell(v)))

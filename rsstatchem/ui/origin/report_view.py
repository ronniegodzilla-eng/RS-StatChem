"""Origin-style hierarchical report sheet view.

A report is a tree of collapsible sections, like Origin's report sheets: Notes, Input Data,
Parameters, Statistics, ANOVA, Fitted Curves Plot, Residual Plots (a parent section holding
several plots)… Click the ⊟ / ⊞ box of a section header to collapse / expand it (the state is
saved with the project). Right-click a table to copy it (HTML for Word plus tab-separated text
for Excel, like the Statistics module's output), double-click an embedded plot to open it as
a graph window.

The lock button at the top left shows the recalculation mode (Origin's green lock): green =
Auto, yellow = Manual, yellow with a red mark = the source data changed and the report is out
of date, grey = None. Its menu switches the mode, recalculates, or re-opens the dialog with
the report's settings (Change Parameters).
"""
from __future__ import annotations

import base64
import html
import math

from PySide6.QtCore import QMimeData, QSize, Qt
from PySide6.QtGui import QAction, QActionGroup, QColor, QImage, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QFrame, QHBoxLayout,
                               QHeaderView, QLabel, QMenu, QScrollArea, QSizePolicy,
                               QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout,
                               QWidget)

from rsstatchem.core.output import Table
from rsstatchem.core.report import clipboard_html, table_tsv
from rsstatchem.ui.origin.icons import icon
from rsstatchem.ui.origin.model import RECALC_MODES, ReportSheet, to_float

_REPORT_CSS = """
body { font-family: Arial, Helvetica, sans-serif; font-size: 9pt; color: #000; }
.title { font-size: 11pt; font-weight: bold; color: #1f3d7a; margin: 4px 0 8px 0; }
.sec { background: #d9e1f0; color: #000; font-weight: bold; padding: 3px 6px;
       border: 1px solid #a6b4cf; margin-top: 10px; }
table.t { border-collapse: collapse; margin: 2px 0 2px 14px; }
table.t td, table.t th { border: 1px solid #b8b8b8; padding: 2px 8px; }
table.t th { background: #ececec; font-weight: normal; }
td.rh { background: #f5f5f5; }
td.num { text-align: right; }
.note { color: #505050; margin-left: 14px; font-size: 8pt; }
.warn { color: #a05a00; margin-left: 14px; }
"""


def _numeric(v) -> bool:
    return not math.isnan(to_float(v)) or v in ("--",)


def item_table(item: dict) -> Table:
    """A report table item as a :class:`~rsstatchem.core.output.Table` (for the clipboard)."""
    cols = [tuple(c) if isinstance(c, list) else c for c in item["columns"]]
    return Table(item.get("title", ""), cols, [list(r) for r in item["rows"]],
                 footnotes=[n for n in item.get("notes", []) if n])


def report_html(report: ReportSheet, doc=None) -> str:
    """Origin-like HTML of a whole report (images registered on the QTextDocument ``doc``
    when given, else embedded as data URIs)."""
    parts = [f"<html><head><style>{_REPORT_CSS}</style></head><body>"]
    if report.title:
        parts.append(f'<div class="title">{html.escape(report.title)}</div>')
    group = None
    for k, item in enumerate(report.items):
        if item.get("group") and item.get("group") != group:
            parts.append(f'<div class="sec">&#8863; {html.escape(item["group"])}</div>')
        group = item.get("group") or None
        title = html.escape(item.get("title", ""))
        if item["kind"] == "table":
            parts.append(f'<div class="sec">&#8863; {title}</div>')
            parts.append('<table class="t" cellspacing="0">')
            cols = item["columns"]
            if any(isinstance(c, list) for c in cols):
                top, j = [], 0
                while j < len(cols):   # merge equal group titles: "95% Confidence Interval"
                    c = cols[j]
                    head = c[0] if isinstance(c, list) else ""
                    span = 1
                    while j + span < len(cols) and isinstance(cols[j + span], list) and \
                            cols[j + span][0] == head and head:
                        span += 1
                    top.append(f'<th nowrap colspan="{span}">{html.escape(head)}</th>')
                    j += span
                parts.append("<tr>" + "".join(top) + "</tr>")
                cols = [c[1] if isinstance(c, list) else c for c in cols]
            parts.append("<tr>" + "".join(f"<th nowrap>{html.escape(str(c))}</th>"
                                          for c in cols) + "</tr>")
            two_col = len(cols) == 2
            for row in item["rows"]:
                cells = []
                for j, v in enumerate(row):
                    numeric = _numeric(v)
                    cls = "rh" if j == 0 or (j == 1 and not numeric and len(cols) > 2) else \
                        ("num" if numeric and not two_col else "")
                    cells.append(f'<td nowrap class="{cls}">{html.escape(str(v))}</td>')
                parts.append("<tr>" + "".join(cells) + "</tr>")
            parts.append("</table>")
            for note in item.get("notes", []):
                if note:
                    parts.append(f'<div class="note">{html.escape(note)}</div>')
        elif item["kind"] == "text":
            cls = "warn" if item.get("style") == "warning" else "note"
            parts.append(f'<div class="sec">&#8863; {title}</div>'
                         f'<div class="{cls}">{html.escape(item.get("text", ""))}</div>')
        elif item["kind"] == "image":
            width = int(item.get("width", 520))
            if doc is not None:
                from PySide6.QtCore import QUrl
                from PySide6.QtGui import QTextDocument
                url = f"img://{k}"
                image = QImage.fromData(base64.b64decode(item["png"]), "PNG")
                doc.addResource(QTextDocument.ResourceType.ImageResource, QUrl(url), image)
            else:
                url = "data:image/png;base64," + item["png"]
            parts.append(f'<div class="sec">&#8863; {title}</div>'
                         f'<p style="margin-left:14px"><img src="{url}" width="{width}"></p>')
    parts.append("</body></html>")
    return "".join(parts)


def report_text(report: ReportSheet) -> str:
    """Tab-separated text of a whole report (tables, notes and titles)."""
    out = [report.title] if report.title else []
    for item in report.items:
        if item["kind"] == "table":
            out.append(table_tsv(item_table(item)))
        elif item["kind"] == "text":
            out.append(f"{item.get('title', '')}\n{item.get('text', '')}")
        else:
            out.append(f"[{item.get('title', '')}]")
    return "\n\n".join(out)


def copy_items(items: list[dict]) -> None:
    """Put tables on the clipboard as HTML (Word keeps the table) and TSV (Excel cells)."""
    tables = [item_table(i) for i in items if i["kind"] == "table"]
    mime = QMimeData()
    if tables:
        mime.setHtml(clipboard_html(tables))
    mime.setText("\n\n".join(table_tsv(t) for t in tables) if tables else
                 "\n".join(i.get("text", i.get("title", "")) for i in items))
    images = [i for i in items if i["kind"] == "image"]
    if images and not tables:
        mime.setImageData(QImage.fromData(base64.b64decode(images[0]["png"]), "PNG"))
    QApplication.clipboard().setMimeData(mime)


# ====================================================================== widgets
class ReportTable(QTableWidget):
    """A report table at its natural size (no scroll bars, read-only, first column shaded as
    the row header)."""

    def __init__(self, item: dict, view: "ReportView"):
        cols = item["columns"]
        super().__init__(len(item["rows"]), len(cols))
        self.item_ = item
        self.view = view
        self.setObjectName("reportTable")
        grouped = any(isinstance(c, list) for c in cols)
        labels = [(f"{c[0]}\n{c[1]}" if grouped else c[1]) if isinstance(c, list) else
                  (f"\n{c}" if grouped else str(c)) for c in cols]
        self.setHorizontalHeaderLabels(labels)
        self.verticalHeader().hide()
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ContiguousSelection)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setWordWrap(False)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        hh = self.horizontalHeader()
        hh.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        two_col = len(cols) == 2
        for r, row in enumerate(item["rows"]):
            for c, value in enumerate(row):
                cell = QTableWidgetItem(str(value))
                numeric = _numeric(value)
                if c == 0 or (c == 1 and not numeric and len(cols) > 2):
                    cell.setBackground(QColor("#f5f5f5"))
                    cell.setTextAlignment(Qt.AlignmentFlag.AlignLeft |
                                          Qt.AlignmentFlag.AlignVCenter)
                elif numeric and not two_col:
                    cell.setTextAlignment(Qt.AlignmentFlag.AlignRight |
                                          Qt.AlignmentFlag.AlignVCenter)
                self.setItem(r, c, cell)
        self.verticalHeader().setDefaultSectionSize(20)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(
            lambda pos: view.item_menu(item, self.viewport().mapToGlobal(pos)))
        self.fit_size()

    def fit_size(self) -> None:
        self.resizeColumnsToContents()
        self.resizeRowsToContents()
        w = sum(self.columnWidth(c) for c in range(self.columnCount())) + 2 * self.frameWidth()
        h = self.horizontalHeader().sizeHint().height() + self.verticalHeader().length() + \
            2 * self.frameWidth() + 2
        self.setFixedSize(max(w + 2, 60), h)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if not getattr(self, "_fitted", False):   # the stylesheet is applied by now
            self._fitted = True
            self.fit_size()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        from PySide6.QtGui import QKeySequence
        if event.matches(QKeySequence.StandardKey.Copy):
            copy_items([self.item_])
            return
        super().keyPressEvent(event)


class _Header(QWidget):
    """Section header: the ⊟/⊞ box and the bold title (Origin's report tree node)."""

    def __init__(self, title: str, collapsed: bool, level: int, toggled, menu):
        super().__init__()
        self.setObjectName("reportSectionHeader")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2 + 14 * level, 1, 6, 1)
        lay.setSpacing(4)
        self.box = QToolButton()
        self.box.setObjectName("reportToggle")
        self.box.setCheckable(True)
        self.box.setChecked(not collapsed)
        self.box.setAutoRaise(True)
        self.box.setFixedSize(16, 16)
        self.box.toggled.connect(toggled)
        self.box.toggled.connect(self._mark)
        self.label = QLabel(title)
        self.label.setTextFormat(Qt.TextFormat.PlainText)
        self.label.setObjectName("reportSectionTitle")
        lay.addWidget(self.box)
        lay.addWidget(self.label)
        lay.addStretch(1)
        self._mark(not collapsed)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(lambda pos: menu(self.mapToGlobal(pos)))

    def _mark(self, expanded: bool) -> None:
        self.box.setText("⊟" if expanded else "⊞")
        self.box.setToolTip("Collapse" if expanded else "Expand")

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.box.toggle()


class ReportSection(QFrame):
    """One collapsible node: a header and a body (a table, text, plot or child sections)."""

    def __init__(self, title: str, body: QWidget, collapsed: bool, level: int, on_toggle,
                 menu):
        super().__init__()
        self.setObjectName("reportSection")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.body = body
        self.header = _Header(title, collapsed, level, self._toggle, menu)
        self._on_toggle = on_toggle
        lay.addWidget(self.header)
        wrap = QWidget()
        wl = QHBoxLayout(wrap)
        wl.setContentsMargins(20 + 14 * level, 2, 0, 6)
        wl.addWidget(body, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        wl.addStretch(1)
        self.wrap = wrap
        lay.addWidget(wrap)
        wrap.setVisible(not collapsed)

    @property
    def expanded(self) -> bool:
        return self.header.box.isChecked()

    def set_expanded(self, on: bool) -> None:
        self.header.box.setChecked(on)

    def _toggle(self, expanded: bool) -> None:
        self.wrap.setVisible(expanded)
        self._on_toggle(not expanded)


class _Image(QLabel):
    def __init__(self, item: dict, view: "ReportView"):
        super().__init__()
        self.item_ = item
        self.view = view
        image = QImage.fromData(base64.b64decode(item["png"]), "PNG")
        pix = QPixmap.fromImage(image)
        width = int(item.get("width", 520))
        if pix.width() > 0:
            pix = pix.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(pix)
        self.setObjectName("reportImage")
        self.setFrameShape(QFrame.Shape.Box)
        self.setToolTip("Double-click to open as a graph window; right-click to copy."
                        if item.get("graph") else "Right-click to copy the picture.")
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(
            lambda pos: view.item_menu(item, self.mapToGlobal(pos)))

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if self.item_.get("graph"):
            self.view.open_graph(self.item_)


class ReportView(QScrollArea):
    """Report sheet tab of a workbook window (``book_window`` may be None in tests)."""

    def __init__(self, report: ReportSheet, book_window=None):
        super().__init__()
        self.report = report
        self.book_window = book_window
        self.sections: list[ReportSection] = []
        self.setObjectName("originReport")
        self.setWidgetResizable(True)
        self._sig = None
        self.rebuild()

    @property
    def main(self):
        return getattr(self.book_window, "main", None)

    # ------------------------------------------------------------------ build
    def _signature(self):
        return (id(self.report.items), len(self.report.items), self.report.mode,
                self.report.dirty, self.report.title)

    def refresh(self) -> None:
        """Rebuild when the report content changed (recalculation) or its lock state."""
        if self._signature() != self._sig:
            self.rebuild()

    def rebuild(self) -> None:
        self._sig = self._signature()
        bar = self.verticalScrollBar().value() if self.widget() is not None else 0
        page = QWidget()
        page.setObjectName("reportPage")
        lay = QVBoxLayout(page)
        lay.setContentsMargins(8, 6, 8, 12)
        lay.setSpacing(3)
        lay.addWidget(self._top_bar())
        self.sections = []
        for group, indices in self.report.sections():
            if group:
                inner = QWidget()
                il = QVBoxLayout(inner)
                il.setContentsMargins(0, 0, 0, 0)
                il.setSpacing(3)
                for k in indices:
                    il.addWidget(self._item_section(self.report.items[k], 1))
                collapsed = bool(self.report.items[indices[0]].get("group_collapsed"))
                sec = ReportSection(group, inner, collapsed, 0,
                                    lambda c, f=indices[0]: self._group_toggled(f, c),
                                    lambda pos, ids=indices: self.section_menu(
                                        [self.report.items[i] for i in ids], pos))
                self.sections.append(sec)
                lay.addWidget(sec)
            else:
                lay.addWidget(self._item_section(self.report.items[indices[0]], 0))
        lay.addStretch(1)
        self.setWidget(page)
        self.verticalScrollBar().setValue(bar)

    def _top_bar(self) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 4)
        self.lock = QToolButton()
        self.lock.setObjectName("reportLock")
        self.lock.setIcon(icon(self.lock_icon()))
        self.lock.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.lock.setAutoRaise(True)
        self.lock.setIconSize(QSize(22, 22))
        self.lock.setToolTip(self.lock_tip())
        self.lock.setMenu(self._lock_menu())
        self.lock.setVisible(bool(self.report.op))
        title = QLabel(self.report.title)
        title.setTextFormat(Qt.TextFormat.PlainText)
        title.setObjectName("reportTitle")
        title.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        title.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        lay.addWidget(self.lock)
        lay.addWidget(title)
        lay.addStretch(1)
        for text, on in (("Expand All", True), ("Collapse All", False)):
            b = QToolButton()
            b.setText(text)
            b.setAutoRaise(True)
            b.clicked.connect(lambda _=False, v=on: self.expand_all(v))
            lay.addWidget(b)
        return w

    def lock_icon(self) -> str:
        if not self.report.op or self.report.mode == "none":
            return "lock_none"
        if self.report.dirty:
            return "lock_dirty"
        return "lock_auto" if self.report.mode == "auto" else "lock_manual"

    def lock_tip(self) -> str:
        if not self.report.op:
            return ""
        text = f"Recalculate: {RECALC_MODES.get(self.report.mode, 'None')}"
        if self.report.dirty:
            text += " — the source data changed; choose Recalculate to update this report."
        return text

    def _lock_menu(self) -> QMenu:
        menu = QMenu(self)
        sub = menu.addMenu("Recalculate &Mode")
        group = QActionGroup(sub)
        for key, text in RECALC_MODES.items():
            act = QAction(text, sub, checkable=True)
            act.setChecked(self.report.mode == key)
            act.triggered.connect(lambda _=False, k=key: self.set_mode(k))
            group.addAction(act)
            sub.addAction(act)
        recalc = menu.addAction("&Recalculate", self.recalculate)
        recalc.setEnabled(bool(self.report.op))
        change = menu.addAction("&Change Parameters...", self.change_parameters)
        change.setEnabled(bool(self.report.op))
        return menu

    def _item_section(self, item: dict, level: int) -> ReportSection:
        kind = item["kind"]
        if kind == "table":
            body = QWidget()
            bl = QVBoxLayout(body)
            bl.setContentsMargins(0, 0, 0, 0)
            bl.setSpacing(2)
            bl.addWidget(ReportTable(item, self))
            for note in item.get("notes", []):
                if note:
                    lab = QLabel(note)
                    lab.setTextFormat(Qt.TextFormat.PlainText)
                    lab.setObjectName("reportNote")
                    lab.setWordWrap(True)
                    lab.setMaximumWidth(720)
                    bl.addWidget(lab)
        elif kind == "image":
            body = _Image(item, self)
        else:
            body = QLabel(item.get("text", ""))
            body.setTextFormat(Qt.TextFormat.PlainText)
            body.setObjectName("reportWarning" if item.get("style") == "warning" else
                               "reportNote")
            body.setWordWrap(True)
            body.setMaximumWidth(720)
            body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        sec = ReportSection(item.get("title", ""), body, bool(item.get("collapsed")), level,
                            lambda c, it=item: it.__setitem__("collapsed", c),
                            lambda pos, it=item: self.item_menu(it, pos))
        self.sections.append(sec)
        return sec

    def _group_toggled(self, first: int, collapsed: bool) -> None:
        # a group's state is kept on its first item (survives saving and recalculation)
        self.report.items[first]["group_collapsed"] = collapsed

    # ------------------------------------------------------------------ commands
    def expand_all(self, on: bool = True) -> None:
        for sec in self.sections:
            sec.set_expanded(on)

    def toPlainText(self) -> str:  # noqa: N802 - same API as the old QTextBrowser view
        return report_text(self.report)

    def copy_all(self) -> None:
        mime = QMimeData()
        mime.setHtml(report_html(self.report))
        mime.setText(report_text(self.report))
        QApplication.clipboard().setMimeData(mime)

    def item_menu(self, item: dict, pos) -> None:
        menu = QMenu(self)
        if item["kind"] == "table":
            menu.addAction("&Copy Table", lambda: copy_items([item]))
        elif item["kind"] == "image":
            menu.addAction("&Copy Picture", lambda: copy_items([item]))
            if item.get("graph"):
                menu.addAction("&Open as Graph Window", lambda: self.open_graph(item))
        else:
            menu.addAction("&Copy", lambda: copy_items([item]))
        menu.addAction("Copy &All (report)", self.copy_all)
        menu.addSeparator()
        menu.addAction("&Expand All", lambda: self.expand_all(True))
        menu.addAction("C&ollapse All", lambda: self.expand_all(False))
        if self.report.op:
            menu.addSeparator()
            menu.addMenu(self._lock_menu()).setText("Recalculate")
        menu.exec(pos)

    def section_menu(self, items: list[dict], pos) -> None:
        menu = QMenu(self)
        menu.addAction("&Copy Tables", lambda: copy_items(items))
        menu.addAction("Copy &All (report)", self.copy_all)
        menu.addSeparator()
        menu.addAction("&Expand All", lambda: self.expand_all(True))
        menu.addAction("C&ollapse All", lambda: self.expand_all(False))
        menu.exec(pos)

    def set_mode(self, mode: str) -> None:
        if self.main is not None and self.book_window is not None:
            self.main.set_report_mode(self.book_window.book, self.report, mode)
        else:
            self.report.op["mode"] = mode
            self.rebuild()

    def recalculate(self) -> None:
        if self.main is not None and self.book_window is not None:
            self.main.recalculate_report(self.book_window.book, self.report)

    def change_parameters(self) -> None:
        if self.main is not None and self.book_window is not None:
            self.main.change_report_parameters(self.book_window.book, self.report)

    def open_graph(self, item: dict) -> None:
        if self.main is not None and self.book_window is not None:
            self.main.open_report_graph(self.book_window.book, self.report, item)

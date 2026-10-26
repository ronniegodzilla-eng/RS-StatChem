"""Building blocks of the RSM views: report panes (Design-Expert-style grids with p-value
colours, copyable as Word HTML + Excel TSV), graph panes (copy / export as image), the
Factors Tool (axis assignment and sliders) and a few small helpers."""
from __future__ import annotations

import html
import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import QByteArray, QMimeData, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QFrame,
                               QGridLayout, QHBoxLayout, QLabel, QMenu, QSizePolicy, QSlider,
                               QTextBrowser, QToolButton, QVBoxLayout, QWidget)

from rsstatchem.core.output import Image, Section, Table, Text
from rsstatchem.core.report import clipboard_html, item_text
from rsstatchem.ui.dx import icons, style


# ---------------------------------------------------------------------------- helpers
def esc(text) -> str:
    return html.escape("" if text is None else str(text))


def looks_numeric(text: str) -> bool:
    t = str(text).strip().replace(",", "").replace("%", "").lstrip("<> +")
    if not t:
        return False
    try:
        float(t)
        return True
    except ValueError:
        return False


def parse_p(text: str) -> float | None:
    t = str(text).strip()
    if not t:
        return None
    if t.startswith("<"):
        return 0.0
    try:
        return float(t)
    except ValueError:
        return None


def copy_items(items: list) -> None:
    """Put tables/notes on the clipboard as Word-ready HTML plus tab-separated text."""
    items = [i for i in items if isinstance(i, (Table, Text, Image, Section))]
    mime = QMimeData()
    mime.setHtml(clipboard_html(items))
    mime.setText("\n\n".join(item_text(i) for i in items))
    QApplication.clipboard().setMimeData(mime)


def copy_png(png: bytes) -> None:
    img = QImage.fromData(QByteArray(png), "PNG")
    mime = QMimeData()
    mime.setImageData(img)
    mime.setData("image/png", QByteArray(png))
    QApplication.clipboard().setMimeData(mime)


def save_png(png: bytes, path: str) -> None:
    suffix = Path(path).suffix.lower()
    if suffix in ("", ".png"):
        Path(path if suffix else path + ".png").write_bytes(png)
        return
    img = QImage.fromData(QByteArray(png), "PNG")
    if suffix in (".jpg", ".jpeg"):
        bg = QImage(img.size(), QImage.Format.Format_RGB32)
        bg.fill(Qt.GlobalColor.white)
        p = QPainter(bg)
        p.drawImage(0, 0, img)
        p.end()
        img = bg
    if not img.save(path):
        raise OSError(f"Could not write {path}.")


def page_title(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("pageTitle")
    return lab


def hint(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("hint")
    lab.setWordWrap(True)
    return lab


def annotation(text: str) -> Text:
    return Text("", text, style="annot")


# ---------------------------------------------------------------------------- report pane
def table_to_html(t: Table) -> str:
    cols = [" ".join(c) if isinstance(c, tuple) else str(c) for c in t.columns]
    pcols = {i for i, c in enumerate(cols) if c.lower().startswith("p-value")
             or c.lower() == "prob > f"}
    remark_col = len(cols) - 1 if cols and cols[-1] == "" else None
    out = []
    if t.title:
        out.append(f'<p class="sub">{esc(t.title)}</p>')
    headed = any(c for c in cols)
    out.append(f'<table class="{"dx" if headed else "eq"}" cellspacing="0" cellpadding="0">')
    if headed:
        out.append("<tr>" + "".join(f'<th align="{"left" if i < t.row_header_cols else "center"}">'
                                    f"{esc(c)}</th>" for i, c in enumerate(cols)) + "</tr>")
    for row in t.rows:
        cells = ["" if v is None else str(v) for v in row]
        remark = cells[remark_col] if remark_col is not None and remark_col < len(cells) else ""
        row_style = ""
        if remark == "Suggested":
            row_style = "background:#FFF4C2; font-weight:bold; text-decoration: underline;"
        elif remark == "Aliased":
            row_style = "color:#8A8F96;"
        tds = []
        for i, text in enumerate(cells):
            style_txt = row_style
            if i in pcols:
                p = parse_p(text)
                if p is not None and remark != "Aliased":
                    style_txt += f"color:{style.p_color(p)}; font-weight:bold;"
            if text in ("significant",):
                style_txt += "color:#B00020;"
            align = "left" if i < t.row_header_cols or not looks_numeric(text) else "right"
            bold = "font-weight:bold;" if i < t.row_header_cols and t.row_header_cols else ""
            lead = len(text) - len(text.lstrip(" "))
            cell = "&nbsp;" * (2 * lead) + esc(text.lstrip(" "))
            if not headed:  # equations: plain lines, no grid
                style_txt += "border:none; padding:1px 6px;"
            tds.append(f'<td align="{align}" style="{style_txt}{bold}">{cell}</td>')
        out.append("<tr>" + "".join(tds) + "</tr>")
    out.append("</table>")
    for f in t.footnotes:
        out.append(f'<p class="foot">{esc(f)}</p>')
    return "\n".join(out)


def items_to_html(items: list, title: str = "", annotate: bool = True) -> str:
    parts = [f'<p class="title">{esc(title)}</p>'] if title else []
    for it in items:
        if isinstance(it, Table):
            parts.append(table_to_html(it))
        elif isinstance(it, Text):
            if it.style == "log":
                continue
            if it.style == "annot":
                if annotate:
                    parts.append(f'<p class="annot">{esc(it.text).replace(chr(10), "<br>")}</p>')
                continue
            if it.title:
                parts.append(f'<p class="sub">{esc(it.title)}</p>')
            cls = "warning" if it.style == "warning" else "note"
            parts.append(f'<p class="{cls}">{esc(it.text).replace(chr(10), "<br>")}</p>')
    return (f"<html><head><style>{style.REPORT_CSS}</style></head><body>"
            + "\n".join(parts) + "</body></html>")


class ReportView(QTextBrowser):
    """Read-only report of tables and notes; right-click copies a table or everything."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.items: list = []
        self.title = ""
        self.annotate = True
        self.setOpenLinks(False)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)

    def set_items(self, items: list, title: str = "") -> None:
        self.items, self.title = list(items), title
        bar = self.verticalScrollBar().value()
        self.setHtml(items_to_html(self.items, title, self.annotate))
        self.verticalScrollBar().setValue(bar)

    def set_annotate(self, on: bool) -> None:
        self.annotate = on
        self.set_items(self.items, self.title)

    def tables(self) -> list[Table]:
        return [i for i in self.items if isinstance(i, Table)]

    def copy_all(self) -> None:
        items = [i for i in self.items if not (isinstance(i, Text) and i.style == "annot")]
        sec = Section(self.title or "Report", items)
        copy_items([sec])

    def _menu(self, pos) -> None:
        menu = QMenu(self)
        if self.textCursor().hasSelection():
            menu.addAction("Copy Selection", self.copy)
        menu.addAction(icons.icon("copy"), "Copy All Tables (Word / Excel)", self.copy_all)
        tables = self.tables()
        if tables:
            sub = menu.addMenu("Copy Table")
            for t in tables:
                sub.addAction(t.title or "Table", lambda _=False, tt=t: copy_items([tt]))
        menu.addSeparator()
        menu.addAction("Select All", self.selectAll)
        menu.exec(self.mapToGlobal(pos))


# ---------------------------------------------------------------------------- graph pane
class GraphView(QWidget):
    """Shows a rendered graph (PNG) scaled to fit; right-click to copy or export it."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.png: bytes | None = None
        self._pix: QPixmap | None = None
        self.message = ""
        self.export_name = "graph"
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)

    def set_png(self, png: bytes | None, name: str = "graph") -> None:
        self.png, self.message, self.export_name = png, "", name
        self._pix = None
        if png:
            pix = QPixmap()
            pix.loadFromData(QByteArray(png), "PNG")
            self._pix = pix
        self.update()

    def set_message(self, text: str) -> None:
        self.png, self._pix, self.message = None, None, text
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(640, 480)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.GlobalColor.white)
        if self._pix is not None and not self._pix.isNull():
            target = self._pix.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
            x = (self.width() - target.width()) // 2
            y = (self.height() - target.height()) // 2
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            p.drawPixmap(x, y, target.width(), target.height(), self._pix)
        elif self.message:
            p.setPen(Qt.GlobalColor.darkGray)
            p.drawText(self.rect().adjusted(20, 20, -20, -20),
                       Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self.message)
        p.end()

    def copy_graph(self) -> None:
        if self.png:
            copy_png(self.png)

    def export_graph(self) -> None:
        if not self.png:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Graph As", f"{self.export_name}.png",
                                              "PNG image (*.png);;JPEG image (*.jpg);;"
                                              "Bitmap (*.bmp);;TIFF image (*.tif)")
        if path:
            save_png(self.png, path)

    def _menu(self, pos) -> None:
        menu = QMenu(self)
        a = menu.addAction(icons.icon("copy"), "Copy Graph", self.copy_graph)
        b = menu.addAction(icons.icon("export"), "Export Graph As...", self.export_graph)
        a.setEnabled(bool(self.png))
        b.setEnabled(bool(self.png))
        menu.exec(self.mapToGlobal(pos))


# ---------------------------------------------------------------------------- tab strip
class TabStrip(QFrame):
    """Design-Expert's row of large tab buttons across the top of an analysis/optimization."""

    changed = Signal(int)

    def __init__(self, tabs: list[tuple[str, str]], parent=None):
        super().__init__(parent)
        self.setObjectName("tabStrip")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 3, 6, 3)
        lay.setSpacing(3)
        self.buttons: list[QToolButton] = []
        for i, (icon_name, text) in enumerate(tabs):
            b = QToolButton()
            b.setObjectName("tabButton")
            b.setText(text)
            b.setIcon(icons.icon(icon_name))
            b.setIconSize(QSize(22, 22))
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            b.setCheckable(True)
            b.setAutoExclusive(True)
            b.clicked.connect(lambda _=False, k=i: self.set_current(k))
            lay.addWidget(b)
            self.buttons.append(b)
        lay.addStretch(1)
        self.extra = QHBoxLayout()
        lay.addLayout(self.extra)
        self.current = -1

    def set_current(self, i: int, emit: bool = True) -> None:
        self.buttons[i].setChecked(True)
        if i != self.current:
            self.current = i
            if emit:
                self.changed.emit(i)

    def labels(self) -> list[str]:
        return [b.text() for b in self.buttons]


# ---------------------------------------------------------------------------- factors tool
SLIDER_STEPS = 1000


class FactorsTool(QFrame):
    """Axis assignment and sliders for the factors not on an axis (Design-Expert's Factors
    Tool). ``n_axes`` = 0 (point prediction), 1 (one factor), 2 (contour) or 3 (ternary)."""

    changed = Signal()

    def __init__(self, space, n_axes: int = 2, parent=None):
        super().__init__(parent)
        self.space = space
        self.n_axes = n_axes
        self._busy = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        title = QLabel("Factors Tool")
        title.setStyleSheet(f"color:{style.BLUE}; font-weight:600;")
        lay.addWidget(title)
        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(4)
        self.grid.setVerticalSpacing(2)
        lay.addLayout(self.grid)
        lay.addStretch(1)
        self.axis_boxes: list[QComboBox] = []
        self.sliders: list[QSlider | None] = []
        self.edits: list[QWidget] = []
        from rsstatchem.core.doe import report

        self._point = report.default_point(space)
        numeric = [j for j, f in enumerate(space.factors) if not f.categoric]
        default_axes = numeric[: n_axes] if not space.mixture else list(range(min(n_axes,
                                                                                  space.k)))
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self.changed.emit)
        for j, f in enumerate(space.factors):
            r = 2 * j
            name = QLabel(f"{f.letter}:{f.name}" + (f" ({f.units})" if f.units else ""))
            name.setStyleSheet("font-weight:600;")
            self.grid.addWidget(name, r, 0, 1, 2)
            box = QComboBox()
            box.addItem("—", -1)
            for a in range(n_axes):
                box.addItem(f"X{a + 1} axis", a)
            if j in default_axes:
                box.setCurrentIndex(default_axes.index(j) + 1)
            box.setVisible(n_axes > 0)
            box.currentIndexChanged.connect(lambda _=0, jj=j: self._axis_changed(jj))
            self.grid.addWidget(box, r, 2)
            self.axis_boxes.append(box)
            if f.categoric:
                combo = QComboBox()
                for i in range(len(f.levels)):
                    combo.addItem(f.level_label(i), float(f.levels[i]))
                combo.currentIndexChanged.connect(lambda _=0, jj=j: self._combo_changed(jj))
                self.grid.addWidget(combo, r + 1, 0, 1, 3)
                self.sliders.append(None)
                self.edits.append(combo)
            else:
                sl = QSlider(Qt.Orientation.Horizontal)
                sl.setRange(0, SLIDER_STEPS)
                sl.valueChanged.connect(lambda v, jj=j: self._slider_changed(jj, v))
                spin = QDoubleSpinBox()
                lo, hi = (f.low, f.high)
                spin.setDecimals(4 if (hi - lo) < 1 else 3 if (hi - lo) < 10 else 2)
                spin.setRange(lo, hi)
                spin.setSingleStep((hi - lo) / 20 or 0.1)
                spin.setKeyboardTracking(False)
                spin.valueChanged.connect(lambda v, jj=j: self._spin_changed(jj, v))
                spin.setMinimumWidth(78)
                self.grid.addWidget(sl, r + 1, 0, 1, 2)
                self.grid.addWidget(spin, r + 1, 2)
                self.sliders.append(sl)
                self.edits.append(spin)
        self.grid.setColumnStretch(0, 1)
        self._axis_of = [b.currentData() for b in self.axis_boxes]
        self._sync_widgets()
        self._update_enabled()

    # ------------------------------------------------------------------ state
    def axes(self) -> list[int]:
        out = [-1] * self.n_axes
        for j, box in enumerate(self.axis_boxes):
            a = box.currentData()
            if a is not None and a >= 0:
                out[a] = j
        return [j for j in out if j >= 0]

    def point(self) -> np.ndarray:
        return self._point.copy()

    def set_point(self, x) -> None:
        self._point = np.array(x, float)
        self._sync_widgets()
        self.changed.emit()

    def set_axes(self, axes: list[int]) -> None:
        self._busy = True
        for j, box in enumerate(self.axis_boxes):
            box.setCurrentIndex(axes.index(j) + 1 if j in axes else 0)
        self._busy = False
        self._axis_of = [b.currentData() for b in self.axis_boxes]
        self._update_enabled()
        self.changed.emit()

    def set_active_axes(self, n: int) -> None:
        """Show only X1..Xn (a contour needs 2 axes, a cube 3, a perturbation plot none);
        factors not on an axis get their slider back."""
        n = min(n, self.n_axes)
        self._busy = True
        for j, box in enumerate(self.axis_boxes):
            cur = self._axis_of[j] if self._axis_of[j] is not None else -1
            box.clear()
            box.addItem("—", -1)
            for a in range(n):
                box.addItem(f"X{a + 1} axis", a)
            if not 0 <= cur < n:
                cur = -1
            box.setCurrentIndex(cur + 1)
            self._axis_of[j] = cur
            box.setVisible(n > 0)
        used = {a for a in self._axis_of if a >= 0}
        # numeric factors first; categoric ones only when nothing else is left
        order = sorted(range(self.space.k), key=lambda j: self.space.factors[j].categoric)
        for a in range(n):
            if a in used:
                continue
            for j in order:
                if self._axis_of[j] < 0:
                    self._axis_of[j] = a
                    self.axis_boxes[j].setCurrentIndex(a + 1)
                    break
        self._busy = False
        self._update_enabled()

    # ------------------------------------------------------------------ internals
    def _sync_widgets(self) -> None:
        self._busy = True
        for j, f in enumerate(self.space.factors):
            v = self._point[j]
            w = self.edits[j]
            if f.categoric:
                lv = np.asarray(f.levels, float)
                w.setCurrentIndex(int(np.argmin(np.abs(lv - v))))
            else:
                span = (f.high - f.low) or 1.0
                self.sliders[j].setValue(int(round((v - f.low) / span * SLIDER_STEPS)))
                w.setValue(float(v))
        self._busy = False

    def _update_enabled(self) -> None:
        on_axis = set(self.axes())
        for j in range(self.space.k):
            free = j not in on_axis
            if self.sliders[j] is not None:
                self.sliders[j].setEnabled(free)
            self.edits[j].setEnabled(free)

    def _axis_changed(self, j: int) -> None:
        if self._busy:
            return
        a = self.axis_boxes[j].currentData()
        old = self._axis_of[j]
        if a is not None and a >= 0:  # swap: the factor that had this axis takes j's old one
            self._busy = True
            for i, box in enumerate(self.axis_boxes):
                if i != j and self._axis_of[i] == a:
                    box.setCurrentIndex(old + 1 if old >= 0 else 0)
                    self._axis_of[i] = old
            self._busy = False
        self._axis_of[j] = a if a is not None else -1
        self._update_enabled()
        self.changed.emit()

    def _set_value(self, j: int, v: float) -> None:
        self._point[j] = v
        sp = self.space
        if sp.mixture:  # keep the blend on the mixture total: rescale the free components
            others = [i for i in range(sp.k) if i != j]
            rest = sp.total - v
            cur = self._point[others]
            lo = sp.lower[others]
            free = cur - lo
            room = rest - lo.sum()
            if room >= 0:
                share = free / free.sum() if free.sum() > 0 else np.full(len(others),
                                                                         1 / len(others))
                self._point[others] = lo + room * share
        self._sync_widgets()
        self._timer.start()

    def _slider_changed(self, j: int, v: int) -> None:
        if self._busy:
            return
        f = self.space.factors[j]
        self._set_value(j, f.low + (f.high - f.low) * v / SLIDER_STEPS)

    def _spin_changed(self, j: int, v: float) -> None:
        if self._busy:
            return
        self._set_value(j, float(v))

    def _combo_changed(self, j: int) -> None:
        if self._busy:
            return
        self._point[j] = float(self.edits[j].currentData())
        self._timer.start()


def fmt(x, sig: int = 4) -> str:
    from rsstatchem.core.doe.report import num

    return num(x, sig)


def finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def tool_button(icon_name: str, text: str, checkable: bool = True) -> QToolButton:
    b = QToolButton()
    b.setText(text)
    b.setIcon(icons.icon(icon_name))
    b.setIconSize(QSize(20, 20))
    b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    b.setCheckable(checkable)
    b.setObjectName("tabButton")
    return b


def make_action(parent, text: str, slot=None, shortcut=None, icon_name: str | None = None,
                checkable: bool = False, tip: str = "") -> QAction:
    act = QAction(text, parent)
    if icon_name:
        act.setIcon(icons.icon(icon_name))
    if shortcut:
        act.setShortcut(shortcut)
    if checkable:
        act.setCheckable(True)
    if tip:
        act.setStatusTip(tip)
        act.setToolTip(tip)
    if slot is not None:
        act.triggered.connect(slot)
    return act

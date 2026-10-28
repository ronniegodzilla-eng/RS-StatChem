"""Plot windows (ImageJ ``PlotWindow``): profile plots, gel lanes, fits, distributions.

The canvas draws the :class:`~rsstatchem.core.imaging.profile.PlotData` series with ImageJ-like
axes and ticks. Buttons as ImageJ 1.54: **List** (table of the values), **Data >>** (Save
Data..., Copy 1st Data Set, Copy All Data), **More >>** (Set Range..., Reset Range, Log X,
Log Y, Legend, Save Plot as PNG...) and **Live** (profile plots follow the selection while it
changes). Drag a rectangle to zoom, double-click to reset. :func:`sync_window` is Analyze >
Tools > Synchronize Windows (selections, slices and zoom).
"""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QAction, QColor, QFont, QGuiApplication, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QHBoxLayout, QInputDialog, QListWidget,
                               QMainWindow, QMenu, QPushButton, QVBoxLayout, QWidget)

from rsstatchem.core.imaging.profile import PlotData
from rsstatchem.ui.imagej import windows as W


def valid(widget) -> bool:
    """False once Qt has deleted the widget (session callbacks may outlive windows)."""
    try:
        import shiboken6
        return shiboken6.isValid(widget)
    except ImportError:  # pragma: no cover
        return True


def nice_ticks(lo: float, hi: float, n: int = 6) -> list[float]:
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return [lo]
    raw = (hi - lo) / max(n, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 5, 10) if m * mag >= raw), default=raw)
    start = math.ceil(lo / step) * step
    ticks = []
    v = start
    while v <= hi + step * 1e-9:
        ticks.append(0.0 if abs(v) < step * 1e-9 else v)
        v += step
    return ticks


def _fmt(v: float) -> str:
    if v == int(v) and abs(v) < 1e7:
        return str(int(v))
    return f"{v:.4g}"


class PlotCanvas(QWidget):
    LEFT, RIGHT, TOP, BOTTOM = 60, 18, 22, 42

    def __init__(self, plot: PlotData, parent=None):
        super().__init__(parent)
        self.plot = plot
        self.view = None            # (xmin, xmax, ymin, ymax) when zoomed
        self._drag = None
        self.setMinimumSize(420, 240)
        self.setMouseTracking(True)
        self.status_cb = None

    def limits(self):
        if self.view is not None:
            return self.view
        xmin, xmax, ymin, ymax = self.plot.limits()
        if self.plot.y_range is None:
            pad = (ymax - ymin) * 0.05
            ymin, ymax = ymin - (0 if ymin == 0 else pad), ymax + pad
        return xmin, xmax, ymin, ymax

    def frame(self) -> QRectF:
        return QRectF(self.LEFT, self.TOP, max(10, self.width() - self.LEFT - self.RIGHT),
                      max(10, self.height() - self.TOP - self.BOTTOM))

    def to_screen(self, x, y):
        xmin, xmax, ymin, ymax = self.limits()
        f = self.frame()
        if self.plot.log_x:
            x = np.log10(np.maximum(x, 1e-300))
            xmin, xmax = math.log10(max(xmin, 1e-300)), math.log10(max(xmax, 1e-300))
        if self.plot.log_y:
            y = np.log10(np.maximum(y, 1e-300))
            ymin, ymax = math.log10(max(ymin, 1e-300)), math.log10(max(ymax, 1e-300))
        sx = f.left() + (np.asarray(x, float) - xmin) / ((xmax - xmin) or 1) * f.width()
        sy = f.bottom() - (np.asarray(y, float) - ymin) / ((ymax - ymin) or 1) * f.height()
        return sx, sy

    def to_data(self, px, py):
        xmin, xmax, ymin, ymax = self.limits()
        f = self.frame()
        return (xmin + (px - f.left()) / f.width() * (xmax - xmin),
                ymin + (f.bottom() - py) / f.height() * (ymax - ymin))

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(255, 255, 255))
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        f = self.frame()
        xmin, xmax, ymin, ymax = self.limits()
        p.setFont(QFont("SansSerif", 9))
        p.setPen(QPen(QColor(0, 0, 0), 1))
        p.drawRect(f)
        for t in nice_ticks(xmin, xmax):
            sx, _ = self.to_screen([t], [ymin])
            p.drawLine(QPointF(sx[0], f.bottom()), QPointF(sx[0], f.bottom() + 4))
            p.drawText(QRectF(sx[0] - 30, f.bottom() + 5, 60, 14),
                       Qt.AlignmentFlag.AlignHCenter, _fmt(t))
        for t in nice_ticks(ymin, ymax):
            _, sy = self.to_screen([xmin], [t])
            p.drawLine(QPointF(f.left() - 4, sy[0]), QPointF(f.left(), sy[0]))
            p.drawText(QRectF(0, sy[0] - 7, f.left() - 6, 14),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, _fmt(t))
        p.drawText(QRectF(f.left(), f.bottom() + 20, f.width(), 16),
                   Qt.AlignmentFlag.AlignHCenter, self.plot.x_label)
        p.save()
        p.translate(12, f.center().y())
        p.rotate(-90)
        p.drawText(QRectF(-f.height() / 2, -10, f.height(), 14), Qt.AlignmentFlag.AlignHCenter,
                   self.plot.y_label)
        p.restore()
        p.setClipRect(f)
        for s in self.plot.series:
            sx, sy = self.to_screen(s["x"], s["y"])
            color = QColor(s.get("color") or "#000000")
            p.setPen(QPen(color, 1))
            style = s.get("style", "line")
            ok = np.isfinite(sx) & np.isfinite(sy)
            if style in ("bar", "separated bar"):
                xs = s["x"]
                w = (xs[1] - xs[0]) if len(xs) > 1 else 1.0
                for x, y in zip(xs, s["y"]):
                    x0, y0 = self.to_screen([x], [0.0])
                    x1, y1 = self.to_screen([x + w], [y])
                    r = QRectF(QPointF(x0[0], y1[0]), QPointF(x1[0], y0[0])).normalized()
                    if style == "separated bar":
                        r.adjust(1, 0, -1, 0)
                    p.fillRect(r, QColor(200, 200, 200))
                    p.drawRect(r)
                continue
            if style in ("line", "connected circle"):
                poly = QPolygonF([QPointF(a, b) for a, b, k in zip(sx, sy, ok) if k])
                p.drawPolyline(poly)
            if style in ("circle", "connected circle"):
                for a, b, k in zip(sx, sy, ok):
                    if k:
                        p.drawEllipse(QPointF(a, b), 3, 3)
        p.setClipping(False)
        p.setPen(QPen(QColor(0, 0, 0)))
        for x, y, text in self.plot.labels:
            sx, sy = self.to_screen([x], [y])
            p.drawText(QPointF(min(max(sx[0], f.left() + 4), f.right() - 100),
                               max(sy[0], f.top() + 12)), text)
        if self.plot.legend:
            yy = f.top() + 14
            for s in self.plot.series:
                if s.get("label"):
                    p.setPen(QPen(QColor(s.get("color") or "#000000")))
                    p.drawText(QPointF(f.right() - 120, yy), s["label"])
                    yy += 14
        if self._drag is not None:
            p.setPen(QPen(QColor(0, 0, 255), 1, Qt.PenStyle.DashLine))
            p.drawRect(QRectF(self._drag[0], self._drag[1]).normalized())
        p.end()

    def mousePressEvent(self, e) -> None:  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = [e.position(), e.position()]

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        if self._drag is not None:
            self._drag[1] = e.position()
            self.update()
        if self.status_cb is not None:
            x, y = self.to_data(e.position().x(), e.position().y())
            self.status_cb(f"X={x:.4g}, Y={y:.4g}")

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if self._drag is None:
            return
        a, b = self._drag
        self._drag = None
        if abs(a.x() - b.x()) > 5 and abs(a.y() - b.y()) > 5:
            x0, y0 = self.to_data(min(a.x(), b.x()), max(a.y(), b.y()))
            x1, y1 = self.to_data(max(a.x(), b.x()), min(a.y(), b.y()))
            self.view = (x0, x1, y0, y1)
        self.update()

    def mouseDoubleClickEvent(self, e) -> None:  # noqa: N802
        self.view = None
        self.update()

    def zoom(self, factor: float) -> None:
        xmin, xmax, ymin, ymax = self.limits()
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        hw, hh = (xmax - xmin) / 2 / factor, (ymax - ymin) / 2 / factor
        self.view = (cx - hw, cx + hw, cy - hh, cy + hh)
        self.update()


class PlotWindow(QMainWindow):
    def __init__(self, bar, plot: PlotData):
        super().__init__()
        self.bar = bar
        self.plot = plot
        self.setWindowTitle(plot.title)
        central = QWidget(self)
        lay = QVBoxLayout(central)
        lay.setContentsMargins(2, 2, 2, 2)
        self.canvas = PlotCanvas(plot, central)
        if bar is not None and hasattr(bar, "show_status"):
            self.canvas.status_cb = bar.show_status
        lay.addWidget(self.canvas, 1)
        row = QHBoxLayout()
        self.list_button = QPushButton("List", central)
        self.list_button.clicked.connect(self.list_values)
        row.addWidget(self.list_button)
        self.data_button = QPushButton("Data >>", central)
        dm = QMenu(self.data_button)
        self._act(dm, "Save Data...", self.save_data)
        self._act(dm, "Copy 1st Data Set", lambda: self.copy_data(False))
        self._act(dm, "Copy All Data", lambda: self.copy_data(True))
        self.data_button.setMenu(dm)
        row.addWidget(self.data_button)
        self.more_button = QPushButton("More >>", central)
        mm = QMenu(self.more_button)
        self._act(mm, "Set Range...", self.set_range)
        self._act(mm, "Reset Range", self.reset_range)
        self._act(mm, "Zoom In", lambda: self.canvas.zoom(1.5))
        self._act(mm, "Zoom Out", lambda: self.canvas.zoom(1 / 1.5))
        self._act(mm, "Log X Axis", lambda: self.toggle_log("x"))
        self._act(mm, "Log Y Axis", lambda: self.toggle_log("y"))
        self._act(mm, "Legend", self.toggle_legend)
        self._act(mm, "Save Plot as PNG...", self.save_png)
        self.more_button.setMenu(mm)
        row.addWidget(self.more_button)
        self.live = QCheckBox("Live", central)
        self.live.setEnabled(plot.source.get("kind") == "profile")
        self.live.toggled.connect(self._live_toggled)
        row.addWidget(self.live)
        row.addStretch(1)
        lay.addLayout(row)
        self.setCentralWidget(central)
        self.resize(560, 340)
        self._listening = False

    def _act(self, menu, text, slot):
        a = QAction(text, self)
        a.triggered.connect(lambda _=False: slot())
        menu.addAction(a)
        return a

    @property
    def session(self):
        return getattr(self.bar, "session", None)

    def set_plot(self, plot: PlotData) -> None:
        self.plot = plot
        self.canvas.plot = plot
        self.setWindowTitle(plot.title)
        self.canvas.update()

    # ------------------------------------------------------------------ buttons
    def list_values(self):
        rt = self.plot.table("Plot Values")
        if self.session is not None:
            self.session.show_table(rt)
        return rt

    def data_text(self, all_series: bool = True) -> str:
        rt = self.plot.table()
        if not all_series:
            for h in list(rt.headings):
                if h not in ("X", "Y"):
                    rt.delete_column(h)
        return rt.to_text("\t", row_numbers=False)

    def copy_data(self, all_series: bool = True) -> str:
        text = self.data_text(all_series)
        QGuiApplication.clipboard().setText(text)
        return text

    def save_data(self, path: str | None = None) -> str | None:
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Save Data", "Values.csv",
                                                  "CSV (*.csv);;Text (*.txt)")
            if not path:
                return None
        self.plot.table().save(path, row_numbers=False)
        return path

    def save_png(self, path: str | None = None) -> str | None:
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Save Plot", self.plot.title + ".png",
                                                  "PNG (*.png)")
            if not path:
                return None
        self.canvas.grab().save(path)
        return path

    def set_range(self, xmin=None, xmax=None, ymin=None, ymax=None) -> None:
        if xmin is None:
            cur = self.canvas.limits()
            text, ok = QInputDialog.getText(self, "Set Range", "X min, X max, Y min, Y max:",
                                            text=", ".join(f"{v:.4g}" for v in cur))
            if not ok:
                return
            xmin, xmax, ymin, ymax = (float(t) for t in text.split(","))
        self.plot.x_range = (float(xmin), float(xmax))
        self.plot.y_range = (float(ymin), float(ymax))
        self.canvas.view = None
        self.canvas.update()

    def reset_range(self) -> None:
        self.plot.x_range = self.plot.y_range = None
        self.canvas.view = None
        self.canvas.update()

    def toggle_log(self, axis: str) -> None:
        if axis == "x":
            self.plot.log_x = not self.plot.log_x
        else:
            self.plot.log_y = not self.plot.log_y
        self.canvas.update()

    def toggle_legend(self) -> None:
        self.plot.legend = not self.plot.legend
        self.canvas.update()

    # ------------------------------------------------------------------ live
    def _live_toggled(self, on: bool) -> None:
        s = self.session
        if s is None:
            return
        if on and not self._listening:
            s.on("image_updated", self._image_updated)
            self._listening = True
        elif not on and self._listening:
            s.off("image_updated", self._image_updated)
            self._listening = False

    def _image_updated(self, doc) -> None:
        if not valid(self):
            return
        src = self.plot.source
        if src.get("kind") != "profile" or doc.id != src.get("image_id"):
            return
        from rsstatchem.core.imaging.profile import profile_plot
        from rsstatchem.core.imaging.roi import shapes as S
        roi = doc.roi
        if roi is None or not (roi.is_line() or isinstance(roi, S.RectRoi)):
            return
        try:
            plot = profile_plot(doc, roi, vertical=src.get("vertical", False))
        except ValueError:
            return
        plot.x_range, plot.y_range = self.plot.x_range, self.plot.y_range
        self.set_plot(plot)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._live_toggled(False)
        event.accept()
        W.unregister(self)


def show_plot(bar, plot: PlotData) -> PlotWindow:
    """Show ``plot`` (a Live profile window of the same image is updated instead)."""
    for w in list(W.WINDOWS):
        if isinstance(w, PlotWindow) and w.live.isChecked() and \
                w.plot.source.get("image_id") == plot.source.get("image_id") and \
                plot.source.get("kind") == "profile":
            w.set_plot(plot)
            w.show()
            return w
    win = PlotWindow(bar, plot)
    W.register(win)
    win.show()
    return win


# ---------------------------------------------------------------------------------- sync
class SyncWindow(QMainWindow):
    """Analyze > Tools > Synchronize Windows: mirrors the selection, the stack position and
    the zoom of the active image onto the checked images."""

    def __init__(self, bar):
        super().__init__()
        self.bar = bar
        self.setWindowTitle("Synchronize Windows")
        central = QWidget(self)
        lay = QVBoxLayout(central)
        self.list = QListWidget(central)
        lay.addWidget(self.list, 1)
        self.sync_roi = QCheckBox("Sync selections (ROIs)", central)
        self.sync_roi.setChecked(True)
        self.sync_slice = QCheckBox("Sync slices (z, t)", central)
        self.sync_slice.setChecked(True)
        self.sync_zoom = QCheckBox("Sync zoom", central)
        for cb in (self.sync_roi, self.sync_slice, self.sync_zoom):
            lay.addWidget(cb)
        row = QHBoxLayout()
        sync_all = QPushButton("Synchronize All", central)
        sync_all.clicked.connect(self.check_all)
        unsync = QPushButton("Unsynchronize All", central)
        unsync.clicked.connect(self.uncheck_all)
        row.addWidget(sync_all)
        row.addWidget(unsync)
        lay.addLayout(row)
        self.setCentralWidget(central)
        self._session = bar.session
        self.refresh()
        self._busy = False
        self._session = bar.session
        bar.session.on("image_updated", self._updated)
        bar.session.on("image_added", self._images_changed)
        bar.session.on("image_closed", self._images_changed)
        self.resize(300, 320)

    def _images_changed(self, _doc) -> None:
        if valid(self):
            self.refresh()

    def refresh(self) -> None:
        checked = set(self.checked_titles()) if self.list.count() else set()
        self.list.clear()
        for im in self._session.images:
            self.list.addItem(im.title)
            it = self.list.item(self.list.count() - 1)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if im.title in checked
                             else Qt.CheckState.Unchecked)

    def checked_titles(self) -> list[str]:
        return [self.list.item(i).text() for i in range(self.list.count())
                if self.list.item(i).checkState() == Qt.CheckState.Checked]

    def check_all(self) -> None:
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(Qt.CheckState.Checked)

    def uncheck_all(self) -> None:
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(Qt.CheckState.Unchecked)

    def _updated(self, doc) -> None:
        if self._busy or not valid(self):
            return
        titles = self.checked_titles()
        if doc.title not in titles:
            return
        s = self._session
        self._busy = True
        try:
            for im in s.images:
                if im is doc or im.title not in titles:
                    continue
                changed = False
                if self.sync_roi.isChecked():
                    im.roi = doc.roi.copy() if doc.roi is not None else None
                    changed = True
                if self.sync_slice.isChecked() and im.stack_size == doc.stack_size and \
                        im.position != doc.position:
                    im.set_position(*doc.position)
                    changed = True
                if self.sync_zoom.isChecked():
                    src, dst = W.window_for(doc), W.window_for(im)
                    if src is not None and dst is not None:
                        dst.canvas.set_magnification(src.canvas.magnification)
                if changed:
                    s.image_updated(im)
        finally:
            self._busy = False

    def closeEvent(self, event) -> None:  # noqa: N802
        s = self._session
        s.off("image_updated", self._updated)
        s.off("image_added", self._images_changed)
        s.off("image_closed", self._images_changed)
        event.accept()
        W.unregister(self)


def sync_window(bar) -> SyncWindow:
    for w in W.WINDOWS:
        if isinstance(w, SyncWindow) and w.bar is bar:
            return w
    win = SyncWindow(bar)
    W.register(win)
    return win

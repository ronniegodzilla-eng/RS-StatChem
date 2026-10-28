"""Cell Counter (in the style of Fiji's Cell Counter; written for RS-StatChem, no Fiji code).

Counter types (Type 1..8, more can be added, renamable), "Initialize" binds the counter to the
active image, then left clicks on the image add a marker of the current type (Delete mode:
remove the nearest marker). Markers are a multi-point selection with counters and stack
positions (``PointRoi.counters``/``positions``), so they can be measured, saved as .roi and
shown by the canvas. Buttons: Add / Remove type, Rename, Initialize, Delete mode, Results
(counts per type and slice), Reset, Export / Import markers (CSV x, y, slice, type),
Show numbers, Measure.

:class:`CounterModel` is the headless part (tests use it directly).
"""
from __future__ import annotations

import csv

import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QColor, QFont, QPen
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QFileDialog, QGridLayout, QInputDialog,
                               QLabel, QMainWindow, QPushButton, QRadioButton, QVBoxLayout,
                               QWidget)

from rsstatchem.core.imaging.results import ResultsTable
from rsstatchem.core.imaging.roi import shapes as S
from rsstatchem.ui.imagej import windows as W
from rsstatchem.ui.imagej.frames.plot_window import valid

TYPE_COLORS = ("#00ffff", "#ff00ff", "#ffff00", "#ff8000", "#00ff00", "#0080ff", "#ff0000",
               "#8000ff", "#ffffff", "#808080")


class CounterModel:
    """Markers of one image: (x, y, slice (1-based), type (0-based))."""

    def __init__(self, n_types: int = 8):
        self.names = [f"Type {i + 1}" for i in range(n_types)]
        self.current = 0
        self.markers: list[tuple[float, float, int, int]] = []
        self.image = None

    def add_type(self) -> None:
        self.names.append(f"Type {len(self.names) + 1}")

    def remove_type(self) -> None:
        if len(self.names) > 1:
            k = len(self.names) - 1
            self.markers = [m for m in self.markers if m[3] != k]
            self.names.pop()
            self.current = min(self.current, len(self.names) - 1)

    def add_marker(self, x: float, y: float, slice_: int = 1, type_: int | None = None):
        t = self.current if type_ is None else type_
        self.markers.append((float(x), float(y), int(slice_), int(t)))

    def delete_nearest(self, x: float, y: float, slice_: int = 1, max_dist: float = 10.0):
        best, bd = None, max_dist
        for i, (mx, my, ms, _t) in enumerate(self.markers):
            if ms != slice_:
                continue
            d = float(np.hypot(mx - x, my - y))
            if d <= bd:
                best, bd = i, d
        if best is not None:
            del self.markers[best]
        return best

    def counts(self, slice_: int | None = None) -> list[int]:
        out = [0] * len(self.names)
        for _x, _y, s, t in self.markers:
            if (slice_ is None or s == slice_) and t < len(out):
                out[t] += 1
        return out

    def reset(self) -> None:
        self.markers.clear()

    def to_roi(self) -> S.PointRoi | None:
        if not self.markers:
            return None
        arr = np.asarray(self.markers, float)
        return S.PointRoi(arr[:, 0], arr[:, 1], counters=arr[:, 3].astype(int),
                          positions=arr[:, 2].astype(int), name="Counter")

    def from_roi(self, roi) -> None:
        self.markers.clear()
        n = len(roi.xs)
        cs = roi.counters if roi.counters is not None else np.zeros(n, int)
        ps = roi.positions if roi.positions is not None else np.ones(n, int)
        for x, y, p, c in zip(roi.xs, roi.ys, ps, cs):
            while int(c) >= len(self.names):
                self.add_type()
            self.markers.append((float(x), float(y), int(p) or 1, int(c)))

    def results(self, n_slices: int = 1) -> ResultsTable:
        rt = ResultsTable("Counts")
        slices = range(1, n_slices + 1) if n_slices > 1 else [None]
        for s in slices:
            row = {"Slice": s} if s is not None else {}
            for name, c in zip(self.names, self.counts(s)):
                row[name] = c
            rt.add_row(row)
        if n_slices > 1:
            row = {"Slice": 0}
            for name, c in zip(self.names, self.counts(None)):
                row[name] = c
            rt.add_row(row, label="Total")
        return rt

    def export_csv(self, path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["X", "Y", "Slice", "Type"])
            for x, y, s, t in self.markers:
                w.writerow([f"{x:g}", f"{y:g}", s, t + 1])

    def import_csv(self, path) -> None:
        self.markers.clear()
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                t = int(row["Type"]) - 1
                while t >= len(self.names):
                    self.add_type()
                self.markers.append((float(row["X"]), float(row["Y"]), int(row["Slice"]), t))


class _ClickFilter(QObject):
    def __init__(self, win):
        super().__init__(win)
        self.win = win

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if not valid(self.win):
            return False
        if event.type() == QEvent.Type.MouseButtonPress and \
                event.button() == Qt.MouseButton.LeftButton and self.win.counting.isChecked():
            x, y = obj.screen_to_image(event.position().x(), event.position().y())
            self.win.click(x, y)
            return True
        return False


class CellCounterWindow(QMainWindow):
    def __init__(self, bar):
        super().__init__()
        self.bar = bar
        self.model = CounterModel()
        self.setWindowTitle("Cell Counter")
        central = QWidget(self)
        lay = QVBoxLayout(central)
        self.grid_holder = QWidget(central)
        self.grid = QGridLayout(self.grid_holder)
        lay.addWidget(self.grid_holder)
        self.group = QButtonGroup(self)
        self.group.idClicked.connect(self._type_clicked)
        self.count_labels: list[QLabel] = []
        self.counting = QCheckBox("Counting (click on the image)", central)
        self.counting.setChecked(True)
        lay.addWidget(self.counting)
        self.delete_mode = QCheckBox("Delete mode", central)
        lay.addWidget(self.delete_mode)
        self.show_numbers = QCheckBox("Show numbers", central)
        self.show_numbers.toggled.connect(lambda _on: self._repaint())
        lay.addWidget(self.show_numbers)
        for text, slot in (("Initialize", self.initialize), ("Add Type", self.add_type),
                           ("Remove Type", self.remove_type), ("Rename Type...", self.rename),
                           ("Results", self.results), ("Measure", self.measure),
                           ("Reset", self.reset), ("Export Markers...", self.export),
                           ("Import Markers...", self.import_)):
            b = QPushButton(text, central)
            b.clicked.connect(lambda _=False, s=slot: s())
            lay.addWidget(b)
        lay.addStretch(1)
        self.setCentralWidget(central)
        self._filter = _ClickFilter(self)
        self._canvas = None
        self._rebuild_types()
        self.resize(240, 520)

    # ------------------------------------------------------------------ types
    def _rebuild_types(self) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        for b in list(self.group.buttons()):
            self.group.removeButton(b)
        self.count_labels = []
        counts = self.model.counts()
        for i, name in enumerate(self.model.names):
            rb = QRadioButton(name, self.grid_holder)
            rb.setStyleSheet(f"color: {TYPE_COLORS[i % len(TYPE_COLORS)]};")
            rb.setChecked(i == self.model.current)
            self.group.addButton(rb, i)
            lb = QLabel(str(counts[i]), self.grid_holder)
            self.grid.addWidget(rb, i, 0)
            self.grid.addWidget(lb, i, 1)
            self.count_labels.append(lb)

    def _type_clicked(self, i: int) -> None:
        self.model.current = i

    def _update_counts(self) -> None:
        for lb, c in zip(self.count_labels, self.model.counts()):
            lb.setText(str(c))

    def add_type(self) -> None:
        self.model.add_type()
        self._rebuild_types()

    def remove_type(self) -> None:
        self.model.remove_type()
        self._rebuild_types()
        self._sync()

    def rename(self, name: str | None = None) -> None:
        if name is None:
            name, ok = QInputDialog.getText(self, "Rename", "Name:",
                                            text=self.model.names[self.model.current])
            if not ok:
                return
        self.model.names[self.model.current] = name
        self._rebuild_types()

    # ------------------------------------------------------------------ image binding
    def initialize(self, image=None) -> None:
        image = image or self.bar.session.active
        if image is None:
            self.bar.error("There are no images open", "Cell Counter")
            return
        self.model.image = image
        if isinstance(image.roi, S.PointRoi):
            self.model.from_roi(image.roi)
            self._rebuild_types()
        win = W.window_for(image)
        if self._canvas is not None and valid(self._canvas):
            self._canvas.removeEventFilter(self._filter)
        self._canvas = None
        if win is not None:
            self._canvas = win.canvas
            self._canvas.installEventFilter(self._filter)
            self._canvas.paint_hooks.append(self._paint)
        self.setWindowTitle(f"Cell Counter - {image.title}")

    def click(self, x: float, y: float) -> None:
        im = self.model.image
        if im is None:
            return
        sl = im.current_index + 1
        if self.delete_mode.isChecked():
            self.model.delete_nearest(x, y, sl)
        else:
            self.model.add_marker(int(x), int(y), sl)
        self._sync()

    def _sync(self) -> None:
        im = self.model.image
        if im is not None:
            im.roi = self.model.to_roi()
            self.bar.session.image_updated(im)
        self._update_counts()

    def _repaint(self) -> None:
        if self._canvas is not None and valid(self._canvas):
            self._canvas.update()

    def _paint(self, p, canvas) -> None:
        if not valid(self) or not self.show_numbers.isChecked():
            return
        im = self.model.image
        if im is None:
            return
        p.setFont(QFont("SansSerif", 8))
        for x, y, s, t in self.model.markers:
            if s != im.current_index + 1:
                continue
            sx, sy = canvas.image_to_screen(x + 0.5, y + 0.5)
            p.setPen(QPen(QColor(TYPE_COLORS[t % len(TYPE_COLORS)])))
            p.drawText(int(sx + 4), int(sy - 4), str(t + 1))

    # ------------------------------------------------------------------ outputs
    def results(self):
        im = self.model.image
        rt = self.model.results(im.stack_size if im is not None else 1)
        self.bar.session.show_table(rt)
        return rt

    def measure(self):
        im = self.model.image
        if im is None or not self.model.markers:
            return None
        im.roi = self.model.to_roi()
        return self.bar.run_command("Measure", "")

    def reset(self) -> None:
        self.model.reset()
        self._sync()

    def export(self, path: str | None = None):
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Export Markers", "CellCounter.csv",
                                                  "CSV (*.csv)")
            if not path:
                return None
        self.model.export_csv(path)
        return path

    def import_(self, path: str | None = None):
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "Import Markers", "", "CSV (*.csv)")
            if not path:
                return None
        self.model.import_csv(path)
        self._rebuild_types()
        self._sync()
        return path

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._canvas is not None and valid(self._canvas):
            self._canvas.removeEventFilter(self._filter)
        event.accept()
        W.unregister(self)


def counter_window(bar, create: bool = True):
    for w in W.WINDOWS:
        if isinstance(w, CellCounterWindow) and w.bar is bar and valid(w):
            return w
    if not create:
        return None
    win = CellCounterWindow(bar)
    W.register(win)
    if bar.session.active is not None:
        win.initialize(bar.session.active)
    return win

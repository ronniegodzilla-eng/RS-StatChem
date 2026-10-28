"""Digitizer (Origin's Tools > Digitizer): pick data points from an image of a graph.

Open a scanned or published graph image, click the two X reference points and the two Y
reference points on the axes and type their values (linear or log axes), then click the
data points (or let Auto Trace follow a dark curve). The points go to a new workbook as an
X/Y column pair. The calibration math is :class:`rsstatchem.core.graph.gadgets.Calibration`.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QImage, QPen, QPixmap
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout, QGraphicsScene,
                               QGraphicsView, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QPushButton, QRadioButton, QTableWidget, QTableWidgetItem,
                               QVBoxLayout)

from rsstatchem.core.graph.gadgets import Calibration, auto_trace

REFS = ("X1", "X2", "Y1", "Y2")
_COLORS = {"X1": "#d62728", "X2": "#d62728", "Y1": "#1f77b4", "Y2": "#1f77b4"}


class _View(QGraphicsView):
    def __init__(self, dlg: "DigitizerDialog"):
        super().__init__(dlg)
        self.dlg = dlg
        self.setScene(QGraphicsScene(self))
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMinimumSize(520, 400)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self.dlg.image is not None:
            pt = self.mapToScene(event.position().toPoint())
            self.dlg.clicked(pt.x(), pt.y())
            return
        super().mousePressEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802
        f = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(f, f)


class DigitizerDialog(QDialog):
    def __init__(self, main=None, parent=None):
        super().__init__(parent or main)
        self.main = main
        self.setWindowTitle("Digitizer")
        self.resize(980, 640)
        self.image: QImage | None = None
        self.refs: dict[str, tuple[float, float]] = {}
        self.points: list[tuple[float, float]] = []
        self.view = _View(self)

        mode_box = QGroupBox("Click sets")
        ml = QVBoxLayout(mode_box)
        self.mode_group = QButtonGroup(self)
        for k, name in enumerate(REFS + ("Points",)):
            rb = QRadioButton(name if name == "Points" else f"{name} reference point")
            rb.setProperty("mode", name)
            self.mode_group.addButton(rb, k)
            ml.addWidget(rb)
        self.mode_group.button(0).setChecked(True)

        ref_box = QGroupBox("Reference values")
        rl = QFormLayout(ref_box)
        self.values = {}
        for name, default in zip(REFS, (0.0, 10.0, 0.0, 10.0)):
            sb = QDoubleSpinBox()
            sb.setRange(-1e15, 1e15)
            sb.setDecimals(6)
            sb.setValue(default)
            self.values[name] = sb
            rl.addRow(name, sb)
        self.x_log = QCheckBox("X axis is log10")
        self.y_log = QCheckBox("Y axis is log10")
        rl.addRow(self.x_log)
        rl.addRow(self.y_log)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["X", "Y"])
        self.status = QLabel("Open an image of a graph.")
        self.status.setWordWrap(True)

        open_btn = QPushButton("Open Image...")
        open_btn.clicked.connect(self.open_image)
        trace_btn = QPushButton("Auto Trace (dark curve)")
        trace_btn.clicked.connect(self.auto_trace)
        undo_btn = QPushButton("Remove Last Point")
        undo_btn.clicked.connect(self.remove_last)
        clear_btn = QPushButton("Clear Points")
        clear_btn.clicked.connect(self.clear_points)
        buttons = QDialogButtonBox()
        self.book_btn = buttons.addButton("Create Workbook",
                                          QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(QDialogButtonBox.StandardButton.Close)
        buttons.accepted.connect(self._create_and_close)
        buttons.rejected.connect(self.reject)

        side = QVBoxLayout()
        side.addWidget(open_btn)
        side.addWidget(mode_box)
        side.addWidget(ref_box)
        row = QHBoxLayout()
        row.addWidget(undo_btn)
        row.addWidget(clear_btn)
        side.addLayout(row)
        side.addWidget(trace_btn)
        side.addWidget(self.table, 1)
        side.addWidget(self.status)
        grid = QGridLayout(self)
        grid.addWidget(self.view, 0, 0)
        grid.addLayout(side, 0, 1)
        grid.addWidget(buttons, 1, 0, 1, 2)
        grid.setColumnStretch(0, 3)
        for sb in self.values.values():
            sb.valueChanged.connect(self.refresh_table)
        self.x_log.toggled.connect(self.refresh_table)
        self.y_log.toggled.connect(self.refresh_table)

    # ------------------------------------------------------------------ image
    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open Graph Image", "",
                                              "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff "
                                              "*.gif)")
        if path:
            self.load_image(path)

    def load_image(self, path_or_image) -> None:
        img = path_or_image if isinstance(path_or_image, QImage) else QImage(str(path_or_image))
        if img.isNull():
            self.status.setText("Could not read the image.")
            return
        self.image = img
        self.refs.clear()
        self.points.clear()
        scene = self.view.scene()
        scene.clear()
        scene.addPixmap(QPixmap.fromImage(img))
        scene.setSceneRect(0, 0, img.width(), img.height())
        self.view.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.refresh_table()
        self.status.setText("Click the X1 reference point on the X axis.")

    # ------------------------------------------------------------------ clicks
    def mode(self) -> str:
        return self.mode_group.checkedButton().property("mode")

    def clicked(self, px: float, py: float) -> None:
        mode = self.mode()
        if mode in REFS:
            self.set_reference(mode, px, py)
            nxt = REFS.index(mode) + 1
            self.mode_group.button(nxt).setChecked(True)
        else:
            self.add_point(px, py)

    def set_reference(self, name: str, px: float, py: float, value: float | None = None):
        self.refs[name] = (float(px), float(py))
        if value is not None:
            self.values[name].setValue(value)
        self._redraw_marks()
        missing = [r for r in REFS if r not in self.refs]
        self.status.setText(f"Click the {missing[0]} reference point." if missing else
                            "Calibrated. Click data points (Points mode) or Auto Trace.")
        self.refresh_table()

    def add_point(self, px: float, py: float) -> None:
        self.points.append((float(px), float(py)))
        self._redraw_marks()
        self.refresh_table()

    def remove_last(self) -> None:
        if self.points:
            self.points.pop()
            self._redraw_marks()
            self.refresh_table()

    def clear_points(self) -> None:
        self.points.clear()
        self._redraw_marks()
        self.refresh_table()

    def auto_trace(self) -> None:
        if self.image is None:
            return
        img = self.image.convertToFormat(QImage.Format.Format_Grayscale8)
        w, h = img.width(), img.height()
        bpl = img.bytesPerLine()
        arr = np.frombuffer(img.constBits(), np.uint8, count=bpl * h).reshape(h, bpl)[:, :w]
        x0 = int(min(p[0] for p in self.refs.values())) + 3 if self.refs else 0
        y1 = int(max(p[1] for p in self.refs.values())) - 3 if self.refs else h
        mask = arr < 100
        mask[:, :max(0, x0)] = False      # skip the Y axis line and labels
        mask[max(0, y1):, :] = False      # skip the X axis line and labels
        self.points = auto_trace(arr, mask, step=max(1, w // 120))
        self._redraw_marks()
        self.refresh_table()

    # ------------------------------------------------------------------ data
    def calibration(self) -> Calibration | None:
        if any(r not in self.refs for r in REFS):
            return None
        try:
            return Calibration((self.refs["X1"], self.refs["X2"]),
                               (self.values["X1"].value(), self.values["X2"].value()),
                               (self.refs["Y1"], self.refs["Y2"]),
                               (self.values["Y1"].value(), self.values["Y2"].value()),
                               self.x_log.isChecked(), self.y_log.isChecked())
        except ValueError as exc:
            self.status.setText(str(exc))
            return None

    def data(self) -> np.ndarray:
        cal = self.calibration()
        if cal is None or not self.points:
            return np.empty((0, 2))
        return cal.many(self.points)

    def refresh_table(self) -> None:
        d = self.data()
        self.table.setRowCount(len(d))
        for r, (x, y) in enumerate(d):
            self.table.setItem(r, 0, QTableWidgetItem(f"{x:.6g}"))
            self.table.setItem(r, 1, QTableWidgetItem(f"{y:.6g}"))

    def _redraw_marks(self) -> None:
        scene = self.view.scene()
        for item in list(scene.items()):
            if item.data(0) == "mark":
                scene.removeItem(item)
        r = 4
        for name, (px, py) in self.refs.items():
            pen = QPen(QColor(_COLORS[name]), 2)
            it = scene.addEllipse(px - r, py - r, 2 * r, 2 * r, pen)
            it.setData(0, "mark")
            t = scene.addText(name)
            t.setDefaultTextColor(QColor(_COLORS[name]))
            t.setPos(QPointF(px + 4, py - 20))
            t.setData(0, "mark")
        for px, py in self.points:
            it = scene.addRect(px - 2.5, py - 2.5, 5, 5, QPen(QColor("#00a000")),
                               QBrush(QColor("#00c000")))
            it.setData(0, "mark")

    def to_workbook(self):
        """New workbook with the digitized X/Y (returns the Workbook, or None)."""
        d = self.data()
        if not len(d):
            self.status.setText("Calibrate the axes and pick some points first.")
            return None
        from rsstatchem.ui.origin.model import Workbook
        name = self.main.project.unique_name("Digitized") if self.main else "Digitized1"
        book = Workbook.new(name, 2, len(d))
        ws = book.sheets[0]
        order = np.argsort(d[:, 0], kind="stable")
        ws.set_column_values(0, [float(v) for v in d[order, 0]])
        ws.set_column_values(1, [float(v) for v in d[order, 1]])
        ws.columns[0].long_name, ws.columns[1].long_name = "X (digitized)", "Y (digitized)"
        if self.main is not None:
            self.main.new_book(book)
            self.main.message(f"Digitizer: {len(d)} points → [{book.name}]")
        return book

    def _create_and_close(self) -> None:
        if self.to_workbook() is not None:
            self.accept()

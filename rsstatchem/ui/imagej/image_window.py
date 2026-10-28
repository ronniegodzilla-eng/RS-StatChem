"""ImageWindow: one top-level window per open image (title with zoom, info line, canvas,
stack slider). **Owned by WP-A after Phase 0** (hyperstack c/z/t sliders, animation, ROI tools,
per-platform menus). Image windows are not registered with the app context; they live in
:data:`rsstatchem.ui.imagej.windows.WINDOWS`.

Contract: ``ImageWindow(doc, bar)``; ``.doc``, ``.canvas`` (:class:`ImageCanvas`),
``.refresh()`` after pixel/display changes, ``.update_title()``; activating the window makes its
image the session's active image; closing it removes the image from the session.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QMainWindow, QScrollBar, QVBoxLayout, QWidget

from rsstatchem.ui.imagej.canvas import ImageCanvas


class ImageWindow(QMainWindow):
    def __init__(self, doc, bar=None):
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.doc = doc
        self.bar = bar
        self.confirm_close = False   # WP-A: ask "Save changes to ...?" for changed images
        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)
        self.info = QLabel()
        self.info.setStyleSheet("font-size: 11px;")
        lay.addWidget(self.info)
        self.canvas = ImageCanvas(doc)
        lay.addWidget(self.canvas, 1)
        self.slider = QScrollBar(Qt.Orientation.Horizontal)
        self.slider.setRange(0, max(doc.stack_size - 1, 0))
        self.slider.setVisible(doc.stack_size > 1)
        self.slider.valueChanged.connect(self._slider_moved)
        lay.addWidget(self.slider)
        self.setCentralWidget(central)
        self.canvas.mouse_moved.connect(self._mouse_moved)
        if bar is not None:  # single-key shortcuts work from image windows, as in ImageJ
            self.addActions(bar.shortcut_actions())
            menubar = bar.image_window_menubar(self)
            if menubar is not None:
                self.setMenuBar(menubar)
        hint = self.canvas.sizeHint()
        self.resize(max(hint.width() + 8, 220), hint.height() + 50)
        self.refresh()

    # ------------------------------------------------------------------ display
    def refresh(self) -> None:
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(self.doc.stack_size - 1, 0))
        self.slider.setValue(self.doc.current_index)
        self.slider.setVisible(self.doc.stack_size > 1)
        self.slider.blockSignals(False)
        self.canvas.refresh()
        self.update_title()

    def update_title(self) -> None:
        m = self.canvas.magnification
        title = self.doc.title
        if m != 1.0:
            title += f" ({m * 100:.0f}%)" if m * 100 == int(m * 100) else f" ({m * 100:.1f}%)"
        self.setWindowTitle(title)
        self.info.setText(self.doc.subtitle())

    def _slider_moved(self, index: int) -> None:
        self.doc.set_position(*self.doc.position_of(index))
        self.refresh()

    def _mouse_moved(self, x: float, y: float) -> None:
        if self.bar is None:
            return
        ix, iy = int(x), int(y)
        if 0 <= ix < self.doc.width and 0 <= iy < self.doc.height:
            self.bar.show_status(self.status_text(ix, iy))

    def status_text(self, x: int, y: int) -> str:
        """ImageJ's "x=12, y=34, value=200" (calibrated coordinates/values when set)."""
        doc, cal = self.doc, self.doc.calibration
        if cal.scaled():
            pos = f"x={cal.get_x(x):.2f}, y={cal.get_y(y, doc.height):.2f}"
        else:
            pos = f"x={x}, y={y}"
        if doc.is_rgb:
            r, g, b = doc.raw_value(x, y)
            return f"{pos}, value={r},{g},{b}"
        raw = doc.raw_value(x, y)
        text = f"{raw:.0f}" if doc.bit_depth != 32 else f"{raw:.4g}"
        if cal.calibrated():
            text = f"{cal.cvalue(raw):.4g} ({raw:.0f})"
        return f"{pos}, value={text}"

    # ------------------------------------------------------------------ window events
    def changeEvent(self, event) -> None:  # noqa: N802
        if event.type() == event.Type.ActivationChange and self.isActiveWindow() and \
                self.bar is not None:
            self.bar.session.set_active(self.doc)
        super().changeEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802
        event.accept()
        if self.bar is not None:
            self.bar.image_window_closed(self)

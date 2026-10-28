"""ImageWindow: one top-level window per open image (``ij/gui/ImageWindow`` and
``StackWindow``). **Owned by WP-A.** Image windows are not registered with the app context;
they live in :data:`rsstatchem.ui.imagej.windows.WINDOWS`.

Contract: ``ImageWindow(doc, bar)``; ``.doc``, ``.canvas`` (:class:`ImageCanvas`),
``.refresh()`` after pixel/display changes, ``.update_title()`` ("blobs.gif (150%)");
activating the window makes its image the session's active image; closing it removes the
image from the session (asking "Save changes to ...?" for changed images).

Stacks get a slider with a play button; hyperstacks get one slider per dimension (c, z, t),
each labelled, with play buttons for z and t (Image > Stacks > Animation). The window grows
with the zoom until it fits on the screen; the mouse wheel changes slices, ctrl/cmd+wheel
zooms; the info line above the image is ImageJ's subtitle. On macOS the window gets its own
menu bar built from the ImageJ bar's actions so the global menu stays ImageJ's.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QMainWindow, QScrollBar, QToolButton,
                               QVBoxLayout, QWidget)

from rsstatchem.ui.imagej.canvas import ImageCanvas, zoom_text
from rsstatchem.ui.imagej.roi_tools import ToolHandler

CHROME_W, CHROME_H = 8, 40      # window frame around the canvas (info line, margins)


class _DimSlider(QWidget):
    def __init__(self, letter: str, play: bool, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.play = None
        if play:
            self.play = QToolButton()
            self.play.setText("▶")
            self.play.setAutoRaise(True)
            self.play.setFixedSize(16, 16)
            self.play.setToolTip("Start/stop animation")
            lay.addWidget(self.play)
        self.label = QLabel(letter)
        self.label.setFixedWidth(12 if letter else 0)
        self.label.setStyleSheet("font-size: 11px;")
        lay.addWidget(self.label)
        self.bar = QScrollBar(Qt.Orientation.Horizontal)
        self.bar.setPageStep(1)
        lay.addWidget(self.bar, 1)


class ImageWindow(QMainWindow):
    def __init__(self, doc, bar=None):
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.doc = doc
        self.bar = bar
        self.confirm_close = True
        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)
        self.info = QLabel()
        self.info.setStyleSheet("font-size: 11px;")
        lay.addWidget(self.info)
        self.canvas = ImageCanvas(doc)
        self.canvas.handler = ToolHandler(self, bar)
        if bar is not None:
            opts = bar.session.options
            self.canvas.options = opts
            self.canvas.selection_color = opts.selection
            self.canvas.interpolate = opts.interpolate_zoomed
        lay.addWidget(self.canvas, 1)
        self.sliders: dict[str, _DimSlider] = {}
        for key in ("c", "z", "t", "stack"):
            s = _DimSlider("" if key == "stack" else key, key in ("z", "t", "stack"))
            s.bar.valueChanged.connect(lambda v, k=key: self._slider_moved(k, v))
            if s.play is not None:
                s.play.clicked.connect(self.toggle_animation)
            self.sliders[key] = s
            lay.addWidget(s)
        # Phase 0 contract name: the single stack slider
        self.slider = self.sliders["stack"].bar
        self.setCentralWidget(central)
        self.canvas.mouse_moved.connect(self._mouse_moved)
        if bar is not None:  # single-key shortcuts work from image windows, as in ImageJ
            self.addActions(bar.shortcut_actions())
            menubar = bar.image_window_menubar(self)
            if menubar is not None:
                self.setMenuBar(menubar)
        hint = self.canvas.sizeHint()
        self._fit_initial(hint.width(), hint.height())
        self.refresh()

    # ------------------------------------------------------------------ size
    def _screen_room(self) -> tuple[int, int]:
        scr = self.screen() if self.screen() is not None else None
        if scr is None:
            return 1600, 1000
        g = scr.availableGeometry()
        return max(200, g.width() - 40), max(200, g.height() - 120)

    def _chrome(self) -> tuple[int, int]:
        extra = sum(s.sizeHint().height() + 2 for s in self.sliders.values()
                    if not s.isHidden())
        mb = self.menuBar().sizeHint().height() if self.menuWidget() is not None and \
            self.menuBar().isVisible() and self.menuBar().isNativeMenuBar() is False else 0
        return CHROME_W, CHROME_H + extra + mb

    def _fit_initial(self, w: int, h: int) -> None:
        """New windows start at 100% or the largest ImageJ zoom level that fits the screen."""
        from rsstatchem.ui.imagej.canvas import ZOOM_LEVELS
        room_w, room_h = self._screen_room()
        m = 1.0
        if w > room_w or h > room_h:
            m = max([z for z in ZOOM_LEVELS if self.doc.width * z <= room_w and
                     self.doc.height * z <= room_h] or [ZOOM_LEVELS[0]])
            self.canvas.magnification = m
        cw = max(round(self.doc.width * m), 120)
        ch = max(round(self.doc.height * m), 16)
        cx, cy = self._chrome()
        self.resize(cw + cx, ch + cy)

    def grow_canvas_to(self, w: int, h: int) -> tuple[int, int]:
        """Enlarge the window so the canvas is ``w`` x ``h`` (bounded by the screen); returns
        the canvas size actually available."""
        room_w, room_h = self._screen_room()
        cw, ch = min(w, room_w), min(h, room_h)
        cur_w, cur_h = self.canvas.width(), self.canvas.height()
        cw, ch = max(cw, min(cur_w, w)), max(ch, min(cur_h, h))
        cx, cy = self._chrome()
        if (cw, ch) != (cur_w, cur_h):
            self.resize(max(cw + cx, 120), ch + cy)
            self.canvas.resize(cw, ch)
        return cw, ch

    # ------------------------------------------------------------------ display
    def refresh(self) -> None:
        doc = self.doc
        c, z, t = doc.dims
        hyper = doc.hyperstack or sum(d > 1 for d in (c, z, t)) > 1 or c > 1
        values = {"c": (c, doc.position[0]), "z": (z, doc.position[1]),
                  "t": (t, doc.position[2]), "stack": (doc.stack_size, doc.current_index)}
        for key, s in self.sliders.items():
            n, v = values[key]
            show = (key == "stack" and doc.stack_size > 1 and not hyper) or \
                (key != "stack" and hyper and n > 1)
            s.bar.blockSignals(True)
            s.bar.setRange(0, max(n - 1, 0))
            s.bar.setValue(v)
            s.bar.blockSignals(False)
            s.setVisible(show)
        self.canvas.doc = doc
        if self.bar is not None:
            self.canvas.selection_color = self.bar.session.options.selection
        self.canvas.refresh()
        self.update_title()

    def update_title(self) -> None:
        m = self.canvas.magnification
        title = self.doc.title
        if abs(m - 1.0) > 1e-9:
            title += f" ({zoom_text(m)})"
        if self.windowTitle() != title:
            self.setWindowTitle(title)
        self.info.setText(self.doc.subtitle())

    def _slider_moved(self, key: str, value: int) -> None:
        doc = self.doc
        c, z, t = doc.position
        if key == "stack":
            doc.set_position(*doc.position_of(value))
        elif key == "c":
            doc.set_position(value, z, t)
        elif key == "z":
            doc.set_position(c, value, t)
        else:
            doc.set_position(c, z, value)
        self.refresh()
        if self.bar is not None:
            self.bar.session.emit("position_changed", doc)

    def step_slice(self, delta: int) -> None:
        """Next/previous slice (z of a hyperstack, else the stack index)."""
        doc = self.doc
        c, z, t = doc.position
        if doc.n_slices > 1 and (doc.hyperstack or doc.n_channels > 1 or doc.n_frames > 1):
            doc.set_position(c, (z + delta) % doc.n_slices, t)
        elif doc.n_frames > 1 and doc.n_slices == 1 and doc.n_channels == 1:
            doc.set_position(c, z, (t + delta) % doc.n_frames)
        elif doc.n_channels > 1 and doc.n_slices == 1 and doc.n_frames == 1:
            doc.set_position((c + delta) % doc.n_channels, z, t)
        else:
            doc.set_position(*doc.position_of((doc.current_index + delta) % doc.stack_size))
        self.refresh()

    def toggle_animation(self) -> None:
        if self.bar is None:
            return
        timer = getattr(self, "_ij_d_animation", None)
        running = timer is not None and timer.isActive()
        label = "Stop Animation" if running else "Start Animation [\\]"
        if label in self.bar.session.registry:
            self.bar.session.set_active(self.doc)
            self.bar.run_command(label)

    def _mouse_moved(self, x: float, y: float) -> None:
        if self.bar is None:
            return
        ix, iy = int(x // 1), int(y // 1)
        if 0 <= ix < self.doc.width and 0 <= iy < self.doc.height:
            self.bar.show_status(self.status_text(ix, iy))

    def status_text(self, x: int, y: int) -> str:
        """ImageJ's status line (``ImagePlus.getLocationAsString`` + ``getValueAsString``):
        "x=12, y=34, value=200"; calibrated as "x=3, y=8.50 (34)"; stacks add ", z=" (0-based
        slice); 32-bit values with 4 significant decimals; RGB "value=255,000,010 (#ff000a)"."""
        from rsstatchem.core.imaging.image import COLOR_256, COLOR_RGB, GRAY32
        from rsstatchem.core.imaging.macro import d2s_sig
        from rsstatchem.core.imaging.io.text_image import d2s as ij_d2s

        doc, cal = self.doc, self.doc.calibration

        def num(v):
            return str(int(v)) if float(v).is_integer() else ij_d2s(v, 2)

        if cal.scaled():
            s = f"x={num(cal.get_x(x))} ({x}), y={num(cal.get_y(y, doc.height))} ({y})"
        else:
            s = f"x={x}, y={y}"
        if doc.is_stack:
            hyper = doc.hyperstack or sum(d > 1 for d in doc.dims) > 1 or doc.n_channels > 1
            z = doc.position[1] if hyper else doc.current_index
            cz = cal.get_z(z)
            s += f", z={num(cz)}" + (f" ({z})" if cal.scaled() and cz != z else "")
        if doc.type == COLOR_RGB:
            r, g, b = doc.raw_value(x, y)
            return f"{s}, value={r:03d},{g:03d},{b:03d} (#{r:02x}{g:02x}{b:02x})"
        raw = doc.raw_value(x, y)
        if doc.type == COLOR_256 and not cal.calibrated():
            r, g, b = (int(v) for v in doc.lut.as_array()[int(raw)])
            return f"{s}, index={int(raw)}, value={r},{g},{b}"
        if doc.type == GRAY32:
            text = f"{int(raw)}.0" if float(raw).is_integer() else (
                "NaN" if raw != raw else d2s_sig(raw, 4, 7))
            return f"{s}, value={text}"
        cv = cal.cvalue(raw)
        if cv == raw:
            return f"{s}, value={int(raw)}"
        return f"{s}, value={ij_d2s(cv, 2)} ({int(raw)})"

    # ------------------------------------------------------------------ window events
    def keyPressEvent(self, event) -> None:  # noqa: N802
        if self.canvas.handler is not None and self.canvas.handler.key_press(self.canvas,
                                                                             event):
            return
        super().keyPressEvent(event)

    def changeEvent(self, event) -> None:  # noqa: N802
        if event.type() == event.Type.ActivationChange and self.isActiveWindow() and \
                self.bar is not None and self.doc in self.bar.session.images:
            self.bar.session.set_active(self.doc)
        super().changeEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.bar is not None and self.confirm_close and self.doc.changed and \
                self.doc in self.bar.session.images:
            answer = self.bar.ask_save_changes(self.doc)
            if answer == "cancel":
                event.ignore()
                return
            if answer == "yes" and not self.bar.save_image(self.doc):
                event.ignore()
                return
        timer = getattr(self, "_ij_d_animation", None)
        if timer is not None:
            timer.stop()
        event.accept()
        if self.bar is not None:
            self.bar.image_window_closed(self)

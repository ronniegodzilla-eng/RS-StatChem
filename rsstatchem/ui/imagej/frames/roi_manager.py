"""The ROI Manager (ImageJ ``ij.plugin.frame.RoiManager``).

The ROIs live in ``session.roi_manager`` (a list); the module-level functions are the
manager's operations (headless, used by the window and by tests): add, update, delete,
rename, measure, multi measure, deselect, combine (OR), AND, XOR, split, sort, specify,
properties, flatten, fill/draw, labels / show all, list, translate, interpolate, remove
positions, open/save ``.roi`` and ``RoiSet.zip``.

The window has ImageJ's buttons (Add [t], Update, Delete, Rename..., Measure, Deselect,
Properties..., Flatten [F], More >>) and the "Show All" and "Labels" checkboxes; clicking an
entry selects that ROI on the active image (and moves to its slice).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QColor, QFont, QPen
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QFileDialog, QHBoxLayout,
                               QInputDialog, QListWidget, QMainWindow, QMenu, QPushButton,
                               QVBoxLayout, QWidget)

from rsstatchem.core.imaging import measure as M
from rsstatchem.core.imaging.commands import CommandError
from rsstatchem.core.imaging.results import ResultsTable
from rsstatchem.core.imaging.roi import geometry as G
from rsstatchem.core.imaging.roi import shapes as S
from rsstatchem.ui.imagej import windows as W
from rsstatchem.ui.imagej.frames.plot_window import valid


# ---------------------------------------------------------------------------------- model
def label_for(roi, image=None) -> str:
    """ImageJ's default ROI Manager name: "yyyy-xxxx" (bounds centre), "ssss-yyyy-xxxx" in
    stacks."""
    x, y, w, h = G.int_bounds(roi)
    yc, xc = y + h // 2, x + w // 2
    if image is not None and image.stack_size > 1:
        pos = roi.position
        sl = pos[1] if pos[1] else image.current_index + 1
        if image.hyperstack or image.is_composite:
            c, z, t = pos if any(pos) else tuple(v + 1 for v in image.position)
            return f"{c:04d}-{z:04d}-{t:04d}-{yc:04d}-{xc:04d}"
        return f"{sl:04d}-{yc:04d}-{xc:04d}"
    return f"{yc:04d}-{xc:04d}"


def unique_name(session, roi, image=None) -> str:
    names = {r.name for r in session.roi_manager}
    base = roi.name if roi.name and roi.name not in names else (roi.name or label_for(roi, image))
    if base not in names:
        return base
    k = 1
    while f"{base}-{k}" in names:
        k += 1
    return f"{base}-{k}"


def add(session, image, roi=None):
    roi = image.roi if roi is None and image is not None else roi
    if roi is None:
        raise CommandError("The active image does not have a selection.")
    r = roi.copy()
    if image is not None and image.stack_size > 1 and r.position == (0, 0, 0):
        if image.hyperstack or image.is_composite:
            c, z, t = image.position
            r.position = (c + 1, z + 1, t + 1)
        else:
            r.position = (0, image.current_index + 1, 0)
    r.name = unique_name(session, r, image)
    session.roi_manager.append(r)
    return r


def update(session, image, index: int):
    if image is None or image.roi is None:
        raise CommandError("The active image does not have a selection.")
    old = session.roi_manager[index]
    r = image.roi.copy()
    r.name = old.name
    if r.position == (0, 0, 0):
        r.position = old.position
    session.roi_manager[index] = r
    return r


def delete(session, indices=None):
    rm = session.roi_manager
    if not indices:
        rm.clear()
        return
    for i in sorted(set(indices), reverse=True):
        if 0 <= i < len(rm):
            del rm[i]


def rename(session, index: int, name: str):
    session.roi_manager[index].name = name


def _selected(session, indices):
    rm = session.roi_manager
    return [rm[i] for i in (indices if indices else range(len(rm)))]


def _goto(image, roi):
    c, z, t = roi.position
    if image.stack_size > 1 and any((c, z, t)):
        if image.hyperstack or image.is_composite:
            image.set_position(max(c - 1, 0) if c else image.position[0],
                               max(z - 1, 0) if z else image.position[1],
                               max(t - 1, 0) if t else image.position[2])
        elif z:
            image.set_position(*image.position_of(z - 1))


def measure(session, image, indices=None) -> ResultsTable:
    rt = session.results
    flags = M.flags_from_keys(session.options.measurements)
    saved_pos, saved_roi = image.position, image.roi
    for roi in _selected(session, indices):
        _goto(image, roi)
        image.roi = roi
        M.Analyzer(image, flags, rt, session.options.decimal_places).measure()
    image.set_position(*saved_pos)
    image.roi = saved_roi
    return rt


def multi_measure(session, image, indices=None, all_slices: bool = True,
                  one_row: bool = True, append: bool = False) -> ResultsTable:
    """More >> Multi Measure: every ROI on every slice; one row per slice with columns
    "<measurement>(<roi name>)" (or one row per ROI and slice)."""
    rois = _selected(session, indices)
    rt = session.results if append else ResultsTable.results("Results")
    if not append:
        rt.reset()
    flags = M.flags_from_keys(session.options.measurements) & ~M.LABELS
    planes = range(image.stack_size) if all_slices else [image.current_index]
    saved_pos, saved_roi = image.position, image.roi
    for index in planes:
        image.set_position(*image.position_of(index))
        if one_row:
            row = rt.increment_counter()
            for roi in rois:
                tmp = ResultsTable.results("tmp")
                image.roi = roi
                M.Analyzer(image, flags, tmp, session.options.decimal_places).measure()
                for h in tmp.headings:
                    if h != "Label":
                        rt.set_value(f"{h}({roi.name})", row, tmp.get_value(h, 0))
        else:
            for roi in rois:
                image.roi = roi
                M.Analyzer(image, flags | M.LABELS, rt, session.options.decimal_places
                           ).measure()
    image.set_position(*saved_pos)
    image.roi = saved_roi
    rt.title = "Results"
    session.results = rt
    return rt


def multi_plot(session, image, indices=None):
    """More >> Multi Plot: the mean of each ROI against the slice number."""
    from rsstatchem.core.imaging.profile import PlotData
    rois = _selected(session, indices)
    plot = PlotData("Profiles", "Slice", "Mean")
    saved_pos = image.position
    for roi in rois:
        means = []
        for index in range(image.stack_size):
            image.set_position(*image.position_of(index))
            rect, mask = M.roi_rect_and_mask(roi, image.width, image.height)
            st = M.get_statistics(image.plane(), image.type, rect, mask, M.MEAN,
                                  image.calibration)
            means.append(st.mean)
        plot.add(np.arange(1, image.stack_size + 1), means, label=roi.name)
    image.set_position(*saved_pos)
    plot.legend = True
    return plot


def combine(session, indices=None, op: str = "or"):
    rois = [r for r in _selected(session, indices) if r.is_area() or isinstance(r, S.PointRoi)]
    if len(rois) < 2 and op != "or":
        raise CommandError("More than one item must be selected, or none")
    if all(isinstance(r, S.PointRoi) for r in rois):
        xs = np.concatenate([r.xs for r in rois])
        ys = np.concatenate([r.ys for r in rois])
        return S.PointRoi(xs, ys)
    rois = [r for r in rois if r.is_area()]
    if not rois:
        raise CommandError("Area selections required")
    if len(rois) == 1:
        return rois[0].copy()
    return S.CompositeRoi([("or" if i == 0 else op, r.copy()) for i, r in enumerate(rois)])


def split(session, index: int):
    roi = session.roi_manager[index]
    if isinstance(roi, S.CompositeRoi):
        parts = [p.copy() for _, p in roi.parts]
    elif isinstance(roi, S.PointRoi):
        parts = [S.PointRoi([x], [y]) for x, y in zip(roi.xs, roi.ys)]
    else:
        raise CommandError("Composite or multi-point selection required")
    for p in parts:
        p.position = roi.position
        p.name = unique_name(session, p)
        session.roi_manager.append(p)
    return parts


def sort(session):
    session.roi_manager.sort(key=lambda r: r.name)


def set_properties(session, indices=None, stroke=None, width=None, fill=None, name=None,
                   position=None):
    for r in _selected(session, indices):
        if stroke is not None:
            r.stroke_color = stroke or None
        if width is not None:
            r.stroke_width = float(width)
        if fill is not None:
            r.fill_color = fill or None
        if position is not None:
            r.position = position
    if name is not None and indices and len(indices) == 1:
        session.roi_manager[indices[0]].name = name


def remove_positions(session, indices=None):
    for r in _selected(session, indices):
        r.position = (0, 0, 0)


def translate(session, dx: float, dy: float, indices=None):
    for r in _selected(session, indices):
        r.translate(dx, dy)


def interpolate_rois(session, image):
    """More >> Interpolate ROIs: linear interpolation of the outlines between slices that
    have a ROI (resampled to the same number of points)."""
    from rsstatchem.core.imaging.roi.geometry import interpolated_polygon
    keyed = {}
    for r in session.roi_manager:
        if r.is_area() and r.position[1]:
            keyed[r.position[1]] = r
    zs = sorted(keyed)
    new = []
    for a, b in zip(zs, zs[1:]):
        if b - a < 2:
            continue
        xa, ya = G.float_polygon(keyed[a])
        xb, yb = G.float_polygon(keyed[b])
        n = 100
        pa = interpolated_polygon(xa, ya, G.polyline_length(xa, ya, True) / n, closed=True)
        pb = interpolated_polygon(xb, yb, G.polyline_length(xb, yb, True) / n, closed=True)
        m = min(len(pa[0]), len(pb[0]))
        for z in range(a + 1, b):
            f = (z - a) / (b - a)
            xs = pa[0][:m] * (1 - f) + pb[0][:m] * f
            ys = pa[1][:m] * (1 - f) + pb[1][:m] * f
            r = S.PolygonRoi(xs, ys, type=S.POLYGON, position=(0, z, 0))
            r.name = unique_name(session, r, image)
            new.append(r)
            session.roi_manager.append(r)
    session.roi_manager.sort(key=lambda r: (r.position[1], r.name))
    return new


def list_table(session) -> ResultsTable:
    rt = ResultsTable("Overlay Elements" if False else "ROI List")
    for i, r in enumerate(session.roi_manager):
        x, y, w, h = G.int_bounds(r)
        rt.add_row({"Index": i, "Name": r.name, "Type": r.type_name, "X": x, "Y": y,
                    "Width": w, "Height": h, "Position": r.position[1],
                    "Color": r.stroke_color or "none"})
    return rt


def fill(session, image, indices=None, value=None):
    image.snapshot("Fill")
    plane = image.plane()
    for r in _selected(session, indices):
        if r.is_area():
            m = r.get_mask((image.height, image.width))
            plane[m] = value if value is not None else (255 if not image.is_rgb else
                                                        np.array([255, 255, 255], np.uint8))
    image.changed = True


def draw(session, image, indices=None, value=None):
    from rsstatchem.core.imaging.particles import draw_polygon
    image.snapshot("Draw")
    plane = image.plane()
    v = value if value is not None else (255 if not image.is_rgb else
                                         np.array([255, 255, 255], np.uint8))
    for r in _selected(session, indices):
        xs, ys = G.float_polygon(r)
        if len(xs):
            draw_polygon(plane, np.floor(xs), np.floor(ys), v, closed=r.is_area())
    image.changed = True


def show_all(image, rois, labels: bool = False, show: bool = True):
    """"Show All": the manager's ROIs as the image overlay (removed again with show=False)."""
    if not show:
        if image.overlay is not None:
            image.overlay.rois = [r for r in image.overlay
                                  if not r.properties.get("roi_manager")]
        return
    ov = image.overlay or S.Overlay()
    ov.rois = [r for r in ov.rois if not r.properties.get("roi_manager")]
    for r in rois:
        c = r.copy()
        c.properties["roi_manager"] = "1"
        ov.add(c)
    ov.show_labels = labels
    ov.draw_names = labels
    image.overlay = ov


def flatten(session, image, indices=None, labels: bool = False):
    from rsstatchem.core.imaging.image import ImageDoc
    from rsstatchem.core.imaging.roi.selection_ops import flatten as fl
    rgb = fl(image, _selected(session, indices), labels)
    title = image.title.rsplit(".", 1)[0] + "-1"
    doc = ImageDoc.from_array(rgb, "YXS", title)
    doc.calibration = image.calibration.copy()
    doc.calibration.disable_density()
    return doc


def save(session, path, indices=None):
    from rsstatchem.core.imaging.roi.roi_io import write_roi, write_rois
    rois = _selected(session, indices)
    if not rois:
        raise CommandError("The ROI Manager is empty")
    p = Path(path)
    if p.suffix.lower() == ".roi" and len(rois) == 1:
        write_roi(p, rois[0])
    else:
        if p.suffix.lower() != ".zip":
            p = p.with_suffix(".zip")
        write_rois(p, rois)
    return p


def open_rois(session, path):
    from rsstatchem.core.imaging.roi.roi_io import read_rois
    new = read_rois(path)
    for r in new:
        r.name = unique_name(session, r)
        session.roi_manager.append(r)
    return new


# ---------------------------------------------------------------------------------- window
class RoiManagerWindow(QMainWindow):
    def __init__(self, bar):
        super().__init__()
        self.bar = bar
        self.setWindowTitle("ROI Manager")
        central = QWidget(self)
        outer = QHBoxLayout(central)
        self.list = QListWidget(central)
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list.itemSelectionChanged.connect(self._selection_changed)
        outer.addWidget(self.list, 1)
        col = QVBoxLayout()
        for text, slot in (("Add [t]", self.add), ("Update", self.update_roi),
                           ("Delete", self.delete), ("Rename...", self.rename),
                           ("Measure", self.measure), ("Deselect", self.deselect),
                           ("Properties...", self.properties), ("Flatten [F]", self.flatten)):
            b = QPushButton(text, central)
            b.clicked.connect(lambda _=False, s=slot: self._safe(s))
            col.addWidget(b)
        more = QPushButton("More >>", central)
        menu = QMenu(more)
        for text, slot in (("Open...", self.open), ("Save...", self.save),
                           ("Fill", self.fill), ("Draw", self.draw),
                           ("AND", lambda: self.combine("and")),
                           ("OR (Combine)", lambda: self.combine("or")),
                           ("XOR", lambda: self.combine("xor")), ("Split", self.split),
                           ("Add Particles", self.add_particles),
                           ("Multi Measure", self.multi_measure),
                           ("Multi Plot", self.multi_plot), ("Sort", self.sort),
                           ("Specify...", self.specify),
                           ("Remove Positions...", self.remove_positions),
                           ("Labels...", self.toggle_labels), ("List", self.list_rois),
                           ("Interpolate ROIs", self.interpolate), ("Translate...", self.translate),
                           ("Reset", self.reset)):
            a = QAction(text, self)
            a.triggered.connect(lambda _=False, s=slot: self._safe(s))
            menu.addAction(a)
        more.setMenu(menu)
        col.addWidget(more)
        self.show_all_box = QCheckBox("Show All", central)
        self.show_all_box.toggled.connect(lambda _on: self._safe(self.apply_show_all))
        col.addWidget(self.show_all_box)
        self.labels_box = QCheckBox("Labels", central)
        self.labels_box.toggled.connect(lambda _on: self._safe(self.apply_show_all))
        col.addWidget(self.labels_box)
        col.addStretch(1)
        outer.addLayout(col)
        self.setCentralWidget(central)
        self.resize(320, 360)
        self._painted = set()
        self.refresh()

    # ------------------------------------------------------------------ helpers
    @property
    def session(self):
        return self.bar.session

    @property
    def image(self):
        return self.session.active

    def _safe(self, fn):
        try:
            return fn()
        except CommandError as exc:
            self.bar.error(str(exc), "ROI Manager")
        return None

    def selected(self) -> list[int]:
        return sorted(self.list.row(i) for i in self.list.selectedItems())

    def refresh(self) -> None:
        if not valid(self):
            return
        sel = set(self.selected())
        self.list.blockSignals(True)
        self.list.clear()
        for r in self.session.roi_manager:
            self.list.addItem(r.name)
        for i in sel:
            if i < self.list.count():
                self.list.item(i).setSelected(True)
        self.list.blockSignals(False)
        if self.show_all_box.isChecked():
            self.apply_show_all()
        self.setWindowTitle(f"ROI Manager ({len(self.session.roi_manager)})"
                            if self.session.roi_manager else "ROI Manager")

    def _image_changed(self, image=None):
        if image is not None:
            self.session.image_updated(image)

    def _need_image(self):
        if self.image is None:
            raise CommandError("There are no images open")
        return self.image

    def _record(self, cmd: str, extra: str = "") -> None:
        rec = self.session.recorder
        if rec is not None:
            rec.record_line(f'roiManager("{cmd}"{", " + extra if extra else ""});')

    # ------------------------------------------------------------------ buttons
    def add(self):
        im = self._need_image()
        add(self.session, im)
        self._record("Add")
        self.refresh()
        self.list.clearSelection()

    def update_roi(self):
        sel = self.selected()
        if len(sel) != 1:
            raise CommandError("Exactly one item in the list must be selected.")
        update(self.session, self._need_image(), sel[0])
        self._record("Update")
        self.refresh()

    def delete(self):
        sel = self.selected()
        delete(self.session, sel)
        self._record("Delete")
        self.refresh()

    def rename(self, name: str | None = None):
        sel = self.selected()
        if len(sel) != 1:
            raise CommandError("Exactly one item in the list must be selected.")
        if name is None:
            name, ok = QInputDialog.getText(self, "Rename", "Name:",
                                            text=self.session.roi_manager[sel[0]].name)
            if not ok:
                return
        rename(self.session, sel[0], name)
        self._record("Rename", f'"{name}"')
        self.refresh()

    def measure(self):
        im = self._need_image()
        rt = measure(self.session, im, self.selected())
        self._record("Measure")
        self.session.show_table(rt)

    def deselect(self):
        self.list.clearSelection()
        self._record("Deselect")

    def properties(self, stroke=None, width=None, fill=None):
        if stroke is None and width is None and fill is None:
            text, ok = QInputDialog.getText(self, "Properties",
                                            "Stroke color, width, fill color:",
                                            text="#ffff00, 1, none")
            if not ok:
                return
            parts = [p.strip() for p in text.split(",")]
            stroke = parts[0] if parts else None
            width = float(parts[1]) if len(parts) > 1 and parts[1] else None
            fill = parts[2] if len(parts) > 2 else None
            if fill == "none":
                fill = ""
        set_properties(self.session, self.selected(), stroke, width, fill)
        self.apply_show_all()

    def flatten(self):
        im = self._need_image()
        doc = flatten(self.session, im, self.selected() or None, self.labels_box.isChecked())
        self.session.add_image(doc)
        return doc

    def open(self, path: str | None = None):
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "Open ROIs", "",
                                                  "ROI (*.roi *.zip)")
            if not path:
                return None
        new = open_rois(self.session, path)
        self._record("Open", f'"{Path(path).as_posix()}"')
        self.refresh()
        return new

    def save(self, path: str | None = None):
        if path is None:
            default = "RoiSet.zip" if len(self.selected()) != 1 else \
                self.session.roi_manager[self.selected()[0]].name + ".roi"
            path, _ = QFileDialog.getSaveFileName(self, "Save ROIs", default,
                                                  "ROI set (*.zip);;ROI (*.roi)")
            if not path:
                return None
        p = save(self.session, path, self.selected())
        self._record("Save", f'"{Path(p).as_posix()}"')
        return p

    def fill(self):
        im = self._need_image()
        fill(self.session, im, self.selected())
        self._image_changed(im)

    def draw(self):
        im = self._need_image()
        draw(self.session, im, self.selected())
        self._image_changed(im)

    def combine(self, op: str = "or"):
        im = self._need_image()
        roi = combine(self.session, self.selected(), op)
        im.roi = roi
        self._record({"or": "Combine", "and": "AND", "xor": "XOR"}[op])
        self._image_changed(im)
        return roi

    def split(self):
        sel = self.selected()
        if len(sel) != 1:
            raise CommandError("Exactly one item in the list must be selected.")
        split(self.session, sel[0])
        self._record("Split")
        self.refresh()

    def add_particles(self):
        """Add the outlines of the last Analyze Particles run (the particle ROIs kept in
        the Results/overlay) — runs Analyze Particles with "add" on the active image."""
        self.bar.run_command("Analyze Particles...", "add")
        self.refresh()

    def multi_measure(self):
        im = self._need_image()
        rt = multi_measure(self.session, im, self.selected())
        self._record("Multi Measure")
        self.session.show_table(rt)
        return rt

    def multi_plot(self):
        im = self._need_image()
        plot = multi_plot(self.session, im, self.selected())
        from rsstatchem.ui.imagej.frames.plot_window import show_plot
        show_plot(self.bar, plot)
        return plot

    def sort(self):
        sort(self.session)
        self._record("Sort")
        self.refresh()

    def specify(self):
        self.bar.run_command("Specify...")
        if self.image is not None and self.image.roi is not None:
            add(self.session, self.image)
            self.refresh()

    def remove_positions(self):
        remove_positions(self.session, self.selected())
        self._record("Remove Slice Info")
        self.refresh()

    def toggle_labels(self):
        self.labels_box.setChecked(not self.labels_box.isChecked())

    def list_rois(self):
        rt = list_table(self.session)
        self.session.show_table(rt)
        return rt

    def interpolate(self):
        im = self._need_image()
        interpolate_rois(self.session, im)
        self.refresh()

    def translate(self, dx: float | None = None, dy: float | None = None):
        if dx is None:
            text, ok = QInputDialog.getText(self, "Translate", "X offset, Y offset:",
                                            text="10, 10")
            if not ok:
                return
            dx, dy = (float(t) for t in text.split(","))
        translate(self.session, float(dx), float(dy), self.selected())
        self._record("Translate", f"{dx:g}, {dy:g}")
        self.apply_show_all()

    def reset(self):
        delete(self.session, None)
        self._record("Reset")
        self.refresh()

    # ------------------------------------------------------------------ selection / overlay
    def _selection_changed(self):
        sel = self.selected()
        im = self.image
        if im is None or not sel:
            return
        rm = self.session.roi_manager
        if len(sel) == 1:
            roi = rm[sel[0]]
            _goto(im, roi)
            im.roi = roi.copy()
        else:
            try:
                im.roi = combine(self.session, sel, "or")
            except CommandError:
                return
        self.session.image_updated(im)

    def apply_show_all(self):
        im = self.image
        if im is None:
            return
        show_all(im, self.session.roi_manager, self.labels_box.isChecked(),
                 self.show_all_box.isChecked())
        self._install_label_hook(im)
        self.session.image_updated(im)

    def _install_label_hook(self, image):
        win = W.window_for(image)
        if win is None or id(win.canvas) in self._painted:
            return
        canvas = win.canvas
        self._painted.add(id(canvas))

        def hook(p, cv, mgr=self):
            if not valid(mgr) or not mgr.labels_box.isChecked() or \
                    not mgr.show_all_box.isChecked():
                return
            p.setPen(QPen(QColor(255, 255, 255)))
            p.setFont(QFont("SansSerif", 9))
            for k, r in enumerate(mgr.session.roi_manager):
                x, y, w, h = G.int_bounds(r)
                sx, sy = cv.image_to_screen(x + w / 2, y + h / 2)
                text = r.name if mgr.labels_box.isChecked() else str(k + 1)
                p.drawText(int(sx - 3 * len(text)), int(sy + 4), text)
        canvas.paint_hooks.append(hook)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_T:
            self._safe(self.add)
            return
        if event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
            self._safe(self.delete)
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802
        event.accept()
        W.unregister(self)


def manager_window(bar, create: bool = True):
    for w in W.WINDOWS:
        if isinstance(w, RoiManagerWindow) and w.bar is bar and valid(w):
            return w
    if not create:
        return None
    win = RoiManagerWindow(bar)
    W.register(win)
    return win

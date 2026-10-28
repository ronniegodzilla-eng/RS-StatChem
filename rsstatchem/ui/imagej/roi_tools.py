"""Mouse and keyboard interaction of the toolbar tools on an image canvas (ImageJ
``ij/gui/Roi``, ``PolygonRoi``, ``Line``, ``PointRoi``, ``TextRoi``, ``ImageCanvas`` tool
handling). **Owned by WP-A.**

* Rectangle (rounded, rotated), Oval (elliptical, selection brush): drag; shift = square /
  circle, ctrl/cmd = from the centre; dragging inside the selection moves it, handles resize
  it; shift-drag adds to and alt-drag subtracts from an existing area selection (composite
  ROI); a click without dragging outside the selection removes it.
* Polygon and segmented line: click vertices, double-click (or click the first vertex)
  finishes, alt-click on a vertex deletes it; freehand area/line: drag; straight line and
  arrow: drag (shift = horizontal/vertical); angle: three clicks.
* Point: click (shift-click adds); multi-point adds points with the current counter,
  alt-click deletes the nearest point; "Auto-measure" / "Add to ROI Manager" options.
* Wand: traces the object under the cursor (Wand Tool options: Legacy / 4- / 8-connected,
  tolerance; the threshold range when the image is thresholded). Uses WP-C's
  ``core/imaging/roi/wand.py`` (``wand_outline(plane, x, y, ...)``) when present.
* Text: click to place, type, Enter for a new line; magnifier: click = zoom in, right/alt
  click = zoom out; scrolling hand (also the space bar held down); colour picker: click =
  foreground, alt-click = background.
* Keys: arrows move the selection (alt+arrows resize it), or change slices (left/right) and
  zoom (up/down) when there is none; Esc aborts; Backspace/Delete clears the selection; other
  keys go to the bar's single-key shortcuts.
"""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPen, QPolygonF

from rsstatchem.core.imaging.roi import shapes as S
from rsstatchem.ui.imagej import toolbar as T
from rsstatchem.ui.imagej.canvas import HANDLE, handle_points

AREA_TOOLS = (T.RECTANGLE, T.OVAL, T.POLYGON, T.FREEROI)
CLOSE_DIST = 6        # screen pixels: click near the first vertex closes a polygon


def _mods(event):
    m = event.modifiers()
    shift = bool(m & Qt.KeyboardModifier.ShiftModifier)
    alt = bool(m & Qt.KeyboardModifier.AltModifier)
    ctrl = bool(m & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier))
    return shift, alt, ctrl


# ---------------------------------------------------------------------------------- wand
def wand_roi(image, x: int, y: int, mode: str = "Legacy", tolerance: float = 0.0):
    """Traced selection of the object containing pixel (x, y)."""
    try:
        from rsstatchem.core.imaging.roi import wand as W  # WP-C port of ij/gui/Wand
        fn = getattr(W, "wand_outline", None)
    except ImportError:
        fn = None
    plane = image.plane()
    if image.is_rgb:
        plane = plane.astype(np.float64).sum(axis=2) / 3.0
    if fn is not None:
        thr = image.threshold
        lower, upper = (thr.lower, thr.upper) if thr is not None else (None, None)
        return fn(plane, x, y, lower=lower, upper=upper, mode=mode, tolerance=tolerance)
    return _wand_fallback(plane, x, y, image.threshold, mode, tolerance)


def _wand_fallback(plane, x, y, threshold, mode, tolerance):
    from scipy import ndimage

    from rsstatchem.core.imaging.io.roi_codec import mask_outlines

    h, w = plane.shape
    if not (0 <= x < w and 0 <= y < h):
        return None
    v = float(plane[y, x])
    if threshold is not None:
        inside = (plane >= threshold.lower) & (plane <= threshold.upper)
        if not inside[y, x]:
            inside = ~inside
    elif tolerance > 0:
        inside = np.abs(plane.astype(np.float64) - v) <= tolerance
    else:
        inside = plane == plane[y, x]
    structure = np.ones((3, 3), bool) if mode != "4-connected" else None
    labels, _ = ndimage.label(inside, structure=structure)
    region = labels == labels[y, x]
    if mode == "Legacy" or threshold is not None:
        region = ndimage.binary_fill_holes(region)
    loops = mask_outlines(region)
    if not loops:
        return None
    best = max(loops, key=lambda lp: abs(_area(lp)))
    return S.PolygonRoi(best[:, 0], best[:, 1], type=S.TRACED_ROI)


def _area(lp) -> float:
    x, y = lp[:, 0], lp[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# ---------------------------------------------------------------------------------- handler
class ToolHandler:
    """Per image window; reads the current tool from ``bar.toolbar`` (``tool`` overrides it
    in tests). Finished selections are recorded through ``bar.record_roi``."""

    def __init__(self, window, bar=None):
        self.window = window
        self.bar = bar
        self.tool: int | None = None
        self.variant: str | None = None
        self.mode: str | None = None
        self.points: list[tuple[float, float]] = []
        self.start = (0.0, 0.0)
        self.last = (0.0, 0.0)
        self.cursor = (0.0, 0.0)
        self.op: str | None = None
        self.base_roi = None
        self.orig = None
        self.handle = -1
        self.space_hand = False
        self.typing = False
        self.brush_mask = None
        self.moved = False

    # ------------------------------------------------------------------ helpers
    @property
    def doc(self):
        return self.window.doc

    def current_tool(self) -> int:
        if self.space_hand:
            return T.HAND
        if self.tool is not None:
            return self.tool
        tb = getattr(self.bar, "toolbar", None)
        return tb.current if tb is not None else T.RECTANGLE

    def current_variant(self) -> str:
        if self.variant is not None:
            return self.variant
        tb = getattr(self.bar, "toolbar", None)
        return tb.current_variant() if tb is not None else ""

    def options(self):
        return self.bar.session.options if self.bar is not None else None

    def _status(self, text: str) -> None:
        if self.bar is not None:
            self.bar.show_status(text)

    def _update(self, canvas) -> None:
        canvas.repaint_overlay()

    def _finish(self, canvas, roi) -> None:
        """Make ``roi`` the selection (combined with the previous one for shift/alt)."""
        doc = self.doc
        if roi is not None and self.op and self.base_roi is not None and \
                self.base_roi.is_area() and roi.is_area():
            parts = list(self.base_roi.parts) if isinstance(self.base_roi, S.CompositeRoi) \
                else [("or", self.base_roi)]
            parts.append((self.op, roi))
            roi = S.CompositeRoi(parts=parts)
        doc.roi = roi
        self.op = None
        self.base_roi = None
        self.mode = None
        self.points = []
        if self.bar is not None:
            self.bar.session.image_updated(doc)
            if roi is not None:
                self.bar.record_roi(roi)
        self._update(canvas)

    def _ip(self, x: float, y: float) -> tuple[int, int]:
        return int(math.floor(x)), int(math.floor(y))

    def _near_handle(self, canvas, roi, x, y) -> int:
        sx, sy = canvas.image_to_screen(x, y)
        for i, (hx, hy) in enumerate(handle_points(roi)):
            hsx, hsy = canvas.image_to_screen(hx, hy)
            if abs(hsx - sx) <= HANDLE and abs(hsy - sy) <= HANDLE:
                return i
        return -1

    def _inside(self, canvas, roi, x, y) -> bool:
        if roi.is_area():
            return roi.contains(int(math.floor(x)), int(math.floor(y)))
        if isinstance(roi, S.TextRoi):
            bx, by, bw, bh = roi.bounds()
            return bx <= x <= bx + bw and by <= y <= by + bh
        xs, ys = roi.polygon()
        tol = 4 / canvas.magnification
        for i in range(len(xs) - 1):
            if _seg_dist(x, y, xs[i], ys[i], xs[i + 1], ys[i + 1]) <= tol:
                return True
        if isinstance(roi, S.PointRoi):
            return any(math.hypot(px + 0.5 - x, py + 0.5 - y) <= tol for px, py in
                       zip(xs, ys))
        return False

    # ------------------------------------------------------------------ mouse
    def mouse_press(self, canvas, event, x, y) -> None:
        tool = self.current_tool()
        shift, alt, ctrl = _mods(event)
        right = event.button() == Qt.MouseButton.RightButton
        self.start = self.last = (x, y)
        self.moved = False
        doc = self.doc
        if tool == T.MAGNIFIER:
            (canvas.zoom_out if right or alt else canvas.zoom_in)((x, y))
            return
        if tool == T.HAND:
            self.mode = "pan"
            self._pan_origin = canvas.image_to_screen(x, y)
            return
        if tool == T.DROPPER:
            self.pick_color(canvas, x, y, background=alt or right)
            return
        if right:
            if self.bar is not None and hasattr(self.bar, "image_popup"):
                self.bar.image_popup(self.window, event.globalPosition().toPoint())
            return
        if self.mode in ("polygon", "polyline", "angle"):
            self._add_vertex(canvas, x, y, alt)
            return
        if self.typing and tool == T.TEXT:
            roi = doc.roi
            if not (isinstance(roi, S.TextRoi) and self._inside(canvas, roi, x, y)):
                self.typing = False
        roi = doc.roi
        # resize / move the current selection
        if roi is not None and not shift and not alt and tool != T.WAND:
            h = self._near_handle(canvas, roi, x, y)
            if h >= 0 and not isinstance(roi, S.PointRoi):
                self.mode, self.handle, self.orig = "handle", h, roi.copy()
                return
            if self._inside(canvas, roi, x, y) and not (tool == T.POINT and
                                                         self._multi()):
                self.mode, self.orig = "move", roi.copy()
                return
        if tool == T.POINT:
            self._point_click(canvas, x, y, shift, alt)
            return
        if tool == T.WAND:
            o = self.options()
            ix, iy = self._ip(x, y)
            r = wand_roi(doc, ix, iy, o.wand_mode if o else "Legacy",
                         o.wand_tolerance if o else 0.0)
            self.op, self.base_roi = self._composite_op(shift, alt)
            self._finish(canvas, r)
            return
        if tool == T.TEXT:
            o = self.options()
            size = float(o.extra.get("font.size", 18)) if o else 18.0
            font = o.extra.get("font.name", "SansSerif") if o else "SansSerif"
            doc.roi = S.TextRoi(x=float(math.floor(x)), y=float(math.floor(y)), text="",
                                font=font, font_size=size)
            self.typing = True
            self._update(canvas)
            return
        self.op, self.base_roi = self._composite_op(shift, alt) if tool in AREA_TOOLS else \
            (None, None)
        if tool == T.POLYGON:
            self.mode = "polygon"
            self.points = [self._snap(x, y)]
        elif tool == T.POLYLINE:
            self.mode = "polyline"
            self.points = [(x, y)]
        elif tool == T.ANGLE:
            self.mode = "angle"
            self.points = [(x, y)]
        elif tool in (T.FREEROI, T.FREELINE):
            self.mode = "freehand"
            self.points = [(x, y)]
        elif tool == T.OVAL and self.current_variant() == "Selection Brush Tool":
            self._brush_start(canvas, x, y)
        else:
            self.mode = "create"
        if self.mode != "create" and self.op is None and self.mode not in ("brush",):
            doc.roi = None
        self._update(canvas)

    def _composite_op(self, shift, alt):
        roi = self.doc.roi
        if roi is not None and roi.is_area() and (shift or alt):
            return ("or" if shift else "not"), roi
        return None, None

    def _snap(self, x, y):
        return float(math.floor(x)), float(math.floor(y))

    def _multi(self) -> bool:
        tb = getattr(self.bar, "toolbar", None)
        return self.current_variant() == "Multi-point Tool" or (tb is not None and
                                                                  tb.multi_point and
                                                                  self.variant is None)

    def mouse_move(self, canvas, event, x, y) -> None:
        self.cursor = (x, y)
        buttons = event.buttons()
        if self.mode == "pan" and buttons:
            sx, sy = canvas.image_to_screen(x, y)
            ox, oy = self._pan_origin
            canvas.scroll_by(sx - ox, sy - oy)
            self._pan_origin = canvas.image_to_screen(x, y)
            return
        if self.mode in ("polygon", "polyline", "angle"):
            self._update(canvas)
            return
        if not buttons:
            return
        self.moved = True
        shift, alt, ctrl = _mods(event)
        doc = self.doc
        if self.mode == "move" and self.orig is not None:
            dx, dy = x - self.start[0], y - self.start[1]
            if not isinstance(self.orig, (S.LineRoi, S.PolygonRoi, S.PointRoi)) or \
                    self.orig.is_area():
                dx, dy = round(dx), round(dy)
            r = self.orig.copy()
            r.translate(dx, dy)
            doc.roi = r
            bx, by, bw, bh = r.bounds()
            self._status(f"x={bx:g}, y={by:g}, w={bw:g}, h={bh:g}")
        elif self.mode == "handle" and self.orig is not None:
            doc.roi = resize_roi(self.orig, self.handle, x, y, shift)
        elif self.mode == "create":
            doc.roi = self._shape(self.start, (x, y), shift, ctrl)
            self._drag_status(doc.roi)
        elif self.mode == "freehand":
            px, py = self.points[-1]
            if math.hypot(px - x, py - y) * canvas.magnification >= 1.5:
                self.points.append((x, y))
        elif self.mode == "brush":
            self._brush_paint(x, y)
        self._update(canvas)

    def mouse_release(self, canvas, event, x, y) -> None:
        doc = self.doc
        if self.mode == "pan":
            self.mode = None
            return
        if self.mode in ("move", "handle"):
            self.mode = None
            if self.moved and self.bar is not None and doc.roi is not None:
                self.bar.record_roi(doc.roi)
            if self.bar is not None:
                self.bar.session.image_updated(doc)
            self._update(canvas)
            return
        if self.mode == "create":
            shift, alt, ctrl = _mods(event)
            roi = self._shape(self.start, (x, y), shift, ctrl) if self.moved else None
            if roi is not None and _degenerate(roi):
                roi = None
            if roi is None and self.op is not None:
                doc.roi = self.base_roi
                self.op = self.base_roi = None
                self.mode = None
                self._update(canvas)
                return
            self._finish(canvas, roi)
        elif self.mode == "freehand":
            self.points.append((x, y))
            tool = self.current_tool()
            if tool == T.FREEROI:
                roi = S.PolygonRoi([p[0] for p in self.points], [p[1] for p in self.points],
                                   type=S.FREEROI) if len(self.points) >= 3 else None
            else:
                roi = S.PolygonRoi([p[0] for p in self.points], [p[1] for p in self.points],
                                   type=S.FREELINE) if len(self.points) >= 2 else None
            self._finish(canvas, roi)
        elif self.mode == "brush":
            self._brush_finish(canvas)

    def mouse_double_click(self, canvas, event, x, y) -> None:
        if self.mode in ("polygon", "polyline"):
            self._close_polygon(canvas)
        elif self.current_tool() == T.MAGNIFIER:
            canvas.original_scale()

    # ------------------------------------------------------------------ shapes
    def _shape(self, a, b, square: bool, center: bool):
        tool, variant = self.current_tool(), self.current_variant()
        if tool in (T.LINE,):
            x1, y1 = a
            x2, y2 = b
            if square:   # horizontal or vertical
                if abs(x2 - x1) >= abs(y2 - y1):
                    y2 = y1
                else:
                    x2 = x1
            o = self.options()
            lw = o.line_width if o else 1.0
            if variant == "Arrow tool":
                aw = float(o.extra.get("arrow_width", 2)) if o else 2.0
                r = S.LineRoi(x1, y1, x2, y2, arrow=True, stroke_width=aw)
                r.properties["_arrow_head"] = str(int(o.extra.get("arrow_size", 10))) if o \
                    else "10"
                if o and o.extra.get("arrow_double"):
                    r.properties["_arrow_double"] = "1"
                return r
            return S.LineRoi(x1, y1, x2, y2, width=lw, stroke_width=lw if lw > 1 else 0.0)
        if tool == T.RECTANGLE and variant == "Rotated Rectangle":
            length = math.hypot(b[0] - a[0], b[1] - a[1])
            return S.RotatedRectRoi(a[0], a[1], b[0], b[1], rect_width=max(1.0, length / 3))
        if tool == T.OVAL and variant == "Elliptical selections":
            return S.EllipseRoi(a[0], a[1], b[0], b[1], aspect_ratio=0.5)
        ax, ay = self._ip(*a)
        bx, by = self._ip(*b)
        w, h = bx - ax, by - ay
        if square:
            s = max(abs(w), abs(h))
            w, h = math.copysign(s, w or 1), math.copysign(s, h or 1)
        if center:
            x0, y0, w, h = ax - abs(w), ay - abs(h), 2 * abs(w), 2 * abs(h)
        else:
            x0, y0 = min(ax, ax + w), min(ay, ay + h)
            w, h = abs(w), abs(h)
        if w <= 0 or h <= 0:
            return None
        if tool == T.OVAL:
            return S.OvalRoi(x0, y0, w, h)
        o = self.options()
        cd = float(o.rounded_rect_diameter or 20) if variant == "Rounded Rectangle" and o \
            else (20.0 if variant == "Rounded Rectangle" else 0.0)
        return S.RectRoi(x0, y0, w, h, corner_diameter=cd)

    def _drag_status(self, roi) -> None:
        if roi is None:
            return
        cal = self.doc.calibration
        if isinstance(roi, S.LineRoi):
            ang = roi.angle()
            length = roi.length(cal.pixel_width, cal.pixel_height)
            self._status(f"x={roi.x2:g}, y={roi.y2:g}, angle={ang:.2f}, length={length:.2f}")
        else:
            x, y, w, h = roi.bounds()
            if cal.scaled():
                self._status(f"x={x * cal.pixel_width:.2f}, y={y * cal.pixel_height:.2f}, "
                             f"w={w * cal.pixel_width:.2f}, h={h * cal.pixel_height:.2f}")
            else:
                self._status(f"x={x:g}, y={y:g}, w={w:g}, h={h:g}")

    # ------------------------------------------------------------------ polygons
    def _add_vertex(self, canvas, x, y, alt) -> None:
        pts = self.points
        if self.mode == "polygon":
            sx, sy = canvas.image_to_screen(*pts[0])
            cx, cy = canvas.image_to_screen(x, y)
            if len(pts) >= 3 and math.hypot(sx - cx, sy - cy) <= CLOSE_DIST:
                self._close_polygon(canvas)
                return
            pts.append(self._snap(x, y))
        else:
            pts.append((x, y))
        if self.mode == "angle" and len(pts) == 3:
            self._finish(canvas, S.PolygonRoi([p[0] for p in pts], [p[1] for p in pts],
                                              type=S.ANGLE))
            return
        self._update(canvas)

    def _close_polygon(self, canvas) -> None:
        pts = _dedupe(self.points)
        if self.mode == "polygon":
            roi = S.PolygonRoi([p[0] for p in pts], [p[1] for p in pts]) if len(pts) >= 3 \
                else None
        else:
            roi = S.PolygonRoi([p[0] for p in pts], [p[1] for p in pts], type=S.POLYLINE) \
                if len(pts) >= 2 else None
        self._finish(canvas, roi)

    # ------------------------------------------------------------------ points
    def _point_click(self, canvas, x, y, shift, alt) -> None:
        doc = self.doc
        px, py = float(math.floor(x)), float(math.floor(y))
        roi = doc.roi
        multi = self._multi()
        o = self.options()
        counter = int(o.extra.get("point_counter", 0)) if o else 0
        if alt and isinstance(roi, S.PointRoi):
            xs, ys = roi.polygon()
            if len(xs):
                d = np.hypot(xs - px, ys - py)
                i = int(np.argmin(d))
                keep = np.arange(len(xs)) != i
                if keep.sum() == 0:
                    self._finish(canvas, None)
                    return
                counters = roi.counters[keep] if roi.counters is not None else None
                doc.roi = S.PointRoi(xs[keep], ys[keep], counters=counters)
                self._update(canvas)
            return
        if (multi or shift) and isinstance(roi, S.PointRoi):
            xs = np.append(roi.xs, px)
            ys = np.append(roi.ys, py)
            counters = None
            if roi.counters is not None or counter:
                old = roi.counters if roi.counters is not None else np.zeros(len(roi.xs), int)
                counters = np.append(old, counter)
            new = S.PointRoi(xs, ys, counters=counters)
        else:
            new = S.PointRoi([px], [py], counters=np.array([counter]) if counter else None)
        doc.roi = new
        if self.bar is not None:
            self.bar.session.image_updated(doc)
            self.bar.record_roi(new)
            if o is not None and o.point_auto_measure and "Measure" in self.bar.session.registry:
                self.bar.run_command("Measure", "")
            if o is not None and o.point_add_to_manager:
                self.bar.session.roi_manager.append(S.PointRoi([px], [py]))
                self.bar.session.emit("roi_manager_updated", self.bar.session.roi_manager)
        self._status(f"x={px:g}, y={py:g}")
        self._update(canvas)

    # ------------------------------------------------------------------ brush
    def _brush_start(self, canvas, x, y) -> None:
        doc = self.doc
        roi = doc.roi
        self.brush_mask = roi.get_mask((doc.height, doc.width)) if roi is not None and \
            roi.is_area() else np.zeros((doc.height, doc.width), bool)
        self.brush_add = roi is None or not roi.is_area() or roi.contains(int(x), int(y))
        self.mode = "brush"
        self._brush_paint(x, y)

    def _brush_paint(self, x, y) -> None:
        o = self.options()
        r = float(o.extra.get("brush_size", 15)) / 2 if o else 7.5
        h, w = self.brush_mask.shape
        y0, y1 = max(int(y - r) - 1, 0), min(int(y + r) + 2, h)
        x0, x1 = max(int(x - r) - 1, 0), min(int(x + r) + 2, w)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        disc = (xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2 <= r * r
        self.brush_mask[y0:y1, x0:x1] = np.where(disc, self.brush_add,
                                                 self.brush_mask[y0:y1, x0:x1])
        self.doc.roi = mask_to_roi(self.brush_mask)

    def _brush_finish(self, canvas) -> None:
        roi = mask_to_roi(self.brush_mask) if self.brush_mask is not None else None
        self.brush_mask = None
        self.op = self.base_roi = None
        self._finish(canvas, roi)

    # ------------------------------------------------------------------ colour picker
    def pick_color(self, canvas, x, y, background: bool = False) -> None:
        from rsstatchem.core.imaging.lut import render
        doc = self.doc
        ix, iy = self._ip(x, y)
        if not (0 <= ix < doc.width and 0 <= iy < doc.height):
            return
        r, g, b = (int(v) for v in render(doc)[iy, ix])
        color = f"#{r:02x}{g:02x}{b:02x}"
        if self.bar is not None:
            self.bar.set_color(color, background)
        self._status(f"{'Background' if background else 'Foreground'}: {r},{g},{b}")

    # ------------------------------------------------------------------ keys
    def key_press(self, canvas, event) -> bool:
        key = event.key()
        shift, alt, ctrl = _mods(event)
        doc = self.doc
        if self.typing and isinstance(doc.roi, S.TextRoi) and not ctrl:
            roi = doc.roi
            if key == Qt.Key.Key_Escape:
                self.typing = False
                if self.bar is not None:
                    self.bar.record_roi(roi)
            elif key in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
                roi.text = roi.text[:-1]
            elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                roi.text += "\n"
            elif event.text() and event.text().isprintable():
                roi.text += event.text()
            else:
                return False
            self._update(canvas)
            return True
        if key == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.space_hand = True
            return True
        if key == Qt.Key.Key_Escape:
            if self.mode in ("polygon", "polyline", "angle", "create", "freehand", "brush"):
                self.mode = None
                self.points = []
                doc.roi = self.base_roi
                self._update(canvas)
            if self.bar is not None:
                self.bar.session.cancel_requested = False
                self.bar.abort()
            return True
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.mode in ("polygon",
                                                                          "polyline"):
            self._close_polygon(canvas)
            return True
        arrows = {Qt.Key.Key_Left: (-1, 0), Qt.Key.Key_Right: (1, 0), Qt.Key.Key_Up: (0, -1),
                  Qt.Key.Key_Down: (0, 1)}
        if key in arrows and not ctrl:
            dx, dy = arrows[key]
            roi = doc.roi
            if roi is not None:
                if alt and isinstance(roi, (S.RectRoi, S.OvalRoi)):
                    r = roi.copy()
                    r.width = max(1.0, r.width + dx)
                    r.height = max(1.0, r.height + dy)
                    doc.roi = r
                else:
                    r = roi.copy()
                    step = 5 if shift else 1
                    r.translate(dx * step, dy * step)
                    doc.roi = r
                if self.bar is not None:
                    self.bar.session.image_updated(doc)
                self._update(canvas)
                return True
            if dy == 0:
                self.window.step_slice(dx)
            elif dy < 0:
                canvas.zoom_in()
            else:
                canvas.zoom_out()
            return True
        if key in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete) and doc.roi is not None and \
                self.bar is not None:
            self.bar.run_command("Clear", "slice" if doc.stack_size > 1 else "")
            return True
        if self.bar is not None:
            return self.bar.handle_key(event)
        return False

    def key_release(self, canvas, event) -> bool:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self.space_hand = False
            return True
        return False

    def abort(self) -> None:
        self.mode = None
        self.points = []
        self.typing = False

    # ------------------------------------------------------------------ painting
    def paint(self, p, canvas) -> None:
        if self.mode in ("polygon", "polyline", "angle") and self.points:
            p.setPen(QPen(QColor(canvas.selection_color), 1))
            pts = [QPointF(*canvas.image_to_screen(x, y)) for x, y in self.points]
            pts.append(QPointF(*canvas.image_to_screen(*self.cursor)))
            p.drawPolyline(QPolygonF(pts))
            if self.mode == "polygon":
                sx, sy = canvas.image_to_screen(*self.points[0])
                p.drawRect(QRectF(sx - 3, sy - 3, 6, 6))
        elif self.mode == "freehand" and len(self.points) > 1:
            p.setPen(QPen(QColor(canvas.selection_color), 1))
            p.drawPolyline(QPolygonF([QPointF(*canvas.image_to_screen(x, y))
                                      for x, y in self.points]))


def _seg_dist(px, py, x1, y1, x2, y2) -> float:
    dx, dy = x2 - x1, y2 - y1
    if dx == dy == 0:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def _dedupe(pts):
    out = []
    for p in pts:
        if not out or (abs(out[-1][0] - p[0]) > 1e-9 or abs(out[-1][1] - p[1]) > 1e-9):
            out.append(p)
    return out


def _degenerate(roi) -> bool:
    if isinstance(roi, S.LineRoi):
        return roi.x1 == roi.x2 and roi.y1 == roi.y2
    x, y, w, h = roi.bounds()
    return w <= 0 or h <= 0


def mask_to_roi(mask: np.ndarray):
    """Area selection from a boolean mask (traced polygon, or a composite of outlines)."""
    from rsstatchem.core.imaging.io.roi_codec import mask_outlines
    loops = mask_outlines(mask)
    if not loops:
        return None
    if len(loops) == 1:
        lp = loops[0]
        return S.PolygonRoi(lp[:, 0], lp[:, 1], type=S.TRACED_ROI)
    return S.CompositeRoi(parts=[("xor" if i else "or", S.PolygonRoi(lp[:, 0], lp[:, 1],
                                                                       type=S.TRACED_ROI))
                                 for i, lp in enumerate(loops)])


def resize_roi(orig, handle: int, x: float, y: float, keep_square: bool = False):
    """``orig`` with handle ``handle`` dragged to (x, y) (see ``canvas.handle_points``)."""
    r = orig.copy()
    if isinstance(r, (S.RectRoi, S.OvalRoi)):
        x0, y0, x1, y1 = r.x, r.y, r.x + r.width, r.y + r.height
        xi, yi = float(math.floor(x + 0.5)), float(math.floor(y + 0.5))
        if handle in (0, 6, 7):
            x0 = xi
        if handle in (2, 3, 4):
            x1 = xi
        if handle in (0, 1, 2):
            y0 = yi
        if handle in (4, 5, 6):
            y1 = yi
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        w, h = max(1.0, x1 - x0), max(1.0, y1 - y0)
        if keep_square:
            w = h = max(w, h)
        r.x, r.y, r.width, r.height = x0, y0, w, h
    elif isinstance(r, S.LineRoi):
        if handle == 0:
            r.x1, r.y1 = x, y
        elif handle == 2:
            r.x2, r.y2 = x, y
        else:
            cx, cy = (orig.x1 + orig.x2) / 2, (orig.y1 + orig.y2) / 2
            r.translate(x - cx, y - cy)
    elif isinstance(r, (S.EllipseRoi, S.RotatedRectRoi)):
        if handle == 0:
            r.x1, r.y1 = x, y
        else:
            r.x2, r.y2 = x, y
    elif isinstance(r, S.PolygonRoi):
        xs, ys = r.xs.copy(), r.ys.copy()
        if 0 <= handle < len(xs):
            snap = r.type == S.POLYGON
            xs[handle] = math.floor(x) if snap else x
            ys[handle] = math.floor(y) if snap else y
        r.xs, r.ys = xs, ys
    return r

"""ImageJ's wand (tracing) tool: outlines of thresholded or similar-valued areas.

``autoOutline`` finds the border right of the start point and traces it; the outline
polygon has its vertices on pixel corners (a vertical boundary at x separates pixels x-1 and
x). Modes: "Legacy" (ImageJ <= 1.42p behaviour: 8-connected for thresholded images, otherwise
4-connected unless on a thin line), "4-connected", "8-connected"; ``tolerance`` selects values
within +/- tolerance of the start pixel.

Ported from ImageJ 1.54 ij/gui/Wand.java (public domain, NIH; M. Schmid, W. Rasband).
"""
from __future__ import annotations

import numpy as np

LEGACY_MODE, FOUR_CONNECTED, EIGHT_CONNECTED = 1, 4, 8
_THRESHOLDED = 256


class Wand:
    """``Wand(ip)`` on a 2-D array (8/16/32-bit values, or an (H, W, 3) RGB array)."""

    def __init__(self, pixels: np.ndarray, rgb_exact: bool = True):
        p = np.asarray(pixels)
        if p.ndim == 3:   # RGB: exact colour match when tolerance is 0, else gray value
            self._rgb = p
            self._packed = (p[..., 0].astype(np.int64) << 16) | (p[..., 1].astype(np.int64) << 8) \
                | p[..., 2].astype(np.int64)
            self._gray = p.astype(np.float64).mean(axis=2)
            self.pixels = self._packed.astype(np.float64)
        else:
            self._rgb = None
            self.pixels = p.astype(np.float64) if p.dtype != np.float64 else p
        self.height, self.width = self.pixels.shape
        self.xpoints: list[int] = []
        self.ypoints: list[int] = []
        self.lower = self.upper = 0.0
        self.all_points = False
        self._inside = None

    @property
    def npoints(self) -> int:
        return len(self.xpoints)

    # ------------------------------------------------------------------ API
    def auto_outline_threshold(self, x: int, y: int, lower: float, upper: float,
                               mode: int = LEGACY_MODE) -> bool:
        """Trace the thresholded object (lower..upper) at or right of (x, y)."""
        self.lower, self.upper = float(np.float32(lower)), float(np.float32(upper))
        return self._auto_outline(x, y, 0.0, mode | _THRESHOLDED)

    def auto_outline(self, x: int, y: int, tolerance: float = 0.0,
                     mode: int = LEGACY_MODE) -> bool:
        """Trace the area of values within ``tolerance`` of the value at (x, y)."""
        return self._auto_outline(x, y, tolerance, mode)

    # ------------------------------------------------------------------ internals
    def _value(self, x, y):
        return self.pixels[y, x]

    def _auto_outline(self, start_x, start_y, tolerance, mode) -> bool:
        self.xpoints, self.ypoints = [], []
        if not (0 <= start_x < self.width and 0 <= start_y < self.height):
            return False
        if self._rgb is not None and tolerance != 0:
            self.pixels = self._gray
        v0 = self._value(start_x, start_y)
        if np.isnan(v0):
            return False
        threshold_mode = bool(mode & _THRESHOLDED)
        legacy = bool(mode & LEGACY_MODE) and tolerance == 0
        if not threshold_mode:
            self.lower = float(np.float32(v0 - tolerance))
            self.upper = float(np.float32(v0 + tolerance))
        with np.errstate(invalid="ignore"):
            pix32 = self.pixels.astype(np.float32)
            ins = (pix32 >= np.float32(self.lower)) & (pix32 <= np.float32(self.upper))
        self._inside = np.pad(ins, 1)
        x, y = start_x, start_y
        if self._ins(x, y):
            seed_x = x
            while True:
                x += 1
                if not self._ins(x, y):
                    break
        else:
            while True:
                x += 1
                if x >= self.width:
                    return False
                if self._ins(x, y):
                    break
            seed_x = x
        if legacy:
            four = (not threshold_mode) and not self._is_line(x, y)
        else:
            four = bool(mode & FOUR_CONNECTED)
        first = True
        while True:
            inside_selected = self._trace_edge(x, y, four)
            if legacy:
                return True
            if inside_selected:
                if first:
                    return True
                if min(self.xpoints) <= seed_x:
                    from matplotlib.path import Path
                    poly = Path(np.column_stack([self.xpoints, self.ypoints]))
                    if poly.contains_point((seed_x, start_y)):
                        return True
            first = False
            if not self._ins(x, y):
                while True:
                    x += 1
                    if x > self.width:
                        raise RuntimeError("Wand Malfunction")
                    if self._ins(x, y):
                        break
            while True:
                x += 1
                if not self._ins(x, y):
                    break

    def _ins(self, x, y) -> bool:
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return bool(self._inside[y + 1, x + 1])

    def _ins_dir(self, x, y, d) -> bool:
        d &= 3
        if d == 0:
            return self._ins(x, y)
        if d == 1:
            return self._ins(x, y - 1)
        if d == 2:
            return self._ins(x - 1, y - 1)
        return self._ins(x - 1, y)

    def _trace_edge(self, start_x, start_y, four) -> bool:
        xs, ys = [], []
        if self._ins(start_x, start_y):
            start_dir = 1
        else:
            start_dir = 3
            start_y += 1
        x, y = start_x, start_y
        direction = start_dir
        ins = self._ins_dir
        allp = self.all_points
        while True:
            if four:
                nd = direction
                while True:
                    if not ins(x, y, nd):
                        break
                    nd += 1
                    if nd >= direction + 2:
                        break
                nd -= 1
            else:
                nd = direction + 1
                while True:
                    if ins(x, y, nd):
                        break
                    nd -= 1
                    if nd < direction:
                        break
            if allp or nd != direction:
                xs.append(x)
                ys.append(y)
            r = nd & 3
            if r == 0:
                x += 1
            elif r == 1:
                y -= 1
            elif r == 2:
                x -= 1
            else:
                y += 1
            direction = nd
            if x == start_x and y == start_y and (direction & 3) == start_dir:
                break
        if xs and xs[0] != x and not allp:
            xs.append(x)
            ys.append(y)
        self.xpoints, self.ypoints = xs, ys
        return direction <= 0

    def _is_line(self, xs, ys) -> bool:
        r = 5
        xmin, xmax = xs, min(xs + 2 * r, self.width - 1)
        ymin, ymax = max(ys - r, 0), min(ys + r, self.height - 1)
        area = (xmax - xmin + 1) * (ymax - ymin + 1)
        if area <= 0:
            return False
        count = int(self._inside[ymin + 1:ymax + 2, xmin + 1:xmax + 2].sum())
        return count / area < 0.25


def do_wand(image, x: int, y: int, tolerance: float = 0.0, mode: str = "Legacy",
            smooth: bool = False):
    """``IJ.doWand``: a traced PolygonRoi of the object at (x, y) of the active plane, using
    the image threshold when one is set (legacy and tolerance 0), else the tolerance."""
    from rsstatchem.core.imaging.roi.shapes import FREEROI, TRACED_ROI, PolygonRoi

    modes = {"Legacy": LEGACY_MODE, "4-connected": FOUR_CONNECTED,
             "8-connected": EIGHT_CONNECTED}
    m = modes.get(mode, LEGACY_MODE)
    w = Wand(image.plane())
    th = image.threshold
    if th is not None and tolerance == 0 and not image.is_rgb:
        ok = w.auto_outline_threshold(x, y, th[0], th[1], m)
    else:
        ok = w.auto_outline(x, y, tolerance, m)
    if not ok or w.npoints == 0:
        return None
    xs, ys = np.asarray(w.xpoints, float), np.asarray(w.ypoints, float)
    if smooth:
        from rsstatchem.core.imaging.roi.geometry import interpolated_polygon
        xs, ys = interpolated_polygon(xs, ys, 1.0, smooth=True)
        return PolygonRoi(xs, ys, type=FREEROI)
    return PolygonRoi(xs, ys, type=TRACED_ROI)

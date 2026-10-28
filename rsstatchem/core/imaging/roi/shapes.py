"""ROI data classes (ImageJ ``ij.gui.Roi`` family) and the overlay.

Coordinates are ImageJ image coordinates: (0, 0) is the top-left corner of the top-left pixel,
pixel (i, j) covers [i, i+1) x [j, j+1) and its centre is (i + 0.5, j + 0.5). Area ROIs are
rasterised by :func:`rsstatchem.core.imaging.roi.mask.rasterize` (owned by WP-C, which ports
ImageJ's PolygonFiller / OvalRoi masks behind the same function): a rectangle with integer bounds
covers exactly its pixel block; polygons include a pixel when its centre is inside. Polygon
vertices of traced ROIs lie on pixel corners; point ROIs at integer (x, y) refer to pixel (x, y).

``position`` is the ImageJ stack position the ROI belongs to: (c, z, t), 1-based, 0 = all, as
stored in .roi files and overlays.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np

# ImageJ Roi.getType() constants
RECTANGLE, OVAL, POLYGON, FREEROI, TRACED_ROI, LINE, POLYLINE, FREELINE, ANGLE, COMPOSITE, \
    POINT = range(11)
TYPE_NAMES = {RECTANGLE: "Rectangle", OVAL: "Oval", POLYGON: "Polygon", FREEROI: "Freehand",
              TRACED_ROI: "Traced", LINE: "Straight Line", POLYLINE: "Polyline",
              FREELINE: "Freeline", ANGLE: "Angle", COMPOSITE: "Composite", POINT: "Point"}


@dataclass
class Roi:
    """Base class: style, name and stack position shared by every ROI."""
    name: str = field(default="", kw_only=True)
    stroke_color: str | None = field(default=None, kw_only=True)   # "#rrggbb" or "#aarrggbb"
    fill_color: str | None = field(default=None, kw_only=True)
    stroke_width: float = field(default=0.0, kw_only=True)
    position: tuple[int, int, int] = field(default=(0, 0, 0), kw_only=True)  # c, z, t; 0 = all
    properties: dict = field(default_factory=dict, kw_only=True)

    type: int = field(default=-1, kw_only=True)   # ImageJ type constant, set per class

    # -- geometry common API ------------------------------------------------------------
    def bounds(self) -> tuple[float, float, float, float]:
        """(x, y, width, height) of the bounding box in image coordinates."""
        xs, ys = self.polygon()
        if len(xs) == 0:
            return 0.0, 0.0, 0.0, 0.0
        return float(min(xs)), float(min(ys)), float(max(xs) - min(xs)), \
            float(max(ys) - min(ys))

    def polygon(self) -> tuple[np.ndarray, np.ndarray]:
        """Float vertex coordinates (ImageJ ``getFloatPolygon``)."""
        raise NotImplementedError

    def is_area(self) -> bool:
        return self.type in (RECTANGLE, OVAL, POLYGON, FREEROI, TRACED_ROI, COMPOSITE)

    def is_line(self) -> bool:
        return self.type in (LINE, POLYLINE, FREELINE)

    def get_mask(self, shape: tuple[int, int]) -> np.ndarray:
        """Boolean (H, W) mask of the pixels inside this area ROI, clipped to ``shape``."""
        from rsstatchem.core.imaging.roi.mask import rasterize
        return rasterize(self, shape)

    def contains(self, x: int, y: int) -> bool:
        """Is pixel (x, y) inside (area ROIs)?"""
        bx, by, bw, bh = self.bounds()
        shape = (int(math.ceil(by + bh)) + 1, int(math.ceil(bx + bw)) + 1)
        if not (0 <= y < shape[0] and 0 <= x < shape[1]):
            return False
        return bool(self.get_mask(shape)[y, x])

    def translate(self, dx: float, dy: float) -> None:
        raise NotImplementedError

    def copy(self) -> "Roi":
        return copy.deepcopy(self)

    @property
    def type_name(self) -> str:
        return TYPE_NAMES.get(self.type, "Roi")


@dataclass
class RectRoi(Roi):
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    corner_diameter: float = 0.0        # rounded rectangle when > 0
    type: int = field(default=RECTANGLE, kw_only=True)

    def bounds(self):
        return float(self.x), float(self.y), float(self.width), float(self.height)

    def polygon(self):
        x, y, w, h = self.bounds()
        return np.array([x, x + w, x + w, x]), np.array([y, y, y + h, y + h])

    def translate(self, dx, dy):
        self.x += dx
        self.y += dy


@dataclass
class OvalRoi(Roi):
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    type: int = field(default=OVAL, kw_only=True)

    def bounds(self):
        return float(self.x), float(self.y), float(self.width), float(self.height)

    def polygon(self, n: int = 72):
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        cx, cy = self.x + self.width / 2, self.y + self.height / 2
        return cx + self.width / 2 * np.cos(a), cy + self.height / 2 * np.sin(a)

    def translate(self, dx, dy):
        self.x += dx
        self.y += dy


@dataclass
class PolygonRoi(Roi):
    """Polygon, freehand, traced (area) and polyline, freeline, angle (open) ROIs."""
    xs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ys: np.ndarray = field(default_factory=lambda: np.zeros(0))
    type: int = field(default=POLYGON, kw_only=True)
    spline_fit: bool = False

    def __post_init__(self):
        self.xs = np.asarray(self.xs, dtype=float)
        self.ys = np.asarray(self.ys, dtype=float)
        if self.type not in (POLYGON, FREEROI, TRACED_ROI, POLYLINE, FREELINE, ANGLE):
            raise ValueError(f"PolygonRoi cannot have type {self.type}")

    def polygon(self):
        return self.xs.copy(), self.ys.copy()

    def translate(self, dx, dy):
        self.xs = self.xs + dx
        self.ys = self.ys + dy


@dataclass
class LineRoi(Roi):
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    width: float = 1.0                  # line width (>1: wide line for profiles)
    arrow: bool = False
    type: int = field(default=LINE, kw_only=True)

    def polygon(self):
        return np.array([self.x1, self.x2]), np.array([self.y1, self.y2])

    def length(self, pixel_width: float = 1.0, pixel_height: float = 1.0) -> float:
        return math.hypot((self.x2 - self.x1) * pixel_width, (self.y2 - self.y1) * pixel_height)

    def angle(self) -> float:
        """ImageJ angle in degrees (counter-clockwise, y up)."""
        return math.degrees(math.atan2(self.y1 - self.y2, self.x2 - self.x1))

    def translate(self, dx, dy):
        self.x1 += dx
        self.x2 += dx
        self.y1 += dy
        self.y2 += dy


@dataclass
class PointRoi(Roi):
    xs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ys: np.ndarray = field(default_factory=lambda: np.zeros(0))
    counters: np.ndarray | None = None      # multi-point counter index per point
    positions: np.ndarray | None = None     # 1-based stack position per point (0 = none)
    type: int = field(default=POINT, kw_only=True)

    def __post_init__(self):
        self.xs = np.asarray(self.xs, dtype=float)
        self.ys = np.asarray(self.ys, dtype=float)

    def polygon(self):
        return self.xs.copy(), self.ys.copy()

    def translate(self, dx, dy):
        self.xs = self.xs + dx
        self.ys = self.ys + dy


@dataclass
class RotatedRectRoi(Roi):
    """ImageJ RotatedRectRoi: the centre line (x1, y1)-(x2, y2) and the rectangle width."""
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    rect_width: float = 0.0
    # ImageJ reports rotated rectangles as polygons
    type: int = field(default=POLYGON, kw_only=True)

    def polygon(self):
        dx, dy = self.x2 - self.x1, self.y2 - self.y1
        n = math.hypot(dx, dy) or 1.0
        px, py = -dy / n * self.rect_width / 2, dx / n * self.rect_width / 2
        return (np.array([self.x1 + px, self.x2 + px, self.x2 - px, self.x1 - px]),
                np.array([self.y1 + py, self.y2 + py, self.y2 - py, self.y1 - py]))

    def translate(self, dx, dy):
        self.x1 += dx
        self.x2 += dx
        self.y1 += dy
        self.y2 += dy


@dataclass
class EllipseRoi(Roi):
    """ImageJ EllipseRoi: major axis (x1, y1)-(x2, y2) and aspect ratio minor/major."""
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    aspect_ratio: float = 1.0
    # ImageJ reports ellipses as freehand ROIs
    type: int = field(default=FREEROI, kw_only=True)

    def polygon(self, n: int = 72):
        cx, cy = (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2
        a = math.hypot(self.x2 - self.x1, self.y2 - self.y1) / 2
        b = a * self.aspect_ratio
        phi = math.atan2(self.y2 - self.y1, self.x2 - self.x1)
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        ex, ey = a * np.cos(t), b * np.sin(t)
        return (cx + ex * math.cos(phi) - ey * math.sin(phi),
                cy + ex * math.sin(phi) + ey * math.cos(phi))

    def translate(self, dx, dy):
        self.x1 += dx
        self.x2 += dx
        self.y1 += dy
        self.y2 += dy


@dataclass
class TextRoi(Roi):
    x: float = 0.0
    y: float = 0.0
    text: str = ""
    font: str = "SansSerif"
    font_size: float = 18.0
    angle: float = 0.0
    # ImageJ TextRoi is a rectangle subclass
    type: int = field(default=RECTANGLE, kw_only=True)

    def polygon(self):
        w = max(len(line) for line in self.text.splitlines() or [""]) * self.font_size * 0.6
        h = max(1, len(self.text.splitlines())) * self.font_size * 1.2
        return (np.array([self.x, self.x + w, self.x + w, self.x]),
                np.array([self.y, self.y, self.y + h, self.y + h]))

    def is_area(self) -> bool:
        return False

    def translate(self, dx, dy):
        self.x += dx
        self.y += dy


@dataclass
class CompositeRoi(Roi):
    """ImageJ ShapeRoi: area ROIs combined in order; ``parts`` = [(op, roi)] with op in
    "or" (add), "and", "xor", "not" (subtract). The first op is ignored (start shape)."""
    parts: list = field(default_factory=list)
    type: int = field(default=COMPOSITE, kw_only=True)

    def polygon(self):
        xs, ys = [], []
        for _, roi in self.parts:
            px, py = roi.polygon()
            xs.append(px)
            ys.append(py)
        if not xs:
            return np.zeros(0), np.zeros(0)
        return np.concatenate(xs), np.concatenate(ys)

    def bounds(self):
        if not self.parts:
            return 0.0, 0.0, 0.0, 0.0
        boxes = [r.bounds() for _, r in self.parts]
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[0] + b[2] for b in boxes)
        y1 = max(b[1] + b[3] for b in boxes)
        return x0, y0, x1 - x0, y1 - y0

    def translate(self, dx, dy):
        for _, roi in self.parts:
            roi.translate(dx, dy)


@dataclass
class Overlay:
    """Non-destructive overlay of an image (ImageJ ``ij.gui.Overlay``)."""
    rois: list = field(default_factory=list)
    show_labels: bool = False
    draw_names: bool = False
    draw_backgrounds: bool = False
    label_color: str | None = None
    label_font_size: int = 9
    hidden: bool = False

    def add(self, roi: Roi) -> None:
        self.rois.append(roi)

    def remove(self, roi: Roi) -> None:
        self.rois.remove(roi)

    def clear(self) -> None:
        self.rois.clear()

    def __len__(self) -> int:
        return len(self.rois)

    def __iter__(self):
        return iter(self.rois)

    def copy(self) -> "Overlay":
        return copy.deepcopy(self)

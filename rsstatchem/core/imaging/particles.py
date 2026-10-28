"""Analyze > Analyze Particles (``ij.plugin.filter.ParticleAnalyzer``).

The image is scanned row by row; each pixel inside the threshold starts a particle: its outer
boundary is traced with the wand (8-connected, or 4-connected), the particle is measured with
the Set Measurements options (pixels of the 8-connected component; with "include holes" every
pixel inside the outline), filtered by size (pixels or calibrated units) and circularity
(4 pi area / perimeter^2 with the traced perimeter), and erased so it is not found again.
Options: exclude on edges (bounding box touches the image or selection edge), include holes,
record starts, composite ROIs (outlines with holes), add to manager, summarize (Count, Total
Area, Average Size, %Area and the means of the measured columns), stacks, and the Show
outputs Outlines, Bare Outlines, Ellipses, Masks, Count Masks, Overlay, Overlay Masks
(in situ: replace the image).

Ported from ImageJ 1.54 ij/plugin/filter/ParticleAnalyzer.java and ij/process/FloodFiller.java
(public domain, NIH; W. Rasband et al.).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from rsstatchem.core.imaging import measure as M
from rsstatchem.core.imaging.image import COLOR_RGB, GRAY8, GRAY16, GRAY32, ImageDoc
from rsstatchem.core.imaging.lut import LUT
from rsstatchem.core.imaging.results import ResultsTable
from rsstatchem.core.imaging.roi import geometry as G
from rsstatchem.core.imaging.roi import shapes as S
from rsstatchem.core.imaging.roi.mask import polygon_mask

SHOW_CHOICES = ("Nothing", "Overlay", "Overlay Masks", "Outlines", "Bare Outlines", "Ellipses",
                "Masks", "Count Masks")

# glasbey-like distinct colours for "Overlay Masks" (index + 5 skips white/black, as ImageJ)
_MASK_COLORS = ["#00ffff", "#ff0000", "#00ff00", "#0000ff", "#ffff00", "#ff00ff", "#ff8000",
                "#0080ff", "#80ff00", "#ff0080", "#8000ff", "#00ff80", "#804000", "#008040",
                "#400080", "#808000", "#008080", "#800080", "#ff8080", "#80ff80", "#8080ff"]


@dataclass
class ParticleOptions:
    min_size: float = 0.0            # in pixels (convert calibrated sizes before)
    max_size: float = math.inf
    min_circularity: float = 0.0
    max_circularity: float = 1.0
    show: str = "Nothing"
    display: bool = False
    exclude: bool = False
    clear: bool = False
    include_holes: bool = False
    summarize: bool = False
    overlay: bool = False
    add_to_manager: bool = False
    composite: bool = False
    record_starts: bool = False
    in_situ: bool = False
    four_connected: bool = False
    stack: bool = False


@dataclass
class ParticleResult:
    results: ResultsTable
    summary: ResultsTable | None = None
    output: ImageDoc | None = None
    rois: list = field(default_factory=list)
    overlay: S.Overlay | None = None
    count: int = 0
    total_count: int = 0


def parse_range(text: str, default_max: float) -> tuple[float, float]:
    """"0-Infinity" / "100-500" / "0.5" -> (min, max) as the dialog parses them."""
    parts = [p for p in str(text).replace(" ", "-").split("-") if p != ""]

    def num(s):
        try:
            return float(s.replace("Infinity", "inf"))
        except ValueError:
            return float("nan")

    lo = num(parts[0]) if parts else 0.0
    hi = num(parts[1]) if len(parts) >= 2 else float("nan")
    return (0.0 if math.isnan(lo) else lo), (default_max if math.isnan(hi) else hi)


def threshold_levels(image, plane, black_background: bool):
    """(level1, level2, fillColor) like ``setThresholdLevels``; raises ValueError when no
    threshold is set and the image is not binary."""
    th = image.threshold
    itype = image.type
    if th is None:
        if itype not in (GRAY8, COLOR_RGB) or plane.ndim == 3:
            raise ValueError("A threshold has not been set using the\nImage>Adjust>Threshold "
                             "tool or the \nsetThreshold(min,max) macro function.")
        vals = np.unique(plane)
        if not np.all(np.isin(vals, (0, 255))):
            raise ValueError("A threshold has not been set using the\nImage>Adjust>Threshold "
                             "tool or the \nsetThreshold(min,max) macro function.")
        level1 = level2 = 255.0
        fill = 64.0
        if not black_background and not image.is_inverted_lut():
            level1 = level2 = 0.0
            fill = 192.0
        return level1, level2, fill, True
    level1, level2 = float(th[0]), float(th[1])
    fill = 0.0
    if itype in (GRAY8, COLOR_RGB):
        if level1 > 0:
            fill = 0.0
        elif level2 < 255:
            fill = 255.0
    elif itype == GRAY16:
        if level1 > 0:
            fill = 0.0
        elif level2 < 65535:
            fill = 65535.0
    else:
        fill = -M.FLOAT_MAX
    return level1, level2, fill, False


def trace_outline(ins_padded: np.ndarray, x: int, y: int, four: bool = False):
    """Wand ``autoOutline`` + ``traceEdge`` from the inside pixel (x, y): the border is
    searched to the right of the seed, then traced (``ins_padded`` has a one-pixel False
    border). Returns integer vertex lists in ImageJ's order."""
    ins = ins_padded

    def inside(px, py, d):
        d &= 3
        if d == 0:
            return ins[py + 1, px + 1]
        if d == 1:
            return ins[py, px + 1]
        if d == 2:
            return ins[py, px]
        return ins[py + 1, px]
    while ins[y + 1, x + 1]:
        x += 1
    xs, ys = [], []
    start_dir = 3          # (x, y) is outside, (x-1, y) inside: start downwards
    y += 1
    start_x, start_y = x, y
    direction = start_dir
    while True:
        if four:
            nd = direction
            while True:
                if not inside(x, y, nd):
                    break
                nd += 1
                if nd >= direction + 2:
                    break
            nd -= 1
        else:
            nd = direction + 1
            while True:
                if inside(x, y, nd):
                    break
                nd -= 1
                if nd < direction:
                    break
        if nd != direction:
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
    if xs and xs[0] != x:
        xs.append(x)
        ys.append(y)
    return xs, ys


def line_to(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, value) -> None:
    """``ImageProcessor.moveTo/lineTo`` with line width 1."""
    dx, dy = x2 - x1, y2 - y1
    n = max(abs(dx), abs(dy))
    xinc = dx / n if dx != 0 and n != 0 else 0.0
    yinc = dy / n if dy != 0 and n != 0 else 0.0
    k = np.arange(n + 1)
    xs = np.floor(x1 + k * xinc + 0.5).astype(np.int64)
    ys = np.floor(y1 + k * yinc + 0.5).astype(np.int64)
    h, w = img.shape[:2]
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    img[ys[ok], xs[ok]] = value


def draw_polygon(img, xs, ys, value, closed=True) -> None:
    n = len(xs)
    for i in range(n - 1 + (1 if closed else 0)):
        j = (i + 1) % n
        line_to(img, int(round(xs[i])), int(round(ys[i])), int(round(xs[j])),
                int(round(ys[j])), value)


def draw_label(img: np.ndarray, text: str, x: int, y: int, value, size: int = 9) -> None:
    """Draw ``text`` with its baseline at (x, y) (ImageJ ``drawString``)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:  # pragma: no cover
        return
    font = ImageFont.load_default()
    canvas = Image.new("L", (len(text) * 8 + 4, size + 8), 0)
    d = ImageDraw.Draw(canvas)
    d.text((0, 0), text, fill=255, font=font)
    arr = np.asarray(canvas) > 127
    ys, xs = np.nonzero(arr)
    if not len(xs):
        return
    top = y - (ys.max() + 1)
    xx, yy = xs + x, ys + top
    h, w = img.shape[:2]
    ok = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
    img[yy[ok], xx[ok]] = value


def text_width(text: str) -> int:
    return 6 * len(text)


def outlines_lut(font_color=(255, 0, 0)) -> LUT:
    """Gray LUT with index 1 = the label colour (``makeCustomLut``)."""
    r = np.arange(256, dtype=np.uint8)
    g, b = r.copy(), r.copy()
    rr = r.copy()
    rr[1], g[1], b[1] = font_color
    return LUT(rr, g, b, "outlines")


class ParticleAnalyzer:
    """Runs the analysis on one image (all slices with ``options.stack``)."""

    def __init__(self, image, flags: int, options: ParticleOptions, rt: ResultsTable | None = None,
                 summary: ResultsTable | None = None, black_background: bool = False,
                 redirect=None, precision: int = 3, roi=None, progress=None,
                 font_size: int = 9):
        self.image = image
        self.opt = options
        self.flags = (flags & ~M.LIMIT)
        if options.summarize:
            self.flags |= M.AREA
        if options.show == "Ellipses":
            self.flags |= M.ELLIPSE
        self.rt = rt if rt is not None else ResultsTable.results()
        self.summary = summary
        self.black_background = black_background
        self.redirect = redirect
        self.precision = precision
        self.roi = image.roi if roi is None else roi
        self.progress = progress
        self.font_size = font_size
        self.rois: list = []
        self.overlay: S.Overlay | None = None
        self.output_planes: list[np.ndarray] = []
        self.total_count = 0

    # ------------------------------------------------------------------ public
    def run(self) -> ParticleResult:
        im = self.image
        opt = self.opt
        if opt.clear:
            self.rt.reset()
        if opt.stack and im.stack_size > 1:
            indices = list(range(im.stack_size))
        else:
            indices = [im.current_index]
        saved = im.position
        count = 0
        for n, index in enumerate(indices):
            im.set_position(*im.position_of(index))
            count = self.analyze_plane(index)
            if self.progress is not None:
                self.progress((n + 1) / len(indices))
        im.set_position(*saved)
        output = self._make_output(indices)
        if self.overlay is not None and len(self.overlay):
            if opt.stack or im.stack_size == 1 or im.overlay is None:
                im.overlay = self.overlay
            else:
                for r in self.overlay:
                    im.overlay.add(r)
        return ParticleResult(self.rt, self.summary, output, self.rois, self.overlay, count,
                              self.total_count)

    # ------------------------------------------------------------------ one plane
    def analyze_plane(self, index: int) -> int:
        im, opt = self.image, self.opt
        plane = im.plane_at(index)
        if plane.ndim == 3:
            raise ValueError("RGB images must be thresholded using\n"
                             "Image>Adjust>Color Threshold.")
        level1, level2, fill, _ = threshold_levels(im, plane, self.black_background)
        h, w = plane.shape
        cal = (self.redirect or im).calibration
        work = plane.copy()
        if im.type == GRAY32:
            with np.errstate(invalid="ignore"):
                ins = (work >= np.float32(level1)) & (work <= np.float32(level2))
        else:
            ins = (work >= level1) & (work <= level2)
            fill = float(np.clip(fill, 0, 255 if work.dtype == np.uint8 else 65535))
        # selection
        rect, sel_mask = M.roi_rect_and_mask(self.roi if (self.roi is not None and
                                                           self.roi.is_area()) else None, w, h)
        rx, ry, rw, rh = rect
        min_x, max_x, min_y, max_y = rx, rx + rw, ry, ry + rh
        outside = np.ones((h, w), bool)
        outside[ry:ry + rh, rx:rx + rw] = False
        if sel_mask is not None:
            outside[ry:ry + rh, rx:rx + rw] |= ~sel_mask
        if outside.any():
            work[outside] = fill
            ins &= ~outside
        if opt.summarize:
            if sel_mask is not None:
                total_area = float(sel_mask.sum()) * cal.pixel_width * cal.pixel_height
            else:
                total_area = rw * cal.pixel_width * rh * cal.pixel_height
        structure = np.ones((3, 3), int) if not opt.four_connected else None
        labels, nlab = ndimage.label(ins, structure=structure)
        flood = not (opt.include_holes or opt.four_connected)
        ins_p = np.pad(ins, 1)
        analyzer = M.Analyzer(im, self.flags, self.rt, self.precision, redirect=self.redirect)
        src_plane = (self.redirect.plane_at(index) if self.redirect is not None and
                     self.redirect.stack_size == im.stack_size else
                     (self.redirect.plane() if self.redirect is not None else None))
        stats_type = (self.redirect or im).type
        if nlab:
            flat = labels.ravel()
            nz = np.flatnonzero(flat)
            labs, first = np.unique(flat[nz], return_index=True)
            order = np.argsort(nz[first])
            starts = nz[first][order]
            slices = ndimage.find_objects(labels)
        else:
            starts, labs, order, slices = [], [], [], []
        erased = np.zeros((h, w), bool) if not flood else None
        count = 0
        if self.output_planes is not None and self._draws_image():
            draw = self._new_draw_plane(w, h)
        else:
            draw = None
        rows_before = self.rt.size
        for k, start in enumerate(starts):
            y0, x0 = divmod(int(start), w)
            if erased is not None and erased[y0, x0]:
                continue
            lab = int(labs[order[k]])
            xs, ys = trace_outline(ins_p, x0, y0, opt.four_connected)
            if not xs:
                continue
            bx, by = min(xs), min(ys)
            bw, bh = max(xs) - bx, max(ys) - by
            roi = S.PolygonRoi(np.asarray(xs, float), np.asarray(ys, float),
                               type=S.TRACED_ROI)
            mask = None
            if bw > 1 and bh > 1:
                if flood:
                    sl = slices[lab - 1]
                    local = np.zeros((bh, bw), bool)
                    comp = labels[sl] == lab
                    oy, ox = sl[0].start - by, sl[1].start - bx
                    local[oy:oy + comp.shape[0], ox:ox + comp.shape[1]] = comp
                    mask = local
                else:
                    mask = polygon_mask(np.asarray(xs, float) - bx, np.asarray(ys, float) - by,
                                        (bh, bw))
            stats_plane = src_plane if src_plane is not None else work
            st = M.get_statistics(stats_plane, stats_type, (bx, by, bw, bh), mask, self.flags,
                                  analyzer.calibration(self.redirect or im))
            include = True
            if opt.exclude:
                if bx == min_x or by == min_y or bx + bw == max_x or by + bh == max_y:
                    include = False
                if include and sel_mask is not None:
                    region = outside[max(by - 1, 0):by + bh + 1, max(bx - 1, 0):bx + bw + 1]
                    if region.any():
                        grown = ndimage.binary_dilation(
                            labels[max(by - 1, 0):by + bh + 1, max(bx - 1, 0):bx + bw + 1] == lab,
                            structure=np.ones((3, 3), bool))
                        if (grown & region).any():
                            include = False
            if opt.composite and flood and mask is not None:
                loops = G.trace_mask_outlines(mask)
                if len(loops) > 1:
                    parts = [("or" if i == 0 else "xor",
                              S.PolygonRoi(lx + bx, ly + by, type=S.TRACED_ROI))
                             for i, (lx, ly) in enumerate(loops)]
                    roi = S.CompositeRoi(parts)
            if opt.min_circularity > 0.0 or opt.max_circularity != 1.0:
                perim = G.roi_length(roi)
                circ = 0.0 if perim == 0 else 4.0 * math.pi * (st.pixel_count / (perim * perim))
                if circ > 1.0 and opt.max_circularity <= 1.0:
                    circ = 1.0
                if circ < opt.min_circularity or circ > opt.max_circularity:
                    include = False
            if opt.min_size <= st.pixel_count <= opt.max_size and include:
                count += 1
                st.xstart, st.ystart = x0, y0
                analyzer.save_results(st, roi)
                if opt.max_circularity > 1.0 and self.rt.column_exists("Circ.") and \
                        self.rt.get_value("Circ.", self.rt.size - 1) == 1.0:
                    perim = G.roi_length(roi, cal.pixel_width, cal.pixel_height)
                    self.rt.add_value("Circ.", 0.0 if perim == 0 else
                                      4.0 * math.pi * (st.pixel_count / (perim * perim)))
                if opt.record_starts:
                    self.rt.add_value("XStart", x0)
                    self.rt.add_value("YStart", y0)
                if opt.add_to_manager:
                    r2 = roi.copy()
                    if im.stack_size > 1:
                        r2.position = self._position(index)
                    r2.name = self._manager_name(r2, im, count)
                    self.rois.append(r2)
                show = self.opt.show
                if self.opt.overlay and show == "Nothing":
                    show = "Overlay"
                self._draw(draw, show, roi, st, mask, bx, by, bw, bh, index)
            # erase the analysed pixels
            region = work[by:by + bh, bx:bx + bw]
            if mask is None:
                region[...] = fill
                if erased is not None:
                    erased[by:by + bh, bx:bx + bw] = True
            else:
                region[mask] = fill
                if erased is not None:
                    erased[by:by + bh, bx:bx + bw] |= mask
        if draw is not None:
            self.output_planes.append(draw)
        self.total_count += count
        if opt.summarize:
            self._summary_row(index, count, total_area, rows_before)
        return count

    # ------------------------------------------------------------------ helpers
    def _position(self, index):
        im = self.image
        if im.hyperstack or im.is_composite:
            c, z, t = im.position_of(index)
            return (c + 1, z + 1, t + 1)
        return (0, index + 1, 0)

    @staticmethod
    def _manager_name(roi, im, count):
        x, y, w, h = G.int_bounds(roi)
        yc, xc = y + h // 2, x + w // 2
        if im.stack_size > 1:
            return f"{im.current_index + 1:04d}-{yc:04d}-{xc:04d}"
        return f"{yc:04d}-{xc:04d}"

    def _draws_image(self) -> bool:
        return self.opt.show in ("Outlines", "Bare Outlines", "Ellipses", "Masks",
                                 "Count Masks") or (self.opt.in_situ and
                                                     self.opt.show == "Nothing")

    def _bb(self) -> bool:
        return self.black_background and self.opt.in_situ

    def _new_draw_plane(self, w, h):
        show = self.opt.show
        if show == "Count Masks":
            return np.zeros((h, w), np.uint16)
        if self._bb():
            return np.zeros((h, w), np.uint8)
        if show == "Masks":
            return np.zeros((h, w), np.uint8)   # inverted LUT: 0 shows white
        return np.full((h, w), 255, np.uint8)

    def _draw(self, draw, show, roi, st, mask, bx, by, bw, bh, index):
        count = self.rt.size
        if show in ("Overlay", "Overlay Masks"):
            if self.overlay is None:
                self.overlay = S.Overlay(show_labels=True)
                self.overlay.label_font_size = self.font_size
            r2 = roi.copy()
            r2.stroke_color = "#00ffff"
            if show == "Overlay Masks":
                r2.fill_color = _MASK_COLORS[(count - 1) % len(_MASK_COLORS)]
            if self.opt.stack or self.image.stack_size > 1:
                r2.position = self._position(index)
            if self.opt.display:
                r2.name = str(count)
            self.overlay.add(r2)
            return
        if draw is None:
            return
        fg = 255 if self._bb() else (255 if show == "Masks" else 0)
        if show in ("Masks",):
            region = draw[by:by + bh, bx:bx + bw]
            if mask is None:
                region[...] = fg
            else:
                region[mask] = fg
        elif show == "Count Masks":
            region = draw[by:by + bh, bx:bx + bw]
            v = min(count, 65535)
            if mask is None:
                region[...] = v
            else:
                region[mask] = v
        elif show in ("Outlines", "Bare Outlines") or (self.opt.in_situ and show == "Nothing"):
            parts = roi.parts if isinstance(roi, S.CompositeRoi) else [("or", roi)]
            for _, p in parts:
                draw_polygon(draw, p.xs, p.ys, fg)
            if show != "Bare Outlines":
                s = str(count)
                draw_label(draw, s, bx + bw // 2 - text_width(s) // 2,
                           by + bh // 2 + self.font_size // 2, 1 if not self.opt.in_situ else fg,
                           self.font_size)
        elif show == "Ellipses" and st.ellipse is not None:
            xc, yc, major, minor, theta = st.ellipse
            dx = major * math.cos(theta) / 2.0
            dy = -major * math.sin(theta) / 2.0
            ell = S.EllipseRoi(xc - dx, yc - dy, xc + dx, yc + dy,
                               minor / major if major else 1.0)
            ex, ey = ell.polygon()
            draw_polygon(draw, np.floor(ex), np.floor(ey), fg)

    def _make_output(self, indices) -> ImageDoc | None:
        if not self.output_planes:
            return None
        im, show = self.image, self.opt.show
        arr = np.stack(self.output_planes)
        if self.opt.in_situ:
            n = im.stack_size
            if arr.shape[0] == n or n == 1:
                px = arr.reshape((im.n_frames, im.n_slices, im.n_channels, im.height, im.width)) \
                    if arr.shape[0] == n else arr.reshape((1, 1, 1, im.height, im.width))
                if px.dtype != im.pixels.dtype and arr.shape[0] == n:
                    im.replace_pixels(px)
                elif arr.shape[0] == n:
                    im.pixels[...] = px
                else:
                    im.set_plane(arr[0])
            if show == "Masks" and not self._bb():
                im.luts = [LUT.grays().inverted() for _ in im.luts]
            return None
        prefix = {"Masks": "Mask of ", "Count Masks": "Count Masks of "}.get(show, "Drawing of ")
        doc = ImageDoc.from_array(arr if len(arr) > 1 else arr[0],
                                  "ZYX" if len(arr) > 1 else "YX", prefix + im.title)
        doc.calibration = im.calibration.copy()
        doc.calibration.disable_density()
        if show == "Masks" and not self._bb():
            doc.luts = [LUT.grays().inverted()]
        elif show == "Outlines":
            doc.luts = [outlines_lut()]
        if show == "Count Masks":
            doc.reset_display_range()
        return doc

    def _summary_row(self, index, count, total_area, rows_before):
        im = self.image
        if self.summary is None:
            self.summary = ResultsTable("Summary" if im.stack_size == 1 else
                                        f"Summary of {im.title}")
        st = self.summary
        rt = self.rt
        if im.stack_size > 1:
            label = im.slice_labels[index]
            label = label.splitlines()[0].strip() if label else ""
            label = label if label else str(index + 1)
        else:
            label = im.title
        st.increment_counter()
        st.set_value("Slice", st.size - 1, label)
        areas = rt.column("Area").astype(np.float32).astype(np.float64) \
            if rt.column_exists("Area") else np.zeros(0)
        start = len(areas) - count
        if start < 0:
            return
        total = float(np.sum(areas[start:]))
        st.add_value("Count", count)
        st.add_value("Total Area", total)
        st.add_value("Average Size", total / count if count else float("nan"))
        st.add_value("%Area", total * 100.0 / total_area if total_area else float("nan"))
        m = self.flags
        cols = []
        if m & M.MEAN:
            cols.append("Mean")
        if m & M.MODE:
            cols.append("Mode")
        if m & M.PERIMETER:
            cols.append("Perim.")
        if m & M.ELLIPSE:
            cols += ["Major", "Minor", "Angle"]
        if m & M.SHAPE_DESCRIPTORS:
            cols += ["Circ.", "Solidity"]
        if m & M.FERET:
            cols += ["Feret", "FeretX", "FeretY", "FeretAngle", "MinFeret"]
        if m & M.INTEGRATED_DENSITY:
            cols.append("IntDen")
        if m & M.MEDIAN:
            cols.append("Median")
        if m & M.SKEWNESS:
            cols.append("Skew")
        if m & M.KURTOSIS:
            cols.append("Kurt")
        for col in cols:
            value = float("nan")
            if len(areas) and rt.column_exists(col):
                c = rt.column(col).astype(np.float32)[start:]
                c64 = c.astype(np.float64)
                with np.errstate(invalid="ignore"):
                    finite = c64[(c64 >= -M.FLOAT_MAX) & (c64 <= M.FLOAT_MAX)]
                value = float(finite.sum() / len(finite)) if len(finite) else float("nan")
            st.add_value(col, value)


def analyze_particles(image, flags: int, options: ParticleOptions, **kw) -> ParticleResult:
    """Convenience wrapper: ``ParticleAnalyzer(image, flags, options, **kw).run()``."""
    return ParticleAnalyzer(image, flags, options, **kw).run()

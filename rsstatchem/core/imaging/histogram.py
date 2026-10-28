"""Analyze > Histogram: the histogram data and statistics that ImageJ's HistogramWindow shows
(N, Mean, StdDev, Min, Max, Mode, Bins, Bin Width), with bins and x range for 16/32-bit
images, the stack histogram, RGB (gray value or one channel) and the List table.

Ported in part from ImageJ 1.54 ij/plugin/Histogram.java, ij/gui/HistogramWindow.java and
ij/process/StackStatistics.java (public domain, NIH).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rsstatchem.core.imaging import measure as M
from rsstatchem.core.imaging.image import COLOR_RGB, GRAY8, GRAY16, GRAY32
from rsstatchem.core.imaging.results import ResultsTable, d2s

RGB_MODES = ("Gray", "Red", "Green", "Blue")


@dataclass
class HistogramData:
    counts: np.ndarray
    bin_starts: np.ndarray       # calibrated start value of each bin
    bin_width: float
    n: int
    mean: float
    std_dev: float
    min: float
    max: float
    mode: float
    mode_count: int
    n_bins: int
    hist_min: float
    hist_max: float
    title: str = ""
    value_unit: str = "Gray Value"
    stack: bool = False
    integer_bins: bool = True

    def stats_text(self) -> str:
        """The statistics block under the histogram (ImageJ wording)."""
        def fmt(v):
            return d2s(v, 0) if float(v).is_integer() else d2s(v, 3)
        lines = [f"N: {self.n}", f"Mean: {fmt(self.mean)}", f"StdDev: {fmt(self.std_dev)}",
                 f"Min: {fmt(self.min)}", f"Max: {fmt(self.max)}",
                 f"Mode: {fmt(self.mode)} ({self.mode_count})"]
        if not self.integer_bins or self.bin_width != 1.0:
            lines.append(f"Bins: {self.n_bins}")
            lines.append(f"Bin Width: {d2s(self.bin_width, 3)}")
        return "\n".join(lines)

    def table(self, title: str = "Histogram") -> ResultsTable:
        """The "List" table: bin start ("value" / "bin start") and count."""
        rt = ResultsTable(title)
        name = "value" if self.integer_bins and self.bin_width == 1.0 else "bin start"
        for v, c in zip(self.bin_starts, self.counts):
            rt.add_row({name: float(v), "count": int(c)})
        if name == "bin start":
            rt.decimal_places[name] = 4
        return rt


def _plane_for(image, index, rgb_mode, weighted):
    p = image.plane_at(index)
    if p.ndim == 3:
        if rgb_mode in ("Red", "Green", "Blue"):
            return p[..., RGB_MODES.index(rgb_mode) - 1], GRAY8
        return M.gray_of_rgb(p, weighted), GRAY8
    return p, image.type


def histogram(image, n_bins: int = 256, hist_min: float = 0.0, hist_max: float = 0.0,
              stack: bool = False, rgb_mode: str = "Gray", weighted_rgb: bool = False,
              roi=None, use_roi: bool = True) -> HistogramData:
    """Histogram of the selection (or whole plane) or of the whole stack."""
    roi = image.roi if (roi is None and use_roi) else roi
    rect, mask = M.roi_rect_and_mask(roi if roi is not None and roi.is_area() else None,
                                     image.width, image.height)
    cal = image.calibration
    itype = image.type
    indices = range(image.stack_size) if stack and image.stack_size > 1 else \
        [image.current_index]
    vals = []
    for i in indices:
        p, t = _plane_for(image, i, rgb_mode, weighted_rgb)
        x, y, w, h = rect
        region = p[y:y + h, x:x + w]
        vals.append(region[mask] if mask is not None else region.ravel())
    v = np.concatenate(vals) if vals else np.zeros(0)
    eight = itype in (GRAY8, COLOR_RGB) or image.type == 3
    table = None
    if cal.function != 20 and image.bit_depth in (8, 16):
        table = np.asarray(cal.ctable(image.bit_depth), dtype=np.float32).astype(float)
    if eight and n_bins == 256 and hist_min == 0 and hist_max == 0:
        counts = np.bincount(v.astype(np.int64), minlength=256)[:256]
        starts = np.arange(256, dtype=float)
        if table is not None:
            starts = table
        width = 1.0
        hmin, hmax = 0.0, 255.0
        integer = True
    else:
        fv = v.astype(np.float64)
        fv = fv[np.isfinite(fv)]
        if itype == GRAY16 and table is None and hist_min == 0 and hist_max == 0 and \
                n_bins == 256 and len(fv):
            hmin, hmax = float(fv.min()), float(fv.max())
            width = (hmax - hmin + 1) / n_bins
            idx = np.minimum(((fv - hmin) / width).astype(np.int64), n_bins - 1)
            integer = True
        else:
            if hist_min == 0 and hist_max == 0:
                hmin = float(fv.min()) if len(fv) else 0.0
                hmax = float(fv.max()) if len(fv) else 0.0
            else:
                hmin, hmax = float(hist_min), float(hist_max)
                fv = fv[(fv >= hmin) & (fv <= hmax)]
            if itype == GRAY16 or eight:
                width = (hmax - hmin + 1) / n_bins
                integer = True
            else:
                width = (hmax - hmin) / n_bins if hmax > hmin else 1.0
                integer = False
            scale = 1.0 / width if width else 0.0
            idx = np.clip((scale * (fv - hmin)).astype(np.int64), 0, n_bins - 1)
        counts = np.bincount(idx, minlength=n_bins)[:n_bins] if len(fv) else \
            np.zeros(n_bins, np.int64)
        starts = hmin + np.arange(n_bins) * width
        v = fv
    cv = table[v.astype(np.int64)] if table is not None and len(v) else v.astype(float)
    n = int(len(cv))
    mean = float(cv.mean()) if n else float("nan")
    sd = float(cv.std(ddof=1)) if n > 1 else 0.0
    k = int(np.argmax(counts)) if len(counts) else 0
    mode = float(starts[k]) + (width / 2 if not integer else 0.0)
    if integer and width != 1.0:
        mode = float(starts[k])
    return HistogramData(counts=counts, bin_starts=starts, bin_width=width, n=n, mean=mean,
                         std_dev=sd, min=float(cv.min()) if n else float("nan"),
                         max=float(cv.max()) if n else float("nan"), mode=mode,
                         mode_count=int(counts[k]) if len(counts) else 0, n_bins=len(counts),
                         hist_min=hmin, hist_max=hmax, title=image.title,
                         value_unit=cal.value_unit, stack=stack, integer_bins=integer)


def default_range(image, stack: bool = False):
    """Default x range of the Histogram dialog for 16/32-bit images (min..max)."""
    if image.type not in (GRAY16, GRAY32):
        return 0.0, 256.0
    if stack:
        v = image.pixels.astype(np.float64)
        v = v[np.isfinite(v)]
        return (float(v.min()), float(v.max())) if v.size else (0.0, 0.0)
    return image.plane_min_max()

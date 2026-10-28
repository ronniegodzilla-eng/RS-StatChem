"""Image statistics and Analyze > Measure with ImageJ semantics.

* :func:`get_statistics` = ``ImageStatistics.getStatistics`` (ByteStatistics, ShortStatistics,
  FloatStatistics, ColorStatistics): area, mean, SD (n-1), mode, min/max, centroid, centre of
  mass, ellipse fit (EllipseFitter), median, skewness, kurtosis, area fraction, histogram;
  "limit to threshold"; value calibration through a float32 calibration table like ImageJ.
* :class:`Analyzer` = ``ij.plugin.filter.Analyzer``: Measure of area/line/point/angle
  selections into a Results table with every Set Measurements option (labels, invert Y,
  redirect, perimeter, bounding rectangle, Feret, shape descriptors, integrated density,
  stack position, MinThr/MaxThr, add to overlay), and Summarize (Mean/SD/Min/Max rows).

Ported from ImageJ 1.54 ij/process/ImageStatistics.java, ByteStatistics.java,
ShortStatistics.java, FloatStatistics.java, ColorStatistics.java, EllipseFitter.java and
ij/plugin/filter/Analyzer.java (public domain, NIH; W. Rasband, M. Schmid et al.).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rsstatchem.core.imaging import calibrate as _calibrate  # noqa: F401 - value functions
from rsstatchem.core.imaging.calibration import NONE, Calibration
from rsstatchem.core.imaging.image import COLOR_RGB, GRAY8, GRAY16, GRAY32
from rsstatchem.core.imaging.results import ResultsTable
from rsstatchem.core.imaging.roi import shapes as S
from rsstatchem.core.imaging.roi import geometry as G

# ij.measure.Measurements
AREA, MEAN, STD_DEV, MODE, MIN_MAX = 1, 2, 4, 8, 16
CENTROID, CENTER_OF_MASS, PERIMETER, LIMIT, RECT = 32, 64, 128, 256, 512
LABELS, ELLIPSE, INVERT_Y, SHAPE_DESCRIPTORS, FERET = 1024, 2048, 4096, 8192, 16384
INTEGRATED_DENSITY, MEDIAN, SKEWNESS, KURTOSIS = 0x8000, 0x10000, 0x20000, 0x40000
AREA_FRACTION, STACK_POSITION, SCIENTIFIC_NOTATION = 0x80000, 0x100000, 0x200000
ADD_TO_OVERLAY, NAN_EMPTY_CELLS = 0x400000, 0x800000
ALL_STATS = AREA | MEAN | STD_DEV | MODE | MIN_MAX | CENTROID | CENTER_OF_MASS | PERIMETER | \
    RECT | ELLIPSE | SHAPE_DESCRIPTORS | FERET | INTEGRATED_DENSITY | MEDIAN | SKEWNESS | \
    KURTOSIS | AREA_FRACTION | STACK_POSITION
NO_THRESHOLD = -808080.0
FLOAT_MAX = 3.4028234663852886e38

# Set Measurements macro keys (Options.measurements) -> flag
KEY_FLAGS = {
    "area": AREA, "mean": MEAN, "standard": STD_DEV, "modal": MODE, "min": MIN_MAX,
    "centroid": CENTROID, "center": CENTER_OF_MASS, "perimeter": PERIMETER,
    "bounding": RECT, "fit": ELLIPSE, "shape": SHAPE_DESCRIPTORS, "feret's": FERET,
    "integrated": INTEGRATED_DENSITY, "median": MEDIAN, "skewness": SKEWNESS,
    "kurtosis": KURTOSIS, "area_fraction": AREA_FRACTION, "stack": STACK_POSITION,
    "limit": LIMIT, "display": LABELS, "invert": INVERT_Y, "scientific": SCIENTIFIC_NOTATION,
    "add": ADD_TO_OVERLAY, "nan": NAN_EMPTY_CELLS,
}


def flags_from_keys(keys) -> int:
    out = 0
    for k in keys:
        out |= KEY_FLAGS.get(k, 0)
    return out


def keys_from_flags(flags: int) -> tuple:
    return tuple(k for k, f in KEY_FLAGS.items() if flags & f)


@dataclass
class ImageStatistics:
    histogram: np.ndarray = field(default_factory=lambda: np.zeros(256, dtype=np.int64))
    histogram16: np.ndarray | None = None
    pixel_count: int = 0
    mode: int = 0
    dmode: float = 0.0
    area: float = 0.0
    min: float = 0.0
    max: float = 0.0
    mean: float = 0.0
    median: float = 0.0
    std_dev: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    x_centroid: float = 0.0
    y_centroid: float = 0.0
    x_center_of_mass: float = 0.0
    y_center_of_mass: float = 0.0
    roi_x: float = 0.0
    roi_y: float = 0.0
    roi_width: float = 0.0
    roi_height: float = 0.0
    umean: float = 0.0
    major: float = 0.0
    minor: float = 0.0
    angle: float = 0.0
    area_fraction: float = 0.0
    lower_threshold: float = float("nan")
    upper_threshold: float = float("nan")
    hist_min: float = 0.0
    hist_max: float = 0.0
    n_bins: int = 256
    bin_size: float = 1.0
    max_count: int = 0
    xstart: int = 0
    ystart: int = 0
    ellipse: tuple | None = None     # (xCenter, yCenter, major, minor, theta) uncalibrated


# ---------------------------------------------------------------------------------- helpers
def gray_of_rgb(rgb: np.ndarray, weighted: bool = False, weights=None) -> np.ndarray:
    """ColorProcessor gray value: (int)(r*rw + g*gw + b*bw + 0.5)."""
    if weights is None:
        weights = (0.299, 0.587, 0.114) if weighted else (1 / 3, 1 / 3, 1 / 3)
    rw, gw, bw = weights
    a = rgb.astype(np.float64)
    return (a[..., 0] * rw + a[..., 1] * gw + a[..., 2] * bw + 0.5).astype(np.int64) \
        .clip(0, 255).astype(np.uint8)


def roi_rect_and_mask(roi, width: int, height: int):
    """``ImageProcessor.setRoi(Roi)``: the selection's integer bounds clipped to the image
    and the matching local mask (None for plain rectangles and the whole image)."""
    if roi is None or not roi.is_area():
        return (0, 0, width, height), None
    bx, by, bw, bh = G.int_bounds(roi)
    x0, y0 = max(bx, 0), max(by, 0)
    x1, y1 = min(bx + bw, width), min(by + bh, height)
    if x1 <= x0 or y1 <= y0:
        return (0, 0, 0, 0), np.zeros((0, 0), bool)
    rect = (x0, y0, x1 - x0, y1 - y0)
    if isinstance(roi, S.RectRoi) and not roi.corner_diameter:
        return rect, None
    full = roi.get_mask((y1, x1))
    return rect, full[y0:y1, x0:x1]


def _ctable(cal: Calibration | None, bit_depth: int):
    if cal is None or cal.function == NONE:
        return None
    t = cal.ctable(bit_depth)
    return None if t is None else np.asarray(t, dtype=np.float32).astype(np.float64)


def _cvalue(cal, raw, table):
    if cal is None or cal.function == NONE:
        return float(raw)
    try:
        return float(cal.cvalue(raw))
    except NotImplementedError:
        return float(table[int(raw)]) if table is not None else float(raw)


def _std(n, s, s2):
    if n > 0:
        sd = (n * s2 - s * s) / n
        return math.sqrt(sd / (n - 1.0)) if sd > 0.0 else 0.0
    return 0.0


def _get_x(cal, x):
    return cal.get_x(x) if cal is not None else x


def _get_y(cal, y, height):
    return cal.get_y(y, height) if cal is not None else y


# ---------------------------------------------------------------------------------- statistics
def get_statistics(plane: np.ndarray, image_type: int = GRAY8, rect=None, mask=None,
                   mflags: int = AREA | MEAN | MODE | MIN_MAX, cal: Calibration | None = None,
                   threshold: tuple | None = None, n_bins: int = 256,
                   hist_range: tuple = (0.0, 0.0), weighted_rgb: bool = False,
                   rgb_weights=None) -> ImageStatistics:
    """Statistics of ``plane`` (2-D, or (H, W, 3) RGB) inside ``rect`` (x, y, w, h) and the
    local boolean ``mask`` (h, w). ``threshold`` = (lower, upper) raw values of the image's
    threshold (used with LIMIT and for Area fraction)."""
    h_img, w_img = plane.shape[:2]
    if rect is None:
        rect = (0, 0, w_img, h_img)
    rx, ry, rw, rh = (int(v) for v in rect)
    st = ImageStatistics()
    pw = cal.pixel_width if cal is not None else 1.0
    ph = cal.pixel_height if cal is not None else 1.0
    st.roi_x = _get_x(cal, rx)
    st.roi_y = _get_y(cal, ry, h_img)
    st.roi_width, st.roi_height = rw * pw, rh * ph
    if plane.ndim == 3 or image_type == COLOR_RGB:
        gray = gray_of_rgb(plane, weighted_rgb, rgb_weights) if plane.ndim == 3 else plane
        return _color_stats(st, gray, rx, ry, rw, rh, mask, mflags, cal, pw, ph, h_img)
    region = plane[ry:ry + rh, rx:rx + rw]
    sel = np.ones(region.shape, dtype=bool) if mask is None else np.asarray(mask, bool)
    ctx = dict(st=st, region=region, sel=sel, rx=rx, ry=ry, rw=rw, rh=rh, mask=mask,
               mflags=mflags, cal=cal, pw=pw, ph=ph, height=h_img, threshold=threshold,
               n_bins=n_bins, hist_range=hist_range)
    if image_type in (GRAY8,) or plane.dtype == np.uint8:
        return _byte_stats(**ctx)
    if image_type == GRAY16 or plane.dtype == np.uint16:
        return _short_stats(**ctx)
    return _float_stats(**ctx)


def _limits(mflags, threshold, lo, hi, integer=True):
    limit = bool(mflags & LIMIT)
    if not limit or threshold is None:
        return lo, hi, limit
    t1, t2 = threshold
    return (int(t1), int(t2), limit) if integer else (float(t1), float(t2), limit)


def _byte_stats(st, region, sel, rx, ry, rw, rh, mask, mflags, cal, pw, ph, height, threshold,
                n_bins, hist_range):
    vals = region[sel].astype(np.int64)
    hist = np.bincount(vals, minlength=256)[:256]
    st.histogram = hist
    min_t, max_t, limit = _limits(mflags, threshold, 0, 255)
    if limit:
        _save_threshold(st, min_t, max_t, cal)
    table = _ctable(cal, 8)
    rng = np.arange(min_t, max_t + 1)
    counts = hist[min_t:max_t + 1]
    if table is not None:
        cv = table[min_t:max_t + 1]
        ok = (counts > 0) & ~np.isnan(cv)
        n = int(counts[ok].sum())
        s = float((cv[ok] * counts[ok]).sum())
        s2 = float((cv[ok] * cv[ok] * counts[ok]).sum())
        isum = float((rng[ok] * counts[ok]).sum())
        st.pixel_count = n
        st.area = n * pw * ph
        st.mean = s / n if n else float("nan")
        st.umean = isum / n if n else float("nan")
        if ok.any():
            masked = np.where(ok, counts, -1)
            st.mode = int(min_t + np.argmax(masked))
            st.max_count = int(masked.max())
        st.dmode = float(table[st.mode])
        st.std_dev = _std(n, s, s2)
    else:
        n = int(counts.sum())
        s = float((rng * counts).sum())
        s2 = float((rng * rng * counts).sum())
        st.pixel_count = n
        st.area = n * pw * ph
        st.mean = s / n if n else float("nan")
        st.umean = st.mean
        if len(counts) and counts.max() > 0:
            st.mode = int(min_t + np.argmax(counts))
            st.max_count = int(counts.max())
        st.dmode = float(st.mode)
        st.std_dev = _std(n, s, s2)
    st.hist_min, st.hist_max = 0.0, 255.0
    if mflags & MIN_MAX:
        if st.pixel_count == 0:
            st.min = st.max = float("nan")
        elif table is not None:
            nz = rng[counts > 0]
            cv = table[nz]
            st.min, st.max = float(cv.min()), float(cv.max())
        else:
            st.min, st.max = _raw_min_max(hist, min_t, max_t, 255)
    lim = (min_t > 0 or max_t < 255)
    vsel = sel & (region >= min_t) & (region <= max_t) if lim else sel
    _shape_stats(st, region, sel, vsel, rx, ry, rw, rh, mask, mflags, cal, height,
                 table, threshold, (min_t, max_t), integer=True)
    if mflags & MEDIAN:
        st.median = _hist_median(hist, min_t, max_t, st.pixel_count, cal, table)
    if mflags & AREA_FRACTION:
        st.area_fraction = _hist_area_fraction(hist, threshold)
    return st


def _short_stats(st, region, sel, rx, ry, rw, rh, mask, mflags, cal, pw, ph, height, threshold,
                 n_bins, hist_range):
    vals = region[sel].astype(np.int64)
    hist = np.bincount(vals, minlength=65536)[:65536]
    st.histogram16 = hist
    min_t, max_t, limit = _limits(mflags, threshold, 0, 65535)
    if limit:
        _save_threshold(st, min_t, max_t, cal)
    max_t = min(max_t, 65535)
    table = _ctable(cal, 16)
    vmin, vmax = _raw_min_max(hist, min_t, max_t, 65535)
    st.min, st.max = vmin, vmax
    imin, imax = int(vmin), int(vmax)
    hmin, hmax = hist_range
    if hmin == 0.0 and hmax == 0.0:
        hmin, hmax = float(imin), float(imax)
    else:
        imin = max(imin, int(hmin))
        imax = min(imax, int(hmax))
    st.n_bins = n_bins
    st.hist_min, st.hist_max = hmin, hmax
    st.bin_size = (hmax - hmin + 1) / n_bins
    scale = 1.0 / st.bin_size
    rng = np.arange(imin, imax + 1)
    counts = hist[imin:imax + 1] if imax >= imin else np.zeros(0, np.int64)
    if len(counts) and counts.max() > 0:
        st.dmode = float(imin + int(np.argmax(counts)))
    n = int(counts.sum())
    v = table[imin:imax + 1] if table is not None else rng.astype(float)
    s = float((v * counts).sum())
    s2 = float((v * v * counts).sum())
    idx = np.minimum((scale * (rng - int(hmin))).astype(np.int64), n_bins - 1)
    h256 = np.zeros(n_bins, dtype=np.int64)
    if len(rng):
        np.add.at(h256, np.clip(idx, 0, n_bins - 1), counts)
    st.histogram = h256
    st.pixel_count = n
    st.area = n * pw * ph
    st.mean = s / n if n else float("nan")
    st.umean = st.mean
    st.std_dev = _std(n, s, s2)
    if table is not None:
        st.dmode = float(table[int(st.dmode)])
    if mflags & MODE:
        st.max_count = int(h256.max()) if len(h256) else 0
        st.mode = int(np.argmax(h256)) if len(h256) else 0
    lim = (min_t > 0 or max_t < 65535)
    vsel = sel & (region >= min_t) & (region <= max_t) if lim else sel
    _shape_stats(st, region, sel, vsel, rx, ry, rw, rh, mask, mflags, cal, height,
                 table, threshold, (min_t, max_t), integer=True)
    if mflags & MIN_MAX and table is not None and len(counts):
        nz = rng[counts > 0]
        if len(nz):
            cv = table[nz]
            st.min, st.max = float(cv.min()), float(cv.max())
    if mflags & MEDIAN:
        st.median = _hist_median(hist, min_t, max_t, st.pixel_count, cal, table) \
            if st.pixel_count > 0 else float("nan")
    if mflags & AREA_FRACTION:
        st.area_fraction = _hist_area_fraction(hist, threshold)
    return st


def _float_stats(st, region, sel, rx, ry, rw, rh, mask, mflags, cal, pw, ph, height, threshold,
                 n_bins, hist_range):
    region = np.asarray(region, dtype=np.float32)
    min_t, max_t, limit = _limits(mflags, threshold, -FLOAT_MAX, FLOAT_MAX, integer=False)
    if limit:
        _save_threshold(st, min_t, max_t, cal)
    v = region[sel].astype(np.float64)
    with np.errstate(invalid="ignore"):
        inside = (v >= min_t) & (v <= max_t)
    vin = v[inside]
    st.min = float(vin.min()) if vin.size else FLOAT_MAX
    st.max = float(vin.max()) if vin.size else -FLOAT_MAX
    hmin, hmax = hist_range
    if hmin == 0.0 and hmax == 0.0:
        hmin, hmax = st.min, st.max
    else:
        st.min = max(st.min, hmin)
        st.max = min(st.max, hmax)
    st.n_bins = n_bins
    st.hist_min, st.hist_max = hmin, hmax
    st.bin_size = (hmax - hmin) / n_bins
    sel_v = vin[(vin >= hmin) & (vin <= hmax)]
    n = int(sel_v.size)
    st.pixel_count = n
    s = float(sel_v.sum())
    s2 = float((sel_v * sel_v).sum())
    hist = np.zeros(n_bins, dtype=np.int64)
    if n:
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = n_bins / (hmax - hmin) if hmax != hmin else 0.0
        idx = (scale * (sel_v - hmin)).astype(np.int64) if scale else np.zeros(n, np.int64)
        np.add.at(hist, np.clip(idx, 0, n_bins - 1), 1)
    st.histogram = hist
    st.area = n * pw * ph
    st.mean = s / n if n else float("nan")
    st.umean = st.mean
    st.std_dev = _std(n, s, s2)
    if mflags & MODE:
        st.max_count = int(hist.max()) if n else 0
        st.mode = int(np.argmax(hist)) if n else 0
        st.dmode = st.hist_min + st.mode * st.bin_size
        if st.bin_size != 1.0:
            st.dmode += st.bin_size / 2.0
    with np.errstate(invalid="ignore"):
        vsel = sel & (region >= min_t) & (region <= max_t)
    _shape_stats(st, region, sel, vsel, rx, ry, rw, rh, mask, mflags, cal, height, None,
                 threshold, (min_t, max_t), integer=False)
    if mflags & MEDIAN:
        if n == 0:
            st.median = float("nan")
        else:
            vals = np.sort(region[vsel].astype(np.float32))
            m = len(vals) // 2
            if len(vals) % 2 == 0:
                st.median = float((vals[m - 1] + vals[m]) / np.float32(2))
            else:
                st.median = float(vals[m])
    if mflags & AREA_FRACTION:
        vv = region[sel]
        total = vv.size
        if threshold is None:
            cnt = int(np.count_nonzero(vv != 0))
        else:
            t1, t2 = np.float32(threshold[0]), np.float32(threshold[1])
            cnt = int(np.count_nonzero((vv >= t1) & (vv <= t2)))
        st.area_fraction = cnt * 100.0 / total if total else float("nan")
    return st


def _color_stats(st, gray, rx, ry, rw, rh, mask, mflags, cal, pw, ph, height):
    region = gray[ry:ry + rh, rx:rx + rw]
    sel = np.ones(region.shape, dtype=bool) if mask is None else np.asarray(mask, bool)
    hist = np.bincount(region[sel].astype(np.int64), minlength=256)[:256]
    st.histogram = hist
    rng = np.arange(256)
    n = int(hist.sum())
    s = float((rng * hist).sum())
    s2 = float((rng * rng * hist).sum())
    st.pixel_count = n
    st.area = n * pw * ph
    st.mean = s / n if n else float("nan")
    st.umean = st.mean
    if n:
        st.mode = int(np.argmax(hist))
        st.max_count = int(hist.max())
    st.dmode = float(st.mode)
    st.std_dev = _std(n, s, s2)
    st.hist_min, st.hist_max = 0.0, 255.0
    if mflags & MIN_MAX:
        st.min, st.max = _raw_min_max(hist, 0, 255, 255)
    _shape_stats(st, region, sel, sel, rx, ry, rw, rh, mask, mflags & ~LIMIT, cal, height,
                 None, None, (0, 255), integer=True)
    if mflags & MEDIAN:
        st.median = _hist_median(hist, 0, 255, n, None, None)
    return st


def _raw_min_max(hist, min_t, max_t, top):
    mn = min_t
    while mn < top and hist[mn] == 0:
        mn += 1
    mx = max_t
    while mx > 0 and hist[mx] == 0:
        mx -= 1
    return float(mn), float(mx)


def _save_threshold(st, lo, hi, cal):
    if cal is not None and cal.function != NONE:
        try:
            lo, hi = cal.cvalue(lo), cal.cvalue(hi)
        except NotImplementedError:
            pass
    st.lower_threshold, st.upper_threshold = float(lo), float(hi)


def _hist_median(hist, first, last, pixel_count, cal, table):
    if pixel_count == 0 or first < 0 or last > len(hist):
        return float("nan")
    seg = np.cumsum(hist[first:last + 1])
    half = pixel_count / 2.0
    above = np.nonzero(seg > half)[0]
    i = first + int(above[0]) if len(above) else last
    i = min(i, last)
    return _cvalue(cal, i, table)


def _hist_area_fraction(hist, threshold):
    total = int(hist.sum())
    if threshold is None:
        s = total - int(hist[0])
    else:
        t1 = int(math.floor(threshold[0] + 0.5))
        t2 = int(math.floor(threshold[1] + 0.5))
        lo, hi = max(t1, 0), min(t2, len(hist) - 1)
        s = int(hist[lo:hi + 1].sum()) if hi >= lo else 0
    return s * 100.0 / total if total else float("nan")


def _shape_stats(st, region, sel, vsel, rx, ry, rw, rh, mask, mflags, cal, height, table,
                 threshold, limits, integer):
    """Centroid / ellipse / moments, shared by the gray-level types."""
    if mflags & (ELLIPSE | SHAPE_DESCRIPTORS):
        ef_mask = mask
        if (mflags & LIMIT) and threshold is not None:
            t1, t2 = threshold
            base = np.ones((rh, rw), bool) if mask is None else np.asarray(mask, bool).copy()
            with np.errstate(invalid="ignore"):
                base &= (region >= t1) & (region <= t2)
            ef_mask = base
        ef = fit_ellipse(ef_mask, rx, ry, rw, rh)
        pw = cal.pixel_width if cal is not None else 1.0
        ph = cal.pixel_height if cal is not None else 1.0
        psize = pw if abs(pw - ph) / pw < 0.01 else 0.0
        st.major, st.minor, st.angle = ef[2] * psize, ef[3] * psize, ef[5]
        st.ellipse = (ef[0], ef[1], ef[2], ef[3], ef[4])
        st.x_centroid = _get_x(cal, ef[0])
        st.y_centroid = _get_y(cal, ef[1], height)
    elif mflags & CENTROID:
        yy, xx = np.nonzero(vsel)
        cnt = len(xx)
        if cnt:
            st.x_centroid = _get_x(cal, (xx.sum() + rx * cnt) / cnt + 0.5)
            st.y_centroid = _get_y(cal, (yy.sum() + ry * cnt) / cnt + 0.5, height)
        else:
            st.x_centroid = st.y_centroid = float("nan")
    if mflags & (CENTER_OF_MASS | SKEWNESS | KURTOSIS):
        yy, xx = np.nonzero(vsel)
        raw = region[yy, xx]
        if table is not None:
            v = table[raw.astype(np.int64)]
        else:
            v = raw.astype(np.float64)
        n = st.pixel_count
        s1 = float(v.sum())
        s2 = float((v * v).sum())
        s3 = float((v * v * v).sum())
        s4 = float((v * v * v * v).sum())
        x = (xx + rx).astype(np.float64)
        y = (yy + ry).astype(np.float64)
        mean = st.mean
        mean2 = mean * mean
        with np.errstate(divide="ignore", invalid="ignore"):
            variance = s2 / n - mean2 if n else float("nan")
            sdev = math.sqrt(variance) if variance >= 0 else float("nan")
            denom = variance * sdev
            st.skewness = ((s3 - 3.0 * mean * s2) / n + 2.0 * mean * mean2) / denom \
                if n and denom != 0 else float("nan")
            st.kurtosis = (((s4 - 4.0 * mean * s3 + 6.0 * mean2 * s2) / n - 3.0 * mean2 * mean2)
                           / (variance * variance) - 3.0) if n and variance != 0 \
                else float("nan")
        if s1 != 0:
            xc = float((x * v).sum()) / s1 + 0.5
            yc = float((y * v).sum()) / s1 + 0.5
        elif len(x):
            xc, yc = float(x.mean()) + 0.5, float(y.mean()) + 0.5
        else:
            xc = yc = float("nan")
        st.x_center_of_mass = _get_x(cal, xc)
        st.y_center_of_mass = _get_y(cal, yc, height)


def fit_ellipse(mask, left, top, width, height):
    """``EllipseFitter.fit``: (xCenter, yCenter, major, minor, theta, angle) in pixels."""
    sqrt_pi = 1.772453851
    halfpi = 1.5707963267949
    if mask is None:
        major = (width * 2) / sqrt_pi
        minor = (height * 2) / sqrt_pi
        angle, theta = 0.0, 0.0
        if major < minor:
            major, minor = minor, major
            angle, theta = 90.0, math.pi / 2.0
        return left + width / 2.0, top + height / 2.0, major, minor, theta, angle
    m = np.asarray(mask, bool)
    cnt_row = m.sum(axis=1).astype(np.int64)
    xs = np.arange(m.shape[1], dtype=np.int64)
    xsum_row = (m * xs[None, :]).sum(axis=1).astype(np.int64)
    x2sum = float((m * (xs * xs)[None, :]).sum())
    ys = np.arange(m.shape[0], dtype=np.int64)
    xsum = float(xsum_row.sum())
    ysum = float((cnt_row * ys).sum())
    xysum = float((xsum_row * ys).sum())
    y2sum = float((ys * ys * cnt_row).sum())
    bit_count = int(cnt_row.sum())
    if bit_count == 0:
        return left + width / 2.0, top + height / 2.0, 0.0, 0.0, 0.0, 0.0
    x2sum += 0.08333333 * bit_count
    y2sum += 0.08333333 * bit_count
    n = float(bit_count)
    x1, y1 = xsum / n, ysum / n
    x2, y2, xy = x2sum / n, y2sum / n, xysum / n
    xm, ym = x1, y1
    u20 = x2 - x1 * x1
    u02 = y2 - y1 * y1
    u11 = xy - x1 * y1
    m4 = 4.0 * abs(u02 * u20 - u11 * u11)
    if m4 < 0.000001:
        m4 = 0.000001
    a11, a12, a22 = u02 / m4, u11 / m4, u20 / m4
    tmp = a11 - a22
    if tmp == 0.0:
        tmp = 0.000001
    theta = 0.5 * math.atan(2.0 * a12 / tmp)
    if theta < 0.0:
        theta += halfpi
    if a12 > 0.0:
        theta += halfpi
    elif a12 == 0.0:
        if a22 > a11:
            theta = 0.0
            a22, a11 = a11, a22
        elif a11 != a22:
            theta = halfpi
    tmp = math.sin(theta)
    if tmp == 0.0:
        tmp = 0.000001
    z = a12 * math.cos(theta) / tmp
    major = math.sqrt(1.0 / abs(a22 + z))
    minor = math.sqrt(1.0 / abs(a11 - z))
    scale = math.sqrt(bit_count / (math.pi * major * minor))
    major = major * scale * 2.0
    minor = minor * scale * 2.0
    angle = 180.0 * theta / math.pi
    if angle == 180.0:
        angle = 0.0
    if major < minor:
        major, minor = minor, major
    return left + xm + 0.5, top + ym + 0.5, major, minor, theta, angle


def ellipse_outline_pixels(xc, yc, major, minor, theta, width, height):
    """``EllipseFitter.drawEllipse``: the pixels ImageJ draws for a fitted ellipse."""
    if major == 0.0 and minor == 0.0:
        return []
    xcen, ycen = int(math.floor(xc + 0.5)), int(math.floor(yc + 0.5))
    sint, cost = math.sin(theta), math.cos(theta)
    rmajor2 = 1.0 / (major / 2) ** 2
    rminor2 = 1.0 / (minor / 2) ** 2
    g11 = rmajor2 * cost ** 2 + rminor2 * sint ** 2
    g12 = (rmajor2 - rminor2) * sint * cost
    g22 = rmajor2 * sint ** 2 + rminor2 * cost ** 2
    k1 = -g12 / g11
    k2 = (g12 ** 2 - g11 * g22) / g11 ** 2
    k3 = 1.0 / g11
    ymax = int(math.floor(math.sqrt(abs(k3 / k2))))
    ymax = min(ymax, height)
    ymin = -ymax
    txmin, txmax = {}, {}
    for y in range(ymin, ymax + 1):
        j2 = math.sqrt(abs(k2 * y * y + k3))
        j1 = k1 * y
        txmin[y] = int(math.floor(j1 - j2 + 0.5))
        txmax[y] = int(math.floor(j1 + j2 + 0.5))
    pts = []
    xs = txmin[ymin] if ymin in txmin else 0
    for y in range(ymin, ymax + 1):
        x0, x1 = txmin[y], txmax[y]
        pts.append((xcen + x0, ycen + y))
        pts.append((xcen + x1, ycen + y))
        xs = x0
    pts = [(x, y) for x, y in pts if 0 <= x < width and 0 <= y < height]
    return pts


# ---------------------------------------------------------------------------------- Analyzer
class Analyzer:
    """Analyze > Measure into a Results table (``ij.plugin.filter.Analyzer``).

    ``image`` is the ImageDoc, ``flags`` the Measurements bit mask, ``precision`` the
    decimal places; ``redirect`` another ImageDoc to take pixel values from."""

    def __init__(self, image, flags: int, rt: ResultsTable | None = None, precision: int = 3,
                 redirect=None, weighted_rgb: bool = False):
        self.image = image
        self.flags = flags
        self.rt = rt if rt is not None else ResultsTable.results()
        self.precision = precision
        self.redirect = redirect
        self.weighted_rgb = weighted_rgb
        self.rt.precision = -precision if flags & SCIENTIFIC_NOTATION else precision
        if flags & NAN_EMPTY_CELLS:
            self.rt.nan_empty_cells = True

    # -------------------------------------------------------------- helpers
    def calibration(self, image=None) -> Calibration:
        cal = (image or self.image).calibration.copy()
        if self.flags & INVERT_Y:
            cal.invert_y = True
        return cal

    def threshold(self, image=None):
        t = (image or self.image).threshold
        return None if t is None else (float(t[0]), float(t[1]))

    def plane(self, image=None) -> np.ndarray:
        return (image or self.image).plane()

    def stats_for(self, roi, mflags: int | None = None, image=None) -> ImageStatistics:
        """``imp.getStatistics(measurements)`` with the selection (redirect aware)."""
        src = image or (self.redirect if self.redirect is not None else self.image)
        mflags = self.flags if mflags is None else mflags
        plane = src.plane()
        rect, mask = roi_rect_and_mask(roi, src.width, src.height)
        return get_statistics(plane, src.type, rect, mask, mflags, self.calibration(src),
                              self.threshold(src), weighted_rgb=self.weighted_rgb)

    def file_name(self, roi=None) -> str:
        im = self.image
        s = self.redirect.title if self.redirect is not None else im.title
        roi = im.roi if roi is None else roi
        name = getattr(roi, "name", "") if roi is not None else ""
        if name and "." not in name:
            if len(name) > 30:
                name = name[:27] + "..."
            s += ":" + name
        if im.stack_size > 1:
            label = im.slice_label()
            label = label.splitlines()[0].strip() if label else ""
            colon = "" if s == "" else ":"
            s += colon + (label[:60] if label else str(im.current_index + 1))
        return s

    # -------------------------------------------------------------- measure
    def measure(self, roi=None, use_image_roi: bool = True) -> None:
        if roi is None and use_image_roi:
            roi = self.image.roi
        if roi is not None and isinstance(roi, S.PointRoi):
            self.measure_point(roi)
            return
        if roi is not None and roi.type == S.ANGLE:
            self.measure_angle(roi)
            return
        if roi is not None and roi.is_line():
            self.measure_length(roi)
            return
        if roi is not None and not roi.is_area():
            roi = None
        stats = self.stats_for(roi)
        self.save_results(stats, roi)

    def measure_point(self, roi) -> None:
        src = self.redirect if self.redirect is not None else self.image
        cal = self.calibration(src)
        n = len(roi.xs)
        for i in range(n):
            px, py = float(np.float32(roi.xs[i])), float(np.float32(roi.ys[i]))
            ix, iy = int(math.floor(px + 0.5)), int(math.floor(py + 0.5))
            position = 0
            if roi.positions is not None and n > 1 and i < len(roi.positions):
                position = int(roi.positions[i])
            if src.stack_size > 1 and 0 < position <= src.stack_size:
                plane = src.plane_at(position - 1)
            else:
                plane = src.plane()
            if 0 <= ix < src.width and 0 <= iy < src.height:
                st = get_statistics(plane, src.type, (ix, iy, 1, 1), None, self.flags, cal,
                                    self.threshold(src), weighted_rgb=self.weighted_rgb)
            else:
                st = ImageStatistics()
            st.x_center_of_mass, st.y_center_of_mass = px, py
            point = S.PointRoi([px], [py], name=roi.name)
            info = None
            if roi.counters is not None and len(np.unique(roi.counters)) > 1 and \
                    i < len(roi.counters):
                counter = int(roi.counters[i]) & 0xFF
                info = (counter, int(np.count_nonzero(np.asarray(roi.counters) == counter)))
            self.save_results(st, point, position=position, counter_info=info)

    def measure_angle(self, roi) -> None:
        self.save_results(ImageStatistics(), roi)

    def measure_length(self, roi) -> None:
        im = self.redirect if self.redirect is not None else self.image
        if not (self.flags & (AREA | MEAN | STD_DEV | MODE | MIN_MAX | CENTROID | MEDIAN)):
            self.rt.increment_counter()
            if self.flags & LABELS:
                self.rt.add_label(self.file_name(roi))
            cal = im.calibration
            self.rt.add_value("Length", G.roi_length(roi, cal.pixel_width, cal.pixel_height))
            if isinstance(roi, S.LineRoi):
                self.rt.add_value("Angle", G.line_angle(roi.x1, roi.y1, roi.x2, roi.y2))
            return
        from rsstatchem.core.imaging.profile import line_profile
        limit = self.flags & LIMIT
        cal = self.calibration(im)
        line_width = int(round(max(getattr(roi, "width", 0) or 0, roi.stroke_width or 0, 1)))
        if line_width > 1:
            from rsstatchem.core.imaging.roi.selection_ops import line_to_area
            area = line_to_area(roi, line_width)
            rect, mask = roi_rect_and_mask(area, im.width, im.height)
            st = get_statistics(im.plane(), im.type, rect, mask,
                                AREA | MEAN | STD_DEV | MODE | MIN_MAX | MEDIAN | limit, cal,
                                self.threshold(im), weighted_rgb=self.weighted_rgb)
        else:
            if limit:
                raw = line_profile(im, roi, interpolate=True, calibrated=False)
                arr = _to_original_depth(raw, im.type)
                ptype = {np.dtype(np.uint8): GRAY8, np.dtype(np.uint16): GRAY16}.get(
                    arr.dtype, GRAY32)
            else:
                vals = line_profile(im, roi, interpolate=True, calibrated=True)
                arr = np.asarray(vals, dtype=np.float32)[None, :]
                ptype = GRAY32
            pcal = cal.copy()
            if ptype == GRAY32:
                pcal.function, pcal.coefficients = NONE, ()
            st = get_statistics(arr, ptype, None, None,
                                AREA | MEAN | STD_DEV | MODE | MIN_MAX | MEDIAN | limit, pcal,
                                self.threshold(im) if limit else None)
        if isinstance(roi, S.LineRoi) and (self.flags & CENTROID):
            x = float(np.float32(roi.x1)) + (float(np.float32(roi.x2)) -
                                             float(np.float32(roi.x1))) / 2.0
            y = float(np.float32(roi.y1)) + (float(np.float32(roi.y2)) -
                                             float(np.float32(roi.y1))) / 2.0
            c2 = self.calibration(self.image)
            st.x_centroid = c2.get_x(x)
            st.y_centroid = c2.get_y(y, im.height)
        self.save_results(st, roi)

    # -------------------------------------------------------------- results
    def save_results(self, stats: ImageStatistics, roi, position: int = 0,
                     counter_info=None) -> None:
        rt, m, im = self.rt, self.flags, self.image
        clear_summary(rt)
        rt.increment_counter()
        cal = self.calibration(im)
        pw, ph = cal.pixel_width, cal.pixel_height
        if m & LABELS:
            rt.add_label(self.file_name(roi))
        if m & AREA:
            rt.add_value("Area", stats.area)
        if m & MEAN:
            rt.add_value("Mean", stats.mean)
        if m & STD_DEV:
            rt.add_value("StdDev", stats.std_dev)
        if m & MODE:
            rt.add_value("Mode", stats.dmode)
        if m & MIN_MAX:
            rt.add_value("Min", stats.min)
            rt.add_value("Max", stats.max)
        if m & CENTROID:
            rt.add_value("X", stats.x_centroid)
            rt.add_value("Y", stats.y_centroid)
        if m & CENTER_OF_MASS:
            rt.add_value("XM", stats.x_center_of_mass)
            rt.add_value("YM", stats.y_center_of_mass)
        if m & (PERIMETER | SHAPE_DESCRIPTORS):
            if roi is not None:
                perimeter = G.roi_length(roi, pw, ph)
            else:
                perimeter = float(im.width * 2 + im.height * 2)
            if m & PERIMETER:
                rt.add_value("Perim.", perimeter)
            if m & SHAPE_DESCRIPTORS:
                circ = 0.0 if perimeter == 0.0 else 4.0 * math.pi * (
                    stats.area / (perimeter * perimeter))
                circ = min(circ, 1.0)
                rt.add_value("Circ.", circ)
                is_area = roi is None or roi.is_area()
                if roi is not None:
                    hx, hy = G.float_convex_hull(roi)
                    convex = G.int_polygon_area(hx, hy)
                else:
                    convex = float(stats.pixel_count)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ar = stats.major / stats.minor if is_area and stats.minor else (
                        0.0 if not is_area else float("inf"))
                    rnd = 4.0 * stats.area / (math.pi * stats.major * stats.major) \
                        if is_area and stats.major else (0.0 if not is_area else float("nan"))
                    sol = (stats.pixel_count / convex if convex else float("nan")) \
                        if is_area else float("nan")
                rt.add_value("AR", ar)
                rt.add_value("Round", rnd)
                rt.add_value("Solidity", sol)
                if rt.size == 1:
                    for col in ("Circ.", "AR", "Round", "Solidity"):
                        rt.decimal_places[col] = self.precision
        if m & RECT:
            if roi is not None and roi.is_line():
                bx, by, bw, bh = G.int_bounds(roi)
                rt.add_value("BX", cal.get_x(bx))
                rt.add_value("BY", cal.get_y(by, im.height))
                rt.add_value("Width", bw * pw)
                rt.add_value("Height", bh * ph)
            else:
                rt.add_value("BX", stats.roi_x)
                rt.add_value("BY", stats.roi_y)
                rt.add_value("Width", stats.roi_width)
                rt.add_value("Height", stats.roi_height)
        if m & ELLIPSE:
            rt.add_value("Major", stats.major)
            rt.add_value("Minor", stats.minor)
            rt.add_value("Angle", stats.angle)
        if m & FERET:
            vals = [float("nan")] * 5
            roi2 = roi if roi is not None else S.RectRoi(0, 0, im.width, im.height)
            fpw, fph = (pw, ph) if roi is not None else (1.0, 1.0)
            a = G.feret_values(roi2, fpw, fph)
            if a is not None:
                vals = [a[0], a[1], a[2], a[3], a[4]]
            rt.add_value("Feret", vals[0])
            rt.add_value("FeretX", vals[3])
            rt.add_value("FeretY", vals[4])
            rt.add_value("FeretAngle", vals[1])
            rt.add_value("MinFeret", vals[2])
        if m & INTEGRATED_DENSITY:
            rt.add_value("IntDen", stats.area * stats.mean)
            rt.add_value("RawIntDen", stats.pixel_count * stats.umean)
        if m & MEDIAN:
            rt.add_value("Median", stats.median)
        if m & SKEWNESS:
            rt.add_value("Skew", stats.skewness)
        if m & KURTOSIS:
            rt.add_value("Kurt", stats.kurtosis)
        if m & AREA_FRACTION:
            rt.add_value("%Area", stats.area_fraction)
        if m & STACK_POSITION:
            c, z, t = im.position
            if im.hyperstack or im.is_composite:
                if im.n_channels > 1:
                    rt.add_value("Ch", c + 1)
                if im.n_slices > 1:
                    rt.add_value("Slice", z + 1)
                if im.n_frames > 1:
                    rt.add_value("Frame", t + 1)
            else:
                rt.add_value("Slice", im.current_index + 1)
        if roi is not None:
            if roi.type == S.ANGLE:
                rt.add_value("Angle", G.polygon_angle(roi))
            elif roi.is_line():
                rt.add_value("Length", G.roi_length(roi, pw, ph))
                if isinstance(roi, S.LineRoi):
                    rt.add_value("Angle", G.line_angle(float(np.float32(roi.x1)),
                                                       float(np.float32(roi.y1)),
                                                       float(np.float32(roi.x2)),
                                                       float(np.float32(roi.y2))))
            elif isinstance(roi, S.PointRoi):
                self._save_points(roi, cal, position, counter_info)
        if (m & LIMIT) and not im.is_rgb:
            rt.add_value("MinThr", stats.lower_threshold)
            rt.add_value("MaxThr", stats.upper_threshold)
        if isinstance(roi, S.RotatedRectRoi):
            length = math.hypot(roi.x2 - roi.x1, roi.y2 - roi.y1)
            p = pw if pw == ph else 1.0
            rt.add_value("RRLength", length * p)
            rt.add_value("RRWidth", roi.rect_width * p)
        if (m & ADD_TO_OVERLAY) and roi is not None:
            from rsstatchem.core.imaging.roi.shapes import Overlay
            if im.overlay is None:
                im.overlay = Overlay(draw_names=True, draw_backgrounds=True,
                                     label_color="#ffffff")
            im.overlay.add(roi.copy())

    def _save_points(self, roi, cal, position, counter_info):
        rt, im = self.rt, self.image
        if self.flags & AREA:
            rt.add_value("Area", 0)
        x, y = float(roi.xs[0]), float(roi.ys[0])
        rt.add_value("X", cal.get_x(x))
        rt.add_value("Y", cal.get_y(y, im.height))
        if im.hyperstack or im.is_composite:
            c, z, t = im.position
            if position > 0:
                c, z, t = im.position_of(position - 1)
            if im.n_channels > 1:
                rt.add_value("Ch", c + 1)
            if im.n_slices > 1:
                rt.add_value("Slice", z + 1)
            if im.n_frames > 1:
                rt.add_value("Frame", t + 1)
        elif im.stack_size > 1:
            rt.add_value("Slice", position if position else im.current_index + 1)
        if counter_info is not None:
            rt.add_value("Counter", counter_info[0])
            rt.add_value("Count", counter_info[1])


def _to_original_depth(values, image_type):
    v = np.asarray(values, dtype=np.float64)
    if image_type in (GRAY8, COLOR_RGB):
        return np.clip(np.floor(v + 0.5), 0, 255).astype(np.uint8)[None, :]
    if image_type == GRAY16:
        return np.clip(np.floor(v + 0.5), 0, 65535).astype(np.uint16)[None, :]
    return v.astype(np.float32)[None, :]


def clear_summary(rt: ResultsTable) -> None:
    """Remove the Mean/SD/Min/Max rows that Summarize appended."""
    if rt.size >= 4 and rt.get_label(rt.size - 1) == "Max":
        for _ in range(4):
            rt.delete_row(rt.size - 1)


def summarize(rt: ResultsTable) -> bool:
    """``Analyzer.summarize``: append Mean, SD, Min, Max rows (SD n-1) to ``rt``."""
    n = rt.size
    if n < 2 or rt.get_label(n - 1) == "Max":
        return False
    heads = [h for h in rt.headings if h != "Label"]
    cols = {h: rt.column(h) for h in heads}
    for lab in ("Mean", "SD", "Min", "Max"):
        r = rt.increment_counter()
        rt.set_label(lab, r)
    for h, v in cols.items():
        s = float(np.sum(v))
        s2 = float(np.sum(v * v))
        rt.set_value(h, n, s / n)
        with np.errstate(invalid="ignore"):
            rt.set_value(h, n + 1, math.sqrt((s2 - s * s / n) / (n - 1))
                         if n > 1 and (s2 - s * s / n) >= 0 else float("nan"))
        finite = v[~np.isnan(v)]
        rt.set_value(h, n + 2, float(finite.min()) if finite.size else 1.7976931348623157e308)
        rt.set_value(h, n + 3, float(finite.max()) if finite.size else -1.7976931348623157e308)
    return True

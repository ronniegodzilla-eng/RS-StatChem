"""Auto Local Threshold (Fiji's "Auto Local Threshold" methods), written clean-room from the
published method descriptions - no code of the GPL Fiji plugin is used:

* Bernsen J. (1986) "Dynamic thresholding of grey-level images", Proc. 8th ICPR;
* Niblack W. (1986) "An Introduction to Digital Image Processing", Prentice-Hall;
* Sauvola J., Pietikainen M. (2000) "Adaptive document image binarization", Pattern
  Recognition 33(2);
* Phansalkar N. et al. (2011) "Adaptive local thresholding for detection of nuclei in
  diversity stained cytology images", ICCSP;
* Otsu N. (1979) "A threshold selection method from gray-level histograms", IEEE SMC 9(1),
  applied to the histogram of each pixel's neighbourhood;
* Contrast, Mean, Median and MidGrey are the textbook local rules (Gonzalez & Woods).

The neighbourhood is ImageJ's circular kernel (``RankFilters.makeLineRadii``, see
:func:`rsstatchem.core.imaging.binary.line_radii`) with edge pixels repeated outside the image
(as ImageJ's RankFilters do). Input is an 8-bit plane; :func:`local_threshold` returns a bool
mask of the *selected* pixels: with ``white_objects`` the pixels above the local threshold,
otherwise those at or below it (the Fiji "White objects on black background" switch).
Parameters 1/2 of 0 mean "method default" (documented defaults below).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from rsstatchem.core.imaging.binary import line_radii

METHODS = ("Bernsen", "Contrast", "Mean", "Median", "MidGrey", "Niblack", "Otsu",
           "Phansalkar", "Sauvola")

# (parameter 1, parameter 2) defaults used when a parameter is 0
DEFAULTS = {
    "Bernsen": (15.0, 0.0),        # contrast threshold
    "Contrast": (0.0, 0.0),
    "Mean": (0.0, 0.0),            # C offset (0 is the documented default)
    "Median": (0.0, 0.0),          # C offset
    "MidGrey": (0.0, 0.0),         # C offset
    "Niblack": (0.2, 0.0),         # k, C offset
    "Otsu": (0.0, 0.0),
    "Phansalkar": (0.25, 0.5),     # k, r (on the [0, 1]-normalised image)
    "Sauvola": (0.5, 128.0),       # k, r (dynamic range of the standard deviation)
}
PHANSALKAR_P, PHANSALKAR_Q = 2.0, 10.0


def footprint(radius: float) -> np.ndarray:
    """Bool kernel of ImageJ's circular neighbourhood of ``radius``."""
    rows = line_radii(radius)
    k = max(abs(dy) for dy, _ in rows)
    fp = np.zeros((2 * k + 1, 2 * k + 1), bool)
    for dy, dx in rows:
        fp[k + dy, k - dx:k + dx + 1] = True
    return fp


def _row_sums(a: np.ndarray, radius: float) -> tuple[np.ndarray, int]:
    """Sum of ``a`` (float64) over the circular kernel, edges repeated; and the kernel size."""
    rows = line_radii(radius)
    k = max(abs(dy) for dy, _ in rows)
    h, w = a.shape
    p = np.pad(a.astype(np.float64), k, mode="edge")
    cs = np.zeros((p.shape[0], p.shape[1] + 1), np.float64)
    np.cumsum(p, axis=1, out=cs[:, 1:])
    total = np.zeros((h, w), np.float64)
    n = 0
    xs = np.arange(w) + k
    for dy, dx in rows:
        r = cs[k + dy:k + dy + h]
        total += r[:, xs + dx + 1] - r[:, xs - dx]
        n += 2 * dx + 1
    return total, n


def local_mean(a: np.ndarray, radius: float) -> np.ndarray:
    s, n = _row_sums(a, radius)
    return s / n


def local_mean_std(a: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean and population standard deviation over the circular kernel."""
    a = a.astype(np.float64)
    s, n = _row_sums(a, radius)
    s2, _ = _row_sums(a * a, radius)
    mean = s / n
    var = np.maximum(s2 / n - mean * mean, 0.0)
    return mean, np.sqrt(var)


def local_min_max(a: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    fp = footprint(radius)
    return (ndi.grey_erosion(a, footprint=fp, mode="nearest"),
            ndi.grey_dilation(a, footprint=fp, mode="nearest"))


def local_median(a: np.ndarray, radius: float) -> np.ndarray:
    return ndi.median_filter(a, footprint=footprint(radius), mode="nearest")


def local_histograms(a: np.ndarray, radius: float, y0: int, y1: int, bins: int = 256
                     ) -> np.ndarray:
    """Counts (bins, y1-y0, W) of the 8-bit values of ``a`` in the circular neighbourhood of
    every pixel of rows y0..y1-1 (edges repeated)."""
    rows = line_radii(radius)
    k = max(abs(dy) for dy, _ in rows)
    h, w = a.shape
    p = np.pad(a, k, mode="edge")[y0:y1 + 2 * k]          # rows y0-k .. y1+k-1
    n = y1 - y0
    xs = np.arange(w) + k
    out = np.zeros((bins, n, w), np.int32)
    present = np.unique(p)
    for b in present:
        ind = (p == b)
        cs = np.zeros((ind.shape[0], ind.shape[1] + 1), np.int32)
        np.cumsum(ind, axis=1, out=cs[:, 1:])
        acc = np.zeros((n, w), np.int32)
        for dy, dx in rows:
            r = cs[k + dy:k + dy + n]
            acc += r[:, xs + dx + 1] - r[:, xs - dx]
        out[int(b)] = acc
    return out


def otsu_from_histograms(hist: np.ndarray) -> np.ndarray:
    """Otsu threshold (largest between-class variance; first maximum) of each histogram along
    axis 0; pixels > threshold belong to the upper class."""
    h = hist.astype(np.float64)
    levels = np.arange(h.shape[0], dtype=np.float64).reshape((-1,) + (1,) * (h.ndim - 1))
    w0 = np.cumsum(h, axis=0)
    mu = np.cumsum(h * levels, axis=0)
    total = w0[-1]
    mu_t = mu[-1]
    w1 = total - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu_t * w0 - mu * total) ** 2 / (w0 * w1)
    between = np.where((w0 > 0) & (w1 > 0), between, -1.0)
    return np.argmax(between, axis=0)


def local_otsu_threshold(a: np.ndarray, radius: float, chunk: int = 32) -> np.ndarray:
    h = a.shape[0]
    out = np.zeros(a.shape, np.int32)
    for y0 in range(0, h, chunk):
        y1 = min(y0 + chunk, h)
        out[y0:y1] = otsu_from_histograms(local_histograms(a, radius, y0, y1))
    return out


def _params(method: str, p1: float, p2: float) -> tuple[float, float]:
    d1, d2 = DEFAULTS[method]
    return (d1 if p1 == 0 else float(p1)), (d2 if p2 == 0 else float(p2))


def local_threshold(plane: np.ndarray, method: str, radius: float = 15.0, p1: float = 0.0,
                    p2: float = 0.0, white_objects: bool = True) -> np.ndarray:
    """Bool mask of the selected pixels of an 8-bit ``plane`` for one ``method``."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    a = np.asarray(plane)
    if a.dtype != np.uint8:
        raise ValueError("8-bit image required")
    v = a.astype(np.float64)
    p1, p2 = _params(method, p1, p2)
    if method == "Bernsen":
        lo, hi = local_min_max(a, radius)
        lo, hi = lo.astype(np.float64), hi.astype(np.float64)
        mid = (lo + hi) / 2.0
        low_contrast = (hi - lo) < p1
        # low contrast: the whole neighbourhood is one class, decided by the mid grey
        above = np.where(low_contrast, mid >= 128, v >= mid)
        return above if white_objects else ~above
    if method == "Contrast":
        lo, hi = local_min_max(a, radius)
        above = (hi.astype(np.float64) - v) <= (v - lo.astype(np.float64))
        return above if white_objects else ~above
    if method == "Mean":
        t = local_mean(v, radius) - p1
    elif method == "Median":
        t = local_median(a, radius).astype(np.float64) - p1
    elif method == "MidGrey":
        lo, hi = local_min_max(a, radius)
        t = (lo.astype(np.float64) + hi.astype(np.float64)) / 2.0 - p1
    elif method == "Niblack":
        mean, sd = local_mean_std(v, radius)
        t = mean + p1 * sd - p2
    elif method == "Otsu":
        t = local_otsu_threshold(a, radius).astype(np.float64)
    elif method == "Phansalkar":
        n = v / 255.0
        mean, sd = local_mean_std(n, radius)
        t = mean * (1.0 + PHANSALKAR_P * np.exp(-PHANSALKAR_Q * mean) +
                    p1 * (sd / p2 - 1.0))
        v = n
    else:  # Sauvola
        mean, sd = local_mean_std(v, radius)
        t = mean * (1.0 + p1 * (sd / p2 - 1.0))
    above = v > t
    return above if white_objects else ~above


def to_mask(selected: np.ndarray) -> np.ndarray:
    """uint8 0/255 mask (255 = selected) as ImageJ masks store objects."""
    return np.where(selected, np.uint8(255), np.uint8(0))


def try_all(plane: np.ndarray, radius: float = 15.0, p1: float = 0.0, p2: float = 0.0,
            white_objects: bool = True) -> list[tuple[str, np.ndarray]]:
    """[(method, 0/255 mask)] for every method (Fiji's "Try all" montage panels)."""
    return [(m, to_mask(local_threshold(plane, m, radius, p1, p2, white_objects)))
            for m in METHODS]


def montage(panels: list[np.ndarray], columns: int = 3, border: int = 0) -> np.ndarray:
    """Tile equally sized 2-D panels row by row (left-to-right, top-to-bottom)."""
    h, w = panels[0].shape
    rows = (len(panels) + columns - 1) // columns
    out = np.zeros((rows * h + (rows - 1) * border, columns * w + (columns - 1) * border),
                   panels[0].dtype)
    for i, p in enumerate(panels):
        r, c = divmod(i, columns)
        y, x = r * (h + border), c * (w + border)
        out[y:y + h, x:x + w] = p
    return out

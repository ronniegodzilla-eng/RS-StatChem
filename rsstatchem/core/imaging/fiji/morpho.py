"""MorphoLibJ-lite: mathematical morphology for 2-D planes, written clean-room from the
literature (no code of the LGPL MorphoLibJ library is used):

* Serra J. (1982) "Image Analysis and Mathematical Morphology"; Soille P. (2003)
  "Morphological Image Analysis" - structuring elements, erosion/dilation/opening/closing,
  top hats, gradients, Laplacian, regional and extended (h-) extrema, fill holes, kill borders;
* Vincent L. (1993) "Morphological grayscale reconstruction in image analysis", IEEE TIP 2(2)
  - reconstruction by dilation/erosion (here: the sequential raster/anti-raster propagation,
  vectorised along rows and columns, repeated until stable);
* Borgefors G. (1986) "Distance transformations in digital images", CVGIP 34 - two-pass
  chamfer distances (3-4, 5-7, 5-7-11 "chessknight", ...); exact Euclidean via scipy;
* Meyer F. (1991) "Un algorithme optimal de ligne de partage des eaux" - priority-flood
  marker-controlled watershed (optionally with watershed lines = "dams");
* connected-component labelling via scipy.ndimage.label (BSD).

Conventions: images are 2-D numpy arrays (uint8, uint16 or float32); results keep the input
dtype unless stated; borders replicate the edge pixels (``mode="nearest"``). Connectivity is
4 or 8. Binary results are uint8 0/255.
"""
from __future__ import annotations

import heapq
import math

import numpy as np
from scipy import ndimage as ndi

SHAPES = ("Square", "Diamond", "Octagon", "Disk", "Line Horizontal", "Line Vertical",
          "Line 45 degrees", "Line 135 degrees")
OPERATIONS = ("Erosion", "Dilation", "Opening", "Closing", "White Top Hat", "Black Top Hat",
              "Gradient", "Laplacian", "Internal Gradient", "External Gradient")
CHAMFER_WEIGHTS = {
    "Chessboard (1,1)": (1.0, 1.0, None),
    "City-Block (1,2)": (1.0, 2.0, None),
    "Quasi-Euclidean (1,1.41)": (1.0, 1.41, None),
    "Borgefors (3,4)": (3.0, 4.0, None),
    "Weights (2,3)": (2.0, 3.0, None),
    "Weights (5,7)": (5.0, 7.0, None),
    "Chessknight (5,7,11)": (5.0, 7.0, 11.0),
    "Euclidean (exact)": None,
}


# ---------------------------------------------------------------------------------- elements
def strel(shape: str, radius: int) -> np.ndarray:
    """Bool footprint (2r+1, 2r+1) of a structuring element centred on the origin."""
    r = max(int(radius), 0)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    if shape == "Square":
        fp = np.ones_like(xx, bool)
    elif shape == "Diamond":
        fp = (np.abs(xx) + np.abs(yy)) <= r
    elif shape == "Octagon":
        # regular octagon of apothem r: square cut by the diamond |x|+|y| <= r*sqrt(2)
        fp = (np.abs(xx) + np.abs(yy)) <= int(math.floor(r * math.sqrt(2.0) + 0.5))
    elif shape == "Disk":
        fp = (xx * xx + yy * yy) <= r * r + r       # digital disk (r + 1/2)^2 rounded down
    elif shape == "Line Horizontal":
        fp = yy == 0
    elif shape == "Line Vertical":
        fp = xx == 0
    elif shape == "Line 45 degrees":
        fp = xx == -yy                               # lower-left to upper-right (y down)
    elif shape == "Line 135 degrees":
        fp = xx == yy
    else:
        raise ValueError(f"unknown structuring element {shape!r}")
    return fp


def _clip_to(values: np.ndarray, dtype) -> np.ndarray:
    if np.dtype(dtype) == np.float32:
        return values.astype(np.float32)
    info = np.iinfo(dtype)
    return np.clip(np.floor(values + 0.5), info.min, info.max).astype(dtype)


def erosion(a: np.ndarray, fp: np.ndarray) -> np.ndarray:
    return ndi.grey_erosion(a, footprint=fp, mode="nearest")


def dilation(a: np.ndarray, fp: np.ndarray) -> np.ndarray:
    return ndi.grey_dilation(a, footprint=fp, mode="nearest")


def morphological_filter(a: np.ndarray, operation: str, shape: str = "Disk",
                         radius: int = 2) -> np.ndarray:
    """One of :data:`OPERATIONS` with the structuring element ``shape``/``radius``.
    Differences are clipped to the dtype; the Laplacian ((dilation - f) - (f - erosion)) is
    offset by half the range (128 for 8-bit, 32768 for 16-bit) so negative values show."""
    a = np.asarray(a)
    fp = strel(shape, radius)
    if operation == "Erosion":
        return erosion(a, fp)
    if operation == "Dilation":
        return dilation(a, fp)
    if operation == "Opening":
        return dilation(erosion(a, fp), fp)
    if operation == "Closing":
        return erosion(dilation(a, fp), fp)
    f = a.astype(np.float64)
    if operation == "White Top Hat":
        out = f - dilation(erosion(a, fp), fp)
    elif operation == "Black Top Hat":
        out = erosion(dilation(a, fp), fp) - f
    elif operation == "Gradient":
        out = dilation(a, fp).astype(np.float64) - erosion(a, fp)
    elif operation == "Internal Gradient":
        out = f - erosion(a, fp)
    elif operation == "External Gradient":
        out = dilation(a, fp) - f
    elif operation == "Laplacian":
        out = dilation(a, fp).astype(np.float64) + erosion(a, fp) - 2.0 * f
        if a.dtype == np.uint8:
            out += 128
        elif a.dtype == np.uint16:
            out += 32768
    else:
        raise ValueError(f"unknown operation {operation!r}")
    return _clip_to(out, a.dtype)


# ---------------------------------------------------------------------------------- reconstruction
def _sweep_rows(f: np.ndarray, g: np.ndarray, conn: int) -> None:
    """Propagate f = min(max(f, previous row neighbours), g) down then up (in place)."""
    h = f.shape[0]
    for order, step in ((range(1, h), -1), (range(h - 2, -1, -1), 1)):
        for y in order:
            prev = f[y + step]
            if conn == 8:
                c = prev.copy()
                np.maximum(c[1:], prev[:-1], out=c[1:])
                np.maximum(c[:-1], prev[1:], out=c[:-1])
            else:
                c = prev
            row = f[y]
            np.maximum(row, c, out=row)
            np.minimum(row, g[y], out=row)


def reconstruct_by_dilation(marker: np.ndarray, mask: np.ndarray, connectivity: int = 4
                            ) -> np.ndarray:
    """Grey reconstruction of ``mask`` from ``marker`` (marker is clipped to the mask)."""
    g = np.asarray(mask, np.float64)
    f = np.minimum(np.asarray(marker, np.float64), g)
    if f.size == 0:
        return f
    while True:
        prev = f.copy()
        _sweep_rows(f, g, connectivity)
        ft, gt = np.ascontiguousarray(f.T), np.ascontiguousarray(g.T)
        _sweep_rows(ft, gt, connectivity)
        f = np.ascontiguousarray(ft.T)
        if np.array_equal(prev, f):
            return f


def reconstruct_by_erosion(marker: np.ndarray, mask: np.ndarray, connectivity: int = 4
                           ) -> np.ndarray:
    return -reconstruct_by_dilation(-np.asarray(marker, np.float64),
                                    -np.asarray(mask, np.float64), connectivity)


def reconstruction(marker: np.ndarray, mask: np.ndarray, by: str = "dilation",
                   connectivity: int = 4) -> np.ndarray:
    """Reconstruction in the dtype of ``mask``."""
    fn = reconstruct_by_dilation if by.lower().endswith("dilation") else reconstruct_by_erosion
    return _clip_to(fn(marker, mask, connectivity), np.asarray(mask).dtype)


# ---------------------------------------------------------------------------------- extrema
def _step(f: np.ndarray) -> float:
    """A positive value smaller than every difference between distinct values of ``f``."""
    u = np.unique(f[np.isfinite(f)])
    if u.size < 2:
        return 1.0
    return float(np.min(np.diff(u))) / 2.0


def _snap(f: np.ndarray) -> np.ndarray:
    """Merge values that differ only by floating-point round-off (relative 1e-9), so that
    plateaus of computed images (h-maxima, normalised distances) stay plateaus."""
    if np.issubdtype(np.asarray(f).dtype, np.integer):
        return np.asarray(f, np.float64)
    f = np.asarray(f, np.float64)
    finite = f[np.isfinite(f)]
    scale = float(np.abs(finite).max()) if finite.size else 0.0
    tol = 1e-9 * max(scale, 1.0)
    return np.where(np.isfinite(f), np.round(f / tol) * tol, f)


def regional_maxima(a: np.ndarray, connectivity: int = 4) -> np.ndarray:
    """Bool mask of the regional maxima (plateaus without a higher neighbour)."""
    f = _snap(a)
    d = _step(f)
    r = reconstruct_by_dilation(f - d, f, connectivity)
    return (f - r) > d / 2.0


def regional_minima(a: np.ndarray, connectivity: int = 4) -> np.ndarray:
    return regional_maxima(-np.asarray(a, np.float64), connectivity)


def extended_maxima(a: np.ndarray, dynamic: float, connectivity: int = 4) -> np.ndarray:
    """Regional maxima of the h-maxima transform (maxima of dynamic >= ``dynamic``)."""
    f = np.asarray(a, np.float64)
    hmax = reconstruct_by_dilation(f - float(dynamic), f, connectivity)
    return regional_maxima(hmax, connectivity)


def extended_minima(a: np.ndarray, dynamic: float, connectivity: int = 4) -> np.ndarray:
    return extended_maxima(-np.asarray(a, np.float64), dynamic, connectivity)


def to_binary(mask: np.ndarray) -> np.ndarray:
    return np.where(mask, np.uint8(255), np.uint8(0))


# ---------------------------------------------------------------------------------- border ops
def fill_holes(a: np.ndarray, connectivity: int = 4) -> np.ndarray:
    """Fill the regional minima not connected to the border (binary or grey)."""
    a = np.asarray(a)
    f = a.astype(np.float64)
    marker = np.full_like(f, f.max() if f.size else 0)
    marker[0, :], marker[-1, :], marker[:, 0], marker[:, -1] = f[0, :], f[-1, :], f[:, 0], \
        f[:, -1]
    return _clip_to(reconstruct_by_erosion(marker, f, connectivity), a.dtype)


def kill_borders(a: np.ndarray, connectivity: int = 4) -> np.ndarray:
    """Remove the structures (bright, binary or grey) connected to the image border."""
    a = np.asarray(a)
    f = a.astype(np.float64)
    marker = np.full_like(f, f.min() if f.size else 0)
    marker[0, :], marker[-1, :], marker[:, 0], marker[:, -1] = f[0, :], f[-1, :], f[:, 0], \
        f[:, -1]
    r = reconstruct_by_dilation(marker, f, connectivity)
    return _clip_to(f - r + (f.min() if f.size else 0), a.dtype)


# ---------------------------------------------------------------------------------- distances
def chamfer_distance(binary: np.ndarray, weights: str = "Borgefors (3,4)",
                     normalize: bool = True) -> np.ndarray:
    """Distance of every foreground (non-zero) pixel to the nearest background pixel (float64;
    background 0). Two-pass chamfer (Borgefors) or exact Euclidean. Outside the image is not
    background; an image without background gives 0."""
    fg = np.asarray(binary) != 0
    w = CHAMFER_WEIGHTS[weights]
    if w is None:
        if not (~fg).any():
            return np.zeros(fg.shape)
        return ndi.distance_transform_edt(fg)
    a, b, c = w
    h, wd = fg.shape
    big = 1e30
    d = np.where(fg, big, 0.0)
    xs = np.arange(wd, dtype=np.float64) * a

    def from_rows(y, dy):
        row = d[y]
        p = d[y + dy] if 0 <= y + dy < h else None
        if p is not None:
            np.minimum(row, p + a, out=row)
            np.minimum(row[1:], p[:-1] + b, out=row[1:])
            np.minimum(row[:-1], p[1:] + b, out=row[:-1])
            if c is not None:
                np.minimum(row[2:], p[:-2] + c, out=row[2:])
                np.minimum(row[:-2], p[2:] + c, out=row[:-2])
        if c is not None and 0 <= y + 2 * dy < h:
            q = d[y + 2 * dy]
            np.minimum(row[1:], q[:-1] + c, out=row[1:])
            np.minimum(row[:-1], q[1:] + c, out=row[:-1])

    for y in range(h):                       # forward: previous rows, then left neighbour
        from_rows(y, -1)
        d[y] = np.minimum.accumulate(d[y] - xs) + xs
    for y in range(h - 1, -1, -1):           # backward: next rows, then right neighbour
        from_rows(y, 1)
        r = d[y][::-1]
        d[y] = (np.minimum.accumulate(r - xs) + xs)[::-1]
    d[d >= big / 2] = 0.0
    return d / a if normalize else d


def convert_distance(d: np.ndarray, output: str) -> np.ndarray:
    """"16 bits" -> uint16 (rounded, clipped), anything else -> float32."""
    if "16" in output:
        return np.clip(np.floor(d + 0.5), 0, 65535).astype(np.uint16)
    return d.astype(np.float32)


# ---------------------------------------------------------------------------------- labelling
def _structure(connectivity: int) -> np.ndarray:
    return ndi.generate_binary_structure(2, 2 if connectivity == 8 else 1)


def label_components(binary: np.ndarray, connectivity: int = 4) -> tuple[np.ndarray, int]:
    """Labels (int32, raster order of first pixel) of the non-zero pixels."""
    lab, n = ndi.label(np.asarray(binary) != 0, structure=_structure(connectivity))
    return lab.astype(np.int32), int(n)


def labels_as(labels: np.ndarray, type_name: str) -> np.ndarray:
    """Label image as "8 bits" (uint8), "16 bits" (uint16) or "float"/"32 bits" (float32);
    raises ValueError when the labels do not fit."""
    n = int(labels.max()) if labels.size else 0
    if "8" in type_name and "16" not in type_name:
        if n > 255:
            raise ValueError(f"Too many labels ({n}) for an 8-bit image")
        return labels.astype(np.uint8)
    if "16" in type_name:
        if n > 65535:
            raise ValueError(f"Too many labels ({n}) for a 16-bit image")
        return labels.astype(np.uint16)
    return labels.astype(np.float32)


# ---------------------------------------------------------------------------------- watershed
def watershed(image: np.ndarray, markers: np.ndarray, mask: np.ndarray | None = None,
              connectivity: int = 4, dams: bool = True) -> np.ndarray:
    """Marker-controlled watershed by priority flooding (Meyer). ``markers`` is a label image
    (> 0 = seed of that label). Pixels reached from two basins become 0 when ``dams``.
    Returns int32 labels (0 = dam / not reached / outside the mask)."""
    f = np.asarray(image, np.float64)
    h, w = f.shape
    w2 = w + 2
    lab = np.zeros((h + 2, w2), np.int64)
    lab[1:-1, 1:-1] = np.asarray(markers, np.int64)
    inside = np.zeros((h + 2, w2), bool)
    inside[1:-1, 1:-1] = True if mask is None else (np.asarray(mask) != 0)
    lab[~inside] = 0
    fp = np.zeros((h + 2, w2))
    fp[1:-1, 1:-1] = f
    if connectivity == 8:
        offs = (-w2 - 1, -w2, -w2 + 1, -1, 1, w2 - 1, w2, w2 + 1)
    else:
        offs = (-w2, -1, 1, w2)
    seeded = lab > 0
    # first front: unlabelled mask pixels touching a seed
    near = ndi.binary_dilation(seeded, structure=_structure(connectivity)) & ~seeded & inside
    labels = lab.ravel().tolist()
    allowed = inside.ravel().tolist()
    values = fp.ravel().tolist()
    queued = seeded.ravel().tolist()
    heap = []
    counter = 0
    for p in np.flatnonzero(near.ravel()).tolist():
        heap.append((values[p], counter, p))
        queued[p] = True
        counter += 1
    heapq.heapify(heap)
    push, pop = heapq.heappush, heapq.heappop
    while heap:
        _, _, p = pop(heap)
        found = 0
        dam = False
        for o in offs:
            lq = labels[p + o]
            if lq > 0:
                if found == 0:
                    found = lq
                elif lq != found:
                    dam = True
                    break
        if dams and dam:
            labels[p] = -1
            continue
        labels[p] = found
        if found <= 0:
            continue
        for o in offs:
            q = p + o
            if allowed[q] and not queued[q]:
                queued[q] = True
                push(heap, (values[q], counter, q))
                counter += 1
    out = np.asarray(labels, np.int64).reshape(h + 2, w2)[1:-1, 1:-1]
    out[out < 0] = 0
    return out.astype(np.int32)


def distance_transform_watershed(binary: np.ndarray, weights: str = "Borgefors (3,4)",
                                 normalize: bool = True, dynamic: float = 1.0,
                                 connectivity: int = 4, dams: bool = True) -> np.ndarray:
    """Split touching objects of a binary image: watershed of the inverted distance map,
    seeded by its extended minima of ``dynamic``. Returns int32 labels (dams 0)."""
    fg = np.asarray(binary) != 0
    dist = chamfer_distance(fg, weights, normalize)
    inv = -dist
    seeds = extended_minima(inv, dynamic, connectivity) & fg
    markers, _ = label_components(seeds, connectivity)
    return watershed(inv, markers, fg, connectivity, dams)

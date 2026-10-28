"""Process > Binary: erode, dilate, open, close, median, outline, fill holes, skeletonize, and the
Euclidean distance map family (Distance Map, Ultimate Points, Watershed, Voronoi).

Ported from ImageJ 1.54p ij/plugin/filter/Binary.java, ij/process/BinaryProcessor.java,
ij/process/ByteProcessor.java (filter ERODE/DILATE, filterEdge) and ij/plugin/filter/EDM.java
(public domain, NIH; W. Rasband, G. Landini, M. Schmid et al.). Results are pixel-identical to
ImageJ (golden case ``process_binary``).

All functions take and return 2-D ``uint8`` planes with values 0/255 (the callers check
"8-bit binary image required"). Which value is the *foreground* follows ImageJ's rule (see
:func:`foreground_value`): with "Black background" off, objects are black, i.e. 0 with a
normal LUT and 255 with an inverting LUT; with it on, objects are white.
"""
from __future__ import annotations

import math

import numpy as np

# EDM output types (EDM.java)
OUTPUT_TYPES = ("Overwrite", "8-bit", "16-bit", "32-bit")
OPERATIONS = ("Nothing", "Erode", "Dilate", "Open", "Close", "Outline", "Fill Holes",
              "Skeletonize")
MAXFINDER_TOLERANCE = 0.5      # EDM.java: tolerance of the MaximumFinder for EDM operations
_FLOAT_MAX = float(np.finfo(np.float32).max)
_INT_MAX = 2147483647


# ---------------------------------------------------------------------------------- basics
def foreground_value(inverted_lut: bool, black_background: bool) -> int:
    """Binary.run: ``fg = blackBackground ? 255 : 0``, flipped for an inverting LUT."""
    fg = 255 if black_background else 0
    return 255 - fg if inverted_lut else fg


def is_binary(plane: np.ndarray) -> bool:
    """ByteProcessor.isBinary: only 0 and 255."""
    a = np.asarray(plane)
    return a.dtype == np.uint8 and bool(np.all((a == 0) | (a == 255)))


def binary_threshold(inverted_lut: bool, black_background: bool) -> float:
    """ImageProcessor.setBinaryThreshold: the invisible threshold level (lower = upper) set
    after a binary operation (255, or 0 when objects are 0)."""
    if (inverted_lut and black_background) or (not inverted_lut and not black_background):
        return 0.0
    return 255.0


def _neighbor_count(mask: np.ndarray, pad_value: bool) -> np.ndarray:
    """Number of the 8 neighbours where ``mask`` is True (out-of-image = ``pad_value``)."""
    p = np.pad(mask, 1, constant_values=pad_value).astype(np.uint8)
    h, w = mask.shape
    n = np.zeros((h, w), np.uint8)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            if dy == 1 and dx == 1:
                continue
            n += p[dy:dy + h, dx:dx + w]
    return n


def erode(plane: np.ndarray, count: int = 1, background: int = 255, iterations: int = 1,
          pad_edges: bool = False) -> np.ndarray:
    """ByteProcessor.erode(count, background), ``iterations`` times: a foreground pixel
    becomes background when at least ``count`` of its 8 neighbours are background. Outside
    the image counts as background, or as foreground with ``pad_edges``."""
    a = np.array(plane, dtype=np.uint8)
    bg = int(background)
    fg = 255 - bg
    for _ in range(int(iterations)):
        bgm = a == bg
        n = _neighbor_count(bgm, not pad_edges)
        a = np.where(bgm | (n >= count), bg, fg).astype(np.uint8)
    return a


def dilate(plane: np.ndarray, count: int = 1, background: int = 255, iterations: int = 1
           ) -> np.ndarray:
    """ByteProcessor.dilate(count, background), ``iterations`` times: a background pixel
    becomes foreground when at least ``count`` neighbours are foreground (outside = bg)."""
    a = np.array(plane, dtype=np.uint8)
    bg = int(background)
    fg = 255 - bg
    for _ in range(int(iterations)):
        fgm = a == fg
        n = _neighbor_count(fgm, False)
        a = np.where(fgm | (n >= count), fg, bg).astype(np.uint8)
    return a


def binary_open(plane, count=1, background=255, iterations=1, pad_edges=False):
    """Binary "Open": erode then dilate (each ``iterations`` times)."""
    return dilate(erode(plane, count, background, iterations, pad_edges), count, background,
                  iterations)


def binary_close(plane, count=1, background=255, iterations=1, pad_edges=False):
    """Binary "Close-": dilate then erode."""
    return erode(dilate(plane, count, background, iterations), count, background, iterations,
                 pad_edges)


def outline(plane: np.ndarray, inverted_lut: bool = False, black_background: bool = False
            ) -> np.ndarray:
    """Binary.outline / BinaryProcessor OUTLINE: pixels that are not background (value 255,
    or 0 with an inverting LUT) and have no background among their 8 neighbours become
    background; the image border rows/columns are kept.

    With "Black background" ImageJ inverts the image first, but BinaryProcessor reads its
    neighbourhoods from the undo snapshot (``setSnapshotCopyMode``), i.e. from the
    *uninverted* pixels, and the result is inverted back. The net effect, reproduced here, is
    that the outline is drawn on the background pixels touching the objects."""
    a = np.array(plane, dtype=np.uint8)
    bg = 0 if inverted_lut else 255
    h, w = a.shape
    out = a.copy()
    if h >= 3 and w >= 3:
        isbg = a == bg
        n = _neighbor_count(isbg, False)
        v = np.where(~isbg & (n == 0), bg, a).astype(np.uint8)
        if black_background:
            v = 255 - v
        out[1:-1, 1:-1] = v[1:-1, 1:-1]
    return out


def fill_holes(plane: np.ndarray, foreground: int, background: int) -> np.ndarray:
    """Binary.fill (G. Landini): background connected (4-connected) to the image border stays
    background, everything else becomes foreground."""
    from scipy import ndimage

    a = np.asarray(plane)
    bgm = a == background
    labels, n = ndimage.label(bgm, structure=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    border = border[border > 0]
    outside = np.isin(labels, border)
    return np.where(outside, background, foreground).astype(np.uint8)


# skeletonize lookup tables (BinaryProcessor.java, incl. the 2012/2013 corrections)
_TABLE = np.array([
    0, 0, 0, 0, 0, 0, 1, 3, 0, 0, 3, 1, 1, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 3, 0, 3, 3,
    0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 2, 2,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 2, 0,
    0, 0, 3, 1, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 3, 1, 3, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0],
    dtype=np.uint8)
_TABLE2 = np.array([
    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 2, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    dtype=np.uint8)
# neighbour bit weights of BinaryProcessor.thin: p1=1 p2=2 p3=4 p6=8 p9=16 p8=32 p7=64 p4=128
_THIN_BITS = ((-1, -1, 1), (-1, 0, 2), (-1, 1, 4), (0, 1, 8), (1, 1, 16), (1, 0, 32),
              (1, -1, 64), (0, -1, 128))


def _thin(fg: np.ndarray, pass_: int, table: np.ndarray) -> int:
    """One BinaryProcessor.thin pass on the bool image ``fg`` (in place, interior only)."""
    h, w = fg.shape
    if h < 3 or w < 3:
        return 0
    c = fg[1:-1, 1:-1]
    index = np.zeros(c.shape, np.int32)
    for dy, dx, bit in _THIN_BITS:
        index |= fg[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx].astype(np.int32) * bit
    code = table[index]
    if pass_ & 1:
        remove = c & ((code == 2) | (code == 3))
    else:
        remove = c & ((code == 1) | (code == 3))
    n = int(remove.sum())
    if n:
        c[remove] = False
    return n


def skeletonize(plane: np.ndarray, foreground: int) -> np.ndarray:
    """BinaryProcessor.skeletonize(foreground): ImageJ's table-driven thinning."""
    if foreground not in (0, 255):
        raise ValueError("Skeletonize: foreground must be 255 or 0")
    a = np.asarray(plane, dtype=np.uint8)
    fgm = a == foreground
    edge = bool(fgm[0].any() or fgm[-1].any() or fgm[:, 0].any() or fgm[:, -1].any())
    work = np.pad(fgm, 1, constant_values=False) if edge else fgm.copy()
    # skeletonize2 draws background lines along the image border
    work[0, :] = work[-1, :] = False
    work[:, 0] = work[:, -1] = False
    pass_ = 0
    while True:
        removed = _thin(work, pass_, _TABLE)
        pass_ += 1
        removed += _thin(work, pass_, _TABLE)
        pass_ += 1
        if removed <= 0:
            break
    while True:
        removed = _thin(work, pass_, _TABLE2)
        pass_ += 1
        removed += _thin(work, pass_, _TABLE2)
        pass_ += 1
        if removed <= 0:
            break
    if edge:
        work = work[1:-1, 1:-1]
    bg = 255 - foreground
    return np.where(work, foreground, bg).astype(np.uint8)


# ---------------------------------------------------------------------------------- rank mean
def line_radii(radius: float) -> list[tuple[int, int]]:
    """RankFilters.makeLineRadii: [(dy, dx)] of the circular kernel rows (x from -dx to dx)."""
    if 1.5 <= radius < 1.75:
        radius = 1.75
    elif 2.5 <= radius < 2.85:
        radius = 2.85
    r2 = int(radius * radius) + 1
    k = int(math.sqrt(r2 + 1e-10))
    rows = []
    for dy in range(-k, k + 1):
        dx = k if dy == 0 else int(math.sqrt(r2 - dy * dy + 1e-10))
        rows.append((dy, dx))
    return rows


def _circular_sum(a: np.ndarray, radius: float) -> tuple[np.ndarray, int]:
    """Sum over ImageJ's circular kernel with edge pixels repeated (RankFilters padding)."""
    rows = line_radii(radius)
    k = max(abs(dy) for dy, _ in rows)
    h, w = a.shape
    p = np.pad(a.astype(np.int64), k, mode="edge")
    cs = np.zeros((p.shape[0], p.shape[1] + 1), np.int64)
    np.cumsum(p, axis=1, out=cs[:, 1:])
    total = np.zeros((h, w), np.int64)
    n = 0
    xs = np.arange(w) + k
    for dy, dx in rows:
        r = cs[k + dy:k + dy + h]
        total += r[:, xs + dx + 1] - r[:, xs - dx]
        n += 2 * dx + 1
    return total, n


def binary_median(plane: np.ndarray, radius: float = 3.0) -> np.ndarray:
    """Binary "Median": RankFilters MEAN with ``radius`` then ``threshold(128, 255)`` (ImageJ
    1.54 implements this command as a circular mean followed by a 50% threshold)."""
    total, n = _circular_sum(np.asarray(plane), radius)
    # 8-bit mean rounded = (int)(sum/n + 0.5) >= 128  <=>  2*sum >= 255*n (n is odd)
    return np.where(2 * total >= 255 * n, 255, 0).astype(np.uint8)


# ---------------------------------------------------------------------------------- EDM
def float_edm(plane: np.ndarray, background_value: int = 0, edges_are_background: bool = False
              ) -> np.ndarray:
    """EDM.makeFloatEDM: distance of every non-background pixel to the nearest background
    pixel (ImageJ's 8SSEDT-like two-pass algorithm, line-by-line port; exact for d < 13,
    and identical to ImageJ beyond). Returns float32."""
    a = np.asarray(plane, dtype=np.uint8)
    h, w = a.shape
    bgv = int(background_value) & 0xFF
    bg_rows = [row.tolist() for row in (a == bgv)]
    big = _FLOAT_MAX
    f = [[0.0 if b else big for b in row] for row in bg_rows]
    NO = -1
    for direction in (1, -1):
        buf0 = [NO] * w
        buf1 = [NO] * w
        ys = range(h) if direction == 1 else range(h - 1, -1, -1)
        for y in ys:
            if edges_are_background:
                ydist = y + 1 if direction == 1 else h - y
            else:
                ydist = _INT_MAX
            _edm_line(bg_rows[y], f[y], buf0, buf1, w, y, ydist)
    d2 = np.array(f, dtype=np.float64)
    out = np.where(d2 <= 0, 0.0, np.sqrt(np.minimum(d2, big)))
    return out.astype(np.float32)


def _edm_line(bg, frow, points0, points1, width, y, ydist):
    """EDM.edmLine with minDist2 inlined (left-to-right, then right-to-left)."""
    NO = -1
    edges = ydist != _INT_MAX
    yshift = y << 16
    for pts, xs in ((points0, range(width)), (points1, range(width - 1, -1, -1))):
        p_prev = NO
        p_diag = NO
        for x in xs:
            p_next_diag = pts[x]
            if bg[x]:
                pts[x] = x | yshift
            else:
                if edges:
                    e = (x + 1) if pts is points0 else (width - x)
                    dist = e * e if e < ydist else ydist * ydist
                else:
                    dist = _INT_MAX
                p0 = pts[x]
                nearest = p0
                if p0 != NO:
                    dx = x - (p0 & 0xFFFF)
                    dy = y - (p0 >> 16)
                    d = dx * dx + dy * dy
                    if d < dist:
                        dist = d
                if p_diag != p0 and p_diag != NO:
                    dx = x - (p_diag & 0xFFFF)
                    dy = y - (p_diag >> 16)
                    d = dx * dx + dy * dy
                    if d < dist:
                        nearest = p_diag
                        dist = d
                if p_prev != p_diag and p_prev != NO:
                    dx = x - (p_prev & 0xFFFF)
                    dy = y - (p_prev >> 16)
                    d = dx * dx + dy * dy
                    if d < dist:
                        nearest = p_prev
                        dist = d
                pts[x] = nearest
                fd = float(dist) if dist < 16777216 else float(np.float32(dist))
                if frow[x] > fd:
                    frow[x] = fd
            p_prev = pts[x]
            p_diag = p_next_diag


def edm_background_is_255(inverted_lut: bool, black_background: bool) -> bool:
    """EDM.setup ``background255``: whether 255 is the background of the binary image."""
    return (inverted_lut and black_background) or (not inverted_lut and not black_background)


def edm_to_byte_overwrite(edm: np.ndarray) -> np.ndarray:
    """ByteProcessor.setPixels(0, fp): (int)(v + 0.5f) clamped to 0..255 (float arithmetic)."""
    v = edm.astype(np.float32) + np.float32(0.5)
    v = np.clip(v, np.float32(0), np.float32(255))
    return v.astype(np.uint8)


def convert_edm(edm: np.ndarray, output: str) -> np.ndarray:
    """EDM output conversions: "8-bit" (setMinAndMax 0-255 + convertToByte(true)), "16-bit"
    (0-65535 + convertToShort(true)), "32-bit" (the float EDM), "Overwrite" (byte setPixels)."""
    if output == "32-bit":
        return edm.astype(np.float32)
    if output == "Overwrite":
        return edm_to_byte_overwrite(edm)
    v = np.maximum(edm.astype(np.float64), 0.0)
    if output == "8-bit":
        return np.minimum((v * 1.0 + 0.5).astype(np.int64), 255).astype(np.uint8)
    if output == "16-bit":
        v = np.minimum(v * 1.0, 65535.0)
        return (v + 0.5).astype(np.int64).astype(np.uint16)
    raise ValueError(f"unknown EDM output {output!r}")


def distance_map(plane: np.ndarray, inverted_lut: bool, black_background: bool,
                 output: str = "Overwrite") -> np.ndarray:
    """Process > Binary > Distance Map (converted to ``output``)."""
    bg255 = edm_background_is_255(inverted_lut, black_background)
    edm = float_edm(plane, 255 if bg255 else 0)
    return convert_edm(edm, output)


def ultimate_points_edm(plane: np.ndarray, inverted_lut: bool, black_background: bool
                        ) -> np.ndarray:
    """Float EDM values at the ultimate eroded points (maxima of the EDM), 0 elsewhere."""
    from rsstatchem.core.imaging import maxima as M

    bg255 = edm_background_is_255(inverted_lut, black_background)
    edm = float_edm(plane, 255 if bg255 else 0)
    res = M.find_maxima(edm, MAXFINDER_TOLERANCE, output=M.SINGLE_POINTS, is_edm=True)
    edm[res.image == 0] = 0
    return edm


def ultimate_points(plane, inverted_lut, black_background, output="Overwrite"):
    """Process > Binary > Ultimate Points (converted to ``output``)."""
    return convert_edm(ultimate_points_edm(plane, inverted_lut, black_background), output)


def watershed(plane: np.ndarray, inverted_lut: bool, black_background: bool) -> np.ndarray:
    """Process > Binary > Watershed: EDM-based segmentation; separating lines are background."""
    from rsstatchem.core.imaging import maxima as M

    bg255 = edm_background_is_255(inverted_lut, black_background)
    edm = float_edm(plane, 255 if bg255 else 0)
    res = M.find_maxima(edm, MAXFINDER_TOLERANCE, output=M.SEGMENTED, is_edm=True)
    out = res.image
    if bg255:
        out = 255 - out
    return out.astype(np.uint8)


def voronoi_edm(plane: np.ndarray, inverted_lut: bool, black_background: bool) -> np.ndarray:
    """Float Voronoi: EDM of the background on the lines between particle zones, 0 inside."""
    from rsstatchem.core.imaging import maxima as M

    bg255 = edm_background_is_255(inverted_lut, black_background)
    edm = float_edm(plane, 0 if bg255 else 255)       # EDM of the background
    neg = (edm * np.float32(-1)).astype(np.float32)
    res = M.find_maxima(neg, MAXFINDER_TOLERANCE, output=M.SEGMENTED, is_edm=False,
                        float_image=True)
    edm[res.image == 255] = 0
    return edm


def voronoi(plane, inverted_lut, black_background, output="Overwrite"):
    """Process > Binary > Voronoi (converted to ``output``)."""
    return convert_edm(voronoi_edm(plane, inverted_lut, black_background), output)

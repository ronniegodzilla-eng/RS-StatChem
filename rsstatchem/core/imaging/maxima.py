"""Process > Find Maxima and the EDM-based segmentation (watershed) used by Process > Binary.

Ported from ImageJ 1.54p ij/plugin/filter/MaximumFinder.java (public domain, NIH; Michael
Schmid, W. Rasband et al.). The flood-fill parts are a line-by-line port (float32 comparisons
as in Java); the per-pixel neighbourhood tests, the 8-bit conversion and the watershed level
expansion are vectorised with numpy without changing the results (golden case
``process_binary``).

:func:`find_maxima` works on a 2-D float32 array of (calibrated) pixel values; callers that
look for minima (light background, or dark background with an inverting LUT) pass the
inverted image, see :func:`values_for_maxima`.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

SINGLE_POINTS, IN_TOLERANCE, SEGMENTED, POINT_SELECTION, LIST, COUNT = range(6)
OUTPUT_TYPE_NAMES = ("Single Points", "Maxima Within Tolerance", "Segmented Particles",
                     "Point Selection", "List", "Count")

MAXIMUM, LISTED, PROCESSED, MAX_AREA, EQUAL, MAX_POINT, ELIMINATED = 1, 2, 4, 8, 16, 32, 64
_OUTPUT_MASKS = (MAX_POINT, MAX_AREA, MAX_AREA)
SQRT2 = np.float32(1.4142135624)
DIR_X = (0, 1, 1, 1, 0, -1, -1, -1)
DIR_Y = (-1, -1, 0, 1, 1, 1, 0, -1)
_F32 = np.float32


class MaximaResult(NamedTuple):
    points: list            # [(x, y)] for POINT_SELECTION / LIST / COUNT (else [])
    image: np.ndarray | None  # uint8 0/255 (normal LUT) for SINGLE_POINTS/IN_TOLERANCE/SEGMENTED


def _f(v) -> float:
    """Round a Python float to float32 (Java float arithmetic)."""
    return float(_F32(v))


# ---------------------------------------------------------------------------------- 1-D
def find_maxima_1d(xx, tolerance: float, edge_mode: int = 0) -> list[int]:
    """MaximumFinder.findMaxima(double[], tolerance, edgeMode) (N. Vischer): positions of
    peaks sorted by decreasing amplitude; edge_mode 0 = include, 1 = exclude, 2 = circular."""
    xx = [float(v) for v in xx]
    orig_len = length = len(xx)
    if length < 2:
        return []
    tolerance = max(tolerance, 0.0)
    if edge_mode == 2:
        xx = xx * 3
        length *= 3
    max_positions = []
    j_start = 0
    while True:
        vmax = vmin = xx[j_start]
        j_start += 1
        if j_start >= len(xx):
            return []
        if not math.isnan(vmin):
            break
    max_pos = j_start - 1
    last_max_pos = -1
    left_valley = edge_mode == 0
    for jj in range(j_start, length):
        val = xx[jj]
        if val > vmin + tolerance:
            left_valley = True
        if val > vmax and left_valley:
            vmax = val
            max_pos = jj
        if left_valley:
            last_max_pos = max_pos
        if val < vmax - tolerance and left_valley:
            max_positions.append(max_pos)
            left_valley = False
            vmin = vmax = val
        if val < vmin:
            vmin = val
            if not left_valley:
                vmax = val
    if edge_mode == 0:
        if max_positions and max_positions[-1] != last_max_pos:
            max_positions.append(last_max_pos)
        if not max_positions and vmax - vmin >= tolerance:
            max_positions.append(last_max_pos)
    values = []
    for k, pos in enumerate(max_positions):
        mid = float(pos)
        while pos < length - 1 and xx[pos] == xx[pos + 1]:
            mid += 0.5
            pos += 1
        max_positions[k] = int(mid)
        values.append(xx[max_positions[k]])
    order = sorted(range(len(values)), key=lambda i: values[i])   # Tools.rank (stable)
    out = [max_positions[i] for i in reversed(order)]
    if edge_mode == 2:
        out = [p - orig_len for p in out if 0 <= p - orig_len < orig_len]
    return out


def find_minima_1d(xx, tolerance: float, edge_mode: int = 0) -> list[int]:
    return find_maxima_1d([-float(v) for v in xx], tolerance, edge_mode)


# ---------------------------------------------------------------------------------- helpers
def values_for_maxima(plane: np.ndarray, find_minima: bool, display_range=None,
                      ctable=None) -> np.ndarray:
    """Float32 values ``ip.getPixelValue`` sees in MaximumFinder.run, inverted (ip.invert())
    when minima are wanted: 8-bit/RGB 255-v, 16-bit max-(v-min) with the data range, 32-bit
    max-(v-min) in float with the display range; a calibration table is negated instead."""
    a = np.asarray(plane)
    if a.ndim == 3:   # RGB: ColorProcessor.getPixelValue = (r+g+b)/3
        rgb = a.astype(np.float64)
        if find_minima:
            rgb = 255.0 - rgb
        return ((rgb[..., 0] + rgb[..., 1] + rgb[..., 2]) / 3.0).astype(np.float32)
    if ctable is not None and a.dtype != np.float32:
        v = np.asarray(ctable, dtype=np.float32)[a]
        return (-v).astype(np.float32) if find_minima else v
    if not find_minima:
        return a.astype(np.float32)
    if a.dtype == np.uint8:
        return (255 - a.astype(np.int32)).astype(np.float32)
    if a.dtype == np.uint16:
        lo, hi = int(a.min()), int(a.max())
        return (hi - (a.astype(np.int64) - lo)).astype(np.float32)
    lo, hi = display_range if display_range is not None else (np.nanmin(a), np.nanmax(a))
    lo, hi = _F32(lo), _F32(hi)
    f = a.astype(np.float32)
    return (hi - (f - lo)).astype(np.float32)


def _shift(a: np.ndarray, d: int, fill):
    """Neighbour in direction d (value at (x+DIR_X[d], y+DIR_Y[d])); out of image = fill."""
    h, w = a.shape
    out = np.full_like(a, fill)
    dx, dy = DIR_X[d], DIR_Y[d]
    ys0, ys1 = max(0, -dy), min(h, h - dy)
    xs0, xs1 = max(0, -dx), min(w, w - dx)
    out[ys0:ys1, xs0:xs1] = a[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
    return out


def true_edm_height(edm: np.ndarray) -> np.ndarray:
    """MaximumFinder.trueEdmHeight for every pixel (interpolated ridge height of an EDM)."""
    v = edm.astype(np.float32)
    h, w = v.shape
    out = v.copy()
    if h < 3 or w < 3:
        return out
    true_h = v + _F32(0.5) * SQRT2
    ridge = np.zeros(v.shape, bool)
    for d in range(4):
        v1 = _shift(v, d, _F32(0))
        v2 = _shift(v, d + 4, _F32(0))
        is_r = (v >= v1) & (v >= v2)
        ridge |= is_r
        hh = np.where(is_r, (v1 + v2) / _F32(2), np.minimum(v1, v2)).astype(np.float32)
        hh = hh + (_F32(1) if d % 2 == 0 else SQRT2)
        true_h = np.minimum(true_h, hh)
    inner = np.zeros(v.shape, bool)
    inner[1:-1, 1:-1] = True
    sel = inner & (v != 0) & ridge
    out[sel] = true_h[sel]
    return out


# ---------------------------------------------------------------------------------- main
def find_maxima(values: np.ndarray, tolerance: float, strict: bool = False,
                threshold: float | None = None, output: int = SINGLE_POINTS,
                exclude_on_edges: bool = False, is_edm: bool = False,
                roi_rect: tuple[int, int, int, int] | None = None,
                roi_mask: np.ndarray | None = None, float_image: bool = False
                ) -> MaximaResult:
    """MaximumFinder.findMaxima(ip, tolerance, strict, threshold, outputType, excludeOnEdges,
    isEDM).

    ``values``: 2-D float32 pixel values (already inverted for minima). ``threshold``: lower
    threshold (calibrated) or None. ``roi_rect`` (x, y, w, h) / ``roi_mask`` (bool, full image
    size) restrict the search like an area selection; ``float_image``: the source is a
    32-bit image (ImageJ then allows for sorting inaccuracy). Returns points (list output types) or
    the uint8 output image (255 = marked, 0 elsewhere)."""
    v = np.ascontiguousarray(values, dtype=np.float32)
    height, width = v.shape
    rx, ry, rw, rh = roi_rect if roi_rect is not None else (0, 0, width, height)
    in_rect = np.zeros((height, width), bool)
    in_rect[ry:ry + rh, rx:rx + rw] = True
    sub = v[ry:ry + rh, rx:rx + rw]
    gmin = _F32(sub.min()) if sub.size else _F32(0)
    gmax = _F32(sub.max()) if sub.size else _F32(0)
    maximum_possible = bool(gmax > gmin)
    if strict and float(gmax - gmin) <= tolerance:
        maximum_possible = False
    if threshold is not None:
        threshold = threshold - float(gmax - gmin) * 1e-6
    exclude_now = exclude_on_edges and output != SEGMENTED

    vtrue = true_edm_height(v) if is_edm else v
    types = np.zeros((height, width), np.uint8)
    if maximum_possible:
        max_offsets = _sorted_max_points(v, vtrue, types, in_rect, exclude_now, is_edm, gmin,
                                         gmax, threshold)
    else:
        max_offsets = np.zeros(0, np.int64)
    max_sorting_error = 0.0
    if is_edm or float_image:     # only for FloatProcessor input (sorting by 32-bit ints)
        max_sorting_error = _f(_F32(1.1) * (SQRT2 / _F32(2) if is_edm else
                                             (gmax - gmin) / _F32(2e9)))
    display_or_count = output in (POINT_SELECTION, LIST, COUNT)
    contains = roi_mask if roi_mask is not None else (in_rect if roi_rect is not None else None)
    flat_types = bytearray(types.tobytes())
    points = _analyze_and_mark(v, vtrue, flat_types, max_offsets, width, height, exclude_now,
                               is_edm, tolerance, strict, display_or_count, exclude_on_edges,
                               contains, max_sorting_error)
    types = np.frombuffer(bytes(flat_types), np.uint8).reshape(height, width).copy()
    if display_or_count:
        return MaximaResult(points, None)
    if output == SEGMENTED:
        out = _make8bit(v, types, is_edm, gmin, gmax, threshold)
        _cleanup_maxima(out, types, max_offsets)
        _watershed_segment(out)
        if not is_edm:
            _cleanup_extra_lines(out)
        out[out < 255] = 0
        if exclude_on_edges:
            _delete_edge_particles(out, types)
    else:
        out = np.where((types & _OUTPUT_MASKS[output]) != 0, 255, 0).astype(np.uint8)
    if roi_rect is not None or roi_mask is not None:
        keep = in_rect if roi_mask is None else (in_rect & roi_mask)
        out[~keep] = 0
    return MaximaResult([], out)


def _sorted_max_points(v, vtrue, types, in_rect, exclude_now, is_edm, gmin, gmax, threshold):
    height, width = v.shape
    cand = in_rect & (v != gmin)
    if exclude_now:
        cand[0, :] = cand[-1, :] = False
        cand[:, 0] = cand[:, -1] = False
    if threshold is not None:
        cand &= ~(v.astype(np.float64) < threshold)
    is_max = cand.copy()
    ninf = _F32(-np.inf)
    for d in range(8):
        vn = _shift(v, d, ninf)
        if is_edm:
            vnt = _shift(vtrue, d, ninf)
            is_max &= ~((vn > v) & (vnt > vtrue))
        else:
            is_max &= ~(vn > v)
    types[is_max] = MAXIMUM
    ys, xs = np.nonzero(is_max)
    if len(ys) == 0:
        return np.zeros(0, np.int64)
    vfactor = _F32(2e9 / float(gmax - gmin))
    fval = vtrue[ys, xs] if is_edm else v[ys, xs]
    prod = ((fval - gmin).astype(np.float32) * vfactor).astype(np.float32)
    ival = np.clip(np.trunc(prod.astype(np.float64)), -2147483648, 2147483647).astype(np.int64)
    offs = (ys.astype(np.int64) * width + xs)
    order = np.lexsort((offs, ival))
    return offs[order]


def _analyze_and_mark(v, vtrue, types, max_offsets, width, height, exclude_now, is_edm,
                      tolerance, strict, display_or_count, exclude_on_edges, contains,
                      max_sorting_error):
    """analyzeAndMarkMaxima: flood within the prominence from each maximum (highest first)."""
    tv = vtrue.ravel().tolist()
    raw = v.ravel().tolist() if is_edm else None
    dir_off = (-width, -width + 1, 1, width + 1, width, width - 1, -1, -width - 1)
    xmax, ymax = width - 1, height - 1
    tol32 = _f(tolerance)
    p_list = [0] * (width * height)
    points = []
    nmax = len(max_offsets)
    for i_max in range(nmax - 1, -1, -1):
        offset0 = int(max_offsets[i_max])
        if types[offset0] & PROCESSED:
            continue
        x0, y0 = offset0 % width, offset0 // width
        v0 = tv[offset0]
        while True:
            p_list[0] = offset0
            types[offset0] |= (EQUAL | LISTED)
            list_len = 1
            list_i = 0
            is_edge_max = x0 == 0 or x0 == xmax or y0 == 0 or y0 == ymax
            sorting_error = False
            max_possible = True
            x_equal, y_equal, n_equal = float(x0), float(y0), 1
            hi = _f(v0 + max_sorting_error)
            lo = _f(v0 - tol32)
            while True:
                offset = p_list[list_i]
                x, y = offset % width, offset // width
                inner = 0 < y < ymax and 0 < x < xmax
                for d in range(8):
                    if not inner:
                        if d == 0:
                            ok = y > 0
                        elif d == 1:
                            ok = x < xmax and y > 0
                        elif d == 2:
                            ok = x < xmax
                        elif d == 3:
                            ok = x < xmax and y < ymax
                        elif d == 4:
                            ok = y < ymax
                        elif d == 5:
                            ok = x > 0 and y < ymax
                        elif d == 6:
                            ok = x > 0
                        else:
                            ok = x > 0 and y > 0
                        if not ok:
                            continue
                    offset2 = offset + dir_off[d]
                    t2 = types[offset2]
                    if t2 & LISTED:
                        continue
                    if is_edm and raw[offset2] <= 0:
                        continue
                    if t2 & PROCESSED:
                        max_possible = False
                        break
                    v2 = tv[offset2]
                    if v2 > hi:
                        max_possible = False
                        break
                    elif v2 >= lo:
                        x2, y2 = x + DIR_X[d], y + DIR_Y[d]
                        if v2 > v0:
                            sorting_error = True
                            offset0, v0, x0, y0 = offset2, v2, x2, y2
                            hi = _f(v0 + max_sorting_error)
                            lo = _f(v0 - tol32)
                        p_list[list_len] = offset2
                        list_len += 1
                        types[offset2] = t2 | LISTED
                        if (x2 == 0 or x2 == xmax or y2 == 0 or y2 == ymax) and \
                                (strict or v2 >= v0):
                            is_edge_max = True
                            if exclude_now:
                                max_possible = False
                                break
                        if v2 == v0:
                            types[offset2] |= EQUAL
                            x_equal += x2
                            y_equal += y2
                            n_equal += 1
                list_i += 1
                if list_i >= list_len:
                    break
            if sorting_error:
                for k in range(list_len):
                    types[p_list[k]] = 0
                continue
            reset_mask = 0xFF & ~(LISTED if max_possible else (LISTED | EQUAL))
            x_equal /= n_equal
            y_equal /= n_equal
            min_dist2 = 1e20
            nearest = 0
            for k in range(list_len):
                off = p_list[k]
                t = (types[off] & reset_mask) | PROCESSED
                if max_possible:
                    t |= MAX_AREA
                    if t & EQUAL:
                        x, y = off % width, off // width
                        dist2 = (x_equal - x) * (x_equal - x) + (y_equal - y) * (y_equal - y)
                        if dist2 < min_dist2:
                            min_dist2 = dist2
                            nearest = k
                types[off] = t
            if max_possible:
                off = p_list[nearest]
                types[off] |= MAX_POINT
                if display_or_count and not (exclude_on_edges and is_edge_max):
                    x, y = off % width, off // width
                    if contains is None or contains[y, x]:
                        points.append((x, y))
            break
    if nmax == 0:
        types[:] = bytes([PROCESSED | MAX_AREA]) * len(types)
    return points


def _make8bit(v, types, is_edm, gmin, gmax, threshold):
    """MaximumFinder.make8bit: scale to 1..254, maximum areas 255, below threshold 0."""
    if is_edm:
        threshold = 0.5
        min_value = 1.0
    else:
        min_value = float(gmin) if threshold is None else threshold
    gmax = float(gmax)
    offset = min_value - (gmax - min_value) * (1.0 / 253 / 2 - 1e-6)
    factor = 253 / (gmax - min_value) if gmax != min_value else math.inf
    if is_edm and factor > 1:
        factor = 1.0
    raw = v.astype(np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        vv = 1 + np.floor((raw - offset) * factor + 0.5)
    out = np.clip(np.nan_to_num(vv, nan=1.0, posinf=254, neginf=1), 1, 254).astype(np.uint8)
    out[(types & MAX_AREA) != 0] = 255
    if threshold is not None:
        out[raw < threshold] = 0
    return out


def _cleanup_maxima(out, types, max_offsets):
    """cleanupMaxima: flatten maxima that did not qualify down to their saddle level."""
    height, width = out.shape
    pix = bytearray(out.tobytes())
    typ = bytearray(types.tobytes())
    dir_off = (-width, -width + 1, 1, width + 1, width, width - 1, -1, -width - 1)
    xmax, ymax = width - 1, height - 1
    p_list = [0] * (width * height)
    for i_max in range(len(max_offsets) - 1, -1, -1):
        offset0 = int(max_offsets[i_max])
        if typ[offset0] & (MAX_AREA | ELIMINATED):
            continue
        level = pix[offset0]
        lo_level = level + 1
        p_list[0] = offset0
        typ[offset0] |= LISTED
        list_len = 1
        last_len = 1
        saddle = False
        while not saddle and lo_level > 0:
            lo_level -= 1
            last_len = list_len
            list_i = 0
            while True:
                offset = p_list[list_i]
                x, y = offset % width, offset // width
                inner = 0 < y < ymax and 0 < x < xmax
                for d in range(8):
                    if not inner and not _within(x, y, d, xmax, ymax):
                        continue
                    offset2 = offset + dir_off[d]
                    t2 = typ[offset2]
                    if t2 & LISTED:
                        continue
                    p2 = pix[offset2]
                    if (t2 & MAX_AREA) or ((t2 & ELIMINATED) and p2 >= lo_level):
                        saddle = True
                        break
                    elif p2 >= lo_level and not (t2 & ELIMINATED):
                        p_list[list_len] = offset2
                        list_len += 1
                        typ[offset2] = t2 | LISTED
                if saddle:
                    break
                list_i += 1
                if list_i >= list_len:
                    break
        for k in range(list_len):
            typ[p_list[k]] &= 0xFF & ~LISTED
        for k in range(last_len):
            off = p_list[k]
            pix[off] = lo_level
            typ[off] |= ELIMINATED
    out[...] = np.frombuffer(bytes(pix), np.uint8).reshape(out.shape)
    types[...] = np.frombuffer(bytes(typ), np.uint8).reshape(types.shape)


def _within(x, y, d, xmax, ymax) -> bool:
    if d == 0:
        return y > 0
    if d == 1:
        return x < xmax and y > 0
    if d == 2:
        return x < xmax
    if d == 3:
        return x < xmax and y < ymax
    if d == 4:
        return y < ymax
    if d == 5:
        return x > 0 and y < ymax
    if d == 6:
        return x > 0
    return x > 0 and y > 0


def _fate_table() -> np.ndarray:
    """MaximumFinder.makeFateTable."""
    table = np.zeros(256, np.int64)
    for item in range(256):
        is_set = [(item >> i) & 1 == 1 for i in range(8)]
        t = 0
        for i in range(8):
            if is_set[(i + 4) % 8]:
                t |= 1 << i
        for i in range(0, 8, 2):
            if is_set[i]:
                is_set[(i + 1) % 8] = True
                is_set[(i + 7) % 8] = True
        transitions = sum(1 for i in range(8) if is_set[i] != is_set[(i + 1) % 8])
        table[item] = 0 if transitions >= 4 else t
    return table


_FATE = _fate_table()
# neighbour bits of processLevel: (dx, dy, bit)
_LEVEL_BITS = ((0, -1, 1), (1, -1, 2), (1, 0, 4), (1, 1, 8), (0, 1, 16), (-1, 1, 32),
               (-1, 0, 64), (-1, -1, 128))


def _watershed_segment(out: np.ndarray) -> None:
    """watershedSegment: dilate the 255 seeds level by level (fate table), in place."""
    height, width = out.shape
    pix = out.ravel()   # view
    hist = np.bincount(pix, minlength=256).astype(np.int64)
    shift = 0
    mult = 1
    while True:
        shift += 1
        mult *= 2
        if mult >= width:
            break
    xmask = mult - 1
    level_start = np.zeros(256, np.int64)
    offset = 0
    highest = 0
    for lv in range(1, 255):
        level_start[lv] = offset
        offset += hist[lv]
        if hist[lv] > 0:
            highest = lv
    mid = (pix > 0) & (pix < 255)
    idx = np.nonzero(mid)[0]
    order = np.argsort(pix[idx], kind="stable")
    idx = idx[order]
    ys, xs = idx // width, idx % width
    coords = np.zeros(max(len(idx), 1) + 0, np.int64)
    coords[:len(idx)] = xs | (ys << shift)
    seq = (7, 3, 1, 5, 0, 4, 2, 6)
    for level in range(highest, 0, -1):
        remaining = int(hist[level])
        idle = 0
        while remaining > 0 and idle < 8:
            d_index = 0
            while True:
                n = _process_level(seq[d_index % 8], pix, width, height, int(level_start[level]),
                                   remaining, coords, xmask, shift)
                remaining -= n
                if n > 0:
                    idle = 0
                d_index += 1
                cont = remaining > 0 and idle < 8
                idle += 1
                if not cont:
                    break
        if remaining > 0 and level > 1:
            next_level = level
            while True:
                next_level -= 1
                if not (next_level > 1 and hist[next_level] == 0):
                    break
            if next_level > 0:
                ls = int(level_start[level])
                pts = coords[ls:ls + remaining].copy()
                x = pts & xmask
                y = pts >> shift
                off = x + y * width
                add = (x == 0) | (y == 0) | (x == width - 1) | (y == height - 1)
                inner = ~add
                if inner.any():
                    oi = off[inner]
                    zero = np.zeros(len(oi), bool)
                    for dx, dy, _ in _LEVEL_BITS:
                        zero |= pix[oi + dx + dy * width] == 0
                    add[inner] = zero
                sel = pts[add]
                start = int(level_start[next_level] + hist[next_level])
                coords[start:start + len(sel)] = sel
                hist[next_level] = start + len(sel) - level_start[next_level]


def _process_level(pass_, pix, width, height, level_start, n_points, coords, xmask, shift):
    pts = coords[level_start:level_start + n_points]
    x = pts & xmask
    y = pts >> shift
    off = x + y * width
    xmax, ymax = width - 1, height - 1
    index = np.zeros(len(pts), np.int64)
    conds = ((y > 0), (x < xmax) & (y > 0), (x < xmax), (x < xmax) & (y < ymax),
             (y < ymax), (x > 0) & (y < ymax), (x > 0), (x > 0) & (y > 0))
    for (dx, dy, bit), cond in zip(_LEVEL_BITS, conds):
        if cond.any():
            o = off[cond] + dx + dy * width
            index[cond] ^= np.where(pix[o] == 255, bit, 0)
    change = (_FATE[index] >> pass_) & 1 == 1
    n_changed = int(change.sum())
    if n_changed:
        keep = pts[~change].copy()
        pix[off[change]] = 255
        coords[level_start:level_start + len(keep)] = keep
    return n_changed


def _n_radii(pix, width, height, x, y) -> int:
    """nRadii: number of 4-connected lines emanating from (x, y) (non-255 = line)."""
    offset = x + y * width
    xmax, ymax = width - 1, height - 1
    inner = 0 < y < ymax and 0 < x < xmax
    dir_off = (-width, -width + 1, 1, width + 1, width, width - 1, -1, -width - 1)
    count = 0
    prev = True
    first = True
    for d in range(8):
        pixel_set = prev
        if inner or _within(x, y, d, xmax, ymax):
            is_set = pix[offset + dir_off[d]] != 255
            if d & 1 == 0:
                pixel_set = is_set
            elif not is_set:
                pixel_set = False
        else:
            pixel_set = True
        if pixel_set and not prev:
            count += 1
        prev = pixel_set
        if d == 0:
            first = pixel_set
    if first and not prev:
        count += 1
    return count


def _cleanup_extra_lines(out: np.ndarray) -> None:
    """cleanupExtraLines: remove single points and dangling line ends of non-EDM watersheds."""
    height, width = out.shape
    pix = bytearray(out.tobytes())
    xmax, ymax = width - 1, height - 1
    dir_off = (-width, -width + 1, 1, width + 1, width, width - 1, -1, -width - 1)
    cand = np.nonzero((out != 0) & (out != 255))
    for y, x in zip(cand[0].tolist(), cand[1].tolist()):
        v = pix[x + y * width]
        if v == 255 or v == 0:
            continue
        nr = _n_radii(pix, width, height, x, y)
        if nr == 0:
            pix[x + y * width] = 255
        elif nr == 1:
            # removeLineFrom
            pix[x + width * y] = 255
            while True:
                cont = False
                inner = 0 < y < ymax and 0 < x < xmax
                for d in range(0, 8, 2):
                    if inner or _within(x, y, d, xmax, ymax):
                        vv = pix[x + width * y + dir_off[d]]
                        if vv != 255 and vv != 0:
                            nr2 = _n_radii(pix, width, height, x + DIR_X[d], y + DIR_Y[d])
                            if nr2 <= 1:
                                x += DIR_X[d]
                                y += DIR_Y[d]
                                pix[x + width * y] = 255
                                cont = nr2 == 1
                                break
                if not cont:
                    break
    out[...] = np.frombuffer(bytes(pix), np.uint8).reshape(out.shape)


def _delete_edge_particles(out: np.ndarray, types: np.ndarray) -> None:
    """deleteEdgeParticles: remove particles whose maximum area touches the image edge
    (Wand legacy 8-connected outline filled, as ImageJ does)."""
    from scipy import ndimage

    height, width = out.shape
    seq = []
    for x in range(width):
        seq += [(x, 0), (x, height - 1)]
    for y in range(1, height - 1):
        seq += [(0, y), (width - 1, y)]
    eight = np.ones((3, 3), bool)
    four = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    for x, y in seq:
        if not (types[y, x] & MAX_AREA) or out[y, x] == 0:
            continue
        lab, _ = ndimage.label(out == 255, structure=eight)
        comp = lab == lab[y, x]
        # the wand starts at the first non-particle pixel to the right of the seed
        xr = x
        while xr < width and comp[y, xr]:
            xr += 1
        rest, _ = ndimage.label(~comp, structure=four)
        if xr < width and _is_hole(rest, rest[y, xr]):
            region = rest == rest[y, xr]            # traced an inner hole: fill the hole
        else:
            border = np.unique(np.concatenate([rest[0], rest[-1], rest[:, 0], rest[:, -1]]))
            region = ~np.isin(rest, border[border > 0])
        out[region] = 0


def _is_hole(rest: np.ndarray, label: int) -> bool:
    if label == 0:
        return False
    return not (np.any(rest[0] == label) or np.any(rest[-1] == label) or
                np.any(rest[:, 0] == label) or np.any(rest[:, -1] == label))

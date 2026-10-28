"""Process > Smooth / Sharpen / Find Edges / Noise / Shadows / Filters (2-D and 3-D).

Ported from ImageJ 1.54p (public domain, NIH; W. Rasband, M. Schmid et al.):
``ij/process/{Byte,Short,Float,Color}Processor.java`` (filter / filter3x3 / convolve3x3 /
noise / setPixels(channel, fp)), ``ij/plugin/filter/{Filters,Shadows,SaltAndPepper,Convolver,
GaussianBlur,RankFilters,UnsharpMask}.java``, ``ij/plugin/{Filters3D,GaussianBlur3D}.java``,
``ij/process/StackProcessor.java`` (filter3D) and ``ij/util/ArrayUtil.java``, plus
``java.util.Random`` (48-bit LCG and polar ``nextGaussian``) so seeded noise is reproducible.

The arithmetic follows Java: 8/16-bit 3x3 filters use integer maths (``(sum+4)/9``,
truncating division), float filters accumulate in float32 in ImageJ's order, the Gaussian uses
ImageJ's kernel with edge correction and its down/upscaling for sigma > 8.5, rank filters use
``makeLineRadii`` circular kernels and write ``(int)(v+0.5f)``; conversions from float use
``setPixels(channel, fp)`` (+0.5, clamp, truncate). Every public function takes a 2-D plane
(``uint8``/``uint16``/``float32``) and returns a new plane of the same dtype; RGB planes
``(H, W, 3)`` are accepted where ImageJ treats colour specially (Sharpen/Shadows: interior
only) or processes the three channels in sequence (noise); otherwise callers filter each
channel as an 8-bit plane, exactly as ``ColorProcessor`` does.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy import ndimage

F32 = np.float32
FLT_MAX = float(np.finfo(np.float32).max)

# ---------------------------------------------------------------------------------- helpers


def _max_value(dtype) -> float:
    return 255.0 if dtype == np.uint8 else 65535.0


def from_float(f: np.ndarray, dtype) -> np.ndarray:
    """``ImageProcessor.setPixels(channel, fp)``: value+0.5f, clamp 0..max, truncate."""
    dtype = np.dtype(dtype)
    if dtype == np.float32:
        return np.asarray(f, dtype=np.float32)
    v = np.asarray(f, dtype=np.float32) + F32(0.5)
    v = np.nan_to_num(v, nan=0.0, posinf=_max_value(dtype), neginf=0.0)
    v = np.clip(v, 0.0, _max_value(dtype))
    return v.astype(np.int64).astype(dtype)


def _round_rank(v: np.ndarray, dtype) -> np.ndarray:
    """RankFilters ``writeLineToPixels``: ``(int)(v+0.5f) & mask`` (no clamping)."""
    dtype = np.dtype(dtype)
    if dtype == np.float32:
        return np.asarray(v, dtype=np.float32)
    w = np.asarray(v, dtype=np.float32) + F32(0.5)
    w = np.nan_to_num(w, nan=0.0)
    w = np.trunc(w).astype(np.int64)
    return (w & (0xff if dtype == np.uint8 else 0xffff)).astype(dtype)


def _java_div(a: np.ndarray, b: int) -> np.ndarray:
    """Java integer division (truncates toward zero)."""
    q = np.abs(a) // abs(b)
    return np.where((a < 0) != (b < 0), -q, q)


def _neighbours(p: np.ndarray):
    """v1..v9 of a 3x3 neighbourhood (edge pixels duplicated, as ImageJ)."""
    h, w = p.shape
    q = np.pad(p, 1, mode="edge")
    return [q[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)]


def _per_channel(plane: np.ndarray, fn, *args, **kw) -> np.ndarray:
    out = np.empty_like(plane)
    for k in range(plane.shape[-1]):
        out[..., k] = fn(np.ascontiguousarray(plane[..., k]), *args, **kw)
    return out


# ---------------------------------------------------------------------------------- 3x3 family
BLUR_MORE, FIND_EDGES = "blur", "edges"


def smooth(plane: np.ndarray) -> np.ndarray:
    """Process > Smooth (``filter(BLUR_MORE)``): 3x3 mean."""
    a = np.asarray(plane)
    if a.ndim == 3:
        return _per_channel(a, smooth)
    if a.shape[1] <= 1:
        return a.copy()
    if a.dtype == np.float32:
        v = _neighbours(a)
        s = v[0]
        for x in v[1:]:
            s = s + x
        return (s * F32(0.11111111)).astype(np.float32)
    v = _neighbours(a.astype(np.int64))
    return ((sum(v) + 4) // 9).astype(a.dtype)


def find_edges(plane: np.ndarray) -> np.ndarray:
    """Process > Find Edges: 3x3 Sobel ``sqrt(sum1^2 + sum2^2)``."""
    a = np.asarray(plane)
    if a.ndim == 3:
        return _per_channel(a, find_edges)
    if a.shape[1] <= 1:
        return a.copy()
    if a.dtype == np.float32:
        v1, v2, v3, v4, _, v6, v7, v8, v9 = _neighbours(a)
        two = F32(2)
        s1 = v1 + two * v2
        s1 = s1 + v3
        s1 = s1 - v7
        s1 = s1 - two * v8
        s1 = s1 - v9
        s2 = v1 + two * v4
        s2 = s2 + v7
        s2 = s2 - v3
        s2 = s2 - two * v6
        s2 = s2 - v9
        q = (s1 * s1 + s2 * s2).astype(np.float64)
        return np.sqrt(q).astype(np.float32)
    v1, v2, v3, v4, _, v6, v7, v8, v9 = _neighbours(a.astype(np.int64))
    s1 = v1 + 2 * v2 + v3 - v7 - 2 * v8 - v9
    s2 = v1 + 2 * v4 + v7 - v3 - 2 * v6 - v9
    r = np.sqrt((s1 * s1 + s2 * s2).astype(np.float64))
    if a.dtype == np.uint8:
        return np.minimum(r.astype(np.int64), 255).astype(np.uint8)
    return np.minimum(r, 65535.0).astype(np.int64).astype(np.uint16)


def convolve3x3(plane: np.ndarray, kernel) -> np.ndarray:
    """``ImageProcessor.convolve3x3`` (Sharpen, Shadows). RGB planes keep their edge pixels
    and divide without rounding, as ``ColorProcessor.convolve3x3``."""
    a = np.asarray(plane)
    k = [int(x) for x in kernel]
    scale = sum(k) or 1
    if a.ndim == 3:   # ColorProcessor: interior only, truncating division
        out = a.copy()
        h, w = a.shape[:2]
        if h < 3 or w < 3:
            return out
        for c in range(3):
            ch = a[..., c].astype(np.int64)
            v = [ch[dy:dy + h - 2, dx:dx + w - 2] for dy in range(3) for dx in range(3)]
            s = sum(kk * vv for kk, vv in zip(k, v))
            s = np.clip(_java_div(s, scale), 0, 255)
            out[1:h - 1, 1:w - 1, c] = s
        return out
    if a.shape[1] <= 1:
        return a.copy()
    if a.dtype == np.float32:
        v = _neighbours(a)
        kf = [F32(x) for x in k]
        s = kf[0] * v[0]
        for kk, vv in zip(kf[1:], v[1:]):
            s = s + kk * vv
        return (s * (F32(1.0) / F32(scale))).astype(np.float32)
    v = _neighbours(a.astype(np.int64))
    s = sum(kk * vv for kk, vv in zip(k, v))
    s = _java_div(s + _java_div(np.int64(scale), 2), scale)
    return np.clip(s, 0, _max_value(a.dtype)).astype(a.dtype)


SHARPEN_KERNEL = (-1, -1, -1, -1, 12, -1, -1, -1, -1)
SHADOW_KERNELS = {
    "north": (1, 2, 1, 0, 1, 0, -1, -2, -1),
    "south": (-1, -2, -1, 0, 1, 0, 1, 2, 1),
    "east": (-1, 0, 1, -2, 1, 2, -1, 0, 1),
    "west": (1, 0, -1, 2, 1, -2, 1, 0, -1),
    "northwest": (2, 1, 0, 1, 1, -1, 0, -1, -2),
    "southeast": (-2, -1, 0, -1, 1, 1, 0, 1, 2),
    "northeast": (0, 1, 2, -1, 1, 1, -2, -1, 0),
    "southwest": (0, -1, -2, 1, 1, -1, 2, 1, 0),
}
SHADOW_ORDER = ("north", "northeast", "east", "southeast", "south", "southwest", "west",
                "northwest")


def sharpen(plane: np.ndarray) -> np.ndarray:
    """Process > Sharpen: kernel -1 ... 12 ... -1 (sum 4)."""
    return convolve3x3(plane, SHARPEN_KERNEL)


def shadow(plane: np.ndarray, direction: str) -> np.ndarray:
    """Process > Shadows > ``direction`` ("north" ... "northwest")."""
    return convolve3x3(plane, SHADOW_KERNELS[direction.lower()])


# ---------------------------------------------------------------------------------- Convolver
def parse_kernel(text: str, macro: bool = True) -> tuple[np.ndarray, int, int]:
    """Convolver ``decodeKernel``: rows separated by newlines; in macro mode a text without a
    newline is a square kernel. Returns (float32 kernel (kh, kw), kw, kh); raises ValueError."""
    text = text.replace("\\n", "\n")
    if macro and "\n" not in text.strip():
        vals = [_parse_double(v, float("nan")) for v in text.split()]
        kw = int(math.sqrt(len(vals)))
        if kw < 3 or not kw & 1:
            raise ValueError(f"Kernel must be square with odd width. This one is {kw}x{kw}.")
        k = np.array(vals[:kw * kw], dtype=np.float32).reshape(kw, kw)
        return k, kw, kw
    rows = [r for r in text.strip().split("\n")]
    kh = len(rows)
    if kh == 0 or not rows[0].strip():
        raise ValueError("Kernel is empty")
    kw = len(rows[0].split())
    vals = []
    for y, row in enumerate(rows):
        items = row.split()
        if len(items) != kw:
            raise ValueError(f"Row {y + 1} is not the same length\nas the first row")
        vals.extend(_parse_double(v, 0.0) for v in items)
    if not (kw & 1) or not (kh & 1):
        raise ValueError(f"Kernel must have odd width and height.\nThis one is {kw}x{kh}.")
    return np.array(vals, dtype=np.float32).reshape(kh, kw), kw, kh


def _parse_double(s: str, default: float) -> float:
    try:
        return float(s)
    except ValueError:
        return default


def convolve(plane: np.ndarray, kernel, normalize: bool = True) -> np.ndarray:
    """Process > Filters > Convolve (``Convolver.convolveFloat``): float products summed in
    double in kernel order, edge pixels duplicated, ``(float)(sum*scale)``; integer images
    go through float and back (CONVERT_TO_FLOAT)."""
    a = np.asarray(plane)
    if a.ndim == 3:
        return _per_channel(a, convolve, kernel, normalize)
    k = np.asarray(kernel, dtype=np.float32)
    if k.ndim == 1:
        n = int(round(math.sqrt(k.size)))
        k = k.reshape(n, n)
    kh, kw = k.shape
    if not (kw & 1) or not (kh & 1):
        raise ValueError(f"Kernel width or height not odd ({kw}x{kh})")
    f = a.astype(np.float32)
    h, w = f.shape
    uc, vc = kw // 2, kh // 2
    p = np.pad(f, ((vc, vc), (uc, uc)), mode="edge")
    total = np.zeros((h, w), dtype=np.float64)
    for v in range(kh):
        for u in range(kw):
            kv = k[v, u]
            total += (p[v:v + h, u:u + w] * kv).astype(np.float64)
    s = float(np.sum(k.astype(np.float64)))
    scale = (1.0 / s if s != 0.0 else 1.0) if normalize else 1.0
    out = (total * scale).astype(np.float32)
    return from_float(out, a.dtype)


# ---------------------------------------------------------------------------------- Gaussian
def make_gaussian_kernel(sigma: float, accuracy: float, max_radius: int):
    """``GaussianBlur.makeGaussianKernel``: (kernel float32[k], running sum float32[k])."""
    k_radius = int(math.ceil(sigma * math.sqrt(-2 * math.log(accuracy)))) + 1
    if max_radius < 50:
        max_radius = 50
    if k_radius > max_radius:
        k_radius = max_radius
    kern = [float(F32(math.exp(-0.5 * i * i / sigma / sigma))) for i in range(k_radius)]
    if k_radius < max_radius and k_radius > 3:      # edge correction
        sqrt_slope = 1.7976931348623157e308
        r = k_radius
        while r > k_radius // 2:
            r -= 1
            a = math.sqrt(kern[r]) / (k_radius - r)
            if a < sqrt_slope:
                sqrt_slope = a
            else:
                break
        for r1 in range(r + 2, k_radius):
            kern[r1] = float(F32(((k_radius - r1) * (k_radius - r1)) * sqrt_slope *
                                 sqrt_slope))
    if k_radius < max_radius:
        total = kern[0]
        for i in range(1, k_radius):
            total += 2 * kern[i]
    else:
        total = sigma * math.sqrt(2 * math.pi)
    rsum = 0.5 + 0.5 * kern[0] / total
    k0 = np.empty(k_radius, np.float32)
    k1 = np.empty(k_radius, np.float32)
    for i in range(k_radius):
        v = kern[i] / total
        k0[i] = v
        rsum -= v
        k1[i] = rsum
    return k0, k1


def _convolve_lines(inp: np.ndarray, kern: np.ndarray, ksum: np.ndarray, write_from: int,
                    write_to: int) -> np.ndarray:
    """``GaussianBlur.convolveLine`` for many lines at once (rows of ``inp``)."""
    n, length = inp.shape
    kr = len(kern)
    first = inp[:, :1]
    last = inp[:, -1:]
    first_part = min(kr, length)
    i_end_inside = min(length - kr, write_to)
    idx = np.arange(write_from, write_to)
    res = inp[:, write_from:write_to] * kern[0]
    part1 = idx < first_part
    part3 = (~part1) & (idx >= i_end_inside)
    first_term = (part1 | (part3 & (idx < kr)))
    last_term = (part1 & (idx + kr > length)) | (part3 & (idx + kr >= length))
    if first_term.any():
        ks = ksum[np.clip(idx, 0, kr - 1)]
        term = np.where(first_term[None, :], ks[None, :] * first, F32(0))
        res = np.where(first_term[None, :], res + term, res)
    if last_term.any():
        ks = ksum[np.clip(length - idx - 1, 0, kr - 1)]
        term = ks[None, :] * last
        res = np.where(last_term[None, :], res + term, res)
    pad = np.zeros((n, length + 2 * kr), dtype=np.float32)
    pad[:, kr:kr + length] = inp
    for k in range(1, kr):
        a = pad[:, kr + write_from - k:kr + write_to - k]
        b = pad[:, kr + write_from + k:kr + write_to + k]
        res = res + kern[k] * (a + b)
    return res.astype(np.float32)


def _downscale_kernel(unit: int) -> np.ndarray:
    mid = unit * 3 // 2
    kernel = np.zeros(3 * unit, np.float32)
    for i in range(unit // 2 + 1):
        x = i / unit
        v = F32((0.75 - x * x) / unit)
        kernel[mid - i] = v
        kernel[mid + i] = v
    for i in range(unit // 2 + 1, (unit * 3 + 1) // 2):
        x = i / unit
        v = F32((0.125 + 0.5 * (x - 1) * (x - 2)) / unit)
        kernel[mid - i] = v
        kernel[mid + i] = v
    return kernel


def _upscale_kernel(unit: int) -> np.ndarray:
    kernel = np.zeros(4 * unit, np.float32)
    mid = 2 * unit
    for i in range(unit):
        x = i / unit
        v = F32(2. / 3. - x * x * (1 - 0.5 * x))
        kernel[mid + i] = v
        kernel[mid - i] = v
    for i in range(unit, 2 * unit):
        x = i / unit
        v = F32((2. - x) * (2. - x) * (2. - x) / 6.)
        kernel[mid + i] = v
        kernel[mid - i] = v
    return kernel


def blur_1direction(pixels: np.ndarray, sigma: float, accuracy: float, x_direction: bool,
                    extra_lines: int, roi: tuple[int, int, int, int] | None = None) -> None:
    """``GaussianBlur.blur1Direction`` on a float32 (H, W) array, in place."""
    height, width = pixels.shape
    rx, ry, rw, rh = roi if roi is not None else (0, 0, width, height)
    view = pixels if x_direction else pixels.T
    length = width if x_direction else height
    line_from = max((ry if x_direction else rx) - extra_lines, 0)
    line_to = min((ry + rh if x_direction else rx + rw) + extra_lines,
                  height if x_direction else width)
    write_from = rx if x_direction else ry
    write_to = rx + rw if x_direction else ry + rh
    if line_to <= line_from or write_to <= write_from:
        return
    upscale_k_radius = 2
    min_downscaled_sigma = 4.0
    do_downscaling = sigma > 2 * min_downscaled_sigma + 0.5
    reduce_by = min(int(math.floor(sigma / min_downscaled_sigma)), length) \
        if do_downscaling else 1
    sigma_gauss = math.sqrt(sigma * sigma / (reduce_by * reduce_by) - 1. / 3. - 1. / 4.) \
        if do_downscaling else sigma
    max_length = ((length + reduce_by - 1) // reduce_by + 2 * (upscale_k_radius + 1)) \
        if do_downscaling else length
    kern, ksum = make_gaussian_kernel(sigma_gauss, accuracy, max_length)
    k_radius = len(kern) * reduce_by
    lines = np.array(view[line_from:line_to, :], dtype=np.float32)
    if not do_downscaling:
        view[line_from:line_to, write_from:write_to] = _convolve_lines(
            lines, kern, ksum, write_from, write_to)
        return
    read_from = max(write_from - k_radius, 0)
    read_to = min(write_to + k_radius, length)
    new_length = (read_to - read_from + reduce_by - 1) // reduce_by + \
        2 * (upscale_k_radius + 1)
    unscaled0 = read_from - (upscale_k_radius + 1) * reduce_by
    dk = _downscale_kernel(reduce_by)
    uk = _upscale_kernel(reduce_by)
    # downscaleLine: block of input for xout = -1 .. new_length
    r = reduce_by
    p0 = unscaled0 - r * 3 // 2
    n = lines.shape[0]
    xouts = np.arange(-1, new_length + 1)
    s0 = np.zeros((n, xouts.size), np.float32)
    s1 = np.zeros_like(s0)
    s2 = np.zeros_like(s0)
    for x in range(r):
        idx = np.clip(p0 + (xouts + 1) * r + x, 0, length - 1)
        v = lines[:, idx]
        s0 = s0 + v * dk[x + 2 * r]
        s1 = s1 + v * dk[x + r]
        s2 = s2 + v * dk[x]
    j = np.arange(new_length)
    cache = (s2[:, j] + s1[:, j + 1]) + s0[:, j + 2]
    cache2 = np.zeros_like(cache)
    cache2[:, 1:new_length - 1] = _convolve_lines(cache, kern, ksum, 1, new_length - 1)
    xo = np.arange(write_from, write_to)
    t = xo - unscaled0 + r - 1
    xin = t // r
    xk = r - 1 - t % r
    out = cache2[:, xin - 2] * uk[xk]
    out = out + cache2[:, xin - 1] * uk[xk + r]
    out = out + cache2[:, xin] * uk[xk + 2 * r]
    out = out + cache2[:, xin + 1] * uk[xk + 3 * r]
    view[line_from:line_to, write_from:write_to] = out


def blur_float(f: np.ndarray, sigma_x: float, sigma_y: float, accuracy: float,
               roi=None) -> np.ndarray:
    """``GaussianBlur.blurFloat`` on a float32 copy of ``f``."""
    out = np.array(f, dtype=np.float32)
    if sigma_x > 0:
        blur_1direction(out, sigma_x, accuracy, True, int(math.ceil(5 * sigma_y)), roi)
    if sigma_y > 0:
        blur_1direction(out, sigma_y, accuracy, False, 0, roi)
    return out


def gaussian_blur(plane: np.ndarray, sigma_x: float, sigma_y: float | None = None,
                  accuracy: float | None = None, roi=None) -> np.ndarray:
    """Process > Filters > Gaussian Blur (accuracy 0.002 for 8-bit/RGB, else 0.0002).
    ``roi`` = (x, y, w, h) rectangle that ImageJ filters (pixels outside are unchanged)."""
    a = np.asarray(plane)
    if sigma_y is None:
        sigma_y = sigma_x
    if a.ndim == 3:
        return _per_channel(a, gaussian_blur, sigma_x, sigma_y,
                            0.002 if accuracy is None else accuracy, roi)
    if accuracy is None:
        accuracy = 0.002 if a.dtype == np.uint8 else 0.0002
    out = blur_float(a.astype(np.float32), sigma_x, sigma_y, accuracy, roi)
    res = from_float(out, a.dtype)
    if roi is not None:
        x, y, w, h = roi
        keep = a.copy()
        keep[y:y + h, x:x + w] = res[y:y + h, x:x + w]
        return keep
    return res


def unsharp_mask(plane: np.ndarray, sigma: float, weight: float, roi=None) -> np.ndarray:
    """Process > Filters > Unsharp Mask: ``(orig - w*gauss)/(1-w)`` (Gaussian accuracy
    0.01, float32)."""
    a = np.asarray(plane)
    if a.ndim == 3:
        return _per_channel(a, unsharp_mask, sigma, weight, roi)
    f = a.astype(np.float32)
    g = blur_float(f, sigma, sigma, 0.01, roi)
    w = F32(weight)
    out = (f - w * g) / (F32(1.0) - w)
    res = from_float(out, a.dtype)
    if roi is not None:
        x, y, rw, rh = roi
        keep = a.copy()
        keep[y:y + rh, x:x + rw] = res[y:y + rh, x:x + rw]
        return keep
    return res


# ---------------------------------------------------------------------------------- RankFilters
MEAN, MIN, MAX, VARIANCE, MEDIAN, OUTLIERS, DESPECKLE, REMOVE_NAN, TOP_HAT = (
    "mean", "min", "max", "variance", "median", "outliers", "despeckle", "remove_nan",
    "top_hat")
RANK_KINDS = (MEAN, MIN, MAX, VARIANCE, MEDIAN, OUTLIERS, DESPECKLE, REMOVE_NAN, TOP_HAT)


def make_line_radii(radius: float) -> list[int]:
    """``RankFilters.makeLineRadii``: [-dx0, dx0, -dx1, dx1, ..., nPoints, kRadius]."""
    if 1.5 <= radius < 1.75:
        radius = 1.75
    elif 2.5 <= radius < 2.85:
        radius = 2.85
    r2 = int(radius * radius) + 1
    k_radius = int(math.sqrt(r2 + 1e-10))
    k_height = 2 * k_radius + 1
    kernel = [0] * (2 * k_height + 2)
    kernel[2 * k_radius] = -k_radius
    kernel[2 * k_radius + 1] = k_radius
    n_points = 2 * k_radius + 1
    for y in range(1, k_radius + 1):
        dx = int(math.sqrt(r2 - y * y + 1e-10))
        kernel[2 * (k_radius - y)] = -dx
        kernel[2 * (k_radius - y) + 1] = dx
        kernel[2 * (k_radius + y)] = -dx
        kernel[2 * (k_radius + y) + 1] = dx
        n_points += 4 * dx + 2
    kernel[-2] = n_points
    kernel[-1] = k_radius
    return kernel


def line_radii(radius: float) -> list[tuple[int, int]]:
    """Circular kernel as [(dy, dx)]: line ``dy`` covers x offsets -dx..dx."""
    lr = make_line_radii(radius)
    kr = lr[-1]
    return [(i - kr, lr[2 * i + 1]) for i in range((len(lr) - 2) // 2)]


def circular_footprint(radius: float) -> np.ndarray:
    """Bool footprint (2k+1, 2k+1) of the RankFilters kernel."""
    lines = line_radii(radius)
    kr = make_line_radii(radius)[-1]
    fp = np.zeros((2 * kr + 1, 2 * kr + 1), bool)
    for dy, dx in lines:
        fp[dy + kr, kr - dx:kr + dx + 1] = True
    return fp


def circular_masks_image() -> tuple[np.ndarray, list[str]]:
    """Process > Filters > Show Circular Masks: (99, 150, 150) float32 stack + labels."""
    w = h = 150
    planes, labels = [], []
    r = 0.5
    while r < 50:
        ip = np.zeros((h, w), np.float32)
        lr = make_line_radii(r)
        kh = (len(lr) - 2) // 2
        kr = lr[-1]
        y0 = h // 2 - kh // 2
        for i in range(kh):
            ip[y0 + i, w // 2 + lr[2 * i]:w // 2 + lr[2 * i + 1] + 1] = 1.0
        planes.append(ip)
        labels.append(f"radius={r!r}, size={2 * kr + 1}")
        r += 0.5
    return np.stack(planes), labels


def _window_sums(f64: np.ndarray, radius: float) -> np.ndarray:
    """Sum over the circular kernel (edge padding), via per-line cumulative sums."""
    lr = make_line_radii(radius)
    kr = lr[-1]
    h, w = f64.shape
    p = np.pad(f64, kr, mode="edge")
    cs = np.zeros((p.shape[0], p.shape[1] + 1), np.float64)
    np.cumsum(p, axis=1, out=cs[:, 1:])
    total = np.zeros((h, w), np.float64)
    cols = np.arange(w)
    for i in range((len(lr) - 2) // 2):
        dy, dx = i - kr, lr[2 * i + 1]
        rows = cs[kr + dy:kr + dy + h]
        total += rows[:, cols + kr + dx + 1] - rows[:, cols + kr - dx]
    return total


def _area_extreme(f: np.ndarray, radius: float, kind: str) -> np.ndarray:
    """Min or max over the kernel; NaN ignored (all NaN -> -/+Float.MAX_VALUE)."""
    fp = circular_footprint(radius)
    a = f
    nan = np.isnan(a)
    if kind == MAX:
        if nan.any():
            a = np.where(nan, -np.inf, a)
        out = ndimage.maximum_filter(a, footprint=fp, mode="nearest")
        if nan.any():
            out = np.where(np.isneginf(out), F32(-FLT_MAX), out)
    else:
        if nan.any():
            a = np.where(nan, np.inf, a)
        out = ndimage.minimum_filter(a, footprint=fp, mode="nearest")
        if nan.any():
            out = np.where(np.isposinf(out), F32(FLT_MAX), out)
    return out.astype(np.float32)


def _nan_aware_median(vals: np.ndarray, guess: float) -> float:
    """``RankFilters.getNaNAwareMedian`` with its guess-dependent choice for even counts."""
    v = vals[~np.isnan(vals)]
    n = v.size
    if n == 0:
        return float("nan")
    half = n // 2
    above = np.sort(v[v > guess])
    below = np.sort(v[v < guess])
    if above.size > half:
        return float(above[above.size - half - 1])
    if below.size > half:
        return float(below[half])
    return guess


def _median(f: np.ndarray, radius: float) -> np.ndarray:
    fp = circular_footprint(radius)
    nan = np.isnan(f) if f.dtype == np.float32 else None
    if nan is None or not nan.any():
        return ndimage.median_filter(f, footprint=fp, mode="nearest").astype(np.float32)
    out = ndimage.median_filter(np.where(nan, 0, f), footprint=fp, mode="nearest").astype(
        np.float32)
    touched = ndimage.binary_dilation(nan, structure=fp)
    kr = fp.shape[0] // 2
    p = np.pad(f, kr, mode="edge")
    for y, x in zip(*np.nonzero(touched)):
        g = f[y, x]
        if np.isnan(g):
            out[y, x] = np.nan
            continue
        win = p[y:y + 2 * kr + 1, x:x + 2 * kr + 1][fp]
        out[y, x] = _nan_aware_median(win, float(g))
    return out


def _remove_nans(f: np.ndarray, radius: float, rect=None) -> np.ndarray:
    fp = circular_footprint(radius)
    kr = fp.shape[0] // 2
    out = f.copy()
    nan = np.isnan(f)
    if not nan.any():
        return out
    h, w = f.shape
    x0, y0, rw, rh = rect if rect is not None else (0, 0, w, h)
    p = np.pad(f, kr, mode="edge")
    for y in range(y0, y0 + rh):
        row_nan = nan[y, x0:x0 + rw]
        if not row_nan.any():
            continue
        guess = 0.0
        for x in range(x0, x0 + rw):
            v = f[y, x]
            if np.isnan(v):
                win = p[y:y + 2 * kr + 1, x:x + 2 * kr + 1][fp]
                out[y, x] = _nan_aware_median(win, guess)
            else:
                guess = float(v)
    return out


def _nan_propagation(nan: np.ndarray, lr: list[int]) -> np.ndarray:
    """Which mean/variance outputs are NaN in ImageJ: the running sums (``addSideSums``)
    stay NaN for one pixel after the window is free again, then a full sum is taken."""
    kr = lr[-1]
    h, w = nan.shape
    p = np.pad(nan.astype(np.int32), kr + 1, mode="edge")
    win = np.zeros((h, w), bool)
    side = np.zeros((h, w), bool)
    o = kr + 1
    for i in range((len(lr) - 2) // 2):
        dy, dx = i - kr, lr[2 * i + 1]
        rows = p[o + dy:o + dy + h]
        cs = np.zeros((h, rows.shape[1] + 1), np.int64)
        np.cumsum(rows, axis=1, out=cs[:, 1:])
        cols = np.arange(w) + o
        win |= (cs[:, cols + dx + 1] - cs[:, cols - dx]) > 0
        side |= (rows[:, cols - dx - 1] > 0) | (rows[:, cols + dx] > 0)
    if kr < 2:          # small kernels: always the full sum
        return win
    out = np.zeros((h, w), bool)
    out[:, 0] = win[:, 0]
    full = np.zeros(h, bool)
    for x in range(1, w):
        cur = np.where(full, win[:, x], out[:, x - 1] | side[:, x])
        full = np.where(full, False, cur)
        out[:, x] = cur
    return out


def _rank_core(f: np.ndarray, radius: float, kind: str, max_value: float) -> np.ndarray:
    """One RankFilters pass on float32 data; returns float32 values (before rounding)."""
    lr = make_line_radii(radius)
    npts = lr[-2]
    if kind in (MIN, MAX):
        return _area_extreme(f, radius, kind)
    if kind in (MEAN, VARIANCE):
        d = f.astype(np.float64)
        nan = np.isnan(d)
        has_nan = nan.any()
        if has_nan:
            d = np.where(nan, 0.0, d)
        s = _window_sums(d, radius)
        if kind == MEAN:
            out = (s / npts).astype(np.float32)
        else:
            s2 = _window_sums(d * d, radius)
            out = ((s2 - s * s / npts) / npts).astype(np.float32)
            if not math.isnan(max_value):
                out = np.minimum(out, F32(max_value))
            out = np.maximum(out, F32(0))
        if has_nan:
            out = np.where(_nan_propagation(nan, lr), F32(np.nan), out)
        return out
    if kind == MEDIAN:
        return _median(f, radius)
    raise ValueError(kind)


def rank_filter(plane: np.ndarray, radius: float, kind: str, threshold: float = 50.0,
                which_outliers: str = "Bright", light_background: bool = False,
                dont_subtract: bool = False, inverted_lut: bool = False,
                rect=None) -> np.ndarray:
    """``RankFilters.rank``: kind in :data:`RANK_KINDS`. RGB planes are filtered per channel.
    ``inverted_lut`` matters for Remove Outliers and Top Hat (as ``ip.isInvertedLut()``);
    ``rect`` (x, y, w, h) only affects Remove NaNs' first median guess per line."""
    a = np.asarray(plane)
    if a.ndim == 3:
        return _per_channel(a, rank_filter, radius, kind, threshold, which_outliers,
                            light_background, dont_subtract, False, rect)
    if kind == DESPECKLE:
        kind, radius = MEDIAN, 1.0
    dtype = a.dtype
    is_float = dtype == np.float32
    max_value = float("nan") if is_float else _max_value(dtype)
    f = a.astype(np.float32)
    if kind == REMOVE_NAN:
        if not is_float:
            raise ValueError('"Remove NaNs" requires a 32-bit image')
        return _remove_nans(f, radius, rect)
    if kind == OUTLIERS:
        dark = str(which_outliers).lower().startswith("d")
        sign = F32(-1.0) if inverted_lut == dark else F32(1.0)
        med = _median(f, radius)
        thr = F32(threshold)
        cond = (f * sign + thr) < (med * sign)
        out = np.where(cond, med, f)
        return _round_rank(out, dtype)
    if kind == TOP_HAT:
        invert = inverted_lut != light_background
        sign = 1.0 if invert else -1.0
        first, second = (MIN, MAX) if sign == -1.0 else (MAX, MIN)
        step1 = _round_rank(_rank_core(f, radius, first, max_value), dtype).astype(np.float32)
        step2 = _round_rank(_rank_core(step1, radius, second, max_value), dtype)
        if dont_subtract:
            return step2
        offset = F32(max_value) if (sign == 1.0 and not is_float) else F32(0)
        out = (f - step2.astype(np.float32)) + offset
        return from_float(out, dtype)
    return _round_rank(_rank_core(f, radius, kind, max_value), dtype)


# ---------------------------------------------------------------------------------- 3-D
def _ellipsoid(rx: float, ry: float, rz: float) -> np.ndarray:
    """``StackProcessor.createKernelEllipsoid`` as bool (2vz+1, 2vy+1, 2vx+1)."""
    rx, ry, rz = float(F32(rx)), float(F32(ry)), float(F32(rz))
    vx, vy, vz = (int(math.ceil(r)) for r in (rx, ry, rz))

    def inv(r):
        r2 = r * r
        return 1.0 / r2 if r2 != 0 else 0.0
    ix, iy, iz = inv(rx), inv(ry), inv(rz)
    k, j, i = np.mgrid[-vz:vz + 1, -vy:vy + 1, -vx:vx + 1]
    dist = (i * i).astype(float) * ix + (j * j).astype(float) * iy + (k * k).astype(float) * iz
    return dist <= 1.0


def filter_3d(stack: np.ndarray, kind: str, rx: float, ry: float, rz: float) -> np.ndarray:
    """Process > Filters > Median/Mean/Minimum/Maximum/Variance 3D (``Filters3D``):
    ellipsoid neighbourhood, out-of-stack voxels left out. ``stack`` is (Z, Y, X) or
    (Z, Y, X, 3) for RGB; variance returns float32, the others keep the dtype."""
    a = np.asarray(stack)
    if a.ndim == 4:
        chans = [filter_3d(np.ascontiguousarray(a[..., c]), kind, rx, ry, rz)
                 for c in range(3)]
        return np.stack(chans, axis=-1)
    ker = _ellipsoid(rx, ry, rz)
    vz, vy, vx = (s // 2 for s in ker.shape)
    d, h, w = a.shape
    offs = np.argwhere(ker) - np.array([vz, vy, vx])     # in kernel (z, y, x) order
    f = a.astype(np.float32)
    p = np.pad(f, ((vz, vz), (vy, vy), (vx, vx)), mode="constant", constant_values=np.nan)
    valid = np.pad(np.ones_like(f, bool), ((vz, vz), (vy, vy), (vx, vx)))
    out = np.empty((d, h, w), np.float64)
    for z in range(d):
        vals = np.empty((len(offs), h, w), np.float32)
        ok = np.empty((len(offs), h, w), bool)
        for n, (dz, dy, dx) in enumerate(offs):
            vals[n] = p[z + vz + dz, vy + dy:vy + dy + h, vx + dx:vx + dx + w]
            ok[n] = valid[z + vz + dz, vy + dy:vy + dy + h, vx + dx:vx + dx + w]
        size = ok.sum(axis=0)
        if kind == "mean":
            out[z] = np.where(ok, vals.astype(np.float64), 0.0).sum(axis=0) / size
        elif kind == "variance":
            v64 = np.where(ok, vals, 0).astype(np.float64)
            sq = np.where(ok, vals * vals, 0).astype(np.float64)
            tot, tot2 = v64.sum(axis=0), sq.sum(axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                var = (tot2 - (tot * tot / size)) / (size - 1)
            out[z] = np.where(size == 1, 0.0, var)
        elif kind in ("min", "max"):
            fill = np.inf if kind == "min" else -np.inf
            m = np.where(ok, vals, fill)
            out[z] = m.min(axis=0) if kind == "min" else m.max(axis=0)
        elif kind == "median":
            m = np.where(ok, vals, np.inf)
            m.sort(axis=0)
            lo = np.take_along_axis(m, ((size - 1) // 2)[None], 0)[0]
            hi = np.take_along_axis(m, (size // 2)[None], 0)[0]
            even = (F32(0.5) * (lo + hi)).astype(np.float64)
            out[z] = np.where(size % 2 == 1, hi.astype(np.float64), even)
        else:
            raise ValueError(kind)
    if kind == "variance" or a.dtype == np.float32:
        return out.astype(np.float32)
    mx = _max_value(a.dtype)
    v = np.clip(out, 0.0, mx) + 0.5      # ImageStack.setVoxel: (byte)(value+0.5)
    return v.astype(np.int64).astype(a.dtype)


def gaussian_blur_3d(stack: np.ndarray, sigma_x: float, sigma_y: float, sigma_z: float
                     ) -> np.ndarray:
    """Process > Filters > Gaussian Blur 3D on a (Z, Y, X[, 3]) volume: 2-D blur of every
    slice, then a z blur through ``ImageStack.get/setVoxels`` (RGB: truncated)."""
    a = np.asarray(stack)
    rgb = a.ndim == 4
    acc = 0.002 if (a.dtype == np.uint8) else 0.0002
    out = a.copy()
    if sigma_x > 0 or sigma_y > 0:
        for z in range(a.shape[0]):
            out[z] = gaussian_blur(a[z], sigma_x, sigma_y, acc)
    if sigma_z > 0:
        chans = range(3) if rgb else [None]
        for c in chans:
            vol = out[..., c] if rgb else out
            d = vol.shape[0]
            f = vol.reshape(d, -1).astype(np.float32)
            blur_1direction(f, sigma_z, acc, False, 0)
            f = f.reshape(vol.shape)
            if rgb:
                res = (f.astype(np.int64) & 0xff).astype(np.uint8)
            elif a.dtype == np.float32:
                res = f
            else:
                mx = _max_value(a.dtype)
                res = (np.clip(f, F32(0), F32(mx)) + F32(0.5)).astype(np.int64).astype(
                    a.dtype)
            if rgb:
                out[..., c] = res
            else:
                out = res
    return out


# ---------------------------------------------------------------------------------- random
class JavaRandom:
    """``java.util.Random`` (48-bit LCG, polar Box-Muller ``nextGaussian``)."""
    MULT = 0x5DEECE66D
    MASK = (1 << 48) - 1

    def __init__(self, seed: int | None = None):
        if seed is None:
            seed = time.time_ns()
        self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        self.seed = (int(seed) ^ self.MULT) & self.MASK
        self.have_next = False
        self.next_gauss = 0.0

    def next(self, bits: int) -> int:
        self.seed = (self.seed * self.MULT + 0xB) & self.MASK
        r = self.seed >> (48 - bits)
        if r & 0x80000000:
            r -= 1 << 32
        return r

    def next_double(self) -> float:
        return ((self.next(26) << 27) + self.next(27)) * (1.0 / (1 << 53))

    def next_int(self, bound: int) -> int:
        if bound <= 0:
            raise ValueError("bound must be positive")
        if bound & -bound == bound:
            return (bound * self.next(31)) >> 31
        while True:
            bits = self.next(31)
            val = bits % bound
            if bits - val + (bound - 1) < (1 << 31):
                return val

    def next_gaussian(self) -> float:
        if self.have_next:
            self.have_next = False
            return self.next_gauss
        while True:
            v1 = 2 * self.next_double() - 1
            v2 = 2 * self.next_double() - 1
            s = v1 * v1 + v2 * v2
            if 0 < s < 1:
                break
        mult = math.sqrt(-2 * math.log(s) / s)
        self.next_gauss = v2 * mult
        self.have_next = True
        return v1 * mult


def _java_round(x: float) -> int:
    return math.floor(x + 0.5)


def add_noise(plane: np.ndarray, sd: float, rng: JavaRandom | None = None,
              rect=None) -> np.ndarray:
    """Process > Noise > Add (Specified) Noise (``ImageProcessor.noise``): Gaussian noise
    with standard deviation ``sd``; 8/16-bit values are re-drawn until in range. With a
    :class:`JavaRandom` the result equals ImageJ's for the same seed (RGB: red, green, blue
    planes in sequence); without one numpy's generator is used (same statistics)."""
    a = np.asarray(plane)
    if a.ndim == 3:
        out = a.copy()
        for c in range(3):
            out[..., c] = add_noise(a[..., c], sd, rng, rect)
        return out
    h, w = a.shape
    x0, y0, rw, rh = rect if rect is not None else (0, 0, w, h)
    out = a.copy()
    if rng is None:
        gen = np.random.default_rng()
        sub = out[y0:y0 + rh, x0:x0 + rw]
        if a.dtype == np.float32:
            sub += gen.normal(0, sd, sub.shape).astype(np.float32)
            return out
        mx = _max_value(a.dtype)
        base = sub.astype(np.int64)
        res = np.empty_like(base)
        todo = np.ones(base.shape, bool)
        while todo.any():
            ran = np.floor(gen.normal(0, 1, int(todo.sum())) * sd + 0.5).astype(np.int64)
            v = base[todo] + ran
            ok = (v >= 0) & (v <= mx)
            idx = np.flatnonzero(todo)
            res.flat[idx[ok]] = v[ok]
            todo.flat[idx[ok]] = False
        out[y0:y0 + rh, x0:x0 + rw] = res
        return out
    if a.dtype == np.float32:
        for y in range(y0, y0 + rh):
            for x in range(x0, x0 + rw):
                out[y, x] = F32(out[y, x] + F32(rng.next_gaussian() * sd))
        return out
    mx = int(_max_value(a.dtype))
    flat = out.astype(np.int64)
    for y in range(y0, y0 + rh):
        row = flat[y]
        for x in range(x0, x0 + rw):
            p = int(row[x])
            while True:
                v = p + _java_round(rng.next_gaussian() * sd)
                if 0 <= v <= mx:
                    row[x] = v
                    break
    return flat.astype(a.dtype)


def salt_and_pepper(plane: np.ndarray, percent: float = 0.05, rng: JavaRandom | None = None,
                    rect=None) -> np.ndarray:
    """Process > Noise > Salt and Pepper (8-bit): ``percent/2`` of the pixels set to 255 and
    as many to 0, at random positions (``SaltAndPepper.add``)."""
    a = np.asarray(plane)
    rng = rng or JavaRandom()
    h, w = a.shape
    x0, y0, rw, rh = rect if rect is not None else (0, 0, w, h)
    n = int(percent * rw * rh)
    out = a.copy()
    for _ in range(n // 2):
        rx = x0 + rng.next_int(rw)
        ry = y0 + rng.next_int(rh)
        out[ry, rx] = 255
        rx = x0 + rng.next_int(rw)
        ry = y0 + rng.next_int(rh)
        out[ry, rx] = 0
    return out

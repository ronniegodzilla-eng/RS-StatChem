"""Geometric transforms of planes and images: resize (Scale..., Adjust > Size), scale in place,
rotate, translate, bin, flips, 90-degree rotations, crop, canvas size and Image <-> Results.

Ported from ImageJ 1.54 ij/process/{ImageProcessor,ByteProcessor,ShortProcessor,FloatProcessor,
ColorProcessor,DownsizeTable}.java (resize/scale/rotate/translate, bilinear and Catmull-Rom
bicubic interpolation with a = 0.5, area-averaging downsizing) and ij/plugin/{Scaler,Resizer,
Binner,CanvasResizer}.java, ij/plugin/filter/{Rotator,Translator}.java (public domain, NIH;
W. Rasband, M. Schmid et al.). The pixel loops are vectorised with numpy but keep ImageJ's
operation order, double/float precision and truncating casts, so results match ImageJ exactly
(goldens ``tests/data_imagej/golden/image_transform``).

Plane "kinds": ``"byte"`` (uint8), ``"short"`` (uint16), ``"float"`` (float32), ``"rgb"``
(uint8 (..., 3); processed per channel like ColorProcessor.filterRGB, which gives the same
numbers as ColorProcessor's own bilinear code).
"""
from __future__ import annotations

import math

import numpy as np

from rsstatchem.core.imaging.image import (COLOR_256, COLOR_RGB, GRAY8, GRAY16, GRAY32, ImageDoc)

NONE, BILINEAR, BICUBIC = 0, 1, 2
METHODS = ("None", "Bilinear", "Bicubic")
_A = 0.5   # Catmull-Rom


def method_index(name) -> int:
    if isinstance(name, int):
        return name
    return {"none": NONE, "bilinear": BILINEAR, "bicubic": BICUBIC}[str(name).lower()]


def kind_of(doc_or_type) -> str:
    t = doc_or_type.type if isinstance(doc_or_type, ImageDoc) else doc_or_type
    return {GRAY8: "byte", COLOR_256: "byte", GRAY16: "short", GRAY32: "float",
            COLOR_RGB: "rgb"}[t]


def _jint(a):
    """Java (int) cast of doubles (truncate, NaN -> 0)."""
    a = np.asarray(a, dtype=np.float64)
    a = np.nan_to_num(a, nan=0.0, posinf=2.0 ** 31 - 1, neginf=-(2.0 ** 31))
    return np.trunc(a).astype(np.int64)


def _store(v, kind):
    """Store an int (already rounded/clamped by the caller) as the plane dtype."""
    if kind == "byte":
        return (np.asarray(v, dtype=np.int64) & 0xFF).astype(np.uint8)
    if kind == "short":
        return (np.asarray(v, dtype=np.int64) & 0xFFFF).astype(np.uint16)
    return np.asarray(v, dtype=np.float32)


def cubic(x):
    """ImageProcessor.cubic (Catmull-Rom, a = 0.5)."""
    x = np.abs(np.asarray(x, dtype=np.float64))
    z1 = x * x * (x * (-_A + 2.0) + (_A - 3.0)) + 1.0
    z2 = -_A * x * x * x + 5.0 * _A * x * x - 8.0 * _A * x + 4.0 * _A
    return np.where(x < 1.0, z1, np.where(x < 2.0, z2, 0.0))


# ---------------------------------------------------------------------------------- sampling
def _bilinear_core(p: np.ndarray, x, y, kind: str) -> np.ndarray:
    """The private getInterpolatedPixel(x, y, pixels) of each processor (coordinates already
    clamped into [0, w-1.001]); float planes use ImageJ's NaN-aware variant."""
    xb = _jint(x)
    yb = _jint(y)
    xf = x - xb
    yf = y - yb
    w = p.shape[1]
    flat = p.reshape(-1)
    off = yb * w + xb
    ll = flat[off].astype(np.float64)
    lr = flat[off + 1].astype(np.float64)
    ur = flat[off + w + 1].astype(np.float64)
    ul = flat[off + w].astype(np.float64)
    if kind != "float":
        upper = ul + xf * (ur - ul)
        lower = ll + xf * (lr - ll)
        return lower + yf * (upper - lower)
    with np.errstate(invalid="ignore"):
        upper = np.where(np.isnan(ul) & (xf >= 0.5), ur,
                         np.where(np.isnan(ur) & (xf < 0.5), ul, ul + xf * (ur - ul)))
        lower = np.where(np.isnan(ll) & (xf >= 0.5), lr,
                         np.where(np.isnan(lr) & (xf < 0.5), ll, ll + xf * (lr - ll)))
        return np.where(np.isnan(lower) & (yf >= 0.5), upper,
                        np.where(np.isnan(upper) & (yf < 0.5), lower,
                                 lower + yf * (upper - lower)))


def _clamped_bilinear(p, x, y, kind):
    """getInterpolatedPixel(x, y) in BILINEAR mode (coordinates clamped)."""
    h, w = p.shape
    x = np.where(x < 0.0, 0.0, x)
    x = np.where(x >= w - 1.0, w - 1.001, x)
    y = np.where(y < 0.0, 0.0, y)
    y = np.where(y >= h - 1.0, h - 1.001, y)
    return _bilinear_core(p, x, y, kind)


def _bicubic(p: np.ndarray, x0, y0, kind: str, bg: float) -> np.ndarray:
    """getBicubicInterpolatedPixel(x0, y0, ip2) with ``bg`` = ip2's background value."""
    h, w = p.shape
    x0 = np.asarray(x0, dtype=np.float64)
    y0 = np.asarray(y0, dtype=np.float64)
    u0 = np.floor(x0).astype(np.int64)
    v0 = np.floor(y0).astype(np.int64)
    edge = (u0 <= 0) | (u0 >= w - 2) | (v0 <= 0) | (v0 >= h - 2)
    out = np.empty(x0.shape, dtype=np.float64)
    inner = ~edge
    if inner.any():
        xi, yi = x0[inner], y0[inner]
        ui, vi = u0[inner], v0[inner]
        q = np.zeros(xi.shape)
        for j in range(4):
            v = vi - 1 + j
            pp = np.zeros(xi.shape)
            for i in range(4):
                u = ui - 1 + i
                pp = pp + p[v, u].astype(np.float64) * cubic(xi - u)
            q = q + pp * cubic(yi - v)
        out[inner] = q
    if edge.any():
        xe, ye = x0[edge], y0[edge]
        ok = (xe >= -1) & (xe < w) & (ye >= -1) & (ye < h)
        val = np.full(xe.shape, float(bg))
        if ok.any():
            val[ok] = _clamped_bilinear(p, xe[ok], ye[ok], kind)
        out[edge] = val
    return out


def _round_store(v, kind):
    """``(int)(v+0.5)`` clamped to the type range (bicubic results) or float cast."""
    if kind == "float":
        return v.astype(np.float32)
    hi = 255 if kind == "byte" else 65535
    return _store(np.clip(_jint(v + 0.5), 0, hi), kind)


def _per_channel(fn, p, kind, *args, bg=None, **kw):
    if kind != "rgb":
        return fn(p, kind, *args, bg=bg, **kw)
    bgs = _rgb_bg(bg)
    return np.stack([fn(np.ascontiguousarray(p[..., k]), "byte", *args, bg=bgs[k], **kw)
                     for k in range(3)], axis=-1)


def _rgb_bg(bg):
    if bg is None:
        return (255, 255, 255)          # ColorProcessor default background: white
    if isinstance(bg, (tuple, list)):
        return tuple(int(v) for v in bg)
    v = int(bg)
    return ((v >> 16) & 255, (v >> 8) & 255, v & 255)


def default_bg(kind: str, inverted_lut: bool = False):
    if kind == "byte":
        return 0 if inverted_lut else 255
    if kind == "rgb":
        return (255, 255, 255)
    return 0.0


# ---------------------------------------------------------------------------------- resize
def _resize_linearly(p: np.ndarray, kind: str, w2: int, h2: int) -> np.ndarray:
    """ImageProcessor.resizeLinearly for 1-pixel-wide or -high images."""
    rotate = p.shape[1] == 1
    q = p.T[::-1] if rotate else p         # rotateLeft
    if rotate:
        w2, h2 = h2, w2
    data1 = q[0].astype(np.float32)
    width1 = data1.size
    scale = (width1 - 1) / (w2 - 1) if w2 > 1 else 0.0
    x = np.arange(w2)
    x1 = _jint(x * scale)
    x2 = np.where(x1 + 1 == width1, width1 - 1, x1 + 1)
    fr = x * scale - x1
    row = ((1.0 - fr) * data1[x1].astype(np.float64) +
           fr * data1[x2].astype(np.float64)).astype(np.float32)
    out = np.repeat(row[None, :], h2, axis=0)
    if kind == "byte":
        out = _from_float(out, "byte")
    elif kind == "short":
        out = _from_float(out, "short")
    if rotate:
        out = out[::-1].T                   # rotateRight
    return np.ascontiguousarray(out)


def _from_float(f: np.ndarray, kind: str) -> np.ndarray:
    """ByteProcessor/ShortProcessor.setPixels(channel, fp): (v+0.5f) clipped, truncated."""
    if kind == "float":
        return f.astype(np.float32)
    v = f.astype(np.float32) + np.float32(0.5)
    hi = np.float32(255 if kind == "byte" else 65535)
    v = np.where(v < 0, np.float32(0), np.where(v > hi, hi, v))
    return _store(_jint(v), kind)


def resize_plane(p: np.ndarray, kind: str, dst_w: int, dst_h: int, method=BILINEAR,
                 roi: tuple | None = None, average: bool = False, bg=None) -> np.ndarray:
    """ImageProcessor.resize(dstWidth, dstHeight, useAveraging) of the plane (or ``roi`` =
    (x, y, w, h) of it)."""
    method = method_index(method)
    if kind == "rgb":
        return _per_channel(resize_plane, p, kind, dst_w, dst_h, method, roi, average, bg=bg)
    h, w = p.shape
    rx, ry, rw, rh = roi if roi is not None else (0, 0, w, h)
    if average and not (dst_w >= rw and dst_h >= rh):
        f = p.astype(np.float32) if kind != "float" else p
        f2 = _downsize(f, dst_w, dst_h, method, (rx, ry, rw, rh))
        return _from_float(f2, kind)
    return _resize(p, kind, dst_w, dst_h, method, (rx, ry, rw, rh), bg)


def _resize(p, kind, dst_w, dst_h, method, roi, bg=None):
    h, w = p.shape
    rx, ry, rw, rh = roi
    if rw == dst_w and rh == dst_h:
        return p[ry:ry + rh, rx:rx + rw].copy()
    if (w == 1 or h == 1) and method != NONE:
        return _resize_linearly(p, kind, dst_w, dst_h)
    src_cx = rx + rw / 2.0
    src_cy = ry + rh / 2.0
    dst_cx = dst_w / 2.0
    dst_cy = dst_h / 2.0
    xscale = dst_w / rw
    yscale = dst_h / rh
    if method != NONE:
        if dst_w != w:
            dst_cx += xscale / 4.0
        if dst_h != h:
            dst_cy += yscale / 4.0
    ys = (np.arange(dst_h, dtype=np.float64) - dst_cy) / yscale + src_cy
    xs = (np.arange(dst_w, dtype=np.float64) - dst_cx) / xscale + src_cx
    if method == BICUBIC:
        Y, X = np.meshgrid(ys, xs, indexing="ij")
        b = default_bg(kind) if bg is None else bg
        return _round_store(_bicubic(p, X, Y, kind, b), kind)
    if method == BILINEAR:
        ys = np.where(ys < 0.0, 0.0, ys)
        ys = np.where(ys >= h - 1.0, h - 1.001, ys)
        xs = np.where(xs < 0.0, 0.0, xs)
        xs = np.where(xs >= w - 1.0, w - 1.001, xs)
        Y, X = np.meshgrid(ys, xs, indexing="ij")
        v = _bilinear_core(p, X, Y, kind)
        if kind == "float":
            return v.astype(np.float32)
        return _store(_jint(v + 0.5), kind)
    yi = _jint(ys)
    xi = _jint(xs)
    return p[yi[:, None], xi[None, :]].copy()


class _DownsizeTable:
    """ij.process.DownsizeTable."""

    KERNEL_SIZES = (1, 2, 4)

    def __init__(self, src_size, src_origin, src_length, dst_size, method):
        self.src_origin, self.scale, self.method = src_origin, src_length / float(dst_size), method
        ks = self.KERNEL_SIZES[method]
        self.kernel_size = ks
        start = int(math.ceil(1e-8 + self._src(-0.5 * ks)))
        self.src_start = max(start, 0)
        end = int(math.floor(1e-8 + self._src(dst_size - 1 + 0.5 * ks)))
        self.src_end = src_size - 1 if end >= src_size else end
        n = (self.src_end - self.src_start + 1) * ks
        indices = [-1] * n
        weights = np.zeros(n, dtype=np.float32)
        for dst in range(dst_size):
            total = 0.0
            lowest = int(math.ceil(1e-8 + self._src(dst - 0.5 * ks)))
            highest = int(math.floor(-1e-8 + self._src(dst + 0.5 * ks)))
            for src in range(lowest, highest + 1):
                s = 0 if src < 0 else (src_size - 1 if src >= src_size else src)
                q = (s - self.src_start) * ks
                while indices[q] != -1 and indices[q] != dst:
                    q += 1
                indices[q] = dst
                wgt = self._kernel(dst - self._dst(src))
                total += float(wgt)
                weights[q] = np.float32(weights[q] + wgt)
            i0 = max((lowest - self.src_start) * ks, 0)
            i1 = min((highest - self.src_start) * ks + (ks - 1), n - 1)
            for i in range(i0, i1 + 1):
                if indices[i] == dst:
                    weights[i] = np.float32(float(weights[i]) / total)
        self.indices = np.array([0 if i == -1 else i for i in indices], dtype=np.int64)
        self.weights = weights

    def _src(self, d):
        return self.src_origin - 0.5 + (d + 0.5) * self.scale

    def _dst(self, s):
        return (s - self.src_origin + 0.5) / self.scale - 0.5

    def _kernel(self, x):
        if self.method == NONE:
            return np.float32(1.0)
        if self.method == BILINEAR:
            return np.float32(1.0) - np.float32(abs(x))
        return np.float32(cubic(x))

    def matrix(self, n_dst):
        """Per source point (src_start..src_end), rows of (dst index, weight) pairs."""
        ks = self.kernel_size
        n_src = self.src_end - self.src_start + 1
        return self.indices.reshape(n_src, ks), self.weights.reshape(n_src, ks)


def _downsize1d(f: np.ndarray, dst_len: int, method: int, roi, x_direction: bool):
    """FloatProcessor.downsize1D: float32 accumulation in ImageJ's order."""
    rx, ry, rw, rh = roi
    h, w = f.shape
    if x_direction:
        dt = _DownsizeTable(w, rx, rw, dst_len, method)
        lines = f[ry:ry + rh, :]                    # (n_lines, w)
    else:
        dt = _DownsizeTable(h, ry, rh, dst_len, method)
        lines = f[:, rx:rx + rw].T                  # (n_lines, h)
    idx, wts = dt.matrix(dst_len)
    out = np.zeros((lines.shape[0], dst_len), dtype=np.float32)
    for k, s in enumerate(range(dt.src_start, dt.src_end + 1)):
        v = lines[:, s].astype(np.float32)
        for i in range(dt.kernel_size):
            out[:, idx[k, i]] += v * wts[k, i]
    return out if x_direction else np.ascontiguousarray(out.T)


def _downsize(f: np.ndarray, dst_w: int, dst_h: int, method: int, roi) -> np.ndarray:
    rx, ry, rw, rh = roi
    cur, cur_roi = f, roi
    if dst_w < rw:
        cur = _downsize1d(cur, dst_w, method, cur_roi, True)
        cur_roi = (0, 0, dst_w, rh)
    if dst_h < rh:
        cur = _downsize1d(cur, dst_h, method, cur_roi, False)
        cur_roi = (0, 0, cur.shape[1], cur.shape[0])
    if cur.shape != (dst_h, dst_w):
        full = (0, 0, cur.shape[1], cur.shape[0]) if cur is not f else roi
        cur = _resize(cur, "float", dst_w, dst_h, method, full)
    return cur


# ---------------------------------------------------------------------------------- scale
def scale_plane(p: np.ndarray, kind: str, xscale: float, yscale: float, method=BILINEAR,
                roi: tuple | None = None, bg=None, display_min: float = 0.0) -> np.ndarray:
    """ImageProcessor.scale (in place, same size); ``display_min`` is the fill value of
    16/32-bit images (ImageJ uses getMin())."""
    method = method_index(method)
    if kind == "rgb":
        bgs = _rgb_bg(bg)
        return np.stack([scale_plane(np.ascontiguousarray(p[..., k]), "byte", xscale, yscale,
                                     method, roi, bgs[k]) for k in range(3)], axis=-1)
    h, w = p.shape
    rx, ry, rw, rh = roi if roi is not None else (0, 0, w, h)
    xc = rx + rw / 2.0
    yc = ry + rh / 2.0
    if xscale > 1.0 and yscale > 1.0:
        xmin = max(int(xc - (xc - rx) * xscale), 0)
        xmax = min(xmin + int(rw * xscale) - 1, w - 1)
        ymin = max(int(yc - (yc - ry) * yscale), 0)
        ymax = min(ymin + int(rh * yscale) - 1, h - 1)
    else:
        xmin, xmax, ymin, ymax = rx, rx + rw - 1, ry, ry + rh - 1
    out = p.copy()
    if bg is None:
        bg = default_bg(kind)
    yy = np.arange(ymin, ymax + 1, dtype=np.float64)
    xx = np.arange(xmin, xmax + 1, dtype=np.float64)
    ys = (yy - yc) / yscale + yc
    xs = (xx - xc) / xscale + xc
    if method == BICUBIC:
        Y, X = np.meshgrid(ys, xs, indexing="ij")
        ip2_bg = bg if kind == "byte" else 0.0
        out[ymin:ymax + 1, xmin:xmax + 1] = _round_store(_bicubic(p, X, Y, kind, ip2_bg), kind)
        return out
    check = xscale < 1.0 or yscale < 1.0
    ysi = _jint(ys)
    ysc = np.where(ys < 0.0, 0.0, ys)
    ysc = np.where(ysc >= h - 1.0, h - 1.001, ysc)
    xsi = _jint(xs)
    Ysi, Xsi = np.meshgrid(ysi, xsi, indexing="ij")
    if method == BILINEAR:
        xsc = np.where(xs < 0.0, 0.0, xs)
        xsc = np.where(xsc >= w - 1.0, w - 1.001, xsc)
        Y, X = np.meshgrid(ysc, xsc, indexing="ij")
        v = _bilinear_core(p, X, Y, kind)
        vals = v.astype(np.float32) if kind == "float" else _store(_jint(v + 0.5), kind)
    else:
        rows = _jint(ysc)
        vals = p[rows[:, None], np.clip(xsi, 0, w - 1)[None, :]]
    if check:
        outside = (Xsi < xmin) | (Xsi > xmax) | (Ysi < ymin) | (Ysi > ymax)
        fill = bg if kind == "byte" else (int(display_min) if kind == "short"
                                          else np.float32(display_min))
        vals = np.where(outside, np.asarray(fill).astype(p.dtype), vals)
    out[ymin:ymax + 1, xmin:xmax + 1] = vals
    return out


# ---------------------------------------------------------------------------------- rotate
def rotate_plane(p: np.ndarray, kind: str, angle: float, method=BILINEAR,
                 roi: tuple | None = None, bg=None) -> np.ndarray:
    """ImageProcessor.rotate(angle) clockwise about the centre of the plane or ``roi``."""
    method = method_index(method)
    if angle % 360 == 0 and kind in ("byte", "rgb"):
        return p.copy()
    if kind == "rgb":
        bgs = _rgb_bg(bg)
        return np.stack([rotate_plane(np.ascontiguousarray(p[..., k]), "byte", angle, method,
                                      roi, bgs[k]) for k in range(3)], axis=-1)
    h, w = p.shape
    rx, ry, rw, rh = roi if roi is not None else (0, 0, w, h)
    if bg is None:
        bg = default_bg(kind)
    cx = rx + (rw - 1) / 2.0
    cy = ry + (rh - 1) / 2.0
    rad = -angle / (180.0 / math.pi)
    ca = math.cos(rad)
    sa = math.sin(rad)
    tmp1 = cy * sa - cx * ca
    tmp2 = -cx * sa - cy * ca
    yy = np.arange(ry, ry + rh, dtype=np.float64)[:, None]
    xx = np.arange(rx, rx + rw, dtype=np.float64)[None, :]
    tmp3 = tmp1 - yy * sa + cx
    tmp4 = tmp2 + yy * ca + cy
    xs = xx * ca + tmp3
    ys = xx * sa + tmp4
    out = p.copy()
    if method == BICUBIC:
        ip2_bg = 0.0 if kind == "short" else bg
        out[ry:ry + rh, rx:rx + rw] = _round_store(_bicubic(p, xs, ys, kind, ip2_bg), kind)
        return out
    inside = (xs >= -0.01) & (xs < w) & (ys >= -0.01) & (ys < h)
    if method == BILINEAR:
        xc = np.where(xs < 0.0, 0.0, xs)
        xc = np.where(xc >= w - 1.0, w - 1.001, xc)
        yc = np.where(ys < 0.0, 0.0, ys)
        yc = np.where(yc >= h - 1.0, h - 1.001, yc)
        xc = np.where(inside, xc, 0.0)
        yc = np.where(inside, yc, 0.0)
        v = _bilinear_core(p, xc, yc, kind)
        vals = v.astype(np.float32) if kind == "float" else _store(_jint(v + 0.5), kind)
    else:
        ixs = np.minimum(_jint(xs + 0.5), w - 1)
        iys = np.minimum(_jint(ys + 0.5), h - 1)
        ixs = np.where(inside, ixs, 0)
        iys = np.where(inside, iys, 0)
        vals = p[iys, ixs]
    fill = np.asarray(bg).astype(np.float32) if kind == "float" else \
        np.asarray(int(bg)).astype(p.dtype)
    out[ry:ry + rh, rx:rx + rw] = np.where(inside, vals, fill)
    return out


def rotated_bounds(width: int, height: int, angle: float) -> tuple[int, int]:
    """Size of the canvas that holds the image rotated by ``angle`` (Rotator.enlargeCanvas:
    the rotated full-image rectangle's bounds, rounded, never smaller than the image)."""
    cx, cy = width / 2.0, height / 2.0
    theta = -angle * math.pi / 180.0
    ca, sa = math.cos(theta), math.sin(theta)
    xs, ys = [], []
    for x, y in ((0, 0), (width, 0), (width, height), (0, height)):
        dx, dy = x - cx, y - cy
        xs.append(cx + dx * ca - dy * sa)
        ys.append(cy + dx * sa + dy * ca)
    bw = max(xs) - min(xs)
    bh = max(ys) - min(ys)
    return max(int(round(bw)), width), max(int(round(bh)), height)


# ---------------------------------------------------------------------------------- translate
def translate_plane(p: np.ndarray, kind: str, dx: float, dy: float, method=NONE,
                    roi: tuple | None = None) -> np.ndarray:
    """ImageProcessor.translate (background 0)."""
    method = method_index(method)
    h, w = p.shape[:2]
    rx, ry, rw, rh = roi if roi is not None else (0, 0, w, h)
    out = p.copy()
    yy = np.arange(ry, ry + rh)[:, None]
    xx = np.arange(rx, rx + rw)[None, :]
    if (dx == int(dx) and dy == int(dy)) or method == NONE:
        sx = xx - int(dx)
        sy = yy - int(dy)
        inside = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
        vals = p[np.clip(sy, 0, h - 1), np.clip(sx, 0, w - 1)]
        zero = np.zeros((), dtype=p.dtype)
        if kind == "rgb":
            inside = inside[..., None]
        out[ry:ry + rh, rx:rx + rw] = np.where(inside, vals, zero)
        return out
    if kind == "rgb":
        if method == BICUBIC:
            return np.stack([translate_plane(np.ascontiguousarray(p[..., k]), "byte", dx, dy,
                                             method, roi) for k in range(3)], axis=-1)
        chans = [translate_plane(np.ascontiguousarray(p[..., k]), "byte", dx, dy, BILINEAR,
                                 roi) for k in range(3)]
        return np.stack(chans, axis=-1)
    xs = (xx - dx).astype(np.float64) + np.zeros((rh, 1))
    ys = (yy - dy).astype(np.float64) + np.zeros((1, rw))
    if method == BILINEAR:
        out_of = (xs < 0.0) | (ys < 0.0) | (xs >= w - 1) | (ys >= h - 1)
        v = _bilinear_core(p, np.where(out_of, 0.0, xs), np.where(out_of, 0.0, ys), kind)
        if kind == "float":
            vals = np.where(out_of, np.float32(0), v.astype(np.float32))
        else:
            hi = 255 if kind == "byte" else 65535
            vals = _store(np.where(out_of, 0, np.clip(np.floor(v + 0.5), 0, hi)), kind)
    else:
        v = _bicubic(p, xs, ys, kind, 0.0)
        vals = _round_store(v, kind)
    out[ry:ry + rh, rx:rx + rw] = vals
    return out


# ---------------------------------------------------------------------------------- binning
BIN_METHODS = ("Average", "Median", "Min", "Max", "Sum")


def _bin_float(f: np.ndarray, xs: int, ys: int, method: str, max_value: float) -> np.ndarray:
    h, w = f.shape[0] // ys, f.shape[1] // xs
    blocks = f[:h * ys, :w * xs].reshape(h, ys, w, xs).transpose(0, 2, 1, 3).reshape(h, w, -1)
    blocks = blocks.astype(np.float32)
    if method in ("Average", "Sum"):
        acc = np.zeros((h, w), dtype=np.float32)
        for k in range(blocks.shape[-1]):
            acc = acc + blocks[..., k]
        if method == "Average":
            return acc / np.float32(xs * ys)
        if max_value > 0:
            acc = np.where(acc > max_value, np.float32(max_value), acc)
        return acc
    big = np.float32(np.finfo(np.float32).max)
    if method == "Min":     # NaN never compares smaller (Binner.getMin)
        return np.min(np.where(np.isnan(blocks), big, blocks), axis=-1).astype(np.float32)
    if method == "Max":
        return np.max(np.where(np.isnan(blocks), -big, blocks), axis=-1).astype(np.float32)
    # Binner.getMedian: repeatedly zero the maximum (half+1 times), then take the maximum
    vals = blocks.copy()
    n = vals.shape[-1]
    for _ in range(n // 2 + 1):
        m = np.max(np.where(vals > 0, vals, 0), axis=-1, keepdims=True)
        has = m > 0
        first = np.where(has[..., 0], np.argmax((vals == m) & has, axis=-1), 0)
        sel = np.zeros_like(vals, dtype=bool)
        np.put_along_axis(sel, first[..., None], True, axis=-1)
        vals = np.where(sel, np.float32(0), vals)
    return np.max(np.where(np.isnan(vals), -big, vals), axis=-1).astype(np.float32)


def bin_plane(p: np.ndarray, kind: str, xs: int, ys: int, method: str = "Average",
              inverted_lut: bool = False, max_value: float | None = None) -> np.ndarray:
    """Binner.shrink of one plane (8/16-bit results stored with (v+0.5f) truncation; RGB sums
    are not clipped and wrap, as in ImageJ)."""
    if kind == "rgb":
        return np.stack([bin_plane(np.ascontiguousarray(p[..., k]), "byte", xs, ys, method,
                                   max_value=0.0) for k in range(3)], axis=-1)
    q = p
    if inverted_lut and kind == "byte":
        q = 255 - p
    if max_value is None:
        max_value = {"byte": 255.0, "short": 65535.0}.get(kind, 0.0)
    f = _bin_float(q.astype(np.float32), xs, ys, method, max_value)
    if kind == "float":
        out = f.astype(np.float32)
    else:
        out = _store(_jint((f + np.float32(0.5)).astype(np.float32)), kind)
    if inverted_lut and kind == "byte":
        out = 255 - out
    return out


def bin_z(planes: np.ndarray, kind: str, zs: int, method: str) -> np.ndarray:
    """Binner.shrinkZ: bin along z (median falls back to average); planes (Z, Y, X[, 3])."""
    if method == "Median":
        method = "Average"
    d = planes.shape[0]
    d2 = d // zs
    if kind == "rgb":
        return np.stack([bin_z(planes[..., k], "byte", zs, method) for k in range(3)], axis=-1)
    f = planes[:d2 * zs].astype(np.float32)
    # x = z in an (h, d) float processor, shrink by (zs, 1); then setVoxel stores
    blocks = f.reshape(d2, zs, *f.shape[1:])
    if method in ("Average", "Sum"):
        acc = np.zeros(blocks.shape[:1] + blocks.shape[2:], dtype=np.float32)
        for k in range(zs):
            acc = acc + blocks[:, k]
        res = acc / np.float32(zs) if method == "Average" else acc
    elif method == "Min":
        res = blocks.min(axis=1)
    else:
        res = blocks.max(axis=1)
    if kind == "float":
        return res.astype(np.float32)
    hi = 255 if kind == "byte" else 65535
    # ImageStack.setVoxel: clamps and rounds for 8/16-bit
    return _store(np.clip(_jint(res.astype(np.float64) + 0.5), 0, hi), kind)


# ---------------------------------------------------------------------------------- simple ops
def flip_horizontal(p: np.ndarray, roi=None) -> np.ndarray:
    out = p.copy()
    x, y, w, h = roi if roi is not None else (0, 0, p.shape[1], p.shape[0])
    out[y:y + h, x:x + w] = p[y:y + h, x:x + w][:, ::-1]
    return out


def flip_vertical(p: np.ndarray, roi=None) -> np.ndarray:
    out = p.copy()
    x, y, w, h = roi if roi is not None else (0, 0, p.shape[1], p.shape[0])
    out[y:y + h, x:x + w] = p[y:y + h, x:x + w][::-1]
    return out


def rotate_right(p: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(p, -1, axes=(0, 1)))


def rotate_left(p: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(p, 1, axes=(0, 1)))


CANVAS_POSITIONS = ("Top-Left", "Top-Center", "Top-Right", "Center-Left", "Center",
                    "Center-Right", "Bottom-Left", "Bottom-Center", "Bottom-Right")


def canvas_offsets(w_old, h_old, w_new, h_new, position: str) -> tuple[int, int]:
    xc, xr = int((w_new - w_old) / 2), w_new - w_old
    yc, yb = int((h_new - h_old) / 2), h_new - h_old
    i = CANVAS_POSITIONS.index(position) if position in CANVAS_POSITIONS else 4
    xoff = (0, xc, xr, 0, xc, xr, 0, xc, xr)[i]
    yoff = (0, 0, 0, yc, yc, yc, yb, yb, yb)[i]
    return xoff, yoff


def insert(dst: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    """ImageProcessor.insert: copy ``src`` at (x, y), clipped to ``dst``."""
    h, w = dst.shape[:2]
    sh, sw = src.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + sw, w), min(y + sh, h)
    if x1 <= x0 or y1 <= y0:
        return
    dst[y0:y1, x0:x1] = src[y0 - y:y1 - y, x0 - x:x1 - x]


def canvas_plane(p: np.ndarray, w_new: int, h_new: int, xoff: int, yoff: int, fill) -> np.ndarray:
    shape = (h_new, w_new) + p.shape[2:]
    out = np.empty(shape, dtype=p.dtype)
    out[...] = fill
    insert(out, p, xoff, yoff)
    return out


# ---------------------------------------------------------------------------------- ImageDoc
def map_planes(doc: ImageDoc, fn, all_planes: bool = True, channel: int | None = None
               ) -> np.ndarray:
    """New pixel array with ``fn(plane)`` applied to every plane (all same output shape)."""
    first = None
    out = None
    for i in range(doc.stack_size):
        c, z, t = doc.position_of(i)
        if (not all_planes and (c, z, t) != doc.position) or \
                (channel is not None and c != channel):
            continue
        r = fn(doc.plane(c, z, t))
        if out is None:
            first = r
            out = np.zeros(doc.pixels.shape[:3] + r.shape, dtype=r.dtype)
            if r.shape == doc.plane().shape:
                out[...] = doc.pixels
        out[t, z, c] = r
    if out is None:
        return doc.pixels.copy()
    del first
    return out


def background_value(doc: ImageDoc, color: str, fill_with_background: bool):
    """Background fill value of the image type (Rotator/Scaler: 0 unless "fill with background
    color"; the colour is mapped like ImageProcessor.setBackgroundColor)."""
    kind = kind_of(doc)
    if not fill_with_background:
        return (0, 0, 0) if kind == "rgb" else 0
    c = color.lstrip("#")[-6:]
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    if kind == "rgb":
        return (r, g, b)
    lut = doc.lut.as_array().astype(np.int64)
    best = int(np.argmin((lut[:, 0] - r) ** 2 + (lut[:, 1] - g) ** 2 + (lut[:, 2] - b) ** 2))
    if kind == "byte":
        return best
    lo, hi = doc.display_range
    return lo + (hi - lo) * (best / 255.0)


def crop_doc(doc: ImageDoc, rect: tuple[int, int, int, int]) -> None:
    x, y, w, h = rect
    x0, y0 = max(int(x), 0), max(int(y), 0)
    x1, y1 = min(int(x + w), doc.width), min(int(y + h), doc.height)
    ranges = list(doc.display_ranges)
    luts = [lut.copy() for lut in doc.luts]
    labels = list(doc.slice_labels)
    doc.replace_pixels(doc.pixels[:, :, :, y0:y1, x0:x1].copy(), doc.type)
    doc.luts, doc.slice_labels = luts, labels
    if doc.bit_depth in (16, 32):
        doc.display_ranges = ranges
    cal = doc.calibration
    if cal.x_origin != 0 or cal.y_origin != 0:
        cal.x_origin -= x0
        cal.y_origin -= y0
    if doc.overlay is not None:
        for roi in doc.overlay:
            roi.translate(-x0, -y0)


def image_to_results(plane: np.ndarray, rect=None):
    """Image > Transform > Image to Results: one column per x ("X0", "X1", ...)."""
    from rsstatchem.core.imaging.results import ResultsTable
    x, y, w, h = rect if rect is not None else (0, 0, plane.shape[1], plane.shape[0])
    sub = plane[y:y + h, x:x + w]
    rt = ResultsTable.results("Results")
    if sub.ndim == 3:
        sub = (sub[..., 0].astype(np.int64) << 16) | (sub[..., 1].astype(np.int64) << 8) | \
            sub[..., 2]
    for row in sub:
        rt.increment_counter()
        for i, v in enumerate(row):
            rt.add_value(f"X{x + i}", float(v))
    return rt


def results_to_image(table) -> ImageDoc:
    """Image > Transform > Results to Image: numeric columns as a 32-bit image."""
    cols = [h for h in table.headings if h not in ("Label",)]
    if not cols or len(table) == 0:
        raise ValueError("Results table is empty")
    data = np.stack([np.asarray(table.column(c), dtype=np.float64) for c in cols], axis=1)
    return ImageDoc.from_array(data.astype(np.float32), "YX", title="Results Table")

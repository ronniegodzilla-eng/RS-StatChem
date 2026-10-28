"""Process > FFT: Fast Hartley Transform, power spectrum, inverse FFT with masks, FFT options
outputs, Bandpass Filter, Custom Filter, FD Math, Swap Quadrants. **WP-B.**

Ported from ImageJ 1.54p ij/process/FHT.java, ij/plugin/FFT.java, ij/plugin/filter/FFTFilter.java,
ij/plugin/filter/FFTCustomFilter.java and ij/plugin/FFTMath.java (public domain, NIH;
W. Rasband, A. Reeves, J. Walter, J. Wesner, M. Schmid et al.). The FHT itself goes back to
Arlo Reeves' NIH Image Pascal code (the FHT patent was placed in the public domain by Stanford
in 1995).

The arithmetic follows the Java code operation by operation in float32 (Java ``float``) so the
results are bit-identical; loops are vectorised only where every element is computed with the
same operations in the same order (the butterflies of one FHT stage touch disjoint elements).
Java ``(int)`` casts of floats (truncation, NaN -> 0, saturation) are reproduced by
:func:`java_int`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

F32 = np.float32
FLOAT_MAX = float(np.finfo(np.float32).max)
FLOAT_MIN_VALUE = float(np.float32(1.4e-45))   # Java Float.MIN_VALUE (smallest denormal)


# ------------------------------------------------------------------------------- Java helpers
def java_int(values) -> np.ndarray:
    """Java ``(int)x`` for float/double arrays: truncation toward zero, NaN -> 0, saturation
    at the int range."""
    v = np.asarray(values, dtype=np.float64)
    out = np.where(np.isnan(v), 0.0, np.clip(v, -2147483648.0, 2147483647.0))
    return np.trunc(out).astype(np.int64)


def is_power_of_2(n: int) -> bool:
    i = 2
    while i < n:
        i *= 2
    return i == n


def rgb_to_hsb(rgb: np.ndarray):
    """``java.awt.Color.RGBtoHSB`` for an (..., 3) uint8 array -> (hue, sat, bri) float32."""
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    bri = cmax.astype(F32) / F32(255.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sat = np.where(cmax != 0, (cmax - cmin).astype(F32) / cmax.astype(F32), F32(0))
        span = (cmax - cmin).astype(F32)
        redc = (cmax - r).astype(F32) / span
        greenc = (cmax - g).astype(F32) / span
        bluec = (cmax - b).astype(F32) / span
        hue = np.where(r == cmax, bluec - greenc,
                       np.where(g == cmax, (F32(2.0) + redc) - bluec,
                                (F32(4.0) + greenc) - redc)).astype(F32)
        hue = (hue / F32(6.0)).astype(F32)
        hue = np.where(hue < 0, hue + F32(1.0), hue).astype(F32)
    hue = np.where(sat == 0, F32(0), hue).astype(F32)
    return hue, sat.astype(F32), bri.astype(F32)


def hsb_to_rgb(hue, sat, bri) -> np.ndarray:
    """``java.awt.Color.HSBtoRGB`` (float32 inputs) -> (..., 3) uint8."""
    hue = np.asarray(hue, F32)
    sat = np.asarray(sat, F32)
    bri = np.asarray(bri, F32)

    def to255(v):
        return java_int((v * F32(255.0) + F32(0.5)).astype(F32))

    h = ((hue - np.floor(hue).astype(F32)) * F32(6.0)).astype(F32)
    f = (h - np.floor(h).astype(F32)).astype(F32)
    p = (bri * (F32(1.0) - sat)).astype(F32)
    q = (bri * (F32(1.0) - (sat * f).astype(F32))).astype(F32)
    t = (bri * (F32(1.0) - (sat * (F32(1.0) - f)).astype(F32))).astype(F32)
    B, P, Q, T = to255(bri), to255(p), to255(q), to255(t)
    ih = java_int(h)
    zero = np.zeros_like(B)
    r = np.select([ih == 0, ih == 1, ih == 2, ih == 3, ih == 4, ih == 5], [B, Q, P, P, T, B], zero)
    g = np.select([ih == 0, ih == 1, ih == 2, ih == 3, ih == 4, ih == 5], [T, B, B, Q, P, P], zero)
    bl = np.select([ih == 0, ih == 1, ih == 2, ih == 3, ih == 4, ih == 5], [P, P, T, B, B, Q], zero)
    gray = sat == 0
    r = np.where(gray, B, r)
    g = np.where(gray, B, g)
    bl = np.where(gray, B, bl)
    return np.stack([r, g, bl], axis=-1).astype(np.uint8)


def get_brightness(rgb: np.ndarray) -> np.ndarray:
    """``ColorProcessor.getBrightness`` (HSB brightness, float32 0..1)."""
    return rgb_to_hsb(rgb)[2]


def set_brightness(rgb: np.ndarray, brightness: np.ndarray) -> np.ndarray:
    """``ColorProcessor.setBrightness``: new RGB with the brightness replaced (clamped 0..1)."""
    h, s, _ = rgb_to_hsb(rgb)
    b = np.clip(np.asarray(brightness, F32), F32(0), F32(1)).astype(F32)
    b = np.where(np.isnan(brightness), np.asarray(brightness, F32), b)
    return hsb_to_rgb(h, s, b)


def float_to_byte_noscale(fp: np.ndarray) -> np.ndarray:
    """``ByteProcessor.setPixels(0, fp)`` (convertToByte(false)): (byte)(v+0.5f) clamped."""
    v = (np.asarray(fp, F32) + F32(0.5)).astype(F32)
    v = np.where(v < 0, F32(0), np.where(v > 255, F32(255), v))
    return java_int(v).astype(np.uint8)


def float_to_short_noscale(fp: np.ndarray) -> np.ndarray:
    """``TypeConverter.convertFloatToShort`` without scaling."""
    v = np.asarray(fp, F32).astype(np.float64)
    v = np.where(v < 0.0, 0.0, np.where(v > 65535.0, 65535.0, v))
    return (java_int(v + 0.5) & 0xFFFF).astype(np.uint16)


def float_min_max(fp: np.ndarray) -> tuple[float, float]:
    """``FloatProcessor.findMinAndMax`` (ignores NaN, keeps infinities)."""
    a = np.asarray(fp, F32)
    finite = a[~np.isnan(a)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(finite.min()), float(finite.max())


def float_to_byte_scaled(fp: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """``FloatProcessor.create8BitImage`` (convertToByte(true)) with display range min..max."""
    a = np.asarray(fp, F32).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        scale = 255.0 / (vmax - vmin) if vmax != vmin else math.inf
        value = a - vmin
        value = np.where(value < 0.0, 0.0, value)
        iv = java_int(value * scale + 0.5)
    iv = np.where(iv > 255, 255, iv)
    return (iv & 0xFF).astype(np.uint8)


def float_to_short_scaled(fp: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """``TypeConverter.convertFloatToShort`` with scaling to the display range."""
    a = np.asarray(fp, F32).astype(np.float64)
    scale = 1.0 if (vmax - vmin) == 0.0 else 65535.0 / (vmax - vmin)
    with np.errstate(invalid="ignore"):
        v = (a - vmin) * scale
        v = np.where(v < 0.0, 0.0, np.where(v > 65535.0, 65535.0, v))
        return (java_int(v + 0.5) & 0xFFFF).astype(np.uint16)


def float_histogram(fp: np.ndarray, nbins: int = 256):
    """``FloatStatistics`` histogram (MIN_MAX): (histogram, histMin, binSize, pixelCount)."""
    v = np.asarray(fp, F32).ravel().astype(np.float64)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return np.zeros(nbins, np.int64), 0.0, 0.0, 0
    hmin, hmax = float(v.min()), float(v.max())
    bin_size = (hmax - hmin) / nbins
    sel = v[(v >= hmin) & (v <= hmax)]
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = nbins / (hmax - hmin) if hmax != hmin else math.inf
        idx = java_int(scale * (sel - hmin))
    idx = np.where(idx >= nbins, nbins - 1, idx)
    idx = np.where(idx < 0, 0, idx)
    hist = np.bincount(idx, minlength=nbins)[:nbins]
    return hist, hmin, bin_size, int(sel.size)


def stretch_min_max(hist, hist_min: float, bin_size: float, pixel_count: int,
                    saturated: float):
    """``ContrastEnhancer.getMinAndMax`` + the min/max of ``stretchHistogram``; None when
    hmax <= hmin (ImageJ then leaves the display range unchanged)."""
    hsize = len(hist)
    threshold = int(pixel_count * saturated / 200.0) if saturated > 0.0 else 0
    i, count = -1, 0
    while True:
        i += 1
        count += int(hist[i])
        if count > threshold or i >= hsize - 1:
            break
    hmin = i
    i, count = hsize, 0
    while True:
        i -= 1
        count += int(hist[i])
        if count > threshold or i <= 0:
            break
    hmax = i
    if hmax <= hmin:
        return None
    return hist_min + hmin * bin_size, hist_min + hmax * bin_size


def float_stretch_range(fp: np.ndarray, saturated: float):
    """Display range ``ContrastEnhancer.stretchHistogram(imp, saturated)`` gives a 32-bit
    plane (as Java floats), or the plane min/max when the histogram cannot be stretched."""
    hist, hmin, bs, n = float_histogram(fp)
    r = stretch_min_max(hist, hmin, bs, n, saturated)
    if r is None or (r[0] == 0.0 and r[1] == 0.0):
        return float_min_max(fp)
    return float(F32(r[0])), float(F32(r[1]))


# ------------------------------------------------------------------------------- FHT core
_TABLES: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _log2(n: int) -> int:
    return n.bit_length() - 1


def _tables(n: int):
    """Sine/cosine tables (theta accumulated like ``makeSinCosTables``) and bit reversal."""
    if n in _TABLES:
        return _TABLES[n]
    q = n // 4
    d_theta = 2.0 * math.pi / n
    theta = np.cumsum(np.concatenate([[0.0], np.full(max(q - 1, 0), d_theta)]))[:q]
    C = np.cos(theta).astype(F32)
    S = np.sin(theta).astype(F32)
    nlog2 = _log2(n)
    idx = np.arange(n)
    rev = np.zeros(n, dtype=np.int64)
    for i in range(nlog2 + 1):
        bit = (idx >> i) & 1
        shift = nlog2 - i - 1
        if shift >= 0:
            rev |= bit << shift
    _TABLES[n] = (C, S, rev)
    return _TABLES[n]


def dfht3(x: np.ndarray, inverse: bool = False) -> np.ndarray:
    """``FHT.dfht3`` applied to every row of ``x`` (rows, n), float32; returns a new array."""
    x = np.asarray(x, F32)
    rows, n = x.shape
    C, S, rev = _tables(n)
    nlog2 = _log2(n)
    x = np.ascontiguousarray(x[:, rev])
    if n >= 4:
        g = x.reshape(rows, n // 4, 4)
        a, b, c, d = g[..., 0].copy(), g[..., 1].copy(), g[..., 2].copy(), g[..., 3].copy()
        rt1, rt2, rt3, rt4 = a + b, a - b, c + d, c - d
        g[..., 0] = rt1 + rt3
        g[..., 1] = rt2 + rt4
        g[..., 2] = rt1 - rt3
        g[..., 3] = rt2 - rt4
    if nlog2 > 2:
        gp, nbfs, ngps = 4, 2, n // 8
        for _stage in range(2, nlog2):
            G = x.reshape(rows, ngps, 2 * gp)
            for k0, k1 in ((0, gp), (gp // 2, gp // 2 + gp)):
                v0 = G[:, :, k0].copy()
                v1 = G[:, :, k1].copy()
                G[:, :, k0] = v0 + v1
                G[:, :, k1] = v0 - v1
            if nbfs > 1:
                bf = np.arange(1, nbfs)
                a1, a2 = G[:, :, bf], G[:, :, bf + gp]
                a3, a4 = G[:, :, gp - bf], G[:, :, 2 * gp - bf]
                cs, sn = C[bf * ngps], S[bf * ngps]
                rt1 = a2 * cs + a4 * sn
                rt2 = a4 * cs - a2 * sn
                G[:, :, bf + gp] = a1 - rt1
                G[:, :, bf] = a1 + rt1
                G[:, :, 2 * gp - bf] = a3 + rt2
                G[:, :, gp - bf] = a3 - rt2
            gp *= 2
            nbfs *= 2
            ngps //= 2
    if inverse:
        x = (x / F32(n)).astype(F32)
    return x


def fht_transform(data: np.ndarray, inverse: bool = False) -> np.ndarray:
    """``FHT.rc2DFHT``: 2-D Hartley transform of a square power-of-2 float32 array."""
    x = np.asarray(data, F32)
    n = x.shape[0]
    if x.shape != (n, n) or not is_power_of_2(n):
        raise ValueError(f"Image not power of 2 size or not square: {x.shape[1]}x{x.shape[0]}")
    x = dfht3(x, inverse)
    x = np.ascontiguousarray(x.T)
    x = dfht3(x, inverse)
    x = np.ascontiguousarray(x.T)
    h = n // 2
    r = np.arange(h + 1)
    mr = (n - r) % n
    R, M = r[:, None], mr[:, None]
    c, mc = r[None, :], mr[None, :]
    A, B, Cc, D = x[R, c], x[M, c], x[R, mc], x[M, mc]
    E = (((A + D) - (B + Cc)) / F32(2)).astype(F32)
    x[R, c] = A - E
    x[M, c] = B + E
    x[R, mc] = Cc + E
    x[M, mc] = D - E
    return x


def _mirror(fht: np.ndarray) -> np.ndarray:
    """fht[(N-r)%N, (N-c)%N]."""
    n = fht.shape[0]
    idx = (-np.arange(n)) % n
    return fht[np.ix_(idx, idx)]


def raw_power_spectrum(fht: np.ndarray) -> np.ndarray:
    """``FHT.getRawPowerSpectrum`` (not quadrant-swapped)."""
    a = np.asarray(fht, F32)
    m = _mirror(a)
    return (((a * a) + (m * m)) / F32(2)).astype(F32)


def power_spectrum(fht: np.ndarray) -> np.ndarray:
    """``FHT.getPowerSpectrum``: 8-bit log-scaled 1..254, quadrants swapped."""
    fps = raw_power_spectrum(fht)
    vmin, vmax = FLOAT_MAX, FLOAT_MIN_VALUE
    finite = fps[~np.isnan(fps)]
    if finite.size:
        vmin = min(vmin, float(finite.min()))
        vmax = max(vmax, float(finite.max()))
    with np.errstate(divide="ignore", invalid="ignore"):
        mx = F32(math.log(vmax)) if vmax > 0 else F32(-math.inf if vmax == 0 else math.nan)
        mn = F32(math.log(vmin)) if vmin > 0 else F32(-math.inf if vmin == 0 else math.nan)
        if np.isnan(mn) or (mx - mn) > 50:
            mn = F32(mx - F32(50))
        scale = F32(253.999 / float(F32(mx - mn)))
        r = ((np.log(fps.astype(np.float64)).astype(F32) - mn) * scale).astype(F32)
        r = np.where(np.isnan(r) | (r < 0), F32(0), r).astype(F32)
        ps = (java_int((r + F32(1)).astype(F32)) & 0xFF).astype(np.uint8)
    return swap_quadrants(ps)


def amplitude_spectrum(fht: np.ndarray) -> np.ndarray:
    a = np.asarray(fht, F32)
    m = _mirror(a)
    amp = np.sqrt(((a * a) + (m * m)).astype(np.float64)).astype(F32)
    return swap_quadrants(amp)


def complex_transform(fht: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``FHT.getComplexTransform``: (real, imaginary), quadrant-swapped."""
    a = np.asarray(fht, F32)
    m = _mirror(a)
    re = ((a + m) * F32(0.5)).astype(F32)
    im = ((-a + m) * F32(0.5)).astype(F32)
    return swap_quadrants(re), swap_quadrants(im)


def swap_quadrants(a: np.ndarray) -> np.ndarray:
    """``FFT.swapQuadrants`` (crop/insert of the four quadrants; works for odd sizes too)."""
    out = np.array(a, copy=True)
    s = out.shape[1] // 2
    t1 = out[0:s, s:2 * s].copy()
    t2 = out[s:2 * s, 0:s].copy()
    out[s:2 * s, 0:s] = t1
    out[0:s, s:2 * s] = t2
    t1 = out[0:s, 0:s].copy()
    t2 = out[s:2 * s, s:2 * s].copy()
    out[s:2 * s, s:2 * s] = t1
    out[0:s, 0:s] = t2
    return out


def fht_multiply(h1: np.ndarray, h2: np.ndarray, conjugate: bool = False) -> np.ndarray:
    """``FHT.multiply`` / ``conjugateMultiply`` (Hartley domain)."""
    a1 = np.asarray(h1, F32)
    a2 = np.asarray(h2, F32)
    m1, m2 = _mirror(a1), _mirror(a2)
    h2e = ((a2 + m2) / F32(2)).astype(F32).astype(np.float64)
    h2o = ((a2 - m2) / F32(2)).astype(F32).astype(np.float64)
    if conjugate:
        out = a1.astype(np.float64) * h2e - m1.astype(np.float64) * h2o
    else:
        out = a1.astype(np.float64) * h2e + m1.astype(np.float64) * h2o
    return out.astype(F32)


def fht_divide(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    """``FHT.divide`` (Hartley domain deconvolution)."""
    a1 = np.asarray(h1, F32)
    a2 = np.asarray(h2, F32)
    m1, m2 = _mirror(a1), _mirror(a2)
    mag = ((a2 * a2) + (m2 * m2)).astype(F32).astype(np.float64)
    mag = np.where(mag < 1e-20, 1e-20, mag)
    h2e = (a2 + m2).astype(F32).astype(np.float64)
    h2o = (a2 - m2).astype(F32).astype(np.float64)
    tmp = a1.astype(np.float64) * h2e - m1.astype(np.float64) * h2o
    return (tmp / mag).astype(F32)


# ------------------------------------------------------------------------------- FFT plugin
@dataclass
class FHTData:
    """What ImageJ keeps as the "FHT" property of an "FFT of ..." image."""
    data: np.ndarray                      # frequency-domain FHT, float32 N x N
    original_width: int = 0               # >0 only when the input was padded
    original_height: int = 0
    original_bit_depth: int = 0           # 8, 16, 24 or 32
    original_lut: object = None           # LUT of the input (not for RGB)
    rgb: np.ndarray | None = None         # RGB input (H, W, 3), brightness is replaced
    quadrant_swap_needed: bool = False
    power_spectrum_mean: float = 0.0
    calibration: object = None
    extra: dict = field(default_factory=dict)

    def copy(self) -> "FHTData":
        return FHTData(self.data.copy(), self.original_width, self.original_height,
                       self.original_bit_depth, self.original_lut,
                       None if self.rgb is None else self.rgb.copy(),
                       self.quadrant_swap_needed, self.power_spectrum_mean, self.calibration,
                       dict(self.extra))


def plane_mean(plane: np.ndarray) -> float:
    """``ImageStatistics.getStatistics(ip, MEAN, null).mean`` (sequential double sum)."""
    a = np.asarray(plane)
    if a.dtype == np.float32:
        v = a.astype(np.float64).ravel()
        v = v[~np.isnan(v)]
        if v.size == 0:
            return float("nan")
        return float(np.cumsum(v)[-1] / v.size)
    return float(a.astype(np.float64).sum() / a.size)


def pad_to_power_of_2(plane: np.ndarray):
    """``FFT.pad``: (padded array (input dtype), padded?, original width, original height).
    Padding to the next power of 2 square, filled with the mean (int-truncated for 8/16-bit)."""
    h, w = plane.shape[:2]
    maxn = max(w, h)
    i = 2
    while i < maxn:
        i *= 2
    if i == maxn and w == h:
        return plane, False, w, h
    if i >= 65536:
        raise ValueError(f"Padded image is too large ({i}x{i})")
    mean = plane_mean(plane)
    if plane.dtype == np.float32:
        fill = F32(mean)
    else:
        top = 255 if plane.dtype == np.uint8 else 65535
        fill = min(max(int(java_int(mean)), 0), top)
    out = np.full((i, i), fill, dtype=plane.dtype)
    out[:h, :w] = plane
    return out, True, w, h


def forward_fft(plane: np.ndarray, bit_depth: int, lut=None) -> tuple[np.ndarray, FHTData]:
    """``FFT.newFHT`` + ``doForwardTransform``: (8-bit power spectrum, FHT data).

    ``plane`` is the (cropped) input: (H, W) gray or (H, W, 3) RGB."""
    rgb = None
    if bit_depth == 24:
        rgb = np.array(plane, copy=True)
        source = get_brightness(plane)
    else:
        source = plane
    padded, was_padded, ow, oh = pad_to_power_of_2(source)
    data = fht_transform(np.asarray(padded, F32))
    info = FHTData(data, ow if was_padded else 0, oh if was_padded else 0, bit_depth,
                   None if bit_depth == 24 else lut, rgb)
    ps = power_spectrum(data)
    info.power_spectrum_mean = float(ps.astype(np.float64).mean())
    return ps, info


def change_values_and_symmetrize(mask: np.ndarray, v1: int, v2: int) -> np.ndarray:
    """``FFT.changeValuesAndSymmetrize`` (square 8-bit)."""
    n = mask.shape[0]
    s = mask == v1
    part = np.zeros_like(s)
    part[1:, 0] = s[1:, 0][::-1]                 # left edge: (k,0) <-> (n-k,0)
    part[0, 1:] = s[0, 1:][::-1]                 # top edge: (0,c) <-> (0,n-c)
    part[1:, 1:] = s[1:, 1:][::-1, ::-1]         # interior: (r,c) <-> (n-r,n-c)
    return np.where(s | part, v1, v2).astype(np.uint8)


def smooth_periodic(mask: np.ndarray) -> np.ndarray:
    """``FFT.smooth``: 3x3 mean with periodic boundaries, (sum+4)/9."""
    a = mask.astype(np.int64)
    acc = np.zeros_like(a)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            acc += np.roll(np.roll(a, dy, axis=0), dx, axis=1)
    return ((acc + 4) // 9).astype(np.uint8)


def apply_spectrum_mask(fht: np.ndarray, mask_plane: np.ndarray, mask_bit_depth: int = 8,
                        power_spectrum_mean: float | None = None) -> tuple[np.ndarray, str]:
    """``FFT.doMasking``: returns (masked FHT, mode) with mode "pass", "filter" or "" (no
    pixel is 0 or 255, nothing done)."""
    n = fht.shape[0]
    mask = np.asarray(mask_plane)
    if mask.dtype != np.uint8:
        mask = float_to_byte_noscale(mask) if mask.dtype == np.float32 else \
            np.minimum(mask, 255).astype(np.uint8)
    if mask.shape != (n, n):
        return fht, ""
    has0, has255 = bool((mask == 0).any()), bool((mask == 255).any())
    if not has0 and not has255:
        return fht, ""
    pass_mode = has255
    if pass_mode:
        m = change_values_and_symmetrize(mask, 255, 0)
    else:
        m = change_values_and_symmetrize(mask, 0, 255)
    for _ in range(3):
        m = smooth_periodic(m)
    m = swap_quadrants(m)
    prod = (np.asarray(fht, F32) * m.astype(F32)).astype(F32)
    out = (prod.astype(np.float64) / 255.0).astype(F32)
    return out, ("pass" if pass_mode else "filter")


def inverse_fft(info: FHTData, mask_plane: np.ndarray | None = None, mask_bit_depth: int = 8,
                stack_size: int = 1, to_float: bool = False):
    """``FFT.doInverseTransform``: (pixels, bit depth). ``mask_plane`` is the edited power
    spectrum (black = filter, white = pass); 8/16-bit results are converted without scaling,
    RGB results get the original hue/saturation with the new brightness."""
    data = info.data.copy()
    if mask_plane is not None and stack_size <= 1:
        data, _ = apply_spectrum_mask(data, mask_plane, mask_bit_depth, info.power_spectrum_mean)
    data = fht_transform(data, inverse=True)
    if info.quadrant_swap_needed:
        data = swap_quadrants(data)
    if info.original_width > 0:
        data = data[:info.original_height, :info.original_width]
    data = np.ascontiguousarray(data)
    bit_depth = info.original_bit_depth or 32
    if to_float and bit_depth != 24:
        bit_depth = 32
    if bit_depth == 8:
        return float_to_byte_noscale(data), 8
    if bit_depth == 16:
        return float_to_short_noscale(data), 16
    if bit_depth == 24:
        if info.rgb is None:
            raise ValueError("Unable to set brightness")
        return set_brightness(info.rgb, data), 24
    return data.astype(F32), 32


def fht_inverse_of_fht_image(plane: np.ndarray) -> np.ndarray:
    """``FFT.doFHTInverseTransform`` (inverse of an "FHT of ..." window)."""
    a = swap_quadrants(np.asarray(plane, F32))
    return fht_transform(a, inverse=True)


def complex_inverse(re: np.ndarray, im: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``FFT.doComplexInverseTransform`` (Real/Imaginary stack, quadrant-swapped input)."""
    rein = swap_quadrants(np.asarray(re, F32))
    imin = swap_quadrants(np.asarray(im, F32))
    mre, mim = _mirror(rein), _mirror(imin)
    f1 = (((rein + mre) - (imin - mim)) * F32(0.5)).astype(F32)
    reout = fht_transform(f1, inverse=True)
    f2 = (((imin + mim) + (rein - mre)) * F32(0.5)).astype(F32)
    imout = fht_transform(f2, inverse=True)
    return reout, imout


# ------------------------------------------------------------------------------- FD Math
FD_OPERATIONS = ("Correlate", "Convolve", "Deconvolve")


def fd_math(a: np.ndarray | FHTData, b: np.ndarray | FHTData, operation: str = "Correlate",
            do_inverse: bool = True):
    """``FFTMath.doMath``: returns a float32 result (inverse, quadrant-swapped) when
    ``do_inverse``, else (power spectrum as float32, FHTData) for a frequency-domain window."""
    def to_fht(x):
        if isinstance(x, FHTData):
            return x.data.copy()
        return fht_transform(np.asarray(x, F32))
    h1 = to_fht(a)
    h2 = h1.copy() if b is a else to_fht(b)
    if h1.shape != h2.shape:
        raise ValueError("Images must be the same size")
    op = operation.lower()
    if op == "correlate":
        result = fht_multiply(h1, h2, conjugate=True)
    elif op == "convolve":
        result = fht_multiply(h1, h2, conjugate=False)
    elif op == "deconvolve":
        result = fht_divide(h1, h2)
    else:
        raise ValueError(f"unknown operation {operation!r}")
    if do_inverse:
        out = fht_transform(result, inverse=True)
        return swap_quadrants(out)
    ps = power_spectrum(result).astype(F32)
    return ps, FHTData(result, 0, 0, 0, None, None, quadrant_swap_needed=True)


# ------------------------------------------------------------------------------- Bandpass
def _insert(dst: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    """``ImageProcessor.insert`` (copyBits COPY with clipping)."""
    h, w = dst.shape[:2]
    sh, sw = src.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + sw, w), min(y + sh, h)
    if x1 <= x0 or y1 <= y0:
        return
    dst[y0:y1, x0:x1] = src[y0 - y:y1 - y, x0 - x:x1 - x]


def tile_mirror(plane: np.ndarray, width: int, height: int, x: int, y: int) -> np.ndarray:
    """``FFTFilter.tileMirror``: the image at (x, y) in a width x height canvas, mirrored
    around its edges."""
    if x < 0 or x > width - 1 or y < 0 or y > height - 1:
        raise ValueError("Image to be tiled is out of bounds.")
    out = np.zeros((height, width), dtype=plane.dtype)
    ip2 = np.array(plane, copy=True)
    h2, w2 = ip2.shape[:2]
    i1 = math.ceil(x / w2)
    i2 = math.ceil((width - x) / w2)
    j1 = math.ceil(y / h2)
    j2 = math.ceil((height - y) / h2)
    if i1 % 2 > 0.5:
        ip2 = ip2[:, ::-1]
    if j1 % 2 > 0.5:
        ip2 = ip2[::-1, :]
    for i in range(-i1, i2, 2):
        for j in range(-j1, j2, 2):
            _insert(out, ip2, x - i * w2, y - j * h2)
    ip2 = ip2[:, ::-1]
    for i in range(-i1 + 1, i2, 2):
        for j in range(-j1, j2, 2):
            _insert(out, ip2, x - i * w2, y - j * h2)
    ip2 = ip2[::-1, :]
    for i in range(-i1 + 1, i2, 2):
        for j in range(-j1 + 1, j2, 2):
            _insert(out, ip2, x - i * w2, y - j * h2)
    ip2 = ip2[:, ::-1]
    for i in range(-i1, i2, 2):
        for j in range(-j1 + 1, j2, 2):
            _insert(out, ip2, x - i * w2, y - j * h2)
    return out


STRIPES = ("None", "Horizontal", "Vertical")


def _fexp(arg) -> np.ndarray:
    return np.exp(np.asarray(arg, np.float64)).astype(F32)


def filter_large_small(fht: np.ndarray, filter_large: float, filter_small: float,
                       stripes: int, scale_stripes: float) -> tuple[np.ndarray, np.ndarray]:
    """``FFTFilter.filterLargeSmall``: returns (filtered FHT, filter) - both float32."""
    fht = np.array(fht, dtype=F32, copy=True)
    n = fht.shape[0]
    h = n // 2
    filt = np.ones((n, n), F32)
    one = F32(1)
    scale_large = filter_large * filter_large
    scale_small = filter_small * filter_small
    scale_stripes = scale_stripes * scale_stripes
    if h > 1:
        k = np.arange(1, h, dtype=np.int64)
        kk = -(k * k).astype(np.float64)
        f_large = _fexp(kk * scale_large)
        f_small = _fexp(kk * scale_small)
        rfl, rfs = f_large[:, None], f_small[:, None]
        cfl, cfs = f_large[None, :], f_small[None, :]
        factor = (((one - (rfl * cfl)) * rfs) * cfs).astype(F32)
        if stripes == 1:
            factor = (factor * (one - _fexp(kk * scale_stripes))[None, :]).astype(F32)
        elif stripes == 2:
            factor = (factor * (one - _fexp(kk * scale_stripes))[:, None]).astype(F32)
        rows, back = k, n - k
        for rr in (rows, back):
            for cc in (rows, back):
                ix = np.ix_(rr, cc)
                fht[ix] = (fht[ix] * factor).astype(F32)
                filt[ix] = (filt[ix] * factor).astype(F32)
    hh = -float(h * h)
    rfl = _fexp(hh * scale_large)
    rfs = _fexp(hh * scale_small)
    fst = _fexp(hh * scale_stripes)

    def mul(arrs, idx, f):
        for a in arrs:
            a[idx] = F32(a[idx] * F32(f))

    both = (fht, filt)
    mul(both, (0, h), (one - rfl) * rfs)
    mul(both, (h, 0), (one - rfl) * rfs)
    mul(both, (h, h), ((one - (rfl * rfl)) * rfs) * rfs)
    if stripes == 1:
        mul(both, (0, h), one - fst)
        fht[h, 0] = 0
        filt[h, 0] = 0
        mul(both, (h, h), one - fst)
    elif stripes == 2:
        fht[0, h] = 0
        filt[0, h] = 0
        mul(both, (h, 0), one - fst)
        mul(both, (h, h), one - fst)
    if h > 1:
        k = np.arange(1, h, dtype=np.int64)
        kk = -(k * k).astype(np.float64)
        cfl = _fexp(kk * scale_large)
        cfs = _fexp(kk * scale_small)
        # row 0 and row n/2 (rowFactLarge/Small belong to n/2)
        if stripes == 0:
            f0 = ((one - cfl) * cfs).astype(F32)
            fm = (((one - (cfl * rfl)) * cfs) * rfs).astype(F32)
        elif stripes == 1:
            fs = _fexp(kk * scale_stripes)
            f0 = (((one - cfl) * cfs) * (one - fs)).astype(F32)
            fm = ((((one - (cfl * rfl)) * cfs) * rfs) * (one - fs)).astype(F32)
        else:
            fs = _fexp(hh * scale_stripes)
            f0 = np.zeros_like(cfl)
            fm = ((((one - (cfl * rfl)) * cfs) * rfs) * (one - fs)).astype(F32)
        for a in both:
            for cc in (k, n - k):
                if stripes == 2:
                    a[0, cc] = 0
                else:
                    a[0, cc] = (a[0, cc] * f0).astype(F32)
                a[h, cc] = (a[h, cc] * fm).astype(F32)
        # column 0 and column n/2 (colFactLarge/Small belong to n/2)
        rfl_j, rfs_j = cfl, cfs
        cflh, cfsh = rfl, rfs
        if stripes == 0:
            g0 = ((one - rfl_j) * rfs_j).astype(F32)
            gm = (((one - (rfl_j * cflh)) * rfs_j) * cfsh).astype(F32)
        elif stripes == 1:
            fs = _fexp(hh * scale_stripes)
            g0 = None
            gm = ((((one - (rfl_j * cflh)) * rfs_j) * cfsh) * (one - fs)).astype(F32)
        else:
            fs = _fexp(kk * scale_stripes)
            g0 = (((one - rfl_j) * rfs_j) * (one - fs)).astype(F32)
            gm = ((((one - (rfl_j * cflh)) * rfs_j) * cfsh) * (one - fs)).astype(F32)
        for a in both:
            for rr in (k, n - k):
                if stripes == 1:
                    a[rr, 0] = 0
                else:
                    a[rr, 0] = (a[rr, 0] * g0).astype(F32)
                a[rr, h] = (a[rr, h] * gm).astype(F32)
    return fht, filt


def bandpass_filter(plane: np.ndarray, bit_depth: int, filter_large: float = 40.0,
                    filter_small: float = 3.0, suppress: str = "None", tolerance: float = 5.0,
                    autoscale: bool = True, saturate: bool = True,
                    rect: tuple[int, int, int, int] | None = None, want_filter: bool = False):
    """``FFTFilter.filter`` on one plane. Returns (new plane, filter image or None).

    ``rect`` = (x, y, w, h) of the selection (default: whole plane); pixels outside it are
    kept. 8/16-bit results are converted back with (autoscale) or without scaling as ImageJ
    does; RGB filters the HSB brightness."""
    plane = np.asarray(plane)
    H, W = plane.shape[:2]
    rx, ry, rw, rh = rect if rect is not None else (0, 0, W, H)
    src = get_brightness(plane) if bit_depth == 24 else plane
    src = src[ry:ry + rh, rx:rx + rw]
    maxn = max(rw, rh)
    sharpness = (100.0 - tolerance) / 100.0
    i = 2
    while i < 1.5 * maxn:
        i *= 2
    f_large = 2.0 * filter_large / float(i)
    f_small = 2.0 * filter_small / float(i)
    fx = int(math.floor((i - rw) / 2.0 + 0.5))
    fy = int(math.floor((i - rh) / 2.0 + 0.5))
    tiled = tile_mirror(src, i, i, fx, fy)
    fht = fht_transform(np.asarray(tiled, F32))
    stripes = STRIPES.index(suppress) if suppress in STRIPES else \
        [s.lower() for s in STRIPES].index(str(suppress).lower())
    fht, filt = filter_large_small(fht, f_large, f_small, stripes, sharpness)
    out = fht_transform(fht, inverse=True)
    ip2 = np.ascontiguousarray(out[fy:fy + rh, fx:fx + rw])
    # the crop inherits the FHT's min/max, which were found on the tiled input (FHT ctor)
    vmin, vmax = float_min_max(np.asarray(tiled, F32))
    if autoscale:
        hist, hmin, bs, n = float_histogram(ip2)
        r = stretch_min_max(hist, hmin, bs, n, 1.0 if saturate else 0.0)
        if r is not None and not (r[0] == 0.0 and r[1] == 0.0):
            vmin, vmax = float(F32(r[0])), float(F32(r[1]))
    result = np.array(plane, copy=True)
    if bit_depth == 8:
        conv = float_to_byte_scaled(ip2, vmin, vmax) if autoscale else float_to_byte_noscale(ip2)
        result[ry:ry + rh, rx:rx + rw] = conv
    elif bit_depth == 16:
        conv = float_to_short_scaled(ip2, vmin, vmax) if autoscale else \
            float_to_short_noscale(ip2)
        result[ry:ry + rh, rx:rx + rw] = conv
    elif bit_depth == 24:
        bright = get_brightness(plane)
        bright[ry:ry + rh, rx:rx + rw] = ip2
        sub = set_brightness(plane, bright)
        result[ry:ry + rh, rx:rx + rw] = sub[ry:ry + rh, rx:rx + rw]
    else:
        result[ry:ry + rh, rx:rx + rw] = ip2
    filter_image = swap_quadrants(filt) if want_filter else None
    return result, filter_image


# ------------------------------------------------------------------------------- custom filter
def resize_nearest(a: np.ndarray, dst_w: int, dst_h: int) -> np.ndarray:
    """``ImageProcessor.resize`` with interpolation NONE (the default)."""
    h, w = a.shape[:2]
    if (w, h) == (dst_w, dst_h):
        return a.copy()
    xs = (np.arange(dst_w) - dst_w / 2.0) / (dst_w / w) + w / 2.0
    ys = (np.arange(dst_h) - dst_h / 2.0) / (dst_h / h) + h / 2.0
    xi = np.clip(java_int(xs), 0, w - 1)
    yi = np.clip(java_int(ys), 0, h - 1)
    return a[np.ix_(yi, xi)]


def filter_to_8bit_or_float(filter_plane: np.ndarray, bit_depth: int,
                            display_range: tuple[float, float] | None = None) -> np.ndarray:
    """``filter.convertToByte(true)`` for non-32-bit filters (32-bit kept)."""
    a = np.asarray(filter_plane)
    if bit_depth == 32:
        return a.astype(F32)
    if bit_depth == 8:
        return a.astype(np.uint8)
    if bit_depth == 24:
        w = 1.0 / 3.0
        v = a[..., 0] * w + a[..., 1] * w + a[..., 2] * w + 0.5
        return (java_int(v) & 0xFF).astype(np.uint8)
    lo, hi = display_range if display_range is not None else (float(a.min()), float(a.max()))
    lo, hi = int(lo), int(hi)
    scale = 256.0 / (hi - lo + 1)
    v = np.maximum(a.astype(np.int64) - lo, 0)
    v = java_int(v * scale + 0.5)
    return np.minimum(v, 255).astype(np.uint8)


def custom_filter(plane: np.ndarray, bit_depth: int, filter_img: np.ndarray) -> np.ndarray:
    """``FFTCustomFilter`` on one plane (filter = 8-bit 0..255 or 32-bit factors in the
    power-spectrum layout, resized to the padded size with nearest neighbour)."""
    plane = np.asarray(plane)
    H, W = plane.shape[:2]
    maxn = max(W, H)
    size = 2
    while size < 1.5 * maxn:
        size *= 2
    rx = int(math.floor((size - W) / 2.0 + 0.5))
    ry = int(math.floor((size - H) / 2.0 + 0.5))
    src = get_brightness(plane) if bit_depth == 24 else plane
    fht = fht_transform(np.asarray(tile_mirror(src, size, size, rx, ry), F32))
    filt = resize_nearest(np.asarray(filter_img), size, size)
    filt = swap_quadrants(filt)
    if filt.dtype == np.float32:
        fht = (fht * filt).astype(F32)
    else:
        fht = (fht.astype(np.float64) * (filt.astype(np.float64) / 255.0)).astype(F32)
    out = fht_transform(fht, inverse=True)
    ip2 = np.ascontiguousarray(out[ry:ry + H, rx:rx + W])
    vmin, vmax = float_min_max(out)      # crop() inherits the padded FHT's min/max
    if bit_depth == 8:
        return float_to_byte_scaled(ip2, vmin, vmax)
    if bit_depth == 16:
        return float_to_short_scaled(ip2, vmin, vmax)
    if bit_depth == 24:
        return set_brightness(plane, ip2)
    return ip2.astype(F32)

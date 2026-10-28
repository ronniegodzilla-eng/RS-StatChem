"""Image > Type conversions with ImageJ's scaling and rounding rules.

Ported from ImageJ 1.54 ij/process/{TypeConverter,ImageConverter,StackConverter,MedianCut,
ColorSpaceConverter}.java, ij/plugin/Converter.java and the HSB code of java.awt.Color as used by
ij/process/ColorProcessor.java (ImageJ is public domain, NIH; W. Rasband, M. Schmid et al.).

Rules (all verified against ImageJ 1.54p goldens, ``tests/data_imagej/golden/image_types``):

* 16 -> 8 bit with "Scale when converting": ``(int)((v-min)*256/(max-min+1)+0.5)`` clipped, with
  the **display range** (integer min/max); without scaling values are clipped at 255.
* 32 -> 8 bit: ``(int)((v-min)*255/(max-min)+0.5)`` in double (FloatProcessor.create8BitImage);
  without scaling ``(int)(v+0.5f)`` clipped 0..255.
* 32 -> 16 bit: ``(short)((v-min)*65535/(max-min)+0.5)``; a scaled stack gets display 0-65535.
* RGB -> 8/16/32 bit: ``r*wr + g*wg + b*wb`` (+0.5 truncated for 8 bit); weights 1/3 each, or
  0.299/0.587/0.114 with "Weighted RGB conversions".
* Gray -> RGB: the image as displayed (display range and LUT), 16/32-bit first scaled to 8 bit.
* 8-bit Color: Heckbert median cut as modified by M. Schmid (2023) - :class:`MedianCut`.
* HSB: ``Color.RGBtoHSB`` in float32, bytes are ``(int)(h*255.0)``; Lab: sRGB/D65 as
  ColorSpaceConverter.
"""
from __future__ import annotations

import numpy as np

from rsstatchem.core.imaging.image import (COLOR_256, COLOR_RGB, GRAY8, GRAY16, GRAY32, ImageDoc)
from rsstatchem.core.imaging.lut import LUT

UNWEIGHTED = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
WEIGHTED = (0.299, 0.587, 0.114)
TYPE_LABELS = ("8-bit", "16-bit", "32-bit", "8-bit Color", "RGB Color", "RGB Stack",
               "HSB Stack", "HSB (32-bit)", "Lab Stack")


class ConversionError(ValueError):
    """Unsupported conversion (ImageJ shows the "Supported Conversions" list)."""


SUPPORTED_TEXT = ("Supported Conversions:\n \n8-bit -> 16-bit*\n8-bit -> 32-bit*\n"
                  "8-bit -> RGB Color*\n16-bit -> 8-bit*\n16-bit -> 32-bit*\n"
                  "16-bit -> RGB Color*\n32-bit -> 8-bit*\n32-bit -> 16-bit\n"
                  "32-bit -> RGB Color*\n8-bit Color -> 8-bit (grayscale)*\n"
                  "8-bit Color -> RGB Color\nRGB -> 8-bit (grayscale)*\nRGB -> 8-bit Color*\n"
                  "RGB -> RGB Stack*\nRGB -> HSB Stack*\nRGB -> Lab Stack\n"
                  "RGB Stack -> RGB Color\nHSB Stack -> RGB Color\n \n* works with stacks\n")


def java_int(a) -> np.ndarray:
    """Java ``(int)`` of doubles: truncation toward zero, NaN -> 0, saturating."""
    a = np.asarray(a, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        out = np.trunc(np.nan_to_num(a, nan=0.0, posinf=2147483647.0, neginf=-2147483648.0))
    return np.clip(out, -2147483648.0, 2147483647.0).astype(np.int64)


def java_round(a) -> np.ndarray:
    """Java ``Math.round`` (floor(x + 0.5))."""
    return np.floor(np.asarray(a, dtype=np.float64) + 0.5).astype(np.int64)


# ---------------------------------------------------------------------------------- display
def byte_display_table(lut: LUT, vmin: float, vmax: float) -> np.ndarray:
    """(256, 3) colours of 8-bit raw values for the display range (ByteProcessor.setMinAndMax)."""
    base = lut.as_array()
    if vmin == 0 and vmax == 255:
        return base
    imin = int(java_round(vmin))
    i = np.arange(256)
    with np.errstate(divide="ignore", invalid="ignore"):
        index = java_int(256.0 * (i - imin) / (vmax - vmin))
    index = np.clip(index, 0, 255)
    index = np.where(i < vmin, 0, np.where(i > vmax, 255, index))
    return base[index]


def short_to_byte(plane: np.ndarray, vmin: float, vmax: float, scale: bool = True) -> np.ndarray:
    """TypeConverter.convertShortToByte."""
    p = plane.astype(np.int64)
    if not scale:
        return np.minimum(p, 255).astype(np.uint8)
    lo, hi = int(vmin), int(vmax)
    s = 256.0 / (hi - lo + 1)
    v = np.maximum(p - lo, 0)
    return np.minimum(java_int(v * s + 0.5), 255).astype(np.uint8)


def float_to_byte(plane: np.ndarray, vmin: float, vmax: float, scale: bool = True) -> np.ndarray:
    """TypeConverter.convertFloatToByte (create8BitImage, or setPixels(0, fp) unscaled)."""
    f = plane.astype(np.float32)
    if not scale:
        v = f + np.float32(0.5)
        v = np.where(v < 0, np.float32(0), np.where(v > 255, np.float32(255), v))
        return java_int(v).astype(np.uint8)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        s = 255.0 / (vmax - vmin) if vmax != vmin else np.inf
        v = f.astype(np.float64) - vmin
        v = np.where(v < 0.0, 0.0, v)
        iv = java_int(v * s + 0.5)
    return np.minimum(iv, 255).astype(np.uint8)


def float_to_short(plane: np.ndarray, vmin: float, vmax: float, scale: bool = True
                   ) -> np.ndarray:
    """TypeConverter.convertFloatToShort."""
    f = plane.astype(np.float32).astype(np.float64)
    s = 1.0 if vmax - vmin == 0.0 else 65535.0 / (vmax - vmin)
    with np.errstate(invalid="ignore"):
        v = (f - vmin) * s if scale else f
        v = np.where(v < 0.0, 0.0, np.where(v > 65535.0, 65535.0, v))
    return java_int(v + 0.5).astype(np.uint16)


def rgb_weights(options=None) -> tuple[float, float, float]:
    if options is not None and getattr(options, "weighted_rgb", False):
        return WEIGHTED
    return UNWEIGHTED


def rgb_gray_double(rgb: np.ndarray, weights=UNWEIGHTED) -> np.ndarray:
    r, g, b = (rgb[..., k].astype(np.float64) for k in range(3))
    return r * weights[0] + g * weights[1] + b * weights[2]


def rgb_to_byte(rgb: np.ndarray, weights=UNWEIGHTED) -> np.ndarray:
    """TypeConverter.convertRGBToByte: ``(byte)(r*rw + g*gw + b*bw + 0.5)``."""
    return (java_int(rgb_gray_double(rgb, weights) + 0.5) & 0xFF).astype(np.uint8)


def rgb_to_float(rgb: np.ndarray, weights=UNWEIGHTED) -> np.ndarray:
    return rgb_gray_double(rgb, weights).astype(np.float32)


def gray_to_rgb(plane: np.ndarray, image_type: int, lut: LUT, vmin: float, vmax: float,
                scale: bool = True) -> np.ndarray:
    """TypeConverter.convertToRGB of one plane: the plane as displayed."""
    if image_type in (GRAY8, COLOR_256):
        return byte_display_table(lut, vmin, vmax)[plane]
    if image_type == GRAY16:
        idx = short_to_byte(plane, vmin, vmax, scale)
    else:
        idx = float_to_byte(plane, vmin, vmax, scale)
    return lut.as_array()[idx]


# ---------------------------------------------------------------------------------- HSB / Lab
def rgb_to_hsb_float(rgb: np.ndarray):
    """java.awt.Color.RGBtoHSB (float32) for an (..., 3) uint8 array -> (h, s, b) float32."""
    f32 = np.float32
    r = rgb[..., 0].astype(np.int64)
    g = rgb[..., 1].astype(np.int64)
    b = rgb[..., 2].astype(np.int64)
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    bright = cmax.astype(f32) / f32(255.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sat = np.where(cmax != 0, (cmax - cmin).astype(f32) / cmax.astype(f32), f32(0))
        span = (cmax - cmin).astype(f32)
        redc = (cmax - r).astype(f32) / span
        greenc = (cmax - g).astype(f32) / span
        bluec = (cmax - b).astype(f32) / span
        hue = np.where(r == cmax, bluec - greenc,
                       np.where(g == cmax, (f32(2.0) + redc) - bluec,
                                (f32(4.0) + greenc) - redc)).astype(f32)
        hue = hue / f32(6.0)
        hue = np.where(hue < 0, hue + f32(1.0), hue)
    hue = np.where(sat == 0, f32(0), hue).astype(f32)
    return hue, sat.astype(f32), bright.astype(f32)


def rgb_to_hsb_bytes(rgb: np.ndarray) -> np.ndarray:
    """ColorProcessor.getHSB(byte[]): (3, ...) uint8 with ``(int)(v*255.0)``."""
    return np.stack([(java_int(c.astype(np.float64) * 255.0) & 0xFF).astype(np.uint8)
                     for c in rgb_to_hsb_float(rgb)])


def hsb_to_rgb(h, s, b) -> np.ndarray:
    """Color.HSBtoRGB for float arrays -> (..., 3) uint8."""
    from rsstatchem.core.imaging.lut_tables import java_hsb_to_rgb
    r, g, bb = java_hsb_to_rgb(h, s, b)
    return np.stack([r, g, bb], axis=-1).astype(np.uint8)


def hsb_bytes_to_rgb(h: np.ndarray, s: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ColorProcessor.setHSB(byte[]): ``(float)((v&0xff)/255.0)`` per component."""
    def f(c):
        return (c.astype(np.float64) / 255.0).astype(np.float32)
    return hsb_to_rgb(f(h), f(s), f(b))


_M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
_MI = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415],
                [0.0557, -0.2040, 1.0570]])
_D65 = (95.0429, 100.0, 108.8900)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """ColorSpaceConverter.RGBtoLAB -> (3, ...) float32 (L*, a*, b*)."""
    c = rgb.astype(np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, np.power((c + 0.055) / 1.055, 2.4)) * 100.0
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    xyz = [(r * _M[i, 0]) + (g * _M[i, 1]) + (b * _M[i, 2]) for i in range(3)]
    t = [xyz[i] / _D65[i] for i in range(3)]
    t = [np.where(v > 0.008856, np.power(v, 1.0 / 3.0), (7.787 * v) + (16.0 / 116.0)) for v in t]
    L = (116.0 * t[1]) - 16.0
    a = 500.0 * (t[0] - t[1])
    bb = 200.0 * (t[1] - t[2])
    return np.stack([L, a, bb]).astype(np.float32)


def lab_to_rgb(L: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ColorSpaceConverter.LABtoRGB + ColorProcessor.putPixel(int[]) -> (..., 3) uint8."""
    L, a, b = (np.asarray(v, dtype=np.float32).astype(np.float64) for v in (L, a, b))
    y = (L + 16.0) / 116.0
    x = (a / 500.0) + y
    z = y - (b / 200.0)

    def f(v):
        v3 = np.power(v, 3.0)
        return np.where(v3 > 0.008856, v3, (v - (16.0 / 116.0)) / 7.787)

    X, Y, Z = f(x) * _D65[0], f(y) * _D65[1], f(z) * _D65[2]
    x, y, z = X / 100.0, Y / 100.0, Z / 100.0
    out = []
    for i in range(3):
        c = (x * _MI[i, 0]) + (y * _MI[i, 1]) + (z * _MI[i, 2])
        with np.errstate(invalid="ignore"):
            c = np.where(c > 0.0031308, (1.055 * np.power(np.maximum(c, 0), 1.0 / 2.4)) - 0.055,
                         c * 12.92)
        c = np.where(c < 0, 0.0, c)
        out.append(java_round(c * 255))
    packed = (out[0] << 16) + (out[1] << 8) + out[2]     # putPixel(x, y, int[]) does not clip
    return np.stack([(packed >> 16) & 0xFF, (packed >> 8) & 0xFF, packed & 0xFF],
                    axis=-1).astype(np.uint8)


# ---------------------------------------------------------------------------------- MedianCut
class MedianCut:
    """Heckbert median cut on a 32x32x32 histogram with M. Schmid's centroid refinement
    (port of ij/process/MedianCut.java)."""

    HSIZE = 32768

    def __init__(self, rgb: np.ndarray):
        rgb = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3).astype(np.int64)
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        self.bgr15 = ((b >> 3) << 10) | ((g >> 3) << 5) | (r >> 3)
        H = self.HSIZE
        self.hist = np.bincount(self.bgr15, minlength=H).astype(np.int64)
        self.sum_r = np.bincount(self.bgr15, weights=r - ((r >> 3) << 3), minlength=H
                                 ).astype(np.int64)
        self.sum_g = np.bincount(self.bgr15, weights=g - ((g >> 3) << 3), minlength=H
                                 ).astype(np.int64)
        self.sum_b = np.bincount(self.bgr15, weights=b - ((b >> 3) << 3), minlength=H
                                 ).astype(np.int64)

    @staticmethod
    def _red(x):
        return (x & 31) << 3

    @staticmethod
    def _green(x):
        return (x >> 2) & 0xF8

    @staticmethod
    def _blue(x):
        return (x >> 7) & 0xF8

    def _shrink(self, cube, ptr):
        cols = np.asarray(ptr[cube["lower"]:cube["upper"] + 1], dtype=np.int64)
        r, g, b = self._red(cols), self._green(cols), self._blue(cols)
        cube.update(rmin=int(r.min()), rmax=int(r.max()), gmin=int(g.min()), gmax=int(g.max()),
                    bmin=int(b.min()), bmax=int(b.max()))

    def convert(self, maxcubes: int = 256) -> tuple[np.ndarray, LUT]:
        """Indices (flat, in input order) and the colour table (``maxcubes`` entries used)."""
        hist = self.hist
        ptr = [int(i) for i in np.nonzero(hist)[0]]
        cubes = [{"lower": 0, "upper": len(ptr) - 1, "count": int(hist.sum()), "level": 0}]
        self._shrink(cubes[0], ptr)
        while len(cubes) < maxcubes:
            level, split = 255, -1
            for k, c in enumerate(cubes):
                if c["lower"] != c["upper"] and c["level"] < level:
                    level, split = c["level"], k
            if split == -1:
                break
            cube = cubes[split]
            lr, lg, lb = (cube["rmax"] - cube["rmin"], cube["gmax"] - cube["gmin"],
                          cube["bmax"] - cube["bmin"])
            longdim = 0
            if lr >= lg and lr >= lb:
                longdim = 0
            if lg >= lr and lg >= lb:
                longdim = 1
            if lb >= lr and lb >= lg:
                longdim = 2
            lo, hi = cube["lower"], cube["upper"]
            seg = ptr[lo:hi + 1]
            if longdim == 0:
                key = lambda c: ((c & 31) << 10) | (c >> 5)                       # noqa: E731
            elif longdim == 1:
                key = lambda c: (((c >> 5) & 31) << 10) | ((c >> 10) << 5) | (c & 31)  # noqa
            else:
                key = lambda c: c                                                  # noqa: E731
            ptr[lo:hi + 1] = sorted(seg, key=key)
            count = 0
            i = lo
            while i <= hi - 1:
                if count >= cube["count"] // 2:
                    break
                count += int(hist[ptr[i]])
                i += 1
            median = i
            a = {"lower": lo, "upper": median - 1, "count": count, "level": cube["level"] + 1}
            self._shrink(a, ptr)
            cubes[split] = a
            b = {"lower": median, "upper": hi, "count": cube["count"] - count,
                 "level": cube["level"] + 1}
            self._shrink(b, ptr)
            cubes.append(b)
        ncubes = len(cubes)
        inverse = np.zeros(self.HSIZE, dtype=np.int64)
        rl, gl, bl = (np.zeros(256, np.int64) for _ in range(3))
        for k, cube in enumerate(cubes):
            cols = np.asarray(ptr[cube["lower"]:cube["upper"] + 1], dtype=np.int64)
            h = hist[cols]
            rs = int((self._red(cols) * h + self.sum_r[cols]).sum())
            gs = int((self._green(cols) * h + self.sum_g[cols]).sum())
            bs = int((self._blue(cols) * h + self.sum_b[cols]).sum())
            rl[k] = java_round(rs / float(cube["count"])) & 0xFF
            gl[k] = java_round(gs / float(cube["count"])) & 0xFF
            bl[k] = java_round(bs / float(cube["count"])) & 0xFF
            inverse[cols] = k
        for _ in range(5):
            if not self._improve(ncubes, inverse, rl, gl, bl):
                break
        idx = inverse[self.bgr15]
        lut = LUT(rl, gl, bl, "")
        return idx.astype(np.uint8), lut

    def _improve(self, ncubes, inverse, rl, gl, bl) -> bool:
        hist = self.hist
        bins = np.nonzero(hist)[0]
        h = hist[bins]
        r_bin = self._red(bins) + self.sum_r[bins] // h      # long / int: integer division
        g_bin = self._green(bins) + self.sum_g[bins] // h
        b_bin = self._blue(bins) + self.sum_b[bins] // h
        R, G, B = rl[:ncubes], gl[:ncubes], bl[:ncubes]
        lut15 = ((B >> 3) << 10) | ((G >> 3) << 5) | (R >> 3)
        d = ((r_bin[:, None] - R[None, :]).astype(np.float64) ** 2 +
             (g_bin[:, None] - G[None, :]).astype(np.float64) ** 2 +
             (b_bin[:, None] - B[None, :]).astype(np.float64) ** 2)
        match = bins[:, None] == lut15[None, :]
        has = match.any(axis=1)
        nearest = np.where(has, np.argmax(match, axis=1), np.argmin(d, axis=1))
        dist = np.where(has, 0.0, d[np.arange(len(bins)), nearest])
        distance = np.zeros(self.HSIZE)
        distance[bins] = dist
        # the exact-match case also fixes inverse (no change flag)
        inverse[bins[has]] = nearest[has]
        diff = nearest != inverse[bins]
        if not diff.any():
            return False
        changed = np.zeros(ncubes, bool)
        changed[nearest[diff]] = True
        changed[inverse[bins[diff]]] = True
        inverse[bins[diff]] = nearest[diff]
        allbins = np.arange(self.HSIZE)
        k = inverse[allbins]
        sel = changed[k] if ncubes else np.zeros(self.HSIZE, bool)
        rs = np.bincount(k[sel], weights=(self._red(allbins) * hist + self.sum_r)[sel],
                         minlength=ncubes)
        gs = np.bincount(k[sel], weights=(self._green(allbins) * hist + self.sum_g)[sel],
                         minlength=ncubes)
        bs = np.bincount(k[sel], weights=(self._blue(allbins) * hist + self.sum_b)[sel],
                         minlength=ncubes)
        npx = np.bincount(k[sel], weights=hist[sel], minlength=ncubes)
        unused = []
        for j in range(ncubes):
            if changed[j]:
                if npx[j] > 0:
                    rl[j] = java_round(rs[j] / npx[j]) & 0xFF
                    gl[j] = java_round(gs[j] / npx[j]) & 0xFF
                    bl[j] = java_round(bs[j] / npx[j]) & 0xFF
                else:
                    unused.append(j)
        for j in unused:
            badness = distance * hist
            worst = int(np.argmax(badness)) if badness.max() > 0 else -1
            if worst < 0:
                continue
            hw = hist[worst]
            rl[j] = java_round(self._red(worst) + self.sum_r[worst] * (1.0 / hw)) & 0xFF
            gl[j] = java_round(self._green(worst) + self.sum_g[worst] * (1.0 / hw)) & 0xFF
            bl[j] = java_round(self._blue(worst) + self.sum_b[worst] * (1.0 / hw)) & 0xFF
            inverse[worst] = j
        return True


# ---------------------------------------------------------------------------------- helpers
def is_pseudo_color_lut(lut: LUT) -> bool:
    """ImageProcessor.isPseudoColorLut (smooth LUTs, plus the 3-3-2 RGB special case)."""
    arr = lut.as_array().astype(np.int64)
    prev = np.vstack([arr[:1], arr[:-1]])
    d = (arr - prev).ravel()
    s, s2 = float(d.sum()), float((d * d).sum())
    sd = (768 * s2 - s * s) / 768.0
    sd = np.sqrt(sd / 767.0) if sd > 0 else 0.0
    return bool(sd < 20.0 or int(sd) == 67)


def is_color_lut(lut: LUT) -> bool:
    return not (np.array_equal(lut.red, lut.green) and np.array_equal(lut.green, lut.blue))


def _is_inverted_ij(lut: LUT) -> bool:
    """ImageProcessor.isInvertedLut: no ascending step in r+g+b over entries 0..254."""
    v = lut.as_array().astype(np.int64).sum(axis=1)
    return not bool(np.any(v[:254] < v[1:255]))


def stack_kind(doc: ImageDoc) -> str | None:
    """ImageStack.isRGB/isHSB/isHSB32/isLab from the first slice label."""
    if doc.stack_size != 3:
        return None
    label = doc.slice_labels[0]
    if label == "Red" and doc.type in (GRAY8, COLOR_256):
        return "rgb"
    if label == "Hue":
        return "hsb" if doc.type == GRAY8 else ("hsb32" if doc.type == GRAY32 else None)
    if label == "L*":
        return "lab"
    return None


def _planes(doc):
    return [doc.plane_at(i) for i in range(doc.stack_size)]


def _restack(doc: ImageDoc, planes: list, shape_czt=None, rgb=False) -> np.ndarray:
    c, z, t = shape_czt or doc.dims
    h, w = planes[0].shape[:2]
    arr = np.zeros((t, z, c, h, w) + ((3,) if rgb else ()), dtype=planes[0].dtype)
    for i, p in enumerate(planes):
        cc, zz, tt = i % c, (i // c) % z, i // (c * z)
        arr[tt, zz, cc] = p
    return arr


def _finish(doc: ImageDoc, pixels: np.ndarray, new_type: int, keep_luts: bool = True) -> None:
    luts = [lut.copy() for lut in doc.luts]
    labels = list(doc.slice_labels)
    mode = doc.composite_mode
    doc.replace_pixels(pixels, new_type)
    if new_type == COLOR_RGB:
        doc.luts = [LUT.grays() for _ in range(doc.n_channels)]
    elif keep_luts and len(luts) == doc.n_channels:
        doc.luts = luts
        doc.composite_mode = mode
    if len(labels) == doc.stack_size:
        doc.slice_labels = labels
    doc.threshold = None


# ---------------------------------------------------------------------------------- converter
def convert(doc: ImageDoc, item: str, scale: bool = True, weights=UNWEIGHTED,
            n_colors: int = 256) -> ImageDoc | None:
    """Image > Type > ``item`` (ImageJ Converter.convert). Converts ``doc`` in place, except
    Lab Stack which returns a new image (ImageJ closes the source). Raises ConversionError."""
    if item not in TYPE_LABELS:
        raise ConversionError(f"Unknown conversion {item!r}")
    t = doc.type
    stack = doc.stack_size > 1
    kind = stack_kind(doc) if stack else None
    if item == "RGB Color":
        if kind is not None:
            _stack_to_rgb(doc, kind)
            return None
        if t == COLOR_RGB:
            return None
        to_rgb(doc, scale)
        return None
    if item == "8-bit":
        to_gray8(doc, scale, weights)
    elif item == "16-bit":
        if t == GRAY16:
            return None
        if t not in (GRAY8, GRAY32, COLOR_RGB) or (stack and t == COLOR_RGB):
            raise ConversionError(SUPPORTED_TEXT)
        to_gray16(doc, scale, weights)
    elif item == "32-bit":
        if t == GRAY32:
            return None
        if t not in (GRAY8, GRAY16, COLOR_RGB):
            raise ConversionError(SUPPORTED_TEXT)
        to_gray32(doc, weights)
    elif item == "8-bit Color":
        if t != COLOR_RGB:
            if t in (GRAY8, COLOR_256) and not stack:
                return None
            raise ConversionError(SUPPORTED_TEXT)
        to_indexed_color(doc, n_colors)
    elif item in ("RGB Stack", "HSB Stack", "HSB (32-bit)"):
        if t != COLOR_RGB:
            if item == "RGB Stack" and doc.is_composite and doc.n_channels == doc.stack_size:
                _composite_to_rgb_stack(doc)
                return None
            raise ConversionError(SUPPORTED_TEXT)
        {"RGB Stack": to_rgb_stack, "HSB Stack": to_hsb_stack,
         "HSB (32-bit)": to_hsb32_stack}[item](doc)
    elif item == "Lab Stack":
        if t != COLOR_RGB or stack:
            raise ConversionError(SUPPORTED_TEXT)
        return to_lab_stack(doc)
    return None


def to_gray8(doc: ImageDoc, scale: bool = True, weights=UNWEIGHTED) -> None:
    t = doc.type
    planes = _planes(doc)
    composite = doc.is_composite
    if t == GRAY8:
        lut = doc.lut
        if is_pseudo_color_lut(lut) or not is_color_lut(lut) or composite:
            if not composite:
                gray = LUT.grays()
                doc.luts = [gray.inverted() if _is_inverted_ij(lut) else gray
                            for _ in doc.luts]
            return
    if t in (COLOR_256, GRAY8) or t == COLOR_RGB:
        out = []
        for i, p in enumerate(planes):
            c = doc.position_of(i)[0]
            if t != COLOR_RGB:
                lo, hi = doc.display_ranges[c]
                p = byte_display_table(doc.luts[c], lo, hi)[p]
            out.append(rgb_to_byte(p, weights))
        pix = _restack(doc, out)
        if t == COLOR_RGB:
            doc.replace_pixels(pix, GRAY8)
        else:
            _finish(doc, pix, GRAY8, keep_luts=False)
            doc.luts = [LUT.grays() for _ in range(doc.n_channels)]
        return
    out = []
    cur = doc.display_range
    for i, p in enumerate(planes):
        c = doc.position_of(i)[0]
        lo, hi = doc.display_ranges[c] if composite else cur
        out.append(short_to_byte(p, lo, hi, scale) if t == GRAY16 else
                   float_to_byte(p, lo, hi, scale))
    cal = doc.calibration
    _finish(doc, _restack(doc, out), GRAY8)
    if cal.calibrated() and not cal.is_signed16():
        pass
    if cal.is_signed16() or t == GRAY32:
        cal.disable_density()
    doc.reset_display_range()


def to_gray16(doc: ImageDoc, scale: bool = True, weights=UNWEIGHTED) -> None:
    t = doc.type
    planes = _planes(doc)
    stack = doc.stack_size > 1
    if t == COLOR_RGB:
        out = [rgb_to_byte(p, weights).astype(np.uint16) for p in planes]
        doc.replace_pixels(_restack(doc, out), GRAY16)
        return
    if t == GRAY8:
        _finish(doc, _restack(doc, [p.astype(np.uint16) for p in planes]), GRAY16)
        return
    cur = doc.display_range
    out = []
    for i, p in enumerate(planes):
        c = doc.position_of(i)[0]
        lo, hi = doc.display_ranges[c] if doc.n_channels > 1 else cur
        out.append(float_to_short(p, lo, hi, scale))
    _finish(doc, _restack(doc, out), GRAY16)
    if stack and scale:
        doc.display_ranges = [(0.0, 65535.0)] * doc.n_channels


def to_gray32(doc: ImageDoc, weights=UNWEIGHTED) -> None:
    t = doc.type
    cal = doc.calibration
    ranges = list(doc.display_ranges)
    planes = _planes(doc)
    if t == COLOR_RGB:
        out = [rgb_to_float(p, weights) for p in planes]
        doc.replace_pixels(_restack(doc, out), GRAY32)
        doc.display_ranges = [(0.0, 255.0)] * doc.n_channels \
            if doc.stack_size == 1 else doc.display_ranges
        return
    table = cal.ctable(doc.bit_depth)
    out = [(table[p] if table is not None else p).astype(np.float32) for p in planes]
    _finish(doc, _restack(doc, out), GRAY32)
    if table is not None:
        ranges = [(float(cal.cvalue(lo)), float(cal.cvalue(hi))) for lo, hi in ranges]
        cal.disable_density()
    doc.display_ranges = ranges


def to_rgb(doc: ImageDoc, scale: bool = True) -> None:
    """Gray / 8-bit colour / composite -> RGB Color."""
    if doc.is_composite:
        from rsstatchem.core.imaging.lut import render
        c0, z0, t0 = doc.position
        arr = np.zeros((doc.n_frames, doc.n_slices, 1, doc.height, doc.width, 3), np.uint8)
        for t in range(doc.n_frames):
            for z in range(doc.n_slices):
                arr[t, z, 0] = render(doc, c0, z, t)
        labels = [doc.slice_labels[doc.stack_index(0, z, t)] for t in range(doc.n_frames)
                  for z in range(doc.n_slices)]
        doc.replace_pixels(arr, COLOR_RGB)
        doc.slice_labels = labels
        doc.luts = [LUT.grays()]
        doc.composite_mode = "color"
        doc.active_channels = [True]
        doc.set_position(0, z0, t0)
        return
    lo, hi = doc.display_range
    lut = doc.lut
    out = [gray_to_rgb(p, doc.type, lut, lo, hi, scale) for p in _planes(doc)]
    labels = list(doc.slice_labels)
    doc.replace_pixels(_restack(doc, out, rgb=True), COLOR_RGB)
    doc.slice_labels = labels
    doc.luts = [LUT.grays()]
    doc.threshold = None


def to_rgb_stack(doc: ImageDoc) -> None:
    rgb = doc.pixels[:, :, 0]                      # (T, Z, Y, X, 3)
    arr = np.moveaxis(rgb, -1, 2)                   # (T, Z, 3, Y, X)
    labels_z = list(doc.slice_labels)
    doc.replace_pixels(np.ascontiguousarray(arr), GRAY8)
    doc.luts = [LUT.grays() for _ in range(3)]
    doc.composite_mode = "grayscale"
    doc.active_channels = [True] * 3
    if doc.stack_size == 3:
        doc.slice_labels = ["Red", "Green", "Blue"]
    else:
        doc.slice_labels = [labels_z[i // 3] for i in range(doc.stack_size)]
    doc.hyperstack = doc.stack_size > 3


def _hsb(doc: ImageDoc, as_float: bool) -> None:
    rgb = doc.pixels[:, :, 0]
    if as_float:
        h, s, b = rgb_to_hsb_float(rgb)
        arr = np.stack([h, s, b], axis=2).astype(np.float32)
    else:
        arr = np.moveaxis(rgb_to_hsb_bytes(rgb), 0, 2)
    n = doc.stack_size
    doc.replace_pixels(np.ascontiguousarray(arr), GRAY32 if as_float else GRAY8)
    doc.luts = [LUT.grays() for _ in range(3)]
    doc.composite_mode = "grayscale"
    doc.active_channels = [True] * 3
    doc.slice_labels = ["Hue", "Saturation", "Brightness"] * n
    if as_float:
        doc.reset_display_range()
    doc.properties["HSB_Stack"] = "true"
    doc.hyperstack = n > 1


def to_hsb_stack(doc: ImageDoc) -> None:
    _hsb(doc, False)


def to_hsb32_stack(doc: ImageDoc) -> None:
    _hsb(doc, True)


def to_lab_stack(doc: ImageDoc) -> ImageDoc:
    lab = rgb_to_lab(doc.plane(0, 0, 0))
    out = ImageDoc.from_array(lab, "CYX", title=doc.title, calibration=doc.calibration.copy())
    out.composite_mode = "grayscale"
    out.luts = [LUT.grays() for _ in range(3)]
    out.slice_labels = ["L*", "a*", "b*"]
    out.reset_display_range()
    return out


def _stack_to_rgb(doc: ImageDoc, kind: str) -> None:
    p = _planes(doc)
    if kind == "rgb":
        b = p[2] if len(p) > 2 else np.zeros_like(p[0])
        rgb = np.stack([p[0], p[1], b], axis=-1)
        if _is_inverted_ij(doc.lut):
            rgb = 255 - rgb
    elif kind == "hsb":
        rgb = hsb_bytes_to_rgb(p[0], p[1], p[2])
    elif kind == "hsb32":
        rgb = hsb_to_rgb(p[0], p[1], p[2])
    else:
        rgb = lab_to_rgb(p[0], p[1], p[2])
    doc.replace_pixels(rgb[None, None, None].astype(np.uint8), COLOR_RGB)
    doc.luts = [LUT.grays()]
    doc.composite_mode = "color"
    doc.active_channels = [True]
    doc.slice_labels = [None]
    doc.properties.pop("HSB_Stack", None)


def _composite_to_rgb_stack(doc: ImageDoc) -> None:
    """Converter.convertCompositeToRGBStack: each channel rendered with its LUT, as RGB."""
    planes = []
    for c in range(doc.n_channels):
        lo, hi = doc.display_ranges[c]
        planes.append(gray_to_rgb(doc.plane(c, 0, 0), doc.type, doc.luts[c], lo, hi))
    arr = np.stack(planes)[None, :, None]
    doc.replace_pixels(np.ascontiguousarray(arr), COLOR_RGB)
    doc.luts = [LUT.grays()]
    doc.composite_mode = "color"
    doc.active_channels = [True]


def to_indexed_color(doc: ImageDoc, n_colors: int = 256) -> None:
    n_colors = min(max(int(n_colors), 2), 256)
    planes = _planes(doc)
    mc = MedianCut(np.concatenate([p.reshape(-1, 3) for p in planes]))
    idx, lut = mc.convert(n_colors)
    h, w = planes[0].shape[:2]
    out = [idx[i * h * w:(i + 1) * h * w].reshape(h, w) for i in range(len(planes))]
    labels = list(doc.slice_labels)
    doc.replace_pixels(_restack(doc, out), COLOR_256)
    doc.luts = [lut]
    doc.slice_labels = labels
    doc.display_ranges = [(0.0, 255.0)]

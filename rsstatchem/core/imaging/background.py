"""Process > Subtract Background: rolling ball and sliding paraboloid. **WP-B.**

Ported from ImageJ 1.54p ij/plugin/filter/BackgroundSubtracter.java (public domain, NIH;
rolling ball after Sternberg 1983 / NIH Image by M. Castle and J. Keller, sliding paraboloid
by M. Schmid 2007). Arithmetic is Java ``float`` (float32) step by step:

* the rolling ball (on the image shrunk by 1, 2, 4 or 8 with a trimmed ball patch) is a grey
  opening with the non-flat ball patch, computed here as an exact vectorised min/max over the
  patch offsets (min/max are order independent, the subtractions/additions are the same
  float32 operations as in Java), followed by ImageJ's bilinear enlargement;
* the sliding paraboloid runs ``lineSlideParabola`` line by line in x, y and the diagonals;
  per line the parabola search over the candidate points is vectorised (ImageJ's search limit
  only skips points that cannot win, so the touching points are the same);
* corner correction, 3x3 maximum/mean pre-smoothing and the light-background inversion follow
  the Java code.
"""
from __future__ import annotations

import numpy as np

from rsstatchem.core.imaging.fft import F32, hsb_to_rgb, java_int, rgb_to_hsb

FLOAT_MAX = np.finfo(np.float32).max


# ------------------------------------------------------------------------------- rolling ball
class RollingBall:
    """The (square, trimmed) ball patch and the shrink factor (``RollingBall``)."""

    def __init__(self, radius: float):
        if radius <= 10:
            self.shrink_factor, trim = 1, 24
        elif radius <= 30:
            self.shrink_factor, trim = 2, 24
        elif radius <= 100:
            self.shrink_factor, trim = 4, 32
        else:
            self.shrink_factor, trim = 8, 40
        small = radius / self.shrink_factor
        if small < 1:
            small = 1.0
        rsquare = small * small
        xtrim = int(trim * small) // 100
        half = int(np.floor(small - xtrim + 0.5))      # Math.round
        self.width = 2 * half + 1
        yy, xx = np.mgrid[0:self.width, 0:self.width]
        xval = (xx - half).astype(np.int64)
        yval = (yy - half).astype(np.int64)
        temp = rsquare - (xval * xval).astype(np.float64) - (yval * yval).astype(np.float64)
        self.data = np.where(temp > 0.0, np.sqrt(np.maximum(temp, 0.0)), 0.0).astype(F32)


def _filter3(a: np.ndarray, kind: str, axis: int) -> tuple[np.ndarray, np.ndarray | None]:
    """``filter3`` along ``axis`` for all lines (edge pixels replicated); returns the new
    array and, for "max", the float32 increments (max - v2)."""
    v2 = a
    if a.shape[axis] == 1:
        v1 = v3 = a
    else:
        first = np.take(a, [0], axis=axis)
        last = np.take(a, [a.shape[axis] - 1], axis=axis)
        n = a.shape[axis]
        v1 = np.concatenate([first, np.take(a, np.arange(0, n - 1), axis=axis)], axis=axis)
        v3 = np.concatenate([np.take(a, np.arange(1, n), axis=axis), last], axis=axis)
    if kind == "max":
        mx = np.where(v1 > v3, v1, v3)
        mx = np.where(v2 > mx, v2, mx).astype(F32)
        return mx, (mx - v2).astype(F32)
    return (((v1 + v2) + v3) * F32(0.33333333)).astype(F32), None


def filter3x3(a: np.ndarray, kind: str) -> tuple[np.ndarray, float]:
    """``filter3x3``: rows then columns; returns (new array, average shift for "max")."""
    out, d1 = _filter3(np.asarray(a, F32), kind, axis=1)
    out, d2 = _filter3(out, kind, axis=0)
    if kind != "max":
        return out, 0.0
    h, w = out.shape
    seq = np.concatenate([d1.ravel(), d2.T.ravel()]).astype(np.float64)
    shift = float(np.cumsum(seq)[-1]) if seq.size else 0.0
    return out, shift / w / h


def shrink_image(a: np.ndarray, sf: int) -> np.ndarray:
    """``shrinkImage``: minimum over sf x sf blocks."""
    h, w = a.shape
    sh, sw = (h + sf - 1) // sf, (w + sf - 1) // sf
    pad = np.full((sh * sf, sw * sf), FLOAT_MAX, F32)
    pad[:h, :w] = a
    return pad.reshape(sh, sf, sw, sf).min(axis=(1, 3)).astype(F32)


def roll_ball(ball: RollingBall, a: np.ndarray) -> np.ndarray:
    """``rollBall``: grey opening of ``a`` with the ball patch (positions beyond the edges
    included, only image pixels considered)."""
    a = np.asarray(a, F32)
    h, w = a.shape
    bw = ball.width
    r = bw // 2
    z = ball.data
    pad = np.full((h + 4 * r, w + 4 * r), np.inf, F32)
    pad[2 * r:2 * r + h, 2 * r:2 * r + w] = a
    eh, ew = h + 2 * r, w + 2 * r
    ero = np.full((eh, ew), np.inf, F32)
    for dy in range(bw):
        for dx in range(bw):
            np.minimum(ero, pad[dy:dy + eh, dx:dx + ew] - z[dy, dx], out=ero)
    out = np.full((h, w), -FLOAT_MAX, F32)
    for by in range(bw):
        for bx in range(bw):
            sy, sx = 2 * r - by, 2 * r - bx
            np.maximum(out, ero[sy:sy + h, sx:sx + w] + z[by, bx], out=out)
    return out


def _interpolation_arrays(length: int, small_length: int, sf: int):
    i = np.arange(length)
    idx = np.trunc((i - sf // 2) / sf).astype(np.int64)        # Java integer division
    idx = np.where(idx >= small_length - 1, small_length - 2, idx)
    distance = ((i.astype(F32) + F32(0.5)) / F32(sf) - (idx.astype(F32) + F32(0.5))).astype(F32)
    return idx, (F32(1) - distance).astype(F32)


def enlarge_image(small: np.ndarray, width: int, height: int, sf: int) -> np.ndarray:
    """``enlargeImage``: bilinear interpolation (extrapolation at the edges)."""
    sh, sw = small.shape
    xi, xw = _interpolation_arrays(width, sw, sf)
    yi, yw = _interpolation_arrays(height, sh, sf)
    lines = (small[:, xi] * xw + small[:, xi + 1] * (F32(1) - xw)).astype(F32)
    l0 = lines[yi]
    l1 = lines[yi + 1]
    wy = yw[:, None]
    return (l0 * wy + l1 * (F32(1) - wy)).astype(F32)


def rolling_ball_float_background(a: np.ndarray, radius: float, invert: bool,
                                  presmooth: bool, ball: RollingBall | None = None) -> np.ndarray:
    """``rollingBallFloatBackground``: the background of a float32 plane."""
    ball = ball or RollingBall(radius)
    px = np.array(a, dtype=F32, copy=True)
    if invert:
        px = -px
    if presmooth:
        px, _ = filter3x3(px, "mean")
    sf = ball.shrink_factor
    if sf > 1:
        small = roll_ball(ball, shrink_image(px, sf))
        px = enlarge_image(small, px.shape[1], px.shape[0], sf)
    else:
        px = roll_ball(ball, px)
    if invert:
        px = -px
    return px.astype(F32)


# ------------------------------------------------------------------------------- paraboloid
def line_slide_parabola(pixels: np.ndarray, idx: np.ndarray, coeff2, corrected: bool = False):
    """``lineSlideParabola`` on the line ``pixels.flat[idx]`` (modified in place). Returns the
    two corrected edge values when ``corrected``."""
    coeff2 = F32(coeff2)
    cache = pixels[idx].astype(F32)
    n = len(cache)
    curvature = F32(F32(1.999) * coeff2)
    if n >= 3:
        mid = cache[1:-1]
        test = (((mid + mid) - cache[:-2]) - cache[2:]) < curvature
        pts = np.nonzero(test)[0] + 1
    else:
        pts = np.zeros(0, np.int64)
    chain = np.concatenate([[0], pts, [n - 1]]).astype(np.int64)
    if n == 1:
        chain = np.array([0], np.int64)
    first_corner, last_corner = n - 1, 0
    i1, k = 0, 0
    while i1 < n - 1:
        v1 = cache[i1]
        cand = chain[k + 1:]
        d = (cand - i1).astype(F32)
        slopes = ((cache[cand] - v1) / d + coeff2 * d).astype(F32)
        valid = ~np.isnan(slopes)
        if not valid.any():
            m = 0
        else:
            m = int(np.argmin(np.where(valid, slopes, np.inf)))
        min_slope = slopes[m]
        i2 = int(cand[m])
        if i1 == 0:
            first_corner = i2
        if i2 == n - 1:
            last_corner = i1
        if i2 - i1 > 1:
            dj = np.arange(1, i2 - i1).astype(F32)
            pixels[idx[i1 + 1:i2]] = (v1 + dj * (min_slope - dj * coeff2)).astype(F32)
        k = k + 1 + m
        i1 = i2
    if not corrected:
        return None
    fc, lc = first_corner, last_corner
    if 4 * fc >= n:
        fc = 0
    if 4 * (n - 1 - lc) >= n:
        lc = n - 1
    v1, v2 = cache[fc], cache[lc]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        span = F32(lc - fc)
        slope = F32((v2 - v1) / span)
        value0 = F32(v1 - F32(slope * F32(fc)))
        coeff6 = F32(0)
        mid = F32(F32(0.5) * F32(lc + fc))
        two = F32(2)
        one = F32(1)
        for i in range((n + 2) // 3, (2 * n) // 3 + 1):
            dx = F32(F32(F32(F32(i) - mid) * two) / span)
            poly6 = F32(F32(dx * dx * dx * dx * dx * dx) - one)
            base = F32(value0 + F32(slope * F32(i)))
            if cache[i] < F32(base + F32(coeff6 * poly6)):
                coeff6 = F32(-F32(base - cache[i]) / poly6)
        dx = F32(F32(F32(F32(fc) - mid) * two) / span)
        e0 = F32(F32(value0 + F32(coeff6 * F32(F32(dx * dx * dx * dx * dx * dx) - one))) +
                 F32(F32(coeff2 * F32(fc)) * F32(fc)))
        dx = F32(F32(F32(F32(lc) - mid) * two) / span)
        k2 = n - 1 - lc
        e1 = F32(F32(F32(value0 + F32(F32(n - 1) * slope)) +
                     F32(coeff6 * F32(F32(dx * dx * dx * dx * dx * dx) - one))) +
                 F32(F32(coeff2 * F32(k2)) * F32(k2)))
    return e0, e1


def _line(start: int, inc: int, length: int) -> np.ndarray:
    return start + inc * np.arange(length, dtype=np.int64)


def correct_corners(px: np.ndarray, w: int, h: int, coeff2) -> None:
    """``correctCorners`` on the flat float32 array ``px`` (in place)."""
    coeff2 = F32(coeff2)
    corners = [F32(0)] * 4
    e = line_slide_parabola(px, _line(0, 1, w), coeff2, True)
    corners[0], corners[1] = e
    e = line_slide_parabola(px, _line((h - 1) * w, 1, w), coeff2, True)
    corners[2], corners[3] = e
    e = line_slide_parabola(px, _line(0, w, h), coeff2, True)
    corners[0] = F32(corners[0] + e[0])
    corners[2] = F32(corners[2] + e[1])
    e = line_slide_parabola(px, _line(w - 1, w, h), coeff2, True)
    corners[1] = F32(corners[1] + e[0])
    corners[3] = F32(corners[3] + e[1])
    diag = min(w, h)
    c2d = F32(F32(2) * coeff2)
    e = line_slide_parabola(px, _line(0, 1 + w, diag), c2d, True)
    corners[0] = F32(corners[0] + e[0])
    e = line_slide_parabola(px, _line(w - 1, -1 + w, diag), c2d, True)
    corners[1] = F32(corners[1] + e[0])
    e = line_slide_parabola(px, _line((h - 1) * w, 1 - w, diag), c2d, True)
    corners[2] = F32(corners[2] + e[0])
    e = line_slide_parabola(px, _line(w * h - 1, -1 - w, diag), c2d, True)
    corners[3] = F32(corners[3] + e[0])
    three = F32(3)
    for i, p in enumerate((0, w - 1, (h - 1) * w, w * h - 1)):
        lim = F32(corners[i] / three)
        if px[p] > lim:
            px[p] = lim


X_DIRECTION, Y_DIRECTION, DIAGONAL_1A, DIAGONAL_1B, DIAGONAL_2A, DIAGONAL_2B = range(6)


def filter1d(px: np.ndarray, w: int, h: int, direction: int, coeff2) -> None:
    """``filter1D``: slide the parabola along every line of one direction (in place)."""
    start_line, n_lines, line_inc, point_inc, length = 0, 0, 0, 0, 0
    if direction == X_DIRECTION:
        n_lines, line_inc, point_inc, length = h, w, 1, w
    elif direction == Y_DIRECTION:
        n_lines, line_inc, point_inc, length = w, 1, w, h
    elif direction == DIAGONAL_1A:
        n_lines, line_inc, point_inc = w - 2, 1, w + 1
    elif direction == DIAGONAL_1B:
        start_line, n_lines, line_inc, point_inc = 1, h - 2, w, w + 1
    elif direction == DIAGONAL_2A:
        start_line, n_lines, line_inc, point_inc = 2, w, 1, w - 1
    elif direction == DIAGONAL_2B:
        start_line, n_lines, line_inc, point_inc = 0, h - 2, w, w - 1
    for i in range(start_line, n_lines):
        start = i * line_inc
        if direction == DIAGONAL_2B:
            start += w - 1
        if direction == DIAGONAL_1A:
            length = min(h, w - i)
        elif direction == DIAGONAL_1B:
            length = min(w, h - i)
        elif direction == DIAGONAL_2A:
            length = min(h, i + 1)
        elif direction == DIAGONAL_2B:
            length = min(w, h - i)
        line_slide_parabola(px, _line(start, point_inc, length), coeff2)


def sliding_paraboloid_float_background(a: np.ndarray, radius: float, invert: bool,
                                        presmooth: bool, correct: bool = True) -> np.ndarray:
    """``slidingParaboloidFloatBackground``: the background of a float32 plane."""
    h, w = a.shape
    radius = F32(radius)
    coeff2 = F32(F32(0.5) / radius)
    coeff2diag = F32(F32(1.0) / radius)
    px = np.array(a, dtype=F32, copy=True)
    if invert:
        px = -px
    shift = F32(0)
    if presmooth:
        px, s = filter3x3(px, "max")
        shift = F32(s)
        px, _ = filter3x3(px, "mean")
    flat = px.ravel()
    if correct:
        correct_corners(flat, w, h, coeff2)
    for direction, c in ((X_DIRECTION, coeff2), (Y_DIRECTION, coeff2), (X_DIRECTION, coeff2),
                         (DIAGONAL_1A, coeff2diag), (DIAGONAL_1B, coeff2diag),
                         (DIAGONAL_2A, coeff2diag), (DIAGONAL_2B, coeff2diag),
                         (DIAGONAL_1A, coeff2diag), (DIAGONAL_1B, coeff2diag)):
        filter1d(flat, w, h, direction, c)
    px = flat.reshape(h, w)
    if invert:
        px = (-(px - shift)).astype(F32)
    elif presmooth:
        px = (px - shift).astype(F32)
    return px


# ------------------------------------------------------------------------------- public API
def background_of(channel: np.ndarray, radius: float, light_background: bool = False,
                  use_paraboloid: bool = False, presmooth: bool = True,
                  inverted_lut: bool = False, correct_corners_: bool = True,
                  ball: RollingBall | None = None) -> np.ndarray:
    """Float32 background of one channel (8/16-bit raw values or float data)."""
    invert = (inverted_lut and not light_background) or (not inverted_lut and light_background)
    fp = np.asarray(channel).astype(F32)
    if use_paraboloid:
        return sliding_paraboloid_float_background(fp, float(radius), invert, presmooth,
                                                   correct_corners_)
    return rolling_ball_float_background(fp, float(radius), invert, presmooth, ball)


def _store(orig: np.ndarray, bg: np.ndarray, create: bool, invert: bool) -> np.ndarray:
    """``rollingBallBackground`` output step for one integer channel (uint8/uint16)."""
    top = F32(255) if orig.dtype == np.uint8 else F32(65535)
    if create:
        v = (bg + F32(0.5)).astype(F32)
    else:
        offset = (top + F32(0.5)) if invert else F32(0.5)
        v = ((orig.astype(F32) - bg) + offset).astype(F32)
    v = np.where(v < 0, F32(0), np.where(v > top, top, v))
    return java_int(v).astype(orig.dtype)


def subtract_background(plane: np.ndarray, radius: float = 50.0, light_background: bool = False,
                        separate_colors: bool = False, create_background: bool = False,
                        use_paraboloid: bool = False, presmooth: bool = True,
                        inverted_lut: bool = False, correct_corners_: bool = True) -> np.ndarray:
    """Subtract Background on one plane (uint8/uint16/float32 (H, W) or uint8 RGB (H, W, 3)).
    Returns a new array of the same dtype (the background with ``create_background``)."""
    plane = np.asarray(plane)
    if radius <= 0.0001:
        raise ValueError("radius must be > 0")
    ball = None if use_paraboloid else RollingBall(radius)
    if plane.ndim == 3 and not separate_colors:
        hue, sat, bri = rgb_to_hsb(plane)
        H = (java_int(hue.astype(np.float64) * 255.0) & 0xFF).astype(np.uint8)
        S = (java_int(sat.astype(np.float64) * 255.0) & 0xFF).astype(np.uint8)
        B = (java_int(bri.astype(np.float64) * 255.0) & 0xFF).astype(np.uint8)
        B2 = subtract_background(B, radius, light_background, False, create_background,
                                 use_paraboloid, presmooth, False, correct_corners_)
        h2 = (H.astype(np.float64) / 255.0).astype(F32)
        s2 = (S.astype(np.float64) / 255.0).astype(F32)
        b2 = (B2.astype(np.float64) / 255.0).astype(F32)
        return hsb_to_rgb(h2, s2, b2)
    invert = (inverted_lut and not light_background) or (not inverted_lut and light_background)
    if plane.ndim == 3:
        inv_rgb = light_background
        out = np.empty_like(plane)
        for k in range(3):
            ch = plane[..., k]
            bg = background_of(ch, radius, light_background, use_paraboloid, presmooth, False,
                               correct_corners_, ball)
            out[..., k] = _store(ch, bg, create_background, inv_rgb)
        return out
    bg = background_of(plane, radius, light_background, use_paraboloid, presmooth,
                       inverted_lut, correct_corners_, ball)
    if plane.dtype == np.float32:
        if create_background:
            return bg
        return (plane.astype(F32) - bg).astype(F32)
    return _store(plane, bg, create_background, invert)

"""Region-of-interest gadgets on a graph (Origin's Gadgets menu) and the digitizer math.

A gadget works on the data of one plot inside an X range [x0, x1] (the ROI rectangle
dragged on the graph): Statistics, Integrate, FFT, Interpolate, Differentiate, Intersect,
Rise Time and the Vertical Cursor. Each function returns an ordered dict of results that the
UI writes to the Results Log or a report sheet. WP-1 adds Quick Fit / Quick Peaks through
:func:`register`. No Qt here.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy import integrate as sint


def roi_data(x, y, x0: float, x1: float, mask=None) -> tuple[np.ndarray, np.ndarray]:
    """Finite (x, y) points with x0 ≤ x ≤ x1, sorted by x. ``mask`` (True = excluded,
    WP-2's masked rows) is honoured when given."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    keep = np.isfinite(x) & np.isfinite(y)
    if mask is not None:
        m = np.zeros(n, bool)
        mk = np.asarray(mask, bool)[:n]
        m[:len(mk)] = mk
        keep &= ~m
    lo, hi = sorted((float(x0), float(x1)))
    keep &= (x >= lo) & (x <= hi)
    order = np.argsort(x[keep], kind="stable")
    return x[keep][order], y[keep][order]


def statistics(x, y, x0, x1, mask=None) -> dict:
    """Statistics gadget: N, mean, SD, SE, min/max (with their X), median, sum."""
    xs, ys = roi_data(x, y, x0, x1, mask)
    if not len(ys):
        raise ValueError("No data points inside the region.")
    n = len(ys)
    sd = float(ys.std(ddof=1)) if n > 1 else float("nan")
    return {"X From": min(x0, x1), "X To": max(x0, x1), "N": n, "Mean": float(ys.mean()),
            "SD": sd, "SE": sd / math.sqrt(n) if n > 1 else float("nan"),
            "Min": float(ys.min()), "X at Min": float(xs[ys.argmin()]),
            "Max": float(ys.max()), "X at Max": float(xs[ys.argmax()]),
            "Median": float(np.median(ys)), "Sum": float(ys.sum()),
            "Range": float(np.ptp(ys))}


def integrate(x, y, x0, x1, baseline: str = "zero", mask=None) -> dict:
    """Integrate gadget: trapezoid area over the ROI, to zero or to the straight line
    between the end points (Origin's "baseline: line"), with peak height, centroid and
    FWHM of the baseline-corrected curve."""
    xs, ys = roi_data(x, y, x0, x1, mask)
    if len(xs) < 2:
        raise ValueError("At least 2 points are needed inside the region.")
    if baseline == "line":
        base = ys[0] + (ys[-1] - ys[0]) * (xs - xs[0]) / ((xs[-1] - xs[0]) or 1.0)
    elif baseline == "min":
        base = np.full_like(ys, ys.min())
    else:
        base = np.zeros_like(ys)
    yc = ys - base
    area = float(sint.trapezoid(yc, xs))
    abs_area = float(sint.trapezoid(np.abs(yc), xs))
    k = int(np.argmax(np.abs(yc)))
    height = float(yc[k])
    centroid = float(sint.trapezoid(xs * yc, xs) / area) if area else float("nan")
    fwhm = _fwhm(xs, yc, k)
    return {"X From": float(xs[0]), "X To": float(xs[-1]), "Baseline": baseline,
            "Area": area, "Absolute Area": abs_area, "Peak X": float(xs[k]),
            "Height": height, "Centroid": centroid, "FWHM": fwhm, "N": len(xs)}


def _fwhm(xs, yc, k) -> float:
    """Full width at half of the extreme at index k (linear interpolation)."""
    half = yc[k] / 2.0
    if half == 0:
        return float("nan")
    s, h = yc * np.sign(half), abs(half)
    left = k
    while left > 0 and s[left] > h:
        left -= 1
    right = k
    while right < len(s) - 1 and s[right] > h:
        right += 1
    if s[left] > h or s[right] > h:
        return float("nan")
    xl = xs[left] + (h - s[left]) * (xs[left + 1] - xs[left]) / (s[left + 1] - s[left])
    xr = xs[right - 1] + (h - s[right - 1]) * (xs[right] - xs[right - 1]) / \
        (s[right] - s[right - 1])
    return float(xr - xl)


def fft(x, y, x0, x1, mask=None) -> dict:
    """FFT gadget: one-sided amplitude spectrum of the ROI (resampled onto an even grid
    when X is uneven) and the dominant frequency."""
    xs, ys = roi_data(x, y, x0, x1, mask)
    if len(xs) < 4:
        raise ValueError("At least 4 points are needed inside the region.")
    dx = np.diff(xs)
    if np.ptp(dx) > 1e-6 * abs(dx.mean()):
        grid = np.linspace(xs[0], xs[-1], len(xs))
        ys = np.interp(grid, xs, ys)
        xs = grid
    step = (xs[-1] - xs[0]) / (len(xs) - 1)
    yy = ys - ys.mean()
    spec = np.fft.rfft(yy)
    freq = np.fft.rfftfreq(len(yy), step)
    amp = np.abs(spec) * 2 / len(yy)
    amp[0] /= 2
    k = int(np.argmax(amp[1:]) + 1) if len(amp) > 1 else 0
    return {"N": len(xs), "Sampling Interval": float(step),
            "Dominant Frequency": float(freq[k]), "Amplitude": float(amp[k]),
            "Period": float(1 / freq[k]) if freq[k] else float("nan"),
            "_frequency": freq, "_amplitude": amp}


def interpolate(x, y, x0, x1, at: float, mask=None) -> dict:
    xs, ys = roi_data(x, y, x0, x1, mask)
    if len(xs) < 2:
        raise ValueError("At least 2 points are needed inside the region.")
    return {"X": float(at), "Y (linear)": float(np.interp(at, xs, ys))}


def differentiate(x, y, x0, x1, mask=None) -> dict:
    xs, ys = roi_data(x, y, x0, x1, mask)
    if len(xs) < 3:
        raise ValueError("At least 3 points are needed inside the region.")
    d = np.gradient(ys, xs)
    k = int(np.argmax(np.abs(d)))
    return {"Mean Slope": float((ys[-1] - ys[0]) / (xs[-1] - xs[0])),
            "Max |dY/dX|": float(d[k]), "X at Max Slope": float(xs[k]),
            "_x": xs, "_dydx": d}


def intersect(x1, y1, x2, y2, x0=-np.inf, xe=np.inf) -> list[tuple[float, float]]:
    """Intersections of two curves (Intersect gadget): linear interpolation of both onto
    the union of their X values, sign changes of the difference."""
    a = roi_data(x1, y1, max(x0, -1e300), min(xe, 1e300))
    b = roi_data(x2, y2, max(x0, -1e300), min(xe, 1e300))
    if len(a[0]) < 2 or len(b[0]) < 2:
        return []
    lo, hi = max(a[0][0], b[0][0]), min(a[0][-1], b[0][-1])
    if hi <= lo:
        return []
    grid = np.unique(np.r_[a[0], b[0]])
    grid = grid[(grid >= lo) & (grid <= hi)]
    d = np.interp(grid, *a) - np.interp(grid, *b)
    out = []
    for k in range(len(grid) - 1):
        if d[k] == 0:
            out.append((float(grid[k]), float(np.interp(grid[k], *a))))
        elif d[k] * d[k + 1] < 0:
            t = d[k] / (d[k] - d[k + 1])
            xv = grid[k] + t * (grid[k + 1] - grid[k])
            out.append((float(xv), float(np.interp(xv, *a))))
    if len(grid) and d[-1] == 0:
        out.append((float(grid[-1]), float(np.interp(grid[-1], *a))))
    return out


def rise_time(x, y, x0, x1, low: float = 10.0, high: float = 90.0, mask=None) -> dict:
    """Rise Time gadget: time between the low % and high % levels of the ROI's
    min→max swing (first crossings)."""
    xs, ys = roi_data(x, y, x0, x1, mask)
    if len(xs) < 3:
        raise ValueError("At least 3 points are needed inside the region.")
    y0, y1 = ys.min(), ys.max()
    lv, hv = y0 + (y1 - y0) * low / 100, y0 + (y1 - y0) * high / 100

    def cross(level):
        for k in range(len(ys) - 1):
            if (ys[k] - level) * (ys[k + 1] - level) <= 0 and ys[k + 1] != ys[k]:
                return xs[k] + (level - ys[k]) / (ys[k + 1] - ys[k]) * (xs[k + 1] - xs[k])
        return float("nan")
    tl, th = cross(lv), cross(hv)
    return {"Low Level": float(lv), "High Level": float(hv), "X at Low": float(tl),
            "X at High": float(th), "Rise Time": float(th - tl)}


def cursor(x, y, at: float) -> dict:
    """Vertical Cursor: Y of the curve at X (interpolated) and the nearest data point."""
    xs, ys = roi_data(x, y, -np.inf, np.inf)
    if not len(xs):
        raise ValueError("The plot has no data.")
    k = int(np.argmin(np.abs(xs - at)))
    return {"X": float(at), "Y (interpolated)": float(np.interp(at, xs, ys)),
            "Nearest X": float(xs[k]), "Nearest Y": float(ys[k]), "Index": k + 1}


GADGETS: dict[str, tuple[str, Callable]] = {
    "statistics": ("Statistics", statistics), "integrate": ("Integrate", integrate),
    "fft": ("FFT", fft), "differentiate": ("Differentiate", differentiate),
    "rise_time": ("Rise Time", rise_time),
}


def register(key: str, label: str, fn: Callable) -> None:
    """Plug in another ROI gadget (fn(x, y, x0, x1, mask=None) -> dict), e.g. WP-1's Quick
    Fit and Quick Peaks."""
    GADGETS[key] = (label, fn)


def public(result: dict) -> dict:
    """Results without the private array entries (keys starting with "_")."""
    return {k: v for k, v in result.items() if not k.startswith("_")}


# ====================================================================== digitizer
class Calibration:
    """Pixel → data mapping of a digitized graph image from two reference points per axis
    (Origin's Digitizer: X1, X2, Y1, Y2), linear or log10 per axis. Pixel Y grows
    downwards; rotated scans are handled because each axis is calibrated along the line
    through its two reference points."""

    def __init__(self, x_px: tuple, x_val: tuple, y_px: tuple, y_val: tuple,
                 x_log: bool = False, y_log: bool = False):
        self.x_px = [np.asarray(p, float) for p in x_px]
        self.y_px = [np.asarray(p, float) for p in y_px]
        self.x_log, self.y_log = x_log, y_log
        self.x_val = [self._t(v, x_log) for v in x_val]
        self.y_val = [self._t(v, y_log) for v in y_val]
        if np.allclose(self.x_px[0], self.x_px[1]) or np.allclose(self.y_px[0], self.y_px[1]):
            raise ValueError("The two reference points of an axis must differ.")
        if self.x_val[0] == self.x_val[1] or self.y_val[0] == self.y_val[1]:
            raise ValueError("The two reference values of an axis must differ.")
        # affine map pixel -> (u, v) where u, v are the axis coordinates in transformed units
        ex = self.x_px[1] - self.x_px[0]
        ey = self.y_px[1] - self.y_px[0]
        A = np.array([[ex[0], ey[0]], [ex[1], ey[1]]])
        if abs(np.linalg.det(A)) < 1e-9:
            raise ValueError("The X and Y reference points must not be on one line.")
        self._Ainv = np.linalg.inv(A)
        # origin: intersection of the two reference lines
        M = np.array([[ex[0], -ey[0]], [ex[1], -ey[1]]])
        st = np.linalg.solve(M, self.y_px[0] - self.x_px[0])
        self._origin = self.x_px[0] + st[0] * ex
        self._x_at_origin = self.x_val[0] + st[0] * (self.x_val[1] - self.x_val[0])
        s_y = (self._origin - self.y_px[0]) @ ey / (ey @ ey)
        self._y_at_origin = self.y_val[0] + s_y * (self.y_val[1] - self.y_val[0])

    @staticmethod
    def _t(v, log):
        v = float(v)
        if log:
            if v <= 0:
                raise ValueError("Log axes need positive reference values.")
            return math.log10(v)
        return v

    def to_data(self, px, py) -> tuple[float, float]:
        d = np.array([px, py], float) - self._origin
        a, b = self._Ainv @ d
        u = self._x_at_origin + a * (self.x_val[1] - self.x_val[0])
        v = self._y_at_origin + b * (self.y_val[1] - self.y_val[0])
        return float(10 ** u if self.x_log else u), float(10 ** v if self.y_log else v)

    def many(self, points) -> np.ndarray:
        return np.array([self.to_data(px, py) for px, py in points], float)


def auto_trace(gray: np.ndarray, color_mask: np.ndarray | None = None,
               step: int = 4) -> list[tuple[float, float]]:
    """Automatic curve extraction: for every ``step``-th pixel column, the centre of the
    dark (or ``color_mask``) pixels. ``gray`` is 0–255 with dark curve pixels."""
    g = np.asarray(gray, float)
    mask = color_mask if color_mask is not None else g < 100
    pts = []
    for c in range(0, mask.shape[1], max(1, step)):
        rows = np.flatnonzero(mask[:, c])
        if len(rows):
            pts.append((float(c), float(rows.mean())))
    return pts


def describe(result: dict) -> str:
    lines = []
    for k, v in public(result).items():
        if isinstance(v, float):
            lines.append(f"{k} = {v:.6g}")
        else:
            lines.append(f"{k} = {v}")
    return "\n".join(lines)


__all__ = ["Calibration", "GADGETS", "auto_trace", "cursor", "describe", "differentiate",
           "fft", "integrate", "interpolate", "intersect", "public", "register",
           "rise_time", "roi_data", "statistics"]

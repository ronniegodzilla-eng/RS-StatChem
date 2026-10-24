"""Origin's Analysis > Mathematics on (x, y) data: Differentiate, Integrate (area under the
curve with a baseline), Interpolate/Extrapolate (linear, cubic spline, Akima, PCHIP) and
Normalize. X defaults to the row number when no X variable is given, as in Origin.

Rows are sorted by x for the calculation; results are mapped back to the original rows so they
can be saved as new variables. Duplicate x values are averaged where a function of x is needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import interpolate as si
from scipy import signal as ss

from rsstatchem.core.analysis.common import (DATA, FIT, PALETTE, AnalysisResult, expand, figure,
                                             num, png, style, var_title, xy_frame)
from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table

INTERP_METHODS = {"linear": "Linear", "cubic": "Cubic Spline", "akima": "Akima",
                  "pchip": "PCHIP (shape-preserving)"}
BASELINES = {"zero": "Y = 0", "line": "Straight line between end points",
             "min": "Minimum of Y", "constant": "Constant value"}
NORMALIZE = {"max": "Divide by maximum", "range": "Normalize to [0, 1]",
             "zscore": "Z-score (mean 0, SD 1)", "area": "Area = 1 (needs X)",
             "sum": "Divide by sum", "value": "Divide by a specified value"}


def unique_mean(x, y) -> tuple[np.ndarray, np.ndarray]:
    """Sorted unique x with y averaged over duplicates."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    keys, inv = np.unique(x, return_inverse=True)
    return keys, np.bincount(inv, y) / np.bincount(inv)


# ---------------------------------------------------------------------- core functions
def differentiate(x, y, order: int = 1, method: str = "central", window: int = 7,
                  polyorder: int = 2) -> np.ndarray:
    """dⁿy/dxⁿ at each point. ``central``: second-order central differences on the (possibly
    uneven) x grid (``numpy.gradient``); ``savgol``: Savitzky-Golay derivative, uniform x."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    order_ = np.argsort(x, kind="stable")
    xs, ys = x[order_], y[order_]
    ux, uy = unique_mean(xs, ys)
    if len(ux) < 3:
        raise ValueError("At least 3 distinct X values are needed.")
    if method == "savgol":
        window = int(window) | 1
        window = min(window, len(ux) - (1 - len(ux) % 2))
        if polyorder >= window:
            raise ValueError("The polynomial order must be smaller than the window size.")
        if polyorder < order:
            raise ValueError("The polynomial order must be at least the derivative order.")
        dx = float(np.mean(np.diff(ux)))
        d = ss.savgol_filter(uy, window, polyorder, deriv=order, delta=dx, mode="interp")
    else:
        d = uy
        for _ in range(order):
            d = np.gradient(d, ux, edge_order=2)
    out = np.empty(len(x))
    out[order_] = np.interp(xs, ux, d)
    return out


def cumulative_area(x, y) -> np.ndarray:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    return np.concatenate([[0.0], np.cumsum(np.diff(x) * (y[1:] + y[:-1]) / 2)])


def integrate(x, y, baseline: str = "zero", value: float = 0.0, x_from: float | None = None,
              x_to: float | None = None) -> dict:
    """Trapezoidal area between the curve and a baseline over [x_from, x_to]."""
    xs, ys = unique_mean(x, y)
    lo = xs[0] if x_from is None else max(float(x_from), xs[0])
    hi = xs[-1] if x_to is None else min(float(x_to), xs[-1])
    if hi <= lo:
        raise ValueError("The integration range is empty.")
    inner = (xs > lo) & (xs < hi)
    xr = np.concatenate([[lo], xs[inner], [hi]])
    yr = np.concatenate([[np.interp(lo, xs, ys)], ys[inner], [np.interp(hi, xs, ys)]])
    if baseline == "zero":
        base = np.zeros_like(yr)
    elif baseline == "line":
        base = yr[0] + (yr[-1] - yr[0]) * (xr - xr[0]) / (xr[-1] - xr[0])
    elif baseline == "min":
        base = np.full_like(yr, yr.min())
    elif baseline == "constant":
        base = np.full_like(yr, float(value))
    else:
        raise ValueError(f"Unknown baseline '{baseline}'")
    h = yr - base
    area = float(cumulative_area(xr, h)[-1])
    abs_area = _abs_area(xr, h)
    imax = int(np.argmax(h))
    centroid = float(cumulative_area(xr, h * xr)[-1] / area) if area else float("nan")
    return {"area": area, "abs_area": abs_area, "x": xr, "y": yr, "base": base,
            "x_from": lo, "x_to": hi, "x_max": float(xr[imax]), "y_max": float(h[imax]),
            "centroid": centroid, "fwhm": _fwhm(xr, h, imax)}


def _abs_area(x, h) -> float:
    """∫|h| with exact splitting of trapezoids that cross zero."""
    total = 0.0
    for i in range(len(x) - 1):
        a, b, dx = h[i], h[i + 1], x[i + 1] - x[i]
        if a * b >= 0:
            total += abs(a + b) / 2 * dx
        else:
            t = a / (a - b) * dx
            total += abs(a) * t / 2 + abs(b) * (dx - t) / 2
    return float(total)


def _fwhm(x, h, imax) -> float:
    half = h[imax] / 2
    if h[imax] <= 0:
        return float("nan")
    i = imax
    while i > 0 and h[i] > half:
        i -= 1
    j = imax
    while j < len(h) - 1 and h[j] > half:
        j += 1
    if h[i] > half or h[j] > half:
        return float("nan")
    xl = np.interp(half, [h[i], h[i + 1]], [x[i], x[i + 1]])
    xr = np.interp(half, [h[j], h[j - 1]], [x[j], x[j - 1]])
    return float(xr - xl)


def interpolate(x, y, xnew, method: str = "linear", extrapolate: bool = False) -> np.ndarray:
    xs, ys = unique_mean(x, y)
    xnew = np.asarray(xnew, float)
    if len(xs) < 2:
        raise ValueError("At least 2 distinct X values are needed.")
    if method == "linear":
        f = si.interp1d(xs, ys, kind="linear", bounds_error=False,
                        fill_value="extrapolate" if extrapolate else np.nan)
        return f(xnew)
    if method == "cubic":
        if len(xs) < 4:
            raise ValueError("A cubic spline needs at least 4 distinct X values.")
        f = si.CubicSpline(xs, ys, bc_type="not-a-knot", extrapolate=extrapolate)
    elif method == "akima":
        if len(xs) < 5:
            raise ValueError("Akima interpolation needs at least 5 distinct X values.")
        f = si.Akima1DInterpolator(xs, ys, extrapolate=extrapolate)
    elif method == "pchip":
        f = si.PchipInterpolator(xs, ys, extrapolate=extrapolate)
    else:
        raise ValueError(f"Unknown method '{method}'")
    out = f(xnew)
    if not extrapolate:
        out = np.where((xnew < xs[0]) | (xnew > xs[-1]), np.nan, out)
    return out


def normalize(y, method: str = "max", x=None, value: float | None = None) -> np.ndarray:
    y = np.asarray(y, float)
    if method == "max":
        d = np.max(y)
        return y / d if d else _zero_div()
    if method == "range":
        lo, hi = np.min(y), np.max(y)
        return (y - lo) / (hi - lo) if hi > lo else _zero_div()
    if method == "zscore":
        sd = np.std(y, ddof=1)
        return (y - np.mean(y)) / sd if sd > 0 else _zero_div()
    if method == "sum":
        s = np.sum(y)
        return y / s if s else _zero_div()
    if method == "value":
        if not value:
            raise ValueError("Enter a non-zero value to divide by.")
        return y / value
    if method == "area":
        if x is None:
            x = np.arange(1, len(y) + 1, dtype=float)
        x = np.asarray(x, float)
        o = np.argsort(x, kind="stable")
        a = cumulative_area(x[o], y[o])[-1]
        return y / a if a else _zero_div()
    raise ValueError(f"Unknown method '{method}'")


def _zero_div():
    raise ValueError("Cannot normalize: the divisor is zero.")


# ---------------------------------------------------------------------- reports
def _xlabel(ds, xname):
    return var_title(ds, xname)


def differentiate_report(ds: Dataset, xname: str | None, ynames: list[str], order: int = 1,
                         method: str = "central", window: int = 7, polyorder: int = 2,
                         plot: bool = True) -> AnalysisResult:
    section = Section("Differentiate")
    section.log(f"DIFFERENTIATE X={xname or '(row)'} Y={' '.join(ynames)} /ORDER={order} "
                f"/METHOD={method.upper()}" + (f"({window},{polyorder})" if method == "savgol"
                                               else "") + ".")
    out = AnalysisResult(section)
    rows = []
    fig = figure(6.2, 2.2 + 1.8 * len(ynames)) if plot else None
    for k, y in enumerate(ynames):
        fr = xy_frame(ds, xname, [y])
        d = differentiate(fr["x"], fr[y], order, method, window, polyorder)
        label = "Derivative" if order == 1 else "2nd Derivative"
        out.saved[f"D{order}_{y}"[:60]] = (f"{label} of {var_title(ds, y)}",
                                           expand(d, fr.index, ds))
        rows.append([var_title(ds, y), str(len(fr)), num(np.min(d)), num(np.max(d)),
                     num(fr["x"].to_numpy()[np.argmax(d)]), num(fr["x"].to_numpy()[np.argmin(d)])])
        if fig is not None:
            ax = fig.add_subplot(len(ynames), 1, k + 1)
            o = np.argsort(fr["x"].to_numpy(), kind="stable")
            ax.plot(fr["x"].to_numpy()[o], d[o], color=FIT, lw=1.4)
            ax.axhline(0, color="#999", lw=0.6)
            style(ax, f"{label} of {var_title(ds, y)}", _xlabel(ds, xname),
                  "dy/dx" if order == 1 else "d²y/dx²")
    section.table(Table("Derivative Summary", ["", "N", "Minimum", "Maximum", "X at Maximum",
                                               "X at Minimum"], rows,
                        footnotes=["Central differences on the actual X spacing (Origin "
                                   "averages the slopes of the two neighbouring segments)."
                                   if method == "central" else
                                   "Savitzky-Golay derivative; assumes evenly spaced X."]))
    if fig is not None:
        section.image("Derivative Plot", png(fig))
    return out


def integrate_report(ds: Dataset, xname: str | None, ynames: list[str], baseline: str = "zero",
                     value: float = 0.0, x_from: float | None = None,
                     x_to: float | None = None, save_cumulative: bool = False,
                     plot: bool = True) -> AnalysisResult:
    section = Section("Integrate")
    section.log(f"INTEGRATE X={xname or '(row)'} Y={' '.join(ynames)} /BASELINE="
                f"{baseline.upper()}" + (f" /RANGE={x_from:g} TO {x_to:g}"
                                         if x_from is not None and x_to is not None else "")
                + ".")
    out = AnalysisResult(section)
    rows = []
    fig = figure(6.2, 2.4 + 1.9 * len(ynames)) if plot else None
    for k, y in enumerate(ynames):
        fr = xy_frame(ds, xname, [y])
        r = integrate(fr["x"], fr[y], baseline, value, x_from, x_to)
        rows.append([var_title(ds, y), num(r["x_from"]), num(r["x_to"]), num(r["area"]),
                     num(r["abs_area"]), num(r["x_max"]), num(r["y_max"]), num(r["fwhm"]),
                     num(r["centroid"])])
        if save_cumulative:
            o = np.argsort(fr["x"].to_numpy(), kind="stable")
            cum = np.empty(len(fr))
            cum[o] = cumulative_area(fr["x"].to_numpy()[o], fr[y].to_numpy()[o])
            out.saved[f"INT_{y}"[:60]] = (f"Cumulative integral of {var_title(ds, y)}",
                                          expand(cum, fr.index, ds))
        if fig is not None:
            ax = fig.add_subplot(len(ynames), 1, k + 1)
            o = np.argsort(fr["x"].to_numpy(), kind="stable")
            ax.plot(fr["x"].to_numpy()[o], fr[y].to_numpy()[o], color=DATA, lw=1.2)
            ax.fill_between(r["x"], r["base"], r["y"], color="#EF9A9A", alpha=0.6, lw=0)
            ax.plot(r["x"], r["base"], color=FIT, lw=1, ls="--")
            style(ax, f"Area of {var_title(ds, y)} = {num(r['area'], 5)}", _xlabel(ds, xname),
                  var_title(ds, y))
    section.table(Table("Integration Result", ["", "X From", "X To", "Area", "Absolute Area",
                                               "X at Ymax", "Ymax", "FWHM", "Centroid"], rows,
                        footnotes=[f"Trapezoidal rule; baseline: {BASELINES[baseline]}"
                                   + (f" ({value:g})" if baseline == "constant" else "")
                                   + ". Ymax, FWHM and centroid refer to Y minus baseline."]))
    if fig is not None:
        section.image("Integration Plot", png(fig))
    return out


def interpolate_report(ds: Dataset, xname: str, yname: str, method: str = "cubic",
                       target: str | None = None, n_points: int = 100,
                       x_from: float | None = None, x_to: float | None = None,
                       extrapolate: bool = False, save: bool = True) -> AnalysisResult:
    fr = xy_frame(ds, xname, [yname])
    x, y = fr["x"].to_numpy(), fr[yname].to_numpy()
    section = Section("Interpolate/Extrapolate")
    section.log(f"INTERPOLATE X={xname} Y={yname} /METHOD={method.upper()}"
                + (f" /AT={target}" if target else f" /POINTS={n_points}")
                + (" /EXTRAPOLATE" if extrapolate else "") + ".")
    out = AnalysisResult(section)
    if target:
        tx = pd.to_numeric(ds.analysis_series(target), errors="coerce").dropna()
        xnew = tx.to_numpy(float)
        ynew = interpolate(x, y, xnew, method, extrapolate)
        if save:
            out.saved[f"IP_{yname}"[:60]] = (f"{INTERP_METHODS[method]} interpolation of "
                                             f"{var_title(ds, yname)} at {target}",
                                             expand(ynew, tx.index, ds))
    else:
        lo = float(np.min(x)) if x_from is None else float(x_from)
        hi = float(np.max(x)) if x_to is None else float(x_to)
        xnew = np.linspace(lo, hi, int(n_points))
        ynew = interpolate(x, y, xnew, method, extrapolate)
        if save:
            if len(xnew) <= ds.n_cases:
                idx = ds.df.index[: len(xnew)]
                out.saved[f"IPX_{yname}"[:60]] = ("Interpolation X grid", expand(xnew, idx, ds))
                out.saved[f"IP_{yname}"[:60]] = (f"{INTERP_METHODS[method]} interpolation of "
                                                 f"{var_title(ds, yname)}",
                                                 expand(ynew, idx, ds))
            else:
                section.note("Note", f"{len(xnew)} points exceed the {ds.n_cases} cases of the "
                             "data file; the grid was not saved.", style="warning")
    outside = int(np.sum((xnew < x.min()) | (xnew > x.max())))
    section.table(Table("Interpolation Summary", ["", "Value"], [
        ["Method", INTERP_METHODS[method]], ["Input points", str(len(x))],
        ["Output points", str(len(xnew))], ["Points outside the data range", str(outside)],
        ["Extrapolation", "Allowed" if extrapolate else "Not allowed (missing)"]]))
    shown = min(len(xnew), 50)
    rows = [[str(i + 1), num(xnew[i]), num(ynew[i])] for i in range(shown)]
    section.table(Table("Interpolated Values" + (f" (first {shown})" if shown < len(xnew)
                                                 else ""),
                        ["", var_title(ds, xname), var_title(ds, yname)], rows))
    fig = figure(6.2, 4.0)
    ax = fig.add_subplot()
    o = np.argsort(xnew)
    ax.plot(xnew[o], ynew[o], color=FIT, lw=1.4, label=INTERP_METHODS[method],
            marker="." if len(xnew) <= 40 else None)
    ax.scatter(x, y, s=22, color=DATA, zorder=3, label="Data")
    style(ax, "Interpolate/Extrapolate", var_title(ds, xname), var_title(ds, yname))
    ax.legend(fontsize=8, frameon=False)
    section.image("Interpolation Plot", png(fig))
    return out


def normalize_report(ds: Dataset, ynames: list[str], method: str = "max",
                     xname: str | None = None, value: float | None = None,
                     plot: bool = True) -> AnalysisResult:
    section = Section("Normalize")
    section.log(f"NORMALIZE Y={' '.join(ynames)} /METHOD={method.upper()}"
                + (f"({value:g})" if method == "value" and value else "") + ".")
    out = AnalysisResult(section)
    rows = []
    fig = figure(6.2, 3.6) if plot else None
    ax = fig.add_subplot() if fig is not None else None
    for k, y in enumerate(ynames):
        fr = xy_frame(ds, xname, [y])
        v = normalize(fr[y].to_numpy(), method, fr["x"].to_numpy() if method == "area" else None,
                      value)
        out.saved[f"N_{y}"[:60]] = (f"{NORMALIZE[method]}: {var_title(ds, y)}",
                                    expand(v, fr.index, ds))
        rows.append([var_title(ds, y), str(len(fr)), num(np.min(v)), num(np.max(v)),
                     num(np.mean(v)), num(np.std(v, ddof=1) if len(v) > 1 else np.nan)])
        if ax is not None:
            o = np.argsort(fr["x"].to_numpy(), kind="stable")
            ax.plot(fr["x"].to_numpy()[o], v[o], lw=1.3, color=PALETTE[k % len(PALETTE)],
                    label=var_title(ds, y))
    section.table(Table("Normalized Variables", ["", "N", "Minimum", "Maximum", "Mean",
                                                 "Std. Deviation"], rows,
                        footnotes=[f"Method: {NORMALIZE[method]}."]))
    if ax is not None:
        style(ax, "Normalized Data", var_title(ds, xname), "Normalized value")
        ax.legend(fontsize=8, frameon=False)
        section.image("Normalized Data", png(fig))
    return out

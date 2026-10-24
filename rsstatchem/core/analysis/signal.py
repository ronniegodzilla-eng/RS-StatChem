"""Origin's Analysis > Signal Processing and Peaks and Baseline tools.

* Smoothing: Savitzky-Golay, adjacent averaging, percentile filter (median at 50 %), FFT
  low-pass filter and LOWESS (statsmodels).
* FFT power spectrum with window functions (one-sided amplitude and mean-square amplitude).
* Baseline: improved iterative modified polynomial (I-ModPoly, Zhao et al. 2007) and
  asymmetric least squares (Eilers & Boelens 2005).
* Peak finding (``scipy.signal.find_peaks``) with position, height, prominence, FWHM, area
  and centroid; multiple-peak fitting with a sum of Gaussian / Lorentzian / Voigt peaks.

Savitzky-Golay, the percentile filter, FFT and ALS assume evenly spaced X; a warning is
printed when the spacing varies by more than 1 %.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage, sparse
from scipy import signal as ss
from scipy.sparse.linalg import spsolve

from rsstatchem.core.analysis import functions as fl
from rsstatchem.core.analysis.common import (DATA, FIT, AnalysisResult, expand, figure, intf,
                                             num, png, style, var_title, xy_frame)
from rsstatchem.core.analysis.fitting import (fit_curve, fit_plot, notes_table,
                                              parameter_table, residual_plot,
                                              statistics_table)
from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table

SMOOTHERS = {"savgol": "Savitzky-Golay", "average": "Adjacent-Averaging",
             "percentile": "Percentile Filter", "fft": "FFT Filter (low-pass)",
             "lowess": "LOWESS"}
WINDOWS = {"none": "Rectangle (none)", "hann": "Hanning", "hamming": "Hamming",
           "blackman": "Blackman"}
BASELINE_METHODS = {"poly": "Polynomial (iterative I-ModPoly)",
                    "als": "Asymmetric Least Squares"}
SHAPES = {"gauss": "Gaussian", "lorentz": "Lorentzian", "voigt": "Voigt"}


def _sorted(x, y):
    o = np.argsort(np.asarray(x, float), kind="stable")
    return o, np.asarray(x, float)[o], np.asarray(y, float)[o]


def uneven(x) -> bool:
    d = np.diff(np.sort(np.asarray(x, float)))
    if len(d) < 2 or np.mean(d) == 0:
        return False
    return bool(np.max(np.abs(d - np.mean(d))) > 0.01 * abs(np.mean(d)))


def _odd_window(window: int, n: int) -> int:
    w = max(3, int(window) | 1)
    return min(w, n if n % 2 else n - 1)


# ---------------------------------------------------------------------- smoothing
def smooth(x, y, method: str = "savgol", window: int = 7, polyorder: int = 2,
           percentile: float = 50.0, cutoff: float = 0.1, frac: float = 0.2) -> np.ndarray:
    """Smoothed y in the original point order. ``cutoff`` is a fraction of the Nyquist
    frequency (FFT filter); ``frac`` the LOWESS span fraction."""
    o, xs, ys = _sorted(x, y)
    n = len(ys)
    if n < 3:
        raise ValueError("At least 3 points are needed.")
    if method == "savgol":
        w = _odd_window(window, n)
        if polyorder >= w:
            raise ValueError("The polynomial order must be smaller than the window size.")
        s = ss.savgol_filter(ys, w, int(polyorder), mode="interp")
    elif method == "average":
        s = ndimage.uniform_filter1d(ys, size=max(1, int(window)), mode="nearest")
    elif method == "percentile":
        s = ndimage.percentile_filter(ys, float(percentile), size=max(1, int(window)),
                                      mode="nearest")
    elif method == "fft":
        if not 0 < cutoff <= 1:
            raise ValueError("The cutoff must be a fraction of the Nyquist frequency (0–1].")
        trend = np.linspace(ys[0], ys[-1], n)  # remove end mismatch to limit ringing
        spec = np.fft.rfft(ys - trend)
        freq = np.fft.rfftfreq(n)  # cycles/sample, Nyquist = 0.5
        spec[freq > cutoff * 0.5] = 0
        s = np.fft.irfft(spec, n) + trend
    elif method == "lowess":
        from statsmodels.nonparametric.smoothers_lowess import lowess
        s = lowess(ys, xs, frac=float(frac), it=3, return_sorted=False)
    else:
        raise ValueError(f"Unknown smoothing method '{method}'")
    out = np.empty(n)
    out[o] = s
    return out


# ---------------------------------------------------------------------- FFT
def power_spectrum(x, y, window: str = "hann", detrend: bool = True) -> dict:
    """One-sided spectrum. ``amplitude`` is the peak amplitude of each sinusoid (window
    coherent-gain corrected); ``power`` is Origin's mean-square amplitude (MSA)."""
    o, xs, ys = _sorted(x, y)
    n = len(ys)
    if n < 4:
        raise ValueError("At least 4 points are needed for an FFT.")
    dt = float(np.mean(np.diff(xs))) if n > 1 else 1.0
    if dt <= 0:
        raise ValueError("X must increase.")
    v = ys - np.mean(ys) if detrend else ys.copy()
    win = np.ones(n) if window == "none" else ss.get_window(window, n, fftbins=False)
    spec = np.fft.rfft(v * win)
    freq = np.fft.rfftfreq(n, d=dt)
    amp = np.abs(spec) / np.sum(win)
    amp[1:] *= 2
    if n % 2 == 0:
        amp[-1] /= 2
    power = amp ** 2 / 2
    power[0] = amp[0] ** 2
    if n % 2 == 0:
        power[-1] = amp[-1] ** 2
    return {"freq": freq, "amplitude": amp, "power": power, "phase": np.angle(spec,
                                                                                deg=True),
            "dt": dt, "n": n, "uneven": uneven(xs)}


# ---------------------------------------------------------------------- baseline
def baseline_poly(x, y, order: int = 2, max_iter: int = 200, tol: float = 1e-3) -> np.ndarray:
    """Improved modified polynomial baseline (I-ModPoly, Zhao et al. 2007): peak points are
    dropped after the first fit, then points above fit + SD are clipped until the residual SD
    stabilises. Unlike plain ModPoly it does not sink into the noise."""
    o, xs, ys = _sorted(x, y)
    if len(ys) <= order + 1:
        raise ValueError("Not enough points for this polynomial order.")
    P = np.polynomial.polynomial
    u = (xs - np.mean(xs)) / (np.ptp(xs) or 1.0)
    coef = P.polyfit(u, ys, order)
    fit = P.polyval(u, coef)
    dev = float(np.std(ys - fit))
    keep = ys <= fit + dev
    if keep.sum() <= order + 1:
        keep = np.ones(len(ys), bool)
    uu, work = u[keep], ys[keep]
    coef = P.polyfit(uu, work, order)
    fit = P.polyval(uu, coef)
    for _ in range(max_iter):
        new_dev = float(np.std(work - fit))
        work = np.minimum(work, fit + new_dev)
        coef = P.polyfit(uu, work, order)
        fit = P.polyval(uu, coef)
        if abs(new_dev - dev) <= tol * (new_dev or 1.0):
            break
        dev = new_dev
    out = np.empty(len(ys))
    out[o] = P.polyval(u, coef)
    return out


def baseline_als(y, lam: float = 1e5, p: float = 0.01, n_iter: int = 10) -> np.ndarray:
    """Eilers & Boelens asymmetric least squares baseline (y in X order, even spacing)."""
    y = np.asarray(y, float)
    n = len(y)
    if n < 3:
        raise ValueError("At least 3 points are needed.")
    D = sparse.diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(n, n - 2))
    H = lam * (D @ D.T)
    w = np.ones(n)
    z = y
    for _ in range(int(n_iter)):
        W = sparse.spdiags(w, 0, n, n)
        z = spsolve(sparse.csc_matrix(W + H), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return np.asarray(z)


def baseline(x, y, method: str = "als", order: int = 2, lam: float = 1e5, p: float = 0.01,
             n_iter: int = 10) -> np.ndarray:
    if method == "poly":
        return baseline_poly(x, y, order)
    if method == "als":
        o, xs, ys = _sorted(x, y)
        z = baseline_als(ys, lam, p, n_iter)
        out = np.empty(len(ys))
        out[o] = z
        return out
    raise ValueError(f"Unknown baseline method '{method}'")


# ---------------------------------------------------------------------- peaks
def find_peaks(x, y, direction: str = "positive", min_height: float | None = None,
               min_prominence: float = 0.05, min_distance: int = 1,
               min_width: float = 0.0) -> list[dict]:
    """Peaks with position, height, prominence, FWHM, area and centroid.

    ``min_prominence`` is a fraction of the Y range; ``min_width`` is in X units.
    For negative peaks the signal is inverted for detection and heights are reported as-is.
    """
    o, xs, ys = _sorted(x, y)
    sign = -1.0 if direction == "negative" else 1.0
    v = sign * ys
    rng = float(np.ptp(v)) or 1.0
    kw = {"prominence": max(min_prominence, 0) * rng, "distance": max(1, int(min_distance))}
    if min_height is not None:
        kw["height"] = sign * min_height
    idx, props = ss.find_peaks(v, **kw)
    if not len(idx):
        return []
    half = ss.peak_widths(v, idx, rel_height=0.5, prominence_data=(props["prominences"],
                                                                   props["left_bases"],
                                                                   props["right_bases"]))
    full = ss.peak_widths(v, idx, rel_height=1.0, prominence_data=(props["prominences"],
                                                                   props["left_bases"],
                                                                   props["right_bases"]))
    pos = np.arange(len(xs))
    valleys = [int(a + np.argmin(v[a:b + 1])) for a, b in zip(idx[:-1], idx[1:])]
    out = []
    for k, i in enumerate(idx):
        xl, xr = np.interp(half[2][k], pos, xs), np.interp(half[3][k], pos, xs)
        left = max(full[2][k], valleys[k - 1]) if k > 0 else full[2][k]
        right = min(full[3][k], valleys[k]) if k < len(idx) - 1 else full[3][k]
        fl_, fr_ = np.interp(left, pos, xs), np.interp(right, pos, xs)
        area, centroid = _segment_area(xs, ys, fl_, fr_)
        fwhm = float(xr - xl)
        if fwhm < min_width:
            continue
        out.append({"index": int(o[i]), "x": float(xs[i]), "y": float(ys[i]),
                    "prominence": float(props["prominences"][k]), "fwhm": fwhm,
                    "left": float(fl_), "right": float(fr_), "half_left": float(xl),
                    "half_right": float(xr), "half_height": float(sign * half[1][k]),
                    "area": area, "centroid": centroid})
    return out


def _segment_area(xs, ys, a, b) -> tuple[float, float]:
    inner = (xs > a) & (xs < b)
    xx = np.concatenate([[a], xs[inner], [b]])
    yy = np.concatenate([[np.interp(a, xs, ys)], ys[inner], [np.interp(b, xs, ys)]])
    area = float(np.trapezoid(yy, xx))
    centroid = float(np.trapezoid(yy * xx, xx) / area) if area else float("nan")
    return area, centroid


def multi_peak_fit(x, y, n_peaks: int | None = None, shape: str = "gauss",
                   baseline: str = "constant", centers=None, min_prominence: float = 0.05,
                   scale_errors: bool = True):
    """Fit a sum of peaks. Starting values come from :func:`find_peaks` (the ``n_peaks`` most
    prominent; all found when None) or from user ``centers``."""
    o, xs, ys = _sorted(x, y)
    span = float(np.ptp(xs)) or 1.0
    base0 = float(np.percentile(ys, 5)) if baseline != "none" else 0.0
    found = find_peaks(xs, ys - base0, min_prominence=min_prominence)
    if centers:
        peaks = []
        for c in centers:
            near = min(found, key=lambda p: abs(p["x"] - c)) if found else None
            if near and abs(near["x"] - c) < 0.05 * span:
                peaks.append({**near, "x": float(c)})
            else:
                yc = float(np.interp(c, xs, ys)) - base0
                peaks.append({"x": float(c), "y": yc + base0, "fwhm": span / 20,
                              "area": yc * span / 20})
    else:
        if not found:
            raise ValueError("No peaks were found; lower the minimum prominence or enter "
                             "peak centres.")
        found.sort(key=lambda p: -p["prominence"])
        peaks = sorted(found[: n_peaks] if n_peaks else found, key=lambda p: p["x"])
        if n_peaks and len(peaks) < n_peaks:
            raise ValueError(f"Only {len(peaks)} peak(s) found; lower the minimum prominence "
                             "or enter the peak centres.")
    model = fl.peak_sum(len(peaks), shape, baseline)
    p0, lo, hi = {}, {}, {}
    if baseline != "none":
        p0["y0"] = base0
    if baseline == "linear":
        p0["k"] = 0.0
    dx = float(np.min(np.diff(np.unique(xs)))) if len(xs) > 1 else span / 100
    for i, pk in enumerate(peaks, start=1):
        h = pk["y"] - base0
        w = max(pk["fwhm"], 2 * dx)
        p0[f"xc{i}"] = pk["x"]
        lo[f"xc{i}"], hi[f"xc{i}"] = xs[0] - 0.1 * span, xs[-1] + 0.1 * span
        if shape == "voigt":
            p0[f"wG{i}"], p0[f"wL{i}"] = 0.7 * w, 0.5 * w
            lo[f"wG{i}"], hi[f"wG{i}"] = dx / 10, 2 * span
            lo[f"wL{i}"], hi[f"wL{i}"] = 0.0, 2 * span
            p0[f"A{i}"] = h * w * 1.2
        else:
            p0[f"w{i}"] = w
            lo[f"w{i}"], hi[f"w{i}"] = dx / 10, 2 * span
            p0[f"A{i}"] = h * w * (1.0645 if shape == "gauss" else math.pi / 2)
    bounds = {k: (lo[k], hi[k]) for k in lo}
    return fit_curve(xs, ys, model, params0=p0, bounds=bounds, scale_errors=scale_errors)


def peak_summary(res) -> list[dict]:
    """Centre, height, FWHM, area (with SE) of each peak of a :func:`multi_peak_fit` result."""
    meta = res.model.meta
    nb, per = meta["n_base"], meta["per"]
    p, e = res.values, res.errors
    total = 0.0
    rows = []
    for i in range(meta["n_peaks"]):
        j = nb + i * len(per)
        q = dict(zip(per, p[j: j + len(per)]))
        qe = dict(zip(per, e[j: j + len(per)]))
        if meta["shape"] == "voigt":
            fwhm = fl.voigt_fwhm(q["wG"], q["wL"])
            fwhm_se = float("nan")
            height = float(fl._voigt(np.array([q["xc"]]), q["xc"], q["A"], q["wG"], q["wL"])[0])
        else:
            fwhm, fwhm_se = abs(q["w"]), qe["w"]
            height = (q["A"] / (abs(q["w"]) / (2 * math.sqrt(2 * math.log(2))))
                      / math.sqrt(2 * math.pi)) if meta["shape"] == "gauss" else \
                2 * q["A"] / (math.pi * abs(q["w"]))
        total += q["A"]
        rows.append({"peak": i + 1, "xc": q["xc"], "xc_se": qe["xc"], "height": height,
                     "fwhm": fwhm, "fwhm_se": fwhm_se, "area": q["A"], "area_se": qe["A"]})
    for r in rows:
        r["area_pct"] = 100 * r["area"] / total if total else float("nan")
    return rows


def peak_components(res, xx) -> list[np.ndarray]:
    """Each fitted peak on the grid ``xx`` (baseline added) for plotting."""
    meta = res.model.meta
    nb, per = meta["n_base"], meta["per"]
    a_pos = per.index("A")
    comps = []
    for i in range(meta["n_peaks"]):
        q = np.array(res.values, float)
        for j in range(meta["n_peaks"]):
            if j != i:
                q[nb + j * len(per) + a_pos] = 0.0
        comps.append(res.model(xx, *q))
    return comps


# ---------------------------------------------------------------------- reports
def _uneven_note(section: Section, x) -> None:
    if uneven(x):
        section.note("Warning", "X is not evenly spaced (spacing varies by more than 1 %); "
                     "this method assumes even spacing.", style="warning")


def smooth_report(ds: Dataset, xname: str | None, ynames: list[str], method: str = "savgol",
                  window: int = 7, polyorder: int = 2, percentile: float = 50.0,
                  cutoff: float = 0.1, frac: float = 0.2, plot: bool = True) -> AnalysisResult:
    section = Section("Smoothing")
    params = {"savgol": f"POINTS={window} ORDER={polyorder}", "average": f"POINTS={window}",
              "percentile": f"POINTS={window} PERCENTILE={percentile:g}",
              "fft": f"CUTOFF={cutoff:g}", "lowess": f"SPAN={frac:g}"}[method]
    section.log(f"SMOOTH X={xname or '(row)'} Y={' '.join(ynames)} /METHOD={method.upper()} "
                f"{params}.")
    out = AnalysisResult(section)
    rows = []
    fig = figure(6.2, 2.4 + 1.9 * len(ynames)) if plot else None
    for k, y in enumerate(ynames):
        fr = xy_frame(ds, xname, [y])
        x, v = fr["x"].to_numpy(), fr[y].to_numpy()
        if k == 0 and method in ("savgol", "percentile", "fft", "average"):
            _uneven_note(section, x)
        s = smooth(x, v, method, window, polyorder, percentile, cutoff, frac)
        out.saved[f"S_{y}"[:60]] = (f"{SMOOTHERS[method]} smoothing of {var_title(ds, y)}",
                                    expand(s, fr.index, ds))
        resid = v - s
        rows.append([var_title(ds, y), intf(len(v)), num(np.std(resid, ddof=1)),
                     num(np.std(v, ddof=1)), num(np.std(s, ddof=1))])
        if fig is not None:
            ax = fig.add_subplot(len(ynames), 1, k + 1)
            o = np.argsort(x, kind="stable")
            ax.plot(x[o], v[o], color="#9e9e9e", lw=0.9, label="Raw")
            ax.plot(x[o], s[o], color=FIT, lw=1.6, label="Smoothed")
            style(ax, f"{SMOOTHERS[method]}: {var_title(ds, y)}", var_title(ds, xname),
                  var_title(ds, y))
            ax.legend(fontsize=8, frameon=False)
    section.table(Table("Smoothing Summary", ["", "N", "SD of Removed Noise", "SD of Raw",
                                              "SD of Smoothed"], rows,
                        footnotes=[f"Method: {SMOOTHERS[method]} ({params.lower()})."]))
    if fig is not None:
        section.image("Smoothed Data", png(fig))
    return out


def fft_report(ds: Dataset, xname: str | None, yname: str, window: str = "hann",
               detrend: bool = True, interval: float | None = None, n_top: int = 10,
               log_scale: bool = False) -> AnalysisResult:
    fr = xy_frame(ds, xname, [yname])
    x = fr["x"].to_numpy()
    if not xname and interval:
        x = (x - 1) * float(interval)
    r = power_spectrum(x, fr[yname].to_numpy(), window, detrend)
    section = Section("FFT")
    section.log(f"FFT X={xname or '(row)'} Y={yname} /WINDOW={window.upper()}"
                + (" /DETREND=MEAN" if detrend else "") + ".")
    if r["uneven"]:
        _uneven_note(section, x)
    f, amp, pw = r["freq"], r["amplitude"], r["power"]
    section.table(Table("FFT Settings", ["", "Value"], [
        ["Number of Points", intf(r["n"])], ["Sampling Interval", num(r["dt"])],
        ["Frequency Resolution", num(1 / (r["n"] * r["dt"]))],
        ["Nyquist Frequency", num(0.5 / r["dt"])], ["Window", WINDOWS[window]],
        ["Mean Removed", "Yes" if detrend else "No"]]))
    cand, _ = ss.find_peaks(np.concatenate([[0], amp[1:], [0]]))
    cand = cand[(cand >= 1) & (cand < len(amp))]
    top = cand[np.argsort(amp[cand])[::-1][:n_top]] if len(cand) else np.array([], int)
    rows = [[intf(k + 1), num(f[i]), num(1 / f[i]) if f[i] else "--", num(amp[i]), num(pw[i]),
             num(r["phase"][i], 4)] for k, i in enumerate(top)]
    section.table(Table("Dominant Frequencies", ["Rank", "Frequency", "Period", "Amplitude",
                                                 "Power (MSA)", "Phase (°)"], rows,
                        row_header_cols=1,
                        footnotes=["Amplitude = peak amplitude of the sinusoid (window "
                                   "gain-corrected); Power = mean square amplitude."]))
    fig = figure(6.2, 5.2)
    ax1 = fig.add_subplot(2, 1, 1)
    o = np.argsort(x, kind="stable")
    ax1.plot(x[o], fr[yname].to_numpy()[o], color=DATA, lw=0.9)
    style(ax1, f"Signal: {var_title(ds, yname)}", var_title(ds, xname, "Time"),
          var_title(ds, yname))
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(f, pw if log_scale else amp, color=FIT, lw=1.1)
    if log_scale:
        ax2.set_yscale("log")
    for i in top[:3]:
        ax2.annotate(num(f[i], 4), (f[i], pw[i] if log_scale else amp[i]), fontsize=7,
                     xytext=(3, 3), textcoords="offset points")
    style(ax2, "Power Spectrum" if log_scale else "Amplitude Spectrum", "Frequency",
          "Power (MSA)" if log_scale else "Amplitude")
    section.image("FFT Spectrum", png(fig))
    return AnalysisResult(section)


def baseline_report(ds: Dataset, xname: str | None, yname: str, method: str = "als",
                    order: int = 2, lam: float = 1e5, p: float = 0.01, n_iter: int = 10,
                    save: bool = True) -> AnalysisResult:
    fr = xy_frame(ds, xname, [yname])
    x, v = fr["x"].to_numpy(), fr[yname].to_numpy()
    b = baseline(x, v, method, order, lam, p, n_iter)
    section = Section("Baseline Correction")
    params = f"ORDER={order}" if method == "poly" else f"LAMBDA={lam:g} P={p:g} ITER={n_iter}"
    section.log(f"BASELINE X={xname or '(row)'} Y={yname} /METHOD={method.upper()} {params}.")
    if method == "als":
        _uneven_note(section, x)
    corr = v - b
    section.table(Table("Baseline Summary", ["", "Value"], [
        ["Method", BASELINE_METHODS[method]], ["Parameters", params.replace(" ", ", ")],
        ["Number of Points", intf(len(v))], ["Baseline Minimum", num(np.min(b))],
        ["Baseline Maximum", num(np.max(b))],
        ["Area of Corrected Signal", num(np.trapezoid(corr[np.argsort(x)], np.sort(x)))]]))
    out = AnalysisResult(section)
    if save:
        out.saved[f"BL_{yname}"[:60]] = (f"Baseline of {var_title(ds, yname)}",
                                         expand(b, fr.index, ds))
        out.saved[f"BC_{yname}"[:60]] = (f"Baseline-corrected {var_title(ds, yname)}",
                                         expand(corr, fr.index, ds))
    fig = figure(6.2, 5.0)
    o = np.argsort(x, kind="stable")
    ax = fig.add_subplot(2, 1, 1)
    ax.plot(x[o], v[o], color=DATA, lw=1.0, label="Signal")
    ax.plot(x[o], b[o], color=FIT, lw=1.6, label="Baseline")
    style(ax, f"Baseline: {var_title(ds, yname)}", "", var_title(ds, yname))
    ax.legend(fontsize=8, frameon=False)
    ax = fig.add_subplot(2, 1, 2)
    ax.plot(x[o], corr[o], color="#1565C0", lw=1.0)
    ax.axhline(0, color="#999", lw=0.6)
    style(ax, "Baseline-corrected", var_title(ds, xname), "Corrected")
    section.image("Baseline Plot", png(fig))
    return out


def peaks_report(ds: Dataset, xname: str | None, yname: str, direction: str = "positive",
                 min_height: float | None = None, min_prominence: float = 0.05,
                 min_distance: int = 1, min_width: float = 0.0) -> AnalysisResult:
    fr = xy_frame(ds, xname, [yname])
    x, v = fr["x"].to_numpy(), fr[yname].to_numpy()
    peaks = find_peaks(x, v, direction, min_height, min_prominence, min_distance, min_width)
    section = Section("Find Peaks")
    section.log(f"PEAKS X={xname or '(row)'} Y={yname} /DIRECTION={direction.upper()} "
                f"/PROMINENCE={min_prominence:g}" + (f" /HEIGHT={min_height:g}"
                                                     if min_height is not None else "")
                + f" /DISTANCE={min_distance}.")
    if not peaks:
        section.note("Find Peaks", "No peaks satisfy the criteria.", style="warning")
    total = sum(p["area"] for p in peaks) or float("nan")
    rows = [[intf(k + 1), intf(p["index"] + 1), num(p["x"]), num(p["y"]), num(p["prominence"]),
             num(p["fwhm"]), num(p["left"]), num(p["right"]), num(p["area"]),
             num(100 * p["area"] / total, 4), num(p["centroid"])] for k, p in enumerate(peaks)]
    section.table(Table("Peak Analysis", ["Peak", "Row", "X", "Height", "Prominence", "FWHM",
                                          "Left X", "Right X", "Area", "Area %", "Centroid"],
                        rows, footnotes=[
                            "Height = Y at the peak; FWHM at half prominence; area = "
                            "trapezoidal integral of Y between the left and right bases "
                            "(baseline Y = 0; correct the baseline first if needed)."]))
    fig = figure(6.2, 4.0)
    ax = fig.add_subplot()
    o = np.argsort(x, kind="stable")
    ax.plot(x[o], v[o], color=DATA, lw=1.0)
    for k, p in enumerate(peaks):
        ax.plot(p["x"], p["y"], "v" if direction == "positive" else "^", color=FIT, ms=7)
        ax.hlines(p["half_height"], p["half_left"], p["half_right"], color="#1565C0", lw=1)
        ax.annotate(str(k + 1), (p["x"], p["y"]), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=8, color=FIT)
    style(ax, f"Peaks of {var_title(ds, yname)}", var_title(ds, xname), var_title(ds, yname))
    section.image("Peak Plot", png(fig))
    return AnalysisResult(section)


def multi_peak_report(ds: Dataset, xname: str | None, yname: str, n_peaks: int | None = None,
                      shape: str = "gauss", baseline: str = "constant", centers=None,
                      min_prominence: float = 0.05, level: float = 0.95,
                      save_fitted: bool = False) -> AnalysisResult:
    fr = xy_frame(ds, xname, [yname])
    x, v = fr["x"].to_numpy(), fr[yname].to_numpy()
    res = multi_peak_fit(x, v, n_peaks, shape, baseline, centers, min_prominence)
    xt, yt = var_title(ds, xname), var_title(ds, yname)
    section = Section(f"Multiple Peak Fit ({res.model.meta['n_peaks']} × {SHAPES[shape]})")
    section.log(f"PEAKFIT X={xname or '(row)'} Y={yname} /SHAPE={shape.upper()} "
                f"/PEAKS={res.model.meta['n_peaks']} /BASELINE={baseline.upper()}.")
    section.table(notes_table(res, xt, yt))
    for note in res.notes:
        section.note("Warning", note, style="warning")
    section.table(parameter_table(res, yt, level))
    rows = [[intf(r["peak"]), num(r["xc"]), num(r["xc_se"]), num(r["height"]), num(r["fwhm"]),
             num(r["fwhm_se"]), num(r["area"]), num(r["area_se"]), num(r["area_pct"], 4)]
            for r in peak_summary(res)]
    section.table(Table("Peak Properties", ["Peak", "Center", "Center SE", "Height", "FWHM",
                                            "FWHM SE", "Area", "Area SE", "Area %"], rows,
                        footnotes=["Height is above the fitted baseline."]))
    section.table(statistics_table(res, yt))
    xx = np.linspace(x.min(), x.max(), 800)
    comps = peak_components(res, xx)
    section.image("Fitted Peaks", fit_plot(res, xt, yt, level, conf=False, pred=False,
                                           title=f"Multiple Peak Fit of {yt}",
                                           components=[(f"Peak {i + 1}", c)
                                                       for i, c in enumerate(comps)], grid=xx))
    section.image("Residual vs. Independent Plot", residual_plot(res, xt, yt))
    out = AnalysisResult(section)
    if save_fitted:
        out.saved["PKFIT"] = (f"Multiple peak fit of {yt}",
                              expand(res.predict(x), fr.index, ds))
    return out

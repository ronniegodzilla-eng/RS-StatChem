"""Analysis > Signal Processing tools of the Origin-style window: extra smoothers, FFT /
IFFT, FFT filters, IIR filters, STFT, convolution / deconvolution, correlation, coherence,
Hilbert transform / envelope, decimation and wavelets (optional PyWavelets).

Same pattern as :mod:`math_dialogs`: ``run_*(main, values, gwin)`` does the work and
``*_dialog(main)`` asks for the settings. Masked points are excluded from the input.
"""
from __future__ import annotations

import numpy as np

from rsstatchem.core.analysis import filters as flt
from rsstatchem.core.analysis import signal as sig
from rsstatchem.core.analysis import transforms as tr
from rsstatchem.core.analysis import wavelets as wv
from rsstatchem.core.analysis.common import num
from rsstatchem.ui.origin.math_dialogs import (_guard, add_columns, add_sheet, ask,
                                               new_matrix_book, plot_into, report, resolve,
                                               result_sheet)

SMOOTH_METHODS = {"loess": "LOESS (local quadratic)", "lowess": "LOWESS (local linear)",
                  "savgol": "Savitzky-Golay", "binomial": "Binomial",
                  "triangular": "Weighted Adjacent-Averaging",
                  "average": "Adjacent-Averaging", "percentile": "Percentile Filter"}


def _dt(s) -> float:
    dt = tr.sampling_interval(s.x)
    return dt


def _uneven_note(s) -> str:
    return " (X is unevenly spaced: the mean spacing was used)" if sig.uneven(s.x) else ""


# ====================================================================== smoothing
def run_smooth(main, v: dict, gwin=None) -> int:
    s = resolve(main, v["input"])
    m = v.get("method", "loess")
    o = np.argsort(s.x, kind="stable")
    if m in ("loess", "lowess", "binomial", "triangular"):
        y = flt.smooth_extra(s.x, s.y, m, int(v.get("window", 7)),
                             float(v.get("span", 0.3)), bool(v.get("robust", False)),
                             int(v.get("order", 2)))
    elif m == "savgol":
        ys = flt.savgol(s.y[o], int(v.get("window", 7)), int(v.get("polyorder", 2)),
                        int(v.get("deriv", 0)), _dt(s) if len(s.x) > 1 else 1.0,
                        v.get("boundary", "interp"))
        y = np.empty(len(ys))
        y[o] = ys
    else:
        y = sig.smooth(s.x, s.y, m, int(v.get("window", 7)), int(v.get("polyorder", 2)),
                       float(v.get("percentile", 50.0)))
    label = SMOOTH_METHODS[m] + (" derivative" if m == "savgol" and v.get("deriv") else "")
    idx = add_columns(main, s, [{"long_name": f"{s.title} ({label})", "values": y}],
                      "Smooth")
    report(main, f"Smooth {s.label}: {label}")
    plot_into(main, gwin, s.book, s.ws, s.xcol, idx[0])
    return idx[0]


def smooth_dialog(main) -> None:
    got = ask(main, "Smooth", [
        ("method", "Method:", "choice", "loess", SMOOTH_METHODS),
        ("span", "Span (fraction of points, LOESS/LOWESS):", "float", 0.3, (0.01, 1, 3)),
        ("robust", "Robust (bisquare iterations):", "bool", False),
        ("window", "Points of Window:", "int", 7, (2, 100001)),
        ("polyorder", "Polynomial Order (SG):", "int", 2, (0, 12)),
        ("deriv", "Derivative Order (SG):", "int", 0, (0, 6)),
        ("boundary", "Boundary (SG):", "choice", "interp", flt.SG_MODES),
        ("order", "Binomial Order:", "int", 2, (1, 50)),
        ("percentile", "Percentile:", "float", 50.0, (0, 100, 1))])
    if got:
        _guard(main, lambda: run_smooth(main, got[0], got[1]), "Smooth")


# ====================================================================== FFT
FFT_OUTPUTS = ("real", "imag", "magnitude", "amplitude", "phase", "msa", "psd", "db")
FFT_TITLES = {"real": "Re", "imag": "Im", "magnitude": "Magnitude", "amplitude": "Amplitude",
              "phase": "Phase", "msa": "Power (MSA)", "psd": "PSD", "db": "dB"}


def run_fft(main, v: dict, gwin=None):
    s = resolve(main, v["input"])
    dt = _dt(s)
    o = np.argsort(s.x, kind="stable")
    r = tr.fft(s.y[o], dt, v.get("window", "rect"), bool(v.get("one_sided", True)),
               bool(v.get("shift", True)), "pow2" if v.get("pad") else None,
               v.get("detrend", "none"), phase_unit=v.get("phase_unit", "deg"),
               unwrap=bool(v.get("unwrap", False)))
    unit = s.ws.columns[s.xcol].units if s.xcol is not None else ""
    cols = [("Frequency", r["freq"], "X", f"1/{unit}" if unit else "",
             f"FFT of {s.label}, window {tr.WINDOWS.get(r['window'], r['window'])}")]
    for k in FFT_OUTPUTS:
        if v.get(k, k in ("real", "imag", "amplitude", "phase")):
            cols.append((FFT_TITLES[k], r[k], "Y", "deg" if k == "phase" and
                         v.get("phase_unit", "deg") == "deg" else ""))
    sheet = add_sheet(main, s.book, result_sheet("FFT", cols), "FFT")
    report(main, f"FFT of {s.label}: N = {r['n']} (FFT length {r['n_fft']}), Δt = "
                 f"{num(dt)}, Δf = {num(r['df'])}" + _uneven_note(s))
    return sheet


def fft_dialog(main) -> None:
    got = ask(main, "FFT", [
        ("window", "Window:", "choice", "rect", tr.WINDOWS),
        ("one_sided", "One-sided spectrum:", "bool", True),
        ("shift", "Shift (two-sided: 0 Hz in the middle):", "bool", True),
        ("pad", "Zero-pad to a power of 2:", "bool", False),
        ("detrend", "Detrend:", "choice", "none", {"none": "None", "mean": "Subtract mean",
                                                   "linear": "Subtract linear trend"}),
        ("phase_unit", "Phase unit:", "choice", "deg", {"deg": "Degrees", "rad": "Radians"}),
        ("unwrap", "Unwrap phase:", "bool", False),
        ("real", "Output Re:", "bool", True), ("imag", "Output Im:", "bool", True),
        ("magnitude", "Output Magnitude:", "bool", False),
        ("amplitude", "Output Amplitude:", "bool", True),
        ("phase", "Output Phase:", "bool", True), ("msa", "Output Power (MSA):", "bool", False),
        ("psd", "Output PSD:", "bool", False), ("db", "Output dB:", "bool", False)])
    if got:
        _guard(main, lambda: run_fft(main, got[0], got[1]), "FFT")


def run_ifft(main, v: dict, gwin=None):
    """Inverse FFT: ``input`` is the Re column (against Frequency), ``input2`` the Im
    column (or none); the spectrum must be two-sided."""
    re_ = resolve(main, v["input"])
    im = resolve(main, v["input2"]).y if v.get("input2") is not None and \
        v.get("use_imag", True) else None
    df = float(np.mean(np.diff(re_.x))) if len(re_.x) > 1 else 1.0
    r = tr.ifft(re_.y, im, abs(df), bool(v.get("shifted", True)))
    sheet = add_sheet(main, re_.book, result_sheet("IFFT", [
        ("Time", r["t"], "X", "", f"IFFT of {re_.label}"), ("Re", r["real"], "Y"),
        ("Im", r["imag"], "Y")]), "IFFT")
    report(main, f"IFFT of {re_.label}: N = {len(r['t'])}, Δt = {num(r['dt'])}")
    return sheet


def ifft_dialog(main) -> None:
    got = ask(main, "IFFT", [("use_imag", "Use the imaginary column:", "bool", True),
                             ("shifted", "Spectrum is shifted (0 in the middle):", "bool",
                              True)], n_inputs=2, input_labels=("Re:", "Im:"))
    if got:
        _guard(main, lambda: run_ifft(main, got[0], got[1]), "IFFT")


# ====================================================================== filters
def run_fft_filter(main, v: dict, gwin=None) -> int:
    s = resolve(main, v["input"])
    o = np.argsort(s.x, kind="stable")
    ys = flt.fft_filter(s.y[o], _dt(s), v.get("kind", "lowpass"),
                               float(v.get("cutoff", 0.1)), float(v.get("cutoff2", 0.0)) or
                               None, float(v.get("threshold", 0.0)),
                               bool(v.get("keep_dc", True)),
                               "ends" if v.get("trend", False) else "none")
    y = np.empty(len(ys))
    y[o] = ys
    label = flt.FFT_FILTERS[v.get("kind", "lowpass")]
    idx = add_columns(main, s, [{"long_name": f"{s.title} ({label})", "values": y}],
                      "FFT Filter")
    report(main, f"FFT Filter {label} on {s.label}" + _uneven_note(s))
    plot_into(main, gwin, s.book, s.ws, s.xcol, idx[0])
    return idx[0]


def fft_filter_dialog(main) -> None:
    got = ask(main, "FFT Filters", [
        ("kind", "Filter Type:", "choice", "lowpass", flt.FFT_FILTERS),
        ("cutoff", "Cutoff Frequency (1/X unit):", "float", 0.1, (0, 1e12, 6)),
        ("cutoff2", "Upper Cutoff (band filters):", "float", 0.0, (0, 1e12, 6)),
        ("threshold", "Amplitude Threshold:", "float", 0.0, (0, 1e12, 6)),
        ("keep_dc", "Keep DC offset:", "bool", True),
        ("trend", "Remove the end-point line first:", "bool", False)])
    if got:
        _guard(main, lambda: run_fft_filter(main, got[0], got[1]), "FFT Filter")


def run_iir(main, v: dict, gwin=None) -> int:
    s = resolve(main, v["input"])
    dt = _dt(s)
    fs = 1.0 / dt
    band = v.get("btype", "lowpass")
    cut = [float(v.get("cutoff", 0.1))] + ([float(v.get("cutoff2", 0.2))]
                                           if band in ("bandpass", "bandstop") else [])
    sos = flt.iir_design(v.get("ftype", "butter"), band, int(v.get("order", 4)), cut, fs,
                         float(v.get("rp", 1.0)), float(v.get("rs", 40.0)))
    o = np.argsort(s.x, kind="stable")
    ys = flt.iir_apply(s.y[o], sos, bool(v.get("zero_phase", True)))
    y = np.empty(len(ys))
    y[o] = ys
    label = f"{flt.IIR_TYPES[v.get('ftype', 'butter')]} {flt.BAND_TYPES[band]}"
    idx = add_columns(main, s, [{"long_name": f"{s.title} ({label})", "values": y}],
                      "IIR Filter")
    if v.get("response", True):
        r = flt.iir_response(sos, fs)
        add_sheet(main, s.book, result_sheet("IIRResponse", [
            ("Frequency", r["freq"], "X", "", label), ("Gain", r["gain"], "Y"),
            ("Gain (dB)", r["db"], "Y"), ("Phase", r["phase"], "Y", "deg"),
            ("Group Delay", r["group_delay"], "Y", "samples")]), "IIR Response", show=False)
    report(main, f"IIR Filter {label}, order {int(v.get('order', 4))}, cutoff "
                 f"{', '.join(num(c) for c in cut)} on {s.label}" +
           (" (zero phase)" if v.get("zero_phase", True) else ""))
    plot_into(main, gwin, s.book, s.ws, s.xcol, idx[0])
    return idx[0]


def iir_dialog(main) -> None:
    got = ask(main, "IIR Filter Design", [
        ("ftype", "Filter:", "choice", "butter", flt.IIR_TYPES),
        ("btype", "Type:", "choice", "lowpass", flt.BAND_TYPES),
        ("order", "Order:", "int", 4, (1, 20)),
        ("cutoff", "Cutoff Frequency (1/X unit):", "float", 0.1, (0, 1e12, 6)),
        ("cutoff2", "Upper Cutoff (band):", "float", 0.2, (0, 1e12, 6)),
        ("rp", "Pass-band ripple (dB):", "float", 1.0, (0.001, 60, 3)),
        ("rs", "Stop-band attenuation (dB):", "float", 40.0, (1, 200, 1)),
        ("zero_phase", "Zero phase (filter forward and backward):", "bool", True),
        ("response", "Output the frequency response sheet:", "bool", True)])
    if got:
        _guard(main, lambda: run_iir(main, got[0], got[1]), "IIR Filter")


# ====================================================================== STFT
def run_stft(main, v: dict, gwin=None):
    s = resolve(main, v["input"])
    o = np.argsort(s.x, kind="stable")
    dt = _dt(s)
    r = tr.stft(s.y[o], dt, int(v.get("nperseg", 64)), int(v.get("noverlap", 32)),
                v.get("window", "hann"), v.get("scale", "magnitude"))
    t0 = float(np.min(s.x))
    times = r["times"] + t0
    mb = new_matrix_book(main, r["values"], (times[0], times[-1], r["freqs"][0],
                                             r["freqs"][-1]), "STFT", f"STFT of {s.label}",
                         (s.x_title, "Frequency", v.get("scale", "magnitude").title()))
    report(main, f"STFT of {s.label}: {r['values'].shape[0]} frequencies × "
                 f"{r['values'].shape[1]} times → {mb.name}")
    return mb


def stft_dialog(main) -> None:
    got = ask(main, "STFT", [
        ("nperseg", "Window length (points):", "int", 64, (4, 1_000_000)),
        ("noverlap", "Overlap (points):", "int", 32, (0, 1_000_000)),
        ("window", "Window:", "choice", "hann", tr.WINDOWS),
        ("scale", "Output:", "choice", "magnitude", {"magnitude": "Magnitude",
                                                     "power": "Power", "db": "dB"})])
    if got:
        _guard(main, lambda: run_stft(main, got[0], got[1]), "STFT")


# ====================================================================== convolution
def run_convolution(main, v: dict, gwin=None, deconvolve: bool = False):
    s, h = resolve(main, v["input"]), resolve(main, v["input2"])
    mode = v.get("mode", "linear")
    if deconvolve:
        out = tr.deconvolve(s.y, h.y, mode, bool(v.get("normalize", False)),
                            float(v.get("regularization", 0.0)))
        name, text = "Deconvolution", "Deconvolve"
    else:
        out = tr.convolve(s.y, h.y, mode, bool(v.get("normalize", False)))
        name, text = "Convolution", "Convolve"
    if len(out) == len(s.y):
        idx = add_columns(main, s, [{"long_name": f"{s.title} ({name.lower()})",
                                     "values": out}], name)
        plot_into(main, gwin, s.book, s.ws, s.xcol, idx[0])
        result = idx[0]
    else:
        dt = tr.sampling_interval(s.x) if len(s.x) > 1 else 1.0
        x = float(np.min(s.x)) + dt * np.arange(len(out))
        result = add_sheet(main, s.book, result_sheet(name, [
            (s.x_title, x, "X", "", f"{name} of {s.label} with {h.label}"),
            (name, out, "Y")]), name)
    report(main, f"{text} {s.label} with {h.label} ({mode})")
    return result


def convolution_dialog(main, deconvolve: bool = False) -> None:
    title = "Deconvolution" if deconvolve else "Convolution"
    fields = [("mode", "Mode:", "choice", "linear", {"linear": "Linear",
                                                     "circular": "Circular"}),
              ("normalize", "Normalize the response (unit sum):", "bool", False)]
    if deconvolve:
        fields.append(("regularization", "Regularization ε (0 = exact):", "float", 0.0,
                       (0, 1, 8)))
    got = ask(main, title, fields, n_inputs=2, input_labels=("Signal:", "Response:"))
    if got:
        _guard(main, lambda: run_convolution(main, got[0], got[1], deconvolve), title)


# ====================================================================== correlation
def run_correlation(main, v: dict, gwin=None):
    a = resolve(main, v["input"])
    auto = v.get("auto", False) or v.get("input2") is None
    b = None if auto else resolve(main, v["input2"])
    dt = tr.sampling_interval(a.x) if len(a.x) > 1 else 1.0
    r = tr.correlate(a.y, None if b is None else b.y, v.get("normalize", "coeff"),
                     bool(v.get("demean", True)), bool(v.get("circular", False)), dt)
    title = "Autocorrelation" if b is None else "Cross-correlation"
    sheet = add_sheet(main, a.book, result_sheet("Corr", [
        ("Lag", r["lag"], "X", "", f"{title} of {a.label}" + ("" if b is None else
                                                              f" and {b.label}")),
        (title, r["r"], "Y")]), "Correlation")
    k = int(np.argmax(r["r"]))
    report(main, f"{title}: maximum {num(r['r'][k])} at lag {num(r['lag'][k])}")
    return sheet


def correlation_dialog(main) -> None:
    got = ask(main, "Correlation", [
        ("auto", "Autocorrelation (ignore the second input):", "bool", False),
        ("normalize", "Normalization:", "choice", "coeff", tr.CORR_NORMS),
        ("demean", "Subtract the means:", "bool", True),
        ("circular", "Circular:", "bool", False)], n_inputs=2,
        input_labels=("Signal 1:", "Signal 2:"))
    if got:
        _guard(main, lambda: run_correlation(main, got[0], got[1]), "Correlation")


def run_coherence(main, v: dict, gwin=None):
    a, b = resolve(main, v["input"]), resolve(main, v["input2"])
    n = min(len(a.y), len(b.y))
    r = tr.coherence(a.y[:n], b.y[:n], _dt(a), int(v.get("nperseg", 64)),
                     v.get("window", "hann"))
    return add_sheet(main, a.book, result_sheet("Coherence", [
        ("Frequency", r["freq"], "X", "", f"Coherence of {a.label} and {b.label}"),
        ("Coherence", r["coherence"], "Y")]), "Coherence")


def coherence_dialog(main) -> None:
    got = ask(main, "Coherence", [
        ("nperseg", "Segment length (points):", "int", 64, (4, 1_000_000)),
        ("window", "Window:", "choice", "hann", tr.WINDOWS)], n_inputs=2,
        input_labels=("Signal 1:", "Signal 2:"))
    if got:
        _guard(main, lambda: run_coherence(main, got[0], got[1]), "Coherence")


# ====================================================================== Hilbert
def run_hilbert(main, v: dict, gwin=None) -> list[int]:
    s = resolve(main, v["input"])
    o = np.argsort(s.x, kind="stable")
    r = tr.hilbert(s.y[o])

    def back(a):
        out = np.empty(len(a))
        out[o] = a
        return out
    cols = []
    if v.get("transform", True):
        cols.append({"long_name": f"Hilbert({s.title})", "values": back(r["hilbert"])})
    if v.get("envelope", True):
        cols.append({"long_name": f"Envelope({s.title})", "values": back(r["envelope"])})
    if v.get("phase", False):
        cols.append({"long_name": f"Phase({s.title})", "values": back(r["phase"]),
                     "units": "rad"})
    idx = add_columns(main, s, cols, "Hilbert Transform")
    for i in idx:
        plot_into(main, gwin, s.book, s.ws, s.xcol, i)
    return idx


def hilbert_dialog(main, envelope_only: bool = False) -> None:
    fields = [] if envelope_only else [
        ("transform", "Output the Hilbert transform:", "bool", True),
        ("envelope", "Output the envelope:", "bool", True),
        ("phase", "Output the instantaneous phase:", "bool", False)]
    got = ask(main, "Envelope" if envelope_only else "Hilbert Transform", fields)
    if got:
        v = got[0]
        if envelope_only:
            v.update(transform=False, envelope=True, phase=False)
        _guard(main, lambda: run_hilbert(main, v, got[1]), "Hilbert")


def run_decimate(main, v: dict, gwin=None):
    s = resolve(main, v["input"])
    q = int(v.get("factor", 2))
    o = np.argsort(s.x, kind="stable")
    y = tr.decimate(s.y[o], q, v.get("ftype", "iir"))
    x = s.x[o][::q][:len(y)]
    return add_sheet(main, s.book, result_sheet("Decimated", [
        (s.x_title, x, "X", "", f"Decimation of {s.label} by {q}"), (s.title, y, "Y")]),
        "Decimation")


def decimate_dialog(main) -> None:
    got = ask(main, "Decimation", [
        ("factor", "Factor:", "int", 2, (2, 1000)),
        ("ftype", "Anti-alias filter:", "choice", "iir", {"iir": "IIR (Chebyshev I, order 8)",
                                                          "fir": "FIR (30 taps)"})])
    if got:
        _guard(main, lambda: run_decimate(main, got[0], got[1]), "Decimation")


# ====================================================================== wavelets
def run_wavelet(main, op: str, v: dict, gwin=None):
    s = resolve(main, v["input"])
    o = np.argsort(s.x, kind="stable")
    ys = s.y[o]
    wav = v.get("wavelet", "db4")

    def back(a):
        out = np.empty(len(a))
        out[o] = a[:len(o)]
        return out
    if op == "cwt":
        r = wv.cwt(ys, _dt(s), v.get("cwt_wavelet", "morl"),
                   n_scales=int(v.get("n_scales", 64)))
        mb = new_matrix_book(main, np.abs(r["coef"]), (float(s.x[o][0]), float(s.x[o][-1]),
                                                       float(r["scales"][0]),
                                                       float(r["scales"][-1])),
                             "CWT", f"CWT of {s.label}", (s.x_title, "Scale", "|Coefficient|"))
        report(main, f"CWT of {s.label}: {len(r['scales'])} scales → {mb.name}")
        return mb
    if op == "dwt":
        r = wv.dwt(ys, wav, v.get("mode", "symmetric"))
        return add_sheet(main, s.book, result_sheet("DWT", [
            ("Approximation", r["approx"], "Y", "", f"DWT ({wav}) of {s.label}"),
            ("Detail", r["detail"], "Y")]), "DWT")
    if op == "decompose":
        r = wv.decompose(ys, wav, int(v.get("level", 0)) or None, v.get("mode", "symmetric"))
        idx = add_columns(main, s, [{"long_name": f"{name} ({wav})", "values": back(c)}
                                    for name, c in r["components"].items()],
                          "Multi-Scale DWT")
        report(main, f"Multi-scale DWT of {s.label}: level {r['level']}, {wav}")
        return idx
    if op == "denoise":
        r = wv.denoise(ys, wav, int(v.get("level", 0)) or None, v.get("threshold_mode", "soft"),
                       float(v.get("threshold")) if v.get("threshold") else None,
                       v.get("rule", "universal"))
        idx = add_columns(main, s, [{"long_name": f"{s.title} (wavelet denoised)",
                                     "values": back(r["signal"])}], "Wavelet Denoise")
        report(main, f"Wavelet denoise of {s.label}: {wav}, level {r['level']}, threshold "
                     f"{', '.join(num(t) for t in r['thresholds'][:3])}")
        plot_into(main, gwin, s.book, s.ws, s.xcol, idx[0])
        return idx[0]
    if op == "smooth":
        y = wv.smooth(ys, wav, int(v.get("drop", 1)), int(v.get("level", 0)) or None)
        idx = add_columns(main, s, [{"long_name": f"{s.title} (wavelet smoothed)",
                                     "values": back(y)}], "Wavelet Smooth")
        plot_into(main, gwin, s.book, s.ws, s.xcol, idx[0])
        return idx[0]
    raise ValueError(f"Unknown wavelet operation {op}")


def wavelet_dialog(main, op: str) -> None:
    if not wv.available():
        main._warn("Wavelet transforms need the optional PyWavelets package.")
        return
    fields = {
        "cwt": [("cwt_wavelet", "Wavelet:", "choice", "morl", wv.CWT_WAVELETS),
                ("n_scales", "Number of scales:", "int", 64, (4, 1024))],
        "dwt": [("wavelet", "Wavelet:", "choice", "db4", wv.DWT_WAVELETS),
                ("mode", "Extension:", "choice", "symmetric", wv.MODES)],
        "decompose": [("wavelet", "Wavelet:", "choice", "db4", wv.DWT_WAVELETS),
                      ("level", "Level (0 = maximum):", "int", 3, (0, 30)),
                      ("mode", "Extension:", "choice", "symmetric", wv.MODES)],
        "denoise": [("wavelet", "Wavelet:", "choice", "db4", wv.DWT_WAVELETS),
                    ("level", "Level (0 = maximum):", "int", 0, (0, 30)),
                    ("rule", "Threshold rule:", "choice", "universal",
                     {"universal": "Universal (σ from finest level)",
                      "level": "Level-dependent universal"}),
                    ("threshold_mode", "Thresholding:", "choice", "soft",
                     {"soft": "Soft", "hard": "Hard"}),
                    ("threshold", "Fixed threshold (0 = rule):", "float", 0.0, (0, 1e12, 6))],
        "smooth": [("wavelet", "Wavelet:", "choice", "db4", wv.DWT_WAVELETS),
                   ("drop", "Detail levels removed:", "int", 1, (1, 30)),
                   ("level", "Level (0 = maximum):", "int", 0, (0, 30))],
    }[op]
    got = ask(main, {"cwt": "Continuous Wavelet Transform", "dwt": "Decompose (DWT)",
                     "decompose": "Multi-Scale DWT", "denoise": "Wavelet Denoise",
                     "smooth": "Wavelet Smooth"}[op], fields)
    if got:
        _guard(main, lambda: run_wavelet(main, op, got[0], got[1]), "Wavelet")

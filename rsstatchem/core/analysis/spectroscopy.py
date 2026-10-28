"""Spectroscopy helpers (Origin's Spectroscopy apps and Simple Spectroscopy tools):
unit conversion between wavelength, wavenumber, energy and frequency, absorbance ↔
transmittance, Kubelka-Munk remission function and the Tauc plot band-gap estimate.

Constants are CODATA 2018 exact values (h, c, e).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

H = 6.62607015e-34          # J s
C = 299792458.0             # m/s
E = 1.602176634e-19         # C
HC_EV_NM = H * C / E * 1e9  # 1239.84198… eV·nm

UNITS = {"nm": "Wavelength (nm)", "um": "Wavelength (µm)", "cm-1": "Wavenumber (cm⁻¹)",
         "eV": "Energy (eV)", "THz": "Frequency (THz)"}
TAUC = {"direct": ("Direct allowed (n = 1/2, (αhν)²)", 2.0),
        "indirect": ("Indirect allowed (n = 2, (αhν)^½)", 0.5),
        "direct_forbidden": ("Direct forbidden (n = 3/2, (αhν)^⅔)", 2 / 3),
        "indirect_forbidden": ("Indirect forbidden (n = 3, (αhν)^⅓)", 1 / 3)}


def _to_nm(v, unit):
    v = np.asarray(v, float)
    with np.errstate(divide="ignore"):
        return {"nm": v, "um": v * 1e3, "cm-1": 1e7 / v, "eV": HC_EV_NM / v,
                "THz": C / (v * 1e12) * 1e9}[unit]


def _from_nm(nm, unit):
    with np.errstate(divide="ignore"):
        return {"nm": nm, "um": nm / 1e3, "cm-1": 1e7 / nm, "eV": HC_EV_NM / nm,
                "THz": C / (nm * 1e-9) / 1e12}[unit]


def convert(values, from_unit: str, to_unit: str) -> np.ndarray:
    """Convert spectral positions, e.g. ``convert(500, "nm", "cm-1") == 20000``."""
    if from_unit not in UNITS or to_unit not in UNITS:
        raise ValueError(f"Units must be one of {', '.join(UNITS)}.")
    return _from_nm(_to_nm(values, from_unit), to_unit)


def absorbance_to_transmittance(a, percent: bool = True) -> np.ndarray:
    t = 10.0 ** (-np.asarray(a, float))
    return t * 100 if percent else t


def transmittance_to_absorbance(t, percent: bool = True) -> np.ndarray:
    t = np.asarray(t, float) / (100.0 if percent else 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.log10(t)


def kubelka_munk(R, percent: bool | None = None) -> np.ndarray:
    """F(R) = (1 − R)²/(2R) of diffuse reflectance R (fraction; percent when any R > 1.5
    unless ``percent`` is given)."""
    R = np.asarray(R, float)
    if percent is None:
        percent = bool(np.nanmax(R) > 1.5)
    r = R / 100.0 if percent else R
    with np.errstate(divide="ignore", invalid="ignore"):
        return (1 - r) ** 2 / (2 * r)


@dataclass
class TaucResult:
    energy: np.ndarray          # hν (eV)
    tauc: np.ndarray            # (α·hν)^(1/n)
    band_gap: float
    slope: float
    intercept: float
    r2: float
    fit_range: tuple[float, float]
    transition: str


def tauc(x, absorbance, x_unit: str = "nm", transition: str = "direct",
         fit_range: tuple[float, float] | None = None, window: int = 7) -> TaucResult:
    """Tauc plot (α·hν)^(1/n) against hν and the band gap from the linear part: the x
    intercept of a straight line fitted over ``fit_range`` (eV), or, automatically, over the
    ``window``-point segment of steepest rise that fits a line well (R² ≥ 0.99)."""
    if transition not in TAUC:
        raise ValueError(f"Transition must be one of {', '.join(TAUC)}.")
    hv = np.asarray(convert(x, x_unit, "eV"), float)
    a = np.asarray(absorbance, float)
    ok = np.isfinite(hv) & np.isfinite(a) & (hv > 0)
    hv, a = hv[ok], a[ok]
    o = np.argsort(hv)
    hv, a = hv[o], a[o]
    power = TAUC[transition][1]
    t = np.power(np.clip(a * hv, 0, None), power)
    if fit_range is not None:
        lo, hi = sorted(fit_range)
        sel = (hv >= lo) & (hv <= hi)
        if sel.sum() < 2:
            raise ValueError("The fit range contains fewer than 2 points.")
    else:
        w = max(3, int(window))
        best = None
        for i in range(0, len(hv) - w + 1):
            xs, ys = hv[i:i + w], t[i:i + w]
            c = np.polyfit(xs, ys, 1)
            pred = np.polyval(c, xs)
            ss = float(np.sum((ys - ys.mean()) ** 2))
            r2 = 1 - float(np.sum((ys - pred) ** 2)) / ss if ss > 0 else 0.0
            if c[0] > 0 and r2 >= 0.99 and (best is None or c[0] > best[0]):
                best = (c[0], i)
        if best is None:
            raise ValueError("No linear rising segment found; give the fit range.")
        i = best[1]
        sel = np.zeros(len(hv), bool)
        sel[i:i + w] = True
    slope, inter = np.polyfit(hv[sel], t[sel], 1)
    pred = slope * hv[sel] + inter
    ss = float(np.sum((t[sel] - t[sel].mean()) ** 2))
    r2 = 1 - float(np.sum((t[sel] - pred) ** 2)) / ss if ss > 0 else float("nan")
    eg = -inter / slope if slope else float("nan")
    return TaucResult(hv, t, float(eg), float(slope), float(inter), float(r2),
                      (float(hv[sel].min()), float(hv[sel].max())), transition)


def photon_energy(nm) -> np.ndarray:
    return HC_EV_NM / np.asarray(nm, float)


def wavenumber_shift(raman_nm, laser_nm: float) -> np.ndarray:
    """Raman shift (cm⁻¹) of scattered wavelengths for a laser line."""
    return 1e7 / laser_nm - 1e7 / np.asarray(raman_nm, float)


def isfinite(v) -> bool:
    return v is not None and math.isfinite(v)

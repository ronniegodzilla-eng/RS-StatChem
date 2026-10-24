"""Solution and laboratory calculators (no RDKit).

The calculators chemistry theses need daily, as found in lab handbooks and in the "Solution
calculator" tools of ChemDraw / Sigma-Aldrich: molarity, molality, normality, mass needed to
prepare a solution (with purity and hydrate correction through the formula's molar mass),
the molarity of concentrated stock reagents from % (w/w) and density, dilution C₁V₁ = C₂V₂,
serial dilutions, concentration unit conversions (M, mM, µM, g/L, mg/L, ppm, ppb, % w/v,
% w/w, % v/v, mol/kg), pH of strong and weak acids/bases (exact quadratic, optionally with
water autoionisation), Henderson–Hasselbalch buffers, the ideal-gas law and general unit
conversions. All functions use SI-friendly inputs (litres, grams, mol/L) and return floats
or small dataclasses so the UI can show the intermediate values.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["molarity", "moles_from_mass", "mass_for_solution", "volume_for_moles",
           "molality", "normality", "stock_molarity", "stock_volume", "dilution",
           "serial_dilution", "SerialStep", "convert_concentration", "CONC_UNITS",
           "PHResult", "ph_strong_acid", "ph_strong_base", "ph_weak_acid", "ph_weak_base",
           "BufferResult", "buffer_ph", "buffer_recipe", "ideal_gas", "convert", "UNITS",
           "KW_25C", "R_GAS_SI"]

KW_25C = 1.0e-14
R_GAS_SI = 8.314462618  # J/(mol·K)


def _pos(name: str, x: float) -> float:
    if x is None or not (x > 0) or math.isinf(x):
        raise ValueError(f"{name} must be a positive number.")
    return float(x)


# --- concentrations ----------------------------------------------------------------------

def moles_from_mass(mass_g: float, molar_mass: float) -> float:
    return _pos("Mass", mass_g) / _pos("Molar mass", molar_mass)


def molarity(volume_L: float, mass_g: float | None = None, molar_mass: float | None = None,
             moles: float | None = None) -> float:
    """c = n / V (mol/L) from moles or from mass and molar mass."""
    if moles is None:
        moles = moles_from_mass(mass_g, molar_mass)
    return _pos("Amount", moles) / _pos("Volume", volume_L)


def mass_for_solution(molarity_M: float, volume_L: float, molar_mass: float,
                      purity_pct: float = 100.0) -> float:
    """Grams of reagent to weigh for ``volume_L`` of ``molarity_M`` (purity in %)."""
    return _pos("Molarity", molarity_M) * _pos("Volume", volume_L) * \
        _pos("Molar mass", molar_mass) / (_pos("Purity", purity_pct) / 100.0)


def volume_for_moles(moles: float, molarity_M: float) -> float:
    return _pos("Amount", moles) / _pos("Molarity", molarity_M)


def molality(moles_solute: float, solvent_kg: float) -> float:
    """b = n(solute) / m(solvent) in mol/kg."""
    return _pos("Amount", moles_solute) / _pos("Solvent mass", solvent_kg)


def normality(molarity_M: float, equivalents: float) -> float:
    """N = M × number of equivalents (H⁺/OH⁻ exchanged, or electrons in redox)."""
    return _pos("Molarity", molarity_M) * _pos("Equivalents", equivalents)


def stock_molarity(percent_ww: float, density_g_mL: float, molar_mass: float) -> float:
    """Molarity of a concentrated reagent: c = 10 · % · ρ / M (e.g. 37 % HCl → 12.1 M)."""
    return 10.0 * _pos("Percentage", percent_ww) * _pos("Density", density_g_mL) / \
        _pos("Molar mass", molar_mass)


def stock_volume(target_M: float, target_volume_L: float, percent_ww: float,
                 density_g_mL: float, molar_mass: float) -> float:
    """Litres of concentrated stock needed for ``target_volume_L`` of ``target_M``."""
    return _pos("Target molarity", target_M) * _pos("Target volume", target_volume_L) / \
        stock_molarity(percent_ww, density_g_mL, molar_mass)


def dilution(c1: float | None = None, v1: float | None = None, c2: float | None = None,
             v2: float | None = None) -> dict:
    """Solve C₁V₁ = C₂V₂ for the one value left as None; returns all four."""
    vals = {"c1": c1, "v1": v1, "c2": c2, "v2": v2}
    missing = [k for k, v in vals.items() if v is None]
    if len(missing) != 1:
        raise ValueError("Leave exactly one of C1, V1, C2, V2 empty.")
    for k, v in vals.items():
        if v is not None:
            _pos(k.upper(), v)
    k = missing[0]
    if k == "c1":
        vals[k] = c2 * v2 / v1
    elif k == "v1":
        vals[k] = c2 * v2 / c1
    elif k == "c2":
        vals[k] = c1 * v1 / v2
    else:
        vals[k] = c1 * v1 / c2
    if vals["c2"] > vals["c1"] * (1 + 1e-12):
        raise ValueError("C2 is larger than C1: a dilution cannot increase the concentration.")
    vals["solvent"] = vals["v2"] - vals["v1"]
    vals["factor"] = vals["c1"] / vals["c2"]
    return vals


@dataclass
class SerialStep:
    step: int
    concentration: float
    transfer_volume: float      # volume of the previous solution transferred
    diluent_volume: float
    cumulative_factor: float


def serial_dilution(c0: float, factor: float, steps: int, final_volume: float
                    ) -> list[SerialStep]:
    """Serial dilution: each tube gets V/factor of the previous one plus diluent to V."""
    _pos("Stock concentration", c0)
    if not factor > 1:
        raise ValueError("The dilution factor must be greater than 1.")
    if not (1 <= steps <= 50):
        raise ValueError("Use 1 to 50 steps.")
    _pos("Final volume", final_volume)
    out = []
    for i in range(1, steps + 1):
        transfer = final_volume / factor
        out.append(SerialStep(i, c0 / factor ** i, transfer, final_volume - transfer,
                              factor ** i))
    return out


# Each unit: (kind, factor to the base of its kind). Bases: mol/L, g/L, mass fraction,
# volume fraction, mol/kg.
CONC_UNITS = {
    "M (mol/L)": ("molar", 1.0), "mM": ("molar", 1e-3), "µM": ("molar", 1e-6),
    "nM": ("molar", 1e-9),
    "g/L": ("massvol", 1.0), "mg/L": ("massvol", 1e-3), "µg/L": ("massvol", 1e-6),
    "mg/mL": ("massvol", 1.0), "µg/mL": ("massvol", 1e-3), "ng/mL": ("massvol", 1e-6),
    "% (w/v)": ("massvol", 10.0),
    "ppm (w/w)": ("massfrac", 1e-6), "ppb (w/w)": ("massfrac", 1e-9),
    "% (w/w)": ("massfrac", 1e-2),
    "% (v/v)": ("volfrac", 1e-2), "ppm (v/v)": ("volfrac", 1e-6),
    "mol/kg": ("molal", 1.0), "mmol/kg": ("molal", 1e-3),
}


def convert_concentration(value: float, from_unit: str, to_unit: str,
                          molar_mass: float | None = None, density: float = 1.0,
                          solute_density: float | None = None) -> float:
    """Convert between concentration units.

    ``density`` is the solution density (g/mL, 1.0 for dilute aqueous solutions, where
    ppm (w/w) ≈ mg/L); ``molar_mass`` is needed between molar and mass units;
    ``solute_density`` (g/mL) is needed for % (v/v)."""
    if from_unit not in CONC_UNITS or to_unit not in CONC_UNITS:
        raise ValueError("Unknown concentration unit.")
    kind, f = CONC_UNITS[from_unit]
    x = float(value) * f
    rho = _pos("Solution density", density) * 1000.0  # g/L of solution
    # → grams of solute per litre of solution
    if kind == "molar":
        g_per_L = x * _need(molar_mass, "molar mass")
    elif kind == "massvol":
        g_per_L = x
    elif kind == "massfrac":
        g_per_L = x * rho
    elif kind == "volfrac":
        g_per_L = x * 1000.0 * _need(solute_density, "solute density")
    else:  # molal: n per kg solvent → g/L of solution
        mm = _need(molar_mass, "molar mass")
        w = x * mm / 1000.0          # g solute per g solvent
        g_per_L = rho * w / (1 + w)
    to_kind, to_f = CONC_UNITS[to_unit]
    if to_kind == "molar":
        y = g_per_L / _need(molar_mass, "molar mass")
    elif to_kind == "massvol":
        y = g_per_L
    elif to_kind == "massfrac":
        y = g_per_L / rho
    elif to_kind == "volfrac":
        y = g_per_L / (1000.0 * _need(solute_density, "solute density"))
    else:
        mm = _need(molar_mass, "molar mass")
        solvent_g = rho - g_per_L
        if solvent_g <= 0:
            raise ValueError("The solute mass exceeds the solution mass; check the density.")
        y = (g_per_L / mm) / (solvent_g / 1000.0)
    return y / to_f


def _need(x: float | None, what: str) -> float:
    if x is None or not x > 0:
        raise ValueError(f"This conversion needs the {what}.")
    return float(x)


# --- pH ------------------------------------------------------------------------------------

@dataclass
class PHResult:
    pH: float
    pOH: float
    h: float                     # [H3O+] mol/L
    oh: float                    # [OH-] mol/L
    method: str
    approx_pH: float | None = None
    ionization_pct: float | None = None
    notes: list[str] = field(default_factory=list)


def _result(h: float, kw: float, method: str, **kw_args) -> PHResult:
    pkw = -math.log10(kw)
    ph = -math.log10(h)
    return PHResult(ph, pkw - ph, h, kw / h, method, **kw_args)


def ph_strong_acid(c: float, protons: int = 1, kw: float = KW_25C,
                   water: bool = True) -> PHResult:
    """Strong acid of molarity ``c`` releasing ``protons`` H⁺ each.

    With ``water`` the exact charge balance [H⁺] = C + Kw/[H⁺] is solved, which matters
    below ~1e-6 M (pH never exceeds 7)."""
    ca = _pos("Concentration", c) * protons
    h = (ca + math.sqrt(ca * ca + 4 * kw)) / 2 if water else ca
    res = _result(h, kw, "Strong acid: [H⁺] = (C + √(C² + 4Kw)) / 2" if water else
                  "Strong acid: [H⁺] = C", approx_pH=-math.log10(ca))
    return res


def ph_strong_base(c: float, hydroxides: int = 1, kw: float = KW_25C,
                   water: bool = True) -> PHResult:
    cb = _pos("Concentration", c) * hydroxides
    oh = (cb + math.sqrt(cb * cb + 4 * kw)) / 2 if water else cb
    pkw = -math.log10(kw)
    res = _result(kw / oh, kw, "Strong base: [OH⁻] = (C + √(C² + 4Kw)) / 2" if water else
                  "Strong base: [OH⁻] = C", approx_pH=pkw + math.log10(cb))
    return res


def _cubic_weak(c: float, k: float, kw: float) -> float:
    """Exact [H⁺] of a monoprotic weak acid with water: h³ + Kh² − (KC + Kw)h − KKw = 0."""
    lo, hi = 0.0, c + math.sqrt(kw) + 1.0
    f = lambda h: h ** 3 + k * h * h - (k * c + kw) * h - k * kw  # noqa: E731
    for _ in range(300):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def ph_weak_acid(c: float, ka: float, kw: float = KW_25C, water: bool = False) -> PHResult:
    """Monoprotic weak acid HA: exact quadratic x = (−Ka + √(Ka² + 4KaC)) / 2.

    ``water=True`` solves the full cubic with water autoionisation (very dilute or very weak
    acids). The approximation √(KaC) and the % ionisation (5 % rule) are reported too."""
    c, ka = _pos("Concentration", c), _pos("Ka", ka)
    if water:
        h = _cubic_weak(c, ka, kw)
        x = h - kw / h
        method = "Weak acid, exact (charge balance incl. water)"
    else:
        x = (-ka + math.sqrt(ka * ka + 4 * ka * c)) / 2
        h = x
        method = "Weak acid: x = (−Ka + √(Ka² + 4·Ka·C)) / 2"
    approx = -math.log10(math.sqrt(ka * c))
    pct = 100.0 * x / c
    res = _result(h, kw, method, approx_pH=approx, ionization_pct=pct)
    res.notes.append(f"Approximation [H⁺] ≈ √(Ka·C) gives pH {approx:.3f}; ionisation "
                     f"{pct:.2f} % ({'≤' if pct <= 5 else '>'} 5 %, approximation "
                     f"{'acceptable' if pct <= 5 else 'not valid'}).")
    return res


def ph_weak_base(c: float, kb: float, kw: float = KW_25C, water: bool = False) -> PHResult:
    c, kb = _pos("Concentration", c), _pos("Kb", kb)
    if water:
        oh = _cubic_weak(c, kb, kw)
        x = oh - kw / oh
        method = "Weak base, exact (charge balance incl. water)"
    else:
        x = (-kb + math.sqrt(kb * kb + 4 * kb * c)) / 2
        oh = x
        method = "Weak base: x = (−Kb + √(Kb² + 4·Kb·C)) / 2"
    pkw = -math.log10(kw)
    approx = pkw + math.log10(math.sqrt(kb * c))
    pct = 100.0 * x / c
    res = _result(kw / oh, kw, method, approx_pH=approx, ionization_pct=pct)
    res.notes.append(f"Approximation [OH⁻] ≈ √(Kb·C) gives pH {approx:.3f}; protonation "
                     f"{pct:.2f} % ({'≤' if pct <= 5 else '>'} 5 %).")
    return res


@dataclass
class BufferResult:
    pH: float
    pKa: float
    acid: float      # mol or mol/L of HA after any addition
    base: float      # mol or mol/L of A⁻
    ratio: float     # base / acid
    notes: list[str] = field(default_factory=list)


def buffer_ph(pka: float, acid: float, base: float, added_strong_acid: float = 0.0,
              added_strong_base: float = 0.0) -> BufferResult:
    """Henderson–Hasselbalch pH = pKa + log([A⁻]/[HA]).

    ``acid`` and ``base`` are moles (or concentrations in the same volume); added strong
    acid converts A⁻ to HA and added strong base converts HA to A⁻ (same units)."""
    ha = _pos("Acid amount", acid) + added_strong_acid - added_strong_base
    a = _pos("Base amount", base) - added_strong_acid + added_strong_base
    if ha <= 0 or a <= 0:
        raise ValueError("The added strong acid/base exceeds the buffer capacity.")
    ratio = a / ha
    res = BufferResult(pka + math.log10(ratio), pka, ha, a, ratio)
    if not (0.1 <= ratio <= 10):
        res.notes.append("The base/acid ratio is outside 0.1–10 (pH outside pKa ± 1): the "
                         "buffer capacity is poor and Henderson–Hasselbalch is less accurate.")
    return res


def buffer_recipe(pka: float, target_ph: float, total_M: float, volume_L: float,
                  acid_molar_mass: float | None = None, base_molar_mass: float | None = None
                  ) -> dict:
    """Moles (and grams) of weak acid and conjugate base for a buffer at ``target_ph``."""
    ratio = 10 ** (target_ph - pka)
    total = _pos("Total concentration", total_M) * _pos("Volume", volume_L)
    n_acid = total / (1 + ratio)
    n_base = total - n_acid
    out = {"ratio": ratio, "acid_mol": n_acid, "base_mol": n_base,
           "acid_M": n_acid / volume_L, "base_M": n_base / volume_L}
    if acid_molar_mass:
        out["acid_g"] = n_acid * acid_molar_mass
    if base_molar_mass:
        out["base_g"] = n_base * base_molar_mass
    return out


# --- gases and units -----------------------------------------------------------------------

def ideal_gas(P: float | None = None, V: float | None = None, n: float | None = None,
              T: float | None = None) -> dict:
    """Solve PV = nRT with P in atm, V in L, n in mol, T in K (one value None)."""
    r = 0.082057366
    vals = {"P": P, "V": V, "n": n, "T": T}
    missing = [k for k, v in vals.items() if v is None]
    if len(missing) != 1:
        raise ValueError("Leave exactly one of P, V, n, T empty.")
    for k, v in vals.items():
        if v is not None:
            _pos(k, v)
    k = missing[0]
    if k == "P":
        vals[k] = n * r * T / V
    elif k == "V":
        vals[k] = n * r * T / P
    elif k == "n":
        vals[k] = P * V / (r * T)
    else:
        vals[k] = P * V / (n * r)
    return vals


UNITS = {
    "Mass": {"kg": 1.0, "g": 1e-3, "mg": 1e-6, "µg": 1e-9, "ng": 1e-12, "lb": 0.45359237,
             "oz": 0.028349523125, "Da (u)": 1.66053906660e-27},
    "Volume": {"m³": 1.0, "L": 1e-3, "dm³": 1e-3, "mL": 1e-6, "cm³": 1e-6, "µL": 1e-9,
               "nL": 1e-12, "gal (US)": 3.785411784e-3},
    "Amount": {"mol": 1.0, "mmol": 1e-3, "µmol": 1e-6, "nmol": 1e-9, "kmol": 1e3,
               "molecules": 1 / 6.02214076e23},
    "Molar concentration": {"mol/L": 1.0, "mmol/L": 1e-3, "µmol/L": 1e-6, "nmol/L": 1e-9,
                            "mol/m³": 1e-3, "mol/mL": 1e3},
    "Pressure": {"Pa": 1.0, "kPa": 1e3, "MPa": 1e6, "bar": 1e5, "mbar": 100.0,
                 "atm": 101325.0, "mmHg": 133.322387415, "Torr": 101325.0 / 760, "psi":
                 6894.757293168},
    "Energy": {"J": 1.0, "kJ": 1e3, "cal": 4.184, "kcal": 4184.0, "eV": 1.602176634e-19,
               "kJ/mol": 1e3 / 6.02214076e23, "kcal/mol": 4184.0 / 6.02214076e23,
               "cm⁻¹": 1.98644586e-23, "Wh": 3600.0},
    "Length": {"m": 1.0, "cm": 1e-2, "mm": 1e-3, "µm": 1e-6, "nm": 1e-9, "pm": 1e-12,
               "Å": 1e-10, "in": 0.0254},
    "Temperature": {"K": None, "°C": None, "°F": None},
}


def convert(value: float, from_unit: str, to_unit: str, category: str | None = None) -> float:
    """Convert ``value`` between units of the same category in :data:`UNITS`."""
    cats = [category] if category else [c for c, u in UNITS.items()
                                        if from_unit in u and to_unit in u]
    if not cats or from_unit not in UNITS[cats[0]] or to_unit not in UNITS[cats[0]]:
        raise ValueError(f"Cannot convert {from_unit} to {to_unit}.")
    cat = cats[0]
    if cat == "Temperature":
        k = {"K": value, "°C": value + 273.15, "°F": (value - 32) * 5 / 9 + 273.15}[from_unit]
        if k < 0:
            raise ValueError("Temperature below absolute zero.")
        return {"K": k, "°C": k - 273.15, "°F": (k - 273.15) * 9 / 5 + 32}[to_unit]
    return value * UNITS[cat][from_unit] / UNITS[cat][to_unit]

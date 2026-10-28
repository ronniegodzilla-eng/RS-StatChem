"""Henry's law constant of a compound in water at 25 °C (ChemDraw's Chemical Properties row
"Henry's Law", 17.1 guide p. 198), as HLC = p / c in Pa m³/mol (air-water partitioning;
larger = more volatile from water).

Sources, in this order:

1. experimental: the PHYSPROP values curated for OPERA (``HL_QR.sdf``, K. Mansouri et al.,
   J. Cheminform. 10 (2018) 10; MIT-licensed OPERA repository, as for the other PHYSPROP data);
2. estimates, the one with the lower benchmark error is shown and both are listed:
   - ``ml``: gradient-boosted trees on the same data (``scripts/train_pka_model.py``), with
     its held-out error and applicability domain;
   - ``vpws``: the thermodynamic ratio HLC = P_vap / S_w of the vapour pressure and the water
     solubility the window already shows (for a gas at 25 °C, P_vap > 1 atm, the solubility
     refers to 1 atm partial pressure, so 1 atm / S_w is used) (experimental where known, else the Nannoolal
     vapour pressure and the ESOL solubility). Valid for sparingly soluble compounds; it fails
     for miscible ones (S_w meaningless) - the benchmark on the PHYSPROP set is
     ``resources/properties/henry_vpws_benchmark.json``.
No Qt here.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

from rsstatchem.core.chem_draw.pka import ModelFile

__all__ = ["henry_models", "HenryEstimate", "experimental", "ml", "vp_ws_estimate",
           "vpws_benchmark", "ATM_M3", "DATA_SOURCE"]

ATM_M3 = 101325.0             # 1 atm m3/mol in Pa m3/mol
DATA_SOURCE = "PHYSPROP via US EPA OPERA (Mansouri et al. 2018)"
henry_models = ModelFile("henry_model.npz", ("loghl",))


@dataclass
class HenryEstimate:
    value: float | None               # Pa m3/mol
    method: str = ""                  # "experimental" | "ml" | "vpws"
    note: str = ""
    experimental_inputs: bool = False  # vpws: both P_vap and S_w experimental
    prediction: object = None


def experimental(mol) -> HenryEstimate | None:
    from rdkit import Chem
    try:
        key = Chem.MolToInchiKey(mol)
    except Exception:  # noqa: BLE001
        return None
    v, note = henry_models.experimental("loghl", key)
    if v is None:
        return None
    return HenryEstimate(10 ** v * ATM_M3, "experimental", note or DATA_SOURCE)


def ml(mol) -> HenryEstimate:
    p = henry_models.predict(mol, "loghl")
    if p.value is None:
        return HenryEstimate(None, "ml", p.error, prediction=p)
    return HenryEstimate(10 ** p.value * ATM_M3, "ml", "", prediction=p)


def vp_ws_estimate(mol, vp_pa: float | None = None, logs: float | None = None,
                   exp_inputs: bool | None = None) -> HenryEstimate:
    """HLC = P_vap / S_w (Pa m3/mol). Without arguments the vapour pressure and solubility come
    from the Chemical Properties rows (experimental or the default estimate)."""
    if vp_pa is None or logs is None:
        from rsstatchem.core.chem_draw import properties
        rows = {p.key: p for p in properties.chemical_properties(mol)}
        vp, ls = rows.get("logvp"), rows.get("logs")
        if vp is None or ls is None or vp.value is None or ls.value is None:
            return HenryEstimate(None, "vpws", "needs a vapour pressure and a water solubility")
        vp_pa, logs = vp.value, ls.value
        exp_inputs = vp.kind == "experimental" and ls.kind == "experimental"
    if vp_pa is None or logs is None or vp_pa <= 0:
        return HenryEstimate(None, "vpws", "needs a vapour pressure and a water solubility")
    s = 10 ** logs * 1000.0                      # mol/L -> mol/m3
    gas = vp_pa > ATM_M3
    # a gas at 25 °C: its water solubility refers to 1 atm partial pressure, not to P_vap
    value = (ATM_M3 if gas else vp_pa) / s
    if not math.isfinite(value) or value <= 0:
        return HenryEstimate(None, "vpws", "out of range")
    note = "P_vap / S_w, " + ("both experimental" if exp_inputs else "with estimated inputs")
    if gas:
        note += "; gas at 25 °C: 1 atm / S_w"
    if logs > 0:
        note += "; very soluble compound: the ratio is unreliable"
    return HenryEstimate(value, "vpws", note, bool(exp_inputs))


_VPWS: dict | None = None


def vpws_benchmark() -> dict:
    """{'all' | 'both_experimental' | 'estimated': {n, rmse, mae, bias}} (log10 units)."""
    global _VPWS
    if _VPWS is None:
        from rsstatchem.paths import resource
        try:
            _VPWS = json.loads(resource("properties/henry_vpws_benchmark.json")
                               .read_text("utf-8"))
        except (OSError, ValueError):
            _VPWS = {}
    return _VPWS

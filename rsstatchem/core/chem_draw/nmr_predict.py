"""¹H and ¹³C NMR shift prediction from spherical atom environments (HOSE-style codes), with
first-order ¹H multiplets - the open counterpart of ChemDraw's licensed ChemNMR
(17.1 guide pp. 217-224).

Method
------
Each atom is described by codes of growing radius in the spirit of Bremser's HOSE codes
(W. Bremser, Anal. Chim. Acta 103 (1978) 355): radius r lists, as a canonical rooted tree,
every heavy atom up to r bonds away with its element, aromaticity, charge and the bond type
from its parent; ring closures inside the sphere are marked. The centre also carries its
number of hydrogens. (The code string is our own canonical form, not Bremser's exact
notation; nothing depends on compatibility with other programs.) The shift table maps the code
of every radius 0..:data:`MAX_RADIUS` to the mean, standard deviation and count of the
observed shifts with that environment. A prediction uses the largest radius found in the
table (more context = more reliable); ``radius`` and ``n_ref`` tell how much data stand behind
it. ¹H shifts are predicted for the heavy atom that carries the hydrogens (its H average).

Data
----
``resources/nmr/hose_shifts.npz`` is built by ``scripts/build_nmr_db.py`` from the assigned
spectra of the BMRB metabolomics archive (CC0, wwPDB core archive). Most of those spectra were
measured in D2O near pH 7.4 (acids as carboxylates, amines protonated), so shifts of groups
next to ionisable sites can differ from CDCl3 values by a few ppm; the benchmark in
``resources/nmr/benchmark.json`` is a held-out compound split. nmrshiftdb2 is not used (its
licence requires OSI-licensed prediction software). No increment tables from Pretsch or other
books are used.

¹H multiplets (first order, n+1 rule) use generic textbook coupling constants: vicinal
H-C-C-H 7 Hz (sp3), aromatic ortho 7.8 Hz, alkene trans 16 Hz / cis 10.5 Hz (from the
drawn E/Z geometry; 13 Hz when undefined), terminal =CH2 to its CH 10/17 Hz, aldehyde CHO-CH
2.5 Hz. Topologically equivalent hydrogens do not split each other; O-H, N-H and S-H are
exchangeable (broad singlets). Long-range and second-order effects are ignored.
No Qt here.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

__all__ = ["MAX_RADIUS", "Shift", "Peak", "predict", "predict_detailed", "atom_codes",
           "build_table", "save_table", "load_table", "table_info", "benchmark",
           "multiplet_lines", "spectrum", "multiplicity_name", "SOLVENT_PEAKS", "normalise"]

MAX_RADIUS = 5
BOND_SYMBOL = {1.0: "-", 2.0: "=", 3.0: "#", 1.5: ":"}
# residual solvent signals (ppm), commonly tabulated values (Fulmer et al., Organometallics
# 29 (2010) 2176) - drawn as markers only
SOLVENT_PEAKS = {"CDCl3": {"1H": 7.26, "13C": 77.16}, "DMSO-d6": {"1H": 2.50, "13C": 39.52},
                 "D2O": {"1H": 4.79, "13C": None}, "CD3OD": {"1H": 3.31, "13C": 49.00},
                 "acetone-d6": {"1H": 2.05, "13C": 29.84}}


class Shift(NamedTuple):
    """One atom's prediction (the WP-A data contract)."""
    atom_idx: int            # heavy atom index (for 1H: the atom carrying the hydrogens)
    shift: float             # ppm
    multiplicity: str        # "s", "d", "t", "q", "dd", ... ("s" for decoupled 13C)
    J: tuple                 # coupling constants in Hz (first order)


@dataclass
class Peak:
    """A group of equivalent atoms (one line / multiplet in the spectrum)."""
    nucleus: str
    atoms: list              # heavy atom indices (equivalent)
    shift: float | None      # ppm
    count: int               # number of nuclei (H atoms for 1H, C atoms for 13C)
    multiplicity: str = "s"
    couplings: list = field(default_factory=list)   # [(J Hz, number of coupled H)]
    radius: int = -1         # code radius used (-1 = no data)
    n_ref: int = 0           # observations behind the value
    std: float = 0.0         # spread of those observations (ppm)
    exchangeable: bool = False
    note: str = ""

    @property
    def J(self) -> tuple:
        return tuple(j for j, _n in self.couplings)


def normalise(nucleus: str) -> str:
    n = nucleus.upper().replace("¹", "1").replace("³", "3")
    if n in ("H", "1H", "PROTON"):
        return "1H"
    if n in ("C", "13C", "CARBON"):
        return "13C"
    raise ValueError(f"unsupported nucleus {nucleus!r} (1H or 13C)")


# --- environment codes -----------------------------------------------------------------------

def _label(atom) -> str:
    s = atom.GetSymbol()
    if atom.GetIsAromatic():
        s = s.lower()
    q = atom.GetFormalCharge()
    if q:
        s += "+" * q if q > 0 else "-" * (-q)
    return s


def _bond(b) -> str:
    if b.GetIsAromatic():
        return ":"
    return BOND_SYMBOL.get(b.GetBondTypeAsDouble(), "~")


def atom_codes(mol, center: int, max_radius: int = MAX_RADIUS) -> list[str]:
    """Codes of radius 0..max_radius of one heavy atom (hydrogens ignored except the centre's
    count). Deterministic and independent of atom order."""
    dist = {center: 0}
    frontier = [center]
    d = 0
    while frontier and d < max_radius:
        d += 1
        nxt = []
        for i in frontier:
            for nb in mol.GetAtomWithIdx(i).GetNeighbors():
                j = nb.GetIdx()
                if nb.GetAtomicNum() == 1 or j in dist:
                    continue
                dist[j] = d
                nxt.append(j)
        frontier = nxt

    def closures(i: int, r: int) -> int:
        """bonds to atoms of the same sphere (ring closures), within radius r."""
        if dist[i] >= r:
            return 0
        return sum(1 for nb in mol.GetAtomWithIdx(i).GetNeighbors()
                   if nb.GetAtomicNum() > 1 and dist.get(nb.GetIdx()) == dist[i])

    def rec(i: int, depth: int, r: int) -> str:
        a = mol.GetAtomWithIdx(i)
        lab = _label(a)
        c = closures(i, r)
        if c:
            lab += "&" * c
        if depth == 0:
            return lab
        kids = []
        for b in a.GetBonds():
            nb = b.GetOtherAtom(a)
            j = nb.GetIdx()
            if nb.GetAtomicNum() > 1 and dist.get(j) == dist[i] + 1:
                kids.append(_bond(b) + rec(j, depth - 1, r))
        if not kids:
            return lab
        return lab + "(" + ",".join(sorted(kids)) + ")"

    ca = mol.GetAtomWithIdx(center)
    head = f"{_label(ca)}H{ca.GetTotalNumHs(includeNeighbors=True)}"
    head += f"D{sum(1 for n in ca.GetNeighbors() if n.GetAtomicNum() > 1)}"
    out = [head]
    for r in range(1, max_radius + 1):
        out.append(head + "|" + rec(center, r, r))
    return out


def _hash(nucleus: str, code: str) -> int:
    return int.from_bytes(hashlib.blake2b(f"{nucleus}|{code}".encode(), digest_size=8).digest(),
                          "little") >> 1      # non-negative int64


# --- table -----------------------------------------------------------------------------------

def _h_shifts(mol, h1: dict) -> dict[int, float]:
    """heavy atom index -> mean shift of its hydrogens (record H shifts are keyed by H index)."""
    acc: dict[int, list[float]] = {}
    for hi, v in h1.items():
        h = mol.GetAtomWithIdx(int(hi))
        nbs = h.GetNeighbors()
        if len(nbs) == 1:
            acc.setdefault(nbs[0].GetIdx(), []).append(float(v))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def build_table(records) -> dict:
    """{hash: [values]} from records with 'mol' (explicit H), 'c13' and 'h1' dicts."""
    table: dict[int, list[float]] = {}
    for rec in records:
        mol = rec["mol"]
        for nucleus, shifts in (("13C", {int(k): float(v) for k, v in rec["c13"].items()}),
                                ("1H", _h_shifts(mol, rec["h1"]))):
            for idx, v in shifts.items():
                for code in atom_codes(mol, idx):
                    table.setdefault(_hash(nucleus, code), []).append(v)
    return table


def _compact(table: dict) -> dict:
    keys = np.array(sorted(table), dtype=np.int64)
    vals = [table[int(k)] for k in keys]
    return {"keys": keys,
            "mean": np.array([float(np.mean(v)) for v in vals], dtype=np.float32),
            "std": np.array([float(np.std(v)) for v in vals], dtype=np.float16),
            "n": np.array([min(len(v), 65535) for v in vals], dtype=np.uint16)}


def save_table(table: dict, path, meta: dict) -> None:
    arr = _compact(table)
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arr)


class _Table:
    def __init__(self, arr: dict, meta: dict):
        self.keys, self.mean, self.std, self.n = arr["keys"], arr["mean"], arr["std"], arr["n"]
        self.meta = meta

    def get(self, h: int):
        i = int(np.searchsorted(self.keys, h))
        if i < len(self.keys) and int(self.keys[i]) == h:
            return float(self.mean[i]), float(self.std[i]), int(self.n[i])
        return None


_TABLE: _Table | None = None
_LOCK = threading.Lock()


def table_path():
    from rsstatchem.paths import resource
    return resource("nmr/hose_shifts.npz")


def load_table() -> _Table | None:
    """The shipped shift table (None when the file is missing)."""
    global _TABLE
    with _LOCK:
        if _TABLE is None:
            try:
                data = np.load(table_path(), allow_pickle=False)
                _TABLE = _Table({k: data[k] for k in ("keys", "mean", "std", "n")},
                                json.loads(str(data["meta"])))
            except (OSError, ValueError, KeyError):
                return None
        return _TABLE


def table_info() -> dict:
    t = load_table()
    return dict(t.meta) if t is not None else {}


def _lookup(table, nucleus: str, codes: list[str]):
    """(value, std, n, radius) from the largest radius present, or None."""
    for r in range(len(codes) - 1, -1, -1):
        hit = table.get(_hash(nucleus, codes[r])) if isinstance(table, _Table) else \
            _dict_get(table, _hash(nucleus, codes[r]))
        if hit is not None:
            return hit[0], hit[1], hit[2], r
    return None


def _dict_get(table: dict, h: int):
    v = table.get(h)
    if not v:
        return None
    return float(np.mean(v)), float(np.std(v)), len(v)


# --- prediction ------------------------------------------------------------------------------

def _prepare(mol):
    from rdkit import Chem
    m = Chem.Mol(mol)
    if any(a.GetAtomicNum() == 1 for a in m.GetAtoms()):
        try:
            m = Chem.RemoveHs(m)
        except Exception:  # noqa: BLE001
            pass
    return m


def _classes(mol) -> list[int]:
    from rdkit import Chem
    return list(Chem.CanonicalRankAtoms(mol, breakTies=False, includeChirality=True))


def _double_bond_hh(mol, a, b) -> float | None:
    """J between the H's across the C=C double bond a=b (each end carrying one H)."""
    from rsstatchem.core.chem_draw import stereo
    _atoms, bonds = stereo.mol_labels(mol)
    bond = mol.GetBondBetweenAtoms(a, b)
    lab = bonds.get(bond.GetIdx()) if bond is not None else None
    ha, hb = (mol.GetAtomWithIdx(i).GetTotalNumHs() for i in (a, b))
    heavy = [sum(1 for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() != j)
             for i, j in ((a, b), (b, a))]
    if ha == 1 and hb == 1 and heavy == [1, 1] and lab in ("E", "Z"):
        return 16.0 if lab == "E" else 10.5
    return None


def _coupling(mol, i: int, j: int) -> float | list | None:
    """Generic 3J between the hydrogens of heavy atoms i and j (bonded), or None."""
    ai, aj = mol.GetAtomWithIdx(i), mol.GetAtomWithIdx(j)
    if ai.GetSymbol() != "C" or aj.GetSymbol() != "C":
        return None
    bond = mol.GetBondBetweenAtoms(i, j)
    if bond.GetIsAromatic():
        return 7.8
    if bond.GetBondTypeAsDouble() == 2:
        hi, hj = ai.GetTotalNumHs(), aj.GetTotalNumHs()
        if hi == 1 and hj == 2:
            return [17.0, 10.5]          # the CH sees trans and cis H of =CH2
        if hi == 2 and hj == 1:
            return 13.5                  # each =CH2 H: average of cis / trans (approximate)
        return _double_bond_hh(mol, i, j) or 13.0
    if bond.GetBondTypeAsDouble() != 1:
        return None
    if any(b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(a).GetSymbol() == "O"
           for a in (ai, aj) for b in a.GetBonds()):
        # aldehyde H - CH vicinal coupling is small
        if (ai.GetTotalNumHs() == 1 and any(b.GetBondTypeAsDouble() == 2 and
                                            b.GetOtherAtom(ai).GetSymbol() == "O"
                                            for b in ai.GetBonds())) or \
           (aj.GetTotalNumHs() == 1 and any(b.GetBondTypeAsDouble() == 2 and
                                            b.GetOtherAtom(aj).GetSymbol() == "O"
                                            for b in aj.GetBonds())):
            return 2.5
    return 7.0


def multiplicity_name(couplings) -> str:
    """'s', 'd', 't', 'q', 'quint', ..., 'dd', 'dt', 'td', 'ddd', or 'm'."""
    names = {1: "d", 2: "t", 3: "q", 4: "quint", 5: "sext", 6: "sept", 7: "oct"}
    groups = [(j, n) for j, n in couplings if n > 0]
    if not groups:
        return "s"
    if len(groups) > 3 or sum(n for _j, n in groups) > 8:
        return "m"
    groups.sort(key=lambda g: -g[0])
    return "".join(names.get(n, "m") for _j, n in groups) if len(groups) > 1 else \
        names.get(groups[0][1], "m")


def _merge(couplings) -> list:
    """Combine couplings of equal J (within 0.6 Hz): first-order n+1 rule."""
    out: list = []
    for j, n in sorted(couplings, key=lambda c: -c[0]):
        for k, (j2, n2) in enumerate(out):
            if abs(j - j2) <= 0.6:
                out[k] = ((j2 * n2 + j * n) / (n2 + n), n2 + n)
                break
        else:
            out.append((j, n))
    return [(round(j, 1), n) for j, n in out]


def predict_detailed(mol, nucleus: str, table=None) -> list[Peak]:
    """One :class:`Peak` per group of equivalent atoms (sorted by shift, high to low)."""
    nucleus = normalise(nucleus)
    table = table if table is not None else load_table()
    m = _prepare(mol)
    ranks = _classes(m)
    groups: dict[int, list[int]] = {}
    for a in m.GetAtoms():
        if nucleus == "13C" and a.GetSymbol() != "C":
            continue
        if nucleus == "1H" and a.GetTotalNumHs() == 0:
            continue
        groups.setdefault(ranks[a.GetIdx()], []).append(a.GetIdx())
    peaks = []
    for _rank, atoms in groups.items():
        first = m.GetAtomWithIdx(atoms[0])
        hit = _lookup(table, nucleus, atom_codes(m, atoms[0])) if table is not None else None
        count = len(atoms) * (first.GetTotalNumHs() if nucleus == "1H" else 1)
        p = Peak(nucleus, atoms, None, count)
        if hit is not None:
            p.shift, p.std, p.n_ref, p.radius = round(hit[0], 2), hit[1], hit[2], hit[3]
        else:
            p.note = "no data for this environment"
        if nucleus == "1H":
            if first.GetSymbol() in ("O", "N", "S"):
                p.exchangeable = True
                p.note = (p.note + "; " if p.note else "") + \
                    "exchangeable H: broad, position depends on solvent and concentration"
            else:
                coup = []
                for nb in first.GetNeighbors():
                    j = nb.GetIdx()
                    if ranks[j] == ranks[atoms[0]] or nb.GetTotalNumHs() == 0:
                        continue
                    jj = _coupling(m, atoms[0], j)
                    if jj is None:
                        continue
                    nh = nb.GetTotalNumHs()
                    if isinstance(jj, list):
                        coup += [(x, 1) for x in jj]
                    else:
                        coup.append((jj, nh))
                p.couplings = _merge(coup)
                p.multiplicity = multiplicity_name(p.couplings)
        peaks.append(p)
    peaks.sort(key=lambda p: -(p.shift if p.shift is not None else -1e9))
    return peaks


def predict(mol, nucleus: str) -> list[Shift]:
    """[(atom_idx, shift, multiplicity, J)] for every carbon (13C) or every heavy atom carrying
    hydrogens (1H); atoms without data are left out."""
    out = []
    for p in predict_detailed(mol, nucleus):
        if p.shift is None:
            continue
        for a in p.atoms:
            out.append(Shift(a, p.shift, p.multiplicity, p.J))
    out.sort(key=lambda s: s.atom_idx)
    return out


# --- spectrum ----------------------------------------------------------------------------------

def multiplet_lines(shift: float, couplings, mhz: float, count: float = 1.0
                    ) -> list[tuple[float, float]]:
    """First-order line positions (ppm) and intensities (sum = count)."""
    lines = {0.0: 1.0}
    for j, n in couplings:
        coeffs = [math.comb(n, k) for k in range(n + 1)]
        new: dict[float, float] = {}
        for off, inten in lines.items():
            for k, c in enumerate(coeffs):
                o = round(off + (k - n / 2) * j, 6)
                new[o] = new.get(o, 0.0) + inten * c
        lines = new
    total = sum(lines.values())
    return sorted((shift + off / mhz, count * inten / total) for off, inten in lines.items())


def spectrum(peaks: list[Peak], mhz: float = 300.0, lo: float | None = None,
             hi: float | None = None, points: int = 8192, width_hz: float = 1.2
             ) -> tuple[np.ndarray, np.ndarray]:
    """(ppm axis, intensity) of the predicted spectrum: Lorentzian lines (FWHM width_hz at the
    1H frequency mhz, never narrower than 2.5 sampling steps; exchangeable H 15 Hz; 13C lines
    4 x width_hz at the 13C frequency, at least 0.08 ppm)."""
    shifts = [p.shift for p in peaks if p.shift is not None]
    nucleus = peaks[0].nucleus if peaks else "1H"
    if lo is None or hi is None:
        lo, hi = (-0.5, 12.5) if nucleus == "1H" else (-5.0, 225.0)
        if shifts:
            lo, hi = min(lo, min(shifts) - 0.5), max(hi, max(shifts) + 0.5)
    x = np.linspace(hi, lo, points)
    y = np.zeros(points)
    for p in peaks:
        if p.shift is None:
            continue
        if nucleus == "13C":
            # 13C observe frequency = 1H frequency x gamma(13C)/gamma(1H); lines drawn at least
            # 0.08 ppm wide so that a sampled spectrum shows every line at full height
            lines = [(p.shift, float(p.count))]
            w_ppm = max(width_hz * 4 / (mhz * 0.2515), 0.08)
        else:
            lines = multiplet_lines(p.shift, p.couplings, mhz, p.count)
            w_ppm = max((15.0 if p.exchangeable else width_hz) / mhz, 2.5 * (hi - lo) / points)
        g = w_ppm / 2
        for pos, inten in lines:
            y += inten * g * g / ((x - pos) ** 2 + g * g)
    return x, y


# --- benchmark (used by scripts/build_nmr_db.py) -----------------------------------------------

def benchmark(table: dict, records) -> dict:
    """Held-out errors: MAE / RMSE per nucleus and per radius used."""
    res = {}
    for nucleus in ("13C", "1H"):
        errs, by_r = [], {}
        missing = 0
        for rec in records:
            mol = rec["mol"]
            obs = ({int(k): float(v) for k, v in rec["c13"].items()} if nucleus == "13C"
                   else _h_shifts(mol, rec["h1"]))
            for idx, v in obs.items():
                hit = _lookup(table, nucleus, atom_codes(mol, idx))
                if hit is None:
                    missing += 1
                    continue
                e = hit[0] - v
                errs.append(e)
                by_r.setdefault(hit[3], []).append(e)
        a = np.array(errs) if errs else np.zeros(0)
        res[nucleus] = {"n": int(len(a)), "missing": missing,
                        "mae": float(np.mean(np.abs(a))) if len(a) else None,
                        "rmse": float(np.sqrt(np.mean(a ** 2))) if len(a) else None,
                        "by_radius": {str(r): {"n": len(v),
                                               "mae": float(np.mean(np.abs(v)))}
                                      for r, v in sorted(by_r.items())}}
    return res

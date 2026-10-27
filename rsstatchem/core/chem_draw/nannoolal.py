"""Nannoolal / Rarey group-contribution methods: normal boiling point, critical temperature,
pressure and volume, and the liquid vapour-pressure curve.

Methods (Y. Nannoolal, J. Rarey, D. Ramjugernath and co-workers):

- normal boiling point: Nannoolal, Rarey, Ramjugernath, Cordes, Fluid Phase Equilib. 226
  (2004) 45-63; Tb = (sum + GI) / (n^a + b) + c with the published a, b, c (``TB_CONSTANTS``);
- critical properties: Nannoolal, Rarey, Ramjugernath, Fluid Phase Equilib. 252 (2007) 1-27;
  Tc = Tb (b + 1 / (a + sum^c)), Pc = M^b / (a + sum)^2 kPa, Vc = sum / n^a + b cm3/mol;
- vapour pressure: Nannoolal, Rarey, Ramjugernath, Fluid Phase Equilib. 269 (2008) 117-133;
  log10(P / atm) = (4.1012 + dB) (T/Tb - 1) / (T/Tb - 1/8), dB = sum + GI + a.

n = number of heavy atoms, M = molar mass (g/mol), GI = sum over all ordered pairs of
interaction groups i != j of C_ij / (n (m - 1)) with m the number of interaction groups.

Sources used here (all openly accessible):

- group definitions and priorities: Y. Nannoolal, "Development and critical evaluation of
  group contribution methods for the estimation of critical properties, liquid vapour pressure
  and liquid viscosity of organic compounds", PhD thesis, University of KwaZulu-Natal, Durban
  2006 (ResearchSpace, http://hdl.handle.net/10413/3766), Appendix B, Tables B-1 and B-2;
  the MSc thesis (Y. Nannoolal, UKZN 2004), Appendix B, for the boiling-point-only details;
- parameters: the DDBST files "2012-01-18 Rarey-Nannoolal {Tb, Tcri, Pcri, vcri, Vap}.txt"
  (https://chemthermo.ddbst.com/Parameters/), see :mod:`nannoolal_tables`; their Tc, Pc, Vc
  and dB values equal the PhD thesis tables (Appendix C) and reproduce its worked examples
  (Appendix D: Tc of 2,2,3,3-tetramethylbutane, Pc of diethylene glycol monomethyl ether, Vc
  of trichlorosilane, dB and vapour pressure of perfluoro-2-propanone), checked in the tests.

The atom-group assignment is RS-StatChem's reading of the published definitions (the original
software is not public); groups are assigned in the published priority order, every heavy atom
must belong to exactly one group, otherwise the molecule gets no estimate. Rings that RDKit
calls aromatic but that carry an exocyclic double bond (pyridones, uracils, caffeine's
pyrimidinedione ring) are treated as non-aromatic. Not implemented (molecules that need them
get no estimate): imines / hydrazines / azo compounds (groups 91, 98, 110, 114), the aminal
group 101, the "CO connected to N" group 109, phosphites (112) and imidazoles (106: the atoms
it spans are not clear from the definition). A zero or blank first-order value in a parameter
file means "not regressed": that property is then n/a.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from rdkit import Chem
from rdkit.Chem import Descriptors

__all__ = ["NannoolalResult", "fragment", "nannoolal", "estimate", "vapour_pressure",
           "INTERACTION_GROUPS", "GROUP_NAMES"]

E_ELEMENTS = {"N", "O", "F", "Cl"}          # "very electronegative neighbours"
HAL_FCL = {"F", "Cl"}
HALOGENS = {"F", "Cl", "Br", "I"}

# interaction group -> first-order groups (identical in every parameter file)
INTERACTION_GROUPS = {
    "OH": (33, 34, 35, 36), "OH(a)": (37,), "NH2": (40, 41), "NH": (42, 97), "SH": (53,),
    "COOH": (44,), "OCN": (80,), "EtherO": (38,), "Epox": (39,), "Ester": (45, 46, 47),
    "Ketone": (51, 92), "Alde": (52, 90), "Teth": (54,), "Nitro": (69,), "AtS": (56,),
    "CN": (57,), "AO": (65,), "AN5": (66,), "AN6": (67,)}
_IGROUP = {g: name for name, gs in INTERACTION_GROUPS.items() for g in gs}

GROUP_NAMES = {
    1: "CH3-", 2: "CH3-(N,O,F,Cl)", 3: "CH3-(aromatic)", 4: "-CH2- (chain)",
    5: ">CH- (chain)", 6: ">C< (chain)", 7: "C (chain) on N,O,F,Cl", 8: "C (chain) on aromatic C",
    9: "-CH2- (ring)", 10: ">CH- (ring)", 11: ">C< (ring)", 12: "C (ring) on N,O,F,Cl",
    13: "C (ring) on ring N,O", 14: "C (ring) on aromatic C", 15: "aromatic CH",
    16: "aromatic C-R", 17: "aromatic C-(N,O,F,Cl)", 18: "aromatic C (fused)",
    19: "F-C", 20: "F-C=C", 21: "F-C(F)<", 22: "F-C(F,Cl)", 23: "F-C(F,Cl)2", 24: "F-aromatic C",
    25: "Cl-C", 26: "Cl-C(F,Cl)", 27: "Cl-C(F,Cl)2", 28: "Cl-aromatic C", 29: "Cl-C=C",
    30: "Br-C", 31: "Br-aromatic C", 32: "I-C", 33: "-OH (tertiary)", 34: "-OH (secondary)",
    35: "-OH (primary, > C4)", 36: "-OH (short chain)", 37: "-OH (phenol)", 38: "-O- (ether)",
    39: "epoxide", 40: "-NH2", 41: "-NH2 (aromatic)", 42: "-NH- (chain)", 43: ">N-",
    44: "-COOH", 45: "-COO- (ester)", 46: "HCOO- (formate)", 47: "-COO- (lactone)",
    48: "-CON<", 49: "-CONH-", 50: "-CONH2", 51: "ketone", 52: "-CHO", 53: "-SH",
    54: "-S- (thioether)", 55: "-S-S-", 56: "aromatic S", 57: "-C#N", 58: ">C=C<",
    59: ">C=C< (on aromatic C)", 60: ">C=C< (on N,O,F,Cl)", 61: "H2C=C<", 62: ">C=C< (ring)",
    63: "-C#C-", 64: "HC#C-", 65: "aromatic O", 66: "aromatic N (5-ring)",
    67: "aromatic N (6-ring)", 68: "-NO2", 69: "-NO2 (aromatic)", 70: ">Si<", 71: ">Si< (3 C/H)",
    72: "nitrate", 73: "phosphate triester", 74: "nitrite", 75: "oxime", 76: "anhydride",
    77: "-COCl", 78: "boric ester", 79: "carbonate diester", 80: "-NCO", 81: "-NCS",
    82: "sulfone", 83: ">Sn<", 84: "-AsCl2", 85: "-GeCl3", 86: ">Ge<", 87: "C=C=C",
    88: "C=C-C=C (ring)", 89: "C=C-C=C (chain)", 90: "-CHO (aromatic)", 92: "ketone (aromatic)",
    93: ">Si< (2 C/H)", 94: "-O-O-", 95: "C#C-C#C", 96: "cyclic anhydride (unsaturated)",
    97: "-NH- (ring)", 99: "carbamate", 100: "urea", 102: "F-C(Cl)<", 103: "cyclic carbonate",
    104: "sulfate", 105: "sulfonamide", 106: "imidazole N-C-N", 107: "sulfoxide",
    108: "thiocyanate", 111: "cyanamide", 113: "phosphine", 115: "isoxazole O-N", 116: "Se",
    117: "Al", 214: "aromatic C (bridge)", 215: ">Si< (1 C/H)", 216: ">Si< (no C/H)",
    119: "C=O on C(F,Cl)2", 120: "C=O on two C(F,Cl)2", 121: "C(F,Cl)3", 122: "(C)2C(F,Cl)2",
    123: "no hydrogen", 124: "one hydrogen", 125: "3/4-ring", 126: "5-ring", 127: "ortho pair",
    128: "meta pair", 129: "para pair", 130: "(C=)(C)C-CC3", 131: "C2C-CC2", 132: "C3C-CC2",
    133: "C3C-CC3", 134: "C=C-C=O", 217: "Si-halogen"}


@dataclass
class NannoolalResult:
    groups: dict = field(default_factory=dict)          # first-order group id -> count
    corrections: dict = field(default_factory=dict)     # correction id -> count
    interaction_types: list = field(default_factory=list)   # one entry per group instance
    n_atoms: int = 0
    mw: float = 0.0
    missing: str = ""

    @property
    def ok(self) -> bool:
        return not self.missing

    def interaction_sum(self, table: dict) -> float:
        """GI = sum_{i != j} C(type_i, type_j) / (n (m - 1))."""
        t = self.interaction_types
        m = len(t)
        if m < 2:
            return 0.0
        s = 0.0
        for i, a in enumerate(t):
            for j, b in enumerate(t):
                if i != j:
                    s += table.get((a, b), table.get((b, a), 0.0))
        return s / (self.n_atoms * (m - 1))

    def interaction_counts(self) -> dict:
        """{(type a, type b): frequency} as in the published examples (N = I / (m - 1))."""
        t = self.interaction_types
        m = len(t)
        out: dict = {}
        if m < 2:
            return out
        for i, a in enumerate(t):
            for j, b in enumerate(t):
                if i != j:
                    key = tuple(sorted((a, b)))
                    out[key] = out.get(key, 0) + 1 / (m - 1)
        return out


# --- molecule preparation ----------------------------------------------------------------------

class _Mol:
    """Atom / bond data with the method's aromaticity (no exocyclic double bonds in aromatic
    rings) and Kekule bond orders."""

    def __init__(self, mol: Chem.Mol):
        self.mol = mol
        kek = Chem.Mol(mol)
        Chem.Kekulize(kek, clearAromaticFlags=True)
        self.n = mol.GetNumAtoms()
        self.sym = [a.GetSymbol() for a in mol.GetAtoms()]
        self.h = [a.GetTotalNumHs() for a in mol.GetAtoms()]
        self.nbrs = [[b.GetIdx() for b in a.GetNeighbors()] for a in mol.GetAtoms()]
        self.kek_order: dict = {}
        for b in kek.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            o = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2,
                 Chem.BondType.TRIPLE: 3}.get(b.GetBondType(), 0)
            self.kek_order[(i, j)] = self.kek_order[(j, i)] = o
        ri = mol.GetRingInfo()
        self.atom_rings = [tuple(r) for r in ri.AtomRings()]
        self.bond_in_ring = {}
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            self.bond_in_ring[(i, j)] = self.bond_in_ring[(j, i)] = b.IsInRing()
        self.in_ring = [a.IsInRing() for a in mol.GetAtoms()]
        # aromatic rings: RDKit aromatic rings without exocyclic double bonds
        rd_arom = [all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r) and
                   all(mol.GetBondBetweenAtoms(r[k], r[(k + 1) % len(r)]).GetIsAromatic()
                       for k in range(len(r))) for r in self.atom_rings]
        self.arom_rings = []
        for r, ar in zip(self.atom_rings, rd_arom):
            if not ar:
                continue
            rs = set(r)
            exo = any(self.kek_order[(i, j)] == 2 for i in r for j in self.nbrs[i]
                      if j not in rs)
            if not exo:
                self.arom_rings.append(r)
        self.arom = [False] * self.n
        self.arom_bond: set = set()
        for r in self.arom_rings:
            for k, i in enumerate(r):
                self.arom[i] = True
                j = r[(k + 1) % len(r)]
                self.arom_bond.add((i, j))
                self.arom_bond.add((j, i))

    # bond order: 'a' for aromatic, else the Kekule order
    def order(self, i: int, j: int):
        return "a" if (i, j) in self.arom_bond else self.kek_order[(i, j)]

    def single(self, i: int) -> bool:
        """All bonds of atom i are non-aromatic single bonds (sp3)."""
        return all(self.order(i, j) == 1 for j in self.nbrs[i])

    def double_partners(self, i: int) -> list[int]:
        return [j for j in self.nbrs[i] if self.order(i, j) == 2]

    def terminal_o(self, i: int, order: int = 2) -> list[int]:
        """Oxygen neighbours bonded to i only, with the given bond order."""
        return [j for j in self.nbrs[i] if self.sym[j] == "O" and len(self.nbrs[j]) == 1 and
                self.order(i, j) == order]

    def ring_sizes(self, i: int, aromatic: bool | None = None) -> set[int]:
        rings = self.arom_rings if aromatic else self.atom_rings
        return {len(r) for r in rings if i in r}


# --- first-order groups ------------------------------------------------------------------------

def _carbonyl(m: _Mol, c: int):
    """Terminal =O of a carbon (non-aromatic), or None."""
    if m.sym[c] != "C" or m.arom[c]:
        return None
    o = m.terminal_o(c, 2)
    return o[0] if len(o) == 1 else None


def _single_o_c(m: _Mol, c: int, used, ring: bool | None = None):
    """Ether-type O neighbours (single bond, two heavy neighbours, the other a C/Si)."""
    out = []
    for o in m.nbrs[c]:
        if m.sym[o] != "O" or o in used or m.order(c, o) != 1 or len(m.nbrs[o]) != 2 or m.h[o]:
            continue
        other = [x for x in m.nbrs[o] if x != c][0]
        if m.sym[other] in ("C", "Si"):
            out.append(o)
    return out


def _assign_multi(m: _Mol, used: set, add, tb_set: bool) -> str:
    """Multi-atom groups in priority order (the PR column of Table B-1). Returns an error
    text when the molecule contains a structure the method does not cover."""
    S, nb = m.sym, m.nbrs

    def free(*atoms):
        return not any(a in used for a in atoms)

    carbonyls = [c for c in range(m.n) if _carbonyl(m, c) is not None]
    # PR 1: 1,2-diketone - not fragmented by the method
    for c in carbonyls:
        for d in nb[c]:
            if d in carbonyls and all(S[x] == "C" for x in nb[c] + nb[d]
                                      if x not in (m.terminal_o(c) + m.terminal_o(d))):
                return "1,2-diketones are not covered"

    def carbonyl_env(c):
        o = _carbonyl(m, c)
        rest = [x for x in nb[c] if x != o]
        return o, rest

    # PR 2: carbamate -O-C(=O)-N< ; PR 3: urea >N-C(=O)-N<
    for c in carbonyls:
        o, rest = carbonyl_env(c)
        ns = [x for x in rest if S[x] == "N" and not m.arom[x] and m.order(c, x) == 1]
        os_ = [x for x in rest if S[x] == "O" and m.order(c, x) == 1 and len(nb[x]) == 2]
        if len(rest) == 2 and len(ns) == 1 and len(os_) == 1 and free(c, o, ns[0], os_[0]):
            add(99, c, o, ns[0], os_[0])
    for c in carbonyls:
        o, rest = carbonyl_env(c)
        ns = [x for x in rest if S[x] == "N" and not m.arom[x] and m.order(c, x) == 1]
        if len(rest) == 2 and len(ns) == 2 and free(c, o, *ns):
            add(100, c, o, *ns)
    # PR 4: ketone with two non-aromatic C
    for c in carbonyls:
        o, rest = carbonyl_env(c)
        if len(rest) == 2 and all(S[x] == "C" and not m.arom[x] for x in rest) and free(c, o):
            add(51, c, o)
    # PR 5: allene; PR 6/7: conjugated dienes; PR 8: conjugated diyne
    for c in range(m.n):
        dp = [x for x in m.double_partners(c) if S[x] == "C"]
        if S[c] == "C" and len(dp) == 2 and free(c, *dp):
            add(87, dp[0], c, dp[1])
    cc_double = [(i, j) for i in range(m.n) for j in nb[i] if i < j and S[i] == S[j] == "C"
                 and m.order(i, j) == 2]
    for (a, b) in cc_double:
        if not free(a, b):
            continue
        done = False
        for x, y in ((a, b), (b, a)):
            for z in nb[y]:
                if z == x or S[z] != "C" or m.order(y, z) != 1:
                    continue
                for (p, q) in cc_double:
                    if z in (p, q) and free(p, q) and x not in (p, q) and y not in (p, q):
                        w = q if z == p else p
                        ring = m.bond_in_ring[(x, y)] and m.bond_in_ring[(z, w)]
                        add(88 if ring else 89, x, y, z, w)
                        done = True
                        break
                if done:
                    break
            if done:
                break
    for i in range(m.n):
        for j in nb[i]:
            if S[i] == S[j] == "C" and m.order(i, j) == 3 and free(i, j):
                for k in nb[j]:
                    if k == i or S[k] != "C" or m.order(j, k) != 1:
                        continue
                    trip = [x for x in nb[k] if x != j and S[x] == "C" and m.order(k, x) == 3]
                    if trip and free(k, trip[0]):
                        add(95, i, j, k, trip[0])
                        break
    # PR 9: phosphate triester P(=O)(OC)3
    for p in range(m.n):
        if S[p] != "P":
            continue
        o = m.terminal_o(p, 2)
        os_ = [x for x in nb[p] if S[x] == "O" and m.order(p, x) == 1 and len(nb[x]) == 2]
        if len(o) == 1 and len(os_) == 3 and len(nb[p]) == 4 and free(p, o[0], *os_):
            add(73, p, o[0], *os_)
    # PR 10/11: anhydrides C(=O)-O-C(=O)
    for oo in range(m.n):
        if S[oo] != "O" or len(nb[oo]) != 2 or oo in used:
            continue
        c1, c2 = nb[oo]
        if c1 in carbonyls and c2 in carbonyls and free(c1, c2):
            o1, o2 = _carbonyl(m, c1), _carbonyl(m, c2)
            r1 = [x for x in nb[c1] if x not in (oo, o1)]
            r2 = [x for x in nb[c2] if x not in (oo, o2)]
            if len(r1) == 1 and len(r2) == 1 and S[r1[0]] == "C" and S[r2[0]] == "C":
                unsat = (m.in_ring[oo] and r1[0] in nb[r2[0]] and
                         m.order(r1[0], r2[0]) in (2, "a"))
                add(96 if unsat else 76, c1, o1, oo, c2, o2)
    # PR 12: GeCl3
    for g in range(m.n):
        cl = [x for x in nb[g] if S[x] == "Cl"]
        if S[g] == "Ge" and len(cl) == 3 and len(nb[g]) == 4 and free(g, *cl):
            add(85, g, *cl)
    # PR 13: nitrate O-N(=O)=O (charge-separated in RDKit)
    for n_ in range(m.n):
        if S[n_] == "N" and len(nb[n_]) == 3 and all(S[x] == "O" for x in nb[n_]):
            term = [x for x in nb[n_] if len(nb[x]) == 1]
            bridge = [x for x in nb[n_] if len(nb[x]) == 2]
            if len(term) == 2 and len(bridge) == 1 and free(n_, *nb[n_]):
                add(72, n_, *nb[n_])
    # PR 14: non-cyclic carbonate diester
    for c in carbonyls:
        o, rest = carbonyl_env(c)
        os_ = [x for x in rest if S[x] == "O" and len(nb[x]) == 2 and m.order(c, x) == 1]
        if len(rest) == 2 and len(os_) == 2 and not m.in_ring[c] and free(c, o, *os_):
            add(79, c, o, *os_)
    # PR 15: boric acid triester (non-cyclic)
    for b in range(m.n):
        os_ = [x for x in nb[b] if S[x] == "O" and len(nb[x]) == 2]
        if S[b] == "B" and len(os_) == 3 and len(nb[b]) == 3 and not m.in_ring[b] and \
                free(b, *os_):
            add(78, b, *os_)
    # PR 16: AsCl2
    for a in range(m.n):
        cl = [x for x in nb[a] if S[x] == "Cl"]
        if S[a] == "As" and len(cl) == 2 and len(nb[a]) == 3 and free(a, *cl):
            add(84, a, *cl)
    # PR 17: sulfone (C)-SO2-(C)
    for s in range(m.n):
        if S[s] != "S":
            continue
        o = m.terminal_o(s, 2)
        rest = [x for x in nb[s] if x not in o]
        if len(o) == 2 and len(rest) == 2 and all(S[x] == "C" for x in rest) and free(s, *o):
            add(82, s, *o)
    # PR 18: acid chloride
    for c in carbonyls:
        o, rest = carbonyl_env(c)
        cl = [x for x in rest if S[x] == "Cl"]
        if len(rest) == 2 and len(cl) == 1 and free(c, o, cl[0]):
            add(77, c, o, cl[0])
    # PR 19: isothiocyanate S=C=N-C
    for c in range(m.n):
        if S[c] != "C" or len(nb[c]) != 2:
            continue
        s_ = [x for x in nb[c] if S[x] == "S" and m.order(c, x) == 2 and len(nb[x]) == 1]
        n_ = [x for x in nb[c] if S[x] == "N" and m.order(c, x) == 2]
        if s_ and n_ and free(c, s_[0], n_[0]):
            add(81, s_[0], c, n_[0])
    # PR 20/21: nitro
    for n_ in range(m.n):
        if S[n_] != "N" or n_ in used:
            continue
        oo = [x for x in nb[n_] if S[x] == "O" and len(nb[x]) == 1]
        cs = [x for x in nb[n_] if S[x] == "C"]
        if len(oo) == 2 and len(cs) == 1 and len(nb[n_]) == 3 and free(*oo):
            add(69 if m.arom[cs[0]] else 68, n_, *oo)
    # PR 22: nitrite O=N-O-C
    for n_ in range(m.n):
        if S[n_] != "N" or n_ in used or len(nb[n_]) != 2:
            continue
        od = [x for x in nb[n_] if S[x] == "O" and len(nb[x]) == 1 and m.order(n_, x) == 2]
        os_ = [x for x in nb[n_] if S[x] == "O" and len(nb[x]) == 2]
        if od and os_ and free(od[0], os_[0]):
            add(74, n_, od[0], os_[0])
    # PR 23: COOH ; PR 24/25: esters ; PR 26: formate
    for c in carbonyls:
        o, rest = carbonyl_env(c)
        oh = [x for x in rest if S[x] == "O" and m.h[x] == 1 and len(nb[x]) == 1]
        cs = [x for x in rest if S[x] == "C"]
        if len(rest) == 2 and oh and cs and free(c, o, oh[0]):
            add(44, c, o, oh[0])
    for c in carbonyls:
        o, rest = carbonyl_env(c)
        eo = _single_o_c(m, c, used)
        cs = [x for x in rest if S[x] == "C"]
        if len(rest) == 2 and len(eo) == 1 and len(cs) == 1 and free(c, o, eo[0]):
            add(47 if m.bond_in_ring[(c, eo[0])] else 45, c, o, eo[0])
    for c in carbonyls:
        o, rest = carbonyl_env(c)
        eo = _single_o_c(m, c, used)
        if len(rest) == 1 and m.h[c] == 1 and len(eo) == 1 and free(c, o, eo[0]):
            add(46, c, o, eo[0])
    # PR 27: -CONH2
    for c in carbonyls:
        o, rest = carbonyl_env(c)
        ns = [x for x in rest if S[x] == "N" and m.h[x] == 2 and len(nb[x]) == 1]
        others = [x for x in rest if x not in ns]
        if ns and all(S[x] == "C" for x in others) and len(rest) <= 2 and free(c, o, ns[0]):
            add(50, c, o, ns[0])
    # PR 28: isocyanate O=C=N-(C,Si)
    for c in range(m.n):
        if S[c] != "C" or len(nb[c]) != 2:
            continue
        o_ = [x for x in nb[c] if S[x] == "O" and m.order(c, x) == 2 and len(nb[x]) == 1]
        n_ = [x for x in nb[c] if S[x] == "N" and m.order(c, x) == 2]
        if o_ and n_ and free(c, o_[0], n_[0]):
            add(80, o_[0], c, n_[0])
    # PR 29: oxime C=N-OH
    for n_ in range(m.n):
        if S[n_] != "N" or n_ in used or m.arom[n_]:
            continue
        cd = [x for x in nb[n_] if S[x] == "C" and m.order(n_, x) == 2]
        oh = [x for x in nb[n_] if S[x] == "O" and m.h[x] == 1 and len(nb[x]) == 1]
        if cd and oh and len(nb[n_]) == 2 and free(cd[0], oh[0]):
            add(75, cd[0], n_, oh[0])
    # PR 31: peroxide
    for o1 in range(m.n):
        for o2 in nb[o1]:
            if o1 < o2 and S[o1] == S[o2] == "O" and m.order(o1, o2) == 1 and free(o1, o2):
                add(94, o1, o2)
    # PR 33: cyclic carbonate
    for c in carbonyls:
        o, rest = carbonyl_env(c)
        os_ = [x for x in rest if S[x] == "O" and len(nb[x]) == 2 and m.order(c, x) == 1]
        if len(rest) == 2 and len(os_) == 2 and free(c, o, *os_):
            add(103, c, o, *os_)
    # PR 34: sulfate ; PR 35: sulfonamide
    for s in range(m.n):
        if S[s] != "S" or s in used:
            continue
        o = m.terminal_o(s, 2)
        if len(o) != 2:
            continue
        os_ = [x for x in nb[s] if S[x] == "O" and len(nb[x]) == 2]
        ns = [x for x in nb[s] if S[x] == "N" and not m.arom[x]]
        if len(os_) == 2 and len(nb[s]) == 4 and free(s, *o, *os_):
            add(104, s, *o, *os_)
        elif len(ns) == 1 and len(nb[s]) == 4 and free(s, *o, ns[0]):
            add(105, s, *o, ns[0])
    # PR 36: imidazole (group 106, "..=CNC=NC=.."): which atoms the group spans is not clear
    # from the definition, and only Tb has a value - imidazoles get no estimate
    for r in m.arom_rings:
        ns = [i for i in r if S[i] == "N"]
        if len(r) == 5 and len(ns) == 2 and all(S[i] in ("C", "N") for i in r) and \
                any(S[i] == "C" and all(x in nb[i] for x in ns) for i in r):
            return "imidazole rings (group 106) are not covered"
    # PR 37: sulfoxide
    for s in range(m.n):
        if S[s] != "S" or s in used:
            continue
        o = m.terminal_o(s, 2)
        rest = [x for x in nb[s] if x not in o]
        if len(o) == 1 and len(rest) == 2 and all(S[x] == "C" for x in rest) and free(s, o[0]):
            add(107, s, o[0])
    # PR 38: thiocyanate S-C#N ; PR 41: cyanamide N-C#N ; PR 55 nitrile handled later
    for c in range(m.n):
        if S[c] != "C" or c in used or len(nb[c]) != 2:
            continue
        nt = [x for x in nb[c] if S[x] == "N" and m.order(c, x) == 3 and len(nb[x]) == 1]
        if not nt:
            continue
        other = [x for x in nb[c] if x != nt[0]][0]
        if S[other] == "S" and free(nt[0]):
            add(108, c, nt[0])
        elif S[other] == "N" and free(nt[0]):
            add(111, c, nt[0])
    # PR 43: phosphine (trivalent P with at least one C or S)
    for p in range(m.n):
        if S[p] == "P" and p not in used and m.single(p) and len(nb[p]) <= 3 and \
                any(S[x] in ("C", "S") for x in nb[p]):
            add(113, p)
    # PR 45: isoxazole -O-N= (aromatic)
    for o in range(m.n):
        if S[o] == "O" and m.arom[o] and o not in used:
            ns = [x for x in nb[o] if S[x] == "N" and m.arom[x] and x not in used]
            if ns:
                add(115, o, ns[0])
    # PR 46: selenium ; PR 47: aluminium
    for a in range(m.n):
        if a in used:
            continue
        if S[a] == "Se" and any(S[x] in ("C", "Si") for x in nb[a]):
            add(116, a)
        elif S[a] == "Al" and any(S[x] in ("C", "Si") for x in nb[a]):
            add(117, a)
    # PR 48/49: amides -CONH- and -CON<
    for c in carbonyls:
        if c in used:
            continue
        o, rest = carbonyl_env(c)
        ns = [x for x in rest if S[x] == "N" and not m.arom[x] and m.order(c, x) == 1
              and x not in used]
        others = [x for x in rest if x not in ns]
        if len(ns) != 1 or len(rest) > 2 or not all(S[x] == "C" for x in others):
            continue
        n_ = ns[0]
        if m.h[n_] == 1 and free(c, o, n_):
            add(49, c, o, n_)
        elif m.h[n_] == 0 and free(c, o, n_):
            add(48, c, o, n_)
    # PR 50: epoxide
    for r in m.atom_rings:
        if len(r) == 3 and sorted(S[i] for i in r) == ["C", "C", "O"] and free(*r):
            o = [i for i in r if S[i] == "O"][0]
            cs = [i for i in r if i != o]
            if all(m.single(i) for i in cs):
                add(39, o, *cs)
    # PR 51: disulfide
    for s1 in range(m.n):
        for s2 in nb[s1]:
            if s1 < s2 and S[s1] == S[s2] == "S" and free(s1, s2) and \
                    len(nb[s1]) == len(nb[s2]) == 2:
                add(55, s1, s2)
    # PR 52/53: aldehydes ; PR 54: aromatic ketone
    for c in carbonyls:
        if c in used:
            continue
        o, rest = carbonyl_env(c)
        if m.h[c] == 1 and len(rest) == 1 and S[rest[0]] == "C" and free(c, o):
            add(90 if m.arom[rest[0]] else 52, c, o)
    for c in carbonyls:
        if c in used:
            continue
        o, rest = carbonyl_env(c)
        if len(rest) == 2 and all(S[x] == "C" for x in rest) and any(m.arom[x] for x in rest) \
                and free(c, o):
            add(92, c, o)
    # PR 55: nitrile on C
    for c in range(m.n):
        if S[c] != "C" or c in used or len(nb[c]) != 2:
            continue
        nt = [x for x in nb[c] if S[x] == "N" and m.order(c, x) == 3 and len(nb[x]) == 1]
        other = [x for x in nb[c] if x not in nt]
        if nt and S[other[0]] == "C" and free(nt[0]):
            add(57, c, nt[0])
    # PR 56: HC#C- ; PR 61: -C#C-
    alk = [(i, j) for i in range(m.n) for j in nb[i] if i < j and S[i] == S[j] == "C"
           and m.order(i, j) == 3]
    for i, j in alk:
        if free(i, j) and (m.h[i] == 1 or m.h[j] == 1):
            add(64, i, j)
    # PR 57-62: C=C groups
    for i, j in cc_double:
        if not free(i, j):
            continue
        if (m.h[i] == 2 and len(nb[i]) == 1) or (m.h[j] == 2 and len(nb[j]) == 1):
            add(61, i, j)
    subs = {}
    for i, j in cc_double:
        others = [x for x in nb[i] + nb[j] if x not in (i, j)]
        subs[(i, j)] = others
    for i, j in cc_double:
        ring = m.bond_in_ring[(i, j)]
        if free(i, j) and not ring and any(S[x] in E_ELEMENTS for x in subs[(i, j)]):
            add(60, i, j)
    for i, j in cc_double:
        ring = m.bond_in_ring[(i, j)]
        if free(i, j) and not ring and any(S[x] == "C" and m.arom[x] for x in subs[(i, j)]):
            add(59, i, j)
    for i, j in cc_double:
        if free(i, j) and m.bond_in_ring[(i, j)]:
            add(62, i, j)
    for i, j in alk:
        if free(i, j) and len(nb[i]) == 2 and len(nb[j]) == 2:
            add(63, i, j)
    for i, j in cc_double:
        if free(i, j) and len(nb[i]) >= 2 and len(nb[j]) >= 2:
            add(58, i, j)
    return ""


def _assign_single(m: _Mol, used: set, add, tb_set: bool, n_carbon: int) -> None:
    """Single-atom groups (halogens, O, N, S, Si, Ge, Sn, C) in priority order."""
    S, nb = m.sym, m.nbrs

    def free_atoms(sym):
        return [a for a in range(m.n) if S[a] == sym and a not in used]

    def halogen_env(x):
        """(carrier atom, number of other F/Cl on it, carrier is sp3 C or Si)."""
        c = nb[x][0]
        others = sum(1 for y in nb[c] if y != x and S[y] in HAL_FCL)
        sp3 = S[c] == "Si" or (S[c] == "C" and not m.arom[c] and m.single(c))
        return c, others, sp3

    # PR 63-67: I, Sn, Br, Ge
    for x in free_atoms("I"):
        if len(nb[x]) == 1 and S[nb[x][0]] in ("C", "Si"):
            add(32, x)
    for x in free_atoms("Sn"):
        if len(nb[x]) == 4 and all(S[y] == "C" for y in nb[x]):
            add(83, x)
    for x in free_atoms("Br"):
        if len(nb[x]) == 1 and S[nb[x][0]] in ("C", "Si"):
            add(31 if m.arom[nb[x][0]] else 30, x)
    for x in free_atoms("Ge"):
        if len(nb[x]) == 4 and all(S[y] == "C" for y in nb[x]):
            add(86, x)
    # PR 68-72: chlorine
    cls = [x for x in free_atoms("Cl") if len(nb[x]) == 1 and S[nb[x][0]] in ("C", "Si")]
    for x in cls:
        c, others, sp3 = halogen_env(x)
        if sp3 and others >= 2:
            add(27, x)
    for x in cls:
        c, others, sp3 = halogen_env(x)
        if x not in used and S[c] == "C" and not m.arom[c] and m.double_partners(c):
            add(29, x)
    for x in cls:
        c, others, sp3 = halogen_env(x)
        if x not in used and sp3 and others == 1:
            add(26, x)
    for x in cls:
        c, others, sp3 = halogen_env(x)
        if x not in used and sp3 and others == 0:
            add(25, x)
    for x in cls:
        if x not in used and m.arom[nb[x][0]]:
            add(28, x)
    # PR 73-75: sulfur
    for x in free_atoms("S"):
        if m.h[x] == 1 and len(nb[x]) == 1 and S[nb[x][0]] == "C":
            add(53, x)
        elif not m.arom[x] and m.h[x] == 0 and len(nb[x]) == 2 and \
                all(S[y] == "C" for y in nb[x]) and m.single(x):
            add(54, x)
        elif m.arom[x]:
            add(56, x)
    # PR 76-80: silicon
    for x in free_atoms("Si"):
        ch = sum(1 for y in nb[x] if S[y] == "C") + m.h[x]
        if tb_set:          # 2004 boiling-point groups: Si(F,Cl) 93, Si(O) 71, else 70
            if any(S[y] in HAL_FCL for y in nb[x]):
                add(93, x)
            elif any(S[y] == "O" for y in nb[x]):
                add(71, x)
            else:
                add(70, x)
        else:
            add({0: 216, 1: 215, 2: 93, 3: 71}.get(ch, 70), x)
    # PR 81-87: fluorine
    fs = [x for x in free_atoms("F") if len(nb[x]) == 1 and S[nb[x][0]] in ("C", "Si")]

    def no_h_4bonded(c):
        return m.h[c] == 0 and len(nb[c]) == 4

    for x in fs:
        c, others, sp3 = halogen_env(x)
        n_f = sum(1 for y in nb[c] if y != x and S[y] == "F")
        if sp3 and no_h_4bonded(c) and n_f >= 1:
            add(21, x)
    for x in fs:
        c, others, sp3 = halogen_env(x)
        n_cl = sum(1 for y in nb[c] if S[y] == "Cl")
        if x not in used and sp3 and no_h_4bonded(c) and n_cl >= 1:
            add(102, x)
    for x in fs:
        c, others, sp3 = halogen_env(x)
        if x not in used and sp3 and others >= 2:
            add(23, x)
    for x in fs:
        c, others, sp3 = halogen_env(x)
        if x not in used and sp3 and others >= 1:
            add(22, x)
    for x in fs:
        c = nb[x][0]
        if x not in used and S[c] == "C" and not m.arom[c] and m.double_partners(c):
            add(20, x)
    for x in fs:
        if x not in used and m.arom[nb[x][0]]:
            add(24, x)
    for x in fs:
        if x not in used and not m.arom[nb[x][0]]:
            add(19, x)
    # PR 88-92: hydroxyl
    ohs = [x for x in free_atoms("O") if m.h[x] == 1 and len(nb[x]) == 1]

    def carbinol(x):
        c = nb[x][0]
        if S[c] not in ("C", "Si") or m.arom[c] or not m.single(c):
            return None
        return c

    for x in ohs:
        c = carbinol(x)
        if c is not None and n_carbon >= 5 and len(nb[c]) == 2 and \
                sum(1 for y in nb[c] if S[y] in ("C", "Si")) == 1:
            add(35, x)
    for x in ohs:
        if x not in used and m.arom[nb[x][0]] and S[nb[x][0]] == "C":
            add(37, x)
    for x in ohs:
        c = carbinol(x)
        if x not in used and c is not None and len(nb[c]) == 3 and m.h[c] == 1 and \
                sum(1 for y in nb[c] if S[y] in ("C", "Si")) == 2:
            add(34, x)
    for x in ohs:
        c = carbinol(x)
        if x not in used and c is not None and len(nb[c]) == 4:
            add(33, x)
    for x in ohs:
        c = carbinol(x)
        if x not in used and c is not None and n_carbon < 5 and len(nb[c]) <= 2 and \
                all(S[y] in ("C", "Si", "O") for y in nb[c]) and \
                not any(m.arom[y] for y in nb[c]):
            add(36, x)
    # PR 93-94: aromatic O, ether O
    for x in free_atoms("O"):
        if m.arom[x] and all(S[y] == "C" and m.arom[y] for y in nb[x]):
            add(65, x)
        elif not m.arom[x] and m.h[x] == 0 and len(nb[x]) == 2 and \
                all(S[y] in ("C", "Si") for y in nb[x]) and m.single(x):
            add(38, x)
    # PR 95-101: nitrogen
    for x in free_atoms("N"):
        if m.h[x] == 2 and len(nb[x]) == 1 and S[nb[x][0]] in ("C", "Si") and m.single(x):
            add(41 if m.arom[nb[x][0]] else 40, x)
    for x in free_atoms("N"):
        if m.arom[x]:
            if 6 in m.ring_sizes(x, aromatic=True):
                add(67, x)
            elif 5 in m.ring_sizes(x, aromatic=True):
                add(66, x)
    for x in free_atoms("N"):
        if not m.single(x) or any(S[y] not in ("C", "Si") for y in nb[x]):
            continue
        if m.h[x] == 1 and len(nb[x]) == 2:
            ring = all(m.bond_in_ring[(x, y)] for y in nb[x])
            add(97 if ring else 42, x)
        elif m.h[x] == 0 and len(nb[x]) == 3:
            add(43, x)
    # PR 103-121: carbon
    for x in free_atoms("C"):
        if m.h[x] == 3 and len(nb[x]) == 1 and m.single(x):
            y = nb[x][0]
            add(2 if S[y] in E_ELEMENTS else 3 if m.arom[y] else 1, x)
    for x in free_atoms("C"):
        if m.arom[x] and m.h[x] == 1:
            add(15, x)
    sp3 = [x for x in free_atoms("C") if not m.arom[x] and m.single(x) and m.h[x] <= 2]

    def on_e(x, ring_bond=None):
        for y in nb[x]:
            if S[y] in HAL_FCL:
                if ring_bond is not True:
                    return True
            elif S[y] in ("N", "O"):
                if ring_bond is None or m.bond_in_ring[(x, y)] == ring_bond:
                    return True
        return False

    def on_arom_c(x):
        return any(S[y] == "C" and m.arom[y] for y in nb[x])

    for x in sp3:
        if m.in_ring[x] and on_arom_c(x):
            add(14, x)
    for x in sp3:
        if x not in used and not m.in_ring[x] and on_e(x):
            add(7, x)
    for x in sp3:
        if x not in used and not m.in_ring[x] and on_arom_c(x):
            add(8, x)
    for x in sp3:
        if x not in used and m.in_ring[x] and on_e(x, ring_bond=False):
            add(12, x)
    for x in sp3:
        if x not in used and m.in_ring[x] and on_e(x, ring_bond=True):
            add(13, x)
    for x in sp3:
        if x not in used and not m.in_ring[x] and m.h[x] == 2:
            add(4, x)
    for x in sp3:
        if x not in used and m.in_ring[x] and m.h[x] == 2:
            add(9, x)
    arom_c = [x for x in free_atoms("C") if m.arom[x] and m.h[x] == 0]

    def exo(x):
        return [y for y in nb[x] if (x, y) not in m.arom_bond]

    for x in arom_c:
        e = exo(x)
        if len(e) == 1 and S[e[0]] in E_ELEMENTS:
            add(17, x)
    for x in arom_c:
        if x in used:
            continue
        e = exo(x)
        if not e and len(nb[x]) == 3:
            add(18, x)
    for x in arom_c:
        if x in used:
            continue
        e = exo(x)
        if len(e) == 1 and m.arom[e[0]]:
            add(18 if tb_set else 214, x)
    for x in arom_c:
        if x not in used and len(exo(x)) == 1:
            add(16, x)
    for x in sp3:
        if x not in used and m.in_ring[x] and m.h[x] == 1:
            add(10, x)
    for x in sp3:
        if x not in used and not m.in_ring[x] and m.h[x] == 1:
            add(5, x)
    for x in sp3:
        if x not in used and m.in_ring[x] and m.h[x] == 0:
            add(11, x)
    for x in sp3:
        if x not in used and not m.in_ring[x] and m.h[x] == 0:
            add(6, x)


def _corrections(m: _Mol, groups: dict, tb_set: bool) -> dict:
    S, nb = m.sym, m.nbrs
    out: dict[int, int] = {}

    def inc(k, n=1):
        if n:
            out[k] = out.get(k, 0) + n

    carbonyls = [c for c in range(m.n) if _carbonyl(m, c) is not None]
    for c in carbonyls:
        cs = [x for x in nb[c] if S[x] == "C"]
        # C=C-C=O: carbonyl on an sp2 carbon (aromatic or C=C)
        if any(m.arom[x] or any(S[y] == "C" and m.order(x, y) == 2 for y in nb[x])
               for x in cs):
            inc(134)
        k = sum(1 for x in cs if sum(1 for y in nb[x] if S[y] in HAL_FCL) >= 2)
        if k == 1:
            inc(119)
        elif k >= 2:
            inc(120)
    for c in range(m.n):
        if S[c] != "C":
            continue
        nh = sum(1 for y in nb[c] if S[y] in HAL_FCL)
        if nh >= 3:
            inc(121)
        elif nh == 2 and sum(1 for y in nb[c] if S[y] == "C") == 2:
            inc(122)
    total_h = sum(m.h)
    if total_h == 0:
        inc(123)
    elif total_h == 1:
        inc(124)
    arom_set = {frozenset(r) for r in m.arom_rings}
    for r in m.atom_rings:
        if frozenset(r) in arom_set:
            continue
        if len(r) in (3, 4):
            inc(125)
        elif len(r) == 5:
            inc(126)
    # ortho / meta / para pairs on benzene rings, counted once and only if alone
    rel: set[str] = set()
    for r in m.arom_rings:
        if len(r) != 6 or any(S[i] != "C" for i in r):
            continue
        sub = [k for k, i in enumerate(r)
               if any((i, y) not in m.arom_bond for y in nb[i])]
        for a in range(len(sub)):
            for b in range(a + 1, len(sub)):
                d = abs(sub[a] - sub[b])
                d = min(d, 6 - d)
                rel.add({1: "o", 2: "m", 3: "p"}[d])
    if len(rel) == 1:
        inc({"o": 127, "m": 128, "p": 129}[rel.pop()])
    # steric corrections around C-C single bonds (carbon neighbours only)
    for i in range(m.n):
        for j in nb[i]:
            if j <= i or S[i] != "C" or S[j] != "C" or m.order(i, j) != 1:
                continue
            ni = [y for y in nb[i] if y != j and S[y] == "C"]
            nj = [y for y in nb[j] if y != i and S[y] == "C"]
            di = sum(1 for y in ni if m.kek_order[(i, y)] == 2)
            dj = sum(1 for y in nj if m.kek_order[(j, y)] == 2)
            a, b = sorted(((len(ni), di), (len(nj), dj)))
            if a[0] == 3 and b[0] == 3 and di == dj == 0:
                inc(133)
            elif (a[0], b[0]) == (2, 3) and di == dj == 0:
                inc(132)
            elif (a[0], b[0]) == (2, 2) and di == dj == 0:
                inc(131)
            elif (a[0], b[0]) == (2, 3) and a[1] == 1 and b[1] == 0:
                inc(130)
    if not tb_set:
        for s in range(m.n):
            if S[s] == "Si" and any(S[y] in HALOGENS for y in nb[s]):
                inc(217)
    return out


def _precheck(mol) -> str:
    if mol is None or mol.GetNumAtoms() == 0:
        return "nothing drawn"
    if len(Chem.GetMolFrags(mol)) > 1:
        return "select a single molecule"
    for a in mol.GetAtoms():
        q = a.GetFormalCharge()
        if not q:
            continue
        nbs = [x.GetSymbol() for x in a.GetNeighbors()]
        if a.GetSymbol() == "N" and q == 1 and nbs.count("O") >= 2:
            continue                                    # nitro / nitrate
        if a.GetSymbol() == "O" and q == -1 and nbs == ["N"]:
            continue
        return "charged atoms have no group"
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        return "radicals have no group"
    return ""


def fragment(mol: Chem.Mol, tb_set: bool = False) -> NannoolalResult:
    """First-order groups, corrections and interaction groups of one molecule. ``tb_set``:
    the boiling-point group set (no groups 214-217, Si groups of the 2004 method)."""
    res = NannoolalResult()
    res.missing = _precheck(mol)
    if res.missing:
        return res
    mol = Chem.RemoveHs(mol, sanitize=True) if any(a.GetAtomicNum() == 1
                                                   for a in mol.GetAtoms()) else mol
    try:
        m = _Mol(mol)
    except Exception:  # noqa: BLE001
        res.missing = "the structure could not be kekulized"
        return res
    used: set[int] = set()
    owner: dict[int, int] = {}
    groups: dict[int, int] = {}
    instances: list[int] = []

    def add(gid: int, *atoms) -> None:
        if any(a in used for a in atoms):
            return
        used.update(atoms)
        for a in atoms:
            owner[a] = gid
        groups[gid] = groups.get(gid, 0) + 1
        instances.append(gid)

    why = _assign_multi(m, used, add, tb_set)
    if why:
        res.missing = why
        return res
    n_carbon = sum(1 for s in m.sym if s in ("C", "Si"))
    _assign_single(m, used, add, tb_set, n_carbon)
    left = [i for i in range(m.n) if i not in used]
    if left:
        a = mol.GetAtomWithIdx(left[0])
        where = "aromatic " if m.arom[left[0]] else ("ring " if a.IsInRing() else "")
        res.missing = f"no Nannoolal group for a {where}{a.GetSymbol()} atom with " \
                      f"{a.GetTotalNumHs()} H"
        return res
    res.groups = groups
    res.corrections = _corrections(m, groups, tb_set)
    res.interaction_types = [_IGROUP[g] for g in instances if g in _IGROUP]
    res.n_atoms = m.n
    res.mw = Descriptors.MolWt(mol)
    return res


# --- property equations ------------------------------------------------------------------------

def _sum(res: NannoolalResult, prop: str, params=None):
    """Group + correction + interaction sum for a property, or (None, why)."""
    from rsstatchem.core.chem_draw import nannoolal_tables as T
    table = (params or {}).get("groups") or T.GROUPS[prop]
    s = 0.0
    for gid, n in res.groups.items():
        v = table.get(gid)
        if v is None:
            return None, f"no {prop.upper()} parameter for group {gid} " \
                         f"({GROUP_NAMES.get(gid, gid)})"
        s += n * v
    for cid, n in res.corrections.items():
        s += n * (table.get(cid) or 0.0)
    inter = (params or {}).get("interactions") or T.INTERACTIONS[prop]
    s += res.interaction_sum(inter)
    return s, ""


def estimate(res: NannoolalResult, prop: str, tb: float | None = None,
             params: dict | None = None):
    """(value, why) for prop in 'tb' (K), 'tc' (K, needs ``tb``), 'pc' (bar), 'vc' (cm3/mol)
    or 'db' (vapour-pressure slope parameter). ``params`` may override the published
    parameters ({'groups', 'interactions', 'constants'}; used by the tests to reproduce the
    MSc-thesis examples)."""
    from rsstatchem.core.chem_draw import nannoolal_tables as T
    if not res.ok:
        return None, res.missing
    s, why = _sum(res, prop, params)
    if s is None:
        return None, why
    k = (params or {}).get("constants") or (TB_CONSTANTS if prop == "tb" else T.PARAMS[prop])
    n = res.n_atoms
    try:
        if prop == "tb":
            a, b, c = k
            return s / (n ** a + b) + c, ""
        if prop == "tc":
            if tb is None:
                return None, "needs the normal boiling point"
            a, b, c = k
            if s <= 0:
                return None, "group sum out of range"
            return tb * (b + 1.0 / (a + s ** c)), ""
        if prop == "pc":
            a, b = k
            return res.mw ** b / (a + s) ** 2 / 100.0, ""       # kPa -> bar
        if prop == "vc":
            a, b = k
            return s / n ** a + b, ""
        if prop == "db":
            return s + k[0], ""
    except (ValueError, ZeroDivisionError, OverflowError):
        return None, "group sum out of range"
    raise ValueError(prop)


def vapour_pressure(t: float, tb: float, db: float) -> float:
    """Liquid vapour pressure in Pa at temperature t (K) from Tb (K) and dB (2008 method)."""
    tr = t / tb
    return 101325.0 * 10 ** ((4.1012 + db) * (tr - 1.0) / (tr - 0.125))


# Tb = sum / (n^a + b) + c with the constants of the 2004 publication (they reproduce its worked
# examples with the group values of the DDBST file; the MSc thesis has 0.6587, 1.6902,
# 84.3359). The DDBST "revised" constants (0.6482, 1.5166, 81.73) with the same group values
# overestimate the open experimental boiling points by 17 K on average (benchmark: RMSE 35.1
# vs 29.5 K), so they are not used.
TB_CONSTANTS = (0.6583, 1.6868, 84.3395)
WALDEN_DSM = 56.5       # J/(mol K), entropy of fusion by Walden's rule (Yalkowsky 1979)


def solid_correction(t: float, tm: float | None) -> float:
    """log10(P_solid / P_subcooled liquid) below the melting point with Walden's rule,
    -dSm (Tm - T) / (ln(10) R T) (S. H. Yalkowsky, Ind. Eng. Chem. Fundam. 18 (1979) 108);
    0 for a liquid or an unknown melting point."""
    if tm is None or not math.isfinite(tm) or tm <= t:
        return 0.0
    return -WALDEN_DSM * (tm - t) / (math.log(10) * 8.314462618 * t)


def nannoolal(mol: Chem.Mol, tb: float | None = None) -> dict:
    """{'tb', 'tc', 'pc', 'vc', 'db'} (None = n/a) plus 'why' notes and the fragmentations.
    Tc uses ``tb`` (e.g. an experimental boiling point) when given, else the estimated Tb."""
    out: dict = {"why": {}}
    rtb = fragment(mol, tb_set=True)
    rot = fragment(mol, tb_set=False)
    out["frag_tb"], out["frag"] = rtb, rot
    v, why = estimate(rtb, "tb")
    out["tb"] = v
    out["why"]["tb"] = why
    for p in ("pc", "vc", "db"):
        v, why = estimate(rot, p)
        out[p] = v if v is None or math.isfinite(v) else None
        out["why"][p] = why
    tb_used = tb if tb is not None else out["tb"]
    v, why = estimate(rot, "tc", tb=tb_used)
    out["tc"] = v
    out["why"]["tc"] = why or (rtb.missing if tb_used is None else "")
    out["tc_tb"] = "given" if tb is not None else "estimated"
    return out

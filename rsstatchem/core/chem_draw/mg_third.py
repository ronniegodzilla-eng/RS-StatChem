"""Marrero & Gani (2001) third-order groups, found with graph logic on RDKit ring information.

J. Marrero, R. Gani, Fluid Phase Equilib. 183-184 (2001) 183-208, Table 9: 66 "level 3"
groups that describe polyfunctional chains (HO-(CH2)m-OH ...), links between ring systems
(aC-aC, aC-CH2-aC ...), aromatic rings fused to other rings ("AROMFUSED[2] s1" ...) and
multi-ring atoms. Their contributions (``gc_tables.MARRERO_GANI_3``) are added to the first-
and second-order sums inside the same equations. Values as tabulated in pychemqt
(lib/newComponent/marrerogani.py; its docstring reproduces the paper's worked examples), the Tb
column cross-checked against Y. Nannoolal, MSc thesis (UKZN 2004), Table A-8.

The group definitions are RS-StatChem's reading of the names, written as Python graph searches
because several of them cannot be expressed as plain SMARTS (chains of any length, "different
rings" vs "fused rings", substitution patterns of fused aromatic rings):

- "aC": aromatic carbon (in the aromaticity form the first-order groups were assigned in);
  "cyclic": ring atom; "fused" ring atom: in two or more SSSR rings.
- "(different rings)": the link between the two ring atoms is outside every ring;
  "(fused rings)": the bond is a ring bond and the aromatic carbon is a fusion atom.
- chains (CHn)m: m >= 1 non-ring sp3 carbons with 0-2 H; "m > 2" means at least 3; the two
  ends are counted once per pair, along the shortest such chain.
- AROMFUSED[k]: a six-membered aromatic carbon ring with k fusion atoms ([2]: two adjacent;
  [3]: three consecutive; [4a]: two separated pairs; [4p]: four consecutive); "s1", "s2" ...:
  substituted positions numbered from a fusion atom; PYRIDINEFUSED[2] (quinoline type, N next
  to a fusion atom), [2-iso] (isoquinoline type), [4] (acridine type), unsubstituted.

Groups used only for Tm with patterns we cannot define unambiguously (329, 333, 336-340, 343)
are not searched.
"""
from __future__ import annotations

from collections import deque

from rdkit import Chem

__all__ = ["third_order_groups"]


class _G:
    def __init__(self, mol: Chem.Mol):
        self.mol = mol
        self.n = mol.GetNumAtoms()
        self.sym = [a.GetSymbol() for a in mol.GetAtoms()]
        self.h = [a.GetTotalNumHs() for a in mol.GetAtoms()]
        self.arom = [a.GetIsAromatic() for a in mol.GetAtoms()]
        self.nb = [[x.GetIdx() for x in a.GetNeighbors()] for a in mol.GetAtoms()]
        ri = mol.GetRingInfo()
        self.nrings = [ri.NumAtomRings(i) for i in range(self.n)]
        self.rings = [tuple(r) for r in ri.AtomRings()]
        self.bond = {}
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            self.bond[(i, j)] = self.bond[(j, i)] = b

    def ring(self, i):
        return self.nrings[i] > 0

    def fused(self, i):
        return self.nrings[i] >= 2

    def rbond(self, i, j):
        return self.bond[(i, j)].IsInRing()

    def btype(self, i, j):
        return self.bond[(i, j)].GetBondType()

    def ac(self, i):
        return self.sym[i] == "C" and self.arom[i]

    def sp3c(self, i):
        return self.sym[i] == "C" and not self.arom[i] and all(
            self.btype(i, j) == Chem.BondType.SINGLE for j in self.nb[i])

    def chain_c(self, i):
        """Non-ring sp3 carbon with 0-2 H (a CHn chain member)."""
        return self.sp3c(i) and not self.ring(i) and self.h[i] <= 2

    def chcyc(self, i):
        """Non-aromatic ring sp3 CH."""
        return self.sp3c(i) and self.ring(i) and self.h[i] == 1

    def chains(self, start, stop_ok):
        """{end atom: chain length} for ends reachable from atom ``start`` through chain
        carbons only (shortest chains; the first chain atom must be a neighbour of start)."""
        out = {}
        seen = {start}
        q = deque()
        for x in self.nb[start]:
            if self.chain_c(x):
                q.append((x, 1))
                seen.add(x)
        while q:
            c, m = q.popleft()
            for y in self.nb[c]:
                if y in seen:
                    continue
                if stop_ok(y):
                    if y not in out or m < out[y]:
                        out[y] = m
                if self.chain_c(y):
                    seen.add(y)
                    q.append((y, m + 1))
        return out


def _ends(g: _G):
    """Functional chain ends: atom index sets."""
    oh = {i for i in range(g.n) if g.sym[i] == "O" and g.h[i] == 1 and len(g.nb[i]) == 1
          and g.sp3c(g.nb[i][0])}
    nh2 = {i for i in range(g.n) if g.sym[i] == "N" and g.h[i] == 2 and len(g.nb[i]) == 1
           and not g.arom[i]}
    sh = {i for i in range(g.n) if g.sym[i] == "S" and g.h[i] == 1 and len(g.nb[i]) == 1}
    cooh = set()
    cn = set()
    for i in range(g.n):
        if g.sym[i] != "C" or g.arom[i]:
            continue
        dbl_o = [x for x in g.nb[i] if g.sym[x] == "O" and g.btype(i, x) == Chem.BondType.DOUBLE]
        oh_ = [x for x in g.nb[i] if g.sym[x] == "O" and g.h[x] == 1 and len(g.nb[x]) == 1]
        if dbl_o and oh_:
            cooh.add(i)
        if any(g.sym[x] == "N" and g.btype(i, x) == Chem.BondType.TRIPLE for x in g.nb[i]):
            cn.add(i)
    nhk = {i for i in range(g.n) if g.sym[i] == "N" and not g.arom[i] and g.h[i] <= 1 and
           not g.ring(i) and all(g.sym[x] == "C" and g.btype(i, x) == Chem.BondType.SINGLE
                                 for x in g.nb[i])}
    return oh, nh2, sh, cooh, cn, nhk


def _pairs(g: _G, a: set, b: set, m_min: int) -> int:
    """Number of (x in a, y in b) end pairs joined by a chain of at least m_min carbons
    (unordered when a is b)."""
    found = set()
    for x in a:
        for y, m in g.chains(x, lambda z: z in b and z != x).items():
            if m >= m_min:
                found.add(frozenset((x, y)) if a is b else (x, y))
    return len(found)


def _via(g: _G, ends: set, centres) -> int:
    """End - (CHp)k - centre - (CHn)m - end with k, m >= 1 (centre = ether O, amine N):
    pairs of ends reached through different neighbours of the centre."""
    count = 0
    for c in centres:
        reach = [{e for e in ends if _side_path(g, c, s, e)} if g.chain_c(s) else set()
                 for s in g.nb[c]]
        pairs = {frozenset((e1, e2)) for k, r1 in enumerate(reach) for r2 in reach[k + 1:]
                 for e1 in r1 for e2 in r2 if e1 != e2}
        count += len(pairs)
    return count


def _side_path(g: _G, centre: int, first: int, end: int) -> bool:
    """Is ``end`` reachable from ``first`` through chain carbons without passing ``centre``?"""
    seen = {centre, first}
    q = deque([first])
    while q:
        c = q.popleft()
        for y in g.nb[c]:
            if y == end:
                return True
            if y not in seen and g.chain_c(y):
                seen.add(y)
                q.append(y)
    return False


def _aromfused(g: _G) -> dict:
    out: dict = {}
    for r in g.rings:
        if len(r) != 6 or not all(g.arom[i] for i in r):
            continue
        syms = [g.sym[i] for i in r]
        fused = [g.fused(i) for i in r]
        if not any(fused):
            continue
        sub = [not fused[k] and syms[k] == "C" and g.h[r[k]] == 0 for k in range(6)]
        n_n = syms.count("N")
        if n_n == 0 and all(s == "C" for s in syms):
            key = _carbon_ring_key(fused, sub)
        elif n_n == 1 and syms.count("C") == 5 and not any(sub):
            key = _pyridine_key(syms, fused)
        else:
            key = None
        if key:
            out[key] = out.get(key, 0) + 1
    return out


def _rotations(seq):
    """Every rotation of the ring sequence, in both directions."""
    for ring in (list(seq), list(reversed(seq))):
        for k in range(len(ring)):
            yield ring[k:] + ring[:k]


def _carbon_ring_key(fused, sub):
    """AROMFUSED key from the fusion / substitution pattern around a benzene ring."""
    pat = ["F" if f else ("S" if s else "H") for f, s in zip(fused, sub)]
    k = pat.count("F")
    for rot in _rotations(pat):
        s = "".join(rot)
        if k == 2 and s.startswith("FF"):
            rest = s[2:]           # positions 1..4 after the second fusion atom
            subs = tuple(i + 1 for i, c in enumerate(rest) if c == "S")
            key = {(): "af2", (1,): "af2_s1", (4,): "af2_s1", (2,): "af2_s2", (3,): "af2_s2",
                   (2, 3): "af2_s2s3", (1, 4): "af2_s1s4", (1, 2): "af2_s1s2",
                   (3, 4): "af2_s1s2", (1, 3): "af2_s1s3", (2, 4): "af2_s1s3"}.get(subs)
            if key:
                return key
        if k == 3 and s == "FFFHHH":
            return "af3"
        if k == 4:
            if s == "FFHFFH":
                return "af4a"
            if s in ("FFSFFH",):
                return "af4a_s1"
            if s == "FFSFFS":
                return "af4a_s1s4"
            if s == "FFFFHH":
                return "af4p"
            if s == "FFFFSS":
                return "af4p_s3s4"
    return None


def _pyridine_key(syms, fused):
    pat = ["N" if s == "N" else ("F" if f else "H") for s, f in zip(syms, fused)]
    for rot in _rotations(pat):
        s = "".join(rot)
        if s == "NFFHHH":
            return "pf2"
        if s == "NHFFHH":
            return "pf2iso"
        if s == "NFFHFF":
            return "pf4"
    return None


def third_order_groups(form: Chem.Mol) -> dict:
    """{third-order group key: occurrences} for a molecule (the aromaticity form used for the
    first-order groups)."""
    g = _G(form)
    out: dict[str, int] = {}

    def add(k, n=1):
        if n:
            out[k] = out.get(k, 0) + n

    oh, nh2, sh, cooh, cn, nhk = _ends(g)
    add("diacid", _pairs(g, cooh, cooh, 3))
    add("aminoalcohol", _pairs(g, nh2, oh, 3))
    add("diol", _pairs(g, oh, oh, 3))
    add("diamine", _pairs(g, nh2, nh2, 3))
    add("nh_nh2", _pairs(g, nhk, nh2, 3))
    add("dithiol", _pairs(g, sh, sh, 3))
    add("dinitrile", _pairs(g, cn, cn, 3))
    ether_o = [i for i in range(g.n) if g.sym[i] == "O" and not g.arom[i] and not g.ring(i)
               and len(g.nb[i]) == 2 and all(g.sp3c(x) for x in g.nb[i])]
    add("oh_o_oh", _via(g, oh, ether_o))
    add("oh_n_oh", _via(g, oh, nhk))
    n = 0
    for o in ether_o:
        for m in g.chains(o, lambda z: z in oh).values():
            if m >= 3:
                n += 1
    add("ether_alcohol", n)
    # links between rings
    for i in range(g.n):
        for j in g.nb[i]:
            if j < i:
                continue
            ring_bond = g.rbond(i, j)
            for a, b in ((i, j), (j, i)):
                if not g.ac(a):
                    continue
                if not ring_bond:
                    if a < b and g.ac(b) and g.btype(a, b) == Chem.BondType.SINGLE:
                        add("ac_ac")
                    if g.sp3c(b) and g.ring(b) and g.h[b] <= 1:
                        add("ac_chcyc_diff")
                    if g.sym[b] == "C" and g.ring(b) and not g.arom[b] and any(
                            g.sym[x] == "N" and g.ring(x) and g.rbond(b, x) and
                            g.btype(b, x) == Chem.BondType.DOUBLE for x in g.nb[b]):
                        add("ac_cn_cyc_diff")
                elif g.fused(a) and not g.arom[b]:
                    if g.sp3c(b) and g.h[b] <= 1:
                        add("ac_chcyc_fused")
                    if g.sym[b] == "C" and any(
                            g.sym[x] == "C" and g.rbond(b, x) and not g.arom[x] and
                            g.btype(b, x) == Chem.BondType.DOUBLE for x in g.nb[b]):
                        add("ac_cc_cyc_fused")
                    if g.sym[b] == "C" and any(
                            g.sym[x] == "O" and len(g.nb[x]) == 1 and
                            g.btype(b, x) == Chem.BondType.DOUBLE for x in g.nb[b]):
                        add("ac_co_cyc_fused")
                    if g.sym[b] == "S" and len(g.nb[b]) == 2:
                        add("ac_s_cyc_fused")
                    if g.sym[b] == "O" and len(g.nb[b]) == 2:
                        add("ac_o_cyc_fused")
                    if g.sym[b] == "N" and g.h[b] <= 1 and all(
                            g.btype(b, x) == Chem.BondType.SINGLE for x in g.nb[b]):
                        add("ac_nh_cyc_fused")
                    if g.sym[b] == "N" and any(
                            g.sym[x] == "C" and g.rbond(b, x) and
                            g.btype(b, x) == Chem.BondType.DOUBLE for x in g.nb[b]):
                        add("ac_nc_cyc_fused")
                    if g.sym[b] == "C" and any(
                            g.sym[x] == "N" and g.rbond(b, x) and
                            g.btype(b, x) == Chem.BondType.DOUBLE for x in g.nb[b]):
                        add("ac_cn_cyc_fused")
    # aC - X - aC, aC - X - Y - aC (different rings: the bridge is outside rings)
    acs = [i for i in range(g.n) if g.ac(i)]
    for x in range(g.n):
        if g.ring(x) or g.arom[x]:
            continue
        a_nb = [y for y in g.nb[x] if g.ac(y)]
        if len(a_nb) != 2:
            continue
        s = g.sym[x]
        if s == "C" and g.sp3c(x):
            add("ac_ch_ac")
        elif s == "C" and any(g.sym[y] == "O" and g.btype(x, y) == Chem.BondType.DOUBLE
                              for y in g.nb[x]):
            add("ac_co_ac")
        elif s == "S" and len(g.nb[x]) == 2:
            add("ac_s_ac")
        elif s == "S" and any(g.sym[y] == "O" for y in g.nb[x]):
            add("ac_so_ac")
        elif s == "N" and g.h[x] == 1:
            add("ac_nh_ac")
        elif s == "O":
            add("ac_o_ac")
    # aC - (CHn)m - aC with m >= 2, aC - (CHn)m - CHcyc, CHcyc - (CHn)m - CHcyc
    a_set = set(acs)
    chcyc = {i for i in range(g.n) if g.chcyc(i)}
    add("ac_chain_ac", _pairs(g, a_set, a_set, 2))
    add("ac_chain_chcyc", _pairs(g, a_set, chcyc, 1))
    add("chcyc_chain_chcyc", _pairs(g, chcyc, chcyc, 1))
    for i in chcyc:
        for j in g.nb[i]:
            if j > i and j in chcyc and not g.rbond(i, j):
                add("chcyc_chcyc")
    add("ch_multiring", sum(1 for i in chcyc if g.fused(i)))
    add("c_multiring", sum(1 for i in range(g.n)
                           if g.sp3c(i) and g.h[i] == 0 and g.fused(i)))
    # stilbene, deoxybenzoin, benzil, benzyl ethers
    for i in range(g.n):
        for j in g.nb[i]:
            if j < i or g.ring(i) or g.ring(j):
                continue
            bt = g.btype(i, j)
            ai = [y for y in g.nb[i] if y != j and g.ac(y)]
            aj = [y for y in g.nb[j] if y != i and g.ac(y)]
            if bt == Chem.BondType.DOUBLE and g.sym[i] == g.sym[j] == "C" and ai and aj:
                add("ac_cc_ac")
            if bt != Chem.BondType.SINGLE or not (ai and aj):
                continue
            si, sj = g.sym[i], g.sym[j]

            def is_co(x):
                return g.sym[x] == "C" and any(g.sym[y] == "O" and len(g.nb[y]) == 1 and
                                               g.btype(x, y) == Chem.BondType.DOUBLE
                                               for y in g.nb[x])
            if (g.sp3c(i) and is_co(j)) or (g.sp3c(j) and is_co(i)):
                add("ac_ch_co_ac")
            elif is_co(i) and is_co(j):
                add("ac_co_co_ac")
            elif (si == "O" and g.sp3c(j)) or (sj == "O" and g.sp3c(i)):
                add("ac_o_ch_ac")
    for o in range(g.n):
        if g.sym[o] != "O" or g.ring(o) or len(g.nb[o]) != 2:
            continue
        c1, c2 = g.nb[o]
        if all(g.sp3c(c) and not g.ring(c) and any(g.ac(y) for y in g.nb[c]) for c in (c1, c2)):
            add("ac_ch_o_ch_ac")
    # (CHm=C)cyc - CHp - (C=CHn)cyc (different rings), used for Tm
    for x in range(g.n):
        if not g.chain_c(x):
            continue
        ends = [y for y in g.nb[x] if g.sym[y] == "C" and g.ring(y) and not g.arom[y] and any(
            g.btype(y, z) == Chem.BondType.DOUBLE and g.rbond(y, z) and g.sym[z] == "C"
            for z in g.nb[y])]
        if len(ends) == 2:
            add("cyc_ch_cyc")
    for k, v in _aromfused(g).items():
        add(k, v)
    return out

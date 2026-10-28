"""Rule-based IUPAC names for teaching-level structures (ChemDraw's *Convert Structure to Name*
is licensed; no mature open generator exists).

Scope: acyclic and monocyclic compounds of C, H, N, O and the halogens - alkanes,
cycloalkanes, alkenes / alkynes, haloalkanes, alcohols and phenols, ethers, aldehydes,
ketones, carboxylic acids, esters, anhydrides, acyl halides, amides, nitriles, amines,
nitro compounds and benzene derivatives, with R/S and E/Z descriptors on the parent.
Fused, bridged and spiro rings, heterocycles and other elements are out of scope.

The rules follow the 2013 IUPAC Recommendations (P-41 seniority of classes, P-44 parent
selection: principal groups, rings before chains, chain length, unsaturation; P-31.1.4
lowest locants: principal groups, multiple bonds, prefixes, first cited prefix; alphanumerical
order of prefixes) in a simplified form. Esters, amides, nitriles, aldehydes, acyl halides
and anhydrides are named through their parent acid ("hexanoic acid" -> "hexanenitrile").

**A name is only returned after OPSIN has parsed it back to the same standard InChIKey**
(:func:`verified_name`); everything else gets None, never an unchecked name. Without Java /
OPSIN nothing is returned. :func:`candidate_name` gives the unverified string (tests only).
No Qt here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rdkit import Chem

__all__ = ["candidate_name", "verified_name", "verified_names", "NamerError", "SOURCE"]

SOURCE = "rule-based IUPAC namer (checked by OPSIN)"

STEMS = ["", "meth", "eth", "prop", "but", "pent", "hex", "hept", "oct", "non", "dec",
         "undec", "dodec", "tridec", "tetradec", "pentadec", "hexadec", "heptadec",
         "octadec", "nonadec", "icos"]
MULT = ["", "", "di", "tri", "tetra", "penta", "hexa", "hepta", "octa", "nona", "deca"]
MULT_COMPLEX = ["", "", "bis", "tris", "tetrakis", "pentakis", "hexakis"]
HALO = {"F": "fluoro", "Cl": "chloro", "Br": "bromo", "I": "iodo"}
HALIDE = {"F": "fluoride", "Cl": "chloride", "Br": "bromide", "I": "iodide"}
RETAINED_SUBST = {"1-methylethyl": "propan-2-yl", "1,1-dimethylethyl": "tert-butyl",
                  "1-methylpropyl": "butan-2-yl", "phenylmethyl": "benzyl",
                  "1-oxoethyl": "acetyl"}
ALKOXY = {"methyl": "methoxy", "ethyl": "ethoxy", "propyl": "propoxy", "butyl": "butoxy",
          "phenyl": "phenoxy", "tert-butyl": "tert-butoxy"}
ALLOWED = {"C", "H", "N", "O", "F", "Cl", "Br", "I"}


class NamerError(Exception):
    """The structure is outside the namer's scope."""


# --- perception --------------------------------------------------------------------------------

def _is_dbl_o(b, c) -> bool:
    o = b.GetOtherAtom(c)
    return b.GetBondType() == Chem.BondType.DOUBLE and o.GetSymbol() == "O"


def _nitro(n) -> bool:
    if n.GetSymbol() != "N" or n.GetFormalCharge() != 1:
        return False
    os_ = [x for x in n.GetNeighbors() if x.GetSymbol() == "O"]
    return len(os_) == 2 and sorted(o.GetFormalCharge() for o in os_) == [-1, 0] and \
        n.GetDegree() == 3


@dataclass
class Perception:
    acid: set = field(default_factory=set)        # COOH carbons
    ketone: set = field(default_factory=set)      # C(=O) with two C neighbours
    aldehyde: set = field(default_factory=set)    # C(=O)H carbons
    nitrile: set = field(default_factory=set)     # C#N carbons
    alcohol: dict = field(default_factory=dict)   # O -> carbon
    amine: dict = field(default_factory=dict)     # N -> [carbon neighbours]
    ester: list = field(default_factory=list)     # (carbonyl C, single O, alkyl C)
    amide: list = field(default_factory=list)     # (carbonyl C, N)
    acyl_halide: list = field(default_factory=list)   # (C, X)
    anhydride: list = field(default_factory=list)     # (C1, O, C2)
    nitro: set = field(default_factory=set)


def perceive(mol) -> Perception:
    p = Perception()
    used: set[int] = set()
    for a in mol.GetAtoms():                      # nitro groups first (their O- is charged)
        if _nitro(a):
            used.update(o.GetIdx() for o in a.GetNeighbors() if o.GetSymbol() == "O")
    for a in mol.GetAtoms():
        if a.GetSymbol() != "C":
            continue
        dbl_o = [b.GetOtherAtom(a) for b in a.GetBonds() if _is_dbl_o(b, a)]
        if len(dbl_o) > 1:
            raise NamerError("carbon with two C=O")
        if a.GetFormalCharge():
            raise NamerError("charged carbon")
        singles = [b.GetOtherAtom(a) for b in a.GetBonds()
                   if b.GetBondType() == Chem.BondType.SINGLE]
        het = [x for x in singles if x.GetSymbol() != "C"]
        cnb = [x for x in a.GetNeighbors() if x.GetSymbol() == "C"]
        triple_n = [b.GetOtherAtom(a) for b in a.GetBonds()
                    if b.GetBondType() == Chem.BondType.TRIPLE and
                    b.GetOtherAtom(a).GetSymbol() == "N"]
        if triple_n:
            n = triple_n[0]
            if a.GetDegree() > 2 or n.GetDegree() != 1:
                raise NamerError("odd nitrile")
            p.nitrile.add(a.GetIdx())
            used.add(n.GetIdx())
            continue
        if not dbl_o:
            continue
        o = dbl_o[0]
        used.add(o.GetIdx())
        if len(het) > 1:
            raise NamerError("carbonic acid derivative")
        if not het:
            if len(cnb) == 2:
                p.ketone.add(a.GetIdx())
            elif a.GetTotalNumHs() >= 1:
                p.aldehyde.add(a.GetIdx())
            else:
                raise NamerError("odd carbonyl")
            continue
        x = het[0]
        sx = x.GetSymbol()
        if x.GetFormalCharge():
            raise NamerError("carboxylate / charged acyl group")
        if sx in HALO:
            p.acyl_halide.append((a.GetIdx(), x.GetIdx()))
            used.add(x.GetIdx())
        elif sx == "O":
            others = [y for y in x.GetNeighbors() if y.GetIdx() != a.GetIdx()]
            if not others:
                p.acid.add(a.GetIdx())
            elif len(others) == 1 and others[0].GetSymbol() == "C":
                y = others[0]
                if x.IsInRing():
                    raise NamerError("lactone / cyclic anhydride")
                if any(_is_dbl_o(b, y) for b in y.GetBonds()):
                    if not any(t[1] == x.GetIdx() for t in p.anhydride):
                        p.anhydride.append((a.GetIdx(), x.GetIdx(), y.GetIdx()))
                else:
                    p.ester.append((a.GetIdx(), x.GetIdx(), y.GetIdx()))
            else:
                raise NamerError("peroxide / odd ester")
            used.add(x.GetIdx())
        elif sx == "N":
            if x.GetFormalCharge() or any(n.GetSymbol() != "C" for n in x.GetNeighbors()):
                raise NamerError("odd amide")
            if any(any(_is_dbl_o(b, n) for b in n.GetBonds()) for n in x.GetNeighbors()
                   if n.GetIdx() != a.GetIdx()):
                raise NamerError("imide")
            p.amide.append((a.GetIdx(), x.GetIdx()))
            used.add(x.GetIdx())
        else:
            raise NamerError("unsupported acyl group")
    for a in mol.GetAtoms():
        s, i = a.GetSymbol(), a.GetIdx()
        if s not in ALLOWED:
            raise NamerError(f"element {s}")
        if a.GetNumRadicalElectrons() or a.GetIsotope():
            raise NamerError("radical / isotope")
        if s == "C" or i in used:
            continue
        if s in HALO:
            if a.GetDegree() != 1 or a.GetNeighbors()[0].GetSymbol() != "C":
                raise NamerError("odd halogen")
            continue
        if _nitro(a):
            p.nitro.add(i)
            for o in a.GetNeighbors():
                if o.GetSymbol() == "O":
                    used.add(o.GetIdx())
            if a.GetNeighbors()[0].GetSymbol() != "C" and \
                    not any(n.GetSymbol() == "C" for n in a.GetNeighbors()):
                raise NamerError("odd nitro")
            continue
        if a.GetFormalCharge():
            raise NamerError("charged atom")
        if any(b.GetBondType() != Chem.BondType.SINGLE for b in a.GetBonds()):
            raise NamerError(f"{s} with a multiple bond")
        nb = list(a.GetNeighbors())
        if any(n.GetSymbol() != "C" for n in nb):
            raise NamerError("heteroatom-heteroatom bond")
        if s == "O":
            if len(nb) == 1:
                p.alcohol[i] = nb[0].GetIdx()
            elif len(nb) != 2:
                raise NamerError("odd oxygen")
        elif s == "N":
            p.amine[i] = [n.GetIdx() for n in nb]
    return p


# --- naming context -----------------------------------------------------------------------------

class Ctx:
    def __init__(self, mol, atom_labels: dict, bond_labels: dict):
        self.mol = mol
        self.p = perceive(mol)
        self.orig = [a.GetIntProp("orig") if a.HasProp("orig") else a.GetIdx()
                     for a in mol.GetAtoms()]
        self.atom_labels = atom_labels       # orig idx -> R/S
        self.bond_labels = bond_labels       # frozenset(orig idx pair) -> E/Z
        ri = mol.GetRingInfo()
        self.rings = [list(r) for r in ri.AtomRings()]
        self.ring_of = {}
        for k, r in enumerate(self.rings):
            for i in r:
                if i in self.ring_of:
                    raise NamerError("fused / spiro rings")
                self.ring_of[i] = k
        for r in self.rings:
            if any(mol.GetAtomWithIdx(i).GetSymbol() != "C" for i in r):
                raise NamerError("heterocycle")
            arom = [mol.GetAtomWithIdx(i).GetIsAromatic() for i in r]
            if any(arom) and not (all(arom) and len(r) == 6):
                raise NamerError("aromatic ring other than benzene")
        self.skeleton = {a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C"} - \
            self.p.nitrile

    def atom(self, i):
        return self.mol.GetAtomWithIdx(i)

    def bond(self, i, j):
        return self.mol.GetBondBetweenAtoms(i, j)

    def order(self, i, j) -> int:
        b = self.bond(i, j)
        if b is None or b.GetIsAromatic():
            return 1
        return {Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3}.get(b.GetBondType(), 1)

    def ordered_ring(self, k: int) -> list[int]:
        ring = self.rings[k]
        rs = set(ring)
        out = [ring[0]]
        prev = None
        while len(out) < len(ring):
            cur = out[-1]
            nxt = [n.GetIdx() for n in self.atom(cur).GetNeighbors()
                   if n.GetIdx() in rs and n.GetIdx() != prev and n.GetIdx() not in out]
            prev = cur
            out.append(nxt[0])
        return out


def _alpha_key(name: str) -> str:
    n = name.replace("tert-", "").replace("sec-", "")
    return re.sub(r"[^a-z]", "", n.lower())


def _complex(name: str) -> bool:
    return bool(re.search(r"[\d(\[,]", name)) or " " in name


def _wrap(name: str) -> str:
    if not _complex(name):
        return name
    if "(" in name:
        return f"[{name}]" if "[" not in name else "{" + name + "}"
    return f"({name})"


def _locs(locs) -> str:
    return ",".join(str(x) for x in locs)


def format_prefixes(prefixes: list[tuple], omit_locants: bool = False) -> str:
    """prefixes: [(locant, name)] -> '2-chloro-3,3-dimethyl' (alphanumerical order)."""
    groups: dict[str, list] = {}
    for loc, nm in prefixes:
        groups.setdefault(nm, []).append(loc)

    def key(nm):
        return (_alpha_key(nm), nm)
    parts = []
    for nm in sorted(groups, key=key):
        locs = sorted(groups[nm], key=lambda x: (isinstance(x, str), str(x).zfill(4)
                                                  if not isinstance(x, str) else x))
        n = len(locs)
        cx = _complex(nm)
        mult = (MULT_COMPLEX if cx else MULT)[n] if n > 1 else ""
        body = mult + (_wrap(nm) if cx else nm)
        if omit_locants:
            parts.append(body)
        else:
            parts.append(f"{_locs(locs)}-{body}")
    if omit_locants:
        return "".join(parts)
    return "-".join(parts)


# --- substituents -----------------------------------------------------------------------------

def sub_name(ctx: Ctx, q: int, p: int) -> str:
    """Prefix name of the branch starting at atom q (attached to atom p)."""
    a = ctx.atom(q)
    s = a.GetSymbol()
    b = ctx.bond(p, q)
    bt = b.GetBondType()
    if bt == Chem.BondType.DOUBLE and not b.GetIsAromatic():
        if s == "O":
            return "oxo"
        if s == "C" and a.GetDegree() == 1:
            return "methylidene"
        raise NamerError("ylidene substituent")
    if bt == Chem.BondType.TRIPLE:
        raise NamerError("triple-bonded substituent")
    if s in HALO:
        return HALO[s]
    if s == "O":
        others = [n.GetIdx() for n in a.GetNeighbors() if n.GetIdx() != p]
        if not others:
            return "hydroxy"
        acyl = _acyl_name(ctx, others[0], q)
        if acyl:
            return acyl + "oxy"                       # ester seen from its alcohol side
        r = sub_name(ctx, others[0], q)
        if r in ALKOXY:
            return ALKOXY[r]
        m = re.match(r"^(.*\d-)?(.*?)(methyl|ethyl|propyl|butyl)$", r)
        if m and not r.endswith("en-1-yl") and "(" not in r:
            return r[:-2] + "oxy"
        return f"{_wrap(r)}oxy" if _complex(r) else f"{r}oxy"
    if s == "N":
        if q in ctx.p.nitro:
            return "nitro"
        others = [n.GetIdx() for n in a.GetNeighbors() if n.GetIdx() != p]
        if not others:
            return "amino"
        names_ = sorted(sub_name(ctx, o, q) for o in others)
        if len(names_) == 2 and names_[0] == names_[1]:
            return f"di{_wrap(names_[0]) if _complex(names_[0]) else names_[0]}amino"
        if len(names_) == 1:
            return f"{_wrap(names_[0])}amino" if _complex(names_[0]) else f"{names_[0]}amino"
        n1, n2 = sorted(names_, key=_alpha_key)
        return f"{n1}({n2})amino"
    # carbon
    if q in ctx.p.nitrile:
        return "cyano"
    if q in ctx.p.aldehyde and a.GetDegree() == 2:
        return "formyl"
    if q in ctx.p.acid:
        return "carboxy"
    if any(q == t[0] for t in ctx.p.ester + ctx.p.amide + ctx.p.acyl_halide) or \
            any(q in (t[0], t[2]) for t in ctx.p.anhydride):
        raise NamerError("acyl substituent")
    if q in ctx.ring_of:
        return _ring_sub(ctx, q, p)
    return _chain_sub(ctx, q, p)


def _acyl_name(ctx: Ctx, c: int, o: int) -> str:
    """'acetyl', 'benzoyl', 'formyl' for the carbonyl carbon c of an ester -O-C(=O)R (seen
    from the ester oxygen o), '' when c is no such carbon; other acyl groups: out of scope."""
    if not any(t[0] == c and t[1] == o for t in ctx.p.ester):
        return ""
    rest = [n for n in ctx.atom(c).GetNeighbors()
            if n.GetIdx() != o and not (n.GetSymbol() == "O" and
                                         ctx.bond(c, n.GetIdx()).GetBondType() ==
                                         Chem.BondType.DOUBLE)]
    if not rest:
        return "formyl"
    r = rest[0]
    if r.GetSymbol() == "C" and r.GetDegree() == 1:
        return "acetyl"
    if r.GetIdx() in ctx.ring_of and r.GetIsAromatic() and \
            all(ctx.atom(i).GetDegree() == 2 for i in ctx.rings[ctx.ring_of[r.GetIdx()]]
                if i != r.GetIdx()):
        return "benzoyl"
    raise NamerError("acyloxy group other than acetyloxy / benzoyloxy")


def _branches(ctx: Ctx, atoms: list[int], exclude: set) -> list[tuple]:
    """[(parent position atom, neighbour)] of branches leaving the given atoms."""
    aset = set(atoms)
    out = []
    for i in atoms:
        for n in ctx.atom(i).GetNeighbors():
            j = n.GetIdx()
            if j in aset or j in exclude:
                continue
            out.append((i, j))
    return out


def _chain_sub(ctx: Ctx, q: int, p: int) -> str:
    """Acyclic substituent starting at q: 'ethyl', '2-methylpropyl', 'prop-2-en-1-yl' ..."""
    paths = []

    def walk(path):
        paths.append(list(path))
        for n in ctx.atom(path[-1]).GetNeighbors():
            j = n.GetIdx()
            if j == p or j in path or j not in ctx.skeleton or j in ctx.ring_of:
                continue
            if j in ctx.p.aldehyde or j in ctx.p.acid:
                continue
            walk(path + [j])
    walk([q])
    best = None
    for path in paths:
        mult = [k + 1 for k in range(len(path) - 1) if ctx.order(path[k], path[k + 1]) > 1]
        enes = [k + 1 for k in range(len(path) - 1) if ctx.order(path[k], path[k + 1]) == 2]
        try:
            prefixes = [(path.index(i) + 1, sub_name(ctx, j, i))
                        for i, j in _branches(ctx, path, {p})]
        except NamerError:
            continue                        # e.g. a double bond leaving this shorter chain
        alpha = [loc for loc, _n in sorted(prefixes, key=lambda x: _alpha_key(x[1]))]
        key = (-len(path), -len(mult), -len(enes), mult, enes, -len(prefixes),
               sorted(x[0] for x in prefixes), alpha)
        if best is None or key < best[0]:
            best = (key, path, mult, enes, prefixes)
    if best is None:
        raise NamerError("substituent chain cannot be named")
    _key, path, _mult, enes, prefixes = best
    n = len(path)
    ynes = [k + 1 for k in range(n - 1) if ctx.order(path[k], path[k + 1]) == 3]
    stem = STEMS[n] if n < len(STEMS) else None
    if stem is None:
        raise NamerError("chain too long")
    if not enes and not ynes:
        base = stem + "yl"
    elif n == 2:
        base = "ethenyl" if enes else "ethynyl"
    else:
        base = stem + ("a" if len(enes) > 1 or len(ynes) > 1 else "")
        if enes:
            base += f"-{_locs(enes)}-{MULT[len(enes)]}en"
        if ynes:
            base += f"-{_locs(ynes)}-{MULT[len(ynes)]}yn"
        base += "-1-yl"
    name = (format_prefixes(prefixes) + base) if prefixes else base
    return RETAINED_SUBST.get(name, name)


def _ring_sub(ctx: Ctx, q: int, p: int) -> str:
    k = ctx.ring_of[q]
    ring = ctx.ordered_ring(k)
    n = len(ring)
    aromatic = ctx.atom(q).GetIsAromatic()
    i0 = ring.index(q)
    best = None
    for d in (1, -1):
        order = [ring[(i0 + d * t) % n] for t in range(n)]
        pos = {a: t + 1 for t, a in enumerate(order)}
        mult, enes = [], []
        if not aromatic:
            for t in range(n):
                o = ctx.order(order[t], order[(t + 1) % n])
                if o > 1:
                    loc = t + 1 if t < n - 1 else n
                    mult.append(loc)
                    if o == 2:
                        enes.append(loc)
        prefixes = [(pos[i], sub_name(ctx, j, i)) for i, j in _branches(ctx, order, {p})]
        alpha = [loc for loc, _n in sorted(prefixes, key=lambda x: _alpha_key(x[1]))]
        key = (sorted(mult), sorted(enes), sorted(x[0] for x in prefixes), alpha)
        if best is None or key < best[0]:
            best = (key, sorted(mult), sorted(enes), prefixes)
    _k, mult, enes, prefixes = best
    ynes = sorted(set(mult) - set(enes))
    if aromatic:
        base = "phenyl"
    else:
        stem = "cyclo" + STEMS[n]
        if not mult:
            base = stem + "yl"
        else:
            base = stem + ("a" if len(enes) > 1 or len(ynes) > 1 else "")
            if enes:
                base += f"-{_locs(enes)}-{MULT[len(enes)]}en"
            if ynes:
                base += f"-{_locs(ynes)}-{MULT[len(ynes)]}yn"
            base += "-1-yl"
    return (format_prefixes(prefixes) + base) if prefixes else base


# --- parent hydride selection and the full name ------------------------------------------------

@dataclass
class Candidate:
    atoms: list            # ring order or chain path
    ring: bool
    aromatic: bool = False


def _principal_kind(p: Perception) -> str | None:
    for kind in ("acid", "ketone", "alcohol", "amine"):
        if getattr(p, kind):
            return kind
    return None


def _principal_sites(ctx: Ctx, kind: str, atoms: list[int], ring: bool) -> list[tuple]:
    """[(parent atom, group atom)] of principal groups on this parent."""
    aset = set(atoms)
    p = ctx.p
    out = []
    if kind == "acid":
        for c in p.acid:
            if not ring and c in aset:
                out.append((c, c))
            elif ring:
                for n in ctx.atom(c).GetNeighbors():
                    if n.GetIdx() in aset:
                        out.append((n.GetIdx(), c))
    elif kind == "ketone":
        out = [(c, c) for c in p.ketone if c in aset]
    elif kind == "alcohol":
        out = [(c, o) for o, c in p.alcohol.items() if c in aset]
    elif kind == "amine":
        for nn, cs in p.amine.items():
            for c in cs:
                if c in aset:
                    out.append((c, nn))
    return out


def _candidates(ctx: Ctx) -> list[Candidate]:
    out = [Candidate(ctx.ordered_ring(k), True, ctx.atom(r[0]).GetIsAromatic())
           for k, r in enumerate(ctx.rings)]
    acyclic = [i for i in ctx.skeleton if i not in ctx.ring_of]
    aset = set(acyclic)
    adj = {i: [n.GetIdx() for n in ctx.atom(i).GetNeighbors() if n.GetIdx() in aset]
           for i in acyclic}
    seen = set()
    for s in acyclic:
        # every simple path from s (the acyclic part is a forest: paths are unique)
        stack = [(s, [s])]
        while stack:
            cur, path = stack.pop()
            key = (min(path[0], path[-1]), max(path[0], path[-1]))
            if key not in seen:
                seen.add(key)
                out.append(Candidate(path, False))
            for j in adj[cur]:
                if j not in path:
                    stack.append((j, path + [j]))
    return out


def _numberings(c: Candidate):
    n = len(c.atoms)
    if not c.ring:
        yield list(c.atoms)
        if n > 1:
            yield list(reversed(c.atoms))
        return
    for i0 in range(n):
        for d in (1, -1):
            yield [c.atoms[(i0 + d * t) % n] for t in range(n)]


@dataclass
class Numbered:
    order: list
    pos: dict
    principal: list
    mult: list
    enes: list
    ynes: list
    prefixes: list          # [(locant, name)]
    n_prefixes: list        # [("N", name)]
    key: tuple


def _evaluate(ctx: Ctx, c: Candidate, kind: str | None, order: list[int]) -> Numbered:
    pos = {a: t + 1 for t, a in enumerate(order)}
    n = len(order)
    sites = _principal_sites(ctx, kind, order, c.ring) if kind else []
    principal = sorted(pos[a] for a, _g in sites)
    mult, enes, ynes = [], [], []
    if not c.aromatic:
        pairs = [(t, t + 1) for t in range(n - 1)]
        if c.ring:
            pairs.append((n - 1, 0))
        for t, u in pairs:
            o = ctx.order(order[t], order[u])
            if o > 1:
                loc = min(t, u) + 1 if not (c.ring and u == 0) else n
                mult.append(loc)
                (enes if o == 2 else ynes).append(loc)
    # what belongs to the suffix groups (not prefixes)
    suffix_atoms: set = set()
    n_prefixes = []
    group_atoms = {g for _a, g in sites}
    for a, g in sites:
        if kind == "acid":
            for nb in ctx.atom(g).GetNeighbors():
                if nb.GetSymbol() == "O":
                    suffix_atoms.add(nb.GetIdx())
            if g != a:
                suffix_atoms.add(g)
        elif kind == "ketone":
            for b in ctx.atom(g).GetBonds():
                if _is_dbl_o(b, ctx.atom(g)):
                    suffix_atoms.add(b.GetOtherAtom(ctx.atom(g)).GetIdx())
        else:
            suffix_atoms.add(g)
    if kind == "amine":
        with_subs = 0
        for a, g in sites:
            others = [cc for cc in ctx.p.amine[g] if cc != a]
            if others:
                with_subs += 1
            for cc in others:
                n_prefixes.append(("N", sub_name(ctx, cc, g)))
        if with_subs > 1:
            raise NamerError("several substituted amino groups")
    prefixes = []
    for i, j in _branches(ctx, order, set()):
        if j in suffix_atoms or (j in group_atoms and kind == "acid" and j not in order):
            continue
        if kind == "acid" and i in ctx.p.acid and not c.ring and \
                ctx.atom(j).GetSymbol() == "O":
            continue                                # the acid's own O atoms
        prefixes.append((pos[i], sub_name(ctx, j, i)))
    alpha = [loc for loc, _n in sorted(prefixes + n_prefixes, key=lambda x: _alpha_key(x[1]))
             if loc != "N"]
    key = (principal, sorted(mult), sorted(enes), -len(prefixes),
           sorted(x[0] for x in prefixes), alpha)
    return Numbered(order, pos, principal, sorted(mult), sorted(enes), sorted(ynes),
                    prefixes, n_prefixes, key)


def _stem(n: int) -> str:
    if n >= len(STEMS):
        raise NamerError("chain too long")
    return STEMS[n]


def _unsat(stem: str, enes, ynes, vowel_suffix: bool) -> str:
    """'hexan', 'hex-2-en', 'hexa-2,4-dien', 'hex-1-en-3-yn' (no final e)."""
    if not enes and not ynes:
        return stem + "an"
    s = stem + ("a" if len(enes) > 1 or len(ynes) > 1 else "")
    if enes:
        s += f"-{_locs(enes)}-{MULT[len(enes)]}en"
    if ynes:
        s += f"-{_locs(ynes)}-{MULT[len(ynes)]}yn"
    return s


def _assemble(ctx: Ctx, c: Candidate, kind: str | None, nb: Numbered) -> str:
    n = len(nb.order)
    k = len(nb.principal)
    if kind and k == 0:
        raise NamerError("principal group not on the parent")
    if nb.n_prefixes and k > 1:
        raise NamerError("N-substituents on one of several amino groups")
    simple_parent = (not c.ring and n == 1)
    stereo = _stereo(ctx, c, nb)
    # parent + suffix
    if c.aromatic:
        if kind == "acid" and k == 1:
            parent = "benzoic acid"
        elif kind == "alcohol" and k == 1:
            parent = "phenol"
        elif kind == "amine" and k == 1:
            parent = "aniline"
        elif kind is None:
            parent = "benzene"
        else:
            suf = {"acid": "carboxylic acid", "alcohol": "ol", "amine": "amine"}[kind]
            parent = f"benzene-{_locs(nb.principal)}-{MULT[k]}{suf}"
    else:
        stem = ("cyclo" if c.ring else "") + _stem(n)
        base = _unsat(stem, nb.enes, nb.ynes, True)
        if kind is None:
            parent = base + "e"
        elif kind == "acid":
            if c.ring:
                parent = base + "e" + ("" if k == 1 else f"-{_locs(nb.principal)}-") + \
                    (MULT[k] if k > 1 else "") + "carboxylic acid"
            elif n == 1:
                parent = "formic acid"
            elif n == 2 and k == 1 and not nb.mult:
                parent = "acetic acid"
            else:
                parent = (base + "oic acid") if k == 1 else (base + "e" + MULT[k] + "oic acid")
        else:
            suf = {"ketone": "one", "alcohol": "ol", "amine": "amine"}[kind]
            if k == 1:
                parent = base + f"-{nb.principal[0]}-{suf}"
            else:
                parent = base + f"e-{_locs(nb.principal)}-{MULT[k]}{suf}"
    # locant omission for unambiguous parents
    all_prefix = nb.prefixes + nb.n_prefixes
    omit = False
    if simple_parent:
        omit = True
        parent = re.sub(r"-1-", "", parent)
    elif parent == "acetic acid":
        omit = True
    elif c.ring and not nb.mult and len(all_prefix) + k <= 1:
        omit = True
        parent = re.sub(r"-1-(ol|one|amine)$", r"\1", parent)
    elif c.aromatic and k == 0 and len(nb.prefixes) == 1:
        omit = True
    elif not c.ring and n == 2 and (k + len(nb.prefixes)) <= 1:
        omit = True
        parent = re.sub(r"-1-(ol|amine)$", r"\1", parent)
    if c.ring and not c.aromatic and k == 0 and not all_prefix and nb.mult == [1] and \
            not stereo:
        parent = re.sub(r"-1-(en|yn)e$", r"\1e", parent)
    if c.ring and not c.aromatic and k == 1 and kind != "acid" and len(all_prefix) == 0 \
            and not nb.mult:
        parent = re.sub(r"-1-(ol|one|amine)$", r"\1", parent)
    if stereo:
        omit = False if not simple_parent else omit
    if omit and parent.startswith(("ethan", "methan", "cyclo", "benz", "eth")):
        parent = re.sub(r"an-1-(ol|amine)$", r"an\1", parent)
    pref_numbered = [(loc, nm) for loc, nm in nb.prefixes]
    text = ""
    if all_prefix:
        text = format_prefixes(pref_numbered + nb.n_prefixes, omit_locants=omit and
                               not nb.n_prefixes)
        if omit and nb.n_prefixes:
            text = format_prefixes(nb.n_prefixes) + (
                ("-" if pref_numbered else "") + format_prefixes(pref_numbered, True)
                if pref_numbered else "")
    name = text + parent if text else parent
    if name and text and text[-1] == "-":
        name = text[:-1] + parent
    if stereo:
        name = f"({stereo})-{name}"
    return name


def _stereo(ctx: Ctx, c: Candidate, nb: Numbered) -> str:
    items = []
    for a in nb.order:
        lab = ctx.atom_labels.get(ctx.orig[a])
        if lab:
            items.append((nb.pos[a], f"{nb.pos[a]}{lab}"))
    n = len(nb.order)
    pairs = [(t, t + 1) for t in range(n - 1)] + ([(n - 1, 0)] if c.ring else [])
    for t, u in pairs:
        key = frozenset((ctx.orig[nb.order[t]], ctx.orig[nb.order[u]]))
        lab = ctx.bond_labels.get(key)
        if lab:
            loc = min(t, u) + 1 if not (c.ring and u == 0) else n
            items.append((loc, f"{loc}{lab}"))
    items.sort()
    return ",".join(s for _l, s in items)


def _name_core(mol, atom_labels, bond_labels) -> str:
    ctx = Ctx(mol, atom_labels, bond_labels)
    p = ctx.p
    kind = _principal_kind(p)
    if p.amide or p.acyl_halide or p.anhydride or (p.ester and kind != "acid"):
        raise NamerError("acid derivative left in the core")
    # esters below an acid: only as acyloxy prefixes; their acyl carbons are no parent
    acyl_c = set()
    for c, o, r in p.ester:
        if not _acyl_name(ctx, c, o):
            raise NamerError("ester below the principal group")
        acyl_c.add(c)
        acyl_c.update(n.GetIdx() for n in ctx.atom(c).GetNeighbors()
                      if n.GetSymbol() == "C" and n.GetDegree() == 1)
    ctx.skeleton -= acyl_c
    if p.aldehyde and kind != "acid":
        raise NamerError("aldehyde not handled as principal group")
    if not ctx.skeleton:
        raise NamerError("no carbon skeleton")
    best = None
    for c in _candidates(ctx):
        sites = len(_principal_sites(ctx, kind, c.atoms, c.ring)) if kind else 0
        nbest = None
        for order in _numberings(c):
            try:
                nb = _evaluate(ctx, c, kind, order)
            except NamerError:
                break                       # this parent cannot be named: try the others
            if nbest is None or nb.key < nbest.key:
                nbest = nb
        if nbest is None:
            continue
        mult = len(nbest.mult)
        dbl = len(nbest.enes)
        if c.ring:
            shape = (0, -len(c.atoms), -(3 if c.aromatic else mult))
        else:
            shape = (1, -len(c.atoms), -mult, -dbl)
        key = (-sites, shape, nbest.key)
        if best is None or key < best[0]:
            best = (key, c, nbest)
    if best is None:
        raise NamerError("no parent")
    _k, c, nb = best
    return _assemble(ctx, c, kind, nb)


# --- acid derivatives via the parent acid ------------------------------------------------------

_ACID_FORMS = {
    # derivative: (chain "oic acid" -> , "dioic acid" -> , "carboxylic acid" ->,
    #              "dicarboxylic acid" ->, retained {formic, acetic, benzoic})
    "amide": ("amide", "diamide", "carboxamide", "dicarboxamide",
              {"formic acid": "formamide", "acetic acid": "acetamide",
               "benzoic acid": "benzamide"}),
    "nitrile": ("enitrile", "dinitrile", "carbonitrile", "dicarbonitrile",
                {"acetic acid": "acetonitrile", "benzoic acid": "benzonitrile"}),
    "aldehyde": ("al", "dial", "carbaldehyde", "dicarbaldehyde",
                 {"formic acid": "formaldehyde", "acetic acid": "acetaldehyde",
                  "benzoic acid": "benzaldehyde"}),
    "ester": ("oate", "dioate", "carboxylate", "dicarboxylate",
              {"formic acid": "formate", "acetic acid": "acetate",
               "benzoic acid": "benzoate"}),
}


def _acid_to(name: str, form: str, halide: str = "") -> str:
    if "carboxy" in re.sub(r"carboxylic acid$", "", name):
        raise NamerError("a converted group ended up as a prefix")
    if form == "acyl_halide":
        table = ("oyl " + halide, "dioyl di" + halide, "carbonyl " + halide,
                 "dicarbonyl di" + halide,
                 {"formic acid": "formyl " + halide, "acetic acid": "acetyl " + halide,
                  "benzoic acid": "benzoyl " + halide})
    else:
        table = _ACID_FORMS[form]
    one, two, ring1, ring2, retained = table
    for old, new in retained.items():
        if name.endswith(old):
            return name[: -len(old)] + new
    if name.endswith("dicarboxylic acid"):
        return name[: -len("dicarboxylic acid")] + ring2
    if name.endswith("carboxylic acid"):
        return name[: -len("carboxylic acid")] + ring1
    if name.endswith("edioic acid"):
        return name[: -len("dioic acid")] + two
    if name.endswith("dioic acid"):
        return name[: -len("dioic acid")] + two
    if name.endswith("oic acid"):
        if form == "nitrile":
            return name[: -len("oic acid")] + "enitrile"
        return name[: -len("oic acid")] + one
    raise NamerError("acid name without an acid suffix")


def _surrogate(mol, p: Perception, kind: str):
    """(acid molecule with 'orig' atom props, extra N-prefix atoms {N orig: [branch atoms]})."""
    rw = Chem.RWMol(mol)
    for a in rw.GetAtoms():
        a.SetIntProp("orig", a.GetIdx())
    remove: set[int] = set()
    n_branches: list[tuple[int, int]] = []        # (N, carbon) branches named as N- prefixes
    if kind == "nitrile":
        for c in p.nitrile:
            n = [x.GetIdx() for x in rw.GetAtomWithIdx(c).GetNeighbors()
                 if x.GetSymbol() == "N"][0]
            rw.GetAtomWithIdx(n).SetAtomicNum(8)
            rw.GetBondBetweenAtoms(c, n).SetBondType(Chem.BondType.DOUBLE)
            o = rw.AddAtom(Chem.Atom(8))
            rw.AddBond(c, o, Chem.BondType.SINGLE)
    elif kind == "aldehyde":
        for c in p.aldehyde:
            o = rw.AddAtom(Chem.Atom(8))
            rw.AddBond(c, o, Chem.BondType.SINGLE)
            at = rw.GetAtomWithIdx(c)
            at.SetNumExplicitHs(0)
            at.SetNoImplicit(False)
    elif kind == "acyl_halide":
        for c, x in p.acyl_halide:
            rw.GetAtomWithIdx(x).SetAtomicNum(8)
    elif kind == "amide":
        for c, n in p.amide:
            for nb in rw.GetAtomWithIdx(n).GetNeighbors():
                if nb.GetIdx() != c:
                    n_branches.append((n, nb.GetIdx()))
                    rw.RemoveBond(n, nb.GetIdx())
            rw.GetAtomWithIdx(n).SetAtomicNum(8)
    elif kind == "ester":
        for c, o, r in p.ester:
            rw.RemoveBond(o, r)
    for a in rw.GetAtoms():
        a.SetNoImplicit(False)
        a.SetNumExplicitHs(0)
    m = rw.GetMol()
    Chem.SanitizeMol(m)
    return m, n_branches


def _largest_fragment_with(mol, orig_atom: int):
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    for f in frags:
        if any(a.GetIntProp("orig") == orig_atom for a in f.GetAtoms()):
            return f
    raise NamerError("fragment lost")


def candidate_name(mol) -> str:
    """Unverified rule-based name (raises :class:`NamerError` outside the scope)."""
    from rsstatchem.core.chem_draw import stereo
    if mol is None or mol.GetNumAtoms() == 0:
        raise NamerError("nothing drawn")
    mol = Chem.RemoveHs(Chem.Mol(mol))
    if len(Chem.GetMolFrags(mol)) != 1:
        raise NamerError("several molecules")
    a_lab, b_lab = stereo.mol_labels(mol)
    blab = {}
    for bi, lab in b_lab.items():
        b = mol.GetBondWithIdx(bi)
        blab[frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))] = lab
    p = perceive(mol)
    if p.anhydride:
        if len(p.anhydride) != 1 or p.ester or p.acid or p.amide or p.acyl_halide:
            raise NamerError("anhydride with other acyl groups")
        c1, o, c2 = p.anhydride[0]
        rw = Chem.RWMol(mol)
        for at in rw.GetAtoms():
            at.SetIntProp("orig", at.GetIdx())
        rw.RemoveBond(o, c2)
        o2 = rw.AddAtom(Chem.Atom(8))
        rw.AddBond(c2, o2, Chem.BondType.SINGLE)
        m = rw.GetMol()
        Chem.SanitizeMol(m)
        n1 = _name_core(_largest_fragment_with(m, c1), a_lab, blab)
        n2 = _name_core(_largest_fragment_with(m, c2), a_lab, blab)
        if not (n1.endswith(" acid") and n2.endswith(" acid")):
            raise NamerError("anhydride halves are not acids")
        s1, s2 = n1[:-5], n2[:-5]
        return f"{s1} anhydride" if s1 == s2 else f"{' '.join(sorted((s1, s2)))} anhydride"
    kind = None
    for k in ("acid", "ester", "acyl_halide", "amide", "nitrile", "aldehyde"):
        if (k == "acid" and p.acid) or (k != "acid" and getattr(p, k)):
            kind = k
            break
    if kind in (None, "acid"):
        return _name_core(mol, a_lab, blab)
    if kind in ("ester",) and (p.amide or p.acyl_halide):
        raise NamerError("ester with other acyl groups")
    if kind == "amide" and p.acyl_halide:
        raise NamerError("amide with acyl halide")
    if kind in ("nitrile", "aldehyde", "amide", "acyl_halide", "ester"):
        lower = {"ester": ("amide", "acyl_halide"), "acyl_halide": ("amide",),
                 "amide": (), "nitrile": (), "aldehyde": ()}[kind]
        if any(getattr(p, x) for x in lower):
            raise NamerError("mixed acid derivatives")
    if kind == "ester" and len(p.ester) > 1:
        ester_o = {o for _c, o, _r in p.ester}
        for c, o, r in p.ester:
            seen, stack = {o, r}, [r]
            while stack:
                cur = stack.pop()
                for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
                    j = nb.GetIdx()
                    if j not in seen:
                        if j in ester_o:
                            raise NamerError("ester groups inside an ester alkyl group")
                        seen.add(j)
                        stack.append(j)
    m, n_branches = _surrogate(mol, p, kind)
    if kind == "ester":
        alkyls = []
        for c, o, r in p.ester:
            ctx0 = Ctx(mol, a_lab, blab)
            alkyls.append(sub_name(ctx0, r, o))
        if len(set(alkyls)) != 1:
            raise NamerError("different ester alkyl groups")
        anchor = p.ester[0][0]
        acid_mol = _largest_fragment_with(m, anchor)
        acid = _name_core(acid_mol, a_lab, blab)
        k = len(p.ester)
        alk = alkyls[0]
        alk_txt = (MULT_COMPLEX[k] + _wrap(alk) if _complex(alk) else MULT[k] + alk) \
            if k > 1 else alk
        return f"{alk_txt} {_acid_to(acid, 'ester')}"
    if kind == "amide":
        if len(p.amide) > 1 and n_branches:
            raise NamerError("several substituted amides")
        ctx0 = Ctx(mol, a_lab, blab)
        n_prefixes = [("N", sub_name(ctx0, cc, n)) for n, cc in n_branches]
        if n_prefixes and re.search(r"amino|amido|imino", "".join(x[1] for x in n_prefixes)):
            raise NamerError("nitrogen inside an N-substituent")
        n_amine = sum(1 for nn in p.amine)
        acid_mol = _largest_fragment_with(m, p.amide[0][0])
        acid = _name_core(acid_mol, a_lab, blab)
        if n_prefixes and n_amine:
            raise NamerError("N-locant would be ambiguous (amino group in the acid part)")
        name = _acid_to(acid, "amide")
        if n_prefixes:
            name = _insert_n_prefixes(name, n_prefixes)
        return name
    if kind == "acyl_halide":
        halides = {mol.GetAtomWithIdx(x).GetSymbol() for _c, x in p.acyl_halide}
        if len(halides) != 1:
            raise NamerError("mixed acyl halides")
        acid = _name_core(m, a_lab, blab)
        return _acid_to(acid, "acyl_halide", HALIDE[halides.pop()])
    if kind == "nitrile":
        acid = _name_core(m, a_lab, blab)
        if acid == "formic acid":
            raise NamerError("hydrogen cyanide")
        return _acid_to(acid, "nitrile")
    acid = _name_core(m, a_lab, blab)
    return _acid_to(acid, "aldehyde")


def _insert_n_prefixes(name: str, n_prefixes: list[tuple]) -> str:
    """Put 'N-methyl' style prefixes in front (after a stereo descriptor)."""
    stereo = ""
    m = re.match(r"^(\([^)]*[RSEZ]\)-)", name)
    if m:
        stereo, name = m.group(1), name[len(m.group(1)):]
    pre = format_prefixes(n_prefixes)
    if name and name[0].isdigit():
        return stereo + pre + "-" + name
    return stereo + pre + name


# --- verification by OPSIN ----------------------------------------------------------------------

def _inchikey(mol) -> str:
    return Chem.MolToInchiKey(Chem.RemoveHs(Chem.Mol(mol)))


def verified_names(mols, opsin=None, timeout: float = 120.0) -> list[str | None]:
    """Names that OPSIN parses back to the same standard InChIKey, else None (one OPSIN run
    for all molecules). Raises ``names.NameLookupError`` when OPSIN cannot run."""
    from rsstatchem.core.chem_draw import names as nm
    opsin = opsin or nm.opsin_structures
    cands: list[str | None] = []
    for m in mols:
        try:
            cands.append(candidate_name(m))
        except (NamerError, Exception):  # noqa: BLE001 - outside the scope: no name
            cands.append(None)
    todo = [c for c in cands if c]
    if not todo:
        return [None] * len(cands)
    smiles = opsin(todo, timeout)
    back = dict(zip(todo, smiles))
    out: list[str | None] = []
    for m, c in zip(mols, cands):
        if not c or not back.get(c):
            out.append(None)
            continue
        parsed = Chem.MolFromSmiles(back[c])
        try:
            ok = parsed is not None and _inchikey(parsed) == _inchikey(m)
        except Exception:  # noqa: BLE001
            ok = False
        out.append(c if ok else None)
    return out


def verified_name(mol, opsin=None) -> str | None:
    return verified_names([mol], opsin)[0]

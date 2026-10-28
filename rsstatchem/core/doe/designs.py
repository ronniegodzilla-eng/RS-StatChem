"""Experimental design generation, mirroring Design-Expert's "New Design" builders.

All designs are built with numpy: two-level full and fractional factorials (generators,
defining relation, resolution, alias structure, blocking by confounding), general multilevel
factorials, Plackett-Burman, central composite (circumscribed / inscribed / face-centred with
rotatable, orthogonal, practical, spherical or custom alpha, blocks), Box-Behnken, 3-level
factorials, D-optimal designs (point exchange from a candidate set), mixture designs (simplex
lattice, simplex centroid, extreme vertices for constrained regions, D-optimal mixtures) and
Taguchi orthogonal arrays, minimum-run resolution V and IV designs for up to 50 factors
(fast columnwise-pairwise / single-entry exchange on the inverse model matrix), foldovers,
categoric factors in factorial / response surface / optimal designs and split-plot designs
with hard-to-change numeric or categoric factors (whole plots, restricted randomisation,
split-plot CCD and D-optimal designs, two-level split plots blocked by confounding whole-plot
interactions).

Optimal (custom) process, mixture and combined mixture-process designs use the engine of
:mod:`rsstatchem.core.doe.optimal` (I/D/A criteria; point, coordinate or best exchange;
blocks, forced runs, edited models) on the regions of :mod:`rsstatchem.core.doe.constraints`
(multiple linear constraints). Further builders: definitive screening designs from Paley
conference matrices, one-factor RSM, orthogonally blocked Box-Behnken designs, simplex
screening, user-defined point types, Latin hypercube and maximin-distance space-filling
designs, blank historical layouts and :func:`design_from_data` (Import Data Set).

:func:`build_design` turns a wizard specification into a :class:`Dataset` with columns
Std, Run, Block, Space Type, the factors in actual units and empty response columns; the design
definition is stored JSON-serialisable in ``ds.meta["doe"]``.
"""
from __future__ import annotations

import functools
import itertools
import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import linalg as sla
from scipy.linalg import LinAlgWarning

from rsstatchem.core.dataset import NOMINAL, ORDINAL, Dataset, Variable, sanitize_name

# Design-Expert skips the letter I (it would read as the identity / intercept).
LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ" + "abcdefghjklmnopqrstuvwxyz"

SPACE_TYPES = {0: "Center", 1: "Factorial", 2: "Axial", 3: "Edge", 4: "Vertex", 5: "Plane",
               6: "Interior", 7: "Centroid", 8: "Run", 9: "Third Edge", 10: "Data"}

DESIGN_TITLES = {
    "factorial": "Regular Two-Level Factorial", "general": "Multilevel Categoric (General "
    "Factorial)", "pb": "Plackett-Burman", "ccd": "Central Composite",
    "bbd": "Box-Behnken", "threelevel": "3-Level Factorial", "dopt": "Optimal (Custom)",
    "lattice": "Simplex Lattice", "centroid": "Simplex Centroid",
    "ev": "Extreme Vertices (constrained mixture)", "mixdopt": "Optimal (Custom) Mixture",
    "taguchi": "Taguchi Orthogonal Array", "mr5": "Min-Run Characterize (Resolution V)",
    "mr4": "Min-Run Screen (Resolution IV)",
    "dsd": "Definitive Screening", "onefactor": "One Factor (RSM)",
    "simscreen": "Simplex Screening", "combined": "Combined Mixture-Process (Optimal)",
    "lhs": "Space Filling: Latin Hypercube", "distance": "Space Filling: Optimal Distance",
    "userdef": "User-Defined", "historical": "Historical Data",
}
FACTORIAL_DESIGNS = ("factorial", "pb", "general", "mr5", "mr4", "dsd")
MIXTURE_DESIGNS = ("lattice", "centroid", "ev", "mixdopt", "simscreen")
COMBINED_DESIGNS = ("combined",)
CRITERIA = {"D": "D-optimal", "I": "I-optimal", "A": "A-optimal"}
EXCHANGES = {"point": "Point exchange", "coordinate": "Coordinate exchange",
             "best": "Best (half of the starts each)"}
CATEGORIC_DESIGNS = ("general", "taguchi")


def letter(i: int) -> str:
    return LETTERS[i] if i < len(LETTERS) else f"X{i + 1}"


def popcount(w: int) -> int:
    return bin(w).count("1")


def word_bits(w: int) -> list[int]:
    return [i for i in range(w.bit_length()) if w >> i & 1]


def word_str(w: int) -> str:
    return "".join(letter(i) for i in word_bits(w)) if w else "I"


def word_key(w: int) -> tuple:
    return popcount(w), word_bits(w)


# ---------------------------------------------------------------------------- factorials
def full_factorial(k: int) -> np.ndarray:
    """2^k design in standard (Yates) order: factor A alternates fastest."""
    idx = np.arange(2 ** k)
    return np.column_stack([np.where(idx >> j & 1, 1.0, -1.0) for j in range(k)]) \
        if k else np.zeros((1, 0))


def general_factorial(levels: list[int]) -> np.ndarray:
    """Level indices 0..L-1 of a full multilevel factorial, first factor changing fastest."""
    rows = itertools.product(*[range(n) for n in reversed(levels)])
    return np.array([list(reversed(r)) for r in rows], dtype=float).reshape(-1, len(levels))


# Minimum-aberration generators (Montgomery, Table 8.14), letters without I.
DEFAULT_GENERATORS = {
    (3, 1): "C=AB", (4, 1): "D=ABC", (5, 1): "E=ABCD", (5, 2): "D=AB, E=AC",
    (6, 1): "F=ABCDE", (6, 2): "E=ABC, F=BCD", (6, 3): "D=AB, E=AC, F=BC",
    (7, 1): "G=ABCDEF", (7, 2): "F=ABCD, G=ABDE", (7, 3): "E=ABC, F=BCD, G=ACD",
    (7, 4): "D=AB, E=AC, F=BC, G=ABC", (8, 1): "H=ABCDEFG", (8, 2): "G=ABCD, H=ABEF",
    (8, 3): "F=ABC, G=ABD, H=BCDE", (8, 4): "E=BCD, F=ACD, G=ABC, H=ABD",
    (9, 2): "H=ACDFG, J=BCEFG", (9, 3): "G=ABCD, H=ACEF, J=CDEF",
    (9, 4): "F=BCDE, G=ACDE, H=ABDE, J=ABCE", (9, 5): "E=ABC, F=BCD, G=ACD, H=ABD, J=ABCD",
    (10, 3): "H=ABCG, J=BCDE, K=ACDF", (10, 4): "G=BCDF, H=ACDF, J=ABDE, K=ABCE",
    (10, 5): "F=ABCD, G=ABCE, H=ABDE, J=ACDE, K=BCDE",
    (10, 6): "E=ABC, F=BCD, G=ACD, H=ABD, J=ABCD, K=AB",
    (11, 5): "G=CDE, H=ABCD, J=ABF, K=BDEF, L=ADEF",
    (11, 6): "F=ABC, G=BCD, H=CDE, J=ACD, K=ADE, L=BDE",
    (11, 7): "E=ABC, F=BCD, G=ACD, H=ABD, J=ABCD, K=AB, L=AC",
}


def default_generators(k: int, p: int) -> str:
    """Generators for a 2^(k-p) design: the textbook table, else a greedy search."""
    if p == 0:
        return ""
    if (k, p) in DEFAULT_GENERATORS:
        return DEFAULT_GENERATORS[(k, p)]
    base = k - p
    if base < 2:
        raise ValueError(f"A 2^({k}-{p}) design is not possible.")
    words = sorted((w for w in range(1, 2 ** base) if popcount(w) >= 2),
                   key=lambda w: (-popcount(w), word_bits(w)))
    chosen: list[tuple[int, int]] = []
    for j in range(p):
        best, best_score = None, None
        for w in words:
            if any(w == c for c, _ in chosen):
                continue
            trial = chosen + [(w, base + j)]
            group = defining_group([((1 << t) | m, 1) for m, t in trial])
            score = (min(popcount(g) for g, _ in group), -sum(1 for g, _ in group
                                                              if popcount(g) == 3))
            if best_score is None or score > best_score:
                best, best_score = w, score
        if best is None:
            raise ValueError(f"A 2^({k}-{p}) design is not possible.")
        chosen.append((best, base + j))
    return ", ".join(f"{letter(t)}={word_str(m)}" for m, t in chosen)


def parse_generators(text: str | list, k: int) -> list[tuple[int, int, int]]:
    """Parse "D=ABC, E=-ABD" into (generated factor index, base word mask, sign)."""
    parts = text if isinstance(text, list) else [s for s in str(text).replace(";", ",")
                                                 .split(",") if s.strip()]
    p = len(parts)
    base = k - p
    out = []
    for part in parts:
        if "=" not in part:
            raise ValueError(f"Generator '{part}' must look like D=ABC.")
        lhs, rhs = (s.strip().replace(" ", "").replace("*", "") for s in part.split("=", 1))
        sign = 1
        if rhs.startswith("-"):
            sign, rhs = -1, rhs[1:]
        rhs = rhs.lstrip("+")
        if lhs not in LETTERS or LETTERS.index(lhs) < base or LETTERS.index(lhs) >= k:
            raise ValueError(f"Generator '{part}': the left side must be one of the generated "
                             f"factors {', '.join(letter(i) for i in range(base, k))}.")
        mask = 0
        for ch in rhs:
            if ch not in LETTERS or LETTERS.index(ch) >= base:
                raise ValueError(f"Generator '{part}': '{ch}' is not a basic factor "
                                 f"({letter(0)}..{letter(base - 1)}).")
            mask ^= 1 << LETTERS.index(ch)
        if popcount(mask) < 2:
            raise ValueError(f"Generator '{part}' must use an interaction of basic factors.")
        out.append((LETTERS.index(lhs), mask, sign))
    if len({t for t, _, _ in out}) != p:
        raise ValueError("Each generated factor needs exactly one generator.")
    return sorted(out)


def defining_group(words: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """All non-identity words (with signs) of the defining relation spanned by ``words``."""
    group = {0: 1}
    for w, s in words:
        for g, sg in list(group.items()):
            group.setdefault(g ^ w, sg * s)
    return sorted(((w, s) for w, s in group.items() if w), key=lambda ws: word_key(ws[0]))


def resolution(group: list[tuple[int, int]]) -> int | None:
    return min(popcount(w) for w, _ in group) if group else None


ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII", 9: "IX",
         10: "X", 11: "XI"}


def fractional_factorial(k: int, generators) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """2^(k-p) design; returns the coded matrix and the defining relation (words, signs)."""
    gens = parse_generators(generators, k) if generators else []
    p = len(gens)
    base = full_factorial(k - p)
    cols = [base[:, j] for j in range(k - p)]
    for _t, mask, sign in gens:
        cols.append(sign * np.prod(base[:, word_bits(mask)], axis=1))
    group = defining_group([((1 << t) | mask, sign) for t, mask, sign in gens])
    return np.column_stack(cols), group


def alias_structure(k: int, group: list[tuple[int, int]], max_order: int = 3,
                    estimated_order: int = 2) -> list[str]:
    """Design-Expert style alias list: "[A] = A + BCD" for intercept, main effects, 2FI."""
    if not group:
        return []
    full = {0: 1}
    full.update(dict(group))
    lines, seen = [], set()

    def member(w, s):
        return ("Intercept" if w == 0 else word_str(w)), s

    terms = [0] + sorted((w for w in range(1, 2 ** k) if popcount(w) <= estimated_order),
                         key=word_key)
    for w in terms:
        if w in seen:
            continue
        cls = sorted(((w ^ g, s) for g, s in full.items()), key=lambda ws: word_key(ws[0]))
        seen.update(m for m, _ in cls)
        if cls[0][0] != w:
            continue
        shown = [member(m, s) for m, s in cls if popcount(m) <= max_order and m != w]
        text = f"[{member(w, 1)[0]}] = {member(w, 1)[0]}"
        for name, s in shown:
            text += f" {'+' if s > 0 else '-'} {name}"
        lines.append(text)
    return lines


def block_words(k: int, group: list[tuple[int, int]], n_blocks: int,
                n_base: int | None = None) -> list[int]:
    """Block generators for 2^b blocks: base interactions whose aliases have highest order."""
    b = int(round(math.log2(n_blocks))) if n_blocks > 1 else 0
    if b == 0:
        return []
    if 2 ** b != n_blocks:
        raise ValueError("The number of blocks of a two-level design must be a power of 2.")
    n_base = k - len({word_bits(w)[-1] for w, _ in group}) if n_base is None else n_base
    full = [0] + [w for w, _ in group]
    order = {w: min(popcount(w ^ g) for g in full) for w in range(1, 2 ** n_base)}
    cands = sorted(order, key=lambda w: (-order[w], word_bits(w)))[:28]
    best, best_score = None, None
    for combo in itertools.combinations(cands, b):
        span = {0}
        for w in combo:
            span |= {s ^ w for s in span}
        if len(span) != 2 ** b:
            continue
        score = tuple(sorted(order[s] for s in span if s))
        if best_score is None or score > best_score:
            best, best_score = combo, score
    if best is None or best_score[0] < 2:
        raise ValueError(f"{n_blocks} blocks would confound blocks with main effects.")
    return list(best)


def assign_blocks(coded: np.ndarray, words: list[int]) -> np.ndarray:
    blocks = np.ones(len(coded), dtype=int)
    for j, w in enumerate(words):
        sign = np.prod(coded[:, word_bits(w)], axis=1)
        blocks += (2 ** j) * (sign > 0)
    return blocks


# ---------------------------------------------------------------------------- Plackett-Burman
PB_GENERATORS = {
    8: "+++-+--",
    12: "++-+++---+-",
    20: "++--++++-+-+----++-",
    24: "+++++-+-++--++--+-+----",
}


def _hadamard(n: int) -> np.ndarray | None:
    """Normalised Hadamard matrix (first column +1) for the run sizes we can construct."""
    if n == 1:
        return np.ones((1, 1))
    if n in PB_GENERATORS and n != 8:
        g = np.array([1.0 if c == "+" else -1.0 for c in PB_GENERATORS[n]])
        rows = [np.roll(g, i) for i in range(n - 1)] + [-np.ones(n - 1)]
        return np.column_stack([np.ones(n), np.array(rows)])
    if n % 2 == 0:
        h = _hadamard(n // 2)
        if h is not None:
            return np.block([[h, h], [h, -h]])
    return None


def pb_runs(k: int, minimum: int = 8) -> int:
    n = max(4, minimum)
    while n < k + 1 or _hadamard(n) is None:
        n += 4
    return n


def plackett_burman(k: int) -> np.ndarray:
    """Plackett-Burman screening design for ``k`` factors (cyclic generators / Sylvester)."""
    n = pb_runs(k)
    if n in PB_GENERATORS:
        g = np.array([1.0 if c == "+" else -1.0 for c in PB_GENERATORS[n]])
        rows = [np.roll(g, i) for i in range(n - 1)] + [-np.ones(n - 1)]
        mat = np.array(rows)
    else:
        mat = _hadamard(n)[:, 1:]
        mat = mat[::-1]  # start like the cyclic designs, end with the all-minus run
    return mat[:, :k]


# ---------------------------------------------------------------------------- RSM designs
CCD_CENTERS = {1: 5, 2: 5, 3: 6, 4: 6, 5: 6, 6: 6, 7: 6, 8: 6, 9: 6, 10: 6}
CCD_FRACTION = {5: 1, 6: 1, 7: 1, 8: 2, 9: 2, 10: 3}  # resolution V fractions used by default


def ccd_alpha(alpha_type: str, k: int, nf: int, n_total: int, n_fact_center: int = 0,
              n_axial_center: int = 0, custom: float | None = None) -> float:
    """Axial distance: rotatable (nF^1/4), orthogonal quadratic, orthogonal blocking,
    practical (k^1/4), spherical (sqrt k), face-centred (1) or a custom value."""
    if alpha_type == "rotatable":
        return nf ** 0.25
    if alpha_type == "orthogonal":
        return (nf * (math.sqrt(n_total) - math.sqrt(nf)) ** 2 / 4) ** 0.25
    if alpha_type == "blocking":
        ns = 2 * k
        return math.sqrt(k * (1 + n_axial_center / ns) / (1 + n_fact_center / nf))
    if alpha_type == "practical":
        return k ** 0.25
    if alpha_type == "spherical":
        return math.sqrt(k)
    if alpha_type == "face":
        return 1.0
    if alpha_type == "custom":
        if not custom or custom <= 0:
            raise ValueError("Enter a positive custom alpha.")
        return float(custom)
    raise ValueError(f"Unknown alpha type {alpha_type!r}.")


def central_composite(k: int, alpha_type: str = "rotatable", ccd_type: str = "circumscribed",
                      center_points: int | None = None, blocks: int = 1,
                      center_axial: int = 2, generators: str = "",
                      custom_alpha: float | None = None) -> dict:
    """Central composite design. ``blocks`` = 1, or factorial blocks + 1 axial block."""
    if k < 2:
        raise ValueError("A central composite design needs at least 2 factors.")
    fact, group = fractional_factorial(k, generators) if generators else (full_factorial(k), [])
    nf = len(fact)
    n_fact_blocks = max(blocks - 1, 1) if blocks > 1 else 1
    c_f = CCD_CENTERS.get(k, 6) if center_points is None else int(center_points)
    if blocks > 1:
        if 2 ** int(round(math.log2(n_fact_blocks))) != n_fact_blocks:
            raise ValueError("Blocks must be 1, 2, 3, 5 or 9 (factorial blocks are a power "
                             "of 2, plus one axial block).")
        n_gen = len(parse_generators(generators, k)) if generators else 0
        words = block_words(k, group, n_fact_blocks, n_base=k - n_gen)
        fblocks = assign_blocks(fact, words)
        n_fact_center, n_ax_center = c_f * n_fact_blocks, int(center_axial)
    else:
        words, fblocks = [], np.ones(nf, dtype=int)
        n_fact_center, n_ax_center = c_f, 0
    n_total = nf + 2 * k + n_fact_center + n_ax_center
    if ccd_type == "face":
        alpha_type = "face"
    if alpha_type == "blocking" and blocks == 1:
        alpha_type = "orthogonal"
    alpha = ccd_alpha(alpha_type, k, nf, n_total, n_fact_center, n_ax_center, custom_alpha)
    axial = np.zeros((2 * k, k))
    for i in range(k):
        axial[2 * i, i], axial[2 * i + 1, i] = -alpha, alpha
    rows, blk, pts = [], [], []
    for b in range(1, n_fact_blocks + 1):
        sel = fact[fblocks == b]
        rows += [sel]
        blk += [b] * len(sel)
        pts += [1] * len(sel)
        if blocks > 1:
            rows.append(np.zeros((c_f, k)))
            blk += [b] * c_f
            pts += [0] * c_f
    ax_block = n_fact_blocks + 1 if blocks > 1 else 1
    rows.append(axial)
    blk += [ax_block] * (2 * k)
    pts += [2] * (2 * k)
    n_c = n_ax_center if blocks > 1 else c_f
    rows.append(np.zeros((n_c, k)))
    blk += [ax_block] * n_c
    pts += [0] * n_c
    coded = np.vstack(rows)
    if ccd_type == "inscribed":
        coded = coded / alpha
    return {"coded": coded, "blocks": np.array(blk), "types": np.array(pts),
            "alpha": float(alpha), "group": group, "block_words": words, "nf": nf}


BBD_SUBSETS = {
    6: [(0, 1, 3), (1, 2, 4), (2, 3, 5), (0, 3, 4), (1, 4, 5), (0, 2, 5)],
    7: [(3, 4, 5), (0, 5, 6), (1, 4, 6), (0, 1, 3), (2, 3, 6), (0, 2, 4), (1, 2, 5)],
}
BBD_CENTERS = {3: 5, 4: 5, 5: 6, 6: 6, 7: 6}


# orthogonal blocking of Box-Behnken designs: every factor appears equally often per block
BBD_BLOCKS = {
    4: [[(0, 1), (2, 3)], [(0, 3), (1, 2)], [(0, 2), (1, 3)]],          # 3 blocks of 9
    5: [[(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)],                          # 2 blocks of 23
        [(0, 2), (2, 4), (1, 4), (1, 3), (0, 3)]],
}
BBD_BLOCK_CENTERS = {4: 1, 5: 3}


def box_behnken(k: int, center_points: int | None = None, blocks: int = 1) -> dict:
    """Box-Behnken design for 3-7 factors (pairs; BIBD triples for 6 and 7 factors).
    ``blocks``: 3 for four factors or 2 for five (orthogonal blocks; ``center_points`` is
    then per block)."""
    if k < 3 or k > 7:
        raise ValueError("Box-Behnken designs are available for 3 to 7 factors.")
    if blocks > 1:
        plan = BBD_BLOCKS.get(k)
        if plan is None or len(plan) != blocks:
            raise ValueError("Blocked Box-Behnken designs: 4 factors in 3 blocks or 5 "
                             "factors in 2 blocks.")
        c = BBD_BLOCK_CENTERS[k] if center_points is None else int(center_points)
        rows, blk = [], []
        for b, pairs in enumerate(plan, start=1):
            for sub in pairs:
                f = full_factorial(2)
                block = np.zeros((4, k))
                block[:, list(sub)] = f
                rows.append(block)
                blk += [b] * 4
            rows.append(np.zeros((c, k)))
            blk += [b] * c
        coded = np.vstack(rows)
        types = np.where(np.all(coded == 0, axis=1), 0, 3)
        return {"coded": coded, "blocks": np.array(blk), "types": types}
    subsets = BBD_SUBSETS.get(k) or list(itertools.combinations(range(k), 2))
    rows = []
    for sub in subsets:
        f = full_factorial(len(sub))
        block = np.zeros((len(f), k))
        block[:, list(sub)] = f
        rows.append(block)
    c = BBD_CENTERS[k] if center_points is None else int(center_points)
    rows.append(np.zeros((c, k)))
    coded = np.vstack(rows)
    types = np.where(np.all(coded == 0, axis=1), 0, 3)
    return {"coded": coded, "blocks": np.ones(len(coded), dtype=int), "types": types}


def three_level(k: int, center_points: int = 0) -> dict:
    coded = general_factorial([3] * k) - 1
    if center_points:
        coded = np.vstack([coded, np.zeros((center_points, k))])
    allpm = np.all(np.abs(coded) == 1, axis=1)
    types = np.where(np.all(coded == 0, axis=1), 0, np.where(allpm, 1, 3))
    return {"coded": coded, "blocks": np.ones(len(coded), dtype=int), "types": types}


# ---------------------------------------------------------------------------- D-optimal
def d_optimal(F: np.ndarray, n: int, rng: np.random.Generator, starts: int = 8,
              max_iter: int = 200) -> np.ndarray:
    """Indices of ``n`` distinct candidate rows maximising det(X'X) (Fedorov point exchange
    with several random starts)."""
    N, p = F.shape
    if n > N:
        raise ValueError(f"Only {N} candidate points for {n} model points.")
    if n < p:
        raise ValueError(f"The model has {p} terms; at least {p} model points are needed.")
    ridge = 1e-8 * np.eye(p)
    best_idx, best_ld = None, -np.inf
    for _ in range(starts):
        idx = rng.choice(N, n, replace=False)
        for _it in range(max_iter):
            M = F[idx].T @ F[idx] + ridge
            Minv = np.linalg.pinv(M)
            C = F[idx] @ Minv @ F.T                       # n x N cross variances
            dd = np.einsum("ij,ij->i", F[idx] @ Minv, F[idx])
            dc = np.einsum("ij,ij->i", F @ Minv, F)
            delta = dc[None, :] - dd[:, None] - dd[:, None] * dc[None, :] + C ** 2
            delta[:, idx] = -np.inf
            i, j = np.unravel_index(np.argmax(delta), delta.shape)
            if delta[i, j] <= 1e-9:
                break
            idx = idx.copy()
            idx[i] = j
        sign, ld = np.linalg.slogdet(F[idx].T @ F[idx])
        ld = ld if sign > 0 else -np.inf
        if ld > best_ld + 1e-9:
            best_idx, best_ld = idx, ld
    if not np.isfinite(best_ld):
        raise ValueError("No non-singular design found for this model and candidate set.")
    return np.sort(best_idx)


def add_distance_points(cand: np.ndarray, chosen: np.ndarray, m: int) -> np.ndarray:
    """Lack-of-fit points: candidates furthest (max-min Euclidean distance) from the design."""
    chosen = list(chosen)
    for _ in range(m):
        d = np.min(np.linalg.norm(cand[:, None, :] - cand[chosen][None, :, :], axis=2), axis=1)
        d[chosen] = -1
        if d.max() <= 1e-9:
            break
        chosen.append(int(np.argmax(d)))
    return np.array(chosen)


def high_leverage(F: np.ndarray, idx: np.ndarray, m: int) -> np.ndarray:
    """Replicates: the ``m`` design points with the highest leverage."""
    X = F[idx]
    h = np.einsum("ij,ij->i", X @ np.linalg.pinv(X.T @ X), X)
    order = np.argsort(-h, kind="stable")[:m]
    return idx[order]


def process_candidates(k: int, model: str = "quadratic") -> np.ndarray:
    """Candidate set for process D-optimal designs: the 3^k grid (vertices, edge and face
    centres, centre), with axial check points added for many factors. Cubic models use a
    4-level grid (thirds of edges) plus the centre."""
    if model == "cubic" and k <= 6:
        grid = np.array(list(itertools.product((-1, -1 / 3, 1 / 3, 1), repeat=k)), float)
        return np.vstack([grid, np.zeros((1, k))])
    if k <= 6:
        return general_factorial([3] * k) - 1
    pts = [full_factorial(k), np.zeros((1, k))]
    for i in range(k):
        for s in (-1, 1):
            e = np.zeros(k)
            e[i] = s
            pts.append(e[None])
    for i, j in itertools.combinations(range(k), 2):
        for a, b in itertools.product((-1, 1), repeat=2):
            e = np.zeros(k)
            e[i], e[j] = a, b
            pts.append(e[None])
    return np.vstack(pts)


# ---------------------------------------------------------------------------- mixtures
def simplex_lattice(q: int, m: int) -> np.ndarray:
    pts = [c for c in itertools.product(range(m + 1), repeat=q) if sum(c) == m]
    pts.sort(key=lambda c: (sum(1 for v in c if v), [-v for v in c]))
    return np.array(pts, dtype=float) / m


def simplex_centroid(q: int) -> np.ndarray:
    pts = []
    for r in range(1, q + 1):
        for sub in itertools.combinations(range(q), r):
            x = np.zeros(q)
            x[list(sub)] = 1.0 / r
            pts.append(x)
    return np.array(pts)


def axial_blends(q: int) -> np.ndarray:
    """Axial check blends halfway between each vertex and the centroid."""
    pts = np.full((q, q), 1.0 / (2 * q))
    np.fill_diagonal(pts, (q + 1) / (2 * q))
    return pts


def mixture_type(x: np.ndarray) -> int:
    nz = int(np.sum(x > 1e-9))
    q = len(x)
    if nz == 1:
        return 4
    if nz == q and np.allclose(x, x.mean()):
        return 7
    if nz == 2 and q > 2:
        return 3
    return 5 if nz < q else 6


def extreme_vertices(lower, upper, total: float = 1.0, tol: float = 1e-9):
    """Vertices of {L <= x <= U, sum x = T} (McLean-Anderson) and their active bounds."""
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    q = len(lower)
    if lower.sum() > total + tol or upper.sum() < total - tol:
        raise ValueError("The component constraints are inconsistent with the mixture total.")
    verts: list[np.ndarray] = []
    for j in range(q):
        others = [i for i in range(q) if i != j]
        for combo in itertools.product((0, 1), repeat=q - 1):
            x = np.empty(q)
            for i, c in zip(others, combo):
                x[i] = upper[i] if c else lower[i]
            x[j] = total - x[others].sum()
            if lower[j] - tol <= x[j] <= upper[j] + tol:
                if not any(np.allclose(x, v, atol=1e-7) for v in verts):
                    verts.append(x)
    active = [frozenset([(i, 0) for i in range(q) if abs(v[i] - lower[i]) < 1e-7]
                        + [(i, 1) for i in range(q) if abs(v[i] - upper[i]) < 1e-7])
              for v in verts]
    return np.array(verts), active


def mixture_candidates(lower, upper, total: float = 1.0,
                       include=("vertex", "edge", "plane", "interior", "centroid")):
    """Extreme vertices, edge centres, constraint-plane centroids, axial check blends and the
    overall centroid of a constrained mixture region. Returns (points, space types)."""
    verts, active = extreme_vertices(lower, upper, total)
    q = verts.shape[1]
    pts, types = [], []

    def add(x, t):
        if not any(np.allclose(x, p, atol=1e-7) for p in pts):
            pts.append(x)
            types.append(t)

    if "vertex" in include:
        for v in verts:
            add(v, 4)
    if "edge" in include and q > 2:
        for a, b in itertools.combinations(range(len(verts)), 2):
            if len(active[a] & active[b]) >= q - 2:
                add((verts[a] + verts[b]) / 2, 3)
    centroid = verts.mean(axis=0)
    if "plane" in include and q > 3:
        for c in sorted(set().union(*active)):
            on = [v for v, act in zip(verts, active) if c in act]
            if len(on) >= 3:
                add(np.mean(on, axis=0), 5)
    if "interior" in include:
        for v in verts:
            add((v + centroid) / 2, 6)
    if "centroid" in include:
        add(centroid, 7)
    return np.array(pts), np.array(types)


# ---------------------------------------------------------------------------- Taguchi
TAGUCHI_ARRAYS = {"L4": (4, [2] * 3), "L8": (8, [2] * 7), "L9": (9, [3] * 4),
                  "L12": (12, [2] * 11), "L16": (16, [2] * 15), "L18": (18, [2] + [3] * 7),
                  "L27": (27, [3] * 13)}

_L12 = """11111111111 11111222222 11222111222 12122122112 12212212121 12221221211
21221122121 21212221112 21122212211 22211112212 22121211122 22112121221"""
_L18 = """11111111 11222222 11333333 12112233 12223311 12331122 13121323 13232131 13313212
21133221 21211332 21322113 22123132 22231213 22312321 23132312 23213123 23321231"""


def _two_level_array(n_bits: int) -> np.ndarray:
    rows = []
    for r in range(2 ** n_bits):
        bits = [(r >> (n_bits - 1 - t)) & 1 for t in range(n_bits)]
        row = []
        for j in range(1, 2 ** n_bits):
            par = sum(bits[t] for t in range(n_bits) if j >> t & 1) % 2
            row.append(par + 1)
        rows.append(row)
    return np.array(rows)


def taguchi_array(name: str) -> np.ndarray:
    """Standard Taguchi orthogonal array with levels coded 1, 2 (, 3)."""
    if name == "L4":
        return _two_level_array(2)
    if name == "L8":
        return _two_level_array(3)
    if name == "L16":
        return _two_level_array(4)
    if name == "L12":
        return np.array([[int(c) for c in r] for r in _L12.split()])
    if name == "L18":
        return np.array([[int(c) for c in r] for r in _L18.split()])
    if name == "L9":
        rows = [[a, b, (a + b) % 3, (2 * a + b) % 3] for a in range(3) for b in range(3)]
        return np.array(rows) + 1
    if name == "L27":
        rows = []
        for a, b, c in itertools.product(range(3), repeat=3):
            rows.append([a, b, (a + b) % 3, (2 * a + b) % 3, c, (a + c) % 3, (2 * a + c) % 3,
                         (b + c) % 3, (a + b + c) % 3, (2 * a + b + c) % 3, (2 * b + c) % 3,
                         (a + 2 * b + c) % 3, (2 * a + 2 * b + c) % 3])
        return np.array(rows) + 1
    raise ValueError(f"Unknown orthogonal array {name!r}.")


# ---------------------------------------------------------------------------- min-run designs
def two_fi_matrix(D: np.ndarray) -> np.ndarray:
    """Model matrix of the two-factor-interaction model (intercept, A..K, AB..JK)."""
    n, k = D.shape
    cols = [np.ones(n)] + [D[:, i] for i in range(k)]
    cols += [D[:, i] * D[:, j] for i, j in itertools.combinations(range(k), 2)]
    return np.column_stack(cols)


def mr5_runs(k: int) -> int:
    """Runs of a minimum-run equireplicated resolution V design: the number of 2FI model
    terms, 1 + k + k(k-1)/2, raised to the next even number (Stat-Ease, "MR5")."""
    p = 1 + k + k * (k - 1) // 2
    return p + (p % 2)


def mr4_runs(k: int, extra: bool = True) -> int:
    """Minimum-run resolution IV: 2k runs, plus 2 (the recommended "MR4+2")."""
    return 2 * k + (2 if extra else 0)


MR_MAX_FACTORS = 50  # Design-Expert screens and characterizes up to 50 factors


def _flip_support(k: int, extra: bool) -> list[np.ndarray]:
    """For each factor c, the columns of the square MR5 model matrix (intercept, A..K, the
    two-factor interactions and, when the run count is the number of terms + 1, the
    interaction of the last three factors) whose sign flips when c is switched in a run."""
    p = 1 + k + k * (k - 1) // 2
    sup = [[1 + c] for c in range(k)]
    for t, (i, j) in enumerate(itertools.combinations(range(k), 2)):
        sup[i].append(1 + k + t)
        sup[j].append(1 + k + t)
    if extra:
        for c in (k - 3, k - 2, k - 1):
            sup[c].append(p)
    return [np.array(s) for s in sup]


def _mr5_square(D: np.ndarray, extra: bool) -> np.ndarray:
    F = two_fi_matrix(D)
    if extra:
        F = np.column_stack([F, D[:, -3] * D[:, -2] * D[:, -1]])
    return F


@functools.lru_cache(maxsize=None)
def _res5_generators(k: int, n: int) -> tuple[int, tuple[int, ...]]:
    """Columns of a regular resolution V two-level fraction with at least ``n`` runs for ``k``
    factors: ``(r, columns)`` with the columns as bit masks h_c of GF(2)^r (run u of the 2^r
    sets factor c to (-1)^(u . h_c)).

    No four or fewer of the h_c may sum to zero (no defining word shorter than five): the set
    is the lexicographically first such set of GF(2)^r, for the smallest r with 2^r >= n that
    holds k of them (17 factors in 256 runs, 21 in 512, 29 in 1024, 38 in 2048, 52 in 4096);
    the 2FI model matrix of such a fraction has orthogonal columns."""
    r = max(1, (n - 1).bit_length())
    while True:
        size = 1 << r
        blocked = np.zeros(size, bool)   # sums of at most three chosen columns (and 0)
        blocked[0] = True
        chosen: list[int] = []
        pairs: list[int] = []
        for v in range(1, size):
            if blocked[v]:
                continue
            blocked[[v] + [v ^ a for a in chosen] + [v ^ s for s in pairs]] = True
            pairs += [v ^ a for a in chosen]
            chosen.append(v)
            if len(chosen) == k:
                return r, tuple(chosen)
        r += 1


def _parity(x: np.ndarray) -> np.ndarray:
    for shift in (16, 8, 4, 2, 1):
        x = x ^ (x >> shift)
    return x & 1


def _mr5_start(k: int, n: int, extra: bool, rng: np.random.Generator) -> np.ndarray:
    """A balanced starting design for the MR5 exchange: n random runs of a regular
    resolution V fraction (columns in random order and signs), then the fewest sign changes
    that balance every column. A random subset of an orthogonal design starts far closer to
    the optimum than random runs (D-efficiency ~0.43 instead of ~0.36 for 50 factors), so the
    exchange needs fewer passes and ends higher. With the extra run the interaction of the
    last three factors is a further model column; the column order keeps it unaliased with
    the two-factor interactions of the fraction."""
    bits, cols = _res5_generators(k, n)
    h = np.array(cols, dtype=np.int64)[rng.permutation(k)]
    if extra:
        pair_sums = {int(a ^ b) for a, b in itertools.combinations(h, 2)}
        if int(h[-3] ^ h[-2] ^ h[-1]) in pair_sums:
            for tri in itertools.combinations(range(k), 3):
                if int(h[tri[0]] ^ h[tri[1]] ^ h[tri[2]]) not in pair_sums:
                    rest = [c for c in range(k) if c not in tri]
                    h = h[rest + list(tri)]
                    break
    runs = rng.choice(1 << bits, n, replace=False)
    D = 1.0 - 2.0 * _parity(runs[:, None] & h[None, :]).astype(float)
    D *= rng.choice([-1.0, 1.0], size=k)
    s = D.sum(axis=0)
    for c in np.flatnonzero(s):
        rows = np.flatnonzero(D[:, c] == np.sign(s[c]))
        D[rng.choice(rows, int(abs(s[c])) // 2, replace=False), c] *= -1
    return D


def _mr5_exchange(D: np.ndarray, extra: bool, rng: np.random.Generator, max_pass: int,
                  candidates: int, tol: float, reps: int = 8, rank: int = 128) -> None:
    """Columnwise-pairwise exchange of the MR5 runs ``D`` (in place); see
    :func:`min_run_res5`."""
    n, k = D.shape
    sup = _flip_support(k, extra)
    dtype = np.float32 if n > 300 else np.float64   # halves the memory traffic of G
    accept = 1 + (1e-5 if dtype is np.float32 else 1e-9)
    eps = dtype(1e-3)
    F = np.asfortranarray(_mr5_square(D, extra), dtype=dtype)  # column blocks F[:, S]
    Fe = F + eps * np.eye(n, dtype=dtype)
    Ut = np.empty((rank, n), dtype)   # G = G0 + Ut[:used]' V[:used] (pending updates)
    V = np.empty((rank, n), dtype)
    used = 0
    refresh = 3 if dtype is np.float32 else 1
    probe = rng.choice(n, min(n, 8), replace=False)
    for _pass in range(max_pass):
        if _pass % refresh == 0:   # recompute G against the drift of the updates
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", LinAlgWarning)
                G0 = sla.inv(Fe, check_finite=False)
            # a singular start is too ill-conditioned for single precision
            resid = Fe[probe] @ G0
            resid[np.arange(len(probe)), probe] -= 1
            if np.abs(resid).max() > 1e-2:
                G0 = np.linalg.inv(Fe.astype(float)).astype(dtype)
            used = 0
        gain, swaps = 0.0, 0
        for c in rng.permutation(k):
            S = sup[c]
            if used + 2 * reps > rank:   # apply the pending rank-2 updates in one GEMM
                G0 += Ut[:used].T @ V[:used]
                used = 0
            GS = G0[S] + Ut[:used, S].T @ V[:used]   # the rows of G that column c needs
            Fc = F[:, S]
            col = D[:, c]
            for _rep in range(reps):
                w = 1 - 2 * np.einsum("ij,ji->i", Fc, GS)  # single-switch ratios
                plus = np.flatnonzero(col > 0)
                minus = np.flatnonzero(col < 0)
                m = min(candidates, len(plus))
                if m < len(plus):
                    P = plus[np.argpartition(-np.abs(w[plus]), m - 1)[:m]]
                    Q = minus[np.argpartition(-np.abs(w[minus]), m - 1)[:m]]
                else:
                    P, Q = plus, minus
                ratio = (np.outer(w[P], w[Q])
                         - 4 * (Fc[P] @ GS[:, Q]) * (Fc[Q] @ GS[:, P]).T)
                idx = int(np.argmax(np.abs(ratio)))
                r = abs(float(ratio.flat[idx]))
                if r <= accept:
                    break
                i, j = P[idx // m], Q[idx % m]
                VG = -2 * (Fc[[i, j]] @ GS)
                a, b, cc, d = 1 + VG[0, i], VG[0, j], VG[1, i], 1 + VG[1, j]
                det = a * d - b * cc
                Y = np.vstack([d * VG[0] - b * VG[1], a * VG[1] - cc * VG[0]]) / det
                # Woodbury: G -= G[:, [i, j]] Y, kept pending as a rank-2 term
                Gij = G0[:, [i, j]] + (V[:used, [i, j]].T @ Ut[:used]).T
                Ut[used:used + 2] = Gij.T
                V[used:used + 2] = -Y
                used += 2
                GS -= Gij[S] @ Y
                Fc[[i, j]] = -Fc[[i, j]]
                F[i, S], F[j, S] = Fc[i], Fc[j]
                Fe[i, S] = F[i, S] + eps * (S == i)
                Fe[j, S] = F[j, S] + eps * (S == j)
                col[i], col[j] = -1.0, 1.0
                swaps += 1
                gain += math.log(r)
        if not swaps or 2 * gain / n < tol:
            break


def min_run_res5(k: int, rng: np.random.Generator, starts: int | None = None,
                 max_pass: int = 40, candidates: int = 64, tol: float = 0.005) -> np.ndarray:
    """Minimum-run equireplicated resolution V design (MR5) for ``k`` two-level factors
    (6 to 50; Oehlert & Whitcomb 2002, Stat-Ease's "Min-Run Characterize").

    The run count is the number of two-factor-interaction model terms raised to an even
    number, so the model matrix F is square (with one extra column, the interaction of the
    last three factors, when a run was added). The runs maximise |det F| — det(F'F) of the
    2FI model itself in the square case and a lower bound of it otherwise — subject to
    equireplication, by a columnwise-pairwise exchange (Li & Wu 1997): within a column a +
    and a - are swapped, which keeps every factor balanced. All swaps of a column are scored
    at once from the inverse G = F^-1: switching factor c in run i multiplies det F by
    1 - 2 sum_s F[i,s] G[s,i] over the columns s that involve c, a swap of two runs by the
    corresponding 2 x 2 determinant, and the best swap updates G by a rank-2 Woodbury step.
    The updates are kept pending (G = G0 + U V) and applied to G0 in one matrix product per
    64 swaps, so a swap costs O(n k) instead of O(n^2).

    Starts alternate between n random runs of a regular resolution V fraction (balanced by
    the fewest sign changes, :func:`_mr5_start`) and random balanced runs; a singular start
    is handled through F + eps I. Passes stop when the D-efficiency gains less than ``tol``
    (relative) in a pass; the best start is kept and the result is deterministic for a given
    seed. 50 factors (1276 runs) take about half a second."""
    if k < 2:
        raise ValueError("Min-Run Characterize needs at least 2 factors.")
    if k > MR_MAX_FACTORS:
        raise ValueError(f"Min-Run Characterize is available for up to {MR_MAX_FACTORS} "
                         "factors.")
    n = mr5_runs(k)
    p = 1 + k + k * (k - 1) // 2
    extra = n > p
    if extra and k < 3:
        raise ValueError("Min-Run Characterize needs at least 3 factors here.")
    if starts is None:
        starts = 24 if n <= 64 else 6 if n <= 200 else 3 if n <= 500 else 1
    best, best_ld = None, -np.inf
    for start in range(starts):
        if start % 2 == 0:
            D = _mr5_start(k, n, extra, rng)
        else:
            D = np.column_stack([rng.permutation(np.repeat([1.0, -1.0], n // 2))
                                 for _ in range(k)])
        _mr5_exchange(D, extra, rng, max_pass, candidates, tol)
        F2 = two_fi_matrix(D)
        sign, ld = np.linalg.slogdet(F2.T @ F2)
        if sign > 0 and ld > best_ld + 1e-9:
            best, best_ld = D.copy(), ld
    if best is None:
        raise ValueError(f"No non-singular minimum-run resolution V design found for {k} "
                         "factors.")
    return best


def max_det_pm1(k: int, rng: np.random.Generator, starts: int = 6) -> np.ndarray:
    """A k x k matrix of +-1 with (locally) maximal |det|: a Hadamard matrix when one can be
    built, else greedy single-entry exchange — every switch scored at once as
    1 - 2 X[i,c] G[c,i] from G = X^-1 and applied by a Sherman-Morrison update."""
    h = _hadamard(k) if k % 4 == 0 else None
    if h is not None:
        return h.copy()
    eps = 1e-3
    best, best_ld = None, -np.inf
    for _ in range(starts):
        X = rng.choice([-1.0, 1.0], size=(k, k))
        Xe = X + eps * np.eye(k)
        G = np.linalg.inv(Xe)
        for flip in range(20 * k * k):
            if flip and flip % (4 * k) == 0:
                G = np.linalg.inv(Xe)  # refresh against round-off
            R = np.abs(1 - 2 * X * G.T)
            idx = int(np.argmax(R))
            if R.flat[idx] <= 1 + 1e-9:
                break
            i, c = divmod(idx, k)
            v = -2 * X[i, c]
            G -= np.outer(G[:, i], G[c, :]) * (v / (1 + v * G[c, i]))
            X[i, c] = -X[i, c]
            Xe[i, c] += v
        sign, ld = np.linalg.slogdet(X)
        if sign != 0 and ld > best_ld + 1e-9:
            best, best_ld = X.copy(), ld
    if best is None:
        raise ValueError(f"No non-singular base design found for {k} factors.")
    return best


def min_run_res4(k: int, rng: np.random.Generator, extra: bool = True) -> np.ndarray:
    """Minimum-run resolution IV design (MR4): the foldover of a k-run main-effects design
    with maximal determinant, so every main effect is clear of all two-factor interactions
    in 2k runs. ``extra`` adds a further fold pair (x, -x), chosen to maximise the
    determinant of the main-effects information ("MR4+2", the Design-Expert default that
    protects against a lost run); the pair keeps both the foldover property and balance."""
    if k < 3:
        raise ValueError("Min-Run Screen needs at least 3 factors.")
    if k > MR_MAX_FACTORS:
        raise ValueError(f"Min-Run Screen is available for up to {MR_MAX_FACTORS} factors.")
    X0 = max_det_pm1(k, rng)
    D = np.vstack([X0, -X0])
    if extra:
        Minv = np.linalg.inv(X0.T @ X0)
        if k <= 14:
            cand = full_factorial(k)
            score = np.einsum("ij,ij->i", cand @ Minv, cand)
            x = cand[int(np.argmax(score))]
        else:
            x = rng.choice([-1.0, 1.0], size=k)
            for _ in range(10):
                for c in range(k):
                    y = x.copy()
                    y[c] = -y[c]
                    if y @ Minv @ y > x @ Minv @ x + 1e-12:
                        x = y
        D = np.vstack([D, x, -x])
    return D


# ---------------------------------------------------------------------------- foldover
def foldover_runs(coded: np.ndarray, kind: str = "full", factor: int | None = None,
                  keep_factor: int | None = None, keep_level: float = 1.0) -> np.ndarray:
    """Foldover of a two-level design (coded +-1; centre points are left out).

    * full: every sign reversed — a resolution III fraction becomes resolution IV.
    * single: only ``factor`` reversed — that factor and its 2FIs are de-aliased.
    * semifold: the single-factor foldover of only the runs where ``keep_factor`` is at
      ``keep_level`` (half the runs, Mee & Peralta 2000)."""
    pts = coded[~np.all(np.abs(coded) < 1e-9, axis=1)]
    if kind == "full":
        return -pts
    if factor is None:
        raise ValueError("Choose the factor to fold on.")
    new = pts.copy()
    new[:, factor] = -new[:, factor]
    if kind == "single":
        return new
    if kind == "semifold":
        if keep_factor is None or keep_factor == factor:
            raise ValueError("Choose a different factor whose level the semifold keeps.")
        return new[np.sign(pts[:, keep_factor]) == np.sign(keep_level)]
    raise ValueError(f"Unknown foldover {kind!r}.")


def folded_group(group: list[tuple[int, int]], kind: str, factor: int | None = None
                 ) -> list[tuple[int, int]] | None:
    """Defining relation after a full or single-factor foldover (words whose sign the fold
    does not change survive); None for a semifold (an irregular fraction)."""
    if kind == "full":
        return [(w, s) for w, s in group if popcount(w) % 2 == 0]
    if kind == "single":
        return [(w, s) for w, s in group if not (w >> factor & 1)]
    return None


# ---------------------------------------------------------------------------- split plots
def _wp_split_words(coded: np.ndarray, members: list[np.ndarray], easy: list[int],
                    extra: int) -> list[int]:
    """``extra`` independent interactions of easy-to-change factors that split every
    hard-to-change group into 2^extra equal whole plots (highest order first)."""
    if extra <= 0:
        return []
    words = sorted((sum(1 << e for e in combo) for r in range(len(easy), 0, -1)
                    for combo in itertools.combinations(easy, r)),
                   key=lambda w: (-popcount(w), word_bits(w)))
    chosen: list[int] = []
    for w in words:
        trial = chosen + [w]
        span = {0}
        for t in trial:
            span |= {s ^ t for s in span}
        if len(span) != 2 ** len(trial):
            continue
        ok = True
        for idx in members:
            keys = [tuple(np.prod(coded[i, word_bits(t)]) > 0 for t in trial) for i in idx]
            counts = {}
            for kk in keys:
                counts[kk] = counts.get(kk, 0) + 1
            if len(counts) != 2 ** len(trial) or len(set(counts.values())) != 1:
                ok = False
                break
        if ok:
            chosen = trial
            if len(chosen) == extra:
                return chosen
    raise ValueError("The easy-to-change factors cannot split the whole plots that finely; "
                     "choose fewer groups.")


def assign_whole_plots(coded: np.ndarray, hard: list[int], groups_per_rep: int | None = None
                       ) -> tuple[np.ndarray, list[int]]:
    """Whole plots of a two-level / multilevel design: one per combination of the
    hard-to-change factors, optionally split further by easy-to-change interactions so
    that there are ``groups_per_rep`` groups (Design-Expert's "groups"). Returns (group id
    0.., the splitting words)."""
    keys = [tuple(np.round(r[hard], 8)) for r in coded]
    uniq = sorted(set(keys))
    members = [np.array([i for i, kk in enumerate(keys) if kk == u]) for u in uniq]
    base = len(uniq)
    words: list[int] = []
    if groups_per_rep and groups_per_rep > base:
        ratio = groups_per_rep / base
        extra = int(round(math.log2(ratio)))
        if 2 ** extra * base != groups_per_rep:
            raise ValueError(f"The number of groups must be {base} times a power of 2.")
        easy = [j for j in range(coded.shape[1]) if j not in hard]
        words = _wp_split_words(coded, members, easy, extra)
    gid = np.empty(len(coded), dtype=int)
    for gi, idx in enumerate(members):
        if words:
            sub = np.array([sum((np.prod(coded[i, word_bits(w)]) > 0) << b
                                for b, w in enumerate(words)) for i in idx])
            gid[idx] = gi * 2 ** len(words) + sub
        else:
            gid[idx] = gi
    return gid, words


def split_plot_block_words(k: int, group: list[tuple[int, int]], hard: list[int],
                           wp_words: list[int], n_blocks: int) -> list[int]:
    """Block generators of a split-plot two-level factorial in 2^b blocks.

    Blocks must hold whole plots intact, so they are confounded with whole-plot contrasts —
    words spanned by the hard-to-change factors and the easy-to-change interactions that split
    the whole plots, modulo the defining relation — chosen, as for completely randomised
    designs, so that the words confounded with blocks (and their aliases) have the highest
    order. Each word is reported by its shortest alias."""
    b = int(round(math.log2(n_blocks))) if n_blocks > 1 else 0
    if b == 0:
        return []
    if 2 ** b != n_blocks:
        raise ValueError("The number of blocks of a two-level design must be a power of 2.")
    full = [0] + [w for w, _ in group]
    basis = [1 << h for h in hard] + list(wp_words)
    span = {0}
    for w in basis:
        span |= {s ^ w for s in span}
    reps = {}
    for w in span - {0}:
        cls = [w ^ g for g in full]
        rep = min(cls, key=word_key)
        reps[rep] = min(popcount(c) for c in cls)
    if len(reps) + 1 < 2 ** b:
        raise ValueError(f"{n_blocks} blocks need more whole plots: there are only "
                         f"{len(reps) + 1} whole plots per replicate to place in blocks.")
    cands = sorted(reps, key=lambda w: (-reps[w], word_key(w)))[:28]
    best, best_score = None, None
    for combo in itertools.combinations(cands, b):
        sp = {0}
        for w in combo:
            sp |= {s ^ w for s in sp}
        if len(sp) != 2 ** b:
            continue
        score = tuple(sorted(min(popcount(s ^ g) for g in full) for s in sp if s))
        if best_score is None or score > best_score:
            best, best_score = combo, score
    if best is None or best_score[0] < 2:
        raise ValueError(f"{n_blocks} blocks would confound blocks with a main effect of this "
                         "split-plot design; use fewer blocks or more groups.")
    return list(best)


def split_plot_ccd(k: int, hard: list[int], alpha_type: str = "rotatable",
                   center_points: int = 1, custom_alpha: float | None = None,
                   generators: str = "") -> dict:
    """Split-plot central composite design (Vining, Kowalski & Montgomery 2005 layout):

    * factorial whole plots: each hard-to-change (HTC) factorial combination, holding the
      easy-to-change (ETC) factorial points;
    * HTC axial whole plots: each HTC factor at +-alpha (others centred), holding the ETC
      factorial points;
    * one centre whole plot with the ETC axial points and the centre runs.
    """
    easy = [j for j in range(k) if j not in hard]
    if not hard or not easy:
        raise ValueError("A split-plot CCD needs both hard- and easy-to-change factors.")
    h, s = len(hard), len(easy)
    nf = 2 ** k
    n_total = nf + 2 * k + center_points
    alpha = ccd_alpha(alpha_type, k, nf, n_total, custom=custom_alpha)
    ef = full_factorial(s)
    rows, groups, types = [], [], []
    g = 0
    for hrow in full_factorial(h):
        for erow in ef:
            x = np.zeros(k)
            x[hard], x[easy] = hrow, erow
            rows.append(x)
            groups.append(g)
            types.append(1)
        g += 1
    for i in range(h):
        for sgn in (-1, 1):
            for erow in ef:
                x = np.zeros(k)
                x[hard[i]] = sgn * alpha
                x[easy] = erow
                rows.append(x)
                groups.append(g)
                types.append(2)
            g += 1
    for i in range(s):
        for sgn in (-1, 1):
            x = np.zeros(k)
            x[easy[i]] = sgn * alpha
            rows.append(x)
            groups.append(g)
            types.append(2)
    for _ in range(center_points):
        rows.append(np.zeros(k))
        groups.append(g)
        types.append(0)
    return {"coded": np.array(rows), "groups": np.array(groups), "types": np.array(types),
            "alpha": float(alpha)}


def split_plot_ccd_categoric(k: int, hard: list[int], cat: list[int], nlev: list[int],
                             alpha_type: str = "rotatable", center_points: int = 1,
                             custom_alpha: float | None = None) -> dict:
    """Split-plot central composite design with categoric factors (``cat``; codes are level
    indices). The numeric factors form a split-plot CCD (:func:`split_plot_ccd`) when some of
    them are hard to change and others easy; when all numeric factors are hard to change,
    every CCD point is its own whole plot; when none is, the whole CCD is run within each
    whole plot. The whole plots are then repeated for every combination of the hard-to-change
    categoric levels (a new group each), and every run is repeated within its group for each
    combination of the easy-to-change categoric levels — the same crossing Design-Expert
    uses for response surface designs with categoric factors. (Stat-Ease offers categoric
    factors in split-plot designs through the optimal builder only.)"""
    num = [j for j in range(k) if j not in cat]
    kn = len(num)
    if kn < 2:
        raise ValueError("A split-plot central composite design needs at least 2 numeric "
                         "factors; use Optimal (custom) or Multilevel Categoric otherwise.")
    nhard = [num.index(j) for j in hard if j in num]
    cat_hard = [j for j in cat if j in hard]
    cat_easy = [j for j in cat if j not in hard]
    if nhard and len(nhard) < kn:
        res = split_plot_ccd(kn, nhard, alpha_type, center_points, custom_alpha)
        nc, ng, nt, alpha = res["coded"], res["groups"], res["types"], res["alpha"]
    else:
        if not cat_easy and nhard:
            raise ValueError("A split-plot design needs at least one easy-to-change factor.")
        res = central_composite(kn, alpha_type, "face" if alpha_type == "face"
                                else "circumscribed", center_points,
                                custom_alpha=custom_alpha)
        nc, nt, alpha = res["coded"], res["types"], res["alpha"]
        ng = np.arange(len(nc)) if nhard else np.zeros(len(nc), dtype=int)
    n_wp = int(ng.max()) + 1
    hard_combos = _level_combos([nlev[j] for j in cat_hard])
    easy_combos = _level_combos([nlev[j] for j in cat_easy])
    rows, groups, types = [], [], []
    for hi, hc in enumerate(hard_combos):
        for g in range(n_wp):
            for r in np.flatnonzero(ng == g):
                for ec in easy_combos:
                    x = np.zeros(k)
                    x[num] = nc[r]
                    x[cat_hard] = hc
                    x[cat_easy] = ec
                    rows.append(x)
                    groups.append(hi * n_wp + g)
                    types.append(int(nt[r]))
    # whole-plot error df of one replicate: groups - 1 - whole-plot fixed-effect df (the
    # hard categoric main effects and interactions, the hard numeric terms of a quadratic)
    n_groups = len(hard_combos) * n_wp
    wp_terms = len(hard_combos) - 1
    if nhard:
        h = len(nhard)
        wp_terms += len(hard_combos) * (2 * h + h * (h - 1) // 2)
    return {"coded": np.array(rows), "groups": np.array(groups), "types": np.array(types),
            "alpha": float(alpha), "whole_plot_df": n_groups - 1 - wp_terms}


def d_optimal_split_plot(levels: list[np.ndarray], hard: list[int], terms_matrix,
                         n_groups: int, group_size: int, rng: np.random.Generator,
                         eta: float = 1.0, starts: int = 4, max_pass: int = 30
                         ) -> np.ndarray:
    """D-optimal split-plot design by coordinate exchange (Jones & Goos 2007): maximise
    det(X' V^-1 X), V = I + eta Z Z', with the hard-to-change factors constant within each
    of ``n_groups`` whole plots of ``group_size`` runs. ``levels[j]`` are the candidate
    coded levels of factor j; ``terms_matrix(coded) -> X``."""
    k = len(levels)
    n = n_groups * group_size
    gid = np.repeat(np.arange(n_groups), group_size)
    c = eta / (1 + eta * group_size)

    def logdet(D):
        X = terms_matrix(D)
        S = np.add.reduceat(X, np.arange(0, n, group_size), axis=0)
        M = X.T @ X - c * S.T @ S
        sign, ld = np.linalg.slogdet(M)
        return ld if sign > 0 else -np.inf

    easy = [j for j in range(k) if j not in hard]
    best, best_ld = None, -np.inf
    for _ in range(starts):
        D = np.empty((n, k))
        for j in range(k):
            if j in hard:
                D[:, j] = np.repeat(rng.choice(levels[j], n_groups), group_size)
            else:
                D[:, j] = rng.choice(levels[j], n)
        cur = logdet(D)
        for _pass in range(max_pass):
            improved = False
            for g in range(n_groups):
                rows = gid == g
                for j in hard:
                    old = D[rows, j][0]
                    for v in levels[j]:
                        if v == old:
                            continue
                        D[rows, j] = v
                        ld = logdet(D)
                        if ld > cur + 1e-9:
                            cur, old, improved = ld, v, True
                        else:
                            D[rows, j] = old
                for i in np.flatnonzero(rows):
                    for j in easy:
                        old = D[i, j]
                        for v in levels[j]:
                            if v == old:
                                continue
                            D[i, j] = v
                            ld = logdet(D)
                            if ld > cur + 1e-9:
                                cur, old, improved = ld, v, True
                            else:
                                D[i, j] = old
            if not improved:
                break
        if cur > best_ld + 1e-9:
            best, best_ld = D.copy(), cur
    if best is None or not np.isfinite(best_ld):
        raise ValueError("No non-singular split-plot design found: add runs or groups.")
    return best


# ---------------------------------------------------------------------------- definitive screening
def _prime_power(q: int) -> tuple[int, int] | None:
    for p in range(2, q + 1):
        if q % p == 0:
            m, r = 0, q
            while r % p == 0:
                r //= p
                m += 1
            return (p, m) if r == 1 else None
    return None


def _gf_elements(p: int, m: int):
    """Elements of GF(p^m) as coefficient tuples with subtraction and the set of squares."""
    elems = list(itertools.product(range(p), repeat=m))
    if m == 1:
        sq = {((a * a) % p,) for (a,) in elems if a}
        return elems, sq
    # an irreducible monic polynomial of degree m: x^m = -(c_{m-1} x^{m-1} + ... + c_0)
    for coefs in itertools.product(range(p), repeat=m):
        if coefs[0] == 0:
            continue
        # irreducible for m <= 3 iff it has no root in GF(p)
        if m <= 3 and all((pow(x, m, p) + sum(c * pow(x, i, p) for i, c in enumerate(coefs)))
                          % p for x in range(p)):
            poly = coefs
            break
    else:
        raise ValueError("No irreducible polynomial found.")

    def mul(a, b):
        prod = [0] * (2 * m - 1)
        for i, ai in enumerate(a):
            for j, bj in enumerate(b):
                prod[i + j] = (prod[i + j] + ai * bj) % p
        for d in range(2 * m - 2, m - 1, -1):
            c = prod[d]
            if c:
                prod[d] = 0
                for i, pc in enumerate(poly):
                    prod[d - m + i] = (prod[d - m + i] - c * pc) % p
        return tuple(prod[:m])

    sq = {mul(a, a) for a in elems if any(a)}
    return elems, sq


def conference_matrix(n: int) -> np.ndarray:
    """Conference matrix of order n (C'C = (n-1) I, zero diagonal) by Paley's construction
    from GF(n-1); n - 1 must be a prime power."""
    q = n - 1
    pm = _prime_power(q)
    if pm is None:
        raise ValueError(f"No Paley conference matrix of order {n}.")
    p, m = pm
    elems, sq = _gf_elements(p, m)
    Q = np.zeros((q, q))
    for i, a in enumerate(elems):
        for j, b in enumerate(elems):
            if i == j:
                continue
            d = tuple((x - y) % p for x, y in zip(b, a))
            Q[i, j] = 1.0 if d in sq else -1.0
    C = np.zeros((n, n))
    C[0, 1:] = 1
    C[1:, 0] = 1 if q % 4 == 1 else -1
    C[1:, 1:] = Q
    if not np.allclose(C.T @ C, (n - 1) * np.eye(n)):
        raise ValueError(f"Conference matrix of order {n} failed its check.")
    return C


def conference_order(m: int) -> int:
    """Smallest even order >= m with a Paley conference matrix."""
    n = m + (m % 2)
    while _prime_power(n - 1) is None:
        n += 2
    return n


def definitive_screening(m: int, n_cat: int = 0, extra_centers: int = 0,
                         C: np.ndarray | None = None) -> dict:
    """Definitive screening design (Jones & Nachtsheim 2011): runs [C; -C; 0] from a
    conference matrix, the last ``n_cat`` columns two-level categoric (their zeros become
    +1 in C and -1 in -C, and two centre runs carry them at -1 and +1; Jones & Nachtsheim
    2013). Returns coded runs (categoric columns -1/+1) and space types."""
    if m < 2:
        raise ValueError("A definitive screening design needs at least 2 factors.")
    if C is None:
        C = conference_matrix(conference_order(m))
    C = np.asarray(C, float)[:, :m] if C.shape[1] >= m else None
    if C is None:
        raise ValueError("The conference matrix has fewer columns than factors.")
    C = C.copy()
    cat = list(range(m - n_cat, m))
    top, bottom = C.copy(), -C.copy()
    for j in cat:
        top[top[:, j] == 0, j] = 1
        bottom[bottom[:, j] == 0, j] = -1
    runs = [top, bottom]
    if cat:
        c1, c2 = np.zeros((1, m)), np.zeros((1, m))
        c1[0, cat], c2[0, cat] = 1, -1
        runs += [c1, c2]
        n_c = 2 + int(extra_centers)
        for i in range(int(extra_centers)):
            e = np.zeros((1, m))
            e[0, cat] = 1 if i % 2 == 0 else -1
            runs.append(e)
    else:
        runs.append(np.zeros((1 + int(extra_centers), m)))
        n_c = 1 + int(extra_centers)
    coded = np.vstack(runs)
    num = [j for j in range(m) if j not in cat]
    zero = np.all(coded[:, num] == 0, axis=1) if num else np.zeros(len(coded), bool)
    types = np.where(zero, 0, 3)
    return {"coded": coded, "types": types, "order": len(C), "centers": n_c}


# ---------------------------------------------------------------------------- small builders
def simplex_screening(q: int) -> tuple[np.ndarray, np.ndarray]:
    """Simplex screening design (Snee & Marquardt 1976) in pseudo components: vertices,
    interior axial check blends, end-effect blends (one component absent) and the overall
    centroid."""
    pts, types = [], []
    for i in range(q):
        e = np.zeros(q)
        e[i] = 1
        pts.append(e)
        types.append(4)
    for x in axial_blends(q):
        pts.append(x)
        types.append(6)
    if q >= 3:
        for i in range(q):
            e = np.full(q, 1.0 / (q - 1))
            e[i] = 0
            pts.append(e)
            types.append(5 if q > 3 else 3)
    pts.append(np.full(q, 1.0 / q))
    types.append(7)
    return np.array(pts), np.array(types)


def latin_hypercube(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Latin hypercube in coded units -1..1, optimised for centred discrepancy."""
    from scipy.stats import qmc

    seed = int(rng.integers(0, 2 ** 31 - 1))
    try:
        sampler = qmc.LatinHypercube(d=k, optimization="random-cd", seed=seed)
    except TypeError:  # very old scipy
        sampler = qmc.LatinHypercube(d=k, seed=seed)
    return 2 * sampler.random(n) - 1


def maximin_points(cand: np.ndarray, n: int, rng: np.random.Generator,
                   start: np.ndarray | None = None, passes: int = 20) -> np.ndarray:
    """``n`` candidate rows with a large minimum distance (greedy farthest point, then
    exchange; Johnson, Moore & Ylvisaker 1990)."""
    if n > len(cand):
        raise ValueError(f"Only {len(cand)} candidate points for {n} runs.")
    centre = cand.mean(axis=0)
    first = int(np.argmax(np.linalg.norm(cand - centre, axis=1)))
    idx = [first] if start is None else []
    base = np.zeros((0, cand.shape[1])) if start is None else np.atleast_2d(start)
    while len(idx) < n:
        ref = np.vstack([base, cand[idx]]) if idx or len(base) else cand[[first]]
        d = np.min(np.linalg.norm(cand[:, None, :] - ref[None, :, :], axis=2), axis=1)
        d[idx] = -1
        idx.append(int(np.argmax(d)))
    idx = np.array(idx)

    def score(sel):
        P = np.vstack([base, cand[sel]])
        D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
        D[np.diag_indices(len(P))] = np.inf
        return D.min(), np.sum(D == D.min())

    cur = score(idx)
    for _ in range(passes):
        improved = False
        for i in rng.permutation(len(idx)):
            for j in rng.permutation(len(cand))[:200]:
                if j in idx:
                    continue
                trial = idx.copy()
                trial[i] = j
                s = score(trial)
                if s[0] > cur[0] + 1e-12 or (abs(s[0] - cur[0]) < 1e-12 and s[1] < cur[1]):
                    idx, cur, improved = trial, s, True
        if not improved:
            break
    return idx


def axial_runs(k: int, alpha: float, factors: list[int] | None = None) -> np.ndarray:
    """2 x len(factors) axial (star) points at +-alpha for augmenting a factorial to a CCD."""
    factors = list(range(k)) if factors is None else list(factors)
    out = np.zeros((2 * len(factors), k))
    for i, j in enumerate(factors):
        out[2 * i, j], out[2 * i + 1, j] = -alpha, alpha
    return out


def classify_points(region, X: np.ndarray) -> np.ndarray:
    """Space types by the number of active bounds / constraints: vertex, edge, plane,
    interior, centre (process 0) or centroid (mixture)."""
    planes = region._hyperplanes()
    cont = region.cont
    d = region.dim
    out = np.empty(len(X), int)
    cen = region.centroid() if region.mix else None
    for r, x in enumerate(X):
        xc = x[cont]
        act = sum(1 for nrm, v, _ in planes if abs(nrm @ xc - v) < 1e-7)
        if not region.mix and np.allclose(xc, 0):
            out[r] = 0
        elif cen is not None and np.allclose(xc, cen, atol=1e-7):
            out[r] = 7
        elif act >= d:
            out[r] = 4
        elif act == d - 1 and d >= 1:
            out[r] = 3
        elif act >= 1:
            out[r] = 5
        else:
            out[r] = 6
    return out


def _parse_constraints(raw, k: int) -> list[dict]:
    from rsstatchem.core.doe import constraints as cons

    out = []
    for c in raw or []:
        if isinstance(c, str):
            if c.strip():
                out.append(cons.parse_constraint(c, k))
        elif isinstance(c, dict):
            d = dict(c)
            if "coefs" not in d and d.get("text"):
                d = cons.parse_constraint(d["text"], k)
            out.append({"coefs": [float(v) for v in d["coefs"]], "low": d.get("low"),
                        "high": d.get("high"),
                        "text": d.get("text") or cons.format_constraint(
                            d["coefs"], d.get("low"), d.get("high"))})
    return out


def _distance_rows(cand: np.ndarray, design: np.ndarray, m: int) -> np.ndarray:
    """Lack-of-fit rows: candidates furthest from the design (max-min distance)."""
    rows = []
    cur = np.asarray(design, float)
    for _ in range(m):
        d = np.min(np.linalg.norm(cand[:, None, :] - cur[None, :, :], axis=2), axis=1)
        if d.max() <= 1e-9:
            break
        x = cand[int(np.argmax(d))]
        rows.append(x)
        cur = np.vstack([cur, x])
    return np.array(rows) if rows else np.zeros((0, cand.shape[1]))


def _leverage_rows(F: np.ndarray, rows: np.ndarray, m: int) -> np.ndarray:
    h = np.einsum("ij,ij->i", F @ np.linalg.pinv(F.T @ F), F)
    order = np.argsort(-h, kind="stable")[:m]
    return rows[order]


def run_optimal(fmat, region, opt: dict, rng, candidates=None, forced=None,
                default_criterion: str = "D") -> dict:
    """Model points by the optimal engine, then lack-of-fit (distance) and replicate
    (leverage) points, optionally in blocks. Returns coded rows, blocks, info."""
    from rsstatchem.core.doe import optimal as optm

    crit = str(opt.get("criterion", default_criterion)).upper()
    method = opt.get("exchange", "point")
    nb = max(int(opt.get("blocks", 1) or 1), 1)
    p_terms = fmat(np.zeros((1, region.k))).shape[1]
    p = p_terms + (nb - 1)
    forced = np.zeros((0, region.k)) if forced is None else np.atleast_2d(forced)
    n_model = max(int(opt.get("model_points", 0) or 0), p - len(forced), 1)
    if candidates is None and method != "coordinate":
        candidates, _ = region.candidates(tuple(opt.get("candidate_types") or (
            "vertex", "edge", "thirds", "plane", "interior", "centroid")))
        if not region.constrained and not region.mix and len(region.cont) <= 6:
            grid = np.array(list(itertools.product(*[
                (np.arange(region.nlevels[j], dtype=float) if region.nlevels[j] else
                 np.array([-1.0, 0.0, 1.0])) for j in range(region.k)])), float)
            if opt.get("model") == "cubic":
                grid = np.array(list(itertools.product(*[
                    (np.arange(region.nlevels[j], dtype=float) if region.nlevels[j] else
                     np.array([-1.0, -1 / 3, 1 / 3, 1.0])) for j in range(region.k)])), float)
            candidates = np.unique(np.vstack([candidates, grid]).round(12), axis=0)
    sizes = None
    if nb > 1:
        base, extra = divmod(n_model, nb)
        sizes = [base + (1 if b < extra else 0) for b in range(nb)]
        if min(sizes) < 1:
            raise ValueError("More blocks than model points.")
    res = optm.optimal_design(fmat, n_model, region, candidates, crit, method,
                              starts=int(opt.get("starts", 0) or 0) or None, rng=rng,
                              forced=forced if len(forced) else None,
                              forced_blocks=np.ones(len(forced)) if len(forced) and sizes
                              else None,
                              block_sizes=sizes, block_start=0,
                              allow_repeats=(bool(opt.get("repeats", False))
                                             or candidates is None
                                             or n_model > len(candidates)),
                              n_grid=int(opt.get("grid", 11) or 11))
    rows = res.coded[len(forced):]
    blocks = res.blocks[len(forced):] if res.blocks is not None else np.ones(len(rows))
    pool = candidates if candidates is not None else region.candidates(
        ("vertex", "edge", "thirds", "plane", "interior", "centroid"))[0]
    lof = _distance_rows(pool, np.vstack([forced, rows]), int(opt.get("lof_points", 5)))
    F = fmat(rows)
    reps = _leverage_rows(F, rows, int(opt.get("replicate_points", 5))) \
        if int(opt.get("replicate_points", 5)) else np.zeros((0, region.k))
    extra_rows = np.vstack([lof, reps]) if len(lof) or len(reps) else np.zeros((0, region.k))
    counts = {b: int(np.sum(blocks == b)) for b in sorted(set(blocks.tolist()))}
    eb = []
    for _ in range(len(extra_rows)):  # balance the block sizes
        b = min(counts, key=lambda kk: (counts[kk], kk))
        eb.append(b)
        counts[b] += 1
    all_rows = np.vstack([rows, extra_rows])
    all_blocks = np.concatenate([blocks, eb]).astype(int) if nb > 1 \
        else np.ones(len(all_rows), int)
    Xall = fmat(all_rows)
    sign, ld = np.linalg.slogdet(Xall.T @ Xall)
    return {"coded": all_rows, "blocks": all_blocks, "criterion": crit, "exchange": method,
            "value": res.value, "log_det": float(ld) if sign > 0 else None,
            "model_points": int(len(rows)), "lof_points": int(len(lof)),
            "replicate_points": int(len(reps))}


def _terms_from_keys(keys) -> list:
    from rsstatchem.core.doe import model as mdl

    return mdl.sort_terms([mdl.key_term(k) if isinstance(k, str) else tuple(k)
                           for k in keys])


# ---------------------------------------------------------------------------- dataset builder
@dataclass
class DesignResult:
    dataset: Dataset
    meta: dict
    notes: list[str] = field(default_factory=list)


def _levels_from_text(values) -> tuple[list, list[str], bool]:
    """Categoric level values typed by the user -> (codes, labels, numeric?)."""
    labels = [str(v).strip() for v in values if str(v).strip()]
    try:
        nums = [float(v.replace(",", ".")) for v in labels]
        if len(set(nums)) == len(nums):
            return nums, labels, True
    except ValueError:
        pass
    return [float(i + 1) for i in range(len(labels))], labels, False


def _decimals(values: np.ndarray) -> int:
    values = values[np.isfinite(values)]
    for d in range(0, 5):
        if np.all(np.abs(np.round(values, d) - values) < 1e-9):
            return d
    return 3


def _unique_name(raw: str, used: set[str], fallback: str) -> str:
    base = sanitize_name(raw or fallback, fallback=fallback)
    name, n = base, 2
    while name.lower() in used:
        name = f"{base}_{n}"
        n += 1
    used.add(name.lower())
    return name


def _model_matrix_for_design(coded: np.ndarray, model: str, mixture: bool,
                             categoric: list[bool] | None = None,
                             nlevels: list[int] | None = None) -> np.ndarray:
    from rsstatchem.core.doe import model as mdl

    k = coded.shape[1]
    categoric = categoric or [False] * k
    terms = (mdl.mixture_terms(k, model) if mixture
             else mdl.process_terms(k, model, categoric))
    return mdl.raw_matrix(coded, terms, mixture=mixture, intercept=not mixture,
                          categoric=categoric if any(categoric) else None,
                          nlevels=nlevels)


def _level_combos(nlev: list[int]) -> np.ndarray:
    """All combinations of categoric level indices (first factor fastest)."""
    if not nlev:
        return np.zeros((1, 0))
    return general_factorial(nlev)


def _cross(num_coded: np.ndarray, num: list[int], cat: list[int], nlev: list[int], k: int,
           *arrays) -> tuple:
    """A numeric design repeated for every combination of the categoric factors' levels
    (Design-Expert's response surface designs with categoric factors)."""
    combos = _level_combos([nlev[j] for j in cat])
    rows = []
    for combo in combos:
        block = np.zeros((len(num_coded), k))
        block[:, num] = num_coded
        block[:, cat] = combo
        rows.append(block)
    out = [np.vstack(rows)]
    for a in arrays:
        out.append(np.concatenate([a] * len(combos)) if a is not None else None)
    return tuple(out)


def _index_two_level(coded: np.ndarray, cat: list[int]) -> np.ndarray:
    """Two-level categoric factors: coded -1 / +1 -> level index 0 / 1."""
    coded = coded.copy()
    for j in cat:
        coded[:, j] = (coded[:, j] > 0).astype(float)
    return coded


def _center_rows(k: int, cat: list[int], nlev: list[int], c: int) -> np.ndarray:
    """``c`` centre runs (numeric factors at 0) at every categoric level combination."""
    combos = _level_combos([nlev[j] for j in cat])
    rows = []
    for combo in combos:
        block = np.zeros((c, k))
        block[:, cat] = combo
        rows.append(block)
    return np.vstack(rows) if rows else np.zeros((0, k))


TWO_LEVEL_DESIGNS = ("factorial", "pb", "mr5", "mr4")
SPLIT_PLOT_DESIGNS = ("factorial", "general", "ccd", "dopt")


def build_design(spec: dict) -> DesignResult:
    """Create the design Dataset from a wizard specification.

    ``spec``: design (key of DESIGN_TITLES), factors [{name, units, type ("numeric" |
    "categoric"), low, high, levels (categoric level names), changes ("easy" | "hard")}],
    responses [{name, units}], options {...}, randomize, seed, name.

    Hard-to-change factors make a split-plot design: the runs come in groups (whole plots)
    within which the hard-to-change settings stay fixed; randomisation is restricted (the
    groups are randomised, then the runs within each group) and the design gets a Group
    column.
    """
    design = spec["design"]
    factors = [dict(f) for f in spec["factors"]]
    responses = [dict(r) for r in spec.get("responses", [])] or [{"name": "R1", "units": ""}]
    opt = dict(spec.get("options", {}))
    k = len(factors)
    rng = np.random.default_rng(spec.get("seed"))
    meta: dict = {"version": 1, "design": design, "title": DESIGN_TITLES[design],
                  "kind": "mixture" if design in MIXTURE_DESIGNS else "process"}
    notes: list[str] = []
    blocks = None
    types = None
    groups = None
    if k < 1:
        raise ValueError("Define at least one factor.")
    process = design not in MIXTURE_DESIGNS and design not in CATEGORIC_DESIGNS
    for f in factors:
        if process and f.get("type") == "categoric":
            codes, labels, numeric = _levels_from_text(f.get("levels") or ["L1", "L2"])
            if len(codes) < 2:
                raise ValueError(f"Categoric factor {f.get('name')} needs at least 2 levels.")
            f.update(type="categoric", levels=codes, labels=labels, numeric_levels=numeric)
        elif process and design == "combined" and f.get("type") == "mixture":
            f["type"] = "mixture"
        elif process:
            f["type"] = "numeric"
    cat = [j for j, f in enumerate(factors) if f.get("type") == "categoric"]
    num = [j for j in range(k) if j not in cat]
    nlev = [len(f["levels"]) if f.get("type") == "categoric" else 0 for f in factors]
    hard = [j for j, f in enumerate(factors) if f.get("changes") == "hard"]
    if design in MIXTURE_DESIGNS or design == "taguchi" or design not in SPLIT_PLOT_DESIGNS:
        if hard:
            raise ValueError("Hard-to-change factors (split-plot designs) are available for "
                             "Regular Two-Level, Multilevel Categoric, Central Composite and "
                             "Optimal (custom) designs.")
    if hard and len(hard) == k:
        raise ValueError("A split-plot design needs at least one easy-to-change factor.")
    if design in TWO_LEVEL_DESIGNS:
        bad = [factors[j].get("name", letter(j)) for j in cat if nlev[j] != 2]
        if bad:
            raise ValueError("Two-level designs need two-level categoric factors ("
                             + ", ".join(bad) + "); use Multilevel Categoric or Optimal "
                             "(custom) for more levels.")
    reps = int(opt.get("replicates", 1))
    block_reps = bool(opt.get("block_per_replicate", False)) and reps > 1

    if design == "factorial":
        p = int(opt.get("fraction", 0))
        gens = opt.get("generators") or default_generators(k, p)
        coded, group = fractional_factorial(k, gens) if p or gens else (full_factorial(k), [])
        p = len(parse_generators(gens, k)) if gens else 0
        nb = int(opt.get("blocks", 1))
        words = block_words(k, group, nb, n_base=k - p) if nb > 1 and not hard else []
        blocks = assign_blocks(coded, words)
        types = np.ones(len(coded), dtype=int)
        meta.update({"generators": [g.strip() for g in gens.split(",")] if gens else [],
                     "defining_relation": ["I"] + [("-" if s < 0 else "") + word_str(w)
                                                   for w, s in group],
                     "resolution": resolution(group), "fraction": p,
                     "aliases": alias_structure(k, group),
                     "block_generators": [word_str(w) for w in words]})
        if hard:
            gid, wp_words = assign_whole_plots(coded, hard, opt.get("groups"))
            if nb > 1:  # blocks confounded with whole-plot interactions
                words = split_plot_block_words(k, group, hard, wp_words, nb)
                blocks = assign_blocks(coded, words)
                meta["block_generators"] = [word_str(w) for w in words]
            order = np.lexsort((gid, blocks))
            coded, gid, types, blocks = coded[order], gid[order], types[order], blocks[order]
            meta["whole_plot_words"] = [word_str(w) for w in wp_words]
            groups = gid
        coded = _index_two_level(coded, cat)
        coded, blocks, types, groups = _replicate_groups(coded, blocks, types, groups, reps,
                                                         block_reps)
        c = int(opt.get("center_points", 0))
        if c:
            if hard and any(j in cat for j in hard):
                raise ValueError("Centre points need numeric hard-to-change factors.")
            nbk = int(blocks.max())
            for b in range(1, nbk + 1):
                rows = _center_rows(k, cat, nlev, c)
                coded = np.vstack([coded, rows])
                blocks = np.concatenate([blocks, np.full(len(rows), b)])
                types = np.concatenate([types, np.zeros(len(rows), dtype=int)])
                if groups is not None:
                    groups = np.concatenate([groups, np.full(len(rows), groups.max() + 1)])
            meta["center_points"] = c
    elif design == "pb":
        coded = _index_two_level(plackett_burman(k), cat)
        blocks, types = np.ones(len(coded), dtype=int), np.ones(len(coded), dtype=int)
        coded, blocks, types = _replicate(coded, blocks, types, reps)
        c = int(opt.get("center_points", 0))
        if c:
            rows = _center_rows(k, cat, nlev, c)
            coded = np.vstack([coded, rows])
            blocks = np.concatenate([blocks, np.ones(len(rows), dtype=int)])
            types = np.concatenate([types, np.zeros(len(rows), dtype=int)])
        meta["pb_runs"] = pb_runs(k)
    elif design in ("mr5", "mr4"):
        if design == "mr5":
            raw = min_run_res5(k, rng)
            meta.update({"resolution": 5, "min_run": "MR5",
                         "alias_note": "Minimum-run equireplicated resolution V: all main "
                                       "effects and two-factor interactions are estimable "
                                       "(not orthogonal; check the Evaluation)."})
        else:
            extra = bool(opt.get("extra_runs", True))
            raw = min_run_res4(k, rng, extra)
            meta.update({"resolution": 4, "min_run": "MR4+2" if extra else "MR4",
                         "alias_note": "Minimum-run resolution IV (foldover): main effects "
                                       "are clear of two-factor interactions; two-factor "
                                       "interactions are aliased with each other."})
        coded = _index_two_level(raw, cat)
        blocks, types = np.ones(len(coded), dtype=int), np.ones(len(coded), dtype=int)
        c = int(opt.get("center_points", 0))
        if c:
            rows = _center_rows(k, cat, nlev, c)
            coded = np.vstack([coded, rows])
            blocks = np.concatenate([blocks, np.ones(len(rows), dtype=int)])
            types = np.concatenate([types, np.zeros(len(rows), dtype=int)])
            meta["center_points"] = c
    elif design == "ccd" and hard:
        alpha_type = "face" if opt.get("ccd_type") == "face" else opt.get("alpha_type",
                                                                          "rotatable")
        res = split_plot_ccd_categoric(k, hard, cat, nlev, alpha_type,
                                       int(opt.get("center_points", 1) or 0), opt.get("alpha"))
        coded, groups, types = res["coded"], res["groups"], res["types"]
        blocks = np.ones(len(coded), dtype=int)
        coded, blocks, types, groups = _replicate_groups(coded, blocks, types, groups, reps,
                                                         block_reps)
        meta.update({"alpha": res["alpha"], "alpha_type": alpha_type,
                     "ccd_type": "split-plot", "factorial_points": 2 ** len(num)})
        if res["whole_plot_df"] < 1 and reps < 2:
            notes.append("Each hard-to-change categoric level is a single group, so the "
                         "whole-plot error has no degrees of freedom: replicate the design "
                         "(2 or more replicates) to test the hard-to-change factors.")
    elif design == "ccd":
        kn = len(num)
        res = central_composite(kn, opt.get("alpha_type", "rotatable"),
                                opt.get("ccd_type", "circumscribed"),
                                opt.get("center_points"), int(opt.get("blocks", 1)),
                                int(opt.get("center_axial", 2)), opt.get("generators", ""),
                                opt.get("alpha"))
        coded, blocks, types = _cross(res["coded"], num, cat, nlev, k, res["blocks"],
                                      res["types"])
        meta.update({"alpha": res["alpha"], "alpha_type": opt.get("alpha_type", "rotatable"),
                     "ccd_type": opt.get("ccd_type", "circumscribed"),
                     "generators": [g.strip() for g in opt.get("generators", "").split(",")
                                    if g.strip()],
                     "resolution": resolution(res["group"]),
                     "block_generators": [word_str(w) for w in res["block_words"]],
                     "factorial_points": res["nf"]})
        if opt.get("ccd_type") == "inscribed":
            notes.append("Inscribed CCD: the axial points are at the entered limits, the "
                         f"factorial points at ±{1 / res['alpha']:.4g} coded.")
        coded, blocks, types = _replicate(coded, blocks, types, reps)
    elif design in ("bbd", "threelevel"):
        kn = len(num)
        res = (box_behnken(kn, opt.get("center_points"), int(opt.get("blocks", 1) or 1))
               if design == "bbd" else three_level(kn, int(opt.get("center_points", 0))))
        coded, blocks, types = _cross(res["coded"], num, cat, nlev, k, res["blocks"],
                                      res["types"])
        coded, blocks, types = _replicate(coded, blocks, types, reps)
    elif design == "dopt":
        from rsstatchem.core.doe import constraints as cons
        from rsstatchem.core.doe import model as mdl

        model = opt.get("model", "quadratic")
        catflags = [j in cat for j in range(k)]
        kn = len(num)
        cons_list = _parse_constraints(opt.get("constraints"), k)
        custom = opt.get("terms")
        terms = _terms_from_keys(custom) if custom else mdl.process_terms(k, model, catflags)
        engine = (str(opt.get("criterion", "D")).upper() != "D"
                  or opt.get("exchange", "point") != "point" or bool(cons_list)
                  or bool(custom) or int(opt.get("blocks", 1) or 1) > 1
                  or bool(opt.get("forced")) or bool(opt.get("candidates")))

        def matrix(c):
            return mdl.raw_matrix(np.atleast_2d(c), terms, intercept=True,
                                  categoric=catflags if any(catflags) else None, nlevels=nlev)

        lows = [float(f.get("low", -1)) if j not in cat else 0.0
                for j, f in enumerate(factors)]
        highs = [float(f.get("high", 1)) if j not in cat else 1.0
                 for j, f in enumerate(factors)]
        if hard:
            numlev = (np.array([-1, -1 / 3, 1 / 3, 1.0]) if model == "cubic"
                      else np.array([-1.0, 0.0, 1.0]))
            levels = [np.arange(nlev[j], dtype=float) if j in cat else numlev
                      for j in range(k)]
            ng, gs = int(opt.get("groups", 8)), int(opt.get("group_size", 4))
            coded = d_optimal_split_plot(levels, hard, matrix, ng, gs, rng,
                                         eta=float(opt.get("variance_ratio", 1.0)),
                                         starts=int(opt.get("starts", 4)))
            groups = np.repeat(np.arange(ng), gs)
            meta.update({"model": model, "groups": ng, "group_size": gs})
            Fm = matrix(coded)
            blocks = np.ones(len(coded), dtype=int)
            nc = coded[:, num] if num else np.zeros((len(coded), 1))
            types = np.where(np.all(nc == 0, axis=1), 0,
                             np.where(np.all(np.abs(nc) == 1, axis=1), 4, 3))
        elif engine:
            region = cons.process_region(lows, highs, cons_list,
                                         [nlev[j] if j in cat else 0 for j in range(k)])
            sp_lo, sp_hi = np.array(lows), np.array(highs)
            centre, half = (sp_lo + sp_hi) / 2, np.where(sp_hi > sp_lo, (sp_hi - sp_lo) / 2, 1)

            def to_coded(rows):
                rows = np.atleast_2d(np.asarray(rows, float))
                out = (rows - centre) / half
                for j in cat:
                    lv = np.asarray(factors[j]["levels"], float)
                    out[:, j] = [int(np.argmin(np.abs(lv - v))) for v in rows[:, j]]
                return out

            forced = to_coded(opt["forced"]) if opt.get("forced") else None
            cands = to_coded(opt["candidates"]) if opt.get("candidates") else None
            if cands is not None:
                cands = cands[region.feasible(cands)]
                if not len(cands):
                    raise ValueError("None of the imported candidate points is feasible.")
            r = run_optimal(matrix, region, opt, rng, cands, forced, "D")
            coded, blocks = r["coded"], r["blocks"]
            if forced is not None:
                coded = np.vstack([forced, coded])
                blocks = np.concatenate([np.full(len(forced), 1), blocks]).astype(int)
            types = classify_points(region, coded)
            if forced is not None:
                types[:len(forced)] = 8
            Fm = matrix(coded)
            meta.update({"model": model if not custom else "custom",
                         "model_points": r["model_points"], "lof_points": r["lof_points"],
                         "replicate_points": r["replicate_points"],
                         "criterion": r["criterion"], "exchange": r["exchange"],
                         "criterion_value": r["value"], "forced_runs": 0 if forced is None
                         else int(len(forced))})
            if custom:
                meta["terms"] = [mdl.term_key(t) for t in terms]
            if cons_list:
                meta["constraints"] = cons_list
        else:
            cand_num = process_candidates(kn, model) if kn else np.zeros((1, 0))
            cand, = _cross(cand_num, num, cat, nlev, k)
            F = matrix(cand)
            p = F.shape[1]
            n_model = max(int(opt.get("model_points", p)), p)
            idx = d_optimal(F, min(n_model, len(cand)), rng, starts=int(opt.get("starts", 8)))
            idx = add_distance_points(cand, idx, int(opt.get("lof_points", 5)))
            reps_idx = high_leverage(F, idx, int(opt.get("replicate_points", 5)))
            all_idx = np.concatenate([idx, reps_idx])
            coded = cand[all_idx]
            Fm = F[all_idx]
            meta.update({"model": model, "model_points": int(len(idx) - int(opt.get(
                "lof_points", 5))), "lof_points": int(opt.get("lof_points", 5)),
                "replicate_points": int(len(reps_idx)), "criterion": "D",
                "exchange": "point"})
            blocks = np.ones(len(coded), dtype=int)
            nc = coded[:, num] if num else np.zeros((len(coded), 1))
            types = np.where(np.all(nc == 0, axis=1), 0,
                             np.where(np.all(np.abs(nc) == 1, axis=1), 4, 3))
        sign, ld = np.linalg.slogdet(Fm.T @ Fm)
        meta["log_det"] = float(ld) if sign > 0 else None
        if groups is not None:
            coded, blocks, types, groups = _replicate_groups(coded, blocks, types, groups,
                                                             reps, block_reps)
    elif design == "combined":
        coded, blocks, types, mixmeta = _combined_design(factors, opt, rng, cat, nlev)
        meta["mixture"] = mixmeta
        meta["kind"] = "combined"
        meta.update({k2: mixmeta.pop(k2) for k2 in ("criterion", "exchange", "model_points",
                                                    "lof_points", "replicate_points",
                                                    "log_det", "criterion_value")
                     if k2 in mixmeta})
        meta["model"] = f"{mixmeta['model']} x {mixmeta['process_model']}"
        if mixmeta.get("terms"):
            meta["terms"] = mixmeta.pop("terms")
        if mixmeta.get("constraints"):
            meta["constraints"] = mixmeta.pop("constraints")
    elif design == "dsd":
        bad = [factors[j].get("name", letter(j)) for j in cat if nlev[j] != 2]
        if bad:
            raise ValueError("Definitive screening designs take two-level categoric factors "
                             "only (" + ", ".join(bad) + ").")
        fake = int(opt.get("fake_factors", 0) or 0)
        order = num + [k + i for i in range(fake)] + cat
        res = definitive_screening(k + fake, len(cat), int(opt.get("center_points", 0) or 0))
        built = res["coded"]
        coded = np.zeros((len(built), k))
        for pos, j in enumerate(order):
            if j < k:
                coded[:, j] = built[:, pos]
        coded = _index_two_level(coded, cat)
        blocks = np.ones(len(coded), dtype=int)
        types = res["types"]
        meta.update({"conference_order": int(res["order"]), "fake_factors": fake,
                     "center_points": int(res["centers"]),
                     "alias_note": "Definitive screening: main effects are orthogonal to each "
                                   "other and to every two-factor interaction and quadratic "
                                   "effect; quadratic effects are estimable; two-factor "
                                   "interactions are partially aliased with each other."})
        coded, blocks, types = _replicate(coded, blocks, types, reps)
    elif design == "onefactor":
        from rsstatchem.core.doe import constraints as cons
        from rsstatchem.core.doe import model as mdl

        if k != 1 or cat:
            raise ValueError("One-factor RSM designs have exactly one numeric factor.")
        model = opt.get("model", "quadratic")
        terms = mdl.process_terms(1, model)
        o = dict(opt)
        o.setdefault("lof_points", 2)
        o.setdefault("replicate_points", 3)
        region = cons.process_region([float(factors[0].get("low", -1))],
                                     [float(factors[0].get("high", 1))])
        grid = np.linspace(-1, 1, 21)[:, None]

        def matrix(c):
            return mdl.raw_matrix(np.atleast_2d(c), terms)

        r = run_optimal(matrix, region, o, rng, grid, None, "I")
        coded, blocks = r["coded"], np.ones(len(r["coded"]), int)
        types = classify_points(region, coded)
        meta.update({"model": model, "criterion": r["criterion"],
                     "model_points": r["model_points"], "lof_points": r["lof_points"],
                     "replicate_points": r["replicate_points"]})
    elif design in ("lhs", "distance", "userdef"):
        coded, types = _space_filling(design, factors, opt, rng, cat, nlev, meta)
        blocks = np.ones(len(coded), dtype=int)
        coded, blocks, types = _replicate(coded, blocks, types, reps)
    elif design == "historical":
        n_rows = int(opt.get("runs", 10) or 10)
        coded = np.full((n_rows, k), np.nan)
        blocks = np.ones(n_rows, dtype=int)
        types = np.full(n_rows, 10)
        meta["editable_factors"] = True
        if opt.get("kind") == "mixture":
            total = float(opt.get("total", 1.0))
            meta["kind"] = "mixture"
            meta["mixture"] = {"total": total,
                               "lower": [float(f.get("low", 0) or 0) for f in factors],
                               "upper": [float(f.get("high", total)) for f in factors],
                               "model": "quadratic", "degree": 2,
                               "pseudo": "L_Pseudo"}
            for f in factors:
                f["type"] = "mixture"
    elif design in MIXTURE_DESIGNS:
        coded, types, mix = _mixture_design(design, factors, opt, rng)
        blocks = np.asarray(mix.pop("_blocks"), int) if "_blocks" in mix \
            else np.ones(len(coded), dtype=int)
        for key in ("criterion", "exchange", "criterion_value", "model_points",
                    "lof_points", "replicate_points", "terms"):
            if key in mix:
                meta[key] = mix.pop(key)
        if mix.get("constraints"):
            meta["constraints"] = mix.pop("constraints")
        meta["mixture"] = mix
        notes += mix.pop("notes", [])
    elif design == "general":
        levels = []
        for f in factors:
            vals = f.get("levels") or ["1", "2"]
            codes, labels, numeric = _levels_from_text(vals)
            if len(codes) < 2:
                raise ValueError(f"Factor {f['name']} needs at least 2 levels.")
            f.update(type="categoric", levels=codes, labels=labels, numeric_levels=numeric)
            levels.append(len(codes))
        if hard:
            easy = [j for j in range(k) if j not in hard]
            wp = general_factorial([levels[j] for j in hard])
            sp = general_factorial([levels[j] for j in easy])
            rows, gid = [], []
            for g, w in enumerate(wp):
                block = np.zeros((len(sp), k))
                block[:, hard] = w
                block[:, easy] = sp
                rows.append(block)
                gid += [g] * len(sp)
            coded, groups = np.vstack(rows), np.array(gid)
        else:
            coded = general_factorial(levels)
        blocks, types = np.ones(len(coded), dtype=int), np.full(len(coded), 1)
        coded, blocks, types, groups = _replicate_groups(coded, blocks, types, groups, reps,
                                                         block_reps)
    elif design == "taguchi":
        name = opt.get("array", "L9")
        arr = taguchi_array(name)
        n_runs, col_levels = TAGUCHI_ARRAYS[name]
        if k > len(col_levels):
            raise ValueError(f"{name} has only {len(col_levels)} columns.")
        for j, f in enumerate(factors):
            nl = col_levels[j]
            vals = f.get("levels") or [str(i + 1) for i in range(nl)]
            codes, labels, numeric = _levels_from_text(vals)
            if len(codes) != nl:
                raise ValueError(f"Column {j + 1} of {name} has {nl} levels; factor "
                                 f"{f['name']} has {len(codes)}.")
            f.update(type="categoric", levels=codes, labels=labels, numeric_levels=numeric)
        coded = arr[:, :k].astype(float) - 1
        blocks, types = np.ones(n_runs, dtype=int), np.full(n_runs, 8)
        meta["taguchi"] = name
        coded, blocks, types = _replicate(coded, blocks, types, reps)
    else:
        raise ValueError(f"Unknown design {design!r}.")

    n = len(coded)
    std = np.arange(1, n + 1)
    randomize = spec.get("randomize", True) and design != "historical"
    run = randomized_run_order(blocks, groups, rng) if randomize else \
        _standard_run_order(blocks, groups)

    used = {"std", "run", "block", "space_type", "group"}
    cols: dict[str, np.ndarray] = {"Std": std.astype(float), "Run": run.astype(float),
                                   "Block": blocks.astype(float)}
    variables = [Variable("Std", decimals=0, label="Standard order", measure=ORDINAL),
                 Variable("Run", decimals=0, label="Run order", measure=ORDINAL),
                 Variable("Block", decimals=0, label="Block", measure=NOMINAL,
                          value_labels={float(b): f"Block {b}" for b in sorted(set(blocks))})]
    if groups is not None:
        cols["Group"] = groups.astype(float) + 1
        variables.append(Variable("Group", decimals=0, label="Group (whole plot)",
                                  measure=NOMINAL))
    cols["Space_Type"] = types.astype(float)
    variables.append(Variable("Space_Type", decimals=0, label="Space Type", measure=NOMINAL,
                              value_labels={float(t): SPACE_TYPES[int(t)]
                                            for t in sorted(set(types))}))
    fmeta = []
    mix = meta.get("mixture")
    for j, f in enumerate(factors):
        name = _unique_name(f.get("name", ""), used, letter(j))
        units = f.get("units", "")
        title = f"{letter(j)}: {f.get('name') or name}" + (f" ({units})" if units else "")
        if f.get("type") == "categoric":
            finite = np.isfinite(coded[:, j])
            idx = np.where(finite, coded[:, j], 0).astype(int)
            values = np.where(finite, np.array(f["levels"], float)[idx], np.nan)
            var = Variable(name, decimals=_decimals(values), label=title, measure=NOMINAL,
                           value_labels={float(c): lab for c, lab in zip(f["levels"],
                                                                         f["labels"])})
            fm = {"name": f.get("name") or name, "column": name, "units": units,
                  "type": "categoric", "levels": [float(c) for c in f["levels"]],
                  "labels": list(f["labels"])}
        elif mix is not None and (meta.get("kind") == "mixture" or f.get("type") == "mixture"):
            values = coded[:, j]
            var = Variable(name, decimals=max(_decimals(values), 2 if mix["total"] <= 1 else 0),
                           label=title)
            fm = {"name": f.get("name") or name, "column": name, "units": units,
                  "type": "mixture", "low": float(mix["lower"][j]),
                  "high": float(mix["upper"][j])}
        else:
            low, high = float(f.get("low", -1)), float(f.get("high", 1))
            if not high > low:
                raise ValueError(f"Factor {f.get('name') or name}: High must exceed Low.")
            values = (low + high) / 2 + coded[:, j] * (high - low) / 2
            var = Variable(name, decimals=_decimals(values), label=title)
            fm = {"name": f.get("name") or name, "column": name, "units": units,
                  "type": "numeric", "low": low, "high": high}
        if j in hard:
            fm["changes"] = "hard"
        cols[name] = values
        variables.append(var)
        fmeta.append(fm)
    rmeta = []
    for j, r in enumerate(responses):
        name = _unique_name(r.get("name", ""), used, f"R{j + 1}")
        units = r.get("units", "")
        cols[name] = np.full(n, np.nan)
        variables.append(Variable(name, decimals=2, label=f"R{j + 1}: {r.get('name') or name}"
                                  + (f" ({units})" if units else "")))
        rmeta.append({"name": r.get("name") or name, "column": name, "units": units})
    df = pd.DataFrame(cols).sort_values("Run", kind="stable").reset_index(drop=True)
    ds = Dataset(df, variables, name=spec.get("name") or "Design")
    columns = {"std": "Std", "run": "Run", "block": "Block", "space_type": "Space_Type"}
    if groups is not None:
        columns["group"] = "Group"
        meta["split_plot"] = {"hard": hard, "groups": int(groups.max() + 1),
                              "variance_ratio": float(opt.get("variance_ratio", 1.0))}
    meta.update({"factors": fmeta, "responses": rmeta, "runs": n,
                 "blocks": int(blocks.max()), "replicates": reps,
                 "randomized": bool(randomize), "seed": spec.get("seed"),
                 "columns": columns,
                 "options": {k2: v for k2, v in opt.items() if _jsonable(v)}})
    if "center_points" not in meta:
        meta["center_points"] = int(np.sum(types == 0))
    ds.meta["doe"] = meta
    return DesignResult(ds, meta, notes)


def randomized_run_order(blocks: np.ndarray, groups: np.ndarray | None,
                         rng: np.random.Generator) -> np.ndarray:
    """Run numbers randomised within blocks; for a split-plot design the groups (whole
    plots) are randomised and then the runs within each group (restricted randomisation)."""
    n = len(blocks)
    run = np.empty(n, dtype=int)
    pos = 0
    for b in sorted(set(np.asarray(blocks).tolist())):
        members = np.flatnonzero(blocks == b)
        if groups is None:
            perm = rng.permutation(members)
            run[perm] = np.arange(pos + 1, pos + len(members) + 1)
            pos += len(members)
            continue
        for g in rng.permutation(np.unique(groups[members])):
            sub = rng.permutation(members[groups[members] == g])
            run[sub] = np.arange(pos + 1, pos + len(sub) + 1)
            pos += len(sub)
    return run


def _standard_run_order(blocks: np.ndarray, groups: np.ndarray | None) -> np.ndarray:
    keys = (blocks,) if groups is None else (groups, blocks)
    order = np.lexsort(keys)
    run = np.empty(len(blocks), dtype=int)
    run[order] = np.arange(1, len(blocks) + 1)
    return run


def _replicate_groups(coded, blocks, types, groups, reps: int, block_reps: bool = False):
    """Replicates of a design; whole plots are replicated as new groups and, optionally,
    each replicate becomes a block (within which any confounded blocks are kept)."""
    if reps <= 1:
        return coded, blocks, types, groups
    n_g = int(groups.max()) + 1 if groups is not None else 0
    nb = int(blocks.max())
    cb = [coded] * reps
    bb = [blocks + r * nb if block_reps else blocks for r in range(reps)]
    gg = [groups + r * n_g for r in range(reps)] if groups is not None else None
    return (np.vstack(cb), np.concatenate(bb), np.concatenate([types] * reps),
            np.concatenate(gg) if gg is not None else None)


def _jsonable(v) -> bool:
    return isinstance(v, (str, int, float, bool, type(None))) or (
        isinstance(v, (list, tuple)) and all(_jsonable(x) for x in v))


def _replicate(coded, blocks, types, reps: int):
    if reps <= 1:
        return coded, blocks, types
    return (np.vstack([coded] * reps), np.concatenate([blocks] * reps),
            np.concatenate([types] * reps))


def _mixture_design(design: str, factors: list[dict], opt: dict, rng):
    q = len(factors)
    if q < 2:
        raise ValueError("A mixture needs at least 2 components.")
    total = float(opt.get("total", 1.0))
    lower = np.array([float(f.get("low", 0.0) or 0.0) for f in factors])
    upper = np.array([float(f.get("high", total)) if f.get("high") not in (None, "")
                      else total for f in factors])
    upper = np.minimum(upper, total)
    if np.any(upper < lower):
        raise ValueError("Each component's upper limit must be at least its lower limit.")
    if lower.sum() > total + 1e-9:
        raise ValueError("The lower limits add up to more than the mixture total.")
    span = total - lower.sum()
    # effective upper bounds implied by the others' lower bounds
    implied = lower + span
    upper = np.minimum(upper, implied)
    constrained_upper = bool(np.any(upper < implied - 1e-9))
    notes = []
    model = opt.get("model", "quadratic")
    cons_list = _parse_constraints(opt.get("constraints"), q)
    blocks_out = None
    extra_meta: dict = {}
    if design in ("lattice", "centroid", "simscreen"):
        if constrained_upper or cons_list:
            raise ValueError("Upper limits or constraints make the region irregular: use an "
                             "extreme-vertices or optimal mixture design.")
        if design == "simscreen":
            pseudo, t0 = simplex_screening(q)
            types = list(t0)
        else:
            pseudo = (simplex_lattice(q, int(opt.get("degree", 2))) if design == "lattice"
                      else simplex_centroid(q))
            types = [mixture_type(x) for x in pseudo]
        if opt.get("augment", False) and design != "simscreen":
            extra = [axial_blends(q), np.full((1, q), 1.0 / q)]
            for e in extra:
                for x in e:
                    if not any(np.allclose(x, p) for p in pseudo):
                        pseudo = np.vstack([pseudo, x])
                        types.append(mixture_type(x) if not np.allclose(x, 1 / q) else 7)
        reps = int(opt.get("center_points", 0))
        if reps:
            pseudo = np.vstack([pseudo, np.full((reps, q), 1.0 / q)])
            types += [7] * reps
        actual = lower + span * pseudo
        types = np.array(types)
    else:
        from rsstatchem.core.doe import constraints as cons
        from rsstatchem.core.doe import model as mdl

        region = cons.mixture_region(lower, upper, total, cons_list)
        engine = (cons_list or str(opt.get("criterion", "D")).upper() != "D"
                  or opt.get("exchange", "point") != "point" or opt.get("terms")
                  or int(opt.get("blocks", 1) or 1) > 1 or design == "ev"
                  and set(opt.get("include", [])) & {"thirds", "interior"})
        if not engine:
            cand, ctypes = mixture_candidates(lower, upper, total)
        else:
            pc, ctypes = region.candidates(("vertex", "edge", "thirds", "plane", "interior",
                                            "centroid"))
            cand = lower + span * pc
        if design == "ev":
            names = {4: "vertex", 3: "edge", 5: "plane", 6: "interior", 7: "centroid",
                     9: "thirds"}
            keep = set(opt.get("include", ["vertex", "edge", "centroid"]))
            mask = np.array([names[int(t)] in keep for t in ctypes])
            actual, types = cand[mask], ctypes[mask]
        elif not engine:
            # enrich with feasible {q,2} and {q,3} lattice blends (thirds of edges, face
            # centroids) so that cubic and special cubic models are estimable
            extra, etypes = [], []
            for m in (2, 3):
                for x in lower + span * simplex_lattice(q, m):
                    if np.all(x <= upper + 1e-9) and not any(np.allclose(x, c, atol=1e-7)
                                                             for c in list(cand) + extra):
                        extra.append(x)
                        etypes.append(mixture_type((x - lower) / span) if span > 0 else 5)
            if extra:
                cand = np.vstack([cand, extra])
                ctypes = np.concatenate([ctypes, etypes])
            pseudo_c = (cand - lower) / span if span > 0 else cand
            F = _model_matrix_for_design(pseudo_c, model, True)
            p = F.shape[1]
            n_model = max(int(opt.get("model_points", p)), p)
            idx = d_optimal(F, min(n_model, len(cand)), rng)
            idx = add_distance_points(pseudo_c, idx, int(opt.get("lof_points", 5)))
            reps = high_leverage(F, idx, int(opt.get("replicate_points", 5)))
            idx = np.concatenate([idx, reps])
            actual, types = cand[idx], ctypes[idx]
            extra_meta = {"criterion": "D", "exchange": "point"}
        else:
            terms = _terms_from_keys(opt["terms"]) if opt.get("terms") else \
                mdl.mixture_terms(q, model)

            def matrix(c):
                return mdl.raw_matrix(np.atleast_2d(c), terms, mixture=True, intercept=False)

            pc_all = (cand - lower) / span
            for m in (2, 3):  # feasible lattice blends enrich the candidates
                lat = simplex_lattice(q, m)
                lat = lat[region.feasible(lat)]
                pc_all = np.unique(np.vstack([pc_all, lat]).round(12), axis=0)
            r = run_optimal(matrix, region, opt, rng, pc_all, None, "D")
            actual = lower + span * r["coded"]
            types = classify_points(region, r["coded"])
            if int(opt.get("blocks", 1) or 1) > 1:
                blocks_out = r["blocks"]
            extra_meta = {"criterion": r["criterion"], "exchange": r["exchange"],
                          "criterion_value": r["value"], "model_points": r["model_points"],
                          "lof_points": r["lof_points"],
                          "replicate_points": r["replicate_points"]}
            if opt.get("terms"):
                extra_meta["terms"] = [mdl.term_key(t) for t in terms]
        reps = int(opt.get("center_points", 0))
        if reps:
            c = cand[ctypes == 7][0] if np.any(ctypes == 7) else cand.mean(axis=0)
            actual = np.vstack([actual, np.tile(c, (reps, 1))])
            types = np.concatenate([types, np.full(reps, 7)])
            if blocks_out is not None:
                blocks_out = np.concatenate([blocks_out, np.ones(reps, int)])
    mix = {"total": total, "lower": lower.tolist(), "upper": upper.tolist(),
           "model": model, "degree": int(opt.get("degree", 2)), "notes": notes,
           "pseudo": "L_Pseudo" if lower.sum() > 0 else "Real"}
    if cons_list:
        mix["constraints"] = cons_list
    mix.update(extra_meta)
    if blocks_out is not None:
        mix["_blocks"] = np.asarray(blocks_out, int)
    return actual, np.asarray(types, dtype=int), mix


def _combined_design(factors, opt, rng, cat, nlev):
    """Optimal combined mixture-process design: L-pseudo components (first columns, returned
    in actual amounts) beside coded process factors."""
    from rsstatchem.core.doe import constraints as cons
    from rsstatchem.core.doe import model as mdl
    from rsstatchem.core.doe import terms_combined as tc

    mixj = [j for j, f in enumerate(factors) if f.get("type") == "mixture"]
    q = len(mixj)
    k = len(factors)
    if q < 2 or mixj != list(range(q)):
        raise ValueError("A combined design lists at least 2 mixture components first, then "
                         "the process factors.")
    if q == k:
        raise ValueError("A combined design needs at least one process factor.")
    total = float(opt.get("total", 1.0))
    lower = np.array([float(factors[j].get("low", 0) or 0) for j in mixj])
    upper = np.array([float(factors[j].get("high", total)) if factors[j].get("high")
                      not in (None, "") else total for j in mixj])
    upper = np.minimum(upper, total)
    if lower.sum() > total + 1e-9:
        raise ValueError("The lower limits add up to more than the mixture total.")
    span = total - lower.sum()
    upper = np.minimum(upper, lower + span)
    kp = k - q
    pcat = [j - q for j in cat]
    plows = [float(factors[j].get("low", -1)) if j not in cat else 0.0 for j in range(q, k)]
    phighs = [float(factors[j].get("high", 1)) if j not in cat else 1.0 for j in range(q, k)]
    mix_cons = _parse_constraints(opt.get("constraints"), q)
    proc_cons = _parse_constraints(opt.get("process_constraints"), kp)
    region = cons.combined_region(lower, upper, total, plows, phighs, mix_cons, proc_cons,
                                  [nlev[j] if j in cat else 0 for j in range(q, k)])
    mix_order = opt.get("mix_model", "quadratic")
    proc_order = opt.get("process_model", "linear")
    catflags = [j in cat for j in range(k)]
    terms = _terms_from_keys(opt["terms"]) if opt.get("terms") else tc.combined_terms(
        q, kp, mix_order, proc_order, [j in pcat for j in range(kp)])

    def matrix(c):
        return tc.combined_matrix(c, terms, catflags if any(catflags) else None, nlev)

    # candidates: constrained-mixture points x process grid (x categoric levels)
    mreg = cons.mixture_region(lower, upper, total, mix_cons)
    mc, _ = mreg.candidates(("vertex", "edge", "plane", "interior", "centroid"))
    if proc_order in ("quadratic", "cubic") or any(len(t) > 1 for t in terms
                                                    if all(i >= q for i in t)):
        levels = (-1.0, 0.0, 1.0)
    else:
        levels = (-1.0, 0.0, 1.0)
    preg = cons.process_region(plows, phighs, proc_cons,
                               [nlev[j] if j in cat else 0 for j in range(q, k)])
    axes = [np.arange(nlev[q + j], dtype=float) if (q + j) in cat else np.array(levels)
            for j in range(kp)]
    pc = np.array(list(itertools.product(*axes)), float)
    pc = pc[preg.feasible(pc)]
    cand = None
    if len(mc) * len(pc) <= 20000:
        cand = np.array([np.concatenate([a, b]) for a in mc for b in pc])
    o = dict(opt)
    if cand is None:
        o["exchange"] = "coordinate"
    r = run_optimal(matrix, region, o, rng, cand, None, "I")
    coded = r["coded"].copy()
    coded[:, :q] = lower + span * coded[:, :q]
    types = classify_points(region, r["coded"])
    mixmeta = {"total": total, "lower": lower.tolist(), "upper": upper.tolist(),
               "model": mix_order, "process_model": proc_order, "n_mix": q,
               "pseudo": "L_Pseudo" if lower.sum() > 0 else "Real", "degree": 2,
               "criterion": r["criterion"], "exchange": r["exchange"],
               "model_points": r["model_points"], "lof_points": r["lof_points"],
               "replicate_points": r["replicate_points"], "log_det": r["log_det"],
               "criterion_value": r["value"]}
    if opt.get("terms"):
        mixmeta["terms"] = [mdl.term_key(t) for t in terms]
    if mix_cons or proc_cons:
        mixmeta["constraints"] = mix_cons
        if proc_cons:
            mixmeta["process_constraints"] = proc_cons
    return coded, r["blocks"], types, mixmeta


def _space_filling(design, factors, opt, rng, cat, nlev, meta):
    """Latin hypercube, optimal (maximin) distance and user-defined process designs."""
    from rsstatchem.core.doe import constraints as cons

    k = len(factors)
    num = [j for j in range(k) if j not in cat]
    lows = [float(f.get("low", -1)) if j not in cat else 0.0 for j, f in enumerate(factors)]
    highs = [float(f.get("high", 1)) if j not in cat else 1.0 for j, f in enumerate(factors)]
    cons_list = _parse_constraints(opt.get("constraints"), k)
    region = cons.process_region(lows, highs, cons_list,
                                 [nlev[j] if j in cat else 0 for j in range(k)])
    if cons_list:
        meta["constraints"] = cons_list
    if design == "lhs":
        n = int(opt.get("runs", 0) or 0) or max(10, 5 * k)
        coded = np.zeros((0, k))
        tries = 0
        while len(coded) < n and tries < 40:
            m = n if not cons_list else 3 * n
            X = np.zeros((m, k))
            if num:
                X[:, num] = latin_hypercube(m, len(num), rng)
            for j in cat:
                X[:, j] = rng.permutation(np.resize(np.arange(nlev[j]), m))
            X = X[region.feasible(X)]
            coded = np.vstack([coded, X])
            tries += 1
        if len(coded) < n:
            raise ValueError("The constraints leave too little room for a Latin hypercube.")
        coded = coded[:n]
        meta["space_filling"] = "Latin hypercube"
        return coded, np.full(len(coded), 6)
    if design == "distance":
        n = int(opt.get("runs", 0) or 0) or max(10, 5 * k)
        levels = int(opt.get("levels", 5) or 5)
        axes = [np.arange(nlev[j], dtype=float) if j in cat else np.linspace(-1, 1, levels)
                for j in range(k)]
        total = int(np.prod([len(a) for a in axes]))
        if total <= 40000:
            grid = np.array(list(itertools.product(*axes)), float)
        else:
            grid = region.sample(4000, rng)
        cand = grid
        if cons_list:
            cand = np.vstack([grid, region.candidates(("vertex", "edge", "centroid"))[0]])
        cand = np.unique(cand[region.feasible(cand)].round(12), axis=0)
        idx = maximin_points(cand, n, rng)
        coded = cand[idx]
        meta["space_filling"] = "Optimal distance (maximin)"
        return coded, classify_points(region, coded)
    # user-defined
    include = tuple(opt.get("include") or ("vertex", "edge", "centroid"))
    pts, types = region.candidates(include)
    c = int(opt.get("center_points", 0) or 0)
    if c:
        cen = region.candidates(("centroid",))[0]
        pts = np.vstack([pts] + [cen] * c)
        types = np.concatenate([types, np.full(len(cen) * c, 7)])
    if not len(pts):
        raise ValueError("Choose at least one point type.")
    types = np.where(types == 7, 0, types)  # the centroid of a process box is the centre
    meta["point_types"] = list(include)
    return pts, types


def design_from_data(frame, factor_cols: list[str], response_cols: list[str],
                     kind: str = "process", factor_info: dict | None = None,
                     total: float | None = None, name: str = "Historical data"
                     ) -> DesignResult:
    """Historical data / Import Data Set: a design document from existing runs. Numeric
    factors are coded from their data range (or ``factor_info[col]`` low / high), text or
    few-valued factors marked categoric in ``factor_info`` get levels from the data."""
    factor_info = factor_info or {}
    frame = frame.reset_index(drop=True)
    factors, values = [], []
    for col in factor_cols:
        info = dict(factor_info.get(col, {}))
        raw = frame[col]
        nums = pd.to_numeric(raw, errors="coerce")
        is_cat = info.get("type") == "categoric" or (nums.isna().all() and raw.notna().any())
        f = {"name": info.get("name") or str(col), "units": info.get("units", "")}
        if is_cat:
            labels = [str(v) for v in pd.unique(raw.dropna())]
            f.update(type="categoric", levels=labels)
            codes = raw.map({v: i for i, v in enumerate(pd.unique(raw.dropna()))})
            values.append(codes.to_numpy(float))
        else:
            v = nums.to_numpy(float)
            fin = v[np.isfinite(v)]
            if not len(fin):
                raise ValueError(f"Column {col} has no numbers.")
            lo = float(info.get("low", fin.min()))
            hi = float(info.get("high", fin.max()))
            if kind == "mixture":
                f.update(type="mixture", low=float(info.get("low", 0.0)), high=hi)
            else:
                if hi <= lo:
                    hi = lo + 1.0
                f.update(low=lo, high=hi)
            values.append(v)
        factors.append(f)
    opt = {"runs": len(frame), "kind": kind}
    if kind == "mixture":
        sums = np.nansum(np.column_stack(values), axis=1)
        opt["total"] = float(total if total else np.round(np.nanmedian(sums), 8))
    spec = {"design": "historical", "factors": factors,
            "responses": [{"name": str(c)} for c in response_cols], "options": opt,
            "randomize": False, "name": name}
    res = build_design(spec)
    ds = res.dataset
    meta = res.meta
    for j, fm in enumerate(meta["factors"]):
        v = values[j]
        if fm["type"] == "categoric":
            lv = np.asarray(fm["levels"], float)
            v = np.where(np.isfinite(v), lv[np.where(np.isfinite(v), v, 0).astype(int)],
                         np.nan)
        ds.df[fm["column"]] = v
        ds.var(fm["column"]).decimals = _decimals(np.asarray(v, float))
    for r, col in zip(meta["responses"], response_cols):
        ds.df[r["column"]] = pd.to_numeric(frame[col], errors="coerce").to_numpy(float)
    meta["imported"] = True
    res.notes.append(f"Built from {len(frame)} existing runs (historical data): the factor "
                     "settings were not chosen by design, so check the Evaluation before "
                     "trusting the model.")
    return res


def _optimal_terms_count(spec: dict) -> int:
    """Model columns of an optimal design's model (with block columns)."""
    from rsstatchem.core.doe import model as mdl

    opt = spec.get("options", {})
    facs = spec["factors"]
    k = len(facs)
    design = spec["design"]
    cat = [f.get("type") == "categoric" for f in facs]
    nl = [max(len([v for v in f.get("levels") or [] if str(v).strip()]), 2) for f in facs]

    def cols(terms, intercept):
        return int(intercept) + sum(int(np.prod([nl[i] - 1 if cat[i] else 1
                                                 for i in set(mdl.term_factors(t))]))
                                    if t and t[0] != -1 else 1 for t in terms)

    nb = max(int(opt.get("blocks", 1) or 1), 1)
    if opt.get("terms"):
        terms = _terms_from_keys(opt["terms"])
        mix = design in ("mixdopt", "combined")
        return cols(terms, not mix) + nb - 1
    if design == "mixdopt":
        return len(mdl.mixture_terms(k, opt.get("model", "quadratic"))) + nb - 1
    if design == "combined":
        from rsstatchem.core.doe import terms_combined as tc

        q = sum(1 for f in facs if f.get("type") == "mixture")
        terms = tc.combined_terms(q, k - q, opt.get("mix_model", "quadratic"),
                                  opt.get("process_model", "linear"), cat[q:])
        return cols(terms, False) + nb - 1
    if design == "onefactor":
        return len(mdl.process_terms(1, opt.get("model", "quadratic"))) + 1
    return cols(mdl.process_terms(k, opt.get("model", "quadratic"), cat), True) + nb - 1


def design_run_count(spec: dict) -> int | str:
    """Number of runs a specification will produce (for the wizard's live summary)."""
    try:
        spec = dict(spec)
        spec["randomize"] = False
        opt = spec.get("options", {})
        k = len(spec["factors"])
        if spec["design"] in ("mr5", "mr4"):  # formula, no exchange for the preview
            n = mr5_runs(k) if spec["design"] == "mr5" else mr4_runs(
                k, bool(opt.get("extra_runs", True)))
            return n + int(opt.get("center_points", 0) or 0)
        hard = any(f.get("changes") == "hard" for f in spec["factors"])
        if spec["design"] == "dopt" and hard:
            return (int(opt.get("groups", 8)) * int(opt.get("group_size", 4))
                    * int(opt.get("replicates", 1) or 1))
        if spec["design"] in ("lhs", "distance"):
            return int(opt.get("runs", 0) or 0) or max(10, 5 * k)
        if spec["design"] == "historical":
            return int(opt.get("runs", 10) or 10)
        if spec["design"] in ("dopt", "mixdopt", "combined", "onefactor"):
            # skip the (slow) exchange for the preview
            p = _optimal_terms_count(spec)
            lof = int(opt.get("lof_points", 2 if spec["design"] == "onefactor" else 5))
            rep = int(opt.get("replicate_points", 3 if spec["design"] == "onefactor" else 5))
            n = max(int(opt.get("model_points", p) or p), p) + lof + rep
            return n + int(opt.get("center_points", 0) or 0)
        return build_design(spec).meta["runs"]
    except Exception as exc:  # noqa: BLE001 - shown as text in the wizard
        return str(exc)

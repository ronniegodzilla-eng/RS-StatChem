"""Experimental design generation, mirroring Design-Expert's "New Design" builders.

All designs are built with numpy: two-level full and fractional factorials (generators,
defining relation, resolution, alias structure, blocking by confounding), general multilevel
factorials, Plackett-Burman, central composite (circumscribed / inscribed / face-centred with
rotatable, orthogonal, practical, spherical or custom alpha, blocks), Box-Behnken, 3-level
factorials, D-optimal designs (point exchange from a candidate set), mixture designs (simplex
lattice, simplex centroid, extreme vertices for constrained regions, D-optimal mixtures) and
Taguchi orthogonal arrays.

:func:`build_design` turns a wizard specification into a :class:`Dataset` with columns
Std, Run, Block, Space Type, the factors in actual units and empty response columns; the design
definition is stored JSON-serialisable in ``ds.meta["doe"]``.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from rsstatchem.core.dataset import NOMINAL, ORDINAL, Dataset, Variable, sanitize_name

# Design-Expert skips the letter I (it would read as the identity / intercept).
LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ" + "abcdefghjklmnopqrstuvwxyz"

SPACE_TYPES = {0: "Center", 1: "Factorial", 2: "Axial", 3: "Edge", 4: "Vertex", 5: "Plane",
               6: "Interior", 7: "Centroid", 8: "Run"}

DESIGN_TITLES = {
    "factorial": "Regular Two-Level Factorial", "general": "Multilevel Categoric (General "
    "Factorial)", "pb": "Plackett-Burman", "ccd": "Central Composite",
    "bbd": "Box-Behnken", "threelevel": "3-Level Factorial", "dopt": "Optimal (D-optimal)",
    "lattice": "Simplex Lattice", "centroid": "Simplex Centroid",
    "ev": "Extreme Vertices (constrained mixture)", "mixdopt": "Optimal (D-optimal) Mixture",
    "taguchi": "Taguchi Orthogonal Array",
}
MIXTURE_DESIGNS = ("lattice", "centroid", "ev", "mixdopt")
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


def box_behnken(k: int, center_points: int | None = None) -> dict:
    """Box-Behnken design for 3-7 factors (pairs; BIBD triples for 6 and 7 factors)."""
    if k < 3 or k > 7:
        raise ValueError("Box-Behnken designs are available for 3 to 7 factors.")
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


def _model_matrix_for_design(coded: np.ndarray, model: str, mixture: bool) -> np.ndarray:
    from rsstatchem.core.doe import model as mdl

    k = coded.shape[1]
    terms = (mdl.mixture_terms(k, model) if mixture
             else mdl.process_terms(k, model, [False] * k))
    return mdl.raw_matrix(coded, terms, mixture=mixture, intercept=not mixture)


def build_design(spec: dict) -> DesignResult:
    """Create the design Dataset from a wizard specification.

    ``spec``: design (key of DESIGN_TITLES), factors [{name, units, low, high, levels?}],
    responses [{name, units}], options {...}, randomize, seed, name.
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
    if k < 1:
        raise ValueError("Define at least one factor.")

    if design == "factorial":
        p = int(opt.get("fraction", 0))
        gens = opt.get("generators") or default_generators(k, p)
        coded, group = fractional_factorial(k, gens) if p or gens else (full_factorial(k), [])
        p = len(parse_generators(gens, k)) if gens else 0
        nb = int(opt.get("blocks", 1))
        words = block_words(k, group, nb, n_base=k - p) if nb > 1 else []
        blocks = assign_blocks(coded, words)
        types = np.ones(len(coded), dtype=int)
        meta.update({"generators": [g.strip() for g in gens.split(",")] if gens else [],
                     "defining_relation": ["I"] + [("-" if s < 0 else "") + word_str(w)
                                                   for w, s in group],
                     "resolution": resolution(group), "fraction": p,
                     "aliases": alias_structure(k, group),
                     "block_generators": [word_str(w) for w in words]})
        coded, blocks, types = _replicate(coded, blocks, types, int(opt.get("replicates", 1)))
        c = int(opt.get("center_points", 0))
        if c:
            nbk = blocks.max()
            for b in range(1, nbk + 1):
                coded = np.vstack([coded, np.zeros((c, k))])
                blocks = np.concatenate([blocks, np.full(c, b)])
                types = np.concatenate([types, np.zeros(c, dtype=int)])
            meta["center_points"] = c
    elif design == "pb":
        coded = plackett_burman(k)
        blocks, types = np.ones(len(coded), dtype=int), np.ones(len(coded), dtype=int)
        coded, blocks, types = _replicate(coded, blocks, types, int(opt.get("replicates", 1)))
        c = int(opt.get("center_points", 0))
        if c:
            coded = np.vstack([coded, np.zeros((c, k))])
            blocks = np.concatenate([blocks, np.ones(c, dtype=int)])
            types = np.concatenate([types, np.zeros(c, dtype=int)])
        meta["pb_runs"] = pb_runs(k)
    elif design == "ccd":
        res = central_composite(k, opt.get("alpha_type", "rotatable"),
                                opt.get("ccd_type", "circumscribed"),
                                opt.get("center_points"), int(opt.get("blocks", 1)),
                                int(opt.get("center_axial", 2)), opt.get("generators", ""),
                                opt.get("alpha"))
        coded, blocks, types = res["coded"], res["blocks"], res["types"]
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
        coded, blocks, types = _replicate(coded, blocks, types, int(opt.get("replicates", 1)))
    elif design == "bbd":
        res = box_behnken(k, opt.get("center_points"))
        coded, blocks, types = res["coded"], res["blocks"], res["types"]
        coded, blocks, types = _replicate(coded, blocks, types, int(opt.get("replicates", 1)))
    elif design == "threelevel":
        res = three_level(k, int(opt.get("center_points", 0)))
        coded, blocks, types = res["coded"], res["blocks"], res["types"]
        coded, blocks, types = _replicate(coded, blocks, types, int(opt.get("replicates", 1)))
    elif design == "dopt":
        model = opt.get("model", "quadratic")
        cand = process_candidates(k, model)
        F = _model_matrix_for_design(cand, model, False)
        p = F.shape[1]
        n_model = max(int(opt.get("model_points", p)), p)
        idx = d_optimal(F, min(n_model, len(cand)), rng, starts=int(opt.get("starts", 8)))
        idx = add_distance_points(cand, idx, int(opt.get("lof_points", 5)))
        reps = high_leverage(F, idx, int(opt.get("replicate_points", 5)))
        all_idx = np.concatenate([idx, reps])
        coded = cand[all_idx]
        blocks = np.ones(len(coded), dtype=int)
        types = np.where(np.all(coded == 0, axis=1), 0,
                         np.where(np.all(np.abs(coded) == 1, axis=1), 4, 3))
        sign, ld = np.linalg.slogdet(F[all_idx].T @ F[all_idx])
        meta.update({"model": model, "model_points": int(len(idx) - int(opt.get(
            "lof_points", 5))), "lof_points": int(opt.get("lof_points", 5)),
            "replicate_points": int(len(reps)),
            "log_det": float(ld) if sign > 0 else None})
    elif design in MIXTURE_DESIGNS:
        coded, types, mix = _mixture_design(design, factors, opt, rng)
        blocks = np.ones(len(coded), dtype=int)
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
        coded = general_factorial(levels)
        blocks, types = np.ones(len(coded), dtype=int), np.full(len(coded), 1)
        coded, blocks, types = _replicate(coded, blocks, types, int(opt.get("replicates", 1)))
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
        coded, blocks, types = _replicate(coded, blocks, types, int(opt.get("replicates", 1)))
    else:
        raise ValueError(f"Unknown design {design!r}.")

    n = len(coded)
    std = np.arange(1, n + 1)
    run = np.empty(n, dtype=int)
    if spec.get("randomize", True):
        pos = 0
        for b in sorted(set(blocks)):
            members = np.flatnonzero(blocks == b)
            perm = rng.permutation(members)
            run[perm] = np.arange(pos + 1, pos + len(members) + 1)
            pos += len(members)
    else:
        order = np.argsort(blocks, kind="stable")
        run[order] = np.arange(1, n + 1)

    used = {"std", "run", "block", "space_type"}
    cols: dict[str, np.ndarray] = {"Std": std.astype(float), "Run": run.astype(float),
                                   "Block": blocks.astype(float),
                                   "Space_Type": types.astype(float)}
    variables = [Variable("Std", decimals=0, label="Standard order", measure=ORDINAL),
                 Variable("Run", decimals=0, label="Run order", measure=ORDINAL),
                 Variable("Block", decimals=0, label="Block", measure=NOMINAL,
                          value_labels={float(b): f"Block {b}" for b in sorted(set(blocks))}),
                 Variable("Space_Type", decimals=0, label="Space Type", measure=NOMINAL,
                          value_labels={float(t): SPACE_TYPES[int(t)]
                                        for t in sorted(set(types))})]
    fmeta = []
    mix = meta.get("mixture")
    for j, f in enumerate(factors):
        name = _unique_name(f.get("name", ""), used, letter(j))
        units = f.get("units", "")
        title = f"{letter(j)}: {f.get('name') or name}" + (f" ({units})" if units else "")
        if f.get("type") == "categoric":
            idx = coded[:, j].astype(int)
            values = np.array(f["levels"], float)[idx]
            var = Variable(name, decimals=_decimals(values), label=title, measure=NOMINAL,
                           value_labels={float(c): lab for c, lab in zip(f["levels"],
                                                                         f["labels"])})
            fm = {"name": f.get("name") or name, "column": name, "units": units,
                  "type": "categoric", "levels": [float(c) for c in f["levels"]],
                  "labels": list(f["labels"])}
        elif mix is not None:
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
    meta.update({"factors": fmeta, "responses": rmeta, "runs": n,
                 "blocks": int(blocks.max()), "replicates": int(opt.get("replicates", 1)),
                 "randomized": bool(spec.get("randomize", True)), "seed": spec.get("seed"),
                 "columns": {"std": "Std", "run": "Run", "block": "Block",
                             "space_type": "Space_Type"},
                 "options": {k2: v for k2, v in opt.items() if _jsonable(v)}})
    if "center_points" not in meta:
        meta["center_points"] = int(np.sum(types == 0))
    ds.meta["doe"] = meta
    return DesignResult(ds, meta, notes)


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
    if design in ("lattice", "centroid"):
        if constrained_upper:
            raise ValueError("Upper limits make the region irregular: use an extreme-vertices "
                             "or D-optimal mixture design.")
        pseudo = (simplex_lattice(q, int(opt.get("degree", 2))) if design == "lattice"
                  else simplex_centroid(q))
        types = [mixture_type(x) for x in pseudo]
        if opt.get("augment", False):
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
        cand, ctypes = mixture_candidates(lower, upper, total)
        if design == "ev":
            keep = set(opt.get("include", ["vertex", "edge", "centroid"]))
            mask = np.array([{4: "vertex", 3: "edge", 5: "plane", 6: "interior",
                              7: "centroid"}[t] in keep for t in ctypes])
            actual, types = cand[mask], ctypes[mask]
        else:
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
        reps = int(opt.get("center_points", 0))
        if reps:
            c = cand[ctypes == 7][0] if np.any(ctypes == 7) else cand.mean(axis=0)
            actual = np.vstack([actual, np.tile(c, (reps, 1))])
            types = np.concatenate([types, np.full(reps, 7)])
    mix = {"total": total, "lower": lower.tolist(), "upper": upper.tolist(),
           "model": model, "degree": int(opt.get("degree", 2)), "notes": notes,
           "pseudo": "L_Pseudo" if lower.sum() > 0 else "Real"}
    return actual, np.asarray(types, dtype=int), mix


def design_run_count(spec: dict) -> int | str:
    """Number of runs a specification will produce (for the wizard's live summary)."""
    try:
        spec = dict(spec)
        spec["randomize"] = False
        if spec["design"] in ("dopt", "mixdopt"):  # skip the (slow) exchange for the preview
            from rsstatchem.core.doe import model as mdl

            opt = spec.get("options", {})
            k = len(spec["factors"])
            order = opt.get("model", "quadratic")
            p = len(mdl.mixture_terms(k, order)) if spec["design"] == "mixdopt" else \
                1 + len(mdl.process_terms(k, order))
            return (max(int(opt.get("model_points", p) or p), p) + int(opt.get("lof_points", 5))
                    + int(opt.get("replicate_points", 5)) + int(opt.get("center_points", 0)))
        return build_design(spec).meta["runs"]
    except Exception as exc:  # noqa: BLE001 - shown as text in the wizard
        return str(exc)

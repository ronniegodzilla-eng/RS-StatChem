"""Analyze > Descriptive Statistics > TURF Analysis (SPSS's Total Unduplicated Reach and
Frequency, STATS TURF).

For every combination of up to k variables: **reach** = number of cases with at least one
positive response among the variables (a response is positive when its value is in the
"responses counted as positive" list, default 1), **frequency** = total number of positive
responses over those variables. Combinations are ranked by reach, then frequency. With too many
combinations the search turns heuristic: the best ``beam`` combinations of size k − 1 are
extended by one variable (a greedy beam search), as the extension's heuristic does after the
first group sizes.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from rsstatchem.core.dataset import Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int
from rsstatchem.core.stats.ml_common import MLError, raw_column
from rsstatchem.core.stats.roc import figure, png, style

TurfError = MLError


def positive_matrix(ds: Dataset, variables: list[str], positive: list) -> np.ndarray:
    cols = []
    for v in variables:
        s = raw_column(ds, v)
        cols.append(s.isin(positive).to_numpy())
    return np.column_stack(cols)


def _bits(P: np.ndarray) -> list[int]:
    """Each variable's positive cases as a Python integer bit set."""
    out = []
    for j in range(P.shape[1]):
        b = np.packbits(P[:, j].astype(np.uint8), bitorder="little")
        out.append(int.from_bytes(b.tobytes(), "little"))
    return out


def turf(P: np.ndarray, max_size: int = 3, top: int = 10, exhaustive_limit: int = 200000,
         beam: int = 50) -> dict:
    """{size: [(reach, frequency, combo), ...] best first} and whether a heuristic was used."""
    n, V = P.shape
    bits = _bits(P)
    freq = P.sum(0).astype(int)
    results = {}
    heuristic = {}
    prev: list[tuple] = []
    for k in range(1, min(max_size, V) + 1):
        ncomb = math.comb(V, k)
        if k == 1 or ncomb <= exhaustive_limit:
            combos = itertools.combinations(range(V), k)
            heuristic[k] = False
        else:
            seen = set()
            combos = []
            for _, _, c in prev[:beam]:
                for j in range(V):
                    if j not in c:
                        cc = tuple(sorted(c + (j,)))
                        if cc not in seen:
                            seen.add(cc)
                            combos.append(cc)
            heuristic[k] = True
        scored = []
        for c in combos:
            acc = 0
            for j in c:
                acc |= bits[j]
            scored.append((acc.bit_count(), int(freq[list(c)].sum()), tuple(c)))
        scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
        results[k] = scored[:max(top, beam)]
        prev = scored
    return {"results": results, "heuristic": heuristic, "n": n,
            "responses": int(freq.sum())}


def turf_analysis(ds: Dataset, variables: list[str], positive: list | None = None,
                  max_size: int = 3, top: int = 10, exhaustive_limit: int = 200000,
                  beam: int = 50, chart: bool = True, detail: bool = True) -> Section:
    if len(variables) < 2:
        raise TurfError("Select at least two variables.")
    positive = list(positive) if positive else [1.0]
    P = positive_matrix(ds, variables, positive)   # missing responses count as not positive
    res =turf(P, int(max_size), int(top), int(exhaustive_limit), int(beam))
    n, tot = res["n"], max(res["responses"], 1)
    section = Section("TURF Analysis")
    section.log(f"STATS TURF VARIABLES={' '.join(variables)}\n  /OPTIONS BESTN={top} "
                f"NUMBERTODISPLAY={top} MAXSETSIZE={max_size}\n  /RESPONSES="
                + " ".join(f"{p:g}" if isinstance(p, float) else str(p) for p in positive)
                + ".")

    def names(c):
        return ", ".join(variables[j] for j in c)

    best = []
    for k, rows in res["results"].items():
        if rows:
            r, f, c = rows[0]
            best.append([fmt_int(k), fmt_int(r), fmt(100 * r / n, 1), fmt_int(f),
                         fmt(100 * f / tot, 1), names(c)])
    section.table(Table("Best Reach and Frequency by Group Size",
                        ["Group Size", "Reach", "Pct of Cases", "Frequency", "Pct of Responses",
                         "Variables"], best,
                        footnotes=[f"N = {n} cases; responses counted as positive: "
                                   + ", ".join(f"{p:g}" if isinstance(p, float) else str(p)
                                               for p in positive) + "."]
                        + (["A heuristic search was used for group size(s) "
                            + ", ".join(str(k) for k, h in res["heuristic"].items() if h) + "."]
                           if any(res["heuristic"].values()) else [])))
    if detail:
        for k, rows in res["results"].items():
            section.table(Table(f"Reach and Frequency (Group Size = {k})",
                                ["Rank", "Reach", "Pct of Cases", "Frequency",
                                 "Pct of Responses", "Variables"],
                                [[fmt_int(i + 1), fmt_int(r), fmt(100 * r / n, 1), fmt_int(f),
                                  fmt(100 * f / tot, 1), names(c)]
                                 for i, (r, f, c) in enumerate(rows[:top])]))
    if chart:
        ks = list(res["results"])
        reach = [100 * res["results"][k][0][0] / n for k in ks]
        fig = figure(5.2, 3.6)
        ax = fig.add_subplot(111)
        ax.plot(ks, reach, "-o", color="#1f5fa8")
        ax.set_xticks(ks)
        ax.set_ylim(0, 105)
        style(ax, "Best Reach by Group Size", "Group Size", "Reach (% of cases)")
        section.image("Reach by Group Size", png(fig), width=440)
    section.result = res  # type: ignore[attr-defined]
    return section


def selftest() -> None:
    P = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 1], [1, 0, 1]], bool)
    r = turf(P, 2, 3)
    assert r["results"][2][0][0] == 4 and r["results"][1][0][0] == 2

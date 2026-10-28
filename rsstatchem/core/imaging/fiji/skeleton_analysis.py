"""Skeleton graph analysis in the style of Fiji's "Analyze Skeleton (2D/3D)", written from
Arganda-Carreras et al. (Microsc Res Tech 2010) and the documented output (no Fiji code; the
plugin is GPL).

Every skeleton voxel is classified by its number of neighbours (8-connected in 2-D,
26-connected in 3-D): end points (< 2), slab voxels (2) and junction voxels (> 2); adjacent
junction voxels form one junction. Branches are traced between nodes (end points and
junctions) through slab voxels; the branch length is the sum of the calibrated distances
between consecutive voxel centres. Closed loops without junctions count as one branch.

Per skeleton (connected component): # Branches, # Junctions, # End-point voxels,
# Junction voxels, # Slab voxels, Average Branch Length, # Triple points, # Quadruple points,
Maximum Branch Length; the branch table lists each branch with its end points and Euclidean
distance; the tagged skeleton marks end points 30, junctions 70 and slabs 127 (Fiji's values).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

END, JUNCTION, SLAB = 30, 70, 127


@dataclass
class Branch:
    skeleton: int
    length: float
    v1: tuple
    v2: tuple
    euclidean: float


@dataclass
class SkeletonResult:
    tagged: np.ndarray
    rows: list = field(default_factory=list)
    branches: list = field(default_factory=list)

    def results_table(self):
        from rsstatchem.core.imaging.results import ResultsTable
        rt = ResultsTable("Results")
        for r in self.rows:
            rt.add_row(r)
        return rt

    def branch_table(self):
        from rsstatchem.core.imaging.results import ResultsTable
        rt = ResultsTable("Branch information")
        for b in self.branches:
            rt.add_row({"Skeleton ID": b.skeleton, "Branch length": b.length,
                        "V1 x": b.v1[2], "V1 y": b.v1[1], "V1 z": b.v1[0],
                        "V2 x": b.v2[2], "V2 y": b.v2[1], "V2 z": b.v2[0],
                        "Euclidean distance": b.euclidean})
        return rt


def analyze_skeleton(skel: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> SkeletonResult:
    """``skel``: boolean (Y, X) or (Z, Y, X) skeleton; ``spacing`` = (dz, dy, dx)."""
    s = np.asarray(skel, bool)
    if s.ndim == 2:
        s = s[None]
    sp = np.asarray(spacing, float)
    if sp.size == 2:
        sp = np.array([1.0, sp[0], sp[1]])
    footprint = np.ones((3, 3, 3), int)
    if s.shape[0] == 1:
        footprint[0] = 0
        footprint[2] = 0
    nb = ndimage.convolve(s.astype(int), footprint, mode="constant") - s
    nb = np.where(s, nb, 0)
    tagged = np.zeros(s.shape, np.uint8)
    end = s & (nb < 2)
    junc = s & (nb > 2)
    slab = s & (nb == 2)
    tagged[end] = END
    tagged[junc] = JUNCTION
    tagged[slab] = SLAB
    structure = footprint.astype(bool) | np.pad(np.ones((1, 1, 1), bool), 1)
    trees, ntrees = ndimage.label(s, structure=structure)
    jl, nj = ndimage.label(junc, structure=structure)
    offsets = [o for o in (tuple(int(v) - 1 for v in oo) for oo in np.argwhere(footprint))
               if o != (0, 0, 0)]
    Z, Y, X = s.shape

    def neighbours(p):
        z, y, x = p
        for dz, dy, dx in offsets:
            q = (z + dz, y + dy, x + dx)
            if 0 <= q[0] < Z and 0 <= q[1] < Y and 0 <= q[2] < X and s[q]:
                yield q

    def dist(p, q):
        return math.sqrt(((p[0] - q[0]) * sp[0]) ** 2 + ((p[1] - q[1]) * sp[1]) ** 2 +
                         ((p[2] - q[2]) * sp[2]) ** 2)

    def node_id(p):
        if junc[p]:
            return ("j", int(jl[p]))
        if end[p]:
            return ("e", p)
        return None

    visited_slab = np.zeros(s.shape, bool)
    used_pairs = set()
    branches: list[Branch] = []
    node_branches: dict = {}
    node_voxels = list(map(tuple, np.argwhere(end | junc)))
    for start in node_voxels:
        sid = node_id(start)
        for q in neighbours(start):
            if junc[q] and node_id(q) == sid:
                continue
            if slab[q]:
                if visited_slab[q]:
                    continue
                length = dist(start, q)
                prev, cur = start, q
                visited_slab[cur] = True
                end_node = None
                while True:
                    others = [r for r in neighbours(cur) if r != prev]
                    if not others:
                        break
                    nxt = others[0]
                    length += dist(cur, nxt)
                    if not slab[nxt]:
                        end_node = nxt
                        break
                    if visited_slab[nxt]:
                        break
                    prev, cur = cur, nxt
                    visited_slab[cur] = True
                v2 = end_node if end_node is not None else cur
                b = Branch(int(trees[start]), length, start, v2, dist(start, v2))
                branches.append(b)
                node_branches.setdefault(sid, []).append(b)
                if end_node is not None and node_id(end_node) is not None:
                    node_branches.setdefault(node_id(end_node), []).append(b)
            else:
                tid = node_id(q)
                if tid is None:
                    continue
                key = tuple(sorted([str(sid), str(tid)])) + (tuple(sorted([start, q])),)
                if key in used_pairs:
                    continue
                used_pairs.add(key)
                b = Branch(int(trees[start]), dist(start, q), start, q, dist(start, q))
                branches.append(b)
                node_branches.setdefault(sid, []).append(b)
                node_branches.setdefault(tid, []).append(b)
    # loops made only of slab voxels
    rest = slab & ~visited_slab
    if rest.any():
        ll, nl = ndimage.label(rest, structure=structure)
        for k in range(1, nl + 1):
            pts = list(map(tuple, np.argwhere(ll == k)))
            p0 = pts[0]
            length = 0.0
            prev, cur = None, p0
            seen = {p0}
            while True:
                nxt = None
                for r in neighbours(cur):
                    if r != prev and r not in seen and rest[r]:
                        nxt = r
                        break
                if nxt is None:
                    length += dist(cur, p0)
                    break
                length += dist(cur, nxt)
                seen.add(nxt)
                prev, cur = cur, nxt
            branches.append(Branch(int(trees[p0]), length, p0, p0, 0.0))
    rows = []
    for t in range(1, ntrees + 1):
        tb = [b for b in branches if b.skeleton == t]
        tmask = trees == t
        jids = np.unique(jl[tmask & junc])
        jids = jids[jids > 0]
        triple = quad = 0
        for j in jids:
            nbr = len(node_branches.get(("j", int(j)), []))
            if nbr == 3:
                triple += 1
            elif nbr == 4:
                quad += 1
        lengths = [b.length for b in tb]
        rows.append({"# Branches": len(tb), "# Junctions": len(jids),
                     "# End-point voxels": int((tmask & end).sum()),
                     "# Junction voxels": int((tmask & junc).sum()),
                     "# Slab voxels": int((tmask & slab).sum()),
                     "Average Branch Length": float(np.mean(lengths)) if lengths else 0.0,
                     "# Triple points": triple, "# Quadruple points": quad,
                     "Maximum Branch Length": float(max(lengths)) if lengths else 0.0})
    out_tag = tagged[0] if skel.ndim == 2 else tagged
    return SkeletonResult(out_tag, rows, branches)

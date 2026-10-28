"""Analyze > Dimension Reduction > Correspondence Analysis, mirroring SPSS CORRESPONDENCE.

Simple correspondence analysis of a two-way table built from a row and a column variable (case
data; use Weight Cases for aggregated counts), with optional category ranges like SPSS's
"Define Range". Chi-square distance, SVD of the standardized residuals, normalization
Symmetrical (default), Principal, Row principal, Column principal or a custom value q in
[-1, 1] (row scores = standard coordinates x sigma^((1+q)/2), column scores x sigma^((1-q)/2)).

Output: Correspondence Table with active margins, Row/Column Profiles, Summary (singular
values, inertia, chi-square, proportion of inertia), Overview Row/Column Points (mass, scores,
inertia, contributions of points to dimensions and of dimensions to points), permuted table and
the biplot / row-points / column-points plots. Results reproduce Greenacre's smoking example
(the table in SPSS's own CORRESPONDENCE tutorial and R's ``ca::smoke``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from rsstatchem.core.dataset import NUMERIC, Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, fmt_p, numeric, value_text
from rsstatchem.core.stats.wp4_util import figure, png, sort_key

NORMALIZATIONS = {"symmetrical": "Symmetrical", "principal": "Principal",
                  "rprincipal": "Row Principal", "cprincipal": "Column Principal",
                  "custom": "Custom"}


def _f(x, d: int = 3) -> str:
    return fmt(x, d)


class CAResult:
    def __init__(self, N: np.ndarray, dims: int, normalization: str = "symmetrical",
                 q: float = 0.0):
        self.N = np.asarray(N, float)
        n = self.N.sum()
        P = self.N / n
        self.n = n
        self.r = P.sum(axis=1)
        self.c = P.sum(axis=0)
        E = np.outer(self.r, self.c)
        S = (P - E) / np.sqrt(E)
        U, sv, Vt = np.linalg.svd(S, full_matrices=False)
        kmax = min(self.N.shape) - 1
        self.sv = sv[:kmax]
        # Sign convention: make the largest absolute column coordinate positive per dimension.
        for k in range(kmax):
            j = np.argmax(np.abs(Vt[k]))
            if Vt[k, j] < 0:
                Vt[k] *= -1
                U[:, k] *= -1
        self.U, self.V = U[:, :kmax], Vt[:kmax].T
        self.X = self.U / np.sqrt(self.r)[:, None]          # standard coordinates
        self.Y = self.V / np.sqrt(self.c)[:, None]
        self.dims = min(dims, kmax)
        self.inertia = self.sv ** 2
        self.total = float(self.inertia.sum())
        self.row_inertia = (S ** 2).sum(axis=1)
        self.col_inertia = (S ** 2).sum(axis=0)
        if normalization == "principal":
            qr, qc = 1.0, 1.0
        elif normalization == "rprincipal":
            qr, qc = 1.0, 0.0
        elif normalization == "cprincipal":
            qr, qc = 0.0, 1.0
        elif normalization == "custom":
            qr, qc = (1 + q) / 2, (1 - q) / 2
        else:
            qr, qc = 0.5, 0.5
        self.row_scores = self.X * self.sv ** qr
        self.col_scores = self.Y * self.sv ** qc
        self.F = self.X * self.sv                               # principal coordinates
        self.G = self.Y * self.sv

    def point_to_dim(self, rows: bool = True) -> np.ndarray:
        m, coords = (self.r, self.X) if rows else (self.c, self.Y)
        return m[:, None] * coords ** 2

    def dim_to_point(self, rows: bool = True) -> np.ndarray:
        m, F, inertia = (self.r, self.F, self.row_inertia) if rows else \
            (self.c, self.G, self.col_inertia)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(inertia[:, None] > 0, m[:, None] * F ** 2 / inertia[:, None], 0.0)


def _categories(ds: Dataset, name: str, rng) -> tuple[pd.Series, list]:
    var = ds.var(name)
    s = numeric(ds, name) if var.type == NUMERIC else ds.analysis_series(name)
    if rng is not None and var.type == NUMERIC:
        s = s.where((s >= rng[0]) & (s <= rng[1]))
    levels = sorted(pd.unique(s.dropna()), key=sort_key)
    return s, levels


def correspondence(ds: Dataset, row: str, col: str, row_range=None, col_range=None,
                   dims: int = 2, normalization: str = "symmetrical", q: float = 0.0,
                   table: bool = True, overview_rows: bool = True, overview_cols: bool = True,
                   row_profiles: bool = False, col_profiles: bool = False,
                   permutation: bool = False, plots: tuple = ("biplot",)) -> Section:
    sec = Section("Correspondence Analysis")
    sec.log(f"CORRESPONDENCE TABLE = {row}" + (f"({row_range[0]:g} {row_range[1]:g})"
                                               if row_range else "") +
            f" BY {col}" + (f"({col_range[0]:g} {col_range[1]:g})" if col_range else "") +
            f"\n  /DIMENSIONS = {dims}\n  /MEASURE = CHISQ\n  /STANDARDIZE = RCMEAN\n"
            f"  /NORMALIZATION = {normalization.upper() if normalization != 'custom' else q}")
    rs, rlev = _categories(ds, row, row_range)
    cs, clev = _categories(ds, col, col_range)
    frame = pd.DataFrame({"r": rs, "c": cs}).dropna()
    N = np.array([[((frame["r"] == a) & (frame["c"] == b)).sum() for b in clev] for a in rlev],
                 float)
    keep_r = N.sum(axis=1) > 0
    keep_c = N.sum(axis=0) > 0
    N = N[keep_r][:, keep_c]
    rlev = [x for x, k in zip(rlev, keep_r) if k]
    clev = [x for x, k in zip(clev, keep_c) if k]
    rv, cv = ds.var(row), ds.var(col)
    rlab = [str(value_text(rv, x)) for x in rlev]
    clab = [str(value_text(cv, x)) for x in clev]
    if min(N.shape) < 2:
        sec.note("Warning", "Correspondence analysis needs at least two row and two column "
                 "categories with data.", "warning")
        return sec
    res = CAResult(N, dims, normalization, q)
    norm_note = f"{NORMALIZATIONS[normalization]} normalization" + (
        f" (q = {q:g})" if normalization == "custom" else "")
    if table:
        rows = [[lab] + [fmt_int(x) for x in N[i]] + [fmt_int(N[i].sum())]
                for i, lab in enumerate(rlab)]
        rows.append(["Active Margin"] + [fmt_int(x) for x in N.sum(axis=0)] +
                    [fmt_int(N.sum())])
        sec.table(Table("Correspondence Table", [rv.title] + [(cv.title, c) for c in clab] +
                        [(cv.title, "Active Margin")], rows))
    if row_profiles:
        prof = N / N.sum(axis=1, keepdims=True)
        rows = [[lab] + [_f(x) for x in prof[i]] + ["1.000"] for i, lab in enumerate(rlab)]
        rows.append(["Mass"] + [_f(x) for x in res.c] + [""])
        sec.table(Table("Row Profiles", [rv.title] + [(cv.title, c) for c in clab] +
                        [(cv.title, "Active Margin")], rows))
    if col_profiles:
        prof = N / N.sum(axis=0, keepdims=True)
        rows = [[lab] + [_f(x) for x in prof[i]] + [_f(res.r[i])] for i, lab in enumerate(rlab)]
        rows.append(["Active Margin"] + ["1.000"] * len(clab) + [""])
        sec.table(Table("Column Profiles", [rv.title] + [(cv.title, c) for c in clab] +
                        [(cv.title, "Mass")], rows))
    # Summary
    chi = res.n * res.total
    df = (N.shape[0] - 1) * (N.shape[1] - 1)
    rows = []
    cum = 0.0
    for k, (s, inr) in enumerate(zip(res.sv, res.inertia)):
        prop = inr / res.total if res.total > 0 else 0.0
        cum += prop
        rows.append([str(k + 1), _f(s), _f(inr), "", "", _f(prop), _f(cum)])
    rows.append(["Total", "", _f(res.total), _f(chi), fmt_p(stats.chi2.sf(chi, df)),
                 "1.000", "1.000"])
    sec.table(Table("Summary", ["Dimension", "Singular Value", "Inertia", "Chi Square", "Sig.ᵃ",
                                ("Proportion of Inertia", "Accounted for"),
                                ("Proportion of Inertia", "Cumulative")], rows,
                    footnotes=[f"a. {df} degrees of freedom.",
                               "The orientation (sign) of each dimension is arbitrary and the "
                               "rule SPSS uses is not documented by IBM; here the largest "
                               "absolute column coordinate of each dimension is positive. "
                               "Reversing a dimension changes no other statistic."]))
    d = res.dims

    def overview(rows_: bool):
        labels = rlab if rows_ else clab
        mass = res.r if rows_ else res.c
        scores = res.row_scores if rows_ else res.col_scores
        inertia = res.row_inertia if rows_ else res.col_inertia
        p2d, d2p = res.point_to_dim(rows_), res.dim_to_point(rows_)
        out = []
        for i, lab in enumerate(labels):
            out.append([lab, _f(mass[i])] + [_f(scores[i, k]) for k in range(d)] +
                       [_f(inertia[i])] + [_f(p2d[i, k]) for k in range(d)] +
                       [_f(d2p[i, k]) for k in range(d)] + [_f(d2p[i, :d].sum())])
        out.append(["Active Total", _f(mass.sum())] + [""] * d + [_f(res.total)] +
                    ["1.000"] * d + [""] * (d + 1))
        name = rv.title if rows_ else cv.title
        cols = [name, "Mass"] + [("Score in Dimension", str(k + 1)) for k in range(d)] + \
            ["Inertia"] + [("Contribution: Of Point to Inertia of Dimension", str(k + 1))
                           for k in range(d)] + \
            [("Contribution: Of Dimension to Inertia of Point", str(k + 1)) for k in range(d)] + \
            [("Contribution: Of Dimension to Inertia of Point", "Total")]
        return Table(f"Overview {'Row' if rows_ else 'Column'} Pointsᵃ", cols, out,
                     footnotes=[f"a. {norm_note}"])

    if overview_rows:
        sec.table(overview(True))
    if overview_cols:
        sec.table(overview(False))
    if permutation:
        ro = np.argsort(res.row_scores[:, 0])
        co = np.argsort(res.col_scores[:, 0])
        rows = [[rlab[i]] + [fmt_int(N[i, j]) for j in co] + [fmt_int(N[i].sum())] for i in ro]
        rows.append(["Active Margin"] + [fmt_int(N[:, j].sum()) for j in co] +
                    [fmt_int(N.sum())])
        sec.table(Table("Permuted Correspondence Table: Dimension 1",
                        [rv.title] + [(cv.title, clab[j]) for j in co] +
                        [(cv.title, "Active Margin")], rows))
    if d >= 2:
        if "biplot" in plots:
            sec.image(f"Row and Column Points ({norm_note})",
                      _plot(res, rlab, clab, rv.title, cv.title, True, True, norm_note))
        if "rows" in plots:
            sec.image(f"Row Points for {rv.title}", _plot(res, rlab, clab, rv.title, cv.title,
                                                          True, False, norm_note))
        if "cols" in plots:
            sec.image(f"Column Points for {cv.title}", _plot(res, rlab, clab, rv.title,
                                                             cv.title, False, True, norm_note))
    return sec


def _plot(res: CAResult, rlab, clab, rname, cname, rows, cols, norm_note) -> bytes:
    fig = figure(6.0, 5.0)
    ax = fig.add_subplot(111)
    if rows:
        ax.scatter(res.row_scores[:, 0], res.row_scores[:, 1], color="#1f4e9c", marker="o",
                   label=rname)
        for (x, y), t in zip(res.row_scores[:, :2], rlab):
            ax.annotate(t, (x, y), fontsize=8, color="#1f4e9c", xytext=(3, 3),
                        textcoords="offset points")
    if cols:
        ax.scatter(res.col_scores[:, 0], res.col_scores[:, 1], color="#c8102e", marker="^",
                   label=cname)
        for (x, y), t in zip(res.col_scores[:, :2], clab):
            ax.annotate(t, (x, y), fontsize=8, color="#c8102e", xytext=(3, 3),
                        textcoords="offset points")
    ax.axhline(0, color="#999999", lw=0.6)
    ax.axvline(0, color="#999999", lw=0.6)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    title = "Row and Column Points" if rows and cols else ("Row Points" if rows else
                                                           "Column Points")
    ax.set_title(f"{title}\n{norm_note}", fontsize=10)
    ax.legend(fontsize=7)
    return png(fig)


SMOKE = np.array([[4, 2, 3, 2], [4, 3, 7, 4], [25, 10, 12, 4], [18, 24, 33, 13],
                  [10, 6, 7, 2]], float)


def selftest() -> None:
    res = CAResult(SMOKE, 2, "principal")
    assert abs(res.inertia[0] - 0.07476) < 1e-4 and abs(res.total - 0.08519) < 1e-4
    rows = np.repeat(np.arange(5), 4)
    cols = np.tile(np.arange(4), 5)
    df = pd.DataFrame({"staff": np.repeat(rows, SMOKE.ravel().astype(int)).astype(float),
                       "smoke": np.repeat(cols, SMOKE.ravel().astype(int)).astype(float)})
    sec = correspondence(Dataset(df), "staff", "smoke", plots=("biplot", "rows", "cols"),
                         row_profiles=True, col_profiles=True, permutation=True)
    assert any(getattr(i, "title", "") == "Summary" for i in sec.items)

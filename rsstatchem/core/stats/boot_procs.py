"""The SPSS *Bootstrap…* option for the individual procedures.

Each function takes the procedure's finished :class:`Section`, re-computes the bootstrapped
statistics from the data on ``spec.samples`` case resamples (see
:mod:`rsstatchem.core.stats.resample`) and changes the output the way SPSS does:

* a **Bootstrap Specifications** table after the syntax,
* bootstrap sub-rows (Bias, Std. Error, CI Lower/Upper) in statistics tables — Descriptives,
  Frequencies "Statistics", Means "Report", One-Sample / Group / Paired Samples Statistics,
  One-Way ANOVA "Descriptives", Correlations, Partial Correlations,
* bootstrap columns appended to Explore "Descriptives" and Crosstabs "Symmetric Measures",
* separate "Bootstrap for …" tables (with Sig. (2-tailed)) for the t tests, One-Way ANOVA
  multiple comparisons and every model's coefficient table (via
  :class:`~rsstatchem.core.stats.resample.ModelInfo` attached to the procedure result).

All statistics are recomputed from the same cases the procedure used; cases are resampled
from the procedure's analysis cases (simple or stratified sampling).
"""
from __future__ import annotations

import itertools
import re

import numpy as np
import pandas as pd
from scipy import stats

from rsstatchem.core.dataset import NUMERIC, Dataset
from rsstatchem.core.output import Section
from rsstatchem.core.stats import resample as rs
from rsstatchem.core.stats.common import numeric, trimmed_mean
from rsstatchem.core.stats.resample import BootSpec

_NUM = re.compile(r"-?\d*\.(\d+)")


# ============================================================================ helpers
def _decimals(text: str, default: int = 3) -> int:
    m = _NUM.search(text or "")
    return len(m.group(1)) if m else default


def strata_values(ds: Dataset, spec: BootSpec, rows=None):
    """Stratum labels for the analysis cases (``rows`` = positions in ``ds``)."""
    if not spec.strata:
        return None
    cols = [ds.analysis_series(s).astype(object).to_numpy() for s in spec.strata
            if ds.has_name(s)]
    if not cols:
        return None
    arr = np.column_stack(cols)
    if rows is not None:
        arr = arr[np.asarray(rows)]
    return arr


def nan_moments(xb: np.ndarray) -> dict[str, np.ndarray]:
    """NaN-aware SPSS moments along the last axis (n-1 variance, G1, G2)."""
    ok = np.isfinite(xb)
    n = ok.sum(axis=-1).astype(float)
    x0 = np.where(ok, xb, 0.0)
    with np.errstate(all="ignore"):
        total = x0.sum(axis=-1)
        mean = total / n
        d = np.where(ok, xb - mean[..., None], 0.0)
        m2 = (d * d).sum(axis=-1)
        var = m2 / (n - 1)
        sd = np.sqrt(var)
        m3 = (d ** 3).sum(axis=-1)
        m4 = (d ** 4).sum(axis=-1)
        skew = n * m3 / ((n - 1) * (n - 2) * sd ** 3)
        kurt = n * (n + 1) * m4 / ((n - 1) * (n - 2) * (n - 3) * var * var) - \
            3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
        sem = sd / np.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "var": var, "skew": skew, "kurt": kurt,
            "sum": total, "sem": sem}


def _rowwise(xb: np.ndarray, fn) -> np.ndarray:
    out = np.full(xb.shape[0], np.nan)
    for i, row in enumerate(xb):
        v = row[np.isfinite(row)]
        if len(v):
            try:
                out[i] = fn(v)
            except (ValueError, ZeroDivisionError, FloatingPointError):
                pass
    return out


def _header(col) -> tuple[str, str]:
    return (col[0], col[1]) if isinstance(col, tuple) else (col, "")


def _replace(section: Section, index: int, table) -> None:
    section.items[index] = table


def _tables(section: Section, prefix: str) -> list[int]:
    return [i for i, it in enumerate(section.items)
            if getattr(it, "kind", "") == "table" and it.title.startswith(prefix)]


def finish(section: Section, spec: BootSpec, seconds: float = 0.0) -> Section:
    rs.add_specifications(section, spec)
    return section


# ============================================================================ Descriptives
_DESC = {"Mean": "mean", "Std. Deviation": "sd", "Variance": "var", "Skewness": "skew",
         "Kurtosis": "kurt", "Sum": "sum"}
_DKEYS = ["mean", "sd", "var", "skew", "kurt", "sum"]


def descriptives(section: Section, ds: Dataset, names: list[str], spec: BootSpec) -> Section:
    idx = rs.find_table(section, "Descriptive Statistics")
    names = [n for n in names if ds.has_name(n) and ds.var(n).type == NUMERIC]
    if idx is None or not names:
        return section
    X = np.column_stack([numeric(ds, n).to_numpy(float) for n in names])
    nk = len(_DKEYS)

    def stat(xb):
        m = nan_moments(np.moveaxis(xb, -1, -2))          # (b, p, n) -> (b, p)
        return np.stack([m[k] for k in _DKEYS], axis=-1).reshape(len(xb), -1)

    res = rs.bootstrap(X, stat, spec, strata_values(ds, spec), vectorized=True)
    table = section.items[idx]
    by_title = {ds.var(n).title: i for i, n in enumerate(names)}
    mapping, decs = {}, {}
    for r, row in enumerate(table.rows):
        vi = by_title.get(row[0])
        if vi is None:
            continue
        for c, col in enumerate(table.columns):
            head, sub = _header(col)
            if c == 0 or sub == "Std. Error" or head not in _DESC:
                continue
            mapping.setdefault(r, {})[c] = (res, vi * nk + _DKEYS.index(_DESC[head]))
            decs[c] = _decimals(row[c])
    _replace(section, idx, rs.insert_boot_rows(table, mapping, spec, decs))
    return finish(section, spec)


# ============================================================================ Frequencies
def frequencies(section: Section, ds: Dataset, names: list[str], spec: BootSpec,
                percentiles=(25, 50, 75)) -> Section:
    idx = rs.find_table(section, "Statistics")
    nums = [n for n in names if ds.has_name(n) and ds.var(n).type == NUMERIC]
    if idx is None or not nums:
        return section
    table = section.items[idx]
    pcts = []
    current = ""
    for row in table.rows:
        current = row[0] or current
        if current == "Percentiles":
            try:
                pcts.append(float(row[1]))
            except ValueError:
                pass
    pcts = list(dict.fromkeys(pcts)) or list(percentiles)
    keys = ["mean", "median", "sd", "var", "skew", "kurt", "sum"] + [f"p{p:g}" for p in pcts]
    X = np.column_stack([numeric(ds, n).to_numpy(float) for n in nums])

    def stat(xb):
        xt = np.moveaxis(xb, -1, -2)                        # (b, p, n)
        m = nan_moments(xt)
        with np.errstate(all="ignore"):
            q = np.nanpercentile(xt, [50] + pcts, axis=-1, method="weibull")
        cols = [m["mean"], q[0], m["sd"], m["var"], m["skew"], m["kurt"], m["sum"]] + \
            [q[i + 1] for i in range(len(pcts))]
        return np.stack(cols, axis=-1).reshape(len(xb), -1)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = rs.bootstrap(X, stat, spec, strata_values(ds, spec), vectorized=True)
    labels = {"Mean": "mean", "Median": "median", "Std. Deviation": "sd", "Variance": "var",
              "Skewness": "skew", "Kurtosis": "kurt", "Sum": "sum"}
    col_of = {}
    for c, col in enumerate(table.columns[2:], start=2):
        for i, n in enumerate(nums):
            if col == ds.var(n).title:
                col_of[c] = i
    mapping, decs = {}, {}
    current = ""
    nk = len(keys)
    for r, row in enumerate(table.rows):
        current = row[0] or current
        key = labels.get(row[0]) if row[0] else None
        if current == "Percentiles":
            try:
                key = f"p{float(row[1]):g}"
            except ValueError:
                key = None
        if key not in keys:
            continue
        for c, vi in col_of.items():
            mapping.setdefault(r, {})[c] = (res, vi * nk + keys.index(key))
            decs[c] = max(decs.get(c, 0), _decimals(row[c]))
    _replace(section, idx, rs.insert_boot_rows(table, mapping, spec, decs))
    return finish(section, spec)


# ============================================================================ Explore
_EXPLORE = ["Mean", "5% Trimmed Mean", "Median", "Variance", "Std. Deviation",
            "Interquartile Range", "Skewness", "Kurtosis"]


def explore(section: Section, ds: Dataset, dependents: list[str], factors: list[str],
            spec: BootSpec, missing: str = "listwise") -> Section:
    from rsstatchem.core.stats.explore import _groups

    idx = rs.find_table(section, "Descriptives")
    if idx is None:
        return section
    frame_all = pd.concat({n: numeric(ds, n) for n in dependents} |
                          {f: ds.analysis_series(f) for f in factors}, axis=1)
    if missing == "report":
        for f in factors:
            frame_all[f] = frame_all[f].astype(object).where(frame_all[f].notna(), "__missing")
    if missing == "listwise":
        frame = frame_all.dropna(subset=list(dependents) + list(factors))
    else:
        frame = frame_all.dropna(subset=list(factors)) if factors else frame_all
    rows_pos = frame.index.to_numpy()
    blocks = []          # (dep column, mask over frame rows)
    splits = list(factors) or [None]
    for dep in dependents:
        sub = frame.dropna(subset=[dep])
        for factor in splits:
            for g in _groups(ds, sub, dep, factor):
                if not len(g.data):
                    continue
                mask = frame[dep].notna().to_numpy().copy()
                if factor is not None:
                    mask &= (frame[factor] == g.value).to_numpy()
                blocks.append((dependents.index(dep), mask))
    Y = frame[list(dependents)].to_numpy(float)
    nk = len(_EXPLORE)

    def evaluate(I):
        out = np.full((len(I), len(blocks) * nk), np.nan)
        for b_i, rows in enumerate(I):
            for k, (j, mask) in enumerate(blocks):
                v = Y[rows, j][mask[rows]]
                v = v[np.isfinite(v)]
                if len(v) < 2:
                    continue
                m = nan_moments(v[None])
                q1, q3 = np.percentile(v, [25, 75], method="weibull")
                vals = [m["mean"][0], trimmed_mean(v), np.median(v), m["var"][0],
                        m["sd"][0], q3 - q1, m["skew"][0], m["kurt"][0]]
                out[b_i, k * nk:(k + 1) * nk] = vals
        return out

    est = evaluate(np.arange(len(frame))[None])[0]
    with np.errstate(all="ignore"):
        res = rs.bootstrap_indices(len(frame), evaluate, est, spec,
                                   strata_values(ds, spec, rows_pos), batch=50)
    table = section.items[idx]
    mapping = {}
    block = -1
    for r, row in enumerate(table.rows):
        name = row[table.row_header_cols - 1]
        if name == "Mean":
            block += 1
        if name in _EXPLORE and 0 <= block < len(blocks):
            mapping[r] = (res, block * nk + _EXPLORE.index(name))
    new = rs.append_boot_columns(table, mapping, spec, decimals=_decimals(
        table.rows[0][table.row_header_cols] if table.rows else "", 3), sig=False)
    _replace(section, idx, new)
    return finish(section, spec)


# ============================================================================ Means
def means(section: Section, ds: Dataset, dependents: list[str], layers: list[list[str]],
          spec: BootSpec) -> Section:
    from rsstatchem.core.stats.means import _sort_key

    layers = [lay for lay in layers if lay]
    combos = list(itertools.product(*layers)) if layers else [()]
    report_idx = _tables(section, "Report")
    if len(report_idx) != len(combos):
        return section
    for combo, t_idx in zip(combos, report_idx):
        frame = pd.DataFrame({d: numeric(ds, d) for d in dependents})
        for v in combo:
            frame[f"__{v}"] = ds.analysis_series(v)
        gcols = [f"__{v}" for v in combo]
        if gcols:
            frame = frame[frame[gcols].notna().all(axis=1)]
        cells: list[np.ndarray] = []

        def walk(mask: np.ndarray, depth: int) -> None:
            if depth == len(combo):
                cells.append(mask)
                return
            col = frame[gcols[depth]]
            for v in sorted(col[mask].unique(), key=_sort_key):
                walk(mask & (col == v).to_numpy(), depth + 1)
            walk(mask, depth + 1)

        walk(np.ones(len(frame), bool), 0)
        Y = frame[list(dependents)].to_numpy(float)
        nd = len(dependents)

        def evaluate(I, cells=cells, Y=Y, nd=nd):
            out = np.full((len(I), len(cells) * nd * 2), np.nan)
            yb = Y[I]                                       # (b, n, nd)
            for c, mask in enumerate(cells):
                mb = mask[I][..., None]                     # (b, n, 1)
                m = nan_moments(np.moveaxis(np.where(mb, yb, np.nan), 1, 2))   # (b, nd)
                out[:, (c * nd) * 2:(c * nd + nd) * 2] = np.stack(
                    [m["mean"], m["sd"]], axis=-1).reshape(len(I), -1)
            return out

        est = evaluate(np.arange(len(frame))[None])[0]
        res = rs.bootstrap_indices(len(frame), evaluate, est, spec,
                                   strata_values(ds, spec, frame.index.to_numpy()),
                                   batch=max(1, int(1e6 // max(len(frame) * nd, 1))))
        table = section.items[t_idx]
        mapping, decs = {}, {}
        if nd == 1:
            for c_i, col in enumerate(table.columns):
                head, sub = _header(col)
                if sub not in ("Mean", "Std. Deviation"):
                    continue
                k = 0 if sub == "Mean" else 1
                for r in range(min(len(cells), len(table.rows))):
                    mapping.setdefault(r, {})[c_i] = (res, r * 2 + k)
                    decs[c_i] = _decimals(table.rows[r][c_i])
        else:
            nh = len(combo)
            labels = [row[nh] for row in table.rows]
            ns = labels[1:].index(labels[0]) + 1 if labels[0] in labels[1:] else len(labels)
            for r, row in enumerate(table.rows):
                label = row[nh]
                if label not in ("Mean", "Std. Deviation"):
                    continue
                k = 0 if label == "Mean" else 1
                cpos = r // ns
                for d in range(nd):
                    c_i = nh + 1 + d
                    mapping.setdefault(r, {})[c_i] = (res, (cpos * nd + d) * 2 + k)
                    decs[c_i] = _decimals(row[c_i])
        _replace(section, t_idx, rs.insert_boot_rows(table, mapping, spec, decs))
    return finish(section, spec)


# ============================================================================ t tests
def one_sample_t(section: Section, ds: Dataset, names: list[str], test_value: float,
                 spec: BootSpec, listwise: bool = False) -> Section:
    frame = pd.concat({n: numeric(ds, n) for n in names}, axis=1)
    if listwise:
        frame = frame.dropna()
    X = frame.to_numpy(float)

    def stat(xb):
        m = nan_moments(np.moveaxis(xb, -1, -2))
        return np.stack([m["mean"], m["sd"], m["sem"], m["mean"] - test_value],
                        axis=-1).reshape(len(xb), -1)

    res = rs.bootstrap(X, stat, spec, strata_values(ds, spec, frame.index.to_numpy()),
                       vectorized=True)
    t = rs.find_table(section, "One-Sample Statistics")
    if t is not None:
        table = section.items[t]
        cols = {"Mean": 0, "Std. Deviation": 1, "Std. Error Mean": 2}
        mapping, decs = {}, {}
        for r, row in enumerate(table.rows[:len(names)]):
            for c, col in enumerate(table.columns):
                if col in cols:
                    mapping.setdefault(r, {})[c] = (res, r * 4 + cols[col])
                    decs[c] = _decimals(row[c])
        _replace(section, t, rs.insert_boot_rows(table, mapping, spec, decs))
    labels = [[ds.var(n).title] for n in names]
    bt = rs.boot_table("Bootstrap for One-Sample Test", [""], labels,
                       res.estimate, res, spec, "Mean Difference",
                       index=[i * 4 + 3 for i in range(len(names))])
    rs.place_after(section, "One-Sample Test", bt, last=False)
    return finish(section, spec)


def independent_t(section: Section, ds: Dataset, names: list[str], group: str, groups,
                  cut, spec: BootSpec, listwise: bool = False) -> Section:
    g = ds.analysis_series(group)
    if cut is not None:
        gnum = pd.to_numeric(g, errors="coerce")
        code = np.where(gnum >= cut, 0.0, np.where(gnum < cut, 1.0, np.nan))
    else:
        code = np.where(g == groups[0], 0.0, np.where(g == groups[1], 1.0, np.nan))
    frame = pd.concat({n: numeric(ds, n) for n in names}, axis=1)
    frame["__g"] = code
    frame = frame[frame["__g"].notna()]
    if listwise:
        frame = frame.dropna()
    data = frame.to_numpy(float)
    p = len(names)

    def stat(xb):
        gb = xb[..., -1]
        out = []
        for j in range(p):
            y = xb[..., j]
            a = nan_moments(np.where(gb == 0, y, np.nan))
            b = nan_moments(np.where(gb == 1, y, np.nan))
            out += [a["mean"], a["sd"], a["sem"], b["mean"], b["sd"], b["sem"],
                    a["mean"] - b["mean"]]
        return np.stack(out, axis=-1)

    res = rs.bootstrap(data, stat, spec, strata_values(ds, spec, frame.index.to_numpy()),
                       vectorized=True)
    t = rs.find_table(section, "Group Statistics")
    if t is not None:
        table = section.items[t]
        cols = {"Mean": 0, "Std. Deviation": 1, "Std. Error Mean": 2}
        mapping, decs = {}, {}
        for r, row in enumerate(table.rows[:2 * p]):
            j, which = divmod(r, 2)
            for c, col in enumerate(table.columns):
                if col in cols:
                    mapping.setdefault(r, {})[c] = (res, j * 7 + which * 3 + cols[col])
                    decs[c] = _decimals(row[c])
        _replace(section, t, rs.insert_boot_rows(table, mapping, spec, decs))
    labels, index = [], []
    for j, n in enumerate(names):
        labels += [[ds.var(n).title, "Equal variances assumed"],
                   ["", "Equal variances not assumed"]]
        index += [j * 7 + 6, j * 7 + 6]
    bt = rs.boot_table("Bootstrap for Independent Samples Test", ["", ""], labels,
                       res.estimate, res, spec, "Mean Difference", index=index)
    rs.place_after(section, "Independent Samples Test", bt, last=False)
    return finish(section, spec)


def paired_t(section: Section, ds: Dataset, pairs: list[tuple[str, str]], spec: BootSpec,
             listwise: bool = False) -> Section:
    names = list(dict.fromkeys([n for pr in pairs for n in pr]))
    frame = pd.concat({n: numeric(ds, n) for n in names}, axis=1)
    if listwise:
        frame = frame.dropna()
    X = frame.to_numpy(float)
    pos = [(names.index(a), names.index(b)) for a, b in pairs]

    def stat(xb):
        out = []
        for ia, ib in pos:
            ok = np.isfinite(xb[..., ia]) & np.isfinite(xb[..., ib])
            a = nan_moments(np.where(ok, xb[..., ia], np.nan))
            b = nan_moments(np.where(ok, xb[..., ib], np.nan))
            d = nan_moments(np.where(ok, xb[..., ia] - xb[..., ib], np.nan))
            with np.errstate(all="ignore"):
                ca = np.where(ok, xb[..., ia] - a["mean"][..., None], 0)
                cb = np.where(ok, xb[..., ib] - b["mean"][..., None], 0)
                r = (ca * cb).sum(-1) / np.sqrt((ca * ca).sum(-1) * (cb * cb).sum(-1))
            out += [a["mean"], a["sd"], a["sem"], b["mean"], b["sd"], b["sem"], r,
                    d["mean"]]
        return np.stack(out, axis=-1)

    res = rs.bootstrap(X, stat, spec, strata_values(ds, spec, frame.index.to_numpy()),
                       vectorized=True)
    t = rs.find_table(section, "Paired Samples Statistics")
    if t is not None:
        table = section.items[t]
        cols = {"Mean": 0, "Std. Deviation": 1, "Std. Error Mean": 2}
        mapping, decs = {}, {}
        for r, row in enumerate(table.rows[:2 * len(pairs)]):
            j, which = divmod(r, 2)
            for c, col in enumerate(table.columns):
                if col in cols:
                    mapping.setdefault(r, {})[c] = (res, j * 8 + which * 3 + cols[col])
                    decs[c] = _decimals(row[c])
        _replace(section, t, rs.insert_boot_rows(table, mapping, spec, decs))
    t = rs.find_table(section, "Paired Samples Correlations")
    if t is not None:
        table = section.items[t]
        c = table.columns.index("Correlation") if "Correlation" in table.columns else None
        if c is not None:
            mapping = {r: {c: (res, r * 8 + 6)} for r in range(min(len(pairs),
                                                                     len(table.rows)))}
            _replace(section, t, rs.insert_boot_rows(table, mapping, spec, 3))
    labels = [[f"Pair {i + 1}", f"{ds.var(a).title} - {ds.var(b).title}"]
              for i, (a, b) in enumerate(pairs)]
    bt = rs.boot_table("Bootstrap for Paired Samples Test", ["", ""], labels, res.estimate,
                       res, spec, "Mean", index=[j * 8 + 7 for j in range(len(pairs))])
    rs.place_after(section, "Paired Samples Test", bt, last=False)
    return finish(section, spec)


# ============================================================================ One-Way ANOVA
def oneway(section: Section, ds: Dataset, dependents: list[str], factor: str,
           spec: BootSpec, posthoc: bool = False, listwise: bool = False) -> Section:
    from rsstatchem.core.stats.common import groups_of

    desc_idx = rs.find_table(section, "Descriptives")
    base = pd.concat({d: numeric(ds, d) for d in dependents}, axis=1)
    base["__f"] = ds.analysis_series(factor)
    if listwise:
        base = base.dropna()
    base = base[base["__f"].notna()]
    comp_tables = []
    desc_maps: list = []
    for d_i, dep in enumerate(dependents):
        frame = base[[dep, "__f"]].dropna()
        groups = groups_of(ds, dep, factor, frame=frame.rename(columns={dep: "y", "__f": "g"}))
        if len(groups) < 2:
            continue
        codes = np.full(len(frame), -1)
        fv = frame["__f"].to_numpy(object)
        for k, g in enumerate(groups):
            codes[fv == g.value] = k
        y = frame[dep].to_numpy(float)
        data = np.column_stack([y, codes])
        K = len(groups)
        pairs = [(i, j) for i in range(K) for j in range(K) if i != j]

        def stat(xb, K=K, pairs=pairs):
            yb, cb = xb[..., 0], xb[..., 1]
            out = []
            means = []
            for k in range(K):
                m = nan_moments(np.where(cb == k, yb, np.nan))
                out += [m["mean"], m["sd"], m["sem"]]
                means.append(m["mean"])
            m = nan_moments(yb)
            out += [m["mean"], m["sd"], m["sem"]]
            out += [means[i] - means[j] for i, j in pairs]
            return np.stack(out, axis=-1)

        res = rs.bootstrap(data, stat, spec, strata_values(ds, spec, frame.index.to_numpy()),
                           vectorized=True)
        desc_maps.append((ds.var(dep).title, K, res))
        if posthoc:
            labels = []
            for i in range(K):
                for jn, j in enumerate([j for j in range(K) if j != i]):
                    labels.append([ds.var(dep).title if not labels else "",
                                   groups[i].label if jn == 0 else "", groups[j].label])
            comp_tables.append((labels, res, [3 * (K + 1) + n for n in range(len(pairs))]))
    if desc_idx is not None and desc_maps:
        table = section.items[desc_idx]
        cols = {"Mean": 0, "Std. Deviation": 1, "Std. Error": 2}
        mapping, decs = {}, {}
        r = 0
        rows = table.rows
        for title, K, res in desc_maps:
            while r < len(rows) and rows[r][0] != title:
                r += 1
            for k in range(K + 1):
                if r + k >= len(rows):
                    break
                for c, col in enumerate(table.columns):
                    if col in cols:
                        mapping.setdefault(r + k, {})[c] = (res, 3 * k + cols[col])
                        decs[c] = _decimals(rows[r + k][c])
            r += K + 1
        _replace(section, desc_idx, rs.insert_boot_rows(table, mapping, spec, decs))
    if comp_tables:
        labels = [lab for labs, _, _ in comp_tables for lab in labs]
        from rsstatchem.core.output import Table
        letter = "a"
        heads = rs.boot_headers(spec, letter)
        rows = []
        for labs, res, index in comp_tables:
            for lab, j in zip(labs, index):
                rows.append(lab + [rs.fmt(res.estimate[j])] + res.cells(j))
        fv = ds.var(factor).title
        bt = Table("Bootstrap for Multiple Comparisons",
                   ["Dependent Variable", f"(I) {fv}", f"(J) {fv}",
                    "Mean Difference (I-J)"] + heads, rows, row_header_cols=3,
                   footnotes=[rs.boot_footnote(spec, letter)])
        rs.place_after(section, "Multiple Comparisons", bt)
    return finish(section, spec)


# ============================================================================ Correlations
def _pearson_pairs(xb, pairs):
    out = []
    for i, j in pairs:
        ok = np.isfinite(xb[..., i]) & np.isfinite(xb[..., j])
        n = ok.sum(-1)
        with np.errstate(all="ignore"):
            a = np.where(ok, xb[..., i], 0.0)
            b = np.where(ok, xb[..., j], 0.0)
            ma = a.sum(-1) / n
            mb = b.sum(-1) / n
            ca = np.where(ok, a - ma[..., None], 0.0)
            cb = np.where(ok, b - mb[..., None], 0.0)
            out.append((ca * cb).sum(-1) / np.sqrt((ca * ca).sum(-1) * (cb * cb).sum(-1)))
    return np.stack(out, axis=-1)


def _rank_pairs(x, pairs, method):
    out = np.full(len(pairs), np.nan)
    for k, (i, j) in enumerate(pairs):
        ok = np.isfinite(x[:, i]) & np.isfinite(x[:, j])
        if ok.sum() < 3:
            continue
        a, b = x[ok, i], x[ok, j]
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            continue
        if method == "spearman":
            out[k] = stats.spearmanr(a, b).statistic
        else:
            out[k] = stats.kendalltau(a, b).statistic
    return out


def correlations(section: Section, ds: Dataset, names: list[str], spec: BootSpec,
                 pearson: bool = True, kendall: bool = False, spearman: bool = False,
                 listwise: bool = False) -> Section:
    frame = pd.concat({n: numeric(ds, n) for n in names}, axis=1)
    if listwise:
        frame = frame.dropna()
    X = frame.to_numpy(float)
    p = len(names)
    pairs = [(i, j) for i in range(p) for j in range(p) if i < j]
    pidx = {pr: k for k, pr in enumerate(pairs)}
    strata = strata_values(ds, spec, frame.index.to_numpy())
    results = {}
    if pearson:
        results["pearson"] = rs.bootstrap(X, lambda xb: _pearson_pairs(xb, pairs), spec, strata,
                                          vectorized=True)
    for method, on in (("kendall", kendall), ("spearman", spearman)):
        if on:
            results[method] = rs.bootstrap(X, lambda x, m=method: _rank_pairs(x, pairs, m),
                                           spec, strata)
    titles = [ds.var(n).title for n in names]
    for t_i in _tables(section, "Correlations"):
        table = section.items[t_i]
        h = table.row_header_cols
        mapping = {}
        method = "pearson" if h == 2 else None
        var = None
        for r, row in enumerate(table.rows):
            if h == 3 and row[0]:
                method = "kendall" if "Kendall" in row[0] else "spearman" \
                    if "Spearman" in row[0] else None
            if row[h - 2]:
                var = row[h - 2]
            if row[h - 1] != "N" or method not in results or var not in titles:
                continue
            i = titles.index(var)
            cells = {}
            for c in range(h, len(table.columns)):
                col = table.columns[c]
                if col not in titles or titles.index(col) == i:
                    continue
                j = titles.index(col)
                cells[c] = (results[method], pidx[(min(i, j), max(i, j))])
            mapping[r] = cells
        if mapping:
            _replace(section, t_i, rs.insert_boot_rows(table, mapping, spec, 3))
    return finish(section, spec)


def partial(section: Section, ds: Dataset, names: list[str], controls: list[str],
            spec: BootSpec) -> Section:
    allv = list(names) + list(controls)
    frame = pd.concat({n: numeric(ds, n) for n in dict.fromkeys(allv)}, axis=1).dropna()
    X = frame.to_numpy(float)
    p, q = len(names), len(controls)
    pairs = [(i, j) for i in range(p) for j in range(p) if i < j]
    pidx = {pr: k for k, pr in enumerate(pairs)}

    def stat(xb):
        c = xb - xb.mean(axis=1, keepdims=True)
        S = np.einsum("bni,bnj->bij", c, c)
        Svv, Svc, Scc = S[:, :p, :p], S[:, :p, p:], S[:, p:, p:]
        try:
            R = Svv - Svc @ np.linalg.solve(Scc, np.swapaxes(Svc, 1, 2))
        except np.linalg.LinAlgError:
            R = Svv - Svc @ np.linalg.pinv(Scc) @ np.swapaxes(Svc, 1, 2)
        d = np.sqrt(np.einsum("bii->bi", R))
        Rn = R / (d[:, :, None] * d[:, None, :])
        return np.stack([Rn[:, i, j] for i, j in pairs], axis=-1)

    res = rs.bootstrap(X, stat, spec, strata_values(ds, spec, frame.index.to_numpy()),
                       vectorized=True)
    titles = [ds.var(n).title for n in names]
    t_i = rs.find_table(section, "Correlations")
    if t_i is None or q == 0:
        return finish(section, spec)
    table = section.items[t_i]
    mapping = {}
    block_ok, var = False, None
    for r, row in enumerate(table.rows):
        if row[0]:
            block_ok = row[0] != "-none-ᵃ" and not row[0].startswith("-none-")
        if row[1]:
            var = row[1]
        if not block_ok or row[2] != "df" or var not in titles:
            continue
        i = titles.index(var)
        cells = {}
        for c in range(3, len(table.columns)):
            col = table.columns[c]
            if col in titles and titles.index(col) != i:
                j = titles.index(col)
                cells[c] = (res, pidx[(min(i, j), max(i, j))])
        mapping[r] = cells
    _replace(section, t_i, rs.insert_boot_rows(table, mapping, spec, 3))
    return finish(section, spec)


# ============================================================================ Crosstabs
_SYM = {"Phi": ("nom", "phi"), "Cramer's V": ("nom", "v"),
        "Contingency Coefficient": ("nom", "cc"), "Kendall's tau-b": ("ord", "btau"),
        "Kendall's tau-c": ("ord", "ctau"), "Gamma": ("ord", "gamma"),
        "Spearman Correlation": ("cor", "spearman"), "Pearson's R": ("cor", "pearson"),
        "Kappa": ("kappa", None)}


def crosstabs(section: Section, ds: Dataset, rows: list[str], cols: list[str],
              spec: BootSpec) -> Section:
    from rsstatchem.core.stats import crosstabs as ct

    sym = _tables(section, "Symmetric Measures")
    pairs = [(r, c) for r in rows for c in cols]
    if len(sym) != len(pairs):
        section.note("", "Bootstrap results are not available for Crosstabs with layers.")
        return finish(section, spec)
    for (rname, cname), t_i in zip(pairs, sym):
        table = section.items[t_i]
        names = [row[1] for row in table.rows]
        wanted = [nm for nm in names if nm in _SYM]
        if not wanted:
            continue
        fr = pd.DataFrame({"r": ds.analysis_series(rname),
                           "c": ds.analysis_series(cname)}).dropna()
        rc, cc = ct._categories(fr["r"]), ct._categories(fr["c"])
        rcode = fr["r"].map({v: i for i, v in enumerate(rc)}).to_numpy(int)
        ccode = fr["c"].map({v: i for i, v in enumerate(cc)}).to_numpy(int)
        rs_, cs_ = ct._scores(ds.var(rname), rc), ct._scores(ds.var(cname), cc)
        R, C = len(rc), len(cc)

        def measures(obs):
            out = []
            keep_r, keep_c = obs.sum(1) > 0, obs.sum(0) > 0
            o2 = obs[keep_r][:, keep_c]
            cache = {}
            for nm in wanted:
                kind, key = _SYM[nm]
                try:
                    if kind == "kappa":
                        out.append(ct.kappa(obs).value)
                        continue
                    if min(o2.shape) < 2:
                        out.append(np.nan)
                        continue
                    if kind not in cache:
                        cache[kind] = ct.nominal_measures(o2) if kind == "nom" else \
                            ct.ordinal_measures(o2) if kind == "ord" else \
                            ct.correlations(o2, rs_[keep_r], cs_[keep_c])
                    out.append(cache[kind][key].value)
                except (KeyError, ValueError, ZeroDivisionError, FloatingPointError):
                    out.append(np.nan)
            return out

        def evaluate(I):
            res = np.full((len(I), len(wanted)), np.nan)
            for b, ix in enumerate(I):
                obs = np.zeros((R, C))
                np.add.at(obs, (rcode[ix], ccode[ix]), 1)
                res[b] = measures(obs)
            return res

        obs0 = np.zeros((R, C))
        np.add.at(obs0, (rcode, ccode), 1)
        est = np.array(measures(obs0), float)
        with np.errstate(all="ignore"):
            res = rs.bootstrap_indices(len(fr), evaluate, est, spec,
                                       strata_values(ds, spec, fr.index.to_numpy()), batch=50)
        mapping = {r: (res, wanted.index(row[1])) for r, row in enumerate(table.rows)
                   if row[1] in wanted}
        _replace(section, t_i, rs.append_boot_columns(table, mapping, spec, 3, sig=False))
    return finish(section, spec)


# ============================================================================ models
def models(section: Section, ds: Dataset, infos, spec: BootSpec,
           title: str | None = None) -> Section:
    """Bootstrap every :class:`ModelInfo` and add one "Bootstrap for …" table."""
    infos = [m for m in (infos or []) if m is not None]
    usable = [m for m in infos if m.bootable and (m.refit or m.refit_batch)]
    notes = [m.note for m in infos if m.note]
    if not usable:
        section.note("Bootstrap", notes[0] if notes else
                     "Bootstrap results are not available for this model.", style="warning")
        return finish(section, spec)
    results = []
    for m in usable:
        strata = strata_values(ds, spec, m.rows) if m.rows is not None else None
        results.append(rs.bootstrap_model(m, spec, strata))
    table = rs.model_boot_table(usable, results, spec, title)
    rs.place_after(section, usable[0].table, table)
    for note in dict.fromkeys(notes):
        section.note("Bootstrap", note)
    return finish(section, spec)


def selftest() -> None:
    from rsstatchem.core.stats import descriptive, regression, ttest

    rng = np.random.default_rng(3)
    df = pd.DataFrame({"y": rng.normal(size=40), "x": rng.normal(size=40),
                       "g": np.repeat([1.0, 2.0], 20)})
    ds = Dataset(df)
    spec = BootSpec(60, 1)
    sec = descriptives(descriptive.descriptives(ds, ["y", "x"]), ds, ["y", "x"], spec)
    assert sec.items[1].title == "Bootstrap Specifications"
    sec = independent_t(ttest.independent(ds, ["y"], "g", (1.0, 2.0)), ds, ["y"], "g",
                        (1.0, 2.0), None, spec)
    assert rs.find_table(sec, "Bootstrap for Independent Samples Test") is not None
    res = regression.linear(ds, "y", ["x"], plots=(), assumptions=())
    models(res.section, ds, res.models, spec)
    assert rs.find_table(res.section, "Bootstrap for Coefficients") is not None

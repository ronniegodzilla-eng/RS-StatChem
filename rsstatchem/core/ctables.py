"""Custom Tables (SPSS Analyze > Tables > Custom Tables, CTABLES) without Qt.

A table has row and column dimensions (each a list of variables that are *nested* or
*stacked*) and optional layers. Categorical variables contribute their categories (plus an
optional Total); a scale variable contributes summary statistics. Multiple response sets
(``$name``, see :mod:`rsstatchem.core.mresponse`) behave as categorical variables whose cases
can fall into several categories; their percentages are based on respondents.

Statistics: Count, Column N %, Row N %, Table N %, Subtable N %, Layer N % for categories;
Mean, Median, Mode, Std. Deviation, Variance, Minimum, Maximum, Range, Sum, Count (N), Std.
Error of Mean, Percentile 25/75 for scale variables. Tests (as CTABLES /SIGTEST and
/COMPARETEST): Pearson chi-square independence, column proportions (z-tests) and column means
(t-tests with the pooled variance of all columns), Bonferroni-adjusted, reported with SPSS's
letter keys (A, B, C…) in the cell of the larger value.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sps

from rsstatchem.core import mresponse as mr
from rsstatchem.core.dataset import NUMERIC, SCALE, STRING, Dataset, format_code
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, fmt_p

CAT_STATS = {"count": "Count", "col_pct": "Column N %", "row_pct": "Row N %",
             "table_pct": "Table N %", "subtable_pct": "Subtable N %",
             "layer_pct": "Layer N %"}
SCALE_STATS = {"mean": "Mean", "median": "Median", "mode": "Mode", "std": "Std. Deviation",
               "var": "Variance", "min": "Minimum", "max": "Maximum", "range": "Range",
               "sum": "Sum", "n": "Count", "se": "Std. Error of Mean",
               "p25": "Percentile 25", "p75": "Percentile 75"}
ORDERS = ("ascending", "descending", "count_desc", "count_asc", "label")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class CTableError(ValueError):
    pass


@dataclass
class CTVar:
    name: str
    scale: bool | None = None          # None: from the measurement level
    totals: bool | None = None         # None: the table-wide setting
    order: str = "ascending"
    show_empty: bool = True            # labelled categories without cases


@dataclass
class CTSpec:
    rows: list = field(default_factory=list)
    cols: list = field(default_factory=list)
    layers: list = field(default_factory=list)
    row_mode: str = "nest"             # "nest" | "stack"
    col_mode: str = "nest"
    cat_stats: tuple = ("count",)
    scale_stats: tuple = ("mean",)
    totals: bool = False
    chisq: bool = False
    compare_props: bool = False
    compare_means: bool = False
    alpha: float = 0.05
    adjust: str = "bonferroni"         # "bonferroni" | "none"
    title: str = "Custom Table"
    decimals: int | None = None        # scale statistics


@dataclass
class Category:
    code: object
    label: str
    mask: np.ndarray                   # bool per case
    total: bool = False


@dataclass
class Element:
    name: str
    label: str
    scale: bool
    valid: np.ndarray
    categories: list = field(default_factory=list)
    values: np.ndarray | None = None   # scale variable data
    mr: bool = False


def _cv(item) -> CTVar:
    return item if isinstance(item, CTVar) else CTVar(str(item))


def _element(ds: Dataset, item: CTVar, spec: CTSpec) -> Element:
    totals = spec.totals if item.totals is None else item.totals
    if mr.is_set_name(ds, item.name):
        ind = mr.indicators(ds, mr.get_set(ds, item.name))
        valid = ind.valid.to_numpy()
        cats = [Category(code, label, (ind.hits.iloc[:, k].to_numpy() & valid))
                for k, (code, label) in enumerate(zip(ind.codes, ind.labels))]
        cats = _order(cats, item.order)
        if totals:
            cats.append(Category(None, "Total", valid.copy(), total=True))
        return Element(ind.mrset.name, ind.mrset.title, False, valid, cats, mr=True)
    var = ds.var(item.name)
    s = ds.analysis_series(var.name)
    scale = item.scale if item.scale is not None else (var.measure == SCALE
                                                       and var.type == NUMERIC)
    if scale:
        values = pd.to_numeric(s, errors="coerce").to_numpy(float)
        return Element(var.name, var.title, True, ~np.isnan(values), values=values)
    valid = s.notna().to_numpy()
    codes = [c for c in pd.unique(s) if not pd.isna(c)]
    if item.show_empty:
        codes += [c for c in var.value_labels if c not in codes and c not in var.missing]
    try:
        codes = sorted(codes)
    except TypeError:
        codes = sorted(codes, key=str)
    arr = s.to_numpy(dtype=object)
    cats = [Category(c, var.label_for(c) or format_code(c), (arr == c) & valid) for c in codes]
    cats = _order(cats, item.order)
    if totals:
        cats.append(Category(None, "Total", valid.copy(), total=True))
    return Element(var.name, var.title, False, valid, cats)


def _order(cats: list[Category], order: str) -> list[Category]:
    if order == "descending":
        return cats[::-1]
    if order == "count_desc":
        return sorted(cats, key=lambda c: -int(c.mask.sum()))
    if order == "count_asc":
        return sorted(cats, key=lambda c: int(c.mask.sum()))
    if order == "label":
        return sorted(cats, key=lambda c: c.label.lower())
    return cats


# ------------------------------------------------------------------ keys
@dataclass
class Key:
    """One row (or column) of the table: categories of the elements in a nesting chain."""
    elements: list            # Element per level
    cats: list                # Category per level (None for a scale element)

    @property
    def mask(self) -> np.ndarray:
        m = None
        for el, cat in zip(self.elements, self.cats):
            part = el.valid if cat is None else cat.mask
            m = part if m is None else (m & part)
        return m

    def outer_mask(self) -> np.ndarray | None:
        """Mask of all levels but the innermost, intersected with the innermost's valid."""
        m = self.elements[-1].valid
        for el, cat in zip(self.elements[:-1], self.cats[:-1]):
            m = m & (el.valid if cat is None else cat.mask)
        return m

    @property
    def scale_element(self) -> Element | None:
        return next((e for e in self.elements if e.scale), None)

    def labels(self) -> tuple:
        out = []
        for el, cat in zip(self.elements, self.cats):
            out.append(el.label if cat is None else f"{el.label}={cat.label}")
        return tuple(out)


def _keys(elements: list[Element], mode: str) -> list[Key]:
    if not elements:
        return []
    if mode == "stack":
        chains = [[e] for e in elements]
    else:
        scale = [e for e in elements if e.scale]
        if len(scale) > 1 or (scale and elements[-1] is not scale[0]):
            raise CTableError("A scale variable can only be the innermost nested variable "
                              "(stack scale variables instead).")
        chains = [elements]
    keys = []
    for chain in chains:
        levels = [[None] if e.scale else e.categories for e in chain]
        for combo in itertools.product(*levels):
            keys.append(Key(list(chain), list(combo)))
    return keys


# ------------------------------------------------------------------ statistics
def _scale_stat(x: np.ndarray, stat: str) -> float:
    x = x[~np.isnan(x)]
    n = len(x)
    if stat == "n":
        return float(n)
    if n == 0:
        return float("nan")
    if stat == "mean":
        return float(x.mean())
    if stat == "median":
        return float(np.median(x))
    if stat == "mode":
        vals, counts = np.unique(x, return_counts=True)
        return float(vals[np.argmax(counts)])
    if stat == "sum":
        return float(x.sum())
    if stat == "min":
        return float(x.min())
    if stat == "max":
        return float(x.max())
    if stat == "range":
        return float(x.max() - x.min())
    if stat in ("std", "var", "se"):
        if n < 2:
            return float("nan")
        v = float(x.var(ddof=1))
        return {"var": v, "std": math.sqrt(v), "se": math.sqrt(v / n)}[stat]
    if stat in ("p25", "p75"):
        from rsstatchem.core.stats.common import percentile
        return float(percentile(x, 25 if stat == "p25" else 75))
    raise CTableError(f"Unknown statistic {stat}.")


@dataclass
class CTResult:
    title: str
    row_keys: list
    col_keys: list
    stats: list                          # (stat key, label) along the columns
    values: dict = field(default_factory=dict)   # (ri, ci, stat) -> float
    letters: dict = field(default_factory=dict)  # (ri, ci) -> "B C"
    col_letters: dict = field(default_factory=dict)  # ci -> "A"
    chisq: list = field(default_factory=list)
    footnotes: list = field(default_factory=list)
    layer_label: str = ""

    def value(self, row_labels: tuple, col_labels: tuple, stat: str) -> float:
        for ri, rk in enumerate(self.row_keys):
            if rk.labels() != tuple(row_labels):
                continue
            for ci, ck in enumerate(self.col_keys or [None]):
                if (ck.labels() if ck else ()) == tuple(col_labels):
                    return self.values[(ri, ci, stat)]
        raise KeyError((row_labels, col_labels, stat))


def _cell_stats(spec: CTSpec, rk: Key, ck: Key | None) -> list[str]:
    scale = rk.scale_element or (ck.scale_element if ck else None)
    return list(spec.scale_stats) if scale is not None else list(spec.cat_stats)


def compute(ds: Dataset, spec: CTSpec, layer_mask: np.ndarray | None = None,
            layer_label: str = "") -> CTResult:
    row_el = [_element(ds, _cv(v), spec) for v in spec.rows]
    col_el = [_element(ds, _cv(v), spec) for v in spec.cols]
    if not row_el and not col_el:
        raise CTableError("Put at least one variable in Rows or Columns.")
    if not row_el:  # SPSS allows column-only tables; treat them as one-row tables
        row_el, col_el = col_el, []
        spec = CTSpec(**{**spec.__dict__, "rows": spec.cols, "cols": [],
                         "row_mode": spec.col_mode})
    row_keys = _keys(row_el, spec.row_mode)
    col_keys = _keys(col_el, spec.col_mode)
    n = ds.n_cases
    base = np.ones(n, dtype=bool) if layer_mask is None else layer_mask
    stat_keys = list(dict.fromkeys(list(spec.cat_stats) + list(spec.scale_stats)))
    res = CTResult(spec.title + (f" ({layer_label})" if layer_label else ""), row_keys,
                   col_keys, [], layer_label=layer_label)
    used_stats: list[str] = []
    for ri, rk in enumerate(row_keys):
        rmask = rk.mask & base
        for ci, ck in enumerate(col_keys or [None]):
            cmask = ck.mask if ck is not None else np.ones(n, dtype=bool)
            cell = rmask & cmask
            scale = rk.scale_element or (ck.scale_element if ck else None)
            for stat in _cell_stats(spec, rk, ck):
                if stat not in used_stats:
                    used_stats.append(stat)
                if scale is not None:
                    value = _scale_stat(scale.values[cell], stat)
                else:
                    value = _cat_stat(stat, cell, rk, ck, base)
                res.values[(ri, ci, stat)] = value
    order = [s for s in stat_keys if s in used_stats]
    res.stats = [(s, CAT_STATS.get(s) or SCALE_STATS.get(s, s)) for s in order]
    if spec.chisq:
        res.chisq = _chisq(row_keys, col_keys, base)
    if spec.compare_props or spec.compare_means:
        _compare(res, spec, base)
    return res


def _cat_stat(stat, cell, rk: Key, ck: Key | None, base) -> float:
    count = float(cell.sum())
    if stat == "count":
        return count
    if stat == "col_pct":
        denom = rk.outer_mask() & base & (ck.mask if ck is not None else True)
    elif stat == "row_pct":
        denom = rk.mask & base & (ck.outer_mask() if ck is not None else True)
    elif stat == "subtable_pct":
        denom = rk.outer_mask() & base & (ck.outer_mask() if ck is not None else True)
    else:  # table / layer: cases valid on every variable of this subtable
        denom = base.copy()
        for e in rk.elements + (ck.elements if ck is not None else []):
            denom &= e.valid
    d = float(np.sum(denom))
    return 100.0 * count / d if d else float("nan")


def _chisq(row_keys: list[Key], col_keys: list[Key], base) -> list[tuple]:
    """Pearson chi-square for each innermost row x innermost column variable pair,
    within each combination of outer categories (totals excluded)."""
    out = []
    groups: dict = {}
    for ri, rk in enumerate(row_keys):
        if rk.scale_element is not None or rk.cats[-1].total or rk.elements[-1].mr:
            continue
        outer = (tuple(e.name for e in rk.elements),
                 tuple(id(c) for c in rk.cats[:-1]))
        groups.setdefault(outer, []).append(rk)
    cgroups: dict = {}
    for ck in col_keys:
        if ck.scale_element is not None or ck.cats[-1].total or ck.elements[-1].mr:
            continue
        outer = (tuple(e.name for e in ck.elements), tuple(id(c) for c in ck.cats[:-1]))
        cgroups.setdefault(outer, []).append(ck)
    for rkeys in groups.values():
        for ckeys in cgroups.values():
            table = np.array([[float(np.sum(rk.mask & ck.mask & base)) for ck in ckeys]
                              for rk in rkeys])
            keep_r = table.sum(axis=1) > 0
            keep_c = table.sum(axis=0) > 0
            t = table[keep_r][:, keep_c]
            if t.shape[0] < 2 or t.shape[1] < 2:
                continue
            chi2, p, dof, expected = sps.chi2_contingency(t, correction=False)
            small = float(np.mean(expected < 5))
            rlab = " / ".join(rkeys[0].labels()[:-1] + (rkeys[0].elements[-1].label,))
            clab = " / ".join(ckeys[0].labels()[:-1] + (ckeys[0].elements[-1].label,))
            out.append((rlab, clab, chi2, dof, p, small))
    return out


def _compare(res: CTResult, spec: CTSpec, base) -> None:
    """Column proportions / means comparisons within each innermost column variable."""
    blocks: dict = {}
    for ci, ck in enumerate(res.col_keys):
        if ck.scale_element is not None or ck.elements[-1].mr:
            continue
        outer = (tuple(e.name for e in ck.elements), tuple(id(c) for c in ck.cats[:-1]))
        blocks.setdefault(outer, []).append(ci)
    for cis in blocks.values():
        members = [ci for ci in cis if not res.col_keys[ci].cats[-1].total]
        for k, ci in enumerate(members):
            res.col_letters[ci] = LETTERS[k % 26]
        if len(members) < 2:
            continue
        pairs = list(itertools.combinations(members, 2))
        factor = len(pairs) if spec.adjust == "bonferroni" else 1
        for ri, rk in enumerate(res.row_keys):
            scale = rk.scale_element
            if scale is None and spec.compare_props and not rk.cats[-1].total:
                _prop_tests(res, ri, rk, pairs, factor, spec.alpha, base)
            elif scale is not None and spec.compare_means:
                _mean_tests(res, ri, rk, members, pairs, factor, spec.alpha, base, scale)
    notes = []
    if spec.compare_props or spec.compare_means:
        kinds = " and ".join(k for k, on in (("proportions", spec.compare_props),
                                             ("means", spec.compare_means)) if on)
        notes.append(f"Comparisons of column {kinds}: results are based on two-sided tests "
                     f"with significance level {fmt(spec.alpha, 2)}. For each significant "
                     "pair, the key of the smaller category appears in the cell of the "
                     "larger category.")
        if spec.adjust == "bonferroni":
            notes.append("Tests are adjusted for all pairwise comparisons within a row of "
                         "each innermost subtable using the Bonferroni correction.")
        if spec.compare_means:
            notes.append("Column means tests assume equal variances (pooled over all "
                         "columns of the subtable).")
    res.footnotes.extend(notes)


def _add_letter(res: CTResult, ri: int, ci: int, letter: str) -> None:
    current = res.letters.get((ri, ci), "")
    res.letters[(ri, ci)] = " ".join(sorted(set(current.split()) | {letter}))


def _prop_tests(res, ri, rk: Key, pairs, factor, alpha, base) -> None:
    rmask = rk.mask & base
    outer = rk.outer_mask() & base
    for a, b in pairs:
        ma, mb = res.col_keys[a].mask, res.col_keys[b].mask
        x1, n1 = float(np.sum(rmask & ma)), float(np.sum(outer & ma))
        x2, n2 = float(np.sum(rmask & mb)), float(np.sum(outer & mb))
        if n1 == 0 or n2 == 0:
            continue
        p1, p2 = x1 / n1, x2 / n2
        if p1 in (0.0, 1.0) and p2 in (0.0, 1.0):
            continue  # SPSS: no test when both proportions are 0 or 1
        pooled = (x1 + x2) / (n1 + n2)
        se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
        if se == 0:
            continue
        z = (p1 - p2) / se
        p = min(1.0, 2 * sps.norm.sf(abs(z)) * factor)
        if p < alpha:
            big, small = (a, b) if p1 > p2 else (b, a)
            _add_letter(res, ri, big, res.col_letters[small])


def _mean_tests(res, ri, rk: Key, members, pairs, factor, alpha, base, scale) -> None:
    rmask = rk.mask & base
    data = {ci: scale.values[rmask & res.col_keys[ci].mask] for ci in members}
    data = {ci: x[~np.isnan(x)] for ci, x in data.items()}
    used = [x for x in data.values() if len(x) > 0]
    n_total = sum(len(x) for x in used)
    df = n_total - len(used)
    if df <= 0:
        return
    sse = sum(float(((x - x.mean()) ** 2).sum()) for x in used)
    mse = sse / df
    for a, b in pairs:
        xa, xb = data[a], data[b]
        if len(xa) == 0 or len(xb) == 0 or mse <= 0:
            continue
        t = (xa.mean() - xb.mean()) / math.sqrt(mse * (1 / len(xa) + 1 / len(xb)))
        p = min(1.0, 2 * sps.t.sf(abs(t), df) * factor)
        if p < alpha:
            big, small = (a, b) if xa.mean() > xb.mean() else (b, a)
            _add_letter(res, ri, big, res.col_letters[small])


# ------------------------------------------------------------------ output table
def _format(stat: str, value: float, spec: CTSpec, scale_el: Element | None,
            ds: Dataset) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if stat in ("count", "n"):
        return fmt_int(value)
    if stat in CAT_STATS:
        return f"{value:.1f}%"
    decimals = spec.decimals
    if decimals is None:
        decimals = 2
        if scale_el is not None and ds.has_name(scale_el.name):
            decimals = min(ds.var(scale_el.name).decimals + 2, 4)
    return f"{value:.{decimals}f}"


def to_table(ds: Dataset, spec: CTSpec, res: CTResult) -> Table:
    depth = max(len(k.elements) for k in res.row_keys)
    rh = 2 * depth
    multi_stat = len(res.stats) > 1
    columns: list = [""] * rh
    col_specs = []   # (ci, stat)
    for ci, ck in enumerate(res.col_keys or [None]):
        stats_here = [s for s, _ in res.stats
                      if any((ri, ci, s) in res.values for ri in range(len(res.row_keys)))]
        for s in stats_here:
            col_specs.append((ci, s))
            label = dict(res.stats)[s]
            if ck is None:
                columns.append(label)
                continue
            outer = [f"{e.label}: {c.label}" if c is not None else e.label
                     for e, c in zip(ck.elements[:-1], ck.cats[:-1])]
            inner_el, inner_cat = ck.elements[-1], ck.cats[-1]
            inner = inner_el.label if inner_cat is None else inner_cat.label
            if ci in res.col_letters:
                inner += f" ({res.col_letters[ci]})"
            if multi_stat or inner_cat is None:
                group = " › ".join(outer + ([inner_el.label + ": " + inner]
                                            if inner_cat is not None else [inner]))
                columns.append((group, label))
            else:
                group = " › ".join(outer + [inner_el.label])
                columns.append((group, inner))
    rows = []
    previous: list = []
    for ri, rk in enumerate(res.row_keys):
        heads = []
        for level in range(depth):
            if level < len(rk.elements):
                el, cat = rk.elements[level], rk.cats[level]
                heads += [el.label, "" if cat is None else cat.label]
            else:
                heads += ["", ""]
        shown = list(heads)
        for level in range(depth):  # print an outer label once per block
            same = len(previous) == len(heads) and \
                previous[:2 * level + 2] == heads[:2 * level + 2]
            if same and level < depth - 1:
                shown[2 * level] = shown[2 * level + 1] = ""
            elif len(previous) == len(heads) and previous[:2 * level + 1] == \
                    heads[:2 * level + 1]:
                shown[2 * level] = ""
        previous = heads
        scale_el = rk.scale_element
        cells = []
        for ci, stat in col_specs:
            ck = res.col_keys[ci] if res.col_keys else None
            el = scale_el or (ck.scale_element if ck else None)
            value = res.values.get((ri, ci, stat))
            text = "" if value is None else _format(stat, value, spec, el, ds)
            first_stat = stat == next((s for c2, s in col_specs if c2 == ci), None)
            letters = res.letters.get((ri, ci))
            if letters and first_stat and text:
                text += f" {letters}"
            cells.append(text)
        rows.append(shown + cells)
    notes = list(res.footnotes)
    if not multi_stat and res.stats and res.col_keys:
        notes.insert(0, f"Cells show: {res.stats[0][1]}.")
    if any(e.mr for k in res.row_keys + res.col_keys for e in k.elements):
        notes.append("Multiple response sets: percentages are based on respondents.")
    return Table(res.title, columns, rows, row_header_cols=rh, footnotes=notes)


def _chisq_table(res: CTResult) -> Table:
    rows = []
    for rlab, clab, chi2, dof, p, small in res.chisq:
        rows.append([rlab, clab, fmt(chi2, 3), fmt_int(dof), fmt_p(p)])
    notes = []
    if any(r[5] > 0.2 for r in res.chisq):
        notes.append("More than 20% of cells in at least one subtable have expected cell "
                     "counts less than 5. Chi-square results may be invalid.")
    title = "Pearson Chi-Square Tests" + (f" ({res.layer_label})" if res.layer_label else "")
    return Table(title, ["", "", "Chi-square", "df", "Sig."], rows, row_header_cols=2,
                 footnotes=notes or ["Results are based on nonempty rows and columns in each "
                                     "innermost subtable."])


def _syntax(spec: CTSpec) -> str:
    def side(items, mode):
        out = []
        for it in items:
            v = _cv(it)
            out.append(f"{v.name} [{'S' if v.scale else 'C'}]")
        return (" > " if mode == "nest" else " + ").join(out)

    lines = [f"CTABLES\n  /TABLE {side(spec.rows, spec.row_mode)}"
             + (f" BY {side(spec.cols, spec.col_mode)}" if spec.cols else "")
             + (f" BY {side(spec.layers, 'stack')}" if spec.layers else "")]
    lines.append("  /SUMMARY " + " ".join(s.upper() for s in spec.cat_stats + spec.scale_stats))
    if spec.totals:
        lines.append("  /CATEGORIES VARIABLES=ALL TOTAL=YES")
    if spec.chisq:
        lines.append(f"  /SIGTEST TYPE=CHISQUARE ALPHA={spec.alpha}")
    if spec.compare_props:
        lines.append(f"  /COMPARETEST TYPE=PROP ALPHA={spec.alpha} ADJUST={spec.adjust.upper()}")
    if spec.compare_means:
        lines.append(f"  /COMPARETEST TYPE=MEAN ALPHA={spec.alpha} ADJUST={spec.adjust.upper()}")
    return "\n".join(lines) + "."


def custom_tables(ds: Dataset, spec: CTSpec) -> Section:
    section = Section("Custom Tables")
    section.log(_syntax(spec))
    layer_specs = [(None, "")]
    if spec.layers:
        layer_el = [_element(ds, _cv(v), CTSpec(totals=False)) for v in spec.layers]
        layer_specs = []
        levels = [[(e, c) for c in e.categories] for e in layer_el if not e.scale]
        for combo in itertools.product(*levels):
            mask = np.ones(ds.n_cases, dtype=bool)
            for e, c in combo:
                mask &= c.mask
            label = ", ".join(f"{e.label} = {c.label}" for e, c in combo)
            layer_specs.append((mask, label))
    for mask, label in layer_specs:
        res = compute(ds, spec, mask, label)
        section.table(to_table(ds, spec, res))
        if res.chisq:
            section.table(_chisq_table(res))
    return section


def measure_of(ds: Dataset, name: str) -> str:
    """"scale", "categorical" or "mrset" (for the builder's icons and defaults)."""
    if mr.is_set_name(ds, name):
        return "mrset"
    var = ds.var(name)
    return "scale" if var.measure == SCALE and var.type == NUMERIC else "categorical"


def available_names(ds: Dataset) -> list[str]:
    return [v.name for v in ds.variables if v.type in (NUMERIC, STRING)] + \
        [s.name for s in mr.get_sets(ds)]

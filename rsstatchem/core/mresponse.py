"""Multiple response sets and the MULT RESPONSE procedures (SPSS Analyze > Multiple Response).

A set groups several variables that together record one question with more than one answer:

* **multiple dichotomy** set: one variable per option; an option is chosen when the variable
  has the counted value (e.g. 1 = "ticked");
* **multiple category** set: each variable holds one answer code (e.g. "first choice",
  "second choice"); the categories are the codes, optionally limited to a range.

Sets are stored with the data in ``ds.meta["mrsets"]`` (like SPSS MRSETS), so Custom Tables
can use them too. A case is valid for a set when it has at least one counted value
(dichotomies) or one valid code (categories). A category is counted once per case.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from rsstatchem.core.dataset import NUMERIC, Dataset, format_code
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int


class MRError(ValueError):
    pass


@dataclass
class MRSet:
    name: str                      # starts with "$"
    label: str = ""
    kind: str = "dichotomy"        # "dichotomy" | "category"
    variables: list = field(default_factory=list)
    counted: object = 1.0          # dichotomy: the counted value
    range: list | None = None      # category: [low, high] (None = all valid codes)
    category_labels: str = "varlabels"   # dichotomy: "varlabels" or "countedlabels"

    @property
    def title(self) -> str:
        return self.label or self.name

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MRSet":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def get_sets(ds: Dataset) -> list[MRSet]:
    return [MRSet.from_dict(d) for d in ds.meta.get("mrsets", [])]


def get_set(ds: Dataset, name: str) -> MRSet:
    for s in get_sets(ds):
        if s.name.lower() == name.lower():
            return s
    raise MRError(f"Unknown multiple response set {name}.")


def is_set_name(ds: Dataset, name: str) -> bool:
    return name.startswith("$") and any(s.name.lower() == name.lower() for s in get_sets(ds))


def validate(ds: Dataset, mrset: MRSet, others: list[MRSet] = ()) -> str | None:
    name = mrset.name
    if not name.startswith("$") or len(name) < 2 or not name[1:].replace("_", "a").isalnum():
        return "A set name starts with $ followed by letters, digits or _ (e.g. $media)."
    if any(o.name.lower() == name.lower() for o in others):
        return f"A set named {name} already exists."
    if len(mrset.variables) < 2:
        return "A set needs at least two variables."
    for v in mrset.variables:
        if not ds.has_name(v):
            return f"Unknown variable {v}."
    types = {ds.var(v).type for v in mrset.variables}
    if len(types) > 1:
        return "All variables of a set must be numeric, or all string."
    if mrset.kind == "category" and mrset.range is not None:
        lo, hi = mrset.range
        if lo is None or hi is None or float(lo) > float(hi):
            return "The category range needs a low value not above the high value."
    return None


def save_sets(ds: Dataset, sets: list[MRSet]) -> None:
    ds.meta["mrsets"] = [s.to_dict() for s in sets]
    ds.modified = True


# ------------------------------------------------------------------ indicators
@dataclass
class Indicators:
    mrset: MRSet
    codes: list            # category identifiers (variable names or codes)
    labels: list[str]
    hits: pd.DataFrame     # bool, cases x categories
    valid: pd.Series       # bool, case counted for the set

    @property
    def responses(self) -> pd.Series:
        return self.hits.sum(axis=0)


def _counted_equal(series: pd.Series, value) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        try:
            return series == float(value)
        except (TypeError, ValueError):
            return pd.Series(False, index=series.index)
    return series.astype(object) == str(value)


def indicators(ds: Dataset, mrset: MRSet) -> Indicators:
    cols = [ds.analysis_series(v) for v in mrset.variables]
    index = ds.df.index
    if mrset.kind == "dichotomy":
        hits = pd.DataFrame({v: _counted_equal(c, mrset.counted).fillna(False).to_numpy()
                             for v, c in zip(mrset.variables, cols)}, index=index)
        labels = []
        for v in mrset.variables:
            var = ds.var(v)
            if mrset.category_labels == "countedlabels":
                key = float(mrset.counted) if var.type == NUMERIC else str(mrset.counted)
                labels.append(var.value_labels.get(key) or var.title)
            else:
                labels.append(var.title)
        valid = hits.any(axis=1)
        return Indicators(mrset, list(mrset.variables), labels, hits, valid)
    values = pd.concat(cols, axis=1, ignore_index=True)
    if mrset.range is not None:
        lo, hi = float(mrset.range[0]), float(mrset.range[1])
        numeric = values.apply(pd.to_numeric, errors="coerce")
        values = values.where((numeric >= lo) & (numeric <= hi))
    codes = pd.unique(values.to_numpy().ravel())
    codes = [c for c in codes if not pd.isna(c)]
    try:
        codes = sorted(codes)
    except TypeError:
        codes = sorted(codes, key=str)
    hits = pd.DataFrame({i: (values == c).any(axis=1).to_numpy() for i, c in enumerate(codes)},
                        index=index)
    hits.columns = range(len(codes))
    labels = []
    for c in codes:
        label = None
        for v in mrset.variables:
            label = ds.var(v).label_for(c)
            if label:
                break
        labels.append(label or format_code(c))
    valid = values.notna().any(axis=1)
    return Indicators(mrset, codes, labels, hits, valid)


def _case_summary(ind_list: list[Indicators], n_total: int, title: str = "Case Summary",
                  names: list[str] | None = None) -> Table:
    rows = []
    for i, ind in enumerate(ind_list):
        valid = int(ind.valid.sum())
        missing = n_total - valid
        name = names[i] if names else ind.mrset.name
        rows.append([name, fmt_int(valid), fmt(100 * valid / n_total if n_total else np.nan, 1)
                     + "%", fmt_int(missing),
                     fmt(100 * missing / n_total if n_total else np.nan, 1) + "%",
                     fmt_int(n_total), "100.0%"])
    cols = ["", ("Cases", "Valid N"), ("Cases", "Valid Percent"), ("Cases", "Missing N"),
            ("Cases", "Missing Percent"), ("Cases", "Total N"), ("Cases", "Total Percent")]
    return Table(title, cols, rows)


def _footnote(mrset: MRSet) -> str:
    if mrset.kind == "dichotomy":
        return f"a. Dichotomy group tabulated at value {format_code(mrset.counted)}."
    return "a. Group of category variables." if mrset.range is None else \
        f"a. Categories {format_code(mrset.range[0])} to {format_code(mrset.range[1])}."


# ------------------------------------------------------------------ frequencies
def frequencies(ds: Dataset, set_names: list[str]) -> Section:
    """MULT RESPONSE /FREQUENCIES: responses, percent of responses, percent of cases."""
    section = Section("Multiple Response")
    sets = [get_set(ds, n) for n in set_names]
    section.log("MULT RESPONSE GROUPS=" + " ".join(
        f"{s.name} '{s.label}' ({' '.join(s.variables)} "
        + (f"({format_code(s.counted)}))" if s.kind == "dichotomy" else
           (f"({format_code(s.range[0])},{format_code(s.range[1])}))" if s.range else "))"))
        for s in sets) + "\n  /FREQUENCIES=" + " ".join(s.name for s in sets) + ".")
    inds = [indicators(ds, s) for s in sets]
    n = ds.n_cases
    section.table(_case_summary(inds, n))
    for ind in inds:
        resp = ind.hits[ind.valid].sum(axis=0)
        total_resp = int(resp.sum())
        valid = int(ind.valid.sum())
        rows = []
        for k, (code, label) in enumerate(zip(ind.codes, ind.labels)):
            count = int(resp.iloc[k])
            rows.append([f"{ind.mrset.title}ᵃ" if k == 0 else "", label, fmt_int(count),
                         fmt(100 * count / total_resp if total_resp else np.nan, 1) + "%",
                         fmt(100 * count / valid if valid else np.nan, 1) + "%"])
        rows.append(["Total", "", fmt_int(total_resp), "100.0%" if total_resp else "",
                     fmt(100 * total_resp / valid if valid else np.nan, 1) + "%"])
        section.table(Table(f"{ind.mrset.name} Frequencies",
                            ["", "", ("Responses", "N"), ("Responses", "Percent"),
                             "Percent of Cases"], rows, row_header_cols=2,
                            footnotes=[_footnote(ind.mrset)]))
    return section


def frequency_counts(ds: Dataset, name: str) -> pd.DataFrame:
    """Responses N, percent of responses and percent of cases (for tests/other modules)."""
    ind = indicators(ds, get_set(ds, name))
    resp = ind.hits[ind.valid].sum(axis=0).astype(int).to_numpy()
    valid = int(ind.valid.sum())
    return pd.DataFrame({"label": ind.labels, "n": resp,
                         "pct_responses": 100 * resp / resp.sum() if resp.sum() else np.nan,
                         "pct_cases": 100 * resp / valid if valid else np.nan},
                        index=ind.codes)


# ------------------------------------------------------------------ crosstabs
def _dimension(ds: Dataset, name: str) -> Indicators:
    """A regular variable seen as a (single response) set, so crosstabs treat both alike."""
    if is_set_name(ds, name):
        return indicators(ds, get_set(ds, name))
    var = ds.var(name)
    s = ds.analysis_series(name)
    codes = [c for c in pd.unique(s) if not pd.isna(c)]
    try:
        codes = sorted(codes)
    except TypeError:
        codes = sorted(codes, key=str)
    hits = pd.DataFrame({i: (s == c).to_numpy() for i, c in enumerate(codes)}, index=s.index)
    labels = [var.label_for(c) or format_code(c) for c in codes]
    pseudo = MRSet(var.name, var.title, "single", [var.name])
    return Indicators(pseudo, codes, labels, hits, s.notna())


def crosstab_counts(ds: Dataset, row: str, col: str, layer_mask: pd.Series | None = None
                    ) -> tuple[Indicators, Indicators, np.ndarray, pd.Series]:
    r, c = _dimension(ds, row), _dimension(ds, col)
    valid = r.valid & c.valid
    if layer_mask is not None:
        valid &= layer_mask
    rh = r.hits[valid].to_numpy(dtype=float)
    ch = c.hits[valid].to_numpy(dtype=float)
    counts = rh.T @ ch
    return r, c, counts, valid


def crosstabs(ds: Dataset, rows: list[str], cols: list[str], layers: list[str] | None = None,
              row_pct: bool = False, col_pct: bool = False, total_pct: bool = False,
              base: str = "cases") -> Section:
    """MULT RESPONSE /TABLES: sets and/or variables crossed; percentages and totals based on
    cases (respondents) or on responses."""
    section = Section("Multiple Response Crosstabs")
    cells = ["COUNT"] + (["ROW"] if row_pct else []) + (["COLUMN"] if col_pct else []) + \
        (["TOTAL"] if total_pct else [])
    section.log("MULT RESPONSE\n  /TABLES=" + " ".join(rows) + " BY " + " ".join(cols)
                + (" BY " + " ".join(layers) if layers else "")
                + f"\n  /CELLS={' '.join(cells)}\n  /BASE={base.upper()}.")
    layer_specs = [(None, None)]
    if layers:
        layer = _dimension(ds, layers[0])
        layer_specs = [(f"{layer.mrset.title} = {lab}", layer.hits.iloc[:, k] & layer.valid)
                       for k, lab in enumerate(layer.labels)]
    for rname in rows:
        for cname in cols:
            for layer_title, mask in layer_specs:
                section.table(_crosstab_table(ds, rname, cname, mask, layer_title, row_pct,
                                              col_pct, total_pct, base))
    return section


def _crosstab_table(ds, rname, cname, mask, layer_title, row_pct, col_pct, total_pct, base):
    r, c, counts, valid = crosstab_counts(ds, rname, cname, mask)
    n_cases = int(valid.sum())
    rh = r.hits[valid].to_numpy(dtype=float)
    ch = c.hits[valid].to_numpy(dtype=float)
    if base == "responses":
        row_tot = counts.sum(axis=1)
        col_tot = counts.sum(axis=0)
        grand = counts.sum()
    else:
        # respondents: a case counts once in a row total even if it has several columns
        row_tot = (rh * (ch.sum(axis=1, keepdims=True) > 0)).sum(axis=0)
        col_tot = (ch * (rh.sum(axis=1, keepdims=True) > 0)).sum(axis=0)
        grand = n_cases
    stats = [("Count", None)]
    if row_pct:
        stats.append((f"% within {r.mrset.title}", "row"))
    if col_pct:
        stats.append((f"% within {c.mrset.title}", "col"))
    if total_pct:
        stats.append(("% of Total", "total"))
    rows = []
    for i, label in enumerate(r.labels):
        for j, (sname, kind) in enumerate(stats):
            values = []
            for k in range(len(c.labels)):
                values.append(_cell(counts[i, k], kind, row_tot[i], col_tot[k], grand))
            values.append(_cell(row_tot[i], kind, row_tot[i], grand, grand, total=True))
            head = f"{r.mrset.title}" + ("ᵃ" if r.mrset.kind != "single" else "") \
                if i == 0 and j == 0 else ""
            rows.append([head, label if j == 0 else "", sname] + values)
    for j, (sname, kind) in enumerate(stats):
        values = [_cell(col_tot[k], kind, grand, col_tot[k], grand, total=True)
                  for k in range(len(c.labels))]
        values.append(_cell(grand, kind, grand, grand, grand, total=True))
        rows.append(["Total" if j == 0 else "", "", sname] + values)
    columns = ["", "", ""] + [(c.mrset.title, lab) for lab in c.labels] + ["Total"]
    title = f"{r.mrset.name}*{c.mrset.name} Crosstabulation"
    if layer_title:
        title += f" ({layer_title})"
    based = "respondents" if base == "cases" else "responses"
    notes = [f"Percentages and totals are based on {based}."]
    for s in (r.mrset, c.mrset):
        if s.kind != "single":
            notes.append(_footnote(s))
    return Table(title, columns, rows, row_header_cols=3, footnotes=notes)


def _cell(count, kind, row_total, col_total, grand, total: bool = False) -> str:
    if kind is None:
        return fmt_int(count)
    denom = {"row": row_total, "col": col_total, "total": grand}[kind]
    return fmt(100 * count / denom, 1) + "%" if denom else ""

"""Data Editor / Data / Transform / Utilities helpers that SPSS has and that need no Qt.

* Edit > Find / Replace (Data View cells as displayed, Variable View attributes);
* Data > Define Variable Properties (scan values, suggest a measurement level), Copy Data
  Properties, Sort Variables, Compare Datasets, Split into Files;
* Transform > Create Dummy Variables, Date and Time Wizard;
* Utilities > Variable Sets and Data File Comments (stored in ``ds.meta``).
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rsstatchem.core import expr_functions as X
from rsstatchem.core.dataset import (DATE, NOMINAL, NUMERIC, ORDINAL, SCALE, STRING, Dataset,
                                     Variable, format_code, format_number, is_valid_name)
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt_int

MODES = ("contains", "entire", "begins", "ends")


class PropsError(ValueError):
    pass


# ------------------------------------------------------------------ find / replace
def _match(text: str, needle: str, mode: str, match_case: bool) -> bool:
    if not match_case:
        text, needle = text.lower(), needle.lower()
    if mode == "entire":
        return text == needle
    if mode == "begins":
        return text.startswith(needle)
    if mode == "ends":
        return text.endswith(needle)
    return needle in text


def find_cells(ds: Dataset, needle: str, columns: list[int] | None = None,
               match_case: bool = False, mode: str = "contains",
               use_labels: bool = False) -> list[tuple[int, int]]:
    """Cells (row, col) whose displayed text matches, in row-major order of the columns."""
    if needle == "" and mode != "entire":
        return []
    cols = list(range(ds.n_vars)) if columns is None else columns
    hits = []
    for c in cols:
        for r in range(ds.n_cases):
            text = ds.display_value(r, c, use_labels)
            alt = ds.edit_text(r, c)
            if _match(text, needle, mode, match_case) or \
                    (alt != text and _match(alt, needle, mode, match_case)):
                hits.append((r, c))
    return sorted(hits, key=lambda rc: (rc[0], cols.index(rc[1]))) if columns is None \
        else hits


def next_match(hits: list[tuple[int, int]], current: tuple[int, int] | None,
               backwards: bool = False) -> tuple[int, int] | None:
    if not hits:
        return None
    if current is None:
        return hits[-1] if backwards else hits[0]
    if backwards:
        before = [h for h in hits if h < current]
        return before[-1] if before else hits[-1]
    after = [h for h in hits if h > current]
    return after[0] if after else hits[0]


def replace_cell_text(ds: Dataset, row: int, col: int, needle: str, new: str,
                      mode: str = "contains", match_case: bool = False) -> str:
    """The new cell text for one replacement (numeric cells are replaced as a whole)."""
    var = ds.variables[col]
    text = ds.edit_text(row, col)
    if var.type != STRING or mode == "entire":
        return new
    flags = 0 if match_case else re.IGNORECASE
    pattern = re.escape(needle)
    if mode == "begins":
        pattern = "^" + pattern
    elif mode == "ends":
        pattern += "$"
    return re.sub(pattern, lambda _: new, text, flags=flags)


def replace_all(ds: Dataset, needle: str, new: str, columns: list[int] | None = None,
                match_case: bool = False, mode: str = "contains") -> int:
    """Replace every match; returns the number of cells changed."""
    hits = find_cells(ds, needle, columns, match_case, mode)
    for r, c in hits:
        ds.set_value(r, c, replace_cell_text(ds, r, c, needle, new, mode, match_case))
    return len(hits)


VAR_FIELDS = ("name", "label", "value_labels")


def find_variables(ds: Dataset, needle: str, field: str = "name", match_case: bool = False,
                   mode: str = "contains") -> list[int]:
    out = []
    for i, v in enumerate(ds.variables):
        if field == "value_labels":
            texts = [str(t) for t in v.value_labels.values()]
        else:
            texts = [str(getattr(v, field, ""))]
        if any(_match(t, needle, mode, match_case) for t in texts):
            out.append(i)
    return out


# ------------------------------------------------------------------ define properties
def scan_values(ds: Dataset, name: str, max_cases: int | None = None,
                max_values: int = 200) -> list[tuple[object, int, bool]]:
    """Distinct values with counts (value, count, is user-missing), like SPSS's scan."""
    var = ds.var(name)
    s = ds.df[var.name]
    if max_cases:
        s = s.iloc[:max_cases]
    s = s[s.notna()]
    if var.type == STRING:
        s = s[s != ""]
    counts = s.value_counts(sort=False)
    try:
        counts = counts.sort_index()
    except TypeError:
        pass
    out = [(v, int(n), v in var.missing) for v, n in counts.items()][:max_values]
    for code in var.value_labels:  # labelled values without data are listed too
        if code not in counts.index:
            out.append((code, 0, code in var.missing))
    return out


def suggest_measure(ds: Dataset, name: str, max_categories: int = 24) -> tuple[str, str]:
    """(level, reason) using SPSS's rules for Define Variable Properties."""
    var = ds.var(name)
    if var.type == STRING:
        return NOMINAL, "String variables are always nominal or ordinal."
    if var.type == DATE:
        return SCALE, "Date variables are scale."
    values = pd.to_numeric(ds.analysis_series(name), errors="coerce").dropna()
    n_unique = values.nunique()
    if len(values) and (values < 0).any():
        return SCALE, "The variable contains negative values."
    if n_unique > max_categories:
        return SCALE, f"More than {max_categories} distinct values."
    if len(values) and not np.all(np.mod(values, 1) == 0):
        return SCALE, "The variable contains non-integer values."
    if var.value_labels:
        return (ORDINAL if n_unique > 2 else NOMINAL,
                "Labelled integer codes: ordinal if the codes have a natural order.")
    return (NOMINAL if n_unique <= 2 else ORDINAL,
            f"{n_unique} distinct integer values (at most {max_categories}).")


def apply_properties(ds: Dataset, name: str, *, label: str | None = None,
                     measure: str | None = None, value_labels: dict | None = None,
                     missing: list | None = None, role: str | None = None,
                     decimals: int | None = None) -> None:
    var = ds.var(name)
    if label is not None:
        var.label = label
    if measure is not None:
        var.measure = measure
    if value_labels is not None:
        var.value_labels = ({float(k): str(v) for k, v in value_labels.items()}
                            if var.type == NUMERIC else
                            {str(k): str(v) for k, v in value_labels.items()})
    if missing is not None:
        var.missing = [float(m) for m in missing] if var.type == NUMERIC else \
            [str(m) for m in missing]
    if role is not None:
        var.role = role
    if decimals is not None and var.type == NUMERIC:
        var.decimals = int(decimals)
    ds.modified = True


# ------------------------------------------------------------------ copy data properties
COPYABLE = ("label", "value_labels", "missing", "measure", "role", "width", "decimals",
            "columns", "align")
FILE_PROPERTIES = ("comments", "mrsets", "varsets", "weight", "dates")


def copy_properties(source: Dataset, target: Dataset, pairs: list[tuple[str, str]],
                    attrs: tuple = COPYABLE, file_props: tuple = (),
                    replace_labels: bool = True, create_missing: bool = False) -> Section:
    """Copy variable (and optional file) properties from ``source`` to ``target``.
    ``pairs`` are (source name, target name). Value labels are replaced or merged."""
    changed = []
    for src_name, dst_name in pairs:
        src = source.var(src_name)
        if not target.has_name(dst_name):
            if not create_missing:
                continue
            target.add_variable(dst_name, src.type)
        dst = target.var(dst_name)
        if src.type != dst.type:
            attrs_here = [a for a in attrs if a not in ("value_labels", "missing")]
        else:
            attrs_here = list(attrs)
        for a in attrs_here:
            value = getattr(src, a)
            if a == "value_labels" and not replace_labels:
                value = {**dst.value_labels, **value}
            setattr(dst, a, dict(value) if isinstance(value, dict) else
                    (list(value) if isinstance(value, list) else value))
        changed.append((src_name, dst_name, ", ".join(attrs_here)))
    for key in file_props:
        if key in source.meta:
            target.meta[key] = copy.deepcopy(source.meta[key])
    target.modified = True
    section = Section("Copy Data Properties")
    section.log("APPLY DICTIONARY FROM=*source\n  /SOURCE VARIABLES=" +
                " ".join(p[0] for p in pairs) + "\n  /TARGET VARIABLES=" +
                " ".join(p[1] for p in pairs) + ".")
    section.table(Table("Properties Copied", ["Source", "Target", "Properties"],
                        [list(c) for c in changed] or [["(none)", "", ""]]))
    if file_props:
        section.note("", "File properties copied: " + ", ".join(file_props))
    return section


# ------------------------------------------------------------------ sort variables
SORT_KEYS = ("name", "type", "width", "decimals", "label", "value_labels", "missing",
             "columns", "align", "measure", "role")


def sort_variables(ds: Dataset, key: str = "name", ascending: bool = True,
                   save_order_as: str | None = None) -> None:
    """Reorder variables by a Variable View attribute (SPSS SORT VARIABLES BY ...)."""
    if key not in SORT_KEYS:
        raise PropsError(f"Unknown sort key {key}.")

    def sort_value(v: Variable):
        value = getattr(v, key)
        if isinstance(value, (dict, list)):
            return (len(value), str(value))
        if isinstance(value, str):
            return (0, value.lower())
        return (0, value)

    if save_order_as:
        ds.meta.setdefault("attributes", {})[save_order_as] = ds.names()
    order = sorted(range(ds.n_vars), key=lambda i: sort_value(ds.variables[i]),
                   reverse=not ascending)
    ds.variables = [ds.variables[i] for i in order]
    ds.df = ds.df[[v.name for v in ds.variables]]
    ds.modified = True


# ------------------------------------------------------------------ dummies
def create_dummies(ds: Dataset, names: list[str], roots: list[str] | None = None,
                   omit: str = "none", use_labels: bool = True) -> Section:
    """Transform > Create Dummy Variables: one 0/1 variable per category of each variable
    (``omit`` = "none", "first" or "last" leaves out a reference category)."""
    section = Section("Create Dummy Variables")
    rows = []
    created_all = []
    for k, name in enumerate(names):
        var = ds.var(name)
        s = ds.analysis_series(name)
        values = sorted(s.dropna().unique(), key=lambda v: (isinstance(v, str), v))
        if len(values) > 100:
            raise PropsError(f"{name} has more than 100 categories.")
        if omit == "first":
            values = values[1:]
        elif omit == "last":
            values = values[:-1]
        root = (roots[k] if roots and k < len(roots) and roots[k] else name)
        if not is_valid_name(root + "_1"):
            raise PropsError(f"'{root}' is not a valid root name.")
        for j, value in enumerate(values, 1):
            new = f"{root}_{j}"
            n = 1
            while ds.has_name(new):
                new = f"{root}_{j}_{n}"
                n += 1
            col = pd.Series(np.where(s.isna(), np.nan, (s == value).astype(float)),
                            index=ds.df.index)
            v = ds.add_variable(new, values=col.to_numpy())
            label = var.label_for(value) if use_labels else None
            v.label = f"{var.title}={label or format_code(value)}"
            v.decimals = 0
            v.measure = NOMINAL
            v.value_labels = {0.0: "No", 1.0: "Yes"}
            created_all.append(new)
            rows.append([new, v.label, fmt_int(np.nansum(col))])
    section.log("SPSSINC CREATE DUMMIES VARIABLE=" + " ".join(names) +
                f"\n  ROOTNAME1={' '.join(roots or names)}\n  /OPTIONS ORDER=A USEVALUELABELS="
                f"{'YES' if use_labels else 'NO'}.")
    section.table(Table("Dummy Variables Created", ["Variable", "Label", "Count of 1"], rows))
    return section


# ------------------------------------------------------------------ date and time wizard
DATE_PARTS = ("year", "quarter", "month", "week", "mday", "jday", "wkday", "hour", "minute",
              "second", "tday", "time", "date")
INPUT_FORMATS = {
    "dd-mmm-yyyy (17-AUG-1945)": "DATE11", "mm/dd/yyyy": "ADATE10", "dd.mm.yyyy": "EDATE10",
    "dd/mm/yyyy": "%d/%m/%Y", "yyyy/mm/dd": "SDATE10", "yyyy-mm-dd": "%Y-%m-%d",
    "yyyy-mm-dd hh:mm:ss": "YMDHMS19", "dd-mmm-yyyy hh:mm:ss": "DATETIME20",
}


def _date_numbers(ds: Dataset, name: str) -> pd.Series:
    var = ds.var(name)
    if var.type == DATE:
        return X.from_timestamps(ds.df[var.name])
    return pd.to_numeric(ds.analysis_series(name), errors="coerce").astype(float)


def _new_date_var(ds: Dataset, target: str, seconds: pd.Series, label: str = "") -> Variable:
    if not is_valid_name(target) or ds.has_name(target):
        raise PropsError(f"'{target}' is not a valid new variable name.")
    var = ds.add_variable(target, DATE, values=X.to_timestamps(seconds).to_numpy())
    var.label = label
    var.width = 11
    return var


def _new_num_var(ds: Dataset, target: str, values: pd.Series, label: str = "",
                 decimals: int = 0) -> Variable:
    if not is_valid_name(target) or ds.has_name(target):
        raise PropsError(f"'{target}' is not a valid new variable name.")
    var = ds.add_variable(target, values=pd.to_numeric(values, errors="coerce").to_numpy())
    var.label = label
    var.decimals = decimals
    return var


def wizard_parse(ds: Dataset, source: str, target: str, pattern: str,
                 label: str = "") -> Section:
    """String with a date (in ``pattern``: an SPSS format such as DATE11 or a strftime
    pattern) -> date variable."""
    s = ds.df[ds.var(source).name]
    if pattern.startswith("%"):
        parsed = pd.to_datetime(s, format=pattern, errors="coerce")
        seconds = X.from_timestamps(parsed)
    else:
        seconds = s.map(lambda v: X.parse_value(v, pattern)).astype(float)
    var = _new_date_var(ds, target, seconds, label)
    section = Section("Date and Time Wizard")
    section.log(f"COMPUTE {target}=NUMBER({source}, {pattern}).\nVARIABLE LEVEL {target}"
                f" (SCALE).\nFORMATS {target} (DATE11).\nEXECUTE.")
    bad = int(seconds.isna().sum() - s.isna().sum() - (s == "").sum())
    section.note("", f"{var.name} created from {source}: {int(seconds.notna().sum())} dates"
                 + (f", {bad} values could not be read" if bad > 0 else "") + ".")
    return section


def wizard_from_parts(ds: Dataset, target: str, year: str, month: str | None = None,
                      day: str | None = None, hour: str | None = None,
                      minute: str | None = None, second: str | None = None,
                      label: str = "") -> Section:
    """Combine numeric year/month/day(/time) variables into a date variable."""
    def num(name, default):
        return _date_numbers(ds, name) if name else pd.Series(default, index=ds.df.index)

    y, m, d = num(year, np.nan), num(month, 1.0), num(day, 1.0)
    t = num(hour, 0.0) * 3600 + num(minute, 0.0) * 60 + num(second, 0.0)
    seconds = pd.Series(X.date_dmy(d, m, y), index=ds.df.index) + t
    _new_date_var(ds, target, seconds, label)
    section = Section("Date and Time Wizard")
    section.log(f"COMPUTE {target}=DATE.DMY({day or 1}, {month or 1}, {year})"
                + (f" + TIME.HMS({hour or 0}, {minute or 0}, {second or 0})" if hour else "")
                + f".\nFORMATS {target} (DATE11).\nEXECUTE.")
    return section


def wizard_extract(ds: Dataset, source: str, target: str, part: str,
                   label: str = "") -> Section:
    if part not in DATE_PARTS:
        raise PropsError(f"Unknown date part {part}.")
    values = pd.Series(X._xdate(part)(_date_numbers(ds, source)), index=ds.df.index)
    if part == "date":
        _new_date_var(ds, target, values, label)
    else:
        _new_num_var(ds, target, values, label)
    section = Section("Date and Time Wizard")
    section.log(f"COMPUTE {target}=XDATE.{part.upper()}({source}).\nEXECUTE.")
    return section


def wizard_difference(ds: Dataset, later: str, earlier: str, target: str, unit: str = "days",
                      label: str = "") -> Section:
    a, b = _date_numbers(ds, later), _date_numbers(ds, earlier)
    values = X._datediff([a, b, unit], "DATEDIFF")
    _new_num_var(ds, target, values, label)
    section = Section("Date and Time Wizard")
    section.log(f'COMPUTE {target}=DATEDIFF({later}, {earlier}, "{unit}").\nEXECUTE.')
    return section


def wizard_add(ds: Dataset, source: str, target: str, amount: float | str, unit: str = "days",
               label: str = "") -> Section:
    """Add a constant (or a numeric variable) of ``unit`` to a date."""
    base = _date_numbers(ds, source)
    n = _date_numbers(ds, amount) if isinstance(amount, str) else float(amount)
    values = X._datesum([base, n, unit], "DATESUM")
    _new_date_var(ds, target, values, label)
    section = Section("Date and Time Wizard")
    section.log(f'COMPUTE {target}=DATESUM({source}, {amount}, "{unit}").\nEXECUTE.')
    return section


# ------------------------------------------------------------------ compare datasets
def compare_datasets(active: Dataset, other: Dataset, ids: list[str] | None = None,
                     variables: list[str] | None = None, flag: str | None = None) -> Section:
    """Data > Compare Datasets: cases matched by ID (or by position), values compared."""
    common = [v.name for v in active.variables if other.has_name(v.name)
              and (not ids or v.name not in ids)]
    variables = [v for v in (variables or common) if other.has_name(v)]
    if ids:
        left = active.df.set_index([active.var(i).name for i in ids])
        right = other.df.set_index([other.var(i).name for i in ids])
        keys = [k for k in left.index if k in right.index]
        only_active = sum(1 for k in left.index if k not in right.index)
        only_other = sum(1 for k in right.index if k not in left.index)
    else:
        n = min(active.n_cases, other.n_cases)
        left, right = active.df.iloc[:n], other.df.iloc[:n]
        keys = list(range(n))
        only_active, only_other = active.n_cases - n, other.n_cases - n
    mismatch_rows, per_var = [], {v: 0 for v in variables}
    flags = pd.Series(0.0, index=active.df.index)
    for pos, key in enumerate(keys):
        diffs = []
        for v in variables:
            a = left.loc[key, active.var(v).name] if ids else left.iloc[pos][active.var(v).name]
            b = right.loc[key, other.var(v).name] if ids else right.iloc[pos][other.var(v).name]
            if isinstance(a, pd.Series):
                a = a.iloc[0]
            if isinstance(b, pd.Series):
                b = b.iloc[0]
            same = (pd.isna(a) and pd.isna(b)) or (not pd.isna(a) and not pd.isna(b) and a == b)
            if not same:
                diffs.append(v)
                per_var[v] += 1
        if diffs:
            label = key if not isinstance(key, tuple) else ", ".join(map(str, key))
            mismatch_rows.append([str(label if ids else pos + 1), ", ".join(diffs)])
            if not ids:
                flags.iloc[pos] = 1.0
            else:
                match = (active.df[[active.var(i).name for i in ids]].apply(tuple, axis=1)
                         == (key if isinstance(key, tuple) else (key,)))
                flags[match] = 1.0
    section = Section("Compare Datasets")
    section.log("COMPARE DATASETS\n  /COMPDATASET=*other\n  /VARIABLES " + " ".join(variables)
                + (f"\n  /CASEID {' '.join(ids)}" if ids else "") + ".")
    section.table(Table("Comparison Summary", ["", "N"], [
        ["Cases compared", fmt_int(len(keys))],
        ["Cases with mismatches", fmt_int(len(mismatch_rows))],
        ["Cases only in the active dataset", fmt_int(only_active)],
        ["Cases only in the compared dataset", fmt_int(only_other)],
        ["Variables compared", fmt_int(len(variables))]]))
    section.table(Table("Mismatches by Variable", ["Variable", "Cases"],
                        [[v, fmt_int(n)] for v, n in per_var.items()]))
    if mismatch_rows:
        section.table(Table("Case Mismatches", ["Case" if not ids else " / ".join(ids),
                                                "Variables that differ"], mismatch_rows[:500]))
    if flag:
        var = active.add_variable(flag, values=flags.to_numpy()) if not active.has_name(flag) \
            else active.var(flag)
        active.df[var.name] = flags
        var.label = "Case mismatch with compared dataset"
        var.decimals = 0
        var.value_labels = {0.0: "Match", 1.0: "Mismatch"}
    return section


# ------------------------------------------------------------------ split into files
def split_into_files(ds: Dataset, split_vars: list[str], folder: str | Path,
                     prefix: str = "") -> tuple[list[Path], Section]:
    """Data > Split into Files: one .rsd per combination of the split variables."""
    from rsstatchem.io.files import save_project

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cols = [ds.var(v).name for v in split_vars]
    paths, rows = [], []
    for key, frame in ds.df.groupby(cols, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        parts = []
        for name, value in zip(cols, key):
            var = ds.var(name)
            text = var.label_for(value) or ("missing" if pd.isna(value) else format_code(value))
            parts.append(re.sub(r"[^\w.-]+", "_", f"{name}_{text}"))
        path = folder / f"{prefix}{'_'.join(parts)}.rsd"
        sub = Dataset(frame.reset_index(drop=True),
                      [Variable.from_dict(v.to_dict()) for v in ds.variables], name=path.stem)
        for v in sub.variables:  # to_dict turns numeric label keys into JSON-able pairs
            if v.type == NUMERIC:
                v.value_labels = {float(k): t for k, t in v.value_labels.items()}
        save_project(sub, path)
        paths.append(path)
        rows.append([" / ".join(str(k) for k in key), fmt_int(len(frame)), path.name])
    section = Section("Split into Files")
    section.log(f"SPSSINC SPLIT DATASET SPLITVAR={' '.join(cols)}\n  /OUTPUT DIRECTORY="
                f"\"{folder}\".")
    section.table(Table("Files Created", ["Split values", "Cases", "File"], rows))
    return paths, section


# ------------------------------------------------------------------ variable sets / comments
@dataclass
class VarSet:
    name: str
    variables: list[str]


def variable_sets(ds: Dataset) -> dict[str, list[str]]:
    return {k: list(v) for k, v in (ds.meta.get("varsets") or {}).items()}


def set_variable_sets(ds: Dataset, sets: dict[str, list[str]]) -> None:
    ds.meta["varsets"] = {k: list(v) for k, v in sets.items() if v}
    active = [s for s in ds.meta.get("varsets_active", []) if s in ds.meta["varsets"]]
    ds.meta["varsets_active"] = active
    ds.modified = True


def use_variable_sets(ds: Dataset, names: list[str] | None) -> None:
    """Activate sets (None = Show All Variables)."""
    if names is None:
        ds.meta.pop("varsets_active", None)
    else:
        ds.meta["varsets_active"] = [n for n in names if n in variable_sets(ds)]


def visible_variables(ds: Dataset) -> list[str] | None:
    """Variables in the active sets, or None when all are shown."""
    active = ds.meta.get("varsets_active")
    if not active:
        return None
    sets = variable_sets(ds)
    keep = {n.lower() for s in active for n in sets.get(s, [])}
    return [v.name for v in ds.variables if v.name.lower() in keep]


def comments(ds: Dataset) -> str:
    return ds.meta.get("comments", "")


def set_comments(ds: Dataset, text: str) -> None:
    ds.meta["comments"] = text
    ds.modified = True


def display_value(ds: Dataset, name: str, value) -> str:
    var = ds.var(name)
    if var.type == NUMERIC and not pd.isna(value):
        return format_number(float(value), var.decimals)
    return "" if pd.isna(value) else str(value)

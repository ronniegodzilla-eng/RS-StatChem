"""Data > Select Cases, Split File and Weight Cases.

The settings live in ``ds.meta`` (so they are saved with the project and undoable):
``filter`` = name of the filter variable (1 = selected), ``split`` = {"vars": [...],
"mode": "compare" | "separate"}, ``weight`` = name of a frequency-weight variable.
Analyses never see these directly: :func:`analysis_views` turns the active dataset into the
dataset(s) an analysis should run on — filtered, weighted (frequency weights, rounded to whole
cases) and split into groups.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rsstatchem.core.dataset import NUMERIC, Dataset, format_code
from rsstatchem.core.expression import ExpressionError, evaluate, is_string_result
from rsstatchem.core.output import Section
from rsstatchem.core.stats.common import value_text

FILTER_VAR = "filter_$"


@dataclass
class AnalysisView:
    label: str               # "" when the file is not split
    ds: Dataset
    rows: np.ndarray | None  # positions in the active dataset (None when cases were replicated)


# ------------------------------------------------------------------ state
def filter_mask(ds: Dataset) -> pd.Series | None:
    name = ds.meta.get("filter")
    if not name or not ds.has_name(name):
        return None
    values = pd.to_numeric(ds.df[ds.var(name).name], errors="coerce")
    return values.fillna(0) != 0  # SPSS: zero or missing = not selected


def describe(ds: Dataset) -> list[str]:
    parts = []
    if filter_mask(ds) is not None:
        parts.append(f"Filter: {ds.meta['filter']}")
    if ds.meta.get("weight") and ds.has_name(ds.meta["weight"]):
        parts.append(f"Weighted by {ds.meta['weight']}")
    split = ds.meta.get("split") or {}
    if split.get("vars"):
        parts.append(f"Split File: {', '.join(split['vars'])}")
    return parts


def analysis_views(ds: Dataset) -> list[AnalysisView]:
    df = ds.df
    rows = np.arange(len(df))
    mask = filter_mask(ds)
    if mask is not None:
        rows = rows[mask.to_numpy()]
    replicated = False
    weight = ds.meta.get("weight")
    if weight and ds.has_name(weight):
        w = pd.to_numeric(ds.analysis_series(weight), errors="coerce").to_numpy()[rows]
        counts = np.where(np.isfinite(w) & (w > 0), np.rint(w), 0).astype(int)
        rows = np.repeat(rows, counts)
        replicated = True

    def make(sub_rows: np.ndarray, label: str) -> AnalysisView:
        sub = Dataset(df.iloc[sub_rows].reset_index(drop=True), copy.deepcopy(ds.variables),
                      name=ds.name)
        sub.meta = {k: v for k, v in ds.meta.items() if k not in ("filter", "weight", "split")}
        return AnalysisView(label, sub, None if replicated else sub_rows)

    split = ds.meta.get("split") or {}
    names = [n for n in split.get("vars", []) if ds.has_name(n)]
    if not names:
        return [make(rows, "")]
    keys = pd.DataFrame({n: ds.analysis_series(n).to_numpy()[rows] for n in names})
    views = []
    for key, idx in keys.groupby(names, sort=True, dropna=False).indices.items():
        key = key if isinstance(key, tuple) else (key,)
        label = ", ".join(
            f"{ds.var(n).title} = {'Missing' if pd.isna(v) else value_text(ds.var(n), v)}"
            for n, v in zip(names, key))
        views.append(make(rows[np.sort(idx)], label))
    return views


def combine(sections: list[tuple[str, Section]], ds: Dataset) -> Section:
    """Merge per-split-group sections into one, with the active Filter/Weight/Split notes."""
    first = sections[0][1]
    out = Section(first.title)
    logs = [i for i in first.items if getattr(i, "style", "") == "log"]
    out.items.extend(logs)
    notes = describe(ds)
    if notes:
        text = " · ".join(notes)
        if ds.meta.get("weight") and ds.has_name(ds.meta["weight"]):
            text += " (weights rounded to whole cases)"
        out.note("", text)
    for label, section in sections:
        body = [i for i in section.items if getattr(i, "style", "") != "log"]
        if label:
            out.note(label, "")
            for item in body:
                if getattr(item, "kind", "") == "table":
                    item.title = f"{item.title} — {label}"
        out.items.extend(body)
    return out


# ------------------------------------------------------------------ Select Cases
def select_cases(ds: Dataset, mode: str, expression: str = "", percent: float | None = None,
                 n_exact: int | None = None, from_first: int | None = None,
                 first: int | None = None, last: int | None = None,
                 variable: str | None = None, output: str = "filter",
                 seed: int | None = None) -> tuple[Section, Dataset | None]:
    """mode: all | if | sample | range | variable;  output: filter | copy | delete.
    Returns the output section and, for ``copy``, the new dataset to open."""
    section = Section("Select Cases")
    if mode == "all":
        ds.meta.pop("filter", None)
        ds.modified = True
        section.log("FILTER OFF.\nUSE ALL.\nEXECUTE.")
        section.note("", "All cases are used in analyses.")
        return section, None

    n = ds.n_cases
    if mode == "if":
        value = evaluate(ds, expression)
        if is_string_result(value):
            raise ExpressionError("The condition must be logical, e.g. kelompok = 1.")
        selected = (value == 1).to_numpy()
        how = f"COMPUTE {FILTER_VAR}=({expression.strip()})."
        label = f"{expression.strip()} (FILTER)"
    elif mode == "sample":
        rng = np.random.default_rng(seed)
        if percent is not None:
            selected = rng.random(n) < percent / 100.0
            how = f"COMPUTE {FILTER_VAR}=(uniform(1)<={percent / 100:.2f})."
            label = f"Approximately {percent:g}% of the cases (SAMPLE)"
        else:
            pool = min(from_first or n, n)
            take = min(n_exact or 0, pool)
            chosen = rng.choice(pool, size=take, replace=False)
            selected = np.zeros(n, bool)
            selected[chosen] = True
            how = f"SAMPLE {take} FROM {pool}."
            label = f"Exactly {take} cases from the first {pool} cases (SAMPLE)"
    elif mode == "range":
        lo, hi = max(1, first or 1), min(n, last or n)
        selected = np.zeros(n, bool)
        selected[lo - 1:hi] = True
        how = f"COMPUTE {FILTER_VAR}=($CASENUM >= {lo} & $CASENUM <= {hi})."
        label = f"Cases {lo} to {hi} (FILTER)"
    elif mode == "variable":
        if not variable or not ds.has_name(variable):
            raise ExpressionError("Choose a filter variable.")
        values = pd.to_numeric(ds.df[ds.var(variable).name], errors="coerce")
        selected = (values.fillna(0) != 0).to_numpy()
        how = f"FILTER BY {ds.var(variable).name}."
        label = None
    else:
        raise ExpressionError(f"Unknown selection mode: {mode}")

    k = int(selected.sum())
    if output == "copy":
        sub = Dataset(ds.df[selected].reset_index(drop=True), copy.deepcopy(ds.variables),
                      name=f"{ds.name}_selected")
        section.log(f"DATASET COPY selected.\n{how}\nSELECT IF ({FILTER_VAR}).\nEXECUTE.")
        section.note("", f"{k} of {n} cases copied to a new dataset.")
        return section, sub
    if output == "delete":
        ds.delete_cases([i for i in range(n) if not selected[i]])
        ds.meta.pop("filter", None)
        section.log(f"{how}\nSELECT IF ({FILTER_VAR}).\nEXECUTE.")
        section.note("", f"{n - k} unselected cases were deleted; {k} remain.")
        return section, None

    if mode == "variable":
        ds.meta["filter"] = ds.var(variable).name
    else:
        if ds.has_name(FILTER_VAR):
            ds.df[ds.var(FILTER_VAR).name] = selected.astype(float)
        else:
            ds.add_variable(FILTER_VAR, NUMERIC, values=selected.astype(float))
        var = ds.var(FILTER_VAR)
        var.decimals, var.label, var.measure = 0, label or "", "nominal"
        var.value_labels = {0.0: "Not Selected", 1.0: "Selected"}
        ds.meta["filter"] = var.name
    ds.modified = True
    section.log(f"USE ALL.\n{how}\nVARIABLE LABELS {FILTER_VAR} '{label}'.\n"
                f"VALUE LABELS {FILTER_VAR} 0 'Not Selected' 1 'Selected'.\n"
                f"FILTER BY {ds.meta['filter']}.\nEXECUTE." if label else
                f"{how}\nEXECUTE.")
    section.note("", f"{k} of {n} cases selected; unselected cases are filtered out "
                     "(struck-through row numbers in Data View).")
    return section, None


# ------------------------------------------------------------------ Split / Weight
def split_file(ds: Dataset, names: list[str], mode: str = "separate",
               sort: bool = True) -> Section:
    section = Section("Split File")
    if mode == "off" or not names:
        ds.meta.pop("split", None)
        ds.modified = True
        section.log("SPLIT FILE OFF.")
        section.note("", "Analyses use all cases together.")
        return section
    names = [ds.var(n).name for n in names]
    if sort:
        ds.sort_cases(names, [True] * len(names))
    ds.meta["split"] = {"vars": names, "mode": mode}
    ds.modified = True
    keyword = "LAYERED" if mode == "compare" else "SEPARATE"
    section.log(("SORT CASES BY " + " ".join(names) + ".\n" if sort else "")
                + f"SPLIT FILE {keyword} BY {' '.join(names)}.")
    groups = ds.df.groupby(names, dropna=False).ngroups
    section.note("", f"Output is split into {groups} group(s) by {', '.join(names)}.")
    return section


def weight_cases(ds: Dataset, variable: str | None) -> Section:
    section = Section("Weight Cases")
    if not variable:
        ds.meta.pop("weight", None)
        ds.modified = True
        section.log("WEIGHT OFF.")
        section.note("", "Cases are not weighted.")
        return section
    var = ds.var(variable)
    if var.type != NUMERIC:
        raise ExpressionError("The frequency variable must be numeric.")
    ds.meta["weight"] = var.name
    ds.modified = True
    w = pd.to_numeric(ds.analysis_series(var.name), errors="coerce")
    section.log(f"WEIGHT BY {var.name}.")
    fractional = bool(((w.dropna() % 1) != 0).any())
    section.note("", f"Cases are weighted by {var.name} (total weighted N = "
                     f"{format_code(float(w[w > 0].sum()))})."
                 + (" Fractional weights are rounded to whole cases in analyses."
                    if fractional else ""))
    return section

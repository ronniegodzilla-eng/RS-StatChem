"""SPSS Data and Transform procedures beyond Compute / Recode (no Qt here).

Transform: Count Values within Cases (COUNT), Automatic Recode (AUTORECODE), Visual Binning,
Rank Cases (RANK), Replace Missing Values (RMV), Shift Values (SHIFT VALUES).
Data: Identify Duplicate Cases, Merge Files > Add Cases / Add Variables (ADD FILES /
MATCH FILES), Aggregate (AGGREGATE), Transpose (FLIP), Restructure (VARSTOCASES /
CASESTOVARS).

Functions that change the active dataset do so in place and return a :class:`Section` whose
log holds the equivalent SPSS syntax. Functions that build a new dataset return
``(Dataset, Section)``. Behaviour, defaults, generated names and output wording follow SPSS
Statistics so results can be compared with it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from rsstatchem.core.dataset import (DATE, NOMINAL, NUMERIC, ORDINAL, SCALE, STRING, Dataset,
                                     Variable, format_code, is_valid_name, sanitize_name)
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int, percentile
from rsstatchem.core.transform import _condition_mask


class DataError(ValueError):
    """A user-correctable problem with the requested data operation."""


# ------------------------------------------------------------------ helpers
def _copy_var(var: Variable, name: str | None = None) -> Variable:
    new = Variable.from_dict(var.to_dict())
    if name:
        new.name = name
    return new


def _rebuild(ds: Dataset, df: pd.DataFrame, variables: list[Variable]) -> None:
    """Replace the dataset content in place, normalising column storage."""
    fresh = Dataset(df.reset_index(drop=True), variables, name=ds.name)
    ds.df = fresh.df
    ds.variables = fresh.variables
    ds.modified = True


def _check_new_name(ds: Dataset, name: str, taken: set[str] | None = None) -> str:
    name = name.strip()
    if not is_valid_name(name):
        raise DataError(f"'{name}' is not a valid variable name.")
    if ds.has_name(name) or (taken is not None and name.lower() in taken):
        raise DataError(f"A variable named '{name}' already exists.")
    if taken is not None:
        taken.add(name.lower())
    return name


def unique_name(ds: Dataset, base: str, taken: set[str] | None = None) -> str:
    """``base`` if free, else ``base_1``, ``base_2``… (the SPSS convention for new names)."""
    base = sanitize_name(base)[:60]
    taken = taken or set()
    name, k = base, 0
    while ds.has_name(name) or name.lower() in taken:
        k += 1
        name = f"{base}_{k}"
    return name


def _sysmis(series: pd.Series, var: Variable) -> pd.Series:
    if var.type == STRING:
        return series.isna() | (series.astype(object).map(
            lambda v: isinstance(v, str) and v.strip() == ""))
    return series.isna()


def _user_missing(series: pd.Series, var: Variable) -> pd.Series:
    if not var.missing:
        return pd.Series(False, index=series.index)
    return series.isin(var.missing) & series.notna()


def _q(value) -> str:
    return f"'{value}'" if isinstance(value, str) else _num(value)


def _num(value) -> str:
    """A number for syntax / tables without float noise (56.275000000000006 → 56.275)."""
    if isinstance(value, (int, float, np.floating)) and not isinstance(value, bool):
        return format_code(float(f"{float(value):.10g}"))
    return format_code(value)


def _freq_table(title: str, series: pd.Series, var: Variable) -> Table:
    """A compact SPSS Frequencies table for a created indicator variable."""
    counts = series.value_counts(dropna=True).sort_index()
    n = len(series)
    valid = int(counts.sum())
    rows, cum = [], 0.0
    for value, count in counts.items():
        pct = 100 * count / n
        vpct = 100 * count / valid if valid else math.nan
        cum += vpct
        rows.append(["" if rows else "Valid", var.label_for(value) or format_code(value),
                     fmt_int(count), fmt(pct, 1), fmt(vpct, 1), fmt(cum, 1)])
    rows.append(["" if rows else "Valid", "Total", fmt_int(valid),
                 fmt(100 * valid / n if n else 0, 1), fmt(100.0, 1), ""])
    if valid < n:
        rows.append(["Missing", "System", fmt_int(n - valid), fmt(100 * (n - valid) / n, 1),
                     "", ""])
        rows.append(["Total", "", fmt_int(n), fmt(100.0, 1), "", ""])
    return Table(title, ["", "", "Frequency", "Percent", "Valid Percent",
                         "Cumulative Percent"], rows, row_header_cols=2)


# ================================================================== COUNT
def spec_text(spec: tuple) -> str:
    """('range', 1, 3) → '1 thru 3' as shown in the Values to Count list."""
    kind = spec[0]
    if kind == "value":
        return _q(spec[1])
    if kind == "range":
        return f"{format_code(spec[1])} thru {format_code(spec[2])}"
    if kind == "lowest":
        return f"Lowest thru {format_code(spec[1])}"
    if kind == "highest":
        return f"{format_code(spec[1])} thru Highest"
    return {"sysmis": "SYSMIS", "missing": "MISSING"}[kind]


def _spec_syntax(spec: tuple) -> str:
    return (spec_text(spec).replace("Lowest", "LOWEST").replace("Highest", "HIGHEST")
            .replace(" thru ", " THRU "))


def _spec_mask(spec: tuple, raw: pd.Series, var: Variable) -> pd.Series:
    kind = spec[0]
    numeric = var.type == NUMERIC
    sysmis = _sysmis(raw, var)
    if kind == "value":
        if numeric:
            return (raw == float(spec[1])) & ~sysmis
        return raw.astype(object).map(lambda v: isinstance(v, str)
                                      and v.rstrip() == str(spec[1]).rstrip())
    if kind == "sysmis":
        return raw.isna() if numeric else pd.Series(False, index=raw.index)
    if kind == "missing":
        return sysmis | _user_missing(raw, var)
    if not numeric:
        raise DataError("Ranges can only be counted for numeric variables.")
    x = raw.astype(float)
    if kind == "range":
        return (x >= spec[1]) & (x <= spec[2])
    if kind == "lowest":
        return x <= spec[1]
    if kind == "highest":
        return x >= spec[1]
    raise DataError(f"Unknown value specification {spec!r}.")


def count_values(ds: Dataset, target: str, variables: list[str], specs: list[tuple],
                 label: str = "", condition: str | None = None) -> Section:
    """COUNT target = vars (values): the number of listed variables whose value matches any
    of ``specs`` in each case. Cases failing ``condition`` keep their old value
    (system-missing for a new variable)."""
    if not variables:
        raise DataError("Select at least one variable.")
    if not specs:
        raise DataError("Define at least one value to count.")
    target = target.strip()
    if not is_valid_name(target):
        raise DataError(f"'{target}' is not a valid target variable name.")
    vars_ = [ds.var(n) for n in variables]
    if len({v.type == STRING for v in vars_}) > 1:
        raise DataError("Numeric and string variables cannot be counted together.")
    mask = _condition_mask(ds, condition)
    total = pd.Series(0.0, index=ds.df.index)
    for var in vars_:
        raw = ds.df[var.name]
        hit = pd.Series(False, index=raw.index)
        for spec in specs:
            hit |= _spec_mask(spec, raw, var).fillna(False).astype(bool)
        total += hit.astype(float)
    exists = ds.has_name(target)
    if exists:
        tvar = ds.var(target)
        if tvar.type != NUMERIC:
            raise DataError(f"The target variable {tvar.name} must be numeric.")
        ds.df[tvar.name] = ds.df[tvar.name].where(~mask, total)
        if label:
            tvar.label = label
        ds.modified = True
    else:
        tvar = ds.add_variable(target, NUMERIC, values=total.where(mask, np.nan).to_numpy())
        tvar.decimals, tvar.label = 0, label
    section = Section("Count Values within Cases")
    spec = " ".join(_spec_syntax(s) for s in specs)
    cmd = f"COUNT {tvar.name}={' '.join(v.name for v in vars_)}({spec})."
    if label:
        cmd += f"\nVARIABLE LABELS {tvar.name} '{label}'."
    if condition and condition.strip():
        cmd = f"DO IF ({condition.strip()}).\n{cmd}\nEND IF."
    section.log(cmd + "\nEXECUTE.")
    section.note("", f"{'Changed existing' if exists else 'Created new'} variable {tvar.name}: "
                     f"{int(mask.sum())} of {ds.n_cases} cases counted.")
    return section


# ================================================================== AUTORECODE
def _autorecode_scheme(values: list, missing: list, descending: bool) -> dict:
    valid = sorted({v for v in values if v not in missing}, reverse=descending)
    miss = sorted({v for v in values if v in missing}, reverse=descending)
    return {v: float(i + 1) for i, v in enumerate(valid + miss)}, set(miss)


def _clean_values(series: pd.Series, var: Variable) -> pd.Series:
    if var.type == STRING:
        return series.astype(object).map(lambda v: v.rstrip() if isinstance(v, str) else "")
    return series


def automatic_recode(ds: Dataset, sources: list[str], targets: list[str],
                     descending: bool = False, blank_missing: bool = False,
                     group: bool = False, print_table: bool = True) -> Section:
    """AUTORECODE: consecutive integers 1..k in ascending (lowest first) or descending order
    of the old values. Old values (or their labels) become value labels; user-missing values
    get the codes after the valid ones and stay user-missing. System-missing stays missing."""
    if not sources:
        raise DataError("Select at least one variable.")
    if len(targets) != len(sources):
        raise DataError("Every variable needs a new name.")
    taken: set[str] = set()
    targets = [_check_new_name(ds, t, taken) for t in targets]
    vars_ = [ds.var(s) for s in sources]
    if group and len({v.type == STRING for v in vars_}) > 1:
        raise DataError("With the same recoding scheme for all variables, the variables must "
                        "all be numeric or all be string.")

    def values_and_missing(var: Variable) -> tuple[list, list]:
        s = _clean_values(ds.df[var.name], var)
        miss = [str(m).rstrip() if var.type == STRING else m for m in var.missing]
        if var.type == STRING:
            if blank_missing:
                miss = miss + [""]
            vals = list(s)
        else:
            vals = list(s.dropna())
        return vals, miss

    shared = None
    if group:
        all_vals, all_miss = [], []
        for var in vars_:
            v, m = values_and_missing(var)
            all_vals += v
            all_miss += m
        shared = _autorecode_scheme(all_vals, all_miss, descending)

    section = Section("Automatic Recode")
    for var, target in zip(vars_, targets):
        vals, miss = values_and_missing(var)
        scheme, miss_set = shared if shared else _autorecode_scheme(vals, miss, descending)
        clean = _clean_values(ds.df[var.name], var)
        new = clean.map(lambda v: scheme.get(v, np.nan) if not (
            isinstance(v, float) and math.isnan(v)) else np.nan).astype(float)
        nvar = ds.add_variable(target, NUMERIC, index=None, values=new.to_numpy())
        nvar.decimals = 0
        nvar.label = var.label
        nvar.measure = NOMINAL if var.type == STRING or var.measure == NOMINAL else ORDINAL
        label_sources = [var] + ([v for v in vars_ if v is not var] if group else [])
        labels = {}
        for old, code in scheme.items():
            text = None
            for src in label_sources:
                text = src.value_labels.get(old)
                if text is None and src.type == STRING:
                    text = next((lbl for k, lbl in src.value_labels.items()
                                 if str(k).rstrip() == old), None)
                if text is not None:
                    break
            labels[code] = text if text is not None else format_code(old)
        nvar.value_labels = labels
        nvar.missing = [scheme[m] for m in sorted(miss_set, key=lambda m: scheme[m])
                        if m in scheme]
        if print_table:
            rows = []
            for old, code in scheme.items():
                marker = " M" if old in miss_set else ""
                rows.append([format_code(old) if old != "" else "(blank)",
                             f"{format_code(code)}{marker}", labels[code]])
            foot = ["M = user-missing value"] if miss_set else []
            section.table(Table(f"{var.name} into {target}", ["Old Value", "New Value",
                                                               "Value Label"],
                                rows, row_header_cols=1, footnotes=foot))
    cmd = (f"AUTORECODE VARIABLES={' '.join(v.name for v in vars_)}\n"
           f"  /INTO {' '.join(targets)}")
    if descending:
        cmd += "\n  /DESCENDING"
    if blank_missing:
        cmd += "\n  /BLANK=MISSING"
    if group:
        cmd += "\n  /GROUP"
    if print_table:
        cmd += "\n  /PRINT"
    section.log(cmd + ".")
    section.items.insert(0, section.items.pop())  # syntax first, as in the other procedures
    return section


# ================================================================== VISUAL BINNING
def equal_width_cutpoints(first: float, n: int, width: float) -> list[float]:
    if n < 1 or width <= 0:
        raise DataError("Enter a positive width and at least one cutpoint.")
    return [round(first + i * width, 10) for i in range(int(n))]


def percentile_cutpoints(x: np.ndarray, n: int) -> list[float]:
    """Equal percentiles based on scanned cases: n cutpoints at 100/(n+1) % steps."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if not len(x):
        raise DataError("The variable has no valid values.")
    q = [100 * (i + 1) / (n + 1) for i in range(int(n))]
    cuts = [float(c) for c in percentile(x, q)]
    return sorted(set(cuts))


def sd_cutpoints(x: np.ndarray, sds: list[int]) -> list[float]:
    """Cutpoints at the mean and at mean ± k standard deviations (k in ``sds``)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        raise DataError("At least two valid values are needed.")
    m, s = float(x.mean()), float(x.std(ddof=1))
    cuts = {m}
    for k in sds:
        cuts |= {m - k * s, m + k * s}
    return sorted(cuts)


def bin_labels(cuts: list[float], decimals: int, upper_included: bool = True) -> list[str]:
    """SPSS "Make Labels": '<= 25', '26 - 35', '36+' (included) or '< 25', '25 - 34', '35+'."""
    unit = 10.0 ** -max(decimals, 0)

    def f(v: float) -> str:
        return f"{v:.{max(decimals, 0)}f}"

    labels = []
    for i, c in enumerate(cuts):
        if upper_included:
            if i == 0:
                labels.append(f"<= {f(c)}")
            else:
                labels.append(f"{f(cuts[i - 1] + unit)} - {f(c)}")
        else:
            if i == 0:
                labels.append(f"< {f(c)}")
            else:
                labels.append(f"{f(cuts[i - 1])} - {f(c - unit)}")
    last = cuts[-1] + unit if upper_included else cuts[-1]
    labels.append(f"{f(last)}+")
    return labels


def bin_codes(x: pd.Series, cuts: list[float], upper_included: bool = True,
              reverse: bool = False) -> pd.Series:
    arr = x.to_numpy(dtype=float)
    side = "left" if upper_included else "right"
    codes = np.searchsorted(np.asarray(cuts, float), arr, side=side).astype(float) + 1
    if reverse:
        codes = len(cuts) + 2 - codes
    codes[np.isnan(arr)] = np.nan
    return pd.Series(codes, index=x.index)


def bin_counts(ds: Dataset, source: str, cuts: list[float],
               upper_included: bool = True) -> list[int]:
    x = ds.analysis_series(source).astype(float)
    codes = bin_codes(x, cuts, upper_included).dropna()
    return [int((codes == k).sum()) for k in range(1, len(cuts) + 2)]


def visual_binning(ds: Dataset, source: str, target: str, cuts: list[float],
                   labels: list[str] | None = None, upper_included: bool = True,
                   reverse: bool = False, target_label: str | None = None) -> Section:
    """Bin a scale variable into ordinal categories 1..k+1 at ``cuts`` (the last interval is
    'HIGH'). User-missing values are copied and stay user-missing."""
    var = ds.var(source)
    if var.type != NUMERIC:
        raise DataError("Only numeric variables can be binned.")
    cuts = sorted(float(c) for c in cuts)
    if not cuts:
        raise DataError("Define at least one cutpoint.")
    if len(set(cuts)) != len(cuts):
        raise DataError("Cutpoints must be different.")
    target = _check_new_name(ds, target)
    raw = ds.df[var.name]
    user = _user_missing(raw, var)
    codes = bin_codes(raw.where(~user), cuts, upper_included, reverse)
    codes = codes.where(~user, raw)
    nvar = ds.add_variable(target, NUMERIC, values=codes.to_numpy())
    nvar.decimals, nvar.measure, nvar.width = 0, ORDINAL, 5
    nvar.label = target_label if target_label is not None else f"{var.title} (Binned)"
    nvar.missing = list(var.missing)
    k = len(cuts) + 1
    labels = labels or [""] * k
    order = list(range(1, k + 1))
    if reverse:
        order = order[::-1]
    nvar.value_labels = {float(code): lbl for code, lbl in zip(order, labels) if lbl}
    for m in var.missing:
        if var.value_labels.get(m):
            nvar.value_labels[m] = var.value_labels[m]

    section = Section("Visual Binning")
    if upper_included:
        parts = [f"(LO THRU {_num(c)}={code})" for c, code in zip(cuts, order)]
        parts.append(f"(LO THRU HI={order[-1]})")
        cmd = (f"RECODE {var.name} (MISSING=COPY) {' '.join(parts)} (ELSE=SYSMIS) "
               f"INTO {target}.")
    else:
        lines = [f"IF ({var.name} < {_num(cuts[0])}) {target}={order[0]}."]
        for i in range(1, len(cuts)):
            lines.append(f"IF ({var.name} >= {_num(cuts[i - 1])} AND {var.name} < "
                         f"{_num(cuts[i])}) {target}={order[i]}.")
        lines.append(f"IF ({var.name} >= {_num(cuts[-1])}) {target}={order[-1]}.")
        cmd = "\n".join(lines)
    cmd += f"\nVARIABLE LABELS {target} '{nvar.label}'."
    cmd += f"\nFORMATS {target} (F5.0)."
    if any(labels):
        cmd += f"\nVALUE LABELS {target} " + " ".join(
            f"{code} '{lbl}'" for code, lbl in zip(order, labels) if lbl) + "."
    if var.missing:
        cmd += f"\nMISSING VALUES {target} ({' '.join(format_code(m) for m in var.missing)})."
    cmd += f"\nVARIABLE LEVEL {target} (ORDINAL).\nEXECUTE."
    section.log(cmd)
    counts = [int((codes[~user] == c).sum()) for c in order]
    rows = []
    for i, (code, lbl, n) in enumerate(zip(order, labels, counts)):
        value = _num(cuts[i]) if i < len(cuts) else "HIGH"
        rows.append([format_code(code), value, lbl, fmt_int(n)])
    section.table(Table(f"Binned Variable {target}",
                        ["Code", "Upper Endpoint", "Label", "Count"], rows,
                        footnotes=["Upper endpoints are "
                                   + ("included (<=)." if upper_included else "excluded (<).")]))
    return section


# ================================================================== RANK
RANK_FUNCTIONS = {
    # function: (prefix, stem when the prefixed name is taken, label)
    "RANK": ("R", "RAN", "Rank"),
    "NTILES": ("N", "NTI", "Percentile Group"),
    "RFRACTION": ("R", "RFR", "Fractional Rank"),
    "PERCENT": ("P", "PER", "Fractional Rank Percent"),
    "N": ("N", "N", "Sum of Case Weights"),
    "PROPORTION": ("P", "PRO", "Proportion Estimate"),
    "NORMAL": ("N", "NOR", "Normal Score"),
    "SAVAGE": ("S", "SAV", "Savage Score"),
}
FRACTIONS = {"BLOM": "Blom's", "TUKEY": "Tukey's", "RANKIT": "Rankit", "VW": "Van der Waerden's"}
TIES = {"MEAN": "average", "LOW": "min", "HIGH": "max", "CONDENSE": "dense"}


def proportion_estimate(r: np.ndarray, w: float, fraction: str = "BLOM") -> np.ndarray:
    r = np.asarray(r, float)
    if fraction == "BLOM":
        return (r - 3 / 8) / (w + 1 / 4)
    if fraction == "TUKEY":
        return (r - 1 / 3) / (w + 1 / 3)
    if fraction == "RANKIT":
        return (r - 1 / 2) / w
    if fraction == "VW":
        return r / (w + 1)
    raise DataError(f"Unknown proportion estimation formula {fraction}.")


def savage_scores(x: np.ndarray, ties: str = "MEAN") -> np.ndarray:
    """Savage (exponential) scores a(j) = sum_{i<=j} 1/(n-i+1) - 1, tied cases share the mean
    (or the low / high score) of their positions."""
    n = len(x)
    a = np.cumsum(1.0 / (n - np.arange(1, n + 1) + 1)) - 1.0
    lo = stats.rankdata(x, "min").astype(int)
    hi = stats.rankdata(x, "max").astype(int)
    if ties == "LOW":
        return a[lo - 1]
    if ties == "HIGH":
        return a[hi - 1]
    csum = np.concatenate([[0.0], np.cumsum(a)])
    return (csum[hi] - csum[lo - 1]) / (hi - lo + 1)


def rank_values(x: np.ndarray, function: str, ties: str = "MEAN", ascending: bool = True,
                ntiles: int = 4, fraction: str = "BLOM") -> np.ndarray:
    """One rank function on the valid values of one group (SPSS RANK algorithms)."""
    x = np.asarray(x, float)
    w = float(len(x))
    if w == 0:
        return x
    key = x if ascending else -x
    r = stats.rankdata(key, TIES[ties])
    if function == "RANK":
        return r
    if function == "RFRACTION":
        return r / w
    if function == "PERCENT":
        return 100 * r / w
    if function == "N":
        return np.full(len(x), w)
    if function == "NTILES":
        return np.floor(r * ntiles / (w + 1)) + 1
    if function == "PROPORTION":
        return proportion_estimate(r, w, fraction)
    if function == "NORMAL":
        return stats.norm.ppf(proportion_estimate(r, w, fraction))
    if function == "SAVAGE":
        return savage_scores(key, "MEAN" if ties == "CONDENSE" else ties)
    raise DataError(f"Unknown rank function {function}.")


def _rank_name(ds: Dataset, function: str, source: str, taken: set[str]) -> str:
    prefix, stem, _ = RANK_FUNCTIONS[function]
    name = (prefix + source)[:64]
    if not ds.has_name(name) and name.lower() not in taken:
        taken.add(name.lower())
        return name
    k = 1
    while ds.has_name(f"{stem}{k:03d}") or f"{stem}{k:03d}".lower() in taken:
        k += 1
    taken.add(f"{stem}{k:03d}".lower())
    return f"{stem}{k:03d}"


def rank_cases(ds: Dataset, variables: list[str], functions: list[str] | None = None,
               by: list[str] | None = None, ties: str = "MEAN", ascending: bool = True,
               ntiles: int = 4, fraction: str = "BLOM", summary: bool = True) -> Section:
    """RANK VARIABLES=… (A|D) BY … /function … /TIES=…. Missing values (system and user) are
    excluded and get system-missing ranks; cases with a missing BY value are not ranked."""
    functions = functions or ["RANK"]
    by = by or []
    if not variables:
        raise DataError("Select at least one variable to rank.")
    for name in variables:
        if ds.var(name).type != NUMERIC:
            raise DataError(f"{name} is not numeric; only numeric variables can be ranked.")
    if ties not in TIES:
        raise DataError(f"Unknown TIES method {ties}.")
    if "NTILES" in functions and ntiles < 2:
        raise DataError("NTILES needs at least 2 groups.")
    by_ok = pd.Series(True, index=ds.df.index)
    for b in by:
        by_ok &= ds.analysis_series(b).notna()
    groups = [ds.df[ds.var(b).name] for b in by]
    taken: set[str] = set()
    created = []
    for name in variables:
        var = ds.var(name)
        x = ds.analysis_series(name).astype(float)
        valid = x.notna() & by_ok
        for function in functions:
            out = pd.Series(np.nan, index=ds.df.index)
            if by:
                for _, idx in x[valid].groupby([g[valid] for g in groups],
                                               dropna=False).groups.items():
                    out.loc[idx] = rank_values(x.loc[idx].to_numpy(), function, ties,
                                               ascending, ntiles, fraction)
            elif valid.any():
                out[valid] = rank_values(x[valid].to_numpy(), function, ties, ascending,
                                         ntiles, fraction)
            new_name = _rank_name(ds, function, var.name, taken)
            label = f"{RANK_FUNCTIONS[function][2]} of {var.name}"
            if function in ("PROPORTION", "NORMAL"):
                label += f" using {FRACTIONS[fraction]} Formula"
            if function == "NTILES":
                label = f"Percentile Group of {var.name}"
            if by:
                label += " by " + " ".join(ds.var(b).name for b in by)
            created.append((var.name, function, new_name, label, out))
    for _, function, new_name, label, out in created:
        nvar = ds.add_variable(new_name, NUMERIC, values=out.to_numpy())
        nvar.label = label
        integral = function in ("N", "NTILES") or (function == "RANK" and ties != "MEAN")
        nvar.decimals = 0 if integral else (3 if function == "RANK" else 5)
        nvar.measure = ORDINAL if function in ("RANK", "NTILES") else SCALE
        nvar.width = 9 if integral else 11

    section = Section("Rank Cases")
    fn_syntax = []
    for function in functions:
        if function == "NTILES":
            fn_syntax.append(f"/NTILES({ntiles})")
        else:
            fn_syntax.append(f"/{function}")
    cmd = f"RANK VARIABLES={' '.join(ds.var(v).name for v in variables)} " \
          f"({'A' if ascending else 'D'})"
    if by:
        cmd += " BY " + " ".join(ds.var(b).name for b in by)
    cmd += "\n  " + " ".join(fn_syntax)
    if any(f in ("PROPORTION", "NORMAL") for f in functions):
        cmd += f"\n  /FRACTION={fraction}"
    cmd += f"\n  /PRINT={'YES' if summary else 'NO'}\n  /TIES={ties}."
    section.log(cmd)
    if summary:
        fn_names = {"RANK": "Rank", "NTILES": f"NTILES({ntiles})", "RFRACTION": "RFRACTION",
                    "PERCENT": "PERCENT", "N": "N", "PROPORTION": "PROPORTION",
                    "NORMAL": "NORMAL", "SAVAGE": "SAVAGE"}
        rows = [[src, fn_names[f], new, lbl] for src, f, new, lbl, _ in created]
        tie_text = {"MEAN": "Mean rank of tied values is used for ties.",
                    "LOW": "Low rank of tied values is used for ties.",
                    "HIGH": "High rank of tied values is used for ties.",
                    "CONDENSE": "Sequential ranks are assigned to unique values (ties "
                                "share a rank)."}[ties]
        foot = [tie_text, f"Ranks are in {'ascending' if ascending else 'descending'} order."]
        if any(f in ("PROPORTION", "NORMAL") for f in functions):
            foot.append(f"{FRACTIONS[fraction]} formula is used to estimate proportions.")
        section.table(Table("Created Variables",
                            ["Source Variable", "Function", "New Variable", "Label"],
                            rows, footnotes=foot))
    return section


# ================================================================== REPLACE MISSING VALUES
RMV_METHODS = {"SMEAN": "Series mean", "MEAN": "Mean of nearby points",
               "MEDIAN": "Median of nearby points", "LINT": "Linear interpolation",
               "TREND": "Linear trend at point"}


def replace_series(x: pd.Series, method: str, span: int | None = 2) -> pd.Series:
    """One RMV function. ``span=None`` means ALL (mean/median of the whole series)."""
    x = x.astype(float).reset_index(drop=True)
    miss = x.isna()
    out = x.copy()
    valid = x.dropna()
    if valid.empty:
        return out
    if method == "SMEAN" or (method == "MEAN" and span is None):
        out[miss] = valid.mean()
    elif method == "MEDIAN" and span is None:
        out[miss] = valid.median()
    elif method in ("MEAN", "MEDIAN"):
        pos = valid.index.to_numpy()
        vals = valid.to_numpy()
        for i in np.flatnonzero(miss.to_numpy()):
            k = np.searchsorted(pos, i)
            if k < span or len(pos) - k < span:
                continue  # not enough valid neighbours on one side: stays missing
            window = np.concatenate([vals[k - span:k], vals[k:k + span]])
            out[i] = window.mean() if method == "MEAN" else np.median(window)
    elif method == "LINT":
        out = x.interpolate(method="index", limit_area="inside")
    elif method == "TREND":
        t = np.arange(1, len(x) + 1, dtype=float)
        ok = ~miss.to_numpy()
        if ok.sum() >= 2:
            slope, intercept = np.polyfit(t[ok], x.to_numpy()[ok], 1)
            out[miss] = intercept + slope * t[miss.to_numpy()]
        else:
            out[miss] = valid.mean()
    else:
        raise DataError(f"Unknown method {method}.")
    return out


def rmv_function_text(method: str, source: str, span: int | None) -> str:
    if method in ("MEAN", "MEDIAN"):
        return f"{method}({source},{'ALL' if span is None else int(span)})"
    return f"{method}({source})"


def replace_missing(ds: Dataset, specs: list[tuple[str, str, str, int | None]]) -> Section:
    """RMV new = FUNCTION(source[, span]) for (source, new name, method, span) specs.
    User-missing values are replaced too."""
    if not specs:
        raise DataError("Select at least one variable.")
    taken: set[str] = set()
    rows, cmds = [], []
    for source, new, method, span in specs:
        var = ds.var(source)
        if var.type != NUMERIC:
            raise DataError(f"{var.name} is not numeric.")
        new = _check_new_name(ds, new, taken)
        x = ds.analysis_series(var.name).astype(float)
        filled = replace_series(x, method, span)
        filled.index = ds.df.index
        nvar = ds.add_variable(new, NUMERIC, values=filled.to_numpy())
        func = rmv_function_text(method, var.name, span)
        nvar.label = func
        nvar.decimals = max(var.decimals, 2) if method != "LINT" else var.decimals
        nvar.measure = var.measure
        nvar.value_labels = dict(var.value_labels)
        replaced = int((x.isna() & filled.notna()).sum())
        ok = np.flatnonzero(filled.notna().to_numpy())
        first = fmt_int(ok[0] + 1) if len(ok) else ""
        last = fmt_int(ok[-1] + 1) if len(ok) else ""
        rows.append([new, fmt_int(replaced), first, last, fmt_int(len(ok)), func])
        cmds.append(f"{new}={func}")
    section = Section("Replace Missing Values")
    section.log("RMV /" + " /".join(cmds) + ".")
    section.table(Table("Result Variables",
                        ["Result Variable", "N of Replaced Missing Values",
                         ("Case Number of Non-Missing Values", "First"),
                         ("Case Number of Non-Missing Values", "Last"), "N of Valid Cases",
                         "Creating Function"], rows))
    return section


# ================================================================== SHIFT VALUES
def shift_values(ds: Dataset, specs: list[tuple[str, str, bool, int]]) -> Section:
    """SHIFT VALUES VARIABLE=v RESULT=new LAG=n | LEAD=n for (source, new, lag, n) specs."""
    if not specs:
        raise DataError("Select at least one variable.")
    taken: set[str] = set()
    cmds, notes = [], []
    for source, new, lag, n in specs:
        var = ds.var(source)
        n = int(n)
        if n < 1:
            raise DataError("The number of cases to shift must be at least 1.")
        new = _check_new_name(ds, new, taken)
        shifted = ds.df[var.name].shift(n if lag else -n)
        nvar = _copy_var(var, new)
        ds.add_variable(new, var.type, values=shifted.to_numpy())
        ds.variables[ds.index_of(new)] = nvar
        cmds.append(f"SHIFT VALUES VARIABLE={var.name} RESULT={new} "
                    f"{'LAG' if lag else 'LEAD'}={n}.")
        notes.append(f"{new} = {'lag' if lag else 'lead'} {n} of {var.name}")
    section = Section("Shift Values")
    section.log("\n".join(cmds))
    section.note("", "Created " + "; ".join(notes) + ".")
    return section


# ================================================================== IDENTIFY DUPLICATES
def identify_duplicates(ds: Dataset, match: list[str], sort_by: list[str] | None = None,
                        sort_ascending: bool = True, primary: str | None = "last",
                        primary_name: str | None = None, sequence_name: str | None = None,
                        move_to_top: bool = False, show_freq: bool = True) -> Section:
    """Identify Duplicate Cases: sorts the file by the matching (and sort-within) variables
    and creates PrimaryLast / PrimaryFirst (1 = unique or primary case, 0 = duplicate) and an
    optional MatchSequence (0 = no matching case, else 1, 2, … within the group)."""
    if not match:
        raise DataError("Define matching cases by at least one variable.")
    sort_by = sort_by or []
    if not primary and not sequence_name:
        raise DataError("Select at least one variable to create.")
    taken: set[str] = set()
    if primary:
        primary_name = _check_new_name(
            ds, primary_name or ("PrimaryLast" if primary == "last" else "PrimaryFirst"), taken)
    if sequence_name:
        sequence_name = _check_new_name(ds, sequence_name, taken)
    mnames = [ds.var(m).name for m in match]
    snames = [ds.var(s).name for s in sort_by if ds.var(s).name not in mnames]
    df = ds.df.sort_values(mnames + snames, ascending=[True] * len(mnames)
                           + [sort_ascending] * len(snames),
                           kind="mergesort", na_position="first").reset_index(drop=True)
    key = df[mnames].astype(object).where(df[mnames].notna(), "\0nan")
    gid = key.groupby(mnames, sort=False).ngroup()
    occ = gid.groupby(gid).cumcount()
    size = gid.map(gid.value_counts())
    is_first = (occ == 0).astype(float)
    is_last = (occ == size - 1).astype(float)
    sequence = np.where(size > 1, occ + 1, 0).astype(float)
    variables = list(ds.variables)
    if primary:
        df[primary_name] = is_last if primary == "last" else is_first
        pv = Variable(primary_name, NUMERIC, width=1, decimals=0, measure=ORDINAL,
                      label=f"Indicator of each {primary} matching case as Primary",
                      value_labels={0.0: "Duplicate Case", 1.0: "Primary Case"})
        variables.append(pv)
    if sequence_name:
        df[sequence_name] = sequence
        variables.append(Variable(sequence_name, NUMERIC, width=7, decimals=0, measure=SCALE,
                                  label="Sequential count of matching cases"))
    if move_to_top:
        order = np.argsort(-(size.to_numpy() > 1).astype(int), kind="mergesort")
        df = df.iloc[order].reset_index(drop=True)
    _rebuild(ds, df, variables)

    sort_spec = " ".join(f"{m}(A)" for m in mnames) + "".join(
        f" {s}({'A' if sort_ascending else 'D'})" for s in snames)
    lines = ["* Identify Duplicate Cases.", f"SORT CASES BY {sort_spec}.",
             "MATCH FILES\n  /FILE=*\n  /BY " + " ".join(mnames)
             + "\n  /FIRST=PrimaryFirst\n  /LAST=PrimaryLast."]
    if sequence_name:
        lines += ["DO IF (PrimaryFirst).", f"COMPUTE {sequence_name}=1-PrimaryLast.", "ELSE.",
                  f"COMPUTE {sequence_name}={sequence_name}+1.", "END IF.",
                  f"LEAVE {sequence_name}.", f"FORMATS {sequence_name} (f7)."]
    if move_to_top:
        seq = sequence_name or "MatchSequence"
        lines += [f"COMPUTE InDupGrp={seq}>0.", "SORT CASES InDupGrp(D)."]
    if primary:
        lines += [f"VARIABLE LABELS {primary_name} 'Indicator of each {primary} matching case "
                  f"as Primary'.", f"VALUE LABELS {primary_name} 0 'Duplicate Case' 1 "
                  f"'Primary Case'.", f"VARIABLE LEVEL {primary_name} (ORDINAL)."]
    if show_freq:
        made = [n for n in (primary_name, sequence_name) if n]
        lines.append(f"FREQUENCIES VARIABLES={' '.join(made)}.")
    lines.append("EXECUTE.")
    section = Section("Identify Duplicate Cases")
    section.log("\n".join(lines))
    n_dup = int((occ > 0).sum())
    section.note("", f"{int(gid.nunique())} distinct combinations of "
                     f"{', '.join(mnames)}; {int((size > 1).sum())} cases are in duplicate "
                     f"groups ({n_dup} duplicates besides the primary cases). The file has been "
                     f"sorted by {sort_spec.replace('(A)', '').replace('(D)', ' (D)')}.")
    if show_freq:
        if primary:
            section.table(_freq_table(ds.var(primary_name).label, ds.df[primary_name],
                                      ds.var(primary_name)))
        if sequence_name:
            section.table(_freq_table(ds.var(sequence_name).label, ds.df[sequence_name],
                                      ds.var(sequence_name)))
    return section


# ================================================================== MERGE: ADD CASES
def _compatible(a: Variable, b: Variable) -> bool:
    return (a.type == STRING) == (b.type == STRING) and (a.type == DATE) == (b.type == DATE)


def match_variables(ds: Dataset, other: Dataset) -> tuple[list[tuple[str, str]], list[str],
                                                          list[str]]:
    """Pairs of same-named variables of compatible type, then unpaired names in the active
    dataset and in the other dataset (SPSS marks them * and +)."""
    paired, un_a, un_b = [], [], []
    for var in ds.variables:
        if other.has_name(var.name) and _compatible(var, other.var(var.name)):
            paired.append((var.name, other.var(var.name).name))
        else:
            un_a.append(var.name)
    used = {b.lower() for _, b in paired}
    un_b = [v.name for v in other.variables if v.name.lower() not in used]
    return paired, un_a, un_b


def _merge_labels(target: Variable, source: Variable) -> None:
    for k, v in source.value_labels.items():
        target.value_labels.setdefault(k, v)
    if not target.label and source.label:
        target.label = source.label


def add_cases(ds: Dataset, other: Dataset, pairs: list[tuple[str, str]] | None = None,
              keep_active: list[str] | None = None, keep_other: list[str] | None = None,
              source_var: str | None = None, other_label: str = "") -> Section:
    """ADD FILES /FILE=* /FILE=other: the other dataset's cases are appended. ``pairs`` maps
    active → other variable names (default: same name and type); unpaired variables listed in
    ``keep_active`` / ``keep_other`` are kept and get missing values for the other file's
    cases; all remaining unpaired variables are dropped."""
    auto, _, _ = match_variables(ds, other)
    pairs = auto if pairs is None else pairs
    keep_active = keep_active or []
    keep_other = keep_other or []
    variables, a_cols, b_cols = [], {}, {}
    paired_a = {a.lower(): b for a, b in pairs}
    for var in ds.variables:
        if var.name.lower() in paired_a:
            bvar = other.var(paired_a[var.name.lower()])
            if not _compatible(var, bvar):
                raise DataError(f"{var.name} and {bvar.name} have different types and cannot "
                                "be paired.")
            new = _copy_var(var)
            _merge_labels(new, bvar)
            new.width = max(var.width, bvar.width)
            variables.append(new)
            a_cols[new.name] = ds.df[var.name]
            b_cols[new.name] = other.df[bvar.name]
        elif any(var.name.lower() == k.lower() for k in keep_active):
            variables.append(_copy_var(var))
            a_cols[var.name] = ds.df[var.name]
    for name in keep_other:
        bvar = other.var(name)
        if ds.has_name(bvar.name) or any(v.name.lower() == bvar.name.lower()
                                         for v in variables):
            raise DataError(f"{bvar.name} already exists in the active dataset; rename it "
                            "before adding it.")
        variables.append(_copy_var(bvar))
        b_cols[bvar.name] = other.df[bvar.name]
    if source_var:
        src = _check_new_name(Dataset(), source_var)
        if any(v.name.lower() == src.lower() for v in variables):
            raise DataError(f"A variable named '{src}' already exists.")
        variables.append(Variable(src, NUMERIC, width=1, decimals=0, measure=NOMINAL,
                                  label="Case source",
                                  value_labels={0.0: "Active dataset",
                                                1.0: other_label or "Other dataset"}))
        a_cols[src] = pd.Series(0.0, index=ds.df.index)
        b_cols[src] = pd.Series(1.0, index=other.df.index)
    names = [v.name for v in variables]
    if not names:
        raise DataError("The new dataset would have no variables.")
    na, nb = ds.n_cases, other.n_cases

    def frame(cols: dict, n: int) -> pd.DataFrame:
        return pd.DataFrame({c: (cols[c].reset_index(drop=True).astype(object) if c in cols
                                 else pd.Series([np.nan] * n, dtype=object)) for c in names})

    df = pd.concat([frame(a_cols, na), frame(b_cols, nb)], ignore_index=True)
    dropped = [v.name for v in ds.variables if v.name not in names] + \
        [v.name for v in other.variables if v.name not in names
         and v.name.lower() not in {b.lower() for _, b in pairs}]
    _rebuild(ds, df, variables)

    section = Section("Add Cases")
    renames = [f"({b}={a})" for a, b in pairs if a != b]
    cmd = "ADD FILES /FILE=*\n  /FILE='" + (other_label or other.name) + "'"
    if renames:
        cmd += "\n  /RENAME " + " ".join(renames)
    if source_var:
        cmd += f"\n  /IN={source_var}"
    cmd += "."
    if source_var:
        cmd += f"\nVARIABLE LABELS {source_var} 'Case source'."
    section.log(cmd + "\nEXECUTE.")
    rows = [["Active dataset", fmt_int(na)], [other_label or other.name, fmt_int(nb)],
            ["New active dataset", fmt_int(na + nb)]]
    section.table(Table("Cases", ["", "N"], rows))
    section.note("", f"{len(pairs)} paired variable(s); {len(names)} variables in the new "
                     "active dataset." + (f" Dropped unpaired variables: {', '.join(dropped)}."
                                          if dropped else ""))
    return section


# ================================================================== MERGE: ADD VARIABLES
def add_variables(ds: Dataset, other: Dataset, method: str = "order",
                  keys: list[str] | None = None, include: list[str] | None = None,
                  renames: dict[str, str] | None = None, lookup: str = "other",
                  other_label: str = "") -> Section:
    """MATCH FILES. ``method``: 'order' (one-to-one by case order), 'key' (one-to-one by key
    values; unmatched cases from both files are kept) or 'table' (one-to-many: every case of
    the case file gets the matching lookup-table record; ``lookup`` = 'other' or 'active').
    ``include`` lists the other dataset's variables to add (default: all whose names are not
    already in the active dataset); ``renames`` renames them first."""
    keys = [k for k in (keys or [])]
    renames = {k.lower(): v for k, v in (renames or {}).items()}
    if method in ("key", "table") and not keys:
        raise DataError("Select at least one key variable.")
    for k in keys:
        if not ds.has_name(k) or not other.has_name(k):
            raise DataError(f"Key variable {k} must exist in both datasets.")
        if not _compatible(ds.var(k), other.var(k)):
            raise DataError(f"Key variable {k} has different types in the two datasets.")
    key_lower = {k.lower() for k in keys}
    if include is None:
        include = [v.name for v in other.variables if v.name.lower() not in key_lower
                   and not ds.has_name(renames.get(v.name.lower(), v.name))]
    add_vars: list[tuple[Variable, str]] = []
    for name in include:
        bvar = other.var(name)
        if bvar.name.lower() in key_lower:
            continue
        new_name = renames.get(bvar.name.lower(), bvar.name)
        if not is_valid_name(new_name):
            raise DataError(f"'{new_name}' is not a valid variable name.")
        if ds.has_name(new_name) or any(v.name.lower() == new_name.lower()
                                        for v, _ in add_vars):
            raise DataError(f"{new_name} already exists in the active dataset; exclude or "
                            "rename it.")
        add_vars.append((_copy_var(bvar, new_name), bvar.name))
    a = ds.df.copy()
    b = pd.DataFrame({nv.name: other.df[src].to_numpy() for nv, src in add_vars},
                     index=other.df.index)
    a_keys = [ds.var(k).name for k in keys]
    for k, ak in zip(keys, a_keys):
        b[ak] = other.df[other.var(k).name].to_numpy()
    warnings = []
    n_matched = None
    if method == "order":
        n = max(len(a), len(b))
        a = a.reindex(pd.RangeIndex(n))
        b = b[[nv.name for nv, _ in add_vars]].reindex(pd.RangeIndex(n))
        df = pd.concat([a, b], axis=1)
        if ds.n_cases != other.n_cases:
            warnings.append(f"The datasets have different numbers of cases ({ds.n_cases} and "
                            f"{other.n_cases}); the extra cases get missing values.")
    elif method == "key":
        for frame, label in ((a, "active dataset"), (b, other_label or "other dataset")):
            if frame.duplicated(a_keys).any():
                warnings.append(f"Duplicate key values in the {label}: cases with the same "
                                "key are matched in order of appearance.")
        a["__occ"] = a.groupby(a_keys, dropna=False).cumcount()
        b["__occ"] = b.groupby(a_keys, dropna=False).cumcount()
        df = a.merge(b, on=a_keys + ["__occ"], how="outer", sort=False, indicator=True)
        n_matched = int((df["_merge"] == "both").sum())
        df = df.sort_values(a_keys + ["__occ"], kind="mergesort", na_position="first")
        df = df.drop(columns=["__occ", "_merge"])
    elif method == "table":
        if lookup == "other":
            case, table = a, b
        else:
            case, table = b, a
        if table.duplicated(a_keys).any():
            warnings.append("Duplicate key values in the lookup table; the first record of each "
                            "key is used.")
            table = table.drop_duplicates(a_keys, keep="first")
        df = case.merge(table, on=a_keys, how="left", sort=False, indicator=True)
        n_matched = int((df["_merge"] == "both").sum())
        df = df.sort_values(a_keys, kind="mergesort", na_position="first").drop(
            columns=["_merge"])
    else:
        raise DataError(f"Unknown merge method {method}.")
    variables = [_copy_var(v) for v in ds.variables] + [nv for nv, _ in add_vars]
    df = df[[v.name for v in variables]].reset_index(drop=True)
    _rebuild(ds, df, variables)

    section = Section("Add Variables")
    src = other_label or other.name
    renamed = [f"({old}={nv.name})" for nv, old in add_vars if old != nv.name]
    drop = [v.name for v in other.variables if v.name.lower() not in key_lower
            and v.name not in {old for _, old in add_vars}]
    if method == "order":
        cmd = f"MATCH FILES /FILE=*\n  /FILE='{src}'"
    elif method == "key":
        cmd = f"MATCH FILES /FILE=*\n  /FILE='{src}'"
    elif lookup == "other":
        cmd = f"MATCH FILES /FILE=*\n  /TABLE='{src}'"
    else:
        cmd = f"MATCH FILES /TABLE=*\n  /FILE='{src}'"
    if renamed:
        cmd += "\n  /RENAME " + " ".join(renamed)
    if keys:
        cmd += "\n  /BY " + " ".join(a_keys)
    if drop:
        cmd += "\n  /DROP=" + " ".join(drop)
    section.log(cmd + ".\nEXECUTE.")
    method_text = {"order": "One-to-one merge based on file order",
                   "key": "One-to-one merge based on key values",
                   "table": "One-to-many merge based on key values"}[method]
    rows = [["Merge method", method_text],
            ["Variables added", ", ".join(nv.name for nv, _ in add_vars) or "(none)"],
            ["Cases in new active dataset", fmt_int(ds.n_cases)]]
    if keys:
        rows.insert(1, ["Key variables", ", ".join(a_keys)])
    if method == "table":
        rows.insert(2, ["Lookup table", "Active dataset" if lookup == "active" else src])
    if n_matched is not None:
        rows.append(["Cases with a matching key", fmt_int(n_matched)])
    section.table(Table("Merge Summary", ["", "Value"], rows))
    for w in warnings:
        section.note("Warning", w, style="warning")
    return section


# ================================================================== AGGREGATE
AGG_FUNCTIONS = {
    # name: (number of value arguments, description)
    "MEAN": (0, "Mean"), "MEDIAN": (0, "Median"), "SUM": (0, "Sum"),
    "SD": (0, "Standard deviation"), "MIN": (0, "Minimum"), "MAX": (0, "Maximum"),
    "FIRST": (0, "First"), "LAST": (0, "Last"),
    "N": (0, "Weighted number of cases"), "NU": (0, "Unweighted number of cases"),
    "NMISS": (0, "Weighted number of missing cases"),
    "NUMISS": (0, "Unweighted number of missing cases"),
    "PGT": (1, "Percentage above"), "PLT": (1, "Percentage below"),
    "PIN": (2, "Percentage inside"), "POUT": (2, "Percentage outside"),
    "FGT": (1, "Fraction above"), "FLT": (1, "Fraction below"),
    "FIN": (2, "Fraction inside"), "FOUT": (2, "Fraction outside"),
    "CGT": (1, "Number above"), "CLT": (1, "Number below"),
    "CIN": (2, "Number inside"), "COUT": (2, "Number outside"),
}
_COUNTING = {"N", "NU", "NMISS", "NUMISS"}


@dataclass
class AggSpec:
    name: str
    function: str
    source: str
    args: tuple = ()
    label: str = ""

    def syntax(self) -> str:
        args = "".join(f", {_q(a)}" for a in self.args)
        return f"{self.name}={self.function}({self.source}{args})"


def default_agg_name(source: str, function: str) -> str:
    return f"{source}_{function.lower()}"


def _agg_group(values: pd.Series, spec: AggSpec, columnwise: bool) -> object:
    valid = values.dropna()
    f = spec.function
    if f in ("N", "NU"):
        return float(len(valid))
    if f in ("NMISS", "NUMISS"):
        return float(len(values) - len(valid))
    if columnwise and len(valid) < len(values):
        return np.nan
    if valid.empty:
        return np.nan
    if f == "FIRST":
        return valid.iloc[0]
    if f == "LAST":
        return valid.iloc[-1]
    if f == "MIN":
        return valid.min()
    if f == "MAX":
        return valid.max()
    x = valid.astype(float)
    if f == "MEAN":
        return float(x.mean())
    if f == "MEDIAN":
        return float(x.median())
    if f == "SUM":
        return float(x.sum())
    if f == "SD":
        return float(x.std(ddof=1)) if len(x) > 1 else np.nan
    a = spec.args
    if f[1:] == "GT":
        hit = x > a[0]
    elif f[1:] == "LT":
        hit = x < a[0]
    else:
        lo, hi = min(a[0], a[1]), max(a[0], a[1])
        inside = (x >= lo) & (x <= hi)
        hit = inside if f[1:] == "IN" else ~inside
    count = float(hit.sum())
    if f[0] == "C":
        return count
    frac = count / len(x)
    return 100 * frac if f[0] == "P" else frac


def aggregate(ds: Dataset, breaks: list[str], specs: list[AggSpec], mode: str = "add",
              n_break: str | None = None, columnwise: bool = False,
              dataset_name: str = "") -> tuple[Dataset | None, Section]:
    """AGGREGATE /BREAK=… /name=FUNCTION(var…). ``mode`` 'add' adds the aggregated values
    to every case of the active dataset (MODE=ADDVARIABLES); 'new' returns a new dataset
    with one case per break group (break variables + aggregated variables). Missing values in
    the summarised variables are excluded; missing break values form their own groups."""
    if not specs and not n_break:
        raise DataError("Select at least one variable to summarise or 'Number of cases'.")
    bnames = [ds.var(b).name for b in breaks]
    names_check = ds if mode == "add" else Dataset()
    taken: set[str] = set() if mode == "add" else {b.lower() for b in bnames}
    for spec in specs:
        if spec.function not in AGG_FUNCTIONS:
            raise DataError(f"Unknown function {spec.function}.")
        nargs = AGG_FUNCTIONS[spec.function][0]
        if len(spec.args) != nargs:
            raise DataError(f"{spec.function} needs {nargs} value argument(s).")
        var = ds.var(spec.source)
        if var.type == STRING and spec.function in ("MEAN", "MEDIAN", "SUM", "SD") \
                or var.type == STRING and nargs:
            raise DataError(f"{spec.function} cannot be used with string variable {var.name}.")
        spec.name = _check_new_name(names_check, spec.name, taken)
    if n_break:
        n_break = _check_new_name(names_check, n_break, taken)

    if bnames:
        keys = [ds.df[b].astype(object).where(ds.df[b].notna(), None) for b in bnames]
        grouped_index = pd.Series(range(ds.n_cases)).groupby(
            [k.reset_index(drop=True) for k in keys], dropna=False, sort=False)
        groups = list(grouped_index.groups.items())
    else:
        groups = [((), ds.df.index)]
    series = {s.source: ds.analysis_series(s.source) for s in specs}
    group_rows = []
    for key, idx in groups:
        idx = pd.Index(idx)
        row = {}
        for spec in specs:
            row[spec.name] = _agg_group(series[spec.source].loc[idx], spec, columnwise)
        if n_break:
            row[n_break] = float(len(idx))
        group_rows.append((key if isinstance(key, tuple) else (key,), idx, row))

    def new_variable(spec: AggSpec) -> Variable:
        src = ds.var(spec.source)
        f = spec.function
        if f in ("FIRST", "LAST", "MIN", "MAX"):
            v = _copy_var(src, spec.name)
            v.label = spec.label
            return v
        v = Variable(spec.name, NUMERIC, label=spec.label, measure=SCALE)
        if f in _COUNTING or f[0] == "C":
            v.decimals = 0
        elif f in ("MEAN", "MEDIAN", "SUM", "SD"):
            v.decimals = max(src.decimals, 2)
        else:
            v.decimals = 2
        return v

    out_vars = [new_variable(s) for s in specs]
    if n_break:
        out_vars.append(Variable(n_break, NUMERIC, decimals=0, measure=SCALE))
    section = Section("Aggregate")
    agg_lines = [f"  /{s.syntax()}" for s in specs]
    if n_break:
        agg_lines.append(f"  /{n_break}=N")
    labels = [f"VARIABLE LABELS {s.name} '{s.label}'." for s in specs if s.label]
    head = "AGGREGATE\n"
    new_ds = None
    if mode == "add":
        for v in out_vars:
            col = pd.Series(np.nan if v.type != STRING else None, index=ds.df.index,
                            dtype=object)
            for _, idx, row in group_rows:
                col.loc[idx] = row[v.name]
            ds.add_variable(v.name, v.type, values=col.to_numpy())
            ds.variables[ds.index_of(v.name)] = v
        head += "  /OUTFILE=* MODE=ADDVARIABLES\n"
    elif mode == "new":
        bvars = [_copy_var(ds.var(b)) for b in bnames]
        data = {b: [k[i] if k[i] is not None else np.nan for k, _, _ in group_rows]
                for i, b in enumerate(bnames)}
        for v in out_vars:
            data[v.name] = [row[v.name] for _, _, row in group_rows]
        df = pd.DataFrame(data)
        if bnames:
            df = df.sort_values(bnames, kind="mergesort", na_position="first")
        new_ds = Dataset(df.reset_index(drop=True), bvars + out_vars,
                         name=dataset_name or "Aggregated")
        head += f"  /OUTFILE='{dataset_name or 'Aggregated'}'\n"
    else:
        raise DataError(f"Unknown aggregate mode {mode}.")
    if columnwise:
        head += "  /MISSING=COLUMNWISE\n"
    cmd = head + f"  /BREAK={' '.join(bnames)}\n" + "\n".join(agg_lines) + "."
    if labels:
        cmd += "\n" + "\n".join(labels)
    section.log(cmd)
    rows = [[s.name, s.function + (f"({', '.join(format_code(a) for a in s.args)})"
                                   if s.args else ""), s.source, s.label] for s in specs]
    if n_break:
        rows.append([n_break, "N", "(cases)", "Number of cases in break group"])
    section.table(Table("Aggregated Variables", ["Variable", "Function", "Source", "Label"],
                        rows, footnotes=[f"Break variable(s): {', '.join(bnames) or '(none)'}; "
                                         f"{len(group_rows)} break group(s)."]))
    if mode == "new":
        section.note("", f"New dataset with {new_ds.n_cases} cases and {new_ds.n_vars} "
                         "variables opened in a new Data Editor window.")
    return new_ds, section


# ================================================================== TRANSPOSE
def transpose(ds: Dataset, variables: list[str], name_var: str | None = None
              ) -> tuple[Dataset, Section]:
    """FLIP VARIABLES=… /NEWNAMES=name_var: variables become cases. The new dataset has a
    string variable CASE_LBL holding the old variable names; the new variables are named from
    the name variable (or VAR001, VAR002…). String variables cannot be transposed."""
    if not variables:
        raise DataError("Select at least one variable to transpose.")
    names = [ds.var(v).name for v in variables if not name_var
             or ds.var(v).name.lower() != name_var.lower()]
    dropped = [n for n in names if ds.var(n).type != NUMERIC]
    names = [n for n in names if ds.var(n).type == NUMERIC]
    if not names:
        raise DataError("Only numeric variables can be transposed.")
    n = ds.n_cases
    if name_var:
        raw = ds.df[ds.var(name_var).name]
        base = [sanitize_name(format_code(v) if not (isinstance(v, float) and math.isnan(v))
                              else "", fallback=f"VAR{i + 1:03d}") for i, v in enumerate(raw)]
    else:
        base = [f"VAR{i + 1:03d}" for i in range(n)]
    new_names, used = [], {"case_lbl"}
    for b in base:
        name, k = b, 1
        while name.lower() in used:
            k += 1
            name = f"{b}_{k}"
        used.add(name.lower())
        new_names.append(name)
    matrix = ds.df[names].to_numpy(dtype=float).T
    df = pd.DataFrame(matrix, columns=new_names)
    df.insert(0, "CASE_LBL", names)
    width = max(8, max(len(x) for x in names))
    variables = [Variable("CASE_LBL", STRING, width=width, decimals=0, columns=min(width, 24),
                          align="left", measure=NOMINAL)]
    for nm in new_names:
        values = df[nm].dropna()
        integral = bool(len(values)) and bool(np.all(np.mod(values, 1) == 0))
        variables.append(Variable(nm, NUMERIC, decimals=0 if integral else 2))
    new_ds = Dataset(df, variables, name="Transposed")
    section = Section("Transpose")
    cmd = f"FLIP VARIABLES={' '.join(names)}"
    if name_var:
        cmd += f"\n  /NEWNAMES={ds.var(name_var).name}"
    section.log(cmd + ".")
    section.table(Table("Transpose Summary", ["", "Value"], [
        ["Variables transposed", fmt_int(len(names))],
        ["Cases in new dataset", fmt_int(new_ds.n_cases)],
        ["Variables in new dataset", fmt_int(new_ds.n_vars)]]))
    if dropped:
        section.note("Warning", "String variables cannot be transposed and were dropped: "
                     + ", ".join(dropped) + ".", style="warning")
    section.note("", "User-missing values are transposed as ordinary values. The transposed "
                     "data were opened in a new Data Editor window.")
    return new_ds, section


# ================================================================== RESTRUCTURE
def vars_to_cases(ds: Dataset, groups: list[tuple[str, list[str]]],
                  index_name: str | None = "index1", index_mode: str = "sequential",
                  id_name: str | None = "id", id_var: str | None = None,
                  fixed: list[str] | None = None, drop_null: bool = False
                  ) -> tuple[Dataset, Section]:
    """VARSTOCASES /MAKE target FROM v1 v2 … /INDEX=index1 /ID=id /KEEP=fixed /NULL=KEEP|DROP.
    Each case becomes one case per variable in the group(s). ``id_var`` uses an existing
    variable as case group identification; otherwise ``id_name`` gets the case number.
    ``index_mode`` 'sequential' (1, 2, …) or 'names' (the first group's variable names)."""
    if not groups:
        raise DataError("Select the variables to be transposed.")
    lengths = {len(vs) for _, vs in groups}
    if len(lengths) != 1 or 0 in lengths:
        raise DataError("Every target variable needs the same number of variables.")
    k = lengths.pop()
    used = {v.lower() for _, vs in groups for v in vs}
    if fixed is None:
        fixed = [v.name for v in ds.variables if v.name.lower() not in used
                 and (not id_var or v.name.lower() != id_var.lower())]
    fixed = [ds.var(f).name for f in fixed]
    taken: set[str] = {f.lower() for f in fixed}
    if id_var:
        taken.add(ds.var(id_var).name.lower())
    blank = Dataset()
    out_vars: list[Variable] = []
    data: dict[str, np.ndarray] = {}
    n = ds.n_cases
    if id_var:
        iv = _copy_var(ds.var(id_var))
        out_vars.append(iv)
        data[iv.name] = np.repeat(ds.df[iv.name].to_numpy(), k)
    elif id_name:
        id_name = _check_new_name(blank, id_name, taken)
        out_vars.append(Variable(id_name, NUMERIC, decimals=0, measure=NOMINAL))
        data[id_name] = np.repeat(np.arange(1, n + 1, dtype=float), k)
    for f in fixed:
        out_vars.append(_copy_var(ds.var(f)))
        data[f] = np.repeat(ds.df[f].to_numpy(), k)
    if index_name:
        index_name = _check_new_name(blank, index_name, taken)
        if index_mode == "names":
            labels = [ds.var(v).name for v in groups[0][1]]
            width = max(8, max(len(x) for x in labels))
            out_vars.append(Variable(index_name, STRING, width=width, decimals=0,
                                     align="left", measure=NOMINAL))
            data[index_name] = np.tile(np.array(labels, dtype=object), n)
        else:
            out_vars.append(Variable(index_name, NUMERIC, decimals=0, measure=NOMINAL))
            data[index_name] = np.tile(np.arange(1, k + 1, dtype=float), n)
    null = np.ones(n * k, dtype=bool)
    for target, vs in groups:
        target = _check_new_name(blank, target, taken)
        src = [ds.var(v) for v in vs]
        if len({v.type for v in src}) > 1:
            raise DataError(f"The variables for {target} must all be of the same type.")
        tv = _copy_var(src[0], target)
        tv.label = ""
        out_vars.append(tv)
        values = ds.df[[v.name for v in src]].to_numpy(dtype=object).reshape(-1)
        data[target] = values
        if src[0].type == STRING:
            null &= np.array([not (isinstance(x, str) and x.strip()) for x in values])
        else:
            null &= pd.isna(pd.Series(values)).to_numpy()
    df = pd.DataFrame(data)
    if drop_null:
        df = df[~null]
    new_ds = Dataset(df.reset_index(drop=True), out_vars, name="Restructured")

    make = [f"  /MAKE {t} FROM {' '.join(ds.var(v).name for v in vs)}" for t, vs in groups]
    cmd = "VARSTOCASES\n" + "\n".join(make)
    if index_name:
        cmd += f"\n  /INDEX={index_name}" + (f"({groups[0][0]})" if index_mode == "names"
                                             else f"({k})")
    if id_name and not id_var:
        cmd += f"\n  /ID={id_name}"
    keep = ([ds.var(id_var).name] if id_var else []) + fixed
    cmd += f"\n  /KEEP={' '.join(keep)}" if keep else ""
    cmd += f"\n  /NULL={'DROP' if drop_null else 'KEEP'}."
    section = Section("Restructure: Variables to Cases")
    section.log(cmd)
    section.table(Table("Processing Statistics", ["", "N"], [
        ["Variables In", fmt_int(ds.n_vars)], ["Variables Out", fmt_int(new_ds.n_vars)],
        ["Cases In", fmt_int(n)], ["Cases Out", fmt_int(new_ds.n_cases)]]))
    gen = [[v.name, v.label] for v in out_vars
           if v.name in ([id_name] if not id_var and id_name else []) + [index_name]
           + [t for t, _ in groups]]
    section.table(Table("Generated Variables", ["Name", "Label"], gen))
    return new_ds, section


def _index_text(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, float):
        return format_code(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y%m%d")
    return sanitize_name(str(value), fallback="blank")


def cases_to_vars(ds: Dataset, ids: list[str], index: list[str] | None = None,
                  groupby: str = "variable", separator: str = ".", autofix: bool = True,
                  sort: bool = True, drop: list[str] | None = None
                  ) -> tuple[Dataset, Section]:
    """CASESTOVARS /ID=… /INDEX=… /GROUPBY=VARIABLE|INDEX /SEPARATOR='.': one case per ID
    group; each varying variable is spread into var.index columns. With ``autofix`` (SPSS
    default) variables constant within every ID group are kept as single fixed variables.
    Without an index variable the cases of a group are numbered 1, 2, … in file order."""
    if not ids:
        raise DataError("Select at least one identifier variable.")
    id_names = [ds.var(i).name for i in ids]
    idx_names = [ds.var(i).name for i in (index or [])]
    drop_l = {d.lower() for d in (drop or [])}
    df = ds.df.copy()
    if sort:
        df = df.sort_values(id_names + idx_names, kind="mergesort",
                            na_position="first").reset_index(drop=True)
    gkey = df[id_names].astype(object).where(df[id_names].notna(), "\0nan")
    gid = gkey.groupby(id_names, sort=False).ngroup().to_numpy()
    warnings = []
    if idx_names:
        bad = df[idx_names].isna().any(axis=1).to_numpy()
        if bad.any():
            warnings.append(f"{int(bad.sum())} case(s) with a missing index value were "
                            "skipped.")
            df, gid = df[~bad].reset_index(drop=True), gid[~bad]
        idx_tuples = list(df[idx_names].itertuples(index=False, name=None))
    else:
        occ = pd.Series(gid).groupby(gid).cumcount().to_numpy() + 1
        idx_tuples = [(float(o),) for o in occ]
    pair = pd.Series(list(zip(gid, idx_tuples)))
    if pair.duplicated().any():
        raise DataError("An index value occurs more than once within an ID group; the "
                        "cases cannot be restructured.")
    levels = sorted(set(idx_tuples))
    lvl_pos = {t: i for i, t in enumerate(levels)}
    n_groups = int(gid.max()) + 1 if len(gid) else 0
    first_rows = pd.Series(range(len(gid))).groupby(gid).first().to_numpy()
    others = [v for v in ds.variables if v.name not in id_names + idx_names
              and v.name.lower() not in drop_l]
    out_vars = [_copy_var(ds.var(i)) for i in id_names]
    data = {i: df[i].to_numpy()[first_rows] for i in id_names}
    fixed_vars, varying = [], []
    for v in others:
        col = df[v.name].astype(object).where(df[v.name].notna(), "\0nan")
        constant = bool((col.groupby(gid).nunique() <= 1).all())
        (fixed_vars if autofix and constant else varying).append(v)
    for v in fixed_vars:
        out_vars.append(_copy_var(v))
        data[v.name] = df[v.name].to_numpy()[first_rows]
    rows_pos = [lvl_pos[t] for t in idx_tuples]
    spread: dict[tuple[str, int], np.ndarray] = {}
    for v in varying:
        values = df[v.name].to_numpy(dtype=object)
        for li in range(len(levels)):
            spread[(v.name, li)] = np.full(n_groups, np.nan, dtype=object)
        for r, (g, li) in enumerate(zip(gid, rows_pos)):
            spread[(v.name, li)][g] = values[r]
    order = ([(v, li) for v in varying for li in range(len(levels))] if groupby == "variable"
             else [(v, li) for li in range(len(levels)) for v in varying])
    taken = {x.lower() for x in data}
    created = []
    for v, li in order:
        suffix = separator.join(_index_text(x) for x in levels[li])
        name = f"{v.name}{separator}{suffix}"
        if not is_valid_name(name):
            name = sanitize_name(name)
        base, k = name, 1
        while name.lower() in taken:
            k += 1
            name = f"{base}_{k}"
        taken.add(name.lower())
        nv = _copy_var(v, name)
        out_vars.append(nv)
        data[name] = spread[(v.name, li)]
        created.append(name)
    new_ds = Dataset(pd.DataFrame(data), out_vars, name="Restructured")
    cmd = f"SORT CASES BY {' '.join(id_names + idx_names)}.\n" if sort else ""
    cmd += f"CASESTOVARS\n  /ID={' '.join(id_names)}"
    if idx_names:
        cmd += f"\n  /INDEX={' '.join(idx_names)}"
    cmd += f"\n  /GROUPBY={'VARIABLE' if groupby == 'variable' else 'INDEX'}"
    if separator != ".":
        cmd += f"\n  /SEPARATOR='{separator}'"
    if not autofix:
        cmd += "\n  /AUTOFIX=NO"
    if drop:
        cmd += f"\n  /DROP={' '.join(drop)}"
    section = Section("Restructure: Cases to Variables")
    section.log(cmd + ".")
    section.table(Table("Processing Statistics", ["", "N"], [
        ["Cases In", fmt_int(ds.n_cases)], ["Cases Out", fmt_int(new_ds.n_cases)],
        ["Variables In", fmt_int(ds.n_vars)], ["Variables Out", fmt_int(new_ds.n_vars)],
        ["Index Values", fmt_int(len(levels))]]))
    gen = [[v.name, ", ".join(c for c in created if c.startswith(v.name + separator))]
           for v in varying]
    if gen:
        section.table(Table("Generated Variables", ["Original Variable", "Result"], gen))
    if fixed_vars:
        section.note("", "Fixed variables (constant within ID groups): "
                     + ", ".join(v.name for v in fixed_vars) + ".")
    for w in warnings:
        section.note("Warning", w, style="warning")
    return new_ds, section

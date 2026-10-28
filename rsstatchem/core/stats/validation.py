"""Data > Validation (SPSS Data Preparation add-on: Load Predefined Rules, Define Rules,
Validate Data / VALIDATEDATA).

Rules live in the dataset's metadata (saved with the .rsd file), as SPSS keeps them in the
data dictionary:

* **single-variable rules**: type (numeric / string / date), valid values as a range
  [minimum, maximum] or a list, "allow noninteger values", "allow user-missing values",
  "allow system-missing values" and "allow blank values" (strings);
* **cross-variable rules**: a logical expression in the Compute language that evaluates to 1
  (true) for the cases that *violate* the rule.

Validate Data runs the **basic checks** (variables: maximum % missing, maximum % of cases in a
single category, maximum % of categories with count 1, minimum coefficient of variation,
minimum standard deviation; cases: incomplete IDs, duplicate IDs, empty cases), applies the
chosen single-variable rules to variables and the cross-variable rules, and reports the
Variable Checks, Rule Descriptions, Variable Summary, Cross-Variable Rules, Case Report and
identifier checks. Saved variables: EmptyCase, IncompleteID, DuplicateIDGroup,
ValidationRuleViolations and optionally one indicator per rule and variable.
"""
from __future__ import annotations

import copy
import re

import numpy as np
import pandas as pd

from rsstatchem.core.dataset import NUMERIC, STRING, Dataset
from rsstatchem.core.output import Section, Table
from rsstatchem.core.stats.common import fmt, fmt_int
from rsstatchem.core.stats.ml_common import MLError

ValidationError = MLError
META_KEY = "validation_rules"

PREDEFINED = [
    {"name": "0 to 1 Dichotomy", "type": "numeric", "domain": "list", "values": [0, 1],
     "noninteger": False, "user_missing": True, "sysmis": True},
    {"name": "1 to 2 Dichotomy", "type": "numeric", "domain": "list", "values": [1, 2],
     "noninteger": False, "user_missing": True, "sysmis": True},
    {"name": "Nonnegative integer", "type": "numeric", "domain": "range", "min": 0,
     "max": None, "noninteger": False, "user_missing": True, "sysmis": True},
    {"name": "Nonnegative number", "type": "numeric", "domain": "range", "min": 0,
     "max": None, "noninteger": True, "user_missing": True, "sysmis": True},
    {"name": "Positive integer", "type": "numeric", "domain": "range", "min": 1, "max": None,
     "noninteger": False, "user_missing": True, "sysmis": True},
    {"name": "Positive number", "type": "numeric", "domain": "range", "min": 1e-300,
     "max": None, "noninteger": True, "user_missing": True, "sysmis": True},
    {"name": "Nonmissing", "type": "numeric", "domain": "range", "min": None, "max": None,
     "noninteger": True, "user_missing": False, "sysmis": False},
    {"name": "Likert 1 to 4", "type": "numeric", "domain": "range", "min": 1, "max": 4,
     "noninteger": False, "user_missing": True, "sysmis": True},
    {"name": "Likert 1 to 5", "type": "numeric", "domain": "range", "min": 1, "max": 5,
     "noninteger": False, "user_missing": True, "sysmis": True},
    {"name": "Likert 1 to 7", "type": "numeric", "domain": "range", "min": 1, "max": 7,
     "noninteger": False, "user_missing": True, "sysmis": True},
    {"name": "Percentage 0 to 100", "type": "numeric", "domain": "range", "min": 0,
     "max": 100, "noninteger": True, "user_missing": True, "sysmis": True},
    {"name": "Proportion 0 to 1", "type": "numeric", "domain": "range", "min": 0, "max": 1,
     "noninteger": True, "user_missing": True, "sysmis": True},
    {"name": "Nonblank string", "type": "string", "domain": "range", "min": None, "max": None,
     "noninteger": True, "user_missing": False, "sysmis": False, "blank": False},
]

BASIC_DEFAULTS = {"max_missing": 70.0, "max_single": 95.0, "max_unique": 90.0,
                  "min_cv": 0.001, "min_sd": 0.0}


def rules(ds: Dataset) -> dict:
    """{'single': [...], 'cross': [...]} stored in the dataset metadata."""
    r = ds.meta.get(META_KEY)
    if not isinstance(r, dict):
        r = {"single": [], "cross": []}
    r.setdefault("single", [])
    r.setdefault("cross", [])
    return copy.deepcopy(r)


def set_rules(ds: Dataset, value: dict) -> None:
    ds.meta[META_KEY] = {"single": list(value.get("single", [])),
                         "cross": list(value.get("cross", []))}


def load_predefined(ds: Dataset, names: list[str] | None = None) -> Section:
    r = rules(ds)
    have = {x["name"].lower() for x in r["single"]}
    added = []
    for rule in PREDEFINED:
        if names is not None and rule["name"] not in names:
            continue
        if rule["name"].lower() not in have:
            r["single"].append(copy.deepcopy(rule))
            added.append(rule["name"])
    set_rules(ds, r)
    section = Section("Load Predefined Rules")
    section.log("* Load predefined validation rules.")
    section.table(Table("Rules Loaded", ["Rule"], [[n] for n in added] or [["(none new)"]]))
    return section


def describe_rule(rule: dict) -> str:
    parts = [rule.get("type", "numeric").capitalize()]
    if rule.get("domain") == "list":
        parts.append("valid values: " + ", ".join(str(v) for v in rule.get("values", [])))
    else:
        lo, hi = rule.get("min"), rule.get("max")
        if lo is not None or hi is not None:
            parts.append(f"range {'' if lo is None else _g(lo)} to {'' if hi is None else _g(hi)}")
    if rule.get("type", "numeric") == "numeric" and not rule.get("noninteger", True):
        parts.append("integers only")
    flags = []
    if rule.get("user_missing", True):
        flags.append("user-missing allowed")
    if rule.get("sysmis", True):
        flags.append("system-missing allowed")
    if rule.get("type") == "string" and rule.get("blank", True):
        flags.append("blanks allowed")
    return "; ".join(parts + flags)


def _g(v):
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


def violations(ds: Dataset, var: str, rule: dict) -> np.ndarray:
    """Boolean per case: the value breaks the single-variable rule."""
    v = ds.var(var)
    raw = ds.df[var]
    n = ds.n_cases
    bad = np.zeros(n, bool)
    if v.type == STRING:
        s = raw.astype(object).where(raw.notna(), "")
        txt = s.map(lambda x: str(x))
        blank = txt.str.strip() == ""
        user_miss = txt.isin([str(m) for m in v.missing])
        if not rule.get("blank", True):
            bad |= blank.to_numpy()
        if not rule.get("user_missing", True):
            bad |= user_miss.to_numpy()
        valid = (~blank & ~user_miss).to_numpy()
        if rule.get("domain") == "list" and rule.get("values"):
            allowed = {str(x) for x in rule["values"]}
            bad |= valid & ~txt.isin(allowed).to_numpy()
        elif rule.get("min") is not None or rule.get("max") is not None:
            lo, hi = rule.get("min"), rule.get("max")
            if lo is not None:
                bad |= valid & (txt < str(lo)).to_numpy()
            if hi is not None:
                bad |= valid & (txt > str(hi)).to_numpy()
        return bad
    x = pd.to_numeric(raw, errors="coerce").to_numpy(float) if v.type == NUMERIC else \
        pd.to_datetime(raw, errors="coerce").map(
            lambda t: t.timestamp() if pd.notna(t) else np.nan).to_numpy(float)
    sysmis = ~np.isfinite(x)
    user_miss = np.isin(x, np.array([m for m in v.missing if isinstance(m, (int, float))],
                                    float)) if v.missing else np.zeros(n, bool)
    if not rule.get("sysmis", True):
        bad |= sysmis
    if not rule.get("user_missing", True):
        bad |= user_miss
    valid = ~sysmis & ~user_miss
    if rule.get("domain") == "list" and rule.get("values"):
        allowed = np.array([float(a) for a in rule["values"]])
        bad |= valid & ~np.isin(x, allowed)
    else:
        lo, hi = rule.get("min"), rule.get("max")
        if lo is not None:
            bad |= valid & (x < float(lo))
        if hi is not None:
            bad |= valid & (x > float(hi))
    if not rule.get("noninteger", True):
        with np.errstate(invalid="ignore"):
            bad |= valid & (np.abs(x - np.round(x)) > 1e-9)
    return bad


def cross_violations(ds: Dataset, rule: dict) -> np.ndarray:
    from rsstatchem.core.expression import evaluate

    res = evaluate(ds, rule["expression"])
    vals = pd.to_numeric(res, errors="coerce").to_numpy(float)
    return np.nan_to_num(vals, nan=0.0) == 1


def _safe(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not s or not s[0].isalpha():
        s = "R_" + s
    return s[:40]


def validate_data(ds: Dataset, variables: list[str], ids: list[str] | None = None,
                  basic: dict | None = None, id_checks: bool = True,
                  empty_check: bool = True, assignments: dict | None = None,
                  cross: list[str] | None = None, case_report_limit: int = 500,
                  save: dict | None = None, minimum_violations: int = 1) -> Section:
    """``assignments`` maps a variable to the names of the single-variable rules applied to
    it; ``cross`` names the cross-variable rules; ``save`` keys: empty, incomplete,
    duplicate, total, indicators (bool) and ``replace``."""
    if not variables and not cross and not ids:
        raise ValidationError("Select analysis variables, identifiers or cross-variable rules.")
    r = rules(ds)
    single = {x["name"]: x for x in r["single"]}
    crossd = {x["name"]: x for x in r["cross"]}
    assignments = {k: list(v) for k, v in (assignments or {}).items() if v}
    for var, names in assignments.items():
        for nm in names:
            if nm not in single:
                raise ValidationError(f"Unknown rule {nm!r}.")
    section = Section("Validate Data")
    section.log("VALIDATEDATA" + (f" VARIABLES={' '.join(variables)}" if variables else "")
                + (f" ID={' '.join(ids)}" if ids else "")
                + (("\n  /VARCHECKS STATUS=ON PCTMISSING={max_missing:g} "
                    "PCTEQUAL={max_single:g} PCTUNEQUAL={max_unique:g} CV={min_cv:g} "
                    "STDDEV={min_sd:g}").format(**{**BASIC_DEFAULTS, **basic}) if basic is not
                   None else "")
                + ("\n  /IDCHECKS INCOMPLETE DUPLICATE" if ids and id_checks else "")
                + ("\n  /CASECHECKS REPORTEMPTY=YES" if empty_check else "")
                + ("\n  /RULESUMMARIES BYVARIABLE" if assignments else "")
                + (f"\n  /CASEREPORT DISPLAY=YES MINVIOLATIONS={minimum_violations}"
                   f" CASELIMIT={case_report_limit}" if assignments or cross else "") + ".")
    warnings = []
    # ------------------------------------------------------------------ basic checks
    if basic is not None:
        b = {**BASIC_DEFAULTS, **basic}
        rows = []
        for v in variables:
            var = ds.var(v)
            s = ds.analysis_series(v)
            n = ds.n_cases
            miss = int(s.isna().sum()) if var.type != STRING else int(
                (s.astype(str).str.strip() == "").sum() + s.isna().sum())
            pmiss = 100 * miss / n if n else 0
            if pmiss > b["max_missing"]:
                rows.append([v, "Missing Values > " + _g(b["max_missing"]) + "%",
                             fmt(pmiss, 1) + "%"])
            valid = s.dropna()
            if var.measure in ("nominal", "ordinal") or var.type == STRING:
                if len(valid):
                    vc = valid.value_counts()
                    top = 100 * vc.iloc[0] / len(valid)
                    if top > b["max_single"]:
                        rows.append([v, "Cases in a Single Category > " + _g(b["max_single"])
                                     + "%", fmt(top, 1) + "%"])
                    ones = 100 * (vc == 1).sum() / len(vc)
                    if ones > b["max_unique"]:
                        rows.append([v, "Categories with Count of 1 > " + _g(b["max_unique"])
                                     + "%", fmt(ones, 1) + "%"])
            elif var.type == NUMERIC and len(valid) > 1:
                x = valid.to_numpy(float)
                sd = x.std(ddof=1)
                m = x.mean()
                cv = abs(sd / m) if m != 0 else np.inf
                if cv < b["min_cv"]:
                    rows.append([v, "Coefficient of Variation < " + _g(b["min_cv"]),
                                 fmt(cv, 4)])
                if sd <= b["min_sd"]:
                    rows.append([v, "Standard Deviation <= " + _g(b["min_sd"]), fmt(sd, 4)])
        if rows:
            section.table(Table("Variable Checks", ["Variable", "Check", "Value"], rows,
                                footnotes=["Variables that fail the basic checks."]))
        else:
            warnings.append("All analysis variables passed the basic checks.")
    # ------------------------------------------------------------------ identifiers / empty
    incomplete = np.zeros(ds.n_cases, bool)
    dup_group = np.full(ds.n_cases, np.nan)
    if ids:
        for i in ids:
            incomplete |= ds.analysis_series(i).isna().to_numpy() if ds.var(i).type != STRING \
                else (ds.df[i].astype(str).str.strip() == "").to_numpy()
        if id_checks:
            key = pd.Series(list(zip(*[ds.df[i].astype(str).to_numpy() for i in ids])))
            dup = key.duplicated(keep=False).to_numpy() & ~incomplete
            groups = {}
            for row in np.where(dup)[0]:
                groups.setdefault(key.iloc[row], len(groups) + 1)
                dup_group[row] = groups[key.iloc[row]]
            rows = [[str(i + 1), "Incomplete ID"] for i in np.where(incomplete)[0]] + \
                [[str(i + 1), f"Duplicate ID (group {int(dup_group[i])})"]
                 for i in np.where(dup)[0]]
            if rows:
                section.table(Table("Identifier Checks", ["Case", "Problem"],
                                    rows[:case_report_limit]))
            else:
                warnings.append("No incomplete or duplicate identifiers.")
    empty = np.zeros(ds.n_cases, bool)
    if empty_check:
        check_vars = [v.name for v in ds.variables if v.name not in (ids or [])]
        if check_vars:
            empty = np.ones(ds.n_cases, bool)
            for v in check_vars:
                s = ds.df[v]
                if ds.var(v).type == STRING:
                    empty &= (s.astype(str).str.strip().isin(["", "nan", "None"])).to_numpy()
                else:
                    empty &= s.isna().to_numpy()
        if empty.any():
            section.table(Table("Empty Cases", ["Case"], [[str(i + 1)] for i in
                                                         np.where(empty)[0]][:case_report_limit]))
    # ------------------------------------------------------------------ rules
    indicator = {}
    total = np.zeros(ds.n_cases, int)
    per_case: dict[int, list[str]] = {}
    if assignments:
        used = sorted({nm for names in assignments.values() for nm in names})
        section.table(Table("Rule Descriptions", ["Rule", "Description"],
                            [[nm, describe_rule(single[nm])] for nm in used]))
        rows = []
        for var, names in assignments.items():
            for nm in names:
                bad = violations(ds, var, single[nm])
                indicator[(nm, var)] = bad
                total += bad
                for i in np.where(bad)[0]:
                    per_case.setdefault(int(i), []).append(f"{nm} ({var})")
                rows.append([var, nm, fmt_int(int(bad.sum()))])
        section.table(Table("Variable Summary", ["Variable", "Rule", "Number of Violations"],
                            rows))
    if cross:
        rows = []
        for nm in cross:
            if nm not in crossd:
                raise ValidationError(f"Unknown cross-variable rule {nm!r}.")
            bad = cross_violations(ds, crossd[nm])
            indicator[(nm, None)] = bad
            total += bad
            for i in np.where(bad)[0]:
                per_case.setdefault(int(i), []).append(nm)
            rows.append([nm, fmt_int(int(bad.sum())), crossd[nm]["expression"]])
        section.table(Table("Cross-Variable Rules", ["Rule", "Number of Violations",
                                                     "Rule Expression"], rows))
    if assignments or cross:
        cases = [i for i in sorted(per_case) if len(per_case[i]) >= minimum_violations]
        if cases:
            id_txt = (lambda i: ", ".join(str(ds.df[x].iloc[i]) for x in ids)) if ids else None
            rows = [[str(i + 1)] + ([id_txt(i)] if id_txt else [])
                    + ["; ".join(per_case[i]), fmt_int(len(per_case[i]))]
                    for i in cases[:case_report_limit]]
            section.table(Table("Case Report", ["Case"] + (["Identifier"] if ids else [])
                                + ["Validation Rule Violations", "Total"], rows,
                                footnotes=([f"Only the first {case_report_limit} cases are "
                                            "listed."] if len(cases) > case_report_limit
                                           else [])))
        else:
            warnings.append("No validation rule violations.")
    for w in warnings:
        section.note("", w)
    if save:
        _save(ds, save, empty, incomplete, dup_group, total, indicator, bool(ids))
    section.result = {"total": total, "indicators": indicator, "empty": empty,  # type: ignore
                      "incomplete": incomplete, "duplicate": dup_group}
    return section


def _save(ds, opts, empty, incomplete, dup, total, indicator, has_ids):
    def put(name, values, label):
        if ds.has_name(name):
            if not opts.get("replace"):
                raise ValidationError(f"A variable named {name} already exists (tick Replace "
                                      "existing summary variables).")
            ds.delete_variables([ds.index_of(name)])
        var = ds.add_variable(name, values=np.asarray(values, float))
        var.label, var.decimals = label, 0
        return var

    if opts.get("empty"):
        put("EmptyCase", empty.astype(float), "Empty case indicator")
    if opts.get("incomplete") and has_ids:
        put("IncompleteID", incomplete.astype(float), "Incomplete ID indicator")
    if opts.get("duplicate") and has_ids:
        put("DuplicateIDGroup", np.where(np.isnan(dup), 0, dup), "Duplicate ID group")
    if opts.get("total"):
        put("ValidationRuleViolations", total.astype(float),
            "Total count of validation rule violations")
    if opts.get("indicators"):
        for (nm, var), bad in indicator.items():
            name = _safe(f"{nm}_{var}" if var else nm)
            put(name, bad.astype(float), f"Violation of rule {nm}" + (f" by {var}" if var
                                                                     else ""))


def selftest() -> None:
    ds = Dataset(pd.DataFrame({"a": [1.0, 2.0, 7.0, np.nan], "b": [0.0, 1.0, 1.0, 2.0]}))
    load_predefined(ds)
    sec = validate_data(ds, ["a", "b"], assignments={"a": ["Likert 1 to 5"],
                                                    "b": ["0 to 1 Dichotomy"]})
    assert list(sec.result["total"]) == [0, 0, 1, 1]

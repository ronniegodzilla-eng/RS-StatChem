"""Transform > Compute Variable and Recode into Same / Different Variables."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from rsstatchem.core.dataset import NUMERIC, STRING, Dataset, format_code, is_valid_name
from rsstatchem.core.expression import ExpressionError, evaluate, is_string_result
from rsstatchem.core.output import Section


def _condition_mask(ds: Dataset, condition: str | None) -> pd.Series:
    if not condition or not condition.strip():
        return pd.Series(True, index=ds.df.index)
    value = evaluate(ds, condition)
    if is_string_result(value):
        raise ExpressionError("The If condition must be a logical (numeric) expression.")
    return value == 1  # false and missing conditions both skip the case, as in SPSS


# ------------------------------------------------------------------ compute
def compute(ds: Dataset, target: str, expression: str, vtype: str | None = None,
            label: str | None = None, width: int | None = None,
            condition: str | None = None) -> Section:
    """Create or replace ``target``. Cases failing ``condition`` keep their old value
    (system-missing for a new variable)."""
    target = target.strip()
    if not is_valid_name(target):
        raise ExpressionError(f"'{target}' is not a valid variable name.")
    value = evaluate(ds, expression)
    result_is_string = is_string_result(value)
    mask = _condition_mask(ds, condition)
    exists = ds.has_name(target)

    if exists:
        var = ds.var(target)
        if (var.type == STRING) != result_is_string:
            kind = "string" if var.type == STRING else "numeric"
            raise ExpressionError(f"{var.name} is a {kind} variable; the expression gives a "
                                  f"{'string' if result_is_string else 'numeric'} result.")
        base = ds.df[var.name].copy()
    else:
        vtype = vtype or (STRING if result_is_string else NUMERIC)
        if (vtype == STRING) != result_is_string:
            raise ExpressionError("Type mismatch: the target type does not match the expression "
                                  "(use STRING() or NUMBER() to convert).")
        base = pd.Series([np.nan] * ds.n_cases, index=ds.df.index,
                         dtype=object if vtype == STRING else float)

    if result_is_string:
        value = value.where(value.notna(), "")
        if width:
            value = value.astype(str).str.slice(0, width)
        new = base.astype(object).where(~mask, value)
    else:
        new = base.astype(float).where(~mask, value.astype(float))

    if exists:
        ds.df[var.name] = new
        if label is not None and label != "":
            var.label = label
        ds.modified = True
    else:
        var = ds.add_variable(target, vtype, values=new.to_numpy())
        if label:
            var.label = label
        if vtype == STRING:
            var.width = width or max(8, int(new.astype(str).str.len().max() or 8))
            var.columns = min(var.width, 24)

    section = Section("Compute Variable")
    syntax = (f"IF ({condition.strip()}) {var.name}={expression.strip()}."
              if condition and condition.strip() else f"COMPUTE {var.name}={expression.strip()}.")
    if label:
        syntax += f"\nVARIABLE LABELS {var.name} '{label}'."
    section.log(syntax + "\nEXECUTE.")
    computed = int(mask.sum())
    missing = int(pd.Series(new).isna().sum()) if not result_is_string else 0
    section.note("", f"{'Changed existing' if exists else 'Created new'} variable {var.name}: "
                     f"{computed} of {ds.n_cases} cases computed"
                     + (f", {missing} system-missing." if missing else "."))
    return section


# ------------------------------------------------------------------ recode
@dataclass(frozen=True)
class Rule:
    """old: ("value", v) | ("sysmis",) | ("missing",) | ("range", lo, hi) | ("lowest", hi)
    | ("highest", lo) | ("else",);  new: ("value", v) | ("sysmis",) | ("copy",)"""
    old: tuple
    new: tuple

    def old_text(self) -> str:
        kind = self.old[0]
        if kind == "value":
            return _quote(self.old[1])
        if kind == "range":
            return f"{format_code(self.old[1])} thru {format_code(self.old[2])}"
        if kind == "lowest":
            return f"Lowest thru {format_code(self.old[1])}"
        if kind == "highest":
            return f"{format_code(self.old[1])} thru Highest"
        return {"sysmis": "SYSMIS", "missing": "MISSING", "else": "ELSE"}[kind]

    def new_text(self) -> str:
        kind = self.new[0]
        if kind == "value":
            return _quote(self.new[1])
        return {"sysmis": "SYSMIS", "copy": "Copy"}[kind]

    def syntax(self) -> str:
        new = "COPY" if self.new[0] == "copy" else self.new_text()
        old = (self.old_text().replace("Lowest", "LOWEST").replace("Highest", "HIGHEST")
               .replace(" thru ", " THRU "))
        return f"({old}={new})"


def _quote(value) -> str:
    return f"'{value}'" if isinstance(value, str) else format_code(value)


def _matches(rule: Rule, raw: pd.Series, missing_codes: list, remaining: pd.Series,
             numeric: bool) -> pd.Series:
    kind = rule.old[0]
    sysmis = raw.isna() if numeric else (raw.isna() | (raw == ""))
    if kind == "value":
        return (raw == rule.old[1]) & ~raw.isna()
    if kind == "sysmis":
        return sysmis
    if kind == "missing":
        return sysmis | raw.isin(missing_codes)
    if kind == "else":
        return remaining
    if not numeric:
        raise ExpressionError("Ranges can only be used with numeric variables.")
    x = pd.to_numeric(raw, errors="coerce")
    if kind == "range":
        return (x >= rule.old[1]) & (x <= rule.old[2])
    if kind == "lowest":
        return x <= rule.old[1]
    if kind == "highest":
        return x >= rule.old[1]
    raise ExpressionError(f"Unknown old value: {rule.old}")


def recode(ds: Dataset, sources: list[str], rules: list[Rule], targets: list[str] | None = None,
           labels: list[str] | None = None, output_string: bool = False, width: int = 8,
           convert: bool = False, condition: str | None = None) -> Section:
    """Recode into the same variables (``targets=None``) or into new ones."""
    if not sources:
        raise ExpressionError("Select at least one variable to recode.")
    if not rules and not convert:
        raise ExpressionError("Specify at least one old → new value pair.")
    into_same = targets is None
    if not into_same:
        if len(targets) != len(sources):
            raise ExpressionError("Every input variable needs an output variable name.")
        for t in targets:
            if not is_valid_name(t):
                raise ExpressionError(f"'{t}' is not a valid output variable name.")
        if len({t.lower() for t in targets}) != len(targets):
            raise ExpressionError("Output variable names must be different.")
    mask = _condition_mask(ds, condition)
    lines, notes = [], []

    for k, src in enumerate(sources):
        svar = ds.var(src)
        numeric_src = svar.type == NUMERIC
        raw = ds.df[svar.name]
        if into_same:
            out_numeric = numeric_src
        else:
            out_numeric = not output_string
        _check_new_values(rules, out_numeric, into_same, svar.name)
        if any(r.new[0] == "copy" for r in rules) and out_numeric != numeric_src and not convert:
            raise ExpressionError("'Copy old value' needs input and output of the same type.")

        if into_same:
            result = raw.copy().astype(float if out_numeric else object)
            target_name = svar.name
        else:
            target_name = targets[k]
            if ds.has_name(target_name):
                tvar = ds.var(target_name)
                if (tvar.type == NUMERIC) != out_numeric:
                    raise ExpressionError(f"Existing variable {target_name} has another type.")
                result = ds.df[tvar.name].copy()
            else:
                result = pd.Series(np.nan, index=raw.index,
                                   dtype=float if out_numeric else object)
        assigned = pd.Series(False, index=raw.index)
        for rule in rules:
            m = _matches(rule, raw, svar.missing, ~assigned, numeric_src) & ~assigned & mask
            if not m.any():
                assigned |= m
                continue
            if rule.new[0] == "value":
                result[m] = rule.new[1]
            elif rule.new[0] == "sysmis":
                result[m] = np.nan if out_numeric else ""
            else:  # copy
                result[m] = raw[m] if numeric_src == out_numeric else raw[m].astype(str)
            assigned |= m
        if convert and not numeric_src and out_numeric:
            rest = ~assigned & mask
            result[rest] = pd.to_numeric(raw[rest].astype(str).str.strip()
                                         .str.replace(",", ".", regex=False), errors="coerce")
            assigned |= rest
        if not into_same:
            # unmatched cases that satisfy the condition become system-missing
            result[~assigned & mask] = np.nan if out_numeric else ""

        changed = int((assigned & mask).sum())
        if into_same:
            ds.df[svar.name] = result.astype(float) if out_numeric else result
            ds.modified = True
        elif ds.has_name(target_name):
            ds.df[ds.var(target_name).name] = result
            ds.modified = True
        else:
            var = ds.add_variable(target_name, NUMERIC if out_numeric else STRING,
                                  values=result.to_numpy())
            if out_numeric:
                values = pd.to_numeric(result, errors="coerce").dropna()
                var.decimals = 0 if len(values) and bool(np.all(np.mod(values, 1) == 0)) else 2
            else:
                var.width = width
            if labels and k < len(labels) and labels[k]:
                var.label = labels[k]
        lines.append(svar.name)
        notes.append(f"{svar.name}{'' if into_same else ' → ' + target_name}: "
                     f"{changed} value(s) recoded")

    section = Section("Recode")
    spec = " ".join(r.syntax() for r in rules) + (" (CONVERT)" if convert else "")
    head = f"DO IF ({condition.strip()}).\n" if condition and condition.strip() else ""
    tail = "\nEND IF." if head else ""
    if into_same:
        cmd = f"RECODE {' '.join(lines)} {spec}."
    else:
        cmd = f"RECODE {' '.join(lines)} {spec} INTO {' '.join(targets)}."
        for t, lbl in zip(targets, labels or []):
            if lbl:
                cmd += f"\nVARIABLE LABELS {t} '{lbl}'."
    section.log(head + cmd + tail + "\nEXECUTE.")
    section.note("", "; ".join(notes) + ".")
    return section


def _check_new_values(rules: list[Rule], out_numeric: bool, into_same: bool, name: str) -> None:
    for r in rules:
        if r.new[0] == "copy" and into_same:
            raise ExpressionError("'Copy old value' is only available for Recode into "
                                  "Different Variables.")
        if r.new[0] == "value":
            if out_numeric and isinstance(r.new[1], str):
                raise ExpressionError(f"New value '{r.new[1]}' must be numeric for {name}.")
            if not out_numeric and not isinstance(r.new[1], str):
                raise ExpressionError(f"New value {r.new[1]} must be a string for {name}.")

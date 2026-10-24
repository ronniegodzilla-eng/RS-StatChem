"""In-memory dataset: a pandas DataFrame plus SPSS-style variable metadata.

Column storage is normalised so the rest of the app can rely on it:
numeric -> float64 (NaN = system missing), string -> object (Python str or NaN),
date -> datetime64[ns] (NaT = missing). The DataFrame always has a RangeIndex.
This module has no Qt dependency.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

NUMERIC, STRING, DATE = "numeric", "string", "date"
TYPES = (NUMERIC, STRING, DATE)
SCALE, ORDINAL, NOMINAL = "scale", "ordinal", "nominal"
MEASURES = (SCALE, ORDINAL, NOMINAL)
ROLES = ("input", "target", "both", "none", "partition", "split")
ALIGNS = ("left", "right", "center")

_NAME_RE = re.compile(r"^[^\W\d][\w.$#@]*$")  # SPSS also allows $ # @ after the first letter
_DATE_FMT = "%Y-%m-%d"


def format_number(value: float, decimals: int) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.{max(decimals, 0)}f}"


def format_code(code: Any) -> str:
    """Show a value-label code without a pointless ``.0``."""
    if isinstance(code, float) and code.is_integer():
        return str(int(code))
    return str(code)


@dataclass
class Variable:
    name: str
    type: str = NUMERIC
    width: int = 8
    decimals: int = 2
    label: str = ""
    value_labels: dict = field(default_factory=dict)
    missing: list = field(default_factory=list)
    columns: int = 8
    align: str = "right"
    measure: str = SCALE
    role: str = "input"

    @property
    def title(self) -> str:
        """Label if set, otherwise name — what SPSS prints in output tables."""
        return self.label or self.name

    def label_for(self, value: Any) -> str | None:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return self.value_labels.get(value)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["value_labels"] = [[k, v] for k, v in self.value_labels.items()]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Variable":
        d = dict(d)
        d["value_labels"] = {k: v for k, v in d.get("value_labels", [])}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def is_valid_name(name: str) -> bool:
    return bool(name) and len(name) <= 64 and bool(_NAME_RE.match(name)) and not name.endswith(".")


def sanitize_name(raw: Any, fallback: str = "VAR") -> str:
    name = re.sub(r"[^\w.]", "_", str(raw).strip())
    name = re.sub(r"_+", "_", name).strip("_.")
    if not name:
        name = fallback
    if not _NAME_RE.match(name):
        name = "V" + name
    return name[:64]


def _coerce_column(series: pd.Series, vtype: str) -> pd.Series:
    if vtype == NUMERIC:
        if pd.api.types.is_bool_dtype(series):
            series = series.astype(float)
        return pd.to_numeric(series, errors="coerce").astype("float64")
    if vtype == DATE:
        return pd.to_datetime(series, errors="coerce").astype("datetime64[ns]")
    out = series.astype(object)
    return out.where(out.isna(), out.map(lambda v: v if isinstance(v, str) else format_code(v)))


def infer_variable(name: str, series: pd.Series) -> Variable:
    if pd.api.types.is_datetime64_any_dtype(series):
        return Variable(name=name, type=DATE, width=10, decimals=0, align="right", measure=SCALE)
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        values = pd.to_numeric(series, errors="coerce").dropna()
        integral = bool(len(values)) and bool(np.all(np.mod(values, 1) == 0))
        return Variable(name=name, type=NUMERIC, width=8, decimals=0 if integral else 2,
                        align="right", measure=SCALE)
    lengths = series.dropna().astype(str).str.len()
    width = int(max(8, lengths.max() if len(lengths) else 8))
    return Variable(name=name, type=STRING, width=width, decimals=0, columns=min(width, 24),
                    align="left", measure=NOMINAL)


class Dataset:
    def __init__(self, df: pd.DataFrame | None = None, variables: list[Variable] | None = None,
                 name: str = "DataSet1"):
        df = pd.DataFrame() if df is None else df
        if variables is None:
            df, variables = self._prepare(df)
        self.variables: list[Variable] = list(variables)
        self.df = pd.DataFrame(
            {v.name: _coerce_column(df[v.name], v.type) for v in self.variables},
            index=pd.RangeIndex(len(df)),
        )
        self.name = name
        self.path: str | None = None
        self.modified = False
        # Free-form, JSON-serialisable metadata saved with the project (e.g. a DOE design).
        self.meta: dict = {}

    @staticmethod
    def _prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, list[Variable]]:
        """Make column names valid and unique; remember the original header as the label."""
        used: set[str] = set()
        rename, variables = {}, []
        for i, col in enumerate(df.columns):
            base = sanitize_name(col, fallback=f"VAR{i + 1:05d}")
            name, k = base, 1
            while name.lower() in used:
                k += 1
                name = f"{base}_{k}"
            used.add(name.lower())
            rename[col] = name
        df = df.copy()
        df.columns = [rename[c] for c in df.columns]
        for orig, name in rename.items():
            var = infer_variable(name, df[name])
            if str(orig) != name:
                var.label = str(orig)
            variables.append(var)
        return df.reset_index(drop=True), variables

    # ----------------------------------------------------------------- queries
    @property
    def n_cases(self) -> int:
        return len(self.df)

    @property
    def n_vars(self) -> int:
        return len(self.variables)

    def names(self) -> list[str]:
        return [v.name for v in self.variables]

    def index_of(self, name: str) -> int:
        lowered = name.lower()
        for i, v in enumerate(self.variables):
            if v.name.lower() == lowered:
                return i
        raise KeyError(name)

    def var(self, name: str) -> Variable:
        return self.variables[self.index_of(name)]

    def has_name(self, name: str, exclude: int | None = None) -> bool:
        lowered = name.lower()
        return any(v.name.lower() == lowered for i, v in enumerate(self.variables) if i != exclude)

    def next_var_name(self) -> str:
        k = 1
        while self.has_name(f"VAR{k:05d}"):
            k += 1
        return f"VAR{k:05d}"

    def get_value(self, row: int, col: int) -> Any:
        return self.df.iat[row, col]

    def display_value(self, row: int, col: int, use_labels: bool = False) -> str:
        var = self.variables[col]
        value = self.df.iat[row, col]
        if pd.isna(value):
            return "" if var.type != NUMERIC else "."
        if use_labels:
            label = var.label_for(value)
            if label is not None:
                return label
        if var.type == NUMERIC:
            return format_number(value, var.decimals)
        if var.type == DATE:
            return value.strftime(_DATE_FMT)
        return str(value)

    def edit_text(self, row: int, col: int) -> str:
        var = self.variables[col]
        value = self.df.iat[row, col]
        if pd.isna(value):
            return ""
        if var.type == NUMERIC:
            return format_code(float(value)) if float(value).is_integer() else repr(float(value))
        if var.type == DATE:
            return value.strftime(_DATE_FMT)
        return str(value)

    def analysis_series(self, name: str) -> pd.Series:
        """Column with user-missing codes (and empty strings) turned into NaN."""
        var = self.var(name)
        s = self.df[var.name]
        if var.type == STRING:
            s = s.where(s != "", np.nan)
        if var.missing:
            s = s.where(~s.isin(var.missing), np.nan)
        return s

    # ----------------------------------------------------------------- edits
    def _touch(self) -> None:
        self.modified = True

    def parse_value(self, col: int, text: Any) -> Any:
        var = self.variables[col]
        if text is None:
            return np.nan if var.type != DATE else pd.NaT
        if not isinstance(text, str):
            return text
        raw = text.strip()
        if var.type == STRING:
            return raw
        if raw in ("", "."):
            return np.nan if var.type == NUMERIC else pd.NaT
        if var.type == NUMERIC:
            for code, label in var.value_labels.items():
                if str(label).lower() == raw.lower():
                    return float(code)
            candidate = raw.replace(",", ".") if "," in raw and "." not in raw else raw
            try:
                return float(candidate)
            except ValueError:
                raise ValueError(f"'{text}' is not a valid number for {var.name}.") from None
        iso = re.match(r"^\d{4}-\d{1,2}-\d{1,2}", raw) is not None
        try:
            return pd.to_datetime(raw, dayfirst=not iso)
        except (ValueError, TypeError):
            raise ValueError(f"'{text}' is not a valid date for {var.name}.") from None

    def set_value(self, row: int, col: int, text: Any) -> None:
        self.ensure_size(0, col + 1)
        value = self.parse_value(col, text)
        self.ensure_size(row + 1, col + 1)
        self.df.iat[row, col] = value
        self._touch()

    def ensure_size(self, n_rows: int, n_cols: int) -> bool:
        grew = False
        while self.n_vars < n_cols:
            self.add_variable()
            grew = True
        if self.n_cases < n_rows:
            self.df = self.df.reindex(pd.RangeIndex(n_rows))
            grew = True
        return grew

    def add_variable(self, name: str | None = None, vtype: str = NUMERIC,
                     index: int | None = None, values: Iterable | None = None) -> Variable:
        name = name or self.next_var_name()
        if not is_valid_name(name):
            raise ValueError(f"'{name}' is not a valid variable name.")
        if self.has_name(name):
            raise ValueError(f"A variable named '{name}' already exists.")
        var = Variable(name=name, type=vtype, align="left" if vtype == STRING else "right",
                       measure=NOMINAL if vtype == STRING else SCALE,
                       decimals=0 if vtype != NUMERIC else 2)
        index = self.n_vars if index is None else max(0, min(index, self.n_vars))
        if values is None:
            column = pd.Series([np.nan] * self.n_cases, index=self.df.index)
        else:
            column = pd.Series(list(values), index=self.df.index)
        self.df.insert(index, name, _coerce_column(column, vtype))
        self.variables.insert(index, var)
        self._touch()
        return var

    def delete_variables(self, indices: Iterable[int]) -> None:
        indices = sorted({i for i in indices if 0 <= i < self.n_vars}, reverse=True)
        if not indices:
            return
        self.df = self.df.drop(columns=[self.variables[i].name for i in indices])
        for i in indices:
            del self.variables[i]
        self._touch()

    def rename_variable(self, col: int, new_name: str) -> None:
        new_name = new_name.strip()
        if not is_valid_name(new_name):
            raise ValueError(
                f"'{new_name}' is not a valid variable name. Names must start with a letter "
                "and contain only letters, digits, '_' or '.'.")
        if self.has_name(new_name, exclude=col):
            raise ValueError(f"A variable named '{new_name}' already exists.")
        old = self.variables[col].name
        self.df = self.df.rename(columns={old: new_name})
        self.variables[col].name = new_name
        self._touch()

    def change_type(self, col: int, new_type: str) -> None:
        var = self.variables[col]
        if new_type == var.type:
            return
        if new_type not in TYPES:
            raise ValueError(new_type)
        series = self.df[var.name]
        if var.type == DATE and new_type == STRING:
            series = series.dt.strftime(_DATE_FMT)
        self.df[var.name] = _coerce_column(series, new_type)
        var.type = new_type
        if new_type == STRING:
            var.align, var.decimals = "left", 0
            var.measure = NOMINAL if var.measure == SCALE else var.measure
            var.value_labels = {format_code(k): v for k, v in var.value_labels.items()}
            var.missing = [format_code(m) for m in var.missing]
        else:
            var.align = "right"
            var.decimals = 2 if new_type == NUMERIC else 0
            if new_type == NUMERIC:
                var.value_labels = _numeric_keys(var.value_labels)
                var.missing = [float(m) for m in var.missing if _is_number(m)]
            else:
                var.value_labels, var.missing = {}, []
        self._touch()

    def insert_cases(self, at: int, count: int = 1) -> None:
        at = max(0, min(at, self.n_cases))
        blank = pd.DataFrame({v.name: _coerce_column(pd.Series([np.nan] * count), v.type)
                              for v in self.variables})
        self.df = pd.concat([self.df.iloc[:at], blank, self.df.iloc[at:]], ignore_index=True)
        self._touch()

    def delete_cases(self, rows: Iterable[int]) -> None:
        rows = [r for r in set(rows) if 0 <= r < self.n_cases]
        if not rows:
            return
        self.df = self.df.drop(index=rows).reset_index(drop=True)
        self._touch()

    def sort_cases(self, names: list[str], ascending: list[bool]) -> None:
        self.df = self.df.sort_values(names, ascending=ascending, kind="mergesort",
                                      na_position="last").reset_index(drop=True)
        self._touch()

    def set_variable_attr(self, col: int, attr: str, value: Any) -> None:
        var = self.variables[col]
        if attr == "name":
            self.rename_variable(col, value)
            return
        if attr == "type":
            self.change_type(col, value)
            return
        setattr(var, attr, value)
        self._touch()


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _numeric_keys(labels: dict) -> dict:
    return {float(k): v for k, v in labels.items() if _is_number(k)}

"""Worksheet data manipulation (Origin's Worksheet menu), on pandas DataFrames, without Qt.

Stack / Unstack columns (wide ↔ long), Pivot Table, Reduce Rows (every n-th, block
statistics), Reduce Duplicate X, Reduce to evenly spaced X, Remove duplicate rows, Trim missing,
multi-key Sort, Split (by a group column or by row count) and the Worksheet Query / Data
Filter expression language (comparisons, ``and``/``or``/``not``, arithmetic, ``abs``,
``sqrt``, ``log``…, ``isnan()`` and ``between(x, a, b)``, string equality).
"""
from __future__ import annotations

import ast
import math
import re

import numpy as np
import pandas as pd

AGGREGATES = {"mean": "Mean", "sum": "Sum", "min": "Minimum", "max": "Maximum",
              "median": "Median", "first": "First", "last": "Last", "count": "Count",
              "std": "Standard Deviation"}


# ---------------------------------------------------------------------- stack / unstack
def stack_columns(df: pd.DataFrame, columns: list, id_columns: list = (),
                  group_name: str = "Group", value_name: str = "Value",
                  drop_missing: bool = True) -> pd.DataFrame:
    """Wide → long (Origin's Stack Columns): every value of ``columns`` in one column with
    the source column name as its group; ``id_columns`` are repeated."""
    out = df.melt(id_vars=list(id_columns), value_vars=list(columns), var_name=group_name,
                  value_name=value_name)
    if drop_missing:
        out = out[out[value_name].notna()]
    return out.reset_index(drop=True)


def unstack_columns(df: pd.DataFrame, value: str, group: str, key: str | None = None) \
        -> pd.DataFrame:
    """Long → wide (Origin's Unstack Columns): one column per group level (in order of first
    appearance). With ``key`` the rows are matched on that column (pivot), otherwise each
    group's values are listed from the top."""
    levels = list(dict.fromkeys(df[group].tolist()))
    if key is not None:
        wide = df.pivot_table(index=key, columns=group, values=value, aggfunc="first",
                              sort=False)
        wide = wide.reindex(columns=levels)
        wide.columns = [str(c) for c in wide.columns]
        return wide.reset_index()
    cols = {str(lv): df.loc[df[group] == lv, value].reset_index(drop=True) for lv in levels}
    return pd.DataFrame(cols)


def pivot(df: pd.DataFrame, index: str, columns: str, values: str,
          aggfunc: str = "mean", fill=None) -> pd.DataFrame:
    """Pivot Table: rows = levels of ``index``, columns = levels of ``columns``, cells =
    ``aggfunc`` of ``values``."""
    if aggfunc not in AGGREGATES:
        raise ValueError(f"Unknown aggregate '{aggfunc}'")
    t = pd.pivot_table(df, index=index, columns=columns, values=values, aggfunc=aggfunc,
                       fill_value=fill, observed=True)
    t.columns = [str(c) for c in t.columns]
    return t.reset_index()


# ---------------------------------------------------------------------- reduce
def reduce_rows(df: pd.DataFrame, n: int, method: str = "first", start: int = 0) \
        -> pd.DataFrame:
    """Every ``n`` rows → one (Origin's Reduce Rows): ``first`` keeps every n-th row from
    ``start``; the other methods aggregate each block of ``n`` rows (numeric columns)."""
    n = max(1, int(n))
    part = df.iloc[int(start):].reset_index(drop=True)
    if method == "first":
        return part.iloc[::n].reset_index(drop=True)
    block = np.arange(len(part)) // n
    num = part.apply(pd.to_numeric, errors="coerce")
    return getattr(num.groupby(block), method)().reset_index(drop=True)


def reduce_duplicate_x(x, ys: dict, method: str = "mean") -> pd.DataFrame:
    """Rows with the same X merged (Origin's Reduce Duplicate X), X sorted."""
    df = pd.DataFrame({"X": np.asarray(x, float), **{k: np.asarray(v, float)
                                                     for k, v in ys.items()}})
    df = df[np.isfinite(df["X"])]
    return getattr(df.groupby("X", sort=True), method)().reset_index()


def reduce_evenly_spaced(x, ys: dict, step: float | None = None, n_points: int | None = None,
                         start: float | None = None, method: str = "mean") -> pd.DataFrame:
    """Reduce to evenly spaced X (Origin's reducexy): X is cut into bins of width ``step``
    (or ``n_points`` bins) from ``start``; each bin gives its centre and the ``method`` of
    the Y values in it (empty bins are dropped)."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    lo = float(np.min(x[ok])) if start is None else float(start)
    hi = float(np.max(x[ok]))
    if step is None:
        if not n_points or n_points < 1:
            raise ValueError("Give the X step or the number of points.")
        step = (hi - lo) / n_points if hi > lo else 1.0
    if step <= 0:
        raise ValueError("The X step must be positive.")
    k = np.floor((x - lo) / step + 1e-9).astype(float)
    if n_points:
        k = np.minimum(k, n_points - 1)
    df = pd.DataFrame({"bin": k, **{name: np.asarray(v, float) for name, v in ys.items()}})
    df = df[ok & (k >= 0)]
    g = getattr(df.groupby("bin", sort=True), method)()
    out = g.reset_index()
    out.insert(0, "X", lo + (out.pop("bin") + 0.5) * step)
    return out


def remove_duplicates(df: pd.DataFrame, subset: list | None = None, keep: str = "first") \
        -> pd.DataFrame:
    return df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)


def trim_missing(df: pd.DataFrame, how: str = "any", subset: list | None = None) \
        -> pd.DataFrame:
    return df.dropna(how=how, subset=subset).reset_index(drop=True)


# ---------------------------------------------------------------------- sort / split
def sort_order(columns: list, ascending: list[bool]) -> np.ndarray:
    """Stable multi-key sort order of rows (first key most significant). Numbers sort
    before text, empty cells last — like Origin's Nested Sort."""
    n = len(columns[0]) if columns else 0
    keys = []
    for vals, asc in zip(columns, ascending):
        vals = list(vals) + [None] * (n - len(vals))
        def key_of(v):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return (2, 0.0, "")
            if isinstance(v, (int, float, np.integer, np.floating)):
                return (0, float(v), "")
            return (1, 0.0, str(v))
        ks = [key_of(v) for v in vals]
        uniq = sorted(set(ks))
        rank = {k: i for i, k in enumerate(uniq)}
        r = np.array([rank[k] for k in ks], dtype=float)
        if not asc:   # descending, but empty cells still last
            last = np.array([k[0] == 2 for k in ks])
            r = np.where(last, len(uniq) + 1, -r)
        keys.append(r)
    if not keys:
        return np.arange(n)
    return np.lexsort(keys[::-1])


def split_by_column(df: pd.DataFrame, column: str) -> dict:
    """{level: rows of that level} in order of first appearance (Origin's Split Worksheet
    by column)."""
    levels = list(dict.fromkeys(df[column].tolist()))
    return {lv: df[df[column] == lv].reset_index(drop=True) for lv in levels}


def split_by_rows(df: pd.DataFrame, n: int) -> list[pd.DataFrame]:
    n = max(1, int(n))
    return [df.iloc[i:i + n].reset_index(drop=True) for i in range(0, len(df), n)]


# ---------------------------------------------------------------------- query
_FUNCS = {"abs": np.abs, "sqrt": np.sqrt, "log": np.log10, "ln": np.log, "exp": np.exp,
          "sin": np.sin, "cos": np.cos, "tan": np.tan, "round": np.round,
          "isnan": lambda a: pd.isna(a), "int": np.trunc,
          "between": lambda a, lo, hi: (a >= lo) & (a <= hi)}
_ALLOWED = (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub,
            ast.UAdd, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
            ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Name,
            ast.Load, ast.Constant, ast.Call, ast.Subscript)


class QueryError(ValueError):
    pass


def _prepare(expr: str) -> str:
    e = expr.strip()
    e = e.replace("&&", " and ").replace("||", " or ").replace("<>", "!=")
    e = re.sub(r"(?<![<>=!])=(?!=)", "==", e)
    e = re.sub(r"(?i)\bcol\(\s*\"?([^\")]+?)\"?\s*\)", r"__col__['\1']", e)
    return e


def query_mask(df: pd.DataFrame, expr: str, names: dict | None = None) -> np.ndarray:
    """Rows of ``df`` where ``expr`` is true. Columns are referred to by name, by
    ``col(A)`` / ``col("Long Name")`` or through ``names`` (alias → column)."""
    if not expr or not expr.strip():
        return np.ones(len(df), dtype=bool)
    text = _prepare(expr)
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise QueryError(f"Syntax error in the condition: {exc.msg}") from None
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise QueryError(f"“{type(node).__name__}” is not allowed in a condition.")
    lookup = {str(c): c for c in df.columns}
    alias = {k: lookup.get(str(v), v) for k, v in (names or {}).items()}

    def column(name):
        key = alias.get(name, lookup.get(name))
        if key is None:
            low = {str(c).lower(): c for c in df.columns}
            key = low.get(str(name).lower())
        if key is None:
            raise QueryError(f"Unknown column “{name}”.")
        s = df[key]
        num = pd.to_numeric(s, errors="coerce")
        return num.to_numpy(float) if num.notna().sum() >= s.notna().sum() else \
            s.astype(object).to_numpy()

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in ("pi",):
                return math.pi
            return column(node.id)
        if isinstance(node, ast.Subscript):   # __col__['A']
            sl = node.slice
            key = sl.value if isinstance(sl, ast.Constant) else ev(sl)
            return column(str(key))
        if isinstance(node, ast.BoolOp):
            vals = [np.asarray(ev(v), dtype=bool) for v in node.values]
            out = vals[0]
            for v in vals[1:]:
                out = (out & v) if isinstance(node.op, ast.And) else (out | v)
            return out
        if isinstance(node, ast.UnaryOp):
            v = ev(node.operand)
            if isinstance(node.op, ast.Not):
                return ~np.asarray(v, dtype=bool)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp):
            a, b = ev(node.left), ev(node.right)
            op = type(node.op)
            return {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
                    ast.Div: np.divide, ast.Pow: np.power, ast.Mod: np.mod}[op](a, b)
        if isinstance(node, ast.Compare):
            left = ev(node.left)
            out = np.ones(len(df), dtype=bool)
            for op, right_node in zip(node.ops, node.comparators):
                right = ev(right_node)
                with np.errstate(invalid="ignore"):
                    res = {ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b,
                           ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b,
                           ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b}[
                        type(op)](left, right)
                out &= np.asarray(res, dtype=bool)
                left = right
            return out
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
                raise QueryError("Unknown function in the condition.")
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise QueryError("Unsupported expression.")
    with np.errstate(invalid="ignore", divide="ignore"):
        res = ev(tree)
    res = np.asarray(res)
    if res.shape == ():
        res = np.full(len(df), bool(res))
    return res.astype(bool)

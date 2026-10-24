"""SPSS-style expression language for Compute Variable and If conditions.

Expressions are tokenized, translated to a Python expression that only uses placeholders
(``__v3`` for variables, ``__f0`` for functions), parsed with :mod:`ast` and evaluated by a
whitelisting walker over pandas Series. Nothing is ever passed to ``eval``.

Semantics follow SPSS:
* user-missing and system-missing values are missing; arithmetic with missing gives missing
* comparisons/logicals give 1 (true), 0 (false) or missing; ``0 AND missing`` is 0
* statistical functions (SUM, MEAN, …) use the valid arguments; ``MEAN.3(…)`` needs ≥ 3
* ``X1 TO X5`` inside a function expands to the variables between them in file order
"""
from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from rsstatchem.core.dataset import NUMERIC, Dataset

_TOKEN = re.compile(r"""
    (?P<space>\s+)
  | (?P<str>'(?:[^']|'')*'|"(?:[^"]|"")*")
  | (?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)
  | (?P<name>[^\W\d][\w.$#@]*)
  | (?P<op><=|>=|<>|~=|\*\*|[-+*/(),<>=&|~])
""", re.VERBOSE | re.UNICODE)

_KEYWORDS = {"AND": " and ", "OR": " or ", "NOT": " not ", "EQ": "==", "NE": "!=",
             "LT": "<", "LE": "<=", "GT": ">", "GE": ">="}
_OPS = {"=": "==", "<>": "!=", "~=": "!=", "&": " and ", "|": " or ", "~": " not "}


class ExpressionError(ValueError):
    pass


# ------------------------------------------------------------------ function library
def _is_str(x) -> bool:
    return isinstance(x, str) or (isinstance(x, pd.Series) and x.dtype == object)


def _num(x, fname: str):
    if _is_str(x):
        raise ExpressionError(f"{fname} expects a numeric argument.")
    return x


def _clean(x):
    """Replace ±inf by missing (SPSS returns system-missing for undefined results)."""
    if isinstance(x, pd.Series):
        return x.replace([np.inf, -np.inf], np.nan)
    if isinstance(x, float) and math.isinf(x):
        return float("nan")
    return x


def _unary(fn: Callable, domain: Callable | None = None):
    def apply(args, name):
        if len(args) != 1:
            raise ExpressionError(f"{name} takes exactly one argument.")
        x = _num(args[0], name)
        with np.errstate(all="ignore"):
            if isinstance(x, pd.Series):
                out = pd.Series(fn(x.to_numpy(float)), index=x.index)
                if domain is not None:
                    out = out.where(domain(x))
                return _clean(out)
            if x is None or (isinstance(x, float) and math.isnan(x)):
                return float("nan")
            if domain is not None and not domain(x):
                return float("nan")
            return _clean(float(fn(x)))
    return apply


def _frame(args, name) -> pd.DataFrame:
    if not args:
        raise ExpressionError(f"{name} needs at least one argument.")
    cols = {}
    for i, a in enumerate(args):
        if _is_str(a):
            raise ExpressionError(f"{name} expects numeric arguments.")
        cols[i] = a
    n = next((len(a) for a in args if isinstance(a, pd.Series)), 1)
    index = next((a.index for a in args if isinstance(a, pd.Series)), pd.RangeIndex(n))
    return pd.DataFrame({k: (v if isinstance(v, pd.Series) else pd.Series(v, index=index))
                         for k, v in cols.items()}, index=index).astype(float)


def _rowstat(method: str, default_min: int = 1):
    def apply(args, name, min_valid=None):
        df = _frame(args, name)
        need = default_min if min_valid is None else min_valid
        valid = df.notna().sum(axis=1)
        if method == "sum":
            out = df.sum(axis=1)
        elif method == "mean":
            out = df.mean(axis=1)
        elif method == "sd":
            out = df.std(axis=1, ddof=1)
        elif method == "variance":
            out = df.var(axis=1, ddof=1)
        elif method == "min":
            out = df.min(axis=1)
        elif method == "max":
            out = df.max(axis=1)
        elif method == "median":
            out = df.median(axis=1)
        else:
            raise ExpressionError(name)
        return out.where(valid >= need)
    return apply


def _rnd(args, name):
    x = _num(args[0], name)
    mult = _num(args[1], name) if len(args) > 1 else 1.0
    with np.errstate(all="ignore"):
        return _clean(np.sign(x) * np.floor(np.abs(x) / mult + 0.5) * mult)


def _mod(args, name):
    if len(args) != 2:
        raise ExpressionError("MOD takes two arguments.")
    with np.errstate(all="ignore"):
        return _clean(np.fmod(_num(args[0], name), _num(args[1], name)))


def _nvalid(args, name):
    return pd.concat([a if isinstance(a, pd.Series) else pd.Series([a]) for a in args],
                     axis=1).notna().sum(axis=1).astype(float)


def _nmiss(args, name):
    return pd.concat([a if isinstance(a, pd.Series) else pd.Series([a]) for a in args],
                     axis=1).isna().sum(axis=1).astype(float)


def _missing(args, name):
    x = args[0]
    return x.isna().astype(float) if isinstance(x, pd.Series) else float(pd.isna(x))


def _range(args, name):
    if len(args) < 3 or (len(args) - 1) % 2:
        raise ExpressionError("RANGE(x, low1, high1 [, low2, high2 …]).")
    x = args[0]
    hit = None
    for lo, hi in zip(args[1::2], args[2::2]):
        cond = (x >= lo) & (x <= hi)
        hit = cond if hit is None else (hit | cond)
    return _logical(hit, x)


def _any(args, name):
    if len(args) < 2:
        raise ExpressionError("ANY(x, value1 [, value2 …]).")
    x = args[0]
    hit = None
    for v in args[1:]:
        cond = x == v
        hit = cond if hit is None else (hit | cond)
    return _logical(hit, x)


def _logical(hit, x):
    if isinstance(hit, pd.Series):
        out = hit.astype(float)
        return out.where(~x.isna()) if isinstance(x, pd.Series) else out
    return float(hit)


def _str_series(x) -> pd.Series | str:
    if isinstance(x, pd.Series):
        return x.where(x.notna(), "").astype(str)
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else str(x)


def _concat(args, name):
    out = _str_series(args[0])
    for a in args[1:]:
        out = out + _str_series(a)
    return out.astype(object) if isinstance(out, pd.Series) else out


def _strmap(fn: Callable):
    def apply(args, name):
        x = args[0]
        if not _is_str(x):
            raise ExpressionError(f"{name} expects a string argument.")
        s = _str_series(x)
        return s.map(fn).astype(object) if isinstance(s, pd.Series) else fn(s)
    return apply


def _length(args, name):
    s = _str_series(args[0])
    return s.str.len().astype(float) if isinstance(s, pd.Series) else float(len(s))


def _number(args, name):
    x = args[0]
    if isinstance(x, pd.Series):
        return pd.to_numeric(x.astype(str).str.strip().str.replace(",", ".", regex=False),
                             errors="coerce").where(x.notna())
    try:
        return float(str(x).strip().replace(",", "."))
    except ValueError:
        return float("nan")


def _string(args, name):
    x = args[0]
    decimals = int(args[1]) if len(args) > 1 else None

    def conv(v):
        if pd.isna(v):
            return ""
        if decimals is not None:
            return f"{v:.{decimals}f}"
        return str(int(v)) if float(v).is_integer() else repr(float(v))

    return x.map(conv).astype(object) if isinstance(x, pd.Series) else conv(x)


def _dist(fn: Callable, nargs: int):
    def apply(args, name):
        if len(args) != nargs:
            raise ExpressionError(f"{name} takes {nargs} arguments.")
        vals = [_num(a, name) for a in args]
        with np.errstate(all="ignore"):
            return _clean(fn(*vals))
    return apply


def _random(fn: Callable, nargs: int):
    def apply(args, name, n=1, index=None):
        if len(args) != nargs:
            raise ExpressionError(f"{name} takes {nargs} arguments.")
        return pd.Series(fn(*[float(a) for a in args], size=n), index=index)
    apply.random = True
    return apply


_RNG = np.random.default_rng()

FUNCTIONS: dict[str, tuple[Callable, str, str]] = {
    # name: (implementation, group, template / description)
    "ABS": (_unary(np.abs), "Arithmetic", "ABS(numexpr) — absolute value"),
    "SQRT": (_unary(np.sqrt, lambda x: x >= 0), "Arithmetic", "SQRT(numexpr) — square root"),
    "EXP": (_unary(np.exp), "Arithmetic", "EXP(numexpr) — e raised to a power"),
    "LN": (_unary(np.log, lambda x: x > 0), "Arithmetic", "LN(numexpr) — natural log"),
    "LG10": (_unary(np.log10, lambda x: x > 0), "Arithmetic", "LG10(numexpr) — base-10 log"),
    "RND": (_rnd, "Arithmetic", "RND(numexpr[, mult]) — round (half away from zero)"),
    "TRUNC": (_unary(np.trunc), "Arithmetic", "TRUNC(numexpr) — truncate toward zero"),
    "MOD": (_mod, "Arithmetic", "MOD(numexpr, modulus) — remainder"),
    "SIN": (_unary(np.sin), "Arithmetic", "SIN(radians)"),
    "COS": (_unary(np.cos), "Arithmetic", "COS(radians)"),
    "ARTAN": (_unary(np.arctan), "Arithmetic", "ARTAN(numexpr) — arctangent"),
    "ARSIN": (_unary(np.arcsin, lambda x: (x >= -1) & (x <= 1)), "Arithmetic",
              "ARSIN(numexpr) — arcsine"),
    "SUM": (_rowstat("sum"), "Statistical", "SUM(numexpr, numexpr[, …]) — sum of valid values"),
    "MEAN": (_rowstat("mean"), "Statistical", "MEAN(numexpr, numexpr[, …]) — mean of valid values"),
    "SD": (_rowstat("sd", 2), "Statistical", "SD(numexpr, numexpr[, …]) — standard deviation"),
    "VARIANCE": (_rowstat("variance", 2), "Statistical", "VARIANCE(numexpr, numexpr[, …])"),
    "MIN": (_rowstat("min"), "Statistical", "MIN(value, value[, …]) — smallest valid value"),
    "MAX": (_rowstat("max"), "Statistical", "MAX(value, value[, …]) — largest valid value"),
    "MEDIAN": (_rowstat("median"), "Statistical", "MEDIAN(numexpr, numexpr[, …])"),
    "NVALID": (_nvalid, "Missing Values", "NVALID(variable[, …]) — number of valid values"),
    "NMISS": (_nmiss, "Missing Values", "NMISS(variable[, …]) — number of missing values"),
    "MISSING": (_missing, "Missing Values", "MISSING(variable) — 1 if user- or system-missing"),
    "SYSMIS": (_missing, "Missing Values", "SYSMIS(variable) — 1 if system-missing"),
    "VALUE": (None, "Missing Values", "VALUE(variable) — value ignoring user-missing codes"),
    "RANGE": (_range, "Search", "RANGE(test, low, high[, …]) — 1 if within a range"),
    "ANY": (_any, "Search", "ANY(test, value[, value…]) — 1 if equal to any value"),
    "CONCAT": (_concat, "String", "CONCAT(strexpr, strexpr[, …]) — join strings"),
    "UPCASE": (_strmap(str.upper), "String", "UPCASE(strexpr) — upper case"),
    "LOWER": (_strmap(str.lower), "String", "LOWER(strexpr) — lower case"),
    "LTRIM": (_strmap(str.lstrip), "String", "LTRIM(strexpr) — remove leading blanks"),
    "RTRIM": (_strmap(str.rstrip), "String", "RTRIM(strexpr) — remove trailing blanks"),
    "LENGTH": (_length, "String", "LENGTH(strexpr) — number of characters"),
    "CHAR.LENGTH": (_length, "String", "CHAR.LENGTH(strexpr) — number of characters"),
    "NUMBER": (_number, "Conversion", "NUMBER(strexpr) — convert text to a number"),
    "STRING": (_string, "Conversion", "STRING(numexpr[, decimals]) — convert number to text"),
    "CDF.NORMAL": (_dist(stats.norm.cdf, 3), "CDF & Inverse",
                   "CDF.NORMAL(quant, mean, stddev) — normal cumulative probability"),
    "IDF.NORMAL": (_dist(stats.norm.ppf, 3), "CDF & Inverse",
                   "IDF.NORMAL(prob, mean, stddev) — normal quantile"),
    "CDFNORM": (_dist(stats.norm.cdf, 1), "CDF & Inverse", "CDFNORM(zvalue) — standard normal"),
    "PROBIT": (_dist(stats.norm.ppf, 1), "CDF & Inverse", "PROBIT(prob) — inverse standard normal"),
    "CDF.T": (_dist(stats.t.cdf, 2), "CDF & Inverse", "CDF.T(quant, df) — Student t"),
    "IDF.T": (_dist(stats.t.ppf, 2), "CDF & Inverse", "IDF.T(prob, df) — t quantile"),
    "CDF.CHISQ": (_dist(stats.chi2.cdf, 2), "CDF & Inverse", "CDF.CHISQ(quant, df)"),
    "CDF.F": (_dist(stats.f.cdf, 3), "CDF & Inverse", "CDF.F(quant, df1, df2)"),
    "RV.NORMAL": (_random(lambda m, s, size: _RNG.normal(m, s, size), 2), "Random Numbers",
                  "RV.NORMAL(mean, stddev) — random normal value"),
    "RV.UNIFORM": (_random(lambda a, b, size: _RNG.uniform(a, b, size), 2), "Random Numbers",
                   "RV.UNIFORM(min, max) — random uniform value"),
}
STAT_FUNCTIONS = {"SUM", "MEAN", "SD", "VARIANCE", "MIN", "MAX", "MEDIAN"}
FUNCTION_GROUPS = ["All"] + sorted({g for _, g, _ in FUNCTIONS.values()})


# ------------------------------------------------------------------ translation
@dataclass
class _Token:
    kind: str
    text: str
    pos: int


@dataclass
class Compiled:
    source: str
    python: str
    tree: ast.Expression
    variables: list[int]
    functions: list[tuple[str, int | None]]


def _tokenize(text: str) -> list[_Token]:
    tokens, pos = [], 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m:
            raise ExpressionError(f"Unexpected character '{text[pos]}' at position {pos + 1}.")
        kind = m.lastgroup
        if kind != "space":
            tokens.append(_Token(kind, m.group(), pos))
        pos = m.end()
    return tokens


def compile_expression(ds: Dataset, text: str) -> Compiled:
    if not text or not text.strip():
        raise ExpressionError("The expression is empty.")
    tokens = _tokenize(text)
    # expand "A TO B" variable ranges
    expanded: list[_Token] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.kind == "name" and t.text.upper() == "TO" and expanded and i + 1 < len(tokens):
            first, last = expanded[-1], tokens[i + 1]
            try:
                a, b = ds.index_of(first.text), ds.index_of(last.text)
            except KeyError:
                raise ExpressionError("TO must be placed between two variable names.") from None
            if a > b:
                raise ExpressionError(f"{first.text} comes after {last.text} in the file.")
            expanded.pop()
            for k, idx in enumerate(range(a, b + 1)):
                if k:
                    expanded.append(_Token("op", ",", t.pos))
                expanded.append(_Token("name", ds.variables[idx].name, t.pos))
            i += 2
            continue
        expanded.append(t)
        i += 1

    parts, variables, functions = [], [], []
    for i, t in enumerate(expanded):
        nxt = expanded[i + 1] if i + 1 < len(expanded) else None
        if t.kind == "str":
            quote = t.text[0]
            parts.append(repr(t.text[1:-1].replace(quote * 2, quote)))
        elif t.kind == "num":
            parts.append(t.text)
        elif t.kind == "op":
            parts.append(_OPS.get(t.text, t.text))
        else:
            upper = t.text.upper()
            if upper in _KEYWORDS:
                parts.append(_KEYWORDS[upper])
            elif nxt is not None and nxt.text == "(":
                base, minimum = upper, None
                if upper not in FUNCTIONS:
                    head, _, suffix = upper.rpartition(".")
                    if head in STAT_FUNCTIONS and suffix.isdigit():
                        base, minimum = head, int(suffix)
                    else:
                        raise ExpressionError(f"Unknown function: {t.text}")
                functions.append((base, minimum))
                parts.append(f"__f{len(functions) - 1}")
            else:
                try:
                    idx = ds.index_of(t.text)
                except KeyError:
                    raise ExpressionError(f"Unknown variable: {t.text}") from None
                variables.append(idx)
                parts.append(f"__v{idx}")
    python = " ".join(parts).strip()  # a leading " not " would read as an indent
    try:
        tree = ast.parse(python, mode="eval")
    except SyntaxError:
        raise ExpressionError("The expression is incomplete or has unbalanced parentheses.") \
            from None
    return Compiled(text, python, tree, variables, functions)


# ------------------------------------------------------------------ evaluation
_ARITH = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
          ast.Div: operator.truediv, ast.Pow: operator.pow}
_CMP = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
        ast.Gt: operator.gt, ast.GtE: operator.ge}


class _Evaluator:
    def __init__(self, ds: Dataset, compiled: Compiled):
        self.ds = ds
        self.c = compiled
        self.index = ds.df.index

    def series(self, x) -> pd.Series:
        if isinstance(x, pd.Series):
            return x
        if isinstance(x, str):
            return pd.Series([x] * len(self.index), index=self.index, dtype=object)
        return pd.Series(float(x) if x is not None else np.nan, index=self.index, dtype=float)

    def visit(self, node):
        method = getattr(self, f"visit_{type(node).__name__}", None)
        if method is None:
            raise ExpressionError("This construction is not allowed in an expression.")
        return method(node)

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Constant(self, node):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float, str)):
            raise ExpressionError("Invalid constant.")
        return float(node.value) if not isinstance(node.value, str) else node.value

    def _var(self, node, raw: bool = False) -> pd.Series:
        if not isinstance(node, ast.Name) or not node.id.startswith("__v"):
            raise ExpressionError("A variable name is required here.")
        var = self.ds.variables[int(node.id[3:])]
        s = self.ds.df[var.name] if raw else self.ds.analysis_series(var.name)
        if var.type == NUMERIC:
            return pd.to_numeric(s, errors="coerce").astype(float)
        if var.type == "date":
            return (s - pd.Timestamp("1582-10-14")).dt.total_seconds()  # SPSS date numbers
        return s.astype(object)

    def visit_Name(self, node):
        if node.id.startswith("__v"):
            return self._var(node)
        raise ExpressionError("A function name must be followed by its arguments.")

    def visit_UnaryOp(self, node):
        x = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            if _is_str(x):
                raise ExpressionError("NOT needs a logical (numeric) value.")
            return 1.0 - (x != 0) if not isinstance(x, pd.Series) else \
                (1.0 - (x != 0).astype(float)).where(x.notna())
        if _is_str(x):
            raise ExpressionError("Arithmetic on a string is not allowed.")
        return -x if isinstance(node.op, ast.USub) else x

    def visit_BinOp(self, node):
        fn = _ARITH.get(type(node.op))
        if fn is None:
            raise ExpressionError("Unsupported operator.")
        a, b = self.visit(node.left), self.visit(node.right)
        if _is_str(a) or _is_str(b):
            raise ExpressionError("Arithmetic on a string is not allowed (use CONCAT).")
        with np.errstate(all="ignore"):
            try:
                return _clean(fn(a, b))
            except ZeroDivisionError:
                return float("nan")

    def _compare(self, a, b, fn):
        if _is_str(a) != _is_str(b):
            raise ExpressionError("Type mismatch: a string is compared with a number.")
        if not isinstance(a, pd.Series) and not isinstance(b, pd.Series):
            if pd.isna(a) or pd.isna(b):
                return float("nan")
            return float(fn(a, b))
        sa, sb = self.series(a), self.series(b)
        missing = sa.isna() | sb.isna()
        fill = "" if _is_str(a) else 0.0
        return fn(sa.where(~missing, fill), sb.where(~missing, fill)).astype(float).where(~missing)

    def visit_Compare(self, node):
        left = self.visit(node.left)
        result = None
        for op, comp in zip(node.ops, node.comparators):
            right = self.visit(comp)
            fn = _CMP.get(type(op))
            if fn is None:
                raise ExpressionError("Unsupported comparison.")
            r = self._compare(left, right, fn)
            result = r if result is None else self._and(result, r)
            left = right
        return result

    def _and(self, a, b):
        sa, sb = self.series(a), self.series(b)
        false = (sa == 0) | (sb == 0)
        true = (sa != 0) & (sb != 0) & sa.notna() & sb.notna()
        return pd.Series(np.where(false, 0.0, np.where(true, 1.0, np.nan)), index=self.index)

    def _or(self, a, b):
        sa, sb = self.series(a), self.series(b)
        true = ((sa != 0) & sa.notna()) | ((sb != 0) & sb.notna())
        false = (sa == 0) & (sb == 0)
        return pd.Series(np.where(true, 1.0, np.where(false, 0.0, np.nan)), index=self.index)

    def visit_BoolOp(self, node):
        values = [self.visit(v) for v in node.values]
        if any(_is_str(v) for v in values):
            raise ExpressionError("AND / OR need logical (numeric) values.")
        out = values[0]
        for v in values[1:]:
            out = self._and(out, v) if isinstance(node.op, ast.And) else self._or(out, v)
        return out

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name) or not node.func.id.startswith("__f") \
                or node.keywords:
            raise ExpressionError("Invalid function call.")
        name, minimum = self.c.functions[int(node.func.id[3:])]
        impl = FUNCTIONS[name][0]
        if name == "VALUE":
            if len(node.args) != 1:
                raise ExpressionError("VALUE takes one variable.")
            return self._var(node.args[0], raw=True)
        if name == "SYSMIS" and len(node.args) == 1 and isinstance(node.args[0], ast.Name):
            return self._var(node.args[0], raw=True).isna().astype(float)
        args = [self.visit(a) for a in node.args]
        if getattr(impl, "random", False):
            return impl(args, name, n=len(self.index), index=self.index)
        if name in STAT_FUNCTIONS:
            return impl(args, name, minimum)
        return impl(args, name)


def evaluate(ds: Dataset, text: str) -> pd.Series:
    """Evaluate an expression for every case. Numeric results are float (NaN = missing);
    string results are object dtype."""
    compiled = compile_expression(ds, text)
    ev = _Evaluator(ds, compiled)
    result = ev.series(ev.visit(compiled.tree))
    if result.dtype != object:
        result = result.astype(float)
    return result


def is_string_result(series: pd.Series) -> bool:
    return series.dtype == object


def check(ds: Dataset, text: str) -> str | None:
    """Return an error message, or None when the expression is valid."""
    try:
        evaluate(ds, text) if ds.n_cases else compile_expression(ds, text)
    except ExpressionError as exc:
        return str(exc)
    except Exception as exc:  # noqa: BLE001 - surface any evaluation failure as a message
        return f"Cannot evaluate: {exc}"
    return None

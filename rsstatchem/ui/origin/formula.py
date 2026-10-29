"""Origin-style column formulas (Set Column Values / the F(x)= header row).

Origin formulas such as ``col(A)*2 + sin(i/10)``, ``(B - mean)^2`` or ``col("Time")/60`` are
translated into the app's expression language (:mod:`rsstatchem.core.expression`) and
evaluated there over a temporary :class:`~rsstatchem.core.dataset.Dataset` made from the
sheet: one variable per column (named by its short name) plus the row index ``i``.

Supported: ``col(A)`` / bare short names / ``col("Long Name")``, ``i`` (row number), ``^``
for powers, ``pi``, and Origin function names (sin, cos, tan, asin, acos, atan, exp, ln, log,
log10, sqrt, abs, int, round, mod, rnd()/ran(), grnd()/normal(), uniform()).
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from rsstatchem.core.dataset import NUMERIC, STRING, Dataset, Variable
from rsstatchem.core.expression import ExpressionError, evaluate
from rsstatchem.ui.origin.model import Worksheet, format_cell

ROW_VAR = "_ROWIDX"
_COL_RE = re.compile(r"""(?i)\bcol\(\s*(?:"([^"]*)"|'([^']*)'|([^)\s]+))\s*\)""")
# Origin name -> expression-language name (applied before an opening parenthesis).
_RENAMES = {"log": "LG10", "log10": "LG10", "int": "TRUNC", "atan": "ARTAN",
            "asin": "ARSIN", "ln": "LN", "sqrt": "SQRT", "abs": "ABS", "exp": "EXP",
            "sin": "SIN", "cos": "COS", "mod": "MOD"}
_RANDOM = {"rnd": "RV.UNIFORM(0, 1)", "ran": "RV.UNIFORM(0, 1)", "rand": "RV.UNIFORM(0, 1)",
           "uniform": "RV.UNIFORM(0, 1)", "grnd": "RV.NORMAL(0, 1)",
           "normal": "RV.NORMAL(0, 1)"}
EXAMPLES = (("col(A)*2", "twice column A"), ("col(B)^2", "square of B"),
            ("i", "row number"), ("sin(col(A))", "sine of A (radians)"),
            ("(col(B)-col(C))/col(C)*100", "percent difference"),
            ("ln(col(B))", "natural log"), ("rnd()", "uniform random 0..1"),
            ("grnd()", "normal random, mean 0, SD 1"))
FUNCTION_HELP = ("sin cos tan asin acos atan exp ln log log10 sqrt abs int round mod rnd() "
                 "grnd() pi")


class FormulaError(ValueError):
    pass


def _matching_paren(text: str, start: int) -> int:
    depth = 0
    for k in range(start, len(text)):
        if text[k] == "(":
            depth += 1
        elif text[k] == ")":
            depth -= 1
            if depth == 0:
                return k
    raise FormulaError("Unbalanced parentheses.")


def _rewrite_call(text: str, name: str, template: str) -> str:
    """Replace ``name(arg)`` by ``template`` with ``{a}`` = the argument text."""
    pattern = re.compile(rf"(?i)(?<![\w.]){name}\s*\(")
    while True:
        m = pattern.search(text)
        if not m:
            return text
        open_at = m.end() - 1
        close_at = _matching_paren(text, open_at)
        arg = text[open_at + 1:close_at]
        text = text[:m.start()] + template.format(a=arg) + text[close_at + 1:]


def _split_args(text: str) -> list[str]:
    """Top-level comma-separated arguments."""
    args, depth, start = [], 0, 0
    for k, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            args.append(text[start:k])
            start = k + 1
    args.append(text[start:])
    return args


def _rewrite_round(text: str) -> str:
    """Origin's round(x, n) rounds to n decimal places; the expression language's RND(x, m)
    rounds to a multiple m, so round(x, n) becomes RND(x*10^n)/10^n."""
    pattern = re.compile(r"(?i)(?<![\w.])round\s*\(")
    pos = 0
    while True:
        m = pattern.search(text, pos)
        if not m:
            return text
        open_at = m.end() - 1
        close_at = _matching_paren(text, open_at)
        args = _split_args(text[open_at + 1:close_at])
        if len(args) == 1:
            repl = f"RND({args[0]})"
        elif len(args) == 2:
            a, n = args[0].strip(), args[1].strip()
            repl = f"(RND(({a})*10**({n}))/10**({n}))"
        else:
            raise FormulaError("round() takes a value and optionally the number of decimals.")
        text = text[:m.start()] + repl + text[close_at + 1:]
        pos = m.start() + len(repl)


def translate(ws: Worksheet, formula: str) -> str:
    """Origin formula -> expression-language text (column references become short names)."""
    text = formula.strip()
    if text.startswith("="):
        text = text[1:]
    if not text:
        raise FormulaError("The formula is empty.")
    refs: list[str] = []

    def col_ref(m: re.Match) -> str:
        quoted = m.group(1) if m.group(1) is not None else m.group(2)
        if quoted is not None:
            col = ws.find_long_name(quoted)
            if col is None:
                try:
                    col = ws.column(quoted)
                except KeyError:
                    raise FormulaError(f'No column with the name "{quoted}".') from None
        else:
            try:
                col = ws.column(m.group(3))
            except KeyError:
                raise FormulaError(f"No column {m.group(3)} in this sheet.") from None
        refs.append(col.name)
        return f"\x00{len(refs) - 1}\x00"

    text = _COL_RE.sub(col_ref, text)
    for name, repl in _RANDOM.items():
        text = re.sub(rf"(?i)(?<![\w.]){name}\s*\(\s*[^()]*\)", repl, text)
    text = _rewrite_call(text, "tan", "(SIN({a})/COS({a}))")
    text = _rewrite_call(text, "acos", "(1.5707963267948966 - ARSIN({a}))")
    text = _rewrite_round(text)
    for name, repl in _RENAMES.items():
        text = re.sub(rf"(?i)(?<![\w.]){name}\s*\(", repl + "(", text)
    text = re.sub(r"(?<![\w.])i(?![\w.(])", ROW_VAR, text)
    text = re.sub(r"(?i)(?<![\w.])pi(?![\w.(])", repr(math.pi), text)
    text = text.replace("^", "**")
    return re.sub("\x00(\\d+)\x00", lambda m: refs[int(m.group(1))], text)


def sheet_dataset(ws: Worksheet, n: int | None = None) -> Dataset:
    """The sheet as a Dataset: numeric columns as numbers, others as text, plus ``i``."""
    n = ws.n_rows if n is None else n
    data, variables = {}, []
    for c in ws.columns:
        if c.is_text():
            data[c.name] = pd.Series([v if v != "" else np.nan for v in c.texts(n)],
                                     dtype=object)
            variables.append(Variable(c.name, STRING, label=c.long_name))
        else:
            data[c.name] = pd.Series(c.numeric(n))
            variables.append(Variable(c.name, NUMERIC, label=c.long_name))
    data[ROW_VAR] = pd.Series(np.arange(1, n + 1, dtype=float))
    variables.append(Variable(ROW_VAR, NUMERIC))
    return Dataset(pd.DataFrame(data, index=pd.RangeIndex(n)), variables, name=ws.name)


def evaluate_formula(ws: Worksheet, formula: str, rows: int | None = None) -> list:
    """Values (float / str / None) of ``formula`` for every row of the sheet."""
    expr = translate(ws, formula)
    ds = sheet_dataset(ws, rows)
    try:
        result = evaluate(ds, expr)
    except ExpressionError as exc:
        raise FormulaError(_origin_message(str(exc))) from None
    except ZeroDivisionError:
        raise FormulaError("Division by zero.") from None
    if result.dtype == object:
        return [None if (v is None or (isinstance(v, float) and math.isnan(v)) or v == "")
                else str(v) for v in result]
    return [None if not math.isfinite(v) else float(v) for v in result.to_numpy(float)]


def _origin_message(msg: str) -> str:
    return msg.replace(ROW_VAR, "i").replace("variable", "column").replace("Variable",
                                                                           "Column")


def apply_formula(ws: Worksheet, col: int, formula: str) -> None:
    """Set Column Values: evaluate and store the formula in the column's F(x)= row."""
    values = evaluate_formula(ws, formula)
    ws.set_column_values(col, values)
    ws.columns[col].formula = formula.strip()


def recalculate(ws: Worksheet, skip: int | None = None) -> list[int]:
    """Re-evaluate every column that has a formula (Origin's Auto Update); returns the
    indices that changed. Formulas that fail are left as they are."""
    changed = []
    for _ in range(2):   # a second pass settles formulas that depend on formulas
        for idx, col in enumerate(ws.columns):
            if not col.formula or idx == skip:
                continue
            try:
                values = evaluate_formula(ws, col.formula)
            except (FormulaError, ValueError):
                continue
            current = (col.values + [None] * ws.n_rows)[:ws.n_rows]
            if [format_cell(v) for v in current] != [format_cell(v) for v in values]:
                ws.set_column_values(idx, values)
                if idx not in changed:
                    changed.append(idx)
    return changed

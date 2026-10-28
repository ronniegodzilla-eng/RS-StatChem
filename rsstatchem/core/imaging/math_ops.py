"""Process > Math and Process > Image Calculator.

Ported from ImageJ 1.54p ij/plugin/filter/ImageMath.java, ij/process/ImageProcessor.java,
ByteProcessor/ShortProcessor/FloatProcessor/ColorProcessor ``process(op, value)``,
ij/plugin/ImageCalculator.java, ij/process/{Byte,Short,Float,Color}Blitter.java and
ij/process/TypeConverter.java (public domain, NIH; W. Rasband et al.).

Java semantics are reproduced exactly: 8-bit and RGB channels go through a 256-entry table
(``ImageProcessor.process``), 16-bit through ``ShortProcessor.process`` with ``(int)``
truncation and the display range for gamma/log/exp, 32-bit in float32 (``c = (float)value``).
``Math.round`` is ``floor(x + 0.5)``; ``(int)`` of a double truncates toward zero, maps NaN to
0 and saturates at the int range. Integer blitter arithmetic overflows like Java ``int``.

The Math > Macro expression evaluator is a small, safe interpreter for the ImageJ macro
subset used by ``Process > Math > Macro...`` (assignments, arithmetic, comparisons, ``if`` /
``else``, blocks, the macro maths functions and ``getPixel``); it evaluates vectorised with
numpy over the whole selection rectangle at once. No Python ``eval`` is used.
"""
from __future__ import annotations

import math
import re

import numpy as np

# Process > Math operation keys
OPS = ("add", "subtract", "multiply", "divide", "and", "or", "xor", "min", "max", "gamma",
       "set", "log", "exp", "square", "sqrt", "reciprocal", "nan_background", "abs")

# Image Calculator operators (ImageCalculator.operators) and blitter modes
CALC_OPERATORS = ("Add", "Subtract", "Multiply", "Divide", "AND", "OR", "XOR", "Min", "Max",
                  "Average", "Difference", "Copy", "Transparent-zero")
_LC_OPERATORS = ("add", "sub", "mul", "div", "and", "or", "xor", "min", "max", "ave", "diff",
                 "copy", "zero")

_INT_MIN, _INT_MAX = -2147483648, 2147483647


# ---------------------------------------------------------------------------------- Java helpers
def java_int(x) -> np.ndarray:
    """Java ``(int)`` of a double (array): truncation toward zero, NaN -> 0, saturation."""
    a = np.asarray(x, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        out = np.trunc(np.nan_to_num(a, nan=0.0, posinf=_INT_MAX, neginf=_INT_MIN))
    return np.clip(out, _INT_MIN, _INT_MAX).astype(np.int64)


def java_round(x) -> np.ndarray:
    """Java ``Math.round(double)`` (floor(x + 0.5)), as int64 (NaN -> 0)."""
    a = np.asarray(x, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return java_int(np.floor(a + 0.5))


def wrap_int32(x) -> np.ndarray:
    """Java ``int`` overflow (two's complement wrap) of an int64 array."""
    a = np.asarray(x, dtype=np.int64)
    return ((a + 2 ** 31) % 2 ** 32) - 2 ** 31


def _ijint_str(value) -> int:
    """Java ``(int)value`` of a scalar double."""
    return int(java_int(value))


def parse_binary(text: str) -> int:
    """``Integer.parseInt(text, 2)`` (ImageMath AND/OR/XOR take a binary string)."""
    s = str(text).strip()
    if not re.fullmatch(r"[+-]?[01]+", s):
        raise ValueError("Binary number required")
    v = int(s, 2)
    if not _INT_MIN <= v <= _INT_MAX:
        raise ValueError("Binary number required")
    return v


# ---------------------------------------------------------------------------------- 8-bit LUT
_BYTE_OPS = {"add", "subtract", "multiply", "divide", "and", "or", "xor", "min", "max",
             "gamma", "set", "log", "exp", "square", "sqrt"}


def byte_table(op: str, value: float = 0.0) -> np.ndarray:
    """The 256-entry table of ``ImageProcessor.process(op, value)`` (8-bit and RGB)."""
    i = np.arange(256, dtype=np.float64)
    ii = np.arange(256, dtype=np.int64)
    scale = 255.0 / math.log(255.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if op == "set":
            v = np.full(256, _ijint_str(value), dtype=np.int64)
        elif op in ("add", "subtract"):
            v = ii + _ijint_str(value if op == "add" else -value)
        elif op in ("multiply", "divide"):
            v = java_round(i * value)
        elif op == "and":
            v = ii & _ijint_str(value)
        elif op == "or":
            v = ii | _ijint_str(value)
        elif op == "xor":
            v = ii ^ _ijint_str(value)
        elif op == "gamma":
            v = java_int(np.exp(np.log(i / 255.0) * value) * 255.0)
        elif op == "log":
            v = np.where(ii == 0, 0, java_int(np.log(i) * scale))
        elif op == "exp":
            v = java_int(np.exp(i / scale))
        elif op == "square":
            v = ii * ii
        elif op == "sqrt":
            v = java_int(np.sqrt(i))
        elif op == "min":
            v = np.where(i < value, _ijint_str(value), ii)
        elif op == "max":
            v = np.where(i > value, _ijint_str(value), ii)
        else:
            v = ii
    return np.clip(v, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------------- 16-bit
def _short_process(p: np.ndarray, op: str, value: float, vmin: float, vmax: float,
                   signed: bool) -> np.ndarray:
    """``ShortProcessor.process(op, value)``; ``vmin/vmax`` = getMin()/getMax()."""
    offset = 32768 if signed else 0
    v1 = p.astype(np.int64) - offset
    rng = float(vmax) - float(vmin)
    min2 = int(java_int(vmin)) - offset
    max2 = int(java_int(vmax)) - offset
    iv = _ijint_str(value)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if op == "set":
            v2 = np.full_like(v1, iv)
        elif op == "add":
            v2 = v1 + iv
        elif op == "subtract":
            v2 = v1 + _ijint_str(-value)
        elif op in ("multiply", "divide"):
            v2 = wrap_int32(java_round(v1 * value))
        elif op == "and":
            v2 = wrap_int32(v1 & iv)
        elif op == "or":
            v2 = wrap_int32(v1 | iv)
        elif op == "xor":
            v2 = wrap_int32(v1 ^ iv)
        elif op == "gamma":
            if rng <= 0.0:
                v2 = v1.copy()
            else:
                g = java_int(np.exp(value * np.log((v1 - min2) / rng)) * rng + min2)
                v2 = np.where(v1 == min2, v1, g)
        elif op == "log":
            factor = max2 / math.log(max2) if max2 > 0 and max2 != 1 else (
                math.inf if max2 == 1 else math.nan)
            v2 = np.where(v1 <= 0, 0, java_int(np.log(v1.astype(np.float64)) * factor))
        elif op == "exp":
            k = (math.log(max2) / max2) if max2 > 0 else math.nan
            v2 = java_int(np.exp(v1 * k))
        elif op == "square":
            d1 = v1.astype(np.float64)
            v2 = java_int(d1 * d1)
        elif op == "sqrt":
            v2 = java_int(np.sqrt(v1.astype(np.float64)))
        elif op == "abs":
            v2 = np.abs(v1)
        elif op == "min":
            v2 = np.where(v1 < value, iv, v1)
        elif op == "max":
            v2 = np.where(v1 > value, iv, v1)
        else:
            v2 = v1
    v2 = v2 + offset
    return np.clip(v2, 0, 65535).astype(np.uint16)


# ---------------------------------------------------------------------------------- 32-bit
def _float_process(p: np.ndarray, op: str, value: float) -> np.ndarray:
    """``FloatProcessor.process(op, value)`` in float32."""
    v1 = p.astype(np.float32)
    c = np.float32(value)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if op == "set":
            return np.full_like(v1, c)
        if op == "add":
            return v1 + c
        if op == "subtract":
            return v1 + np.float32(-value)
        if op in ("multiply", "divide"):
            return v1 * c
        if op == "gamma":
            d = v1.astype(np.float64)
            out = np.exp(np.float64(c) * np.log(d)).astype(np.float32)
            return np.where(v1 <= 0, np.float32(0), out)
        if op == "log":
            return np.log(v1.astype(np.float64)).astype(np.float32)
        if op == "exp":
            return np.exp(v1.astype(np.float64)).astype(np.float32)
        if op == "square":
            return v1 * v1
        if op == "sqrt":
            d = v1.astype(np.float64)
            out = np.sqrt(d).astype(np.float32)
            return np.where(v1 <= 0, np.float32(0), out)
        if op == "abs":
            return np.abs(v1)
        if op == "min":
            return np.where(v1.astype(np.float64) < value, np.float32(value), v1)
        if op == "max":
            return np.where(v1.astype(np.float64) > value, np.float32(value), v1)
        if op == "reciprocal":
            return np.where(v1 == 0, np.float32(np.nan), np.float32(1) / v1)
    return v1


# ---------------------------------------------------------------------------------- math_op
def math_op(plane: np.ndarray, op: str, value: float | str | None = None,
            display_range: tuple[float, float] | None = None, signed16: bool = False,
            threshold: tuple[float, float] | None = None) -> np.ndarray:
    """One Process > Math operation on a plane (Y, X) or RGB (Y, X, 3); returns a new array of
    the same dtype. Operations ImageJ does not apply to a type return an unchanged copy
    (e.g. Reciprocal on 8-bit, Abs on unsigned 16-bit, Divide by 0 on 8/16-bit).

    ``value``: number, or a binary string (or int) for and/or/xor. ``display_range`` is the
    16-bit ``getMin()/getMax()`` used by gamma/log/exp (default: plane min/max).
    ``threshold`` = (lower, upper) for "nan_background" (32-bit only)."""
    if op not in OPS:
        raise ValueError(f"unknown Math operation {op!r}")
    a = np.asarray(plane)
    if op in ("and", "or", "xor") and isinstance(value, str):
        value = parse_binary(value)
    value = 0.0 if value is None else float(value)
    dtype = a.dtype
    rgb = a.ndim == 3
    if op == "divide":
        if value == 0.0 and dtype != np.float32:
            return a.copy()
        with np.errstate(divide="ignore"):
            value = (1.0 / value) if value != 0.0 else math.inf
    if dtype == np.float32:
        if op in ("and", "or", "xor"):
            return a.copy()
        if op == "nan_background":
            if threshold is None:
                raise ValueError("Thresholded 32-bit float image required")
            lo, hi = float(threshold[0]), float(threshold[1])
            d = a.astype(np.float64)
            return np.where((d < lo) | (d > hi), np.float32(np.nan), a).astype(np.float32)
        return _float_process(a, op, value).astype(np.float32)
    if op in ("reciprocal", "nan_background"):
        return a.copy()
    if dtype == np.uint16:
        if op == "abs" and not signed16:
            return a.copy()
        if display_range is None:
            display_range = (float(a.min()), float(a.max()))
        return _short_process(a, op, value, display_range[0], display_range[1], signed16)
    # 8-bit and RGB
    if op == "abs":
        return a.copy()
    if rgb and op == "set":
        # ColorProcessor.set(value): fill with the int colour (int)value
        c = _ijint_str(value) & 0xFFFFFF
        out = np.empty_like(a)
        out[..., 0], out[..., 1], out[..., 2] = (c >> 16) & 255, (c >> 8) & 255, c & 255
        return out
    return byte_table(op, value)[a]


# ---------------------------------------------------------------------------------- Math > Macro
class MacroError(ValueError):
    pass


_TOKEN_RE = re.compile(r"""
    (?P<num>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?|0[xX][0-9a-fA-F]+)
  | (?P<name>[A-Za-z_][A-Za-z_0-9]*(?:\.[A-Za-z_][A-Za-z_0-9]*)?)
  | (?P<op>\+\+|--|\+=|-=|\*=|/=|==|!=|<=|>=|&&|\|\||<<|>>|[-+*/%^&|<>=!(){};,?:~])
  | (?P<ws>\s+|//[^\n]*|/\*.*?\*/)
""", re.VERBOSE | re.DOTALL)


def _tokenize(code: str) -> list[tuple[str, str]]:
    out, pos = [], 0
    while pos < len(code):
        m = _TOKEN_RE.match(code, pos)
        if not m:
            raise MacroError(f"Unexpected character in macro: {code[pos]!r}")
        pos = m.end()
        kind = m.lastgroup
        if kind == "ws":
            continue
        text = m.group()
        if kind == "num" and text[:2].lower() == "0x":
            text = str(int(text, 16))
        out.append((kind, text))
    out.append(("eof", ""))
    return out


class _Parser:
    """Recursive descent parser producing tuples:
    ("block", [stmts]) ("assign", name, op, expr) ("if", cond, then, else) ("expr", e)
    expressions: ("num", v) ("var", name) ("call", name, [args]) ("un", op, e)
    ("bin", op, a, b) ("tern", c, a, b)."""

    _PREC = [("||",), ("&&",), ("|",), ("^",), ("&",), ("==", "!="), ("<", "<=", ">", ">="),
             ("<<", ">>"), ("+", "-"), ("*", "/", "%")]

    def __init__(self, code: str):
        self.toks = _tokenize(code)
        self.i = 0

    def peek(self, k=0):
        return self.toks[self.i + k]

    def next(self):
        t = self.toks[self.i]
        self.i += 1
        return t

    def accept(self, text):
        if self.peek()[0] in ("op", "name") and self.peek()[1] == text:
            self.i += 1
            return True
        return False

    def expect(self, text):
        if not self.accept(text):
            raise MacroError(f"'{text}' expected but '{self.peek()[1]}' found")

    def program(self):
        stmts = []
        while self.peek()[0] != "eof":
            if self.accept(";"):
                continue
            stmts.append(self.statement())
        return ("block", stmts)

    def statement(self):
        t = self.peek()
        if t == ("op", "{"):
            self.next()
            stmts = []
            while not self.accept("}"):
                if self.peek()[0] == "eof":
                    raise MacroError("'}' expected")
                if self.accept(";"):
                    continue
                stmts.append(self.statement())
            return ("block", stmts)
        if t == ("name", "if"):
            self.next()
            self.expect("(")
            cond = self.expr()
            self.expect(")")
            then = self.statement()
            while self.accept(";"):
                pass
            other = None
            if self.accept("else"):
                other = self.statement()
            return ("if", cond, then, other)
        if t == ("name", "var"):
            self.next()
            return self.statement()
        if t[0] == "name" and self.peek(1)[0] == "op" and \
                self.peek(1)[1] in ("=", "+=", "-=", "*=", "/="):
            name = self.next()[1]
            op = self.next()[1]
            e = self.expr()
            self.accept(";")
            return ("assign", name, op, e)
        if t[0] == "name" and self.peek(1)[0] == "op" and self.peek(1)[1] in ("++", "--"):
            name = self.next()[1]
            op = self.next()[1]
            self.accept(";")
            return ("assign", name, "+=" if op == "++" else "-=", ("num", 1.0))
        e = self.expr()
        self.accept(";")
        return ("expr", e)

    def expr(self):
        cond = self.binary(0)
        if self.accept("?"):
            a = self.expr()
            self.expect(":")
            b = self.expr()
            return ("tern", cond, a, b)
        return cond

    def binary(self, level):
        if level == len(self._PREC):
            return self.unary()
        left = self.binary(level + 1)
        while self.peek()[0] == "op" and self.peek()[1] in self._PREC[level]:
            op = self.next()[1]
            right = self.binary(level + 1)
            left = ("bin", op, left, right)
        return left

    def unary(self):
        if self.peek()[0] == "op" and self.peek()[1] in ("-", "+", "!", "~"):
            op = self.next()[1]
            return ("un", op, self.unary())
        return self.primary()

    def primary(self):
        kind, text = self.next()
        if kind == "num":
            return ("num", float(text))
        if kind == "name":
            if text in ("true", "false"):
                return ("num", 1.0 if text == "true" else 0.0)
            if self.accept("("):
                args = []
                if not self.accept(")"):
                    args.append(self.expr())
                    while self.accept(","):
                        args.append(self.expr())
                    self.expect(")")
                return ("call", text, args)
            return ("var", text)
        if (kind, text) == ("op", "("):
            e = self.expr()
            self.expect(")")
            return e
        raise MacroError(f"Unexpected '{text}' in macro")


def _jround(x):
    return np.floor(np.asarray(x, dtype=np.float64) + 0.5)


def _as_int(x):
    return java_int(x)


_FUNCS1 = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "atan": np.arctan, "asin": np.arcsin,
    "acos": np.arccos, "sqrt": np.sqrt, "exp": np.exp, "log": np.log, "abs": np.abs,
    "floor": np.floor, "round": _jround, "isNaN": np.isnan,
    "Math.sin": np.sin, "Math.cos": np.cos, "Math.tan": np.tan, "Math.atan": np.arctan,
    "Math.asin": np.arcsin, "Math.acos": np.arccos, "Math.sqrt": np.sqrt,
    "Math.exp": np.exp, "Math.log": np.log, "Math.abs": np.abs, "Math.floor": np.floor,
    "Math.ceil": np.ceil, "Math.round": _jround, "Math.log10": np.log10,
    "Math.sqr": lambda a: a * a, "Math.signum": np.sign, "Math.sinh": np.sinh,
    "Math.cosh": np.cosh, "Math.tanh": np.tanh, "Math.erf": None,
}
_FUNCS2 = {
    "atan2": np.arctan2, "pow": np.power, "maxOf": np.maximum, "minOf": np.minimum,
    "Math.atan2": np.arctan2, "Math.pow": np.power, "Math.max": np.maximum,
    "Math.min": np.minimum,
}
_CONSTS = {"PI": math.pi, "NaN": math.nan, "Math.PI": math.pi, "Math.E": math.e}


def _erf(a):
    from scipy.special import erf
    return erf(a)


_FUNCS1["Math.erf"] = _erf


class _Evaluator:
    def __init__(self, env: dict, shape, getpixel=None, rng=None):
        self.env = env
        self.shape = shape
        self.getpixel = getpixel
        self.rng = rng or np.random.default_rng()

    def run(self, node, mask=None):
        kind = node[0]
        if kind == "block":
            for st in node[1]:
                self.run(st, mask)
        elif kind == "assign":
            _, name, op, e = node
            val = np.asarray(self.eval(e), dtype=np.float64)
            if op != "=":
                cur = np.asarray(self.var(name), dtype=np.float64)
                val = {"+=": cur + val, "-=": cur - val, "*=": cur * val,
                       "/=": cur / val}[op]
            val = np.broadcast_to(val, self.shape)
            if mask is None:
                self.env[name] = np.array(val, dtype=np.float64)
            else:
                cur = np.broadcast_to(np.asarray(self.env.get(name, 0.0), np.float64),
                                      self.shape)
                self.env[name] = np.where(mask, val, cur)
        elif kind == "if":
            _, cond, then, other = node
            c = np.broadcast_to(self.truth(self.eval(cond)), self.shape)
            m_then = c if mask is None else (mask & c)
            self.run(then, m_then)
            if other is not None:
                m_else = ~c if mask is None else (mask & ~c)
                self.run(other, m_else)
        elif kind == "expr":
            self.eval(node[1])

    @staticmethod
    def truth(v):
        a = np.asarray(v)
        if a.dtype == bool:
            return a
        return a != 0

    def var(self, name):
        if name in self.env:
            return self.env[name]
        if name in _CONSTS:
            return _CONSTS[name]
        raise MacroError(f"Undefined variable: {name}")

    def eval(self, node):
        kind = node[0]
        if kind == "num":
            return node[1]
        if kind == "var":
            return self.var(node[1])
        if kind == "un":
            v = self.eval(node[2])
            if node[1] == "-":
                return -np.asarray(v, dtype=np.float64)
            if node[1] == "+":
                return v
            if node[1] == "!":
                return ~self.truth(v)
            return (~_as_int(v)).astype(np.float64)
        if kind == "tern":
            c = self.truth(self.eval(node[1]))
            return np.where(c, self.eval(node[2]), self.eval(node[3]))
        if kind == "bin":
            op = node[1]
            if op in ("&&", "||"):
                a = self.truth(self.eval(node[2]))
                b = self.truth(self.eval(node[3]))
                return (a & b) if op == "&&" else (a | b)
            a = self.eval(node[2])
            b = self.eval(node[3])
            fa = np.asarray(a, dtype=np.float64) if not isinstance(a, np.ndarray) or \
                a.dtype != bool else a.astype(np.float64)
            fb = np.asarray(b, dtype=np.float64) if not isinstance(b, np.ndarray) or \
                b.dtype != bool else b.astype(np.float64)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                if op == "+":
                    return fa + fb
                if op == "-":
                    return fa - fb
                if op == "*":
                    return fa * fb
                if op == "/":
                    return fa / fb
                if op == "%":
                    return np.fmod(fa, fb)
                if op == "==":
                    return fa == fb
                if op == "!=":
                    return fa != fb
                if op == "<":
                    return fa < fb
                if op == "<=":
                    return fa <= fb
                if op == ">":
                    return fa > fb
                if op == ">=":
                    return fa >= fb
                ia, ib = _as_int(fa), _as_int(fb)
                if op == "&":
                    return (ia & ib).astype(np.float64)
                if op == "|":
                    return (ia | ib).astype(np.float64)
                if op == "^":
                    return (ia ^ ib).astype(np.float64)
                if op == "<<":
                    return wrap_int32(ia << (ib & 31)).astype(np.float64)
                if op == ">>":
                    return (wrap_int32(ia) >> (ib & 31)).astype(np.float64)
            raise MacroError(f"Unsupported operator {op}")
        if kind == "call":
            name, args = node[1], [self.eval(a) for a in node[2]]
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                if name in _FUNCS1 and len(args) == 1:
                    return _FUNCS1[name](np.asarray(args[0], dtype=np.float64))
                if name in _FUNCS2 and len(args) == 2:
                    return _FUNCS2[name](np.asarray(args[0], np.float64),
                                         np.asarray(args[1], np.float64))
                if name in ("random", "Math.random") and not args:
                    return self.rng.random(self.shape)
                if name == "getPixel" and len(args) == 2 and self.getpixel is not None:
                    return self.getpixel(np.broadcast_to(np.asarray(args[0], np.float64),
                                                         self.shape),
                                         np.broadcast_to(np.asarray(args[1], np.float64),
                                                         self.shape))
            raise MacroError(f"Unrecognized function: {name}")
        raise MacroError("bad expression")


def parse_macro(code: str):
    """Parse Math > Macro code (``"v=v*2"``; a bare expression gets ``v=`` like ImageJ)."""
    if "=" not in code:
        code = "v=" + code
    return _Parser(code).program()


def macro_uses_getpixel(code: str) -> bool:
    """Does Math > Macro ``code`` call getPixel (ImageJ then ignores the selection)?"""
    return _has_word(code, "getPixel")


def _has_word(code: str, word: str) -> bool:
    return any(k == "name" and t == word for k, t in _tokenize(code))


def _bilinear(src: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """ImageProcessor.getInterpolatedValue (bilinear; 0 outside like getPixel)."""
    h, w = src.shape
    x0 = np.floor(x)
    y0 = np.floor(y)
    fx, fy = x - x0, y - y0
    x0i, y0i = x0.astype(np.int64), y0.astype(np.int64)

    def at(xx, yy):
        ok = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        return np.where(ok, src[np.clip(yy, 0, h - 1), np.clip(xx, 0, w - 1)], 0.0)
    ll, lr = at(x0i, y0i + 1), at(x0i + 1, y0i + 1)
    ul, ur = at(x0i, y0i), at(x0i + 1, y0i)
    upper = ul + fx * (ur - ul)
    lower = ll + fx * (lr - ll)
    return upper + fy * (lower - upper)


def macro_expression(plane: np.ndarray, code: str, z: int = 0,
                     rect: tuple[int, int, int, int] | None = None,
                     signed16: bool = False, rng=None) -> np.ndarray:
    """Process > Math > Macro (``ImageMath.applyMacro``) on one plane; returns a new array.

    Variables: v (pixel value), x, y, z (0-based slice), w, h, d (distance from the centre
    (w/2, h/2)), a (angle in radians, 0..2pi, counter-clockwise, y up). Only the rectangle
    ``rect`` = (x, y, w, h) is computed (pixels outside it are kept). Code that calls getPixel
    makes ImageJ drop the selection (the macro's temporary ImagePlus resets the processor's
    ROI), so the whole plane is computed then. RGB: v runs per channel, the three runs share
    the other variables (as ImageJ's interpreter does), unless the code uses getPixel, then it
    runs once on the packed 0xRRGGBB int."""
    a = np.asarray(plane)
    H, W = a.shape[:2]
    tree = parse_macro(code)
    has_getpixel = macro_uses_getpixel(code)
    if has_getpixel:
        rect = None
    rx, ry, rw, rh = rect if rect is not None else (0, 0, W, H)
    yy, xx = np.mgrid[ry:ry + rh, rx:rx + rw]
    w2, h2 = W // 2, H // 2
    base_env = {"x": xx.astype(np.float64), "y": yy.astype(np.float64), "z": float(z),
                "w": float(W), "h": float(H)}
    dx, dy = xx - w2, yy - h2
    base_env["d"] = np.sqrt((dx * dx + dy * dy).astype(np.float64))
    ang = np.arctan2(((H - yy - 1) - h2).astype(np.float64), dx.astype(np.float64))
    base_env["a"] = np.where(ang < 0, ang + 2 * math.pi, ang)
    shape = (rh, rw)

    def run(values: np.ndarray, getpixel_src: np.ndarray | None, env: dict | None = None):
        env = dict(base_env) if env is None else env
        env["v"] = values.astype(np.float64)
        gp = None
        if getpixel_src is not None:
            src = getpixel_src.astype(np.float64)
            # getPixel outside the image: 0 (byte/short/RGB), NaN (float)
            outside = np.nan if getpixel_src.dtype == np.float32 else 0.0

            def gp(px, py):
                integral = (px == np.floor(px)) & (py == np.floor(py))
                pxi, pyi = px.astype(np.int64), py.astype(np.int64)
                ok = (pxi >= 0) & (pxi < W) & (pyi >= 0) & (pyi < H)
                direct = np.where(ok, src[np.clip(pyi, 0, H - 1), np.clip(pxi, 0, W - 1)],
                                  outside)
                if integral.all():
                    return direct
                return np.where(integral, direct, _bilinear(src, px, py))
        ev = _Evaluator(env, shape, gp, rng)
        ev.run(tree)
        return np.broadcast_to(np.asarray(env["v"], dtype=np.float64), shape)

    out = a.copy()
    sl = (slice(ry, ry + rh), slice(rx, rx + rw))
    if a.ndim == 3:
        if has_getpixel:
            packed = (a[..., 0].astype(np.int64) << 16) | (a[..., 1].astype(np.int64) << 8) | \
                a[..., 2].astype(np.int64)
            res = java_int(run(packed[sl], packed)) & 0xFFFFFF
            out[sl + (0,)] = (res >> 16) & 255
            out[sl + (1,)] = (res >> 8) & 255
            out[sl + (2,)] = res & 255
        else:
            env = dict(base_env)
            for k in range(3):
                res = np.clip(java_int(run(a[sl + (k,)], None, env)), 0, 255)
                out[sl + (k,)] = res
        return out
    gsrc = a if has_getpixel else None
    if a.dtype == np.uint8:
        out[sl] = np.clip(java_int(run(a[sl], gsrc)), 0, 255)
    elif a.dtype == np.uint16:
        if signed16:
            vals = a.astype(np.float64) - 32768.0
            res = run(vals[sl], vals if has_getpixel else None)
            # putPixelValue on a signed image: (int)Math.round(value + 32768), clamped
            out[sl] = np.clip(java_round(res + 32768.0), 0, 65535)
        else:
            out[sl] = np.clip(java_int(run(a[sl], gsrc)), 0, 65535)
    else:
        out[sl] = run(a[sl], gsrc).astype(np.float32)
    return out


# ---------------------------------------------------------------------------------- conversions
def to_byte_scaled(plane: np.ndarray, display_range: tuple[float, float] | None = None
                   ) -> np.ndarray:
    """``ip.convertToByte(true)`` of a gray plane (TypeConverter, scaled by the display
    range); RGB -> (r+g+b)/3 luminance."""
    a = np.asarray(plane)
    if a.ndim == 3:
        return rgb_to_gray(a).astype(np.uint8)
    if a.dtype == np.uint8:
        return a.copy()
    if display_range is None:
        if a.dtype == np.float32:
            f = a[np.isfinite(a)]
            display_range = (float(f.min()), float(f.max())) if f.size else (0.0, 0.0)
        else:
            display_range = (float(a.min()), float(a.max()))
    if a.dtype == np.uint16:
        lo, hi = int(display_range[0]), int(display_range[1])
        scale = 256.0 / (hi - lo + 1)
        v = np.maximum(a.astype(np.int64) - lo, 0)
        return np.minimum(java_int(v * scale + 0.5), 255).astype(np.uint8)
    from rsstatchem.core.imaging.lut import to_8bit
    return to_8bit(a, display_range[0], display_range[1])


def rgb_to_gray(rgb: np.ndarray, weights=(1 / 3, 1 / 3, 1 / 3)) -> np.ndarray:
    """TypeConverter.convertRGBToByte: (int)(r*rw + g*gw + b*bw + 0.5)."""
    r, g, b = (rgb[..., k].astype(np.float64) for k in range(3))
    return java_int(r * weights[0] + g * weights[1] + b * weights[2] + 0.5).astype(np.uint8)


def rgb_to_float(rgb: np.ndarray, weights=(1 / 3, 1 / 3, 1 / 3)) -> np.ndarray:
    """TypeConverter.convertRGBToFloat: (float)(r*rw + g*gw + b*bw)."""
    r, g, b = (rgb[..., k].astype(np.float64) for k in range(3))
    return (r * weights[0] + g * weights[1] + b * weights[2]).astype(np.float32)


def to_short(plane: np.ndarray) -> np.ndarray:
    """``ip.convertToShort(false)`` (no scaling)."""
    a = np.asarray(plane)
    if a.ndim == 3:
        return rgb_to_gray(a).astype(np.uint16)
    if a.dtype == np.float32:
        v = np.clip(np.nan_to_num(a.astype(np.float64), nan=0.0), 0.0, 65535.0)
        return java_int(v + 0.5).astype(np.uint16)
    return a.astype(np.uint16)


def to_float(plane: np.ndarray, ctable: np.ndarray | None = None) -> np.ndarray:
    """``ip.convertToFloat()`` (with the calibration table for 8/16-bit when given)."""
    a = np.asarray(plane)
    if a.ndim == 3:
        return rgb_to_float(a)
    if ctable is not None and a.dtype in (np.uint8, np.uint16):
        return np.asarray(ctable, dtype=np.float32)[a]
    return a.astype(np.float32)


def to_rgb(plane: np.ndarray, lut_array: np.ndarray | None = None,
           display_range: tuple[float, float] | None = None) -> np.ndarray:
    """``ip.convertToRGB()``: gray -> 8-bit (scaled) -> through the LUT."""
    a = np.asarray(plane)
    if a.ndim == 3:
        return a.copy()
    b = to_byte_scaled(a, display_range)
    table = np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, 1) if lut_array is None \
        else np.asarray(lut_array, dtype=np.uint8)
    return table[b]


# ---------------------------------------------------------------------------------- blitters
def blit(dst: np.ndarray, src: np.ndarray, op: str, divide_by_zero: float = math.inf
         ) -> np.ndarray:
    """``copyBits(src, 0, 0, mode)`` for ``dst`` and ``src`` of the same dtype/kind; the
    overlapping top-left region is combined, the rest of ``dst`` is kept. ``op`` is an
    Image Calculator operator (case-insensitive, e.g. "Add", "Transparent-zero")."""
    mode = calculator_mode(op)
    out = np.array(dst, copy=True)
    h = min(dst.shape[0], src.shape[0])
    w = min(dst.shape[1], src.shape[1])
    d = out[:h, :w]
    s = np.asarray(src)[:h, :w]
    if out.dtype == np.float32:
        out[:h, :w] = _blit_float(d, s.astype(np.float32), mode, divide_by_zero)
    elif out.ndim == 3:
        if mode == "copy":
            out[:h, :w] = s
        elif mode == "zero":
            zero = (s[..., 0] == 0) & (s[..., 1] == 0) & (s[..., 2] == 0)
            out[:h, :w] = np.where(zero[..., None], d, s)
        else:
            out[:h, :w] = _blit_int(d, s, mode, 255)
    else:
        maxv = 255 if out.dtype == np.uint8 else 65535
        out[:h, :w] = _blit_int(d, s, mode, maxv)
    return out


def _blit_int(d: np.ndarray, s: np.ndarray, mode: str, maxv: int) -> np.ndarray:
    di = d.astype(np.int64)
    si = s.astype(np.int64)
    if mode == "copy":
        r = si
    elif mode == "zero":
        r = np.where(si == 0, di, si)
    elif mode == "add":
        r = np.clip(si + di, 0, maxv)
    elif mode == "ave":
        r = (si + di) // 2
    elif mode == "sub":
        r = np.clip(di - si, 0, maxv)
    elif mode == "diff":
        r = np.minimum(np.abs(di - si), maxv)
    elif mode == "mul":
        r = np.clip(wrap_int32(si * di), 0, maxv)
    elif mode == "div":
        r = np.where(si == 0, maxv, di // np.where(si == 0, 1, si))
    elif mode == "and":
        r = si & di
    elif mode == "or":
        r = si | di
    elif mode == "xor":
        r = si ^ di
    elif mode == "min":
        r = np.minimum(si, di)
    elif mode == "max":
        r = np.maximum(si, di)
    else:
        raise ValueError(mode)
    return (r & maxv).astype(d.dtype)


def _blit_float(d: np.ndarray, s: np.ndarray, mode: str, dbz: float) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if mode == "copy":
            return s.copy()
        if mode == "zero":
            return np.where(s == 0, d, s)
        if mode == "add":
            return s + d
        if mode == "ave":
            return (s + d) / np.float32(2)
        if mode == "diff":
            return np.abs(d - s)
        if mode == "sub":
            return d - s
        if mode == "mul":
            return s * d
        if mode == "div":
            dbz32 = np.float32(dbz)
            q = d / s
            if np.isinf(dbz32):
                return q
            return np.where(s == 0, dbz32, q)
        if mode in ("and", "or", "xor"):
            a, b = java_int(s), java_int(d)
            r = {"and": a & b, "or": a | b, "xor": a ^ b}[mode]
            return wrap_int32(r).astype(np.float32)
        if mode == "min":
            return np.where(s < d, s, d)
        if mode == "max":
            return np.where(s > d, s, d)
    raise ValueError(mode)


def calculator_mode(operation: str) -> str:
    """ImageCalculator.getOperator: "xor" first, then the first lower-case operator found."""
    o = str(operation).lower()
    if "xor" in o:
        return "xor"
    for k in _LC_OPERATORS:
        if k in o:
            return k
    raise ValueError(f"No valid operator: {operation!r}")


def calculate_planes(p1: np.ndarray, p2: np.ndarray, operation: str,
                     float_result: bool = False, divide_by_zero: float = math.inf,
                     range1: tuple[float, float] | None = None,
                     range2: tuple[float, float] | None = None,
                     lut1: np.ndarray | None = None, lut2: np.ndarray | None = None,
                     ctable1=None, ctable2=None, create: bool = False) -> np.ndarray:
    """``ImageCalculator.doOperation`` for one plane pair. Returns the result plane: the
    in-place result for image1 (same shape/type as ``p1``), or with ``create``/``float_result``
    the new image (size = the overlap, float32 with ``float_result``)."""
    rgb1, rgb2 = p1.ndim == 3, p2.ndim == 3
    if create or float_result:
        h, w = min(p1.shape[0], p2.shape[0]), min(p1.shape[1], p2.shape[1])
        base = np.asarray(p1)[:h, :w]
        if float_result and not rgb1:
            base = to_float(base, ctable1)
        dst = np.array(base, copy=True)
        r1 = range1
    else:
        dst = np.array(p1, copy=True)
        r1 = range1
    src = np.asarray(p2)
    if float_result and not rgb2:
        src = to_float(src, ctable2)
    # dst.copyBits(src): convert src to dst's kind
    if dst.ndim == 3:
        src_c = to_rgb(src, lut2, range2) if src.ndim != 3 else src
        res = blit(dst, src_c, operation, divide_by_zero)
        if float_result:
            res = rgb_to_float(res)
        return res
    if dst.dtype == np.float32:
        src_c = to_float(src) if src.ndim != 3 else rgb_to_float(src)
        return blit(dst, src_c, operation, divide_by_zero)
    mode = calculator_mode(operation)
    if src.dtype == np.float32 and src.ndim == 2 and mode in ("mul", "div"):
        # temporaryFloat: FloatBlitter on a float copy, then setPixels(1, fp) (rounded)
        res = blit(dst.astype(np.float32), src, operation, divide_by_zero)
        maxv = 255.0 if dst.dtype == np.uint8 else 65535.0
        v = np.clip(np.nan_to_num(res.astype(np.float64), nan=0.0, posinf=maxv,
                                  neginf=0.0), 0.0, maxv)
        return java_int(v + 0.5).astype(dst.dtype)
    if dst.dtype == np.uint8:
        src_c = to_byte_scaled(src, range2) if not (src.dtype == np.uint8 and src.ndim == 2) \
            else src
        return blit(dst, src_c, operation, divide_by_zero)
    src_c = to_short(src)
    del r1
    return blit(dst, src_c, operation, divide_by_zero)
